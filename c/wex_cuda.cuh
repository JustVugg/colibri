#ifndef COLI_WEX_CUDA_CUH
#define COLI_WEX_CUDA_CUH

/* Optional CUDA reconstruction for WEX3 palette3, Neural-rANS (WNR1), WNR2 and WNR3.
 * The host bridge uses the Neural-rANS paths; bench_wex_cuda.cu also tests device output.
 * Input: complete compressed record on device + validated tile offsets.
 * Output: canonical raw MXFP4 slab, ready for existing kernels, no D2H copy.
 */
#include <cuda_runtime.h>
#include "wex_format.h"
#ifndef WEX_CUDA_NR_LUT_BITS
#define WEX_CUDA_NR_LUT_BITS 8u
#endif
#define WEX_CUDA_NR_LUT_SLOTS (1u << WEX_CUDA_NR_LUT_BITS)

static inline int wex_cuda_prepare_palette3(const WexEntry *entry,
    const unsigned char *record, size_t record_capacity, uint32_t *tile_offsets,
    size_t tile_capacity, uint32_t *scale_encoded) {
    uint64_t raw, aligned, disk, aligned_disk;
    size_t scale_stream, weights, consumed = 0, produced = 0, tile = 0;
    if (!entry || !record || !tile_offsets || !scale_encoded || entry->flags != WEX_ENTRY_PALETTE3 ||
        wex_entry_raw_sizes(entry, &raw, &aligned) || entry->scale_bytes % 16u ||
        record_capacity < entry->total_bytes || entry->total_bytes < WEX_PALETTE3_PREFIX_BYTES ||
        wex_read_u32_le(record) != WEX_PALETTE3_MAGIC || wex_read_u32_le(record + 12u)) return -1;
    scale_stream = wex_read_u32_le(record + 4u);
    weights = wex_read_u32_le(record + 8u);
    disk = (uint64_t)WEX_PALETTE3_PREFIX_BYTES + scale_stream + weights;
    if (!scale_stream || weights != (uint64_t)entry->weight_bytes / 4u * 3u ||
        wex_aligned_size(disk, &aligned_disk) || aligned_disk != entry->total_bytes ||
        disk > entry->total_bytes || aligned_disk >= aligned ||
        wex_unpack_scale_tiles(record + WEX_PALETTE3_PREFIX_BYTES, scale_stream, NULL, entry->scale_bytes)) return -1;
    while (produced < entry->scale_bytes) {
        size_t count = entry->scale_bytes - produced;
        unsigned int bits = record[WEX_PALETTE3_PREFIX_BYTES + consumed + 1u];
        if (tile >= tile_capacity) return -1;
        if (count > WEX_SCALE_TILE_VALUES) count = WEX_SCALE_TILE_VALUES;
        tile_offsets[tile++] = (uint32_t)consumed;
        consumed += 2u + (count * bits + 7u) / 8u;
        produced += count;
    }
    *scale_encoded = (uint32_t)scale_stream;
    return 0;
}

static __global__ void wex_cuda_scale_tiles(const unsigned char *record,
    const uint32_t *tile_offsets, unsigned char *output, uint32_t count, uint32_t prefix_bytes) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned char *tile = record + prefix_bytes + tile_offsets[index / 256u];
    unsigned int base = tile[0], bits = tile[1], value = 0;
    if (bits) {
        unsigned int bit = (index % 256u) * bits, pos = bit / 8u, shift = bit % 8u;
        value = tile[2u + pos] >> shift;
        if (shift + bits > 8u) value |= (unsigned int)tile[3u + pos] << (8u - shift);
        value &= (1u << bits) - 1u;
    }
    output[index] = (unsigned char)(base + value);
}

static __global__ void wex_cuda_palette3_weights(const unsigned char *record,
    uint32_t scale_stream, unsigned char *output, uint32_t scale_bytes,
    uint32_t groups0, uint32_t groups1) {
    uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= scale_bytes) return;
    unsigned int matrix = group < groups0 ? 0u : group < groups0 + groups1 ? 1u : 2u;
    const unsigned char *input = record + WEX_PALETTE3_PREFIX_BYTES + scale_stream + (size_t)group * 12u;
    const uint32_t *palettes = (const uint32_t *)(record + WEX_PALETTE3_HEADER_BYTES + matrix * 1024u);
    uint32_t palette = palettes[output[group]];
    uint4 values;
#pragma unroll
    for (unsigned int block = 0; block < 4u; block++) {
        uint32_t word = (uint32_t)input[block * 3u] | ((uint32_t)input[block * 3u + 1u] << 8) |
                        ((uint32_t)input[block * 3u + 2u] << 16);
        uint32_t packed = 0;
#pragma unroll
        for (unsigned int code = 0; code < 8u; code++)
            packed |= ((palette >> (((word >> (code * 3u)) & 7u) * 4u)) & 15u) << (code * 4u);
        if (block == 0) values.x = packed;
        if (block == 1) values.y = packed;
        if (block == 2) values.z = packed;
        if (block == 3) values.w = packed;
    }
    ((uint4 *)(output + scale_bytes))[group] = values;
}

static inline cudaError_t wex_cuda_launch_palette3(const WexEntry *entry,
    uint32_t scale_stream, const unsigned char *device_record, const uint32_t *device_tile_offsets,
    unsigned char *device_raw, cudaStream_t stream) {
    uint32_t groups0 = (uint32_t)entry->scale_shapes[0][0] * entry->scale_shapes[0][1];
    uint32_t groups1 = (uint32_t)entry->scale_shapes[1][0] * entry->scale_shapes[1][1];
    wex_cuda_scale_tiles<<<(entry->scale_bytes + 255u) / 256u, 256, 0, stream>>>(
        device_record, device_tile_offsets, device_raw, entry->scale_bytes, WEX_PALETTE3_PREFIX_BYTES);
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) return result;
    wex_cuda_palette3_weights<<<(entry->scale_bytes + 255u) / 256u, 256, 0, stream>>>(
        device_record, scale_stream, device_raw, entry->scale_bytes, groups0, groups1);
    return cudaGetLastError();
}

/* Internal offset builder: caller has validated both the record and the entire
 * scale stream. Keep that validation single-pass in the host bridge. */
static inline int wex_cuda_neural_tile_offsets(const WexNeuralRans *v,
    uint32_t scale_bytes, uint32_t *tile_offsets, size_t tile_capacity) {
    size_t consumed = 0, produced = 0, tile = 0;
    while (produced < scale_bytes) {
        size_t count = scale_bytes - produced;
        unsigned int bits = v->scales[consumed + 1u];
        if (tile >= tile_capacity) return -1;
        if (count > 256u) count = 256u;
        tile_offsets[tile++] = (uint32_t)consumed;
        consumed += 2u + (count * bits + 7u) / 8u; produced += count;
    }
    return 0;
}

static inline int wex_cuda_prepare_neural_rans(const WexEntry *entry,
    const unsigned char *record, size_t capacity, uint32_t *tile_offsets,
    size_t tile_capacity, uint32_t *metadata_bytes) {
    WexNeuralRans v;
    uint64_t raw, aligned;
    if (!entry || !record || !metadata_bytes || !tile_offsets || entry->flags != WEX_ENTRY_NEURAL_RANS ||
        wex_entry_raw_sizes(entry, &raw, &aligned) || entry->scale_bytes % 16u ||
        capacity < entry->total_bytes || entry->total_bytes >= aligned ||
        wex_nr_parse(record, entry->total_bytes, entry->weight_bytes, &v) ||
        wex_unpack_scale_tiles(v.scales, v.scale_encoded, NULL, entry->scale_bytes)) return -1;
    if (wex_cuda_neural_tile_offsets(&v, entry->scale_bytes, tile_offsets, tile_capacity)) return -1;
    *metadata_bytes = v.metadata_bytes;
    return 0;
}

static __device__ __forceinline__ uint32_t wex_cuda_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

template<bool ScaleOnly>
static __global__ void wex_cuda_neural_rans_weights(const unsigned char *record,
    unsigned char *output, uint32_t scale_bytes, uint32_t weight_bytes,
    uint32_t streams, uint32_t classes, uint32_t metadata_bytes, uint32_t scale_encoded, uint32_t chunk_symbols,
    const uint32_t *__restrict__ decode_lut, uint32_t *error) {
    __shared__ unsigned char input_cache[32768u];
    uint32_t stream = blockIdx.x * blockDim.x + threadIdx.x;
    const uint16_t *mapping = (const uint16_t *)(record + 32u);
    // CDF size is even but not necessarily four-byte aligned: use byte loads
    // for offsets below, rather than dereferencing this typed pointer.
    const unsigned char *off = record + 544u + classes * WEX_NR_CDF_BYTES;
    const unsigned char *payload = record + metadata_bytes + scale_encoded;
    uint32_t block_first = blockIdx.x * blockDim.x;
    uint32_t block_last = block_first + blockDim.x < streams ? block_first + blockDim.x : streams;
    uint32_t cache_begin = wex_cuda_u32(off + (size_t)block_first * 4u);
    uint32_t cache_end = wex_cuda_u32(off + (size_t)block_last * 4u);
    bool cached = cache_end - cache_begin <= sizeof(input_cache);
    if (cached) {
        // Contiguous cooperative reads replace the per-lane scattered global
        // loads on rANS renormalization. Oversized/rare streams use global
        // memory without changing the wire format or truncating any input.
        for (uint32_t p = threadIdx.x; p < cache_end - cache_begin; p += blockDim.x)
            input_cache[p] = payload[cache_begin + p];
    }
    __syncthreads(); // inactive lanes also take part in the cooperative copy
    if (stream >= streams) return;
    uint32_t first_byte = wex_cuda_u32(off + (size_t)stream * 4u);
    uint32_t last_byte = wex_cuda_u32(off + (size_t)(stream + 1u) * 4u);
    const unsigned char *source = cached ? input_cache : payload;
    uint32_t origin = cached ? cache_begin : 0u;
    const unsigned char *ptr = source + (first_byte - origin), *end = source + (last_byte - origin);
    uint32_t state = (uint32_t)ptr[0] << 24 | (uint32_t)ptr[1] << 16 | (uint32_t)ptr[2] << 8 | ptr[3];
    ptr += 4;
    if (state < WEX_NR_L || state >= (WEX_NR_L << 8)) { atomicExch(error, 1u); return; }
    size_t first = (size_t)stream * chunk_symbols;
    for (uint32_t group = 0; group < chunk_symbols / 32u; group++) {
        size_t position = first + (size_t)group * 32u;
        if (position >= (size_t)weight_bytes * 2u) break;
        uint32_t previous = 16u, cls = mapping[output[position / 32u]];
        if (cls >= classes) { atomicExch(error, 1u); return; }
        uint4 native = make_uint4(0,0,0,0);
#pragma unroll 1
        for (uint32_t j = 0; j < 32u; j++) {
            uint32_t slot = state & 4095u;
            size_t context = ScaleOnly ? cls : (size_t)cls * 17u + previous;
            uint32_t packed = __ldg(decode_lut + context * WEX_CUDA_NR_LUT_SLOTS + (slot >> (12u-WEX_CUDA_NR_LUT_BITS)));
            uint32_t lo = packed & 15u, frequency = (packed >> 4) & 4095u, start = packed >> 16;
            // Exact coarse lookup, not probability quantization: if a bucket
            // spans CDF boundaries, finish with the original integer CDF.
            if (slot >= start + frequency) {
                size_t row = ScaleOnly ? (size_t)cls * 17u + 16u : context;
                const uint16_t *cdf = (const uint16_t *)(record + 544u) + row * 17u;
                do {
                    start += frequency; ++lo;
                    frequency = __ldg(cdf + lo + 1u) - start;
                } while (slot >= start + frequency);
            }
            state = frequency * (state >> 12) + slot - start;
            while (state < WEX_NR_L) {
                if (ptr == end) { atomicExch(error, 1u); return; }
                state = (state << 8) | *ptr++;
            }
            uint32_t value = lo << ((j & 7u) * 4u);
            if (j < 8u) native.x |= value;
            else if (j < 16u) native.y |= value;
            else if (j < 24u) native.z |= value;
            else native.w |= value;
            previous = lo;
        }
        ((uint4 *)(output + scale_bytes))[position / 32u] = native;
    }
    if (ptr != end || state != WEX_NR_L) atomicExch(error, 1u);
}

template<bool ScaleOnly>
static __global__ void wex_cuda_neural_rans_build_lut(const unsigned char *record,
    uint32_t classes, uint32_t *output) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)classes * (ScaleOnly ? 1u : 17u) * WEX_CUDA_NR_LUT_SLOTS) return;
    uint32_t slot = (uint32_t)(index % WEX_CUDA_NR_LUT_SLOTS) << (12u-WEX_CUDA_NR_LUT_BITS), lo = 0, hi = 16u;
    size_t row = ScaleOnly ? (index / WEX_CUDA_NR_LUT_SLOTS) * 17u + 16u : index / WEX_CUDA_NR_LUT_SLOTS;
    const uint16_t *cdf = (const uint16_t *)(record + 544u) + row * 17u;
    while (lo + 1u < hi) {
        uint32_t mid = (lo + hi) / 2u;
        if (cdf[mid] <= slot) lo = mid; else hi = mid;
    }
    output[index] = lo | (uint32_t)(cdf[lo + 1u] - cdf[lo]) << 4 | (uint32_t)cdf[lo] << 16;
}

static inline cudaError_t wex_cuda_launch_neural_rans(const WexEntry *entry,
    const WexNeuralRans *view, const unsigned char *record, const uint32_t *tile_offsets,
    unsigned char *raw, uint32_t *decode_lut, uint32_t *error, cudaStream_t stream) {
    cudaError_t status = cudaMemsetAsync(error,0,4u,stream);
    if (status != cudaSuccess) return status;
    size_t slots = (size_t)view->classes * (view->scale_only ? 1u : 17u) * WEX_CUDA_NR_LUT_SLOTS;
    if(view->scale_only)
        wex_cuda_neural_rans_build_lut<true><<<(unsigned int)((slots + 255u) / 256u),256,0,stream>>>(record,view->classes,decode_lut);
    else
        wex_cuda_neural_rans_build_lut<false><<<(unsigned int)((slots + 255u) / 256u),256,0,stream>>>(record,view->classes,decode_lut);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    wex_cuda_scale_tiles<<<(entry->scale_bytes + 255u) / 256u, 256, 0, stream>>>(
        record, tile_offsets, raw, entry->scale_bytes, view->metadata_bytes);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    unsigned int threads = view->chunk_symbols >= 1024u ? 64u : 128u;
    unsigned int blocks = (view->streams + threads - 1u) / threads;
    if(view->scale_only)
        wex_cuda_neural_rans_weights<true><<<blocks, threads, 0, stream>>>(record, raw,
            entry->scale_bytes, entry->weight_bytes, view->streams, view->classes,
            view->metadata_bytes, view->scale_encoded, view->chunk_symbols, decode_lut, error);
    else
        wex_cuda_neural_rans_weights<false><<<blocks, threads, 0, stream>>>(record, raw,
            entry->scale_bytes, entry->weight_bytes, view->streams, view->classes,
            view->metadata_bytes, view->scale_encoded, view->chunk_symbols, decode_lut, error);
    return cudaGetLastError();
}

/* ---------------------------------------------------------------------------
 * WNR2 (wex_neural_rans.h): bit-identical to wex_nr2_decode.
 *
 * Per model and device, wex_cuda_build_nr2_tables lays out once, from a device
 * copy of the WNM2 blob (validated on the host by wex_nr2_model_open):
 *   weight rows   [3][567] x 32 B   cdf[0..15] as u16 pairs (cdf[16] = 1 << 14)
 *   scale direct  [3][1 << 14] u32  slot -> (frequency - 1) | (slot - start) << 14
 *   scale symbol  [3][1 << 14] B    slot -> symbol
 *   (optional, lookup 0) weight buckets [3][567][1 << WEX_CUDA_NR2_BUCKET_BITS]
 *                 u32 symbol | (frequency - 1) << 4 | start << 18 at each
 *                 bucket's first slot, finished by an exact CDF walk (as WNR1)
 * Host: wex_cuda_plan_nr2 validates the record with wex_nr2_parse and maps
 * every stream to its (matrix, first symbol), scales and weights separately;
 * wex_cuda_nr2_offsets prefix-sums the u16 lengths into absolute record
 * offsets (streams + 1 u32), uploaded with the record.
 * Kernel A: one thread per scale stream, the matrix's direct tables in shared
 * memory. Kernel B: one thread per weight stream, the exact context rule (at
 * i % 32 == 0: srel from the decoded scale and median[m], previous = maximum
 * = 8), its matrix's 567 CDF rows in shared memory and a branch-free 16-way
 * compare per symbol, one uint4 per 32 symbols. Every block serves one matrix.
 *
 * Payload bytes: every lane owns a 64-word ring in shared memory, refilled by
 * cp.async at warp-uniform points (stream open, then every 32 symbols) with
 * the words up to 40 ahead of its position; each symbol reads its next two
 * bytes with two shared loads issued before the lookup. (Register prefetch
 * stalls the warp: lanes refill at different symbols, and a register still
 * being loaded for one lane blocks every lane that reads it. Measured on an
 * RTX 5090: scale kernel 193 -> 75 us, weight kernel 300 -> 253 us per expert.)
 * Any violation (initial state outside [L, 256 L), reading past the end, an
 * early end, final state != L) sets *error; the host checks it, or the commit
 * kernel does, before anything reaches the caller's destination.
 *
 * WNR3 (codec 5, plan.nr3, a uniform branch): each lane folds its stream's
 * decoded coded bytes into MurmurHash3 (ident {model, owner, g, word24} first)
 * and requires final state L + h23 (error bit 2); a CZ record must not decode
 * nibble 8 (bit 8); in a CEIL matrix a block with coded max 5 is doubled back
 * (SWAR) before its uint4 store and its scale byte, already read for the
 * context, is decremented in place (scale 0: bit 4). The hash always sees the
 * coded bytes, so contexts and hashes never depend on the inverse.
 * ------------------------------------------------------------------------- */
#define WEX_CUDA_ERR_STRUCT 1u
#define WEX_CUDA_ERR_HASH 2u
#define WEX_CUDA_ERR_CEIL 4u
#define WEX_CUDA_ERR_CZ 8u
#define WEX_CUDA_NR2_THREADS 32u          /* lanes per block, both kernels */
#define WEX_CUDA_NR2_RING_WORDS 64u       /* per-lane payload ring, power of two */
#define WEX_CUDA_NR2_RING_AHEAD 40u       /* words kept requested ahead of the read position */
#ifndef WEX_CUDA_NR2_BUCKET_BITS
#define WEX_CUDA_NR2_BUCKET_BITS 10u
#endif
/* Weight symbol lookup used by wex_cuda_launch_nr2:
 * 0 = global coarse buckets + exact walk, 1 = shared-memory CDF compare.
 * Measured per expert (chunk 4096): 0 ~ 925 us (the 7 MB bucket table misses
 * L1), 1 ~ 253 us. */
#ifndef WEX_CUDA_NR2_LOOKUP
#define WEX_CUDA_NR2_LOOKUP 1
#endif
#define WEX_CUDA_NR2_ROWS (3u * WEX_NR2_CONTEXTS)
#define WEX_CUDA_NR2_RING_BYTES (WEX_CUDA_NR2_RING_WORDS * WEX_CUDA_NR2_THREADS * 4u)
/* The scale kernel's 64 KB direct table stays in global memory (read via
 * __ldg) in code built for pre-Ampere targets (__CUDA_ARCH__ < 800), whose
 * per-block shared limit is below 88 KB (sm_6x 48 KB, sm_75 64 KB), and
 * everywhere with WEX_CUDA_NR2_SCALE_DIRECT_GLOBAL=1. Shared then holds symbols
 * + ring, 24 KB; the host sizes it per device (wex_cuda_nr2_scale_smem), so a
 * fat build keeps the shared table on sm_80+. Pre-Ampere helper: nvcc ...
 * -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75
 * (sm_61 needs CUDA <= 12.9). */
#ifndef WEX_CUDA_NR2_SCALE_DIRECT_GLOBAL
#define WEX_CUDA_NR2_SCALE_DIRECT_GLOBAL 0
#endif
#define WEX_CUDA_NR2_SCALE_SMEM (WEX_NR2_TOTAL * (WEX_CUDA_NR2_SCALE_DIRECT_GLOBAL ? 1u : 5u) + WEX_CUDA_NR2_RING_BYTES)   /* 88 KB, opt-in (24 KB global-direct) */
#define WEX_CUDA_NR2_SCALE_SMEM_GLOBAL (WEX_NR2_TOTAL + WEX_CUDA_NR2_RING_BYTES)   /* 24 KB */

struct WexCudaNr2Tables {
    const unsigned char *blob;          /* device copy of the validated WNM2 blob */
    const uint4 *weight_rows;           /* [3 * 567][2] */
    const uint32_t *weight_buckets;     /* [3 * 567][1 << bits], or NULL */
    const uint32_t *scale_direct;       /* [3][1 << 14] */
    const unsigned char *scale_symbol;  /* [3][1 << 14] */
};

struct WexCudaNr2Plan {
    uint32_t chunk[2];              /* symbols per stream: [0] scales, [1] weights */
    uint32_t count[2][3];           /* scales and 4-bit weights of matrix m */
    uint32_t stream_base[2][4];     /* first stream of matrix m in the offset table; [k][3] ends kind k */
    uint32_t block_base[2][4];      /* first block of matrix m in kernel A / B; [k][3] = grid size */
    uint32_t scale_offset[3];       /* decoded scales of matrix m in the device raw buffer */
    uint32_t weight_offset[3];      /* decoded packed weights of matrix m (16-byte aligned) */
    uint32_t median[3];
    uint32_t streams;               /* scale + weight streams; the offset table has one more */
    uint32_t weight_base;           /* first weight byte in the device raw buffer */
    uint32_t weight_payload;        /* record offset of the weight payload: kernel A needs only [0, it) */
    uint32_t model_id;
    uint32_t nr3, ceil_mask, cz, owner, word24;   /* WNR3 record fields (all 0 for WNR2) */
};

static inline size_t wex_cuda_nr2_align256(size_t v) { return (v + 255u) & ~(size_t)255u; }

/* One device allocation per model and device: blob, rows, scale tables, buckets. */
static inline size_t wex_cuda_nr2_table_bytes(int with_buckets) {
    size_t bytes = wex_cuda_nr2_align256(WEX_NR2_MODEL_BYTES);
    bytes += wex_cuda_nr2_align256((size_t)WEX_CUDA_NR2_ROWS * 32u);
    bytes += 3u * WEX_NR2_TOTAL * 4u;
    bytes += 3u * WEX_NR2_TOTAL;
    if (with_buckets) bytes += (size_t)WEX_CUDA_NR2_ROWS * (1u << WEX_CUDA_NR2_BUCKET_BITS) * 4u;
    return bytes;
}

static inline void wex_cuda_nr2_table_view(unsigned char *base, int with_buckets, WexCudaNr2Tables *t) {
    size_t at = 0;
    t->blob = base;
    at += wex_cuda_nr2_align256(WEX_NR2_MODEL_BYTES);
    t->weight_rows = (const uint4 *)(base + at);
    at += wex_cuda_nr2_align256((size_t)WEX_CUDA_NR2_ROWS * 32u);
    t->scale_direct = (const uint32_t *)(base + at);
    at += 3u * WEX_NR2_TOTAL * 4u;
    t->scale_symbol = base + at;
    at += 3u * WEX_NR2_TOTAL;
    t->weight_buckets = with_buckets ? (const uint32_t *)(base + at) : NULL;
}

static __device__ __forceinline__ const uint16_t *wex_cuda_nr2_weight_cdf(const unsigned char *blob, uint32_t row) {
    return (const uint16_t *)(blob + WEX_NR2_MODEL_HEADER_BYTES) + (size_t)row * 17u;
}
static __device__ __forceinline__ const uint16_t *wex_cuda_nr2_scale_cdf(const unsigned char *blob, uint32_t m) {
    return (const uint16_t *)(blob + WEX_NR2_MODEL_HEADER_BYTES + WEX_NR2_WCDF_BYTES) + (size_t)m * 257u;
}

static __global__ void wex_cuda_nr2_build_rows(const unsigned char *blob, uint32_t *rows) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= WEX_CUDA_NR2_ROWS * 8u) return;
    const uint16_t *cdf = wex_cuda_nr2_weight_cdf(blob, index / 8u) + (index % 8u) * 2u;
    rows[index] = (uint32_t)cdf[0] | (uint32_t)cdf[1] << 16;
}

static __global__ void wex_cuda_nr2_build_scales(const unsigned char *blob,
    uint32_t *direct, unsigned char *symbol_table) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 3u * WEX_NR2_TOTAL) return;
    const uint16_t *cdf = wex_cuda_nr2_scale_cdf(blob, index / WEX_NR2_TOTAL);
    uint32_t slot = index % WEX_NR2_TOTAL, lo = 0, hi = 256u;
    while (lo + 1u < hi) {   /* the reference binary search */
        uint32_t mid = (lo + hi) / 2u;
        if (cdf[mid] <= slot) lo = mid; else hi = mid;
    }
    /* frequency <= 2^14 - 255 and slot - start < frequency: 14 bits each */
    direct[index] = (uint32_t)(cdf[lo + 1u] - cdf[lo] - 1u) | (slot - cdf[lo]) << 14;
    symbol_table[index] = (unsigned char)lo;
}

static __global__ void wex_cuda_nr2_build_buckets(const unsigned char *blob, uint32_t *buckets) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= WEX_CUDA_NR2_ROWS << WEX_CUDA_NR2_BUCKET_BITS) return;
    const uint16_t *cdf = wex_cuda_nr2_weight_cdf(blob, index >> WEX_CUDA_NR2_BUCKET_BITS);
    uint32_t slot = (index & ((1u << WEX_CUDA_NR2_BUCKET_BITS) - 1u)) << (WEX_NR2_PRECISION - WEX_CUDA_NR2_BUCKET_BITS);
    uint32_t lo = 0, hi = 16u;
    while (lo + 1u < hi) {
        uint32_t mid = (lo + hi) / 2u;
        if (cdf[mid] <= slot) lo = mid; else hi = mid;
    }
    buckets[index] = lo | (uint32_t)(cdf[lo + 1u] - cdf[lo] - 1u) << 4 | (uint32_t)cdf[lo] << 18;
}

/* Host: validate a WNR2 or WNR3 record (any model id; the caller resolves the
 * model and checks the WNR3 owner) and plan its decode. record holds
 * entry->total_bytes. */
static inline int wex_cuda_plan_nr2(const WexEntry *entry, const unsigned char *record, size_t capacity,
    WexNr2Record *view, WexCudaNr2Plan *plan) {
    uint64_t raw, aligned;
    WexNr2Layout layout;
    uint32_t m, k;
    if (!entry || !record || !view || !plan || !wex_entry_nr_model(entry) ||
        wex_entry_raw_sizes(entry, &raw, &aligned) || entry->total_bytes >= aligned ||
        capacity < entry->total_bytes) return -1;
    wex_nr2_layout(entry, &layout);
    if (wex_nrx_parse(record, entry->total_bytes, &layout, 0, wex_entry_nr_magic(entry), view)) return -1;
    memset(plan, 0, sizeof(*plan));
    plan->chunk[0] = view->scale_chunk;
    plan->chunk[1] = view->weight_chunk;
    plan->weight_base = (entry->scale_bytes + 15u) & ~15u;   /* uint4 stores */
    for (m = 0; m < 3u; m++) {
        plan->count[0][m] = layout.scale_count[m];
        plan->count[1][m] = layout.symbol_count[m];
        plan->scale_offset[m] = entry->scale_offsets[m];
        plan->weight_offset[m] = plan->weight_base + (entry->weight_offsets[m] - entry->scale_bytes);
        plan->median[m] = view->median[m];
    }
    for (k = 0; k < 2u; k++) {
        uint32_t stream = k ? view->scale_streams : 0u, block = 0;
        for (m = 0; m < 3u; m++) {
            uint32_t streams = wex_nr2_streams(plan->count[k][m], plan->chunk[k]);
            plan->stream_base[k][m] = stream;
            plan->block_base[k][m] = block;
            stream += streams;
            block += (streams + WEX_CUDA_NR2_THREADS - 1u) / WEX_CUDA_NR2_THREADS;
        }
        plan->stream_base[k][3] = stream;
        plan->block_base[k][3] = block;
    }
    plan->streams = view->scale_streams + view->weight_streams;
    plan->weight_payload = view->metadata_bytes + view->scale_payload_bytes;
    plan->model_id = view->model_id;
    if (view->magic == WEX_NR3_MAGIC) {
        plan->nr3 = 1u;
        plan->ceil_mask = view->flags & WEX_NR3_FLAG_CEIL_MASK;
        plan->cz = (view->flags & WEX_NR3_FLAG_CZ) != 0;
        plan->owner = view->owner;
        plan->word24 = view->word24;
    }
    return plan->stream_base[1][3] == plan->streams ? 0 : -1;
}

/* Absolute record offsets of every stream (scales, then weights) and the end. */
static inline void wex_cuda_nr2_offsets(const WexNr2Record *view, const WexCudaNr2Plan *plan, uint32_t *offsets) {
    uint32_t at = view->metadata_bytes, s;
    for (s = 0; s < plan->streams; s++) {
        offsets[s] = at;
        at += wex_nr_u16(view->lengths + (size_t)s * 2u);
    }
    offsets[plan->streams] = at;
}

/* ---- payload reader: per-lane shared-memory ring ------------------------ */
struct WexCudaNr2Reader {
    const uint32_t *words;   /* the record, as words */
    uint32_t *ring;          /* this lane's ring: word w at ring[(w % RING_WORDS) * 32] */
    uint32_t pos;            /* record offset of the next renormalization byte */
    uint32_t requested;      /* next word to request */
    uint32_t last;           /* the stream's last word */
};
struct WexCudaNr2Bytes { uint32_t a, b, shift; };

static __device__ __forceinline__ uint32_t wex_cuda_be32(uint32_t x) { return __byte_perm(x, 0u, 0x0123u); }

static __device__ __forceinline__ void wex_cuda_nr2_copy(uint32_t *ring_slot, const uint32_t *word, bool predicate) {
#if __CUDA_ARCH__ >= 800
    unsigned destination = (unsigned)__cvta_generic_to_shared(ring_slot);
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %2, 0;\n @p cp.async.ca.shared.global [%0], [%1], 4;\n}\n"
                 :: "r"(destination), "l"(word), "r"((int)predicate) : "memory");
#else   /* pre-Ampere: no cp.async; each lane owns its ring, so a plain copy needs no barrier */
    if (predicate) *ring_slot = __ldg(word);
#endif
}
#if __CUDA_ARCH__ >= 800
#define WEX_CUDA_NR2_WAIT(n) asm volatile("cp.async.commit_group;\ncp.async.wait_group " #n ";\n" ::: "memory")
#else
#define WEX_CUDA_NR2_WAIT(n) ((void)0)
#endif
static __device__ __forceinline__ uint32_t *wex_cuda_nr2_slot(const WexCudaNr2Reader &s, uint32_t w) {
    return s.ring + (w & (WEX_CUDA_NR2_RING_WORDS - 1u)) * WEX_CUDA_NR2_THREADS;
}

/* Opens [begin, end) (length >= 4, checked by wex_nr2_parse): its initial
 * state. Words past the stream's last word are never requested; a ring slot
 * read past it holds stale bytes, and consuming one is caught by pos != end. */
static __device__ __forceinline__ uint32_t wex_cuda_nr2_open(const unsigned char *record, uint32_t *ring_lane,
    uint32_t begin, uint32_t end, WexCudaNr2Reader &s) {
    uint32_t first = begin >> 2, target, w;
    s.words = (const uint32_t *)record;
    s.ring = ring_lane;
    s.last = (end - 1u) >> 2;
    target = min(first + WEX_CUDA_NR2_RING_AHEAD, s.last + 1u);
    for (w = first; w < target; w++) wex_cuda_nr2_copy(wex_cuda_nr2_slot(s, w), s.words + w, true);
    s.requested = target;
    WEX_CUDA_NR2_WAIT(0);
    uint32_t e0 = wex_cuda_be32(*wex_cuda_nr2_slot(s, first)), e1 = wex_cuda_be32(*wex_cuda_nr2_slot(s, first + 1u));
    s.pos = begin + 4u;
    return __funnelshift_l(e1, e0, (begin & 3u) * 8u);   /* big-endian bytes begin..begin+3 */
}

/* Warp-uniform, at most every 32 symbols (<= 64 bytes): keeps RING_AHEAD words
 * requested, then waits for the previous group (needed from here on). */
static __device__ __forceinline__ void wex_cuda_nr2_topup(WexCudaNr2Reader &s) {
    uint32_t target = min((s.pos >> 2) + WEX_CUDA_NR2_RING_AHEAD, s.last + 1u);
#pragma unroll
    for (uint32_t k = 0; k < 16u; k++) {
        uint32_t w = s.requested + k;
        wex_cuda_nr2_copy(wex_cuda_nr2_slot(s, w), s.words + w, w < target);
    }
    s.requested = max(s.requested, min(s.requested + 16u, target));
    WEX_CUDA_NR2_WAIT(1);
}

/* The two bytes at pos, read before the symbol lookup. */
static __device__ __forceinline__ WexCudaNr2Bytes wex_cuda_nr2_peek(const WexCudaNr2Reader &s) {
    WexCudaNr2Bytes r;
    r.a = *wex_cuda_nr2_slot(s, s.pos >> 2);
    r.b = *wex_cuda_nr2_slot(s, (s.pos >> 2) + 1u);
    r.shift = (s.pos & 3u) * 8u;
    return r;
}

/* while (state < L) state = state << 8 | next byte, branch-free: from a valid
 * state, decoding leaves at least 2^9, so 0, 1 or 2 bytes. */
static __device__ __forceinline__ uint32_t wex_cuda_nr2_renorm(uint32_t state, WexCudaNr2Reader &s,
    const WexCudaNr2Bytes &p) {
    uint32_t two = __funnelshift_l(wex_cuda_be32(p.b), wex_cuda_be32(p.a), p.shift);
    uint32_t bits = state < (1u << 15) ? 16u : state < WEX_NR_L ? 8u : 0u;
    s.pos += bits >> 3;
    return __funnelshift_l(two, state, bits);
}

static __device__ __forceinline__ uint32_t wex_cuda_nr2_matrix(const uint32_t base[4], uint32_t block) {
    return block >= base[2] ? 2u : block >= base[1] ? 1u : 0u;
}

/* Kernel A: scale streams. Dynamic shared memory wex_cuda_nr2_scale_smem(). */
static __global__ void __launch_bounds__(WEX_CUDA_NR2_THREADS) wex_cuda_nr2_scales(
    const unsigned char *record, const uint32_t *offsets, WexCudaNr2Plan plan,
    WexCudaNr2Tables tables, unsigned char *raw, uint32_t *error) {
    extern __shared__ uint4 wex_cuda_nr2_dynamic[];
    uint32_t m = wex_cuda_nr2_matrix(plan.block_base[0], blockIdx.x);
#if WEX_CUDA_NR2_SCALE_DIRECT_GLOBAL || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
    const uint32_t *direct = tables.scale_direct + (size_t)m * WEX_NR2_TOTAL;
    unsigned char *symbol_table = (unsigned char *)wex_cuda_nr2_dynamic;
    const uint4 *source;
#define WEX_CUDA_NR2_DIRECT(slot) __ldg(direct + (slot))
#else
    uint32_t *direct = (uint32_t *)wex_cuda_nr2_dynamic;
    unsigned char *symbol_table = (unsigned char *)(direct + WEX_NR2_TOTAL);
    const uint4 *source = (const uint4 *)(tables.scale_direct + (size_t)m * WEX_NR2_TOTAL);
    for (uint32_t i = threadIdx.x; i < WEX_NR2_TOTAL / 4u; i += blockDim.x) wex_cuda_nr2_dynamic[i] = __ldg(source + i);
#define WEX_CUDA_NR2_DIRECT(slot) direct[slot]
#endif
    uint32_t *ring = (uint32_t *)(symbol_table + WEX_NR2_TOTAL);
    source = (const uint4 *)(tables.scale_symbol + (size_t)m * WEX_NR2_TOTAL);
    for (uint32_t i = threadIdx.x; i < WEX_NR2_TOTAL / 16u; i += blockDim.x) ((uint4 *)symbol_table)[i] = __ldg(source + i);
    __syncthreads();
    uint32_t j = (blockIdx.x - plan.block_base[0][m]) * blockDim.x + threadIdx.x;
    if (j >= plan.stream_base[0][m + 1u] - plan.stream_base[0][m]) return;
    uint32_t stream = plan.stream_base[0][m] + j, begin = offsets[stream], end = offsets[stream + 1u];
    uint32_t first = j * plan.chunk[0], count = min(plan.chunk[0], plan.count[0][m] - first);
    unsigned char *out = raw + plan.scale_offset[m] + first;
    WexCudaNr2Reader s;
    uint32_t state = wex_cuda_nr2_open(record, ring + threadIdx.x, begin, end, s);
    if (state < WEX_NR_L || state >= (WEX_NR_L << 8)) { atomicOr(error, WEX_CUDA_ERR_STRUCT); return; }
    bool vector = !((uintptr_t)out & 15u);
    uint32_t hash = 0, pending = 0, held = 0;   /* WNR3: MurmurHash3 state, bytes not yet folded */
    if (plan.nr3) {
        const uint32_t ident[4] = {plan.model_id, plan.owner, stream, plan.word24};
        hash = wex_nr3_ident_hash(ident);
    }
    for (uint32_t i = 0; i < count; i += 32u) {
        wex_cuda_nr2_topup(s);
        uint32_t n = min(32u, count - i);
        if (vector && n == 32u) {
            uint32_t w[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
#pragma unroll
            for (uint32_t k = 0; k < 32u; k++) {
                WexCudaNr2Bytes p = wex_cuda_nr2_peek(s);
                uint32_t slot = state & (WEX_NR2_TOTAL - 1u), e = WEX_CUDA_NR2_DIRECT(slot);
                w[k / 4u] |= (uint32_t)symbol_table[slot] << ((k % 4u) * 8u);
                state = ((e & 0x3fffu) + 1u) * (state >> WEX_NR2_PRECISION) + (e >> 14);
                state = wex_cuda_nr2_renorm(state, s, p);
            }
            ((uint4 *)(out + i))[0] = make_uint4(w[0], w[1], w[2], w[3]);
            ((uint4 *)(out + i))[1] = make_uint4(w[4], w[5], w[6], w[7]);
            if (plan.nr3) {   /* i % 32 == 0: nothing pending */
#pragma unroll
                for (uint32_t k = 0; k < 8u; k++) hash = wex_nr3_h32_block(hash, w[k]);
            }
        } else {
            for (uint32_t k = 0; k < n; k++) {
                WexCudaNr2Bytes p = wex_cuda_nr2_peek(s);
                uint32_t slot = state & (WEX_NR2_TOTAL - 1u), e = WEX_CUDA_NR2_DIRECT(slot);
                out[i + k] = symbol_table[slot];
                if (plan.nr3) {
                    pending |= (uint32_t)symbol_table[slot] << (held * 8u);
                    if (++held == 4u) { hash = wex_nr3_h32_block(hash, pending); pending = held = 0; }
                }
                state = ((e & 0x3fffu) + 1u) * (state >> WEX_NR2_PRECISION) + (e >> 14);
                state = wex_cuda_nr2_renorm(state, s, p);
            }
        }
    }
    uint32_t expect = WEX_NR_L;
    if (plan.nr3) {
        if (held) hash = wex_nr3_h32_tail(hash, pending);
        expect += wex_nr3_h32_final(hash, 16u + count) & WEX_NR3_HASH_MASK;
    }
    if (s.pos != end) atomicOr(error, WEX_CUDA_ERR_STRUCT);
    else if (state != expect) atomicOr(error, plan.nr3 ? WEX_CUDA_ERR_HASH : WEX_CUDA_ERR_STRUCT);
}

/* Each halfword becomes 0xffff if its bit 15 is set, else 0: PRMT with the
 * sign-replicate selector bit (__byte_perm drops that bit, hence the PTX). */
static __device__ __forceinline__ uint32_t wex_cuda_half_sign(uint32_t x) {
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0xbb99;" : "=r"(r) : "r"(x));
    return r;
}

/* Symbol of `slot` in a 16-symbol CDF row a:b (cdf[0..15] as u16 pairs,
 * cdf[16] = 1 << 14): per halfword, (slot | 0x8000) - cdf has bit 15 set iff
 * cdf <= slot (all values < 2^15, no borrow between halves). */
static __device__ __forceinline__ void wex_cuda_nr2_row_find(uint4 a, uint4 b, uint32_t slot,
    uint32_t &symbol, uint32_t &start, uint32_t &frequency) {
    const uint32_t g = slot * 0x10001u | 0x80008000u;
    /* 0xffff in halfwords with cdf <= slot */
    uint32_t m0 = wex_cuda_half_sign(g - a.x), m1 = wex_cuda_half_sign(g - a.y);
    uint32_t m2 = wex_cuda_half_sign(g - a.z), m3 = wex_cuda_half_sign(g - a.w);
    uint32_t m4 = wex_cuda_half_sign(g - b.x), m5 = wex_cuda_half_sign(g - b.y);
    uint32_t m6 = wex_cuda_half_sign(g - b.z), m7 = wex_cuda_half_sign(g - b.w);
    uint32_t bits = (m0 & 0x00010001u) | (m1 & 0x00020002u) | (m2 & 0x00040004u) | (m3 & 0x00080008u) |
                    (m4 & 0x00100010u) | (m5 & 0x00200020u) | (m6 & 0x00400040u) | (m7 & 0x00800080u);
    symbol = (uint32_t)__popc(bits) - 1u;   /* cdf[0] = 0 always counts */
    uint32_t lo = __vmaxu2(__vmaxu2(__vmaxu2(a.x & m0, a.y & m1), __vmaxu2(a.z & m2, a.w & m3)),
                           __vmaxu2(__vmaxu2(b.x & m4, b.y & m5), __vmaxu2(b.z & m6, b.w & m7)));
    uint32_t hi = __vminu2(__vminu2(__vminu2(a.x | m0, a.y | m1), __vminu2(a.z | m2, a.w | m3)),
                           __vminu2(__vminu2(b.x | m4, b.y | m5), __vminu2(b.z | m6, b.w | m7)));
    start = max(lo & 0xffffu, lo >> 16);
    frequency = min(min(hi & 0xffffu, hi >> 16), WEX_NR2_TOTAL) - start;
}

/* WNR3 ceil-form inverse of 8 nibbles (lab wnr3-5090-20261107): magnitudes
 * 0..5 -> 0,2,4,5,6,7 (a + min(a, 2)), sign bits kept. */
static __device__ __forceinline__ uint32_t wex_cuda_ceil_dbl8(uint32_t w) {
    const uint32_t m = w & 0x77777777u;
    const uint32_t g = ((m >> 1) | (m >> 2)) & 0x11111111u;          /* 1 per nibble iff m >= 2 */
    const uint32_t mn = (g << 1) | (m & 0x11111111u & ~g);            /* min(m, 2) */
    return (w & 0x88888888u) | (m + mn);
}
/* Nonzero iff some nibble of w is 8 (-0). */
static __device__ __forceinline__ uint32_t wex_cuda_has_nibble8(uint32_t w) {
    const uint32_t t = w ^ 0x88888888u;
    return (t - 0x11111111u) & ~t & 0x88888888u;
}

/* Kernel B: weight streams. Lookup 1 stages the matrix's 567 CDF rows in
 * shared memory; lookup 0 reads global coarse buckets. */
template<int Lookup>
static __global__ void __launch_bounds__(WEX_CUDA_NR2_THREADS) wex_cuda_nr2_weights(
    const unsigned char *record, const uint32_t *offsets, WexCudaNr2Plan plan,
    WexCudaNr2Tables tables, unsigned char *raw, uint32_t *error) {
    __shared__ uint4 rows[Lookup == 1 ? WEX_NR2_CONTEXTS * 2u : 1u];
    __shared__ uint32_t ring[WEX_CUDA_NR2_RING_WORDS * WEX_CUDA_NR2_THREADS];
    uint32_t m = wex_cuda_nr2_matrix(plan.block_base[1], blockIdx.x);
    if (Lookup == 1) {
        const uint4 *source = tables.weight_rows + (size_t)m * WEX_NR2_CONTEXTS * 2u;
        for (uint32_t i = threadIdx.x; i < WEX_NR2_CONTEXTS * 2u; i += blockDim.x) rows[i] = __ldg(source + i);
        __syncthreads();
    }
    uint32_t j = (blockIdx.x - plan.block_base[1][m]) * blockDim.x + threadIdx.x;
    if (j >= plan.stream_base[1][m + 1u] - plan.stream_base[1][m]) return;
    uint32_t stream = plan.stream_base[1][m] + j, begin = offsets[stream], end = offsets[stream + 1u];
    uint32_t first = j * plan.chunk[1], blocks = min(plan.chunk[1], plan.count[1][m] - first) / 32u;
    unsigned char *scales = raw + plan.scale_offset[m] + first / 32u;   /* WNR3 ceil: decremented in place */
    uint4 *out = (uint4 *)(raw + plan.weight_offset[m] + first / 2u);
    const uint32_t *buckets = Lookup == 1 ? NULL :
        tables.weight_buckets + ((size_t)m * WEX_NR2_CONTEXTS << WEX_CUDA_NR2_BUCKET_BITS);
    const uint16_t *cdfs = Lookup == 1 ? NULL : wex_cuda_nr2_weight_cdf(tables.blob, m * WEX_NR2_CONTEXTS);
    WexCudaNr2Reader s;
    uint32_t state = wex_cuda_nr2_open(record, ring + threadIdx.x, begin, end, s);
    if (state < WEX_NR_L || state >= (WEX_NR_L << 8)) { atomicOr(error, WEX_CUDA_ERR_STRUCT); return; }
    int median = (int)plan.median[m];
    uint32_t scale = scales[0], hash = 0;
    const bool ceil_matrix = plan.nr3 && ((plan.ceil_mask >> m) & 1u);
    if (plan.nr3) {
        const uint32_t ident[4] = {plan.model_id, plan.owner, stream, plan.word24};
        hash = wex_nr3_ident_hash(ident);
    }
#pragma unroll 1
    for (uint32_t block = 0; block < blocks; block++) {
        wex_cuda_nr2_topup(s);
        uint32_t next_scale = block + 1u < blocks ? scales[block + 1u] : 0u;
        int d = (int)scale - median;
        uint32_t base = (uint32_t)(min(max(d, -3), 3) + 3) * 81u;
        uint32_t context = base + 80u, maximum = 0, word[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (uint32_t k = 0; k < 32u; k++) {
            WexCudaNr2Bytes p = wex_cuda_nr2_peek(s);
            uint32_t slot = state & (WEX_NR2_TOTAL - 1u), symbol, start, frequency;
            if (Lookup == 1) {
                wex_cuda_nr2_row_find(rows[context * 2u], rows[context * 2u + 1u], slot, symbol, start, frequency);
            } else {
                uint32_t e = __ldg(buckets + (context << WEX_CUDA_NR2_BUCKET_BITS) +
                                   (slot >> (WEX_NR2_PRECISION - WEX_CUDA_NR2_BUCKET_BITS)));
                symbol = e & 15u; frequency = ((e >> 4) & 0x3fffu) + 1u; start = e >> 18;
                if (slot >= start + frequency) {   /* exact walk inside a mixed bucket */
                    const uint16_t *cdf = cdfs + (size_t)context * 17u;
                    do {
                        start += frequency; ++symbol;
                        frequency = __ldg(cdf + symbol + 1u) - start;
                    } while (slot >= start + frequency);
                }
            }
            state = frequency * (state >> WEX_NR2_PRECISION) + slot - start;
            state = wex_cuda_nr2_renorm(state, s, p);
            word[k / 8u] |= symbol << ((k % 8u) * 4u);
            uint32_t previous = symbol & 7u;
            maximum = max(maximum, previous);   /* = previous after the first symbol */
            context = base + previous * 9u + maximum;
        }
        if (plan.nr3) {   /* coded bytes: hash, CZ check, then the ceil inverse */
            hash = wex_nr3_h32_block(wex_nr3_h32_block(wex_nr3_h32_block(wex_nr3_h32_block(hash,
                       word[0]), word[1]), word[2]), word[3]);
            if (plan.cz && (wex_cuda_has_nibble8(word[0]) | wex_cuda_has_nibble8(word[1]) |
                            wex_cuda_has_nibble8(word[2]) | wex_cuda_has_nibble8(word[3]))) {
                atomicOr(error, WEX_CUDA_ERR_CZ); return;
            }
            if (ceil_matrix && maximum == 5u) {
                if (!scale) { atomicOr(error, WEX_CUDA_ERR_CEIL); return; }
#pragma unroll
                for (uint32_t w = 0; w < 4u; w++) word[w] = wex_cuda_ceil_dbl8(word[w]);
                scales[block] = (unsigned char)(scale - 1u);   /* read for its context already; next_scale is block + 1 */
            }
        }
        out[block] = make_uint4(word[0], word[1], word[2], word[3]);
        scale = next_scale;
    }
    uint32_t expect = WEX_NR_L;
    if (plan.nr3) expect += wex_nr3_h32_final(hash, 16u + blocks * 16u) & WEX_NR3_HASH_MASK;
    if (s.pos != end) atomicOr(error, WEX_CUDA_ERR_STRUCT);
    else if (state != expect) atomicOr(error, plan.nr3 ? WEX_CUDA_ERR_HASH : WEX_CUDA_ERR_STRUCT);
}

/* Dynamic shared memory of the scale kernel on the current device. A device
 * below compute 8.0 can only run code built for __CUDA_ARCH__ < 800, which
 * keeps the direct table in global memory; on sm_80+ the 88 KB covers either
 * build (a pre-Ampere PTX JIT there uses 24 KB of it). */
static inline unsigned wex_cuda_nr2_scale_smem(void) {
    int device = 0, major = 8;
    if (!WEX_CUDA_NR2_SCALE_DIRECT_GLOBAL &&
        (cudaGetDevice(&device) != cudaSuccess ||
         cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess)) {
        (void)cudaGetLastError();
        major = 8;
    }
    return major < 8 ? WEX_CUDA_NR2_SCALE_SMEM_GLOBAL : WEX_CUDA_NR2_SCALE_SMEM;
}

/* The WNM2 blob must already be at base (wex_cuda_nr2_table_view layout).
 * Validate it on the host first (wex_nr2_model_open): the tables rely on it.
 * Also sets the scale kernel's shared-memory opt-in for the current device;
 * build the tables on every device (and in every CUDA runtime) that decodes. */
static inline cudaError_t wex_cuda_build_nr2_tables(unsigned char *base, int with_buckets,
    cudaStream_t stream, WexCudaNr2Tables *tables) {
    cudaError_t status = cudaFuncSetAttribute(wex_cuda_nr2_scales,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wex_cuda_nr2_scale_smem());
    if (status != cudaSuccess) return status;
    wex_cuda_nr2_table_view(base, with_buckets, tables);
    wex_cuda_nr2_build_rows<<<(WEX_CUDA_NR2_ROWS * 8u + 255u) / 256u, 256, 0, stream>>>(
        base, (uint32_t *)tables->weight_rows);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    wex_cuda_nr2_build_scales<<<3u * WEX_NR2_TOTAL / 256u, 256, 0, stream>>>(
        base, (uint32_t *)tables->scale_direct, (unsigned char *)tables->scale_symbol);
    status = cudaGetLastError();
    if (status != cudaSuccess || !with_buckets) return status;
    wex_cuda_nr2_build_buckets<<<(WEX_CUDA_NR2_ROWS << WEX_CUDA_NR2_BUCKET_BITS) / 256u, 256, 0, stream>>>(
        base, (uint32_t *)tables->weight_buckets);
    return cudaGetLastError();
}

/* Kernel A (scales), kernel B (weights, after A), or wex_cuda_launch_nr2: clear
 * *error, A, B on one stream. record and offsets on device; raw receives
 * scales at plan->scale_offset and weights at plan->weight_offset (the
 * canonical slab when plan->weight_base == scale_bytes). Launched apart, A
 * may start once record[0, plan->weight_payload) is on the device. */
static inline cudaError_t wex_cuda_launch_nr2_scales(const WexCudaNr2Plan *plan, const WexCudaNr2Tables *tables,
    const unsigned char *record, const uint32_t *offsets, unsigned char *raw, uint32_t *error, cudaStream_t stream) {
    wex_cuda_nr2_scales<<<plan->block_base[0][3], WEX_CUDA_NR2_THREADS, wex_cuda_nr2_scale_smem(), stream>>>(
        record, offsets, *plan, *tables, raw, error);
    return cudaGetLastError();
}
template<int Lookup>
static inline cudaError_t wex_cuda_launch_nr2_weights(const WexCudaNr2Plan *plan, const WexCudaNr2Tables *tables,
    const unsigned char *record, const uint32_t *offsets, unsigned char *raw, uint32_t *error, cudaStream_t stream) {
    if (Lookup == 0 && !tables->weight_buckets) return cudaErrorInvalidValue;
    wex_cuda_nr2_weights<Lookup><<<plan->block_base[1][3], WEX_CUDA_NR2_THREADS, 0, stream>>>(
        record, offsets, *plan, *tables, raw, error);
    return cudaGetLastError();
}
template<int Lookup>
static inline cudaError_t wex_cuda_launch_nr2_lookup(const WexCudaNr2Plan *plan, const WexCudaNr2Tables *tables,
    const unsigned char *record, const uint32_t *offsets, unsigned char *raw, uint32_t *error, cudaStream_t stream) {
    cudaError_t status = cudaMemsetAsync(error, 0, 4u, stream);
    if (status == cudaSuccess) status = wex_cuda_launch_nr2_scales(plan, tables, record, offsets, raw, error, stream);
    if (status == cudaSuccess) status = wex_cuda_launch_nr2_weights<Lookup>(plan, tables, record, offsets, raw, error, stream);
    return status;
}
static inline cudaError_t wex_cuda_launch_nr2(const WexCudaNr2Plan *plan, const WexCudaNr2Tables *tables,
    const unsigned char *record, const uint32_t *offsets, unsigned char *raw, uint32_t *error, cudaStream_t stream) {
    return wex_cuda_launch_nr2_lookup<WEX_CUDA_NR2_LOOKUP>(plan, tables, record, offsets, raw, error, stream);
}

/* Commit: copy decoded regions to caller buffers only when *error is clear
 * (same stream, after the decode), sparing a host round trip. */
struct WexCudaCommit {
    unsigned char *destination[6];
    uint32_t source[6];   /* offsets in raw */
    uint32_t bytes[6];
};
static __global__ void wex_cuda_commit_regions(const unsigned char *raw, WexCudaCommit commit, const uint32_t *error) {
    if (*(const volatile uint32_t *)error) return;
    uint32_t region = blockIdx.y;
    const unsigned char *from = raw + commit.source[region];
    unsigned char *to = commit.destination[region];
    size_t bytes = commit.bytes[region], stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (!(((uintptr_t)from | (uintptr_t)to | bytes) & 15u)) {
        for (; i < bytes / 16u; i += stride) ((uint4 *)to)[i] = ((const uint4 *)from)[i];
    } else {
        for (; i < bytes; i += stride) to[i] = from[i];
    }
}
static inline cudaError_t wex_cuda_launch_commit(const unsigned char *raw, const WexCudaCommit *commit,
    const uint32_t *error, cudaStream_t stream) {
    wex_cuda_commit_regions<<<dim3(64u, 6u), 256, 0, stream>>>(raw, *commit, error);
    return cudaGetLastError();
}

#endif
