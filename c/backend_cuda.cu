#include "backend_cuda.h"

#include "backend_gpu_compat.h"

/* Optional fmt=8 decode candidate (COLI_CUDA_F8_WARP=2): cuda_fp8.h maps
 * __nv_cvt_fp8_to_halfraw to an sm_89+ cvt instruction, with a bit-manip
 * fallback below 890. CUDA-only; the HIP build keeps the LUT decode. */
#if !(defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)) && defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define COLI_F8_HWCVT 1
#else
#define COLI_F8_HWCVT 0
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef COLI_ANS
#include <dietgpu/ans/GpuANSCodec.h>
#include <dietgpu/utils/StackDeviceMemory.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <sys/stat.h>
#endif

struct RaggedKVEntry {
    const void *key;
    const float *host_l,*host_r;
    float **latent_pages,**rope_pages;
    int length,page_count,K,R;
};

#ifndef COLI_KV_PAGE_TOKENS
#define COLI_KV_PAGE_TOKENS 64
#endif

static void ragged_kv_clear(RaggedKVEntry *e) {
    for (int i=0;i<e->page_count;i++) {
        if (e->latent_pages[i]) cudaFree(e->latent_pages[i]);
        if (e->rope_pages[i]) cudaFree(e->rope_pages[i]);
    }
    std::free(e->latent_pages);
    std::free(e->rope_pages);
    e->latent_pages=e->rope_pages=nullptr;
    e->length=e->page_count=0;
}

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int gs;                    /* quant group size; 0 = per-row scales (#334) */
    int ng;                    /* number of scale groups per row = ceil(I/gs) for fmt=4 */
    size_t scale_count;        /* floats in `scales`: O per-row, O*ng grouped */
    int tracked;
    int weights_owned;
#ifdef COLI_ANS
    size_t archive_bytes;
    int compressed;
#endif
    RaggedKVEntry ragged[512];
    int ragged_count;
};

#ifdef COLI_ANS
struct AnsArenaChunk { uint8_t *p; size_t used,cap; };
#endif
typedef struct {
    int device;
    int compute_major,compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y,*host_kv; size_t host_x_cap,host_y_cap,host_kv_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *pipe_buf[27]; size_t pipe_cap[27];   /* scratch persistenti del resident pipeline */
    cudaStream_t stream;
    cudaEvent_t ev_done; int ev_done_ok;        /* resident-group issue completion (#431 PR-C0) */
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
    int group_pending; size_t group_pending_bytes;   /* async expert-group in flight (Inc.4) */
#ifdef COLI_ANS
    void *ans_raw; size_t ans_raw_cap;
    void *ans_host; size_t ans_host_cap;
    int ans_copy_pending;
    dietgpu::StackDeviceMemory *ans_scratch;
    std::vector<AnsArenaChunk> *ans_chunks;
#endif
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
    int ggs,ugs,dgs;      /* per-tensor quant group size; 0 = per-row scales (#334 fmt=4) */
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static uint64_t g_device_group_calls[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_experts[COLI_CUDA_MAX_DEVICES];
static uint64_t g_device_group_rows[COLI_CUDA_MAX_DEVICES];
static double g_device_group_h2d_ms[COLI_CUDA_MAX_DEVICES];
static double g_device_group_kernel_ms[COLI_CUDA_MAX_DEVICES];
static double g_device_group_d2h_ms[COLI_CUDA_MAX_DEVICES];
static std::mutex g_group_stats_mu;
#ifdef COLI_ANS
static FILE *g_ans_sidecar;
static int g_ans_sidecar_pack;
#if defined(__linux__)
static int g_ans_direct_fd=-1;
static off_t g_ans_direct_off;
#endif
static uint64_t g_ans_load_records;
static double g_ans_header_s,g_ans_read_s,g_ans_stage_s,g_ans_enqueue_s;
static int g_ans_profile_printed;
static double ans_now_s(){
    using clock=std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
#endif

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    (void)cudaGetLastError();   /* consume the sticky error: a failed call must
                                   not poison the next launch's error check */
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!cuda_ok(cudaSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

struct ColiGpuContext {
    int device;
    cudaStream_t stream;
    cudaStream_t upload_stream;
    int clock_rate_khz;
    uint64_t caps;
    int healthy;
    ColiGpuFaultPoint fault_point;
    int fault_occurrence;
    int fault_seen;
    ColiGpuTelemetry telemetry;
};

struct ColiGpuTensor {
    ColiGpuContext *ctx;
    void *data;
    float *scales;
    size_t data_bytes;
    size_t scale_count;
    int format;
    int rows;
    int columns;
    int group_size;
    int groups;
};

struct ColiGpuArena {
    ColiGpuContext *ctx;
    unsigned char *data;
    size_t capacity;
};

struct ColiGpuRouter {
    ColiGpuContext *ctx;
    ColiGpuRouteConfig config;
    int max_rows;
    int rows;
    float *scores;
    float *choices;
    int *selected;
    float *weights;
    void *allocation;
    int *host_selected;
    volatile int *host_status;
    int *device_status;
};

typedef struct ColiGpuExpertTransfer {
    unsigned char *host_staging;
    cudaEvent_t ready;
    int published;
    struct ColiGpuExpertTransfer *next;
} ColiGpuExpertTransfer;

typedef struct {
    unsigned char *allocation;
    void *gate_data;
    float *gate_scales;
    void *up_data;
    float *up_scales;
    void *down_data;
    float *down_scales;
    cudaEvent_t use_done;
    int use_recorded;
} ColiGpuExpertBank;

typedef struct {
    ColiGpuExpertBank *bank;
    ColiGpuExpertTransfer *publication;
} ColiGpuExpertSnapshot;

typedef struct {
    ColiGpuExpertBank bank[2];
    int active_bank;
    int expert_id;
    uint64_t generation;
    int published;
    ColiGpuExpertTransfer *publication;
} ColiGpuExpertSlot;

struct ColiGpuExpertCache {
    ColiGpuContext *ctx;
    ColiGpuExpertCacheConfig config;
    ColiGpuExpertSlot *slots;
    ColiGpuExpertSnapshot *snapshots;
    size_t snapshot_capacity;
    size_t gate_data_bytes;
    size_t gate_scale_bytes;
    size_t down_data_bytes;
    size_t down_scale_bytes;
    size_t bank_bytes;
    ColiGpuExpertFaultPoint fault;
    int fault_occurrence;
    unsigned test_delay_ms;
    std::atomic<int> test_hold_snapshot;
    std::atomic<int> test_snapshot_entered;
    ColiGpuExpertTransfer *transfers;
    std::mutex mutex;
    int healthy;
    volatile int *host_status;
    int *device_status;
};

struct ColiGpuKdaState {
    ColiGpuContext *ctx;
    ColiGpuArena *arena;
    size_t state_offset;
    size_t window_offset;
    ColiGpuKdaConfig config;
    int position;
};

typedef struct {
    float *latent;
    float *index_keys;
    float *index_gates;
    void *allocation;
} ColiGpuMlaPage;

struct ColiGpuMlaState {
    ColiGpuContext *ctx;
    ColiGpuMlaConfig config;
    ColiGpuMlaPage *pages;
    ColiGpuMlaPage *device_pages;
    int page_count;
    int page_table_capacity;
    int max_pages;
    int length;
    int capacity;
    uint64_t payload_copy_bytes;
    ColiGpuMlaFaultPoint fault_point;
    int fault_occurrence;
    int fault_seen;
    volatile int *host_status;
    int *device_status;
    ColiGpuMlaLaunchInfo launch_info;
};

static int select_device_ordinal(int device) {
    if (g_current_device == device) return 1;
    if (!cuda_ok(cudaSetDevice(device), "select device")) return 0;
    g_current_device = device;
    return 1;
}

extern "C" int coli_gpu_context_create(ColiGpuContext **out, int device) {
    int available = 0;
    cudaDeviceProp prop{};
    ColiGpuContext *ctx = nullptr;
    if (!out) return 0;
    *out = nullptr;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    if (device < 0 || device >= available) {
        std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n",
                     device, available - 1);
        return 0;
    }
    if (!select_device_ordinal(device)) return 0;
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) return 0;
    ctx = static_cast<ColiGpuContext *>(std::calloc(1, sizeof(*ctx)));
    if (!ctx) return 0;
    ctx->device = device;
    ctx->clock_rate_khz = prop.clockRate;
    ctx->caps = COLI_GPU_CAP_STREAM_ORDERED |
                COLI_GPU_CAP_INT4_GS64;
    ctx->healthy = 1;
    if (!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking),
                 "context stream creation")) {
        std::free(ctx);
        return 0;
    }
    if (!cuda_ok(cudaStreamCreateWithFlags(
            &ctx->upload_stream, cudaStreamNonBlocking),
            "context upload stream creation")) {
        (void)cudaStreamDestroy(ctx->stream);
        std::free(ctx);
        return 0;
    }
    *out = ctx;
    return 1;
}

extern "C" int coli_gpu_context_probe(ColiGpuContext *ctx, uint64_t required_caps) {
    if (!ctx || !ctx->healthy) return 0;
    if ((required_caps & ~ctx->caps) != 0) return 0;
    return 1;
}

extern "C" int coli_gpu_context_healthy(const ColiGpuContext *ctx) {
    return ctx && ctx->healthy;
}

extern "C" int coli_gpu_context_memory_info(
    ColiGpuContext *ctx, size_t *free_bytes, size_t *total_bytes) {
    return ctx && ctx->healthy && free_bytes && total_bytes &&
           select_device_ordinal(ctx->device) &&
           cuda_ok(cudaMemGetInfo(free_bytes, total_bytes),
                   "context memory info");
}

extern "C" int coli_gpu_context_inject_fault(
    ColiGpuContext *ctx, ColiGpuFaultPoint point, int occurrence) {
    if (!ctx || !ctx->healthy || point <= COLI_GPU_FAULT_NONE ||
        point > COLI_GPU_FAULT_STREAM_SYNC || occurrence < 0)
        return 0;
    ctx->fault_point = point;
    ctx->fault_occurrence = occurrence;
    ctx->fault_seen = 0;
    return 1;
}

extern "C" int coli_gpu_context_consume_fault(
    ColiGpuContext *ctx, ColiGpuFaultPoint point) {
    if (!ctx || ctx->fault_point != point) return 0;
    if (ctx->fault_seen++ != ctx->fault_occurrence) return 0;
    ctx->fault_point = COLI_GPU_FAULT_NONE;
    ctx->fault_seen = 0;
    return 1;
}

extern "C" int coli_gpu_context_sync(ColiGpuContext *ctx) {
    if (!ctx || !ctx->healthy) return 0;
    if (coli_gpu_context_consume_fault(ctx, COLI_GPU_FAULT_STREAM_SYNC)) {
        ctx->healthy = 0;
        return 0;
    }
    if (!select_device_ordinal(ctx->device)) {
        ctx->healthy = 0;
        return 0;
    }
    if (!cuda_ok(cudaStreamSynchronize(ctx->stream), "context stream synchronize")) {
        ctx->healthy = 0;
        return 0;
    }
    return 1;
}

extern "C" void coli_gpu_context_mark_unhealthy(ColiGpuContext *ctx) {
    if (ctx) ctx->healthy = 0;
}

extern "C" int coli_gpu_context_advertise_pipeline(ColiGpuContext *ctx) {
    if (!ctx || !ctx->healthy) return 0;
    ctx->caps |= COLI_GPU_CAP_PIPELINE;
    return 1;
}

extern "C" void coli_gpu_context_telemetry(const ColiGpuContext *ctx,
                                            ColiGpuTelemetry *out) {
    if (!out) return;
    if (ctx) *out = ctx->telemetry;
    else std::memset(out, 0, sizeof(*out));
}

extern "C" void coli_gpu_context_destroy(ColiGpuContext *ctx) {
    if (!ctx) return;
    if (select_device_ordinal(ctx->device)) {
        if (ctx->upload_stream) (void)cudaStreamDestroy(ctx->upload_stream);
        if (ctx->stream) (void)cudaStreamDestroy(ctx->stream);
    }
    std::free(ctx);
}

/* fmt=6 (E8/IQ3) geometry, mirroring quant.h. A super-block packs 256 weights
 * into 98 bytes: 64 codebook indices, 8 words of (4x7 signs + 4-bit sub-scale),
 * and one fp16 super-scale. Scales live INSIDE the block, so fmt=6 tensors carry
 * no separate scale array (#452). */
#define COLI_E8_QK      256
#define COLI_E8_SUB      32
#define COLI_E8_BBYTES   98

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2 || fmt == 4) return (size_t)(I + 1) / 2;   /* fmt=4: same packed int4 */
    if (fmt == 3) return (size_t)(I + 3) / 4;
    if (fmt == 4) return (size_t)(I + 1) / 2;   /* grouped int4: nibbles like fmt 2 */
    if (fmt == 7) return (size_t)(I + 1) / 2;   /* MXFP4: e2m1 nibbles, 2 per byte */
    if (fmt == 6) return (size_t)(((int64_t)I + COLI_E8_QK - 1) / COLI_E8_QK) * COLI_E8_BBYTES;
    if (fmt == 8) return (size_t)I;             /* fp8-e4m3: raw bytes, layout of fmt=1 */
    return 0;
}

/* The E8 codebook, uploaded once per device from quant.h's e8_grid so the table
 * has a single source of truth and cannot drift from the CPU decoder. */
__constant__ uint8_t c_e8_grid[256][4];

/* The fmt=8 e4m3 decode table, uploaded once per device from quant.h's
 * E4M3_LUT — same single-source-of-truth arrangement as c_e8_grid. Uploads of
 * fmt=8 tensors are refused until it is published (g_fp8_lut_ready): a kernel
 * reading the zero-initialized table would compute silent zeros, the exact
 * failure mode this format's dispatch work exists to prevent. */
__constant__ float c_e4m3[256];
static int g_fp8_lut_ready;

/* A super-block is 98 bytes, so nothing inside it is guaranteed 4- or 2-byte
 * aligned: assemble the words byte-wise instead of dereferencing. */
__device__ __forceinline__ uint32_t e8_ld_u32(const uint8_t *p){
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
/* Mirrors e8_fp16_to_f32 rather than calling __half2float, so the two decoders
 * cannot disagree on subnormals. */
__device__ __forceinline__ float e8_fp16(const uint8_t *p){
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1]<<8);
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1F, man=h&0x3FF, bits;
    if (!exp)         bits = man ? (sign|((127u-15u+1u-1u)<<23)|(man<<13)) : sign;
    else if (exp==31) bits = sign|0x7F800000u|(man<<13);
    else              bits = sign|((exp+112u)<<23)|(man<<13);
    float f; memcpy(&f,&bits,4); return f;
}
/* Expand one 32-weight sub-block; mirrors e8_expand_sub in quant.h. */
__device__ __forceinline__ void e8_expand_sub_dev(const uint8_t *blk, int ib, float d, float *out){
    uint32_t word = e8_ld_u32(blk + COLI_E8_QK/4 + ib*4);
    float db = d * (0.5f + (float)((word>>28)&0xF)) * 0.5f;
    const uint8_t *idx = blk + ib*8;
    for (int l=0;l<4;l++){
        uint32_t seven=(word>>(7*l))&0x7F;
        const uint8_t *g0=c_e8_grid[idx[l*2+0]], *g1=c_e8_grid[idx[l*2+1]];
        int par=0;
        for (int j=0;j<8;j++){
            int neg = j<7 ? (int)((seven>>j)&1) : 0;
            if (j<7) par^=neg; else neg=par;        /* odd parity closes the block */
            float mag = (j<4 ? (float)g0[j] : (float)g1[j-4]) * 0.5f;
            out[l*8+j] = neg ? -mag*db : mag*db;
        }
    }
}

/* ---- MXFP4 (OCP microscaling FP4), fmt=7 -----------------------------------
 * Same layout the CPU path decodes in quant.h's matmul_mxfp4, and the same two
 * tricks, so the two agree bit for bit:
 *
 *   packed [O, I/2]  u8 — e2m1 nibbles, LOW nibble = even column, bit3 = sign,
 *                         bits 0..2 index {0,.5,1,1.5,2,3,4,6}
 *   scales [O, I/32] u8 — ue8m0 exponent per 32-column group, w = v * 2^(s-127)
 *
 * The exponent is decoded as a bit pattern rather than exp2f: (uint32)s << 23
 * reinterpreted as float IS 2^(s-127) for s in [1,254], and reproduces the CPU
 * path's documented edge behaviour exactly -- s=0 gives +0 and s=255 gives +inf
 * on both sides. Using exp2f here would agree for the normal range and diverge
 * at the ends, which is precisely where a silent mismatch would hide.
 *
 * The LUT holds DOUBLED values so every entry is an exact small integer; the
 * compensating 0.5f rides along in mx4_scale_dev, as it does on the CPU. */
/* Decoded arithmetically rather than from a __constant__ table: a file-scope
 * __constant__ array with static linkage is initialised per translation unit,
 * and this kernel is also compiled into the HIP build and the DLL, where that
 * silently yields garbage. The magnitude is 2^(exp-1) * 0.5 for exp in 1..3 and
 * 0 for exp 0, which is exactly the OCP e2m1 table {0,.5,1,1.5,2,3,4,6}. */
__device__ static inline float mx4_decode(int n) {
    int mant = n & 1, exp = (n >> 1) & 3;
    float mag = exp ? ldexpf(1.0f + 0.5f * (float)mant, exp - 1) : 0.5f * (float)mant;
    return (n & 8) ? -mag : mag;
}

__device__ static inline float mx4_scale_dev(uint8_t s) {
    union { uint32_t u; float f; } b;
    b.u = static_cast<uint32_t>(s) << 23;
    return b.f;
}

/* e2m1 nibble at column i of a packed row. */
__device__ static inline float mx4_weight_at(const uint8_t *q, int i) {
    uint8_t v = q[i >> 1];
    return mx4_decode((i & 1) ? (v >> 4) : (v & 15));
}

/* Generic per-element weight decode for the kernels that need one (the absorb
 * kernels and the generic grouped-expert path). fmt=3 (int2) is now an EXPLICIT
 * branch and the fall-through is a refusal.
 *
 * It used to be the other way round: int2 was the fall-through, so every format
 * this function does not decode -- fmt=5 (int3-g64), fmt=6 (E8/IQ3), fmt=8
 * (fp8-e4m3), and anything added later -- was read as 2-bit values and returned
 * numbers. Meanwhile the CPU functions doing the same job on the same tensor,
 * qt_addrow and qt_matvec_rows (colibri.c), both exit(1) naming the function and
 * the fmt. Two backends, identical unsupported input, one refusing and one
 * fabricating: that asymmetry is the defect, independent of any particular
 * format's arrival.
 *
 * WHY __trap() AND NOT A DIAGNOSTIC. This is device code inside a running
 * kernel; there is no stderr to name the tensor on and no way to unwind. __trap
 * aborts the kernel and poisons the context, so the next cuda_ok() call on the
 * host reports a failure instead of the caller consuming fabricated values --
 * the same "stop rather than misread" outcome as the CPU's exit(1), reached the
 * only way device code can reach it.
 *
 * IT IS A BACKSTOP, NOT THE PRIMARY GATE -- and the gates above it are several
 * DIFFERENT checks, not one. Naming them exactly, because "every launcher checks
 * coli_cuda_weight_at_supported" would be false and a reader will verify it:
 *   - UPLOAD is the widest gate. coli_cuda_tensor_upload{,_g} refuse when
 *     row_bytes(fmt,I) == 0, which is every fmt this file has no row stride for --
 *     fmt=5 (int3-g64), every negative fmt, every unknown fmt. Those can never
 *     become a ColiCudaTensor at all, so the uploadable set is {0,1,2,3,4,6,7,8}.
 *   - quant_matmul dispatches 6, 7, 4 and 8 in explicit branches of its own, so
 *     the generic else that calls this function sees only 0/1/2/3 out of that set.
 *   - the generic grouped-expert path refuses gf>3 || uf>3 || df>3 on the host.
 *   - the absorb call sites (the other caller of this function) are the ones
 *     gated by coli_cuda_weight_at_supported, via absorb_fmt_ok below -- one
 *     caller, not "every launcher".
 * Each of those is sufficient on its own for the sites it covers; this trap
 * exists because none of them is a property of THIS function. Reaching this line
 * means a launch site got past its own gate, which is a bug in that gate. */
__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) { /* per-row int4 after offset_to_signed_s4 XOR 0x88 */
        uint8_t v = q[i >> 1];
        int n = (i & 1) ? (v >> 4) : (v & 15);
        return static_cast<float>(n & 8 ? n - 16 : n);
    }
    if (fmt == 4) { /* Colibri grouped int4 stores level + 8 (0..15). */
        uint8_t v = q[i >> 1];
        int n = (i & 1) ? (v >> 4) : (v & 15);
        return static_cast<float>(n - 8);
    }
    if (fmt == 3) {                                           /* int2 */
        uint8_t v = q[i >> 2];
        return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
    }
    __trap();
    return 0.0f;   /* not reached: __trap() does not return */
}

/* Scale for output `row`, input element `k`. fmt=4 (grouped int4) stores ng
 * scales per row at scales[row*ng + k/gs]; every other quantized format has
 * one scale per row at scales[row]. Mirrors quant_matmul's fmt==4 branch so the
 * attention absorb kernels apply per-group scales instead of the per-row
 * (fmt=2) semantic that crashed #298's g64 kv_b. */
__device__ static float absorb_scale(const float *wscale, int fmt, int gs, int ng, int row, int k) {
    if (!fmt) return 1.f;
    if (fmt != 4) return wscale[row];
    int g = k / gs; if (g >= ng) g = ng - 1;   /* tail of the last (partial) group */
    return wscale[(size_t)row * ng + g];
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

/* ---- fmt=8 (fp8-e4m3) warp decode/accumulate helpers -----------------------
 * The fmt=8 dot products are compared against the CPU reference (quant.h
 * matmul_fp8) by the cross-tier parity tests, so this TU must never be built
 * with fast-math: nvcc's --use_fast_math implies -ftz=true, which flushes the
 * scale*subnormal contributions the denormal-scale test pins down. (-ftz has
 * no macro of its own to test; the Makefile pins -ftz=false on the nvcc line.) */
#if defined(__USE_FAST_MATH__) || defined(__FAST_MATH__)
#error "backend_cuda.cu: fmt=8 kernels forbid fast-math builds (FTZ breaks CPU parity)"
#endif

/* Fixed-order 5-shuffle reduce over a 32-lane logical warp. Width pinned to 32
 * so a HIP wave64 device does not fold two output rows into one reduction. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define f8_shfl_down(v,o) __shfl_down((v),(o),32)
#else
#define f8_shfl_down(v,o) __shfl_down_sync(0xffffffffu,(v),(o),32)
#endif

/* Decode one e4m3 byte. Default: a __shared__ copy of c_e4m3 — per-thread
 * data-dependent indices are the __constant__ cache's documented worst case
 * (accesses to different addresses within a warp serialize), while shared
 * memory takes them at full rate. HW=1 (COLI_CUDA_F8_WARP=2, CUDA-only): the
 * cuda_fp8.h route. Every finite e4m3 value is exactly representable in f16,
 * so the two paths should agree bit for bit — but the on-device 256-value
 * sweep is the authority, and the LUT stays the default until the sweep
 * certifies the cvt path on the target silicon. */
template<int HW>
__device__ __forceinline__ float f8_dec(const float *slut, uint8_t b){
#if COLI_F8_HWCVT
    if (HW) return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
#endif
    return slut[b];
}

/* One 128-column scale block, one warp: 32 lanes x 4 bytes is the exact
 * FP8_BLOCK fit. Weights are decoded once and reused across ns activation
 * rows (weights are the traffic; activations come from cache), then each
 * row's per-lane products are tree-reduced in fixed order — the block partial
 * lands in LANE 0 of gp[]/up_[] (other lanes hold partials). vec picks the
 * LOAD width only (uchar4/float4 on aligned full blocks, guarded bytes on
 * tails and unaligned bases — e.g. zero-copy views: correct, slower); both
 * load paths feed ONE shared product/accumulate sequence below, and the
 * vec/byte parity test asserts their bit-equality on identical data. Tail
 * padding contributes +0.0f terms, which can at most flip a -0.0 partial to
 * +0.0. DUAL folds a second weight row (gate+up) against the same
 * activations without re-reading x. The per-row arithmetic never depends on
 * ns or s, so S-tiling cannot reorder a dot (B5). */
template<int HW,int DUAL>
__device__ __forceinline__ void f8w_block(const float *slut,const uint8_t *gr,
        const uint8_t *ur,const float *xs,int xstride,int ns,int base,int len,
        int vec,float *gp,float *up_){
    int lane=threadIdx.x&31,i0=base+lane*4,full=vec&&len==128;
    float gw[4],uw[4]={0,0,0,0};
    if(full){
        uchar4 q=*(const uchar4*)(gr+i0);
        gw[0]=f8_dec<HW>(slut,q.x);gw[1]=f8_dec<HW>(slut,q.y);
        gw[2]=f8_dec<HW>(slut,q.z);gw[3]=f8_dec<HW>(slut,q.w);
        if(DUAL){uchar4 r=*(const uchar4*)(ur+i0);
            uw[0]=f8_dec<HW>(slut,r.x);uw[1]=f8_dec<HW>(slut,r.y);
            uw[2]=f8_dec<HW>(slut,r.z);uw[3]=f8_dec<HW>(slut,r.w);}
    }else for(int k=0;k<4;k++){
        int in=i0+k<base+len;
        gw[k]=in?f8_dec<HW>(slut,gr[i0+k]):0.f;
        if(DUAL)uw[k]=in?f8_dec<HW>(slut,ur[i0+k]):0.f;
    }
    for(int s=0;s<ns;s++){
        const float *xr=xs+(size_t)s*xstride;
        float xv[4];
        if(full){
            float4 xf=*(const float4*)(xr+i0);
            xv[0]=xf.x;xv[1]=xf.y;xv[2]=xf.z;xv[3]=xf.w;
        }else for(int k=0;k<4;k++) xv[k]=i0+k<base+len?xr[i0+k]:0.f;
        float g=0,u=0;
        for(int k=0;k<4;k++){g+=xv[k]*gw[k];if(DUAL)u+=xv[k]*uw[k];}
        for(int off=16;off;off>>=1){g+=f8_shfl_down(g,off);if(DUAL)u+=f8_shfl_down(u,off);}
        gp[s]=g;if(DUAL)up_[s]=u;
    }
}

/* Dense fmt=8 rework: quant_matmul's fmt=8 branch as its OWN kernel, so
 * COLI_CUDA_F8_WARP=0 restores the fully original dense behavior too (the
 * host picks in quant_matmul_launch). Same grid/block contract as
 * quant_matmul: one 256-thread block per (o,s). Accumulation mirrors the CPU
 * reference: f32 partial per 128-block (f8w_block), the scale applied ONCE
 * per partial, double across blocks. Warps stride the block axis; the
 * cross-warp double sum runs in fixed warp order, so the reduction order is
 * a pure function of the dims. Decode reads a shared copy of c_e4m3
 * (data-dependent __constant__ indices serialize); the cvt candidate stays
 * grouped-path-only until certified. NaN bytes decode to NaN and propagate,
 * same policy as the CPU path. */
__global__ static void quant_matmul_f8w(float *y,const float *x,const void *weights,
                                        const float *scales,int S,int I,int O){
    int o=blockIdx.x,s=blockIdx.y;
    const float *xs=x+(size_t)s*I;
    __shared__ float slut[256];
    __shared__ double dsum[32];
    for(int i=threadIdx.x;i<256;i+=blockDim.x) slut[i]=c_e4m3[i];
    __syncthreads();
    const uint8_t *wrow=(const uint8_t*)weights+(size_t)o*I;
    const float *scl=scales+(size_t)(o>>7)*(size_t)((I+127)>>7);
    int warp=threadIdx.x>>5,nw=blockDim.x>>5,nblk=(I+127)>>7;
    int vec=!(I&3)&&!((size_t)wrow&3)&&!((size_t)xs&15);
    double a=0;
    for(int bi=warp;bi<nblk;bi+=nw){
        int base=bi<<7,len=I-base<128?I-base:128;
        float p;
        f8w_block<0,0>(slut,wrow,nullptr,xs,0,1,base,len,vec,&p,nullptr);
        if(!(threadIdx.x&31)) a+=(double)p*scl[bi];
    }
    if(!(threadIdx.x&31)) dsum[warp]=a;
    __syncthreads();
    if(!threadIdx.x){
        for(int w=1;w<nw;w++) a+=dsum[w];
        y[(size_t)s*O+o]=(float)a;
    }
    (void)S;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb, int gs, int ng) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    if (fmt == 6) {
        /* E8/IQ3: decode is per 32-weight sub-block, so threads stride over
         * sub-blocks rather than elements -- expanding once per 32 weights
         * instead of redoing the word/parity work for every element. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        int nsub = (I + COLI_E8_SUB - 1) / COLI_E8_SUB;
        for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
            const uint8_t *blk = wrow + (size_t)(sb / (COLI_E8_QK/COLI_E8_SUB)) * COLI_E8_BBYTES;
            float w[COLI_E8_SUB];
            e8_expand_sub_dev(blk, sb % (COLI_E8_QK/COLI_E8_SUB), e8_fp16(blk+96), w);
            int off = sb*COLI_E8_SUB, n = I-off < COLI_E8_SUB ? I-off : COLI_E8_SUB;
            for (int k=0;k<n;k++) sum += xs[off+k]*w[k];
        }
    } else if (fmt == 7) {
        /* MXFP4: one ue8m0 exponent per 32 columns. The SCALAR CPU reference
         * (quant.h matmul_mxfp4's scalar loop -- its AVX2 sibling applies the
         * scale per 8-lane FMA instead and disagrees with it at s=255)
         * accumulates each 32-group UNSCALED and multiplies the group
         * subtotal by the scale once. For scales in [0,254] -- every value a
         * real checkpoint contains -- the scale is a finite power of two,
         * scaling each element or the subtotal is exact either way, so
         * threads stride over columns exactly as before and outputs stay
         * bit-identical with prior builds. s=255 decodes to +inf (the
         * documented edge), and there the application point is visible:
         * per-element scaling turns every zero product into 0*inf = NaN,
         * where the scalar reference's finite-subtotal-times-inf keeps the
         * group's sign. Rows carrying a 255 scale byte therefore take the
         * per-group path mirroring that scalar loop; the row's +-inf/NaN
         * classification does not depend on summation order unless several
         * large-finite (s~254) group results overflow only in aggregate --
         * out-of-domain either way -- so the tree reduce below needs no
         * change. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const uint8_t *scl = reinterpret_cast<const uint8_t *>(scales) + (size_t)o * ng;
        __shared__ int scale_inf;
        if (!threadIdx.x) scale_inf = 0;
        __syncthreads();
        for (int g = threadIdx.x; g < ng; g += blockDim.x)
            if (scl[g] == 255) scale_inf = 1;
        __syncthreads();
        if (!scale_inf) {
            for (int i = threadIdx.x; i < I; i += blockDim.x) {
                int g = i >> 5;
                if (g >= ng) g = ng - 1;
                sum += xs[i] * mx4_weight_at(wrow, i) * mx4_scale_dev(scl[g]);
            }
        } else {
            for (int g = threadIdx.x; g < ng; g += blockDim.x) {
                /* same element->group mapping as the clamp above: the last
                 * group takes every remaining column, a group past the data
                 * contributes nothing */
                int base = g << 5, end = base + 32;
                if (g == ng - 1 || end > I) end = I;
                if (base >= end) continue;
                float ga = 0.0f;
                for (int i = base; i < end; i++)
                    ga += xs[i] * mx4_weight_at(wrow, i);
                sum += ga * mx4_scale_dev(scl[g]);
            }
        }
    } else if (fmt == 4) {
        /* Grouped int4: one f32 scale per gs elements along I (ng groups per row).
         * Scale layout: scales[o*ng + g]. Each thread strides through I, applying
         * the appropriate group scale as it crosses group boundaries. This matches
         * the CPU matmul_i4_grouped accumulation exactly. */
        const float *scl = scales + (size_t)o * ng;
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            int g = i / gs;
            if (g >= ng) g = ng - 1;  /* tail elements in the last (partial) group */
            sum += xs[i] * weight_at(weights, fmt, row, i) * scl[g];
        }
    } else if (fmt == 8) {
        /* fp8-e4m3 (matmul_fp8): one byte per weight (layout of fmt=1), one f32
         * scale per 128x128 BLOCK of [O,I] — scales[(o/128)*ceil(I/128) + i/128].
         * The block edge is a fixed property of the format (FP8_BLOCK), so the
         * geometry derives from I alone and gs/ng are ignored: this branch is
         * correct no matter which call site launched it. NaN bytes decode to NaN
         * through the LUT and propagate, same policy as the CPU path. This is
         * the ORIGINAL dense path, kept for COLI_CUDA_F8_WARP=0; the default
         * routes fmt=8 to quant_matmul_f8w instead (quant_matmul_launch). */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const float *scl = scales + (size_t)(o >> 7) * (size_t)((I + 127) >> 7);
        for (int i = threadIdx.x; i < I; i += blockDim.x)
            sum += xs[i] * c_e4m3[wrow[i]] * scl[i >> 7];
    } else {
        for (int i = threadIdx.x; i < I; i += blockDim.x)
            sum += xs[i] * weight_at(weights, fmt, row, i);
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        /* fmt 4/6/7/8 already applied their scaling inside the loop: 4 and 7 are
         * per-group (one scale per gs / per 32 columns), 6 carries it in the
         * block header, 8 reads a per-128 block scale alongside the weights.
         * Only the per-row formats get the trailing multiply --
         * and for fmt=7 `scales` points at ue8m0 BYTES, so reading it as float
         * here does not merely double-scale, it reads garbage. */
        y[(size_t)s * O + o] = (fmt && fmt != 4 && fmt != 6 && fmt != 7 && fmt != 8) ? partial[0] * scales[o] : partial[0];
}

/* fmt=6 activation rotation, y = Q^T x for Q = D*H/sqrt(n) (#452). One block per
 * row; the power-of-two block is staged in shared memory, capping n at 4096
 * floats -- which covers every block GLM produces (6144 -> 2048+4096, 1536 ->
 * 512+1024). The sign stream is regenerated in-kernel from the same xorshift64*
 * that quant.h's e8_signs uses, so no rotation data is stored or uploaded.
 *
 * Placement note: all routed experts of a layer share one gate/up input, so that
 * rotation belongs to the CALLER (once per layer). This kernel exists for the
 * down projection, whose input is the per-expert silu(gate)*up product and so
 * cannot be shared -- mirroring colibri.c's split at moe(). */
__global__ static void e8_rot_rows_kernel(float *rows, int dim, int off, int n){
    extern __shared__ float sh[];
    __shared__ uint8_t sbits[4096/8];
    if (!threadIdx.x) {
        uint64_t s = 417u + (uint64_t)n;
        for (int i=0;i<(n+7)/8;i++){
            s^=s>>12; s^=s<<25; s^=s>>27;
            sbits[i] = (uint8_t)((s*2685821657736338717ULL)>>56);
        }
    }
    __syncthreads();
    float *row = rows + (size_t)blockIdx.x*dim + off;
    for (int i=threadIdx.x;i<n;i+=blockDim.x){
        float v=row[i];
        sh[i] = (sbits[i>>3]>>(i&7)&1) ? -v : v;
    }
    __syncthreads();
    for (int len=1;len<n;len<<=1){
        for (int j=threadIdx.x;j<n/2;j+=blockDim.x){
            int i = (j/len)*(len<<1) + (j%len);
            float u=sh[i], v=sh[i+len];
            sh[i]=u+v; sh[i+len]=u-v;
        }
        __syncthreads();
    }
    float sc=rsqrtf((float)n);
    for (int i=threadIdx.x;i<n;i+=blockDim.x) row[i]=sh[i]*sc;
}

/* Rotate nr rows in place, tiling non-power-of-two dims block-diagonally exactly
 * as e8_rot_rows does. Returns 0 if a block exceeds the shared-memory cap. */
static int e8_rot_rows_dev(float *rows, int nr, int dim, cudaStream_t stream){
    int off = 0;
    while (off < dim) {
        int rem = dim-off, b = rem & (-rem);
        while (b > 4096) b >>= 1;
        e8_rot_rows_kernel<<<(unsigned)nr, 256, (size_t)b*sizeof(float), stream>>>(rows, dim, off, b);
        if (cudaGetLastError() != cudaSuccess) return 0;
        off += b;
    }
    return 1;
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

__global__ static void swiglu_clamped_kernel(
    float *gate, const float *up, size_t n, float limit) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = fminf(gate[i], limit);
        float u = fmaxf(-limit, fminf(up[i], limit));
        gate[i] = (g / (1.0f + expf(-g))) * u;
    }
}

/* Four warps share one A tile and compute 16x64 outputs.  This matters for
 * prefill: the first prototype reloaded/converter A once per 16 output cols. */
__global__ static void w4a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);           /* [Ntile,Ktile] == B col-major */
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* Gate and up use the same input.  Eight warps compute both 16x64 projections
 * while sharing the FP32->FP16 conversion of A. */
__global__ static void w4a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;size_t rb=(size_t)(K+1)/2;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    __syncwarp();   /* i due fratelli sopra la portano; stessa lettura cross-lane di out[warp]
                     * subito sotto -> stesso requisito di visibilita' (racecheck: 3 hazard qui).
                     * EN: the two sibling kernels carry this; same cross-lane read follows. */
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

/* Native fmt=6 expert groups.  One block owns one (expert,row,output) and
 * expands each 32-weight E8 sub-block once for both gate/up projections. */
__global__ static void grouped_hidden_e8_dual(float *gate,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    size_t rb=row_bytes(6,D);
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*rb;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    int nsub=(D+COLI_E8_SUB-1)/COLI_E8_SUB;
    for(int sb=threadIdx.x;sb<nsub;sb+=blockDim.x){
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *gb=gr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        const uint8_t *ub=ur+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float gw[COLI_E8_SUB],uw[COLI_E8_SUB];
        e8_expand_sub_dev(gb,ib,e8_fp16(gb+96),gw);
        e8_expand_sub_dev(ub,ib,e8_fp16(ub+96),uw);
        int off=sb*COLI_E8_SUB,n=D-off<COLI_E8_SUB?D-off:COLI_E8_SUB;
        for(int k=0;k<n;k++){float v=xs[off+k];ga+=v*gw[k];ua+=v*uw[k];}
    }
    __shared__ float gp[256],up[256];gp[threadIdx.x]=ga;up[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];up[threadIdx.x]+=up[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){float g=gp[0];gate[(size_t)(d.offset+s)*I+o]=(g/(1.f+expf(-g)))*up[0];}
}

__global__ static void grouped_down_e8(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    size_t rb=row_bytes(6,I);const uint8_t *wr=(const uint8_t*)d.d+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    int nsub=(I+COLI_E8_SUB-1)/COLI_E8_SUB;
    for(int sb=threadIdx.x;sb<nsub;sb+=blockDim.x){
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *blk=wr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float w[COLI_E8_SUB];e8_expand_sub_dev(blk,ib,e8_fp16(blk+96),w);
        int off=sb*COLI_E8_SUB,n=I-off<COLI_E8_SUB?I-off:COLI_E8_SUB;
        for(int k=0;k<n;k++)sum+=xs[off+k]*w[k];
    }
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* Fused epilogue: silu(gate)*up lands here instead of a third kernel —
     * the exact silu_mul expression on the exact same inputs, so bit-identical,
     * and the up[] round-trip through global memory disappears. up stays a
     * param so the launch sites keep their signature. */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0]*d.gs[o],u=upv[0]*d.us[o];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

/* fmt=4 grouped-int4 variants (#334): identical structure to the w4 kernels,
 * but the scale varies along the input dimension — sc[o*ng + i/gs], applied
 * per element inside the accumulation (gs is even, so a packed byte never
 * straddles a group). gs<=0 degrades to per-row (ng=1), so mixed fmt2/fmt4
 * groups run correctly through this one kernel family. */
__global__ static void grouped_hidden_g4_dual(float *gate,float *up,const float *x,
                                              const GroupDesc *desc,int I,int D,
                                              float swiglu_limit){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    int ggs=d.ggs>0?d.ggs:D, ugs=d.ugs>0?d.ugs:D;
    const float *gsc=d.gs+(size_t)o*(size_t)((D+ggs-1)/ggs);
    const float *usc=d.us+(size_t)o*(size_t)((D+ugs-1)/ugs);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;float gv=gsc[i/ggs],uv=usc[i/ugs];
        ga+=xs[i]*g0*gv;ua+=xs[i]*u0*uv;
        if(i+1<D){ga+=xs[i+1]*g1*gv;ua+=xs[i+1]*u1*uv;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* same epilogue fusion as the w4 dual above (per-group scales already
     * applied inside the accumulation, so silu runs on the raw sums) */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0],u=upv[0];
        if(swiglu_limit>0.0f){
            g=fminf(g,swiglu_limit);
            u=fminf(fmaxf(u,-swiglu_limit),swiglu_limit);
        }
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}
__global__ static void grouped_down_g4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    int dgs=d.dgs>0?d.dgs:I;
    const float *dsc=d.ds+(size_t)o*(size_t)((I+dgs-1)/dgs);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;float sv=dsc[i/dgs];
        sum+=xs[i]*a*sv;if(i+1<I)sum+=xs[i+1]*z*sv;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

/* fmt=8 fp8-e4m3 variants: same structure as the g4 kernels, but one byte per
 * weight (decoded through c_e4m3) and the scale is per 128x128 BLOCK of the
 * member's [O,I] matrix — sc[(o/128)*ceil(I/128) + i/128]. The block edge is a
 * property of the format, so the geometry derives from the dims alone; these
 * kernels require every member to be fmt=8 (no ride-along: a per-row member's
 * scales are [O], which this indexing would read out of bounds). */
__global__ static void grouped_hidden_f8_dual(float *gate,float *up,const float *x,
                                              const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*D;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*D;
    int nblk=(D+127)>>7;
    const float *gsc=d.gs+(size_t)(o>>7)*nblk;
    const float *usc=d.us+(size_t)(o>>7)*nblk;
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int i=threadIdx.x;i<D;i+=blockDim.x){float xv=xs[i];int b=i>>7;
        ga+=xv*c_e4m3[gr[i]]*gsc[b];ua+=xv*c_e4m3[ur[i]]*usc[b];}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* same fused epilogue as the w4/g4 duals: scales applied in the
     * accumulation, silu(gate)*up lands in gate[], up[] is never written */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}
__global__ static void grouped_down_f8(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*I;
    int nblk=(I+127)>>7;
    const float *dsc=d.ds+(size_t)(o>>7)*nblk;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x)sum+=xs[i]*c_e4m3[row[i]]*dsc[i>>7];
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

/* fmt=8 warp rework (COLI_CUDA_F8_WARP, default on): one WARP owns one
 * (expert,output-row) pair and walks the row block-major through f8w_block —
 * weights stream once per tile of up to 4 activation rows instead of once per
 * (o,s) 256-thread block, and the accumulation is the CPU reference's (f32
 * block partial, scale once per block, double across blocks) instead of the
 * old flat per-element-scale f32 sum. Same descriptor surface as the old
 * kernels; grid (ceil(O/8), count): 8 warps per 256-thread block, rows looped
 * in-kernel. The old kernels stay compiled and selectable (=0) as the field
 * escape hatch. */
template<int HW>
__global__ static void grouped_hidden_f8w_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    __shared__ float slut[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)slut[i]=c_e4m3[i];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int o=blockIdx.x*(blockDim.x>>5)+warp,c=blockIdx.y;GroupDesc d=desc[c];
    if(o>=I)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*D;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*D;
    int nblk=(D+127)>>7;
    const float *gsc=d.gs+(size_t)(o>>7)*nblk;
    const float *usc=d.us+(size_t)(o>>7)*nblk;
    int vec=!(D&3)&&!(((size_t)gr|(size_t)ur)&3)&&!((size_t)x&15);
    for(int s0=0;s0<d.rows;s0+=4){
        int ns=d.rows-s0<4?d.rows-s0:4;
        const float *xs=x+(size_t)(d.offset+s0)*D;
        double ga[4]={0,0,0,0},ua[4]={0,0,0,0};
        for(int bi=0;bi<nblk;bi++){
            int base=bi<<7,len=D-base<128?D-base:128;
            float gp[4],up_[4];
            f8w_block<HW,1>(slut,gr,ur,xs,D,ns,base,len,vec,gp,up_);
            if(!lane){float sg=gsc[bi],su=usc[bi];
                for(int s=0;s<ns;s++){ga[s]+=(double)gp[s]*sg;ua[s]+=(double)up_[s]*su;}}
        }
        /* same fused epilogue as the old dual: silu(gate)*up lands in gate[],
         * up[] is never written */
        if(!lane)for(int s=0;s<ns;s++){
            float g=(float)ga[s],u=(float)ua[s];
            gate[(size_t)(d.offset+s0+s)*I+o]=(g/(1.f+expf(-g)))*u;
        }
    }
    (void)up;
}
template<int HW>
__global__ static void grouped_down_f8w(float *y,const float *x,const GroupDesc *desc,int D,int I){
    __shared__ float slut[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)slut[i]=c_e4m3[i];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int o=blockIdx.x*(blockDim.x>>5)+warp,c=blockIdx.y;GroupDesc d=desc[c];
    if(o>=D)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*I;
    int nblk=(I+127)>>7;
    const float *dsc=d.ds+(size_t)(o>>7)*nblk;
    int vec=!(I&3)&&!((size_t)row&3)&&!((size_t)x&15);
    for(int s0=0;s0<d.rows;s0+=4){
        int ns=d.rows-s0<4?d.rows-s0:4;
        const float *xs=x+(size_t)(d.offset+s0)*I;
        double a[4]={0,0,0,0};
        for(int bi=0;bi<nblk;bi++){
            int base=bi<<7,len=I-base<128?I-base:128;
            float p[4];
            f8w_block<HW,0>(slut,row,nullptr,xs,I,ns,base,len,vec,p,nullptr);
            if(!lane){float sd=dsc[bi];
                for(int s=0;s<ns;s++)a[s]+=(double)p[s]*sd;}
        }
        if(!lane)for(int s=0;s<ns;s++)
            y[(size_t)(d.offset+s0+s)*D+o]=(float)a[s];
    }
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale,
                                                int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

__global__ static void attention_absorb_batch_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale,
        int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K;
        const float *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
#ifdef COLI_ABSORB_RACE_STRESS
    if(!tid)red[blockDim.x-1]=(float)((clock64()>>10)&1ull);__syncthreads();
    int delay_first=(int)red[blockDim.x-1];
    if((delay_first&&tid<warpSize)||(!delay_first&&tid>=warpSize)){
        unsigned long long until=clock64()+100000ull;while(clock64()<until){}
    }
    float mx=((volatile float *)red)[0];
#else
    float mx=red[0];
#endif
    local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    __syncthreads(); /* all waves must read the maximum before red[] is reused */
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

/* Independent device-resident KV sequence per row. lengths selects the valid
 * prefix; latent/rope point at paged caches updated by the host wrapper. */
__global__ static void attention_absorb_ragged_kernel(float *ctx,const float *q,
        const float *const *latent,const float *const *rope,const int *lengths,
        const void *weights,const float *wscale,int fmt,int S,int H,int Q,int R,
        int V,int K,int T,int page_stride,float scale,int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=lengths[s],rbase=h*(Q+V);
    if(s>=S||nt<1||nt>T)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;int pg=t/COLI_KV_PAGE_TOKENS,pt=t%COLI_KV_PAGE_TOKENS;
        const float *lt=latent[(size_t)s*page_stride+pg]+(size_t)pt*K;
        const float *rt=rope[(size_t)s*page_stride+pg]+(size_t)pt*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
#ifdef COLI_ABSORB_RACE_STRESS
    if(!tid)red[blockDim.x-1]=(float)((clock64()>>10)&1ull);__syncthreads();
    int delay_first=(int)red[blockDim.x-1];
    if((delay_first&&tid<warpSize)||(!delay_first&&tid>=warpSize)){
        unsigned long long until=clock64()+100000ull;while(clock64()<until){}
    }
    float mx=((volatile float *)red)[0];
#else
    float mx=red[0];
#endif
    local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    __syncthreads(); /* all waves must read the maximum before red[] is reused */
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++){
        const float *lt=latent[(size_t)s*page_stride+t/COLI_KV_PAGE_TOKENS]+
                        (size_t)(t%COLI_KV_PAGE_TOKENS)*K;
        a+=scores[t]*lt[k];}cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*
            absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

__global__ static void ragged_kv_append(float *const *latent,float *const *rope,
        const float *packed,const int *old_len,const int *add,const int *offset,
        int K,int R,int page_stride){
    int s=blockIdx.x,n=add[s],base=offset[s];
    for(int t=0;t<n;t++){
        int pos=old_len[s]+t,pg=pos/COLI_KV_PAGE_TOKENS,pt=pos%COLI_KV_PAGE_TOKENS;
        float *lp=latent[(size_t)s*page_stride+pg]+(size_t)pt*K;
        float *rp=rope[(size_t)s*page_stride+pg]+(size_t)pt*R;
        for(int k=threadIdx.x;k<K;k+=blockDim.x)lp[k]=packed[base+(size_t)t*K+k];
        for(int r=threadIdx.x;r<R;r+=blockDim.x)rp[r]=packed[base+(size_t)n*K+(size_t)t*R+r];
    }
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

#ifdef COLI_ANS
static void *ans_arena_alloc(DeviceContext *ctx,size_t bytes){
    bytes=(bytes+255)&~size_t(255);
    if(!ctx->ans_chunks)ctx->ans_chunks=new std::vector<AnsArenaChunk>;
    if(ctx->ans_chunks->empty()||ctx->ans_chunks->back().cap-ctx->ans_chunks->back().used<bytes){
        size_t cap=256ull<<20;if(cap<bytes)cap=bytes;
        uint8_t *p=nullptr;if(!cuda_ok(cudaMalloc(&p,cap),"ANS arena chunk"))return nullptr;
        ctx->ans_chunks->push_back({p,0,cap});
    }
    AnsArenaChunk &c=ctx->ans_chunks->back();void *p=c.p+c.used;c.used+=bytes;return p;
}
static int ans_host_reserve(DeviceContext *ctx,size_t bytes){
    if(ctx->ans_copy_pending){
        if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"ANS sidecar upload synchronize"))return 0;
        ctx->ans_copy_pending=0;
    }
    if(ctx->ans_host_cap>=bytes)return 1;
    if(ctx->ans_host)cudaFreeHost(ctx->ans_host);
    ctx->ans_host=nullptr;ctx->ans_host_cap=0;
    if(!cuda_ok(cudaMallocHost(&ctx->ans_host,bytes),"ANS pinned staging allocation"))return 0;
    ctx->ans_host_cap=bytes;return 1;
}
static int prepare_group_weights(DeviceContext *ctx,
        ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,
        ColiCudaTensor *const *downs,int count,GroupDesc *host){
    int n=0; size_t total=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *q[3]={gates[c],ups[c],downs[c]};
        for(int k=0;k<3;k++) if(q[k]->compressed){n++;total+=q[k]->weight_bytes;}
    }
    if(!n) return 1;
    if(!g_ans_profile_printed&&std::getenv("COLI_ANS_PROFILE")){
        g_ans_profile_printed=1;
        std::fprintf(stderr,
            "[ANS] load profile: %llu records | header %.2fs | read %.2fs | "
            "staging/alloc %.2fs | enqueue %.2fs\n",
            (unsigned long long)g_ans_load_records,g_ans_header_s,g_ans_read_s,
            g_ans_stage_s,g_ans_enqueue_s);
    }
    if(!ctx->ans_scratch||!reserve_bytes(&ctx->ans_raw,&ctx->ans_raw_cap,total)) return 0;
    std::vector<const void*> in; in.reserve(n);
    std::vector<void*> out; out.reserve(n);
    std::vector<uint32_t> cap; cap.reserve(n);
    size_t off=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *q[3]={gates[c],ups[c],downs[c]};
        const void **dst[3]={&host[c].g,&host[c].u,&host[c].d};
        for(int k=0;k<3;k++) if(q[k]->compressed){
            void *raw=(uint8_t*)ctx->ans_raw+off;
            in.push_back(q[k]->weights);out.push_back(raw);cap.push_back((uint32_t)q[k]->weight_bytes);
            *dst[k]=raw;off+=q[k]->weight_bytes;
        }
    }
    dietgpu::ANSCodecConfig config(11,false);
    dietgpu::ansDecodeBatchPointer(*ctx->ans_scratch,config,(uint32_t)n,in.data(),out.data(),
                                   cap.data(),nullptr,nullptr,ctx->stream);
    return cuda_ok(cudaGetLastError(),"ANS expert decode launch");
}
#else
static int prepare_group_weights(DeviceContext *,ColiCudaTensor *const *,
        ColiCudaTensor *const *,ColiCudaTensor *const *,int,GroupDesc *){return 1;}
#endif

/* Publish quant.h's E8 codebook to every configured device. __constant__ memory
 * is per-device, so this walks the contexts; the engine calls it once after init
 * rather than the backend carrying a second copy of the table that could drift
 * from the CPU decoder's (#452). Safe to call before any fmt=6 upload only. */
extern "C" int coli_cuda_e8_set_grid(const void *grid) {
    if (!grid || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(cudaMemcpyToSymbol(c_e8_grid, grid, sizeof(c_e8_grid)), "E8 codebook upload"))
            return 0;
    }
    return 1;
}

/* Publish quant.h's E4M3_LUT the same way — one source of truth for the fmt=8
 * decode on CPU and GPU. Until this succeeds, fmt=8 uploads are refused. */
extern "C" int coli_cuda_fp8_set_lut(const float *lut) {
    if (!lut || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(cudaMemcpyToSymbol(c_e4m3, lut, sizeof(c_e4m3)), "e4m3 LUT upload"))
            return 0;
    }
    g_fp8_lut_ready = 1;
    return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    /* #509: the ROCm runtime (comgr, MIOpen, roctracer) reads $TEMP as a temp-dir
     * path. A stray numeric TEMP (the engine's legacy sampling alias) makes comgr's
     * lazy init fail inside the first stream create -- SIGSEGV in the error-unwind
     * on gfx1100, clean hipErrorOutOfMemory on gfx1030. The engine has already
     * parsed g_temp by the time we get here, so a TEMP that is not a real directory
     * is safe to drop before the first ROCm call; a genuine temp-dir is preserved. */
    {
        const char *t = std::getenv("TEMP");
        struct stat st;
        /* Same test on both hosts; only the CRT spelling differs. The MSVC CRT
         * (Windows hipcc's host pass) has no S_ISDIR and no unsetenv — it spells
         * the directory bit _S_IFDIR/_S_IFMT and clears a variable by assigning
         * an empty value. This is an OS/CRT difference, NOT a vendor one, so it
         * stays a _WIN32 branch and adds no CUDA-vs-HIP conditional. */
#ifdef _WIN32
        if (t && *t && (stat(t, &st) != 0 ||
                        (st.st_mode & _S_IFMT) != _S_IFDIR)) _putenv_s("TEMP", "");
#else
        if (t && *t && (stat(t, &st) != 0 || !S_ISDIR(st.st_mode))) unsetenv("TEMP");
#endif
    }
#endif
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        ctx->compute_major=prop.major;ctx->compute_minor=prop.minor;
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
#ifdef COLI_ANS
        if(std::getenv("CUDA_RAW_EXPERTS")){
            ctx->ans_scratch = new dietgpu::StackDeviceMemory(device, 1ull << 30);
            ctx->ans_chunks = new std::vector<AnsArenaChunk>;
        }
#endif
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" int coli_cuda_available_device_count(void) {
    int available = 0;
    if (cudaGetDeviceCount(&available) != cudaSuccess) return 0;
    return available;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        for(int b=0;b<27;b++) if(ctx->pipe_buf[b]) cudaFree(ctx->pipe_buf[b]);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->host_kv) cudaFreeHost(ctx->host_kv);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
#ifdef COLI_ANS
        if(ctx->ans_copy_pending)cudaStreamSynchronize(ctx->stream);
        if(ctx->ans_host)cudaFreeHost(ctx->ans_host);
        if (ctx->ans_raw) cudaFree(ctx->ans_raw);
        if(ctx->ans_chunks){for(auto &c:*ctx->ans_chunks)cudaFree(c.p);delete ctx->ans_chunks;}
        delete ctx->ans_scratch;
        ctx->ans_scratch=nullptr;ctx->ans_chunks=nullptr;ctx->ans_raw=nullptr;ctx->ans_raw_cap=0;
        ctx->ans_host=nullptr;ctx->ans_host_cap=0;ctx->ans_copy_pending=0;
#endif
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->host_x=ctx->host_y=ctx->host_kv=nullptr;ctx->stream=nullptr;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=ctx->host_kv_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
    }
    g_nctx = 0;
#ifdef COLI_ANS
    if(g_ans_sidecar){std::fclose(g_ans_sidecar);g_ans_sidecar=nullptr;}
#if defined(__linux__)
    if(g_ans_direct_fd>=0){close(g_ans_direct_fd);g_ans_direct_fd=-1;g_ans_direct_off=0;}
#endif
#endif
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

/* #653: 1 when the device shares physical memory with the host (Grace-Blackwell /
 * GB10, Jetson, integrated GPUs). On these the expert tier and the RAM cache draw
 * from the same pool, so the RAM budget must account for the tier; on a discrete GPU
 * VRAM is a separate pool and this returns 0. */
extern "C" int coli_cuda_device_integrated(int device) {
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) return 0;
    return prop.integrated ? 1 : 0;
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" void coli_cuda_group_stats_device(
    int device, uint64_t *calls, uint64_t *experts, uint64_t *rows,
    double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    std::lock_guard<std::mutex> lock(g_group_stats_mu);
    int index=-1;
    for(int i=0;i<g_nctx;i++) if(g_ctx[i].device==device){ index=i; break; }
    if(calls) *calls=index<0?0:g_device_group_calls[index];
    if(experts) *experts=index<0?0:g_device_group_experts[index];
    if(rows) *rows=index<0?0:g_device_group_rows[index];
    if(h2d_ms) *h2d_ms=index<0?0:g_device_group_h2d_ms[index];
    if(kernel_ms) *kernel_ms=index<0?0:g_device_group_kernel_ms[index];
    if(d2h_ms) *d2h_ms=index<0?0:g_device_group_d2h_ms[index];
}

/* group size for the NEXT upload on this thread (fmt=4): routed through a
 * thread_local so the widely-wired upload signature (and the Windows DLL ABI)
 * stays untouched. pin_load uploads in parallel, hence thread_local. */
static thread_local int g_upload_gs = 0;
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs);
extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    if (!tensor) return 0;
    if (*tensor) {
        /* Cached device copy: usable even when the caller's host pointers are
         * gone. CUDA_RELEASE_HOST slots null their host pointers after upload,
         * and with the old order (!weights checked first) every later matmul
         * on such a slot failed here — the GPU tier silently never computed
         * for host-released slab experts. */
        ColiCudaTensor *t = *tensor;
        int want_gs = (fmt==4 && g_upload_gs>0) ? g_upload_gs : 0;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device && t->gs == want_gs;
    }
    DeviceContext *ctx = find_ctx(device);
    if (!weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    /* fmt=6 keeps its scales inside each 98-byte block, so it is the one
     * quantized format that legitimately arrives with scales == NULL. */
    if (!rb || (fmt && fmt != 6 && !scales)) return 0;
    if (fmt == 8 && !g_fp8_lut_ready) return 0;   /* kernels would read a zero LUT */
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    t->gs = (fmt==4 && g_upload_gs>0) ? g_upload_gs : 0;
    t->ng = t->gs ? (I + t->gs - 1) / t->gs : 1;
    t->scale_count = t->gs ? (size_t)O * (size_t)t->ng : (size_t)O;
    if (fmt == 8) {   /* per-128x128-block scales: [ceil(O/128), ceil(I/128)] */
        t->ng = (I + 127) / 128;
        t->scale_count = (size_t)((O + 127) / 128) * (size_t)t->ng;
    }
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    /* Ownership is a fact of the allocation, not the copy: set it BEFORE the
     * memcpy, or a failed H2D upload frees the tensor while weights_owned is
     * still 0 and free()'s ownership gate leaks the device buffer. */
    t->weights_owned=1;
    if (!cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2||fmt==4){ /* same nibble layout: offset-binary -> signed in place */
        offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt && fmt != 6) {
        if (!cuda_ok(cudaMalloc(&t->scales, t->scale_count * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, t->scale_count * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    if (fmt == 6) t->scale_count = 0;      /* in-block scales: nothing separate to track */
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + ((fmt && fmt != 6) ? t->scale_count * sizeof(float) : 0);
    *tensor = t;
    return 1;
}
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs){
    g_upload_gs = gs>0 ? gs : 0;
    int r = coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device);
    g_upload_gs = 0;
    return r;
}

#ifdef COLI_ANS
struct AnsSidecarHeader {
    uint32_t magic,raw_bytes,archive_bytes,fmt,I,O;
};
static FILE *ans_sidecar(void){
    if(g_ans_sidecar) return g_ans_sidecar;
    const char *path=std::getenv("COLI_ANS_SIDECAR");
    if(!path||!*path) return nullptr;
    g_ans_sidecar_pack=std::getenv("COLI_ANS_PACK")&&std::atoi(std::getenv("COLI_ANS_PACK"));
    g_ans_sidecar=std::fopen(path,g_ans_sidecar_pack?"wb":"rb");
#if defined(__linux__)
    if(g_ans_sidecar&&!g_ans_sidecar_pack&&std::getenv("COLI_ANS_DIRECT")&&
       std::atoi(std::getenv("COLI_ANS_DIRECT"))){
        g_ans_direct_fd=open(path,O_RDONLY|O_DIRECT);
        if(g_ans_direct_fd<0)std::fprintf(stderr,"[ANS] O_DIRECT unavailable; using buffered sidecar\n");
    }
    if(g_ans_sidecar&&!g_ans_sidecar_pack&&g_ans_direct_fd<0)
        posix_fadvise(fileno(g_ans_sidecar),0,0,POSIX_FADV_SEQUENTIAL);
#endif
    return g_ans_sidecar;
}
extern "C" int coli_cuda_tensor_upload_compressed(ColiCudaTensor **tensor,
        const void *weights,const float *scales,int fmt,int I,int O,int device){
    if(fmt!=2 || !tensor || *tensor) return 0;  /* prototype: per-row int4 experts only */
    FILE *sidecar=ans_sidecar();
    if(sidecar&&!g_ans_sidecar_pack){
        AnsSidecarHeader h{};
        size_t expected=((size_t)I+1)/2*(size_t)O;
        double t0=ans_now_s();
#if defined(__linux__)
        size_t direct_delta=0,direct_bytes=0,direct_data=0;
        if(g_ans_direct_fd>=0){
            alignas(4096) uint8_t first[8192];
            off_t start=g_ans_direct_off&~off_t(4095);
            direct_delta=(size_t)(g_ans_direct_off-start);
            ssize_t got=pread(g_ans_direct_fd,first,sizeof(first),start);
            if(got<(ssize_t)(direct_delta+sizeof(h))){
                std::fprintf(stderr,"[ANS] direct header read failed at %lld: got %lld errno %d\n",
                    (long long)g_ans_direct_off,(long long)got,errno);
                return 0;
            }
            std::memcpy(&h,first+direct_delta,sizeof(h));
        }else
#endif
        if(std::fread(&h,sizeof(h),1,sidecar)!=1)return 0;
        if(h.magic!=0x31534e41u||
           h.fmt!=(uint32_t)fmt||h.I!=(uint32_t)I||h.O!=(uint32_t)O||
           expected>UINT32_MAX||h.raw_bytes!=(uint32_t)expected||
           !h.archive_bytes||h.archive_bytes>dietgpu::getMaxCompressedSize(h.raw_bytes)){
            std::fprintf(stderr,"[ANS] invalid or mismatched sidecar record\n");
            return 0;
        }
        g_ans_header_s+=ans_now_s()-t0;
        DeviceContext *ctx=find_ctx(device); if(!ctx||!select_ctx(ctx)) return 0;
        ColiCudaTensor *t=(ColiCudaTensor*)std::calloc(1,sizeof(*t)); if(!t)return 0;
        t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->weight_bytes=h.raw_bytes;
        t->scale_count=(size_t)O;t->archive_bytes=h.archive_bytes;t->compressed=1;
        t0=ans_now_s();
        t->weights=ans_arena_alloc(ctx,h.archive_bytes);
        size_t scale_bytes=(size_t)O*sizeof(float),scale_off;
#if defined(__linux__)
        if(g_ans_direct_fd>=0){
            size_t record_bytes=sizeof(h)+(size_t)h.archive_bytes;
            direct_data=direct_delta+sizeof(h);
            direct_bytes=(direct_delta+record_bytes+4095)&~size_t(4095);
            scale_off=(direct_bytes+255)&~size_t(255);
        }else
#endif
            scale_off=(h.archive_bytes+255)&~size_t(255);
        if(!t->weights||!ans_host_reserve(ctx,scale_off+scale_bytes)||
           !cuda_ok(cudaMalloc(&t->scales,scale_bytes),"ANS sidecar scales")){
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_stage_s+=ans_now_s()-t0;
        t0=ans_now_s();
#if defined(__linux__)
        if(g_ans_direct_fd>=0){
            off_t start=g_ans_direct_off&~off_t(4095);
            ssize_t got=pread(g_ans_direct_fd,ctx->ans_host,direct_bytes,start);
            if(got<(ssize_t)(direct_data+h.archive_bytes)){
                std::fprintf(stderr,
                    "[ANS] direct record read failed at %lld: need %zu got %lld errno %d\n",
                    (long long)g_ans_direct_off,direct_data+h.archive_bytes,
                    (long long)got,errno);
                coli_cuda_tensor_free(t);return 0;
            }
            g_ans_direct_off+=(off_t)sizeof(h)+(off_t)h.archive_bytes;
        }else
#endif
        if(std::fread(ctx->ans_host,h.archive_bytes,1,sidecar)!=1){
            std::fprintf(stderr,"[ANS] truncated sidecar record\n");
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_read_s+=ans_now_s()-t0;
        std::memcpy((uint8_t*)ctx->ans_host+scale_off,scales,scale_bytes);
        t0=ans_now_s();
        void *archive_src=
#if defined(__linux__)
            g_ans_direct_fd>=0?(uint8_t*)ctx->ans_host+direct_data:
#endif
            ctx->ans_host;
        if(!cuda_ok(cudaMemcpyAsync(t->weights,archive_src,h.archive_bytes,
                                   cudaMemcpyHostToDevice,ctx->stream),"ANS sidecar upload")||
           !cuda_ok(cudaMemcpyAsync(t->scales,(uint8_t*)ctx->ans_host+scale_off,scale_bytes,
                                   cudaMemcpyHostToDevice,ctx->stream),"ANS sidecar scale upload")){
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_enqueue_s+=ans_now_s()-t0;g_ans_load_records++;
        ctx->ans_copy_pending=1;
        t->tracked=1;ctx->tensor_count++;ctx->tensor_bytes+=h.archive_bytes+(size_t)O*sizeof(float);
        *tensor=t;return 1;
    }
    if(!sidecar||!g_ans_sidecar_pack) return 0;
    if(!coli_cuda_tensor_upload(tensor,weights,scales,fmt,I,O,device)) return 0;
    ColiCudaTensor *t=*tensor;
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->ans_scratch||!select_ctx(ctx)){ coli_cuda_tensor_free(t);*tensor=nullptr;return 0; }
    uint32_t raw=(uint32_t)t->weight_bytes;
    uint32_t bound=dietgpu::getMaxCompressedSize(raw), *dsize=nullptr;
    void *tmp=nullptr;
    if(!cuda_ok(cudaMalloc(&tmp,bound),"ANS archive allocation")||
       !cuda_ok(cudaMalloc(&dsize,sizeof(*dsize)),"ANS size allocation")){
        if(tmp)cudaFree(tmp);if(dsize)cudaFree(dsize);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;
    }
    dietgpu::ANSCodecConfig config(11,false);
    /* tensor_upload converted offset-binary nibbles on the legacy stream.
     * ctx->stream is explicitly non-blocking, so it does not inherit the
     * legacy-stream dependency. Finish that one-time conversion before the
     * encoder reads the bytes. */
    if(!cuda_ok(cudaStreamSynchronize(0),"ANS source conversion synchronize")){
        cudaFree(tmp);cudaFree(dsize);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;
    }
    dietgpu::ansEncodeBatchStride(*ctx->ans_scratch,config,1,t->weights,raw,raw,nullptr,
                                  tmp,bound,dsize,ctx->stream);
    uint32_t used=0;
    int ok=cuda_ok(cudaMemcpyAsync(&used,dsize,sizeof(used),cudaMemcpyDeviceToHost,ctx->stream),
                   "ANS size download")&&
           cuda_ok(cudaStreamSynchronize(ctx->stream),"ANS encode synchronize")&&used>0&&used<raw;
    cudaFree(dsize);
    if(!ok){cudaFree(tmp);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;}
    if(g_ans_sidecar_pack){
        std::vector<uint8_t> archive(used);
        AnsSidecarHeader h{0x31534e41u,raw,used,(uint32_t)fmt,(uint32_t)I,(uint32_t)O};
        ok=cuda_ok(cudaMemcpy(archive.data(),tmp,used,cudaMemcpyDeviceToHost),"ANS sidecar download")&&
           std::fwrite(&h,sizeof(h),1,sidecar)==1&&
           std::fwrite(archive.data(),archive.size(),1,sidecar)==1;
        cudaFree(tmp);coli_cuda_tensor_free(t);*tensor=nullptr;
        if(!ok)return 0;
        t=(ColiCudaTensor*)std::calloc(1,sizeof(*t));if(!t)return 0;
        t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->weight_bytes=raw;
        t->archive_bytes=used;t->compressed=1;*tensor=t;
        return 1;
    }
    return 0;
}
#endif

extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                          const void *weights,
                                          const float *scales) {
    if (!tensor || !weights || (tensor->fmt && tensor->fmt != 6 && !scales)) return 0;
#ifdef COLI_ANS
    if(tensor->compressed) return 0;
#endif
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaMemcpy(tensor->weights,weights,tensor->weight_bytes,
                            cudaMemcpyHostToDevice),"tensor refresh")) return 0;
    if(tensor->fmt==2||tensor->fmt==4){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight refresh")) return 0;
    }
    /* fmt=6 has no scale buffer at all (scales live in-block, scale_count 0), and
     * the fallback below would otherwise copy O floats out of a NULL host pointer. */
    return !tensor->fmt || tensor->fmt==6 || cuda_ok(cudaMemcpy(tensor->scales,scales,
        (tensor->scale_count?tensor->scale_count:(size_t)tensor->O)*sizeof(float),
        cudaMemcpyHostToDevice),"scale refresh");
}

/* Test hook: COLI_GPU_FAIL_AFTER=N makes every GPU COMPUTE entry point report
 * failure after N successful calls (N=0: every call fails), exercising the
 * engine's CPU fallbacks and host-rematerialization end-to-end without real
 * hardware faults. Uploads/queries are not gated. Unset: no effect. */
static long g_gpu_calls;
static int fault_injected(void) {
    const char *fa = std::getenv("COLI_GPU_FAIL_AFTER");
    return fa && g_gpu_calls++ >= std::atol(fa);
}

/* COLI_CUDA_F8_WARP mode, re-read per dispatch like its siblings. Strict
 * parse: a non-numeric value selects the DEFAULT, not atoi's silent 0.
 * Default 1 (warp kernels) on CUDA; 0 (original kernels) on HIP — the warp
 * kernels' wave64 width-32 shuffle sub-grouping has never been validated on
 * AMD silicon, and a scoring instrument must not default onto an untested
 * reduction. Opt in explicitly with =1/=2 once hip-test certifies it. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define COLI_F8_DEFAULT 0
#else
#define COLI_F8_DEFAULT 1
#endif
static int f8_warp_mode(void) {
    const char *e = std::getenv("COLI_CUDA_F8_WARP");
    if (!e || !*e) return COLI_F8_DEFAULT;
    char *end; long v = std::strtol(e, &end, 10);
    return *end ? COLI_F8_DEFAULT : (int)v;
}

/* One launch site for the dense matvec so fmt=8 honors the same toggle as
 * f8_group_launch: mode 0 runs the original quant_matmul branch (fully
 * original behavior), anything else the warp/shared-LUT rework. */
static void quant_matmul_launch(float *y, const float *x, const void *w,
        const float *sc, int fmt, int S, int I, int O, size_t rb, int gs, int ng) {
    dim3 grid((unsigned)O, (unsigned)S);
    if (fmt == 8 && f8_warp_mode())
        quant_matmul_f8w<<<grid, 256>>>(y, x, w, sc, S, I, O);
    else
        quant_matmul<<<grid, 256>>>(y, x, w, sc, fmt, S, I, O, rb, gs, ng);
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device, int gs) {
    if (fault_injected()) return 0;
    /* fmt=4 carries [O, ceil(I/gs)] scales: without the group size the plain
     * upload truncates the buffer to O floats and quant_matmul divides by
     * gs==0. Callers must come through the gs>0 path (upload_g) or stay on
     * the CPU (#298, #334). */
    if (fmt == 4 && gs <= 0) return 0;
    if (S < 1) return 0;
    if (gs > 0) { if (!coli_cuda_tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs)) return 0; }
    else        { if (!coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0; }
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    quant_matmul_launch(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb, t->gs, t->ng);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

/* MXFP4 matmul, stateless. Separate from coli_cuda_matmul on purpose: that one
 * takes scales as const float* and caches an uploaded tensor, while MXFP4
 * scales are ue8m0 BYTES -- passing them through the float* parameter would
 * compile and silently reinterpret the buffer. Kimi K3's routed experts stream
 * (a fill-once tier at decode), so there is nothing to cache here anyway; the
 * weights go up with the call.
 *
 * Returns 0 and leaves y untouched on any failure, which is the contract the
 * engine's GPU paths already use to fall back to CPU. */
extern "C" int coli_cuda_matmul_mxfp4(float *y, const float *x,
                                      const uint8_t *q4, const uint8_t *e8s,
                                      int S, int I, int O) {
    if (fault_injected()) return 0;
    if (S < 1 || I < 1 || O < 1 || !y || !x || !q4 || !e8s) return 0;
    DeviceContext *ctx = find_ctx(0);
    if (!select_ctx(ctx)) return 0;

    size_t rb = (size_t)(I + 1) / 2, ng = (size_t)(I + 31) / 32;
    size_t wb = (size_t)O * rb, sb = (size_t)O * ng;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);

    uint8_t *dw = nullptr, *ds = nullptr;
    if (!cuda_ok(cudaMalloc(&dw, wb), "mxfp4 weight alloc")) return 0;
    if (!cuda_ok(cudaMalloc(&ds, sb), "mxfp4 scale alloc")) { cudaFree(dw); return 0; }

    int ok = reserve(&ctx->x, &ctx->x_cap, xb) && reserve(&ctx->y, &ctx->y_cap, yb) &&
             cuda_ok(cudaMemcpy(dw, q4, wb, cudaMemcpyHostToDevice), "mxfp4 weight upload") &&
             cuda_ok(cudaMemcpy(ds, e8s, sb, cudaMemcpyHostToDevice), "mxfp4 scale upload") &&
             cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "mxfp4 input upload");
    if (ok) {
        dim3 grid((unsigned)O, (unsigned)S);
        quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, dw, reinterpret_cast<const float *>(ds),
                                    7, S, I, O, rb, 32, (int)ng);
        ok = cuda_ok(cudaGetLastError(), "mxfp4 launch") &&
             cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "mxfp4 output download");
    }
    cudaFree(dw);
    cudaFree(ds);
    return ok;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (fault_injected()) return 0;
    /* same reason as coli_cuda_matmul: fmt=4 without recorded group info would
     * misread the scales (and divide by gs==0 in the kernel). */
    if (gate && ((gate->fmt == 4 && gate->gs <= 0) ||
                 (up && up->fmt == 4 && up->gs <= 0) ||
                 (down && down->fmt == 4 && down->gs <= 0))) return 0;
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    quant_matmul_launch(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D),gate->gs,gate->ng);
    quant_matmul_launch(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D),up->gs,up->ng);
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    /* fmt=6: the down projection stores W@Q, so its input needs Q^T applied. This
     * one is per-expert (the silu product is not shared), unlike the gate/up input
     * rotation, which the caller does once per layer -- same split as moe(). */
    if (down->fmt == 6 && !e8_rot_rows_dev(ctx->gate, S, I, 0)) return 0;
    quant_matmul_launch(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I),down->gs,down->ng);
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if (fault_injected()) return 0;
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||!COLI_GPU_HAS_WMMA||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "shared w4a16 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    w4a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    w4a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"shared w4a16 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "shared w4a16 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"shared w4a16 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

/* Single launch site for the fmt=8 group kernels, shared by the sync and the
 * issue/take dispatches so both stay one-line call sites (rebase-tolerant
 * against #935's e8_group_launch refactor of the adjacent branch).
 * COLI_CUDA_F8_WARP (docs/ENVIRONMENT.md, parsed by f8_warp_mode): 1 = warp
 * kernels (the CUDA default; HIP defaults to 0), 0 = the original per-(o,s)
 * kernels (field escape hatch, dense path included via quant_matmul_launch),
 * 2 = warp kernels decoding through cuda_fp8.h — a real cvt instruction only
 * on sm_89+, the header's bit-manip emulation below that, and plain =1
 * behavior where cuda_fp8.h is absent (HIP); experimental until the
 * 256-value sweep certifies it bit-identical on the target silicon. */
static void f8_group_launch(DeviceContext *ctx,GroupDesc *dev,int I,int D,
                            int max_rows,int count){
    int mode=f8_warp_mode();
    if(mode){
        dim3 hg((unsigned)((I+7)/8),(unsigned)count),og((unsigned)((D+7)/8),(unsigned)count);
#if COLI_F8_HWCVT
        if(mode==2){
            grouped_hidden_f8w_dual<1><<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
            grouped_down_f8w<1><<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
            return;
        }
#endif
        grouped_hidden_f8w_dual<0><<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        grouped_down_f8w<0><<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
        return;
    }
    dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
    grouped_hidden_f8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
    grouped_down_f8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
}

static int expert_group_impl(ColiCudaTensor *const *gates,
                             ColiCudaTensor *const *ups,
                             ColiCudaTensor *const *downs,
                             const int *rows, int count,
                             float *y, const float *x,
                             int pin_small_batch) {
    if (fault_injected()) return 0;
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    int all_s4=1,all_q4=1,any_g4=0,any_e8=0,all_e8=1,any_f8=0,all_f8=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        any_f8|=g->fmt==8||u->fmt==8||d->fmt==8;
        all_f8&=g->fmt==8&&u->fmt==8&&d->fmt==8;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    /* Mixed E8/FP8 groups cannot use a homogeneous grouped kernel. */
    if((any_e8&&!all_e8)||(any_f8&&!all_f8)){
        int off=0;
        for(int c=0;c<count;c++){
            if(!coli_cuda_expert_mlp(gates[c],ups[c],downs[c],
                    y+(size_t)off*D,x+(size_t)off*D,rows[c])) return 0;
            off+=rows[c];
        }
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
        return 1;
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")){
        for(int j=0;j<i;j++) cudaEventDestroy(ev[j]); profile=0; break; }   /* (#B8) don't leak the events already created */
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    /* grouped_s4_wmma's body needs __CUDA_ARCH__>=750: on builds where the
     * WMMA kernels are compiled out (COLI_HIP_NO_WMMA) the launch would
     * succeed with an EMPTY kernel and the output buffer would silently keep
     * stale data. Gate the branch like TC_W4A16 below does. */
    tc=tc&&!pin_small_batch&&COLI_GPU_HAS_WMMA&&all_s4&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=rows[c]>=tc_min;
    if(all_e8){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_e8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        grouped_down_e8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_f8){
        /* fp8-e4m3 groups: silu fused in the dual epilogue, like the w4/g4 duals. */
        f8_group_launch(ctx,dev,I,D,max_rows,count);
    }else if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(!pin_small_batch&&all_s4&&COLI_GPU_HAS_WMMA&&ctx->compute_major>=7&&getenv("COLI_CUDA_TC_W4A16")&&
             atoi(getenv("COLI_CUDA_TC_W4A16"))&&
             [&]{ int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
                  for(int c=0;c<count;c++) if(rows[c]>=tc16_min) return 1;
                  return 0; }()){
        /* At least one expert has enough rows for a Tensor Core tile. Groups
         * where EVERY expert is below the threshold (decode: r=1) fall through
         * to the grouped-W4 path below — 3 launches for the whole group instead
         * of 4 per expert (#431: the launch flood measured at ~981 micro-kernels
         * per token came from decode riding this branch's per-expert fallback). */
        /* W4A16 Tensor Core per gruppo: attivazioni fp16 per tile (lossless al
         * contrario del path W4A4), un lancio per expert dentro lo stream —
         * l'overhead di lancio e' trascurabile rispetto ai GEMM. */
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                dim3 hg16((unsigned)((I+63)/64),(unsigned)((r+15)/16));
                dim3 og16((unsigned)((D+63)/64),(unsigned)((r+15)/16));
                w4a16_gate_up<<<hg16,256,0,ctx->stream>>>(g16,u16,x16,
                    (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                w4a16_matmul<<<og16,128,0,ctx->stream>>>(y16,g16,
                    (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
                    host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
                    host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
                    host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
            }
            off16+=r;
        }
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{   /* non-dual path has no fused epilogue: silu stays a kernel here */
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_q4&&any_g4){
        /* grouped-int4 (fmt=4) present: per-group scales (#334). fmt=2 members
         * ride along as the ng=1 special case. silu fused in the dual epilogue. */
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        const char *limit_env=getenv("COLI_SWIGLU_LIMIT");
        float swiglu_limit=limit_env?(float)atof(limit_env):0.0f;
        grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D,swiglu_limit);
        grouped_down_g4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else{
        /* generic path decodes fmt 0/1/2/3 only — refuse everything else rather
         * than whitelist known offenders: a fmt=4 group that slipped the gates
         * above (odd gs) must NOT be silently decoded as int2 (#334), and any
         * group/block-scaled format that gains CUDA tensors later (fmt=5, fmt=8)
         * carries scale geometry this kernel's epilogue does not apply.
         * STRICTER than weight_at's own admissible set on purpose: fmt=4 belongs
         * on the g4 path above, and returning 0 here keeps the CORRECT CPU
         * fallback, which is the outcome we want — weight_at's device-side
         * __trap backstop is for a launch that got past a gate like this one,
         * not a substitute for having the gate. */
        for(int c=0;c<count;c++)
            if(host[c].gf>3||host[c].uf>3||host[c].df>3) return 0;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          int index=(int)(ctx-g_ctx);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c;
          g_device_group_h2d_ms[index]+=a;
          g_device_group_kernel_ms[index]+=b;
          g_device_group_d2h_ms[index]+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      int index=(int)(ctx-g_ctx);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_device_group_calls[index]++; g_device_group_experts[index]+=(uint64_t)count;
      g_device_group_rows[index]+=(uint64_t)total; }
    return 1;
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    return expert_group_impl(gates,ups,downs,rows,count,y,x,0);
}

extern "C" int coli_cuda_expert_group_pinned(ColiCudaTensor *const *gates,
                                               ColiCudaTensor *const *ups,
                                               ColiCudaTensor *const *downs,
                                               const int *rows, int count,
                                               float *y, const float *x,
                                               int pin_small_batch) {
    return expert_group_impl(gates,ups,downs,rows,count,y,x,pin_small_batch);
}

/* ---- Async expert group (Inc.4): issue/take split of coli_cuda_expert_group ----
 * The measured cost of the sync call at decode is ~0.45 ms/call of HOST-side wait
 * (stream sync + staging), vs ~0.18 ms of actual GPU work — 70% tax, paid ~5x per
 * layer because a token's 8 experts scatter across devices. issue() stages and
 * launches on the device stream and returns immediately; take() syncs and hands
 * back the pinned result rows. One issue may be outstanding per device; moe()
 * takes at each layer end, which also orders the next layer's reuse of the ctx
 * scratch buffers. Small batches only (decode/spec): bigger totals keep the sync
 * path with its TC variants. Numerics are the sync path's small-batch kernels,
 * so greedy output is byte-identical by construction. */
extern "C" int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              const float *x) {
    if (!gates || !ups || !downs || !rows || !x || count < 1 || count > 64) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0,all_s4=1,any_e8=0,all_e8=1,
        all_q4=1,any_g4=0,any_f8=0,all_f8=1;
    GroupDesc host[64];
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_f8|=g->fmt==8||u->fmt==8||d->fmt==8;
        all_f8&=g->fmt==8&&u->fmt==8&&d->fmt==8;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    if(any_e8&&!all_e8) return 0;
    if(any_f8&&!all_f8) return 0;   /* mixed FP8: no homogeneous kernel, sync path has the per-expert loop */
    if(total>8) return 0;                       /* decode-scale only */
    DeviceContext *ctx=find_ctx(device); if(!ctx||ctx->group_pending||!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)) return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue descriptors")||
       !cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue upload")) return 0;
    if(all_e8){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_e8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        grouped_down_e8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_f8){
        /* fp8-e4m3 groups on the async decode path: same launch helper as the
         * sync dispatch, silu fused in the dual epilogue. */
        f8_group_launch(ctx,(GroupDesc*)ctx->group_desc,I,D,max_rows,count);
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual) grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else {
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(
                ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    } else if(all_q4&&any_g4){
        /* grouped int4 (fmt=4) present in the async decode path: per-group
         * scales via the #334 kernels (fmt=2 members ride along as ng=1). The
         * previous fallback ran quant_matmul with gs=0,ng=1, which silently
         * applied one per-row scale to a grouped container -> wrong output. */
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        /* silu is fused in the dual kernel's epilogue (like the sync path):
         * an extra silu_mul here would re-apply it against the never-written
         * ctx->up buffer. */
        const char *limit_env=getenv("COLI_SWIGLU_LIMIT");
        float swiglu_limit=limit_env?(float)atof(limit_env):0.0f;
        grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D,swiglu_limit);
        grouped_down_g4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    } else {
        /* Fallback runs quant_matmul with gs=0,ng=1 — per-row-scale semantics.
         * That is only correct for fmt 0/1/2/3: refuse group/block-scaled
         * members (fmt=4 with odd gs today; fmt=5/8 if they ever gain CUDA
         * tensors) instead of silently mis-scaling them, mirroring the sync
         * path's refusal (#334). fmt=6 cannot reach here (any_e8 gates above). */
        for(int c=0;c<count;c++)
            if(host[c].gf>3||host[c].uf>3||host[c].df>3) return 0;
        for(int c=0;c<count;c++){
        int r=rows[c];
        float *g16=ctx->gate+(size_t)host[c].offset*I,*u16=ctx->up+(size_t)host[c].offset*I;
        float *x16=ctx->x+(size_t)host[c].offset*D,*y16=ctx->y+(size_t)host[c].offset*D;
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
            host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
            host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
        silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
        quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
            host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
    }}
    if(!cuda_ok(cudaGetLastError(),"expert group issue launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                "expert group issue download")) return 0;
    ctx->group_pending=1; ctx->group_pending_bytes=xb;
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      int index=(int)(ctx-g_ctx);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total;
      g_device_group_calls[index]++; g_device_group_experts[index]+=(uint64_t)count;
      g_device_group_rows[index]+=(uint64_t)total; }
    return 1;
}

extern "C" const float *coli_cuda_expert_group_take(int device) {
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->group_pending) return nullptr;
    ctx->group_pending=0;
    if(!select_ctx(ctx)) return nullptr;
    if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group take")) return nullptr;
    return ctx->host_y;
}


/* The absorb kernels decode `w` through weight_at + absorb_scale, which know
 * per-row and fmt=4 group scales only. Refuse anything else (fmt=5/6/8) rather
 * than mis-decode it — the caller keeps its CPU attention path. (`proj`
 * tensors are exempt: they run through quant_matmul, which dispatches every
 * format it uploads.) A dedicated block-scale absorb for fmt=8 is follow-up
 * work, same shape as routing fmt=4 through the grouped kernels was.
 *
 * The admissible set is weight_at's own, taken from the shared predicate rather
 * than restated as `fmt <= 4`: this gate and weight_at's device-side backstop
 * must not be able to drift apart, and the old inequality also admitted
 * NEGATIVE fmt values, which weight_at would then have fallen through on. Same
 * truth table for every fmt a container can actually carry (0..8), so no
 * existing container changes behaviour here. */
static int absorb_fmt_ok(const ColiCudaTensor *w){
    return w && coli_cuda_weight_at_supported(w->fmt);
}

extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,int S,int H,int Q,int R,int V,
        int K,int T,float scale){
    if(!absorb_fmt_ok(w)||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention batch launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_ragged(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!absorb_fmt_ok(w)||!proj||!out||!q||!keys||!latent||!rope||!lengths||S<1||S>512||T<1||T>8192||
       H<1||Q<1||R<1||V<1||K<1||K>512||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);
    if(!select_ctx(dc))return 0;
    int *old=(int*)std::malloc((size_t)S*sizeof(*old));
    int *add=(int*)std::malloc((size_t)S*sizeof(*add));
    int *off=(int*)std::malloc((size_t)S*sizeof(*off));int packed_n=0;
    if(!old||!add||!off){std::free(old);std::free(add);std::free(off);return 0;}
    int page_stride=0;
    for(int s=0;s<S;s++){
        if(!keys[s]||lengths[s]<1||lengths[s]>T){std::free(old);std::free(add);std::free(off);return 0;}
        RaggedKVEntry *e=nullptr;
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){e=&w->ragged[i];break;}
        if(!e){
            if(w->ragged_count>=512){std::free(old);std::free(add);std::free(off);return 0;}
            e=&w->ragged[w->ragged_count++];std::memset(e,0,sizeof(*e));e->key=keys[s];
        }
        if(e->K!=K||e->R!=R||e->host_l!=latent[s]||e->host_r!=rope[s]||lengths[s]<e->length){
            ragged_kv_clear(e);
            e->K=K;e->R=R;e->host_l=latent[s];e->host_r=rope[s];
        }
        int need=(lengths[s]+COLI_KV_PAGE_TOKENS-1)/COLI_KV_PAGE_TOKENS;
        if(need>e->page_count){
            float **nl=(float**)std::calloc((size_t)need,sizeof(*nl));
            float **nr=(float**)std::calloc((size_t)need,sizeof(*nr));
            if(!nl||!nr){std::free(nl);std::free(nr);std::free(old);std::free(add);std::free(off);return 0;}
            for(int i=0;i<e->page_count;i++){nl[i]=e->latent_pages[i];nr[i]=e->rope_pages[i];}
            int made=e->page_count;
            for(;made<need;made++){
                if(!cuda_ok(cudaMalloc(&nl[made],(size_t)COLI_KV_PAGE_TOKENS*K*sizeof(float)),"ragged KV latent page")||
                   !cuda_ok(cudaMalloc(&nr[made],(size_t)COLI_KV_PAGE_TOKENS*R*sizeof(float)),"ragged KV rope page"))break;
            }
            if(made<need){
                if(nl[made])cudaFree(nl[made]);if(nr[made])cudaFree(nr[made]);
                for(int i=e->page_count;i<made;i++){cudaFree(nl[i]);cudaFree(nr[i]);}
                std::free(nl);std::free(nr);std::free(old);std::free(add);std::free(off);return 0;
            }
            std::free(e->latent_pages);std::free(e->rope_pages);
            e->latent_pages=nl;e->rope_pages=nr;e->page_count=need;
        }
        if(e->page_count>page_stride)page_stride=e->page_count;
        old[s]=e->length;add[s]=lengths[s]-e->length;
        off[s]=packed_n;packed_n+=add[s]*(K+R);
    }
    size_t table_n=(size_t)S*page_stride;
    float **dl=(float**)std::calloc(table_n,sizeof(*dl));
    float **dr=(float**)std::calloc(table_n,sizeof(*dr));
    if(!dl||!dr){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
    for(int s=0;s<S;s++)for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){
        for(int p=0;p<w->ragged[i].page_count;p++){
            dl[(size_t)s*page_stride+p]=w->ragged[i].latent_pages[p];
            dr[(size_t)s*page_stride+p]=w->ragged[i].rope_pages[p];
        }
        break;
    }
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float);
    size_t cb=(size_t)S*H*V*sizeof(float),ob=(size_t)S*proj->O*sizeof(float);
    size_t pb=(size_t)packed_n*sizeof(float);
    size_t desc=2*table_n*sizeof(float*)+(size_t)S*4*sizeof(int);
    int ok=reserve(&dc->aq,&dc->aq_cap,qb)&&reserve(&dc->ac,&dc->ac_cap,cb)&&
           reserve(&dc->y,&dc->y_cap,ob)&&reserve_bytes(&dc->group_desc,&dc->group_desc_cap,desc)&&
           (!pb||(reserve(&dc->al,&dc->al_cap,pb)&&reserve_pinned(&dc->host_kv,&dc->host_kv_cap,pb)));
    char *db=(char*)dc->group_desc;float **ddl=(float**)db,**ddr=ddl+table_n;
    int *dn=(int*)(ddr+table_n),*dold=dn+S,*dadd=dold+S,*doff=dadd+S;
    if(ok&&pb){
        for(int s=0;s<S;s++)if(add[s]){
            float *p=dc->host_kv+off[s];
            std::memcpy(p,latent[s]+(size_t)old[s]*K,(size_t)add[s]*K*sizeof(float));
            std::memcpy(p+(size_t)add[s]*K,rope[s]+(size_t)old[s]*R,(size_t)add[s]*R*sizeof(float));
        }
        ok=cuda_ok(cudaMemcpyAsync(dc->al,dc->host_kv,pb,cudaMemcpyHostToDevice,dc->stream),"ragged KV append upload");
    }
    if(ok)ok=cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"ragged q upload")&&
             cuda_ok(cudaMemcpyAsync(ddl,dl,table_n*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged latent page table")&&
             cuda_ok(cudaMemcpyAsync(ddr,dr,table_n*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged rope page table")&&
             cuda_ok(cudaMemcpyAsync(dn,lengths,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged lengths upload")&&
             cuda_ok(cudaMemcpyAsync(dold,old,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged old lengths")&&
             cuda_ok(cudaMemcpyAsync(dadd,add,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append lengths")&&
             cuda_ok(cudaMemcpyAsync(doff,off,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append offsets");
    if(ok&&pb)ragged_kv_append<<<S,256,0,dc->stream>>>(ddl,ddr,dc->al,dold,dadd,doff,K,R,page_stride);
    if(ok)for(int s=0;s<S;s++){
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){w->ragged[i].length=lengths[s];break;}
    }
    std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);if(!ok)return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_ragged_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,ddl,ddr,
        dn,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,page_stride,scale,w->gs,w->ng);
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    return cuda_ok(cudaGetLastError(),"ragged attention launch")&&
           cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"ragged output download")&&
           cuda_ok(cudaStreamSynchronize(dc->stream),"ragged attention synchronize");
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        /* Must mirror the upload's accounting exactly -- literally the same
         * expression upload uses to charge (scale_count * sizeof(float), gated
         * on fmt=6 never having a separate scale buffer), so the two can no
         * longer drift independently. Over-subtracting here trips the >= guard
         * below, which silently leaves the tensor's bytes on the device counter
         * forever. */
        size_t storage_bytes =
#ifdef COLI_ANS
            tensor->compressed ? tensor->archive_bytes :
#endif
            tensor->weight_bytes;
        size_t bytes = storage_bytes +
            ((tensor->fmt && tensor->fmt != 6) ? tensor->scale_count * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights&&tensor->weights_owned) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    for(int i=0;i<tensor->ragged_count;i++)ragged_kv_clear(&tensor->ragged[i]);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    /* Must mirror upload's and free's accounting exactly -- literally the same
     * expression they use (scale_count * sizeof(float), gated on fmt=6 never
     * having a separate scale buffer) -- so all three can no longer drift
     * independently. The prior `O * ng` shape over-reported for fmt=8 (real
     * footprint is (O+127)/128 * ng block scales, not O * ng) and for fmt=6
     * (which has no separate scale buffer at all). */
    size_t storage_bytes =
#ifdef COLI_ANS
        tensor->compressed ? tensor->archive_bytes :
#endif
        tensor->weight_bytes;
    return storage_bytes +
        ((tensor->fmt && tensor->fmt != 6) ? tensor->scale_count * sizeof(float) : 0);
}

/* What a cudaMalloc of `bytes` actually takes off the card.
 *
 * coli_cuda_tensor_bytes() above is the LOGICAL size and must stay that way -
 * it mirrors upload and free so the three cannot drift. This is a different
 * question: the allocator rounds a request up, and nothing was charging the
 * difference to the expert budget.
 *
 * Measured on an RTX 3090 (sm_86, CUDA 13.1):
 *
 *     request              actual      overhead
 *     0.75 MiB          1.00 MiB        +0.25     <- int4-g64 scale array
 *     1.00 MiB          1.00 MiB        +0.00
 *     1.00 MiB + 1 B    2.00 MiB        +1.00
 *     1.50 MiB          2.00 MiB        +0.50
 *     2.00 MiB          2.00 MiB        +0.00
 *     2.00 MiB + 1 B    4.00 MiB        +2.00
 *     6.00 MiB          6.00 MiB        +0.00     <- int4-g64 weight tensor
 *
 * A GLM-5.2 int4-g64 expert is three 6 MiB weight arrays, which are already on
 * a boundary and cost nothing extra, and three 0.75 MiB scale arrays, each
 * padded to 1 MiB. 3 x 0.25 = 0.75 MiB per expert unaccounted - which is the
 * 0.741 MiB/expert (sigma 0.019) measured independently on H100 and H200 in
 * #687, from the other direction.
 *
 * PROBED, not modelled. The table above is not a rule this hardcodes: the
 * rounding is a driver and architecture property, and a formula fitted to one
 * card would be silently wrong on the next. Probed once per distinct size
 * and cached, so the cost does not scale with the tier.
 *
 * cudaMemGetInfo is the obvious alternative and it does not survive contact
 * with the numbers: it is O(live allocations), measured 0.52 us empty and
 * 68.2 us with 12,000 live, so charging it per expert would be quadratic
 * across a tier that places thousands.
 *
 * Returns `bytes` unchanged if the probe cannot run. Under-reporting is the
 * pre-existing behaviour and degrades to exactly what this replaced; refusing
 * to size at all would be worse.
 */
#define COLI_CUDA_FOOTPRINT_CACHE 16
static struct { size_t req, real; } g_fp_cache[COLI_CUDA_FOOTPRINT_CACHE];
static int g_fp_cache_n = 0;

/* The probe MUST allocate several buffers and divide, not one and measure it.
 *
 * A single cudaMalloc reserves a whole 2 MiB VMM page, so one allocation of
 * anything smaller reads as a flat 2 MiB and the answer is the page size
 * rather than the per-allocation cost. Later allocations suballocate from
 * pages already reserved, so the real amortised figure only appears once
 * enough of them are live to fill a page. Measured both ways on sm_86, same
 * 786,432-byte request:
 *
 *     1 allocation    2.00 MiB   <- the page, not the allocation
 *     64 allocations  1.00 MiB   <- the truth, and what the tier will pay
 *
 * The single-shot version of this function shipped in an earlier draft and its
 * own test caught it: it over-reported every expert by 3x and would have
 * shrunk the tier far more than the bug it fixes.
 */
#define COLI_CUDA_FOOTPRINT_PROBE_BYTES (24u << 20)   /* transient probe budget */
#define COLI_CUDA_FOOTPRINT_PROBE_MAX   64

extern "C" size_t coli_cuda_alloc_footprint(size_t bytes) {
    if (!bytes) return 0;
    for (int i = 0; i < g_fp_cache_n; i++)
        if (g_fp_cache[i].req == bytes) return g_fp_cache[i].real;

    /* Enough allocations to cross a page boundary, capped so the probe never
     * takes a meaningful bite out of a card that is about to be filled. */
    int want = (int)(COLI_CUDA_FOOTPRINT_PROBE_BYTES / bytes);
    if (want < 8) want = 8;
    if (want > COLI_CUDA_FOOTPRINT_PROBE_MAX) want = COLI_CUDA_FOOTPRINT_PROBE_MAX;

    size_t free_before = 0, free_after = 0, total = 0, real = bytes;
    void *p[COLI_CUDA_FOOTPRINT_PROBE_MAX];
    int made = 0;
    if (cudaMemGetInfo(&free_before, &total) == cudaSuccess) {
        for (int i = 0; i < want; i++) {
            if (cudaMalloc(&p[i], bytes) != cudaSuccess) break;
            made++;
        }
        if (made > 0 && cudaMemGetInfo(&free_after, &total) == cudaSuccess &&
            free_before > free_after) {
            size_t avg = (free_before - free_after) / (size_t)made;
            /* Only ever round UP. A concurrent free elsewhere on the device
             * can make the delta read small, and charging less than the
             * logical size is the failure this exists to fix. */
            if (avg > real) real = avg;
        }
        for (int i = 0; i < made; i++) cudaFree(p[i]);
    }
    /* Cache even a failed probe: `real` is then the logical size, which is the
     * pre-existing behaviour, and retrying per expert on a card that just
     * refused 8 allocations would be the worst possible time to try again. */
    if (g_fp_cache_n < COLI_CUDA_FOOTPRINT_CACHE)
        g_fp_cache[g_fp_cache_n++] = { bytes, real };
    return real;
}

/* Same split as coli_cuda_tensor_bytes, but per ALLOCATION rather than summed
 * first: the weights and the scales are two separate cudaMallocs and each is
 * rounded on its own. Summing then rounding once would miss the scale array's
 * padding entirely, which is the whole term. */
extern "C" size_t coli_cuda_tensor_vram(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    size_t storage_bytes =
#ifdef COLI_ANS
        tensor->compressed ? tensor->archive_bytes :
#endif
        tensor->weight_bytes;
    size_t total = coli_cuda_alloc_footprint(storage_bytes);
    if (tensor->fmt && tensor->fmt != 6)
        total += coli_cuda_alloc_footprint(tensor->scale_count * sizeof(float));
    return total;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}

/* ==== explicit-context resident core =====================================
 * Unlike the older pipe helpers below, these objects are tied to the
 * ColiGpuContext stream. Their only host synchronization points are upload,
 * download, and explicit context sync. */

static int gpu_range_ok(const ColiGpuArena *arena, size_t offset, size_t bytes) {
    return arena && offset <= arena->capacity && bytes <= arena->capacity - offset;
}

static int gpu_ranges_overlap(size_t a_offset, size_t a_bytes,
                              size_t b_offset, size_t b_bytes) {
    if (a_offset <= b_offset) return b_offset - a_offset < a_bytes;
    return a_offset - b_offset < b_bytes;
}

__host__ __device__ static size_t gpu_tensor_row_bytes(int format, int columns) {
    if (format == 0) return (size_t)columns * sizeof(float);
    if (format == 1) return (size_t)columns;
    if (format == 2 || format == 4) return (size_t)(columns + 1) / 2;
    return 0;
}

static int gpu_tensor_same_context(const ColiGpuArena *arena,
                                   const ColiGpuTensor *tensor) {
    return arena && tensor && arena->ctx == tensor->ctx;
}

extern "C" int coli_gpu_tensor_create(ColiGpuTensor **out,
                                       ColiGpuContext *ctx,
                                       const ColiGpuTensorDesc *desc) {
    if (!out) return 0;
    *out = nullptr;
    if (!ctx || !ctx->healthy || !desc || !desc->data ||
        desc->rows < 1 || desc->columns < 1 ||
        (desc->format != 0 && desc->format != 1 &&
         desc->format != 2 && desc->format != 4) ||
        (desc->format == 4 && desc->group_size < 1) ||
        (desc->format != 4 && desc->group_size != 0))
        return 0;
    if (!select_device_ordinal(ctx->device)) return 0;
    size_t row = gpu_tensor_row_bytes(desc->format, desc->columns);
    if (!row || (size_t)desc->rows > SIZE_MAX / row) return 0;
    if (desc->format == 0) {
        size_t count = (size_t)desc->rows * (size_t)desc->columns;
        const float *values = static_cast<const float *>(desc->data);
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(values[i])) return 0;
    } else {
        int groups = desc->format == 4
            ? (desc->columns + desc->group_size - 1) / desc->group_size : 1;
        size_t count = (size_t)desc->rows * (size_t)groups;
        if (!desc->scales) return 0;
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(desc->scales[i])) return 0;
    }
    ColiGpuTensor *tensor =
        static_cast<ColiGpuTensor *>(std::calloc(1, sizeof(*tensor)));
    if (!tensor) return 0;
    tensor->ctx = ctx;
    tensor->format = desc->format;
    tensor->rows = desc->rows;
    tensor->columns = desc->columns;
    tensor->group_size = desc->group_size;
    tensor->groups = desc->format == 4
        ? (desc->columns + desc->group_size - 1) / desc->group_size : 1;
    tensor->data_bytes = (size_t)desc->rows * row;
    tensor->scale_count = desc->format == 0 ? 0 :
        (size_t)desc->rows * tensor->groups;
    if (!cuda_ok(cudaMalloc(&tensor->data, tensor->data_bytes),
                 "resident tensor allocation")) {
        std::free(tensor);
        return 0;
    }
    ctx->telemetry.device_allocations++;
    if (!cuda_ok(cudaMemcpyAsync(tensor->data, desc->data, tensor->data_bytes,
                                 cudaMemcpyHostToDevice, ctx->stream),
                 "resident tensor upload")) {
        cudaFree(tensor->data);
        std::free(tensor);
        return 0;
    }
    ctx->telemetry.h2d_copies++;
    ctx->telemetry.h2d_bytes += tensor->data_bytes;
    if (tensor->scale_count) {
        if (!desc->scales ||
            !cuda_ok(cudaMalloc(&tensor->scales,
                                tensor->scale_count * sizeof(float)),
                     "resident tensor scales allocation")) {
            cudaFree(tensor->data);
            std::free(tensor);
            return 0;
        }
        ctx->telemetry.device_allocations++;
        if (!cuda_ok(cudaMemcpyAsync(tensor->scales, desc->scales,
                                     tensor->scale_count * sizeof(float),
                                     cudaMemcpyHostToDevice, ctx->stream),
                     "resident tensor scales upload")) {
            cudaFree(tensor->scales);
            cudaFree(tensor->data);
            std::free(tensor);
            return 0;
        }
        ctx->telemetry.h2d_copies++;
        ctx->telemetry.h2d_bytes += tensor->scale_count * sizeof(float);
    }
    if (!coli_gpu_context_sync(ctx)) {
        if (tensor->scales) cudaFree(tensor->scales);
        cudaFree(tensor->data);
        std::free(tensor);
        return 0;
    }
    *out = tensor;
    return 1;
}

extern "C" void coli_gpu_tensor_destroy(ColiGpuTensor *tensor) {
    if (!tensor) return;
    ColiGpuContext *ctx = tensor->ctx;
    if (ctx && select_device_ordinal(ctx->device)) {
        if (tensor->scales) cudaFree(tensor->scales);
        if (tensor->data) cudaFree(tensor->data);
    }
    std::free(tensor);
}

extern "C" int coli_gpu_arena_create(ColiGpuArena **out,
                                      ColiGpuContext *ctx,
                                      size_t capacity) {
    if (!out) return 0;
    *out = nullptr;
    if (!ctx || !ctx->healthy || !capacity ||
        !select_device_ordinal(ctx->device))
        return 0;
    ColiGpuArena *arena =
        static_cast<ColiGpuArena *>(std::calloc(1, sizeof(*arena)));
    if (!arena) return 0;
    arena->ctx = ctx;
    arena->capacity = capacity;
    if (!cuda_ok(cudaMalloc(&arena->data, capacity),
                 "resident arena allocation")) {
        std::free(arena);
        return 0;
    }
    ctx->telemetry.device_allocations++;
    *out = arena;
    return 1;
}

extern "C" void coli_gpu_arena_destroy(ColiGpuArena *arena) {
    if (!arena) return;
    if (arena->ctx && select_device_ordinal(arena->ctx->device) && arena->data)
        cudaFree(arena->data);
    std::free(arena);
}

extern "C" size_t coli_gpu_arena_capacity(const ColiGpuArena *arena) {
    return arena ? arena->capacity : 0;
}

extern "C" int coli_gpu_arena_upload(ColiGpuArena *arena, size_t offset,
                                      const void *src, size_t bytes) {
    if (!src || !bytes || !gpu_range_ok(arena, offset, bytes) ||
        !arena->ctx->healthy || !select_device_ordinal(arena->ctx->device))
        return 0;
    if (!cuda_ok(cudaMemcpyAsync(arena->data + offset, src, bytes,
                                 cudaMemcpyHostToDevice, arena->ctx->stream),
                 "resident arena upload") ||
        !coli_gpu_context_sync(arena->ctx))
        return 0;
    arena->ctx->telemetry.h2d_copies++;
    arena->ctx->telemetry.h2d_bytes += bytes;
    return 1;
}

extern "C" int coli_gpu_arena_download(ColiGpuArena *arena, size_t offset,
                                        void *dst, size_t bytes) {
    if (!dst || !bytes || !gpu_range_ok(arena, offset, bytes) ||
        !arena->ctx->healthy || !select_device_ordinal(arena->ctx->device))
        return 0;
    if (!cuda_ok(cudaMemcpyAsync(dst, arena->data + offset, bytes,
                                 cudaMemcpyDeviceToHost, arena->ctx->stream),
                 "resident arena download") ||
        !coli_gpu_context_sync(arena->ctx))
        return 0;
    arena->ctx->telemetry.d2h_copies++;
    arena->ctx->telemetry.d2h_bytes += bytes;
    return 1;
}

extern "C" int coli_gpu_arena_upload_activation(
    ColiGpuArena *arena, size_t offset, const void *src, size_t bytes) {
    if (!coli_gpu_arena_upload(arena, offset, src, bytes)) return 0;
    arena->ctx->telemetry.host_activation_h2d_copies++;
    return 1;
}

extern "C" int coli_gpu_arena_download_activation(
    ColiGpuArena *arena, size_t offset, void *dst, size_t bytes) {
    if (!coli_gpu_arena_download(arena, offset, dst, bytes)) return 0;
    arena->ctx->telemetry.host_activation_d2h_copies++;
    return 1;
}

__global__ static void gpu_embedding_kernel(float *output, const int32_t *tokens,
                                             const float *embedding,
                                             int streams, int hidden) {
    int row = (int)blockIdx.x;
    int token = tokens[row];
    for (int i = (int)threadIdx.x; i < streams * hidden; i += (int)blockDim.x)
        output[(size_t)row * streams * hidden + i] =
            embedding[(size_t)token * hidden + i % hidden];
}

__global__ static void gpu_rmsnorm_kernel(float *output, const float *input,
                                           const float *weight, int hidden,
                                           float eps) {
    int row = (int)blockIdx.x;
    const float *x = input + (size_t)row * hidden;
    float *y = output + (size_t)row * hidden;
    __shared__ double partial[256];
    double sum = 0.0;
    for (int d = (int)threadIdx.x; d < hidden; d += (int)blockDim.x)
        sum += (double)x[d] * x[d];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = (int)blockDim.x / 2; width; width >>= 1) {
        if ((int)threadIdx.x < width)
            partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    float inverse = rsqrtf((float)(partial[0] / hidden) + eps);
    for (int d = (int)threadIdx.x; d < hidden; d += (int)blockDim.x)
        y[d] = x[d] * inverse * weight[d];
}

__global__ static void gpu_layernorm_kernel(float *output, const float *input,
                                             const float *weight,
                                             const float *bias, int hidden,
                                             float eps) {
    int row = (int)blockIdx.x;
    const float *x = input + (size_t)row * hidden;
    float *y = output + (size_t)row * hidden;
    __shared__ double partial[256];
    double sum = 0.0;
    for (int d = (int)threadIdx.x; d < hidden; d += (int)blockDim.x)
        sum += x[d];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = (int)blockDim.x / 2; width; width >>= 1) {
        if ((int)threadIdx.x < width)
            partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    float mean = (float)(partial[0] / hidden);
    sum = 0.0;
    for (int d = (int)threadIdx.x; d < hidden; d += (int)blockDim.x) {
        float centered = x[d] - mean;
        sum += (double)centered * centered;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = (int)blockDim.x / 2; width; width >>= 1) {
        if ((int)threadIdx.x < width)
            partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    float inverse = rsqrtf((float)(partial[0] / hidden) + eps);
    for (int d = (int)threadIdx.x; d < hidden; d += (int)blockDim.x)
        y[d] = (x[d] - mean) * inverse * weight[d] + bias[d];
}

__device__ static float gpu_stable_sigmoid(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

__global__ static void gpu_mhc_pre_kernel(
    float *collapsed, float *post_out, float *comb_out, const float *residual,
    const float *fn, const float *scale, const float *base,
    int streams, int hidden, float norm_eps, float hc_eps) {
    int row = (int)blockIdx.x;
    int flattened = streams * hidden;
    int mix_count = (2 + streams) * streams;
    const float *input = residual + (size_t)row * flattened;
    __shared__ double partial[256];
    __shared__ float mixes[80];
    __shared__ float pre[8];
    __shared__ float post[8];
    __shared__ float comb[64];
    double square = 0.0;
    for (int i = (int)threadIdx.x; i < flattened; i += (int)blockDim.x)
        square += (double)input[i] * input[i];
    partial[threadIdx.x] = square;
    __syncthreads();
    for (int width = (int)blockDim.x / 2; width; width >>= 1) {
        if ((int)threadIdx.x < width)
            partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    float inverse = rsqrtf((float)(partial[0] / flattened) + norm_eps);
    for (int mix = 0; mix < mix_count; ++mix) {
        double sum = 0.0;
        const float *w = fn + (size_t)mix * flattened;
        for (int i = (int)threadIdx.x; i < flattened; i += (int)blockDim.x)
            sum += (double)w[i] * input[i];
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (int width = (int)blockDim.x / 2; width; width >>= 1) {
            if ((int)threadIdx.x < width)
                partial[threadIdx.x] += partial[threadIdx.x + width];
            __syncthreads();
        }
        if (!threadIdx.x) mixes[mix] = (float)partial[0] * inverse;
        __syncthreads();
    }
    if (!threadIdx.x) {
        for (int index = 0; index < streams; ++index) {
            pre[index] = gpu_stable_sigmoid(
                mixes[index] * scale[0] + base[index]) + hc_eps;
            post[index] = 2.0f * gpu_stable_sigmoid(
                mixes[streams + index] * scale[1] + base[streams + index]);
        }
        int matrix_offset = 2 * streams;
        for (int r = 0; r < streams; ++r) {
            float maximum = -INFINITY;
            for (int c = 0; c < streams; ++c) {
                int index = matrix_offset + r * streams + c;
                float value = mixes[index] * scale[2] + base[index];
                comb[r * streams + c] = value;
                maximum = fmaxf(maximum, value);
            }
            float sum = 0.0f;
            for (int c = 0; c < streams; ++c) {
                float value = expf(comb[r * streams + c] - maximum);
                comb[r * streams + c] = value;
                sum += value;
            }
            for (int c = 0; c < streams; ++c)
                comb[r * streams + c] =
                    comb[r * streams + c] / sum + hc_eps;
        }
        for (int c = 0; c < streams; ++c) {
            float sum = 0.0f;
            for (int r = 0; r < streams; ++r)
                sum += comb[r * streams + c];
            for (int r = 0; r < streams; ++r)
                comb[r * streams + c] /= sum + hc_eps;
        }
        for (int iteration = 1; iteration < 20; ++iteration) {
            for (int r = 0; r < streams; ++r) {
                float sum = 0.0f;
                for (int c = 0; c < streams; ++c)
                    sum += comb[r * streams + c];
                for (int c = 0; c < streams; ++c)
                    comb[r * streams + c] /= sum + hc_eps;
            }
            for (int c = 0; c < streams; ++c) {
                float sum = 0.0f;
                for (int r = 0; r < streams; ++r)
                    sum += comb[r * streams + c];
                for (int r = 0; r < streams; ++r)
                    comb[r * streams + c] /= sum + hc_eps;
            }
        }
        for (int i = 0; i < streams; ++i)
            post_out[(size_t)row * streams + i] = post[i];
        for (int i = 0; i < streams * streams; ++i)
            comb_out[(size_t)row * streams * streams + i] = comb[i];
    }
    __syncthreads();
    for (int d = (int)threadIdx.x; d < hidden; d += (int)blockDim.x) {
        float sum = 0.0f;
        for (int stream = 0; stream < streams; ++stream)
            sum += pre[stream] * input[(size_t)stream * hidden + d];
        collapsed[(size_t)row * hidden + d] = sum;
    }
}

__global__ static void gpu_mhc_post_kernel(
    float *output, const float *branch, const float *residual,
    const float *post, const float *comb, int streams, int hidden) {
    int row = (int)blockIdx.x;
    int total = streams * hidden;
    for (int i = (int)threadIdx.x; i < total; i += (int)blockDim.x) {
        int destination = i / hidden;
        int column = i % hidden;
        float value = 0.0f;
        for (int source = 0; source < streams; ++source)
            value += comb[((size_t)row * streams + source) * streams + destination] *
                     residual[((size_t)row * streams + source) * hidden + column];
        value += post[(size_t)row * streams + destination] *
                 branch[(size_t)row * hidden + column];
        output[(size_t)row * total + i] = value;
    }
}

__global__ static void gpu_collapse_kernel(float *output, const float *input,
                                            int streams, int hidden) {
    int row = (int)blockIdx.x;
    for (int d = (int)threadIdx.x; d < hidden; d += (int)blockDim.x) {
        float sum = 0.0f;
        for (int stream = 0; stream < streams; ++stream)
            sum += input[((size_t)row * streams + stream) * hidden + d];
        output[(size_t)row * hidden + d] = sum / streams;
    }
}

__global__ static void gpu_projection_kernel(
    float *output, const float *input, const void *weight,
    const float *scales, int format, int groups, int group_size,
    int input_size, int output_size) {
    int out = (int)blockIdx.x;
    int row_index = (int)blockIdx.y;
    if (out >= output_size) return;
    const float *x = input + (size_t)row_index * input_size;
    size_t packed_row = (size_t)out * gpu_tensor_row_bytes(format, input_size);
    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < input_size; i += (int)blockDim.x) {
        float value = weight_at(weight, format, packed_row, i);
        float scale_value = 1.0f;
        if (format == 1 || format == 2) scale_value = scales[out];
        else if (format == 4)
            scale_value = scales[(size_t)out * groups + i / group_size];
        sum += x[i] * value * scale_value;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = (int)blockDim.x / 2; width; width >>= 1) {
        if ((int)threadIdx.x < width)
            partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    if (!threadIdx.x)
        output[(size_t)row_index * output_size + out] = partial[0];
}

extern "C" int coli_gpu_embedding(ColiGpuArena *arena,
                                   size_t streams_offset,
                                   size_t token_ids_offset,
                                   const ColiGpuTensor *embedding,
                                   int rows, int streams, int hidden) {
    size_t output_bytes = (size_t)rows * streams * hidden * sizeof(float);
    size_t token_bytes = (size_t)rows * sizeof(int32_t);
    if (rows < 1 || streams < 1 || hidden < 1 ||
        !gpu_tensor_same_context(arena, embedding) ||
        embedding->format != 0 || embedding->columns != hidden ||
        !gpu_range_ok(arena, streams_offset, output_bytes) ||
        !gpu_range_ok(arena, token_ids_offset, token_bytes) ||
        !select_device_ordinal(arena->ctx->device))
        return 0;
    gpu_embedding_kernel<<<rows, 256, 0, arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + streams_offset),
        reinterpret_cast<const int32_t *>(arena->data + token_ids_offset),
        reinterpret_cast<const float *>(embedding->data), streams, hidden);
    return cuda_ok(cudaGetLastError(), "resident embedding launch");
}

extern "C" int coli_gpu_rmsnorm(ColiGpuArena *arena, size_t output_offset,
                                 size_t input_offset,
                                 const ColiGpuTensor *weight,
                                 int rows, int hidden, float eps) {
    size_t bytes = (size_t)rows * hidden * sizeof(float);
    if (rows < 1 || hidden < 1 || eps < 0.0f ||
        !gpu_tensor_same_context(arena, weight) ||
        weight->format != 0 || weight->rows != 1 ||
        weight->columns != hidden ||
        !gpu_range_ok(arena, output_offset, bytes) ||
        !gpu_range_ok(arena, input_offset, bytes) ||
        !select_device_ordinal(arena->ctx->device))
        return 0;
    gpu_rmsnorm_kernel<<<rows, 256, 0, arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + output_offset),
        reinterpret_cast<const float *>(arena->data + input_offset),
        reinterpret_cast<const float *>(weight->data), hidden, eps);
    return cuda_ok(cudaGetLastError(), "resident RMSNorm launch");
}

extern "C" int coli_gpu_layernorm(ColiGpuArena *arena, size_t output_offset,
                                   size_t input_offset,
                                   const ColiGpuTensor *weight,
                                   const ColiGpuTensor *bias,
                                   int rows, int hidden, float eps) {
    size_t bytes = (size_t)rows * hidden * sizeof(float);
    if (rows < 1 || hidden < 1 || eps < 0.0f ||
        !gpu_tensor_same_context(arena, weight) ||
        !gpu_tensor_same_context(arena, bias) ||
        weight->format != 0 || bias->format != 0 ||
        weight->rows != 1 || bias->rows != 1 ||
        weight->columns != hidden || bias->columns != hidden ||
        !gpu_range_ok(arena, output_offset, bytes) ||
        !gpu_range_ok(arena, input_offset, bytes) ||
        !select_device_ordinal(arena->ctx->device))
        return 0;
    gpu_layernorm_kernel<<<rows, 256, 0, arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + output_offset),
        reinterpret_cast<const float *>(arena->data + input_offset),
        reinterpret_cast<const float *>(weight->data),
        reinterpret_cast<const float *>(bias->data), hidden, eps);
    return cuda_ok(cudaGetLastError(), "resident LayerNorm launch");
}

extern "C" int coli_gpu_mhc_pre_norm(
    ColiGpuArena *arena, size_t collapsed_offset, size_t normed_offset,
    size_t post_offset, size_t comb_offset, size_t residual_offset,
    const ColiGpuTensor *fn, const ColiGpuTensor *scale,
    const ColiGpuTensor *base, const ColiGpuTensor *norm_weight,
    int rows, int streams, int hidden, float norm_eps, float hc_eps) {
    int mix_count = (2 + streams) * streams;
    size_t residual_bytes = (size_t)rows * streams * hidden * sizeof(float);
    size_t row_bytes_f = (size_t)rows * hidden * sizeof(float);
    size_t post_bytes = (size_t)rows * streams * sizeof(float);
    size_t comb_bytes = (size_t)rows * streams * streams * sizeof(float);
    if (rows < 1 || streams < 1 || streams > 8 || hidden < 1 ||
        norm_eps < 0.0f || hc_eps < 0.0f ||
        !gpu_tensor_same_context(arena, fn) ||
        !gpu_tensor_same_context(arena, scale) ||
        !gpu_tensor_same_context(arena, base) ||
        !gpu_tensor_same_context(arena, norm_weight) ||
        fn->format != 0 || fn->rows != mix_count ||
        fn->columns != streams * hidden ||
        scale->format != 0 || scale->rows != 1 || scale->columns != 3 ||
        base->format != 0 || base->rows != 1 || base->columns != mix_count ||
        norm_weight->format != 0 || norm_weight->rows != 1 ||
        norm_weight->columns != hidden ||
        !gpu_range_ok(arena, residual_offset, residual_bytes) ||
        !gpu_range_ok(arena, collapsed_offset, row_bytes_f) ||
        !gpu_range_ok(arena, normed_offset, row_bytes_f) ||
        !gpu_range_ok(arena, post_offset, post_bytes) ||
        !gpu_range_ok(arena, comb_offset, comb_bytes) ||
        !select_device_ordinal(arena->ctx->device))
        return 0;
    gpu_mhc_pre_kernel<<<rows, 256, 0, arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + collapsed_offset),
        reinterpret_cast<float *>(arena->data + post_offset),
        reinterpret_cast<float *>(arena->data + comb_offset),
        reinterpret_cast<const float *>(arena->data + residual_offset),
        reinterpret_cast<const float *>(fn->data),
        reinterpret_cast<const float *>(scale->data),
        reinterpret_cast<const float *>(base->data),
        streams, hidden, norm_eps, hc_eps);
    if (!cuda_ok(cudaGetLastError(), "resident mHC pre launch")) return 0;
    return coli_gpu_rmsnorm(arena, normed_offset, collapsed_offset, norm_weight,
                            rows, hidden, norm_eps);
}

extern "C" int coli_gpu_mhc_post(
    ColiGpuArena *arena, size_t output_offset, size_t branch_offset,
    size_t residual_offset, size_t post_offset, size_t comb_offset,
    int rows, int streams, int hidden) {
    size_t residual_bytes = (size_t)rows * streams * hidden * sizeof(float);
    size_t branch_bytes = (size_t)rows * hidden * sizeof(float);
    size_t post_bytes = (size_t)rows * streams * sizeof(float);
    size_t comb_bytes = (size_t)rows * streams * streams * sizeof(float);
    if (rows < 1 || streams < 1 || streams > 8 || hidden < 1 ||
        !gpu_range_ok(arena, output_offset, residual_bytes) ||
        !gpu_range_ok(arena, residual_offset, residual_bytes) ||
        !gpu_range_ok(arena, branch_offset, branch_bytes) ||
        !gpu_range_ok(arena, post_offset, post_bytes) ||
        !gpu_range_ok(arena, comb_offset, comb_bytes) ||
        !select_device_ordinal(arena->ctx->device))
        return 0;
    gpu_mhc_post_kernel<<<rows, 256, 0, arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + output_offset),
        reinterpret_cast<const float *>(arena->data + branch_offset),
        reinterpret_cast<const float *>(arena->data + residual_offset),
        reinterpret_cast<const float *>(arena->data + post_offset),
        reinterpret_cast<const float *>(arena->data + comb_offset),
        streams, hidden);
    return cuda_ok(cudaGetLastError(), "resident mHC post launch");
}

extern "C" int coli_gpu_mhc_site(
    ColiGpuArena *arena, size_t output_offset, size_t collapsed_offset,
    size_t normed_offset, size_t post_offset, size_t comb_offset,
    size_t residual_offset, size_t branch_offset,
    const ColiGpuTensor *fn, const ColiGpuTensor *scale,
    const ColiGpuTensor *base, const ColiGpuTensor *norm_weight,
    int rows, int streams, int hidden, float norm_eps, float hc_eps) {
    return coli_gpu_mhc_pre_norm(
               arena, collapsed_offset, normed_offset, post_offset, comb_offset,
               residual_offset, fn, scale, base, norm_weight,
               rows, streams, hidden, norm_eps, hc_eps) &&
           coli_gpu_mhc_post(
               arena, output_offset, branch_offset, residual_offset,
               post_offset, comb_offset, rows, streams, hidden);
}

extern "C" int coli_gpu_collapse_streams(
    ColiGpuArena *arena, size_t output_offset, size_t streams_offset,
    int rows, int streams, int hidden) {
    size_t output_bytes = (size_t)rows * hidden * sizeof(float);
    size_t input_bytes = (size_t)rows * streams * hidden * sizeof(float);
    if (rows < 1 || streams < 1 || hidden < 1 ||
        !gpu_range_ok(arena, output_offset, output_bytes) ||
        !gpu_range_ok(arena, streams_offset, input_bytes) ||
        !select_device_ordinal(arena->ctx->device))
        return 0;
    gpu_collapse_kernel<<<rows, 256, 0, arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + output_offset),
        reinterpret_cast<const float *>(arena->data + streams_offset),
        streams, hidden);
    return cuda_ok(cudaGetLastError(), "resident stream collapse launch");
}

extern "C" int coli_gpu_projection(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    const ColiGpuTensor *weight, int rows, int input_size, int output_size) {
    size_t input_bytes = (size_t)rows * input_size * sizeof(float);
    size_t output_bytes = (size_t)rows * output_size * sizeof(float);
    if (rows < 1 || input_size < 1 || output_size < 1 ||
        !gpu_tensor_same_context(arena, weight) ||
        weight->columns != input_size || weight->rows != output_size ||
        !gpu_range_ok(arena, input_offset, input_bytes) ||
        !gpu_range_ok(arena, output_offset, output_bytes) ||
        !select_device_ordinal(arena->ctx->device))
        return 0;
    gpu_projection_kernel<<<dim3((unsigned)output_size, (unsigned)rows), 256, 0,
                                arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + output_offset),
        reinterpret_cast<const float *>(arena->data + input_offset),
        weight->data, weight->scales, weight->format, weight->groups,
        weight->group_size, input_size, output_size);
    return cuda_ok(cudaGetLastError(), "resident projection launch");
}

extern "C" int coli_gpu_dense_mlp(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t gate_offset, size_t up_offset,
    const ColiGpuTensor *gate, const ColiGpuTensor *up,
    const ColiGpuTensor *down, int rows, int hidden, int intermediate,
    float swiglu_limit) {
    if (!arena || rows < 1 || hidden < 1 || intermediate < 1 ||
        !std::isfinite(swiglu_limit) || swiglu_limit <= 0.0f)
        return 0;
    size_t hidden_bytes = (size_t)rows * hidden * sizeof(float);
    size_t intermediate_bytes =
        (size_t)rows * intermediate * sizeof(float);
    if (!gpu_range_ok(arena, input_offset, hidden_bytes) ||
        !gpu_range_ok(arena, output_offset, hidden_bytes) ||
        !gpu_range_ok(arena, gate_offset, intermediate_bytes) ||
        !gpu_range_ok(arena, up_offset, intermediate_bytes) ||
        gpu_ranges_overlap(input_offset, hidden_bytes,
                           output_offset, hidden_bytes) ||
        gpu_ranges_overlap(input_offset, hidden_bytes,
                           gate_offset, intermediate_bytes) ||
        gpu_ranges_overlap(input_offset, hidden_bytes,
                           up_offset, intermediate_bytes) ||
        gpu_ranges_overlap(output_offset, hidden_bytes,
                           gate_offset, intermediate_bytes) ||
        gpu_ranges_overlap(output_offset, hidden_bytes,
                           up_offset, intermediate_bytes) ||
        gpu_ranges_overlap(gate_offset, intermediate_bytes,
                           up_offset, intermediate_bytes))
        return 0;
    if (!coli_gpu_projection(arena, gate_offset, input_offset, gate,
                             rows, hidden, intermediate) ||
        !coli_gpu_projection(arena, up_offset, input_offset, up,
                             rows, hidden, intermediate))
        return 0;
    size_t values = (size_t)rows * intermediate;
    swiglu_clamped_kernel<<<(unsigned)((values + 255) / 256), 256, 0,
                            arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + gate_offset),
        reinterpret_cast<const float *>(arena->data + up_offset),
        values, swiglu_limit);
    if (!cuda_ok(cudaGetLastError(), "resident dense SwiGLU launch"))
        return 0;
    return coli_gpu_projection(arena, output_offset, gate_offset, down,
                               rows, intermediate, hidden);
}

__global__ static void gpu_router_score_kernel(
    const float *input, const float *weight, const float *bias,
    float *scores, float *choices, int hidden, int experts,
    int *status) {
    int expert = (int)blockIdx.x;
    int row = (int)blockIdx.y;
    const float *x = input + (size_t)row * hidden;
    const float *w = weight + (size_t)expert * hidden;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (int column = (int)threadIdx.x; column < hidden;
         column += (int)blockDim.x) {
        float xv = x[column];
        if (!isfinite(xv)) atomicMax(status, 1);
        sum += xv * w[column];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = (int)blockDim.x / 2; width; width >>= 1) {
        if ((int)threadIdx.x < width)
            partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    if (!threadIdx.x) {
        float score = 1.0f / (1.0f + expf(-partial[0]));
        float choice = score + (bias ? bias[expert] : 0.0f);
        if (!isfinite(score) || !isfinite(choice)) atomicMax(status, 2);
        scores[(size_t)row * experts + expert] = score;
        choices[(size_t)row * experts + expert] = choice;
    }
}

__global__ static void gpu_router_select_kernel(
    const float *scores, const float *choices, int *selected, float *weights,
    int rows, int experts, int topk, int normalize, float routed_scale,
    int *status) {
    int row = (int)blockIdx.x;
    if (threadIdx.x || row >= rows) return;
    const float *row_scores = scores + (size_t)row * experts;
    const float *row_choices = choices + (size_t)row * experts;
    int *row_selected = selected + (size_t)row * topk;
    float *row_weights = weights + (size_t)row * topk;
    float total = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int best = -1;
        float best_value = -INFINITY;
        for (int expert = 0; expert < experts; ++expert) {
            int used = 0;
            for (int prior = 0; prior < k; ++prior)
                if (row_selected[prior] == expert) used = 1;
            float value = row_choices[expert];
            if (!used && value > best_value) {
                best = expert;
                best_value = value;
            }
        }
        if (best < 0) {
            atomicMax(status, 2);
            return;
        }
        row_selected[k] = best;
        row_weights[k] = row_scores[best];
        total += row_weights[k];
    }
    for (int k = 0; k < topk; ++k) {
        float value = row_weights[k];
        if (normalize) value /= total + 1e-20f;
        value *= routed_scale;
        if (!isfinite(value)) atomicMax(status, 2);
        row_weights[k] = value;
    }
}

static int gpu_route_config_ok(const ColiGpuRouteConfig *config,
                               int max_rows) {
    return config && config->hidden > 0 && config->experts > 0 &&
           config->experts <= 4096 && config->topk > 0 &&
           config->topk <= config->experts && config->topk <= 64 &&
           (config->normalize_topk == 0 || config->normalize_topk == 1) &&
           std::isfinite(config->routed_scale) && max_rows > 0 &&
           (size_t)max_rows <= SIZE_MAX / (size_t)config->experts &&
           (size_t)max_rows <= SIZE_MAX / (size_t)config->topk;
}

extern "C" int coli_gpu_router_create(
    ColiGpuRouter **out, ColiGpuContext *ctx,
    const ColiGpuRouteConfig *config, int max_rows) {
    if (!out) return 0;
    *out = nullptr;
    if (!ctx || !ctx->healthy || !gpu_route_config_ok(config, max_rows) ||
        !select_device_ordinal(ctx->device))
        return 0;
    ColiGpuRouter *router =
        static_cast<ColiGpuRouter *>(std::calloc(1, sizeof(*router)));
    if (!router) return 0;
    router->ctx = ctx;
    router->config = *config;
    router->max_rows = max_rows;
    size_t score_count = (size_t)max_rows * config->experts;
    size_t selected_count = (size_t)max_rows * config->topk;
    size_t score_bytes = score_count * sizeof(float);
    size_t selected_bytes = selected_count * sizeof(int);
    size_t weight_bytes = selected_count * sizeof(float);
    size_t total = score_bytes * 2 + selected_bytes + weight_bytes;
    void *mapped = nullptr;
    void *selected_host = nullptr;
    if (!cuda_ok(cudaMalloc(&router->allocation, total),
                 "resident router allocation") ||
        !cuda_ok(cudaHostAlloc(&selected_host, selected_bytes, 0),
                 "resident router selected-id staging allocation") ||
        !cuda_ok(cudaHostAlloc(&mapped, sizeof(int), cudaHostAllocMapped),
                 "resident router status allocation")) {
        if (selected_host) (void)cudaFreeHost(selected_host);
        if (router->allocation) (void)cudaFree(router->allocation);
        std::free(router);
        return 0;
    }
    router->scores = static_cast<float *>(router->allocation);
    router->choices = reinterpret_cast<float *>(
        reinterpret_cast<unsigned char *>(router->allocation) + score_bytes);
    router->selected = reinterpret_cast<int *>(
        reinterpret_cast<unsigned char *>(router->allocation) + score_bytes * 2);
    router->weights = reinterpret_cast<float *>(
        reinterpret_cast<unsigned char *>(router->selected) + selected_bytes);
    router->host_selected = static_cast<int *>(selected_host);
    router->host_status = static_cast<volatile int *>(mapped);
    if (!cuda_ok(cudaHostGetDevicePointer(
            reinterpret_cast<void **>(&router->device_status), mapped, 0),
            "resident router status pointer")) {
        (void)cudaFreeHost(mapped);
        (void)cudaFreeHost(selected_host);
        (void)cudaFree(router->allocation);
        std::free(router);
        return 0;
    }
    *router->host_status = 0;
    ctx->telemetry.device_allocations++;
    *out = router;
    return 1;
}

extern "C" void coli_gpu_router_destroy(ColiGpuRouter *router) {
    if (!router) return;
    if (router->ctx && select_device_ordinal(router->ctx->device)) {
        coli_gpu_context_sync(router->ctx);
        if (router->allocation) (void)cudaFree(router->allocation);
    }
    if (router->host_status)
        (void)cudaFreeHost(const_cast<int *>(router->host_status));
    if (router->host_selected) (void)cudaFreeHost(router->host_selected);
    std::free(router);
}

extern "C" int coli_gpu_router_run(
    ColiGpuRouter *router, ColiGpuArena *arena, size_t input_offset,
    const ColiGpuTensor *weight, const ColiGpuTensor *correction_bias,
    int rows) {
    if (!router || !arena || router->ctx != arena->ctx ||
        rows < 1 || rows > router->max_rows ||
        !gpu_tensor_same_context(arena, weight) || weight->format != 0 ||
        weight->rows != router->config.experts ||
        weight->columns != router->config.hidden ||
        (correction_bias &&
         (!gpu_tensor_same_context(arena, correction_bias) ||
          correction_bias->format != 0 || correction_bias->rows != 1 ||
          correction_bias->columns != router->config.experts)) ||
        !gpu_range_ok(arena, input_offset,
                      (size_t)rows * router->config.hidden * sizeof(float)) ||
        !router->ctx->healthy || !select_device_ordinal(router->ctx->device))
        return 0;
    *router->host_status = 0;
    gpu_router_score_kernel<<<
        dim3((unsigned)router->config.experts, (unsigned)rows), 256, 0,
        router->ctx->stream>>>(
        reinterpret_cast<const float *>(arena->data + input_offset),
        static_cast<const float *>(weight->data),
        correction_bias
            ? static_cast<const float *>(correction_bias->data) : nullptr,
        router->scores, router->choices, router->config.hidden,
        router->config.experts, router->device_status);
    gpu_router_select_kernel<<<rows, 1, 0, router->ctx->stream>>>(
        router->scores, router->choices, router->selected, router->weights,
        rows, router->config.experts, router->config.topk,
        router->config.normalize_topk, router->config.routed_scale,
        router->device_status);
    size_t id_bytes =
        (size_t)rows * router->config.topk * sizeof(int);
    if (!cuda_ok(cudaGetLastError(), "resident router launch") ||
        !cuda_ok(cudaMemcpyAsync(
            router->host_selected, router->selected, id_bytes,
            cudaMemcpyDeviceToHost, router->ctx->stream),
            "resident router selected metadata") ||
        !coli_gpu_context_sync(router->ctx) || *router->host_status)
        return 0;
    router->ctx->telemetry.d2h_copies++;
    router->ctx->telemetry.d2h_bytes += id_bytes;
    router->rows = rows;
    router->ctx->telemetry.route_launches++;
    router->ctx->telemetry.selected_expert_count +=
        (uint64_t)rows * (uint64_t)router->config.topk;
    return 1;
}

extern "C" int coli_gpu_router_download(
    ColiGpuRouter *router, int *selected_ids, float *routing_weights,
    size_t selected_count, int rows) {
    if (!router || !selected_ids || !routing_weights || rows < 1 ||
        rows != router->rows ||
        selected_count != (size_t)rows * router->config.topk ||
        !select_device_ordinal(router->ctx->device))
        return 0;
    size_t id_bytes = selected_count * sizeof(int);
    size_t weight_bytes = selected_count * sizeof(float);
    std::memcpy(selected_ids, router->host_selected, id_bytes);
    if (!cuda_ok(cudaMemcpyAsync(routing_weights, router->weights, weight_bytes,
                                 cudaMemcpyDeviceToHost, router->ctx->stream),
                 "resident router weights download") ||
        !coli_gpu_context_sync(router->ctx))
        return 0;
    router->ctx->telemetry.d2h_copies++;
    router->ctx->telemetry.d2h_bytes += weight_bytes;
    return 1;
}

static size_t gpu_align256(size_t value) {
    return (value + 255u) & ~(size_t)255u;
}

static int gpu_expert_source_ok(const ColiGpuExpertCache *cache,
                                const ColiGpuExpertSource *source) {
    if (!cache || !source) return 0;
    const ColiGpuTensorDesc *all[3] = {
        &source->gate, &source->up, &source->down
    };
    const int rows[3] = {
        cache->config.intermediate,
        cache->config.intermediate,
        cache->config.hidden
    };
    const int columns[3] = {
        cache->config.hidden,
        cache->config.hidden,
        cache->config.intermediate
    };
    for (int tensor = 0; tensor < 3; ++tensor) {
        const ColiGpuTensorDesc *desc = all[tensor];
        if (!desc->data || !desc->scales || desc->format != 4 ||
            desc->rows != rows[tensor] || desc->columns != columns[tensor] ||
            desc->group_size != cache->config.group_size)
            return 0;
        size_t groups = ((size_t)desc->columns + desc->group_size - 1) /
                        desc->group_size;
        size_t scale_count = (size_t)desc->rows * groups;
        for (size_t index = 0; index < scale_count; ++index)
            if (!std::isfinite(desc->scales[index])) return 0;
    }
    return 1;
}

static void gpu_expert_bank_layout(ColiGpuExpertCache *cache,
                                   ColiGpuExpertBank *bank) {
    size_t gate_data = 0;
    size_t gate_scales = gpu_align256(
        gate_data + cache->gate_data_bytes);
    size_t up_data = gpu_align256(
        gate_scales + cache->gate_scale_bytes);
    size_t up_scales = gpu_align256(
        up_data + cache->gate_data_bytes);
    size_t down_data = gpu_align256(
        up_scales + cache->gate_scale_bytes);
    size_t down_scales = gpu_align256(
        down_data + cache->down_data_bytes);
    unsigned char *device = bank->allocation;
    bank->gate_data = device + gate_data;
    bank->gate_scales = reinterpret_cast<float *>(device + gate_scales);
    bank->up_data = device + up_data;
    bank->up_scales = reinterpret_cast<float *>(device + up_scales);
    bank->down_data = device + down_data;
    bank->down_scales = reinterpret_cast<float *>(device + down_scales);
}

extern "C" int coli_gpu_expert_cache_create(
    ColiGpuExpertCache **out, ColiGpuContext *ctx,
    const ColiGpuExpertCacheConfig *config) {
    if (!out) return 0;
    *out = nullptr;
    if (!ctx || !ctx->healthy || !config || config->experts < 1 ||
        config->experts > 4096 || config->slots < 1 ||
        config->slots > config->experts || config->hidden < 1 ||
        config->intermediate < 1 || config->group_size != 64 ||
        config->max_rows < 1 || !std::isfinite(config->swiglu_limit) ||
        config->swiglu_limit <= 0.0f ||
        !select_device_ordinal(ctx->device))
        return 0;
    ColiGpuExpertCache *cache = new (std::nothrow) ColiGpuExpertCache{};
    if (!cache) return 0;
    cache->ctx = ctx;
    cache->config = *config;
    cache->healthy = 1;
    cache->gate_data_bytes =
        (size_t)config->intermediate * ((config->hidden + 1u) / 2u);
    cache->gate_scale_bytes =
        (size_t)config->intermediate *
        ((config->hidden + config->group_size - 1u) / config->group_size) *
        sizeof(float);
    cache->down_data_bytes =
        (size_t)config->hidden * ((config->intermediate + 1u) / 2u);
    cache->down_scale_bytes =
        (size_t)config->hidden *
        ((config->intermediate + config->group_size - 1u) /
         config->group_size) * sizeof(float);
    size_t cursor = 0;
    cursor = gpu_align256(cursor + cache->gate_data_bytes);
    cursor = gpu_align256(cursor + cache->gate_scale_bytes);
    cursor = gpu_align256(cursor + cache->gate_data_bytes);
    cursor = gpu_align256(cursor + cache->gate_scale_bytes);
    cursor = gpu_align256(cursor + cache->down_data_bytes);
    cache->bank_bytes = gpu_align256(cursor + cache->down_scale_bytes);
    if (!cache->bank_bytes ||
        (size_t)config->slots > SIZE_MAX / sizeof(*cache->slots)) {
        delete cache;
        return 0;
    }
    cache->slots = static_cast<ColiGpuExpertSlot *>(
        std::calloc((size_t)config->slots, sizeof(*cache->slots)));
    size_t max_topk =
        (size_t)(config->experts < 64 ? config->experts : 64);
    if ((size_t)config->max_rows > SIZE_MAX / max_topk) {
        std::free(cache->slots);
        delete cache;
        return 0;
    }
    cache->snapshot_capacity = (size_t)config->max_rows * max_topk;
    cache->snapshots = static_cast<ColiGpuExpertSnapshot *>(
        std::calloc(cache->snapshot_capacity, sizeof(*cache->snapshots)));
    if (!cache->slots || !cache->snapshots) {
        std::free(cache->snapshots);
        std::free(cache->slots);
        delete cache;
        return 0;
    }
    void *mapped_status = nullptr;
    if (!cuda_ok(cudaHostAlloc(
            &mapped_status, sizeof(int), cudaHostAllocMapped),
            "expert cache status allocation") ||
        !cuda_ok(cudaHostGetDevicePointer(
            reinterpret_cast<void **>(&cache->device_status),
            mapped_status, 0), "expert cache status pointer")) {
        if (mapped_status) (void)cudaFreeHost(mapped_status);
        std::free(cache->snapshots);
        std::free(cache->slots);
        delete cache;
        return 0;
    }
    cache->host_status = static_cast<volatile int *>(mapped_status);
    *cache->host_status = 0;
    for (int slot = 0; slot < config->slots; ++slot) {
        cache->slots[slot].expert_id = -1;
        for (int copy = 0; copy < 2; ++copy) {
            ColiGpuExpertBank *bank = &cache->slots[slot].bank[copy];
            if (!cuda_ok(cudaMalloc(
                    reinterpret_cast<void **>(&bank->allocation),
                    cache->bank_bytes), "expert cache bank allocation") ||
                !cuda_ok(cudaEventCreateWithFlags(
                    &bank->use_done, cudaEventDisableTiming),
                    "expert cache bank use event")) {
                coli_gpu_expert_cache_destroy(cache);
                return 0;
            }
            gpu_expert_bank_layout(cache, bank);
            ctx->telemetry.device_allocations++;
        }
    }
    *out = cache;
    return 1;
}

extern "C" void coli_gpu_expert_cache_destroy(
    ColiGpuExpertCache *cache) {
    if (!cache) return;
    if (cache->ctx && select_device_ordinal(cache->ctx->device)) {
        coli_gpu_context_sync(cache->ctx);
        (void)cudaStreamSynchronize(cache->ctx->upload_stream);
    }
    if (cache->slots) {
        for (int slot = 0; slot < cache->config.slots; ++slot) {
            for (int copy = 0; copy < 2; ++copy) {
                ColiGpuExpertBank *bank = &cache->slots[slot].bank[copy];
                if (bank->use_done) (void)cudaEventDestroy(bank->use_done);
                if (bank->allocation) (void)cudaFree(bank->allocation);
            }
        }
    }
    if (cache->host_status)
        (void)cudaFreeHost(const_cast<int *>(cache->host_status));
    while (cache->transfers) {
        ColiGpuExpertTransfer *transfer = cache->transfers;
        cache->transfers = transfer->next;
        if (transfer->ready) (void)cudaEventDestroy(transfer->ready);
        if (transfer->host_staging)
            (void)cudaFreeHost(transfer->host_staging);
        std::free(transfer);
    }
    std::free(cache->slots);
    std::free(cache->snapshots);
    delete cache;
}

__global__ static void gpu_expert_upload_delay_kernel(
    unsigned long long cycles) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

static int gpu_expert_stream_wait_event(
    ColiGpuExpertCache *cache, cudaStream_t stream, cudaEvent_t event,
    int inject_error, const char *label) {
    cache->ctx->telemetry.expert_event_wait_calls++;
    if (inject_error) {
        cache->ctx->telemetry.expert_event_wait_failures++;
        return 0;
    }
    cudaError_t status = cudaStreamWaitEvent(stream, event, 0);
    if (status != cudaSuccess) {
        cache->ctx->telemetry.expert_event_wait_failures++;
        (void)cuda_ok(status, label);
        return 0;
    }
    return 1;
}

static void gpu_expert_retire_transfers_locked(ColiGpuExpertCache *cache) {
    ColiGpuExpertTransfer **link = &cache->transfers;
    while (*link) {
        ColiGpuExpertTransfer *transfer = *link;
        if (transfer->published) {
            link = &transfer->next;
            continue;
        }
        cudaError_t status = cudaEventQuery(transfer->ready);
        if (status == cudaErrorNotReady) {
            link = &transfer->next;
            continue;
        }
        if (status != cudaSuccess) {
            cache->healthy = 0;
            return;
        }
        *link = transfer->next;
        (void)cudaEventDestroy(transfer->ready);
        (void)cudaFreeHost(transfer->host_staging);
        std::free(transfer);
    }
}

static int gpu_expert_record_retirement(
    ColiGpuExpertCache *cache, ColiGpuExpertTransfer *transfer) {
    if (cuda_ok(cudaEventRecord(
            transfer->ready, cache->ctx->upload_stream),
            "expert cache failed-transfer retirement"))
        return 1;
    if (!cuda_ok(cudaStreamSynchronize(cache->ctx->upload_stream),
                 "expert cache failed-transfer drain"))
        cache->healthy = 0;
    return 0;
}

extern "C" int coli_gpu_expert_cache_upload(
    ColiGpuExpertCache *cache, int expert_id, int slot,
    const ColiGpuExpertSource *source, ColiGpuExpertHandle *out) {
    if (out) std::memset(out, 0, sizeof(*out));
    if (!cache || !out || expert_id < 0 ||
        expert_id >= cache->config.experts || slot < 0 ||
        slot >= cache->config.slots ||
        !gpu_expert_source_ok(cache, source) ||
        !cache->ctx->healthy || !select_device_ordinal(cache->ctx->device))
        return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (!cache->healthy) {
        cache->ctx->telemetry.unhealthy_cache_rejections++;
        return 0;
    }
    ColiGpuExpertFaultPoint fault = cache->fault;
    int fault_occurrence = cache->fault_occurrence;
    cache->fault = COLI_GPU_EXPERT_FAULT_NONE;
    cache->fault_occurrence = 0;
    gpu_expert_retire_transfers_locked(cache);
    if (!cache->healthy ||
        fault == COLI_GPU_EXPERT_FAULT_ALLOCATION ||
        (fault == COLI_GPU_EXPERT_FAULT_UPLOAD && fault_occurrence == 0))
        return 0;
    ColiGpuExpertSlot *target = &cache->slots[slot];
    if (target->generation == UINT64_MAX) {
        cache->healthy = 0;
        cache->ctx->telemetry.generation_exhaustions++;
        return 0;
    }
    int bank_index = target->published ? 1 - target->active_bank : 0;
    ColiGpuExpertBank *bank = &target->bank[bank_index];
    if (bank->use_recorded &&
        !gpu_expert_stream_wait_event(
            cache, cache->ctx->upload_stream, bank->use_done, 0,
            "expert cache bank reuse wait"))
        return 0;

    ColiGpuExpertTransfer *transfer =
        static_cast<ColiGpuExpertTransfer *>(
            std::calloc(1, sizeof(*transfer)));
    void *staging = nullptr;
    if (!transfer ||
        !cuda_ok(cudaHostAlloc(&staging, cache->bank_bytes, 0),
                 "expert cache transfer staging allocation") ||
        !cuda_ok(cudaEventCreateWithFlags(
            &transfer->ready, cudaEventDisableTiming),
            "expert cache generation publication event")) {
        if (staging) (void)cudaFreeHost(staging);
        std::free(transfer);
        return 0;
    }
    transfer->host_staging = static_cast<unsigned char *>(staging);
    transfer->next = cache->transfers;
    cache->transfers = transfer;

    size_t gate_data = 0;
    size_t gate_scales = gpu_align256(gate_data + cache->gate_data_bytes);
    size_t up_data = gpu_align256(gate_scales + cache->gate_scale_bytes);
    size_t up_scales = gpu_align256(up_data + cache->gate_data_bytes);
    size_t down_data = gpu_align256(up_scales + cache->gate_scale_bytes);
    size_t down_scales = gpu_align256(down_data + cache->down_data_bytes);
    std::memcpy(transfer->host_staging + gate_data, source->gate.data,
                cache->gate_data_bytes);
    std::memcpy(transfer->host_staging + gate_scales, source->gate.scales,
                cache->gate_scale_bytes);
    std::memcpy(transfer->host_staging + up_data, source->up.data,
                cache->gate_data_bytes);
    std::memcpy(transfer->host_staging + up_scales, source->up.scales,
                cache->gate_scale_bytes);
    std::memcpy(transfer->host_staging + down_data, source->down.data,
                cache->down_data_bytes);
    std::memcpy(transfer->host_staging + down_scales, source->down.scales,
                cache->down_scale_bytes);
    if (cache->test_delay_ms) {
        unsigned long long cycles =
            (unsigned long long)cache->test_delay_ms *
            (unsigned long long)cache->ctx->clock_rate_khz;
        gpu_expert_upload_delay_kernel<<<
            1, 1, 0, cache->ctx->upload_stream>>>(cycles);
        cache->test_delay_ms = 0;
        if (!cuda_ok(cudaGetLastError(), "expert cache upload delay")) {
            gpu_expert_record_retirement(cache, transfer);
            return 0;
        }
    }
    void *destinations[6] = {
        bank->gate_data, bank->gate_scales, bank->up_data, bank->up_scales,
        bank->down_data, bank->down_scales
    };
    const size_t offsets[6] = {
        gate_data, gate_scales, up_data, up_scales, down_data, down_scales
    };
    const size_t bytes[6] = {
        cache->gate_data_bytes, cache->gate_scale_bytes,
        cache->gate_data_bytes, cache->gate_scale_bytes,
        cache->down_data_bytes, cache->down_scale_bytes
    };
    const char *labels[6] = {
        "expert cache gate upload", "expert cache gate scales upload",
        "expert cache up upload", "expert cache up scales upload",
        "expert cache down upload", "expert cache down scales upload"
    };
    for (int copy = 0; copy < 6; ++copy) {
        if (!cuda_ok(cudaMemcpyAsync(
                destinations[copy], transfer->host_staging + offsets[copy],
                bytes[copy], cudaMemcpyHostToDevice,
                cache->ctx->upload_stream), labels[copy])) {
            gpu_expert_record_retirement(cache, transfer);
            return 0;
        }
        cache->ctx->telemetry.h2d_copies++;
        cache->ctx->telemetry.h2d_bytes += bytes[copy];
        cache->ctx->telemetry.expert_upload_bytes += bytes[copy];
        if (fault == COLI_GPU_EXPERT_FAULT_UPLOAD &&
            fault_occurrence == copy + 1) {
            gpu_expert_record_retirement(cache, transfer);
            return 0;
        }
    }
    if (fault == COLI_GPU_EXPERT_FAULT_EVENT_RECORD) {
        gpu_expert_record_retirement(cache, transfer);
        return 0;
    }
    if (!cuda_ok(cudaEventRecord(
            transfer->ready, cache->ctx->upload_stream),
                 "expert cache publication record")) {
        if (!cuda_ok(cudaStreamSynchronize(cache->ctx->upload_stream),
                     "expert cache publication failure drain"))
            cache->healthy = 0;
        return 0;
    }
    if (target->published && target->expert_id != expert_id)
        cache->ctx->telemetry.expert_cache_evictions++;
    uint64_t generation = target->generation + 1;
    target->active_bank = bank_index;
    target->expert_id = expert_id;
    target->generation = generation;
    target->published = 1;
    if (target->publication) target->publication->published = 0;
    target->publication = transfer;
    transfer->published = 1;
    out->expert_id = expert_id;
    out->slot = slot;
    out->generation = generation;
    cache->ctx->telemetry.expert_upload_events++;
    cache->ctx->telemetry.expert_publications++;
    return 1;
}

extern "C" int coli_gpu_expert_cache_lookup(
    ColiGpuExpertCache *cache, int expert_id, ColiGpuExpertHandle *out) {
    if (out) std::memset(out, 0, sizeof(*out));
    if (!cache || !out || expert_id < 0 ||
        expert_id >= cache->config.experts)
        return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (!cache->healthy) {
        cache->ctx->telemetry.unhealthy_cache_rejections++;
        return 0;
    }
    for (int slot = 0; slot < cache->config.slots; ++slot) {
        ColiGpuExpertSlot *candidate = &cache->slots[slot];
        if (candidate->published && candidate->expert_id == expert_id) {
            out->expert_id = expert_id;
            out->slot = slot;
            out->generation = candidate->generation;
            cache->ctx->telemetry.expert_cache_hits++;
            return 1;
        }
    }
    cache->ctx->telemetry.expert_cache_misses++;
    return 0;
}

static int gpu_expert_handle_validate_locked(
    ColiGpuExpertCache *cache, const ColiGpuExpertHandle *handle) {
    if (!cache->healthy) {
        cache->ctx->telemetry.unhealthy_cache_rejections++;
        return 0;
    }
    if (!handle || handle->slot < 0 ||
        handle->slot >= cache->config.slots) {
        cache->ctx->telemetry.expert_handle_range_rejections++;
        return 0;
    }
    const ColiGpuExpertSlot *slot = &cache->slots[handle->slot];
    if (!slot->published) {
        cache->ctx->telemetry.unpublished_slot_rejections++;
        return 0;
    }
    if (slot->generation != handle->generation) {
        cache->ctx->telemetry.stale_generation_rejections++;
        return 0;
    }
    if (handle->expert_id < 0 ||
        handle->expert_id >= cache->config.experts ||
        slot->expert_id != handle->expert_id) {
        cache->ctx->telemetry.wrong_expert_rejections++;
        return 0;
    }
    return 1;
}

extern "C" int coli_gpu_expert_cache_validate(
    ColiGpuExpertCache *cache, const ColiGpuExpertHandle *handle) {
    if (!cache) return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);
    return gpu_expert_handle_validate_locked(cache, handle);
}

extern "C" int coli_gpu_expert_cache_slot_info(
    const ColiGpuExpertCache *cache, int slot, ColiGpuExpertSlotInfo *out) {
    if (!cache || !out || slot < 0 || slot >= cache->config.slots) return 0;
    std::lock_guard<std::mutex> lock(
        const_cast<ColiGpuExpertCache *>(cache)->mutex);
    std::memset(out, 0, sizeof(*out));
    out->slot = slot;
    out->expert_id = cache->slots[slot].expert_id;
    out->generation = cache->slots[slot].generation;
    out->published = cache->slots[slot].published;
    return 1;
}

extern "C" int coli_gpu_expert_cache_inject_fault(
    ColiGpuExpertCache *cache, ColiGpuExpertFaultPoint point) {
    if (!cache || point < COLI_GPU_EXPERT_FAULT_NONE ||
        point > COLI_GPU_EXPERT_FAULT_LAUNCH)
        return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->fault = point;
    cache->fault_occurrence = 0;
    return 1;
}

extern "C" int coli_gpu_expert_cache_inject_fault_at(
    ColiGpuExpertCache *cache, ColiGpuExpertFaultPoint point,
    int occurrence) {
    if (!cache || point < COLI_GPU_EXPERT_FAULT_NONE ||
        point > COLI_GPU_EXPERT_FAULT_LAUNCH || occurrence < 0)
        return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->fault = point;
    cache->fault_occurrence = occurrence;
    return 1;
}

extern "C" int coli_gpu_expert_cache_healthy(
    const ColiGpuExpertCache *cache) {
    if (!cache) return 0;
    std::lock_guard<std::mutex> lock(
        const_cast<ColiGpuExpertCache *>(cache)->mutex);
    return cache->healthy;
}

extern "C" int coli_gpu_expert_cache_test_set_generation(
    ColiGpuExpertCache *cache, int slot, uint64_t generation) {
    if (!cache || slot < 0 || slot >= cache->config.slots ||
        generation == 0)
        return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (!cache->healthy || !cache->slots[slot].published) return 0;
    cache->slots[slot].generation = generation;
    return 1;
}

extern "C" int coli_gpu_expert_cache_test_delay_upload(
    ColiGpuExpertCache *cache, unsigned milliseconds) {
    if (!cache || !milliseconds) return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (!cache->healthy) return 0;
    cache->test_delay_ms = milliseconds;
    return 1;
}

extern "C" int coli_gpu_expert_cache_test_hold_snapshot(
    ColiGpuExpertCache *cache, int hold) {
    if (!cache || (hold != 0 && hold != 1)) return 0;
    cache->test_hold_snapshot.store(hold, std::memory_order_release);
    if (hold)
        cache->test_snapshot_entered.store(0, std::memory_order_release);
    return 1;
}

extern "C" int coli_gpu_expert_cache_test_snapshot_entered(
    const ColiGpuExpertCache *cache) {
    return cache &&
        cache->test_snapshot_entered.load(std::memory_order_acquire);
}

typedef struct {
    const void *data;
    const float *scales;
    int format;
    int columns;
    int groups;
    int group_size;
} ColiGpuMoeWeightView;

__device__ static float gpu_moe_weight(
    ColiGpuMoeWeightView weight, int row, int column) {
    size_t packed_row =
        (size_t)row * gpu_tensor_row_bytes(weight.format, weight.columns);
    float value = weight_at(weight.data, weight.format, packed_row, column);
    if (weight.format == 1 || weight.format == 2)
        value *= weight.scales[row];
    else if (weight.format == 4)
        value *= weight.scales[
            (size_t)row * weight.groups + column / weight.group_size];
    return value;
}

__global__ static void gpu_moe_hidden_kernel(
    float *activated, const float *input,
    ColiGpuMoeWeightView gate, ColiGpuMoeWeightView up,
    int intermediate, float swiglu_limit, const int *status) {
    if (*status) return;
    int out = (int)blockIdx.x;
    if (out >= intermediate) return;
    __shared__ float gate_partial[256];
    __shared__ float up_partial[256];
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (int column = (int)threadIdx.x; column < gate.columns;
         column += (int)blockDim.x) {
        float value = input[column];
        gate_sum += value * gpu_moe_weight(gate, out, column);
        up_sum += value * gpu_moe_weight(up, out, column);
    }
    gate_partial[threadIdx.x] = gate_sum;
    up_partial[threadIdx.x] = up_sum;
    __syncthreads();
    for (int width = (int)blockDim.x / 2; width; width >>= 1) {
        if ((int)threadIdx.x < width) {
            gate_partial[threadIdx.x] += gate_partial[threadIdx.x + width];
            up_partial[threadIdx.x] += up_partial[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        float gate_value = gate_partial[0];
        float up_value = up_partial[0];
        gate_value = fminf(gate_value, swiglu_limit);
        up_value = fminf(fmaxf(up_value, -swiglu_limit), swiglu_limit);
        activated[out] =
            gate_value / (1.0f + expf(-gate_value)) * up_value;
    }
}

__global__ static void gpu_moe_down_kernel(
    float *output, const float *activated,
    ColiGpuMoeWeightView down, int hidden, const int *status) {
    if (*status) return;
    int out = (int)blockIdx.x;
    if (out >= hidden) return;
    __shared__ float partial[256];
    float sum = 0.0f;
    for (int column = (int)threadIdx.x; column < down.columns;
         column += (int)blockDim.x)
        sum += activated[column] * gpu_moe_weight(down, out, column);
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = (int)blockDim.x / 2; width; width >>= 1) {
        if ((int)threadIdx.x < width)
            partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    if (!threadIdx.x) output[out] = partial[0];
}

__global__ static void gpu_moe_accumulate_kernel(
    float *output, const float *expert, const float *weight, int hidden,
    const int *status) {
    if (*status) return;
    for (int column = (int)threadIdx.x; column < hidden;
         column += (int)blockDim.x)
        output[column] += *weight * expert[column];
}

__global__ static void gpu_moe_clear_kernel(
    float *output, size_t count, const int *status) {
    if (*status) return;
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride) output[index] = 0.0f;
}

__global__ static void gpu_moe_validate_kernel(
    const float *values, size_t count, int *status) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; index < count; index += stride)
        if (!isfinite(values[index])) atomicMax(status, 1);
}

static ColiGpuMoeWeightView gpu_moe_tensor_view(
    const ColiGpuTensor *tensor) {
    ColiGpuMoeWeightView view = {
        tensor->data, tensor->scales, tensor->format, tensor->columns,
        tensor->groups, tensor->group_size
    };
    return view;
}

static ColiGpuMoeWeightView gpu_moe_bank_view(
    const void *data, const float *scales, int columns, int group_size) {
    ColiGpuMoeWeightView view = {
        data, scales, 4, columns,
        (columns + group_size - 1) / group_size, group_size
    };
    return view;
}

static void gpu_moe_launch_expert(
    ColiGpuContext *ctx, float *activated, float *temporary,
    const float *input, ColiGpuMoeWeightView gate,
    ColiGpuMoeWeightView up, ColiGpuMoeWeightView down,
    int hidden, int intermediate, float swiglu_limit, int *status) {
    gpu_moe_hidden_kernel<<<intermediate, 256, 0, ctx->stream>>>(
        activated, input, gate, up, intermediate, swiglu_limit, status);
    gpu_moe_down_kernel<<<hidden, 256, 0, ctx->stream>>>(
        temporary, activated, down, hidden, status);
    ctx->telemetry.moe_compute_launches += 2;
}

extern "C" size_t coli_gpu_moe_scratch_bytes(
    const ColiGpuExpertCacheConfig *config) {
    if (!config || config->hidden < 1 || config->intermediate < 1 ||
        !std::isfinite(config->swiglu_limit) ||
        config->swiglu_limit <= 0.0f ||
        (size_t)config->hidden >
            SIZE_MAX / sizeof(float) - (size_t)config->intermediate)
        return 0;
    return ((size_t)config->hidden + config->intermediate) * sizeof(float);
}

extern "C" int coli_gpu_expert_primitive(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t scratch_offset, ColiGpuExpertCache *cache,
    const ColiGpuMoeSharedWeights *weights, int rows) {
    if (!arena || !cache || cache->ctx != arena->ctx || !weights ||
        rows < 1 || rows > cache->config.max_rows ||
        !std::isfinite(cache->config.swiglu_limit) ||
        cache->config.swiglu_limit <= 0.0f || !cache->ctx->healthy ||
        !select_device_ordinal(cache->ctx->device))
        return 0;
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        if (!cache->healthy) {
            cache->ctx->telemetry.unhealthy_cache_rejections++;
            return 0;
        }
    }
    const int hidden = cache->config.hidden;
    const int intermediate = cache->config.intermediate;
    const ColiGpuTensor *all[3] = {
        weights->gate, weights->up, weights->down
    };
    if (!all[0] || !all[1] || !all[2] ||
        all[0]->ctx != cache->ctx || all[1]->ctx != cache->ctx ||
        all[2]->ctx != cache->ctx ||
        all[0]->rows != intermediate || all[1]->rows != intermediate ||
        all[0]->columns != hidden || all[1]->columns != hidden ||
        all[2]->rows != hidden || all[2]->columns != intermediate)
        return 0;
    for (int index = 0; index < 3; ++index)
        if (all[index]->format != 0 && all[index]->format != 1 &&
            all[index]->format != 2 && all[index]->format != 4)
            return 0;
    size_t activation_bytes = (size_t)rows * hidden * sizeof(float);
    size_t scratch_bytes = coli_gpu_moe_scratch_bytes(&cache->config);
    if (!gpu_range_ok(arena, input_offset, activation_bytes) ||
        !gpu_range_ok(arena, output_offset, activation_bytes) ||
        !gpu_range_ok(arena, scratch_offset, scratch_bytes) ||
        gpu_ranges_overlap(input_offset, activation_bytes,
                           output_offset, activation_bytes) ||
        gpu_ranges_overlap(input_offset, activation_bytes,
                           scratch_offset, scratch_bytes) ||
        gpu_ranges_overlap(output_offset, activation_bytes,
                           scratch_offset, scratch_bytes))
        return 0;
    float *input =
        reinterpret_cast<float *>(arena->data + input_offset);
    float *output =
        reinterpret_cast<float *>(arena->data + output_offset);
    float *activated =
        reinterpret_cast<float *>(arena->data + scratch_offset);
    *cache->host_status = 0;
    int blocks = (int)((activation_bytes / sizeof(float) + 255u) / 256u);
    if (blocks > 65535) blocks = 65535;
    gpu_moe_validate_kernel<<<blocks, 256, 0, cache->ctx->stream>>>(
        input, activation_bytes / sizeof(float), cache->device_status);
    cache->ctx->telemetry.moe_compute_launches++;
    if (!cuda_ok(cudaGetLastError(), "expert primitive input validation") ||
        !coli_gpu_context_sync(cache->ctx) || *cache->host_status)
        return 0;
    ColiGpuMoeWeightView gate = gpu_moe_tensor_view(weights->gate);
    ColiGpuMoeWeightView up = gpu_moe_tensor_view(weights->up);
    ColiGpuMoeWeightView down = gpu_moe_tensor_view(weights->down);
    for (int row = 0; row < rows; ++row)
        gpu_moe_launch_expert(
            cache->ctx, activated, output + (size_t)row * hidden,
            input + (size_t)row * hidden, gate, up, down, hidden,
            intermediate, cache->config.swiglu_limit, cache->device_status);
    *cache->host_status = 0;
    gpu_moe_validate_kernel<<<blocks, 256, 0, cache->ctx->stream>>>(
        output, activation_bytes / sizeof(float), cache->device_status);
    cache->ctx->telemetry.moe_compute_launches++;
    return cuda_ok(cudaGetLastError(), "expert primitive launch") &&
           coli_gpu_context_sync(cache->ctx) && !*cache->host_status;
}

extern "C" int coli_gpu_moe_site(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t scratch_offset, ColiGpuRouter *router,
    ColiGpuExpertCache *cache, const ColiGpuMoeSharedWeights *shared,
    const ColiGpuExpertHandle *handles, size_t handle_count, int rows) {
    if (!arena || !cache || cache->ctx != arena->ctx ||
        rows < 1 || rows > cache->config.max_rows ||
        (!router && !shared) || !cache->ctx->healthy ||
        !select_device_ordinal(cache->ctx->device))
        return 0;
    const int hidden = cache->config.hidden;
    const int intermediate = cache->config.intermediate;
    size_t activation_bytes = (size_t)rows * hidden * sizeof(float);
    size_t scratch_bytes = coli_gpu_moe_scratch_bytes(&cache->config);
    if (!gpu_range_ok(arena, input_offset, activation_bytes) ||
        !gpu_range_ok(arena, output_offset, activation_bytes) ||
        !gpu_range_ok(arena, scratch_offset, scratch_bytes) ||
        gpu_ranges_overlap(input_offset, activation_bytes,
                           output_offset, activation_bytes) ||
        gpu_ranges_overlap(input_offset, activation_bytes,
                           scratch_offset, scratch_bytes) ||
        gpu_ranges_overlap(output_offset, activation_bytes,
                           scratch_offset, scratch_bytes))
        return 0;
    if (shared) {
        const ColiGpuTensor *all[3] = {
            shared->gate, shared->up, shared->down
        };
        if (!all[0] || !all[1] || !all[2] ||
            all[0]->ctx != cache->ctx || all[1]->ctx != cache->ctx ||
            all[2]->ctx != cache->ctx ||
            all[0]->rows != intermediate || all[1]->rows != intermediate ||
            all[0]->columns != hidden || all[1]->columns != hidden ||
            all[2]->rows != hidden || all[2]->columns != intermediate)
            return 0;
        for (int index = 0; index < 3; ++index)
            if (all[index]->format != 0 && all[index]->format != 1 &&
                all[index]->format != 2 && all[index]->format != 4)
                return 0;
    }
    size_t required_handles = 0;
    if (router) {
        if (router->ctx != cache->ctx || router->rows != rows ||
            router->config.hidden != hidden ||
            router->config.experts != cache->config.experts)
            return 0;
        required_handles = (size_t)rows * router->config.topk;
        if (!handles || handle_count != required_handles ||
            required_handles > cache->snapshot_capacity)
            return 0;
    } else if (handles || handle_count) {
        return 0;
    }

    float *input = reinterpret_cast<float *>(
        arena->data + input_offset);
    float *output = reinterpret_cast<float *>(
        arena->data + output_offset);
    float *activated = reinterpret_cast<float *>(
        arena->data + scratch_offset);
    float *temporary = activated + intermediate;
    int blocks = (int)((activation_bytes / sizeof(float) + 255u) / 256u);
    if (blocks > 65535) blocks = 65535;
    std::unique_lock<std::mutex> lock(cache->mutex);
    if (!cache->healthy) {
        cache->ctx->telemetry.unhealthy_cache_rejections++;
        return 0;
    }
    ColiGpuExpertFaultPoint fault = cache->fault;
    int fault_occurrence = cache->fault_occurrence;
    cache->fault = COLI_GPU_EXPERT_FAULT_NONE;
    cache->fault_occurrence = 0;
    if (router) {
        for (size_t index = 0; index < required_handles; ++index) {
            if (handles[index].expert_id != router->host_selected[index]) {
                cache->ctx->telemetry.wrong_expert_rejections++;
                return 0;
            }
            if (!gpu_expert_handle_validate_locked(cache, &handles[index]))
                return 0;
            ColiGpuExpertSlot *slot =
                &cache->slots[handles[index].slot];
            cache->snapshots[index].bank =
                &slot->bank[slot->active_bank];
            cache->snapshots[index].publication = slot->publication;
        }
        if (cache->test_hold_snapshot.load(std::memory_order_acquire)) {
            cache->test_snapshot_entered.store(1, std::memory_order_release);
            while (cache->test_hold_snapshot.load(std::memory_order_acquire))
                std::this_thread::yield();
        }
        for (size_t index = 0; index < required_handles; ++index) {
            const ColiGpuExpertTransfer *publication =
                cache->snapshots[index].publication;
            if (!publication ||
                !gpu_expert_stream_wait_event(
                    cache, cache->ctx->stream, publication->ready,
                    fault == COLI_GPU_EXPERT_FAULT_EVENT_WAIT &&
                    (fault_occurrence == 0 ||
                     fault_occurrence == (int)index + 1),
                    "expert cache wait before use")) {
                cache->test_snapshot_entered.store(
                    0, std::memory_order_release);
                return 0;
            }
        }
    }
    if (fault == COLI_GPU_EXPERT_FAULT_LAUNCH) {
        cache->test_snapshot_entered.store(0, std::memory_order_release);
        return 0;
    }

    *cache->host_status = 0;
    gpu_moe_validate_kernel<<<blocks, 256, 0, cache->ctx->stream>>>(
        input, activation_bytes / sizeof(float), cache->device_status);
    cache->ctx->telemetry.moe_compute_launches++;
    if (!shared) {
        gpu_moe_clear_kernel<<<blocks, 256, 0, cache->ctx->stream>>>(
            output, activation_bytes / sizeof(float),
            cache->device_status);
        cache->ctx->telemetry.moe_compute_launches++;
    }
    if (shared) {
        ColiGpuMoeWeightView gate = gpu_moe_tensor_view(shared->gate);
        ColiGpuMoeWeightView up = gpu_moe_tensor_view(shared->up);
        ColiGpuMoeWeightView down = gpu_moe_tensor_view(shared->down);
        for (int row = 0; row < rows; ++row)
            gpu_moe_launch_expert(
                cache->ctx, activated, output + (size_t)row * hidden,
                input + (size_t)row * hidden, gate, up, down, hidden,
                intermediate, cache->config.swiglu_limit,
                cache->device_status);
    }
    if (router) {
        for (size_t index = 0; index < required_handles; ++index) {
            int row = (int)(index / (size_t)router->config.topk);
            ColiGpuExpertBank *bank = cache->snapshots[index].bank;
            gpu_moe_launch_expert(
                cache->ctx, activated, temporary,
                input + (size_t)row * hidden,
                gpu_moe_bank_view(bank->gate_data, bank->gate_scales,
                                  hidden, cache->config.group_size),
                gpu_moe_bank_view(bank->up_data, bank->up_scales,
                                  hidden, cache->config.group_size),
                gpu_moe_bank_view(bank->down_data, bank->down_scales,
                                  intermediate, cache->config.group_size),
                hidden, intermediate, cache->config.swiglu_limit,
                cache->device_status);
            gpu_moe_accumulate_kernel<<<1, 256, 0, cache->ctx->stream>>>(
                output + (size_t)row * hidden, temporary,
                router->weights + index, hidden, cache->device_status);
            cache->ctx->telemetry.moe_compute_launches++;
            if (!cuda_ok(cudaEventRecord(
                    bank->use_done, cache->ctx->stream),
                    "expert cache bank use record")) {
                cache->healthy = 0;
                return 0;
            }
            bank->use_recorded = 1;
        }
        cache->test_snapshot_entered.store(0, std::memory_order_release);
    }
    lock.unlock();
    gpu_moe_validate_kernel<<<blocks, 256, 0, cache->ctx->stream>>>(
        output, activation_bytes / sizeof(float), cache->device_status);
    cache->ctx->telemetry.moe_compute_launches++;
    return cuda_ok(cudaGetLastError(), "resident MoE launch") &&
           coli_gpu_context_sync(cache->ctx) && !*cache->host_status;
}

static int gpu_kda_config_ok(const ColiGpuKdaConfig *config) {
    if (!config || config->heads <= 0 || config->head_dim <= 0 ||
        config->head_dim > 512 || config->kernel <= 0 ||
        config->kernel > 8 || config->max_rows <= 0 ||
        config->max_context <= 0 ||
        config->recurrent_norm_eps < 0.0f ||
        config->output_norm_eps < 0.0f ||
        !std::isfinite(config->recurrent_norm_eps) ||
        !std::isfinite(config->output_norm_eps) ||
        !std::isfinite(config->gate_lower_bound))
        return 0;
    size_t heads = (size_t)config->heads;
    size_t dim = (size_t)config->head_dim;
    if (heads > SIZE_MAX / dim) return 0;
    size_t projection = heads * dim;
    /* These products are narrowed to int for tensor geometry, loop bounds,
     * launch counts, and kernel arguments. Reject them before any allocation
     * or launch rather than relying on a 64-bit byte-range check. */
    return projection <= (size_t)INT_MAX / 3u &&
           (size_t)config->max_rows <= (size_t)INT_MAX / projection &&
           (size_t)config->max_rows <=
               (size_t)INT_MAX / (size_t)config->heads;
}

extern "C" size_t coli_gpu_kda_state_bytes(
    const ColiGpuKdaConfig *config) {
    if (!gpu_kda_config_ok(config)) return 0;
    size_t values = (size_t)config->heads * config->head_dim;
    if (values > SIZE_MAX / (size_t)config->head_dim ||
        values * (size_t)config->head_dim > SIZE_MAX / sizeof(float))
        return 0;
    return values * (size_t)config->head_dim * sizeof(float);
}

extern "C" size_t coli_gpu_kda_window_bytes(
    const ColiGpuKdaConfig *config) {
    if (!gpu_kda_config_ok(config)) return 0;
    size_t values = (size_t)config->heads * config->head_dim;
    if (values > SIZE_MAX / 3u ||
        values * 3u > SIZE_MAX / (size_t)config->kernel ||
        values * 3u * (size_t)config->kernel > SIZE_MAX / sizeof(float))
        return 0;
    return values * 3u * (size_t)config->kernel * sizeof(float);
}

extern "C" size_t coli_gpu_kda_scratch_bytes(
    const ColiGpuKdaConfig *config, int rows, int hidden) {
    if (!gpu_kda_config_ok(config) || rows < 1 || rows > config->max_rows ||
        hidden < 1)
        return 0;
    size_t projection = (size_t)config->heads * config->head_dim;
    if (projection > (SIZE_MAX - 2u * (size_t)config->head_dim -
                      (size_t)config->heads) / 7u)
        return 0;
    size_t per_row = 7u * projection + 2u * (size_t)config->head_dim +
                     (size_t)config->heads;
    if ((size_t)rows > SIZE_MAX / per_row ||
        (size_t)rows * per_row > SIZE_MAX / sizeof(float))
        return 0;
    return (size_t)rows * per_row * sizeof(float);
}

extern "C" int coli_gpu_kda_state_create(
    ColiGpuKdaState **out, ColiGpuContext *ctx, ColiGpuArena *arena,
    size_t state_offset, size_t window_offset,
    const ColiGpuKdaConfig *config) {
    if (!out) return 0;
    *out = nullptr;
    size_t state_bytes = coli_gpu_kda_state_bytes(config);
    size_t window_bytes = coli_gpu_kda_window_bytes(config);
    if (!ctx || !arena || arena->ctx != ctx || !ctx->healthy ||
        !state_bytes || !window_bytes ||
        !gpu_range_ok(arena, state_offset, state_bytes) ||
        !gpu_range_ok(arena, window_offset, window_bytes) ||
        (state_offset < window_offset + window_bytes &&
         window_offset < state_offset + state_bytes))
        return 0;
    ColiGpuKdaState *state =
        static_cast<ColiGpuKdaState *>(std::calloc(1, sizeof(*state)));
    if (!state) return 0;
    state->ctx = ctx;
    state->arena = arena;
    state->state_offset = state_offset;
    state->window_offset = window_offset;
    state->config = *config;
    *out = state;
    return 1;
}

extern "C" void coli_gpu_kda_state_destroy(ColiGpuKdaState *state) {
    std::free(state);
}

extern "C" int coli_gpu_kda_state_reset(ColiGpuKdaState *state) {
    if (!state || !state->ctx || !state->arena ||
        state->arena->ctx != state->ctx || !state->ctx->healthy ||
        !select_device_ordinal(state->ctx->device))
        return 0;
    size_t state_bytes = coli_gpu_kda_state_bytes(&state->config);
    size_t window_bytes = coli_gpu_kda_window_bytes(&state->config);
    if (!cuda_ok(cudaMemsetAsync(state->arena->data + state->state_offset, 0,
                                 state_bytes, state->ctx->stream),
                 "resident KDA state reset") ||
        !cuda_ok(cudaMemsetAsync(state->arena->data + state->window_offset, 0,
                                 window_bytes, state->ctx->stream),
                 "resident KDA window reset"))
        return 0;
    state->position = 0;
    return 1;
}

extern "C" int coli_gpu_kda_state_download(
    ColiGpuKdaState *state, float *matrix, size_t matrix_floats,
    float *window, size_t window_floats) {
    if (!state || !matrix || !window) return 0;
    size_t state_bytes = coli_gpu_kda_state_bytes(&state->config);
    size_t window_bytes = coli_gpu_kda_window_bytes(&state->config);
    if (matrix_floats != state_bytes / sizeof(float) ||
        window_floats != window_bytes / sizeof(float))
        return 0;
    return coli_gpu_arena_download(state->arena, state->state_offset,
                                   matrix, state_bytes) &&
           coli_gpu_arena_download(state->arena, state->window_offset,
                                   window, window_bytes);
}

__global__ static void gpu_kda_transform_decay_kernel(
    float *decay, const float *dt_bias, const float *a_log,
    int count, int projection, int dim, float lower_bound) {
    int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (index >= count) return;
    int channel = index % projection;
    int head = channel / dim;
    decay[index] = lower_bound * gpu_stable_sigmoid(
        expf(a_log[head]) * (decay[index] + dt_bias[channel]));
}

__global__ static void gpu_kda_sigmoid_kernel(float *values, int count) {
    int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (index < count) values[index] = gpu_stable_sigmoid(values[index]);
}

/* qkv is projection-major [3, rows, heads * dim]. One block owns one head,
 * and one lane deliberately performs that head's token sequence in reference
 * loop order. Heads are independent, so this is deterministic without
 * atomics while retaining head-level parallelism. */
__global__ static void gpu_kda_recurrent_kernel(
    float *output, float *qkv, const float *decay, const float *beta,
    float *state, float *window, const float *conv,
    int rows, int heads, int dim, int kernel, float norm_eps) {
    int head = (int)blockIdx.x;
    if (head >= heads || threadIdx.x != 0) return;
    int projection = heads * dim;
    extern __shared__ float memory[];
    float query_scale = 1.0f / sqrtf((float)dim);
    float *matrix = state + (size_t)head * dim * dim;
    for (int row = 0; row < rows; ++row) {
        for (int part = 0; part < 3; ++part)
            for (int d = 0; d < dim; ++d) {
                int channel = part * projection + head * dim + d;
                float *history = window + (size_t)channel * kernel;
                for (int tap = 0; tap + 1 < kernel; ++tap)
                    history[tap] = history[tap + 1];
                float *slot = qkv + ((size_t)part * rows + row) * projection +
                              head * dim + d;
                history[kernel - 1] = *slot;
                float sum = 0.0f;
                const float *taps = conv + (size_t)channel * kernel;
                for (int tap = 0; tap < kernel; ++tap)
                    sum += taps[tap] * history[tap];
                *slot = sum / (1.0f + expf(-sum));
            }
        const float *query = qkv + (size_t)row * projection + head * dim;
        const float *key = qkv + ((size_t)rows + row) * projection +
                           head * dim;
        const float *value = qkv + ((size_t)2 * rows + row) * projection +
                             head * dim;
        float query_square = norm_eps;
        float key_square = norm_eps;
        for (int d = 0; d < dim; ++d) {
            query_square += query[d] * query[d];
            key_square += key[d] * key[d];
        }
        float query_norm = query_scale / sqrtf(query_square);
        float key_norm = 1.0f / sqrtf(key_square);
        for (int vd = 0; vd < dim; ++vd) memory[vd] = 0.0f;
        for (int kd = 0; kd < dim; ++kd) {
            float *matrix_row = matrix + (size_t)kd * dim;
            float alpha = expf(decay[(size_t)row * projection +
                                     head * dim + kd]);
            float scaled_key = key[kd] * key_norm;
            for (int vd = 0; vd < dim; ++vd) {
                matrix_row[vd] *= alpha;
                memory[vd] += scaled_key * matrix_row[vd];
            }
        }
        float *result = output + (size_t)row * projection + head * dim;
        for (int vd = 0; vd < dim; ++vd) result[vd] = 0.0f;
        for (int kd = 0; kd < dim; ++kd) {
            float *matrix_row = matrix + (size_t)kd * dim;
            float scaled_key = key[kd] * key_norm;
            float scaled_query = query[kd] * query_norm;
            for (int vd = 0; vd < dim; ++vd) {
                matrix_row[vd] += scaled_key * (value[vd] - memory[vd]) *
                                  beta[(size_t)row * heads + head];
                result[vd] += scaled_query * matrix_row[vd];
            }
        }
    }
}

static int gpu_kda_state_call_ok(ColiGpuArena *arena,
                                 ColiGpuKdaState *state,
                                 int rows, int start_position) {
    return arena && state && state->arena == arena &&
           state->ctx == arena->ctx && rows > 0 &&
           rows <= state->config.max_rows && start_position >= 0 &&
           start_position == state->position &&
           start_position <= state->config.max_context - rows &&
           arena->ctx->healthy;
}

typedef struct {
    size_t offset;
    size_t bytes;
} ColiGpuKdaRange;

static int gpu_kda_ranges_overlap(ColiGpuKdaRange a, ColiGpuKdaRange b) {
    if (a.offset <= b.offset) return b.offset - a.offset < a.bytes;
    return a.offset - b.offset < b.bytes;
}

/* All arena ranges participating in one generic KDA call are exclusive.
 * This intentionally rejects aliases that might happen to be safe for one
 * current launch order: qkv, scratch, state, and windows are mutated in place,
 * and the API must not make correctness depend on undocumented ordering. */
static int gpu_kda_ranges_disjoint(const ColiGpuKdaState *state,
                                   const ColiGpuKdaRange *ranges,
                                   int range_count) {
    if (!state || !ranges || range_count < 1) return 0;
    ColiGpuKdaRange persistent[2] = {
        {state->state_offset, coli_gpu_kda_state_bytes(&state->config)},
        {state->window_offset, coli_gpu_kda_window_bytes(&state->config)}
    };
    for (int i = 0; i < range_count; ++i) {
        if (!ranges[i].bytes) return 0;
        for (int p = 0; p < 2; ++p)
            if (gpu_kda_ranges_overlap(ranges[i], persistent[p])) return 0;
        for (int j = 0; j < i; ++j)
            if (gpu_kda_ranges_overlap(ranges[i], ranges[j])) return 0;
    }
    return 1;
}

extern "C" int coli_gpu_kda_recurrent(
    ColiGpuArena *arena, size_t output_offset, size_t qkv_offset,
    size_t decay_offset, size_t beta_offset, ColiGpuKdaState *state,
    const ColiGpuTensor *conv, int rows, int start_position) {
    if (!gpu_kda_state_call_ok(arena, state, rows, start_position))
        return 0;
    int heads = state->config.heads;
    int dim = state->config.head_dim;
    int projection = heads * dim;
    size_t output_bytes = (size_t)rows * projection * sizeof(float);
    size_t qkv_bytes = output_bytes * 3u;
    size_t decay_bytes = output_bytes;
    size_t beta_bytes = (size_t)rows * heads * sizeof(float);
    ColiGpuKdaRange ranges[] = {
        {output_offset, output_bytes},
        {qkv_offset, qkv_bytes},
        {decay_offset, decay_bytes},
        {beta_offset, beta_bytes}
    };
    if (!gpu_tensor_same_context(arena, conv) ||
        conv->format != 0 || conv->rows != 3 * projection ||
        conv->columns != state->config.kernel ||
        !gpu_range_ok(arena, output_offset, output_bytes) ||
        !gpu_range_ok(arena, qkv_offset, qkv_bytes) ||
        !gpu_range_ok(arena, decay_offset, decay_bytes) ||
        !gpu_range_ok(arena, beta_offset, beta_bytes) ||
        !gpu_kda_ranges_disjoint(
            state, ranges, (int)(sizeof(ranges) / sizeof(ranges[0]))) ||
        !select_device_ordinal(arena->ctx->device))
        return 0;
    gpu_kda_recurrent_kernel<<<heads, 1, (size_t)dim * sizeof(float),
                               arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + output_offset),
        reinterpret_cast<float *>(arena->data + qkv_offset),
        reinterpret_cast<const float *>(arena->data + decay_offset),
        reinterpret_cast<const float *>(arena->data + beta_offset),
        reinterpret_cast<float *>(arena->data + state->state_offset),
        reinterpret_cast<float *>(arena->data + state->window_offset),
        reinterpret_cast<const float *>(conv->data), rows, heads, dim,
        state->config.kernel, state->config.recurrent_norm_eps);
    if (!cuda_ok(cudaGetLastError(), "resident KDA recurrence launch"))
        return 0;
    state->position += rows;
    return 1;
}

__global__ static void gpu_kda_output_norm_gate_kernel(
    float *output, const float *core, const float *gate,
    const float *norm, int heads, int dim, float eps) {
    int row_head = (int)blockIdx.x;
    if (threadIdx.x != 0) return;
    (void)heads;
    int base = row_head * dim;
    float square = 0.0f;
    for (int d = 0; d < dim; ++d)
        square += core[base + d] * core[base + d];
    float inverse = 1.0f / sqrtf(square / dim + eps);
    for (int d = 0; d < dim; ++d)
        output[base + d] = core[base + d] * inverse * norm[d] *
                           gpu_stable_sigmoid(gate[base + d]);
}

static int gpu_kda_weight_shape(const ColiGpuArena *arena,
                                const ColiGpuTensor *tensor,
                                int rows, int columns, int f32_only) {
    return gpu_tensor_same_context(arena, tensor) &&
           tensor->rows == rows && tensor->columns == columns &&
           (!f32_only || tensor->format == 0);
}

extern "C" int coli_gpu_kda_site(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t scratch_offset, ColiGpuKdaState *state,
    const ColiGpuKdaWeights *weights, int rows, int start_position,
    int hidden) {
    if (!weights || !gpu_kda_state_call_ok(arena, state, rows, start_position) ||
        hidden < 1)
        return 0;
    const int heads = state->config.heads;
    const int dim = state->config.head_dim;
    const int projection = heads * dim;
    size_t input_bytes = (size_t)rows * hidden * sizeof(float);
    size_t output_bytes = input_bytes;
    size_t scratch_bytes =
        coli_gpu_kda_scratch_bytes(&state->config, rows, hidden);
    ColiGpuKdaRange ranges[] = {
        {input_offset, input_bytes},
        {output_offset, output_bytes},
        {scratch_offset, scratch_bytes}
    };
    if (!scratch_bytes ||
        !gpu_range_ok(arena, input_offset, input_bytes) ||
        !gpu_range_ok(arena, output_offset, output_bytes) ||
        !gpu_range_ok(arena, scratch_offset, scratch_bytes) ||
        !gpu_kda_ranges_disjoint(
            state, ranges, (int)(sizeof(ranges) / sizeof(ranges[0]))) ||
        !gpu_kda_weight_shape(arena, weights->q_proj, projection, hidden, 0) ||
        !gpu_kda_weight_shape(arena, weights->k_proj, projection, hidden, 0) ||
        !gpu_kda_weight_shape(arena, weights->v_proj, projection, hidden, 0) ||
        !gpu_kda_weight_shape(arena, weights->o_proj, hidden, projection, 0) ||
        !gpu_kda_weight_shape(arena, weights->gate_a_proj, dim, hidden, 0) ||
        !gpu_kda_weight_shape(arena, weights->gate_b_proj, projection, dim, 0) ||
        !gpu_kda_weight_shape(arena, weights->decay_a_proj, dim, hidden, 0) ||
        !gpu_kda_weight_shape(arena, weights->decay_b_proj, projection, dim, 0) ||
        !gpu_kda_weight_shape(arena, weights->beta_proj, heads, hidden, 0) ||
        !gpu_kda_weight_shape(arena, weights->conv, 3 * projection,
                              state->config.kernel, 1) ||
        !gpu_kda_weight_shape(arena, weights->dt_bias, 1, projection, 1) ||
        !gpu_kda_weight_shape(arena, weights->a_log, 1, heads, 1) ||
        !gpu_kda_weight_shape(arena, weights->o_norm, 1, dim, 1))
        return 0;

    size_t cursor = scratch_offset;
    const size_t qkv = cursor;
    cursor += (size_t)rows * 3 * projection * sizeof(float);
    const size_t decay_low = cursor;
    cursor += (size_t)rows * dim * sizeof(float);
    const size_t decay = cursor;
    cursor += (size_t)rows * projection * sizeof(float);
    const size_t beta = cursor;
    cursor += (size_t)rows * heads * sizeof(float);
    const size_t gate_low = cursor;
    cursor += (size_t)rows * dim * sizeof(float);
    const size_t gate = cursor;
    cursor += (size_t)rows * projection * sizeof(float);
    const size_t core = cursor;
    cursor += (size_t)rows * projection * sizeof(float);
    const size_t normed = cursor;

    if (!coli_gpu_projection(arena, qkv, input_offset, weights->q_proj,
                             rows, hidden, projection) ||
        !coli_gpu_projection(arena, qkv + (size_t)rows * projection * sizeof(float),
                             input_offset, weights->k_proj,
                             rows, hidden, projection) ||
        !coli_gpu_projection(arena, qkv + (size_t)2 * rows * projection * sizeof(float),
                             input_offset, weights->v_proj,
                             rows, hidden, projection) ||
        !coli_gpu_projection(arena, decay_low, input_offset,
                             weights->decay_a_proj, rows, hidden, dim) ||
        !coli_gpu_projection(arena, decay, decay_low, weights->decay_b_proj,
                             rows, dim, projection) ||
        !coli_gpu_projection(arena, beta, input_offset, weights->beta_proj,
                             rows, hidden, heads))
        return 0;
    int decay_count = rows * projection;
    int blocks = 1 + (decay_count - 1) / 256;
    gpu_kda_transform_decay_kernel<<<blocks, 256, 0, arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + decay),
        reinterpret_cast<const float *>(weights->dt_bias->data),
        reinterpret_cast<const float *>(weights->a_log->data),
        decay_count, projection, dim, state->config.gate_lower_bound);
    if (!cuda_ok(cudaGetLastError(), "resident KDA decay transform launch"))
        return 0;
    int beta_count = rows * heads;
    gpu_kda_sigmoid_kernel<<<1 + (beta_count - 1) / 256, 256, 0,
                              arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + beta), beta_count);
    if (!cuda_ok(cudaGetLastError(), "resident KDA beta transform launch") ||
        !coli_gpu_kda_recurrent(arena, core, qkv, decay, beta, state,
                                weights->conv, rows, start_position) ||
        !coli_gpu_projection(arena, gate_low, input_offset,
                             weights->gate_a_proj, rows, hidden, dim) ||
        !coli_gpu_projection(arena, gate, gate_low, weights->gate_b_proj,
                             rows, dim, projection))
        return 0;
    gpu_kda_output_norm_gate_kernel<<<rows * heads, 1, 0,
                                      arena->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + normed),
        reinterpret_cast<const float *>(arena->data + core),
        reinterpret_cast<const float *>(arena->data + gate),
        reinterpret_cast<const float *>(weights->o_norm->data),
        heads, dim, state->config.output_norm_eps);
    if (!cuda_ok(cudaGetLastError(), "resident KDA output norm/gate launch"))
        return 0;
    return coli_gpu_projection(arena, output_offset, normed, weights->o_proj,
                               rows, projection, hidden);
}

static int gpu_mla_config_ok(const ColiGpuMlaConfig *config) {
    if (!config || config->hidden < 1 || config->heads < 1 ||
        config->q_lora < 1 || config->kv_lora < 1 ||
        config->qk_nope < 1 || config->qk_rope != 0 ||
        config->value_dim < 1 || config->index_heads < 1 ||
        config->index_dim < 1 || config->index_pool < 1 ||
        config->index_topk < config->index_pool ||
        config->index_topk % config->index_pool ||
        (config->index_select_tail != 0 &&
         config->index_select_tail != 1) ||
        config->max_rows < 1 || config->max_context < 1 ||
        config->max_rows > config->max_context ||
        config->page_tokens < 1 ||
        config->page_tokens > config->max_context ||
        config->rms_norm_eps < 0.0f ||
        config->index_norm_eps < 0.0f ||
        !std::isfinite(config->rms_norm_eps) ||
        !std::isfinite(config->index_norm_eps))
        return 0;
    size_t heads = (size_t)config->heads;
    size_t index_heads = (size_t)config->index_heads;
    return heads <= (size_t)INT_MAX / (size_t)config->qk_nope &&
           heads <= (size_t)INT_MAX / (size_t)config->kv_lora &&
           heads <= (size_t)INT_MAX / (size_t)config->value_dim &&
           index_heads <= (size_t)INT_MAX / (size_t)config->index_dim &&
           (size_t)config->max_context <=
               (size_t)INT_MAX / (size_t)config->kv_lora &&
           (size_t)config->max_context <=
               (size_t)INT_MAX / (size_t)config->index_dim;
}

static int gpu_mla_width(const ColiGpuMlaConfig *config) {
    if (!gpu_mla_config_ok(config)) return 0;
    if (config->index_select_tail &&
        config->index_topk > INT_MAX - config->index_pool + 1)
        return 0;
    return config->index_topk +
           (config->index_select_tail ? config->index_pool - 1 : 0);
}

extern "C" size_t coli_gpu_mla_selected_bytes(
    const ColiGpuMlaConfig *config, int rows) {
    int width = gpu_mla_width(config);
    if (!width || rows < 1 || rows > config->max_rows ||
        (size_t)rows > SIZE_MAX / (size_t)width ||
        (size_t)rows * (size_t)width > SIZE_MAX / sizeof(int))
        return 0;
    return (size_t)rows * (size_t)width * sizeof(int);
}

extern "C" size_t coli_gpu_mla_scratch_bytes(
    const ColiGpuMlaConfig *config) {
    int width = gpu_mla_width(config);
    if (!width) return 0;
    size_t pools = ((size_t)config->max_context +
                    (size_t)config->index_pool - 1u) /
                   (size_t)config->index_pool;
    size_t attention_scores = (size_t)config->heads * (size_t)width;
    size_t score_count =
        pools > attention_scores ? pools : attention_scores;
    size_t values_per_row = (size_t)config->q_lora +
        (size_t)config->heads * (size_t)config->qk_nope +
        (size_t)config->heads * (size_t)config->kv_lora +
        (size_t)config->index_heads * (size_t)config->index_dim +
        (size_t)config->index_heads + score_count +
        (size_t)config->heads * (size_t)config->value_dim;
    if ((size_t)config->max_rows > SIZE_MAX / values_per_row ||
        (size_t)config->max_rows * values_per_row >
            SIZE_MAX / sizeof(float))
        return 0;
    return (size_t)config->max_rows * values_per_row * sizeof(float);
}

static size_t gpu_mla_page_bytes(const ColiGpuMlaConfig *config) {
    return (size_t)config->page_tokens *
           ((size_t)config->kv_lora + 2u * (size_t)config->index_dim) *
           sizeof(float);
}

static int gpu_mla_async_alloc(ColiGpuMlaState *state, void **out,
                               size_t bytes, const char *what) {
    *out = nullptr;
    if (!cuda_ok(cudaMallocAsync(out, bytes, state->ctx->stream), what))
        return 0;
    state->ctx->telemetry.device_allocations++;
    return 1;
}

static int gpu_mla_async_free(ColiGpuMlaState *state, void *pointer) {
    return !pointer ||
           cuda_ok(cudaFreeAsync(pointer, state->ctx->stream),
                   "resident MLA stream-ordered free");
}

static void gpu_mla_host_free(void *pointer) {
    if (!pointer) return;
    cudaError_t error = cudaFreeHost(pointer);
    if (error != cudaSuccess)
        std::fprintf(stderr, "[CUDA] resident MLA host status free: %s\n",
                     cudaGetErrorString(error));
}

static int gpu_mla_alloc_page(ColiGpuMlaState *state,
                              ColiGpuMlaPage *page) {
    std::memset(page, 0, sizeof(*page));
    size_t latent_values =
        (size_t)state->config.page_tokens * state->config.kv_lora;
    size_t index_values =
        (size_t)state->config.page_tokens * state->config.index_dim;
    if (!gpu_mla_async_alloc(
            state, &page->allocation, gpu_mla_page_bytes(&state->config),
            "resident MLA page allocation"))
        return 0;
    page->latent = static_cast<float *>(page->allocation);
    page->index_keys = page->latent + latent_values;
    page->index_gates = page->index_keys + index_values;
    if (!cuda_ok(cudaMemsetAsync(page->allocation, 0,
                                 gpu_mla_page_bytes(&state->config),
                                 state->ctx->stream),
                 "resident MLA page initialize")) {
        gpu_mla_async_free(state, page->allocation);
        std::memset(page, 0, sizeof(*page));
        return 0;
    }
    return 1;
}

__global__ static void gpu_mla_publish_page_kernel(
    ColiGpuMlaPage *table, int index, ColiGpuMlaPage page) {
    if (!blockIdx.x && !threadIdx.x) table[index] = page;
}

static int gpu_mla_fault(ColiGpuMlaState *state,
                         ColiGpuMlaFaultPoint point) {
    if (state->fault_point != point) return 0;
    int occurrence = state->fault_seen++;
    if (occurrence != state->fault_occurrence) return 0;
    state->fault_point = COLI_GPU_MLA_FAULT_NONE;
    return 1;
}

extern "C" int coli_gpu_mla_state_create(
    ColiGpuMlaState **out, ColiGpuContext *ctx,
    const ColiGpuMlaConfig *config) {
    if (!out) return 0;
    *out = nullptr;
    if (!ctx || !ctx->healthy || !gpu_mla_config_ok(config) ||
        !coli_gpu_mla_scratch_bytes(config) ||
        !select_device_ordinal(ctx->device))
        return 0;
    ColiGpuMlaState *state =
        static_cast<ColiGpuMlaState *>(std::calloc(1, sizeof(*state)));
    if (!state) return 0;
    state->ctx = ctx;
    state->config = *config;
    state->max_pages =
        (config->max_context + config->page_tokens - 1) /
        config->page_tokens;
    state->pages = static_cast<ColiGpuMlaPage *>(
        std::calloc((size_t)state->max_pages, sizeof(*state->pages)));
    void *mapped_status = nullptr;
    if (!state->pages ||
        !cuda_ok(cudaHostAlloc(&mapped_status, sizeof(int),
                               cudaHostAllocMapped),
                 "resident MLA mapped status allocation")) {
        std::free(state->pages);
        std::free(state);
        return 0;
    }
    state->host_status = static_cast<volatile int *>(mapped_status);
    if (!cuda_ok(cudaHostGetDevicePointer(
            reinterpret_cast<void **>(&state->device_status),
            mapped_status, 0),
            "resident MLA mapped status device pointer")) {
        gpu_mla_host_free(mapped_status);
        std::free(state->pages);
        std::free(state);
        return 0;
    }
    *state->host_status = COLI_GPU_MLA_STATUS_OK;
    if (!gpu_mla_alloc_page(state, &state->pages[0]) ||
        !gpu_mla_async_alloc(
            state, reinterpret_cast<void **>(&state->device_pages),
            sizeof(ColiGpuMlaPage), "resident MLA page-table allocation")) {
        coli_gpu_mla_state_destroy(state);
        return 0;
    }
    gpu_mla_publish_page_kernel<<<1, 1, 0, ctx->stream>>>(
        state->device_pages, 0, state->pages[0]);
    if (!cuda_ok(cudaGetLastError(), "resident MLA initial page publication")) {
        coli_gpu_mla_state_destroy(state);
        return 0;
    }
    state->page_count = 1;
    state->page_table_capacity = 1;
    state->capacity = config->page_tokens < config->max_context
        ? config->page_tokens : config->max_context;
    *out = state;
    return 1;
}

extern "C" void coli_gpu_mla_state_destroy(ColiGpuMlaState *state) {
    if (!state) return;
    if (state->ctx && select_device_ordinal(state->ctx->device)) {
        for (int page = 0; page < state->page_count; ++page)
            gpu_mla_async_free(state, state->pages[page].allocation);
        gpu_mla_async_free(state, state->device_pages);
        coli_gpu_context_sync(state->ctx);
    }
    if (state->host_status)
        gpu_mla_host_free(const_cast<int *>(state->host_status));
    std::free(state->pages);
    std::free(state);
}

extern "C" int coli_gpu_mla_state_reset(ColiGpuMlaState *state) {
    if (!state || !state->ctx || !state->ctx->healthy ||
        !select_device_ordinal(state->ctx->device))
        return 0;
    for (int page = 0; page < state->page_count; ++page)
        if (!cuda_ok(cudaMemsetAsync(
                state->pages[page].allocation, 0,
                gpu_mla_page_bytes(&state->config), state->ctx->stream),
                "resident MLA page reset"))
            return 0;
    state->length = 0;
    *state->host_status = COLI_GPU_MLA_STATUS_OK;
    return 1;
}

extern "C" int coli_gpu_mla_state_length(const ColiGpuMlaState *state) {
    return state ? state->length : 0;
}

extern "C" int coli_gpu_mla_state_capacity(const ColiGpuMlaState *state) {
    return state ? state->capacity : 0;
}

extern "C" int coli_gpu_mla_state_cache_info(
    const ColiGpuMlaState *state, ColiGpuMlaCacheInfo *out) {
    if (!state || !out) return 0;
    out->logical_length = state->length;
    out->capacity = state->capacity;
    out->page_count = state->page_count;
    out->page_table_capacity = state->page_table_capacity;
    out->payload_copy_bytes = state->payload_copy_bytes;
    return 1;
}

extern "C" int coli_gpu_mla_state_inject_growth_fault(
    ColiGpuMlaState *state, ColiGpuMlaFaultPoint point, int occurrence) {
    if (!state || point < COLI_GPU_MLA_FAULT_NONE ||
        point > COLI_GPU_MLA_FAULT_TABLE_PUBLISH || occurrence < 0)
        return 0;
    state->fault_point = point;
    state->fault_occurrence = occurrence;
    state->fault_seen = 0;
    return 1;
}

extern "C" ColiGpuMlaStatus coli_gpu_mla_state_status(
    const ColiGpuMlaState *state) {
    return state && state->host_status
        ? static_cast<ColiGpuMlaStatus>(*state->host_status)
        : COLI_GPU_MLA_STATUS_NONFINITE_RESULT;
}

extern "C" int coli_gpu_mla_state_launch_info(
    const ColiGpuMlaState *state, ColiGpuMlaLaunchInfo *out) {
    if (!state || !out) return 0;
    *out = state->launch_info;
    return 1;
}

extern "C" int coli_gpu_mla_state_test_corrupt_cache(
    ColiGpuMlaState *state, ColiGpuMlaCacheKind kind,
    int position, int channel, float value) {
    if (!state || position < 0 || position >= state->length ||
        kind < COLI_GPU_MLA_CACHE_LATENT ||
        kind > COLI_GPU_MLA_CACHE_INDEX_GATE ||
        !select_device_ordinal(state->ctx->device))
        return 0;
    ColiGpuMlaPage &page = state->pages[position / state->config.page_tokens];
    int columns = kind == COLI_GPU_MLA_CACHE_LATENT
        ? state->config.kv_lora : state->config.index_dim;
    if (channel < 0 || channel >= columns) return 0;
    float *base = kind == COLI_GPU_MLA_CACHE_LATENT ? page.latent :
                  kind == COLI_GPU_MLA_CACHE_INDEX_KEY
                      ? page.index_keys : page.index_gates;
    float *at = base +
        (size_t)(position % state->config.page_tokens) * columns + channel;
    return cuda_ok(cudaMemcpyAsync(
                       at, &value, sizeof(value), cudaMemcpyHostToDevice,
                       state->ctx->stream),
                   "resident MLA test cache corruption") &&
           coli_gpu_context_sync(state->ctx);
}

static void gpu_mla_record_launch(
    ColiGpuMlaState *state, int blocks, int threads) {
    state->launch_info.kernel_launches++;
    state->launch_info.total_grid_blocks += (uint64_t)blocks;
    if (blocks > state->launch_info.max_grid_blocks)
        state->launch_info.max_grid_blocks = blocks;
    if (threads > state->launch_info.max_block_threads)
        state->launch_info.max_block_threads = threads;
}

static int gpu_mla_reserve(ColiGpuMlaState *state, int required) {
    if (required <= state->capacity) return 1;
    if (required > state->config.max_context) return 0;
    int required_pages =
        (required + state->config.page_tokens - 1) /
        state->config.page_tokens;
    int target_pages = state->page_count;
    while (target_pages < required_pages) {
        if (target_pages > state->max_pages / 2) {
            target_pages = state->max_pages;
            break;
        }
        target_pages *= 2;
    }
    if (target_pages < required_pages) target_pages = required_pages;
    if (gpu_mla_fault(state, COLI_GPU_MLA_FAULT_TABLE_ALLOC))
        return 0;
    ColiGpuMlaPage *new_table = nullptr;
    if (!gpu_mla_async_alloc(
            state, reinterpret_cast<void **>(&new_table),
            (size_t)target_pages * sizeof(*new_table),
            "resident MLA grown page-table allocation"))
        return 0;
    if (gpu_mla_fault(state, COLI_GPU_MLA_FAULT_TABLE_COPY) ||
        !cuda_ok(cudaMemcpyAsync(
            new_table, state->device_pages,
            (size_t)state->page_count * sizeof(*new_table),
            cudaMemcpyDeviceToDevice, state->ctx->stream),
            "resident MLA page-table copy")) {
        gpu_mla_async_free(state, new_table);
        return 0;
    }
    std::vector<ColiGpuMlaPage> pending(
        (size_t)(target_pages - state->page_count));
    int created = 0;
    for (int page = state->page_count; page < target_pages; ++page) {
        if (gpu_mla_fault(state, COLI_GPU_MLA_FAULT_PAGE_ALLOC) ||
            !gpu_mla_alloc_page(state, &pending[(size_t)created]))
            goto rollback;
        created++;
        if (gpu_mla_fault(state, COLI_GPU_MLA_FAULT_PAGE_PUBLISH))
            goto rollback;
        gpu_mla_publish_page_kernel<<<1, 1, 0, state->ctx->stream>>>(
            new_table, page, pending[(size_t)created - 1]);
        if (!cuda_ok(cudaGetLastError(),
                     "resident MLA page publication"))
            goto rollback;
    }
    if (gpu_mla_fault(state, COLI_GPU_MLA_FAULT_TABLE_PUBLISH))
        goto rollback;
    if (!gpu_mla_async_free(state, state->device_pages))
        goto rollback;
    for (int i = 0; i < created; ++i)
        state->pages[state->page_count + i] = pending[(size_t)i];
    state->device_pages = new_table;
    state->page_count = target_pages;
    state->page_table_capacity = target_pages;
    state->capacity = target_pages * state->config.page_tokens;
    if (state->capacity > state->config.max_context)
        state->capacity = state->config.max_context;
    return 1;

rollback:
    for (int i = 0; i < created; ++i)
        gpu_mla_async_free(state, pending[(size_t)i].allocation);
    gpu_mla_async_free(state, new_table);
    return 0;
}

extern "C" int coli_gpu_mla_state_reserve(
    ColiGpuMlaState *state, int required) {
    return state && required >= 0 && gpu_mla_reserve(state, required);
}

extern "C" int coli_gpu_mla_state_download(
    ColiGpuMlaState *state,
    float *latent, size_t latent_floats,
    float *index_keys, size_t index_key_floats,
    float *index_gates, size_t index_gate_floats) {
    if (!state) return 0;
    size_t latent_count =
        (size_t)state->length * state->config.kv_lora;
    size_t index_count =
        (size_t)state->length * state->config.index_dim;
    if (latent_floats != latent_count ||
        index_key_floats != index_count ||
        index_gate_floats != index_count ||
        (latent_count && !latent) || (index_count && (!index_keys || !index_gates)))
        return 0;
    if (!latent_count) return 1;
    if (!select_device_ordinal(state->ctx->device)) return 0;
    for (int page = 0, copied = 0; copied < state->length; ++page) {
        int count = state->length - copied;
        if (count > state->config.page_tokens)
            count = state->config.page_tokens;
        size_t latent_bytes =
            (size_t)count * state->config.kv_lora * sizeof(float);
        size_t index_bytes =
            (size_t)count * state->config.index_dim * sizeof(float);
        if (!cuda_ok(cudaMemcpyAsync(
                latent + (size_t)copied * state->config.kv_lora,
                state->pages[page].latent, latent_bytes,
                cudaMemcpyDeviceToHost, state->ctx->stream),
                "resident MLA latent page download") ||
            !cuda_ok(cudaMemcpyAsync(
                index_keys + (size_t)copied * state->config.index_dim,
                state->pages[page].index_keys, index_bytes,
                cudaMemcpyDeviceToHost, state->ctx->stream),
                "resident MLA index-key page download") ||
            !cuda_ok(cudaMemcpyAsync(
                index_gates + (size_t)copied * state->config.index_dim,
                state->pages[page].index_gates, index_bytes,
                cudaMemcpyDeviceToHost, state->ctx->stream),
                "resident MLA index-gate page download"))
            return 0;
        state->ctx->telemetry.d2h_copies += 3;
        state->ctx->telemetry.d2h_bytes +=
            latent_bytes + 2u * index_bytes;
        copied += count;
    }
    return coli_gpu_context_sync(state->ctx);
}

typedef struct {
    const void *data;
    const float *scales;
    int format;
    int columns;
    int groups;
    int group_size;
} ColiGpuMlaWeightView;

__device__ static float gpu_mla_weight(
    ColiGpuMlaWeightView weight, int row, int column) {
    size_t row_offset =
        (size_t)row * gpu_tensor_row_bytes(weight.format, weight.columns);
    float value = weight_at(weight.data, weight.format, row_offset, column);
    if (!weight.format) return value;
    if (weight.format == 4)
        return value * weight.scales[(size_t)row * weight.groups +
                                     column / weight.group_size];
    return value * weight.scales[row];
}

__device__ static void gpu_mla_matvec(
    float *out, ColiGpuMlaWeightView weight, const float *input, int rows) {
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < weight.columns; ++column)
            sum += gpu_mla_weight(weight, row, column) * input[column];
        out[row] = sum;
    }
}

__global__ static void gpu_mla_validate_finite_kernel(
    const float *values, size_t count, int *status, int failure_status) {
    size_t at = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; at < count; at += stride)
        if (!isfinite(values[at])) atomicMax(status, failure_status);
}

typedef struct {
    ColiGpuMlaWeightView qa, qan, qb, kva, kvan, kvbk, kvbv, output;
    ColiGpuMlaWeightView iwq, iwk, iwp, iknw, iknb, ape, gate;
} ColiGpuMlaKernelWeights;

__device__ static float *gpu_mla_page_row(
    ColiGpuMlaPage *pages, int page_tokens, int position,
    int columns, int member) {
    ColiGpuMlaPage &page = pages[position / page_tokens];
    float *base = member == 0 ? page.latent :
                  member == 1 ? page.index_keys : page.index_gates;
    return base + (size_t)(position % page_tokens) * columns;
}

__global__ static void gpu_mla_site_kernel(
    float *output, const float *input, float *scratch, int *selected,
    ColiGpuMlaPage *pages, int *status,
    ColiGpuMlaConfig c, ColiGpuMlaKernelWeights w,
    int rows, int start, int width) {
    if (blockIdx.x || threadIdx.x) return;
    if (*status != COLI_GPU_MLA_STATUS_OK) return;
    float *qn = scratch;
    float *query = qn + c.q_lora;
    float *absorbed = query + (size_t)c.heads * c.qk_nope;
    float *iq = absorbed + (size_t)c.heads * c.kv_lora;
    float *head_w = iq + (size_t)c.index_heads * c.index_dim;
    float *pooled_index = head_w + c.index_heads;
    float *scores = pooled_index + c.index_dim;
    size_t pools_max =
        ((size_t)c.max_context + c.index_pool - 1u) / c.index_pool;
    size_t score_count = pools_max > (size_t)width ? pools_max : (size_t)width;
    float *pooled = scores + score_count;
    float *context = pooled + c.kv_lora;
    const float attention_scale = 1.0f / sqrtf((float)c.qk_nope);
    const float index_scale = 1.0f / sqrtf((float)c.index_dim);
    const float head_scale = 1.0f / sqrtf((float)c.index_heads);

    for (int token = 0; token < rows; ++token) {
        int absolute = start + token;
        int seen = absolute + 1;
        const float *x = input + (size_t)token * c.hidden;
        gpu_mla_matvec(qn, w.qa, x, c.q_lora);
        float square = 0.0f;
        for (int d = 0; d < c.q_lora; ++d) square += qn[d] * qn[d];
        float inverse = rsqrtf(square / c.q_lora + c.rms_norm_eps);
        for (int d = 0; d < c.q_lora; ++d)
            qn[d] *= inverse * gpu_mla_weight(w.qan, 0, d);
        gpu_mla_matvec(query, w.qb, qn, c.heads * c.qk_nope);

        float *latent =
            gpu_mla_page_row(pages, c.page_tokens, absolute, c.kv_lora, 0);
        gpu_mla_matvec(latent, w.kva, x, c.kv_lora);
        square = 0.0f;
        for (int d = 0; d < c.kv_lora; ++d)
            square += latent[d] * latent[d];
        inverse = rsqrtf(square / c.kv_lora + c.rms_norm_eps);
        for (int d = 0; d < c.kv_lora; ++d)
            latent[d] *= inverse * gpu_mla_weight(w.kvan, 0, d);
        for (int h = 0; h < c.heads; ++h)
            for (int d = 0; d < c.kv_lora; ++d) {
                float sum = 0.0f;
                for (int q = 0; q < c.qk_nope; ++q)
                    sum += gpu_mla_weight(
                               w.kvbk, h * c.kv_lora + d, q) *
                           query[(size_t)h * c.qk_nope + q];
                absorbed[(size_t)h * c.kv_lora + d] = sum;
            }

        gpu_mla_matvec(iq, w.iwq, qn, c.index_heads * c.index_dim);
        float *key =
            gpu_mla_page_row(pages, c.page_tokens, absolute, c.index_dim, 1);
        gpu_mla_matvec(key, w.iwk, x, c.index_dim);
        float mean = 0.0f;
        for (int d = 0; d < c.index_dim; ++d) mean += key[d];
        mean /= c.index_dim;
        float variance = 0.0f;
        for (int d = 0; d < c.index_dim; ++d) {
            float centered = key[d] - mean;
            variance += centered * centered;
        }
        inverse = rsqrtf(variance / c.index_dim + c.index_norm_eps);
        for (int d = 0; d < c.index_dim; ++d)
            key[d] = (key[d] - mean) * inverse *
                         gpu_mla_weight(w.iknw, 0, d) +
                     gpu_mla_weight(w.iknb, 0, d);
        gpu_mla_matvec(
            gpu_mla_page_row(
                pages, c.page_tokens, absolute, c.index_dim, 2),
            w.gate, x, c.index_dim);
        gpu_mla_matvec(head_w, w.iwp, x, c.index_heads);
        for (int h = 0; h < c.index_heads; ++h) head_w[h] *= head_scale;

        int *chosen = selected + (size_t)token * width;
        for (int slot = 0; slot < width; ++slot) chosen[slot] = -1;
        int pools = seen / c.index_pool;
        for (int p = 0; p < pools; ++p) {
            for (int d = 0; d < c.index_dim; ++d) {
                float maximum = -3.402823466e+38F;
                for (int j = 0; j < c.index_pool; ++j) {
                    int position = p * c.index_pool + j;
                    float logit =
                        gpu_mla_page_row(
                            pages, c.page_tokens, position,
                            c.index_dim, 2)[d] +
                        gpu_mla_weight(w.ape, j, d);
                    maximum = fmaxf(maximum, logit);
                }
                float total = 0.0f;
                for (int j = 0; j < c.index_pool; ++j)
                    total += expf(
                        gpu_mla_page_row(
                            pages, c.page_tokens, p * c.index_pool + j,
                            c.index_dim, 2)[d] +
                        gpu_mla_weight(w.ape, j, d) - maximum);
                float mixed = 0.0f;
                for (int j = 0; j < c.index_pool; ++j) {
                    int position = p * c.index_pool + j;
                    float weight = expf(
                        gpu_mla_page_row(
                            pages, c.page_tokens, position,
                            c.index_dim, 2)[d] +
                        gpu_mla_weight(w.ape, j, d) - maximum) / total;
                    mixed += weight *
                        gpu_mla_page_row(
                            pages, c.page_tokens, position,
                            c.index_dim, 1)[d];
                }
                pooled_index[d] = mixed;
            }
            float score = 0.0f;
            for (int h = 0; h < c.index_heads; ++h) {
                float dot = 0.0f;
                for (int d = 0; d < c.index_dim; ++d)
                    dot += iq[(size_t)h * c.index_dim + d] * pooled_index[d];
                if (dot > 0.0f) score += head_w[h] * dot * index_scale;
            }
            scores[p] = score;
        }
        int wanted = c.index_topk / c.index_pool;
        for (int rank = 0; rank < wanted; ++rank) {
            int best = -1;
            for (int p = 0; p < pools; ++p) {
                int taken = 0;
                for (int prior = 0; prior < rank; ++prior)
                    if (chosen[prior * c.index_pool] == p * c.index_pool)
                        taken = 1;
                if (!taken && (best < 0 || scores[p] > scores[best])) best = p;
            }
            if (best < 0) break;
            for (int j = 0; j < c.index_pool; ++j)
                chosen[rank * c.index_pool + j] =
                    best * c.index_pool + j;
        }
        if (c.index_select_tail) {
            int tail = seen % c.index_pool;
            int tail_start = seen - tail;
            for (int j = 0; j < tail; ++j)
                chosen[c.index_topk + j] = tail_start + j;
        }

        for (int h = 0; h < c.heads; ++h) {
            float maximum = -3.402823466e+38F;
            int used = 0;
            for (int slot = 0; slot < width; ++slot) {
                int at = chosen[slot];
                if (at < 0 || at >= seen) continue;
                float dot = 0.0f;
                for (int d = 0; d < c.kv_lora; ++d)
                    dot += absorbed[(size_t)h * c.kv_lora + d] *
                           gpu_mla_page_row(
                               pages, c.page_tokens, at,
                               c.kv_lora, 0)[d];
                scores[used] = dot * attention_scale;
                maximum = fmaxf(maximum, scores[used++]);
            }
            for (int v = 0; v < c.value_dim; ++v)
                context[(size_t)h * c.value_dim + v] = 0.0f;
            if (!used) continue;
            double total = 0.0;
            for (int i = 0; i < used; ++i) {
                scores[i] = expf(scores[i] - maximum);
                total += scores[i];
            }
            for (int d = 0; d < c.kv_lora; ++d) pooled[d] = 0.0f;
            int score_index = 0;
            for (int slot = 0; slot < width; ++slot) {
                int at = chosen[slot];
                if (at < 0 || at >= seen) continue;
                float weight = (float)(scores[score_index++] / total);
                for (int d = 0; d < c.kv_lora; ++d)
                    pooled[d] += weight *
                        gpu_mla_page_row(
                            pages, c.page_tokens, at,
                            c.kv_lora, 0)[d];
            }
            for (int v = 0; v < c.value_dim; ++v) {
                float sum = 0.0f;
                for (int d = 0; d < c.kv_lora; ++d)
                    sum += gpu_mla_weight(
                               w.kvbv, h * c.value_dim + v, d) *
                           pooled[d];
                context[(size_t)h * c.value_dim + v] = sum;
            }
        }
        gpu_mla_matvec(output + (size_t)token * c.hidden, w.output,
                       context, c.hidden);
    }
}

__device__ static float gpu_mla_dot_row(
    ColiGpuMlaWeightView weight, int row, const float *input) {
    float sum = 0.0f;
    for (int column = 0; column < weight.columns; ++column)
        sum += gpu_mla_weight(weight, row, column) * input[column];
    return sum;
}

__device__ static float gpu_mla_block_sum(float value, float *shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step >>= 1) {
        if ((int)threadIdx.x < step)
            shared[threadIdx.x] += shared[threadIdx.x + step];
        __syncthreads();
    }
    return shared[0];
}

__global__ static void gpu_mla_prepare_kernel(
    const float *input, float *scratch, ColiGpuMlaPage *pages, int *status,
    ColiGpuMlaConfig c, ColiGpuMlaKernelWeights w,
    int rows, int start, int width) {
    if (*status != COLI_GPU_MLA_STATUS_OK) return;
    int token = (int)blockIdx.x;
    if (token >= rows) return;
    size_t qn_count = (size_t)rows * c.q_lora;
    size_t query_count = (size_t)rows * c.heads * c.qk_nope;
    size_t absorbed_count = (size_t)rows * c.heads * c.kv_lora;
    size_t iq_count = (size_t)rows * c.index_heads * c.index_dim;
    size_t head_count = (size_t)rows * c.index_heads;
    size_t pools_max =
        ((size_t)c.max_context + c.index_pool - 1u) / c.index_pool;
    size_t attention_scores = (size_t)c.heads * width;
    size_t score_stride =
        pools_max > attention_scores ? pools_max : attention_scores;
    float *all_qn = scratch;
    float *all_query = all_qn + qn_count;
    float *all_absorbed = all_query + query_count;
    float *all_iq = all_absorbed + absorbed_count;
    float *all_head = all_iq + iq_count;
    float *qn = all_qn + (size_t)token * c.q_lora;
    float *query =
        all_query + (size_t)token * c.heads * c.qk_nope;
    float *absorbed =
        all_absorbed + (size_t)token * c.heads * c.kv_lora;
    float *iq =
        all_iq + (size_t)token * c.index_heads * c.index_dim;
    float *head_w = all_head + (size_t)token * c.index_heads;
    (void)head_count;
    (void)score_stride;
    int absolute = start + token;
    const float *x = input + (size_t)token * c.hidden;
    __shared__ float reduction[256];

    for (int d = threadIdx.x; d < c.q_lora; d += blockDim.x)
        qn[d] = gpu_mla_dot_row(w.qa, d, x);
    __syncthreads();
    float partial = 0.0f;
    for (int d = threadIdx.x; d < c.q_lora; d += blockDim.x)
        partial += qn[d] * qn[d];
    float inverse = rsqrtf(
        gpu_mla_block_sum(partial, reduction) / c.q_lora +
        c.rms_norm_eps);
    for (int d = threadIdx.x; d < c.q_lora; d += blockDim.x)
        qn[d] *= inverse * gpu_mla_weight(w.qan, 0, d);
    __syncthreads();

    for (int d = threadIdx.x; d < c.heads * c.qk_nope;
         d += blockDim.x)
        query[d] = gpu_mla_dot_row(w.qb, d, qn);
    __syncthreads();
    for (int d = threadIdx.x; d < c.heads * c.kv_lora;
         d += blockDim.x) {
        int h = d / c.kv_lora;
        int latent_d = d % c.kv_lora;
        float sum = 0.0f;
        for (int q = 0; q < c.qk_nope; ++q)
            sum += gpu_mla_weight(
                       w.kvbk, h * c.kv_lora + latent_d, q) *
                   query[(size_t)h * c.qk_nope + q];
        absorbed[d] = sum;
    }

    float *latent =
        gpu_mla_page_row(pages, c.page_tokens, absolute, c.kv_lora, 0);
    for (int d = threadIdx.x; d < c.kv_lora; d += blockDim.x)
        latent[d] = gpu_mla_dot_row(w.kva, d, x);
    __syncthreads();
    partial = 0.0f;
    for (int d = threadIdx.x; d < c.kv_lora; d += blockDim.x)
        partial += latent[d] * latent[d];
    inverse = rsqrtf(
        gpu_mla_block_sum(partial, reduction) / c.kv_lora +
        c.rms_norm_eps);
    for (int d = threadIdx.x; d < c.kv_lora; d += blockDim.x)
        latent[d] *= inverse * gpu_mla_weight(w.kvan, 0, d);

    for (int d = threadIdx.x; d < c.index_heads * c.index_dim;
         d += blockDim.x)
        iq[d] = gpu_mla_dot_row(w.iwq, d, qn);
    float *key =
        gpu_mla_page_row(pages, c.page_tokens, absolute, c.index_dim, 1);
    for (int d = threadIdx.x; d < c.index_dim; d += blockDim.x)
        key[d] = gpu_mla_dot_row(w.iwk, d, x);
    __syncthreads();
    partial = 0.0f;
    for (int d = threadIdx.x; d < c.index_dim; d += blockDim.x)
        partial += key[d];
    float mean = gpu_mla_block_sum(partial, reduction) / c.index_dim;
    partial = 0.0f;
    for (int d = threadIdx.x; d < c.index_dim; d += blockDim.x) {
        float centered = key[d] - mean;
        partial += centered * centered;
    }
    inverse = rsqrtf(
        gpu_mla_block_sum(partial, reduction) / c.index_dim +
        c.index_norm_eps);
    for (int d = threadIdx.x; d < c.index_dim; d += blockDim.x)
        key[d] = (key[d] - mean) * inverse *
                     gpu_mla_weight(w.iknw, 0, d) +
                 gpu_mla_weight(w.iknb, 0, d);
    float *gate =
        gpu_mla_page_row(pages, c.page_tokens, absolute, c.index_dim, 2);
    for (int d = threadIdx.x; d < c.index_dim; d += blockDim.x)
        gate[d] = gpu_mla_dot_row(w.gate, d, x);
    float head_scale = rsqrtf((float)c.index_heads);
    for (int h = threadIdx.x; h < c.index_heads; h += blockDim.x)
        head_w[h] = gpu_mla_dot_row(w.iwp, h, x) * head_scale;
}

__global__ static void gpu_mla_attention_kernel(
    float *output, float *scratch, int *selected,
    ColiGpuMlaPage *pages, int *status,
    ColiGpuMlaConfig c, ColiGpuMlaKernelWeights w,
    int rows, int start, int width) {
    if (*status != COLI_GPU_MLA_STATUS_OK) return;
    int token = (int)blockIdx.x;
    if (token >= rows) return;
    size_t qn_count = (size_t)rows * c.q_lora;
    size_t query_count = (size_t)rows * c.heads * c.qk_nope;
    size_t absorbed_count = (size_t)rows * c.heads * c.kv_lora;
    size_t iq_count = (size_t)rows * c.index_heads * c.index_dim;
    size_t head_count = (size_t)rows * c.index_heads;
    size_t pools_max =
        ((size_t)c.max_context + c.index_pool - 1u) / c.index_pool;
    size_t attention_scores = (size_t)c.heads * width;
    size_t score_stride =
        pools_max > attention_scores ? pools_max : attention_scores;
    float *all_qn = scratch;
    float *all_query = all_qn + qn_count;
    float *all_absorbed = all_query + query_count;
    float *all_iq = all_absorbed + absorbed_count;
    float *all_head = all_iq + iq_count;
    float *all_scores = all_head + head_count;
    float *all_context = all_scores + (size_t)rows * score_stride;
    float *absorbed =
        all_absorbed + (size_t)token * c.heads * c.kv_lora;
    float *iq =
        all_iq + (size_t)token * c.index_heads * c.index_dim;
    float *head_w = all_head + (size_t)token * c.index_heads;
    float *scores = all_scores + (size_t)token * score_stride;
    float *context =
        all_context + (size_t)token * c.heads * c.value_dim;
    int *chosen = selected + (size_t)token * width;
    int seen = start + token + 1;
    int pools = seen / c.index_pool;
    float index_scale = rsqrtf((float)c.index_dim);

    for (int p = threadIdx.x; p < pools; p += blockDim.x) {
        float score = 0.0f;
        for (int h = 0; h < c.index_heads; ++h) {
            float dot = 0.0f;
            for (int d = 0; d < c.index_dim; ++d) {
                float maximum = -3.402823466e+38F;
                for (int j = 0; j < c.index_pool; ++j) {
                    float logit = gpu_mla_page_row(
                        pages, c.page_tokens, p * c.index_pool + j,
                        c.index_dim, 2)[d] +
                        gpu_mla_weight(w.ape, j, d);
                    maximum = fmaxf(maximum, logit);
                }
                float total = 0.0f;
                float mixed = 0.0f;
                for (int j = 0; j < c.index_pool; ++j) {
                    int position = p * c.index_pool + j;
                    float weight = expf(
                        gpu_mla_page_row(
                            pages, c.page_tokens, position,
                            c.index_dim, 2)[d] +
                        gpu_mla_weight(w.ape, j, d) - maximum);
                    total += weight;
                    mixed += weight * gpu_mla_page_row(
                        pages, c.page_tokens, position,
                        c.index_dim, 1)[d];
                }
                dot += iq[(size_t)h * c.index_dim + d] *
                       (mixed / total);
            }
            if (dot > 0.0f) score += head_w[h] * dot * index_scale;
        }
        scores[p] = score;
    }
    __syncthreads();
    if (!threadIdx.x) {
        for (int slot = 0; slot < width; ++slot) chosen[slot] = -1;
        int wanted = c.index_topk / c.index_pool;
        for (int rank = 0; rank < wanted; ++rank) {
            int best = -1;
            for (int p = 0; p < pools; ++p) {
                int taken = 0;
                for (int prior = 0; prior < rank; ++prior)
                    if (chosen[prior * c.index_pool] ==
                        p * c.index_pool)
                        taken = 1;
                if (!taken && (best < 0 || scores[p] > scores[best]))
                    best = p;
            }
            if (best < 0) break;
            for (int j = 0; j < c.index_pool; ++j)
                chosen[rank * c.index_pool + j] =
                    best * c.index_pool + j;
        }
        if (c.index_select_tail) {
            int tail = seen % c.index_pool;
            int tail_start = seen - tail;
            for (int j = 0; j < tail; ++j)
                chosen[c.index_topk + j] = tail_start + j;
        }
    }
    __syncthreads();

    float attention_scale = rsqrtf((float)c.qk_nope);
    for (int hs = threadIdx.x; hs < c.heads * width;
         hs += blockDim.x) {
        int h = hs / width;
        int slot = hs % width;
        int at = chosen[slot];
        float score = -3.402823466e+38F;
        if (at >= 0 && at < seen) {
            float dot = 0.0f;
            const float *latent = gpu_mla_page_row(
                pages, c.page_tokens, at, c.kv_lora, 0);
            for (int d = 0; d < c.kv_lora; ++d)
                dot += absorbed[(size_t)h * c.kv_lora + d] * latent[d];
            score = dot * attention_scale;
        }
        scores[(size_t)h * width + slot] = score;
    }
    __syncthreads();
    for (int h = threadIdx.x; h < c.heads; h += blockDim.x) {
        float maximum = -3.402823466e+38F;
        for (int slot = 0; slot < width; ++slot)
            maximum = fmaxf(maximum, scores[(size_t)h * width + slot]);
        double total = 0.0;
        for (int slot = 0; slot < width; ++slot) {
            int at = chosen[slot];
            float value = at >= 0 && at < seen
                ? expf(scores[(size_t)h * width + slot] - maximum)
                : 0.0f;
            scores[(size_t)h * width + slot] = value;
            total += value;
        }
        if (total)
            for (int slot = 0; slot < width; ++slot)
                scores[(size_t)h * width + slot] =
                    (float)(scores[(size_t)h * width + slot] / total);
    }
    __syncthreads();
    for (int hv = threadIdx.x; hv < c.heads * c.value_dim;
         hv += blockDim.x) {
        int h = hv / c.value_dim;
        int v = hv % c.value_dim;
        float sum = 0.0f;
        for (int slot = 0; slot < width; ++slot) {
            int at = chosen[slot];
            if (at < 0 || at >= seen) continue;
            const float *latent = gpu_mla_page_row(
                pages, c.page_tokens, at, c.kv_lora, 0);
            float expanded = 0.0f;
            for (int d = 0; d < c.kv_lora; ++d)
                expanded +=
                    gpu_mla_weight(w.kvbv, h * c.value_dim + v, d) *
                    latent[d];
            sum += scores[(size_t)h * width + slot] * expanded;
        }
        context[hv] = sum;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < c.hidden; d += blockDim.x)
        output[(size_t)token * c.hidden + d] =
            gpu_mla_dot_row(w.output, d, context);
}

static ColiGpuMlaWeightView gpu_mla_view(const ColiGpuTensor *tensor) {
    ColiGpuMlaWeightView view = {};
    if (tensor) {
        view.data = tensor->data;
        view.scales = tensor->scales;
        view.format = tensor->format;
        view.columns = tensor->columns;
        view.groups = tensor->groups;
        view.group_size = tensor->group_size;
    }
    return view;
}

static int gpu_mla_weight_shape(
    const ColiGpuArena *arena, const ColiGpuTensor *tensor,
    int rows, int columns, int f32_only) {
    return gpu_tensor_same_context(arena, tensor) &&
           tensor->rows == rows && tensor->columns == columns &&
           (!f32_only || tensor->format == 0);
}

static int gpu_mla_validate_range(
    const float *values, size_t count, int failure_status,
    ColiGpuMlaState *state) {
    if (!count) return 1;
    int blocks = (int)((count + 255u) / 256u);
    if (blocks > 65535) blocks = 65535;
    gpu_mla_validate_finite_kernel<<<blocks, 256, 0, state->ctx->stream>>>(
        values, count, state->device_status, failure_status);
    return cuda_ok(cudaGetLastError(),
                   "resident MLA finite validation launch");
}

extern "C" int coli_gpu_mla_site(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t scratch_offset, size_t selected_offset,
    ColiGpuMlaState *state, const ColiGpuMlaWeights *weights,
    int rows, int start_position) {
    if (!arena || !state || !weights || arena->ctx != state->ctx ||
        !state->ctx->healthy || rows < 1 || rows > state->config.max_rows ||
        start_position < 0 || start_position != state->length ||
        start_position > state->config.max_context - rows)
        return 0;
    const ColiGpuMlaConfig &c = state->config;
    int width = gpu_mla_width(&c);
    size_t row_bytes = (size_t)rows * c.hidden * sizeof(float);
    size_t scratch_bytes = coli_gpu_mla_scratch_bytes(&c);
    size_t selected_bytes = coli_gpu_mla_selected_bytes(&c, rows);
    ColiGpuKdaRange ranges[] = {
        {output_offset, row_bytes},
        {input_offset, row_bytes},
        {scratch_offset, scratch_bytes},
        {selected_offset, selected_bytes}
    };
    if (!gpu_range_ok(arena, output_offset, row_bytes) ||
        !gpu_range_ok(arena, input_offset, row_bytes) ||
        !gpu_range_ok(arena, scratch_offset, scratch_bytes) ||
        !gpu_range_ok(arena, selected_offset, selected_bytes))
        return 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < i; ++j)
            if (gpu_kda_ranges_overlap(ranges[i], ranges[j])) return 0;
    if (!gpu_mla_weight_shape(arena, weights->q_a_proj,
                              c.q_lora, c.hidden, 0) ||
        !gpu_mla_weight_shape(arena, weights->q_a_norm, 1, c.q_lora, 1) ||
        !gpu_mla_weight_shape(arena, weights->q_b_proj,
                              c.heads * c.qk_nope, c.q_lora, 0) ||
        !gpu_mla_weight_shape(arena, weights->kv_a_proj,
                              c.kv_lora, c.hidden, 0) ||
        !gpu_mla_weight_shape(arena, weights->kv_a_norm, 1, c.kv_lora, 1) ||
        !gpu_mla_weight_shape(arena, weights->kv_b_key,
                              c.heads * c.kv_lora, c.qk_nope, 0) ||
        !gpu_mla_weight_shape(arena, weights->kv_b_value,
                              c.heads * c.value_dim, c.kv_lora, 0) ||
        !gpu_mla_weight_shape(arena, weights->o_proj,
                              c.hidden, c.heads * c.value_dim, 0) ||
        !gpu_mla_weight_shape(arena, weights->index_q_proj,
                              c.index_heads * c.index_dim, c.q_lora, 0) ||
        !gpu_mla_weight_shape(arena, weights->index_k_proj,
                              c.index_dim, c.hidden, 0) ||
        !gpu_mla_weight_shape(arena, weights->index_weight_proj,
                              c.index_heads, c.hidden, 0) ||
        !gpu_mla_weight_shape(arena, weights->index_key_norm,
                              1, c.index_dim, 1) ||
        !gpu_mla_weight_shape(arena, weights->index_key_bias,
                              1, c.index_dim, 1) ||
        !gpu_mla_weight_shape(arena, weights->index_pool_ape,
                              c.index_pool, c.index_dim, 1) ||
        !gpu_mla_weight_shape(arena, weights->index_pool_gate,
                              c.index_dim, c.hidden, 0) ||
        !select_device_ordinal(state->ctx->device) ||
        !gpu_mla_reserve(state, start_position + rows))
        return 0;
    ColiGpuMlaKernelWeights views = {
        gpu_mla_view(weights->q_a_proj),
        gpu_mla_view(weights->q_a_norm),
        gpu_mla_view(weights->q_b_proj),
        gpu_mla_view(weights->kv_a_proj),
        gpu_mla_view(weights->kv_a_norm),
        gpu_mla_view(weights->kv_b_key),
        gpu_mla_view(weights->kv_b_value),
        gpu_mla_view(weights->o_proj),
        gpu_mla_view(weights->index_q_proj),
        gpu_mla_view(weights->index_k_proj),
        gpu_mla_view(weights->index_weight_proj),
        gpu_mla_view(weights->index_key_norm),
        gpu_mla_view(weights->index_key_bias),
        gpu_mla_view(weights->index_pool_ape),
        gpu_mla_view(weights->index_pool_gate)
    };
    *state->host_status = COLI_GPU_MLA_STATUS_OK;
    if (!gpu_mla_validate_range(
            reinterpret_cast<const float *>(arena->data + input_offset),
            (size_t)rows * c.hidden, COLI_GPU_MLA_STATUS_NONFINITE_INPUT,
            state))
        return 0;
    for (int page = 0, checked = 0; checked < state->length; ++page) {
        int count = state->length - checked;
        if (count > c.page_tokens) count = c.page_tokens;
        if (!gpu_mla_validate_range(
                state->pages[page].latent, (size_t)count * c.kv_lora,
                COLI_GPU_MLA_STATUS_NONFINITE_CACHE, state) ||
            !gpu_mla_validate_range(
                state->pages[page].index_keys, (size_t)count * c.index_dim,
                COLI_GPU_MLA_STATUS_NONFINITE_CACHE, state) ||
            !gpu_mla_validate_range(
                state->pages[page].index_gates, (size_t)count * c.index_dim,
                COLI_GPU_MLA_STATUS_NONFINITE_CACHE, state))
            return 0;
        checked += count;
    }
    gpu_mla_prepare_kernel<<<rows, 256, 0, state->ctx->stream>>>(
        reinterpret_cast<const float *>(arena->data + input_offset),
        reinterpret_cast<float *>(arena->data + scratch_offset),
        state->device_pages, state->device_status,
        c, views, rows, start_position, width);
    gpu_mla_record_launch(state, rows, 256);
    if (!cuda_ok(cudaGetLastError(), "resident MLA preparation launch"))
        return 0;
    int end = start_position + rows;
    for (int page = start_position / c.page_tokens,
             checked = start_position;
         checked < end; ++page) {
        int offset = checked % c.page_tokens;
        int count = c.page_tokens - offset;
        if (count > end - checked) count = end - checked;
        if (!gpu_mla_validate_range(
                state->pages[page].latent + (size_t)offset * c.kv_lora,
                (size_t)count * c.kv_lora,
                COLI_GPU_MLA_STATUS_NONFINITE_RESULT, state) ||
            !gpu_mla_validate_range(
                state->pages[page].index_keys +
                    (size_t)offset * c.index_dim,
                (size_t)count * c.index_dim,
                COLI_GPU_MLA_STATUS_NONFINITE_RESULT, state) ||
            !gpu_mla_validate_range(
                state->pages[page].index_gates +
                    (size_t)offset * c.index_dim,
                (size_t)count * c.index_dim,
                COLI_GPU_MLA_STATUS_NONFINITE_RESULT, state))
            return 0;
        checked += count;
    }
    gpu_mla_attention_kernel<<<rows, 256, 0, state->ctx->stream>>>(
        reinterpret_cast<float *>(arena->data + output_offset),
        reinterpret_cast<float *>(arena->data + scratch_offset),
        reinterpret_cast<int *>(arena->data + selected_offset),
        state->device_pages, state->device_status,
        c, views, rows, start_position, width);
    gpu_mla_record_launch(state, rows, 256);
    if (!cuda_ok(cudaGetLastError(), "resident MLA attention launch") ||
        !gpu_mla_validate_range(
            reinterpret_cast<const float *>(arena->data + output_offset),
            (size_t)rows * c.hidden, COLI_GPU_MLA_STATUS_NONFINITE_RESULT,
            state))
        return 0;
    if (!coli_gpu_context_sync(state->ctx) ||
        *state->host_status != COLI_GPU_MLA_STATUS_OK)
        return 0;
    state->length += rows;
    return 1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

__global__ static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    const float *xr=x+(size_t)blockIdx.x*xstride; float *yr=y+(size_t)blockIdx.x*ystride;
    __shared__ double sh[256];
    double a=0; for(int i=threadIdx.x;i<D;i+=blockDim.x){ double v=xr[i]; a+=v*v; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float r=rsqrtf((float)(sh[0]/D)+eps);
    for(int i=threadIdx.x;i<D;i+=blockDim.x) yr[i]=xr[i]*r*w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
__global__ static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    float *p=v+(size_t)blockIdx.x*stride+offset;
    int half=R/2, ps=pos?pos[blockIdx.x/heads]:pos_base+(int)(blockIdx.x/heads);
    __shared__ float in[256];
    for(int j=threadIdx.x;j<R;j+=blockDim.x) in[j]=p[j];
    __syncthreads();
    for(int j=threadIdx.x;j<half;j+=blockDim.x){
        float inv=__powf(theta,-2.0f*j/R);
        float ang=ps*inv, cs=__cosf(ang), sn=__sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

__global__ static void pipe_add_n(float *x,const float *t,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
__global__ static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    float *xr=x+(size_t)rows[blockIdx.x]*D;
    const float *pr=partial+(size_t)blockIdx.x*D;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) xr[i]+=pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=27||!select_ctx(ctx)) return NULL;
    if(!reserve(&ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
extern "C" void *coli_cuda_pipe_alloc(int device,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if(!cuda_ok(cudaMalloc(&p,bytes),"pipe alloc")) return NULL;
    return p;
}
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    cudaFree(p);
}
extern "C" int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"pipe upload");
}
extern "C" int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"pipe download");
}
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,D,D);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,xstride,ystride);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,pos_dev,0,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,NULL,pos_base,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope base");
}
/* ---- device router (#431 PR-A) -------------------------------------------
 * Router for one decode row, entirely on the layer's home device: logits GEMV
 * (E x D, tiny) + sigmoid, bias-augmented top-K selection, route-level TOPP
 * truncation, norm_topk and routed_scale — a float-faithful clone of moe()'s
 * plain routing path (colibri.c FASE A). Selection runs single-thread so the
 * argmax order, tie-breaking (strict >, lowest index wins) and weight math
 * match the CPU reference exactly; only the dot/expf rounding can differ,
 * which is the documented kernel-family divergence class (#100/#163).
 * Results are packed [idx[K] | w[K] | keff] in one scratch buffer and read
 * back with a single tiny D2H. */
__global__ void pipe_router_logits(const float *__restrict__ x,
                                   const float *__restrict__ W,
                                   const float *__restrict__ bias,
                                   int D, float *logit, float *choice){
    int e = blockIdx.x;
    const float *w = W + (size_t)e*D;
    float acc = 0.f;
    for(int i=threadIdx.x; i<D; i+=blockDim.x) acc += x[i]*w[i];
    __shared__ float sh[128];
    sh[threadIdx.x]=acc; __syncthreads();
    for(int s=blockDim.x>>1; s>0; s>>=1){
        if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s];
        __syncthreads();
    }
    if(!threadIdx.x){
        float lg = 1.f/(1.f+expf(-sh[0]));
        logit[e]=lg; choice[e]=lg+bias[e];
    }
}
__global__ void pipe_router_select(const float *__restrict__ logit,
                                   const float *__restrict__ choice, int E,
                                   int Ksel, float topp, int norm_topk,
                                   float routed_scale, char *out){
    if(threadIdx.x||blockIdx.x) return;
    int   *idx = (int*)out;
    float *w   = (float*)(out + Ksel*sizeof(int));
    int   *keff= (int*)(out + Ksel*(sizeof(int)+sizeof(float)));
    for(int kk=0;kk<Ksel;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
            if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
        idx[kk]=best; w[kk]=logit[best];
    }
    int Ke=Ksel;
    if(topp>0.f && topp<1.f){
        for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
            while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
        float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
        float cum=0.f; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
    }
    if(norm_topk){ float sm=0.f; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f;
                   for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
    for(int kk=0;kk<Ke;kk++) w[kk]*=routed_scale;
    *keff=Ke;
}
extern "C" int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host){
    DeviceContext *ctx=find_ctx(device);
    if(!x_dev||!rw_dev||!rb_dev||D<1||E<1||E>4096||Ksel<1||Ksel>64||!select_ctx(ctx)) return 0;
    size_t pack=(size_t)Ksel*(sizeof(int)+sizeof(float))+sizeof(int);
    float *logit=coli_cuda_pipe_scratch(device,22,(size_t)E*sizeof(float));
    float *chc  =coli_cuda_pipe_scratch(device,23,(size_t)E*sizeof(float));
    char  *out  =(char*)coli_cuda_pipe_scratch(device,24,pack);
    if(!logit||!chc||!out) return 0;
    pipe_router_logits<<<E,128>>>(x_dev,(const float*)rw_dev,(const float*)rb_dev,D,logit,chc);
    pipe_router_select<<<1,1>>>(logit,chc,E,Ksel,topp,norm_topk,routed_scale,out);
    if(!cuda_ok(cudaGetLastError(),"pipe router launch")) return 0;
    char buf[64*(sizeof(int)+sizeof(float))+sizeof(int)];
    if(!cuda_ok(cudaMemcpy(buf,out,pack,cudaMemcpyDeviceToHost),"pipe router readback")) return 0;
    memcpy(idx_host,buf,(size_t)Ksel*sizeof(int));
    memcpy(w_host,buf+Ksel*sizeof(int),(size_t)Ksel*sizeof(float));
    memcpy(keff_host,buf+Ksel*(sizeof(int)+sizeof(float)),sizeof(int));
    return 1;
}
/* ---- resident expert-group accumulation (#431 PR-C0) ----------------------
 * Decode-time (S=1) expert groups without the host round-trip: the input row
 * is P2P'd from the layer's home device, the group runs through the grouped-W4
 * kernels on its own stream, the down-projection outputs are weighted and
 * reduced ON DEVICE (fixed expert order), and the device's partial sum is
 * peer-pushed into a per-issue slot on the home device. take() makes the home
 * legacy stream wait on every issue event and reduces the slots in issue order
 * — deterministic, no atomics, no host bytes. The CPU tier overlaps with all
 * of it exactly as before. */
__global__ static void bcast_row(float *dst,const float *src,int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float v=src[i];
        for(int c=0;c<count;c++) dst[(size_t)c*D+i]=v;
    }
}
__global__ static void weighted_sum_rows(float *out,const float *y,const float *w,
                                         int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int c=0;c<count;c++) acc+=w[c]*y[(size_t)c*D+i];   /* fixed order */
        out[i]=acc;
    }
}
__global__ static void sum_slots(float *dst,const float *slots,int n,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int s=0;s<n;s++) acc+=slots[(size_t)s*D+i];        /* issue order */
        dst[i]=acc;
    }
}
extern "C" int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev){
    if(!gates||!ups||!downs||!weights||count<1||count>64||!x_src_dev||!partial_slot_dev) return 0;
    ColiCudaTensor *first=gates[0]; if(!first) return 0;
    int device=first->device,D=first->I,I=first->O;
    GroupDesc host[64];
    int total=0,all_s4=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,1,total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        total++;
    }
    if(!all_s4) return 0;                       /* resident path: per-row int4 only */
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    if(!ctx->ev_done_ok){
        if(!cuda_ok(cudaEventCreateWithFlags(&ctx->ev_done,cudaEventDisableTiming),
                    "resident group event")) return 0;
        ctx->ev_done_ok=1;
    }
    /* size for the 64-expert cap, not for `count`: reserve() reallocs on growth,
     * and a realloc here could free a buffer the PREVIOUS layer's still-queued
     * async work on this stream reads. Fixed caps make re-issue realloc-free. */
    size_t xb=(size_t)64*D*sizeof(float), ib=(size_t)64*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->ac,&ctx->ac_cap,(size_t)(D+64)*sizeof(float))||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)64*sizeof(GroupDesc)))
        return 0;
    float *w_dev=ctx->ac+D, *partial_local=ctx->ac;
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),"resident group desc")||
       !cuda_ok(cudaMemcpyAsync(w_dev,weights,(size_t)count*sizeof(float),
                                cudaMemcpyHostToDevice,ctx->stream),"resident group weights"))
        return 0;
    /* input row: P2P from the home device. The caller guarantees x_src_dev is
     * materialized (the pre-moe nrm download already synced the home stream). */
    if(!cuda_ok(cudaMemcpyPeerAsync(ctx->x,device,x_src_dev,home_device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident group x p2p"))
        return 0;
    bcast_row<<<64,256,0,ctx->stream>>>(ctx->x,ctx->x,count,D);   /* row 0 -> rows 1..count-1 (in-place safe: row 0 rewritten with itself) */
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    dim3 hg((unsigned)I,1,(unsigned)count),og((unsigned)D,1,(unsigned)count);
    grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);  /* silu fused in epilogue */
    grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    weighted_sum_rows<<<48,256,0,ctx->stream>>>(partial_local,ctx->y,w_dev,count,D);
    if(!cuda_ok(cudaMemcpyPeerAsync(partial_slot_dev,home_device,partial_local,device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident partial p2p"))
        return 0;
    if(!cuda_ok(cudaEventRecord(ctx->ev_done,ctx->stream),"resident event record")) return 0;
    return cuda_ok(cudaGetLastError(),"resident group launch");
}
extern "C" int coli_cuda_expert_group_resident_take(int home_device,const int *devices,int n_issued,
                                           float *slots_dev,float *acc_dev,int D){
    if(n_issued<1||!slots_dev||!acc_dev||D<1) return 0;
    DeviceContext *home=find_ctx(home_device); if(!select_ctx(home)) return 0;
    for(int i=0;i<n_issued;i++){
        DeviceContext *src=find_ctx(devices[i]);
        if(!src||!src->ev_done_ok) return 0;
        if(!cuda_ok(cudaStreamWaitEvent(0,src->ev_done,0),"resident take wait")) return 0;
    }
    sum_slots<<<48,256>>>(acc_dev,slots_dev,n_issued,D);          /* legacy stream: ordered with pipe_* */
    return cuda_ok(cudaGetLastError(),"resident take reduce");
}
extern "C" int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                                     int spitch,int width,int height){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy2D(dst,(size_t)dpitch*4,src,(size_t)spitch*4,
        (size_t)width*4,height,cudaMemcpyDeviceToDevice),"pipe copy2d");
}
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
extern "C" int coli_cuda_attention_project_batch_dev(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"pipe attention download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(gate_dev,up_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    pipe_add_n<<<(unsigned)((n+255)/256),256>>>(x_dev,t_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rows_add<<<nrows,256>>>(x_dev,partial_dev,rows_dev,D);
    return cuda_ok(cudaGetLastError(),"pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S){
    if (fault_injected()) return 0;
    if(!t||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    dim3 grid((unsigned)t->O,(unsigned)S);
    quant_matmul<<<grid,256>>>(y_dev,x_dev,t->weights,t->scales,t->fmt,S,t->I,t->O,
        row_bytes(t->fmt,t->I),t->gs,t->ng);
    return cuda_ok(cudaGetLastError(),"pipe gemm");
}
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                                        const float *src,size_t bytes){
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),"pipe intra copy"); }
    return cuda_ok(cudaMemcpyPeer(dst,dst_dev,src,src_dev,bytes),"pipe peer copy");
}
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch (dev out)"))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(out_dev,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch (dev out)"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync (dev out)");
}
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(ctx_dev,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe shard attention launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe shard attention sync");
}
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    if(!absorb_fmt_ok(w)||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev q upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,1),256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,1,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"kvdev absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev absorb sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_sync(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaStreamSynchronize(ctx->stream),"pipe sync");
}
