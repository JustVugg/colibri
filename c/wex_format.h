#ifndef COLI_WEX_FORMAT_H
#define COLI_WEX_FORMAT_H

/* Portable WEX wire format, lossless scales and opt-in palette3 weights.
 * No engine dependency.
 * Integer fields on disk are little-endian. V2 deliberately has a new magic:
 * old readers checked only the magic and cannot safely consume compressed data.
 */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include "wex_palette.h"
#include "wex_neural_rans.h"

#define WEX_MAGIC_V1 UINT32_C(0x57455831)
#define WEX_MAGIC_V2 UINT32_C(0x57455832)
#define WEX_MAGIC_V3 UINT32_C(0x57455833)
#define WEX_MAGIC WEX_MAGIC_V1
#define WEX_ALIGN 4096u
#define WEX_IS_ALIGNED(x, a) (((uintptr_t)(x) & ((uintptr_t)(a) - 1u)) == 0)
#define WEX_ALIGN_UP(x, a) (((x) + ((uint64_t)(a) - 1u)) & ~((uint64_t)(a) - 1u))

#define WEX_HEADER_SCALE_TILES UINT32_C(1)
#define WEX_HEADER_PALETTE3 UINT32_C(2)
#define WEX_HEADER_DEDUP UINT32_C(4)
#define WEX_HEADER_NEURAL_RANS UINT32_C(8)
#define WEX_HEADER_NEURAL_RANS2 UINT32_C(16)  /* WNR2 records + a WNM2 model before the table */
#define WEX_HEADER_NEURAL_RANS3 UINT32_C(32)  /* WNR3 records + a WNM2 model; excludes 8 and 16 */
#define WEX_HEADER_CANONICAL_ZERO UINT32_C(64)  /* every WNR3 record value-exact (-0 stored as +0); needs 32 */
#define WEX_ENTRY_SCALE_TILES UINT32_C(1)
#define WEX_ENTRY_PALETTE3 UINT32_C(2)
#define WEX_ENTRY_NEURAL_RANS UINT32_C(3)
#define WEX_ENTRY_NEURAL_RANS2 UINT32_C(4)
#define WEX_ENTRY_NEURAL_RANS3 UINT32_C(5)
#define WEX_ENTRY_CODEC_MASK UINT32_C(0xff)
#define WEX_ENTRY_SCALE_PAGES_SHIFT 8u
#define WEX_SCALE_STREAM_MAGIC UINT32_C(0x31534357)
#define WEX_SCALE_STREAM_HEADER_BYTES 12u
#define WEX_SCALE_TILE_VALUES 256u

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t num_layers;
    uint32_t num_experts;
    uint64_t aligned_expert_bytes;   /* maximum decoded, 4096-aligned slab */
    uint64_t table_offset;
    uint32_t flags;
    uint32_t reserved;
} WexHeader;

typedef struct {
    uint64_t file_offset;
    uint32_t total_bytes;            /* aligned on-disk read size */
    uint32_t scale_bytes;            /* decoded scales, w1/w2/w3 */
    uint32_t weight_bytes;           /* raw packed FP4, w1/w2/w3 */
    uint16_t scale_shapes[3][2];
    uint16_t weight_shapes[3][2];
    uint32_t scale_offsets[3];       /* canonical decoded slab offsets */
    uint32_t weight_offsets[3];
    uint32_t flags;                  /* codec low 8 bits, scale pages high 24 */
} WexEntry;
#pragma pack(pop)

typedef char WexHeader_wire_size_must_be_40[(sizeof(WexHeader) == 40) ? 1 : -1];
typedef char WexEntry_wire_size_must_be_72[(sizeof(WexEntry) == 72) ? 1 : -1];

static inline uint32_t wex_read_u32_le(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* The checked helper, rather than WEX_ALIGN_UP, is used on untrusted sizes. */
static inline int wex_aligned_size(uint64_t bytes, uint64_t *aligned) {
    if (bytes > UINT64_MAX - (WEX_ALIGN - 1u)) return -1;
    *aligned = (bytes + (WEX_ALIGN - 1u)) & ~((uint64_t)WEX_ALIGN - 1u);
    return 0;
}

static inline int wex_validate_header(const WexHeader *hdr, uint64_t file_bytes,
                                      int expected_layers, int expected_experts,
                                      size_t *table_bytes,
                                      size_t *aligned_table_bytes) {
    uint64_t count, bytes, aligned;
    if (table_bytes) *table_bytes = 0;
    if (aligned_table_bytes) *aligned_table_bytes = 0;
    if (!hdr || expected_layers < 0 || expected_experts < 0) return -1;
    if (hdr->magic == WEX_MAGIC_V1 && hdr->version == 1u) {
        if (hdr->flags != 0) return -1;
    } else if (hdr->magic == WEX_MAGIC_V2 && hdr->version == 2u) {
        if (hdr->flags & ~WEX_HEADER_SCALE_TILES) return -1;
    } else if (hdr->magic == WEX_MAGIC_V3 && hdr->version == 3u) {
        if (hdr->flags & ~(WEX_HEADER_SCALE_TILES | WEX_HEADER_PALETTE3 | WEX_HEADER_DEDUP |
                           WEX_HEADER_NEURAL_RANS | WEX_HEADER_NEURAL_RANS2 |
                           WEX_HEADER_NEURAL_RANS3 | WEX_HEADER_CANONICAL_ZERO)) return -1;
        if (((hdr->flags & WEX_HEADER_NEURAL_RANS3) &&
             (hdr->flags & (WEX_HEADER_NEURAL_RANS | WEX_HEADER_NEURAL_RANS2))) ||
            ((hdr->flags & WEX_HEADER_NEURAL_RANS3) && (hdr->flags & WEX_HEADER_DEDUP)) ||   /* WNR3 never dedups */
            ((hdr->flags & WEX_HEADER_CANONICAL_ZERO) && !(hdr->flags & WEX_HEADER_NEURAL_RANS3))) return -1;
    } else return -1;
    if (hdr->reserved || !hdr->num_layers || !hdr->num_experts ||
        hdr->num_layers > INT_MAX || hdr->num_experts > INT_MAX ||
        (expected_layers && hdr->num_layers != (uint32_t)expected_layers) ||
        (expected_experts && hdr->num_experts != (uint32_t)expected_experts))
        return -1;
    if (!hdr->aligned_expert_bytes ||
        hdr->aligned_expert_bytes > UINT32_MAX ||
        hdr->aligned_expert_bytes > SIZE_MAX ||
        (hdr->aligned_expert_bytes & (WEX_ALIGN - 1u)) ||
        hdr->table_offset < WEX_ALIGN ||
        (hdr->table_offset & (WEX_ALIGN - 1u)) || file_bytes < WEX_ALIGN)
        return -1;
    count = (uint64_t)hdr->num_layers * hdr->num_experts;
    if (count > UINT64_MAX / sizeof(WexEntry)) return -1;
    bytes = count * sizeof(WexEntry);
    if (wex_aligned_size(bytes, &aligned) || bytes > SIZE_MAX ||
        aligned > SIZE_MAX || hdr->table_offset > file_bytes ||
        aligned > file_bytes - hdr->table_offset) return -1;
    if (table_bytes) *table_bytes = (size_t)bytes;
    if (aligned_table_bytes) *aligned_table_bytes = (size_t)aligned;
    return 0;
}

/* Validate the canonical raw layout shared by V1 and V2. */
static inline int wex_entry_raw_sizes(const WexEntry *entry,
                                      uint64_t *raw_bytes,
                                      uint64_t *aligned_raw_bytes) {
    uint64_t scales = 0, weights = 0;
    unsigned int matrix;
    if (!entry) return -1;
    for (matrix = 0; matrix < 3; matrix++) {
        uint32_t sr = entry->scale_shapes[matrix][0];
        uint32_t sc = entry->scale_shapes[matrix][1];
        uint32_t wr = entry->weight_shapes[matrix][0];
        uint32_t wc = entry->weight_shapes[matrix][1];
        if (!sr || !sc || !wr || !wc || sr != wr || sc * 16u != wc ||
            entry->scale_offsets[matrix] != scales)
            return -1;
        scales += (uint64_t)sr * sc;
    }
    if (scales != entry->scale_bytes) return -1;
    for (matrix = 0; matrix < 3; matrix++) {
        if (entry->weight_offsets[matrix] != scales + weights) return -1;
        weights += (uint64_t)entry->weight_shapes[matrix][0] *
                   entry->weight_shapes[matrix][1];
    }
    if (weights != entry->weight_bytes || scales + weights > UINT32_MAX)
        return -1;
    *raw_bytes = scales + weights;
    if (wex_aligned_size(*raw_bytes, aligned_raw_bytes) ||
        *aligned_raw_bytes > UINT32_MAX || *aligned_raw_bytes > SIZE_MAX)
        return -1;
    return 0;
}

static inline size_t wex_record_read_offset(const WexEntry *entry) {
    uint64_t prefix;
    if (!entry || !entry->flags) return 0;
    if (entry->flags == WEX_ENTRY_PALETTE3 || entry->flags == WEX_ENTRY_NEURAL_RANS ||
        entry->flags == WEX_ENTRY_NEURAL_RANS2 || entry->flags == WEX_ENTRY_NEURAL_RANS3) {
        uint64_t raw, aligned;
        if (wex_entry_raw_sizes(entry, &raw, &aligned) || entry->total_bytes >= aligned) return 0;
        return (size_t)(aligned - entry->total_bytes);
    }
    if ((entry->flags & WEX_ENTRY_CODEC_MASK) != WEX_ENTRY_SCALE_TILES)
        return 0;
    prefix = (uint64_t)(entry->flags >> WEX_ENTRY_SCALE_PAGES_SHIFT) * WEX_ALIGN;
    if (!prefix || prefix >= entry->scale_bytes ||
        (entry->scale_bytes & (WEX_ALIGN - 1u))) return 0;
    return (size_t)(entry->scale_bytes - prefix);
}

/* Codec 4 (WNR2) or 5 (WNR3): records decoded with the container's WNM2 model. */
static inline int wex_entry_nr_model(const WexEntry *e) {
    return e && (e->flags == WEX_ENTRY_NEURAL_RANS2 || e->flags == WEX_ENTRY_NEURAL_RANS3);
}
/* Record magic of a codec 4/5 entry. */
static inline uint32_t wex_entry_nr_magic(const WexEntry *e) {
    return e && e->flags == WEX_ENTRY_NEURAL_RANS3 ? WEX_NR3_MAGIC : WEX_NR2_MAGIC;
}
/* WNR3 owner check: 1 unless a codec-5 record names another (layer, expert)
 * (record bytes 28..31 = layer << 16 | expert) or its canonical-zero flag (byte 27
 * bit 3) differs from the container's (header_flags & WEX_HEADER_CANONICAL_ZERO). */
static inline int wex_nr3_owner_ok(const WexEntry *e, const unsigned char *record, uint32_t layer, uint32_t expert,
                                   uint32_t header_flags) {
    if (!e || e->flags != WEX_ENTRY_NEURAL_RANS3) return 1;
    return record && wex_read_u32_le(record + 28u) == (layer << 16 | expert) &&
           !(record[27] & WEX_NR3_FLAG_CZ) == !(header_flags & WEX_HEADER_CANONICAL_ZERO);
}

/* WNR2: per-matrix counts of an entry (the caller has validated its sizes). */
static inline void wex_nr2_layout(const WexEntry *entry, WexNr2Layout *layout) {
    unsigned int m;
    for (m = 0; m < 3u; m++) {
        layout->scale_count[m] = (uint32_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1];
        layout->symbol_count[m] = (uint32_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1] * 2u;
    }
}

/* WNR2 containers name their model in the header page, at byte 64:
 * {u32 "WNMD", u32 bytes, u64 offset, u32 model id, u32 0}. The model lies in
 * [offset, offset + bytes), 4096-aligned, between the header page and the
 * entry table; the id is the FNV-1a hash every record repeats. */
#define WEX_NR2_DESCRIPTOR_MAGIC UINT32_C(0x444d4e57)
#define WEX_NR2_DESCRIPTOR_OFFSET 64u
static inline int wex_nr2_model_location(const WexHeader *hdr, const unsigned char *header_page,
                                         uint64_t *offset, uint32_t *bytes, uint32_t *id) {
    const unsigned char *d;
    uint64_t at, aligned;
    if (!hdr || !header_page || hdr->magic != WEX_MAGIC_V3 || hdr->version != 3u ||
        !(hdr->flags & (WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_NEURAL_RANS3))) return -1;
    d = header_page + WEX_NR2_DESCRIPTOR_OFFSET;
    at = (uint64_t)wex_read_u32_le(d + 8u) | (uint64_t)wex_read_u32_le(d + 12u) << 32;
    if (wex_read_u32_le(d) != WEX_NR2_DESCRIPTOR_MAGIC || wex_read_u32_le(d + 4u) != WEX_NR2_MODEL_BYTES ||
        !wex_read_u32_le(d + 16u) || wex_read_u32_le(d + 20u) || at < WEX_ALIGN || (at & (WEX_ALIGN - 1u)) ||
        wex_aligned_size(WEX_NR2_MODEL_BYTES, &aligned) || at > hdr->table_offset ||
        aligned > hdr->table_offset - at) return -1;
    *offset = at;
    *bytes = WEX_NR2_MODEL_BYTES;
    *id = wex_read_u32_le(d + 16u);
    return 0;
}

static inline int wex_validate_entry(const WexHeader *hdr,
                                     const WexEntry *entry,
                                     uint64_t file_bytes) {
    uint64_t raw, aligned_raw, table_end, prefix, disk_bytes, aligned_disk;
    size_t table_bytes, aligned_table_bytes;
    unsigned int byte;
    if (!entry || wex_validate_header(hdr, file_bytes, 0, 0,
                                      &table_bytes, &aligned_table_bytes))
        return -1;
    if (!entry->total_bytes) {
        const unsigned char *p = (const unsigned char *)entry;
        for (byte = 0; byte < sizeof(*entry); byte++)
            if (p[byte]) return -1;
        return 0; /* dense/missing entry is represented by all zero bytes */
    }
    if (wex_entry_raw_sizes(entry, &raw, &aligned_raw) ||
        aligned_raw > hdr->aligned_expert_bytes ||
        (entry->total_bytes & (WEX_ALIGN - 1u)) ||
        (entry->file_offset & (WEX_ALIGN - 1u))) return -1;
    table_end = hdr->table_offset + aligned_table_bytes;
    if (entry->file_offset < table_end || entry->file_offset > file_bytes ||
        entry->total_bytes > file_bytes - entry->file_offset) return -1;
    if (!entry->flags) return entry->total_bytes == aligned_raw ? 0 : -1;
    if (entry->flags == WEX_ENTRY_NEURAL_RANS) {
        uint64_t streams = ((uint64_t)entry->weight_bytes * 2u + WEX_NR_CHUNK_SYMBOLS - 1u) / WEX_NR_CHUNK_SYMBOLS;
        uint64_t minimum = 32u + 512u + WEX_NR_CDF_BYTES + (streams + 1u) * 4u + streams * 4u +
                           ((uint64_t)entry->scale_bytes + 255u) / 256u * 2u;
        return hdr->version == 3u && (hdr->flags & WEX_HEADER_NEURAL_RANS) &&
               !wex_aligned_size(minimum, &aligned_disk) && entry->total_bytes >= aligned_disk &&
               entry->total_bytes < aligned_raw ? 0 : -1;
    }
    if (entry->flags == WEX_ENTRY_NEURAL_RANS2 || entry->flags == WEX_ENTRY_NEURAL_RANS3) {
        /* smallest legal record: longest chunks, 4-byte streams */
        uint32_t need = entry->flags == WEX_ENTRY_NEURAL_RANS3 ? WEX_HEADER_NEURAL_RANS3 : WEX_HEADER_NEURAL_RANS2;
        uint64_t streams = 0, minimum;
        unsigned int m;
        for (m = 0; m < 3u; m++)
            streams += wex_nr2_streams((uint32_t)entry->scale_shapes[m][0] * entry->scale_shapes[m][1], WEX_NR2_MAX_CHUNK) +
                       wex_nr2_streams((uint32_t)entry->weight_shapes[m][0] * entry->weight_shapes[m][1] * 2u, WEX_NR2_MAX_CHUNK);
        minimum = WEX_NR2_HEADER_BYTES + streams * 6u;
        return hdr->version == 3u && (hdr->flags & need) &&
               !wex_aligned_size(minimum, &aligned_disk) && entry->total_bytes >= aligned_disk &&
               entry->total_bytes < aligned_raw ? 0 : -1;
    }
    if (entry->flags == WEX_ENTRY_PALETTE3) {
        uint64_t minimum = WEX_PALETTE3_PREFIX_BYTES +
                           ((uint64_t)entry->scale_bytes + 255u) / 256u * 2u +
                           (uint64_t)entry->weight_bytes / 4u * 3u;
        return hdr->magic == WEX_MAGIC_V3 && hdr->version == 3u &&
               (hdr->flags & WEX_HEADER_PALETTE3) &&
               !wex_aligned_size(minimum, &aligned_disk) &&
               entry->total_bytes >= aligned_disk && entry->total_bytes < aligned_raw &&
               wex_record_read_offset(entry) + WEX_PALETTE3_PREFIX_BYTES >= entry->scale_bytes ? 0 : -1;
    }
    if (!((hdr->magic == WEX_MAGIC_V2 && hdr->version == 2u) ||
          (hdr->magic == WEX_MAGIC_V3 && hdr->version == 3u)) ||
        !(hdr->flags & WEX_HEADER_SCALE_TILES) ||
        (entry->flags & WEX_ENTRY_CODEC_MASK) != WEX_ENTRY_SCALE_TILES ||
        !wex_record_read_offset(entry)) return -1;
    prefix = (uint64_t)(entry->flags >> WEX_ENTRY_SCALE_PAGES_SHIFT) * WEX_ALIGN;
    disk_bytes = prefix + entry->weight_bytes;
    if (wex_aligned_size(disk_bytes, &aligned_disk) ||
        aligned_disk != entry->total_bytes || aligned_disk >= aligned_raw ||
        wex_record_read_offset(entry) + aligned_disk > aligned_raw)
        return -1;
    return 0;
}

/* Validate V3 aliases once at open, not in the cache/lookup path. New physical
 * records are canonical and contiguous; aliases must match all 72 entry bytes.
 * V1/V2 retain their existing per-entry compatibility rules, and one bound on
 * the header's aligned_expert_bytes. A reader allocates slabs of that size (the
 * standalone store: 24 per layer when it opens), and the per-entry rule only
 * says that no record decodes to more than it, so a V1/V2 header could state
 * any multiple of 4096 below 4 GiB whatever its records were. V3 requires the
 * field to equal the largest aligned decoded record. V1/V2 never did, and a
 * writer may round its slab up further (to a larger page, to a power of two):
 * the field is accepted up to twice the largest aligned decoded record of the
 * table, which covers a power of two and any granule no larger than that
 * record. Above it the table does not explain the field and the container is
 * refused. A table without records needs no slab: one page, the smallest value
 * wex_validate_header accepts.
 */
static inline int wex_validate_table(const WexHeader *hdr, const WexEntry *entries,
                                      uint64_t file_bytes) {
    size_t bytes, aligned_table, count, used = 0, i;
    uint64_t next, maximum = 0, largest = 0;
    const WexEntry **physical = NULL;
    int result = -1;
    if (!entries || wex_validate_header(hdr, file_bytes, 0, 0, &bytes, &aligned_table)) return -1;
    count = bytes / sizeof(*entries);
    next = hdr->table_offset + aligned_table;
    if (hdr->version == 3u) {
        physical = (const WexEntry **)malloc(count * sizeof(*physical));
        if (!physical) return -1;
    }
    for (i = 0; i < count; i++) {
        const WexEntry *entry = &entries[i];
        if (wex_validate_entry(hdr, entry, file_bytes)) goto finish;
        if (!entry->total_bytes) continue;
        if (!physical) {
            uint64_t raw, aligned;   /* at most UINT32_MAX each: twice it cannot wrap */
            if (wex_entry_raw_sizes(entry, &raw, &aligned)) goto finish;
            if (aligned > largest) largest = aligned;
            continue;
        }
        if (entry->file_offset == next) {
            uint64_t raw, aligned;
            physical[used++] = entry;
            next += entry->total_bytes;
            if (wex_entry_raw_sizes(entry, &raw, &aligned)) goto finish;
            if (aligned > maximum) maximum = aligned;
        } else {
            size_t lo = 0, hi = used;
            if (!(hdr->flags & WEX_HEADER_DEDUP)) goto finish;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2u;
                if (physical[mid]->file_offset < entry->file_offset) lo = mid + 1u;
                else hi = mid;
            }
            if (lo == used || memcmp(physical[lo], entry, sizeof(*entry))) goto finish;
        }
    }
    if (physical && (next != file_bytes || maximum != hdr->aligned_expert_bytes)) goto finish;
    if (!physical && hdr->aligned_expert_bytes > (largest ? 2u * largest : WEX_ALIGN)) goto finish;
    result = 0;
finish:
    free(physical);
    return result;
}

/* The caller has proved base + ((1 << bits) - 1) <= 255. Eight values occupy
 * exactly 'bits' bytes; fixed-width groups avoid per-value variable shifts and
 * crossing tests. The generic tail never reads past a partial tile's payload.
 */
static inline void wex_unpack_bounded_tile(const unsigned char *src,
                                           unsigned char *dst, size_t count,
                                           unsigned int base, unsigned int bits) {
    size_t index = 0, full = count & ~(size_t)7u;
    unsigned int mask = (1u << bits) - 1u;
#define WEX_TILE_VALUE(B, N) \
    ((src[((N) * (B)) / 8u] >> (((N) * (B)) % 8u)) | \
     ((((N) * (B)) % 8u + (B) > 8u) ? \
      (unsigned int)src[((N) * (B)) / 8u + 1u] << (8u - (((N) * (B)) % 8u)) : 0u))
#define WEX_TILE_STORE(B, N) \
    dst[index + (N)] = (unsigned char)(base + (WEX_TILE_VALUE(B, N) & ((1u << (B)) - 1u)))
#define WEX_TILE_CASE(B) \
    case B: \
        for (; index < full; index += 8u, src += (B)) { \
            WEX_TILE_STORE(B, 0); WEX_TILE_STORE(B, 1); \
            WEX_TILE_STORE(B, 2); WEX_TILE_STORE(B, 3); \
            WEX_TILE_STORE(B, 4); WEX_TILE_STORE(B, 5); \
            WEX_TILE_STORE(B, 6); WEX_TILE_STORE(B, 7); \
        } \
        break
    switch (bits) {
        WEX_TILE_CASE(1);
        WEX_TILE_CASE(2);
        WEX_TILE_CASE(3);
        WEX_TILE_CASE(4);
        WEX_TILE_CASE(5);
        WEX_TILE_CASE(6);
        WEX_TILE_CASE(7);
        case 8:
            if (!base) memcpy(dst, src, count);
            else for (; index < count; index++) dst[index] = (unsigned char)(base + src[index]);
            return;
    }
#undef WEX_TILE_CASE
#undef WEX_TILE_STORE
#undef WEX_TILE_VALUE
    for (; index < count; index++) {
        size_t bit = (index - full) * bits;
        size_t pos = bit >> 3;
        unsigned int shift = (unsigned int)(bit & 7u);
        unsigned int value = src[pos] >> shift;
        if (shift + bits > 8u) value |= (unsigned int)src[pos + 1u] << (8u - shift);
        dst[index] = (unsigned char)(base + (value & mask));
    }
}

/* Parse one tile stream. A NULL output performs validation only. */
static inline int wex_unpack_scale_tiles(const unsigned char *stream,
                                          size_t stream_bytes,
                                          unsigned char *output,
                                          size_t decoded_bytes) {
    size_t consumed = 0, produced = 0;
    if (!stream && (stream_bytes || decoded_bytes)) return -1;
    while (produced < decoded_bytes) {
        size_t count = decoded_bytes - produced;
        size_t payload, index;
        unsigned int base, bits, mask;
        if (count > WEX_SCALE_TILE_VALUES) count = WEX_SCALE_TILE_VALUES;
        if (stream_bytes - consumed < 2u) return -1;
        base = stream[consumed++];
        bits = stream[consumed++];
        if (bits > 8u) return -1;
        payload = (count * bits + 7u) / 8u;
        if (!bits) {
            if (output) memset(output + produced, (int)base, count);
            produced += count;
            continue;
        }
        if (!payload || payload > stream_bytes - consumed) return -1;
        mask = (1u << bits) - 1u;
        /* A final partial byte is canonical only when its unused bits are zero. */
        if (payload && ((count * bits) & 7u) &&
            (stream[consumed + payload - 1u] >> ((count * bits) & 7u)))
            return -1;
        if (base + mask <= UINT8_MAX) {
            /* Every representable residual is safe: validation needs only the
             * tile header, payload bounds and final padding, not each scale.
             */
            if (output)
                wex_unpack_bounded_tile(stream + consumed, output + produced, count, base, bits);
        } else {
            /* A large base is valid when all actual residuals fit. Preserve
             * this case, checking each value before narrowing to a byte.
             */
            for (index = 0; index < count; index++) {
                size_t bit = index * bits;
                size_t pos = bit >> 3;
                unsigned int shift = (unsigned int)(bit & 7u);
                unsigned int value = stream[consumed + pos] >> shift;
                if (shift + bits > 8u)
                    value |= (unsigned int)stream[consumed + pos + 1u] << (8u - shift);
                value &= mask;
                if (base + value > UINT8_MAX) return -1;
                if (output) output[produced + index] = (unsigned char)(base + value);
            }
        }
        consumed += payload;
        produced += count;
    }
    return consumed == stream_bytes ? 0 : -1;
}

static inline void wex_unpack_palette3_range(const WexEntry *entry,
    const unsigned char *stream, size_t stream_raw_base, size_t begin, size_t end,
    unsigned char *slab, const unsigned char tables[3][256][16]) {
    unsigned int matrix;
    for (matrix = 0; matrix < 3u; matrix++) {
        size_t matrix_start = entry->weight_offsets[matrix] - entry->scale_bytes;
        size_t matrix_end = matrix_start + (size_t)entry->weight_shapes[matrix][0] * entry->weight_shapes[matrix][1];
        size_t lo = begin > matrix_start ? begin : matrix_start;
        size_t hi = end < matrix_end ? end : matrix_end;
        if (lo < hi)
            wex_unpack_palette3(stream + (lo - stream_raw_base) / 4u * 3u,
                slab + entry->scale_bytes + lo, hi - lo,
                slab + entry->scale_offsets[matrix] + (lo - matrix_start) / 16u, tables[matrix]);
    }
}

/* WEX3 expands forward without copying the whole compressed weight payload.
 * Disk/raw padding can differ by at most 4095 bytes. Save the last 16 KiB of
 * decoded weights (12 KiB compressed) before writing: before that tail, the
 * remaining compression savings exceed the worst padding skew, so forward
 * stores cannot clobber unread input. Matrix
 * and scale-group boundaries are all multiples of 16 raw bytes.
 */
static inline int wex_decode_palette3_record(const WexEntry *entry,
                                              unsigned char *slab, size_t offset) {
    const unsigned char *header = slab + offset;
    const unsigned char *scales, *weights;
    unsigned char tail[3u * WEX_ALIGN];
    unsigned char packed_tables[WEX_PALETTE3_TABLE_BYTES];
    unsigned char tables[3][256][16];
    size_t scale_stream, weight_stream, tail_raw, tail_start;
    uint64_t disk, aligned;
    unsigned int matrix;
    if (entry->flags != WEX_ENTRY_PALETTE3 ||
        entry->total_bytes < WEX_PALETTE3_PREFIX_BYTES ||
        offset + WEX_PALETTE3_PREFIX_BYTES < entry->scale_bytes ||
        wex_read_u32_le(header) != WEX_PALETTE3_MAGIC || wex_read_u32_le(header + 12u)) return -1;
    scale_stream = wex_read_u32_le(header + 4u);
    weight_stream = wex_read_u32_le(header + 8u);
    disk = (uint64_t)WEX_PALETTE3_PREFIX_BYTES + scale_stream + weight_stream;
    if (!scale_stream || weight_stream != (uint64_t)entry->weight_bytes / 4u * 3u ||
        wex_aligned_size(disk, &aligned) || aligned != entry->total_bytes || disk > entry->total_bytes)
        return -1;
    scales = header + WEX_PALETTE3_PREFIX_BYTES;
    if (wex_unpack_scale_tiles(scales, scale_stream, NULL, entry->scale_bytes)) return -1;
    weights = scales + scale_stream;
    memcpy(packed_tables, header + WEX_PALETTE3_HEADER_BYTES, sizeof(packed_tables));
    tail_raw = entry->weight_bytes < 4u * WEX_ALIGN ? entry->weight_bytes : 4u * WEX_ALIGN;
    tail_start = entry->weight_bytes - tail_raw;
    memcpy(tail, weights + tail_start / 4u * 3u, tail_raw / 4u * 3u);
    /* The scale stream is above the complete scale destination. */
    if (wex_unpack_scale_tiles(scales, scale_stream, slab, entry->scale_bytes)) return -1;
    for (matrix = 0; matrix < 3u; matrix++)
        wex_palette_expand_table(packed_tables + matrix * 1024u, tables[matrix]);
    wex_unpack_palette3_range(entry, weights, 0, 0, tail_start, slab,
                               (const unsigned char (*)[256][16])tables);
    wex_unpack_palette3_range(entry, tail, tail_start, tail_start, entry->weight_bytes,
                               slab, (const unsigned char (*)[256][16])tables);
    return 0;
}

/* WNR2/WNR3 into a slab with the container's model; the slab is untouched
 * unless it returns 0 (decodes into scratch first, like the WNR1 reference).
 * The WNR3 owner check is the caller's (wex_nr3_owner_ok). */
static inline int wex_decode_nr2_record(const WexEntry *entry, unsigned char *slab, size_t capacity,
                                        const WexNr2Model *model) {
    uint64_t raw, aligned_raw;
    size_t read_offset;
    WexNr2Layout layout;
    WexNr2Record view;
    unsigned char *decoded, *scales[3], *weights[3];
    unsigned int m;
    int result;
    if (!entry || !slab || !model || !wex_entry_nr_model(entry) ||
        wex_entry_raw_sizes(entry, &raw, &aligned_raw) || aligned_raw > capacity) return -1;
    read_offset = wex_record_read_offset(entry);
    if (!read_offset || read_offset > capacity || entry->total_bytes > capacity - read_offset) return -1;
    wex_nr2_layout(entry, &layout);
    if (wex_nrx_parse(slab + read_offset, entry->total_bytes, &layout, model->id, wex_entry_nr_magic(entry), &view))
        return -1;
    decoded = (unsigned char *)malloc((size_t)raw);
    if (!decoded) return -1;
    for (m = 0; m < 3u; m++) {
        scales[m] = decoded + entry->scale_offsets[m];
        weights[m] = decoded + entry->weight_offsets[m];
    }
    result = wex_nr2_decode(&view, &layout, model, scales, weights);
    if (!result) memcpy(slab, decoded, (size_t)raw);
    free(decoded);
    return result;
}

/* The caller reads total_bytes at slab + wex_record_read_offset(entry).
 * V2 weights arrive at their final offsets; only scales need heap scratch.
 * V3 palette3 expands in place with bounded palette/tail stack workspace.
 * Invalid streams are rejected before any byte of the slab is overwritten.
 */
static inline int wex_decode_record(const WexEntry *entry, unsigned char *slab,
                                     size_t capacity) {
    uint64_t raw, aligned_raw, prefix, aligned_disk;
    size_t read_offset, stream_bytes;
    unsigned char *scratch;
    const unsigned char *header;
    int result;
    if (!entry || !slab || wex_entry_raw_sizes(entry, &raw, &aligned_raw) ||
        aligned_raw > capacity) return -1;
    if (!entry->flags) return entry->total_bytes == aligned_raw ? 0 : -1;
    if (wex_entry_nr_model(entry)) return -1;   /* needs its model: wex_decode_nr2_record */
    read_offset = wex_record_read_offset(entry);
    if (!read_offset) return -1;
    if (entry->flags == WEX_ENTRY_PALETTE3) {
        if (read_offset > capacity || entry->total_bytes > capacity - read_offset) return -1;
        return wex_decode_palette3_record(entry, slab, read_offset);
    }
    if (entry->flags == WEX_ENTRY_NEURAL_RANS) {
        WexNeuralRans view;
        unsigned char *decoded;
        if (read_offset > capacity || entry->total_bytes > capacity - read_offset ||
            wex_nr_parse(slab + read_offset, entry->total_bytes, entry->weight_bytes, &view)) return -1;
        decoded = (unsigned char *)malloc((size_t)raw);
        if (!decoded) return -1;
        result = wex_unpack_scale_tiles(view.scales, view.scale_encoded, decoded, entry->scale_bytes);
        if (!result) result = wex_nr_decode_weights(&view, decoded, entry->weight_bytes, decoded + entry->scale_bytes);
        if (!result) memcpy(slab, decoded, (size_t)raw);
        free(decoded);
        return result;
    }
    prefix = (uint64_t)(entry->flags >> WEX_ENTRY_SCALE_PAGES_SHIFT) * WEX_ALIGN;
    if (wex_aligned_size(prefix + entry->weight_bytes, &aligned_disk) ||
        aligned_disk != entry->total_bytes || aligned_disk >= aligned_raw ||
        read_offset > capacity || entry->total_bytes > capacity - read_offset)
        return -1;
    header = slab + read_offset;
    if (wex_read_u32_le(header) != WEX_SCALE_STREAM_MAGIC ||
        wex_read_u32_le(header + 4u) != entry->scale_bytes) return -1;
    stream_bytes = wex_read_u32_le(header + 8u);
    if (!stream_bytes || stream_bytes > prefix - WEX_SCALE_STREAM_HEADER_BYTES)
        return -1;
    scratch = (unsigned char *)malloc(stream_bytes);
    if (!scratch) return -1;
    memcpy(scratch, header + WEX_SCALE_STREAM_HEADER_BYTES, stream_bytes);
    result = wex_unpack_scale_tiles(scratch, stream_bytes, NULL, entry->scale_bytes);
    if (!result)
        result = wex_unpack_scale_tiles(scratch, stream_bytes, slab, entry->scale_bytes);
    free(scratch);
    return result;
}

#endif /* COLI_WEX_FORMAT_H */
