#ifndef COLI_WEX_NEURAL_RANS_H
#define COLI_WEX_NEURAL_RANS_H
/* Exact-symbol conditional rANS using integer CDFs learned by a neural net.
 * Same byte-renormalized state equations as a plain byte-wise rANS. Contexts reset
 * every 32 weights, so independent chunks can encode/decode on CUDA.
 * This header contains the portable, strictly checked CPU reference.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef __CUDACC__
#define WEX_NR_HD __host__ __device__
#else
#define WEX_NR_HD
#endif
#define WEX_NR_MAGIC UINT32_C(0x31524e57)
#define WEX_NR_CHUNK_SYMBOLS 1024u
#define WEX_NR_L (1u << 23)
#define WEX_NR_CDF_BYTES (17u * 17u * 2u)

typedef struct {
    const unsigned char *mapping, *cdf, *offsets, *scales, *payload;
    uint32_t scale_encoded, payload_bytes, classes, streams, metadata_bytes, chunk_symbols, scale_only;
} WexNeuralRans;

static inline uint32_t wex_nr_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static inline uint32_t wex_nr_u16(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

static inline int wex_nr_parse(const unsigned char *record, size_t total,
    uint32_t weight_bytes, WexNeuralRans *view) {
    uint64_t meta, disk;
    uint32_t classes, streams, chunk, i, previous;
    if (!record || !view || total < 32u || wex_nr_u32(record) != WEX_NR_MAGIC ||
        wex_nr_u32(record + 24u) > 1u || wex_nr_u32(record + 28u)) return -1;
    chunk = wex_nr_u32(record + 20u);
    if (chunk != 256u && chunk != 512u && chunk != 1024u) return -1;
    classes = wex_nr_u32(record + 12u); streams = wex_nr_u32(record + 16u);
    if (!classes || classes > 256u || !streams ||
        streams != ((uint64_t)weight_bytes * 2u + chunk - 1u) / chunk) return -1;
    meta = 32u + 512u + (uint64_t)classes * WEX_NR_CDF_BYTES + ((uint64_t)streams + 1u) * 4u;
    disk = meta + wex_nr_u32(record + 4u) + wex_nr_u32(record + 8u);
    if (!wex_nr_u32(record + 4u) || meta > total || disk > total ||
        ((disk + 4095u) & ~UINT64_C(4095)) != total) return -1;
    view->classes = classes; view->streams = streams; view->metadata_bytes = (uint32_t)meta;
    view->chunk_symbols = chunk;
    view->scale_only = wex_nr_u32(record + 24u);
    view->scale_encoded = wex_nr_u32(record + 4u); view->payload_bytes = wex_nr_u32(record + 8u);
    view->mapping = record + 32u; view->cdf = record + 544u;
    view->offsets = view->cdf + classes * WEX_NR_CDF_BYTES;
    view->scales = record + (size_t)meta; view->payload = view->scales + view->scale_encoded;
    for (i = 0; i < 256u; i++) {
        uint32_t value = wex_nr_u16(view->mapping + i * 2u);
        if (value != 65535u && value >= classes) return -1;
    }
    for (i = 0; i < classes * 17u; i++) {
        const unsigned char *cdf = view->cdf + i * 34u;
        uint32_t symbol;
        if (wex_nr_u16(cdf) || wex_nr_u16(cdf + 32u) != 4096u) return -1;
        for (symbol = 0; symbol < 16u; symbol++)
            if (wex_nr_u16(cdf + symbol * 2u) >= wex_nr_u16(cdf + symbol * 2u + 2u)) return -1;
        if (view->scale_only) {
            uint32_t byte;
            const unsigned char *first_cdf = view->cdf + (i / 17u) * WEX_NR_CDF_BYTES;
            for (byte = 0; byte < 34u; byte++) if (cdf[byte] != first_cdf[byte]) return -1;
        }
    }
    previous = wex_nr_u32(view->offsets);
    if (previous) return -1;
    for (i = 0; i < streams; i++) {
        uint32_t next = wex_nr_u32(view->offsets + (size_t)(i + 1u) * 4u);
        if (next < previous || next - previous < 4u || next > view->payload_bytes) return -1;
        previous = next;
    }
    return previous == view->payload_bytes ? 0 : -1;
}

static inline int wex_nr_decode_weights(const WexNeuralRans *v,
    const unsigned char *scales, uint32_t weight_bytes, unsigned char *output) {
    uint32_t stream;
    for (stream = 0; stream < v->streams; stream++) {
        const unsigned char *ptr = v->payload + wex_nr_u32(v->offsets + (size_t)stream * 4u);
        const unsigned char *end = v->payload + wex_nr_u32(v->offsets + (size_t)(stream + 1u) * 4u);
        uint32_t state = (uint32_t)ptr[0] << 24 | (uint32_t)ptr[1] << 16 | (uint32_t)ptr[2] << 8 | ptr[3];
        uint32_t previous = 16u, j;
        uint64_t first = (uint64_t)stream * v->chunk_symbols;
        ptr += 4;
        if (state < WEX_NR_L || state >= (WEX_NR_L << 8)) return -1;
        for (j = 0; j < v->chunk_symbols && first + j < (uint64_t)weight_bytes * 2u; j++) {
            size_t position = (size_t)(first + j);
            uint32_t cls = wex_nr_u16(v->mapping + (size_t)scales[position / 32u] * 2u);
            const unsigned char *cdf;
            uint32_t slot = state & 4095u, lo = 0, hi = 16u, start, frequency;
            if (!(j % 32u) || v->scale_only) previous = 16u;
            if (cls >= v->classes) return -1;
            cdf = v->cdf + ((size_t)cls * 17u + previous) * 34u;
            while (lo + 1u < hi) {
                uint32_t mid = (lo + hi) / 2u;
                if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid;
                else hi = mid;
            }
            start = wex_nr_u16(cdf + lo * 2u);
            frequency = wex_nr_u16(cdf + lo * 2u + 2u) - start;
            state = frequency * (state >> 12) + slot - start;
            while (state < WEX_NR_L) {
                if (ptr == end) return -1;
                state = (state << 8) | *ptr++;
            }
            if (!(position & 1u)) output[position / 2u] = (unsigned char)lo;
            else output[position / 2u] |= (unsigned char)(lo << 4);
            previous = lo;
        }
        if (ptr != end || state != WEX_NR_L) return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * WNR2: the same exact rANS, with the probability model moved out of the
 * records into one container-level model blob ("WNM2", trained on the experts
 * of that container), richer contexts and long streams.
 *
 * Weights: matrix m (gate w1, down w2, up w3), symbol i in row-major order,
 * 32-weight block b = i / 32 whose UE8M0 scale is s. At each block start
 * srel = clamp(s - median_m, -3, 3) + 3, previous magnitude pm = 8 (none) and
 * block maximum so far mx = 8 (none); after symbol x, pm = x & 7 and mx =
 * max(mx, x & 7). Context = (srel * 9 + pm) * 9 + mx, one of 567, each with a
 * 16-symbol CDF. Scales: one 256-symbol CDF per matrix. All CDFs are integer,
 * total 1 << 14, every symbol encodable (frequency >= 1).
 *
 * Record: 32-byte header {magic, model_id, scale_payload_bytes,
 * weight_payload_bytes, weight_chunk, scale_chunk, median[3], 0, 0}, then
 * u16 lengths of every scale stream and every weight stream, the scale
 * payload and the weight payload, zero-padded to 4096. Streams never cross a
 * matrix: matrix m has ceil(count_m / chunk) of them. Each stream is a 4-byte
 * big-endian initial state plus renormalization bytes; it must end exactly at
 * its length with the state back at WEX_NR_L.
 * ------------------------------------------------------------------------- */
#define WEX_NR2_MAGIC UINT32_C(0x32524e57)        /* "WNR2" record */
#define WEX_NR2_MODEL_MAGIC UINT32_C(0x324d4e57)  /* "WNM2" model */
#define WEX_NR2_PRECISION 14u
#define WEX_NR2_TOTAL (1u << WEX_NR2_PRECISION)
#define WEX_NR2_CONTEXTS 567u
#define WEX_NR2_HEADER_BYTES 32u
#define WEX_NR2_MODEL_HEADER_BYTES 32u
#define WEX_NR2_WCDF_BYTES (3u * WEX_NR2_CONTEXTS * 17u * 2u)
#define WEX_NR2_SCDF_BYTES (3u * 257u * 2u)
#define WEX_NR2_MODEL_BYTES (WEX_NR2_MODEL_HEADER_BYTES + WEX_NR2_WCDF_BYTES + WEX_NR2_SCDF_BYTES)
#define WEX_NR2_MIN_WEIGHT_CHUNK 512u
#define WEX_NR2_MIN_SCALE_CHUNK 256u
#define WEX_NR2_MAX_CHUNK 16384u

typedef struct {
    uint32_t scale_count[3];    /* UE8M0 scales of each matrix */
    uint32_t symbol_count[3];   /* 4-bit weights of each matrix (two per byte) */
} WexNr2Layout;

typedef struct {
    const unsigned char *blob;  /* WEX_NR2_MODEL_BYTES, validated */
    uint32_t id;
} WexNr2Model;

typedef struct {
    const unsigned char *lengths, *scale_payload, *weight_payload;
    uint32_t model_id, scale_payload_bytes, weight_payload_bytes;
    uint32_t weight_chunk, scale_chunk, scale_streams, weight_streams, metadata_bytes;
    unsigned char median[3];    /* WNR2 median, WNR3 rule-g anchor */
    unsigned char flags;        /* WNR3 record byte 27 (WNR2: 0) */
    uint32_t magic, owner, word24;   /* WNR3: owner = layer << 16 | expert, word24 = bytes 24..27 LE */
} WexNr2Record;

/* ---------------------------------------------------------------------------
 * WNR3 (codec 5): WNR2's record skeleton and WNM2 model, plus
 *  - ceil form (record flag bit m = matrix m): an "even" block (max magnitude
 *    7, no magnitude 1 or 3, scale <= 254) is stored halved (0,2,4,5,6,7 ->
 *    0..5) with scale + 1; the decoder undoes every block whose coded max is 5;
 *  - canonical zero (flag CZ): no coded nibble 8 (-0 stored as +0);
 *  - per-stream integrity hash: the stream's initial state is L + h23, h23 =
 *    MurmurHash3_x86_32({model_id, owner, g, bytes 24..27} || the stream's
 *    decoded coded bytes, seed WEX_NR3_HASH_SEED) & 0x7fffff, and the decoder
 *    requires that final state.
 * Record bytes 24..26 = rule-g anchors of the coded scales, 27 = flags,
 * 28..31 = owner. Contexts and hashes use the coded domain only.
 * ------------------------------------------------------------------------- */
#define WEX_NR3_MAGIC UINT32_C(0x33524e57)        /* "WNR3" record */
#define WEX_NR3_FLAG_CEIL_MASK 0x07u
#define WEX_NR3_FLAG_CZ 0x08u
#define WEX_NR3_HASH_SEED UINT32_C(0x33524e57)
#define WEX_NR3_HASH_MASK UINT32_C(0x007fffff)

static WEX_NR_HD inline uint32_t wex_nr3_rotl(uint32_t x, int r) { return x << r | x >> (32 - r); }
static WEX_NR_HD inline uint32_t wex_nr3_h32_tail(uint32_t h, uint32_t k) {
    k *= UINT32_C(0xcc9e2d51); k = wex_nr3_rotl(k, 15); k *= UINT32_C(0x1b873593);
    return h ^ k;
}
static WEX_NR_HD inline uint32_t wex_nr3_h32_block(uint32_t h, uint32_t k) {
    h = wex_nr3_rotl(wex_nr3_h32_tail(h, k), 13);
    return h * 5u + UINT32_C(0xe6546b64);
}
static WEX_NR_HD inline uint32_t wex_nr3_h32_final(uint32_t h, uint32_t len) {
    h ^= len;
    h ^= h >> 16; h *= UINT32_C(0x85ebca6b); h ^= h >> 13; h *= UINT32_C(0xc2b2ae35); h ^= h >> 16;
    return h;
}
/* MurmurHash3_x86_32 of p[0..n) continuing from h after `done` bytes. */
static inline uint32_t wex_nr3_h32_bytes(uint32_t h, const unsigned char *p, size_t n, uint32_t done) {
    size_t i;
    for (i = 0; i + 4u <= n; i += 4u) h = wex_nr3_h32_block(h, wex_nr_u32(p + i));
    if (n & 3u) {
        uint32_t k = 0;
        size_t t;
        for (t = n & 3u; t-- > 0;) k = k << 8 | p[i + t];
        h = wex_nr3_h32_tail(h, k);
    }
    return wex_nr3_h32_final(h, (uint32_t)(done + n));
}
static inline uint32_t wex_nr3_h32(const unsigned char *p, size_t n, uint32_t seed) {
    return wex_nr3_h32_bytes(seed, p, n, 0);
}
static WEX_NR_HD inline uint32_t wex_nr3_ident_hash(const uint32_t ident[4]) {
    uint32_t h = WEX_NR3_HASH_SEED;
    h = wex_nr3_h32_block(h, ident[0]); h = wex_nr3_h32_block(h, ident[1]);
    h = wex_nr3_h32_block(h, ident[2]); return wex_nr3_h32_block(h, ident[3]);
}
/* L + h23 of one stream: ident = {model_id, owner, g, word24}, data = its coded bytes. */
static inline uint32_t wex_nr3_stream_x0(const uint32_t ident[4], const unsigned char *data, uint32_t n) {
    return WEX_NR_L + (wex_nr3_h32_bytes(wex_nr3_ident_hash(ident), data, n, 16u) & WEX_NR3_HASH_MASK);
}

static inline uint32_t wex_nr2_fnv1a(const unsigned char *p, size_t n) {
    uint32_t h = UINT32_C(2166136261);
    size_t i;
    for (i = 0; i < n; i++) { h ^= p[i]; h *= UINT32_C(16777619); }
    return h ? h : 1u;   /* 0 never names a model */
}

static inline const unsigned char *wex_nr2_weight_cdf(const WexNr2Model *m, uint32_t matrix, uint32_t context) {
    return m->blob + WEX_NR2_MODEL_HEADER_BYTES + ((size_t)matrix * WEX_NR2_CONTEXTS + context) * 34u;
}
static inline const unsigned char *wex_nr2_scale_cdf(const WexNr2Model *m, uint32_t matrix) {
    return m->blob + WEX_NR2_MODEL_HEADER_BYTES + WEX_NR2_WCDF_BYTES + (size_t)matrix * 514u;
}

static inline int wex_nr2_cdf_ok(const unsigned char *cdf, uint32_t symbols) {
    uint32_t s;
    if (wex_nr_u16(cdf) || wex_nr_u16(cdf + symbols * 2u) != WEX_NR2_TOTAL) return 0;
    for (s = 0; s < symbols; s++)
        if (wex_nr_u16(cdf + s * 2u) >= wex_nr_u16(cdf + s * 2u + 2u)) return 0;
    return 1;
}

/* Validate a model blob; model->id is its FNV-1a hash. */
static inline int wex_nr2_model_open(const unsigned char *blob, size_t bytes, WexNr2Model *model) {
    uint32_t i;
    WexNr2Model m;
    if (!blob || !model || bytes != WEX_NR2_MODEL_BYTES ||
        wex_nr_u32(blob) != WEX_NR2_MODEL_MAGIC || wex_nr_u32(blob + 4u) != 1u ||
        wex_nr_u32(blob + 8u) != WEX_NR2_PRECISION || wex_nr_u32(blob + 12u) != WEX_NR2_CONTEXTS ||
        wex_nr_u32(blob + 16u) != 3u || wex_nr_u32(blob + 20u) || wex_nr_u32(blob + 24u) ||
        wex_nr_u32(blob + 28u)) return -1;
    m.blob = blob; m.id = 0;
    for (i = 0; i < 3u * WEX_NR2_CONTEXTS; i++)
        if (!wex_nr2_cdf_ok(wex_nr2_weight_cdf(&m, i / WEX_NR2_CONTEXTS, i % WEX_NR2_CONTEXTS), 16u)) return -1;
    for (i = 0; i < 3u; i++)
        if (!wex_nr2_cdf_ok(wex_nr2_scale_cdf(&m, i), 256u)) return -1;
    m.id = wex_nr2_fnv1a(blob, bytes);
    *model = m;
    return 0;
}

static inline uint32_t wex_nr2_streams(uint32_t count, uint32_t chunk) {
    return (uint32_t)(((uint64_t)count + chunk - 1u) / chunk);
}
static inline int wex_nr2_pow2_in(uint32_t v, uint32_t lo, uint32_t hi) {
    return v >= lo && v <= hi && !(v & (v - 1u));
}

/* Parse and validate a record (magic WNR2 or WNR3) against its layout;
 * expected_model 0 skips the model check (the caller then compares
 * view->model_id itself). WNR3: reserved flag bits must be 0; the owner is
 * checked by the caller. */
static inline int wex_nrx_parse(const unsigned char *record, size_t total, const WexNr2Layout *layout,
                                uint32_t expected_model, uint32_t magic, WexNr2Record *view) {
    uint64_t meta, disk, sum = 0;
    uint32_t m, s, scale_streams = 0, weight_streams = 0;
    WexNr2Record v;
    if (!record || !layout || !view || total < WEX_NR2_HEADER_BYTES || (total & 4095u) ||
        (magic != WEX_NR2_MAGIC && magic != WEX_NR3_MAGIC) || wex_nr_u32(record) != magic) return -1;
    if (magic == WEX_NR2_MAGIC ? (record[27] || wex_nr_u32(record + 28u))
                               : (record[27] & ~(WEX_NR3_FLAG_CEIL_MASK | WEX_NR3_FLAG_CZ)) != 0) return -1;
    v.magic = magic;
    v.flags = record[27];
    v.owner = wex_nr_u32(record + 28u);
    v.word24 = wex_nr_u32(record + 24u);
    v.model_id = wex_nr_u32(record + 4u);
    if (!v.model_id || (expected_model && v.model_id != expected_model)) return -1;
    v.scale_payload_bytes = wex_nr_u32(record + 8u);
    v.weight_payload_bytes = wex_nr_u32(record + 12u);
    v.weight_chunk = wex_nr_u32(record + 16u);
    v.scale_chunk = wex_nr_u32(record + 20u);
    if (!wex_nr2_pow2_in(v.weight_chunk, WEX_NR2_MIN_WEIGHT_CHUNK, WEX_NR2_MAX_CHUNK) ||
        !wex_nr2_pow2_in(v.scale_chunk, WEX_NR2_MIN_SCALE_CHUNK, WEX_NR2_MAX_CHUNK)) return -1;
    for (m = 0; m < 3u; m++) {
        v.median[m] = record[24u + m];
        if (!layout->scale_count[m] || !layout->symbol_count[m] || layout->symbol_count[m] % 32u ||
            (uint64_t)layout->scale_count[m] * 32u != layout->symbol_count[m]) return -1;
        scale_streams += wex_nr2_streams(layout->scale_count[m], v.scale_chunk);
        weight_streams += wex_nr2_streams(layout->symbol_count[m], v.weight_chunk);
    }
    v.scale_streams = scale_streams; v.weight_streams = weight_streams;
    meta = WEX_NR2_HEADER_BYTES + 2u * ((uint64_t)scale_streams + weight_streams);
    disk = meta + v.scale_payload_bytes + v.weight_payload_bytes;
    if (disk > total || ((disk + 4095u) & ~UINT64_C(4095)) != total) return -1;
    v.metadata_bytes = (uint32_t)meta;
    v.lengths = record + WEX_NR2_HEADER_BYTES;
    v.scale_payload = record + meta;
    v.weight_payload = v.scale_payload + v.scale_payload_bytes;
    for (s = 0; s < scale_streams + weight_streams; s++) {
        uint32_t n = wex_nr_u16(v.lengths + (size_t)s * 2u);
        if (n < 4u) return -1;
        sum += n;
        if (s + 1u == scale_streams) {
            if (sum != v.scale_payload_bytes) return -1;
            sum = 0;
        }
    }
    if (sum != v.weight_payload_bytes) return -1;
    for (disk = meta + v.scale_payload_bytes + v.weight_payload_bytes; disk < total; disk++)
        if (record[disk]) return -1;   /* canonical zero padding */
    *view = v;
    return 0;
}
static inline int wex_nr2_parse(const unsigned char *record, size_t total, const WexNr2Layout *layout,
                                uint32_t expected_model, WexNr2Record *view) {
    return wex_nrx_parse(record, total, layout, expected_model, WEX_NR2_MAGIC, view);
}
static inline int wex_nr3_parse(const unsigned char *record, size_t total, const WexNr2Layout *layout,
                                uint32_t expected_model, WexNr2Record *view) {
    return wex_nrx_parse(record, total, layout, expected_model, WEX_NR3_MAGIC, view);
}

static inline uint32_t wex_nr2_context(uint32_t srel, uint32_t previous, uint32_t maximum) {
    return (srel * 9u + previous) * 9u + maximum;
}
static inline uint32_t wex_nr2_srel(uint32_t scale, uint32_t median) {
    int d = (int)scale - (int)median;
    return (uint32_t)((d < -3 ? -3 : d > 3 ? 3 : d) + 3);
}

/* Decode one symbol with a validated CDF (binary search). */
static inline int wex_nr2_decode_symbol(const unsigned char *cdf, uint32_t symbols, uint32_t *state,
                                        const unsigned char **ptr, const unsigned char *end, uint32_t *symbol) {
    uint32_t slot = *state & (WEX_NR2_TOTAL - 1u), lo = 0, hi = symbols, start, frequency;
    while (lo + 1u < hi) {
        uint32_t mid = (lo + hi) / 2u;
        if (wex_nr_u16(cdf + mid * 2u) <= slot) lo = mid; else hi = mid;
    }
    start = wex_nr_u16(cdf + lo * 2u);
    frequency = wex_nr_u16(cdf + lo * 2u + 2u) - start;
    *state = frequency * (*state >> WEX_NR2_PRECISION) + slot - start;
    while (*state < WEX_NR_L) {
        if (*ptr == end) return -1;
        *state = (*state << 8) | *(*ptr)++;
    }
    *symbol = lo;
    return 0;
}

static inline int wex_nr2_stream_open(const unsigned char *begin, uint32_t length,
                                      const unsigned char **ptr, const unsigned char **end, uint32_t *state) {
    if (length < 4u) return -1;
    *state = (uint32_t)begin[0] << 24 | (uint32_t)begin[1] << 16 | (uint32_t)begin[2] << 8 | begin[3];
    *ptr = begin + 4; *end = begin + length;
    return *state >= WEX_NR_L && *state < (WEX_NR_L << 8) ? 0 : -1;
}

/* ---- WNR3 transforms (CPU reference; the GPU decoder mirrors the inverse) -- */
static const unsigned char wex_nr3_halve[8] = {0, 0, 1, 0, 2, 3, 4, 5};   /* H on {0,2,4,5,6,7} */
static const unsigned char wex_nr3_double[8] = {0, 2, 4, 5, 6, 7, 0, 0};  /* D = a + min(a, 2) on 0..5 */

static inline uint32_t wex_nr3_block_max(const unsigned char *block16, uint32_t *odd) {
    uint32_t k, maximum = 0, o = 0;
    for (k = 0; k < 32u; k++) {
        uint32_t a = (block16[k / 2u] >> ((k & 1u) * 4u)) & 7u;
        if (a > maximum) maximum = a;
        o |= a == 1u || a == 3u;
    }
    if (odd) *odd = o;
    return maximum;
}

/* Rule g: lower value of the densest adjacent pair of the (coded) scales,
 * first maximum wins. Trainer and encoder use it; decoders read the byte. */
static inline unsigned char wex_nr3_anchor(const unsigned char *scales, uint32_t n) {
    uint32_t histogram[256] = {0}, i, best = 0, top = 0;
    for (i = 0; i < n; i++) histogram[scales[i]]++;
    for (i = 0; i < 255u; i++)
        if (histogram[i] + histogram[i + 1u] > top) { top = histogram[i] + histogram[i + 1u]; best = i; }
    return (unsigned char)best;
}

/* Canonical zero: every weight nibble 8 (-0) becomes 0; returns how many. */
static inline uint64_t wex_nr3_canonical_zero(unsigned char *weights, size_t bytes) {
    uint64_t mapped = 0;
    size_t i;
    for (i = 0; i < bytes; i++) {
        if ((weights[i] & 15u) == 8u) { weights[i] &= 0xf0u; mapped++; }
        if ((weights[i] >> 4) == 8u) { weights[i] &= 0x0fu; mapped++; }
    }
    return mapped;
}

/* Ceil form in place on the caller's copy. Matrix m is transformed only if no
 * block has max magnitude 5 (the bijection condition); returns the ceil mask.
 * stats (optional) [m] = {transformed blocks, max-5 blocks, even blocks kept at scale 255}. */
static inline uint32_t wex_nr3_ceil_forward(const WexNr2Layout *layout, unsigned char *const scales[3],
                                            unsigned char *const weights[3], uint32_t stats[3][3]) {
    uint32_t m, b, k, mask = 0;
    for (m = 0; m < 3u; m++) {
        uint32_t fives = 0, done = 0, kept = 0, odd;
        for (b = 0; b < layout->scale_count[m]; b++) fives += wex_nr3_block_max(weights[m] + (size_t)b * 16u, NULL) == 5u;
        if (!fives) {
            mask |= 1u << m;
            for (b = 0; b < layout->scale_count[m]; b++) {
                unsigned char *q = weights[m] + (size_t)b * 16u;
                if (wex_nr3_block_max(q, &odd) != 7u || odd) continue;
                if (scales[m][b] == 255u) { kept++; continue; }
                for (k = 0; k < 16u; k++)
                    q[k] = (unsigned char)((q[k] & 0x88u) | wex_nr3_halve[q[k] & 7u] | wex_nr3_halve[(q[k] >> 4) & 7u] << 4);
                scales[m][b]++;
                done++;
            }
        }
        if (stats) { stats[m][0] = done; stats[m][1] = fives; stats[m][2] = kept; }
    }
    return mask;
}

/* Inverse for every matrix in mask: blocks with coded max 5 are doubled back
 * and their scale decremented; -1 on a coded scale 0 in such a block. */
static inline int wex_nr3_ceil_inverse(const WexNr2Layout *layout, uint32_t mask, unsigned char *const scales[3],
                                       unsigned char *const weights[3]) {
    uint32_t m, b, k;
    for (m = 0; m < 3u; m++) {
        if (!(mask >> m & 1u)) continue;
        for (b = 0; b < layout->scale_count[m]; b++) {
            unsigned char *q = weights[m] + (size_t)b * 16u;
            if (wex_nr3_block_max(q, NULL) != 5u) continue;
            if (!scales[m][b]) return -1;
            for (k = 0; k < 16u; k++)
                q[k] = (unsigned char)((q[k] & 0x88u) | wex_nr3_double[q[k] & 7u] | wex_nr3_double[(q[k] >> 4) & 7u] << 4);
            scales[m][b]--;
        }
    }
    return 0;
}

/* CPU reference: scales of matrix m to scales_out[m], packed weights to
 * weights_out[m] (symbol 2k in the low nibble). Outputs may be partially
 * written when it fails; callers decode into scratch. WNR3 records (view
 * magic) also check each stream's hash, reject nibble 8 under CZ and undo
 * the ceil form. */
static inline int wex_nr2_decode(const WexNr2Record *v, const WexNr2Layout *layout, const WexNr2Model *model,
                                 unsigned char *const scales_out[3], unsigned char *const weights_out[3]) {
    const unsigned char *payload = v->scale_payload;
    uint32_t stream = 0, m, nr3 = v->magic == WEX_NR3_MAGIC, cz = nr3 && (v->flags & WEX_NR3_FLAG_CZ);
    uint32_t ident[4];
    if (v->model_id != model->id || (v->magic != WEX_NR2_MAGIC && !nr3)) return -1;
    ident[0] = v->model_id; ident[1] = v->owner; ident[3] = v->word24;
    for (m = 0; m < 3u; m++) {
        const unsigned char *cdf = wex_nr2_scale_cdf(model, m);
        uint32_t first;
        for (first = 0; first < layout->scale_count[m]; first += v->scale_chunk, stream++) {
            uint32_t length = wex_nr_u16(v->lengths + (size_t)stream * 2u), state, j, symbol;
            uint32_t count = layout->scale_count[m] - first < v->scale_chunk ? layout->scale_count[m] - first : v->scale_chunk;
            const unsigned char *ptr, *end;
            if (wex_nr2_stream_open(payload, length, &ptr, &end, &state)) return -1;
            for (j = 0; j < count; j++) {
                if (wex_nr2_decode_symbol(cdf, 256u, &state, &ptr, end, &symbol)) return -1;
                scales_out[m][first + j] = (unsigned char)symbol;
            }
            ident[2] = stream;
            if (ptr != end || state != (nr3 ? wex_nr3_stream_x0(ident, scales_out[m] + first, count) : WEX_NR_L)) return -1;
            payload += length;
        }
    }
    payload = v->weight_payload;
    for (m = 0; m < 3u; m++) {
        uint32_t first;
        for (first = 0; first < layout->symbol_count[m]; first += v->weight_chunk, stream++) {
            uint32_t length = wex_nr_u16(v->lengths + (size_t)stream * 2u), state, j, symbol;
            uint32_t count = layout->symbol_count[m] - first < v->weight_chunk ? layout->symbol_count[m] - first : v->weight_chunk;
            uint32_t srel = 0, previous = 8u, maximum = 8u;
            const unsigned char *ptr, *end;
            if (wex_nr2_stream_open(payload, length, &ptr, &end, &state)) return -1;
            for (j = 0; j < count; j++) {
                uint32_t i = first + j;
                if (!(i % 32u)) {
                    srel = wex_nr2_srel(scales_out[m][i / 32u], v->median[m]);
                    previous = 8u; maximum = 8u;
                }
                if (wex_nr2_decode_symbol(wex_nr2_weight_cdf(model, m, wex_nr2_context(srel, previous, maximum)),
                                          16u, &state, &ptr, end, &symbol) || (cz && symbol == 8u)) return -1;
                if (!(i & 1u)) weights_out[m][i / 2u] = (unsigned char)symbol;
                else weights_out[m][i / 2u] |= (unsigned char)(symbol << 4);
                previous = symbol & 7u;
                if (maximum == 8u || previous > maximum) maximum = previous;
            }
            ident[2] = stream;
            if (ptr != end || state != (nr3 ? wex_nr3_stream_x0(ident, weights_out[m] + first / 2u, count / 2u) : WEX_NR_L))
                return -1;
            payload += length;
        }
    }
    return nr3 ? wex_nr3_ceil_inverse(layout, v->flags & WEX_NR3_FLAG_CEIL_MASK, scales_out, weights_out) : 0;
}

/* ---- Reference encoder and model builder (tools and tests) ------------- */

/* Lower median of n UE8M0 values. */
static inline unsigned char wex_nr2_median(const unsigned char *scales, uint32_t n) {
    uint32_t histogram[256] = {0}, i, seen = 0;
    for (i = 0; i < n; i++) histogram[scales[i]]++;
    for (i = 0; i < 256u; i++) {
        seen += histogram[i];
        if (seen * 2u >= n) return (unsigned char)i;
    }
    return 255u;
}

/* Integer CDF (symbols + 1 u16 entries) from counts: every symbol gets at
 * least 1, the rest in proportion, remainders largest first (ties: lower
 * symbol). Deterministic; empty counts give a near-uniform CDF. */
static inline void wex_nr2_quantize(const uint64_t *counts, uint32_t symbols, unsigned char *cdf) {
    uint64_t total = 0, remainder[256];
    uint32_t frequency[256], spare = WEX_NR2_TOTAL - symbols, used = 0, s, extra;
    unsigned char taken[256] = {0};
    for (s = 0; s < symbols; s++) total += counts[s];
    for (s = 0; s < symbols; s++) {
        uint64_t scaled = total ? counts[s] * spare : spare;
        uint64_t divisor = total ? total : symbols;
        frequency[s] = 1u + (uint32_t)(scaled / divisor);
        remainder[s] = scaled % divisor;
        used += frequency[s];
    }
    for (extra = WEX_NR2_TOTAL - used; extra; extra--) {
        uint32_t best = symbols;
        for (s = 0; s < symbols; s++)
            if (!taken[s] && (best == symbols || remainder[s] > remainder[best])) best = s;
        taken[best] = 1; frequency[best]++;
    }
    used = 0;
    for (s = 0; s <= symbols; s++) {
        cdf[s * 2u] = (unsigned char)used; cdf[s * 2u + 1u] = (unsigned char)(used >> 8);
        if (s < symbols) used += frequency[s];
    }
}

/* Model blob from counts[3][567][16] (weights) and counts[3][256] (scales). */
static inline void wex_nr2_model_build(const uint64_t *weight_counts, const uint64_t *scale_counts,
                                       unsigned char *blob) {
    uint32_t i;
    static const uint32_t header[8] = {WEX_NR2_MODEL_MAGIC, 1u, WEX_NR2_PRECISION, WEX_NR2_CONTEXTS, 3u, 0, 0, 0};
    for (i = 0; i < 32u; i++) blob[i] = (unsigned char)(header[i / 4u] >> ((i % 4u) * 8u));
    for (i = 0; i < 3u * WEX_NR2_CONTEXTS; i++)
        wex_nr2_quantize(weight_counts + (size_t)i * 16u, 16u, blob + WEX_NR2_MODEL_HEADER_BYTES + (size_t)i * 34u);
    for (i = 0; i < 3u; i++)
        wex_nr2_quantize(scale_counts + (size_t)i * 256u, 256u,
                         blob + WEX_NR2_MODEL_HEADER_BYTES + WEX_NR2_WCDF_BYTES + (size_t)i * 514u);
}

/* Accumulate the training counts of one expert (same contexts as decode),
 * srel relative to anchor[m]. */
static inline void wex_nr2_count_anchored(const WexNr2Layout *layout, const unsigned char *const scales[3],
                                          const unsigned char *const weights[3], const unsigned char anchor[3],
                                          uint64_t *weight_counts, uint64_t *scale_counts) {
    uint32_t m, i;
    for (m = 0; m < 3u; m++) {
        unsigned char median = anchor[m];
        uint32_t srel = 0, previous = 8u, maximum = 8u;
        for (i = 0; i < layout->scale_count[m]; i++) scale_counts[m * 256u + scales[m][i]]++;
        for (i = 0; i < layout->symbol_count[m]; i++) {
            uint32_t x = (weights[m][i / 2u] >> ((i & 1u) * 4u)) & 15u;
            if (!(i % 32u)) { srel = wex_nr2_srel(scales[m][i / 32u], median); previous = 8u; maximum = 8u; }
            weight_counts[((size_t)m * WEX_NR2_CONTEXTS + wex_nr2_context(srel, previous, maximum)) * 16u + x]++;
            previous = x & 7u;
            if (maximum == 8u || previous > maximum) maximum = previous;
        }
    }
}
/* WNR2 training counts (median anchors). */
static inline void wex_nr2_count(const WexNr2Layout *layout, const unsigned char *const scales[3],
                                 const unsigned char *const weights[3], uint64_t *weight_counts,
                                 uint64_t *scale_counts) {
    unsigned char median[3];
    uint32_t m;
    for (m = 0; m < 3u; m++) median[m] = wex_nr2_median(scales[m], layout->scale_count[m]);
    wex_nr2_count_anchored(layout, scales, weights, median, weight_counts, scale_counts);
}
/* WNR3 training counts on CODED data (after CZ and ceil form): rule-g anchors. */
static inline void wex_nr3_count(const WexNr2Layout *layout, const unsigned char *const scales[3],
                                 const unsigned char *const weights[3], uint64_t *weight_counts,
                                 uint64_t *scale_counts) {
    unsigned char anchor[3];
    uint32_t m;
    for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(scales[m], layout->scale_count[m]);
    wex_nr2_count_anchored(layout, scales, weights, anchor, weight_counts, scale_counts);
}

/* Encode `count` symbols backwards into the stream ending at *cursor (bytes
 * grow downwards from there), CDF chosen per symbol by cdf_of(index). The
 * stream is then [state 4 B big-endian][renormalization bytes]. */
static inline int wex_nr2_put(uint32_t *state, unsigned char **cursor, const unsigned char *floor,
                              const unsigned char *cdf, uint32_t symbol) {
    uint32_t start = wex_nr_u16(cdf + symbol * 2u), frequency = wex_nr_u16(cdf + symbol * 2u + 2u) - start;
    uint32_t maximum = ((WEX_NR_L >> WEX_NR2_PRECISION) << 8) * frequency;
    while (*state >= maximum) {
        if (*cursor == floor) return -1;
        *--*cursor = (unsigned char)*state; *state >>= 8;
    }
    *state = ((*state / frequency) << WEX_NR2_PRECISION) + *state % frequency + start;
    return 0;
}
static inline int wex_nr2_flush(uint32_t state, unsigned char **cursor, const unsigned char *floor) {
    int b;
    for (b = 0; b < 4; b++) {
        if (*cursor == floor) return -1;
        *--*cursor = (unsigned char)state; state >>= 8;
    }
    return 0;
}

/* Reference encoder (WNR2 or WNR3 by magic): writes the unpadded record to
 * out and its size to *written. anchor = srel anchors (header bytes 24..26).
 * WNR3: flags/owner go to bytes 27..31, the data must already be coded (CZ,
 * ceil form) and each stream starts at L + h23. Scratch: 2 * WEX_NR2_MAX_CHUNK
 * + 8 bytes per stream at most. */
static inline int wex_nrx_encode(const WexNr2Model *model, const WexNr2Layout *layout,
                                 const unsigned char *const scales[3], const unsigned char *const weights[3],
                                 uint32_t weight_chunk, uint32_t scale_chunk, const unsigned char anchor[3],
                                 uint32_t magic, uint32_t flags, uint32_t owner,
                                 unsigned char *out, size_t capacity, size_t *written) {
    uint32_t m, streams = 0, scale_streams = 0, stream = 0, nr3 = magic == WEX_NR3_MAGIC, ident[4];
    unsigned char median[3], tmp[2u * WEX_NR2_MAX_CHUNK + 8u];
    size_t meta, cursor;
    uint64_t scale_bytes = 0, weight_bytes = 0;
    if (!model || !layout || !out || !written || !anchor ||
        (magic != WEX_NR2_MAGIC && !nr3) || (!nr3 && (flags || owner)) ||
        (flags & ~(WEX_NR3_FLAG_CEIL_MASK | WEX_NR3_FLAG_CZ)) ||
        !wex_nr2_pow2_in(weight_chunk, WEX_NR2_MIN_WEIGHT_CHUNK, WEX_NR2_MAX_CHUNK) ||
        !wex_nr2_pow2_in(scale_chunk, WEX_NR2_MIN_SCALE_CHUNK, WEX_NR2_MAX_CHUNK)) return -1;
    ident[0] = model->id; ident[1] = owner;
    ident[3] = (uint32_t)anchor[0] | (uint32_t)anchor[1] << 8 | (uint32_t)anchor[2] << 16 | flags << 24;
    for (m = 0; m < 3u; m++) {
        if (!layout->scale_count[m] || (uint64_t)layout->scale_count[m] * 32u != layout->symbol_count[m]) return -1;
        median[m] = anchor[m];
        scale_streams += wex_nr2_streams(layout->scale_count[m], scale_chunk);
        streams += wex_nr2_streams(layout->symbol_count[m], weight_chunk);
    }
    streams += scale_streams;
    meta = WEX_NR2_HEADER_BYTES + 2u * (size_t)streams;
    if (capacity < meta) return -1;
    cursor = meta;
    for (m = 0; m < 6u; m++) {
        uint32_t matrix = m % 3u, scale_pass = m < 3u, first;
        uint32_t count_all = scale_pass ? layout->scale_count[matrix] : layout->symbol_count[matrix];
        uint32_t chunk = scale_pass ? scale_chunk : weight_chunk;
        for (first = 0; first < count_all; first += chunk, stream++) {
            uint32_t count = count_all - first < chunk ? count_all - first : chunk, j, state = WEX_NR_L, length;
            unsigned char *end = tmp + sizeof(tmp), *p = end;
            uint16_t contexts[WEX_NR2_MAX_CHUNK];
            if (!scale_pass) {   /* forward contexts, backward coding */
                uint32_t srel = 0, previous = 8u, maximum = 8u;
                for (j = 0; j < count; j++) {
                    uint32_t i = first + j, x = (weights[matrix][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                    if (!(i % 32u)) { srel = wex_nr2_srel(scales[matrix][i / 32u], median[matrix]); previous = 8u; maximum = 8u; }
                    contexts[j] = (uint16_t)wex_nr2_context(srel, previous, maximum);
                    previous = x & 7u;
                    if (maximum == 8u || previous > maximum) maximum = previous;
                }
            }
            if (nr3) {
                ident[2] = stream;
                state = scale_pass ? wex_nr3_stream_x0(ident, scales[matrix] + first, count)
                                   : wex_nr3_stream_x0(ident, weights[matrix] + first / 2u, count / 2u);
            }
            for (j = count; j-- > 0;) {
                uint32_t i = first + j;
                if (scale_pass) {
                    if (wex_nr2_put(&state, &p, tmp, wex_nr2_scale_cdf(model, matrix), scales[matrix][i])) return -1;
                } else {
                    uint32_t x = (weights[matrix][i / 2u] >> ((i & 1u) * 4u)) & 15u;
                    if (wex_nr2_put(&state, &p, tmp, wex_nr2_weight_cdf(model, matrix, contexts[j]), x)) return -1;
                }
            }
            if (wex_nr2_flush(state, &p, tmp)) return -1;
            length = (uint32_t)(end - p);
            if (length > 65535u || capacity - cursor < length) return -1;
            memcpy(out + cursor, p, length);
            cursor += length;
            out[WEX_NR2_HEADER_BYTES + (size_t)stream * 2u] = (unsigned char)length;
            out[WEX_NR2_HEADER_BYTES + (size_t)stream * 2u + 1u] = (unsigned char)(length >> 8);
            if (scale_pass) scale_bytes += length; else weight_bytes += length;
        }
    }
    {
        uint32_t fields[8] = {magic, model->id, (uint32_t)scale_bytes, (uint32_t)weight_bytes,
                              weight_chunk, scale_chunk, ident[3], owner};
        uint32_t i;
        for (i = 0; i < 32u; i++) out[i] = (unsigned char)(fields[i / 4u] >> ((i % 4u) * 8u));
    }
    *written = cursor;
    return 0;
}
static inline int wex_nr2_encode(const WexNr2Model *model, const WexNr2Layout *layout,
                                 const unsigned char *const scales[3], const unsigned char *const weights[3],
                                 uint32_t weight_chunk, uint32_t scale_chunk,
                                 unsigned char *out, size_t capacity, size_t *written) {
    unsigned char median[3];
    uint32_t m;
    if (!layout || !scales) return -1;
    for (m = 0; m < 3u; m++) median[m] = wex_nr2_median(scales[m], layout->scale_count[m]);
    return wex_nrx_encode(model, layout, scales, weights, weight_chunk, scale_chunk, median, WEX_NR2_MAGIC, 0, 0,
                          out, capacity, written);
}
/* WNR3 encoder on CODED data (the caller applied CZ and wex_nr3_ceil_forward
 * and passes the resulting flags); rule-g anchors of the coded scales. */
static inline int wex_nr3_encode(const WexNr2Model *model, const WexNr2Layout *layout,
                                 const unsigned char *const scales[3], const unsigned char *const weights[3],
                                 uint32_t weight_chunk, uint32_t scale_chunk, uint32_t flags, uint32_t owner,
                                 unsigned char *out, size_t capacity, size_t *written) {
    unsigned char anchor[3];
    uint32_t m;
    if (!layout || !scales || !weights) return -1;
    for (m = 0; m < 3u; m++) {
        uint32_t i;
        anchor[m] = wex_nr3_anchor(scales[m], layout->scale_count[m]);
        if (flags & WEX_NR3_FLAG_CZ)   /* a CZ record must not carry -0: the decoder rejects it */
            for (i = 0; i < layout->symbol_count[m] / 2u; i++)
                if ((weights[m][i] & 15u) == 8u || (weights[m][i] >> 4) == 8u) return -1;
    }
    return wex_nrx_encode(model, layout, scales, weights, weight_chunk, scale_chunk, anchor, WEX_NR3_MAGIC, flags, owner,
                          out, capacity, written);
}
#endif
