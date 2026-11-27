/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4: lossless record codec for MXFP4 and NVFP4 routed-expert slabs, built
 * for wide parallel decode (docs/WNR4-SPEC.md). Header-only C99; the CUDA
 * decoder is wnr4/wnr4_cuda.cuh and shares the helpers marked WNR4_HD.
 *
 * Model: WNR3's (per block, class of the block scale relative to the rule-g
 * anchor, previous magnitude and running maximum; 16-symbol, 14-bit tables;
 * scales order 0 (MX) or, for NV, coded as a delta from the scale of the row above with the up-left and
 * up-right deltas as context).
 * Skeleton: a record is a list of tiles; a tile is 32 rANS lanes (32-bit
 * state, L = 2^16, 16-bit renormalisation) sharing one word stream. Lane l of
 * a tile owns cells c0 + l, c0 + 32 + l, ...; a cell is 32 nibbles (16 bytes)
 * with its scale (MX: 1 scale of a 32-block; NV: 2 scales of two 16-blocks).
 * The decoder runs the lanes in lockstep, symbol by symbol; at each symbol the
 * lanes that renormalise read the next words in lane order.
 * Integrity: lanes 0 and 1 start at L + the low / high half of the tile hash
 * (MurmurHash3 of the coded cells, per lane, XOR over lanes), lanes 2..31 at L + their 4 tail nibbles;
 * the decoder requires those final states and that the tile's words are used
 * exactly. Coded domain as WNR3 (canonical zero, ceil form for MX).
 * ------------------------------------------------------------------------- */
#ifndef WNR4_H
#define WNR4_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__) && !defined(__CUDACC__)
#include <immintrin.h>
#endif

#ifdef __CUDACC__
#define WNR4_HD __host__ __device__
#else
#define WNR4_HD
#endif

#define WNR4_MAGIC UINT32_C(0x34524e57)         /* "WNR4" record */
#define WNM4_MAGIC UINT32_C(0x344d4e57)         /* "WNM4" model blob */
#define WNR4_L UINT32_C(0x10000)
#define WNR4_PREC 14u
#define WNR4_TOTAL (1u << WNR4_PREC)
#define WNR4_LANES 32u
#define WNR4_HEADER_BYTES 64u
#define WNR4_TILE_STATE_BYTES (4u * WNR4_LANES)
#define WNR4_FLAG_CEIL_MASK 0x07u
#define WNR4_FLAG_CZ 0x08u
#define WNR4_KIND_MX 0u                          /* MXFP4: UE8M0 scale per 32 nibbles */
#define WNR4_KIND_NV 1u                          /* NVFP4: E4M3 scale per 16 nibbles */
#define WNR4_MAX_G 4096u
#define WNR4_HASH_SEED WNR4_MAGIC

/* per kind: classes (2R+1), scale contexts, scales per cell, nibbles per block */
#define WNR4_NCLS(k) ((k) ? 13u : 7u)
#define WNR4_R(k) ((k) ? 6 : 3)
#define WNR4_NSCTX(k) ((k) ? 4160u : 1u)   /* NV: 65 buckets of the scale above x 8 x 8 neighbour deltas */
#define WNR4_SPC(k) ((k) ? 2u : 1u)
#define WNR4_WCTX(k) (WNR4_NCLS(k) * 81u)
#define WNR4_ERR_STRUCT 1u
#define WNR4_ERR_HASH 2u
#define WNR4_ERR_CEIL 4u
#define WNR4_ERR_CZ 8u

/* ---- small helpers shared with CUDA ----------------------------------- */
static WNR4_HD inline uint32_t wnr4_rotl(uint32_t x, int r) { return x << r | x >> (32 - r); }
static WNR4_HD inline uint32_t wnr4_h32_block(uint32_t h, uint32_t k) {
    k *= UINT32_C(0xcc9e2d51); k = wnr4_rotl(k, 15); k *= UINT32_C(0x1b873593);
    h ^= k; h = wnr4_rotl(h, 13);
    return h * 5u + UINT32_C(0xe6546b64);
}
static WNR4_HD inline uint32_t wnr4_h32_final(uint32_t h, uint32_t len) {
    h ^= len;
    h ^= h >> 16; h *= UINT32_C(0x85ebca6b); h ^= h >> 13; h *= UINT32_C(0xc2b2ae35); h ^= h >> 16;
    return h;
}
/* lane hash start: ident words {model, owner, tile, lane, word, anchors} */
static WNR4_HD inline uint32_t wnr4_lane_seed(uint32_t model, uint32_t owner, uint32_t tile, uint32_t lane,
                                              uint32_t word, uint32_t anchors) {
    uint32_t h = WNR4_HASH_SEED;
    h = wnr4_h32_block(h, model); h = wnr4_h32_block(h, owner); h = wnr4_h32_block(h, tile);
    h = wnr4_h32_block(h, lane); h = wnr4_h32_block(h, word);
    return wnr4_h32_block(h, anchors);
}
/* class of a scale */
static WNR4_HD inline uint32_t wnr4_cls(uint32_t s, uint32_t anchor, int R) {
    int d = (int)s - (int)anchor;
    d = d < -R ? -R : d > R ? R : d;
    return (uint32_t)(d + R);
}
/* WNR3 ceil-form inverse of 8 nibbles: magnitudes 0..5 -> 0,2,4,5,6,7, signs kept */
static WNR4_HD inline uint32_t wnr4_ceil_dbl8(uint32_t w) {
    const uint32_t m = w & 0x77777777u;
    const uint32_t g = ((m >> 1) | (m >> 2)) & 0x11111111u;
    const uint32_t mn = (g << 1) | (m & 0x11111111u & ~g);
    return (w & 0x88888888u) | (m + mn);
}
static inline uint32_t wnr4_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static inline void wnr4_put32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

/* ---- geometry ----------------------------------------------------------- */
/* An expert slab: per matrix m its scales (scount[m] bytes at soff[m]) and
 * nibbles (B * scount[m], packed two per byte, at woff[m]); B = 32 (MX) or 16 (NV). */
typedef struct {
    uint32_t kind;
    uint32_t scount[3], scols[3], soff[3], woff[3];   /* scales, scales per row, offsets */
    uint32_t cells[3];          /* nibbles / 32 */
    uint64_t raw;               /* slab bytes = max end of the six regions */
} Wnr4Geo;

static inline int wnr4_geo(Wnr4Geo *g, uint32_t kind, const uint32_t scount[3], const uint32_t scols[3],
                           const uint32_t soff[3], const uint32_t woff[3]) {
    uint32_t m, spc;
    if (kind > 1u) return -1;
    memset(g, 0, sizeof(*g));
    g->kind = kind;
    spc = WNR4_SPC(kind);
    uint64_t lo[6], hi[6];
    uint32_t i, j;
    for (m = 0; m < 3u; m++) {
        uint64_t end;
        /* a cell count below 2^27 keeps every tile, cell and scale index of the codec inside uint32 */
        if (!scount[m] || scount[m] % spc || !scols[m] || scount[m] % scols[m] || scount[m] / spc >= (1u << 27))
            return -1;
        g->scount[m] = scount[m]; g->scols[m] = scols[m]; g->soff[m] = soff[m]; g->woff[m] = woff[m];
        g->cells[m] = scount[m] / spc;
        lo[2 * m] = soff[m]; hi[2 * m] = (uint64_t)soff[m] + scount[m];
        lo[2 * m + 1] = woff[m]; hi[2 * m + 1] = (uint64_t)woff[m] + (uint64_t)g->cells[m] * 16u;
        end = hi[2 * m]; if (end > g->raw) g->raw = end;
        end = hi[2 * m + 1]; if (end > g->raw) g->raw = end;
    }
    for (i = 0; i < 6u; i++)   /* the six regions must not overlap: decoders read scales back from the slab */
        for (j = i + 1u; j < 6u; j++)
            if (lo[i] < hi[j] && lo[j] < hi[i]) return -1;
    return 0;
}

/* ---- model blob (WNM4) ----------------------------------------------------
 * 32-byte header: u32 magic WNM4, version 1, kind, precision 14, weight contexts
 * rows (NCLS * 81 * pb), scale contexts, pb (position buckets: 1, or 5 = positions
 * below B-4 together and the last four apart), 0; then 3 x wctx weight tables of 17 u16
 * (cdf[0] = 0 .. cdf[16] = 16384) and 3 x sctx scale tables of 257 u16, LE.
 * Id = FNV-1a of the blob (0 -> 1). */
typedef struct {
    uint32_t kind, id, wctx, sctx, pb;   /* wctx = weight rows per matrix = NCLS * 81 * pb */
    uint16_t *wcdf;     /* [3][wctx][17] */
    uint16_t *scdf;     /* [3][sctx][257] */
} Wnr4Model;

static inline size_t wnr4_model_bytes(uint32_t kind, uint32_t pb) {
    return 32u + 3u * (size_t)WNR4_WCTX(kind) * pb * 34u + 3u * (size_t)WNR4_NSCTX(kind) * 514u;
}
static inline uint32_t wnr4_fnv1a(const unsigned char *p, size_t n) {
    uint32_t h = UINT32_C(2166136261);
    size_t i;
    for (i = 0; i < n; i++) { h ^= p[i]; h *= UINT32_C(16777619); }
    return h ? h : 1u;
}
static inline int wnr4_cdf_ok(const uint16_t *c, uint32_t n) {
    uint32_t i;
    if (c[0] != 0 || c[n] != WNR4_TOTAL) return 0;
    for (i = 0; i < n; i++) if (c[i + 1] <= c[i]) return 0;
    return 1;
}
static inline void wnr4_model_free(Wnr4Model *m) { free(m->wcdf); free(m->scdf); m->wcdf = m->scdf = NULL; }
static inline int wnr4_model_open(const unsigned char *blob, size_t bytes, Wnr4Model *m) {
    uint32_t kind, i, n, pb;
    memset(m, 0, sizeof(*m));
    if (!blob || bytes < 32u) return -1;
    kind = wnr4_u32(blob + 8); pb = wnr4_u32(blob + 24);
    if (wnr4_u32(blob) != WNM4_MAGIC || wnr4_u32(blob + 4) != 1u || kind > 1u || (pb != 1u && pb != 5u) ||
        bytes != wnr4_model_bytes(kind, pb) || wnr4_u32(blob + 12) != WNR4_PREC ||
        wnr4_u32(blob + 16) != WNR4_WCTX(kind) * pb || wnr4_u32(blob + 20) != WNR4_NSCTX(kind) || wnr4_u32(blob + 28))
        return -1;
    m->kind = kind; m->pb = pb; m->wctx = WNR4_WCTX(kind) * pb; m->sctx = WNR4_NSCTX(kind);
    m->wcdf = (uint16_t *)malloc(3u * (size_t)m->wctx * 17u * 2u);
    m->scdf = (uint16_t *)malloc(3u * (size_t)m->sctx * 257u * 2u);
    if (!m->wcdf || !m->scdf) { wnr4_model_free(m); return -1; }
    n = 3u * m->wctx * 17u;
    for (i = 0; i < n; i++) m->wcdf[i] = (uint16_t)(blob[32 + 2 * i] | blob[33 + 2 * i] << 8);
    for (i = 0; i < 3u * m->sctx * 257u; i++) {
        size_t at = 32u + (size_t)n * 2u + 2u * i;
        m->scdf[i] = (uint16_t)(blob[at] | blob[at + 1] << 8);
    }
    for (i = 0; i < 3u * m->wctx; i++) if (!wnr4_cdf_ok(m->wcdf + (size_t)i * 17u, 16u)) { wnr4_model_free(m); return -1; }
    for (i = 0; i < 3u * m->sctx; i++) if (!wnr4_cdf_ok(m->scdf + (size_t)i * 257u, 256u)) { wnr4_model_free(m); return -1; }
    m->id = wnr4_fnv1a(blob, bytes);
    return 0;
}

/* quantize counts to a 14-bit cdf of n symbols (every symbol >= 1), WNR3's rule */
static inline void wnr4_quantize(const uint64_t *counts, uint32_t n, uint16_t *cdf) {
    uint64_t total = 0, rem[256];
    uint32_t f[256], spare = WNR4_TOTAL - n, used = 0, s, extra;
    unsigned char taken[256] = {0};
    for (s = 0; s < n; s++) total += counts[s];
    for (s = 0; s < n; s++) {
        uint64_t scaled = total ? counts[s] * spare : spare, div = total ? total : n;
        f[s] = 1u + (uint32_t)(scaled / div); rem[s] = scaled % div; used += f[s];
    }
    for (extra = WNR4_TOTAL - used; extra; extra--) {
        uint32_t best = n;
        for (s = 0; s < n; s++) if (!taken[s] && (best == n || rem[s] > rem[best])) best = s;
        taken[best] = 1; f[best]++;
    }
    used = 0;
    for (s = 0; s <= n; s++) { cdf[s] = (uint16_t)used; if (s < n) used += f[s]; }
}

/* counts: weights [3][wctx][16], scales [3][sctx][256] (uint64) */
static inline size_t wnr4_counts_words(uint32_t kind, uint32_t pb) {
    return 3u * (size_t)WNR4_WCTX(kind) * pb * 16u + 3u * (size_t)WNR4_NSCTX(kind) * 256u;
}
static inline void wnr4_model_build(uint32_t kind, uint32_t pb, const uint64_t *counts, unsigned char *blob) {
    uint32_t wctx = WNR4_WCTX(kind) * pb, sctx = WNR4_NSCTX(kind), i, k;
    uint32_t hdr[8] = {WNM4_MAGIC, 1u, kind, WNR4_PREC, wctx, sctx, pb, 0};
    uint16_t cdf[257];
    unsigned char *p = blob + 32;
    for (i = 0; i < 8u; i++) wnr4_put32(blob + 4 * i, hdr[i]);
    for (i = 0; i < 3u * wctx; i++) {
        wnr4_quantize(counts + (size_t)i * 16u, 16u, cdf);
        for (k = 0; k < 17u; k++) { *p++ = (unsigned char)cdf[k]; *p++ = (unsigned char)(cdf[k] >> 8); }
    }
    for (i = 0; i < 3u * sctx; i++) {
        wnr4_quantize(counts + 3u * (size_t)wctx * 16u + (size_t)i * 256u, 256u, cdf);
        for (k = 0; k < 257u; k++) { *p++ = (unsigned char)cdf[k]; *p++ = (unsigned char)(cdf[k] >> 8); }
    }
}


/* ---- coded-domain access --------------------------------------------------- */
static inline uint32_t wnr4_nib(const unsigned char *w, uint32_t cell, uint32_t i) {
    return (w[(size_t)cell * 16u + i / 2u] >> ((i & 1u) * 4u)) & 15u;
}
/* Scale context (MX: order 0, ctx 0, pred 0). NV: from the three scales of the row above (up-left,
 * up, up-right), when the row above is inside [lo, i) and rows are >= 33 scales (so every one of
 * them was decoded at an earlier lockstep step): ctx = (1 + up / 4) * 64 + dul * 8 + dur, d = clamp(
 * neighbour - up, -3, 3) + 3 or 7 when absent; the symbol is (s - up) mod 256, *pred = up. Without
 * the row above: ctx 63, pred 0. */
static WNR4_HD inline uint32_t wnr4_sctx(uint32_t kind, const unsigned char *s, uint32_t i, uint32_t lo, uint32_t cols,
                                         uint32_t *pred) {
    uint32_t up, col, dul = 7u, dur = 7u;
    int d;
    *pred = 0;
    if (!kind) return 0u;
    if (cols < 33u || i < lo + cols) return 63u;
    up = s[i - cols]; col = i % cols;
    if (col > 0u && i - cols - 1u >= lo) { d = (int)s[i - cols - 1u] - (int)up; dul = (uint32_t)((d < -3 ? -3 : d > 3 ? 3 : d) + 3); }
    if (col + 1u < cols) { d = (int)s[i - cols + 1u] - (int)up; dur = (uint32_t)((d < -3 ? -3 : d > 3 ? 3 : d) + 3); }
    *pred = up;
    return (1u + (up >> 2)) * 64u + dul * 8u + dur;
}
/* weight context row after the previous magnitude pm and running maximum mx (8 = none) */
static WNR4_HD inline uint32_t wnr4_wctx(uint32_t cls, uint32_t pm, uint32_t mx) { return (cls * 9u + pm) * 9u + mx; }
/* weight row of nibble position p (0..B-1) of a block: pb = 1, or 5 (the last four positions apart) */
static WNR4_HD inline uint32_t wnr4_wrow(uint32_t ctx, uint32_t pb, uint32_t p, uint32_t B) {
    return pb == 1u ? ctx : ctx * 5u + (p + 5u <= B ? 0u : p + 5u - B);
}
static WNR4_HD inline uint32_t wnr4_mx(uint32_t mx, uint32_t a) { return mx == 8u ? a : (a > mx ? a : mx); }

/* training counts of one coded slab, added to counts (wnr4_counts_words(kind) u64) */
static inline void wnr4_count(const Wnr4Geo *g, const unsigned char *coded, const unsigned char anchor[3], uint32_t pb,
                              uint64_t *counts) {
    const uint32_t kind = g->kind, spc = WNR4_SPC(kind), wctx = WNR4_WCTX(kind) * pb, B = kind ? 16u : 32u;
    const int R = WNR4_R(kind);
    uint64_t *sc = counts + 3u * (size_t)wctx * 16u;
    uint32_t m, c, i;
    for (m = 0; m < 3u; m++) {
        const unsigned char *s = coded + g->soff[m], *w = coded + g->woff[m];
        for (i = 0; i < g->scount[m]; i++) {
            uint32_t pred, ctx = wnr4_sctx(kind, s, i, 0u, g->scols[m], &pred);
            sc[((size_t)m * WNR4_NSCTX(kind) + ctx) * 256u + ((s[i] - pred) & 255u)]++;
        }
        for (c = 0; c < g->cells[m]; c++) {
            uint32_t pm = 8, mx = 8, cls = 0;
            for (i = 0; i < 32u; i++) {
                uint32_t x = wnr4_nib(w, c, i);
                if (!(i % B)) { cls = wnr4_cls(s[(size_t)c * spc + i / B], anchor[m], R); pm = mx = 8; }
                counts[((size_t)m * wctx + wnr4_wrow(wnr4_wctx(cls, pm, mx), pb, i % B, B)) * 16u + x]++;
                pm = x & 7u; mx = wnr4_mx(mx, pm);
            }
        }
    }
}

/* rule g anchor (WNR3 4.3) */
static inline unsigned char wnr4_anchor(const unsigned char *s, uint32_t n) {
    uint32_t h[256] = {0}, i, best = 0, top = 0;
    for (i = 0; i < n; i++) h[s[i]]++;
    for (i = 0; i < 255u; i++) if (h[i] + h[i + 1] > top) { top = h[i] + h[i + 1]; best = i; }
    return (unsigned char)best;
}

/* ---- record layout --------------------------------------------------------
 * [0,64) header: u32 magic, model_id, owner; u8 kind, flags; u16 G; u8 anchor[3],
 * u8 0; u32 ntiles, payload bytes; 36 zero bytes.
 * [64, 64 + 4(ntiles + 1)) u32 tile offsets into the payload (offset[0] = 0,
 * offset[ntiles] = payload bytes, every tile length >= 128 and even).
 * payload: per tile 32 u32 initial states, then its u16 words. Zero padding to
 * align4096. Tiles: matrix 0 then 1 then 2, each ceil(cells / (32 G)) tiles.
 * Inside a tile (n cells from c0): phase 1, the tile's n*spc scales in steps k
 * of 32, lane l owning scale c0*spc + 32 k + l; phase 2, rounds j of 32
 * cells, lane l owning cell c0 + 32 j + l, its 32 nibbles in order. */
typedef struct {
    uint32_t model_id, owner, kind, flags, G, ntiles, payload;
    unsigned char anchor[3];
    const unsigned char *offsets, *data;   /* offsets table, payload start */
    uint32_t tile_base[4];                 /* first tile of each matrix, [3] = ntiles */
} Wnr4Record;

static WNR4_HD inline uint32_t wnr4_tiles(uint32_t cells, uint32_t G) { return (cells + 32u * G - 1u) / (32u * G); }

static inline int wnr4_parse(const unsigned char *rec, size_t total, const Wnr4Geo *g, Wnr4Record *r) {
    uint32_t m, t, prev = 0;
    size_t used;
    memset(r, 0, sizeof(*r));
    if (!rec || total < WNR4_HEADER_BYTES || (total & 4095u)) return -1;
    if (wnr4_u32(rec) != WNR4_MAGIC) return -1;
    r->model_id = wnr4_u32(rec + 4); r->owner = wnr4_u32(rec + 8);
    r->kind = rec[12]; r->flags = rec[13]; r->G = (uint32_t)rec[14] | (uint32_t)rec[15] << 8;
    r->anchor[0] = rec[16]; r->anchor[1] = rec[17]; r->anchor[2] = rec[18];
    r->ntiles = wnr4_u32(rec + 20); r->payload = wnr4_u32(rec + 24);
    if (!r->model_id || r->kind != g->kind || rec[19] || r->G < 1u || r->G > WNR4_MAX_G) return -1;
    if (r->flags & ~(WNR4_FLAG_CEIL_MASK | WNR4_FLAG_CZ)) return -1;
    if (r->kind == WNR4_KIND_NV && (r->flags & WNR4_FLAG_CEIL_MASK)) return -1;
    for (t = 28; t < 64; t++) if (rec[t]) return -1;
    for (m = 0, t = 0; m < 3u; m++) { r->tile_base[m] = t; t += wnr4_tiles(g->cells[m], r->G); }
    r->tile_base[3] = t;
    if (t != r->ntiles) return -1;
    used = WNR4_HEADER_BYTES + 4u * ((size_t)r->ntiles + 1u) + r->payload;
    if (used > total || ((used + 4095u) & ~(size_t)4095u) != total) return -1;
    r->offsets = rec + WNR4_HEADER_BYTES;
    r->data = r->offsets + 4u * ((size_t)r->ntiles + 1u);
    if (wnr4_u32(r->offsets)) return -1;
    for (t = 1; t <= r->ntiles; t++) {
        uint32_t o = wnr4_u32(r->offsets + 4u * t);
        if (o < prev || o - prev < WNR4_TILE_STATE_BYTES || ((o - prev) & 1u)) return -1;
        prev = o;
    }
    if (prev != r->payload) return -1;
    for (; used < total; used++) if (rec[used]) return -1;
    return 0;
}

/* tile t -> matrix, first cell, cells of the tile */
static WNR4_HD inline void wnr4_tile_cells(const uint32_t tile_base[4], const uint32_t cells[3], uint32_t G,
                                           uint32_t t, uint32_t *m, uint32_t *c0, uint32_t *n) {
    uint32_t mm = t >= tile_base[2] ? 2u : t >= tile_base[1] ? 1u : 0u;
    uint32_t first = (t - tile_base[mm]) * 32u * G, left = cells[mm] - first;
    *m = mm; *c0 = first; *n = left < 32u * G ? left : 32u * G;
}
static WNR4_HD inline uint32_t wnr4_word(uint32_t kind, uint32_t flags, uint32_t G) { return kind | flags << 8 | G << 16; }
/* scales and cells owned by lane l of a tile of n cells (ns scales) */
static WNR4_HD inline uint32_t wnr4_lane_scales(uint32_t ns, uint32_t l) { return l < ns ? (ns - l + 31u) / 32u : 0u; }
static WNR4_HD inline uint32_t wnr4_lane_cells(uint32_t n, uint32_t l) { return l < n ? (n - l + 31u) / 32u : 0u; }
static WNR4_HD inline uint32_t wnr4_lane_hlen(uint32_t nsl, uint32_t ncl) { return 24u + 4u * nsl + 16u * ncl; }
/* Tail in the state: for a lane l >= 2 with cells, the last 4 nibbles of its last cell are not
 * rANS-coded; the lane starts at L + (n28 | n29 << 4 | n30 << 8 | n31 << 12) and ends there. */
static WNR4_HD inline int wnr4_tail_skip(uint32_t l, uint32_t j, uint32_t k, uint32_t n) {
    return l >= 2u && k >= 28u && l < n && j == (n - l - 1u) / 32u;
}

/* ---- decode (reference, CPU) ---------------------------------------------- */
static inline uint32_t wnr4_find16(const uint16_t *cdf, uint32_t slot) {
#if defined(__AVX2__) && !defined(__CUDACC__)
    /* cdf[0..15] < 2^15: one signed 16-bit compare; symbol = (number of cdf[k] <= slot) - 1 */
    const __m256i gt = _mm256_cmpgt_epi16(_mm256_loadu_si256((const __m256i *)cdf), _mm256_set1_epi16((short)slot));
    return 15u - ((uint32_t)__builtin_popcount((unsigned)_mm256_movemask_epi8(gt)) >> 1);
#else
    uint32_t s = 0, k;
    for (k = 1; k < 16u; k++) s += cdf[k] <= slot;
    return s;
#endif
}
static inline uint32_t wnr4_find256(const uint16_t *cdf, uint32_t slot) {
    uint32_t lo = 0, hi = 256;   /* largest s with cdf[s] <= slot */
    while (hi - lo > 1u) { uint32_t mid = (lo + hi) / 2u; if (cdf[mid] <= slot) lo = mid; else hi = mid; }
    return lo;
}

/* Decodes tile t into out (an output-domain slab laid out by g). Returns 0 or
 * error bits. out may hold partial data on failure (decode into scratch when
 * the destination must stay untouched). */
static inline uint32_t wnr4_decode_tile(const Wnr4Model *model, const Wnr4Geo *g, const Wnr4Record *r, uint32_t t,
                                        unsigned char *out) {
    const uint32_t kind = g->kind, spc = WNR4_SPC(kind), B = kind ? 16u : 32u, wctx = model->wctx;
    const int R = WNR4_R(kind);
    uint32_t m, c0, n, o0 = wnr4_u32(r->offsets + 4u * t), o1 = wnr4_u32(r->offsets + 4u * t + 4u);
    const unsigned char *tile = r->data + o0;
    const uint32_t nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u;
    uint32_t x[32], hash[32], ptr = 0, l, j, k, err = 0, H = 0;
    const uint32_t anchors = (uint32_t)r->anchor[0] | (uint32_t)r->anchor[1] << 8 | (uint32_t)r->anchor[2] << 16;
    wnr4_tile_cells(r->tile_base, g->cells, r->G, t, &m, &c0, &n);
    const uint16_t *wt = model->wcdf + (size_t)m * wctx * 17u, *st = model->scdf + (size_t)m * model->sctx * 257u;
    const int ceil = (r->flags >> m) & 1u, cz = (r->flags & WNR4_FLAG_CZ) != 0;
    unsigned char *so = out + g->soff[m], *wo = out + g->woff[m];
    const uint32_t ns = n * spc, S0 = c0 * spc, cols = g->scols[m];
    if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || o1 > r->payload) return WNR4_ERR_STRUCT;   /* re-read: re-check */
    for (l = 0; l < 32u; l++) {
        x[l] = wnr4_u32(tile + 4u * l);
        if (x[l] < WNR4_L) return WNR4_ERR_STRUCT;
        hash[l] = wnr4_lane_seed(r->model_id, r->owner, t, l, wnr4_word(kind, r->flags, r->G), anchors);
    }
#define WNR4_RENORM(L_)                                                              \
    if (x[L_] < WNR4_L) {                                                            \
        if (ptr >= nwords) return WNR4_ERR_STRUCT;                                   \
        x[L_] = x[L_] << 16 | (uint32_t)(tile[128u + 2u * ptr] | tile[129u + 2u * ptr] << 8); ptr++; \
    }
    /* phase 1: scales, row-major across the lanes */
    for (k = 0; 32u * k < ns; k++)
        for (l = 0; l < 32u; l++) {
            uint32_t i = 32u * k + l, slot, y, pred;
            const uint16_t *cdf;
            if (i >= ns) continue;
            i += S0;
            cdf = st + (size_t)wnr4_sctx(kind, so, i, S0, cols, &pred) * 257u;
            slot = x[l] & (WNR4_TOTAL - 1u); y = wnr4_find256(cdf, slot);
            x[l] = (uint32_t)(cdf[y + 1] - cdf[y]) * (x[l] >> WNR4_PREC) + slot - cdf[y];
            y = (y + pred) & 255u;
            so[i] = (unsigned char)y;
            hash[l] = wnr4_h32_block(hash[l], y);
            WNR4_RENORM(l)
        }
    /* phase 2: cells */
    for (j = 0; 32u * j < n; j++) {
        uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u;
        uint32_t pm[32], mx[32], cls[32], wd[32][4];
        for (l = 0; l < act; l++) wd[l][0] = wd[l][1] = wd[l][2] = wd[l][3] = 0;
        for (k = 0; k < 32u; k++) {
            if (!(k % B))
                for (l = 0; l < act; l++) {
                    cls[l] = wnr4_cls(so[(size_t)(c0 + 32u * j + l) * spc + k / B], r->anchor[m], R);
                    pm[l] = mx[l] = 8;
                }
            for (l = 0; l < act; l++) {
                if (wnr4_tail_skip(l, j, k, n)) continue;
                const uint16_t *cdf = wt + (size_t)wnr4_wrow(wnr4_wctx(cls[l], pm[l], mx[l]), model->pb, k % B, B) * 17u;
                uint32_t slot = x[l] & (WNR4_TOTAL - 1u), y = wnr4_find16(cdf, slot);
                x[l] = (uint32_t)(cdf[y + 1] - cdf[y]) * (x[l] >> WNR4_PREC) + slot - cdf[y];
                if (cz && y == 8u) err |= WNR4_ERR_CZ;
                wd[l][k / 8u] |= y << ((k % 8u) * 4u);
                pm[l] = y & 7u; mx[l] = wnr4_mx(mx[l], pm[l]);
                WNR4_RENORM(l)
            }
        }
        for (l = 0; l < act; l++) {
            uint32_t cell = c0 + 32u * j + l;
            if (wnr4_tail_skip(l, j, 31u, n)) {   /* tail nibbles come from the final state */
                uint32_t tail = x[l] - WNR4_L, q;
                if (x[l] < WNR4_L || tail > 0xffffu) return err | WNR4_ERR_STRUCT;
                for (q = 0; q < 4u; q++) {
                    uint32_t y = (tail >> (4u * q)) & 15u;
                    if (cz && y == 8u) err |= WNR4_ERR_CZ;
                    mx[l] = wnr4_mx(mx[l], y & 7u);
                }
                wd[l][3] |= tail << 16;
                x[l] = WNR4_L;
            }
            for (k = 0; k < 4u; k++) hash[l] = wnr4_h32_block(hash[l], wd[l][k]);
            if (ceil && mx[l] == 5u) {   /* MX: one block per cell, mx = its maximum */
                if (!so[cell]) err |= WNR4_ERR_CEIL;
                else { so[cell]--; for (k = 0; k < 4u; k++) wd[l][k] = wnr4_ceil_dbl8(wd[l][k]); }
            }
            for (k = 0; k < 4u; k++) wnr4_put32(wo + (size_t)cell * 16u + 4u * k, wd[l][k]);
        }
    }
#undef WNR4_RENORM
    for (l = 0; l < 32u; l++)
        H ^= wnr4_h32_final(hash[l], wnr4_lane_hlen(wnr4_lane_scales(ns, l), wnr4_lane_cells(n, l)));
    if (ptr != nwords) err |= WNR4_ERR_STRUCT;
    if (x[0] != WNR4_L + (H & 0xffffu) || x[1] != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    for (l = 2; l < 32u; l++) if (x[l] != WNR4_L) err |= WNR4_ERR_STRUCT;
    return err;
}

/* whole record into out (g->raw bytes); returns 0 or error bits */
static inline uint32_t wnr4_decode(const Wnr4Model *model, const Wnr4Geo *g, const unsigned char *rec, size_t total,
                                   uint32_t owner, unsigned char *out, int threads) {
    Wnr4Record r;
    uint32_t err = 0;
    int t;
    if (model->kind != g->kind || wnr4_parse(rec, total, g, &r) || r.model_id != model->id || r.owner != owner)
        return WNR4_ERR_STRUCT;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1) reduction(| : err)
#endif
    for (t = 0; t < (int)r.ntiles; t++) err |= wnr4_decode_tile(model, g, &r, (uint32_t)t, out);
    (void)threads;
    return err;
}

/* n records of one geometry, every tile of every record in one parallel loop (small records keep all
 * threads busy). status[i] = 0 or error bits; outs[i] may hold partial data when status[i] != 0.
 * Returns the number of failed records. */
static inline int wnr4_decode_batch(const Wnr4Model *model, const Wnr4Geo *g, int n, const unsigned char *const *recs,
                                    const size_t *totals, const uint32_t *owners, unsigned char *const *outs,
                                    uint32_t *status, int threads) {
    Wnr4Record *r = (Wnr4Record *)calloc((size_t)(n > 0 ? n : 1), sizeof(Wnr4Record));
    long long *first = (long long *)calloc((size_t)(n > 0 ? n : 1) + 1u, sizeof(long long));
    long long t, total = 0;
    int i, bad = 0;
    if (!r || !first) { free(r); free(first); for (i = 0; i < n; i++) status[i] = WNR4_ERR_STRUCT; return n; }
    for (i = 0; i < n; i++) {
        status[i] = model->kind != g->kind || wnr4_parse(recs[i], totals[i], g, &r[i]) || r[i].model_id != model->id ||
                    r[i].owner != owners[i] ? WNR4_ERR_STRUCT : 0u;
        first[i] = total;
        if (!status[i]) total += r[i].ntiles;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1)
#endif
    for (t = 0; t < total; t++) {
        int lo = 0, hi = n;   /* the last record whose first tile is <= t: a record without tiles never is */
        while (hi - lo > 1) { int mid = (lo + hi) / 2; if (first[mid] <= t) lo = mid; else hi = mid; }
        uint32_t e = wnr4_decode_tile(model, g, &r[lo], (uint32_t)(t - first[lo]), outs[lo]);
        if (e) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            status[lo] |= e;
        }
    }
    for (i = 0; i < n; i++) bad += status[i] != 0;
    free(r); free(first);
    (void)threads;
    return bad;
}

/* ---- encode ------------------------------------------------------------------ */
/* Encodes the coded slab (flags: ceil mask (MX) | CZ; already in the coded
 * domain) into rec (capacity cap). Returns record bytes (align4096), or 0 on
 * error / overflow. */
static inline size_t wnr4_encode(const Wnr4Model *model, const Wnr4Geo *g, const unsigned char *coded,
                                 uint32_t flags, uint32_t owner, uint32_t G, unsigned char *rec, size_t cap) {
    const uint32_t kind = g->kind, spc = WNR4_SPC(kind), B = kind ? 16u : 32u, wctx = model->wctx;
    const int R = WNR4_R(kind);
    unsigned char anchor[3];
    uint32_t ntiles = 0, m, t, l, k, j, tile_base[4];
    size_t at, payload0, nsym = (size_t)32u * G * (32u + spc) + 64u, total;
    uint16_t *stack;
    uint32_t *sym;   /* per symbol in decode order: bit 31 scale table, 24..28 lane, 8..23 table, 0..7 symbol */
    if (!G || G > WNR4_MAX_G || model->kind != kind || (kind == WNR4_KIND_NV && (flags & WNR4_FLAG_CEIL_MASK)) ||
        (flags & ~(WNR4_FLAG_CEIL_MASK | WNR4_FLAG_CZ))) return 0;
    for (m = 0; m < 3u; m++) {
        anchor[m] = wnr4_anchor(coded + g->soff[m], g->scount[m]);
        tile_base[m] = ntiles; ntiles += wnr4_tiles(g->cells[m], G);
    }
    tile_base[3] = ntiles;
    payload0 = WNR4_HEADER_BYTES + 4u * ((size_t)ntiles + 1u);
    if (payload0 > cap) return 0;
    memset(rec, 0, payload0);
    wnr4_put32(rec, WNR4_MAGIC); wnr4_put32(rec + 4, model->id); wnr4_put32(rec + 8, owner);
    rec[12] = (unsigned char)kind; rec[13] = (unsigned char)flags; rec[14] = (unsigned char)G; rec[15] = (unsigned char)(G >> 8);
    rec[16] = anchor[0]; rec[17] = anchor[1]; rec[18] = anchor[2];
    wnr4_put32(rec + 20, ntiles);
    const uint32_t anchors = (uint32_t)anchor[0] | (uint32_t)anchor[1] << 8 | (uint32_t)anchor[2] << 16;
    stack = (uint16_t *)malloc(nsym * 2u);
    sym = (uint32_t *)malloc(nsym * 4u);
    if (!stack || !sym) { free(stack); free(sym); return 0; }
    at = payload0;
    for (t = 0; t < ntiles; t++) {
        uint32_t c0, n, x[32], hash[32], H = 0, ns, S0, cols;
        size_t sp = 0, p, q = 0;
        wnr4_tile_cells(tile_base, g->cells, G, t, &m, &c0, &n);
        const unsigned char *sc = coded + g->soff[m], *wt = coded + g->woff[m];
        const uint32_t wbase = m * wctx, sbase = m * model->sctx;
        ns = n * spc; S0 = c0 * spc; cols = g->scols[m];
        for (l = 0; l < 32u; l++)
            hash[l] = wnr4_lane_seed(model->id, owner, t, l, wnr4_word(kind, flags, G), anchors);
        /* forward, in decode order */
        for (k = 0; 32u * k < ns; k++)
            for (l = 0; l < 32u; l++) {
                uint32_t i = 32u * k + l, y, pred, ctx;
                if (i >= ns) continue;
                i += S0; y = sc[i];
                ctx = wnr4_sctx(kind, sc, i, S0, cols, &pred);
                sym[q++] = 0x80000000u | l << 24 | (sbase + ctx) << 8 | ((y - pred) & 255u);
                hash[l] = wnr4_h32_block(hash[l], y);
            }
        for (j = 0; 32u * j < n; j++) {
            uint32_t act = n - 32u * j < 32u ? n - 32u * j : 32u, pm[32], mx[32], cls[32];
            for (k = 0; k < 32u; k++) {
                if (!(k % B))
                    for (l = 0; l < act; l++) {
                        cls[l] = wnr4_cls(sc[(size_t)(c0 + 32u * j + l) * spc + k / B], anchor[m], R);
                        pm[l] = mx[l] = 8;
                    }
                for (l = 0; l < act; l++) {
                    uint32_t y = wnr4_nib(wt, c0 + 32u * j + l, k);
                    if (!wnr4_tail_skip(l, j, k, n))
                        sym[q++] = l << 24 | (wbase + wnr4_wrow(wnr4_wctx(cls[l], pm[l], mx[l]), model->pb, k % B, B)) << 8 | y;
                    pm[l] = y & 7u; mx[l] = wnr4_mx(mx[l], pm[l]);
                }
            }
            for (l = 0; l < act; l++)
                for (k = 0; k < 4u; k++)
                    hash[l] = wnr4_h32_block(hash[l], wnr4_u32(wt + (size_t)(c0 + 32u * j + l) * 16u + 4u * k));
        }
        for (l = 0; l < 32u; l++)
            H ^= wnr4_h32_final(hash[l], wnr4_lane_hlen(wnr4_lane_scales(ns, l), wnr4_lane_cells(n, l)));
        for (l = 0; l < 32u; l++) {
            x[l] = l == 0 ? WNR4_L + (H & 0xffffu) : l == 1 ? WNR4_L + (H >> 16) : WNR4_L;
            if (l >= 2u && l < n) {   /* the lane's last cell: nibbles 28..31 = bytes 14..15 */
                const unsigned char *last = wt + (size_t)(c0 + l + 32u * ((n - l - 1u) / 32u)) * 16u;
                x[l] += (uint32_t)last[14] | (uint32_t)last[15] << 8;
            }
        }
        /* backward: every symbol in reverse decode order; words on a stack */
        for (p = q; p-- > 0;) {
            uint32_t e = sym[p], y = e & 0xffu, ln = (e >> 24) & 31u, ti = (e >> 8) & 0xffffu, f, s0;
            const uint16_t *cdf = (e & 0x80000000u) ? model->scdf + (size_t)ti * 257u : model->wcdf + (size_t)ti * 17u;
            s0 = cdf[y]; f = (uint32_t)cdf[y + 1] - s0;
            if (x[ln] >= (f << (32u - WNR4_PREC))) { stack[sp++] = (uint16_t)x[ln]; x[ln] >>= 16; }
            x[ln] = ((x[ln] / f) << WNR4_PREC) + x[ln] % f + s0;
        }
        /* the tile offsets and P are u32: a payload past 4 GiB is refused, as wnr4_i4.h does */
        if (at + WNR4_TILE_STATE_BYTES + 2u * sp > cap || at + WNR4_TILE_STATE_BYTES + 2u * sp - payload0 > 0xffffffffu) {
            free(stack); free(sym); return 0;
        }
        for (l = 0; l < 32u; l++) wnr4_put32(rec + at + 4u * l, x[l]);
        for (p = 0; p < sp; p++) {   /* reversed: the last word pushed is read first */
            uint16_t w = stack[sp - 1u - p];
            rec[at + 128u + 2u * p] = (unsigned char)w; rec[at + 129u + 2u * p] = (unsigned char)(w >> 8);
        }
        at += WNR4_TILE_STATE_BYTES + 2u * sp;
        wnr4_put32(rec + WNR4_HEADER_BYTES + 4u * (t + 1u), (uint32_t)(at - payload0));
    }
    free(stack); free(sym);
    wnr4_put32(rec + 24, (uint32_t)(at - payload0));
    total = (at + 4095u) & ~(size_t)4095u;
    if (total > cap) return 0;
    memset(rec + at, 0, total - at);
    return total;
}

#endif
