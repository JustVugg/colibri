/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* WNR4 CUDA decoder (docs/WNR4-SPEC.md). One warp decodes one tile: its 32
 * rANS lanes run in lockstep and share the tile's 16-bit word stream (ballot +
 * popc gives each renormalising lane its word). Every tile of every record of
 * a batch is one job; one launch decodes the whole batch straight into the
 * callers' destinations (device memory, e.g. the expert's VRAM slot).
 * Destinations of a record that fails may hold partial data: status != 0
 * means "do not use the slot". Host side: wnr4_cuda_plan() (jobs, from the
 * records' host copies), wnr4_cuda_tables() (model upload), wnr4_cuda_launch(). */
#ifndef WNR4_CUDA_CUH
#define WNR4_CUDA_CUH

#include <cuda_runtime.h>
#include <atomic>
#include <cstring>
#include <vector>
#include "wnr4.h"

#define WNR4_CUDA_WARPS_MAX 16u   /* warps per block (4, 8 or 16, wnr4_cuda_warps); a block's jobs share one matrix */

struct Wnr4CudaTables {      /* device copy of a model */
    uint32_t kind, id, rows, sctx, pb;   /* rows per matrix = classes * 37 states * pb */
    const uint4 *wrows;      /* [3][rows][2]: cdf[0..15] of each weight row, compact state order */
    const uint16_t *scdf;    /* [3][sctx][257] */
    const uint4 *srows;      /* [3][sctx][34]: 2 uint4 = cdf[0], cdf[16], .., cdf[240]; then cdf[0..255] (two-level search) */
};
struct Wnr4CudaRec {         /* per record of a batch */
    uint32_t model_id, owner, flags, G, anchors, kind;
    uint32_t scols[3];
    unsigned char *dst_s[3], *dst_w[3];
};
struct Wnr4CudaJob {         /* one tile */
    const unsigned char *tile;   /* device: its 32 states then words */
    uint32_t nwords, rec, t, m, c0, n;
};

static __device__ __forceinline__ uint32_t wnr4_half_sign(uint32_t x) {
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0xbb99;" : "=r"(r) : "r"(x));
    return r;
}
/* symbol, start, frequency of slot in a 16-symbol row (cdf[16] = 2^14): as WNR3's row find */
static __device__ __forceinline__ void wnr4_row_find(uint4 a, uint4 b, uint32_t slot, uint32_t &symbol,
                                                     uint32_t &start, uint32_t &frequency) {
    const uint32_t g = slot * 0x10001u | 0x80008000u;
    uint32_t m0 = wnr4_half_sign(g - a.x), m1 = wnr4_half_sign(g - a.y);
    uint32_t m2 = wnr4_half_sign(g - a.z), m3 = wnr4_half_sign(g - a.w);
    uint32_t m4 = wnr4_half_sign(g - b.x), m5 = wnr4_half_sign(g - b.y);
    uint32_t m6 = wnr4_half_sign(g - b.z), m7 = wnr4_half_sign(g - b.w);
    uint32_t bits = (m0 & 0x00010001u) | (m1 & 0x00020002u) | (m2 & 0x00040004u) | (m3 & 0x00080008u) |
                    (m4 & 0x00100010u) | (m5 & 0x00200020u) | (m6 & 0x00400040u) | (m7 & 0x00800080u);
    symbol = (uint32_t)__popc(bits) - 1u;
    uint32_t lo = __vmaxu2(__vmaxu2(__vmaxu2(a.x & m0, a.y & m1), __vmaxu2(a.z & m2, a.w & m3)),
                           __vmaxu2(__vmaxu2(b.x & m4, b.y & m5), __vmaxu2(b.z & m6, b.w & m7)));
    uint32_t hi = __vminu2(__vminu2(__vminu2(a.x | m0, a.y | m1), __vminu2(a.z | m2, a.w | m3)),
                           __vminu2(__vminu2(b.x | m4, b.y | m5), __vminu2(b.z | m6, b.w | m7)));
    start = max(lo & 0xffffu, lo >> 16);
    frequency = min(min(hi & 0xffffu, hi >> 16), WNR4_TOTAL) - start;
}
/* symbol of slot in a 16-symbol row: number of cdf[k] <= slot, minus 1 (the comparison half of row_find) */
static __device__ __forceinline__ uint32_t wnr4_row_sym(uint4 a, uint4 b, uint32_t slot) {
    const uint32_t g = slot * 0x10001u | 0x80008000u;
    uint32_t bits = (wnr4_half_sign(g - a.x) & 0x00010001u) | (wnr4_half_sign(g - a.y) & 0x00020002u) |
                    (wnr4_half_sign(g - a.z) & 0x00040004u) | (wnr4_half_sign(g - a.w) & 0x00080008u) |
                    (wnr4_half_sign(g - b.x) & 0x00100010u) | (wnr4_half_sign(g - b.y) & 0x00200020u) |
                    (wnr4_half_sign(g - b.z) & 0x00400040u) | (wnr4_half_sign(g - b.w) & 0x00800080u);
    return (uint32_t)__popc(bits) - 1u;
}
/* compact (pm, mx) state: 0 = block start (8, 8), else 1 + mx (mx + 1) / 2 + pm (pm <= mx) */
static WNR4_HD inline uint32_t wnr4_state(uint32_t pm, uint32_t mx) { return pm == 8u ? 0u : 1u + mx * (mx + 1u) / 2u + pm; }
/* largest s with cdf[s] <= slot (256 symbols) */
template<typename Load>
static __device__ __forceinline__ uint32_t wnr4_find256_dev(Load cdf, uint32_t slot) {
    uint32_t s = 0;
#pragma unroll
    for (uint32_t step = 128; step; step >>= 1) if (cdf(s + step) <= slot) s += step;
    return s;
}

/* Renormalise the lanes that need a word. Warp-uniform call (all 32 lanes). */
static __device__ __forceinline__ bool wnr4_renorm(uint32_t &x, bool active, const uint16_t *words, uint32_t &ptr,
                                                    uint32_t nwords, uint32_t lane) {
    bool need = active && x < WNR4_L;
    uint32_t vote = __ballot_sync(0xffffffffu, need);
    uint32_t total = __popc(vote);
    if (ptr + total > nwords) return false;
    if (need) x = x << 16 | __ldg(words + ptr + __popc(vote & ((1u << lane) - 1u)));
    ptr += total;
    return true;
}

/* Word window: lane l holds words base + l (w0) and base + 32 + l (w1) of the tile; a renormalising
 * lane takes its word by shuffle; when ptr passes base + 32 the window slides by 32 words with one
 * coalesced 64-byte load, consumed several steps later (its latency off the per-symbol chain). */
struct Wnr4Win { uint32_t w0, w1, base; };
static __device__ __forceinline__ uint32_t wnr4_win_load(const uint16_t *words, uint32_t i, uint32_t nwords) {
    return i < nwords ? (uint32_t)__ldg(words + i) : 0u;
}
static __device__ __forceinline__ void wnr4_win_init(Wnr4Win &w, const uint16_t *words, uint32_t nwords, uint32_t lane) {
    w.base = 0; w.w0 = wnr4_win_load(words, lane, nwords); w.w1 = wnr4_win_load(words, 32u + lane, nwords);
}
static __device__ __forceinline__ bool wnr4_renorm_win(uint32_t &x, bool active, const uint16_t *words, uint32_t &ptr,
                                                        uint32_t nwords, uint32_t lane, Wnr4Win &w) {
    bool need = active && x < WNR4_L;
    uint32_t vote = __ballot_sync(0xffffffffu, need);
    if (!vote) return true;
    uint32_t total = __popc(vote);
    if (ptr + total > nwords) return false;
    uint32_t idx = ptr - w.base + __popc(vote & ((1u << lane) - 1u));   /* < 64 */
    uint32_t v0 = __shfl_sync(0xffffffffu, w.w0, idx & 31u), v1 = __shfl_sync(0xffffffffu, w.w1, idx & 31u);
    if (need) x = x << 16 | (idx < 32u ? v0 : v1);
    ptr += total;
    if (ptr - w.base >= 32u) { w.w0 = w.w1; w.base += 32u; w.w1 = wnr4_win_load(words, w.base + 32u + lane, nwords); }
    return true;
}
#define WNR4_RENORM_CALL(x_, act_) (Win ? wnr4_renorm_win(x_, act_, words, ptr, job.nwords, lane, win) \
                                        : wnr4_renorm(x_, act_, words, ptr, job.nwords, lane))

/* RG: weight rows read through L1 (__ldg) instead of being staged in shared memory (no shared-memory
 * occupancy limit; for NVFP4's 77 KB tables). Launch bounds ask for 32 / W blocks per SM (32 warps, <= 64 registers). */
/* LD: start and end of the decoded symbol by two 16-bit loads from its row instead of the min/max trees
 * of row_find (about 30 fewer instructions per symbol, one more dependent load). */
template<uint32_t Kind, bool Win, uint32_t W, bool RG, bool LD>
static __global__ void __launch_bounds__(W * 32u, 32u / W) wnr4_cuda_kernel(
    const Wnr4CudaJob *jobs, uint32_t njobs, const Wnr4CudaRec *recs, Wnr4CudaTables tab, uint32_t *errors) {
    constexpr uint32_t SPC = Kind ? 2u : 1u, B = Kind ? 16u : 32u;
    constexpr int R = Kind ? 6 : 3;
    extern __shared__ uint4 rows[];   /* tab.rows * 2 */
    __shared__ uint16_t s0cdf[Kind ? 1u : 258u];   /* MX: the matrix's order-0 scale table */
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t first = blockIdx.x * W;
    if (first >= njobs) return;
    const uint32_t m = jobs[first].m;   /* the host groups jobs so a block has one matrix */
    if (!RG) for (uint32_t i = threadIdx.x; i < tab.rows * 2u; i += blockDim.x) rows[i] = tab.wrows[(size_t)m * tab.rows * 2u + i];
    const uint4 *grows = tab.wrows + (size_t)m * tab.rows * 2u;
    if (!Kind) for (uint32_t i = threadIdx.x; i < 257u; i += blockDim.x) s0cdf[i] = tab.scdf[(size_t)m * 257u + i];
    __syncthreads();
    const uint32_t jid = first + warp;
    if (jid >= njobs) return;
    const Wnr4CudaJob job = jobs[jid];
    if (!job.n) return;   /* padding job */
    if (job.m != m) { if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT); return; }   /* mis-planned block: fail closed */
    const Wnr4CudaRec rec = recs[job.rec];
    const uint16_t *st = tab.scdf + (size_t)m * tab.sctx * 257u;
    const uint16_t *words = (const uint16_t *)(job.tile + WNR4_TILE_STATE_BYTES);
    uint32_t x = (uint32_t)__ldg((const uint16_t *)job.tile + 2u * lane) |
                 (uint32_t)__ldg((const uint16_t *)job.tile + 2u * lane + 1u) << 16;
    uint32_t ptr = 0, err = 0;
    bool ok = x >= WNR4_L;
    Wnr4Win win = {0u, 0u, 0u};
    if (Win) wnr4_win_init(win, words, job.nwords, lane);
    uint32_t hash = wnr4_lane_seed(rec.model_id, rec.owner, job.t, lane, wnr4_word(Kind, rec.flags, rec.G), rec.anchors);
    const uint32_t n = job.n, ns = n * SPC, S0 = job.c0 * SPC, cols = rec.scols[m];
    const uint32_t anchor = (rec.anchors >> (8u * m)) & 0xffu;
    unsigned char *so = rec.dst_s[m];
    /* phase 1: scales, row-major across the lanes (NV contexts read the row above from so) */
    for (uint32_t k = 0; 32u * k < ns && __all_sync(0xffffffffu, ok); k++) {
        uint32_t i = 32u * k + lane;
        const bool active = i < ns;
        if (active) {
            i += S0;
            uint32_t slot = x & (WNR4_TOTAL - 1u), y, c, f;
            if (Kind) {   /* two-level: the summary row picks a block of 16 symbols, the block row the symbol */
                uint32_t pred, kb, sS, fS, jb;
                const uint4 *row = tab.srows + ((size_t)m * tab.sctx + wnr4_sctx(Kind, so, i, S0, cols, &pred)) * 34u;
                wnr4_row_find(__ldg(row), __ldg(row + 1), slot, kb, sS, fS);
                wnr4_row_find(__ldg(row + 2u + 2u * kb), __ldg(row + 3u + 2u * kb), slot, jb, c, f);
                if (c + f == WNR4_TOTAL) f = sS + fS - c;   /* the block's last symbol: its end is the next summary value */
                y = 16u * kb + jb;
                x = f * (x >> WNR4_PREC) + slot - c;
                y = (y + pred) & 255u;
            } else {
                y = wnr4_find256_dev([&](uint32_t q) { return (uint32_t)s0cdf[q]; }, slot);
                c = s0cdf[y]; f = (uint32_t)s0cdf[y + 1u] - c;
                x = f * (x >> WNR4_PREC) + slot - c;
            }
            so[i] = (unsigned char)y;
            hash = wnr4_h32_block(hash, y);
        }
        if (!WNR4_RENORM_CALL(x, active)) ok = false;
        if (Kind) __syncwarp();   /* the next step reads this step's scales */
    }
    __syncwarp();   /* phase 2 reads scales written by other lanes */
    const bool ceil = !Kind && ((rec.flags >> m) & 1u), cz = (rec.flags & WNR4_FLAG_CZ) != 0;
    uint4 *wo = (uint4 *)rec.dst_w[m];
    uint32_t pw[4] = {0u, 0u, 0u, 0u}, pmx = 8, ps = 0, pcell = 0;   /* a tail lane's last cell, finished at the end */
    bool pending = false;
    for (uint32_t j = 0; 32u * j < n && __all_sync(0xffffffffu, ok); j++) {
        const uint32_t cell = job.c0 + 32u * j + lane;
        const bool active = 32u * j + lane < n;
        uint32_t s[SPC], w[4] = {0u, 0u, 0u, 0u}, cls = 0, pm = 8, mx = 8;
#pragma unroll
        for (uint32_t q = 0; q < SPC; q++) s[q] = active ? so[(size_t)cell * SPC + q] : 0u;
        const uint32_t kend = active && wnr4_tail_skip(lane, j, 31u, n) ? 28u : 32u;   /* tail lanes stop at 28 */
#pragma unroll
        for (uint32_t k = 0; k < 32u; k++) {
            if (!(k % B)) { cls = wnr4_cls(s[k / B], anchor, R); pm = mx = 8; }
            const bool coded = active && k < kend;
            if (coded) {
                const uint32_t st5 = (cls * 37u + wnr4_state(pm, mx)) * tab.pb;
                const uint32_t ctx = tab.pb == 1u ? st5 : st5 + (k % B + 5u <= B ? 0u : k % B + 5u - B);
                uint32_t slot = x & (WNR4_TOTAL - 1u), y, c, f;
                if (LD) {
                    const uint4 *rp = RG ? grows + ctx * 2u : rows + ctx * 2u;
                    const uint16_t *r16 = (const uint16_t *)rp;
                    y = RG ? wnr4_row_sym(__ldg(rp), __ldg(rp + 1), slot) : wnr4_row_sym(rp[0], rp[1], slot);
                    c = RG ? __ldg(r16 + y) : r16[y];
                    f = (y == 15u ? WNR4_TOTAL : (uint32_t)(RG ? __ldg(r16 + y + 1u) : r16[y + 1u])) - c;
                } else if (RG) wnr4_row_find(__ldg(grows + ctx * 2u), __ldg(grows + ctx * 2u + 1u), slot, y, c, f);
                else wnr4_row_find(rows[ctx * 2u], rows[ctx * 2u + 1u], slot, y, c, f);
                x = f * (x >> WNR4_PREC) + slot - c;
                w[k / 8u] |= y << ((k % 8u) * 4u);
                pm = y & 7u; mx = wnr4_mx(mx, pm);
            }
            ok = WNR4_RENORM_CALL(x, coded) && ok;   /* no break: the loop stays provably convergent */
        }
        if (active && ok && wnr4_tail_skip(lane, j, 31u, n)) {
            pending = true; pw[0] = w[0]; pw[1] = w[1]; pw[2] = w[2]; pw[3] = w[3]; pmx = mx; ps = s[0]; pcell = cell;
        } else if (active && ok) {
            hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, w[0]), w[1]), w[2]), w[3]);
            if (cz) {
#pragma unroll
                for (uint32_t q = 0; q < 4u; q++) {
                    const uint32_t tq = w[q] ^ 0x88888888u;
                    if ((tq - 0x11111111u) & ~tq & 0x88888888u) err |= WNR4_ERR_CZ;
                }
            }
            if (ceil && mx == 5u) {
                if (!s[0]) err |= WNR4_ERR_CEIL;
                else {
                    so[cell] = (unsigned char)(s[0] - 1u);
#pragma unroll
                    for (uint32_t q = 0; q < 4u; q++) w[q] = wnr4_ceil_dbl8(w[q]);
                }
            }
            wo[cell] = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
    if (!__all_sync(0xffffffffu, ok)) { if (!lane) atomicOr(errors + job.rec, WNR4_ERR_STRUCT); return; }
    if (pending) {   /* nibbles 28..31 of the lane's last cell are its final state - L */
        const uint32_t tail = x - WNR4_L;
        if (tail > 0xffffu) err |= WNR4_ERR_STRUCT;
        pw[3] |= tail << 16;
#pragma unroll
        for (uint32_t q = 0; q < 4u; q++) pmx = wnr4_mx(pmx, (tail >> (4u * q)) & 7u);
        x = WNR4_L;
        hash = wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(wnr4_h32_block(hash, pw[0]), pw[1]), pw[2]), pw[3]);
        if (cz) {
#pragma unroll
            for (uint32_t q = 0; q < 4u; q++) {
                const uint32_t tq = pw[q] ^ 0x88888888u;
                if ((tq - 0x11111111u) & ~tq & 0x88888888u) err |= WNR4_ERR_CZ;
            }
        }
        if (ceil && pmx == 5u) {
            if (!ps) err |= WNR4_ERR_CEIL;
            else {
                so[pcell] = (unsigned char)(ps - 1u);
#pragma unroll
                for (uint32_t q = 0; q < 4u; q++) pw[q] = wnr4_ceil_dbl8(pw[q]);
            }
        }
        wo[pcell] = make_uint4(pw[0], pw[1], pw[2], pw[3]);
    }
    uint32_t H = wnr4_h32_final(hash, wnr4_lane_hlen(wnr4_lane_scales(ns, lane), wnr4_lane_cells(n, lane)));
#pragma unroll
    for (uint32_t d = 16; d; d >>= 1) H ^= __shfl_xor_sync(0xffffffffu, H, d);
    if (lane == 0 && x != WNR4_L + (H & 0xffffu)) err |= WNR4_ERR_HASH;
    if (lane == 1 && x != WNR4_L + (H >> 16)) err |= WNR4_ERR_HASH;
    if (lane >= 2u && x != WNR4_L) err |= WNR4_ERR_STRUCT;
    if (lane == 0 && ptr != job.nwords) err |= WNR4_ERR_STRUCT;
    if (err) atomicOr(errors + job.rec, err);
}

/* ---- host side --------------------------------------------------------------- */
/* Model tables for the device: rows (16-byte aligned) + scale cdfs, in one buffer. */
static inline uint32_t wnr4_cuda_rows(const Wnr4Model *m) { return WNR4_NCLS(m->kind) * 37u * m->pb; }
static inline size_t wnr4_cuda_scale_bytes(const Wnr4Model *m) { return ((3u * (size_t)m->sctx * 514u) + 15u) & ~(size_t)15u; }
static inline size_t wnr4_cuda_table_bytes(const Wnr4Model *m) {
    return 3u * (size_t)wnr4_cuda_rows(m) * 32u + wnr4_cuda_scale_bytes(m) + 3u * (size_t)m->sctx * 544u;
}
/* device image: weight rows in compact state order (only reachable (pm, mx) pairs), then the scale cdfs */
static inline void wnr4_cuda_table_image(const Wnr4Model *m, unsigned char *img) {
    const uint32_t rows = wnr4_cuda_rows(m), ncls = WNR4_NCLS(m->kind);
    uint32_t mm, cls, pm, mx, b, k;
    for (mm = 0; mm < 3u; mm++)
        for (cls = 0; cls < ncls; cls++)
            for (mx = 0; mx <= 8u; mx++)
                for (pm = 0; pm <= 8u; pm++) {
                    if (!((pm == 8u && mx == 8u) || (pm <= mx && mx < 8u))) continue;
                    for (b = 0; b < m->pb; b++) {
                        const uint32_t src = mm * m->wctx + wnr4_wctx(cls, pm, mx) * m->pb + b;
                        const uint32_t dst = mm * rows + (cls * 37u + wnr4_state(pm, mx)) * m->pb + b;
                        for (k = 0; k < 16u; k++) memcpy(img + (size_t)dst * 32u + 2u * k, m->wcdf + (size_t)src * 17u + k, 2u);
                    }
                }
    memcpy(img + 3u * (size_t)rows * 32u, m->scdf, 3u * (size_t)m->sctx * 514u);
    {   /* two-level scale rows */
        unsigned char *sr = img + 3u * (size_t)rows * 32u + wnr4_cuda_scale_bytes(m);
        uint32_t t, q;
        for (t = 0; t < 3u * m->sctx; t++) {
            const uint16_t *cdf = m->scdf + (size_t)t * 257u;
            uint16_t *o = (uint16_t *)(sr + (size_t)t * 544u);
            for (q = 0; q < 16u; q++) o[q] = cdf[16u * q];
            for (q = 0; q < 256u; q++) o[16u + q] = cdf[q];
        }
    }
}
static inline void wnr4_cuda_tables_view(const Wnr4Model *m, const unsigned char *dev, Wnr4CudaTables *t) {
    t->kind = m->kind; t->id = m->id; t->rows = wnr4_cuda_rows(m); t->sctx = m->sctx; t->pb = m->pb;
    t->wrows = (const uint4 *)dev;
    t->scdf = (const uint16_t *)(dev + 3u * (size_t)t->rows * 32u);
    t->srows = (const uint4 *)(dev + 3u * (size_t)t->rows * 32u + wnr4_cuda_scale_bytes(m));
}

/* Jobs of one record (host copy rec_host, device copy rec_dev of the same
 * bytes), appended per matrix to by_m[3]. Returns 0 or -1 (invalid record). */
static inline int wnr4_cuda_plan(const Wnr4Geo *g, const Wnr4Model *model, const unsigned char *rec_host, size_t total,
                                 uint32_t owner, const unsigned char *rec_dev, uint32_t index,
                                 unsigned char *const dst_s[3], unsigned char *const dst_w[3],
                                 Wnr4CudaRec *info, std::vector<Wnr4CudaJob> by_m[3]) {
    Wnr4Record r;
    uint32_t t, m;
    if (model->kind != g->kind || wnr4_parse(rec_host, total, g, &r) || r.model_id != model->id || r.owner != owner)
        return -1;
    info->model_id = r.model_id; info->owner = r.owner; info->flags = r.flags; info->G = r.G; info->kind = r.kind;
    info->anchors = (uint32_t)r.anchor[0] | (uint32_t)r.anchor[1] << 8 | (uint32_t)r.anchor[2] << 16;
    for (m = 0; m < 3u; m++) { info->scols[m] = g->scols[m]; info->dst_s[m] = dst_s[m]; info->dst_w[m] = dst_w[m]; }
    const size_t data_at = (size_t)(r.data - rec_host);
    for (t = 0; t < r.ntiles; t++) {   /* the offsets are read again here: check them again before any job is kept */
        uint32_t o0 = wnr4_u32(r.offsets + 4u * t), o1 = wnr4_u32(r.offsets + 4u * t + 4u);
        if (o1 < o0 || o1 - o0 < WNR4_TILE_STATE_BYTES || o1 > r.payload) return -1;
    }
    for (t = 0; t < r.ntiles; t++) {
        uint32_t o0 = wnr4_u32(r.offsets + 4u * t), o1 = wnr4_u32(r.offsets + 4u * t + 4u);
        Wnr4CudaJob j;
        wnr4_tile_cells(r.tile_base, g->cells, r.G, t, &j.m, &j.c0, &j.n);
        j.tile = rec_dev + data_at + o0; j.nwords = (o1 - o0 - WNR4_TILE_STATE_BYTES) / 2u; j.rec = index; j.t = t;
        by_m[j.m].push_back(j);
    }
    return 0;
}
static int wnr4_cuda_window = 1;         /* 1: word window in registers (default); 0: one global load per renormalisation */
static uint32_t wnr4_cuda_warps = 0u;     /* warps per block: 4, 8 or 16; 0 = by batch size (wnr4_cuda_jobs) */
static int wnr4_cuda_rows_global = -1;    /* 1: weight rows through L1, 0: shared memory, -1: L1 for NVFP4 at 16 warps */
static int wnr4_cuda_lds = 1;             /* 1: symbol start/end by two row loads (LD), 0: row_find min/max trees */
static thread_local uint32_t wnr4_cuda_planned_warps = 16u;   /* block size of the last wnr4_cuda_jobs on this thread */
/* Concatenates the per-matrix lists, padding each to a multiple of the block size (padding jobs have n = 0).
 * Block size: wnr4_cuda_warps, or by the number of tiles (few tiles: small blocks, lower latency; many:
 * 16 warps share one table copy, more resident warps per SM). Returns it; wnr4_cuda_launch uses it. */
static inline uint32_t wnr4_cuda_choose_warps(size_t tiles) {
    /* 1 V4.1 record = 1080 tiles: W=4 -> 270 blocks (fills all 170 SMs on RTX 5090),
     * 2 records = 2160 tiles -> W=8 (270 blocks), >=3 records -> W=16. */
    return wnr4_cuda_warps ? wnr4_cuda_warps : tiles < 1200u ? 4u : tiles < 2400u ? 8u : 16u;
}
static inline size_t wnr4_cuda_jobs_cap(const std::vector<Wnr4CudaJob> by_m[3], uint32_t *w_out) {
    const size_t tiles = by_m[0].size() + by_m[1].size() + by_m[2].size();
    const uint32_t W = wnr4_cuda_choose_warps(tiles);
    if (w_out) *w_out = W;
    size_t n = 0;
    for (uint32_t m = 0; m < 3u; m++) n += ((by_m[m].size() + W - 1u) / W) * W;
    return n;
}
static inline uint32_t wnr4_cuda_jobs_into(const std::vector<Wnr4CudaJob> by_m[3], Wnr4CudaJob *out, size_t *njobs_out) {
    uint32_t W = 16u;
    (void)wnr4_cuda_jobs_cap(by_m, &W);
    size_t pos = 0;
    for (uint32_t m = 0; m < 3u; m++) {
        const size_t sz = by_m[m].size();
        if (sz) { std::memcpy(out + pos, by_m[m].data(), sz * sizeof(Wnr4CudaJob)); pos += sz; }
        Wnr4CudaJob pad = {};
        pad.m = m;
        while (pos % W) out[pos++] = pad;
    }
    if (njobs_out) *njobs_out = pos;
    wnr4_cuda_planned_warps = W;
    return W;
}
static inline uint32_t wnr4_cuda_jobs(std::vector<Wnr4CudaJob> by_m[3], std::vector<Wnr4CudaJob> &jobs) {
    uint32_t W = 16u;
    const size_t cap = wnr4_cuda_jobs_cap(by_m, &W);
    jobs.resize(cap);
    size_t nj = 0;
    if (cap) wnr4_cuda_jobs_into(by_m, jobs.data(), &nj);
    else wnr4_cuda_planned_warps = W;
    jobs.resize(nj);
    return W;
}
static inline cudaError_t wnr4_cuda_launch(const Wnr4CudaTables &tab, const Wnr4CudaJob *jobs_dev, uint32_t njobs,
                                           const Wnr4CudaRec *recs_dev, uint32_t *errors_dev, cudaStream_t stream) {
    const uint32_t W = wnr4_cuda_planned_warps, blocks = (njobs + W - 1u) / W;
    const bool rg = wnr4_cuda_rows_global < 0 ? (tab.kind == 1u && W == 16u) : wnr4_cuda_rows_global != 0;
    const int smem = rg ? 0 : (int)(tab.rows * 32u);
    if (!blocks) return cudaSuccess;
    int dev_id = 0;
    if (smem && cudaGetDevice(&dev_id) != cudaSuccess) dev_id = 0;
#define WNR4_LAUNCH3(K_, I_, W_, G_, L_) do { \
        if (smem) { \
            static std::atomic<int> smem_done_[64]; \
            const bool cached_ = dev_id >= 0 && dev_id < 64 && smem_done_[dev_id].load(std::memory_order_relaxed) >= smem; \
            if (!cached_) { \
                cudaError_t e_ = cudaFuncSetAttribute(wnr4_cuda_kernel<K_, I_, W_, G_, L_>, \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                if (e_ != cudaSuccess) return e_; \
                if (dev_id >= 0 && dev_id < 64) smem_done_[dev_id].store(smem, std::memory_order_relaxed); \
            } \
        } \
        wnr4_cuda_kernel<K_, I_, W_, G_, L_><<<blocks, W_ * 32u, smem, stream>>>(jobs_dev, njobs, recs_dev, tab, errors_dev); \
    } while (0)
#define WNR4_LAUNCH2(K_, I_, W_, G_) do { if (wnr4_cuda_lds) WNR4_LAUNCH3(K_, I_, W_, G_, true); \
        else WNR4_LAUNCH3(K_, I_, W_, G_, false); } while (0)
#define WNR4_LAUNCH1(K_, I_, W_) do { if (rg) WNR4_LAUNCH2(K_, I_, W_, true); \
        else WNR4_LAUNCH2(K_, I_, W_, false); } while (0)
#define WNR4_LAUNCH(K_, I_) do { if (W == 4u) WNR4_LAUNCH1(K_, I_, 4u); else if (W == 8u) WNR4_LAUNCH1(K_, I_, 8u); \
        else WNR4_LAUNCH1(K_, I_, 16u); } while (0)
    if (W != 4u && W != 8u && W != 16u) return cudaErrorInvalidValue;
    if (tab.kind) WNR4_LAUNCH(1, true);   /* the word window is always on (A/B of 2026-11-21: +7..+19%) */
    else WNR4_LAUNCH(0, true);
#undef WNR4_LAUNCH
#undef WNR4_LAUNCH1
#undef WNR4_LAUNCH2
#undef WNR4_LAUNCH3
    return cudaGetLastError();
}

#endif
