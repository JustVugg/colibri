/* bench_sycl_bf16_head -- what the card can stream, and the lm head's kernel against it.
 *
 * DeepSeek V4.1's head is 129280 x 5120 BF16 (1.32 GB), read once a token by coli_cuda_bf16t_mv's bf16t2_kernel: a
 * work-item an output, 16 bytes of its own row a step, so the 16 lanes of a sub-group ask for 16 different cache lines.
 * This program has no part of the backend in it (its own queue, the default device: ONEAPI_DEVICE_SELECTOR chooses):
 *
 *  1. STREAM: read-only kernels over the same 1.32 GB (xor of every dword, checked against the host's), one per way
 *     of asking for the bytes: a lane its own row (16 or 64 bytes a load), a sub-group one contiguous run (lanes 16
 *     or 64 bytes apart, or 4 bytes apart = consecutive dwords), sub-groups of 16 and 32. The best is the ceiling.
 *  2. HEAD: the head's product in three layouts of the same bytes, each output's arithmetic exactly mvb_cpu's (eight
 *     accumulators, columns j, j+8, ... in order, fma from +0, ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))):
 *       row  today's (bf16t2_kernel's loads)
 *       l16  the rows of a sub-group interleaved 16 bytes a lane: a step of the sub-group is one contiguous run
 *       soa  the same interleaved a dword a lane: a step is four loads of consecutive dwords
 *     every variant memcmp-compared with `row`, and `row` with the host's reference on a small shape. 1, 2, 4 rows.
 *  3. AROUND: the input upload and the logits' download (pageable and page-locked host memory), and the kernel
 *     writing its logits straight into host USM.
 *
 * "BF16 HEAD BENCH PASS" when every comparison held; the numbers are the BENCH-* lines. */
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int fails;
static uint32_t state = 0x2545f491u;
static uint32_t rnd() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float bf(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t wrand() {   /* a finite bf16, subnormals included, no Inf/NaN (test_sycl_bf16t's) */
    uint16_t b;
    do b = (uint16_t)rnd(); while ((b & 0x7f80) == 0x7f80);
    if ((rnd() & 7) == 0) b &= 0x807f;
    else b = (uint16_t)((b & 0x807f) | ((0x70 + (rnd() % 24)) << 7));
    return b;
}
static void reference(float *y, const uint16_t *w, const float *x, int S, int I, int O) {
    for (int r = 0; r < S; r++)
        for (int o = 0; o < O; o++) {
            const uint16_t *row = w + (size_t)o * I;
            const float *xr = x + (size_t)r * I;
            float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int i = 0; i < I; i += 8)
                for (int j = 0; j < 8; j++) a[j] = std::fma(bf(row[i + j]), xr[i + j], a[j]);
            const float s04 = a[0] + a[4], s26 = a[2] + a[6], s15 = a[1] + a[5], s37 = a[3] + a[7];
            const float lo = s04 + s26, hi = s15 + s37;
            y[(size_t)r * O + o] = lo + hi;
        }
}

enum { ROW = 0, L16 = 1, SOA = 2 };
static const char *const lay_name[] = { "row", "l16", "soa" };

/* the layouts: `row` is the matrix as it is; the others hold blocks of SG outputs (the last one padded with zero rows),
 * a block's step t (8 columns = 4 dwords an output) as  l16: [lane][k]   soa: [k][lane] */
static size_t lay_dwords(int lay, int SG, int O, int I) {
    return lay == ROW ? (size_t)O * (I / 2) : (size_t)((O + SG - 1) / SG) * SG * (I / 2);
}
static void lay_build(uint32_t *dst, const uint16_t *w, int lay, int SG, int O, int I) {
    const uint32_t *src = (const uint32_t *)w;
    const int nit = I / 8;
    if (lay == ROW) { std::memcpy(dst, src, (size_t)O * (I / 2) * 4); return; }
    const int nb = (O + SG - 1) / SG;
    for (int b = 0; b < nb; b++) {
        uint32_t *blk = dst + (size_t)b * nit * SG * 4;
        for (int lane = 0; lane < SG; lane++) {
            const int o = b * SG + lane;
            for (int t = 0; t < nit; t++)
                for (int k = 0; k < 4; k++) {
                    const uint32_t v = o < O ? src[(size_t)o * (I / 2) + t * 4 + k] : 0u;
                    if (lay == L16) blk[((size_t)t * SG + lane) * 4 + k] = v;
                    else blk[((size_t)t * 4 + k) * SG + lane] = v;
                }
        }
    }
}

/* ---- the head's product ---- */
template <int SG, int LAY, int USELID> struct HeadK {
    const uint32_t *w; const float *x; float *y; int S, I, O;
    [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
        size_t b; int lane;
        if (USELID) { const size_t g = it.get_global_id(0); b = g / SG; lane = (int)(g % SG); }
        else {
            auto sg = it.get_sub_group();
            b = it.get_group(0) * (it.get_local_range(0) / SG) + sg.get_group_id()[0];
            lane = (int)sg.get_local_id()[0];
        }
        const size_t o = b * SG + lane;
        /* nothing past the matrix is read: `row` has no padding rows, the others pad their last block only (a whole
         * sub-group leaves together there; no sub-group operation below) */
        if (LAY == ROW ? o >= (size_t)O : b * SG >= (size_t)O) return;
        const int nit = I / 8;
        const uint32_t *base = LAY == ROW ? w + o * (size_t)(I / 2) : w + b * (size_t)nit * (SG * 4);
        float acc[4][8];
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int j = 0; j < 8; j++) acc[r][j] = 0.0f;
#pragma unroll 4
        for (int t = 0; t < nit; t++) {
            uint32_t wd[4];
            if (LAY == ROW) {
                const sycl::uint4 q = *(const sycl::uint4 *)(base + (size_t)t * 4);
                wd[0] = q.x(); wd[1] = q.y(); wd[2] = q.z(); wd[3] = q.w();
            } else if (LAY == L16) {
                const sycl::uint4 q = *(const sycl::uint4 *)(base + ((size_t)t * SG + lane) * 4);
                wd[0] = q.x(); wd[1] = q.y(); wd[2] = q.z(); wd[3] = q.w();
            } else {
#pragma unroll
                for (int k = 0; k < 4; k++) wd[k] = base[((size_t)t * 4 + k) * SG + lane];
            }
            float wv[8];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                wv[2 * k] = sycl::bit_cast<float>(wd[k] << 16);
                wv[2 * k + 1] = sycl::bit_cast<float>(wd[k] & 0xffff0000u);
            }
#pragma unroll
            for (int r = 0; r < 4; r++) {
                if (r < S) {
                    const float *xr = x + (size_t)r * I + (size_t)t * 8;
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[r][j] = sycl::fma(wv[j], xr[j], acc[r][j]);
                }
            }
        }
        if (o < (size_t)O) {
#pragma unroll
            for (int r = 0; r < 4; r++) {
                if (r < S) {
                    const float a0 = acc[r][0] + acc[r][4], a1 = acc[r][1] + acc[r][5], a2 = acc[r][2] + acc[r][6], a3 = acc[r][3] + acc[r][7];
                    const float b0 = a0 + a2, b1 = a1 + a3;
                    y[(size_t)r * O + o] = b0 + b1;
                }
            }
        }
    }
};

/* ---- the read-only kernels: xor of every dword of [0, nd), R dwords an output "row" ---- */
enum { P_ROW16 = 0, P_CLU16 = 1, P_SOA4 = 2, P_ROW64 = 3, P_CLU64 = 4, P_N = 5 };
static const char *const pat_name[] = { "row16B", "clu16B", "soa4B", "row64B", "clu64B" };
template <int SG, int PAT> struct StreamK {
    const uint32_t *w; uint32_t *out; int R;   /* R dwords a work-item, a multiple of 16 */
    [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
        auto sg = it.get_sub_group();
        const size_t b = it.get_group(0) * (it.get_local_range(0) / SG) + sg.get_group_id()[0];
        const int lane = (int)sg.get_local_id()[0];
        const size_t g = b * SG + lane;
        const uint32_t *blk = w + b * (size_t)SG * R, *row = w + g * (size_t)R;
        uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if (PAT == P_ROW16 || PAT == P_CLU16) {
#pragma unroll 4
            for (int t = 0; t < R / 4; t++) {
                const sycl::uint4 q = PAT == P_ROW16 ? *(const sycl::uint4 *)(row + (size_t)t * 4)
                                                     : *(const sycl::uint4 *)(blk + ((size_t)t * SG + lane) * 4);
                a0 ^= q.x(); a1 ^= q.y(); a2 ^= q.z(); a3 ^= q.w();
            }
        } else if (PAT == P_SOA4) {
#pragma unroll 4
            for (int t = 0; t < R / 4; t++) {
                a0 ^= blk[((size_t)t * 4 + 0) * SG + lane]; a1 ^= blk[((size_t)t * 4 + 1) * SG + lane];
                a2 ^= blk[((size_t)t * 4 + 2) * SG + lane]; a3 ^= blk[((size_t)t * 4 + 3) * SG + lane];
            }
        } else {
            for (int t = 0; t < R / 16; t++) {
                const sycl::vec<uint32_t, 16> q = PAT == P_ROW64 ? *(const sycl::vec<uint32_t, 16> *)(row + (size_t)t * 16)
                                                                 : *(const sycl::vec<uint32_t, 16> *)(blk + ((size_t)t * SG + lane) * 16);
#pragma unroll
                for (int e = 0; e < 16; e += 4) { a0 ^= q[e]; a1 ^= q[e + 1]; a2 ^= q[e + 2]; a3 ^= q[e + 3]; }
            }
        }
        out[g] = a0 ^ a1 ^ a2 ^ a3;
    }
};

struct Stat { double min_ms, med_ms; };
template <typename F> static Stat timed(int warm, int n, F f) {
    for (int k = 0; k < warm; k++) f();
    std::vector<double> t(n);
    for (int k = 0; k < n; k++) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        t[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::sort(t.begin(), t.end());
    return { t[0], t[n / 2] };
}
template <typename K> static void launch(sycl::queue &q, size_t items, int local, const K &k) {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(items), sycl::range<1>(local)), k).wait();
}
template <int SG, int LAY, int USELID>
static void head_run(sycl::queue &q, const uint32_t *w, const float *x, float *y, int S, int I, int O, int local) {
    const size_t items = (size_t)((O + local - 1) / local) * local;
    launch(q, items, local, HeadK<SG, LAY, USELID>{w, x, y, S, I, O});
}
static void head_any(sycl::queue &q, int sg, int lay, int uselid, const uint32_t *w, const float *x, float *y, int S, int I, int O, int local) {
#define HK(SGV, L, U) if (sg == SGV && lay == L && uselid == U) { head_run<SGV, L, U>(q, w, x, y, S, I, O, local); return; }
    HK(16, ROW, 1) HK(16, ROW, 0) HK(16, L16, 0) HK(16, L16, 1) HK(16, SOA, 0) HK(16, SOA, 1)
#ifndef BENCH_NO_SG32
    HK(32, ROW, 1) HK(32, L16, 0) HK(32, SOA, 0) HK(32, SOA, 1)
#endif
#undef HK
    std::fprintf(stderr, "no such head variant\n"); std::exit(2);
}
static void stream_any(sycl::queue &q, int sg, int pat, const uint32_t *w, uint32_t *out, int R, size_t items, int local) {
#define SK(SGV, P) if (sg == SGV && pat == P) { launch(q, items, local, StreamK<SGV, P>{w, out, R}); return; }
    SK(16, P_ROW16) SK(16, P_CLU16) SK(16, P_SOA4) SK(16, P_ROW64) SK(16, P_CLU64)
#ifndef BENCH_NO_SG32
    SK(32, P_ROW16) SK(32, P_CLU16) SK(32, P_SOA4) SK(32, P_ROW64) SK(32, P_CLU64)
#endif
#undef SK
    std::fprintf(stderr, "no such stream variant\n"); std::exit(2);
}

static bool sg_offered(sycl::queue &q, int sg) {
    for (size_t s : q.get_device().get_info<sycl::info::device::sub_group_sizes>()) if ((int)s == sg) return true;
    return false;
}

/* ---- round 2: today's kernel as it is (prod), and the same product with the rows a compile-time count (NR), the
 * weights in blocks of LB outputs (0 = row; 16 or 32 = a block's step [lane][k], served by sub-groups of 16 or 32),
 * and x either [r][I] (XI 0) or a step's rows together, [t][NR][8] (XI 1: one or two uniform loads a step) ---- */
#define BF2_ROWS 4
struct ProdK {   /* bf16t2_kernel, its lines */
    float *y; int ystride; const float *x; int xstride; const uint16_t *w; int S, I, O;
    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<3> item) const {
        const int o = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
        const int r0 = item.get_group(1) * BF2_ROWS;
        const int nr = S - r0 < BF2_ROWS ? S - r0 : BF2_ROWS;
        if (o >= O) return;
        const uint32_t *row = (const uint32_t *)(w + (size_t)o * I);
        float acc[BF2_ROWS][8];
#pragma unroll
        for (int r = 0; r < BF2_ROWS; r++)
#pragma unroll
            for (int j = 0; j < 8; j++) acc[r][j] = 0.0f;
#pragma unroll 4
        for (int i = 0; i < I; i += 8) {
            const sycl::uint4 q = *(const sycl::uint4 *)(row + (i >> 1));
            const uint32_t wd[4] = { q.x(), q.y(), q.z(), q.w() };
            float wv[8];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                wv[2 * k] = sycl::bit_cast<float>(wd[k] << 16);
                wv[2 * k + 1] = sycl::bit_cast<float>(wd[k] & 0xffff0000u);
            }
#pragma unroll
            for (int r = 0; r < BF2_ROWS; r++) {
                if (r < nr) {
                    const float *xr = x + (size_t)(r0 + r) * xstride + i;
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[r][j] = sycl::fma(wv[j], xr[j], acc[r][j]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < BF2_ROWS; r++) {
            if (r < nr) {
                const float a0 = acc[r][0] + acc[r][4], a1 = acc[r][1] + acc[r][5], a2 = acc[r][2] + acc[r][6], a3 = acc[r][3] + acc[r][7];
                const float b0 = a0 + a2, b1 = a1 + a3;
                y[(size_t)(r0 + r) * ystride + o] = b0 + b1;
            }
        }
    }
};
static void prod_run(sycl::queue &q, const uint32_t *w, const float *x, float *y, int S, int I, int O) {
    const sycl::range<3> grid(1, (size_t)((S + BF2_ROWS - 1) / BF2_ROWS), (size_t)((O + 255) / 256)), block(1, 1, 256);
    q.parallel_for(sycl::nd_range<3>(grid * block, block), ProdK{y, O, x, I, (const uint16_t *)w, S, I, O}).wait();
}

static size_t blk_dwords(int LB, int O, int I) { return LB ? (size_t)((O + LB - 1) / LB) * LB * (I / 2) : (size_t)O * (I / 2); }
static void blk_build(uint32_t *dst, const uint16_t *w, int LB, int O, int I) { lay_build(dst, w, LB ? L16 : ROW, LB ? LB : 16, O, I); }
/* x for XI 1: [t][NR][8] */
static void xi_build(float *dst, const float *x, int NR, int S, int I) {
    for (int t = 0; t < I / 8; t++)
        for (int r = 0; r < NR; r++)
            for (int j = 0; j < 8; j++) dst[((size_t)t * NR + r) * 8 + j] = r < S ? x[(size_t)r * I + t * 8 + j] : 0.0f;
}
template <int SG, int LB, int NR, int XI> struct Head2K {
    const uint32_t *w; const float *x; float *y; int S, I, O;   /* S <= NR rows are written */
    [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
        auto sg = it.get_sub_group();
        const size_t sgi = it.get_group(0) * (it.get_local_range(0) / SG) + sg.get_group_id()[0];
        const int lane = (int)sg.get_local_id()[0];
        const size_t o = sgi * SG + lane;
        const int nit = I / 8;
        const uint32_t *base; size_t step;
        if (LB == 0) {
            if (o >= (size_t)O) return;
            base = w + o * (size_t)(I / 2); step = 4;
        } else {
            if (sgi * SG >= (size_t)O) return;   /* past the last block: the whole sub-group */
            size_t blk; int bl;
            if (LB == SG) { blk = sgi; bl = lane; }
            else if (LB == 2 * SG) { blk = sgi >> 1; bl = (int)(sgi & 1) * SG + lane; }   /* half a block: every other run */
            else {                                                                        /* SG == 2 * LB: two blocks */
                blk = sgi * 2 + (lane >= LB ? 1 : 0); bl = lane & (LB - 1);
                if (blk * LB >= (size_t)O) return;   /* the second may be past the last one */
            }
            base = w + blk * (size_t)nit * (LB * 4) + (size_t)bl * 4; step = (size_t)LB * 4;
        }
        float acc[NR][8];
#pragma unroll
        for (int r = 0; r < NR; r++)
#pragma unroll
            for (int j = 0; j < 8; j++) acc[r][j] = 0.0f;
#pragma unroll 4
        for (int t = 0; t < nit; t++) {
            const sycl::uint4 q = *(const sycl::uint4 *)(base + (size_t)t * step);
            const uint32_t wd[4] = { q.x(), q.y(), q.z(), q.w() };
            float wv[8];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                wv[2 * k] = sycl::bit_cast<float>(wd[k] << 16);
                wv[2 * k + 1] = sycl::bit_cast<float>(wd[k] & 0xffff0000u);
            }
            if (XI) {
                const float *xs = x + (size_t)t * NR * 8;
                float xv[NR * 8];
                if (NR == 1) {
                    const sycl::vec<float, 8> a = *(const sycl::vec<float, 8> *)xs;
#pragma unroll
                    for (int j = 0; j < 8; j++) xv[j] = a[j];
                } else {   /* NR 2 or 4 (3 rows by step run as 4) */
#pragma unroll
                    for (int j = 0; j < NR * 8; j++) xv[j] = 0.0f;
#pragma unroll
                    for (int h = 0; h < NR / 2; h++) {
                        const sycl::vec<float, 16> a = *(const sycl::vec<float, 16> *)(xs + h * 16);
#pragma unroll
                        for (int j = 0; j < 16; j++) xv[h * 16 + j] = a[j];
                    }
                }
#pragma unroll
                for (int r = 0; r < NR; r++)
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[r][j] = sycl::fma(wv[j], xv[r * 8 + j], acc[r][j]);
            } else {
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    const float *xr = x + (size_t)r * I + (size_t)t * 8;
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[r][j] = sycl::fma(wv[j], xr[j], acc[r][j]);
                }
            }
        }
        if (o < (size_t)O) {
#pragma unroll
            for (int r = 0; r < NR; r++) {
                if (r < S) {
                    const float a0 = acc[r][0] + acc[r][4], a1 = acc[r][1] + acc[r][5], a2 = acc[r][2] + acc[r][6], a3 = acc[r][3] + acc[r][7];
                    const float b0 = a0 + a2, b1 = a1 + a3;
                    y[(size_t)r * O + o] = b0 + b1;
                }
            }
        }
    }
};
struct V2 { int sg, lb, xi; };
static const V2 v2s[] = { {16, 0, 0}, {16, 0, 1}, {32, 0, 1}, {16, 16, 0}, {16, 16, 1}, {32, 16, 1}, {16, 32, 0}, {16, 32, 1}, {32, 32, 0}, {32, 32, 1} };   /* {16, 32, 0}: the backend's bf16t3_kernel */
static void head2_any(sycl::queue &q, const V2 &v, int NR, const uint32_t *w, const float *x, float *y, int S, int I, int O) {
    const size_t items = (size_t)((O + 255) / 256) * 256;
#define H2(SGV, LBV, XIV) \
    if (v.sg == SGV && v.lb == LBV && v.xi == XIV) { \
        if (NR == 1) launch(q, items, 256, Head2K<SGV, LBV, 1, XIV>{w, x, y, S, I, O}); \
        else if (NR == 2) launch(q, items, 256, Head2K<SGV, LBV, 2, XIV>{w, x, y, S, I, O}); \
        else if (NR == 3 && !XIV) launch(q, items, 256, Head2K<SGV, LBV, 3, 0>{w, x, y, S, I, O}); \
        else launch(q, items, 256, Head2K<SGV, LBV, 4, XIV>{w, x, y, S, I, O}); \
        return; }
    H2(16, 0, 0) H2(16, 0, 1) H2(32, 0, 1) H2(16, 16, 0) H2(16, 16, 1) H2(32, 16, 1) H2(16, 32, 0) H2(16, 32, 1) H2(32, 32, 0) H2(32, 32, 1)
#undef H2
    std::fprintf(stderr, "no such head2 variant\n"); std::exit(2);
}
struct Stat3 { double min_ms, med_ms, p90_ms; };
template <typename F> static Stat3 timed3(int warm, int n, F f) {
    for (int k = 0; k < warm; k++) f();
    std::vector<double> t(n);
    for (int k = 0; k < n; k++) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        t[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::sort(t.begin(), t.end());
    return { t[0], t[n / 2], t[(size_t)(n * 0.9)] };
}
static void head2_suite(sycl::queue &q, int O, int I, int reps, bool with_ref) {
    std::vector<uint16_t> w((size_t)O * I);
    for (auto &b : w) b = wrand();
    const double gb = (double)O * I * 2 / 1e9;
    std::vector<float> x((size_t)4 * I), xi((size_t)4 * I);
    for (auto &v : x) v = ((int)(rnd() % 20001) - 10000) / 7919.0f;
    float *dx = sycl::aligned_alloc_device<float>(4096, (size_t)4 * I, q), *dxi = sycl::aligned_alloc_device<float>(4096, (size_t)4 * I, q);
    float *dy = sycl::malloc_device<float>((size_t)4 * O, q);
    q.memcpy(dx, x.data(), (size_t)4 * I * 4).wait();
    uint32_t *dw[3] = { nullptr, nullptr, nullptr };   /* row, blocks of 16, blocks of 32 */
    const int lbs[3] = { 0, 16, 32 };
    for (int k = 0; k < 3; k++) {
        const size_t nd = blk_dwords(lbs[k], O, I);
        std::vector<uint32_t> host(nd);
        blk_build(host.data(), w.data(), lbs[k], O, I);
        dw[k] = sycl::aligned_alloc_device<uint32_t>(4096, nd, q);
        if (!dw[k]) { std::fprintf(stderr, "FAIL: device allocation\n"); std::exit(1); }
        q.memcpy(dw[k], host.data(), nd * 4).wait();
    }
    std::vector<float> got((size_t)4 * O), keep((size_t)4 * O), ref;
    for (int S : {1, 2, 3, 4}) {
        const int NRx = S == 3 ? 4 : S;   /* x by step: 1, 2 or 4 rows a step; x by row: exactly S */
        /* today's kernel: the bits every other is held to */
        q.fill(dy, NAN, (size_t)4 * O).wait();
        prod_run(q, dw[0], dx, dy, S, I, O);
        q.memcpy(keep.data(), dy, (size_t)S * O * 4).wait();
        if (with_ref) {
            ref.assign((size_t)S * O, 0.0f);
            reference(ref.data(), w.data(), x.data(), S, I, O);
            size_t d = 0;
            for (size_t i = 0; i < (size_t)S * O; i++) d += std::memcmp(&keep[i], &ref[i], 4) != 0;
            std::printf("  %s prod O %d I %d S %d against the host reference: %zu of %zu differ\n", d ? "FAIL" : "ok  ", O, I, S, d, (size_t)S * O);
            if (d) fails++;
        }
        if (reps) {
            const Stat3 st = timed3(3, reps, [&] { prod_run(q, dw[0], dx, dy, S, I, O); });
            std::printf("BENCH-HEAD2 prod (today's kernel)        S %d: min %.3f median %.3f p90 %.3f ms, %.0f GB/s (median)\n", S, st.min_ms, st.med_ms, st.p90_ms, gb / (st.med_ms * 1e-3));
        }
        xi_build(xi.data(), x.data(), NRx, S, I);
        q.memcpy(dxi, xi.data(), (size_t)NRx * I * 4).wait();
        for (const V2 &v : v2s) {
            if (!sg_offered(q, v.sg)) continue;
            const uint32_t *wl = dw[v.lb == 0 ? 0 : v.lb == 16 ? 1 : 2];
            const float *xl = v.xi ? dxi : dx;
            const int NR = v.xi ? NRx : S;
            q.fill(dy, NAN, (size_t)4 * O).wait();
            head2_any(q, v, NR, wl, xl, dy, S, I, O);
            q.memcpy(got.data(), dy, (size_t)S * O * 4).wait();
            size_t d = 0;
            for (size_t i = 0; i < (size_t)S * O; i++) d += std::memcmp(&got[i], &keep[i], 4) != 0;
            if (d) fails++;
            if (!reps) { std::printf("  %s sg%d lb%d xi%d nr%d O %d I %d S %d: %zu of %zu differ from prod\n", d ? "FAIL" : "ok  ", v.sg, v.lb, v.xi, NR, O, I, S, d, (size_t)S * O); continue; }
            const Stat3 st = timed3(3, reps, [&] { head2_any(q, v, NR, wl, xl, dy, S, I, O); });
            std::printf("BENCH-HEAD2 sg%d blocks %2d x %s rows %d S %d: min %.3f median %.3f p90 %.3f ms, %.0f GB/s (median), %s\n",
                        v.sg, v.lb, v.xi ? "by step" : "by row ", NR, S, st.min_ms, st.med_ms, st.p90_ms, gb / (st.med_ms * 1e-3), d ? "BITS DIFFER" : "same bits");
        }
    }
    for (int k = 0; k < 3; k++) sycl::free(dw[k], q);
    sycl::free(dx, q); sycl::free(dxi, q); sycl::free(dy, q);
}

struct Variant { int sg, lay, uselid; };
static const Variant variants[] = { {16, ROW, 1}, {16, ROW, 0}, {16, L16, 0}, {16, L16, 1}, {16, SOA, 0}, {16, SOA, 1},
                                    {32, ROW, 1}, {32, L16, 0}, {32, SOA, 0}, {32, SOA, 1} };

/* every variant against `row` (and `row` against the host when small); timing when `reps` */
static void head_suite(sycl::queue &q, int O, int I, int local, int reps, bool with_ref) {
    std::vector<uint16_t> w((size_t)O * I);
    for (auto &b : w) b = wrand();
    const double gb = (double)O * I * 2 / 1e9;
    std::vector<float> x((size_t)4 * I);
    for (auto &v : x) v = ((int)(rnd() % 20001) - 10000) / 7919.0f;
    float *dx = sycl::malloc_device<float>((size_t)4 * I, q);
    float *dy = sycl::malloc_device<float>((size_t)4 * O, q);
    q.memcpy(dx, x.data(), (size_t)4 * I * 4).wait();
    std::vector<float> got((size_t)4 * O), ref, keep[5];
    for (const Variant &v : variants) {
        if (!sg_offered(q, v.sg)) continue;
        const size_t nd = lay_dwords(v.lay, v.sg, O, I);
        std::vector<uint32_t> host(nd);
        lay_build(host.data(), w.data(), v.lay, v.sg, O, I);
        uint32_t *dw = sycl::aligned_alloc_device<uint32_t>(4096, nd, q);
        if (!dw) { std::fprintf(stderr, "FAIL: device allocation of %zu dwords\n", nd); std::exit(1); }
        q.memcpy(dw, host.data(), nd * 4).wait();
        std::vector<uint32_t>().swap(host);
        for (int S : {1, 2, 4}) {
            q.fill(dy, NAN, (size_t)4 * O).wait();
            head_any(q, v.sg, v.lay, v.uselid, dw, dx, dy, S, I, O, local);
            q.memcpy(got.data(), dy, (size_t)S * O * 4).wait();
            const bool first = &v == &variants[0];
            if (first) {
                if (with_ref) {
                    ref.assign((size_t)S * O, 0.0f);
                    reference(ref.data(), w.data(), x.data(), S, I, O);
                    size_t d = 0;
                    for (size_t i = 0; i < (size_t)S * O; i++) d += std::memcmp(&got[i], &ref[i], 4) != 0;
                    std::printf("  %s row sg16 O %d I %d S %d against the host reference: %zu of %zu differ\n", d ? "FAIL" : "ok  ", O, I, S, d, (size_t)S * O);
                    if (d) fails++;
                }
            }
            if (first) keep[S].assign(got.begin(), got.begin() + (size_t)S * O);
            size_t d = 0;
            for (size_t i = 0; i < (size_t)S * O; i++) d += std::memcmp(&got[i], &keep[S][i], 4) != 0;
            if (d) fails++;
            if (!reps) { std::printf("  %s %s sg%d %s O %d I %d S %d: %zu of %zu differ from row\n", d ? "FAIL" : "ok  ", lay_name[v.lay], v.sg, v.uselid ? "lid" : "sgid", O, I, S, d, (size_t)S * O); continue; }
            const Stat st = timed(3, reps, [&] { head_any(q, v.sg, v.lay, v.uselid, dw, dx, dy, S, I, O, local); });
            std::printf("BENCH-HEAD %s sg%d %s local %d O %d I %d S %d: min %.3f ms median %.3f ms, %.0f GB/s of weights (min), %s\n",
                        lay_name[v.lay], v.sg, v.uselid ? "lid " : "sgid", local, O, I, S, st.min_ms, st.med_ms, gb / (st.min_ms * 1e-3),
                        d ? "BITS DIFFER" : "same bits as row");
        }
        sycl::free(dw, q);
    }
    sycl::free(dx, q); sycl::free(dy, q);
}

static void stream_suite(sycl::queue &q, int O, int I, int reps) {
    const int R = I / 2;
    const size_t nd = (size_t)O * R;
    std::vector<uint32_t> host(nd);
    uint32_t want = 0;
    for (auto &v : host) { v = rnd(); want ^= v; }
    uint32_t *dw = sycl::aligned_alloc_device<uint32_t>(4096, nd, q), *dw2 = sycl::aligned_alloc_device<uint32_t>(4096, nd, q);
    uint32_t *dout = sycl::malloc_device<uint32_t>((size_t)O, q);
    if (!dw || !dw2 || !dout) { std::fprintf(stderr, "FAIL: stream allocation\n"); std::exit(1); }
    q.memcpy(dw, host.data(), nd * 4).wait();
    const double gb = (double)nd * 4 / 1e9;
    std::vector<uint32_t> out((size_t)O);
    for (int local : {256, 64, 128})
        for (int sg : {16, 32})
            for (int pat = 0; pat < P_N; pat++) {
                if (O % local || !sg_offered(q, sg)) continue;
                if (local != 256 && pat != P_ROW16 && pat != P_CLU64 && pat != P_SOA4) continue;
                q.fill(dout, 0u, (size_t)O).wait();
                stream_any(q, sg, pat, dw, dout, R, (size_t)O, local);
                q.memcpy(out.data(), dout, (size_t)O * 4).wait();
                uint32_t got = 0;
                for (uint32_t v : out) got ^= v;
                if (got != want) fails++;
                const Stat st = timed(3, reps, [&] { stream_any(q, sg, pat, dw, dout, R, (size_t)O, local); });
                std::printf("BENCH-STREAM %s sg%d local %d (%.2f GB): min %.3f ms median %.3f ms, %.0f GB/s (min) %.0f GB/s (median), xor %s\n",
                            pat_name[pat], sg, local, gb, st.min_ms, st.med_ms, gb / (st.min_ms * 1e-3), gb / (st.med_ms * 1e-3), got == want ? "ok" : "WRONG");
            }
    {   /* the driver's own copy, device to device: a read and a write of every byte */
        const Stat st = timed(2, reps, [&] { q.memcpy(dw2, dw, nd * 4).wait(); });
        std::printf("BENCH-STREAM memcpy device->device (%.2f GB): min %.3f ms median %.3f ms, %.0f GB/s read (and as much written)\n", gb, st.min_ms, st.med_ms, gb / (st.min_ms * 1e-3));
    }
    sycl::free(dw, q); sycl::free(dw2, q); sycl::free(dout, q);
}

/* the call's other parts: x up (S x 5120 floats), the logits down (S x 129280 floats) */
static void around_suite(sycl::queue &q, int O, int I, int reps) {
    float *dx = sycl::malloc_device<float>((size_t)4 * I, q), *dy = sycl::malloc_device<float>((size_t)4 * O, q);
    float *hy = sycl::malloc_host<float>((size_t)4 * O, q);
    std::vector<float> x((size_t)4 * I, 0.5f), y((size_t)4 * O);
    q.fill(dy, 1.0f, (size_t)4 * O).wait();
    for (int S : {1, 2, 4}) {
        const Stat up = timed(3, reps, [&] { q.memcpy(dx, x.data(), (size_t)S * I * 4).wait(); });
        const Stat dn = timed(3, reps, [&] { q.memcpy(y.data(), dy, (size_t)S * O * 4).wait(); });
        std::printf("BENCH-AROUND S %d: input upload %.3f ms (median %.3f), logits download to pageable memory %.3f ms (median %.3f)",
                    S, up.min_ms, up.med_ms, dn.min_ms, dn.med_ms);
        if (hy) {
            const Stat dh = timed(3, reps, [&] { q.memcpy(hy, dy, (size_t)S * O * 4).wait(); });
            std::printf(", to page-locked host USM %.3f ms (median %.3f)", dh.min_ms, dh.med_ms);
        }
        std::printf("\n");
    }
    {   /* an empty kernel's submit + wait: the floor of any launch */
        const Stat e = timed(3, reps, [&] { q.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)), [=](sycl::nd_item<1> it) { if (it.get_global_id(0) == 0) dy[0] = 1.0f; }).wait(); });
        std::printf("BENCH-AROUND a one-group kernel, submit + wait: %.3f ms (median %.3f)\n", e.min_ms, e.med_ms);
    }
    sycl::free(dx, q); sycl::free(dy, q); if (hy) sycl::free(hy, q);
}

/* the head's kernel writing its logits into host USM (no download), against device USM + download */
static void hostwrite_suite(sycl::queue &q, int O, int I, int reps) {
    std::vector<uint16_t> w((size_t)O * I);
    for (auto &b : w) b = wrand();
    const size_t nd = lay_dwords(ROW, 16, O, I);
    uint32_t *dw = sycl::aligned_alloc_device<uint32_t>(4096, nd, q);
    q.memcpy(dw, w.data(), nd * 4).wait();
    float *dx = sycl::malloc_device<float>((size_t)4 * I, q), *dy = sycl::malloc_device<float>((size_t)4 * O, q);
    float *hy = sycl::malloc_host<float>((size_t)4 * O, q);
    std::vector<float> x((size_t)4 * I, 0.25f), y((size_t)4 * O);
    q.memcpy(dx, x.data(), (size_t)4 * I * 4).wait();
    for (int S : {1, 4}) {
        const Stat a = timed(3, reps, [&] { head_any(q, 16, ROW, 1, dw, dx, dy, S, I, O, 256); q.memcpy(y.data(), dy, (size_t)S * O * 4).wait(); });
        std::printf("BENCH-HOSTWRITE S %d: kernel to device memory + download %.3f ms (median %.3f)", S, a.min_ms, a.med_ms);
        if (hy) {
            const Stat b = timed(3, reps, [&] { head_any(q, 16, ROW, 1, dw, dx, hy, S, I, O, 256); });
            size_t d = 0;
            for (size_t i = 0; i < (size_t)S * O; i++) d += std::memcmp(&hy[i], &y[i], 4) != 0;
            if (d) fails++;
            std::printf(", kernel writing host USM %.3f ms (median %.3f), %s", b.min_ms, b.med_ms, d ? "BITS DIFFER" : "same bits");
        }
        std::printf("\n");
    }
    sycl::free(dw, q); sycl::free(dx, q); sycl::free(dy, q); if (hy) sycl::free(hy, q);
}

int main(int argc, char **argv) {
    bool small = false;
    int O = 129280, I = 5120, reps = 25;   /* --dims O I (O a multiple of 256, I of 32) and --reps N: a smaller run */
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--small")) small = true;
        else if (!std::strcmp(argv[i], "--dims") && i + 2 < argc) { O = std::atoi(argv[i + 1]); I = std::atoi(argv[i + 2]); i += 2; }
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) i++;   /* the harness passes it: the selector has chosen */
        else if (!std::strncmp(argv[i], "--device=", 9)) {}
        else { std::fprintf(stderr, "usage: %s [--small] [--dims O I] [--reps N] [--device N]\n", argv[0]); return 2; }
    }
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
    const auto dev = q.get_device();
    std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("sub-group sizes:");
    for (size_t s : dev.get_info<sycl::info::device::sub_group_sizes>()) std::printf(" %zu", s);
    std::printf("\n");
    if (!sg_offered(q, 16)) { std::printf("sub-group size 16 is not offered: nothing run\nBF16 HEAD BENCH SKIP\n"); return 0; }
    /* bits: small shapes against the host, odd O (padding in the interleaved layouts), I a multiple of 8 only */
    const char *only = std::getenv("BENCH_SUITES");   /* e.g. head2 : only those suites (default all) */
    auto want = [&](const char *name) { return !only || std::strstr(only, name); };
    head_suite(q, 384, 5120, 256, 0, true);
    head_suite(q, 513, 1032, 256, 0, true);
    head_suite(q, 1000, 64, 256, 0, true);
    head2_suite(q, 384, 5120, 0, true);
    head2_suite(q, 513, 1032, 0, true);
    head2_suite(q, 1000, 64, 0, true);
    if (small) { std::printf(fails ? "BF16 HEAD BENCH FAIL\n" : "BF16 HEAD BENCH PASS\n"); return fails ? 1 : 0; }
    if (O < 256 || O % 256 || I < 32 || I % 32 || reps < 1) { std::fprintf(stderr, "--dims: O a multiple of 256, I of 32\n"); return 2; }
    if (want("stream")) stream_suite(q, O, I, reps);
    if (want("head1")) head_suite(q, O, I, 256, reps, false);
    if (want("head2")) head2_suite(q, O, I, 2 * reps, false);
    if (want("around")) around_suite(q, O, I, 2 * reps);
    if (want("hostwrite")) hostwrite_suite(q, O, I, reps);
    std::printf(fails ? "BF16 HEAD BENCH FAIL\n" : "BF16 HEAD BENCH PASS\n");
    return fails ? 1 : 0;
}
