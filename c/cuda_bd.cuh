/* cuda_bd.cuh -- a GGUF's block formats on the CUDA device (backend_cuda.cu): fmt 100 + t is
 * ggml block type t, each row as the file stores it, padded to whole words (rowWords), and
 * blockdec_host.h's table (the IQ grids, ksigns, kvalues) in device memory. The CUDA twin
 * of the Vulkan shaders' decoders, the same steps and the same table:
 *   - bdi_step: shaders/bdstep_i8.glsl, the CPU's integer form (ggml_quants.h's
 *     gq_quant_act_ref and gq_dot_i8): x quantized to int8 per 32 (the CPU's codes, scales
 *     and code sums: IEEE division, rounding to nearest even), each 32-weight step decoded
 *     to int8 once and dotted with up to NR rows of x in packed 4 x 8 products (dp4a), per
 *     group dx * (sw * sum(w8 * xq) - mw * xs). The same integers and scales as the CPU's:
 *     the two differ only in the order of the float sums. Types 8 Q8_0, 10 Q2_K, 11 Q3_K,
 *     12 Q4_K, 13 Q5_K, 14 Q6_K, 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS, 19 IQ1_S, 20 IQ4_NL,
 *     21 IQ3_S, 22 IQ2_S, 23 IQ4_XS, 29 IQ1_M, 42 Q2_0 (bdi_has).
 *   - bdf_decode: shaders/blockdec.glsl's general path, every block type, a 32-weight unit
 *     as int weights with a scale and a min per 16, in floats (the other types: Q4_0, Q4_1,
 *     Q5_0, Q5_1, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0).
 * A row's words: W (32-bit, 16-byte aligned rows for Q4_K and Q5_K, whose 144- and 176-byte
 * blocks keep it); a field at any even byte comes from one word or two (bd_ldu). */
#ifndef CUDA_BD_CUH
#define CUDA_BD_CUH
#if !(defined(__HIP__) || defined(__HIP_PLATFORM_AMD__))
#include <cuda_fp16.h>   /* HIP: hip_fp16.h, through backend_gpu_compat.h */
#endif
#include "blockdec_host.h"

typedef unsigned int bd_u32;

__device__ __forceinline__ bd_u32 bd_w(const bd_u32 *__restrict__ W, bd_u32 i) { return __ldg(W + i); }
__device__ __forceinline__ uint4 bd_w4(const bd_u32 *__restrict__ W, bd_u32 i) { return __ldg(reinterpret_cast<const uint4 *>(W) + i); }
/* four bytes from an even byte offset: one word when aligned, two otherwise; the second word
 * only when the four bytes run into it, so a read never leaves the row */
__device__ __forceinline__ bd_u32 bd_ldu(const bd_u32 *__restrict__ W, bd_u32 b) {
    bd_u32 a = b >> 2;
    return (b & 2u) == 0u ? bd_w(W, a) : (bd_w(W, a) >> 16) | (bd_w(W, a + 1u) << 16);
}
__device__ __forceinline__ float bd_h2f(bd_u32 h) { return __half2float(__ushort_as_half((unsigned short)(h & 0xffffu))); }
__device__ __forceinline__ float bd_f16at(const bd_u32 *__restrict__ W, bd_u32 b) { return bd_h2f(bd_w(W, b >> 2) >> ((b & 2u) * 8u)); }
__device__ __forceinline__ bd_u32 bd_u8at(const bd_u32 *__restrict__ W, bd_u32 b) { return (bd_w(W, b >> 2) >> ((b & 3u) * 8u)) & 0xffu; }
__device__ __forceinline__ bd_u32 bd_u16at(const bd_u32 *__restrict__ W, bd_u32 b) { return (bd_w(W, b >> 2) >> ((b & 2u) * 8u)) & 0xffffu; }
__device__ __forceinline__ float bd_sbyte(bd_u32 w, int k) { return (float)(int)(signed char)(w >> (8 * k)); }
__device__ __forceinline__ int bd_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610 && !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
    return __dp4a(a, b, c);
#else
    return c + (int)(signed char)a * (int)(signed char)b + (int)(signed char)(a >> 8) * (int)(signed char)(b >> 8) +
           (int)(signed char)(a >> 16) * (int)(signed char)(b >> 16) + (int)(signed char)(a >> 24) * (int)(signed char)(b >> 24);
#endif
}

/* 1 for the types with an integer step (bdi_step) */
__host__ __device__ constexpr bool bdi_has(int t) {
    return t == 8 || t == 10 || t == 11 || t == 12 || t == 13 || t == 14 || t == 16 || t == 17 || t == 18 || t == 19 || t == 20 ||
           t == 21 || t == 22 || t == 23 || t == 29 || t == 42;
}
/* the type's tables in shared memory, in words: its grid, ksigns, kvalues */
__host__ __device__ constexpr int bd_grid_words(int t) { return t == 22 ? 2048 : t == 17 ? 1024 : t == 21 || t == 16 ? 512 : t == 18 ? 256 : 0; }
__host__ __device__ constexpr int bd_grid_base(int t) {
    return t == 22 ? BDT_IQ2S : t == 21 ? BDT_IQ3S : t == 17 ? BDT_IQ2XS : t == 18 ? BDT_IQ3XXS : BDT_IQ2XXS;
}
__host__ __device__ constexpr int bd_ks_words(int t) { return t == 16 || t == 17 || t == 18 ? 128 : 0; }
__host__ __device__ constexpr int bd_kv_words(int t) { return t == 20 || t == 23 ? 16 : 0; }
__host__ __device__ constexpr int bd_tab_words(int t) { return (bd_grid_words(t) + bd_ks_words(t) + bd_kv_words(t) + 3) & ~3; }

struct BdTabs { const bd_u32 *grid, *ks; const int *kvt; const bd_u32 *tab; };
/* type T's tables into shared memory at sm (bd_tab_words(T) words), by the block's threads */
template <int T>
__device__ __forceinline__ BdTabs bd_tables_fill(bd_u32 *sm, const bd_u32 *__restrict__ tab) {
    constexpr int GN = bd_grid_words(T), KN = bd_ks_words(T), VN = bd_kv_words(T);
    for (int i = threadIdx.x; i < GN; i += blockDim.x) sm[i] = __ldg(tab + bd_grid_base(T) + i);
    for (int i = threadIdx.x; i < KN; i += blockDim.x) sm[GN + i] = __ldg(tab + BDT_KSIGNS + i);
    for (int i = threadIdx.x; i < VN; i += blockDim.x) sm[GN + KN + i] = __ldg(tab + BDT_IQ4NL + i);
    BdTabs tb = {sm, sm + GN, reinterpret_cast<const int *>(sm + GN + KN), tab};
    return tb;
}

/* The rows of x in the integer form, in shared memory: row r's codes at q + r * qr (uint4s,
 * one of padding after every eight: a lane's step reads two, the next lane's two further
 * on), its scale per 32 at dx + r * nb, its code sums per 16 at xs + r * 2 nb. */
struct BdX { const uint4 *q; const float *dx; const int *xs; int qr, nb; };
__host__ __device__ constexpr int bd_x4(int k) { return k + (k >> 3); }
__host__ __device__ constexpr int bd_xq_row(int I) { return I / 16 + I / 128 + 1; }
/* shared bytes of nr rows of I inputs in that form */
__host__ __device__ constexpr size_t bd_x_bytes(int I, int nr) { return (size_t)nr * ((size_t)bd_xq_row(I) * 16 + (size_t)(I / 32) * 4 + (size_t)(I / 16) * 4); }
__device__ __forceinline__ void bdx_u4(const BdX &X, int r, int k, int *w) {
    uint4 v = X.q[r * X.qr + bd_x4(k)];
    w[0] = (int)v.x; w[1] = (int)v.y; w[2] = (int)v.z; w[3] = (int)v.w;
}
/* rows s0 .. s0 + nrow - 1 of x (I inputs each) as gq_quant_act_ref quantizes them, a block
 * of 32 a thread, into the shared memory at q (BdX's layout); the block syncs after */
__device__ __forceinline__ void bdx_quant(const float *__restrict__ x, int nrow, int I, uint4 *q, float *dx, int *xs) {
    int nb = I / 32, qr = bd_xq_row(I);
    for (int k = threadIdx.x; k < nrow * nb; k += blockDim.x) {
        int r = k / nb, b = k - r * nb;
        const float4 *src = reinterpret_cast<const float4 *>(x + (size_t)r * I + 32 * b);
        float v[32], amax = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float4 f = src[j];
            v[4 * j] = f.x; v[4 * j + 1] = f.y; v[4 * j + 2] = f.z; v[4 * j + 3] = f.w;
        }
        #pragma unroll
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(v[j]));
        float d = __fdiv_rn(amax, 127.f), id = amax > 0.f ? __fdiv_rn(127.f, amax) : 0.f;
        int s0 = 0, s1 = 0;
        bd_u32 w[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            bd_u32 pk = 0;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                int qv = __float2int_rn(__fmul_rn(v[4 * j + c], id));
                qv = qv > 127 ? 127 : qv < -127 ? -127 : qv;
                if (j < 4) s0 += qv; else s1 += qv;
                pk |= ((bd_u32)qv & 255u) << (8 * c);
            }
            w[j] = pk;
        }
        q[r * qr + bd_x4(2 * b)] = make_uint4(w[0], w[1], w[2], w[3]);
        q[r * qr + bd_x4(2 * b + 1)] = make_uint4(w[4], w[5], w[6], w[7]);
        dx[r * nb + b] = d;
        xs[r * 2 * nb + 2 * b] = s0; xs[r * 2 * nb + 2 * b + 1] = s1;
    }
}

/* 0xFF in each byte whose bit (of the low four) is set: those bytes are negated */
__device__ __forceinline__ bd_u32 bdi_smask(bd_u32 b4) { return ((b4 * 0x00204081u) & 0x01010101u) * 0xFFu; }
__device__ __forceinline__ bd_u32 bdi_neg(bd_u32 g, bd_u32 b4) { bd_u32 m = bdi_smask(b4 & 15u); return (g ^ m) + (m & 0x01010101u); }
/* four kvalues (IQ4) of the nibbles at bit sh of each byte of w, as packed int8 */
__device__ __forceinline__ int bdi_kv4(const int *kvt, bd_u32 w, int sh) {
    return (int)(((bd_u32)kvt[(w >> sh) & 15u] & 255u) | (((bd_u32)kvt[(w >> (sh + 8)) & 15u] & 255u) << 8) |
                 (((bd_u32)kvt[(w >> (sh + 16)) & 15u] & 255u) << 16) | (((bd_u32)kvt[(w >> (sh + 24)) & 15u] & 255u) << 24));
}
/* Q4_K / Q5_K: byte j of the block's 12 scale bytes (words 1..3), the 6-bit scale and min of group g */
__device__ __forceinline__ bd_u32 bdi_kb(uint4 hd, bd_u32 j) { bd_u32 w = j < 4u ? hd.y : j < 8u ? hd.z : hd.w; return (w >> (8u * (j & 3u))) & 255u; }
__device__ __forceinline__ uint2 bdi_k4sm(uint4 hd, bd_u32 g) {
    if (g < 4u) return make_uint2(bdi_kb(hd, g) & 63u, bdi_kb(hd, g + 4u) & 63u);
    return make_uint2((bdi_kb(hd, g + 4u) & 15u) | ((bdi_kb(hd, g - 4u) >> 6) << 4), (bdi_kb(hd, g + 4u) >> 4) | ((bdi_kb(hd, g) >> 6) << 4));
}
/* row r's codes: words 8q .. 8q + 7 (a step's 32 inputs) */
__device__ __forceinline__ void bdx_step8(const BdX &X, int r, bd_u32 q, int *xw) { bdx_u4(X, r, 2 * q, xw); bdx_u4(X, r, 2 * q + 1, xw + 4); }

/* Step q (weights 32q .. 32q + 31 of the row W, ggml type T) for the rows r < nrow of X,
 * each adding its sum to acc[r]: bdstep_i8.glsl's bsi_step */
template <int T, int NR>
__device__ __forceinline__ void bdi_step(const bd_u32 *__restrict__ W, bd_u32 q, const BdX &X, int nrow, const BdTabs &tb, float *acc) {
    if (T == 12 || T == 13) {   /* chunk c (64 weights), half h: weights 64c + 16h + 0..15 (low nibbles) and 64c + 32 + 16h + 0..15 (high) */
        bd_u32 b = q >> 3, c = (q >> 1) & 3u, h = q & 1u;
        bd_u32 base = b * (T == 12 ? 144u : 176u);
        uint4 hd = bd_w4(W, base >> 4);
        float dmx = bd_h2f(hd.x), dmy = bd_h2f(hd.x >> 16);
        uint2 lo = bdi_k4sm(hd, 2u * c), hi = bdi_k4sm(hd, 2u * c + 1u);
        uint4 qs = bd_w4(W, (base + (T == 12 ? 16u : 48u) + 32u * c + 16u * h) >> 4);
        uint4 qh = T == 13 ? bd_w4(W, (base + 16u + 16u * h) >> 4) : make_uint4(0u, 0u, 0u, 0u);
        bd_u32 e = b * 64u + c * 16u + h * 4u;   /* the low run's first word of codes */
        bd_u32 qa[4] = {qs.x, qs.y, qs.z, qs.w}, qb[4] = {qh.x, qh.y, qh.z, qh.w};
        int wl[4], wh[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            bd_u32 l = qa[k] & 0x0F0F0F0Fu, hh = (qa[k] >> 4) & 0x0F0F0F0Fu;
            if (T == 13) { bd_u32 hb = qb[k] >> (2u * c); l |= (hb & 0x01010101u) << 4; hh |= ((hb >> 1) & 0x01010101u) << 4; }
            wl[k] = (int)l; wh[k] = (int)hh;
        }
        bd_u32 gl = e >> 2, gh = gl + 2u;   /* their groups of 16 (and blocks of 32: g / 2) */
        float sl0 = dmx * (float)lo.x, ml0 = dmy * (float)lo.y, sh0 = dmx * (float)hi.x, mh0 = dmy * (float)hi.y;
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            int xl[4], xh[4];
            bdx_u4(X, r, (int)(e >> 2), xl); bdx_u4(X, r, (int)(e >> 2) + 2, xh);
            int sl = 0, sh = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) { sl = bd_dp4a(wl[k], xl[k], sl); sh = bd_dp4a(wh[k], xh[k], sh); }
            const float *dx = X.dx + r * X.nb; const int *xs = X.xs + r * 2 * X.nb;
            acc[r] += dx[gl >> 1] * (sl0 * (float)sl - ml0 * (float)xs[gl]) + dx[gh >> 1] * (sh0 * (float)sh - mh0 * (float)xs[gh]);
        }
        return;
    }
    int w[8];
    float dq = 0.f;   /* the step's scale where one covers it */
    if (T == 14) {   /* Q6_K: (6-bit code) - 32, the codes' dot less 32 times the sum of x */
        bd_u32 b = q >> 3, j = q & 7u, n = j >> 2, m = j & 3u;
        bd_u32 base = b * 210u, qlo = base + 64u * n + 32u * (m & 1u), qhi = base + 128u + 32u * n;
        bd_u32 s2 = bd_ldu(W, base + 192u + 8u * n + 2u * m);
        float d = bd_f16at(W, base + 208u);
        bd_u32 sh = (m >> 1) * 4u, hs = 2u * m;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            bd_u32 lw = bd_ldu(W, qlo + 4u * k), hw = bd_ldu(W, qhi + 4u * k);
            w[k] = (int)(((lw >> sh) & 0x0F0F0F0Fu) | (((hw >> hs) & 0x03030303u) << 4));
        }
        float c0 = d * bd_sbyte(s2, 0), c1 = d * bd_sbyte(s2, 1);
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            int xw[8]; bdx_step8(X, r, q, xw);
            int s0 = 0, s1 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) { s0 = bd_dp4a(w[k], xw[k], s0); s1 = bd_dp4a(w[k + 4], xw[k + 4], s1); }
            const int *xs = X.xs + r * 2 * X.nb;
            s0 -= 32 * xs[2u * q]; s1 -= 32 * xs[2u * q + 1u];
            acc[r] += X.dx[r * X.nb + q] * (c0 * (float)s0 + c1 * (float)s1);
        }
        return;
    }
    if (T == 10) {   /* Q2_K: 2-bit codes, a scale and a min per 16 */
        bd_u32 b = q >> 3, s = q & 7u, base = b * 84u, sh = 2u * (s & 3u), qa = base + 16u + 32u * (s >> 2);
        bd_u32 s0 = bd_u8at(W, base + 2u * s), s1 = bd_u8at(W, base + 2u * s + 1u);
        bd_u32 dmw = bd_ldu(W, base + 80u);
        float dmx = bd_h2f(dmw), dmy = bd_h2f(dmw >> 16);
        #pragma unroll
        for (int k = 0; k < 8; k++) w[k] = (int)((bd_ldu(W, qa + 4u * k) >> sh) & 0x03030303u);
        float a0s = dmx * (float)(s0 & 15u), a0m = dmy * (float)(s0 >> 4), a1s = dmx * (float)(s1 & 15u), a1m = dmy * (float)(s1 >> 4);
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            int xw[8]; bdx_step8(X, r, q, xw);
            int a0 = 0, a1 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) { a0 = bd_dp4a(w[k], xw[k], a0); a1 = bd_dp4a(w[k + 4], xw[k + 4], a1); }
            const int *xs = X.xs + r * 2 * X.nb;
            acc[r] += X.dx[r * X.nb + q] * (a0s * (float)a0 - a0m * (float)xs[2u * q] + a1s * (float)a1 - a1m * (float)xs[2u * q + 1u]);
        }
        return;
    }
    if (T == 11) {   /* Q3_K: codes 0..7 (the 2 low bits, hmask's bit as 4), less 4 times the sum of x; a scale less 32 per 16 */
        bd_u32 b = q >> 3, s = q & 7u, base = b * 110u, sh = 2u * (s & 3u), qa = base + 32u + 32u * (s >> 2);
        float d = bd_f16at(W, base + 108u), sw[2];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            bd_u32 k = 2u * s + (bd_u32)h;
            bd_u32 l = k < 8u ? (bd_u8at(W, base + 96u + k) & 15u) : (bd_u8at(W, base + 88u + k) >> 4);
            bd_u32 hh = (bd_u8at(W, base + 104u + (k & 3u)) >> (2u * (k >> 2))) & 3u;
            sw[h] = d * (float)((int)(l | (hh << 4)) - 32);
        }
        #pragma unroll
        for (int k = 0; k < 8; k++)
            w[k] = (int)(((bd_ldu(W, qa + 4u * k) >> sh) & 0x03030303u) | (((bd_ldu(W, base + 4u * k) >> s) & 0x01010101u) << 2));
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            int xw[8]; bdx_step8(X, r, q, xw);
            int a0 = 0, a1 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) { a0 = bd_dp4a(w[k], xw[k], a0); a1 = bd_dp4a(w[k + 4], xw[k + 4], a1); }
            const int *xs = X.xs + r * 2 * X.nb;
            a0 -= 4 * xs[2u * q]; a1 -= 4 * xs[2u * q + 1u];
            acc[r] += X.dx[r * X.nb + q] * (sw[0] * (float)a0 + sw[1] * (float)a1);
        }
        return;
    }
    if (T == 22 || T == 17) {   /* IQ2_S, IQ2_XS: four chunks of 8 weights, a grid row each; a scale per 16 */
        bd_u32 b = q >> 3, ib = q & 7u;
        float d, sc0, sc1;
        if (T == 22) {
            bd_u32 base = b * 82u;
            bd_u32 ix = bd_ldu(W, base + 2u + 4u * ib), sg = bd_ldu(W, base + 34u + 4u * ib);
            bd_u32 hb = bd_u8at(W, base + 66u + ib), scb = bd_u8at(W, base + 74u + ib);
            d = bd_f16at(W, base);
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                bd_u32 idx = ((ix >> (8u * l)) & 255u) | (((hb >> (2u * l)) & 3u) << 8), s8 = sg >> (8u * l);
                w[2 * l] = (int)bdi_neg(tb.grid[2u * idx], s8); w[2 * l + 1] = (int)bdi_neg(tb.grid[2u * idx + 1u], s8 >> 4);
            }
            sc0 = (float)(scb & 15u); sc1 = (float)(scb >> 4);
        } else {
            bd_u32 base = b * 74u;
            bd_u32 i0 = bd_ldu(W, base + 2u + 8u * ib), i1 = bd_ldu(W, base + 6u + 8u * ib), scb = bd_u8at(W, base + 66u + ib);
            d = bd_f16at(W, base);
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                bd_u32 e = ((l < 2 ? i0 : i1) >> (16u * (l & 1))) & 0xFFFFu, idx = e & 511u, s8 = tb.ks[e >> 9];
                w[2 * l] = (int)bdi_neg(tb.grid[2u * idx], s8); w[2 * l + 1] = (int)bdi_neg(tb.grid[2u * idx + 1u], s8 >> 4);
            }
            sc0 = (float)(scb & 15u); sc1 = (float)(scb >> 4);
        }
        float c0 = d * (0.5f + sc0) * 0.25f, c1 = d * (0.5f + sc1) * 0.25f;
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            int xw[8]; bdx_step8(X, r, q, xw);
            int s0 = 0, s1 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) { s0 = bd_dp4a(w[k], xw[k], s0); s1 = bd_dp4a(w[k + 4], xw[k + 4], s1); }
            acc[r] += X.dx[r * X.nb + q] * (c0 * (float)s0 + c1 * (float)s1);
        }
        return;
    }
    if (T == 19 || T == 29) {   /* IQ1_S / IQ1_M: 8 * (the grid rows' dot) + the delta's sign times the sum of x */
        bd_u32 b = q >> 3, ib = q & 7u, base = b * (T == 19 ? 50u : 56u);
        bd_u32 qs, qh; float sw0, sw1;
        if (T == 19) {
            qs = bd_ldu(W, base + 2u + 4u * ib); qh = bd_u16at(W, base + 34u + 2u * ib);
            sw0 = sw1 = bd_f16at(W, base) * (float)(2u * ((qh >> 12) & 7u) + 1u) * 0.125f;
        } else {
            qs = bd_ldu(W, base + 4u * ib); qh = bd_u16at(W, base + 32u + 2u * ib);
            bd_u32 s01 = bd_ldu(W, base + 48u), s23 = bd_ldu(W, base + 52u);
            float d = bd_h2f(((s01 >> 12) & 0xFu) | ((s01 >> 24) & 0xF0u) | ((s23 >> 4) & 0xF00u) | ((s23 >> 16) & 0xF000u));
            bd_u32 scw = (ib >> 1) == 0u ? s01 : (ib >> 1) == 1u ? s01 >> 16 : (ib >> 1) == 2u ? s23 : s23 >> 16;
            sw0 = d * (float)(2u * ((scw >> (3u * ((2u * ib) & 3u))) & 7u) + 1u) * 0.125f;
            sw1 = d * (float)(2u * ((scw >> (3u * ((2u * ib + 1u) & 3u))) & 7u) + 1u) * 0.125f;
        }
        int sg[4];
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            bd_u32 idx;
            if (T == 19) { idx = ((qs >> (8u * l)) & 255u) | (((qh >> (3u * l)) & 7u) << 8); sg[l] = (qh & 0x8000u) != 0u ? -1 : 1; }
            else { bd_u32 h = (qh >> (4u * l)) & 15u; idx = ((qs >> (8u * l)) & 255u) | ((h & 7u) << 8); sg[l] = (h & 8u) != 0u ? -1 : 1; }
            w[2 * l] = (int)__ldg(tb.tab + BDT_IQ1S + 2u * idx); w[2 * l + 1] = (int)__ldg(tb.tab + BDT_IQ1S + 2u * idx + 1u);
        }
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            int xw[8]; bdx_step8(X, r, q, xw);
            int a0 = 0, a1 = 0;
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int g = bd_dp4a(w[2 * l + 1], xw[2 * l + 1], bd_dp4a(w[2 * l], xw[2 * l], 0));
                int xs8 = bd_dp4a(0x01010101, xw[2 * l + 1], bd_dp4a(0x01010101, xw[2 * l], 0));   /* the 8 codes' sum */
                int v = 8 * g + sg[l] * xs8;
                if (l < 2) a0 += v; else a1 += v;
            }
            acc[r] += X.dx[r * X.nb + q] * (sw0 * (float)a0 + sw1 * (float)a1);
        }
        return;
    }
    /* the types with one scale a step: the codes' dot (less a sum of x for Q2_0) times it */
    int sub = 0;   /* Q2_0: the codes' offset of 1 */
    if (T == 8) {   /* Q8_0: the step is the block */
        bd_u32 base = q * 34u;
        #pragma unroll
        for (int k = 0; k < 8; k++) w[k] = (int)bd_ldu(W, base + 2u + 4u * k);
        dq = bd_f16at(W, base);
    } else if (T == 20 || T == 23) {   /* IQ4_NL (the step is the block) / IQ4_XS (8 steps a block): kvalues of the nibbles */
        bd_u32 qa;
        if (T == 20) { bd_u32 base = q * 18u; dq = bd_f16at(W, base); qa = base + 2u; }
        else {
            bd_u32 b = q >> 3, ib = q & 7u, base = b * 136u;
            bd_u32 w0 = bd_w(W, base >> 2), w2 = bd_w(W, (base >> 2) + 1u);
            bd_u32 ls = ((w2 >> (8u * (ib >> 1) + 4u * (ib & 1u))) & 15u) | (((w0 >> (16u + 2u * ib)) & 3u) << 4);
            dq = bd_h2f(w0) * (float)((int)ls - 32); qa = base + 8u + 16u * ib;
        }
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            bd_u32 wd = T == 20 ? bd_ldu(W, qa + 4u * k) : bd_w(W, (qa >> 2) + (bd_u32)k);
            w[k] = bdi_kv4(tb.kvt, wd, 0); w[k + 4] = bdi_kv4(tb.kvt, wd, 4);
        }
    } else if (T == 21) {   /* IQ3_S: 8 chunks of 4 weights, a grid row each, its signs */
        bd_u32 b = q >> 3, ib = q & 7u, base = b * 110u;
        bd_u32 i0 = bd_ldu(W, base + 2u + 8u * ib), i1 = bd_ldu(W, base + 6u + 8u * ib);
        bd_u32 hb = bd_u8at(W, base + 66u + ib), sg = bd_ldu(W, base + 74u + 4u * ib);
        bd_u32 sc = (bd_u8at(W, base + 106u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u;
        #pragma unroll
        for (int l = 0; l < 8; l++) {
            bd_u32 idx = (((l < 4 ? i0 : i1) >> (8u * (l & 3))) & 255u) | (((hb >> l) & 1u) << 8);
            w[l] = (int)bdi_neg(tb.grid[idx], sg >> (4u * l));
        }
        dq = bd_f16at(W, base) * (float)(1u + 2u * sc);
    } else if (T == 18) {   /* IQ3_XXS: 8 grid rows of 4, their signs (ksigns) and a 4-bit scale */
        bd_u32 b = q >> 3, ib = q & 7u, base = b * 98u;
        bd_u32 g0 = bd_ldu(W, base + 2u + 8u * ib), g1 = bd_ldu(W, base + 6u + 8u * ib), a = bd_ldu(W, base + 66u + 4u * ib);
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            bd_u32 gw = l < 2 ? g0 : g1, sh = 16u * (l & 1), s8 = tb.ks[(a >> (7u * l)) & 127u];
            w[2 * l] = (int)bdi_neg(tb.grid[(gw >> sh) & 255u], s8); w[2 * l + 1] = (int)bdi_neg(tb.grid[(gw >> (sh + 8u)) & 255u], s8 >> 4);
        }
        dq = bd_f16at(W, base) * (0.5f + (float)(a >> 28)) * 0.5f;
    } else if (T == 16) {   /* IQ2_XXS: 4 grid indices, their signs (ksigns) and a 4-bit scale */
        bd_u32 b = q >> 3, ib = q & 7u, base = b * 66u;
        bd_u32 i0 = bd_ldu(W, base + 2u + 8u * ib), i1 = bd_ldu(W, base + 6u + 8u * ib);
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            bd_u32 idx = (i0 >> (8u * l)) & 255u, s8 = tb.ks[(i1 >> (7u * l)) & 127u];
            w[2 * l] = (int)bdi_neg(tb.grid[2u * idx], s8); w[2 * l + 1] = (int)bdi_neg(tb.grid[2u * idx + 1u], s8 >> 4);
        }
        dq = bd_f16at(W, base) * (0.5f + (float)(i1 >> 28)) * 0.25f;
    } else if (T == 42) {   /* Q2_0: codes 0..3, w = code - 1: the codes' dot less the sum of x */
        bd_u32 b = q >> 1, h = q & 1u, base = b * 18u;
        bd_u32 c[2] = {bd_ldu(W, base + 2u + 8u * h), bd_ldu(W, base + 6u + 8u * h)};
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                bd_u32 by = c[k] >> (8u * j);
                w[4 * k + j] = (int)((by & 3u) | ((by << 6) & 0x300u) | ((by << 12) & 0x30000u) | ((by << 18) & 0x3000000u));
            }
        }
        dq = bd_f16at(W, base); sub = 1;
    }
    #pragma unroll
    for (int r = 0; r < NR; r++) if (r < nrow) {
        int xw[8]; bdx_step8(X, r, q, xw);
        int s = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) s = bd_dp4a(w[k], xw[k], s);
        if (sub) { const int *xs = X.xs + r * 2 * X.nb; s -= xs[2u * q] + xs[2u * q + 1u]; }
        acc[r] += X.dx[r * X.nb + q] * dq * (float)s;
    }
}

/* The general decoder (blockdec.glsl's unit): unit u (weights 32u .. 32u + 31 of the row W,
 * fmt 100 + t, any block type with blocks of 32 or more) as int weights wv with a scale and
 * a min for each run of 16: weight = s * wv - m. The float form, for the types without an
 * integer step. */
__device__ __forceinline__ void bdf_decode(int fmt, const bd_u32 *__restrict__ W, bd_u32 u, const bd_u32 *__restrict__ tab, int *wv,
                                           float &sA, float &mA, float &sB, float &mB) {
    sA = mA = sB = mB = 0.f;
    int t = fmt - 100;
    bd_u32 xb = u * 32u, B = __ldg(tab + BDT_BLK + t), Tb = __ldg(tab + BDT_BYTES + t);
    bd_u32 e0 = xb % B, off = (xb / B) * Tb, g0 = e0 / 16u;
    #define BD_U8(o) bd_u8at(W, (o))
    auto ld32 = [&](bd_u32 o) -> bd_u32 {   /* four bytes from any offset */
        bd_u32 a = o >> 2, sh = (o & 3u) << 3;
        bd_u32 lo = bd_w(W, a);
        return sh == 0u ? lo : (lo >> sh) | (bd_w(W, a + 1u) << (32u - sh));
    };
    auto f16 = [&](bd_u32 o) -> float { return bd_h2f(ld32(o)); };
    auto g2 = [&](bd_u32 base, bd_u32 i, bd_u32 &lo, bd_u32 &hi) { lo = __ldg(tab + base + 2u * i); hi = __ldg(tab + base + 2u * i + 1u); };
    auto gbyte = [](bd_u32 lo, bd_u32 hi, bd_u32 j) -> bd_u32 { return ((j < 4u ? lo : hi) >> ((j & 3u) * 8u)) & 0xffu; };
    if (t == 2 || t == 3 || t == 6 || t == 7) {   /* Q4_0, Q4_1, Q5_0, Q5_1 */
        bool mn = t == 3 || t == 7;
        float d = f16(off), m = mn ? f16(off + 2u) : 0.f;
        bd_u32 qh = t == 6 ? ld32(off + 2u) : t == 7 ? ld32(off + 4u) : 0u;
        bd_u32 qs = off + (t == 2 ? 2u : t == 3 ? 4u : t == 6 ? 6u : 8u);
        int z = t == 2 ? 8 : t == 6 ? 16 : 0;
        for (bd_u32 j = 0u; j < 16u; j++) {
            bd_u32 by = BD_U8(qs + j);
            int lo = (int)(by & 15u), hi = (int)(by >> 4);
            if (t >= 6) { lo |= (int)((qh >> j) & 1u) << 4; hi |= (int)((qh >> (j + 16u)) & 1u) << 4; }
            wv[j] = lo - z; wv[j + 16u] = hi - z;
        }
        sA = sB = d; mA = mB = -m;
    } else if (t == 8) {   /* Q8_0 */
        float d = f16(off);
        for (bd_u32 j = 0u; j < 32u; j++) wv[j] = (int)(signed char)BD_U8(off + 2u + j);
        sA = sB = d;
    } else if (t == 20 || t == 39) {   /* IQ4_NL, MXFP4 */
        float d;
        if (t == 20) d = f16(off);
        else { bd_u32 x8 = BD_U8(off); d = __uint_as_float(x8 < 2u ? (0x00200000u << x8) : ((x8 - 1u) << 23)); }
        bd_u32 qs = off + (t == 20 ? 2u : 1u), kv = t == 20 ? BDT_IQ4NL : BDT_MXFP4;
        for (bd_u32 j = 0u; j < 16u; j++) {
            bd_u32 by = BD_U8(qs + j);
            wv[j] = (int)__ldg(tab + kv + (by & 15u)); wv[j + 16u] = (int)__ldg(tab + kv + (by >> 4));
        }
        sA = sB = d;
    } else if (t == 42) {   /* Q2_0: element 4k+r in bits 2r of byte k */
        float d = f16(off);
        for (bd_u32 j = 0u; j < 32u; j++) { bd_u32 e = e0 + j; wv[j] = (int)((BD_U8(off + 2u + (e >> 2)) >> ((e & 3u) * 2u)) & 3u) - 1; }
        sA = sB = d;
    } else if (t == 41) {   /* Q1_0 */
        float d = f16(off);
        for (bd_u32 j = 0u; j < 32u; j++) { bd_u32 e = e0 + j; wv[j] = ((BD_U8(off + 2u + (e >> 3)) >> (e & 7u)) & 1u) != 0u ? 1 : -1; }
        sA = sB = d;
    } else if (t == 40) {   /* NVFP4: 4 sub-blocks of 16 */
        bd_u32 qs = off + 4u;
        for (bd_u32 hh = 0u; hh < 2u; hh++) {
            bd_u32 sb = g0 + hh, x8 = BD_U8(off + sb);
            float s;
            if (x8 == 0u || x8 == 0x7fu) s = 0.f;
            else { bd_u32 e = (x8 >> 3) & 15u, m = x8 & 7u; s = e == 0u ? ldexpf((float)m, -10) : ldexpf((float)(8u + m), (int)e - 11); }
            if (hh == 0u) sA = s; else sB = s;
            for (bd_u32 j = 0u; j < 8u; j++) {
                bd_u32 by = BD_U8(qs + 8u * sb + j);
                wv[16u * hh + j] = (int)__ldg(tab + BDT_MXFP4 + (by & 15u)); wv[16u * hh + 8u + j] = (int)__ldg(tab + BDT_MXFP4 + (by >> 4));
            }
        }
    } else if (t == 10) {   /* Q2_K */
        float d = f16(off + 80u), dmin = f16(off + 82u);
        bd_u32 c = e0 >> 7, sh = (e0 >> 5) & 3u, qq = off + 16u + c * 32u;
        for (bd_u32 j = 0u; j < 32u; j++) wv[j] = (int)((BD_U8(qq + j) >> (2u * sh)) & 3u);
        bd_u32 a = BD_U8(off + g0), b = BD_U8(off + g0 + 1u);
        sA = d * (float)(a & 15u); mA = dmin * (float)(a >> 4); sB = d * (float)(b & 15u); mB = dmin * (float)(b >> 4);
    } else if (t == 11) {   /* Q3_K */
        float d = f16(off + 108u);
        for (bd_u32 hh = 0u; hh < 2u; hh++) {
            bd_u32 k = g0 + hh;
            bd_u32 l = k < 8u ? (BD_U8(off + 96u + k) & 15u) : (BD_U8(off + 96u + k - 8u) >> 4);
            bd_u32 h = (BD_U8(off + 96u + 8u + (k & 3u)) >> (2u * (k >> 2))) & 3u;
            float s = d * (float)((int)(l | (h << 4)) - 32);
            if (hh == 0u) sA = s; else sB = s;
        }
        for (bd_u32 j = 0u; j < 32u; j++) {
            bd_u32 i = e0 + j, c = i >> 7, sh = (i >> 5) & 3u, jj = i & 31u;
            int ql = (int)((BD_U8(off + 32u + c * 32u + jj) >> (2u * sh)) & 3u);
            int hb = (int)((BD_U8(off + (i & 31u)) >> (i >> 5)) & 1u);
            wv[j] = ql - ((hb ^ 1) << 2);
        }
    } else if (t == 12 || t == 13) {   /* Q4_K, Q5_K */
        float d = f16(off), dmin = f16(off + 2u);
        bd_u32 g = e0 >> 5, sc = off + 4u, qs = off + (t == 12 ? 16u : 48u), s, m;
        if (g < 4u) { s = BD_U8(sc + g) & 63u; m = BD_U8(sc + g + 4u) & 63u; }
        else {
            s = (BD_U8(sc + g + 4u) & 15u) | ((BD_U8(sc + g - 4u) >> 2) & 48u);
            m = (BD_U8(sc + g + 4u) >> 4) | ((BD_U8(sc + g) >> 2) & 48u);
        }
        sA = sB = d * (float)s; mA = mB = dmin * (float)m;
        bd_u32 qq = qs + (g >> 1) * 32u, sh = (g & 1u) * 4u;
        for (bd_u32 j = 0u; j < 32u; j++) {
            int v = (int)((BD_U8(qq + j) >> sh) & 15u);
            if (t == 13) v |= (int)((BD_U8(off + 16u + j) >> g) & 1u) << 4;
            wv[j] = v;
        }
    } else if (t == 14) {   /* Q6_K */
        float d = f16(off + 208u);
        bd_u32 c = e0 >> 7, r0 = e0 & 127u, nib = r0 >> 6, sh = r0 >> 5;
        bd_u32 ql = off + c * 64u + (r0 & 63u), qh = off + 128u + c * 32u;
        for (bd_u32 j = 0u; j < 32u; j++) {
            int lo = (int)((BD_U8(ql + j) >> (4u * nib)) & 15u), hi = (int)((BD_U8(qh + j) >> (2u * sh)) & 3u);
            wv[j] = (lo | (hi << 4)) - 32;
        }
        sA = d * (float)(signed char)BD_U8(off + 192u + g0); sB = d * (float)(signed char)BD_U8(off + 192u + g0 + 1u);
    } else if (t == 16) {   /* IQ2_XXS */
        float d = f16(off);
        bd_u32 ib = e0 >> 5, qs = off + 2u + 8u * ib, a0 = ld32(qs), a1 = ld32(qs + 4u);
        sA = sB = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
        for (bd_u32 l = 0u; l < 4u; l++) {
            bd_u32 lo, hi; g2(BDT_IQ2XXS, (a0 >> (8u * l)) & 255u, lo, hi);
            bd_u32 sg = __ldg(tab + BDT_KSIGNS + ((a1 >> (7u * l)) & 127u));
            for (bd_u32 j = 0u; j < 8u; j++) { int v = (int)gbyte(lo, hi, j); wv[l * 8u + j] = ((sg >> j) & 1u) != 0u ? -v : v; }
        }
    } else if (t == 17 || t == 22) {   /* IQ2_XS, IQ2_S */
        float d = f16(off);
        bd_u32 sc = off + (t == 17 ? 66u : 74u);
        for (bd_u32 hh = 0u; hh < 2u; hh++) {
            bd_u32 si = g0 + hh;
            float s = d * (0.5f + (float)((BD_U8(sc + (si >> 1)) >> (4u * (si & 1u))) & 15u)) * 0.25f;
            if (hh == 0u) sA = s; else sB = s;
        }
        for (bd_u32 l = 0u; l < 4u; l++) {
            bd_u32 k = (e0 >> 3) + l, lo, hi, sg;
            if (t == 17) { bd_u32 qv = ld32(off + 2u + 2u * k) & 0xffffu; g2(BDT_IQ2XS, qv & 511u, lo, hi); sg = __ldg(tab + BDT_KSIGNS + (qv >> 9)); }
            else {
                bd_u32 idx = BD_U8(off + 2u + k) | (((BD_U8(off + 66u + (k >> 2)) >> (2u * (k & 3u))) & 3u) << 8);
                g2(BDT_IQ2S, idx, lo, hi); sg = BD_U8(off + 34u + k);
            }
            for (bd_u32 j = 0u; j < 8u; j++) { int v = (int)gbyte(lo, hi, j); wv[l * 8u + j] = ((sg >> j) & 1u) != 0u ? -v : v; }
        }
    } else if (t == 18) {   /* IQ3_XXS */
        float d = f16(off);
        bd_u32 ib = e0 >> 5, qs = off + 2u, a = ld32(off + 66u + 4u * ib);
        sA = sB = d * (0.5f + (float)(a >> 28)) * 0.5f;
        for (bd_u32 l = 0u; l < 4u; l++) {
            bd_u32 sg = __ldg(tab + BDT_KSIGNS + ((a >> (7u * l)) & 127u));
            bd_u32 gg1 = __ldg(tab + BDT_IQ3XXS + BD_U8(qs + 8u * ib + 2u * l)), gg2 = __ldg(tab + BDT_IQ3XXS + BD_U8(qs + 8u * ib + 2u * l + 1u));
            for (bd_u32 j = 0u; j < 4u; j++) {
                int v1 = (int)((gg1 >> (8u * j)) & 255u), v2 = (int)((gg2 >> (8u * j)) & 255u);
                wv[l * 8u + j] = ((sg >> j) & 1u) != 0u ? -v1 : v1;
                wv[l * 8u + j + 4u] = ((sg >> (j + 4u)) & 1u) != 0u ? -v2 : v2;
            }
        }
    } else if (t == 21) {   /* IQ3_S */
        float d = f16(off);
        bd_u32 ib = e0 >> 5;
        sA = sB = d * (float)(1u + 2u * ((BD_U8(off + 106u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u));
        for (bd_u32 l = 0u; l < 8u; l++) {
            bd_u32 k = (e0 >> 2) + l;
            bd_u32 idx = BD_U8(off + 2u + k) | (((BD_U8(off + 66u + (k >> 3)) >> (k & 7u)) & 1u) << 8);
            bd_u32 gr = __ldg(tab + BDT_IQ3S + idx);
            for (bd_u32 j = 0u; j < 4u; j++) {
                bd_u32 e = k * 4u + j; int v = (int)((gr >> (8u * j)) & 255u);
                wv[l * 4u + j] = ((BD_U8(off + 74u + (e >> 3)) >> (e & 7u)) & 1u) != 0u ? -v : v;
            }
        }
    } else if (t == 19) {   /* IQ1_S */
        float d = f16(off);
        bd_u32 ib = e0 >> 5, qh = ld32(off + 34u + 2u * ib) & 0xffffu;
        float dl = d * (float)(2u * ((qh >> 12) & 7u) + 1u);
        int dsign = (qh & 0x8000u) != 0u ? -1 : 1;
        sA = sB = dl * 0.125f;
        for (bd_u32 l = 0u; l < 4u; l++) {
            bd_u32 lo, hi; g2(BDT_IQ1S, BD_U8(off + 2u + 4u * ib + l) | (((qh >> (3u * l)) & 7u) << 8), lo, hi);
            for (bd_u32 j = 0u; j < 8u; j++) wv[l * 8u + j] = 8 * (int)(signed char)gbyte(lo, hi, j) + dsign;
        }
    } else if (t == 29) {   /* IQ1_M */
        bd_u32 s0 = ld32(off + 48u) & 0xffffu, s1 = ld32(off + 50u) & 0xffffu, s2 = ld32(off + 52u) & 0xffffu, s3 = ld32(off + 54u) & 0xffffu;
        float d = bd_h2f((s0 >> 12) | ((s1 >> 8) & 0xf0u) | ((s2 >> 4) & 0xf00u) | (s3 & 0xf000u));
        for (bd_u32 hh = 0u; hh < 2u; hh++) {
            bd_u32 s16 = g0 + hh, qq = s16 >> 2, sw = qq == 0u ? s0 : qq == 1u ? s1 : qq == 2u ? s2 : s3;
            float s = d * (float)(2u * ((sw >> (3u * (s16 & 3u))) & 7u) + 1u) * 0.125f;
            if (hh == 0u) sA = s; else sB = s;
        }
        for (bd_u32 l = 0u; l < 4u; l++) {
            bd_u32 k = (e0 >> 3) + l;
            bd_u32 h = (BD_U8(off + 32u + (k >> 1)) >> (4u * (k & 1u))) & 15u;
            bd_u32 lo, hi; g2(BDT_IQ1S, BD_U8(off + k) | ((h & 7u) << 8), lo, hi);
            int dsign = (h & 8u) != 0u ? -1 : 1;
            for (bd_u32 j = 0u; j < 8u; j++) wv[l * 8u + j] = 8 * (int)(signed char)gbyte(lo, hi, j) + dsign;
        }
    } else if (t == 23) {   /* IQ4_XS */
        float d = f16(off);
        bd_u32 sh = ld32(off + 2u) & 0xffffu, ib = e0 >> 5;
        bd_u32 ls = ((BD_U8(off + 4u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4);
        sA = sB = d * (float)((int)ls - 32);
        bd_u32 qq = off + 8u + 16u * ib;
        for (bd_u32 j = 0u; j < 16u; j++) {
            bd_u32 by = BD_U8(qq + j);
            wv[j] = (int)__ldg(tab + BDT_IQ4NL + (by & 15u)); wv[j + 16u] = (int)__ldg(tab + BDT_IQ4NL + (by >> 4));
        }
    } else if (t == 35) {   /* TQ2_0 */
        float d = f16(off + 64u);
        for (bd_u32 j = 0u; j < 32u; j++) {
            bd_u32 i = e0 + j, c = i >> 7, s = (i >> 5) & 3u, jj = i & 31u;
            wv[j] = (int)((BD_U8(off + c * 32u + jj) >> (2u * s)) & 3u) - 1;
        }
        sA = sB = d;
    } else if (t == 34) {   /* TQ1_0: base-3 digits */
        float d = f16(off + 52u);
        const bd_u32 pw[5] = {1u, 3u, 9u, 27u, 81u};
        for (bd_u32 j = 0u; j < 32u; j++) {
            bd_u32 e = e0 + j, by, n;
            if (e < 160u) { n = e >> 5; by = BD_U8(off + (e & 31u)); }
            else if (e < 240u) { bd_u32 f = e - 160u; n = f >> 4; by = BD_U8(off + 32u + (f & 15u)); }
            else { bd_u32 f = e - 240u; n = f >> 2; by = BD_U8(off + 48u + (f & 3u)); }
            bd_u32 v = (by * pw[n]) & 255u;
            wv[j] = (int)((v * 3u) >> 8) - 1;
        }
        sA = sB = d;
    } else {
        for (int j = 0; j < 32; j++) wv[j] = 0;
    }
    #undef BD_U8
}
/* 16 consecutive weights e0 .. e0 + 15 (e0 a multiple of 16) of the row W, ggml type t, as
 * floats with their scales (weight = s * code - m): the f16 tiles' fill (backend_cuda.cu,
 * hcg_fetch_w<100>), word loads, only the 16. 0 for a type without one here (then
 * bdf_decode's general path). The same weights as bdf_decode's. */
__device__ __forceinline__ void bd_bytes16(const bd_u32 *__restrict__ W, bd_u32 off, bd_u32 *b) {   /* off even */
    b[0] = bd_ldu(W, off); b[1] = bd_ldu(W, off + 4u); b[2] = bd_ldu(W, off + 8u); b[3] = bd_ldu(W, off + 12u);
}
__device__ __forceinline__ bd_u32 bd_byte_of(const bd_u32 *b, int j) { return (b[j >> 2] >> (8 * (j & 3))) & 0xffu; }
/* bd_dec16's 16 weights for the types whose codes come to f16 exactly by their bits (Q8_0,
 * Q4_K, Q5_K, Q6_K), as eight half2: four codes of a word, each a byte, go to two half2 of
 * 1024 + code (0x64 above the byte: f16's integers from 1024 are exact), 1024 (and a code's
 * offset) less, then times the scale less the min in half2: about two instructions a weight
 * where bd_dec16 and the f16 rounding after it took six (Qwen3.6 UD-Q4_K_M, a 1K-token
 * prompt on an RTX PRO 4000: the experts' gate and up 133 ms, 24 with no decode at all). The
 * same codes; the scale and the min rounded to f16 before the product, not after it. */
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ bd_u32 bd_h2u(__half2 h) { return *reinterpret_cast<bd_u32 *>(&h); }
__device__ __forceinline__ __half2 bd_u2h(bd_u32 u) { return *reinterpret_cast<__half2 *>(&u); }
__device__ __forceinline__ void bd_codes_h(bd_u32 c4, __half2 off, __half2 s, __half2 m, bd_u32 *o) {
    /* four byte codes: bytes 0, 1 and 2, 3 with 0x64 above each, then (1024 + c - off) * s - m */
    __half2 a = __hsub2(bd_u2h(__byte_perm(c4, 0x64646464u, 0x5140u)), off);
    __half2 b = __hsub2(bd_u2h(__byte_perm(c4, 0x64646464u, 0x5342u)), off);
    o[0] = bd_h2u(__hfma2(a, s, m)); o[1] = bd_h2u(__hfma2(b, s, m));
}
__device__ __forceinline__ bool bd_dec16h(int t, const bd_u32 *__restrict__ W, bd_u32 e0, bd_u32 *out) {
    if (t == 12 || t == 13) {   /* Q4_K, Q5_K: groups of 32, a 6-bit scale and min each */
        bd_u32 b = e0 >> 8, e = e0 & 255u, c = e >> 6, pos = e & 63u, hi = pos >= 32u, g = 2u * c + hi, p = pos & 31u;
        bd_u32 base = b * (t == 12 ? 144u : 176u);
        uint4 hd = bd_w4(W, base >> 4);
        uint2 sm = bdi_k4sm(hd, g);
        float d = bd_h2f(hd.x) * (float)sm.x, m = bd_h2f(hd.x >> 16) * (float)sm.y;
        uint4 q = bd_w4(W, (base + (t == 12 ? 16u : 48u) + 32u * c + p) >> 4);
        uint4 h = t == 13 ? bd_w4(W, (base + 16u + p) >> 4) : make_uint4(0u, 0u, 0u, 0u);
        bd_u32 qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {h.x, h.y, h.z, h.w}, sh = hi ? 4u : 0u;
        __half2 off = __float2half2_rn(1024.f), s2 = __float2half2_rn(d), m2 = __float2half2_rn(-m);
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            bd_u32 v = (qw[k] >> sh) & 0x0F0F0F0Fu;
            if (t == 13) v |= ((hw[k] >> g) & 0x01010101u) << 4;
            bd_codes_h(v, off, s2, m2, out + 2 * k);
        }
        return true;
    }
    if (t == 14) {   /* Q6_K: (6-bit code - 32), a signed scale per 16 */
        bd_u32 b = e0 >> 8, e = e0 & 255u, n = e >> 7, r = e & 127u, base = b * 210u;
        bd_u32 ql[4], qh[4];
        bd_bytes16(W, base + 64u * n + (r & 63u), ql); bd_bytes16(W, base + 128u + 32u * n + (r & 31u), qh);
        bd_u32 ns = 4u * (r >> 6), hs = 2u * (r >> 5);
        float d = bd_f16at(W, base + 208u) * (float)(int)(signed char)bd_u8at(W, base + 192u + (e >> 4));
        __half2 off = __float2half2_rn(1056.f), s2 = __float2half2_rn(d), z2 = __float2half2_rn(0.f);
        #pragma unroll
        for (int k = 0; k < 4; k++) bd_codes_h(((ql[k] >> ns) & 0x0F0F0F0Fu) | (((qh[k] >> hs) & 0x03030303u) << 4), off, s2, z2, out + 2 * k);
        return true;
    }
    if (t == 8) {   /* Q8_0: blocks of 32, signed codes (their top bit flipped: code + 128) */
        bd_u32 b = e0 >> 5, base = b * 34u, w[4];
        bd_bytes16(W, base + 2u + 16u * ((e0 >> 4) & 1u), w);
        __half2 off = __float2half2_rn(1152.f), s2 = __float2half2_rn(bd_f16at(W, base)), z2 = __float2half2_rn(0.f);
        #pragma unroll
        for (int k = 0; k < 4; k++) bd_codes_h(w[k] ^ 0x80808080u, off, s2, z2, out + 2 * k);
        return true;
    }
    return false;
}
#endif
__device__ __forceinline__ bool bd_dec16(int t, const bd_u32 *__restrict__ W, bd_u32 e0, const bd_u32 *__restrict__ tab, float *out) {
    if (t == 12 || t == 13) {   /* Q4_K, Q5_K: groups of 32, a 6-bit scale and min each */
        bd_u32 b = e0 >> 8, e = e0 & 255u, c = e >> 6, pos = e & 63u, hi = pos >= 32u, g = 2u * c + hi, p = pos & 31u;
        bd_u32 base = b * (t == 12 ? 144u : 176u);
        uint4 hd = bd_w4(W, base >> 4);
        uint2 sm = bdi_k4sm(hd, g);
        float d = bd_h2f(hd.x) * (float)sm.x, m = bd_h2f(hd.x >> 16) * (float)sm.y;
        uint4 q = bd_w4(W, (base + (t == 12 ? 16u : 48u) + 32u * c + p) >> 4);
        uint4 h = t == 13 ? bd_w4(W, (base + 16u + p) >> 4) : make_uint4(0u, 0u, 0u, 0u);
        bd_u32 qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {h.x, h.y, h.z, h.w}, sh = hi ? 4u : 0u;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            bd_u32 v = (qw[j >> 2] >> (8 * (j & 3) + sh)) & 15u;
            if (t == 13) v |= ((hw[j >> 2] >> (8 * (j & 3) + g)) & 1u) << 4;
            out[j] = d * (float)v - m;
        }
        return true;
    }
    if (t == 14) {   /* Q6_K: (6-bit code - 32), a signed scale per 16 */
        bd_u32 b = e0 >> 8, e = e0 & 255u, n = e >> 7, r = e & 127u, base = b * 210u;
        bd_u32 ql[4], qh[4];
        bd_bytes16(W, base + 64u * n + (r & 63u), ql); bd_bytes16(W, base + 128u + 32u * n + (r & 31u), qh);
        bd_u32 ns = 4u * (r >> 6), hs = 2u * (r >> 5);
        float d = bd_f16at(W, base + 208u) * (float)(int)(signed char)bd_u8at(W, base + 192u + (e >> 4));
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int v = (int)(((ql[j >> 2] >> (8 * (j & 3) + ns)) & 15u) | (((qh[j >> 2] >> (8 * (j & 3) + hs)) & 3u) << 4)) - 32;
            out[j] = d * (float)v;
        }
        return true;
    }
    if (t == 8) {   /* Q8_0: blocks of 32 */
        bd_u32 b = e0 >> 5, base = b * 34u, w[4];
        bd_bytes16(W, base + 2u + 16u * ((e0 >> 4) & 1u), w);
        float d = bd_f16at(W, base);
        #pragma unroll
        for (int j = 0; j < 16; j++) out[j] = d * (float)(int)(signed char)bd_byte_of(w, j);
        return true;
    }
    if (t == 20 || t == 23) {   /* IQ4_NL (blocks of 32), IQ4_XS (256): kvalues of the nibbles, low for 0..15, high for 16..31 */
        bd_u32 w[4], hs = 4u * ((e0 >> 4) & 1u);
        float d;
        if (t == 20) { bd_u32 base = (e0 >> 5) * 18u; d = bd_f16at(W, base); bd_bytes16(W, base + 2u, w); }
        else {
            bd_u32 b = e0 >> 8, ib = (e0 & 255u) >> 5, base = b * 136u;
            bd_u32 w0 = bd_w(W, base >> 2), w2 = bd_w(W, (base >> 2) + 1u);
            bd_u32 ls = ((w2 >> (8u * (ib >> 1) + 4u * (ib & 1u))) & 15u) | (((w0 >> (16u + 2u * ib)) & 3u) << 4);
            d = bd_h2f(w0) * (float)((int)ls - 32);
            bd_u32 qa = (base + 8u + 16u * ib) >> 2;
            w[0] = bd_w(W, qa); w[1] = bd_w(W, qa + 1u); w[2] = bd_w(W, qa + 2u); w[3] = bd_w(W, qa + 3u);
        }
        #pragma unroll
        for (int j = 0; j < 16; j++) out[j] = d * (float)(int)__ldg(tab + BDT_IQ4NL + ((bd_byte_of(w, j) >> hs) & 15u));
        return true;
    }
    if (t == 16 || t == 18) {   /* IQ2_XXS, IQ3_XXS: a sub-block of 32, its grid indices, ksigns and a 4-bit scale */
        bd_u32 b = e0 >> 8, e = e0 & 255u, ib = e >> 5, l0 = ((e & 31u) >> 3), base = b * (t == 16 ? 66u : 98u);
        float d = bd_f16at(W, base);
        if (t == 16) {
            bd_u32 a0 = bd_ldu(W, base + 2u + 8u * ib), a1 = bd_ldu(W, base + 6u + 8u * ib);
            float s = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
            #pragma unroll
            for (int ll = 0; ll < 2; ll++) {
                bd_u32 l = l0 + (bd_u32)ll, idx = (a0 >> (8u * l)) & 255u, sg = __ldg(tab + BDT_KSIGNS + ((a1 >> (7u * l)) & 127u));
                bd_u32 g0 = __ldg(tab + BDT_IQ2XXS + 2u * idx), g1 = __ldg(tab + BDT_IQ2XXS + 2u * idx + 1u);
                #pragma unroll
                for (int j = 0; j < 8; j++) { float v = (float)(((j < 4 ? g0 : g1) >> (8 * (j & 3))) & 255u); out[8 * ll + j] = (sg >> j) & 1u ? -s * v : s * v; }
            }
        } else {
            bd_u32 a = bd_ldu(W, base + 66u + 4u * ib), qw = bd_ldu(W, base + 2u + 8u * ib + 2u * l0);
            float s = d * (0.5f + (float)(a >> 28)) * 0.5f;
            #pragma unroll
            for (int ll = 0; ll < 2; ll++) {
                bd_u32 l = l0 + (bd_u32)ll, sg = __ldg(tab + BDT_KSIGNS + ((a >> (7u * l)) & 127u));
                bd_u32 g0 = __ldg(tab + BDT_IQ3XXS + ((qw >> (16 * ll)) & 255u)), g1 = __ldg(tab + BDT_IQ3XXS + ((qw >> (16 * ll + 8)) & 255u));
                #pragma unroll
                for (int j = 0; j < 8; j++) { float v = (float)(((j < 4 ? g0 : g1) >> (8 * (j & 3))) & 255u); out[8 * ll + j] = (sg >> j) & 1u ? -s * v : s * v; }
            }
        }
        return true;
    }
    if (t == 17 || t == 22) {   /* IQ2_XS, IQ2_S: two chunks of 8, a grid row each, a 4-bit scale per 16 */
        bd_u32 b = e0 >> 8, e = e0 & 255u, k0 = e >> 3, base = b * (t == 17 ? 74u : 82u);
        float d = bd_f16at(W, base);
        bd_u32 si = e >> 4, scb = bd_u8at(W, base + (t == 17 ? 66u : 74u) + (si >> 1));
        float s = d * (0.5f + (float)((scb >> (4u * (si & 1u))) & 15u)) * 0.25f;
        #pragma unroll
        for (int ll = 0; ll < 2; ll++) {
            bd_u32 k = k0 + (bd_u32)ll, g0, g1, sg;
            if (t == 17) {
                bd_u32 qv = bd_u16at(W, base + 2u + 2u * k);
                g0 = __ldg(tab + BDT_IQ2XS + 2u * (qv & 511u)); g1 = __ldg(tab + BDT_IQ2XS + 2u * (qv & 511u) + 1u);
                sg = __ldg(tab + BDT_KSIGNS + (qv >> 9));
            } else {
                bd_u32 idx = bd_u8at(W, base + 2u + k) | (((bd_u8at(W, base + 66u + (k >> 2)) >> (2u * (k & 3u))) & 3u) << 8);
                g0 = __ldg(tab + BDT_IQ2S + 2u * idx); g1 = __ldg(tab + BDT_IQ2S + 2u * idx + 1u);
                sg = bd_u8at(W, base + 34u + k);
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) { float v = (float)(((j < 4 ? g0 : g1) >> (8 * (j & 3))) & 255u); out[8 * ll + j] = (sg >> j) & 1u ? -s * v : s * v; }
        }
        return true;
    }
    if (t == 21) {   /* IQ3_S: four chunks of 4, a 9-bit grid index each, sign bits, a scale per 32 */
        bd_u32 b = e0 >> 8, e = e0 & 255u, ib = e >> 5, base = b * 110u, k0 = e >> 2;
        float s = bd_f16at(W, base) * (float)(1u + 2u * ((bd_u8at(W, base + 106u + (ib >> 1)) >> (4u * (ib & 1u))) & 15u));
        bd_u32 sgw = bd_u16at(W, base + 74u + (e >> 3));   /* the 16 weights' sign bits */
        bd_u32 i4 = bd_ldu(W, base + 2u + k0);   /* k0 a multiple of 4: even */
        bd_u32 hb = (bd_u8at(W, base + 66u + (k0 >> 3)) >> (k0 & 7u)) & 15u;
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            bd_u32 idx = ((i4 >> (8 * l)) & 255u) | (((hb >> l) & 1u) << 8), gr = __ldg(tab + BDT_IQ3S + idx);
            #pragma unroll
            for (int j = 0; j < 4; j++) { float v = (float)((gr >> (8 * j)) & 255u); out[4 * l + j] = (sgw >> (4 * l + j)) & 1u ? -s * v : s * v; }
        }
        return true;
    }
    if (t == 10 || t == 11) {   /* Q2_K, Q3_K: 2-bit codes (Q3_K: the hmask bit), a scale per 16 */
        bd_u32 b = e0 >> 8, e = e0 & 255u, c = e >> 7, sh = 2u * ((e >> 5) & 3u), p = e & 31u, k = e >> 4;
        bd_u32 base = b * (t == 10 ? 84u : 110u), q[4];
        bd_bytes16(W, base + (t == 10 ? 16u : 32u) + 32u * c + p, q);
        float s, m = 0.f;
        bd_u32 hm[4] = {0u, 0u, 0u, 0u};
        if (t == 10) {
            bd_u32 sc = bd_u8at(W, base + k), dm = bd_ldu(W, base + 80u);
            s = bd_h2f(dm) * (float)(sc & 15u); m = bd_h2f(dm >> 16) * (float)(sc >> 4);
        } else {
            bd_u32 l = k < 8u ? (bd_u8at(W, base + 96u + k) & 15u) : (bd_u8at(W, base + 88u + k) >> 4);
            bd_u32 h = (bd_u8at(W, base + 104u + (k & 3u)) >> (2u * (k >> 2))) & 3u;
            s = bd_f16at(W, base + 108u) * (float)((int)(l | (h << 4)) - 32);
            bd_bytes16(W, base + p, hm);
        }
        bd_u32 hs = e >> 5;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int v = (int)((q[j >> 2] >> (8 * (j & 3) + sh)) & 3u);
            if (t == 11) v -= ((hm[j >> 2] >> (8 * (j & 3) + hs)) & 1u) ? 0 : 4;
            out[j] = s * (float)v - m;
        }
        return true;
    }
    return false;
}

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
#define bd_shfl_xor(v, o) __shfl_xor((v), (o), 32)
#else
#define bd_shfl_xor(v, o) __shfl_xor_sync(0xffffffffu, (v), (o), 32)
#endif

/* ---- the GEMVs: y[s, o] = sum_i x[s, i] * w[o, i] for a few rows of x ----------------------
 * A row of the matrix over lpr lanes (a power of two up to the warp, a few steps a lane:
 * bd_lpr), so a warp works on 32 / lpr rows at once; a block of 256 threads takes nr rows of
 * x (up to NR: a speculative verify's), each lane's steps against all of them, so the weights
 * are read once for the nr; at most BD_GEMV_BLOCKS blocks, the rest by a grid-stride (each
 * block prepares x once for more rows). chain_gemv_bd.comp's arrangement. */
#define BD_GEMV_NR 4
#define BD_GEMV_BLOCKS 256
/* lanes a row: about six 32-weight steps each (an RTX PRO 4000, Q4_K 10240 x 2560 against
 * four and eight lanes: 53 -> 46 us for one row, chain_gemv_bd's measure) */
__host__ __device__ inline int bd_lpr(int I) { int steps = I / 32, l = 1; while (l < 32 && l * 6 < steps) l *= 2; return l; }
/* the integer form's shared bytes for type T and nr rows of I inputs */
__host__ __device__ inline size_t bd_gemv_i8_smem(int T, int I, int nr) { return (size_t)bd_tab_words(T) * 4 + bd_x_bytes(I, nr); }

template <int T, int NR>
__global__ static void __launch_bounds__(256) bd_gemv_i8_k(float *__restrict__ y, const float *__restrict__ x, const bd_u32 *__restrict__ W,
        const bd_u32 *__restrict__ tab, int I, int O, int rowWords, int S, int nr, int lpr) {
    COLI_PDL_WAIT();
    extern __shared__ uint4 bd_sm4[];
    constexpr int TW = bd_tab_words(T);
    BdTabs tb = bd_tables_fill<T>(reinterpret_cast<bd_u32 *>(bd_sm4), tab);
    int s0 = (int)blockIdx.y * nr, nrow = min(nr, S - s0);
    int nb = I / 32, qr = bd_xq_row(I);
    uint4 *xq = bd_sm4 + TW / 4;   /* nr rows: what the launch sized (bd_gemv_i8_smem) */
    float *dx = reinterpret_cast<float *>(xq + nr * qr);
    int *xs = reinterpret_cast<int *>(dx + nr * nb);
    bdx_quant(x + (size_t)s0 * I, nrow, I, xq, dx, xs);
    __syncthreads();
    BdX X = {xq, dx, xs, qr, nb};
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int rps = 32 / lpr, sub = lane / lpr, ln = lane % lpr;
    int stride = (int)gridDim.x * 8 * rps;
    int steps = I / 32;
    for (int o0 = ((int)blockIdx.x * 8 + warp) * rps; o0 < O; o0 += stride) {
        int o = o0 + sub;
        bool live = o < O;
        const bd_u32 *Wr = W + (size_t)(live ? o : O - 1) * rowWords;
        float acc[NR];
        #pragma unroll
        for (int r = 0; r < NR; r++) acc[r] = 0.f;
        if (live) for (int q = ln; q < steps; q += lpr) bdi_step<T, NR>(Wr, (bd_u32)q, X, nrow, tb, acc);
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            float v = acc[r];
            for (int off = lpr / 2; off > 0; off >>= 1) v += bd_shfl_xor(v, off);
            if (ln == 0 && live) y[(size_t)(s0 + r) * O + o] = v;
        }
    }
}

/* The float form for every block type (bdf_decode): x as floats in shared memory (float4s
 * with gx4's padding), a warp a row, a 32-weight unit a lane, decoded once for the nr rows. */
__host__ __device__ inline size_t bd_gemv_f_smem(int I, int nr) { return (size_t)nr * (size_t)(bd_x4(I / 4) + 1) * 16; }
template <int NR>
__global__ static void __launch_bounds__(256) bd_gemv_f_k(float *__restrict__ y, const float *__restrict__ x, const bd_u32 *__restrict__ W,
        const bd_u32 *__restrict__ tab, int fmt, int I, int O, int rowWords, int S, int nr) {
    COLI_PDL_WAIT();
    extern __shared__ float4 bd_smf[];
    int s0 = (int)blockIdx.y * nr, nrow = min(nr, S - s0);
    int I4 = I / 4, XR = bd_x4(I4) + 1;
    const float4 *x4 = reinterpret_cast<const float4 *>(x + (size_t)s0 * I);
    for (int k = threadIdx.x; k < nrow * I4; k += blockDim.x) { int r = k / I4, kk = k - r * I4; bd_smf[r * XR + bd_x4(kk)] = x4[(size_t)r * I4 + kk]; }
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, units = I / 32;
    for (int o = (int)blockIdx.x * 8 + warp; o < O; o += (int)gridDim.x * 8) {
        const bd_u32 *Wr = W + (size_t)o * rowWords;
        float acc[NR];
        #pragma unroll
        for (int r = 0; r < NR; r++) acc[r] = 0.f;
        for (int u = lane; u < units; u += 32) {
            int wv[32]; float sA, mA, sB, mB;
            bdf_decode(fmt, Wr, (bd_u32)u, tab, wv, sA, mA, sB, mB);
            #pragma unroll
            for (int r = 0; r < NR; r++) if (r < nrow) {
                const float4 *xr = bd_smf + r * XR;
                float a = 0.f, b = 0.f, xa = 0.f, xb = 0.f;
                #pragma unroll
                for (int j4 = 0; j4 < 4; j4++) {
                    float4 p = xr[bd_x4(8 * u + j4)], q = xr[bd_x4(8 * u + 4 + j4)];
                    a += (float)wv[4 * j4] * p.x + (float)wv[4 * j4 + 1] * p.y + (float)wv[4 * j4 + 2] * p.z + (float)wv[4 * j4 + 3] * p.w;
                    b += (float)wv[16 + 4 * j4] * q.x + (float)wv[16 + 4 * j4 + 1] * q.y + (float)wv[16 + 4 * j4 + 2] * q.z + (float)wv[16 + 4 * j4 + 3] * q.w;
                    xa += p.x + p.y + p.z + p.w; xb += q.x + q.y + q.z + q.w;
                }
                acc[r] += sA * a - mA * xa + sB * b - mB * xb;
            }
        }
        #pragma unroll
        for (int r = 0; r < NR; r++) if (r < nrow) {
            float v = acc[r];
            for (int off = 16; off > 0; off >>= 1) v += bd_shfl_xor(v, off);
            if (lane == 0) y[(size_t)(s0 + r) * O + o] = v;
        }
    }
}

/* The launch for a block-format matrix (fmt 100 + t, rows of rowWords words at W, I a
 * multiple of 32 and of the type's block) on S rows of x: the integer form where the type
 * has one (each verify's rows a block, up to BD_GEMV_NR), else the float form. smem_max:
 * the shared bytes a block may take (the device's opt-in limit). 0 when it cannot run. */
/* a kernel's dynamic shared memory past the 48 KiB default: asked for once a device (bit
 * `device` of *raised) */
static inline void bd_smem_raise(const void *k, size_t sm, size_t smem_max, unsigned *raised) {
    int dev = 0;
    if (sm <= 48 * 1024 || cudaGetDevice(&dev) != cudaSuccess || dev > 31 || (*raised >> dev & 1u)) return;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_max);
    *raised |= 1u << dev;
}
template <int T>
static int bd_gemv_i8_launch(float *y, const float *x, const bd_u32 *W, const bd_u32 *tab, int I, int O, int rowWords, int S,
                             size_t smem_max, cudaStream_t st) {
    int nr = S < BD_GEMV_NR ? S : BD_GEMV_NR;
    while (nr > 1 && bd_gemv_i8_smem(T, I, nr) > smem_max) nr--;
    size_t sm = bd_gemv_i8_smem(T, I, nr);
    if (sm > smem_max) return 0;
    int lpr = bd_lpr(I), rows_b = 256 / lpr, bx = (O + rows_b - 1) / rows_b;
    if (bx > BD_GEMV_BLOCKS) bx = BD_GEMV_BLOCKS;
    dim3 g((unsigned)bx, (unsigned)((S + nr - 1) / nr));
    if (nr > 1) {
        static unsigned raised;   /* past 48 KiB a kernel asks for it, once a device */
        bd_smem_raise((const void *)bd_gemv_i8_k<T, BD_GEMV_NR>, sm, smem_max, &raised);
        coli_launch(bd_gemv_i8_k<T, BD_GEMV_NR>, g, 256, sm, st, y, x, W, tab, I, O, rowWords, S, nr, lpr);
    } else {
        static unsigned raised;
        bd_smem_raise((const void *)bd_gemv_i8_k<T, 1>, sm, smem_max, &raised);
        coli_launch(bd_gemv_i8_k<T, 1>, g, 256, sm, st, y, x, W, tab, I, O, rowWords, S, 1, lpr);
    }
    return 1;
}
static int bd_gemv_f_launch(float *y, const float *x, const bd_u32 *W, const bd_u32 *tab, int fmt, int I, int O, int rowWords, int S,
                            size_t smem_max, cudaStream_t st) {
    int nr = S < BD_GEMV_NR ? S : BD_GEMV_NR;
    while (nr > 1 && bd_gemv_f_smem(I, nr) > smem_max) nr--;
    size_t sm = bd_gemv_f_smem(I, nr);
    if (sm > smem_max) return 0;
    int bx = (O + 7) / 8;
    if (bx > 4 * BD_GEMV_BLOCKS) bx = 4 * BD_GEMV_BLOCKS;
    dim3 g((unsigned)bx, (unsigned)((S + nr - 1) / nr));
    if (nr > 1) {
        static unsigned raised;   /* past 48 KiB a kernel asks for it, once a device */
        bd_smem_raise((const void *)bd_gemv_f_k<BD_GEMV_NR>, sm, smem_max, &raised);
        coli_launch(bd_gemv_f_k<BD_GEMV_NR>, g, 256, sm, st, y, x, W, tab, fmt, I, O, rowWords, S, nr);
    } else {
        static unsigned raised;
        bd_smem_raise((const void *)bd_gemv_f_k<1>, sm, smem_max, &raised);
        coli_launch(bd_gemv_f_k<1>, g, 256, sm, st, y, x, W, tab, fmt, I, O, rowWords, S, 1);
    }
    return 1;
}
/* 1 = fmt is a block format this file decodes (I a multiple of its block, checked by the caller) */
static inline int bd_fmt_ok(int fmt) {
    if (fmt < 102 || fmt >= 143) return 0;
    int t = fmt - 100;
    return BD_BLK[t] >= 32;
}
static int bd_gemv_launch(float *y, const float *x, const bd_u32 *W, const bd_u32 *tab, int fmt, int I, int O, int rowWords, int S,
                          size_t smem_max, int force_float, cudaStream_t st) {
    if (!bd_fmt_ok(fmt) || I % 32 || I % BD_BLK[fmt - 100]) return 0;
    if (!force_float) switch (fmt - 100) {
#define BD_I8_CASE(t) case t: return bd_gemv_i8_launch<t>(y, x, W, tab, I, O, rowWords, S, smem_max, st);
        BD_I8_CASE(8) BD_I8_CASE(10) BD_I8_CASE(11) BD_I8_CASE(12) BD_I8_CASE(13) BD_I8_CASE(14) BD_I8_CASE(16) BD_I8_CASE(17)
        BD_I8_CASE(18) BD_I8_CASE(19) BD_I8_CASE(20) BD_I8_CASE(21) BD_I8_CASE(22) BD_I8_CASE(23) BD_I8_CASE(29) BD_I8_CASE(42)
#undef BD_I8_CASE
        default: break;
    }
    return bd_gemv_f_launch(y, x, W, tab, fmt, I, O, rowWords, S, smem_max, st);
}
#endif
