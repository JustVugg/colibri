/* vq2_g32.h — colibri's vq2 routed-expert format: two-stage vector codes, and its kernels.
 *
 *   codes   a row of I weights (I % 32 == 0) is I/8 vectors of 8 weights; each vector is
 *           two bytes, an index into codebook A and one into codebook B (I/4 bytes a row,
 *           2 bits a weight), in column order.
 *   values  the vector is A[a] + B[b]: two 256 x 8 int8 codebooks of the model (the
 *           sidecar's index carries them), scaled so that in every coordinate the
 *           largest |A| plus the largest |B| is at most 127: any sum is an int8 and the
 *           integer path of int2_g32.h / lowbit_g16.h reads it as it reads their levels.
 *           Fitted by k-means to the experts' weight vectors (over the RMS of their group
 *           of 32), then rounded.
 *   scales  one 4-bit scale per 32 inputs (two to a byte: group 2k low, 2k+1 high, I/64
 *           bytes a row) under one f32 per row: w = d[row] * q * (A[a] + B[b]), the
 *           codebooks' own scale folded into d.
 *   size    2.125 bits per weight, plus 32 bits per row.
 *   x       the activation contract of int2_g32.h (x2_quant_act): int8 per 32, one f32
 *           scale each.
 *
 * Measured on 23 Qwen3.8 experts against the rows they receive (held-out tokens, GPTQ
 * under the same H): output error 0.371 against 0.398 for int2-g16 and 0.435 for ISTA's
 * Q2_0, at 2.125 bits a weight against 2.27 and 2.25; on 6 of them the converter's beam
 * of 4 and int8 codebooks cost 0.004 over a beam of 16 in floats.
 *
 * The scalar reference defines the numerics (lane l = elements 4l..4l+3 of a group, one
 * fmaf per group with d*dx[g] on the lane's integer sum times q, the lanes summed as
 * x2_hsum8); the AVX2 kernel reproduces it bit for bit for every number of rows.
 * Header-only, all static, pure compute. */
#ifndef COLI_VQ2_G32_H
#define COLI_VQ2_G32_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "int2_g32.h"   /* x2_quant_act, x2_hsum8_scalar, X2_MAX_ROWS */

#define VQ2_BOOK 256
typedef struct { int8_t a[VQ2_BOOK][8], b[VQ2_BOOK][8]; } Vq2Books;

static inline int    vq2_layout_ok(int I)    { return I > 0 && I % 32 == 0; }
static inline size_t vq2_code_bytes(int I)   { return (size_t)I / 4; }
static inline size_t vq2_scale_bytes(int I)  { return (size_t)I / 64 + (I % 64 ? 1 : 0); }
static inline int    vq2_scale(const uint8_t *q, int g) { return (q[g / 2] >> (4 * (g & 1))) & 15; }
/* weight value (before d and q) of element i of a row */
static inline int vq2_value(const Vq2Books *bk, const uint8_t *row, int i) {
    const uint8_t *v = row + (size_t)(i / 8) * 2;
    return bk->a[v[0]][i % 8] + bk->b[v[1]][i % 8];
}

/* y = sum_i w[i] * x[i] for one weight row and one activation row (S = 1) */
static inline float vq2_dot_ref(const Vq2Books *bk, const uint8_t *codes, const uint8_t *q, float d,
                                const int8_t *xq, const float *dx, int I) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int g = 0; g < I / 32; g++) {
        const int8_t *xb = xq + (size_t)g * 32;
        const float sc = d * dx[g];
        const int qs = vq2_scale(q, g);
        for (int l = 0; l < 8; l++) {
            int32_t s = 0;
            for (int j = 4 * l; j < 4 * l + 4; j++) s += vq2_value(bk, codes, g * 32 + j) * xb[j];
            acc[l] = fmaf((float)(s * qs), sc, acc[l]);
        }
    }
    return x2_hsum8_scalar(acc);
}

#ifdef X2_HAVE_AVX2
#define VQ2_HAVE_AVX2 1
static inline __attribute__((always_inline)) void vq2_dot_body(float *y, int ystride, const Vq2Books *bk,
                                                              const uint8_t *codes, const uint8_t *q, float d,
                                                              const int8_t *xq, const float *dx, int I, const int S) {
    const int ng = I / 32;
    __m256 acc[X2_MAX_ROWS];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    for (int g = 0; g < ng; g++) {
        const uint8_t *v = codes + (size_t)g * 8;   /* four vectors: a0 b0 a1 b1 a2 b2 a3 b3 */
        int64_t a0, a1, a2, a3, b0, b1, b2, b3;
        memcpy(&a0, bk->a[v[0]], 8); memcpy(&b0, bk->b[v[1]], 8);
        memcpy(&a1, bk->a[v[2]], 8); memcpy(&b1, bk->b[v[3]], 8);
        memcpy(&a2, bk->a[v[4]], 8); memcpy(&b2, bk->b[v[5]], 8);
        memcpy(&a3, bk->a[v[6]], 8); memcpy(&b3, bk->b[v[7]], 8);
        const __m256i w = _mm256_add_epi8(_mm256_set_epi64x(a3, a2, a1, a0), _mm256_set_epi64x(b3, b2, b1, b0));
        const __m256i aw = _mm256_sign_epi8(w, w), qv = _mm256_set1_epi16((short)vq2_scale(q, g));
        for (int s = 0; s < S; s++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (size_t)s * I + (size_t)g * 32));
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)), qv);
            acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[(size_t)s * ng + g]), acc[s]);
        }
    }
    for (int s = 0; s < S; s++) y[(size_t)s * ystride] = x2_hsum8(acc[s]);
}
#endif

/* S (1..X2_MAX_ROWS) activation rows of one weight row; xq/dx row-major [S][I], [S][I/32] */
static inline void vq2_dot(float *y, int ystride, const Vq2Books *bk, const uint8_t *codes, const uint8_t *q, float d,
                           const int8_t *xq, const float *dx, int I, int S) {
#ifdef VQ2_HAVE_AVX2
    switch (S) {
    case 1: vq2_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 1); break;
    case 2: vq2_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 2); break;
    case 3: vq2_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 3); break;
    default: vq2_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 4); break;
    }
#else
    for (int s = 0; s < S; s++)
        y[(size_t)s * ystride] = vq2_dot_ref(bk, codes, q, d, xq + (size_t)s * I, dx + (size_t)s * (I / 32), I);
#endif
}

/* y[s*O + o] = W[o,:] . x[s,:] for a vq2 matrix: codes [O][I/4], scales [O][vq2_scale_bytes],
 * d [O]. Activations quantized once, rows four at a time. 0, or -1 when out of memory. */
static inline int vq2_matmul(float *y, const float *x, const Vq2Books *bk, const uint8_t *codes, const uint8_t *q,
                             const float *d, int S, int I, int O) {
    const size_t cb = vq2_code_bytes(I), qb = vq2_scale_bytes(I); const int ng = I / 32;
    int8_t *xq = (int8_t *)malloc((size_t)S * I); float *dx = (float *)malloc((size_t)S * ng * sizeof(float));
    if (!xq || !dx) { free(xq); free(dx); return -1; }
    for (int s = 0; s < S; s++) x2_quant_act(x + (size_t)s * I, I, xq + (size_t)s * I, dx + (size_t)s * ng);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        for (int s0 = 0; s0 < S; s0 += X2_MAX_ROWS) {
            const int n = S - s0 < X2_MAX_ROWS ? S - s0 : X2_MAX_ROWS;
            vq2_dot(y + (size_t)s0 * O + o, O, bk, codes + (size_t)o * cb, q + (size_t)o * qb, d[o],
                    xq + (size_t)s0 * I, dx + (size_t)s0 * ng, I, n);
        }
    free(xq); free(dx);
    return 0;
}


/* ---- rows interleaved by 16 (vq2r16_*) -------------------------------------------
 * As lowbit_g16.h's xl16: 16 rows share a register (lane r = row r), so the float work is
 * one fma per 32 inputs for all of them and nothing is summed across lanes. A group of 16
 * rows holds their 16 f32 row scales, then per 32 inputs (g) a block of
 *   8 bytes    the 16 rows' 4-bit scales of g (row r: byte r / 2, nibble r % 2)
 *   128 bytes  the four vectors' codes, vector v: the 16 rows' (a, b) byte pairs (32 bytes)
 * the same bytes as the rows (2.125 bits a weight and 4 a row). The vectors are still looked
 * up in the codebooks, 16 rows at a time; A[a] + B[b] is an int8 (|.| <= 127), taken as
 * value + 128 by vpdpbusd (signed with the sign trick where there is no VNNI).
 * Numerics: per 32 inputs the exact integer sum value * x times the scale (below 2^24:
 * exact as a float), then one fma with d * dx: vq2r16_ref, not vq2_dot's eight lane sums. */
static inline size_t vq2r16_gblock(void) { return 136; }
static inline size_t vq2r16_group_bytes(int I) { return 64 + (size_t)(I / 32) * vq2r16_gblock(); }
static inline int vq2r16_layout_ok(int rows, int I) { return rows % 16 == 0 && I % 64 == 0; }
static void vq2r16_pack(uint8_t *dst, const uint8_t *codes, const uint8_t *q, const float *d, int rows, int I) {
    const size_t cb = vq2_code_bytes(I), qb = vq2_scale_bytes(I), gb = vq2r16_group_bytes(I);
    for (int g = 0; g < rows / 16; g++) {
        uint8_t *grp = dst + (size_t)g * gb;
        memcpy(grp, d + 16 * g, 64);
        for (int gi = 0; gi < I / 32; gi++) {
            uint8_t *o = grp + 64 + (size_t)gi * vq2r16_gblock();
            memset(o, 0, 8);
            for (int r = 0; r < 16; r++) {
                const uint8_t *row = codes + (size_t)(16 * g + r) * cb;
                o[r / 2] |= (uint8_t)(vq2_scale(q + (size_t)(16 * g + r) * qb, gi) << (4 * (r % 2)));
                for (int v = 0; v < 4; v++) memcpy(o + 8 + 32 * v + 2 * r, row + (size_t)gi * 8 + 2 * v, 2);
            }
        }
    }
}
static void vq2r16_unpack(uint8_t *codes, uint8_t *q, float *d, const uint8_t *src, int rows, int I) {
    const size_t cb = vq2_code_bytes(I), qb = vq2_scale_bytes(I), gb = vq2r16_group_bytes(I);
    memset(q, 0, qb * (size_t)rows);
    for (int g = 0; g < rows / 16; g++) {
        const uint8_t *grp = src + (size_t)g * gb;
        memcpy(d + 16 * g, grp, 64);
        for (int gi = 0; gi < I / 32; gi++) {
            const uint8_t *o = grp + 64 + (size_t)gi * vq2r16_gblock();
            for (int r = 0; r < 16; r++) {
                uint8_t *row = codes + (size_t)(16 * g + r) * cb, *qr = q + (size_t)(16 * g + r) * qb;
                qr[gi / 2] |= (uint8_t)(((o[r / 2] >> (4 * (r % 2))) & 15) << (4 * (gi % 2)));
                for (int v = 0; v < 4; v++) memcpy(row + (size_t)gi * 8 + 2 * v, o + 8 + 32 * v + 2 * r, 2);
            }
        }
    }
}
static inline float vq2r16_ref(const Vq2Books *bk, const uint8_t *codes, const uint8_t *q, float d,
                               const int8_t *xq, const float *dx, int I) {
    float acc = 0.f;
    for (int g = 0; g < I / 32; g++) {
        int32_t p = 0;
        for (int j = 0; j < 32; j++) p += vq2_value(bk, codes, g * 32 + j) * xq[(size_t)g * 32 + j];
        acc = fmaf((float)(p * vq2_scale(q, g)), d * dx[g], acc);
    }
    return acc;
}
#if defined(__AVX2__) && defined(__FMA__)
#define VQ2R16_ON 1
static inline int32_t vq2r16_ld32(const int8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
static inline int64_t vq2r16_row(const int8_t (*t)[8], int i) { int64_t v; memcpy(&v, t[i], 8); return v; }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
/* the eight sums A[a] + B[b] of rows r0 .. r0+7 of vector v (pairs at c) */
static inline __m512i vq2r16_vec8(const Vq2Books *bk, const uint8_t *c) {
    const __m512i a = _mm512_set_epi64(vq2r16_row(bk->a, c[14]), vq2r16_row(bk->a, c[12]), vq2r16_row(bk->a, c[10]), vq2r16_row(bk->a, c[8]),
                                       vq2r16_row(bk->a, c[6]), vq2r16_row(bk->a, c[4]), vq2r16_row(bk->a, c[2]), vq2r16_row(bk->a, c[0]));
    const __m512i b = _mm512_set_epi64(vq2r16_row(bk->b, c[15]), vq2r16_row(bk->b, c[13]), vq2r16_row(bk->b, c[11]), vq2r16_row(bk->b, c[9]),
                                       vq2r16_row(bk->b, c[7]), vq2r16_row(bk->b, c[5]), vq2r16_row(bk->b, c[3]), vq2r16_row(bk->b, c[1]));
    return _mm512_add_epi8(a, b);
}
/* the 16 rows' codes of every g as value + 128, eight registers a g (for many activation rows) */
static inline void vq2r16_expand512(const Vq2Books *bk, const uint8_t *grp, int I, __m512i *buf) {
    const __m512i ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), od = _mm512_add_epi32(ev, _mm512_set1_epi32(1));
    const __m512i flip = _mm512_set1_epi8((char)0x80);
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *o = grp + 64 + (size_t)g * vq2r16_gblock();
        for (int v = 0; v < 4; v++) {
            const __m512i lo = vq2r16_vec8(bk, o + 8 + 32 * v), hi = vq2r16_vec8(bk, o + 8 + 32 * v + 16);
            _mm512_storeu_si512(buf + 8 * g + 2 * v, _mm512_xor_si512(_mm512_permutex2var_epi32(lo, ev, hi), flip));
            _mm512_storeu_si512(buf + 8 * g + 2 * v + 1, _mm512_xor_si512(_mm512_permutex2var_epi32(lo, od, hi), flip));
        }
    }
}
static inline __attribute__((always_inline)) void vq2r16_body512(float *y, int ystride, const Vq2Books *bk, const uint8_t *grp,
                                                                const int8_t *xq, const float *dx, const int32_t *xs, int I, const int S,
                                                                const __m512i *buf) {
    const __m512i ev = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), od = _mm512_add_epi32(ev, _mm512_set1_epi32(1));
    const __m512i flip = _mm512_set1_epi8((char)0x80);
    const __m512 dv = _mm512_loadu_ps((const float *)grp);
    __m512 acc[8];
    for (int s = 0; s < S; s++) acc[s] = _mm512_setzero_ps();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *o = grp + 64 + (size_t)g * vq2r16_gblock();
        const __m128i qb = _mm_loadl_epi64((const __m128i *)o);
        const __m128i qn = _mm_unpacklo_epi8(_mm_and_si128(qb, _mm_set1_epi8(15)), _mm_and_si128(_mm_srli_epi16(qb, 4), _mm_set1_epi8(15)));
        const __m512 sq = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qn));
        __m512i c[8];
        if (buf) for (int j = 0; j < 8; j++) c[j] = _mm512_loadu_si512(buf + 8 * g + j);
        else for (int v = 0; v < 4; v++) {
            const __m512i lo = vq2r16_vec8(bk, o + 8 + 32 * v), hi = vq2r16_vec8(bk, o + 8 + 32 * v + 16);
            c[2 * v] = _mm512_xor_si512(_mm512_permutex2var_epi32(lo, ev, hi), flip);
            c[2 * v + 1] = _mm512_xor_si512(_mm512_permutex2var_epi32(lo, od, hi), flip);
        }
        for (int s = 0; s < S; s++) {
            const int8_t *xb = xq + (size_t)s * I + 32 * (size_t)g;
            __m512i p = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c[0], _mm512_set1_epi32(vq2r16_ld32(xb)));
            for (int j = 1; j < 8; j++) p = _mm512_dpbusd_epi32(p, c[j], _mm512_set1_epi32(vq2r16_ld32(xb + 4 * j)));
            const int32_t *x16 = xs + (size_t)s * (I / 16) + 2 * (size_t)g;
            p = _mm512_sub_epi32(p, _mm512_set1_epi32(128 * (x16[0] + x16[1])));
            acc[s] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(p), sq), _mm512_mul_ps(dv, _mm512_set1_ps(dx[(size_t)s * (I / 32) + g])), acc[s]);
        }
    }
    for (int s = 0; s < S; s++) _mm512_storeu_ps(y + (size_t)s * ystride, acc[s]);
}
#define VQ2R16_SMAX 8
#else
/* rows 8h .. 8h+7 of vector v: two registers of quads (elements 0-3, 4-7) */
static inline void vq2r16_vec_h(const Vq2Books *bk, const uint8_t *c, __m256i *q0, __m256i *q1) {
    const __m256i s0 = _mm256_add_epi8(_mm256_set_epi64x(vq2r16_row(bk->a, c[6]), vq2r16_row(bk->a, c[4]), vq2r16_row(bk->a, c[2]), vq2r16_row(bk->a, c[0])),
                                       _mm256_set_epi64x(vq2r16_row(bk->b, c[7]), vq2r16_row(bk->b, c[5]), vq2r16_row(bk->b, c[3]), vq2r16_row(bk->b, c[1])));
    const __m256i s1 = _mm256_add_epi8(_mm256_set_epi64x(vq2r16_row(bk->a, c[14]), vq2r16_row(bk->a, c[12]), vq2r16_row(bk->a, c[10]), vq2r16_row(bk->a, c[8])),
                                       _mm256_set_epi64x(vq2r16_row(bk->b, c[15]), vq2r16_row(bk->b, c[13]), vq2r16_row(bk->b, c[11]), vq2r16_row(bk->b, c[9])));
    const __m256i sel = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i p0 = _mm256_permutevar8x32_epi32(s0, sel), p1 = _mm256_permutevar8x32_epi32(s1, sel);
    *q0 = _mm256_permute2x128_si256(p0, p1, 0x20); *q1 = _mm256_permute2x128_si256(p0, p1, 0x31);
}
/* half h's codes of every g as the products take them (value + 128 with VNNI, signed without) */
static inline void vq2r16_expand256(const Vq2Books *bk, const uint8_t *grp, int I, int h, __m256i *buf) {
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *o = grp + 64 + (size_t)g * vq2r16_gblock();
        for (int v = 0; v < 4; v++) vq2r16_vec_h(bk, o + 8 + 32 * v + 16 * h, &buf[8 * g + 2 * v], &buf[8 * g + 2 * v + 1]);
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
        for (int j = 0; j < 8; j++) buf[8 * g + j] = _mm256_xor_si256(buf[8 * g + j], _mm256_set1_epi8((char)0x80));
#endif
    }
}
static inline __attribute__((always_inline)) void vq2r16_body256(float *y, int ystride, const Vq2Books *bk, const uint8_t *grp,
                                                                const int8_t *xq, const float *dx, const int32_t *xs, int I, const int S, const int h,
                                                                const __m256i *buf) {
    const __m256 dv = _mm256_loadu_ps((const float *)grp + 8 * h);
    __m256 acc[4];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *o = grp + 64 + (size_t)g * vq2r16_gblock();
        const __m128i qb = _mm_cvtsi32_si128(vq2r16_ld32((const int8_t *)o + 4 * h));
        const __m128i qn = _mm_unpacklo_epi8(_mm_and_si128(qb, _mm_set1_epi8(15)), _mm_and_si128(_mm_srli_epi16(qb, 4), _mm_set1_epi8(15)));
        const __m256 sq = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qn));
        __m256i c[8];
        if (buf) for (int j = 0; j < 8; j++) c[j] = buf[8 * g + j];
        else {
            for (int v = 0; v < 4; v++) vq2r16_vec_h(bk, o + 8 + 32 * v + 16 * h, &c[2 * v], &c[2 * v + 1]);
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
            for (int j = 0; j < 8; j++) c[j] = _mm256_xor_si256(c[j], _mm256_set1_epi8((char)0x80));
#endif
        }
        for (int s = 0; s < S; s++) {
            const int8_t *xb = xq + (size_t)s * I + 32 * (size_t)g;
            __m256i p = _mm256_setzero_si256();
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
            for (int j = 0; j < 8; j++)
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                p = _mm256_dpbusd_epi32(p, c[j], _mm256_set1_epi32(vq2r16_ld32(xb + 4 * j)));
#else
                p = _mm256_dpbusd_avx_epi32(p, c[j], _mm256_set1_epi32(vq2r16_ld32(xb + 4 * j)));
#endif
            const int32_t *x16 = xs + (size_t)s * (I / 16) + 2 * (size_t)g;
            p = _mm256_sub_epi32(p, _mm256_set1_epi32(128 * (x16[0] + x16[1])));
#else   /* the signed values, |w| <= 127: a maddubs pair stays below 2^15 */
            for (int j = 0; j < 8; j++)
                p = _mm256_add_epi32(p, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(c[j], c[j]),
                        _mm256_sign_epi8(_mm256_set1_epi32(vq2r16_ld32(xb + 4 * j)), c[j])), _mm256_set1_epi16(1)));
#endif
            acc[s] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(p), sq), _mm256_mul_ps(dv, _mm256_set1_ps(dx[(size_t)s * (I / 32) + g])), acc[s]);
        }
    }
    for (int s = 0; s < S; s++) _mm256_storeu_ps(y + (size_t)s * ystride + 8 * h, acc[s]);
}
#define VQ2R16_SMAX 4
#endif
/* group grp's 16 rows (outputs y .. y + 15) against S activation rows (the activation
 * contract plus the sums per 16 of its bytes, gq_act's q, d, s). Past one pass of rows the
 * codes are looked up once into a per-thread buffer (16 x I bytes) and read from there. */
static void vq2r16_rows(float *y, int ystride, const Vq2Books *bk, const uint8_t *grp, const int8_t *xq, const float *dx,
                        const int32_t *xs, int I, int S) {
#if VQ2R16_SMAX == 8
    typedef __m512i vq2r16_v;
#else
    typedef __m256i vq2r16_v;
#endif
    static __thread vq2r16_v *buf; static __thread void *raw; static __thread int cap;
    const int many = S > VQ2R16_SMAX, need = I / 32 * 8;
    if (many && cap < need) {   /* aligned by hand: Windows' C library has no aligned_alloc */
        free(raw); raw = malloc((size_t)need * sizeof(vq2r16_v) + 63);
        buf = raw ? (vq2r16_v *)(((uintptr_t)raw + 63) & ~(uintptr_t)63) : NULL; cap = buf ? need : 0;
        if (!buf) { fprintf(stderr, "OOM vq2 rows\n"); exit(1); }
    }
#if VQ2R16_SMAX == 8
    if (many) vq2r16_expand512(bk, grp, I, buf);
    for (int s0 = 0; s0 < S; s0 += 8) {
        const int n = S - s0 < 8 ? S - s0 : 8;
        float *ys = y + (size_t)s0 * ystride; const int8_t *xb = xq + (size_t)s0 * I;
        const float *db = dx + (size_t)s0 * (I / 32); const int32_t *sb = xs + (size_t)s0 * (I / 16);
        const vq2r16_v *bf = many ? buf : NULL;
        switch (n) { case 1: vq2r16_body512(ys, ystride, bk, grp, xb, db, sb, I, 1, bf); break;
                     case 2: vq2r16_body512(ys, ystride, bk, grp, xb, db, sb, I, 2, bf); break;
                     case 3: vq2r16_body512(ys, ystride, bk, grp, xb, db, sb, I, 3, bf); break;
                     case 4: vq2r16_body512(ys, ystride, bk, grp, xb, db, sb, I, 4, bf); break;
                     case 8: vq2r16_body512(ys, ystride, bk, grp, xb, db, sb, I, 8, bf); break;
                     default: vq2r16_body512(ys, ystride, bk, grp, xb, db, sb, I, n, bf); break; }
    }
#else
    for (int h = 0; h < 2; h++) {
        if (many) vq2r16_expand256(bk, grp, I, h, buf);
        for (int s0 = 0; s0 < S; s0 += 4) {
            const int n = S - s0 < 4 ? S - s0 : 4;
            float *ys = y + (size_t)s0 * ystride; const int8_t *xb = xq + (size_t)s0 * I;
            const float *db = dx + (size_t)s0 * (I / 32); const int32_t *sb = xs + (size_t)s0 * (I / 16);
            const vq2r16_v *bf = many ? buf : NULL;
            switch (n) { case 1: vq2r16_body256(ys, ystride, bk, grp, xb, db, sb, I, 1, h, bf); break;
                         case 2: vq2r16_body256(ys, ystride, bk, grp, xb, db, sb, I, 2, h, bf); break;
                         case 3: vq2r16_body256(ys, ystride, bk, grp, xb, db, sb, I, 3, h, bf); break;
                         default: vq2r16_body256(ys, ystride, bk, grp, xb, db, sb, I, 4, h, bf); break; }
        }
    }
#endif
}
#endif

#endif /* COLI_VQ2_G32_H */
