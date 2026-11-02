/* vq1_g64.h — colibri's vq1 routed-expert format: 16-dim vector codes at ~1.3 bits a weight.
 *
 *   codes   a row of I weights (I % 128 == 0) is I/16 vectors of 16 weights; each vector is
 *           an index into codebook A and one into codebook B, 10 bits each. The eight
 *           indices of a group of 64 weights (four vectors: a0 b0 a1 b1 a2 b2 a3 b3) are
 *           packed into 10 bytes, index k at bits 10k..10k+9, least significant first:
 *           I/64 x 10 bytes a row.
 *   values  the vector is A[a] + B[b]: two 1024 x 16 int8 codebooks of the model (the
 *           sidecar's index carries them), scaled so that in every coordinate the
 *           largest |A| plus the largest |B| is at most 127, so any sum is an int8.
 *   scales  one 4-bit scale per 64 inputs (two to a byte: group 2k low, 2k+1 high, I/128
 *           bytes a row) under one f32 per row: w = d[row] * q * (A[a] + B[b]).
 *   size    1.3125 bits per weight, plus 32 bits per row.
 *   x       the activation contract of int2_g32.h (x2_quant_act): int8 per 32, one f32
 *           scale each, so a group of 64 weights meets two activation scales.
 *
 * The scalar reference defines the numerics as vq2_g32.h's: per block of 32 inputs, lane
 * l = elements 4l..4l+3, one fmaf with d*dx[block] on the lane's integer sum times the
 * group's q, the lanes summed as x2_hsum8; the AVX2 kernel reproduces it bit for bit for
 * every number of rows. Header-only, all static, pure compute. */
#ifndef COLI_VQ1_G64_H
#define COLI_VQ1_G64_H

#include <string.h>
#include "int2_g32.h"   /* x2_quant_act, x2_hsum8_scalar, X2_MAX_ROWS */

#define VQ1_BOOK 1024
typedef struct { int8_t a[VQ1_BOOK][16], b[VQ1_BOOK][16]; } Vq1Books;

static inline int    vq1_layout_ok(int I)    { return I > 0 && I % 128 == 0; }
static inline size_t vq1_code_bytes(int I)   { return (size_t)I / 64 * 10; }
static inline size_t vq1_scale_bytes(int I)  { return (size_t)I / 128; }
static inline int    vq1_scale(const uint8_t *q, int g) { return (q[g / 2] >> (4 * (g & 1))) & 15; }
/* the eight 10-bit indices of group g of a row */
static inline void vq1_indices(const uint8_t *row, int g, int idx[8]) {
    uint64_t lo; uint16_t hi;
    memcpy(&lo, row + (size_t)g * 10, 8); memcpy(&hi, row + (size_t)g * 10 + 8, 2);
    for (int k = 0; k < 6; k++) idx[k] = (int)((lo >> (10 * k)) & 1023);
    idx[6] = (int)(((lo >> 60) | ((uint64_t)hi << 4)) & 1023);
    idx[7] = (int)((hi >> 6) & 1023);
}
/* weight value (before d and q) of element i of a row */
static inline int vq1_value(const Vq1Books *bk, const uint8_t *row, int i) {
    int idx[8]; vq1_indices(row, i / 64, idx);
    const int v = (i % 64) / 16;
    return bk->a[idx[2 * v]][i % 16] + bk->b[idx[2 * v + 1]][i % 16];
}

/* y = sum_i w[i] * x[i] for one weight row and one activation row (S = 1) */
static inline float vq1_dot_ref(const Vq1Books *bk, const uint8_t *codes, const uint8_t *q, float d,
                                const int8_t *xq, const float *dx, int I) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int j = 0; j < I / 32; j++) {
        const int8_t *xb = xq + (size_t)j * 32;
        const float sc = d * dx[j];
        const int qs = vq1_scale(q, j / 2);
        for (int l = 0; l < 8; l++) {
            int32_t s = 0;
            for (int t = 4 * l; t < 4 * l + 4; t++) s += vq1_value(bk, codes, j * 32 + t) * xb[t];
            acc[l] = fmaf((float)(s * qs), sc, acc[l]);
        }
    }
    return x2_hsum8_scalar(acc);
}

#ifdef X2_HAVE_AVX2
#define VQ1_HAVE_AVX2 1
static inline __attribute__((always_inline)) __m256i vq1_pair(const Vq1Books *bk, int a0, int b0, int a1, int b1) {
    const __m128i v0 = _mm_add_epi8(_mm_loadu_si128((const __m128i *)bk->a[a0]), _mm_loadu_si128((const __m128i *)bk->b[b0]));
    const __m128i v1 = _mm_add_epi8(_mm_loadu_si128((const __m128i *)bk->a[a1]), _mm_loadu_si128((const __m128i *)bk->b[b1]));
    return _mm256_set_m128i(v1, v0);
}
static inline __attribute__((always_inline)) void vq1_dot_body(float *y, int ystride, const Vq1Books *bk,
                                                              const uint8_t *codes, const uint8_t *q, float d,
                                                              const int8_t *xq, const float *dx, int I, const int S) {
    const int nb = I / 32;
    __m256 acc[X2_MAX_ROWS];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    for (int g = 0; g < I / 64; g++) {
        int idx[8]; vq1_indices(codes, g, idx);
        const __m256i qv = _mm256_set1_epi16((short)vq1_scale(q, g));
        for (int h = 0; h < 2; h++) {
            const int j = 2 * g + h;
            const __m256i w = vq1_pair(bk, idx[4 * h], idx[4 * h + 1], idx[4 * h + 2], idx[4 * h + 3]);
            const __m256i aw = _mm256_sign_epi8(w, w);
            for (int s = 0; s < S; s++) {
                const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (size_t)s * I + (size_t)j * 32));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)), qv);
                acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[(size_t)s * nb + j]), acc[s]);
            }
        }
    }
    for (int s = 0; s < S; s++) y[(size_t)s * ystride] = x2_hsum8(acc[s]);
}
#endif

/* S (1..X2_MAX_ROWS) activation rows of one weight row; xq/dx row-major [S][I], [S][I/32] */
static inline void vq1_dot(float *y, int ystride, const Vq1Books *bk, const uint8_t *codes, const uint8_t *q, float d,
                           const int8_t *xq, const float *dx, int I, int S) {
#ifdef VQ1_HAVE_AVX2
    switch (S) {
    case 1: vq1_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 1); break;
    case 2: vq1_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 2); break;
    case 3: vq1_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 3); break;
    default: vq1_dot_body(y, ystride, bk, codes, q, d, xq, dx, I, 4); break;
    }
#else
    for (int s = 0; s < S; s++)
        y[(size_t)s * ystride] = vq1_dot_ref(bk, codes, q, d, xq + (size_t)s * I, dx + (size_t)s * (I / 32), I);
#endif
}

/* y[s*O + o] = W[o,:] . x[s,:] for a vq1 matrix: codes [O][I/64*10], scales [O][I/128], d [O].
 * Activations quantized once, rows four at a time. 0, or -1 when out of memory. */
static inline int vq1_matmul(float *y, const float *x, const Vq1Books *bk, const uint8_t *codes, const uint8_t *q,
                             const float *d, int S, int I, int O) {
    const size_t cb = vq1_code_bytes(I), qb = vq1_scale_bytes(I); const int nb = I / 32;
    int8_t *xq = (int8_t *)malloc((size_t)S * I); float *dx = (float *)malloc((size_t)S * nb * sizeof(float));
    if (!xq || !dx) { free(xq); free(dx); return -1; }
    for (int s = 0; s < S; s++) x2_quant_act(x + (size_t)s * I, I, xq + (size_t)s * I, dx + (size_t)s * nb);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        for (int s0 = 0; s0 < S; s0 += X2_MAX_ROWS) {
            const int n = S - s0 < X2_MAX_ROWS ? S - s0 : X2_MAX_ROWS;
            vq1_dot(y + (size_t)s0 * O + o, O, bk, codes + (size_t)o * cb, q + (size_t)o * qb, d[o],
                    xq + (size_t)s0 * I, dx + (size_t)s0 * nb, I, n);
        }
    free(xq); free(dx);
    return 0;
}

#endif /* COLI_VQ1_G64_H */
