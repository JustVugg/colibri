/* int2_g32.h — colibri's 2-bit routed-expert format ("int2-g32") and its kernels.
 *
 *   layout  a row of I weights (I % 128 == 0) is I/128 blocks of 32 bytes; byte k
 *           of a block holds the 2-bit codes of elements k, k+32, k+64, k+96 in
 *           bits 0-1, 2-3, 4-5, 6-7. A shift by 0, 2, 4 or 6 and `and 3` yield
 *           the block's four 32-element groups already in order: no shuffle, no
 *           blend (ggml's Q2_0 interleaves element 4k+r into byte k and pays a
 *           shuffle and three blends per 32 elements to undo it).
 *   values  code c -> level X2_LEVELS[c] = -10, -3, 3, 10: the four-level
 *           Lloyd-Max grid of a bell-shaped weight (+-0.453 and +-1.510 sigma) in
 *           integers. Fitted per layer and matrix on Qwen3.8's experts the levels
 *           came out the same to three decimals, so the grid is fixed and one
 *           pshufb table decodes it. No level is zero: Q2_0's grid (-1, 0, 1, 2)
 *           spends a code on zero (41% of the codes in a released Q2_0 file) and
 *           flips the sign of half its scales to make up for the asymmetry. On
 *           24 Qwen3.8 experts at the same 2.25 bits, relative L2 error 0.337
 *           here against 0.444 for that file's blocks.
 *   scales  one u8 per (row, 32-group) under one f32 per row: a weight is
 *           d[row] * q[row][g] * level. A row's group scales span 0.38..1 of
 *           its largest, which 8 linear bits cover with no measurable loss
 *           against f32 scales (0.3369 against 0.3369).
 *   size    2 + 8/32 = 2.25 bits per weight, plus 32 bits per row.
 *   x       the activation contract of ggml_quants.h (gq_quant_act): int8 per
 *           32 elements with one f32 scale each. Per group the integer dot is
 *           exact; then one f32 fma per group and lane with d*q*dx.
 *
 * The scalar reference defines the numerics: lane l of a group sums its
 * elements 4l..4l+3 as integers, each lane accumulates fmaf(lane, d*q*dx)
 * over the groups in order, and the 8 lanes are summed as GQ_HSUM8 does.
 * The AVX2 kernels reproduce it bit for bit, for every S (rows do not share
 * accumulators). Header-only, all static, pure compute. */
#ifndef COLI_INT2_G32_H
#define COLI_INT2_G32_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define X2_GROUP 32                    /* elements per scale and per activation block */
#define X2_BLOCK 128                   /* elements per 32-byte code block */
#define X2_BLOCK_BYTES 32
#define X2_MAX_ROWS 4                  /* activation rows one call takes */

static const int8_t X2_LEVELS[4] = { -10, -3, 3, 10 };

static inline int    x2_layout_ok(int I)  { return I > 0 && I % X2_BLOCK == 0; }
static inline size_t x2_code_bytes(int I) { return (size_t)I / 4; }
static inline int    x2_groups(int I)     { return I / X2_GROUP; }

/* code of element i of a row (tests, packing checks) */
static inline int x2_code(const uint8_t *codes, int i) {
    return (codes[(size_t)(i / X2_BLOCK) * X2_BLOCK_BYTES + i % 32] >> (2 * ((i % X2_BLOCK) / 32))) & 3;
}

static inline float x2_hsum8_scalar(const float *l) {   /* GQ_HSUM8's order */
    float h0 = l[0] + l[4], h1 = l[1] + l[5], h2 = l[2] + l[6], h3 = l[3] + l[7];
    return (h0 + h2) + (h1 + h3);
}

/* y = sum_i w[i] * x[i] for one weight row and one activation row (S = 1) */
static inline float x2_dot_ref(const uint8_t *codes, const uint8_t *q, float d,
                               const int8_t *xq, const float *dx, int I) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int g = 0; g < I / X2_GROUP; g++) {
        const uint8_t *blk = codes + (size_t)(g / 4) * X2_BLOCK_BYTES;
        const int sh = 2 * (g % 4);
        const int8_t *xb = xq + (size_t)g * X2_GROUP;
        const float sc = d * (float)q[g] * dx[g];
        for (int l = 0; l < 8; l++) {
            int32_t s = 0;
            for (int j = 4 * l; j < 4 * l + 4; j++) s += X2_LEVELS[(blk[j] >> sh) & 3] * xb[j];
            acc[l] = fmaf((float)s, sc, acc[l]);
        }
    }
    return x2_hsum8_scalar(acc);
}

#if defined(__AVX2__) && defined(__FMA__)
#define X2_HAVE_AVX2 1
static inline float x2_hsum8(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}
/* S activation rows (constant at each call site) of one weight row: y[s * ystride].
 * A group is decoded once (shift, mask, pshufb) and multiplied into every row with
 * the sign trick: |w| (<= 10) unsigned times sign(x, w), pairs summed in int16
 * (<= 2540) and the pairs in int32 lanes, which is lane l = elements 4l..4l+3. */
static inline __attribute__((always_inline)) void x2_dot_body(float *y, int ystride, const uint8_t *codes, const uint8_t *q, float d,
                                                             const int8_t *xq, const float *dx, int I, const int S) {
    const __m256i lut = _mm256_setr_epi8(-10, -3, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         -10, -3, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    const int ng = I / X2_GROUP;
    __m256 acc[X2_MAX_ROWS];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    for (int b = 0; b < I / X2_BLOCK; b++) {
        const __m256i bb = _mm256_loadu_si256((const __m256i *)(codes + (size_t)b * X2_BLOCK_BYTES));
        for (int h = 0; h < 4; h++) {
            const int g = 4 * b + h;
            const __m256i c = _mm256_and_si256(h == 0 ? bb : _mm256_srli_epi16(bb, 2 * h), m3);
            const __m256i w = _mm256_shuffle_epi8(lut, c), aw = _mm256_sign_epi8(w, w);
            const float wq = d * (float)q[g];
            for (int s = 0; s < S; s++) {
                const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (size_t)s * I + (size_t)g * X2_GROUP));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)), ones);
                acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(wq * dx[(size_t)s * ng + g]), acc[s]);
            }
        }
    }
    for (int s = 0; s < S; s++) y[(size_t)s * ystride] = x2_hsum8(acc[s]);
}
#endif

/* S (1..X2_MAX_ROWS) activation rows of one weight row; xq/dx row-major [S][I], [S][I/32] */
static inline void x2_dot(float *y, int ystride, const uint8_t *codes, const uint8_t *q, float d,
                          const int8_t *xq, const float *dx, int I, int S) {
#ifdef X2_HAVE_AVX2
    switch (S) {
    case 1: x2_dot_body(y, ystride, codes, q, d, xq, dx, I, 1); break;
    case 2: x2_dot_body(y, ystride, codes, q, d, xq, dx, I, 2); break;
    case 3: x2_dot_body(y, ystride, codes, q, d, xq, dx, I, 3); break;
    default: x2_dot_body(y, ystride, codes, q, d, xq, dx, I, 4); break;
    }
#else
    for (int s = 0; s < S; s++)
        y[(size_t)s * ystride] = x2_dot_ref(codes, q, d, xq + (size_t)s * I, dx + (size_t)s * (I / X2_GROUP), I);
#endif
}

/* The activation contract: int8 per 32 elements, one f32 scale (amax/127) each,
 * rounded to nearest; the arithmetic of ggml_quants.h's gq_quant_act. */
static inline void x2_quant_act_ref(const float *x, int I, int8_t *xq, float *dx) {
    for (int b = 0; b < I / X2_GROUP; b++) {
        const float *v = x + (size_t)b * X2_GROUP; float amax = 0.f;
        for (int i = 0; i < X2_GROUP; i++) { float a = fabsf(v[i]); if (a > amax) amax = a; }
        const float id = amax > 0.f ? 127.f / amax : 0.f;
        dx[b] = amax / 127.f;
        for (int i = 0; i < X2_GROUP; i++) {
            int t = (int)lrintf(v[i] * id); if (t > 127) t = 127; if (t < -127) t = -127;
            xq[(size_t)b * X2_GROUP + i] = (int8_t)t;
        }
    }
}
/* x2_quant_act_ref eight lanes at a time, byte for byte (see gq_quant_block_avx2) */
static inline void x2_quant_act(const float *x, int I, int8_t *xq, float *dx) {
#if defined(__AVX2__)
    const __m256 sign = _mm256_set1_ps(-0.f);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (int b = 0; b < I / X2_GROUP; b++) {
        const float *v = x + (size_t)b * X2_GROUP;
        const __m256 v0 = _mm256_loadu_ps(v), v1 = _mm256_loadu_ps(v + 8), v2 = _mm256_loadu_ps(v + 16), v3 = _mm256_loadu_ps(v + 24);
        const __m256 a = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                       _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m)); m = _mm_max_ss(m, _mm_movehdup_ps(m));
        const float amax = _mm_cvtss_f32(m);
        const float id = amax > 0.f ? 127.f / amax : 0.f;
        dx[b] = amax / 127.f;
        const __m256 idv = _mm256_set1_ps(id);
        const __m256i q0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v0, idv)), lo), hi);
        const __m256i q1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v1, idv)), lo), hi);
        const __m256i q2 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v2, idv)), lo), hi);
        const __m256i q3 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v3, idv)), lo), hi);
        const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(q0, q1), _mm256_packs_epi32(q2, q3));
        _mm256_storeu_si256((__m256i *)(xq + (size_t)b * X2_GROUP), _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
    }
#else
    x2_quant_act_ref(x, I, xq, dx);
#endif
}

/* y[s*O + o] = W[o,:] . x[s,:] for an int2-g32 matrix: codes [O][I/4], q [O][I/32],
 * d [O]. The activations are quantized once; every output row takes them four at a
 * time, so a row's bits do not depend on S. 0, or -1 when out of memory. */
static inline int x2_matmul(float *y, const float *x, const uint8_t *codes, const uint8_t *q, const float *d,
                            int S, int I, int O) {
    const size_t cb = x2_code_bytes(I); const int ng = x2_groups(I);
    int8_t *xq = (int8_t *)malloc((size_t)S * I); float *dx = (float *)malloc((size_t)S * ng * sizeof(float));
    if (!xq || !dx) { free(xq); free(dx); return -1; }
    for (int s = 0; s < S; s++) x2_quant_act(x + (size_t)s * I, I, xq + (size_t)s * I, dx + (size_t)s * ng);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        for (int s0 = 0; s0 < S; s0 += X2_MAX_ROWS) {
            const int n = S - s0 < X2_MAX_ROWS ? S - s0 : X2_MAX_ROWS;
            x2_dot(y + (size_t)s0 * O + o, O, codes + (size_t)o * cb, q + (size_t)o * ng, d[o],
                   xq + (size_t)s0 * I, dx + (size_t)s0 * ng, I, n);
        }
    free(xq); free(dx);
    return 0;
}

#endif /* COLI_INT2_G32_H */
