/* matmul_f16.h — y[S,O] = x[S,I] @ W^T with W[O,I] stored as IEEE half or BF16 (header-only, static).
 *
 * Same contract as matmul_f32.h: the weight is widened to f32 (exact) and every dot product
 * runs the very same fixed lanes, the same fused multiply-add and the same halving tree as
 * dot_f32_lanes(). The result is therefore bit-identical to matmul() on the widened matrix
 * (tests/test_matmul_f16.c compares the bits), while the matrix costs half the DRAM bytes.
 * A decode step that streams its attention and lm_head weights from memory is
 * bandwidth-bound, so keeping a half checkpoint's tensors half-sized is a straight win.
 *
 * IEEE half needs F16C to widen 8 halves per instruction; without it matmul_f16w is not
 * defined (COLI_MATMUL_F16 stays undefined). BF16 needs nothing: its widening is a 16-bit
 * shift, so matmul_bf16w is defined on every target, ARM included. */
#ifndef COLI_MATMUL_F16_H
#define COLI_MATMUL_F16_H

#include <string.h>
#include "matmul_f32.h"

#if defined(__AVX2__) || (defined(__F16C__) && defined(__AVX__))
#include <immintrin.h>
#endif
#if defined(__F16C__) && defined(__AVX__)
#define COLI_MATMUL_F16 1

static inline float dot_f16w_lanes(const float *x, const uint16_t *w, int I){
    float a[MATMUL_F32_LANES] = {0};
    int i = 0;
    for (; i + MATMUL_F32_LANES <= I; i += MATMUL_F32_LANES){
        float t[MATMUL_F32_LANES];
        for (int j = 0; j < MATMUL_F32_LANES; j += 8)
            _mm256_storeu_ps(t + j, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + j))));
        for (int j = 0; j < MATMUL_F32_LANES; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], t[j]);
    }
    for (int j = 0; i + j < I; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], _cvtsh_ss(w[i+j]));
    for (int h = MATMUL_F32_LANES/2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) a[j] += a[j+h];
    return a[0];
}

static void matmul_f16w(float *y, const float *x, const uint16_t *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++){
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = dot_f16w_lanes(x + (int64_t)s * I, w, I);
    }
}
#endif

static inline float bf16w_to_f32(uint16_t h){
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, sizeof f); return f;
}

#if defined(__AVX2__)
static inline float dot_bf16w_lanes(const float *x, const uint16_t *w, int I){
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= I; i += 32) {
        __m128i h0 = _mm_loadu_si128((const __m128i *)(w + i));
        __m128i h1 = _mm_loadu_si128((const __m128i *)(w + i + 8));
        __m128i h2 = _mm_loadu_si128((const __m128i *)(w + i + 16));
        __m128i h3 = _mm_loadu_si128((const __m128i *)(w + i + 24));
        __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16));
        __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        __m256 w2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h2), 16));
        __m256 w3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h3), 16));
#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),      w0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8),  w1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), w2, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), w3, acc3);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(x + i),      w0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(x + i + 8),  w1));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(x + i + 16), w2));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(x + i + 24), w3));
#endif
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < I; i++) sum += x[i] * bf16w_to_f32(w[i]);
    return sum;
}
#elif defined(__ARM_NEON)
static inline float dot_bf16w_lanes(const float *x, const uint16_t *w, int I){
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 16 <= I; i += 16) {
        uint16x8_t h0 = vld1q_u16(w + i);
        uint16x8_t h1 = vld1q_u16(w + i + 8);
        float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16));
        float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h0), 16));
        float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
        float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
        acc0 = vfmaq_f32(acc0, vld1q_f32(x + i),      w0);
        acc1 = vfmaq_f32(acc1, vld1q_f32(x + i + 4),  w1);
        acc2 = vfmaq_f32(acc2, vld1q_f32(x + i + 8),  w2);
        acc3 = vfmaq_f32(acc3, vld1q_f32(x + i + 12), w3);
    }
    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vaddvq_f32(acc);
    for (; i < I; i++) sum += x[i] * bf16w_to_f32(w[i]);
    return sum;
}
#else
static inline float dot_bf16w_lanes(const float *x, const uint16_t *w, int I){
    float a[MATMUL_F32_LANES] = {0};
    int i = 0;
    for (; i + MATMUL_F32_LANES <= I; i += MATMUL_F32_LANES)
        for (int j = 0; j < MATMUL_F32_LANES; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], bf16w_to_f32(w[i+j]));
    for (int j = 0; i + j < I; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], bf16w_to_f32(w[i+j]));
    for (int h = MATMUL_F32_LANES/2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) a[j] += a[j+h];
    return a[0];
}
#endif

static void matmul_bf16w(float *y, const float *x, const uint16_t *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++){
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = dot_bf16w_lanes(x + (int64_t)s * I, w, I);
    }
}

#endif
