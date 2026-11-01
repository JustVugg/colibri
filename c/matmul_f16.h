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

#if defined(__F16C__) && defined(__AVX__)
#include <immintrin.h>
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

static void matmul_bf16w(float *y, const float *x, const uint16_t *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++){
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = dot_bf16w_lanes(x + (int64_t)s * I, w, I);
    }
}

#endif
