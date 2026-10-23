/* matmul_f32.h — y[S,O] = x[S,I] @ W^T with W[O,I] f32 (header-only, static).
 * One kernel for every engine that keeps f32 matrices: quant.h (colibri, glm53,
 * kimi_k3, deepseek_v4/v41, qwen38) and olmoe, inkling and qwen36, which used
 * to carry their own copy of the loop (#442). */
#ifndef COLI_MATMUL_F32_H
#define COLI_MATMUL_F32_H

#include <math.h>
#include <stdint.h>

/* Each dot product runs in MATMUL_F32_LANES fixed lanes: lane j accumulates
 * the elements i == j (mod lanes) in order, then the lanes are summed in a
 * fixed halving tree. Every lane is its own chain, so the compiler vectorizes
 * the loop without reassociating anything, and the result does not depend on
 * the vector width: NEON runs 8 chains of 4, AVX2 4 of 8, AVX-512 2 of 16, and
 * all of them produce the same bits.
 *
 * The one-chain loops this replaces were either scalar (`a+=x*w`, which has
 * toolchain-dependent bits: gcc and Apple clang keep the adds unfused, x86
 * clang fuses them) or `#pragma omp simd reduction`, which lets the compiler
 * pick the order, so the bits change with the target. */
#define MATMUL_F32_LANES 32

#if defined(__FMA__) || defined(__ARM_FEATURE_FMA)
#define MATMUL_F32_MADD(a, x, w) fmaf((x), (w), (a))
#else
/* No hardware FMA: fmaf would be a libm call. The lanes still fix the order. */
#define MATMUL_F32_MADD(a, x, w) ((a) + (x) * (w))
#endif

static inline float dot_f32_lanes(const float *x, const float *w, int I){
    float a[MATMUL_F32_LANES] = {0};
    int i = 0;
    for (; i + MATMUL_F32_LANES <= I; i += MATMUL_F32_LANES)
        for (int j = 0; j < MATMUL_F32_LANES; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], w[i+j]);
    for (int j = 0; i + j < I; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], w[i+j]);
    for (int h = MATMUL_F32_LANES/2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) a[j] += a[j+h];
    return a[0];
}

static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++){
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = dot_f32_lanes(x + (int64_t)s * I, w, I);
    }
}

#endif
