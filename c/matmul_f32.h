/* matmul_f32.h — y[S,O] = x[S,I] @ W^T with W[O,I] f32 (header-only, static).
 * One kernel for every engine that keeps f32 matrices: quant.h (colibri, glm53,
 * kimi_k3, deepseek_v4/v41, qwen38) and olmoe, inkling and qwen36, which used
 * to carry their own copy of the loop (#442). */
#ifndef COLI_MATMUL_F32_H
#define COLI_MATMUL_F32_H

#include <math.h>
#include <stdint.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

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
#if defined(__AVX2__) && defined(__FMA__)
    if (S >= 2 && I % MATMUL_F32_LANES == 0 && MATMUL_F32_LANES == 32) {
        /* Several rows (a prompt step): 16 outputs a task, their rows 16 at a time in L2
         * (the loop below rereads every row for every output), each 32 weights loaded
         * once for 4 rows. A row's 32 lanes are dot_f32_lanes': the same fused adds in
         * order, then its halving tree (16, 8, 4, 2, 1 lanes apart) -- the same bytes. */
        const int nob = (O + 15) / 16;
        #pragma omp parallel for schedule(static)
        for (int ob = 0; ob < nob; ob++) {
            const int oe = ob * 16 + 16 < O ? ob * 16 + 16 : O;
            for (int s0 = 0; s0 < S; s0 += 16) {
                const int se = s0 + 16 < S ? s0 + 16 : S;
                for (int o = ob * 16; o < oe; o++) {
                    const float *w = W + (int64_t)o * I;
                    for (int s1 = s0; s1 < se; s1 += 4) {
                        const int ns = se - s1 < 4 ? se - s1 : 4;
                        const float *xr[4];
                        for (int r = 0; r < 4; r++) xr[r] = x + (int64_t)(s1 + (r < ns ? r : ns - 1)) * I;
                        __m256 a[4][4];
                        for (int r = 0; r < 4; r++) for (int q = 0; q < 4; q++) a[r][q] = _mm256_setzero_ps();
                        for (int i = 0; i < I; i += 32) {
                            __m256 wq[4];
                            for (int q = 0; q < 4; q++) wq[q] = _mm256_loadu_ps(w + i + 8 * q);
                            for (int r = 0; r < 4; r++)
                                for (int q = 0; q < 4; q++)
                                    a[r][q] = _mm256_fmadd_ps(_mm256_loadu_ps(xr[r] + i + 8 * q), wq[q], a[r][q]);
                        }
                        for (int r = 0; r < ns; r++) {
                            __m256 h0 = _mm256_add_ps(a[r][0], a[r][2]), h1 = _mm256_add_ps(a[r][1], a[r][3]);
                            h0 = _mm256_add_ps(h0, h1);
                            __m128 l = _mm_add_ps(_mm256_castps256_ps128(h0), _mm256_extractf128_ps(h0, 1));
                            l = _mm_add_ps(l, _mm_movehl_ps(l, l));
                            l = _mm_add_ss(l, _mm_shuffle_ps(l, l, 1));
                            y[(int64_t)(s1 + r) * O + o] = _mm_cvtss_f32(l);
                        }
                    }
                }
            }
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++){
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = dot_f32_lanes(x + (int64_t)s * I, w, I);
    }
}

#endif
