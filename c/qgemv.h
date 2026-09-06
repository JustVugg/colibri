#ifndef COLIBRI_QGEMV_H
#define COLIBRI_QGEMV_H
/* Plain (non-group-scaled) int8 GEMV: one f32 scale per output row, q[O,I]
 * int8 row-major. Its own header so tests/test_qgemv.c can link the very
 * kernel the engine runs; qwen36.c carries a main and cannot be linked into
 * a test.
 *
 * This is qwen36's dense-projection kernel (lm_head and any non-expert
 * quantized matmul); the group-scaled sibling used for expert matmuls lives
 * in gsgemv.h. Its float operations must stay in exactly this order -- float
 * addition is not associative and the engine's token stream is required to
 * be byte-identical to the reference. tests/test_qgemv.c holds the
 * pre-restructure kernel verbatim and compares raw float bits, so any
 * reassociation fails there rather than surfacing as drifted text much
 * later.
 *
 * The SSE4.1 tier below is the exception to that byte-identical rule, by
 * design: it is new code with no pre-existing output to match, and it is
 * checked by tests/test_qgemv.c against a tolerance, not memcmp. It routes
 * its FMA and float loads through sse41_kernels.h -- the same shared
 * primitives header olmoe.c uses -- rather than inlining its own copy. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__SSE4_1__)
#include <immintrin.h>
#endif
#if defined(__SSE4_1__)
#include "sse41_kernels.h"
#endif

#if defined(__ARM_NEON)
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#endif

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    /* IDOT is opt-in, not default-on: this path quantizes the ACTIVATIONS to
     * Q8_0 per 16-element block, which the scalar path does not, so the two are
     * not numerically equivalent. olmoe shipped it default-on and it cost
     * token-exactness end to end (#1044, fixed in af48fe8 by making it opt-in);
     * qwen36 inherited the same default from the same family of kernels. The
     * tiny-oracle gate would not have caught it -- that job runs on x86. */
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = (e && atoi(e)); }
    if (idot && I % 16 == 0 && I <= 4096) {
        int nb = I / 16; int8_t xi[4096]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*16;
            float am = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b]*(float)dot_i8_16(xi+b*16, w+b*16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    /* Hand-vectorized int8->f32 GEMV (gcc does not auto-vectorize the
     * convert+accumulate chain). 32 weights per iteration, FMA accumulate. */
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 32 <= I; i += 32) {
            __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
            __m128i b1 = _mm_loadu_si128((const __m128i*)(w + i + 16));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8))), a3);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
        s = _mm_add_ps(s, _mm_movehl_ps(s,s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
        float acc = _mm_cvtss_f32(s);
        for (; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#elif defined(__SSE4_1__)
    /* SSE4.1 tier: one rung below the AVX2 branch above, same shape --
     * single output row per iteration, no row-interleaving and no runtime
     * gate (matmul_q has no group scale to gate on, unlike matmul_q_gs's
     * SSE4.1 branch in gsgemv.h). 128-bit lanes instead of 256-bit: four
     * accumulators of 4 floats each, 16 weights per iteration where AVX2
     * manages 32. The multiply-add and the float loads route through
     * sse41_kernels.h (COLIBRI_FMA / colibri_sse41_loadu_ps), the same
     * shared primitives olmoe.c's canary uses, instead of an inline copy of
     * the intrinsics. This tier is new code with no pre-existing
     * byte-identical output to match, so tests/test_qgemv.c checks it
     * against a tolerance, not memcmp -- unlike the AVX2 and scalar tiers
     * above/below. */
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
        __m128 a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
        int i = 0;
        for (; i + 16 <= I; i += 16) {
            __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
            a0 = COLIBRI_FMA(colibri_sse41_loadu_ps(x+i),    _mm_cvtepi32_ps(_mm_cvtepi8_epi32(b0)), a0);
            a1 = COLIBRI_FMA(colibri_sse41_loadu_ps(x+i+4),  _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(b0,4))), a1);
            a2 = COLIBRI_FMA(colibri_sse41_loadu_ps(x+i+8),  _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(b0,8))), a2);
            a3 = COLIBRI_FMA(colibri_sse41_loadu_ps(x+i+12), _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(b0,12))), a3);
        }
        a0 = _mm_add_ps(_mm_add_ps(a0,a1), _mm_add_ps(a2,a3));
        a0 = _mm_add_ps(a0, _mm_movehl_ps(a0,a0));
        a0 = _mm_add_ss(a0, _mm_shuffle_ps(a0,a0,1));
        float acc = _mm_cvtss_f32(a0);
        for (; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#else
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#endif
}

#endif /* COLIBRI_QGEMV_H */
