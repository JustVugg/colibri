#ifndef COLIBRI_GSGEMV_H
#define COLIBRI_GSGEMV_H
/* Group-scaled int8 GEMV: one f32 scale per `gs` input elements per row, the
 * layout the gs64 expert containers use.  Row layout of `scale`: [O][I/gs]
 * row-major.  Its own header so tests/test_gsgemv.c can link the very kernel
 * the engine runs; qwen36.c carries a main and cannot be linked into a test.
 *
 * This is qwen36's hottest kernel: every expert matmul goes through it, three
 * per expert, eight experts, forty layers.  Each row's float operations must
 * stay in exactly this order -- float addition is not associative and the
 * engine's token stream is required to be byte-identical to the reference.
 * tests/test_gsgemv.c holds the pre-restructure kernel verbatim and compares
 * raw float bits, so any reassociation fails there rather than surfacing as
 * drifted text much later.
 *
 * The SSE4.1 tier below is the exception to that byte-identical rule, by
 * design: it is new code with no pre-existing output to match, its tree
 * reduction is a genuinely different shape than the scalar reference, and it
 * is checked by tests/test_gsgemv.c against a tolerance, not memcmp. It
 * routes its FMA and float loads through sse41_kernels.h -- the same shared
 * primitives header olmoe.c uses -- rather than inlining its own copy. */
#include <stdint.h>
#include <math.h>
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__SSE4_1__)
#include <immintrin.h>
#endif
#if defined(__SSE4_1__)
#include "sse41_kernels.h"
#endif

#if defined(__AVX2__) && defined(__FMA__)
/* The group reduction, lifted verbatim so the four-row body and the tail row
 * cannot drift apart. Order of the adds is load-bearing, not stylistic. */
static inline float gs_group_sum(__m256 a0, __m256 a1) {
    a0 = _mm256_add_ps(a0, a1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
    s = _mm_add_ps(s, _mm_movehl_ps(s,s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
    return _mm_cvtss_f32(s);
}
#elif defined(__SSE4_1__)
/* Same reduction as gs_group_sum above, one level shallower: a0/a1 are
 * already the two four-lane halves, so there is no 256->128 fold first. */
static inline float gs_group_sum_sse41(__m128 a0, __m128 a1) {
    a0 = _mm_add_ps(a0, a1);
    a0 = _mm_add_ps(a0, _mm_movehl_ps(a0,a0));
    a0 = _mm_add_ss(a0, _mm_shuffle_ps(a0,a0,1));
    return _mm_cvtss_f32(a0);
}
#endif

static void matmul_q_gs(float *y, const float *x, const int8_t *q, const float *scale,
                        int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
#if defined(__AVX2__) && defined(__FMA__)
    if ((gs & 31) == 0) {
        /* Four output rows in flight at once. Each row keeps its own pair of
         * accumulators, its own reduction and its own running acc, so its float
         * operations happen in exactly the order the single-row loop used them
         * -- the interleave is a schedule change, not an algebraic one.
         *
         * Why it pays: one group is 8 FMAs but a dependency chain of roughly 36
         * cycles (accumulate, then the reduction tree, then the loop-carried
         * acc +=), against a throughput floor near 4. The single-row loop stalls
         * on that chain about nine times out of ten. Four independent rows fill
         * the gaps, and the x block gets loaded once instead of four times. */
        int o4 = O & ~3;
        #pragma omp parallel for schedule(static) if(O >= 256)
        for (int ob = 0; ob < o4; ob += 4) {
            const int8_t *w0 = q + (int64_t)(ob+0) * I, *w1 = q + (int64_t)(ob+1) * I;
            const int8_t *w2 = q + (int64_t)(ob+2) * I, *w3 = q + (int64_t)(ob+3) * I;
            const float *sc0 = scale + (int64_t)(ob+0) * ng, *sc1 = scale + (int64_t)(ob+1) * ng;
            const float *sc2 = scale + (int64_t)(ob+2) * ng, *sc3 = scale + (int64_t)(ob+3) * ng;
            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                for (int i = base; i + 16 <= end; i += 16) {
                    __m256 xl = _mm256_loadu_ps(x+i), xh = _mm256_loadu_ps(x+i+8);
                    __m128i b0 = _mm_loadu_si128((const __m128i*)(w0 + i));
                    a00 = _mm256_fmadd_ps(xl, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a00);
                    a01 = _mm256_fmadd_ps(xh, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a01);
                    __m128i b1 = _mm_loadu_si128((const __m128i*)(w1 + i));
                    a10 = _mm256_fmadd_ps(xl, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), a10);
                    a11 = _mm256_fmadd_ps(xh, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8))), a11);
                    __m128i b2 = _mm_loadu_si128((const __m128i*)(w2 + i));
                    a20 = _mm256_fmadd_ps(xl, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b2)), a20);
                    a21 = _mm256_fmadd_ps(xh, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b2,8))), a21);
                    __m128i b3 = _mm_loadu_si128((const __m128i*)(w3 + i));
                    a30 = _mm256_fmadd_ps(xl, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b3)), a30);
                    a31 = _mm256_fmadd_ps(xh, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b3,8))), a31);
                }
                /* fmaf, not `acc += sum * sc`: the shipped single-row loop was
                 * written as a separate multiply and add, but -ffp-contract=fast
                 * fused it, and that fused form is what produced the reference
                 * output. Whether the optimizer also fuses it in this differently
                 * shaped loop is not something to leave to chance -- spelling it
                 * out pins the single rounding the reference depends on. */
                acc0 = fmaf(gs_group_sum(a00, a01), sc0[gi], acc0);
                acc1 = fmaf(gs_group_sum(a10, a11), sc1[gi], acc1);
                acc2 = fmaf(gs_group_sum(a20, a21), sc2[gi], acc2);
                acc3 = fmaf(gs_group_sum(a30, a31), sc3[gi], acc3);
            }
            y[ob+0] = acc0; y[ob+1] = acc1; y[ob+2] = acc2; y[ob+3] = acc3;
        }
        for (int o = o4; o < O; o++) {          /* at most three rows */
            const int8_t *w = q + (int64_t)o * I;
            const float *sc = scale + (int64_t)o * ng;
            float acc = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                for (int i = base; i + 16 <= end; i += 16) {
                    __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a1);
                }
                acc = fmaf(gs_group_sum(a0, a1), sc[gi], acc);
            }
            y[o] = acc;
        }
        return;
    }
#elif defined(__SSE4_1__)
    if ((gs & 15) == 0) {
        /* Same four-row interleave as the AVX2 tier above, sized down to this
         * ISA: each __m128 lane holds 4 lanes instead of 8, so a group step is
         * 8 int8s, not 16 -- the gate halves to 16 and each row keeps two
         * __m128 accumulators instead of two __m256. The multiply-add goes
         * through COLIBRI_FMA (sse41_kernels.h): on real SSE4.1-only hardware
         * (no __FMA__) that macro expands to the same explicit mul-then-add
         * this tier always used, so nothing changes there; it only takes the
         * single-rounded _mm_fmadd_ps path if __FMA__ is somehow defined
         * without __AVX2__, which does not happen on real hardware but keeps
         * the tier correct rather than silently wrong if it ever did. Float
         * loads of `x` go through colibri_sse41_loadu_ps for the same reason
         * olmoe.c does: one definition shared with future consumers instead
         * of a fourth inline copy of _mm_loadu_ps. */
        int o4 = O & ~3;
        #pragma omp parallel for schedule(static) if(O >= 256)
        for (int ob = 0; ob < o4; ob += 4) {
            const int8_t *w0 = q + (int64_t)(ob+0) * I, *w1 = q + (int64_t)(ob+1) * I;
            const int8_t *w2 = q + (int64_t)(ob+2) * I, *w3 = q + (int64_t)(ob+3) * I;
            const float *sc0 = scale + (int64_t)(ob+0) * ng, *sc1 = scale + (int64_t)(ob+1) * ng;
            const float *sc2 = scale + (int64_t)(ob+2) * ng, *sc3 = scale + (int64_t)(ob+3) * ng;
            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                __m128 a00 = _mm_setzero_ps(), a01 = _mm_setzero_ps();
                __m128 a10 = _mm_setzero_ps(), a11 = _mm_setzero_ps();
                __m128 a20 = _mm_setzero_ps(), a21 = _mm_setzero_ps();
                __m128 a30 = _mm_setzero_ps(), a31 = _mm_setzero_ps();
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                for (int i = base; i + 8 <= end; i += 8) {
                    __m128 xl = colibri_sse41_loadu_ps(x+i), xh = colibri_sse41_loadu_ps(x+i+4);
                    __m128i b0 = _mm_loadl_epi64((const __m128i*)(w0 + i));
                    a00 = COLIBRI_FMA(xl, _mm_cvtepi32_ps(_mm_cvtepi8_epi32(b0)), a00);
                    a01 = COLIBRI_FMA(xh, _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(b0,4))), a01);
                    __m128i b1 = _mm_loadl_epi64((const __m128i*)(w1 + i));
                    a10 = COLIBRI_FMA(xl, _mm_cvtepi32_ps(_mm_cvtepi8_epi32(b1)), a10);
                    a11 = COLIBRI_FMA(xh, _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(b1,4))), a11);
                    __m128i b2 = _mm_loadl_epi64((const __m128i*)(w2 + i));
                    a20 = COLIBRI_FMA(xl, _mm_cvtepi32_ps(_mm_cvtepi8_epi32(b2)), a20);
                    a21 = COLIBRI_FMA(xh, _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(b2,4))), a21);
                    __m128i b3 = _mm_loadl_epi64((const __m128i*)(w3 + i));
                    a30 = COLIBRI_FMA(xl, _mm_cvtepi32_ps(_mm_cvtepi8_epi32(b3)), a30);
                    a31 = COLIBRI_FMA(xh, _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(b3,4))), a31);
                }
                acc0 = fmaf(gs_group_sum_sse41(a00, a01), sc0[gi], acc0);
                acc1 = fmaf(gs_group_sum_sse41(a10, a11), sc1[gi], acc1);
                acc2 = fmaf(gs_group_sum_sse41(a20, a21), sc2[gi], acc2);
                acc3 = fmaf(gs_group_sum_sse41(a30, a31), sc3[gi], acc3);
            }
            y[ob+0] = acc0; y[ob+1] = acc1; y[ob+2] = acc2; y[ob+3] = acc3;
        }
        for (int o = o4; o < O; o++) {          /* at most three rows */
            const int8_t *w = q + (int64_t)o * I;
            const float *sc = scale + (int64_t)o * ng;
            float acc = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                for (int i = base; i + 8 <= end; i += 8) {
                    __m128i b0 = _mm_loadl_epi64((const __m128i*)(w + i));
                    a0 = COLIBRI_FMA(colibri_sse41_loadu_ps(x+i),   _mm_cvtepi32_ps(_mm_cvtepi8_epi32(b0)), a0);
                    a1 = COLIBRI_FMA(colibri_sse41_loadu_ps(x+i+4), _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(b0,4))), a1);
                }
                acc = fmaf(gs_group_sum_sse41(a0, a1), sc[gi], acc);
            }
            y[o] = acc;
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        const float *sc = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int gi = 0; gi < ng; gi++) {
            int base = gi * gs, end = base + gs; if (end > I) end = I;
            float part = 0.f;
            for (int i = base; i < end; i++) part += x[i] * (float)w[i];
            acc += part * sc[gi];
        }
        y[o] = acc;
    }
}

#endif /* COLIBRI_GSGEMV_H */
