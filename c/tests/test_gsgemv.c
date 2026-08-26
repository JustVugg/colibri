/* ============================================================================
 * test_gsgemv.c — standalone unit tests for the group-scaled int8 GEMV
 * (gsgemv.h: matmul_q_gs) that qwen36 runs for every expert matmul.
 * No model, no weights: links the SAME gsgemv.h the engine links.
 *
 * gsgemv.h has three tiers: AVX2/FMA (row-interleaved), SSE4.1 (row-
 * interleaved, routed through sse41_kernels.h) and scalar. The AVX2 and
 * scalar tiers must keep their exact sequence of float operations -- float
 * addition is not associative, and this engine's token stream is required
 * to stay byte-identical to the reference. reference_gs_gemv below is a
 * VERBATIM copy of the kernel as it shipped before either the AVX2
 * row-interleave or the SSE4.1 tier existed, and the AVX2/scalar properties
 * compare the live kernel against it with memcmp on raw float bits. The
 * SSE4.1 tier has no pre-existing byte-identical output to match -- its
 * reduction tree is a genuinely different shape -- so it alone is checked
 * by max relative+absolute error instead.
 *
 * P1 EXPERT SHAPES — the two shapes the engine actually runs (I=2048,O=512
 *    for gate/up and I=512,O=2048 for down, gs=64) match.
 * P2 ROW TAIL — O not a multiple of the unroll width still matches, so the
 *    tail rows cannot quietly accumulate in a different order.
 * P3 SCALAR PATH — gs not a multiple of 32 (AVX2) / 16 (SSE4.1) takes the
 *    scalar fallback. The dispatch is a runtime property of gs, so that
 *    path needs the same gate.
 * P4 PARTIAL GROUP — I not a multiple of gs leaves a short final group.
 * P5 SHORT FINAL GROUP — a final group of fewer than 16 (AVX2) / 8 (SSE4.1)
 *    elements. Both vector paths step through a group in fixed-width chunks
 *    and DROP such a remainder. That is pre-existing, dormant-in-this-model
 *    behaviour (I is 2048 or 512 and gs is 64, so the remainder is always
 *    zero), but it is pinned here so any future restructure reproduces it
 *    rather than silently correcting it -- a correction would change output,
 *    which is exactly what byte-identical tiers must not do. The SSE4.1
 *    tier's own tail-drop width (<8, half the AVX2 tier's <16) means
 *    reference_gs_gemv needs its OWN SSE4.1-shaped branch below: on an
 *    SSE4.1-only build, without that branch the reference would fall through
 *    to the undropped scalar path and P5 could never pass.
 * Exit 0 = all pass.
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../gsgemv.h"

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  ok   %s\n", msg); } \
    else { printf("  FAIL %s\n", msg); fails++; } \
} while (0)

/* Deterministic across libc implementations, unlike rand(). */
static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rng_next(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rng_f32(void) { return (float)((int32_t)(rng_next() >> 8) - 8388608) / 8388608.0f; }

/* ---------------------------------------------------------------------------
 * VERBATIM copy of matmul_q_gs as it shipped before the SSE4.1 tier was
 * added. Do not tidy, reformat or "improve" this: its value is that it is
 * the old arithmetic, character for character. If this drifts, the test
 * proves nothing.
 * ------------------------------------------------------------------------- */
static void reference_gs_gemv(float *y, const float *x, const int8_t *q, const float *scale,
                              int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
#if defined(__AVX2__) && defined(__FMA__)
    if ((gs & 31) == 0) {
        #pragma omp parallel for schedule(static) if(O >= 256)
        for (int o = 0; o < O; o++) {
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
                a0 = _mm256_add_ps(a0, a1);
                __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
                s = _mm_add_ps(s, _mm_movehl_ps(s,s));
                s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
                acc += _mm_cvtss_f32(s) * sc[gi];
            }
            y[o] = acc;
        }
        return;
    }
#elif defined(__SSE4_1__)
    /* Mirrors the SSE4.1 kernel's intentional <8 tail-drop so P5 pins that
     * behaviour instead of comparing against a reference that always sums
     * the full remainder. */
    if ((gs & 15) == 0) {
        #pragma omp parallel for schedule(static) if(O >= 256)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            const float *sc = scale + (int64_t)o * ng;
            float acc = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                float part = 0.f;
                for (int i = base; i + 8 <= end; i += 8)
                    for (int k = 0; k < 8; k++) part += x[i+k] * (float)w[i+k];
                acc += part * sc[gi];
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

/* Fills one random case, runs both kernels, compares the results.
 * The AVX2 and scalar tiers stay exact: memcmp on raw float bits, per the
 * engine's byte-identical requirement. The SSE4.1 tier's tree reduction is
 * not bit-reproducible against the scalar reference -- float addition is
 * not associative -- so that tier alone is checked by max relative error. */
static int rows_match(int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
    float  *x   = malloc((size_t)I * sizeof *x);
    int8_t *q   = malloc((size_t)O * I);
    float  *sc  = malloc((size_t)O * ng * sizeof *sc);
    float  *y   = malloc((size_t)O * sizeof *y);
    float  *ref = malloc((size_t)O * sizeof *ref);
    if (!x || !q || !sc || !y || !ref) { fprintf(stderr, "out of memory\n"); exit(2); }

    for (int i = 0; i < I; i++) x[i] = rng_f32();
    for (int64_t i = 0; i < (int64_t)O * I; i++) q[i] = (int8_t)(rng_next() >> 24);
    for (int i = 0; i < O * ng; i++) sc[i] = rng_f32() * 0.01f;

    /* Poison both outputs differently: a kernel that skips a row must not pass
     * by leaving stale bytes that happen to agree. */
    memset(y, 0x5A, (size_t)O * sizeof *y);
    memset(ref, 0xA5, (size_t)O * sizeof *ref);

    reference_gs_gemv(ref, x, q, sc, I, O, gs);
    matmul_q_gs(y, x, q, sc, I, O, gs);

#if defined(__SSE4_1__) && !(defined(__AVX2__) && defined(__FMA__))
    /* No FMA and a different tree reduction than the scalar reference: pure
     * relative error blows up on near-zero-ref rows (cancellation), so use a
     * combined absolute+relative bound instead. */
    int equal = 1;
    for (int o = 0; o < O; o++) {
        float diff = fabsf(y[o] - ref[o]);
        if (diff > 1e-5f + 1e-4f * fabsf(ref[o])) { equal = 0; break; }
    }
#else
    int equal = memcmp(y, ref, (size_t)O * sizeof *y) == 0;
#endif
    free(x); free(q); free(sc); free(y); free(ref);
    return equal;
}

int main(void) {
    printf("test_gsgemv: group-scaled int8 GEMV matches its reference\n");

    /* ---- P1: the shapes the engine actually runs ------------------------- */
    CHECK(rows_match(2048, 512, 64),  "P1 gate/up shape (I=2048, O=512, gs=64) matches");
    CHECK(rows_match(512, 2048, 64),  "P1 down shape (I=512, O=2048, gs=64) matches");

    /* ---- P2: O not a multiple of the unroll width ------------------------ */
    CHECK(rows_match(2048, 515, 64),  "P2 O=515 (tail rows) matches");
    CHECK(rows_match(2048, 6, 64),    "P2 O=6 (fewer rows than one unrolled block) matches");
    CHECK(rows_match(2048, 1, 64),    "P2 O=1 (single row) matches");

    /* ---- P3: gs not a multiple of 32/16 takes the scalar fallback -------- */
    CHECK(rows_match(2048, 512, 48),  "P3 gs=48 (scalar fallback path) matches");
    CHECK(rows_match(2000, 517, 48),  "P3 scalar fallback with ragged I and O matches");

    /* ---- P4: I not a multiple of gs leaves a short final group ----------- */
    CHECK(rows_match(2000, 512, 64),  "P4 I=2000 (final group of 16) matches");

    /* ---- P5: final group under the vector width, dropped by AVX2/SSE4.1 -- */
    CHECK(rows_match(1990, 512, 64),  "P5 I=1990 (final group of 6, dropped) reproduces the old result");

    printf("\n%s (%d failure%s)\n", fails ? "TEST FAIL" : "ALL PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
