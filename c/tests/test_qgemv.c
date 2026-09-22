/* ============================================================================
 * test_qgemv.c — standalone unit tests for the plain (non-group-scaled) int8
 * GEMV (qgemv.h: matmul_q) that qwen36 runs for lm_head and any non-expert
 * quantized matmul. No model, no weights: links the SAME qgemv.h the engine
 * links.
 *
 * qgemv.h has an AVX2/FMA tier, an SSE4.1 tier (routed through
 * sse41_kernels.h) and a scalar tier (plus an ARM NEON/IDOT tier, dead code
 * on this x86 test build). The AVX2 and scalar tiers must keep their exact
 * sequence of float operations — float addition is not associative, and this
 * engine's token stream is required to stay byte-identical to the reference.
 * reference_q_gemv below is a VERBATIM copy of the kernel as it shipped
 * before the SSE4.1 tier was added, and the AVX2/scalar properties compare
 * the live kernel against it with memcmp on raw float bits. The SSE4.1 tier
 * has no pre-existing byte-identical output to match — its reduction tree is
 * a genuinely different shape — so it alone is checked by max relative error
 * instead.
 *
 * P1 DENSE SHAPES — the two shapes the engine actually runs (I=2048,O=512 and
 *    I=512,O=2048) are bit-identical.
 * P2 ROW COUNT EDGE CASES — O values around and below the OpenMP
 *    parallel-for threshold (O=515, O=6, O=1) are bit-identical. matmul_q has
 *    no row-interleaving, so this only exercises the scheduling boundary,
 *    not an unroll tail.
 * P4 I TAIL — I not a multiple of the unroll width (32 for AVX2, 16 for
 *    SSE4.1) leaves a final chunk that the unconditional scalar tail loop
 *    must still fold in correctly.
 * P5 SHORT I TAIL — a final chunk of just a few elements. Unlike
 *    matmul_q_gs's AVX2 tier, matmul_q's tail loop never drops a remainder,
 *    so this must match exactly, not reproduce a dropped-value quirk.
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

#include "../qgemv.h"

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
 * VERBATIM copy of matmul_q as it shipped before the SSE4.1 tier was added.
 * Do not tidy, reformat or "improve" this: its value is that it is the old
 * arithmetic, character for character. If this drifts, the test proves nothing.
 * ------------------------------------------------------------------------- */
static void reference_q_gemv(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__AVX2__) && defined(__FMA__)
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

/* Fills one random case, runs both kernels, compares the results.
 * The AVX2 and scalar tiers stay exact: memcmp on raw float bits, per the
 * engine's byte-identical requirement. The SSE4.1 tier's tree reduction is
 * not bit-reproducible against the scalar reference -- float addition is
 * not associative -- so that tier alone is checked by max relative error. */
static int rows_match(int I, int O) {
    float  *x   = malloc((size_t)I * sizeof *x);
    int8_t *q   = malloc((size_t)O * I);
    float  *sc  = malloc((size_t)O * sizeof *sc);
    float  *y   = malloc((size_t)O * sizeof *y);
    float  *ref = malloc((size_t)O * sizeof *ref);
    if (!x || !q || !sc || !y || !ref) { fprintf(stderr, "out of memory\n"); exit(2); }

    for (int i = 0; i < I; i++) x[i] = rng_f32();
    for (int64_t i = 0; i < (int64_t)O * I; i++) q[i] = (int8_t)(rng_next() >> 24);
    for (int o = 0; o < O; o++) sc[o] = rng_f32() * 0.01f;

    /* Poison both outputs differently: a kernel that skips a row must not pass
     * by leaving stale bytes that happen to agree. */
    memset(y, 0x5A, (size_t)O * sizeof *y);
    memset(ref, 0xA5, (size_t)O * sizeof *ref);

    reference_q_gemv(ref, x, q, sc, I, O);
    matmul_q(y, x, q, sc, I, O);

#if defined(__SSE4_1__) && !(defined(__AVX2__) && defined(__FMA__))
    /* Combined absolute+relative tolerance (numpy allclose-style), not pure
     * relative error: a tree-reduced SSE4.1 sum and a sequential scalar sum
     * disagree by float-rounding noise, and when a reference output happens
     * to land near zero that noise alone blows the relative error past any
     * pure-relative threshold. Measured worst-case noise across all shapes
     * below is ~4.2e-6 absolute; the 1e-5 floor gives headroom above that
     * without loosening the check for the far more common non-near-zero
     * outputs, where 1e-4 relative still rules. */
    int equal = 1;
    for (int o = 0; o < O; o++) {
        float diff  = fabsf(y[o] - ref[o]);
        float denom = fabsf(ref[o]);
        if (diff > 1e-5f + 1e-4f * denom) { equal = 0; break; }
    }
#else
    int equal = memcmp(y, ref, (size_t)O * sizeof *y) == 0;
#endif
    free(x); free(q); free(sc); free(y); free(ref);
    return equal;
}

int main(void) {
    printf("test_qgemv: plain int8 GEMV matches its reference\n");

    /* ---- P1: the shapes the engine actually runs ------------------------- */
    CHECK(rows_match(2048, 512),  "P1 lm_head-like shape (I=2048, O=512) matches");
    CHECK(rows_match(512, 2048),  "P1 dense shape (I=512, O=2048) matches");

    /* ---- P2: row count around the OpenMP parallel-for threshold ---------- */
    CHECK(rows_match(2048, 515),  "P2 O=515 (just over the parallel threshold) matches");
    CHECK(rows_match(2048, 6),    "P2 O=6 (few rows) matches");
    CHECK(rows_match(2048, 1),    "P2 O=1 (single row) matches");

    /* ---- P4: I not a multiple of the unroll width leaves a final chunk --- */
    CHECK(rows_match(2000, 512),  "P4 I=2000 (final chunk of 16) matches");

    /* ---- P5: a short final chunk, folded in by the unconditional tail ---- */
    CHECK(rows_match(1990, 512),  "P5 I=1990 (final chunk of 6) matches");
    CHECK(rows_match(1, 4),       "P5 I=1 (no full unrolled chunk at all) matches");

    printf("\n%s (%d failure%s)\n", fails ? "TEST FAIL" : "ALL PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
