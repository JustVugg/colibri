/* olmoe matmul_q: the IDOT path must be compared against FP32 ACTIVATIONS,
 * not against an integer reference.
 *
 * This is the test gap behind issue #1044. tests/test_idot.c asserts that the
 * integer dot kernel matches a plain-C integer reference, which it does --
 * exactly, because integer arithmetic has no rounding. But matmul_q's IDOT
 * branch quantizes the ACTIVATIONS to Q8_0 per 16-element block before calling
 * that kernel, and the scalar fallback does not. Comparing int8 against int8
 * can never see that. So a lossy path shipped as the default on every AVX2
 * machine (since 2c4e9de) with a green test suite.
 *
 * What this file asserts:
 *   1. the scalar fallback is EXACT against an fp64 reference;
 *   2. the IDOT path is NOT exact, and its error sits at the magnitude the
 *      activation quantization predicts (~1e-3 relative, not ~1e-7);
 *   3. IDOT is OPT-IN: with the variable unset, matmul_q takes the exact path.
 *
 * (2) is deliberately a two-sided bound. Asserting only "error is small" would
 * pass if someone silently made IDOT exact, and asserting only "error exists"
 * would pass if it got much worse. Both directions are load-bearing: the point
 * is that the two paths DIVERGE and by how much.
 *
 * Build: make -C c tests/test_olmoe_matmul_q
 */
#define OLMOE_TESTING 1
#define main coli_olmoe_main_unused
#include "../olmoe.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t xr(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float frand(void) { return (float)((double)(xr() % 20001) / 10000.0 - 1.0); }

/* Reference in double precision: y[o] = scale[o] * sum_i x[i]*q[o,i].
 * fp64 so that fp32 rounding in the implementation is visible as error rather
 * than being hidden by an equally-rounded reference. */
static void ref_matmul(double *y, const float *x, const int8_t *q,
                       const float *scale, int I, int O) {
    for (int o = 0; o < O; o++) {
        double acc = 0.0;
        const int8_t *w = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) acc += (double)x[i] * (double)w[i];
        y[o] = acc * (double)scale[o];
    }
}

/* Error normalised by the MAGNITUDE OF THE SUMMANDS, not by the result.
 *
 * y[o] is a sum of I signed products that largely cancel: with random
 * activations and random int8 weights the result sits near zero while the
 * individual terms do not. Dividing by |y[o]| then reports catastrophic
 * cancellation in the reference as if it were kernel error -- an early version
 * of this test did exactly that and reported 0.86 relative error for a kernel
 * that is correct to 1e-7. Normalising by sum|x_i * w_i| measures what we
 * actually care about: error relative to the work performed. */
static double max_rel_err(const float *got, const double *want,
                          const float *x, const int8_t *q, const float *scale,
                          int I, int O) {
    double worst = 0.0;
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        double mag = 0.0;
        for (int i = 0; i < I; i++) mag += fabs((double)x[i] * (double)w[i]);
        mag *= fabs((double)scale[o]);
        double d = fabs((double)got[o] - want[o]);
        double r = mag > 1e-12 ? d / mag : d;
        if (r > worst) worst = r;
    }
    return worst;
}

int main(void) {
    const int I = 512, O = 64;   /* I % 16 == 0 and I <= 4096: the IDOT gate */
    int failures = 0;

    float *x = malloc((size_t)I * sizeof *x);
    int8_t *q = malloc((size_t)O * I);
    float *scale = malloc((size_t)O * sizeof *scale);
    float *y = malloc((size_t)O * sizeof *y);
    double *ref = malloc((size_t)O * sizeof *ref);
    if (!x || !q || !scale || !y || !ref) { fprintf(stderr, "oom\n"); return 1; }

    for (int i = 0; i < I; i++) x[i] = frand();
    for (int64_t i = 0; i < (int64_t)O * I; i++) q[i] = (int8_t)((int)(xr() % 256) - 128);
    for (int o = 0; o < O; o++) scale[o] = 0.001f + (float)(xr() % 1000) * 1e-6f;

    ref_matmul(ref, x, q, scale, I, O);

#if !defined(HAVE_FAST_DOT_I8)
    printf("SKIP: no fast int8 dot compiled in; IDOT path unreachable\n");
    printf("      (build with -mavx2 on x86 or on an ARM NEON target)\n");
    free(x); free(q); free(scale); free(y); free(ref);
    return 0;
#else
    /* --- 1. THE SHIPPING DEFAULT must be exact ---------------------------
     * Checked FIRST and with the hook untouched, because matmul_q reads IDOT
     * once into a static. Running this after any forced path would test the
     * cache, not the default. This is the assertion that would have caught
     * issue #1044 the day 2c4e9de landed. */
    unsetenv("IDOT");
    matmul_q(y, x, q, scale, I, O);
    double err_default = max_rel_err(y, ref, x, q, scale, I, O);
    if (err_default < 1e-5) {
        printf("PASS  shipping default (IDOT unset) is exact (max rel %.3e)\n", err_default);
    } else {
        printf("FAIL  shipping default is LOSSY (max rel %.3e).\n", err_default);
        printf("      IDOT must be opt-in: a default that silently quantizes\n");
        printf("      activations is exactly issue #1044.\n");
        failures++;
    }

    /* --- 2. the scalar fallback must be exact when forced ----------------- */
    matmul_q_idot_force = 0;
    matmul_q(y, x, q, scale, I, O);
    double err_scalar = max_rel_err(y, ref, x, q, scale, I, O);
    if (err_scalar < 1e-5) {
        printf("PASS  scalar path exact vs fp64 reference (max rel %.3e)\n", err_scalar);
    } else {
        printf("FAIL  scalar path should be ~exact, got max rel %.3e\n", err_scalar);
        failures++;
    }

    /* --- 3. the IDOT path must diverge, by the predicted magnitude ------- */
    matmul_q_idot_force = 1;
    matmul_q(y, x, q, scale, I, O);
    double err_idot = max_rel_err(y, ref, x, q, scale, I, O);

    /* Q8_0 over a 16-block: the activation is rounded to a 1/127 grid of the
     * block maximum, so relative error lands around 1e-3, orders of magnitude
     * above fp32 noise. Bound it on both sides. */
    if (err_idot > 20.0 * err_scalar + 1e-7 && err_idot < 1e-2) {
        printf("PASS  IDOT path diverges as expected (max rel %.3e vs scalar %.3e)\n",
               err_idot, err_scalar);
        printf("      this is the activation quantization, not a kernel bug\n");
    } else if (err_idot <= 20.0 * err_scalar + 1e-7) {
        printf("FAIL  IDOT path is now as exact as scalar (%.3e vs %.3e).\n",
               err_idot, err_scalar);
        printf("      If the activation quantization was removed, delete this test\n");
        printf("      and the IDOT opt-in note in matmul_q -- do not just relax it.\n");
        failures++;
    } else {
        printf("FAIL  IDOT error %.3e is far larger than activation quantization\n", err_idot);
        printf("      predicts (~1e-3). Suspect a real kernel bug.\n");
        failures++;
    }

    /* --- 4. IDOT=1 must still be reachable ------------------------------- */
    matmul_q_idot_force = 1;
    matmul_q(y, x, q, scale, I, O);
    if (max_rel_err(y, ref, x, q, scale, I, O) > 20.0 * err_scalar + 1e-7) {
        printf("PASS  IDOT=1 still opts in to the fast path\n");
    } else {
        printf("FAIL  IDOT=1 no longer reaches the fast path\n");
        failures++;
    }

    free(x); free(q); free(scale); free(y); free(ref);
    printf("\n%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
#endif
}
