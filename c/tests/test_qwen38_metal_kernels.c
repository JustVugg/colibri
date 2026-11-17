/* Metal kernel tests for Qwen3.8: the qwen38 kernels against a double-precision
 * reference on random data, at the real model's shapes and at the tiny
 * fixture's. No model, no checkpoint. Build and run:
 * make test-qwen38-metal-kernels
 *
 * The qwen38 kernels share qwen36's device and command batch, so a test may
 * chain q36 and q38 dispatches on the same tensors and sync once. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../qwen38_gpu.h"

/* ---- helpers (copies of tests/test_qwen36_kernels.c's) ------------------- */

static uint32_t g_rng = 0x2545f491u;
static float frand(void) {                      /* uniform in [-1, 1) */
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float)(g_rng & 0xffffffu) / 8388608.0f - 1.0f;
}

static int g_failures;

/* got vs ref, with |got-ref| <= tol * scale[i] (scale NULL: |ref| + 1e-3). */
static void check(const char *what, const float *got, const double *ref, const double *scale, size_t n, double tol) {
    double worst = 0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        double s = scale ? scale[i] : fabs(ref[i]) + 1e-3;
        if (!isfinite(got[i]) || !isfinite(ref[i]) || !isfinite(s)) {
            worst = INFINITY;
            at = i;
            break;
        }
        double e = fabs((double)got[i] - ref[i]) / (s > 1e-30 ? s : 1e-30);
        if (!(e <= worst)) { worst = e; at = i; }
    }
    int ok = worst <= tol;
    printf("%-34s %s  worst=%.3g at %zu (got %.9g ref %.9g)\n", what, ok ? "ok  " : "FAIL", worst, at,
           (double)got[at], ref[at]);
    if (!ok) g_failures++;
}

static q36_tensor *rand_tensor(size_t count, float scale) {
    q36_tensor *t = q36_tensor_new(count * sizeof(float));
    float *p = q36_tensor_data(t);
    for (size_t i = 0; i < count; i++) p[i] = frand() * scale;
    return t;
}

static float *F(q36_tensor *t) { return q36_tensor_data(t); }

static void sync_or_die(void) {
    if (!q36_gpu_sync()) { fprintf(stderr, "GPU batch failed\n"); exit(1); }
}

static void expect(const char *what, int ok) {
    printf("%-34s %s\n", what, ok ? "ok  " : "FAIL");
    if (!ok) g_failures++;
}

/* ---- plumbing ------------------------------------------------------------- */

/* The reference norm of the original layout: x * rsqrt(mean(x^2) + eps) * (1 + w). */
static void ref_norm(double *y, const double *x, const float *w, int rows, int dim, double eps) {
    for (int r = 0; r < rows; r++) {
        double ss = 0;
        for (int j = 0; j < dim; j++) ss += x[(size_t)r * dim + j] * x[(size_t)r * dim + j];
        double k = 1.0 / sqrt(ss / dim + eps);
        for (int j = 0; j < dim; j++) y[(size_t)r * dim + j] = x[(size_t)r * dim + j] * k * (1.0 + w[j]);
    }
}

/* q36_rmsnorm -> q38_copy -> q36_rmsnorm on the same tensors, one sync at the
 * end: the result is norm(copy(norm(x))) only if the q38 dispatch runs in the
 * q36 stream, after the first norm and before the second. (Out of order, the
 * copy reads b's zeros or the second norm reads x itself.) */
static void test_copy_interleaved(int rows, int dim, int exact) {
    size_t n = (size_t)rows * dim;
    const float eps = 1e-6f;
    q36_gpu_set_exact(exact);
    q36_tensor *a = rand_tensor(n, 3.0f), *w = rand_tensor((size_t)dim, 0.5f);
    q36_tensor *b = q36_tensor_new(n * sizeof(float));
    double *x = malloc(n * sizeof(double)), *ref = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) x[i] = F(a)[i];
    ref_norm(ref, x, F(w), rows, dim, eps);
    ref_norm(x, ref, F(w), rows, dim, eps);     /* x <- norm(copy(norm(x))) */

    int ok = q36_rmsnorm(b, a, w, rows, dim, eps, 0) && q38_copy(a, b, n) && q36_rmsnorm(b, a, w, rows, dim, eps, 0);
    sync_or_die();
    char name[64];
    snprintf(name, sizeof name, "copy interleaved %dx%d %s", rows, dim, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(b), x, NULL, n, exact ? 1e-5 : 3e-4);
    free(x); free(ref);
    q36_tensor_free(a); q36_tensor_free(w); q36_tensor_free(b);
}

/* q38_copy declines what does not fit, and enqueues nothing. */
static void test_copy_declines(void) {
    q36_tensor *x = rand_tensor(16, 1.0f), *y = q36_tensor_new(8 * sizeof(float));
    int ok = !q38_copy(y, x, 16) && !q38_copy(x, y, 16) && !q38_copy(y, x, 0) && !q38_copy(NULL, x, 4) &&
             !q38_copy(y, NULL, 4);
    sync_or_die();
    for (int i = 0; i < 8; i++) ok = ok && F(y)[i] == 0.0f;
    expect("copy declines bad sizes", ok);
    q36_tensor_free(x); q36_tensor_free(y);
}

int main(void) {
    /* As tests/test_qwen36_kernels.c: GitHub's Apple Paravirtual device never completes
     * command buffers, so nothing runs there. */
    const char *device = q36_gpu_default_device_name();
    if (!*device) { puts("SKIPPED: no Metal device"); return 0; }
    if (strstr(device, "Apple Paravirtual device")) {
        printf("SKIPPED: %s: none of the kernels runs here\n", device);
        return 0;
    }
    if (!q38_gpu_init()) { puts("FAILED: no usable Metal device, or the kernels did not compile (see above)"); return 1; }
    printf("Metal device: %s\n", q36_gpu_device_name());

    for (int exact = 1; exact >= 0; exact--) {
        test_copy_interleaved(1, 2560, exact);      /* real: hidden 2560 */
        test_copy_interleaved(4, 2560, exact);
        test_copy_interleaved(3, 32, exact);        /* tiny fixture: hidden 32 */
    }
    test_copy_declines();

    printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    q36_gpu_cleanup();
    return g_failures ? 1 : 0;
}
