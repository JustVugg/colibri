/* test_v41_amx.c -- v41_amx.h (v2: weights as the A operand) against a double reference; needs an AMX-BF16 CPU and
 * skips otherwise. The weights come in each format the engine hands it -- bf16, e4m3 with 32x32 ue8m0 tiles, e2m1 pairs
 * with ue8m0 groups -- through the header's own decoders, and the reference decodes them independently (scalar).
 * V41_AMX=1 within the bf16 rounding of the activations, V41_AMX=2 near float32. Shapes: partial row tiles (S % 16,
 * S % 32), partial output tiles (O % 32, O % 16), the row-group split, strided x and y (the gap past O untouched).
 * Then timings: a trunk-sized product and an expert-sized one (33 rows), the case found slow before.
 * Build: gcc -O2 -fopenmp -march=x86-64-v4 test_v41_amx.c -o test_v41_amx -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#include "../v41_amx.h"

#if !V41_AMX_HAVE
int main(void) { printf("SKIP: not x86-64 Linux\n"); return 0; }
#else
static uint32_t rs = 12345;
static uint32_t rnu(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float rnd(void) { return (rnu() / 16777216.0f) * 2.0f - 1.0f; }

enum { F_BF16, F_FP8, F_MX4 };
typedef struct { int fmt, O, I; uint16_t *b; uint8_t *q, *s; V41AmxBf16 cb; V41AmxFp8 c8; V41Mx4 c4; } Wt;
static double e4_ref(uint8_t b) {   /* e4m3 by its definition: bias 7, subnormals 2^-6 * m/8 */
    const int e = (b >> 3) & 15, m = b & 7; const double v = e ? ldexp(1.0 + m / 8.0, e - 7) : ldexp(m / 8.0, -6);
    return (b & 0x80) ? -v : v;
}
static double mx4_ref(int nib) { static const double v[8] = {0, .5, 1, 1.5, 2, 3, 4, 6}; return (nib & 8) ? -v[nib & 7] : v[nib & 7]; }
static double wref(const Wt *w, int o, int i) {
    if (w->fmt == F_BF16) return v41_bf2f(w->b[(size_t)o * w->I + i]);
    if (w->fmt == F_FP8) return e4_ref(w->q[(size_t)o * w->I + i]) * ldexp(1.0, w->s[(size_t)(o / 32) * (w->I / 32) + i / 32] - 127);
    const uint8_t byte = w->q[(size_t)o * (w->I / 2) + i / 2];
    return mx4_ref((i & 1) ? byte >> 4 : byte & 15) * ldexp(1.0, w->s[(size_t)o * (w->I / 32) + i / 32] - 127);
}
static void wmake(Wt *w, int fmt, int O, int I) {
    memset(w, 0, sizeof *w); w->fmt = fmt; w->O = O; w->I = I;
    if (fmt == F_BF16) { w->b = malloc((size_t)O * I * 2); for (size_t k = 0; k < (size_t)O * I; k++) w->b[k] = v41_f2bf(rnd() * 0.5f); w->cb.w = w->b; }
    else if (fmt == F_FP8) {
        w->q = malloc((size_t)O * I); w->s = malloc((size_t)((O + 31) / 32) * (I / 32));
        for (size_t k = 0; k < (size_t)O * I; k++) { uint8_t b; do b = (uint8_t)rnu(); while ((b & 0x7f) == 0x7f); w->q[k] = b; }
        for (size_t k = 0; k < (size_t)((O + 31) / 32) * (I / 32); k++) w->s[k] = (uint8_t)(118 + rnu() % 8);
        w->c8.q = w->q; w->c8.s = w->s;
    } else {
        w->q = malloc((size_t)O * I / 2); w->s = malloc((size_t)O * (I / 32));
        for (size_t k = 0; k < (size_t)O * I / 2; k++) w->q[k] = (uint8_t)rnu();
        for (size_t k = 0; k < (size_t)O * (I / 32); k++) w->s[k] = (uint8_t)(122 + rnu() % 6);
        w->c4.q4 = w->q; w->c4.e8s = w->s;
    }
}
static void wfree(Wt *w) { free(w->b); free(w->q); free(w->s); }
static int wgemm(const Wt *w, float *y, int ys, const float *x, int xs, int S) {
    if (w->fmt == F_BF16) return v41_amx_gemm(y, ys, x, xs, S, w->I, w->O, v41_amx_dec_bf16, &w->cb);
    if (w->fmt == F_FP8) return v41_amx_gemm(y, ys, x, xs, S, w->I, w->O, v41_amx_dec_fp8, &w->c8);
    return v41_amx_gemm(y, ys, x, xs, S, w->I, w->O, v41_amx_dec_mxfp4, &w->c4);
}
static void arm(int mode) { g_amx_mode = -1; setenv("V41_AMX", mode == 1 ? "1" : "2", 1); if (!v41_amx_on()) { printf("SKIP: no AMX-BF16\n"); exit(0); } }

static int one(int mode, int fmt, int S, int I, int O, int xs, int ys) {
    static const char *fn[] = {"bf16", "fp8 ", "mxf4"};
    Wt w; wmake(&w, fmt, O, I);
    float *x = malloc((size_t)S * xs * 4), *y = malloc((size_t)S * ys * 4);
    for (size_t k = 0; k < (size_t)S * xs; k++) x[k] = rnd();
    for (size_t k = 0; k < (size_t)S * ys; k++) y[k] = -12345.0f;
    arm(mode);
    if (!wgemm(&w, y, ys, x, xs, S)) { printf("FAIL: gemm refused\n"); return 1; }
    double *wd = malloc((size_t)O * I * sizeof(double));
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) wd[(size_t)o * I + i] = wref(&w, o, i);
    double worst = 0; int bad = 0;
    for (int r = 0; r < S; r++) {
        for (int o = 0; o < O; o++) {
            double ref = 0, mag = 0;
            for (int i = 0; i < I; i++) { const double p = wd[(size_t)o * I + i] * x[(size_t)r * xs + i]; ref += p; mag += fabs(p); }
            const double err = fabs(y[(size_t)r * ys + o] - ref) / (mag > 0 ? mag : 1);
            if (!(err <= worst)) worst = err;   /* a NaN counts as the worst */
        }
        for (int o = O; o < ys; o++) if (y[(size_t)r * ys + o] != -12345.0f) bad++;
    }
    const double lim = mode == 1 ? 8e-3 : 1e-4;
    const int ok = worst < lim && !bad;
    printf("%s mode %d %s S %4d I %5d O %5d xs %5d ys %5d: worst rel err %.2e (limit %.0e)%s\n", ok ? "ok  " : "FAIL", mode,
           fn[fmt], S, I, O, xs, ys, worst, lim, bad ? ", wrote outside O" : "");
    wfree(&w); free(x); free(y); free(wd);
    return !ok;
}
static void perf(int fmt, int S, int I, int O, const char *what) {
    Wt w; wmake(&w, fmt, O, I);
    float *x = malloc((size_t)S * I * 4), *y = malloc((size_t)S * O * 4);
    for (size_t k = 0; k < (size_t)S * I; k++) x[k] = rnd();
    for (int mode = 1; mode <= 2; mode++) {
        arm(mode);
        wgemm(&w, y, O, x, I, S);
        const int reps = S < 256 ? 50 : 3;
        const double t0 = now_s(); for (int k = 0; k < reps; k++) wgemm(&w, y, O, x, I, S); const double t = (now_s() - t0) / reps;
        printf("perf %-7s mode %d: S %4d I %d O %d in %.3f ms, %.0f GFLOP/s (%d threads)\n", what, mode, S, I, O, t * 1e3,
               2.0 * S * I * (double)O / t / 1e9, V41_AMX_NTH());
    }
    wfree(&w); free(x); free(y);
}
int main(void) {
    int f = 0;
    const int sh[][5] = { {16, 64, 32, 64, 32}, {17, 96, 40, 96, 48}, {33, 128, 1000, 160, 1000}, {100, 5120, 48, 5120, 64},
                          {512, 1024, 576, 1024, 576}, {31, 512, 4096, 512, 4096}, {300, 256, 77, 256, 80}, {33, 64, 8, 64, 8} };
    for (int mode = 1; mode <= 2; mode++)
        for (int fmt = 0; fmt < 3; fmt++)
            for (size_t k = 0; k < sizeof sh / sizeof *sh; k++) f |= one(mode, fmt, sh[k][0], sh[k][1], sh[k][2], sh[k][3], sh[k][4]);
    perf(F_FP8, 2048, 5120, 4096, "trunk");
    perf(F_MX4, 33, 5120, 3072, "expert");
    perf(F_MX4, 33, 3072, 5120, "expdown");
    printf(f ? "FAILED\n" : "PASS\n");
    return f;
}
#endif
