/* The tier's MXFP4 copy mode (fmt 7) on a REAL device, against quant.h's CPU expert.
 *
 * qt_init_mxfp4 -> qt_note (staging, upload) -> qt_issue / qt_take (the asynchronous
 * coli_cuda_expert_group_issue path, the grouped MXFP4 kernels), every routed output
 * against matmul_mxfp4 + silu(gate) * up + down as the engine computes an expert, and
 * a swap that writes a newcomer into a victim's tensors (coli_cuda_tensor_overwrite)
 * checked the same way. Shapes: Qwen3.8's (hidden 2560, inter 640, top-10: the 16-byte
 * loads), the tiny fixture's (32, 8: one short group a down row) and 96 x 40 (a short
 * last group on the down side). Fails above 1e-5 of the output's largest magnitude
 * (the kernels' warp sums add in another order than the CPU's loop).
 * Build: make tests/test_qwen36_tier_mxfp4_gpu CUDA=1 (Windows: CUDA_DLL=1, with
 * coli_cuda.dll beside the executable). No device: SKIP. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"
#include "../quant.h"
#include "../qwen36_tier.c"

static int fails;
static double worst_rel;

static void ref_expert(float *y, const float *x, const uint8_t *gw, const uint8_t *uw, const uint8_t *dw,
                       const uint8_t *gs, const uint8_t *us, const uint8_t *ds, int D, int I) {
    float *g = malloc(sizeof(float) * I), *u = malloc(sizeof(float) * I);
    matmul_mxfp4(g, x, gw, gs, 1, D, I);
    matmul_mxfp4(u, x, uw, us, 1, D, I);
    for (int i = 0; i < I; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
    matmul_mxfp4(y, g, dw, ds, 1, I, D);
    free(g); free(u);
}
static void compare(const char *what, const float *got, const float *want, int n) {
    double mx = 0, mag = 0; int same = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)got[i] - want[i]);
        if (!(d <= mx)) mx = d;   /* a NaN wins */
        if (fabs(want[i]) > mag) mag = fabs(want[i]);
        same += !memcmp(&got[i], &want[i], sizeof(float));
    }
    double rel = mag > 0 ? mx / mag : mx;
    if (!(rel <= worst_rel)) worst_rel = rel;
    printf("  %-40s max|d| %.3g of max|ref| %.3g (rel %.3g), bit-identical %d/%d\n", what, mx, mag, rel, same, n);
    if (!(rel <= 1e-5)) { printf("  FAIL: %s\n", what); fails++; }
}

/* one layer's experts as a record holds them: gate|up|down nibbles, gate|up|down ue8m0 bytes */
typedef struct { int L, E, D, I; size_t mb_gu, mb_d, sb_gu, sb_d; uint8_t *w, *s; } Experts;
static uint32_t rng = 12345;
static uint32_t nxt(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void ex_make(Experts *m) {
    m->mb_gu = (size_t)m->I * (m->D / 2); m->mb_d = (size_t)m->D * (m->I / 2);
    m->sb_gu = (size_t)m->I * ((m->D + 31) / 32); m->sb_d = (size_t)m->D * ((m->I + 31) / 32);
    size_t nw = (size_t)m->L * m->E * (2 * m->mb_gu + m->mb_d), ns = (size_t)m->L * m->E * (2 * m->sb_gu + m->sb_d);
    m->w = malloc(nw); m->s = malloc(ns);
    for (size_t i = 0; i < nw; i++) m->w[i] = (uint8_t)nxt();
    for (size_t i = 0; i < ns; i++) m->s[i] = (uint8_t)(118 + nxt() % 6);   /* 2^-9 .. 2^-4, as Quark's */
}
static void ex_ptr(const Experts *m, int l, int e, const uint8_t **g, const uint8_t **u, const uint8_t **d,
                   const uint8_t **gs, const uint8_t **us, const uint8_t **ds) {
    size_t k = (size_t)l * m->E + e;
    *g = m->w + k * (2 * m->mb_gu + m->mb_d); *u = *g + m->mb_gu; *d = *u + m->mb_gu;
    *gs = m->s + k * (2 * m->sb_gu + m->sb_d); *us = *gs + m->sb_gu; *ds = *us + m->sb_gu;
}
static void note(const Experts *m, int l, int e) {
    const uint8_t *g, *u, *d, *gs, *us, *ds;
    ex_ptr(m, l, e, &g, &u, &d, &gs, &us, &ds);
    qt_note(l, e, g, u, d, (const float *)gs, (const float *)us, (const float *)ds);
}

/* route K residents per layer and token; every output against the CPU's experts */
static void route(const char *name, const Experts *m, int K, int tokens, int pick_from) {
    float *x = malloc(sizeof(float) * m->D), *got = malloc(sizeof(float) * m->D),
          *want = malloc(sizeof(float) * m->D), *y = malloc(sizeof(float) * m->D);
    const uint8_t *g, *u, *d, *gs, *us, *ds;
    for (int t = 0; t < tokens; t++)
        for (int l = 0; l < m->L; l++) {
            int eids[QT_MAX_ROWS]; float val[QT_MAX_ROWS];
            for (int k = 0; k < K; k++) {
                int e, dup;
                do { e = (int)(nxt() % (unsigned)pick_from); dup = 0; for (int j = 0; j < k; j++) dup |= eids[j] == e; } while (dup);
                eids[k] = e; val[k] = 0.05f + (float)(nxt() % 100) / 100.f;
            }
            for (int i = 0; i < m->D; i++) x[i] = (float)((int)(nxt() % 2001) - 1000) / 400.f;
            uint32_t mask = qt_issue(l, eids, K, x);
            memset(got, 0, sizeof(float) * m->D);
            if (mask != (K == 32 ? 0xFFFFFFFFu : (1u << K) - 1) || !qt_take(mask, val, K, got)) {
                printf("  FAIL: %s: issue mask %#x or take\n", name, mask); fails++; continue;
            }
            memset(want, 0, sizeof(float) * m->D);
            for (int k = 0; k < K; k++) {
                ex_ptr(m, l, eids[k], &g, &u, &d, &gs, &us, &ds);
                ref_expert(y, x, g, u, d, gs, us, ds, m->D, m->I);
                for (int i = 0; i < m->D; i++) want[i] += val[k] * y[i];
            }
            char what[96]; snprintf(what, sizeof what, "%s token %d layer %d", name, t, l);
            compare(what, got, want, m->D);
        }
    free(x); free(got); free(want); free(y);
}

static void tier_case(const char *name, Experts *m, int K) {
    printf(" %s: %d layer(s) of %d experts, hidden %d, inter %d, top-%d\n", name, m->L, m->E, m->D, m->I, K);
    if (!qt_init_mxfp4(m->L, m->E, m->D, m->I, 2, K)) { printf("  FAIL: qt_init_mxfp4\n"); fails++; return; }
    for (int l = 0; l < m->L; l++) for (int e = 0; e < m->E; e++) { note(m, l, e); qt_fill_wait(); }
    int resident = 0;
    for (int l = 0; l < m->L; l++) for (int e = 0; e < m->E; e++) resident += qt_is_resident(l, e);
    if (resident != m->L * m->E) { printf("  FAIL: %d of %d experts resident\n", resident, m->L * m->E); fails++; }
    route(name, m, K, 3, m->E);
    qt_shutdown();
}

/* a budget of K experts: a hot newcomer is written into a victim's buffers in place */
static void swap_case(Experts *m, int K) {
    printf(" swap in place: hidden %d, inter %d, a budget of %d experts\n", m->D, m->I, K);
    char gb[64];
    if (!qt_init_mxfp4(1, m->E, m->D, m->I, 2, K)) { printf("  FAIL: qt_init_mxfp4\n"); fails++; return; }
    size_t eb = G.exp_bytes;
    qt_shutdown();
    snprintf(gb, sizeof gb, "%.15f", (double)(K * eb + eb / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    if (!qt_init_mxfp4(1, m->E, m->D, m->I, 2, K)) { printf("  FAIL: qt_init_mxfp4 (budget)\n"); fails++; return; }
    for (int e = 0; e < K; e++) { note(m, 0, e); qt_fill_wait(); }
    for (int i = 0; i < 64 && !qt_is_resident(0, K); i++) { note(m, 0, K); qt_fill_wait(); }
    int resident = 0; for (int e = 0; e < m->E; e++) resident += qt_is_resident(0, e);
    printf("  swaps %llu, in place %llu, residents %d\n", (unsigned long long)G.swaps, (unsigned long long)G.inplace, resident);
    if (!qt_is_resident(0, K) || resident != K || G.inplace < 1) { printf("  FAIL: no swap in place\n"); fails++; }
    /* route among the residents only: the newcomer and the K-1 that stayed */
    int res[QT_MAX_ROWS], n = 0;
    for (int e = 0; e <= K; e++) if (qt_is_resident(0, e)) res[n++] = e;
    float *x = malloc(sizeof(float) * m->D), *got = calloc((size_t)m->D, sizeof(float)),
          *want = calloc((size_t)m->D, sizeof(float)), *y = malloc(sizeof(float) * m->D);
    float val[QT_MAX_ROWS];
    for (int i = 0; i < m->D; i++) x[i] = (float)((int)(nxt() % 2001) - 1000) / 400.f;
    for (int k = 0; k < n; k++) val[k] = 0.1f * (float)(k + 1);
    uint32_t mask = qt_issue(0, res, n, x);
    if (mask != (1u << n) - 1 || !qt_take(mask, val, n, got)) { printf("  FAIL: swap issue mask %#x\n", mask); fails++; }
    else {
        const uint8_t *g, *u, *d, *gs, *us, *ds;
        for (int k = 0; k < n; k++) {
            ex_ptr(m, 0, res[k], &g, &u, &d, &gs, &us, &ds);
            ref_expert(y, x, g, u, d, gs, us, ds, m->D, m->I);
            for (int i = 0; i < m->D; i++) want[i] += val[k] * y[i];
        }
        compare("residents after the swap", got, want, m->D);
    }
    free(x); free(got); free(want); free(y);
    qt_shutdown();
    setenv("CUDA_EXPERT_GB", "2", 1);
}

int main(void) {
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    setenv("CUDA_EXPERT_GB", "2", 1);
    setenv("QT_UPLOAD_SYNC", "1", 1);
    if (coli_cuda_available_device_count() < 1) { puts("SKIP no CUDA device"); return 0; }

    Experts big = {.L = 1, .E = 16, .D = 2560, .I = 640};
    ex_make(&big);
    tier_case("Qwen3.8 shapes", &big, 10);
    Experts tiny = {.L = 4, .E = 4, .D = 32, .I = 8};
    ex_make(&tiny);
    tier_case("the tiny fixture's shapes", &tiny, 2);
    Experts odd = {.L = 2, .E = 6, .D = 96, .I = 40};
    ex_make(&odd);
    tier_case("a short last group", &odd, 3);
    swap_case(&big, 4);

    printf("worst relative max|gpu-cpu| = %.3g\n", worst_rel);
    if (fails) { printf("test_qwen36_tier_mxfp4_gpu: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_mxfp4_gpu: ok\n");
    return 0;
}
