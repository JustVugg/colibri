/* COLI_CUDA_F8_MMA oracle + bench: the tensor-core fmt=8 expert group (cuda_f8_mma.cuh)
 * against the warp kernels (grouped_*_f8w, the default) and a CPU double reference.
 * Needs a GPU with compute capability >= 8.0; no model. Run: make f8-mma-test
 * (or: tests/test_f8_mma_cuda [D I count], defaults 2048 512 10).
 *
 * Checks (FAIL exits 1):
 *   1. CPU reference, rows per expert T = 1, 8, 16, 64, 256 and a mixed group: every
 *      checked output of the MMA path within the per-output bound below. The warp
 *      kernels are held to the same bound (they sit far inside it: f32 activations).
 *   2. The path was taken (g_f8_mma_calls) when on, and not when off or pinned.
 *   3. Row-tiling invariance: one expert's rows computed in a 64-row call (row tiles of
 *      32/64) equal, bit for bit, the same rows in an 8-row call (row tiles of 16).
 *   4. issue/take == sync, bit for bit, on a group the async path admits (<= 32 rows).
 *   5. A pinned call (coli_cuda_expert_group_pinned, pin 1) ignores COLI_CUDA_F8_MMA:
 *      bit-identical to the warp kernels.
 * Bench (kernel time only, cudaEvents on the group stream, after warm-up): the MMA
 * launch vs the warp-kernel launch for T = 1, 8, 16, 64, 256 rows per expert, `count`
 * experts, plus the effective weight bandwidth (each expert's 3 matrices read once).
 *
 * Tolerance (per output, derived, not tuned). Weights decode exactly (every finite e4m3
 * code is a bf16, test_f8_mma_idx); products of two bf16 are exact in f32. Errors:
 *   x -> bf16 (RN):           |dx| <= 2^-8 |x|
 *   tensor-core sums:         <= 2^-13 sum|terms| per 128-k block. Assumption, stated:
 *                             NVIDIA's bf16 mma.sync with f32 accumulation is not IEEE RN
 *                             (Fasi et al. 2021 measured truncation after alignment to
 *                             the largest term); 8 chained k16 steps of 17 terms losing
 *                             at most 2^-23 of the largest each is <= 2^-14.9. 2^-13 keeps
 *                             a 4x margin and is still 32x below the bf16 term.
 *   acc = fmaf(part, s, acc): KB roundings of 2^-24 over K/128 blocks.
 * so for a projection z = sum_k s w_k x_k with A = sum_k |s w_k x_k|:
 *   |z^ - z| <= c A,  c = 2^-8 + 1.01 (2^-13 + KB 2^-24)                      (gate, up)
 * h = silu(g) u in f32 (|silu'| <= 1.1, ~6 roundings incl. expf's 2 ulp):
 *   Eh = 1.1 Eg (|u| + Eu) + |silu(g)| Eu + 6 2^-24 (|silu(g)| + 1.1 Eg)(|u| + Eu)
 * the down kernel reads bf16(h^): |bf16(h^) - h| <= Eh + 2^-8 (|h| + Eh) =: Dh, and
 *   Ey = sum_i |s w_i| Dh_i + 1.01 (2^-13 + KB_I 2^-24) sum_i |s w_i| (|h_i| + Dh_i)
 * plus 1e-12 of the magnitude for the double reference itself. The ratio max|err|/Ey is
 * printed: well under 1 is expected (the bound is worst case); above 1 fails.
 *
 * What justifies turning it on (COLI_CUDA_F8_MMA=1) on a card: every check passes, and the
 * bench shows mma faster than f8w from some T on; set COLI_CUDA_F8_MMA_MIN_ROWS to the
 * printed crossover. That is kernel time only: an end-to-end A/B of the model (prompt
 * throughput, and greedy output / verify acceptance, since bf16 activations change the
 * bits) is the second gate. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

#include "../backend_cuda.cu"

#ifdef _WIN32
static int setenv(const char *name, const char *value, int) { return _putenv_s(name, value); }
#endif

#if !COLI_F8_MMA
int main(void) { printf("test_f8_mma_cuda: COLI_F8_MMA not built (HIP), nothing to test\n"); return 0; }
#else

static float e4m3_ref(uint8_t b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    float v = e ? ldexpf(1.f + m / 8.f, e - 7) : ldexpf(m / 8.f, -6);
    return s ? -v : v;
}
static float g_lut[256];   /* e4m3_ref, tabled for the reference loops */
static uint64_t rng = 0x2545F4914F6CDD1Dull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static float frnd(void) { return (float)((rnd() >> 8) * (1.0 / 16777216.0)); }
static uint8_t rnd_e4m3(void) { uint8_t b = (uint8_t)rnd(); if ((b & 0x7F) == 0x7F) b &= (uint8_t)~1; return b; }

struct Expert { uint8_t *g, *u, *d; float *gs, *us, *ds; ColiCudaTensor *tg, *tu, *td; };

/* z = W x in double, with the magnitude sum A = sum |s w x| */
static void ref_proj(const uint8_t *w, const float *sc, int K, int O, const double *x, double *z, double *A) {
    int KB = K / 128;
    for (int o = 0; o < O; o++) {
        const uint8_t *r = w + (size_t)o * K; const float *s = sc + (size_t)(o / 128) * KB;
        double a = 0, m = 0;
        for (int k = 0; k < K; k++) { double p = (double)g_lut[r[k]] * s[k / 128] * x[k]; a += p; m += fabs(p); }
        z[o] = a; A[o] = m;
    }
}

/* one row through the expert: y and its bound Ey (header) */
static void ref_row(const Expert &e, int D, int I, const float *xf, double *y, double *Ey) {
    const double cg = ldexp(1, -8) + 1.01 * (ldexp(1, -13) + (D / 128) * ldexp(1, -24));
    const double cd = 1.01 * (ldexp(1, -13) + (I / 128) * ldexp(1, -24));
    double *x = (double *)malloc(sizeof(double) * D), *g = (double *)malloc(sizeof(double) * I),
           *u = (double *)malloc(sizeof(double) * I), *Ag = (double *)malloc(sizeof(double) * I),
           *Au = (double *)malloc(sizeof(double) * I), *h = (double *)malloc(sizeof(double) * I),
           *Dh = (double *)malloc(sizeof(double) * I);
    for (int k = 0; k < D; k++) x[k] = xf[k];
    ref_proj(e.g, e.gs, D, I, x, g, Ag);
    ref_proj(e.u, e.us, D, I, x, u, Au);
    for (int i = 0; i < I; i++) {
        double sg = g[i] / (1 + exp(-g[i])), Eg = cg * Ag[i], Eu = cg * Au[i];
        double Eh = 1.1 * Eg * (fabs(u[i]) + Eu) + fabs(sg) * Eu + 6 * ldexp(1, -24) * (fabs(sg) + 1.1 * Eg) * (fabs(u[i]) + Eu);
        h[i] = sg * u[i];
        Dh[i] = Eh + ldexp(1, -8) * (fabs(h[i]) + Eh);
    }
    for (int o = 0; o < D; o++) {
        const uint8_t *r = e.d + (size_t)o * I; const float *s = e.ds + (size_t)(o / 128) * (I / 128);
        double a = 0, m = 0, b1 = 0, b2 = 0;
        for (int i = 0; i < I; i++) {
            double sw = (double)g_lut[r[i]] * s[i / 128];
            a += sw * h[i]; m += fabs(sw * h[i]);
            b1 += fabs(sw) * Dh[i]; b2 += fabs(sw) * (fabs(h[i]) + Dh[i]);
        }
        y[o] = a; Ey[o] = b1 + cd * b2 + 1e-12 * m + 1e-30;
    }
    free(x); free(g); free(u); free(Ag); free(Au); free(h); free(Dh);
}

static void set_mma(const char *v) { setenv("COLI_CUDA_F8_MMA", v, 1); }

/* sync group with the MMA path on (MIN_ROWS 1) or off; returns 0 on API failure */
static int run(ColiCudaTensor **tg, ColiCudaTensor **tu, ColiCudaTensor **td, const int *rows, int n, float *y,
               const float *x, int mma) {
    set_mma(mma ? "1" : "0");
    uint64_t c0 = g_f8_mma_calls;
    int ok = coli_cuda_expert_group(tg, tu, td, rows, n, y, x);
    if (ok && (g_f8_mma_calls != c0) != (mma != 0)) { printf("FAIL: MMA path %s when %s\n", mma ? "not taken" : "taken", mma ? "on" : "off"); return 0; }
    return ok;
}

int main(int argc, char **argv) {
    int D = argc > 1 ? atoi(argv[1]) : 2048, I = argc > 2 ? atoi(argv[2]) : 512, NE = argc > 3 ? atoi(argv[3]) : 10;
    if (D < 128 || I < 128 || D % 128 || I % 128 || NE < 3 || NE > 64) { printf("usage: %s [D I count]: D, I multiples of 128, 3 <= count <= 64\n", argv[0]); return 2; }
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("FAIL cuda init (no device?)\n"); return 1; }
    DeviceContext *ctx = find_ctx(0);
    int cc = ctx->compute_major * 10 + ctx->compute_minor;
    if (cc < 80) { printf("SKIP: compute capability %d < 80, the MMA path is never taken here\n", cc); return 0; }
    for (int i = 0; i < 256; i++) g_lut[i] = e4m3_ref((uint8_t)i);
    if (!coli_cuda_fp8_set_lut(g_lut)) { printf("FAIL set_lut\n"); return 1; }
    setenv("COLI_CUDA_F8_WARP", "1", 1);
    setenv("COLI_CUDA_F8_MMA_MIN_ROWS", "1", 1);
    printf("device cc %d, D=%d I=%d experts=%d\n", cc, D, I, NE);

    Expert *E = (Expert *)calloc(NE, sizeof(Expert));
    ColiCudaTensor **tg = (ColiCudaTensor **)calloc(NE, sizeof(void *)), **tu = (ColiCudaTensor **)calloc(NE, sizeof(void *)),
                   **td = (ColiCudaTensor **)calloc(NE, sizeof(void *));
    const int nbD = D / 128, nbI = I / 128;
    for (int c = 0; c < NE; c++) {
        Expert &e = E[c];
        e.g = (uint8_t *)malloc((size_t)I * D); e.u = (uint8_t *)malloc((size_t)I * D); e.d = (uint8_t *)malloc((size_t)D * I);
        e.gs = (float *)malloc((size_t)nbI * nbD * 4); e.us = (float *)malloc((size_t)nbI * nbD * 4); e.ds = (float *)malloc((size_t)nbD * nbI * 4);
        for (size_t i = 0; i < (size_t)I * D; i++) { e.g[i] = rnd_e4m3(); e.u[i] = rnd_e4m3(); e.d[i] = rnd_e4m3(); }
        for (int i = 0; i < nbI * nbD; i++) { e.gs[i] = ldexpf(1.f + frnd(), -12); e.us[i] = ldexpf(1.f + frnd(), -12); e.ds[i] = ldexpf(1.f + frnd(), -10); }
        if (!coli_cuda_tensor_upload(&e.tg, e.g, e.gs, 8, D, I, 0) || !coli_cuda_tensor_upload(&e.tu, e.u, e.us, 8, D, I, 0) ||
            !coli_cuda_tensor_upload(&e.td, e.d, e.ds, 8, I, D, 0)) { printf("FAIL upload\n"); return 1; }
        tg[c] = e.tg; tu[c] = e.tu; td[c] = e.td;
    }
    const int TMAX = 256;
    size_t xcap = (size_t)NE * TMAX * D;
    float *x = (float *)malloc(xcap * 4), *ym = (float *)malloc(xcap * 4), *yw = (float *)malloc(xcap * 4);
    for (size_t i = 0; i < xcap; i++) x[i] = (frnd() - .5f) * 4.f;
    double *yr = (double *)malloc(sizeof(double) * D), *Ey = (double *)malloc(sizeof(double) * D);
    int fail = 0;

    /* ---- 1+2: accuracy against the CPU reference ------------------------------------ */
    struct Case { const char *name; int n; int rows[4]; } cases[] = {
        {"T=1", 3, {1, 1, 1}}, {"T=8", 3, {8, 8, 8}}, {"T=16", 3, {16, 16, 16}},
        {"T=64", 2, {64, 64}}, {"T=256", 1, {256}}, {"mixed 40/1/17", 3, {40, 1, 17}}};
    for (auto &cs : cases) {
        int total = 0; for (int c = 0; c < cs.n; c++) total += cs.rows[c];
        if (!run(tg, tu, td, cs.rows, cs.n, ym, x, 1) || !run(tg, tu, td, cs.rows, cs.n, yw, x, 0)) { printf("FAIL %s: group call\n", cs.name); return 1; }
        double rm = 0, rw = 0, em = 0, ew = 0, ref2 = 0; long nck = 0, badm = 0, badw = 0;
        int off = 0;
        for (int c = 0; c < cs.n; c++) {
            for (int r = 0; r < cs.rows[c]; r++) {
                if (cs.rows[c] > 16 && r % 8 != 3 && r != cs.rows[c] - 1) continue;   /* a sample of the big tiles */
                size_t z = (size_t)(off + r) * D;
                ref_row(E[c], D, I, x + z, yr, Ey);
                for (int o = 0; o < D; o++) {
                    double dm = fabs(ym[z + o] - yr[o]), dw = fabs(yw[z + o] - yr[o]);
                    if (!(dm <= Ey[o])) badm++;
                    if (!(dw <= Ey[o])) badw++;
                    rm = fmax(rm, dm / Ey[o]); rw = fmax(rw, dw / Ey[o]);
                    em += dm * dm; ew += dw * dw; ref2 += yr[o] * yr[o]; nck++;
                }
            }
            off += cs.rows[c];
        }
        printf("%-14s rows %3d: mma max err/bound %.3g, rel RMS %.3g | f8w max err/bound %.3g, rel RMS %.3g | %ld outputs checked\n",
               cs.name, total, rm, sqrt(em / ref2), rw, sqrt(ew / ref2), nck);
        if (badm || badw) { printf("FAIL %s: %ld mma / %ld f8w outputs outside the bound\n", cs.name, badm, badw); fail = 1; }
        if (rm == 0) { printf("FAIL %s: mma output equals the double reference exactly (bf16 path not exercised?)\n", cs.name); fail = 1; }
    }

    /* ---- 3: row-tiling invariance ---------------------------------------------------- */
    {
        int r64[1] = {64}, r8[1] = {8};
        float *y8 = (float *)malloc((size_t)8 * D * 4);
        if (!run(tg, tu, td, r64, 1, ym, x, 1)) return 1;
        int bad = 0;
        for (int b = 0; b < 8; b++) {
            if (!run(tg, tu, td, r8, 1, y8, x + (size_t)b * 8 * D, 1)) return 1;
            bad += memcmp(y8, ym + (size_t)b * 8 * D, (size_t)8 * D * 4) != 0;
        }
        printf("row-tiling invariance (64 rows in one call vs 8 calls of 8): %d of 8 slices differ\n", bad);
        if (bad) { printf("FAIL row-tiling invariance\n"); fail = 1; }
        free(y8);
    }

    /* ---- 4: issue/take == sync ------------------------------------------------------- */
    {
        int rows[3] = {12, 9, 3};
        if (!run(tg, tu, td, rows, 3, ym, x, 1)) return 1;
        set_mma("1");
        uint64_t c0 = g_f8_mma_calls;
        if (!coli_cuda_expert_group_issue(tg, tu, td, rows, 3, x)) { printf("FAIL issue\n"); return 1; }
        const float *ya = coli_cuda_expert_group_take(0);
        if (!ya || g_f8_mma_calls == c0) { printf("FAIL take / path not taken on issue\n"); return 1; }
        int same = !memcmp(ya, ym, (size_t)24 * D * 4);
        printf("issue/take vs sync (24 rows, MMA): %s\n", same ? "bitwise OK" : "DIFFER");
        if (!same) fail = 1;
    }

    /* ---- 5: pinned ignores the toggle ------------------------------------------------ */
    {
        int rows[2] = {16, 16};
        if (!run(tg, tu, td, rows, 2, yw, x, 0)) return 1;
        set_mma("1");
        uint64_t c0 = g_f8_mma_calls;
        if (!coli_cuda_expert_group_pinned(tg, tu, td, rows, 2, ym, x, 1)) { printf("FAIL pinned call\n"); return 1; }
        int same = !memcmp(ym, yw, (size_t)32 * D * 4) && g_f8_mma_calls == c0;
        printf("pinned call with COLI_CUDA_F8_MMA=1: %s\n", same ? "warp kernels, bitwise OK" : "FAIL");
        if (!same) fail = 1;
    }

    /* ---- bench ------------------------------------------------------------------------ */
    {
        const int Ts[5] = {1, 8, 16, 64, 256};
        int *rows = (int *)malloc(sizeof(int) * NE);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        double wbytes = (double)NE * 3 * D * I;
        int cross = -1;
        printf("bench: %d experts, kernel time per group call (median of 5 runs of 20)\n", NE);
        printf("  T   rows   f8w us   mma us  speedup  f8w GB/s  mma GB/s\n");
        for (int ti = 0; ti < 5; ti++) {
            int T = Ts[ti], total = NE * T;
            for (int c = 0; c < NE; c++) rows[c] = T;
            /* one real call stages x and the descriptors in ctx (and warms both paths) */
            if (!run(tg, tu, td, rows, NE, ym, x, 0) || !run(tg, tu, td, rows, NE, ym, x, 1)) return 1;
            GroupDesc *dev = (GroupDesc *)ctx->group_desc;
            GroupDesc host[64];
            cudaMemcpy(host, dev, sizeof(GroupDesc) * NE, cudaMemcpyDeviceToHost);
            double us[2];
            for (int p = 0; p < 2; p++) {
                set_mma(p ? "1" : "0");
                float med[5];
                for (int rep = 0; rep < 5; rep++) {
                    cudaEventRecord(a, ctx->stream);
                    for (int it = 0; it < 20; it++)
                        if (!f8_group_launch(ctx, dev, host, I, D, T, NE, total, 0)) { printf("FAIL launch\n"); return 1; }
                    cudaEventRecord(b, ctx->stream); cudaEventSynchronize(b);
                    cudaEventElapsedTime(&med[rep], a, b);
                }
                for (int i = 0; i < 5; i++) for (int j = i + 1; j < 5; j++) if (med[j] < med[i]) { float t = med[i]; med[i] = med[j]; med[j] = t; }
                us[p] = med[2] * 1000.0 / 20;
            }
            if (cudaGetLastError() != cudaSuccess) { printf("FAIL bench launch error\n"); return 1; }
            printf("%4d %6d %8.1f %8.1f %8.2fx %9.0f %9.0f\n", T, total, us[0], us[1], us[0] / us[1],
                   wbytes / us[0] / 1e3, wbytes / us[1] / 1e3);
            if (us[1] < us[0]) { if (cross < 0) cross = T; } else cross = -1;
        }
        if (cross > 0) printf("mma faster from T=%d on (of 1,8,16,64,256): COLI_CUDA_F8_MMA_MIN_ROWS=%d is the measured crossover\n", cross, cross);
        else printf("mma not faster at T=256 on this card: keep COLI_CUDA_F8_MMA off\n");
        free(rows);
    }
    printf(fail ? "test_f8_mma_cuda: FAILED\n" : "test_f8_mma_cuda: all checks passed\n");
    return fail;
}
#endif
