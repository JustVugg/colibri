/* Physical CUDA diagnostic for a deterministic Q2_0 SwiGLU expert.
 *
 * Build from the repository's c/ directory with a MinGW-w64 compiler that has
 * OpenMP (for example, MSYS2 UCRT64):
 *   gcc -std=c11 -O2 -Wall -Wextra -fopenmp -DCOLI_CUDA \
 *       tests/test_q2_numeric_probe.c backend_loader.c \
 *       -o q2_numeric_probe.exe -lm -lpsapi
 * Run q2_numeric_probe.exe from c/. The already-built coli_cuda.dll must be
 * beside the executable or on PATH; keep its dependent CUDA runtime DLLs
 * discoverable too. This program does not build CUDA kernels.
 *
 * A working physical NVIDIA CUDA device and compatible coli_cuda.dll are
 * required. Missing DLLs, chain APIs, or a device fail the diagnostic; there
 * is no CPU-only or mocked success path. The CPU integer-activation reference
 * uses this checkout's production quantized CPU kernels, so it is useful for
 * isolating the CPU/CUDA arithmetic paths but is not an independent accuracy
 * oracle. The metrics and finite checks are diagnostic only, not an accuracy
 * guarantee or a production acceptance threshold. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../backend_cuda.h"
#include "../cuda_chain.h"
#include "../ggml_quants.h"

static unsigned rng = 0x6d2b79f5u;
static unsigned next_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

static void make_q2(uint8_t *w, size_t bytes, unsigned salt, int test_scale_profile) {
    /* Q2_0 block: f16 d followed by 64 two-bit codes. The stress profile uses
     * d=0.0625; the comparison profile matches test_cuda_bd's 0x2400..0x2600. */
    for (size_t b = 0, block = 0; b < bytes; b += 18, block++) {
        const unsigned scale_noise = next_u32();
        w[b] = 0x00;
        w[b + 1] = test_scale_profile
            ? (uint8_t)(0x24 + (scale_noise + salt + (unsigned)block) % 3u) : 0x2c;
        for (int j = 0; j < 16; j++)
            w[b + 2 + j] = (uint8_t)(next_u32() + salt + (unsigned)block * 37u + (unsigned)j * 13u);
    }
}

typedef struct { double abs_rms, max_abs, cosine, rel_max; size_t worst; } Metrics;
static Metrics metrics(const float *a, const float *b, size_t n) {
    Metrics m = {0}; double se = 0, aa = 0, bb = 0, ab = 0;
    for (size_t i = 0; i < n; i++) {
        double av = a[i], bv = b[i], d = fabs(av - bv);
        se += d * d; aa += av * av; bb += bv * bv; ab += av * bv;
        if (d > m.max_abs) { m.max_abs = d; m.worst = i; }
        double rel = d / (fabs(bv) + 1e-3);
        if (rel > m.rel_max) m.rel_max = rel;
    }
    m.abs_rms = sqrt(se / (double)n);
    m.cosine = (aa > 0 && bb > 0) ? ab / sqrt(aa * bb) : 1.0;
    return m;
}
static void print_metrics(const char *name, Metrics m) {
    printf("  %-22s abs_rms=%.9g max_abs=%.9g cosine=%.9g rel_max=%.9g worst=%zu\n",
           name, m.abs_rms, m.max_abs, m.cosine, m.rel_max, m.worst);
}
static double output_term_relmax(const float *got, const float *ref, const float *hidden,
                                 const uint8_t *down, size_t rb, int S, int D, int I) {
    float *row = (float *)malloc((size_t)I * sizeof(float));
    if (!row) return INFINITY;
    double worst = 0.0;
    for (int o = 0; o < D; o++) {
        gq_dequant_row(GQ_Q2_0, down + (size_t)o * rb, row, I);
        for (int s = 0; s < S; s++) {
            double mag = 0.0;
            for (int i = 0; i < I; i++) mag += fabs((double)row[i] * hidden[(size_t)s * I + i]);
            double e = fabs((double)got[(size_t)s * D + o] - ref[(size_t)s * D + o]) / (mag + 1e-6);
            if (!(e <= worst)) worst = e;
        }
    }
    free(row); return worst;
}

static int run_case(int S, int D, int I, int test_scale_profile) {
    rng = 0x6d2b79f5u;
    const int K = 1, E = 1, fmt = 100 + GQ_Q2_0;
    const size_t rb_g = (size_t)gq_row_bytes(GQ_Q2_0, D);
    const size_t rb_d = (size_t)gq_row_bytes(GQ_Q2_0, I);
    const size_t gb = rb_g * (size_t)I, db = rb_d * (size_t)D;
    const size_t nx = (size_t)S * D, nh = (size_t)S * I, ny = nx;
    uint8_t *G = (uint8_t *)malloc(gb), *U = (uint8_t *)malloc(gb), *W = (uint8_t *)malloc(db);
    float *x = (float *)malloc(nx * sizeof(float));
    float *ig = (float *)malloc(nh * sizeof(float)), *iu = (float *)malloc(nh * sizeof(float));
    float *ih = (float *)malloc(nh * sizeof(float)), *iy = (float *)malloc(ny * sizeof(float));
    float *fg = (float *)malloc(nh * sizeof(float)), *fu = (float *)malloc(nh * sizeof(float));
    float *fh = (float *)malloc(nh * sizeof(float)), *fy = (float *)malloc(ny * sizeof(float));
    if (!G || !U || !W || !x || !ig || !iu || !ih || !iy || !fg || !fu || !fh || !fy) {
        fprintf(stderr, "FAIL S=%d CPU allocation\n", S); return 1;
    }
    make_q2(G, gb, 11, test_scale_profile); make_q2(U, gb, 71, test_scale_profile);
    make_q2(W, db, 131, test_scale_profile);
    for (size_t i = 0; i < nx; i++) x[i] = ((int)(next_u32() % 20001u) - 10000) * 1.0e-5f;
    for (size_t i = 0; i < nh; i++) fg[i] = fu[i] = fh[i] = NAN;
    for (size_t i = 0; i < ny; i++) fy[i] = NAN;

    gq_act ax = {0}, ah = {0};
    int ok = !gq_act_set(&ax, x, S, D);
    if (ok) { gq_matmul_act(ig, &ax, G, GQ_Q2_0, I); gq_matmul_act(iu, &ax, U, GQ_Q2_0, I); }
    if (ok) {
        for (size_t i = 0; i < nh; i++) ih[i] = (ig[i] / (1.0f + expf(-ig[i]))) * iu[i];
        ok = !gq_act_set(&ah, ih, S, I);
    }
    if (ok) gq_matmul_act(iy, &ah, W, GQ_Q2_0, D);
    ok = ok && !gq_matmul_f32(fg, x, G, GQ_Q2_0, S, D, I)
            && !gq_matmul_f32(fu, x, U, GQ_Q2_0, S, D, I);
    if (ok) {
        for (size_t i = 0; i < nh; i++) fh[i] = (fg[i] / (1.0f + expf(-fg[i]))) * fu[i];
        ok = !gq_matmul_f32(fy, fh, W, GQ_Q2_0, S, I, D);
    }
    if (!ok) { fprintf(stderr, "FAIL S=%d CPU reference setup\n", S); return 1; }

    ColiCudaTensor *tg = NULL, *tu = NULL, *td = NULL;
    CcBuf *tab = NULL, *xb = NULL, *idsb = NULL, *wb = NULL, *hidb = NULL, *yb = NULL;
    CcExpert expert; memset(&expert, 0, sizeof expert);
    int *ids = (int *)malloc((size_t)S * sizeof(int));
    float *route_w = (float *)malloc((size_t)S * sizeof(float));
    int cuda_ok = ids && route_w;
    if (!cuda_ok) fprintf(stderr, "FAIL S=%d route allocation\n", S);
    if (cuda_ok) {
        for (int s = 0; s < S; s++) { ids[s] = 0; route_w[s] = 1.0f; }
    }
    if (cuda_ok && !coli_cuda_tensor_upload(&tg, G, NULL, fmt, D, I, 0)) { fprintf(stderr, "FAIL S=%d gate upload\n", S); cuda_ok = 0; }
    if (cuda_ok && !coli_cuda_tensor_upload(&tu, U, NULL, fmt, D, I, 0)) { fprintf(stderr, "FAIL S=%d up upload\n", S); cuda_ok = 0; }
    if (cuda_ok && !coli_cuda_tensor_upload(&td, W, NULL, fmt, I, D, 0)) { fprintf(stderr, "FAIL S=%d down upload\n", S); cuda_ok = 0; }
    if (cuda_ok && !cc_tensor_dev(tg, &expert.g, &expert.gs, NULL)) { fprintf(stderr, "FAIL S=%d gate device view\n", S); cuda_ok = 0; }
    if (cuda_ok && !cc_tensor_dev(tu, &expert.u, &expert.us, NULL)) { fprintf(stderr, "FAIL S=%d up device view\n", S); cuda_ok = 0; }
    if (cuda_ok && !cc_tensor_dev(td, &expert.d, &expert.ds, NULL)) { fprintf(stderr, "FAIL S=%d down device view\n", S); cuda_ok = 0; }
    if (cuda_ok) {
        tab = cc_buf(sizeof expert, CC_UP); xb = cc_buf(nx * sizeof(float), CC_UP);
        idsb = cc_buf((size_t)S * sizeof(int), CC_UP); wb = cc_buf((size_t)S * sizeof(float), CC_UP);
        hidb = cc_buf(cc_moe_hid_floats(S, K, D, I, E) * sizeof(float), CC_DEV);
        yb = cc_buf(ny * sizeof(float), CC_DOWN);
        cuda_ok = tab && xb && idsb && wb && hidb && yb;
        if (!cuda_ok) fprintf(stderr, "FAIL S=%d chain buffer allocation\n", S);
    }
    if (cuda_ok) {
        memcpy(cc_ptr(tab), &expert, sizeof expert); memcpy(cc_ptr(xb), x, nx * sizeof(float));
        memcpy(cc_ptr(idsb), ids, (size_t)S * sizeof(int)); memcpy(cc_ptr(wb), route_w, (size_t)S * sizeof(float));
        CcMoe p; memset(&p, 0, sizeof p);
        p.S = S; p.K = K; p.D = D; p.I = I; p.fmt = fmt; p.fmt_d = fmt; p.parts = 3; p.E = E;
        p.x_row = D; p.i_row = K; p.w_row = K; p.y_row = D; p.h_off = -1;
        if (!cc_moe(tab, xb, idsb, wb, hidb, yb, NULL, &p)) { fprintf(stderr, "FAIL S=%d cc_moe\n", S); cuda_ok = 0; }
        else if (!cc_finish()) { fprintf(stderr, "FAIL S=%d cc_finish after moe\n", S); cuda_ok = 0; }
    }
    float *ch = NULL, *cy = NULL;
    if (cuda_ok) { ch = (float *)malloc(nh * sizeof(float)); cy = (float *)malloc(ny * sizeof(float));
        cuda_ok = ch && cy; }
    /* Read the hidden and output regions through the chain's synchronous read API. */
    if (cuda_ok) {
        const ColiCudaChainOps *ops = cc_ops();
        cuda_ok = ops && ops->read(hidb, 0, ch, nh * sizeof(float));
        if (!cuda_ok) fprintf(stderr, "FAIL S=%d read CUDA hidden\n", S);
        if (cuda_ok) {
            cuda_ok = ops->read(yb, 0, cy, ny * sizeof(float));
            if (!cuda_ok) fprintf(stderr, "FAIL S=%d read CUDA output\n", S);
        }
    }
    if (!cuda_ok) fprintf(stderr, "FAIL S=%d physical CUDA chain execution / transfer failed\n", S);
    else {
        Metrics m;
        printf("case S=%d D=%d I=%d fmt=%d scale_profile=%s\n", S, D, I, fmt,
               test_scale_profile ? "test_cuda_bd (0x24..0x26)" : "stress (0x2c)");
        m = metrics(ih, fh, nh); print_metrics("CPU integer vs float hidden", m);
        m = metrics(iy, fy, ny); print_metrics("CPU integer vs float output", m);
        m = metrics(ch, ih, nh); print_metrics("CUDA hidden vs integer", m);
        m = metrics(ch, fh, nh); print_metrics("CUDA hidden vs float", m);
        m = metrics(cy, iy, ny); print_metrics("CUDA output vs integer", m);
        m = metrics(cy, fy, ny); print_metrics("CUDA output vs float", m);
        Metrics rh = metrics(ch, S > 8 ? fh : ih, nh), ry = metrics(cy, S > 8 ? fy : iy, ny);
        double term_rel = S > 8 ? output_term_relmax(cy, fy, fh, W, rb_d, S, D, I) : NAN;
        if (S <= 8) printf("  CPU_integer_comparison_rel_max_ref_plus_1e-3=%.9g (test_cuda_bd decode context: 2e-3)\n", ry.rel_max);
        else printf("  adapted_final_output_term_norm=%.9g (sum |deq(Wd)*CPU_float_hidden| + 1e-6; diagnostic only)\n", term_rel);
        printf("  compared_hidden_rel_ref_plus_1e-3=%.9g compared_output_rel_ref_plus_1e-3=%.9g\n", rh.rel_max, ry.rel_max);
        int finite = 1;
        for (size_t i = 0; i < nh; i++) finite &= isfinite(ch[i]) && isfinite(ih[i]) && isfinite(fh[i]);
        for (size_t i = 0; i < ny; i++) finite &= isfinite(cy[i]) && isfinite(iy[i]) && isfinite(fy[i]);
        if (!finite || (S > 8 && !isfinite(term_rel))) {
            fprintf(stderr, "FAIL S=%d non-finite diagnostic value\n", S); cuda_ok = 0;
        }
    }
    if (yb) cc_free(yb);
    if (hidb) cc_free(hidb);
    if (wb) cc_free(wb);
    if (idsb) cc_free(idsb);
    if (xb) cc_free(xb);
    if (tab) cc_free(tab);
    if (tg) coli_cuda_tensor_free(tg);
    if (tu) coli_cuda_tensor_free(tu);
    if (td) coli_cuda_tensor_free(td);
    cc_finish();
    free(ch); free(cy); free(ids); free(route_w); gq_act_free(&ax); gq_act_free(&ah);
    free(G); free(U); free(W); free(x); free(ig); free(iu); free(ih); free(iy); free(fg); free(fu); free(fh); free(fy);
    return cuda_ok ? 0 : 1;
}

int main(void) {
    int device = 0;
    if (!coli_cuda_init(&device, 1)) {
        fprintf(stderr, "FAIL physical CUDA required: coli_cuda.dll/device 0 unavailable\n");
        return 2;
    }
    if (!cc_available() || !cc_init(device) || !cc_ready()) {
        fprintf(stderr, "FAIL physical CUDA required: chain API/device %d unavailable\n", device);
        coli_cuda_shutdown();
        return 2;
    }
    printf("physical CUDA diagnostic: loaded coli_cuda.dll, device=%d\n", device);
    int rc = run_case(1, 256, 128, 0);
    if (!rc) rc = run_case(21, 2560, 640, 1);
    if (!rc) rc = run_case(21, 2560, 640, 0);
    cc_shutdown(); coli_cuda_shutdown();
    if (!rc) puts("test_q2_numeric_probe: completed (physical CUDA; finite-value checks only, metrics are diagnostics)");
    return rc;
}
