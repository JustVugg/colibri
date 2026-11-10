/* Kimi K3's two ops on the CUDA chain (cuda_k3c.cuh) against kimi_k3.c's arithmetic:
 *   ares  res_mix: the softmax over nb block snapshots and the prefix of their weighted RMS
 *         scores (the dot and the squares in double), then the mix; nb 0 (the prefix alone, its
 *         bits), 1, 3 and 15, Kimi K3's width (7168) and a small one, rows with offsets and strides;
 *   situ  SiTU-GLU's b1*tanh(g/b1)*sigmoid(g)*b2*tanh(u/b2), Kimi K3's betas (4, 25), small and
 *         large inputs, offsets.
 * CUDA's expf and tanhf are within 2 ulp of the CPU's, and the CPU may fuse the mix's multiply and
 * add: the bound is 2e-6 relative to the largest output. make CUDA=1 cuda-k3c-check */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../backend_cuda.h"
#include "../cuda_chain.h"

static const ColiCudaChainOps *T;
static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; }
static unsigned g_rng = 4242;
static unsigned rnd() { g_rng = g_rng * 1103515245u + 12345u; return (g_rng >> 8) & 0xffffff; }
static float frnd() { return ((int)(rnd() % 2001) - 1000) / 1000.0f; }
static std::vector<float> fvec(size_t n, float scale) { std::vector<float> v(n); for (auto &x : v) x = frnd() * scale; return v; }
static CcBuf *upf(const std::vector<float> &v) {
    CcBuf *b = T->buf(v.size() * 4 ? v.size() * 4 : 4, CC_DEV);
    if (!b || !T->begin() || !T->write(b, 0, v.data(), v.size() * 4) || !T->submit(1)) { printf("  FAIL upload\n"); fails++; }
    return b;
}
static std::vector<float> down(CcBuf *b, size_t n) {
    std::vector<float> v(n);
    if (!T->read(b, 0, v.data(), n * 4)) { printf("  FAIL read back\n"); fails++; }
    return v;
}
static double rel_to_max(const float *a, const float *b, size_t n) {
    double w = 0, m = 0;
    for (size_t i = 0; i < n; i++) { if (!std::isfinite(a[i])) return 1e30; w = fmax(w, fabs((double)a[i] - b[i])); m = fmax(m, fabs((double)b[i])); }
    return m > 0 ? w / m : w;
}

/* kimi_k3.c's res_mix, one row */
static void ref_mix(float *out, const float *prefix, const float *bres, int nb, int D, const float *sw, float eps) {
    const float *v[16]; float sc[16];
    for (int e = 0; e < nb; e++) v[e] = bres + (size_t)e * D;
    v[nb] = prefix;
    for (int e = 0; e <= nb; e++) {
        double ms = 0, dot = 0;
        for (int d = 0; d < D; d++) { double x = v[e][d]; ms += x * x; dot += x * (double)sw[d]; }
        sc[e] = (float)(dot / sqrt(ms / D + eps));
    }
    float m = sc[0];
    for (int e = 1; e <= nb; e++) if (sc[e] > m) m = sc[e];
    float s = 0;
    for (int e = 0; e <= nb; e++) { sc[e] = expf(sc[e] - m); s += sc[e]; }
    for (int e = 0; e <= nb; e++) sc[e] /= s;
    for (int d = 0; d < D; d++) { float a = 0; for (int e = 0; e <= nb; e++) a += sc[e] * v[e][d]; out[d] = a; }
}
static void test_ares(int S, int D, int nb, int x_pad, int b_pad) {
    const int NBX = 15, x_row = D + x_pad, b_row = NBX * D + b_pad, w_off = 3, y_row = D + 5;
    std::vector<float> x = fvec((size_t)S * x_row + 7, 1.f), blk = fvec((size_t)S * b_row + 9, 1.f), prm = fvec((size_t)w_off + D, 0.2f);
    CcBuf *bx = upf(x), *bb = upf(blk), *bp = upf(prm), *by = T->buf((size_t)(S * y_row + 2) * 4, CC_DEV);
    CcAres a = {S, D, nb, 7, x_row, 9, b_row, w_off, 2, y_row, 1e-5f};
    int ok = by && T->begin() && T->ares_mix(bx, bb, bp, by, &a) && T->submit(1);
    std::vector<float> y = down(by, (size_t)S * y_row + 2), want(D);
    double worst = 0; int exact = 1;
    for (int s = 0; s < S && ok; s++) {
        ref_mix(want.data(), x.data() + 7 + (size_t)s * x_row, blk.data() + 9 + (size_t)s * b_row, nb, D, prm.data() + w_off, 1e-5f);
        const float *got = y.data() + 2 + (size_t)s * y_row;
        worst = fmax(worst, rel_to_max(got, want.data(), D));
        if (nb == 0) exact &= !memcmp(got, x.data() + 7 + (size_t)s * x_row, (size_t)D * 4);
    }
    char what[160];
    snprintf(what, sizeof what, "ares mix S=%d D=%d nb=%d (pads %d, %d): rel %.2e%s", S, D, nb, x_pad, b_pad, worst,
             nb == 0 ? (exact ? ", the prefix's bits" : ", NOT the prefix's bits") : "");
    ck(ok && worst < 2e-6 && exact, what);
    T->free(bx); T->free(bb); T->free(bp); T->free(by);
}
static float ref_situ(float g, float u, float b1, float b2) { return b1 * tanhf(g / b1) * (1.f / (1.f + expf(-g))) * b2 * tanhf(u / b2); }
static void test_situ(int n, float scale) {
    std::vector<float> g = fvec((size_t)n + 4, scale), u = fvec((size_t)n + 6, scale * 3);
    CcBuf *bg = upf(g), *bu = upf(u), *by = T->buf((size_t)(n + 1) * 4, CC_DEV);
    CcSitu p = {n, 4, 6, 1, 4.f, 25.f};
    int ok = by && T->begin() && T->situ(bg, bu, by, &p) && T->submit(1);
    std::vector<float> y = down(by, (size_t)n + 1), want(n);
    for (int i = 0; i < n; i++) want[i] = ref_situ(g[4 + i], u[6 + i], 4.f, 25.f);
    double r = rel_to_max(y.data() + 1, want.data(), n);
    char what[96];
    snprintf(what, sizeof what, "situ n=%d inputs to %g: rel %.2e", n, scale, r);
    ck(ok && r < 2e-6, what);
    T->free(bg); T->free(bu); T->free(by);
}

int main(void) {
    T = coli_cuda_chain_ops();
    if (!T || !T->init(0) || !T->ready()) { printf("cuda-k3c-check: no CUDA chain on this machine\n"); return 1; }
    if (T->size < offsetof(ColiCudaChainOps, situ) + sizeof T->situ || !T->ares_mix || !T->situ) {
        printf("cuda-k3c-check: the module has no Kimi K3 ops\n"); return 1;
    }
    printf("ares (kimi_k3.c's res_mix)\n");
    test_ares(1, 7168, 0, 0, 0);
    test_ares(3, 7168, 1, 5, 3);
    test_ares(5, 7168, 3, 0, 0);
    test_ares(2, 7168, 15, 1, 2);
    test_ares(9, 128, 7, 3, 1);
    printf("situ (SiTU-GLU, betas 4 and 25)\n");
    test_situ(1, 1.f);
    test_situ(3072 * 3, 3.f);
    test_situ(100003, 60.f);
    test_situ(77, 1e-4f);
    printf(fails ? "cuda-k3c-check: %d FAILED\n" : "cuda-k3c-check: all passed\n", fails);
    return fails ? 1 : 0;
}
