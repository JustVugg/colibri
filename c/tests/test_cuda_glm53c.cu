/* GLM-5.3's layer ops on the CUDA chain (cuda_glm53c.cuh) against the CPU's references, the cases
 * and bounds tests/test_vk_chain.c holds the Vulkan shaders to:
 *   kda  glm53.c's KDA layer around delta_attention.h's coli_kda_step (the decay, beta and
 *        output gate as glm53.c makes them), S rows over two calls (the state and the window
 *        carried on the device between them); the block of rows kernel (CC_KDA_ROWS rows and
 *        more in one call) against calls of fewer rows, bit for bit, GLM's form and Kimi K3's.
 * The kernels round the CPU's steps in its order and contract nothing into an FMA (the reference is
 * built without contraction too): with an exp correctly rounded in their place the outputs are the
 * CPU's within 5e-7. CUDA's expf, within 2 ulp, gives 2-5e-5 on the outputs once the recurrence and
 * the output norm have carried it: the outputs' bound is 1e-4, the state's 1e-5.
 * make CUDA=1 cuda-glm53c-check (the references in tests/cuda_glm53c_ref.c) */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../backend_cuda.h"
#include "../cuda_chain.h"
extern "C" {   /* tests/cuda_glm53c_ref.c */
int glm53c_ref_kda_scratch(int H, int KD, int VD);
int glm53c_ref_kda_step(float *out, float *state, float *window, const float *qkv, const float *conv_w, const float *gate,
                        const float *beta, int H, int KD, int VD, int K, float neps, float *scratch);
}

static const ColiCudaChainOps *T;
static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; }
static unsigned g_rng = 12345;
static unsigned rnd() { g_rng = g_rng * 1103515245u + 12345u; return (g_rng >> 8) & 0xffffff; }
static float frnd() { return ((int)(rnd() % 2001) - 1000) / 1000.0f; }
static std::vector<float> fvec(size_t n, float scale) { std::vector<float> v(n); for (auto &x : v) x = frnd() * scale; return v; }
static CcBuf *up(const void *v, size_t bytes) {
    CcBuf *b = T->buf(bytes ? bytes : 4, CC_DEV);
    if (!b || !T->begin() || !T->write(b, 0, v, bytes) || !T->submit(1)) { printf("  FAIL upload\n"); fails++; }
    return b;
}
static CcBuf *upf(const std::vector<float> &v) { return up(v.data(), v.size() * 4); }
template <class X> static std::vector<X> down(CcBuf *b, size_t n) {
    std::vector<X> v(n);
    if (!T->read(b, 0, v.data(), n * sizeof(X))) { printf("  FAIL read back\n"); fails++; }
    return v;
}
static double relerr(const float *a, const float *b, size_t n, double floor_) {
    double w = 0;
    for (size_t i = 0; i < n; i++) {
        if (std::isinf(a[i]) || std::isinf(b[i])) { if (a[i] != b[i]) return 1e9; continue; }
        double d = fabs((double)a[i] - b[i]) / (fabs((double)b[i]) > floor_ ? fabs((double)b[i]) : floor_);
        if (!(d <= w)) w = d;
    }
    return w;
}
static int bad(const float *a, size_t n) { for (size_t i = 0; i < n; i++) if (!std::isfinite(a[i])) return 1; return 0; }
static float sigm(float v) { if (v >= 0) { float d = expf(-v); return 1.f / (1.f + d); } float g = expf(v); return g / (1.f + g); }

/* the device's inputs of one KDA layer step: q, k, v in three blocks [3][S][P] (in_part = S*P) */
struct Kda {
    int H, KD, VD, P, C, S, K;
    std::vector<float> qkv, f, b, g, taps, win, state, prm;
};
static Kda kda_make(int H, int KD, int K, int S, float xs) {
    Kda k; k.H = H; k.KD = KD; k.VD = KD; k.P = H * KD; k.C = 3 * k.P; k.S = S; k.K = K;
    k.qkv = fvec((size_t)S * k.C, xs); k.f = fvec((size_t)S * k.P, 1.f); k.b = fvec((size_t)S * H, 2.f); k.g = fvec((size_t)S * k.P, 2.f);
    k.taps = fvec((size_t)k.C * K, 0.7f); k.win = fvec((size_t)k.C * K, 1.f); k.state = fvec((size_t)H * KD * k.VD, 0.3f);
    k.prm.resize((size_t)H + k.P + k.VD);
    for (int i = 0; i < H; i++) k.prm[i] = frnd() * 0.5f;
    for (int i = 0; i < k.P; i++) k.prm[H + i] = frnd() * 0.5f;
    for (int i = 0; i < k.VD; i++) k.prm[H + k.P + i] = 0.5f + (rnd() % 100) / 100.f;
    return k;
}
/* the device's run of the S rows in calls of `per` rows; y, the state and the window back */
static int kda_run(const Kda &k, int per, int flags, std::vector<float> &y, std::vector<float> &st, std::vector<float> &win) {
    const int S = k.S, P = k.P, C = k.C;
    const float lb = -5.f, neps = 1e-6f, eps = 1e-5f;
    std::vector<float> blk((size_t)3 * S * P);
    for (int s = 0; s < S; s++) for (int part = 0; part < 3; part++)
        memcpy(&blk[(size_t)part * S * P + (size_t)s * P], &k.qkv[(size_t)s * C + (size_t)part * P], (size_t)P * 4);
    CcBuf *ib = upf(blk), *tb = upf(k.taps), *wb = upf(k.win), *sb = upf(k.state);
    CcBuf *fb = upf(k.f), *bb = upf(k.b), *gb = upf(k.g), *pb = upf(k.prm);
    CcBuf *mb = T->buf((size_t)S * C * 4, CC_DEV), *yb = T->buf((size_t)S * P * 4, CC_DEV);
    int ok = 1;
    for (int s0 = 0; s0 < S && ok; s0 += per) {
        const int n = S - s0 < per ? S - s0 : per;
        CcKdaConv cp = {n, C, k.K, P, s0 * P, P, S * P, 0, C, 0, 0};
        CcKdaRec rp = {n, k.H, k.VD, P, 0, C, s0 * P, P, s0 * k.H, k.H, s0 * P, P, s0 * P, P, 0, 0, lb, neps, eps};
        ok = T->begin() && T->kda_conv(ib, tb, wb, mb, &cp) && T->kda_rec(k.KD, mb, fb, bb, gb, pb, sb, yb, &rp, flags) && T->submit(1);
    }
    y = down<float>(yb, (size_t)S * P); st = down<float>(sb, (size_t)k.H * k.KD * k.VD); win = down<float>(wb, (size_t)C * k.K);
    CcBuf *all[] = {ib, tb, wb, sb, fb, bb, gb, pb, mb, yb};
    for (CcBuf *x : all) T->free(x);
    return ok;
}
static void test_kda(int H, int KD, int K, float xs) {
    Kda k = kda_make(H, KD, K, 5, xs);
    const int S = k.S, P = k.P, C = k.C, VD = k.VD;
    const float lb = -5.f, neps = 1e-6f, eps = 1e-5f;
    /* reference: glm53.c's kda_layer row by row */
    std::vector<float> rs = k.state, rw = k.win, ref((size_t)S * P), core(P), dk(P), bt(H);
    std::vector<float> scratch((size_t)glm53c_ref_kda_scratch(H, KD, VD));
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H; h++) for (int d = 0; d < KD; d++) {
            const int i = h * KD + d; const float sv = expf(k.prm[h]) * (k.f[s * P + i] + k.prm[H + i]);
            dk[i] = lb * sigm(sv);
        }
        for (int h = 0; h < H; h++) bt[h] = sigm(k.b[s * H + h]);
        glm53c_ref_kda_step(core.data(), rs.data(), rw.data(), &k.qkv[(size_t)s * C], k.taps.data(), dk.data(), bt.data(), H, KD, VD, K, neps,
                            scratch.data());
        for (int h = 0; h < H; h++) {
            float sq = 0; for (int d = 0; d < VD; d++) sq += core[h * VD + d] * core[h * VD + d];
            const float inv = 1.f / sqrtf(sq / VD + eps);
            for (int d = 0; d < VD; d++) ref[s * P + h * VD + d] = core[h * VD + d] * inv * k.prm[H + P + d] * sigm(k.g[s * P + h * VD + d]);
        }
    }
    std::vector<float> y, st, win;
    const int ok = kda_run(k, 3, 0, y, st, win);
    const double e = relerr(y.data(), ref.data(), y.size(), 1e-3), es = relerr(st.data(), rs.data(), st.size(), 1e-3),
                 ew = relerr(win.data(), rw.data(), win.size(), 1e-3);
    char what[160];
    snprintf(what, sizeof what, "kda H %d KD %d K %d xs %g: out %.2e state %.2e window %.2e", H, KD, K, xs, e, es, ew);
    ck(ok && e < 1e-4 && es < 1e-5 && ew == 0 && !bad(y.data(), y.size()), what);
}
/* the rows kernel (a call of S >= CC_KDA_ROWS rows) against calls of fewer rows: the same bits */
static void test_kda_rows(int H, int KD, int S, int flags) {
    Kda k = kda_make(H, KD, 4, S, 1.f);
    std::vector<float> y1, st1, w1, y2, st2, w2;
    const int ok = kda_run(k, S, flags, y1, st1, w1) && kda_run(k, CC_KDA_ROWS - 1, flags, y2, st2, w2);
    const int same = ok && !memcmp(y1.data(), y2.data(), y1.size() * 4) && !memcmp(st1.data(), st2.data(), st1.size() * 4) &&
                     !memcmp(w1.data(), w2.data(), w1.size() * 4) && !bad(y1.data(), y1.size());
    char what[160];
    snprintf(what, sizeof what, "kda rows H %d KD %d S %d flags %d: the row kernel's bits", H, KD, S, flags);
    ck(same, what);
}

int main(void) {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { printf("no CUDA device: skipped\n"); return 0; }
    T = coli_cuda_chain_ops();
    if (!T || T->size < offsetof(ColiCudaChainOps, kda_rec) + sizeof T->kda_rec || !T->kda_rec) { printf("FAIL: the backend has no GLM-5.3 ops\n"); return 1; }
    if (!T->init(0) || !T->ready()) { printf("FAIL: no chain context\n"); return 1; }
    printf("kda\n");
    test_kda(2, 64, 4, 1.f); test_kda(3, 128, 4, 1.f); test_kda(2, 128, 4, 0.01f); test_kda(4, 64, 3, 0.05f);
    test_kda_rows(3, 128, 21, 0); test_kda_rows(2, 64, 8, 0); test_kda_rows(3, 128, 19, 2); test_kda_rows(2, 64, 30, 3);
    T->shutdown();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
