/* Inkling's and Inkling-2's ops on the CUDA chain (cuda_inkc.cuh) against the CPU's arithmetic
 * (inkling.c's sconv_apply and attention, inkling2.h's attention residuals, L2 clip and dynamic
 * convolution), as tests/test_vk_chain.c holds the Vulkan chain's:
 *   sconv   the short convolution over two steps with its ring carried, and the divide
 *   relattn the relative bias, tau, a window over a wrapped ring, the step's own rows, the sink
 *   mix     nb 0, 8 and 15 block states, Inkling-2-Small's width
 *   clip    with and without the learned scale, a threshold that binds and one that does not
 *   dconv   two steps, the state carried, 1 to 9 taps
 *   relattn a decode step's split over its positions (a block a tile of 128, then joined), up to 1500
 *   loopgate Inkling-2-Mini's loop gate (CC_EW_LOOPGATE), in place too, and the backend's CC_INK2_* bits
 * make CUDA=1 cuda-inkc-check */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../backend_cuda.h"
#include "../cuda_chain.h"

static const ColiCudaChainOps *T;
static int fails;
static unsigned g_rng = 777;
static unsigned rnd() { g_rng = g_rng * 1103515245u + 12345u; return (g_rng >> 8) & 0xffffff; }
static float frnd() { return ((int)(rnd() % 2001) - 1000) / 1000.0f; }
static std::vector<float> fvec(size_t n, float scale) { std::vector<float> v(n); for (auto &x : v) x = frnd() * scale; return v; }
static CcBuf *upf(const std::vector<float> &v) {
    CcBuf *b = T->buf(v.empty() ? 4 : v.size() * 4, CC_DEV);
    if (!b || !T->begin() || !T->write(b, 0, v.data(), v.size() * 4) || !T->submit(1)) { printf("  FAIL upload\n"); fails++; }
    return b;
}
static std::vector<float> down(CcBuf *b, size_t off, size_t n) {
    std::vector<float> v(n);
    if (!T->read(b, off, v.data(), n * 4)) { printf("  FAIL read back\n"); fails++; }   /* off in floats */
    return v;
}
static double rel_to_max(const float *a, const float *b, size_t n) {
    double w = 0, m = 0;
    for (size_t i = 0; i < n; i++) { if (!std::isfinite(a[i])) return 1e30; w = fmax(w, fabs((double)a[i] - b[i])); m = fmax(m, fabs((double)b[i])); }
    return m > 0 ? w / m : w;
}
static void check(double e, double bound, const char *what) {
    printf("  %s %s: %.2e\n", e < bound ? "ok  " : "FAIL", what, e);
    if (!(e < bound)) fails++;
}

static void test_sconv(int C, int CK, int S1, int S2) {
    int S = S1 + S2, P = CK - 1, xo = 3, wo = 5, ro = 2;
    std::vector<float> x = fvec((size_t)xo + (size_t)S * C, 1.f), w = fvec((size_t)wo + (size_t)C * CK, 0.5f), ring = fvec((size_t)ro + (size_t)C * P + 1, 1.f);
    std::vector<float> ref((size_t)S * C);
    for (int c = 0; c < C; c++) {
        std::vector<float> col(P + S);
        for (int j = 0; j < P; j++) col[j] = ring[ro + c * P + j];
        for (int t = 0; t < S; t++) col[P + t] = x[xo + (size_t)t * C + c];
        for (int t = 0; t < S; t++) {
            float acc = 0.f;
            for (int j = 0; j < CK; j++) acc += w[wo + (size_t)c * CK + j] * col[t + j];
            ref[(size_t)t * C + c] = (acc + col[P + t]) / 2.5f;
        }
    }
    CcBuf *bx = upf(x), *bw = upf(w), *br = upf(ring);
    CcSconv p1 = {0, S1, C, CK, xo, C, wo, ro, 0, 1.f}, p2 = {0, S2, C, CK, xo + S1 * C, C, wo, ro, 0, 1.f};
    CcSconv dv = {2, 0, 0, 0, xo, 0, 0, 0, S * C, 2.5f};
    int ok = T->begin() && T->sconv(bx, bw, br, &p1) && T->submit(1) && T->begin() && T->sconv(bx, bw, br, &p2) &&
             T->sconv(bx, NULL, NULL, &dv) && T->submit(1);
    std::vector<float> y = down(bx, xo, (size_t)S * C);
    char what[96]; snprintf(what, sizeof what, "sconv %d channels, %d taps, rows %d + %d", C, CK, S1, S2);
    check(ok ? rel_to_max(y.data(), ref.data(), ref.size()) : 1e30, 2e-6, what);
    T->free(bx); T->free(bw); T->free(br);
}

static void test_relattn(int S, int pos_base, int window, int cap, int hd, int dr, int ext, int sink) {
    int H = 4, KVH = 2, kvd = KVH * hd, Tn = pos_base + S, koff = 32, roff = 16, poff = 24, toff = 8, soff = toff + S + 3;
    std::vector<float> q = fvec((size_t)S * H * hd, 1.f), kt = fvec((size_t)Tn * kvd, 1.f), vt = fvec((size_t)Tn * kvd, 1.f);
    std::vector<float> r = fvec((size_t)S * H * dr, 0.5f), relp = fvec((size_t)dr * ext, 0.5f), snk = fvec((size_t)S * H, 3.f), tau(S);
    for (int s = 0; s < S; s++) tau[s] = 1.f + (rnd() % 1000) / 2000.f;
    float scale = 1.f / sqrtf((float)hd);
    std::vector<float> ref((size_t)S * H * hd);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        int kvh = h / (H / KVH), qpos = pos_base + s, t0 = window > 0 && qpos - window + 1 > 0 ? qpos - window + 1 : 0, n = qpos - t0 + 1;
        std::vector<double> sc(n); double mx = -1e300, sum = 0;
        for (int j = 0; j < n; j++) {
            int t = t0 + j, dist = qpos - t;
            double a = 0; for (int d = 0; d < hd; d++) a += (double)q[((size_t)s * H + h) * hd + d] * kt[(size_t)t * kvd + kvh * hd + d];
            double b = 0; if (dist < ext) for (int k = 0; k < dr; k++) b += (double)r[((size_t)s * H + h) * dr + k] * relp[k * ext + dist];
            sc[j] = tau[s] * (a * scale + b); if (sc[j] > mx) mx = sc[j];
        }
        double sk = snk[s * H + h];
        if (sink && sk > mx) mx = sk;
        if (sink) sum = exp(sk - mx);
        for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
        for (int d = 0; d < hd; d++) {
            double a = 0; for (int j = 0; j < n; j++) a += sc[j] / sum * vt[(size_t)(t0 + j) * kvd + kvh * hd + d];
            ref[((size_t)s * H + h) * hd + d] = (float)a;
        }
    }
    size_t nr = (size_t)koff + (size_t)KVH * cap * hd;
    std::vector<float> kr(nr), vr(nr);
    for (int t = 0; t < pos_base; t++) for (int h = 0; h < KVH; h++) {
        memcpy(&kr[koff + ((size_t)h * cap + t % cap) * hd], &kt[(size_t)t * kvd + h * hd], hd * 4);
        memcpy(&vr[koff + ((size_t)h * cap + t % cap) * hd], &vt[(size_t)t * kvd + h * hd], hd * 4);
    }
    size_t ks = 4, vs = ks + (size_t)S * kvd, nsc = vs + (size_t)S * kvd;
    std::vector<float> scr(nsc), rb(roff + (size_t)S * H * dr), pb(poff + (size_t)dr * ext), tb((size_t)soff + (size_t)S * H);
    memcpy(&scr[ks], &kt[(size_t)pos_base * kvd], (size_t)S * kvd * 4); memcpy(&scr[vs], &vt[(size_t)pos_base * kvd], (size_t)S * kvd * 4);
    memcpy(&rb[roff], r.data(), r.size() * 4); memcpy(&pb[poff], relp.data(), relp.size() * 4);
    memcpy(&tb[toff], tau.data(), S * 4); memcpy(&tb[soff], snk.data(), snk.size() * 4);
    CcBuf *bq = upf(q), *bk = upf(kr), *bv = upf(vr), *bs = upf(scr), *brr = upf(rb), *bp = upf(pb), *bt = upf(tb);
    CcBuf *bo = T->buf((size_t)S * H * hd * 4, CC_DEV);
    CcRelAttn p = {S, H, KVH, hd, pos_base, cap, window, ext, dr, 0, H * hd, 0, H * hd, koff, koff, (int)ks, (int)vs, kvd,
                   roff, H * dr, poff, toff, scale, sink, sink ? soff : 0};
    int ok = T->begin() && T->relattn(bq, bk, bv, bo, bs, brr, bp, bt, &p) && T->submit(1);
    std::vector<float> y = down(bo, 0, ref.size());
    char what[128]; snprintf(what, sizeof what, "relattn S %d pos %d window %d cap %d hd %d d_rel %d ext %d sink %d", S, pos_base, window, cap, hd, dr, ext, sink);
    check(ok ? rel_to_max(y.data(), ref.data(), ref.size()) : 1e30, 2e-5, what);
    CcBuf *all[] = {bq, bk, bv, bs, brr, bp, bt, bo};
    for (CcBuf *b : all) T->free(b);
}

static void test_mix(int S, int D, int nb) {
    int bro = (nb + 2) * D, xo = 5, bo = 7, wo = 3, yo = 2;
    float eps = 1e-6f, scale = 1.f / sqrtf((float)D);
    std::vector<float> x = fvec((size_t)xo + (size_t)S * D, 2.f), blk = fvec((size_t)bo + (size_t)S * bro, 2.f), q = fvec((size_t)wo + D, 1.f);
    std::vector<float> ref((size_t)S * D);
    for (int s = 0; s < S; s++) {
        double w[16], rr[16], mx = -1e300, sum = 0;
        for (int e = 0; e <= nb; e++) {
            const float *v = e < nb ? &blk[bo + (size_t)s * bro + (size_t)e * D] : &x[xo + (size_t)s * D];
            double ms = 0, dt = 0;
            for (int d = 0; d < D; d++) { ms += (double)v[d] * v[d]; dt += (double)v[d] * q[wo + d]; }
            rr[e] = 1.0 / sqrt(ms / D + eps); w[e] = dt * rr[e] * scale; if (w[e] > mx) mx = w[e];
        }
        for (int e = 0; e <= nb; e++) { w[e] = exp(w[e] - mx); sum += w[e]; }
        for (int d = 0; d < D; d++) {
            double a = 0;
            for (int e = 0; e <= nb; e++) a += w[e] / sum * (e < nb ? blk[bo + (size_t)s * bro + (size_t)e * D + d] : x[xo + (size_t)s * D + d]) * rr[e];
            ref[(size_t)s * D + d] = (float)a;
        }
    }
    CcBuf *bx = upf(x), *bb = upf(blk), *bw = upf(q), *by = T->buf(((size_t)yo + (size_t)S * D) * 4, CC_DEV);
    CcInk2Mix p = {S, D, nb, xo, D, bo, bro, wo, yo, D, eps, scale};
    int ok = T->begin() && T->ink2_mix(bx, bb, bw, by, &p) && T->submit(1);
    std::vector<float> y = down(by, yo, ref.size());
    char what[96]; snprintf(what, sizeof what, "ink2 mix S %d D %d nb %d", S, D, nb);
    check(ok ? rel_to_max(y.data(), ref.data(), ref.size()) : 1e30, 2e-5, what);
    T->free(bx); T->free(bb); T->free(bw); T->free(by);
}

static void test_clip(int rows, int per_row, int n, int has_w, float thr) {
    int xo = 4, row = per_row * n + 3, wo = 6, nseg = rows * per_row;
    std::vector<float> x = fvec((size_t)xo + (size_t)rows * row, 3.f), w = fvec((size_t)wo + (size_t)per_row * n, 2.f), ref = x;
    for (int g = 0; g < nseg; g++) {
        float *v = &ref[xo + (size_t)(g / per_row) * row + (size_t)(g % per_row) * n];
        if (has_w) for (int i = 0; i < n; i++) v[i] *= w[wo + (g % per_row) * n + i];
        double ss = 0; for (int i = 0; i < n; i++) ss += (double)v[i] * v[i];
        float f = thr / ((float)sqrt(ss) + 1e-6f);
        if (f < 1.f) for (int i = 0; i < n; i++) v[i] *= f;
    }
    CcBuf *bx = upf(x), *bw = upf(w);
    CcInk2Clip p = {nseg, n, per_row, xo, row, has_w, wo, thr};
    int ok = T->begin() && T->ink2_clip(bx, has_w ? bw : NULL, &p) && T->submit(1);
    std::vector<float> y = down(bx, 0, x.size());
    char what[96]; snprintf(what, sizeof what, "ink2 clip %d rows of %d x %d, scale %d, thr %g", rows, per_row, n, has_w, thr);
    check(ok ? rel_to_max(y.data(), ref.data(), ref.size()) : 1e30, 2e-6, what);
    T->free(bx); T->free(bw);
}

static void test_dconv(int KV, int hd, int K, int S1, int S2) {
    int C = KV * hd, P = K - 1, S = S1 + S2, xo = 3, dro = KV * K + 2, wo = 5, ro = 4;
    float fc = sqrtf((float)(2 * K + 1));
    std::vector<float> x = fvec((size_t)xo + (size_t)S * C, 1.f), dyn = fvec((size_t)S * dro, 0.5f), st = fvec((size_t)wo + (size_t)K * C, 0.5f);
    std::vector<float> ring = fvec((size_t)ro + (size_t)C * P + 1, 1.f), ref((size_t)S * C);
    for (int c = 0; c < C; c++) {
        int h = c / hd;
        std::vector<float> col(P + S);
        for (int j = 0; j < P; j++) col[j] = ring[ro + c * P + j];
        for (int t = 0; t < S; t++) col[P + t] = x[xo + (size_t)t * C + c];
        for (int t = 0; t < S; t++) {
            float acc = 0.f;
            for (int lag = 0; lag < K; lag++) acc += col[P + t - lag] * (st[wo + lag * C + c] + dyn[(size_t)t * dro + h * K + lag]);
            ref[(size_t)t * C + c] = (acc + col[P + t]) / fc;
        }
    }
    CcBuf *bx = upf(x), *bd = upf(dyn), *bw = upf(st), *br = upf(ring);
    CcInk2Dconv p1 = {S1, C, K, hd, xo, C, 0, dro, wo, ro, fc}, p2 = {S2, C, K, hd, xo + S1 * C, C, S1 * dro, dro, wo, ro, fc};
    int ok = T->begin() && T->ink2_dconv(bx, bd, bw, br, &p1) && T->submit(1) && T->begin() && T->ink2_dconv(bx, bd, bw, br, &p2) && T->submit(1);
    std::vector<float> y = down(bx, xo, ref.size());
    char what[96]; snprintf(what, sizeof what, "ink2 dconv %d heads of %d, %d taps, rows %d + %d", KV, hd, K, S1, S2);
    check(ok ? rel_to_max(y.data(), ref.data(), ref.size()) : 1e30, 2e-6, what);
    T->free(bx); T->free(bd); T->free(bw); T->free(br);
}

/* y = (1 - tanh(a)) c + tanh(b) e, ink2_loop_gate's last line; y the same buffer as c when inplace */
static void test_loopgate(int rows, int D, int inplace) {
    const int n = rows * D;
    std::vector<float> a = fvec(n, 3.f), b = fvec(n, 3.f), c = fvec(n, 2.f), e = fvec(n, 2.f), ref(n);
    for (int i = 0; i < n; i++) ref[i] = (1.f - tanhf(a[i])) * c[i] + tanhf(b[i]) * e[i];
    CcBuf *ba = upf(a), *bb = upf(b), *bc = upf(c), *be = upf(e), *by = inplace ? bc : T->buf((size_t)n * 4, CC_DEV);
    CcEw p = {CC_EW_LOOPGATE, n, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    int ok = T->begin() && T->ew(by, ba, bb, bc, be, &p) && T->submit(1);
    std::vector<float> y = down(by, 0, (size_t)n);
    char what[96]; snprintf(what, sizeof what, "loop gate %d x %d%s", rows, D, inplace ? ", in place" : "");
    check(ok ? rel_to_max(y.data(), ref.data(), ref.size()) : 1e30, 1e-6, what);
    T->free(ba); T->free(bb); T->free(bc); T->free(be); if (!inplace) T->free(by);
}

int main(void) {
    T = coli_cuda_chain_ops();
    if (!T || !T->init(0) || !T->ready()) { printf("cuda-inkc-check: no CUDA chain on this machine\n"); return 1; }
    if (T->size < offsetof(ColiCudaChainOps, ink2_dconv) + sizeof T->ink2_dconv || !T->sconv || !T->relattn) {
        printf("cuda-inkc-check: the module has no Inkling ops\n"); return 1;
    }
    printf("sconv\n");
    test_sconv(37, 4, 3, 1); test_sconv(6144, 4, 1, 1); test_sconv(9, 1, 2, 2); test_sconv(5, 9, 6, 3);
    printf("relattn\n");
    test_relattn(1, 0, 0, 64, 16, 4, 32, 0);
    test_relattn(5, 200, 0, 300, 32, 16, 64, 0);
    test_relattn(1, 40, 16, 16, 32, 4, 16, 1);
    test_relattn(20, 30, 16, 16, 16, 4, 16, 1);
    test_relattn(130, 3, 0, 200, 24, 4, 200, 1);
    test_relattn(3, 513, 512, 512, 128, 16, 512, 1);
    test_relattn(2, 9, 0, 64, 256, 64, 8, 1);
    test_relattn(1, 1500, 0, 1501, 128, 16, 512, 1);   /* a decode step at 1500 positions: twelve tiles split and joined */
    test_relattn(8, 700, 0, 708, 64, 8, 64, 0);
    printf("loop gate\n");
    {
        const int caps = T->size >= offsetof(ColiCudaChainOps, ink2_caps) + sizeof T->ink2_caps && T->ink2_caps ? T->ink2_caps() : 0;
        printf("  %s the backend's Inkling-2-Mini bits: %d\n", (caps & 3) == 3 ? "ok  " : "FAIL", caps);
        if ((caps & 3) != 3) fails++;
    }
    test_loopgate(1, 2048, 0); test_loopgate(3, 300, 1);
    printf("ink2 mix, clip, dconv\n");
    test_mix(1, 64, 0); test_mix(3, 4608, 8); test_mix(2, 300, 15);
    test_clip(3, 4, 128, 1, 2.f); test_clip(2, 2, 16, 0, 0.5f); test_clip(1, 48, 300, 1, 22.6f);
    test_dconv(2, 16, 4, 3, 1); test_dconv(8, 128, 4, 1, 1); test_dconv(1, 8, 9, 5, 4); test_dconv(3, 4, 1, 2, 2);
    printf(fails ? "cuda-inkc-check: %d FAILED\n" : "cuda-inkc-check: all passed\n", fails);
    return fails ? 1 : 0;
}
