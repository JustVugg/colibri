/* cc_moe on fmt 7 (MXFP4, compressed-tensors: [O][I/2] e2m1 nibbles, the low nibble the even column, bit 3 the sign,
 * bits 0..2 index {0,.5,1,1.5,2,3,4,6}; one ue8m0 exponent byte a 32 inputs, [O][I/32], w = v * 2^(s-127)) as DeepSeek
 * V4.1's routed experts hold it, through the dense chain's table (T->moe, coli_cuda_moe_async): gate and up [I][D],
 * down [D][I], resident through coli_cuda_tensor_upload(fmt 7) and the table's entries from T->tensor_dev (the scales
 * BYTES behind CcExpert's float pointers). Activations act 0 (silu(gate) * up) and act 1 (deepseek_v41.c's clamped
 * SwiGLU: up to [-lim, lim], gate to at most lim, lim 7 with gate and up large enough to reach it). Shapes: DeepSeek
 * V4.1's (D 5120, I 3072) and a small one (D 256, I 512); S = 1, 2, 4 rows, K = 6 slots over 8 experts (an empty slot,
 * experts shared between rows); parts 3 with every expert on the device, and parts 1 then 2 with two experts the
 * host's (their table entries empty, the host's row hy at h_off 0). The reference decodes exactly and computes gate,
 * up, the activation, down and the rank-order sum in double; a case passes at max |err| / max |ref| < 2e-5. One line
 * a case ("MOE7 ... ok" / "FAIL"), "MOE7 SUITE PASS" at the end. Run it also with COLI_CUDA_MOE_GV=0 (the warp-a-row
 * kernels moe_gu_k / moe_dn_k instead of moe_gv_*). */
#include "../../backend_cuda.h"
#include "../../cuda_chain.h"
#include "sycl_test_args.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const ColiCudaChainOps *T;
static int fails;
static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL MOE7: %s\n", what); std::exit(1); }
}
static uint32_t state = 0x2545f491u;
static uint32_t rnd() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float urand() { return (float)(rnd() & 0xffffff) / 16777216.0f * 2.0f - 1.0f; }   /* [-1, 1) */

static const double e2m1[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};

struct Mat {   /* one MXFP4 matrix, O rows of I, on the host and resident */
    int O = 0, I = 0;
    std::vector<uint8_t> q, s;
    ColiCudaTensor *t = nullptr;
};
/* scale bytes around e0 (+-1): e0 = 127 + log2 of the wanted step so an output of I random inputs lands near `target` */
static void make(Mat &m, int O, int I, double target, int dev) {
    m.O = O; m.I = I;
    m.q.resize((size_t)O * I / 2); m.s.resize((size_t)O * (I / 32));
    for (auto &b : m.q) b = (uint8_t)rnd();
    const int e0 = 127 + (int)std::lround(std::log2(target / (1.45 * std::sqrt((double)I))));
    for (auto &b : m.s) b = (uint8_t)(e0 + (int)(rnd() % 3) - 1);
    require(coli_cuda_tensor_upload(&m.t, m.q.data(), (const float *)(const void *)m.s.data(), 7, I, O, dev), "fmt 7 upload");
}
static void row(const Mat &m, int o, std::vector<double> &w) {   /* row o exactly, in double */
    w.resize(m.I);
    const uint8_t *q = m.q.data() + (size_t)o * m.I / 2, *s = m.s.data() + (size_t)o * (m.I / 32);
    for (int g = 0; g < m.I / 32; g++) {
        const double sc = std::ldexp(1.0, (int)s[g] - 127);   /* exact: a power of two */
        for (int i = g * 32; i < g * 32 + 32; i++) {
            const unsigned nib = i & 1 ? q[i >> 1] >> 4 : q[i >> 1] & 15u;
            w[i] = e2m1[nib] * sc;
        }
    }
}
static double act_of(double g, double u, int act, double lim, long *hits) {
    if (act == 1) {
        if (u > lim) { u = lim; (*hits)++; }
        if (u < -lim) { u = -lim; (*hits)++; }
        if (g > lim) { g = lim; (*hits)++; }
    }
    return g / (1.0 + std::exp(-g)) * u;
}
static CcBuf *up(const void *v, size_t bytes, int kind = CC_DEV) {
    CcBuf *b = T->buf(bytes, kind);
    require(b && T->begin() && T->write(b, 0, v, bytes) && T->submit(1), "upload to a chain buffer");
    return b;
}

static void suite(int dev, int D, int I) {
    const int SM = 4, K = 6, E = 8;
    const float lim = 7.0f;
    std::vector<Mat> G(E), U(E), Dn(E);
    for (int e = 0; e < E; e++) { make(G[e], I, D, 10.0, dev); make(U[e], I, D, 10.0, dev); make(Dn[e], D, I, 0.05, dev); }
    std::vector<CcExpert> tab(E + 2), part;   /* the layer's entries from t_off 2 */
    std::memset(tab.data(), 0, tab.size() * sizeof(CcExpert));
    for (int e = 0; e < E; e++) {
        CcExpert &x = tab[2 + e];
        require(T->tensor_dev(G[e].t, &x.g, &x.gs, NULL) && T->tensor_dev(U[e].t, &x.u, &x.us, NULL) &&
                T->tensor_dev(Dn[e].t, &x.d, &x.ds, NULL), "tensor_dev");
    }
    const int host_a = 1, host_b = 6;   /* the host's experts in the parts 1 + 2 cases */
    part = tab;
    std::memset(&part[2 + host_a], 0, sizeof(CcExpert)); std::memset(&part[2 + host_b], 0, sizeof(CcExpert));
    const int ids[SM * K] = {3, 0, 7, -1, 5, 1,
                             2, 3, 6, 4, 0, 7,
                             1, 4, 2, 6, 3, -1,
                             7, 6, 5, 0, 2, 4};
    std::vector<float> w(SM * K), x(4 + (size_t)SM * D), hy((size_t)SM * D);
    for (auto &v : w) v = 0.05f + 0.35f * (urand() * 0.5f + 0.5f);
    for (auto &v : x) v = urand();
    for (auto &v : hy) v = urand();

    /* the reference: each expert's rows decoded once, against every (row, slot) that chose it */
    std::vector<std::vector<double>> gq(SM * K), uq(SM * K), out[2];
    out[0].resize(SM * K); out[1].resize(SM * K);
    long hits[SM] = {0, 0, 0, 0};   /* act 1's clamps a row */
    std::vector<double> wr;
    for (int e = 0; e < E; e++) {
        std::vector<int> pairs;
        for (int a = 0; a < SM * K; a++) if (ids[a] == e) pairs.push_back(a);
        if (pairs.empty()) continue;
        for (int a : pairs) { gq[a].assign(I, 0.0); uq[a].assign(I, 0.0); }
        for (int o = 0; o < I; o++) {
            row(G[e], o, wr);
            for (int a : pairs) { const float *xr = x.data() + 4 + (size_t)(a / K) * D; double v = 0; for (int i = 0; i < D; i++) v += wr[i] * xr[i]; gq[a][o] = v; }
            row(U[e], o, wr);
            for (int a : pairs) { const float *xr = x.data() + 4 + (size_t)(a / K) * D; double v = 0; for (int i = 0; i < D; i++) v += wr[i] * xr[i]; uq[a][o] = v; }
        }
        for (int act = 0; act < 2; act++) {
            std::vector<std::vector<double>> h(pairs.size(), std::vector<double>(I));
            for (size_t p = 0; p < pairs.size(); p++)
                for (int o = 0; o < I; o++) {
                    long dummy = 0;
                    h[p][o] = act_of(gq[pairs[p]][o], uq[pairs[p]][o], act, lim, act ? &hits[pairs[p] / K] : &dummy);
                }
            for (int a : pairs) out[act][a].assign(D, 0.0);
            for (int o = 0; o < D; o++) {
                row(Dn[e], o, wr);
                for (size_t p = 0; p < pairs.size(); p++) { double v = 0; for (int i = 0; i < I; i++) v += wr[i] * h[p][i]; out[act][pairs[p]][o] = v; }
            }
        }
    }

    /* the tier's form of the same experts: each ue8m0 byte as the float 2^(s-127) (vk_tier.c), in device buffers whose
     * addresses replace the scale pointers; cc_moe reads them with CcMoe.gs 32 (the bytes with gs 0) */
    std::vector<CcBuf *> fsb;
    auto fscales = [&](const Mat &m) -> const float * {
        std::vector<float> f(m.s.size());
        for (size_t i = 0; i < f.size(); i++) f[i] = (float)std::ldexp(1.0, (int)m.s[i] - 127);
        CcBuf *b = up(f.data(), f.size() * 4);
        fsb.push_back(b);
        return (const float *)T->dev_addr(b);
    };
    std::vector<CcExpert> tabf = tab;
    for (int e = 0; e < E; e++) { CcExpert &x = tabf[2 + e]; x.gs = fscales(G[e]); x.us = fscales(U[e]); x.ds = fscales(Dn[e]);
                                  require(x.gs && x.us && x.ds, "float scales' device addresses"); }
    std::vector<CcExpert> partf = tabf;
    std::memset(&partf[2 + host_a], 0, sizeof(CcExpert)); std::memset(&partf[2 + host_b], 0, sizeof(CcExpert));
    CcBuf *tb = up(tab.data(), tab.size() * sizeof(CcExpert)), *pb = up(part.data(), part.size() * sizeof(CcExpert));
    CcBuf *tbf = up(tabf.data(), tabf.size() * sizeof(CcExpert)), *pbf = up(partf.data(), partf.size() * sizeof(CcExpert));
    CcBuf *xb = up(x.data(), x.size() * 4), *ib = up(ids, sizeof ids), *wb = up(w.data(), w.size() * 4);
    CcBuf *hyb = up(hy.data(), hy.size() * 4, CC_UP);
    CcBuf *hb = T->buf(cc_moe_hid_floats(SM, K, D, I, E) * 4, CC_DEV), *yb = T->buf(((size_t)SM * D + 8) * 4, CC_DEV);
    require(hb && yb, "hid and y buffers");
    std::vector<float> y((size_t)SM * D + 8);
    for (int fs = 0; fs < 2; fs++)
    for (int act = 0; act < 2; act++)
        for (int S : {1, 2, 4})
            for (int split = 0; split < 2; split++) {   /* 0: parts 3, every expert on the device; 1: parts 1 then 2, the host's two */
                CcMoe m; std::memset(&m, 0, sizeof m);
                m.S = S; m.K = K; m.D = D; m.I = I; m.fmt = 7; m.gs = fs ? 32 : 0; m.act = act; m.parts = 3;
                CcBuf *t3 = fs ? tbf : tb, *t12 = fs ? pbf : pb;
                m.t_off = 2; m.x_off = 4; m.x_row = D; m.i_off = 0; m.i_row = K; m.w_off = 0; m.w_row = K;
                m.y_off = 8; m.y_row = D; m.h_off = -1; m.E = E; m.act_lim = act ? lim : 0.f;
                int ok;
                if (!split) ok = T->begin() && T->moe(t3, xb, ib, wb, hb, yb, NULL, &m) && T->submit(1);
                else {
                    CcMoe m1 = m, m2 = m; m1.parts = 1; m2.parts = 2; m2.h_off = 0;
                    ok = T->begin() && T->moe(t12, xb, ib, wb, hb, yb, hyb, &m1) && T->submit(0) &&
                         T->begin() && T->moe(t12, xb, ib, wb, hb, yb, hyb, &m2) && T->submit(1);
                }
                ok = ok && T->read(yb, 0, y.data(), y.size() * 4);
                size_t hdiff = 0;
                if (!split) {   /* the half-bits decode (FMT 77, the default) against mx4_d2's (COLI_SYCL_MX4_HALF=0): the same bits */
                    std::vector<float> y0(y.size());
                    setenv("COLI_SYCL_MX4_HALF", "0", 1);
                    ok = ok && T->begin() && T->moe(t3, xb, ib, wb, hb, yb, NULL, &m) && T->submit(1) && T->read(yb, 0, y0.data(), y0.size() * 4);
                    unsetenv("COLI_SYCL_MX4_HALF");
                    for (int s = 0; s < S; s++)
                        for (int d = 0; d < D; d++) hdiff += std::memcmp(&y[8 + (size_t)s * D + d], &y0[8 + (size_t)s * D + d], 4) != 0;
                    std::printf("MOE7   half-bits decode vs mx4_d2: %zu of %zu values differ\n", hdiff, (size_t)S * D);
                }
                double me = 0, mr = 0;
                long nh = 0;
                for (int s = 0; s < S; s++) {
                    nh += hits[s];
                    for (int d = 0; d < D; d++) {
                        double r = split ? (double)hy[(size_t)s * D + d] : 0.0;
                        for (int k = 0; k < K; k++) {   /* rank order, the device's slots */
                            const int e = ids[s * K + k];
                            if (e < 0 || (split && (e == host_a || e == host_b))) continue;
                            r += (double)w[s * K + k] * out[act][s * K + k][d];
                        }
                        me = std::fmax(me, std::fabs((double)y[8 + (size_t)s * D + d] - r));
                        mr = std::fmax(mr, std::fabs(r));
                    }
                }
                const double rel = mr > 0 ? me / mr : me;
                const bool reached = act == 0 || nh > 0;   /* act 1 must have met its limit, or it tested nothing */
                const bool pass = ok && rel < 2e-5 && reached && hdiff == 0;
                std::printf("MOE7 D=%d I=%d S=%d K=%d %s act=%d%s parts=%s: max|err|/max|ref| = %.3g (max|ref| %.4g)%s %s\n", D, I, S, K,
                            fs ? "scales f32 (the tier's, gs 32)" : "scales ue8m0 (gs 0)", act,
                            act ? " lim=7" : "", split ? "1+2 host rows" : "3", rel, mr,
                            act ? (nh ? "" : " [limit never reached]") : "", pass ? "ok" : ok ? "FAIL" : "FAIL (the call or frame failed)");
                if (act) std::printf("MOE7   clamps in the rows' gate/up: %ld\n", nh);
                std::fflush(stdout);
                if (!pass) fails++;
            }
    for (CcBuf *b : fsb) T->free(b);
    T->free(tbf); T->free(pbf);
    T->free(tb); T->free(pb); T->free(xb); T->free(ib); T->free(wb); T->free(hyb); T->free(hb); T->free(yb);
    for (int e = 0; e < E; e++) { coli_cuda_tensor_free(G[e].t); coli_cuda_tensor_free(U[e].t); coli_cuda_tensor_free(Dn[e].t); }
}

int main(int argc, char **argv) {
    int dev[1]; int n = sycl_test_devices(argc, argv, dev, 1); if (n < 0) return 0;
    require(coli_cuda_init(dev, n), "init");
    T = coli_cuda_chain_ops();
    require(T && T->size == sizeof(ColiCudaChainOps) && T->init(dev[0]) && T->ready(), "the chain on the device");
    suite(dev[0], 256, 512);
    suite(dev[0], 5120, 3072);   /* DeepSeek V4.1's routed experts */
    std::printf(fails ? "MOE7 SUITE FAIL\n" : "MOE7 SUITE PASS\n");
    return fails ? 1 : 0;
}
