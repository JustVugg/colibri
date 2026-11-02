// mla_call_bench.cu -- k3cu_mla whole, its copies from ordinary (pageable) host memory included: a layer of a
// P-token prompt as calls of 1,024 rows (the caller before) and as one call for the step
#include "../../../c/backend_cuda_k3.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint32_t rng = 1;
static float rndf() { rng = rng * 1664525u + 1013904223u; return ((float)(rng >> 8) / 16777216.0f - 0.5f) * 4.0f; }

int main(int argc, char **argv) {
    const int H = 96, qn = 128, qr = 64, vh = 128, kvl = 512, P = argc > 1 ? atoi(argv[1]) : 5862;
    const int qh = qn + qr, O = H * (qn + vh);
    if (!k3cu_init(0)) { printf("init failed\n"); return 1; }
    std::vector<float> Lc((size_t)P * kvl), Rc((size_t)P * qr), qv((size_t)P * H * qh), gv((size_t)P * H * vh);
    std::vector<float> c1((size_t)P * H * vh), c2((size_t)P * H * vh);
    std::vector<int8_t> kvb((size_t)O * kvl); std::vector<float> kvbs(O);
    for (auto &v : Lc) v = rndf(); for (auto &v : Rc) v = rndf(); for (auto &v : qv) v = rndf() * 0.3f; for (auto &v : gv) v = rndf();
    for (auto &v : kvb) v = (int8_t)((int)(rndf() * 60)); for (auto &v : kvbs) v = 0.004f;
    const float scale = 1.0f / sqrtf((float)qh);
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    for (int rep = 0; rep < reps; rep++) {
        double t0 = now_s();
        int ok = 1;
        for (int c0 = 0; c0 < P && ok; c0 += 1024) {
            const int n = P - c0 < 1024 ? P - c0 : 1024;
            ok = k3cu_mla(Lc.data(), Rc.data(), c0 + n, c0, n, qv.data() + (size_t)c0 * H * qh, gv.data() + (size_t)c0 * H * vh,
                          kvb.data(), kvbs.data(), H, qn, qr, vh, kvl, scale, c1.data() + (size_t)c0 * H * vh);
        }
        double t1 = now_s();
        ok &= k3cu_mla(Lc.data(), Rc.data(), P, 0, P, qv.data(), gv.data(), kvb.data(), kvbs.data(), H, qn, qr, vh, kvl, scale, c2.data());
        double t2 = now_s();
        printf("a layer of %d tokens: calls of 1,024 rows %.1f ms, one call %.1f ms (x24: %.2f / %.2f s); %s, outputs %s\n", P,
               (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t1 - t0) * 24, (t2 - t1) * 24, ok ? "ok" : "FAILED",
               memcmp(c1.data(), c2.data(), c1.size() * 4) ? "DIFFER" : "the same");
    }
    return 0;
}
