/* MXFP4 (fmt=7) expert groups on CUDA (grouped_hidden_mx4_dual, grouped_down_mx4)
 * against quant.h's matmul_mxfp4, the CPU path the engines already run.
 *
 * Each case uploads `count` experts (gate, up, down as fmt 7: e2m1 nibbles plus
 * one ue8m0 byte per 32 inputs), gives each its own number of rows, and checks
 * the group's output rows against the reference expert MLP
 *   y = down(silu(gate(x)) * up(x))
 * computed with mxfp4_ref (quant.h, built as C: tests/mxfp4_ref.c). Shapes: K a
 * multiple of 32 (the 16-byte loads), short K (a last group of fewer than 32,
 * read a byte at a time), the release's 2560 x 640 experts; 1 to 9 rows an
 * expert (tiles of 4 rows and their tails). The issue/take and the pinned calls
 * run the same kernels, so they must give the sync call's bits exactly. An
 * exponent of 255 in a down row is +inf on both sides: there NaN must agree with
 * NaN and inf with inf of the same sign. A group mixing fmt 7 with another
 * format is refused (the caller keeps its CPU path).
 *
 * Tolerance, not bits: the CPU sums a row's groups in column order, the GPU
 * gives the groups to the 32 lanes and folds the lanes' sums, so the float sums
 * differ in the last bits. An element passes within 1e-4 of its own magnitude
 * plus 1e-5 of the largest magnitude in its output row; a wrong nibble, parity
 * or exponent moves results by whole factors.
 *
 *   make -C c cuda-test       (runs it among the CUDA tests)
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#if defined(__HIPCC__)
#include "../backend_gpu_compat.h"
#else
#include <cuda_runtime.h>
#endif
#include "../backend_cuda.h"
extern "C" void mxfp4_ref(float *y, const float *x, const unsigned char *q4,
                          const unsigned char *e8s, int S, int I, int O);

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

struct Mat { std::vector<uint8_t> w, s; int I, O; ColiCudaTensor *t = nullptr; };

static void make_mat(Mat &m, int I, int O, int smin, int sspan) {
    m.I = I; m.O = O;
    m.w.resize((size_t)O * ((I + 1) / 2));
    m.s.resize((size_t)O * ((I + 31) / 32));
    for (auto &b : m.w) b = (uint8_t)rnd();
    for (auto &b : m.s) b = (uint8_t)(smin + rnd() % sspan);
}
static int upload(Mat &m) {
    return coli_cuda_tensor_upload(&m.t, m.w.data(), (const float *)m.s.data(), 7, m.I, m.O, 0);
}

static void reference(std::vector<float> &y, const std::vector<float> &x, const Mat &g, const Mat &u,
                      const Mat &d, int rows, int D, int I) {
    std::vector<float> a((size_t)rows * I), b(a.size());
    mxfp4_ref(a.data(), x.data(), g.w.data(), g.s.data(), rows, D, I);
    mxfp4_ref(b.data(), x.data(), u.w.data(), u.s.data(), rows, D, I);
    for (size_t j = 0; j < a.size(); j++) a[j] = (a[j] / (1.f + expf(-a[j]))) * b[j];
    y.resize((size_t)rows * D);
    mxfp4_ref(y.data(), a.data(), d.w.data(), d.s.data(), rows, I, D);
}

static int compare(const char *what, const float *want, const float *got, int rows, int D) {
    int bad = 0, nonfinite = 0;
    double worst = 0;
    for (int r = 0; r < rows; r++) {
        float scale = 0;
        for (int i = 0; i < D; i++) if (std::isfinite(want[(size_t)r * D + i])) scale = fmaxf(scale, fabsf(want[(size_t)r * D + i]));
        for (int i = 0; i < D; i++) {
            float a = want[(size_t)r * D + i], b = got[(size_t)r * D + i];
            if (std::isnan(a) || std::isnan(b)) { nonfinite++; bad += std::isnan(a) != std::isnan(b); continue; }
            if (std::isinf(a) || std::isinf(b)) { nonfinite++; bad += !(std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)); continue; }
            double err = fabs((double)a - b), tol = 1e-4 * fabs(a) + 1e-5 * scale;
            if (err > tol) { if (bad < 4) printf("  %s row %d col %d: want %.9g got %.9g\n", what, r, i, a, b); bad++; }
            if (scale > 0 && err / scale > worst) worst = err / scale;
        }
    }
    printf("  %s %s: %d rows x %d, worst |err|/row max %.2e%s\n", bad ? "FAIL" : "ok  ", what, rows, D, worst,
           nonfinite ? " (non-finite elements agree)" : "");
    return bad != 0;
}

/* count experts of D x I, rows[c] rows each; down exponents of 255 in odd rows' group 1 if inf_down */
static int run_case(const char *name, int D, int I, int count, const int *rows, int inf_down) {
    std::vector<Mat> g(count), u(count), dn(count);
    std::vector<ColiCudaTensor *> tg(count), tu(count), td(count);
    int total = 0, failed = 0;
    for (int c = 0; c < count; c++) total += rows[c];
    std::vector<float> x((size_t)total * D), want((size_t)total * D), got(want.size()), y;
    for (auto &v : x) v = (float)((int)(rnd() % 2001) - 1000) / 1000.f;
    for (int c = 0; c < count; c++) {
        make_mat(g[c], D, I, 119, 5); make_mat(u[c], D, I, 119, 5); make_mat(dn[c], I, D, 118, 5);
        if (inf_down)
            for (int o = 1; o < D; o += 2) dn[c].s[(size_t)o * ((I + 31) / 32) + 1] = 255;
        if (!upload(g[c]) || !upload(u[c]) || !upload(dn[c])) { printf("  FAIL %s: upload refused\n", name); return 1; }
        tg[c] = g[c].t; tu[c] = u[c].t; td[c] = dn[c].t;
    }
    for (int c = 0, off = 0; c < count; c++) {
        std::vector<float> xs(x.begin() + (size_t)off * D, x.begin() + (size_t)(off + rows[c]) * D);
        reference(y, xs, g[c], u[c], dn[c], rows[c], D, I);
        std::copy(y.begin(), y.end(), want.begin() + (size_t)off * D);
        off += rows[c];
    }
    char what[160];
    if (!coli_cuda_expert_group(tg.data(), tu.data(), td.data(), rows, count, got.data(), x.data())) {
        printf("  FAIL %s: the group was refused\n", name); failed = 1;
    } else {
        snprintf(what, sizeof what, "%s, sync", name);
        failed |= compare(what, want.data(), got.data(), total, D);
        std::vector<float> again(got.size());
        if (!coli_cuda_expert_group_pinned(tg.data(), tu.data(), td.data(), rows, count, again.data(), x.data(), 1) ||
            memcmp(again.data(), got.data(), got.size() * sizeof(float))) {
            printf("  FAIL %s: the pinned call does not give the sync call's bits\n", name); failed = 1;
        }
        if (total <= 32) {
            const float *taken = nullptr;
            if (!coli_cuda_expert_group_issue(tg.data(), tu.data(), td.data(), rows, count, x.data()) ||
                !(taken = coli_cuda_expert_group_take(0)) || memcmp(taken, got.data(), got.size() * sizeof(float))) {
                printf("  FAIL %s: issue/take does not give the sync call's bits\n", name); failed = 1;
            }
        }
    }
    for (int c = 0; c < count; c++) { coli_cuda_tensor_free(tg[c]); coli_cuda_tensor_free(tu[c]); coli_cuda_tensor_free(td[c]); }
    return failed;
}

/* fmt 7 with an int4 (fmt 2) member: no kernel takes the mix, so the group is refused */
static int mixed_refused(void) {
    const int D = 64, I = 32, rows[2] = {1, 1};
    Mat g, u, dn; make_mat(g, D, I, 120, 3); make_mat(u, D, I, 120, 3); make_mat(dn, I, D, 120, 3);
    std::vector<uint8_t> w4((size_t)I * (D / 2), 0x11), d4((size_t)D * (I / 2), 0x11);
    std::vector<float> s4g(I, 0.01f), s4d(D, 0.01f), x((size_t)2 * D, 0.5f), y((size_t)2 * D, 0.f);
    ColiCudaTensor *a = nullptr, *b = nullptr, *c = nullptr;
    int failed = 0;
    if (!upload(g) || !upload(u) || !upload(dn) ||
        !coli_cuda_tensor_upload(&a, w4.data(), s4g.data(), 2, D, I, 0) ||
        !coli_cuda_tensor_upload(&b, w4.data(), s4g.data(), 2, D, I, 0) ||
        !coli_cuda_tensor_upload(&c, d4.data(), s4d.data(), 2, I, D, 0)) { puts("  FAIL mixed: upload refused"); return 1; }
    ColiCudaTensor *gs[2] = {g.t, a}, *us[2] = {u.t, b}, *ds[2] = {dn.t, c};
    if (coli_cuda_expert_group(gs, us, ds, rows, 2, y.data(), x.data()) ||
        coli_cuda_expert_group_issue(gs, us, ds, rows, 2, x.data())) { puts("  FAIL a fmt 7 + fmt 2 group was accepted"); failed = 1; }
    else puts("  ok   a fmt 7 + fmt 2 group is refused (sync and issue)");
    coli_cuda_tensor_free(g.t); coli_cuda_tensor_free(u.t); coli_cuda_tensor_free(dn.t);
    coli_cuda_tensor_free(a); coli_cuda_tensor_free(b); coli_cuda_tensor_free(c);
    return failed;
}

int main() {
    int n = 0, device = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n) { puts("SKIP no CUDA device"); return 0; }
    if (!coli_cuda_init(&device, 1)) { puts("FAIL coli_cuda_init"); return 1; }
    const int r3[3] = {1, 4, 9}, r2[2] = {2, 5}, r10[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
              r8[8] = {3, 1, 2, 4, 1, 5, 2, 1}, r1[1] = {7};
    int failed = 0;
    failed |= run_case("K%32==0 (64 x 96), 1/4/9 rows", 64, 96, 3, r3, 0);
    failed |= run_case("short K (48 x 40), 2/5 rows", 48, 40, 2, r2, 0);
    failed |= run_case("release 2560 x 640, 10 experts x 1 row", 2560, 640, 10, r10, 0);
    failed |= run_case("release 2560 x 640, 8 experts x 1-5 rows", 2560, 640, 8, r8, 0);
    failed |= run_case("short K, exponent 255 in down rows", 48, 40, 1, r1, 1);
    failed |= mixed_refused();
    coli_cuda_shutdown();
    puts(failed ? "test_mxfp4_group_cuda: FAIL" : "test_mxfp4_group_cuda: ok");
    return failed;
}
