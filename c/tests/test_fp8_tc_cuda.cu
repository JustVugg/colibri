/* fmt=8 (fp8-e4m3) expert groups on the tensor cores (COLI_CUDA_F8_TC=1, grouped_f8_tc).
 *
 *   1. Qwen3.8-Flash-Next's expert shape (hidden 2560, inter 640) and a tail shape (inputs 320,
 *      outputs 192: a partial tile of outputs, a partial last scale block), experts of 1 to 100
 *      rows: the tensor cores' path against the CPU reference (the warp kernels' accumulation:
 *      float within a 128-block, double across), relative RMS under 2e-3 (rows and weights
 *      through half), and against the warp kernels.
 *   2. A group whose largest expert has fewer rows than COLI_CUDA_F8_TC_MIN_ROWS stays on the
 *      warp kernels: bit for bit their output.
 *   3. COLI_CUDA_F8_TC unset: the warp kernels, bit for bit.
 *   BENCH=1: the kernels' time (COLI_CUDA_PROFILE's) of 64-expert groups of the Qwen3.8 shape at
 *   8, 20, 40, 80 and 160 rows an expert (a 2048-row chunk's top 10 of 512 gives 40), warp
 *   kernels against the tensor cores.
 *
 * Build: make fp8-tc-test (nvcc, a CUDA card).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../backend_cuda.cu"

#ifdef _WIN32
static int setenv(const char *name, const char *value, int overwrite) { (void)overwrite; return _putenv_s(name, value); }
static int unsetenv(const char *name) { return _putenv_s(name, ""); }
#endif

static float e4m3_ref(uint8_t b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    float v = e ? ldexpf(1.f + m / 8.f, e - 7) : ldexpf(m / 8.f, -6);
    return s ? -v : v;
}
static uint8_t rnd_e4m3(void) { uint8_t b = (uint8_t)(rand() & 255); if ((b & 0x7F) == 0x7F) b &= (uint8_t)~1; return b; }

static void cpu_gemv_f8(const uint8_t *q, const float *sc, int K, int O, const float *x, float *y) {
    int nblk = (K + 127) / 128;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q + (size_t)o * K; const float *scl = sc + (size_t)(o / 128) * nblk;
        double a = 0;
        for (int b = 0; b * 128 < K; b++) {
            int base = b * 128, len = K - base < 128 ? K - base : 128; float acc = 0;
            for (int i = base; i < base + len; i++) acc += e4m3_ref(w[i]) * x[i];
            a += (double)acc * scl[b];
        }
        y[o] = (float)a;
    }
}

struct Expert { std::vector<uint8_t> g, u, d; std::vector<float> gs, us, ds; ColiCudaTensor *tg = nullptr, *tu = nullptr, *td = nullptr; };

static int make_expert(Expert &e, int D, int I) {
    const int nbD = (D + 127) / 128, nbI = (I + 127) / 128;
    e.g.resize((size_t)I * D); e.u.resize((size_t)I * D); e.d.resize((size_t)D * I);
    e.gs.resize((size_t)nbI * nbD); e.us.resize((size_t)nbI * nbD); e.ds.resize((size_t)nbD * nbI);
    for (auto &b : e.g) b = rnd_e4m3();
    for (auto &b : e.u) b = rnd_e4m3();
    for (auto &b : e.d) b = rnd_e4m3();
    for (auto &v : e.gs) v = ldexpf(1.f + rand() / (float)RAND_MAX, -10);
    for (auto &v : e.us) v = ldexpf(1.f + rand() / (float)RAND_MAX, -10);
    for (auto &v : e.ds) v = ldexpf(1.f + rand() / (float)RAND_MAX, -10);
    return coli_cuda_tensor_upload(&e.tg, e.g.data(), e.gs.data(), 8, D, I, 0) &&
           coli_cuda_tensor_upload(&e.tu, e.u.data(), e.us.data(), 8, D, I, 0) &&
           coli_cuda_tensor_upload(&e.td, e.d.data(), e.ds.data(), 8, I, D, 0);
}

static void cpu_expert(const Expert &e, int D, int I, const float *x, float *y) {
    std::vector<float> g(I), u(I);
    cpu_gemv_f8(e.g.data(), e.gs.data(), D, I, x, g.data());
    cpu_gemv_f8(e.u.data(), e.us.data(), D, I, x, u.data());
    for (int i = 0; i < I; i++) g[i] = (g[i] / (1.f + expf(-g[i]))) * u[i];
    cpu_gemv_f8(e.d.data(), e.ds.data(), I, D, g.data(), y);
}

static double rel_rms(const float *a, const float *b, size_t n) {
    double e = 0, m = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i] - b[i]; e += d * d; m += (double)b[i] * b[i]; }
    return m > 0 ? sqrt(e / m) : sqrt(e);
}

static int fails = 0;
static void check(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; }

static int run_group(std::vector<Expert> &ex, const int *rows, int n, const float *x, float *y) {
    std::vector<ColiCudaTensor *> g(n), u(n), d(n);
    for (int c = 0; c < n; c++) { g[c] = ex[c].tg; u[c] = ex[c].tu; d[c] = ex[c].td; }
    return coli_cuda_expert_group(g.data(), u.data(), d.data(), rows, n, y, x);
}

static void shape(int D, int I, const char *name) {
    const int n = 8; int rows[n] = {1, 16, 17, 40, 64, 33, 100, 5}, total = 0;
    for (int c = 0; c < n; c++) total += rows[c];
    std::vector<Expert> ex(n);
    for (int c = 0; c < n; c++) if (!make_expert(ex[c], D, I)) { check(0, "upload"); return; }
    std::vector<float> x((size_t)total * D), want((size_t)total * D), yw((size_t)total * D), yt((size_t)total * D), y2((size_t)total * D);
    for (auto &v : x) v = (rand() / (float)RAND_MAX - .5f) * 2.f;
    for (int c = 0, off = 0; c < n; c++) { for (int s = 0; s < rows[c]; s++) cpu_expert(ex[c], D, I, &x[(size_t)(off + s) * D], &want[(size_t)(off + s) * D]); off += rows[c]; }
    unsetenv("COLI_CUDA_F8_TC");
    int ok = run_group(ex, rows, n, x.data(), yw.data());
    setenv("COLI_CUDA_F8_TC", "1", 1);
    ok = ok && run_group(ex, rows, n, x.data(), yt.data());
    char msg[200];
    const double ew = rel_rms(yw.data(), want.data(), yw.size()), et = rel_rms(yt.data(), want.data(), yt.size()), ewt = rel_rms(yt.data(), yw.data(), yt.size());
    printf("  %s: relative RMS against the CPU: warp %.2e, tensor cores %.2e; tensor cores against warp %.2e\n", name, ew, et, ewt);
    snprintf(msg, sizeof msg, "%s: the tensor cores within 2e-3 of the CPU", name);
    check(ok && et < 2e-3, msg);
    setenv("COLI_CUDA_F8_TC_MIN_ROWS", "101", 1);   /* the largest expert has 100: the warp kernels */
    ok = run_group(ex, rows, n, x.data(), y2.data());
    unsetenv("COLI_CUDA_F8_TC_MIN_ROWS");
    snprintf(msg, sizeof msg, "%s: under COLI_CUDA_F8_TC_MIN_ROWS the warp kernels' bits", name);
    check(ok && !memcmp(y2.data(), yw.data(), y2.size() * sizeof(float)), msg);
    unsetenv("COLI_CUDA_F8_TC");
    ok = run_group(ex, rows, n, x.data(), y2.data());
    snprintf(msg, sizeof msg, "%s: COLI_CUDA_F8_TC unset, the warp kernels' bits", name);
    check(ok && !memcmp(y2.data(), yw.data(), y2.size() * sizeof(float)), msg);
    for (auto &e : ex) { coli_cuda_tensor_free(e.tg); coli_cuda_tensor_free(e.tu); coli_cuda_tensor_free(e.td); }
}

static void bench(void) {
    const int D = 2560, I = 640, n = 64, reps = 10;
    std::vector<Expert> ex(n);
    for (int c = 0; c < n; c++) if (!make_expert(ex[c], D, I)) { check(0, "bench upload"); return; }
    setenv("COLI_CUDA_PROFILE", "1", 1);
    printf("  64 experts of 2560 x 640 (Qwen3.8-Flash-Next), the kernels' time of a group, mean of %d:\n", reps);
    for (int r : {8, 20, 40, 80, 160}) {
        std::vector<int> rows(n, r);
        std::vector<float> x((size_t)n * r * D), y((size_t)n * r * D);
        for (auto &v : x) v = (rand() / (float)RAND_MAX - .5f) * 2.f;
        double ms[2] = {0, 0};
        for (int mode = 0; mode < 2; mode++) {
            if (mode) setenv("COLI_CUDA_F8_TC", "1", 1); else unsetenv("COLI_CUDA_F8_TC");
            run_group(ex, rows.data(), n, x.data(), y.data());   /* warm */
            const double k0 = g_group_kernel_ms;
            for (int k = 0; k < reps; k++) run_group(ex, rows.data(), n, x.data(), y.data());
            ms[mode] = (g_group_kernel_ms - k0) / reps;
        }
        const double gb = (double)n * 3 * D * I / 1e9, flop = 2.0 * 3 * D * I * (double)n * r;
        printf("    %3d rows an expert: warp %7.3f ms, tensor cores %7.3f ms (%.2fx; %.0f vs %.0f GB/s of weights, %.1f TFLOP/s)\n",
               r, ms[0], ms[1], ms[0] / ms[1], gb / (ms[0] / 1e3), gb / (ms[1] / 1e3), flop / (ms[1] / 1e3) / 1e12);
    }
    unsetenv("COLI_CUDA_F8_TC"); unsetenv("COLI_CUDA_PROFILE");
    for (auto &e : ex) { coli_cuda_tensor_free(e.tg); coli_cuda_tensor_free(e.tu); coli_cuda_tensor_free(e.td); }
}

int main(void) {
    srand(7);
    setenv("COLI_CUDA_F8_WARP", "1", 1);
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("SKIP no CUDA device\n"); return 0; }
    float lut[256]; for (int i = 0; i < 256; i++) lut[i] = e4m3_ref((uint8_t)i);
    if (!coli_cuda_fp8_set_lut(lut)) { printf("FAIL set_lut\n"); return 1; }
    printf(" fmt=8 groups on the tensor cores (COLI_CUDA_F8_TC)\n");
    shape(2560, 640, "2560 x 640 (Qwen3.8-Flash-Next)");
    shape(320, 192, "tail 320 x 192");
    const char *b = getenv("BENCH");
    if (b && *b == '1') bench();
    if (fails) { printf("test_fp8_tc_cuda: %d failure(s)\n", fails); return 1; }
    printf("OK test_fp8_tc_cuda\n");
    return 0;
}
