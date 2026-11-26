/* coli_cuda_bf16t_* (deepseek_v41's V41_HEAD_GPU / V41_BF16_GPU): every output must be, byte for byte, deepseek_v41.c's
 * mvb_cpu on an AVX2 build: per output and row, lanes j = 0..7 take columns j, j+8, ... in order, acc = fma(w, x, acc)
 * from +0; y = ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)). The reference replays that with std::fma, each operation a
 * statement. Three kernels are held to it, each chosen when the matrix is uploaded:
 *   v3  bf16t3_kernel: the matrix in blocks of 32 outputs, the row count a template argument (COLI_SYCL_BF16T_LB=32
 *       here, for every shape)
 *   v2  bf16t2_kernel: the rows as they are (COLI_SYCL_BF16T_LB=0)
 *   v1  bf16t_kernel (COLI_SYCL_BF16T_V1=1)
 * Shapes: DeepSeek V4.1's router and compressor shapes and odd ones (O not a multiple of 256 or 32, I a multiple of 8
 * only), 1..9 rows (more than four: several passes), each also with strides wider than the rows, in all three modes.
 * Then the lm head (129280 x 5120, 1.32 GB), uploaded ONCE a program (the card's host-to-card path is not to be
 * exercised in loops) in the mode BF16T_HEAD names: unset = nothing set (the backend's default: v3 from 64 MiB),
 * v2, v3, v1, none. It is held to the reference at 5 rows (two passes), each of 1..4 rows to the same rows (every row
 * count's kernel), the 5 rows' bytes hashed (BF16T-HASH: the same line in every mode), and timed at 1..4 rows
 * (BF16T-TIME: a call = input upload, kernel, logits download; GB/s of weights). "BF16T SUITE PASS" at the end. */
#include "../../backend_cuda.h"
#include "sycl_test_args.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static int fails;
static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL BF16T: %s\n", what); std::exit(1); }
}
static uint32_t state = 0x2545f491u;
static uint32_t rnd() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float bf(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

static void reference(float *y, const uint16_t *w, const float *x, int S, int I, int O) {
    for (int r = 0; r < S; r++)
        for (int o = 0; o < O; o++) {
            const uint16_t *row = w + (size_t)o * I;
            const float *xr = x + (size_t)r * I;
            float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int i = 0; i < I; i += 8)
                for (int j = 0; j < 8; j++) a[j] = std::fma(bf(row[i + j]), xr[i + j], a[j]);
            const float s04 = a[0] + a[4], s26 = a[2] + a[6], s15 = a[1] + a[5], s37 = a[3] + a[7];
            const float lo = s04 + s26, hi = s15 + s37;
            y[(size_t)r * O + o] = lo + hi;
        }
}
static uint16_t wrand() {   /* a finite bf16, subnormals included, no Inf/NaN */
    uint16_t b;
    do b = (uint16_t)rnd(); while ((b & 0x7f80) == 0x7f80);
    if ((rnd() & 7) == 0) b &= 0x807f;   /* a subnormal now and then */
    else b = (uint16_t)((b & 0x807f) | ((0x70 + (rnd() % 24)) << 7));   /* exponents near 1 */
    return b;
}
enum { V3 = 0, V2 = 1, V1 = 2, DEFAULT = 3 };
static const char *const mode_name[] = { "v3", "v2", "v1", "default" };
static void mode_set(int mode) {   /* read by coli_cuda_bf16t_upload (the layout) and, for v1, by the mv */
    unsetenv("COLI_SYCL_BF16T_V1"); unsetenv("COLI_SYCL_BF16T_LB");
    if (mode == V3) setenv("COLI_SYCL_BF16T_LB", "32", 1);
    if (mode == V2) setenv("COLI_SYCL_BF16T_LB", "0", 1);
    if (mode == V1) setenv("COLI_SYCL_BF16T_V1", "1", 1);
}
static void shape(int dev, int O, int I, int S, int mode) {
    std::vector<uint16_t> w((size_t)O * I);
    for (auto &b : w) b = wrand();
    std::vector<float> x((size_t)S * I), ref((size_t)S * O), got((size_t)S * O, NAN);
    for (auto &v : x) v = ((int)(rnd() % 20001) - 10000) / 7919.0f;
    reference(ref.data(), w.data(), x.data(), S, I, O);
    mode_set(mode);
    void *h = coli_cuda_bf16t_upload(dev, w.data(), O, I);
    require(h != nullptr, "upload");
    const int ok = coli_cuda_bf16t_mv(h, got.data(), O, x.data(), I, S);
    size_t diff = 0;
    for (size_t i = 0; i < got.size(); i++) diff += std::memcmp(&got[i], &ref[i], 4) != 0;
    std::printf("  %s %s O %d I %d S %d: %zu of %zu values differ\n", ok && !diff ? "ok  " : "FAIL", mode_name[mode], O, I, S, diff, got.size());
    if (!ok || diff) fails++;
    if (S > 1) {   /* strides wider than the rows: the per-row upload and download */
        const int xs = I + 8, ys = O + 5;
        std::vector<float> xw((size_t)S * xs, NAN), yw((size_t)S * ys, NAN);
        for (int r = 0; r < S; r++) std::memcpy(&xw[(size_t)r * xs], &x[(size_t)r * I], (size_t)I * 4);
        const int ok2 = coli_cuda_bf16t_mv(h, yw.data(), ys, xw.data(), xs, S);
        size_t d2 = 0;
        for (int r = 0; r < S; r++) d2 += std::memcmp(&yw[(size_t)r * ys], &ref[(size_t)r * O], (size_t)O * 4) != 0;
        if (!ok2 || d2) { std::printf("  FAIL %s O %d I %d S %d with wide strides: %zu rows differ\n", mode_name[mode], O, I, S, d2); fails++; }
    }
    coli_cuda_bf16t_free(h);
    mode_set(DEFAULT);
}
struct Stat { double min_ms, med_ms; };
static Stat time_mv(void *h, float *y, const float *x, int O, int I, int S, int n) {
    for (int k = 0; k < 3; k++) require(coli_cuda_bf16t_mv(h, y, O, x, I, S), "warm");
    std::vector<double> t(n);
    for (int k = 0; k < n; k++) {
        const auto t0 = std::chrono::steady_clock::now();
        require(coli_cuda_bf16t_mv(h, y, O, x, I, S), "timed");
        t[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::sort(t.begin(), t.end());
    return { t[0], t[n / 2] };
}
static uint64_t fnv(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
/* the head, one upload: the reference at 5 rows, 1..4 rows against it, the hash, the timings */
static void head(int dev, const char *which) {
    const int mode = !std::strcmp(which, "v2") ? V2 : !std::strcmp(which, "v3") ? V3 : !std::strcmp(which, "v1") ? V1 : DEFAULT;
    const int O = 129280, I = 5120, SM = 5;
    std::vector<uint16_t> w((size_t)O * I);
    for (auto &b : w) b = wrand();
    std::vector<float> x((size_t)SM * I), ref((size_t)SM * O), y((size_t)SM * O, NAN);
    for (auto &v : x) v = ((int)(rnd() % 20001) - 10000) / 7919.0f;
    {   /* the reference on a few threads: each (row, output) is its own chain */
        const int nt = 16;
        std::vector<std::thread> th;
        for (int k = 0; k < nt; k++)
            th.emplace_back([&, k] {
                const int o0 = (int)((long long)O * k / nt), o1 = (int)((long long)O * (k + 1) / nt);
                for (int r = 0; r < SM; r++) reference(ref.data() + (size_t)r * O + o0, w.data() + (size_t)o0 * I, x.data() + (size_t)r * I, 1, I, o1 - o0);
            });
        for (auto &t : th) t.join();
    }
    mode_set(mode);
    const auto t0 = std::chrono::steady_clock::now();
    void *h = coli_cuda_bf16t_upload(dev, w.data(), O, I);   /* the program's one upload of the head */
    const double up = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    require(h != nullptr, "upload (the head)");
    require(coli_cuda_bf16t_mv(h, y.data(), O, x.data(), I, SM), "the head, 5 rows");
    size_t d5 = 0;
    for (size_t i = 0; i < (size_t)SM * O; i++) d5 += std::memcmp(&y[i], &ref[i], 4) != 0;
    std::printf("  %s the head %s S %d against the reference: %zu of %zu values differ\n", d5 ? "FAIL" : "ok  ", mode_name[mode], SM, d5, (size_t)SM * O);
    if (d5) fails++;
    std::printf("BF16T-HASH the head's %d rows (fnv1a-64 of the bytes): device %016llx reference %016llx\n", SM,
                (unsigned long long)fnv(y.data(), (size_t)SM * O * 4), (unsigned long long)fnv(ref.data(), (size_t)SM * O * 4));
    for (int S = 1; S <= 4; S++) {
        std::fill(y.begin(), y.end(), NAN);
        require(coli_cuda_bf16t_mv(h, y.data(), O, x.data(), I, S), "the head, 1..4 rows");
        size_t d = 0;
        for (size_t i = 0; i < (size_t)S * O; i++) d += std::memcmp(&y[i], &ref[i], 4) != 0;
        std::printf("  %s the head %s S %d against the reference: %zu of %zu values differ\n", d ? "FAIL" : "ok  ", mode_name[mode], S, d, (size_t)S * O);
        if (d) fails++;
    }
    Stat st[5];
    for (int S = 1; S <= 4; S++) st[S] = time_mv(h, y.data(), x.data(), O, I, S, 40);
    coli_cuda_bf16t_free(h);
    mode_set(DEFAULT);
    const double gb = (double)O * I * 2 / 1e9;
    std::printf("BF16T-TIME %s the head's upload: %.2f s\n", mode_name[mode], up);
    for (int S = 1; S <= 4; S++)
        std::printf("BF16T-TIME %s O %d I %d S %d: min %.3f median %.3f ms a call (input upload, kernel, logits download), %.0f GB/s of weights (median)\n",
                    mode_name[mode], O, I, S, st[S].min_ms, st[S].med_ms, gb / (st[S].med_ms * 1e-3));
}
int main(int argc, char **argv) {
    int dev[1]; int n = sycl_test_devices(argc, argv, dev, 1); if (n < 0) return 0;
    require(coli_cuda_init(dev, n), "init");
    const char *hd = std::getenv("BF16T_HEAD");   /* unset: nothing set; v2 | v3 | v1; none: no 1.3 GB shape */
    const int sh[][3] = { {64, 8, 1}, {384, 5120, 1}, {384, 5120, 4}, {513, 5120, 3}, {1000, 1032, 9}, {257, 4096, 5},
                          {200, 64, 8}, {33, 520, 2}, {31, 64, 6}, {96, 5120, 7}, {2048, 5120, 5} };
    for (int mode : {V3, V2, V1})
        for (auto &p : sh) shape(dev[0], p[0], p[1], p[2], mode);
    if (!hd || std::strcmp(hd, "none")) head(dev[0], hd && *hd ? hd : "default");
    std::printf(fails ? "BF16T SUITE FAIL\n" : "BF16T SUITE PASS\n");
    return fails ? 1 : 0;
}
