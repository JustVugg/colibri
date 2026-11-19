/* bench_sycl_fp8_group: the FP8 expert group (coli_cuda_expert_group) at Qwen3.8's shape -- ten experts of
 * gate/up 2560x640 and down 640x2560, as a decode layer routes them -- timed, and its output written so that two
 * kernel families can be compared byte for byte (COLI_SYCL_F8_CPU_ORDER=1 against =2: separate runs, same seed).
 * The kernels, the host SiLU and the copies are the engine's: set the engine's FP8 settings in the environment.
 *   FP8B_ROWS    comma list of rows an expert (default 1,4,16,64): a group of FP8B_EXPERTS experts with that many
 *                rows each, inputs the same for every family
 *   FP8B_EXPERTS experts a group (default 10)
 *   FP8B_ITERS   timed calls a row count after 3 untimed (default 0: one call, a finite check -- a quick test)
 *   FP8B_OUT     file to write every row count's output to (float32, in FP8B_ROWS order)
 * Device choice as the other GPU tests (--device N, --list). Prints one line a row count:
 *   FP8B rows=R experts=E us_per_call=T ... */
#include "../../backend_cuda.h"
#include "sycl_test_args.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL FP8B: %s\n", what); std::exit(1); }
}
static uint32_t state = 0x5eed1234u;
static uint32_t random_u32() { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float sample() { return (random_u32() & 65535) / 32768.f - 1.f; }
static float decode(uint8_t b) {
    int e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
    return b & 128 ? -v : v;
}
static int env_int(const char *name, int dflt) { const char *e = std::getenv(name); return e && *e ? std::atoi(e) : dflt; }

struct Weight {
    int K, O; std::vector<uint8_t> q; std::vector<float> scale; ColiCudaTensor *device = nullptr;
    Weight(int k, int o) : K(k), O(o), q((size_t)k * o), scale((size_t)((k + 127) / 128) * ((o + 127) / 128)) {
        for (auto &b : q) { b = (uint8_t)random_u32(); if ((b & 127) == 127) b &= 254; }   /* no NaN code */
        for (auto &s : scale) s = std::ldexp(1.5f + sample() * .5f, -12);
    }
    void upload(int dev) { require(coli_cuda_tensor_upload(&device, q.data(), scale.data(), 8, K, O, dev), "upload"); }
    ~Weight() { if (device) coli_cuda_tensor_free(device); }
    Weight(const Weight &) = delete; Weight &operator=(const Weight &) = delete;
};

int main(int argc, char **argv) {
    int dev[1]; int n = sycl_test_devices(argc, argv, dev, 1); if (n < 0) return 0;
    const int D = 2560, I = 640, E = env_int("FP8B_EXPERTS", 10), iters = env_int("FP8B_ITERS", 0);
    require(E >= 1 && E <= 64 && iters >= 0, "FP8B_EXPERTS 1..64, FP8B_ITERS >= 0");
    std::vector<int> rowlist;
    {
        const char *r = std::getenv("FP8B_ROWS"); std::string s = r && *r ? r : "1,4,16,64";
        for (size_t p = 0; p < s.size();) {
            size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
            int v = std::atoi(s.substr(p, q - p).c_str()); require(v >= 1 && v <= 512, "FP8B_ROWS 1..512"); rowlist.push_back(v); p = q + 1;
        }
    }
    require(coli_cuda_init(dev, n), "init");
    float lut[256]; for (int b = 0; b < 256; b++) lut[b] = decode((uint8_t)b);
    require(coli_cuda_fp8_set_lut(lut), "publish LUT");
    std::vector<Weight *> g, u, d; std::vector<ColiCudaTensor *> dg, du, dd;
    for (int e = 0; e < E; e++) {
        g.push_back(new Weight(D, I)); u.push_back(new Weight(D, I)); d.push_back(new Weight(I, D));
        g.back()->upload(dev[0]); u.back()->upload(dev[0]); d.back()->upload(dev[0]);
        dg.push_back(g.back()->device); du.push_back(u.back()->device); dd.push_back(d.back()->device);
    }
    FILE *out = nullptr;
    if (const char *o = std::getenv("FP8B_OUT")) if (*o) { out = std::fopen(o, "wb"); require(out != nullptr, "open FP8B_OUT"); }
    const char *mode = std::getenv("COLI_SYCL_F8_CPU_ORDER");
    for (int R : rowlist) {
        std::vector<int> rows((size_t)E, R);
        const size_t S = (size_t)E * R;
        std::vector<float> x(S * D), y(S * D);
        uint32_t keep = state; state = 0xabcd0000u + (uint32_t)R;   /* the inputs depend on R only */
        for (auto &v : x) v = sample();
        state = keep;
        require(coli_cuda_expert_group(dg.data(), du.data(), dd.data(), rows.data(), E, y.data(), x.data()), "group");
        for (size_t z = 0; z < y.size(); z++) require(std::isfinite(y[z]), "finite output");
        double us = 0;
        if (iters) {
            for (int w = 0; w < 3; w++) require(coli_cuda_expert_group(dg.data(), du.data(), dd.data(), rows.data(), E, y.data(), x.data()), "warm");
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; it++) require(coli_cuda_expert_group(dg.data(), du.data(), dd.data(), rows.data(), E, y.data(), x.data()), "timed");
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        }
        uint64_t h = 1469598103934665603ull;
        for (size_t z = 0; z < y.size(); z++) { uint32_t b; std::memcpy(&b, &y[z], 4); for (int k = 0; k < 4; k++) { h ^= (b >> (8 * k)) & 255u; h *= 1099511628211ull; } }
        std::printf("FP8B rows=%d experts=%d cpu_order=%s us_per_call=%.1f weights_GBps=%.1f hash=%016llx\n", R, E,
                    mode ? mode : "unset", us, us > 0 ? (double)E * 3 * D * I / (us * 1e3) : 0.0, (unsigned long long)h);
        std::fflush(stdout);
        if (out) require(std::fwrite(y.data(), sizeof(float), y.size(), out) == y.size(), "write FP8B_OUT");
    }
    if (out) std::fclose(out);
    for (int e = 0; e < E; e++) { delete g[e]; delete u[e]; delete d[e]; }
    coli_cuda_shutdown();
    std::puts("FP8 GROUP BENCH PASS");
    return 0;
}
