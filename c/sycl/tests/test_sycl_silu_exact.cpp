/* COLI_SYCL_SILU_CPU=2 on the card against the host, exhaustively (sycl/coli_silu_exact.hpp):
 *   1. glibc_expf_dev on the device == the host's expf (glibc's libm: this runner is linked by GCC, as the engine is)
 *      for all 2^32 float inputs, NaN results equal when both are NaN;
 *   2. glibc_form_silu_mul(g, 1) on the device == coli_fp8_host_silu_mul(g, 1) (sycl/fp8_host_silu.h) for all 2^32 g;
 *   3. the same for 2^28 (g, u) pairs: g pseudo-random bit patterns over the whole float range, u random.
 * The host reference calls glibc's expf through a pointer (no inlining, no vector variant) and does
 * coli_fp8_host_silu_mul's float steps; that the header function itself gives the same bits is checked too.
 * Prints SILU EXACT PASS, or the counts and the first mismatches and exits 1. Runs on the first GPU the runtime shows
 * (one card in the campaign's container). */
#include <sycl/sycl.hpp>
#include "../coli_silu_exact.hpp"
#include "../fp8_host_silu.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float (*volatile g_expf)(float) = expf;   /* glibc's, through a pointer */
static float host_silu_ref(float g, float u) {   /* coli_fp8_host_silu_mul's steps with that expf */
    const float z = g_expf(g >= 0.f ? -g : g);
    const float sig = g >= 0.f ? 1.f / (1.f + z) : z / (1.f + z);
    const float hidden = g * sig;
    return hidden * u;
}
static bool same(uint32_t a, uint32_t b) {
    const float fa = bitsf(a), fb = bitsf(b);
    return a == b || (std::isnan(fa) && std::isnan(fb));
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    if (!q.get_device().has(sycl::aspect::fp64)) { std::printf("SKIP: no fp64 on this device\n"); return 0; }
    const size_t CH = (size_t)1 << 26;
    uint32_t *dev = sycl::malloc_device<uint32_t>(CH, q);
    uint32_t *devu = sycl::malloc_device<uint32_t>(CH, q);
    std::vector<uint32_t> got(CH), gu(CH);
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    long long bad[3] = {0, 0, 0}; std::atomic<int> shown{0}; std::atomic<long long> header_bad{0};
    auto host_check = [&](int which, uint64_t base, const uint32_t *ub) {
        std::atomic<long long> nb{0};
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; t++) th.emplace_back([&, t] {
            long long local = 0;
            for (size_t i = t; i < CH; i += nt) {
                const uint32_t xb = which == 2 ? (uint32_t)(base + i) * 2654435761u : (uint32_t)(base + i);
                const float x = bitsf(xb);
                uint32_t want;
                const float u = which == 2 ? bitsf(ub[i]) : 1.0f;
                if (which == 0) want = fbits(g_expf(x));
                else {
                    want = fbits(host_silu_ref(x, u));
                    if (!same(want, fbits(coli_fp8_host_silu_mul(x, u)))) header_bad++;
                }
                if (!same(want, got[i])) {
                    local++;
                    if (shown.fetch_add(1) < 8) {
                        std::printf("  mismatch test %d: x=%a u=%a want %a got %a\n", which + 1, x, u,
                                    bitsf(want), bitsf(got[i])); }
                }
            }
            nb += local;
        });
        for (auto &h : th) h.join();
        return nb.load();
    };
    for (int which = 0; which < 3; which++) {
        const uint64_t chunks = which == 2 ? 4 : ((uint64_t)1 << 32) / CH;
        for (uint64_t c = 0; c < chunks; c++) {
            const uint64_t base = c * CH;
            if (which == 2) {   /* u: random floats in [-8, 8) and every tenth a raw bit pattern */
                uint32_t s = 0x9e3779b9u ^ (uint32_t)c;
                for (size_t i = 0; i < CH; i++) {
                    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                    gu[i] = (i % 10 == 9) ? s : fbits((float)((int32_t)s) * (8.0f / 2147483648.0f));
                }
                q.memcpy(devu, gu.data(), CH * sizeof(uint32_t)).wait();
            }
            const int w = which;
            q.parallel_for(sycl::range<1>(CH), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const uint32_t xb = w == 2 ? (uint32_t)(base + i) * 2654435761u : (uint32_t)(base + i);
                const float x = sycl::bit_cast<float>(xb);
                float r;
                if (w == 0) r = glibc_expf_dev(x);
                else if (w == 1) r = glibc_form_silu_mul(x, 1.0f);
                else r = glibc_form_silu_mul(x, sycl::bit_cast<float>(devu[i]));
                dev[i] = sycl::bit_cast<uint32_t>(r);
            }).wait();
            q.memcpy(got.data(), dev, CH * sizeof(uint32_t)).wait();
            bad[which] += host_check(which, base, gu.data());
        }
        std::printf("test %d (%s): %lld mismatches\n", which + 1,
                    which == 0 ? "expf, all 2^32 inputs" : which == 1 ? "SiLU(g, 1), all 2^32 g" : "SiLU(g, u), 2^28 pairs",
                    bad[which]);
        std::fflush(stdout);
    }
    sycl::free(dev, q); sycl::free(devu, q);
    std::printf("host: coli_fp8_host_silu_mul against the glibc reference: %lld mismatches\n", header_bad.load());
    if (bad[0] || bad[1] || bad[2] || header_bad.load()) { std::printf("SILU EXACT FAIL\n"); return 1; }
    std::printf("SILU EXACT PASS\n");
    return 0;
}
