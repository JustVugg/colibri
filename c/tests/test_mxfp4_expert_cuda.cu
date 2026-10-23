/* Streaming SiTU-GLU expert vs the CPU MXFP4 decoder, including odd widths. */
#include <cmath>
#include <cstdio>
#include <vector>
#if defined(__HIPCC__)
#include "../backend_gpu_compat.h"
#else
#include <cuda_runtime.h>
#endif
#include "../backend_cuda.h"
extern "C" void mxfp4_ref(float *, const float *, const unsigned char *, const unsigned char *, int, int, int);

static int check_case(int S, int D, int I, float b1, float b2) {
    std::vector<unsigned char> gw((size_t)I * ((D + 1) / 2)), uw(gw.size()), dw((size_t)D * ((I + 1) / 2));
    std::vector<unsigned char> gs((size_t)I * ((D + 31) / 32)), us(gs.size()), ds((size_t)D * ((I + 31) / 32));
    std::vector<float> x((size_t)S * D), g((size_t)S * I), u(g.size()), want(x.size()), got(x.size());
    for (size_t j = 0; j < gw.size(); j++) { gw[j] = (unsigned char)(j * 13 + 2); uw[j] = (unsigned char)(j * 7 + 5); }
    for (size_t j = 0; j < dw.size(); j++) dw[j] = (unsigned char)(j * 11 + 3);
    for (size_t j = 0; j < gs.size(); j++) { gs[j] = 121 + j % 4; us[j] = 122 + j % 3; }
    for (size_t j = 0; j < ds.size(); j++) ds[j] = 123 + j % 4;
    for (size_t j = 0; j < x.size(); j++) x[j] = (int(j % 17) - 8) * 0.125f;
    for (int pass = 0; pass < 2; pass++) {
        mxfp4_ref(g.data(), x.data(), gw.data(), gs.data(), S, D, I);
        mxfp4_ref(u.data(), x.data(), uw.data(), us.data(), S, D, I);
        for (size_t j = 0; j < g.size(); j++)
            g[j] = b1 * tanhf(g[j] / b1) * (1.f / (1.f + expf(-g[j]))) * b2 * tanhf(u[j] / b2);
        mxfp4_ref(want.data(), g.data(), dw.data(), ds.data(), S, I, D);
        if (!coli_cuda_expert_mxfp4(got.data(), x.data(), gw.data(), gs.data(), uw.data(), us.data(),
                                   dw.data(), ds.data(), S, D, I, b1, b2)) return 1;
        for (size_t j = 0; j < got.size(); j++) {
            if (!std::isfinite(got[j]) || fabsf(got[j] - want[j]) > 2e-5f + 2e-4f * fabsf(want[j])) {
                printf("FAIL S=%d D=%d I=%d row-element=%zu want=%g got=%g\n", S, D, I, j, want[j], got[j]);
                return 1;
            }
        }
        /* Reused host addresses now contain different expert bytes. */
        for (size_t j = 0; j < gw.size(); j++) gw[j] ^= 0x88;
        for (size_t j = 0; j < ds.size(); j++) ds[j]++;
    }
    got[0] = 123.f;
    if (coli_cuda_expert_mxfp4(got.data(), x.data(), gw.data(), gs.data(), uw.data(), us.data(),
                             dw.data(), ds.data(), S, D, I, 0.f, b2) || got[0] != 123.f) return 1;
    printf("ok SiTU expert S=%d D=%d I=%d b1=%g b2=%g\n", S, D, I, b1, b2);
    return 0;
}
int main() {
    int n = 0, device = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n) { puts("SKIP no CUDA device"); return 0; }
    if (!coli_cuda_init(&device, 1)) return 1;
    int failed = check_case(1, 64, 128, 1.5f, 2.5f) |
                 check_case(3, 33, 65, 4.f, 3.f) |
                 check_case(2, 129, 31, 0.5f, 0.75f);
    coli_cuda_shutdown();
    if (!coli_cuda_init(&device, 1)) return 1;
    failed |= check_case(1, 64, 128, 1.5f, 2.5f);
    coli_cuda_shutdown();
    return failed;
}
