#include "../backend_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

static int close_enough(const float *got, const float *want, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - want[i]) > 1e-4f) {
            std::fprintf(stderr, "mismatch %d: got %.6f want %.6f\n", i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES], ndev = argc > 1 ? argc - 1 : 1;
    if (ndev > COLI_CUDA_MAX_DEVICES) return 2;
    for (int i = 0; i < ndev; i++) devices[i] = argc > 1 ? std::atoi(argv[i + 1]) : 0;
    if (!coli_cuda_init(devices, ndev)) return 77;
    if (coli_cuda_device_count() != ndev) return 1;
    int d0 = devices[0], d1 = devices[ndev > 1 ? 1 : 0];
    size_t count = 99, bytes = 99;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    const float x[8] = {1, -2, 3, -4, 2, 1, -1, 0.5f};
    float got[4];

    const int8_t q8[8] = {1, 2, 3, 4, -1, 2, -3, 4};
    const float s8[2] = {0.5f, 2.0f};
    const float want8[4] = {-5.0f, -60.0f, 1.5f, 10.0f};
    ColiCudaTensor *t8 = nullptr;
    if (!coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d0)) return 1;
    if (coli_cuda_tensor_upload(&t8, q8, s8, 1, 5, 2, d0)) return 1;
    if (ndev > 1 && coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d1)) return 1;
    if (!coli_cuda_matmul(&t8, got, x, q8, s8, 1, 2, 4, 2, d0) || !close_enough(got, want8, 4)) return 1;
    /* Cached tensor must stay callable without live host pointers (VRAM-only
     * slots free their host slab after upload) — including SUSTAINED reuse,
     * not just the first call after the free. */
    for (int rep = 0; rep < 64; rep++)
        if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0) ||
            !close_enough(got, want8, 4)) return 1;
    /* A tensor uploaded from a TEMPORARY host buffer (the VRAM-only load path:
     * load slab -> upload -> free slab) must survive the buffer being reused. */
    {
        int8_t *tmpw = static_cast<int8_t *>(std::malloc(8));
        float  *tmps = static_cast<float *>(std::malloc(2 * sizeof(float)));
        if (!tmpw || !tmps) return 2;
        for (int i = 0; i < 8; i++) tmpw[i] = q8[i];
        tmps[0] = s8[0]; tmps[1] = s8[1];
        ColiCudaTensor *tt = nullptr;
        if (!coli_cuda_tensor_upload(&tt, tmpw, tmps, 1, 4, 2, d0)) return 1;
        for (int i = 0; i < 8; i++) tmpw[i] = 99;      /* scribble, then free */
        std::free(tmpw); std::free(tmps);
        if (!coli_cuda_matmul(&tt, got, x, nullptr, nullptr, 1, 2, 4, 2, d0) ||
            !close_enough(got, want8, 4)) return 1;
        coli_cuda_tensor_free(tt);
    }
    /* Upload failures must be graceful and must not corrupt accounting. */
    {
        size_t c0 = 0, b0 = 0, c1 = 0, b1 = 0;
        coli_cuda_stats(-1, &c0, &b0);
        ColiCudaTensor *bad = nullptr;
        if (coli_cuda_tensor_upload(&bad, q8, s8, 1, 4, 2, 9999)) return 1;   /* invalid device */
        if (coli_cuda_tensor_upload(&bad, q8, s8, 7, 4, 2, d0)) return 1;     /* invalid format */
        if (coli_cuda_tensor_upload(&bad, q8, nullptr, 1, 4, 2, d0)) return 1;/* missing scales */
        if (coli_cuda_tensor_upload(&bad, nullptr, s8, 1, 4, 2, d0)) return 1;/* no host data on FIRST upload */
        if (coli_cuda_tensor_upload(&bad, q8, s8, 1, 1 << 20, 1 << 24, d0)) return 1; /* ~16 TB: must fail cleanly */
        if (bad) return 1;                              /* no partial tensor may leak out */
        coli_cuda_stats(-1, &c1, &b1);
        if (c0 != c1 || b0 != b1) return 1;             /* failed uploads must not change stats */
    }
    /* Fault injection: COLI_GPU_FAIL_AFTER=0 must fail every matmul (the hook
     * the engine-level repair validation uses), and unsetting it must restore. */
    if (setenv("COLI_GPU_FAIL_AFTER", "0", 1)) return 2;
    if (coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0)) return 1;
    if (unsetenv("COLI_GPU_FAIL_AFTER")) return 2;
    if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0) ||
        !close_enough(got, want8, 4)) return 1;

    /* Rows [-8,-1,0,7] and [1,2,3,4], packed low nibble first. */
    const uint8_t q4[4] = {0x70, 0xf8, 0xa9, 0xcb};
    const float s4[2] = {1.0f, 0.25f};
    const float want4[2] = {-34.0f, -2.5f};
    ColiCudaTensor *t4 = nullptr;
    if (!coli_cuda_matmul(&t4, got, x, q4, s4, 2, 1, 4, 2, d1) || !close_enough(got, want4, 2)) return 1;

    const uint8_t q2[2] = {0xe4, 0x1b};
    const float s2[2] = {0.5f, 2.0f};
    const float want2[2] = {-2.0f, 12.0f};
    ColiCudaTensor *t2 = nullptr;
    if (!coli_cuda_matmul(&t2, got, x, q2, s2, 3, 1, 4, 2, d1) || !close_enough(got, want2, 2)) return 1;

    const float wf[8] = {1, 0, -1, 2, 0.5f, 0.5f, 0.5f, 0.5f};
    const float wantf[2] = {-10.0f, -1.0f};
    ColiCudaTensor *tf = nullptr;
    if (!coli_cuda_matmul(&tf, got, x, wf, nullptr, 0, 1, 4, 2, d0) || !close_enough(got, wantf, 2)) return 1;

    coli_cuda_stats(-1, &count, &bytes);
    if (count != 4 || bytes != 70) {
        std::fprintf(stderr, "unexpected CUDA stats: %zu tensors, %zu bytes\n", count, bytes);
        return 1;
    }
    if (coli_cuda_tensor_device(t8) != d0 || coli_cuda_tensor_device(tf) != d0 ||
        coli_cuda_tensor_device(t4) != d1 || coli_cuda_tensor_device(t2) != d1) return 1;
    coli_cuda_stats(d0, &count, &bytes);
    if (ndev > 1) {
        if (count != 2 || bytes != 48) return 1;
        coli_cuda_stats(d1, &count, &bytes);
        if (count != 2 || bytes != 22) return 1;
    } else if (count != 4 || bytes != 70) return 1;

    coli_cuda_tensor_free(t8);
    coli_cuda_tensor_free(t4);
    coli_cuda_tensor_free(t2);
    coli_cuda_tensor_free(tf);
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    coli_cuda_shutdown();
    std::printf("cuda backend: q8/q4/q2/f32 correctness ok on %d device(s)\n", ndev);
    return 0;
}
