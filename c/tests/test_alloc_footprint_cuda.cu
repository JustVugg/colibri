/* coli_cuda_alloc_footprint / coli_cuda_tensor_vram (#687)
 *
 * The expert tier charged `remaining` the LOGICAL byte count of each expert
 * while the allocator was taking more, so auto could claim the card to within
 * a few MiB and then fail every lazy dense upload afterwards. These two
 * functions report what an allocation really costs.
 *
 * Everything here is measured against the live allocator rather than against a
 * rounding rule, because the rounding is a driver and architecture property
 * and a test that hardcoded one card's table would fail on the next.
 *
 * build: nvcc $(GPUFLAGS) backend_cuda.cu tests/test_alloc_footprint_cuda.cu
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "../backend_cuda.h"

static int failures = 0;

static void check(int cond, const char *what, const char *detail) {
    std::printf("%-4s %-56s %s\n", cond ? "ok" : "FAIL", what, detail ? detail : "");
    if (!cond) failures++;
}

/* Ground truth, independent of the function under test: allocate n buffers of
 * `bytes` and read how much free VRAM actually went away. Averaged over n
 * because the driver can serve one request from a pool it already holds, so a
 * single allocation can read as free. */
static double measured_footprint(size_t bytes, int n) {
    size_t fb = 0, fa = 0, t = 0;
    if (cudaMemGetInfo(&fb, &t) != cudaSuccess) return 0.0;
    std::vector<void *> p(n, nullptr);
    int made = 0;
    for (int i = 0; i < n; i++) {
        if (cudaMalloc(&p[i], bytes) != cudaSuccess) break;
        made++;
    }
    cudaMemGetInfo(&fa, &t);
    for (int i = 0; i < made; i++) cudaFree(p[i]);
    return made ? (double)(fb - fa) / made : 0.0;
}

int main(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        std::printf("no CUDA device (count=%d)\n", count);
        return 77;
    }
    int devices[8], ndev = 0;
    for (int i = 0; i < count && ndev < 8; i++) devices[ndev++] = i;
    if (!coli_cuda_init(devices, ndev)) {
        std::printf("cuda init failed\n");
        return 77;
    }

    /* 1. Never under-reports. This is the load-bearing property: the whole bug
     *    was a budget charged less than reality, so a footprint below the
     *    request would reintroduce it in a new place. */
    const size_t sizes[] = { 4096, 262144, 524288, 786432, 786433,
                             1048576, 1048577, 1572864, 2097152, 6291456 };
    for (size_t b : sizes) {
        char d[128];
        size_t got = coli_cuda_alloc_footprint(b);
        std::snprintf(d, sizeof d, "%zu -> %zu", b, got);
        check(got >= b, "footprint never below the request", d);
    }

    /* 2. It tracks the real allocator, not a formula. */
    for (size_t b : { (size_t)786432, (size_t)1572864, (size_t)6291456 }) {
        double real = measured_footprint(b, 64);
        size_t got = coli_cuda_alloc_footprint(b);
        char d[160];
        std::snprintf(d, sizeof d, "%zu: probe %zu vs measured %.0f", b, got, real);
        check(real > 0 && (double)got == real, "footprint agrees with a live measurement", d);
    }

    /* 3. An exactly-boundary-aligned request pays nothing. Without this the
     *    suite would pass on a function that just doubled everything. */
    {
        size_t b = 6291456;               /* 6 MiB, the int4-g64 weight array */
        size_t got = coli_cuda_alloc_footprint(b);
        char d[128];
        std::snprintf(d, sizeof d, "6 MiB -> %zu (overhead %lld)", got, (long long)got - (long long)b);
        check(got == b, "an aligned request is charged exactly, not inflated", d);
    }

    /* 4. Cached: the second call must not allocate again. Checked by free VRAM
     *    rather than by timing, which would be flaky under load. */
    {
        size_t fb = 0, fa = 0, t = 0;
        coli_cuda_alloc_footprint(786432);          /* prime */
        cudaMemGetInfo(&fb, &t);
        for (int i = 0; i < 500; i++) coli_cuda_alloc_footprint(786432);
        cudaMemGetInfo(&fa, &t);
        char d[128];
        std::snprintf(d, sizeof d, "500 repeat calls moved free VRAM by %lld B",
                      (long long)fb - (long long)fa);
        check(fb == fa, "repeat calls are cached, not re-probed", d);
    }

    /* 5. The tensor-level question the sizing path actually asks. A GLM-5.2
     *    int4-g64 expert matrix: fmt=4, gs=64, weights on a boundary and a
     *    0.75 MiB scale array that is not. */
    {
        const int I = 6144, O = 2048, gs = 64;
        size_t wb = (size_t)O * ((I + 1) / 2);
        size_t ng = (I + gs - 1) / gs, sc = (size_t)O * ng;
        std::vector<unsigned char> w(wb, 0x11);
        std::vector<float> s(sc, 1.0f);
        ColiCudaTensor *t = nullptr;
        if (!coli_cuda_tensor_upload_g(&t, w.data(), s.data(), 4, I, O, 0, gs)) {
            std::printf("FAIL upload\n"); return 1;
        }
        size_t logical = coli_cuda_tensor_bytes(t);
        size_t vram    = coli_cuda_tensor_vram(t);
        char d[192];
        std::snprintf(d, sizeof d, "logical %zu, vram %zu, padding %lld",
                      logical, vram, (long long)vram - (long long)logical);
        check(vram >= logical, "tensor_vram is never below tensor_bytes", d);
        check(vram > logical, "the 0.75 MiB scale array's padding is counted", d);
        std::snprintf(d, sizeof d, "%.4f MiB", 3.0 * (vram - logical) / 1048576.0);
        std::printf("     per-expert unaccounted (x3 matrices): %s\n", d);
        std::printf("     #687 measured 0.741 MiB/expert, sigma 0.019\n");
        coli_cuda_tensor_free(t);
    }

    std::printf(failures ? "\ntest_alloc_footprint_cuda: FAILED\n"
                         : "\ntest_alloc_footprint_cuda: ok\n");
    return failures ? 1 : 0;
}
