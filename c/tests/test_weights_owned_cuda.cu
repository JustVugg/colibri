/* weights_owned ownership-vs-copy VRAM leak (upload failure path).
 *
 * At the defect (U6 audit F3): coli_cuda_tensor_upload sets
 * t->weights_owned = 1 only AFTER the H2D weight memcpy succeeds. When the
 * cudaMalloc succeeds and the memcpy fails, the failure branch calls
 * coli_cuda_tensor_free(t) while weights_owned is still 0, free's
 * `weights && weights_owned` gate skips the cudaFree, and the device buffer
 * leaks — invisibly to the ledger (tracked=0), visibly to the consumer
 * (cudaMemGetInfo free memory drops by weight_bytes per failed upload).
 * Ownership is a fact of the allocation, not the copy.
 *
 * Contract pinned here:
 *   1. Control: N successful upload+free cycles leave device free memory
 *      unchanged (within noise), and the ledger returns to zero.
 *   2. N upload attempts whose weight H2D memcpy fails (injected below)
 *      leave device free memory unchanged — a failed upload releases every
 *      byte it allocated. On the defect this loses weight_bytes per cycle.
 *   3. The ledger never sees the failed uploads (charge only on success).
 *   4. The upload path still works after the injection is disarmed.
 *
 * Injection seam: the backend is included directly (kernel-level pattern of
 * tests/test_fp8_warp_cuda.cu), with every `cudaMemcpy` token in the included
 * backend source renamed to a hook via an object-like macro. The runtime
 * header itself is untouched: it is in scope before the macro exists (nvcc
 * pre-includes cuda_runtime.h; under HIP it is included explicitly below),
 * so the real declaration survives and the macro only rewrites the
 * backend's call sites. The hook fails HostToDevice
 * copies of exactly the armed byte count and passes everything else through
 * to the real runtime call, so ONLY the weight upload of the target tensor
 * shape is hit. On HIP the compat header maps cudaMemcpy -> hipMemcpy, so
 * the rename is applied to hipMemcpy and chains identically.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_weights_owned_cuda.cu -o tests/weights_owned_test
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* Forward declaration first, then the rename, then the backend. Under HIP
 * the runtime header is included explicitly — the SAME header
 * backend_gpu_compat.h pulls, unreachable under nvcc — so the declaration
 * does not depend on hipcc pre-including it; under CUDA, nvcc pre-includes
 * cuda_runtime.h before every TU. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <hip/hip_runtime.h>
extern "C" hipError_t coli_test_memcpy_hook(void *dst, const void *src,
                                            size_t count, hipMemcpyKind kind);
#define hipMemcpy coli_test_memcpy_hook
#else
extern "C" cudaError_t coli_test_memcpy_hook(void *dst, const void *src,
                                             size_t count, enum cudaMemcpyKind kind);
#define cudaMemcpy coli_test_memcpy_hook
#endif

/* No direct <cuda_runtime.h>: the backend include pulls backend_gpu_compat.h,
 * which supplies the runtime surface for BOTH vendors. */
#include "../backend_cuda.cu"

/* backend_gpu_compat.h maps only the names the backend itself uses:
 * cudaMemcpyKind (the backend never names the kind TYPE) and
 * cudaErrorInvalidValue (it never names this error) are not among them, so
 * under hipcc they are undefined. Alias the two locally per vendor rather
 * than widening the product header for a test-only need — the same
 * arrangement the repo's other direct-include tests use for their
 * off-surface names. */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#undef hipMemcpy
#define COLI_REAL_MEMCPY      hipMemcpy
#define cudaMemcpyKind        hipMemcpyKind
#define cudaErrorInvalidValue hipErrorInvalidValue
#else
#undef cudaMemcpy
#define COLI_REAL_MEMCPY      cudaMemcpy
#endif

static size_t g_fail_h2d_bytes = 0;   /* 0 = pass everything through */
static int    g_hook_hits      = 0;

extern "C" cudaError_t coli_test_memcpy_hook(void *dst, const void *src,
                                             size_t count, cudaMemcpyKind kind) {
    if (g_fail_h2d_bytes && kind == cudaMemcpyHostToDevice && count == g_fail_h2d_bytes) {
        g_hook_hits++;
        return cudaErrorInvalidValue;
    }
    return COLI_REAL_MEMCPY(dst, src, count, kind);
}

static int device_free_bytes(size_t *out) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    *out = free_b;
    return 1;
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES], ndev = argc > 1 ? argc - 1 : 1;
    if (ndev > COLI_CUDA_MAX_DEVICES) return 2;
    for (int i = 0; i < ndev; i++) devices[i] = argc > 1 ? std::atoi(argv[i + 1]) : 0;
    if (!coli_cuda_init(devices, ndev)) return 77;
    int d0 = devices[0];

    /* fmt=0 (float weights, no scale buffer): weight_bytes is exactly
     * I*O*4, giving the hook an unambiguous byte-count target, and a 64 MB
     * signal that dwarfs allocator/driver jitter (SLACK below). */
    const int I = 4096, O = 4096, N = 8;
    const size_t wb = (size_t)I * O * sizeof(float);      /* 64 MiB */
    const size_t SLACK = wb / 2;                          /* driver noise allowance */
    std::vector<float> w((size_t)I * O, 0.5f);

    size_t count = 99, bytes = 99;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;

    /* Warmup: the first real allocation can trigger lazy driver/module
     * setup whose memory never returns; absorb it before any sampling. */
    {
        ColiCudaTensor *t = nullptr;
        if (!coli_cuda_tensor_upload(&t, w.data(), nullptr, 0, I, O, d0) || !t) {
            std::fprintf(stderr, "warmup upload failed\n");
            return 1;
        }
        coli_cuda_tensor_free(t);
    }

    /* ---- control: success path allocates and frees to zero delta -------- */
    size_t free_before = 0, free_now = 0;
    if (!device_free_bytes(&free_before)) return 2;
    for (int n = 0; n < N; n++) {
        ColiCudaTensor *t = nullptr;
        if (!coli_cuda_tensor_upload(&t, w.data(), nullptr, 0, I, O, d0) || !t) {
            std::fprintf(stderr, "control upload %d failed\n", n);
            return 1;
        }
        coli_cuda_tensor_free(t);
    }
    if (!device_free_bytes(&free_now)) return 2;
    long long control_delta = (long long)free_before - (long long)free_now;
    std::printf("control: %d upload+free cycles of %zu MiB, free-mem delta %lld bytes\n",
                N, wb >> 20, control_delta);
    if (control_delta > (long long)SLACK || -control_delta > (long long)SLACK) {
        std::fprintf(stderr, "control cycles moved device free memory by %lld bytes\n",
                     control_delta);
        return 1;
    }
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) {
        std::fprintf(stderr, "ledger not restored after control cycles: %zu tensors %zu bytes\n",
                     count, bytes);
        return 1;
    }

    /* ---- injected H2D failure: a failed upload must release its bytes --- */
    std::fprintf(stderr, "expecting %d upload-failure lines next (deliberate injected H2D failures):\n", N);
    g_fail_h2d_bytes = wb;
    if (!device_free_bytes(&free_before)) return 2;
    for (int n = 0; n < N; n++) {
        ColiCudaTensor *t = nullptr;
        if (coli_cuda_tensor_upload(&t, w.data(), nullptr, 0, I, O, d0) || t) {
            std::fprintf(stderr, "upload %d ignored the injected memcpy failure\n", n);
            g_fail_h2d_bytes = 0;
            return 1;
        }
        if (!device_free_bytes(&free_now)) { g_fail_h2d_bytes = 0; return 2; }
        std::printf("fail-cycle %d: free %zu bytes (cumulative delta %lld)\n",
                    n, free_now, (long long)free_before - (long long)free_now);
    }
    g_fail_h2d_bytes = 0;
    if (g_hook_hits != N) {
        std::fprintf(stderr, "injection seam engaged %d times, want %d\n", g_hook_hits, N);
        return 1;
    }
    long long leaked = (long long)free_before - (long long)free_now;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) {
        std::fprintf(stderr, "failed uploads reached the ledger: %zu tensors %zu bytes\n",
                     count, bytes);
        return 1;
    }
    if (leaked > (long long)SLACK) {
        std::fprintf(stderr,
                     "FAIL: %d failed uploads leaked %lld bytes of device memory "
                     "(%.1f MiB, ~%zu MiB per cycle expected on the defect)\n",
                     N, leaked, leaked / (1024.0 * 1024.0), wb >> 20);
        return 1;
    }

    /* ---- disarmed: the upload path still works ---------------------------- */
    {
        ColiCudaTensor *t = nullptr;
        if (!coli_cuda_tensor_upload(&t, w.data(), nullptr, 0, I, O, d0) || !t) {
            std::fprintf(stderr, "healthy upload after disarm failed\n");
            return 1;
        }
        coli_cuda_tensor_free(t);
        coli_cuda_stats(-1, &count, &bytes);
        if (count || bytes) return 1;
    }

    std::printf("weights_owned upload-failure test passed (%d fail cycles, %lld bytes cumulative)\n",
                N, leaked);
    return 0;
}
