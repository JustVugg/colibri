// zc_probe.cu: 17.5 MB host-to-device copies from pinned memory, one at a time and waited for
// (k3cu_moe_expert's), while a kernel writes floats into mapped pinned host memory at a set rate:
// k3_down's pattern (a warp: 8 rows of 4 consecutive floats, 16-byte pieces) or whole 128-byte rows.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void writer(float *y, int LT, int rows, long long spin, int iters, int coalesced) {
    const int l = threadIdx.x & 7, q = threadIdx.x >> 3, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    for (int it = 0; it < iters; it++) {
        const int o0 = ((blockIdx.x + it * gridDim.x) * 32) % LT, p0 = ((blockIdx.x * 8 + it * 64) % rows) & ~7;
        if (coalesced) y[(size_t)(p0 + wp) * LT + o0 + lane] = (float)it;   // warp w: row p0 + w, 32 floats
        else y[(size_t)(p0 + l) * LT + o0 + q] = (float)it;                  // k3_down: 8 rows x 4 floats a warp
        const long long t0 = clock64();
        while (clock64() - t0 < spin) {}
    }
}
int main(int argc, char **argv) {
    const long long spin = argc > 1 ? atoll(argv[1]) : 100000;
    const size_t E = (size_t)17u * 1024 * 1024 + 512 * 1024, NB = 40;
    char *src = nullptr; cudaHostAlloc((void **)&src, NB * E, cudaHostAllocMapped | cudaHostAllocPortable);
    memset(src, 1, NB * E);
    const int LT = 3584, rows = 8192;
    float *y = nullptr; cudaHostAlloc((void **)&y, (size_t)rows * LT * 4, cudaHostAllocMapped | cudaHostAllocPortable);
    float *yd = nullptr; cudaHostGetDevicePointer((void **)&yd, y, 0);
    char *d = nullptr; cudaMalloc(&d, 8 * E);
    cudaStream_t sc, sk; cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking); cudaStreamCreateWithFlags(&sk, cudaStreamNonBlocking);
    cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    for (int mode = -1; mode < 2; mode++) {   // -1: no writer
        const int blocks = 256, iters = 4000;
        double tw0 = now();
        if (mode >= 0) writer<<<blocks, 256, 0, sk>>>(yd, LT, rows, spin, iters, mode);
        const int n = 300;
        double t0 = now();
        for (int i = 0; i < n; i++) {
            cudaMemcpyAsync(d + (i % 8) * E, src + (i % NB) * E, E, cudaMemcpyHostToDevice, sc);
            cudaEventRecord(ev, sc); cudaEventSynchronize(ev);
        }
        const double tc = now() - t0;
        const bool busy = mode >= 0 && cudaStreamQuery(sk) == cudaErrorNotReady;
        cudaStreamSynchronize(sk);
        const double tw = now() - tw0;
        const double wb = mode >= 0 ? (double)blocks * iters * 256 * 4 : 0;
        printf("%-28s copies %.1f GB/s; writer %.2f GB/s of floats over %.1f s%s\n",
               mode < 0 ? "no writes" : mode ? "writes of 128 B rows" : "writes of 16 B (k3_down's)",
               n * E / tc / 1e9, wb / tw / 1e9, tw, mode >= 0 && !busy ? " (writer ended before the copies)" : "");
    }
    return 0;
}
