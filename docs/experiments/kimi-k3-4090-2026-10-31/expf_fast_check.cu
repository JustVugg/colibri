// expf_fast_check.cu -- k3d_expf (float pairs) against k3d_expf_d (exp_f's bits in doubles) on every one of the
// 2^32 floats, and both functions' speed on a softmax's inputs (scores minus their max)
#include "../../../c/backend_cuda_k3.cu"
#include <cstdio>

__device__ unsigned long long d_bad, d_first;
__global__ void all_floats(uint64_t first, uint32_t n) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint32_t b = (uint32_t)(first + i);
        const float x = __uint_as_float(b);
        const uint32_t a = __float_as_uint(k3d_expf(x)), c = __float_as_uint(k3d_expf_d(x));
        if (a != c) { if (atomicAdd(&d_bad, 1ull) == 0) d_first = b; }
    }
}
template <int FAST>
__global__ void speed(const float *x, int n, float *out) {
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = x[i];
#pragma unroll 4
        for (int r = 0; r < 16; r++) s = __fadd_rn(s, FAST ? k3d_expf(__fsub_rn(v, (float)r)) : k3d_expf_d(__fsub_rn(v, (float)r)));
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
int main() {
    if (!k3cu_init(0)) { printf("init failed\n"); return 1; }
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(d_bad, &zero, 8);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (uint64_t f = 0; f < (1ull << 32); f += 1ull << 28) all_floats<<<4096, 256>>>(f, 1u << 28);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    unsigned long long bad, first; cudaMemcpyFromSymbol(&bad, d_bad, 8); cudaMemcpyFromSymbol(&first, d_first, 8);
    printf("all 2^32 floats: %llu differ%s (%.1f s) %s\n", bad, bad ? "" : "", ms / 1000, cudaGetErrorString(cudaGetLastError()));
    if (bad) { float x; uint32_t b = (uint32_t)first; memcpy(&x, &b, 4); printf("  first: %a (0x%08x)\n", x, b); }
    // speed: scores minus their max, spread over [-40, 0]
    const int n = 1 << 24;
    float *hx = (float *)malloc((size_t)n * 4), *dx, *dout;
    uint32_t r = 7; for (int i = 0; i < n; i++) { r = r * 1664525u + 1013904223u; hx[i] = -40.0f * (float)(r >> 8) / 16777216.0f; }
    cudaMalloc(&dx, (size_t)n * 4); cudaMalloc(&dout, 4096 * 256 * 4); cudaMemcpy(dx, hx, (size_t)n * 4, cudaMemcpyHostToDevice);
    for (int fast = 0; fast < 2; fast++) {
        for (int rep = 0; rep < 2; rep++) {
            cudaEventRecord(e0);
            if (fast) speed<1><<<1024, 256>>>(dx, n, dout); else speed<0><<<1024, 256>>>(dx, n, dout);
            cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
        }
        printf("%s: %.2f ms for %d exps, %.1f G/s\n", fast ? "k3d_expf (pairs)" : "k3d_expf_d (doubles)", ms, n * 16, n * 16.0 / ms / 1e6);
    }
    return 0;
}
