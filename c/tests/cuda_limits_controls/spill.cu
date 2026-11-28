// Planted control for tools/check_cuda_limits.py: __launch_bounds__(1024, 2)
// caps the kernel at 32 registers, and 96 live accumulators cannot fit, so
// ptxas spills to local memory; the controls have no spill baseline.
// expect: spill
#include <cuda_runtime.h>
__global__ void __launch_bounds__(1024, 2) spill_k(float *y, const float *x, int n) {
    float acc[96];
#pragma unroll
    for (int j = 0; j < 96; j++) acc[j] = x[threadIdx.x + j * n];
    for (int it = 0; it < n; it++) {
#pragma unroll
        for (int j = 0; j < 96; j++) acc[j] = acc[j] * acc[(j + 1) % 96] + x[it + j];
    }
#pragma unroll
    for (int j = 0; j < 96; j++) y[threadIdx.x * 96 + j] = acc[j];
}
void spill_launch(float *y, const float *x, int n) { spill_k<<<1, 1024>>>(y, x, n); }
