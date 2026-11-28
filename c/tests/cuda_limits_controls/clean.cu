// Planted control for tools/check_cuda_limits.py: a kernel inside every sm_120
// limit, launched inside its bound. The gate must report nothing here (a
// finding would be a false positive in the checker itself).
// expect: none
#include <cuda_runtime.h>
#define CLEAN_THREADS 256
__global__ void __launch_bounds__(CLEAN_THREADS) clean_k(float *y, const float *x, int n) {
    __shared__ float s[CLEAN_THREADS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = i < n ? x[i] : 0.f;
    __syncthreads();
    if (i < n) y[i] = s[threadIdx.x] + s[(threadIdx.x + 1) % CLEAN_THREADS];
}
void clean_launch(float *y, const float *x, int n, cudaStream_t st) {
    clean_k<<<(n + CLEAN_THREADS - 1) / CLEAN_THREADS, CLEAN_THREADS, 0, st>>>(y, x, n);
}
