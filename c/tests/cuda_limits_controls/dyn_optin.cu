// Planted control for tools/check_cuda_limits.py: 64 KiB of dynamic shared
// memory with no cudaFuncSetAttribute(MaxDynamicSharedMemorySize): the launch
// fails on any card, the 48 KiB default is never raised.
// expect: optin
#include <cuda_runtime.h>
__global__ void __launch_bounds__(256) dyn_optin_k(float *y) {
    extern __shared__ float dyn[];
    dyn[threadIdx.x] = (float)threadIdx.x;
    __syncthreads();
    y[threadIdx.x] = dyn[(threadIdx.x * 5) % 256];
}
void dyn_optin_launch(float *y) { dyn_optin_k<<<1, 256, 65536>>>(y); }
