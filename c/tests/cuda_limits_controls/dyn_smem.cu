// Planted control for tools/check_cuda_limits.py: 40 KiB static plus a 64 KiB
// dynamic opt-in is 104 KiB, over sm_120's 99 KiB a block.
// expect: smem
#include <cuda_runtime.h>
__global__ void dyn_smem_k(float *y) {
    __shared__ float fixed[10240];
    extern __shared__ float dyn[];
    fixed[threadIdx.x] = dyn[threadIdx.x];
    __syncthreads();
    y[threadIdx.x] = fixed[(threadIdx.x * 3) % 10240];
}
void dyn_smem_launch(float *y) {
    cudaFuncSetAttribute(dyn_smem_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
    dyn_smem_k<<<1, 256, 64 * 1024>>>(y);
}
