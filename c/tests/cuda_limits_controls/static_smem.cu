// Planted control for tools/check_cuda_limits.py: 100 KiB of static shared
// memory, over sm_120's 48 KiB static limit (ptxas refuses it: the gate must
// report that refusal as a shared-memory finding).
// expect: smem
#include <cuda_runtime.h>
__global__ void static_smem_k(float *y) {
    __shared__ float big[25600];
    big[threadIdx.x] = (float)threadIdx.x;
    __syncthreads();
    y[threadIdx.x] = big[(threadIdx.x * 7) % 25600];
}
