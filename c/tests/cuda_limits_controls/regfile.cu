// Planted control for tools/check_cuda_limits.py: no __launch_bounds__, so
// ptxas gives the 96 live accumulators all the registers they want (> 64),
// then a 1024-thread launch needs more than sm_120's 64 Ki register file.
// expect: regfile
#include <cuda_runtime.h>
__global__ void regfile_k(float *y, const float *x, int n) {
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
void regfile_launch(float *y, const float *x, int n) { regfile_k<<<1, 1024>>>(y, x, n); }
