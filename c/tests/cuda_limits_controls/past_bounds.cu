// Planted control for tools/check_cuda_limits.py: a kernel declared for 128
// threads launched with 256 (cudaErrorInvalidValue / too many resources at
// run time), once through <<<>>> and once through coli_launch's form.
// expect: bounds
#include <cuda_runtime.h>
template <typename... P, typename... A>
static inline void coli_launch(void (*k)(P...), dim3 g, dim3 b, size_t sm, cudaStream_t st, A... a) { k<<<g, b, sm, st>>>(a...); }
__global__ void __launch_bounds__(128) past_bounds_k(float *y) { y[threadIdx.x] = 1.f; }
void past_bounds_launch(float *y) {
    past_bounds_k<<<1, 2 * 128>>>(y);
    coli_launch(past_bounds_k, dim3(1), dim3(256), 0, (cudaStream_t)0, y);
}
