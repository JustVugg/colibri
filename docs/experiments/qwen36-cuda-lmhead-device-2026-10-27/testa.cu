// testa.cu: lm_head as the chain runs it on this machine. gemv_i8_rows<1,1> copied from backend_cuda.cu (dev
// c1c06ac6, NR 1, PAD 1: a warp a row, x in shared memory), 248320 x 2048 int8 rows in the device's memory; y into
// the device's memory, then into page-locked host memory mapped into the device (ch->outd, as the chain does), then
// into the device's memory followed by a 1 MB copy to the host. Best of 20, CUDA events. The photo's probe, no PR.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define O 248320
#define I 2048
__host__ __device__ static inline int gx4(int k) { return k + (k >> 3); }
__global__ static void __launch_bounds__(256) gemv(float *__restrict__ y, const float *__restrict__ x,
        const int8_t *__restrict__ w, const float *__restrict__ sc) {
    extern __shared__ float4 xs4[];
    int I4 = I / 4;
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    for (int k = threadIdx.x; k < I4; k += 256) xs4[gx4(k)] = x4[k];
    __syncthreads();
    int lane = threadIdx.x & 31, o = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (o >= O) return;
    const int8_t *row = w + (size_t)o * I;
    float acc = 0.f;
    for (int i0 = lane * 16; i0 < I; i0 += 32 * 16) {
        int4 v = *reinterpret_cast<const int4 *>(row + i0);
        const int8_t *b = reinterpret_cast<const int8_t *>(&v);
        float wf[16];
        #pragma unroll
        for (int k = 0; k < 16; k++) wf[k] = (float)b[k];
        int k4 = i0 / 4; float4 q[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) q[j] = xs4[gx4(k4 + j)];
        const float *xv = reinterpret_cast<const float *>(q);
        #pragma unroll
        for (int k = 0; k < 16; k++) acc += xv[k] * wf[k];
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[o] = acc * sc[o];
}
int main() {
    int8_t *w; float *x, *sc, *yd, *yh, *yhd;
    cudaMalloc(&w, (size_t)O * I); cudaMemset(w, 3, (size_t)O * I);
    cudaMalloc(&x, I * 4); cudaMemset(x, 0, I * 4); cudaMalloc(&sc, O * 4); cudaMemset(sc, 0, O * 4);
    cudaMalloc(&yd, O * 4);
    cudaHostAlloc(&yh, O * 4, cudaHostAllocMapped); cudaHostGetDevicePointer((void **)&yhd, yh, 0);
    size_t smem = (size_t)(gx4(I / 4) + 1) * 16;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const char *what[3] = {"y in the device's memory          ", "y in mapped host memory (chain)   ",
                           "y in the device + 1 MB copy       "};
    for (int m = 0; m < 3; m++) {
        float best = 1e9f;
        for (int r = 0; r < 20; r++) {
            cudaEventRecord(a);
            gemv<<<O / 8, 256, smem>>>(m == 1 ? yhd : yd, x, w, sc);
            if (m == 2) cudaMemcpyAsync(yh, yd, O * 4, cudaMemcpyDeviceToHost);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); if (ms < best) best = ms;
        }
        printf("%s %.3f ms  (%.0f GB/s of weights)\n", what[m], best, (double)O * I / best / 1e6);
    }
    printf("err %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
