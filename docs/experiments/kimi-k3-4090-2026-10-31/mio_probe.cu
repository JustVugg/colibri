// mio_probe.cu: cycles a warp instruction (an SM's 16 warps) for LDS.32/.64/.128 with 8 distinct addresses
// a warp, SHFL.BFLY, and FFMA alone: what the matmul's shared-memory pipe can take.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
template <int K>
__global__ void k(float *out, int iters, long long *cyc) {
    __shared__ __align__(16) float sm[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) sm[i] = (float)i;
    __syncthreads();
    const int ln = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
    const uint32_t off = 16u * (uint32_t)(ln & 7);   // 8 distinct 16-byte pieces a warp
    uint32_t acc = ln;
    float f0 = ln, f1 = 1.0001f;
    const long long c0 = clock64();
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        const uint32_t a0 = sbase + ((off + 1024u * (uint32_t)((it + wp) & 3)) & 16383u);
#pragma unroll
        for (int u = 0; u < 16; u++) {
            uint32_t a, b, c, d;
            const uint32_t ad = sbase + ((a0 - sbase + 128u * (uint32_t)u) & 16383u);
            if (K == 0) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a) : "r"(ad)); acc ^= a; }
            if (K == 1) { asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(a), "=r"(b) : "r"(ad)); acc ^= (u & 1) ? a : b; }
            if (K == 2) { asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(ad)); acc ^= (u & 1) ? a : d; }
            if (K == 3) { asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(a) : "r"(acc), "r"(1 + (u & 1))); acc ^= a; }
            if (K == 4) { f0 = __fmaf_rn(f0, f1, 0.5f); }
        }
    }
    const long long c1 = clock64();
    if (ln == 0) atomicMax((unsigned long long *)cyc, (unsigned long long)(c1 - c0));
    if (acc == 0x12345u || f0 == 1.2345f) out[0] = (float)acc + f0;
}
int main() {
    float *o; long long *cy; cudaMalloc(&o, 4); cudaMalloc(&cy, 8);
    const int iters = 4096, warps = 16;
    const char *names[] = {"LDS.32, 8 distinct", "LDS.64, 8 distinct", "LDS.128, 8 distinct", "SHFL.BFLY (+LOP)", "FFMA chain x16 warps"};
    for (int p = 0; p < 5; p++) {
        auto run = [&](void) {
            switch (p) {
            case 0: k<0><<<128, 32 * warps>>>(o, iters, cy); break;
            case 1: k<1><<<128, 32 * warps>>>(o, iters, cy); break;
            case 2: k<2><<<128, 32 * warps>>>(o, iters, cy); break;
            case 3: k<3><<<128, 32 * warps>>>(o, iters, cy); break;
            default: k<4><<<128, 32 * warps>>>(o, iters, cy); break;
            }
        };
        run(); cudaDeviceSynchronize(); cudaMemset(cy, 0, 8); run(); cudaDeviceSynchronize();
        long long c; cudaMemcpy(&c, cy, 8, cudaMemcpyDeviceToHost);
        printf("%-24s %.2f cycles a warp instruction (an SM)\n", names[p], c / ((double)iters * 16 * warps));
    }
    return 0;
}
