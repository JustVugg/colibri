// lds_probe.cu: what a warp's 16-byte shared-memory load costs on this card, by its pattern of addresses:
// cycles a warp instruction (per SM: all warps' loads over the kernel's clocks).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// pattern p: the float4 index thread ln reads (of 256 float4s = 4 KB), before the per-iteration offset
__device__ __forceinline__ int pat(int p, int ln) {
    switch (p) {
    case 0: return ln;                                   // 32 distinct consecutive: 512 B
    case 1: return 0;                                    // one address
    case 2: return ln & 7;                               // 8 distinct consecutive (128 B), each 4 threads (ln, ln+8, ..)
    case 3: return ln >> 2;                              // 8 distinct consecutive, each by 4 consecutive threads
    case 4: { const int q = ln & 3, b0 = q & 1, rt = ln >> 4, la = 2 * b0 + (q >> 1);   // the kernel's x loads
              const int s = ((la & 1) << 2) | (la & 2) | (la >> 2);
              return la * 8 + ((2 * rt + b0) ^ s); }
    case 5: { const int q = ln & 3, b1 = q >> 1, ct = (ln >> 2) & 3, la = 2 * (q & 1) + b1;   // the kernel's w loads
              return la * 32 + 2 * ct + b1; }
    case 6: return (ln & 15);                            // 16 distinct consecutive (256 B), each by 2 threads
    case 7: return ln >> 1;                              // 16 distinct, each by 2 consecutive threads
    default: return 0;
    }
}

template <int P>
__global__ void k(float *out, int iters, long long *cyc) {
    __shared__ __align__(16) float4 sm[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) sm[i] = make_float4(i, i + 1, i + 2, i + 3);
    __syncthreads();
    const int ln = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int base = pat(P, ln);
    uint32_t acc = 0;
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
    const long long c0 = clock64();
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        const uint32_t a0 = 16u * (uint32_t)((base + 256 * ((it + wp) & 3)) & 1023);
#pragma unroll
        for (int u = 0; u < 16; u++) {
            uint32_t a, b, c, d;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(sbase + ((a0 + 1024u * (uint32_t)u) & 16383u)));
            acc ^= (u & 1) ? a : d;
        }
    }
    const long long c1 = clock64();
    if (ln == 0) atomicMax((unsigned long long *)cyc, (unsigned long long)(c1 - c0));
    if (acc == 0x12345u) out[0] = (float)acc;
}

int main() {
    float *o; long long *cy; cudaMalloc(&o, 4); cudaMalloc(&cy, 8);
    const int iters = 4096, warps = 16;
    const char *names[] = {"32 distinct", "1 address", "8 distinct (ln&7)", "8 distinct (ln>>2)", "kernel x load", "kernel w load",
                           "16 distinct (ln&15)", "16 distinct (ln>>1)"};
    for (int p = 0; p < 8; p++) {
        cudaMemset(cy, 0, 8);
        auto run = [&](void) {
            switch (p) {
            case 0: k<0><<<128, 32 * warps>>>(o, iters, cy); break;
            case 1: k<1><<<128, 32 * warps>>>(o, iters, cy); break;
            case 2: k<2><<<128, 32 * warps>>>(o, iters, cy); break;
            case 3: k<3><<<128, 32 * warps>>>(o, iters, cy); break;
            case 4: k<4><<<128, 32 * warps>>>(o, iters, cy); break;
            case 5: k<5><<<128, 32 * warps>>>(o, iters, cy); break;
            case 6: k<6><<<128, 32 * warps>>>(o, iters, cy); break;
            default: k<7><<<128, 32 * warps>>>(o, iters, cy); break;
            }
        };
        run(); cudaDeviceSynchronize(); cudaMemset(cy, 0, 8); run(); cudaDeviceSynchronize();
        long long c; cudaMemcpy(&c, cy, 8, cudaMemcpyDeviceToHost);
        const double loads = (double)iters * 16 * warps;   // warp instructions a block (one block an SM)
        printf("%-22s %.2f cycles a warp's LDS.128\n", names[p], c / loads);
    }
    return 0;
}
