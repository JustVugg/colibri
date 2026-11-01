// fp32peak.cu: what the 4090 issues per clock for the instruction mixes the matmul uses
#include <cstdio>
#include <cuda_runtime.h>

/* 8 inputs x 8 weights = 64 accumulators, like a register tile; MODE 0: FFMA, 1: FMUL+FFMA,
 * 2: + the int4 nibble decode of the 8 weights each step, 3: + the int8 decode */
template <int MODE>
__global__ void k(float *out, float a0, int n) {
    float acc[8][8], xv[8], wv[8];
    unsigned q = threadIdx.x * 2654435761u + blockIdx.x;
#pragma unroll
    for (int r = 0; r < 8; r++) { xv[r] = a0 + r * 1e-3f + threadIdx.x * 1e-7f; wv[r] = 1.0f - r * 1e-3f; }
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[r][j] = (r * 8 + j) * 1e-3f;
    const float s = 0.5f;
    for (int it = 0; it < n; it++) {
        if (MODE >= 2) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned b = (q >> (j * 3)) & 255u;
                wv[j] = MODE == 2 ? __int_as_float(((b >> (q & 4)) & 15u) ^ 0x4B000008u) - 8388616.0f
                                  : __int_as_float(b ^ 0x4B000080u) - 8388736.0f;
            }
            q = q * 1664525u + 1013904223u;
        }
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
            for (int j = 0; j < 8; j++) {
                if (MODE == 0 || MODE == 3) acc[r][j] = __fmaf_rn(xv[r], wv[j], acc[r][j]);
                else acc[r][j] = __fmaf_rn(__fmul_rn(xv[r], wv[j]), s, acc[r][j]);
            }
#pragma unroll
        for (int r = 0; r < 8; r++) xv[r] = __fmul_rn(xv[r], 1.0000001f);
    }
    float t = 0;
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
        for (int j = 0; j < 8; j++) t += acc[r][j];
    if (t == 1.2345f) out[0] = t + q;
}

int main() {
    float *o; cudaMalloc(&o, 4);
    int n = 2048;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int mode = 0; mode < 4; mode++) {
        for (int rep = 0; rep < 2; rep++) {
            cudaEventRecord(a);
            if (mode == 0) k<0><<<128 * 8, 256>>>(o, 1.f, n);
            if (mode == 1) k<1><<<128 * 8, 256>>>(o, 1.f, n);
            if (mode == 2) k<2><<<128 * 8, 256>>>(o, 1.f, n);
            if (mode == 3) k<3><<<128 * 8, 256>>>(o, 1.f, n);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            double macs = 128.0 * 8 * 256 * n * 64;
            int fp = (mode == 0 || mode == 3) ? 1 : 2;
            if (rep) printf("mode %d: %.2f T MAC/s, %.2f T FP32 instr/s (MACs only)\n", mode, macs / ms / 1e9, macs * fp / ms / 1e9);
        }
    }
    return 0;
}
