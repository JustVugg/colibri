// expf_fallback_count.cu -- how many floats k3d_expf's float-pair path hands to k3d_expf_d: in [-87, 88]
// (the uncertain roundings) and past it
#include "../../../c/backend_cuda_k3.cu"
#include <cstdio>

// k3d_expf's test alone: 1 where the pair decides, 0 where it falls back
__device__ int pair_sure(float x) {
    if (!(x >= -87.0f && x <= 88.0f)) return 0;
    const float kf = rintf(__fmul_rn(x, 0x1.715476p+6f));
    const int k = (int)kf, j = k & 63;
    const float r1 = __fmaf_rn(-kf, 0x1.63p-7f, x);
    const float ph = __fmul_rn(kf, -0x1.bd0106p-19f), pl = __fmaf_rn(kf, -0x1.bd0106p-19f, -ph);
    const float sh = __fsub_rn(r1, ph), bb = __fsub_rn(sh, r1);
    const float er = __fadd_rn(__fsub_rn(r1, __fsub_rn(sh, bb)), __fsub_rn(-ph, bb));
    const float rl = __fsub_rn(__fsub_rn(er, pl), __fmul_rn(kf, 0x1.cf79acp-46f));
    const float qh = __fmul_rn(sh, sh), ql = __fmaf_rn(sh, sh, -qh), hq = __fmul_rn(qh, 0.5f);
    const float tail = __fmul_rn(__fmul_rn(qh, sh), __fmaf_rn(__fmaf_rn(sh, 1.0f / 120.0f, 1.0f / 24.0f), sh, 1.0f / 6.0f));
    const float Eh = __fadd_rn(sh, hq);
    float El = __fadd_rn(__fsub_rn(hq, __fsub_rn(Eh, sh)), rl);
    El = __fadd_rn(El, __fmul_rn(ql, 0.5f));
    El = __fmaf_rn(sh, rl, El);
    El = __fmaf_rn(hq, rl, El);
    El = __fadd_rn(El, tail);
    const float2 T = __ldg(&k3d_e2j[j]);
    const float p2 = __fmul_rn(T.x, Eh), p2l = __fmaf_rn(T.x, Eh, -p2);
    const float yh = __fadd_rn(T.x, p2);
    float yl = __fadd_rn(__fsub_rn(p2, __fsub_rn(yh, T.x)), p2l);
    yl = __fadd_rn(yl, T.y);
    yl = __fmaf_rn(T.x, El, yl);
    yl = __fmaf_rn(T.y, Eh, yl);
    const float f = __fadd_rn(yh, yl), d = __fsub_rn(yl, __fsub_rn(f, yh));
    const uint32_t fb = __float_as_uint(f);
    const float hu = __uint_as_float((fb & 0x7f800000u) - (24u << 23)), hb = fb & 0x007fffffu ? hu : __fmul_rn(hu, 0.5f);
    return d >= 0.0f ? d < __fsub_rn(hu, 0x1p-42f) : -d < __fsub_rn(hb, 0x1p-42f);
}
__device__ unsigned long long d_in, d_unsure, d_out;
__global__ void count(uint64_t first, uint32_t n) {
    unsigned long long in = 0, unsure = 0, out = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float x = __uint_as_float((uint32_t)(first + i));
        if (x >= -87.0f && x <= 88.0f) { in++; unsure += !pair_sure(x); } else out++;
    }
    atomicAdd(&d_in, in); atomicAdd(&d_unsure, unsure); atomicAdd(&d_out, out);
}
int main() {
    if (!k3cu_init(0)) { printf("init failed\n"); return 1; }
    for (uint64_t f = 0; f < (1ull << 32); f += 1ull << 28) count<<<4096, 256>>>(f, 1u << 28);
    unsigned long long in, unsure, out;
    cudaMemcpyFromSymbol(&in, d_in, 8); cudaMemcpyFromSymbol(&unsure, d_unsure, 8); cudaMemcpyFromSymbol(&out, d_out, 8);
    printf("floats in [-87, 88]: %llu, the pair unsure on %llu (1 in %.0f); past it or NaN: %llu\n", in, unsure,
           (double)in / (double)unsure, out);
    return 0;
}
