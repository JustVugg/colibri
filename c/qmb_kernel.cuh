/* qmb_kernel.cuh -- quant_matmul_blocked and its input copy, in one place for the two
 * modules that run it: coli_cuda.dll (backend_cuda.cu, coli_cuda_matmul_tiled) and
 * coli_cuda_k3.dll (backend_cuda_k3.cu, Kimi K3's KDA layer on the matrices coli_cuda.dll
 * holds), so both give the same bytes. Include it once per translation unit. */
#ifndef QMB_KERNEL_CUH
#define QMB_KERNEL_CUH
#include <cstdint>

/* quant_matmul_tiled's bytes again, register-blocked for a prompt step at Kimi K3's trunk
 * shapes (whole 256-column chunks, I a template constant, O a multiple of QMB_RO, int4 in
 * groups of 64): thread t still sums quant_matmul's columns t, t + 256, ... for QMB_RS rows
 * by QMB_RO outputs, with quant_matmul's product and fused add in its column order. The
 * rows come from a copy in blocks of QMB_RS (qmb_rows: a column's QMB_RS rows adjacent, two
 * 16-byte loads a step), each weight is decoded once for QMB_RS rows ((byte & m) ^ c reads
 * as 2^23 + 16^p (n + 8), p the nibble's place; one fused add takes the place and the 8
 * out, exactly), the scales sit in shared memory by group, and the next step's inputs and
 * weights are in flight during this one. Then quant_matmul's tree: the levels of 128, 64
 * and 32 across the warps (thread t's warp is t >> 5), 16 to 1 within a warp. Measured on
 * the RTX 4090 at 2,637 rows: 12288x7168 int4 72 -> 24 ms, 7168x12288 int8 109 -> 27 ms
 * (the int4 one bound by its instructions: a weight's decode, the product, the fused add).
 * COLI_CUDA_QMB=0: quant_matmul_tiled again. */
#define QMB_RS 8
#define QMB_RO 16
#define QMB_P 16                     /* (row, output) pairs a round of the tree */
__global__ static void qmb_rows(float *__restrict__ xt, const float *__restrict__ x, int S, int I, long long n) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const int sb = (int)(e / I), i = (int)(e % I);
    float v[QMB_RS];
#pragma unroll
    for (int r = 0; r < QMB_RS; r++) { const int s = sb * QMB_RS + r; v[r] = s < S ? x[(size_t)s * I + i] : 0.0f; }
    float4 *d = reinterpret_cast<float4 *>(xt + (size_t)e * QMB_RS);
#pragma unroll
    for (int q = 0; q < QMB_RS / 4; q++) d[q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
}
template <int FMT, int I>
__global__ static void __launch_bounds__(256, 1) quant_matmul_blocked(float *__restrict__ y, const float *__restrict__ xt,
        const uint8_t *__restrict__ w, const float *__restrict__ scales, int S, int O) {
    constexpr int RS = QMB_RS, RO = QMB_RO, K = I / 256, RB = FMT == 4 ? I / 2 : I, NG = FMT == 4 ? I / 64 : 1;
    static_assert(I % 256 == 0 && (RS * RO) % QMB_P == 0 && RS % 4 == 0 && RO % 4 == 0, "shape");
    __shared__ __align__(16) float ssc[NG][RO];
    __shared__ float red[QMB_P][256];
    const int t = threadIdx.x, o0 = blockIdx.x * RO, sb = blockIdx.y;
    if (FMT == 4) {
        for (int e = t; e < NG * RO; e += 256) { const int j = e / NG, g = e % NG; ssc[g][j] = scales[(size_t)(o0 + j) * NG + g]; }
        __syncthreads();
    }
    float acc[RS][RO];
#pragma unroll
    for (int r = 0; r < RS; r++)
#pragma unroll
        for (int j = 0; j < RO; j++) acc[r][j] = 0.0f;
    const float *xp = xt + ((size_t)sb * I + t) * RS;
    const uint8_t *wp = w + (size_t)o0 * RB + (FMT == 4 ? (t >> 1) : t);
    const uint32_t dm = (t & 1) ? 0xF0u : 0x0Fu, dc = (t & 1) ? 0x4B000080u : 0x4B000008u;
    const float dmul = (t & 1) ? 0.0625f : 1.0f, dadd = (t & 1) ? -524296.0f : -8388616.0f;
    float xn[RS];
    uint32_t bn[RO];
#pragma unroll
    for (int q = 0; q < RS / 4; q++) {
        const float4 v = __ldg(reinterpret_cast<const float4 *>(xp) + q);
        xn[4 * q] = v.x; xn[4 * q + 1] = v.y; xn[4 * q + 2] = v.z; xn[4 * q + 3] = v.w;
    }
#pragma unroll
    for (int j = 0; j < RO; j++) bn[j] = __ldg(wp + (size_t)j * RB);
#pragma unroll 2
    for (int k = 0; k < K; k++) {
        float xv[RS];
        uint32_t b[RO];
#pragma unroll
        for (int r = 0; r < RS; r++) xv[r] = xn[r];
#pragma unroll
        for (int j = 0; j < RO; j++) b[j] = bn[j];
        if (k + 1 < K) {
#pragma unroll
            for (int q = 0; q < RS / 4; q++) {
                const float4 v = __ldg(reinterpret_cast<const float4 *>(xp + (size_t)(k + 1) * 256 * RS) + q);
                xn[4 * q] = v.x; xn[4 * q + 1] = v.y; xn[4 * q + 2] = v.z; xn[4 * q + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < RO; j++) bn[j] = __ldg(wp + (size_t)j * RB + (size_t)(k + 1) * (FMT == 4 ? 128 : 256));
        }
        float sc[RO];
        if (FMT == 4) {
            const int g = 4 * k + (t >> 6);
#pragma unroll
            for (int q = 0; q < RO / 4; q++) {
                const float4 v = *reinterpret_cast<const float4 *>(&ssc[g][4 * q]);
                sc[4 * q] = v.x; sc[4 * q + 1] = v.y; sc[4 * q + 2] = v.z; sc[4 * q + 3] = v.w;
            }
        }
#pragma unroll
        for (int j = 0; j < RO; j++) {
            float wv;
            if (FMT == 4) wv = __fmaf_rn(__int_as_float((b[j] & dm) ^ dc), dmul, dadd);
            else wv = __int_as_float(b[j] ^ 0x4B000080u) - 8388736.0f;
#pragma unroll
            for (int r = 0; r < RS; r++) {
                if (FMT == 4) acc[r][j] = __fmaf_rn(__fmul_rn(xv[r], wv), sc[j], acc[r][j]);
                else acc[r][j] = __fmaf_rn(xv[r], wv, acc[r][j]);
            }
        }
    }
    const int h = t >> 5, c = t & 31;
#pragma unroll
    for (int p0 = 0; p0 < RS * RO; p0 += QMB_P) {
#pragma unroll
        for (int p = 0; p < QMB_P; p++) red[p][t] = acc[(p0 + p) / RO][(p0 + p) % RO];
        __syncthreads();
#pragma unroll
        for (int p = h; p < QMB_P; p += 8) {
            float v[8];
#pragma unroll
            for (int q = 0; q < 8; q++) v[q] = red[p][32 * q + c];
            const float a0 = v[0] + v[4], a1 = v[1] + v[5], a2 = v[2] + v[6], a3 = v[3] + v[7];
            float s = (a0 + a2) + (a1 + a3);
            s += __shfl_down_sync(0xffffffffu, s, 16);
            s += __shfl_down_sync(0xffffffffu, s, 8);
            s += __shfl_down_sync(0xffffffffu, s, 4);
            s += __shfl_down_sync(0xffffffffu, s, 2);
            s += __shfl_down_sync(0xffffffffu, s, 1);
            const int r = (p0 + p) / RO, j = (p0 + p) % RO, row = sb * RS + r, o = o0 + j;
            if (!c && row < S) y[(size_t)row * O + o] = FMT == 1 ? s * scales[o] : s;
        }
        __syncthreads();
    }
}
#endif
