// bench_s.cu: two tile kernels with the weights decoded once a block into shared memory, against
// k3_dense_b / k3_dense_b8 (the same bytes?, T multiply-adds a second at Kimi K3's shapes):
//   ks_dense_s4 (int4 groups of 64): a thread all 8 lanes of 4 rows x 8 outputs, lane by lane in
//     hsum256's order (a running stack of three partial sums), the group's fmaf into the total;
//   ko_dense_o8 (int8 rows): 8 threads a tile of 8 rows x 16 outputs, thread l lane l over the whole
//     row; hsum256 by xor shuffles 4, 2, 1 at the end.
//   nvcc -O3 -std=c++17 -arch=sm_89 -ftz=false -I<c> bench_s.cu -o bench_s.exe
// Run against 254338ab's backend_cuda_k3.cu, whose k3_dense_launch then took k3_dense_b and k3_dense_b8 (the
// "old" column); since, k3_dense_launch takes these two kernels as k3_dense_s4 and k3_dense_o8 (renamed here
// so this file still builds beside them; bench_tiles.cu keeps the old ones). KS_EXP: the timing experiments
// (wrong outputs): 1 no staging in the loop, 2 no barrier; KS_MID 0: the int4 staging spread between the lanes.
#include "backend_cuda_k3.cu"
#include <vector>

#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);}}while(0)

#ifndef KS_EXP
#define KS_EXP 0
#endif
#ifndef KS_MID
#define KS_MID 1     // int4: the staging's 8 parts all after lane 6 (the kernel adopted; 0: one after each lane, which spills)
#endif

// ---- int4: ks_dense_s4 ----------------------------------------------------------------------------
#define KS_BR 64                    // rows a block
#define KS_BO 128                   // outputs a block
#define KS_SC 64                    // columns a stage: one group
#define KS_SMEM ((2 * KS_SC * KS_BR + 2 * KS_SC * KS_BO + 2 * KS_BO) * 4)
template <int L>
__device__ __forceinline__ void ks_lane(const float *bx, const float *bw, int xo, int wo, float (&a)[4][8]) {
    // lane L's 8 columns L, L+8, .., fused from +0: x a column's 4 rows (row quads swizzled by the column's lane), w its 8 outputs
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) a[i][j] = 0.0f;
#pragma unroll
    for (int m = 0; m < 8; m++) {
        const float4 xv = *reinterpret_cast<const float4 *>(bx + (8 * m + L) * KS_BR + xo);
        const float4 w0 = *reinterpret_cast<const float4 *>(bw + (8 * m + L) * KS_BO + wo);
        const float4 w1 = *reinterpret_cast<const float4 *>(bw + (8 * m + L) * KS_BO + wo + 4);
        const float xr[4] = {xv.x, xv.y, xv.z, xv.w}, wr[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) a[i][j] = __fmaf_rn(xr[i], wr[j], a[i][j]);
    }
}
template <int F>
__global__ void __launch_bounds__(256, 1) ks_dense_s4(const uint8_t *__restrict__ w, const float *__restrict__ scale,
                                                     const float *__restrict__ x, int S, int I, int O, float *__restrict__ y) {
    extern __shared__ __align__(16) float ks_sm[];
    float *const sx = ks_sm;                          // [2][KS_SC][KS_BR]: a column's rows, row quad q at q ^ (column & 7)
    float *const sw = ks_sm + 2 * KS_SC * KS_BR;      // [2][KS_SC][KS_BO]: a column's outputs
    float *const ss = sw + 2 * KS_SC * KS_BO;         // [2][KS_BO]: the group's scales
    const int t = threadIdx.x, wp = t >> 5, ln = t & 31;
    const int rg = 8 * (wp >> 2) + (ln & 7), og = 4 * (wp & 3) + (ln >> 3);   // rows 4 rg .. +3, outputs 8 og .. +7
    const int s0 = blockIdx.x * KS_BR, o0 = blockIdx.y * KS_BO, nst = I / KS_SC;
    // the compute side: lane L reads row quad rg at rg ^ L
    const int wo = 8 * og;
    // staging: x column xc of rows 16 xq .. +15; outputs 4u .. 4u+3, their word m
    // x: thread (cg, rq) of its warp's 8 x 4 brings columns 32 (wp & 1) + 4 cg .. +3 of rows 16 (wp >> 1) + 4 rq .. +3
    const int xcg = 32 * (wp & 1) + 4 * (ln & 7), xrq = 4 * (wp >> 1) + (ln >> 3);
    const int xr0 = s0 + 4 * xrq, xlast = S - 1 - xr0;          // row xr0 + i, clamped: xr0 + min(i, xlast)
    const float *xb = x + (size_t)xr0 * I + xcg;
    const int u = 8 * (wp & 3) + (ln & 7), m = 4 * (wp >> 2) + (ln >> 3);
    const int wlast = O - 1 - (o0 + 4 * u);                     // output o0 + 4u + k, clamped likewise
    const uint8_t *wb = w + (size_t)(o0 + 4 * u) * (I / 2) + 4 * m;
    const int wrb = I / 2;
    const int ng = I / 64;
    int so = 0;
    if (t < KS_BO) { int o = o0 + t; o = o < O ? o : O - 1; so = o * ng; }
    float4 xv[4];   // rows 0..3 of the thread's four
    float sv = 0.0f;
    uint32_t wv[4];
    auto load = [&](int st) {
#pragma unroll
        for (int i = 0; i < 4; i++) xv[i] = __ldg(reinterpret_cast<const float4 *>(xb + (size_t)(i < xlast ? i : xlast) * I + st * KS_SC));
#pragma unroll
        for (int k = 0; k < 4; k++) wv[k] = __ldg(reinterpret_cast<const uint32_t *>(wb + (size_t)(k < wlast ? k : wlast) * wrb + st * 32));
        if (t < KS_BO) sv = __ldg(scale + so + st);
    };
    // the staging in 8 parts p (after each lane): the weights of column 8m + p; x's row quad p (p < 4);
    // the scale (p 0); then the loads for the stage after: x rows 4 (p - 4) .. +3 (p >= 4), the weights (p 7)
    auto part = [&](int p, int b, int nx) {
        float *bx = sx + b * KS_SC * KS_BR, *bw = sw + b * KS_SC * KS_BO;
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) v[k] = kb_lane_v(F ? wv[k] ^ 0x88888888u : wv[k], p);
        *reinterpret_cast<float4 *>(bw + (8 * m + p) * KS_BO + 4 * u) = make_float4(v[0], v[1], v[2], v[3]);
        if (p < 4) {   // column xcg + p: its 4 rows, row quad xrq at xrq ^ (column & 7)
            const float c0 = p == 0 ? xv[0].x : p == 1 ? xv[0].y : p == 2 ? xv[0].z : xv[0].w;
            const float c1 = p == 0 ? xv[1].x : p == 1 ? xv[1].y : p == 2 ? xv[1].z : xv[1].w;
            const float c2 = p == 0 ? xv[2].x : p == 1 ? xv[2].y : p == 2 ? xv[2].z : xv[2].w;
            const float c3 = p == 0 ? xv[3].x : p == 1 ? xv[3].y : p == 2 ? xv[3].z : xv[3].w;
            *reinterpret_cast<float4 *>(bx + (xcg + p) * KS_BR + 4 * (xrq ^ ((xcg + p) & 7))) = make_float4(c0, c1, c2, c3);
        }
        if (p == 0 && t < KS_BO) ss[b * KS_BO + t] = sv;
        if (p == 1 && t < KS_BO) sv = __ldg(scale + so + nx);
        if (p >= 4) {
            const int i = p - 4;
            xv[i] = __ldg(reinterpret_cast<const float4 *>(xb + (size_t)(i < xlast ? i : xlast) * I + nx * KS_SC));
        }
        if (p == 7)
#pragma unroll
            for (int k = 0; k < 4; k++) wv[k] = __ldg(reinterpret_cast<const uint32_t *>(wb + (size_t)(k < wlast ? k : wlast) * wrb + nx * 32));
    };
    float tot[4][8], s0v[4][8], s1v[4][8], s2v[4][8], a[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) tot[i][j] = 0.0f;
    load(0);
#pragma unroll
    for (int p = 0; p < 8; p++) part(p, 0, nst > 1 ? 1 : 0);
    __syncthreads();
#pragma unroll 1
    for (int st = 0; st < nst; st++) {
        const int b = st & 1, nx = st + 2 < nst ? st + 2 : nst - 1;
        const float *bx = sx + b * KS_SC * KS_BR, *bw = sw + b * KS_SC * KS_BO;
        // hsum256's tree as the lanes finish: ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)); between the lanes the
        // next stage's staging, a part at a time
        ks_lane<0>(bx, bw, 4 * (rg ^ 0), wo, s0v);
        if (!(KS_EXP & 1) && !KS_MID) part(0, b ^ 1, nx);
        ks_lane<4>(bx, bw, 4 * (rg ^ 4), wo, a);
        if (!(KS_EXP & 1) && !KS_MID) part(1, b ^ 1, nx);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) s0v[i][j] = __fadd_rn(s0v[i][j], a[i][j]);
        ks_lane<2>(bx, bw, 4 * (rg ^ 2), wo, s1v);
        if (!(KS_EXP & 1) && !KS_MID) part(2, b ^ 1, nx);
        ks_lane<6>(bx, bw, 4 * (rg ^ 6), wo, a);
        if (!(KS_EXP & 1) && !KS_MID) part(3, b ^ 1, nx);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) s0v[i][j] = __fadd_rn(s0v[i][j], __fadd_rn(s1v[i][j], a[i][j]));
        if (!(KS_EXP & 1) && KS_MID)
#pragma unroll
            for (int p = 0; p < (KS_MID == 2 ? 4 : 8); p++) part(p, b ^ 1, nx);
        ks_lane<1>(bx, bw, 4 * (rg ^ 1), wo, s1v);
        if (!(KS_EXP & 1) && !KS_MID) part(4, b ^ 1, nx);
        ks_lane<5>(bx, bw, 4 * (rg ^ 5), wo, a);
        if (!(KS_EXP & 1) && !KS_MID) part(5, b ^ 1, nx);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) s1v[i][j] = __fadd_rn(s1v[i][j], a[i][j]);
        ks_lane<3>(bx, bw, 4 * (rg ^ 3), wo, s2v);
        if (!(KS_EXP & 1) && !KS_MID) part(6, b ^ 1, nx);
        if (!(KS_EXP & 1) && KS_MID == 2)
#pragma unroll
            for (int p = 4; p < 8; p++) part(p, b ^ 1, nx);
        ks_lane<7>(bx, bw, 4 * (rg ^ 7), wo, a);
        if (!(KS_EXP & 1) && !KS_MID) part(7, b ^ 1, nx);
        const float4 g0 = *reinterpret_cast<const float4 *>(ss + b * KS_BO + wo), g1 = *reinterpret_cast<const float4 *>(ss + b * KS_BO + wo + 4);
        const float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++)
                tot[i][j] = __fmaf_rn(__fadd_rn(s0v[i][j], __fadd_rn(s1v[i][j], __fadd_rn(s2v[i][j], a[i][j]))), g[j], tot[i][j]);
        if (!(KS_EXP & 2)) __syncthreads();
    }
    const int oo = o0 + wo;
    if (oo < O) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = s0 + 4 * rg + i;
            if (row < S) {
                *reinterpret_cast<float4 *>(y + (size_t)row * O + oo) = make_float4(tot[i][0], tot[i][1], tot[i][2], tot[i][3]);
                *reinterpret_cast<float4 *>(y + (size_t)row * O + oo + 4) = make_float4(tot[i][4], tot[i][5], tot[i][6], tot[i][7]);
            }
        }
    }
}

// ---- int8: ko_dense_o8 ----------------------------------------------------------------------------
#define KO_BR 32
#define KO_BO 128
#define KO_SC 64
#define KO_SMEM ((2 * KO_SC * KO_BR + 2 * KO_SC * KO_BO) * 4)
__global__ void __launch_bounds__(256, 1) ko_dense_o8(const int8_t *__restrict__ w, const float *__restrict__ scale,
                                                     const float *__restrict__ x, int S, int I, int O, float *__restrict__ y) {
    extern __shared__ __align__(16) float ko_sm[];
    float *const sx = ko_sm;                          // [2][KO_SC][KO_BR]: a column's rows, row quad q at q ^ (column & 7)
    float *const sw = ko_sm + 2 * KO_SC * KO_BR;      // [2][KO_SC][KO_BO]: a column's outputs, output quad f at f ^ (column & 7)
    const int t = threadIdx.x, wp = t >> 5, ln = t & 31;
    const int L = ln & 7, oc = ln >> 3, rt = oc >> 1, ct = oc & 1;   // the lane; the warp's 4 tiles: 2 of rows by 2 of outputs
    const int wr = wp >> 2, wc = wp & 3;
    const int r8 = 16 * wr + 8 * rt, o16 = 32 * wc + 16 * ct;       // the tile: rows r8 .. +7, outputs o16 .. +15
    const int s0 = blockIdx.x * KO_BR, o0 = blockIdx.y * KO_BO, nst = I / KO_SC;
    const int xa0 = 4 * ((r8 / 4) ^ L), xa1 = 4 * ((r8 / 4 + 1) ^ L);
    int wa[4];
#pragma unroll
    for (int j = 0; j < 4; j++) wa[j] = 4 * ((o16 / 4 + j) ^ L);
    const int xc = t & 63, xq = t >> 6, xl = xc & 7;
    int xo[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { int r = s0 + 8 * xq + i; r = r < S ? r : S - 1; xo[i] = r * I + xc; }
    const int u = 8 * (wp & 3) + (ln & 7), m = 4 * (wp >> 2) + (ln >> 3);
    const int8_t *wq[4];
#pragma unroll
    for (int k = 0; k < 4; k++) { int o = o0 + 4 * u + k; o = o < O ? o : O - 1; wq[k] = w + (size_t)o * I + 8 * m; }
    float xv[8];
    uint32_t wv[8];
    auto load = [&](int st) {
#pragma unroll
        for (int i = 0; i < 8; i++) xv[i] = __ldg(x + xo[i] + st * KO_SC);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint2 p = __ldg(reinterpret_cast<const uint2 *>(wq[k] + (size_t)st * KO_SC));
            wv[2 * k] = p.x ^ 0x80808080u; wv[2 * k + 1] = p.y ^ 0x80808080u;
        }
    };
    // the staging in 8 parts p (after each column): the weights of column 8m + p; x's row quad p (p < 2);
    // then the loads for the stage after: x rows 4 (p - 2) .. +3 (p 2, 3), the weights (p 7)
    auto part = [&](int p, int b, int nx) {
        float *bx = sx + b * KO_SC * KO_BR, *bw = sw + b * KO_SC * KO_BO;
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) v[k] = k8_val(wv[2 * k + (p >> 2)], p & 3);
        *reinterpret_cast<float4 *>(bw + (8 * m + p) * KO_BO + 4 * (u ^ p)) = make_float4(v[0], v[1], v[2], v[3]);
        if (p < 2)
            *reinterpret_cast<float4 *>(bx + xc * KO_BR + 4 * ((2 * xq + p) ^ xl)) = make_float4(xv[4 * p], xv[4 * p + 1], xv[4 * p + 2], xv[4 * p + 3]);
        if (p == 2 || p == 3)
#pragma unroll
            for (int i = 4 * (p - 2); i < 4 * (p - 2) + 4; i++) xv[i] = __ldg(x + xo[i] + nx * KO_SC);
        if (p == 3 || p == 7)   // the words parts 0-3 (4-7) decode, once they have
#pragma unroll
            for (int k = 0; k < 4; k++)
                wv[2 * k + (p >> 2)] = __ldg(reinterpret_cast<const uint32_t *>(wq[k] + (size_t)nx * KO_SC) + (p >> 2)) ^ 0x80808080u;
    };
    float a[8][16];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 16; j++) a[i][j] = 0.0f;
    load(0);
#pragma unroll
    for (int p = 0; p < 8; p++) part(p, 0, nst > 1 ? 1 : 0);
    __syncthreads();
#pragma unroll 1
    for (int st = 0; st < nst; st++) {
        const int b = st & 1, nx = st + 2 < nst ? st + 2 : nst - 1;
        const float *bx = sx + b * KO_SC * KO_BR + L * KO_BR, *bw = sw + b * KO_SC * KO_BO + L * KO_BO;
#pragma unroll
        for (int mm = 0; mm < 8; mm++) {   // column 8 mm + L
            const float4 x0 = *reinterpret_cast<const float4 *>(bx + mm * 8 * KO_BR + xa0);
            const float4 x1 = *reinterpret_cast<const float4 *>(bx + mm * 8 * KO_BR + xa1);
            float wr[16];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 wv4 = *reinterpret_cast<const float4 *>(bw + mm * 8 * KO_BO + wa[j]);
                wr[4 * j] = wv4.x; wr[4 * j + 1] = wv4.y; wr[4 * j + 2] = wv4.z; wr[4 * j + 3] = wv4.w;
            }
            const float xr[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 16; j++) a[i][j] = __fmaf_rn(xr[i], wr[j], a[i][j]);
            if (!(KS_EXP & 1)) part(mm, b ^ 1, nx);
        }
        if (!(KS_EXP & 2)) __syncthreads();
    }
    // hsum256: xor 4 pairs lanes l and l^4, then 2, then 1; each step keeps half the tile (the rows, then
    // the outputs, then the outputs again by the lane's bit), so lane l ends with 16 of the 128
    const int k4 = (L >> 2) & 1, k2 = (L >> 1) & 1, k1 = L & 1;
    float h1[4][16];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const float mine = k4 ? a[i + 4][j] : a[i][j], other = k4 ? a[i][j] : a[i + 4][j];
            h1[i][j] = __fadd_rn(mine, __shfl_xor_sync(0xffffffffu, other, 4));
        }
    float h2[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float mine = k2 ? h1[i][j + 8] : h1[i][j], other = k2 ? h1[i][j] : h1[i][j + 8];
            h2[i][j] = __fadd_rn(mine, __shfl_xor_sync(0xffffffffu, other, 2));
        }
    float h3[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float mine = k1 ? h2[i][j + 4] : h2[i][j], other = k1 ? h2[i][j] : h2[i][j + 4];
            h3[i][j] = __fadd_rn(mine, __shfl_xor_sync(0xffffffffu, other, 1));
        }
    // lane l holds rows r8 + 4 k4 + i, outputs o16 + 8 k2 + 4 k1 + j
    const int oo = o0 + o16 + 8 * k2 + 4 * k1;
    if (oo < O) {
        float sc[4];
#pragma unroll
        for (int j = 0; j < 4; j++) sc[j] = __ldg(scale + oo + j);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = s0 + r8 + 4 * k4 + i;
            if (row < S)
                *reinterpret_cast<float4 *>(y + (size_t)row * O + oo) =
                    make_float4(__fmul_rn(h3[i][0], sc[0]), __fmul_rn(h3[i][1], sc[1]), __fmul_rn(h3[i][2], sc[2]), __fmul_rn(h3[i][3], sc[3]));
        }
    }
}

static void ks_launch(int fmt, int flip, const uint8_t *w, const float *s, const float *x, int S, int I, int O, float *y) {
    static int set = 0;
    if (!set) {
        CK(cudaFuncSetAttribute(ks_dense_s4<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM));
        CK(cudaFuncSetAttribute(ks_dense_s4<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, KS_SMEM));
        CK(cudaFuncSetAttribute(ko_dense_o8, cudaFuncAttributeMaxDynamicSharedMemorySize, KO_SMEM));
        set = 1;
    }
    if (fmt == 4) {
        const dim3 g((unsigned)((S + KS_BR - 1) / KS_BR), (unsigned)((O + KS_BO - 1) / KS_BO));
        if (flip) ks_dense_s4<1><<<g, 256, KS_SMEM>>>(w, s, x, S, I, O, y);
        else ks_dense_s4<0><<<g, 256, KS_SMEM>>>(w, s, x, S, I, O, y);
    } else {
        const dim3 g((unsigned)((S + KO_BR - 1) / KO_BR), (unsigned)((O + KO_BO - 1) / KO_BO));
        ko_dense_o8<<<g, 256, KO_SMEM>>>(reinterpret_cast<const int8_t *>(w), s, x, S, I, O, y);
    }
}

static float timeit(int which, int fmt, int flip, const uint8_t *w, const float *s, const float *x, int S, int I, int O, float *y, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto go = [&]() {
        if (which == 0) k3_dense_launch(fmt, w, s, x, S, I, O, 64, y, 0, flip);
        else ks_launch(fmt, flip, w, s, x, S, I, O, y);
    };
    go(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) go();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

static uint32_t rng = 12345u;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float rx(int kind) {   // kind 1: zeros of both signs, subnormals, large values among them
    const uint32_t r = rnd();
    if (kind) {
        switch (r & 31) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: { uint32_t b = (rnd() & 0x007fffffu) | (r & 0x80000000u); float f; memcpy(&f, &b, 4); return f; }
        case 3: return (float)((int)(rnd() % 2001) - 1000) * 1e30f;
        default: break;
        }
    }
    return (float)((int)(rnd() % 200001) - 100000) / 100000.f * (1.0f + (r & 7));
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 10, quick = argc > 2;
    struct { const char *name; int O, I, fmt; } sh[] = {
        {"KDA q/shared gate 12288x7168 i4", 12288, 7168, 4}, {"KDA o/shared down 7168x12288 i4", 7168, 12288, 4},
        {"latent down 3584x7168 i4", 3584, 7168, 4}, {"latent up 7168x3584 i4", 7168, 3584, 4},
        {"MLA g 12288x7168 i8", 12288, 7168, 1}, {"MLA o 7168x12288 i8", 7168, 12288, 1},
        {"MLA qb 18432x1536 i8", 18432, 1536, 1}, {"MLA qa 1536x7168 i8", 1536, 7168, 1},
        {"MLA kva 576x7168 i8", 576, 7168, 1}};
    const int rows_all[] = {64, 256, 300, 512, 1024}, rows_q[] = {1024};
    const int *rows = quick ? rows_q : rows_all, nrows = quick ? 1 : 5;
    CK(cudaSetDevice(0));
    long long bad_total = 0;
    int hi = -1;
    for (auto &h : sh) {
        hi++;
        if (quick && hi != 0 && hi != 4) continue;
        const size_t wb = h.fmt == 1 ? (size_t)h.O * h.I : (size_t)h.O * h.I / 2, sn = h.fmt == 1 ? (size_t)h.O : (size_t)h.O * h.I / 64;
        std::vector<uint8_t> hw(wb); std::vector<float> hs(sn);
        for (auto &v : hw) v = (uint8_t)rnd();
        for (size_t i = 0; i < sn; i++) hs[i] = (i % 97 == 5 ? -1.0f : 1.0f) * 1e-3f * (1 + (rnd() & 255));
        const int Smax = 1024;
        std::vector<float> hx((size_t)Smax * h.I);
        for (size_t i = 0; i < hx.size(); i++) hx[i] = rx(i % 7 == 3);
        uint8_t *dw; float *ds, *dx, *dy0, *dy1;
        CK(cudaMalloc(&dw, wb)); CK(cudaMalloc(&ds, sn * 4)); CK(cudaMalloc(&dx, hx.size() * 4));
        CK(cudaMalloc(&dy0, (size_t)Smax * h.O * 4)); CK(cudaMalloc(&dy1, (size_t)Smax * h.O * 4));
        CK(cudaMemcpy(dw, hw.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds, hs.data(), sn * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        printf("%-34s", h.name);
        for (int ri = 0; ri < nrows; ri++) {
            const int S = rows[ri];
            const double mac = (double)S * h.O * h.I;
            for (int flip = 0; flip < (h.fmt == 4 && !quick ? 2 : 1); flip++) {
                CK(cudaMemset(dy0, 0xff, (size_t)S * h.O * 4)); CK(cudaMemset(dy1, 0xee, (size_t)S * h.O * 4));
                if (quick) {
                    const float t1 = timeit(1, h.fmt, h.fmt == 4, dw, ds, dx, S, h.I, h.O, dy1, reps);
                    printf(" | %4d: new %5.1f", S, mac / (t1 * 1e-3) / 1e12);
                    continue;
                }
                const float t0 = timeit(0, h.fmt, flip, dw, ds, dx, S, h.I, h.O, dy0, reps);
                const float t1 = timeit(1, h.fmt, flip, dw, ds, dx, S, h.I, h.O, dy1, reps);
                std::vector<uint32_t> a((size_t)S * h.O), b((size_t)S * h.O);
                CK(cudaMemcpy(a.data(), dy0, a.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b.data(), dy1, b.size() * 4, cudaMemcpyDeviceToHost));
                long long bad = 0;
                for (size_t i = 0; i < a.size(); i++) if (a[i] != b[i]) bad++;
                bad_total += bad;
                if (flip == 0 || h.fmt != 4)
                    printf(" | %4d: %5.1f -> %5.1f%s", S, mac / (t0 * 1e-3) / 1e12, mac / (t1 * 1e-3) / 1e12, bad ? " BAD" : "");
                else if (bad) printf(" (flip BAD %lld)", bad);
            }
        }
        printf("  T/s\n");
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy0); cudaFree(dy1);
    }
    if (!quick) printf("outputs differing: %lld\n", bad_total);
    return bad_total != 0;
}
