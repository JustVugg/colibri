// cuda_q38_moe.cu -- Qwen3.8's routed experts for a prompt's rows on the CUDA device, in the CPU's
// arithmetic to the bit: the chain's q38_moe and q38_sum ops (cuda_chain.h), driven by
// qwen38_cuda_chain.h's layer-major prompt (q38cc_forward_f8).
//
// qwen38_core.h's grouped path computes an expert's gate and up for a row as q38_fp8_rows does: for each
// 128 x 128 block of the e4m3 matrix, eight lanes of fused multiply-adds (lane l: the block's columns
// l, l+8, .., l+120 in that order, from +0), hsum256's tree ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)), and
// the block's scale fused into a double total, fma(scale, part, total), rounded to float at the end;
// then h = (g sigmoid(g)) u, the sigmoid through exp_f (1/(1+exp_f(-g)) from 0 on, else z/(1+z) with
// z = exp_f(g)); down the same way over h; and each row's experts in rank order, fma(gate, out, sum)
// from +0. The kernels here do exactly that, so a prompt's routed sum is the CPU's bytes.
//
// q38m_mm is Kimi K3's k3_dense_o8 tile (backend_cuda_k3.cu) for e4m3 rows: a block stages 64 columns of
// its 32 rows and of its 128 outputs at a time in shared memory (the weights decoded once, as floats), 8
// threads a tile of 8 rows by 16 outputs, thread l of them lane l; after each 128-column block the lanes
// go through hsum256's tree by xor shuffles 4, 2, 1 (each keeping half of what is left), and each thread
// fuses its 16 sums with the block's scale into double totals. The gate/up form's 128 outputs are 64
// rows of gate and the same 64 of up, so its epilogue has both for h. A block's rows are assignments of
// one expert (a row tile of the batch table: blockIdx.x), its outputs blockIdx.y's.
//
// exp_f: backend_cuda_k3.cu's k3d_expf (a float pair at FP32's rate, double arithmetic and the table of
// k3_exact_tables.h for the floats it leaves undecided), checked there on all 2^32 floats; copied here so
// that coli_cuda.dll does not depend on coli_cuda_k3.dll.
//
// CUDA only: the lanes are a 32-thread warp's. Under HIP this file is empty and cuda_chain.cu's table has
// no q38 ops (cc_q38_ready says so). tests/test_cuda_q38_moe.cu checks it against the CPU's arithmetic.
#if !defined(__HIP__) && !defined(__HIP_PLATFORM_AMD__)
#if defined(__USE_FAST_MATH__) || defined(__FAST_MATH__)
#error "cuda_q38_moe.cu is the CPU's arithmetic to the bit: no fast math (and -ftz=false, the Makefile's)"
#endif
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "cuda_chain.h"
#include "k3_exact.h"
#include "k3_exact_tables.h"

#define Q38M_SC 64                  // columns a stage (a 128-column block is two)
#define Q38M_BR CC_Q38_TILE         // rows a block: 32 assignments of one expert
#define Q38M_BO 128                 // outputs a block
#define Q38M_SMEM ((2 * Q38M_SC * Q38M_BR + 2 * Q38M_SC * Q38M_BO) * 4)

// ---- exp_f, exact -------------------------------------------------------------------------------
__device__ uint32_t q38m_exp_in[K3X_EXP_N], q38m_exp_out[K3X_EXP_N];
__device__ __forceinline__ float q38m_expf_d(float x) {
    float f;
    if (k3x_expf_fast(x, &f)) return f;
    return k3x_table(q38m_exp_in, q38m_exp_out, K3X_EXP_N, x);
}
__device__ float2 q38m_e2j[64] = {
    {0x1.000000p+0f, 0x0.0p+0f}, {0x1.02c9a4p+0f, -0x1.887fa0p-28f}, {0x1.059b0ep+0f, -0x1.9d4f52p-25f},
    {0x1.087452p+0f, -0x1.e2990ep-26f}, {0x1.0b5586p+0f, 0x1.9f3122p-25f}, {0x1.0e3ec4p+0f, -0x1.a585ccp-25f},
    {0x1.11301ep+0f, -0x1.fdb496p-25f}, {0x1.1429aap+0f, 0x1.d525bcp-25f}, {0x1.172b84p+0f, -0x1.c15742p-27f},
    {0x1.1a35bep+0f, 0x1.6df96ep-25f}, {0x1.1d4874p+0f, -0x1.d2e8cap-25f}, {0x1.2063b8p+0f, 0x1.0c519ap-25f},
    {0x1.2387a6p+0f, 0x1.ceac48p-25f}, {0x1.26b456p+0f, 0x1.789f38p-26f}, {0x1.29e9e0p+0f, -0x1.5c0424p-25f},
    {0x1.2d285ap+0f, 0x1.b900c2p-26f}, {0x1.306fe0p+0f, 0x1.4636e2p-25f}, {0x1.33c08cp+0f, -0x1.b37d20p-25f},
    {0x1.371a74p+0f, -0x1.18aac6p-25f}, {0x1.3a7db4p+0f, -0x1.634c02p-25f}, {0x1.3dea64p+0f, 0x1.824684p-25f},
    {0x1.4160a2p+0f, 0x1.f72e2ap-28f}, {0x1.44e086p+0f, 0x1.8624b4p-30f}, {0x1.486a2cp+0f, -0x1.47d866p-25f},
    {0x1.4bfdaep+0f, -0x1.593abcp-25f}, {0x1.4f9b28p+0f, -0x1.2c5a6cp-25f}, {0x1.5342b6p+0f, -0x1.2c5610p-25f},
    {0x1.56f474p+0f, -0x1.295b04p-25f}, {0x1.5ab07ep+0f, -0x1.5bd5ecp-27f}, {0x1.5e76f2p+0f, -0x1.4a5bd6p-25f},
    {0x1.6247ecp+0f, -0x1.f8b550p-25f}, {0x1.662388p+0f, 0x1.2a9112p-27f}, {0x1.6a09e6p+0f, 0x1.9fcef4p-26f},
    {0x1.6dfb24p+0f, -0x1.cd72e8p-27f}, {0x1.71f75ep+0f, 0x1.1d8beep-25f}, {0x1.75feb6p+0f, -0x1.37b306p-25f},
    {0x1.7a1148p+0f, -0x1.829fd0p-25f}, {0x1.7e2f34p+0f, -0x1.261634p-25f}, {0x1.82589ap+0f, -0x1.accc7cp-26f},
    {0x1.868d9ap+0f, -0x1.2edb44p-26f}, {0x1.8ace54p+0f, 0x1.15506ep-27f}, {0x1.8f1aeap+0f, -0x1.baa232p-26f},
    {0x1.93737cp+0f, -0x1.e64744p-25f}, {0x1.97d82ap+0f, -0x1.0d8d84p-31f}, {0x1.9c4918p+0f, 0x1.51f848p-27f},
    {0x1.a0c668p+0f, -0x1.2886a6p-26f}, {0x1.a5503cp+0f, -0x1.b83b54p-25f}, {0x1.a9e6b6p+0f, -0x1.50c048p-25f},
    {0x1.ae89fap+0f, -0x1.a94b14p-26f}, {0x1.b33a2cp+0f, -0x1.ec3a82p-26f}, {0x1.b7f770p+0f, -0x1.a09438p-25f},
    {0x1.bcc1eap+0f, -0x1.f687c6p-25f}, {0x1.c199bep+0f, -0x1.3d56b2p-27f}, {0x1.c67f12p+0f, 0x1.cafa2ap-25f},
    {0x1.cb720ep+0f, -0x1.8837ccp-27f}, {0x1.d072d4p+0f, 0x1.40f130p-25f}, {0x1.d5818ep+0f, -0x1.822dbcp-27f},
    {0x1.da9e60p+0f, 0x1.ed9942p-27f}, {0x1.dfc974p+0f, -0x1.908c94p-25f}, {0x1.e502eep+0f, 0x1.e2cffep-26f},
    {0x1.ea4afap+0f, 0x1.52486cp-27f}, {0x1.efa1bep+0f, 0x1.cc2b44p-25f}, {0x1.f50766p+0f, -0x1.246eb0p-26f},
    {0x1.fa7c18p+0f, 0x1.9e90d8p-28f}};
__device__ __forceinline__ float q38m_expf(float x) {
    if (!(x >= -87.0f && x <= 88.0f)) return q38m_expf_d(x);
    const float kf = rintf(__fmul_rn(x, 0x1.715476p+6f));
    const int k = (int)kf, j = k & 63, m = k >> 6;
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
    const float2 T = q38m_e2j[j];
    const float p2 = __fmul_rn(T.x, Eh), p2l = __fmaf_rn(T.x, Eh, -p2);
    const float yh = __fadd_rn(T.x, p2);
    float yl = __fadd_rn(__fsub_rn(p2, __fsub_rn(yh, T.x)), p2l);
    yl = __fadd_rn(yl, T.y);
    yl = __fmaf_rn(T.x, El, yl);
    yl = __fmaf_rn(T.y, Eh, yl);
    const float f = __fadd_rn(yh, yl), d = __fsub_rn(yl, __fsub_rn(f, yh));
    const uint32_t fb = __float_as_uint(f);
    const float hu = __uint_as_float((fb & 0x7f800000u) - (24u << 23)), hb = fb & 0x007fffffu ? hu : __fmul_rn(hu, 0.5f);
    if (d >= 0.0f ? d < __fsub_rn(hu, 0x1p-42f) : -d < __fsub_rn(hb, 0x1p-42f))
        return __fmul_rn(f, __int_as_float((m + 127) << 23));
    return q38m_expf_d(x);
}
// q38_silu: x * q38_sigmoid(x), q38_sigmoid's two branches
__device__ __forceinline__ float q38m_silu(float x) {
    float s;
    if (x >= 0.0f) { const float z = q38m_expf(-x); s = __fdiv_rn(1.0f, __fadd_rn(1.0f, z)); }
    else { const float z = q38m_expf(x); s = __fdiv_rn(z, __fadd_rn(1.0f, z)); }
    return __fmul_rn(x, s);
}
// e4m3_decode8's value of byte b: its seven low bits at bit 20, its sign at 31, times 2^120 (exact, the
// subnormals too); 0x7f and 0xff the quiet NaN its blend puts there
__device__ __forceinline__ float q38m_e4m3(uint32_t b) {
    const uint32_t lo = b & 0x7fu;
    if (lo == 0x7fu) return __uint_as_float(0x7fc00000u);
    return __fmul_rn(__uint_as_float(((b & 0x80u) << 24) | (lo << 20)), 0x1p120f);
}

// ---- the matrices -------------------------------------------------------------------------------
// UP 1: rows of gate and up (D inputs, the segment's rows x picked by rows[]) into h (I floats an
// assignment, from place hp0 on); UP 0: down (I inputs: h from place hp0 on) into y (D floats an
// assignment). An assignment's index is its place in the segment's expert order (CcQ38Exp.p0 + its
// rank among the expert's rows).
template <int UP>
__global__ void __launch_bounds__(256, 1) q38m_mm(const float *__restrict__ x, const int *__restrict__ rows,
                                                 const CcQ38Exp *__restrict__ tab, int ntab, int D, int I, int hp0,
                                                 float *__restrict__ y) {
    extern __shared__ __align__(16) float q38m_sm[];
    float *const sx = q38m_sm;                         // [2][SC][BR]: a column's rows
    float *const sw = sx + 2 * Q38M_SC * Q38M_BR;      // [2][SC][BO]: a column's outputs
    __shared__ int s_e;
    const int t = threadIdx.x;
    if (t == 0) {   // the block's expert: the last entry whose first tile is at most this one (t0 ascending)
        int lo = 0, hi = ntab - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (__ldg(&tab[mid].t0) <= (int)blockIdx.x) lo = mid; else hi = mid - 1;
        }
        s_e = lo;
    }
    __syncthreads();
    const CcQ38Exp *ex = tab + s_e;
    const int n = ex->n, p0 = ex->p0, a0 = ((int)blockIdx.x - ex->t0) * Q38M_BR;
    const int K = UP ? D : I, nst = K / Q38M_SC, nblk = K / 128;
    const int o0 = UP ? (int)blockIdx.y * 64 : (int)blockIdx.y * Q38M_BO;
    const int wp = t >> 5, ln = t & 31, L = ln & 7, tl = ln >> 3;
    const int r8 = 16 * (wp >> 2) + 8 * (tl >> 1), o16 = 32 * (wp & 3) + 16 * (tl & 1);   // the tile's rows, outputs
    const int xa0 = 4 * ((r8 / 4) ^ L), xa1 = 4 * ((r8 / 4 + 1) ^ L);
    int wa[4];
#pragma unroll
    for (int j = 0; j < 4; j++) wa[j] = 4 * ((o16 / 4 + j) ^ L);
    // the staging: x column xc of rows 8 xq .. +7; bytes 8m .. 8m+7 of outputs 4u .. 4u+3. Rows past the
    // expert's n read its last row (not stored).
    const int xc = t & 63, xq = t >> 6, xl = xc & 7;
    int xo[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int a = a0 + 8 * xq + i;
        a = a < n ? a : n - 1;
        xo[i] = UP ? __ldg(rows + p0 + a) * D + xc : (p0 + a - hp0) * I + xc;
    }
    const int u = 8 * (wp & 3) + (ln & 7), m = 4 * (wp >> 2) + (ln >> 3);
    const uint8_t *wq[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int oi = 4 * u + k;
        wq[k] = UP ? (const uint8_t *)(oi < 64 ? ex->g : ex->u) + (size_t)(o0 + (oi & 63)) * D + 8 * m
                   : (const uint8_t *)ex->d + (size_t)(o0 + oi) * I + 8 * m;
    }
    float xv[8];
    uint32_t wv[8];
    auto load = [&](int st) {
#pragma unroll
        for (int i = 0; i < 8; i++) xv[i] = __ldg(x + xo[i] + st * Q38M_SC);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint2 q = __ldg(reinterpret_cast<const uint2 *>(wq[k] + (size_t)st * Q38M_SC));
            wv[2 * k] = q.x; wv[2 * k + 1] = q.y;
        }
    };
    auto store = [&](int b) {
        float *bx = sx + b * Q38M_SC * Q38M_BR, *bw = sw + b * Q38M_SC * Q38M_BO;
        *reinterpret_cast<float4 *>(bx + xc * Q38M_BR + 4 * ((2 * xq) ^ xl)) = make_float4(xv[0], xv[1], xv[2], xv[3]);
        *reinterpret_cast<float4 *>(bx + xc * Q38M_BR + 4 * ((2 * xq + 1) ^ xl)) = make_float4(xv[4], xv[5], xv[6], xv[7]);
#pragma unroll
        for (int l = 0; l < 8; l++) {
            float v[4];
#pragma unroll
            for (int k = 0; k < 4; k++) v[k] = q38m_e4m3((wv[2 * k + (l >> 2)] >> (8 * (l & 3))) & 0xffu);
            *reinterpret_cast<float4 *>(bw + (8 * m + l) * Q38M_BO + 4 * (u ^ l)) = make_float4(v[0], v[1], v[2], v[3]);
        }
    };
    // hsum256 by xor 4 (rows), 2 then 1 (outputs): lane l ends with rows r8 + 4 k4 + i, outputs o16 + 8 k2 + 4 k1 + j
    const int k4 = (L >> 2) & 1, k2 = (L >> 1) & 1, k1 = L & 1;
    const int oo = o16 + 8 * k2 + 4 * k1;
    const float *scl = UP ? (oo < 64 ? ex->gs : ex->us) : ex->ds;
    const int srow = (UP ? o0 / 128 : (int)blockIdx.y) * nblk;   // the outputs' row of the matrix's 128 x 128 scales
    float a[8][16];
    double tot[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) tot[i][j] = 0.0;
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 16; j++) a[i][j] = 0.0f;
    load(0); store(0); load(nst > 1 ? 1 : 0);
    __syncthreads();
#pragma unroll 1
    for (int st = 0; st < nst; st++) {
        const int b = st & 1;
        const float *bx = sx + b * Q38M_SC * Q38M_BR + L * Q38M_BR, *bw = sw + b * Q38M_SC * Q38M_BO + L * Q38M_BO;
#pragma unroll
        for (int mm = 0; mm < 8; mm++) {   // column 8 mm + L of the stage
            const float4 x0 = *reinterpret_cast<const float4 *>(bx + mm * 8 * Q38M_BR + xa0);
            const float4 x1 = *reinterpret_cast<const float4 *>(bx + mm * 8 * Q38M_BR + xa1);
            float wr[16];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 w4 = *reinterpret_cast<const float4 *>(bw + mm * 8 * Q38M_BO + wa[j]);
                wr[4 * j] = w4.x; wr[4 * j + 1] = w4.y; wr[4 * j + 2] = w4.z; wr[4 * j + 3] = w4.w;
            }
            const float xr[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 16; j++) a[i][j] = __fmaf_rn(xr[i], wr[j], a[i][j]);
            if (mm == 3) {   // the next stage into the other buffer, the one after into registers
                store(b ^ 1);
                load(st + 2 < nst ? st + 2 : nst - 1);
            }
        }
        if (st & 1) {   // a 128-column block done: its eight lanes in hsum256's pairs, then the scale
            float h1[4][16];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 16; j++)
                    h1[i][j] = __fadd_rn(k4 ? a[i + 4][j] : a[i][j], __shfl_xor_sync(0xffffffffu, k4 ? a[i][j] : a[i + 4][j], 4));
            float h2[4][8];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++)
                    h2[i][j] = __fadd_rn(k2 ? h1[i][j + 8] : h1[i][j], __shfl_xor_sync(0xffffffffu, k2 ? h1[i][j] : h1[i][j + 8], 2));
            const double s = (double)__ldg(scl + srow + (st >> 1));
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float h3 = __fadd_rn(k1 ? h2[i][j + 4] : h2[i][j], __shfl_xor_sync(0xffffffffu, k1 ? h2[i][j] : h2[i][j + 4], 1));
                    tot[i][j] = __fma_rn(s, (double)h3, tot[i][j]);
                }
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 16; j++) a[i][j] = 0.0f;
        }
        __syncthreads();
    }
    if (!UP) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int ar = a0 + r8 + 4 * k4 + i;
            if (ar < n)
                *reinterpret_cast<float4 *>(y + (size_t)(p0 + ar) * D + o0 + oo) =
                    make_float4(__double2float_rn(tot[i][0]), __double2float_rn(tot[i][1]),
                                __double2float_rn(tot[i][2]), __double2float_rn(tot[i][3]));
        }
        return;
    }
    // gate and up of the tile's rows side by side in shared memory (the loop's last barrier passed), then
    // h = (g sigmoid(g)) u: thread t row t / 8, the tile's columns 8 (t % 8) .. +7 of gate's 64
    float *tile = sw;
#pragma unroll
    for (int i = 0; i < 4; i++)
        *reinterpret_cast<float4 *>(tile + (r8 + 4 * k4 + i) * Q38M_BO + oo) =
            make_float4(__double2float_rn(tot[i][0]), __double2float_rn(tot[i][1]),
                        __double2float_rn(tot[i][2]), __double2float_rn(tot[i][3]));
    __syncthreads();
    const int hr = t >> 3, hc = 8 * (t & 7);
    if (a0 + hr < n) {
        float h[8];
#pragma unroll
        for (int c = 0; c < 8; c++) h[c] = __fmul_rn(q38m_silu(tile[hr * Q38M_BO + hc + c]), tile[hr * Q38M_BO + 64 + hc + c]);
        float *yp = y + (size_t)(p0 + a0 + hr - hp0) * I + o0 + hc;
        *reinterpret_cast<float4 *>(yp) = make_float4(h[0], h[1], h[2], h[3]);
        *reinterpret_cast<float4 *>(yp + 4) = make_float4(h[4], h[5], h[6], h[7]);
    }
}

// each row's experts in rank order: sum = fma(gate, out, sum) from +0 (pos -1: none)
__global__ void q38m_sum(const float *__restrict__ out, const int *__restrict__ pos, const float *__restrict__ gates,
                         int K, int D, float *__restrict__ y) {
    const int r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.0f;
    for (int z = 0; z < K; z++) {
        const int p = __ldg(pos + (size_t)r * K + z);
        if (p >= 0) acc = __fmaf_rn(__ldg(gates + (size_t)r * K + z), __ldg(out + (size_t)p * D + d), acc);
    }
    y[(size_t)r * D + d] = acc;
}

static int q38m_ok(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] qwen38 prompt experts: %s: %s\n", what, cudaGetErrorString(e));
    return 0;
}
// the exp_f table and the kernels' shared memory, once (the chain runs on one device)
static int q38m_init(void) {
    static int done = 0;
    if (done) return done > 0;
    done = q38m_ok(cudaMemcpyToSymbol(q38m_exp_in, k3x_EXP_in, sizeof k3x_EXP_in), "exp_f table") &&
           q38m_ok(cudaMemcpyToSymbol(q38m_exp_out, k3x_EXP_out, sizeof k3x_EXP_out), "exp_f table") &&
           q38m_ok(cudaFuncSetAttribute(q38m_mm<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, Q38M_SMEM), "shared memory") &&
           q38m_ok(cudaFuncSetAttribute(q38m_mm<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, Q38M_SMEM), "shared memory") ? 1 : -1;
    return done > 0;
}

extern "C" int coli_cuda_q38_moe(const float *x, const int *rows, const CcQ38Exp *tab, const CcQ38Moe *p,
                                 float *h, float *out, void *stream) {
    if (!x || !rows || !tab || !p || !h || !out || p->ntab < 1 || p->ntiles < 1 || p->D % 128 || p->I % 128) return 0;
    if (!q38m_init()) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    q38m_mm<1><<<dim3((unsigned)p->ntiles, (unsigned)(p->I / 64)), 256, Q38M_SMEM, st>>>(x, rows, tab, p->ntab, p->D, p->I, p->hp0, h);
    if (!q38m_ok(cudaGetLastError(), "gate and up")) return 0;
    q38m_mm<0><<<dim3((unsigned)p->ntiles, (unsigned)(p->D / Q38M_BO)), 256, Q38M_SMEM, st>>>(h, nullptr, tab, p->ntab, p->D, p->I, p->hp0, out);
    return q38m_ok(cudaGetLastError(), "down");
}
extern "C" int coli_cuda_q38_sum(const float *out, const int *pos, const float *gates, int S, int K, int D, float *y,
                                 void *stream) {
    if (!out || !pos || !gates || !y || S < 1 || K < 1 || D < 1) return 0;
    q38m_sum<<<dim3((unsigned)((D + 255) / 256), (unsigned)S), 256, 0, (cudaStream_t)stream>>>(out, pos, gates, K, D, y);
    return q38m_ok(cudaGetLastError(), "the rows' sums");
}
#endif   /* CUDA only */
