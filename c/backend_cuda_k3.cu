// backend_cuda_k3.cu -- coli_cuda_k3.dll: Kimi K3's prompt-step routed experts on the device,
// in kimi_k3.c's CPU arithmetic to the bit (backend_cuda_k3.h says what that means).
//
// An expert's three MXFP4 matrices (gate w1, up w3: [MI][LT]; down w2: [LT][MI]; e2m1 codes two
// to a byte, one e8m0 scale a group of 32) against the step's rows quantized to int8 per group
// of 32 (mxfp4_qx). matmul_mxfp4_i8_pre computes an output as 8 lanes: lane l of a group holds
// the integer sum of the products of its elements 4l..4l+3 (the e2m1 values doubled: 0..12 and
// their negatives), times the group's scale (e8m0 times 0.5, times the row's scale, two float
// multiplies), fused into the lane's sum, group after group; the output is hsum256 of the 8
// lanes: ((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7)). Here a lane is a thread: dp4a gives its
// integer sum, the same two multiplies and fused add follow, and three xor shuffles add the
// lanes in hsum256's pairs. SiTU, the quantization of its outputs for down (mxfp4_qx) and
// down follow in the same way, so every output is the CPU's bytes.
//
// The host drives it from one thread: k3cu_moe_begin sends the step's rows up, k3cu_moe_expert
// copies an expert's bytes into one of K3_SLOTS device slots (on the copy stream; the slot's
// last kernels done first) and queues its kernels on the slot's stream, which write the
// outputs straight into the caller's pinned buffer; k3cu_moe_poll names the experts done.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include "backend_cuda_k3.h"
#include "k3_exact.h"
#include "k3_exact_tables.h"
#include "qmb_kernel.cuh"   // quant_matmul_blocked: the KDA layer's projections on coli_cuda.dll's matrices

#define K3_TP 8       // rows a block: one an output's lane
#define K3_TO 32      // outputs a block: a group of 32, so the block quantizes what it computes
#define K3_SLOTS 8
#define K3_RCAP 1024  // rows an expert's piece (its quantized SiTU outputs' room)

static int ready, failed, dev_id = -1;
static int checked(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 1;
    fprintf(stderr, "[K3-CUDA] experts: %s: %s\n", what, cudaGetErrorString(e));
    failed = 1;
    return 0;
}

// ---- exp_f and tanhf, exact (k3_exact.h; the undecided floats in its tables) -----------------
__device__ uint32_t d_exp_in[K3X_EXP_N], d_exp_out[K3X_EXP_N], d_tanh_in[K3X_TANH_N], d_tanh_out[K3X_TANH_N];
__device__ __forceinline__ float k3d_expf(float x) {
    float f;
    if (k3x_expf_fast(x, &f)) return f;
    return k3x_table(d_exp_in, d_exp_out, K3X_EXP_N, x);
}
__device__ __forceinline__ float k3d_tanhf(float x) {
    float f;
    if (k3x_tanhf_fast(x, &f)) return f;
    return k3x_table(d_tanh_in, d_tanh_out, K3X_TANH_N, x);
}
// situf_: b1*tanhf(g/b1)*sigmoidf_(g) * b2*tanhf(u/b2), left to right; sigmoidf_ 1/(1+exp_f(-g))
__device__ __forceinline__ float k3d_situ(float g, float u, float b1, float b2) {
    float t1 = k3d_tanhf(__fdiv_rn(g, b1));
    float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, k3d_expf(-g)));
    float t2 = k3d_tanhf(__fdiv_rn(u, b2));
    return __fmul_rn(__fmul_rn(__fmul_rn(__fmul_rn(b1, t1), s), b2), t2);
}
__global__ void k3_exact_kernel(int kind, uint32_t first, uint32_t count, uint32_t *out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = __uint_as_float(first + i);
    out[i] = __float_as_uint(kind ? k3d_tanhf(x) : k3d_expf(x));
}

// ---- the expert kernels ------------------------------------------------------------------------
// a byte of two e2m1 codes -> the two doubled values (quant.h's lut2), low code first
__device__ __forceinline__ void k3d_lut_init(uint16_t *lut) {
    const int8_t v[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int b = threadIdx.x; b < 256; b += blockDim.x)
        lut[b] = (uint16_t)((uint8_t)v[b & 15] | ((uint16_t)(uint8_t)v[b >> 4] << 8));
}
// hsum256's pairs over the 8 lanes of an output (threads 8k .. 8k+7); every lane gets the total
__device__ __forceinline__ float k3d_hsum8(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}
// K3_TP rows of x (int8, n bytes each) and their scales (ng each) into shared memory: row p is
// src row idx(p0 + p), the last of the r rows standing in past r
__device__ __forceinline__ void k3d_load_rows(int8_t *sx, float *sxs, const int8_t *x, const float *xs,
                                              const int *rows, int p0, int r, int n, int ng) {
    const int w = n >> 2;
    for (int i = threadIdx.x; i < K3_TP * w; i += blockDim.x) {
        int p = i / w, c = i - p * w, pr = p0 + p < r ? p0 + p : r - 1;
        size_t src = rows ? (size_t)rows[pr] : (size_t)pr;
        ((int *)sx)[i] = ((const int *)(x + src * n))[c];
    }
    for (int i = threadIdx.x; i < K3_TP * ng; i += blockDim.x) {
        int p = i / ng, g = i - p * ng, pr = p0 + p < r ? p0 + p : r - 1;
        size_t src = rows ? (size_t)rows[pr] : (size_t)pr;
        sxs[i] = xs[src * ng + g];
    }
}
// gate and up of rows p0.. (blockIdx.y) for outputs o0 .. o0+31 (blockIdx.x), SiTU, then the 32
// SiTU outputs of each row quantized as mxfp4_qx does into gq [r][MI], gs [r][MI/32]
__global__ void __launch_bounds__(256) k3_gate_up(const uint8_t *__restrict__ w1p, const uint8_t *__restrict__ w1s,
                                                  const uint8_t *__restrict__ w3p, const uint8_t *__restrict__ w3s,
                                                  const int8_t *__restrict__ zq, const float *__restrict__ zs,
                                                  const int *__restrict__ rows, int r, int LT, int MI, float b1, float b2,
                                                  int8_t *__restrict__ gq, float *__restrict__ gs) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int ng = LT >> 5;
    uint16_t *lut = (uint16_t *)smem;
    float *sg = (float *)(smem + 512);                            // [K3_TP][K3_TO]
    float *sxs = sg + K3_TP * K3_TO;                              // [K3_TP][ng]
    int8_t *sx = (int8_t *)(sxs + K3_TP * ng);                    // [K3_TP][LT]
    const int l = threadIdx.x & 7, q = threadIdx.x >> 3;
    const int o = blockIdx.x * K3_TO + q, p0 = blockIdx.y * K3_TP;
    k3d_lut_init(lut);
    k3d_load_rows(sx, sxs, zq, zs, rows, p0, r, LT, ng);
    __syncthreads();
    float a1[K3_TP], a3[K3_TP];
#pragma unroll
    for (int p = 0; p < K3_TP; p++) { a1[p] = 0.0f; a3[p] = 0.0f; }
    const uint8_t *r1 = w1p + (size_t)o * (LT >> 1) + 2 * l, *r3 = w3p + (size_t)o * (LT >> 1) + 2 * l;
    const uint8_t *e1 = w1s + (size_t)o * ng, *e3 = w3s + (size_t)o * ng;
    for (int g = 0; g < ng; g++) {
        const uint32_t c1 = *(const uint16_t *)(r1 + g * 16), c3 = *(const uint16_t *)(r3 + g * 16);
        const int w1 = (int)((uint32_t)lut[c1 & 255] | (uint32_t)lut[c1 >> 8] << 16);
        const int w3 = (int)((uint32_t)lut[c3 & 255] | (uint32_t)lut[c3 >> 8] << 16);
        const float h1 = __fmul_rn(__uint_as_float((uint32_t)e1[g] << 23), 0.5f);
        const float h3 = __fmul_rn(__uint_as_float((uint32_t)e3[g] << 23), 0.5f);
#pragma unroll
        for (int p = 0; p < K3_TP; p++) {
            const int x4 = *(const int *)(sx + p * LT + g * 32 + 4 * l);
            const float xs = sxs[p * ng + g];
            a1[p] = __fmaf_rn(__int2float_rn(__dp4a(w1, x4, 0)), __fmul_rn(h1, xs), a1[p]);
            a3[p] = __fmaf_rn(__int2float_rn(__dp4a(w3, x4, 0)), __fmul_rn(h3, xs), a3[p]);
        }
    }
    float gv = 0.0f, uv = 0.0f;
#pragma unroll
    for (int p = 0; p < K3_TP; p++) {
        const float s1 = k3d_hsum8(a1[p]), s3 = k3d_hsum8(a3[p]);
        if (l == p) { gv = s1; uv = s3; }
    }
    sg[l * K3_TO + q] = k3d_situ(gv, uv, b1, b2);   // lane l: row p0 + l
    __syncthreads();
    // warp w quantizes row p0 + w: lane j its output o0 + j (mxfp4_qx's group)
    const int w = threadIdx.x >> 5, j = threadIdx.x & 31;
    if (p0 + w < r) {
        const float x = sg[w * K3_TO + j];
        float am = fabsf(x);
        am = am > 0.0f ? am : 0.0f;                  // a NaN counts as 0, as `if(a>am)am=a` skips it
#pragma unroll
        for (int s = 16; s; s >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, s));
        float sc = __fdiv_rn(am, 127.0f);
        if (sc < 1e-20f) sc = 1e-20f;
        const float inv = __fdiv_rn(1.0f, sc), pr = __fmul_rn(x, inv);
        int v = pr != pr ? -127 : __float2int_rn(pr);   // lrintf's NaN: INT_MIN, clamped
        v = v > 127 ? 127 : v < -127 ? -127 : v;
        gq[(size_t)(p0 + w) * MI + blockIdx.x * K3_TO + j] = (int8_t)v;
        if (j == 0) gs[(size_t)(p0 + w) * (MI >> 5) + blockIdx.x] = sc;
    }
}
// down of rows p0.. for outputs o0 .. o0+31: gq/gs [r][MI] -> y rows [r][LT]
__global__ void __launch_bounds__(256) k3_down(const uint8_t *__restrict__ w2p, const uint8_t *__restrict__ w2s,
                                               const int8_t *__restrict__ gq, const float *__restrict__ gs,
                                               int r, int MI, int LT, float *__restrict__ y) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int ng = MI >> 5;
    uint16_t *lut = (uint16_t *)smem;
    float *sxs = (float *)(smem + 512);                           // [K3_TP][ng]
    int8_t *sx = (int8_t *)(sxs + K3_TP * ng);                    // [K3_TP][MI]
    const int l = threadIdx.x & 7, q = threadIdx.x >> 3;
    const int o = blockIdx.x * K3_TO + q, p0 = blockIdx.y * K3_TP;
    k3d_lut_init(lut);
    k3d_load_rows(sx, sxs, gq, gs, nullptr, p0, r, MI, ng);
    __syncthreads();
    float a[K3_TP];
#pragma unroll
    for (int p = 0; p < K3_TP; p++) a[p] = 0.0f;
    const uint8_t *r2 = w2p + (size_t)o * (MI >> 1) + 2 * l, *e2 = w2s + (size_t)o * ng;
    for (int g = 0; g < ng; g++) {
        const uint32_t c = *(const uint16_t *)(r2 + g * 16);
        const int w = (int)((uint32_t)lut[c & 255] | (uint32_t)lut[c >> 8] << 16);
        const float h = __fmul_rn(__uint_as_float((uint32_t)e2[g] << 23), 0.5f);
#pragma unroll
        for (int p = 0; p < K3_TP; p++) {
            const int x4 = *(const int *)(sx + p * MI + g * 32 + 4 * l);
            a[p] = __fmaf_rn(__int2float_rn(__dp4a(w, x4, 0)), __fmul_rn(h, sxs[p * ng + g]), a[p]);
        }
    }
    float v = 0.0f;
#pragma unroll
    for (int p = 0; p < K3_TP; p++) {
        const float s = k3d_hsum8(a[p]);
        if (l == p) v = s;
    }
    // y is the caller's pinned memory, written over PCIe: a warp writes one row's 32 outputs, 128
    // bytes (8 rows of 4 floats a warp, 16-byte pieces, slowed the expert copies the other way
    // from 26 to 20 GB/s)
    __shared__ float sy[K3_TP][K3_TO];
    sy[l][q] = v;
    __syncthreads();
    const int wr = threadIdx.x >> 5, j = threadIdx.x & 31;
    if (p0 + wr < r) y[(size_t)(p0 + wr) * LT + blockIdx.x * K3_TO + j] = sy[wr][j];
}

// ---- the dense matrices the device does not hold, in the CPU's arithmetic ------------------------
// matmul_i4_grouped (fmt 4): a row's output is, group by group, 8 lanes summing x[i+l] times the
// weight (its nibble less 8) and then x[i+8+l] times its weight, for each 16 columns, fused; the
// group's hsum256 times its scale fused into the total (fmaf). matmul_q (fmt 1): 8 lanes summing
// x[i+l] times the int8 weight, fused, over the whole row; hsum256 times the row's scale. Lane l
// is thread l of the output's 8, as for the experts; x comes in chunks of K3_KC columns.
#define K3_KC 512
__global__ void __launch_bounds__(256) k3_dense(int fmt, const uint8_t *__restrict__ w, const float *__restrict__ scale,
                                                const float *__restrict__ x, int S, int I, int O, int gs,
                                                float *__restrict__ y) {
    __shared__ float sx[K3_TP * K3_KC];
    const int l = threadIdx.x & 7, q = threadIdx.x >> 3;
    const int o = blockIdx.x * K3_TO + q, p0 = blockIdx.y * K3_TP, oc = o < O ? o : O - 1;
    float tot[K3_TP], acc[K3_TP];
#pragma unroll
    for (int p = 0; p < K3_TP; p++) { tot[p] = 0.0f; acc[p] = 0.0f; }
    const uint8_t *w4 = w + (size_t)oc * (I >> 1);
    const int8_t *w8 = (const int8_t *)w + (size_t)oc * I;
    const float *sc = scale + (size_t)oc * (fmt == 4 ? I / gs : 1);
    for (int k0 = 0; k0 < I; k0 += K3_KC) {
        const int kn = I - k0 < K3_KC ? I - k0 : K3_KC;
        __syncthreads();
        for (int t = threadIdx.x; t < K3_TP * kn; t += blockDim.x) {
            const int p = t / kn, c = t - p * kn, pr = p0 + p < S ? p0 + p : S - 1;
            sx[p * K3_KC + c] = x[(size_t)pr * I + k0 + c];
        }
        __syncthreads();
        if (fmt == 4) {
            for (int base = 0; base < kn; base += gs) {
#pragma unroll
                for (int p = 0; p < K3_TP; p++) acc[p] = 0.0f;
                for (int i = base; i < base + gs; i += 16) {
                    const int e0 = k0 + i + l, e1 = e0 + 8;
                    const uint8_t b0 = w4[e0 >> 1], b1 = w4[e1 >> 1];
                    const float v0 = (float)((int)((e0 & 1) ? b0 >> 4 : b0 & 15) - 8);
                    const float v1 = (float)((int)((e1 & 1) ? b1 >> 4 : b1 & 15) - 8);
#pragma unroll
                    for (int p = 0; p < K3_TP; p++) {
                        acc[p] = __fmaf_rn(sx[p * K3_KC + i + l], v0, acc[p]);
                        acc[p] = __fmaf_rn(sx[p * K3_KC + i + 8 + l], v1, acc[p]);
                    }
                }
                const float g = sc[(k0 + base) / gs];
#pragma unroll
                for (int p = 0; p < K3_TP; p++) tot[p] = __fmaf_rn(k3d_hsum8(acc[p]), g, tot[p]);
            }
        } else {
            for (int i = 0; i < kn; i += 8) {
                const float v = (float)w8[k0 + i + l];
#pragma unroll
                for (int p = 0; p < K3_TP; p++) acc[p] = __fmaf_rn(sx[p * K3_KC + i + l], v, acc[p]);
            }
        }
    }
    float v = 0.0f;
#pragma unroll
    for (int p = 0; p < K3_TP; p++) {
        const float t = fmt == 4 ? tot[p] : __fmul_rn(k3d_hsum8(acc[p]), sc[0]);
        if (l == p) v = t;
    }
    if (o < O && p0 + l < S) y[(size_t)(p0 + l) * O + o] = v;
}

// matmul_i4_grouped's arithmetic again for groups of 64, a thread's 8 lanes in its own registers
// (k3_dense gives them to 8 threads and adds them with shuffles every group, its inputs one float
// a load): thread (tr, to) takes rows 4 tr .. +3 of the block's 32 by outputs 6 to .. +5 of its
// 192. In a group lane l's chain is columns l, l+8, .., l+56, fused in that order from +0; the
// lanes go 0, 4, 2, 6, 1, 5, 3, 7, so each finished one joins hsum256's tree at once,
// ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)), three kept a pair; then the group's fmaf into the total.
// Lane l's weight m is nibble l of the group's word m: (w & 15 << 4p) | 2^23 read as a float, one
// fused add takes the place p and the 8 out, exactly. The block's rows of x go to shared memory by
// column, 128 columns a stage, double-buffered. About 4x k3_dense's rate from 64 rows (13-15 T
// multiply-adds a second at Kimi K3's shapes on an RTX 4090), the same bytes; K3_GPU_DENSE_B=0: k3_dense.
#define KB_RS 4
#define KB_RO 6
#define KB_BR 32                    // rows a block
#define KB_BO 192                   // outputs a block
#define KB_SC 128                   // columns a stage
#define KB_PAD 36                   // a column's 32 rows and 4 spare: 16-byte stores without conflicts
#define KB_MINROWS 64               // fewer rows: k3_dense (too few blocks)
template <int P>
__device__ __forceinline__ float kb_nib(uint32_t w) {   // the offset nibble at place P (0..4) of w, less 8
    const float mul = P == 0 ? 1.0f : P == 1 ? 0.0625f : P == 2 ? 0.00390625f : P == 3 ? 0.000244140625f : 1.52587890625e-05f;
    const float add = P == 0 ? -8388616.0f : P == 1 ? -524296.0f : P == 2 ? -32776.0f : P == 3 ? -2056.0f : -136.0f;
    return __fmaf_rn(__uint_as_float((w & (15u << (4 * P))) | 0x4B000000u), mul, add);
}
template <int L>
__device__ __forceinline__ float kb_lane_w(uint32_t w) {   // lane L's nibble of a word
    if (L <= 4) return kb_nib<L>(w);
    else return kb_nib<L - 3>(w >> 12);
}
template <int L>
__device__ __forceinline__ void kb_lane(const float (*sxc)[KB_PAD], int r0, const uint32_t (&W)[KB_RO][8],
                                        float (&a)[KB_RS][KB_RO], int cbase) {
#pragma unroll
    for (int r = 0; r < KB_RS; r++)
#pragma unroll
        for (int j = 0; j < KB_RO; j++) a[r][j] = 0.0f;
#pragma unroll
    for (int m = 0; m < 8; m++) {
        const float4 xv = *reinterpret_cast<const float4 *>(&sxc[cbase + L + 8 * m][r0]);
        const float xr[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
        for (int j = 0; j < KB_RO; j++) {
            const float wv = kb_lane_w<L>(W[j][m]);
#pragma unroll
            for (int r = 0; r < KB_RS; r++) a[r][j] = __fmaf_rn(xr[r], wv, a[r][j]);
        }
    }
}
__global__ void __launch_bounds__(256, 1) k3_dense_b(const uint8_t *__restrict__ w, const float *__restrict__ scale,
                                                     const float *__restrict__ x, int S, int I, int O,
                                                     float *__restrict__ y) {
    __shared__ __align__(16) float sx[2][KB_SC][KB_PAD];
    const int t = threadIdx.x, tr = t & 7, to = t >> 3;
    const int s0 = blockIdx.y * KB_BR, ob = blockIdx.x * KB_BO + to * KB_RO, r0 = tr * KB_RS;
    const int ng = I / 64;
    float tot[KB_RS][KB_RO], s0v[KB_RS][KB_RO], s1v[KB_RS][KB_RO], s2v[KB_RS][KB_RO], a[KB_RS][KB_RO];
#pragma unroll
    for (int r = 0; r < KB_RS; r++)
#pragma unroll
        for (int j = 0; j < KB_RO; j++) tot[r][j] = 0.0f;
    // staging: thread t loads column t & 127 of rows 16 (t >> 7) .. +15 (past S: the last row)
    const int sc_c = t & 127, sc_r = (t >> 7) * 16;
    float xs[16];
    auto xload = [&](int k0) {
#pragma unroll
        for (int q = 0; q < 16; q++) {
            const int row = s0 + sc_r + q, rr = row < S ? row : S - 1;
            xs[q] = __ldg(x + (size_t)rr * I + k0 + sc_c);
        }
    };
    auto xstore = [&](int b) {
#pragma unroll
        for (int q = 0; q < 4; q++)
            *reinterpret_cast<float4 *>(&sx[b][sc_c][sc_r + 4 * q]) = make_float4(xs[4 * q], xs[4 * q + 1], xs[4 * q + 2], xs[4 * q + 3]);
    };
    const int nst = I / KB_SC;
    xload(0); xstore(0);
    if (nst > 1) xload(KB_SC);
    __syncthreads();
    const uint8_t *wrow[KB_RO];
#pragma unroll
    for (int j = 0; j < KB_RO; j++) { const int o = ob + j < O ? ob + j : O - 1; wrow[j] = w + (size_t)o * (I >> 1); }
    for (int st = 0; st < nst; st++) {
        const int b = st & 1;
#pragma unroll 1
        for (int gg = 0; gg < KB_SC / 64; gg++) {
            const int g = st * (KB_SC / 64) + gg;
            uint32_t W[KB_RO][8];
#pragma unroll
            for (int j = 0; j < KB_RO; j++) {
                const uint4 u0 = __ldg(reinterpret_cast<const uint4 *>(wrow[j] + (size_t)g * 32));
                const uint4 u1 = __ldg(reinterpret_cast<const uint4 *>(wrow[j] + (size_t)g * 32 + 16));
                W[j][0] = u0.x; W[j][1] = u0.y; W[j][2] = u0.z; W[j][3] = u0.w;
                W[j][4] = u1.x; W[j][5] = u1.y; W[j][6] = u1.z; W[j][7] = u1.w;
            }
            const float (*sxc)[KB_PAD] = sx[b];
            const int cb = gg * 64;
            kb_lane<0>(sxc, r0, W, a, cb);
#pragma unroll
            for (int r = 0; r < KB_RS; r++)
#pragma unroll
                for (int j = 0; j < KB_RO; j++) s0v[r][j] = a[r][j];
            kb_lane<4>(sxc, r0, W, a, cb);
#pragma unroll
            for (int r = 0; r < KB_RS; r++)
#pragma unroll
                for (int j = 0; j < KB_RO; j++) s0v[r][j] = __fadd_rn(s0v[r][j], a[r][j]);
            kb_lane<2>(sxc, r0, W, a, cb);
#pragma unroll
            for (int r = 0; r < KB_RS; r++)
#pragma unroll
                for (int j = 0; j < KB_RO; j++) s1v[r][j] = a[r][j];
            kb_lane<6>(sxc, r0, W, a, cb);
#pragma unroll
            for (int r = 0; r < KB_RS; r++)
#pragma unroll
                for (int j = 0; j < KB_RO; j++) s0v[r][j] = __fadd_rn(s0v[r][j], __fadd_rn(s1v[r][j], a[r][j]));
            kb_lane<1>(sxc, r0, W, a, cb);
#pragma unroll
            for (int r = 0; r < KB_RS; r++)
#pragma unroll
                for (int j = 0; j < KB_RO; j++) s1v[r][j] = a[r][j];
            kb_lane<5>(sxc, r0, W, a, cb);
#pragma unroll
            for (int r = 0; r < KB_RS; r++)
#pragma unroll
                for (int j = 0; j < KB_RO; j++) s1v[r][j] = __fadd_rn(s1v[r][j], a[r][j]);
            kb_lane<3>(sxc, r0, W, a, cb);
#pragma unroll
            for (int r = 0; r < KB_RS; r++)
#pragma unroll
                for (int j = 0; j < KB_RO; j++) s2v[r][j] = a[r][j];
            kb_lane<7>(sxc, r0, W, a, cb);
#pragma unroll
            for (int j = 0; j < KB_RO; j++) {
                const int o = ob + j < O ? ob + j : O - 1;
                const float gsc = __ldg(scale + (size_t)o * ng + g);
#pragma unroll
                for (int r = 0; r < KB_RS; r++) {
                    const float f = __fadd_rn(s1v[r][j], __fadd_rn(s2v[r][j], a[r][j]));
                    tot[r][j] = __fmaf_rn(__fadd_rn(s0v[r][j], f), gsc, tot[r][j]);
                }
            }
        }
        if (st + 1 < nst) {
            __syncthreads();
            xstore(b ^ 1);
            __syncthreads();
            if (st + 2 < nst) xload((st + 2) * KB_SC);
        }
    }
#pragma unroll
    for (int r = 0; r < KB_RS; r++) {
        const int row = s0 + r0 + r;
        if (row >= S) continue;
#pragma unroll
        for (int j = 0; j < KB_RO; j++) if (ob + j < O) y[(size_t)row * O + ob + j] = tot[r][j];
    }
}
static int kb_mode(void) {
    static int m = -1;
    if (m < 0) { const char *e = getenv("K3_GPU_DENSE_B"); m = !e || !*e || atoi(e) != 0; }
    return m;
}
// y [r][O] = x [r][I] . W^T in the CPU's arithmetic on stream st: k3_dense_b where it was built for it
static void k3_dense_launch(int fmt, const uint8_t *w, const float *s, const float *x, int r, int I, int O, int gs,
                            float *y, cudaStream_t st) {
    if (fmt == 4 && gs == 64 && I % KB_SC == 0 && r >= KB_MINROWS && kb_mode()) {
        const dim3 g((unsigned)((O + KB_BO - 1) / KB_BO), (unsigned)((r + KB_BR - 1) / KB_BR));
        k3_dense_b<<<g, 256, 0, st>>>(w, s, x, r, I, O, y);
    } else {
        const dim3 g((unsigned)((O + K3_TO - 1) / K3_TO), (unsigned)((r + K3_TP - 1) / K3_TP));
        k3_dense<<<g, 256, 0, st>>>(fmt, w, s, x, r, I, O, gs, y);
    }
}

// ---- a prompt step's gated MLA attention, mla_forward's loop -----------------------------------
// For row tt (position pos0 + tt, nt = pos0 + tt + 1 positions) and head h, as the CPU does it:
//   qabs[i] = sum over d < qk_nope, in order, fused: (kv_b scale of row rbase+d times q[d]) times
//             kv_b[rbase+d][i]                                          (w_addrow, fmt 1)
//   s[t]    = (qabs . Lc[t] then + q_rope . Rc[t]: products rounded, added in order from zero)
//             times the scale                                           (k3_mla_scores_g)
//   p[t]    = exp_f(s[t] - max) / (their sum, in t's order)             (softmax_)
//   clat[i] = sum over t < nt, in order, fused: p[t] times Lc[t][i]
//   ctx[d]  = (kv_b[rbase+qk_nope+d] . clat: products rounded, added in order) times the row's
//             scale, times sigmoidf_(gate[d])                           (k3_rowdot8_i8)
// The scores and clat are matrix products whose every output sums its k in order, fused or not:
// one tiled kernel (k3_gemm); a row's positions past its nt have p = 0, which adds nothing.
#define K3G_T 64
#define K3G_K 16
template <int FUSED, int BT>
__global__ void __launch_bounds__(256) k3_gemm(int M, int N, int K, const float *__restrict__ A, int lda,
                                               const float *__restrict__ B, int ldb, float *__restrict__ Cm, int ldc,
                                               float after) {
    __shared__ float As[K3G_K][K3G_T + 4], Bs[K3G_K][K3G_T + 4];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int m0 = blockIdx.y * K3G_T, n0 = blockIdx.x * K3G_T;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += K3G_K) {
        for (int e = threadIdx.x; e < K3G_T * K3G_K; e += 256) {
            const int r = e / K3G_K, kk = e - r * K3G_K, gm = m0 + r, gk = k0 + kk;
            As[kk][r] = gm < M && gk < K ? A[(size_t)gm * lda + gk] : 0.0f;
        }
        for (int e = threadIdx.x; e < K3G_T * K3G_K; e += 256) {
            int c, kk;
            if (BT) { c = e / K3G_K; kk = e - c * K3G_K; } else { kk = e / K3G_T; c = e - kk * K3G_T; }
            const int gn = n0 + c, gk = k0 + kk;
            Bs[kk][c] = gn < N && gk < K ? (BT ? B[(size_t)gn * ldb + gk] : B[(size_t)gk * ldb + gn]) : 0.0f;
        }
        __syncthreads();
        const int kn = K - k0 < K3G_K ? K - k0 : K3G_K;
        for (int kk = 0; kk < kn; kk++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = As[kk][ty * 4 + i]; b[i] = Bs[kk][tx * 4 + i]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = FUSED ? __fmaf_rn(a[i], b[j], acc[i][j]) : __fadd_rn(acc[i][j], __fmul_rn(a[i], b[j]));
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int gm = m0 + ty * 4 + i, gn = n0 + tx * 4 + j;
            if (gm < M && gn < N) Cm[(size_t)gm * ldc + gn] = after != 0.0f ? __fmul_rn(acc[i][j], after) : acc[i][j];
        }
}
// q_abs and the rope part of every (row, head) of the chunk: Q [rows*H][kvl+qr]
__global__ void k3_mla_q(const float *__restrict__ qv, int H, int qh, int qn, int qr, int vh, int kvl,
                         const int8_t *__restrict__ kvb, const float *__restrict__ kvbs, float *__restrict__ Q) {
    const int row = blockIdx.x, tt = row / H, h = row - tt * H;
    const float *qp = qv + ((size_t)tt * H + h) * qh;
    const int rbase = h * (qn + vh);
    float *q = Q + (size_t)row * (kvl + qr);
    for (int i = threadIdx.x; i < kvl; i += blockDim.x) {
        float acc = 0.0f;
        for (int d = 0; d < qn; d++)
            acc = __fmaf_rn(__fmul_rn(kvbs[rbase + d], qp[d]), (float)kvb[(size_t)(rbase + d) * kvl + i], acc);
        q[i] = acc;
    }
    for (int j = threadIdx.x; j < qr; j += blockDim.x) q[kvl + j] = qp[qn + j];
}
// softmax_ of every row's first nt scores (nt = first + row / H + 1), the rest of the row 0
__global__ void k3_mla_softmax(float *__restrict__ S, int N, int H, int first) {
    __shared__ float red[256];
    __shared__ float total;
    float *x = S + (size_t)blockIdx.x * N;
    const int nt = first + (int)blockIdx.x / H + 1;
    float m = x[0];
    for (int t = threadIdx.x; t < nt; t += blockDim.x) m = x[t] > m ? x[t] : m;
    red[threadIdx.x] = m;
    __syncthreads();
    for (int w = blockDim.x / 2; w; w >>= 1) {
        if ((int)threadIdx.x < w && red[threadIdx.x + w] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + w];
        __syncthreads();
    }
    m = red[0];
    for (int t = threadIdx.x; t < nt; t += blockDim.x) x[t] = k3d_expf(__fsub_rn(x[t], m));
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int t = 0; t < nt; t++) s = __fadd_rn(s, x[t]);
        total = s;
    }
    __syncthreads();
    const float s = total;
    for (int t = threadIdx.x; t < N; t += blockDim.x) x[t] = t < nt ? __fdiv_rn(x[t], s) : 0.0f;
}
// ctx = (kv_b's v rows . clat) * scale * sigmoidf_(gate): [rows][H][vh]
__global__ void k3_mla_out(const float *__restrict__ clat, const float *__restrict__ gv, int H, int qn, int vh, int kvl,
                           const int8_t *__restrict__ kvb, const float *__restrict__ kvbs, float *__restrict__ ctx) {
    const int row = blockIdx.x, tt = row / H, h = row - tt * H;
    const float *cl = clat + (size_t)row * kvl;
    for (int d = threadIdx.x; d < vh; d += blockDim.x) {
        const int r = h * (qn + vh) + qn + d;
        const int8_t *w = kvb + (size_t)r * kvl;
        float acc = 0.0f;
        for (int i = 0; i < kvl; i++) acc = __fadd_rn(acc, __fmul_rn((float)w[i], cl[i]));
        const float g = gv[((size_t)tt * H + h) * vh + d];
        const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, k3d_expf(-g)));
        ctx[((size_t)tt * H + h) * vh + d] = __fmul_rn(__fmul_rn(acc, kvbs[r]), sg);
    }
}

// ---- the router (k3_router_scores, router16) and the f32 matrices (matmul_f32.h's matmul) -------
// An output is 32 lanes, lane j fusing x[i+j] times the weight (bf16 widened, or f32) for i = 0, 32,
// ..., then halved: a[j] += a[j+16], += a[j+8], ... += a[j+1] (MATMUL_F32_LANES, matmul_f32.h); a
// lane past the last column adds nothing, as in dot_f32_lanes' last round. Lane j is a warp's lane
// j; the halving is shfl_down's. A block: 8 outputs (a warp each) over K3R_RT rows, their x in
// shared memory K3R_KC columns at a time.
#define K3R_RT 32
#define K3R_KC 256
__device__ __forceinline__ float k3d_wval(uint16_t w) { return __uint_as_float((uint32_t)w << 16); }
__device__ __forceinline__ float k3d_wval(float w) { return w; }
template <typename WT>
__global__ void __launch_bounds__(256) k3_lanes32(const WT *__restrict__ w, const float *__restrict__ x,
                                                  int C, int D, int E, float *__restrict__ sco) {
    __shared__ float xs[K3R_RT * K3R_KC];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int e = blockIdx.x * 8 + wp, ec = e < E ? e : E - 1, p0 = blockIdx.y * K3R_RT;
    const WT *wr = w + (size_t)ec * D;
    float acc[K3R_RT];
#pragma unroll
    for (int r = 0; r < K3R_RT; r++) acc[r] = 0.0f;
    for (int k0 = 0; k0 < D; k0 += K3R_KC) {
        const int kn = D - k0 < K3R_KC ? D - k0 : K3R_KC;
        __syncthreads();
        for (int t = threadIdx.x; t < K3R_RT * kn; t += blockDim.x) {
            const int r = t / kn, c = t - r * kn, pr = p0 + r < C ? p0 + r : C - 1;
            xs[r * K3R_KC + c] = x[(size_t)pr * D + k0 + c];
        }
        __syncthreads();
        for (int i = 0; i < kn; i += 32) {
            if (i + lane >= kn) break;
            const float wv = k3d_wval(wr[k0 + i + lane]);
#pragma unroll
            for (int r = 0; r < K3R_RT; r++) acc[r] = __fmaf_rn(xs[r * K3R_KC + i + lane], wv, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < K3R_RT; r++) {
        float v = acc[r];
        for (int h = 16; h; h >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, h));
        if (lane == 0 && e < E && p0 + r < C) sco[(size_t)(p0 + r) * E + e] = v;
    }
}

// ---- the host side ---------------------------------------------------------------------------
struct K3Slot {
    uint8_t *w; int *rows; int8_t *gq; float *gs;
    cudaStream_t s; cudaEvent_t done;
    int busy, tag;
};
static K3Slot S[K3_SLOTS];
static cudaStream_t s_copy;
static cudaEvent_t e_copied;
static size_t slot_bytes;                // the expert bytes a slot holds
static int8_t *d_zq; static float *d_zs; static size_t zq_cap, zs_cap;
static int m_on, m_LT, m_MI, m_n, m_next, m_npairs;
static float m_b1, m_b2, *m_Ydev;
static size_t m_ebytes, m_off[6];
static int m_late[K3_SLOTS], m_nlate;    // tags found done while expert() waited for their slot
static unsigned long long n_steps, n_experts, n_rows;
static double t_copy, t_wait;

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// every entry point runs on the device k3cu_init chose, from whichever thread calls it
static int on_device() {
    static thread_local int set = -1;
    if (set == dev_id) return 1;
    if (!checked(cudaSetDevice(dev_id), "device")) return 0;
    set = dev_id;
    return 1;
}

extern "C" int k3cu_init(int device) {
    if (ready) return device == dev_id;
    if (failed) return 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || device < 0 || device >= n) { cudaGetLastError(); return 0; }
    if (!checked(cudaSetDevice(device), "device") ||
        !checked(cudaMemcpyToSymbol(d_exp_in, k3x_EXP_in, sizeof k3x_EXP_in), "tables") ||
        !checked(cudaMemcpyToSymbol(d_exp_out, k3x_EXP_out, sizeof k3x_EXP_out), "tables") ||
        !checked(cudaMemcpyToSymbol(d_tanh_in, k3x_TANH_in, sizeof k3x_TANH_in), "tables") ||
        !checked(cudaMemcpyToSymbol(d_tanh_out, k3x_TANH_out, sizeof k3x_TANH_out), "tables") ||
        !checked(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking), "streams") ||
        !checked(cudaEventCreateWithFlags(&e_copied, cudaEventDisableTiming), "events")) return 0;
    for (int b = 0; b < K3_SLOTS; b++)
        if (!checked(cudaStreamCreateWithFlags(&S[b].s, cudaStreamNonBlocking), "streams") ||
            !checked(cudaEventCreateWithFlags(&S[b].done, cudaEventDisableTiming), "events")) return 0;
    dev_id = device; ready = 1;
    return on_device();
}
extern "C" size_t k3cu_free_bytes(void) {
    size_t fr = 0, tot = 0;
    if (!ready || cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); return 0; }
    return fr;
}
extern "C" void *k3cu_host_alloc(size_t bytes) {
    void *p = nullptr;
    if (!ready) return nullptr;
    const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (e != cudaSuccess) {
        static int told;
        if (!told++) fprintf(stderr, "[K3-CUDA] pinned memory: %.2f GB more refused (%s)\n", bytes / 1e9, cudaGetErrorString(e));
        cudaGetLastError();
        return nullptr;
    }
    return p;
}
extern "C" void k3cu_host_free(void *p) { if (p) cudaFreeHost(p); }

extern "C" int k3cu_exact_eval(int kind, uint32_t first, uint32_t count, uint32_t *out) {
    if (!ready || failed || !out || !count) return 0;
    uint32_t *d = nullptr;
    if (!checked(cudaMalloc(&d, (size_t)count * 4), "exact eval")) return 0;
    k3_exact_kernel<<<(count + 255) / 256, 256>>>(kind, first, count, d);
    int ok = checked(cudaGetLastError(), "exact eval") && checked(cudaMemcpy(out, d, (size_t)count * 4, cudaMemcpyDeviceToHost), "exact eval");
    cudaFree(d);
    return ok;
}

static void slots_free() {
    for (int b = 0; b < K3_SLOTS; b++) {
        cudaFree(S[b].w); cudaFree(S[b].rows); cudaFree(S[b].gq); cudaFree(S[b].gs);
        S[b].w = nullptr; S[b].rows = nullptr; S[b].gq = nullptr; S[b].gs = nullptr;
    }
    slot_bytes = 0;
}
static size_t smem_gate_up(int LT) { return 512 + (size_t)K3_TP * K3_TO * 4 + (size_t)K3_TP * (LT / 32) * 4 + (size_t)K3_TP * LT; }
static size_t smem_down(int MI) { return 512 + (size_t)K3_TP * (MI / 32) * 4 + (size_t)K3_TP * MI; }

extern "C" int k3cu_moe_begin(const int8_t *zq, const float *zs, int n, int LT, int MI, float b1, float b2,
                              size_t ebytes, const size_t *off, float *Y, int npairs) {
    if (ready && !on_device()) return 0;
    if (!ready || failed || m_on || !zq || !zs || !off || !Y || n < 1 || npairs < 1 ||
        LT < 32 || MI < 32 || LT % 32 || MI % 32 || !ebytes) return 0;
    if (smem_gate_up(LT) > 48 * 1024 || smem_down(MI) > 48 * 1024) return 0;   // K3: 33.8 and 28.2 KiB
    if (ebytes != slot_bytes) {
        slots_free();
        for (int b = 0; b < K3_SLOTS; b++)
            if (!checked(cudaMalloc(&S[b].w, ebytes), "expert slots") ||
                !checked(cudaMalloc(&S[b].rows, (size_t)K3_RCAP * sizeof(int)), "expert slots") ||
                !checked(cudaMalloc(&S[b].gq, (size_t)K3_RCAP * MI), "expert slots") ||
                !checked(cudaMalloc(&S[b].gs, (size_t)K3_RCAP * (MI / 32) * sizeof(float)), "expert slots")) { slots_free(); return 0; }
        slot_bytes = ebytes;
    }
    const size_t zb = (size_t)n * LT, sb = (size_t)n * (LT / 32) * sizeof(float);
    if (zb > zq_cap) { cudaFree(d_zq); d_zq = nullptr; zq_cap = 0; if (!checked(cudaMalloc(&d_zq, zb), "rows")) return 0; zq_cap = zb; }
    if (sb > zs_cap) { cudaFree(d_zs); d_zs = nullptr; zs_cap = 0; if (!checked(cudaMalloc(&d_zs, sb), "rows")) return 0; zs_cap = sb; }
    void *yd = nullptr;
    if (!checked(cudaHostGetDevicePointer(&yd, Y, 0), "outputs (not k3cu_host_alloc's?)") ||
        !checked(cudaMemcpyAsync(d_zq, zq, zb, cudaMemcpyHostToDevice, s_copy), "rows") ||
        !checked(cudaMemcpyAsync(d_zs, zs, sb, cudaMemcpyHostToDevice, s_copy), "rows") ||
        !checked(cudaStreamSynchronize(s_copy), "rows")) return 0;
    m_LT = LT; m_MI = MI; m_n = n; m_b1 = b1; m_b2 = b2; m_ebytes = ebytes;
    memcpy(m_off, off, sizeof m_off);
    m_Ydev = (float *)yd; m_npairs = npairs;
    for (int b = 0; b < K3_SLOTS; b++) S[b].busy = 0;
    m_next = 0; m_nlate = 0; m_on = 1;
    n_steps++;
    return 1;
}

extern "C" int k3cu_moe_expert(const uint8_t *bytes, const int *rows, int r, int pair0, int tag) {
    if (ready && !on_device()) return -1;
    if (!m_on || failed || !bytes || !rows || r < 1 || pair0 < 0 || pair0 + r > m_npairs) return 0;
    const int b = m_next;
    K3Slot *s = &S[b];
    if (s->busy) {   // its kernels before must be done before its bytes change
        double t0 = now_s();
        if (!checked(cudaEventSynchronize(s->done), "experts")) return 0;
        t_wait += now_s() - t0;
        s->busy = 0;
        m_late[m_nlate++] = s->tag;
    }
    for (int i = 0; i < r; i++) if (rows[i] < 0 || rows[i] >= m_n) return 0;
    double t0 = now_s();
    if (!checked(cudaMemcpyAsync(s->w, bytes, m_ebytes, cudaMemcpyHostToDevice, s_copy), "expert bytes") ||
        !checked(cudaEventRecord(e_copied, s_copy), "expert bytes") ||
        !checked(cudaEventSynchronize(e_copied), "expert bytes")) return 0;   // the caller's bytes are free again
    t_copy += now_s() - t0;
    const uint8_t *w = s->w;
    const uint8_t *w1p = w + m_off[0], *w1s = w + m_off[1], *w2p = w + m_off[2], *w2s = w + m_off[3],
                  *w3p = w + m_off[4], *w3s = w + m_off[5];
    const size_t s1 = smem_gate_up(m_LT), s2 = smem_down(m_MI);
    for (int p = 0; p < r; p += K3_RCAP) {
        const int pr = r - p < K3_RCAP ? r - p : K3_RCAP;
        // the piece's rows: a synchronous copy from pageable memory is done when it returns,
        // and the slot's stream orders it behind the slot's kernels before
        if (!checked(cudaMemcpyAsync(s->rows, rows + p, (size_t)pr * sizeof(int), cudaMemcpyHostToDevice, s->s), "expert rows")) return 0;
        dim3 g1((unsigned)(m_MI / K3_TO), (unsigned)((pr + K3_TP - 1) / K3_TP)), g2((unsigned)(m_LT / K3_TO), (unsigned)((pr + K3_TP - 1) / K3_TP));
        k3_gate_up<<<g1, 256, s1, s->s>>>(w1p, w1s, w3p, w3s, d_zq, d_zs, s->rows, pr, m_LT, m_MI, m_b1, m_b2, s->gq, s->gs);
        k3_down<<<g2, 256, s2, s->s>>>(w2p, w2s, s->gq, s->gs, pr, m_MI, m_LT, m_Ydev + (size_t)(pair0 + p) * m_LT);
        if (!checked(cudaGetLastError(), "expert kernels")) return 0;
        if (p + K3_RCAP < r && !checked(cudaStreamSynchronize(s->s), "expert pieces")) return 0;   // rows' room reused
    }
    if (!checked(cudaEventRecord(s->done, s->s), "experts")) return 0;
    s->busy = 1; s->tag = tag;
    m_next = (b + 1) % K3_SLOTS;
    n_experts++; n_rows += (unsigned long long)r;
    return 1;
}

extern "C" int k3cu_moe_poll(int *tags, int cap) {
    if (ready && !on_device()) return -1;
    if (!m_on || failed || !tags) return -1;
    int n = 0;
    while (m_nlate && n < cap) tags[n++] = m_late[--m_nlate];
    for (int b = 0; b < K3_SLOTS && n < cap; b++) {
        if (!S[b].busy) continue;
        cudaError_t e = cudaEventQuery(S[b].done);
        if (e == cudaErrorNotReady) continue;
        if (!checked(e, "experts")) return -1;
        S[b].busy = 0; tags[n++] = S[b].tag;
    }
    return n;
}

// the oldest expert in flight, waited for: 1 (the next poll names it), 0 when none is, -1 on error
extern "C" int k3cu_moe_wait(void) {
    if (ready && !on_device()) return -1;
    if (!m_on || failed) return -1;
    for (int k = 0; k < K3_SLOTS; k++) {
        const int b = (m_next + k) % K3_SLOTS;
        if (!S[b].busy) continue;
        return checked(cudaEventSynchronize(S[b].done), "experts") ? 1 : -1;
    }
    return 0;
}

extern "C" int k3cu_moe_end(void) {
    if (ready && !on_device()) return -1;
    if (!m_on) return 0;
    m_on = 0;
    int ok = !failed;
    for (int b = 0; b < K3_SLOTS; b++) {
        if (S[b].busy && !checked(cudaEventSynchronize(S[b].done), "experts")) ok = 0;
        S[b].busy = 0;
    }
    m_nlate = 0;
    return ok;
}

// y [S][O] = x [S][I] . W^T in the CPU's arithmetic (fmt 4: int4 groups of gs, scales [O][I/gs];
// fmt 1: int8 rows, a scale each), from host memory to host memory, rows K3_MMROWS at a time.
#define K3_MMROWS 1024
static cudaStream_t s_mm;
static uint8_t *mm_w; static float *mm_s, *mm_x, *mm_y; static size_t mm_wcap, mm_scap, mm_xcap, mm_ycap;
static unsigned long long n_mm, n_mm_rows; static double t_mm;
static int mm_room(void **p, size_t *cap, size_t need) {
    if (need <= *cap) return 1;
    cudaFree(*p); *p = nullptr; *cap = 0;
    if (!checked(cudaMalloc(p, need), "dense buffers")) return 0;
    *cap = need;
    return 1;
}
extern "C" int k3cu_matmul(int fmt, float *y, const float *x, const void *w, const float *scale,
                           int S, int I, int O, int gs) {
    if (!ready || failed || !y || !x || !w || !scale || S < 1 || I < 8 || O < 1 || !on_device()) return 0;
    if (fmt == 4 ? (gs < 16 || gs % 16 || I % gs || K3_KC % gs) : fmt != 1 || I % 8) return 0;
    double t0 = now_s();
    if (!s_mm && !checked(cudaStreamCreateWithFlags(&s_mm, cudaStreamNonBlocking), "dense stream")) return 0;
    const size_t wb = fmt == 4 ? (size_t)O * (I / 2) : (size_t)O * I, sb = (size_t)O * (fmt == 4 ? I / gs : 1) * sizeof(float);
    const int R = S < K3_MMROWS ? S : K3_MMROWS;
    if (!mm_room((void **)&mm_w, &mm_wcap, wb) || !mm_room((void **)&mm_s, &mm_scap, sb) ||
        !mm_room((void **)&mm_x, &mm_xcap, (size_t)R * I * sizeof(float)) ||
        !mm_room((void **)&mm_y, &mm_ycap, (size_t)R * O * sizeof(float))) return 0;
    if (!checked(cudaMemcpyAsync(mm_w, w, wb, cudaMemcpyHostToDevice, s_mm), "dense weights") ||
        !checked(cudaMemcpyAsync(mm_s, scale, sb, cudaMemcpyHostToDevice, s_mm), "dense scales")) return 0;
    for (int s0 = 0; s0 < S; s0 += R) {
        const int r = S - s0 < R ? S - s0 : R;
        if (!checked(cudaMemcpyAsync(mm_x, x + (size_t)s0 * I, (size_t)r * I * sizeof(float), cudaMemcpyHostToDevice, s_mm), "dense rows")) return 0;
        k3_dense_launch(fmt, mm_w, mm_s, mm_x, r, I, O, gs, mm_y, s_mm);
        if (!checked(cudaGetLastError(), "dense kernel") ||
            !checked(cudaMemcpyAsync(y + (size_t)s0 * O, mm_y, (size_t)r * O * sizeof(float), cudaMemcpyDeviceToHost, s_mm), "dense outputs")) return 0;
    }
    if (!checked(cudaStreamSynchronize(s_mm), "dense")) return 0;
    n_mm++; n_mm_rows += (unsigned long long)S; t_mm += now_s() - t0;
    return 1;
}

// The buffers of the layer calls that never run at once (the MLA core, the KDA layer: the engine's
// one thread, each call done when it returns): one device allocation carved per call, as large as
// the largest call has needed, so the KDA layer's pieces take no memory past what the MLA core
// already held (about 460 MB; the card has 2-2.5 GiB beside the resident matrices).
static char *ar_mem; static size_t ar_cap;
static int ar_carve(void **const *ptr, const size_t *bytes, int n) {
    size_t tot = 0;
    for (int i = 0; i < n; i++) tot += (bytes[i] + 255) & ~(size_t)255;
    if (tot > ar_cap) {
        cudaFree(ar_mem); ar_mem = nullptr; ar_cap = 0;
        if (!checked(cudaMalloc(&ar_mem, tot), "layer buffers")) return 0;
        ar_cap = tot;
    }
    size_t at = 0;
    for (int i = 0; i < n; i++) { *ptr[i] = ar_mem + at; at += (bytes[i] + 255) & ~(size_t)255; }
    return 1;
}

// A prompt step's MLA attention core (k3cu_mla, backend_cuda_k3.h)
static cudaStream_t s_mla;
static float *ml_LR, *ml_qv, *ml_gv, *ml_s, *ml_Q, *ml_S, *ml_cl, *ml_ctx; static int8_t *ml_w;
static unsigned long long n_mla, n_mla_rows; static double t_mla;
extern "C" int k3cu_mla(const float *Lc, const float *Rc, int N, int pos0, int C, const float *qv, const float *gv,
                        const int8_t *kvb, const float *kvbs, int H, int qn, int qr, int vh, int kvl, float scale,
                        float *ctx) {
    if (!ready || failed || !Lc || !qv || !gv || !kvb || !kvbs || !ctx || C < 1 || N != pos0 + C || H < 1 ||
        kvl < 8 || (qr && !Rc) || !on_device()) return 0;
    double t0 = now_s();
    const int qh = qn + qr, KD = kvl + qr, O = H * (qn + vh);
    if (!s_mla && !checked(cudaStreamCreateWithFlags(&s_mla, cudaStreamNonBlocking), "attention stream")) return 0;
    /* rows a chunk: their scores within about 128 MB (the card has 2-2.5 GiB beside the resident matrices) */
    long long tq = (128ll << 20) / ((long long)H * N * 4);
    const int TQ = tq < 1 ? 1 : tq > C ? C : (int)tq;
    void **const mp[9] = {(void **)&ml_LR, (void **)&ml_qv, (void **)&ml_gv, (void **)&ml_w, (void **)&ml_s,
                          (void **)&ml_Q, (void **)&ml_S, (void **)&ml_cl, (void **)&ml_ctx};
    const size_t mb[9] = {(size_t)N * KD * 4, (size_t)C * H * qh * 4, (size_t)C * H * vh * 4, (size_t)O * kvl, (size_t)O * 4,
                          (size_t)TQ * H * KD * 4, (size_t)TQ * H * N * 4, (size_t)TQ * H * kvl * 4, (size_t)C * H * vh * 4};
    if (!ar_carve(mp, mb, 9)) return 0;
    cudaStream_t st = s_mla;
    if (!checked(cudaMemcpy2DAsync(ml_LR, (size_t)KD * 4, Lc, (size_t)kvl * 4, (size_t)kvl * 4, (size_t)N, cudaMemcpyHostToDevice, st), "attention cache") ||
        (qr && !checked(cudaMemcpy2DAsync(ml_LR + kvl, (size_t)KD * 4, Rc, (size_t)qr * 4, (size_t)qr * 4, (size_t)N, cudaMemcpyHostToDevice, st), "attention cache")) ||
        !checked(cudaMemcpyAsync(ml_qv, qv, (size_t)C * H * qh * 4, cudaMemcpyHostToDevice, st), "attention rows") ||
        !checked(cudaMemcpyAsync(ml_gv, gv, (size_t)C * H * vh * 4, cudaMemcpyHostToDevice, st), "attention rows") ||
        !checked(cudaMemcpyAsync(ml_w, kvb, (size_t)O * kvl, cudaMemcpyHostToDevice, st), "kv_b") ||
        !checked(cudaMemcpyAsync(ml_s, kvbs, (size_t)O * 4, cudaMemcpyHostToDevice, st), "kv_b")) return 0;
    for (int c0 = 0; c0 < C; c0 += TQ) {
        const int rows = C - c0 < TQ ? C - c0 : TQ, M = rows * H;
        k3_mla_q<<<M, 256, 0, st>>>(ml_qv + (size_t)c0 * H * qh, H, qh, qn, qr, vh, kvl, ml_w, ml_s, ml_Q);
        dim3 gs((unsigned)((N + K3G_T - 1) / K3G_T), (unsigned)((M + K3G_T - 1) / K3G_T));
        k3_gemm<0, 1><<<gs, 256, 0, st>>>(M, N, KD, ml_Q, KD, ml_LR, KD, ml_S, N, scale);
        k3_mla_softmax<<<M, 256, 0, st>>>(ml_S, N, H, pos0 + c0);
        dim3 gp((unsigned)((kvl + K3G_T - 1) / K3G_T), (unsigned)((M + K3G_T - 1) / K3G_T));
        k3_gemm<1, 0><<<gp, 256, 0, st>>>(M, kvl, N, ml_S, N, ml_LR, KD, ml_cl, kvl, 0.0f);
        k3_mla_out<<<M, 128, 0, st>>>(ml_cl, ml_gv + (size_t)c0 * H * vh, H, qn, vh, kvl, ml_w, ml_s, ml_ctx + (size_t)c0 * H * vh);
        if (!checked(cudaGetLastError(), "attention kernels")) return 0;
    }
    if (!checked(cudaMemcpyAsync(ctx, ml_ctx, (size_t)C * H * vh * 4, cudaMemcpyDeviceToHost, st), "attention outputs") ||
        !checked(cudaStreamSynchronize(st), "attention")) return 0;
    n_mla++; n_mla_rows += (unsigned long long)C; t_mla += now_s() - t0;
    return 1;
}

// sco [C][E] = the rows x [C][D] against the bf16 router w [E][D], k3_router_scores's bytes
static float *rt_x, *rt_s; static uint16_t *rt_w; static size_t rt_xc, rt_sc, rt_wc;
static unsigned long long n_rt; static double t_rt;
extern "C" int k3cu_router(float *sco, const float *x, const uint16_t *w, int C, int D, int E) {
    if (!ready || failed || !sco || !x || !w || C < 1 || E < 1 || D < 32 || D % 32 || !on_device()) return 0;
    double t0 = now_s();
    if (!s_mm && !checked(cudaStreamCreateWithFlags(&s_mm, cudaStreamNonBlocking), "dense stream")) return 0;
    if (!mm_room((void **)&rt_x, &rt_xc, (size_t)C * D * 4) || !mm_room((void **)&rt_s, &rt_sc, (size_t)C * E * 4) ||
        !mm_room((void **)&rt_w, &rt_wc, (size_t)E * D * 2)) return 0;
    dim3 g((unsigned)((E + 7) / 8), (unsigned)((C + K3R_RT - 1) / K3R_RT));
    if (!checked(cudaMemcpyAsync(rt_w, w, (size_t)E * D * 2, cudaMemcpyHostToDevice, s_mm), "router") ||
        !checked(cudaMemcpyAsync(rt_x, x, (size_t)C * D * 4, cudaMemcpyHostToDevice, s_mm), "router rows")) return 0;
    k3_lanes32<uint16_t><<<g, 256, 0, s_mm>>>(rt_w, rt_x, C, D, E, rt_s);
    if (!checked(cudaGetLastError(), "router kernel") ||
        !checked(cudaMemcpyAsync(sco, rt_s, (size_t)C * E * 4, cudaMemcpyDeviceToHost, s_mm), "router scores") ||
        !checked(cudaStreamSynchronize(s_mm), "router")) return 0;
    n_rt++; t_rt += now_s() - t0;
    return 1;
}

// The shared experts of a prompt step: sd [S][D] = down(situ(gate x, up x)), every matrix int4
// groups (matmul_i4_grouped's arithmetic), SiTU situf_'s; rows K3_MMROWS at a time.
__global__ void k3_situ_rows(float *__restrict__ g, const float *__restrict__ u, size_t n, float b1, float b2) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = k3d_situ(g[i], u[i], b1, b2);
}
static uint8_t *sh_w[3]; static float *sh_s[3], *sh_x, *sh_g, *sh_u, *sh_y;
static size_t sh_wc[3], sh_sc[3], sh_xc, sh_gc, sh_uc, sh_yc;
static unsigned long long n_sh; static double t_sh;
extern "C" int k3cu_shared(float *sd, const float *x, int S, int D, int I, const uint8_t *gq, const float *gs,
                           const uint8_t *uq, const float *us, int ggs, const uint8_t *dq, const float *ds, int dgs,
                           float b1, float b2) {
    if (!ready || failed || !sd || !x || !gq || !uq || !dq || S < 1 || !on_device()) return 0;
    if (ggs < 16 || ggs % 16 || D % ggs || K3_KC % ggs || dgs < 16 || dgs % 16 || I % dgs || K3_KC % dgs) return 0;
    double t0 = now_s();
    if (!s_mm && !checked(cudaStreamCreateWithFlags(&s_mm, cudaStreamNonBlocking), "dense stream")) return 0;
    const int R = S < K3_MMROWS ? S : K3_MMROWS;
    const size_t wb[3] = {(size_t)I * (D / 2), (size_t)I * (D / 2), (size_t)D * (I / 2)};
    const size_t sb[3] = {(size_t)I * (D / ggs) * 4, (size_t)I * (D / ggs) * 4, (size_t)D * (I / dgs) * 4};
    const uint8_t *wq[3] = {gq, uq, dq}; const float *ws[3] = {gs, us, ds};
    for (int k = 0; k < 3; k++)
        if (!mm_room((void **)&sh_w[k], &sh_wc[k], wb[k]) || !mm_room((void **)&sh_s[k], &sh_sc[k], sb[k]) ||
            !checked(cudaMemcpyAsync(sh_w[k], wq[k], wb[k], cudaMemcpyHostToDevice, s_mm), "shared weights") ||
            !checked(cudaMemcpyAsync(sh_s[k], ws[k], sb[k], cudaMemcpyHostToDevice, s_mm), "shared scales")) return 0;
    if (!mm_room((void **)&sh_x, &sh_xc, (size_t)R * D * 4) || !mm_room((void **)&sh_g, &sh_gc, (size_t)R * I * 4) ||
        !mm_room((void **)&sh_u, &sh_uc, (size_t)R * I * 4) || !mm_room((void **)&sh_y, &sh_yc, (size_t)R * D * 4)) return 0;
    for (int s0 = 0; s0 < S; s0 += R) {
        const int r = S - s0 < R ? S - s0 : R;
        const size_t n = (size_t)r * I;
        if (!checked(cudaMemcpyAsync(sh_x, x + (size_t)s0 * D, (size_t)r * D * 4, cudaMemcpyHostToDevice, s_mm), "shared rows")) return 0;
        k3_dense_launch(4, sh_w[0], sh_s[0], sh_x, r, D, I, ggs, sh_g, s_mm);
        k3_dense_launch(4, sh_w[1], sh_s[1], sh_x, r, D, I, ggs, sh_u, s_mm);
        k3_situ_rows<<<(unsigned)((n + 255) / 256), 256, 0, s_mm>>>(sh_g, sh_u, n, b1, b2);
        k3_dense_launch(4, sh_w[2], sh_s[2], sh_g, r, I, D, dgs, sh_y, s_mm);
        if (!checked(cudaGetLastError(), "shared kernels") ||
            !checked(cudaMemcpyAsync(sd + (size_t)s0 * D, sh_y, (size_t)r * D * 4, cudaMemcpyDeviceToHost, s_mm), "shared outputs")) return 0;
    }
    if (!checked(cudaStreamSynchronize(s_mm), "shared")) return 0;
    n_sh++; t_sh += now_s() - t0;
    return 1;
}

// ---- a prompt step's KDA layer (k3cu_kda): kda_forward's by-step path to the bit ---------------
// The conv: channel d of q, k or v (c = which P + d) keeps the window of its K latest inputs (the
// step's before it in win, [3][P][K]); a row shifts it and takes the row's input, its K taps fused
// into +0 in order, then SiLU as exp_f8's path computes it, x / (1 + exp_f(-x)). In place; the
// window goes back to win.
template <int K>
__global__ void __launch_bounds__(256) k3_kda_conv(float *__restrict__ q, float *__restrict__ k, float *__restrict__ v,
                                                   float *__restrict__ win, const float *__restrict__ taps, int R, int P) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 3 * P) return;
    const int which = c / P, d = c - which * P;
    float *val = (which == 0 ? q : which == 1 ? k : v) + d;
    float w[K], tp[K];
#pragma unroll
    for (int j = 0; j < K; j++) { w[j] = win[(size_t)c * K + j]; tp[j] = taps[(size_t)c * K + j]; }
    for (int t = 0; t < R; t++) {
#pragma unroll
        for (int j = 0; j < K - 1; j++) w[j] = w[j + 1];
        w[K - 1] = val[(size_t)t * P];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < K; j++) acc = __fmaf_rn(tp[j], w[j], acc);
        val[(size_t)t * P] = __fdiv_rn(acc, __fadd_rn(1.0f, k3d_expf(-acc)));
    }
#pragma unroll
    for (int j = 0; j < K; j++) win[(size_t)c * K + j] = w[j];
}
// A head's row (i = t H + h): the squares of q and of k rounded and added in order from 0 (GCC's
// in-order reduction of k3_kda_head's loop), 1 / sqrt(sum + 1e-6); beta = sigmoidf_(b).
__global__ void k3_kda_norms(const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ b,
                             int n, int hd, float *__restrict__ rq, float *__restrict__ rk, float *__restrict__ beta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *qh = q + (size_t)i * hd, *kh = k + (size_t)i * hd;
    float sq = 0.0f, sk = 0.0f;
    for (int e = 0; e < hd; e++) {
        sq = __fadd_rn(sq, __fmul_rn(qh[e], qh[e]));
        sk = __fadd_rn(sk, __fmul_rn(kh[e], kh[e]));
    }
    rq[i] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(sq, 1e-6f)));
    rk[i] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(sk, 1e-6f)));
    beta[i] = __fdiv_rn(1.0f, __fadd_rn(1.0f, k3d_expf(-b[i])));
}
// In place: q (q r_q) qscale, k r_k, and the decay alpha = exp_f(lb / (1 + exp_f(-(A_h (graw + dt)))))
// in g (k3_kda_head's exp_f8 loop)
__global__ void k3_kda_prep(float *__restrict__ q, float *__restrict__ k, float *__restrict__ g,
                            const float *__restrict__ rq, const float *__restrict__ rk, const float *__restrict__ A,
                            const float *__restrict__ dt, size_t n, int H, int hd, float qscale, float lb) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t th = i / hd;
    const int h = (int)(th % H), p = (int)(i % ((size_t)H * hd));
    q[i] = __fmul_rn(__fmul_rn(q[i], rq[th]), qscale);
    k[i] = __fmul_rn(k[i], rk[th]);
    const float z = __fadd_rn(g[i], dt[p]);
    const float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, k3d_expf(-__fmul_rn(A[h], z))));
    g[i] = k3d_expf(__fmul_rn(lb, s));
}
// The recurrence: block h, thread j owns column j of the head's state S [HD][HD] in registers. A row:
// row kk times alpha[kk], kS fusing k[kk] times it in kk's order from 0; vt = (v - kS) beta; row kk
// plus k[kk] vt fused, oh fusing q[kk] times it in kk's order from 0 (k3_kda_head's two sweeps).
// oh replaces v. The rows' q, k and alpha come KR_T at a time through shared memory.
#define KR_T 8
template <int HD>
__global__ void __launch_bounds__(HD) k3_kda_rec(float *__restrict__ state, const float *__restrict__ qn,
                                                 const float *__restrict__ kn, const float *__restrict__ al,
                                                 float *__restrict__ v, const float *__restrict__ beta, int R, int H) {
    __shared__ __align__(16) float sq[KR_T][HD], sk[KR_T][HD], sa[KR_T][HD];
    __shared__ float sb[KR_T];
    const int h = blockIdx.x, j = threadIdx.x, P = H * HD;
    float *S = state + (size_t)h * HD * HD + j;
    float s[HD];
#pragma unroll
    for (int kk = 0; kk < HD; kk++) s[kk] = S[(size_t)kk * HD];
    for (int t0 = 0; t0 < R; t0 += KR_T) {
        const int n = R - t0 < KR_T ? R - t0 : KR_T;
        __syncthreads();
        for (int tt = 0; tt < n; tt++) {
            const size_t src = (size_t)(t0 + tt) * P + (size_t)h * HD + j;
            sq[tt][j] = qn[src]; sk[tt][j] = kn[src]; sa[tt][j] = al[src];
        }
        if (j < n) sb[j] = beta[(size_t)(t0 + j) * H + h];
        __syncthreads();
        for (int tt = 0; tt < n; tt++) {
            float kS = 0.0f;
#pragma unroll
            for (int kk = 0; kk < HD; kk++) { s[kk] = __fmul_rn(s[kk], sa[tt][kk]); kS = __fmaf_rn(sk[tt][kk], s[kk], kS); }
            const size_t iv = (size_t)(t0 + tt) * P + (size_t)h * HD + j;
            const float vt = __fmul_rn(__fsub_rn(v[iv], kS), sb[tt]);
            float oh = 0.0f;
#pragma unroll
            for (int kk = 0; kk < HD; kk++) { s[kk] = __fmaf_rn(sk[tt][kk], vt, s[kk]); oh = __fmaf_rn(sq[tt][kk], s[kk], oh); }
            v[iv] = oh;
        }
    }
#pragma unroll
    for (int kk = 0; kk < HD; kk++) S[(size_t)kk * HD] = s[kk];
}
// A head's row: 1 / sqrtf((float)(sum of oh^2 in double, in order) / hd) + eps) (the per-head RMSNorm)
__global__ void k3_kda_rms(const float *__restrict__ oh, int n, int hd, float eps, float *__restrict__ rr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *o = oh + (size_t)i * hd;
    double ms = 0.0;
    for (int e = 0; e < hd; e++) { const double d = (double)o[e]; ms = __dadd_rn(ms, __dmul_rn(d, d)); }
    rr[i] = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__double2float_rn(__ddiv_rn(ms, (double)hd)), eps)));
}
// In place in g: ((oh r) onw) sigmoidf_(g), the gated heads o_proj takes
__global__ void k3_kda_gate(const float *__restrict__ oh, float *__restrict__ g, const float *__restrict__ rr,
                            const float *__restrict__ onw, size_t n, int hd) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, k3d_expf(-g[i])));
    g[i] = __fmul_rn(__fmul_rn(__fmul_rn(oh[i], rr[i / hd]), onw[i % hd]), sg);
}

#define KD_ROWS 512   // rows a piece (K3_GPU_KDA_ROWS)
static cudaStream_t s_kda;
static unsigned long long n_kda, n_kda_rows; static double t_kda;
static int kd_rows(void) {
    static int r = -1;
    if (r < 0) { const char *e = getenv("K3_GPU_KDA_ROWS"); r = e && atoi(e) > 0 ? atoi(e) : KD_ROWS; r = (r + 7) / 8 * 8; }
    return r;
}
// a projection on coli_cuda.dll's copy: quant_matmul_blocked's shapes
static int kd_dev_ok(const K3cuProj *p, int I, int O) {
    return p->w && p->s && (p->fmt == 1 || (p->fmt == 4 && p->gs == 64)) && (I == 7168 || I == 12288) && O % QMB_RO == 0;
}
static void kd_mm(const K3cuProj *p, float *y, const float *xb, int r, int I, int O, cudaStream_t st) {
    const dim3 grid((unsigned)(O / QMB_RO), (unsigned)((r + QMB_RS - 1) / QMB_RS));
    const uint8_t *w = static_cast<const uint8_t *>(p->w);
    if (p->fmt == 4 && I == 7168) quant_matmul_blocked<4, 7168><<<grid, 256, 0, st>>>(y, xb, w, p->s, r, O);
    else if (p->fmt == 4) quant_matmul_blocked<4, 12288><<<grid, 256, 0, st>>>(y, xb, w, p->s, r, O);
    else if (I == 7168) quant_matmul_blocked<1, 7168><<<grid, 256, 0, st>>>(y, xb, w, p->s, r, O);
    else quant_matmul_blocked<1, 12288><<<grid, 256, 0, st>>>(y, xb, w, p->s, r, O);
}
static void kd_block(float *xb, const float *x, int r, int I, cudaStream_t st) {
    const long long n = (long long)((r + QMB_RS - 1) / QMB_RS) * I;
    qmb_rows<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(xb, x, r, I, n);
}
template <int HD>
static void kd_rec(float *st8, const float *q, const float *k, const float *a, float *v, const float *b, int r, int H, cudaStream_t s) {
    k3_kda_rec<HD><<<H, HD, 0, s>>>(st8, q, k, a, v, b, r, H);
}
static int kd_conv(int K, float *q, float *k, float *v, float *win, const float *taps, int r, int P, cudaStream_t st) {
    const unsigned g = (unsigned)((3 * P + 255) / 256);
    switch (K) {
    case 1: k3_kda_conv<1><<<g, 256, 0, st>>>(q, k, v, win, taps, r, P); return 1;
    case 2: k3_kda_conv<2><<<g, 256, 0, st>>>(q, k, v, win, taps, r, P); return 1;
    case 3: k3_kda_conv<3><<<g, 256, 0, st>>>(q, k, v, win, taps, r, P); return 1;
    case 4: k3_kda_conv<4><<<g, 256, 0, st>>>(q, k, v, win, taps, r, P); return 1;
    case 5: k3_kda_conv<5><<<g, 256, 0, st>>>(q, k, v, win, taps, r, P); return 1;
    case 6: k3_kda_conv<6><<<g, 256, 0, st>>>(q, k, v, win, taps, r, P); return 1;
    case 7: k3_kda_conv<7><<<g, 256, 0, st>>>(q, k, v, win, taps, r, P); return 1;
    }
    return 0;
}
extern "C" int k3cu_kda(const K3cuKda *a, const float *x, float *out) {
    if (!ready || failed || !a || !x || !on_device()) return 0;
    const int C = a->C, D = a->D, H = a->H, hd = a->hd, K = a->K, P = H * hd;
    if (C < 1 || D < 1 || H < 1 || K < 1 || K > 7 || !(hd == 16 || hd == 32 || hd == 64 || hd == 128) ||
        !a->fa || !a->fb || !a->bp || !a->A || !a->dt || !a->onw || !a->state) return 0;
    const K3cuProj *pr[4] = {&a->q, &a->k, &a->v, &a->g};
    int dev_in = 0;
    for (int i = 0; i < 4; i++) {
        if (pr[i]->w ? !kd_dev_ok(pr[i], D, P) : !pr[i]->y) return 0;
        dev_in |= pr[i]->w != nullptr;
    }
    if (a->o.w ? !kd_dev_ok(&a->o, P, D) || !out : !a->o.y) return 0;
    for (int w = 0; w < 3; w++) if (!a->taps[w] || !a->win[w]) return 0;
    double t0 = now_s();
    if (!s_kda && !checked(cudaStreamCreateWithFlags(&s_kda, cudaStreamNonBlocking), "KDA stream")) return 0;
    cudaStream_t st = s_kda;
    const int R = C < kd_rows() ? C : kd_rows(), RB = (R + QMB_RS - 1) / QMB_RS * QMB_RS;
    const size_t PK = (size_t)P * K;
    // the layer's constants: fa, fb, bp, A, dt, onw, the q, k, v taps; its state: S, the q, k, v windows
    const size_t nfa = (size_t)hd * D, nfb = (size_t)P * hd, nbp = (size_t)H * D;
    const size_t ncon = nfa + nfb + nbp + H + P + hd + 3 * PK, nst = (size_t)H * hd * hd + 3 * PK;
    float *dx, *dxb, *dq, *dk, *dv, *da, *dc, *ds, *dsm;
    void **const kp[9] = {(void **)&dx, (void **)&dxb, (void **)&dq, (void **)&dk, (void **)&dv, (void **)&da,
                          (void **)&dc, (void **)&ds, (void **)&dsm};
    const size_t kb[9] = {(size_t)R * D * 4, (size_t)RB * D * 4, (size_t)RB * P * 4, (size_t)R * P * 4, (size_t)R * P * 4,
                          (size_t)R * P * 4, ncon * 4, nst * 4, ((size_t)R * hd + 4 * (size_t)R * H) * 4};
    if (!ar_carve(kp, kb, 9)) return 0;
    float *cfa = dc, *cfb = cfa + nfa, *cbp = cfb + nfb, *cA = cbp + nbp, *cdt = cA + H, *conw = cdt + P, *ctap = conw + hd;
    float *sS = ds, *swin = ds + (size_t)H * hd * hd;
    float *t1 = dsm, *br = t1 + (size_t)R * hd, *rq = br + (size_t)R * H, *rk = rq + (size_t)R * H, *be = rk + (size_t)R * H;
    const struct { float *d; const float *h; size_t n; } up[10] = {
        {cfa, a->fa, nfa}, {cfb, a->fb, nfb}, {cbp, a->bp, nbp}, {cA, a->A, (size_t)H}, {cdt, a->dt, (size_t)P},
        {conw, a->onw, (size_t)hd}, {ctap, a->taps[0], PK}, {ctap + PK, a->taps[1], PK}, {ctap + 2 * PK, a->taps[2], PK},
        {sS, a->state, (size_t)H * hd * hd}};
    for (int i = 0; i < 10; i++)
        if (!checked(cudaMemcpyAsync(up[i].d, up[i].h, up[i].n * 4, cudaMemcpyHostToDevice, st), "KDA constants")) return 0;
    for (int w = 0; w < 3; w++)
        if (!checked(cudaMemcpyAsync(swin + w * PK, a->win[w], PK * 4, cudaMemcpyHostToDevice, st), "KDA windows")) return 0;
    float *dqkv[3] = {dq, dk, dv};
    for (int s0 = 0; s0 < C; s0 += R) {
        const int r = C - s0 < R ? C - s0 : R;
        const size_t rp = (size_t)r * P;
        const unsigned grh = (unsigned)((r * H + 255) / 256), grp = (unsigned)((rp + 255) / 256);
        if (!checked(cudaMemcpyAsync(dx, x + (size_t)s0 * D, (size_t)r * D * 4, cudaMemcpyHostToDevice, st), "KDA rows")) return 0;
        if (dev_in) kd_block(dxb, dx, r, D, st);
        for (int i = 0; i < 3; i++) {
            if (pr[i]->w) kd_mm(pr[i], dqkv[i], dxb, r, D, P, st);
            else if (!checked(cudaMemcpyAsync(dqkv[i], pr[i]->y + (size_t)s0 * P, rp * 4, cudaMemcpyHostToDevice, st), "KDA projections")) return 0;
        }
        kd_conv(K, dq, dk, dv, swin, ctap, r, P, st);
        k3_lanes32<float><<<dim3((unsigned)((hd + 7) / 8), (unsigned)((r + K3R_RT - 1) / K3R_RT)), 256, 0, st>>>(cfa, dx, r, D, hd, t1);
        k3_lanes32<float><<<dim3((unsigned)((P + 7) / 8), (unsigned)((r + K3R_RT - 1) / K3R_RT)), 256, 0, st>>>(cfb, t1, r, hd, P, da);
        k3_lanes32<float><<<dim3((unsigned)((H + 7) / 8), (unsigned)((r + K3R_RT - 1) / K3R_RT)), 256, 0, st>>>(cbp, dx, r, D, H, br);
        k3_kda_norms<<<grh, 256, 0, st>>>(dq, dk, br, r * H, hd, rq, rk, be);
        k3_kda_prep<<<grp, 256, 0, st>>>(dq, dk, da, rq, rk, cA, cdt, rp, H, hd, a->qscale, a->gate_lb);
        if (hd == 128) kd_rec<128>(sS, dq, dk, da, dv, be, r, H, st);
        else if (hd == 64) kd_rec<64>(sS, dq, dk, da, dv, be, r, H, st);
        else if (hd == 32) kd_rec<32>(sS, dq, dk, da, dv, be, r, H, st);
        else kd_rec<16>(sS, dq, dk, da, dv, be, r, H, st);
        k3_kda_rms<<<grh, 256, 0, st>>>(dv, r * H, hd, a->eps, rq);
        if (a->g.w) kd_mm(&a->g, dk, dxb, r, D, P, st);
        else if (!checked(cudaMemcpyAsync(dk, a->g.y + (size_t)s0 * P, rp * 4, cudaMemcpyHostToDevice, st), "KDA gate")) return 0;
        k3_kda_gate<<<grp, 256, 0, st>>>(dv, dk, rq, conw, rp, hd);
        if (!checked(cudaGetLastError(), "KDA kernels")) return 0;
        if (a->o.w) {
            kd_block(dq, dk, r, P, st);
            kd_mm(&a->o, dx, dq, r, P, D, st);
            if (!checked(cudaGetLastError(), "KDA output") ||
                !checked(cudaMemcpyAsync(out + (size_t)s0 * D, dx, (size_t)r * D * 4, cudaMemcpyDeviceToHost, st), "KDA output")) return 0;
        } else if (!checked(cudaMemcpyAsync(a->o.y + (size_t)s0 * P, dk, rp * 4, cudaMemcpyDeviceToHost, st), "KDA heads")) return 0;
    }
    if (!checked(cudaStreamSynchronize(st), "KDA")) return 0;   // every piece done: the host state may change now
    if (!checked(cudaMemcpyAsync(a->state, sS, (size_t)H * hd * hd * 4, cudaMemcpyDeviceToHost, st), "KDA state")) return 0;
    for (int w = 0; w < 3; w++)
        if (!checked(cudaMemcpyAsync(a->win[w], swin + w * PK, PK * 4, cudaMemcpyDeviceToHost, st), "KDA windows")) return 0;
    if (!checked(cudaStreamSynchronize(st), "KDA")) return 0;
    n_kda++; n_kda_rows += (unsigned long long)C; t_kda += now_s() - t0;
    return 1;
}

extern "C" void k3cu_report(void) {
    if (!ready) return;
    if (n_steps)
        fprintf(stderr, "[K3-CUDA] experts: %llu steps, %llu experts, %llu rows on the GPU; copying their bytes %.1fs, waiting for slots %.1fs%s\n",
                n_steps, n_experts, n_rows, t_copy, t_wait, failed ? "; the device failed" : "");
    if (n_mm) fprintf(stderr, "[K3-CUDA] the dense matrices the GPU does not hold: %llu matmuls, %llu rows, %.1fs\n", n_mm, n_mm_rows, t_mm);
    if (n_mla) fprintf(stderr, "[K3-CUDA] the gated MLA's attention: %llu layer steps, %llu rows, %.1fs\n", n_mla, n_mla_rows, t_mla);
    if (n_rt) fprintf(stderr, "[K3-CUDA] the routers: %llu steps, %.1fs\n", n_rt, t_rt);
    if (n_sh) fprintf(stderr, "[K3-CUDA] the shared experts: %llu steps, %.1fs\n", n_sh, t_sh);
    if (n_kda) fprintf(stderr, "[K3-CUDA] the KDA layers: %llu layer steps, %llu rows, %.1fs\n", n_kda, n_kda_rows, t_kda);
}

extern "C" void k3cu_shutdown(void) {
    if (!ready) return;
    if (m_on) k3cu_moe_end();
    slots_free();
    cudaFree(d_zq); cudaFree(d_zs); d_zq = nullptr; d_zs = nullptr; zq_cap = zs_cap = 0;
    cudaFree(mm_w); cudaFree(mm_s); cudaFree(mm_x); cudaFree(mm_y); mm_w = nullptr; mm_s = mm_x = mm_y = nullptr;
    mm_wcap = mm_scap = mm_xcap = mm_ycap = 0;
    cudaFree(ar_mem); ar_mem = nullptr; ar_cap = 0;
    if (s_mm) { cudaStreamDestroy(s_mm); s_mm = nullptr; }
    if (s_kda) { cudaStreamDestroy(s_kda); s_kda = nullptr; }
    for (int b = 0; b < K3_SLOTS; b++) { cudaStreamDestroy(S[b].s); cudaEventDestroy(S[b].done); }
    cudaStreamDestroy(s_copy); cudaEventDestroy(e_copied);
    ready = 0;
}
