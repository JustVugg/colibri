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
    if (p0 + l < r) y[(size_t)(p0 + l) * LT + o] = v;
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
    if (!ready || cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) { cudaGetLastError(); return nullptr; }
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
        dim3 g((unsigned)((O + K3_TO - 1) / K3_TO), (unsigned)((r + K3_TP - 1) / K3_TP));
        if (!checked(cudaMemcpyAsync(mm_x, x + (size_t)s0 * I, (size_t)r * I * sizeof(float), cudaMemcpyHostToDevice, s_mm), "dense rows")) return 0;
        k3_dense<<<g, 256, 0, s_mm>>>(fmt, mm_w, mm_s, mm_x, r, I, O, gs, mm_y);
        if (!checked(cudaGetLastError(), "dense kernel") ||
            !checked(cudaMemcpyAsync(y + (size_t)s0 * O, mm_y, (size_t)r * O * sizeof(float), cudaMemcpyDeviceToHost, s_mm), "dense outputs")) return 0;
    }
    if (!checked(cudaStreamSynchronize(s_mm), "dense")) return 0;
    n_mm++; n_mm_rows += (unsigned long long)S; t_mm += now_s() - t0;
    return 1;
}

extern "C" void k3cu_report(void) {
    if (!ready || !n_steps) return;
    fprintf(stderr, "[K3-CUDA] experts: %llu steps, %llu experts, %llu rows on the GPU; copying their bytes %.1fs, waiting for slots %.1fs%s\n",
            n_steps, n_experts, n_rows, t_copy, t_wait, failed ? "; the device failed" : "");
    if (n_mm) fprintf(stderr, "[K3-CUDA] the dense matrices the GPU does not hold: %llu matmuls, %llu rows, %.1fs\n", n_mm, n_mm_rows, t_mm);
}

extern "C" void k3cu_shutdown(void) {
    if (!ready) return;
    if (m_on) k3cu_moe_end();
    slots_free();
    cudaFree(d_zq); cudaFree(d_zs); d_zq = nullptr; d_zs = nullptr; zq_cap = zs_cap = 0;
    cudaFree(mm_w); cudaFree(mm_s); cudaFree(mm_x); cudaFree(mm_y); mm_w = nullptr; mm_s = mm_x = mm_y = nullptr;
    mm_wcap = mm_scap = mm_xcap = mm_ycap = 0;
    if (s_mm) { cudaStreamDestroy(s_mm); s_mm = nullptr; }
    for (int b = 0; b < K3_SLOTS; b++) { cudaStreamDestroy(S[b].s); cudaEventDestroy(S[b].done); }
    cudaStreamDestroy(s_copy); cudaEventDestroy(e_copied);
    ready = 0;
}
