// bench_qmt.cu: coli_cuda.dll's quant_matmul_tiled (verbatim) against new kernels with
// the same bytes, at Kimi K3's resident shapes. nvcc -O3 -arch=sm_89
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);}}while(0)

/* ---------------- the reference: backend_cuda.cu's quant_matmul_tiled ---------------- */
#define QMT_TO 8
#define QMT_TS 8
__global__ static void __launch_bounds__(256) quant_matmul_tiled(float *y, const float *x,
        const void *weights, const float *scales, int fmt, int S, int I, int O,
        size_t rb, int gs, int ng) {
    __shared__ float red[2][QMT_TS][256];
    const int t = threadIdx.x, o0 = blockIdx.x * QMT_TO, s0 = blockIdx.y * QMT_TS;
    const uint8_t *wb = static_cast<const uint8_t *>(weights);
    float acc[QMT_TO][QMT_TS];
#pragma unroll
    for (int a = 0; a < QMT_TO; a++)
#pragma unroll
        for (int b = 0; b < QMT_TS; b++) acc[a][b] = 0.0f;
    const int no = O - o0 < QMT_TO ? O - o0 : QMT_TO, ns = S - s0 < QMT_TS ? S - s0 : QMT_TS;
    for (int i = t; i < I; i += 256) {
        float xv[QMT_TS];
#pragma unroll
        for (int b = 0; b < QMT_TS; b++) xv[b] = b < ns ? x[(size_t)(s0 + b) * I + i] : 0.0f;
        int g = 0;
        if (fmt == 4) { g = i / gs; if (g >= ng) g = ng - 1; }
#pragma unroll
        for (int a = 0; a < QMT_TO; a++) {
            if (a >= no) break;
            const uint8_t *row = wb + (size_t)(o0 + a) * rb;
            float w;
            if (fmt == 4) {
                uint8_t v = row[i >> 1];
                int n = (i & 1) ? (v >> 4) : (v & 15);
                w = static_cast<float>(n & 8 ? n - 16 : n);
                float sc = scales[(size_t)(o0 + a) * ng + g];
#pragma unroll
                for (int b = 0; b < QMT_TS; b++) acc[a][b] = __fmaf_rn(__fmul_rn(xv[b], w), sc, acc[a][b]);
            } else {
                w = static_cast<float>(reinterpret_cast<const int8_t *>(row)[i]);
#pragma unroll
                for (int b = 0; b < QMT_TS; b++) acc[a][b] = __fmaf_rn(xv[b], w, acc[a][b]);
            }
        }
    }
    const int w = t >> 5, lane = t & 31;
#pragma unroll
    for (int a0 = 0; a0 < QMT_TO; a0 += 2) {
        if (a0 >= no) break;
#pragma unroll
        for (int q = 0; q < 2; q++)
#pragma unroll
            for (int b = 0; b < QMT_TS; b++) red[q][b][t] = acc[a0 + q][b];
        __syncthreads();
#pragma unroll
        for (int n = 128; n >= 32; n >>= 1) {
            if (t < n)
#pragma unroll
                for (int q = 0; q < 2; q++)
#pragma unroll
                    for (int b = 0; b < QMT_TS; b++) red[q][b][t] += red[q][b][t + n];
            __syncthreads();
        }
#pragma unroll
        for (int q = 0; q < 2; q++) {
            float v = red[q][w][lane];
            v += __shfl_down_sync(0xffffffffu, v, 16);
            v += __shfl_down_sync(0xffffffffu, v, 8);
            v += __shfl_down_sync(0xffffffffu, v, 4);
            v += __shfl_down_sync(0xffffffffu, v, 2);
            v += __shfl_down_sync(0xffffffffu, v, 1);
            if (!lane && w < ns && a0 + q < no) {
                int o = o0 + a0 + q;
                y[(size_t)(s0 + w) * O + o] = fmt == 1 ? v * scales[o] : v;
            }
        }
        __syncthreads();
    }
}

/* ---------------- new: rows in blocks of RS, each thread chain t of RS x RO outputs ---------------- */
/* xt[(sb * I + i) * RS + r] = x[(sb * RS + r) * I + i], rows past S zero */
template <int RS>
__global__ static void qmb_xt(float *__restrict__ xt, const float *__restrict__ x, int S, int I, long long n) {
    long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const int sb = (int)(e / I), i = (int)(e % I);
    float v[RS];
#pragma unroll
    for (int r = 0; r < RS; r++) { const int s = sb * RS + r; v[r] = s < S ? x[(size_t)s * I + i] : 0.0f; }
    float4 *d = reinterpret_cast<float4 *>(xt + (size_t)e * RS);
#pragma unroll
    for (int q = 0; q < RS / 4; q++) d[q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
}

#define QMB_P 32   /* pairs a reduction round */
template <int FMT, int I, int RS, int RO, int PROBE = 0>
__global__ static void __launch_bounds__(256, 1) qmb_kernel(float *__restrict__ y, const float *__restrict__ xt,
        const uint8_t *__restrict__ w, const float *__restrict__ scales, int S, int O) {
    constexpr int K = I / 256, RB = FMT == 4 ? I / 2 : I, NG = FMT == 4 ? I / 64 : 1;
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
    const float *xp = xt + ((size_t)(PROBE == 1 ? 0 : sb) * I + t) * RS;
    const uint8_t *wp = w + (size_t)(PROBE == 2 ? 0 : o0) * RB + (FMT == 4 ? (t >> 1) : t);
    /* a weight's nibble to its value: (byte & m) ^ c reads as 2^23 + 16^p (n + 8), p the
     * nibble's place, then one fused add takes the place and the offset out, exactly */
    const uint32_t dm = (t & 1) ? 0xF0u : 0x0Fu, dc = (t & 1) ? 0x4B000080u : 0x4B000008u;
    const float dmul = (t & 1) ? 0.0625f : 1.0f, dadd = (t & 1) ? -524296.0f : -8388616.0f;
    /* two register sets, A and B: one step's input and weights in flight while the other's compute */
    auto load = [&](float (&xd)[RS], uint32_t (&bd)[RO], int k) {
#pragma unroll
        for (int q = 0; q < RS / 4; q++) {
            const float4 v = __ldg(reinterpret_cast<const float4 *>(xp + (size_t)k * 256 * RS) + q);
            xd[4 * q] = v.x; xd[4 * q + 1] = v.y; xd[4 * q + 2] = v.z; xd[4 * q + 3] = v.w;
        }
#pragma unroll
        for (int j = 0; j < RO; j++) bd[j] = __ldg(wp + (size_t)j * RB + (size_t)k * (FMT == 4 ? 128 : 256));
    };
    auto step = [&](const float (&xv)[RS], const uint32_t (&b)[RO], int k) {
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
    };
    static_assert(K % 2 == 0, "steps in pairs");
    float xa[RS], xb[RS];
    uint32_t ba[RO], bb[RO];
    load(xa, ba, 0);
#pragma unroll 1
    for (int k = 0; k < K; k += 2) {
        load(xb, bb, k + 1);
        step(xa, ba, k);
        if (k + 2 < K) load(xa, ba, k + 2);
        step(xb, bb, k + 1);
    }
    /* quant_matmul's tree: levels 128, 64, 32 across the warps (h = t >> 5), 16 .. 1 in a warp */
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
            float a0 = v[0] + v[4], a1 = v[1] + v[5], a2 = v[2] + v[6], a3 = v[3] + v[7];
            float b0 = a0 + a2, b1 = a1 + a3;
            float s = b0 + b1;
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

/* NH threads a chain: thread (t, hf) holds chain t of RS rows by RO outputs, its half hf of
 * the block's NH * RO outputs; the halves read the same inputs (the second from L1). */
#define QMH_P 16
template <int FMT, int I, int RS, int RO, int NH, int PROBE = 0>
__global__ static void __launch_bounds__(256 * NH, 1) qmh_kernel(float *__restrict__ y, const float *__restrict__ xt,
        const uint8_t *__restrict__ w, const float *__restrict__ scales, int S, int O) {
    constexpr int K = I / 256, RB = FMT == 4 ? I / 2 : I, NG = FMT == 4 ? I / 64 : 1, BO = RO * NH;
    static_assert(I % 256 == 0 && (RS * RO) % QMH_P == 0 && RS % 4 == 0 && RO % 4 == 0, "shape");
    __shared__ __align__(16) float ssc[NG][BO];
    __shared__ float red[NH][QMH_P][256];
    const int tid = threadIdx.x, t = tid & 255, hf = tid >> 8, ob = blockIdx.x * BO, o0 = ob + hf * RO, sb = blockIdx.y;
    if (FMT == 4) {
        for (int e = tid; e < NG * BO; e += 256 * NH) { const int j = e / NG, g = e % NG; ssc[g][j] = scales[(size_t)(ob + j) * NG + g]; }
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
                const float4 v = __ldg(reinterpret_cast<const float4 *>(xp + (size_t)(PROBE == 3 ? ((k + 1) & 1) : k + 1) * 256 * RS) + q);
                xn[4 * q] = v.x; xn[4 * q + 1] = v.y; xn[4 * q + 2] = v.z; xn[4 * q + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < RO; j++) bn[j] = __ldg(wp + (size_t)j * RB + (size_t)(PROBE == 4 ? ((k + 1) & 1) : k + 1) * (FMT == 4 ? 128 : 256));
        }
        float sc[RO];
        if (FMT == 4) {
            const int g = 4 * k + (t >> 6);
#pragma unroll
            for (int q = 0; q < RO / 4; q++) {
                const float4 v = *reinterpret_cast<const float4 *>(&ssc[g][hf * RO + 4 * q]);
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
    for (int p0 = 0; p0 < RS * RO; p0 += QMH_P) {
#pragma unroll
        for (int p = 0; p < QMH_P; p++) red[hf][p][t] = acc[(p0 + p) / RO][(p0 + p) % RO];
        __syncthreads();
#pragma unroll
        for (int p = h; p < QMH_P; p += 8) {
            float v[8];
#pragma unroll
            for (int q = 0; q < 8; q++) v[q] = red[hf][p][32 * q + c];
            float a0 = v[0] + v[4], a1 = v[1] + v[5], a2 = v[2] + v[6], a3 = v[3] + v[7];
            float b0 = a0 + a2, b1 = a1 + a3;
            float s = b0 + b1;
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

static float run_ref(float *dy, const float *dx, const uint8_t *dw, const float *dsc, int fmt, int S, int I, int O, int reps);
template <int FMT, int I, int RS, int RO, int NH, int PROBE = 0>
static void one_h(const char *name, int O, int S, int reps) {
    size_t rb = FMT == 1 ? (size_t)I : (size_t)I / 2;
    size_t sn = FMT == 1 ? (size_t)O : (size_t)O * (I / 64);
    std::vector<uint8_t> w(rb * O); std::vector<float> sc(sn), x((size_t)S * I);
    for (auto &v : w) v = (uint8_t)(rand() * 7);
    for (auto &v : sc) v = (float)(rand() % 1000 + 1) * 1e-4f;
    for (size_t i = 0; i < x.size(); i++) { int r = rand(); x[i] = r % 997 == 0 ? 0.0f : ((float)(r % 20001) - 10000.f) * 3.1e-4f; }
    uint8_t *dw; float *dsc, *dx, *dxt, *dy0, *dy1;
    const int SB = (S + RS - 1) / RS;
    CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dsc, sn * 4)); CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&dxt, (size_t)SB * RS * I * 4)); CK(cudaMalloc(&dy0, (size_t)S * O * 4)); CK(cudaMalloc(&dy1, (size_t)S * O * 4));
    CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsc, sc.data(), sn * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    float tr = run_ref(dy0, dx, dw, dsc, FMT, S, I, O, reps);
    long long n = (long long)SB * I;
    qmb_xt<RS><<<(unsigned)((n + 255) / 256), 256>>>(dxt, dx, S, I, n);
    dim3 grid(O / (RO * NH), SB);
    qmh_kernel<FMT, I, RS, RO, NH, PROBE><<<grid, 256 * NH>>>(dy1, dxt, dw, dsc, S, O);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) qmh_kernel<FMT, I, RS, RO, NH, PROBE><<<grid, 256 * NH>>>(dy1, dxt, dw, dsc, S, O);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float tk; cudaEventElapsedTime(&tk, a, b); tk /= reps;
    std::vector<float> y0((size_t)S * O), y1((size_t)S * O);
    CK(cudaMemcpy(y0.data(), dy0, y0.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < y0.size(); i++) bad += memcmp(&y0[i], &y1[i], 4) != 0;
    printf("%-26s P%d S=%5d RS=%2d RO=%2d NH=%d: ref %8.3f ms  kernel %8.3f ms (%5.2f TMAC/s)  %s\n", name, PROBE, S, RS, RO, NH, tr, tk,
           (double)S * O * I / tk / 1e9, bad ? "DIFFER" : "same bytes");
    cudaFree(dw); cudaFree(dsc); cudaFree(dx); cudaFree(dxt); cudaFree(dy0); cudaFree(dy1);
}

/* The weights through shared memory: a block's RO rows two steps at a time, each thread
 * one or two 16-byte loads a chunk, double-buffered (the next chunk in registers during
 * this one's steps); the inputs straight from global memory, a step ahead. */
template <int FMT, int I, int RS, int RO>
__global__ static void __launch_bounds__(256, 1) qmw_kernel(float *__restrict__ y, const float *__restrict__ xt,
        const uint8_t *__restrict__ w, const float *__restrict__ scales, int S, int O) {
    constexpr int K = I / 256, RB = FMT == 4 ? I / 2 : I, NG = FMT == 4 ? I / 64 : 1;
    constexpr int RBK = FMT == 4 ? 128 : 256;                    /* a row's bytes a step */
    constexpr int CB = 2 * RO * RBK, PT = CB / (256 * 16);       /* a chunk's bytes, 16-byte pieces a thread */
    constexpr int WSB = 2 * CB, REDB = QMB_P * 256 * 4, RAW = WSB > REDB ? WSB : REDB;
    static_assert(I % 512 == 0 && (RS * RO) % QMB_P == 0 && RS % 4 == 0 && RO % 4 == 0 && PT >= 1 && CB % 4096 == 0, "shape");
    __shared__ __align__(16) float ssc[NG][RO];
    __shared__ __align__(16) uint8_t raw[RAW];
    uint8_t (*ws)[2][RO][RBK] = reinterpret_cast<uint8_t (*)[2][RO][RBK]>(raw);   /* [buffer][step][row][byte] */
    float (*red)[256] = reinterpret_cast<float (*)[256]>(raw);
    const int t = threadIdx.x, o0 = blockIdx.x * RO, sb = blockIdx.y;
    if (FMT == 4)
        for (int e = t; e < NG * RO; e += 256) { const int j = e / NG, g = e % NG; ssc[g][j] = scales[(size_t)(o0 + j) * NG + g]; }
    float acc[RS][RO];
#pragma unroll
    for (int r = 0; r < RS; r++)
#pragma unroll
        for (int j = 0; j < RO; j++) acc[r][j] = 0.0f;
    const float *xp = xt + ((size_t)sb * I + t) * RS;
    const uint8_t *wb = w + (size_t)o0 * RB;
    /* piece e = t + 256 p of a chunk: step e / (RO * RBK / 16), row, 16-byte column */
    auto gload = [&](uint4 (&v)[PT], int c) {
#pragma unroll
        for (int p = 0; p < PT; p++) {
            const int e = t + 256 * p, s = e / (RO * RBK / 16), j = (e / (RBK / 16)) % RO, m = e % (RBK / 16);
            v[p] = __ldg(reinterpret_cast<const uint4 *>(wb + (size_t)j * RB + (size_t)(2 * c + s) * RBK) + m);
        }
    };
    auto sstore = [&](const uint4 (&v)[PT], int buf) {
#pragma unroll
        for (int p = 0; p < PT; p++) {
            const int e = t + 256 * p, s = e / (RO * RBK / 16), j = (e / (RBK / 16)) % RO, m = e % (RBK / 16);
            *reinterpret_cast<uint4 *>(&ws[buf][s][j][16 * m]) = v[p];
        }
    };
    const uint32_t dm = (t & 1) ? 0xF0u : 0x0Fu, dc = (t & 1) ? 0x4B000080u : 0x4B000008u;
    const float dmul = (t & 1) ? 0.0625f : 1.0f, dadd = (t & 1) ? -524296.0f : -8388616.0f;
    const int wcol = FMT == 4 ? (t >> 1) : t;
    uint4 wv4[PT];
    gload(wv4, 0); sstore(wv4, 0);
    gload(wv4, 1);
    float xn[RS];
#pragma unroll
    for (int q = 0; q < RS / 4; q++) {
        const float4 v = __ldg(reinterpret_cast<const float4 *>(xp) + q);
        xn[4 * q] = v.x; xn[4 * q + 1] = v.y; xn[4 * q + 2] = v.z; xn[4 * q + 3] = v.w;
    }
    __syncthreads();
#pragma unroll 1
    for (int c = 0; c < K / 2; c++) {
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const int k = 2 * c + s;
            float xv[RS];
#pragma unroll
            for (int r = 0; r < RS; r++) xv[r] = xn[r];
            if (k + 1 < K) {
#pragma unroll
                for (int q = 0; q < RS / 4; q++) {
                    const float4 v = __ldg(reinterpret_cast<const float4 *>(xp + (size_t)(k + 1) * 256 * RS) + q);
                    xn[4 * q] = v.x; xn[4 * q + 1] = v.y; xn[4 * q + 2] = v.z; xn[4 * q + 3] = v.w;
                }
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
                const uint32_t b = ws[c & 1][s][j][wcol];
                float wvf;
                if (FMT == 4) wvf = __fmaf_rn(__int_as_float((b & dm) ^ dc), dmul, dadd);
                else wvf = __int_as_float(b ^ 0x4B000080u) - 8388736.0f;
#pragma unroll
                for (int r = 0; r < RS; r++) {
                    if (FMT == 4) acc[r][j] = __fmaf_rn(__fmul_rn(xv[r], wvf), sc[j], acc[r][j]);
                    else acc[r][j] = __fmaf_rn(xv[r], wvf, acc[r][j]);
                }
            }
        }
        if (c + 1 < K / 2) sstore(wv4, (c + 1) & 1);
        __syncthreads();
        if (c + 2 < K / 2) gload(wv4, c + 2);
    }
    /* quant_matmul's tree: levels 128, 64, 32 across the warps (h = t >> 5), 16 .. 1 in a warp */
    const int h = t >> 5, cl = t & 31;
#pragma unroll
    for (int p0 = 0; p0 < RS * RO; p0 += QMB_P) {
#pragma unroll
        for (int p = 0; p < QMB_P; p++) red[p][t] = acc[(p0 + p) / RO][(p0 + p) % RO];
        __syncthreads();
#pragma unroll
        for (int p = h; p < QMB_P; p += 8) {
            float v[8];
#pragma unroll
            for (int q = 0; q < 8; q++) v[q] = red[p][32 * q + cl];
            float a0 = v[0] + v[4], a1 = v[1] + v[5], a2 = v[2] + v[6], a3 = v[3] + v[7];
            float b0 = a0 + a2, b1 = a1 + a3;
            float s = b0 + b1;
            s += __shfl_down_sync(0xffffffffu, s, 16);
            s += __shfl_down_sync(0xffffffffu, s, 8);
            s += __shfl_down_sync(0xffffffffu, s, 4);
            s += __shfl_down_sync(0xffffffffu, s, 2);
            s += __shfl_down_sync(0xffffffffu, s, 1);
            const int r = (p0 + p) / RO, j = (p0 + p) % RO, row = sb * RS + r, o = o0 + j;
            if (!cl && row < S) y[(size_t)row * O + o] = FMT == 1 ? s * scales[o] : s;
        }
        __syncthreads();
    }
}

static uint32_t rng = 12345;
static uint32_t nxt(void);
template <int FMT, int I, int RS, int RO>
static void one_w(const char *name, int O, int S, int reps) {
    size_t rb = FMT == 1 ? (size_t)I : (size_t)I / 2;
    size_t sn = FMT == 1 ? (size_t)O : (size_t)O * (I / 64);
    std::vector<uint8_t> w(rb * O); std::vector<float> sc(sn), x((size_t)S * I);
    for (auto &v : w) v = (uint8_t)nxt();
    for (auto &v : sc) v = (float)(nxt() % 1000 + 1) * 1e-4f;
    for (size_t i = 0; i < x.size(); i++) {
        uint32_t r = nxt();
        float v = ((float)(r % 20001) - 10000.f) * 3.1e-4f;
        if (r % 997 == 0) v = 0.0f; else if (r % 991 == 0) v = -0.0f;
        else if (r % 983 == 0) v = 1e-39f * (float)(r % 7);
        else if (r % 977 == 0) v = 3e4f;
        x[i] = v;
    }
    uint8_t *dw; float *dsc, *dx, *dxt, *dy0, *dy1;
    const int SB = (S + RS - 1) / RS;
    CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dsc, sn * 4)); CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&dxt, (size_t)SB * RS * I * 4)); CK(cudaMalloc(&dy0, (size_t)S * O * 4)); CK(cudaMalloc(&dy1, (size_t)S * O * 4));
    CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsc, sc.data(), sn * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dy1, 0xee, (size_t)S * O * 4));
    float tr = run_ref(dy0, dx, dw, dsc, FMT, S, I, O, reps);
    long long n = (long long)SB * I;
    qmb_xt<RS><<<(unsigned)((n + 255) / 256), 256>>>(dxt, dx, S, I, n);
    dim3 grid(O / RO, SB);
    qmw_kernel<FMT, I, RS, RO><<<grid, 256>>>(dy1, dxt, dw, dsc, S, O);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) qmw_kernel<FMT, I, RS, RO><<<grid, 256>>>(dy1, dxt, dw, dsc, S, O);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float tk; cudaEventElapsedTime(&tk, a, b); tk /= reps;
    std::vector<float> y0((size_t)S * O), y1((size_t)S * O);
    CK(cudaMemcpy(y0.data(), dy0, y0.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < y0.size(); i++) bad += memcmp(&y0[i], &y1[i], 4) != 0;
    printf("%-26s W  S=%5d RS=%2d RO=%2d: ref %8.3f ms  kernel %8.3f ms (%5.2f TMAC/s)  %s\n", name, S, RS, RO, tr, tk,
           (double)S * O * I / tk / 1e9, bad ? "DIFFER" : "same bytes");
    cudaFree(dw); cudaFree(dsc); cudaFree(dx); cudaFree(dxt); cudaFree(dy0); cudaFree(dy1);
}

/* ---------------- harness ---------------- */
static uint32_t nxt(void){ rng = rng * 1664525u + 1013904223u; return rng >> 8; }

__global__ void l2bw(const float4 *p, size_t n, int reps, float *out) {
    float4 a = make_float4(0, 0, 0, 0);
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
            float4 v = __ldcg(p + i); a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
        }
    if (a.x + a.y + a.z + a.w == 123.f) out[0] = a.x;
}

template <int FMT, int I, int RS, int RO, int PROBE = 0>
static float run_new(float *dy, float *dxt, const float *dx, const uint8_t *dw, const float *dsc, int S, int O, int reps, bool with_xt = true) {
    const int SB = (S + RS - 1) / RS;
    long long n = (long long)SB * I;
    dim3 grid(O / RO, SB);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    qmb_xt<RS><<<(unsigned)((n + 255) / 256), 256>>>(dxt, dx, S, I, n);
    qmb_kernel<FMT, I, RS, RO, PROBE><<<grid, 256>>>(dy, dxt, dw, dsc, S, O);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) {
        if (with_xt) qmb_xt<RS><<<(unsigned)((n + 255) / 256), 256>>>(dxt, dx, S, I, n);
        qmb_kernel<FMT, I, RS, RO, PROBE><<<grid, 256>>>(dy, dxt, dw, dsc, S, O);
    }
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

static float run_ref(float *dy, const float *dx, const uint8_t *dw, const float *dsc, int fmt, int S, int I, int O, int reps) {
    size_t rb = fmt == 1 ? (size_t)I : (size_t)I / 2;
    int gs = fmt == 4 ? 64 : 0, ng = fmt == 4 ? I / 64 : 1;
    dim3 grid((O + QMT_TO - 1) / QMT_TO, (S + QMT_TS - 1) / QMT_TS);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    quant_matmul_tiled<<<grid, 256>>>(dy, dx, dw, dsc, fmt, S, I, O, rb, gs, ng);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) quant_matmul_tiled<<<grid, 256>>>(dy, dx, dw, dsc, fmt, S, I, O, rb, gs, ng);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

template <int FMT, int I, int RS, int RO, int PROBES = 0>
static void one(const char *name, int O, int S, int reps) {
    size_t rb = FMT == 1 ? (size_t)I : (size_t)I / 2;
    size_t sn = FMT == 1 ? (size_t)O : (size_t)O * (I / 64);
    std::vector<uint8_t> w(rb * O); std::vector<float> sc(sn), x((size_t)S * I);
    for (auto &v : w) v = (uint8_t)nxt();
    for (auto &v : sc) v = (float)(nxt() % 1000 + 1) * 1e-4f;
    for (size_t i = 0; i < x.size(); i++) {
        uint32_t r = nxt();
        float v = ((float)(r % 20001) - 10000.f) * 3.1e-4f;
        if (r % 997 == 0) v = 0.0f; else if (r % 991 == 0) v = -0.0f;
        else if (r % 983 == 0) v = 1e-39f * (float)(r % 7);          /* subnormal inputs */
        else if (r % 977 == 0) v = 3e4f;
        x[i] = v;
    }
    uint8_t *dw; float *dsc, *dx, *dxt, *dy0, *dy1;
    const int SB = (S + RS - 1) / RS;
    CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dsc, sn * 4)); CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&dxt, (size_t)SB * RS * I * 4)); CK(cudaMalloc(&dy0, (size_t)S * O * 4)); CK(cudaMalloc(&dy1, (size_t)S * O * 4));
    CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsc, sc.data(), sn * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dy0, 0xff, (size_t)S * O * 4)); CK(cudaMemset(dy1, 0xee, (size_t)S * O * 4));
    float tr = run_ref(dy0, dx, dw, dsc, FMT, S, I, O, reps);
    float tn = run_new<FMT, I, RS, RO>(dy1, dxt, dx, dw, dsc, S, O, reps);
    float tk = run_new<FMT, I, RS, RO>(dy1, dxt, dx, dw, dsc, S, O, reps, false);
    std::vector<float> y0((size_t)S * O), y1((size_t)S * O);
    CK(cudaMemcpy(y0.data(), dy0, y0.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
    float p1 = PROBES ? run_new<FMT, I, RS, RO, 1>(dy0, dxt, dx, dw, dsc, S, O, reps, false) : 0;
    float p2 = PROBES ? run_new<FMT, I, RS, RO, 2>(dy0, dxt, dx, dw, dsc, S, O, reps, false) : 0;
    size_t bad = 0, first = (size_t)-1;
    for (size_t i = 0; i < y0.size(); i++) if (memcmp(&y0[i], &y1[i], 4)) { if (!bad) first = i; bad++; }
    double macs = (double)S * O * I;
    printf("%-26s S=%5d RS=%2d RO=%2d: ref %8.3f ms (%5.2f TMAC/s)  new %8.3f ms (%5.2f), kernel alone %8.3f  %s",
           name, S, RS, RO, tr, macs / tr / 1e9, tn, macs / tn / 1e9, tk, bad ? "DIFFER" : "same bytes");
    if (bad) printf(" (%zu, first row %zu out %zu: %.9g vs %.9g)", bad, first / O, first % O, y0[first], y1[first]);
    if (PROBES) printf("  [x from one tile %.3f ms, w from one tile %.3f ms]", p1, p2);
    printf("\n");
    cudaFree(dw); cudaFree(dsc); cudaFree(dx); cudaFree(dxt); cudaFree(dy0); cudaFree(dy1);
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    int S = argc > 2 ? atoi(argv[2]) : 2637;
    {   /* L2: 32 MB read over and over */
        size_t n = (32u << 20) / 16; float4 *p; float *o; CK(cudaMalloc(&p, n * 16)); CK(cudaMalloc(&o, 4));
        CK(cudaMemset(p, 0, n * 16));
        l2bw<<<128 * 8, 256>>>(p, n, 2, o); CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a); l2bw<<<128 * 8, 256>>>(p, n, 50, o); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("L2 read: %.2f TB/s\n", 50.0 * n * 16 / ms / 1e9);
        size_t n2 = (1024u << 20) / 16; float4 *p2; CK(cudaMalloc(&p2, n2 * 16)); CK(cudaMemset(p2, 0, n2 * 16));
        cudaEventRecord(a); l2bw<<<128 * 8, 256>>>(p2, n2, 3, o); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
        printf("DRAM read: %.2f TB/s\n", 3.0 * n2 * 16 / ms / 1e9);
        cudaFree(p); cudaFree(p2); cudaFree(o);
    }
    int which = argc > 3 ? atoi(argv[3]) : 0;
    if (which == 0 || which == 1) {
        one<4, 7168, 8, 16, 1>("KDA q int4", 12288, S, reps);
        one<1, 7168, 8, 16, 1>("MLA g int8", 12288, S, reps);
    }
    if (which == 4) {
        one_h<4, 7168, 8, 8, 2>("KDA q/k/v/g int4", 12288, S, reps);
        one_h<4, 12288, 8, 8, 2>("KDA o int4", 7168, S, reps);
        one_h<1, 12288, 8, 8, 2>("MLA o int8", 7168, S, reps);
        one_h<1, 7168, 8, 8, 2>("MLA g int8", 12288, S, reps);
        one_h<4, 7168, 8, 16, 1>("KDA q/k/v/g int4", 12288, S, reps);
        one_h<4, 7168, 4, 16, 2>("KDA q/k/v/g int4", 12288, S, reps);
    }
    if (which == 5) {
        one_h<4, 7168, 8, 16, 1, 0>("KDA q/k/v/g int4", 12288, S, reps);
        one_h<4, 7168, 8, 16, 1, 3>("KDA q x in L1", 12288, S, reps);
        one_h<4, 7168, 8, 16, 1, 4>("KDA q w in L1", 12288, S, reps);
        one_h<1, 7168, 8, 16, 1, 0>("MLA g int8", 12288, S, reps);
        one_h<1, 7168, 8, 16, 1, 3>("MLA g x in L1", 12288, S, reps);
        one_h<1, 7168, 8, 16, 1, 4>("MLA g w in L1", 12288, S, reps);
    }
    if (which == 6) {
        one_w<4, 7168, 8, 16>("KDA q/k/v/g int4", 12288, S, reps);
        one_w<4, 12288, 8, 16>("KDA o int4", 7168, S, reps);
        one_w<1, 12288, 8, 16>("MLA o int8", 7168, S, reps);
        one_w<1, 7168, 8, 16>("MLA g int8", 12288, S, reps);
        one_w<4, 7168, 8, 16>("KDA q/k/v/g int4", 12288, 37, reps);
        one_w<4, 7168, 8, 16>("KDA q/k/v/g int4", 12288, 1, reps);
    }
    if (which == 3) {
        one<4, 7168, 8, 16>("KDA q/k/v/g int4", 12288, S, reps);
        one<4, 12288, 8, 16>("KDA o int4", 7168, S, reps);
        one<1, 12288, 8, 16>("MLA o int8", 7168, S, reps);
        one<1, 7168, 8, 16>("MLA g int8", 12288, S, reps);
        one<4, 7168, 8, 16>("KDA q/k/v/g int4", 12288, 37, reps);
        one<4, 7168, 8, 16>("KDA q/k/v/g int4", 12288, 1, reps);
    }
    if (which == 0 || which == 2) {
        one<4, 7168, 4, 32>("KDA q int4", 12288, S, reps);
        one<4, 7168, 4, 16>("KDA q int4", 12288, S, reps);
        one<4, 7168, 8, 8>("KDA q int4", 12288, S, reps);
        one<4, 7168, 16, 8>("KDA q int4", 12288, S, reps);
        one<4, 7168, 12, 8>("KDA q int4", 12288, S, reps);
        one<4, 7168, 8, 12>("KDA q int4", 12288, S, reps);
    }
    return 0;
}
