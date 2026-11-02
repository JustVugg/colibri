// bench_k3d.cu: coli_cuda_k3.dll's k3_dense (verbatim, fmt 4) against k3_dense_b, the same CPU
// arithmetic (matmul_i4_grouped: 8 lanes of fused adds a group of 64, hsum256, fmaf a group) with
// a thread's 8 lanes in its own registers, at Kimi K3's non-resident shapes. nvcc -O3 -arch=sm_89
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);}}while(0)

/* ---------------- the reference: backend_cuda_k3.cu's k3_dense ---------------- */
#define K3_TP 8
#define K3_TO 32
#define K3_KC 512
__device__ __forceinline__ float k3d_hsum8(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}
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

/* ---------------- new: a thread's 8 lanes in its registers ---------------- */
/* Thread (tr, to) of the block: rows r0 = 4 tr .. +3 of the block's 32, outputs 6 to .. +5 of its
 * 192. A group of 64 columns: lane l's chain is columns l, l+8, .., l+56, fused in that order
 * from +0, the lanes taken 0, 4, 2, 6, 1, 5, 3, 7 so that each finished one joins hsum256's
 * tree at once ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)) with three kept a pair; then the group's
 * fmaf into the total. Lane l's weight m of output o is nibble l of the group's word m (the
 * CPU's offset nibbles): (w & 15 << 4p) | 2^23 read as a float, one fused add takes the place
 * p and the 8 out. The block's rows of x by column in shared memory, two groups a stage. */
#define KB_RS 4
#define KB_RO 6
#define KB_BR 32
#define KB_BO 192
#define KB_SC 128                   /* columns a stage */
#define KB_PAD 36                   /* a column's 32 rows and 4 spare: 16-byte stores without conflicts */
template <int P>
__device__ __forceinline__ float kb_nib(uint32_t w) {   /* the offset nibble at place P (0..4) of w, less 8 */
    const float mul = P == 0 ? 1.0f : P == 1 ? 0.0625f : P == 2 ? 0.00390625f : P == 3 ? 0.000244140625f : 1.52587890625e-05f;
    const float add = P == 0 ? -8388616.0f : P == 1 ? -524296.0f : P == 2 ? -32776.0f : P == 3 ? -2056.0f : -136.0f;
    return __fmaf_rn(__uint_as_float((w & (15u << (4 * P))) | 0x4B000000u), mul, add);
}
template <int L>
__device__ __forceinline__ float kb_lane_w(uint32_t w) {   /* lane L's nibble of a word */
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
    /* staging: thread t loads column t & 127 of rows 16 (t >> 7) .. +15 */
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
            __syncthreads();                 /* every thread done with buffer b ^ 1's last stage */
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

/* ---------------- harness ---------------- */
static uint32_t rng = 777;
static uint32_t nxt(void){ rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static void one(const char *name, int O, int I, int S, int reps) {
    const int gs = 64, ng = I / gs;
    std::vector<uint8_t> w((size_t)O * I / 2); std::vector<float> sc((size_t)O * ng), x((size_t)S * I);
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
    uint8_t *dw; float *dsc, *dx, *dy0, *dy1;
    CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dsc, sc.size() * 4)); CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&dy0, (size_t)S * O * 4)); CK(cudaMalloc(&dy1, (size_t)S * O * 4));
    CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dy1, 0xee, (size_t)S * O * 4));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    dim3 g0((O + K3_TO - 1) / K3_TO, (S + K3_TP - 1) / K3_TP), g1((O + KB_BO - 1) / KB_BO, (S + KB_BR - 1) / KB_BR);
    k3_dense<<<g0, 256>>>(4, dw, dsc, dx, S, I, O, gs, dy0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for (int i = 0; i < reps; i++) k3_dense<<<g0, 256>>>(4, dw, dsc, dx, S, I, O, gs, dy0);
    cudaEventRecord(b); CK(cudaEventSynchronize(b)); float t0; cudaEventElapsedTime(&t0, a, b); t0 /= reps;
    k3_dense_b<<<g1, 256>>>(dw, dsc, dx, S, I, O, dy1); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for (int i = 0; i < reps; i++) k3_dense_b<<<g1, 256>>>(dw, dsc, dx, S, I, O, dy1);
    cudaEventRecord(b); CK(cudaEventSynchronize(b)); float t1; cudaEventElapsedTime(&t1, a, b); t1 /= reps;
    std::vector<float> y0((size_t)S * O), y1((size_t)S * O);
    CK(cudaMemcpy(y0.data(), dy0, y0.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < y0.size(); i++) if (memcmp(&y0[i], &y1[i], 4)) { if (!bad) first = i; bad++; }
    const double macs = (double)S * O * I;
    printf("%-24s O=%5d I=%5d S=%5d: k3_dense %8.3f ms (%5.2f TMAC/s)  k3_dense_b %8.3f ms (%5.2f)  %s", name, O, I, S,
           t0, macs / t0 / 1e9, t1, macs / t1 / 1e9, bad ? "DIFFER" : "same bytes");
    if (bad) printf(" (%zu, first row %zu out %zu: %.9g vs %.9g)", bad, first / O, first % O, y0[first], y1[first]);
    printf("\n");
    cudaFree(dw); cudaFree(dsc); cudaFree(dx); cudaFree(dy0); cudaFree(dy1);
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    if (argc > 2) {
        for (int S : {32, 48, 64, 96, 128, 192, 256, 512}) one("shared gate/up", 6144, 7168, S, reps);
        for (int S : {32, 64, 128, 256}) one("latent down", 3584, 7168, S, reps);
        return 0;
    }
    one("shared gate/up", 6144, 7168, 1024, reps);
    one("shared down", 7168, 6144, 1024, reps);
    one("latent down", 3584, 7168, 1024, reps);
    one("latent up", 7168, 3584, 1024, reps);
    one("KDA q (non-resident)", 12288, 7168, 1024, reps);
    one("odd rows and outputs", 200, 7168, 77, reps);
    one("one row", 6144, 7168, 1, reps);
    return 0;
}
