// bench_tiles.cu: coli_cuda_k3.dll's tile kernels as backend_cuda_k3.cu has them now (k3_dense_s4, k3_dense_o8:
// a stage's weights decoded once a block into shared memory) against their predecessors k3_dense_b and
// k3_dense_b8 (a thread's own weights decoded from global memory; copied below verbatim from 254338ab), the
// kernels alone at Kimi K3's shapes and row counts: every output the same bytes, T multiply-adds a second
// (the best of 3 runs of `reps` launches, the two kernels alternated). Both on int4 copies as coli_cuda.dll
// leaves them (nibbles flipped) and as the host has them.
//   nvcc -O3 -std=c++17 -arch=sm_89 -ftz=false -I<c> bench_tiles.cu -o bench_tiles.exe && bench_tiles.exe [reps]
#include "backend_cuda_k3.cu"
#include <vector>

#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);}}while(0)

// ---- 254338ab's k3_dense_b and k3_dense_b8, verbatim (kb_nib, kb_lane_w and k8_val are backend_cuda_k3.cu's) ----
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
// F: coli_cuda.dll's copy (its nibbles flipped back as the words come in).
#define KB_RS 4
#define KB_RO 6
#define KB_BR 32                    // rows a block
#define KB_BO 192                   // outputs a block
#define KB_SC 128                   // columns a stage
#define KB_PAD 36                   // a column's 32 rows and 4 spare: 16-byte stores without conflicts
#define KB_MINROWS 64               // fewer rows: k3_dense (too few blocks)
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
template <int F>
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
                if (F)
#pragma unroll
                    for (int m = 0; m < 8; m++) W[j][m] ^= 0x88888888u;
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
// matmul_q's arithmetic for several rows: output o's lane l sums columns l, l+8, l+16, .. of the whole
// row, fused from +0; then hsum256's tree ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)) times the row's scale.
// Thread (tr, to): rows 4 tr .. +3 of the block's 32 by outputs 4 to .. +3 of its 128, the 8 lanes of
// each in its own registers (128 sums); 16 weights a load, each made a float by one byte permute and
// one add: (b ^ 0x80) under 2^23's exponent, less 2^23 + 128. x staged as k3_dense_b stages it.
#define OB8_RO 4
#define OB8_BO 128                   // outputs a block
__global__ void __launch_bounds__(256, 1) k3_dense_b8(const int8_t *__restrict__ w, const float *__restrict__ scale,
                                                      const float *__restrict__ x, int S, int I, int O,
                                                      float *__restrict__ y) {
    __shared__ __align__(16) float sx[2][KB_SC][KB_PAD];
    const int t = threadIdx.x, tr = t & 7, to = t >> 3;
    const int s0 = blockIdx.y * KB_BR, ob = blockIdx.x * OB8_BO + to * OB8_RO, r0 = tr * KB_RS;
    float a[8][KB_RS][OB8_RO];
#pragma unroll
    for (int l = 0; l < 8; l++)
#pragma unroll
        for (int r = 0; r < KB_RS; r++)
#pragma unroll
            for (int j = 0; j < OB8_RO; j++) a[l][r][j] = 0.0f;
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
    const int8_t *wrow[OB8_RO];
#pragma unroll
    for (int j = 0; j < OB8_RO; j++) { const int o = ob + j < O ? ob + j : O - 1; wrow[j] = w + (size_t)o * I; }
    for (int st = 0; st < nst; st++) {
        const float (*sxc)[KB_PAD] = sx[st & 1];
#pragma unroll 1
        for (int c = 0; c < KB_SC; c += 16) {
            uint32_t W[OB8_RO][4];
#pragma unroll
            for (int j = 0; j < OB8_RO; j++) {
                const uint4 u = __ldg(reinterpret_cast<const uint4 *>(wrow[j] + (size_t)st * KB_SC + c));
                W[j][0] = u.x ^ 0x80808080u; W[j][1] = u.y ^ 0x80808080u; W[j][2] = u.z ^ 0x80808080u; W[j][3] = u.w ^ 0x80808080u;
            }
#pragma unroll
            for (int k = 0; k < 16; k++) {   // column c + k, lane k & 7
                const float4 xv = *reinterpret_cast<const float4 *>(&sxc[c + k][r0]);
                const float xr[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int j = 0; j < OB8_RO; j++) {
                    const float wv = k8_val(W[j][k >> 2], k & 3);
#pragma unroll
                    for (int r = 0; r < KB_RS; r++) a[k & 7][r][j] = __fmaf_rn(xr[r], wv, a[k & 7][r][j]);
                }
            }
        }
        if (st + 1 < nst) {
            __syncthreads();
            xstore((st & 1) ^ 1);
            __syncthreads();
            if (st + 2 < nst) xload((st + 2) * KB_SC);
        }
    }
#pragma unroll
    for (int j = 0; j < OB8_RO; j++) {
        const int o = ob + j;
        const float sc = __ldg(scale + (o < O ? o : O - 1));
#pragma unroll
        for (int r = 0; r < KB_RS; r++) {
            const int row = s0 + r0 + r;
            const float v = __fadd_rn(__fadd_rn(__fadd_rn(a[0][r][j], a[4][r][j]), __fadd_rn(a[2][r][j], a[6][r][j])),
                                      __fadd_rn(__fadd_rn(a[1][r][j], a[5][r][j]), __fadd_rn(a[3][r][j], a[7][r][j])));
            if (row < S && o < O) y[(size_t)row * O + o] = __fmul_rn(v, sc);
        }
    }
}

static void old_launch(int fmt, int flip, const uint8_t *w, const float *s, const float *x, int r, int I, int O, float *y) {
    const unsigned rb = (unsigned)((r + KB_BR - 1) / KB_BR);
    if (fmt == 1) k3_dense_b8<<<dim3((unsigned)((O + OB8_BO - 1) / OB8_BO), rb), 256>>>(reinterpret_cast<const int8_t *>(w), s, x, r, I, O, y);
    else if (flip) k3_dense_b<1><<<dim3((unsigned)((O + KB_BO - 1) / KB_BO), rb), 256>>>(w, s, x, r, I, O, y);
    else k3_dense_b<0><<<dim3((unsigned)((O + KB_BO - 1) / KB_BO), rb), 256>>>(w, s, x, r, I, O, y);
}

static float timeit(int which, int fmt, int flip, const uint8_t *w, const float *s, const float *x, int S, int I, int O, float *y, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto go = [&]() {
        if (which == 0) old_launch(fmt, flip, w, s, x, S, I, O, y);
        else k3_dense_launch(fmt, w, s, x, S, I, O, 64, y, 0, flip);
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
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    struct { const char *name; int O, I, fmt, flip; } sh[] = {
        {"KDA q/k/v/g 12288x7168 i4 (card)", 12288, 7168, 4, 1}, {"KDA o 7168x12288 i4 (card)", 7168, 12288, 4, 1},
        {"shared gate/up 12288x7168 i4", 12288, 7168, 4, 0}, {"shared down 7168x12288 i4", 7168, 12288, 4, 0},
        {"latent down 3584x7168 i4", 3584, 7168, 4, 0}, {"latent up 7168x3584 i4", 7168, 3584, 4, 0},
        {"MLA gate 12288x7168 i8", 12288, 7168, 1, 0}, {"MLA o 7168x12288 i8", 7168, 12288, 1, 0},
        {"MLA qb 18432x1536 i8", 18432, 1536, 1, 0}, {"MLA qa 1536x7168 i8", 1536, 7168, 1, 0},
        {"MLA kva 576x7168 i8", 576, 7168, 1, 0}};
    const int rows[] = {64, 230, 256, 512, 1024};
    CK(cudaSetDevice(0));
    long long bad_total = 0;
    for (auto &h : sh) {
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
        for (int S : rows) {
            const double mac = (double)S * h.O * h.I;
            CK(cudaMemset(dy0, 0xff, (size_t)S * h.O * 4)); CK(cudaMemset(dy1, 0xee, (size_t)S * h.O * 4));
            float t0 = 1e30f, t1 = 1e30f;
            for (int k = 0; k < 3; k++) {
                const float a = timeit(0, h.fmt, h.flip, dw, ds, dx, S, h.I, h.O, dy0, reps); if (a < t0) t0 = a;
                const float b = timeit(1, h.fmt, h.flip, dw, ds, dx, S, h.I, h.O, dy1, reps); if (b < t1) t1 = b;
            }
            std::vector<uint32_t> a((size_t)S * h.O), b((size_t)S * h.O);
            CK(cudaMemcpy(a.data(), dy0, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), dy1, b.size() * 4, cudaMemcpyDeviceToHost));
            long long bad = 0;
            for (size_t i = 0; i < a.size(); i++) if (a[i] != b[i]) bad++;
            bad_total += bad;
            printf(" | %4d: %4.1f -> %4.1f%s", S, mac / (t0 * 1e-3) / 1e12, mac / (t1 * 1e-3) / 1e12, bad ? " BAD" : "");
        }
        printf("  T/s\n");
        cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy0); cudaFree(dy1);
    }
    printf("outputs differing between the old kernels and the new: %lld\n", bad_total);
    return bad_total != 0;
}
