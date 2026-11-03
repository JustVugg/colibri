// bench_q.cu: a tile kernel with the weights decoded once a block into shared memory (a quad of
// threads an 8x8 tile, two of the CPU's 8 lanes each) against k3_dense_b / k3_dense_b8: the same
// bytes?, T multiply-adds a second at Kimi K3's shapes.
//   nvcc -O3 -std=c++17 -arch=sm_89 -ftz=false -I<c> bench_q.cu -o bench_q.exe
// Run against 254338ab's backend_cuda_k3.cu, whose k3_dense_launch then took k3_dense_b and k3_dense_b8 (the
// "old" column); since, k3_dense_launch takes k3_dense_s4 and k3_dense_o8 (bench_tiles.cu keeps the old ones).
#include "backend_cuda_k3.cu"
#include <vector>

#define CK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);}}while(0)

__host__ __device__ constexpr int kq_brev3(int l) { return ((l & 1) << 2) | (l & 2) | ((l >> 2) & 1); }

#define KQ_BR 32                    // rows a block
#define KQ_BO 128                   // outputs a block
#define KQ_SC 64                    // columns a stage (int4: a group)
#define KQ_SMEM ((2 * KQ_SC * KQ_BR + 2 * KQ_SC * KQ_BO + 2 * KQ_BO) * 4)

#ifndef KQ_ROWFAST
#define KQ_ROWFAST 1
#endif
#ifndef KQ_EXP
#define KQ_EXP 0     // timing experiments (wrong outputs): 1 no staging in the loop, 2 no barrier, 4 no int4 tail, 8 operands loaded once a lane
#endif

template <int FMT, int F>
__global__ void __launch_bounds__(256, 1) k3_dense_q(const uint8_t *__restrict__ w, const float *__restrict__ scale,
                                                    const float *__restrict__ x, int S, int I, int O, float *__restrict__ y) {
    extern __shared__ __align__(16) float kq_sm[];
    float *const sx = kq_sm;                          // [2][KQ_SC][KQ_BR]: a column's rows, row quads swizzled
    float *const sw = kq_sm + 2 * KQ_SC * KQ_BR;      // [2][KQ_SC][KQ_BO]: a column's outputs
    float *const ss = sw + 2 * KQ_SC * KQ_BO;         // [2][KQ_BO]: the stage's group scales (int4)
    const int t = threadIdx.x, wp = t >> 5, ln = t & 31;
    const int b0 = ln & 1, b1 = (ln >> 1) & 1, ct = (ln >> 2) & 3, rt = ln >> 4, wr = wp >> 2, wc = wp & 3;
#if KQ_ROWFAST
    const int s0 = blockIdx.x * KQ_BR, o0 = blockIdx.y * KQ_BO;
#else
    const int s0 = blockIdx.y * KQ_BR, o0 = blockIdx.x * KQ_BO;
#endif
    const int nst = I / KQ_SC;
    const int LA = 2 * b0 + b1, SA = kq_brev3(LA), SB = SA | 1;
    const int iA = 4 * wr + 2 * rt + b0, fA = 8 * wc + 2 * ct + b1;
    const int cxa0 = LA * KQ_BR + 4 * (iA ^ SA), cxa1 = LA * KQ_BR + 4 * (iA ^ 1 ^ SA);
    const int cxb0 = (LA + 4) * KQ_BR + 4 * (iA ^ SB), cxb1 = (LA + 4) * KQ_BR + 4 * (iA ^ 1 ^ SB);
    const int cwa0 = LA * KQ_BO + 4 * fA, cwa1 = LA * KQ_BO + 4 * (fA ^ 1);
    const int cwb0 = cwa0 + 4 * KQ_BO, cwb1 = cwa1 + 4 * KQ_BO;
    // staging: x column xc of rows 8 xq .. +7; the weights of outputs 4u .. 4u+3, their word (int8: 8 bytes) m
    const int xc = t & 63, xq = t >> 6, xS = kq_brev3(xc & 7);
    const int sxo0 = xc * KQ_BR + 4 * ((2 * xq) ^ xS), sxo1 = xc * KQ_BR + 4 * ((2 * xq + 1) ^ xS);
    int xo[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { int r = s0 + 8 * xq + i; r = r < S ? r : S - 1; xo[i] = r * I + xc; }
    const int u = 8 * (wp & 3) + (ln & 7), m = 4 * (wp >> 2) + (ln >> 3);
    const size_t rb = FMT == 4 ? (size_t)I / 2 : (size_t)I;
    const uint8_t *wq[4];
#pragma unroll
    for (int k = 0; k < 4; k++) { int o = o0 + 4 * u + k; o = o < O ? o : O - 1; wq[k] = w + (size_t)o * rb + (FMT == 4 ? 4 * m : 8 * m); }
    const int swo = 8 * m * KQ_BO + 4 * u;
    const int ng = I / 64;
    int so = 0;
    if (t < KQ_BO) { int o = o0 + t; o = o < O ? o : O - 1; so = o * ng; }
    float xv[8], sv = 0.0f;
    uint32_t wv[FMT == 4 ? 4 : 8];
    auto load = [&](int st) {
#pragma unroll
        for (int i = 0; i < 8; i++) xv[i] = __ldg(x + xo[i] + st * KQ_SC);
        if (FMT == 4) {
#pragma unroll
            for (int k = 0; k < 4; k++) wv[k] = __ldg(reinterpret_cast<const uint32_t *>(wq[k] + st * 32));
            if (t < KQ_BO) sv = __ldg(scale + so + st);
        } else {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint2 p = __ldg(reinterpret_cast<const uint2 *>(wq[k] + (size_t)st * 64));
                wv[2 * k] = p.x; wv[2 * k + 1] = p.y;
            }
        }
    };
    auto store = [&](int b) {
        float *bx = sx + b * KQ_SC * KQ_BR, *bw = sw + b * KQ_SC * KQ_BO;
        *reinterpret_cast<float4 *>(bx + sxo0) = make_float4(xv[0], xv[1], xv[2], xv[3]);
        *reinterpret_cast<float4 *>(bx + sxo1) = make_float4(xv[4], xv[5], xv[6], xv[7]);
#pragma unroll
        for (int l = 0; l < 8; l++) {
            float v[4];
#pragma unroll
            for (int k = 0; k < 4; k++)
                v[k] = FMT == 4 ? kb_lane_v(F ? wv[k] ^ 0x88888888u : wv[k], l) : k8_val(wv[2 * k + (l >> 2)] ^ 0x80808080u, l & 3);
            *reinterpret_cast<float4 *>(bw + swo + l * KQ_BO) = make_float4(v[0], v[1], v[2], v[3]);
        }
        if (FMT == 4 && t < KQ_BO) ss[b * KQ_BO + t] = sv;
    };
    float tot[4][4], acA[8][8], acB[8][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) tot[i][j] = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) { acA[i][j] = 0.0f; acB[i][j] = 0.0f; }
    load(0); store(0);
    if (nst > 1) load(1);
    __syncthreads();
#pragma unroll 1
    for (int st = 0; st < nst; st++) {
        const float *bx = sx + (st & 1) * KQ_SC * KQ_BR, *bw = sw + (st & 1) * KQ_SC * KQ_BO;
        if (FMT == 4) {
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) { acA[i][j] = 0.0f; acB[i][j] = 0.0f; }
        }
#pragma unroll
        for (int mm = 0; mm < 8; mm++) {
            const int mo = (KQ_EXP & 8) ? 0 : mm;
            const float4 x0 = *reinterpret_cast<const float4 *>(bx + cxa0 + mo * 8 * KQ_BR);
            const float4 x1 = *reinterpret_cast<const float4 *>(bx + cxa1 + mo * 8 * KQ_BR);
            const float4 w0 = *reinterpret_cast<const float4 *>(bw + cwa0 + mo * 8 * KQ_BO);
            const float4 w1 = *reinterpret_cast<const float4 *>(bw + cwa1 + mo * 8 * KQ_BO);
            const float xr[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const float wr8[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acA[i][j] = __fmaf_rn(xr[i], wr8[j], acA[i][j]);
        }
#pragma unroll
        for (int mm = 0; mm < 8; mm++) {
            const int mo = (KQ_EXP & 8) ? 0 : mm;
            const float4 x0 = *reinterpret_cast<const float4 *>(bx + cxb0 + mo * 8 * KQ_BR);
            const float4 x1 = *reinterpret_cast<const float4 *>(bx + cxb1 + mo * 8 * KQ_BR);
            const float4 w0 = *reinterpret_cast<const float4 *>(bw + cwb0 + mo * 8 * KQ_BO);
            const float4 w1 = *reinterpret_cast<const float4 *>(bw + cwb1 + mo * 8 * KQ_BO);
            const float xr[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const float wr8[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acB[i][j] = __fmaf_rn(xr[i], wr8[j], acB[i][j]);
        }
        if (FMT == 4 && !(KQ_EXP & 4)) {
            // the group's tree: the pair in the thread, (a0+a4) etc.; rows 4..7 to the partner xor 1, then
            // outputs 4..7 of the rest to the partner xor 2; then its scale, fused into the total
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acA[i][j] = __fadd_rn(acA[i][j], acB[i][j]);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acA[i][j] = __fadd_rn(acA[i][j], __shfl_xor_sync(0xffffffffu, acA[i + 4][j], 1));
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acA[i][j] = __fadd_rn(acA[i][j], __shfl_xor_sync(0xffffffffu, acA[i][j + 4], 2));
            const float4 g4 = *reinterpret_cast<const float4 *>(ss + (st & 1) * KQ_BO + 4 * fA);
            const float g[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) tot[i][j] = __fmaf_rn(acA[i][j], g[j], tot[i][j]);
        }
        if (st + 1 < nst) {
            if (!(KQ_EXP & 1)) store((st + 1) & 1);
            if (!(KQ_EXP & 2)) __syncthreads();
            if (!(KQ_EXP & 1) && st + 2 < nst) load(st + 2);
        }
    }
    if (FMT == 1) {
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) acA[i][j] = __fadd_rn(acA[i][j], acB[i][j]);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) acA[i][j] = __fadd_rn(acA[i][j], __shfl_xor_sync(0xffffffffu, acA[i + 4][j], 1));
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acA[i][j] = __fadd_rn(acA[i][j], __shfl_xor_sync(0xffffffffu, acA[i][j + 4], 2));
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int o = o0 + 4 * fA + j; o = o < O ? o : O - 1;
            const float sc = __ldg(scale + o);
#pragma unroll
            for (int i = 0; i < 4; i++) tot[i][j] = __fmul_rn(acA[i][j], sc);
        }
    }
    const int oo = o0 + 4 * fA;
    if (oo < O) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = s0 + 16 * wr + 8 * rt + 4 * b0 + i;
            if (row < S) *reinterpret_cast<float4 *>(y + (size_t)row * O + oo) = make_float4(tot[i][0], tot[i][1], tot[i][2], tot[i][3]);
        }
    }
}

template <int FMT, int F>
static void kq_launch(const uint8_t *w, const float *s, const float *x, int S, int I, int O, float *y) {
    static int set = 0;
    if (!set) { CK(cudaFuncSetAttribute(k3_dense_q<FMT, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, KQ_SMEM)); set = 1; }
    const unsigned rb = (unsigned)((S + KQ_BR - 1) / KQ_BR), ob = (unsigned)((O + KQ_BO - 1) / KQ_BO);
#if KQ_ROWFAST
    k3_dense_q<FMT, F><<<dim3(rb, ob), 256, KQ_SMEM>>>(w, s, x, S, I, O, y);
#else
    k3_dense_q<FMT, F><<<dim3(ob, rb), 256, KQ_SMEM>>>(w, s, x, S, I, O, y);
#endif
}

static float timeit(int which, int fmt, int flip, const uint8_t *w, const float *s, const float *x, int S, int I, int O, float *y, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto go = [&]() {
        if (which == 0) k3_dense_launch(fmt, w, s, x, S, I, O, 64, y, 0, flip);
        else if (fmt == 4) { if (flip) kq_launch<4, 1>(w, s, x, S, I, O, y); else kq_launch<4, 0>(w, s, x, S, I, O, y); }
        else kq_launch<1, 0>(w, s, x, S, I, O, y);
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
static float rx(int kind) {   // kind 0: ordinary; 1: some zeros of both signs, subnormals, large values
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
    const int reps = argc > 1 ? atoi(argv[1]) : 10;
    struct { const char *name; int O, I, fmt; } sh[] = {
        {"KDA q/shared gate 12288x7168 i4", 12288, 7168, 4}, {"KDA o/shared down 7168x12288 i4", 7168, 12288, 4},
        {"latent down 3584x7168 i4", 3584, 7168, 4}, {"latent up 7168x3584 i4", 7168, 3584, 4},
        {"MLA g 12288x7168 i8", 12288, 7168, 1}, {"MLA o 7168x12288 i8", 7168, 12288, 1},
        {"MLA qb 18432x1536 i8", 18432, 1536, 1}, {"MLA qa 1536x7168 i8", 1536, 7168, 1},
        {"MLA kva 576x7168 i8", 576, 7168, 1}};
    const int quick = argc > 2;   // the first i4 and i8 shapes at 1,024 rows, the new kernel alone
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
    printf("outputs differing: %lld\n", bad_total);
    return bad_total != 0;
}
