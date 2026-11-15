/* A GGUF's block formats on the CUDA device (cuda_bd.cuh): every ggml block type the
 * backend takes, random blocks with sane scale fields, against the CPU's decoders
 * (ggml_quants.h through tests/cuda_bd_ref.c):
 *   - the integer form's types against the CPU's integer form (the same codes, scales and
 *     integer sums, in double), the float form's against the CPU's dequantized rows;
 *   - a verify's 2, 3 and 4 rows, sharing every weight load, each with the bits of the
 *     same row alone (a decode step's);
 *   - the routed experts of the dense chain (coli_cuda_moe_async, CcMoe.fmt >= 100) against
 *     the CPU's integer form, a row and a verify's; with some of them in page-locked host
 *     memory (a host pool), read over the bus or staged, the same bits;
 *   - the backend's path: coli_cuda_tensor_upload of the file's rows (padded to words on the
 *     device where a row is not), the dense chain's matmul, the same bits as the kernel's;
 *   - a SwiGLU's gate and up of a decode step's rows in one pass (bd_glu_launch), the bits of the
 *     two GEMVs and the elementwise product;
 *   - the tensor cores' integer steps (bd_mmg_launch, sm_80 on) of 1 to 8 rows, each row the bits
 *     of the GEMV's same row alone; again with x's quantization kept (g_bd_x_same), and after
 *     another x at the same address without it.
 * BENCH=1 also times the GEMV at a few shapes, 1 to 4 rows. Build from c/:
 *   gcc -O2 -c tests/cuda_bd_ref.c -o tests/cuda_bd_ref.o
 *   nvcc -O2 -std=c++17 -arch=native tests/test_cuda_bd.cu tests/cuda_bd_ref.o -o tests/test_cuda_bd */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include "../backend_cuda.cu"

extern "C" {
int ref_unpack(int t, const uint8_t *row, int n, int8_t *w8, float *sw, float *mw);
void ref_quant(const float *x, int n, int8_t *xq, float *dx, int32_t *xs);
int ref_dequant(int t, const uint8_t *row, float *out, int n);
long ref_row_bytes(int t, int n);
int ref_block_bytes(int t);
int ref_block_elems(int t);
const char *ref_name(int t);
}

static int fails;
static unsigned long long g_rng = 0x9e3779b97f4a7c15ull;
static unsigned rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (unsigned)(g_rng >> 11); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* sane scale fields in a random block (test_vk_chain.c's bd_fix, with the float-form types) */
static void bd_fix(int t, uint8_t *b) {
#define F16AT(o) do { b[o] = 0x00; b[(o) + 1] = (uint8_t)(0x24 + rnd() % 3); } while (0)
    switch (t) {
    case 12: case 13: F16AT(0); F16AT(2); break;   /* Q4_K, Q5_K */
    case 3: F16AT(0); F16AT(2); break;             /* Q4_1 */
    /* Q5_1: its min -16 d, as a quantizer sets it for a block about zero (its least weight). A random
     * positive one made every weight of a block positive, and 40 rows of {Q5_1, Q5_1} through the f16
     * tiles came 0.0195 off (0.0121 with the min's sign alone), through hcg's tiles and hcm's alike */
    case 7: F16AT(0); b[2] = 0x00; b[3] = (uint8_t)(0x80 | (b[1] + 0x10)); break;
    case 10: F16AT(80); F16AT(82); break;          /* Q2_K */
    case 11: F16AT(108); break;                    /* Q3_K */
    case 14: F16AT(208); break;                    /* Q6_K */
    /* F16, BF16: b is one value here, of either sign, magnitudes 2^-9 .. 2^-1 */
    case 1: b[1] = (uint8_t)((rnd() & 0x80) | (0x18 + rnd() % 20)); break;
    case 30: { int e = 118 + (int)(rnd() % 8); b[1] = (uint8_t)((rnd() & 0x80) | (e >> 1)); b[0] = (uint8_t)(((e & 1) << 7) | (b[0] & 0x7f)); break; }
    case 34: F16AT(52); break;                     /* TQ1_0 */
    case 35: F16AT(64); break;                     /* TQ2_0 */
    case 39: b[0] = (uint8_t)(122 + rnd() % 4); break;   /* MXFP4: e8m0 */
    case 40: break;                                /* NVFP4: ue4m3 scales, any byte finite */
    case 29: {   /* IQ1_M: d is the scales' four top nibbles: 0x24n0, n of 0..2 */
        int n1 = (int)(rnd() % 3);
        b[49] = (uint8_t)(b[49] & 0x0F); b[51] = (uint8_t)((b[51] & 0x0F) | (n1 << 4));
        b[53] = (uint8_t)((b[53] & 0x0F) | (4 << 4)); b[55] = (uint8_t)((b[55] & 0x0F) | (2 << 4));
        break; }
    default: F16AT(0); break;
    }
#undef F16AT
}

static bd_u32 *g_tab;
static size_t g_smem;
static int g_glu;   /* types and shapes whose gate and up went through bd_glu_launch */
/* silu(g) * u as CC_EW_SWIGLU (cuda_chain.cu's cc_ew_k) has it */
__global__ static void swiglu_ref_k(float *y, const float *g, const float *u, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) { float a = g[i]; y[i] = (a / (1.f + expf(-a))) * u[i]; }
}

/* type t at I inputs, O rows: the rows against the CPU, a verify's rows against the rows alone */
static void check(int t, int I, int O, int bench) {
    long rb = ref_row_bytes(t, I);
    int bb = ref_block_bytes(t), be = ref_block_elems(t);
    if (rb <= 0 || I % be) return;
    int rw = (int)((rb + 3) / 4), fmt = 100 + t;
    uint8_t *w = (uint8_t *)calloc((size_t)O, (size_t)rw * 4);
    for (int o = 0; o < O; o++) {
        uint8_t *r = w + (size_t)o * rw * 4;
        for (long i = 0; i < rb; i++) r[i] = (uint8_t)rnd();
        for (long k = 0; k + bb <= rb; k += bb) bd_fix(t, r + k);
    }
    const int SM = 4;
    float *x = (float *)malloc((size_t)SM * I * 4);
    for (int i = 0; i < SM * I; i++) x[i] = ((int)(rnd() % 20001) - 10000) * 1e-4f;
    bd_u32 *wd; float *xd, *yd;
    cudaMalloc(&wd, (size_t)O * rw * 4); cudaMalloc(&xd, (size_t)SM * I * 4); cudaMalloc(&yd, (size_t)SM * O * 4);
    cudaMemcpy(wd, w, (size_t)O * rw * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(xd, x, (size_t)SM * I * 4, cudaMemcpyHostToDevice);
    float *one = (float *)malloc((size_t)SM * O * 4), *got = (float *)malloc((size_t)SM * O * 4);
    int ok = 1;
    for (int s = 0; s < SM; s++) ok &= bd_gemv_launch(yd + (size_t)s * O, xd + (size_t)s * I, wd, g_tab, fmt, I, O, rw, 1, g_smem, 0, 0);
    ok &= cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(one, yd, (size_t)SM * O * 4, cudaMemcpyDeviceToHost);
    if (!ok) { printf("FAIL %s [%d x %d]: the launch failed (%s)\n", ref_name(t), O, I, cudaGetErrorString(cudaGetLastError())); fails++; goto out; }
    {
        /* against the CPU: the integer form's sums from its own codes, or the dequantized row */
        int idot = bdi_has(t);
        int8_t *w8 = (int8_t *)malloc(I), *xq = (int8_t *)malloc(I);
        float *sw = (float *)malloc((size_t)I / 16 * 4), *mw = (float *)malloc((size_t)I / 16 * 4), *dx = (float *)malloc((size_t)I / 32 * 4);
        float *deq = (float *)malloc((size_t)I * 4);
        int32_t *xs = (int32_t *)malloc((size_t)I / 16 * 4);
        double worst = 0;
        for (int s = 0; s < SM && ok; s++) {
            ref_quant(x + (size_t)s * I, I, xq, dx, xs);
            for (int o = 0; o < O; o++) {
                const uint8_t *r = w + (size_t)o * rw * 4;
                double ref = 0, mag = 0;
                if (idot) {
                    if (ref_unpack(t, r, I, w8, sw, mw)) { ok = 0; break; }
                    for (int b = 0; b < I / 32; b++) {
                        double s0 = 0, s1 = 0, a0 = 0, a1 = 0;
                        for (int i = 0; i < 16; i++) { s0 += w8[32 * b + i] * xq[32 * b + i]; a0 += fabs((double)w8[32 * b + i] * xq[32 * b + i]); }
                        for (int i = 16; i < 32; i++) { s1 += w8[32 * b + i] * xq[32 * b + i]; a1 += fabs((double)w8[32 * b + i] * xq[32 * b + i]); }
                        ref += (double)dx[b] * ((double)sw[2 * b] * s0 + (double)sw[2 * b + 1] * s1 - (double)mw[2 * b] * xs[2 * b] - (double)mw[2 * b + 1] * xs[2 * b + 1]);
                        mag += fabs((double)dx[b]) * (fabs((double)sw[2 * b]) * a0 + fabs((double)sw[2 * b + 1]) * a1 +
                                                      fabs((double)mw[2 * b] * xs[2 * b]) + fabs((double)mw[2 * b + 1] * xs[2 * b + 1]));
                    }
                } else {
                    if (ref_dequant(t, r, deq, I)) { ok = 0; break; }
                    for (int i = 0; i < I; i++) { ref += (double)deq[i] * x[(size_t)s * I + i]; mag += fabs((double)deq[i] * x[(size_t)s * I + i]); }
                }
                double e = fabs(one[(size_t)s * O + o] - ref) / (mag + 1e-6);
                if (!(e <= worst)) worst = e;   /* NaN too */
            }
        }
        if (!ok) { printf("FAIL %s: the CPU's decoder refused the row\n", ref_name(t)); fails++; }
        else if (!(worst <= 2e-5)) { printf("FAIL %s [%d x %d]: %s off by %.3g of the terms' magnitude\n", ref_name(t), O, I, idot ? "integer form" : "float form", worst); fails++; }
        else printf("%-8s [%6d x %5d] %s: within %.2g of the CPU's\n", ref_name(t), O, I, idot ? "integer form" : "float form  ", worst);
        free(w8); free(xq); free(sw); free(mw); free(dx); free(deq); free(xs);
    }
    /* the backend's upload (the file's rows, unpadded) and the chain's matmul: the kernel's bits */
    if (ok) {
        uint8_t *packed = (uint8_t *)malloc((size_t)O * rb);
        for (int o = 0; o < O; o++) memcpy(packed + (size_t)o * rb, w + (size_t)o * rw * 4, (size_t)rb);
        ColiCudaTensor *ct = NULL;
        /* the backend's block: ggml's, or 32 values for F16 and BF16 (blockdec_host.h) */
        int okb = coli_cuda_tensor_upload(&ct, packed, NULL, fmt, I, O, 0) && coli_cuda_block_elems(fmt) == (t == 1 || t == 30 ? 32 : be);
        cudaMemset(yd, 0, (size_t)SM * O * 4);
        for (int s = 0; s < SM && okb; s++) okb = coli_cuda_tensor_gemm_async(ct, yd + (size_t)s * O, xd + (size_t)s * I, 1, NULL);
        okb = okb && cudaDeviceSynchronize() == cudaSuccess;
        cudaMemcpy(got, yd, (size_t)SM * O * 4, cudaMemcpyDeviceToHost);
        if (!okb || memcmp(got, one, (size_t)SM * O * 4)) { printf("FAIL %s [%d x %d]: the backend's upload and matmul differ from the kernel's\n", ref_name(t), O, I); fails++; }
        /* a prompt chunk's rows (40: past the GEMV's 8) through the f16 tiles where the device has
         * them, against the dequantized rows in double: f16 operands, to 1e-2 of the terms' magnitude */
        if (okb && I % 64 == 0) {
            const int SG = 40;
            float *xg = (float *)malloc((size_t)SG * I * 4), *yg = (float *)malloc((size_t)SG * O * 4), *deq = (float *)malloc((size_t)I * 4);
            float *dxg, *dyg;
            for (int i = 0; i < SG * I; i++) xg[i] = ((int)(rnd() % 20001) - 10000) * 1e-4f;
            cudaMalloc(&dxg, (size_t)SG * I * 4); cudaMalloc(&dyg, (size_t)SG * O * 4);
            cudaMemcpy(dxg, xg, (size_t)SG * I * 4, cudaMemcpyHostToDevice);
            int okg = coli_cuda_tensor_gemm_async(ct, dyg, dxg, SG, NULL) && cudaDeviceSynchronize() == cudaSuccess;
            cudaMemcpy(yg, dyg, (size_t)SG * O * 4, cudaMemcpyDeviceToHost);
            double worst = 0;
            for (int o = 0; o < O && okg; o++) {
                if (ref_dequant(t, w + (size_t)o * rw * 4, deq, I)) { okg = 0; break; }
                for (int r = 0; r < SG; r += 13) {
                    double ref = 0, mag = 0;
                    for (int i = 0; i < I; i++) { ref += (double)deq[i] * xg[(size_t)r * I + i]; mag += fabs((double)deq[i] * xg[(size_t)r * I + i]); }
                    double e = fabs(yg[(size_t)r * O + o] - ref) / (mag + 1e-6);
                    if (!(e <= worst)) worst = e;
                }
            }
            if (!okg || !(worst <= 1e-2)) { printf("FAIL %s [%d x %d]: %d rows off by %.3g\n", ref_name(t), O, I, SG, worst); fails++; }
            else if (I == 2048 && O == 300) printf("%-8s [%6d x %5d] %d rows (tiles): within %.2g of the dequantized rows\n", ref_name(t), O, I, SG, worst);
            cudaFree(dxg); cudaFree(dyg); free(xg); free(yg); free(deq);
        }
        if (ct) coli_cuda_tensor_free(ct);
        free(packed);
    }
    /* a verify's rows together: the rows alone's bits; x quantized by each block, once for all of
     * them (bdx_quant_k) and read from global memory, and once and copied into each block's shared
     * memory (bd_gemv_xs: three rows or more, where x fits) alike */
    static const char *const how[3] = {"by each block", "once", "once, in shared memory"};
    for (int pq = 0; pq < 3 && ok && bdi_has(t); pq++) {
        g_bd_preq_force = pq > 0; g_bd_xs_force = pq == 2;
        for (int S = 1; S <= SM && ok; S++) {
            cudaMemset(yd, 0, (size_t)SM * O * 4);
            int okl = bd_gemv_launch(yd, xd, wd, g_tab, fmt, I, O, rw, S, g_smem, 0, 0) && cudaDeviceSynchronize() == cudaSuccess;
            cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
            if (!okl || memcmp(got, one, (size_t)S * O * 4)) {
                printf("FAIL %s [%d x %d]: %d rows (x quantized %s) differ from the rows alone\n", ref_name(t), O, I, S, how[pq]);
                fails++;
            }
        }
        g_bd_preq_force = -1; g_bd_xs_force = -1;
    }
    for (int S = 2; S <= SM && ok && !bdi_has(t); S++) {
        cudaMemset(yd, 0, (size_t)SM * O * 4);
        int okl = bd_gemv_launch(yd, xd, wd, g_tab, fmt, I, O, rw, S, g_smem, 0, 0) && cudaDeviceSynchronize() == cudaSuccess;
        cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
        if (!okl || memcmp(got, one, (size_t)S * O * 4)) { printf("FAIL %s [%d x %d]: %d rows differ from the rows alone\n", ref_name(t), O, I, S); fails++; }
    }
    /* a SwiGLU's gate and up in one pass (bd_glu_launch: a decode step's shared expert), 1 to 4
     * rows: the bits of the two GEMVs and CC_EW_SWIGLU's product */
    if (ok && bdi_has(t)) {
        uint8_t *wu = (uint8_t *)calloc((size_t)O, (size_t)rw * 4);
        for (int o = 0; o < O; o++) {
            uint8_t *r = wu + (size_t)o * rw * 4;
            for (long i = 0; i < rb; i++) r[i] = (uint8_t)rnd();
            for (long k = 0; k + bb <= rb; k += bb) bd_fix(t, r + k);
        }
        bd_u32 *ud; float *gd, *vd;
        cudaMalloc(&ud, (size_t)O * rw * 4); cudaMalloc(&gd, (size_t)SM * O * 4); cudaMalloc(&vd, (size_t)SM * O * 4);
        cudaMemcpy(ud, wu, (size_t)O * rw * 4, cudaMemcpyHostToDevice);
        float *want = (float *)malloc((size_t)SM * O * 4);
        /* x in each of its forms, as the verify's rows above: the one pass's bits in all of them */
        for (int pq = 0; pq < 3; pq++) {
            g_bd_preq_force = pq > 0; g_bd_xs_force = pq == 2;
            for (int S = 1; S <= SM; S++) {
                cudaMemset(yd, 0, (size_t)SM * O * 4);
                int okg = bd_gemv_launch(gd, xd, wd, g_tab, fmt, I, O, rw, S, g_smem, 0, 0) && bd_gemv_launch(vd, xd, ud, g_tab, fmt, I, O, rw, S, g_smem, 0, 0);
                if (okg) swiglu_ref_k<<<(S * O + 255) / 256, 256>>>(gd, gd, vd, S * O);
                okg = okg && bd_glu_launch(yd, xd, wd, ud, g_tab, fmt, I, O, rw, S, g_smem, 0) && cudaDeviceSynchronize() == cudaSuccess;
                cudaMemcpy(want, gd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
                if (!okg || memcmp(got, want, (size_t)S * O * 4)) {
                    printf("FAIL %s [%d x %d]: the gate and up of %d rows in one pass (x quantized %s) differ\n", ref_name(t), O, I, S, how[pq]);
                    fails++;
                }
            }
        }
        g_bd_preq_force = -1; g_bd_xs_force = -1;
        g_glu++;
        free(want); free(wu); cudaFree(ud); cudaFree(gd); cudaFree(vd);
    }
    if (bench) {
        /* copies of the matrix past the card's L2 (BENCH_MB, default 256), read in turn: a
         * decode step reads every matrix once, nothing of it is in L2 the next time */
        size_t mb = (size_t)O * rw * 4, want = (size_t)(getenv("BENCH_MB") ? atoi(getenv("BENCH_MB")) : 256) << 20;
        int ncopy = mb >= want ? 1 : (int)(want / mb);
        bd_u32 *wb = NULL;
        if (cudaMalloc(&wb, mb * ncopy) != cudaSuccess) { ncopy = 1; wb = wd; }
        else for (int c = 0; c < ncopy; c++) cudaMemcpy((uint8_t *)wb + mb * c, wd, mb, cudaMemcpyDeviceToDevice);
        int smax = getenv("BENCH_ROWS") ? atoi(getenv("BENCH_ROWS")) : 4;
        printf("  %-8s [%6d x %5d] %6.1f MB:", ref_name(t), O, I, (double)mb / 1e6);
        for (int S = 1; S <= smax; S++) {
            double best = 1e9;
            for (int k = 0; k < 5; k++) {
                cudaDeviceSynchronize(); double t0 = now();
                for (int j = 0; j < 40; j++)
                    bd_gemv_launch(yd, xd, (const bd_u32 *)((uint8_t *)wb + mb * (j % ncopy)), g_tab, fmt, I, O, rw, S, g_smem, 0, 0);
                cudaDeviceSynchronize(); double dt = (now() - t0) / 40;
                if (dt < best) best = dt;
            }
            printf(" %d: %.1f us (%.0f GB/s)", S, best * 1e6, (double)mb / best / 1e9);
        }
        printf("\n");
        if (wb != wd) cudaFree(wb);
    }
out:
    cudaFree(wd); cudaFree(xd); cudaFree(yd);
    free(w); free(x); free(one); free(got);
}

/* the CPU's integer form of row r (type t, n inputs) against x quantized (xq, dx, xs), in double */
static double ref_dot(int t, const uint8_t *r, int n, const int8_t *xq, const float *dx, const int32_t *xs) {
    static int8_t w8[16384]; static float sw[1024], mw[1024];
    if (ref_unpack(t, r, n, w8, sw, mw)) return NAN;
    double acc = 0;
    for (int b = 0; b < n / 32; b++) {
        double s0 = 0, s1 = 0;
        for (int i = 0; i < 16; i++) s0 += w8[32 * b + i] * xq[32 * b + i];
        for (int i = 16; i < 32; i++) s1 += w8[32 * b + i] * xq[32 * b + i];
        acc += (double)dx[b] * ((double)sw[2 * b] * s0 + (double)sw[2 * b + 1] * s1 - (double)mw[2 * b] * xs[2 * b] - (double)mw[2 * b + 1] * xs[2 * b + 1]);
    }
    return acc;
}
/* the dequantized row r (type t, n inputs) against x, in double */
static double ref_fdot(int t, const uint8_t *r, int n, const float *x) {
    static float deq[16384];
    if (ref_dequant(t, r, deq, n)) return NAN;
    double acc = 0;
    for (int i = 0; i < n; i++) acc += (double)deq[i] * x[i];
    return acc;
}
/* The chain's routed experts (coli_cuda_moe_async) of a GGUF: E experts of gate/up type tg
 * and down type td, S rows routed to K of them: up to 8 rows (the GEMV kernels) against the
 * CPU's integer form (each row quantized, then the hidden row quantized for down), more (a prompt
 * chunk's, the sorted f16 tiles) against the dequantized rows. */
static void check_moe(int tg, int td, int S, int K, int E, int D, int I) {
    long rg = ref_row_bytes(tg, D), rd = ref_row_bytes(td, I);
    if (rg <= 0 || rd <= 0) return;
    int bg = ref_block_bytes(tg), bdb = ref_block_bytes(td);
    size_t gb = (size_t)rg * I, db = (size_t)rd * D;
    uint8_t *G = (uint8_t *)malloc(gb * E), *U = (uint8_t *)malloc(gb * E), *Dn = (uint8_t *)malloc(db * E);
    for (size_t i = 0; i < gb * E; i++) { G[i] = (uint8_t)rnd(); U[i] = (uint8_t)rnd(); }
    for (size_t i = 0; i < db * E; i++) Dn[i] = (uint8_t)rnd();
    for (int e = 0; e < E; e++) {
        for (int o = 0; o < I; o++) for (long b = 0; b + bg <= rg; b += bg) { bd_fix(tg, G + e * gb + o * rg + b); bd_fix(tg, U + e * gb + o * rg + b); }
        for (int o = 0; o < D; o++) for (long b = 0; b + bdb <= rd; b += bdb) bd_fix(td, Dn + e * db + o * rd + b);
    }
    ColiCudaTensor **tt = (ColiCudaTensor **)calloc((size_t)3 * E, sizeof *tt);
    CcExpert *tab = (CcExpert *)calloc((size_t)E, sizeof *tab);
    int ok = 1;
    for (int e = 0; e < E && ok; e++) {
        ok = coli_cuda_tensor_upload(&tt[3 * e], G + e * gb, NULL, 100 + tg, D, I, 0) && coli_cuda_tensor_upload(&tt[3 * e + 1], U + e * gb, NULL, 100 + tg, D, I, 0) &&
             coli_cuda_tensor_upload(&tt[3 * e + 2], Dn + e * db, NULL, 100 + td, I, D, 0);
        const float *sc;
        int gsz;
        ok = ok && coli_cuda_tensor_dev(tt[3 * e], &tab[e].g, &sc, &gsz) && coli_cuda_tensor_dev(tt[3 * e + 1], &tab[e].u, &sc, &gsz) &&
             coli_cuda_tensor_dev(tt[3 * e + 2], &tab[e].d, &sc, &gsz);
    }
    float *x = (float *)malloc((size_t)S * D * 4), *w = (float *)malloc((size_t)S * K * 4), *y = (float *)malloc((size_t)S * D * 4);
    int *ids = (int *)malloc((size_t)S * K * 4);
    /* random K-quant blocks: the hidden rows of an x of 1e-4 run past f16's range in the tiles
     * (an operand held to 65504), a tenth keeps them in it, as test_vk_chain's moe rows do */
    for (int i = 0; i < S * D; i++) x[i] = ((int)(rnd() % 20001) - 10000) * (S > 8 ? 1e-5f : 1e-4f);
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int e;
            do { e = (int)(rnd() % E); int dup = 0; for (int j = 0; j < k; j++) dup |= ids[s * K + j] == e; if (!dup) break; } while (1);
            ids[s * K + k] = e; w[s * K + k] = (float)(1 + rnd() % 100) / 100.f;
        }
    if (S > 1) ids[K + 1] = -1;   /* a slot no expert holds adds nothing */
    CcExpert *dtab; float *dx_, *dw, *dy, *dh; int *di;
    size_t hf = cc_moe_hid_floats(S, K, D, I, E);
    cudaMalloc(&dtab, (size_t)E * sizeof *tab); cudaMalloc(&dx_, (size_t)S * D * 4); cudaMalloc(&dw, (size_t)S * K * 4);
    cudaMalloc(&dy, (size_t)S * D * 4); cudaMalloc(&dh, hf * 4); cudaMalloc(&di, (size_t)S * K * 4);
    cudaMemcpy(dtab, tab, (size_t)E * sizeof *tab, cudaMemcpyHostToDevice); cudaMemcpy(dx_, x, (size_t)S * D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, (size_t)S * K * 4, cudaMemcpyHostToDevice); cudaMemcpy(di, ids, (size_t)S * K * 4, cudaMemcpyHostToDevice);
    CcMoe mp = {};
    mp.S = S; mp.K = K; mp.D = D; mp.I = I; mp.fmt = 100 + tg; mp.fmt_d = 100 + td; mp.parts = 3; mp.E = E;
    mp.x_row = D; mp.i_row = K; mp.w_row = K; mp.y_row = D; mp.h_off = -1;
    ok = ok && coli_cuda_moe_async(0, dtab, dx_, di, dw, dh, dy, NULL, &mp, NULL) && cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(y, dy, (size_t)S * D * 4, cudaMemcpyDeviceToHost);
    double worst = 0;
    if (ok) {
        int8_t *xq = (int8_t *)malloc(16384); float *dxs = (float *)malloc(4096); int32_t *xss = (int32_t *)malloc(4096);
        float *h = (float *)malloc((size_t)I * 4);
        for (int s = 0; s < S; s++) {
            double *ref = (double *)calloc((size_t)D, sizeof(double)), *mag = (double *)calloc((size_t)D, sizeof(double));
            for (int k = 0; k < K; k++) {
                int e = ids[s * K + k];
                if (e < 0) continue;
                if (S <= 8) {   /* the GEMV kernels: the CPU's integer form, row and hidden row quantized */
                    ref_quant(x + (size_t)s * D, D, xq, dxs, xss);
                    for (int o = 0; o < I; o++) {
                        double g = ref_dot(tg, G + e * gb + o * rg, D, xq, dxs, xss), u = ref_dot(tg, U + e * gb + o * rg, D, xq, dxs, xss);
                        h[o] = (float)(g / (1 + exp(-g)) * u);
                    }
                    ref_quant(h, I, xq, dxs, xss);
                    for (int o = 0; o < D; o++) {
                        double v = ref_dot(td, Dn + e * db + o * rd, I, xq, dxs, xss);
                        ref[o] += w[s * K + k] * v; mag[o] += fabs(w[s * K + k] * v);
                    }
                } else {        /* the f16 tiles: the dequantized rows in double, nothing quantized */
                    for (int o = 0; o < I; o++) {
                        double g = ref_fdot(tg, G + e * gb + o * rg, D, x + (size_t)s * D), u = ref_fdot(tg, U + e * gb + o * rg, D, x + (size_t)s * D);
                        h[o] = (float)(g / (1 + exp(-g)) * u);
                    }
                    for (int o = 0; o < D; o++) {
                        double v = ref_fdot(td, Dn + e * db + o * rd, I, h);
                        ref[o] += w[s * K + k] * v; mag[o] += fabs(w[s * K + k] * v);
                    }
                }
            }
            for (int o = 0; o < D; o++) { double e = fabs(y[(size_t)s * D + o] - ref[o]) / (mag[o] + 1e-3); if (!(e <= worst)) worst = e; }
            free(ref); free(mag);
        }
        free(xq); free(dxs); free(xss); free(h);
    }
    double tol = S > 8 ? 1e-2 : 2e-3;   /* past 8 rows the f16 tiles */
    /* the same experts, the odd ones in page-locked host memory the device reads over the bus
     * (CcExpert.host: an engine's host pool, gate, up and down from 256-byte boundaries): read
     * there, copied to a staging first (coli_cuda_moe_stage), and with a staging of one place
     * (the rest read where they are); each the bits of the run from device memory */
    if (ok && S <= 8) {
        size_t rgw = ((size_t)rg + 3) / 4 * 4, rdw = ((size_t)rd + 3) / 4 * 4;
        size_t gpad = ((size_t)I * rgw + 255) / 256 * 256, slot = (2 * gpad + (size_t)D * rdw + 255) / 256 * 256;
        char *hp = NULL, *dp = NULL, *stg = NULL;
        CcExpert *htab = (CcExpert *)malloc((size_t)E * sizeof *htab), *dtab2 = NULL;
        float *y2 = (float *)malloc((size_t)S * D * 4);
        int hok = htab && y2 && cudaHostAlloc((void **)&hp, slot * E, cudaHostAllocMapped) == cudaSuccess &&
                  cudaHostGetDevicePointer((void **)&dp, hp, 0) == cudaSuccess && cudaMalloc(&stg, CC_MOE_STAGE_MAP + 4 * slot) == cudaSuccess &&
                  cudaMalloc(&dtab2, (size_t)E * sizeof *htab) == cudaSuccess;
        if (hok) {
            memcpy(htab, tab, (size_t)E * sizeof *htab);
            memset(hp, 0, slot * E);
            for (int e = 1; e < E; e += 2) {
                char *b = hp + (size_t)e * slot;
                for (int o = 0; o < I; o++) { memcpy(b + o * rgw, G + e * gb + o * rg, (size_t)rg); memcpy(b + gpad + o * rgw, U + e * gb + o * rg, (size_t)rg); }
                for (int o = 0; o < D; o++) memcpy(b + 2 * gpad + o * rdw, Dn + e * db + o * rd, (size_t)rd);
                htab[e].g = dp + (size_t)e * slot; htab[e].u = dp + (size_t)e * slot + gpad; htab[e].d = dp + (size_t)e * slot + 2 * gpad; htab[e].host = 1;
            }
            hok = cudaMemcpy(dtab2, htab, (size_t)E * sizeof *htab, cudaMemcpyHostToDevice) == cudaSuccess;
        }
        static const char *how[3] = {"read over the bus", "staged (4 places)", "staged (1 place)"};
        for (int md = 0; md < 3; md++) {
            int same = hok && coli_cuda_moe_stage(0, md ? stg : NULL, slot, md == 1 ? 4 : 1) && cudaMemset(dy, 0, (size_t)S * D * 4) == cudaSuccess &&
                       coli_cuda_moe_async(0, dtab2, dx_, di, dw, dh, dy, NULL, &mp, NULL) && cudaDeviceSynchronize() == cudaSuccess &&
                       cudaMemcpy(y2, dy, (size_t)S * D * 4, cudaMemcpyDeviceToHost) == cudaSuccess && !memcmp(y2, y, (size_t)S * D * 4);
            if (!same) { printf("FAIL moe %s/%s S %d: the host's experts %s differ from the device's\n", ref_name(tg), ref_name(td), S, how[md]); fails++; }
        }
        if (hok) printf("moe %-7s/%-7s S %d: the host's experts read over the bus, staged, one place staged: the device's bits\n", ref_name(tg), ref_name(td), S);
        coli_cuda_moe_stage(0, NULL, 0, 0);
        if (hp) cudaFreeHost(hp);
        cudaFree(stg); cudaFree(dtab2); free(htab); free(y2);
    }
    /* a prompt chunk's two lanes (CcMoe.lay_S): the sums of its first third of rows and of the
     * rest apart, over the parts 1 above (one call for the chunk's rows), each the chunk's bits */
    if (ok && S > 8) {
        const int r0 = S / 3;
        float *dy2 = NULL, *y2 = (float *)malloc((size_t)S * D * 4);
        CcMoe la = mp, lb = mp;
        la.parts = lb.parts = 2; la.S = r0; la.lay_S = lb.lay_S = S; lb.S = S - r0; lb.lay_s0 = r0;
        lb.i_off = r0 * K; lb.w_off = r0 * K; lb.y_off = r0 * D;
        int lok = y2 && cudaMalloc(&dy2, (size_t)S * D * 4) == cudaSuccess && coli_cuda_moe_async(0, dtab, NULL, di, dw, dh, dy2, NULL, &la, NULL) &&
                  coli_cuda_moe_async(0, dtab, NULL, di, dw, dh, dy2, NULL, &lb, NULL) && cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(y2, dy2, (size_t)S * D * 4, cudaMemcpyDeviceToHost) == cudaSuccess && !memcmp(y2, y, (size_t)S * D * 4);
        CcMoe bad = mp; bad.lay_S = S + 4;   /* a parts 1 over another layout: refused */
        lok = lok && !coli_cuda_moe_async(0, dtab, dx_, di, dw, dh, dy2, NULL, &bad, NULL);
        if (!lok) { printf("FAIL moe %s/%s S %d: the lanes' sums (lay_S) differ from the chunk's\n", ref_name(tg), ref_name(td), S); fails++; }
        else printf("moe %-7s/%-7s S %d: two lanes' sums over the chunk's parts 1, the chunk's bits\n", ref_name(tg), ref_name(td), S);
        cudaFree(dy2); free(y2);
    }
    /* the tier's expert groups (coli_cuda_expert_group: the frame's kernels, a row a slot, K 1)
     * on row 0's experts, each expert's down output against the CPU's integer form */
    if (ok && S == 1) {
        ColiCudaTensor *gg[32], *uu[32], *dd[32]; int rr[32], cnt = 0;
        for (int k = 0; k < K; k++) { int e = ids[k]; if (e < 0) continue; gg[cnt] = tt[3 * e]; uu[cnt] = tt[3 * e + 1]; dd[cnt] = tt[3 * e + 2]; rr[cnt++] = 1; }
        float *xr = (float *)malloc((size_t)cnt * D * 4), *yr = (float *)malloc((size_t)cnt * D * 4);
        for (int c = 0; c < cnt; c++) memcpy(xr + (size_t)c * D, x, (size_t)D * 4);
        int okg = coli_cuda_expert_group(gg, uu, dd, rr, cnt, yr, xr);
        int8_t *xq = (int8_t *)malloc(16384); float *dxs = (float *)malloc(4096), *h = (float *)malloc((size_t)I * 4); int32_t *xss = (int32_t *)malloc(4096);
        double gw = 0;
        for (int c = 0, k = 0; okg && k < K; k++) {
            int e = ids[k]; if (e < 0) continue;
            ref_quant(x, D, xq, dxs, xss);
            for (int o = 0; o < I; o++) {
                double g = ref_dot(tg, G + e * gb + o * rg, D, xq, dxs, xss), u = ref_dot(tg, U + e * gb + o * rg, D, xq, dxs, xss);
                h[o] = (float)(g / (1 + exp(-g)) * u);
            }
            ref_quant(h, I, xq, dxs, xss);
            double vmax = 1e-6, emax = 0;   /* to the expert's largest output */
            for (int o = 0; o < D; o++) {
                double v = ref_dot(td, Dn + e * db + o * rd, I, xq, dxs, xss), er = fabs(yr[(size_t)c * D + o] - v);
                if (fabs(v) > vmax) vmax = fabs(v);
                if (!(er <= emax)) emax = er;
            }
            if (!(emax / vmax <= gw)) gw = emax / vmax;
            c++;
        }
        if (!okg || !(gw <= 1e-5)) { printf("FAIL moe group %s/%s: %s %.3g\n", ref_name(tg), ref_name(td), okg ? "off by" : "the group failed", gw); fails++; }
        else printf("moe group %-7s/%-7s %d experts: within %.2g of the CPU's integer form\n", ref_name(tg), ref_name(td), cnt, gw);
        free(xr); free(yr); free(xq); free(dxs); free(h); free(xss);
    }
    if (!ok || !(worst <= tol)) { printf("FAIL moe %s/%s S %d K %d E %d [%d x %d]: %s %.3g\n", ref_name(tg), ref_name(td), S, K, E, D, I, ok ? "off by" : "the call failed", worst); fails++; }
    else printf("moe %-7s/%-7s S %d K %d E %d [%d x %d]: within %.2g of the %s\n", ref_name(tg), ref_name(td), S, K, E, D, I, worst,
                S > 8 ? "dequantized rows (f16 tiles)" : "CPU's integer form");
    cudaFree(dtab); cudaFree(dx_); cudaFree(dw); cudaFree(dy); cudaFree(dh); cudaFree(di);
    for (int i = 0; i < 3 * E; i++) if (tt[i]) coli_cuda_tensor_free(tt[i]);
    free(tt); free(tab); free(G); free(U); free(Dn); free(x); free(w); free(y); free(ids);
}

/* BENCH_MOE: a prompt chunk's routed experts through the f16 tiles, timed: E experts of tg/tg/td with
 * random blocks, S rows of K distinct routes, parts 1 and 2 (the tiles and the sum), ms a call */
static void check_ok(int ok, const char *what) { if (ok) printf("  %s\n", what); else { printf("FAIL %s\n", what); fails++; } }
/* check: no timing; the rows through coli_cuda_moe_cols_async at slices of 128, 896 and all of D
 * outputs against one coli_cuda_moe_async of parts 3, to the bit */
static void bench_moe(int tg, int td, int S, int K, int E, int D, int I, int reps, int check = 0) {
    long rg = ref_row_bytes(tg, D), rd = ref_row_bytes(td, I);
    if (rg <= 0 || rd <= 0) return;
    int bg = ref_block_bytes(tg), bdb = ref_block_bytes(td);
    size_t gb = (size_t)rg * I, db = (size_t)rd * D;
    uint8_t *G = (uint8_t *)malloc(gb), *U = (uint8_t *)malloc(gb), *Dn = (uint8_t *)malloc(db);
    ColiCudaTensor **tt = (ColiCudaTensor **)calloc((size_t)3 * E, sizeof *tt);
    CcExpert *tab = (CcExpert *)calloc((size_t)E, sizeof *tab);
    int ok = 1;
    for (int e = 0; e < E && ok; e++) {   /* each expert its own random blocks */
        for (size_t i = 0; i < gb; i++) { G[i] = (uint8_t)rnd(); U[i] = (uint8_t)rnd(); }
        for (size_t i = 0; i < db; i++) Dn[i] = (uint8_t)rnd();
        for (int o = 0; o < I; o++) for (long b = 0; b + bg <= rg; b += bg) { bd_fix(tg, G + o * rg + b); bd_fix(tg, U + o * rg + b); }
        for (int o = 0; o < D; o++) for (long b = 0; b + bdb <= rd; b += bdb) bd_fix(td, Dn + o * rd + b);
        ok = coli_cuda_tensor_upload(&tt[3 * e], G, NULL, 100 + tg, D, I, 0) && coli_cuda_tensor_upload(&tt[3 * e + 1], U, NULL, 100 + tg, D, I, 0) &&
             coli_cuda_tensor_upload(&tt[3 * e + 2], Dn, NULL, 100 + td, I, D, 0);
        const float *sc; int gsz;
        ok = ok && coli_cuda_tensor_dev(tt[3 * e], &tab[e].g, &sc, &gsz) && coli_cuda_tensor_dev(tt[3 * e + 1], &tab[e].u, &sc, &gsz) &&
             coli_cuda_tensor_dev(tt[3 * e + 2], &tab[e].d, &sc, &gsz);
    }
    float *x = (float *)malloc((size_t)S * D * 4), *w = (float *)malloc((size_t)S * K * 4);
    int *ids = (int *)malloc((size_t)S * K * 4);
    for (size_t i = 0; i < (size_t)S * D; i++) x[i] = ((int)(rnd() % 20001) - 10000) * 1e-5f;
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int e;
            do { e = (int)(rnd() % E); int dup = 0; for (int j = 0; j < k; j++) dup |= ids[s * K + j] == e; if (!dup) break; } while (1);
            ids[s * K + k] = e; w[s * K + k] = (float)(1 + rnd() % 100) / 100.f;
        }
    CcExpert *dtab; float *dx_, *dw, *dy, *dh; int *di;
    size_t hf = cc_moe_hid_floats(S, K, D, I, E);
    cudaMalloc(&dtab, (size_t)E * sizeof *tab); cudaMalloc(&dx_, (size_t)S * D * 4); cudaMalloc(&dw, (size_t)S * K * 4);
    cudaMalloc(&dy, (size_t)S * D * 4); cudaMalloc(&dh, hf * 4); cudaMalloc(&di, (size_t)S * K * 4);
    cudaMemcpy(dtab, tab, (size_t)E * sizeof *tab, cudaMemcpyHostToDevice); cudaMemcpy(dx_, x, (size_t)S * D * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w, (size_t)S * K * 4, cudaMemcpyHostToDevice); cudaMemcpy(di, ids, (size_t)S * K * 4, cudaMemcpyHostToDevice);
    CcMoe mp = {};
    mp.S = S; mp.K = K; mp.D = D; mp.I = I; mp.fmt = 100 + tg; mp.fmt_d = 100 + td; mp.parts = 3; mp.E = E;
    mp.x_row = D; mp.i_row = K; mp.w_row = K; mp.y_row = D; mp.h_off = -1;
    if (check) {
        std::vector<float> ref((size_t)S * D), got((size_t)S * D);
        ok = ok && coli_cuda_moe_async(0, dtab, dx_, di, dw, dh, dy, NULL, &mp, NULL) && cudaDeviceSynchronize() == cudaSuccess &&
             cudaMemcpy(ref.data(), dy, ref.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        const int cl[3] = {128, 896, (D + 127) / 128 * 128};
        for (int c = 0; c < 3 && ok; c++) {
            const size_t need = cc_moe_hid_floats(S, K, cl[c] < D ? cl[c] : D, I, E);
            int r = need <= hf ? 1 : -2;
            cudaMemset(dy, 0xff, (size_t)S * D * 4);
            if (r == 1) r = coli_cuda_moe_cols_async(0, dtab, dx_, di, dw, dh, dy, &mp, cl[c], NULL);
            ok = r == 1 && cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
            int same = ok && !memcmp(got.data(), ref.data(), got.size() * 4);
            char what[160];
            snprintf(what, sizeof what, "moe %s/%s S %d K %d E %d [%d x %d]: down by %d outputs, the bits of one call", ref_name(tg), ref_name(td),
                     S, K, E, D, I, cl[c]);
            check_ok(same, what);
        }
        CcMoe few = mp; few.S = CC_MOE_FRAME_ROWS; few.h_off = 0;
        check_ok(coli_cuda_moe_cols_async(0, dtab, dx_, di, dw, dh, dy, &few, 128, NULL) == -1, "moe_cols: host rows or a decode step's rows not taken");
        few.h_off = -1; few.S = S;
        check_ok(coli_cuda_moe_cols_async(0, dtab, dx_, di, dw, dh, dy, &few, 100, NULL) == -1, "moe_cols: a slice not of 128s not taken");
        cudaFree(dtab); cudaFree(dx_); cudaFree(dw); cudaFree(dy); cudaFree(dh); cudaFree(di);
        for (int i = 0; i < 3 * E; i++) if (tt[i]) coli_cuda_tensor_free(tt[i]);
        free(tt); free(tab); free(G); free(U); free(Dn); free(x); free(w); free(ids);
        return;
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 2 && ok; r++) ok = coli_cuda_moe_async(0, dtab, dx_, di, dw, dh, dy, NULL, &mp, NULL);
    ok = ok && cudaDeviceSynchronize() == cudaSuccess;
    cudaEventRecord(e0);
    for (int r = 0; r < reps && ok; r++) ok = coli_cuda_moe_async(0, dtab, dx_, di, dw, dh, dy, NULL, &mp, NULL);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    const double per = ms / reps, wbytes = (double)E * (2 * gb + db), flop = 2.0 * S * K * (3.0 * D * I);
    if (!ok) { printf("FAIL bench moe %s/%s: the call failed\n", ref_name(tg), ref_name(td)); fails++; }
    else printf("BENCH moe %-7s/%-7s S %d K %d E %d [%d x %d]: %.3f ms a call, %.1f TFLOP/s, the experts %.0f MB\n",
                ref_name(tg), ref_name(td), S, K, E, D, I, per, flop / per / 1e9, wbytes / 1e6);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dtab); cudaFree(dx_); cudaFree(dw); cudaFree(dy); cudaFree(dh); cudaFree(di);
    for (int i = 0; i < 3 * E; i++) if (tt[i]) coli_cuda_tensor_free(tt[i]);
    free(tt); free(tab); free(G); free(U); free(Dn); free(x); free(w); free(ids);
}

/* bd_mmg_launch of type t at I inputs, O rows (16 outputs a block, a part one; the GEMV's 8, 16 or 32
 * classes): every row count's rows the bits of the GEMV's rows alone */
static int g_mmq;
static void check_mmq(int t, int I, int O) {
    const int SM = 8, Ss[] = {1, 2, 3, 4, 5, 6, 7, 8};
    long rb = ref_row_bytes(t, I);
    int bb = ref_block_bytes(t);
    int rw = (int)((rb + 3) / 4), fmt = 100 + t;
    uint8_t *w = (uint8_t *)calloc((size_t)O, (size_t)rw * 4);
    for (int o = 0; o < O; o++) {
        uint8_t *r = w + (size_t)o * rw * 4;
        for (long i = 0; i < rb; i++) r[i] = (uint8_t)rnd();
        for (long k = 0; k + bb <= rb; k += bb) bd_fix(t, r + k);
    }
    float *x = (float *)malloc((size_t)SM * I * 4), *x2 = (float *)malloc((size_t)SM * I * 4);
    for (int i = 0; i < SM * I; i++) { x[i] = ((int)(rnd() % 20001) - 10000) * 1e-4f; x2[i] = ((int)(rnd() % 20001) - 10000) * 1e-4f; }
    bd_u32 *wd; float *xd, *yd;
    cudaMalloc(&wd, (size_t)O * rw * 4); cudaMalloc(&xd, (size_t)SM * I * 4); cudaMalloc(&yd, (size_t)SM * O * 4);
    cudaMemcpy(wd, w, (size_t)O * rw * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(xd, x, (size_t)SM * I * 4, cudaMemcpyHostToDevice);
    float *one = (float *)malloc((size_t)SM * O * 4), *two = (float *)malloc((size_t)SM * O * 4), *got = (float *)malloc((size_t)SM * O * 4);
    int ok = 1;
    /* the GEMV's rows alone, of x and of x2 */
    for (int s = 0; s < SM && ok; s++) ok = bd_gemv_launch(yd + (size_t)s * O, xd + (size_t)s * I, wd, g_tab, fmt, I, O, rw, 1, g_smem, 0, 0);
    ok = ok && cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(one, yd, (size_t)SM * O * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(xd, x2, (size_t)SM * I * 4, cudaMemcpyHostToDevice);
    for (int s = 0; s < SM && ok; s++) ok = bd_gemv_launch(yd + (size_t)s * O, xd + (size_t)s * I, wd, g_tab, fmt, I, O, rw, 1, g_smem, 0, 0);
    ok = ok && cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(two, yd, (size_t)SM * O * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(xd, x, (size_t)SM * I * 4, cudaMemcpyHostToDevice);
    if (!ok) { printf("FAIL %s [%d x %d]: the GEMV's launch failed (%s)\n", ref_name(t), O, I, cudaGetErrorString(cudaGetLastError())); fails++; goto out; }
    for (int k = 0; k < (int)(sizeof Ss / sizeof *Ss) && ok; k++) {
        const int S = Ss[k];
        for (int rep = 0; rep < 2 && ok; rep++) {   /* twice: the scratch reused */
            cudaMemset(yd, 0, (size_t)SM * O * 4);
            int okl = bd_mmg_launch(yd, xd, wd, fmt, I, O, rw, S, 0, g_tab) && cudaDeviceSynchronize() == cudaSuccess;
            cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
            if (!okl || memcmp(got, one, (size_t)S * O * 4)) {
                printf("FAIL %s [%d x %d]: %d rows on the tensor cores differ from the GEMV's rows alone (%s)\n", ref_name(t), O, I, S,
                       cudaGetErrorString(cudaGetLastError()));
                fails++; ok = 0;
            }
        }
    }
    if (ok) {
        /* x's quantization kept for the next matmul (the chain's word): x changed behind it, the kept
         * codes' bits (the word is the caller's promise); without the word quantized again, the new x's
         * rows' bits; with it but another row count, quantized again as well */
        const int S = 7;
        int okl = bd_mmg_launch(yd, xd, wd, fmt, I, O, rw, S, 0, g_tab) && cudaDeviceSynchronize() == cudaSuccess;
        cudaMemcpy(xd, x2, (size_t)SM * I * 4, cudaMemcpyHostToDevice);
        cudaMemset(yd, 0, (size_t)SM * O * 4);
        g_bd_x_same = 1;
        okl = okl && bd_mmg_launch(yd, xd, wd, fmt, I, O, rw, S, 0, g_tab) && cudaDeviceSynchronize() == cudaSuccess;
        g_bd_x_same = 0;
        cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
        if (!okl || memcmp(got, one, (size_t)S * O * 4)) { printf("FAIL %s [%d x %d]: x's quantization not kept\n", ref_name(t), O, I); fails++; ok = 0; }
        okl = bd_mmg_launch(yd, xd, wd, fmt, I, O, rw, S, 0, g_tab) && cudaDeviceSynchronize() == cudaSuccess;
        cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
        if (!okl || memcmp(got, two, (size_t)S * O * 4)) { printf("FAIL %s [%d x %d]: another x at the same address, not quantized again\n", ref_name(t), O, I); fails++; ok = 0; }
        g_bd_x_same = 1;
        okl = bd_mmg_launch(yd, xd, wd, fmt, I, O, rw, 7, 0, g_tab) && bd_mmg_launch(yd, xd, wd, fmt, I, O, rw, 4, 0, g_tab) && cudaDeviceSynchronize() == cudaSuccess;
        g_bd_x_same = 0;
        cudaMemcpy(got, yd, (size_t)4 * O * 4, cudaMemcpyDeviceToHost);
        if (!okl || memcmp(got, two, (size_t)4 * O * 4)) { printf("FAIL %s [%d x %d]: the word with another row count, other bits\n", ref_name(t), O, I); fails++; ok = 0; }
    }
    if (ok) g_mmq++;
out:
    cudaFree(wd); cudaFree(xd); cudaFree(yd);
    free(w); free(x); free(x2); free(one); free(two); free(got);
}

/* the backend's matmuls (coli_cuda_tensor_gemm_async) with the chain's word: A (6 rows, the tensor
 * cores) quantizes x; x changes; B (float rows, which quantize nothing) without the word; A with it:
 * x quantized again (B forgot the scratch's x), the new x's bits */
static void check_xq_forget(void) {
    const int t = 12, I = 2048, OA = 2048, OB = 16, S = 6;
    long rb = ref_row_bytes(t, I);
    int bb = ref_block_bytes(t);
    uint8_t *w = (uint8_t *)malloc((size_t)OA * rb);
    for (size_t i = 0; i < (size_t)OA * rb; i++) w[i] = (uint8_t)rnd();
    for (size_t k = 0; k + bb <= (size_t)OA * rb; k += bb) bd_fix(t, w + k);
    float *x = (float *)malloc((size_t)S * I * 4), *x2 = (float *)malloc((size_t)S * I * 4);
    float *want = (float *)malloc((size_t)S * OA * 4), *got = (float *)malloc((size_t)S * OA * 4);
    for (int i = 0; i < S * I; i++) { x[i] = ((int)(rnd() % 20001) - 10000) * 1e-4f; x2[i] = ((int)(rnd() % 20001) - 10000) * 1e-4f; }
    ColiCudaTensor *ta = NULL, *tb = NULL;
    float *xd, *ya, *yb;
    cudaMalloc(&xd, (size_t)S * I * 4); cudaMalloc(&ya, (size_t)S * OA * 4); cudaMalloc(&yb, (size_t)S * OB * 4);
    float *wf = (float *)malloc((size_t)OB * I * 4);
    for (int i = 0; i < OB * I; i++) wf[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
    int ok = coli_cuda_tensor_upload(&ta, w, NULL, 100 + t, I, OA, 0) && coli_cuda_tensor_upload(&tb, wf, NULL, 0, I, OB, 0);
    free(wf);
    cudaMemcpy(xd, x2, (size_t)S * I * 4, cudaMemcpyHostToDevice);
    ok = ok && coli_cuda_tensor_gemm_async(ta, ya, xd, S, NULL) && cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(want, ya, (size_t)S * OA * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(xd, x, (size_t)S * I * 4, cudaMemcpyHostToDevice);
    ok = ok && coli_cuda_tensor_gemm_async(ta, ya, xd, S, NULL);
    cudaMemcpy(xd, x2, (size_t)S * I * 4, cudaMemcpyHostToDevice);
    ok = ok && coli_cuda_tensor_gemm_async(tb, yb, xd, S, NULL);
    coli_cuda_bd_x_same(1);
    ok = ok && coli_cuda_tensor_gemm_async(ta, ya, xd, S, NULL) && cudaDeviceSynchronize() == cudaSuccess;
    coli_cuda_bd_x_same(0);
    cudaMemcpy(got, ya, (size_t)S * OA * 4, cudaMemcpyDeviceToHost);
    if (!ok || memcmp(got, want, (size_t)S * OA * 4)) { printf("FAIL the chain's word after a matmul without it: a stale quantization of x\n"); fails++; }
    else printf("mmq: a matmul without the chain's word forgets the scratch's x, the next one with it quantizes again\n");
    if (ta) coli_cuda_tensor_free(ta);
    if (tb) coli_cuda_tensor_free(tb);
    cudaFree(xd); cudaFree(ya); cudaFree(yb);
    free(w); free(x); free(x2); free(want); free(got);
}

int main(void) {
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("SKIP: no CUDA device\n"); return 0; }
    int bench = getenv("BENCH") && *getenv("BENCH") == '1';
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    g_smem = optin > 48 * 1024 ? (size_t)optin : 48 * 1024;
    uint32_t *tab = (uint32_t *)malloc((size_t)BDT_WORDS * 4);
    bdtab_fill(tab, NULL);
    cudaMalloc(&g_tab, (size_t)BDT_WORDS * 4);
    cudaMemcpy(g_tab, tab, (size_t)BDT_WORDS * 4, cudaMemcpyHostToDevice);
    free(tab);
    const int types[] = {8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 42, 2, 3, 6, 7, 34, 35, 39, 40, 41, 1, 30};
    const int shapes[][2] = {{2048, 300}, {4096, 70}, {768, 1000}, {12288, 40}, {2816, 33}};
    /* BENCH_MOE="tg:td:S,...": a prompt chunk's routed experts of Qwen3.8's shape (10 of 512 a row,
     * 2560 x 640) through the f16 tiles, timed */
    if (getenv("BENCH_MOE")) {
        char *spec = strdup(getenv("BENCH_MOE")), *save = NULL;
        for (char *tok = strtok_r(spec, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
            int tg = 0, td = 0, S = 0;
            if (sscanf(tok, "%d:%d:%d", &tg, &td, &S) == 3) bench_moe(tg, td, S, 10, 512, 2560, 640, 10);
        }
        free(spec); cudaFree(g_tab);
        printf(fails ? "FAILED: %d\n" : "OK\n", fails);
        return fails != 0;
    }
    /* BENCH_ONLY="t:I:O,...": time those (and check them) only, e.g. LFM3's decode matrices */
    if (getenv("BENCH_ONLY")) {
        char *spec = strdup(getenv("BENCH_ONLY")), *save = NULL;
        printf("BENCH (rows of x: 1 to BENCH_ROWS)\n");
        for (char *tok = strtok_r(spec, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
            int t = 0, I = 0, O = 0;
            if (sscanf(tok, "%d:%d:%d", &t, &I, &O) == 3) check(t, I, O, 1);
        }
        free(spec); cudaFree(g_tab);
        printf(fails ? "FAILED: %d\n" : "OK\n", fails);
        return fails != 0;
    }
    for (size_t i = 0; i < sizeof types / sizeof *types; i++)
        for (size_t k = 0; k < sizeof shapes / sizeof *shapes; k++) check(types[i], shapes[k][0], shapes[k][1], 0);
    printf("glu: %d types and shapes, gate and up of 1 to 4 rows in one pass, the bits of the two GEMVs and the SwiGLU\n", g_glu);
    /* the routed experts: the commonest pairs of the published files, one row and a verify's */
    /* {22, 6} and {12, 7}: unsloth's Dynamic V3.1 Qwen3.8 files, IQ2_S gate and up with a Q5_0 down (UD-IQ2_XS),
     * Q4_K with Q5_1 (UD-Q4_K_XL); {16, 42}, {16, 20} and {17, 42}: their IQ2_XXS and IQ2_XS layers (UD-IQ1_M,
     * UD-Q2_0, UD-IQ2_XS); {18, 20}, {18, 6} and {23, 6}: their IQ3_XXS and IQ4_XS layers (UD-Q2_K_XL, UD-IQ3_XXS,
     * UD-Q3_K_XL, UD-IQ4_XS) */
    const int pairs[][2] = {{12, 14}, {16, 18}, {17, 17}, {22, 23}, {11, 13}, {10, 12}, {19, 29}, {21, 20}, {8, 8}, {42, 42}, {22, 42}, {22, 8},
                            {22, 6}, {12, 7}, {6, 6}, {16, 42}, {17, 42}, {16, 20}, {17, 16}, {7, 7}, {18, 20}, {18, 6}, {23, 6}};
    for (size_t i = 0; i < sizeof pairs / sizeof *pairs; i++) {
        check_moe(pairs[i][0], pairs[i][1], 1, 8, 16, 2048, 512);
        check_moe(pairs[i][0], pairs[i][1], 3, 8, 16, 1024, 768);
        check_moe(pairs[i][0], pairs[i][1], 40, 8, 16, 1024, 768);
    }
    for (int g : {22, 16, 29}) for (int S : {2, 4}) check_moe(g, 42, S, 10, 24, 2560, 640);   /* Qwen3.8-Flash-Next IQ2_XS's experts, a verify's rows */
    /* a prompt's rows of a layer in one call, down by slices (coli_cuda_moe_cols_async): Qwen3.8's shape, 40 experts,
     * the 64-row tiles (S 100) and the 128-row ones (S 2100), against one call of every output */
    for (int g : {42, 16}) for (int S : {100, 2100}) bench_moe(g, 42, S, 10, 40, 2560, 640, 0, 1);
    /* last: the tests above keep their random draws */
    {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        if (major >= 8) {
            /* the GEMV's 16, 32, 32 and 8 classes (1, 4, 4 and 1 warps a block), part tiles of 16 outputs */
            const int mshape[][2] = {{2048, 300}, {5120, 1000}, {17408, 40}, {1024, 8200}};
            for (int t : {8, 12, 13, 14, 20, 23, 10, 11, 16, 17, 18, 19, 21, 22}) for (auto &sh : mshape) check_mmq(t, sh[0], sh[1]);
            printf("mmq: %d types and shapes, 1 to 8 rows on the tensor cores, each the bits of the GEMV's row alone\n", g_mmq);
            check_xq_forget();
        } else printf("mmq: SKIP (compute capability %d, the tensor cores' integer steps need 8)\n", major);
    }
    if (bench) {
        printf("BENCH (rows of x: 1 2 3 4)\n");
        const int bt[] = {12, 14, 8, 17, 22, 16, 21, 23, 11, 10};
        const int bs[][2] = {{2048, 248320}, {2048, 8192}, {8192, 2048}, {2048, 512}};
        for (size_t i = 0; i < sizeof bt / sizeof *bt; i++)
            for (size_t k = 0; k < sizeof bs / sizeof *bs; k++) check(bt[i], bs[k][0], bs[k][1], 1);
    }
    cudaFree(g_tab);
    printf(fails ? "FAILED: %d\n" : "OK\n", fails);
    return fails != 0;
}
