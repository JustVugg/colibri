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
 *     device where a row is not), the dense chain's matmul, the same bits as the kernel's.
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
    case 3: case 7: F16AT(0); F16AT(2); break;     /* Q4_1, Q5_1 */
    case 10: F16AT(80); F16AT(82); break;          /* Q2_K */
    case 11: F16AT(108); break;                    /* Q3_K */
    case 14: F16AT(208); break;                    /* Q6_K */
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
        int okb = coli_cuda_tensor_upload(&ct, packed, NULL, fmt, I, O, 0) && coli_cuda_block_elems(fmt) == be;
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
    /* a verify's rows together: the rows alone's bits */
    for (int S = 2; S <= SM && ok; S++) {
        cudaMemset(yd, 0, (size_t)SM * O * 4);
        int okl = bd_gemv_launch(yd, xd, wd, g_tab, fmt, I, O, rw, S, g_smem, 0, 0) && cudaDeviceSynchronize() == cudaSuccess;
        cudaMemcpy(got, yd, (size_t)S * O * 4, cudaMemcpyDeviceToHost);
        if (!okl || memcmp(got, one, (size_t)S * O * 4)) { printf("FAIL %s [%d x %d]: %d rows differ from the rows alone\n", ref_name(t), O, I, S); fails++; }
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
    const int types[] = {8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 42, 2, 3, 6, 7, 34, 35, 39, 40, 41};
    const int shapes[][2] = {{2048, 300}, {4096, 70}, {768, 1000}, {12288, 40}, {2816, 33}};
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
    /* the routed experts: the commonest pairs of the published files, one row and a verify's */
    const int pairs[][2] = {{12, 14}, {16, 18}, {17, 17}, {22, 23}, {11, 13}, {10, 12}, {19, 29}, {21, 20}, {8, 8}, {42, 42}};
    for (size_t i = 0; i < sizeof pairs / sizeof *pairs; i++) {
        check_moe(pairs[i][0], pairs[i][1], 1, 8, 16, 2048, 512);
        check_moe(pairs[i][0], pairs[i][1], 3, 8, 16, 1024, 768);
        check_moe(pairs[i][0], pairs[i][1], 40, 8, 16, 1024, 768);
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
