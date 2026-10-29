/* bench_cuda_bdgemm: a prompt chunk's rows of a GGUF block tensor on the CUDA backend
 * (bd_tensor_gemv's tiles: hcm_dense_k for Q8_0 and the K-quants, COLI_CUDA_HCM_TILE and
 * COLI_CUDA_HCM_SPLIT to pick its tile and split), against a float reference of the same codes
 * and scales on the CPU, and timed.
 *
 *   bench_cuda_bdgemm [S]      every type and shape below; S rows (default 512)
 *
 * The weights are random blocks with sane scales; the reference decodes them with
 * blockdec_host.h (the CPU's own block decoder) and sums in double. A type's line gives the
 * worst relative error of the tensor-core path, and the microseconds and TFLOP/s of a call
 * (BENCH_R calls; BENCH_SEQ "type.shape ..." picks the cases, their indices below). BENCH_F32=1:
 * f32 weights instead (f32_gemm_launch), every sampled output against a double sum.
 *
 * Build (the backend in the same unit):  nvcc -O3 -std=c++17 -arch=native -I. tests/bench_cuda_bdgemm.cu -o bench_bdgemm -lcuda */
#include "../backend_cuda.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>

static uint64_t rng = 88172645463325252ULL;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static uint16_t f16_bits(float f) { __half h = __float2half(f); uint16_t u; memcpy(&u, &h, 2); return u; }
static float f16_val(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

/* one row of I weights of ggml type t, random codes, its scales small and positive */
static void fill_row(int t, uint8_t *r, int I) {
    if (t == 8) {
        for (int b = 0; b < I / 32; b++) {
            uint8_t *p = r + 34 * b; uint16_t d = f16_bits(1e-3f + (rnd() % 100) * 2e-5f); memcpy(p, &d, 2);
            for (int j = 0; j < 32; j++) p[2 + j] = (uint8_t)rnd();
        }
    } else if (t == 12 || t == 13) {
        int B = t == 12 ? 144 : 176;
        for (int b = 0; b < I / 256; b++) {
            uint8_t *p = r + B * b;
            for (int j = 0; j < B; j++) p[j] = (uint8_t)rnd();
            uint16_t d = f16_bits(1e-3f + (rnd() % 100) * 2e-5f), m = f16_bits(5e-4f + (rnd() % 100) * 1e-5f);
            memcpy(p, &d, 2); memcpy(p + 2, &m, 2);
        }
    } else {   /* Q6_K */
        for (int b = 0; b < I / 256; b++) {
            uint8_t *p = r + 210 * b;
            for (int j = 0; j < 208; j++) p[j] = (uint8_t)rnd();
            for (int j = 0; j < 16; j++) p[192 + j] = (uint8_t)(int8_t)((int)(rnd() % 61) - 30);
            uint16_t d = f16_bits(1e-3f + (rnd() % 100) * 2e-5f); memcpy(p + 208, &d, 2);
        }
    }
}
/* the weights as floats: the GGUF formulas */
static void decode_row(int t, const uint8_t *r, int I, float *w) {
    if (t == 8) {
        for (int b = 0; b < I / 32; b++) { uint16_t d; memcpy(&d, r + 34 * b, 2); for (int j = 0; j < 32; j++) w[32 * b + j] = f16_val(d) * (float)(int8_t)r[34 * b + 2 + j]; }
        return;
    }
    if (t == 12 || t == 13) {
        int B = t == 12 ? 144 : 176;
        for (int b = 0; b < I / 256; b++) {
            const uint8_t *p = r + B * b, *sc = p + 4, *qs = p + (t == 12 ? 16 : 48), *qh = p + 16;
            uint16_t dd, mm; memcpy(&dd, p, 2); memcpy(&mm, p + 2, 2);
            float d = f16_val(dd), dmin = f16_val(mm);
            for (int g = 0; g < 8; g++) {
                int s, m;
                if (g < 4) { s = sc[g] & 63; m = sc[g + 4] & 63; }
                else { s = (sc[g + 4] & 0xF) | ((sc[g - 4] >> 6) << 4); m = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4); }
                for (int l = 0; l < 32; l++) {
                    int c = g / 2, q = (qs[32 * c + l] >> (4 * (g & 1))) & 15;
                    if (t == 13) q |= ((qh[l] >> g) & 1) << 4;
                    w[256 * b + 32 * g + l] = d * s * q - dmin * m;
                }
            }
        }
        return;
    }
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *p = r + 210 * b, *ql = p, *qh = p + 128; const int8_t *sc = (const int8_t *)(p + 192);
        uint16_t dd; memcpy(&dd, p + 208, 2); float d = f16_val(dd);
        for (int n = 0; n < 2; n++)
            for (int l = 0; l < 32; l++) {
                int is = 8 * n + l / 16;
                const uint8_t *L = ql + 64 * n, *H = qh + 32 * n;
                int q1 = ((L[l] & 0xF) | (((H[l] >> 0) & 3) << 4)) - 32, q2 = ((L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((L[l] >> 4) | (((H[l] >> 4) & 3) << 4)) - 32, q4 = ((L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4)) - 32;
                float *o = w + 256 * b + 128 * n;
                o[l] = d * sc[is] * q1; o[l + 32] = d * sc[is + 2] * q2; o[l + 64] = d * sc[is + 4] * q3; o[l + 96] = d * sc[is + 6] * q4;
            }
    }
}
static int bench_row_bytes(int t, int I) { return t == 8 ? I / 32 * 34 : I / 256 * (t == 12 ? 144 : t == 13 ? 176 : 210); }

int main(int argc, char **argv) {
    int S = argc > 1 ? atoi(argv[1]) : 512;
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { fprintf(stderr, "no CUDA device\n"); return 1; }
    DeviceContext *ctx = find_ctx(0);
    select_ctx(ctx);
    /* I, O: LFM3's MLP, conv in_proj, out_proj; then its K and V, a heads' gate and an odd width (4, 5, 6) */
    const int shapes[][2] = {{2048, 7680}, {7680, 2048}, {2048, 6144}, {2048, 2048}, {2048, 512}, {2048, 40}, {2048, 1001}};
    const int types[] = {12, 14, 8, 13};
    int fails = 0;
    if (getenv("BENCH_F32")) {   /* f32 weights (f32_gemm_launch: a router, DeltaNet's b and a rows): every output against doubles */
        const int fs[][2] = {{2048, 257}, {2048, 64}, {2048, 2048}, {4096, 40}};
        for (int ci = 0; ci < 4; ci++) {
            int I = fs[ci][0], O = fs[ci][1];
            std::vector<float> w((size_t)O * I), x((size_t)S * I), y((size_t)S * O);
            for (auto &v : w) v = ((int)(rnd() % 20001) - 10000) * 1e-5f;
            for (auto &v : x) v = ((int)(rnd() % 20001) - 10000) * 1e-4f;
            float *dw, *dx, *dy;
            cudaMalloc(&dw, w.size() * 4); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dy, y.size() * 4);
            cudaMemcpyAsync(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice, ctx->stream);
            cudaMemcpyAsync(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice, ctx->stream);
            cudaStreamSynchronize(ctx->stream);
            f32_gemm_launch(ctx, dy, dx, dw, S, I, O, ctx->stream);
            cudaStreamSynchronize(ctx->stream);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            double worst = 0, rms = 0; long n = 0;
            for (int r = 0; r < S; r += (S > 64 ? 7 : 1))
                for (int o = 0; o < O; o++) {
                    double ref = 0; for (int k = 0; k < I; k++) ref += (double)x[(size_t)r * I + k] * w[(size_t)o * I + k];
                    double e = fabs(y[(size_t)r * O + o] - ref); rms += ref * ref; n++; if (e > worst) worst = e;
                }
            rms = sqrt(rms / n);
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            for (int r = 0; r < 3; r++) f32_gemm_launch(ctx, dy, dx, dw, S, I, O, ctx->stream);
            cudaEventRecord(a, ctx->stream);
            for (int r = 0; r < 20; r++) f32_gemm_launch(ctx, dy, dx, dw, S, I, O, ctx->stream);
            cudaEventRecord(b, ctx->stream); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); ms /= 20;
            int bad = worst / rms > 1e-5;
            fails += bad;
            printf("f32   I %5d O %5d S %4d: %8.1f us  %6.1f TFLOP/s   worst error %.2e of the rms%s\n", I, O, S, ms * 1e3,
                   2.0 * S * I * O / (ms * 1e-3) / 1e12, worst / rms, bad ? "  BAD" : "");
            cudaFree(dw); cudaFree(dx); cudaFree(dy);
        }
        printf("bench_cuda_bdgemm: %s\n", fails ? "FAILED" : "ok");
        return fails != 0;
    }
    const char *seq = getenv("BENCH_SEQ");   /* "type.shape type.shape ...": these cases in this order */
    int cases[64][2], nc = 0;
    if (seq) { for (const char *p = seq; *p && nc < 64; ) { int a, b, k; if (sscanf(p, "%d.%d%n", &a, &b, &k) != 2) break; cases[nc][0] = a; cases[nc][1] = b; nc++; p += k; while (*p == ' ') p++; } }
    else for (int ti = 0; ti < 4; ti++) for (int si = 0; si < 4; si++) { cases[nc][0] = ti; cases[nc][1] = si; nc++; }
    for (int ci = 0; ci < nc; ci++) {
        int ti = cases[ci][0], si = cases[ci][1];
        int t = types[ti], I = shapes[si][0], O = shapes[si][1];
        int rb = bench_row_bytes(t, I);
        std::vector<uint8_t> w((size_t)rb * O);
        for (int o = 0; o < O; o++) fill_row(t, w.data() + (size_t)rb * o, I);
        std::vector<float> x((size_t)S * I), y((size_t)S * O);
        for (auto &v : x) v = ((int)(rnd() % 2001) - 1000) * 1e-3f;
        ColiCudaTensor *tn = NULL;
        if (!coli_cuda_tensor_upload(&tn, w.data(), NULL, 100 + t, I, O, 0)) { fprintf(stderr, "upload failed\n"); return 1; }
        float *dx, *dy;
        cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dy, y.size() * 4);
        /* on the backend's stream: a pageable cudaMemcpy may return before its DMA lands, and the
         * backend's stream does not wait for the default one */
        cudaMemcpyAsync(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice, ctx->stream);
        cudaStreamSynchronize(ctx->stream);
        if (!bd_tensor_gemv(ctx, tn, dy, dx, S, ctx->stream)) { fprintf(stderr, "gemm failed\n"); return 1; }
        cudaStreamSynchronize(ctx->stream);
        cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
        /* the reference on a sample of rows and outputs */
        std::vector<float> wf(I);
        double worst = 0, rms = 0; int n = 0, wo = 0, ws_ = 0; double wgot = 0, wref = 0;
        for (int o = 0; o < O; o += O / 61 + 1) {
            decode_row(t, w.data() + (size_t)rb * o, I, wf.data());
            for (int s = 0; s < S; s += S / 23 + 1) {
                double ref = 0; for (int k = 0; k < I; k++) ref += (double)x[(size_t)s * I + k] * wf[k];
                double e = y[(size_t)s * O + o] - ref; rms += ref * ref; n++;
                if (fabs(e) > worst) { worst = fabs(e); wo = o; ws_ = s; wgot = y[(size_t)s * O + o]; wref = ref; }
            }
        }
        rms = sqrt(rms / n);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int r = 0; r < 3; r++) bd_tensor_gemv(ctx, tn, dy, dx, S, ctx->stream);
        cudaEventRecord(a, ctx->stream);
        const int R = getenv("BENCH_R") ? atoi(getenv("BENCH_R")) : 20;
        for (int r = 0; r < R; r++) bd_tensor_gemv(ctx, tn, dy, dx, S, ctx->stream);
        cudaEventRecord(b, ctx->stream); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); ms /= R;
        double tf = 2.0 * S * I * O / (ms * 1e-3) / 1e12;
        int bad = worst / rms > 2e-2;
        fails += bad;
        printf("%-5s I %5d O %5d S %4d: %8.1f us  %6.1f TFLOP/s   worst error %.2e of the rms%s\n",
               t == 8 ? "Q8_0" : t == 12 ? "Q4_K" : t == 13 ? "Q5_K" : "Q6_K", I, O, S, ms * 1e3, tf, worst / rms, bad ? "  BAD" : "");
        if (bad) {
            printf("      row %d output %d: %.6f, the reference %.6f\n", ws_, wo, wgot, wref);
            if (getenv("BENCH_MAP")) {   /* which rows and outputs are off: every output of rows 0, 31, 32, 63 */
                for (int s2 = 0; s2 < S; s2 += S > 32 ? 31 : 1) {
                    int nbad = 0, first = -1;
                    for (int o = 0; o < O; o++) {
                        decode_row(t, w.data() + (size_t)rb * o, I, wf.data());
                        double ref = 0; for (int k = 0; k < I; k++) ref += (double)x[(size_t)s2 * I + k] * wf[k];
                        if (fabs(y[(size_t)s2 * O + o] - ref) > 2e-2 * rms) { nbad++; if (first < 0) first = o; }
                    }
                    printf("      row %d: %d of %d outputs off, the first %d\n", s2, nbad, O, first);
                }
            }
        }
        coli_cuda_tensor_free(tn); cudaFree(dx); cudaFree(dy);
    }
    printf("bench_cuda_bdgemm: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
