/* coli_cuda_matmul_tiled gives coli_cuda_matmul's bytes: int8 rows and grouped int4,
 * odd shapes (partial tiles of outputs and rows, columns not a multiple of 256, a last
 * int4 group shorter than the rest), 1 to 37 rows, and a row's result the same alone and
 * among others. One row (coli_cuda_matmul's matvec kernels: quant_matvec_warp for rows
 * of whole 256-column chunks, quant_matvec_rows for the rest) is also checked against
 * quant_matmul's arithmetic done on the CPU, at Kimi K3's trunk shapes and odd ones;
 * COLI_CUDA_QMR=0 checks quant_matmul itself against it. Kimi K3's trunk shapes (7168 or
 * 12288 columns, outputs a multiple of 16) take quant_matmul_blocked, the others
 * quant_matmul_tiled; the inputs hold zeros of both signs, subnormals and large values.
 * Needs a CUDA build and a device (CUDA=1, or CUDA_DLL=1 with
 * coli_cuda.dll beside the binary); exits 77 (skipped) without a device.
 *   make tests/test_cuda_matmul_tiled.exe CUDA_DLL=1 && tests/test_cuda_matmul_tiled.exe */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../backend_cuda.h"

static uint32_t rng = 12345;
static uint32_t next(void){ rng = rng * 1664525u + 1013904223u; return rng >> 8; }

/* quant_matmul's output o for one row: thread t sums the columns t, t + 256, ... (int4:
 * the product rounded, then a fused add of it times the group's scale; int8: a fused add
 * of the product), then the tree over the 256 partial sums, then int8's row scale. The
 * device holds int4 as two's complement nibbles (offset_to_signed_s4): n - 8 here. */
static float quant_matmul_out(const float *x, const uint8_t *w, const float *sc, int fmt,
                              int I, int o, size_t rb, int gs, int ng){
    float part[256];
    const uint8_t *row = w + (size_t)o * rb;
    for(int t = 0; t < 256; t++){
        float s = 0.0f;
        for(int i = t; i < I; i += 256){
            if(fmt == 4){
                int n = (i & 1) ? row[i >> 1] >> 4 : row[i >> 1] & 15, g = i / gs;
                if(g >= ng) g = ng - 1;
                float p = x[i] * (float)(n - 8);
                s = fmaf(p, sc[(size_t)o * ng + g], s);
            } else s = fmaf(x[i], (float)(int8_t)row[i], s);
        }
        part[t] = s;
    }
    for(int n = 128; n; n >>= 1) for(int t = 0; t < n; t++) part[t] += part[t + n];
    return fmt == 1 ? part[0] * sc[o] : part[0];
}

int main(void){
    int dev = 0;
    if(!coli_cuda_init(&dev, 1)){ fprintf(stderr, "no CUDA device: skipped\n"); return 77; }
    struct { int fmt, O, I, gs; } cases[] = {
        {4, 37, 7168, 64}, {4, 64, 3584, 64}, {4, 13, 1000, 64}, {4, 9, 200, 32},
        {1, 37, 7168, 0}, {1, 64, 12288, 0}, {1, 11, 300, 0},
        {4, 48, 7168, 64}, {4, 32, 12288, 64}, {1, 48, 7168, 0}, {1, 32, 12288, 0},
    };
    int bad = 0, checked = 0;
    for(size_t c = 0; c < sizeof cases / sizeof cases[0]; c++){
        int fmt = cases[c].fmt, O = cases[c].O, I = cases[c].I, gs = cases[c].gs;
        int ng = gs ? (I + gs - 1) / gs : 1;
        size_t rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
        size_t sn = fmt == 1 ? (size_t)O : (size_t)O * ng;
        uint8_t *w = malloc(rb * O); float *sc = malloc(sn * sizeof(float));
        for(size_t i = 0; i < rb * O; i++) w[i] = (uint8_t)next();
        for(size_t i = 0; i < sn; i++) sc[i] = (float)(next() % 1000 + 1) * 1e-4f;
        ColiCudaTensor *t = NULL;
        for(int S = 1; S <= 101; S += (S < 10 ? 1 : S < 37 ? 9 : 64)){
            float *x = malloc((size_t)S * I * sizeof(float));
            float *a = malloc((size_t)S * O * sizeof(float)), *b = malloc((size_t)S * O * sizeof(float));
            float *one = malloc((size_t)O * sizeof(float));
            for(size_t i = 0; i < (size_t)S * I; i++){
                uint32_t r = next();
                x[i] = r % 997 == 0 ? 0.0f : r % 991 == 0 ? -0.0f : r % 983 == 0 ? 1e-39f * (float)(r % 7)
                     : r % 977 == 0 ? 3e4f : ((float)(r % 20001) - 10000.f) * 3.1e-4f;
            }
            if(!coli_cuda_matmul(&t, a, x, w, sc, fmt, S, I, O, dev, gs) ||
               !coli_cuda_matmul_tiled(&t, b, x, w, sc, fmt, S, I, O, dev, gs)){
                fprintf(stderr, "case %zu S=%d: a call failed\n", c, S); return 1;
            }
            if(memcmp(a, b, (size_t)S * O * sizeof(float))){
                int first = -1;
                for(int i = 0; i < S * O; i++) if(memcmp(&a[i], &b[i], 4)){ first = i; break; }
                fprintf(stderr, "fmt %d O=%d I=%d S=%d: differs first at row %d out %d (%.9g vs %.9g)\n",
                        fmt, O, I, S, first / O, first % O, a[first], b[first]);
                bad++;
            }
            /* the last row alone through the tiled kernel: the bytes it had among the others */
            if(!coli_cuda_matmul_tiled(&t, one, x + (size_t)(S - 1) * I, w, sc, fmt, 1, I, O, dev, gs)){
                fprintf(stderr, "case %zu: a call failed\n", c); return 1;
            }
            if(memcmp(one, b + (size_t)(S - 1) * O, (size_t)O * sizeof(float))){
                fprintf(stderr, "fmt %d O=%d I=%d S=%d: the last row alone differs\n", fmt, O, I, S);
                bad++;
            }
            checked++;
            free(x); free(a); free(b); free(one);
        }
        coli_cuda_tensor_free(t);
        free(w); free(sc);
    }
    struct { int fmt, O, I, gs; } one_row[] = {
        {4, 12288, 7168, 64}, {4, 7168, 12288, 64}, {1, 7168, 12288, 0}, {4, 3584, 7168, 64},
        {1, 1, 7168, 0}, {4, 1, 7168, 64}, {4, 15, 1000, 48}, {4, 9, 999, 64}, {1, 23, 257, 0},
        {4, 130, 3000, 128}, {4, 5, 100, 32}, {1, 17, 100, 0}, {4, 13, 512, 32},
        {4, 11, 768, 128}, {1, 9, 256, 0}, {4, 3, 256, 64}, {4, 21, 1536, 64}, {1, 6145, 7168, 0},
    };
    for(size_t c = 0; c < sizeof one_row / sizeof one_row[0]; c++){
        int fmt = one_row[c].fmt, O = one_row[c].O, I = one_row[c].I, gs = one_row[c].gs;
        int ng = gs ? (I + gs - 1) / gs : 1;
        size_t rb = fmt == 1 ? (size_t)I : (size_t)(I + 1) / 2;
        size_t sn = fmt == 1 ? (size_t)O : (size_t)O * ng;
        uint8_t *w = malloc(rb * O); float *sc = malloc(sn * sizeof(float));
        float *x = malloc((size_t)I * sizeof(float));
        float *a = malloc((size_t)O * sizeof(float)), *b = malloc((size_t)O * sizeof(float));
        for(size_t i = 0; i < rb * O; i++) w[i] = (uint8_t)next();
        for(size_t i = 0; i < sn; i++) sc[i] = (float)(next() % 1000 + 1) * 1e-4f;
        for(int i = 0; i < I; i++) x[i] = ((float)(next() % 20001) - 10000.f) * 3.1e-4f;
        ColiCudaTensor *t = NULL;
        if(!coli_cuda_matmul(&t, a, x, w, sc, fmt, 1, I, O, dev, gs) ||
           !coli_cuda_matmul_tiled(&t, b, x, w, sc, fmt, 1, I, O, dev, gs)){
            fprintf(stderr, "one row, case %zu: a call failed\n", c); return 1;
        }
        int off = 0, off_tiled = 0;
        for(int o = 0; o < O; o++){
            float e = quant_matmul_out(x, w, sc, fmt, I, o, rb, gs, ng);
            off += memcmp(&e, &a[o], 4) != 0;
            off_tiled += memcmp(&a[o], &b[o], 4) != 0;
        }
        if(off || off_tiled){
            fprintf(stderr, "fmt %d O=%d I=%d gs=%d, one row: %d outputs differ from quant_matmul's "
                    "arithmetic, %d from the tiled kernel's\n", fmt, O, I, gs, off, off_tiled);
            bad++;
        }
        checked++;
        coli_cuda_tensor_free(t);
        free(w); free(sc); free(x); free(a); free(b);
    }
    printf("%s: %d shape/row-count combinations, %d differ\n", bad ? "FAIL" : "ok", checked, bad);
    return bad ? 1 : 0;
}
