/* coli_cuda_matmul_tiled gives coli_cuda_matmul's bytes: int8 rows and grouped int4,
 * odd shapes (partial tiles of outputs and rows, columns not a multiple of 256, a last
 * int4 group shorter than the rest), 1 to 37 rows, and a row's result the same alone and
 * among others. Needs a CUDA build and a device (CUDA=1, or CUDA_DLL=1 with
 * coli_cuda.dll beside the binary); exits 77 (skipped) without a device.
 *   make tests/test_cuda_matmul_tiled.exe CUDA_DLL=1 && tests/test_cuda_matmul_tiled.exe */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../backend_cuda.h"

static uint32_t rng = 12345;
static uint32_t next(void){ rng = rng * 1664525u + 1013904223u; return rng >> 8; }

int main(void){
    int dev = 0;
    if(!coli_cuda_init(&dev, 1)){ fprintf(stderr, "no CUDA device: skipped\n"); return 77; }
    struct { int fmt, O, I, gs; } cases[] = {
        {4, 37, 7168, 64}, {4, 64, 3584, 64}, {4, 13, 1000, 64}, {4, 9, 200, 32},
        {1, 37, 7168, 0}, {1, 64, 12288, 0}, {1, 11, 300, 0},
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
        for(int S = 1; S <= 37; S += (S < 10 ? 1 : 9)){
            float *x = malloc((size_t)S * I * sizeof(float));
            float *a = malloc((size_t)S * O * sizeof(float)), *b = malloc((size_t)S * O * sizeof(float));
            float *one = malloc((size_t)O * sizeof(float));
            for(size_t i = 0; i < (size_t)S * I; i++) x[i] = ((float)(next() % 20001) - 10000.f) * 3.1e-4f;
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
    printf("%s: %d shape/row-count combinations, %d differ\n", bad ? "FAIL" : "ok", checked, bad);
    return bad ? 1 : 0;
}
