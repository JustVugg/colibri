/* matmul_mxfp4_i8_pre and matmul_i4_grouped give several rows the bytes they give each
 * of them alone: their multi-row paths (an output's weights decoded once for 4 or 8 rows,
 * matmul_i4_grouped's rows 16 at a time) against one call a row, for 1 to 37 rows (whole
 * and partial blocks), Kimi K3's shapes (experts 3584 -> 3072 and back, the shared experts'
 * and latent projections' 7168 columns in groups of 64) and small ones.
 *   make tests/test_mxfp4_i8_rows.exe && tests/test_mxfp4_i8_rows.exe */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../quant.h"

static uint32_t rng = 2026;
static uint32_t next(void){ rng = rng * 1664525u + 1013904223u; return rng >> 8; }

int main(void){
    struct { int I, O; } shapes[] = { {32, 1}, {64, 7}, {3584, 64}, {3072, 37}, {96, 130} };
    int bad = 0, checked = 0;
    for(size_t k = 0; k < sizeof shapes / sizeof shapes[0]; k++){
        int I = shapes[k].I, O = shapes[k].O, ng = I / 32, rb = I / 2;
        uint8_t *q4 = malloc((size_t)O * rb), *e8 = malloc((size_t)O * ng);
        for(size_t i = 0; i < (size_t)O * rb; i++) q4[i] = (uint8_t)next();
        for(size_t i = 0; i < (size_t)O * ng; i++) e8[i] = (uint8_t)(120 + next() % 14);
        for(int S = 1; S <= 13; S++){
            int8_t *xq = malloc((size_t)S * I); float *xs = malloc((size_t)S * ng * sizeof(float));
            float *all = malloc((size_t)S * O * sizeof(float)), *one = malloc((size_t)O * sizeof(float));
            for(size_t i = 0; i < (size_t)S * I; i++) xq[i] = (int8_t)((int)(next() % 255) - 127);
            for(size_t i = 0; i < (size_t)S * ng; i++) xs[i] = (float)(next() % 1000 + 1) * 1.7e-4f;
            matmul_mxfp4_i8_pre(all, xq, xs, q4, e8, S, I, O);
            for(int s = 0; s < S; s++){
                matmul_mxfp4_i8_pre(one, xq + (size_t)s * I, xs + (size_t)s * ng, q4, e8, 1, I, O);
                if(memcmp(one, all + (size_t)s * O, (size_t)O * sizeof(float))){
                    fprintf(stderr, "I=%d O=%d S=%d: row %d differs from its call alone\n", I, O, S, s);
                    bad++;
                    break;
                }
            }
            checked++;
            free(xq); free(xs); free(all); free(one);
        }
        free(q4); free(e8);
    }
    struct { int I, O, gs; } i4[] = { {7168, 33, 64}, {3584, 17, 64}, {64, 5, 64}, {96, 3, 32}, {160, 7, 16}, {96, 4, 48} };
    for(size_t k = 0; k < sizeof i4 / sizeof i4[0]; k++){
        int I = i4[k].I, O = i4[k].O, gs = i4[k].gs, ng = (I + gs - 1) / gs, rb = (I + 1) / 2;
        uint8_t *q4 = malloc((size_t)O * rb); float *sc = malloc((size_t)O * ng * sizeof(float));
        for(size_t i = 0; i < (size_t)O * rb; i++) q4[i] = (uint8_t)next();
        for(size_t i = 0; i < (size_t)O * ng; i++) sc[i] = (float)(next() % 1000 + 1) * 1e-4f;
        for(int S = 1; S <= 37; S += S < 10 ? 1 : 9){
            float *x = malloc((size_t)S * I * sizeof(float));
            float *all = malloc((size_t)S * O * sizeof(float)), *one = malloc((size_t)O * sizeof(float));
            for(size_t i = 0; i < (size_t)S * I; i++) x[i] = ((float)(next() % 20001) - 10000.f) * 3.1e-4f;
            matmul_i4_grouped(all, x, q4, sc, S, I, O, gs);
            for(int s = 0; s < S; s++){
                matmul_i4_grouped(one, x + (size_t)s * I, q4, sc, 1, I, O, gs);
                if(memcmp(one, all + (size_t)s * O, (size_t)O * sizeof(float))){
                    fprintf(stderr, "i4 I=%d O=%d gs=%d S=%d: row %d differs from its call alone\n", I, O, gs, S, s);
                    bad++;
                    break;
                }
            }
            checked++;
            free(x); free(all); free(one);
        }
        free(q4); free(sc);
    }
    printf("%s: %d shape/row-count combinations, %d differ\n", bad ? "FAIL" : "ok", checked, bad);
    return bad ? 1 : 0;
}
