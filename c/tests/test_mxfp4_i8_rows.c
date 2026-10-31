/* matmul_mxfp4_i8_pre gives several rows the bytes it gives each of them alone: the
 * multi-row path (an output's weights decoded once for 4 rows) against one call a row,
 * for 1 to 13 rows (whole and partial blocks of 4), Kimi K3's expert shapes (3584 -> 3072
 * and 3072 -> 3584) and small ones, every e2m1 code and scales around 2^0.
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
    printf("%s: %d shape/row-count combinations, %d differ\n", bad ? "FAIL" : "ok", checked, bad);
    return bad ? 1 : 0;
}
