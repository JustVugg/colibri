/* The grid types' fused row kernels (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) and Q2_K's through
 * gq_row_all against the unpack path: the row as int8 (gq_unpack_row), then gq_dot_i8. The same
 * bits for every row and for one to eight activation rows, random blocks with small scales. On
 * an AVX-512 VNNI build gq_row_all takes the AVX-512 kernels (GQ_DOT_SBU256_512, GQ_DOT_SBK_512;
 * any S, four rows at a time), on an AVX2 one the AVX2 kernels for S <= 4 and the unpack for more. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ggml_quants.h"

static uint32_t rs = 0x2545F491u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

int main(void) {
#if !(defined(__AVX2__) && defined(__FMA__))
    puts("test_gq_grid512: the kernels are AVX2's and AVX-512's, not in this build: nothing to check");
    return 0;
#else
    const int types[] = { GQ_IQ2_XXS, GQ_IQ2_XS, GQ_IQ2_S, GQ_IQ3_XXS, GQ_IQ3_S, GQ_Q2_K };
    const int I = 2048, O = 64, SMAX = 8;
    float *x = malloc((size_t)SMAX * I * sizeof(float)), *y = malloc((size_t)SMAX * sizeof(float));
    int8_t *w8 = malloc((size_t)I); float *sw = malloc((size_t)(I / 16) * sizeof(float)), *mw = malloc((size_t)(I / 16) * sizeof(float));
    if (!x || !y || !w8 || !sw || !mw) { puts("FAIL alloc"); return 1; }
    for (int i = 0; i < SMAX * I; i++) x[i] = ((float)(xr() % 4001) - 2000.f) / 400.f;
    gq_act a[8]; memset(a, 0, sizeof a);
    for (int S = 1; S <= SMAX; S++) if (gq_act_set(&a[S - 1], x, S, I)) { puts("FAIL activation"); return 1; }
    int fails = 0, checked = 0;
    for (size_t k = 0; k < sizeof types / sizeof *types; k++) {
        const int t = types[k]; const int64_t rb = gq_row_bytes(t, I), bs = rb / (I / 256);   /* every type here: 256 a block */
        const int q2k = t == GQ_Q2_K;
        if (rb <= 0) { puts("FAIL row bytes"); return 1; }
        uint8_t *W = malloc((size_t)rb * O);
        if (!W) { puts("FAIL alloc"); return 1; }
        for (int64_t i = 0; i < rb * O; i++) W[i] = (uint8_t)(xr() & 0xFF);
        for (int64_t o = 0; o < O; o++)   /* the block's f16 scales (bytes 0..1; Q2_K's d and dmin at 80..83): small, finite */
            for (int64_t b = 0; b < rb; b += bs)
                for (int f = q2k ? 80 : 0; f < (q2k ? 84 : 2); f += 2) { W[o * rb + b + f] = (uint8_t)(xr() & 0xFF); W[o * rb + b + f + 1] = (uint8_t)(0x18 + xr() % 16); }
        for (int S = 1; S <= SMAX && !fails; S++) {
            const gq_act *ac = &a[S - 1];
            for (int o = 0; o < O; o++) {
                gq_row_all(y, 1, ac, W + o * rb, t, I);
                gq_unpack_row(t, W + o * rb, I, w8, sw, mw);
                for (int s = 0; s < S; s++) {
                    const float old = gq_dot_i8(w8, sw, mw, gq_has_min(t), ac->q + (int64_t)s * I, ac->d + (int64_t)s * (I / 32), ac->s + (int64_t)s * (I / 16), I);
                    if (memcmp(&y[s], &old, sizeof old)) {
                        printf("FAIL %s S=%d row %d act %d: %.9g != %.9g\n", gq_type_name(t), S, o, s, (double)y[s], (double)old);
                        fails++; break;
                    }
                }
                if (fails) break;
                checked++;
            }
        }
        free(W);
    }
    if (fails) return 1;
#ifdef GQ_VNNI512
    printf("test_gq_grid512: OK (%d rows, 6 types, S = 1..8, AVX-512 kernels)\n", checked);
#else
    printf("test_gq_grid512: OK (%d rows, 6 types, S = 1..8, AVX2 kernels)\n", checked);
#endif
    return 0;
#endif
}
