/* Q2_K's fused row kernel (ggml_quants.h gq_dot_q2_k, GQ_DOT_SBK) against the path it
 * replaced for one to four activation rows: the row unpacked to int8 (gq_unpack_row), then
 * gq_dot_i8. The same bits for every row and S, random blocks with small scales; gq_matmul
 * through the new kernel too. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ggml_quants.h"

static uint32_t rs = 0x9E3779B9u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

int main(void) {
#if !(defined(__AVX2__) && defined(__FMA__))
    puts("test_gq_q2_k: the kernel is AVX2's, not in this build: nothing to check");
    return 0;
#else
    const int I = 2048, O = 96;
    const int64_t rb = gq_row_bytes(GQ_Q2_K, I);
    uint8_t *W = malloc((size_t)rb * O);
    float *x = malloc((size_t)4 * I * sizeof(float)), *y = malloc((size_t)4 * O * sizeof(float)), *ym = malloc((size_t)4 * O * sizeof(float));
    int8_t *w8 = malloc((size_t)I); float *sw = malloc((size_t)(I / 16) * sizeof(float)), *mw = malloc((size_t)(I / 16) * sizeof(float));
    if (!W || !x || !y || !ym || !w8 || !sw || !mw) { puts("FAIL alloc"); return 1; }
    for (int64_t i = 0; i < rb * O; i++) W[i] = (uint8_t)(xr() & 0xFF);
    for (int64_t o = 0; o < O; o++)   /* d and dmin (bytes 80..83 of a block): small, finite */
        for (int64_t b = 0; b < rb; b += 84) for (int k = 80; k < 84; k += 2) { W[o * rb + b + k] = (uint8_t)(xr() & 0xFF); W[o * rb + b + k + 1] = (uint8_t)(0x18 + xr() % 16); }
    for (int i = 0; i < 4 * I; i++) x[i] = ((float)(xr() % 4001) - 2000.f) / 400.f;
    gq_act a[4]; memset(a, 0, sizeof a);
    for (int S = 1; S <= 4; S++) if (gq_act_set(&a[S - 1], x, S, I)) { puts("FAIL activation"); return 1; }
    int fails = 0, checked = 0;
    for (int S = 1; S <= 4; S++) {
        const gq_act *ac = &a[S - 1];
        for (int o = 0; o < O; o++) {
            float now[4], old[4];
            gq_dot_q2_k(now, 1, W + o * rb, ac->q, ac->d, ac->s, I, S);
            gq_unpack_row(GQ_Q2_K, W + o * rb, I, w8, sw, mw);
            for (int s = 0; s < S; s++)
                old[s] = gq_dot_i8(w8, sw, mw, 1, ac->q + (int64_t)s * I, ac->d + (int64_t)s * (I / 32), ac->s + (int64_t)s * (I / 16), I);
            if (memcmp(now, old, (size_t)S * sizeof(float))) {
                printf("FAIL Q2_K S=%d row %d: %.9g != %.9g\n", S, o, (double)now[0], (double)old[0]); fails++; break;
            }
            checked++;
        }
        /* the whole matrix through gq_matmul: row o of activation row s is the old path's */
        if (gq_matmul(ym, x, W, GQ_Q2_K, S, I, O)) { puts("FAIL gq_matmul"); return 1; }
        for (int o = 0; o < O && !fails; o++) {
            gq_unpack_row(GQ_Q2_K, W + o * rb, I, w8, sw, mw);
            for (int s = 0; s < S; s++) {
                const float old = gq_dot_i8(w8, sw, mw, 1, ac->q + (int64_t)s * I, ac->d + (int64_t)s * (I / 32), ac->s + (int64_t)s * (I / 16), I);
                if (memcmp(&ym[(int64_t)s * O + o], &old, sizeof old)) {
                    printf("FAIL gq_matmul Q2_K S=%d row %d: %.9g != %.9g\n", S, o, (double)ym[(int64_t)s * O + o], (double)old); fails++; break;
                }
            }
        }
    }
    if (fails) return 1;
    printf("test_gq_q2_k: OK (%d rows, S = 1..4, and gq_matmul)\n", checked);
    return 0;
#endif
}
