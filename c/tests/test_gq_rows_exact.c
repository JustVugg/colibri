/* test_gq_rows_exact: a speculative verify's rows get the bits of plain decoding (#2874).
 *
 * For every block type with an integer path, 16 rows of random codes and sane scales against
 * eight activation rows: gq_matmul_act with S = 1 (a decode step) for each row alone, then with
 * S = 2 .. GQ_EXACT_ROWS (a verify with drafts) on the first S rows together; every output must
 * have the same bits. And gq_rows_f8, which moe_gq_decode reads the experts with and moe_gq_run
 * now reads a verify's experts with, row alone against rows together. Above GQ_EXACT_ROWS the
 * prompt's kernels may sum in another order: not checked. The counter-proof: gq_rows_f8 against
 * the fused kernels (what a verify of up to four rows read before), which differ for some types,
 * so a row-count-dependent choice between them would show here. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../ggml_quants.h"

static uint64_t rng = 0x2874C0FFEEULL;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
/* an f16 for a positive normal value (the scales below), round to nearest */
static uint16_t f16_of(float v) {
    int e; float m = frexpf(v, &e);
    int E = e - 1 + 15; uint32_t mant = (uint32_t)lrintf((m * 2.f - 1.f) * 1024.f);
    if (mant == 1024) { mant = 0; E++; }
    return (uint16_t)((E << 10) | mant);
}
/* rows of type t with random codes and sane f16 scales (as tests/test_gq_r16.c) */
static void fill(int t, uint8_t *w, int rows, int I) {
    const int bb = GQ_TYPES[t].bytes, nb = I / GQ_TYPES[t].blck;
    for (int64_t i = 0; i < (int64_t)rows * nb * bb; i++) w[i] = (uint8_t)rnd();
    for (int64_t k = 0; k < (int64_t)rows * nb; k++) {
        uint8_t *b = w + k * bb; uint16_t d = f16_of(1e-3f + (rnd() % 1000) * 1e-5f), m = f16_of(1e-3f + (rnd() % 1000) * 1e-5f);
        switch (t) {
        case GQ_Q4_K: case GQ_Q5_K: memcpy(b, &d, 2); memcpy(b + 2, &m, 2); break;
        case GQ_Q6_K: memcpy(b + 208, &d, 2); break;
        case GQ_Q3_K: memcpy(b + 108, &d, 2); break;
        case GQ_Q2_K: memcpy(b + 80, &d, 2); memcpy(b + 82, &m, 2); break;
        case GQ_MXFP4: b[0] = (uint8_t)(118 + rnd() % 8); break;
        case GQ_IQ1_M:
            for (int i = 0; i < 4; i++) b[49 + 2 * i] = (uint8_t)((b[49 + 2 * i] & 0x0F) | (((d >> (4 * i)) & 15) << 4));
            break;
        case GQ_Q4_1: case GQ_Q5_1: memcpy(b, &d, 2); memcpy(b + 2, &m, 2); break;
        default: memcpy(b, &d, 2); break;
        }
    }
}

/* gq_act_set through a function gcc does not specialise: on the test's constant sizes its
 * constant-propagated copy warned about an iteration count no call reaches */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noipa))
#endif
static int act_set(gq_act *a, const float *x, int S, int I) { return gq_act_set(a, x, S, I); }

int main(void) {
    static const int types[] = { GQ_Q4_0, GQ_Q4_1, GQ_Q5_0, GQ_Q5_1, GQ_Q8_0, GQ_IQ4_NL, GQ_MXFP4, GQ_Q2_K, GQ_Q3_K,
                                 GQ_Q4_K, GQ_Q5_K, GQ_Q6_K, GQ_IQ4_XS, GQ_IQ2_XXS, GQ_IQ2_XS, GQ_IQ2_S, GQ_IQ3_XXS,
                                 GQ_IQ3_S, GQ_IQ1_S, GQ_IQ1_M, GQ_Q2_0 };
    enum { I = 512, O = 16, SMAX = GQ_EXACT_ROWS };
    float *x = malloc(sizeof(float) * SMAX * I), *y1 = malloc(sizeof(float) * SMAX * O), *yS = malloc(sizeof(float) * SMAX * O);
    uint8_t *w = malloc((size_t)O * I * 2);
    int bad = 0, checked = 0, f8_types = 0, f8_vs_fused = 0;
    for (size_t k = 0; k < sizeof types / sizeof types[0]; k++) {
        const int t = types[k];
        if (!gq_has_int(t) || I % GQ_TYPES[t].blck) continue;
        fill(t, w, O, I);
        for (int i = 0; i < SMAX * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) / 300.f;
        const int64_t rb = gq_row_bytes(t, I);
        gq_act a1, aS; memset(&a1, 0, sizeof a1); memset(&aS, 0, sizeof aS);
        for (int s = 0; s < SMAX; s++) {   /* each activation row alone: a decode step */
            if (act_set(&a1, x + (int64_t)s * I, 1, I)) { puts("test_gq_rows_exact: no memory"); return 1; }
            gq_matmul_act(y1 + (int64_t)s * O, &a1, w, t, O);
        }
        for (int S = 2; S <= SMAX; S++) {
            if (act_set(&aS, x, S, I)) { puts("test_gq_rows_exact: no memory"); return 1; }
            gq_matmul_act(yS, &aS, w, t, O);
            for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
                checked++;
                if (memcmp(&yS[(int64_t)s * O + o], &y1[(int64_t)s * O + o], sizeof(float))) {
                    if (bad < 6) printf("test_gq_rows_exact: %s, S = %d, row %d output %d: %.9g alone, %.9g in the verify\n",
                                        gq_type_name(t), S, s, o, y1[(int64_t)s * O + o], yS[(int64_t)s * O + o]);
                    bad++;
                }
            }
        }
        if (gq_f8_ok(t, I)) {   /* the experts' kernel: a row alone against the rows together */
            f8_types++;
            float *f1 = malloc(sizeof(float) * SMAX * O), *fS = malloc(sizeof(float) * SMAX * O);
            for (int s = 0; s < SMAX; s++) {
                act_set(&a1, x + (int64_t)s * I, 1, I);
                gq_rows_f8(f1 + (int64_t)s * O, O, &a1, w, rb, t, I, O);
            }
            act_set(&aS, x, SMAX, I);
            gq_rows_f8(fS, O, &aS, w, rb, t, I, O);
            for (int i = 0; i < SMAX * O; i++) {
                checked++;
                if (memcmp(&f1[i], &fS[i], sizeof(float))) {
                    if (bad < 6) printf("test_gq_rows_exact: %s gq_rows_f8: output %d alone %.9g, together %.9g\n", gq_type_name(t), i, f1[i], fS[i]);
                    bad++;
                }
                if (memcmp(&f1[i], &y1[i], sizeof(float))) f8_vs_fused++;
            }
            free(f1); free(fS);
        }
        gq_act_free(&a1); gq_act_free(&aS);
    }
    printf("test_gq_rows_exact: %d outputs checked, %d differ from the row alone; gq_rows_f8 differs from the fused "
           "kernels in %d outputs of its %d types (the reason a verify must not switch between them)\n",
           checked, bad, f8_vs_fused, f8_types);
    free(x); free(y1); free(yS); free(w);
    return bad ? 1 : 0;
}
