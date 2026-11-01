/* test_tile16: a prompt's rows of a type with no R16 form through 16 file rows unpacked once
 * (gq_tile16_pack, gq_tile16_rows: ggml_quants.h), every type it takes.
 *
 *   test_tile16          for each type, widths and activation rows: the kernel against a scalar
 *                        reference of its numerics (gq_ublk's integers and scale per 16: per 16
 *                        elements the exact integer dot, then its fmas) to the bit; a row's result
 *                        the same with any S; the gap to the row kernels below 1e-4 of the outputs'
 *                        rms; and gq_matmul_act, which takes the tile from GQ_TILE16_MIN_S rows on,
 *                        the same bits as the tile on every whole group of 16 rows.
 *                        Exit 0 on a build without the R16 kernels, where the tile is never taken.
 *   test_tile16 bench    rows a second on one core, 512 x 2048 against 32 activation rows: the row
 *                        kernels, gq_rows_f8 where the type has it, and the tile
 *
 * Compile with the arch under test, as CI does twice (native, and x86-64-v3 for the release
 * builds without VNNI):  gcc -O3 -march=native -fopenmp -I. tests/test_tile16.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../ggml_quants.h"

#ifndef GQ_TILE16_BIAS
int main(void) { puts("test_tile16: no R16 kernels in this build, nothing to check (the tile is never taken)"); return 0; }
#else
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ULL;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static uint16_t f16_of(float v) { return (uint16_t)_cvtss_sh(v, 0); }
static const int TYPES[] = {GQ_IQ2_XXS, GQ_IQ2_XS, GQ_IQ2_S, GQ_IQ3_XXS, GQ_IQ3_S, GQ_IQ1_S, GQ_IQ1_M};
/* rows of type t with random codes and sane f16 scales (IQ1_M keeps its scale in the
 * sub-block bytes: random ones are what it reads) */
static void fill(int t, uint8_t *w, int rows, int I) {
    const int bb = GQ_TYPES[t].bytes, nb = I / GQ_TYPES[t].blck;
    for (int64_t i = 0; i < (int64_t)rows * nb * bb; i++) w[i] = (uint8_t)rnd();
    if (t == GQ_IQ1_M) return;
    for (int64_t k = 0; k < (int64_t)rows * nb; k++) {
        uint16_t d = f16_of(1e-3f + (rnd() % 1000) * 1e-5f);
        memcpy(w + k * bb, &d, 2);
    }
}
/* the tile's numerics in plain C: per 32 elements p0, p1 the exact integer dots of the two
 * halves, then fma(fma(p1, sv1, p0 * sv0), d, acc), the R16 kernels' order */
static void ref(int t, float *y, int ys, const uint8_t *w, const gq_act *a, int rows, int I) {
    const int64_t rb = gq_row_bytes(t, I);
    int8_t *w8 = malloc((size_t)I); float *sw = malloc((size_t)(I / 16) * sizeof(float)), *mw = malloc((size_t)(I / 16) * sizeof(float));
    for (int r = 0; r < rows; r++) {
        gq_unpack_row(t, w + rb * r, I, w8, sw, mw);
        for (int s = 0; s < a->S; s++) {
            float acc = 0.f;
            for (int g32 = 0; g32 < I / 32; g32++) {
                const int8_t *x = a->q + (int64_t)s * I + 32 * g32;
                int32_t p0 = 0, p1 = 0;
                for (int e = 0; e < 16; e++) { p0 += w8[32 * g32 + e] * x[e]; p1 += w8[32 * g32 + 16 + e] * x[16 + e]; }
                const float t_ = fmaf((float)p1, sw[2 * g32 + 1], (float)p0 * sw[2 * g32]);
                acc = fmaf(t_, a->d[(int64_t)s * (I / 32) + g32], acc);
            }
            y[(int64_t)s * ys + r] = acc;
        }
    }
    free(w8); free(sw); free(mw);
}
/* the tile kernel on R rows (a multiple of 16) */
static void tiled(int t, float *y, int ys, const uint8_t *w, const gq_act *a, int R, int I) {
    const int64_t rb = gq_row_bytes(t, I);
    uint8_t *tile = malloc(gq_tile16_bytes(I)); int8_t *w8 = malloc((size_t)I); float *sw = malloc((size_t)(I / 16) * 2 * sizeof(float));
    for (int g = 0; g < R / 16; g++) {
        gq_tile16_pack(tile, w + rb * 16 * g, rb, t, I, w8, sw, sw + I / 16);
        gq_tile16_rows(y + 16 * g, ys, tile, a, I);
    }
    free(tile); free(w8); free(sw);
}
int main(int argc, char **argv) {
    const int NT = (int)(sizeof TYPES / sizeof *TYPES);
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        const int R = 512, I = 2048, S = 32;
        float *x = malloc((size_t)S * I * 4), *y = malloc((size_t)S * R * 4);
        for (int i = 0; i < S * I; i++) x[i] = sinf(i * 0.37f);
        gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, I);
        for (int k = 0; k < NT; k++) {
            const int t = TYPES[k]; const int64_t rb = gq_row_bytes(t, I);
            uint8_t *w = malloc((size_t)rb * R); fill(t, w, R, I);
            double best[3] = {1e9, 1e9, 1e9};
            for (int rep = 0; rep < 20; rep++)
                for (int m = 0; m < 3; m++) {
                    if (m == 1 && !gq_f8_ok(t, I)) continue;
                    double t0 = now();
                    if (m == 0) for (int r = 0; r < R; r++) gq_row_all(y + r, R, &a, w + rb * r, t, I);
                    else if (m == 1) gq_rows_f8(y, R, &a, w, rb, t, I, R);
                    else tiled(t, y, R, w, &a, R, I);
                    double dt = now() - t0; if (dt < best[m]) best[m] = dt;
                }
            printf("%-8s rows %8.0f/s   by eight %8.0f/s   tile %8.0f/s   %.2fx the rows\n", gq_type_name(t),
                   R / best[0], best[1] < 1e9 ? R / best[1] : 0.0, R / best[2], best[0] / best[2]);
            free(w);
        }
        return 0;
    }
    int fails = 0;
    for (int k = 0; k < NT; k++) {
        const int t = TYPES[k], bl = GQ_TYPES[t].blck;
        int bad = 0, badS = 0, badM = 0, n = 0; double worst = 0;
        for (int I = 256; I <= 2560; I += 768) for (int S = 1; S <= 21; S += S < 9 ? 1 : 6) {
            const int R = 48, O = R + 5; const int64_t rb = gq_row_bytes(t, I);
            if (I % bl || !gq_tile16_ok(t, I)) continue;
            uint8_t *src = malloc((size_t)rb * O);
            fill(t, src, O, I);
            float *x = malloc((size_t)S * I * 4); for (int i = 0; i < S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
            for (int i = 0; i < S; i++) x[(int64_t)i * I] *= 40.f;   /* a large activation, as real hidden states have */
            gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, I);
            float *y0 = malloc((size_t)S * R * 4), *y1 = malloc((size_t)S * R * 4), *y2 = malloc((size_t)S * R * 4), *y3 = malloc((size_t)S * R * 4);
            float *ym = malloc((size_t)S * O * 4);
            ref(t, y0, R, src, &a, R, I);
            tiled(t, y1, R, src, &a, R, I);
            for (int r = 0; r < R; r++) gq_row_all(y3 + r, R, &a, src + rb * r, t, I);
            n++;
            if (memcmp(y0, y1, (size_t)S * R * 4)) { bad++; if (bad < 3) printf("  %s I=%d S=%d: %.9g against %.9g\n", gq_type_name(t), I, S, y1[0], y0[0]); }
            for (int s = 0; s < S; s++) {   /* each activation row alone: its own bits */
                gq_act one; memset(&one, 0, sizeof one); gq_act_set(&one, x + (int64_t)s * I, 1, I);
                tiled(t, y2, R, src, &one, R, I);
                badS += memcmp(y2, y1 + (int64_t)s * R, (size_t)R * 4) != 0;
                gq_act_free(&one);
            }
            /* gq_matmul_act over O rows (three whole groups and five more): the tile's bits on
             * the groups from GQ_TILE16_MIN_S activation rows on */
            gq_matmul_act(ym, &a, src, t, O);
            if (S >= GQ_TILE16_MIN_S)
                for (int s = 0; s < S; s++) badM += memcmp(ym + (int64_t)s * O, y1 + (int64_t)s * R, (size_t)R * 4) != 0;
            double rms = 0; for (int i = 0; i < S * R; i++) rms += (double)y3[i] * y3[i]; rms = sqrt(rms / (S * R)) + 1e-30;
            for (int i = 0; i < S * R; i++) { double e = fabs((double)y3[i] - y1[i]) / rms; if (e > worst) worst = e; }
            free(src); free(x); free(y0); free(y1); free(y2); free(y3); free(ym); gq_act_free(&a);
        }
        printf("%-8s %d of %d cases differ from the reference; %d rows depend on S; %d rows of gq_matmul_act differ; "
               "worst gap to the row kernels %.2e of the rms\n", gq_type_name(t), bad, n, badS, badM, worst);
        fails += bad + badS + badM + (worst > 1e-4) + (n == 0);
    }
    printf("test_tile16: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
#endif
