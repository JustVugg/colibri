/* test_rows_f8: a GGUF matrix's file rows eight at a time (gq_rows_f8, ggml_quants.h), every type
 * that has the kernel.
 *
 *   test_rows_f8             for each type, widths and activation rows: the kernel against a scalar
 *                            reference of its numerics (gq_ublk's integers and scales: per 32
 *                            elements, or 16 where the type's scale is per 16, the exact integer dot
 *                            and its fma) to the bit; where the type has an R16 form, against
 *                            gq_r16_rows on the packed rows to the bit; a row's result the same with
 *                            any S; the gap to the row kernels below 1e-4 of the outputs' rms.
 *                            Exit 0 on a build without AVX2, where the kernel is never taken.
 *   test_rows_f8 bench       GB/s on one core, 512 x 2048 rows in cache, against the row kernels
 *
 * Compile with the arch under test, as CI does twice (native, and x86-64-v3 for the release
 * builds without VNNI):  gcc -O3 -march=native -fopenmp -I. tests/test_rows_f8.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../ggml_quants.h"

#ifndef GQ_F8_ON
int main(void) { puts("test_rows_f8: no AVX2 in this build, nothing to check (gq_f8_ok is 0 here)"); return 0; }
#else
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ULL;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static uint16_t f16_of(float v) { return (uint16_t)_cvtss_sh(v, 0); }
static const int TYPES[] = {GQ_IQ4_XS, GQ_IQ2_XXS, GQ_IQ2_XS, GQ_IQ2_S, GQ_IQ3_XXS, GQ_IQ3_S, GQ_Q4_K, GQ_Q5_K, GQ_Q6_K, GQ_Q3_K, GQ_Q8_0};
/* rows of type t with random codes and sane f16 scales */
static void fill(int t, uint8_t *w, int rows, int I) {
    const int bb = GQ_TYPES[t].bytes, nb = I / GQ_TYPES[t].blck;
    for (int64_t i = 0; i < (int64_t)rows * nb * bb; i++) w[i] = (uint8_t)rnd();
    for (int64_t k = 0; k < (int64_t)rows * nb; k++) {
        uint8_t *b = w + k * bb; uint16_t d = f16_of(1e-3f + (rnd() % 1000) * 1e-5f), m = f16_of(1e-3f + (rnd() % 1000) * 1e-5f);
        switch (t) {
        case GQ_Q4_K: case GQ_Q5_K: memcpy(b, &d, 2); memcpy(b + 2, &m, 2); break;
        case GQ_Q6_K: memcpy(b + 208, &d, 2); break;
        case GQ_Q3_K: memcpy(b + 108, &d, 2); break;
        default: memcpy(b, &d, 2); break;
        }
    }
}
static int per16(int t) { return t == GQ_Q6_K || t == GQ_Q3_K || t == GQ_IQ2_XS || t == GQ_IQ2_S; }
/* the kernel's numerics in plain C, from gq_ublk's integers and per-16 scales and mins */
static void ref(int t, float *y, int ys, const uint8_t *w, const gq_act *a, int rows, int I) {
    const int bl = GQ_TYPES[t].blck, bb = GQ_TYPES[t].bytes, nb = I / bl;
    int8_t w8[256]; float sw[16], mw[16];
    for (int r = 0; r < rows; r++) for (int s = 0; s < a->S; s++) {
        float acc = 0.f;
        for (int b = 0; b < nb; b++) {
            gq_ublk(t, w + ((int64_t)r * nb + b) * bb, w8, sw, mw);
            for (int k = 0; k < bl / 32; k++) {
                const int g32 = (b * bl) / 32 + k; const int8_t *x = a->q + (int64_t)s * I + 32 * g32;
                const float d = a->d[(int64_t)s * (I / 32) + g32];
                int32_t p0 = 0, p1 = 0;
                for (int e = 0; e < 16; e++) { p0 += w8[32 * k + e] * x[e]; p1 += w8[32 * k + 16 + e] * x[16 + e]; }
                if (per16(t)) acc = fmaf(fmaf((float)p1, sw[2 * k + 1], (float)p0 * sw[2 * k]), d, acc);
                else {
                    acc = fmaf((float)(p0 + p1), sw[2 * k] * d, acc);
                    if (t == GQ_Q4_K || t == GQ_Q5_K) {
                        const int32_t x32 = a->s[(int64_t)s * (I / 16) + 2 * g32] + a->s[(int64_t)s * (I / 16) + 2 * g32 + 1];
                        acc = fmaf(-mw[2 * k], d * (float)x32, acc);
                    }
                }
            }
        }
        y[(int64_t)s * ys + r] = acc;
    }
}
int main(int argc, char **argv) {
    const int NT = (int)(sizeof TYPES / sizeof *TYPES);
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        const int R = 512, I = 2048;
        float *x = malloc((size_t)I * 4), *y = malloc((size_t)R * 4);
        for (int i = 0; i < I; i++) x[i] = sinf(i * 0.37f);
        gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, 1, I);
        for (int k = 0; k < NT; k++) {
            const int t = TYPES[k]; const int64_t rb = gq_row_bytes(t, I);
            uint8_t *w = malloc((size_t)rb * R); fill(t, w, R, I);
            double best[2] = {1e9, 1e9};
            for (int rep = 0; rep < 200; rep++)
                for (int m = 0; m < 2; m++) {
                    double t0 = now();
                    if (m) gq_rows_f8(y, R, &a, w, rb, t, I, R);
                    else for (int r = 0; r < R; r++) gq_row_all(y + r, R, &a, w + rb * r, t, I);
                    double dt = now() - t0; if (dt < best[m]) best[m] = dt;
                }
            printf("%-8s rows %6.1f GB/s   by eight %6.1f GB/s   %.2fx\n", gq_type_name(t), rb * R / best[0] / 1e9, rb * R / best[1] / 1e9, best[0] / best[1]);
            free(w);
        }
        return 0;
    }
    int fails = 0;
    for (int k = 0; k < NT; k++) {
        const int t = TYPES[k], bl = GQ_TYPES[t].blck, t16 = gq_r16_of(t);
        int bad = 0, bad16 = 0, badS = 0, n = 0; double worst = 0;
        for (int I = 256; I <= 2560; I += 768) for (int S = 1; S <= 9; S += S < 5 ? 1 : 4) {
            const int R = 48; const int64_t rb = gq_row_bytes(t, I);
            if (I % bl || !gq_f8_ok(t, I)) continue;
            uint8_t *src = malloc((size_t)rb * R);
            fill(t, src, R, I);
            float *x = malloc((size_t)S * I * 4); for (int i = 0; i < S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
            for (int i = 0; i < S; i++) x[(int64_t)i * I] *= 40.f;   /* a large activation, as real hidden states have */
            gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, I);
            float *y0 = malloc((size_t)S * R * 4), *y1 = malloc((size_t)S * R * 4), *y2 = malloc((size_t)S * R * 4), *y3 = malloc((size_t)S * R * 4);
            ref(t, y0, R, src, &a, R, I);
            gq_rows_f8(y1, R, &a, src, rb, t, I, R);
            for (int r = 0; r < R; r++) gq_row_all(y3 + r, R, &a, src + rb * r, t, I);
            n++;
            if (memcmp(y0, y1, (size_t)S * R * 4)) { bad++; if (bad < 3) printf("  %s I=%d S=%d: %.9g against %.9g\n", gq_type_name(t), I, S, y1[0], y0[0]); }
            const int64_t hb = t16 >= 0 ? gq_row_bytes(t16, I) : 0;
            if (hb > 0) {   /* the packed rows' kernel: the same bits */
                uint8_t *g = malloc((size_t)hb * R);
                gq_r16_pack(t16, g, src, R, I);
                for (int gi = 0; gi < R / 16; gi++) gq_r16_rows(t16, y2 + 16 * gi, R, g + (size_t)hb * 16 * gi, &a, I);
                if (memcmp(y1, y2, (size_t)S * R * 4)) { bad16++; if (bad16 < 3) printf("  %s I=%d S=%d: %.9g, R16 %.9g\n", gq_type_name(t), I, S, y1[0], y2[0]); }
                free(g);
            }
            for (int s = 0; s < S; s++) {   /* each activation row alone: its own bits */
                gq_act one; memset(&one, 0, sizeof one); gq_act_set(&one, x + (int64_t)s * I, 1, I);
                gq_rows_f8(y2, R, &one, src, rb, t, I, R);
                badS += memcmp(y2, y1 + (int64_t)s * R, (size_t)R * 4) != 0;
                gq_act_free(&one);
            }
            double rms = 0; for (int i = 0; i < S * R; i++) rms += (double)y3[i] * y3[i]; rms = sqrt(rms / (S * R)) + 1e-30;
            for (int i = 0; i < S * R; i++) { double e = fabs((double)y3[i] - y1[i]) / rms; if (e > worst) worst = e; }
            free(src); free(x); free(y0); free(y1); free(y2); free(y3); gq_act_free(&a);
        }
        printf("%-8s %d of %d cases differ from the reference%s; %d rows depend on S; worst gap to the row kernels %.2e of the rms\n",
               gq_type_name(t), bad, n, t16 < 0 ? "" : bad16 ? ", some from R16" : ", none from R16", badS, worst);
        fails += bad + bad16 + badS + (worst > 1e-4) + (n == 0);
    }
    printf("test_rows_f8: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
#endif
