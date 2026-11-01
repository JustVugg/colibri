/* test_gq_r16: the row-interleaved forms (GQ_*_R16, ggml_quants.h) of every block type that has one.
 *
 *   test_gq_r16              for each type: pack then unpack gives the file's rows back, and so
 *                            does the row reader; the kernel against a scalar reference of its
 *                            numerics (gq_ublk's integers and scales: per 32 elements, or 16 where
 *                            the type's scale is per 16, the exact integer dot and one fma), to
 *                            the bit; the gap to the row kernels (the same integers, other
 *                            roundings) below 1e-4 of the outputs' rms. Exit 0 on a build
 *                            without AVX2, where the forms are never made.
 *   test_gq_r16 bench [S] [reps]   ms per 2560 x 2560 matrix (48 of them, past the caches),
 *                            interleaved against the row kernels, S activation rows */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../ggml_quants.h"

#ifndef GQ_R16_ON
int main(void) { puts("test_gq_r16: no AVX2 in this build, nothing to check (the forms are made only where their kernels are)"); return 0; }
#else
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ULL;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static uint16_t f16_of(float v) { return (uint16_t)_cvtss_sh(v, 0); }
/* rows of base type t with random codes and sane f16 scales */
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
/* the kernels' numerics in plain C, from gq_ublk's integers and per-16 scales and mins */
static void ref(int t, float *y, int ys, const uint8_t *w, const gq_act *a, int rows, int I) {
    const int bl = GQ_TYPES[t].blck, bb = GQ_TYPES[t].bytes, nb = I / bl, per16 = t == GQ_Q6_K || t == GQ_Q3_K;
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
                if (per16) acc = fmaf(fmaf((float)p1, sw[2 * k + 1], (float)p0 * sw[2 * k]), d, acc);
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
    if (argc < 2 || strcmp(argv[1], "bench")) {
        int fails = 0;
        for (int t16 = GQ_Q2_0_R16; t16 < GQ_R16_END; t16++) {
            const int t = gq_r16_base(t16), bl = GQ_TYPES[t].blck; int bad = 0, n = 0, trip = 0; double worst = 0;
            for (int I = 256; I <= 2560; I += 768) for (int S = 1; S <= 19; S += S < 9 ? 1 : 5) {
                const int R = 48; const int64_t rb = gq_row_bytes(t, I), hb = gq_row_bytes(t16, I);
                if (I % bl) continue;
                uint8_t *src = malloc((size_t)rb * R), *g = malloc((size_t)hb * R), *back = malloc((size_t)rb * R);
                fill(t, src, R, I);
                float *x = malloc((size_t)S * I * 4); for (int i = 0; i < S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
                gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, I);
                gq_r16_pack(t16, g, src, R, I); gq_r16_unpack(t16, back, g, R, I);
                trip += memcmp(back, src, (size_t)rb * R) != 0;
                float *y1 = malloc((size_t)S * R * 4), *y2 = malloc((size_t)S * R * 4), *y3 = malloc((size_t)S * R * 4), *row = malloc((size_t)I * 4), *row2 = malloc((size_t)I * 4);
                for (int r = 0; r < R; r++) {   /* the row reader */
                    gq_dequant_row(t, src + (int64_t)r * rb, row, I); gq_r16_dequant_row(t16, g, r, row2, I);
                    if (memcmp(row, row2, (size_t)I * 4)) trip++;
                }
                ref(t, y1, R, src, &a, R, I); gq_matmul_act(y2, &a, g, t16, R); gq_matmul_act(y3, &a, src, t, R);
                double rms = 0; for (int i = 0; i < S * R; i++) rms += (double)y3[i] * y3[i]; rms = sqrt(rms / (S * R)) + 1e-30;
                for (int i = 0; i < S * R; i++) { double e = fabs((double)y3[i] - y2[i]) / rms; if (e > worst) worst = e; }
                n++; if (memcmp(y1, y2, (size_t)S * R * 4)) { bad++; if (bad < 3) printf("  %s I=%d S=%d: %.9g against %.9g\n", gq_type_name(t16), I, S, y2[0], y1[0]); }
                free(src); free(g); free(back); free(x); free(y1); free(y2); free(y3); free(row); free(row2); gq_act_free(&a);
            }
            printf("%-10s %d of %d cases differ from the reference; round trip %s; worst gap to the row kernels %.2e of the rms\n",
                   gq_type_name(t16), bad, n, trip ? "BROKEN" : "exact", worst);
            fails += bad + trip + (worst > 1e-4);
        }
        printf("test_gq_r16: %s\n", fails ? "FAILED" : "ok");
        return fails != 0;
    }
    const int S = argc > 2 ? atoi(argv[2]) : 1, reps = argc > 3 ? atoi(argv[3]) : 5, M = 48, D = 2560;
    float *x = malloc((size_t)S * D * 4), *y = malloc((size_t)S * D * 4);
    for (int i = 0; i < S * D; i++) x[i] = sinf(i * 0.37f);
    gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, D);
    for (int t16 = GQ_Q2_0_R16; t16 < GQ_R16_END; t16++) {
        const int t = gq_r16_base(t16); const int64_t rb = gq_row_bytes(t, D), hb = gq_row_bytes(t16, D);
        uint8_t **w = malloc(M * sizeof *w), **h = malloc(M * sizeof *h);
        for (int i = 0; i < M; i++) { w[i] = malloc((size_t)rb * D); h[i] = malloc((size_t)hb * D); fill(t, w[i], D, D); gq_r16_pack(t16, h[i], w[i], D, D); }
        double pk = 1e9;   /* packing (single thread, as a loader thread does it per expert) */
        for (int r = 0; r < 3; r++) { double t0 = now(); gq_r16_pack(t16, h[r % M], w[r % M], D, D); double dt = now() - t0; if (dt < pk) pk = dt; }
        double best[2] = {1e9, 1e9};
        for (int r = 0; r <= reps; r++)
            for (int k = 0; k < 2; k++) {
                double t0 = now();
                for (int i = 0; i < M; i++) gq_matmul_act(y, &a, k ? h[i] : w[i], k ? t16 : t, D);
                double dt = (now() - t0) / M; if (r && dt < best[k]) best[k] = dt;
            }
        printf("%-7s S=%-3d rows %.3f ms (%.1f GB/s)   interleaved %.3f ms (%.1f GB/s)   %.2fx   packing %.2f GB/s\n", GQ_TYPES[t].name, S,
               best[0] * 1e3, rb * D / best[0] / 1e9, best[1] * 1e3, hb * D / best[1] / 1e9, best[0] / best[1], rb * D / pk / 1e9);
        for (int i = 0; i < M; i++) { free(w[i]); free(h[i]); }
        free(w); free(h);
    }
    return 0;
}
#endif
