/* test_xf16: int4-g64 rows interleaved by 16 (expert_ffn.h xf16_*).
 *
 *   test_xf16             pack then unpack gives the planar codes and the scales back; xf_act32
 *                         against xf_act32_ref to the byte; the kernel against xf16_ref (its
 *                         numerics) to the bit for 1..19 activation rows, each row from its own
 *                         pointer, and xf_dot_a32 (the same numbers on a planar row) too; its gap to xf_dot_f32 (f32 activations, the reference
 *                         of the rows) no wider than xf_dot_i8's (one int8 scale a row, the
 *                         engines' default before), measured as the worst output over the
 *                         outputs' rms. Exit 0 on a build without AVX2.
 *   test_xf16 bench [S]   ms per 512 x 2048 matrix (480 of them, an expert forward's worth):
 *                         the interleaved rows against xf_dot_i8 and xf_dot_f32 once per row */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../expert_ffn.h"

#ifndef XF16_ON
int main(void) { puts("test_xf16: no AVX2 in this build, nothing to check"); return 0; }
#else
static uint64_t rng = 88172645463325252ULL;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "bench")) {
        int bad = 0, n = 0, trip = 0, actbad = 0, a32bad = 0; double worst = 0, worst_i8 = 0;
        for (int I = 64; I <= 2560; I += I < 512 ? 192 : 1536) for (int S = 1; S <= 19; S += S < 9 ? 1 : 5) {
            const int R = 48, nb = I / XF_BLOCK; const size_t rb = xf_row_bytes(I), gb = xf16_group_bytes(I);
            uint8_t *w = malloc(rb * R), *g = malloc(gb * (R / 16)), *w2 = malloc(rb * R);
            float *sc = malloc((size_t)R * nb * 4), *sc2 = malloc((size_t)R * nb * 4);
            for (size_t i = 0; i < rb * R; i++) w[i] = (uint8_t)rnd();
            for (int i = 0; i < R * nb; i++) sc[i] = 1e-4f * (1 + rnd() % 1000);
            xf16_pack(g, w, sc, R, I); xf16_unpack(w2, sc2, g, R, I);
            trip += memcmp(w2, w, rb * R) || memcmp(sc2, sc, (size_t)R * nb * 4);
            /* activation rows apart, each its own allocation, in reverse order */
            float **x = malloc(S * sizeof *x), **dx = malloc(S * sizeof *dx), **y = malloc(S * sizeof *y);
            int8_t **xq = malloc(S * sizeof *xq); int32_t **xs = malloc(S * sizeof *xs);
            for (int s = S - 1; s >= 0; s--) {
                x[s] = malloc((size_t)I * 4); dx[s] = malloc((size_t)(I / 32) * 4); xq[s] = malloc(I); xs[s] = malloc((size_t)(I / 16) * 4); y[s] = malloc(R * 4);
                for (int i = 0; i < I; i++) x[s][i] = ((int)(rnd() % 2001) - 1000) * 1e-3f * (1 + (i % 7));
                int8_t *q2 = malloc(I); float *d2 = malloc((size_t)(I / 32) * 4); int32_t *s2 = malloc((size_t)(I / 16) * 4);
                xf_act32(x[s], I, xq[s], dx[s], xs[s]); xf_act32_ref(x[s], I, q2, d2, s2);
                actbad += memcmp(q2, xq[s], I) || memcmp(d2, dx[s], (size_t)(I / 32) * 4) || memcmp(s2, xs[s], (size_t)(I / 16) * 4);
                free(q2); free(d2); free(s2);
            }
            for (int grp = 0; grp < R / 16; grp++) xf16_rows(y, 16 * grp, g + grp * gb, (const int8_t *const *)xq, (const float *const *)dx, (const int32_t *const *)xs, I, S);
            double rms = 0; float *ref = malloc((size_t)S * R * 4), *f32 = malloc((size_t)S * R * 4), *i8 = malloc((size_t)S * R * 4);
            int8_t *q1 = malloc(I); float *xsum = malloc((size_t)nb * 4);
            for (int s = 0; s < S; s++) {
                const float sx = xf_act_i8(x[s], I, q1, xsum);
                for (int r = 0; r < R; r++) {
                    ref[s * R + r] = xf16_ref(w + r * rb, sc + r * nb, xq[s], dx[s], I);
                    f32[s * R + r] = xf_dot_f32(w + r * rb, sc + r * nb, x[s], I);
                    i8[s * R + r] = xf_dot_i8(w + r * rb, sc + r * nb, q1, xsum, sx, I);
                    rms += (double)f32[s * R + r] * f32[s * R + r];
                }
            }
            rms = sqrt(rms / (S * R)) + 1e-30;
            int diff = 0;
            for (int s = 0; s < S; s++) for (int r = 0; r < R; r++) {
                double e = fabs((double)f32[s * R + r] - y[s][r]) / rms; if (e > worst) worst = e;
                double e8 = fabs((double)f32[s * R + r] - i8[s * R + r]) / rms; if (e8 > worst_i8) worst_i8 = e8;
                if (memcmp(&ref[s * R + r], &y[s][r], 4)) diff = 1;
                const float a = xf_dot_a32(w + r * rb, sc + r * nb, xq[s], dx[s], xs[s], I);
                if (memcmp(&ref[s * R + r], &a, 4)) a32bad++;
            }
            free(q1); free(xsum); free(i8);
            n++; if (diff) { bad++; if (bad < 3) printf("  I=%d S=%d: %.9g against %.9g\n", I, S, y[0][0], ref[0]); }
            for (int s = 0; s < S; s++) { free(x[s]); free(dx[s]); free(xq[s]); free(xs[s]); free(y[s]); }
            free(x); free(dx); free(xq); free(xs); free(y); free(ref); free(f32); free(w); free(g); free(w2); free(sc); free(sc2);
        }
        printf("int4-g64 x16: %d of %d cases differ from xf16_ref; planar rows (xf_dot_a32): %d outputs differ; round trip %s; activations %s; "
               "worst gap to xf_dot_f32 %.2e of the rms (xf_dot_i8 %.2e)\n",
               bad, n, a32bad, trip ? "BROKEN" : "exact", actbad ? "DIFFER from xf_act32_ref" : "as xf_act32_ref", worst, worst_i8);
        const int fails = bad + a32bad + trip + actbad + (worst > worst_i8);
        printf("test_xf16: %s\n", fails ? "FAILED" : "ok");
        return fails != 0;
    }
    const int S = argc > 2 ? atoi(argv[2]) : 1, M = 480, O = 512, I = 2048, nb = I / XF_BLOCK;
    const size_t rb = xf_row_bytes(I), gb = xf16_group_bytes(I);
    uint8_t **w = malloc(M * sizeof *w), **g = malloc(M * sizeof *g); float **sc = malloc(M * sizeof *sc);
    for (int m = 0; m < M; m++) {
        w[m] = malloc(rb * O); g[m] = malloc(gb * (O / 16)); sc[m] = malloc((size_t)O * nb * 4);
        for (size_t i = 0; i < rb * O; i++) w[m][i] = (uint8_t)rnd();
        for (int i = 0; i < O * nb; i++) sc[m][i] = 1e-3f;
        xf16_pack(g[m], w[m], sc[m], O, I);
    }
    float *x = malloc((size_t)S * I * 4), *dx = malloc((size_t)S * (I / 32) * 4), *y = malloc((size_t)S * O * 4), *xsum = malloc((size_t)S * nb * 4), *sx = malloc(S * 4);
    int8_t *xq = malloc((size_t)S * I), *xq1 = malloc((size_t)S * I); int32_t *xs = malloc((size_t)S * (I / 16) * 4);
    const int8_t *xp[256]; const float *dp[256], *fp[256]; const int32_t *sp[256]; float *yp[256];
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = sinf(i * 0.37f);
    for (int s = 0; s < S; s++) {
        xf_act32(x + (size_t)s * I, I, xq + (size_t)s * I, dx + (size_t)s * (I / 32), xs + (size_t)s * (I / 16));
        sx[s] = xf_act_i8(x + (size_t)s * I, I, xq1 + (size_t)s * I, xsum + (size_t)s * nb);
        xp[s] = xq + (size_t)s * I; dp[s] = dx + (size_t)s * (I / 32); sp[s] = xs + (size_t)s * (I / 16); fp[s] = x + (size_t)s * I; yp[s] = y + (size_t)s * O;
    }
    double pk = 1e9;   /* packing, one thread */
    for (int rep = 0; rep < 3; rep++) { double t0 = now(); xf16_pack(g[rep], w[rep], sc[rep], O, I); double dt = now() - t0; if (dt < pk) pk = dt; }
    double best[3] = { 1e9, 1e9, 1e9 };
    for (int rep = 0; rep < 5; rep++) for (int k = 0; k < 3; k++) {
        double t0 = now();
        for (int m = 0; m < M; m++) {
            if (k == 2) {
                #pragma omp parallel for schedule(static)
                for (int grp = 0; grp < O / 16; grp++) xf16_rows(yp, 16 * grp, g[m] + grp * gb, xp, dp, sp, I, S);
            } else {
                #pragma omp parallel for schedule(static)
                for (int o = 0; o < O; o++)
                    for (int s = 0; s < S; s++)
                        y[(size_t)s * O + o] = k ? xf_dot_i8(w[m] + o * rb, sc[m] + o * nb, xq1 + (size_t)s * I, xsum + (size_t)s * nb, sx[s], I)
                                                 : xf_dot_f32(w[m] + o * rb, sc[m] + o * nb, fp[s], I);
            }
        }
        double dt = now() - t0; if (rep && dt < best[k]) best[k] = dt;
    }
    const double bytes = (double)M * O * (rb + nb * 4);
    printf("int4-g64 S=%-3d f32 rows %.2f ms (%.1f GB/s)   i8 rows %.2f ms (%.1f GB/s)   interleaved %.2f ms (%.1f GB/s)   %.2fx on i8   packing %.2f GB/s\n",
           S, best[0] * 1e3, bytes / best[0] / 1e9, best[1] * 1e3, bytes / best[1] / 1e9, best[2] * 1e3, bytes / best[2] / 1e9, best[1] / best[2],
           (double)O * (rb + nb * 4) / pk / 1e9);
    return 0;
}
#endif
