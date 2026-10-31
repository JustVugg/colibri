/* test_idot_r16: the int8 trunk rows interleaved by 16 (idot.h, matmul_q_idot_r16).
 *
 *   test_idot_r16              against matmul_q_idot (the row kernel) to the bit, for several
 *                              widths and 1..19 activation rows; pack then unpack gives the rows
 *                              back. Exit 0 on a build without AVX2 (nothing to check).
 *   test_idot_r16 bench [S]    ms per 2560 x 2560 matrix (48 of them, past the caches) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "../idot.h"

#ifndef IDOT_R16
int main(void) { puts("test_idot_r16: no AVX2 in this build, nothing to check"); return 0; }
#else
static uint64_t rng = 88172645463325252ULL;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void act(const float *x, int S, int I, int8_t *xq, float *sx, int32_t *xsum) {
    for (int s = 0; s < S; s++) {
        sx[s] = dense_act_i8(x + (size_t)s * I, I, xq + (size_t)s * I, NULL);
        int32_t t = 0; for (int i = 0; i < I; i++) t += xq[(size_t)s * I + i]; xsum[s] = t;
    }
}
int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "bench")) {
        int bad = 0, n = 0, trip = 0;
        const int widths[] = { 4, 64, 640, 2560, 10240, 4100 };
        for (int O = 48; O <= 64; O += 16) for (int wi = 0; wi < 6; wi++) for (int S = 1; S <= 19; S += S < 9 ? 1 : 5) {
            const int I = widths[wi];   /* 48 rows: an odd number of groups; 64: pairs of groups */
            int8_t *w = malloc((size_t)O * I), *r = malloc((size_t)O * I), *back = malloc((size_t)O * I), *xq = malloc((size_t)S * I);
            float *sc = malloc(O * 4), *x = malloc((size_t)S * I * 4), *y1 = malloc((size_t)S * O * 4), *y2 = malloc((size_t)S * O * 4), *sx = malloc(S * 4);
            int32_t *xsum = malloc(S * 4);
            for (size_t i = 0; i < (size_t)O * I; i++) w[i] = (int8_t)((int)(rnd() % 255) - 127);
            for (int o = 0; o < O; o++) sc[o] = 1e-3f * (1 + rnd() % 1000) / 7.f;
            for (size_t i = 0; i < (size_t)S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
            act(x, S, I, xq, sx, xsum);
            i8r16_pack(r, w, O, I); i8r16_unpack(back, r, O, I); trip += memcmp(back, w, (size_t)O * I) != 0;
            matmul_q_idot(y1, xq, sx, w, sc, S, I, O); matmul_q_idot_r16(y2, xq, sx, xsum, r, sc, S, I, O);
            n++; if (memcmp(y1, y2, (size_t)S * O * 4)) { bad++; if (bad < 3) printf("  I=%d S=%d: %.9g against %.9g\n", I, S, y2[0], y1[0]); }
            free(w); free(r); free(back); free(xq); free(sc); free(x); free(y1); free(y2); free(sx); free(xsum);
        }
        printf("test_idot_r16: %d of %d cases differ from matmul_q_idot; round trip %s\n", bad, n, trip ? "BROKEN" : "exact");
        return bad || trip;
    }
    const int S = argc > 2 ? atoi(argv[2]) : 1, M = 48, D = 2560;
    int8_t **w = malloc(M * sizeof *w), **r = malloc(M * sizeof *r), *xq = malloc((size_t)S * D);
    float *sc = malloc(D * 4), *x = malloc((size_t)S * D * 4), *y = malloc((size_t)S * D * 4), *sx = malloc(S * 4); int32_t *xsum = malloc(S * 4);
    for (int i = 0; i < M; i++) { w[i] = malloc((size_t)D * D); r[i] = malloc((size_t)D * D);
        for (size_t k = 0; k < (size_t)D * D; k++) w[i][k] = (int8_t)((int)(rnd() % 255) - 127);
        i8r16_pack(r[i], w[i], D, D); }
    for (int o = 0; o < D; o++) sc[o] = 1e-3f;
    for (size_t i = 0; i < (size_t)S * D; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
    act(x, S, D, xq, sx, xsum);
    double best[2] = { 1e9, 1e9 };
    for (int rep = 0; rep < 6; rep++) for (int k = 0; k < 2; k++) {
        double t0 = now();
        for (int i = 0; i < M; i++) { if (k) matmul_q_idot_r16(y, xq, sx, xsum, r[i], sc, S, D, D); else matmul_q_idot(y, xq, sx, w[i], sc, S, D, D); }
        double dt = (now() - t0) / M; if (rep && dt < best[k]) best[k] = dt;
    }
    printf("int8 rows S=%-3d rows %.3f ms (%.1f GB/s)   interleaved %.3f ms (%.1f GB/s)   %.2fx\n", S, best[0] * 1e3,
           (double)D * D / best[0] / 1e9, best[1] * 1e3, (double)D * D / best[1] / 1e9, best[0] / best[1]);
    return 0;
}
#endif
