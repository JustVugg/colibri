/* test_vq2r16: vq2 rows interleaved by 16 (vq2_g32.h vq2r16_*).
 *
 *   test_vq2r16             pack then unpack gives the codes, scales and row scales back; the
 *                           kernel against vq2r16_ref (its numerics) to the bit for 1..19
 *                           activation rows; the gap to vq2_dot (eight lane sums) below 1e-5 of
 *                           the outputs' rms. Exit 0 on a build without AVX2.
 *   test_vq2r16 bench [S]   ms per 640 x 2560 matrix (480 of them, an expert forward's worth) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../vq2_g32.h"

#ifndef VQ2R16_ON
int main(void) { puts("test_vq2r16: no AVX2 in this build, nothing to check"); return 0; }
#else
static uint64_t rng = 88172645463325252ULL;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void act(const float *x, int S, int I, int8_t *xq, float *dx, int32_t *xs) {
    for (int s = 0; s < S; s++) {
        x2_quant_act(x + (size_t)s * I, I, xq + (size_t)s * I, dx + (size_t)s * (I / 32));
        for (int k = 0; k < I / 16; k++) { int32_t t = 0; for (int i = 0; i < 16; i++) t += xq[(size_t)s * I + 16 * k + i]; xs[(size_t)s * (I / 16) + k] = t; }
    }
}
int main(int argc, char **argv) {
    Vq2Books *bk = malloc(sizeof *bk);
    for (int i = 0; i < VQ2_BOOK; i++) for (int e = 0; e < 8; e++) { bk->a[i][e] = (int8_t)((int)(rnd() % 127) - 63); bk->b[i][e] = (int8_t)((int)(rnd() % 129) - 64); }
    if (argc < 2 || strcmp(argv[1], "bench")) {
        int bad = 0, n = 0, trip = 0; double worst = 0;
        for (int I = 64; I <= 2560; I += I < 640 ? 192 : 1920) for (int S = 1; S <= 19; S += S < 9 ? 1 : 5) {
            const int R = 48; const size_t cb = vq2_code_bytes(I), qb = vq2_scale_bytes(I), gb = vq2r16_group_bytes(I);
            uint8_t *codes = malloc(cb * R), *q = malloc(qb * R), *g = malloc(gb * (R / 16)), *c2 = malloc(cb * R), *q2 = malloc(qb * R);
            float *d = malloc(R * 4), *d2 = malloc(R * 4);
            for (size_t i = 0; i < cb * R; i++) codes[i] = (uint8_t)rnd();
            for (size_t i = 0; i < qb * R; i++) q[i] = (uint8_t)rnd();
            for (int r = 0; r < R; r++) d[r] = 1e-4f * (1 + rnd() % 1000);
            vq2r16_pack(g, codes, q, d, R, I); vq2r16_unpack(c2, q2, d2, g, R, I);
            trip += memcmp(c2, codes, cb * R) || memcmp(q2, q, qb * R) || memcmp(d2, d, R * 4);
            float *x = malloc((size_t)S * I * 4), *dx = malloc((size_t)S * (I / 32) * 4); int8_t *xq = malloc((size_t)S * I); int32_t *xs = malloc((size_t)S * (I / 16) * 4);
            for (size_t i = 0; i < (size_t)S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
            act(x, S, I, xq, dx, xs);
            float *y1 = malloc((size_t)S * R * 4), *y2 = malloc((size_t)S * R * 4), *y3 = malloc((size_t)S * R * 4);
            for (int r = 0; r < R; r++) for (int s = 0; s < S; s++) {
                y1[s * R + r] = vq2r16_ref(bk, codes + r * cb, q + r * qb, d[r], xq + (size_t)s * I, dx + (size_t)s * (I / 32), I);
                y3[s * R + r] = vq2_dot_ref(bk, codes + r * cb, q + r * qb, d[r], xq + (size_t)s * I, dx + (size_t)s * (I / 32), I);
            }
            for (int grp = 0; grp < R / 16; grp++) vq2r16_rows(y2 + 16 * grp, R, bk, g + grp * gb, xq, dx, xs, I, S);
            double rms = 0; for (int i = 0; i < S * R; i++) rms += (double)y3[i] * y3[i]; rms = sqrt(rms / (S * R)) + 1e-30;
            for (int i = 0; i < S * R; i++) { double e = fabs((double)y3[i] - y2[i]) / rms; if (e > worst) worst = e; }
            n++; if (memcmp(y1, y2, (size_t)S * R * 4)) { bad++; if (bad < 3) printf("  I=%d S=%d: %.9g against %.9g\n", I, S, y2[0], y1[0]); }
            free(codes); free(q); free(g); free(c2); free(q2); free(d); free(d2); free(x); free(dx); free(xq); free(xs); free(y1); free(y2); free(y3);
        }
        printf("vq2 x16: %d of %d cases differ from vq2r16_ref; round trip %s; worst gap to vq2_dot %.2e of the rms\n", bad, n, trip ? "BROKEN" : "exact", worst);
        const int fails = bad + trip + (worst > 1e-5);
        printf("test_vq2r16: %s\n", fails ? "FAILED" : "ok");
        return fails != 0;
    }
    const int S = argc > 2 ? atoi(argv[2]) : 1, M = 480, O = 640, I = 2560;
    const size_t cb = vq2_code_bytes(I), qb = vq2_scale_bytes(I), gb = vq2r16_group_bytes(I);
    uint8_t **c = malloc(M * sizeof *c), **q = malloc(M * sizeof *q), **g = malloc(M * sizeof *g); float **d = malloc(M * sizeof *d);
    for (int m = 0; m < M; m++) {
        c[m] = malloc(cb * O); q[m] = malloc(qb * O); g[m] = malloc(gb * (O / 16)); d[m] = malloc(O * 4);
        for (size_t i = 0; i < cb * O; i++) c[m][i] = (uint8_t)rnd();
        for (size_t i = 0; i < qb * O; i++) q[m][i] = (uint8_t)rnd();
        for (int r = 0; r < O; r++) d[m][r] = 1e-3f;
    }
    double pk = 1e9;
    for (int m = 0; m < M; m++) { double t0 = now(); vq2r16_pack(g[m], c[m], q[m], d[m], O, I); double dt = now() - t0; if (dt < pk) pk = dt; }
    float *x = malloc((size_t)S * I * 4), *dx = malloc((size_t)S * (I / 32) * 4), *y = malloc((size_t)S * O * 4);
    int8_t *xq = malloc((size_t)S * I); int32_t *xs = malloc((size_t)S * (I / 16) * 4);
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = sinf(i * 0.37f);
    act(x, S, I, xq, dx, xs);
    double best[2] = { 1e9, 1e9 };
    for (int rep = 0; rep < 5; rep++) for (int k = 0; k < 2; k++) {
        double t0 = now();
        for (int m = 0; m < M; m++) {
            if (k) {
                #pragma omp parallel for schedule(static)
                for (int grp = 0; grp < O / 16; grp++) vq2r16_rows(y + 16 * grp, O, bk, g[m] + grp * gb, xq, dx, xs, I, S);
            } else {
                #pragma omp parallel for schedule(static)
                for (int o = 0; o < O; o++)
                    for (int s0 = 0; s0 < S; s0 += X2_MAX_ROWS)
                        vq2_dot(y + (size_t)s0 * O + o, O, bk, c[m] + o * cb, q[m] + o * qb, d[m][o], xq + (size_t)s0 * I, dx + (size_t)s0 * (I / 32), I,
                                S - s0 < X2_MAX_ROWS ? S - s0 : X2_MAX_ROWS);
            }
        }
        double dt = now() - t0; if (rep && dt < best[k]) best[k] = dt;
    }
    const double bytes = (double)M * O * (cb + qb + 4);
    printf("vq2 S=%-3d rows %.2f ms (%.1f GB/s)   interleaved %.2f ms (%.1f GB/s)   %.2fx   packing %.2f GB/s\n", S, best[0] * 1e3, bytes / best[0] / 1e9,
           best[1] * 1e3, bytes / best[1] / 1e9, best[0] / best[1], (double)O * (cb + qb + 4) / pk / 1e9);
    return 0;
}
#endif
