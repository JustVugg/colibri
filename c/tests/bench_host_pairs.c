/* bench_host_pairs: vkt_host_pairs (the hybrid steps' CPU pairs) alone, on colibri's int4 rows
 * (fmt 2 a row's scale, fmt 4 groups of 64), int8 (fmt 1) and FP8 e4m3 (fmt 12, groups of 128),
 * the shapes of Qwen3.6-35B-A3B's expert (D 2048, I 512) or those given (Qwen3.8-Flash-Next:
 * 12 2560 640). No device: the host pool's rows are malloc'd here.
 *   ./tests/bench_host_pairs [fmt] [D] [I] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "../vk_tier.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static float frnd(void) { return (float)rand() / RAND_MAX * 2.f - 1.f; }
typedef struct { uint8_t *g, *u, *d; float *sg, *su, *sd; } Ex;
/* rows of O x I in fmt (1 int8, 2 int4 pairs v+8, 4 int4 pairs gs 64, 12 FP8 e4m3 gs 128); scales per row or per group */
static uint8_t *mk(int fmt, int O, int I, int gs, float **sc) {
    size_t rb = fmt == 1 || fmt == 12 ? (size_t)I : (size_t)I / 2, ns = gs ? (size_t)O * (((size_t)I + gs - 1) / gs) : (size_t)O;   /* a row's last group may be partial */
    uint8_t *r = malloc(rb * O); *sc = malloc(ns * sizeof(float));
    for (size_t i = 0; i < ns; i++) (*sc)[i] = 0.01f + 0.02f * (float)rand() / RAND_MAX;
    for (size_t i = 0; i < rb * O; i++) r[i] = fmt == 1 ? (uint8_t)(int8_t)(rand() % 255 - 127) : (uint8_t)(rand() & 0xff);
    return r;
}
int main(int argc, char **argv) {
    int fmt = argc > 1 ? atoi(argv[1]) : 2, D = argc > 2 ? atoi(argv[2]) : 2048, I = argc > 3 ? atoi(argv[3]) : 512, NE = 64;
    int gs = fmt == 4 ? 64 : fmt == 12 ? 128 : 0;
    const int byte_rows = fmt == 1 || fmt == 12;   /* a byte a weight */
    int nps[] = {1, 2, 8}; int nts[] = {1, 2, 4, 12, 24}, nnt = 5;
    if (argc > 4) { nts[0] = atoi(argv[4]); nnt = 1; }   /* one thread count: [fmt] [D] [I] [threads] [calls] */
    const int calls = argc > 5 ? atoi(argv[5]) : 400;
    Ex *ex = malloc((size_t)NE * sizeof *ex); VktHostExpert *hv = malloc((size_t)NE * sizeof *hv);
    for (int e = 0; e < NE; e++) {
        ex[e].g = mk(fmt, I, D, gs, &ex[e].sg); ex[e].u = mk(fmt, I, D, gs, &ex[e].su); ex[e].d = mk(fmt, D, I, gs, &ex[e].sd);
        hv[e] = (VktHostExpert){ex[e].g, ex[e].u, ex[e].d, -fmt, -fmt, byte_rows ? (size_t)D : (size_t)D / 2, byte_rows ? (size_t)I : (size_t)I / 2,
                                ex[e].sg, ex[e].su, ex[e].sd, gs, gs};
    }
    float *x = malloc((size_t)8 * D * sizeof(float)); for (int i = 0; i < 8 * D; i++) x[i] = frnd();
    float *gu = malloc((size_t)8 * 2 * I * sizeof(float)), *hh = malloc((size_t)8 * I * sizeof(float)), *dn = malloc((size_t)8 * D * sizeof(float));
    double bytes = byte_rows ? 3.0 * D * I : 1.5 * D * I;
    printf("fmt %d D %d I %d, %.2f MB a pair\n", fmt, D, I, bytes / 1e6);
    for (int a = 0; a < 3; a++) for (int b = 0; b < nnt; b++) {
        int np = nps[a], nt = nts[b];
        const VktHostExpert *hp[8]; const float *xp[8];
        int reps = calls / np > 0 ? calls / np : 1, e0 = 0; double best = 1e9, tot = 0;
        for (int r = -2; r < 5; r++) {
            double t0 = now();
            for (int k = 0; k < reps; k++) {
                for (int p = 0; p < np; p++) { hp[p] = &hv[(e0 + p) % NE]; xp[p] = x + (size_t)(p % 8) * D; }
                e0 += np;
                if (!vkt_host_pairs(hp, xp, np, D, I, 0.f, nt, gu, hh, dn)) { printf("failed\n"); return 1; }
            }
            double t = (now() - t0) / reps;
            if (r >= 0) { if (t < best) best = t; tot += t; }
        }
        printf("np %d nt %2d: %7.1f us a call, %6.1f us a pair (best; mean %6.1f), %5.1f GB/s\n", np, nt, best * 1e3, best * 1e3 / np,
               tot / 5 * 1e3 / np, bytes * np / (best * 1e-3) / 1e9);
    }
    return 0;
}
