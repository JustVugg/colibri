/* Throughput of gq_matmul per type on an expert-shaped matrix (O x I). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include "../ggml_quants.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc, char **argv){
    int O = argc > 1 ? atoi(argv[1]) : 512, I = argc > 2 ? atoi(argv[2]) : 2048, reps = 200;
    int types[] = { GQ_Q4_K, GQ_Q5_K, GQ_Q6_K, GQ_Q8_0, GQ_Q2_K, GQ_Q3_K, GQ_IQ2_XXS, GQ_IQ2_S, GQ_IQ3_S, GQ_IQ3_XXS, GQ_IQ4_XS, GQ_IQ1_M, GQ_MXFP4 };
    float *x = malloc((size_t)I*4), *y = malloc((size_t)O*4);
    for (int i = 0; i < I; i++) x[i] = (float)((i*37)%17) * 0.01f;
    printf("%d threads, matrix %d x %d, S=1\n", omp_get_max_threads(), O, I);
    for (size_t k = 0; k < sizeof types/sizeof *types; k++) {
        int t = types[k]; int64_t rb = gq_row_bytes(t, I);
        uint8_t *W = malloc((size_t)rb*O); for (int64_t i = 0; i < rb*O; i++) W[i] = (uint8_t)(i*131u + 7u);
        for (int64_t o = 0; o < O; o++) for (int64_t b = 0; b < rb; b += GQ_TYPES[t].bytes) { if (t != GQ_MXFP4 && t != GQ_IQ1_M) { W[o*rb+b] = 0x00; W[o*rb+b+1] = 0x30; } }
        float *yr = malloc((size_t)O*4);
        gq_matmul_f32(yr, x, W, t, 1, I, O); gq_matmul(y, x, W, t, 1, I, O);
        double err = 0, mag = 0; for (int o = 0; o < O; o++) { err = fmax(err, fabs(y[o]-yr[o])); mag = fmax(mag, fabs(yr[o])); }
        double t0 = now(); for (int r = 0; r < reps; r++) gq_matmul_f32(y, x, W, t, 1, I, O); double df = (now()-t0)/reps;
        t0 = now(); for (int r = 0; r < reps; r++) gq_matmul(y, x, W, t, 1, I, O); double dt = (now()-t0)/reps;
        printf("%-8s f32 %7.3f ms | int8 %7.3f ms  %5.1fx  %6.2f GB/s  | max|diff|/max|y| %.2e\n", gq_type_name(t), df*1e3, dt*1e3, df/dt, rb*O/dt/1e9, err/mag);
        free(yr);
        free(W);
    }
    return 0;
}
