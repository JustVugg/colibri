/* Microbenchmark: the pre-#442 scalar f32 matmul vs quant.h's fmaf row-blocked
 * matmul, at the GLM-5.2 router shape (colibri.c, D=6144 E=256) and a smaller
 * router (D=2048 E=64), for decode (S=1) and a prefill chunk (S=16).
 *
 * NOT a unit test: tests/test_matmul_f32 gates exactness. Each arm runs
 * SAMPLES timed samples (each the mean of enough calls to last ~50 ms) after
 * one warm-up sample; the median is reported, arms interleaved per sample.
 *
 * Run:  make tests/bench_matmul_f32 && ./tests/bench_matmul_f32
 *       OMP_NUM_THREADS=1 ./tests/bench_matmul_f32   (single thread) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "../quant.h"

#define SAMPLES 5

/* ---- OLD: verbatim pre-#442 quant.h matmul ---- */
static void matmul_old(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmp_d(const void *a, const void *b){ double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }

typedef void (*mm_fn)(float*,const float*,const float*,int,int,int);
static double sample_us(mm_fn f, float *y, const float *x, const float *W, int S, int I, int O, int reps){
    mm_fn volatile call=f;   /* opaque call: without OpenMP the unused result would be optimized away */
    double t=now_s(); for (int r=0;r<reps;r++) call(y,x,W,S,I,O); return (now_s()-t)/reps*1e6;
}

int main(void){
    static const struct { const char *name; int S, I, O; } shapes[]={
        {"GLM-5.2 router decode",  1, 6144, 256}, {"GLM-5.2 router prefill", 16, 6144, 256},
        {"small router decode",    1, 2048,  64}, {"small router prefill",   16, 2048,  64} };
    uint64_t rng=0x9E3779B97F4A7C15ull;
#ifdef _OPENMP
    printf("threads: %d\n", omp_get_max_threads());
#else
    printf("threads: 1 (no OpenMP)\n");
#endif
    for (size_t k=0;k<sizeof shapes/sizeof shapes[0];k++){
        int S=shapes[k].S, I=shapes[k].I, O=shapes[k].O;
        float *x=malloc(sizeof(float)*(size_t)S*I), *W=malloc(sizeof(float)*(size_t)I*O), *y=malloc(sizeof(float)*(size_t)S*O);
        for (size_t i=0;i<(size_t)S*I;i++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; x[i]=(float)(rng>>40)/(float)(1<<24)*2.f-1.f; }
        for (size_t i=0;i<(size_t)I*O;i++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; W[i]=((float)(rng>>40)/(float)(1<<24)*2.f-1.f)*0.05f; }
        int reps=1; while (sample_us(matmul_old,y,x,W,S,I,O,reps)*reps<50e3 && reps<1<<20) reps*=2;
        sample_us(matmul_old,y,x,W,S,I,O,reps); sample_us(matmul,y,x,W,S,I,O,reps);   /* warm-up */
        double to[SAMPLES], tn[SAMPLES];
        for (int r=0;r<SAMPLES;r++){ to[r]=sample_us(matmul_old,y,x,W,S,I,O,reps); tn[r]=sample_us(matmul,y,x,W,S,I,O,reps); }
        printf("%-24s S=%-2d I=%d O=%d  old:", shapes[k].name, S, I, O);
        for (int r=0;r<SAMPLES;r++) printf(" %.2f", to[r]);
        printf("  new:"); for (int r=0;r<SAMPLES;r++) printf(" %.2f", tn[r]);
        qsort(to,SAMPLES,sizeof(double),cmp_d); qsort(tn,SAMPLES,sizeof(double),cmp_d);
        printf("  us/call; median %.2f -> %.2f (%.2fx)\n", to[SAMPLES/2], tn[SAMPLES/2], to[SAMPLES/2]/tn[SAMPLES/2]);
        free(x); free(W); free(y);
    }
    return 0;
}
