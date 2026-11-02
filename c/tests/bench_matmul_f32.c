/* Microbenchmark: matmul_f32.h's lane kernel against the f32 loops it replaced,
 * at the shapes the engines run it on: routers (GLM-5.2 in colibri.c, Inkling),
 * Qwen3.6's DeltaNet a/b projections and OLMoE's attention and lm_head.
 *   scalar   the plain `a+=x*w` loop (quant.h before #442, inkling, qwen36)
 *   ompsimd  the same loop under `#pragma omp simd reduction` (olmoe)
 *   fmaf8    eight rows of one fmaf chain each (quant.h after #1790)
 *   lanes    matmul_f32.h
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

static void matmul_scalar(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}
static void matmul_ompsimd(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0;
            #pragma omp simd reduction(+:a)
            for(int i=0;i<I;i++) a+=xs[i]*w[i];
            y[(int64_t)s*O+o]=a; } }
}
static void matmul_fmaf8(float *y, const float *x, const float *W, int S, int I, int O){
    int O8=O&~7;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O8;o+=8){ const float *w0=W+(int64_t)o*I, *w1=w0+I, *w2=w1+I, *w3=w2+I, *w4=w3+I, *w5=w4+I, *w6=w5+I, *w7=w6+I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
            for(int i=0;i<I;i++){ float xi=xs[i];
                a0=fmaf(xi,w0[i],a0); a1=fmaf(xi,w1[i],a1); a2=fmaf(xi,w2[i],a2); a3=fmaf(xi,w3[i],a3);
                a4=fmaf(xi,w4[i],a4); a5=fmaf(xi,w5[i],a5); a6=fmaf(xi,w6[i],a6); a7=fmaf(xi,w7[i],a7); }
            float *ys=y+(int64_t)s*O+o; ys[0]=a0; ys[1]=a1; ys[2]=a2; ys[3]=a3; ys[4]=a4; ys[5]=a5; ys[6]=a6; ys[7]=a7; } }
    for (int o=O8;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a=fmaf(xs[i],w[i],a); y[(int64_t)s*O+o]=a; } }
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
        {"GLM-5.2 router decode",   1, 6144,   256}, {"GLM-5.2 router prefill", 16, 6144, 256},
        {"Inkling router decode",   1, 6144,   258}, {"Qwen3.6 DeltaNet a/b",    1, 2048,  32},
        {"OLMoE router decode",     1, 2048,    64}, {"OLMoE q/k/v/o decode",    1, 2048, 2048},
        {"OLMoE q/k/v/o prefill",  16, 2048,  2048}, {"OLMoE lm_head decode",    1, 2048, 50304} };
    static const mm_fn arm[]={ matmul_scalar, matmul_ompsimd, matmul_fmaf8, matmul };
    static const char *arm_name[]={ "scalar", "ompsimd", "fmaf8", "lanes" };
    enum { ARMS=4 };
    uint64_t rng=0x9E3779B97F4A7C15ull;
#ifdef _OPENMP
    printf("threads: %d\n", omp_get_max_threads());
#else
    printf("threads: 1 (no OpenMP)\n");
#endif
    printf("%-24s %-22s", "shape", "");
    for (int a=0;a<ARMS;a++) printf(" %9s", arm_name[a]);
    printf("   median us/call; lanes vs each\n");
    for (size_t k=0;k<sizeof shapes/sizeof shapes[0];k++){
        int S=shapes[k].S, I=shapes[k].I, O=shapes[k].O;
        float *x=malloc(sizeof(float)*(size_t)S*I), *W=malloc(sizeof(float)*(size_t)I*O), *y=malloc(sizeof(float)*(size_t)S*O);
        for (size_t i=0;i<(size_t)S*I;i++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; x[i]=(float)(rng>>40)/(float)(1<<24)*2.f-1.f; }
        for (size_t i=0;i<(size_t)I*O;i++){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; W[i]=((float)(rng>>40)/(float)(1<<24)*2.f-1.f)*0.05f; }
        int reps=1; while (sample_us(matmul_scalar,y,x,W,S,I,O,reps)*reps<50e3 && reps<1<<20) reps*=2;
        for (int a=0;a<ARMS;a++) sample_us(arm[a],y,x,W,S,I,O,reps);   /* warm-up */
        double t[ARMS][SAMPLES];
        for (int r=0;r<SAMPLES;r++) for (int a=0;a<ARMS;a++) t[a][r]=sample_us(arm[a],y,x,W,S,I,O,reps);
        for (int a=0;a<ARMS;a++) qsort(t[a],SAMPLES,sizeof(double),cmp_d);
        double lanes=t[ARMS-1][SAMPLES/2];
        printf("%-24s S=%-2d I=%-4d O=%-6d ", shapes[k].name, S, I, O);
        for (int a=0;a<ARMS;a++) printf(" %9.2f", t[a][SAMPLES/2]);
        printf("  ");
        for (int a=0;a<ARMS-1;a++) printf(" %.2fx", t[a][SAMPLES/2]/lanes);
        printf("\n");
        free(x); free(W); free(y);
    }
    return 0;
}
