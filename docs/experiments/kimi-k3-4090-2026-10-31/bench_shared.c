/* bench_shared.c -- k3cu_shared at Kimi K3's size: 4096 rows, hidden 7168, 2 x 6144 inter, int4 g64 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "backend_cuda_k3.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc, char **argv){
    int S = argc > 1 ? atoi(argv[1]) : 4096, D = 7168, I = 12288, gs = 64;
    if(!k3cu_init(0)){ printf("no device\n"); return 1; }
    uint8_t *g = malloc((size_t)I*D/2), *u = malloc((size_t)I*D/2), *d = malloc((size_t)D*I/2);
    float *gsc = malloc((size_t)I*(D/gs)*4), *usc = malloc((size_t)I*(D/gs)*4), *dsc = malloc((size_t)D*(I/gs)*4);
    float *x = malloc((size_t)S*D*4), *y = malloc((size_t)S*D*4);
    for(size_t i=0;i<(size_t)I*D/2;i++){ g[i]=(uint8_t)(i*2654435761u>>24); u[i]=(uint8_t)(i*40503u>>8); d[i]=(uint8_t)(i*2246822519u>>24); }
    for(size_t i=0;i<(size_t)I*(D/gs);i++){ gsc[i]=0.01f; usc[i]=0.01f; dsc[i]=0.01f; }
    for(size_t i=0;i<(size_t)S*D;i++) x[i]=(float)((int)(i%97)-48)*0.02f;
    double best=1e9;
    for(int k=0;k<4;k++){ double t0=now(); if(!k3cu_shared(y,x,S,D,I,g,gsc,u,usc,gs,d,dsc,gs,4.f,25.f)){ printf("failed\n"); return 1; } double t=now()-t0; if(k&&t<best) best=t; }
    printf("k3cu_shared %d rows: %.1f ms (best of 3 after a warm-up)\n", S, best*1e3);
    return 0;
}
