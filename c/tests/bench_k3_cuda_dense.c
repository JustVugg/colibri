/* Time coli_cuda.dll's resident matmuls at Kimi K3's trunk shapes (K3_CUDA_DENSE):
 * one decode row and prompt chunks, int4-g64 and int8 rows, against the CPU kernels
 * kimi_k3 runs the same matrices with. Random weights; nothing is compared but time.
 *   make tests/bench_k3_cuda_dense.exe CUDA_DLL=1  (coli_cuda.dll beside it, or on PATH)
 *   tests/bench_k3_cuda_dense.exe [repeats] */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../compat.h"
#include "../quant.h"
#include "../backend_cuda.h"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

typedef struct { const char *name; int O, I, fmt; } Shape;

int main(int argc, char **argv){
    int reps=argc>1?atoi(argv[1]):20, dev=0;
    if(!coli_cuda_init(&dev,1)){ fprintf(stderr,"no CUDA device (coli_cuda.dll?)\n"); return 1; }
    Shape sh[]={
        {"KDA q/k/v/g 12288x7168 int4-g64",12288,7168,4},
        {"KDA o 7168x12288 int4-g64",7168,12288,4},
        {"MLA o 7168x12288 int8",7168,12288,1},
        {"shared gate/up 6144x7168 int4-g64",6144,7168,4},
        {"latent down 3584x7168 int4-g64",3584,7168,4},
        {"head 163840x7168 int8",163840,7168,1},
    };
    int rows[]={1,32,512};
    for(size_t k=0;k<sizeof sh/sizeof sh[0];k++){
        int O=sh[k].O, I=sh[k].I, fmt=sh[k].fmt, gs=64, ng=I/gs;
        size_t wb=fmt==1?(size_t)O*I:(size_t)O*(I/2), sn=fmt==1?(size_t)O:(size_t)O*ng;
        uint8_t *wq=malloc(wb); float *sc=malloc(sn*sizeof(float));
        for(size_t i=0;i<wb;i++) wq[i]=(uint8_t)(rand()&0xff);
        for(size_t i=0;i<sn;i++) sc[i]=1e-3f*(1+(rand()&7));
        ColiCudaTensor *t=NULL;
        if(!coli_cuda_tensor_upload_g(&t,wq,sc,fmt,I,O,dev,fmt==4?gs:0)){ printf("%s: upload failed\n",sh[k].name); continue; }
        for(size_t r=0;r<sizeof rows/sizeof rows[0];r++){
            int S=rows[r];
            if(O>100000&&S>1) continue;
            float *x=malloc((size_t)S*I*sizeof(float)), *y=malloc((size_t)S*O*sizeof(float));
            for(size_t i=0;i<(size_t)S*I;i++) x[i]=(float)(rand()%200-100)*0.01f;
            coli_cuda_matmul(&t,y,x,wq,sc,fmt,S,I,O,dev,fmt==4?gs:0);
            double t0=now_s();
            for(int i=0;i<reps;i++) coli_cuda_matmul(&t,y,x,wq,sc,fmt,S,I,O,dev,fmt==4?gs:0);
            double g=(now_s()-t0)/reps;
            double c=0;
            if(S<=32){
                t0=now_s();
                int cr=S==1?reps:2;
                for(int i=0;i<cr;i++){
                    if(fmt==1) matmul_q(y,x,(const int8_t*)wq,sc,S,I,O);
                    else matmul_i4_grouped(y,x,wq,sc,S,I,O,gs);
                }
                c=(now_s()-t0)/cr;
            }
            printf("%-36s S=%4d: GPU %8.3f ms (%6.1f GB/s of weights)  CPU %8.3f ms\n",sh[k].name,S,g*1e3,
                   (wb+sn*4)/g/1e9,c*1e3);
            free(x); free(y);
        }
        coli_cuda_tensor_free(t);
        free(wq); free(sc);
    }
    return 0;
}
