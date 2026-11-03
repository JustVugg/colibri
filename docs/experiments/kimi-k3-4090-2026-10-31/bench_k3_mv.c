/* bench_k3_mv.c -- a decode row (and 1-4 rows) on coli_cuda.dll's copies of Kimi K3's held matrices:
 * coli_cuda_matmul (quant_matvec_warp, quant_matmul's order) against k3cu_matmul_dev (the CPU's
 * arithmetic), the whole call with its copies, the median of REPS. Built from c/:
 *   gcc ... bench_k3_mv.c backend_cuda_k3_loader.c backend_loader.o cu_tier.o -o tests/bench_k3_mv.exe */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "backend_cuda.h"
#include "backend_cuda_k3.h"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmpd(const void *a, const void *b){ double x=*(const double*)a, y=*(const double*)b; return x<y?-1:x>y; }
typedef struct { const char *name; int O, I, fmt, n; } Shape;   /* n: how many a decode token runs */

int main(int argc, char **argv){
    int reps=argc>1?atoi(argv[1]):50, dev=0;
    if(!coli_cuda_init(&dev,1)||!k3cu_init(0)){ fprintf(stderr,"no CUDA device or DLLs\n"); return 1; }
    Shape sh[]={
        {"KDA q/k/v/g 12288x7168 int4-g64",12288,7168,4,250},
        {"KDA o 7168x12288 int4-g64",7168,12288,4,67},
        {"MLA g 12288x7168 int8",12288,7168,1,24},
        {"MLA o 7168x12288 int8",7168,12288,1,24},
        {"dense gate/up 33792x7168 int4-g64",33792,7168,4,2},
        {"dense down 7168x33792 int4-g64",7168,33792,4,1},
        {"head 163840x7168 int8",163840,7168,1,1},
    };
    double tot_old=0, tot_new=0;
    for(size_t k=0;k<sizeof sh/sizeof sh[0];k++){
        const int O=sh[k].O, I=sh[k].I, fmt=sh[k].fmt, gs=fmt==4?64:0;
        size_t wb=fmt==1?(size_t)O*I:(size_t)O*(I/2), sn=fmt==1?(size_t)O:(size_t)O*(I/64);
        uint8_t *wq=malloc(wb); float *sc=malloc(sn*sizeof(float));
        for(size_t i=0;i<wb;i++) wq[i]=(uint8_t)(rand()&0xff);
        for(size_t i=0;i<sn;i++) sc[i]=1e-3f*(1+(rand()&7));
        float *x=malloc((size_t)4*I*4), *y=malloc((size_t)4*O*4);
        for(int i=0;i<4*I;i++) x[i]=(float)(rand()%2001-1000)/1000.f;
        ColiCudaTensor *t=NULL;
        if(!coli_cuda_tensor_upload_g(&t,wq,sc,fmt,I,O,dev,gs)){ fprintf(stderr,"upload failed\n"); return 1; }
        const void *dw=NULL; const float *ds=NULL; int tgs=0;
        coli_cuda_tensor_dev_ptrs(t,&dw,&ds,&tgs);
        double *tm=malloc((size_t)reps*sizeof(double)), new1=0;
        printf("%-36s", sh[k].name);
        for(int R=1;R<=4;R+=R<2?1:2){
            double med[2];
            for(int which=0;which<2;which++){
                for(int r=-3;r<reps;r++){
                    double t0=now_s();
                    int ok=which==0?coli_cuda_matmul(&t,y,x,wq,sc,fmt,R,I,O,dev,gs):k3cu_matmul_dev(fmt,y,x,dw,ds,R,I,O,tgs);
                    if(!ok){ fprintf(stderr,"call failed\n"); return 1; }
                    if(r>=0) tm[r]=now_s()-t0;
                }
                qsort(tm,(size_t)reps,sizeof(double),cmpd);
                med[which]=tm[reps/2]*1e3;
            }
            printf("  %d row%s: old %.3f ms, new %.3f ms", R, R>1?"s":"", med[0], med[1]);
            if(R==1){ tot_old+=med[0]*sh[k].n; tot_new+=med[1]*sh[k].n; new1=med[1]; }
        }
        printf("  (new, 1 row: %.0f GB/s)\n", (double)(wb+sn*4)/(new1*1e-3)/1e9);
        coli_cuda_tensor_free(t); free(wq); free(sc); free(x); free(y); free(tm);
    }
    printf("a decode token's held matrices (counts as on the 4090: 317 KDA, 48 MLA, 3 dense, head): old %.1f ms, new %.1f ms\n",
           tot_old, tot_new);
    return 0;
}
