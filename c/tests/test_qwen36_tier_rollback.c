/* Fail each matrix upload in turn: unpublished handles must be reclaimed. */
#include <stdio.h>
#include "../compat.h"
#define coli_cuda_tensor_upload fixture_upload
#define coli_cuda_tensor_upload_g fixture_upload_g
#define coli_cuda_tensor_free fixture_free
#include "qwen36_fake_cuda.h"
#undef coli_cuda_tensor_upload
#undef coli_cuda_tensor_upload_g
#undef coli_cuda_tensor_free
static int attempt, fail_at, freed, expected_fmt, unexpected_fmt;
int coli_cuda_tensor_upload(ColiCudaTensor **t,const void *w,const float *s,
                           int fmt,int I,int O,int dev) {
    if(fmt!=expected_fmt) unexpected_fmt=1;
    if(++attempt==fail_at) return 0;
    return fixture_upload(t,w,s,fmt,I,O,dev);
}
int coli_cuda_tensor_upload_g(ColiCudaTensor **t,const void *w,const float *s,
                             int fmt,int I,int O,int dev,int gs) {
    if(fmt!=expected_fmt) unexpected_fmt=1;
    if(++attempt==fail_at) return 0;
    return fixture_upload_g(t,w,s,fmt,I,O,dev,gs);
}
void coli_cuda_tensor_free(ColiCudaTensor *t) { if(t) freed++; fixture_free(t); }
#include "../qwen36_tier.c"
int main(void) {
    enum { D=64 };
    unsigned char w[D*D]={0}; float sc[D], lut[256]={0};
    const int formats[]={1,2,4,8};
    for(int i=0;i<D;i++) sc[i]=1;
    setenv("COLI_CUDA","1",1); setenv("COLI_GPUS","0",1);
    setenv("QT_NO_WARMSTART","1",1); setenv("HEAT_FILE","",1);
    setenv("CUDA_EXPERT_GB","0.0625",1);
    for(int mode=0;mode<4;mode++) for(fail_at=1;fail_at<=3;fail_at++) {
        expected_fmt=formats[mode];
        attempt=freed=fake_uploads=unexpected_fmt=0;
        int started=expected_fmt==8 ? qt_init_fp8(1,1,D,D,1,1,lut)
                    : qt_init(1,1,D,D,1,1,expected_fmt==4?D:0,expected_fmt!=1);
        if(!started) return 1;
        qt_note(0,0,w,w,w,sc,sc,sc); qt_fill_wait();
        QSlot *s=qs(0,0);
        if(unexpected_fmt || attempt!=fail_at || fake_uploads!=fail_at-1 || freed!=fake_uploads ||
           s->resident || s->queued || s->tg || s->tu || s->td ||
           G.inflight || G.qn || G.used[0]) {
            fprintf(stderr,"rollback failed: fmt=%d matrix=%d uploads=%d frees=%d\n",
                    expected_fmt,fail_at,fake_uploads,freed); return 1;
        }
        if(expected_fmt==8 && (s->g4 || s->u4 || s->d4 || s->gs || s->us || s->ds)) {
            fputs("FP8 failure retained borrowed slot pointers\n",stderr); return 1;
        }
        qt_shutdown();
        /* This fixture targets failed uploads; older tier teardown retains
         * its host allocations. Release those here for leak sanitizers. */
        free(G.slot); G.slot=NULL; free(G.is_x); G.is_x=NULL;
        free(G.fill_order); G.fill_order=NULL; free(G.heat0); G.heat0=NULL;
    }
    puts("tier upload rollback: ok (12 failure cases: int8, per-row int4, grouped int4, FP8)"); return 0;
}
