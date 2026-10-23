/* Every failed initialization must unwind only the resources it acquired. */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include "../compat.h"
#include "qwen36_fake_cuda.h"
static void *owned[128];
static int live, mutexes, conds, fail_stage, sync_step;
static void *remember(void *p) {
    if(p) { if(live==128) abort(); owned[live++]=p; }
    return p;
}
static void *test_malloc(size_t n) {
    if(fail_stage==1 && n==32*64*sizeof(float)) return NULL;
    return remember(malloc(n));
}
static void *test_calloc(size_t n,size_t s) { return remember(calloc(n,s)); }
static void test_free(void *p) {
    if(p) for(int i=0;i<live;i++) if(owned[i]==p) { owned[i]=owned[--live]; break; }
    free(p);
}
static int test_mutex_init(pthread_mutex_t *p,const pthread_mutexattr_t *a) {
    if(++sync_step==fail_stage-1) return EAGAIN;
    int rc=pthread_mutex_init(p,a);if(!rc) mutexes++;return rc;
}
static int test_cond_init(pthread_cond_t *p,const pthread_condattr_t *a) {
    if(++sync_step==fail_stage-1) return EAGAIN;
    int rc=pthread_cond_init(p,a);if(!rc) conds++;return rc;
}
static int test_mutex_destroy(pthread_mutex_t *p) {
    int rc=pthread_mutex_destroy(p);if(!rc) mutexes--;return rc;
}
static int test_cond_destroy(pthread_cond_t *p) {
    int rc=pthread_cond_destroy(p);if(!rc) conds--;return rc;
}
static int test_thread_create(pthread_t *t,const pthread_attr_t *a,
                              void *(*fn)(void *),void *arg) {
    if(fail_stage==5) return EAGAIN;
    return pthread_create(t,a,fn,arg);
}
#define malloc test_malloc
#define calloc test_calloc
#define free test_free
#define pthread_mutex_init test_mutex_init
#define pthread_cond_init test_cond_init
#define pthread_mutex_destroy test_mutex_destroy
#define pthread_cond_destroy test_cond_destroy
#define pthread_create test_thread_create
#include "../qwen36_tier.c"
int main(void) {
    char path[128];snprintf(path,sizeof path,"qt-init-heat-%ld.tmp",(long)getpid());
    FILE *f=fopen(path,"wb");if(!f) return 1;
    uint32_t data[]={0x51544831u,1,2,8,4};
    if(fwrite(data,sizeof data,1,f)!=1){fclose(f);remove(path);return 1;}fclose(f);
    setenv("COLI_CUDA","1",1);setenv("COLI_GPUS","0",1);
    setenv("HEAT_FILE",path,1);setenv("CUDA_EXPERT_GB","0.0625",1);
    int failed=0;
    for(fail_stage=0;fail_stage<=5;fail_stage++) for(int retry=0;retry<2;retry++) {
        setenv("COLI_PLACE",fail_stage==0?"experts=cpu,lmhead=0":"experts=0,lmhead=0",1);
        sync_step=0;
        int rc=qt_init(1,2,64,32,2,1,0,1);
        if(rc || G.on || G.slot || G.heat0 || G.is_x || G.is_x_floats ||
           G_lmh.dev_ok || live || mutexes || conds){
            fprintf(stderr,"init failure stage=%d retry=%d rc=%d allocations=%d mutexes=%d conds=%d\n",
                    fail_stage,retry,rc,live,mutexes,conds);
            failed=1;goto done;
        }
    }
    fail_stage=0;
    float lut[256]={0}, other_lut[256]={0};
    for(int fp8=0;fp8<2;fp8++){
        int rc=fp8 ? qt_init_fp8(1,2,64,32,2,1,lut) : qt_init(1,2,64,32,2,1,0,1);
        if(!rc){failed=1;goto done;}
        QSlot *slots=G.slot;
        float *inputs=G.is_x;
        pthread_t uploader_thread=G.th;
        int allocations=live;
        const float *table=G_fp8_lut;
        if(qt_init(2,2,64,32,2,1,0,1) ||
           qt_init_fp8(2,2,64,32,2,1,other_lut) ||
           !G.on || G.nl!=1 || G.slot!=slots || G.is_x!=inputs ||
           !pthread_equal(G.th,uploader_thread) || live!=allocations ||
           G_fp8_stream!=fp8 || G_fp8_lut!=table){
            fputs("active tier initialization corrupted live state\n",stderr);
            failed=1;goto done;
        }
        qt_shutdown();
        /* Until the separate shutdown PR lands, reclaim its retained host state. */
        free(G.slot);G.slot=NULL;free(G.is_x);G.is_x=NULL;
        free(G.heat0);G.heat0=NULL;free(G.fill_order);G.fill_order=NULL;
        if(conds){pthread_cond_destroy(&G.cv);pthread_cond_destroy(&G.cv_take);}
        if(mutexes) pthread_mutex_destroy(&G.mx);
        if(live || mutexes || conds){failed=1;goto done;}
    }
done:
    remove(path);
    if(!failed) puts("tier init failure: ok (12 failure cases and active int4/FP8 preservation)");
    return failed;
}
