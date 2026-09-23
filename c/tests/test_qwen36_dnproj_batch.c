/* Run DeltaNet itself: batching must preserve outputs, conv history and state. */
#define main qwen_main_unused
#include "../qwen36.c"
#undef main

/* Define COLI_DNPROJ_REAL_CUDA and link --wrap=coli_cuda_matmul to run
 * the same state-parity test against the real CUDA backend on Linux. */
#ifdef COLI_DNPROJ_REAL_CUDA
#include "../backend_cuda.h"
int __real_coli_cuda_matmul(ColiCudaTensor **,float *,const float *,const void *,
                           const float *,int,int,int,int,int,int);
#define dn_fake_matmul __real_coli_cuda_matmul
#define DN_MATMUL __wrap_coli_cuda_matmul
#else
/* Wrap the shared fake so this test can inject a failure at a block boundary. */
#define coli_cuda_matmul dn_fake_matmul
#include "qwen36_fake_cuda.h"
#undef coli_cuda_matmul
#define DN_MATMUL coli_cuda_matmul
#endif
static int calls, fail_call, max_rows, last_rows;
int DN_MATMUL(ColiCudaTensor **t,float *y,const float *x,const void *w,
                    const float *sc,int fmt,int S,int I,int O,int dev,int gs) {
    calls++; if (S > max_rows) max_rows = S; last_rows = S;
    if (calls == fail_call) { for(int i=0;i<S*O;i++) y[i] = NAN; return 0; }
    return dn_fake_matmul(t,y,x,w,sc,fmt,S,I,O,dev,gs);
}
#include "../qwen36_tier.c"

enum { H=8, KH=1, VH=2, KD=2, VD=3, C=10, Z=6, O=C+Z, CK=3 };
static int failed;
#define CHECK(c) do { if(!(c)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);failed++;} } while(0)
static float *weights(int n) {
    float *p = calloc(n,sizeof(float));
    for(int i=0;i<n;i++) p[i] = (i%11 - 5)*0.03125f;
    return p;
}
static void state(Model *m) {
    m->DN_rec = calloc(1,sizeof(float *)); m->DN_conv = calloc(1,sizeof(float *));
    m->DN_rec[0] = calloc(VH*KD*VD,sizeof(float)); m->DN_conv[0] = calloc(C*(CK-1),sizeof(float));
}
static void clear_state(Model *m) {
    memset(m->DN_rec[0],0,VH*KD*VD*sizeof(float));
    memset(m->DN_conv[0],0,C*(CK-1)*sizeof(float));
}
static void agree(const float *a,const float *b,int n) {
    for(int i=0;i<n;i++) CHECK(isfinite(a[i]) && isfinite(b[i]) && fabsf(a[i]-b[i]) < 1e-5f);
}
int main(void) {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    setenv("COLI_DENSE_IDOT","0",1);
    Model m={0}; Layer l={0};
    m.c.hidden=H; m.c.dn_kheads=KH; m.c.dn_vheads=VH;
    m.c.dn_kdim=KD; m.c.dn_vdim=VD; m.c.dn_convk=CK; m.c.dn_conv_dim=C; m.c.eps=1e-6f;
    l.dn_qkv=weights(C*H); l.dn_z=weights(Z*H); l.dn_out=weights(H*Z);
    l.dn_b=weights(VH*H); l.dn_a=weights(VH*H); l.dn_conv=weights(C*CK);
    l.dn_alog=weights(VH); l.dn_dtbias=weights(VH); l.dn_norm=weights(VD);
    qdw_register(l.dn_qkv,H,C); qdw_register(l.dn_z,H,Z);
    int8_t q[O*H]; float sc[O];
    memcpy(q,g_qdw[0].q,C*H); memcpy(q+C*H,g_qdw[1].q,Z*H);
    memcpy(sc,g_qdw[0].sc,C*sizeof(float)); memcpy(sc+C,g_qdw[1].sc,Z*sizeof(float));
    state(&m);
    enum { S=259 };
    float *x=weights(S*H), cpu[S*H], gpu[S*H], rec[VH*KD*VD], conv[C*(CK-1)];
    deltanet(&m,&l,0,x,S,0,cpu);
    memcpy(rec,m.DN_rec[0],sizeof(rec)); memcpy(conv,m.DN_conv[0],sizeof(conv));
#ifdef COLI_DNPROJ_REAL_CUDA
    int device=0;
    if(!coli_cuda_init(&device,1)) return 1;
#else
    fake_dense_compute=1;
#endif
    for(int failure=0;failure<=2;failure++) {
        clear_state(&m); calls=0; max_rows=0; fail_call=failure;
        CHECK(qt_dnproj_init(0,q,sc,H,O,0));
        deltanet(&m,&l,0,x,S,0,gpu);
        CHECK(calls == (failure==1 ? 1 : 2));
        CHECK(max_rows==256);
        if(failure!=1) CHECK(last_rows==3);
        agree(cpu,gpu,S*H); agree(rec,m.DN_rec[0],VH*KD*VD); agree(conv,m.DN_conv[0],C*(CK-1));
        CHECK(qt_dnproj_ready(0) == !failure);
        coli_cuda_tensor_free(G_dnp[0].t); memset(&G_dnp[0],0,sizeof(G_dnp[0]));
        qt_shutdown();
    }
    CHECK(dnproj_batch_rows(10000,65536,65536)==64);
    CHECK(dnproj_batch_rows(1,H,O)==1);
    /* Decode still uses a single projection and the same state transition. */
    clear_state(&m); deltanet(&m,&l,0,x,1,0,cpu);
    memcpy(rec,m.DN_rec[0],sizeof(rec)); memcpy(conv,m.DN_conv[0],sizeof(conv));
    clear_state(&m); calls=0; fail_call=0;
    CHECK(qt_dnproj_init(0,q,sc,H,O,0));
    deltanet(&m,&l,0,x,1,0,gpu);
    CHECK(calls==1 && last_rows==1);
    agree(cpu,gpu,H); agree(rec,m.DN_rec[0],VH*KD*VD); agree(conv,m.DN_conv[0],C*(CK-1));
    coli_cuda_tensor_free(G_dnp[0].t); memset(&G_dnp[0],0,sizeof(G_dnp[0]));
    qt_shutdown();
    /* Prefix extension must preserve state across separate calls, including a
     * split immediately before, on, and after the 256-row block boundary. */
    const int splits[]={1,255,256,257};
    for(int split_index=0;split_index<4;split_index++) for(int failure=0;failure<2;failure++){
        int split=splits[split_index];
        float cpu_next[H], gpu_next[H];
        clear_state(&m);
        deltanet(&m,&l,0,x,S,0,cpu);
        deltanet(&m,&l,0,x,1,S,cpu_next);
        memcpy(rec,m.DN_rec[0],sizeof(rec)); memcpy(conv,m.DN_conv[0],sizeof(conv));
        clear_state(&m);calls=0;
        fail_call=failure ? (split+255)/256+1 : 0;
        CHECK(qt_dnproj_init(0,q,sc,H,O,0));
        deltanet(&m,&l,0,x,split,0,gpu);
        deltanet(&m,&l,0,x+(size_t)split*H,S-split,split,gpu+(size_t)split*H);
        CHECK(calls==(split+255)/256+(failure?1:(S-split+255)/256));
        int before_decode=calls;
        deltanet(&m,&l,0,x,1,S,gpu_next);
        CHECK(calls==before_decode+!failure);
        if(!failure) CHECK(last_rows==1);
        CHECK(qt_dnproj_ready(0)==!failure);
        agree(cpu,gpu,S*H); agree(cpu_next,gpu_next,H);
        agree(rec,m.DN_rec[0],VH*KD*VD); agree(conv,m.DN_conv[0],C*(CK-1));
        coli_cuda_tensor_free(G_dnp[0].t); memset(&G_dnp[0],0,sizeof(G_dnp[0]));
        qt_shutdown();
    }
#ifdef COLI_DNPROJ_REAL_CUDA
    coli_cuda_shutdown();
#endif
    free(x); free(m.DN_rec[0]); free(m.DN_conv[0]); free(m.DN_rec); free(m.DN_conv);
    free(l.dn_qkv); free(l.dn_z); free(l.dn_out); free(l.dn_b); free(l.dn_a); free(l.dn_conv);
    free(l.dn_alog); free(l.dn_dtbias); free(l.dn_norm);
    for(int i=0;i<g_qdw_n;i++){free(g_qdw[i].q);free(g_qdw[i].sc);} g_qdw_n=0;
    printf("DeltaNet batch: %s\n",failed?"FAIL":"outputs and recurrent state match; 259 calls -> 2");
    return failed!=0;
}
