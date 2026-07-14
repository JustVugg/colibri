/* Same-binary A/B microbenchmark for the runtime-dispatched AArch64 SMMLA
 * expert kernel. Uses the real GLM-5.2 expert shapes and reports medians after
 * alternating path order, so neither path always receives the warmer cache. */
#define main coli_glm_main_unused
#include "../glm.c"
#undef main

#if COLI_HAVE_I8MM
static uint32_t rng_state=0x9e3779b9u;
static uint32_t xr(void){ rng_state^=rng_state<<13; rng_state^=rng_state>>17; rng_state^=rng_state<<5; return rng_state; }
static int cmp_double(const void *a,const void *b){
    double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y);
}

static void call_path(int fmt, int use_i8mm, float *y, const int8_t *x, const float *sx,
                      const uint8_t *w, const float *sc, int S, int I, int O){
    g_i8mm=use_i8mm;
    if(fmt==1) matmul_q_idot(y,x,sx,(const int8_t*)w,sc,S,I,O);
    else matmul_i4_idot(y,x,sx,w,sc,S,I,O);
}

static int bench_shape(int fmt, int S, int I, int O, const char *label){
    enum { REPS=21 };
    double tn[REPS],tm[REPS]; volatile float sink=0; int ok=0;
    int rb=(I+1)/2; size_t wn=fmt==1?(size_t)O*I:(size_t)O*rb;
    uint8_t *w=malloc(wn); int8_t *x=malloc((size_t)S*I);
    float *yn=malloc((size_t)S*O*sizeof(float)),*ym=malloc((size_t)S*O*sizeof(float));
    float *sc=malloc((size_t)O*sizeof(float)),*sx=malloc((size_t)S*sizeof(float));
    if(!w||!x||!yn||!ym||!sc||!sx){ fprintf(stderr,"bench_i8mm: OOM\n"); goto cleanup; }
    for(size_t i=0;i<wn;i++) w[i]=(uint8_t)xr();
    for(int i=0;i<S*I;i++) x[i]=(int8_t)((int)(xr()%255)-127);
    for(int i=0;i<O;i++) sc[i]=0.001f+(float)(i%17)*0.0001f;
    for(int i=0;i<S;i++) sx[i]=0.002f+(float)i*0.0001f;

    call_path(fmt,0,yn,x,sx,w,sc,S,I,O);
    call_path(fmt,1,ym,x,sx,w,sc,S,I,O);
    if(memcmp(yn,ym,(size_t)S*O*sizeof(float))!=0){
        fprintf(stderr,"bench_i8mm: numeric mismatch: %s\n",label); goto cleanup;
    }
    for(int r=0;r<REPS;r++){
        int first=r&1;
        double t=now_s(); call_path(fmt,first,first?ym:yn,x,sx,w,sc,S,I,O); double dt0=now_s()-t;
        t=now_s(); call_path(fmt,!first,first?yn:ym,x,sx,w,sc,S,I,O); double dt1=now_s()-t;
        if(first){ tm[r]=dt0; tn[r]=dt1; } else { tn[r]=dt0; tm[r]=dt1; }
        sink+=yn[r%(S*O)]+ym[(r*7)%(S*O)];
    }
    qsort(tn,REPS,sizeof(double),cmp_double); qsort(tm,REPS,sizeof(double),cmp_double);
    const char *selected=(fmt==2&&S==1&&I<6144)?"NEON guard":"SMMLA";
    printf("%-4s %-18s S=%-2d %7.3f %8.3f %7.3fx  %s\n",
           fmt==1?"i8":"i4",label,S,tn[REPS/2]*1e3,tm[REPS/2]*1e3,
           tn[REPS/2]/tm[REPS/2],selected);
    (void)sink; ok=1;
cleanup:
    free(w);free(x);free(yn);free(ym);free(sc);free(sx); return ok;
}

int main(void){
    if(!cpu_has_i8mm()){ puts("i8mm benchmark: skipped (SMMLA unavailable)"); return 0; }
    puts("format shape              S   NEON-ms SMMLA-ms speedup  dispatch");
    int ok=1;
    for(int fmt=1;fmt<=2;fmt++){
        ok&=bench_shape(fmt,1,6144,2048,"gate/up");
        ok&=bench_shape(fmt,1,2048,6144,"down");
        ok&=bench_shape(fmt,2,6144,2048,"gate/up");
        ok&=bench_shape(fmt,2,2048,6144,"down");
        ok&=bench_shape(fmt,4,6144,2048,"gate/up");
        ok&=bench_shape(fmt,4,2048,6144,"down");
    }
    return ok?0:1;
}
#else
int main(void){ puts("i8mm benchmark: skipped (not an AArch64 i8mm build)"); return 0; }
#endif
