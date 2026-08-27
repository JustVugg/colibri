/* Optional local A/B for Qwen's default dense-int8 prefill path.  These are
 * the real Qwen3.6 shared-expert dimensions; one matrix is benchmarked at a
 * time so peak RSS stays below 10 MiB.
 *
 *   make tests/bench_qwen36_dense_batch ARCH=native
 *   OMP_NUM_THREADS=<physical cores> ./tests/bench_qwen36_dense_batch
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

enum { S=16, I=2048, O=512, REPS=7 };

static float value(int64_t i, int salt) {
    int v=(int)((i*31+salt*17)%251); return (float)(v-125)/(float)(109+salt);
}

static double scalar(float *y,const float *x,const int8_t *q,const float *sc) {
    double t0=now_s();
    for(int s=0;s<S;s++)matmul_q(y+(int64_t)s*O,x+(int64_t)s*I,q,sc,I,O);
    return now_s()-t0;
}
static double batch(float *y,const float *x,const int8_t *q,const float *sc) {
    double t0=now_s();matmul_q_batch(y,x,q,sc,S,I,O);return now_s()-t0;
}

static void env_set(const char *name,const char *value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static void env_unset(const char *name) {
#ifdef _WIN32
    _putenv_s(name,"");
#else
    unsetenv(name);
#endif
}

static double shared_run(Model *m,Layer *l,const float *x,const float *seed,
                         float *out,const char *mode) {
    float *g=falloc(O),*u=falloc(O),*hh=falloc(I);
    if(mode){env_set("QWEN_SHARED_BATCH",mode);env_set("QWEN_DENSE_BATCH",mode);}
    else{env_unset("QWEN_SHARED_BATCH");env_unset("QWEN_DENSE_BATCH");}
    memcpy(out,seed,(size_t)S*I*sizeof(float));double t0=now_s();
    qwen_shared_experts_cpu(m,l,x,S,out,g,u,hh);double dt=now_s()-t0;
    free(g);free(u);free(hh);return dt;
}

static int shared_benchmark(void) {
    Model m;memset(&m,0,sizeof(m));m.c.hidden=I;m.c.shared_inter=O;
    Layer l;memset(&l,0,sizeof(l));
    l.sh_g=falloc((int64_t)O*I);l.sh_u=falloc((int64_t)O*I);
    l.sh_d=falloc((int64_t)I*O);l.sh_gate=falloc(I);
    for(int64_t i=0;i<(int64_t)O*I;i++){l.sh_g[i]=value(i,2);l.sh_u[i]=value(i,3);}
    for(int64_t i=0;i<(int64_t)I*O;i++)l.sh_d[i]=value(i,4);
    for(int i=0;i<I;i++)l.sh_gate[i]=value(i,5);
    qdw_register(l.sh_g,I,O);qdw_register(l.sh_u,I,O);qdw_register(l.sh_d,O,I);
    if(g_qdw_n!=3){fprintf(stderr,"FAIL: expected three dense-int8 copies, got %d\n",g_qdw_n);return 1;}
    float *x=falloc((int64_t)S*I),*seed=falloc((int64_t)S*I);
    float *a=falloc((int64_t)S*I),*b=falloc((int64_t)S*I);
    for(int64_t i=0;i<(int64_t)S*I;i++){x[i]=value(i,6);seed[i]=value(i,7);}
    shared_run(&m,&l,x,seed,a,"0");shared_run(&m,&l,x,seed,b,NULL);
    if(memcmp(a,b,(size_t)S*I*sizeof(float))){fputs("FAIL: shared batch is not exact\n",stderr);return 1;}
    double ts=0,tb=0;enum{R=5};
    for(int r=0;r<R;r++){
        if(r&1){tb+=shared_run(&m,&l,x,seed,b,NULL);ts+=shared_run(&m,&l,x,seed,a,"0");}
        else{ts+=shared_run(&m,&l,x,seed,a,"0");tb+=shared_run(&m,&l,x,seed,b,NULL);}
    }
    ts/=R;tb/=R;
    printf("qwen shared int8: S=%d D=%d I=%d weights=%.1f MiB\n",S,I,O,
           3.0*I*O/1048576.0);
    printf("scalar %.6f s  batch %.6f s  speedup %.2fx  calls %d -> 3\n",ts,tb,ts/tb,S*3);
    for(int i=0;i<g_qdw_n;i++){free(g_qdw[i].q);free(g_qdw[i].sc);}g_qdw_n=0;
    free(l.sh_g);free(l.sh_u);free(l.sh_d);free(l.sh_gate);
    free(x);free(seed);free(a);free(b);env_unset("QWEN_SHARED_BATCH");env_unset("QWEN_DENSE_BATCH");
    return 0;
}

int main(void) {
    env_unset("COLI_DENSE_I8");
    float *x=falloc((int64_t)S*I),*a=falloc((int64_t)S*O),*b=falloc((int64_t)S*O);
    int8_t *q=malloc((size_t)O*I);float *sc=falloc(O);
    if(!q){fputs("OOM qwen dense benchmark\n",stderr);return 2;}
    for(int64_t i=0;i<(int64_t)S*I;i++)x[i]=value(i,1);
    for(int64_t i=0;i<(int64_t)O*I;i++)q[i]=(int8_t)((i*13+5)%255-127);
    for(int o=0;o<O;o++)sc[o]=0.001f*(float)(1+(o*7)%29);
    scalar(a,x,q,sc);batch(b,x,q,sc);
    if(memcmp(a,b,(size_t)S*O*sizeof(float))){fputs("FAIL: batch is not exact\n",stderr);return 1;}
    double ts=0,tb=0;
    for(int r=0;r<REPS;r++){
        if(r&1){tb+=batch(b,x,q,sc);ts+=scalar(a,x,q,sc);}
        else{ts+=scalar(a,x,q,sc);tb+=batch(b,x,q,sc);}
    }
    ts/=REPS;tb/=REPS;
    printf("qwen dense int8: S=%d I=%d O=%d weights=%.1f MiB\n",S,I,O,
           (double)I*O/1048576.0);
    printf("scalar %.6f s  batch %.6f s  speedup %.2fx\n",ts,tb,ts/tb);
    free(x);free(a);free(b);free(q);free(sc);return shared_benchmark();
}
