/* Optional A/B for Inkling shared-expert prefill batching.  The dimensions
 * make the bf16 shared weights larger than the LLC of an ordinary desktop,
 * while keeping the whole process comfortably below 100 MiB.  Correctness is
 * a bit-exact precondition; timing is reported but never used as a CI gate.
 *
 *   make tests/bench_inkling_shared_batch ARCH=native
 *   OMP_NUM_THREADS=<physical cores> ./tests/bench_inkling_shared_batch
 */
#define COLI_INKLING_SHARED_BATCH_TEST 1
#define main inkling_main_unused
#include "../inkling.c"
#undef main

enum { S = 16, D = 2048, I = 1024, K = 2, NS = 2, REPS = 3 };

static uint16_t to_bf16(float x) {
    union { float f; uint32_t u; } v = { x };
    return (uint16_t)(v.u >> 16);
}

static float val(int64_t i, int salt) {
    int v = (int)((i * 29 + salt * 43) % 127);
    return (float)(v - 63) / (float)(256 + salt);
}

static void fill_f32(float *p, int64_t n, int salt) {
    for (int64_t i = 0; i < n; i++) p[i] = val(i, salt);
}

static void fill_bf16(uint16_t *p, int64_t n, int salt) {
    for (int64_t i = 0; i < n; i++) p[i] = to_bf16(val(i, salt));
}

static void init_q4(Wt *w, int rows, int cols, int salt) {
    const int gs=64, rb=(cols+1)/2, ng=(cols+gs-1)/gs;
    w->qbits=4; w->gs=gs; w->qn=(int64_t)rows*rb;
    w->q4=malloc((size_t)w->qn); w->qs=malloc((size_t)rows*ng*sizeof(float));
    if(!w->q4||!w->qs){fputs("OOM q4 weights\n",stderr);exit(2);}
    for(int64_t i=0;i<w->qn;i++){
        int lo=(int)((i*5+salt)%16),hi=(int)((i*11+salt*3)%16);
        w->q4[i]=(uint8_t)(lo|(hi<<4));
    }
    for(int64_t i=0;i<(int64_t)rows*ng;i++)w->qs[i]=0.01f*(float)(1+(i+salt)%7);
}

static double one(Model *m, Layer *l, const float *x, const float *wgt,
                  const float *seed, float *out, const char *mode) {
    float *g = falloc(2 * I), *u = g + I, *hh = falloc(D);
    if (mode) setenv("INK_SHARED_BATCH", mode, 1);
    else unsetenv("INK_SHARED_BATCH");
    memcpy(out, seed, (size_t)S * D * sizeof(float));
    double t0 = now_s();
    shared_experts_cpu(m, l, x, S, out, wgt, g, u, hh);
    double dt = now_s() - t0;
    free(g); free(hh);
    return dt;
}

static int measure(const char *format, double weight_mib, Model *m, Layer *l,
                   const float *x, const float *wgt, const float *seed,
                   float *scalar, float *batch) {
    one(m,l,x,wgt,seed,scalar,"0"); one(m,l,x,wgt,seed,batch,NULL);
    if(memcmp(scalar,batch,(size_t)S*D*sizeof(float))){
        int shown=0;float worst=0.f;
        for(int q=0;q<S*D;q++)if(scalar[q]!=batch[q]){
            float d=fabsf(scalar[q]-batch[q]);if(d>worst)worst=d;
            if(shown++<4)fprintf(stderr,"%s diff[%d] scalar=%a batch=%a delta=%g\n",
                                 format,q,scalar[q],batch[q],d);
        }
        fprintf(stderr,"FAIL %s: %d values differ, worst=%g\n",format,shown,worst);return 1;
    }
    double ts=0.0,tb=0.0;
    for(int r=0;r<REPS;r++){
        if(r&1){tb+=one(m,l,x,wgt,seed,batch,NULL);ts+=one(m,l,x,wgt,seed,scalar,"0");}
        else{ts+=one(m,l,x,wgt,seed,scalar,"0");tb+=one(m,l,x,wgt,seed,batch,NULL);}
    }
    ts/=REPS;tb/=REPS;
    printf("inkling shared %s: S=%d D=%d I=%d shared=%d bytes=%.1f MiB\n",
           format,S,D,I,NS,weight_mib);
    printf("scalar %.6f s  batch %.6f s  speedup %.2fx  calls %d -> %d\n",
           ts,tb,ts/tb,S*NS*3,NS*3);
    return 0;
}

int main(void) {
    Model m; memset(&m, 0, sizeof(m));
    m.c.hidden = D; m.c.moe_inter = I; m.c.topk = K; m.c.n_shared = NS;
    Layer l; memset(&l, 0, sizeof(l));
    l.sh_g.h = malloc((size_t)NS * I * D * sizeof(uint16_t));
    l.sh_u.h = malloc((size_t)NS * I * D * sizeof(uint16_t));
    l.sh_d.h = malloc((size_t)NS * D * I * sizeof(uint16_t));
    float *x = falloc((int64_t)S * D);
    float *wgt = falloc((int64_t)S * (K + NS));
    float *seed = falloc((int64_t)S * D);
    float *scalar = falloc((int64_t)S * D), *batch = falloc((int64_t)S * D);
    if (!l.sh_g.h || !l.sh_u.h || !l.sh_d.h) { fputs("OOM weights\n", stderr); return 2; }
    fill_bf16(l.sh_g.h, (int64_t)NS * I * D, 1);
    fill_bf16(l.sh_u.h, (int64_t)NS * I * D, 2);
    fill_bf16(l.sh_d.h, (int64_t)NS * D * I, 3);
    fill_f32(x, (int64_t)S * D, 4);
    fill_f32(wgt, (int64_t)S * (K + NS), 5);
    fill_f32(seed, (int64_t)S * D, 6);

    int failed=measure("bf16",3.0*NS*D*I*sizeof(uint16_t)/1048576.0,
                       &m,&l,x,wgt,seed,scalar,batch);
    free(l.sh_g.h);free(l.sh_u.h);free(l.sh_d.h);
    memset(&l,0,sizeof(l));
    init_q4(&l.sh_g,NS*I,D,1);init_q4(&l.sh_u,NS*I,D,2);init_q4(&l.sh_d,NS*D,I,3);
    double qbytes=(double)(l.sh_g.qn+l.sh_u.qn+l.sh_d.qn);
    qbytes+=(double)(NS*I*((D+63)/64)*2+NS*D*((I+63)/64))*sizeof(float);
    failed|=measure("int4-g64",qbytes/1048576.0,&m,&l,x,wgt,seed,scalar,batch);

    unsetenv("INK_SHARED_BATCH");
    free(l.sh_g.q4);free(l.sh_g.qs);free(l.sh_u.q4);free(l.sh_u.qs);
    free(l.sh_d.q4);free(l.sh_d.qs);
    free(x); free(wgt); free(seed); free(scalar); free(batch);
    return failed;
}
