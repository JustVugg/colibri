/* Milestone 3 gate: the C training path (train/train_model.h) must reproduce
 * PyTorch on the tiny GLM oracle — forward loss, adapter gradients dA/dB, and
 * an 8-step AdamW loss trajectory ending in matching adapter tensors.
 *
 * Needs artifacts produced by torch tooling (not part of make check unless
 * present — prints "skipped" otherwise):
 *   python3 tools/make_glm_oracle.py
 *   python3 tools/make_lora_adapter.py --model ./glm_tiny --out /tmp/ad_train \
 *       --rank 4 --alpha 8 --init random --seed 3
 *   python3 tools/make_train_oracle.py --adapter /tmp/ad_train --out /tmp/train_fixture.json
 *
 * Tolerances (f32 end-to-end vs f32 PyTorch, different summation order,
 * 5 layers + softmax/CE): documented per check below, tightened from measured
 * headroom, not loosened to pass. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include "../train/qlora_ops.h"
#include "../train/train_model.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static jval *jreq(jval *o, const char *k){
    jval *v=json_get(o,k);
    if(!v){ fprintf(stderr,"fixture: missing %s\n",k); exit(1); }
    return v;
}
static void jfloats(jval *arr, float *out, int64_t n){
    if(arr->t!=J_ARR||arr->len!=n){ fprintf(stderr,"fixture: bad array (want %lld got %d)\n",(long long)n,arr?arr->len:-1); exit(1); }
    for(int64_t i=0;i<n;i++) out[i]=(float)arr->kids[i]->num;
}
/* max |a-b| / (max|b|+eps) over n — one scale-free number per tensor */
static float relerr(const float *a, const float *b, int64_t n){
    float mx=0,mb=0;
    for(int64_t i=0;i<n;i++){ float d=fabsf(a[i]-b[i]); if(d>mx)mx=d;
        if(fabsf(b[i])>mb)mb=fabsf(b[i]); }
    return mx/(mb+1e-8f);
}

int main(void){
    const char *snap="./glm_tiny";
    const char *fx=getenv("TRAIN_FIXTURE")?getenv("TRAIN_FIXTURE"):"/tmp/train_fixture.json";
    const char *ad=getenv("TRAIN_ADAPTER")?getenv("TRAIN_ADAPTER"):"/tmp/ad_train";
    struct stat sb;
    char cfgp[512]; snprintf(cfgp,sizeof(cfgp),"%s/config.json",snap);
    if(stat(cfgp,&sb)||stat(fx,&sb)){
        printf("tiny training oracle test: skipped (need %s + %s; see file header)\n",snap,fx);
        return 0;
    }
    FILE *f=fopen(fx,"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char *txt=malloc(sz+1); CHECK(fread(txt,1,sz,f)==(size_t)sz); fclose(f); txt[sz]=0;
    char *arena=NULL; jval *J=json_parse(txt,&arena); CHECK(J);

    unsetenv("ADAPTER");                      /* we attach the adapter ourselves */
    Model M; model_init(&M,snap,64,16,16);
    LoraAdapter *lora=lora_load(ad,cfg_fingerprint(&M.c),0);
    CHECK(lora);

    jval *jt=jreq(J,"tokens");
    int T=jt->len, *tok=malloc(T*sizeof(int));
    for(int i=0;i<T;i++) tok[i]=(int)jt->kids[i]->num;
    int steps=(int)jreq(J,"steps")->num;
    float lr=(float)jreq(J,"lr")->num;

    TT *tt=tt_init(&M,lora,T,0,16);   /* cache fits every expert */

    /* 1) forward loss parity */
    float loss0=(float)jreq(J,"loss0")->num;
    float loss=tt_forward(tt,tok);
    float lerr=fabsf(loss-loss0)/loss0;
    printf("loss0: C %.6f vs torch %.6f (rel %.2e)\n",loss,loss0,lerr);
    CHECK(lerr<2e-5f);

    /* 2) gradient parity per adapted layer */
    tt_backward(tt,tok);
    jval *jg=jreq(J,"grads");
    float gmax=0;
    for(int i=0;i<lora->n;i++){
        LoraTensor *t=&lora->t[i];
        char key[16]; snprintf(key,sizeof(key),"%d",t->layer);
        jval *g=jreq(jg,key);
        int64_t na=(int64_t)t->rank*t->I, nb=(int64_t)t->O*t->rank;
        float *ra=malloc(na*4), *rb=malloc(nb*4);
        jfloats(jreq(g,"dA"),ra,na); jfloats(jreq(g,"dB"),rb,nb);
        float ea=relerr(tt->dA[i],ra,na), eb=relerr(tt->dB[i],rb,nb);
        printf("layer %d: dA rel %.2e | dB rel %.2e\n",t->layer,ea,eb);
        if(ea>gmax)gmax=ea; if(eb>gmax)gmax=eb;
        free(ra); free(rb);
    }
    CHECK(gmax<5e-4f);

    /* 2b) M4: activation-checkpointed mode — same loss, same gradients (the
     * recompute replays routing through the same kernels in the same order, so
     * parity should be essentially bitwise), with O(1-layer) stash memory. */
    TT *tc=tt_init(&M,lora,T,1,4);    /* cap 4 < 8 experts/layer: forces real streaming */
    float lck=tt_forward(tc,tok);
    CHECK(fabsf(lck-loss)<=1e-7f);
    tt_backward(tc,tok);
    float cmax=0;
    for(int i=0;i<lora->n;i++){
        LoraTensor *t=&lora->t[i];
        float ea=relerr(tc->dA[i],tt->dA[i],(int64_t)t->rank*t->I);
        float eb=relerr(tc->dB[i],tt->dB[i],(int64_t)t->O*t->rank);
        if(ea>cmax)cmax=ea; if(eb>cmax)cmax=eb;
    }
    printf("checkpointed vs retained: grad max rel %.2e | stash %.2f MB vs %.2f MB\n",
           cmax,tc->bytes_stash/1048576.0,tt->bytes_stash/1048576.0);
    CHECK(cmax<1e-6f);
    CHECK(tc->bytes_stash*2<tt->bytes_stash);   /* bounded: ~1 layer + checkpoints */

    /* 2c) M5: streamed experts — deduplicated layer-wise loads, honest I/O.
     * 2 sparse layers x 8 experts = 16 unique (layer,expert) pairs max; without
     * dedup a fwd+bwd pass would issue >= 2*2*T*K = 256 loads. The 4-slot
     * trainer must evict and reload (loads > uniques) yet match bitwise. */
    printf("expert I/O: retained(cap16) %llu loads %llu hits | ckpt(cap4) %llu loads %llu hits | %.2f MB read\n",
           (unsigned long long)tt->ec_loads,(unsigned long long)tt->ec_hits,
           (unsigned long long)tc->ec_loads,(unsigned long long)tc->ec_hits,
           (tt->ec_bytes+tc->ec_bytes)/1048576.0);
    CHECK(tt->ec_loads<=16);                    /* one load per unique expert */
    CHECK(tt->ec_hits>0);                       /* backward reuses the cache */
    CHECK(tc->ec_loads>tt->ec_loads);           /* small cache streams (evicts+reloads) */
    CHECK(tc->ec_bytes>0);

    /* 3) AdamW trajectory + final adapter tensors — run in CHECKPOINTED mode:
     * training end-to-end must work with recompute, not just one backward. */
    jval *jl=jreq(J,"losses"); CHECK(jl->len==steps+1);
    AdamW opt=adamw_default(lr);
    float **sm=calloc(lora->n*2,sizeof(float*)), **sv=calloc(lora->n*2,sizeof(float*));
    for(int i=0;i<lora->n;i++){
        LoraTensor *t=&lora->t[i];
        sm[2*i]=calloc((size_t)t->rank*t->I,4);   sv[2*i]=calloc((size_t)t->rank*t->I,4);
        sm[2*i+1]=calloc((size_t)t->O*t->rank,4); sv[2*i+1]=calloc((size_t)t->O*t->rank,4);
    }
    float trajmax=0;
    for(int s=0;s<steps;s++){
        float l=(s==0)?lck:tt_forward(tc,tok);
        float ref=(float)jl->kids[s]->num, e=fabsf(l-ref)/ref;
        if(e>trajmax)trajmax=e;
        tt_backward(tc,tok);
        adamw_tick(&opt);
        for(int i=0;i<lora->n;i++){
            LoraTensor *t=&lora->t[i];
            CHECK(adamw_step(&opt,t->A,tc->dA[i],sm[2*i],sv[2*i],(int64_t)t->rank*t->I)==0);
            CHECK(adamw_step(&opt,t->B,tc->dB[i],sm[2*i+1],sv[2*i+1],(int64_t)t->O*t->rank)==0);
        }
    }
    float lfin=tt_forward(tc,tok);
    float reff=(float)jl->kids[steps]->num, ef=fabsf(lfin-reff)/reff;
    if(ef>trajmax)trajmax=ef;
    printf("trajectory: %.4f -> %.4f (torch %.4f -> %.4f), max rel %.2e\n",
           loss,lfin,(float)jl->kids[0]->num,reff,trajmax);
    CHECK(trajmax<1e-4f);
    CHECK(lfin<loss);                          /* it must actually learn */

    jval *jf=jreq(J,"final");
    float fmax=0;
    for(int i=0;i<lora->n;i++){
        LoraTensor *t=&lora->t[i];
        char key[16]; snprintf(key,sizeof(key),"%d",t->layer);
        jval *g=jreq(jf,key);
        int64_t na=(int64_t)t->rank*t->I, nb=(int64_t)t->O*t->rank;
        float *ra=malloc(na*4), *rb=malloc(nb*4);
        jfloats(jreq(g,"A"),ra,na); jfloats(jreq(g,"B"),rb,nb);
        float ea=relerr(t->A,ra,na), eb=relerr(t->B,rb,nb);
        if(ea>fmax)fmax=ea; if(eb>fmax)fmax=eb;
        free(ra); free(rb);
    }
    printf("final adapters after %d steps: max rel %.2e\n",steps,fmax);
    CHECK(fmax<1e-3f);

    puts("tiny training oracle test: ok");
    return 0;
}
