/* Resume acceptance test (AGENTS.md §15) + dataset reader test, on the tiny
 * oracle model with a synthetic coli-sft-v1 dataset the test writes itself:
 *
 *   run A: 20 uninterrupted training steps;
 *   run B: 10 steps -> tckpt_save -> fresh trainer + lora_load + tckpt_load
 *          -> 10 more steps.
 *
 * Same seeds, same kernels, same order => adapters and losses must match to
 * float noise (expected bitwise; asserted < 1e-6 rel). Also exercises: masked
 * CE (prompt/pad tokens carry no loss), deterministic epoch reshuffle, and the
 * dataset open-time validation. Skips without ./glm_tiny (no torch needed). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include "../train/qlora_ops.h"
#include "../train/train_model.h"
#include "../train/dataset.h"
#include "../train/checkpoint.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

enum { RANK=4, SEQ=24, NSAMP=13, STEPS_A=20, STEPS_HALF=10 };
static const float ALPHA=8.f, LR=1e-2f;
static const uint64_t DSEED=77;

static uint32_t rs=0xC0FFEE;
static uint32_t xr(void){ rs^=rs<<13; rs^=rs>>17; rs^=rs<<5; return rs; }
static float fr(void){ return (float)(int32_t)xr()/2147483648.0f; }

/* write a synthetic coli-sft-v1 split: NSAMP samples, vocab-bounded ids,
 * fake prompt/completion mask, one deliberately-too-short sample (skipped) */
static void write_dataset(const char *dir, int vocab){
    char p[1024]; mkdir(dir,0755);
    int lens[NSAMP]; int64_t tot=0;
    for(int i=0;i<NSAMP;i++){ lens[i]=(i==5)?1:(int)(6+xr()%40); tot+=lens[i]; }
    uint32_t *tok=malloc(tot*4); uint8_t *msk=malloc(tot);
    for(int64_t i=0;i<tot;i++){ tok[i]=xr()%vocab; msk[i]=1; }
    int64_t off=0;
    for(int i=0;i<NSAMP;i++){ int cut=lens[i]>2?lens[i]/3:0;
        for(int j=0;j<cut;j++) msk[off+j]=0;
        off+=lens[i]; }
    snprintf(p,sizeof(p),"%s/train.bin",dir); FILE *f=fopen(p,"wb"); fwrite(tok,4,tot,f); fclose(f);
    snprintf(p,sizeof(p),"%s/train.msk",dir); f=fopen(p,"wb"); fwrite(msk,1,tot,f); fclose(f);
    snprintf(p,sizeof(p),"%s/train.idx",dir); f=fopen(p,"wb");
    int64_t hdr[2]={TDS_MAGIC,NSAMP}; fwrite(hdr,8,2,f);
    int64_t cum=0; fwrite(&cum,8,1,f);
    for(int i=0;i<NSAMP;i++){ cum+=lens[i]; fwrite(&cum,8,1,f); }
    fclose(f);
    free(tok); free(msk);
}

typedef struct {
    Model *M; LoraAdapter *lora; TT *tt; TDataset ds;
    AdamW opt; float **sm,**sv;
} Trainer;

static void trainer_new(Trainer *tr, Model *M, const char *adir, const char *ddir){
    tr->M=M;
    tr->lora=lora_load(adir,cfg_fingerprint(&M->c),0);
    if(!tr->lora){ fprintf(stderr,"adapter load failed\n"); exit(1); }
    tr->tt=tt_init(M,tr->lora,SEQ+1,1,8);
    if(tds_open(&tr->ds,ddir,"train",SEQ,DSEED)){ fprintf(stderr,"dataset open failed\n"); exit(1); }
    tr->opt=adamw_default(LR);
    tr->sm=calloc(tr->lora->n*2,sizeof(float*)); tr->sv=calloc(tr->lora->n*2,sizeof(float*));
    for(int i=0;i<tr->lora->n;i++){
        LoraTensor *t=&tr->lora->t[i];
        tr->sm[2*i]=calloc((size_t)t->rank*t->I,4);   tr->sv[2*i]=calloc((size_t)t->rank*t->I,4);
        tr->sm[2*i+1]=calloc((size_t)t->O*t->rank,4); tr->sv[2*i+1]=calloc((size_t)t->O*t->rank,4);
    }
}

static float trainer_step(Trainer *tr){
    uint32_t tok32[SEQ+1]; uint8_t msk[SEQ+1]; int tok[SEQ+1];
    tds_next(&tr->ds,tok32,msk);
    for(int i=0;i<SEQ+1;i++) tok[i]=(int)tok32[i];
    float l=tt_forward_masked(tr->tt,tok,msk);
    tt_backward_masked(tr->tt,tok,msk);
    adamw_tick(&tr->opt);
    for(int i=0;i<tr->lora->n;i++){
        LoraTensor *t=&tr->lora->t[i];
        adamw_step(&tr->opt,t->A,tr->tt->dA[i],tr->sm[2*i],tr->sv[2*i],(int64_t)t->rank*t->I);
        adamw_step(&tr->opt,t->B,tr->tt->dB[i],tr->sm[2*i+1],tr->sv[2*i+1],(int64_t)t->O*t->rank);
    }
    return l;
}

static float maxrel(const float *a,const float *b,int64_t n){
    float mx=0,mb=0;
    for(int64_t i=0;i<n;i++){ float d=fabsf(a[i]-b[i]); if(d>mx)mx=d;
        if(fabsf(b[i])>mb)mb=fabsf(b[i]); }
    return mx/(mb+1e-8f);
}

int main(void){
    const char *snap="./glm_tiny";
    struct stat sb; char cfgp[512]; snprintf(cfgp,sizeof(cfgp),"%s/config.json",snap);
    if(stat(cfgp,&sb)){ printf("resume test: skipped (need %s; python3 tools/make_glm_oracle.py)\n",snap); return 0; }
    const char *tmp=getenv("TMPDIR")?getenv("TMPDIR"):"/tmp";
    char ddir[1024],adir[1024],cdir[1024];
    snprintf(ddir,sizeof(ddir),"%s/coli_resume_data_%d",tmp,(int)getpid());
    snprintf(adir,sizeof(adir),"%s/coli_resume_ad_%d",tmp,(int)getpid());
    snprintf(cdir,sizeof(cdir),"%s/coli_resume_ck_%d",tmp,(int)getpid());

    unsetenv("ADAPTER");
    Model M; model_init(&M,snap,64,16,16);
    write_dataset(ddir,M.c.vocab);

    /* seed adapter: rank-4 random on every layer's o_proj, saved once, loaded
     * by both runs — identical starting point, exercised through lora_save/load */
    {
        LoraAdapter seed={0};
        seed.n=M.c.n_layers; seed.rank=RANK; seed.alpha=ALPHA;
        seed.base_fp=cfg_fingerprint(&M.c);
        snprintf(seed.base_model,sizeof(seed.base_model),"glm_tiny");
        seed.t=calloc(seed.n,sizeof(LoraTensor));
        for(int li=0;li<seed.n;li++){
            LoraTensor *t=&seed.t[li];
            t->layer=li; t->tgt=LORA_T_O; t->rank=RANK;
            t->I=M.L[li].o.I; t->O=M.L[li].o.O; t->scale=ALPHA/RANK;
            t->A=falloc((int64_t)RANK*t->I); t->B=falloc((int64_t)t->O*RANK);
            for(int i=0;i<RANK*t->I;i++) t->A[i]=0.05f*fr();
            for(int i=0;i<t->O*RANK;i++) t->B[i]=0.05f*fr();
        }
        CHECK(lora_save(adir,&seed)==0);
    }

    /* run A: 20 uninterrupted steps */
    Trainer A; trainer_new(&A,&M,adir,ddir);
    float lossA_last=0;
    for(int s=0;s<STEPS_A;s++) lossA_last=trainer_step(&A);
    CHECK(isfinite(lossA_last));
    CHECK(A.ds.epoch>0);                       /* 20 steps over 12 usable samples wrapped */

    /* run B: 10 steps, checkpoint, fresh everything, resume, 10 more */
    Trainer B; trainer_new(&B,&M,adir,ddir);
    for(int s=0;s<STEPS_HALF;s++) trainer_step(&B);
    TCkptState st={.step=STEPS_HALF,.opt_t=B.opt.t,
                   .epoch=B.ds.epoch,.cursor=B.ds.cursor,
                   .data_seed=DSEED,.base_fp=cfg_fingerprint(&M.c)};
    CHECK(tckpt_save(cdir,B.lora,&st,B.sm,B.sv)==0);

    Trainer C; trainer_new(&C,&M,cdir,ddir);   /* adapter reloaded from checkpoint */
    TCkptState rst;
    CHECK(tckpt_load(cdir,C.lora,&rst,C.sm,C.sv)==0);
    CHECK(rst.step==STEPS_HALF && rst.data_seed==DSEED && rst.base_fp==cfg_fingerprint(&M.c));
    C.opt.t=rst.opt_t;
    C.ds.epoch=rst.epoch; C.ds.cursor=rst.cursor; tds_shuffle(&C.ds);
    float lossC_last=0;
    for(int s=0;s<STEPS_HALF;s++) lossC_last=trainer_step(&C);

    /* A vs resumed B+C: same losses, same adapters */
    float er=fabsf(lossC_last-lossA_last)/(fabsf(lossA_last)+1e-8f);
    float wmax=0;
    for(int i=0;i<A.lora->n;i++){
        LoraTensor *ta=&A.lora->t[i], *tc=&C.lora->t[i];
        float ea=maxrel(tc->A,ta->A,(int64_t)ta->rank*ta->I);
        float eb=maxrel(tc->B,ta->B,(int64_t)ta->O*ta->rank);
        if(ea>wmax)wmax=ea; if(eb>wmax)wmax=eb;
    }
    printf("resume: lossA %.6f lossResumed %.6f (rel %.2e) | adapter max rel %.2e | epochs %d\n",
           lossA_last,lossC_last,er,wmax,A.ds.epoch);
    CHECK(er<1e-6f);
    CHECK(wmax<1e-6f);

    /* corrupt-state rejection: truncated train_state.bin must fail cleanly */
    {
        char p[1200]; snprintf(p,sizeof(p),"%s/train_state.bin",cdir);
        FILE *f=fopen(p,"rb+"); CHECK(f!=NULL);
        CHECK(ftruncate(fileno(f),64)==0); fclose(f);
        TCkptState junk;
        CHECK(tckpt_load(cdir,C.lora,&junk,C.sm,C.sv)!=0);
    }

    char cmd[4096];
    snprintf(cmd,sizeof(cmd),"rm -rf %s %s %s",ddir,adir,cdir); system(cmd);
    puts("dataset + checkpoint resume tests: ok");
    return 0;
}
