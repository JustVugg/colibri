/* coli_train — streamed QLoRA/SFT trainer CLI (AGENTS.md §24, pre-M7).
 *
 * Assembles the proven pieces: model_init (frozen base, snapshot fmts),
 * lora.h adapters (o_proj allowlist), train_model.h checkpointed backward with
 * streamed experts, coli-sft-v1 dataset, AdamW, budget manager, atomic
 * checkpoint/resume. Micro-batch is 1 sequence; --grad-accum averages
 * gradients across windows before each optimizer step.
 *
 *   ./coli_train --model ./glm_tiny --data /tmp/sft --adapter-out ./ad-run1 \
 *       --ram 52 --seq-len 128 --grad-accum 4 --rank 8 --alpha 16 \
 *       --lr 1e-4 --steps 100 --save-every 25
 *
 * Fresh runs use standard LoRA init (A ~ N(0,.02), B = 0 -> step-0 model
 * IDENTICAL to base). --resume DIR continues a previous run exactly (§15).
 * Every step prints the §19 metrics line; the budget manager aborts the run
 * on ceiling breach or swap growth — a swapping run is a failed run. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include "qlora_ops.h"
#include "budget.h"       /* BEFORE train_model.h: activates the in-step cache-shrink guard */
#include "train_model.h"
#include "dataset.h"
#include "checkpoint.h"
#include <signal.h>

static volatile sig_atomic_t g_train_stop=0;
static void on_int(int sig){ (void)sig; g_train_stop=1; }

static uint64_t lcg=0x2545F4914F6CDD1Dull;
static float nrand(void){                       /* Box-Muller-ish cheap normal */
    lcg=lcg*6364136223846793005ull+1442695040888963407ull;
    double u=((lcg>>11)+1)*(1.0/9007199254740992.0);
    lcg=lcg*6364136223846793005ull+1442695040888963407ull;
    double v=((lcg>>11)+1)*(1.0/9007199254740992.0);
    return (float)(sqrt(-2.0*log(u))*cos(2.0*M_PI*v));
}

static const char *arg_s(int argc,char**argv,const char*k,const char*d){
    for(int i=1;i<argc-1;i++) if(!strcmp(argv[i],k)) return argv[i+1];
    return d;
}
static double arg_f(int argc,char**argv,const char*k,double d){
    const char *v=arg_s(argc,argv,k,NULL); return v?atof(v):d;
}
static int arg_i(int argc,char**argv,const char*k,int d){
    const char *v=arg_s(argc,argv,k,NULL); return v?atoi(v):d;
}

int main(int argc, char **argv){
    const char *model=arg_s(argc,argv,"--model",NULL);
    const char *data =arg_s(argc,argv,"--data",NULL);
    const char *out  =arg_s(argc,argv,"--adapter-out",NULL);
    const char *resume=arg_s(argc,argv,"--resume",NULL);
    if(!model||!data||!out){
        fprintf(stderr,"usage: coli_train --model DIR --data DIR --adapter-out DIR\n"
                       "  [--resume DIR] [--ram GB=52] [--seq-len N=128] [--grad-accum N=4]\n"
                       "  [--rank N=8] [--alpha F=2*rank] [--lr F=1e-4] [--steps N=100]\n"
                       "  [--save-every N=25] [--seed N=0] [--ebits N=16] [--dbits N=16]\n"
                       "  [--expert-slots N=auto] [--targets attention]\n");
        return 1;
    }
    int seq=arg_i(argc,argv,"--seq-len",128), accum=arg_i(argc,argv,"--grad-accum",4);
    int rank=arg_i(argc,argv,"--rank",8), steps=arg_i(argc,argv,"--steps",100);
    int save_every=arg_i(argc,argv,"--save-every",25);
    float alpha=(float)arg_f(argc,argv,"--alpha",2.0*rank);
    float lr=(float)arg_f(argc,argv,"--lr",1e-4);
    uint64_t seed=(uint64_t)arg_i(argc,argv,"--seed",0);
    double ram_gb=arg_f(argc,argv,"--ram",52);
    const char *targets=arg_s(argc,argv,"--targets","attention");
    if(strcmp(targets,"attention")){ fprintf(stderr,"v1 supports --targets attention only\n"); return 1; }

    unsetenv("ADAPTER");                        /* the trainer owns the adapter */
    signal(SIGINT,on_int);

#ifdef COLI_METAL
    /* same opt-in as the inference engine: COLI_METAL=1 enables the M6
     * training kernels (train_qt_bwd_dx dispatches large-S calls to t_tmul) */
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))){
        g_metal_enabled = coli_metal_init();
        if(!g_metal_enabled){ fprintf(stderr,"[METAL] backend requested but not available\n"); return 2; }
        fprintf(stderr,"[METAL] training mode: t_tmul dispatch enabled (COLI_TRAIN_METAL_MIN rows)\n");
    }
#else
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL")))
        fprintf(stderr,"[METAL] requested but this coli_train is CPU-only; rebuild with: make coli_train METAL=1\n");
#endif

    Model M;
    model_init(&M,model,64,arg_i(argc,argv,"--ebits",16),arg_i(argc,argv,"--dbits",16));
    Cfg *c=&M.c;
    uint64_t fp=cfg_fingerprint(c);

    /* ---- adapter: resume or fresh (A~N(0,.02), B=0 -> exact base at step 0) */
    LoraAdapter *lora=NULL;
    TCkptState st={.data_seed=seed,.base_fp=fp};
    if(resume){
        lora=lora_load(resume,fp,0);
        if(!lora){ fprintf(stderr,"resume adapter load failed\n"); return 1; }
    } else {
        LoraAdapter fresh={0};
        fresh.n=c->n_layers; fresh.rank=rank; fresh.alpha=alpha; fresh.base_fp=fp;
        snprintf(fresh.base_model,sizeof(fresh.base_model),"glm_moe_dsa");
        fresh.t=calloc(fresh.n,sizeof(LoraTensor));
        lcg^=seed*0x9E3779B97F4A7C15ull;
        for(int li=0;li<fresh.n;li++){
            LoraTensor *t=&fresh.t[li];
            t->layer=li; t->tgt=LORA_T_O; t->rank=rank;
            t->I=M.L[li].o.I; t->O=M.L[li].o.O; t->scale=alpha/rank;
            t->A=falloc((int64_t)rank*t->I); t->B=falloc((int64_t)t->O*rank);
            for(int i=0;i<rank*t->I;i++) t->A[i]=0.02f*nrand();
            memset(t->B,0,(size_t)t->O*rank*4);
        }
        if(lora_save(out,&fresh)){ fprintf(stderr,"cannot write %s\n",out); return 1; }
        lora=lora_load(out,fp,0);
        if(!lora){ fprintf(stderr,"fresh adapter reload failed\n"); return 1; }
    }
    int64_t param_bytes=0;
    for(int i=0;i<lora->n;i++)
        param_bytes+=((int64_t)lora->t[i].rank*lora->t[i].I+(int64_t)lora->t[i].O*lora->t[i].rank)*4;

    /* ---- trainer + budget: expert-slot count derived from the plan ---- */
    TT *tt=tt_init(&M,lora,seq+1,1,/*ecap placeholder*/1);
    TBudget bud={0};
    bud.total=(int64_t)(ram_gb*1073741824.0);
    bud.dense=M.resident_bytes;
    bud.adapter_state=4*param_bytes;            /* params + grads + AdamW m/v */
    bud.train_scratch=tt->bytes_stash + (int64_t)(seq+1)*(int64_t)(seq+1)*c->n_heads*4 + (256ll<<20);
    bud.ckpt_reserve=0;                         /* included in bytes_stash (ckpt mode) */
    if(tbudget_plan(&bud)){ fprintf(stderr,"[budget] refusing to start\n"); return 1; }
    int slots=arg_i(argc,argv,"--expert-slots",0);
    if(slots<=0){
        /* size the cache from the MEASURED footprint, not the planned
         * categories: dense f32 conversions, KV buffers and allocator overhead
         * make the real baseline bigger than the estimate, and an optimistic
         * slot count just meets the soft limit and thrashes (seen on the first
         * real-model run). One expert is loaded to measure its resident cost. */
        int li0=-1; for(int li=0;li<c->n_layers;li++) if(M.L[li].sparse){ li0=li; break; }
        if(li0>=0){
            ttec_get(tt,li0,0);
            int64_t per=tt->ec_bytes>0?tt->ec_bytes:1;
            int64_t base=tb_footprint();
            int64_t avail=(bud.total-bud.os_margin)-base-(2ll<<30);   /* keep 2 GB slack under soft */
            slots=(int)(avail/per);
            int maxs=c->n_layers*c->n_experts;
            if(slots<8){
                fprintf(stderr,"[budget] baseline footprint %.1f GB leaves room for only %d expert "
                               "slots under --ram %.0f — training will be I/O-bound. Raise --ram "
                               "or close other applications.\n",base/1073741824.0,slots,ram_gb);
                if(slots<1) slots=1;
            }
            if(slots>maxs) slots=maxs;
            fprintf(stderr,"[budget] baseline footprint %.2f GB | %.1f MB/expert | %d cache slots\n",
                    base/1073741824.0,per/1048576.0,slots);
        } else slots=1;
    }
    /* re-init trainer with the real slot budget (frees nothing big: tiny structs) */
    tt->ecap=slots; free(tt->ec); tt->ec=calloc(slots,sizeof(TTESlot));
    tt->ec_loads=tt->ec_hits=0; tt->ec_bytes=0; tt->ec_time=0;
    /* footprint soft limit: everything the process may hold; the OS margin
     * stays outside. Crossing it shrinks the expert cache mid-step (§11). */
    tt->mem_soft=bud.total-bud.os_margin;

    TDataset ds;
    if(tds_open(&ds,data,"train",seq,seed)) return 1;
    for(int64_t i=0;i<ds.n_tok;i++) if(ds.tok[i]>=(uint32_t)c->vocab){
        fprintf(stderr,"[data] token id %u at %lld >= model vocab %d — dataset was "
                       "tokenized for a different model\n",ds.tok[i],(long long)i,c->vocab);
        return 1;
    }

    AdamW opt=adamw_default(lr);
    float **sm=calloc(lora->n*2,sizeof(float*)), **sv=calloc(lora->n*2,sizeof(float*));
    float **gA=calloc(lora->n,sizeof(float*)), **gB=calloc(lora->n,sizeof(float*));
    for(int i=0;i<lora->n;i++){
        LoraTensor *t=&lora->t[i];
        int64_t nA=(int64_t)t->rank*t->I, nB=(int64_t)t->O*t->rank;
        sm[2*i]=calloc(nA,4); sv[2*i]=calloc(nA,4);
        sm[2*i+1]=calloc(nB,4); sv[2*i+1]=calloc(nB,4);
        gA[i]=calloc(nA,4); gB[i]=calloc(nB,4);
    }
    int64_t step0=0;
    if(resume){
        if(tckpt_load(resume,lora,&st,sm,sv)){ fprintf(stderr,"resume state load failed\n"); return 1; }
        if(st.base_fp!=fp){ fprintf(stderr,"resume: base fingerprint mismatch\n"); return 1; }
        opt.t=st.opt_t; step0=st.step;
        ds.epoch=st.epoch; ds.cursor=st.cursor; ds.seed=st.data_seed; tds_shuffle(&ds);
    }

    fprintf(stderr,"[train] %s | %d layers rank=%d alpha=%g | seq=%d accum=%d lr=%g | "
                   "expert slots=%d | adapter %.1f MB (state %.1f MB)\n",
            model,lora->n,lora->rank,lora->alpha,seq,accum,lr,slots,
            param_bytes/1048576.0,4.0*param_bytes/1048576.0);
    tbudget_log(&bud,stderr);

    int *tok=malloc((seq+1)*sizeof(int));
    uint32_t *tok32=malloc((seq+1)*4); uint8_t *msk=malloc(seq+1);
    int swap_strikes=0;
    int64_t ceiling=(int64_t)(56ll<<30)<bud.total?(56ll<<30):bud.total+(4ll<<30);
    double t_start=now_s();
    int64_t last_done=step0;   /* last COMPLETED step; the interrupt save must not claim more */

    for(int64_t s=step0; s<step0+steps && !g_train_stop; s++){
        double ts=now_s(), t_fwd=0,t_bwd=0;
        uint64_t loads0=tt->ec_loads, hits0=tt->ec_hits; int64_t bytes0=tt->ec_bytes;
        double loss_sum=0; int64_t tok_sum=0;
        for(int i=0;i<lora->n;i++){
            memset(gA[i],0,(size_t)lora->rank*lora->t[i].I*4);
            memset(gB[i],0,(size_t)lora->t[i].O*lora->rank*4);
        }
        for(int a=0;a<accum;a++){
            tds_next(&ds,tok32,msk);
            for(int i=0;i<seq+1;i++) tok[i]=(int)tok32[i];
            uint64_t pl0=tt->ec_loads, ph0=tt->ec_hits;
            double t0=now_s();
            double lpass=tt_forward_masked(tt,tok,msk); loss_sum+=lpass;
            double dfwd=now_s()-t0; t_fwd+=dfwd; t0=now_s();
            tt_backward_masked(tt,tok,msk);
            double dbwd=now_s()-t0; t_bwd+=dbwd;
            tok_sum+=tt->loss_np;
            if(accum>1)   /* real-model passes are minutes-to-hours: show life between [step] lines */
                fprintf(stderr,"  [micro %d/%d] loss %.4f | fwd %.1fs bwd %.1fs | experts +%llu loads +%llu hits\n",
                        a+1,accum,lpass,dfwd,dbwd,
                        (unsigned long long)(tt->ec_loads-pl0),(unsigned long long)(tt->ec_hits-ph0));
            for(int i=0;i<lora->n;i++){
                int64_t nA=(int64_t)lora->rank*lora->t[i].I, nB=(int64_t)lora->t[i].O*lora->rank;
                for(int64_t j=0;j<nA;j++) gA[i][j]+=tt->dA[i][j];
                for(int64_t j=0;j<nB;j++) gB[i][j]+=tt->dB[i][j];
            }
        }
        double t0=now_s();
        adamw_tick(&opt);
        int bad=0;
        for(int i=0;i<lora->n;i++){
            LoraTensor *t=&lora->t[i];
            int64_t nA=(int64_t)t->rank*t->I, nB=(int64_t)t->O*t->rank;
            for(int64_t j=0;j<nA;j++) gA[i][j]/=accum;
            for(int64_t j=0;j<nB;j++) gB[i][j]/=accum;
            bad|=adamw_step(&opt,t->A,gA[i],sm[2*i],sv[2*i],nA);
            bad|=adamw_step(&opt,t->B,gB[i],sm[2*i+1],sv[2*i+1],nB);
        }
        double t_opt=now_s()-t0, dt=now_s()-ts;
        if(bad){ fprintf(stderr,"[train] NaN/Inf gradient at step %lld — aborting\n",(long long)s+1); break; }
        float loss=(float)(loss_sum/accum);
        fprintf(stderr,"[step %lld] loss %.4f | lr %g | %lld tok %.2f tok/s | "
                       "%.2fs (fwd %.2f bwd %.2f opt %.2f) | experts +%llu loads +%llu hits %.1f MB read | epoch %d\n",
                (long long)s+1,loss,lr,(long long)tok_sum,tok_sum/dt,dt,t_fwd,t_bwd,t_opt,
                (unsigned long long)(tt->ec_loads-loads0),(unsigned long long)(tt->ec_hits-hits0),
                (tt->ec_bytes-bytes0)/1048576.0,ds.epoch);
        last_done=s+1;
        /* any abort below must not lose the step that just completed (§15) */
        #define SAVE_NOW() do{ st.step=s+1; st.opt_t=opt.t; st.epoch=ds.epoch; st.cursor=ds.cursor; \
            if(!tckpt_save(out,lora,&st,sm,sv)) \
                fprintf(stderr,"[ckpt] saved %s @ step %lld (abort path)\n",out,(long long)s+1); }while(0)
        if(!isfinite(loss)){ fprintf(stderr,"[train] non-finite loss — aborting\n"); SAVE_NOW(); break; }
        int viol=tbudget_violated(&bud,ceiling);
        if(viol==1){ tbudget_log(&bud,stderr);
            fprintf(stderr,"[train] footprint over hard ceiling — aborting per AGENTS.md §11\n");
            SAVE_NOW(); break; }
        if(viol==2){
            /* system swap grew: §11 says the expert cache shrinks BEFORE the
             * process leans on swap. Streaming 100s of GB also pressures the
             * macOS file cache, which can page OTHER apps — so give memory
             * back and continue; abort only when the cache has nothing left
             * to give twice in a row. */
            int64_t grew=tb_swap_used()-bud.swap_base;
            int shed=ttec_shed(tt,grew+(1ll<<30));
            bud.swap_base=tb_swap_used();
            if(shed){ swap_strikes=0;
                fprintf(stderr,"[mem] swap grew %+.0f MB — shed %d expert slots and continuing\n",
                        grew/1048576.0,shed); }
            else if(++swap_strikes>=2){ tbudget_log(&bud,stderr);
                fprintf(stderr,"[train] swap keeps growing with an empty expert cache — aborting per §11\n");
                SAVE_NOW(); break; }
            else fprintf(stderr,"[mem] swap grew %+.0f MB with empty cache (strike 1/2) — continuing\n",
                         grew/1048576.0);
        } else swap_strikes=0;
        tbudget_log(&bud,stderr);   /* real-model steps are minutes: log every step */
        if(tt->ec_shrinks) fprintf(stderr,"[mem] cache shrinks so far: %llu slots\n",
                                   (unsigned long long)tt->ec_shrinks);
        if((s+1)%save_every==0 || s+1==step0+steps || g_train_stop){
            st.step=s+1; st.opt_t=opt.t; st.epoch=ds.epoch; st.cursor=ds.cursor;
            if(tckpt_save(out,lora,&st,sm,sv)) fprintf(stderr,"[ckpt] save failed (continuing)\n");
            else fprintf(stderr,"[ckpt] saved %s @ step %lld\n",out,(long long)s+1);
        }
    }
    if(g_train_stop){
        st.step=last_done; st.opt_t=opt.t; st.epoch=ds.epoch; st.cursor=ds.cursor;
        tckpt_save(out,lora,&st,sm,sv);
        fprintf(stderr,"[train] interrupted — checkpoint saved to %s @ step %lld\n",
                out,(long long)last_done);
    }
    fprintf(stderr,"[train] done in %.1fs | total expert I/O: %llu loads %llu hits %.1f MB %.1fs\n",
            now_s()-t_start,(unsigned long long)tt->ec_loads,(unsigned long long)tt->ec_hits,
            tt->ec_bytes/1048576.0,tt->ec_time);
    tbudget_log(&bud,stderr);
    return 0;
}
