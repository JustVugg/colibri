/* Milestones 3+4: training forward + manual backward through the GLM
 * transformer, updating ONLY LoRA adapters on self_attn.o_proj (base + router
 * frozen), with optional activation checkpointing.
 *
 * Two memory modes (tt_init ckpt flag):
 *   ckpt=0  M3 semantics — every layer retains its full activation stash.
 *   ckpt=1  M4 semantics — each layer retains ONLY its input x_in [T,D] and
 *           its routing choices [T,K]; every other intermediate lives in ONE
 *           scratch stash shared by all layers. Backward re-runs the layer
 *           forward from the checkpoint (routing REPLAYED from the saved ids,
 *           same kernels, same order -> bit-identical recompute) and then does
 *           the layer backward. Stash memory is O(1 layer), not O(n_layers).
 *
 * Backward per AGENTS.md §8: dW is never allocated for any frozen tensor;
 * every frozen matmul contributes dx = W^T dy only (train_qt_bwd_dx). Routing
 * top-k is a frozen selection; gradients DO flow through the sigmoid gate
 * VALUES and their normalization (PyTorch autograd does the same: gather is
 * differentiable in the gathered values, not the indices).
 *
 * Correctness-first scope for the tiny oracle model: f32 dense QTs (fmt=0,
 * dbits=16), full causal attention (index_topk >> T so DSA selection is the
 * identity — the property the inference oracle relies on), MTP absent.
 * logits [T,V] stay retained in both modes (recomputing the head per position
 * is an M5+ concern; documented, not hidden).
 *
 * Include after glm.c (needs Model/QT/matmul/rmsnorm) + train/qlora_ops.h. */
#ifndef TRAIN_MODEL_H
#define TRAIN_MODEL_H

/* ---------- small backward primitives ---------- */

/* y_i = x_i * r * w_i with r = 1/sqrt(mean(x^2)+eps)  ->  dx (w frozen).
 * Mirrors glm.c rmsnorm: ms accumulated in double, r formed in f32. */
static void tt_rmsnorm_bwd(const float *x, const float *w, const float *dy,
                           float *dx, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps);
    double dot=0; for(int i=0;i<D;i++) dot+=(double)dy[i]*w[i]*x[i];
    float c=(float)(dot)*r*r*r/(float)D;
    for(int i=0;i<D;i++) dx[i]+=r*dy[i]*w[i]-c*x[i];
}

/* rope_interleave maps in[2j],in[2j+1] -> out[j],out[half+j] with a rotation;
 * backward is the transposed (inverse) rotation back to interleaved slots. */
static void tt_rope_bwd(float *dv, int pos, const Cfg *c){
    int half=c->qk_rope/2; float din[256];
    for(int j=0;j<half;j++){
        float inv=powf(c->theta,-2.0f*j/c->qk_rope),ang=pos*inv;
        float cs=cosf(ang),sn=sinf(ang);
        float go=dv[j],gh=dv[half+j];
        din[2*j]  = go*cs+gh*sn;
        din[2*j+1]=-go*sn+gh*cs;
    }
    memcpy(dv,din,c->qk_rope*sizeof(float));
}

/* softmax bwd on one row: dx_i = p_i*(dy_i - sum_j p_j dy_j) */
static void tt_softmax_bwd(const float *p, const float *dy, float *dx, int n){
    double dot=0; for(int i=0;i<n;i++) dot+=(double)p[i]*dy[i];
    for(int i=0;i<n;i++) dx[i]=p[i]*(dy[i]-(float)dot);
}

static inline float tt_dsilu(float x){ float s=1.f/(1.f+expf(-x)); return s*(1.f+x*(1.f-s)); }

/* SwiGLU y = down( silu(g) * u ), g=gate@x, u=up@x — forward stashing g,u.
 * Row-batched: S rows share ONE pass over the weights (matmul_qt streams each
 * weight byte once per call and is OpenMP+SIMD inside). */
static void tt_swiglu_fwd_b(const QT *gate, const QT *up, const QT *down,
                            const float *x, float *g, float *u, float *y, int S){
    int N=gate->O;
    matmul_qt(g,x,(QT*)gate,S);
    matmul_qt(u,x,(QT*)up,S);
    float *h=falloc((int64_t)S*N);
    for(int64_t i=0;i<(int64_t)S*N;i++) h[i]=siluf(g[i])*u[i];
    matmul_qt(y,h,(QT*)down,S);
    free(h);
}
static void tt_swiglu_fwd(const QT *gate, const QT *up, const QT *down,
                          const float *x, float *g, float *u, float *y){
    tt_swiglu_fwd_b(gate,up,down,x,g,u,y,1);
}
/* backward: dx += gate^T dg + up^T du (frozen weights, no dW anywhere) */
static void tt_swiglu_bwd_b(const QT *gate, const QT *up, const QT *down,
                            const float *g, const float *u, const float *dy,
                            float *dx, int S){
    int N=gate->O;
    float *dh=falloc((int64_t)S*N), *dg=falloc((int64_t)S*N), *du=falloc((int64_t)S*N);
    memset(dh,0,(int64_t)S*N*sizeof(float));
    train_qt_bwd_dx(down,dy,dh,S);
    for(int64_t i=0;i<(int64_t)S*N;i++){ du[i]=dh[i]*siluf(g[i]); dg[i]=dh[i]*u[i]*tt_dsilu(g[i]); }
    train_qt_bwd_dx(gate,dg,dx,S);
    train_qt_bwd_dx(up,du,dx,S);
    free(dh); free(dg); free(du);
}
static void tt_swiglu_bwd(const QT *gate, const QT *up, const QT *down,
                          const float *g, const float *u, const float *dy, float *dx){
    tt_swiglu_bwd_b(gate,up,down,g,u,dy,dx,1);
}

/* ---------- per-layer activation stash ---------- */
typedef struct {
    /* CHECKPOINT fields — always private per layer */
    float *x_in;                 /* [T,D] residual entering the layer */
    int   *eidx;                 /* [T,K] routing choices (sparse layers) */
    /* RECOMPUTABLE fields — private (ckpt=0) or aliased to the shared scratch */
    float *nrm1;                 /* [T,D] */
    float *qa,*qan,*q;           /* [T,qlr] [T,qlr] [T,H*qh] (q roped) */
    float *kva;                  /* [T,kvl+R] raw kv_a output (pre-norm/pre-rope) */
    float *ckn,*kr;              /* [T,kvl] normed latent, [T,R] roped k_rope */
    float *kvb;                  /* [T,H*(nope+vh)] */
    float *probs;                /* [H,T,T] */
    float *ctx;                  /* [T,H*vh] */
    float *z;                    /* [T,rank] LoRA cache (o_proj), NULL if unadapted */
    float *x_mid;                /* [T,D] residual entering mlp */
    float *nrm2;                 /* [T,D] */
    float *g,*u;                 /* dense mlp or shared expert pre-activations [T,N] */
    float *ew,*ep;               /* [T,K] final weights, [T,K] sigmoid p */
    float *eg,*eu,*eout;         /* [T,K,mi] [T,K,mi] [T,K,D] */
} TTLayer;

/* streamed-expert cache slot: (layer,eid) -> f32 QTs, LRU-evicted (M5) */
typedef struct { int layer,eid,valid; uint64_t last; QT w[3]; } TTESlot;

typedef struct {
    Model *m; LoraAdapter *lora;
    int T,D,H,qh,vh,nope,kvl,R,E,K,mi;
    int ckpt;                    /* 1 = M4 checkpointed mode */
    TTLayer *L;
    TTLayer scratch;             /* shared recompute stash (ckpt=1) */
    float *x;                    /* [T,D] residual stream (final state after fwd) */
    float *fn;                   /* [T,D] final rmsnorm out */
    float *logits;               /* [T,V] */
    float loss; int loss_np;     /* CE positions in the last forward */
    float **dA,**dB;             /* adapter grads, indexed like lora->t */
    int64_t bytes_stash;         /* retained activation bytes (accounting) */
    /* M5: streamed routed experts — never fully resident, loaded per layer
     * DEDUPLICATED across the micro-batch, cached under a slot budget. */
    TTESlot *ec; int ecap;
    uint64_t ec_clock, ec_loads, ec_hits;    /* honest I/O counters */
    int64_t  ec_bytes;                       /* bytes read from the snapshot */
    double   ec_time;                        /* seconds spent loading */
    int64_t  mem_soft;                       /* footprint soft limit: cache SHRINKS
                                                before the system swaps (§11).
                                                0 = disabled (tests). Enforced only
                                                when budget.h is included first. */
    uint64_t ec_shrinks;                     /* slots dropped under pressure */
    int      ec_cooldown;                    /* misses to skip shrink checks after
                                                an exhausted-cache warning */
} TT;

static void ttec_drop(TTESlot *s){
    for(int k=0;k<3;k++){
        free(s->w[k].qf); free(s->w[k].q8); free(s->w[k].q4); free(s->w[k].s);
        memset(&s->w[k],0,sizeof(QT));
    }
    s->valid=0;
}

/* shed ~`bytes` of expert cache (LRU-first). Returns slots dropped; 0 means
 * the cache was already empty — the caller has nothing elastic left. */
static int ttec_shed(TT *tt, int64_t bytes){
    int dropped=0; int64_t freed=0;
    int64_t per = tt->ec_loads ? tt->ec_bytes/(int64_t)tt->ec_loads : (16ll<<20);
    while(freed<bytes){
        TTESlot *old=NULL;
        for(int i=0;i<tt->ecap;i++){ TTESlot *s=&tt->ec[i];
            if(s->valid && (!old||s->last<old->last)) old=s; }
        if(!old) break;
        ttec_drop(old); dropped++; freed+=per; tt->ec_shrinks++;
    }
    return dropped;
}

/* fetch expert weights, loading from the snapshot on miss (drop=1: streaming —
 * the page cache is told not to keep the data; the slot cache is the budget). */
static QT *ttec_get(TT *tt, int layer, int eid){
    TTESlot *lru=&tt->ec[0];
    for(int i=0;i<tt->ecap;i++){
        TTESlot *s=&tt->ec[i];
        if(s->valid&&s->layer==layer&&s->eid==eid){
            s->last=++tt->ec_clock; tt->ec_hits++; return s->w; }
        if(!s->valid){ if(lru->valid) lru=s; }
        else if(lru->valid&&s->last<lru->last) lru=s;
    }
#ifdef TRAIN_BUDGET_H
    /* miss under memory pressure: §11 — the expert cache is the ONLY elastic
     * category, so it gives memory back BEFORE macOS starts swapping. Checked
     * here (inside the step) because one real-model step is long enough for
     * the cache to balloon between the per-step budget checks.
     * Hysteresis: shrink to soft-1GB, not to soft — dropping exactly to the
     * line makes the very next miss shrink again (measured thrash: every load
     * evicted the whole cache, hit rate ~0). If the cache is empty and the
     * footprint is STILL over the line, the base+scratch exceed the plan:
     * nothing left to give back — warn once and cool down instead of looping. */
    if(tt->mem_soft>0 && tt->ec_cooldown<=0 && tb_footprint()>tt->mem_soft){
        int dropped=0; int64_t tgt=tt->mem_soft-(1ll<<30);
        for(;;){
            if(tb_footprint()<=tgt) break;
            TTESlot *old=NULL;
            for(int i=0;i<tt->ecap;i++){ TTESlot *s2=&tt->ec[i];
                if(s2->valid && (!old||s2->last<old->last)) old=s2; }
            if(!old) break;
            ttec_drop(old); dropped++; tt->ec_shrinks++;
        }
        if(tb_footprint()>tt->mem_soft){
            tt->ec_cooldown=2048;               /* ~a layer pass of quiet */
            fprintf(stderr,"[mem] cache empty but footprint %.1f GB > soft %.1f GB — "
                           "base+scratch exceed the plan; expert caching is effectively "
                           "disabled. Lower other categories or raise --ram.\n",
                    tb_footprint()/1073741824.0,tt->mem_soft/1073741824.0);
        } else if(dropped)
            fprintf(stderr,"[mem] expert cache shrunk by %d slots (footprint %.1f GB, target %.1f GB)\n",
                    dropped,tb_footprint()/1073741824.0,tgt/1073741824.0);
    }
    if(tt->ec_cooldown>0) tt->ec_cooldown--;
#endif
    double t0=now_s();
    static const char *suf[3]={"gate_proj","up_proj","down_proj"};
    int Os[3]={tt->mi,tt->mi,tt->D}, Is[3]={tt->D,tt->D,tt->mi};
    char nm[256];
    for(int k=0;k<3;k++){
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
        qt_from_disk(tt->m,nm,Os[k],Is[k],16,1,&lru->w[k]);
        tt->ec_bytes+=st_nbytes(&tt->m->S,nm);
    }
    lru->valid=1; lru->layer=layer; lru->eid=eid; lru->last=++tt->ec_clock;
    tt->ec_loads++; tt->ec_time+=now_s()-t0;
    return lru->w;
}

/* allocate the recomputable fields of one stash; sized for the WORST layer so a
 * single scratch serves them all. Returns bytes allocated. */
static int64_t tt_alloc_recompute(TT *tt, TTLayer *s, int any_sparse, int any_lora){
    Cfg *c=&tt->m->c;
    int T=tt->T,D=tt->D,H=tt->H,qh=tt->qh,vh=tt->vh,kvl=tt->kvl,R=tt->R,K=tt->K,mi=tt->mi;
    int N=c->dense_inter; if(any_sparse && mi*c->n_shared>N) N=mi*c->n_shared;
    int64_t b=0;
    #define AL(f,n) do{ s->f=falloc(n); b+=(int64_t)(n)*4; }while(0)
    AL(nrm1,(int64_t)T*D); AL(qa,(int64_t)T*c->q_lora); AL(qan,(int64_t)T*c->q_lora);
    AL(q,(int64_t)T*H*qh); AL(kva,(int64_t)T*(kvl+R)); AL(ckn,(int64_t)T*kvl);
    AL(kr,(int64_t)T*R);   AL(kvb,(int64_t)T*H*(tt->nope+vh));
    AL(probs,(int64_t)H*T*T); AL(ctx,(int64_t)T*H*vh);
    AL(x_mid,(int64_t)T*D); AL(nrm2,(int64_t)T*D);
    AL(g,(int64_t)T*N); AL(u,(int64_t)T*N);
    if(any_sparse){
        AL(ew,(int64_t)T*K); AL(ep,(int64_t)T*K);
        AL(eg,(int64_t)T*K*mi); AL(eu,(int64_t)T*K*mi); AL(eout,(int64_t)T*K*D);
    }
    if(any_lora) AL(z,(int64_t)T*tt->lora->rank);
    #undef AL
    return b;
}

static TT *tt_init(Model *m, LoraAdapter *lora, int T, int ckpt, int ecap){
    TT *tt=calloc(1,sizeof(TT));
    Cfg *c=&m->c;
    /* fixed-buffer contracts (lg/ch[4096], dwl/dp[64], rope[256]): fail loudly
     * instead of smashing the stack on an architecture that violates them */
    if(c->n_experts>4096||c->topk>64||c->qk_rope>256){
        fprintf(stderr,"tt_init: dims exceed trainer buffer contracts (E=%d topk=%d rope=%d)\n",
                c->n_experts,c->topk,c->qk_rope);
        exit(1);
    }
    tt->m=m; tt->lora=lora; tt->T=T; tt->D=c->hidden; tt->H=c->n_heads;
    tt->qh=c->qk_head; tt->vh=c->v_head; tt->nope=c->qk_nope; tt->kvl=c->kv_lora;
    tt->R=c->qk_rope; tt->E=c->n_experts; tt->K=c->topk; tt->mi=c->moe_inter;
    tt->ckpt=ckpt;
    tt->L=calloc(c->n_layers,sizeof(TTLayer));
    int D=tt->D,K=tt->K;
    int any_sparse=0; for(int li=0;li<c->n_layers;li++) any_sparse|=m->L[li].sparse;
    if(ckpt) tt->bytes_stash+=tt_alloc_recompute(tt,&tt->scratch,any_sparse,lora->n>0);
    for(int li=0;li<c->n_layers;li++){
        TTLayer *s=&tt->L[li];
        s->x_in=falloc((int64_t)T*D); tt->bytes_stash+=(int64_t)T*D*4;
        if(m->L[li].sparse){ s->eidx=calloc((size_t)T*K,sizeof(int)); tt->bytes_stash+=(int64_t)T*K*4; }
        if(ckpt){
            /* recomputable fields alias the shared scratch (x_in/eidx stay private) */
            float *xi=s->x_in; int *ei=s->eidx;
            *s=tt->scratch; s->x_in=xi; s->eidx=ei;
            if(!lora_find(lora,li,LORA_T_O)) s->z=NULL;
        } else {
            tt->bytes_stash+=tt_alloc_recompute(tt,s,m->L[li].sparse,0);
            if(m->L[li].sparse){ /* keep */ } else { s->ew=s->ep=s->eg=s->eu=s->eout=NULL; }
            if(lora_find(lora,li,LORA_T_O)){ s->z=falloc((int64_t)T*lora->rank); tt->bytes_stash+=(int64_t)T*lora->rank*4; }
            else s->z=NULL;
        }
    }
    tt->x=falloc((int64_t)T*D); tt->fn=falloc((int64_t)T*D);
    tt->logits=falloc((int64_t)T*c->vocab);
    tt->dA=calloc(lora->n,sizeof(float*)); tt->dB=calloc(lora->n,sizeof(float*));
    for(int i=0;i<lora->n;i++){
        tt->dA[i]=falloc((int64_t)lora->rank*lora->t[i].I);
        tt->dB[i]=falloc((int64_t)lora->t[i].O*lora->rank);
    }
    if(ecap<1) ecap=1;
    tt->ecap=ecap; tt->ec=calloc(ecap,sizeof(TTESlot));
    return tt;
}

/* ---------- one layer forward from s->x_in ----------
 * Fills the stash; if x_out != NULL also writes the layer output stream
 * (x_in + attn + mlp) — x_out may alias the buffer x_in was copied from.
 * replay=1 (checkpointed backward): routing choices are NOT re-selected, they
 * are replayed from s->eidx; gate values are recomputed deterministically. */
static void tt_layer_forward(TT *tt, int li, TTLayer *s, float *x_out, int replay){
    Model *m=tt->m; Cfg *c=&m->c; Layer *l=&m->L[li];
    int T=tt->T,D=tt->D,H=tt->H,qh=tt->qh,vh=tt->vh,nope=tt->nope,kvl=tt->kvl,R=tt->R,K=tt->K,mi=tt->mi;
    for(int t=0;t<T;t++) rmsnorm(s->nrm1+(int64_t)t*D,s->x_in+(int64_t)t*D,l->in_ln,D,c->eps);
    for(int t=0;t<T;t++){
        const float *n1=s->nrm1+(int64_t)t*D;
        matmul_qt(s->qa+(int64_t)t*c->q_lora,n1,&l->q_a,1);
        rmsnorm(s->qan+(int64_t)t*c->q_lora,s->qa+(int64_t)t*c->q_lora,l->q_a_ln,c->q_lora,c->eps);
        matmul_qt(s->q+(int64_t)t*H*qh,s->qan+(int64_t)t*c->q_lora,&l->q_b,1);
        for(int h=0;h<H;h++) rope_interleave(s->q+(int64_t)t*H*qh+(int64_t)h*qh+nope,t,c);
        matmul_qt(s->kva+(int64_t)t*(kvl+R),n1,&l->kv_a,1);
        rmsnorm(s->ckn+(int64_t)t*kvl,s->kva+(int64_t)t*(kvl+R),l->kv_a_ln,kvl,c->eps);
        memcpy(s->kr+(int64_t)t*R,s->kva+(int64_t)t*(kvl+R)+kvl,R*sizeof(float));
        rope_interleave(s->kr+(int64_t)t*R,t,c);
        matmul_qt(s->kvb+(int64_t)t*H*(nope+vh),s->ckn+(int64_t)t*kvl,&l->kv_b,1);
    }
    /* causal attention, full materialization */
    for(int h=0;h<H;h++) for(int t=0;t<T;t++){
        const float *qp=s->q+(int64_t)t*H*qh+(int64_t)h*qh, *qr=qp+nope;
        float *p=s->probs+((int64_t)h*T+t)*T;
        for(int t2=0;t2<=t;t2++){
            const float *kn=s->kvb+(int64_t)t2*H*(nope+vh)+(int64_t)h*(nope+vh);
            const float *kr=s->kr+(int64_t)t2*R;
            float a=0; for(int d=0;d<nope;d++) a+=qp[d]*kn[d];
            for(int d=0;d<R;d++) a+=qr[d]*kr[d];
            p[t2]=a*c->attn_scale;
        }
        softmax(p,t+1);
        for(int t2=t+1;t2<T;t2++) p[t2]=0;
        float *cx=s->ctx+(int64_t)t*H*vh+(int64_t)h*vh;
        for(int d=0;d<vh;d++) cx[d]=0;
        for(int t2=0;t2<=t;t2++){
            const float *v=s->kvb+(int64_t)t2*H*(nope+vh)+(int64_t)h*(nope+vh)+nope;
            float a=p[t2]; for(int d=0;d<vh;d++) cx[d]+=a*v[d];
        }
    }
    /* o_proj + LoRA residual -> x_mid = x_in + attn_out */
    {
        const LoraTensor *lt=lora_find(tt->lora,li,LORA_T_O);
        matmul_qt(s->x_mid,s->ctx,&l->o,T);
        if(lt) train_lora_fwd(lt,s->ctx,s->x_mid,s->z,T);
        for(int64_t i=0;i<(int64_t)T*D;i++) s->x_mid[i]+=s->x_in[i];
    }
    for(int t=0;t<T;t++) rmsnorm(s->nrm2+(int64_t)t*D,s->x_mid+(int64_t)t*D,l->post_ln,D,c->eps);
    if(x_out) memcpy(x_out,s->x_mid,(int64_t)T*D*sizeof(float));
    /* mlp: dense / shared-expert SwiGLU batched over all T rows (one weight
     * stream per matmul instead of one per token) */
    {
        float *Y=falloc((int64_t)T*D);
        if(!l->sparse)
            tt_swiglu_fwd_b(&l->gate_proj,&l->up_proj,&l->down_proj,s->nrm2,s->g,s->u,Y,T);
        else
            tt_swiglu_fwd_b(&l->sh_gate,&l->sh_up,&l->sh_down,s->nrm2,s->g,s->u,Y,T);
        if(x_out) for(int64_t i=0;i<(int64_t)T*D;i++) x_out[i]+=Y[i];
        free(Y);
    }
    if(l->sparse) for(int t=0;t<T;t++){
        const float *n2=s->nrm2+(int64_t)t*D;
        /* router: sigmoid probs; selection top-K by p+bias — or replay */
        float lg[4096];
        matmul(lg,n2,l->router,1,D,tt->E);
        for(int e=0;e<tt->E;e++) lg[e]=sigmoidf(lg[e]);
        int *idx=s->eidx+(int64_t)t*K; float *w=s->ew+(int64_t)t*K, *pp=s->ep+(int64_t)t*K;
        if(!replay){
            float ch[4096];
            for(int e=0;e<tt->E;e++) ch[e]=lg[e]+l->router_bias[e];
            for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
                for(int e=0;e<tt->E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                    if(!tk&&ch[e]>bv){bv=ch[e];best=e;} }
                idx[kk]=best;
            }
        }
        for(int kk=0;kk<K;kk++) pp[kk]=lg[idx[kk]];
        float sm=0; for(int kk=0;kk<K;kk++) sm+=pp[kk];
        for(int kk=0;kk<K;kk++)
            w[kk]=(c->norm_topk ? pp[kk]/(sm+1e-20f) : pp[kk])*c->routed_scale;
    }
    /* routed experts, GROUPED BY EXPERT across the whole micro-batch: each
     * needed expert is fetched ONCE per layer pass, its routed rows are
     * GATHERED and run as one S=nr batch — one weight stream serves every
     * row (AGENTS.md §10 — never load per token). */
    if(l->sparse){
        int *rt=malloc(sizeof(int)*2*T);
        float *bx=falloc((int64_t)T*D), *bg=falloc((int64_t)T*mi);
        float *bu=falloc((int64_t)T*mi), *by=falloc((int64_t)T*D);
        for(int e=0;e<tt->E;e++){
            int nr=0;
            for(int t=0;t<T;t++){ const int *idx=s->eidx+(int64_t)t*K;
                for(int kk=0;kk<K;kk++) if(idx[kk]==e){ rt[2*nr]=t; rt[2*nr+1]=kk; nr++; } }
            if(!nr) continue;
            QT *ew=ttec_get(tt,li,e);
            for(int r=0;r<nr;r++)
                memcpy(bx+(int64_t)r*D, s->nrm2+(int64_t)rt[2*r]*D, D*sizeof(float));
            tt_swiglu_fwd_b(&ew[0],&ew[1],&ew[2],bx,bg,bu,by,nr);
            for(int r=0;r<nr;r++){
                int t=rt[2*r], kk=rt[2*r+1];
                memcpy(s->eg+((int64_t)t*K+kk)*mi, bg+(int64_t)r*mi, mi*sizeof(float));
                memcpy(s->eu+((int64_t)t*K+kk)*mi, bu+(int64_t)r*mi, mi*sizeof(float));
                memcpy(s->eout+((int64_t)t*K+kk)*D, by+(int64_t)r*D, D*sizeof(float));
                if(x_out){ float wk=s->ew[(int64_t)t*K+kk];
                    for(int d=0;d<D;d++) x_out[(int64_t)t*D+d]+=wk*by[(int64_t)r*D+d]; }
            }
        }
        free(rt); free(bx); free(bg); free(bu); free(by);
    }
}

/* ---------- one layer backward (stash must be valid for this layer) ----------
 * dx enters as dL/d(layer output) [T,D] and leaves as dL/d(layer input). */
static void tt_layer_backward(TT *tt, int li, TTLayer *s, float *dx){
    Model *m=tt->m; Cfg *c=&m->c; Layer *l=&m->L[li];
    int T=tt->T,D=tt->D,H=tt->H,qh=tt->qh,vh=tt->vh,nope=tt->nope,kvl=tt->kvl,R=tt->R,K=tt->K,mi=tt->mi;
    /* ---- mlp backward: dx currently holds dL/d(x after mlp add) ---- */
    float *dn2=falloc((int64_t)T*D); memset(dn2,0,(int64_t)T*D*sizeof(float));
    /* dense / shared-expert SwiGLU backward batched over all T rows */
    if(!l->sparse)
        tt_swiglu_bwd_b(&l->gate_proj,&l->up_proj,&l->down_proj,s->g,s->u,dx,dn2,T);
    else
        tt_swiglu_bwd_b(&l->sh_gate,&l->sh_up,&l->sh_down,s->g,s->u,dx,dn2,T);
    if(l->sparse) for(int t=0;t<T;t++){
        const float *dy=dx+(int64_t)t*D;
        {
            /* dL/dw_k needs only the STASHED expert outputs — no expert weights */
            const int *idx=s->eidx+(int64_t)t*K; const float *pp=s->ep+(int64_t)t*K;
            float dwl[64];
            for(int kk=0;kk<K;kk++){
                const float *eo=s->eout+((int64_t)t*K+kk)*D;
                double a=0; for(int d=0;d<D;d++) a+=(double)eo[d]*dy[d];
                dwl[kk]=(float)a;
            }
            /* through gate values: w_k = scale * p_k / sum(p) (norm_topk) */
            float dp[64];
            if(c->norm_topk){
                float sm=0; for(int kk=0;kk<K;kk++) sm+=pp[kk]; sm+=1e-20f;
                double wd=0; for(int kk=0;kk<K;kk++) wd+=(double)dwl[kk]*pp[kk];
                for(int kk=0;kk<K;kk++)
                    dp[kk]=c->routed_scale*(dwl[kk]/sm-(float)(wd/((double)sm*sm)));
            } else for(int kk=0;kk<K;kk++) dp[kk]=c->routed_scale*dwl[kk];
            for(int kk=0;kk<K;kk++){
                float dl=dp[kk]*pp[kk]*(1.f-pp[kk]);  /* sigmoid' */
                const float *rr=l->router+(int64_t)idx[kk]*D;
                for(int d=0;d<D;d++) dn2[(int64_t)t*D+d]+=dl*rr[d];
            }
        }
    }
    /* routed expert backward, GROUPED BY EXPERT (one fetch per needed expert
     * per layer pass — the backward reload is counted by the same I/O metrics,
     * never hidden). Routed rows are GATHERED into one S=nr batch so the three
     * weight streams serve every row. Frozen experts receive dX only, no dW
     * exists anywhere. */
    if(l->sparse){
        int *rt=malloc(sizeof(int)*2*T);
        float *bdy=falloc((int64_t)T*D), *bg=falloc((int64_t)T*mi);
        float *bu=falloc((int64_t)T*mi), *bdx=falloc((int64_t)T*D);
        for(int e=0;e<tt->E;e++){
            int nr=0;
            for(int t=0;t<T;t++){ const int *idx=s->eidx+(int64_t)t*K;
                for(int kk=0;kk<K;kk++) if(idx[kk]==e){ rt[2*nr]=t; rt[2*nr+1]=kk; nr++; } }
            if(!nr) continue;
            QT *ew=ttec_get(tt,li,e);
            for(int r=0;r<nr;r++){
                int t=rt[2*r], kk=rt[2*r+1]; float wk=s->ew[(int64_t)t*K+kk];
                const float *dy=dx+(int64_t)t*D;
                for(int d=0;d<D;d++) bdy[(int64_t)r*D+d]=wk*dy[d];
                memcpy(bg+(int64_t)r*mi, s->eg+((int64_t)t*K+kk)*mi, mi*sizeof(float));
                memcpy(bu+(int64_t)r*mi, s->eu+((int64_t)t*K+kk)*mi, mi*sizeof(float));
            }
            memset(bdx,0,(int64_t)nr*D*sizeof(float));
            tt_swiglu_bwd_b(&ew[0],&ew[1],&ew[2],bg,bu,bdy,bdx,nr);
            for(int r=0;r<nr;r++){
                float *dn=dn2+(int64_t)rt[2*r]*D;
                for(int d=0;d<D;d++) dn[d]+=bdx[(int64_t)r*D+d];
            }
        }
        free(rt); free(bdy); free(bg); free(bu); free(bdx);
    }
    /* post_ln backward; residual passthrough keeps dx as-is */
    for(int t=0;t<T;t++)
        tt_rmsnorm_bwd(s->x_mid+(int64_t)t*D,l->post_ln,dn2+(int64_t)t*D,dx+(int64_t)t*D,D,c->eps);
    free(dn2);
    /* ---- attention backward: dx holds dL/d(x after attn add) ---- */
    float *dctx=falloc((int64_t)T*H*vh); memset(dctx,0,(int64_t)T*H*vh*sizeof(float));
    const LoraTensor *lt=lora_find(tt->lora,li,LORA_T_O);
    train_qt_bwd_dx(&l->o,dx,dctx,T);
    if(lt){
        int ai=-1; for(int i=0;i<tt->lora->n;i++) if(&tt->lora->t[i]==lt){ ai=i; break; }
        train_lora_bwd(lt,s->ctx,s->z,dx,tt->dA[ai],tt->dB[ai],dctx,T);
    }
    float *dq=falloc((int64_t)T*H*qh);       memset(dq,0,(int64_t)T*H*qh*sizeof(float));
    float *dkvb=falloc((int64_t)T*H*(nope+vh)); memset(dkvb,0,(int64_t)T*H*(nope+vh)*sizeof(float));
    float *dkr=falloc((int64_t)T*R);         memset(dkr,0,(int64_t)T*R*sizeof(float));
    float *dp=falloc(T), *ds=falloc(T);
    for(int h=0;h<H;h++) for(int t=0;t<T;t++){
        const float *p=s->probs+((int64_t)h*T+t)*T;
        const float *dcx=dctx+(int64_t)t*H*vh+(int64_t)h*vh;
        for(int t2=0;t2<=t;t2++){
            const float *v=s->kvb+(int64_t)t2*H*(nope+vh)+(int64_t)h*(nope+vh)+nope;
            float a=0; for(int d=0;d<vh;d++) a+=dcx[d]*v[d];
            dp[t2]=a;
            float *dv=dkvb+(int64_t)t2*H*(nope+vh)+(int64_t)h*(nope+vh)+nope;
            for(int d=0;d<vh;d++) dv[d]+=p[t2]*dcx[d];
        }
        tt_softmax_bwd(p,dp,ds,t+1);
        const float *qp=s->q+(int64_t)t*H*qh+(int64_t)h*qh, *qr=qp+nope;
        float *dqp=dq+(int64_t)t*H*qh+(int64_t)h*qh, *dqr=dqp+nope;
        for(int t2=0;t2<=t;t2++){
            float a=ds[t2]*c->attn_scale;
            const float *kn=s->kvb+(int64_t)t2*H*(nope+vh)+(int64_t)h*(nope+vh);
            const float *kr=s->kr+(int64_t)t2*R;
            float *dkn=dkvb+(int64_t)t2*H*(nope+vh)+(int64_t)h*(nope+vh);
            for(int d=0;d<nope;d++){ dqp[d]+=a*kn[d]; dkn[d]+=a*qp[d]; }
            for(int d=0;d<R;d++){ dqr[d]+=a*kr[d]; dkr[(int64_t)t2*R+d]+=a*qr[d]; }
        }
    }
    free(dp); free(ds);
    /* projections backward into dnrm1, then rmsnorm into dx (residual keeps dx) */
    float *dn1=falloc((int64_t)T*D); memset(dn1,0,(int64_t)T*D*sizeof(float));
    float *dckn=falloc(kvl), *dkva=falloc(kvl+R), *dqan=falloc(c->q_lora), *dqa=falloc(c->q_lora);
    for(int t=0;t<T;t++){
        /* kv path */
        memset(dckn,0,kvl*sizeof(float));
        train_qt_bwd_dx(&l->kv_b,dkvb+(int64_t)t*H*(nope+vh),dckn,1);
        memset(dkva,0,(kvl+R)*sizeof(float));
        tt_rmsnorm_bwd(s->kva+(int64_t)t*(kvl+R),l->kv_a_ln,dckn,dkva,kvl,c->eps);
        float drr[256]; memcpy(drr,dkr+(int64_t)t*R,R*sizeof(float));
        tt_rope_bwd(drr,t,c);
        for(int d=0;d<R;d++) dkva[kvl+d]+=drr[d];
        train_qt_bwd_dx(&l->kv_a,dkva,dn1+(int64_t)t*D,1);
        /* q path: un-rope grad per head, then q_b^T, q_a_ln, q_a^T */
        float *dqt=dq+(int64_t)t*H*qh;
        for(int h=0;h<H;h++) tt_rope_bwd(dqt+(int64_t)h*qh+nope,t,c);
        memset(dqan,0,c->q_lora*sizeof(float));
        train_qt_bwd_dx(&l->q_b,dqt,dqan,1);
        memset(dqa,0,c->q_lora*sizeof(float));
        tt_rmsnorm_bwd(s->qa+(int64_t)t*c->q_lora,l->q_a_ln,dqan,dqa,c->q_lora,c->eps);
        train_qt_bwd_dx(&l->q_a,dqa,dn1+(int64_t)t*D,1);
        tt_rmsnorm_bwd(s->x_in+(int64_t)t*D,l->in_ln,dn1+(int64_t)t*D,dx+(int64_t)t*D,D,c->eps);
    }
    free(dckn); free(dkva); free(dqan); free(dqa);
    free(dn1); free(dq); free(dkvb); free(dkr); free(dctx);
}

/* ---------- full forward ----------
 * msk: per-token loss mask (coli-sft-v1 semantics — position t contributes
 * iff msk[t+1]==1), NULL = every next-token position counts. */
static float tt_forward_masked(TT *tt, const int *tok, const unsigned char *msk){
    Model *m=tt->m; Cfg *c=&m->c;
    int T=tt->T,D=tt->D;
    for(int t=0;t<T;t++) embed_row(m,tok[t],tt->x+(int64_t)t*D);
    for(int li=0;li<c->n_layers;li++){
        TTLayer *s=&tt->L[li];
        memcpy(s->x_in,tt->x,(int64_t)T*D*sizeof(float));
        tt_layer_forward(tt,li,s,tt->x,0);
    }
    /* head + mean CE over unmasked positions t predicting tok[t+1] */
    double lsum=0; int V=c->vocab, np=0;
    for(int t=0;t<T;t++){
        rmsnorm(tt->fn+(int64_t)t*D,tt->x+(int64_t)t*D,m->final_norm,D,c->eps);
        matmul_qt(tt->logits+(int64_t)t*V,tt->fn+(int64_t)t*D,&m->lm_head,1);
    }
    for(int t=0;t<T-1;t++){
        if(msk && !msk[t+1]) continue;
        const float *lo=tt->logits+(int64_t)t*V;
        double mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
        double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
        lsum += (mx+log(se)) - (double)lo[tok[t+1]];
        np++;
    }
    tt->loss_np=np;
    tt->loss = np ? (float)(lsum/np) : 0.f;
    return tt->loss;
}
static float tt_forward(TT *tt, const int *tok){ return tt_forward_masked(tt,tok,NULL); }

/* ---------- full backward ----------
 * ckpt=0: every layer's stash is still valid from tt_forward.
 * ckpt=1: only x_in + routing ids survived; recompute the layer stash
 * (routing replayed) right before its backward — memory stays O(1 layer). */
static void tt_backward_masked(TT *tt, const int *tok, const unsigned char *msk){
    Model *m=tt->m; Cfg *c=&m->c;
    int T=tt->T,D=tt->D,V=c->vocab,np=tt->loss_np;
    if(!np) return;
    for(int i=0;i<tt->lora->n;i++){
        memset(tt->dA[i],0,(size_t)tt->lora->rank*tt->lora->t[i].I*sizeof(float));
        memset(tt->dB[i],0,(size_t)tt->lora->t[i].O*tt->lora->rank*sizeof(float));
    }
    float *dx=falloc((int64_t)T*D); memset(dx,0,(int64_t)T*D*sizeof(float));
    /* CE + lm_head + final norm (np = unmasked positions from the forward) */
    {
        float *dlog=falloc(V), *dfn=falloc(D);
        for(int t=0;t<T-1;t++){
            if(msk && !msk[t+1]) continue;
            const float *lo=tt->logits+(int64_t)t*V;
            double mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
            double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
            for(int i=0;i<V;i++) dlog[i]=(float)(exp((double)lo[i]-mx)/se/np);
            dlog[tok[t+1]]-=1.f/np;
            memset(dfn,0,D*sizeof(float));
            train_qt_bwd_dx(&m->lm_head,dlog,dfn,1);
            tt_rmsnorm_bwd(tt->x+(int64_t)t*D,m->final_norm,dfn,dx+(int64_t)t*D,D,c->eps);
        }
        free(dlog); free(dfn);
    }
    for(int li=c->n_layers-1;li>=0;li--){
        TTLayer *s=&tt->L[li];
        if(tt->ckpt) tt_layer_forward(tt,li,s,NULL,1);   /* recompute from checkpoint */
        tt_layer_backward(tt,li,s,dx);
    }
    free(dx);
}
static void tt_backward(TT *tt, const int *tok){ tt_backward_masked(tt,tok,NULL); }

#endif /* TRAIN_MODEL_H */
