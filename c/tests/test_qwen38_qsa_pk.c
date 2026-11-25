/* Q38_QSA_PK: the host QSA's pooled-key cache gives the bits re-pooling gives.
 *
 * Part 1 pools synthetic index keys through q38_qsa_pk_sync and compares every
 * block, bit for bit, with a verbatim copy of the loop q38_attention ran
 * before the cache existed: decode steps, prompt chunks, a rollback the host
 * runs again, a rollback-and-rewrite it does not see, a bulk replacement of
 * the rows, a buffer swap, a rewrite with the same values, and an old block
 * rewritten by a writer that names its first row (q38_qsa_pk_drop_from).
 *
 * Part 2 runs q38_attention itself on a fabricated attention layer (F32
 * weights) with Q38_QSA_PK=0, 1 and 2 side by side: one sequence through a
 * prompt chunk, decode steps, rollbacks and rows rewritten behind the host's
 * back, then a multiplexed step over two conversations. Every output row
 * and every cached K/V/IK row must match the reference bit for bit, and mode
 * 2 must count no differing block. A last case shows the one write the cache
 * cannot see (rows of an old block changed with the last block left alone)
 * does change the output, so the test is able to fail.
 *
 * Part 3 benchmarks long-context decode steps with Q38_QSA_PK=0 vs 1,
 * verifying bit-identical outputs and measuring the host QSA speedup. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#define QWEN38_TEST_SERVE
#include "../qwen38.c"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);return 1; \
} } while(0)

static uint64_t g_rng=0x9E3779B97F4A7C15ull;
static float frand(void){   /* xorshift64*, uniform in [-1, 1) */
    g_rng^=g_rng>>12;g_rng^=g_rng<<25;g_rng^=g_rng>>27;
    return (float)((g_rng*0x2545F4914F6CDD1Dull)>>40)/(float)(1u<<23)-1.f;
}
static void fill(float *x,int64_t n,float scale){for(int64_t i=0;i<n;i++)x[i]=scale*frand();}

/* The pooling exactly as q38_attention wrote it before Q38_QSA_PK (master daaaa99). */
static void ref_pool(float *pool,const float *IKl,int b,int R,int ID,const float *idx_kn,
                     float eps,int rotary_dim,float theta){
    memset(pool,0,(size_t)ID*sizeof(float));for(int r=0;r<R;r++){const float *raw=IKl+(int64_t)(b*R+r)*ID;for(int d=0;d<ID;d++)pool[d]+=raw[d]/R;}
    q38_rms0(pool,pool,idx_kn,ID,eps);q38_rope(pool,ID,rotary_dim,b*R,theta);
}

typedef struct { int R, ID, rotary_dim; float eps, theta; const float *kn; } PoolCfg;

/* every block of e below visible equals the reference re-pooled from IKl */
static int blocks_match(const Q38PkCache *e,const float *IKl,int visible,const PoolCfg *p){
    int need=visible/p->R;
    if(e->len<need)return 0;
    float *ref=falloc(p->ID);
    for(int b=0;b<need;b++){
        ref_pool(ref,IKl,b,p->R,p->ID,p->kn,p->eps,p->rotary_dim,p->theta);
        if(memcmp(ref,e->pk+(int64_t)b*p->ID,(size_t)p->ID*sizeof(float))){
            fprintf(stderr,"block %d of %d differs\n",b,need);free(ref);return 0;
        }
    }
    free(ref);return 1;
}

static void pk_sync(Q38PkCache *e,const float *IKl,int first,int visible,const PoolCfg *p){
    q38_qsa_pk_sync(e,IKl,first,visible,p->R,p->ID,p->kn,p->eps,p->rotary_dim,p->theta);
}

static int part1(int R,int ID,int rotary_dim){
    const int cap=12*R+400;   /* the rows the steps below reach, with room */
    float *kn=falloc(ID);fill(kn,ID,0.3f);
    PoolCfg p={R,ID,rotary_dim,1e-6f,10000000.f,kn};
    float *IK=falloc((int64_t)cap*ID),*IK2=falloc((int64_t)cap*ID);
    Q38PkCache e={0};
    unsigned long long resets;
    int pos=0;
    /* a prompt in chunks of uneven sizes, then decode steps */
    int chunks[]={1,7,R,R+1,3*R-1,64,129};
    for(size_t i=0;i<sizeof chunks/sizeof chunks[0];i++){
        fill(IK+(int64_t)pos*ID,(int64_t)chunks[i]*ID,2.f);
        pk_sync(&e,IK,pos,pos+chunks[i],&p);pos+=chunks[i];
        CHECK(blocks_match(&e,IK,pos,&p));
    }
    for(int t=0;t<3*R+5;t++){
        fill(IK+(int64_t)pos*ID,ID,2.f);pk_sync(&e,IK,pos,pos+1,&p);pos++;
        CHECK(blocks_match(&e,IK,pos,&p));
    }
    /* the host rolls back (a rejected draft) and runs the rows again with new values */
    for(int back=1;back<=2*R+1;back+=R/2+1){
        pos-=back;resets=g_q38_qsa_pk_resets;
        fill(IK+(int64_t)pos*ID,(int64_t)(back+2)*ID,2.f);
        pk_sync(&e,IK,pos,pos+back+2,&p);pos+=back+2;
        CHECK(blocks_match(&e,IK,pos,&p));
        CHECK(g_q38_qsa_pk_resets==resets);   /* the host's own rewrite drops only its blocks, no rebuild */
    }
    /* rolled back and run forward by someone else (the chain's copy-back): rows
     * [x, pos) rewritten without a sync, then the host's next step at pos */
    for(int back=1;back<=3*R;back+=R>1?R-1:1){
        int x=pos-back,P=pos;
        resets=g_q38_qsa_pk_resets;
        fill(IK+(int64_t)x*ID,(int64_t)(P-x)*ID,2.f);
        fill(IK+(int64_t)P*ID,ID,2.f);pk_sync(&e,IK,P,P+1,&p);pos++;
        CHECK(blocks_match(&e,IK,pos,&p));
        /* a rewrite that reached a cached block (all below P/R*R) was noticed; one that did not, was not needed */
        CHECK(g_q38_qsa_pk_resets==resets+(x<P/R*R?1u:0u));
    }
    /* someone else rolls back below the host and runs forward past where it was */
    {
        int x=pos-5*R-2,to=pos+2*R+1;
        resets=g_q38_qsa_pk_resets;
        fill(IK+(int64_t)x*ID,(int64_t)(to-x)*ID,2.f);pos=to;
        fill(IK+(int64_t)pos*ID,ID,2.f);pk_sync(&e,IK,pos,pos+1,&p);pos++;
        CHECK(blocks_match(&e,IK,pos,&p));CHECK(g_q38_qsa_pk_resets==resets+1);
    }
    /* every row replaced in place at the same length (a prefix or park restore) */
    resets=g_q38_qsa_pk_resets;
    fill(IK,(int64_t)pos*ID,2.f);
    fill(IK+(int64_t)pos*ID,ID,2.f);pk_sync(&e,IK,pos,pos+1,&p);pos++;
    CHECK(blocks_match(&e,IK,pos,&p));CHECK(g_q38_qsa_pk_resets==resets+1);
    /* rewritten with the very same values: nothing to rebuild */
    resets=g_q38_qsa_pk_resets;
    memcpy(IK2,IK,(size_t)pos*ID*sizeof(float));memcpy(IK,IK2,(size_t)pos*ID*sizeof(float));
    fill(IK+(int64_t)pos*ID,ID,2.f);pk_sync(&e,IK,pos,pos+1,&p);pos++;
    CHECK(blocks_match(&e,IK,pos,&p));CHECK(g_q38_qsa_pk_resets==resets);
    /* another buffer (a conversation swapped in): starts from it */
    memcpy(IK2,IK,(size_t)pos*ID*sizeof(float));fill(IK2,(int64_t)ID*R,2.f);
    fill(IK2+(int64_t)pos*ID,ID,2.f);pk_sync(&e,IK2,pos,pos+1,&p);
    CHECK(e.ik==IK2&&blocks_match(&e,IK2,pos+1,&p));
    /* invalidated: rebuilt from IK */
    q38_qsa_pk_invalidate(&e,0);
    pk_sync(&e,IK,pos,pos,&p);CHECK(blocks_match(&e,IK,pos,&p));
    /* an old block rewritten, the last one left as it was: the check cannot see
     * it, a writer that names its first row (q38_qsa_pk_drop_from) is exact */
    {
        int x=R/2;
        CHECK(x+R<=(pos/R-1)*R&&pos+1<=cap);
        fill(IK+(int64_t)x*ID,(int64_t)R*ID,2.f);
        q38_qsa_pk_drop_from(&e,0,x,R);CHECK(e.len==x/R);
        fill(IK+(int64_t)pos*ID,ID,2.f);pk_sync(&e,IK,pos,pos+1,&p);pos++;
        CHECK(blocks_match(&e,IK,pos,&p));
        q38_qsa_pk_drop_from(&e,0,pos+5*R,R);CHECK(e.len==pos/R);   /* past the cache: nothing dropped */
        q38_qsa_pk_drop_from(NULL,0,0,R);
    }
    /* a whole long prompt at once (the parallel build) */
    {
        Q38PkCache f={0};fill(IK,(int64_t)cap*ID,2.f);
        pk_sync(&f,IK,0,cap,&p);CHECK(blocks_match(&f,IK,cap,&p));CHECK(f.len==cap/R);
        free(f.pk);
    }
    /* empty and sub-block steps */
    {
        Q38PkCache f={0};
        pk_sync(&f,IK,0,R-1,&p);CHECK(f.len==0);
        pk_sync(&f,IK,R-1,R,&p);CHECK(f.len==1&&blocks_match(&f,IK,R,&p));
        free(f.pk);
    }
    free(e.pk);free(IK);free(IK2);free(kn);
    return 0;
}

/* ---- part 2: q38_attention with and without the cache ---------------------- */
enum { TH=48, TQH=4, TKVH=2, TD=16, TROT=8, TIQ=3, TID=16, TRR=4, TBUDGET=8, TCAP=320 };

static Q38Weight w_f32(int rows,int cols,float scale){
    Q38Weight w;memset(&w,0,sizeof w);
    w.kind=Q38_WEIGHT_F32;w.rows=rows;w.cols=cols;w.elements=(int64_t)rows*cols;
    w.data=falloc(w.elements);fill((float*)w.data,w.elements,scale);
    return w;
}

typedef struct { Model m; Layer L[2]; Q38Seq seq[2]; } Fake;

static void fake_kv(float ***K,float ***V,float ***IK){
    *K=(float**)calloc(2,sizeof(float*));*V=(float**)calloc(2,sizeof(float*));*IK=(float**)calloc(2,sizeof(float*));
    /* zeroed: the rows no step wrote are compared too */
    (*K)[0]=(float*)calloc((size_t)TKVH*TCAP*TD,sizeof(float));(*V)[0]=(float*)calloc((size_t)TKVH*TCAP*TD,sizeof(float));
    (*IK)[0]=(float*)calloc((size_t)TCAP*TID,sizeof(float));
}

/* every run gets the same weights: the generator is reset to the same seed */
static void fake_init(Fake *f){
    memset(f,0,sizeof *f);
    uint64_t saved=g_rng;g_rng=0x1234567887654321ull;
    Cfg *c=&f->m.c;
    c->hidden=TH;c->layers=1;c->q_heads=TQH;c->kv_heads=TKVH;c->head_dim=TD;c->rotary_dim=TROT;
    c->idx_qheads=TIQ;c->idx_kheads=1;c->idx_dim=TID;c->idx_ratio=TRR;c->idx_budget=TBUDGET;
    c->eps=1e-6f;c->theta=10000000.f;c->mtp_theta=c->theta;
    Layer *l=&f->L[0];
    l->q=w_f32(TQH*2*TD,TH,0.2f);l->k=w_f32(TKVH*TD,TH,0.2f);l->v=w_f32(TKVH*TD,TH,0.2f);l->o=w_f32(TH,TQH*TD,0.2f);
    l->idx_qk=w_f32((TIQ+1)*TID,TH,0.3f);
    l->qn=falloc(TD);l->kn=falloc(TD);l->idx_qn=falloc(TID);l->idx_kn=falloc(TID);
    fill(l->qn,TD,0.2f);fill(l->kn,TD,0.2f);fill(l->idx_qn,TID,0.2f);fill(l->idx_kn,TID,0.2f);
    f->m.L=f->L;f->m.kv_cap=TCAP;
    fake_kv(&f->m.K,&f->m.V,&f->m.IK);
    for(int q=0;q<2;q++)fake_kv(&f->seq[q].K,&f->seq[q].V,&f->seq[q].IK);
    g_rng=saved;
}
static void kv_free(float **K,float **V,float **IK){
    if(K){free(K[0]);free(K);} if(V){free(V[0]);free(V);} if(IK){free(IK[0]);free(IK);}
}
static void fake_free(Fake *f){
    Layer *l=&f->L[0];
    free(l->q.data);free(l->k.data);free(l->v.data);free(l->o.data);free(l->idx_qk.data);
    free(l->qn);free(l->kn);free(l->idx_qn);free(l->idx_kn);
    kv_free(f->m.K,f->m.V,f->m.IK);q38_qsa_pk_free(&f->m.qsa_pk,1);
    for(int q=0;q<2;q++){kv_free(f->seq[q].K,f->seq[q].V,f->seq[q].IK);q38_qsa_pk_free(&f->seq[q].qsa_pk,1);}
}

#define NMODE 3
static Fake g_f[NMODE];   /* [0] Q38_QSA_PK=0 (the reference), [1] =1, [2] =2 */

static int same_rows(const float *a,const float *b,int64_t n){return !memcmp(a,b,(size_t)n*sizeof(float));}
static int state_same(float **K0,float **V0,float **IK0,float **K1,float **V1,float **IK1){
    return same_rows(K0[0],K1[0],(int64_t)TKVH*TCAP*TD)&&same_rows(V0[0],V1[0],(int64_t)TKVH*TCAP*TD)&&
           same_rows(IK0[0],IK1[0],(int64_t)TCAP*TID);
}

/* one q38_attention call in every mode on the same input; outputs compared to mode 0 */
static int attend(const float *x,int S,int pos_base,const Q38Row *rows0,const Q38Row *rows1,const Q38Row *rows2){
    float *out[NMODE];
    const Q38Row *rows[NMODE]={rows0,rows1,rows2};
    for(int k=0;k<NMODE;k++){
        g_q38_qsa_pk=k;g_f[k].m.mux_rows=rows[k];
        out[k]=falloc((int64_t)S*TH);
        q38_attention(&g_f[k].m,&g_f[k].L[0],0,x,S,pos_base,out[k]);
        g_f[k].m.mux_rows=NULL;
    }
    int ok=1;
    for(int k=1;k<NMODE;k++){
        if(!same_rows(out[0],out[k],(int64_t)S*TH)){fprintf(stderr,"Q38_QSA_PK=%d: output differs (S=%d pos %d)\n",k,S,pos_base);ok=0;}
        if(!state_same(g_f[0].m.K,g_f[0].m.V,g_f[0].m.IK,g_f[k].m.K,g_f[k].m.V,g_f[k].m.IK)){fprintf(stderr,"Q38_QSA_PK=%d: K/V/IK rows differ\n",k);ok=0;}
        for(int q=0;q<2;q++)if(!state_same(g_f[0].seq[q].K,g_f[0].seq[q].V,g_f[0].seq[q].IK,
                                           g_f[k].seq[q].K,g_f[k].seq[q].V,g_f[k].seq[q].IK))ok=0;
    }
    for(int k=0;k<NMODE;k++)free(out[k]);
    return ok;
}
static int attn_step(int S,int pos_base){
    float *x=falloc((int64_t)S*TH);fill(x,(int64_t)S*TH,1.f);
    int ok=attend(x,S,pos_base,NULL,NULL,NULL);free(x);return ok;
}
/* rows [x, to) of every model's IK rewritten with the same values, no attention call */
static void rewrite_ik(int x,int to){
    float *v=falloc((int64_t)(to-x)*TID);fill(v,(int64_t)(to-x)*TID,1.f);
    for(int k=0;k<NMODE;k++)memcpy(g_f[k].m.IK[0]+(int64_t)x*TID,v,(size_t)(to-x)*TID*sizeof(float));
    free(v);
}

static int part2(void){
    for(int k=0;k<NMODE;k++)fake_init(&g_f[k]);
    g_q38_qsa_pk_mism=0;
    int pos=0;
    /* a prompt in chunks, so the ranking has many blocks to choose TBUDGET/TRR of */
    int chunks[]={37,1,64,9};
    for(size_t i=0;i<sizeof chunks/sizeof chunks[0];i++){CHECK(attn_step(chunks[i],pos));pos+=chunks[i];}
    for(int t=0;t<2*TRR+3;t++){CHECK(attn_step(1,pos));pos++;}
    /* an MTP verify (S rows) rejected after its first row: the host runs from there */
    CHECK(attn_step(4,pos));pos+=1;CHECK(attn_step(3,pos));pos+=3;
    /* a rollback across a block boundary, run again by the host */
    pos-=TRR+2;CHECK(attn_step(1,pos));pos++;CHECK(attn_step(1,pos));pos++;
    /* rows rewritten behind the host's back (the chain ran them), suffix to pos */
    rewrite_ik(pos-2*TRR-1,pos);CHECK(attn_step(1,pos));pos++;
    /* behind its back far below, run past where it was */
    rewrite_ik(pos-6*TRR,pos+TRR+1);pos+=TRR+1;CHECK(attn_step(1,pos));pos++;
    for(int t=0;t<TRR+1;t++){CHECK(attn_step(1,pos));pos++;}
    CHECK(g_q38_qsa_pk_mism==0);
    CHECK(g_f[1].m.qsa_pk&&g_f[1].m.qsa_pk[0].len==pos/TRR);

    /* multiplexed: two parked conversations at their own positions, one row each,
     * then a step with two rows of the first and one of the second */
    int sp[2]={0,0};
    for(int q=0;q<2;q++){
        int n=q?53:71;
        for(int k=0;k<NMODE;k++){
            /* the Model holds the conversation it prefills: swap it in, prefill, swap it out */
            q38_seq_swap(&g_f[k].m,&g_f[k].seq[q]);
        }
        float *x=falloc((int64_t)n*TH);fill(x,(int64_t)n*TH,1.f);
        CHECK(attend(x,n,0,NULL,NULL,NULL));free(x);
        for(int k=0;k<NMODE;k++)q38_seq_swap(&g_f[k].m,&g_f[k].seq[q]);
        sp[q]=n;
    }
    for(int t=0;t<2*TRR+1;t++){
        Q38Row r[NMODE][3];int S=t==TRR?3:2;
        for(int k=0;k<NMODE;k++){
            r[k][0]=(Q38Row){&g_f[k].seq[0],sp[0]};
            r[k][1]=(Q38Row){&g_f[k].seq[1],sp[1]};
            if(S==3)r[k][2]=(Q38Row){&g_f[k].seq[0],sp[0]+1};
        }
        float *x=falloc((int64_t)S*TH);fill(x,(int64_t)S*TH,1.f);
        CHECK(attend(x,S,0,r[0],r[1],r[2]));free(x);
        sp[0]+=S==3?2:1;sp[1]++;
    }
    CHECK(g_q38_qsa_pk_mism==0);
    CHECK(g_f[1].seq[0].qsa_pk&&g_f[1].seq[0].qsa_pk[0].len==sp[0]/TRR);
    CHECK(g_f[1].seq[1].qsa_pk&&g_f[1].seq[1].qsa_pk[0].len==sp[1]/TRR);

    /* The write the check cannot see: rows of an old block changed, the last
     * cached block left as it was. Mode 1 keeps the stale key, mode 2 counts
     * it. This is why every IK writer is a suffix or a whole replacement. */
    {
        unsigned long long mism=g_q38_qsa_pk_mism;
        rewrite_ik(0,(pos/TRR-1)*TRR);   /* every cached block but the last one */
        float *x=falloc(TH);fill(x,TH,1.f);
        float *o0=falloc(TH),*o1=falloc(TH),*o2=falloc(TH);
        g_q38_qsa_pk=0;q38_attention(&g_f[0].m,&g_f[0].L[0],0,x,1,pos,o0);
        g_q38_qsa_pk=1;q38_attention(&g_f[1].m,&g_f[1].L[0],0,x,1,pos,o1);
        g_q38_qsa_pk=2;q38_attention(&g_f[2].m,&g_f[2].L[0],0,x,1,pos,o2);
        CHECK(g_q38_qsa_pk_mism>mism);       /* mode 2 saw it */
        CHECK(same_rows(o0,o2,TH));           /* and ranked from the re-pooled keys */
        CHECK(!same_rows(o0,o1,TH));          /* mode 1 ranked stale keys: the comparison can fail */
        /* an explicit invalidation brings mode 1 back */
        q38_qsa_pk_invalidate(g_f[1].m.qsa_pk,1);
        pos++;
        CHECK(attn_step(1,pos));
        free(x);free(o0);free(o1);free(o2);
    }
    for(int k=0;k<NMODE;k++)fake_free(&g_f[k]);
    return 0;
}

/* ---- part 3: long-context decode benchmark (Q38_QSA_PK=0 vs 1) ------------- */
static int part3_bench(void){
    const int H=64,QH=4,KVH=2,D=64,ROT=32,IQ=4,ID=128,R=16,BUDGET=256;
    const int CTX=8192,STEPS=64,CAP=CTX+STEPS+16;
    Model m[2];Layer L[2];
    memset(m,0,sizeof m);memset(L,0,sizeof L);
    for(int k=0;k<2;k++){
        uint64_t saved=g_rng;g_rng=0xCAFEBABE12345678ull;
        Cfg *c=&m[k].c;
        c->hidden=H;c->layers=1;c->q_heads=QH;c->kv_heads=KVH;c->head_dim=D;c->rotary_dim=ROT;
        c->idx_qheads=IQ;c->idx_kheads=1;c->idx_dim=ID;c->idx_ratio=R;c->idx_budget=BUDGET;
        c->eps=1e-6f;c->theta=10000000.f;c->mtp_theta=c->theta;
        L[k].q=w_f32(QH*2*D,H,0.2f);L[k].k=w_f32(KVH*D,H,0.2f);L[k].v=w_f32(KVH*D,H,0.2f);L[k].o=w_f32(H,QH*D,0.2f);
        L[k].idx_qk=w_f32((IQ+1)*ID,H,0.3f);
        L[k].qn=falloc(D);L[k].kn=falloc(D);L[k].idx_qn=falloc(ID);L[k].idx_kn=falloc(ID);
        fill(L[k].qn,D,0.2f);fill(L[k].kn,D,0.2f);fill(L[k].idx_qn,ID,0.2f);fill(L[k].idx_kn,ID,0.2f);
        m[k].L=&L[k];m[k].kv_cap=CAP;
        m[k].K=(float**)calloc(2,sizeof(float*));m[k].V=(float**)calloc(2,sizeof(float*));m[k].IK=(float**)calloc(2,sizeof(float*));
        m[k].K[0]=falloc((int64_t)KVH*CAP*D);m[k].V[0]=falloc((int64_t)KVH*CAP*D);m[k].IK[0]=falloc((int64_t)CAP*ID);
        fill(m[k].K[0],(int64_t)KVH*CTX*D,0.5f);fill(m[k].V[0],(int64_t)KVH*CTX*D,0.5f);fill(m[k].IK[0],(int64_t)CTX*ID,1.0f);
        g_rng=saved;
    }
    float *xs=falloc((int64_t)STEPS*H);fill(xs,(int64_t)STEPS*H,1.f);
    float *o0=falloc((int64_t)STEPS*H),*o1=falloc((int64_t)STEPS*H);
    /* warm the cache for mode 1 with one step before timing steady-state decode */
    g_q38_qsa_pk=0;q38_attention(&m[0],&L[0],0,xs,1,CTX,o0);
    g_q38_qsa_pk=1;q38_attention(&m[1],&L[1],0,xs,1,CTX,o1);
    CHECK(same_rows(o0,o1,H));
    m[0].timers.seconds[Q38_TM_QSA_INDEX]=0.0;
    m[1].timers.seconds[Q38_TM_QSA_INDEX]=0.0;
    double t0=now_s();
    g_q38_qsa_pk=0;
    for(int s=0;s<STEPS;s++)q38_attention(&m[0],&L[0],0,xs+(int64_t)s*H,1,CTX+1+s,o0+(int64_t)s*H);
    double dt0=now_s()-t0;
    double t1=now_s();
    g_q38_qsa_pk=1;
    for(int s=0;s<STEPS;s++)q38_attention(&m[1],&L[1],0,xs+(int64_t)s*H,1,CTX+1+s,o1+(int64_t)s*H);
    double dt1=now_s()-t1;
    CHECK(same_rows(o0,o1,(int64_t)STEPS*H));
    double idx0=m[0].timers.seconds[Q38_TM_QSA_INDEX],idx1=m[1].timers.seconds[Q38_TM_QSA_INDEX];
    printf("test_qwen38_qsa_pk: bench ctx=%d blocks=%d steps=%d: index %.2fms -> %.2fms (%.2fx), attn %.2fms -> %.2fms (%.2fx)\n",
           CTX,CTX/R,STEPS,idx0*1e3,idx1*1e3,idx1>0?idx0/idx1:0.0,dt0*1e3,dt1*1e3,dt1>0?dt0/dt1:0.0);
    free(xs);free(o0);free(o1);
    for(int k=0;k<2;k++){
        free(L[k].q.data);free(L[k].k.data);free(L[k].v.data);free(L[k].o.data);free(L[k].idx_qk.data);
        free(L[k].qn);free(L[k].kn);free(L[k].idx_qn);free(L[k].idx_kn);
        kv_free(m[k].K,m[k].V,m[k].IK);q38_qsa_pk_free(&m[k].qsa_pk,1);
    }
    return 0;
}

int main(void){
    /* the pooling helper is the old loop */
    {
        float kn[64],IK[8*64],a[64],b[64];fill(kn,64,0.3f);fill(IK,8*64,2.f);
        for(int blk=0;blk<2;blk++){
            ref_pool(a,IK,blk,4,64,kn,1e-6f,32,10000000.f);
            q38_qsa_pool_block(b,IK,blk,4,64,kn,1e-6f,32,10000000.f);
            CHECK(!memcmp(a,b,sizeof a));
        }
    }
    int shapes[][3]={{4,64,32},{8,128,64},{3,16,8},{1,32,32},{128,128,64}};
    for(size_t i=0;i<sizeof shapes/sizeof shapes[0];i++)
        if(part1(shapes[i][0],shapes[i][1],shapes[i][2])){
            fprintf(stderr,"part 1 failed at R=%d ID=%d rotary=%d\n",shapes[i][0],shapes[i][1],shapes[i][2]);return 1;
        }
    if(part2())return 1;
    if(part3_bench())return 1;
    printf("test_qwen38_qsa_pk: ok (rebuilds %llu, blocks built %llu)\n",g_q38_qsa_pk_resets,g_q38_qsa_pk_built);
    return 0;
}
