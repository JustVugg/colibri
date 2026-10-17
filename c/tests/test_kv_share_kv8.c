/* #2096 gate: COLI_KV_SHARE cross-slot KV adoption must copy the quantized
 * cache — not just the f32 rows — when the engine runs with KV8=1 or KV_TQ=1.
 *
 * Two KVStates play the two slots. The DONOR is "fully prefilled": its rows
 * [0,blen) are quantized natively from deterministic f32 reference rows, the
 * way a real prefill writes them. The RECEIVER holds its own [0,from) prefix
 * and a poisoned [from,blen) window (0xA5 bytes, NaN scales — standing in for
 * the uninitialized malloc the #2096 bug left behind). coli_kv_share_adopt_rows
 * — the exact routine mux_submit calls — then adopts [from,blen).
 *
 * For BOTH KV8 and KV_TQ the test checks:
 *   1. the adopted byte rows (Lc8/Rc8) and scales (Lsc/Rsc) are bit-identical
 *      to the donor's — the raw copy is exact — including the asymmetric
 *      scale strides: under KV8, Lsc holds coli_kv8_nscale(kv_lora,g_kv8_gs)
 *      group scales per row while Rsc holds one float per row; under KV_TQ
 *      both hold one float per row (the radius);
 *   2. the adopted rows dequantize to exactly the donor's rows, and within
 *      quantization tolerance of the pre-quantization f32 reference — i.e.
 *      the rows are valid cache, not garbage (the #2096 failure mode was
 *      attending over uninitialized memory);
 *   3. the receiver's own [0,from) prefix rows and the token history are
 *      intact — adoption copies exactly [from,blen), nothing else.
 * No model file needed: kv_alloc's CPU path only reads c->n_layers /
 * kv_lora / qk_rope, and the quant/dequant helpers are header-inline. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int fails=0;
#define CHECK(cond, ...) do{ if(!(cond)){ fails++; \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); fprintf(stderr,__VA_ARGS__); fputc('\n',stderr);} }while(0)

/* deterministic pseudo-random floats (no libc rand: reproducible across platforms) */
static uint32_t s_rng=0x2096u;
static float s_frand(void){ s_rng=s_rng*1664525u+1013904223u;
    return (float)((s_rng>>8)&0xFFFFFF)/(float)0x1000000 * 6.f - 3.f; }

static float s_l2diff(const float *a,const float *b,int n){
    double s=0; for(int i=0;i<n;i++){ double d=(double)a[i]-b[i]; s+=d*d; } return (float)sqrt(s); }
static float s_norm2(const float *a,int n){
    double s=0; for(int i=0;i<n;i++) s+=(double)a[i]*a[i]; return (float)sqrt(s); }

#define KVL 512   /* kv_lora: power of two (KV_TQ requires it) */
#define KVR 64    /* qk_rope: power of two (KV_TQ requires it) */
#define NL  2     /* n_layers -> NR = n_layers+1 layer rows */
#define FROM 5    /* receiver's own prefix length: adopt starts at a nonzero offset */
#define AN  11    /* adopted rows */
#define BLEN (FROM+AN)
#define MAXT 32

/* quantize one f32 reference row pair into the donor slot, exactly the way the
 * hot prefill producer writes them (c/colibri.c kv write path). */
static void s_prefill_row(KVState *dn, int l, int t,
                          const float *Ls, const float *Rs, int nsl){
    if(g_tq){
        int lb=coli_kvq_row_bytes(KVL,g_tq_bits,g_tq_codec);
        int rb=coli_kvq_row_bytes(KVR,g_tq_bits,g_tq_codec);
        dn->Lsc[l][t]=coli_kvq_quant_row(Ls,coli_kv_row8(dn->Lc8[l],t,lb),KVL,g_tq_bits,g_tq_codec);
        dn->Rsc[l][t]=coli_kvq_quant_row(Rs,coli_kv_row8(dn->Rc8[l],t,rb),KVR,g_tq_bits,g_tq_codec);
    }else{
        dn->Rsc[l][t]=coli_kv8_quant_row(Rs,coli_kv_row8(dn->Rc8[l],t,KVR),KVR);
        coli_kv8_quant_row_gs(Ls,coli_kv_row8(dn->Lc8[l],t,KVL),
                              dn->Lsc[l]+(int64_t)t*nsl,KVL,g_kv8_gs);
    }
}

static void s_dequant_row(KVState *k, int l, int t, float *Ld, float *Rd, int nsl){
    if(g_tq){
        int lb=coli_kvq_row_bytes(KVL,g_tq_bits,g_tq_codec);
        int rb=coli_kvq_row_bytes(KVR,g_tq_bits,g_tq_codec);
        coli_kvq_dequant_row(coli_kv_row8(k->Lc8[l],t,lb),k->Lsc[l][t],Ld,KVL,g_tq_bits,g_tq_codec);
        coli_kvq_dequant_row(coli_kv_row8(k->Rc8[l],t,rb),k->Rsc[l][t],Rd,KVR,g_tq_bits,g_tq_codec);
    }else{
        coli_kv8_dequant_row_gs(coli_kv_row8(k->Lc8[l],t,KVL),k->Lsc[l]+(int64_t)t*nsl,Ld,KVL,g_kv8_gs);
        coli_kv8_dequant_row(coli_kv_row8(k->Rc8[l],t,KVR),k->Rsc[l][t],Rd,KVR);
    }
}

/* one full adopt scenario for the current g_kv8/g_tq (and gs/bits/codec) setting */
static void s_run_case(const char *name, float tol){
    static Model m;
    memset(&m,0,sizeof(m));
    m.c.n_layers=NL; m.c.kv_lora=KVL; m.c.qk_rope=KVR;
    int nsl=g_kv8?coli_kv8_nscale(KVL,g_kv8_gs):1;
    int lb=g_tq?coli_kvq_row_bytes(KVL,g_tq_bits,g_tq_codec):KVL;
    int rb=g_tq?coli_kvq_row_bytes(KVR,g_tq_bits,g_tq_codec):KVR;

    KVState *donor=calloc(1,sizeof(KVState));
    KVState *recv =calloc(1,sizeof(KVState));
    m.kv=donor; kv_alloc(&m,MAXT);
    /* the bug's premise, pinned: under KV8/KV_TQ the f32 row arrays stay NULL */
    for(int l=0;l<NL+1;l++)
        CHECK(donor->Lc[l]==NULL&&donor->Rc[l]==NULL, "%s: f32 rows must be NULL (l=%d)",name,l);

    /* donor = full prefill of [0,BLEN): deterministic reference rows, quantized natively */
    float *refL=malloc(sizeof(float)*(size_t)BLEN*KVL);
    float *refR=malloc(sizeof(float)*(size_t)BLEN*KVR);
    for(int t=0;t<BLEN;t++){
        for(int i=0;i<KVL;i++) refL[(size_t)t*KVL+i]=s_frand();
        for(int i=0;i<KVR;i++) refR[(size_t)t*KVR+i]=s_frand();
    }
    for(int l=0;l<NL+1;l++)
        for(int t=0;t<BLEN;t++)
            s_prefill_row(donor,l,t,refL+(size_t)t*KVL,refR+(size_t)t*KVR,nsl);

    /* receiver: its own [0,FROM) prefix (different data), adopted window poisoned */
    m.kv=recv; kv_alloc(&m,MAXT);
    int *dhist=malloc(sizeof(int)*MAXT), *rhist=malloc(sizeof(int)*MAXT);
    for(int t=0;t<BLEN;t++) dhist[t]=1000+t;
    for(int t=0;t<MAXT;t++) rhist[t]=-1;
    for(int l=0;l<NL+1;l++){
        for(int t=0;t<FROM;t++){
            float Ls[KVL], Rs[KVR];
            for(int i=0;i<KVL;i++) Ls[i]=s_frand()*0.5f+10.f;   /* distinct from donor */
            for(int i=0;i<KVR;i++) Rs[i]=s_frand()*0.5f-10.f;
            s_prefill_row(recv,l,t,Ls,Rs,nsl);
        }
        /* snapshot the prefix so we can prove adoption did not touch it */
        uint8_t *preL=malloc((size_t)FROM*lb), *preR=malloc((size_t)FROM*rb);
        float *preLs=malloc(sizeof(float)*(size_t)FROM*nsl), *preRs=malloc(sizeof(float)*FROM);
        memcpy(preL,recv->Lc8[l],(size_t)FROM*lb);
        memcpy(preR,recv->Rc8[l],(size_t)FROM*rb);
        memcpy(preLs,recv->Lsc[l],sizeof(float)*(size_t)FROM*nsl);
        memcpy(preRs,recv->Rsc[l],sizeof(float)*FROM);
        /* poison [FROM,BLEN): the uninitialized memory the #2096 bug attended over */
        memset(coli_kv_row8(recv->Lc8[l],FROM,lb),0xA5,(size_t)AN*lb);
        memset(coli_kv_row8(recv->Rc8[l],FROM,rb),0xA5,(size_t)AN*rb);
        for(int t=FROM;t<BLEN;t++){
            for(int k=0;k<nsl;k++) recv->Lsc[l][(int64_t)t*nsl+k]=NAN;
            recv->Rsc[l][t]=NAN;
        }
        /* ---- the adoption under test (the exact mux_submit routine) ---- */
        coli_kv_share_adopt_rows(&m.c,recv,donor,rhist,dhist,FROM,AN);

        /* 1. raw copy is exact, incl. the asymmetric scale strides */
        CHECK(memcmp(coli_kv_row8(recv->Lc8[l],FROM,lb),coli_kv_row8(donor->Lc8[l],FROM,lb),(size_t)AN*lb)==0,
              "%s: Lc8 adopted rows differ (l=%d)",name,l);
        CHECK(memcmp(coli_kv_row8(recv->Rc8[l],FROM,rb),coli_kv_row8(donor->Rc8[l],FROM,rb),(size_t)AN*rb)==0,
              "%s: Rc8 adopted rows differ (l=%d)",name,l);
        CHECK(memcmp(recv->Lsc[l]+(int64_t)FROM*nsl,donor->Lsc[l]+(int64_t)FROM*nsl,
                    sizeof(float)*(size_t)AN*nsl)==0,
              "%s: Lsc adopted scales differ (l=%d, nsl=%d)",name,l,nsl);
        CHECK(memcmp(recv->Rsc[l]+FROM,donor->Rsc[l]+FROM,sizeof(float)*AN)==0,
              "%s: Rsc adopted scales differ (l=%d)",name,l);

        /* 2. adopted rows dequantize to the donor's rows exactly, and to the
         * pre-quantization reference within quantization tolerance */
        for(int t=FROM;t<BLEN;t++){
            float dLd[KVL], dRd[KVR], rLd[KVL], rRd[KVR];
            s_dequant_row(donor,l,t,dLd,dRd,nsl);
            s_dequant_row(recv,l,t,rLd,rRd,nsl);
            CHECK(memcmp(rLd,dLd,sizeof(float)*KVL)==0&&memcmp(rRd,dRd,sizeof(float)*KVR)==0,
                  "%s: adopted row dequant != donor row (l=%d t=%d)",name,l,t);
            float eL=s_l2diff(rLd,refL+(size_t)t*KVL,KVL)/(s_norm2(refL+(size_t)t*KVL,KVL)+1e-6f);
            float eR=s_l2diff(rRd,refR+(size_t)t*KVR,KVR)/(s_norm2(refR+(size_t)t*KVR,KVR)+1e-6f);
            CHECK(eL<tol&&eR<tol,"%s: adopted row not within quant tolerance of reference (l=%d t=%d eL=%g eR=%g tol=%g)",
                  name,l,t,eL,eR,tol);
        }

        /* 3. the receiver's own prefix is untouched, history adopted exactly */
        CHECK(memcmp(recv->Lc8[l],preL,(size_t)FROM*lb)==0, "%s: receiver Lc8 prefix clobbered (l=%d)",name,l);
        CHECK(memcmp(recv->Rc8[l],preR,(size_t)FROM*rb)==0, "%s: receiver Rc8 prefix clobbered (l=%d)",name,l);
        CHECK(memcmp(recv->Lsc[l],preLs,sizeof(float)*(size_t)FROM*nsl)==0, "%s: receiver Lsc prefix clobbered (l=%d)",name,l);
        CHECK(memcmp(recv->Rsc[l],preRs,sizeof(float)*FROM)==0, "%s: receiver Rsc prefix clobbered (l=%d)",name,l);
        for(int t=FROM;t<BLEN;t++)
            CHECK(rhist[t]==dhist[t],"%s: hist not adopted at %d (%d!=%d)",name,t,rhist[t],dhist[t]);
        for(int t=0;t<FROM;t++)
            CHECK(rhist[t]==-1,"%s: hist prefix clobbered at %d",name,t);

        free(preL); free(preR); free(preLs); free(preRs);
    }
    free(dhist); free(rhist);
    free(refL); free(refR);
    /* (KVStates are intentionally not freed: same convention as test_kv_alloc) */
    if(!fails) fprintf(stderr,"[info] %s: adopt exact, rows valid, prefix+hist intact\n",name);
}

int main(void){
    /* KV8 with per-row scales and with 128-wide group scales (the asymmetric
     * stride trap: Lsc multi-float vs Rsc single float per row). */
    g_kv8=1; g_tq=0; g_kv8_gs=0;   s_run_case("KV8 gs=0",0.10f);
    g_kv8=1; g_tq=0; g_kv8_gs=128; s_run_case("KV8 gs=128",0.10f);
    g_kv8=0;
    /* KV_TQ: the production default codec (rotated int4) and PolarQuant. */
    g_tq=1; g_tq_bits=4; g_tq_codec=1; s_run_case("KV_TQ codec=1 bits=4",0.60f);
    g_tq=1; g_tq_bits=4; g_tq_codec=0; s_run_case("KV_TQ codec=0 bits=4",0.60f);
    g_tq=0;

    if(fails){ fprintf(stderr,"%d failure(s)\n",fails); return 1; }
    printf("OK kv_share_kv8: cross-slot adoption copies KV8/KV_TQ rows+scales exactly\n");
    return 0;
}
