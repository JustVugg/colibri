/* Kimi K3 inference engine in pure C — sibling of colibri.c (GLM-5.2) / olmoe.c /
 * inkling.c, sharing st.h / json.h / tok.h / quant.h.
 *
 * Architecture (2.8T total / 104B active, 93 layers, hidden 7168):
 *   - Hybrid attention: 69 KDA (Kimi Delta Attention, linear/recurrent) +
 *     24 gated MLA layers (every 4th, plus the final layer). MLA is NoPE —
 *     no positional encoding anywhere in the model; position lives in KDA's
 *     decay/conv. Both attention types carry a full-rank sigmoid output gate.
 *   - AttnRes (Attention Residuals) REPLACE the plain residual stream: a
 *     running prefix_sum plus block snapshots (layers 0,12,24,...,84), mixed
 *     by softmax twice per layer (before attention, before MLP) and once at
 *     the end. Weights: per-layer {self_attention,mlp}_res_{norm,proj}.
 *   - Stable LatentMoE: router (sigmoid + e_score_correction_bias, top-16 of
 *     896, renormalized raw scores), shared latent down/up projections
 *     (7168<->3584), per-expert GLU in the 3584 latent with moe_inter 3072,
 *     RMSNorm on the aggregate, plus 2 fused shared experts (inter 6144) at
 *     full width. Activation is SiTU-GLU:
 *         b1*tanh(g/b1)*sigmoid(g) * b2*tanh(u/b2),  b1=4, b2=25.
 *   - Routed experts are NATIVE MXFP4 (QAT; e2m1 nibbles + ue8m0 scale per
 *     32, compressed-tensors "mxfp4-pack-quantized") and are streamed
 *     straight from the original HF shards — never re-encoded, never
 *     converted. Everything else is BF16 in the checkpoint and quantized at
 *     LOAD TIME into RAM (int8 per-row / int4-g64 / f32, see K3_*BITS).
 *
 * KDA per-head recurrence (head dim 128, 96 heads; fla fused_recurrent_kda):
 *     q,k,v = SiLU(ShortConv4(W{q,k,v} x));  q,k L2-normalized (eps 1e-6
 *     inside the sqrt), q *= 128^-0.5
 *     z = W_fb(W_fa x) + dt_bias            (per channel, dt_bias[12288])
 *     gk = gmin * sigmoid(exp(A_log[h]) * z),  gmin=-5;  alpha = exp(gk)
 *     S = (I - beta k k^T) Diag(alpha) S + beta k v^T,  beta = sigmoid(W_b x)
 *     o = S^T q;  out = W_o [ sigmoid(W_g x) * RMSNorm_head(o) ]
 *   (checkpoint A_log is [128] = per-head [96] zero-padded, first 96 used)
 *
 * Model dir = the HF snapshot (config.json + model-*-of-000096.safetensors).
 * tokenizer.json is synthesized once by tools/k3_tokenizer.py (the HF repo
 * ships only tiktoken.model); without it the engine still runs on raw ids.
 *
 * ENV:
 *   K3_BITS=4|8|32       load-time quant of KDA/latent/shared/dense (default 4)
 *   K3_MLA_BITS=8|4|32   MLA projections (default 8)
 *   K3_HEAD_BITS=8|4|32  lm_head (default 8)
 *   K3_EXPERT_GB=N       routed-expert LRU cache budget (default 8)
 *   K3_DIRECT=0|1        O_DIRECT expert reads (default 1; buffered fallback)
 *   K3_LAYERS=N          truncate to first N layers (validation; skips head)
 *   K3_TRACE=path        dump f32 hidden state after every layer (validation)
 *   K3_MAXT=N            KV/context capacity (default prompt+ngen)
 *   COLI_TEMP=F          0 = greedy (default), else softmax temperature
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "st.h"
#include "tok.h"
#include "quant.h"

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, vocab, first_dense, dense_inter;
    /* MLA */
    int n_heads, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head;
    float attn_scale;
    /* KDA */
    int kda_heads, kda_hd, kda_proj, conv_k;
    float gate_lb;
    /* MoE */
    int n_experts, topk, moe_inter, latent, n_shared;
    float situ_b1, situ_b2;
    /* AttnRes */
    int res_bs;
    float eps;
    int8_t is_kda[128];
    int bos, eos[8], n_eos;
} Cfg;

/* ---------- RAM-resident weight, quantized at load ---------- */
typedef struct { int fmt; float *f; int8_t *q8; uint8_t *q4; float *s; int O, I, gs; } W;

typedef struct {                          /* KDA layer */
    W q, k, v, o, g;
    float *conv_q, *conv_k, *conv_v;      /* [proj*4] depthwise taps, oldest first */
    float *fa, *fb;                       /* decay low-rank, f32 [hd,hidden] [proj,hd] */
    float *bp;                            /* beta proj f32 [heads,hidden] */
    float *dt, *A, *onw;                  /* dt_bias[proj], exp(A_log)[heads], o_norm[hd] */
} Kda;

typedef struct {                          /* gated MLA layer */
    W qa, qb, kva, kvb, o, g;
    float *qa_ln, *kva_ln;
} Mla;

typedef struct {                          /* LatentMoE */
    float *router, *rbias, *lat_norm;     /* [E,hidden] f32, [E], [latent] */
    W lat_down, lat_up, sh_gate, sh_up, sh_down;
} Moe;

typedef struct {
    int kda, sparse;
    Kda a; Mla m; Moe moe;
    W d_gate, d_up, d_down;               /* dense layer only */
    float *in_ln, *post_ln;
    float *attn_sw, *mlp_sw;              /* AttnRes score weights: norm.w * proj.w */
} Layer;

/* ---------- routed-expert streaming (native MXFP4 from the HF shards) ---- */
typedef struct { int fd[6]; int64_t off[6]; int contig; } ERef;  /* w1p w1s w2p w2s w3p w3s */
typedef struct { int eid; uint8_t *buf, *base; uint64_t used; } Slot;
                          /* base = 4K-aligned allocation (O_DIRECT target);
                           * buf = expert data view inside it (= base + off%4K) */
typedef struct { Slot *s; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    char pfx[40];                         /* "language_model." or "" */
    Layer *L;
    float *final_norm, *out_sw;
    W lm_head;
    int has_head;
    Slot ws[64];                          /* working set: parallel loads land here,
                                           * then swap into the layer LRU */
    /* KDA state */
    float **kstate;                       /* [layer] -> [heads*hd*hd], S[k][v] */
    float **cwq, **cwk, **cwv;            /* conv windows [proj*conv_k], oldest first */
    /* MLA cache */
    float **Lc, **Rc; int max_t;
    /* experts */
    ERef *eref;                           /* [n_layers][n_experts] (dense rows zeroed) */
    LCache *ecache;
    int64_t e_w1p, e_w1s, e_w2p, e_w2s, e_slot;
    uint64_t clock, hits, miss, ebytes;
    double t_attn, t_moe, t_eload, t_head;
    FILE *trace;
} Model;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#if defined(__APPLE__)
    return r.ru_maxrss/(1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss/(1024.0*1024.0);
#endif
}
static float *falloc(int64_t n){ float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} return p; }
static float *fcalloc(int64_t n){ float *p=calloc((size_t)n,sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} return p; }
static inline float sigmoidf_(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf_(float x){ return x/(1.f+expf(-x)); }
static void softmax_(float *x, int n){ float m=x[0]; for(int i=1;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; } for(int i=0;i<n;i++) x[i]/=s; }
static void rmsnorm_(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps);
    for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}

/* ---------- W: load-time quantization + matvec ---------- */
static void w_matmul(float *y, const float *x, const W *w, int S){
    if(w->fmt==0)      matmul(y,x,w->f,S,w->I,w->O);
    else if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==4) matmul_i4_grouped(y,x,w->q4,w->s,S,w->I,w->O,w->gs);
    else { fprintf(stderr,"w_matmul: bad fmt %d\n",w->fmt); exit(1); }
}
/* acc[0..I) += coef * row r (MLA absorb builds q_abs from kv_b rows) */
static void w_addrow(const W *w, int r, float coef, float *acc){
    int I=w->I;
    if(w->fmt==0){ const float *p=w->f+(int64_t)r*I; for(int i=0;i<I;i++) acc[i]+=coef*p[i]; }
    else if(w->fmt==1){ const int8_t *p=w->q8+(int64_t)r*I; float s=w->s[r]*coef;
        for(int i=0;i<I;i++) acc[i]+=s*p[i]; }
    else { int rb=(I+1)/2, ng=(I+w->gs-1)/w->gs; const uint8_t *p=w->q4+(int64_t)r*rb;
        const float *scl=w->s+(int64_t)r*ng;
        for(int g=0;g*w->gs<I;g++){ float s=scl[g]*coef; int e=(g+1)*w->gs; if(e>I)e=I;
            for(int i=g*w->gs;i<e;i+=2){ uint8_t b=p[i>>1];
                acc[i]+=s*(float)((int)(b&0xF)-8);
                if(i+1<e) acc[i+1]+=s*(float)((int)(b>>4)-8); } } }
}
static float w_rowdot(const W *w, int r, const float *x){
    int I=w->I; float a=0;
    if(w->fmt==0){ const float *p=w->f+(int64_t)r*I; for(int i=0;i<I;i++) a+=x[i]*p[i]; return a; }
    if(w->fmt==1){ const int8_t *p=w->q8+(int64_t)r*I; for(int i=0;i<I;i++) a+=x[i]*p[i]; return a*w->s[r]; }
    { int rb=(I+1)/2, ng=(I+w->gs-1)/w->gs; const uint8_t *p=w->q4+(int64_t)r*rb;
      const float *scl=w->s+(int64_t)r*ng;
      for(int g=0;g*w->gs<I;g++){ float ga=0; int e=(g+1)*w->gs; if(e>I)e=I;
          for(int i=g*w->gs;i<e;i+=2){ uint8_t b=p[i>>1];
              ga+=x[i]*(float)((int)(b&0xF)-8);
              if(i+1<e) ga+=x[i+1]*(float)((int)(b>>4)-8); }
          a+=ga*scl[g]; } }
    return a;
}

#define QCHUNK 1024                      /* rows per load-quantize pass */
static int g_bits_env=0;                 /* K3_BITS explicitly set: enables the
                                          * int8-container -> int4 load downcast */
static int g_k3_direct=-1;               /* K3_DIRECT: O_DIRECT expert reads */
static void w_load(Model *m, W *w, const char *name, int O, int I, int bits){
    char nm[512]; snprintf(nm,sizeof(nm),"%s%s",m->pfx,name);
    st_tensor *t=st_find(&m->S,nm);
    if(!t) st_die_missing(&m->S,nm);
    memset(w,0,sizeof(*w)); w->O=O; w->I=I;
    if(t->dtype==3){
        /* repacked container (tools/k3_repack.py): pre-quantized U8 + .qs f32
         * scales — no load-time quantization, K3_BITS is ignored for these
         * (except the explicit =4 downcast below) */
        char qn[560]; snprintf(qn,sizeof(qn),"%s.qs",nm);
        st_tensor *ts=st_find(&m->S,qn);
        if(!ts){ fprintf(stderr,"%s: quantized (U8) but no %s scale sidecar\n",nm,qn); exit(1); }
        if(t->nbytes==(int64_t)O*I && ts->numel==O){                  /* int8 per-row */
            if(g_bits_env && bits==4 && I%64==0){
                /* EXPLICIT K3_BITS=4 on an int8 container: downcast to int4-g64
                 * at load. Halves resident RAM (the 62 GB box cannot hold the
                 * 93-layer non-expert set at int8 next to a desktop session);
                 * the int8 grid is 16x finer than int4, so the double-quant
                 * noise ~ the direct-int4 noise. Unset K3_BITS keeps the
                 * container's own bits — the default is untouched. */
                int gs=64, rb=I/2, ng=I/gs;
                int8_t *q8=malloc((size_t)O*I); float *s8=falloc(O);
                if(!q8){fprintf(stderr,"OOM int8 tmp %s\n",nm);exit(1);}
                st_read_raw(&m->S,nm,q8,1); st_read_f32(&m->S,qn,s8,0);
                w->fmt=4; w->gs=gs;
                w->q4=malloc((int64_t)O*rb); w->s=falloc((int64_t)O*ng);
                if(!w->q4){fprintf(stderr,"OOM int4 %s\n",nm);exit(1);}
                for(int r=0;r<O;r++){
                    const int8_t *src=q8+(int64_t)r*I; float sc8=s8[r];
                    uint8_t *dst=w->q4+(int64_t)r*rb; float *scl=w->s+(int64_t)r*ng;
                    for(int g=0;g<ng;g++){ const int8_t *gp=src+g*gs;
                        int am=0; for(int i=0;i<gs;i++){ int a=gp[i]<0?-gp[i]:gp[i]; if(a>am)am=a; }
                        float s=am*sc8/7.f; if(s<1e-20f)s=1e-20f; scl[g]=s; float inv=sc8/s;
                        for(int i=0;i<gs;i+=2){
                            int v0=(int)lrintf(gp[i]*inv);   if(v0>7)v0=7; if(v0<-8)v0=-8;
                            int v1=(int)lrintf(gp[i+1]*inv); if(v1>7)v1=7; if(v1<-8)v1=-8;
                            dst[(g*gs+i)>>1]=(uint8_t)((v0+8)|((v1+8)<<4)); } } }
                free(q8); free(s8);
                return;
            }
            w->fmt=1; w->q8=malloc((size_t)O*I);
            if(!w->q8){fprintf(stderr,"OOM int8 %s\n",nm);exit(1);}
            st_read_raw(&m->S,nm,w->q8,1);
            w->s=falloc(O); st_read_f32(&m->S,qn,w->s,0);
        } else if(I%64==0 && t->nbytes==(int64_t)O*(I/2) && ts->numel==(int64_t)O*(I/64)){
            w->fmt=4; w->gs=64;                                       /* int4-g64 */
            w->q4=malloc((size_t)O*(I/2));
            if(!w->q4){fprintf(stderr,"OOM int4 %s\n",nm);exit(1);}
            st_read_raw(&m->S,nm,w->q4,1);
            w->s=falloc((int64_t)O*(I/64)); st_read_f32(&m->S,qn,w->s,0);
        } else {
            fprintf(stderr,"%s: U8 tensor is %lld bytes / %lld scales — matches neither int8 [%d,%d] nor int4-g64, refusing (untrusted container)\n",
                    nm,(long long)t->nbytes,(long long)ts->numel,O,I); exit(1);
        }
        return;
    }
    if(t->numel!=(int64_t)O*I){ fprintf(stderr,"%s: numel %lld != %dx%d\n",nm,(long long)t->numel,O,I); exit(1); }
    if(bits>=32){ w->fmt=0; w->f=falloc((int64_t)O*I); st_read_f32(&m->S,nm,w->f,0); return; }
    int gs=64;
    if(bits<=4 && I%gs){ bits=8; }        /* int4-g64 wants I%64==0; fall back */
    float *scr=falloc((int64_t)QCHUNK*I);
    if(bits>4){ w->fmt=1; w->q8=malloc((int64_t)O*I); w->s=falloc(O);
        if(!w->q8){fprintf(stderr,"OOM int8 %s\n",nm);exit(1);}
        for(int r0=0;r0<O;r0+=QCHUNK){ int n=O-r0<QCHUNK?O-r0:QCHUNK;
            st_read_slice_f32(&m->S,nm,(int64_t)r0*I,(int64_t)n*I,scr,1);
            for(int r=0;r<n;r++){ const float *src=scr+(int64_t)r*I;
                float am=0; for(int i=0;i<I;i++){ float a=fabsf(src[i]); if(a>am)am=a; }
                float s=am/127.f; if(s<1e-20f)s=1e-20f; w->s[r0+r]=s; float inv=1.f/s;
                int8_t *dst=w->q8+(int64_t)(r0+r)*I;
                for(int i=0;i<I;i++){ int v=(int)lrintf(src[i]*inv); if(v>127)v=127; if(v<-127)v=-127; dst[i]=(int8_t)v; } } }
    } else { w->fmt=4; w->gs=gs; int rb=I/2, ng=I/gs;
        w->q4=malloc((int64_t)O*rb); w->s=falloc((int64_t)O*ng);
        if(!w->q4){fprintf(stderr,"OOM int4 %s\n",nm);exit(1);}
        for(int r0=0;r0<O;r0+=QCHUNK){ int n=O-r0<QCHUNK?O-r0:QCHUNK;
            st_read_slice_f32(&m->S,nm,(int64_t)r0*I,(int64_t)n*I,scr,1);
            for(int r=0;r<n;r++){ const float *src=scr+(int64_t)r*I;
                uint8_t *dst=w->q4+(int64_t)(r0+r)*rb; float *scl=w->s+(int64_t)(r0+r)*ng;
                for(int g=0;g<ng;g++){ const float *gp=src+g*gs;
                    float am=0; for(int i=0;i<gs;i++){ float a=fabsf(gp[i]); if(a>am)am=a; }
                    float s=am/7.f; if(s<1e-20f)s=1e-20f; scl[g]=s; float inv=1.f/s;
                    for(int i=0;i<gs;i+=2){
                        int v0=(int)lrintf(gp[i]*inv);   if(v0>7)v0=7; if(v0<-8)v0=-8;
                        int v1=(int)lrintf(gp[i+1]*inv); if(v1>7)v1=7; if(v1<-8)v1=-8;
                        dst[(g*gs+i)>>1]=(uint8_t)((v0+8)|((v1+8)<<4)); } } } }
    }
    free(scr);
}
static float *f32_load(Model *m, const char *name, int64_t want){
    char nm[512]; snprintf(nm,sizeof(nm),"%s%s",m->pfx,name);
    st_tensor *t=st_find(&m->S,nm);
    if(!t) st_die_missing(&m->S,nm);
    if(want>0 && t->numel!=want){ fprintf(stderr,"%s: numel %lld != %lld\n",nm,(long long)t->numel,(long long)want); exit(1); }
    float *p=falloc(t->numel); st_read_f32(&m->S,nm,p,0); return p;
}

/* ---------- config ---------- */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static void load_cfg(Cfg *c, const char *snap){
    char path[2048]; snprintf(path,sizeof(path),"%s/config.json",snap);
    long n; char *buf;
    { FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
      fseek(f,0,SEEK_END); n=ftell(f); fseek(f,0,SEEK_SET);
      if(n<0||n>(64L<<20)){ fprintf(stderr,"%s: bad size\n",path); exit(1); }
      buf=malloc((size_t)n+1); if(!buf){fprintf(stderr,"OOM cfg\n");exit(1);}
      if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); }
      buf[n]=0; fclose(f); }
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    jval *tc=json_get(root,"text_config"); if(!tc||tc->t!=J_OBJ) tc=root;
    memset(c,0,sizeof(*c));
    c->hidden      =(int)req_num(tc,"hidden_size");
    c->n_layers    =(int)req_num(tc,"num_hidden_layers");
    c->vocab       =(int)req_num(tc,"vocab_size");
    c->first_dense =(int)req_num(tc,"first_k_dense_replace");
    c->dense_inter =(int)req_num(tc,"intermediate_size");
    c->n_heads     =(int)req_num(tc,"num_attention_heads");
    c->q_lora      =(int)req_num(tc,"q_lora_rank");
    c->kv_lora     =(int)req_num(tc,"kv_lora_rank");
    c->qk_nope     =(int)req_num(tc,"qk_nope_head_dim");
    c->qk_rope     =(int)req_num(tc,"qk_rope_head_dim");
    c->v_head      =(int)req_num(tc,"v_head_dim");
    c->n_experts   =(int)req_num(tc,"num_experts");
    c->topk        =(int)req_num(tc,"num_experts_per_token");
    c->moe_inter   =(int)req_num(tc,"moe_intermediate_size");
    c->latent      =(int)req_num(tc,"routed_expert_hidden_size");
    c->n_shared    =(int)req_num(tc,"num_shared_experts");
    c->res_bs      =(int)req_num(tc,"attn_res_block_size");
    c->situ_b1     =(float)req_num(tc,"activation_situ_beta");
    c->situ_b2     =(float)req_num(tc,"activation_situ_linear_beta");
    jval *ep=json_get(tc,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    c->qk_head=c->qk_nope+c->qk_rope;
    c->attn_scale=1.f/sqrtf((float)c->qk_head);
    jval *la=json_get(tc,"linear_attn_config");
    if(!la||la->t!=J_OBJ){ fprintf(stderr,"config.json: missing linear_attn_config\n"); exit(1); }
    c->kda_heads=(int)req_num(la,"num_heads");
    c->kda_hd   =(int)req_num(la,"head_dim");
    c->conv_k   =(int)req_num(la,"short_conv_kernel_size");
    jval *lb=json_get(la,"gate_lower_bound"); c->gate_lb=lb?(float)lb->num:-5.f;
    c->kda_proj=c->kda_heads*c->kda_hd;
    if(c->hidden<1||c->hidden>65536||c->n_layers<1||c->n_layers>128||
       c->n_experts<1||c->n_experts>4096||c->topk<1||c->topk>64||c->topk>c->n_experts||
       c->vocab<1||c->vocab>(1<<22)||c->kda_proj<1||c->kda_proj>(1<<20)||
       c->conv_k<1||c->conv_k>8||c->latent<32||c->latent%32||c->moe_inter%32||
       c->res_bs<1||c->kda_hd>512||c->kv_lora>4096){
        fprintf(stderr,"config.json: dimension out of range\n"); exit(1); }
    jval *kl=json_get(la,"kda_layers");
    if(!kl||kl->t!=J_ARR){ fprintf(stderr,"config.json: missing kda_layers\n"); exit(1); }
    for(int i=0;i<kl->len;i++){ int v=(int)kl->kids[i]->num;      /* 1-indexed */
        if(v>=1&&v<=c->n_layers) c->is_kda[v-1]=1; }
    jval *b=json_get(root,"bos_token_id"); if(!b) b=json_get(tc,"bos_token_id");
    c->bos = b&&b->t==J_NUM ? (int)b->num : -1;
    jval *e=json_get(root,"eos_token_id"); if(!e) e=json_get(tc,"eos_token_id");
    if(e&&e->t==J_NUM) c->eos[c->n_eos++]=(int)e->num;
    else if(e&&e->t==J_ARR) for(int i=0;i<e->len&&c->n_eos<8;i++) c->eos[c->n_eos++]=(int)e->kids[i]->num;
    free(buf); (void)arena;
}

/* ---------- init ---------- */
static void expert_table_init(Model *m){
    Cfg *c=&m->c;
    m->e_w1p=(int64_t)c->moe_inter*(c->latent/2);   m->e_w1s=(int64_t)c->moe_inter*(c->latent/32);
    m->e_w2p=(int64_t)c->latent*(c->moe_inter/2);   m->e_w2s=(int64_t)c->latent*(c->moe_inter/32);
    m->e_slot=2*(m->e_w1p+m->e_w1s)+m->e_w2p+m->e_w2s;
    m->eref=calloc((size_t)c->n_layers*c->n_experts,sizeof(ERef));
    if(!m->eref){fprintf(stderr,"OOM expert table\n");exit(1);}
    const char *mat[3]={"w1","w2","w3"};
    int64_t want[6]={m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s,m->e_w1p,m->e_w1s};
    int missing=0;
    for(int li=0;li<c->n_layers;li++){
        if(!m->L[li].sparse) continue;
        for(int e2=0;e2<c->n_experts;e2++){
            ERef *er=&m->eref[(int64_t)li*c->n_experts+e2];
            for(int k=0;k<6;k++){
                char nm[512];
                snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.experts.%d.%s.weight_%s",
                         m->pfx,li,e2,mat[k/2],(k&1)?"scale":"packed");
                st_tensor *t=st_find(&m->S,nm);
                if(!t){ missing++; er->fd[k]=-1; continue; }
                if(t->nbytes!=want[k]){ fprintf(stderr,"%s: %lld bytes, expected %lld — refusing (untrusted container)\n",
                        nm,(long long)t->nbytes,(long long)want[k]); exit(1); }
                er->fd[k]=t->fd; er->off[k]=t->off;
            }
            /* HF shards store the six tensors back-to-back (measured: 0 gaps
             * across a whole layer) — collapse the load to ONE pread when so */
            er->contig=1;
            for(int k=0;k<5;k++)
                if(er->fd[k]!=er->fd[k+1]||er->off[k]+want[k]!=er->off[k+1]) er->contig=0;
        }
    }
    if(missing) fprintf(stderr,"[K3] WARNING: %d expert tensors missing (incomplete download?) — touching one aborts\n",missing);
}

static void model_init(Model *m, const char *snap, int n_layers_env){
    memset(m,0,sizeof(*m));
    load_cfg(&m->c,snap);
    Cfg *c=&m->c;
    if(n_layers_env>0&&n_layers_env<c->n_layers) c->n_layers=n_layers_env;
    st_init_multi(&m->S,snap,getenv("K3_DIRS"));   /* K3_DIRS: extra shard dirs (multi-drive split) */
    m->pfx[0]=0;   /* probe a layer-0 tensor: embed/head live in one of the LAST shards */
    if(!st_has(&m->S,"model.layers.0.input_layernorm.weight")&&
       st_has(&m->S,"language_model.model.layers.0.input_layernorm.weight"))
        snprintf(m->pfx,sizeof(m->pfx),"language_model.");
    if((c->n_layers+c->res_bs-1)/c->res_bs+1>16){ fprintf(stderr,"attn_res: too many blocks\n"); exit(1); }
    g_bits_env = getenv("K3_BITS")!=NULL;
    g_k3_direct = getenv("K3_DIRECT")?atoi(getenv("K3_DIRECT")):1;
    int bits   = getenv("K3_BITS")?atoi(getenv("K3_BITS")):4;
    int mbits  = getenv("K3_MLA_BITS")?atoi(getenv("K3_MLA_BITS")):8;
    int hbits  = getenv("K3_HEAD_BITS")?atoi(getenv("K3_HEAD_BITS")):8;
    double t0=now_s();
    m->L=calloc(c->n_layers,sizeof(Layer));
    m->kstate=calloc(c->n_layers,sizeof(float*));
    m->cwq=calloc(c->n_layers,sizeof(float*));
    m->cwk=calloc(c->n_layers,sizeof(float*));
    m->cwv=calloc(c->n_layers,sizeof(float*));
    char nm[512];
    #define NM(...) (snprintf(nm,sizeof(nm),__VA_ARGS__),nm)
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        l->kda=c->is_kda[i];
        l->sparse=(i>=c->first_dense);
        l->in_ln  =f32_load(m,NM("model.layers.%d.input_layernorm.weight",i),c->hidden);
        l->post_ln=f32_load(m,NM("model.layers.%d.post_attention_layernorm.weight",i),c->hidden);
        /* AttnRes score weight = res_norm.weight * res_proj.weight (elementwise) */
        { float *rn=f32_load(m,NM("model.layers.%d.self_attention_res_norm.weight",i),c->hidden);
          float *rp=f32_load(m,NM("model.layers.%d.self_attention_res_proj.weight",i),c->hidden);
          l->attn_sw=falloc(c->hidden); for(int d=0;d<c->hidden;d++) l->attn_sw[d]=rn[d]*rp[d];
          free(rn); free(rp);
          rn=f32_load(m,NM("model.layers.%d.mlp_res_norm.weight",i),c->hidden);
          rp=f32_load(m,NM("model.layers.%d.mlp_res_proj.weight",i),c->hidden);
          l->mlp_sw=falloc(c->hidden); for(int d=0;d<c->hidden;d++) l->mlp_sw[d]=rn[d]*rp[d];
          free(rn); free(rp); }
        if(l->kda){
            Kda *a=&l->a; int P=c->kda_proj;
            w_load(m,&a->q,NM("model.layers.%d.self_attn.q_proj.weight",i),P,c->hidden,bits);
            w_load(m,&a->k,NM("model.layers.%d.self_attn.k_proj.weight",i),P,c->hidden,bits);
            w_load(m,&a->v,NM("model.layers.%d.self_attn.v_proj.weight",i),P,c->hidden,bits);
            w_load(m,&a->g,NM("model.layers.%d.self_attn.g_proj.weight",i),P,c->hidden,bits);
            w_load(m,&a->o,NM("model.layers.%d.self_attn.o_proj.weight",i),c->hidden,P,bits);
            a->conv_q=f32_load(m,NM("model.layers.%d.self_attn.q_conv1d.weight",i),(int64_t)P*c->conv_k);
            a->conv_k=f32_load(m,NM("model.layers.%d.self_attn.k_conv1d.weight",i),(int64_t)P*c->conv_k);
            a->conv_v=f32_load(m,NM("model.layers.%d.self_attn.v_conv1d.weight",i),(int64_t)P*c->conv_k);
            a->fa=f32_load(m,NM("model.layers.%d.self_attn.f_a_proj.weight",i),(int64_t)c->kda_hd*c->hidden);
            a->fb=f32_load(m,NM("model.layers.%d.self_attn.f_b_proj.weight",i),(int64_t)P*c->kda_hd);
            a->bp=f32_load(m,NM("model.layers.%d.self_attn.b_proj.weight",i),(int64_t)c->kda_heads*c->hidden);
            a->dt=f32_load(m,NM("model.layers.%d.self_attn.dt_bias",i),P);
            a->onw=f32_load(m,NM("model.layers.%d.self_attn.o_norm.weight",i),c->kda_hd);
            { /* A_log in the checkpoint is [kda_hd] = per-head zero-padded */
              char an[512]; snprintf(an,sizeof(an),"%smodel.layers.%d.self_attn.A_log",m->pfx,i);
              st_tensor *t=st_find(&m->S,an); if(!t) st_die_missing(&m->S,an);
              if(t->numel<c->kda_heads){ fprintf(stderr,"%s: %lld < heads\n",an,(long long)t->numel); exit(1); }
              float *al=falloc(t->numel); st_read_f32(&m->S,an,al,0);
              a->A=falloc(c->kda_heads);
              for(int h=0;h<c->kda_heads;h++) a->A[h]=expf(al[h]);
              free(al); }
            m->kstate[i]=fcalloc((int64_t)c->kda_heads*c->kda_hd*c->kda_hd);
            m->cwq[i]=fcalloc((int64_t)P*c->conv_k);
            m->cwk[i]=fcalloc((int64_t)P*c->conv_k);
            m->cwv[i]=fcalloc((int64_t)P*c->conv_k);
        } else {
            Mla *a=&l->m;
            w_load(m,&a->qa,NM("model.layers.%d.self_attn.q_a_proj.weight",i),c->q_lora,c->hidden,mbits);
            w_load(m,&a->qb,NM("model.layers.%d.self_attn.q_b_proj.weight",i),c->n_heads*c->qk_head,c->q_lora,mbits);
            w_load(m,&a->kva,NM("model.layers.%d.self_attn.kv_a_proj_with_mqa.weight",i),c->kv_lora+c->qk_rope,c->hidden,mbits);
            w_load(m,&a->kvb,NM("model.layers.%d.self_attn.kv_b_proj.weight",i),c->n_heads*(c->qk_nope+c->v_head),c->kv_lora,mbits);
            w_load(m,&a->o,NM("model.layers.%d.self_attn.o_proj.weight",i),c->hidden,c->n_heads*c->v_head,mbits);
            w_load(m,&a->g,NM("model.layers.%d.self_attn.g_proj.weight",i),c->n_heads*c->v_head,c->hidden,mbits);
            a->qa_ln =f32_load(m,NM("model.layers.%d.self_attn.q_a_layernorm.weight",i),c->q_lora);
            a->kva_ln=f32_load(m,NM("model.layers.%d.self_attn.kv_a_layernorm.weight",i),c->kv_lora);
        }
        if(l->sparse){
            Moe *o=&l->moe;
            o->router=f32_load(m,NM("model.layers.%d.block_sparse_moe.gate.weight",i),(int64_t)c->n_experts*c->hidden);
            o->rbias =f32_load(m,NM("model.layers.%d.block_sparse_moe.gate.e_score_correction_bias",i),c->n_experts);
            o->lat_norm=f32_load(m,NM("model.layers.%d.block_sparse_moe.routed_expert_norm.weight",i),c->latent);
            w_load(m,&o->lat_down,NM("model.layers.%d.block_sparse_moe.routed_expert_down_proj.weight",i),c->latent,c->hidden,bits);
            w_load(m,&o->lat_up,NM("model.layers.%d.block_sparse_moe.routed_expert_up_proj.weight",i),c->hidden,c->latent,bits);
            int shi=c->moe_inter*c->n_shared;
            w_load(m,&o->sh_gate,NM("model.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight",i),shi,c->hidden,bits);
            w_load(m,&o->sh_up,NM("model.layers.%d.block_sparse_moe.shared_experts.up_proj.weight",i),shi,c->hidden,bits);
            w_load(m,&o->sh_down,NM("model.layers.%d.block_sparse_moe.shared_experts.down_proj.weight",i),c->hidden,shi,bits);
        } else {
            w_load(m,&l->d_gate,NM("model.layers.%d.mlp.gate_proj.weight",i),c->dense_inter,c->hidden,bits);
            w_load(m,&l->d_up,NM("model.layers.%d.mlp.up_proj.weight",i),c->dense_inter,c->hidden,bits);
            w_load(m,&l->d_down,NM("model.layers.%d.mlp.down_proj.weight",i),c->hidden,c->dense_inter,bits);
        }
        if(i%8==0) fprintf(stderr,"[K3] loaded layer %d/%d (%.1fs, RSS %.1f GB)\n",i+1,c->n_layers,now_s()-t0,rss_gb());
    }
    snprintf(nm,sizeof(nm),"%smodel.norm.weight",m->pfx);
    m->has_head = st_has(&m->S,nm);
    if(m->has_head){
        m->final_norm=f32_load(m,"model.norm.weight",c->hidden);
        { float *rn=f32_load(m,"model.output_attn_res_norm.weight",c->hidden);
          float *rp=f32_load(m,"model.output_attn_res_proj.weight",c->hidden);
          m->out_sw=falloc(c->hidden); for(int d=0;d<c->hidden;d++) m->out_sw[d]=rn[d]*rp[d];
          free(rn); free(rp); }
        w_load(m,&m->lm_head,"lm_head.weight",c->vocab,c->hidden,hbits);
    } else fprintf(stderr,"[K3] final norm/lm_head not present — trace-only mode\n");
    expert_table_init(m);
    /* expert LRU cache, per-layer slots from the global budget */
    double egb = getenv("K3_EXPERT_GB")?atof(getenv("K3_EXPERT_GB")):8.0;
    int nmoe=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nmoe++;
    int cap=(int)((egb*1e9)/((double)m->e_slot*(nmoe?nmoe:1)));
    /* floor 1, NOT topk: experts are loaded and consumed one at a time inside
     * a token, so slots never need to hold a whole top-k set. A topk floor
     * would silently commit topk*nmoe slots (~26 GB on the 93-layer model)
     * regardless of K3_EXPERT_GB. */
    if(cap<1) cap=1;
    if(cap>c->n_experts) cap=c->n_experts;
    m->ecache=calloc(c->n_layers,sizeof(LCache));
    for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse){
        m->ecache[i].cap=cap; m->ecache[i].s=calloc(cap,sizeof(Slot));
        for(int j2=0;j2<cap;j2++) m->ecache[i].s[j2].eid=-1;
    }
    fprintf(stderr,"[K3] init done in %.1fs | %d layers | expert cache %d/layer (%.1f MB/slot) | RSS %.1f GB\n",
            now_s()-t0,c->n_layers,cap,m->e_slot/1e6,rss_gb());
    #undef NM
}

/* ---------- AttnRes softmax mix over [bres rows..., prefix] ---------- */
static void res_mix(float *out, const float *prefix, const float *bres, int nb, int D,
                    const float *sw, float eps){
    const float *v[16]; float sc[16];
    for(int e=0;e<nb;e++) v[e]=bres+(int64_t)e*D;
    v[nb]=prefix;
    for(int e=0;e<=nb;e++){
        double ms=0,dot=0;
        for(int d=0;d<D;d++){ double x=v[e][d]; ms+=x*x; dot+=x*(double)sw[d]; }
        sc[e]=(float)(dot/sqrt(ms/D+eps));
    }
    softmax_(sc,nb+1);
    for(int d=0;d<D;d++){ float a=0; for(int e=0;e<=nb;e++) a+=sc[e]*v[e][d]; out[d]=a; }
}

/* ---------- KDA layer (single token) ---------- */
static void kda_forward(Model *m, Layer *l, int li, const float *x, float *out){
    Cfg *c=&m->c; Kda *a=&l->a;
    int P=c->kda_proj, H=c->kda_heads, hd=c->kda_hd, K=c->conv_k;
    float *q=falloc(P), *k=falloc(P), *v=falloc(P), *gp=falloc(P), *on=falloc(P);
    float *t1=falloc(c->kda_hd), *graw=falloc(P), *braw=falloc(H);
    w_matmul(q,x,&a->q,1); w_matmul(k,x,&a->k,1); w_matmul(v,x,&a->v,1);
    w_matmul(gp,x,&a->g,1);
    matmul(t1,x,a->fa,1,c->hidden,c->kda_hd);
    matmul(graw,t1,a->fb,1,c->kda_hd,P);
    matmul(braw,x,a->bp,1,c->hidden,H);
    /* depthwise causal conv (window: oldest..newest) + SiLU, window rolls forward */
    float *wins[3]={m->cwq[li],m->cwk[li],m->cwv[li]};
    float *vecs[3]={q,k,v}; float *taps[3]={a->conv_q,a->conv_k,a->conv_v};
    for(int t=0;t<3;t++){
        float *win=wins[t], *vec=vecs[t]; const float *cw=taps[t];
        #pragma omp parallel for schedule(static)
        for(int d=0;d<P;d++){
            float *wd=win+(int64_t)d*K;
            for(int j=0;j<K-1;j++) wd[j]=wd[j+1];
            wd[K-1]=vec[d];
            float acc=0; const float *cd=cw+(int64_t)d*K;
            for(int j=0;j<K;j++) acc+=cd[j]*wd[j];
            vec[d]=siluf_(acc);
        }
    }
    float qscale=1.f/sqrtf((float)hd);
    #pragma omp parallel for schedule(static)
    for(int h=0;h<H;h++){
        const float *qh=q+(int64_t)h*hd, *kh=k+(int64_t)h*hd, *vh=v+(int64_t)h*hd;
        float qn[512], kn[512], alpha[512], kS[512], vt[512], oh[512];
        float sq=0,sk=0;
        for(int i=0;i<hd;i++){ sq+=qh[i]*qh[i]; sk+=kh[i]*kh[i]; }
        sq=1.f/sqrtf(sq+1e-6f); sk=1.f/sqrtf(sk+1e-6f);
        for(int i=0;i<hd;i++){ qn[i]=qh[i]*sq*qscale; kn[i]=kh[i]*sk; }
        for(int i=0;i<hd;i++){
            float z=graw[(int64_t)h*hd+i]+a->dt[(int64_t)h*hd+i];
            alpha[i]=expf(c->gate_lb*sigmoidf_(a->A[h]*z));
        }
        float beta=sigmoidf_(braw[h]);
        float *S=m->kstate[li]+(int64_t)h*hd*hd;
        memset(kS,0,hd*sizeof(float));
        for(int kk=0;kk<hd;kk++){
            float *row=S+(int64_t)kk*hd; float al=alpha[kk], kv=kn[kk];
            for(int vv=0;vv<hd;vv++){ row[vv]*=al; kS[vv]+=kv*row[vv]; }
        }
        for(int vv=0;vv<hd;vv++) vt[vv]=(vh[vv]-kS[vv])*beta;
        memset(oh,0,hd*sizeof(float));
        for(int kk=0;kk<hd;kk++){
            float *row=S+(int64_t)kk*hd; float kv=kn[kk], qq=qn[kk];
            for(int vv=0;vv<hd;vv++){ row[vv]+=kv*vt[vv]; oh[vv]+=qq*row[vv]; }
        }
        /* per-head RMSNorm * sigmoid(full-rank gate) */
        double ms=0; for(int vv=0;vv<hd;vv++) ms+=(double)oh[vv]*oh[vv];
        float r=1.f/sqrtf((float)(ms/hd)+c->eps);
        float *dst=on+(int64_t)h*hd;
        for(int vv=0;vv<hd;vv++) dst[vv]=oh[vv]*r*a->onw[vv]*sigmoidf_(gp[(int64_t)h*hd+vv]);
    }
    w_matmul(out,on,&a->o,1);
    free(q);free(k);free(v);free(gp);free(on);free(t1);free(graw);free(braw);
}

/* ---------- gated MLA layer (single token, NoPE, absorb) ---------- */
static void mla_forward(Model *m, Layer *l, int li, const float *x, int pos, float *out){
    Cfg *c=&m->c; Mla *a=&l->m;
    int H=c->n_heads, qh=c->qk_head, vh=c->v_head, kvl=c->kv_lora, qr=c->qk_rope;
    float *qa=falloc(c->q_lora), *qv=falloc((int64_t)H*qh), *ckv=falloc(kvl+qr);
    float *gv=falloc((int64_t)H*vh), *ctx=falloc((int64_t)H*vh);
    w_matmul(qa,x,&a->qa,1);
    rmsnorm_(qa,qa,a->qa_ln,c->q_lora,c->eps);
    w_matmul(qv,qa,&a->qb,1);
    w_matmul(ckv,x,&a->kva,1);
    float *Lrow=m->Lc[li]+(int64_t)pos*kvl, *Rrow=m->Rc[li]+(int64_t)pos*qr;
    rmsnorm_(Lrow,ckv,a->kva_ln,kvl,c->eps);
    memcpy(Rrow,ckv+kvl,qr*sizeof(float));           /* NoPE: cached raw, no rotation */
    w_matmul(gv,x,&a->g,1);
    int nt=pos+1;
    #pragma omp parallel for schedule(static)
    for(int h=0;h<H;h++){
        const float *qp=qv+(int64_t)h*qh, *qrp=qp+c->qk_nope;
        int rbase=h*(c->qk_nope+vh);
        float qabs[4096]; memset(qabs,0,kvl*sizeof(float));
        for(int d=0;d<c->qk_nope;d++) w_addrow(&a->kvb,rbase+d,qp[d],qabs);
        float *sc=falloc(nt);
        for(int t=0;t<nt;t++){
            const float *Lt=m->Lc[li]+(int64_t)t*kvl, *Rt=m->Rc[li]+(int64_t)t*qr;
            float s2=0; for(int i=0;i<kvl;i++) s2+=qabs[i]*Lt[i];
            for(int i=0;i<qr;i++) s2+=qrp[i]*Rt[i];
            sc[t]=s2*c->attn_scale;
        }
        softmax_(sc,nt);
        float clat[4096]; memset(clat,0,kvl*sizeof(float));
        for(int t=0;t<nt;t++){
            const float *Lt=m->Lc[li]+(int64_t)t*kvl; float s2=sc[t];
            for(int i=0;i<kvl;i++) clat[i]+=s2*Lt[i];
        }
        free(sc);
        float *cx=ctx+(int64_t)h*vh;
        for(int d=0;d<vh;d++)
            cx[d]=w_rowdot(&a->kvb,rbase+c->qk_nope+d,clat)*sigmoidf_(gv[(int64_t)h*vh+d]);
    }
    w_matmul(out,ctx,&a->o,1);
    free(qa);free(qv);free(ckv);free(gv);free(ctx);
}

/* ---------- routed experts: LRU + pread from the shards ----------
 * Loads are issued in PARALLEL (OMP over the token's misses, working-set
 * slots) and, when K3_DIRECT=1 (default) and st.h has an O_DIRECT twin fd,
 * bypass the page cache: measured on the box 7.1 GB/s direct vs 2.9 buffered
 * (and ~1.8 effective once the resident weights leave no cache headroom). */
static Slot *slot_find(Model *m, int li, int eid){
    LCache *lc=&m->ecache[li];
    for(int i=0;i<lc->n;i++) if(lc->s[i].eid==eid){ m->hits++; lc->s[i].used=++m->clock; return &lc->s[i]; }
    return NULL;
}
static void expert_read(Model *m, int li, int eid, Slot *s){
    if(!s->base){
        if(posix_memalign((void**)&s->base,4096,(size_t)m->e_slot+8192)){
            fprintf(stderr,"OOM expert slot\n"); exit(1); }
    }
    ERef *er=&m->eref[(int64_t)li*m->c.n_experts+eid];
    int64_t sizes[6]={m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s,m->e_w1p,m->e_w1s};
    if(er->fd[0]<0){ fprintf(stderr,"[K3] expert L%d E%d missing on disk\n",li,eid); exit(1); }
    if(er->contig){
        int dfd = g_k3_direct ? st_direct_fd(&m->S,er->fd[0]) : -1;
        if(dfd>=0){
            /* aligned window read; sub-4K head/tail slack handled explicitly.
             * The tail past the last aligned block (or past EOF) is fetched
             * with a tiny buffered pread — O_DIRECT wants aligned lengths. */
            int64_t a0=er->off[0]&~4095LL, pad=er->off[0]-a0;
            int64_t want=pad+m->e_slot;
            struct stat sb;
            int64_t dlen=(want+4095)&~4095LL;
            if(fstat(dfd,&sb)==0 && a0+dlen>sb.st_size) dlen=(sb.st_size-a0)&~4095LL;
            if(dlen>0) st_pread_full(dfd,s->base,dlen,a0,"pread expert direct");
            if(dlen<want)
                st_pread_full(er->fd[0],s->base+dlen,want-dlen,a0+dlen,"pread expert tail");
            s->buf=s->base+pad;
        } else {
            st_pread_full(er->fd[0],s->base,m->e_slot,er->off[0],"pread expert");
            s->buf=s->base;
        }
    } else {
        uint8_t *dst=s->base;
        for(int k=0;k<6;k++){
            if(er->fd[k]<0){ fprintf(stderr,"[K3] expert L%d E%d tensor %d missing on disk\n",li,eid,k); exit(1); }
            st_pread_full(er->fd[k],dst,sizes[k],er->off[k],"pread expert");
            dst+=sizes[k];
        }
        s->buf=s->base;
    }
    s->eid=eid;
}

static inline float situf_(float g, float u, float b1, float b2){
    return b1*tanhf(g/b1)*sigmoidf_(g) * b2*tanhf(u/b2);
}

/* u += wk * E(z) for one loaded expert slot (SiTU-GLU in the latent).
 * gate/up are [moe_inter] scratch, hz is [latent] scratch. */
static void expert_apply(Model *m, Slot *s, const float *z, float wk,
                         float *u, float *gate, float *up, float *hz){
    Cfg *c=&m->c;
    uint8_t *w1p=s->buf, *w1s=w1p+m->e_w1p, *w2p=w1s+m->e_w1s, *w2s=w2p+m->e_w2p,
            *w3p=w2s+m->e_w2s, *w3s=w3p+m->e_w1p;
    matmul_mxfp4(gate,z,w1p,w1s,1,c->latent,c->moe_inter);
    matmul_mxfp4(up,z,w3p,w3s,1,c->latent,c->moe_inter);
    for(int i=0;i<c->moe_inter;i++) gate[i]=situf_(gate[i],up[i],c->situ_b1,c->situ_b2);
    matmul_mxfp4(hz,gate,w2p,w2s,1,c->moe_inter,c->latent);
    for(int i=0;i<c->latent;i++) u[i]+=wk*hz[i];
}

/* the full routed-expert pass for one token+layer: resolve (LRU hit or
 * working-set slot), load misses IN PARALLEL, compute, promote into the LRU. */
static void experts_run(Model *m, int li, int n, const int *ids, const float *w,
                        const float *z, float *u, float *gate, float *up, float *hz){
    Slot *use[64]; int missk[64]; int nmiss=0;
    for(int j=0;j<n;j++){
        use[j]=slot_find(m,li,ids[j]);
        if(!use[j]){ m->miss++; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; }
    }
    if(nmiss){
        double t0=now_s();
        #pragma omp parallel for schedule(dynamic,1)
        for(int q=0;q<nmiss;q++) expert_read(m,li,ids[missk[q]],&m->ws[q]);
        m->t_eload+=now_s()-t0;
        m->ebytes+=(uint64_t)nmiss*(uint64_t)m->e_slot;
    }
    for(int j=0;j<n;j++)
        expert_apply(m,use[j],z,w[j],u,gate,up,hz);
    /* promotion: swap the freshly-read slots into the layer LRU */
    LCache *lc=&m->ecache[li];
    int promo = nmiss<lc->cap ? nmiss : lc->cap;
    for(int a=0;a<promo;a++){
        int q=nmiss-1-a; Slot *dst;
        if(lc->n<lc->cap) dst=&lc->s[lc->n++];
        else { int lru=0; for(int i=1;i<lc->n;i++) if(lc->s[i].used<lc->s[lru].used) lru=i; dst=&lc->s[lru]; }
        Slot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;
        dst->used=++m->clock;
    }
}

static void moe_forward(Model *m, Layer *l, int li, const float *x, float *out){
    Cfg *c=&m->c; Moe *o=&l->moe;
    int E=c->n_experts, K=c->topk, LT=c->latent, MI=c->moe_inter;
    float *sco=falloc(E);
    matmul(sco,x,o->router,1,c->hidden,E);
    for(int e=0;e<E;e++) sco[e]=sigmoidf_(sco[e]);
    int idx[64]; float wsel[64];
    for(int kk=0;kk<K;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){
            int taken=0; for(int j=0;j<kk;j++) if(idx[j]==e){taken=1;break;}
            float sv=sco[e]+o->rbias[e];
            if(!taken&&sv>bv){ bv=sv; best=e; }
        }
        idx[kk]=best; wsel[kk]=sco[best];             /* weight = RAW sigmoid score */
    }
    { float sm=0; for(int kk=0;kk<K;kk++) sm+=wsel[kk];
      for(int kk=0;kk<K;kk++) wsel[kk]/=(sm+1e-20f); }
    /* keep loads in DISK-OFFSET order (experts are NOT id-ordered inside the
     * HF shards — measured 169/895; helps the buffered fallback, harmless
     * under O_DIRECT). WILLNEED prefetch only for the buffered path. */
    for(int a2=0;a2<K-1;a2++) for(int b2=a2+1;b2<K;b2++){
        ERef *ea=&m->eref[(int64_t)li*E+idx[a2]], *eb=&m->eref[(int64_t)li*E+idx[b2]];
        if(eb->fd[0]<ea->fd[0]||(eb->fd[0]==ea->fd[0]&&eb->off[0]<ea->off[0])){
            int t=idx[a2];idx[a2]=idx[b2];idx[b2]=t; float tw=wsel[a2];wsel[a2]=wsel[b2];wsel[b2]=tw; }
    }
    if(!g_k3_direct)
        for(int kk=0;kk<K;kk++){
            ERef *er=&m->eref[(int64_t)li*E+idx[kk]];
            int64_t sizes[6]={m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s,m->e_w1p,m->e_w1s};
            if(er->contig){ if(er->fd[0]>=0) posix_fadvise(er->fd[0],er->off[0],m->e_slot,POSIX_FADV_WILLNEED); }
            else for(int k2=0;k2<6;k2++) if(er->fd[k2]>=0) posix_fadvise(er->fd[k2],er->off[k2],sizes[k2],POSIX_FADV_WILLNEED);
        }
    float *z=falloc(LT), *u=falloc(LT), *gate=falloc(MI), *up=falloc(MI), *hz=falloc(LT);
    w_matmul(z,x,&o->lat_down,1);
    memset(u,0,LT*sizeof(float));
    experts_run(m,li,K,idx,wsel,z,u,gate,up,hz);
    rmsnorm_(u,u,o->lat_norm,LT,c->eps);
    w_matmul(out,u,&o->lat_up,1);
    /* shared experts at full width */
    int shi=MI*c->n_shared;
    float *sg=falloc(shi), *su=falloc(shi), *sd=falloc(c->hidden);
    w_matmul(sg,x,&o->sh_gate,1); w_matmul(su,x,&o->sh_up,1);
    for(int i=0;i<shi;i++) sg[i]=situf_(sg[i],su[i],c->situ_b1,c->situ_b2);
    w_matmul(sd,sg,&o->sh_down,1);
    for(int d=0;d<c->hidden;d++) out[d]+=sd[d];
    free(sco);free(z);free(u);free(gate);free(up);free(hz);free(sg);free(su);free(sd);
}

static void dense_forward(Model *m, Layer *l, const float *x, float *out){
    Cfg *c=&m->c; int DI=c->dense_inter;
    float *g=falloc(DI), *u=falloc(DI);
    w_matmul(g,x,&l->d_gate,1); w_matmul(u,x,&l->d_up,1);
    for(int i=0;i<DI;i++) g[i]=situf_(g[i],u[i],c->situ_b1,c->situ_b2);
    w_matmul(out,g,&l->d_down,1);
    free(g);free(u);
}

/* ---------- one token through the stack; returns logits (or NULL pre-head) --- */
static float *g_x0=NULL; static int g_x0_n=0;  /* K3_X0: injected inputs (validation) */
static float *step(Model *m, int id, int pos){
    Cfg *c=&m->c; int D=c->hidden;
    int nbmax=(c->n_layers+c->res_bs-1)/c->res_bs;
    float *hidden=falloc(D), *bres=falloc((int64_t)nbmax*D), *prefix=falloc(D);
    float *nrm=falloc(D), *att=falloc(D), *mix=falloc(D), *mlp=falloc(D);
    int nb=0;
    if(g_x0){
        if(pos>=g_x0_n){ fprintf(stderr,"K3_X0: pos %d beyond %d injected rows\n",pos,g_x0_n); exit(1); }
        memcpy(hidden,g_x0+(int64_t)pos*D,D*sizeof(float));
    } else {
        char nm[512]; snprintf(nm,sizeof(nm),"%smodel.embed_tokens.weight",m->pfx);
        st_read_slice_f32(&m->S,nm,(int64_t)id*D,D,hidden,0);
    }
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        memcpy(prefix,hidden,D*sizeof(float));            /* prefix_sum at entry */
        int have_prefix=1;
        if(nb>0) res_mix(hidden,prefix,bres,nb,D,l->attn_sw,c->eps);
        if(i%c->res_bs==0){
            memcpy(bres+(int64_t)nb*D,prefix,D*sizeof(float)); nb++;
            have_prefix=0;
        }
        double t0=now_s();
        for(int d=0;d<D;d++) nrm[d]=hidden[d];
        rmsnorm_(nrm,nrm,l->in_ln,D,c->eps);
        if(l->kda) kda_forward(m,l,i,nrm,att);
        else       mla_forward(m,l,i,nrm,pos,att);
        m->t_attn+=now_s()-t0;
        if(have_prefix){ for(int d=0;d<D;d++) prefix[d]+=att[d]; }
        else           { memcpy(prefix,att,D*sizeof(float)); }
        res_mix(mix,prefix,bres,nb,D,l->mlp_sw,c->eps);
        rmsnorm_(nrm,mix,l->post_ln,D,c->eps);
        t0=now_s();
        if(l->sparse) moe_forward(m,l,i,nrm,mlp);
        else          dense_forward(m,l,nrm,mlp);
        m->t_moe+=now_s()-t0;
        for(int d=0;d<D;d++) prefix[d]+=mlp[d];
        memcpy(hidden,prefix,D*sizeof(float));
        if(m->trace) fwrite(hidden,sizeof(float),D,m->trace);
    }
    float *logits=NULL;
    if(m->has_head){
        double t0=now_s();
        res_mix(mix,hidden,bres,nb,D,m->out_sw,c->eps);
        rmsnorm_(mix,mix,m->final_norm,D,c->eps);
        if(m->trace) fwrite(mix,sizeof(float),D,m->trace);
        logits=falloc(c->vocab);
        w_matmul(logits,mix,&m->lm_head,1);
        m->t_head+=now_s()-t0;
    }
    free(hidden);free(bres);free(prefix);free(nrm);free(att);free(mix);free(mlp);
    return logits;
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c; m->max_t=max_t;
    m->Lc=calloc(c->n_layers,sizeof(float*));
    m->Rc=calloc(c->n_layers,sizeof(float*));
    for(int i=0;i<c->n_layers;i++) if(!m->L[i].kda){
        m->Lc[i]=falloc((int64_t)max_t*c->kv_lora);
        m->Rc[i]=falloc((int64_t)max_t*c->qk_rope);
    }
}

static int sample_tok(const float *lo, int V, float temp){
    if(temp<=0.f){ int b=0; for(int i=1;i<V;i++) if(lo[i]>lo[b]) b=i; return b; }
    float *p=falloc(V); float mx=lo[0];
    for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double s=0; for(int i=0;i<V;i++){ p[i]=expf((lo[i]-mx)/temp); s+=p[i]; }
    double r=((double)rand()/RAND_MAX)*s, acc=0; int pick=V-1;
    for(int i=0;i<V;i++){ acc+=p[i]; if(acc>=r){ pick=i; break; } }
    free(p); return pick;
}

int main(int argc, char **argv){
    if(argc<2){
        fprintf(stderr,"usage: %s <model_dir> [prompt] [--ids \"1 2 3\"] [--ngen N]\n",argv[0]);
        return 1;
    }
    const char *snap=argv[1], *prompt=NULL, *idstr=NULL;
    int ngen=32;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--ngen")&&i+1<argc) ngen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ids")&&i+1<argc) idstr=argv[++i];
        else if(!prompt) prompt=argv[i];
    }
    float temp=getenv("COLI_TEMP")?(float)atof(getenv("COLI_TEMP")):0.f;
    int nlayers=getenv("K3_LAYERS")?atoi(getenv("K3_LAYERS")):0;
    Model m;
    model_init(&m,snap,nlayers);
    if(getenv("K3_TRACE")){
        m.trace=fopen(getenv("K3_TRACE"),"wb");
        if(!m.trace){ perror(getenv("K3_TRACE")); return 1; }
    }
    if(getenv("K3_X0")){       /* injected input rows [T,hidden] f32, bypasses embed */
        FILE *f=fopen(getenv("K3_X0"),"rb");
        if(!f){ perror(getenv("K3_X0")); return 1; }
        fseek(f,0,SEEK_END); long fn=ftell(f); fseek(f,0,SEEK_SET);
        g_x0_n=(int)(fn/((long)m.c.hidden*4));
        g_x0=falloc((int64_t)g_x0_n*m.c.hidden);
        if(fread(g_x0,4,(size_t)g_x0_n*m.c.hidden,f)!=(size_t)g_x0_n*m.c.hidden){ fprintf(stderr,"K3_X0 short read\n"); return 1; }
        fclose(f);
        fprintf(stderr,"[K3] K3_X0: %d injected input rows\n",g_x0_n);
    }
    /* tokenize */
    int ids[65536], np=0;
    Tok T; int has_tok=0;
    { char tp[2048]; snprintf(tp,sizeof(tp),"%s/tokenizer.json",snap);
      FILE *f=fopen(tp,"rb"); if(f){ fclose(f); tok_load(&T,tp); has_tok=1;
          fprintf(stderr,"[K3] tokenizer.json loaded (family=%s)\n",T.kimi?"kimi":(T.o200k?"o200k":"cl100k")); } }
    if(idstr){
        const char *p=idstr;
        while(*p&&np<65536){ while(*p==' '||*p==',')p++; if(!*p)break; ids[np++]=(int)strtol(p,(char**)&p,10); }
    } else if(prompt){
        if(!has_tok){ fprintf(stderr,"no tokenizer.json — pass --ids (generate one with tools/k3_tokenizer.py)\n"); return 1; }
        if(m.c.bos>=0) ids[np++]=m.c.bos;
        np+=tok_encode(&T,prompt,(int)strlen(prompt),ids+np,65536-np);
    } else { fprintf(stderr,"no prompt and no --ids\n"); return 1; }
    fprintf(stderr,"[K3] prompt: %d tokens | ngen %d | temp %.2f\n",np,ngen,temp);
    int max_t=getenv("K3_MAXT")?atoi(getenv("K3_MAXT")):np+ngen;
    kv_alloc(&m,max_t);
    double t0=now_s(); float *lo=NULL;
    for(int i=0;i<np;i++){
        if(lo) free(lo);
        lo=step(&m,ids[i],i);
        fprintf(stderr,"\r[K3] prefill %d/%d (%.1fs)",i+1,np,now_s()-t0);
    }
    fprintf(stderr,"\n[K3] prefill done in %.1fs (%.2f tok/s)\n",now_s()-t0,np/(now_s()-t0));
    if(!m.has_head||!lo){
        fprintf(stderr,"[K3] no head — trace written, stopping after prefill\n");
        if(m.trace) fclose(m.trace);
        return 0;
    }
    double tg=now_s(); int ntok=0;
    char buf[512];
    for(int s=0;s<ngen;s++){
        int t=sample_tok(lo,m.c.vocab,temp);
        free(lo); lo=NULL;
        int is_eos=0; for(int e=0;e<m.c.n_eos;e++) if(t==m.c.eos[e]) is_eos=1;
        if(has_tok){ int n2=tok_decode(&T,&t,1,buf,sizeof(buf)-1); fwrite(buf,1,n2,stdout); fflush(stdout); }
        else { printf("%d ",t); fflush(stdout); }
        ntok++;
        if(is_eos){ fprintf(stderr,"\n[K3] eos\n"); break; }
        if(np+ntok>=max_t){ fprintf(stderr,"\n[K3] context full\n"); break; }
        lo=step(&m,t,np+ntok-1);
        double el=now_s()-tg;
        fprintf(stderr,"  [tok %d: %.1fs/tok, hit %.0f%%, %.1f GB read]\n",
                ntok,el/ntok,100.0*m.hits/(m.hits+m.miss+1e-9),m.ebytes/1e9);
    }
    if(lo) free(lo);
    double dt=now_s()-tg;
    fprintf(stderr,"\n[K3] decode %d tokens in %.1fs (%.2f tok/s) | expert hit %.1f%% (%llu/%llu) | %.1f GB streamed\n",
            ntok,dt,ntok/dt,100.0*m.hits/(m.hits+m.miss+1e-9),
            (unsigned long long)m.hits,(unsigned long long)(m.hits+m.miss),m.ebytes/1e9);
    fprintf(stderr,"[K3] time: attn %.1fs moe %.1fs (eload %.1fs) head %.1fs | RSS %.1f GB\n",
            m.t_attn,m.t_moe,m.t_eload,m.t_head,rss_gb());
    if(m.trace) fclose(m.trace);
    return 0;
}
