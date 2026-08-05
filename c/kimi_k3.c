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
 *   K3_VK=0|1            Vulkan tier (build with `make VK=1 kimi_k3`; default
 *                        1 when built): shared experts VRAM-resident + a
 *                        fill-once native-MXFP4 routed-expert tier; resident
 *                        experts skip disk AND CPU at decode. CPU fallback
 *                        everywhere; output identical.
 *   K3_VK_GB=N           VRAM cap for the tier (default: driver budget)
 *   K3_VK_UP=N           routed-expert uploads per step (default 8)
 *   K3_DIRECT=0|1        O_DIRECT expert reads (default 1; buffered fallback)
 *   K3_IDOT=0|1          int8-activation expert matmuls (default 1; 0 = float)
 *   K3_PIPE=0|1          overlap expert loads with compute (default 1)
 *   K3_LOAD_THREADS=N    loader threads for K3_PIPE (default 4)
 *   K3_TOPP=F            keep routed experts to cumulative weight F (0 = off)
 *   K3_CHUNK=N           prefill chunk size (default 32; 1 = token-at-a-time)
 *   K3_THINK=0|1         chat mode: open the think channel (default 1)
 *   K3_LAYERS=N          truncate to first N layers (validation; skips head)
 *   K3_TRACE=path        dump f32 hidden state after every layer (validation)
 *   K3_LOGITS=path       dump f32 logits per PREFILL position (teacher-forced
 *                        bit-width comparisons; use with --ngen 0)
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
#include <sys/select.h>
#include <unistd.h>
#endif
#include <pthread.h>
#include <stdatomic.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "st.h"
#include "tok.h"
#include "quant.h"
#include "omp_tune.h"
#include "route_trace.h"
#include "kv_prefix.h"                    /* KV prefix reuse (shared) */

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
typedef struct { int fmt; float *f; int8_t *q8; uint8_t *q4; float *s; int O, I, gs;
                 void *vk; /* ColiVkTensor* once uploaded (K3_VK); NULL = CPU only */ } W;

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
static char g_k3_usage[2100];   /* <snap>/.coli_usage, or COLI_USAGE */
typedef struct { int eid; uint8_t *buf, *base; uint64_t used; int pinned; } Slot;
/* pinned: seeded from .coli_usage at startup and never evicted. The LRU adapts to
 * THIS session; the pin knows the history of every session before it. Capped at
 * half the layer budget so the adaptive half always survives — an all-pinned cache
 * that guessed wrong is slower than no pin at all (measured on GLM: 0.17 vs 0.25
 * tok/s when the pins came from a single prompt). */
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
    /* KV prefix reuse: what the current state was built from (kv_prefix.h).
     * K3 has no single KV to inspect — 69 KDA layers carry a RECURRENT state
     * and only the 24 MLA layers keep Lc/Rc — so an explicit record of the
     * tokens fed is the only description of it that cannot drift. */
    kv_prefix kvp;
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
/* ---------- Vulkan tier (K3_VK, build with `make VK=1 kimi_k3`) ----------
 * Two residency classes on the card, both with transparent CPU fallback:
 *  - dense W tensors uploaded once at init (shared experts first): computed
 *    by the existing fmt=1/4 shaders whenever S==1 (w_matmul hook below);
 *  - a fill-once routed-expert tier in fmt=7 (native MXFP4, never re-encoded):
 *    experts enter from freshly-read RAM slots (K3_VK_UP per step) until the
 *    VRAM budget (K3_VK_GB) is reached; resident experts then skip BOTH the
 *    disk read and the CPU matmuls at decode (C==1). */
static int g_k3_vk=0;                     /* backend live (K3_VK=0 disables) */
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
typedef struct { void *w1, *w2, *w3; } VkExp;   /* ColiVkTensor* triple */
static VkExp *g_vkexp; static int64_t g_vkexp_n;
static int g_vk_upcap=8, g_vk_up_left=0, g_vk_full=0;
static long g_vk_hit=0, g_vk_res=0;
static double g_vk_gb=0;                  /* K3_VK_GB cap (0 = driver budget) */
static const char *k3_vk_spv(char *buf, size_t n){
    const char *env=getenv("COLI_VK_SHADERS");
    struct stat st;
    if(env&&*env){
        if(!stat(env,&st)&&S_ISDIR(st.st_mode)){ snprintf(buf,n,"%s/qmatmul.spv",env); return buf; }
        return env;
    }
#ifdef __linux__
    ssize_t k=readlink("/proc/self/exe",buf,n-1);
    if(k>0){
        buf[k]=0;
        char *sl=strrchr(buf,'/');
        if(sl&&(size_t)(sl+1-buf)+sizeof("shaders/qmatmul.spv")<=n){
            strcpy(sl+1,"shaders/qmatmul.spv");
            if(!stat(buf,&st)) return buf;
        }
    }
#endif
    return "shaders/qmatmul.spv";
}
static int vk_budget_full(void){
    double used=0,budget=0;
    if(!coli_vk_mem_budget(&used,&budget)) return 0;
    double cap=budget-0.5; if(g_vk_gb>0&&g_vk_gb<cap) cap=g_vk_gb;
    return used>=cap;
}
static int w_vk_upload(W *w){
    if(w->vk) return 1;
    int fmt = w->fmt==1?1 : w->fmt==4?4 : -1;
    if(fmt<0) return 0;
    return coli_vk_tensor_ensure((ColiVkTensor**)&w->vk,
        fmt==1?(const void*)w->q8:(const void*)w->q4,w->s,fmt,w->I,w->O,w->gs);
}
#endif
static void w_matmul(float *y, const float *x, const W *w, int S){
#ifdef COLI_VULKAN
    if(g_k3_vk&&S==1&&w->vk&&
       coli_vk_matmul((ColiVkTensor**)&((W*)w)->vk,y,x,NULL,NULL,w->fmt,1,w->I,w->O,w->gs))
        return;
#endif
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
static int g_k3_idot=1;                  /* K3_IDOT: int8-activation expert matmuls */
static int g_k3_pipe=1;                  /* K3_PIPE: overlap loads with compute */
static float g_k3_topp=0.f;              /* K3_TOPP: routed-expert top-p pruning */
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
    g_k3_idot  = getenv("K3_IDOT")?atoi(getenv("K3_IDOT")):1;
    g_k3_pipe  = getenv("K3_PIPE")?atoi(getenv("K3_PIPE")):1;
    g_k3_topp  = getenv("K3_TOPP")?(float)atof(getenv("K3_TOPP")):0.f;
    if(g_k3_topp>0.f)
        fprintf(stderr,"[K3] TOPP=%.2f: routed experts pruned to cumulative weight (quality lever — A/B with K3_LOGITS)\n",g_k3_topp);
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
#ifdef COLI_VULKAN
    { const char *ev=getenv("K3_VK");
      if(!ev||atoi(ev)){
        char sbuf[512]; const char *spv=k3_vk_spv(sbuf,sizeof(sbuf));
        g_k3_vk=coli_vk_init(spv);
        if(!g_k3_vk) fprintf(stderr,"[K3-VK] Vulkan unavailable (tried %s) — CPU only\n",spv);
      }
      if(g_k3_vk){
        g_vk_gb=getenv("K3_VK_GB")?atof(getenv("K3_VK_GB")):0;
        g_vk_upcap=getenv("K3_VK_UP")?atoi(getenv("K3_VK_UP")):8;
        g_vkexp_n=(int64_t)c->n_layers*c->n_experts;
        g_vkexp=calloc((size_t)g_vkexp_n,sizeof(VkExp));
        if(!g_vkexp) g_k3_vk=0;
      }
      if(g_k3_vk){
        /* dense residency: shared experts first — they run every token and are
         * the biggest always-on RAM-bandwidth slice that fits VRAM */
        int nsh=0;
        for(int i=0;i<c->n_layers&&!vk_budget_full();i++){
            if(!m->L[i].sparse) continue;
            Moe *sm2=&m->L[i].moe;
            nsh+=w_vk_upload(&sm2->sh_gate)+w_vk_upload(&sm2->sh_up)+w_vk_upload(&sm2->sh_down);
        }
        double used=0,budget=0; coli_vk_mem_budget(&used,&budget);
        fprintf(stderr,"[K3-VK] resident: %d shared-expert mats (%.1f/%.1f GB); routed MXFP4 tier fills at decode (K3_VK_UP=%d/step, cap %s)\n",
                nsh,used,budget,g_vk_upcap,g_vk_gb>0?"K3_VK_GB":"driver budget");
      }
    }
#endif
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
    { int ncl=c->n_layers>0?c->n_layers:1;
      m->ecache=calloc((size_t)ncl,sizeof(LCache)); }
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

/* ---------- KDA layer (chunk of C tokens; projections batched, recurrence
 * sequential per token, AVX2 on the state sweeps) ---------- */
static void kda_forward(Model *m, Layer *l, int li, const float *x, int C, float *out){
    Cfg *c=&m->c; Kda *a=&l->a;
    int P=c->kda_proj, H=c->kda_heads, hd=c->kda_hd, K=c->conv_k;
    float *q=falloc((int64_t)C*P), *k=falloc((int64_t)C*P), *v=falloc((int64_t)C*P);
    float *gp=falloc((int64_t)C*P), *on=falloc((int64_t)C*P);
    float *t1=falloc((int64_t)C*c->kda_hd), *graw=falloc((int64_t)C*P), *braw=falloc((int64_t)C*H);
    w_matmul(q,x,&a->q,C); w_matmul(k,x,&a->k,C); w_matmul(v,x,&a->v,C);
    w_matmul(gp,x,&a->g,C);
    matmul(t1,x,a->fa,C,c->hidden,c->kda_hd);
    matmul(graw,t1,a->fb,C,c->kda_hd,P);
    matmul(braw,x,a->bp,C,c->hidden,H);
    float qscale=1.f/sqrtf((float)hd);
    for(int t=0;t<C;t++){
        float *qt=q+(int64_t)t*P, *kt=k+(int64_t)t*P, *tv=v+(int64_t)t*P;
        float *gpt=gp+(int64_t)t*P, *ont=on+(int64_t)t*P;
        const float *rgt=graw+(int64_t)t*P, *bt=braw+(int64_t)t*H;
        /* depthwise causal conv (window: oldest..newest) + SiLU, rolls forward */
        float *wins[3]={m->cwq[li],m->cwk[li],m->cwv[li]};
        float *vecs[3]={qt,kt,tv}; float *taps[3]={a->conv_q,a->conv_k,a->conv_v};
        for(int w2=0;w2<3;w2++){
            float *win=wins[w2], *vec=vecs[w2]; const float *cw=taps[w2];
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
        #pragma omp parallel for schedule(static)
        for(int h=0;h<H;h++){
            const float *qh=qt+(int64_t)h*hd, *kh=kt+(int64_t)h*hd, *vh=tv+(int64_t)h*hd;
            float qn[512], kn[512], alpha[512], kS[512], vt[512], oh[512];
            float sq=0,sk=0;
            for(int i=0;i<hd;i++){ sq+=qh[i]*qh[i]; sk+=kh[i]*kh[i]; }
            sq=1.f/sqrtf(sq+1e-6f); sk=1.f/sqrtf(sk+1e-6f);
            for(int i=0;i<hd;i++){ qn[i]=qh[i]*sq*qscale; kn[i]=kh[i]*sk; }
            for(int i=0;i<hd;i++){
                float z=rgt[(int64_t)h*hd+i]+a->dt[(int64_t)h*hd+i];
                alpha[i]=expf(c->gate_lb*sigmoidf_(a->A[h]*z));
            }
            float beta=sigmoidf_(bt[h]);
            float *S=m->kstate[li]+(int64_t)h*hd*hd;
            memset(kS,0,sizeof(kS));
#ifdef __AVX2__
            if(!(hd&7)){
                for(int kk=0;kk<hd;kk++){
                    float *row=S+(int64_t)kk*hd;
                    __m256 al8=_mm256_set1_ps(alpha[kk]), kv8=_mm256_set1_ps(kn[kk]);
                    for(int vv=0;vv<hd;vv+=8){
                        __m256 r=_mm256_mul_ps(_mm256_loadu_ps(row+vv),al8);
                        _mm256_storeu_ps(row+vv,r);
                        _mm256_storeu_ps(kS+vv,_mm256_fmadd_ps(kv8,r,_mm256_loadu_ps(kS+vv)));
                    }
                }
                for(int vv=0;vv<hd;vv++) vt[vv]=(vh[vv]-kS[vv])*beta;
                memset(oh,0,sizeof(oh));
                for(int kk=0;kk<hd;kk++){
                    float *row=S+(int64_t)kk*hd;
                    __m256 kv8=_mm256_set1_ps(kn[kk]), qq8=_mm256_set1_ps(qn[kk]);
                    for(int vv=0;vv<hd;vv+=8){
                        __m256 r=_mm256_fmadd_ps(kv8,_mm256_loadu_ps(vt+vv),_mm256_loadu_ps(row+vv));
                        _mm256_storeu_ps(row+vv,r);
                        _mm256_storeu_ps(oh+vv,_mm256_fmadd_ps(qq8,r,_mm256_loadu_ps(oh+vv)));
                    }
                }
            } else {
#endif
            for(int kk=0;kk<hd;kk++){
                float *row=S+(int64_t)kk*hd; float al=alpha[kk], kv=kn[kk];
                for(int vv=0;vv<hd;vv++){ row[vv]*=al; kS[vv]+=kv*row[vv]; }
            }
            for(int vv=0;vv<hd;vv++) vt[vv]=(vh[vv]-kS[vv])*beta;
            memset(oh,0,sizeof(oh));
            for(int kk=0;kk<hd;kk++){
                float *row=S+(int64_t)kk*hd; float kv=kn[kk], qq=qn[kk];
                for(int vv=0;vv<hd;vv++){ row[vv]+=kv*vt[vv]; oh[vv]+=qq*row[vv]; }
            }
#ifdef __AVX2__
            }
#endif
            /* per-head RMSNorm * sigmoid(full-rank gate) */
            double ms=0; for(int vv=0;vv<hd;vv++) ms+=(double)oh[vv]*oh[vv];
            float r=1.f/sqrtf((float)(ms/hd)+c->eps);
            float *dst=ont+(int64_t)h*hd;
            for(int vv=0;vv<hd;vv++) dst[vv]=oh[vv]*r*a->onw[vv]*sigmoidf_(gpt[(int64_t)h*hd+vv]);
        }
    }
    w_matmul(out,on,&a->o,C);
    free(q);free(k);free(v);free(gp);free(on);free(t1);free(graw);free(braw);
}

/* ---------- gated MLA layer (chunk of C tokens, NoPE, absorb; projections
 * batched, per-token causal attention — token t attends to 0..pos0+t) ------ */
static void mla_forward(Model *m, Layer *l, int li, const float *x, int pos0, int C, float *out){
    Cfg *c=&m->c; Mla *a=&l->m;
    int H=c->n_heads, qh=c->qk_head, vh=c->v_head, kvl=c->kv_lora, qr=c->qk_rope;
    float *qa=falloc((int64_t)C*c->q_lora), *qv=falloc((int64_t)C*H*qh);
    float *ckv=falloc((int64_t)C*(kvl+qr));
    float *gv=falloc((int64_t)C*H*vh), *ctx=falloc((int64_t)C*H*vh);
    w_matmul(qa,x,&a->qa,C);
    for(int t=0;t<C;t++)
        rmsnorm_(qa+(int64_t)t*c->q_lora,qa+(int64_t)t*c->q_lora,a->qa_ln,c->q_lora,c->eps);
    w_matmul(qv,qa,&a->qb,C);
    w_matmul(ckv,x,&a->kva,C);
    for(int t=0;t<C;t++){                            /* append the whole chunk to the
                                                      * cache first: token t's scores
                                                      * only read rows 0..pos0+t */
        float *Lrow=m->Lc[li]+(int64_t)(pos0+t)*kvl, *Rrow=m->Rc[li]+(int64_t)(pos0+t)*qr;
        const float *cv=ckv+(int64_t)t*(kvl+qr);
        rmsnorm_(Lrow,cv,a->kva_ln,kvl,c->eps);
        memcpy(Rrow,cv+kvl,qr*sizeof(float));        /* NoPE: cached raw, no rotation */
    }
    w_matmul(gv,x,&a->g,C);
    for(int tt=0;tt<C;tt++){
        int nt=pos0+tt+1;
        const float *qvt=qv+(int64_t)tt*H*qh, *gvt=gv+(int64_t)tt*H*vh;
        float *ctxt=ctx+(int64_t)tt*H*vh;
        #pragma omp parallel for schedule(static)
        for(int h=0;h<H;h++){
            const float *qp=qvt+(int64_t)h*qh, *qrp=qp+c->qk_nope;
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
            float *cx=ctxt+(int64_t)h*vh;
            for(int d=0;d<vh;d++)
                cx[d]=w_rowdot(&a->kvb,rbase+c->qk_nope+d,clat)*sigmoidf_(gvt[(int64_t)h*vh+d]);
        }
    }
    w_matmul(out,ctx,&a->o,C);
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
    void (*mm)(float*,const float*,const uint8_t*,const uint8_t*,int,int,int)
        = g_k3_idot ? matmul_mxfp4_i8 : matmul_mxfp4;
    mm(gate,z,w1p,w1s,1,c->latent,c->moe_inter);
    mm(up,z,w3p,w3s,1,c->latent,c->moe_inter);
    for(int i=0;i<c->moe_inter;i++) gate[i]=situf_(gate[i],up[i],c->situ_b1,c->situ_b2);
    mm(hz,gate,w2p,w2s,1,c->moe_inter,c->latent);
    for(int i=0;i<c->latent;i++) u[i]+=wk*hz[i];
}

#ifdef COLI_VULKAN
/* GPU apply for a tier-resident expert (decode, S==1): w1/w3 in one paired
 * submit, SiTU-GLU on CPU, w2 down. Returns 0 untouched on any failure so
 * the caller can run the normal disk+CPU path. */
static int vk_expert_apply(Model *m, int li, int eid, const float *z, float wk,
                           float *u, float *gate, float *up, float *hz){
    Cfg *c=&m->c;
    VkExp *v=&g_vkexp[(int64_t)li*c->n_experts+eid];
    if(!v->w1||!v->w2||!v->w3) return 0;
    if(!coli_vk_matmul_pair((ColiVkTensor**)&v->w1,gate,NULL,NULL,c->moe_inter,
                            (ColiVkTensor**)&v->w3,up,NULL,NULL,c->moe_inter,
                            7,z,1,c->latent,32)) return 0;
    for(int i=0;i<c->moe_inter;i++) gate[i]=situf_(gate[i],up[i],c->situ_b1,c->situ_b2);
    if(!coli_vk_matmul((ColiVkTensor**)&v->w2,hz,gate,NULL,NULL,7,1,c->moe_inter,c->latent,32))
        return 0;
    for(int i=0;i<c->latent;i++) u[i]+=wk*hz[i];
    g_vk_hit++;
    return 1;
}
/* Fill the tier from a freshly-read RAM slot (main thread only). The ue8m0
 * exponents expand to f32 group scales at upload (the shader is float-only). */
static void vk_expert_try_upload(Model *m, int li, int eid, Slot *s){
    if(!g_k3_vk||g_vk_full||g_vk_up_left<=0) return;
    VkExp *v=&g_vkexp[(int64_t)li*m->c.n_experts+eid];
    if(v->w1) return;
    if(vk_budget_full()){ g_vk_full=1;
        fprintf(stderr,"[K3-VK] expert tier full: %ld experts resident\n",g_vk_res);
        return; }
    uint8_t *w1p=s->buf, *w1s=w1p+m->e_w1p, *w2p=w1s+m->e_w1s, *w2s=w2p+m->e_w2p,
            *w3p=w2s+m->e_w2s, *w3s=w3p+m->e_w1p;
    int LT=m->c.latent, MI=m->c.moe_inter;
    int64_t n1=m->e_w1s, n2=m->e_w2s;          /* scale counts = scale bytes (u8) */
    float *sc=falloc(n1>n2?n1:n2);
    int ok=1;
    for(int64_t i=0;i<n1;i++) sc[i]=mx4_scale(w1s[i]);
    ok=ok&&coli_vk_tensor_ensure((ColiVkTensor**)&v->w1,w1p,sc,7,LT,MI,32);
    if(ok){ for(int64_t i=0;i<n2;i++) sc[i]=mx4_scale(w2s[i]);
        ok=ok&&coli_vk_tensor_ensure((ColiVkTensor**)&v->w2,w2p,sc,7,MI,LT,32); }
    if(ok){ for(int64_t i=0;i<n1;i++) sc[i]=mx4_scale(w3s[i]);
        ok=ok&&coli_vk_tensor_ensure((ColiVkTensor**)&v->w3,w3p,sc,7,LT,MI,32); }
    free(sc);
    if(!ok){
        if(v->w1){ coli_vk_tensor_free(v->w1); v->w1=NULL; }
        if(v->w2){ coli_vk_tensor_free(v->w2); v->w2=NULL; }
        if(v->w3){ coli_vk_tensor_free(v->w3); v->w3=NULL; }
        g_vk_full=1;
        fprintf(stderr,"[K3-VK] expert tier full: %ld experts resident\n",g_vk_res);
        return;
    }
    g_vk_res++; g_vk_up_left--;
}
#endif

/* ---------- async loader pool (K3_PIPE): expert preads overlap compute ----
 * A batch of jobs is submitted per token+layer; the compute loop below waits
 * per-expert on its ready flag, so expert j's math runs while j+1.. load.
 * One batch in flight at a time (the submitter consumes every job before the
 * next submit), so the flags need no generation counter. */
#define LP_MAX 64
typedef struct { int li, eid; Slot *s; } LJob;
static struct {
    pthread_t th[16]; int nth, started;
    pthread_mutex_t mx; pthread_cond_t cv;
    Model *m;
    LJob job[LP_MAX];
    _Atomic int ready[LP_MAX];
    _Atomic int next; int count;
} g_lp = { .mx=PTHREAD_MUTEX_INITIALIZER, .cv=PTHREAD_COND_INITIALIZER };

static void *lp_main(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&g_lp.mx);
        while(atomic_load_explicit(&g_lp.next,memory_order_relaxed)>=g_lp.count)
            pthread_cond_wait(&g_lp.cv,&g_lp.mx);
        int idx=atomic_fetch_add_explicit(&g_lp.next,1,memory_order_relaxed);
        pthread_mutex_unlock(&g_lp.mx);
        if(idx>=g_lp.count) continue;
        LJob *j=&g_lp.job[idx];
        expert_read(g_lp.m,j->li,j->eid,j->s);
        atomic_store_explicit(&g_lp.ready[idx],1,memory_order_release);
    }
    return NULL;
}
static void lp_start(void){
    if(g_lp.started) return;
    g_lp.nth = getenv("K3_LOAD_THREADS")?atoi(getenv("K3_LOAD_THREADS")):4;
    if(g_lp.nth<1) g_lp.nth=1;
    if(g_lp.nth>16) g_lp.nth=16;
    g_lp.count=0; atomic_store(&g_lp.next,0);
    for(int i=0;i<g_lp.nth;i++)
        if(pthread_create(&g_lp.th[i],NULL,lp_main,NULL)){
            fprintf(stderr,"[K3] K3_PIPE: pthread_create failed, falling back\n");
            g_k3_pipe=0; g_lp.nth=i; break;
        }
    g_lp.started=1;
}
static void lp_submit(Model *m, int n){
    pthread_mutex_lock(&g_lp.mx);
    g_lp.m=m;
    for(int q=0;q<n;q++) atomic_store_explicit(&g_lp.ready[q],0,memory_order_relaxed);
    atomic_store_explicit(&g_lp.next,0,memory_order_relaxed);
    g_lp.count=n;
    pthread_cond_broadcast(&g_lp.cv);
    pthread_mutex_unlock(&g_lp.mx);
}

/* the general expert pass for one layer over a CHUNK of positions: nu unique
 * experts, expert j applied to pcnt[j] positions (poslist/wlist rows starting
 * at pfirst[j]). Loads run in blocks of <=LP_MAX working-set slots, pipelined
 * with compute under K3_PIPE (expert j's matmuls overlap expert j+1's read),
 * else all-parallel up front. Z/U are [C, stride] position-major. */
static void experts_apply_union(Model *m, int li, int nu, const int *uids,
                                const int *pfirst, const int *pcnt,
                                const int *poslist, const float *wlist,
                                const float *Z, int stride, float *U,
                                float *gate, float *up, float *hz){
    for(int base=0;base<nu;base+=LP_MAX){
        int nb=nu-base<LP_MAX?nu-base:LP_MAX;
        Slot *use[LP_MAX]; int missk[LP_MAX]; int qof[LP_MAX]; int nmiss=0;
        for(int j=0;j<nb;j++){
            use[j]=slot_find(m,li,uids[base+j]); qof[j]=-1;
            if(!use[j]){ m->miss++; use[j]=&m->ws[nmiss]; qof[j]=nmiss; missk[nmiss++]=j; }
        }
        if(nmiss){
            if(g_k3_pipe && !g_lp.started) lp_start();
            if(g_k3_pipe){
                for(int q=0;q<nmiss;q++){
                    g_lp.job[q].li=li; g_lp.job[q].eid=uids[base+missk[q]]; g_lp.job[q].s=&m->ws[q];
                }
                lp_submit(m,nmiss);
            } else {
                double t0=now_s();
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++) expert_read(m,li,uids[base+missk[q]],&m->ws[q]);
                m->t_eload+=now_s()-t0;
            }
            m->ebytes+=(uint64_t)nmiss*(uint64_t)m->e_slot;
        }
        for(int j=0;j<nb;j++){
            if(g_k3_pipe && qof[j]>=0 &&
               !atomic_load_explicit(&g_lp.ready[qof[j]],memory_order_acquire)){
                double t0=now_s();      /* t_eload = UN-hidden I/O (wait) time */
                while(!atomic_load_explicit(&g_lp.ready[qof[j]],memory_order_acquire))
                    usleep(50);
                m->t_eload+=now_s()-t0;
            }
#ifdef COLI_VULKAN
            if(g_k3_vk&&qof[j]>=0) vk_expert_try_upload(m,li,uids[base+j],use[j]);
#endif
            int f=pfirst[base+j];
            for(int p2=0;p2<pcnt[base+j];p2++){
                int t=poslist[f+p2];
                expert_apply(m,use[j],Z+(int64_t)t*stride,wlist[f+p2],
                             U+(int64_t)t*stride,gate,up,hz);
            }
        }
        /* promotion: swap the freshly-read slots into the layer LRU */
        LCache *lc=&m->ecache[li];
        int promo = nmiss<lc->cap ? nmiss : lc->cap;
        for(int a=0;a<promo;a++){
            int q=nmiss-1-a; Slot *dst;
            if(lc->n<lc->cap) dst=&lc->s[lc->n++];
            else {
                /* LRU over the UNPINNED slots. pin_seed caps pinned at cap/2, so a
                 * victim always exists; the -1 fallback is a belt-and-braces guard
                 * against a future caller pinning everything and deadlocking here. */
                int lru=-1;
                for(int i=0;i<lc->n;i++){
                    if(lc->s[i].pinned) continue;
                    if(lru<0 || lc->s[i].used<lc->s[lru].used) lru=i;
                }
                if(lru<0) break;                    /* every slot pinned: keep the read */
                dst=&lc->s[lru];
            }
            Slot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;
            dst->used=++m->clock;
        }
    }
}

/* ---- AUTOPIN: seed the layer caches from the accumulated history ----------
 *
 * K3 counted every routed selection (rt_route) and could READ a history file,
 * but only when COLI_USAGE named one by hand, and it never WROTE one — so the
 * read path was unreachable for anyone who did not already have a profile from
 * somewhere else. colibri.c has seeded itself from <snap>/.coli_usage since the
 * beginning and inkling.c does too; this brings K3 in line.
 *
 * The quota scales with CONFIDENCE in the history, exactly as colibri.c does it:
 * a handful of turns is a bad predictor and pinning on it steals slots from the
 * LRU, which at least adapts to the session in front of it. Below 5,000 recorded
 * selections nothing is pinned at all; the quota reaches its cap of half the
 * layer budget at 200,000, which is a few hours of real use.
 *
 * Cost is bounded and paid once: at most cap/2 experts per sparse layer, read
 * sequentially at startup rather than demand-faulted mid-token. */
static void k3_usage(const char *argv0){
    fprintf(stderr,
      "usage: %s <model_dir> [prompt] [options]\n"
      "\n"
      "  --chat              apply the K3 chat template to the prompt\n"
      "  --system TEXT       system message (implies --chat)\n"
      "  --ngen N            tokens to generate (default 32)\n"
      "  --ids \"1 2 3\"       raw token ids instead of a prompt\n"
      "                      (needed when the snapshot has no tokenizer.json;\n"
      "                       generate one with tools/k3_tokenizer.py)\n"
      "\n"
      "  %s /path/to/kimi \"What is the capital of France?\" --ngen 64\n"
      "\n"
      "For an interactive session use the launcher instead, which starts this\n"
      "engine for you and does not need the GLM engine built:\n"
      "  coli chat --model /path/to/kimi\n"
      "\n"
      "environment: SERVE=1 serve mode (SNAP=<dir>) - COLI_VULKAN=1 GPU path\n"
      "             OMP_NUM_THREADS=<physical cores> - AUTOPIN=0 no pin seeding\n",
      argv0, argv0);
}

static void pin_seed(Model *m, int64_t hist){
    Cfg *c=&m->c;
    if(getenv("AUTOPIN") && atoi(getenv("AUTOPIN"))==0) return;
    if(hist<5000) return;
    double conf=(double)hist/200000.0; if(conf>1) conf=1;
    int pinned_total=0;
    for(int li=0; li<c->n_layers; li++){
        if(!m->L[li].sparse) continue;
        LCache *lc=&m->ecache[li];
        int quota=(int)(lc->cap*0.5*conf);
        if(quota<1) continue;
        if(quota>lc->cap/2) quota=lc->cap/2;
        const uint32_t *u=rt_counts(li);
        if(!u) continue;
        /* top-`quota` by recorded use: a partial selection sort, since quota is
         * small (half a cache) and E is 896 — no allocation, no qsort callback. */
        for(int slot=0; slot<quota && lc->n<lc->cap; slot++){
            int best=-1; uint32_t bestc=0;
            for(int e=0;e<c->n_experts;e++){
                if(u[e]<=bestc) continue;
                int taken=0;
                for(int i=0;i<lc->n;i++) if(lc->s[i].eid==e){ taken=1; break; }
                if(!taken){ best=e; bestc=u[e]; }
            }
            if(best<0) break;                       /* history is exhausted */
            Slot *d=&lc->s[lc->n];
            expert_read(m,li,best,d);
            d->eid=best; d->used=++m->clock; d->pinned=1;
            lc->n++; pinned_total++;
        }
    }
    if(pinned_total)
        fprintf(stderr,"[PIN] %d experts pinned from history (%lld selections, %.0f%% confidence)\n",
                pinned_total,(long long)hist,100.0*conf);
}

static void moe_forward(Model *m, Layer *l, int li, const float *x, int C, float *out){
    Cfg *c=&m->c; Moe *o=&l->moe;
    int E=c->n_experts, K=c->topk, LT=c->latent, MI=c->moe_inter;
    float *sco=falloc((int64_t)C*E);
    matmul(sco,x,o->router,C,c->hidden,E);
    int *idxs=malloc((size_t)C*K*sizeof(int)); float *wsels=falloc((int64_t)C*K);
    int *keff=malloc((size_t)C*sizeof(int));
    if(!idxs||!keff){fprintf(stderr,"OOM moe sel\n");exit(1);}
    for(int t=0;t<C;t++){
        float *st=sco+(int64_t)t*E;
        for(int e=0;e<E;e++) st[e]=sigmoidf_(st[e]);
        int *idx=idxs+(int64_t)t*K; float *wsel=wsels+(int64_t)t*K;
        for(int kk=0;kk<K;kk++){
            int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){
                int taken=0; for(int j=0;j<kk;j++) if(idx[j]==e){taken=1;break;}
                float sv=st[e]+o->rbias[e];
                if(!taken&&sv>bv){ bv=sv; best=e; }
            }
            idx[kk]=best; wsel[kk]=st[best];          /* weight = RAW sigmoid score */
        }
        { float sm=0; for(int kk=0;kk<K;kk++) sm+=wsel[kk];
          for(int kk=0;kk<K;kk++) wsel[kk]/=(sm+1e-20f); }
        int Kt=K;
        /* K3_TOPP: drop the low-weight tail — the only lever that cuts expert
         * I/O AND compute proportionally. Weights renormalize over the kept
         * set (GLM's TOPP semantics). Quality-gate via K3_LOGITS. */
        if(g_k3_topp>0.f && g_k3_topp<1.f){
            for(int a2=1;a2<Kt;a2++){ int e=idx[a2]; float w2=wsel[a2]; int b2=a2-1;
                while(b2>=0&&wsel[b2]<w2){ idx[b2+1]=idx[b2]; wsel[b2+1]=wsel[b2]; b2--; }
                idx[b2+1]=e; wsel[b2+1]=w2; }
            float cum=0; int keep=Kt;
            for(int kk=0;kk<Kt;kk++){ cum+=wsel[kk]; if(cum>=g_k3_topp){ keep=kk+1; break; } }
            if(keep<Kt){
                float sm=0; for(int kk=0;kk<keep;kk++) sm+=wsel[kk];
                for(int kk=0;kk<keep;kk++) wsel[kk]/=(sm+1e-20f);
                Kt=keep;
            }
        }
        keff[t]=Kt;
        rt_route(li,t,idx,wsel,Kt);        /* traces and counts, one call */
    }
    float *z=falloc((int64_t)C*LT), *u=falloc((int64_t)C*LT);
    float *gate=falloc(MI), *up=falloc(MI), *hz=falloc(LT);
    w_matmul(z,x,&o->lat_down,C);
    memset(u,0,(size_t)C*LT*sizeof(float));
    /* union across the chunk: each unique expert loads ONCE and applies to
     * every position that selected it (position lists via counting sort).
     * With QB-flat routing the dedup is modest (~15% at C=32), but the loads
     * arrive as one deep burst for the NVMe and the dense side above/below
     * batches perfectly. */
    {
        int *map=malloc((size_t)E*sizeof(int));
        int *uid=malloc((size_t)C*K*sizeof(int));
        int *pcnt=malloc((size_t)C*K*sizeof(int)), *pfirst=malloc((size_t)C*K*sizeof(int));
        int *poslist=malloc((size_t)C*K*sizeof(int)); float *wlist=falloc((int64_t)C*K);
        int *cur=malloc((size_t)C*K*sizeof(int));
        if(!map||!uid||!pcnt||!pfirst||!poslist||!cur){fprintf(stderr,"OOM moe union\n");exit(1);}
        for(int e=0;e<E;e++) map[e]=-1;
        int nu=0;
        for(int t=0;t<C;t++) for(int kk=0;kk<keff[t];kk++){
            int e=idxs[(int64_t)t*K+kk];
            if(map[e]<0){ map[e]=nu; uid[nu]=e; pcnt[nu]=0; nu++; }
            pcnt[map[e]]++;
        }
        int acc=0;
        for(int j=0;j<nu;j++){ pfirst[j]=acc; cur[j]=acc; acc+=pcnt[j]; }
        for(int t=0;t<C;t++) for(int kk=0;kk<keff[t];kk++){
            int j=map[idxs[(int64_t)t*K+kk]];
            poslist[cur[j]]=t; wlist[cur[j]]=wsels[(int64_t)t*K+kk]; cur[j]++;
        }
        /* keep loads in DISK-OFFSET order (experts are NOT id-ordered inside
         * the HF shards — measured 169/895); permute the list heads with the
         * ids. WILLNEED prefetch only for the buffered path. */
        for(int a2=0;a2<nu-1;a2++) for(int b2=a2+1;b2<nu;b2++){
            ERef *ea=&m->eref[(int64_t)li*E+uid[a2]], *eb=&m->eref[(int64_t)li*E+uid[b2]];
            if(eb->fd[0]<ea->fd[0]||(eb->fd[0]==ea->fd[0]&&eb->off[0]<ea->off[0])){
                int tt=uid[a2];uid[a2]=uid[b2];uid[b2]=tt;
                tt=pcnt[a2];pcnt[a2]=pcnt[b2];pcnt[b2]=tt;
                tt=pfirst[a2];pfirst[a2]=pfirst[b2];pfirst[b2]=tt; }
        }
        if(!g_k3_direct)
            for(int j=0;j<nu;j++){
                ERef *er=&m->eref[(int64_t)li*E+uid[j]];
                int64_t sizes[6]={m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s,m->e_w1p,m->e_w1s};
                if(er->contig){ if(er->fd[0]>=0) posix_fadvise(er->fd[0],er->off[0],m->e_slot,POSIX_FADV_WILLNEED); }
                else for(int k2=0;k2<6;k2++) if(er->fd[k2]>=0) posix_fadvise(er->fd[k2],er->off[k2],sizes[k2],POSIX_FADV_WILLNEED);
            }
#ifdef COLI_VULKAN
        /* decode: tier-resident experts run on the GPU and drop out of the
         * disk union — that skip is the I/O saving. C>1 prefill stays on the
         * CPU-batched path (and still feeds the tier via the upload hook). */
        if(g_k3_vk&&C==1){
            int keep=0;
            for(int j=0;j<nu;j++){
                int handled=0;
                if(pcnt[j]==1){
                    int t=poslist[pfirst[j]];
                    handled=vk_expert_apply(m,li,uid[j],z+(int64_t)t*LT,
                                            wlist[pfirst[j]],u+(int64_t)t*LT,gate,up,hz);
                }
                if(!handled){ uid[keep]=uid[j]; pcnt[keep]=pcnt[j]; pfirst[keep]=pfirst[j]; keep++; }
            }
            nu=keep;
        }
#endif
        experts_apply_union(m,li,nu,uid,pfirst,pcnt,poslist,wlist,z,LT,u,gate,up,hz);
        free(map);free(uid);free(pcnt);free(pfirst);free(poslist);free(wlist);free(cur);
    }
    for(int t=0;t<C;t++)
        rmsnorm_(u+(int64_t)t*LT,u+(int64_t)t*LT,o->lat_norm,LT,c->eps);
    w_matmul(out,u,&o->lat_up,C);
    /* shared experts at full width */
    int shi=MI*c->n_shared;
    float *sg=falloc((int64_t)C*shi), *su=falloc((int64_t)C*shi), *sd=falloc((int64_t)C*c->hidden);
    w_matmul(sg,x,&o->sh_gate,C); w_matmul(su,x,&o->sh_up,C);
    for(int64_t i=0;i<(int64_t)C*shi;i++) sg[i]=situf_(sg[i],su[i],c->situ_b1,c->situ_b2);
    w_matmul(sd,sg,&o->sh_down,C);
    for(int64_t d=0;d<(int64_t)C*c->hidden;d++) out[d]+=sd[d];
    free(sco);free(idxs);free(wsels);free(keff);
    free(z);free(u);free(gate);free(up);free(hz);free(sg);free(su);free(sd);
}

static void dense_forward(Model *m, Layer *l, const float *x, int C, float *out){
    Cfg *c=&m->c; int DI=c->dense_inter;
    float *g=falloc((int64_t)C*DI), *u=falloc((int64_t)C*DI);
    w_matmul(g,x,&l->d_gate,C); w_matmul(u,x,&l->d_up,C);
    for(int64_t i=0;i<(int64_t)C*DI;i++) g[i]=situf_(g[i],u[i],c->situ_b1,c->situ_b2);
    w_matmul(out,g,&l->d_down,C);
    free(g);free(u);
}

/* ---------- a CHUNK of C tokens through the stack, layer-major: every dense
 * matmul batches over the chunk (weights stream from RAM once per chunk), the
 * MoE loads each unique expert once. Sequential state (KDA recurrence, MLA
 * cache, AttnRes bookkeeping) advances per token inside each layer, which is
 * exactly the original order — chunked results are bit-identical to C=1.
 * Returns the LAST position's logits (falloc'd), or NULL pre-head. ---------- */
static float *g_x0=NULL; static int g_x0_n=0;  /* K3_X0: injected inputs (validation) */
static FILE *g_lfp=NULL;                       /* K3_LOGITS: per-position logit dump */
static float *step_chunk(Model *m, const int *ids, int pos0, int C){
    Cfg *c=&m->c; int D=c->hidden;
#ifdef COLI_VULKAN
    g_vk_up_left=g_vk_upcap;              /* routed-tier upload budget per step */
#endif
    int nbmax=(c->n_layers+c->res_bs-1)/c->res_bs;
    float *hidden=falloc((int64_t)C*D), *bres=falloc((int64_t)C*nbmax*D);
    float *prefix=falloc((int64_t)C*D), *nrm=falloc((int64_t)C*D);
    float *att=falloc((int64_t)C*D), *mix=falloc(D), *mlp=falloc((int64_t)C*D);
    int nb=0;
    for(int t=0;t<C;t++){
        if(g_x0){
            if(pos0+t>=g_x0_n){ fprintf(stderr,"K3_X0: pos %d beyond %d injected rows\n",pos0+t,g_x0_n); exit(1); }
            memcpy(hidden+(int64_t)t*D,g_x0+(int64_t)(pos0+t)*D,D*sizeof(float));
        } else {
            char nm[512]; snprintf(nm,sizeof(nm),"%smodel.embed_tokens.weight",m->pfx);
            st_read_slice_f32(&m->S,nm,(int64_t)ids[t]*D,D,hidden+(int64_t)t*D,0);
        }
    }
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        int snap=(i%c->res_bs==0);                    /* block boundary: same for all t */
        for(int t=0;t<C;t++){
            float *h=hidden+(int64_t)t*D, *p=prefix+(int64_t)t*D;
            memcpy(p,h,D*sizeof(float));              /* prefix_sum at entry */
            if(nb>0) res_mix(h,p,bres+(int64_t)t*nbmax*D,nb,D,l->attn_sw,c->eps);
            if(snap) memcpy(bres+(int64_t)t*nbmax*D+(int64_t)nb*D,p,D*sizeof(float));
            rmsnorm_(nrm+(int64_t)t*D,h,l->in_ln,D,c->eps);
        }
        int have_prefix=!snap;
        if(snap) nb++;
        double t0=now_s();
        if(l->kda) kda_forward(m,l,i,nrm,C,att);
        else       mla_forward(m,l,i,nrm,pos0,C,att);
        m->t_attn+=now_s()-t0;
        for(int t=0;t<C;t++){
            float *p=prefix+(int64_t)t*D, *a=att+(int64_t)t*D;
            if(have_prefix){ for(int d=0;d<D;d++) p[d]+=a[d]; }
            else           { memcpy(p,a,D*sizeof(float)); }
            res_mix(mix,p,bres+(int64_t)t*nbmax*D,nb,D,l->mlp_sw,c->eps);
            rmsnorm_(nrm+(int64_t)t*D,mix,l->post_ln,D,c->eps);
        }
        t0=now_s();
        if(l->sparse) moe_forward(m,l,i,nrm,C,mlp);
        else          dense_forward(m,l,nrm,C,mlp);
        m->t_moe+=now_s()-t0;
        for(int t=0;t<C;t++){
            float *p=prefix+(int64_t)t*D;
            for(int d=0;d<D;d++) p[d]+=mlp[(int64_t)t*D+d];
            memcpy(hidden+(int64_t)t*D,p,D*sizeof(float));
            if(m->trace) fwrite(hidden+(int64_t)t*D,sizeof(float),D,m->trace);
        }
    }
    float *logits=NULL;
    if(m->has_head){
        double t0=now_s();
        for(int t=0;t<C;t++){
            /* head only where needed: the chunk's last token (feeds sampling)
             * and every position when K3_LOGITS dumps teacher-forced logits */
            if(!g_lfp && t<C-1) continue;
            res_mix(mix,hidden+(int64_t)t*D,bres+(int64_t)t*nbmax*D,nb,D,m->out_sw,c->eps);
            rmsnorm_(mix,mix,m->final_norm,D,c->eps);
            if(m->trace) fwrite(mix,sizeof(float),D,m->trace);
            float *lo=falloc(c->vocab);
            w_matmul(lo,mix,&m->lm_head,1);
            if(g_lfp) fwrite(lo,sizeof(float),(size_t)c->vocab,g_lfp);
            if(t==C-1) logits=lo; else free(lo);
        }
        m->t_head+=now_s()-t0;
    }
    /* record what was just fed, at the positions it went to (kv_prefix.h) */
    kv_prefix_record(&m->kvp, ids, pos0, C);
    free(hidden);free(bres);free(prefix);free(nrm);free(att);free(mix);free(mlp);
    return logits;
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    /* Serve calls this once per request, and it used to calloc Lc/Rc over the
     * old pointers without freeing them: n_layers x max_t x (kv_lora +
     * qk_rope) floats leaked every turn, hundreds of MB over a conversation on
     * the 24 MLA layers at 4k context.
     *
     * GROW, DO NOT RESTART. Freeing and re-allocating also discards every
     * position already computed, which defeats KV prefix reuse in the one case
     * it exists for: a conversation whose prompt is longer every turn asks for
     * a larger max_t every turn, so the state would be thrown away immediately
     * before the point of using it. (Caught by CI on the Inkling side, where
     * the same shape of bug sat in the same place.)
     *
     * Lc/Rc are laid out [position][kv_lora] and [position][qk_rope], so unlike
     * inkling's head-major K/V a grow is a straight prefix copy — no re-layout.
     * The 69 KDA layers are untouched here: their recurrent state does not
     * scale with max_t and survives on its own. */
    if(m->Lc && max_t<=m->max_t) return;

    float **oldL=m->Lc, **oldR=m->Rc;
    int keep=(m->Lc && m->kvp.len>0 && m->kvp.len<=max_t) ? m->kvp.len : 0;

    m->max_t=max_t;
    m->Lc=calloc(c->n_layers,sizeof(float*));
    m->Rc=calloc(c->n_layers,sizeof(float*));
    for(int i=0;i<c->n_layers;i++) if(!m->L[i].kda){
        m->Lc[i]=falloc((int64_t)max_t*c->kv_lora);
        m->Rc[i]=falloc((int64_t)max_t*c->qk_rope);
        if(keep){
            memcpy(m->Lc[i], oldL[i], (size_t)keep*c->kv_lora*sizeof(float));
            memcpy(m->Rc[i], oldR[i], (size_t)keep*c->qk_rope*sizeof(float));
        }
    }
    if(oldL) for(int i=0;i<c->n_layers;i++){ free(oldL[i]); free(oldR[i]); }
    free(oldL); free(oldR);

    /* the record describes those same positions, so it survives with them */
    if(!kv_prefix_grow(&m->kvp,max_t,keep)) kv_prefix_clear(&m->kvp);
}

typedef struct { float p; int id; } SampleProb;
static int sample_prob_desc(const void *a,const void *b){
    float d=((const SampleProb*)b)->p-((const SampleProb*)a)->p;
    return d>0?1:d<0?-1:0;
}
static int sample_tok(const float *lo, int V, float temp, float top_p){
    if(temp<=0.f){ int b=0; for(int i=1;i<V;i++) if(lo[i]>lo[b]) b=i; return b; }
    SampleProb *rank=malloc((size_t)V*sizeof(SampleProb)); float mx=lo[0];
    if(!rank){ fprintf(stderr,"OOM sampling\n"); exit(1); }
    for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double sum=0;
    for(int i=0;i<V;i++){ float p=expf((lo[i]-mx)/temp); sum+=p; rank[i]=(SampleProb){p,i}; }
    qsort(rank,(size_t)V,sizeof(SampleProb),sample_prob_desc);
    double cut=(top_p>0.f&&top_p<1.f)?top_p*sum:sum, kept=0; int n=0;
    while(n<V&&kept<cut) kept+=rank[n++].p;
    double r=((double)rand()/RAND_MAX)*kept, acc=0; int pick=rank[0].id;
    for(int i=0;i<n;i++){ acc+=rank[i].p; if(acc>=r){ pick=rank[i].id; break; } }
    free(rank); return pick;
}

/* ---------- K3 XTML chat format (faithful to the shipped encoding_k3.py) --
 * Only <|open|>, <|close|>, <|sep|>, <|end_of_msg|> are special TOKENS; tag
 * names and attributes are ordinary text, encoded as the same standalone
 * segments as the reference (segment boundaries are token boundaries). A
 * turn renders as
 *   <|open|>message role="user"<|sep|>TEXT<|close|>message<|sep|><|end_of_msg|>
 * and the generation prompt opens the assistant message plus its structural
 * thinking channel:
 *   <|open|>message role="assistant"<|sep|><|open|>think<|sep|>
 * (K3_THINK=0 opens <response> directly = non-thinking mode). The model then
 * closes think, opens response, and finishes with <|end_of_msg|> (the eos). */
typedef struct { Tok *T; int *ids; int n, cap;
                 int sp_open, sp_close, sp_sep, sp_eom; } ChatB;
static void cb_special(ChatB *b, int id){
    if(b->n>=b->cap){ fprintf(stderr,"chat prompt too long\n"); exit(1); }
    b->ids[b->n++]=id;
}
static void cb_text(ChatB *b, const char *s){
    if(!*s) return;
    b->n+=tok_encode(b->T,s,(int)strlen(s),b->ids+b->n,b->cap-b->n);
}
static void cb_open(ChatB *b, const char *tag, const char *role){
    cb_special(b,b->sp_open); cb_text(b,tag);
    if(role){ cb_text(b," role"); cb_text(b,"=\""); cb_text(b,role); cb_text(b,"\""); }
    cb_special(b,b->sp_sep);
}
static void cb_close(ChatB *b, const char *tag){
    cb_special(b,b->sp_close); cb_text(b,tag); cb_special(b,b->sp_sep);
}
static int chat_special(Tok *T, const char *s){
    int l=(int)strlen(s);
    for(int i=0;i<T->nsp;i++)
        if(T->sp[i].len==l && !memcmp(T->sp[i].str,s,l)) return T->sp[i].id;
    return -1;
}
/* returns prompt length; sp[4] = {open, close, sep, end_of_msg} ids */
static int chat_build(Tok *T, const char *sys, const char *user, int thinking,
                      int *ids, int cap, int *sp){
    ChatB b={T,ids,0,cap,
        chat_special(T,"<|open|>"), chat_special(T,"<|close|>"),
        chat_special(T,"<|sep|>"),  chat_special(T,"<|end_of_msg|>")};
    if(b.sp_open<0||b.sp_close<0||b.sp_sep<0||b.sp_eom<0){
        fprintf(stderr,"chat: XTML special tokens not in tokenizer.json\n"); exit(1); }
    sp[0]=b.sp_open; sp[1]=b.sp_close; sp[2]=b.sp_sep; sp[3]=b.sp_eom;
    if(sys&&*sys){
        cb_open(&b,"message","system"); cb_text(&b,sys);
        cb_close(&b,"message"); cb_special(&b,b.sp_eom);
    }
    cb_open(&b,"message","user"); cb_text(&b,user);
    cb_close(&b,"message"); cb_special(&b,b.sp_eom);
    cb_open(&b,"message","assistant");
    cb_open(&b,thinking?"think":"response",NULL);
    return b.n;
}

static void chat_message(ChatB *b, const char *role, const char *text, int assistant){
    cb_open(b,"message",role);
    if(assistant) cb_open(b,"response",NULL);
    cb_text(b,text);
    if(assistant) cb_close(b,"response");
    cb_close(b,"message");
    cb_special(b,b->sp_eom);
}

static void chat_assistant(ChatB *b, const char *reasoning, const char *text){
    cb_open(b,"message","assistant");
    cb_open(b,"think",NULL); cb_text(b,reasoning); cb_close(b,"think");
    cb_open(b,"response",NULL); cb_text(b,text); cb_close(b,"response");
    cb_close(b,"message"); cb_special(b,b->sp_eom);
}

/* Internal gateway payload. Length framing keeps arbitrary UTF-8/newlines in
 * message content while preserving the segment boundaries required by K3's
 * rank-BPE chat template:
 *   K3CHAT1\n
 *   M <role> <utf8-bytes>\n<content> ...
 *   G <thinking>\n
 */
static int chat_build_wire(Tok *T, const char *wire, int nwire, int *thinking,
                           int *ids, int cap, int *sp){
    ChatB b={T,ids,0,cap,
        chat_special(T,"<|open|>"), chat_special(T,"<|close|>"),
        chat_special(T,"<|sep|>"),  chat_special(T,"<|end_of_msg|>")};
    /* -2, not -1: the caller must be able to tell a bad payload from a snapshot
     * whose tokenizer has no XTML tokens. Serve reported both as "invalid K3
     * chat payload", which sent at least one user hunting through a request
     * body that was perfectly well formed. The CLI path has always named the
     * real cause; serve now does too. */
    if(b.sp_open<0||b.sp_close<0||b.sp_sep<0||b.sp_eom<0) return -2;
    sp[0]=b.sp_open; sp[1]=b.sp_close; sp[2]=b.sp_sep; sp[3]=b.sp_eom;
    const char *p=wire, *end=wire+nwire;
    if(nwire<8||memcmp(p,"K3CHAT1\n",8)) return -1;
    p+=8; *thinking=0;
    while(p<end){
        const char *nl=memchr(p,'\n',(size_t)(end-p));
        if(!nl) return -1;
        if(*p=='G'){
            int v=0;
            if(sscanf(p,"G %d",&v)!=1) return -1;
            *thinking=!!v; p=nl+1; break;
        }
        if(*p=='A'){
            int nr=-1, nt=-1;
            if(sscanf(p,"A %d %d",&nr,&nt)!=2||nr<0||nt<0||nl+1+nr+nt>end) return -1;
            char *reason=malloc((size_t)nr+1), *text=malloc((size_t)nt+1);
            if(!reason||!text){ fprintf(stderr,"OOM chat assistant\n"); exit(1); }
            memcpy(reason,nl+1,(size_t)nr); reason[nr]=0;
            memcpy(text,nl+1+nr,(size_t)nt); text[nt]=0;
            chat_assistant(&b,reason,text);
            free(reason); free(text); p=nl+1+nr+nt; continue;
        }
        char role[16]; int nb=-1;
        if(sscanf(p,"M %15s %d",role,&nb)!=2||nb<0||nl+1+nb>end) return -1;
        char *text=malloc((size_t)nb+1);
        if(!text){ fprintf(stderr,"OOM chat message\n"); exit(1); }
        memcpy(text,nl+1,(size_t)nb); text[nb]=0;
        const char *r=!strcmp(role,"developer")?"system":role;
        if(strcmp(r,"system")&&strcmp(r,"user")&&strcmp(r,"assistant")){ free(text); return -1; }
        chat_message(&b,r,text,!strcmp(r,"assistant"));
        free(text); p=nl+1+nb;
    }
    cb_open(&b,"message","assistant");
    cb_open(&b,*thinking?"think":"response",NULL);
    return b.n;
}

/* ---------- serve mode: shared openai_server.py protocol ---------- */
typedef struct {
    char id[64];
    int max_tok;
    float temp, top_p;
    char *payload;
    int plen;
} ServeReq;

static void model_state_reset(Model *m){
    Cfg *c=&m->c;
    kv_prefix_clear(&m->kvp);   /* the record describes the state we are dropping */
    for(int i=0;i<c->n_layers;i++){
        if(m->L[i].kda){
            memset(m->kstate[i],0,(size_t)c->kda_heads*c->kda_hd*c->kda_hd*sizeof(float));
            memset(m->cwq[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
            memset(m->cwk[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
            memset(m->cwv[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
        }
        if(m->Lc&&m->Lc[i]) free(m->Lc[i]);
        if(m->Rc&&m->Rc[i]) free(m->Rc[i]);
    }
    free(m->Lc); free(m->Rc);
    m->Lc=NULL; m->Rc=NULL; m->max_t=0;
}

static int serve_stdin_readable(void){
    /* Windows non ha fd_set/select in questa forma: la build falliva del tutto.
     * La versione portabile (con i fix #139/#195) vive in compat.h, incluso via st.h. */
    return coli_stdin_readable();
}

static int serve_read_req(ServeReq *q, const char *active){
    char line[512], cmd[16], id[64];
    if(!fgets(line,sizeof(line),stdin)) return -1;
    if(sscanf(line,"%15s %63s",cmd,id)<2) return 0;
    if(!strcmp(cmd,"CANCEL")||!strcmp(cmd,"STOP")) return active&&!strcmp(active,id);
    if(strcmp(cmd,"SUBMIT")) return 0;
    int slot, plen, max_tok; float temp, top_p;
    if(sscanf(line,"%*s %*s %d %d %d %f %f",&slot,&plen,&max_tok,&temp,&top_p)!=5||
       plen<0||plen>(1<<24)||max_tok<1){
        printf("ERROR %s bad submit header\n",id); fflush(stdout); return 0;
    }
    (void)slot;
    char *payload=malloc((size_t)plen+1);
    if(!payload){ printf("ERROR %s out of memory\n",id); fflush(stdout); return 0; }
    if(fread(payload,1,(size_t)plen,stdin)!=(size_t)plen){ free(payload); return -1; }
    (void)fgetc(stdin); payload[plen]=0;
    snprintf(q->id,sizeof(q->id),"%s",id);
    q->max_tok=max_tok; q->temp=temp; q->top_p=top_p;
    q->payload=payload; q->plen=plen;
    return 2;
}

static void serve_data(const char *id, const char *p, int n){
    if(n<=0) return;
    printf("DATA %s %d\n",id,n);
    fwrite(p,1,(size_t)n,stdout); fputc('\n',stdout); fflush(stdout);
}

static void serve_one(Model *m, Tok *T, ServeReq *q){
    int cap=65536, *ids=malloc((size_t)cap*sizeof(int)), np=0;
    if(!ids){ printf("ERROR %s out of memory\n",q->id); fflush(stdout); return; }
    int sp[4]={-1,-1,-1,-1}, chat=0, thinking=0;
    if(m->c.bos>=0) ids[np++]=m->c.bos;
    if(q->plen>=8&&!memcmp(q->payload,"K3CHAT1\n",8)){
        int n=chat_build_wire(T,q->payload,q->plen,&thinking,ids+np,cap-np,sp);
        if(n==-2){
            /* Not the request's fault: this snapshot's tokenizer.json has no
             * <|open|>/<|close|>/<|sep|>/<|end_of_msg|>, so no chat turn can be
             * built from it. Say that, and say where a usable one comes from. */
            printf("ERROR %s tokenizer.json has no XTML chat tokens "
                   "(<|open|> <|close|> <|sep|> <|end_of_msg|>); "
                   "regenerate it with tools/k3_tokenizer.py\n",q->id);
            fflush(stdout);
            fprintf(stderr,"[K3] chat: XTML special tokens not in tokenizer.json — "
                           "regenerate with tools/k3_tokenizer.py\n");
            free(ids); return; }
        if(n<0){ printf("ERROR %s invalid K3 chat payload\n",q->id); fflush(stdout); free(ids); return; }
        np+=n; chat=1;
    } else {
        np+=tok_encode(T,q->payload,q->plen,ids+np,cap-np);
    }
    int max_ctx=getenv("K3_MAXT")?atoi(getenv("K3_MAXT")):8192;
    if(np<1||np+q->max_tok>max_ctx){
        printf("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",
               q->id,np,q->max_tok,max_ctx);
        fflush(stdout); free(ids); return;
    }
    printf("ACCEPT %s %d\n",q->id,np); fflush(stdout);
    /* KV PREFIX REUSE (#639 for GLM; this engine re-prefilled every turn).
     * A chat client resends the whole transcript each turn, so turn N used to
     * re-process turns 1..N-1 from scratch — the cost of a message grew with
     * the conversation, and every replayed position pulled its experts off
     * disk again. When this prompt begins with the sequence the state already
     * holds, that state IS the state at that position: keep it and prefill
     * only the tail. This is why kv_alloc above must run FIRST and must keep
     * its buffers: growing them discards the positions fed[] describes.
     * At least one new token is required, since the state cannot be rewound.
     * Either the reused positions are token-identical or nothing is reused;
     * the emitted tokens are unchanged in both cases. */
    kv_alloc(m,np+q->max_tok+8);
    int reuse=kv_prefix_reuse(&m->kvp, ids, np);
    if(getenv("K3_PREFIX_LOG")){
        /* Report the decision either way, with the state behind a "no".
         * "It did not get faster" is otherwise the same observation as
         * "reuse is not wired up" — for a user as much as for a test. */
        if(reuse)
            fprintf(stderr,"[PREFIX] reusing %d of %d prompt tokens (%.0f%%)\n",
                    reuse,np,100.0*reuse/np);
        else
            fprintf(stderr,"[PREFIX] no reuse: held=%d cap=%d prompt=%d%s\n",
                    m->kvp.len,m->kvp.cap,np,
                    (m->kvp.len>0 && m->kvp.len<np) ? " (diverged)" : "");
        fflush(stderr);
    }
    if(!reuse) model_state_reset(m);
    int chunk=getenv("K3_CHUNK")?atoi(getenv("K3_CHUNK")):32;
    if(chunk<1) chunk=1; if(chunk>512) chunk=512;
    double t0=now_s(), a0=m->t_attn, e0=m->t_moe, d0=m->t_eload, h0=m->t_head;
    uint64_t hit0=m->hits, miss0=m->miss;
    float *lo=NULL;
    /* `i` is the ABSOLUTE position: attention and the MLA Lc/Rc slots are
     * position-indexed, so the loop starts at `reuse`, not at 0. */
    for(int i=reuse;i<np;i+=chunk){
        int C=np-i<chunk?np-i:chunk;
        free(lo); lo=step_chunk(m,ids+i,i,C);
    }
    int gen=0, limited=1, cancelled=0, xsup=0, xopen=0, xtl=0;
    char buf[512], xtag[64];
    double tg=now_s();
    for(int s=0;s<q->max_tok&&!cancelled;s++){
        int tk=sample_tok(lo,m->c.vocab,q->temp,q->top_p);
        free(lo); lo=NULL;
        int eos=0; for(int i=0;i<m->c.n_eos;i++) if(tk==m->c.eos[i]) eos=1;
        int show=!eos;
        if(chat&&sp[0]>=0){
            if(tk==sp[0]||tk==sp[1]){
                xsup=1; xopen=(tk==sp[0]); xtl=0; show=0;
            } else if(tk==sp[2]){
                if(xsup){
                    xsup=0; xtag[xtl]=0;
                    if(xopen&&!strcmp(xtag,"response")&&thinking)
                        serve_data(q->id,"</think>",8);
                }
                show=0;
            } else if(xsup){
                int nb=tok_decode(T,&tk,1,buf,sizeof(buf)-1);
                if(xtl+nb<(int)sizeof(xtag)){ memcpy(xtag+xtl,buf,(size_t)nb); xtl+=nb; }
                show=0;
            } else if(tk==sp[3]) show=0;
        }
        if(show){
            int nb=tok_decode(T,&tk,1,buf,sizeof(buf)-1);
            serve_data(q->id,buf,nb);
        }
        if(!eos) gen++;
        while(serve_stdin_readable()){
            ServeReq queued={0};
            int r=serve_read_req(&queued,q->id);
            if(r<0){ cancelled=1; break; }
            if(r==1) cancelled=1;
            if(r==2){
                printf("ERROR %s engine busy\n",queued.id); fflush(stdout); free(queued.payload);
            }
        }
        if(cancelled){ limited=0; break; }
        if(eos){ limited=0; break; }
        if(s+1<q->max_tok) lo=step_chunk(m,&tk,np+s,1);
    }
    free(lo); free(ids);
    double dt=now_s()-t0, decode=now_s()-tg;
    uint64_t hits=m->hits-hit0, misses=m->miss-miss0, total=hits+misses;
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n",q->id,gen,
           decode>0?gen/decode:0.0,total?100.0*hits/total:0.0,rss_gb(),np,limited);
    double moe=m->t_moe-e0, disk=m->t_eload-d0;
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %d\n",
           dt,np,gen,disk,0.0,moe>disk?moe-disk:moe,m->t_attn-a0,m->t_head-h0,gen+1);
    fflush(stdout);
#ifdef COLI_VULKAN
    if(g_k3_vk) fprintf(stderr,"[K3-VK] routed tier: %ld resident, %ld GPU hits so far\n",
                        g_vk_res,g_vk_hit);
#endif
}

static void serve_loop(Model *m, Tok *T){
    /* PRIMA del sentinella: su Windows stdout in modalita' TEXT trasforma il \n
     * finale in \r\n, il gateway non lo riconosce e resta in attesa per sempre
     * (#748). Vive in compat.h perche' colibri.c ce l'ha da #195 e questo motore
     * e' nato senza. */
    coli_serve_binary_mode();
    setvbuf(stdin,NULL,_IONBF,0);
    fputs("\x01\x01READY\x01\x01\n",stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n",rss_gb());
    fflush(stdout);
    for(;;){
        ServeReq q={0}; int r;
        do r=serve_read_req(&q,NULL); while(r==0);
        if(r<0) return;
        if(r==2){ serve_one(m,T,&q); free(q.payload); }
    }
}

int main(int argc, char **argv){
    coli_omp_tune_threads("kimi_k3");   /* squadra sui core fisici, niente spin-wait: vedi omp_tune.h */
    int serving=getenv("SERVE")&&getenv("SERVE")[0]=='1';
    /* Usage was printed only when there were NO arguments, so `--help` fell
     * through as the model directory and the engine went looking for
     * "--help/config.json". Nobody should have to read the source to find the
     * argument order. Print it for the help flags too, and from every refusal
     * below, so an error says what to do instead of only what went wrong. */
    if(!serving && (argc<2 || !strcmp(argv[1],"--help") || !strcmp(argv[1],"-h")
                          || !strcmp(argv[1],"help"))){
        k3_usage(argv[0]);
        return argc<2 ? 1 : 0;          /* asking for help is not a failure */
    }
    const char *snap=serving?getenv("SNAP"):argv[1], *prompt=NULL, *idstr=NULL, *sysmsg=NULL, *wirepath=NULL;
    if(!snap||!*snap){ fprintf(stderr,"set SNAP=<Kimi K3 snapshot directory>\n"); return 1; }
    int ngen=32, chat=0;
    for(int i=serving?1:2;i<argc;i++){
        if(!strcmp(argv[i],"--ngen")&&i+1<argc) ngen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ids")&&i+1<argc) idstr=argv[++i];
        else if(!strcmp(argv[i],"--chat")) chat=1;
        else if(!strcmp(argv[i],"--system")&&i+1<argc) sysmsg=argv[++i];
        else if(!strcmp(argv[i],"--wire-test")&&i+1<argc) wirepath=argv[++i];
        else if(!prompt) prompt=argv[i];
    }
    if(wirepath){
        char tp[2048]; snprintf(tp,sizeof(tp),"%s/tokenizer.json",snap);
        Tok wt; tok_load(&wt,tp);
        FILE *wf=fopen(wirepath,"rb"); if(!wf){ perror(wirepath); return 1; }
        fseek(wf,0,SEEK_END); long wn=ftell(wf); fseek(wf,0,SEEK_SET);
        if(wn<0||wn>(1<<24)){ fprintf(stderr,"wire payload too large\n"); fclose(wf); return 1; }
        char *wire=malloc((size_t)wn+1); int *wid=malloc(65536*sizeof(int));
        if(!wire||!wid){ fprintf(stderr,"OOM wire test\n"); return 1; }
        if(fread(wire,1,(size_t)wn,wf)!=(size_t)wn){ fprintf(stderr,"short wire read\n"); return 1; }
        fclose(wf); wire[wn]=0;
        int thinking=0, wsp[4], n=chat_build_wire(&wt,wire,(int)wn,&thinking,wid,65536,wsp);
        if(n<0){ fprintf(stderr,"invalid K3 chat wire payload\n"); return 1; }
        for(int i=0;i<n;i++) printf("%s%d",i?" ":"",wid[i]);
        printf("\n"); free(wire); free(wid); return 0;
    }
    float temp=getenv("COLI_TEMP")?(float)atof(getenv("COLI_TEMP")):0.f;
    int nlayers=getenv("K3_LAYERS")?atoi(getenv("K3_LAYERS")):0;
    Model m;
    model_init(&m,snap,nlayers);
    rt_init("kimi_k3",m.c.n_layers,m.c.n_experts);   /* counters, identity, ROUTE_TRACE */
    /* A layer with no counter row is a layer that cannot be credited: dense layers do not
     * route, and K3 has no MTP row. Without this a history record naming one of them is
     * silently absorbed and written back out. See docs/routing-telemetry.md. */
    for(int i=0;i<m.c.n_layers;i++) if(!m.L[i].sparse) rt_drop_row(i);
    rt_drop_row(m.c.n_layers);
    /* LEARNED CACHE. Expert use accumulates in <snap>/.coli_usage across sessions
     * and seeds the pins at startup, the same convention colibri.c and inkling.c
     * already follow. COLI_USAGE overrides the location; USAGE_SAVE=0 makes the
     * run read-only, for benchmark loops that would otherwise skew the profile
     * they are measuring. */
    { const char *up=getenv("COLI_USAGE");
      if(up&&*up) snprintf(g_k3_usage,sizeof(g_k3_usage),"%s",up);
      else        snprintf(g_k3_usage,sizeof(g_k3_usage),"%s/.coli_usage",snap);
      int64_t h=rt_load(g_k3_usage);
      if(h>0) fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",
                      (long long)h,g_k3_usage);
      /* #780: without a half-life the ranking freezes — after ~18M recorded
       * selections one more turn moves it by 0.2% and the profile stops
       * following the workload. */
      if(getenv("COLI_USAGE_DECAY")) rt_decay();
      pin_seed(&m,h); }
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
    if(serving){
        if(!has_tok){ fprintf(stderr,"serve mode needs tokenizer.json\n"); return 1; }
        serve_loop(&m,&T);
        return 0;
    }
    int sp[4]={-1,-1,-1,-1};
    int think=getenv("K3_THINK")?atoi(getenv("K3_THINK")):1;
    if(idstr){
        const char *p=idstr;
        while(*p&&np<65536){ while(*p==' '||*p==',')p++; if(!*p)break; ids[np++]=(int)strtol(p,(char**)&p,10); }
    } else if(chat){
        if(!has_tok){ fprintf(stderr,"--chat needs tokenizer.json\n"); return 1; }
        if(!prompt){ fprintf(stderr,"--chat needs a user message\n"); return 1; }
        if(m.c.bos>=0) ids[np++]=m.c.bos;
        np+=chat_build(&T,sysmsg,prompt,think,ids+np,65536-np,sp);
        if(getenv("K3_CHAT_IDS")){
            fprintf(stderr,"[K3] chat ids:");
            for(int i=0;i<np;i++) fprintf(stderr," %d",ids[i]);
            fprintf(stderr,"\n");
        }
    } else if(prompt){
        if(!has_tok){ fprintf(stderr,"no tokenizer.json in the snapshot — pass --ids instead\n\n");
                      k3_usage(argv[0]); return 1; }
        if(m.c.bos>=0) ids[np++]=m.c.bos;
        np+=tok_encode(&T,prompt,(int)strlen(prompt),ids+np,65536-np);
    } else { fprintf(stderr,"no prompt and no --ids: nothing to generate from\n\n");
             k3_usage(argv[0]); return 1; }
    fprintf(stderr,"[K3] prompt: %d tokens | ngen %d | temp %.2f\n",np,ngen,temp);
    int max_t=getenv("K3_MAXT")?atoi(getenv("K3_MAXT")):np+ngen;
    kv_alloc(&m,max_t);
    if(getenv("K3_LOGITS")){
        g_lfp=fopen(getenv("K3_LOGITS"),"wb");
        if(!g_lfp){ perror(getenv("K3_LOGITS")); return 1; }
    }
    int chunk=getenv("K3_CHUNK")?atoi(getenv("K3_CHUNK")):32;
    if(chunk<1) chunk=1;
    if(chunk>512) chunk=512;
    if(m.trace && chunk>1){
        chunk=1;                       /* trace rows are token-major by contract */
        fprintf(stderr,"[K3] K3_TRACE set: prefill chunk forced to 1\n");
    }
    double t0=now_s(); float *lo=NULL;
    for(int i=0;i<np;i+=chunk){
        int Cc=np-i<chunk?np-i:chunk;
        if(lo) free(lo);
        lo=step_chunk(&m,ids+i,i,Cc);
        fprintf(stderr,"\r[K3] prefill %d/%d (%.1fs)",i+Cc,np,now_s()-t0);
    }
    if(g_lfp){ fclose(g_lfp); g_lfp=NULL; }
    fprintf(stderr,"\n[K3] prefill done in %.1fs (%.2f tok/s)\n",now_s()-t0,np/(now_s()-t0));
    if(!m.has_head||!lo){
        fprintf(stderr,"[K3] no head — trace written, stopping after prefill\n");
        if(m.trace) fclose(m.trace);
        return 0;
    }
    double tg=now_s(); int ntok=0;
    char buf[512];
    /* chat print filter: hide the XTML structure, label the channels.
     * Structural runs are <|open|>/<|close|> TAGTEXT <|sep|> — suppress them
     * and print a channel banner when the response channel opens. */
    int xsup=0, xopen=0; char xtag[64]; int xtl=0;
    if(chat&&think){ printf("[think] "); fflush(stdout); }
    for(int s=0;s<ngen;s++){
        int t=sample_tok(lo,m.c.vocab,temp,1.f);
        free(lo); lo=NULL;
        int is_eos=0; for(int e=0;e<m.c.n_eos;e++) if(t==m.c.eos[e]) is_eos=1;
        int show=1;
        if(chat&&sp[0]>=0){
            if(t==sp[0]||t==sp[1]){ xsup=1; xopen=(t==sp[0]); xtl=0; show=0; }
            else if(t==sp[2]){
                if(xsup){ xsup=0; xtag[xtl]=0;
                    if(xopen&&!strcmp(xtag,"response")){ printf("\n\n[response] "); fflush(stdout); }
                }
                show=0;
            } else if(xsup){
                if(has_tok){ int n2=tok_decode(&T,&t,1,buf,sizeof(buf)-1);
                    if(xtl+n2<(int)sizeof(xtag)){ memcpy(xtag+xtl,buf,n2); xtl+=n2; } }
                show=0;
            } else if(t==sp[3]) show=0;
        }
        if(show){
            if(has_tok){ int n2=tok_decode(&T,&t,1,buf,sizeof(buf)-1); fwrite(buf,1,n2,stdout); fflush(stdout); }
            else { printf("%d ",t); fflush(stdout); }
        }
        ntok++;
        if(is_eos){ fprintf(stderr,"\n[K3] eos\n"); break; }
        if(np+ntok>=max_t){ fprintf(stderr,"\n[K3] context full\n"); break; }
        lo=step_chunk(&m,&t,np+ntok-1,1);
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
    { const char *sv=getenv("USAGE_SAVE");
      if(!(sv && atoi(sv)==0) && g_k3_usage[0])
          rt_save(g_k3_usage,0); }                   /* same bytes as every other engine */
    return 0;
}
