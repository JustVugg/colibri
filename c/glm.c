/* Motore GLM-5.2 (architettura glm_moe_dsa) in C puro.
 * Stadio B: replica fedele del forward di transformers (modeling_glm_moe_dsa.py):
 *   - attenzione MLA (q/kv-LoRA, RoPE interleaved parziale)
 *   - router sigmoid + noaux_tc (n_group=1) con routed_scaling_factor
 *   - shared expert + expert routed in streaming dal disco (per-expert)
 *   - primi first_k_dense_replace layer densi
 * Il DSA indexer e' un NO-OP per seq <= index_topk (seleziona tutte le key): qui si usa
 * attenzione causale densa -> output identico all'oracolo su prompt corti.
 *
 * QUANTIZZAZIONE: gli expert (streaming) e la parte DENSA residente (attenzione, lm_head,
 * embed, mlp densa, shared expert) sono tenuti in int8 per-riga + scala (dequant-on-use).
 * E' cio' che fa entrare GLM-5.2 nei 15 GB: ~17B param residenti a int4 ~= 8.7 GB.
 * Norme/router/bias restano f32 (piccoli e sensibili).
 *
 * Validazione: stessi token id di ref_glm.json (oracolo transformers, c/tools/make_glm_oracle.py).
 *   build: make glm   run: SNAP=./glm_tiny ./glm <cap> <expert_bits> <dense_bits>
 *   TF=1 -> teacher-forcing (valida il prefill su tutta la sequenza)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>                              /* thread I/O del PILOTA */
#include <stdatomic.h>                            /* PIPE ready-flags/job queue + PILOT_REAL cross-layer handshake */
#include <sched.h>                                /* sched_yield: PIPE spin / PILOT barrier */
#include <unistd.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/select.h>                             /* select() serve-loop polling (#68); not on native MinGW */
#endif
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#include <sys/mman.h>                             /* mlock: inchioda le pagine in RAM / wire pages into RAM */
#include <sys/stat.h>                             /* fstat per mmap degli shard (COLI_MMAP) */
#endif
#include "st.h"
#include "tok.h"
#include "tier.h"
#include "grammar.h"                              /* metodo F: draft grammaticali (#48) */
#include "decode_batch.h"
#ifdef _OPENMP
#include <omp.h>                                  /* scratch per-thread nell'attention */
#else
static inline int omp_get_max_threads(void){ return 1; }
static inline int omp_get_thread_num(void){ return 0; }
#endif
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif
#ifdef COLI_METAL
#include "backend_metal.h"
#include <omp.h>
static int g_metal_enabled;
static int g_metal_gemm_min=16;   /* COLI_METAL_GEMM_MIN: min rows to send a matmul_qt GEMM to GPU */
/* routing precalcolata dalla GPU (layer CB): moe() la usa e salta la FASE A */
static const int *g_pre_idx; static const float *g_pre_w; static const int *g_pre_keff;
static const float *g_pre_sh;   /* output dello shared expert gia' calcolato su GPU */
#endif
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){            /* somma orizzontale di 8 float */
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>                             /* Apple Silicon / aarch64: kernel NEON */
#elif defined(__VSX__)
#include <altivec.h>                              /* POWER8+ (ppc64le): kernel VSX */
#undef vector                                     /* igiene: si usano __vector/__bool espliciti */
#undef pixel
#undef bool
#endif
#ifdef __APPLE__
#include <mach/mach.h>                            /* host_statistics64: MemAvailable di macOS */
#endif

typedef struct {
    int hidden, n_layers, n_heads, n_experts, topk, moe_inter, dense_inter;
    int first_dense, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head, n_shared, vocab;
    int n_group, topk_group, norm_topk;
    int stop_ids[8], n_stop;                     /* eos_token_id dal config (GLM-5.2 ne ha 3!) */
    int index_topk, index_nh, index_hd;          /* DSA lightning indexer */
    int8_t idx_type[128];                        /* per layer: 1=full (calcola), 0=shared (riusa) */
    float eps, theta, attn_scale, routed_scale;
} Cfg;

/* tensore [O,I] in uno di tre formati:
 *   fmt=0 F32   -> qf
 *   fmt=1 INT8  -> q8 (1 byte/param) + scala per riga
 *   fmt=2 INT4  -> q4 (2 valori per byte, impacchettati) + scala per riga
 * INT4 e' cio' che fa stare la densa residente nei 15 GB (0.5 byte/param). */
/* fmt: 0 F32, 1 INT8, 2 INT4 (2/byte), 3 INT2 (4/byte). q4 ospita sia int4 che int2 packed. */
typedef struct {
    int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I;
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
    int cuda_eligible, cuda_failed, cuda_device;  /* resident tensor, never a reused expert slot */
} QT;
static int64_t qt_bytes(const QT *t){    /* byte residenti del tensore */
    int64_t n=(int64_t)t->O*t->I;
    if(t->fmt==0) return n*4;
    if(t->fmt==1) return n + (int64_t)t->O*4;
    if(t->fmt==3) return (int64_t)t->O*((t->I+3)/4) + (int64_t)t->O*4;
    return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*4;
}

typedef struct {
    float *in_ln, *post_ln;
    /* MLA (densa, quantizzata) */
    QT q_a, q_b, kv_a, kv_b, o; float *q_a_ln, *kv_a_ln;
    int sparse;
    /* dense mlp (sparse==0) */
    QT gate_proj, up_proj, down_proj;
    /* moe (sparse==1) */
    float *router, *router_bias;                 /* router f32 (sensibile) */
    QT sh_gate, sh_up, sh_down;                  /* shared expert */
} Layer;

/* slot di un expert: pesi quantizzati + scale. Nel container pre-quantizzato g/u/d sono
 * VISTE dentro `slab` (una sola pread coalescente); nel fallback hanno buffer propri.
 * slab_cap/fslab_cap: capienza allocata — gli slot ws[] sono riusati TRA layer e gli
 * expert non hanno tutti la stessa taglia (layer MTP int8 = 2x i layer int4). */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used; } ESlot;

typedef struct {
    float **Lc, **Rc, **Ic;
    int *kv_start, max_t;
    int disk_nrec;
    char disk_path[2048];
} KVState;

typedef struct {
    KVState *kv;
    int token, pos;
} DecodeRow;

typedef struct {
    Cfg c; shards S;
    int ebits, dbits;                            /* bit expert / bit densa */
    QT embed, lm_head; float *final_norm;
    Layer *L;
    /* KV-cache MLA COMPRESSA: per token si tiene solo il latente normato [kv_lora] e
     * k_rot [qk_rope] (576 vs 32768 valori/token). k_nope e value si ricostruiscono al
     * volo con kv_b. E' cio' che rende gestibile il contesto su 15 GB (64 teste, no GQA). */
    float **Lc, **Rc; int max_t;                 /* alias della KVState attiva */
    int *kv_start;                               /* prima pos valida nella KV del layer (MTP: parziale) */
    KVState *kv;
    ESlot **ecache; int *ecn; int ecap;          /* LRU expert per-layer */
    ESlot ws[64];                                /* working set del layer corrente (load paralleli) */
    ESlot **pin; int *npin;                      /* HOT-STORE: expert pinnati in RAM (mai evicted) */
    uint32_t **eusage;                           /* contatori persistenti (per STATS/PIN) */
    uint32_t **eheat;                            /* calore recente per promotion/demotion live */
    uint32_t **elast, eaccess_clock;              /* recency per LFRU session-local */
    /* DSA lightning indexer (attivo solo se i pesi out-idx-* sono presenti) */
    int has_dsa;
    QT *ix_wq, *ix_wk, *ix_wp;                   /* per layer FULL: wq_b, wk, weights_proj */
    float **ix_knw, **ix_knb;                    /* k_norm (LayerNorm, eps 1e-6) */
    float **Ic;                                  /* alias KVState: cache indexer [max_t*hd] */
    int *dsa_sel, *dsa_nsel; int dsa_scap;       /* selezione per posizione del batch corrente */
    /* testa MTP (layer n_layers, stile DeepSeek-V3): draft nativi ad alta acceptance */
    int has_mtp; Layer mtpL; QT eh_proj;
    float *enorm, *hnorm, *mtp_norm;
    float *hlast, *h_all;                        /* hidden pre-norm: ultima pos / tutte le pos batch */
    uint64_t mtp_prop, mtp_acc;                  /* statistica acceptance */
    int **eroute; int *enr;                      /* metodo C: routing dell'ULTIMO token per layer */
    uint64_t eclock, hits, miss, ereq;
    uint64_t gpu_expert_calls; int gpu_expert_count; int64_t gpu_expert_bytes;
    uint64_t n_fw, n_emit;                       /* metodo E: forward di decode / token emessi */
    double t_edisk, t_ewait, t_emm, t_attn, t_kvb, t_head;/* profiling: dove va il tempo */
    double t_aproj,t_acore,t_aout;                     /* attention breakdown */
    int64_t resident_bytes;
} Model;

static void usage_save(Model *m);
static void tiers_emit(Model *m);
static void ehit_mark(Model *m, int layer, int eid);
static void emap_emit(Model *m);
static void hits_emit(Model *m);
static void hwinfo_emit(Model *m);        /* cache che impara: definita accanto a stats_dump */
#ifdef COLI_CUDA
static int g_cuda_enabled;
static double g_cuda_expert_gb;
static int g_cuda_dense;
static int g_cuda_release_host;
static int g_cuda_devices[COLI_CUDA_MAX_DEVICES], g_cuda_ndev, g_cuda_rr;
static int64_t g_cuda_dense_projected[COLI_CUDA_MAX_DEVICES];
static void qt_cuda_reset(QT *t){
    if(t->cuda){ coli_cuda_tensor_free(t->cuda); t->cuda=NULL; }
    t->cuda_failed=0;
}
static int qt_cuda_upload(QT *t){
    const void *weights = t->fmt==0 ? (const void*)t->qf
                        : t->fmt==1 ? (const void*)t->q8 : (const void*)t->q4;
    return coli_cuda_tensor_upload(&t->cuda,weights,t->s,t->fmt,t->I,t->O,t->cuda_device);
}
static void cuda_stats_print(void){
    size_t n=0,b=0; coli_cuda_stats(-1,&n,&b);
    fprintf(stderr,"[CUDA] resident set: %zu tensors, %.2f GB VRAM\n",n,b/1e9);
    if(g_cuda_ndev>1) for(int i=0;i<g_cuda_ndev;i++){
        coli_cuda_stats(g_cuda_devices[i],&n,&b);
        fprintf(stderr,"[CUDA]   device %d: %zu tensors, %.2f GB\n",g_cuda_devices[i],n,b/1e9);
    }
    uint64_t calls=0,experts=0,rows=0; double h2d=0,kernel=0,d2h=0;
    coli_cuda_group_stats(&calls,&experts,&rows,&h2d,&kernel,&d2h);
    if(calls) fprintf(stderr,"[CUDA] expert groups: %llu call, %llu expert, %llu righe "
        "(%.2f expert/call)%s\n",(unsigned long long)calls,(unsigned long long)experts,
        (unsigned long long)rows,(double)experts/calls,
        getenv("COLI_CUDA_PROFILE")?"; timing sotto":"");
    if(calls&&getenv("COLI_CUDA_PROFILE")) fprintf(stderr,
        "[CUDA] expert groups timing: H2D %.1f ms | kernel %.1f ms | D2H %.1f ms\n",h2d,kernel,d2h);
}
static int parse_cuda_devices(const char *list, int *out){
    if(!list||!*list) return 0;
    int n=0; const char *p=list;
    while(*p){
        char *end=NULL; long v=strtol(p,&end,10);
        if(end==p||v<0||v>INT_MAX||n>=COLI_CUDA_MAX_DEVICES) return 0;
        for(int i=0;i<n;i++) if(out[i]==(int)v) return 0;
        out[n++]=(int)v; p=end;
        while(*p==' '||*p=='\t') p++;
        if(!*p) break;
        if(*p++!=',') return 0;
        while(*p==' '||*p=='\t') p++;
        if(!*p) return 0;
    }
    return n;
}
#endif
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);   /* macOS: ru_maxrss in BYTE */
#else
    return r.ru_maxrss/(1024.0*1024.0);          /* Linux: in KB */
#endif
}
static float *falloc(int64_t n){
    /* guardia anti-wrap (report PR #25): n assurdo da file modello ostili non deve
     * diventare una malloc piccola. Niente calloc: il memset nel percorso caldo costa. */
    if(n<0 || (uint64_t)n > SIZE_MAX/sizeof(float)){ fprintf(stderr,"falloc: n=%lld is out of range\n",(long long)n); exit(1); }
    float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }

/* ---- Accumulatore int4->float a 512 bit / 512-bit int4->float accumulator ----
 * Stessa matematica lossless di matmul_i4 (nibble->f32, FMA), ma 32 pesi/iter su
 * due catene FMA indipendenti. NON bit-identico al vecchio ordine: la riduzione
 * ad albero accumula MENO errore della somma sequenziale (misurato 2-4x più
 * vicino all'oracolo double sulle forme reali; perplexity invariata, +4-7% sul
 * decode con routing CPU-heavy — vedi docs/experiments/glm52-6x5090-2026-07-12.md).
 * EN: same lossless math as matmul_i4, 32 weights/iter on two independent FMA
 * chains. Not bit-identical to the old order: tree reduction accumulates LESS
 * rounding than sequential summation. I4_ACC512=0 restores the old order (A/B). */
#if defined(__AVX512F__) && defined(__AVX512BW__)
static int g_i4_acc512=1;
static inline float dot_i4f_avx512(const uint8_t *w,const float *x,int I){
    const __m128i m4=_mm_set1_epi8(0x0F); const __m512i b8=_mm512_set1_epi32(8);
    __m512 acc0=_mm512_setzero_ps(),acc1=_mm512_setzero_ps(); int i=0;
    for(;i+32<=I;i+=32){ __m128i by=_mm_loadu_si128((const __m128i*)(w+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi),n1=_mm_unpackhi_epi8(lo,hi);
        __m512 w0=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n0),b8));
        __m512 w1=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n1),b8));
        acc0=_mm512_fmadd_ps(_mm512_loadu_ps(x+i),w0,acc0);
        acc1=_mm512_fmadd_ps(_mm512_loadu_ps(x+i+16),w1,acc1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0,acc1));
}
/* selftest contro il riferimento scalare (I4_ACC512_TEST=1): copre l'ordine dei
 * nibble e ogni multiplo di 32. / selftest vs the scalar reference. */
static int i4_acc512_selftest(void){
    enum { N=224 }; uint8_t w[(N+1)/2]; float x[N];
    for(int i=0;i<N;i++){
        int q=((i*13+5)&15)-8;
        if(!(i&1)) w[i>>1]=(uint8_t)(q+8);
        else w[i>>1]|=(uint8_t)((q+8)<<4);
        x[i]=(float)(((i*29+7)%101)-50)/37.f;
    }
    for(int n=32;n<=N;n+=32){
        float ref=0; for(int i=0;i<n;i++) ref+=x[i]*(float)(((w[i>>1]>>((i&1)*4))&15)-8);
        float got=dot_i4f_avx512(w,x,n),tol=2e-5f*(1.f+fabsf(ref));
        if(fabsf(got-ref)>tol){ fprintf(stderr,"AVX512 i4 selftest n=%d: %.9g != %.9g\n",n,got,ref); return 0; }
    }
    return 1;
}
#endif

/* y[S,O] = x[S,I] @ W^T, W[O,I] f32 */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}
/* y[S,O] = x[S,I] @ W^T con W quantizzato int8 per-riga + scala[O] (dequant-on-use) */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            __m256 acc=_mm256_setzero_ps();
            for(;i+8<=I;i+=8){ __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+i)));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), _mm256_cvtepi32_ps(wi), acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+8<=I;i+=8){ int16x8_t w16=vmovl_s8(vld1_s8(w+i));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++) a+=xs[i]*(float)w[i]; y[(int64_t)s*O+o]=a*sc; } }
}
/* y[S,O] = x[S,I] @ W^T con W int4 impacchettato (2 valori/byte) + scala[O]. */
static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
            if(g_i4_acc512){ a=dot_i4f_avx512(w,xs,I); i=I&~31; }
            else {
#endif
#ifdef __AVX2__
            const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));   /* 8 byte=16 nibble */
                __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib=_mm_unpacklo_epi8(lo,hi);                                       /* nibble in ordine */
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint8x8_t by=vld1_u8(w+(i>>1));               /* 8 byte=16 nibble */
                uint8x8x2_t z=vzip_u8(vand_u8(by,m4), vshr_n_u8(by,4));        /* nibble in ordine */
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
            }
#endif
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += xs[i]*(float)lo; }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* Decode hot path for gate+up: same exact q4 dot products as matmul_i4, but one
 * OpenMP dispatch covers both matrices. KTransformers uses persistent pools;
 * this keeps colibri dependency-free while removing one team launch/expert. */
static void matmul_i4_pair(float *yg, float *yu, const float *x,
                           const uint8_t *qg, const float *sg,
                           const uint8_t *qu, const float *su, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int z=0;z<2*O;z++){
        int o=z<O?z:z-O; const uint8_t *w=(z<O?qg:qu)+(int64_t)o*rb;
        float a=0; int i=0;
#ifdef __AVX2__
        const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
        __m256 acc=_mm256_setzero_ps();
        for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
            __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
            __m128i nib=_mm_unpacklo_epi8(lo,hi);
            __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
            __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),w0,acc);
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),w1,acc); }
        a=hsum256(acc);
#elif defined(__ARM_NEON)
        const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
        float32x4_t ac0=vdupq_n_f32(0),ac1=vdupq_n_f32(0);
        for(;i+16<=I;i+=16){ uint8x8_t by=vld1_u8(w+(i>>1));
            uint8x8x2_t n=vzip_u8(vand_u8(by,m4),vshr_n_u8(by,4));
            int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(n.val[0]),b8));
            int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(n.val[1]),b8));
            ac0=vfmaq_f32(ac0,vld1q_f32(x+i),vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
            ac1=vfmaq_f32(ac1,vld1q_f32(x+i+4),vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
            ac0=vfmaq_f32(ac0,vld1q_f32(x+i+8),vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
            ac1=vfmaq_f32(ac1,vld1q_f32(x+i+12),vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
        a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
        for(;i+1<I;i+=2){ uint8_t b=w[i>>1]; a+=x[i]*(float)((b&15)-8)+x[i+1]*(float)((b>>4)-8); }
        if(i<I) a+=x[i]*(float)((w[i>>1]&15)-8);
        (z<O?yg:yu)[o]=a*(z<O?sg:su)[o];
    }
}

static void matmul_qt(float *y,const float *x,QT *w,int S);
static int g_no_fused_pair=0;  /* COLI_NO_FUSED_PAIR=1: disable the gate+up kernel fusion
 * that changes OMP scheduling vs separate matmul_qt calls — this
 * shifts floating-point accumulation order and can collapse MTP
 * draft acceptance by flipping near-ties (#163). */
static void expert_gate_up(float *g,float *u,const float *x,QT *wg,QT *wu,int S){
    if(!g_no_fused_pair&&S==1&&wg->fmt==2&&wu->fmt==2&&wg->I==wu->I&&wg->O==wu->O)
        matmul_i4_pair(g,u,x,wg->q4,wg->s,wu->q4,wu->s,wg->I,wg->O);
    else { matmul_qt(g,x,wg,S); matmul_qt(u,x,wu,S); }
}
/* y[S,O] = x[S,I] @ W^T con W int2 impacchettato (4 valori/byte) + scala[O]. nibble 2-bit -> [-2,1]. */
static void matmul_i2(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O){
    int rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q2+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m2=_mm_set1_epi8(0x03); const __m256i b2=_mm256_set1_epi32(2);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_cvtsi32_si128(*(const int*)(w+(i>>2)));    /* 4 byte=16 valori */
                __m128i p0=_mm_and_si128(by,m2), p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                __m128i lo=_mm_unpacklo_epi8(p0,p1), hi=_mm_unpacklo_epi8(p2,p3);
                __m128i nib=_mm_unpacklo_epi16(lo,hi);                                      /* 16 valori in ordine */
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b2));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b2));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m2v=vdup_n_u8(3); const int8x8_t b2v=vdup_n_s8(2);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint32_t wd; memcpy(&wd, w+(i>>2), 4);        /* 4 byte=16 valori */
                uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),              vand_u8(vshr_n_u8(by,2),m2v));
                uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v), vshr_n_u8(by,6));
                uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[0]),b2v));  /* 16 valori in ordine */
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[1]),b2v));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++){ uint8_t byte=w[i>>2]; int sh=(i&3)*2; a += xs[i]*(float)((int)((byte>>sh)&3)-2); }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* ---- KERNEL INTERI (IDOT): attivazioni quantizzate a int8 per riga (absmax/127,
 * stile Q8_0), prodotto scalare INTERO via maddubs/madd AVX2 — niente conversione
 * f32 dei pesi nel ciclo caldo. ~2-3x sui matmul quantizzati; errore aggiunto ~0.3%
 * RMS per matmul (attivazione int8), IDOT=0 torna al percorso f32 esatto. */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define IDOT_KERNEL "avx512-vnni"
#elif defined(__AVXVNNI__) && defined(__AVX2__)
#define IDOT_KERNEL "avx-vnni"
#elif defined(__AVX2__)
#define IDOT_KERNEL "avx2"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#elif defined(__VSX__)
#define IDOT_KERNEL "vsx"
#else
#define IDOT_KERNEL "scalar"
#endif
static int g_lmhead_exact=0;   /* PROBE: LMHEAD_EXACT=1 -> pin lm_head to the exact int8 kernel */
static int g_idot=1;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int g_i4s=1;   /* SDOT presente: int4 IDOT conviene anche a S=1 (decode). Misurato
                       * su Apple M-series: +14%%, expert-matmul -16%%. EN: with SDOT, int4
                       * IDOT pays even at S=1 (decode); measured on Apple M-series. */
#elif defined(__VSX__)
static int g_i4s=1;   /* POWER8 vec_msum: qui il fallback f32 e' SCALARE, quindi l'IDOT
                       * int4 conviene anche a S=1. Misurato su POWER8 S824 (vedi PR).
                       * EN: on VSX the f32 fallback is plain scalar C, so int4 IDOT
                       * pays even at S=1. Measured on a POWER8 S824 (see PR). */
#else
static int g_i4s=2;   /* senza SDOT / altrove: soglia originale (misura AVX2 dell'autore).
                       * EN: without SDOT / elsewhere: original threshold (author's AVX2). */
#endif
static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
#ifdef __AVX2__
static inline int hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
#endif
#if defined(__AVXVNNI__) && defined(__AVX2__)
/* hsum di un __m128i a 4 lane s32 (l'AVX-VNNI 128-bit accumula su 4 lane). */
static inline int hsum128_i32(__m128i v){
    v=_mm_hadd_epi32(v,v); v=_mm_hadd_epi32(v,v); return _mm_cvtsi128_si32(v);
}
#endif
/* dot int8·int8: trucco del segno (|w| unsigned × x·sign(w) signed). Sicuro:
 * coppie <= 128*127*2 = 32512 < 32767, accumulo s32 fino a I=16384. */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* VNNI: vpdpbusd u8*s8 -> s32 directly, 64 bytes/iter, no 16-bit intermediate.
     * AVX-512 has no vpsignb: |w| via abs, sign folded into x with a mask-negate
     * (w==0 -> product 0 either way). |x|<=127 (qrow_i8), |w|<=128 as u8: each
     * s32 lane adds <= 4*128*127, safe up to I=16384 like the AVX2 bound. */
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m512i wv=_mm512_loadu_si512((const void*)(w+i));
        __m512i xv=_mm512_loadu_si512((const void*)(x+i));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVXVNNI__) && defined(__AVX2__)
    /* AVX-VNNI 128-bit: vpdpbusd u8*s8 -> s32, 16 byte/iter. Stesso trucco del
     * segno della variante 512-bit: |w| via abs, segno piegato in x con maschera
     * (w==0 -> product 0). __AVX2__ serve per _mm_sign_epi8 / abs. */
    __m128i acc=_mm_setzero_si128();
    for(;i+16<=I;i+=16){
        __m128i wv=_mm_loadu_si128((const __m128i*)(w+i));
        __m128i xv=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i xs=_mm_sign_epi8(xv,wv);              /* x * sign(w); _mm_sign zona __AVX2__ */
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(wv),xs);
    }
    sum=hsum128_i32(acc);
#elif defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256(); const __m256i ones=_mm256_set1_epi16(1);
    for(;i+32<=I;i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    /* ARM: SDOT nativo se disponibile (Apple Silicon: sempre); altrimenti vmull/vpadal.
     * Stesso bound anti-overflow del trucco AVX2: coppie <= 128*127*2 = 32512 < 32767. */
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){
        int8x16_t wv=vld1q_s8(w+i), xv=vld1q_s8(x+i);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,wv,xv);
#else
        int16x8_t p=vmull_s8(vget_low_s8(wv),vget_low_s8(xv));
        p=vmlal_s8(p,vget_high_s8(wv),vget_high_s8(xv));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#elif defined(__VSX__)
    /* POWER8: vec_msum (s8 x u8 -> s32) somma i prodotti byte DIRETTAMENTE in lane
     * s32, 16 byte/iter: il bound anti-saturazione a 16 bit di maddubs qui non serve.
     * Stesso trucco del segno (|w| u8 per x*sign(w) s8), ma |w| via select+sub MODULO
     * e non vec_abs: -128 deve diventare 128 u8, non saturare a 127.
     * EN: vec_msum accumulates byte products straight into s32 lanes; |w| is built
     * with a modulo subtract select instead of vec_abs so w=-128 wraps to 128 (u8)
     * rather than saturating to 127. |x|<=127 from qrow_i8, so x negation is safe. */
    __vector signed int acc=vec_splats(0);
    const __vector signed char vz=vec_splats((signed char)0);
    for(;i+16<=I;i+=16){
        __vector signed char wv=vec_xl(0,(const signed char*)(w+i));
        __vector signed char xv=vec_xl(0,(const signed char*)(x+i));
        __vector __bool char neg=vec_cmplt(wv,vz);
        __vector signed char xs=vec_sel(xv,vec_sub(vz,xv),neg);
        __vector unsigned char wa=(__vector unsigned char)vec_sel(wv,vec_sub(vz,wv),neg);
        acc=vec_msum(xs,wa,acc);
    }
    sum=vec_extract(acc,0)+vec_extract(acc,1)+vec_extract(acc,2)+vec_extract(acc,3);
#endif
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
/* dot int4(packed)·int8: nibble -> int8 [-8,7] al volo, poi stesso trucco */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* 32 bytes = 64 nibbles -> int8 in [-8,7], one vpdpbusd per 64 values.
     * 256-bit unpack leaves values in per-128-lane order [0-15][32-47]/[16-31][48-63];
     * dot pairing is order-invariant, so permute x's 128-bit blocks to match
     * instead of re-ordering w (one vpermq per iter, off the critical unpack path). */
    const __m256i m4v=_mm256_set1_epi8(0x0F);
    const __m512i b8v=_mm512_set1_epi8(8);
    const __m512i xidx=_mm512_setr_epi64(0,1,4,5,2,3,6,7);
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        __m256i z0=_mm256_unpacklo_epi8(lo,hi), z1=_mm256_unpackhi_epi8(lo,hi);
        __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(z0),z1,1),b8v);
        __m512i xv=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i)));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVXVNNI__) && defined(__AVX2__)
    /* AVX-VNNI 128-bit, int4: 16 byte = 32 nibble -> int8 [-8,7] in due half
     * (n0/n1), ciascuno alimentato a un vpdpbusd da 16 byte. Stesso unpack
     * 128-bit del ramo AVX2 sotto; 32 elementi/iter come li. */
    const __m128i m4=_mm_set1_epi8(0x0F); const __m128i b8=_mm_set1_epi8(8);
    __m128i acc=_mm_setzero_si128();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));   /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);  /* nibble in ordine */
        __m128i w0=_mm_sub_epi8(n0,b8), w1=_mm_sub_epi8(n1,b8);
        __m128i x0=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i x1=_mm_loadu_si128((const __m128i*)(x+i+16));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(w0),_mm_sign_epi8(x0,w0));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(w1),_mm_sign_epi8(x1,w1));
    }
    sum=hsum128_i32(acc);
#elif defined(__AVX2__)
    const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi8(8);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));   /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);   /* in ordine */
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));                          /* 16 byte = 32 nibble */
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4)); /* nibble in ordine */
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,w0,x0); acc=vdotq_s32(acc,w1,x1);
#else
        int16x8_t p=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));      /* |w|<=8: nessun overflow */
        p=vmlal_s8(p,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p);
        p=vmull_s8(vget_low_s8(w1),vget_low_s8(x1));
        p=vmlal_s8(p,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#elif defined(__VSX__)
    /* 16 byte = 32 nibble. vec_mergeh/vec_mergel su ppc64le (GCC) interallacciano come
     * unpacklo/unpackhi x86 (verificato empiricamente su POWER8): i nibble escono in
     * ordine di memoria. |w|<=8 dopo il -8, quindi stesso trucco del segno di dot_i8i8.
     * EN: vec_mergeh/l on ppc64le interleave like x86 unpacklo/hi (verified on POWER8),
     * so nibbles come out in memory order; then the same sign trick as dot_i8i8. */
    const __vector unsigned char m4v=vec_splats((unsigned char)0x0F);
    const __vector unsigned char sh4=vec_splats((unsigned char)4);
    const __vector signed char b8v=vec_splats((signed char)8);
    const __vector signed char vz=vec_splats((signed char)0);
    __vector signed int acc=vec_splats(0);
    for(;i+32<=I;i+=32){
        __vector unsigned char by=vec_xl(0,w4+(i>>1));               /* 16 byte = 32 nibble */
        __vector unsigned char lo=vec_and(by,m4v), hi=vec_sr(by,sh4);
        __vector signed char w0=vec_sub((__vector signed char)vec_mergeh(lo,hi),b8v);
        __vector signed char w1=vec_sub((__vector signed char)vec_mergel(lo,hi),b8v);
        __vector signed char x0=vec_xl(0,(const signed char*)(x+i));
        __vector signed char x1=vec_xl(0,(const signed char*)(x+i+16));
        __vector __bool char n0=vec_cmplt(w0,vz), n1=vec_cmplt(w1,vz);
        acc=vec_msum(vec_sel(x0,vec_sub(vz,x0),n0),
                     (__vector unsigned char)vec_sel(w0,vec_sub(vz,w0),n0),acc);
        acc=vec_msum(vec_sel(x1,vec_sub(vz,x1),n1),
                     (__vector unsigned char)vec_sel(w1,vec_sub(vz,w1),n1),acc);
    }
    sum=vec_extract(acc,0)+vec_extract(acc,1)+vec_extract(acc,2)+vec_extract(acc,3);
#endif
    for(;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
static void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                          const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                           const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}

typedef struct { int8_t *xq; size_t xq_cap; float *sx; size_t sx_cap; } QScratch;
static _Thread_local QScratch g_qscratch;
static void quant_scratch(size_t xn, size_t sn, int8_t **xq, float **sx){
    if(xn>g_qscratch.xq_cap){
        int8_t *p=realloc(g_qscratch.xq,xn);
        if(!p){ fprintf(stderr,"OOM quant scratch\n"); exit(1); }
        g_qscratch.xq=p; g_qscratch.xq_cap=xn;
    }
    if(sn>g_qscratch.sx_cap){
        float *p=realloc(g_qscratch.sx,sn*sizeof(float));
        if(!p){ fprintf(stderr,"OOM quant scales\n"); exit(1); }
        g_qscratch.sx=p; g_qscratch.sx_cap=sn;
    }
    *xq=g_qscratch.xq; *sx=g_qscratch.sx;
}

/* allow_idot=0: forza il kernel int4/int8 ESATTO (attivazioni f32). Serve alle proiezioni di
 * attenzione: sono sensibili alla quantizzazione int8 delle attivazioni dell'IDOT. Misurato su
 * GLM-5.2 int4, 1023 token, log-lik -5040.33 (esatto) -> -5160.47 (IDOT) = +0.117 nat/token,
 * ~+12% perplexity. Gli altri matmul del prefill (o_proj, kv_b, expert) tengono l'IDOT.
 * EN: allow_idot=0 forces the EXACT int4/int8 kernel (f32 activations). The attention
 * projections need it: IDOT's int8 activation quantization costs +0.117 nats/token there
 * (~+12% perplexity), measured. Every other prefill matmul keeps IDOT as before. */
static void matmul_qt_ex(float *y, const float *x, QT *w, int S, int allow_idot);
static void matmul_qt(float *y, const float *x, QT *w, int S){ matmul_qt_ex(y,x,w,S,1); }
static void matmul_qt_ex(float *y, const float *x, QT *w, int S, int allow_idot){
#ifdef COLI_METAL
    /* Large row-batches (prefill: kv_b reconstruction, o_proj, dense MLP, step_all logits)
     * amortize Metal's ~5ms submit latency; small-S decode matmuls stay on CPU (NEON wins).
     * Weights must be registered (all dense QT allocs are, via qalloc). */
    if(g_metal_enabled && S>=g_metal_gemm_min && (w->fmt==1||w->fmt==2) && !omp_in_parallel()){
        const void *wp = w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_metal_gemm(y,x,wp,w->s,w->fmt,S,w->I,w->O)) return;
    }
#endif
#ifdef COLI_CUDA
    /* The CUDA backend owns persistent copies only for model-resident tensors.
     * Streaming expert slots are reused for different IDs and must never enter
     * this cache. Nested OpenMP calls stay on CPU because each device context
     * intentionally owns one synchronous scratch stream in this stage. */
    if(g_cuda_enabled && w->cuda_eligible && !w->cuda_failed && !omp_in_parallel()){
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_cuda_matmul(&w->cuda,y,x,weights,w->s,w->fmt,S,w->I,w->O,w->cuda_device)) return;
        w->cuda_failed=1;
        fprintf(stderr,"[CUDA] tensor [%d,%d] on device %d disabled after an error; falling back to CPU\n",
            w->O,w->I,w->cuda_device);
    }
#endif
    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    /* int8 IDOT vince sempre (1.4-2.5x). int4 IDOT: l'autore su AVX2 trovo' che a S=1
     * non ripaga (soglia S>=2); ma su ARM/SDOT il singolo token CONVIENE (vedi g_i4s /
     * PR #9 per il gemello VNNI). Soglia configurabile con I4S.
     * EN: int8 IDOT always wins (1.4-2.5x). int4 IDOT: on AVX2 the author found S=1 didn't
     * pay (S>=2 gate); on ARM/SDOT single-token DOES pay (see g_i4s / PR #9 for the VNNI
     * twin). Threshold configurable via I4S. */
    if(allow_idot && g_idot && (w->fmt==1 || (w->fmt==2 && S>=g_i4s))){
        int I=w->I; int8_t *xq; float *sx;
        if(S<0 || I<0 || (size_t)S>SIZE_MAX/(size_t)(I?I:1)){ fprintf(stderr,"matmul_qt: shape overflow\n"); exit(1); }
        quant_scratch((size_t)S*I,(size_t)S,&xq,&sx);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        return;
    }
    if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==3) matmul_i2(y,x,w->q4,w->s,S,w->I,w->O);
    else matmul_i4(y,x,w->q4,w->s,S,w->I,w->O);
}

/* quantizza w[O,I] f32 -> int8 q[O,I] + scala[O] simmetrica per riga */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        int8_t *qr=q+(int64_t)o*I;
        for(int i=0;i<I;i++){ int v=(int)lrintf(wr[i]/s); if(v>qmax)v=qmax; if(v<-qmax-1)v=-qmax-1; qr[i]=(int8_t)v; }
    }
}
/* quantizza w[O,I] f32 -> int4 impacchettato (2/byte) + scala[O].
 * bits<=4: valori in [-qmax-1,qmax] stanno in un nibble [-8,7]; memorizzati come v+8 (0..15). */
static void pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q4+(int64_t)o*rb;
        for(int i=0;i<I;i+=2){
            int v0=(int)lrintf(wr[i]/s); if(v0>qmax)v0=qmax; if(v0<-8)v0=-8;
            int v1=0; if(i+1<I){ v1=(int)lrintf(wr[i+1]/s); if(v1>qmax)v1=qmax; if(v1<-8)v1=-8; }
            qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
        }
    }
}

/* quantizza w[O,I] f32 -> int2 impacchettato (4/byte) + scala[O]. valori nibble 2-bit in [-2,1]. */
static void pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q2+(int64_t)o*rb;
        for(int i=0;i<I;i+=4){ uint8_t byte=0;
            for(int k=0;k<4 && i+k<I;k++){ int v=(int)lrintf(wr[i+k]/s); if(v>qmax)v=qmax; if(v<-2)v=-2; byte|=(uint8_t)((v+2)<<(k*2)); }
            qr[i>>2]=byte;
        }
    }
}

static int g_nopack=0;   /* NOPACK=1 -> tiene i valori <=4bit in contenitore int8 (per validare il packing) */
static int g_drop=0;     /* DROP=1 -> scarta le pagine expart dopo l'uso. Default 0: le lascia in
                          * page-cache (buff/cache, NON RSS) come L2 gratuito -> sfrutta lo
                          * sbilanciamento del routing MoE (pochi expert "caldi" riusati). */
static int g_prefetch=0; /* PREFETCH=1 -> riabilita il WILLNEED cross-layer (metodo C). Default
                          * OFF: i load VERI in parallelo lo hanno reso superfluo, e sotto
                          * pressione di memoria il readahead speculativo veniva rievictato. */
static int g_direct=0;   /* DIRECT=1 -> O_DIRECT sugli slab expert. Default OFF: su questo host
                          * (VHDX su NVMe DRAM-less, latenza serializzata ~60ms/req) il buffered
                          * liscio e' risultato il migliore; su NVMe veri DIRECT=1 rende di piu'. */
static float g_temp=-1;  /* TEMP: temperatura di sampling sui TOKEN. <0 = auto (1.0 in chat/testo,
                          * 0=greedy in validazione). 0 = greedy puro. */
static float g_nuc=0.95f;/* NUCLEUS: top-p sul vocabolario (default dal generation_config GLM-5.2) */
static int g_topk=0;     /* TOPK=n -> usa n expert/token invece di config (ricerca: meno disco) */
static float g_topp=0;   /* TOPP=p (0..1) -> top-p adattivo: tieni gli expert fino a peso cumulato p */
static int g_spec=1;     /* metodo C: SPEC=0 disabilita il prefetch speculativo cross-layer */
static int g_draft=0;    /* metodo E: DRAFT=n token auto-speculati per forward via n-gram lookup
                          * (0=off). LOSSLESS: verifica = output identico al greedy. Default OFF:
                          * misurato sul run reale (2026-07-03) acceptance ~5% -> ogni draft
                          * rifiutato paga comunque i suoi expert dal disco = ~3x piu' lento.
                          * Opt-in (DRAFT=4) per testi ripetitivi dove l'acceptance e' alta. */
/* metodo F (#48): GRAMMAR=<file.gbnf> -> terza sorgente di draft, la grammatica stessa.
 * Nei workload a output vincolato (JSON/NDJSON, function calling) i byte FORZATI dalla
 * grammatica (chiavi, punteggiatura, valori enum) sono draft gratuiti ad acceptance ~1:
 * nessuna testa, nessuna lookup table, e si aggancia anche dove la testa MTP int4 non
 * parte (#8). MAI un vincolo sul sampling: solo proposte, la verifica batch-union
 * decide — grammatica sbagliata = draft rifiutati, output identico.
 * GRAMMAR_DRAFT=n (default 24) limita i token forzati per forward. */
static Grammar g_gram; static GrState g_gst;
static Tok *g_gr_T=NULL;
static int g_gr_on=0;     /* grammatica caricata e walker vivo */
static int g_gr_armed=0;  /* lazy: parte dal primo byte ammesso dalla radice (salta i preamboli) */
static int g_gr_max=24;
static uint64_t g_gr_prop=0, g_gr_acc=0;
static FILE *g_route_fp=NULL; /* ROUTE_TRACE=<path>: dump per-position top-K routing (ids:gates)
                               * per layer — offline co-activation / coupling analysis. Zero
                               * effect on computation; measurement only. */
static int g_route_call=0;
/* COUPLE=<.coli_pairs>: coupling-scored cross-layer prefetch. The routing of layer L
 * strongly constrains the routing of L+1/L+2 (measured: median co-activation lift 1.8x
 * over independence, p99 40x, and the structure TRANSFERS across workloads — it is a
 * property of the model, not the session). An offline table (tools/route_pairs.py,
 * built from ROUTE_TRACE dumps) maps (layer, expert) -> top co-activated experts of the
 * next layer(s); after FASE A routing we score candidates by summing counts over the
 * position's routed set and enqueue the top COUPLE_K non-resident ones into the SAME
 * pilot ring (worker, residency re-check, safety invariants unchanged). Unlike PILOT,
 * no router matmul is needed — prediction is a table lookup on ids the layer just
 * produced. Hints only: a wrong prediction costs bandwidth, never output. */
#define CP_M 16
static int g_couple=0, g_couple_k=8, g_couple_d=1;
static int16_t *cp_pred=NULL;    /* [(L*2+(dL-1))*E + e]*CP_M + j -> target id (-1 none) */
static float   *cp_cnt=NULL;
static long g_cp_enq=0;
static void couple_prefetch(Model *m, int layer, const int *idx, int Ke);
static int g_looka=0;    /* LOOKA=1: misura (solo contatori, zero effetti) quanto il routing MoE
                          * e' predicibile IN ANTICIPO — la domanda che decide se un prefetch
                          * pilotato dal router puo' riempire i tempi morti del disco.
                          * [0] token precedente, stesso layer (cio' che usa gia' SPEC/PREFETCH)
                          * [1] ingresso del layer -> routing dello STESSO layer (salta l'attention)
                          * [2] post-attention del layer L -> routing di L+1 (un residuo MoE e
                          *     un'attention di anticipo: il punto dove il prefetch avrebbe
                          *     un intero giro di disco per lavorare in ombra). */
static int64_t la_hit[3], la_tot[3];
static int la_pred[2][130][16]; static signed char la_val[2][130];
static int g_pilot=0;    /* PILOT=1: prefetch pilotato dal router (vedi pilot_prefetch) */
static int g_pilot_k=8;  /* PILOT_K=k: prefetcha solo le prime k predizioni per posizione */
/* Aligned allocator for dense QT weights/scales: under METAL, page-align + register so the
 * GPU reads them zero-copy (no upload duplicate). Plain malloc otherwise. */
static void *qalloc(size_t n){
#ifdef COLI_METAL
    if(g_metal_enabled){ void *p; size_t r=(n+16383)&~(size_t)16383;
        if(posix_memalign(&p,16384,r)){fprintf(stderr,"OOM qalloc\n");exit(1);}
        coli_metal_register(p,r); return p; }
#endif
    return malloc(n);
}
static float *qsalloc(int O){ return (float*)qalloc((size_t)O*sizeof(float)); }
static int g_pilot_real=0;/* PILOT_REAL=1: il pilota fa LOAD VERI cross-layer dentro ecache[L+1]
                          * (non il semplice WILLNEED). Implica PILOT=1. Default OFF: hint-only. */
/* Handshake main<->pilota per il load-vero cross-layer. Invariante di sicurezza in DUE parti:
 *  1) Percorso MATMUL (moe): il pilota scrive SOLO ecache[layer] con layer > g_cur_moe_layer;
 *     il matmul in moe() legge SOLO ecache[layer]==g_cur_moe_layer, e la barriera a inizio moe()
 *     aspetta l'eventuale load in volo su QUEL layer. Quindi NESSUNO slot mezzo-caricato viene
 *     mai matmul-ato: il matmul e il pilota non toccano mai lo stesso layer contemporaneamente.
 *  2) Percorso SCAN (pilot_prefetch, anch'esso sul MAIN): la scansione di residenza gira sul
 *     layer FUTURO (lnext = layer corrente + 1), esattamente il layer che il pilota sta scrivendo
 *     -> QUI i due thread toccano davvero la stessa ecache. Percio' quella scansione prende
 *     g_pilot_mx (lo stesso lock del worker): letture e pubblicazione degli slot sono serializzate,
 *     niente torn read di ecn[]/eid. Il pilota non altera MAI il valore di un expert, solo QUALE
 *     expert e' residente: con un load andato a buon fine l'output resta byte-identico all'OFF. */
static pthread_mutex_t g_pilot_mx=PTHREAD_MUTEX_INITIALIZER;
static _Atomic int g_cur_moe_layer=-1;   /* massimo layer moe in cui il MAIN e' entrato (per forward) */
static _Atomic int g_pilot_inflight=-1;  /* layer che il worker sta REAL-caricando adesso (-1 = idle) */
static _Atomic long g_pilot_loads=0;     /* load cross-layer VERI completati (banda spesa) */
static _Atomic long g_pilot_drops=0;     /* predizioni scartate perche' il main possiede gia' il layer */
/* sceglie il formato da `bits`: >=16 f32, 5..8 int8, <=4 int4-packed */
static void qt_alloc(QT *t, int O, int I, int bits){
    t->O=O; t->I=I; t->qf=NULL; t->q8=NULL; t->q4=NULL; t->s=NULL;
    if(bits>=16){ t->fmt=0; t->qf=falloc((int64_t)O*I); }
    else if(bits>=5 || g_nopack){ t->fmt=1; t->q8=qalloc((int64_t)O*I); t->s=qsalloc(O); }
    else if(bits>=3){ t->fmt=2; t->q4=qalloc((int64_t)O*((I+1)/2)); t->s=qsalloc(O); }
    else { t->fmt=3; t->q4=qalloc((int64_t)O*((I+3)/4)); t->s=qsalloc(O); }
}
static void qt_fill(QT *t, const float *w, int bits){
    if(t->fmt==0) memcpy(t->qf, w, (int64_t)t->O*t->I*sizeof(float));
    else if(t->fmt==1) quantize_rows(w, t->q8, t->s, t->O, t->I, bits);
    else if(t->fmt==3) pack_int2(w, t->q4, t->s, t->O, t->I, bits);
    else pack_int4(w, t->q4, t->s, t->O, t->I, bits);
}

static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
/* LayerNorm classica (media+varianza, weight+bias) — usata dal k_norm dell'indexer DSA */
static void layernorm(float *v, const float *w, const float *b, int n, float eps){
    double mu=0; for(int i=0;i<n;i++) mu+=v[i]; mu/=n;
    double var=0; for(int i=0;i<n;i++){ double d=v[i]-mu; var+=d*d; } var/=n;
    float r=1.f/sqrtf((float)var+eps);
    for(int i=0;i<n;i++) v[i]=((float)(v[i]-mu))*r*w[i]+b[i];
}
static void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }
static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf(float x){ return x/(1.f+expf(-x)); }

/* RoPE interleaved su un vettore di dimensione qk_rope a posizione pos */
static void rope_interleave(float *v, int pos, const Cfg *c){
    int half = c->qk_rope/2;
    /* Validate against the fixed buffer: the config checker allows qk_rope up
     * to 1<<16 but this stack buffer holds 256 floats. Abort cleanly instead
     * of smashing the stack. (GLM-5.2 qk_rope=64, well within bounds.) */
    if(c->qk_rope > 256){ fprintf(stderr,"qk_rope=%d exceeds rope_interleave buffer (256)\n",c->qk_rope); exit(1); }
    float in[256]; memcpy(in,v,c->qk_rope*sizeof(float));
    for(int j=0;j<half;j++){
        float inv = powf(c->theta, -2.0f*j/c->qk_rope);
        float ang = pos*inv, cs=cosf(ang), sn=sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        v[j]      = a*cs - b*sn;
        v[half+j] = b*cs + a*sn;
    }
}

/* ---------- config ---------- */
static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); size_t got=fread(b,1,n,f); b[got]=0; fclose(f);
    if((long)got!=n) fprintf(stderr,"warning: short read on %s (%ld of %ld)\n",p,(long)got,n);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
static void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar);
    c->hidden=gi(r,"hidden_size"); c->n_layers=gi(r,"num_hidden_layers");
    c->n_heads=gi(r,"num_attention_heads"); c->n_experts=gi(r,"n_routed_experts");
    c->topk=gi(r,"num_experts_per_tok"); c->moe_inter=gi(r,"moe_intermediate_size");
    c->dense_inter=gi(r,"intermediate_size"); c->first_dense=gi(r,"first_k_dense_replace");
    c->q_lora=gi(r,"q_lora_rank"); c->kv_lora=gi(r,"kv_lora_rank");
    c->qk_nope=gi(r,"qk_nope_head_dim"); c->qk_rope=gi(r,"qk_rope_head_dim");
    c->v_head=gi(r,"v_head_dim"); c->n_shared=gi(r,"n_shared_experts"); c->vocab=gi(r,"vocab_size");
    c->n_group=gi(r,"n_group"); c->topk_group=gi(r,"topk_group");
    jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:0;
    jval *ep=json_get(r,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    jval *rs=json_get(r,"routed_scaling_factor"); c->routed_scale=rs?(float)rs->num:1.f;
    jval *rp=json_get(r,"rope_parameters"); jval *th=rp?json_get(rp,"rope_theta"):NULL;
    c->theta = th?(float)th->num:10000.f;
    /* token di stop: GLM-5.2 ne ha TRE (endoftext, user, observation). Fermarsi solo sul
     * primo = generare spazzatura invisibile dopo la fine del turno (5-10x token sprecati). */
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len && c->n_stop<8;i++)
                c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }
    /* DSA lightning indexer: parametri + tipo per-layer (lista esplicita o formula freq/offset) */
    c->index_topk=gi(r,"index_topk"); c->index_nh=gi(r,"index_n_heads"); c->index_hd=gi(r,"index_head_dim");
    { jval *it=json_get(r,"indexer_types");
      int freq=gi(r,"index_topk_freq"); if(freq<1) freq=1;
      jval *of=json_get(r,"index_skip_topk_offset"); int off=of?(int)of->num:2;
      for(int i=0;i<c->n_layers && i<128;i++){
          if(it && it->t==J_ARR && i<it->len && it->kids[i]->str)
              c->idx_type[i] = !strcmp(it->kids[i]->str,"full");
          else { int v=i-off+1; if(v<0) v=0; c->idx_type[i] = (v%freq)==0; }
      } }
    c->qk_head=c->qk_nope+c->qk_rope;
    c->attn_scale = 1.f / sqrtf((float)c->qk_head);
    if(c->n_group!=1){ fprintf(stderr,"this engine requires n_group=1 (GLM-5.2)\n"); exit(1); }
    /* VALIDAZIONE (report PR #25): il config.json arriva da mirror non fidati — dimensioni
     * ostili non devono superare questo punto. Un solo choke point protegge ogni alloc a valle. */
    #define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ \
        fprintf(stderr,"config: %s=%d is outside [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi)); exit(1); }
    CKR("hidden_size",c->hidden,1,1<<20)         CKR("num_hidden_layers",c->n_layers,1,128)
    CKR("num_attention_heads",c->n_heads,1,1024) CKR("n_routed_experts",c->n_experts,1,4096)
    CKR("num_experts_per_tok",c->topk,1,64)      CKR("moe_intermediate_size",c->moe_inter,1,1<<20)
    CKR("intermediate_size",c->dense_inter,1,1<<24) CKR("first_k_dense_replace",c->first_dense,0,c->n_layers)
    CKR("q_lora_rank",c->q_lora,0,1<<20)         CKR("kv_lora_rank",c->kv_lora,1,1<<20)
    CKR("qk_nope_head_dim",c->qk_nope,1,1<<16)   CKR("qk_rope_head_dim",c->qk_rope,1,1<<16)
    CKR("v_head_dim",c->v_head,1,1<<16)          CKR("n_shared_experts",c->n_shared,0,64)
    CKR("vocab_size",c->vocab,1,1<<24)           CKR("index_topk",c->index_topk,0,1<<20)
    CKR("index_n_heads",c->index_nh,0,1024)      CKR("index_head_dim",c->index_hd,0,1<<16)
    #undef CKR
    free(ar);
}

/* costruisce un QT [O,I] dal disco in `t` (buffer riusabili tra chiamate).
 *  - se esiste `name.qs`: pesi GIA' quantizzati nel container (U8 qdata + F32 scala) -> letti diretti
 *  - altrimenti: tensore pieno (f32/bf16) -> quantizzato a runtime a `bits` (oracolo tiny / pesi pieni)
 * drop=1 -> fadvise DONTNEED (streaming expert). */
static void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t){
    char sn[300]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->S,sn)){
        int64_t nb=st_nbytes(&m->S,name);
        int fmt = (nb==(int64_t)O*I)?1 : (nb==(int64_t)O*((I+1)/2))?2 : 3;  /* int8 / int4 / int2 dai byte */
        if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->q8=qalloc(nb); t->s=qsalloc(O); } st_read_raw(&m->S,name,t->q8,drop); }
        else      { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->q4=qalloc(nb); t->s=qsalloc(O); } st_read_raw(&m->S,name,t->q4,drop); }
        st_read_f32(&m->S,sn,t->s,drop);
    } else {
        if(!t->qf && !t->q8 && !t->q4) qt_alloc(t,O,I,bits);
        if(t->fmt==0) st_read_f32(&m->S,name,t->qf,drop);
        else { float *tmp=falloc((int64_t)O*I); st_read_f32(&m->S,name,tmp,drop); qt_fill(t,tmp,bits); free(tmp); }
    }
}
static QT qt_load(Model *m, const char *name, int O, int I, int bits){
    QT t; memset(&t,0,sizeof(t)); qt_from_disk(m,name,O,I,bits,0,&t);
#ifdef COLI_CUDA
    if(g_cuda_enabled&&g_cuda_dense){
        t.cuda_eligible=1;
        int slot=g_cuda_rr++%g_cuda_ndev; t.cuda_device=g_cuda_devices[slot];
        g_cuda_dense_projected[slot]+=qt_bytes(&t);
    }
#endif
    return t;
}
static float *ld(Model *m, const char *name){   /* tensore 1D f32 residente (norme/bias) */
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"missing %s\n",name);exit(1);}
    float *p=(float*)qalloc((size_t)n*sizeof(float));   /* registrato per la GPU sotto METAL */
    st_read_f32(&m->S,name,p,0); return p;
}

static void model_init(Model *m, const char *snap, int cap, int ebits, int dbits){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[256]; int H=c->n_heads, D=c->hidden;
    /* embed e lm_head sono il confine I/O: tenerli ad alta precisione (come i quant dynamic
     * reali). A bf16 ~1.9GB su GLM reale: trascurabile. dbits>=8 -> qui f32; piu' basso -> dbits. */
    int io_bits = dbits>=8 ? 16 : dbits;
    m->embed   = qt_load(m,"model.embed_tokens.weight", c->vocab, D, io_bits);
    m->lm_head = qt_load(m,"lm_head.weight", c->vocab, D, io_bits);
    m->final_norm = ld(m,"model.norm.weight");
    m->L=calloc(c->n_layers,sizeof(Layer));
    int NR=c->n_layers+1;                        /* +1: riga del layer MTP */
    m->ecap=cap; m->ecache=calloc(NR,sizeof(ESlot*)); m->ecn=calloc(NR,sizeof(int));
    m->eroute=calloc(NR,sizeof(int*)); m->enr=calloc(NR,sizeof(int));
    m->pin=calloc(NR,sizeof(ESlot*)); m->npin=calloc(NR,sizeof(int));
    m->eusage=calloc(NR,sizeof(uint32_t*)); m->eheat=calloc(NR,sizeof(uint32_t*));
    m->elast=calloc(NR,sizeof(uint32_t*));
    m->kv=calloc(1,sizeof(KVState));
    m->kv_start=m->kv->kv_start=calloc(NR,sizeof(int));
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        #define P(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
        l->in_ln=ld(m,P("input_layernorm.weight"));
        l->post_ln=ld(m,P("post_attention_layernorm.weight"));
        l->q_a   = qt_load(m,P("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
        l->q_a_ln= ld(m,P("self_attn.q_a_layernorm.weight"));
        l->q_b   = qt_load(m,P("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
        l->kv_a  = qt_load(m,P("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
        l->kv_a_ln= ld(m,P("self_attn.kv_a_layernorm.weight"));
        l->kv_b  = qt_load(m,P("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
        l->o     = qt_load(m,P("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
        l->sparse = (i >= c->first_dense);
        if(!l->sparse){
            l->gate_proj = qt_load(m,P("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->up_proj   = qt_load(m,P("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->down_proj = qt_load(m,P("mlp.down_proj.weight"), D, c->dense_inter, dbits);
        } else {
            l->router=ld(m,P("mlp.gate.weight"));
            l->router_bias=ld(m,P("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,P("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,P("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,P("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));      /* metodo C: ultimo routing del layer */
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast[i]=calloc(c->n_experts,sizeof(uint32_t));
        }
        #undef P
    }
    /* testa MTP (layer n_layers): presente solo se convertita con --mtp */
    {
        /* MTP attiva SOLO se il set e' COMPLETO (i tensori vivono su 3 shard: durante la
         * conversione parziale ne esiste solo una parte). MTP=0 la disabilita comunque. */
        const char *req[]={"eh_proj.weight","enorm.weight","hnorm.weight","shared_head.norm.weight",
            "input_layernorm.weight","post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight","self_attn.q_b_proj.weight","self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight","self_attn.o_proj.weight","mlp.gate.weight",
            "mlp.shared_experts.gate_proj.weight","mlp.shared_experts.down_proj.weight",
            "mlp.experts.0.gate_proj.weight","mlp.experts.255.down_proj.weight"};
        char mn[256]; m->has_mtp=1;
        for(unsigned q=0;q<sizeof(req)/sizeof(req[0]);q++){
            snprintf(mn,sizeof(mn),"model.layers.%d.%s",c->n_layers,req[q]);
            if(!st_has(&m->S,mn)){ m->has_mtp=0; break; }
        }
        if(getenv("MTP") && atoi(getenv("MTP"))==0) m->has_mtp=0;
        if(m->has_mtp){
            int i=c->n_layers; Layer *l=&m->mtpL;
            #define PM(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
            l->in_ln=ld(m,PM("input_layernorm.weight"));
            l->post_ln=ld(m,PM("post_attention_layernorm.weight"));
            l->q_a   = qt_load(m,PM("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
            l->q_a_ln= ld(m,PM("self_attn.q_a_layernorm.weight"));
            l->q_b   = qt_load(m,PM("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
            l->kv_a  = qt_load(m,PM("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
            l->kv_a_ln= ld(m,PM("self_attn.kv_a_layernorm.weight"));
            l->kv_b  = qt_load(m,PM("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
            l->o     = qt_load(m,PM("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
            l->sparse=1;
            l->router=ld(m,PM("mlp.gate.weight"));
            l->router_bias=ld(m,PM("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,PM("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,PM("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,PM("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            m->eh_proj = qt_load(m,PM("eh_proj.weight"), D, 2*D, dbits);
            m->enorm=ld(m,PM("enorm.weight")); m->hnorm=ld(m,PM("hnorm.weight"));
            m->mtp_norm=ld(m,PM("shared_head.norm.weight"));
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->elast[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->kv_start[i]=-1;                    /* KV MTP: parte dalla prima posizione di decode */
            #undef PM
        }
    }
    /* DSA lightning indexer: attivo SOLO se i pesi (conversione --indexer) ci sono per
     * TUTTI i layer full. Auto-rilevamento come per MTP: niente flag, niente passi extra. */
    {
        m->has_dsa = (c->index_topk>0 && c->index_nh>0 && c->index_hd>0 && c->index_hd<=256);
        char inm[300];
        for(int i=0;i<c->n_layers && m->has_dsa;i++){
            if(!c->idx_type[i]) continue;
            snprintf(inm,sizeof(inm),"model.layers.%d.self_attn.indexer.wq_b.weight",i);
            if(!st_has(&m->S,inm)) m->has_dsa=0;
        }
        if(getenv("DSA") && atoi(getenv("DSA"))==0) m->has_dsa=0;
        if(m->has_dsa){
            m->ix_wq=calloc(c->n_layers,sizeof(QT)); m->ix_wk=calloc(c->n_layers,sizeof(QT));
            m->ix_wp=calloc(c->n_layers,sizeof(QT));
            m->ix_knw=calloc(c->n_layers,sizeof(float*)); m->ix_knb=calloc(c->n_layers,sizeof(float*));
            for(int i=0;i<c->n_layers;i++){
                if(!c->idx_type[i]) continue;
                #define PI(s) (snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.indexer." s,i),nm)
                m->ix_wq[i]=qt_load(m,PI("wq_b.weight"), c->index_nh*c->index_hd, c->q_lora, dbits);
                m->ix_wk[i]=qt_load(m,PI("wk.weight"), c->index_hd, D, dbits);
                m->ix_wp[i]=qt_load(m,PI("weights_proj.weight"), c->index_nh, D, dbits);
                m->ix_knw[i]=ld(m,PI("k_norm.weight")); m->ix_knb[i]=ld(m,PI("k_norm.bias"));
                #undef PI
            }
            fprintf(stderr,"[DSA] indexer active: top-%d sparse attention beyond %d context tokens\n",
                c->index_topk, c->index_topk);
        }
    }
    m->hlast=falloc(D); m->h_all=falloc((int64_t)64*D);

    /* byte della parte DENSA residente (embed+lm_head+attn+mlp densa+shared+norme) */
    int64_t rb=qt_bytes(&m->embed)+qt_bytes(&m->lm_head);
    for(int i=0;i<c->n_layers;i++){ Layer *l=&m->L[i];
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        if(!l->sparse) rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
        else rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down);
    }
    if(m->has_mtp){ Layer *l=&m->mtpL;
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down)+qt_bytes(&m->eh_proj);
    }
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        rb+=qt_bytes(&m->ix_wq[i])+qt_bytes(&m->ix_wk[i])+qt_bytes(&m->ix_wp[i]);
    m->resident_bytes=rb;
}

/* embed: dequantizza la riga del token (scala per-riga) in x[hidden] */
static void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    if(e->fmt==0){ memcpy(x, e->qf+(int64_t)tok*D, D*sizeof(float)); return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok];
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok];   /* int4 */
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; }
        return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok];   /* int2 */
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}

/* COLI_MMAP=1: gli expert diventano VISTE dentro mmap dei file safetensors (niente pread,
 * niente slab, niente copia: la page cache del kernel E' la cache). Le mappe sono
 * registrate con Metal (newBufferWithBytesNoCopy su pagine file-backed, come llama.cpp),
 * quindi la GPU legge gli stessi byte. Fallback allo slab path su disallineamento. */
static int g_mmap=0;
static struct { int fd; void *base; size_t len; } g_maps[512]; static int g_nmaps;
static pthread_mutex_t g_map_mtx = PTHREAD_MUTEX_INITIALIZER;   /* expert_load e' OMP-parallel */
static void *map_of_fd(int fd){
    pthread_mutex_lock(&g_map_mtx);
    for(int i=0;i<g_nmaps;i++) if(g_maps[i].fd==fd){ void *b=g_maps[i].base; pthread_mutex_unlock(&g_map_mtx); return b; }
    void *base=NULL;
#if defined(__APPLE__) || defined(__linux__)
    struct stat st;
    if(g_nmaps<512 && fstat(fd,&st)==0){
        size_t len=((size_t)st.st_size+16383)&~(size_t)16383;
        void *p=mmap(NULL,len,PROT_READ,MAP_SHARED,fd,0);
        if(p!=MAP_FAILED){
            base=p; g_maps[g_nmaps].fd=fd; g_maps[g_nmaps].base=p; g_maps[g_nmaps].len=len; g_nmaps++;
#ifdef COLI_METAL
            if(g_metal_enabled) coli_metal_register(p,len);
#endif
        }
    }
#endif
    pthread_mutex_unlock(&g_map_mtx);
    return base;
}

/* carica un expert nello slot. Container pre-quantizzato: le 3 matrici sono contigue nel
 * file -> UNA pread coalescente da ~19 MB dentro `slab` (+ le scale in fslab); i QT sono
 * viste dentro lo slab (zero copie). Fallback per modelli non quantizzati (oracolo tiny).
 * THREAD-SAFE su slot distinti (pread posizionale, st_find read-only). */
/* Load one expert's weights into slot `s`. Returns 0 on success, -1 on failure.
 * fatal=1 (all main / on-demand / REPIN / pin callers): preserve the original
 * exit-on-error contract byte-for-byte — any missing tensor, OOM, short read or
 * pread error aborts the process. fatal=0 (speculative pilot only): the same
 * errors instead abandon the load and return -1 without touching s->eid, so a
 * mispredicted cross-layer prefetch can never kill the server. */
static int expert_load(Model *m, int layer, int eid, ESlot *s, int fatal){
#ifdef COLI_CUDA
    /* A live REPIN may reuse a GPU-enabled pinned slot for a different expert.
     * Keep its tier assignment, but invalidate the old device weights. */
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    char nm[3][288]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[300]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(!st_has(&m->S,qn)){                       /* fallback: tensori pieni, quantizza a runtime.
                                                  * Reachable ONLY for unquantized models (no .qs);
                                                  * GLM always has .qs, so the pilot never hits it. */
        qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
        qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
        qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
        s->eid=eid; return 0;
    }
    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm[k]); tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ fprintf(stderr,"missing %s\n",nm[k]); if(fatal) exit(1); return -1; }
    }
    if(g_mmap){
        void *bw[3],*bq[3]; int okm=1;
        for(int k=0;k<3;k++){
            bw[k]=map_of_fd(tw[k]->fd); bq[k]=map_of_fd(tq[k]->fd);
            if(!bw[k]||!bq[k]||((tw[k]->off)&3)||((tq[k]->off)&3)) okm=0;
        }
        if(okm){
            QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
            for(int k=0;k<3;k++){
                int64_t nb=tw[k]->nbytes;
                int fmt=(nb==(int64_t)OO[k]*II[k])?1:(nb==(int64_t)OO[k]*((II[k]+1)/2))?2:3;
                qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
                qt[k]->q8=(int8_t*)((char*)bw[k]+tw[k]->off); qt[k]->q4=(uint8_t*)((char*)bw[k]+tw[k]->off);
                qt[k]->s=(float*)((char*)bq[k]+tq[k]->off);
            }
            /* CPU pre-touch: fault the pages in HERE (cheap, parallel, overlapped with the
             * resident-experts GPU submit) so the GPU never demand-faults file-backed pages
             * (measured catastrophic). madvise starts async readahead, the touch guarantees
             * residency. This is pread's I/O without the copy and without the slab. */
            for(int k=0;k<3;k++){
                char *p=(char*)bw[k]+tw[k]->off; size_t n=(size_t)tw[k]->nbytes;
#if defined(__APPLE__) || defined(__linux__)
                madvise((void*)((uintptr_t)p & ~16383UL), n+16384, MADV_WILLNEED);
#endif
                volatile char acc=0;
                for(size_t i=0;i<n;i+=4096) acc+=p[i];
                acc+=p[n-1]; (void)acc;
                char *q=(char*)bq[k]+tq[k]->off; size_t nq=(size_t)tq[k]->nbytes;
                for(size_t i=0;i<nq;i+=4096) acc+=q[i];
            }
            s->eid=eid; return 0;
        }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    /* rialloca se lo slot (riusato tra layer) e' troppo piccolo per QUESTO expert:
     * pread oltre la mappatura = short-read o CORRUZIONE silenziosa dei vicini */
    if(!s->slab || wtot+8192 > s->slab_cap){
#ifdef COLI_METAL
        /* page-align + zero-copy wrap: the GPU reads this slab in place (unified memory) */
        if(s->slab && g_metal_enabled) coli_metal_unregister(s->slab);
        compat_aligned_free(s->slab);
        size_t need=((size_t)wtot+8192+16383)&~(size_t)16383;
        if(posix_memalign((void**)&s->slab,16384,need)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=need;
        if(g_metal_enabled) coli_metal_register(s->slab,need);
#else
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n"); if(fatal) exit(1); s->slab=NULL; s->slab_cap=0; return -1;}
        s->slab_cap=wtot+8192;
#endif
    }
    if(!s->fslab || ftot > s->fslab_cap){
#ifdef COLI_METAL
        /* page-align + register: the GPU reads the scales in place (unified memory).
         * Honours `fatal` exactly like the CPU arm below — a speculative pilot load
         * that hits OOM must unwind into a clean hidden slot, never exit(). */
        if(s->fslab && g_metal_enabled) coli_metal_unregister(s->fslab);
        free(s->fslab);
        size_t fb=(((size_t)ftot*sizeof(float))+16383)&~(size_t)16383;
        if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
           posix_memalign((void**)&s->fslab,16384,fb)){
            fprintf(stderr,"OOM fslab\n"); if(fatal) exit(1);
            compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0;  /* clean, hidden slot (eid stays -1) */
            s->fslab=NULL; s->fslab_cap=0; return -1;
        }
        s->fslab_cap=ftot;
        if(g_metal_enabled) coli_metal_register(s->fslab,fb);
#else
        free(s->fslab);
        if(fatal){ s->fslab=falloc(ftot); }          /* main path: byte-identical exit-on-OOM */
        else {                                        /* speculative pilot: checked alloc, never exit() */
            /* replicate falloc's anti-wrap guard + malloc (no zeroing/alignment) */
            if(ftot<0 || (uint64_t)ftot > SIZE_MAX/sizeof(float) ||
               !(s->fslab=malloc((size_t)ftot*sizeof(float)))){
                fprintf(stderr,"OOM fslab\n");
                compat_aligned_free(s->slab); s->slab=NULL; s->slab_cap=0; /* leave a clean, hidden slot (eid stays -1) */
                s->fslab=NULL; s->fslab_cap=0; return -1;
            }
        }
        s->fslab_cap=ftot;
#endif
    }
    int ord[3]={0,1,2};                          /* ordina per offset nel file */
    for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++) if(tw[ord[bb]]->off<tw[ord[a]]->off){ int t=ord[a]; ord[a]=ord[bb]; ord[bb]=t; }
    int contig = tw[ord[0]]->fd==tw[ord[1]]->fd && tw[ord[1]]->fd==tw[ord[2]]->fd
              && tw[ord[0]]->off+tw[ord[0]]->nbytes==tw[ord[1]]->off
              && tw[ord[1]]->off+tw[ord[1]]->nbytes==tw[ord[2]]->off;
    int64_t pos[3]; int done=0;
    if(contig){
        int64_t off0=tw[ord[0]]->off;
        int dfd = g_direct ? st_direct_fd(&m->S, tw[ord[0]]->fd) : -1;
        if(dfd>=0){                              /* O_DIRECT: offset/len allineati a 4K */
            int64_t base=off0 & ~4095LL, need=(off0-base)+wtot;
            int64_t len=(need+4095)&~4095LL;
            ssize_t r=pread(dfd, s->slab, len, base);
            if(r>=need){
                pos[ord[0]]=off0-base; pos[ord[1]]=pos[ord[0]]+tw[ord[0]]->nbytes;
                pos[ord[2]]=pos[ord[1]]+tw[ord[1]]->nbytes; done=1;
            }
        }
        if(!done){                               /* fallback bufferizzato */
            if(pread(tw[ord[0]]->fd, s->slab, wtot, off0)!=wtot){ perror("pread expert"); if(fatal) exit(1); return -1; }
            pos[ord[0]]=0; pos[ord[1]]=tw[ord[0]]->nbytes; pos[ord[2]]=tw[ord[0]]->nbytes+tw[ord[1]]->nbytes; done=1;
        }
    }
    if(!done){                                   /* non contigui: 3 pread bufferizzate */
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            if(pread(tw[k]->fd, s->slab+o, tw[k]->nbytes, tw[k]->off)!=tw[k]->nbytes){ perror("pread expert"); if(fatal) exit(1); return -1; }
            pos[k]=o; o+=tw[k]->nbytes; }
    }
    float *fp[3]; int64_t fo=0;                  /* scale (piccole) */
    for(int k=0;k<3;k++){
        if(pread(tq[k]->fd, (char*)(s->fslab+fo), tq[k]->nbytes, tq[k]->off)!=tq[k]->nbytes){ perror("pread qs"); if(fatal) exit(1); return -1; }
        fp[k]=s->fslab+fo; fo+=tq[k]->nbytes/4; }
    if(g_drop){                                  /* scarta subito le pagine: evita che la page
                                                  * cache in pressione strangoli il throughput */
        posix_fadvise(tw[ord[0]]->fd, tw[ord[0]]->off, wtot, POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(tq[k]->fd, tq[k]->off, tq[k]->nbytes, POSIX_FADV_DONTNEED);
    }
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
    for(int k=0;k<3;k++){
        int64_t nb=tw[k]->nbytes;
        int fmt = (nb==(int64_t)OO[k]*II[k])?1 : (nb==(int64_t)OO[k]*((II[k]+1)/2))?2 : 3;
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=fp[k];
    }
    s->eid=eid; return 0;
}

/* ============================ PIPE: load ‖ matmul ============================
 * Overlap NVMe expert-weight loads with expert matmul. A small persistent pool
 * of I/O worker pthreads runs the misses' pread (expert_load) into distinct
 * ws[] slabs and sets a per-slot `ready` flag; the MAIN thread walks the block's
 * experts in order, waiting on ready[q] only for the expert it needs right now,
 * and does all matmul_qt on itself (matmul_qt parallelises internally via OpenMP
 * and checks !omp_in_parallel() for GPU dispatch — so it must stay off the omp
 * team and off these I/O threads).
 *
 * Cross-generation safety is provided by a single generation-tagged, lock-free
 * cursor `cur = (gen<<8) | index`. The main thread is the sole writer of `gen`
 * (monotonic bump, so no ABA); workers grab jobs by CAS-advancing the low 8-bit
 * index. THE INVARIANT: a worker reads eids[i]/layer only AFTER its winning CAS,
 * and that CAS's comparand carries the generation — so if `cur`'s gen advanced
 * (a new batch was published), the CAS fails and the worker re-reads, seeing the
 * new generation. A straggler preempted anywhere (wake gap, post-cursor) can
 * therefore NEVER grab a wrong-generation job or read torn batch state: its
 * first act is a gen-checked CAS. dispatch publishes all batch state with
 * relaxed stores and then RELEASE-stores `cur`; each worker ACQUIRE-loads `cur`,
 * so the ready[] reset + eids[]/njobs/layer are visible before any worker acts.
 * The per-expert pipe_wait(ready[q]) in the matmul loop makes every grabbed job
 * complete before the block ends, so no grab outlives its generation — which is
 * why the old `active` counter AND the end-of-block drain barrier are gone (both
 * were redundant with those per-slot waits + the gen-tagged cursor). The mutex/
 * condvar exist ONLY to park/wake idle workers, never for correctness. Gated
 * behind PIPE=1; OFF => the original blocking-load + serial-matmul path runs
 * byte-identically. */
static int g_pipe=0;      /* PIPE=1: async expert-load pipeline (default OFF) */
static int g_pipe_nw=8;   /* PIPE_WORKERS=n: I/O worker threads (disk-parallel reads) */
typedef struct {
    _Atomic uint64_t cur;                         /* (gen<<8)|index; gen main-only, index 0..njobs (≤64) */
    _Atomic int njobs;                            /* current batch job count */
    _Atomic int eids[64];                         /* current batch expert ids */
    _Atomic int layer;                            /* current batch layer */
    _Atomic int ready[64];                        /* per-slot load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;        /* ONLY for parking/waking idle workers */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
static PipePool g_pp;

static void *pipe_worker(void *arg){
    (void)arg; PipePool *p=&g_pp; uint64_t seen=0;
    for(;;){
        pthread_mutex_lock(&p->mx);
        while((atomic_load_explicit(&p->cur,memory_order_relaxed)>>8)==seen)
            pthread_cond_wait(&p->cv,&p->mx);
        pthread_mutex_unlock(&p->mx);
        for(;;){
            uint64_t c=atomic_load_explicit(&p->cur,memory_order_acquire);
            seen=c>>8;
            uint32_t i=(uint32_t)(c & 0xFF);
            if(i >= (uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed))
                break;                                /* batch drained → re-park */
            if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,
                    memory_order_acq_rel,memory_order_relaxed)){
                int L  =atomic_load_explicit(&p->layer,memory_order_relaxed);
                int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed); /* AFTER winning CAS */
                expert_load(p->m,L,eid,&p->m->ws[i],1);  /* needed-now load: fatal on I/O error (matches serial path) */
                atomic_store_explicit(&p->ready[i],1,memory_order_release);
            }
            /* CAS failed → another worker advanced index (or gen advanced): re-loop */
        }
    }
    return NULL;
}
static void pipe_init(Model *m){
    if(g_pp.started) return;
    g_pp.m=m; g_pp.nw=g_pipe_nw; if(g_pp.nw>16) g_pp.nw=16; if(g_pp.nw<1) g_pp.nw=1;
    atomic_store(&g_pp.cur,0); atomic_store(&g_pp.njobs,0);
    pthread_mutex_init(&g_pp.mx,NULL); pthread_cond_init(&g_pp.cv,NULL);
    for(int i=0;i<g_pp.nw;i++) pthread_create(&g_pp.th[i],NULL,pipe_worker,NULL);
    g_pp.started=1;
}
/* enqueue `njobs` loads (slots ws[0..njobs)); returns immediately, workers run ahead.
 * Order is load-bearing: write all batch state RELAXED, then RELEASE-store cur to
 * publish it, then wake parked workers. */
static void pipe_dispatch(Model *m,int layer,const int *eids,int njobs){
    g_pp.m=m;
    atomic_store_explicit(&g_pp.njobs,njobs,memory_order_relaxed);
    atomic_store_explicit(&g_pp.layer,layer,memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.eids[q],eids[q],memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.ready[q],0,memory_order_relaxed); /* reset BEFORE publish */
    uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
    atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);                          /* PUBLISH */
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
}
static inline void pipe_wait(int q){
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
}

#ifdef COLI_CUDA
static void expert_host_release(Model *m, ESlot *s){
    if(!s->slab&&!s->fslab) return;
#if defined(__APPLE__) || defined(__linux__)
    if(s->slab) munlock(s->slab,(size_t)s->slab_cap);
    if(s->fslab) munlock(s->fslab,(size_t)s->fslab_cap*sizeof(float));
#elif defined(_WIN32)
    if(s->slab) compat_munlock(s->slab,(size_t)s->slab_cap);
    if(s->fslab) compat_munlock(s->fslab,(size_t)s->fslab_cap*sizeof(float));
#endif
    int64_t bytes=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
    free(s->slab); free(s->fslab); s->slab=NULL; s->fslab=NULL; s->slab_cap=s->fslab_cap=0;
    QT *q[3]={&s->g,&s->u,&s->d};
    for(int k=0;k<3;k++){ q[k]->qf=NULL; q[k]->q8=NULL; q[k]->q4=NULL; q[k]->s=NULL; }
    m->resident_bytes-=bytes; if(m->resident_bytes<0) m->resident_bytes=0;
}
static void expert_host_ensure(Model *m, int layer, ESlot *s){
    if(!s->slab) expert_load(m,layer,s->eid,s,1);
}
#endif

/* prefetch asincrono dei pesi di un expert (e delle sue scale .qs): avvia il readahead
 * cosi' le letture sincrone successive trovano la page-cache calda. */
static void expert_prefetch(Model *m, int layer, int eid){
    char nm[300];
    const char *suf[3]={"gate_proj.weight","up_proj.weight","down_proj.weight"};
    for(int k=0;k<3;k++){
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s",layer,eid,suf[k]); st_prefetch(&m->S,nm);
        char qs[320]; snprintf(qs,sizeof(qs),"%s.qs",nm); st_prefetch(&m->S,qs);
    }
}

/* ---- helper per l'ABSORPTION: accesso per-riga ai QT quantizzati ---- */
/* acc[0..I) += coef * W[row,:] (dequant al volo) */
static void qt_addrow(const QT *t, int row, float coef, float *acc){
    int I=t->I;
    if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=coef*w[i]; return; }
    float c=coef*t->s[row];
    if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=c*(float)w[i]; return; }
    if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2);
        for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc[i]+=c*((int)(b&0xF)-8); acc[i+1]+=c*((int)(b>>4)-8); }
        if(I&1){ uint8_t b=w[I>>1]; acc[I-1]+=c*((int)(b&0xF)-8); } return; }
    const uint8_t *w=t->q4+(int64_t)row*((I+3)/4);
    for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc[i]+=c*((int)((b>>((i&3)*2))&3)-2); }
}
/* y[0..n) = W[r0+j,:]·x  (matvec su una FETTA di righe del QT) */
static void qt_matvec_rows(const QT *t, int r0, int n, const float *x, float *y){
    int I=t->I;
    for(int j=0;j<n;j++){ int row=r0+j; double a=0;
        if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) a+=(double)w[i]*x[i]; }
        else if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; float s=t->s[row];
            float acc=0; for(int i=0;i<I;i++) acc+=(float)w[i]*x[i]; a=acc*s; }
        else if(t->fmt==2){ const uint8_t *w=t->q4+(int64_t)row*((I+1)/2); float s=t->s[row]; float acc=0;
            for(int i=0;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
            if(I&1){ uint8_t b=w[I>>1]; acc+=((int)(b&0xF)-8)*x[I-1]; } a=acc*s; }
        else { const uint8_t *w=t->q4+(int64_t)row*((I+3)/4); float s=t->s[row]; float acc=0;
            for(int i=0;i<I;i++){ uint8_t b=w[i>>2]; acc+=((int)((b>>((i&3)*2))&3)-2)*x[i]; } a=acc*s; }
        y[j]=(float)a;
    }
}
static int g_absorb=-1;   /* ABSORB: -1 auto (decode S<=4), 0 mai, 1 sempre (test) */
static int g_dsa_force=0; /* DSA_FORCE=1: selezione sempre attiva (test: top-min(k,T)=denso) */
static int cmp_fdesc(const void *a,const void *b){
    float x=*(const float*)a, y=*(const float*)b; return x<y?1:x>y?-1:0; }

/* attenzione MLA con KV-cache compressa, su token nuovi x[S,hidden], pos_base = pos del primo */
/* kvs/pos describe a ragged decode batch: each row may belong to a different
 * sequence.  NULL keeps the original contiguous, currently-bound KV path. */
static void attention_rows(Model *m, Layer *l, int layer, float *x, int S, int pos_base,
                           KVState *const *kvs, const int *positions, float *out){
    Cfg *c=&m->c; int H=c->n_heads, D=c->hidden, qh=c->qk_head, vh=c->v_head;
    int kvb_dim=H*(c->qk_nope+vh), Tk=pos_base+S;
    double ta0=now_s();
#ifdef COLI_METAL
    /* Fused decode attention on GPU: whole layer in one command buffer (keeps the GPU hot).
     * S<=4 absorption path with st0==0, DSA selection inactive, and GLM-5.2 int4 dims. */
    if(g_metal_enabled && S<=4 && (g_absorb==1||(g_absorb<0&&S<=4)) && m->kv_start[layer]==0
       && D==6144 && H==64 && c->q_lora==2048 && c->kv_lora==512 && c->qk_nope==192
       && c->qk_rope==64 && vh==256 && l->kv_b.fmt==2){
        int sel_active = m->has_dsa && layer<c->n_layers && c->idx_type[layer] && (pos_base+S) > c->index_topk;
        if(!sel_active){
            if(m->has_dsa && layer<c->n_layers && c->idx_type[layer]){   /* index keys for future selection */
                for(int s=0;s<S;s++){ int pos=pos_base+s; float *kd=m->Ic[layer]+(int64_t)pos*c->index_hd;
                    matmul_qt(kd, x+(int64_t)s*D, &m->ix_wk[layer], 1);
                    layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], c->index_hd, 1e-6f);
                    rope_interleave(kd, pos, c); }
            }
            #define WP_(q) ((q).fmt==1?(const void*)(q).q8:(const void*)(q).q4)
            int ok = coli_metal_attn_decode(x,
                WP_(l->q_a), l->q_a.s, l->q_a.fmt, l->q_a_ln,
                WP_(l->q_b), l->q_b.s, l->q_b.fmt,
                WP_(l->kv_a), l->kv_a.s, l->kv_a.fmt, l->kv_a_ln,
                WP_(l->kv_b), l->kv_b.s, l->kv_b.fmt,
                WP_(l->o), l->o.s, l->o.fmt,
                m->Lc[layer], m->Rc[layer], S, pos_base, m->kv_start[layer], c->eps, c->theta, c->attn_scale, out);
            #undef WP_
            if(ok){ m->t_attn += now_s()-ta0; return; }
        }
    }
#endif
    float *ctx=falloc((int64_t)S*H*vh);
    float *Q=falloc((int64_t)S*H*qh);                  /* query (roped) dei token nuovi */
    int cw=c->kv_lora+c->qk_rope;
    float *QR=falloc((int64_t)S*c->q_lora), *comp=falloc((int64_t)S*cw);
    /* 1) query roped + latente normato e k_rot roped -> in cache.
     * QR tiene il residuo q_a per TUTTE le posizioni: serve anche all'indexer DSA.
     *
     * BATCH-ROWS: le tre proiezioni girano su tutte le S righe in un colpo solo, come gia' fa
     * o_proj (matmul_qt(...,S) sotto) e come fa moe() con la batch-union. Una riga per volta
     * il peso veniva ri-letto per OGNI token; a S righe si legge una volta sola.
     * matmul_qt_ex(...,0): restano sul kernel int4 ESATTO. Con l'IDOT (che il gate S>=g_i4s
     * abiliterebbe da solo appena S>1) il prefill sarebbe molto piu' veloce ma la qualita'
     * cala: -5040.33 -> -5158.68 di log-lik su 1023 token (~+12% perplexity). Il batch da
     * solo e' bit-identical all'originale; il kernel no. Vedi issue.
     * EN: batch the three projections over all S rows, like o_proj below and moe()'s
     * batch-union. matmul_qt_ex(...,0) keeps them on the EXACT int4 kernel: letting S>1 pull
     * them into IDOT is much faster but costs ~12% perplexity (measured). Batching alone is
     * bit-identical to upstream; the kernel switch is not. */
    matmul_qt_ex(QR, x, &l->q_a, S, 0);
    for(int s=0;s<S;s++){ float *qr=QR+(int64_t)s*c->q_lora;
        rmsnorm(qr, qr, l->q_a_ln, c->q_lora, c->eps); }         /* q_b legge il residuo NORMATO */
    matmul_qt_ex(Q, QR, &l->q_b, S, 0);
    matmul_qt_ex(comp, x, &l->kv_a, S, 0);
    for(int s=0;s<S;s++){
        KVState *ks=kvs?kvs[s]:m->kv;
        int pos=positions?positions[s]:pos_base+s;
        float *qfull=Q+(int64_t)s*H*qh;
        for(int h=0;h<H;h++) rope_interleave(qfull+(int64_t)h*qh+c->qk_nope, pos, c);
        const float *cs=comp+(int64_t)s*cw;
        float *Ldst=coli_kv_row(ks->Lc[layer],pos,c->kv_lora);
        float *Rdst=coli_kv_row(ks->Rc[layer],pos,c->qk_rope);
        memcpy(Ldst, cs, c->kv_lora*sizeof(float));
        rmsnorm(Ldst, Ldst, l->kv_a_ln, c->kv_lora, c->eps);     /* latente normato */
        memcpy(Rdst, cs+c->kv_lora, c->qk_rope*sizeof(float));
        rope_interleave(Rdst, pos, c);                            /* k_rot roped, condiviso fra teste */
    }
    /* ---- DSA lightning indexer ----
     * Layer FULL: k_idx dei token nuovi in cache + selezione top-k per query (riusata
     * dai layer SHARED successivi). Selezione attiva solo con contesto > index_topk
     * (o DSA_FORCE=1 per il test: selezionare TUTTO deve dare l'output denso esatto). */
    const int *dsel=NULL, *dnsel=NULL; int dtopk=0;
    if(m->has_dsa && layer<c->n_layers && ((!kvs && m->kv_start[layer]==0) || kvs)){
        int nh=c->index_nh, hd=c->index_hd; dtopk=c->index_topk;
        if(c->idx_type[layer]){
            /* BATCH-ROWS, come le proiezioni di attenzione sopra: ix_wk (D x index_hd) veniva
             * ri-letto per OGNI token. matmul_qt_ex(...,0) lo tiene sul kernel int4 ESATTO:
             * il batch da solo supererebbe il gate S>=g_i4s e cambierebbe la quantizzazione
             * delle attivazioni. Cosi' l'output resta bit-identical.
             * EN: batch ix_wk over all S rows like the attention projections; allow_idot=0
             * keeps it on the exact int4 kernel so the result stays bit-identical. */
            float *KD=falloc((int64_t)S*hd);
            matmul_qt_ex(KD, x, &m->ix_wk[layer], S, 0);
            for(int s=0;s<S;s++){
                KVState *ks=kvs?kvs[s]:m->kv;
                int pos=positions?positions[s]:pos_base+s;
                float *kd=coli_kv_row(ks->Ic[layer],pos,hd);
                memcpy(kd, KD+(int64_t)s*hd, (size_t)hd*sizeof(float));
                layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], hd, 1e-6f);
                rope_interleave(kd, pos, c);                 /* primi qk_rope dim, interleaved */
            }
            free(KD);
            if((int64_t)S*dtopk > m->dsa_scap){
                free(m->dsa_sel); free(m->dsa_nsel);
                m->dsa_scap=(int64_t)S*dtopk;
                m->dsa_sel=malloc((size_t)m->dsa_scap*sizeof(int));
                m->dsa_nsel=malloc((size_t)S*sizeof(int));
            }
            #pragma omp parallel for schedule(dynamic,1)
            for(int s=0;s<S;s++){
                KVState *ks=kvs?kvs[s]:m->kv;
                int pos=positions?positions[s]:pos_base+s, nk=pos+1;
                if(ks->kv_start[layer]!=0){ m->dsa_nsel[s]=0; continue; }
                if(nk<=dtopk && !g_dsa_force){ m->dsa_nsel[s]=0; continue; }
                int keep = nk<dtopk ? nk : dtopk;
                float *qi=falloc((int64_t)nh*hd);
                matmul_qt(qi, QR+(int64_t)s*c->q_lora, &m->ix_wq[layer], 1);
                for(int h=0;h<nh;h++) rope_interleave(qi+(int64_t)h*hd, pos, c);
                float *w32=falloc(nh);
                matmul_qt(w32, x+(int64_t)s*D, &m->ix_wp[layer], 1);
                float wsc=1.f/sqrtf((float)nh), rs=1.f/sqrtf((float)hd);
                float *isc=falloc(nk);
                for(int t=0;t<nk;t++){
                    const float *kt=coli_kv_row(ks->Ic[layer],t,hd);
                    float a=0;
                    for(int h=0;h<nh;h++){ const float *qhp=qi+(int64_t)h*hd;
                        float d0=0; for(int i=0;i<hd;i++) d0+=qhp[i]*kt[i];
                        d0*=rs; if(d0>0) a+=w32[h]*d0;       /* ReLU sullo score, poi peso */
                    }
                    isc[t]=a*wsc;
                }
                /* top-keep: soglia via qsort desc, poi scan in ordine di posizione */
                float *tmp=falloc(nk); memcpy(tmp,isc,nk*sizeof(float));
                qsort(tmp,nk,sizeof(float),cmp_fdesc);
                float thr=tmp[keep-1];
                int *dst=m->dsa_sel+(int64_t)s*dtopk, nd=0;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
                m->dsa_nsel[s]=nd;
                free(qi); free(w32); free(isc); free(tmp);
            }
        }
        if(m->dsa_nsel){ dsel=m->dsa_sel; dnsel=m->dsa_nsel; }
    }
    /* WEIGHT ABSORPTION (DeepSeek): per S piccoli (decode/verifica MTP) NON si ricostruisce
     * k/v per ogni token del contesto. Per linearita':
     *   q·k_nope_t = (W_K^hT q_nope)·L_t      ctx^h = W_V^h (Σ_t a_t L_t)
     * costo per step ~O(T·kv_lora) invece di O(T·H·(nope+vh)) del matmul kvb_all. */
    int absorb = kvs || g_absorb==1 || (g_absorb<0 && S<=4);
    if(absorb && c->kv_lora<=512){
        m->t_aproj+=now_s()-ta0; double tac=now_s();
        int kvl=c->kv_lora, r0v=c->qk_nope;      /* offset righe V dentro il blocco di testa */
        /* Punteggi per-thread sul HEAP. Il cap DEVE essere il massimo nt effettivo del
         * batch, non Tk+1: Tk=pos_base+S vale solo quando pos==pos_base+s. Il percorso
         * batched (step_decode_batch da run_serve_mux) passa positions[] e kv_start
         * per-slot, quindi nt=pos+1-st0 puo' superare Tk+1 -> heap-buffer-overflow su
         * sc[jj]. Si conta esattamente come il loop sotto. */
        int64_t sc_cap = 1;
        for(int s=0;s<S;s++){
            KVState *ks=kvs?kvs[s]:m->kv;
            int pos=positions?positions[s]:pos_base+s;
            int st0=ks->kv_start[layer];
            int ns=(dnsel && dnsel[s]>0)?dnsel[s]:0;      /* DSA: top-k, altrimenti range pieno */
            int64_t nt = ns ? (int64_t)ns : (int64_t)pos+1-st0;
            if(nt>sc_cap) sc_cap=nt;
        }
        float *sc_all = falloc((int64_t)omp_get_max_threads()*sc_cap);
        int cuda_core=0;
#ifdef COLI_CUDA
        if(S<=4&&g_cuda_enabled&&getenv("COLI_CUDA_ATTN")&&atoi(getenv("COLI_CUDA_ATTN"))&&
           l->kv_b.cuda_eligible&&qt_cuda_upload(&l->kv_b)){
            cuda_core=1;
            for(int s=0;s<S&&cuda_core;s++){
                KVState *ks=kvs?kvs[s]:m->kv;int pos=positions?positions[s]:pos_base+s;
                int st0=ks->kv_start[layer],nt=pos+1-st0;
                if(dnsel&&dnsel[s]>0){cuda_core=0;break;}
                cuda_core=coli_cuda_attention_absorb(l->kv_b.cuda,ctx+(int64_t)s*H*vh,
                    Q+(int64_t)s*H*qh,coli_kv_row(ks->Lc[layer],st0,kvl),
                    coli_kv_row(ks->Rc[layer],st0,c->qk_rope),H,c->qk_nope,c->qk_rope,
                    vh,kvl,nt,c->attn_scale);
            }
        }
#endif
        if(!cuda_core){
        #pragma omp parallel for collapse(2) schedule(static)
        for(int s=0;s<S;s++) for(int h=0;h<H;h++){
            KVState *ks=kvs?kvs[s]:m->kv;
            int pos=positions?positions[s]:pos_base+s;
            const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;
            const float *qr=qp+c->qk_nope;
            int rbase=h*(c->qk_nope+vh);
            float qabs[512]; memset(qabs,0,kvl*sizeof(float));
            for(int d=0;d<c->qk_nope;d++) qt_addrow(&l->kv_b, rbase+d, qp[d], qabs);
            float *sc = sc_all + (int64_t)omp_get_thread_num()*sc_cap;
            int st0=ks->kv_start[layer];
            int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;    /* DSA: lista top-k o range pieno */
            const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
            int nt = ns ? ns : pos+1-st0;
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=coli_kv_row(ks->Lc[layer],t,kvl);
                const float *kr=coli_kv_row(ks->Rc[layer],t,c->qk_rope);
                float a=0; for(int i=0;i<kvl;i++) a+=qabs[i]*Lt[i];
                for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
                sc[jj]=a*c->attn_scale;
            }
            softmax(sc,nt);
            float clat[512]; memset(clat,0,kvl*sizeof(float));
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=coli_kv_row(ks->Lc[layer],t,kvl);
                float a=sc[jj]; for(int i=0;i<kvl;i++) clat[i]+=a*Lt[i]; }
            qt_matvec_rows(&l->kv_b, rbase+r0v, vh, clat, ctx+((int64_t)s*H+h)*vh);
        }
        }
        m->t_acore+=now_s()-tac; double tao=now_s();
        matmul_qt(out, ctx, &l->o, S); m->t_aout+=now_s()-tao;
        free(ctx); free(Q); free(QR); free(comp); free(sc_all);
        m->t_attn += now_s()-ta0;
        return;
    }
    /* 2) ricostruzione di k_nope+value per TUTTI i token 0..Tk-1 (un solo matmul su kv_b) */
    m->t_aproj+=now_s()-ta0; double tk0=now_s();
    int stL=m->kv_start[layer];
    float *kvb_all=falloc((int64_t)Tk*kvb_dim);
    matmul_qt(kvb_all+(int64_t)stL*kvb_dim, m->Lc[layer]+(int64_t)stL*c->kv_lora, &l->kv_b, Tk-stL);
    m->t_kvb += now_s()-tk0;
    /* 3) attenzione causale: score = q_pass·k_nope + q_rot·k_rot
     * (punteggi sul heap, per-thread: vedi il commento nel ramo absorb) */
    int64_t sc_cap = Tk - stL;
    float *sc_all = falloc((int64_t)omp_get_max_threads()*sc_cap);
    double tac=now_s();
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s;
        const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;          /* [qk_nope | qk_rope] */
        const float *qr=qp+c->qk_nope;
        float *sc = sc_all + (int64_t)omp_get_thread_num()*sc_cap;
        int st0=m->kv_start[layer];
        int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;        /* DSA: lista top-k o range pieno */
        const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
        int nt = ns ? ns : pos+1-st0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            const float *kn=kvb_all+(int64_t)t*kvb_dim+(int64_t)h*(c->qk_nope+vh);
            const float *kr=m->Rc[layer]+(int64_t)t*c->qk_rope;
            float a=0; for(int d=0;d<c->qk_nope;d++) a+=qp[d]*kn[d];
            for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
            sc[jj]=a*c->attn_scale;
        }
        softmax(sc,nt);
        float *cx=ctx+((int64_t)s*H+h)*vh; for(int d=0;d<vh;d++) cx[d]=0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            const float *vv=kvb_all+(int64_t)t*kvb_dim+(int64_t)h*(c->qk_nope+vh)+c->qk_nope;
            float a=sc[jj]; for(int d=0;d<vh;d++) cx[d]+=a*vv[d]; }
    }
    m->t_acore+=now_s()-tac; double tao=now_s();
    matmul_qt(out, ctx, &l->o, S); m->t_aout+=now_s()-tao;
    free(ctx); free(Q); free(QR); free(comp); free(kvb_all); free(sc_all);
    m->t_attn += now_s()-ta0;
}

static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    attention_rows(m,l,layer,x,S,pos_base,NULL,NULL,out);
}

/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
 * BATCH-UNION: per S>1 (prefill, verifica MTP) ogni expert UNICO del batch viene caricato
 * una volta sola e moltiplicato per tutte le posizioni che lo usano (pesi letti 1 volta);
 * lo shared expert e' un unico matmul a S righe. Per posizione l'accumulo resta
 * nell'ordine (routed nel loro ordine di union, poi shared). */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out){
    if(g_pilot_real){   /* barriera cross-layer: prendi possesso di QUESTO layer e aspetta
                         * l'eventuale load-pilota in volo sullo stesso layer (dopodiche' il
                         * worker droppa ogni nuovo load <= layer -> ecache[layer] e' stabile
                         * per tutto il resolve/matmul/promozione qui sotto). */
        for(;;){
            pthread_mutex_lock(&g_pilot_mx);
            atomic_store_explicit(&g_cur_moe_layer,layer,memory_order_release);
            int inf=atomic_load_explicit(&g_pilot_inflight,memory_order_acquire);
            pthread_mutex_unlock(&g_pilot_mx);
            if(inf!=layer) break;
            sched_yield();
        }
    }
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    float *logit=falloc(E), *choice=falloc(E);
    int sI=c->moe_inter*c->n_shared;
    /* ---- FASE A: routing di tutte le S posizioni ---- */
    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));
    int *keff=malloc(S*sizeof(int));
#ifdef COLI_METAL
    if(g_pre_idx){                               /* routing gia' calcolata dal layer CB (GPU) */
        memcpy(idxs,g_pre_idx,(size_t)S*K*sizeof(int));
        memcpy(ws,g_pre_w,(size_t)S*K*sizeof(float));
        memcpy(keff,g_pre_keff,(size_t)S*sizeof(int));
        for(int s=0;s<S;s++){
            m->ereq+=keff[s];
            for(int kk=0;kk<keff[s];kk++){
                m->eusage[layer][idxs[(int64_t)s*K+kk]]++;
                ehit_mark(m,layer,idxs[(int64_t)s*K+kk]);
                if(m->eheat[layer][idxs[(int64_t)s*K+kk]]<UINT32_MAX) m->eheat[layer][idxs[(int64_t)s*K+kk]]++;
            }
            for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
        }
    } else
#endif
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D;
        matmul(logit, xs, l->router, 1, D, E);
        for(int e=0;e<E;e++){ logit[e]=sigmoidf(logit[e]); choice[e]=logit[e]+l->router_bias[e]; }
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
            idx[kk]=best; w[kk]=logit[best];
        }
        int Ke=Ksel;
        if(g_topp>0 && g_topp<1.f){
            for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
                while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
            float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
            float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=g_topp*tot){ Ke=kk+1; break; } }
        }
        keff[s]=Ke; m->ereq+=Ke;
        for(int kk=0;kk<Ke;kk++){
            m->eusage[layer][idx[kk]]++;
            ehit_mark(m,layer,idx[kk]);
            if(m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++;
            m->elast[layer][idx[kk]]=++m->eaccess_clock;
        }
        if(c->norm_topk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int kk=0;kk<Ke;kk++) w[kk]*=c->routed_scale;
        if(g_route_fp){                       /* ROUTE_TRACE: one line per (position, layer) */
            fprintf(g_route_fp,"%d %d %d",g_route_call,s,layer);
            for(int kk=0;kk<Ke;kk++) fprintf(g_route_fp," %d:%.4f",idx[kk],w[kk]);
            fputc('\n',g_route_fp);
        }
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
    }
    if(g_route_fp) g_route_call++;
    if(g_couple && cp_pred && S<=8)
        for(int s2=0;s2<S;s2++) couple_prefetch(m,layer,idxs+(int64_t)s2*K,keff[s2]);
    if(g_looka && S==1 && layer<c->n_layers){
        int Ke=keff[0];
        if(m->enr[layer]>0){                       /* [0] vs routing del token precedente */
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<m->enr[layer];z++)
                if(m->eroute[layer][z]==idxs[kk]){ la_hit[0]++; break; }
            la_tot[0]+=Ke;
        }
        for(int kind=0;kind<2;kind++) if(la_val[kind][layer]){   /* [1]/[2] vs predizioni */
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<K;z++)
                if(la_pred[kind][layer][z]==idxs[kk]){ la_hit[1+kind]++; break; }
            la_tot[1+kind]+=Ke; la_val[kind][layer]=0;
        }
    }
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
    /* ---- FASE B: union degli expert del batch ---- */
    int *uniq=malloc((size_t)E*sizeof(int)); int nu=0;
    unsigned char seen[E]; memset(seen,0,(size_t)E);
    for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){
        int e=idxs[(int64_t)s*K+kk];
        if(!seen[e]){ seen[e]=1; uniq[nu++]=e; }
    }
    /* ---- FASE C/D: risolvi (pin/cache/disco) e calcola, a blocchi di 64 unici ---- */
    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);
    int *rows=malloc(S*sizeof(int)); float *rw=malloc(S*sizeof(float));
#ifdef COLI_CUDA
    int group_enabled=S<=64;
    float *group_x=group_enabled?falloc((int64_t)S*K*D):NULL;
    float *group_y=group_enabled?falloc((int64_t)S*K*D):NULL;
    int *group_row=group_enabled?malloc((size_t)64*S*sizeof(int)):NULL;
    float *group_weight=group_enabled?malloc((size_t)64*S*sizeof(float)):NULL;
#endif
    int shared_on_gpu=0; (void)shared_on_gpu;   /* set by the Metal path when Phase E was fused */
    for(int base=0;base<nu;base+=64){
        int nb = nu-base<64 ? nu-base : 64;
        ESlot *use[64]; int missk[64]; int qof[64]; int nmiss=0;
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; use[j]=NULL; qof[j]=-1;
            ESlot *P=m->pin[layer];
            for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ m->hits++; use[j]=&P[z]; break; }
            if(!use[j]){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ m->hits++; Sl[z].used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED); use[j]=&Sl[z]; break; } }
            if(!use[j]){ qof[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
        }
        int metal_done=0;
#ifdef COLI_METAL
        /* GPU/disk OVERLAP: submit the RESIDENT experts (pin/LRU hits, + shared expert on
         * the first block) to the GPU BEFORE loading the missed experts from disk, so the
         * preads run while the GPU computes; the missed subset follows in a second submit.
         * Per-subset CPU fallback on unresolved slab / bad fmt / GPU fault. */
        int is_miss[64]={0}; ColiMetalMoeHandle *mh=NULL;
        int cpu_res=1, cpu_miss=1, mh_shared=0, nbb=0, Rtot=0, mfmt=-1, sh_in=0;
        const void *MG[65],*MU[65],*MD[65]; const float *MGS[65],*MUS[65],*MDS[65];
        int xoffb[65],nrb[65];
        float *mxg=NULL; int *mrows=NULL; float *mrw=NULL;
        /* subset builder: experts with is_miss==WANTMISS (+ shared expert when TRY_SH) */
        #define MB_BUILD(WANTMISS, TRY_SH) do{ \
            nbb=0; Rtot=0; mfmt=-1; sh_in=0; \
            for(int j=0;j<nb;j++){ if(is_miss[j]!=(WANTMISS)) continue; \
                int eid=uniq[base+j]; ESlot *e=use[j]; int cnt=0; \
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++) \
                    if(idxs[(int64_t)s*K+kk]==eid){ cnt++; break; } \
                if(!cnt) continue; \
                if(mfmt<0) mfmt=e->g.fmt; \
                MG[nbb]=e->g.fmt==1?(const void*)e->g.q8:(const void*)e->g.q4; \
                MU[nbb]=e->u.fmt==1?(const void*)e->u.q8:(const void*)e->u.q4; \
                MD[nbb]=e->d.fmt==1?(const void*)e->d.q8:(const void*)e->d.q4; \
                MGS[nbb]=e->g.s; MUS[nbb]=e->u.s; MDS[nbb]=e->d.s; \
                xoffb[nbb]=Rtot; nrb[nbb]=cnt; Rtot+=cnt; nbb++; \
            } \
            if(TRY_SH){ int shf = mfmt<0 ? l->sh_gate.fmt : mfmt; \
                if(c->n_shared==1 && sI==I && l->sh_gate.fmt==shf && l->sh_up.fmt==shf && l->sh_down.fmt==shf){ \
                    if(mfmt<0) mfmt=shf; \
                    MG[nbb]=shf==1?(const void*)l->sh_gate.q8:(const void*)l->sh_gate.q4; \
                    MU[nbb]=shf==1?(const void*)l->sh_up.q8  :(const void*)l->sh_up.q4; \
                    MD[nbb]=shf==1?(const void*)l->sh_down.q8:(const void*)l->sh_down.q4; \
                    MGS[nbb]=l->sh_gate.s; MUS[nbb]=l->sh_up.s; MDS[nbb]=l->sh_down.s; \
                    xoffb[nbb]=Rtot; nrb[nbb]=S; Rtot+=S; nbb++; sh_in=1; } } \
            int p=0; \
            for(int j=0;j<nb;j++){ if(is_miss[j]!=(WANTMISS)) continue; int eid=uniq[base+j]; \
                for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++) \
                    if(idxs[(int64_t)s*K+kk]==eid){ \
                        memcpy(mxg+(int64_t)p*D, x+(int64_t)s*D, D*sizeof(float)); \
                        mrows[p]=s; mrw[p]=ws[(int64_t)s*K+kk]; p++; break; } } \
            if(sh_in) for(int s=0;s<S;s++){ \
                memcpy(mxg+(int64_t)p*D, x+(int64_t)s*D, D*sizeof(float)); \
                mrows[p]=s; mrw[p]=1.0f; p++; } \
        }while(0)
        if(g_metal_enabled){
            for(int q=0;q<nmiss;q++) is_miss[missk[q]]=1;
            mxg=falloc((int64_t)(nb+1)*S*D);
            mrows=malloc((size_t)(nb+1)*S*sizeof(int)); mrw=malloc((size_t)(nb+1)*S*sizeof(float));
            MB_BUILD(0, base==0 && !g_pre_sh);
            if(nbb>0){
                double t0=now_s();
                mh=coli_metal_moe_block_begin(nbb,D,I,mfmt,MG,MU,MD,MGS,MUS,MDS,mxg,xoffb,nrb,mrows,mrw);
                m->t_emm += now_s()-t0;
                if(mh){ cpu_res=0; mh_shared=sh_in; }
            } else cpu_res=0;
        }
#endif
        /* Expert loads run HERE, after the resident-experts GPU submit above: under METAL the
         * preads overlap the GPU compute (that submit is async). With METAL off the submit block
         * is a no-op / compiled out, so this sits exactly where dev put it and CPU behaviour is
         * unchanged. */
        if(nmiss){
            if(g_pipe){                            /* PIPE: launch loads async, matmul overlaps them */
                if(!g_pp.started) pipe_init(m);
                double t0=now_s();
                int eids[64]; for(int q=0;q<nmiss;q++) eids[q]=uniq[base+missk[q]];
                pipe_dispatch(m,layer,eids,nmiss);
                m->t_edisk += now_s()-t0;           /* dispatch only; real reads hide behind matmul */
            } else { double t0=now_s();             /* ORIGINALE: blocking parallel load */
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++) expert_load(m,layer,uniq[base+missk[q]],&m->ws[q],1);
                m->t_edisk += now_s()-t0; }
        }
        /* I/O ASINCRONO: readahead (WILLNEED) del blocco SUCCESSIVO mentre calcoliamo
         * questo — il kernel legge in background, le pread dopo trovano cache calda */
        if(base+64<nu){
            int nb2 = nu-(base+64)<64 ? nu-(base+64) : 64;
            for(int j=0;j<nb2;j++){ int eid=uniq[base+64+j]; int found=0;
                ESlot *P=m->pin[layer];
                for(int z=0;z<m->npin[layer] && !found;z++) if(P[z].eid==eid) found=1;
                ESlot *Sl=m->ecache[layer];
                for(int z=0;z<m->ecn[layer] && !found;z++) if(Sl[z].eid==eid) found=1;
                if(!found) expert_prefetch(m,layer,eid);
            }
        }
#ifdef COLI_CUDA
        ESlot *group_e[64]; int group_n[64]; int ngroup=0;
#endif
#ifdef COLI_METAL
        if(g_metal_enabled){
            /* PIPE drain. Two reasons this barrier is mandatory here, and not optional:
             *  1) MB_BUILD(1) hands the missed experts' slabs straight to the GPU — a slot still
             *     being pread by an I/O worker would be matmul-ed half-loaded.
             *  2) PIPE's only drain barrier is the per-expert pipe_wait() in the CPU matmul loop
             *     below, which metal_done SKIPS ENTIRELY. Without this, a still-writing worker
             *     would race the end-of-block LRU swap that recycles ws[].
             * pipe_wait() is an idempotent spin on ready[q], so the per-expert waits below stay
             * correct (and free) when a subset falls back to the CPU. */
            if(g_pipe && nmiss){ double tw=now_s();
                for(int q=0;q<nmiss;q++) pipe_wait(q);
                m->t_edisk += now_s()-tw; }
            MB_BUILD(1, 0);                                   /* missed experts, now loaded */
            if(nbb>0){
                double t0=now_s();
                if(coli_metal_moe_block(nbb,D,I,mfmt,MG,MU,MD,MGS,MUS,MDS,mxg,xoffb,nrb,mrows,mrw,out,S)) cpu_miss=0;
                m->t_emm += now_s()-t0;
            } else cpu_miss=0;
            if(mh){ double t0=now_s();
                if(coli_metal_moe_block_end(mh,out)){ if(mh_shared) shared_on_gpu=1; }
                else cpu_res=1;
                m->t_emm += now_s()-t0; mh=NULL; }
            metal_done = (!cpu_res && !cpu_miss);
            free(mxg); free(mrows); free(mrw);
        }
        #undef MB_BUILD
#endif
        if(!metal_done)
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            /* Drain this miss's async load BEFORE the nr==0 early-exit below: every
             * dispatched slot must be waited before the end-of-block LRU swap can reuse
             * its ws[] slab, so correctness does not depend on the nr>=1 routing invariant.
             * Stays ABOVE the METAL skip: a subset that fell back to the CPU still needs its
             * slot drained here, and under METAL the block-level drain above already ran (this
             * spin is then a no-op). */
            if(g_pipe && qof[j]>=0){ double tw=now_s(); pipe_wait(qof[j]); m->t_edisk += now_s()-tw; }
#ifdef COLI_METAL
            /* skip the subsets already computed on GPU */
            if(g_metal_enabled && ((is_miss[j] && !cpu_miss) || (!is_miss[j] && !cpu_res))) continue;
#endif
            int nr=0;                                 /* righe (posizioni) che usano questo expert */
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
#ifdef COLI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
            if(group_enabled && g_cuda_enabled && e->g.cuda_eligible && e->u.cuda_eligible && e->d.cuda_eligible &&
               !omp_in_parallel()){
                group_e[ngroup]=e; group_n[ngroup]=nr;
                for(int r=0;r<nr;r++){ group_row[(int64_t)ngroup*S+r]=rows[r]; group_weight[(int64_t)ngroup*S+r]=rw[r]; }
                ngroup++; continue;
            }
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
#ifdef COLI_CUDA
            if(!group_enabled && g_cuda_enabled && e->g.cuda_eligible && e->u.cuda_eligible &&
               e->d.cuda_eligible && !omp_in_parallel() &&
               coli_cuda_expert_mlp(e->g.cuda,e->u.cuda,e->d.cuda,hh,xg,nr)){
                for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D,wgt=rw[r],*hr=hh+(int64_t)r*D;
                    for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
                m->t_emm+=now_s()-t0; continue;
            }
            if(!e->slab) expert_host_ensure(m,layer,e);
#endif
            expert_gate_up(gg,uu,xg,&e->g,&e->u,nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            matmul_qt(hh, gg, &e->d, nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            m->t_emm += now_s()-t0;
        }
#ifdef COLI_CUDA
        ColiCudaTensor *dev_g[COLI_CUDA_MAX_DEVICES][64],*dev_u[COLI_CUDA_MAX_DEVICES][64];
        ColiCudaTensor *dev_d[COLI_CUDA_MAX_DEVICES][64];
        int dev_rows[COLI_CUDA_MAX_DEVICES][64],dev_which[COLI_CUDA_MAX_DEVICES][64];
        int dev_nc[COLI_CUDA_MAX_DEVICES]={0},dev_total[COLI_CUDA_MAX_DEVICES]={0};
        int dev_off[COLI_CUDA_MAX_DEVICES]={0},dev_ok[COLI_CUDA_MAX_DEVICES]={0};
        for(int di=0;di<g_cuda_ndev;di++) for(int q=0;q<ngroup;q++)
            if(group_e[q]->g.cuda_device==g_cuda_devices[di]) dev_total[di]+=group_n[q];
        for(int di=1;di<g_cuda_ndev;di++) dev_off[di]=dev_off[di-1]+dev_total[di-1];
        for(int di=0;di<g_cuda_ndev;di++){
            int cursor=0,device=g_cuda_devices[di];
            for(int q=0;q<ngroup;q++) if(group_e[q]->g.cuda_device==device){
                int nc=dev_nc[di]++; ESlot *e=group_e[q];
                dev_g[di][nc]=e->g.cuda; dev_u[di][nc]=e->u.cuda; dev_d[di][nc]=e->d.cuda;
                dev_rows[di][nc]=group_n[q]; dev_which[di][nc]=q;
                for(int r=0;r<group_n[q];r++) memcpy(group_x+(int64_t)(dev_off[di]+cursor+r)*D,
                    x+(int64_t)group_row[(int64_t)q*S+r]*D,D*sizeof(float));
                cursor+=group_n[q];
            }
        }
        double tg=now_s();
        #pragma omp parallel for if(g_cuda_ndev>1) schedule(static)
        for(int di=0;di<g_cuda_ndev;di++) if(dev_nc[di])
            dev_ok[di]=coli_cuda_expert_group(dev_g[di],dev_u[di],dev_d[di],dev_rows[di],dev_nc[di],
                group_y+(int64_t)dev_off[di]*D,group_x+(int64_t)dev_off[di]*D);
        for(int di=0;di<g_cuda_ndev;di++){
            int off=dev_off[di];
            for(int q=0;q<dev_nc[di];q++){
                int gi=dev_which[di][q],nr=group_n[gi]; ESlot *e=group_e[gi];
                if(!dev_ok[di]){
                    for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D,x+(int64_t)group_row[(int64_t)gi*S+r]*D,D*sizeof(float));
                    if(!coli_cuda_expert_mlp(e->g.cuda,e->u.cuda,e->d.cuda,hh,xg,nr)){
                        expert_host_ensure(m,layer,e);
                        expert_gate_up(gg,uu,xg,&e->g,&e->u,nr);
                        for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
                        matmul_qt(hh,gg,&e->d,nr);
                    }
                }
                float *src=dev_ok[di]?group_y+(int64_t)off*D:hh;
                for(int r=0;r<nr;r++){ float *os=out+(int64_t)group_row[(int64_t)gi*S+r]*D,wgt=group_weight[(int64_t)gi*S+r];
                    for(int d=0;d<D;d++) os[d]+=wgt*src[(int64_t)r*D+d]; }
                off+=nr;
            }
        }
        m->t_emm+=now_s()-tg;
#endif
        /* No drain barrier: the per-expert pipe_wait(qof[j]) above (issued for every
         * dispatched miss slot, before the nr==0 skip) already waited on all ws[] loads
         * for this block, so they are complete before the LRU swap — and the gen-tagged
         * cursor keeps any still-spinning worker off a wrong-generation slot. */
        { ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];   /* promozione LRU (swap buffer) */
          int promo = nmiss<m->ecap ? nmiss : m->ecap;
          for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
              if(*nn<m->ecap) dst=&Sl[(*nn)++];
              else { int lru=0; for(int z=1;z<*nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp; dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED); }
        }
    }
    /* ---- FASE E: shared expert, un matmul a S righe (skipped se fuso nel blocco GPU) ---- */
    float *sg=falloc((int64_t)S*sI), *su=falloc((int64_t)S*sI);
#ifdef COLI_METAL
    if(g_pre_sh){ for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=g_pre_sh[z]; shared_on_gpu=1; }
#endif
    if(!shared_on_gpu){
        matmul_qt(sg, x, &l->sh_gate, S);
        matmul_qt(su, x, &l->sh_up,   S);
        for(int64_t z=0;z<(int64_t)S*sI;z++) sg[z]=siluf(sg[z])*su[z];
        matmul_qt(hh, sg, &l->sh_down, S);
        for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=hh[z];
    }
    free(logit); free(choice); free(idxs); free(ws); free(keff); free(uniq);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw); free(sg); free(su);
#ifdef COLI_CUDA
    free(group_x); free(group_y); free(group_row); free(group_weight);
#endif
}

static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g, x, &l->gate_proj, S);
    matmul_qt(u, x, &l->up_proj,   S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=siluf(g[i])*u[i];
    matmul_qt(out, g, &l->down_proj, S);
    free(g); free(u);
}

/* LOOKA: predice il top-K del router del layer `target` dallo stato h (residual stream),
 * usando la STESSA pipeline del routing vero (post_ln -> router -> sigmoid+bias, top-K).
 * kind 0 = stesso layer saltando l'attention, kind 1 = layer successivo. */
static void la_predict(Model *m, int target, const float *h, int kind){
    Cfg *c=&m->c; Layer *l=&m->L[target]; int D=c->hidden, E=c->n_experts, K=c->topk;
    float *nrm=falloc(D), *ch=falloc(E);
    rmsnorm(nrm,h,l->post_ln,D,c->eps);
    matmul(ch,nrm,l->router,1,D,E);
    for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
    int *pred=la_pred[kind][target];
    for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(pred[j]==e){tk=1;break;}
            if(!tk && ch[e]>bv){bv=ch[e];best=e;} }
        pred[kk]=best; }
    la_val[kind][target]=1;
    free(nrm); free(ch);
}

/* PILOTA: prefetch guidato dal router. Predice il top-K del layer L+1 dallo stato
 * post-attention di L (recall misurato 71.6% su GLM-5.2, vs 41.3% del token precedente)
 * e lancia il WILLNEED degli expert mancanti MENTRE il MoE di L legge i suoi: il disco
 * lavora nei tempi morti del calcolo invece di aspettare il routing vero. Con MTP attiva
 * predice per TUTTE le posizioni del draft: la speculazione pilota anche l'I/O.
 * PILOT_K limita alle prime k predizioni (la testa del ranking e' piu' affidabile
 * della coda: meno banda sprecata sulle predizioni sbagliate).
 *
 * I WILLNEED partono da un THREAD I/O dedicato: con la coda disco satura la submit
 * del fadvise BLOCCA (~0.5ms x 169k chiamate = +92s/48 token, misurato) — inline
 * il pilota costava piu' di quanto rendesse. Ring lock-free 1P/1C; pieno = scarta
 * (un hint perso non e' un errore). */
static struct { int l,e; } pilot_q[4096];
static volatile unsigned pilot_w=0, pilot_r=0;
static Model *pilot_m=NULL;
/* PILOT_REAL: load VERO dell'expert predetto dentro la LRU del layer FUTURO. Vedi
 * l'invariante di sicurezza accanto a g_pilot_real. Il pread (lento) gira FUORI dal lock;
 * il lock protegge solo la scelta/pubblicazione dello slot e l'handshake col main. */
static void pilot_realload(Model *m, int layer, int eid){
    pthread_mutex_lock(&g_pilot_mx);
    if(layer <= atomic_load_explicit(&g_cur_moe_layer,memory_order_acquire)){
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
        pthread_mutex_unlock(&g_pilot_mx); return;      /* il main possiede gia' questo layer */
    }
    ESlot *P=m->pin[layer];                             /* gia' residente (pin o ecache)? skip */
    for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ pthread_mutex_unlock(&g_pilot_mx); return; }
    ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
    for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ pthread_mutex_unlock(&g_pilot_mx); return; }
    int slot,isnew;                                     /* cresci se c'e' posto, altrimenti LRU */
    if(nn<m->ecap){ slot=nn; isnew=1; }
    else { int lru=0; for(int z=1;z<nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; slot=lru; isnew=0; }
    ESlot *dst=&Sl[slot];
    dst->eid=-1;                                        /* nascondi dagli scan-hint mentre carica */
    atomic_store_explicit(&g_pilot_inflight,layer,memory_order_release);
    pthread_mutex_unlock(&g_pilot_mx);

    int rc=expert_load(m,layer,eid,dst,0);              /* pread VERO — fuori dal lock, sovrapposto al compute; fatal=0: un errore su una speculazione NON deve uccidere il server */

    pthread_mutex_lock(&g_pilot_mx);
    if(rc==0){
        dst->used=(uint64_t)__atomic_add_fetch(&m->eclock,1,__ATOMIC_RELAXED);
        if(isnew) m->ecn[layer]=slot+1;                 /* pubblica lo slot SOLO ora che eid e' valido */
        atomic_fetch_add_explicit(&g_pilot_loads,1,memory_order_relaxed);
    } else {
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed); /* load fallito: slot resta nascosto (eid=-1), mai pubblicato */
    }
    atomic_store_explicit(&g_pilot_inflight,-1,memory_order_release);
    pthread_mutex_unlock(&g_pilot_mx);
    if(rc!=0)                                            /* mai swallow silenzioso: logga (una riga) e prosegui */
        fprintf(stderr,"[PILOT] load speculativo abbandonato: layer %d expert %d (I/O error/short read) — nessun impatto sull'output\n",layer,eid);
}
static void *pilot_worker(void *arg){
    (void)arg;
    for(;;){
        unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
        unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
        if(r==w){ usleep(200); continue; }
        if(g_pilot_real) pilot_realload(pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        else             expert_prefetch(pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        __atomic_store_n(&pilot_r,r+1,__ATOMIC_RELEASE);
    }
    return NULL;
}
/* parse .coli_pairs (see tools/route_pairs.py): "COLIPAIRS 1 <n>" then
 * "<L> <dL> <e> f:c f:c ..." lines. Needs c->n_experts/n_layers -> called post-init. */
static void couple_load(Model *m, const char *path){
    Cfg *c=&m->c; int E=c->n_experts, NL=c->n_layers;
    FILE *f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[COUPLE] cannot open %s\n",path); return; }
    char magic[16]; int ver=0; long n=0;
    if(fscanf(f,"%15s %d %ld",magic,&ver,&n)!=3 || strcmp(magic,"COLIPAIRS") || ver!=1){
        fprintf(stderr,"[COUPLE] %s: bad header\n",path); fclose(f); return; }
    size_t cells=(size_t)NL*2*E*CP_M;
    cp_pred=malloc(cells*sizeof(int16_t)); cp_cnt=calloc(cells,sizeof(float));
    if(!cp_pred||!cp_cnt){ fprintf(stderr,"[COUPLE] OOM\n"); free(cp_pred); free(cp_cnt); cp_pred=NULL; fclose(f); return; }
    for(size_t i=0;i<cells;i++) cp_pred[i]=-1;
    long used=0;
    char *ln=NULL; size_t lcap=0;
    while(getline(&ln,&lcap,f)>0){          /* line-based: a malformed line cannot eat the next */
        char *p=ln; int L,dL,e; int nc=0;
        if(sscanf(p,"%d %d %d%n",&L,&dL,&e,&nc)!=3) continue;
        p+=nc;
        if(L<0||L>=NL||(dL!=1&&dL!=2)||e<0||e>=E) continue;
        size_t base=((size_t)(L*2+(dL-1))*E+e)*CP_M;
        int j=0;
        while(j<CP_M){
            int fe; float fc;
            if(sscanf(p," %d:%f%n",&fe,&fc,&nc)!=2) break;
            p+=nc;
            if(fe>=0&&fe<E){ cp_pred[base+j]=(int16_t)fe; cp_cnt[base+j]=fc; j++; }
        }
        if(j) used++;
    }
    free(ln);
    fclose(f);
    g_couple=1;
    fprintf(stderr,"[COUPLE] %s: %ld conditioning entries, K=%d depth=%d\n",path,used,g_couple_k,g_couple_d);
}
/* score + enqueue: called from moe() after FASE A with the position's routed set */
static void couple_prefetch(Model *m, int layer, const int *idx, int Ke){
    Cfg *c=&m->c; int E=c->n_experts;
    if(E>512) return;
    if(!pilot_m){ pilot_m=m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    for(int dL=1; dL<=g_couple_d; dL++){
        int lt=layer+dL;
        if(lt>=c->n_layers || !m->L[lt].sparse) continue;
        float sc[512]; memset(sc,0,(size_t)E*sizeof(float));
        for(int kk=0;kk<Ke;kk++){
            size_t base=((size_t)(layer*2+(dL-1))*E+idx[kk])*CP_M;
            for(int j=0;j<CP_M && cp_pred[base+j]>=0;j++) sc[cp_pred[base+j]]+=cp_cnt[base+j];
        }
        for(int kk=0;kk<g_couple_k;kk++){
            int best=-1; float bv=0;
            for(int e=0;e<E;e++) if(sc[e]>bv){bv=sc[e];best=e;}
            if(best<0) break;
            sc[best]=0;
            int found=0;                            /* residency scan, same locking as pilot */
            pthread_mutex_lock(&g_pilot_mx);
            ESlot *P=m->pin[lt];
            for(int z=0;z<m->npin[lt] && !found;z++) if(P[z].eid==best) found=1;
            ESlot *Sl=m->ecache[lt];
            for(int z=0;z<m->ecn[lt] && !found;z++) if(Sl[z].eid==best) found=1;
            pthread_mutex_unlock(&g_pilot_mx);
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    pilot_q[w&4095].l=lt; pilot_q[w&4095].e=best;
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                    g_cp_enq++;
                }
            }
        }
    }
}
static void pilot_prefetch(Model *m, int lnext, const float *x, int S){
    Cfg *c=&m->c; Layer *l=&m->L[lnext]; int D=c->hidden, E=c->n_experts;
    int K = g_pilot_k<c->topk ? g_pilot_k : c->topk;
    if(!pilot_m){ pilot_m=m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    float *nrm=falloc(D), *ch=falloc(E);
    for(int s=0;s<S;s++){
        rmsnorm(nrm, x+(int64_t)s*D, l->post_ln, D, c->eps);
        matmul(ch, nrm, l->router, 1, D, E);
        for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
        for(int kk=0;kk<K;kk++){
            int best=0; for(int e=1;e<E;e++) if(ch[e]>ch[best]) best=e;
            ch[best]=-2e30f;
            /* Residency scan of the FUTURE layer lnext under g_pilot_mx: with
             * PILOT_REAL=1 the pilot worker mutates ecache[lnext]/ecn[lnext]
             * concurrently, so read them under the same lock (Option A). Decide
             * under the lock, then enqueue AFTER unlocking — the pilot_q ring is
             * lock-free (pilot_w/pilot_r atomics, not g_pilot_mx) so there is no
             * re-entrant double-lock, and the worker re-checks residency under the
             * lock anyway, making a racing redundant enqueue harmless. */
            int found=0;
            pthread_mutex_lock(&g_pilot_mx);
            ESlot *P=m->pin[lnext];
            for(int z=0;z<m->npin[lnext] && !found;z++) if(P[z].eid==best) found=1;
            ESlot *Sl=m->ecache[lnext];
            for(int z=0;z<m->ecn[lnext] && !found;z++) if(Sl[z].eid==best) found=1;
            pthread_mutex_unlock(&g_pilot_mx);
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    pilot_q[w&4095].l=lnext; pilot_q[w&4095].e=best;
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                }
            }
        }
    }
    free(nrm); free(ch);
}

/* forward di UN layer (usato dai 78 principali e dal layer MTP) */
static void layer_forward_rows(Model *m, Layer *l, int li, float *x, int S, int pos_base,
                               KVState *const *kvs, const int *positions, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++) expert_prefetch(m,li,m->eroute[li][z]);
    if(g_looka && S==1 && li<c->n_layers && l->sparse) la_predict(m,li,x,0);
#ifdef COLI_METAL
    /* FULL-LAYER CB: in_ln + attention + residuo + post_ln + shared expert + router/top-K
     * in un solo submit GPU; la CPU legge il routing e fa solo resolve/disk/expert-CB.
     * Fallback: qualsiasi condizione mancante -> percorso CPU intero qui sotto. */
    if(g_metal_enabled && S<=4 && li<c->n_layers && l->sparse
       && (g_absorb==1||(g_absorb<0&&S<=4)) && m->kv_start[li]==0
       && D==6144 && c->n_heads==64 && c->q_lora==2048 && c->kv_lora==512
       && c->qk_nope==192 && c->qk_rope==64 && c->v_head==256 && l->kv_b.fmt==2
       && c->n_experts==256 && c->topk==8 && c->n_shared==1 && c->moe_inter==2048){
        int sel_active = m->has_dsa && c->idx_type[li] && (pos_base+S) > c->index_topk;
        if(!sel_active){
            static float *linrm,*lnrm,*lsh,*lw; static int *lidx,*lkeff;
            if(!linrm){ linrm=falloc(4*(int64_t)D); lnrm=falloc(4*(int64_t)D); lsh=falloc(4*(int64_t)D);
                        lidx=malloc(4*8*sizeof(int)); lw=malloc(4*8*sizeof(float)); lkeff=malloc(4*sizeof(int)); }
            int Ksel = g_topk>0 ? (g_topk<8?g_topk:8) : 8;
            float tp = (g_topp>0 && g_topp<1.f) ? g_topp : 0.f;
            double ta0=now_s();
            #define WP_(q) ((q).fmt==1?(const void*)(q).q8:(const void*)(q).q4)
            int ok = coli_metal_layer_decode(x, l->in_ln, l->post_ln,
                WP_(l->q_a), l->q_a.s, l->q_a.fmt, l->q_a_ln,
                WP_(l->q_b), l->q_b.s, l->q_b.fmt,
                WP_(l->kv_a), l->kv_a.s, l->kv_a.fmt, l->kv_a_ln,
                WP_(l->kv_b), l->kv_b.s, l->kv_b.fmt,
                WP_(l->o), l->o.s, l->o.fmt,
                WP_(l->sh_gate), l->sh_gate.s, l->sh_gate.fmt,
                WP_(l->sh_up),   l->sh_up.s,   l->sh_up.fmt,
                WP_(l->sh_down), l->sh_down.s, l->sh_down.fmt,
                l->router, l->router_bias,
                c->n_experts, c->topk, Ksel, tp, c->norm_topk, c->routed_scale,
                m->Lc[li], m->Rc[li], S, pos_base, m->kv_start[li],
                c->eps, c->theta, c->attn_scale,
                linrm, lnrm, lsh, lidx, lw, lkeff);
            #undef WP_
            if(ok){
                m->t_attn += now_s()-ta0;
                if(m->has_dsa && c->idx_type[li]){            /* index key per selezioni future */
                    for(int s=0;s<S;s++){ int pos=pos_base+s;
                        float *kd=m->Ic[li]+(int64_t)pos*c->index_hd;
                        matmul_qt(kd, linrm+(int64_t)s*D, &m->ix_wk[li], 1);
                        layernorm(kd, m->ix_knw[li], m->ix_knb[li], c->index_hd, 1e-6f);
                        rope_interleave(kd, pos, c);
                    }
                }
                if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
                if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse) la_predict(m,li+1,x,1);
                g_pre_idx=lidx; g_pre_w=lw; g_pre_keff=lkeff; g_pre_sh=lsh;
                moe(m,l,li,lnrm,S,tmp);
                g_pre_idx=NULL; g_pre_w=NULL; g_pre_keff=NULL; g_pre_sh=NULL;
                for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
                return;
            }
        }
    }
#endif
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention_rows(m,l,li,nrm,S,pos_base,kvs,positions,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
    if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse) la_predict(m,li+1,x,1);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp); else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    layer_forward_rows(m,l,li,x,S,pos_base,NULL,NULL,nrm,tmp);
}
static void layers_forward_rows(Model *m, float *x, int S, int pos_base,
                                KVState *const *kvs, const int *positions){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_pilot_real){   /* nuovo forward: il possesso-layer riparte da -1 (i layer si rifanno da 0) */
        pthread_mutex_lock(&g_pilot_mx);
        atomic_store_explicit(&g_cur_moe_layer,-1,memory_order_release);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    for(int i=0;i<c->n_layers;i++){
        /* progresso su stderr per i batch grossi (prefill): il primo byte di risposta
         * puo' arrivare dopo MINUTI di streaming — al buio sembra un blocco. */
        if(S>=8 && (i%4==0 || i==c->n_layers-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token\n", i+1, c->n_layers, S);
        layer_forward_rows(m,&m->L[i],i,x,S,pos_base,kvs,positions,nrm,tmp);
    }
    free(nrm); free(tmp);
}
static void layers_forward(Model *m, float *x, int S, int pos_base){
    layers_forward_rows(m,x,S,pos_base,NULL,NULL);
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    KVState *k=m->kv;
    if(k->Lc){ for(int i=0;i<c->n_layers+1;i++){
#ifdef COLI_METAL
        if(g_metal_enabled){ coli_metal_unregister(k->Lc[i]); coli_metal_unregister(k->Rc[i]); }
#endif
        free(k->Lc[i]); free(k->Rc[i]); } free(k->Lc); free(k->Rc); }
    if(k->Ic){ for(int i=0;i<c->n_layers;i++) free(k->Ic[i]); free(k->Ic); k->Ic=NULL; }
    if(m->has_dsa){
        k->Ic=calloc(c->n_layers,sizeof(float*));
        for(int i=0;i<c->n_layers;i++) if(c->idx_type[i]) k->Ic[i]=falloc((int64_t)max_t*c->index_hd);
    }
    k->max_t=max_t;
    int NR=c->n_layers+1;                        /* riga extra: KV del layer MTP */
    k->Lc=calloc(NR,sizeof(float*)); k->Rc=calloc(NR,sizeof(float*));
    for(int i=0;i<NR;i++){ k->Lc[i]=falloc((int64_t)max_t*c->kv_lora);
        k->Rc[i]=falloc((int64_t)max_t*c->qk_rope);
#ifdef COLI_METAL
        /* page-align + register Lc/Rc for zero-copy GPU attention. falloc isn't 16K-aligned,
         * so re-allocate aligned and register the exact byte length. */
        if(g_metal_enabled){
            size_t lb=(((size_t)max_t*c->kv_lora*sizeof(float))+16383)&~(size_t)16383;
            size_t rb=(((size_t)max_t*c->qk_rope*sizeof(float))+16383)&~(size_t)16383;
            free(k->Lc[i]); free(k->Rc[i]); void *lp,*rp;
            if(posix_memalign(&lp,16384,lb)||posix_memalign(&rp,16384,rb)){fprintf(stderr,"OOM kv\n");exit(1);}
            k->Lc[i]=lp; k->Rc[i]=rp;
            coli_metal_register(k->Lc[i],lb); coli_metal_register(k->Rc[i],rb);
        }
#endif
    }
    m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic; m->max_t=k->max_t; m->kv_start=k->kv_start;
}

static void kv_bind(Model *m, KVState *k){
    m->kv=k; m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic;
    m->max_t=k->max_t; m->kv_start=k->kv_start;
}

static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base);
static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    if(m->has_mtp && S>=2 && g_draft>0) mtp_absorb(m, ids+1, x, S-1, pos_base);
    float *last=falloc(D); rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    double th0=now_s();
    float *logit=falloc(c->vocab); matmul_qt_ex(logit,last,&m->lm_head,1,!g_lmhead_exact);
    m->t_head += now_s()-th0;
    free(x); free(last); return logit;
}

/* come step(), ma ritorna i logits di TUTTE le S posizioni [S,vocab] (per la verifica spec) */
static float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->h_all) memcpy(m->h_all, x, (int64_t)S*D*sizeof(float));   /* hidden di TUTTE le pos (S<=64) */
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    float *lo=falloc((int64_t)S*c->vocab), *row=falloc(D);
    for(int s=0;s<S;s++){ rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt_ex(lo+(int64_t)s*c->vocab, row, &m->lm_head, 1, !g_lmhead_exact); }
    free(x); free(row); return lo;
}

/* One decode token from each independent sequence, evaluated as a single MoE
 * batch.  Prefill and speculative batches retain their contiguous-KV path. */
static float *step_decode_batch(Model *m, const DecodeRow *rows, int S){
    Cfg *c=&m->c; int D=c->hidden;
    /* Ragged KV currently uses MLA absorption; the stack kernel is sized to 512. */
    if(!rows || S<1 || S>64 || c->kv_lora>512) return NULL;
    KVState *kvs[64]; int positions[64];
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++){
        if(!rows[s].kv || !rows[s].kv->Lc || !rows[s].kv->Rc || !rows[s].kv->kv_start ||
           rows[s].token<0 || rows[s].token>=c->vocab ||
           rows[s].pos<0 || rows[s].pos>=rows[s].kv->max_t){
            free(x); return NULL;
        }
        for(int l=0;l<c->n_layers;l++){
            if(!rows[s].kv->Lc[l] || !rows[s].kv->Rc[l] ||
               rows[s].kv->kv_start[l]<0 || rows[s].kv->kv_start[l]>rows[s].pos ||
               (m->has_dsa && c->idx_type[l] &&
                (!rows[s].kv->Ic || !rows[s].kv->Ic[l]))){ free(x); return NULL; }
        }
        for(int p=0;p<s;p++) if(rows[p].kv==rows[s].kv){ free(x); return NULL; }
        kvs[s]=rows[s].kv; positions[s]=rows[s].pos;
        embed_row(m,rows[s].token,x+(int64_t)s*D);
    }
    layers_forward_rows(m,x,S,0,kvs,positions);
    float *norm=falloc((int64_t)S*D);
    for(int s=0;s<S;s++)
        rmsnorm(norm+(int64_t)s*D,x+(int64_t)s*D,m->final_norm,D,c->eps);
    double th0=now_s();
    float *logit=falloc((int64_t)S*c->vocab);
    matmul_qt_ex(logit,norm,&m->lm_head,S,!g_lmhead_exact);
    m->t_head+=now_s()-th0;
    free(x); free(norm);
    return logit;
}

/* METODO E — prompt-lookup: cerca l'occorrenza piu' recente dell'ultimo bigramma nel
 * contesto e propone i token che la seguirono. Zero pesi extra, zero costo: e' solo
 * un'ipotesi che il modello verifichera'. */
static int ngram_draft(const int *ids, int len, int G, int *draft){
    if(len<4 || G<1) return 0;
    int a=ids[len-2], b=ids[len-1];
    for(int i=len-3;i>=1;i--)
        if(ids[i-1]==a && ids[i]==b){
            int n=0; for(int j=i+1;j<len && n<G;j++) draft[n++]=ids[j];
            return n;
        }
    return 0;
}

/* METODO MTP: propone fino a G draft con la testa multi-token nativa di GLM-5.2.
 * Input: next_tok (appena emesso, posizione kv) e hlast (hidden pre-norm della pos kv-1).
 * Catena DeepSeek-V3: h' = Layer78( eh_proj[ enorm(emb(tok)) ; hnorm(h) ] ),
 * draft = argmax(lm_head(shared_head.norm(h'))). La KV del layer MTP vive alla riga n_layers
 * ed e' valida da kv_start (niente prefill: finestra di solo-decode, basta per il draft). */
static int mtp_argmax(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}
static int mtp_draft(Model *m, int next_tok, int kv, int G, int *draft){
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    int p=kv-1; if(p<0||G<1) return 0;
    if(m->kv_start[li]<0 || m->kv_start[li]>p) m->kv_start[li]=p;
    float *x=falloc(D), *cat=falloc(2*D), *hx=falloc(D), *nrm=falloc(D), *tmp=falloc(D);
    float *row=falloc(D), *logit=falloc(c->vocab), *h=falloc(D);
    memcpy(h, m->hlast, D*sizeof(float));
    int tok=next_tok, n=0;
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int g=0; g<G; g++){
        int pos=p+g; if(pos+2>=m->max_t) break;
        embed_row(m, tok, x);
        rmsnorm(x, x, m->enorm, D, c->eps);
        if(g==0 && !prenorm) rmsnorm(h, h, m->final_norm, D, c->eps);  /* h vero: post model.norm */
        rmsnorm(h, h, m->hnorm, D, c->eps);
        if(getenv("MTP_SWAP")){ memcpy(cat, h, D*sizeof(float)); memcpy(cat+D, x, D*sizeof(float)); }
        else { memcpy(cat, x, D*sizeof(float)); memcpy(cat+D, h, D*sizeof(float)); }
        matmul_qt(hx, cat, &m->eh_proj, 1);
        double n_eh=0; for(int d=0;d<D;d++) n_eh+=hx[d]*hx[d];
        int dbg = getenv("MTP_DEBUG") && atoi(getenv("MTP_DEBUG"))>=2;
        int t_pre=-1;
        if(dbg){ rmsnorm(row, hx, m->mtp_norm, D, c->eps); matmul_qt_ex(logit, row, &m->lm_head, 1, !g_lmhead_exact);
                 t_pre=mtp_argmax(logit, c->vocab); }
        layer_forward(m, &m->mtpL, li, hx, 1, pos, nrm, tmp);
        double n_post=0; for(int d=0;d<D;d++) n_post+=hx[d]*hx[d];
        rmsnorm(row, hx, m->mtp_norm, D, c->eps);
        matmul_qt_ex(logit, row, &m->lm_head, 1, !g_lmhead_exact);
        int t2=mtp_argmax(logit, c->vocab);
        if(dbg) fprintf(stderr,"[mtp2] pos=%d in_tok=%d ||eh||=%.1f ||post||=%.1f pre_blk=%d post_blk=%d\n",
                        pos, tok, sqrt(n_eh), sqrt(n_post), t_pre, t2);
        draft[n++]=t2; tok=t2; memcpy(h, hx, D*sizeof(float));
    }
    free(x); free(cat); free(hx); free(nrm); free(tmp); free(row); free(logit); free(h);
    return n;
}
/* assorbe nella KV della testa MTP le coppie VERIFICATE (emb(token@pos+1), h_vero@pos):
 * next_ids[i] = token alla posizione pos_base+i+1; x[i] = hidden VERO a pos_base+i.
 * Un solo passaggio batch del layer MTP (il batch-union rende economici gli expert). */
static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base){
    if(!m->has_mtp || S<1) return;
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    if(m->kv_start[li]<0 || m->kv_start[li]>pos_base) m->kv_start[li]=pos_base;
    float *hx=falloc((int64_t)S*D), *cat=falloc(2*D), *e=falloc(D), *hn=falloc(D), *hf=falloc(D);
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int i=0;i<S;i++){
        embed_row(m,next_ids[i],e);
        rmsnorm(e,e,m->enorm,D,c->eps);
        if(prenorm) rmsnorm(hn,x+(int64_t)i*D,m->hnorm,D,c->eps);
        else { rmsnorm(hf,x+(int64_t)i*D,m->final_norm,D,c->eps);   /* vLLM: h POST model.norm */
               rmsnorm(hn,hf,m->hnorm,D,c->eps); }
        if(getenv("MTP_SWAP")){ memcpy(cat,hn,D*sizeof(float)); memcpy(cat+D,e,D*sizeof(float)); }
        else { memcpy(cat,e,D*sizeof(float)); memcpy(cat+D,hn,D*sizeof(float)); }
        matmul_qt(hx+(int64_t)i*D, cat, &m->eh_proj, 1);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    layer_forward(m,&m->mtpL,li,hx,S,pos_base,nrm,tmp);
    free(hx); free(cat); free(e); free(hn); free(hf); free(nrm); free(tmp);
}

static inline int argmax_v(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}

/* ---- METODO F: draft grammaticale (#48) ----
 * gr_feed consuma i byte di ogni token EMESSO e tiene il walker in sync con l'output;
 * grammar_draft propone lo span FORZATO successivo (un solo byte legale per posizione)
 * gia' tokenizzato. Il confine di tokenizzazione non e' garantito coincidere con quello
 * del modello: la verifica assorbe la differenza (al peggio l'ultimo draft e' rifiutato). */
static void grammar_setup(Tok *T){
    const char *gf=getenv("GRAMMAR"); if(!gf||!*gf) return;
    FILE *f=fopen(gf,"rb");
    if(!f){ fprintf(stderr,"[GRAMMAR] cannot open %s\n",gf); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *txt=malloc((size_t)n+1);
    if(!txt || fread(txt,1,(size_t)n,f)!=(size_t)n){
        fprintf(stderr,"[GRAMMAR] failed to read %s\n",gf); fclose(f); free(txt); return; }
    fclose(f); txt[n]=0;
    if(gr_parse(&g_gram,txt)){ fprintf(stderr,"[GRAMMAR] %s: %s\n",gf,g_gram.err); free(txt); return; }
    free(txt);
    gr_state_init(&g_gst,&g_gram);
    if(!g_gst.alive){ fprintf(stderr,"[GRAMMAR] %s: grammar cannot be evaluated (left recursion?)\n",gf); return; }
    if(getenv("GRAMMAR_DRAFT")) g_gr_max=atoi(getenv("GRAMMAR_DRAFT"));
    if(g_gr_max<1) g_gr_max=1;
    if(g_gr_max>48) g_gr_max=48;
    g_gr_T=T; g_gr_on=1;
    fprintf(stderr,"[GRAMMAR] %s: %d rules, forced span capped at %d tokens/forward\n",gf,g_gram.n,g_gr_max);
}
/* stato pulito all'inizio di ogni RISPOSTA (non tra i \x02MORE, che continuano) */
static void grammar_reset(void){
    if(!g_gr_on) return;
    gr_state_init(&g_gst,&g_gram); g_gr_armed=0;
    if(!g_gst.alive) g_gr_on=0;
}
/* consuma i byte di un token emesso. Preambolo (prima dell'arming): ignorato.
 * Desync dopo l'arming: si riarma in attesa del prossimo inizio valido — al peggio
 * i draft vengono rifiutati dalla verifica, l'output non cambia MAI. */
static void gr_feed(int t){
    if(!g_gr_on||!g_gr_T) return;
    char b[64]; int n=tok_decode(g_gr_T,&t,1,b,63);
    for(int i=0;i<n;i++){
        int r=gr_accept(&g_gst,(unsigned char)b[i]);
        if(r==1){ g_gr_armed=1; continue; }
        if(r<0){ g_gr_on=0; return; }                 /* walker spento: fine dei draft */
        if(!g_gr_armed) continue;                     /* preambolo: aspetta l'inizio */
        gr_state_init(&g_gst,&g_gram); g_gr_armed=0;  /* desync: riparti dalla radice */
        if(!g_gst.alive){ g_gr_on=0; return; }
        if(gr_accept(&g_gst,(unsigned char)b[i])==1) g_gr_armed=1;
    }
}
/* propone lo span forzato come token (max cap); 0 se la grammatica dirama qui */
static int grammar_draft(int *draft, int cap){
    if(!g_gr_on||!g_gr_armed||!g_gr_T||cap<1) return 0;
    if(g_gr_prop>=32 && g_gr_acc*2<g_gr_prop){        /* guardia adattiva, come per MTP:
        acceptance sotto il 50% = tokenizzazione fuori asse, meglio spegnersi */
        g_gr_on=0;
        fprintf(stderr,"[GRAMMAR] %.0f%% acceptance after %llu proposals: grammar drafts disabled\n",
            100.0*g_gr_acc/g_gr_prop,(unsigned long long)g_gr_prop);
        return 0;
    }
    char fb[512]; int nb=gr_forced(&g_gst,fb,(int)sizeof fb-1);
    if(nb<=0) return 0;
    int g=tok_encode(g_gr_T,fb,nb,draft,cap);
    return g>0?g:0;
}

/* ---- SAMPLING (temperatura + nucleus) con verifica speculativa LOSSLESS ----
 * Il draft (MTP/n-gram) e' DETERMINISTICO (argmax della testa): q = massa puntuale.
 * Rejection sampling di Leviathan: accetta il draft x_d con prob p(x_d); al rifiuto
 * ricampiona da p con x_d azzerato e rinormalizzato. La distribuzione risultante e'
 * ESATTAMENTE p: la speculazione resta invisibile all'output anche col sampling. */
static uint64_t g_rng=0x9E3779B97F4A7C15ULL;
static inline double rndu(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17;
    return (double)(g_rng>>11)*(1.0/9007199254740992.0); }
static float *g_pbuf=NULL; static int *g_pidx=NULL;   /* buffer riusati (decode single-thread) */
static int cmp_pdesc(const void *a,const void *b){
    float pa=g_pbuf[*(const int*)a], pb=g_pbuf[*(const int*)b];
    return pa<pb ? 1 : pa>pb ? -1 : 0; }
/* costruisce in g_pbuf la distribuzione target: softmax(lo/temp) troncata a top-p g_nuc */
static void dist_build(const float *lo, int V){
    if(!g_pbuf){ g_pbuf=falloc(V); g_pidx=malloc(V*sizeof(int)); }
    float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double s=0; float invt=1.f/(g_temp>1e-4f?g_temp:1e-4f);
    for(int i=0;i<V;i++){ g_pbuf[i]=expf((lo[i]-mx)*invt); s+=g_pbuf[i]; }
    for(int i=0;i<V;i++) g_pbuf[i]/=(float)s;
    if(g_nuc>0 && g_nuc<1.f){
        for(int i=0;i<V;i++) g_pidx[i]=i;
        qsort(g_pidx,V,sizeof(int),cmp_pdesc);
        double cum=0; int keep=V;
        for(int i=0;i<V;i++){ cum+=g_pbuf[g_pidx[i]]; if(cum>=g_nuc){ keep=i+1; break; } }
        double s2=0; for(int i=keep;i<V;i++) g_pbuf[g_pidx[i]]=0;
        for(int i=0;i<keep;i++) s2+=g_pbuf[g_pidx[i]];
        for(int i=0;i<keep;i++) g_pbuf[g_pidx[i]]/=(float)s2;
    }
}
/* campiona da g_pbuf; ban>=0 -> quel token e' escluso (rinormalizzando al volo) */
static int dist_sample(int V, int ban){
    double z = 1.0 - (ban>=0 ? g_pbuf[ban] : 0.0); if(z<=1e-12) z=1e-12;
    double u = rndu()*z, cum=0;
    for(int i=0;i<V;i++){ if(i==ban) continue; cum+=g_pbuf[i]; if(cum>=u) return i; }
    for(int i=V-1;i>=0;i--) if(i!=ban && g_pbuf[i]>0) return i;
    return 0;
}
/* prossimo token dai logits: greedy se g_temp<=0, altrimenti sampling.
 * ban = token escluso perche' rifiutato dalla verifica speculativa precedente. */
static int pick_tok(const float *lo, int V, int ban){
    if(g_temp<=0) return argmax_v(lo,V);
    dist_build(lo,V);
    return dist_sample(V,ban);
}

/* stop-set attivo (popolato da run_text/run_serve dal config; vuoto in validazione,
 * dove si genera un numero fisso di token da confrontare con l'oracolo) */
static int g_stop[9], g_nstop=0;
static inline int is_stop(int t){ for(int i=0;i<g_nstop;i++) if(t==g_stop[i]) return 1; return 0; }
static void stops_arm(const Cfg *c, int tok_eos){
    g_nstop=0;
    for(int i=0;i<c->n_stop;i++) g_stop[g_nstop++]=c->stop_ids[i];
    if(tok_eos>=0 && !is_stop(tok_eos)) g_stop[g_nstop++]=tok_eos;
    fprintf(stderr,"[stop] %d stop tokens:",g_nstop);
    for(int i=0;i<g_nstop;i++) fprintf(stderr," %d",g_stop[i]);
    fprintf(stderr,"\n");
}

/* decode greedy con SELF-SPECULATION n-gram: LOSSLESS (output identico al greedy puro).
 * Ogni forward verifica fino a g_draft token proposti dal contesto: i token accettati
 * costano UNA sola passata sui pesi -> disco e banda RAM ammortizzati su piu' token.
 * all: storia token (capacita' >= kv+n_new+g_draft+2), kv = token gia' in KV.
 * logit = logits della posizione kv-1 (dal prefill); viene liberato qui.
 * emit(tok,ud) per ogni token emesso. Ritorna i token emessi; *kv_out = nuova kv. */
static int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                       void (*emit)(int,void*), void *ud, int *kv_out){
    Cfg *c=&m->c; int V=c->vocab; int emitted=0, done=0;
    int draft[64]; if(g_draft>63) g_draft=63;
    int carry_ban=-1;                    /* token rifiutato dalla verifica: escluso dal resample */
    while(emitted<n_new && !done){
        int next=pick_tok(logit,V,carry_ban); carry_ban=-1; free(logit); logit=NULL;
        if((eos>=0 && next==eos) || is_stop(next)) break;
        emit(next,ud); all[kv]=next; emitted++; m->n_emit++;
        gr_feed(next);                                  /* il walker segue l'output emesso */
        if(emitted>=n_new) break;                       /* l'ultimo token non serve forwardarlo */
        int g = 0, gsrc = 0;                            /* sorgente: 1=grammatica 2=MTP/n-gram */
        if(g_gr_on){                                    /* metodo F: prima la grammatica — dove
                                                         * forza, l'acceptance e' ~1 (#48) */
            g=grammar_draft(draft,g_gr_max);
            if(g>0) gsrc=1;
        }
        if(!g && g_draft>0){
            /* auto-off adattivo: draft che non vengono mai accettati = solo tassa disco */
            if(m->has_mtp && m->mtp_prop>=24 && m->mtp_acc*10 < m->mtp_prop){
                g_draft=0;
                fprintf(stderr,"[MTP] %.0f%% acceptance after %llu proposals: drafts disabled\n",
                    100.0*m->mtp_acc/m->mtp_prop, (unsigned long long)m->mtp_prop);
            }
        }
        if(!g && g_draft>0){
            if(m->has_mtp){ g=mtp_draft(m,next,kv,g_draft,draft); m->mtp_prop+=g; if(g)gsrc=2; }
            else { g=ngram_draft(all,kv+1,g_draft,draft); if(g)gsrc=2; }
        }
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        if(gsrc==1) g_gr_prop+=(uint64_t)g;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        float *lo=step_all(m,batch,S,kv); m->n_fw++;
        int k=0;                                        /* verifica: accetta finche' coincide */
        if(g>0 && getenv("MTP_DEBUG")){ int veri=argmax_v(lo,V);
            fprintf(stderr,"[mtpdbg] draft0=%d verified=%d %s\n", draft[0], veri, draft[0]==veri?"HIT":"miss"); }
        while(k<g && emitted<n_new){
            int accept;
            if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V);          /* rejection sampling: p(draft) */
                   accept = (rndu() < g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0 && draft[k]==eos) || is_stop(draft[k])){ done=1; break; }
            emit(draft[k],ud); all[kv+1+k]=draft[k]; emitted++; m->n_emit++;
            gr_feed(draft[k]); k++;
        }
        if(gsrc==1) g_gr_acc+=(uint64_t)k;
        else if(gsrc==2 && m->has_mtp) m->mtp_acc+=k;
        if(m->has_mtp && k>=1) mtp_absorb(m, all+kv+1, m->h_all, k, kv);   /* KV MTP in sync coi verificati */
        /* hlast deve corrispondere all'ultima posizione ACCETTATA (kv+k), non a fine batch */
        if(m->h_all && k<S-1) memcpy(m->hlast, m->h_all+(int64_t)k*m->c.hidden, m->c.hidden*sizeof(float));
        kv += 1+k;                                      /* KV oltre kv e' stantia: verra' sovrascritta */
        logit=falloc(V); memcpy(logit, lo+(int64_t)k*V, V*sizeof(float)); free(lo);
    }
    if(logit) free(logit);
    if(kv_out) *kv_out=kv;
    return emitted;
}

/* emit callback: accumula in un array (validazione) */
typedef struct { int *dst; int n; } EmitStore;
static void emit_store(int t, void *ud){ EmitStore *e=(EmitStore*)ud; e->dst[e->n++]=t; }
/* emit callback: detokenizza e stampa in streaming (chat/run), con heartbeat */
typedef struct { Tok *T; Model *m; double t0; int count; int quiet; } EmitStream;
static void emit_stream(int t, void *ud){
    EmitStream *e=(EmitStream*)ud; char dec[64];
    int dn=tok_decode(e->T,&t,1,dec,63); dec[dn]=0; fputs(dec,stdout); fflush(stdout);
    if(!e->quiet && ++e->count%16==0){ double tt=e->m->hits+e->m->miss;
        fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  %.2f tok/s  %.2f tok/fw]\n", e->count,
            rss_gb(), tt?100.0*e->m->hits/tt:0.0, e->count/(now_s()-e->t0),
            e->m->n_fw?(double)e->m->n_emit/e->m->n_fw:1.0); }
}

/* teacher-forcing: un solo forward su ids[S], argmax per posizione in pred[S] */
static void forward_all(Model *m, const int *ids, int S, int *pred){
    Cfg *c=&m->c; int D=c->hidden;
    kv_alloc(m,S);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,0);
    float *lo=falloc(c->vocab);
    float *row=falloc(D);
    for(int s=0;s<S;s++){
        rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);   /* heap row (#183) */
        matmul_qt_ex(lo, row, &m->lm_head, 1, !g_lmhead_exact);   /* exact-mode knob (#152) */
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
    }
    free(x); free(lo); free(row);
}

/* log-prob (log-softmax) del token target dato il vettore di logit; *am=1 se e' l'argmax */
static double logprob_target(const float *lo, int V, int target, int *am){
    float mx=lo[0]; int best=0; for(int i=1;i<V;i++){ if(lo[i]>mx){mx=lo[i];best=i;} }
    double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
    if(am)*am=(best==target);
    return (double)(lo[target]-mx) - log(se);
}
/* modalita' SCORING per i benchmark (stile lm-eval, log-likelihood):
 * input: file con righe "<ctxlen> <contlen> <id0> .. <id_{T-1}>"  (T=ctxlen+contlen)
 * output: riga "<logprob_continuazione> <contlen> <greedy 0/1>" per richiesta.
 * Un solo forward per richiesta (teacher-forcing): niente generazione -> fattibile a bassa velocita'. */
static void run_score(Model *m, const char *path){
    Cfg *c=&m->c; int D=c->hidden;
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    int maxT=1; { char *ln=NULL; size_t cp=0;
        while(getline(&ln,&cp,f)>0){ int a,b; if(sscanf(ln,"%d %d",&a,&b)==2 && a+b>maxT) maxT=a+b; }
        free(ln); }
    kv_alloc(m,maxT);
    float *x=falloc((int64_t)maxT*D), *lo=falloc(c->vocab), *row=falloc(D);
    int *ids=malloc(maxT*sizeof(int));
    rewind(f); char *ln=NULL; size_t cp=0; int nreq=0; double t0=now_s();
    while(getline(&ln,&cp,f)>0){
        char *p=ln; int ctxlen=strtol(p,&p,10), contlen=strtol(p,&p,10), T=ctxlen+contlen;
        if(T<=0||ctxlen<1){ printf("0 0 0\n"); fflush(stdout); continue; }
        for(int i=0;i<T;i++) ids[i]=strtol(p,&p,10);
        for(int s=0;s<T;s++) embed_row(m, ids[s], x+(int64_t)s*D);
        layers_forward(m,x,T,0);
        double lp=0; int greedy=1;
        for(int pos=ctxlen-1; pos<T-1; pos++){
            rmsnorm(row, x+(int64_t)pos*D, m->final_norm, D, c->eps);
            matmul_qt_ex(lo,row,&m->lm_head,1,!g_lmhead_exact);
            int am; lp += logprob_target(lo,c->vocab,ids[pos+1],&am); if(!am) greedy=0;
        }
        printf("%.6f %d %d\n", lp, contlen, greedy); fflush(stdout);
        if(++nreq%5==0) fprintf(stderr,"[score %d req | %.1fs | RSS %.2f GB | hit %.0f%%]\n",
            nreq, now_s()-t0, rss_gb(), (m->hits+m->miss)?100.0*m->hits/(m->hits+m->miss):0.0);
    }
    free(ln); free(ids); free(x); free(lo); free(row); fclose(f);
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out){
    kv_alloc(m,np+n_new+g_draft+2);
    for(int i=0;i<np;i++) out[i]=prompt[i];
    float *logit=step(m,prompt,np,0);
    EmitStore es={out+np,0};
    spec_decode(m,out,np,n_new,-1,logit,emit_store,&es,NULL);
}

static void profile_print(Model *m, double elapsed){
    double accounted=m->t_ewait+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILE: expert-disk %.3fs service / %.3fs wait | expert-matmul %.3fs | attention %.3fs "
           "(including kvb %.3fs) | lm_head %.3fs | other %.3fs\n",
        m->t_edisk,m->t_ewait,m->t_emm,m->t_attn,m->t_kvb,m->t_head,elapsed-accounted);
    printf("ATTENTION: projection/RoPE %.3fs | score-softmax-value %.3fs | output projection %.3fs\n",
        m->t_aproj,m->t_acore,m->t_aout);
#ifdef COLI_METAL
    if(g_metal_enabled){ uint64_t ok=0,fb=0,ex=0; double su=0,gp=0,sc=0;
        coli_metal_moe_counts(&ok,&fb,&ex); coli_metal_moe_times(&su,&gp,&sc);
        { uint64_t aok=0; double aw=0,ak=0; coli_metal_attn_counts(&aok,&aw,&ak);
          if(aok){ double ks=0,gs=0; coli_metal_attn_lat(&ks,&gs);
          printf("METAL-ATTN: layer GPU %llu | gpu-wall %.2fs (kernel %.2fs | cpu-sched %.2fs gpu-sched %.2fs)\n",(unsigned long long)aok,aw,ak,ks,gs); } }
        printf("METAL: blocchi GPU %llu | fallback CPU %llu | expert su GPU %llu | setup %.2fs gpu-wall %.2fs (kernel %.2fs) scatter %.2fs\n",
               (unsigned long long)ok,(unsigned long long)fb,(unsigned long long)ex,su,gp,coli_metal_moe_kernel_time(),sc); }
#endif
}

/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\n"); return; }
    kv_alloc(m,nfull+2);
    float *logit=step(m,full,np-1,0); free(logit);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0;
    m->t_edisk=m->t_ewait=m->t_emm=m->t_attn=m->t_kvb=m->t_head=0;
    m->t_aproj=m->t_acore=m->t_aout=0;
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        logit=step(m,full+i,1,i); free(logit); steps++;
    }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
}

/* generazione reale: tokenizza PROMPT, prefill + decode greedy con stop su EOS,
 * detokenizza e stampa il testo in streaming. */
static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    Cfg *c=&m->c; char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    grammar_setup(&T);                   /* metodo F: GRAMMAR=file.gbnf (#48) */
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int cap=(int)strlen(prompt)+16; int *pids=malloc(cap*sizeof(int));
    int np=tok_encode(&T,prompt,(int)strlen(prompt),pids,cap);
    if(np<1){ fprintf(stderr,"prompt is empty after tokenization\n"); return; }
    printf("prompt: %d tokens | generating up to %d (EOS stop=%d) | n-gram draft=%d\n", np, ngen, eos, g_draft);
    fputs(prompt,stdout); fflush(stdout);
    kv_alloc(m, np+ngen+g_draft+2);
    int *all=malloc((np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,np*sizeof(int));
    double t=now_s();
    float *logit=step(m,pids,np,0);
    EmitStream es={&T,m,t,0,0};
    grammar_reset();
    int produced=spec_decode(m,all,np,ngen,eos,logit,emit_stream,&es,NULL);
    double dt=now_s()-t;
    double tot=m->hits+m->miss;
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    printf("\n---\n%d tokens in %.2fs (%.2f tok/s) | expert hit rate %.1f%% | RSS %.2f GB\n",
        produced, dt, produced/dt, tot?100.0*m->hits/tot:0.0, rss_gb());
    printf("experts loaded/token: %.1f (per-layer %.2f across %d; baseline topk=%d) | TOPK=%d TOPP=%.2f\n",
        produced?(double)m->ereq/produced:0.0, (produced&&nsp)?(double)m->ereq/produced/nsp:0.0, nsp, c->topk, g_topk, g_topp);
    printf("speculation: %.2f tokens/forward (%llu forwards per %llu tokens) | MTP acceptance %.0f%% (%llu/%llu)\n",
        m->n_fw?(double)m->n_emit/m->n_fw:1.0, (unsigned long long)m->n_fw, (unsigned long long)m->n_emit,
        m->mtp_prop?100.0*m->mtp_acc/m->mtp_prop:0.0, (unsigned long long)m->mtp_acc, (unsigned long long)m->mtp_prop);
    if(g_cp_enq) printf("couple: %ld cross-layer prefetch hints enqueued\n", g_cp_enq);
    if(g_gr_prop) printf("grammar: %.0f%% acceptance (%llu/%llu forced drafts)\n",
        100.0*g_gr_acc/g_gr_prop, (unsigned long long)g_gr_acc, (unsigned long long)g_gr_prop);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    profile_print(m,dt);
    if(g_pilot_real) printf("PILOT_REAL: %ld load cross-layer completati, %ld scartati (main gia' sul layer) | PILOT_K=%d\n",
        (long)atomic_load_explicit(&g_pilot_loads,memory_order_relaxed),
        (long)atomic_load_explicit(&g_pilot_drops,memory_order_relaxed), g_pilot_k);
    if(g_looka){
        const char *nm[3]={"previous token (=SPEC prefetch)","layer input, skip attention","next layer (one step ahead)"};
        printf("LOOKAHEAD routing — recall of true experts in predicted top-8:\n");
        for(int i=0;i<3;i++) printf("  %-38s %5.1f%%  (%lld/%lld)\n", nm[i],
            la_tot[i]?100.0*la_hit[i]/la_tot[i]:0.0, (long long)la_hit[i], (long long)la_tot[i]);
    }
    free(pids); free(all);
    usage_save(m);
}

/* modalita' SERVE (per la CLI 'coli'): carica il modello UNA volta, poi CHAT conversazionale.
 * KV-cache PERSISTENTE tra i turni: la storia resta in cache, si fa il prefill solo dei
 * token NUOVI -> il modello RICORDA la conversazione e non ri-processa il passato (lossless,
 * piu' umano, piu' veloce). Template chat GLM con token speciali (CHAT_TEMPLATE=0 -> grezzo).
 * Protocollo: "\x01\x01" "READY" "\x01\x01\n" dopo il load; risposta in streaming; "\x01\x01" "END" "\x01\x01\n" a fine turno.
 * ":reset" (riga "\x02RESET") azzera la memoria. EOF -> esce. */
/* ---- RFC: RE-PIN A CALDO / LIVE RE-PIN (opt-in, REPIN=n, default OFF) ----
 * Upstream fa AUTOPIN allo START (dalla storia .coli_usage). Questo aggiunge un re-pin
 * TRA I TURNI: nel punto sicuro dopo la risposta scambia i pin peggiori con i non-pinnati
 * piu' caldi, cosi' l'hot-store insegue il carico VIVO senza un profilo a parte. Isteresi
 * 25% (+4) contro il ping-pong; max 4 scambi/passata (~20 MB di disco l'uno). Una heat
 * map separata decade a ogni passata: la storia persistente .coli_usage resta intatta.
 * EN: upstream AUTOPINs at START (from .coli_usage). This adds a between-turns re-pin: at
 * the safe point after the reply, swap the worst pins for the hottest unpinned, so the
 * hot-store tracks the LIVE workload without a separate profile. 25% (+4) hysteresis vs
 * ping-pong; max 4 swaps/pass (~20 MB disk each). A separate decaying heat map keeps
 * persistent .coli_usage intact while adapting to the current workload. */
static int g_repin=0;
static uint64_t g_last_repin=0;
typedef struct { long gain; int l, slot, eid; } RepinCand;
static int repin_pick(Model *m, RepinCand *out, int maxc){
    Cfg *c=&m->c; int nb=0;
    for(int l=0;l<c->n_layers;l++){
        if(!m->npin || m->npin[l]<1 || !m->eheat[l]) continue;
        ESlot *P=m->pin[l]; int ids[4096], zp, eu; long g;
        int np=m->npin[l]; if(np>4096) np=4096;
        for(int z=0;z<np;z++) ids[z]=P[z].eid;
        if(!tier_pick_lfru(m->eheat[l],m->elast[l],m->eaccess_clock,
                           c->n_experts,ids,np,&zp,&eu,&g)) continue;
        if(nb<maxc){ out[nb].gain=g; out[nb].l=l; out[nb].slot=zp; out[nb].eid=eu; nb++; }
        else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain) w=b;
               if(g>out[w].gain){ out[w].gain=g; out[w].l=l; out[w].slot=zp; out[w].eid=eu; } }
    }
    return nb;
}
static void repin_pass(Model *m){
    if(g_repin<=0) return;
    if(m->n_emit - g_last_repin < (uint64_t)g_repin) return;
    g_last_repin = m->n_emit;
    RepinCand cd[4]; int nb=repin_pick(m,cd,4);
    for(int b=0;b<nb;b++){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        int old=s->eid;
        uint32_t old_heat=m->eheat[cd[b].l][old], new_heat=m->eheat[cd[b].l][cd[b].eid];
#ifdef COLI_CUDA
        int gpu=s->g.cuda_eligible;
        int64_t old_gpu=gpu ? (int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                             +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                             +(int64_t)coli_cuda_tensor_bytes(s->d.cuda) : 0;
#endif
        double t0=now_s();
        expert_load(m,cd[b].l,cd[b].eid,s,1);       /* disk -> RAM, same resident slot */
        const char *tier="RAM";
#ifdef COLI_CUDA
        if(gpu){                                  /* refresh the same VRAM slot now, not lazily */
            if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                int64_t now_gpu=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                m->gpu_expert_bytes+=now_gpu-old_gpu; tier="VRAM";
                if(g_cuda_release_host) expert_host_release(m,s);
            } else {
                qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                m->gpu_expert_count--; m->gpu_expert_bytes-=old_gpu;
                fprintf(stderr,"[REPIN] VRAM upload failed; slot downgraded to RAM\n");
            }
        }
#endif
        fprintf(stderr,"[REPIN] %s layer %d: evict %d (heat=%u) <- admit %d (heat=%u) in %.0f ms\n",
            tier,cd[b].l,old,old_heat,cd[b].eid,new_heat,(now_s()-t0)*1e3);
    }
    for(int l=0;l<m->c.n_layers;l++) if(m->eheat[l]) tier_decay(m->eheat[l],m->c.n_experts);
}
/* ---- KV SU DISCO: la conversazione si riapre CALDA (KVSAVE=0 disattiva) ----
 * Il re-prefill di una chat riaperta costa ore su questo disco; la KV compressa MLA
 * costa ~182 KB/token. File <SNAP>/.coli_kv append-only: header (magic + dimensioni +
 * nrec) e un record per posizione [tok i32][Lc+Rc dei 78 layer][Ic DSA]. A fine turno
 * si appendono SOLO le posizioni nuove e si riscrive nrec per ultimo: un crash a meta'
 * append lascia nrec vecchio = file coerente. La riga KV del layer MTP non si salva:
 * al resume kv_start=-1 e la finestra di draft riparte da sola. */
static int g_kvsave=1;
#define KV_MAGIC "COLIKV1\0"
static void kv_hdr(Model *m, int32_t *h, int nrec){
    Cfg *c=&m->c; int nic=0;
    for(int i=0;i<c->n_layers;i++) if(m->Ic && m->Ic[i]) nic++;
    h[0]=c->n_layers; h[1]=c->kv_lora; h[2]=c->qk_rope;
    h[3]=m->has_dsa?c->index_hd:0; h[4]=nic; h[5]=c->vocab; h[6]=nrec; h[7]=0;
}
static void kv_disk_truncate(Model *m, int nrec){
    if(!g_kvsave) return;
    KVState *k=m->kv;
    FILE *f=fopen(k->disk_path,"r+b");
    if(!f){ k->disk_nrec=0; return; }
    k->disk_nrec=nrec;
    int32_t nr=nrec; fseek(f,8+6*4,SEEK_SET); fwrite(&nr,4,1,f); fclose(f);
}
static void kv_disk_reset(Model *m){ kv_disk_truncate(m,0); }
static void kv_disk_append(Model *m, const int *hist, int len){
    KVState *k=m->kv;
    if(!g_kvsave || len<=k->disk_nrec) return;
    Cfg *c=&m->c;
    FILE *f=fopen(k->disk_path,"r+b");
    if(!f){ f=fopen(k->disk_path,"wb"); if(!f) return;
        int32_t h[8]; kv_hdr(m,h,0); fwrite(KV_MAGIC,1,8,f); fwrite(h,4,8,f); }
    int64_t rec = 4 + (int64_t)c->n_layers*(c->kv_lora+c->qk_rope)*4;
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i]) rec+=(int64_t)c->index_hd*4;
    fseek(f, 8+8*4 + (int64_t)k->disk_nrec*rec, SEEK_SET);
    for(int p=k->disk_nrec;p<len;p++){
        int32_t tk=hist[p]; fwrite(&tk,4,1,f);
        for(int i=0;i<c->n_layers;i++){
            fwrite(m->Lc[i]+(int64_t)p*c->kv_lora, 4, c->kv_lora, f);
            fwrite(m->Rc[i]+(int64_t)p*c->qk_rope, 4, c->qk_rope, f);
        }
        if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i])
            fwrite(m->Ic[i]+(int64_t)p*c->index_hd, 4, c->index_hd, f);
    }
    fflush(f);                                   /* dati prima, contatore poi */
    int32_t nr=len; fseek(f,8+6*4,SEEK_SET); fwrite(&nr,4,1,f); fclose(f);
    k->disk_nrec=len;
}
static int kv_disk_load(Model *m, int *hist, int maxctx){
    if(!g_kvsave) return 0;
    KVState *k=m->kv;
    Cfg *c=&m->c;
    FILE *f=fopen(k->disk_path,"rb"); if(!f) return 0;
    char mg[8]; int32_t h[8], w[8]; kv_hdr(m,w,0);
    if(fread(mg,1,8,f)!=8 || memcmp(mg,KV_MAGIC,8) || fread(h,4,8,f)!=8 ||
       h[0]!=w[0]||h[1]!=w[1]||h[2]!=w[2]||h[3]!=w[3]||h[4]!=w[4]||h[5]!=w[5]){
        fprintf(stderr,"[KV] ignoring .coli_kv from a different model or version\n"); fclose(f); return 0; }
    int nrec=h[6];
    if(nrec<1){ fclose(f); return 0; }
    if(nrec>=maxctx-8-g_draft){
        fprintf(stderr,"[KV] saved conversation (%d tokens) exceeds the context: starting over\n",nrec);
        fclose(f); return 0; }
    double t0=now_s();
    for(int p=0;p<nrec;p++){
        int32_t tk; if(fread(&tk,4,1,f)!=1){ nrec=p; break; } hist[p]=tk;
        for(int i=0;i<c->n_layers;i++){
            if(fread(m->Lc[i]+(int64_t)p*c->kv_lora, 4, c->kv_lora, f)!=(size_t)c->kv_lora ||
               fread(m->Rc[i]+(int64_t)p*c->qk_rope, 4, c->qk_rope, f)!=(size_t)c->qk_rope){ nrec=p; goto out; }
        }
        if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(m->Ic[i])
            if(fread(m->Ic[i]+(int64_t)p*c->index_hd, 4, c->index_hd, f)!=(size_t)c->index_hd){ nrec=p; goto out; }
    }
out:
    fclose(f);
    if(nrec>0){
        if(m->has_mtp) m->kv_start[c->n_layers]=-1;    /* la finestra MTP riparte da sola */
        fprintf(stderr,"[KV] resumed conversation from disk: %d tokens in %.1fs (no re-prefill)\n",
            nrec, now_s()-t0);
    }
    k->disk_nrec=nrec;
    return nrec;
}

typedef struct { KVState kv; int *hist, len, first; } ServeCtx;
static double kv_pool_bytes(Model *m, int max_ctx);

static void serve_ctx_init(Model *m, ServeCtx *s, const char *snap, int slot, int maxctx){
    s->kv.kv_start=calloc(m->c.n_layers+1,sizeof(int));
    if(m->has_mtp) s->kv.kv_start[m->c.n_layers]=-1;
    kv_bind(m,&s->kv); kv_alloc(m,maxctx);
    s->hist=malloc(maxctx*sizeof(int));
    if(!s->hist){ fprintf(stderr,"OOM serve_ctx_init hist\n"); exit(1); }
    s->first=1;
    if(slot==0) snprintf(s->kv.disk_path,sizeof(s->kv.disk_path),"%s/.coli_kv",snap);
    else snprintf(s->kv.disk_path,sizeof(s->kv.disk_path),"%s/.coli_kv.%d",snap,slot);
    s->len=kv_disk_load(m,s->hist,maxctx); if(s->len>0) s->first=0;
}

static void serve_ctx_free(Model *m, ServeCtx *s){
    KVState *k=&s->kv; int NR=m->c.n_layers+1;
    if(k->Lc) for(int i=0;i<NR;i++){ free(k->Lc[i]); free(k->Rc[i]); }
    if(k->Ic) for(int i=0;i<m->c.n_layers;i++) free(k->Ic[i]);
    free(k->Lc); free(k->Rc); free(k->Ic); free(k->kv_start); free(s->hist);
}

typedef struct {
    int active, pending, emitted, maximum, prompt_tokens, length_limited;
    unsigned long long id;
    float temp, top_p;
    double started;
    uint64_t hits0, miss0;
} ServeReq;

static void mux_data(Tok *T, unsigned long long id, int token){
    char out[256]; int n=tok_decode(T,&token,1,out,sizeof(out));
    printf("DATA %llu %d\n",id,n); if(n>0) fwrite(out,1,(size_t)n,stdout); putchar('\n');
    fflush(stdout);
}

static void mux_done(Model *m, ServeCtx *sc, ServeReq *r){
    double dt=now_s()-r->started; if(dt<1e-6) dt=1e-6;
    double dh=(double)(m->hits-r->hits0), dm=(double)(m->miss-r->miss0);
    hwinfo_emit(m);
    tiers_emit(m);
    emap_emit(m);
    hits_emit(m);
    printf("DONE %llu STAT %d %.2f %.1f %.2f %d %d\n",r->id,r->emitted,
           r->emitted/dt,(dh+dm)>0?100.0*dh/(dh+dm):0.0,rss_gb(),
           r->prompt_tokens,r->length_limited);
    fflush(stdout); kv_bind(m,&sc->kv); kv_disk_append(m,sc->hist,sc->len);
    r->active=0;
}

/* Read and prefill one request. Returns -1 on EOF, 0 for a rejected frame and
 * 1 for an accepted request. Prefill deliberately remains serial: continuous
 * batching starts at decode, where every active slot contributes one row. */
static int mux_submit(Model *m, Tok *T, ServeCtx *ctx, ServeReq *req, int nctx,
                      int maxctx, int eos){
    char *line=NULL; size_t cap=0; ssize_t nr=getline(&line,&cap,stdin);
    if(nr<0){ free(line); return -1; }
    if(nr && line[nr-1]=='\n') line[--nr]=0;
    if(!strncmp(line,"CANCEL ",7)){
        unsigned long long id=0; char tail;
        if(sscanf(line+7,"%llu %c",&id,&tail)!=1 || id==0){
            printf("ERROR 0 BAD_REQUEST\n"); fflush(stdout); free(line); return 0;
        }
        for(int i=0;i<nctx;i++) if(req[i].active && req[i].id==id){
            req[i].active=0; kv_bind(m,&ctx[i].kv);
            kv_disk_append(m,ctx[i].hist,ctx[i].len);
            printf("ERROR %llu CANCELLED\n",id); fflush(stdout); free(line); return 0;
        }
        printf("ERROR %llu NOT_FOUND\n",id); fflush(stdout); free(line); return 0;
    }
    ColiSubmit sub; int valid=coli_submit_parse(line,&sub);
    if(!valid){ printf("ERROR 0 BAD_REQUEST\n"); fflush(stdout); free(line); return 0; }
    char *raw=malloc((size_t)sub.bytes+1);
    if(!raw){ fprintf(stderr,"OOM multiplex payload\n"); exit(1); }
    if(fread(raw,1,(size_t)sub.bytes,stdin)!=(size_t)sub.bytes){ free(raw); free(line); return -1; }
    int delim=fgetc(stdin);
    if(delim!='\n'){
        printf("ERROR %llu BAD_FRAME\n",sub.id); fflush(stdout);
        free(raw); free(line); return -1;
    }
    raw[sub.bytes]=0;
    if(sub.slot>=nctx || memchr(raw,0,(size_t)sub.bytes)){
        printf("ERROR %llu BAD_REQUEST\n",sub.id); fflush(stdout); free(raw); free(line); return 0;
    }
    if(req[sub.slot].active){
        printf("ERROR %llu SLOT_BUSY\n",sub.id); fflush(stdout); free(raw); free(line); return 0;
    }
    for(int i=0;i<nctx;i++) if(req[i].active && req[i].id==sub.id){
        printf("ERROR %llu DUPLICATE_ID\n",sub.id); fflush(stdout); free(raw); free(line); return 0;
    }
    ServeCtx *sc=&ctx[sub.slot]; kv_bind(m,&sc->kv);
    int *tmp=malloc(maxctx*sizeof(int));
    if(!tmp){ fprintf(stderr,"OOM mux_submit tmp\n"); free(raw); free(line); exit(1); }
    int nt=tok_encode(T,raw,(int)sub.bytes,tmp,maxctx-2);
    free(raw); free(line);
    if(nt<1){ free(tmp); printf("ERROR %llu EMPTY_PROMPT\n",sub.id); fflush(stdout); return 0; }
    int prefix=0; while(prefix<sc->len && prefix<nt && sc->hist[prefix]==tmp[prefix]) prefix++;
    if(prefix<sc->len){ sc->len=prefix; if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
        kv_disk_truncate(m,sc->len); }
    int add=nt-sc->len;
    if(add>0) memcpy(sc->hist+sc->len,tmp+sc->len,(size_t)add*sizeof(int));
    free(tmp);
    float *logit = add>0 ? step(m,sc->hist+sc->len,add,sc->len)
                         : step(m,sc->hist+sc->len-1,1,sc->len-1);
    sc->len+=add; sc->first=0;
    ServeReq *r=&req[sub.slot]; memset(r,0,sizeof(*r));
    r->id=sub.id; r->maximum=sub.max_tokens; r->temp=sub.temperature; r->top_p=sub.top_p;
    r->prompt_tokens=nt; r->started=now_s(); r->hits0=m->hits; r->miss0=m->miss;
    int room=maxctx-sc->len-1; if(r->maximum>room){r->maximum=room; r->length_limited=1;}
    g_temp=r->temp; g_nuc=r->top_p;
    int next=pick_tok(logit,m->c.vocab,-1); free(logit);
    if(r->maximum<=0 || next==eos || is_stop(next)){ mux_done(m,sc,r); return 1; }
    r->pending=next; r->emitted=1; r->active=1; sc->hist[sc->len]=next; m->n_emit++;
    mux_data(T,r->id,next);
    if(r->emitted>=r->maximum) mux_done(m,sc,r);
    return 1;
}

static void run_serve_mux(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp); int eos=tok_id_of(&T,"<|endoftext|>"); stops_arm(&m->c,eos);
    g_draft=0; /* one scheduler owns every forward; MTP/speculation is not ragged-safe */
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int nctx=getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
    if(nctx<1||nctx>16){fprintf(stderr,"KV_SLOTS deve essere tra 1 e 16\n");exit(2);}
    g_kvsave=getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    KVState *initial=m->kv; free(initial->kv_start); free(initial);
    ServeCtx *ctx=calloc(nctx,sizeof(*ctx)); ServeReq *req=calloc(nctx,sizeof(*req));
    for(int i=0;i<nctx;i++) serve_ctx_init(m,&ctx[i],snap,i,maxctx);
    setvbuf(stdin,NULL,_IONBF,0);
    printf("\x01\x01READY\x01\x01\nSTAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout);
    hwinfo_emit(m);
    tiers_emit(m);
    emap_emit(m);
    int eof=0;
    for(;;){
        int active=0; for(int i=0;i<nctx;i++) active+=req[i].active;
        /* Poll stdin for available input without blocking. On POSIX this is
         * select(); on Windows, select() on a pipe handle routes to winsock
         * and always returns -1 (SOCKET_ERROR), so the batch loop could never
         * accept a request (#139). PeekNamedPipe on the stdin OS handle is
         * the Windows equivalent: it reports bytes available without reading. */
        int ready=0;
        if(!eof){
#if defined(__APPLE__) || defined(__linux__)
            fd_set rfds; FD_ZERO(&rfds); FD_SET(STDIN_FILENO,&rfds);
            struct timeval tv={0,0}, *ptv=active?&tv:NULL;
            ready=select(STDIN_FILENO+1,&rfds,NULL,NULL,ptv);
            if(ready>0 && FD_ISSET(STDIN_FILENO,&rfds))
#elif defined(_WIN32)
            HANDLE ih=(HANDLE)_get_osfhandle(_fileno(stdin));
            DWORD avail=0;
            /* WaitForSingleObject(handle, 0) is non-blocking and works on both
             * pipe and console handles; for pipes PeekNamedPipe gives the byte
             * count. Either returns "data is there right now". When a decode
             * is active we poll (timeout 0); when idle we block until input. */
            if(active){
                ready=(WaitForSingleObject(ih,0)==WAIT_OBJECT_0)?1:0;
            } else {
                ready=(WaitForSingleObject(ih,INFINITE)==WAIT_OBJECT_0)?1:0;
            }
            if(ready && PeekNamedPipe(ih,NULL,0,NULL,&avail,NULL) && avail>0)
#endif
                if(mux_submit(m,&T,ctx,req,nctx,maxctx,eos)<0) eof=1;
        }
        active=0; for(int i=0;i<nctx;i++) active+=req[i].active;
        if(!active){ if(eof) break; continue; }
        DecodeRow rows[16]; int slots[16], S=0;
        for(int i=0;i<nctx;i++) if(req[i].active){
            rows[S]=(DecodeRow){&ctx[i].kv,req[i].pending,ctx[i].len}; slots[S++]=i;
        }
        float *lo=step_decode_batch(m,rows,S); if(!lo){fprintf(stderr,"decode batch failed\n");break;}
        m->n_fw++;
        for(int s=0;s<S;s++){
            int i=slots[s]; ServeCtx *sc=&ctx[i]; ServeReq *r=&req[i];
            sc->len++; g_temp=r->temp; g_nuc=r->top_p;
            int next=pick_tok(lo+(int64_t)s*m->c.vocab,m->c.vocab,-1);
            if(next==eos || is_stop(next)){mux_done(m,sc,r);continue;}
            r->pending=next; sc->hist[sc->len]=next; r->emitted++; m->n_emit++;
            mux_data(&T,r->id,next);
            if(r->emitted>=r->maximum) mux_done(m,sc,r);
        }
        free(lo);
    }
    usage_save(m);
    for(int i=0;i<nctx;i++) serve_ctx_free(m,&ctx[i]); free(ctx); free(req);
    m->kv=NULL; m->Lc=m->Rc=m->Ic=NULL; m->kv_start=NULL; m->max_t=0;
}

static void run_serve(Model *m, const char *snap){
    /* Serve mode speaks a byte protocol over BOTH stdout and stdin:
     *   stdout: \x01\x01READY\x01\x01\n, STAT lines, \x01\x01END\x01\x01\n
     *   stdin:  text lines plus \x02RESET / \x02MORE control bytes.
     * 'coli' matches the sentinels with endswith() and a "^STAT ..." regex,
     * so they must arrive byte-exact (LF, no CR). On Windows the CRT opens
     * both handles in TEXT mode: stdout translates '\n'->'\r\n' (so the READY
     * sentinel never matches and chat hangs at ~10 GB resident), and stdin
     * translates '\r\n'->'\n' and rejects writes of raw bytes with EINVAL,
     * breaking the control protocol. Put BOTH handles in BINARY mode so the
     * protocol bytes are exact in both directions. No-op on Linux/macOS. */
#ifdef _WIN32
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    setvbuf(stdout, NULL, _IONBF, 0);
#endif
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    grammar_setup(&T);                   /* metodo F: GRAMMAR=file.gbnf (#48) */
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):256;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int templ=getenv("CHAT_TEMPLATE")?atoi(getenv("CHAT_TEMPLATE")):1;
    g_kvsave = getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    int nctx=getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
    if(nctx<1||nctx>16){ fprintf(stderr,"KV_SLOTS must be between 1 and 16\n"); exit(2); }
    KVState *initial=m->kv; free(initial->kv_start); free(initial);
    ServeCtx *ctx=calloc(nctx,sizeof(ServeCtx));
    for(int i=0;i<nctx;i++) serve_ctx_init(m,&ctx[i],snap,i,maxctx);
    int active=0; ServeCtx *sc=&ctx[0]; kv_bind(m,&sc->kv);
    fprintf(stderr,"[KV] context slots: %d x %d tokens, projected pool %.2f GB\n",
        nctx,maxctx,kv_pool_bytes(m,maxctx)/1e9);
    #define hist  (sc->hist)
    #define len   (sc->len)
    #define first (sc->first)
    char *line=NULL; size_t cap=0; ssize_t nr; char *buf=malloc(1<<16);
    printf("\x01\x01" "READY" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout);
    tiers_emit(m);
    while((nr=getline(&line,&cap,stdin))>0){
        if(nr>0 && line[nr-1]=='\n') line[--nr]=0;
        if(!strcmp(line,"\x02RESET")){ len=0; first=1; if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
            kv_disk_reset(m);
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        if(!strcmp(line,"\x02MORE")){                /* continua la risposta troncata da NGEN:
            la storia e' gia' in KV, basta ri-forwardare l'ULTIMO token per riavere i logits */
            if(len<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
            int cur=ngen; if(len+cur+g_draft+2>=maxctx) cur=maxctx-len-g_draft-2;
            uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
            float *logit=step(m,hist+len-1,1,len-1);
            EmitStream es={&T,m,now_s(),0,1};
            int prod=0;
            if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
            else free(logit);
            double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
            double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
            printf("\n\x01\x01" "END" "\x01\x01\n");
            printf("STAT %d %.2f %.1f %.2f\n", prod, prod/tdt, (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb());
            fflush(stdout); kv_disk_append(m,hist,len); repin_pass(m); continue; }   /* RFC: re-pin a caldo tra i turni / live re-pin between turns */
        if(nr<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        /* API mode: an exact, length-prefixed prompt. Unlike the interactive
         * line protocol this accepts newlines. The tokenized prompt is matched
         * against hist so the common KV prefix survives stateless HTTP turns.
         * Per-request generation controls follow the byte count:
         *   \x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n */
        char *raw=NULL, *input=line;
        int input_n=(int)nr, raw_mode=0, req_ngen=ngen, prompt_tokens=0;
        float base_temp=g_temp, base_nuc=g_nuc;
        if(!strncmp(line,"\x02PROMPT ",8)){
            unsigned long long nb=0; double rt=0, rp=0; int slot=0;
            int nf=sscanf(line+8,"%llu %d %lf %lf %d",&nb,&req_ngen,&rt,&rp,&slot);
            if(nf<4 || nb>(16u<<20) || req_ngen<1 || rt<0 || rt>2 || rp<=0 || rp>1 ||
               slot<0 || slot>=nctx){
                printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;
            }
            active=slot; sc=&ctx[active]; kv_bind(m,&sc->kv);
            raw=malloc((size_t)nb+1); if(!raw){fprintf(stderr,"OOM raw prompt\n");exit(1);}
            if(fread(raw,1,(size_t)nb,stdin)!=(size_t)nb){free(raw);break;}
            int delim=fgetc(stdin); if(delim!='\n' && delim!=EOF) ungetc(delim,stdin);
            if(memchr(raw,0,(size_t)nb)){free(raw); printf("\x01\x01" "END" "\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;}
            raw[nb]=0; input=raw; input_n=(int)nb; raw_mode=1;
            if(req_ngen>ngen) req_ngen=ngen;
            g_temp=(float)rt; g_nuc=(float)rp;
        } else { active=0; sc=&ctx[0]; kv_bind(m,&sc->kv); }
        int bl=0, k=0;                           /* costruisce/tokenizza il turno */
        /* template UFFICIALE GLM-5.2 (chat_template.jinja): niente \n dopo i ruoli, e dopo
         * <|assistant|> serve SEMPRE il blocco think — <think></think> lo DISATTIVA (nothink):
         * col template sbagliato il modello farfuglia e non emette mai lo stop. THINK=1 lo abilita. */
        const char *tk = getenv("THINK")&&atoi(getenv("THINK"))? "<think>" : "<think></think>";
        if(raw_mode){
            int *tmp=malloc(maxctx*sizeof(int)); if(!tmp){fprintf(stderr,"OOM raw tokens\n");exit(1);}
            prompt_tokens=tok_encode(&T,input,input_n,tmp,maxctx-8-g_draft);
            int old_len=len, prefix=0;
            while(prefix<old_len && prefix<prompt_tokens && hist[prefix]==tmp[prefix]) prefix++;
            if(prefix<old_len){
                len=prefix;
                if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
                kv_disk_truncate(m,len);           /* il prossimo append sovrascrive solo la coda */
            }
            k=prompt_tokens-len;
            if(k>0) memcpy(hist+len,tmp+len,k*sizeof(int));
            fprintf(stderr,"[API] KV slot %d prefix %d/%d token, prefill %d\n",
                active,len,prompt_tokens,k);
            free(tmp);
        } else {
            if(templ){ if(first) bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop>");
                       bl+=snprintf(buf+bl,(1<<16)-bl,"<|user|>%s<|assistant|>%s",input,tk); }
            else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",input);
            k=tok_encode(&T,buf,bl,hist+len,maxctx-len); prompt_tokens=k;
            if(len+k+8+g_draft>=maxctx){ len=0; first=1; kv_disk_reset(m);
                bl=0; if(templ){ bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop><|user|>%s<|assistant|>%s",input,tk); }
                else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",input);
                k=tok_encode(&T,buf,bl,hist,maxctx); if(k>maxctx-8-g_draft) k=maxctx-8-g_draft;
                prompt_tokens=k;
            }
        }
        if(prompt_tokens<1){ free(raw); g_temp=base_temp; g_nuc=base_nuc;
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb()); fflush(stdout); continue; }
        first=0;
        int cur=req_ngen; if(len+k+cur+g_draft+2>=maxctx) cur=maxctx-len-k-g_draft-2;
        uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
        float *logit;
        if(k>0){ logit=step(m,hist+len,k,len); len+=k; }
        else logit=step(m,hist+len-1,1,len-1);   /* prompt identico/prefisso: rigenera i logits */
        EmitStream es={&T,m,now_s(),0,1};
        int prod=0;
        grammar_reset();                         /* nuova risposta = nuovo documento (MORE invece continua) */
        if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
        else free(logit);
        double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
        double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
        printf("%s\x01\x01" "END" "\x01\x01\n",raw_mode?"":"\n");
        printf("STAT %d %.2f %.1f %.2f %d %d\n", prod, prod/tdt,
            (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb(), prompt_tokens, prod>=cur);
        fflush(stdout);
        free(raw); g_temp=base_temp; g_nuc=base_nuc;
        usage_save(m);                   /* la cache che impara: storia aggiornata a ogni turno */
        kv_disk_append(m,hist,len);      /* KV su disco: il prossimo avvio riparte da qui */
        repin_pass(m);                   /* safe request boundary: adapt session-local hot tier */
    }
    free(line); free(buf);
    usage_save(m);
    #undef hist
    #undef len
    #undef first
    for(int i=0;i<nctx;i++) serve_ctx_free(m,&ctx[i]);
    free(ctx); m->kv=NULL; m->Lc=m->Rc=m->Ic=NULL; m->kv_start=NULL; m->max_t=0;
}

static int *read_arr(jval*o,const char*k,int*n){
    jval*a=json_get(o,k);
    if(!a){ *n=0; return NULL; }
    int*r=malloc(a->len*sizeof(int));
    if(!r){ fprintf(stderr,"OOM read_arr\n"); exit(1); }
    for(int i=0;i<a->len;i++) r[i]=(int)a->kids[i]->num; *n=a->len; return r; }

/* byte residenti di un tensore [O,I] al numero di bit dato (specchio di qt_bytes) */
static int64_t tbytes(int O,int I,int bits){
    if(bits>=16) return (int64_t)O*I*4;
    if(bits>=5)  return (int64_t)O*I + (int64_t)O*4;
    return (int64_t)O*((I+1)/2) + (int64_t)O*4;
}
/* byte VERI di un expert: dal container se pre-quantizzato, altrimenti stima da ebits */
static int64_t expert_bytes_probe(Model *m, int ebits){
    Cfg *c=&m->c; int64_t eb=0; char nm[256];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.gate_proj.weight",c->first_dense);
    if(st_nbytes(&m->S,nm)>0){
        const char *suf[3]={"gate_proj","up_proj","down_proj"};
        for(int k=0;k<3;k++){
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight",c->first_dense,suf[k]);
            eb+=st_nbytes(&m->S,nm);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight.qs",c->first_dense,suf[k]);
            int64_t q=st_nbytes(&m->S,nm); if(q>0) eb+=q;
        }
    }
    if(eb<=0) eb = tbytes(c->moe_inter,c->hidden,ebits)*2 + tbytes(c->hidden,c->moe_inter,ebits);
    return eb;
}

/* TIERS: fotografia della piramide expert per la dashboard web —
 * "TIERS <vram> <ram> <disk> <vram_gb> <ram_gb>" sul canale di protocollo.
 * ram = pinnati non-VRAM + LRU corrente; disk = tutto il resto. */
/* BRAIN MAP (dashboard): per-turn expert hit bitmap + full residency/heat map.
 * g_ehit[layer][eid]=1 quando l'expert viene instradato in questo turno;
 * hits_emit lo serializza e lo azzera. emap_emit fotografa tier+heat di TUTTI. */
static uint8_t **g_ehit;
static void ehit_mark(Model *m, int layer, int eid){
    if(!g_ehit){ Cfg *c=&m->c;
        g_ehit=calloc(c->n_layers+1,sizeof(uint8_t*));
        for(int i=0;i<=c->n_layers;i++) g_ehit[i]=calloc(c->n_experts,1);
    }
    g_ehit[layer][eid]=1;
}

/* HWINFO: hardware snapshot for the web dashboard — emitted once at READY. */
static void hwinfo_emit(Model *m){
    Cfg *c=&m->c;
    /* CPU */
    char cpu[256]=""; FILE *ci=fopen("/proc/cpuinfo","r");
    if(ci){ char ln[256];
        while(fgets(ln,sizeof(ln),ci)) if(!strncmp(ln,"model name",10)){
            char *p=strchr(ln,':'); if(p){ p++; while(*p==' ')p++;
            int n=(int)strlen(p); if(n>0&&p[n-1]=='\n')p[--n]=0;
            snprintf(cpu,sizeof(cpu),"%s",p); } break; }
        fclose(ci); }
    int cores=0;
#ifdef _SC_NPROCESSORS_ONLN
    cores=(int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    /* RAM */
    double ram_total=0,ram_avail=0;
    FILE *mi=fopen("/proc/meminfo","r");
    if(mi){ char ln[256]; double mt=0,ma=0;
        while(fgets(ln,sizeof(ln),mi)){
            if(sscanf(ln,"MemTotal: %lf",&mt)==1) ram_total=mt/1e6;
            if(sscanf(ln,"MemAvailable: %lf",&ma)==1) ram_avail=ma/1e6;
        } fclose(mi); }
    /* GPU */
    int ngpu=0; double vram_total=0;
    char gpu_name[128]="";
#ifdef COLI_CUDA
    ngpu=g_cuda_ndev; vram_total=m->gpu_expert_bytes/1e9;
    for(int i=0;i<g_cuda_ndev;i++){
        size_t fr=0,to=0; coli_cuda_mem_info(g_cuda_devices[i],&fr,&to);
        if(!i) vram_total=(double)to*g_cuda_ndev/1e9;
    }
    /* GPU name from the first device — already printed at init */
    if(g_cuda_ndev>0){
        /* We don't have a device-name API; parse from the init log line stored in stderr.
         * Simpler: just read the nvidia driver sysfs or use a fixed label. */
        snprintf(gpu_name,sizeof(gpu_name),"CUDA device x%d",g_cuda_ndev);
    }
#endif
    printf("HWINFO %d %.1f %.1f %d %.1f %s|%s\n",
        cores,ram_total,ram_avail,ngpu,vram_total,cpu,gpu_name);
    fflush(stdout);
}

static void tiers_emit(Model *m){
    Cfg *c=&m->c; int nsp=0;
    for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    int total=(nsp+(m->has_mtp?1:0))*c->n_experts;
    int pinned=0,lru=0;
    for(int i=0;i<=c->n_layers;i++){ pinned+=m->npin?m->npin[i]:0; lru+=m->ecn?m->ecn[i]:0; }
    int vram=0; double vram_gb=0;
#ifdef COLI_CUDA
    vram=m->gpu_expert_count; vram_gb=m->gpu_expert_bytes/1e9;
#endif
    int ram=pinned-vram+lru; if(ram<0) ram=0;
    int disk=total-vram-ram; if(disk<0) disk=0;
    double eb=(double)expert_bytes_probe(m,m->ebits);
    printf("TIERS %d %d %d %.2f %.2f\n",vram,ram,disk,vram_gb,ram*eb/1e9);
    fflush(stdout);
}

/* EMAP: 1 byte/expert (2bit tier: 0=disk 1=RAM 2=VRAM | 6bit heat log2-bucket),
 * righe = layer sparsi in ordine (+MTP se presente), colonne = n_experts. Hex. */
static void emap_emit(Model *m){
    Cfg *c=&m->c;
    int rows=0;
    for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) rows++;
    int has_mtp = m->has_mtp && m->eusage[c->n_layers];
    if(has_mtp) rows++;
    int cols=c->n_experts;
    char *hex=malloc((size_t)rows*cols*2+1); int w=0;
    for(int i=0;i<=c->n_layers;i++){
        int is_row = (i<c->n_layers && m->L[i].sparse) || (i==c->n_layers && has_mtp);
        if(!is_row) continue;
        for(int e=0;e<cols;e++){
            int tier=0;
            ESlot *P=m->pin[i];
            for(int z=0;z<m->npin[i];z++) if(P[z].eid==e){
#ifdef COLI_CUDA
                tier = P[z].g.cuda?2:1;
#else
                tier = 1;
#endif
                break; }
            if(!tier && m->ecache && m->ecache[i])
                for(int z=0;z<m->ecn[i];z++) if(m->ecache[i][z].eid==e){ tier=1; break; }
            uint32_t u = m->eusage[i]?m->eusage[i][e]:0;
            int heat=0; while(u){ heat++; u>>=1; } if(heat>63) heat=63;
            int b=(tier<<6)|heat;
            hex[w++]="0123456789abcdef"[b>>4]; hex[w++]="0123456789abcdef"[b&15];
        }
    }
    hex[w]=0;
    printf("EMAP %d %d %s\n",rows,cols,hex); fflush(stdout); free(hex);
}

/* HITS: bitmap 1bit/expert (stesso ordine di EMAP), poi azzera per il turno dopo. */
static void hits_emit(Model *m){
    Cfg *c=&m->c; if(!g_ehit) return;
    int rows=0;
    for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) rows++;
    int has_mtp = m->has_mtp && m->eusage[c->n_layers];
    if(has_mtp) rows++;
    int cols=c->n_experts, nb=(rows*cols+7)/8;
    uint8_t *bm=calloc(nb,1); int bit=0;
    for(int i=0;i<=c->n_layers;i++){
        int is_row = (i<c->n_layers && m->L[i].sparse) || (i==c->n_layers && has_mtp);
        if(!is_row) continue;
        for(int e=0;e<cols;e++,bit++)
            if(g_ehit[i][e]){ bm[bit>>3]|=1<<(bit&7); g_ehit[i][e]=0; }
    }
    char *hex=malloc((size_t)nb*2+1); int w=0;
    for(int b=0;b<nb;b++){ hex[w++]="0123456789abcdef"[bm[b]>>4]; hex[w++]="0123456789abcdef"[bm[b]&15]; }
    hex[w]=0;
    printf("HITS %d %d %s\n",rows,cols,hex); fflush(stdout); free(hex); free(bm);
}

/* scarica su file l'istogramma d'uso degli expert: righe "layer eid count" (per PIN).
 * Include la riga MTP (layer n_layers). Scrittura atomica (tmp+rename): viene chiamata
 * anche a ogni turno di serve e il processo puo' morire in qualsiasi momento. */
static void stats_dump_q(Model *m, const char *path, int quiet){
    char tmp[2100]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *f=fopen(tmp,"w"); if(!f){ if(!quiet) perror(tmp); return; }
    Cfg *c=&m->c; int64_t tot=0, nz=0;
    for(int i=0;i<=c->n_layers;i++){ if(!m->eusage[i]) continue;
        for(int e=0;e<c->n_experts;e++) if(m->eusage[i][e]){ fprintf(f,"%d %d %u\n",i,e,m->eusage[i][e]); tot+=m->eusage[i][e]; nz++; } }
    fclose(f); rename(tmp,path);
    if(!quiet) fprintf(stderr,"[STATS] %lld selections across %lld distinct experts -> %s\n",(long long)tot,(long long)nz,path);
}
static void stats_dump(Model *m, const char *path){ stats_dump_q(m,path,0); }

/* CACHE CHE IMPARA: istogramma d'uso PERSISTENTE in <SNAP>/.coli_usage.
 * Caricato all'avvio (i contatori ripartono dalla storia), salvato a ogni turno:
 * piu' usi colibri', meglio l'auto-pin conosce i TUOI expert caldi. */
static char g_usage_path[2100]="";
static int64_t usage_load(Model *m, const char *path){
    FILE *f=fopen(path,"r"); if(!f) return 0;
    Cfg *c=&m->c; int l,e; uint32_t cnt; int64_t tot=0;
    while(fscanf(f,"%d %d %u",&l,&e,&cnt)==3)
        if(l>=0&&l<=c->n_layers&&e>=0&&e<c->n_experts&&m->eusage[l]){ m->eusage[l][e]+=cnt; tot+=cnt; }
    fclose(f); return tot;
}
static void usage_save(Model *m){ if(g_usage_path[0]) stats_dump_q(m,g_usage_path,1); }

/* HOT-STORE ("il redis del colibri'"): carica in RAM, UNA VOLTA e per sempre, i top expert
 * per frequenza d'uso misurata (file STATS di un run precedente), entro un budget in GB.
 * Ogni hit evita una lettura dal disco lento. */
/* MLOCK: inchioda in RAM fisica gli expert pinnati cosi' il compressore di memoria di
 * macOS non li comprime/evacua (visto: RSS reale < residente previsto -> "hit" lenti).
 * -1 = auto (ON su macOS dove serve e RLIMIT_MEMLOCK e' permissivo; OFF altrove, dove
 * il limite e' spesso minuscolo e va alzato a mano), 0 = off, 1 = force.
 * EN: MLOCK: wire pinned experts into physical RAM so macOS's memory compressor can't
 * compress/evict them (we saw actual RSS < intended resident -> slow "hits"). -1 = auto
 * (ON on macOS where it matters and RLIMIT_MEMLOCK is permissive; OFF elsewhere, where the
 * limit is often tiny and must be raised by hand), 0 = off, 1 = force. */
static int g_mlock=-1;
static int mem_should_wire(void){
    if(g_mlock>=0) return g_mlock;
#if defined(__APPLE__)
    return 1;                                     /* macOS: default ON */
#else
    return 0;                                     /* Linux/altri: opt-in via MLOCK=1 / opt-in */
#endif
}
/* Inchioda [addr,addr+len) in RAM fisica. No-op fuori da POSIX (Windows ecc.).
 * EN: wire [addr,addr+len) into physical RAM. No-op off POSIX (Windows, etc.). */
static int mem_wire(void *addr, size_t len){
#if defined(__APPLE__) || defined(__linux__)
    return mlock(addr, len);
#elif defined(_WIN32)
    return compat_mlock(addr, len);   /* VirtualLock + working-set growth */
#else
    (void)addr; (void)len; return 0;
#endif
}
/* Inchioda tutti gli slab degli expert pinnati (pesi + scale). Non fatale se fallisce.
 * EN: wire all pinned-expert slabs (weights + scales). Non-fatal on failure. */
static void pin_wire(Model *m){
    if(!mem_should_wire()) return;
    Cfg *c=&m->c; double t0=now_s(); int64_t wired=0; long failed=0;
    for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
        ESlot *s=&m->pin[i][z];
        if(s->slab){  if(mem_wire(s->slab, s->slab_cap)==0) wired+=s->slab_cap; else failed++; }
        if(s->fslab){ size_t fl=(size_t)s->fslab_cap*sizeof(float);
                      if(mem_wire(s->fslab, fl)==0) wired+=fl; else failed++; }
    }
    if(failed)
        fprintf(stderr,"[PIN] mlock: %.1f GB wired, %ld allocations failed "
            "(raise the limit: ulimit -l unlimited) in %.0fs\n", wired/1e9, failed, now_s()-t0);
    else
        fprintf(stderr,"[PIN] mlock: %.1f GB wired in physical RAM "
            "(no compression) in %.0fs\n", wired/1e9, now_s()-t0);
}

typedef struct { int l,e; uint32_t c; } PinRec;
static int pin_rec_cmp(const void *a,const void *b){
    const PinRec *x=a,*y=b; return x->c<y->c?1:x->c>y->c?-1:0;
}
static void pin_load(Model *m, const char *statspath, double gb){
    FILE *f=fopen(statspath,"r"); if(!f){ perror(statspath); return; }
    Cfg *c=&m->c; int cap=(c->n_layers+1)*c->n_experts;
    PinRec *r=malloc((size_t)cap*sizeof(PinRec)); int n=0;
    unsigned char *seen=calloc((size_t)(c->n_layers+1)*c->n_experts,1);
    int l,e; uint32_t cnt;
    while(n<cap && fscanf(f,"%d %d %u",&l,&e,&cnt)==3){
        int ok = l>=0 && e>=0 && e<c->n_experts &&
                 ((l<c->n_layers && m->L[l].sparse) || (l==c->n_layers && m->has_mtp));
        int64_t key=(int64_t)l*c->n_experts+e;
        if(ok&&!seen[key]){ r[n++]=(PinRec){l,e,cnt}; seen[key]=1; }
    }
    fclose(f);
    int fill=getenv("PIN_FILL")?atoi(getenv("PIN_FILL")):0;
#ifdef COLI_CUDA
    if(!getenv("PIN_FILL")&&g_cuda_release_host) fill=1;
#endif
    if(fill) for(int li=0;li<=c->n_layers;li++){
        int sparse=(li<c->n_layers&&m->L[li].sparse)||(li==c->n_layers&&m->has_mtp);
        if(sparse) for(int ei=0;ei<c->n_experts;ei++) if(!seen[(int64_t)li*c->n_experts+ei])
            r[n++]=(PinRec){li,ei,0};
    }
    free(seen);
    qsort(r,(size_t)n,sizeof(*r),pin_rec_cmp);
    int64_t eb=expert_bytes_probe(m,m->ebits);
    int npin=(int)(gb*1e9/eb); if(npin>n) npin=n;
    if(npin<1){ free(r); return; }
    int *cnt_l=calloc(c->n_layers+1,sizeof(int));   /* +1: riga MTP */
    for(int a=0;a<npin;a++) cnt_l[r[a].l]++;
    for(int i=0;i<=c->n_layers;i++) if(cnt_l[i]) m->pin[i]=calloc(cnt_l[i],sizeof(ESlot));
    int *slot_of=malloc((size_t)npin*sizeof(int)), *next=calloc(c->n_layers+1,sizeof(int));
    for(int a=0;a<npin;a++) slot_of[a]=next[r[a].l]++;
    for(int i=0;i<=c->n_layers;i++) m->npin[i]=cnt_l[i];
    double t0=now_s();
#ifdef COLI_CUDA
    double remaining[COLI_CUDA_MAX_DEVICES]={0}, placed_b[COLI_CUDA_MAX_DEVICES]={0};
    int placed_n[COLI_CUDA_MAX_DEVICES]={0}, gpu_prefix=0;
    double budget=g_cuda_expert_gb*1e9, safe_total=0;
    if(g_cuda_enabled&&g_cuda_expert_gb>0) for(int i=0;i<g_cuda_ndev;i++){
        size_t free_b=0,total_b=0;
        if(coli_cuda_mem_info(g_cuda_devices[i],&free_b,&total_b)){
            remaining[i]=(double)free_b-(double)g_cuda_dense_projected[i]-2e9;
            if(remaining[i]<0) remaining[i]=0; safe_total+=remaining[i];
        }
    }
    if(budget>safe_total) budget=safe_total;
    if(g_cuda_enabled&&g_cuda_release_host&&budget>0){ gpu_prefix=(int)(budget/eb)+g_cuda_ndev; if(gpu_prefix>npin)gpu_prefix=npin; }
#else
    int gpu_prefix=0;
#endif
    /* Load the VRAM-ranked prefix first.  Once uploaded its host backing is
     * released before the disjoint RAM-ranked suffix is allocated. */
    #pragma omp parallel for schedule(dynamic,1)
    for(int a=0;a<(gpu_prefix?gpu_prefix:npin);a++)
        expert_load(m,r[a].l,r[a].e,&m->pin[r[a].l][slot_of[a]],1);
    m->resident_bytes+=(int64_t)(gpu_prefix?gpu_prefix:npin)*eb;
#ifdef COLI_CUDA
    if(g_cuda_enabled && g_cuda_expert_gb>0){
        int gpu_limit=gpu_prefix?gpu_prefix:npin;
        for(int a=0;a<gpu_limit && m->gpu_expert_bytes<budget;a++){
            int li=r[a].l;
            { ESlot *s=&m->pin[li][slot_of[a]];
                int64_t need=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
                if(m->gpu_expert_bytes+need>budget) break;
                int tried[COLI_CUDA_MAX_DEVICES]={0}, placed=0;
                for(int attempt=0;attempt<g_cuda_ndev && !placed;attempt++){
                    int best=-1;
                    for(int i=0;i<g_cuda_ndev;i++) if(!tried[i] && remaining[i]>=need &&
                        (best<0||placed_b[i]<placed_b[best])) best=i;
                    if(best<0) break;
                    tried[best]=1;
                    s->g.cuda_device=s->u.cuda_device=s->d.cuda_device=g_cuda_devices[best];
                    s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=1;
                    if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                        int64_t actual=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                                      +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                                      +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                        m->gpu_expert_count++; m->gpu_expert_bytes+=actual;
                        remaining[best]-=actual; placed_b[best]+=actual; placed_n[best]++;
                        if(g_cuda_release_host) expert_host_release(m,s);
                        placed=1;
                    } else {
                        qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                        s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                        remaining[best]=0;             /* device rejected its projected capacity */
                    }
                }
            }
        }
        fprintf(stderr,"[CUDA] hot expert tier: %d/%d experts, VRAM %.2f GB (total budget %.1f GB)\n",
            m->gpu_expert_count,npin,m->gpu_expert_bytes/1e9,g_cuda_expert_gb);
        for(int i=0;i<g_cuda_ndev;i++) fprintf(stderr,"[CUDA]   device %d: %d experts, %.2f GB\n",
            g_cuda_devices[i],placed_n[i],placed_b[i]/1e9);
    }
#endif
    if(gpu_prefix>0&&gpu_prefix<npin){
        #pragma omp parallel for schedule(dynamic,1)
        for(int a=gpu_prefix;a<npin;a++)
            expert_load(m,r[a].l,r[a].e,&m->pin[r[a].l][slot_of[a]],1);
        m->resident_bytes+=(int64_t)(npin-gpu_prefix)*eb;
    }
    fprintf(stderr,"[PIN] placement: %d VRAM + %d RAM expert (%.1f GB warm) in %.0fs da %s\n",
        m->gpu_expert_count,npin-m->gpu_expert_count,(npin-m->gpu_expert_count)*eb/1e9,now_s()-t0,statspath);
    pin_wire(m);                                   /* inchioda in RAM (no compressione) / wire in RAM (no compression) */
    free(r); free(cnt_l); free(slot_of); free(next);
}

static double g_mem_avail_boot=0;   /* MemAvailable all'avvio, prima di caricare il modello */
/* RAM disponibile ADESSO (GB): e' il tetto vero, non il totale. Linux: MemAvailable
 * da /proc/meminfo. macOS: pagine free+inactive+purgeable da host_statistics64
 * (stessa semantica: recuperabili senza swap). Senza questo ramo il fallback
 * "assumo 8 GB" castrava la cache expert proprio sulle macchine con piu' RAM. */
static double mem_available_gb(void){
#ifdef __APPLE__
    mach_msg_type_number_t cnt=HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&cnt)!=KERN_SUCCESS) return 0;
    return ((double)vm.free_count+(double)vm.inactive_count+(double)vm.purgeable_count)
           * (double)sysconf(_SC_PAGESIZE) / 1e9;
#elif defined(_WIN32)
    double total, avail;
    compat_meminfo(&total, &avail);
    return avail;
#else
    FILE *f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char ln[256]; double kb=0;
    while(fgets(ln,sizeof(ln),f)) if(sscanf(ln,"MemAvailable: %lf",&kb)==1) break;
    fclose(f); return kb/1e6;
#endif
}

static int kv_slot_count(void){
    if(!getenv("SERVE")) return 1;
    return getenv("KV_SLOTS")?atoi(getenv("KV_SLOTS")):1;
}

static double kv_pool_bytes(Model *m, int max_ctx){
    Cfg *c=&m->c; double one=(double)(c->n_layers+1)*max_ctx*(c->kv_lora+c->qk_rope)*4.0;
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        one+=(double)max_ctx*c->index_hd*4.0;
    int slots=kv_slot_count(); if(slots<1||slots>16) slots=1;
    return one*slots;
}

/* byte disponibili per gli expert (pin + LRU) nel budget — specchio del conto di cap_for_ram */
static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int64_t eb=expert_bytes_probe(m,ebits);
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double slack = 1.2e9 + 2.5e9 + 64.0*(double)eb
        + kv_pool_bytes(m,max_ctx)
        + (double)max_ctx*c->n_heads*(c->qk_nope+c->v_head)*4.0;
    return ram_gb*1e9 - (double)m->resident_bytes - slack;
}

/* clampa la cache expert a un budget RAM (GB): cap t.c. residente + cache + slack <= budget.
 * ram_gb<=0 -> budget AUTO = 88% della RAM disponibile adesso (lascia respiro a OS+wrapper:
 * sforare = OOM-kill del kernel a meta' generazione, molto peggio di una cache piu' piccola). */
static void cap_for_ram(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    if(m->has_mtp) nsp+=2;                       /* riga cache MTP: conta ~doppia (expert int8 = 2x int4) */
    int64_t eb=expert_bytes_probe(m,ebits);
    int auto_b = ram_gb<=0;
    if(auto_b){ ram_gb = g_mem_avail_boot*0.88;   /* misurata PRIMA del load: il residente gia'
                                                   * allocato viene sottratto sotto, non due volte */
        if(ram_gb<4){ fprintf(stderr,"[RAM] MemAvailable is unreadable or too low; assuming 8 GB\n"); ram_gb=8; } }
    /* slack ONESTO, non forfettario (l'OOM del 2026-07-04 veniva da qui):
     *  ws[64] slab del working-set (si materializzano TUTTI nel prefill batch-union),
     *  KV cache a max_ctx, kvb_all della ricostruzione k/v in attention,
     *  attivazioni+logits+overhead ~1.2 GB */
    double ws_b  = 64.0*(double)eb;
    double kv_b  = kv_pool_bytes(m,max_ctx);
    double kvb_b = (double)max_ctx*c->n_heads*(c->qk_nope+c->v_head)*4.0;
    /* RISERVA PAGE-CACHE (misurato 2026-07-06): strangolarla fa crollare le pread
     * buffered da ~800 a ~180 MB/s — gli ultimi GB di LRU rendono MENO di quanto
     * costino in banda disco persa. 2.5 GB restano SEMPRE al kernel. */
    double pc_b  = 2.5e9;
    double slack = 1.2e9 + pc_b + ws_b + kv_b + kvb_b;
    double avail = ram_gb*1e9 - (double)m->resident_bytes - slack;
    int capmax = (avail>0 && nsp>0) ? (int)(avail/((double)nsp*eb)) : 0;
    if(capmax<1) capmax=1;
    if(capmax < m->ecap){
        fprintf(stderr,"[RAM_GB=%.1f%s] resident %.1f GB + reserve %.1f GB (ws %.1f, KV %dx%d %.1f, kvb %.1f), "
            "experts %.1f MB x %d layers -> cap lowered %d->%d (projected peak %.1f GB)\n",
            ram_gb,auto_b?" auto":"",m->resident_bytes/1e9,slack/1e9,ws_b/1e9,
            kv_slot_count(),max_ctx,kv_b/1e9,kvb_b/1e9,
            eb/1e6, nsp, m->ecap, capmax,
            (m->resident_bytes + (double)capmax*nsp*eb + slack)/1e9);
        m->ecap=capmax;
    } else {
        /* AUTO-RAISE (issue #12): il budget consente PIU' cache di quella chiesta.
         * Senza questo, una macchina da 128 GB girava con la LRU di una da 16
         * (cap=8 di default in coli): hit 23-28% con decine di GB inutilizzati.
         * Tetto a n_experts: oltre, ogni layer avrebbe slot che non puo' riempire.
         * CAP_RAISE=0 ripristina il comportamento fisso. */
        int raise_on = getenv("CAP_RAISE")?atoi(getenv("CAP_RAISE")):1;
        int newcap = capmax>c->n_experts ? c->n_experts : capmax;
        if(raise_on && newcap>m->ecap){
            for(int i=0;i<=c->n_layers;i++) if(m->ecache[i]){
                m->ecache[i]=realloc(m->ecache[i],(size_t)newcap*sizeof(ESlot));
                memset(m->ecache[i]+m->ecap,0,(size_t)(newcap-m->ecap)*sizeof(ESlot));
            }
            fprintf(stderr,"[RAM_GB=%.1f%s] cap raised %d->%d: budget allows it "
                "(projected peak %.1f GB; set CAP_RAISE=0 to disable)\n",
                ram_gb, auto_b?" auto":"", m->ecap, newcap,
                (m->resident_bytes + (double)newcap*nsp*eb + slack)/1e9);
            m->ecap=newcap;
        } else
            fprintf(stderr,"[RAM_GB=%.1f%s] cap=%d ok (projected peak %.1f GB)\n", ram_gb, auto_b?" auto":"", m->ecap,
                (m->resident_bytes + (double)m->ecap*nsp*eb + slack)/1e9);
    }
}

int main(int argc, char **argv){
    /* ---- Permanent OpenMP hot-thread tuning. The per-expert matmul regions are
     * tiny and back-to-back; with the default passive wait policy libgomp parks
     * the worker team between regions and the re-wake latency dominates. Keeping
     * the threads hot (active spin) collapses that overhead — measured matmul
     * time 66.9s -> 20.9s on the Zen5 build, with no change to numerical output.
     *
     * libgomp reads the OMP_ / GOMP_ vars in a CONSTRUCTOR that runs before
     * main(), so setenv() here and continuing would be too late (verified:
     * setenv-in-main is ignored by the already-initialised runtime). Instead, on
     * first entry seed the winning defaults — respecting anything the user
     * already set (overwrite=0) — then re-exec self once so a fresh libgomp
     * constructor picks them up. The COLI_OMP_TUNED sentinel guards the exec so
     * we re-exec at most once. Fully overridable: any explicit OMP_/GOMP_ env the
     * user sets wins (overwrite=0), pre-setting COLI_OMP_TUNED=1 skips the
     * re-exec entirely (runs with whatever policy the environment already has),
     * and COLI_NO_OMP_TUNE=1 is a documented kill-switch that disables the whole
     * re-exec + tuning path (distinct from the internal COLI_OMP_TUNED sentinel).
     *
     * Must remain the FIRST statement in main(): argv is passed verbatim to execv(). */
    if(!getenv("COLI_OMP_TUNED") && !getenv("COLI_NO_OMP_TUNE") &&
       !getenv("COLI_CUDA") && !getenv("COLI_METAL")){
        setenv("OMP_WAIT_POLICY","active",0);  /* keep the team hot across the tiny per-expert matmul regions */
        setenv("GOMP_SPINCOUNT","200000",0);   /* spin briefly, then yield so long disk waits don't burn a core */
        setenv("OMP_PROC_BIND","close",0);     /* pack the team onto adjacent cores for cache locality */
        setenv("OMP_DYNAMIC","FALSE",0);       /* fixed team size: no per-region thread-count churn */
        setenv("COLI_OMP_TUNED","1",1);
#ifdef __linux__
        fprintf(stderr,"[OMP] hot-thread tuning: re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/self/exe", argv);         /* returns only on failure -> fall through and run untuned */
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if(getenv("I4_ACC512")) g_i4_acc512=atoi(getenv("I4_ACC512"))!=0;
    if(getenv("I4_ACC512_TEST")){
        if(!i4_acc512_selftest()) return 1;
        puts("AVX512 i4 selftest: ok"); return 0;
    }
#endif
    const char *snap=getenv("SNAP"); if(!snap){fprintf(stderr,"SNAP=<dir>\n");return 1;}
    g_nopack = getenv("NOPACK")?1:0;
    g_drop = getenv("DROP")?1:0;
    g_prefetch = getenv("PREFETCH")?atoi(getenv("PREFETCH")):0;
    g_mmap = getenv("COLI_MMAP")?atoi(getenv("COLI_MMAP")):0;
    if(g_mmap) fprintf(stderr,"[MMAP] expert = viste zero-copy nei file (page cache = cache)\n");
    g_topk = getenv("TOPK")?atoi(getenv("TOPK")):0;
    g_topp = getenv("TOPP")?atof(getenv("TOPP")):0;
    const char *policy=getenv("COLI_POLICY"); if(!policy) policy="quality";
    int experimental=!strcmp(policy,"experimental-fast");
    if(strcmp(policy,"quality")&&strcmp(policy,"balanced")&&!experimental){
        fprintf(stderr,"COLI_POLICY non valida: quality, balanced o experimental-fast\n"); return 2;
    }
    if(!experimental&&(g_topk>0||g_topp>0)){
        fprintf(stderr,"[policy] --topp/--topk drop low-weight experts (~1.6x fewer reads, small quality cost)\n");
    }
    g_mlock  = getenv("MLOCK")?atoi(getenv("MLOCK")):-1;   /* -1 auto (ON macOS), 0 off, 1 force / auto (ON macOS), 0 off, 1 force */
    g_spec = getenv("SPEC")?atoi(getenv("SPEC")):1;
    g_draft = getenv("DRAFT")?atoi(getenv("DRAFT")):-1;
    g_no_fused_pair = getenv("COLI_NO_FUSED_PAIR")?atoi(getenv("COLI_NO_FUSED_PAIR")):0;   /* -1 = auto: 3 se MTP, 0 senza */
    g_looka = getenv("LOOKA")?atoi(getenv("LOOKA")):0;    /* 1 = misura predicibilita' routing */
    g_pilot = getenv("PILOT")?atoi(getenv("PILOT")):0;    /* 1 = prefetch pilotato dal router */
    g_pilot_real = getenv("PILOT_REAL")?atoi(getenv("PILOT_REAL")):0; /* default OFF: load VERI cross-layer (value-preserving prefetch); PILOT_REAL=1 opta in */
    if(g_pilot_real) g_pilot=1;                           /* PILOT_REAL implica il pilota attivo */
    /* Default K: hint-only PILOT keeps 8 (WILLNEED hints are free, no eviction).
     * Under PILOT_REAL the speculative loads are REAL and create LRU eviction
     * pressure, so at ~28% mispredict a large K thrashes the cache — default to 6
     * (best-measured this session) unless the user set PILOT_K explicitly. */
    g_pilot_k = getenv("PILOT_K")?atoi(getenv("PILOT_K")):(g_pilot_real?6:8);
    if(g_pilot_k<1) g_pilot_k=1;
    g_pipe = getenv("PIPE")?atoi(getenv("PIPE")):0;       /* default OFF: overlap expert load ‖ matmul (byte-identical; reorders I/O). PIPE=1 opts in */
    g_pipe_nw = getenv("PIPE_WORKERS")?atoi(getenv("PIPE_WORKERS")):8; /* I/O worker threads */
    if(g_pipe_nw<1) g_pipe_nw=1;
    g_direct = getenv("DIRECT")?atoi(getenv("DIRECT")):0;
    g_idot = getenv("IDOT")?atoi(getenv("IDOT")):1;        /* 0 = kernel f32 esatti (A/B) */
    if(getenv("ROUTE_TRACE")&&*getenv("ROUTE_TRACE")){
        g_route_fp=fopen(getenv("ROUTE_TRACE"),"w");
        if(!g_route_fp) fprintf(stderr,"[ROUTE_TRACE] cannot open %s\n",getenv("ROUTE_TRACE"));
        else fprintf(stderr,"[ROUTE_TRACE] logging routing to %s\n",getenv("ROUTE_TRACE"));
    }
    g_repin = getenv("REPIN")?atoi(getenv("REPIN")):0;     /* RFC: re-pin ogni n token emessi (0=off) / live re-pin every n emitted tokens (0=off) */
    g_absorb = getenv("ABSORB")?atoi(getenv("ABSORB")):-1; /* -1 auto: assorbita per S<=4 */
    g_dsa_force = getenv("DSA_FORCE")?atoi(getenv("DSA_FORCE")):0;
    g_lmhead_exact = getenv("LMHEAD_EXACT")?atoi(getenv("LMHEAD_EXACT")):0;
    /* matmul_qt documenta la soglia int4-IDOT come "configurabile con I4S" ma il getenv non
     * c'era: la variabile non aveva alcun effetto. I4S=<n> -> IDOT int4 solo per S>=n.
     * EN: matmul_qt documents the int4 IDOT threshold as "configurable via I4S", but the
     * getenv was missing, so the knob did nothing. I4S=<n> -> int4 IDOT only for S>=n. */
    if(getenv("I4S")) g_i4s=atoi(getenv("I4S"));
    g_temp = getenv("TEMP")?atof(getenv("TEMP")):-1;       /* -1 = auto (1.0 chat/testo, greedy altrove) */
    g_nuc  = getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.90f;  /* piu' stretto dell'ufficiale 0.95: la coda int4 e' rumore */
    if(getenv("SEED")) g_rng = (uint64_t)atoll(getenv("SEED"))*0x9E3779B97F4A7C15ULL+1;
    else { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); g_rng ^= (uint64_t)ts.tv_nsec<<20 ^ (uint64_t)getpid(); }
    if(g_draft>63) g_draft=63;                             /* -1 = auto, risolto dopo model_init */
    int cap  = argc>1?atoi(argv[1]):64;
    int ebits= argc>2?atoi(argv[2]):8;
    int dbits= argc>3?atoi(argv[3]):ebits;
    if(getenv("SERVE") && (kv_slot_count()<1 || kv_slot_count()>16)){
        fprintf(stderr,"KV_SLOTS must be between 1 and 16\n"); return 2;
    }
#ifdef COLI_CUDA
    if(getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))){
        const char *one=getenv("COLI_GPU"), *many=getenv("COLI_GPUS");
        if(one&&many){ fprintf(stderr,"use COLI_GPU or COLI_GPUS, not both\n"); return 2; }
        if(many) g_cuda_ndev=parse_cuda_devices(many,g_cuda_devices);
        else if(one) g_cuda_ndev=parse_cuda_devices(one,g_cuda_devices);
        else { g_cuda_ndev=1; g_cuda_devices[0]=0; }
        if(g_cuda_ndev<1){ fprintf(stderr,"invalid COLI_GPUS: use a list such as 0,1,2\n"); return 2; }
        g_cuda_enabled=coli_cuda_init(g_cuda_devices,g_cuda_ndev);
        if(!g_cuda_enabled){ fprintf(stderr,"[CUDA] requested backend is unavailable\n"); return 2; }
    }
    g_cuda_dense=getenv("CUDA_DENSE")?atoi(getenv("CUDA_DENSE")):0;
    g_cuda_expert_gb=getenv("CUDA_EXPERT_GB")?atof(getenv("CUDA_EXPERT_GB")):0;
    g_cuda_release_host=getenv("CUDA_RELEASE_HOST")?atoi(getenv("CUDA_RELEASE_HOST")):(g_cuda_ndev>1);
    if((getenv("COLI_GPU")||getenv("COLI_GPUS"))&&!g_cuda_enabled){ fprintf(stderr,"COLI_GPU(S) requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_dense&&!g_cuda_enabled){ fprintf(stderr,"CUDA_DENSE requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_expert_gb>0 && !g_cuda_enabled){ fprintf(stderr,"CUDA_EXPERT_GB requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_enabled) fprintf(stderr,"[CUDA] mode: routed experts%s%s\n",
        g_cuda_dense?" + resident dense tensors":" only (resident dense on CPU)",
        g_cuda_release_host?"; VRAM experts without host backing":"");
#else
    if((getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))) ||
       getenv("COLI_GPU") || getenv("COLI_GPUS") ||
       (getenv("CUDA_DENSE") && atoi(getenv("CUDA_DENSE"))) ||
       (getenv("CUDA_EXPERT_GB") && atof(getenv("CUDA_EXPERT_GB"))>0)){
        fprintf(stderr,"CUDA was requested, but this binary is CPU-only; rebuild with: make CUDA=1\n");
        return 2;
    }
#endif
#ifdef COLI_METAL
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))){
        g_metal_enabled = coli_metal_init();
        if(!g_metal_enabled){ fprintf(stderr,"[METAL] backend requested but not available\n"); return 2; }
        fprintf(stderr,"[METAL] mode: batched routed experts on GPU (unified-memory zero-copy)\n");
        if(getenv("COLI_METAL_SPIN") && atoi(getenv("COLI_METAL_SPIN"))){ coli_metal_spin_start(); fprintf(stderr,"[METAL] keep-alive spinner ON\n"); }
        if(getenv("COLI_METAL_GEMM_MIN")) g_metal_gemm_min=atoi(getenv("COLI_METAL_GEMM_MIN"));
    }
#else
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))){
        fprintf(stderr,"METAL was requested, but this binary has no Metal backend; rebuild with: make METAL=1\n");
        return 2;
    }
#endif
    printf("== GLM C engine (glm_moe_dsa), cache=%d experts/layer | experts@%d-bit dense@%d-bit | idot: " IDOT_KERNEL " ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    if(g_draft<0){
#ifdef COLI_CUDA
        g_draft = (m.has_mtp&&!g_cuda_enabled) ? 3 : 0;
#else
        g_draft = m.has_mtp ? 3 : 0;
#endif
    }
    if(getenv("DSA_TOPK")) m.c.index_topk=atoi(getenv("DSA_TOPK"));   /* override per test */
    printf("loaded in %.2fs | resident dense: %.2f MB | layers=%d experts=%d | MTP %s (draft=%d)\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), m.c.n_layers, m.c.n_experts,
           m.has_mtp?"ACTIVE":"absent", g_draft);
    /* anche su stderr: e' il canale che le UI (coli) mostrano all'utente */
    fprintf(stderr,"[MTP] %s (draft=%d)\n", m.has_mtp?"active: native speculative decoding":"absent", g_draft);
    if(!strncmp(snap,"/mnt/",5))
        fprintf(stderr,"WARNING: the model is on %s (slow 9p/Windows filesystem; fadvise is ineffective).\n"
                       "         Keep it on ext4 (for example, /home/...) for memory efficiency and speed.\n", snap);
    /* HOT-STORE: PIN=<statsfile> [PIN_GB=g] -> top expert per frequenza fissi in RAM.
     * Va PRIMA di cap_for_ram: i pinnati contano nel residente. */
    if(getenv("PIN")) pin_load(&m, getenv("PIN"), getenv("PIN_GB")?atof(getenv("PIN_GB")):10.0);
    if(getenv("COUPLE")&&*getenv("COUPLE")){    /* coupling-scored cross-layer prefetch */
        g_couple_k=getenv("COUPLE_K")?atoi(getenv("COUPLE_K")):8;
        if(g_couple_k<1)g_couple_k=1; if(g_couple_k>32)g_couple_k=32;
        g_couple_d=getenv("COUPLE_D")?atoi(getenv("COUPLE_D")):1;
        if(g_couple_d<1)g_couple_d=1; if(g_couple_d>2)g_couple_d=2;
        couple_load(&m, getenv("COUPLE"));
    }
    /* CACHE CHE IMPARA: l'uso degli expert si accumula in <SNAP>/.coli_usage tra le sessioni;
     * all'avvio i piu' usati vengono auto-pinnati in RAM (meta' del budget expert: il pin
     * conosce la TUA storia, la LRU si adatta alla sessione). AUTOPIN=0 disattiva. */
    { double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default di run_serve */
      snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
      int64_t hist = usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",(long long)hist,g_usage_path);
      int autopin = getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN") && autopin && hist>=5000){
          /* quota pin proporzionale alla FIDUCIA nella storia: con pochi dati il pin
           * sbaglia expert e ruba slot alla LRU adattiva; a regime (>=200k selezioni,
           * qualche ora di chat) arriva a meta' del budget expert. */
          double conf = (double)hist/200000.0; if(conf>1) conf=1;
          double pin_gb = expert_avail(&m,ram_env,ebits,est_ctx)*0.5*conf/1e9;
          if(pin_gb>=0.5) pin_load(&m, g_usage_path, pin_gb);
      }
      /* SEMPRE: senza clamp la LRU cresce fino a cap*76 layer = decine di GB -> OOM-kill.
       * RAM_GB assente o <=0 = budget automatico da MemAvailable. */
      cap_for_ram(&m, ram_env, ebits, est_ctx); }
    const char *stats=getenv("STATS");   /* STATS=<file> -> istogramma uso expert a fine run */

    /* modo scoring per benchmark: SCORE=<requests.txt> -> log-likelihood per riga */
    if(getenv("SCORE")){ run_score(&m, getenv("SCORE")); if(stats) stats_dump(&m,stats); return 0; }

    /* modo serve persistente per la CLI 'coli': SERVE=1 */
    if(getenv("SERVE")){
        if(getenv("SERVE_BATCH") && atoi(getenv("SERVE_BATCH"))) run_serve_mux(&m,snap);
        else run_serve(&m,snap);
        if(stats) stats_dump(&m,stats); return 0;
    }

    /* modo testo reale: PROMPT="..." [NGEN=n] -> tokenizza, genera, detokenizza */
    if(getenv("PROMPT")){
        int ngen=getenv("NGEN")?atoi(getenv("NGEN")):64;
        run_text(&m, snap, getenv("PROMPT"), ngen);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    /* altrimenti: validazione contro l'oracolo (ref_glm.json) */
    const char *refpath=getenv("REF")?getenv("REF"):"ref_glm.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); size_t got=fread(b,1,n,f); b[got]=0; fclose(f);
    if((long)got!=n) fprintf(stderr,"warning: short read on %s (%ld of %ld)\n",refpath,(long)got,n);
    char *ar=NULL; jval *ref=json_parse(b,&ar);
    int np=0,nfull=0; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    if(!prompt||!full||np<1||nfull<np){ fprintf(stderr,"ref file missing prompt_ids/full_ids or empty\n"); return 1; }
    int n_new=nfull-np;
    /* L'oracolo (ref_glm.json in repo) e' del modello TINY: contro il 744B da' 0/20
     * garantito su OGNI piattaforma (prompt-token tiny = spazzatura per il modello vero).
     * Non e' un bug del motore — vedi #76. */
    { int maxid=0; for(int i=0;i<nfull;i++) if(full[i]>maxid) maxid=full[i];
      if(m.c.vocab>1000 && maxid<1000 && !getenv("REF_FORCE")){
        fprintf(stderr,"ERRORE: ref_glm.json e' l'oracolo del modello TINY (token max %d, ma il tuo vocab e' %d).\n"
                       "        Self-test motore:  SNAP=./glm_tiny TF=1 ./glm 64 16 16   (atteso 32/32)\n"
                       "        Prova reale:       PROMPT=\"Ciao\" NGEN=32 SNAP=<modello> ./glm 64\n"
                       "        REF_FORCE=1 per eseguire comunque il confronto (senza senso).\n", maxid, m.c.vocab);
        return 1;
      } }

    if(getenv("REPLAY")){
        run_replay(&m,full,nfull,np);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    if(getenv("TF")){
        int *tf=read_arr(ref,"tf_pred",&(int){0});
        int *pred=malloc(nfull*sizeof(int)); double tt=now_s();
        forward_all(&m, full, nfull, pred); double tdt=now_s()-tt;
        int ok=0; for(int i=0;i<nfull;i++){
            if(pred[i]==tf[i]) ok++;
            else fprintf(stderr,"[ORACLE] mismatch pos=%d expected=%d got=%d\n",i,tf[i],pred[i]);
        }
        printf("PREFILL (teacher-forcing) C vs oracle: %d/%d positions | %.1f pos/s\n",
            ok,nfull,nfull/tdt);
        if(ok<nfull) fprintf(stderr,
            "[ORACLE] %d/%d mismatches — run: TF=1 DEBUG_LOGITS=1 for top-5 logit dump\n",
            nfull-ok,nfull);
        profile_print(&m,tdt);
#ifdef COLI_CUDA
        if(g_cuda_enabled) cuda_stats_print();
#endif
        return 0;
    }
    int *out=malloc((np+n_new)*sizeof(int));
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nReference (oracle): "); for(int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nGLM C engine      : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double tot=m.hits+m.miss;
    printf("N-gram speculation (DRAFT=%d): %.2f tokens/forward (%llu forwards per %llu tokens)\n",
        g_draft, m.n_fw?(double)m.n_emit/m.n_fw:1.0, (unsigned long long)m.n_fw, (unsigned long long)m.n_emit);
    printf("Expert cache hit rate: %.1f%% (hit=%llu miss=%llu) | RSS: %.2f GB | %.1f tok/s\n",
           tot?100.0*m.hits/tot:0.0, (unsigned long long)m.hits, (unsigned long long)m.miss, rss_gb(), n_new/dt);
    profile_print(&m,dt);
#ifdef COLI_CUDA
    if(m.gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m.gpu_expert_count,m.gpu_expert_bytes/1e9,(unsigned long long)m.gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    if(g_looka){
        const char *nm[3]={"previous token (=SPEC prefetch)","layer input, skip attention","next layer (one step ahead)"};
        printf("LOOKAHEAD routing — recall of true experts in predicted top-8:\n");
        for(int i=0;i<3;i++) printf("  %-38s %5.1f%%  (%lld/%lld)\n", nm[i],
            la_tot[i]?100.0*la_hit[i]/la_tot[i]:0.0, (long long)la_hit[i], (long long)la_tot[i]);
    }
    if(stats) stats_dump(&m,stats);
    return 0;
}
