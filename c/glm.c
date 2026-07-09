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
 * Validazione: stessi token id di ref_glm.json (oracolo transformers, c/make_glm_oracle.py).
 *   build: make glm   run: SNAP=./glm_tiny ./glm <cap> <expert_bits> <dense_bits>
 *   TF=1 -> teacher-forcing (valida il prefill su tutta la sequenza)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>                             /* mlock: inchioda le pagine in RAM / wire pages into RAM */
#endif
#include "st.h"
#include "tok.h"
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){            /* somma orizzontale di 8 float */
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>                             /* Apple Silicon / aarch64: kernel NEON */
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
typedef struct { int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I; } QT;
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
    Cfg c; shards S;
    int ebits, dbits;                            /* bit expert / bit densa */
    QT embed, lm_head; float *final_norm;
    Layer *L;
    /* KV-cache MLA COMPRESSA: per token si tiene solo il latente normato [kv_lora] e
     * k_rot [qk_rope] (576 vs 32768 valori/token). k_nope e value si ricostruiscono al
     * volo con kv_b. E' cio' che rende gestibile il contesto su 15 GB (64 teste, no GQA). */
    float **Lc, **Rc; int max_t;
    int *kv_start;                               /* prima pos valida nella KV del layer (MTP: parziale) */
    ESlot **ecache; int *ecn; int ecap;          /* LRU expert per-layer */
    ESlot ws[64];                                /* working set del layer corrente (load paralleli) */
    ESlot **pin; int *npin;                      /* HOT-STORE: expert pinnati in RAM (mai evicted) */
    uint32_t **eusage;                           /* contatori uso expert per layer (per STATS/PIN) */
    /* DSA lightning indexer (attivo solo se i pesi out-idx-* sono presenti) */
    int has_dsa;
    QT *ix_wq, *ix_wk, *ix_wp;                   /* per layer FULL: wq_b, wk, weights_proj */
    float **ix_knw, **ix_knb;                    /* k_norm (LayerNorm, eps 1e-6) */
    float **Ic;                                  /* cache k dell'indexer per layer full [max_t*hd] */
    int *dsa_sel, *dsa_nsel; int dsa_scap;       /* selezione per posizione del batch corrente */
    /* testa MTP (layer n_layers, stile DeepSeek-V3): draft nativi ad alta acceptance */
    int has_mtp; Layer mtpL; QT eh_proj;
    float *enorm, *hnorm, *mtp_norm;
    float *hlast, *h_all;                        /* hidden pre-norm: ultima pos / tutte le pos batch */
    uint64_t mtp_prop, mtp_acc;                  /* statistica acceptance */
    int **eroute; int *enr;                      /* metodo C: routing dell'ULTIMO token per layer */
    uint64_t eclock, hits, miss, ereq;
    uint64_t n_fw, n_emit;                       /* metodo E: forward di decode / token emessi */
    double t_edisk, t_emm, t_attn, t_kvb, t_head;/* profiling: dove va il tempo (sempre attivo) */
    int64_t resident_bytes;
} Model;

static void usage_save(Model *m);        /* cache che impara: definita accanto a stats_dump */
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);   /* macOS: ru_maxrss in BYTE */
#else
    return r.ru_maxrss/(1024.0*1024.0);          /* Linux: in KB */
#endif
}
static float *falloc(int64_t n){ float *p=malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }

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
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += xs[i]*(float)lo; }
            y[(int64_t)s*O+o]=a*sc; } }
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
#elif defined(__AVX2__)
#define IDOT_KERNEL "avx2"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#else
#define IDOT_KERNEL "scalar"
#endif
static int g_idot=1;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int g_i4s=1;   /* SDOT presente: int4 IDOT conviene anche a S=1 (decode). Misurato
                       * su Apple M-series: +14%%, expert-matmul -16%%. EN: with SDOT, int4
                       * IDOT pays even at S=1 (decode); measured on Apple M-series. */
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

static void matmul_qt(float *y, const float *x, const QT *w, int S){
    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    /* int8 IDOT vince sempre (1.4-2.5x). int4 IDOT: l'autore su AVX2 trovo' che a S=1
     * non ripaga (soglia S>=2); ma su ARM/SDOT il singolo token CONVIENE (vedi g_i4s /
     * PR #9 per il gemello VNNI). Soglia configurabile con I4S.
     * EN: int8 IDOT always wins (1.4-2.5x). int4 IDOT: on AVX2 the author found S=1 didn't
     * pay (S>=2 gate); on ARM/SDOT single-token DOES pay (see g_i4s / PR #9 for the VNNI
     * twin). Threshold configurable via I4S. */
    if(g_idot && (w->fmt==1 || (w->fmt==2 && S>=g_i4s))){
        int I=w->I;
        int8_t *xq=malloc((size_t)S*I); float sxb[64]; float *sx=S<=64?sxb:falloc(S);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        free(xq); if(sx!=sxb) free(sx);
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
/* sceglie il formato da `bits`: >=16 f32, 5..8 int8, <=4 int4-packed */
static void qt_alloc(QT *t, int O, int I, int bits){
    t->O=O; t->I=I; t->qf=NULL; t->q8=NULL; t->q4=NULL; t->s=NULL;
    if(bits>=16){ t->fmt=0; t->qf=falloc((int64_t)O*I); }
    else if(bits>=5 || g_nopack){ t->fmt=1; t->q8=malloc((int64_t)O*I); t->s=falloc(O); }
    else if(bits>=3){ t->fmt=2; t->q4=malloc((int64_t)O*((I+1)/2)); t->s=falloc(O); }
    else { t->fmt=3; t->q4=malloc((int64_t)O*((I+3)/4)); t->s=falloc(O); }
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
    int half = c->qk_rope/2; float in[256]; memcpy(in,v,c->qk_rope*sizeof(float));
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
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
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
    if(c->n_group!=1){ fprintf(stderr,"questo motore assume n_group=1 (GLM-5.2)\n"); exit(1); }
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
        if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->q8=malloc(nb); t->s=falloc(O); } st_read_raw(&m->S,name,t->q8,drop); }
        else      { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->q4=malloc(nb); t->s=falloc(O); } st_read_raw(&m->S,name,t->q4,drop); }
        st_read_f32(&m->S,sn,t->s,drop);
    } else {
        if(!t->qf && !t->q8 && !t->q4) qt_alloc(t,O,I,bits);
        if(t->fmt==0) st_read_f32(&m->S,name,t->qf,drop);
        else { float *tmp=falloc((int64_t)O*I); st_read_f32(&m->S,name,tmp,drop); qt_fill(t,tmp,bits); free(tmp); }
    }
}
static QT qt_load(Model *m, const char *name, int O, int I, int bits){
    QT t; memset(&t,0,sizeof(t)); qt_from_disk(m,name,O,I,bits,0,&t); return t;
}
static float *ld(Model *m, const char *name){   /* tensore 1D f32 residente (norme/bias) */
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"manca %s\n",name);exit(1);}
    float *p=falloc(n); st_read_f32(&m->S,name,p,0); return p;
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
    m->eusage=calloc(NR,sizeof(uint32_t*));
    m->kv_start=calloc(NR,sizeof(int));
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
            fprintf(stderr,"[DSA] indexer attivo: attenzione sparsa top-%d oltre %d token di contesto\n",
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

/* carica un expert nello slot. Container pre-quantizzato: le 3 matrici sono contigue nel
 * file -> UNA pread coalescente da ~19 MB dentro `slab` (+ le scale in fslab); i QT sono
 * viste dentro lo slab (zero copie). Fallback per modelli non quantizzati (oracolo tiny).
 * THREAD-SAFE su slot distinti (pread posizionale, st_find read-only). */
static void expert_load(Model *m, int layer, int eid, ESlot *s){
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    char nm[3][288]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[300]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(!st_has(&m->S,qn)){                       /* fallback: tensori pieni, quantizza a runtime */
        qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
        qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
        qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
        s->eid=eid; return;
    }
    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm[k]); tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ fprintf(stderr,"manca %s\n",nm[k]); exit(1); }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    /* rialloca se lo slot (riusato tra layer) e' troppo piccolo per QUESTO expert:
     * pread oltre la mappatura = short-read o CORRUZIONE silenziosa dei vicini */
    if(!s->slab || wtot+8192 > s->slab_cap){
        free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n");exit(1);}
        s->slab_cap=wtot+8192;
    }
    if(!s->fslab || ftot > s->fslab_cap){ free(s->fslab); s->fslab=falloc(ftot); s->fslab_cap=ftot; }
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
            if(pread(tw[ord[0]]->fd, s->slab, wtot, off0)!=wtot){ perror("pread expert"); exit(1); }
            pos[ord[0]]=0; pos[ord[1]]=tw[ord[0]]->nbytes; pos[ord[2]]=tw[ord[0]]->nbytes+tw[ord[1]]->nbytes; done=1;
        }
    }
    if(!done){                                   /* non contigui: 3 pread bufferizzate */
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            if(pread(tw[k]->fd, s->slab+o, tw[k]->nbytes, tw[k]->off)!=tw[k]->nbytes){ perror("pread expert"); exit(1); }
            pos[k]=o; o+=tw[k]->nbytes; }
    }
    float *fp[3]; int64_t fo=0;                  /* scale (piccole) */
    for(int k=0;k<3;k++){
        if(pread(tq[k]->fd, (char*)(s->fslab+fo), tq[k]->nbytes, tq[k]->off)!=tq[k]->nbytes){ perror("pread qs"); exit(1); }
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
    s->eid=eid;
}

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
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    Cfg *c=&m->c; int H=c->n_heads, D=c->hidden, qh=c->qk_head, vh=c->v_head;
    int kvb_dim=H*(c->qk_nope+vh), Tk=pos_base+S;
    double ta0=now_s();
    float *ctx=falloc((int64_t)S*H*vh);
    float *Q=falloc((int64_t)S*H*qh);                  /* query (roped) dei token nuovi */
    float *QR=falloc((int64_t)S*c->q_lora), *comp=falloc(c->kv_lora+c->qk_rope);
    /* 1) per ogni token nuovo: query roped + latente normato e k_rot roped -> in cache.
     * QR tiene il residuo q_a per TUTTE le posizioni: serve anche all'indexer DSA. */
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D; int pos=pos_base+s;
        float *qresid=QR+(int64_t)s*c->q_lora;
        matmul_qt(qresid, xs, &l->q_a, 1);
        rmsnorm(qresid, qresid, l->q_a_ln, c->q_lora, c->eps);
        float *qfull=Q+(int64_t)s*H*qh; matmul_qt(qfull, qresid, &l->q_b, 1);
        for(int h=0;h<H;h++) rope_interleave(qfull+(int64_t)h*qh+c->qk_nope, pos, c);
        matmul_qt(comp, xs, &l->kv_a, 1);
        float *Ldst=m->Lc[layer]+(int64_t)pos*c->kv_lora, *Rdst=m->Rc[layer]+(int64_t)pos*c->qk_rope;
        memcpy(Ldst, comp, c->kv_lora*sizeof(float));
        rmsnorm(Ldst, Ldst, l->kv_a_ln, c->kv_lora, c->eps);     /* latente normato */
        memcpy(Rdst, comp+c->kv_lora, c->qk_rope*sizeof(float));
        rope_interleave(Rdst, pos, c);                            /* k_rot roped, condiviso fra teste */
    }
    /* ---- DSA lightning indexer ----
     * Layer FULL: k_idx dei token nuovi in cache + selezione top-k per query (riusata
     * dai layer SHARED successivi). Selezione attiva solo con contesto > index_topk
     * (o DSA_FORCE=1 per il test: selezionare TUTTO deve dare l'output denso esatto). */
    const int *dsel=NULL, *dnsel=NULL; int dtopk=0;
    if(m->has_dsa && layer<c->n_layers && m->kv_start[layer]==0){
        int nh=c->index_nh, hd=c->index_hd; dtopk=c->index_topk;
        if(c->idx_type[layer]){
            for(int s=0;s<S;s++){
                const float *xs=x+(int64_t)s*D; int pos=pos_base+s;
                float *kd=m->Ic[layer]+(int64_t)pos*hd;
                matmul_qt(kd, xs, &m->ix_wk[layer], 1);
                layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], hd, 1e-6f);
                rope_interleave(kd, pos, c);                 /* primi qk_rope dim, interleaved */
            }
            if((int64_t)S*dtopk > m->dsa_scap){
                free(m->dsa_sel); free(m->dsa_nsel);
                m->dsa_scap=(int64_t)S*dtopk;
                m->dsa_sel=malloc((size_t)m->dsa_scap*sizeof(int));
                m->dsa_nsel=malloc((size_t)S*sizeof(int));
            }
            #pragma omp parallel for schedule(dynamic,1)
            for(int s=0;s<S;s++){
                int pos=pos_base+s, nk=pos+1;
                if(nk<=dtopk && !g_dsa_force){ m->dsa_nsel[s]=0; continue; }
                int keep = nk<dtopk ? nk : dtopk;
                float *qi=falloc((int64_t)nh*hd);
                matmul_qt(qi, QR+(int64_t)s*c->q_lora, &m->ix_wq[layer], 1);
                for(int h=0;h<nh;h++) rope_interleave(qi+(int64_t)h*hd, pos, c);
                float w32[64];
                matmul_qt(w32, x+(int64_t)s*D, &m->ix_wp[layer], 1);
                float wsc=1.f/sqrtf((float)nh), rs=1.f/sqrtf((float)hd);
                float *isc=falloc(nk);
                for(int t=0;t<nk;t++){
                    const float *kt=m->Ic[layer]+(int64_t)t*hd;
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
                free(qi); free(isc); free(tmp);
            }
        }
        if(m->dsa_nsel){ dsel=m->dsa_sel; dnsel=m->dsa_nsel; }
    }
    /* WEIGHT ABSORPTION (DeepSeek): per S piccoli (decode/verifica MTP) NON si ricostruisce
     * k/v per ogni token del contesto. Per linearita':
     *   q·k_nope_t = (W_K^hT q_nope)·L_t      ctx^h = W_V^h (Σ_t a_t L_t)
     * costo per step ~O(T·kv_lora) invece di O(T·H·(nope+vh)) del matmul kvb_all. */
    int absorb = g_absorb==1 || (g_absorb<0 && S<=4);
    if(absorb && c->kv_lora<=512){
        int kvl=c->kv_lora, r0v=c->qk_nope;      /* offset righe V dentro il blocco di testa */
        #pragma omp parallel for collapse(2) schedule(static)
        for(int s=0;s<S;s++) for(int h=0;h<H;h++){
            int pos=pos_base+s;
            const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;
            const float *qr=qp+c->qk_nope;
            int rbase=h*(c->qk_nope+vh);
            float qabs[512]; memset(qabs,0,kvl*sizeof(float));
            for(int d=0;d<c->qk_nope;d++) qt_addrow(&l->kv_b, rbase+d, qp[d], qabs);
            float sc[8192];
            int st0=m->kv_start[layer];
            int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;    /* DSA: lista top-k o range pieno */
            const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
            int nt = ns ? ns : pos+1-st0;
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=m->Lc[layer]+(int64_t)t*kvl;
                const float *kr=m->Rc[layer]+(int64_t)t*c->qk_rope;
                float a=0; for(int i=0;i<kvl;i++) a+=qabs[i]*Lt[i];
                for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
                sc[jj]=a*c->attn_scale;
            }
            softmax(sc,nt);
            float clat[512]; memset(clat,0,kvl*sizeof(float));
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=m->Lc[layer]+(int64_t)t*kvl;
                float a=sc[jj]; for(int i=0;i<kvl;i++) clat[i]+=a*Lt[i]; }
            qt_matvec_rows(&l->kv_b, rbase+r0v, vh, clat, ctx+((int64_t)s*H+h)*vh);
        }
        matmul_qt(out, ctx, &l->o, S);
        free(ctx); free(Q); free(QR); free(comp);
        m->t_attn += now_s()-ta0;
        return;
    }
    /* 2) ricostruzione di k_nope+value per TUTTI i token 0..Tk-1 (un solo matmul su kv_b) */
    double tk0=now_s();
    int stL=m->kv_start[layer];
    float *kvb_all=falloc((int64_t)Tk*kvb_dim);
    matmul_qt(kvb_all+(int64_t)stL*kvb_dim, m->Lc[layer]+(int64_t)stL*c->kv_lora, &l->kv_b, Tk-stL);
    m->t_kvb += now_s()-tk0;
    /* 3) attenzione causale: score = q_pass·k_nope + q_rot·k_rot */
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s;
        const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;          /* [qk_nope | qk_rope] */
        const float *qr=qp+c->qk_nope;
        float sc[8192];
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
    matmul_qt(out, ctx, &l->o, S);
    free(ctx); free(Q); free(QR); free(comp); free(kvb_all);
    m->t_attn += now_s()-ta0;
}

/* MoE GLM su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, + shared expert).
 * BATCH-UNION: per S>1 (prefill, verifica MTP) ogni expert UNICO del batch viene caricato
 * una volta sola e moltiplicato per tutte le posizioni che lo usano (pesi letti 1 volta);
 * lo shared expert e' un unico matmul a S righe. Per posizione l'accumulo resta
 * nell'ordine (routed nel loro ordine di union, poi shared). */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out){
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    float *logit=falloc(E), *sig=falloc(E), *choice=falloc(E);
    int sI=c->moe_inter*c->n_shared;
    /* ---- FASE A: routing di tutte le S posizioni ---- */
    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));
    int *keff=malloc(S*sizeof(int));
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D;
        matmul(logit, xs, l->router, 1, D, E);
        for(int e=0;e<E;e++){ sig[e]=sigmoidf(logit[e]); choice[e]=sig[e]+l->router_bias[e]; }
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
            idx[kk]=best; w[kk]=sig[best];
        }
        int Ke=Ksel;
        if(g_topp>0 && g_topp<1.f){
            for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
                while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
            float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
            float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=g_topp*tot){ Ke=kk+1; break; } }
        }
        keff[s]=Ke; m->ereq+=Ke;
        for(int kk=0;kk<Ke;kk++) m->eusage[layer][idx[kk]]++;
        if(c->norm_topk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int kk=0;kk<Ke;kk++) w[kk]*=c->routed_scale;
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
    }
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
    /* ---- FASE B: union degli expert del batch ---- */
    int *uniq=malloc((size_t)E*sizeof(int)); int nu=0;
    { char *seen=calloc(E,1);
      for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){ int e=idxs[(int64_t)s*K+kk];
          if(!seen[e]){ seen[e]=1; uniq[nu++]=e; } }
      free(seen); }
    /* ---- FASE C/D: risolvi (pin/cache/disco) e calcola, a blocchi di 64 unici ---- */
    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);
    int *rows=malloc(S*sizeof(int)); float *rw=malloc(S*sizeof(float));
    for(int base=0;base<nu;base+=64){
        int nb = nu-base<64 ? nu-base : 64;
        ESlot *use[64]; int missk[64]; int nmiss=0;
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; use[j]=NULL;
            ESlot *P=m->pin[layer];
            for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ m->hits++; use[j]=&P[z]; break; }
            if(!use[j]){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ m->hits++; Sl[z].used=++m->eclock; use[j]=&Sl[z]; break; } }
            if(!use[j]){ use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
        }
        if(nmiss){ double t0=now_s();
            #pragma omp parallel for schedule(dynamic,1)
            for(int q=0;q<nmiss;q++) expert_load(m,layer,uniq[base+missk[q]],&m->ws[q]);
            m->t_edisk += now_s()-t0; }
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
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            int nr=0;                                 /* righe (posizioni) che usano questo expert */
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
            matmul_qt(gg, xg, &e->g, nr);
            matmul_qt(uu, xg, &e->u, nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            matmul_qt(hh, gg, &e->d, nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            m->t_emm += now_s()-t0;
        }
        { ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];   /* promozione LRU (swap buffer) */
          int promo = nmiss<m->ecap ? nmiss : m->ecap;
          for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
              if(*nn<m->ecap) dst=&Sl[(*nn)++];
              else { int lru=0; for(int z=1;z<*nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp; dst->used=++m->eclock; }
        }
    }
    /* ---- FASE E: shared expert, un matmul a S righe ---- */
    float *sg=falloc((int64_t)S*sI), *su=falloc((int64_t)S*sI);
    matmul_qt(sg, x, &l->sh_gate, S);
    matmul_qt(su, x, &l->sh_up,   S);
    for(int64_t z=0;z<(int64_t)S*sI;z++) sg[z]=siluf(sg[z])*su[z];
    matmul_qt(hh, sg, &l->sh_down, S);
    for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=hh[z];
    free(logit); free(sig); free(choice); free(idxs); free(ws); free(keff); free(uniq);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw); free(sg); free(su);
}

static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g, x, &l->gate_proj, S);
    matmul_qt(u, x, &l->up_proj,   S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=siluf(g[i])*u[i];
    matmul_qt(out, g, &l->down_proj, S);
    free(g); free(u);
}

/* forward di UN layer (usato dai 78 principali e dal layer MTP) */
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++) expert_prefetch(m,li,m->eroute[li][z]);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention(m,l,li,nrm,S,pos_base,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp); else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}
static void layers_forward(Model *m, float *x, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    for(int i=0;i<c->n_layers;i++){
        /* progresso su stderr per i batch grossi (prefill): il primo byte di risposta
         * puo' arrivare dopo MINUTI di streaming — al buio sembra un blocco. */
        if(S>=8 && (i%4==0 || i==c->n_layers-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token\n", i+1, c->n_layers, S);
        layer_forward(m,&m->L[i],i,x,S,pos_base,nrm,tmp);
    }
    free(nrm); free(tmp);
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    if(m->Lc){ for(int i=0;i<c->n_layers+1;i++){ free(m->Lc[i]); free(m->Rc[i]); } free(m->Lc); free(m->Rc); }
    if(m->Ic){ for(int i=0;i<c->n_layers;i++) free(m->Ic[i]); free(m->Ic); m->Ic=NULL; }
    if(m->has_dsa){
        m->Ic=calloc(c->n_layers,sizeof(float*));
        for(int i=0;i<c->n_layers;i++) if(c->idx_type[i]) m->Ic[i]=falloc((int64_t)max_t*c->index_hd);
    }
    m->max_t=max_t;
    int NR=c->n_layers+1;                        /* riga extra: KV del layer MTP */
    m->Lc=calloc(NR,sizeof(float*)); m->Rc=calloc(NR,sizeof(float*));
    for(int i=0;i<NR;i++){ m->Lc[i]=falloc((int64_t)max_t*c->kv_lora);
        m->Rc[i]=falloc((int64_t)max_t*c->qk_rope); }
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
    float *logit=falloc(c->vocab); matmul_qt(logit,last,&m->lm_head,1);
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
        matmul_qt(lo+(int64_t)s*c->vocab, row, &m->lm_head, 1); }
    free(x); free(row); return lo;
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
        if(dbg){ rmsnorm(row, hx, m->mtp_norm, D, c->eps); matmul_qt(logit, row, &m->lm_head, 1);
                 t_pre=mtp_argmax(logit, c->vocab); }
        layer_forward(m, &m->mtpL, li, hx, 1, pos, nrm, tmp);
        double n_post=0; for(int d=0;d<D;d++) n_post+=hx[d]*hx[d];
        rmsnorm(row, hx, m->mtp_norm, D, c->eps);
        matmul_qt(logit, row, &m->lm_head, 1);
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
    fprintf(stderr,"[stop] %d token di stop:",g_nstop);
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
        if(emitted>=n_new) break;                       /* l'ultimo token non serve forwardarlo */
        int g = 0;
        if(g_draft>0){
            /* auto-off adattivo: draft che non vengono mai accettati = solo tassa disco */
            if(m->has_mtp && m->mtp_prop>=24 && m->mtp_acc*10 < m->mtp_prop){
                g_draft=0;
                fprintf(stderr,"[MTP] acceptance %.0f%% dopo %llu proposte: draft disattivati\n",
                    100.0*m->mtp_acc/m->mtp_prop, (unsigned long long)m->mtp_prop);
            }
        }
        if(g_draft>0){
            if(m->has_mtp){ g=mtp_draft(m,next,kv,g_draft,draft); m->mtp_prop+=g; }
            else g=ngram_draft(all,kv+1,g_draft,draft);
        }
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        float *lo=step_all(m,batch,S,kv); m->n_fw++;
        int k=0;                                        /* verifica: accetta finche' coincide */
        if(g>0 && getenv("MTP_DEBUG")){ int veri=argmax_v(lo,V);
            fprintf(stderr,"[mtpdbg] draft0=%d verita=%d %s\n", draft[0], veri, draft[0]==veri?"HIT":"miss"); }
        while(k<g && emitted<n_new){
            int accept;
            if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V);          /* rejection sampling: p(draft) */
                   accept = (rndu() < g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0 && draft[k]==eos) || is_stop(draft[k])){ done=1; break; }
            emit(draft[k],ud); all[kv+1+k]=draft[k]; emitted++; m->n_emit++; k++;
        }
        if(m->has_mtp) m->mtp_acc+=k;
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
    for(int s=0;s<S;s++){
        float row[8192]; rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo, row, &m->lm_head, 1);
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
    }
    free(x); free(lo);
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
            matmul_qt(lo,row,&m->lm_head,1);
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

/* generazione reale: tokenizza PROMPT, prefill + decode greedy con stop su EOS,
 * detokenizza e stampa il testo in streaming. */
static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    Cfg *c=&m->c; char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int cap=(int)strlen(prompt)+16; int *pids=malloc(cap*sizeof(int));
    int np=tok_encode(&T,prompt,(int)strlen(prompt),pids,cap);
    if(np<1){ fprintf(stderr,"prompt vuoto dopo tokenizzazione\n"); return; }
    printf("prompt: %d token | genero fino a %d (stop EOS=%d) | draft n-gram=%d\n", np, ngen, eos, g_draft);
    fputs(prompt,stdout); fflush(stdout);
    kv_alloc(m, np+ngen+g_draft+2);
    int *all=malloc((np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,np*sizeof(int));
    double t=now_s();
    float *logit=step(m,pids,np,0);
    EmitStream es={&T,m,t,0,0};
    int produced=spec_decode(m,all,np,ngen,eos,logit,emit_stream,&es,NULL);
    double dt=now_s()-t;
    double tot=m->hits+m->miss;
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    printf("\n---\n%d token in %.2fs (%.2f tok/s) | hit-rate expert %.1f%% | RSS %.2f GB\n",
        produced, dt, produced/dt, tot?100.0*m->hits/tot:0.0, rss_gb());
    printf("expert caricati/token: %.1f (per-layer %.2f su %d; baseline topk=%d) | TOPK=%d TOPP=%.2f\n",
        produced?(double)m->ereq/produced:0.0, (produced&&nsp)?(double)m->ereq/produced/nsp:0.0, nsp, c->topk, g_topk, g_topp);
    printf("speculazione: %.2f token/forward (%llu fw per %llu tok) | MTP acceptance %.0f%% (%llu/%llu)\n",
        m->n_fw?(double)m->n_emit/m->n_fw:1.0, (unsigned long long)m->n_fw, (unsigned long long)m->n_emit,
        m->mtp_prop?100.0*m->mtp_acc/m->mtp_prop:0.0, (unsigned long long)m->mtp_acc, (unsigned long long)m->mtp_prop);
    double acc=m->t_edisk+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILO: expert-disk %.1fs | expert-matmul %.1fs | attention %.1fs (di cui kvb %.1fs) | lm_head %.1fs | altro %.1fs\n",
        m->t_edisk, m->t_emm, m->t_attn, m->t_kvb, m->t_head, dt-acc);
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
 * 25% (+4) contro il ping-pong; max 4 scambi/passata (~20 MB di disco l'uno). Usa eusage
 * cumulativo (l'aging LFU del mio fork non e' incluso qui: vedi PR.md).
 * EN: upstream AUTOPINs at START (from .coli_usage). This adds a between-turns re-pin: at
 * the safe point after the reply, swap the worst pins for the hottest unpinned, so the
 * hot-store tracks the LIVE workload without a separate profile. 25% (+4) hysteresis vs
 * ping-pong; max 4 swaps/pass (~20 MB disk each). Uses cumulative eusage (my fork's LFU
 * aging is NOT included here: see PR.md). */
static int g_repin=0;
static uint64_t g_last_repin=0;
typedef struct { long gain; int l, slot, eid; } RepinCand;
static int repin_pick(Model *m, RepinCand *out, int maxc){
    Cfg *c=&m->c; int nb=0;
    for(int l=0;l<c->n_layers;l++){
        if(!m->npin || m->npin[l]<1 || !m->eusage[l]) continue;
        uint32_t *u=m->eusage[l]; ESlot *P=m->pin[l];
        int zp=0; for(int z=1;z<m->npin[l];z++) if(u[P[z].eid]<u[P[zp].eid]) zp=z;   /* pin piu' freddo / coldest pin */
        int eu=-1; uint32_t fu=0;
        for(int e=0;e<c->n_experts;e++){
            int pinned=0; for(int z=0;z<m->npin[l];z++) if(P[z].eid==e){pinned=1;break;}
            if(!pinned && u[e]>fu){ fu=u[e]; eu=e; }                                 /* non-pin piu' caldo / hottest unpinned */
        }
        if(eu<0) continue;
        uint32_t fp=u[P[zp].eid];
        if(fu <= fp + (fp>>2) + 4) continue;                                         /* isteresi 25% / hysteresis */
        long g=(long)fu-(long)fp;
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
        int old=m->pin[cd[b].l][cd[b].slot].eid;
        double t0=now_s();
        expert_load(m, cd[b].l, cd[b].eid, &m->pin[cd[b].l][cd[b].slot]);
        fprintf(stderr,"[REPIN] layer %d: esce/out %d (f=%u) <- entra/in %d (f=%u) in %.0f ms\n",
            cd[b].l, old, m->eusage[cd[b].l][old], cd[b].eid, m->eusage[cd[b].l][cd[b].eid],
            (now_s()-t0)*1e3);
    }
}
static void run_serve(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    if(g_temp<0) g_temp=0.7f;            /* auto: 0.7, NON l'1.0 ufficiale — la coda della
                                          * distribuzione int4 e' rumore di quantizzazione */
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):256;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int templ=getenv("CHAT_TEMPLATE")?atoi(getenv("CHAT_TEMPLATE")):1;
    kv_alloc(m,maxctx);
    int len=0, first=1;                          /* len = contesto gia' in KV (persiste tra turni) */
    int *hist=malloc(maxctx*sizeof(int));        /* storia token (= contenuto della KV): serve
                                                  * al lookup n-gram e resta allineata a len */
    char *line=NULL; size_t cap=0; ssize_t nr; char *buf=malloc(1<<16);
    printf("\x01\x01" "READY" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout);
    while((nr=getline(&line,&cap,stdin))>0){
        if(nr>0 && line[nr-1]=='\n') line[--nr]=0;
        if(!strcmp(line,"\x02RESET")){ len=0; first=1; if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
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
            fflush(stdout); repin_pass(m); continue; }   /* RFC: re-pin a caldo tra i turni / live re-pin between turns */
        if(nr<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        int bl=0;                                /* costruisce il testo del turno (con template) */
        /* template UFFICIALE GLM-5.2 (chat_template.jinja): niente \n dopo i ruoli, e dopo
         * <|assistant|> serve SEMPRE il blocco think — <think></think> lo DISATTIVA (nothink):
         * col template sbagliato il modello farfuglia e non emette mai lo stop. THINK=1 lo abilita. */
        const char *tk = getenv("THINK")&&atoi(getenv("THINK"))? "<think>" : "<think></think>";
        if(templ){ if(first) bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop>");
                   bl+=snprintf(buf+bl,(1<<16)-bl,"<|user|>%s<|assistant|>%s",line,tk); }
        else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",line);
        int k=tok_encode(&T,buf,bl,hist+len,maxctx-len);
        if(k<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        if(len+k+8+g_draft>=maxctx){ len=0; first=1;   /* contesto pieno: azzera e ricomincia */
            bl=0; if(templ){ bl+=snprintf(buf+bl,(1<<16)-bl,"[gMASK]<sop><|user|>%s<|assistant|>%s",line,tk); }
            else bl+=snprintf(buf+bl,(1<<16)-bl,"%s",line);
            k=tok_encode(&T,buf,bl,hist,maxctx); if(k>maxctx-8-g_draft) k=maxctx-8-g_draft; }
        first=0;
        int cur=ngen; if(len+k+cur+g_draft+2>=maxctx) cur=maxctx-len-k-g_draft-2;
        uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
        float *logit=step(m,hist+len,k,len); len+=k;
        EmitStream es={&T,m,now_s(),0,1};
        int prod=0;
        if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
        else free(logit);
        double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
        double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
        printf("\n\x01\x01" "END" "\x01\x01\n");
        printf("STAT %d %.2f %.1f %.2f\n", prod, prod/tdt, (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb());
        fflush(stdout);
        usage_save(m);                   /* la cache che impara: storia aggiornata a ogni turno */
    }
    free(line); free(hist); free(buf);
    usage_save(m);
}

static int *read_arr(jval*o,const char*k,int*n){ jval*a=json_get(o,k); int*r=malloc(a->len*sizeof(int));
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
    if(!quiet) fprintf(stderr,"[STATS] %lld selezioni su %lld expert distinti -> %s\n",(long long)tot,(long long)nz,path);
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
        fprintf(stderr,"[PIN] mlock: %.1f GB inchiodati/wired, %ld alloc fallite/failed "
            "(alza il limite / raise it: ulimit -l unlimited) in %.0fs\n", wired/1e9, failed, now_s()-t0);
    else
        fprintf(stderr,"[PIN] mlock: %.1f GB inchiodati in RAM fisica / wired in physical RAM "
            "(niente compressione/no compression) in %.0fs\n", wired/1e9, now_s()-t0);
}

static void pin_load(Model *m, const char *statspath, double gb){
    FILE *f=fopen(statspath,"r"); if(!f){ perror(statspath); return; }
    typedef struct { int l,e; uint32_t c; } Rec;
    Cfg *c=&m->c; int cap=(c->n_layers+1)*c->n_experts;
    Rec *r=malloc((size_t)cap*sizeof(Rec)); int n=0;
    int l,e; uint32_t cnt;
    while(n<cap && fscanf(f,"%d %d %u",&l,&e,&cnt)==3){
        int ok = l>=0 && e>=0 && e<c->n_experts &&
                 ((l<c->n_layers && m->L[l].sparse) || (l==c->n_layers && m->has_mtp));
        if(ok) r[n++]=(Rec){l,e,cnt};
    }
    fclose(f);
    for(int a=0;a<n;a++){ int best=a;                       /* selection sort parziale, poi taglio */
        for(int b=a+1;b<n;b++) if(r[b].c>r[best].c) best=b;
        Rec t=r[a]; r[a]=r[best]; r[best]=t;
        if(a>4095) break;                                    /* bastano i top ~4k */
    }
    int64_t eb=expert_bytes_probe(m,m->ebits);
    int npin=(int)(gb*1e9/eb); if(npin>n) npin=n; if(npin>4096) npin=4096;
    if(npin<1){ free(r); return; }
    int *cnt_l=calloc(c->n_layers+1,sizeof(int));   /* +1: riga MTP */
    for(int a=0;a<npin;a++) cnt_l[r[a].l]++;
    for(int i=0;i<=c->n_layers;i++) if(cnt_l[i]) m->pin[i]=calloc(cnt_l[i],sizeof(ESlot));
    double t0=now_s();
    #pragma omp parallel for schedule(dynamic,1)
    for(int a=0;a<npin;a++){
        int li=r[a].l, slot;
        #pragma omp critical
        slot=m->npin[li]++;
        expert_load(m,li,r[a].e,&m->pin[li][slot]);
    }
    m->resident_bytes += (int64_t)npin*eb;
    fprintf(stderr,"[PIN] hot-store: %d expert in RAM (%.1f GB) in %.0fs da %s\n",
        npin, npin*eb/1e9, now_s()-t0, statspath);
    pin_wire(m);                                   /* inchioda in RAM (no compressione) / wire in RAM (no compression) */
    free(r); free(cnt_l);
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
#else
    FILE *f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char ln[256]; double kb=0;
    while(fgets(ln,sizeof(ln),f)) if(sscanf(ln,"MemAvailable: %lf",&kb)==1) break;
    fclose(f); return kb/1e6;
#endif
}

/* byte disponibili per gli expert (pin + LRU) nel budget — specchio del conto di cap_for_ram */
static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int64_t eb=expert_bytes_probe(m,ebits);
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double slack = 1.2e9 + 2.5e9 + 64.0*(double)eb
        + (double)(c->n_layers+1)*max_ctx*(c->kv_lora+c->qk_rope)*4.0
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
        if(ram_gb<4){ fprintf(stderr,"[RAM] MemAvailable illeggibile/troppo bassa, assumo 8 GB\n"); ram_gb=8; } }
    /* slack ONESTO, non forfettario (l'OOM del 2026-07-04 veniva da qui):
     *  ws[64] slab del working-set (si materializzano TUTTI nel prefill batch-union),
     *  KV cache a max_ctx, kvb_all della ricostruzione k/v in attention,
     *  attivazioni+logits+overhead ~1.2 GB */
    double ws_b  = 64.0*(double)eb;
    double kv_b  = (double)(c->n_layers+1)*max_ctx*(c->kv_lora+c->qk_rope)*4.0;
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
        fprintf(stderr,"[RAM_GB=%.1f%s] residente %.1f GB + slack %.1f GB (ws %.1f, KV@%d %.1f, kvb %.1f), "
            "expert %.1f MB x %d layer -> cap abbassato %d->%d (proiezione picco %.1f GB)\n",
            ram_gb, auto_b?" auto":"", m->resident_bytes/1e9, slack/1e9, ws_b/1e9, max_ctx, kv_b/1e9, kvb_b/1e9,
            eb/1e6, nsp, m->ecap, capmax,
            (m->resident_bytes + (double)capmax*nsp*eb + slack)/1e9);
        m->ecap=capmax;
    } else {
        fprintf(stderr,"[RAM_GB=%.1f%s] cap=%d ok (proiezione picco %.1f GB)\n", ram_gb, auto_b?" auto":"", m->ecap,
            (m->resident_bytes + (double)m->ecap*nsp*eb + slack)/1e9);
    }
}

int main(int argc, char **argv){
    /* i thread OMP non devono girare a vuoto mentre il main aspetta il disco */
    if(!getenv("OMP_WAIT_POLICY")) setenv("OMP_WAIT_POLICY","passive",1);
    const char *snap=getenv("SNAP"); if(!snap){fprintf(stderr,"SNAP=<dir>\n");return 1;}
    g_nopack = getenv("NOPACK")?1:0;
    g_drop = getenv("DROP")?1:0;
    g_prefetch = getenv("PREFETCH")?atoi(getenv("PREFETCH")):0;
    g_topk = getenv("TOPK")?atoi(getenv("TOPK")):0;
    g_topp = getenv("TOPP")?atof(getenv("TOPP")):0;
    g_mlock  = getenv("MLOCK")?atoi(getenv("MLOCK")):-1;   /* -1 auto (ON macOS), 0 off, 1 force / auto (ON macOS), 0 off, 1 force */
    g_spec = getenv("SPEC")?atoi(getenv("SPEC")):1;
    g_draft = getenv("DRAFT")?atoi(getenv("DRAFT")):-1;   /* -1 = auto: 3 se MTP, 0 senza */
    g_direct = getenv("DIRECT")?atoi(getenv("DIRECT")):0;
    g_idot = getenv("IDOT")?atoi(getenv("IDOT")):1;        /* 0 = kernel f32 esatti (A/B) */
    g_repin = getenv("REPIN")?atoi(getenv("REPIN")):0;     /* RFC: re-pin ogni n token emessi (0=off) / live re-pin every n emitted tokens (0=off) */
    g_absorb = getenv("ABSORB")?atoi(getenv("ABSORB")):-1; /* -1 auto: assorbita per S<=4 */
    g_dsa_force = getenv("DSA_FORCE")?atoi(getenv("DSA_FORCE")):0;
    g_temp = getenv("TEMP")?atof(getenv("TEMP")):-1;       /* -1 = auto (1.0 chat/testo, greedy altrove) */
    g_nuc  = getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.90f;  /* piu' stretto dell'ufficiale 0.95: la coda int4 e' rumore */
    if(getenv("SEED")) g_rng = (uint64_t)atoll(getenv("SEED"))*0x9E3779B97F4A7C15ULL+1;
    else { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); g_rng ^= (uint64_t)ts.tv_nsec<<20 ^ (uint64_t)getpid(); }
    if(g_draft>63) g_draft=63;                             /* -1 = auto, risolto dopo model_init */
    int cap  = argc>1?atoi(argv[1]):64;
    int ebits= argc>2?atoi(argv[2]):8;
    int dbits= argc>3?atoi(argv[3]):ebits;
    printf("== Motore C GLM (glm_moe_dsa), cache=%d expert/layer | expert@%d-bit densa@%d-bit | idot: " IDOT_KERNEL " ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    if(g_draft<0) g_draft = m.has_mtp ? 3 : 0;
    if(getenv("DSA_TOPK")) m.c.index_topk=atoi(getenv("DSA_TOPK"));   /* override per test */
    printf("caricato in %.2fs | densa residente: %.2f MB | layers=%d experts=%d | MTP %s (draft=%d)\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), m.c.n_layers, m.c.n_experts,
           m.has_mtp?"ATTIVA":"assente", g_draft);
    /* anche su stderr: e' il canale che le UI (coli) mostrano all'utente */
    fprintf(stderr,"[MTP] %s (draft=%d)\n", m.has_mtp?"attiva: decodifica speculativa nativa":"assente", g_draft);
    if(!strncmp(snap,"/mnt/",5))
        fprintf(stderr,"ATTENZIONE: il modello e' su %s (filesystem 9p/Windows, lento e fadvise inefficace).\n"
                       "            Per RAM e velocita' tienilo su ext4 (es. /home/...).\n", snap);
    /* HOT-STORE: PIN=<statsfile> [PIN_GB=g] -> top expert per frequenza fissi in RAM.
     * Va PRIMA di cap_for_ram: i pinnati contano nel residente. */
    if(getenv("PIN")) pin_load(&m, getenv("PIN"), getenv("PIN_GB")?atof(getenv("PIN_GB")):10.0);
    /* CACHE CHE IMPARA: l'uso degli expert si accumula in <SNAP>/.coli_usage tra le sessioni;
     * all'avvio i piu' usati vengono auto-pinnati in RAM (meta' del budget expert: il pin
     * conosce la TUA storia, la LRU si adatta alla sessione). AUTOPIN=0 disattiva. */
    { double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default di run_serve */
      snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
      int64_t hist = usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] storia expert: %lld selezioni (%s)\n",(long long)hist,g_usage_path);
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
    if(getenv("SERVE")){ run_serve(&m, snap); if(stats) stats_dump(&m,stats); return 0; }

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
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    char *ar=NULL; jval *ref=json_parse(b,&ar);
    int np,nfull; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    int n_new=nfull-np;

    if(getenv("TF")){
        int *tf=read_arr(ref,"tf_pred",&(int){0});
        int *pred=malloc(nfull*sizeof(int)); forward_all(&m, full, nfull, pred);
        int ok=0; for(int i=0;i<nfull;i++) ok+=(pred[i]==tf[i]);
        printf("PREFILL (teacher-forcing) C vs oracolo: %d/%d posizioni\n", ok, nfull);
        return 0;
    }
    int *out=malloc((np+n_new)*sizeof(int));
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nRiferimento (oracolo): "); for(int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nMotore C GLM         : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
    printf("\nToken coincidenti: %d/%d\n", match, n_new);
    double tot=m.hits+m.miss;
    printf("Speculazione n-gram (DRAFT=%d): %.2f token/forward (%llu fw per %llu tok)\n",
        g_draft, m.n_fw?(double)m.n_emit/m.n_fw:1.0, (unsigned long long)m.n_fw, (unsigned long long)m.n_emit);
    printf("Hit-rate cache expert: %.1f%% (hit=%llu miss=%llu) | RSS: %.2f GB | %.1f tok/s\n",
           tot?100.0*m.hits/tot:0.0, (unsigned long long)m.hits, (unsigned long long)m.miss, rss_gb(), n_new/dt);
    if(stats) stats_dump(&m,stats);
    return 0;
}
