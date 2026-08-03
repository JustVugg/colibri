/* Hy3 (hy_v3) streaming MoE inference engine for Colibri.
 * GQA attention with per-head Q/K RMSNorm + rotate_half RoPE.
 * Expert weights streamed from disk (int8/int4 per-row quant, dequant-on-use).
 * MoE router: sigmoid + expert bias + top-k (HYV3TopKRouter math).
 *
 * Validation: TF=1 SNAP=./hy3_tiny ./hy3 64 16 16
 *   (use 16-bit dense on fp32 oracle; int4 after convert_hy3.py)
 * Build: make hy3   (same CFLAGS/LDFLAGS as glm.c)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <unistd.h>
#include <errno.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#include <sys/mman.h>
#endif
#if defined(__linux__) && defined(COLI_IOURING)
#include <liburing.h>
#endif
#include "st.h"
#include "json.h"
#include "compat.h"
#include "tok.h"
#include "tier.h"
#ifdef COLI_CUDA
#include <omp.h>
#include "backend_cuda.h"
#endif
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
static inline float dot_avx2(const float *a, const float *b, int n){
    __m256 acc=_mm256_setzero_ps(); int i=0;
    for(;i+8<=n;i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    float s=hsum256(acc);
    for(;i<n;i++) s+=a[i]*b[i];
    return s;
}
static inline void axpy_avx2(float *y, const float *x, float w, int n){
    __m256 wv=_mm256_set1_ps(w); int i=0;
    for(;i+8<=n;i+=8){
        __m256 yv=_mm256_loadu_ps(y+i);
        yv=_mm256_fmadd_ps(_mm256_loadu_ps(x+i), wv, yv);
        _mm256_storeu_ps(y+i, yv);
    }
    for(;i<n;i++) y[i]+=w*x[i];
}
static inline float dot_f_i8(const float *a, const int8_t *b, int n){
    float s=0; int i=0;
#ifdef __AVX2__
    __m256 acc=_mm256_setzero_ps();
    for(;i+8<=n;i+=8){
        __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(b+i)));
        acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_cvtepi32_ps(wi), acc);
    }
    s=hsum256(acc);
#endif
    for(;i<n;i++) s+=a[i]*(float)b[i];
    return s;
}
static inline void axpy_f_i8(float *y, const int8_t *x, float w, int n){
    int i=0;
#ifdef __AVX2__
    __m256 wv=_mm256_set1_ps(w);
    for(;i+8<=n;i+=8){
        __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(x+i)));
        __m256 yv=_mm256_loadu_ps(y+i);
        yv=_mm256_fmadd_ps(_mm256_cvtepi32_ps(wi), wv, yv);
        _mm256_storeu_ps(y+i, yv);
    }
#endif
    for(;i<n;i++) y[i]+=w*(float)x[i];
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifndef __AVX2__
static inline float dot_avx2(const float *a, const float *b, int n){
    float s=0; for(int i=0;i<n;i++) s+=a[i]*b[i]; return s;
}
static inline void axpy_avx2(float *y, const float *x, float w, int n){
    for(int i=0;i<n;i++) y[i]+=w*x[i];
}
static inline float dot_f_i8(const float *a, const int8_t *b, int n){
    float s=0; for(int i=0;i<n;i++) s+=a[i]*(float)b[i]; return s;
}
static inline void axpy_f_i8(float *y, const int8_t *x, float w, int n){
    for(int i=0;i<n;i++) y[i]+=w*(float)x[i];
}
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#endif

typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, moe_inter, dense_inter, first_dense, n_shared, vocab;
    int route_norm;
    float eps, theta, router_scale;
    int stop_ids[8], n_stop;
} Cfg;

typedef struct {
    int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I;
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
    int cuda_eligible, cuda_failed, cuda_device;
} QT;

static int64_t qt_bytes(const QT *t){
    int64_t n=(int64_t)t->O*t->I;
    if(t->fmt==0) return n*4;
    if(t->fmt==1) return n+(int64_t)t->O*4;
    if(t->fmt==3) return (int64_t)t->O*((t->I+3)/4)+(int64_t)t->O*4;
    return (int64_t)t->O*((t->I+1)/2)+(int64_t)t->O*4;
}

typedef struct {
    float *in_ln, *post_ln;
    QT q, k, v, o;
    float *qn, *kn;
    int sparse;
    QT gate_proj, up_proj, down_proj;
    float *router, *router_bias;
    QT sh_gate, sh_up, sh_down;
} Layer;

typedef struct {
    int eid; QT g,u,d; uint8_t *slab; float *fslab;
    int64_t slab_cap, fslab_cap; uint64_t used;
} ESlot;

typedef struct {
    Cfg c; shards S;
    int ebits, dbits;
    QT embed, lm_head; float *final_norm;
    Layer *L;
    float **K, **V; int max_t;
    int8_t **Kq, **Vq;
    float **Kscale, **Vscale;
    int *kv_start;                               /* prima pos valida nella KV (MTP: parziale) */
    ESlot **ecache; int *ecn; int ecap;
    ESlot **pin; int *npin;
    ESlot ws[64];
    uint64_t eclock, hits, miss, ereq;
    uint32_t **eusage;
    uint32_t **eheat;
    /* testa MTP (layer n_layers, stile DeepSeek-V3): draft nativi ad alta acceptance */
    int has_mtp; Layer mtpL; QT eh_proj;
    float *enorm, *hnorm, *mtp_norm;
    float *hlast, *h_all;                        /* hidden pre-norm: ultima pos / tutte le pos batch */
    uint64_t mtp_prop, mtp_acc;                  /* statistica acceptance */
    uint64_t gpu_expert_calls; int gpu_expert_count; int64_t gpu_expert_bytes;
    uint64_t n_fw, n_emit;
    int64_t resident_bytes;
    double t_edisk, t_emm, t_attn, t_head;
    uint8_t *attn_vis;                           /* TREE_DRAFT: mask [S][max_t] flattened */
    int attn_vis_s;
} Model;

static void perf_report(Model *m);

#ifdef COLI_CUDA
static int g_cuda_enabled;
static double g_cuda_expert_gb;
static int g_cuda_dense;
static int g_cuda_attn;
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
    fprintf(stderr,COLI_ACCEL_TAG " resident set: %zu tensors, %.2f GB VRAM\n",n,b/1e9);
    if(g_cuda_ndev>1) for(int i=0;i<g_cuda_ndev;i++){
        coli_cuda_stats(g_cuda_devices[i],&n,&b);
        fprintf(stderr,COLI_ACCEL_TAG "   device %d: %zu tensors, %.2f GB\n",g_cuda_devices[i],n,b/1e9);
    }
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

static float g_temp=-1;
static float g_nuc=0.90f;
static int g_topk=0;
static float g_topp=0;
static int g_draft=-1;   /* -1 = auto: 3 se MTP, 0 senza */

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss/(1024.0*1024.0);
#endif
}
static float *falloc(int64_t n){
    if(n<0||(uint64_t)n>SIZE_MAX/sizeof(float)){ fprintf(stderr,"falloc: n=%lld out of range\n",(long long)n); exit(1); }
    float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p;
}

static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0;
            for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
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

static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib=_mm_unpacklo_epi8(lo,hi);
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint8x8_t by=vld1_u8(w+(i>>1));
                uint8x8x2_t z=vzip_u8(vand_u8(by,m4), vshr_n_u8(by,4));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a+=xs[i]*(float)lo+xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; a+=xs[i]*(float)((int)(byte&0xF)-8); }
            y[(int64_t)s*O+o]=a*sc; } }
}

static void matmul_i2(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O){
    int rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q2+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m2=_mm_set1_epi8(0x03); const __m256i b2=_mm256_set1_epi32(2);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_cvtsi32_si128(*(const int*)(w+(i>>2)));
                __m128i p0=_mm_and_si128(by,m2), p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                __m128i lo=_mm_unpacklo_epi8(p0,p1), hi=_mm_unpacklo_epi8(p2,p3);
                __m128i nib=_mm_unpacklo_epi16(lo,hi);
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b2));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b2));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m2v=vdup_n_u8(3); const int8x8_t b2v=vdup_n_s8(2);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint32_t wd; memcpy(&wd,w+(i>>2),4);
                uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v), vand_u8(vshr_n_u8(by,2),m2v));
                uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v), vshr_n_u8(by,6));
                uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[0]),b2v));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[1]),b2v));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++){ uint8_t byte=w[i>>2]; int sh=(i&3)*2;
                a+=xs[i]*(float)((int)((byte>>sh)&3)-2); }
            y[(int64_t)s*O+o]=a*sc; } }
}

static int g_idot=1, g_i4s=2, g_nopack=0, g_drop=0, g_direct=0;
static int g_perf=0, g_kv_i8=0, g_tree_draft=0;
static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f)s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv); return s;
}
#ifdef __AVX2__
static inline int hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
#endif
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0;
#ifdef __AVX2__
    __m256i acc=_mm256_setzero_si256(); const __m256i ones=_mm256_set1_epi16(1);
    int i=0;
    for(;i+32<=I;i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,xv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
#elif defined(__ARM_NEON)
    int32x4_t acc=vdupq_n_s32(0); int i=0;
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
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
#else
    for(int i=0;i<I;i++) sum+=(int32_t)w[i]*x[i];
#endif
    return sum;
}
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0;
    for(int i=0;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(I&1){ uint8_t b=w4[I>>1]; sum+=((int)(b&0xF)-8)*x[I-1]; }
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
    if(xn>g_qscratch.xq_cap){ int8_t *p=realloc(g_qscratch.xq,xn);
        if(!p){fprintf(stderr,"OOM quant scratch\n");exit(1);} g_qscratch.xq=p; g_qscratch.xq_cap=xn; }
    if(sn>g_qscratch.sx_cap){ float *p=realloc(g_qscratch.sx,sn*sizeof(float));
        if(!p){fprintf(stderr,"OOM quant scales\n");exit(1);} g_qscratch.sx=p; g_qscratch.sx_cap=sn; }
    *xq=g_qscratch.xq; *sx=g_qscratch.sx;
}

static void matmul_qt(float *y, const float *x, QT *w, int S){
#ifdef COLI_CUDA
    if(g_cuda_enabled && w->cuda_eligible && !w->cuda_failed && !omp_in_parallel()){
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_cuda_matmul(&w->cuda,y,x,weights,w->s,w->fmt,S,w->I,w->O,w->cuda_device)) return;
        w->cuda_failed=1;
        fprintf(stderr,COLI_ACCEL_TAG " tensor [%d,%d] on device %d disabled after an error; falling back to CPU\n",
            w->O,w->I,w->cuda_device);
    }
#endif
    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    if(g_idot&&(w->fmt==1||(w->fmt==2&&S>=g_i4s))){
        int I=w->I; int8_t *xq; float *sx;
        quant_scratch((size_t)S*I,(size_t)S,&xq,&sx);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I,xq+(int64_t)s*I,I);
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        return;
    }
    if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==3) matmul_i2(y,x,w->q4,w->s,S,w->I,w->O);
    else matmul_i4(y,x,w->q4,w->s,S,w->I,w->O);
}

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
            qr[i>>1]=(uint8_t)((v0+8)|((v1+8)<<4));
        }
    }
}
static void pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q2+(int64_t)o*rb;
        for(int i=0;i<I;i+=4){ uint8_t byte=0;
            for(int k=0;k<4&&i+k<I;k++){ int v=(int)lrintf(wr[i+k]/s); if(v>qmax)v=qmax; if(v<-2)v=-2;
                byte|=(uint8_t)((v+2)<<(k*2)); }
            qr[i>>2]=byte;
        }
    }
}

static void qt_alloc(QT *t, int O, int I, int bits){
    t->O=O; t->I=I; t->qf=NULL; t->q8=NULL; t->q4=NULL; t->s=NULL;
    if(bits>=16){ t->fmt=0; t->qf=falloc((int64_t)O*I); }
    else if(bits>=5||g_nopack){ t->fmt=1; t->q8=malloc((int64_t)O*I); t->s=falloc(O); }
    else if(bits>=3){ t->fmt=2; t->q4=malloc((int64_t)O*((I+1)/2)); t->s=falloc(O); }
    else { t->fmt=3; t->q4=malloc((int64_t)O*((I+3)/4)); t->s=falloc(O); }
}
static void qt_fill(QT *t, const float *w, int bits){
    if(t->fmt==0) memcpy(t->qf,w,(int64_t)t->O*t->I*sizeof(float));
    else if(t->fmt==1) quantize_rows(w,t->q8,t->s,t->O,t->I,bits);
    else if(t->fmt==3) pack_int2(w,t->q4,t->s,t->O,t->I,bits);
    else pack_int4(w,t->q4,t->s,t->O,t->I,bits);
}

static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
static void rmsnorm_head(float *out, const float *x, const float *w, int hd, float eps){
    double ms=0; for(int i=0;i<hd;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/hd)+eps); for(int i=0;i<hd;i++) out[i]=x[i]*r*w[i];
}
static void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }
static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf(float x){ return x/(1.f+expf(-x)); }

static int argmax_v(const float *lo, int V){
    int best=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){ bv=lo[i]; best=i; } return best;
}
static uint64_t g_rng=0x9E3779B97F4A7C15ULL;
static inline double rndu(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17;
    return (double)(g_rng>>11)*(1.0/9007199254740992.0); }
static float *g_pbuf=NULL; static int *g_pidx=NULL;
static int cmp_pdesc(const void *a,const void *b){
    float pa=g_pbuf[*(const int*)a], pb=g_pbuf[*(const int*)b];
    return pa<pb ? 1 : pa>pb ? -1 : 0; }
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
static int dist_sample(int V, int ban){
    double z=1.0-(ban>=0?g_pbuf[ban]:0.0); if(z<=1e-12) z=1e-12;
    double u=rndu()*z, cum=0;
    for(int i=0;i<V;i++){ if(i==ban) continue; cum+=g_pbuf[i]; if(cum>=u) return i; }
    for(int i=V-1;i>=0;i--) if(i!=ban && g_pbuf[i]>0) return i;
    return 0;
}
static int pick_tok(const float *lo, int V, int ban){
    if(g_temp<=0) return argmax_v(lo,V);
    dist_build(lo,V);
    return dist_sample(V,ban);
}

/* rotate_half RoPE on one head vector (split-half, HuggingFace Hy3 style) */
static void rope_rotate_half(float *x, int pos, float theta, int hd){
    int h2=hd/2; float tmp[512];
    if((size_t)hd>sizeof(tmp)/sizeof(float)){ fprintf(stderr,"head_dim too large\n"); exit(1); }
    memcpy(tmp,x,(size_t)hd*sizeof(float));
    for(int j=0;j<h2;j++){
        float inv=powf(theta,-2.0f*j/(float)hd), ang=pos*inv, c=cosf(ang), s=sinf(ang);
        float x1=tmp[j], x2=tmp[j+h2];
        x[j]=x1*c-x2*s; x[j+h2]=x2*c+x1*s;
    }
}

static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
static float gf(jval*r,const char*k, float def){ jval*v=json_get(r,k); return v?(float)v->num:def; }

static void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar);
    c->hidden=gi(r,"hidden_size"); c->n_layers=gi(r,"num_hidden_layers");
    c->n_heads=gi(r,"num_attention_heads"); c->n_kv_heads=gi(r,"num_key_value_heads");
    c->n_experts=gi(r,"num_experts"); c->topk=gi(r,"num_experts_per_tok");
    c->moe_inter=gi(r,"moe_intermediate_size"); c->dense_inter=gi(r,"intermediate_size");
    c->first_dense=gi(r,"first_k_dense_replace");
    c->head_dim=gi(r,"head_dim"); if(!c->head_dim) c->head_dim=c->hidden/c->n_heads;
    c->n_shared=gi(r,"num_shared_experts"); if(!c->n_shared) c->n_shared=1;
    c->vocab=gi(r,"vocab_size");
    jval *rn=json_get(r,"route_norm"); c->route_norm=(rn&&rn->t==J_BOOL)?rn->boolean:1;
    c->router_scale=gf(r,"router_scaling_factor",1.f);
    jval *ep=json_get(r,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    jval *rp=json_get(r,"rope_parameters"); jval *th=rp?json_get(rp,"rope_theta"):NULL;
    c->theta=th?(float)th->num:10000.f;
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len&&c->n_stop<8;i++) c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }
    if(c->n_heads%c->n_kv_heads){ fprintf(stderr,"n_heads must divide n_kv_heads evenly\n"); exit(1); }
    #define CK(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ fprintf(stderr,"config: %s=%d out of range\n",name,(int)(v)); exit(1); }
    CK("hidden_size",c->hidden,1,1<<20) CK("num_hidden_layers",c->n_layers,1,128)
    CK("num_attention_heads",c->n_heads,1,1024) CK("num_experts",c->n_experts,1,4096)
    CK("num_experts_per_tok",c->topk,1,64) CK("head_dim",c->head_dim,1,1<<16)
    #undef CK
    free(ar);
}

static void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t){
    char sn[300]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->S,sn)){
        int64_t nb=st_nbytes(&m->S,name);
        int fmt=(nb==(int64_t)O*I)?1:(nb==(int64_t)O*((I+1)/2))?2:3;
        if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->q8=malloc(nb); t->s=falloc(O); }
            st_read_raw(&m->S,name,t->q8,drop); }
        else { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->q4=malloc(nb); t->s=falloc(O); }
            st_read_raw(&m->S,name,t->q4,drop); }
        st_read_f32(&m->S,sn,t->s,drop);
    } else {
        if(!t->qf&&!t->q8&&!t->q4) qt_alloc(t,O,I,bits);
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
static float *ld(Model *m, const char *name){
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"missing %s\n",name);exit(1);}
    float *p=falloc(n); st_read_f32(&m->S,name,p,0); return p;
}
static float *ld_first(Model *m, const char *a, const char *b){
    if(st_has(&m->S,a)) return ld(m,a);
    if(b&&st_has(&m->S,b)) return ld(m,b);
    fprintf(stderr,"missing %s (also tried %s)\n",a,b?b:""); exit(1);
    return NULL;
}
static QT qt_load_first(Model *m, const char *a, const char *b, int O, int I, int bits){
    if(st_has(&m->S,a)) return qt_load(m,a,O,I,bits);
    if(b&&st_has(&m->S,b)) return qt_load(m,b,O,I,bits);
    fprintf(stderr,"missing %s (also tried %s)\n",a,b?b:""); exit(1);
    QT z; memset(&z,0,sizeof(z)); return z;
}
static int st_has_any(Model *m, const char *a, const char *b){
    if(st_has(&m->S,a)) return 1;
    if(b&&st_has(&m->S,b)) return 1;
    return 0;
}

static void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    if(e->fmt==0){ memcpy(x,e->qf+(int64_t)tok*D,D*sizeof(float)); return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok];
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok];
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; } return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok];
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}

static void expert_finalize(Model *m, int layer, int eid, ESlot *s,
        st_tensor *tw[3], st_tensor *tq[3], const int ord[3], const int64_t pos[3]){
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden;
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    if(g_drop){
        posix_fadvise(tw[ord[0]]->fd,tw[ord[0]]->off,wtot,POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(tq[k]->fd,tq[k]->off,tq[k]->nbytes,POSIX_FADV_DONTNEED);
    }
    float *fp[3]; int64_t fo=0;
    for(int k=0;k<3;k++){ fp[k]=s->fslab+fo; fo+=tq[k]->nbytes/4; }
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
    for(int k=0;k<3;k++){
        int64_t nb=tw[k]->nbytes;
        int fmt=(nb==(int64_t)OO[k]*II[k])?1:(nb==(int64_t)OO[k]*((II[k]+1)/2))?2:3;
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=fp[k];
    }
    s->eid=eid; (void)layer; (void)m;
}

static int expert_load(Model *m, int layer, int eid, ESlot *s){
#ifdef COLI_CUDA
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    char nm[3][288]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[300]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(!st_has(&m->S,qn)){
        qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
        qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
        qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
        s->eid=eid; return 1;
    }
    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm[k]); tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ fprintf(stderr,"missing %s\n",nm[k]); exit(1); }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    if(!s->slab||wtot+8192>s->slab_cap){
        compat_aligned_free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n");exit(1);}
        s->slab_cap=wtot+8192;
    }
    if(!s->fslab||ftot>s->fslab_cap){ free(s->fslab); s->fslab=falloc(ftot); s->fslab_cap=ftot; }
    int ord[3]={0,1,2};
    for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++) if(tw[ord[bb]]->off<tw[ord[a]]->off){ int t=ord[a]; ord[a]=ord[bb]; ord[bb]=t; }
    int contig=tw[ord[0]]->fd==tw[ord[1]]->fd&&tw[ord[1]]->fd==tw[ord[2]]->fd
        &&tw[ord[0]]->off+tw[ord[0]]->nbytes==tw[ord[1]]->off
        &&tw[ord[1]]->off+tw[ord[1]]->nbytes==tw[ord[2]]->off;
    int64_t pos[3]; int done=0;
    if(contig){
        int64_t off0=tw[ord[0]]->off;
        int dfd=g_direct?st_direct_fd(&m->S,tw[ord[0]]->fd):-1;
        if(dfd>=0){
            int64_t base=off0&~4095LL, need=(off0-base)+wtot, len=(need+4095)&~4095LL;
            if(pread(dfd,s->slab,len,base)>=need){
                pos[ord[0]]=off0-base; pos[ord[1]]=pos[ord[0]]+tw[ord[0]]->nbytes;
                pos[ord[2]]=pos[ord[1]]+tw[ord[1]]->nbytes; done=1;
            }
        }
        if(!done){
            if(pread(tw[ord[0]]->fd,s->slab,wtot,off0)!=wtot){perror("pread expert");exit(1);}
            pos[ord[0]]=0; pos[ord[1]]=tw[ord[0]]->nbytes; pos[ord[2]]=tw[ord[0]]->nbytes+tw[ord[1]]->nbytes; done=1;
        }
    }
    if(!done){ int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            if(pread(tw[k]->fd,s->slab+o,tw[k]->nbytes,tw[k]->off)!=tw[k]->nbytes){perror("pread expert");exit(1);}
            pos[k]=o; o+=tw[k]->nbytes; }
    }
    int64_t fo=0;
    for(int k=0;k<3;k++){
        if(pread(tq[k]->fd,(char*)(s->fslab+fo),tq[k]->nbytes,tq[k]->off)!=tq[k]->nbytes){perror("pread qs");exit(1);}
        fo+=tq[k]->nbytes/4;
    }
    if(g_drop){
        posix_fadvise(tw[ord[0]]->fd,tw[ord[0]]->off,wtot,POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(tq[k]->fd,tq[k]->off,tq[k]->nbytes,POSIX_FADV_DONTNEED);
    }
    expert_finalize(m,layer,eid,s,tw,tq,ord,pos);
    return 1;
}

static int g_pipe=0, g_pipe_nw=8;

#if defined(__linux__) && defined(COLI_IOURING)
typedef struct {
    ESlot *slot; Model *m; int layer, eid;
    _Atomic int pending;
    st_tensor *tw[3], *tq[3];
    int ord[3]; int64_t pos[3];
} ExpertIoJob;

static struct { struct io_uring ring; int ok; ExpertIoJob jobs[64]; int njobs; } g_uring;

static void uring_on_cqe(ExpertIoJob *job, int res){
    if(res<0){ fprintf(stderr,"io_uring read failed: %s\n",strerror(-res)); exit(1); }
    if(atomic_fetch_sub_explicit(&job->pending,1,memory_order_acq_rel)==1)
        expert_finalize(job->m,job->layer,job->eid,job->slot,job->tw,job->tq,job->ord,job->pos);
}

static void uring_pipe_init(void){
    if(g_uring.ok) return;
    int r=io_uring_queue_init(4096,&g_uring.ring,0);
    if(r<0){ fprintf(stderr,"[PIPE] io_uring_queue_init: %s, falling back to thread pool\n",strerror(-r)); g_pipe=1; return; }
    g_uring.ok=1;
    fprintf(stderr,"[PIPE] io_uring async expert load enabled (PIPE=2)\n");
}

static void uring_sqe_read(ExpertIoJob *job, int fd, void *buf, size_t len, off_t off){
    for(int tries=0;;tries++){
        struct io_uring_sqe *sqe=io_uring_get_sqe(&g_uring.ring);
        if(sqe){
            io_uring_prep_read(sqe,fd,buf,len,off);
            io_uring_sqe_set_data(sqe,job);
            atomic_fetch_add_explicit(&job->pending,1,memory_order_relaxed);
            return;
        }
        io_uring_submit(&g_uring.ring);
        struct io_uring_cqe *cqe;
        if(io_uring_wait_cqe(&g_uring.ring,&cqe)==0){
            ExpertIoJob *cj=(ExpertIoJob*)io_uring_cqe_get_data(cqe);
            uring_on_cqe(cj,cqe->res);
            io_uring_cqe_seen(&g_uring.ring,cqe);
        }
        if(tries>16384){
            fprintf(stderr,"io_uring_get_sqe failed: ring saturated (layer=%d eid=%d)\n",
                job->layer,job->eid);
            exit(1);
        }
    }
}

static void uring_pipe_dispatch(Model *m,int layer,const int *eids,int njobs){
    g_uring.njobs=njobs;
    for(int q=0;q<njobs;q++){
        ExpertIoJob *job=&g_uring.jobs[q];
        ESlot *s=&m->ws[q]; int eid=eids[q];
#ifdef COLI_CUDA
        if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
        memset(job,0,sizeof(*job));
        job->slot=s; job->m=m; job->layer=layer; job->eid=eid;
        atomic_store_explicit(&job->pending,0,memory_order_relaxed);
        Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
        char nm[3][288], qs[3][320]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
        for(int k=0;k<3;k++){
            snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
            snprintf(qs[k],sizeof(qs[k]),"%.*s.qs",(int)(sizeof(nm[k])-1),nm[k]);
        }
        if(!st_has(&m->S,qs[0])){
            qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
            qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
            qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
            s->eid=eid; continue;
        }
        for(int k=0;k<3;k++){
            job->tw[k]=st_find(&m->S,nm[k]);
            job->tq[k]=st_find(&m->S,qs[k]);
            if(!job->tw[k]||!job->tq[k]){ fprintf(stderr,"missing %s\n",nm[k]); exit(1); }
        }
        int64_t wtot=job->tw[0]->nbytes+job->tw[1]->nbytes+job->tw[2]->nbytes;
        int64_t ftot=(job->tq[0]->nbytes+job->tq[1]->nbytes+job->tq[2]->nbytes)/4;
        if(!s->slab||wtot+8192>s->slab_cap){
            compat_aligned_free(s->slab);
            if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n");exit(1);}
            s->slab_cap=wtot+8192;
        }
        if(!s->fslab||ftot>s->fslab_cap){ free(s->fslab); s->fslab=falloc(ftot); s->fslab_cap=ftot; }
        for(int a=0;a<3;a++) job->ord[a]=a;
        for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++)
            if(job->tw[job->ord[bb]]->off<job->tw[job->ord[a]]->off){
                int t=job->ord[a]; job->ord[a]=job->ord[bb]; job->ord[bb]=t; }
        int contig=job->tw[job->ord[0]]->fd==job->tw[job->ord[1]]->fd
            &&job->tw[job->ord[1]]->fd==job->tw[job->ord[2]]->fd
            &&job->tw[job->ord[0]]->off+job->tw[job->ord[0]]->nbytes==job->tw[job->ord[1]]->off
            &&job->tw[job->ord[1]]->off+job->tw[job->ord[1]]->nbytes==job->tw[job->ord[2]]->off;
        if(contig){
            int64_t off0=job->tw[job->ord[0]]->off;
            uring_sqe_read(job,job->tw[job->ord[0]]->fd,s->slab,wtot,off0);
            job->pos[job->ord[0]]=0;
            job->pos[job->ord[1]]=job->tw[job->ord[0]]->nbytes;
            job->pos[job->ord[2]]=job->tw[job->ord[0]]->nbytes+job->tw[job->ord[1]]->nbytes;
        } else {
            int64_t o=0;
            for(int a=0;a<3;a++){ int k=job->ord[a];
                uring_sqe_read(job,job->tw[k]->fd,s->slab+o,job->tw[k]->nbytes,job->tw[k]->off);
                job->pos[k]=o; o+=job->tw[k]->nbytes; }
        }
        int64_t fo=0;
        for(int k=0;k<3;k++){
            uring_sqe_read(job,job->tq[k]->fd,(char*)(s->fslab+fo),job->tq[k]->nbytes,job->tq[k]->off);
            fo+=job->tq[k]->nbytes/4;
        }
        if(atomic_load_explicit(&job->pending,memory_order_relaxed)==0)
            expert_finalize(m,layer,eid,s,job->tw,job->tq,job->ord,job->pos);
    }
    io_uring_submit(&g_uring.ring);
}

static void uring_pipe_wait(int q){
    ExpertIoJob *job=&g_uring.jobs[q];
    while(atomic_load_explicit(&job->pending,memory_order_acquire)>0){
        struct io_uring_cqe *cqe;
        if(io_uring_wait_cqe(&g_uring.ring,&cqe)<0){ perror("io_uring_wait_cqe"); exit(1); }
        ExpertIoJob *cj=(ExpertIoJob*)io_uring_cqe_get_data(cqe);
        uring_on_cqe(cj,cqe->res);
        io_uring_cqe_seen(&g_uring.ring,cqe);
    }
}
#endif

typedef struct {
    _Atomic uint64_t cur;
    _Atomic int njobs, eids[64], layer, ready[64];
    pthread_mutex_t mx; pthread_cond_t cv;
    Model *m; pthread_t th[16]; int nw, started;
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
            seen=c>>8; uint32_t i=(uint32_t)(c&0xFF);
            if(i>=(uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed)) break;
            if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,memory_order_acq_rel,memory_order_relaxed)){
                int L=atomic_load_explicit(&p->layer,memory_order_relaxed);
                int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed);
                expert_load(p->m,L,eid,&p->m->ws[i]);
                atomic_store_explicit(&p->ready[i],1,memory_order_release);
            }
        }
    }
    return NULL;
}
static void pipe_init(Model *m){
    if(g_pipe==2){
#if defined(__linux__) && defined(COLI_IOURING)
        uring_pipe_init();
        if(g_pipe==2) return;
#endif
        fprintf(stderr,"[PIPE] io_uring unavailable, using thread pool (PIPE=1)\n");
        g_pipe=1;
    }
    if(g_pp.started) return;
    g_pp.m=m; g_pp.nw=g_pipe_nw; if(g_pp.nw>16)g_pp.nw=16; if(g_pp.nw<1)g_pp.nw=1;
    atomic_store(&g_pp.cur,0); atomic_store(&g_pp.njobs,0);
    pthread_mutex_init(&g_pp.mx,NULL); pthread_cond_init(&g_pp.cv,NULL);
    for(int i=0;i<g_pp.nw;i++) pthread_create(&g_pp.th[i],NULL,pipe_worker,NULL);
    g_pp.started=1;
}
static void pipe_dispatch(Model *m,int layer,const int *eids,int njobs){
#if defined(__linux__) && defined(COLI_IOURING)
    if(g_pipe==2){ uring_pipe_dispatch(m,layer,eids,njobs); return; }
#endif
    g_pp.m=m;
    atomic_store_explicit(&g_pp.njobs,njobs,memory_order_relaxed);
    atomic_store_explicit(&g_pp.layer,layer,memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.eids[q],eids[q],memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.ready[q],0,memory_order_relaxed);
    uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
    atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
}
static inline void pipe_wait(int q){
#if defined(__linux__) && defined(COLI_IOURING)
    if(g_pipe==2){ uring_pipe_wait(q); return; }
#endif
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
}

static void expert_prefetch(Model *m, int layer, int eid){
    char nm[300]; const char *suf[3]={"gate_proj.weight","up_proj.weight","down_proj.weight"};
    for(int k=0;k<3;k++){
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s",layer,eid,suf[k]); st_prefetch(&m->S,nm);
        char qs[320]; snprintf(qs,sizeof(qs),"%s.qs",nm); st_prefetch(&m->S,qs);
    }
}

static void model_init(Model *m, const char *snap, int cap, int ebits, int dbits){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[512], nm2[512]; int D=c->hidden, hd=c->head_dim;
    int qo=c->n_heads*hd, kvo=c->n_kv_heads*hd;
    int io_bits=dbits>=8?16:dbits;
    m->embed=qt_load(m,"model.embed_tokens.weight",c->vocab,D,io_bits);
    if(st_has(&m->S,"lm_head.weight"))
        m->lm_head=qt_load(m,"lm_head.weight",c->vocab,D,io_bits);
    else
        m->lm_head=m->embed; /* tie_word_embeddings */
    m->final_norm=ld(m,"model.norm.weight");
    m->L=calloc(c->n_layers,sizeof(Layer));
    int nrows=c->n_layers+1;
    m->ecap=cap; m->ecache=calloc(nrows,sizeof(ESlot*)); m->ecn=calloc(nrows,sizeof(int));
    m->pin=calloc(nrows,sizeof(ESlot*)); m->npin=calloc(nrows,sizeof(int));
    m->eusage=calloc(nrows,sizeof(uint32_t*));
    m->eheat=calloc(nrows,sizeof(uint32_t*));
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        #define P(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
        l->in_ln=ld(m,P("input_layernorm.weight"));
        l->post_ln=ld(m,P("post_attention_layernorm.weight"));
        l->q=qt_load(m,P("self_attn.q_proj.weight"),qo,D,dbits);
        l->k=qt_load(m,P("self_attn.k_proj.weight"),kvo,D,dbits);
        l->v=qt_load(m,P("self_attn.v_proj.weight"),kvo,D,dbits);
        l->o=qt_load(m,P("self_attn.o_proj.weight"),D,qo,dbits);
        l->qn=ld(m,P("self_attn.q_norm.weight"));
        l->kn=ld(m,P("self_attn.k_norm.weight"));
        l->sparse=(i>=c->first_dense);
        if(!l->sparse){
            l->gate_proj=qt_load(m,P("mlp.gate_proj.weight"),c->dense_inter,D,dbits);
            l->up_proj=qt_load(m,P("mlp.up_proj.weight"),c->dense_inter,D,dbits);
            l->down_proj=qt_load(m,P("mlp.down_proj.weight"),D,c->dense_inter,dbits);
        } else {
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.router.gate.weight",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.weight",i);
            l->router=ld_first(m,nm2,nm);
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.expert_bias",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.e_score_correction_bias",i);
            l->router_bias=ld_first(m,nm2,nm);
            int sI=c->moe_inter*c->n_shared;
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.shared_experts.gate_proj.weight",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_mlp.gate_proj.weight",i);
            l->sh_gate=qt_load_first(m,nm2,nm,sI,D,dbits);
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.shared_experts.up_proj.weight",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_mlp.up_proj.weight",i);
            l->sh_up=qt_load_first(m,nm2,nm,sI,D,dbits);
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.shared_experts.down_proj.weight",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_mlp.down_proj.weight",i);
            l->sh_down=qt_load_first(m,nm2,nm,D,sI,dbits);
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
        }
        #undef P
    }
    /* testa MTP (layer n_layers): presente solo se convertita con --mtp */
    {
        char ex_last[64]; snprintf(ex_last,sizeof(ex_last),"mlp.experts.%d.down_proj.weight",c->n_experts-1);
        const char *req[]={"eh_proj.weight","enorm.weight","hnorm.weight",
            "input_layernorm.weight","post_attention_layernorm.weight",
            "self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight",
            "self_attn.o_proj.weight","self_attn.q_norm.weight","self_attn.k_norm.weight",
            "mlp.experts.0.gate_proj.weight",ex_last};
        char mn[256]; m->has_mtp=1;
        for(unsigned q=0;q<sizeof(req)/sizeof(req[0]);q++){
            snprintf(mn,sizeof(mn),"model.layers.%d.%s",c->n_layers,req[q]);
            if(!st_has(&m->S,mn)){ m->has_mtp=0; break; }
        }
        /* Hy3 source naming: shared_mlp / final_layernorm; Colibri rename: shared_experts / shared_head.norm */
        if(m->has_mtp){
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.router.gate.weight",c->n_layers);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.weight",c->n_layers);
            if(!st_has_any(m,nm2,nm)) m->has_mtp=0;
        }
        if(m->has_mtp){
            snprintf(nm2,sizeof(nm2),"model.layers.%d.shared_head.norm.weight",c->n_layers);
            snprintf(nm,sizeof(nm),"model.layers.%d.final_layernorm.weight",c->n_layers);
            if(!st_has_any(m,nm2,nm)) m->has_mtp=0;
        }
        if(m->has_mtp){
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.shared_experts.gate_proj.weight",c->n_layers);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_mlp.gate_proj.weight",c->n_layers);
            if(!st_has_any(m,nm2,nm)) m->has_mtp=0;
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.shared_experts.down_proj.weight",c->n_layers);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_mlp.down_proj.weight",c->n_layers);
            if(!st_has_any(m,nm2,nm)) m->has_mtp=0;
        }
        if(getenv("MTP") && atoi(getenv("MTP"))==0) m->has_mtp=0;
        if(m->has_mtp){
            int i=c->n_layers; Layer *l=&m->mtpL;
            #define PM(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
            l->in_ln=ld(m,PM("input_layernorm.weight"));
            l->post_ln=ld(m,PM("post_attention_layernorm.weight"));
            l->q=qt_load(m,PM("self_attn.q_proj.weight"),qo,D,dbits);
            l->k=qt_load(m,PM("self_attn.k_proj.weight"),kvo,D,dbits);
            l->v=qt_load(m,PM("self_attn.v_proj.weight"),kvo,D,dbits);
            l->o=qt_load(m,PM("self_attn.o_proj.weight"),D,qo,dbits);
            l->qn=ld(m,PM("self_attn.q_norm.weight"));
            l->kn=ld(m,PM("self_attn.k_norm.weight"));
            l->sparse=1;
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.router.gate.weight",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.weight",i);
            l->router=ld_first(m,nm2,nm);
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.expert_bias",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.e_score_correction_bias",i);
            l->router_bias=ld_first(m,nm2,nm);
            int sI=c->moe_inter*c->n_shared;
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.shared_experts.gate_proj.weight",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_mlp.gate_proj.weight",i);
            l->sh_gate=qt_load_first(m,nm2,nm,sI,D,dbits);
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.shared_experts.up_proj.weight",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_mlp.up_proj.weight",i);
            l->sh_up=qt_load_first(m,nm2,nm,sI,D,dbits);
            snprintf(nm2,sizeof(nm2),"model.layers.%d.mlp.shared_experts.down_proj.weight",i);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_mlp.down_proj.weight",i);
            l->sh_down=qt_load_first(m,nm2,nm,D,sI,dbits);
            m->eh_proj=qt_load(m,PM("eh_proj.weight"),D,2*D,dbits);
            m->enorm=ld(m,PM("enorm.weight")); m->hnorm=ld(m,PM("hnorm.weight"));
            char mtp_na[512], mtp_nb[512];
            snprintf(mtp_na,sizeof(mtp_na),"model.layers.%d.shared_head.norm.weight",i);
            snprintf(mtp_nb,sizeof(mtp_nb),"model.layers.%d.final_layernorm.weight",i);
            m->mtp_norm=ld_first(m,mtp_na,mtp_nb);
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            #undef PM
        }
    }
    m->hlast=falloc(D); m->h_all=falloc((int64_t)64*D);
    int64_t rb=qt_bytes(&m->embed);
    if(m->lm_head.qf!=m->embed.qf&&m->lm_head.q8!=m->embed.q8&&m->lm_head.q4!=m->embed.q4)
        rb+=qt_bytes(&m->lm_head);
    for(int i=0;i<c->n_layers;i++){ Layer *l=&m->L[i];
        rb+=qt_bytes(&l->q)+qt_bytes(&l->k)+qt_bytes(&l->v)+qt_bytes(&l->o);
        if(!l->sparse) rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
        else rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down);
    }
    if(m->has_mtp){ Layer *l=&m->mtpL;
        rb+=qt_bytes(&l->q)+qt_bytes(&l->k)+qt_bytes(&l->v)+qt_bytes(&l->o);
        rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down)+qt_bytes(&m->eh_proj);
    }
    m->resident_bytes=rb;
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    int NR=c->n_layers+(m->has_mtp?1:0);
    if(m->K){ for(int i=0;i<NR;i++){ free(m->K[i]); free(m->V[i]); }
        free(m->K); free(m->V); m->K=m->V=NULL; }
    if(m->Kq){ for(int i=0;i<NR;i++){ free(m->Kq[i]); free(m->Vq[i]); free(m->Kscale[i]); free(m->Vscale[i]); }
        free(m->Kq); free(m->Vq); free(m->Kscale); free(m->Vscale);
        m->Kq=NULL; m->Vq=NULL; m->Kscale=NULL; m->Vscale=NULL; }
    free(m->kv_start); m->kv_start=NULL;
    m->max_t=max_t;
    m->kv_start=calloc(NR,sizeof(int));
    int64_t slot=(int64_t)c->n_kv_heads*max_t*c->head_dim;
    int64_t sc_slot=(int64_t)c->n_kv_heads*max_t;
    if(g_kv_i8){
        m->Kq=calloc(NR,sizeof(int8_t*)); m->Vq=calloc(NR,sizeof(int8_t*));
        m->Kscale=calloc(NR,sizeof(float*)); m->Vscale=calloc(NR,sizeof(float*));
        for(int i=0;i<NR;i++){
            m->Kq[i]=malloc((size_t)slot); m->Vq[i]=malloc((size_t)slot);
            m->Kscale[i]=falloc(sc_slot); m->Vscale[i]=falloc(sc_slot);
            if(!m->Kq[i]||!m->Vq[i]){fprintf(stderr,"OOM kv i8\n");exit(1);}
            m->kv_start[i]=(m->has_mtp && i==c->n_layers)?-1:0;
        }
    } else {
        m->K=calloc(NR,sizeof(float*)); m->V=calloc(NR,sizeof(float*));
        for(int i=0;i<NR;i++){
            m->K[i]=falloc(slot); m->V[i]=falloc(slot);
            m->kv_start[i]=(m->has_mtp && i==c->n_layers)?-1:0;
        }
    }
}

static inline void quantize_row_i8(const float *src, int8_t *dst, float *scale, int n){
    float mx=1e-8f; for(int i=0;i<n;i++){ float a=fabsf(src[i]); if(a>mx) mx=a; }
    float s=mx/127.0f; *scale=s;
    float inv=1.0f/s;
    for(int i=0;i<n;i++){
        float v=src[i]*inv;
        if(v>127.f) v=127.f; else if(v<-127.f) v=-127.f;
        dst[i]=(int8_t)lrintf(v);
    }
}

/* GQA attention: per-head QK norm, rotate_half RoPE, repeat_kv for scores */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    Cfg *c=&m->c; int H=c->n_heads, Hkv=c->n_kv_heads, hd=c->head_dim;
    int nrep=H/Hkv; float scale=1.f/sqrtf((float)hd);
    double ta0=now_s();
    int64_t qsz=(int64_t)S*H*hd, kv_sz=(int64_t)S*Hkv*hd;
    float *Q=falloc(qsz), *Kp=falloc(kv_sz), *Vp=falloc(kv_sz);
    matmul_qt(Q,x,&l->q,S); matmul_qt(Kp,x,&l->k,S); matmul_qt(Vp,x,&l->v,S);
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        for(int h=0;h<H;h++){
            float *qh=Q+(int64_t)s*H*hd+(int64_t)h*hd;
            rmsnorm_head(qh,qh,l->qn,hd,c->eps);
            rope_rotate_half(qh,pos,c->theta,hd);
        }
        for(int kh=0;kh<Hkv;kh++){
            float *kk=Kp+(int64_t)s*Hkv*hd+(int64_t)kh*hd;
            float *vv=Vp+(int64_t)s*Hkv*hd+(int64_t)kh*hd;
            rmsnorm_head(kk,kk,l->kn,hd,c->eps);
            rope_rotate_half(kk,pos,c->theta,hd);
            if(g_kv_i8){
                int64_t off=((int64_t)kh*m->max_t+pos)*hd;
                int64_t soff=(int64_t)kh*m->max_t+pos;
                quantize_row_i8(kk,m->Kq[layer]+off,m->Kscale[layer]+soff,hd);
                quantize_row_i8(vv,m->Vq[layer]+off,m->Vscale[layer]+soff,hd);
            } else {
                memcpy(m->K[layer]+((int64_t)kh*m->max_t+pos)*hd,kk,(size_t)hd*sizeof(float));
                memcpy(m->V[layer]+((int64_t)kh*m->max_t+pos)*hd,vv,(size_t)hd*sizeof(float));
            }
        }
    }
    float *ctx=falloc(qsz);
    int use_cuda=0;
#ifdef COLI_CUDA
    if(g_cuda_enabled&&g_cuda_attn&&!g_kv_i8&&m->K&&!m->attn_vis){
        int st0=(m->kv_start&&m->kv_start[layer]>=0)?m->kv_start[layer]:0;
        if(coli_cuda_gqa_attention(ctx,Q,m->K[layer],m->V[layer],S,H,Hkv,hd,st0,pos_base,m->max_t,g_cuda_devices[0]))
            use_cuda=1;
    }
#endif
    if(!use_cuda){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s, kvh=h/nrep;
        const float *qv=Q+(int64_t)s*H*hd+(int64_t)h*hd;
        int st0=(m->kv_start && m->kv_start[layer]>=0)?m->kv_start[layer]:0;
        int nt=pos+1-st0;
        float sc[8192];
        int nv=0;
        for(int jj=0;jj<nt;jj++){
            int t=st0+jj;
            if(m->attn_vis && !m->attn_vis[(int64_t)s*m->max_t+t]) continue;
            if(g_kv_i8){
                if(t==pos){
                    const float *kv=Kp+(int64_t)s*Hkv*hd+(int64_t)kvh*hd;
                    sc[nv]=dot_avx2(qv,kv,hd)*scale;
                } else {
                    int64_t off=((int64_t)kvh*m->max_t+t)*hd;
                    int64_t soff=(int64_t)kvh*m->max_t+t;
                    sc[nv]=m->Kscale[layer][soff]*dot_f_i8(qv,m->Kq[layer]+off,hd)*scale;
                }
            } else {
                const float *kv=m->K[layer]+((int64_t)kvh*m->max_t+t)*hd;
                sc[nv]=dot_avx2(qv,kv,hd)*scale;
            }
            nv++;
        }
        if(nv<1){ float *cx=ctx+(int64_t)s*H*hd+(int64_t)h*hd; for(int d=0;d<hd;d++) cx[d]=0; continue; }
        softmax(sc,nv);
        float *cx=ctx+(int64_t)s*H*hd+(int64_t)h*hd;
        for(int d=0;d<hd;d++) cx[d]=0;
        int vi=0;
        for(int jj=0;jj<nt;jj++){
            int t=st0+jj;
            if(m->attn_vis && !m->attn_vis[(int64_t)s*m->max_t+t]) continue;
            if(g_kv_i8){
                if(t==pos){
                    const float *vv=Vp+(int64_t)s*Hkv*hd+(int64_t)kvh*hd;
                    axpy_avx2(cx,vv,sc[vi],hd);
                } else {
                    int64_t off=((int64_t)kvh*m->max_t+t)*hd;
                    int64_t soff=(int64_t)kvh*m->max_t+t;
                    axpy_f_i8(cx,m->Vq[layer]+off,sc[vi]*m->Vscale[layer][soff],hd);
                }
            } else {
                const float *vv=m->V[layer]+((int64_t)kvh*m->max_t+t)*hd;
                axpy_avx2(cx,vv,sc[vi],hd);
            }
            vi++;
        }
    }
    }
    matmul_qt(out,ctx,&l->o,S);
    m->t_attn+=now_s()-ta0;
    free(Q); free(Kp); free(Vp); free(ctx);
}

static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g,x,&l->gate_proj,S); matmul_qt(u,x,&l->up_proj,S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=siluf(g[i])*u[i];
    matmul_qt(out,g,&l->down_proj,S);
    free(g); free(u);
}

/* MoE: HYV3TopKRouter math (sigmoid, bias for selection, normalize, router_scaling_factor) */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out){
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    int sI=c->moe_inter*c->n_shared;
    float *logit=falloc(E), *choice=falloc(E);
    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));
    int *keff=malloc(S*sizeof(int));
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D;
        matmul(logit,xs,l->router,1,D,E);
        for(int e=0;e<E;e++){ logit[e]=sigmoidf(logit[e]); choice[e]=logit[e]+l->router_bias[e]; }
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel=g_topk>0?(g_topk<K?g_topk:K):K;
        for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                if(!tk&&choice[e]>bv){bv=choice[e];best=e;} }
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
            if(m->eusage[layer]) m->eusage[layer][idx[kk]]++;
            if(m->eheat[layer]&&m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++;
        }
        if(c->route_norm){ float sm=1e-20f; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int kk=0;kk<Ke;kk++) w[kk]*=c->router_scale;
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
    }
    int *uniq=malloc((size_t)E*sizeof(int)); int nu=0;
    unsigned char seen[4096]; memset(seen,0,(size_t)E);
    for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){
        int e=idxs[(int64_t)s*K+kk]; if(!seen[e]){ seen[e]=1; uniq[nu++]=e; }
    }
    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);
    int *rows=malloc(S*sizeof(int)); float *rw=malloc(S*sizeof(float));
    for(int base=0;base<nu;base+=64){
        int nb=nu-base<64?nu-base:64;
        ESlot *use[64]; int missk[64], qof[64], nmiss=0;
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; use[j]=NULL; qof[j]=-1;
            ESlot *P=m->pin[layer];
            for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ m->hits++; use[j]=&P[z]; break; }
            if(!use[j]){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ m->hits++; Sl[z].used=++m->eclock; use[j]=&Sl[z]; break; } }
            if(!use[j]){ qof[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
        }
        if(nmiss){
            if(g_pipe){ if(!g_pp.started) pipe_init(m);
                double t0=now_s(); int eids[64];
                for(int q=0;q<nmiss;q++) eids[q]=uniq[base+missk[q]];
                pipe_dispatch(m,layer,eids,nmiss); m->t_edisk+=now_s()-t0;
            } else { double t0=now_s();
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++) expert_load(m,layer,uniq[base+missk[q]],&m->ws[q]);
                m->t_edisk+=now_s()-t0; }
        }
        if(base+64<nu){
            int nb2=nu-(base+64)<64?nu-(base+64):64;
            for(int j=0;j<nb2;j++){ int eid=uniq[base+64+j], found=0;
                ESlot *Sl=m->ecache[layer];
                for(int z=0;z<m->ecn[layer]&&!found;z++) if(Sl[z].eid==eid) found=1;
                if(!found) expert_prefetch(m,layer,eid);
            }
        }
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            if(g_pipe&&qof[j]>=0){ double tw=now_s(); pipe_wait(qof[j]); m->t_edisk+=now_s()-tw; }
            int nr=0;
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
#ifdef COLI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D,x+(int64_t)rows[r]*D,D*sizeof(float));
            double t0=now_s();
            matmul_qt(gg,xg,&e->g,nr); matmul_qt(uu,xg,&e->u,nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            matmul_qt(hh,gg,&e->d,nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            m->t_emm+=now_s()-t0;
        }
        { ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];
          int promo=nmiss<m->ecap?nmiss:m->ecap;
          for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
              if(*nn<m->ecap) dst=&Sl[(*nn)++];
              else { int lru=0; for(int z=1;z<*nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp; dst->used=++m->eclock; }
        }
    }
    float *sg=falloc((int64_t)S*sI), *su=falloc((int64_t)S*sI);
    matmul_qt(sg,x,&l->sh_gate,S); matmul_qt(su,x,&l->sh_up,S);
    for(int64_t z=0;z<(int64_t)S*sI;z++) sg[z]=siluf(sg[z])*su[z];
    matmul_qt(hh,sg,&l->sh_down,S);
    for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=hh[z];
    free(logit); free(choice); free(idxs); free(ws); free(keff); free(uniq);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw); free(sg); free(su);
}

static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D,x+(int64_t)s*D,l->in_ln,D,c->eps);
    attention(m,l,li,nrm,S,pos_base,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D,x+(int64_t)s*D,l->post_ln,D,c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp);
    else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}

static void layers_forward(Model *m, float *x, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    for(int i=0;i<c->n_layers;i++){
        if(S>=8&&(i%4==0||i==c->n_layers-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token\n",i+1,c->n_layers,S);
        layer_forward(m,&m->L[i],i,x,S,pos_base,nrm,tmp);
    }
    free(nrm); free(tmp);
}

static void forward_all(Model *m, const int *ids, int S, int *pred){
    Cfg *c=&m->c; int D=c->hidden;
    kv_alloc(m,S);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m,ids[s],x+(int64_t)s*D);
    layers_forward(m,x,S,0);
    float *lo=falloc(c->vocab), row[8192];
    for(int s=0;s<S;s++){
        rmsnorm(row,x+(int64_t)s*D,m->final_norm,D,c->eps);
        matmul_qt(lo,row,&m->lm_head,1);
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
    }
    free(x); free(lo);
}

static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base);
static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    if((!m->K && !m->Kq)||pos_base+S>m->max_t) kv_alloc(m,pos_base+S+16);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m,ids[s],x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->hlast) memcpy(m->hlast,x+(int64_t)(S-1)*D,D*sizeof(float));
    if(m->has_mtp && S>=2 && g_draft>0) mtp_absorb(m,ids+1,x,S-1,pos_base);
    float *last=falloc(D); rmsnorm(last,x+(int64_t)(S-1)*D,m->final_norm,D,c->eps);
    double th0=now_s();
    float *logit=falloc(c->vocab); matmul_qt(logit,last,&m->lm_head,1);
    m->t_head+=now_s()-th0;
    free(x); free(last); return logit;
}

/* come step(), ma ritorna i logits di TUTTE le S posizioni [S,vocab] (per la verifica spec) */
static float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    if((!m->K && !m->Kq)||pos_base+S>m->max_t) kv_alloc(m,pos_base+S+16);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m,ids[s],x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->h_all) memcpy(m->h_all,x,(int64_t)S*D*sizeof(float));
    if(m->hlast) memcpy(m->hlast,x+(int64_t)(S-1)*D,D*sizeof(float));
    float *lo=falloc((int64_t)S*c->vocab), *row=falloc(D);
    for(int s=0;s<S;s++){ rmsnorm(row,x+(int64_t)s*D,m->final_norm,D,c->eps);
        matmul_qt(lo+(int64_t)s*c->vocab,row,&m->lm_head,1); }
    free(x); free(row); return lo;
}

static int ngram_draft(const int *ids, int len, int G, int *draft){
    if(len<4||G<1) return 0;
    int a=ids[len-2], b=ids[len-1];
    for(int i=len-3;i>=1;i--)
        if(ids[i-1]==a&&ids[i]==b){
            int n=0; for(int j=i+1;j<len&&n<G;j++) draft[n++]=ids[j];
            return n;
        }
    return 0;
}

static int mtp_argmax(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}

static int mtp_one_forward(Model *m, int tok, int pos, float *logit){
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    if(m->kv_start[li]<0||m->kv_start[li]>pos) m->kv_start[li]=pos;
    float *x=falloc(D), *cat=falloc(2*D), *hx=falloc(D), *nrm=falloc(D), *tmp=falloc(D);
    float *row=falloc(D), *h=falloc(D);
    memcpy(h,m->hlast,D*sizeof(float));
    int prenorm=getenv("MTP_PRENORM")!=NULL;
    embed_row(m,tok,x);
    rmsnorm(x,x,m->enorm,D,c->eps);
    if(!prenorm) rmsnorm(h,h,m->final_norm,D,c->eps);
    rmsnorm(h,h,m->hnorm,D,c->eps);
    if(getenv("MTP_SWAP")){ memcpy(cat,h,D*sizeof(float)); memcpy(cat+D,x,D*sizeof(float)); }
    else { memcpy(cat,x,D*sizeof(float)); memcpy(cat+D,h,D*sizeof(float)); }
    matmul_qt(hx,cat,&m->eh_proj,1);
    layer_forward(m,&m->mtpL,li,hx,1,pos,nrm,tmp);
    rmsnorm(row,hx,m->mtp_norm,D,c->eps);
    matmul_qt(logit,row,&m->lm_head,1);
    memcpy(m->hlast,hx,D*sizeof(float));
    free(x); free(cat); free(hx); free(nrm); free(tmp); free(row); free(h);
    return 1;
}

static void mtp_top2(const float *lo, int V, int *a, int *b){
    int t0=0,t1=0; float v0=lo[0],v1=-1e30f;
    for(int i=1;i<V;i++){
        if(lo[i]>v0){v1=v0;t1=t0;v0=lo[i];t0=i;}
        else if(lo[i]>v1){v1=lo[i];t1=i;}
    }
    *a=t0; *b=t1;
}

static int mtp_draft_tree(Model *m, int next_tok, int kv, int *nodes, int *parent){
    Cfg *c=&m->c; int V=c->vocab;
    float *logit=falloc(V);
    int p=kv; if(p<0){ free(logit); return 0; }
    mtp_one_forward(m,next_tok,p,logit);
    int r0,r1; mtp_top2(logit,V,&r0,&r1);
    nodes[0]=r0; nodes[1]=r1; parent[0]=parent[1]=-1;
    int n=2;
    for(int ri=0;ri<2;ri++){
        int root=nodes[ri], pos=p+1+ri;
        mtp_one_forward(m,root,pos,logit);
        int c0,c1; mtp_top2(logit,V,&c0,&c1);
        nodes[n]=c0; parent[n]=ri; n++;
        nodes[n]=c1; parent[n]=ri; n++;
    }
    free(logit);
    return n;
}

static void build_tree_attn_mask(Model *m, int kv, int S, const int *parent){
    if(!m->attn_vis) m->attn_vis=calloc((size_t)S*m->max_t,1);
    memset(m->attn_vis,0,(size_t)S*m->max_t);
    m->attn_vis_s=S;
    for(int s=0;s<S;s++){
        for(int t=0;t<=kv;t++) m->attn_vis[(int64_t)s*m->max_t+t]=1;
        int pos=kv+1+s;
        for(int a=s;;){
            int pt=kv+1+a;
            if(pt<=pos) m->attn_vis[(int64_t)s*m->max_t+pt]=1;
            if(a<0) break;
            a=parent[a];
        }
    }
}

static void clear_tree_attn_mask(Model *m){
    free(m->attn_vis); m->attn_vis=NULL; m->attn_vis_s=0;
}

static int mtp_draft(Model *m, int next_tok, int kv, int G, int *draft){
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    int p=kv-1; if(p<0||G<1) return 0;
    if(m->kv_start[li]<0||m->kv_start[li]>p) m->kv_start[li]=p;
    float *x=falloc(D), *cat=falloc(2*D), *hx=falloc(D), *nrm=falloc(D), *tmp=falloc(D);
    float *row=falloc(D), *logit=falloc(c->vocab), *h=falloc(D);
    memcpy(h,m->hlast,D*sizeof(float));
    int tok=next_tok, n=0;
    int prenorm=getenv("MTP_PRENORM")!=NULL;
    for(int g=0;g<G;g++){
        int pos=p+g; if(pos+2>=m->max_t) break;
        embed_row(m,tok,x);
        rmsnorm(x,x,m->enorm,D,c->eps);
        if(g==0&&!prenorm) rmsnorm(h,h,m->final_norm,D,c->eps);
        rmsnorm(h,h,m->hnorm,D,c->eps);
        if(getenv("MTP_SWAP")){ memcpy(cat,h,D*sizeof(float)); memcpy(cat+D,x,D*sizeof(float)); }
        else { memcpy(cat,x,D*sizeof(float)); memcpy(cat+D,h,D*sizeof(float)); }
        matmul_qt(hx,cat,&m->eh_proj,1);
        double n_eh=0; for(int d=0;d<D;d++) n_eh+=hx[d]*hx[d];
        int dbg=getenv("MTP_DEBUG")&&atoi(getenv("MTP_DEBUG"))>=2;
        int t_pre=-1;
        if(dbg){ rmsnorm(row,hx,m->mtp_norm,D,c->eps); matmul_qt(logit,row,&m->lm_head,1);
                 t_pre=mtp_argmax(logit,c->vocab); }
        layer_forward(m,&m->mtpL,li,hx,1,pos,nrm,tmp);
        double n_post=0; for(int d=0;d<D;d++) n_post+=hx[d]*hx[d];
        rmsnorm(row,hx,m->mtp_norm,D,c->eps);
        matmul_qt(logit,row,&m->lm_head,1);
        int t2=mtp_argmax(logit,c->vocab);
        if(dbg) fprintf(stderr,"[mtp2] pos=%d in_tok=%d ||eh||=%.1f ||post||=%.1f pre_blk=%d post_blk=%d\n",
                        pos,tok,sqrt(n_eh),sqrt(n_post),t_pre,t2);
        draft[n++]=t2; tok=t2; memcpy(h,hx,D*sizeof(float));
    }
    free(x); free(cat); free(hx); free(nrm); free(tmp); free(row); free(logit); free(h);
    return n;
}
static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base){
    if(!m->has_mtp||S<1) return;
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    if(m->kv_start[li]<0||m->kv_start[li]>pos_base) m->kv_start[li]=pos_base;
    float *hx=falloc((int64_t)S*D), *cat=falloc(2*D), *e=falloc(D), *hn=falloc(D), *hf=falloc(D);
    int prenorm=getenv("MTP_PRENORM")!=NULL;
    for(int i=0;i<S;i++){
        embed_row(m,next_ids[i],e);
        rmsnorm(e,e,m->enorm,D,c->eps);
        if(prenorm) rmsnorm(hn,x+(int64_t)i*D,m->hnorm,D,c->eps);
        else { rmsnorm(hf,x+(int64_t)i*D,m->final_norm,D,c->eps);
               rmsnorm(hn,hf,m->hnorm,D,c->eps); }
        if(getenv("MTP_SWAP")){ memcpy(cat,hn,D*sizeof(float)); memcpy(cat+D,e,D*sizeof(float)); }
        else { memcpy(cat,e,D*sizeof(float)); memcpy(cat+D,hn,D*sizeof(float)); }
        matmul_qt(hx+(int64_t)i*D,cat,&m->eh_proj,1);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    layer_forward(m,&m->mtpL,li,hx,S,pos_base,nrm,tmp);
    free(hx); free(cat); free(e); free(hn); free(hf); free(nrm); free(tmp);
}

static int is_stop(const Cfg *c, int tok){
    for(int i=0;i<c->n_stop;i++) if(c->stop_ids[i]==tok) return 1;
    return 0;
}

static int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                       void (*emit)(int,void*), void *ud, int *kv_out);

typedef struct { int *dst; int n; } EmitStore;
static void emit_store(int t, void *ud){ EmitStore *e=(EmitStore*)ud; e->dst[e->n++]=t; }

typedef struct { Tok *T; Model *m; double t0; int count; } EmitStream;
static void emit_stream(int t, void *ud){
    EmitStream *e=(EmitStream*)ud; char dec[512];
    int dn=tok_decode(e->T,&t,1,dec,511); dec[dn]=0; fputs(dec,stdout); fflush(stdout);
    if(++e->count%16==0){ double tt=e->m->hits+e->m->miss;
        fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  %.2f tok/s  %.2f tok/fw]\n",e->count,
            rss_gb(),tt?100.0*e->m->hits/tt:0.0,e->count/(now_s()-e->t0),
            e->m->n_fw?(double)e->count/e->m->n_fw:1.0); }
    if(g_perf && e->count%100==0) perf_report(e->m);
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out){
    kv_alloc(m,np+n_new+g_draft+2);
    for(int i=0;i<np;i++) out[i]=prompt[i];
    float *logit=step(m,prompt,np,0);
    EmitStore es={out+np,0};
    spec_decode(m,out,np,n_new,-1,logit,emit_store,&es,NULL);
}

static int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                       void (*emit)(int,void*), void *ud, int *kv_out){
    Cfg *c=&m->c; int V=c->vocab; int emitted=0, done=0;
    int draft[64]; int gd=g_draft; if(gd>63) gd=63;
    int carry_ban=-1;
    while(emitted<n_new&&!done){
        int next=pick_tok(logit,V,carry_ban); carry_ban=-1; free(logit); logit=NULL;
        if((eos>=0&&next==eos)||is_stop(&m->c,next)) break;
        emit(next,ud); all[kv]=next; emitted++; m->n_emit++;
        if(g_perf && m->n_emit%100==0) perf_report(m);
        if(emitted>=n_new) break;
        int g=0, gsrc=0;
        if(gd>0){
            if(m->has_mtp&&m->mtp_prop>=24&&m->mtp_acc*10<m->mtp_prop){
                gd=0; g_draft=0;
                fprintf(stderr,"[MTP] %.0f%% acceptance after %llu proposals: drafts disabled\n",
                    100.0*m->mtp_acc/m->mtp_prop,(unsigned long long)m->mtp_prop);
            }
        }
        if(!g&&gd>0){
            if(m->has_mtp&&g_tree_draft){
                int nodes[8], parent[8], tn=mtp_draft_tree(m,next,kv,nodes,parent);
                if(tn>=4){
                    build_tree_attn_mask(m,kv,tn,parent);
                    float *lo=step_all(m,nodes,tn,kv+1); m->n_fw++;
                    m->mtp_prop+=tn;
                    int paths[4][2]={{0,2},{0,3},{1,4},{1,5}};
                    int best_p=0,best_len=0;
                    for(int p=0;p<4;p++){
                        int r=paths[p][0],l=paths[p][1];
                        int len=0;
                        if(g_temp<=0){
                            if(argmax_v(lo+(int64_t)r*V,V)==nodes[l]) len=2;
                        } else {
                            dist_build(lo+(int64_t)r*V,V);
                            if(rndu()<g_pbuf[nodes[l]]) len=2;
                        }
                        if(len>best_len){ best_len=len; best_p=p; }
                    }
                    clear_tree_attn_mask(m);
                    int r=paths[best_p][0], l=paths[best_p][1];
                    int k=0;
                    if(best_len>=1){
                        if((eos>=0&&nodes[r]==eos)||is_stop(&m->c,nodes[r])) done=1;
                        else { emit(nodes[r],ud); all[kv+1]=nodes[r]; emitted++; m->n_emit++; k=1; }
                    }
                    if(!done&&best_len>=2&&emitted<n_new){
                        if((eos>=0&&nodes[l]==eos)||is_stop(&m->c,nodes[l])) done=1;
                        else { emit(nodes[l],ud); all[kv+2]=nodes[l]; emitted++; m->n_emit++; k=2; }
                    }
                    if(m->has_mtp) m->mtp_acc+=k;
                    if(m->has_mtp&&k>=1) mtp_absorb(m,all+kv+1,m->h_all,k,kv);
                    kv+=k;
                    logit=falloc(V);
                    if(k>0&&k<tn) memcpy(logit,lo+(int64_t)(k-1)*V,V*sizeof(float));
                    else if(k>=2) memcpy(logit,lo+(int64_t)l*V,V*sizeof(float));
                    else memcpy(logit,lo,V*sizeof(float));
                    free(lo);
                    continue;
                }
            }
            if(m->has_mtp){ g=mtp_draft(m,next,kv,gd,draft); m->mtp_prop+=g; if(g)gsrc=2; }
            else { g=ngram_draft(all,kv+1,gd,draft); if(g)gsrc=2; }
        }
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        float *lo=step_all(m,batch,S,kv); m->n_fw++;
        int k=0;
        if(g>0&&getenv("MTP_DEBUG")){ int veri=argmax_v(lo,V);
            fprintf(stderr,"[mtpdbg] draft0=%d verified=%d %s\n",draft[0],veri,draft[0]==veri?"HIT":"miss"); }
        while(k<g&&emitted<n_new){
            int accept;
            if(g_temp<=0) accept=(argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V); accept=(rndu()<g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0&&draft[k]==eos)||is_stop(&m->c,draft[k])){ done=1; break; }
            emit(draft[k],ud); all[kv+1+k]=draft[k]; emitted++; m->n_emit++;
            k++;
        }
        if(gsrc==2&&m->has_mtp) m->mtp_acc+=k;
        if(m->has_mtp&&k>=1) mtp_absorb(m,all+kv+1,m->h_all,k,kv);
        if(m->h_all&&k<S-1) memcpy(m->hlast,m->h_all+(int64_t)k*c->hidden,c->hidden*sizeof(float));
        kv+=1+k;
        logit=falloc(V); memcpy(logit,lo+(int64_t)k*V,V*sizeof(float)); free(lo);
    }
    if(logit) free(logit);
    if(kv_out) *kv_out=kv;
    return emitted;
}

static int *read_arr(jval*o,const char*k,int*n){
    jval*a=json_get(o,k); int*r=malloc(a->len*sizeof(int));
    for(int i=0;i<a->len;i++) r[i]=(int)a->kids[i]->num;
    *n=a->len; return r;
}

static int *parse_ids_env(int *n){
    const char *s=getenv("IDS"); if(!s||!*s) return NULL;
    int cap=64, len=0; int *ids=malloc(cap*sizeof(int));
    const char *p=s;
    while(*p){
        char *end=NULL; long v=strtol(p,&end,10);
        if(end==p) break;
        if(len>=cap){ cap*=2; ids=realloc(ids,cap*sizeof(int)); }
        ids[len++]=(int)v; p=end;
        while(*p==','||*p==' ') p++;
    }
    *n=len; return ids;
}

static double g_mem_avail_boot=0;
static double mem_available_gb(void){
#ifdef __APPLE__
    mach_msg_type_number_t cnt=HOST_VM_INFO64_COUNT; vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&cnt)!=KERN_SUCCESS) return 0;
    return ((double)vm.free_count+(double)vm.inactive_count+(double)vm.purgeable_count)
           *(double)sysconf(_SC_PAGESIZE)/1e9;
#elif defined(_WIN32)
    double total, avail; compat_meminfo(&total,&avail); return avail;
#else
    FILE *f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char ln[256]; double kb=0;
    while(fgets(ln,sizeof(ln),f)) if(sscanf(ln,"MemAvailable: %lf",&kb)==1) break;
    fclose(f); return kb/1e6;
#endif
}

static int64_t tbytes(int O,int I,int bits){
    if(bits>=16) return (int64_t)O*I*4;
    if(bits>=5) return (int64_t)O*I+(int64_t)O*4;
    return (int64_t)O*((I+1)/2)+(int64_t)O*4;
}
static int64_t expert_bytes_probe(Model *m, int ebits){
    Cfg *c=&m->c; int64_t eb=0; char nm[256];
    int layer=c->first_dense<c->n_layers?c->first_dense:c->n_layers-1;
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.gate_proj.weight",layer);
    if(st_nbytes(&m->S,nm)>0){
        const char *suf[3]={"gate_proj","up_proj","down_proj"};
        for(int k=0;k<3;k++){
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight",layer,suf[k]);
            eb+=st_nbytes(&m->S,nm);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight.qs",layer,suf[k]);
            int64_t q=st_nbytes(&m->S,nm); if(q>0) eb+=q;
        }
    }
    if(eb<=0) eb=tbytes(c->moe_inter,c->hidden,ebits)*2+tbytes(c->hidden,c->moe_inter,ebits);
    return eb;
}

static double kv_pool_bytes(Model *m, int max_ctx){
    Cfg *c=&m->c;
    int nl=c->n_layers+(m->has_mtp?1:0);
    if(g_kv_i8){
        int64_t per=(int64_t)c->n_kv_heads*max_ctx;
        return (double)nl*(2.0*(double)per*c->head_dim+2.0*(double)per*4.0);
    }
    return (double)nl*max_ctx*c->n_kv_heads*c->head_dim*4.0*2.0;
}

static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int64_t eb=expert_bytes_probe(m,ebits);
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double slack=1.2e9+2.5e9+64.0*(double)eb+kv_pool_bytes(m,max_ctx)
        +(double)max_ctx*c->n_heads*c->head_dim*4.0;
    return ram_gb*1e9-(double)m->resident_bytes-slack;
}

static int mem_wire(void *addr, size_t len){
#if defined(__APPLE__) || defined(__linux__)
    return mlock(addr,len);
#else
    (void)addr; (void)len; return 0;
#endif
}
static void pin_wire(Model *m){
    Cfg *c=&m->c; double t0=now_s(); int64_t wired=0; long failed=0;
    for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
        ESlot *s=&m->pin[i][z];
        if(s->slab){ if(mem_wire(s->slab,s->slab_cap)==0) wired+=s->slab_cap; else failed++; }
        if(s->fslab){ size_t fl=(size_t)s->fslab_cap*sizeof(float);
            if(mem_wire(s->fslab,fl)==0) wired+=fl; else failed++; }
    }
    if(wired>0) fprintf(stderr,"[PIN] mlock: %.1f GB wired%s in %.0fs\n",
        wired/1e9, failed?" (some failed; try ulimit -l unlimited)":"", now_s()-t0);
}
static void pin_load(Model *m, const char *statspath, double gb){
    FILE *f=fopen(statspath,"r"); if(!f){ perror(statspath); return; }
    typedef struct { int l,e; uint32_t c; } Rec;
    Cfg *c=&m->c; int cap=(c->n_layers+1)*c->n_experts;
    Rec *r=malloc((size_t)cap*sizeof(Rec)); int n=0;
    int l,e; uint32_t cnt;
    while(n<cap && fscanf(f,"%d %d %u",&l,&e,&cnt)==3){
        int ok=l>=0&&e>=0&&e<c->n_experts&&
            ((l<c->n_layers&&m->L[l].sparse)||(l==c->n_layers&&m->has_mtp));
        if(ok) r[n++]=(Rec){l,e,cnt};
    }
    fclose(f);
    for(int a=0;a<n;a++){ int best=a;
        for(int b=a+1;b<n;b++) if(r[b].c>r[best].c) best=b;
        Rec t=r[a]; r[a]=r[best]; r[best]=t;
        if(a>4095) break;
    }
    int64_t eb=expert_bytes_probe(m,m->ebits);
    int npin=(int)(gb*1e9/eb); if(npin>n) npin=n; if(npin>4096) npin=4096;
    if(npin<1){ free(r); return; }
    int *cnt_l=calloc(c->n_layers+1,sizeof(int));
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
    m->resident_bytes+=(int64_t)npin*eb;
    fprintf(stderr,"[PIN] hot store: %d experts in RAM (%.1f GB) loaded in %.0fs from %s\n",
        npin,npin*eb/1e9,now_s()-t0,statspath);
#ifdef COLI_CUDA
    if(g_cuda_enabled && g_cuda_expert_gb>0){
        double remaining[COLI_CUDA_MAX_DEVICES]={0}, placed_b[COLI_CUDA_MAX_DEVICES]={0};
        int placed_n[COLI_CUDA_MAX_DEVICES]={0};
        double budget=g_cuda_expert_gb*1e9, safe_total=0;
        for(int i=0;i<g_cuda_ndev;i++){
            size_t free_b=0,total_b=0;
            if(coli_cuda_mem_info(g_cuda_devices[i],&free_b,&total_b)){
                remaining[i]=(double)free_b-(double)g_cuda_dense_projected[i]-2e9;
                if(remaining[i]<0) remaining[i]=0;
                safe_total+=remaining[i];
            }
        }
        if(budget>safe_total) budget=safe_total;
        for(int a=0;a<npin && m->gpu_expert_bytes<budget;a++){
            int li=r[a].l;
            for(int z=0;z<m->npin[li];z++) if(m->pin[li][z].eid==r[a].e){
                ESlot *s=&m->pin[li][z];
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
                        placed=1;
                    } else {
                        qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                        s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                        remaining[best]=0;
                    }
                }
                break;
            }
        }
        fprintf(stderr,COLI_ACCEL_TAG " hot expert tier: %d/%d experts, VRAM %.2f GB (total budget %.1f GB)\n",
            m->gpu_expert_count,npin,m->gpu_expert_bytes/1e9,g_cuda_expert_gb);
        for(int i=0;i<g_cuda_ndev;i++) fprintf(stderr,COLI_ACCEL_TAG "   device %d: %d experts, %.2f GB\n",
            g_cuda_devices[i],placed_n[i],placed_b[i]/1e9);
    }
#endif
    pin_wire(m);
    free(r); free(cnt_l);
}

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
        if(!tier_pick_swap(m->eheat[l],c->n_experts,ids,np,&zp,&eu,&g)) continue;
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
        expert_load(m,cd[b].l,cd[b].eid,s);
        const char *tier="RAM";
#ifdef COLI_CUDA
        if(gpu){
            if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                int64_t now_gpu=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                m->gpu_expert_bytes+=now_gpu-old_gpu; tier="VRAM";
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

static void cap_for_ram(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    if(m->has_mtp) nsp+=2;
    int64_t eb=expert_bytes_probe(m,ebits);
    int auto_b=ram_gb<=0;
    if(auto_b){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double ws_b=64.0*(double)eb, kv_b=kv_pool_bytes(m,max_ctx);
    double attn_b=(double)max_ctx*c->n_heads*c->head_dim*4.0;
    double pc_b=2.5e9, slack=1.2e9+pc_b+ws_b+kv_b+attn_b;
    double avail=ram_gb*1e9-(double)m->resident_bytes-slack;
    int capmax=(avail>0&&nsp>0)?(int)(avail/((double)nsp*eb)):0;
    if(capmax<1) capmax=1;
    if(capmax<m->ecap){
        fprintf(stderr,"[RAM_GB=%.1f%s] resident %.1f GB + reserve %.1f GB (ws %.1f, KV %.1f, attn %.1f), "
            "experts %.1f MB x %d sparse layers -> cap lowered %d->%d (peak ~%.1f GB)\n",
            ram_gb,auto_b?" auto":"",m->resident_bytes/1e9,slack/1e9,ws_b/1e9,kv_b/1e9,attn_b/1e9,
            eb/1e6,nsp,m->ecap,capmax,(m->resident_bytes+(double)capmax*nsp*eb+slack)/1e9);
        m->ecap=capmax;
    } else {
        int raise_on=getenv("CAP_RAISE")?atoi(getenv("CAP_RAISE")):1;
        int newcap=capmax>c->n_experts?c->n_experts:capmax;
        if(raise_on&&newcap>m->ecap){
            for(int i=0;i<=c->n_layers;i++) if(m->ecache[i]){
                m->ecache[i]=realloc(m->ecache[i],(size_t)newcap*sizeof(ESlot));
                memset(m->ecache[i]+m->ecap,0,(size_t)(newcap-m->ecap)*sizeof(ESlot));
            }
            fprintf(stderr,"[RAM_GB=%.1f%s] cap raised %d->%d (peak ~%.1f GB)\n",
                ram_gb,auto_b?" auto":"",m->ecap,newcap,
                (m->resident_bytes+(double)newcap*nsp*eb+slack)/1e9);
            m->ecap=newcap;
        } else
            fprintf(stderr,"[RAM_GB=%.1f%s] cap=%d ok (peak ~%.1f GB)\n",ram_gb,auto_b?" auto":"",m->ecap,
                (m->resident_bytes+(double)m->ecap*nsp*eb+slack)/1e9);
    }
}

static void profile_print(Model *m, double elapsed){
    double acc=m->t_edisk+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILE: expert-disk %.3fs | expert-matmul %.3fs | attention %.3fs | lm_head %.3fs | other %.3fs\n",
        m->t_edisk,m->t_emm,m->t_attn,m->t_head,elapsed-acc);
}

static void perf_report(Model *m){
    double total=m->t_attn+m->t_edisk+m->t_emm+m->t_head;
    if(total<1e-9) return;
    fprintf(stderr,"[perf] attn=%.1f%% disk=%.1f%% expert_mm=%.1f%% head=%.1f%%\n",
        100.0*m->t_attn/total, 100.0*m->t_edisk/total,
        100.0*m->t_emm/total, 100.0*m->t_head/total);
}

static char g_usage_path[2100]="";
static void stats_dump_q(Model *m, const char *path, int quiet){
    char tmp[2100]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *f=fopen(tmp,"w"); if(!f){ if(!quiet) perror(tmp); return; }
    Cfg *c=&m->c; int64_t tot=0, nz=0;
    for(int i=0;i<c->n_layers;i++){ if(!m->eusage[i]) continue;
        for(int e=0;e<c->n_experts;e++) if(m->eusage[i][e]){
            fprintf(f,"%d %d %u\n",i,e,m->eusage[i][e]); tot+=m->eusage[i][e]; nz++; } }
    fclose(f); rename(tmp,path);
    if(!quiet) fprintf(stderr,"[STATS] %lld selections / %lld experts -> %s\n",(long long)tot,(long long)nz,path);
}
static void stats_dump(Model *m, const char *path){ stats_dump_q(m,path,0); }
static int64_t usage_load(Model *m, const char *path){
    FILE *f=fopen(path,"r"); if(!f) return 0;
    Cfg *c=&m->c; int l,e; uint32_t cnt; int64_t tot=0;
    while(fscanf(f,"%d %d %u",&l,&e,&cnt)==3)
        if(l>=0&&l<c->n_layers&&e>=0&&e<c->n_experts&&m->eusage[l]){ m->eusage[l][e]+=cnt; tot+=cnt; }
    fclose(f); return tot;
}
static void usage_save(Model *m){ if(g_usage_path[0]) stats_dump_q(m,g_usage_path,1); }

/* log-likelihood scoring (SCORE=requests.txt): one forward per line, teacher-forcing */
static double logprob_target(const float *lo, int V, int target, int *am){
    float mx=lo[0]; int best=0; for(int i=1;i<V;i++){ if(lo[i]>mx){mx=lo[i];best=i;} }
    double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
    if(am)*am=(best==target);
    return (double)(lo[target]-mx)-log(se);
}
static void run_score(Model *m, const char *path){
    Cfg *c=&m->c; int D=c->hidden;
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    int maxT=1; { char *ln=NULL; size_t cp=0;
        while(getline(&ln,&cp,f)>0){ int a,b; if(sscanf(ln,"%d %d",&a,&b)==2&&a+b>maxT) maxT=a+b; }
        free(ln); }
    kv_alloc(m,maxT);
    float *x=falloc((int64_t)maxT*D), *lo=falloc(c->vocab), *row=falloc(D);
    int *ids=malloc(maxT*sizeof(int));
    rewind(f); char *ln=NULL; size_t cp=0; int nreq=0; double t0=now_s();
    while(getline(&ln,&cp,f)>0){
        char *p=ln; int ctxlen=strtol(p,&p,10), contlen=strtol(p,&p,10), T=ctxlen+contlen;
        if(T<=0||ctxlen<1){ printf("0 0 0\n"); fflush(stdout); continue; }
        for(int i=0;i<T;i++) ids[i]=strtol(p,&p,10);
        for(int s=0;s<T;s++) embed_row(m,ids[s],x+(int64_t)s*D);
        layers_forward(m,x,T,0);
        double lp=0; int greedy=1;
        for(int pos=ctxlen-1; pos<T-1; pos++){
            rmsnorm(row,x+(int64_t)pos*D,m->final_norm,D,c->eps);
            matmul_qt(lo,row,&m->lm_head,1);
            int am; lp+=logprob_target(lo,c->vocab,ids[pos+1],&am); if(!am) greedy=0;
        }
        printf("%.6f %d %d\n",lp,contlen,greedy); fflush(stdout);
        if(++nreq%5==0) fprintf(stderr,"[score %d req | %.1fs | RSS %.2f GB | hit %.0f%%]\n",
            nreq,now_s()-t0,rss_gb(),(m->hits+m->miss)?100.0*m->hits/(m->hits+m->miss):0.0);
    }
    free(ln); free(ids); free(x); free(lo); free(row); fclose(f);
}

static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires prompt+continuation\n"); return; }
    kv_alloc(m,nfull+2);
    float *logit=step(m,full,np-1,0); free(logit);
    m->hits=m->miss=m->ereq=0; m->t_edisk=m->t_emm=m->t_attn=m->t_head=0;
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){ logit=step(m,full+i,1,i); free(logit); steps++; }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
}

static int decode_tokens(Model *m, Tok *T, int *all, int kv, int n_new, int eos, float *logit,
                         double *decode_sec){
    EmitStream es={T,m,now_s(),0};
    int n=spec_decode(m,all,kv,n_new,eos,logit,emit_stream,&es,NULL);
    if(decode_sec) *decode_sec=now_s()-es.t0;
    return n;
}

static const char *hy3_effort(int think){
    if(!think) return "reasoning_effort:no_think";
    const char *e=getenv("REASONING_EFFORT");
    if(e){
        if(!strcmp(e,"high")||!strcmp(e,"xhigh")) return "reasoning_effort:high";
        if(!strcmp(e,"max")) return "reasoning_effort:max";
        if(!strcmp(e,"low")||!strcmp(e,"minimal")) return "reasoning_effort:low";
        if(!strcmp(e,"medium")) return "reasoning_effort:medium";
    }
    return "reasoning_effort:high";
}
static int hy3_wrap(char *buf, int cap, const char *user, int think){
    const char *eff=hy3_effort(think);
    const char *tk=think?"<think:opensource>":"<think:opensource></think:opensource>";
    return snprintf(buf,cap,
        "<\xEF\xBD\x9Chy_begin_of_sentence:opensource\xEF\xBD\x9C>"
        "<\xEF\xBD\x9Creasoning_mode:opensource\xEF\xBD\x9C>%s"
        "<\xEF\xBD\x9Chy_User:opensource\xEF\xBD\x9C>%s"
        "<\xEF\xBD\x9Chy_Assistant:opensource\xEF\xBD\x9C>%s",
        eff,user,tk);
}

static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    if(eos<0) eos=tok_id_of(&T,"<|end|>");
    if(g_temp<0) g_temp=0.7f;
    int think=getenv("THINK")?atoi(getenv("THINK")):0;
    char wrapped[1<<16]; int bl=hy3_wrap(wrapped,sizeof(wrapped),prompt,think);
    int cap=bl+16; int *pids=malloc((size_t)cap*sizeof(int));
    int np=tok_encode(&T,wrapped,bl,pids,cap);
    if(np<1){ fprintf(stderr,"prompt is empty after tokenization\n"); free(pids); return; }
    printf("prompt: %d tokens | generating up to %d (temp=%.2f nucleus=%.2f) | draft=%d\n",
        np,ngen,g_temp,g_nuc,g_draft);
    kv_alloc(m,np+ngen+g_draft+2);
    int *all=malloc((size_t)(np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,(size_t)np*sizeof(int));
    double t=now_s();
    float *logit=step(m,pids,np,0);
    int produced=decode_tokens(m,&T,all,np,ngen,eos,logit,NULL);
    double dt=now_s()-t, tot=m->hits+m->miss;
    int nsp=0; for(int i=0;i<m->c.n_layers;i++) if(m->L[i].sparse) nsp++;
    printf("\n---\n%d tokens in %.2fs (%.2f tok/s) | expert hit %.1f%% | RSS %.2f GB\n",
        produced,dt,produced/dt,tot?100.0*m->hits/tot:0.0,rss_gb());
    printf("experts loaded/token: %.1f (per-layer %.2f across %d; topk=%d) | TOPK=%d TOPP=%.2f\n",
        produced?(double)m->ereq/produced:0.0,(produced&&nsp)?(double)m->ereq/produced/nsp:0.0,
        nsp,m->c.topk,g_topk,g_topp);
    printf("speculation: %.2f tokens/forward (%llu forwards per %llu tokens) | MTP acceptance %.0f%% (%llu/%llu)\n",
        m->n_fw?(double)m->n_emit/m->n_fw:1.0,(unsigned long long)m->n_fw,(unsigned long long)m->n_emit,
        m->mtp_prop?100.0*m->mtp_acc/m->mtp_prop:0.0,(unsigned long long)m->mtp_acc,(unsigned long long)m->mtp_prop);
    profile_print(m,dt);
    free(pids); free(all);
    usage_save(m);
}

static void run_serve(Model *m, const char *snap){
    coli_serve_binary_mode();   /* #748: TEXT-mode stdout mangles the READY sentinel */
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    if(eos<0) eos=tok_id_of(&T,"<|end|>");
    if(g_temp<0) g_temp=0.7f;
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):256;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int templ=getenv("CHAT_TEMPLATE")?atoi(getenv("CHAT_TEMPLATE")):1;
    int think=getenv("THINK")?atoi(getenv("THINK")):0;
    int *hist=malloc((size_t)maxctx*sizeof(int));
    int len=0, first=1;
    char *line=NULL; size_t cap=0; ssize_t nr;
    char *buf=malloc(1<<16);
    kv_alloc(m,maxctx);
    printf("\x01\x01" "READY" "\x01\x01\n");
    printf("STAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout);
    while((nr=getline(&line,&cap,stdin))>0){
        if(nr>0&&line[nr-1]=='\n') line[--nr]=0;
        if(!strcmp(line,"\x02RESET")){ len=0; first=1; kv_alloc(m,4096);
            if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout); continue; }
        if(!strcmp(line,"\x02MORE")){
            if(len<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout); continue; }
            uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
            float *logit=step(m,hist+len-1,1,len-1);
            double decode_t=0;
            int prod=decode_tokens(m,&T,hist,len,ngen,eos,logit,&decode_t); len+=prod;
            double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
            double decode_tps=prod>0&&decode_t>1e-6?prod/decode_t:0.0;
            double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
            printf("\n\x01\x01" "END" "\x01\x01\n");
            printf("STAT %d %.2f %.1f %.2f 0 0 %.2f\n",prod,prod/tdt,(dh+dm)>0?100.0*dh/(dh+dm):0.0,
                rss_gb(),decode_tps);
            fflush(stdout); usage_save(m); repin_pass(m); continue;
        }
        if(nr<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n",rss_gb()); fflush(stdout); continue; }
        char *raw=NULL, *input=line;
        int input_n=(int)nr, raw_mode=0, req_ngen=ngen, prompt_tokens=0;
        float base_temp=g_temp, base_nuc=g_nuc;
        if(!strncmp(line,"\x02PROMPT ",8)){
            unsigned long long nb=0; double rt=0, rp=0; int slot=0;
            int nf=sscanf(line+8,"%llu %d %lf %lf %d",&nb,&req_ngen,&rt,&rp,&slot);
            (void)slot;
            if(nf<4||nb>(16u<<20)||req_ngen<1||rt<0||rt>2||rp<=0||rp>1){
                printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;
            }
            raw=malloc((size_t)nb+1); if(!raw){fprintf(stderr,"OOM raw prompt\n");exit(1);}
            if(fread(raw,1,(size_t)nb,stdin)!=(size_t)nb){free(raw);break;}
            int delim=fgetc(stdin); if(delim!='\n'&&delim!=EOF) ungetc(delim,stdin);
            if(memchr(raw,0,(size_t)nb)){free(raw); printf("\x01\x01" "END" "\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;}
            raw[nb]=0; input=raw; input_n=(int)nb; raw_mode=1;
            if(req_ngen>ngen) req_ngen=ngen;
            g_temp=(float)rt; g_nuc=(float)rp;
        }
        int bl=0, k=0;
        if(raw_mode){
            int *tmp=malloc(maxctx*sizeof(int)); if(!tmp){fprintf(stderr,"OOM raw tokens\n");exit(1);}
            prompt_tokens=tok_encode(&T,input,input_n,tmp,maxctx-8);
            int old_len=len, prefix=0;
            while(prefix<old_len&&prefix<prompt_tokens&&hist[prefix]==tmp[prefix]) prefix++;
            if(prefix<old_len) len=prefix;
            k=prompt_tokens-len;
            if(k>0) memcpy(hist+len,tmp+len,k*sizeof(int));
            fprintf(stderr,"[API] KV prefix %d/%d token, prefill %d\n",len,prompt_tokens,k);
            free(tmp);
        } else {
            if(templ) bl=hy3_wrap(buf,1<<16,input,think);
            else bl=snprintf(buf,1<<16,"%s",input);
            k=tok_encode(&T,buf,bl,hist+len,maxctx-len-8); prompt_tokens=len+k;
            if(len+k+req_ngen+2>=maxctx){ len=0; first=1; kv_alloc(m,4096);
                if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;
                bl=0; if(templ) bl=hy3_wrap(buf,1<<16,input,think);
                else bl=snprintf(buf,1<<16,"%s",input);
                k=tok_encode(&T,buf,bl,hist,maxctx-8); if(k>maxctx-8) k=maxctx-8;
                prompt_tokens=k;
            }
        }
        if(prompt_tokens<1){ free(raw); g_temp=base_temp; g_nuc=base_nuc;
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue; }
        (void)first; first=0;
        int cur=req_ngen; if(len+k+cur+g_draft+2>=maxctx) cur=maxctx-len-k-g_draft-2;
        uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
        float *logit;
        if(k>0){ logit=step(m,hist+len,k,len); len+=k; }
        else logit=step(m,hist+len-1,1,len-1);
        int prod=0; double decode_t=0;
        if(cur>0) prod=decode_tokens(m,&T,hist,len,cur,eos,logit,&decode_t);
        else free(logit);
        len+=prod;
        double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
        double decode_tps=prod>0&&decode_t>1e-6?prod/decode_t:0.0;
        double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
        printf("%s\x01\x01" "END" "\x01\x01\n",raw_mode?"":"\n");
        printf("STAT %d %.2f %.1f %.2f %d %d %.2f\n",prod,prod/tdt,(dh+dm)>0?100.0*dh/(dh+dm):0.0,
            rss_gb(),prompt_tokens,prod>=cur,decode_tps);
        fflush(stdout); usage_save(m); repin_pass(m);
        free(raw); g_temp=base_temp; g_nuc=base_nuc;
    }
    free(line); free(buf); free(hist); usage_save(m);
}

static void run_prompt_ids(Model *m, int ngen){
    int np=0; int *prompt=parse_ids_env(&np);
    if(!prompt||np<1){ fprintf(stderr,"PROMPT mode: set IDS=1,2,3,...\n"); exit(1); }
    int cap=np+ngen; int *out=malloc(cap*sizeof(int));
    double t=now_s(); generate(m,prompt,np,ngen,out); double dt=now_s()-t;
    printf("generated %d tokens in %.2fs (%.2f tok/s)\n",ngen,dt,ngen/dt);
    printf("tokens:"); for(int i=np;i<np+ngen;i++) printf(" %d",out[i]); printf("\n");
    double tot=m->hits+m->miss;
    printf("expert hit rate: %.1f%% | RSS %.2f GB\n",tot?100.0*m->hits/tot:0.0,rss_gb());
    free(prompt); free(out);
}

int main(int argc, char **argv){
/* Windows: stdin/stdout default to text mode, which mangles the \x02RESET / \x02MORE
 * binary control sequences and the STAT protocol. Force binary + unbuffered. */
#ifdef _WIN32
#include <fcntl.h>
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    setvbuf(stdout, NULL, _IONBF, 0);
#endif    
    const char *snap=getenv("SNAP"); if(!snap){fprintf(stderr,"SNAP=<dir>\n");return 1;}
    g_nopack=getenv("NOPACK")?1:0;
    g_drop=getenv("DROP")?1:0;
    g_direct=getenv("DIRECT")?atoi(getenv("DIRECT")):0;
    g_idot=getenv("IDOT")?atoi(getenv("IDOT")):0;  /* IDOT=1 breaks int4/int8 on Hy3 until validated */
    g_i4s=getenv("I4S")?atoi(getenv("I4S")):g_i4s;
    g_pipe=getenv("PIPE")?atoi(getenv("PIPE")):0;
    if(g_pipe==2){
#if !defined(__linux__) || !defined(COLI_IOURING)
        fprintf(stderr,"[PIPE] io_uring requires Linux build with IOURING=1, using thread pool\n");
        g_pipe=1;
#endif
    }
    g_pipe_nw=getenv("PIPE_WORKERS")?atoi(getenv("PIPE_WORKERS")):8;
    if(g_pipe_nw<1) g_pipe_nw=1;
    g_perf=getenv("PERF")?atoi(getenv("PERF")):0;
    g_kv_i8=getenv("KV_I8")?atoi(getenv("KV_I8")):0;
    g_tree_draft=getenv("TREE_DRAFT")?atoi(getenv("TREE_DRAFT")):0;
    if(g_tree_draft) fprintf(stderr,"[TREE_DRAFT] tree speculative decoding enabled\n");
    g_temp=getenv("TEMP")?atof(getenv("TEMP")):-1;
    g_nuc=getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.90f;
    g_topk=getenv("TOPK")?atoi(getenv("TOPK")):0;
    g_topp=getenv("TOPP")?atof(getenv("TOPP")):0;
    g_repin=getenv("REPIN")?atoi(getenv("REPIN")):0;
    g_draft=getenv("DRAFT")?atoi(getenv("DRAFT")):-1;
    if(g_draft>63) g_draft=63;
    int cap=argc>1?atoi(argv[1]):64;
    int ebits=argc>2?atoi(argv[2]):8;
    int dbits=argc>3?atoi(argv[3]):ebits;
#ifdef COLI_CUDA
    if(getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))){
        const char *one=getenv("COLI_GPU"), *many=getenv("COLI_GPUS");
        if(one&&many){ fprintf(stderr,"use COLI_GPU or COLI_GPUS, not both\n"); return 2; }
        if(many) g_cuda_ndev=parse_cuda_devices(many,g_cuda_devices);
        else if(one) g_cuda_ndev=parse_cuda_devices(one,g_cuda_devices);
        else { g_cuda_ndev=1; g_cuda_devices[0]=0; }
        if(g_cuda_ndev<1){ fprintf(stderr,"invalid COLI_GPUS: use a list such as 0,1,2\n"); return 2; }
        g_cuda_enabled=coli_cuda_init(g_cuda_devices,g_cuda_ndev);
        if(!g_cuda_enabled){ fprintf(stderr,COLI_ACCEL_TAG " requested backend is unavailable\n"); return 2; }
    }
    g_cuda_dense=getenv("CUDA_DENSE")?atoi(getenv("CUDA_DENSE")):0;
    g_cuda_attn=getenv("CUDA_ATTN")?atoi(getenv("CUDA_ATTN")):0;
    g_cuda_expert_gb=getenv("CUDA_EXPERT_GB")?atof(getenv("CUDA_EXPERT_GB")):0;
    if((getenv("COLI_GPU")||getenv("COLI_GPUS"))&&!g_cuda_enabled){ fprintf(stderr,"COLI_GPU(S) requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_dense&&!g_cuda_enabled){ fprintf(stderr,"CUDA_DENSE requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_expert_gb>0 && !g_cuda_enabled){ fprintf(stderr,"CUDA_EXPERT_GB requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_enabled) fprintf(stderr,COLI_ACCEL_TAG " mode: routed experts%s\n",g_cuda_dense?" + resident dense tensors":" only (resident dense on CPU)");
#else
    if((getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))) ||
       getenv("COLI_GPU") || getenv("COLI_GPUS") ||
       (getenv("CUDA_DENSE") && atoi(getenv("CUDA_DENSE"))) ||
       (getenv("CUDA_EXPERT_GB") && atof(getenv("CUDA_EXPERT_GB"))>0)){
        fprintf(stderr,"CUDA was requested, but this binary is CPU-only; rebuild with: make hy3 CUDA=1\n");
        return 2;
    }
#endif
    printf("== Hy3 C engine, cache=%d experts/layer | experts@%d-bit dense@%d-bit ==\n",cap,ebits,dbits);
    g_mem_avail_boot=mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    if(g_draft<0) g_draft=m.has_mtp?3:0;
    fprintf(stderr,"[MTP] %s (draft=%d)\n",
        m.has_mtp?"active: native speculative decoding":"absent",g_draft);
    printf("loaded in %.2fs | resident dense: %.2f MB | layers=%d experts=%d%s\n",
        now_s()-t0,m.resident_bytes/(1024.0*1024.0),m.c.n_layers,m.c.n_experts,
        m.has_mtp?" | MTP head active":"");
    if(g_kv_i8){
        int est_ctx=getenv("CTX")?atoi(getenv("CTX")):4096;
        double i8b=kv_pool_bytes(&m,est_ctx);
        int save=g_kv_i8; g_kv_i8=0; double f32b=kv_pool_bytes(&m,est_ctx); g_kv_i8=save;
        fprintf(stderr,"[KV] int8 cache: %.2f GB vs f32 %.2f GB at ctx=%d\n",i8b/1e9,f32b/1e9,est_ctx);
    }
    if(!strncmp(snap,"/mnt/",5))
        fprintf(stderr,"WARNING: model on %s (slow 9p mount). Use ext4 (e.g. /home/ or native /mnt/d/) for speed.\n",snap);
    if(getenv("PIN")) pin_load(&m,getenv("PIN"),getenv("PIN_GB")?atof(getenv("PIN_GB")):10.0);
    { double ram_env=getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx=getenv("CTX")?atoi(getenv("CTX")):4096;
      snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
      int64_t hist=usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",(long long)hist,g_usage_path);
      int autopin=getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN")&&autopin&&hist>=5000){
          double conf=(double)hist/200000.0; if(conf>1) conf=1;
          double pin_gb=expert_avail(&m,ram_env,ebits,est_ctx)*0.5*conf/1e9;
          if(pin_gb>=0.5) pin_load(&m,g_usage_path,pin_gb);
      }
      cap_for_ram(&m,ram_env,ebits,est_ctx); }

    const char *stats=getenv("STATS");
    if(getenv("SCORE")){ run_score(&m,getenv("SCORE")); if(stats) stats_dump(&m,stats); usage_save(&m); return 0; }

    if(getenv("SERVE")){ run_serve(&m,snap); if(stats) stats_dump(&m,stats); usage_save(&m); return 0; }
    if(getenv("PROMPT")){
        if(getenv("IDS")) run_prompt_ids(&m,getenv("NGEN")?atoi(getenv("NGEN")):32);
        else run_text(&m,snap,getenv("PROMPT"),getenv("NGEN")?atoi(getenv("NGEN")):32);
        if(stats) stats_dump(&m,stats); usage_save(&m); return 0;
    }

    const char *refpath=getenv("REF")?getenv("REF"):"ref_hy3.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    char *ar=NULL; jval *ref=json_parse(b,&ar);

    if(getenv("TF")){
        int nfull=0; int *full=read_arr(ref,"full_ids",&nfull);
        int ntf=0; int *tf=read_arr(ref,"tf_pred",&ntf);
        if(ntf!=nfull){ fprintf(stderr,"tf_pred length mismatch\n"); return 1; }
        { int maxid=0; for(int i=0;i<nfull;i++) if(full[i]>maxid) maxid=full[i];
          if(m.c.vocab>1000&&maxid<1000&&!getenv("REF_FORCE")){
            fprintf(stderr,"ERRORE: ref_hy3.json is the tiny oracle (max id %d, vocab %d).\n"
                           "  Self-test: SNAP=./hy3_tiny TF=1 ./hy3 64 16 16\n",maxid,m.c.vocab);
            return 1;
          } }
        int *pred=malloc(nfull*sizeof(int)); double tt=now_s();
        forward_all(&m,full,nfull,pred); double tdt=now_s()-tt;
        int ok=0; for(int i=0;i<nfull;i++){
            if(pred[i]==tf[i]) ok++;
            else fprintf(stderr,"[ORACLE] mismatch pos=%d expected=%d got=%d\n",i,tf[i],pred[i]);
        }
        printf("PREFILL (teacher-forcing) C vs oracle: %d/%d positions | %.1f pos/s\n",ok,nfull,nfull/tdt);
        profile_print(&m,tdt);
#ifdef COLI_CUDA
        if(m.gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
            m.gpu_expert_count,m.gpu_expert_bytes/1e9,(unsigned long long)m.gpu_expert_calls);
        if(g_cuda_enabled) cuda_stats_print();
#endif
        free(full); free(tf); free(pred); free(b); free(ar); return ok==nfull?0:1;
    }

    int np,nfull; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    int n_new=nfull-np;

    if(getenv("REPLAY")){ run_replay(&m,full,nfull,np); free(b); free(ar); free(prompt); free(full); return 0; }
    int *out=malloc((np+n_new)*sizeof(int));
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nReference: "); for(int i=np;i<nfull;i++) printf("%d ",full[i]);
    printf("\nHy3 C engine: "); for(int i=np;i<nfull;i++){ printf("%d ",out[i]); if(out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n",match,n_new);
    double tot=m.hits+m.miss;
    printf("Expert cache hit rate: %.1f%% | RSS: %.2f GB | %.1f tok/s\n",
        tot?100.0*m.hits/tot:0.0,rss_gb(),n_new/dt);
    profile_print(&m,dt);
#ifdef COLI_CUDA
    if(m.gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m.gpu_expert_count,m.gpu_expert_bytes/1e9,(unsigned long long)m.gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    if(stats) stats_dump(&m,stats);
    usage_save(&m);
    free(b); free(ar); free(prompt); free(full); free(out);
    return 0;
}
