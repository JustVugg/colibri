/* quant.h — quantized matmul kernels (header-only, all functions static).
 * Multi-architecture SIMD: AVX2 / AVX-512 / AVX-VNNI / ARM NEON / NEON-SDOT /
 * NEON-i8mm / POWER VSX.  Pure compute — no Model or QT dependency. */
#ifndef COLI_QUANT_H
#define COLI_QUANT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- SIMD includes -------------------------------------------------------- */
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
static inline int hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
#endif
#if defined(__AVXVNNI__) && defined(__AVX2__)
static inline int hsum128_i32(__m128i v){
    v=_mm_hadd_epi32(v,v); v=_mm_hadd_epi32(v,v); return _mm_cvtsi128_si32(v);
}
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __VSX__
#include <altivec.h>
#undef vector
#undef pixel
#undef bool
#endif

/* ---- AVX-512 int4->float accumulator -------------------------------------- */
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

/* ---- y[S,O] = x[S,I] @ W^T, W[O,I] f32 ---------------------------------- */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}

/* ---- y[S,O] = x[S,I] @ W^T, W int8 per-row + scale[O] ------------------- */
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

/* ---- y[S,O] = x[S,I] @ W^T, W int4 packed (2/byte) + scale[O] ------------ */
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
            for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib=_mm_unpacklo_epi8(lo,hi);
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
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
#if defined(__AVX512F__) && defined(__AVX512BW__)
            }
#endif
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += xs[i]*(float)lo; }
            y[(int64_t)s*O+o]=a*sc; } }
}

/* ---- y[S,O] = x[S,I] @ W^T, W int4 packed + per-GROUP scales (fmt=4) ----- */
static void matmul_i4_grouped(float *y, const float *x, const uint8_t *q4, const float *scale,
                              int S, int I, int O, int gs){
    int rb=(I+1)/2; int ng=(I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs; int glen=gs; if(base+glen>I) glen=I-base;
                float sc=scl[g];
                int i=base;
#ifdef __AVX2__
                const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
                __m256 acc=_mm256_setzero_ps();
                for(; i+16<=base+glen; i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                    __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
                a+=hsum256(acc)*sc;
#endif
                for(; i<base+glen; i+=2){
                    if(i+1<base+glen){ uint8_t byte=w[i>>1];
                        a+=(xs[i]*(float)((int)(byte&0xF)-8)+xs[i+1]*(float)((int)(byte>>4)-8))*sc; }
                    else { uint8_t byte=w[i>>1]; a+=xs[i]*(float)((int)(byte&0xF)-8)*sc; }
                }
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ---- fused gate+up: one OMP dispatch for both matrices -------------------- */
static void matmul_i4_pair(float *yg, float *yu, const float *x,
                           const uint8_t *qg, const float *sg,
                           const uint8_t *qu, const float *su, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int z=0;z<2*O;z++){
        int o=z<O?z:z-O; const uint8_t *w=(z<O?qg:qu)+(int64_t)o*rb;
        float a=0; int i=0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
        if(g_i4_acc512){ a=dot_i4f_avx512(w,x,I); i=I&~31; }
        else {
#endif
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
#if defined(__AVX512F__) && defined(__AVX512BW__)
        }
#endif
        for(;i+1<I;i+=2){ uint8_t b=w[i>>1]; a+=x[i]*(float)((b&15)-8)+x[i+1]*(float)((b>>4)-8); }
        if(i<I) a+=x[i]*(float)((w[i>>1]&15)-8);
        (z<O?yg:yu)[o]=a*(z<O?sg:su)[o];
    }
}

/* ---- y[S,O] = x[S,I] @ W^T, W int2 packed (4/byte) + scale[O] ------------ */
static void matmul_i2(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O){
    int rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q2+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
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
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m2v=vdup_n_u8(3); const int8x8_t b2v=vdup_n_s8(2);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint32_t wd; memcpy(&wd, w+(i>>2), 4);
                uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),              vand_u8(vshr_n_u8(by,2),m2v));
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
            for(;i<I;i++){ uint8_t byte=w[i>>2]; int sh=(i&3)*2; a += xs[i]*(float)((int)((byte>>sh)&3)-2); }
            y[(int64_t)s*O+o]=a*sc; } }
}

/* ---- IDOT: integer dot kernels (int8-quantized activations) --------------- */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define IDOT_KERNEL "avx512-vnni"
#elif defined(__AVXVNNI__) && defined(__AVX2__)
#define IDOT_KERNEL "avx-vnni"
#elif defined(__AVX2__)
#define IDOT_KERNEL "avx2"
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
#define IDOT_KERNEL "neon-i8mm"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#elif defined(__VSX__)
#define IDOT_KERNEL "vsx"
#else
#define IDOT_KERNEL "scalar"
#endif
static int g_idot=1;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int g_i4s=1;
#elif defined(__VSX__)
static int g_i4s=1;
#else
static int g_i4s=2;
#endif

static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}

/* dot int8*int8 */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
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
    __m128i acc=_mm_setzero_si128();
    for(;i+16<=I;i+=16){
        __m128i wv=_mm_loadu_si128((const __m128i*)(w+i));
        __m128i xv=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i xs=_mm_sign_epi8(xv,wv);
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
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
    for(;i+64<=I;i+=64){
        a0=vdotq_s32(a0,vld1q_s8(w+i),   vld1q_s8(x+i));
        a1=vdotq_s32(a1,vld1q_s8(w+i+16),vld1q_s8(x+i+16));
        a2=vdotq_s32(a2,vld1q_s8(w+i+32),vld1q_s8(x+i+32));
        a3=vdotq_s32(a3,vld1q_s8(w+i+48),vld1q_s8(x+i+48));
    }
    int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
    for(;i+16<=I;i+=16) acc=vdotq_s32(acc,vld1q_s8(w+i),vld1q_s8(x+i));
    sum=vaddvq_s32(acc);
#else
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){
        int8x16_t wv=vld1q_s8(w+i), xv=vld1q_s8(x+i);
        int16x8_t p=vmull_s8(vget_low_s8(wv),vget_low_s8(xv));
        p=vmlal_s8(p,vget_high_s8(wv),vget_high_s8(xv));
        acc=vpadalq_s16(acc,p);
    }
    sum=vaddvq_s32(acc);
#endif
#elif defined(__VSX__)
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

/* dot int4(packed)*int8 */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
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
    const __m128i m4=_mm_set1_epi8(0x0F); const __m128i b8=_mm_set1_epi8(8);
    __m128i acc=_mm_setzero_si128();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);
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
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
    for(;i+64<=I;i+=64){
        uint8x16_t byA=vld1q_u8(w4+(i>>1)), byB=vld1q_u8(w4+(i>>1)+16);
        uint8x16x2_t zA=vzipq_u8(vandq_u8(byA,m4q), vshrq_n_u8(byA,4));
        uint8x16x2_t zB=vzipq_u8(vandq_u8(byB,m4q), vshrq_n_u8(byB,4));
        a0=vdotq_s32(a0,vsubq_s8(vreinterpretq_s8_u8(zA.val[0]),b8q),vld1q_s8(x+i));
        a1=vdotq_s32(a1,vsubq_s8(vreinterpretq_s8_u8(zA.val[1]),b8q),vld1q_s8(x+i+16));
        a2=vdotq_s32(a2,vsubq_s8(vreinterpretq_s8_u8(zB.val[0]),b8q),vld1q_s8(x+i+32));
        a3=vdotq_s32(a3,vsubq_s8(vreinterpretq_s8_u8(zB.val[1]),b8q),vld1q_s8(x+i+48));
    }
    int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4));
        acc=vdotq_s32(acc,vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q),vld1q_s8(x+i));
        acc=vdotq_s32(acc,vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q),vld1q_s8(x+i+16));
    }
    sum=vaddvq_s32(acc);
#else
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4));
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
        int16x8_t p=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));
        p=vmlal_s8(p,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p);
        p=vmull_s8(vget_low_s8(w1),vget_low_s8(x1));
        p=vmlal_s8(p,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p);
    }
    sum=vaddvq_s32(acc);
#endif
#elif defined(__VSX__)
    const __vector unsigned char m4v=vec_splats((unsigned char)0x0F);
    const __vector unsigned char sh4=vec_splats((unsigned char)4);
    const __vector signed char b8v=vec_splats((signed char)8);
    const __vector signed char vz=vec_splats((signed char)0);
    __vector signed int acc=vec_splats(0);
    for(;i+32<=I;i+=32){
        __vector unsigned char by=vec_xl(0,w4+(i>>1));
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

/* ---- ARM i8mm SMMLA tiled kernels ---------------------------------------- */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
static inline int32x4_t mm_tile16(int32x4_t acc, int8x16_t wo, int8x16_t wo1,
                                  int8x16_t xs, int8x16_t xs1){
    acc=vmmlaq_s32(acc, vcombine_s8(vget_low_s8(wo), vget_low_s8(wo1)),
                        vcombine_s8(vget_low_s8(xs), vget_low_s8(xs1)));
    return vmmlaq_s32(acc, vcombine_s8(vget_high_s8(wo), vget_high_s8(wo1)),
                           vcombine_s8(vget_high_s8(xs), vget_high_s8(xs1)));
}
static void matmul_q_idot_mm(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                             const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<(O&~1);o+=2){
        const int8_t *wo=q+(int64_t)o*I, *wo1=q+(int64_t)(o+1)*I;
        float sc0=scale[o], sc1=scale[o+1];
        for(int s=0;s<(S&~1);s+=2){
            const int8_t *xs=xq+(int64_t)s*I, *xs1=xq+(int64_t)(s+1)*I;
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0); int i=0;
            for(;i+64<=I;i+=64){
                a0=mm_tile16(a0,vld1q_s8(wo+i),   vld1q_s8(wo1+i),   vld1q_s8(xs+i),   vld1q_s8(xs1+i));
                a1=mm_tile16(a1,vld1q_s8(wo+i+16),vld1q_s8(wo1+i+16),vld1q_s8(xs+i+16),vld1q_s8(xs1+i+16));
                a2=mm_tile16(a2,vld1q_s8(wo+i+32),vld1q_s8(wo1+i+32),vld1q_s8(xs+i+32),vld1q_s8(xs1+i+32));
                a3=mm_tile16(a3,vld1q_s8(wo+i+48),vld1q_s8(wo1+i+48),vld1q_s8(xs+i+48),vld1q_s8(xs1+i+48));
            }
            for(;i+16<=I;i+=16)
                a0=mm_tile16(a0,vld1q_s8(wo+i),vld1q_s8(wo1+i),vld1q_s8(xs+i),vld1q_s8(xs1+i));
            int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
            int32_t d00=vgetq_lane_s32(acc,0), d01=vgetq_lane_s32(acc,1);
            int32_t d10=vgetq_lane_s32(acc,2), d11=vgetq_lane_s32(acc,3);
            for(;i<I;i++){ int a=wo[i],b=wo1[i],u=xs[i],v=xs1[i];
                d00+=a*u; d01+=a*v; d10+=b*u; d11+=b*v; }
            y[(int64_t)s*O+o]        =(float)d00*sc0*sx[s];
            y[(int64_t)s*O+(o+1)]    =(float)d10*sc1*sx[s];
            y[(int64_t)(s+1)*O+o]    =(float)d01*sc0*sx[s+1];
            y[(int64_t)(s+1)*O+(o+1)]=(float)d11*sc1*sx[s+1];
        }
        if(S&1){ int s=S-1; const int8_t *xs=xq+(int64_t)s*I;
            y[(int64_t)s*O+o]    =(float)dot_i8i8(wo, xs,I)*sc0*sx[s];
            y[(int64_t)s*O+(o+1)]=(float)dot_i8i8(wo1,xs,I)*sc1*sx[s]; }
    }
    if(O&1){ int o=O-1; const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        #pragma omp parallel for schedule(static)
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot_mm(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                              const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<(O&~1);o+=2){
        const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
        const uint8_t *wo=q4+(int64_t)o*rb, *wo1=q4+(int64_t)(o+1)*rb;
        float sc0=scale[o], sc1=scale[o+1];
        for(int s=0;s<(S&~1);s+=2){
            const int8_t *xs=xq+(int64_t)s*I, *xs1=xq+(int64_t)(s+1)*I;
            int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0); int i=0;
            for(;i+64<=I;i+=64){
                uint8x16_t byo=vld1q_u8(wo+(i>>1)), byo1=vld1q_u8(wo1+(i>>1));
                uint8x16_t cyo=vld1q_u8(wo+(i>>1)+16), cyo1=vld1q_u8(wo1+(i>>1)+16);
                uint8x16x2_t zo =vzipq_u8(vandq_u8(byo, m4q), vshrq_n_u8(byo, 4));
                uint8x16x2_t zo1=vzipq_u8(vandq_u8(byo1,m4q), vshrq_n_u8(byo1,4));
                uint8x16x2_t ko =vzipq_u8(vandq_u8(cyo, m4q), vshrq_n_u8(cyo, 4));
                uint8x16x2_t ko1=vzipq_u8(vandq_u8(cyo1,m4q), vshrq_n_u8(cyo1,4));
                a0=mm_tile16(a0, vsubq_s8(vreinterpretq_s8_u8(zo.val[0]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(zo1.val[0]),b8q),
                                 vld1q_s8(xs+i), vld1q_s8(xs1+i));
                a1=mm_tile16(a1, vsubq_s8(vreinterpretq_s8_u8(zo.val[1]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(zo1.val[1]),b8q),
                                 vld1q_s8(xs+i+16), vld1q_s8(xs1+i+16));
                a2=mm_tile16(a2, vsubq_s8(vreinterpretq_s8_u8(ko.val[0]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(ko1.val[0]),b8q),
                                 vld1q_s8(xs+i+32), vld1q_s8(xs1+i+32));
                a3=mm_tile16(a3, vsubq_s8(vreinterpretq_s8_u8(ko.val[1]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(ko1.val[1]),b8q),
                                 vld1q_s8(xs+i+48), vld1q_s8(xs1+i+48));
            }
            for(;i+32<=I;i+=32){
                uint8x16_t byo=vld1q_u8(wo+(i>>1)), byo1=vld1q_u8(wo1+(i>>1));
                uint8x16x2_t zo =vzipq_u8(vandq_u8(byo, m4q), vshrq_n_u8(byo, 4));
                uint8x16x2_t zo1=vzipq_u8(vandq_u8(byo1,m4q), vshrq_n_u8(byo1,4));
                a0=mm_tile16(a0, vsubq_s8(vreinterpretq_s8_u8(zo.val[0]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(zo1.val[0]),b8q),
                                 vld1q_s8(xs+i), vld1q_s8(xs1+i));
                a1=mm_tile16(a1, vsubq_s8(vreinterpretq_s8_u8(zo.val[1]),b8q),
                                 vsubq_s8(vreinterpretq_s8_u8(zo1.val[1]),b8q),
                                 vld1q_s8(xs+i+16), vld1q_s8(xs1+i+16));
            }
            int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
            int32_t d00=vgetq_lane_s32(acc,0), d01=vgetq_lane_s32(acc,1);
            int32_t d10=vgetq_lane_s32(acc,2), d11=vgetq_lane_s32(acc,3);
            for(;i+1<I;i+=2){ uint8_t bo=wo[i>>1], bo1=wo1[i>>1];
                int a0=(int)(bo&0xF)-8, a1=(int)(bo>>4)-8, b0=(int)(bo1&0xF)-8, b1=(int)(bo1>>4)-8;
                int u0=xs[i],u1=xs[i+1],v0=xs1[i],v1=xs1[i+1];
                d00+=a0*u0+a1*u1; d01+=a0*v0+a1*v1; d10+=b0*u0+b1*u1; d11+=b0*v0+b1*v1; }
            if(i<I){ uint8_t bo=wo[i>>1], bo1=wo1[i>>1];
                int a0=(int)(bo&0xF)-8, b0=(int)(bo1&0xF)-8;
                d00+=a0*xs[i]; d01+=a0*xs1[i]; d10+=b0*xs[i]; d11+=b0*xs1[i]; }
            y[(int64_t)s*O+o]        =(float)d00*sc0*sx[s];
            y[(int64_t)s*O+(o+1)]    =(float)d10*sc1*sx[s];
            y[(int64_t)(s+1)*O+o]    =(float)d01*sc0*sx[s+1];
            y[(int64_t)(s+1)*O+(o+1)]=(float)d11*sc1*sx[s+1];
        }
        if(S&1){ int s=S-1; const int8_t *xs=xq+(int64_t)s*I;
            y[(int64_t)s*O+o]    =(float)dot_i4i8(wo, xs,I)*sc0*sx[s];
            y[(int64_t)s*O+(o+1)]=(float)dot_i4i8(wo1,xs,I)*sc1*sx[s]; }
    }
    if(O&1){ int o=O-1; const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        #pragma omp parallel for schedule(static)
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
#endif

/* ---- IDOT dispatch (int8-quantized activations) --------------------------- */
static void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                          const float *scale, int S, int I, int O){
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    if(S>=2){ matmul_q_idot_mm(y,xq,sx,q,scale,S,I,O); return; }
#endif
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                           const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    if(S>=2){ matmul_i4_idot_mm(y,xq,sx,q4,scale,S,I,O); return; }
#endif
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}

/* ---- per-thread quantization scratch -------------------------------------- */
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

/* ---- f32 -> quantized packing --------------------------------------------- */
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
            qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
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
            for(int k=0;k<4 && i+k<I;k++){ int v=(int)lrintf(wr[i+k]/s); if(v>qmax)v=qmax; if(v<-2)v=-2; byte|=(uint8_t)((v+2)<<(k*2)); }
            qr[i>>2]=byte;
        }
    }
}

#endif /* COLI_QUANT_H */
