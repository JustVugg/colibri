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

#include "idot.h"   /* SIMD prelude + the integer dot kernels, shared with qwen36 */
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
/* acc[0..I) += coef * dequant(int4 row) — the axpy twin of dot_i4f_avx512, for the
 * MLA absorption path (qt_addrow). Each acc[i] receives exactly ONE fma per call
 * (no cross-element accumulation), so this is bit-identical to the scalar loop
 * (gcc -O3 -ffp-contract already emits scalar fma there). Tail handled scalar. */
static inline void axpy_i4f_avx512(const uint8_t *w,float coef,float *acc,int I){
    const __m128i m4=_mm_set1_epi8(0x0F); const __m512i b8=_mm512_set1_epi32(8);
    const __m512 cv=_mm512_set1_ps(coef); int i=0;
    for(;i+32<=I;i+=32){ __m128i by=_mm_loadu_si128((const __m128i*)(w+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi),n1=_mm_unpackhi_epi8(lo,hi);
        __m512 w0=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n0),b8));
        __m512 w1=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(n1),b8));
        _mm512_storeu_ps(acc+i,   _mm512_fmadd_ps(cv,w0,_mm512_loadu_ps(acc+i)));
        _mm512_storeu_ps(acc+i+16,_mm512_fmadd_ps(cv,w1,_mm512_loadu_ps(acc+i+16)));
    }
    for(;i+1<I;i+=2){ uint8_t b=w[i>>1]; acc[i]+=coef*(float)((int)(b&0xF)-8); acc[i+1]+=coef*(float)((int)(b>>4)-8); }
    if(i<I){ uint8_t b=w[i>>1]; acc[i]+=coef*(float)((int)(b&0xF)-8); }
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
                /* Pinned as an fma in the SOURCE. With the default
                 * -ffp-contract=fast the compiler may fuse this multiply-add
                 * (one rounding) or not (two), so the same source produced
                 * different bits depending on the flag -- see the PR for a
                 * reproduction. That made every bit-exactness gate, including
                 * the glm_tiny token oracle, depend on build flags rather than
                 * on the code. */
                a=fmaf(hsum256(acc),sc,a);
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

/* ---- int3-g64 (fmt=5): 3-bit weights with ONE f32 scale per 64-input group -
 * Per group: 16B low plane (2 bits/val, int2 layout) + 8B high plane (1 bit/val),
 * values in [-4,3] stored v+4. 3.5 bits/weight effective — the quality/size point
 * the #132 OLMoE ablation measured BEATING per-row int4. */
#define I3_GROUP 64
#define I3_GBYTES 24                     /* 16B low plane + 8B high plane per group */
static inline int64_t i3_groups(int I){ return ((int64_t)I + I3_GROUP - 1) / I3_GROUP; }
static inline int64_t i3_rowbytes(int I){ return i3_groups(I) * I3_GBYTES; }

#if defined(__AVX512F__) && defined(__AVX512BW__)
static int g_i3_avx512=1;
/* one full 64-value group -> f32 partial. Relies on immintrin.h arriving via the
 * __AVX2__-gated include above (AVX512F implies AVX2 on clang/gcc/MSVC), same as
 * dot_i4f_avx512. */
static inline float dot_i3g64_avx512(const uint8_t *lo, const uint8_t *hi, const float *x){
    const __m128i m2=_mm_set1_epi8(3); const __m512i c4=_mm512_set1_epi8(4);
    __m128i by=_mm_loadu_si128((const __m128i*)lo);
    __m128i p0=_mm_and_si128(by,m2),                   p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
    __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
    __m128i l01=_mm_unpacklo_epi8(p0,p1), h01=_mm_unpackhi_epi8(p0,p1);
    __m128i l23=_mm_unpacklo_epi8(p2,p3), h23=_mm_unpackhi_epi8(p2,p3);
    __m512i lov=_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(
        _mm512_castsi128_si512(_mm_unpacklo_epi16(l01,l23)),
        _mm_unpackhi_epi16(l01,l23),1),
        _mm_unpacklo_epi16(h01,h23),2),
        _mm_unpackhi_epi16(h01,h23),3);            /* byte k = low 2 bits of value k */
    uint64_t hb; memcpy(&hb,hi,8);                 /* mask bit k = high bit of value k */
    __m512i wq=_mm512_sub_epi8(_mm512_mask_add_epi8(lov,(__mmask64)hb,lov,c4),c4); /* [-4,3] in order */
    __m512 ac0=_mm512_setzero_ps(), ac1=_mm512_setzero_ps();
    ac0=_mm512_fmadd_ps(_mm512_loadu_ps(x),    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_castsi512_si128(wq))),      ac0);
    ac1=_mm512_fmadd_ps(_mm512_loadu_ps(x+16), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq,1))), ac1);
    ac0=_mm512_fmadd_ps(_mm512_loadu_ps(x+32), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq,2))), ac0);
    ac1=_mm512_fmadd_ps(_mm512_loadu_ps(x+48), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq,3))), ac1);
    return _mm512_reduce_add_ps(_mm512_add_ps(ac0,ac1));
}
static int i3_avx512_selftest(void){
    /* fixed group, asymmetric in every lane: pseudo-random 3-bit values with
     * distinct nonzero integer activations. All terms and partials are small
     * integers (exact in f32 under ANY summation order), so the compare is
     * exact — any lane permutation, bias error or plane mix-up shifts the sum. */
    uint8_t lo[16]={0}, hi[8]={0}; float x[I3_GROUP]; double ref=0;
    uint64_t r=0x9E3779B97F4A7C15ull;
    for(int k=0;k<I3_GROUP;k++){
        r^=r<<13; r^=r>>7; r^=r<<17;
        unsigned u=(unsigned)(r&7);
        lo[k>>2]|=(uint8_t)((u&3)<<((k&3)*2));
        hi[k>>3]|=(uint8_t)((u>>2)<<(k&7));
        x[k]=(k&1)?-(float)(k+1):(float)(k+1);
        ref+=(double)x[k]*((int)u-4);
    }
    float got=dot_i3g64_avx512(lo,hi,x);
    if(got!=(float)ref){ fprintf(stderr,"AVX512 i3 selftest: %.9g != %.9g\n",got,ref); return 0; }
    return 1;
}
#endif

/* Dequant-on-use with PER-GROUP scale. Exact f32 path only (no IDOT in v1: int8
 * activations don't compose with per-group accumulation without a kernel
 * restructure — follow-up). NEON: low plane = matmul_i2's unpack, high plane
 * expanded via vtst on bit masks. AVX2: same unpack, high plane via bit-test
 * against per-lane masks. AVX-512(F+BW): same unpack at 128-bit, high plane
 * loaded as a __mmask64 (bit k = value k) driving a masked +4; one full group
 * per iteration (dot_i3g64_avx512; I3_AVX512=0 falls back to scalar). All
 * vector arms reorder fma WITHIN a group only; the per-group partial is
 * scaled by scale[g] and added to the row accumulator in scalar order,
 * exactly like the scalar loop. */
static void matmul_i3(float *y, const float *x, const uint8_t *q3, const float *scale, int S, int I, int O){
    int64_t ng=i3_groups(I), rb=i3_rowbytes(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *wrow=q3+(int64_t)o*rb;
        const float *srow=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            float acc=0;
            for(int64_t g=0; g<ng; g++){
                const uint8_t *lo=wrow+g*I3_GBYTES, *hi=lo+16;
                int base=(int)(g*I3_GROUP), n = I-base < I3_GROUP ? I-base : I3_GROUP;
                float a=0; int k=0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
                if(g_i3_avx512 && n==I3_GROUP){ a=dot_i3g64_avx512(lo,hi,xs+base); k=I3_GROUP; }
#elif defined(__ARM_NEON)
                if(n==I3_GROUP){
                    const uint8x8_t m2v=vdup_n_u8(3); const int8x16_t b4q=vdupq_n_s8(4);
                    const uint8x16_t bitm={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
                    const uint8x16_t fourq=vdupq_n_u8(4);
                    float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
                    for(;k+16<=I3_GROUP;k+=16){
                        uint32_t wd; memcpy(&wd, lo+(k>>2), 4);            /* 4 bytes = 16 low-plane values */
                        uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                        uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),              vand_u8(vshr_n_u8(by,2),m2v));
                        uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v), vshr_n_u8(by,6));
                        uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
                        uint8x16_t lov=vcombine_u8(vreinterpret_u8_u16(zz.val[0]), vreinterpret_u8_u16(zz.val[1]));
                        uint8x16_t hv=vcombine_u8(vdup_n_u8(hi[k>>3]), vdup_n_u8(hi[(k>>3)+1]));
                        uint8x16_t hb=vandq_u8(vtstq_u8(hv,bitm), fourq);   /* 4 where high bit set */
                        int8x16_t wq=vsubq_s8(vreinterpretq_s8_u8(vaddq_u8(lov,hb)), b4q); /* [-4,3] in order */
                        int16x8_t w0=vmovl_s8(vget_low_s8(wq)), w1=vmovl_s8(vget_high_s8(wq));
                        ac0=vfmaq_f32(ac0, vld1q_f32(xs+base+k),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                        ac1=vfmaq_f32(ac1, vld1q_f32(xs+base+k+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                        ac0=vfmaq_f32(ac0, vld1q_f32(xs+base+k+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                        ac1=vfmaq_f32(ac1, vld1q_f32(xs+base+k+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
                    }
                    a=vaddvq_f32(vaddq_f32(ac0,ac1));
                }
#elif defined(__AVX2__)
                if(n==I3_GROUP){
                    const __m128i m2=_mm_set1_epi8(0x03);
                    const __m128i bsel=_mm_set_epi8(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
                    const __m128i bitm=_mm_set_epi8((char)128,64,32,16,8,4,2,1,(char)128,64,32,16,8,4,2,1);
                    const __m128i four8=_mm_set1_epi8(4);
                    const __m256i b4=_mm256_set1_epi32(4);
                    __m256 ac0=_mm256_setzero_ps(), ac1=_mm256_setzero_ps();
                    for(;k+16<=I3_GROUP;k+=16){
                        __m128i by=_mm_cvtsi32_si128(*(const int*)(lo+(k>>2)));  /* 4 bytes = 16 low-plane values */
                        __m128i p0=_mm_and_si128(by,m2), p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                        __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                        __m128i l01=_mm_unpacklo_epi8(p0,p1), h23=_mm_unpacklo_epi8(p2,p3);
                        __m128i lov=_mm_unpacklo_epi16(l01,h23);                  /* 16 vals in order */
                        __m128i hv=_mm_shuffle_epi8(_mm_cvtsi32_si128(hi[k>>3]|(hi[(k>>3)+1]<<8)),bsel);
                        __m128i hb=_mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hv,bitm),bitm),four8); /* 4 where high bit set */
                        __m128i u=_mm_add_epi8(lov,hb);                           /* [0,7] stored v+4 */
                        __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(u),b4));
                        __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(u,8)),b4));
                        ac0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+base+k),  w0,ac0);
                        ac1=_mm256_fmadd_ps(_mm256_loadu_ps(xs+base+k+8),w1,ac1);
                    }
                    a=hsum256(_mm256_add_ps(ac0,ac1));
                }
#endif
                for(;k<n;k++){
                    unsigned u=((lo[k>>2]>>((k&3)*2))&3) | (((hi[k>>3]>>(k&7))&1)<<2);
                    a += xs[base+k]*(float)((int)u-4);
                }
                acc += a*srow[g];
            }
            y[(int64_t)s*O+o]=acc;
        }
    }
}

/* ---- fmt=8: native FP8-e4m3 passthrough (Z.ai GLM-5.2-FP8 read path) ------
 * PUBLIC ordinal. This format was minted fmt=6 during original development,
 * then re-tagged fmt=100 (PRIVATE ORDINAL BLOCK, see colibri.c's QT struct
 * comment) after #465 (E8/IQ3, above) claimed ordinal 6 upstream out from
 * under it; graduated to fmt=7 when the maintainer assigned that ordinal on
 * #524; renumbered a second time to fmt=8 after #705 merged into dev claiming
 * fmt=7 for MXFP4 (Kimi K3 Vulkan tier, backend_vulkan.c) while this PR was
 * still open. Both renumbers were pure retags, per the private-block
 * convention's own "find-and-replace, zero on-disk impact" promise: nothing
 * on disk carries the ordinal. See qt_resolve_fmt in colibri.c ("THE DESIGN
 * LANDMINE") for the disambiguation this needs against BOTH fmt=1 (int8) and
 * fmt=6 (E8/IQ3).
 *
 * fmt=8 weight bytes are O*I raw e4m3 bytes -- byte-identical to int8 (fmt=1).
 * What makes this a DIFFERENT format is the scale: one f32 per 128x128 BLOCK of
 * the [O,I] weight matrix (shape [ceil(O/128),ceil(I/128)]), not one f32 per
 * output row. Dequant is w[o,i] = e4m3_decode(byte) * scale[o/128, i/128]
 * (MULTIPLY, not divide) -- mirrors tools/convert_fp8_to_int4.py's dequant()
 * exactly, which is the authoritative reference for how Z.ai's checkpoints read
 * on the source side. This f32 encoding is a declared PROPERTY of the format,
 * not the only one it can carry -- DeepSeek-V4 ships this identical weight
 * geometry with UE8M0 (1-byte, power-of-two) block scales instead; qt_resolve_fmt
 * recognizes that byte signature and refuses it BY NAME rather than
 * misreading it as f32 (UE8M0 decode itself is not implemented in this build).
 *
 * 256-entry decode LUT, cross-checked byte-for-byte against PyTorch's
 * torch.float8_e4m3fn (the exact dtype safetensors reports for these shards):
 * sign(1) exp(4,bias=7) mant(3), subnormal at exp==0, and the OCP E4M3-FN
 * convention that exp==0xF is NOT reserved for infinity -- only mant==0x7 at
 * exp==0xF is NaN (max finite is exp=0xF,mant=0x6 -> 448). A static compile-time
 * table (not a lazily-initialized one) is used deliberately: matmul_fp8 below
 * runs inside #pragma omp parallel for, and a lazy-init global would race under
 * concurrent first use.
 *
 * NaN POLICY (decided, documented per the build spec): a NaN byte (0x7F/0xFF)
 * decodes to a real IEEE NaN and is left to propagate through the dot product,
 * exactly like any other source of a NaN weight (fmt=0/f32 tensors are never
 * scrubbed either). The engine already has a dedicated, TESTED safety net for
 * NaN reaching the sampler (argmax_v/dist_build, see tests/test_logit_nan.c:
 * "degrade + diagnose, never silently corrupt") -- fmt=8 relies on that existing
 * downstream net rather than adding a second, redundant weight-level scrub. */
static const float E4M3_LUT[256] = {
    0x0.0p+0f,0x1.0000000000000p-9f,0x1.0000000000000p-8f,0x1.8000000000000p-8f,0x1.0000000000000p-7f,0x1.4000000000000p-7f,0x1.8000000000000p-7f,0x1.c000000000000p-7f,
    0x1.0000000000000p-6f,0x1.2000000000000p-6f,0x1.4000000000000p-6f,0x1.6000000000000p-6f,0x1.8000000000000p-6f,0x1.a000000000000p-6f,0x1.c000000000000p-6f,0x1.e000000000000p-6f,
    0x1.0000000000000p-5f,0x1.2000000000000p-5f,0x1.4000000000000p-5f,0x1.6000000000000p-5f,0x1.8000000000000p-5f,0x1.a000000000000p-5f,0x1.c000000000000p-5f,0x1.e000000000000p-5f,
    0x1.0000000000000p-4f,0x1.2000000000000p-4f,0x1.4000000000000p-4f,0x1.6000000000000p-4f,0x1.8000000000000p-4f,0x1.a000000000000p-4f,0x1.c000000000000p-4f,0x1.e000000000000p-4f,
    0x1.0000000000000p-3f,0x1.2000000000000p-3f,0x1.4000000000000p-3f,0x1.6000000000000p-3f,0x1.8000000000000p-3f,0x1.a000000000000p-3f,0x1.c000000000000p-3f,0x1.e000000000000p-3f,
    0x1.0000000000000p-2f,0x1.2000000000000p-2f,0x1.4000000000000p-2f,0x1.6000000000000p-2f,0x1.8000000000000p-2f,0x1.a000000000000p-2f,0x1.c000000000000p-2f,0x1.e000000000000p-2f,
    0x1.0000000000000p-1f,0x1.2000000000000p-1f,0x1.4000000000000p-1f,0x1.6000000000000p-1f,0x1.8000000000000p-1f,0x1.a000000000000p-1f,0x1.c000000000000p-1f,0x1.e000000000000p-1f,
    0x1.0000000000000p+0f,0x1.2000000000000p+0f,0x1.4000000000000p+0f,0x1.6000000000000p+0f,0x1.8000000000000p+0f,0x1.a000000000000p+0f,0x1.c000000000000p+0f,0x1.e000000000000p+0f,
    0x1.0000000000000p+1f,0x1.2000000000000p+1f,0x1.4000000000000p+1f,0x1.6000000000000p+1f,0x1.8000000000000p+1f,0x1.a000000000000p+1f,0x1.c000000000000p+1f,0x1.e000000000000p+1f,
    0x1.0000000000000p+2f,0x1.2000000000000p+2f,0x1.4000000000000p+2f,0x1.6000000000000p+2f,0x1.8000000000000p+2f,0x1.a000000000000p+2f,0x1.c000000000000p+2f,0x1.e000000000000p+2f,
    0x1.0000000000000p+3f,0x1.2000000000000p+3f,0x1.4000000000000p+3f,0x1.6000000000000p+3f,0x1.8000000000000p+3f,0x1.a000000000000p+3f,0x1.c000000000000p+3f,0x1.e000000000000p+3f,
    0x1.0000000000000p+4f,0x1.2000000000000p+4f,0x1.4000000000000p+4f,0x1.6000000000000p+4f,0x1.8000000000000p+4f,0x1.a000000000000p+4f,0x1.c000000000000p+4f,0x1.e000000000000p+4f,
    0x1.0000000000000p+5f,0x1.2000000000000p+5f,0x1.4000000000000p+5f,0x1.6000000000000p+5f,0x1.8000000000000p+5f,0x1.a000000000000p+5f,0x1.c000000000000p+5f,0x1.e000000000000p+5f,
    0x1.0000000000000p+6f,0x1.2000000000000p+6f,0x1.4000000000000p+6f,0x1.6000000000000p+6f,0x1.8000000000000p+6f,0x1.a000000000000p+6f,0x1.c000000000000p+6f,0x1.e000000000000p+6f,
    0x1.0000000000000p+7f,0x1.2000000000000p+7f,0x1.4000000000000p+7f,0x1.6000000000000p+7f,0x1.8000000000000p+7f,0x1.a000000000000p+7f,0x1.c000000000000p+7f,0x1.e000000000000p+7f,
    0x1.0000000000000p+8f,0x1.2000000000000p+8f,0x1.4000000000000p+8f,0x1.6000000000000p+8f,0x1.8000000000000p+8f,0x1.a000000000000p+8f,0x1.c000000000000p+8f,       NAN,
     -0x0.0p+0f,-0x1.0000000000000p-9f,-0x1.0000000000000p-8f,-0x1.8000000000000p-8f,-0x1.0000000000000p-7f,-0x1.4000000000000p-7f,-0x1.8000000000000p-7f,-0x1.c000000000000p-7f,
    -0x1.0000000000000p-6f,-0x1.2000000000000p-6f,-0x1.4000000000000p-6f,-0x1.6000000000000p-6f,-0x1.8000000000000p-6f,-0x1.a000000000000p-6f,-0x1.c000000000000p-6f,-0x1.e000000000000p-6f,
    -0x1.0000000000000p-5f,-0x1.2000000000000p-5f,-0x1.4000000000000p-5f,-0x1.6000000000000p-5f,-0x1.8000000000000p-5f,-0x1.a000000000000p-5f,-0x1.c000000000000p-5f,-0x1.e000000000000p-5f,
    -0x1.0000000000000p-4f,-0x1.2000000000000p-4f,-0x1.4000000000000p-4f,-0x1.6000000000000p-4f,-0x1.8000000000000p-4f,-0x1.a000000000000p-4f,-0x1.c000000000000p-4f,-0x1.e000000000000p-4f,
    -0x1.0000000000000p-3f,-0x1.2000000000000p-3f,-0x1.4000000000000p-3f,-0x1.6000000000000p-3f,-0x1.8000000000000p-3f,-0x1.a000000000000p-3f,-0x1.c000000000000p-3f,-0x1.e000000000000p-3f,
    -0x1.0000000000000p-2f,-0x1.2000000000000p-2f,-0x1.4000000000000p-2f,-0x1.6000000000000p-2f,-0x1.8000000000000p-2f,-0x1.a000000000000p-2f,-0x1.c000000000000p-2f,-0x1.e000000000000p-2f,
    -0x1.0000000000000p-1f,-0x1.2000000000000p-1f,-0x1.4000000000000p-1f,-0x1.6000000000000p-1f,-0x1.8000000000000p-1f,-0x1.a000000000000p-1f,-0x1.c000000000000p-1f,-0x1.e000000000000p-1f,
    -0x1.0000000000000p+0f,-0x1.2000000000000p+0f,-0x1.4000000000000p+0f,-0x1.6000000000000p+0f,-0x1.8000000000000p+0f,-0x1.a000000000000p+0f,-0x1.c000000000000p+0f,-0x1.e000000000000p+0f,
    -0x1.0000000000000p+1f,-0x1.2000000000000p+1f,-0x1.4000000000000p+1f,-0x1.6000000000000p+1f,-0x1.8000000000000p+1f,-0x1.a000000000000p+1f,-0x1.c000000000000p+1f,-0x1.e000000000000p+1f,
    -0x1.0000000000000p+2f,-0x1.2000000000000p+2f,-0x1.4000000000000p+2f,-0x1.6000000000000p+2f,-0x1.8000000000000p+2f,-0x1.a000000000000p+2f,-0x1.c000000000000p+2f,-0x1.e000000000000p+2f,
    -0x1.0000000000000p+3f,-0x1.2000000000000p+3f,-0x1.4000000000000p+3f,-0x1.6000000000000p+3f,-0x1.8000000000000p+3f,-0x1.a000000000000p+3f,-0x1.c000000000000p+3f,-0x1.e000000000000p+3f,
    -0x1.0000000000000p+4f,-0x1.2000000000000p+4f,-0x1.4000000000000p+4f,-0x1.6000000000000p+4f,-0x1.8000000000000p+4f,-0x1.a000000000000p+4f,-0x1.c000000000000p+4f,-0x1.e000000000000p+4f,
    -0x1.0000000000000p+5f,-0x1.2000000000000p+5f,-0x1.4000000000000p+5f,-0x1.6000000000000p+5f,-0x1.8000000000000p+5f,-0x1.a000000000000p+5f,-0x1.c000000000000p+5f,-0x1.e000000000000p+5f,
    -0x1.0000000000000p+6f,-0x1.2000000000000p+6f,-0x1.4000000000000p+6f,-0x1.6000000000000p+6f,-0x1.8000000000000p+6f,-0x1.a000000000000p+6f,-0x1.c000000000000p+6f,-0x1.e000000000000p+6f,
    -0x1.0000000000000p+7f,-0x1.2000000000000p+7f,-0x1.4000000000000p+7f,-0x1.6000000000000p+7f,-0x1.8000000000000p+7f,-0x1.a000000000000p+7f,-0x1.c000000000000p+7f,-0x1.e000000000000p+7f,
    -0x1.0000000000000p+8f,-0x1.2000000000000p+8f,-0x1.4000000000000p+8f,-0x1.6000000000000p+8f,-0x1.8000000000000p+8f,-0x1.a000000000000p+8f,-0x1.c000000000000p+8f,       NAN,
};
static inline float e4m3_decode(uint8_t b){ return E4M3_LUT[b]; }

#ifdef __AVX2__
/* Eight e4m3 bytes to eight floats without touching a lookup table.
 *
 * A float wants sign, exponent, mantissa in that order, and e4m3 already has
 * them in that order: place the byte's seven low bits at bit 20 and its sign at
 * bit 31, and the result is the right number with the wrong exponent bias,
 * 127 where e4m3 means 7. One multiply by 2^120 fixes it, and it fixes the
 * subnormals too, because scaling a subnormal by a power of two into the normal
 * range is exact.
 *
 * Verified byte for byte against quant.h's E4M3_LUT: 254 of the 256 encodings
 * come out identical. The two that do not are 0x7F and 0xFF, e4m3's NaNs, which
 * this would turn into +-480 -- a corrupt weight quietly becoming a plausible
 * one. They are blended back, because a NaN that propagates is how you find out.
 */
static inline __m256 e4m3_decode8(const uint8_t *p) {
    __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
    __m256i low = _mm256_and_si256(b, _mm256_set1_epi32(0x7F));
    __m256i bits = _mm256_or_si256(
        _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24),
        _mm256_slli_epi32(low, 20));
    __m256 value = _mm256_mul_ps(_mm256_castsi256_ps(bits), _mm256_set1_ps(0x1p120f));
    __m256i nan = _mm256_cmpeq_epi32(low, _mm256_set1_epi32(0x7F));
    return _mm256_blendv_ps(value, _mm256_set1_ps(NAN), _mm256_castsi256_ps(nan));
}

/* Eight bf16 into eight floats: bf16 IS the top half of a float, so the whole
 * conversion is a shift. Exact for every encoding, NaNs and infinities
 * included, which is why there is no table to disagree with. */
static inline __m256 bf16_decode8(const uint16_t *p) {
    __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
}
#endif

#define FP8_BLOCK 128
static inline int64_t fp8_nblk(int n){ return ((int64_t)n + FP8_BLOCK - 1) / FP8_BLOCK; }

/* y[S,O] = x[S,I] @ W^T, W raw e4m3 bytes (byte-identical layout to fmt=1) +
 * per-128x128-BLOCK f32 scale [ceil(O/128),ceil(I/128)]. Scalar reference path
 * (no SIMD in v1 -- BW-bound like the Metal kernel, and this format's hot path
 * is the GPU one; a vectorized CPU kernel is future work if measured needed).
 * Mirrors matmul_i3's double-accumulate-across-groups / float-within-group
 * convention so cross-block cancellation doesn't cost precision unfairly. */
/* Two kernels, selected by compiler, because the faster one is only
   BIT-EXACT under clang.

   The four-accumulator form is algebraically identical to the one-accumulator
   loop - same operands, same column order - so any difference comes purely
   from the compiler contracting multiply-adds differently in the four-chain
   shape, which drifts results by ~1 ulp.  That would break the byte-exact
   contract tests/test_qwen38_native_weights.c pins against its independent
   reference, and a one-ulp logit can flip an argmax in the token-exact gates.

   Under GCC that contraction cannot be controlled from source.  Measured on
   gcc 13.4, clean build per cell, against this kernel's own test:
     - #pragma GCC optimize ("fp-contract=off")     ignored (no-op)
     - #pragma GCC optimize ("-ffp-contract=off")   ignored (no-op)
     - __attribute__((optimize("-ffp-contract=off"))) ignored (no-op)
     - #pragma STDC FP_CONTRACT OFF                 unimplemented: GCC warns
                                                    "ignoring '#pragma STDC
                                                    FP_CONTRACT'"
   Only the command-line -ffp-contract=off works, and that is a global numerics
   decision this kernel has no business making for the whole project.  With no
   guard, the four-row form is exact on -march=znver3 and -march=x86-64-v3 but
   NOT on -march=haswell, so "it passed on my machine" is not evidence here.

   clang contracts the reference and this kernel identically, so the four-row
   form is bit-exact there: verified on arm64 (binary byte-identical to the
   one-row build) and on x86 clang at -march=haswell, znver3 and x86-64-v3.
   So clang gets the fast kernel and GCC keeps upstream's, which is exact on
   every arch tested. */
#if defined(__clang__)
static void matmul_fp8(float *y, const float *x, const uint8_t *q8, const float *bscale,
                       int S, int I, int O){
    int64_t nblkI = fp8_nblk(I);
    /* Four output rows per pass, each with its own accumulator.  Every row's
       addition sequence is identical to the one-row form - same operands, same
       column order - so results are bit-identical; only the interleaving of four
       independent dependency chains differs.  The gain is latency hiding plus one
       load of xs[i] feeding four rows.  A tail of fewer than four rows clamps the
       spare indices onto the last valid row and the stores are guarded, which
       avoids a separate remainder loop. */
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o+=4){
        int o1=o+1<O ? o+1 : o, o2=o+2<O ? o+2 : o, o3=o+3<O ? o+3 : o;
        const uint8_t *w0=q8+(int64_t)o*I,  *w1=q8+(int64_t)o1*I;
        const uint8_t *w2=q8+(int64_t)o2*I, *w3=q8+(int64_t)o3*I;
        const float *scl0=bscale+((int64_t)o /FP8_BLOCK)*nblkI;
        const float *scl1=bscale+((int64_t)o1/FP8_BLOCK)*nblkI;
        const float *scl2=bscale+((int64_t)o2/FP8_BLOCK)*nblkI;
        const float *scl3=bscale+((int64_t)o3/FP8_BLOCK)*nblkI;
        for(int s=0;s<S;s++){
            const float *xs = x + (int64_t)s*I;
            double a0=0,a1=0,a2=0,a3=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                float acc0=0,acc1=0,acc2=0,acc3=0;
                for(int i=base;i<base+blen;i++){
                    float xv=xs[i];
                    acc0 += e4m3_decode(w0[i])*xv;
                    acc1 += e4m3_decode(w1[i])*xv;
                    acc2 += e4m3_decode(w2[i])*xv;
                    acc3 += e4m3_decode(w3[i])*xv;
                }
                a0 += (double)acc0*scl0[bi];
                a1 += (double)acc1*scl1[bi];
                a2 += (double)acc2*scl2[bi];
                a3 += (double)acc3*scl3[bi];
            }
            y[(int64_t)s*O+o]=(float)a0;
            if(o1!=o) y[(int64_t)s*O+o1]=(float)a1;
            if(o2!=o) y[(int64_t)s*O+o2]=(float)a2;
            if(o3!=o) y[(int64_t)s*O+o3]=(float)a3;
        }
    }
}
#else
/* GCC and everything else: upstream's one-row kernel, unchanged.  Exact on
   every -march tested; see the note above for why it is not simply replaced. */
static void matmul_fp8(float *y, const float *x, const uint8_t *q8, const float *bscale,
                       int S, int I, int O){
    int64_t nblkI = fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w = q8 + (int64_t)o*I;
        int64_t blkO = o / FP8_BLOCK;
        const float *scl = bscale + blkO*nblkI;
        for(int s=0;s<S;s++){
            const float *xs = x + (int64_t)s*I;
            double a=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                float sc=scl[bi]; float acc=0;
                for(int i=base;i<base+blen;i++) acc += e4m3_decode(w[i])*xs[i];
                a += (double)acc*sc;
            }
            y[(int64_t)s*O+o]=(float)a;
        }
    }
}
#endif


/* f32 planare (fmt=2): stesso ordine di accumulazione di matmul_i4 (sequenza
 * 0..63 in blocchi di 8 lane, stessa lane per elemento) -> bit-identico. */
static void matmul_i4p(float *y, const float *x, const uint8_t *q4, const float *scale,
                       int S, int I, int O){
#ifdef __AVX2__
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            const __m256i m4=_mm256_set1_epi32(0xF), b8=_mm256_set1_epi32(8);
            __m256 acc=_mm256_setzero_ps();
            int i=0;
            for(;i+64<=I;i+=64){
                __m128i lo0=_mm_loadu_si128((const __m128i*)(w+(i>>1)));      /* byte 0..15 */
                __m128i lo1=_mm_loadu_si128((const __m128i*)(w+(i>>1)+16));   /* byte 16..31 */
                /* elementi 0..31: nibble bassi in ordine */
                __m256i e0=_mm256_and_si256(_mm256_cvtepu8_epi32(lo0),m4);
                __m256i e1=_mm256_and_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(lo0,8)),m4);
                __m256i e2=_mm256_and_si256(_mm256_cvtepu8_epi32(lo1),m4);
                __m256i e3=_mm256_and_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(lo1,8)),m4);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(e0,b8)),acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8),
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(e1,b8)),acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+16),
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(e2,b8)),acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+24),
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(e3,b8)),acc);
                /* elementi 32..63: nibble alti in ordine */
                __m256i h0=_mm256_srli_epi32(_mm256_cvtepu8_epi32(lo0),4);
                __m256i h1=_mm256_srli_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(lo0,8)),4);
                __m256i h2=_mm256_srli_epi32(_mm256_cvtepu8_epi32(lo1),4);
                __m256i h3=_mm256_srli_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(lo1,8)),4);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+32),
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(h0,m4),b8)),acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+40),
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(h1,m4),b8)),acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+48),
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(h2,m4),b8)),acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+56),
                                    _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(h3,m4),b8)),acc);
            }
            /* coda: PRIMA il loop vettoriale a 16 del gemello (stesso acc,
             * stessa sequenza per-lane {8m+L}: nessuna cucitura), POI la coda
             * scalare identica. EN: tail = the pair kernel's 16-wide loop into
             * the same acc (seamless per-lane order), then its scalar tail. */
            {
                const __m128i m4p=_mm_set1_epi8(0x0F); const __m256i b8p=_mm256_set1_epi32(8);
                for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4p), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4p);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8p));
                    __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8p));
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            }
            float a=hsum256(acc);
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += xs[i]*(float)lo; }
            y[(int64_t)s*O+o]=a*sc;
        }
    }
#else
    /* scalare: blocchi planari + coda a coppie */
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0; int i=0;
            for(;i+64<=I;i+=64){
                const uint8_t *blk=w+(i>>1);
                for(int k=0;k<32;k++){
                    a+=xs[i+k]*(float)((int)(blk[k]&0xF)-8);
                    a+=xs[i+k+32]*(float)((int)(blk[k]>>4)-8);
                }
            }
            for(;i<I;i+=2){
                uint8_t byte=w[i>>1];
                a+=xs[i]*(float)((int)(byte&0xF)-8);
                if(i+1<I) a+=xs[i+1]*(float)((int)(byte>>4)-8);
            }
            y[(int64_t)s*O+o]=a*sc;
        }
    }
#endif
}
/* ============================ fine blocco K1 ============================== */

/* ---- per-thread quantization scratch -------------------------------------- */
typedef struct { int8_t *xq; size_t xq_cap; float *sx; size_t sx_cap; } QScratch;
static _Thread_local QScratch g_qscratch;
static void quant_scratch(size_t xn, size_t sn, int8_t **xq, float **sx){
    if(xn>g_qscratch.xq_cap){
        int8_t *p=(int8_t*)realloc(g_qscratch.xq,xn);
        if(!p){ fprintf(stderr,"OOM quant scratch\n"); exit(1); }
        g_qscratch.xq=p; g_qscratch.xq_cap=xn;
    }
    if(sn>g_qscratch.sx_cap){
        float *p=(float*)realloc(g_qscratch.sx,sn*sizeof(float));
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
/* quantize w[O,I] f32 -> int3-g64 (fmt=5): per 64-input group, symmetric absmax
 * (qmax=3, clamp [-4,3], stored v+4), 16B low plane + 8B high plane, ONE f32 scale
 * per group. Same math as tools/quant_ablation.py `_quant_last_dim(bits=3, group=64)`
 * (#132), here with real bit packing. */
static void pack_int3_g64(const float *w, uint8_t *q3, float *scale, int O, int I){
    int64_t ng=i3_groups(I), rb=i3_rowbytes(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const float *wr=w+(int64_t)o*I;
        uint8_t *qr=q3+(int64_t)o*rb;
        float   *sr=scale+(int64_t)o*ng;
        for(int64_t g=0; g<ng; g++){
            int base=(int)(g*I3_GROUP), n = I-base < I3_GROUP ? I-base : I3_GROUP;
            float amax=0;
            for(int k=0;k<n;k++){ float a=fabsf(wr[base+k]); if(a>amax)amax=a; }
            float s=amax/3.f; if(s<1e-8f)s=1e-8f; sr[g]=s;
            uint8_t *lo=qr+g*I3_GBYTES, *hi=lo+16;
            memset(lo,0,I3_GBYTES);
            for(int k=0;k<n;k++){
                int v=(int)lrintf(wr[base+k]/s); if(v>3)v=3; if(v<-4)v=-4;
                unsigned u=(unsigned)(v+4);                     /* 0..7 */
                lo[k>>2] |= (uint8_t)((u&3)<<((k&3)*2));
                hi[k>>3] |= (uint8_t)(((u>>2)&1)<<(k&7));
            }
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

/* ---- fmt=6: E8/IQ3 lattice container (#452) --------------------------------
 * 98 bytes per 256 weights = 3.0625 bits/weight. Per super-block:
 *   [ 0..63]  uint8  grid index per 4-dim magnitude block
 *   [64..95]  uint32 x8 - four 7-bit sign words + 4-bit sub-scale, per 32 weights
 *   [96..97]  fp16   super-scale
 * value = d * (0.5 + code) * 0.5 * grid[idx][j] * sign, with the 8th sign of every
 * eight derived from odd parity (that is what buys the 8th bit back).
 * Byte layout and arithmetic mirror tools/iq3_pack.py exactly - that codec is the
 * oracle this kernel is tested against.
 *
 * Decode strategy: expand one 32-weight sub-block into a stack buffer, then FMA it
 * against the activations. Per-weight table lookups would dominate; per-sub-block
 * expansion keeps the grid rows (16 bytes each) hot in L1 and lets the compiler
 * vectorize the multiply-accumulate. */
#define E8_QK      256                  /* weights per super-block */
#define E8_SUB     32                   /* weights per sign/scale word */
#define E8_BBYTES  98                   /* bytes per super-block */
static inline int64_t e8_blocks(int I){ return ((int64_t)I + E8_QK - 1) / E8_QK; }
static inline int64_t e8_rowbytes(int I){ return e8_blocks(I) * E8_BBYTES; }

/* The published 256x4 magnitude grid, stored doubled (4,12,..,62 mean 2,6,..,31).
 * From ggml-common.h (MIT); tools/iq3xxs_grid.json is the same table for the
 * Python codec, and tests/test_e8_kernel.c checks the two agree through the
 * fixture. Header-local like the rest of quant.h - the engine is a single
 * translation unit and the tests include this header directly. */
static const uint8_t e8_grid[256][4] = {
    {  4,  4,  4,  4},
    { 20,  4,  4,  4},
    { 36,  4,  4,  4},
    { 12, 12,  4,  4},
    { 28, 12,  4,  4},
    { 62, 12,  4,  4},
    {  4, 20,  4,  4},
    { 20, 20,  4,  4},
    { 12, 28,  4,  4},
    { 20, 36,  4,  4},
    { 28, 62,  4,  4},
    { 44, 62,  4,  4},
    { 12,  4, 12,  4},
    { 28,  4, 12,  4},
    {  4, 12, 12,  4},
    { 20, 12, 12,  4},
    { 12, 20, 12,  4},
    { 44, 20, 12,  4},
    {  4, 28, 12,  4},
    { 20, 28, 12,  4},
    { 12, 36, 12,  4},
    { 36, 44, 12,  4},
    {  4, 62, 12,  4},
    {  4,  4, 20,  4},
    { 20,  4, 20,  4},
    { 36,  4, 20,  4},
    { 12, 12, 20,  4},
    {  4, 20, 20,  4},
    { 20, 20, 20,  4},
    { 12, 28, 20,  4},
    { 28, 28, 20,  4},
    { 62, 28, 20,  4},
    { 12, 44, 20,  4},
    { 62, 44, 20,  4},
    { 44, 62, 20,  4},
    { 12,  4, 28,  4},
    { 62,  4, 28,  4},
    {  4, 12, 28,  4},
    { 20, 12, 28,  4},
    { 44, 20, 28,  4},
    {  4, 62, 28,  4},
    { 28, 12, 36,  4},
    { 62, 28, 36,  4},
    { 36, 36, 36,  4},
    { 62, 44, 36,  4},
    { 28, 62, 36,  4},
    { 44, 62, 36,  4},
    { 12,  4, 44,  4},
    { 62,  4, 44,  4},
    { 20, 28, 44,  4},
    { 20, 44, 44,  4},
    { 44, 28, 52,  4},
    { 36, 52, 52,  4},
    {  4, 12, 62,  4},
    { 36, 12, 62,  4},
    { 52, 12, 62,  4},
    { 28, 36, 62,  4},
    { 12, 52, 62,  4},
    { 12,  4,  4, 12},
    { 28,  4,  4, 12},
    {  4, 12,  4, 12},
    { 20, 12,  4, 12},
    { 12, 20,  4, 12},
    { 28, 20,  4, 12},
    {  4, 28,  4, 12},
    { 20, 28,  4, 12},
    { 36, 28,  4, 12},
    { 62, 36,  4, 12},
    {  4, 44,  4, 12},
    {  4,  4, 12, 12},
    { 20,  4, 12, 12},
    { 12, 12, 12, 12},
    {  4, 20, 12, 12},
    { 20, 20, 12, 12},
    { 12,  4, 20, 12},
    { 28,  4, 20, 12},
    {  4, 12, 20, 12},
    { 20, 12, 20, 12},
    { 12, 20, 20, 12},
    {  4, 28, 20, 12},
    { 20, 62, 20, 12},
    {  4,  4, 28, 12},
    { 20,  4, 28, 12},
    {  4, 20, 28, 12},
    { 12, 28, 28, 12},
    { 52, 36, 28, 12},
    { 52, 52, 28, 12},
    { 12,  4, 36, 12},
    { 44,  4, 36, 12},
    {  4, 44, 36, 12},
    {  4, 20, 44, 12},
    { 36, 20, 44, 12},
    { 52, 36, 44, 12},
    { 12, 62, 44, 12},
    { 44,  4, 52, 12},
    { 20, 20, 62, 12},
    {  4, 36, 62, 12},
    {  4,  4,  4, 20},
    { 20,  4,  4, 20},
    { 12, 12,  4, 20},
    { 28, 12,  4, 20},
    {  4, 20,  4, 20},
    { 20, 20,  4, 20},
    { 52, 20,  4, 20},
    { 12, 28,  4, 20},
    { 20, 36,  4, 20},
    { 12,  4, 12, 20},
    { 28,  4, 12, 20},
    { 44,  4, 12, 20},
    {  4, 12, 12, 20},
    { 20, 12, 12, 20},
    { 12, 20, 12, 20},
    {  4, 28, 12, 20},
    { 28, 52, 12, 20},
    { 62, 52, 12, 20},
    {  4, 62, 12, 20},
    {  4,  4, 20, 20},
    { 20,  4, 20, 20},
    { 12, 12, 20, 20},
    { 62, 12, 20, 20},
    {  4, 20, 20, 20},
    { 20, 20, 20, 20},
    { 62, 28, 20, 20},
    {  4, 36, 20, 20},
    { 44, 44, 20, 20},
    { 12,  4, 28, 20},
    {  4, 12, 28, 20},
    { 36, 12, 28, 20},
    {  4, 62, 28, 20},
    { 36, 62, 28, 20},
    { 44, 28, 36, 20},
    { 28, 44, 36, 20},
    { 28,  4, 44, 20},
    { 62, 20, 44, 20},
    { 12, 36, 44, 20},
    { 36, 62, 44, 20},
    { 12,  4, 62, 20},
    { 28,  4, 62, 20},
    { 52, 12, 62, 20},
    { 44, 36, 62, 20},
    { 12,  4,  4, 28},
    {  4, 12,  4, 28},
    { 20, 12,  4, 28},
    { 12, 20,  4, 28},
    { 28, 20,  4, 28},
    {  4, 44,  4, 28},
    { 44, 52,  4, 28},
    { 20, 62,  4, 28},
    {  4,  4, 12, 28},
    { 20,  4, 12, 28},
    {  4, 20, 12, 28},
    { 12, 28, 12, 28},
    { 36, 36, 12, 28},
    { 52, 36, 12, 28},
    { 12,  4, 20, 28},
    { 28,  4, 20, 28},
    {  4, 12, 20, 28},
    { 44, 20, 20, 28},
    { 20, 44, 20, 28},
    { 20, 62, 20, 28},
    { 12, 12, 28, 28},
    { 28, 28, 28, 28},
    {  4, 28, 36, 28},
    { 62, 36, 36, 28},
    { 20, 62, 36, 28},
    {  4,  4, 44, 28},
    { 52,  4, 44, 28},
    { 20, 20, 44, 28},
    { 44, 44, 44, 28},
    { 36, 12, 52, 28},
    { 52, 28, 52, 28},
    { 28, 52, 52, 28},
    { 28, 28, 62, 28},
    {  4, 52, 62, 28},
    { 36,  4,  4, 36},
    { 62, 12,  4, 36},
    { 44, 28,  4, 36},
    { 62, 28,  4, 36},
    { 28, 44,  4, 36},
    { 62, 44,  4, 36},
    { 36, 62, 12, 36},
    {  4, 20, 20, 36},
    { 62, 28, 20, 36},
    {  4, 36, 20, 36},
    {  4, 52, 20, 36},
    { 52, 52, 20, 36},
    { 62,  4, 28, 36},
    { 44, 36, 28, 36},
    { 36,  4, 36, 36},
    { 12, 44, 36, 36},
    { 36, 52, 36, 36},
    { 44, 20, 44, 36},
    { 28, 36, 44, 36},
    {  4, 62, 44, 36},
    { 44,  4, 62, 36},
    {  4, 12, 62, 36},
    { 20, 12, 62, 36},
    {  4, 28, 62, 36},
    { 20, 12,  4, 44},
    { 12, 36,  4, 44},
    {  4, 62,  4, 44},
    {  4,  4, 12, 44},
    { 52,  4, 12, 44},
    { 52, 20, 12, 44},
    { 44, 44, 12, 44},
    { 36, 12, 20, 44},
    { 20, 28, 20, 44},
    { 20, 62, 20, 44},
    { 20,  4, 28, 44},
    { 28, 44, 28, 44},
    {  4, 12, 36, 44},
    { 28, 20, 36, 44},
    { 62, 20, 36, 44},
    { 20, 62, 36, 44},
    { 20,  4, 44, 44},
    { 12, 28, 44, 44},
    {  4, 44, 52, 44},
    { 36, 20, 62, 44},
    { 20, 36, 62, 44},
    { 36, 20,  4, 52},
    { 36, 36,  4, 52},
    { 52, 36,  4, 52},
    { 36, 52,  4, 52},
    { 12, 20, 12, 52},
    { 12, 52, 12, 52},
    { 62, 12, 20, 52},
    { 36, 52, 20, 52},
    {  4, 28, 28, 52},
    { 52, 28, 28, 52},
    { 36, 36, 36, 52},
    { 44,  4, 44, 52},
    { 20, 44, 44, 52},
    { 28, 28, 52, 52},
    { 28,  4, 62, 52},
    { 12, 20, 62, 52},
    { 28,  4,  4, 62},
    { 44,  4,  4, 62},
    { 62,  4,  4, 62},
    {  4, 12,  4, 62},
    { 20, 28,  4, 62},
    { 20, 44,  4, 62},
    { 52, 20, 12, 62},
    {  4, 36, 12, 62},
    { 20, 12, 20, 62},
    { 44, 36, 20, 62},
    { 20, 44, 20, 62},
    {  4,  4, 28, 62},
    { 44, 12, 28, 62},
    { 28, 28, 28, 62},
    {  4, 52, 28, 62},
    { 12, 20, 36, 62},
    { 12, 36, 36, 62},
    {  4,  4, 44, 62},
    { 20,  4, 44, 62},
    { 36, 20, 44, 62},
    {  4, 28, 52, 62}
};

static inline float e8_fp16_to_f32(uint16_t h){
    uint32_t sign=(uint32_t)(h>>15)<<31, exp=(h>>10)&0x1F, man=h&0x3FF, bits;
    if(!exp)      bits = man ? (sign | ((127-15+1)<<23) | (man<<13)) : sign;  /* subnormal->approx */
    else if(exp==0x1F) bits = sign | 0x7F800000u | (man<<13);
    else          bits = sign | ((exp+112)<<23) | (man<<13);
    float f; memcpy(&f,&bits,4); return f;
}

/* Expand one 32-weight sub-block. `out` must hold 32 floats. */
static inline void e8_expand_sub(const uint8_t *blk, int ib, float d, float *out){
    uint32_t word; memcpy(&word, blk + E8_QK/4 + ib*4, 4);
    float db = d * (0.5f + (float)((word>>28)&0xF)) * 0.5f;
    const uint8_t *idx = blk + ib*8;
    for(int l=0;l<4;l++){
        uint32_t seven=(word>>(7*l))&0x7F;
        const uint8_t *g0=e8_grid[idx[l*2+0]], *g1=e8_grid[idx[l*2+1]];
        int par=0;
        for(int j=0;j<8;j++){
            int neg = j<7 ? (int)((seven>>j)&1) : 0;
            if(j<7) par^=neg; else neg=par;            /* odd parity closes the block */
            float mag = (j<4 ? (float)g0[j] : (float)g1[j-4]) * 0.5f;
            out[l*8+j] = neg ? -mag*db : mag*db;
        }
    }
}

/* Fast Walsh-Hadamard transform with the per-tensor sign flip: y = Q^T x for
 * Q = D*H/sqrt(n). fmt=6 stores W@Q, so activations must be transformed before
 * the matmul (#452). Placement is the engine's job and it matters: all routed
 * experts of a layer share one input row, so ONE transform per (layer,
 * projection group) costs ~1.4 ms/token on GLM dims, while doing it per expert
 * costs ~11 ms. n must be a power of two >= the real dim; the tail is zero-pad.
 * Self-inverse up to the sign flip, so the same routine serves both directions. */
static inline void e8_fwht(float *a, int n, const uint8_t *signbits){
    if(signbits) for(int i=0;i<n;i++) if(signbits[i>>3]>>(i&7)&1) a[i]=-a[i];
    for(int len=1;len<n;len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){ float u=a[j],v=a[j+len]; a[j]=u+v; a[j+len]=u-v; }
    float s=1.0f/sqrtf((float)n);
    for(int i=0;i<n;i++) a[i]*=s;
}
static inline int e8_pow2_ceil(int n){ int p=1; while(p<n) p<<=1; return p; }

/* Rotation sign bits, regenerated — not stored. xorshift64* seeded 417+n, one
 * bit per element; tools/iq3_pack.py signs() draws the identical stream, so the
 * container carries no rotation data and the two sides cannot drift apart
 * without the oracle fixture catching it (#452). */
static inline void e8_signs(uint8_t *bits, int n){
    uint64_t s=417u+(uint64_t)n;
    for(int i=0;i<(n+7)/8;i++){
        s^=s>>12; s^=s<<25; s^=s>>27;
        bits[i]=(uint8_t)((s*2685821657736338717ULL)>>56);
    }
}
/* Apply the fmt=6 activation rotation Q^T in place, row-major [nr,dim].
 * Non-power-of-two dims tile block-diagonally; each block is the largest power
 * of two dividing the remainder (= its lowest set bit): 6144 -> 2048+4096,
 * 1536 -> 512+1024. Blocks over 32768 halve until they fit the sign buffer.
 * The converter rotates weight rows with this exact routine (W@Q and Q^T x are
 * the same transform: Q is symmetric-orthogonal up to the sign flip). */
static inline void e8_rot_rows(float *rows, int nr, int dim){
    int off=0;
    while(off<dim){
        int rem=dim-off, b=rem&(-rem);
        while(b>32768) b>>=1;
        uint8_t bits[32768/8];
        e8_signs(bits,b);
        for(int r=0;r<nr;r++) e8_fwht(rows+(int64_t)r*dim+off,b,bits);
        off+=b;
    }
}

/* ---- MXFP4 (OCP microscaling FP4) ----------------------------------------
 * The native layout of compressed-tensors "mxfp4-pack-quantized" checkpoints
 * (Kimi K3 routed experts are QAT in this format — pass-through, never
 * re-encoded):
 *   packed [O, I/2]  u8 — e2m1 nibbles, LOW nibble = even column, bit3 = sign,
 *                         bits 0..2 index {0,.5,1,1.5,2,3,4,6}
 *   scales [O, I/32] u8 — ue8m0 exponent per 32-column group, w = v * 2^(s-127)
 * = 4.25 bits/weight all-in. The exponent is decoded with the bit trick
 * (s<<23 as a float): exact for s in [1,254]; s=0 (2^-127, denormal) decodes
 * to +0 and s=255 to +inf on BOTH the scalar and SIMD paths, so the two stay
 * bit-identical — real checkpoints never contain either. */
static const float mx4_lut[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                                  -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static inline float mx4_scale(uint8_t s){
    union { uint32_t u; float f; } b; b.u = (uint32_t)s << 23; return b.f;
}
static void matmul_mxfp4(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s,
                         int S, int I, int O){
    int rb=(I+1)/2, ng=(I+31)/32;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const uint8_t *scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
#ifdef __AVX2__
            if(I%32==0){
                /* doubled e2m1 values are exact int8 -> one pshufb decodes a
                 * nibble vector; the 0.5f un-doubling rides the group scale */
                const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
                const __m128i m4=_mm_set1_epi8(0x0F);
                __m256 acc=_mm256_setzero_ps();
                for(int g=0;g<ng;g++){
                    __m128i by=_mm_loadu_si128((const __m128i*)(w+g*16));
                    __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i n0=_mm_shuffle_epi8(lut2,_mm_unpacklo_epi8(lo,hi));  /* cols g*32+0..15  */
                    __m128i n1=_mm_shuffle_epi8(lut2,_mm_unpackhi_epi8(lo,hi));  /* cols g*32+16..31 */
                    const float *xg=xs+g*32;
                    __m256 ga=_mm256_mul_ps(_mm256_loadu_ps(xg),
                                            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n0)));
                    ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+8),
                                       _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n0,8))),ga);
                    ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+16),
                                       _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n1)),ga);
                    ga=_mm256_fmadd_ps(_mm256_loadu_ps(xg+24),
                                       _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n1,8))),ga);
                    acc=_mm256_fmadd_ps(ga,_mm256_set1_ps(mx4_scale(scl[g])*0.5f),acc);
                }
                y[(int64_t)s*O+o]=hsum256(acc);
                continue;
            }
#endif
            for(int g=0;g<ng;g++){
                int base=g*32, glen=32; if(base+glen>I) glen=I-base;
                float sc=mx4_scale(scl[g]), ga=0;
                for(int i=base;i<base+glen;i+=2){
                    uint8_t byte=w[i>>1];
                    ga+=xs[i]*mx4_lut[byte&0xF];
                    if(i+1<base+glen) ga+=xs[i+1]*mx4_lut[byte>>4];
                }
                a+=ga*sc;
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* IDOT variant of matmul_mxfp4: per-32-group int8 activation quantization +
 * integer dots. The doubled e2m1 values are exact int8 (same LUT as the float
 * path), so a group reduces to maddubs(|w|, sign(x,w)) like dot_i4i8 — no
 * per-weight int->float conversion. Group-exact scale folding:
 *     y = sum_g idot_g * (2^(e8-127) * 0.5) * xscale_g
 * Activation-quant noise (~0.4%/group) rides on top of the e2m1 grid; gate
 * with K3_IDOT=0 in the K3 engine for exact-float A/B. */
/* Quantizzazione int8 per-gruppo-di-32 delle attivazioni per l'IDOT mxfp4.
 * Estratta da matmul_mxfp4_i8 (ordine dei loop IDENTICO -> stessi byte) così i
 * chiamanti possono sollevarla a livello layer: in kimi_k3 gate e up consumano
 * lo stesso z, e ogni expert del layer consuma le stesse righe — prima ogni
 * chiamata riquantizzava da capo, con un malloc/free a giro (issue del PR #1071
 * per colibri.c; qui è la stessa malattia in forma mxfp4).
 * EN: per-32-group int8 activation quantization for the mxfp4 IDOT kernel.
 * Extracted with the loop order unchanged (bit-identical) so callers can hoist
 * it to layer level: gate/up share one z, and every expert of the layer
 * consumes the same rows. */
static void mxfp4_qx(const float *x, int S, int I, int8_t *xq, float *xsc){
    int ng=I/32;
    for(int s=0;s<S;s++)
        for(int g=0;g<ng;g++){
            const float *xg=x+(int64_t)s*I+g*32;
            float am=0; for(int i=0;i<32;i++){ float a=fabsf(xg[i]); if(a>am)am=a; }
            float sc=am/127.f; if(sc<1e-20f)sc=1e-20f;
            xsc[(int64_t)s*ng+g]=sc; float inv=1.f/sc;
            int8_t *qg=xq+(int64_t)s*I+g*32;
            for(int i=0;i<32;i++){
                int v=(int)lrintf(xg[i]*inv);
                if(v>127)v=127; if(v<-127)v=-127; qg[i]=(int8_t)v;
            }
        }
}

/* Compute su attivazioni GIA' quantizzate (xq/xsc da mxfp4_qx). Contratto: I%32==0.
 * EN: compute on pre-quantized activations. Contract: I%32==0. */
static void matmul_mxfp4_i8_pre(float *y, const int8_t *xq, const float *xsc,
                                const uint8_t *q4, const uint8_t *e8s,
                                int S, int I, int O){
    int rb=I/2, ng=I/32;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const uint8_t *scl=e8s+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const int8_t *xr=xq+(int64_t)s*I;
            const float *xsr=xsc+(int64_t)s*ng;
#ifdef __AVX2__
            {
                const __m128i lut2=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
                const __m128i m4=_mm_set1_epi8(0x0F);
                const __m256i ones=_mm256_set1_epi16(1);
                __m256 acc=_mm256_setzero_ps();
                for(int g=0;g<ng;g++){
                    __m128i by=_mm_loadu_si128((const __m128i*)(w+g*16));
                    __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i n0=_mm_shuffle_epi8(lut2,_mm_unpacklo_epi8(lo,hi));
                    __m128i n1=_mm_shuffle_epi8(lut2,_mm_unpackhi_epi8(lo,hi));
                    __m256i wv=_mm256_set_m128i(n1,n0);        /* signed doubled e2m1 */
                    __m256i xv=_mm256_loadu_si256((const __m256i*)(xr+g*32));
                    /* |w| <= 12, |x| <= 127: pair sums <= 3048, no int16 overflow */
                    __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),
                                                   _mm256_sign_epi8(xv,wv));
                    acc=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p,ones)),
                                        _mm256_set1_ps(mx4_scale(scl[g])*0.5f*xsr[g]),acc);
                }
                y[(int64_t)s*O+o]=hsum256(acc);
                continue;
            }
#endif
            {
                static const int8_t l2[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
                float a=0;
                for(int g=0;g<ng;g++){
                    const int8_t *qg=xr+g*32;
                    int32_t gi=0;
                    for(int i=0;i<32;i+=2){
                        uint8_t byte=w[(g*32+i)>>1];
                        gi+=(int32_t)l2[byte&0xF]*qg[i]+(int32_t)l2[byte>>4]*qg[i+1];
                    }
                    a+=(float)gi*(mx4_scale(scl[g])*0.5f*xsr[g]);
                }
                y[(int64_t)s*O+o]=a;
            }
        }
    }
}

/* Wrapper storico: quantizza in scratch _Thread_local (niente più malloc/free
 * per chiamata) e delega al kernel _pre. Stessi valori, stessi ordini.
 * EN: historical entry point — quantize into the growable thread-local scratch
 * (no more per-call malloc/free) and delegate to the _pre kernel. */
static void matmul_mxfp4_i8(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s,
                            int S, int I, int O){
    if(I%32){ matmul_mxfp4(y,x,q4,e8s,S,I,O); return; }
    int ng=I/32;
    int8_t *xq; float *xsc;
    quant_scratch((size_t)S*I,(size_t)S*ng,&xq,&xsc);
    mxfp4_qx(x,S,I,xq,xsc);
    matmul_mxfp4_i8_pre(y,xq,xsc,q4,e8s,S,I,O);
}

#ifdef __ARM_NEON
/* Chemin NEON pour fmt=6 (E8/IQ3).
 *
 * Le noyau amont n'a qu'un chemin AVX2 et un repli scalaire : sur aarch64
 * (GB10 / DGX Spark, Apple silicon) tout le deballage retombe donc sur du C
 * scalaire, mesure en amont a ~92 % du temps de decodage. Cette version
 * transpose la structure du chemin AVX2, qui suit deja la forme du format :
 *
 *   - une "lane" = 8 poids = deux lignes de grille de 4 octets + 8 signes,
 *     soit exactement DEUX registres NEON de 4 flottants ;
 *   - les 8 octets de grille s'elargissent par vmovl_u8 puis vmovl_u16 ;
 *   - les 7 bits de signe stockes plus le 8e derive de la parite deviennent
 *     des masques par comparaison contre le vecteur de selection, appliques
 *     par XOR du bit de signe flottant -- aucun branchement, ce qui est
 *     precisement ce qui rendait la version scalaire couteuse ;
 *   - le 0.5 de la convention demi-unite de la grille est replie dans
 *     l'echelle du sous-bloc.
 *
 * Deux accumulateurs evitent une chaine de dependance unique sur les 32 FMA
 * d'un super-bloc, et une seule reduction horizontale par 256 poids remplace
 * une par 32.
 */
static void matmul_e8_neon(float *y, const float *x, const uint8_t *q, const float *unused,
                           int S, int I, int O){
    (void)unused;                                  /* les echelles vivent dans les blocs */
    int64_t nb=e8_blocks(I), rb=e8_rowbytes(I);
    const uint32x4_t sel_lo=(uint32x4_t){1,2,4,8};
    const uint32x4_t sel_hi=(uint32x4_t){16,32,64,128};
    const uint32x4_t sgn=vdupq_n_u32(0x80000000u);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *wrow=q+(int64_t)o*rb;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            float acc=0;
            for(int64_t b=0;b<nb;b++){
                const uint8_t *blk=wrow+b*E8_BBYTES;
                uint16_t dh; memcpy(&dh, blk+96, 2);
                float d=e8_fp16_to_f32(dh);
                int base=(int)(b*E8_QK);
                if(base+E8_QK<=I){
                    float32x4_t ac0=vdupq_n_f32(0.0f), ac1=vdupq_n_f32(0.0f);
                    for(int ib=0; ib<E8_QK/E8_SUB; ib++){
                        uint32_t word; memcpy(&word, blk+E8_QK/4+ib*4, 4);
                        float db=d*(0.5f+(float)((word>>28)&0xF))*0.5f;
                        const float32x4_t vdb=vdupq_n_f32(0.5f*db);
                        const uint8_t *ix=blk+ib*8;
                        int off=base+ib*E8_SUB;
                        for(int l=0;l<4;l++){
                            uint32_t sv=(word>>(7*l))&0x7Fu;
                            uint32_t s8=sv|((uint32_t)__builtin_parity(sv)<<7); /* parite impaire ferme la lane */
                            uint8_t g[8];
                            memcpy(g,   e8_grid[ix[l*2+0]], 4);
                            memcpy(g+4, e8_grid[ix[l*2+1]], 4);
                            uint16x8_t w16=vmovl_u8(vld1_u8(g));
                            float32x4_t v0=vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16))),  vdb);
                            float32x4_t v1=vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16))), vdb);
                            uint32x4_t sd=vdupq_n_u32(s8);
                            uint32x4_t m0=vceqq_u32(vandq_u32(sd,sel_lo),sel_lo);
                            uint32x4_t m1=vceqq_u32(vandq_u32(sd,sel_hi),sel_hi);
                            v0=vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v0),vandq_u32(m0,sgn)));
                            v1=vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v1),vandq_u32(m1,sgn)));
                            ac0=vfmaq_f32(ac0,v0,vld1q_f32(xs+off+l*8));
                            ac1=vfmaq_f32(ac1,v1,vld1q_f32(xs+off+l*8+4));
                        }
                    }
                    acc+=vaddvq_f32(vaddq_f32(ac0,ac1));
                    continue;
                }
                /* Queue non alignee : on garde le deballage de reference. */
                for(int ib=0; ib<E8_QK/E8_SUB; ib++){
                    int off=base+ib*E8_SUB;
                    if(off>=I) break;
                    float w[E8_SUB];
                    e8_expand_sub(blk, ib, d, w);
                    int n = I-off < E8_SUB ? I-off : E8_SUB;
                    float a=0;
                    for(int k=0;k<n;k++) a += xs[off+k]*w[k];
                    acc+=a;
                }
            }
            y[(int64_t)s*O+o]=acc;
        }
    }
}
#endif /* __ARM_NEON */

static void matmul_e8(float *y, const float *x, const uint8_t *q, const float *unused,
                      int S, int I, int O){
    (void)unused;                                  /* scales live inside the blocks */
#ifdef __ARM_NEON
    /* aarch64 : le corps ci-dessous na quun chemin AVX2 et un repli scalaire,
     * donc tout retombait sur du C non vectorise. Mesure ici (expert GLM
     * 6144x2048) : 3,6x plus rapide a S=1, et 2,2x PLUS PRECIS contre une
     * reference en double precision. */
    matmul_e8_neon(y, x, q, unused, S, I, O);
    return;
#endif
    int64_t nb=e8_blocks(I), rb=e8_rowbytes(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *wrow=q+(int64_t)o*rb;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            float acc=0;
            for(int64_t b=0;b<nb;b++){
                const uint8_t *blk=wrow+b*E8_BBYTES;
                uint16_t dh; memcpy(&dh, blk+96, 2);
                float d=e8_fp16_to_f32(dh);
                int base=(int)(b*E8_QK);
#ifdef __AVX2__
                /* One 8-weight lane is exactly one AVX2 register, which is what the
                 * format's own shape suggests: a lane is two 4-dim grid rows (8
                 * contiguous codebook bytes) plus 8 signs. So instead of expanding a
                 * sub-block into a stack buffer and re-reading it, each lane is
                 * decoded straight into a register and FMA'd against x:
                 *   - the 8 grid bytes widen with one vpmovzxbd,
                 *   - the sign byte (7 stored bits + the parity-derived 8th) expands
                 *     to 8 lane masks with an AND/CMPEQ against the bit-select vector
                 *     and is applied as an XOR of the float sign bit — no branches,
                 *     which is what made the scalar expansion expensive,
                 *   - 0.5 (the grid's half-unit convention) folds into the sub-scale.
                 * Two accumulators over the 32 FMAs of a super-block keep this off a
                 * single dependency chain, and one horizontal add per 256 weights
                 * replaces one per 32. */
                if(base+E8_QK<=I){
                    const __m256i sel=_mm256_setr_epi32(1,2,4,8,16,32,64,128);
                    const __m256i sgn=_mm256_set1_epi32((int)0x80000000u);
                    __m256 ac[2]={_mm256_setzero_ps(),_mm256_setzero_ps()};
                    for(int ib=0; ib<E8_QK/E8_SUB; ib++){
                        uint32_t word; memcpy(&word, blk+E8_QK/4+ib*4, 4);
                        float db=d*(0.5f+(float)((word>>28)&0xF))*0.5f;
                        __m256 vdb=_mm256_set1_ps(0.5f*db);
                        const uint8_t *ix=blk+ib*8;
                        int off=base+ib*E8_SUB;
                        for(int l=0;l<4;l++){
                            uint32_t sv=(word>>(7*l))&0x7Fu;
                            uint32_t s8=sv|((uint32_t)__builtin_parity(sv)<<7); /* odd parity closes the lane */
                            uint32_t g0,g1;
                            memcpy(&g0,e8_grid[ix[l*2+0]],4);
                            memcpy(&g1,e8_grid[ix[l*2+1]],4);
                            __m128i by=_mm_cvtsi64_si128((long long)((uint64_t)g0|((uint64_t)g1<<32)));
                            __m256 v=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(by)),vdb);
                            __m256i m=_mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32((int)s8),sel),sel);
                            v=_mm256_xor_ps(v,_mm256_castsi256_ps(_mm256_and_si256(m,sgn)));
                            ac[l&1]=_mm256_fmadd_ps(v,_mm256_loadu_ps(xs+off+l*8),ac[l&1]);
                        }
                    }
                    __m256 t=_mm256_add_ps(ac[0],ac[1]);
                    __m128 h=_mm_add_ps(_mm256_castps256_ps128(t),_mm256_extractf128_ps(t,1));
                    h=_mm_add_ps(h,_mm_movehl_ps(h,h));
                    h=_mm_add_ss(h,_mm_shuffle_ps(h,h,1));
                    acc+=_mm_cvtss_f32(h);
                    continue;
                }
#endif
                for(int ib=0; ib<E8_QK/E8_SUB; ib++){
                    int off=base+ib*E8_SUB;
                    if(off>=I) break;
                    float w[E8_SUB];
                    e8_expand_sub(blk, ib, d, w);
                    int n = I-off < E8_SUB ? I-off : E8_SUB;
                    float a=0;
                    for(int k=0;k<n;k++) a += xs[off+k]*w[k];
                    acc+=a;
                }
            }
            y[(int64_t)s*O+o]=acc;
        }
    }
}

#endif /* COLI_QUANT_H */
