/* idot.h -- integer dot-product kernels shared by every engine.
 *
 * The activation is quantized to int8 once (one scale per row, amax/127,
 * qrow_i8) and the weights, int8 rows or int4 planar blocks with per-row or
 * per-group scales, are multiplied with integer instructions: maddubs on
 * AVX2, vpdpbusd on AVX-VNNI and AVX-512 VNNI, sdot/smmla on NEON, and the
 * AMX tile kernel where the CPU has it. Exact int32 sums, scaled once per
 * output, so every ISA path is bit-identical to the scalar reference.
 *
 * Lived inside quant.h; moved here so an engine that has its own dense
 * kernels (qwen36.c uses qgemv.h/gsgemv.h, whose matmul_q would collide
 * with quant.h's) can still take the integer path for its dense trunk.
 * quant.h includes this file at the top, so its other consumers see the
 * same declarations in the same place as before. */
#ifndef COLIBRI_IDOT_H
#define COLIBRI_IDOT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
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
    /* 4 accumulatori indipendenti (64 byte/iter): un solo acc incatena i vpdpbusd
     * (latenza-bound ~5c). Somme intere associative -> bit-identico. Stessa struttura
     * dei 4 accumulatori del ramo NEON piu' sotto.
     * EN: four independent accumulators break the serial vpdpbusd->acc chain; integer
     * adds are associative, so the result is bit-identical (mirrors the NEON path). */
    __m128i a0=_mm_setzero_si128(),a1=_mm_setzero_si128(),a2=_mm_setzero_si128(),a3=_mm_setzero_si128();
    for(;i+64<=I;i+=64){
        __m128i w0=_mm_loadu_si128((const __m128i*)(w+i)),    x0=_mm_loadu_si128((const __m128i*)(x+i));
        __m128i w1=_mm_loadu_si128((const __m128i*)(w+i+16)), x1=_mm_loadu_si128((const __m128i*)(x+i+16));
        __m128i w2=_mm_loadu_si128((const __m128i*)(w+i+32)), x2=_mm_loadu_si128((const __m128i*)(x+i+32));
        __m128i w3=_mm_loadu_si128((const __m128i*)(w+i+48)), x3=_mm_loadu_si128((const __m128i*)(x+i+48));
        a0=_mm_dpbusd_epi32(a0,_mm_abs_epi8(w0),_mm_sign_epi8(x0,w0));
        a1=_mm_dpbusd_epi32(a1,_mm_abs_epi8(w1),_mm_sign_epi8(x1,w1));
        a2=_mm_dpbusd_epi32(a2,_mm_abs_epi8(w2),_mm_sign_epi8(x2,w2));
        a3=_mm_dpbusd_epi32(a3,_mm_abs_epi8(w3),_mm_sign_epi8(x3,w3));
    }
    __m128i acc=_mm_add_epi32(_mm_add_epi32(a0,a1),_mm_add_epi32(a2,a3));
    for(;i+16<=I;i+=16){
        __m128i wv=_mm_loadu_si128((const __m128i*)(w+i));
        __m128i xv=_mm_loadu_si128((const __m128i*)(x+i));
        acc=_mm_dpbusd_epi32(acc,_mm_abs_epi8(wv),_mm_sign_epi8(xv,wv));
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
    /* 4 accumulatori indipendenti (64 elementi = 32 byte packed/iter): un solo acc
     * incatena i vpdpbusd (latenza-bound ~5c). Somme intere associative -> bit-identico.
     * Stessa struttura dei 4 accumulatori del ramo NEON piu' sotto.
     * EN: four independent accumulators break the serial vpdpbusd->acc chain; integer
     * adds are associative, so the result is bit-identical (mirrors the NEON path). */
    const __m128i m4=_mm_set1_epi8(0x0F); const __m128i b8=_mm_set1_epi8(8);
    __m128i a0=_mm_setzero_si128(),a1=_mm_setzero_si128(),a2=_mm_setzero_si128(),a3=_mm_setzero_si128();
    for(;i+64<=I;i+=64){
        __m128i by0=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));       /* elem i..i+31  */
        __m128i by1=_mm_loadu_si128((const __m128i*)(w4+(i>>1)+16));    /* elem i+32..i+63 */
        __m128i lo0=_mm_and_si128(by0,m4), hi0=_mm_and_si128(_mm_srli_epi16(by0,4),m4);
        __m128i lo1=_mm_and_si128(by1,m4), hi1=_mm_and_si128(_mm_srli_epi16(by1,4),m4);
        __m128i w0=_mm_sub_epi8(_mm_unpacklo_epi8(lo0,hi0),b8), w1=_mm_sub_epi8(_mm_unpackhi_epi8(lo0,hi0),b8);
        __m128i w2=_mm_sub_epi8(_mm_unpacklo_epi8(lo1,hi1),b8), w3=_mm_sub_epi8(_mm_unpackhi_epi8(lo1,hi1),b8);
        __m128i x0=_mm_loadu_si128((const __m128i*)(x+i)),    x1=_mm_loadu_si128((const __m128i*)(x+i+16));
        __m128i x2=_mm_loadu_si128((const __m128i*)(x+i+32)), x3=_mm_loadu_si128((const __m128i*)(x+i+48));
        a0=_mm_dpbusd_epi32(a0,_mm_abs_epi8(w0),_mm_sign_epi8(x0,w0));
        a1=_mm_dpbusd_epi32(a1,_mm_abs_epi8(w1),_mm_sign_epi8(x1,w1));
        a2=_mm_dpbusd_epi32(a2,_mm_abs_epi8(w2),_mm_sign_epi8(x2,w2));
        a3=_mm_dpbusd_epi32(a3,_mm_abs_epi8(w3),_mm_sign_epi8(x3,w3));
    }
    __m128i acc=_mm_add_epi32(_mm_add_epi32(a0,a1),_mm_add_epi32(a2,a3));
    for(;i+32<=I;i+=32){   /* 32-nibble remainder: 2 dpbusd, same unpack */
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i w0=_mm_sub_epi8(_mm_unpacklo_epi8(lo,hi),b8), w1=_mm_sub_epi8(_mm_unpackhi_epi8(lo,hi),b8);
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


/* ================= K1: layout int4 A PIANI (fmt 2/4, opzionale) ==============
 * Layout classico ("a coppie"): byte k = elementi (2k, 2k+1). Ogni kernel paga
 * unpacklo/unpackhi per riordinare i nibble, e l'IDOT paga sub+abs+sign per
 * forzare l'operando con segno dentro vpdpbusd (che e' nativamente u8 x s8).
 *
 * Layout A PIANI, per blocco di 64 elementi consecutivi (32 byte):
 *   byte k del blocco = (elem k + 8) | ((elem k+32 + 8) << 4)      k = 0..31
 * `and 0x0F` produce gli elementi 0..31 GIA' in ordine; `srli 4 + and` produce
 * 32..63 in ordine: l'unpack sparisce. La coda (I mod 64) resta a coppie.
 *
 * L'algebra dell'IDOT: i nibble memorizzati sono u = v+8 (unsigned 0..15), e
 *   dot(v, x) = dot(u, x) - 8 * sum(x)
 * dove sum(x) e' UN int32 per riga di attivazione, condiviso da ogni riga di
 * output. vpdpbusd consuma (u8, s8) direttamente: via sub+abs+sign (3 op
 * vettoriali per dpbusd). Misurato su questa famiglia: 1.87x L3-resident,
 * 25.3 GB/s DRAM (al tetto), 0 differenze di bit.
 * NB: allargare a 256 bit SENZA cambiare layout e' stato misurato 0.79-0.81x
 * (il fixup di lane costa piu' della larghezza): il layout e' la precondizione
 * della larghezza, non un'ottimizzazione indipendente.
 *
 * I kernel f32 planari preservano ESATTAMENTE l'ordine di accumulazione dei
 * gemelli a coppie (stesse lane, stessa sequenza 0..63): bit-identici anche
 * loro. EN: plane-nibble int4 layout. Low nibble of block byte k = element k,
 * high = element k+32; the unpack disappears, and the stored-unsigned nibbles
 * feed vpdpbusd natively via dot(v,x) = dot(u,x) - 8*sum(x). The f32 planar
 * kernels keep the pair kernels' exact accumulation order: bit-identical. */

/* riordina in place una riga (o un tensore [O,rb] riga per riga) da coppie a
 * piani; la coda I%64 resta a coppie. EN: in-place pair->planar repack. */
static void planarize_i4_row(uint8_t *row, int I){
    uint8_t tmp[32];
    int nb=I/64;
    for(int b=0;b<nb;b++){
        uint8_t *blk=row+b*32;
        /* pair: byte j ha (elem 2j, elem 2j+1). planar: byte k = (elem k, elem k+32) */
        for(int k=0;k<32;k++){
            int src_lo=k, src_hi=k+32;                       /* elementi voluti */
            uint8_t nib_lo=(blk[src_lo>>1]>>((src_lo&1)*4))&0xF;
            uint8_t nib_hi=(blk[src_hi>>1]>>((src_hi&1)*4))&0xF;
            tmp[k]=(uint8_t)(nib_lo|(nib_hi<<4));
        }
        memcpy(blk,tmp,32);
    }
}
static void planarize_i4(uint8_t *q4, int O, int I){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++) planarize_i4_row(q4+(int64_t)o*rb, I);
}

/* dot unsigned-nibble planare * int8: ritorna dot(u, x) SENZA la correzione
 * -8*sum(x) (la applica il chiamante, una volta per riga di output).
 * Coda I%64: nibble a coppie letti come unsigned, stessa identita'. */
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
#if defined(__AVXVNNI__) && !defined(__AVX512VNNI__)
#define coli_dpbusd256 _mm256_dpbusd_avx_epi32
#else
#define coli_dpbusd256 _mm256_dpbusd_epi32
#endif
#endif
static inline int32_t dot_i4p_u(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(coli_dpbusd256)
    const __m256i m4=_mm256_set1_epi8(0x0F);
    __m256i a0=_mm256_setzero_si256(),a1=_mm256_setzero_si256();
    __m256i a2=_mm256_setzero_si256(),a3=_mm256_setzero_si256();
    for(;i+128<=I;i+=128){
        __m256i b0=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i b1=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)+32));
        a0=coli_dpbusd256(a0,_mm256_and_si256(b0,m4),
                          _mm256_loadu_si256((const __m256i*)(x+i)));
        a1=coli_dpbusd256(a1,_mm256_and_si256(_mm256_srli_epi16(b0,4),m4),
                          _mm256_loadu_si256((const __m256i*)(x+i+32)));
        a2=coli_dpbusd256(a2,_mm256_and_si256(b1,m4),
                          _mm256_loadu_si256((const __m256i*)(x+i+64)));
        a3=coli_dpbusd256(a3,_mm256_and_si256(_mm256_srli_epi16(b1,4),m4),
                          _mm256_loadu_si256((const __m256i*)(x+i+96)));
    }
    __m256i acc=_mm256_add_epi32(_mm256_add_epi32(a0,a1),_mm256_add_epi32(a2,a3));
    for(;i+64<=I;i+=64){
        __m256i b0=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        acc=coli_dpbusd256(acc,_mm256_and_si256(b0,m4),
                           _mm256_loadu_si256((const __m256i*)(x+i)));
        acc=coli_dpbusd256(acc,_mm256_and_si256(_mm256_srli_epi16(b0,4),m4),
                           _mm256_loadu_si256((const __m256i*)(x+i+32)));
    }
    sum=hsum256_i32(acc);
#elif defined(__AVX2__)
    const __m256i m4=_mm256_set1_epi8(0x0F);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(;i+64<=I;i+=64){
        __m256i b0=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        /* maddubs(u8, s8): u<=15, |x|<=127 -> coppia <= 3810, int16 sicuro */
        __m256i p0=_mm256_maddubs_epi16(_mm256_and_si256(b0,m4),
                                        _mm256_loadu_si256((const __m256i*)(x+i)));
        __m256i p1=_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(b0,4),m4),
                                        _mm256_loadu_si256((const __m256i*)(x+i+32)));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p0,ones));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p1,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+64<=I;i+=64){
        uint8x16_t b0=vld1q_u8(w4+(i>>1)), b1=vld1q_u8(w4+(i>>1)+16);
        int8x16_t lo0=vreinterpretq_s8_u8(vandq_u8(b0,vdupq_n_u8(0x0F)));
        int8x16_t lo1=vreinterpretq_s8_u8(vandq_u8(b1,vdupq_n_u8(0x0F)));
        int8x16_t hi0=vreinterpretq_s8_u8(vshrq_n_u8(b0,4));
        int8x16_t hi1=vreinterpretq_s8_u8(vshrq_n_u8(b1,4));
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,lo0,vld1q_s8(x+i));
        acc=vdotq_s32(acc,lo1,vld1q_s8(x+i+16));
        acc=vdotq_s32(acc,hi0,vld1q_s8(x+i+32));
        acc=vdotq_s32(acc,hi1,vld1q_s8(x+i+48));
#else
        int8x16_t xs0=vld1q_s8(x+i), xs1=vld1q_s8(x+i+16);
        int8x16_t xs2=vld1q_s8(x+i+32), xs3=vld1q_s8(x+i+48);
        int16x8_t m;
        m=vmull_s8(vget_low_s8(lo0),vget_low_s8(xs0));  acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_high_s8(lo0),vget_high_s8(xs0)); acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_low_s8(lo1),vget_low_s8(xs1));  acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_high_s8(lo1),vget_high_s8(xs1)); acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_low_s8(hi0),vget_low_s8(xs2));  acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_high_s8(hi0),vget_high_s8(xs2)); acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_low_s8(hi1),vget_low_s8(xs3));  acc=vpadalq_s16(acc,m);
        m=vmull_s8(vget_high_s8(hi1),vget_high_s8(xs3)); acc=vpadalq_s16(acc,m);
#endif
    }
    sum=vaddvq_s32(acc);
#endif
    for(;i+64<=I;i+=64){          /* fallback scalare sui blocchi planari */
        const uint8_t *blk=w4+(i>>1);
        for(int k=0;k<32;k++){
            sum+=(int32_t)(blk[k]&0xF)*x[i+k];
            sum+=(int32_t)(blk[k]>>4)*x[i+k+32];
        }
    }
    for(;i<I;i+=2){               /* coda a coppie, unsigned (u=v+8 memorizzato) */
        uint8_t byte=w4[i>>1];
        sum+=(int32_t)(byte&0xF)*x[i];
        if(i+1<I) sum+=(int32_t)(byte>>4)*x[i+1];
    }
    return sum;
}

/* ---- K1b (OPT-IN, IDOT_GS=1): IDOT planare A GRUPPI (fmt=4, gs%64==0) -----
 * Con gs=64 il gruppo di scala COINCIDE col blocco-piano da 64 elementi: il
 * dot unsigned del blocco (2 dpbusd) -> int32 di gruppo, meno 8*somma(x) del
 * gruppo, per la scala f32 del gruppo. Attivazioni int8 (stessa famiglia
 * qrow_i8 del resto dell'IDOT): NON bit-identico al kernel f32 a gruppi --
 * per questo e' dietro flag, in attesa dell'ablazione. xsg = somme int32
 * per (riga, gruppo), calcolate dal chiamante in una passata esatta.
 * EN: grouped planar IDOT, opt-in. With gs=64 the scale group IS the plane
 * block; per-group unsigned dot minus 8*group-sum, times the group scale.
 * int8 activations: not bit-identical to the f32 grouped kernel, hence the
 * flag until the ablation blesses a default.
 *
 * Three execution shapes below, one contract: per (row, output) the float
 * accumulation is the SAME sequence of fmaf((float)group_int, scale[g], a)
 * in ascending g, and every group_int is an exact int32 — so the per-row
 * path, the 1x4 row tile, and the AMX tile are bit-identical to each other
 * and to the pure-C reference on every ISA. */

/* one planar 64-element block, unpacked once and shared across a row tile:
 * lo nibbles = elements base..base+31 in order, hi = base+32..base+63. */
#if defined(coli_dpbusd256)
static inline void i4p_blk256(const uint8_t *blk, __m256i *lo, __m256i *hi){
    const __m256i m4=_mm256_set1_epi8(0x0F);
    __m256i b=_mm256_loadu_si256((const __m256i*)blk);
    *lo=_mm256_and_si256(b,m4);
    *hi=_mm256_and_si256(_mm256_srli_epi16(b,4),m4);
}
#endif
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
/* whole block in one zmm: lanes 0..31 = lo elements, 32..63 = hi — matches a
 * straight 64-byte load of the activation block, so ONE dpbusd per block. */
static inline __m512i i4p_blk512(const uint8_t *blk){
    const __m256i m4=_mm256_set1_epi8(0x0F);
    __m256i b=_mm256_loadu_si256((const __m256i*)blk);
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(b,m4)),
                              _mm256_and_si256(_mm256_srli_epi16(b,4),m4),1);
}
#endif

/* ---- K1c: AMX int8 tile kernel (Sapphire Rapids+, opt-in via the same
 * IDOT_GS=1 family gate; AMX=0 kills it, AMX_S_MIN sets the row threshold).
 * With gs a multiple of 64, K=64 tile-multiplies cover a scale group exactly:
 * B tile = 16 output rows' int4 block unpacked once to SIGNED int8 (v-8, so
 * tdpbssd returns d - 8*sum(x_group) directly — the same int32 the vector
 * path computes as d_unsigned - 8*xsg), A tile = up to 16 activation rows.
 * The unpack cost is paid once per (output tile, group) and amortized over
 * every activation row — the multi-row reuse the pair-layout kernels lack.
 * Linux-only arming: tile data needs an ARCH_REQ_XCOMP_PERM handshake. */
#if defined(__AMX_INT8__) && defined(__AMX_TILE__) && defined(__AVX512F__)
#define COLI_HAVE_AMX_I4P 1
#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
static int coli_amx_state=-1;
static int coli_amx_s_min=8;
static int coli_amx_ok(void){
    if(coli_amx_state<0){
        const char *e=getenv("AMX");
        const char *sm=getenv("AMX_S_MIN"); if(sm&&atoi(sm)>0) coli_amx_s_min=atoi(sm);
#if defined(__linux__)
        /* ARCH_REQ_XCOMP_PERM(XFEATURE_XTILEDATA): kernel >= 5.16 grants tile
         * state per-process; without it the first tile op SIGILLs. */
        coli_amx_state=(e&&*e=='0')?0:(syscall(SYS_arch_prctl,0x1023,18)==0);
#elif defined(_WIN32)
        /* Windows 11: tile data is an opt-in per-process XState feature.
         * Dynamic lookup keeps older kernels building/running (arming just
         * fails closed there). XSTATE_AMX_TILE_DATA = 18. */
        if(e&&*e=='0') coli_amx_state=0;
        else {
            HMODULE k32=GetModuleHandleA("kernel32.dll");
            typedef BOOL (WINAPI *coli_xstate_fn)(ULONG64);
            coli_xstate_fn f=k32?(coli_xstate_fn)(void*)GetProcAddress(k32,"EnableProcessOptionalXStateFeatures"):NULL;
            coli_amx_state=(f && f(1ull<<18))?1:0;
        }
#else
        (void)e; coli_amx_state=0;
#endif
        if(coli_amx_state)
            fprintf(stderr,"[K1c] AMX int8 tile kernel armed for gs64 tensors "
                           "(AMX=0 disables, engages at S>=%d)\n",coli_amx_s_min);
    }
    return coli_amx_state;
}
/* ldtilecfg layout (palette 1): tmm0=C [rows x 16 i32], tmm1=A [rows x 64 i8],
 * tmm2=B [16 x 64 i8 VNNI]. rows<16 reconfigures for the last partial s-tile. */
struct coli_tilecfg { uint8_t palette,start_row,rsvd[14]; uint16_t colsb[16]; uint8_t rows[16]; };
static void coli_amx_cfg(int arows){
    struct coli_tilecfg c; memset(&c,0,sizeof c); c.palette=1;
    c.rows[0]=(uint8_t)arows; c.colsb[0]=64;
    c.rows[1]=(uint8_t)arows; c.colsb[1]=64;
    c.rows[2]=16;             c.colsb[2]=64;
    _tile_loadconfig(&c);
}
static void matmul_i4p_gidot_amx(float *y, const int8_t *xq, const float *sx,
                                 const uint8_t *q4, const float *scale,
                                 int S, int I, int O16, int O, int gs){
    int rb=(I+1)/2, ng=(I+gs-1)/gs, bpg=gs/64;
    #pragma omp parallel
    {
        float *acc=malloc((size_t)S*16*sizeof(float));
        if(!acc){ fprintf(stderr,"OOM: amx acc\n"); exit(1); }
        int8_t  bstage[4*1024] __attribute__((aligned(64)));   /* bpg<=4 gated below */
        int32_t cbuf[16*16]    __attribute__((aligned(64)));
        float   sclT[16];
        int cur=16; coli_amx_cfg(16);
        #pragma omp for schedule(static)
        for(int ot=0; ot<O16; ot+=16){
            memset(acc,0,(size_t)S*16*sizeof(float));
            for(int g=0; g<ng; g++){
                for(int j=0;j<16;j++) sclT[j]=scale[(int64_t)(ot+j)*ng+g];
                for(int b=0;b<bpg;b++){
                    /* B tile, VNNI [k/4][n][4]: element k of output row ot+n
                     * lands at row k/4, byte n*4+k%4 — signed (v-8) at unpack. */
                    int8_t *dst=bstage+(size_t)b*1024;
                    int64_t boff=((int64_t)g*gs+b*64)>>1;
                    for(int n=0;n<16;n++){
                        const uint8_t *blk=q4+(int64_t)(ot+n)*rb+boff;
                        for(int k=0;k<32;k++){
                            dst[(k>>2)*64+n*4+(k&3)]          =(int8_t)((blk[k]&0xF)-8);
                            dst[((k+32)>>2)*64+n*4+((k+32)&3)]=(int8_t)((blk[k]>>4)-8);
                        }
                    }
                }
                for(int st=0; st<S; st+=16){
                    int rows=S-st<16?S-st:16;
                    if(rows!=cur){ coli_amx_cfg(rows); cur=rows; }
                    _tile_zero(0);
                    for(int b=0;b<bpg;b++){
                        _tile_loadd(1, xq+(int64_t)st*I+(int64_t)g*gs+b*64, (size_t)I);
                        _tile_loadd(2, bstage+(size_t)b*1024, 64);
                        _tile_dpbssd(0,1,2);
                    }
                    _tile_stored(0, cbuf, 64);
                    /* per-lane fmadd == the vector path's per-(s,o) scalar fmaf:
                     * same values, same ascending-g order, single rounding. */
                    for(int r=0;r<rows;r++){
                        __m512 cv=_mm512_cvtepi32_ps(_mm512_load_si512((const void*)(cbuf+(size_t)r*16)));
                        __m512 av=_mm512_loadu_ps(acc+(int64_t)(st+r)*16);
                        _mm512_storeu_ps(acc+(int64_t)(st+r)*16,
                                         _mm512_fmadd_ps(cv,_mm512_loadu_ps(sclT),av));
                    }
                }
            }
            for(int s=0;s<S;s++)
                _mm512_storeu_ps(y+(int64_t)s*O+ot,
                                 _mm512_mul_ps(_mm512_loadu_ps(acc+(int64_t)s*16),
                                               _mm512_set1_ps(sx[s])));
        }
        _tile_release();
        free(acc);
    }
}
#endif

/* vector body over an output range [o0,o1): the 1x4 row tile pays the weight
 * block's load+mask once per 4 activation rows (the fmt=2 K2 tile's idea,
 * brought to the grouped family — the prefill batch-union and the serve mux
 * deliver exactly these multi-row calls). Integer group dots in any lane
 * order are exact, and each row keeps its own ascending-g fmaf chain. */
static void i4p_gidot_rows(float *y, const int8_t *xq, const float *sx,
                           const int32_t *xsg, const uint8_t *q4,
                           const float *scale, int S, int I, int O, int gs,
                           int o0, int o1){
    int rb=(I+1)/2, ng=(I+gs-1)/gs, bpg=gs/64;   /* blocchi-piano per gruppo */
    #pragma omp parallel for schedule(static)
    for(int o=o0;o<o1;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        int s=0;
        for(; s+4<=S; s+=4){
            const int8_t  *x0=xq +(int64_t)s*I,  *x1=x0+I,  *x2=x1+I,  *x3=x2+I;
            const int32_t *g0=xsg+(int64_t)s*ng, *g1=g0+ng, *g2=g1+ng, *g3=g2+ng;
            float a0=0,a1=0,a2=0,a3=0;
            int g=0;
            for(; (g+1)*gs<=I; g++){
                int32_t d0,d1,d2,d3;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
                __m512i A0=_mm512_setzero_si512(),A1=_mm512_setzero_si512();
                __m512i A2=_mm512_setzero_si512(),A3=_mm512_setzero_si512();
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    __m512i wz=i4p_blk512(w+(base>>1));
                    A0=_mm512_dpbusd_epi32(A0,wz,_mm512_loadu_si512((const void*)(x0+base)));
                    A1=_mm512_dpbusd_epi32(A1,wz,_mm512_loadu_si512((const void*)(x1+base)));
                    A2=_mm512_dpbusd_epi32(A2,wz,_mm512_loadu_si512((const void*)(x2+base)));
                    A3=_mm512_dpbusd_epi32(A3,wz,_mm512_loadu_si512((const void*)(x3+base)));
                }
                d0=_mm512_reduce_add_epi32(A0); d1=_mm512_reduce_add_epi32(A1);
                d2=_mm512_reduce_add_epi32(A2); d3=_mm512_reduce_add_epi32(A3);
#elif defined(coli_dpbusd256)
                __m256i A0=_mm256_setzero_si256(),A1=_mm256_setzero_si256();
                __m256i A2=_mm256_setzero_si256(),A3=_mm256_setzero_si256();
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64; __m256i lo,hi;
                    i4p_blk256(w+(base>>1),&lo,&hi);
                    A0=coli_dpbusd256(A0,lo,_mm256_loadu_si256((const __m256i*)(x0+base)));
                    A0=coli_dpbusd256(A0,hi,_mm256_loadu_si256((const __m256i*)(x0+base+32)));
                    A1=coli_dpbusd256(A1,lo,_mm256_loadu_si256((const __m256i*)(x1+base)));
                    A1=coli_dpbusd256(A1,hi,_mm256_loadu_si256((const __m256i*)(x1+base+32)));
                    A2=coli_dpbusd256(A2,lo,_mm256_loadu_si256((const __m256i*)(x2+base)));
                    A2=coli_dpbusd256(A2,hi,_mm256_loadu_si256((const __m256i*)(x2+base+32)));
                    A3=coli_dpbusd256(A3,lo,_mm256_loadu_si256((const __m256i*)(x3+base)));
                    A3=coli_dpbusd256(A3,hi,_mm256_loadu_si256((const __m256i*)(x3+base+32)));
                }
                d0=hsum256_i32(A0); d1=hsum256_i32(A1);
                d2=hsum256_i32(A2); d3=hsum256_i32(A3);
#elif defined(__AVX2__)
                const __m256i ones=_mm256_set1_epi16(1);
                const __m256i m4=_mm256_set1_epi8(0x0F);
                __m256i A0=_mm256_setzero_si256(),A1=_mm256_setzero_si256();
                __m256i A2=_mm256_setzero_si256(),A3=_mm256_setzero_si256();
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    __m256i bb=_mm256_loadu_si256((const __m256i*)(w+(base>>1)));
                    __m256i lo=_mm256_and_si256(bb,m4);
                    __m256i hi=_mm256_and_si256(_mm256_srli_epi16(bb,4),m4);
                    /* maddubs(u8,s8): u<=15, |x|<=127 -> pair <= 3810, int16-safe */
                    A0=_mm256_add_epi32(A0,_mm256_madd_epi16(_mm256_maddubs_epi16(lo,_mm256_loadu_si256((const __m256i*)(x0+base))),ones));
                    A0=_mm256_add_epi32(A0,_mm256_madd_epi16(_mm256_maddubs_epi16(hi,_mm256_loadu_si256((const __m256i*)(x0+base+32))),ones));
                    A1=_mm256_add_epi32(A1,_mm256_madd_epi16(_mm256_maddubs_epi16(lo,_mm256_loadu_si256((const __m256i*)(x1+base))),ones));
                    A1=_mm256_add_epi32(A1,_mm256_madd_epi16(_mm256_maddubs_epi16(hi,_mm256_loadu_si256((const __m256i*)(x1+base+32))),ones));
                    A2=_mm256_add_epi32(A2,_mm256_madd_epi16(_mm256_maddubs_epi16(lo,_mm256_loadu_si256((const __m256i*)(x2+base))),ones));
                    A2=_mm256_add_epi32(A2,_mm256_madd_epi16(_mm256_maddubs_epi16(hi,_mm256_loadu_si256((const __m256i*)(x2+base+32))),ones));
                    A3=_mm256_add_epi32(A3,_mm256_madd_epi16(_mm256_maddubs_epi16(lo,_mm256_loadu_si256((const __m256i*)(x3+base))),ones));
                    A3=_mm256_add_epi32(A3,_mm256_madd_epi16(_mm256_maddubs_epi16(hi,_mm256_loadu_si256((const __m256i*)(x3+base+32))),ones));
                }
                d0=hsum256_i32(A0); d1=hsum256_i32(A1);
                d2=hsum256_i32(A2); d3=hsum256_i32(A3);
#else
                d0=d1=d2=d3=0;
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    const uint8_t *blk=w+(base>>1);
                    for(int k=0;k<32;k++){
                        int32_t ul=(int32_t)(blk[k]&0xF), uh=(int32_t)(blk[k]>>4);
                        d0+=ul*x0[base+k]+uh*x0[base+k+32];
                        d1+=ul*x1[base+k]+uh*x1[base+k+32];
                        d2+=ul*x2[base+k]+uh*x2[base+k+32];
                        d3+=ul*x3[base+k]+uh*x3[base+k+32];
                    }
                }
#endif
                a0=fmaf((float)(d0-8*g0[g]),scl[g],a0);
                a1=fmaf((float)(d1-8*g1[g]),scl[g],a1);
                a2=fmaf((float)(d2-8*g2[g]),scl[g],a2);
                a3=fmaf((float)(d3-8*g3[g]),scl[g],a3);
            }
            if(g*gs<I){                                  /* coda: gruppo parziale, nibble a coppie */
                int32_t d0=0,d1=0,d2=0,d3=0;
                for(int i=g*gs;i<I;i++){
                    uint8_t byte=w[i>>1];
                    int32_t u=(int32_t)((i&1)?(byte>>4):(byte&0xF));
                    d0+=u*x0[i]; d1+=u*x1[i]; d2+=u*x2[i]; d3+=u*x3[i];
                }
                a0=fmaf((float)(d0-8*g0[g]),scl[g],a0);
                a1=fmaf((float)(d1-8*g1[g]),scl[g],a1);
                a2=fmaf((float)(d2-8*g2[g]),scl[g],a2);
                a3=fmaf((float)(d3-8*g3[g]),scl[g],a3);
            }
            y[(int64_t)s*O+o]    =a0*sx[s];
            y[(int64_t)(s+1)*O+o]=a1*sx[s+1];
            y[(int64_t)(s+2)*O+o]=a2*sx[s+2];
            y[(int64_t)(s+3)*O+o]=a3*sx[s+3];
        }
        for(; s<S; s++){
            const int8_t *xr=xq+(int64_t)s*I;
            const int32_t *xg=xsg+(int64_t)s*ng;
            float a=0; int g=0;
            for(; (g+1)*gs<=I; g++){                     /* gruppi interi */
                int32_t d=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
                __m512i acc=_mm512_setzero_si512();
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    acc=_mm512_dpbusd_epi32(acc,i4p_blk512(w+(base>>1)),
                                            _mm512_loadu_si512((const void*)(xr+base)));
                }
                d=_mm512_reduce_add_epi32(acc);
#else
                for(int b=0;b<bpg;b++){
                    int base=g*gs+b*64;
                    const uint8_t *blk=w+(base>>1);
                    const int8_t *xb=xr+base;
#if defined(coli_dpbusd256)
                    __m256i lo,hi; i4p_blk256(blk,&lo,&hi);
                    __m256i acc=_mm256_setzero_si256();
                    acc=coli_dpbusd256(acc,lo,_mm256_loadu_si256((const __m256i*)xb));
                    acc=coli_dpbusd256(acc,hi,_mm256_loadu_si256((const __m256i*)(xb+32)));
                    d+=hsum256_i32(acc);
#elif defined(__AVX2__)
                    const __m256i m4=_mm256_set1_epi8(0x0F);
                    const __m256i ones=_mm256_set1_epi16(1);
                    __m256i bb=_mm256_loadu_si256((const __m256i*)blk);
                    __m256i p0=_mm256_maddubs_epi16(_mm256_and_si256(bb,m4),
                                                    _mm256_loadu_si256((const __m256i*)xb));
                    __m256i p1=_mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(bb,4),m4),
                                                    _mm256_loadu_si256((const __m256i*)(xb+32)));
                    __m256i acc=_mm256_add_epi32(_mm256_madd_epi16(p0,ones),
                                                 _mm256_madd_epi16(p1,ones));
                    d+=hsum256_i32(acc);
#else
                    for(int k=0;k<32;k++){
                        d+=(int32_t)(blk[k]&0xF)*xb[k];
                        d+=(int32_t)(blk[k]>>4)*xb[k+32];
                    }
#endif
                }
#endif
                a=fmaf((float)(d-8*xg[g]),scl[g],a);
            }
            if(g*gs<I){                                  /* coda: gruppo parziale, nibble a coppie */
                int32_t d=0;
                for(int i=g*gs;i<I;i++){
                    uint8_t byte=w[i>>1];
                    d+=(int32_t)((i&1)?(byte>>4):(byte&0xF))*xr[i];
                }
                a=fmaf((float)(d-8*xg[g]),scl[g],a);
            }
            y[(int64_t)s*O+o]=a*sx[s];
        }
    }
}

static void matmul_i4p_grouped_idot(float *y, const int8_t *xq, const float *sx,
                                    const int32_t *xsg, const uint8_t *q4,
                                    const float *scale, int S, int I, int O, int gs){
#if defined(COLI_HAVE_AMX_I4P)
    /* AMX takes the aligned bulk (full 16-output tiles, full groups only —
     * the real gs64 checkpoints have I%gs==0); the vector path finishes any
     * output remainder. Below the row threshold the B-tile unpack does not
     * amortize and the vector tile is the better kernel. */
    if(coli_amx_ok() && S>=coli_amx_s_min && gs%64==0 && gs<=256 && I%gs==0 && O>=16){
        int O16=O&~15;
        matmul_i4p_gidot_amx(y,xq,sx,q4,scale,S,I,O16,O,gs);
        if(O16<O) i4p_gidot_rows(y,xq,sx,xsg,q4,scale,S,I,O,gs,O16,O);
        return;
    }
#endif
    i4p_gidot_rows(y,xq,sx,xsg,q4,scale,S,I,O,gs,0,O);
}

/* matmul IDOT planare (fmt=2): y = (dot_u - 8*xsum[s]) * scale[o] * sx[s].
 * Bit-identico a matmul_i4_idot: somme intere, identita' esatta. */
static void matmul_i4p_idot(float *y, const int8_t *xq, const float *sx, const int32_t *xsum,
                            const uint8_t *q4, const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        int s=0;
#if defined(coli_dpbusd256)
        /* K2: tile 1x4 sulla union — il blocco pesi (load + 2 and + srli) si
         * paga UNA volta per 4 righe di attivazione invece che per riga. La
         * union del prefill consegna nr=2..16 righe per expert: e' esattamente
         * la finestra che il per-riga serviva peggio. Somme intere ->
         * bit-identico al path per-riga per associativita'.
         * EN: 1x4 register tile — the weight block's load+mask cost is paid
         * once per 4 activation rows. Integer sums: bit-identical to the
         * per-row path by associativity. */
        const __m256i m4t=_mm256_set1_epi8(0x0F);
        for(;s+4<=S;s+=4){
            const int8_t *x0=xq+(int64_t)s*I, *x1=x0+I, *x2=x1+I, *x3=x2+I;
            __m256i a0=_mm256_setzero_si256(), a1=_mm256_setzero_si256();
            __m256i a2=_mm256_setzero_si256(), a3=_mm256_setzero_si256();
            int i=0;
            for(;i+64<=I;i+=64){
                __m256i b =_mm256_loadu_si256((const __m256i*)(w+(i>>1)));
                __m256i lo=_mm256_and_si256(b,m4t);
                __m256i hi=_mm256_and_si256(_mm256_srli_epi16(b,4),m4t);
                a0=coli_dpbusd256(a0,lo,_mm256_loadu_si256((const __m256i*)(x0+i)));
                a0=coli_dpbusd256(a0,hi,_mm256_loadu_si256((const __m256i*)(x0+i+32)));
                a1=coli_dpbusd256(a1,lo,_mm256_loadu_si256((const __m256i*)(x1+i)));
                a1=coli_dpbusd256(a1,hi,_mm256_loadu_si256((const __m256i*)(x1+i+32)));
                a2=coli_dpbusd256(a2,lo,_mm256_loadu_si256((const __m256i*)(x2+i)));
                a2=coli_dpbusd256(a2,hi,_mm256_loadu_si256((const __m256i*)(x2+i+32)));
                a3=coli_dpbusd256(a3,lo,_mm256_loadu_si256((const __m256i*)(x3+i)));
                a3=coli_dpbusd256(a3,hi,_mm256_loadu_si256((const __m256i*)(x3+i+32)));
            }
            int32_t d0=hsum256_i32(a0), d1=hsum256_i32(a1);
            int32_t d2=hsum256_i32(a2), d3=hsum256_i32(a3);
            /* coda a coppie, unsigned: stessa identita' -8*xsum del per-riga */
            for(;i<I;i+=2){
                uint8_t byte=w[i>>1];
                d0+=(int32_t)(byte&0xF)*x0[i]; d1+=(int32_t)(byte&0xF)*x1[i];
                d2+=(int32_t)(byte&0xF)*x2[i]; d3+=(int32_t)(byte&0xF)*x3[i];
                if(i+1<I){
                    d0+=(int32_t)(byte>>4)*x0[i+1]; d1+=(int32_t)(byte>>4)*x1[i+1];
                    d2+=(int32_t)(byte>>4)*x2[i+1]; d3+=(int32_t)(byte>>4)*x3[i+1];
                }
            }
            y[(int64_t)(s+0)*O+o]=(float)(d0-8*xsum[s+0])*sc*sx[s+0];
            y[(int64_t)(s+1)*O+o]=(float)(d1-8*xsum[s+1])*sc*sx[s+1];
            y[(int64_t)(s+2)*O+o]=(float)(d2-8*xsum[s+2])*sc*sx[s+2];
            y[(int64_t)(s+3)*O+o]=(float)(d3-8*xsum[s+3])*sc*sx[s+3];
        }
#endif
        for(;s<S;s++){
            int32_t d=dot_i4p_u(w,xq+(int64_t)s*I,I)-8*xsum[s];
            y[(int64_t)s*O+o]=(float)d*sc*sx[s];
        }
    }
}

#endif /* COLIBRI_IDOT_H */
