/* fused_simd.h — FUSED3: faster AVX2 kernels for the int8 (Q8_0-per-row)
 * routed-expert path of olmoe.c. Three cooperating pieces:
 *
 *  1) dot_i8idot_avx2_v2 / matmul_q_idot_v2: re-vectorized per-row IDOT matmul
 *     (2 independent accumulator chains instead of matmul_q's 1 per row).
 *  2) quant_x_q8_avx2: AVX2 activation quantization, BIT-IDENTICAL to the
 *     scalar loop in matmul_q (see the proof comment above it).
 *  3) matmul_q_idot_v3 / matmul_q_idot_pair_v3: v2 + vectorized quant, plus a
 *     gate/up pair that quantizes the shared input ONCE instead of twice.
 *
 * Numerics: all integer dots are exact (sign-extend + madd, as dot_i8_16);
 * activation quantization is byte-exact vs the scalar path, so v3/pair-v3
 * outputs are bit-identical to the stock matmul_q calls — verified by
 * memcmp in tests/bench_fused3.c. The flag only changes instruction
 * scheduling, never values.
 *
 * Header-only, all static, no dependencies beyond immintrin — mirrors quant.h
 * conventions. Guarded by __AVX2__; callers keep the stock fallback.
 */
#ifndef COLI_FUSED_SIMD_H
#define COLI_FUSED_SIMD_H

#include <stdint.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>

/* One Q8_0-style row dot: y = sum_b xs[b] * dot16(xi_b, w_b), 16 int8 per
 * block — the same activation block contract as olmoe.c matmul_q's IDOT
 * branch. Integer dots are exact (cvtepi8_epi16 + madd_epi16), so the only
 * numeric delta vs the stock path is fp32 summation order of the per-block
 * fold, exactly as in matmul_q. Requires nb%2==0 for the paired fast loop
 * (activation blocks come in pairs); tail block scalar with v1 semantics. */
static inline float dot_i8idot_avx2_v2(const int8_t *xi, const int8_t *w,
                                       const float *xs, int nb){
    __m256 tot0=_mm256_setzero_ps(), tot1=_mm256_setzero_ps();
    int b=0;
    for(; b+2<=nb; b+=2){
        __m256i pa=_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xi+b*16))),
                                     _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w +b*16))));
        tot0=_mm256_fmadd_ps(_mm256_cvtepi32_ps(pa),_mm256_set1_ps(xs[b]),tot0);
        __m256i pb=_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(xi+b*16+16))),
                                     _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w +b*16+16))));
        tot1=_mm256_fmadd_ps(_mm256_cvtepi32_ps(pb),_mm256_set1_ps(xs[b+1]),tot1);
    }
    __m256 tot=_mm256_add_ps(tot0,tot1);
    __m128 lo4=_mm256_castps256_ps128(tot), hi4=_mm256_extractf128_ps(tot,1);
    lo4=_mm_add_ps(lo4,hi4); __m128 sh=_mm_movehl_ps(lo4,lo4); lo4=_mm_add_ps(lo4,sh);
    sh=_mm_shuffle_ps(lo4,lo4,1); lo4=_mm_add_ss(lo4,sh);
    float acc=_mm_cvtss_f32(lo4);
    for(; b<nb; b++){   /* odd trailing block: v1 semantics, scalar */
        const int8_t *xb=xi+b*16, *wb=w+b*16; int32_t d=0;
        for(int i=0;i<16;i++) d+=(int32_t)xb[i]*wb[i];
        acc+=xs[b]*(float)d;
    }
    return acc;
}

/* Drop-in faster replacement for olmoe.c:matmul_q's IDOT branch (S==1 GEMV,
 * I%16==0, I<=4096). Activation quantization is byte-identical to v1 (scalar,
 * same rounding), so xq/xs match; only the row dot is re-vectorized. */
static void matmul_q_idot_v2(float *y, const float *x, const int8_t *q,
                             const float *scale, int I, int O){
    int nb=I/16; int8_t xi[4096]; float xs[256];
    for(int b=0;b<nb;b++){
        const float *xb=x+b*16;
        float am=0.f; for(int i=0;i<16;i++){ float a=fabsf(xb[i]); if(a>am) am=a; }
        float s=am/127.f; if(s<1e-12f) s=1e-12f;
        xs[b]=s; float inv=1.f/s;
        for(int i=0;i<16;i++) xi[b*16+i]=(int8_t)lrintf(xb[i]*inv);
    }
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const int8_t *w=q+(int64_t)o*I;
        y[o]=dot_i8idot_avx2_v2(xi,w,xs,nb)*scale[o];
    }
}

/* ---- FUSED3: vectorized activation quantization + gate/up pair ------------
 * Profiling the v2 path (bench_fused3 i8 shapes) showed the SCALAR activation
 * quantization — not the row dots — dominating: ~0.12 ms of ~0.19 ms per
 * [1024,2048] call (2048 fabsf + 2048 lrintf, the latter a msvcrt libcall
 * under MinGW). The IDOT row kernel already runs at ~30 GB/s, near the memory
 * ceiling of the mobile APUs this engine targets, so the remaining win is
 * quantizing x with AVX2 and doing it ONCE for the shared gate/up input
 * instead of twice.
 *
 * quant_x_q8_avx2 is BIT-IDENTICAL to the scalar loop above, line by line:
 *  - amax: vector abs (andnot sign) + max is exact, order-free;
 *  - s = am/127.f, clamp, inv = 1.f/s: the same scalar fp ops per block;
 *  - xi = (int8_t)lrintf(x*inv): lrintf honors the current rounding mode
 *    (nearest-even by default) and so does vcvtps2dq; |x|<=am keeps every
 *    product inside [-127,127], so the packs_epi16 saturation never engages
 *    differently from the scalar (int8_t) cast;
 *  - the lane fix-ups (packs_epi32 interleaves 128-bit lanes -> permute4x64)
 *    only reorder, they do not round.
 * Verified byte-for-byte (xi memcmp + xs bit-compare) in bench_fused3.
 * Requires I%16==0, I<=4096 — the same contract as the v2 IDOT path. */
static inline void quant_x_q8_avx2(const float *x, int8_t *xi, float *xs, int I){
    const __m256 sgn=_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    int nb=I/16;
    for(int b=0;b<nb;b++){
        const float *xb=x+b*16;
        __m256 v0=_mm256_andnot_ps(sgn,_mm256_loadu_ps(xb));
        __m256 v1=_mm256_andnot_ps(sgn,_mm256_loadu_ps(xb+8));
        __m256 mx=_mm256_max_ps(v0,v1);
        __m128 m4=_mm_max_ps(_mm256_castps256_ps128(mx),_mm256_extractf128_ps(mx,1));
        m4=_mm_max_ps(m4,_mm_movehl_ps(m4,m4));
        m4=_mm_max_ps(m4,_mm_shuffle_ps(m4,m4,1));
        float am=_mm_cvtss_f32(m4);
        float s=am/127.f; if(s<1e-12f) s=1e-12f;
        xs[b]=s; float inv=1.f/s;
        __m256 vi=_mm256_set1_ps(inv);
        __m256i q0=_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb),  vi));
        __m256i q1=_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb+8),vi));
        __m256i p16=_mm256_packs_epi32(q0,q1);              /* [q0_0..3 q1_0..3 | q0_4..7 q1_4..7] */
        p16=_mm256_permute4x64_epi64(p16,0xD8);             /* [q0(8) | q1(8)] int16 */
        __m128i p8=_mm_packs_epi16(_mm256_castsi256_si128(p16),
                                   _mm256_extracti128_si256(p16,1));
        _mm_storeu_si128((__m128i*)(xi+b*16),p8);
    }
}

/* v3 single-projection: v2 with the activation quantization vectorized.
 * Row dots unchanged (dot_i8idot_avx2_v2) -> output bit-identical to v2. */
static void matmul_q_idot_v3(float *y, const float *x, const int8_t *q,
                             const float *scale, int I, int O){
    int nb=I/16; int8_t xi[4096]; float xs[256];
    quant_x_q8_avx2(x,xi,xs,I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const int8_t *w=q+(int64_t)o*I;
        y[o]=dot_i8idot_avx2_v2(xi,w,xs,nb)*scale[o];
    }
}

/* v3 gate+up pair: one quantization of the shared input feeds both weight
 * streams (v2 runs the scalar quant twice — once per matmul_q call). Each
 * row dot is unchanged, so yg/yu are bit-identical to two v2/v3 calls. */
static void matmul_q_idot_pair_v3(float *yg, float *yu, const float *x,
                                  const int8_t *qg, const float *sg,
                                  const int8_t *qu, const float *su,
                                  int I, int O){
    int nb=I/16; int8_t xi[4096]; float xs[256];
    quant_x_q8_avx2(x,xi,xs,I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const int8_t *wg=qg+(int64_t)o*I, *wu=qu+(int64_t)o*I;
        yg[o]=dot_i8idot_avx2_v2(xi,wg,xs,nb)*sg[o];
        yu[o]=dot_i8idot_avx2_v2(xi,wu,xs,nb)*su[o];
    }
}
#endif /* __AVX2__ */

#endif /* COLI_FUSED_SIMD_H */
