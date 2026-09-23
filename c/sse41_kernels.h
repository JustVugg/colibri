#ifndef COLIBRI_SSE41_KERNELS_H
#define COLIBRI_SSE41_KERNELS_H
/*
 * sse41_kernels.h — shared SSE 4.1 primitives for Colibri engines.
 *
 * The engines (c/deepseek_v4.c, c/colibri.c, c/kimi_k3.c, c/olmoe.c, c/inkling.c)
 * have an `#if defined(__AVX2__)` dispatch for fast paths and fall through to scalar
 * on pre-Haswell hardware (Sandy Bridge: AVX 1.0, no FMA, no AVX-2). This header
 * provides the missing middle tier: 128-bit SIMD primitives, FMA-free.
 *
 * Why a separate header (not just inline in each .c):
 * - 109 AVX2 sites total across 5 engines need patching. Copy-pasting the
 *   128-bit intrinsics 109 times is a typo factory. A single macro definition
 *   is the difference between correct and wrong-on-100-sites.
 * - The single most critical shared piece is the FMA-emulation macro: on
 *   Sandy Bridge, _mm_mul_ps + _mm_add_ps has double rounding vs. hardware
 *   FMA's single rounding, so the output is NOT bit-identical to AVX2 (1-2 ULP
 *   difference). A typo in a single copy is a silent correctness bug.
 *
 * This is the minimum needed for the SSE 4.1 fallback. More primitives can be
 * added as additional engines are patched.
 */
#if defined(__SSE2__)

#include <immintrin.h>
#include <stdint.h>

/*
 * COLIBRI_FMA: emulate FMA on non-FMA hardware.
 *
 * On FMA hardware: maps to _mm_fmadd_ps (single rounding, 1 instruction).
 * On Sandy Bridge (no FMA): separate mul+add, double rounding, 2 instructions.
 *
 * On Sandy Bridge this is NOT bit-identical to AVX2/FMA output — typically
 * within 1-2 ULP. Test tolerance must accommodate this.
 */
#if defined(__FMA__)
#  define COLIBRI_FMA(a, b, c)  _mm_fmadd_ps((a), (b), (c))
#else
#  define COLIBRI_FMA(a, b, c)  _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#endif

/*
 * SSE 4.1 (or lower) load/store helpers. Sandy Bridge has these natively.
 * _mm_load_ps is aligned; _mm_loadu_ps is unaligned. For 128-bit (16-byte)
 * data, aligned loads are faster but UB on misaligned pointers. Default to
 * unaligned: buffers from malloc / numa_slab_bind have no 16-byte guarantee.
 * Aligned loads can be added as a profiled follow-up.
 */
static inline __m128 colibri_sse41_loadu_ps(const float *p) { return _mm_loadu_ps(p); }
static inline void  colibri_sse41_storeu_ps(float *p, __m128 v) { _mm_storeu_ps(p, v); }

/*
 * Min/max (SSE 4.1 native). Identical to AVX2, just narrower width.
 */
static inline __m128 colibri_sse41_min_ps(__m128 a, __m128 b) { return _mm_min_ps(a, b); }
static inline __m128 colibri_sse41_max_ps(__m128 a, __m128 b) { return _mm_max_ps(a, b); }

/*
 * Prefetch (SSE 1+, always available). Identical to AVX2/FMA path.
 */
static inline void colibri_sse41_prefetch(const void *p) { _mm_prefetch(p, _MM_HINT_T0); }

/* Grouped-int4 kernels for the SSE4.1 tier. Callers select even group sizes
 * and pass a multiple of four output rows; scalar dispatch handles the rest.
 * Keep the scalar pair-sum, scale-multiply and accumulator-add order intact. */
#if defined(__SSE4_1__) && !defined(__AVX2__)
/* Load one packed byte from each of four output rows, then unpack their low and
 * high offset nibbles into four f32 lanes. Keeping independent output rows in
 * the lanes preserves the scalar operation order within every row. */
static inline void colibri_sse41_i4_rows4(const uint8_t *q4,int rb,int o,int byte,
                                    __m128 *lo,__m128 *hi){
    const __m128i m4=_mm_set1_epi8(0x0F), b8=_mm_set1_epi8(8);
    /* Read exactly one byte per row, including when a row is one byte wide. */
    __m128i by=_mm_cvtsi32_si128(q4[(int64_t)(o+0)*rb+byte]);
    by=_mm_insert_epi8(by,q4[(int64_t)(o+1)*rb+byte],1);
    by=_mm_insert_epi8(by,q4[(int64_t)(o+2)*rb+byte],2);
    by=_mm_insert_epi8(by,q4[(int64_t)(o+3)*rb+byte],3);
    __m128i qlo=_mm_sub_epi8(_mm_and_si128(by,m4),b8);
    __m128i qhi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(by,4),m4),b8);
    *lo=_mm_cvtepi32_ps(_mm_cvtepi8_epi32(qlo));
    *hi=_mm_cvtepi32_ps(_mm_cvtepi8_epi32(qhi));
}

static inline __m128 colibri_sse41_f32_rows4(const float *p,int stride,int o,int i){
    return _mm_set_ps(p[(int64_t)(o+3)*stride+i],p[(int64_t)(o+2)*stride+i],
                      p[(int64_t)(o+1)*stride+i],p[(int64_t)(o+0)*stride+i]);
}

/* Process four output rows at once without a horizontal reduction. Each lane
 * uses the scalar kernel's pair sum, scale multiply, and accumulator add in
 * the same order, so the result can remain byte-identical on pre-FMA CPUs. */
static void matmul_i4_grouped_sse41_rows4(float *y,const float *x,
                                           const uint8_t *q4,const float *scale,
                                           int S,int I,int O,int gs,int rb,int ng,
                                           int o4){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<o4;o+=4){
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; __m128 a=_mm_setzero_ps();
            for(int g=0;g*gs<I;g++){
                int base=g*gs,end=base+gs; if(end>I) end=I;
                __m128 sc=colibri_sse41_f32_rows4(scale,ng,o,g); int i=base;
                for(;i+1<end;i+=2){
                    __m128 lo,hi; colibri_sse41_i4_rows4(q4,rb,o,i>>1,&lo,&hi);
                    __m128 pair=_mm_add_ps(_mm_mul_ps(_mm_set1_ps(xs[i]),lo),
                                           _mm_mul_ps(_mm_set1_ps(xs[i+1]),hi));
                    a=_mm_add_ps(a,_mm_mul_ps(pair,sc));
                }
                if(i<end){
                    __m128 lo,hi; colibri_sse41_i4_rows4(q4,rb,o,i>>1,&lo,&hi); (void)hi;
                    a=_mm_add_ps(a,_mm_mul_ps(_mm_mul_ps(_mm_set1_ps(xs[i]),lo),sc));
                }
            }
            colibri_sse41_storeu_ps(y+(int64_t)s*O+o,a);
        }
    }
}

/* Fused gate/up shares activation loads while keeping independent accumulators. */
static void matmul_i4_grouped_pair_sse41_rows4(float *yg,float *yu,const float *x,
                                                const uint8_t *qg,const float *sg,
                                                const uint8_t *qu,const float *su,
                                                int S,int I,int O,int gs,int rb,
                                                int ng,int o4){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<o4;o+=4){
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            __m128 ag=_mm_setzero_ps(),au=_mm_setzero_ps();
            for(int g=0;g*gs<I;g++){
                int base=g*gs,end=base+gs; if(end>I) end=I;
                __m128 scg=colibri_sse41_f32_rows4(sg,ng,o,g);
                __m128 scu=colibri_sse41_f32_rows4(su,ng,o,g); int i=base;
                for(;i+1<end;i+=2){
                    __m128 gl,gh,ul,uh;
                    colibri_sse41_i4_rows4(qg,rb,o,i>>1,&gl,&gh);
                    colibri_sse41_i4_rows4(qu,rb,o,i>>1,&ul,&uh);
                    __m128 x0=_mm_set1_ps(xs[i]),x1=_mm_set1_ps(xs[i+1]);
                    __m128 gp=_mm_add_ps(_mm_mul_ps(x0,gl),_mm_mul_ps(x1,gh));
                    __m128 up=_mm_add_ps(_mm_mul_ps(x0,ul),_mm_mul_ps(x1,uh));
                    ag=_mm_add_ps(ag,_mm_mul_ps(gp,scg));
                    au=_mm_add_ps(au,_mm_mul_ps(up,scu));
                }
                if(i<end){
                    __m128 gl,gh,ul,uh;
                    colibri_sse41_i4_rows4(qg,rb,o,i>>1,&gl,&gh);
                    colibri_sse41_i4_rows4(qu,rb,o,i>>1,&ul,&uh); (void)gh; (void)uh;
                    __m128 xi=_mm_set1_ps(xs[i]);
                    ag=_mm_add_ps(ag,_mm_mul_ps(_mm_mul_ps(xi,gl),scg));
                    au=_mm_add_ps(au,_mm_mul_ps(_mm_mul_ps(xi,ul),scu));
                }
            }
            colibri_sse41_storeu_ps(yg+(int64_t)s*O+o,ag);
            colibri_sse41_storeu_ps(yu+(int64_t)s*O+o,au);
        }
    }
}
#endif /* __SSE4_1__ && !__AVX2__ */

#endif /* __SSE2__ */
#endif /* COLIBRI_SSE41_KERNELS_H */
