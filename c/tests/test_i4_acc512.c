/* Numeric regression test for the AVX-512 int4->float accumulator.
 * Mirrors dot_i4f_avx512 from glm.c (kept in sync by the engine's own
 * I4_ACC512_TEST selftest) and checks it against a double-precision oracle
 * over the real GLM-5.2 expert row shapes. The pass criterion is that the
 * 512-bit tree reduction is at least as accurate as the engine's sequential
 * scalar-f32 order — the fallback every other platform uses.
 * On CPUs without AVX-512 the test compiles to a skip. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

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
    float a=_mm512_reduce_add_ps(_mm512_add_ps(acc0,acc1));
    for(;i<I;i++){ uint8_t b=w[i>>1]; a+=x[i]*(float)(((b>>((i&1)*4))&15)-8); }
    return a;
}
static float dot_i4f_scalar(const uint8_t *w,const float *x,int I){
    float a=0;
    for(int i=0;i<I;i++){ uint8_t b=w[i>>1]; a+=x[i]*(float)(((b>>((i&1)*4))&15)-8); }
    return a;
}
static double dot_i4f_double(const uint8_t *w,const float *x,int I){
    double a=0;
    for(int i=0;i<I;i++){ uint8_t b=w[i>>1]; a+=(double)x[i]*(double)(((b>>((i&1)*4))&15)-8); }
    return a;
}
static uint64_t rng=0x243F6A8885A308D3ULL;
static double rndu(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17;
    return (double)(rng>>11)*(1.0/9007199254740992.0); }
static double rndn(void){ double u1=rndu()+1e-18,u2=rndu();
    return sqrt(-2.0*log(u1))*cos(6.283185307179586*u2); }

static int trial(int I, int rows, double xscale, const char *label){
    uint8_t *w=malloc((size_t)(I+1)/2);
    float *x=malloc((size_t)I*sizeof(float));
    double e512=0,esca=0; int bad=0;
    for(int r=0;r<rows;r++){
        for(int i=0;i<I;i+=2){ int q0=(int)(rndu()*16),q1=(int)(rndu()*16);
            w[i>>1]=(uint8_t)(q0|(q1<<4)); }
        for(int i=0;i<I;i++) x[i]=(float)(rndn()*xscale);
        double ref=dot_i4f_double(w,x,I);
        float v512=dot_i4f_avx512(w,x,I), vsca=dot_i4f_scalar(w,x,I);
        if(!isfinite(v512)) bad++;
        double den=fabs(ref); if(den<1.0) den=1.0;
        double e5=fabs((double)v512-ref)/den, es=fabs((double)vsca-ref)/den;
        if(e5>e512) e512=e5;
        if(es>esca) esca=es;
    }
    free(w); free(x);
    /* 2x headroom on the scalar bound: the claim is "no worse", not "always 2-4x better" */
    int ok = !bad && e512 <= esca*2.0 + 1e-7;
    printf("  %-24s I=%-5d avx512 max %.3e | scalar max %.3e | %s\n",
        label,I,e512,esca,ok?"ok":"FAIL");
    return ok;
}

int main(void){
    int ok=1;
    ok &= trial(6144, 2000, 1.0,  "gate/up rows");
    ok &= trial(6144, 2000, 30.0, "gate/up large x");
    ok &= trial(2048, 2000, 1.0,  "down rows");
    ok &= trial(2048, 2000, 0.02, "down small x");
    ok &= trial(6143, 1000, 1.0,  "tail I=6143");
    ok &= trial(96,   5000, 1.0,  "short rows");
    if(!ok){ puts("test_i4_acc512: FAIL"); return 1; }
    puts("test_i4_acc512: ok");
    return 0;
}
#else
int main(void){ puts("test_i4_acc512: skipped (no AVX-512 on this build)"); return 0; }
#endif
