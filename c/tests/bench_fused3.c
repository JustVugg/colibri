/* bench_fused3.c — microbenchmark + correctness harness for the FUSED3
 * kernels (c/fused_simd.h) on OLMoE expert shapes: stock matmul_q IDOT branch
 * (v1, copied verbatim below) vs v2 (re-vectorized row dots) vs v3 (+
 * vectorized activation quant, gate/up pair).
 * Build: gcc -O3 -mavx2 -mfma -fopenmp -I.. bench_fused3.c -o bench_fused3.exe -lm
 * Prints KEY=VALUE lines. Correctness gates:
 *   (a) quant_x_q8_avx2 output byte-identical to the scalar quant
 *       (xi memcmp, xs bit-compare), incl. the all-zero clamp path;
 *   (b) v3/pair-v3 row outputs bit-identical to matmul_q_idot_v2. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "fused_simd.h"

/* dot_i8_16: verbatim AVX2 copy from olmoe.c (static there) — v1 baseline dot */
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    __m256i va16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)a));
    __m256i vb16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)b));
    __m256i prod = _mm256_madd_epi16(va16, vb16);
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(prod), _mm256_extractf128_si256(prod, 1));
    __m128i hi64   = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64  = _mm_add_epi32(sum128, hi64);
    __m128i hi32   = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32  = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

static double now_s(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

static unsigned rng_s=12345;
static unsigned rnd(void){ rng_s=rng_s*1664525u+1013904223u; return rng_s>>8; }

int main(int argc, char **argv){
    int reps = argc>1 ? atoi(argv[1]) : 200;
    printf("# bench_fused3  (S=1 GEMV, int8 Q8_0-per-row, olmoe expert path)\n");
#ifdef _OPENMP
    printf("omp_threads=%d\n", omp_get_max_threads());
#else
    printf("omp_threads=1\n");
#endif

    /* ---- int8 per-row IDOT (olmoe.c expert path) -------------------------- */
    /* v1 baseline: verbatim copy of olmoe.c matmul_q's HAVE_FAST_DOT_I8 branch */
    int i8shapes[2][2] = { {1024,2048}, {2048,1024} };
    for(int sh=0; sh<2; sh++){
        int O=i8shapes[sh][0], I=i8shapes[sh][1];
        int8_t *q8=malloc((int64_t)O*I);
        float *scl=malloc((int64_t)O*sizeof(float));
        float *x=malloc(I*sizeof(float));
        float *y1=malloc((int64_t)O*sizeof(float));
        float *y2=malloc((int64_t)O*sizeof(float));
        if(!q8||!scl||!x||!y1||!y2){ fprintf(stderr,"OOM\n"); return 1; }
        for(int64_t i=0;i<(int64_t)O*I;i++) q8[i]=(int8_t)(rnd()%255-127);
        for(int o=0;o<O;o++) scl[o]=0.001f+((int)(rnd()%1000))/1e6f;
        for(int i=0;i<I;i++) x[i]=((int)(rnd()%2000)-1000)/500.f;

        /* v1 (copy of olmoe.c matmul_q AVX2 IDOT branch) */
        #define MATMUL_Q_IDOT_V1(Y) do{ \
            int nb=I/16; int8_t xi[4096]; float xs[256]; \
            for(int b=0;b<nb;b++){ const float *xb=x+b*16; \
                float am=0.f; for(int i=0;i<16;i++){ float a=fabsf(xb[i]); if(a>am) am=a; } \
                float s=am/127.f; if(s<1e-12f) s=1e-12f; \
                xs[b]=s; float inv=1.f/s; \
                for(int i=0;i<16;i++) xi[b*16+i]=(int8_t)lrintf(xb[i]*inv); } \
            _Pragma("omp parallel for schedule(static)") \
            for(int o=0;o<O;o++){ const int8_t *w=q8+(int64_t)o*I; float acc=0.f; \
                for(int b=0;b<nb;b++) acc+=xs[b]*(float)dot_i8_16(xi+b*16,w+b*16); \
                (Y)[o]=acc*scl[o]; } }while(0)

        /* correctness vs int64 reference (int dot is exact; fp fold differs) */
        double e1=0,e2=0;
        for(int o=0;o<O;o+=O/8+1){
            int nb=I/16; int8_t xi[4096]; float xs[256];
            for(int b=0;b<nb;b++){ const float *xb=x+b*16;
                float am=0.f; for(int i=0;i<16;i++){ float a=fabsf(xb[i]); if(a>am) am=a; }
                float s=am/127.f; if(s<1e-12f) s=1e-12f;
                xs[b]=s; float inv=1.f/s;
                for(int i=0;i<16;i++) xi[b*16+i]=(int8_t)lrintf(xb[i]*inv); }
            double r=0; const int8_t *w=q8+(int64_t)o*I;
            for(int b=0;b<nb;b++){ int32_t d=0; for(int i=0;i<16;i++) d+=(int32_t)xi[b*16+i]*w[b*16+i];
                r+=(double)xs[b]*d; }
            r*=scl[o];
            MATMUL_Q_IDOT_V1(y1);
            matmul_q_idot_v2(y2,x,q8,scl,I,O);
            double d1=fabs(y1[o]-r)/(1+fabs(r)), d2=fabs(y2[o]-r)/(1+fabs(r));
            if(d1>e1)e1=d1; if(d2>e2)e2=d2;
        }
        printf("i8_O%d_I%d: relerr_v1=%.3g relerr_v2=%.3g\n",O,I,e1,e2);

        double t1=1e30,t2=1e30;
        for(int k=0;k<3;k++){ MATMUL_Q_IDOT_V1(y1); matmul_q_idot_v2(y2,x,q8,scl,I,O); }
        for(int r=0;r<reps;r++){
            double a=now_s(); MATMUL_Q_IDOT_V1(y1); double b=now_s();
            if(b-a<t1)t1=b-a;
            a=now_s(); matmul_q_idot_v2(y2,x,q8,scl,I,O); b=now_s();
            if(b-a<t2)t2=b-a;
        }
        double wbytes=(double)O*I;
        printf("i8_O%d_I%d: v1_ms=%.4f v2_ms=%.4f speedup=%.3f v1_GBs=%.1f v2_GBs=%.1f\n",
               O,I,t1*1e3,t2*1e3,t1/t2,wbytes/t1/1e9,wbytes/t2/1e9);
        free(q8);free(scl);free(x);free(y1);free(y2);
        #undef MATMUL_Q_IDOT_V1
    }

    /* ---- FUSED3: vectorized quant + gate/up pair ---------------------------
     * Contract: (a) quant_x_q8_avx2 output byte-identical to the scalar quant
     * (xi memcmp, xs bit-compare); (b) v3/pair-v3 row outputs bit-identical
     * to matmul_q_idot_v2; (c) timing on the two real expert shapes, with the
     * per-expert total = gate+up pair + down. */
    {
        int O=1024, I=2048, Od=2048, Id=1024;   /* gate/up then down (OLMoE) */
        int8_t *qg=malloc((int64_t)O*I), *qu=malloc((int64_t)O*I), *qd=malloc((int64_t)Od*Id);
        float *sg=malloc(O*sizeof(float)), *su=malloc(O*sizeof(float)), *sd=malloc(Od*sizeof(float));
        float *x=malloc(I*sizeof(float)), *xd=malloc(Id*sizeof(float));
        float *yg=malloc(O*sizeof(float)), *yu=malloc(O*sizeof(float)), *yd=malloc(Od*sizeof(float));
        float *yg2=malloc(O*sizeof(float)), *yu2=malloc(O*sizeof(float)), *yd2=malloc(Od*sizeof(float));
        if(!qg||!qu||!qd||!sg||!su||!sd||!x||!xd||!yg||!yu||!yd||!yg2||!yu2||!yd2){ fprintf(stderr,"OOM\n"); return 1; }
        for(int64_t i=0;i<(int64_t)O*I;i++){ qg[i]=(int8_t)(rnd()%255-127); qu[i]=(int8_t)(rnd()%255-127); }
        for(int64_t i=0;i<(int64_t)Od*Id;i++) qd[i]=(int8_t)(rnd()%255-127);
        for(int o=0;o<O;o++){ sg[o]=0.001f+((int)(rnd()%1000))/1e6f; su[o]=0.001f+((int)(rnd()%1000))/1e6f; }
        for(int o=0;o<Od;o++) sd[o]=0.001f+((int)(rnd()%1000))/1e6f;
        for(int i=0;i<I;i++) x[i]=((int)(rnd()%2000)-1000)/500.f;
        for(int i=0;i<Id;i++) xd[i]=((int)(rnd()%2000)-1000)/500.f;
        x[37]=0.f; x[38]=0.f;  /* near-zero block elements: exercise the s<1e-12 clamp path partially */

        /* (a) quant bit-identity vs the scalar loop */
        int nb=I/16; static int8_t xi_s[4096], xi_v[4096]; static float xs_s[256], xs_v[256];
        for(int b=0;b<nb;b++){ const float *xb=x+b*16;
            float am=0.f; for(int i=0;i<16;i++){ float a=fabsf(xb[i]); if(a>am) am=a; }
            float s=am/127.f; if(s<1e-12f) s=1e-12f;
            xs_s[b]=s; float inv=1.f/s;
            for(int i=0;i<16;i++) xi_s[b*16+i]=(int8_t)lrintf(xb[i]*inv); }
        quant_x_q8_avx2(x,xi_v,xs_v,I);
        int qbad=memcmp(xi_s,xi_v,I)!=0 || memcmp(xs_s,xs_v,nb*sizeof(float))!=0;
        /* all-zero block: clamp path */
        { float xz[64]={0}; int8_t xz_s[64],xz_v[64]; float xs2_s[4],xs2_v[4];
          for(int b=0;b<4;b++){ float am=0.f; float s=am/127.f; if(s<1e-12f) s=1e-12f;
              xs2_s[b]=s; float inv=1.f/s;
              for(int i=0;i<16;i++) xz_s[b*16+i]=(int8_t)lrintf(xz[b*16+i]*inv); }
          quant_x_q8_avx2(xz,xz_v,xs2_v,64);
          if(memcmp(xz_s,xz_v,64)!=0 || memcmp(xs2_s,xs2_v,4*sizeof(float))!=0) qbad=1; }
        printf("v3_quant_bitexact=%s\n", qbad?"FAIL":"yes");

        /* (b) v3 / pair-v3 outputs bit-identical to v2 */
        matmul_q_idot_v2(yg,x,qg,sg,I,O);  matmul_q_idot_v3(yg2,x,qg,sg,I,O);
        int bid = memcmp(yg,yg2,O*sizeof(float))==0;
        matmul_q_idot_v2(yg,x,qg,sg,I,O);  matmul_q_idot_v2(yu,x,qu,su,I,O);
        matmul_q_idot_pair_v3(yg2,yu2,x,qg,sg,qu,su,I,O);
        bid = bid && memcmp(yg,yg2,O*sizeof(float))==0 && memcmp(yu,yu2,O*sizeof(float))==0;
        matmul_q_idot_v2(yd,xd,qd,sd,Id,Od); matmul_q_idot_v3(yd2,xd,qd,sd,Id,Od);
        bid = bid && memcmp(yd,yd2,Od*sizeof(float))==0;
        printf("v3_output_bitidentical_v2=%s\n", bid?"yes":"FAIL");

        /* (c) per-expert timing: v2 (3 separate calls) vs v3 (pair + down) */
        double t2e=1e30,t3e=1e30;
        for(int k=0;k<3;k++){
            matmul_q_idot_v2(yg,x,qg,sg,I,O); matmul_q_idot_v2(yu,x,qu,su,I,O); matmul_q_idot_v2(yd,xd,qd,sd,Id,Od);
            matmul_q_idot_pair_v3(yg2,yu2,x,qg,sg,qu,su,I,O); matmul_q_idot_v3(yd,xd,qd,sd,Id,Od); }
        for(int r=0;r<reps;r++){
            double a=now_s();
            matmul_q_idot_v2(yg,x,qg,sg,I,O); matmul_q_idot_v2(yu,x,qu,su,I,O); matmul_q_idot_v2(yd,xd,qd,sd,Id,Od);
            double b=now_s(); if(b-a<t2e)t2e=b-a;
            a=now_s();
            matmul_q_idot_pair_v3(yg2,yu2,x,qg,sg,qu,su,I,O); matmul_q_idot_v3(yd,xd,qd,sd,Id,Od);
            b=now_s(); if(b-a<t3e)t3e=b-a;
        }
        printf("v3_expert: v2_3calls_ms=%.4f v3_pair+down_ms=%.4f speedup=%.3f\n",
               t2e*1e3,t3e*1e3,t2e/t3e);
        free(qg);free(qu);free(qd);free(sg);free(su);free(sd);free(x);free(xd);
        free(yg);free(yu);free(yd);free(yg2);free(yu2);free(yd2);
    }
    return 0;
}
