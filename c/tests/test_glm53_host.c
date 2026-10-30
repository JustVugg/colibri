/* Focused numerics for the actual decode host implementation, CPU only. */
/* Test execution must not disappear when NDEBUG is defined. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "../glm53_host.h"
static uint32_t state = 0x726adb31;
static uint32_t rnd(void) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }
static float val(void) { return ((int)(rnd() & 65535) - 32768) / 32000.f; }
static void same(float a, float b) { if (memcmp(&a,&b,4)) { fprintf(stderr,"mismatch %.9g %.9g\n",a,b); abort(); } }

static void dots(void) {
#if defined(__AVX2__) && defined(__FMA__)
    const int sizes[] = {64,128,192,256,320,512,2048,4096};
    for (int t=0;t<8;t++) {
        int I=sizes[t];uint8_t *mem=malloc(I/2+1),*w=mem+1;
        float *xm=malloc((I+1)*4),*x=xm+1,*sc=malloc((I/64)*4);
        for(int r=0;r<40;r++) {
            for(int i=0;i<I/2;i++)w[i]=(uint8_t)rnd();
            for(int i=0;i<I;i++)x[i]=r==0?0.f:r==1?-0.f:r==2?((i&1)?FLT_MIN:-FLT_MIN):val();
            for(int i=0;i<I/64;i++)sc[i]=(r&1)?fabsf(val())*0.02f:val();
            same(i4g_dot(w,sc,x,I,64),g53h_dot64(w,sc,x,I));
        }
        free(sc);free(xm);free(mem);
    }
#endif
}
static G53HostMat mat(int O,int I,int gs) {
    size_t nq=(size_t)O*((I+1)/2),ns=(size_t)O*((I+gs-1)/gs);
    uint8_t *q=malloc(nq);float *s=malloc(ns*4);
    for(size_t i=0;i<nq;i++)q[i]=(uint8_t)rnd();
    for(size_t i=0;i<ns;i++)s[i]=fabsf(val())*0.015f;
    return (G53HostMat){q,s};
}
static void layer(int n,int H,int F,int gs,int threads) {
    G53HostMat g[5],u[5],d[5];float *dst[5],*ref[5],*x=malloc(H*4),*a=malloc(F*4),*b=malloc(F*4);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    for(int i=0;i<H;i++)x[i]=val();
    for(int e=0;e<n;e++) {
        g[e]=mat(F,H,gs);u[e]=mat(F,H,gs);d[e]=mat(H,F,gs);dst[e]=malloc(H*4);ref[e]=malloc(H*4);
        /* Exercise the existing matmul entry point, including its SSE rows4
         * dispatch, rather than using the fused kernel's row helper as oracle. */
        matmul_i4_grouped(a,x,g[e].q4,g[e].s,1,H,F,gs);
        matmul_i4_grouped(b,x,u[e].q4,u[e].s,1,H,F,gs);
        for(int o=0;o<F;o++){float v=a[o]>10.f?10.f:a[o],w=b[o]<-10.f?-10.f:(b[o]>10.f?10.f:b[o]);a[o]=(v/(1.f+expf(-v)))*w;}
        matmul_i4_grouped(ref[e],a,d[e].q4,d[e].s,1,F,H,gs);
    }
    for(int mode=0;mode<=1;mode++) {
        for(int e=0;e<n;e++)for(int i=0;i<H;i++)dst[e][i]=17.f;
        const int computed=g53h_compute(n,F,H,H,gs,g,u,d,x,10.f,dst,mode);
#if defined(__SSE4_1__) && !defined(__AVX2__)
        const int expected=0;
#else
        const int expected=gs==64 && H%64==0 && F%64==0;
#endif
        if(computed!=expected){fprintf(stderr,"unexpected fusion eligibility: gs=%d H=%d F=%d got=%d expected=%d\n",gs,H,F,computed,expected);abort();}
        /* Declining fusion must leave every output untouched. */
        for(int e=0;e<n;e++)for(int i=0;i<H;i++)same(dst[e][i],computed?ref[e][i]:17.f);
    }
    for(int e=0;e<n;e++){free((void*)g[e].q4);free((void*)g[e].s);free((void*)u[e].q4);free((void*)u[e].s);free((void*)d[e].q4);free((void*)d[e].s);free(dst[e]);free(ref[e]);}
    free(b);free(a);free(x);
}
int main(void) {
    dots();
    for(int t=1;t<=16;t*=4){layer(1,320,192,64,t);layer(5,4096,2048,64,t);layer(2,70,98,64,t);layer(2,128,96,32,t);}
#if defined(__SSE4_1__) && !defined(__AVX2__)
    puts("glm53_host: SSE-only fallback preserves output buffers PASS");
#elif defined(__AVX2__) && defined(__FMA__)
    puts("glm53_host: exact AVX2 dots, g64 layers and unsupported-shape fallback PASS");
#else
    puts("glm53_host: g64 layer parity and unsupported-shape fallback PASS (AVX2 dots not built)");
#endif
    return 0;
}
