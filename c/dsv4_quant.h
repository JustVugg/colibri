#ifndef DSV4_QUANT_H
#define DSV4_QUANT_H
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float dsv4_q_bf16(float x){
    uint32_t u; memcpy(&u,&x,4); u+=0x7fff+((u>>16)&1);u&=0xffff0000u;
    memcpy(&x,&u,4);return x;
}

/* IEEE-like finite-only E4M3 used by torch.float8_e4m3fn. */
static inline float dsv4_e4m3(uint8_t v){
    int sign=v>>7, exp=(v>>3)&15, man=v&7;
    if(exp==15&&man==7) return NAN;
    float x=exp ? ldexpf(1.f+man*.125f,exp-7) : ldexpf((float)man,-9);
    return sign ? -x : x;
}

/* Unsigned exponent-only scale used by torch.float8_e8m0fnu. */
static inline float dsv4_e8m0(uint8_t v){
    return v==255 ? NAN : ldexpf(1.f,(int)v-127);
}

static inline float dsv4_e4m3_round(float x){
    int best=0; float bd=INFINITY;
    for(int v=0;v<255;v++){ float q=dsv4_e4m3((uint8_t)v); if(isnan(q))continue;
        float d=fabsf(q-x);
        if(d<bd || (d==bd && !(v&1) && (best&1))){bd=d;best=v;} }
    return dsv4_e4m3((uint8_t)best);
}

static void dsv4_fp8_simulate(float *x,int n,int block){
    for(int a=0;a<n;a+=block){ int z=a+block<n?a+block:n; float mx=1e-4f;
        for(int i=a;i<z;i++)if(fabsf(x[i])>mx)mx=fabsf(x[i]);
        float raw=mx/448.f; int e; frexpf(raw,&e); float scale=ldexpf(1.f,e-1);
        if(scale<raw)scale*=2.f;
        for(int i=a;i<z;i++)x[i]=dsv4_q_bf16(dsv4_e4m3_round(fmaxf(-448.f,fminf(448.f,x[i]/scale)))*scale);
    }
}

/* Reference W8 block-scaled matvec. Weight is [O,I], scale is
 * [ceil(O/128),ceil(I/128)]. Production GPU kernels can replace this without
 * changing the checkpoint representation. */
static void dsv4_fp8_matvec(float *y, const float *x, const uint8_t *w,
                            const uint8_t *scale, int O, int I){
    int SI=(I+127)/128;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        float sum=0.f;
        for(int i=0;i<I;i++)
            sum+=x[i]*dsv4_e4m3(w[(long)o*I+i])*dsv4_e8m0(scale[(o/128)*SI+i/128]);
        y[o]=sum;
    }
}

static void dsv4_fp8_matvec_bf16w(float *y,const float *x,const uint8_t *w,
                                  const uint8_t *scale,int O,int I){
    int SI=(I+127)/128;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ float sum=0.f;
        for(int i=0;i<I;i++){
            float q=dsv4_q_bf16(dsv4_e4m3(w[(long)o*I+i])*dsv4_e8m0(scale[(o/128)*SI+i/128]));
            sum+=x[i]*q;
        } y[o]=sum;
    }
}
#endif
