#ifndef DSV4_MHC_H
#define DSV4_MHC_H
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float dsv4_bf16(float x){
    uint32_t u; memcpy(&u,&x,4); u+=0x7fff+((u>>16)&1); u&=0xffff0000u;
    memcpy(&x,&u,4); return x;
}

/* Reference mHC path. Layouts match the checkpoint and vLLM torch oracle:
 * residual [M,H], fn [2M+M*M,M*H], base [2M+M*M]. */
static void dsv4_mhc_pre(const float *residual, const float *fn,
                         const float scale[3], const float *base,
                         int M, int H, float rms_eps, float pre_eps,
                         float sink_eps, float post_mult, int sink_iters,
                         float *post, float *comb, float *layer_input){
    int MH=M*H, N=2*M+M*M;
    float ss=0.f; for(int i=0;i<MH;i++) ss+=residual[i]*residual[i];
    float inv=1.f/sqrtf(ss/MH+rms_eps);
    float mix[288]; /* M<=16 => 2M+M^2<=288 (validated by config loader) */
    for(int n=0;n<N;n++){
        float v=0.f; const float *w=fn+(long)n*MH;
        for(int i=0;i<MH;i++) v+=residual[i]*w[i];
        mix[n]=v*inv;
    }
    for(int i=0;i<M;i++){
        float pre=1.f/(1.f+expf(-(mix[i]*scale[0]+base[i])))+pre_eps;
        post[i]=(1.f/(1.f+expf(-(mix[M+i]*scale[1]+base[M+i]))))*post_mult;
        for(int h=0;h<H;h++) layer_input[h]+=pre*residual[i*H+h];
    }
    for(int h=0;h<H;h++) layer_input[h]=dsv4_bf16(layer_input[h]);
    for(int i=0;i<M;i++){
        float mx=-INFINITY, sum=0.f;
        for(int j=0;j<M;j++){
            float v=mix[2*M+i*M+j]*scale[2]+base[2*M+i*M+j];
            comb[i*M+j]=v; if(v>mx) mx=v;
        }
        for(int j=0;j<M;j++){ float v=expf(comb[i*M+j]-mx); comb[i*M+j]=v; sum+=v; }
        for(int j=0;j<M;j++) comb[i*M+j]=comb[i*M+j]/sum+sink_eps;
    }
    for(int it=0;it<sink_iters;it++){
        if(it){ for(int i=0;i<M;i++){ float s=0.f; for(int j=0;j<M;j++) s+=comb[i*M+j];
              for(int j=0;j<M;j++) comb[i*M+j]/=s+sink_eps; } }
        for(int j=0;j<M;j++){ float s=0.f; for(int i=0;i<M;i++) s+=comb[i*M+j];
            for(int i=0;i<M;i++) comb[i*M+j]/=s+sink_eps; }
    }
}

static void dsv4_mhc_post(const float *x, const float *residual,
                          const float *post, const float *comb,
                          int M, int H, float *out){
    for(int j=0;j<M;j++) for(int h=0;h<H;h++){
        float v=post[j]*x[h];
        for(int i=0;i<M;i++) v+=comb[i*M+j]*residual[i*H+h];
        out[j*H+h]=dsv4_bf16(v);
    }
}
#endif
