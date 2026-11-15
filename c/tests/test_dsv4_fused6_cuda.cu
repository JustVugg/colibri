/* Bitwise gate for the fused 6+1 decode MoE (DSV4_CUDA_MOE_FUSED6).
 * Runs BOTH paths in one process (the env var is re-read per call): the launch
 * sequence and the single cooperative kernel must give IDENTICAL float bits on
 * the same inputs. GPU-only, ~300 MB VRAM, no model load. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "../backend_cuda_dsv4.h"

static double now_ms(void){
#ifdef _WIN32
    LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart*1000.0/(double)f.QuadPart;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6;
#endif
}

/* The backend reads the switch with getenv on every call. On Windows only the CRT's
 * own table is visible to getenv, so the switch goes through _putenv_s, not
 * SetEnvironmentVariableA (which getenv would not see). */
static inline void set_fused6(const char *v){
#ifdef _WIN32
    _putenv_s("DSV4_CUDA_MOE_FUSED6",v);
#else
    setenv("DSV4_CUDA_MOE_FUSED6",v,1);
#endif
}

static uint32_t xs_state=0x9E3779B9u;
static uint32_t xs32(void){xs_state^=xs_state<<13;xs_state^=xs_state>>17;xs_state^=xs_state<<5;return xs_state;}

static int upload4(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I){
    return dsv4_cuda_upload_fp4(t,w,s,O,I,0)?0:1;
}

int main(void){
    enum { H=4096, I=2048 };
    int dev=0; if(!dsv4_cuda_init(&dev,1)){puts("fused6 bitwise: no CUDA device, skipped");return 77;}
    size_t w1=(size_t)I*H/2, s1=(size_t)I*H/32, w2=(size_t)H*I/2, s2=(size_t)H*I/32;
    size_t fw8=(size_t)I*H, fs8=(size_t)((I+127)/128)*((H+127)/128);
    uint8_t *wg[3],*ws[3],*ug[3],*us[3],*dg[3],*ds[3];
    for(int k=0;k<3;k++){wg[k]=(uint8_t*)malloc(w1);ws[k]=(uint8_t*)malloc(s1);ug[k]=(uint8_t*)malloc(w1);us[k]=(uint8_t*)malloc(s1);
        dg[k]=(uint8_t*)malloc(w2);ds[k]=(uint8_t*)malloc(s2);
        if(!wg[k]||!ws[k]||!ug[k]||!us[k]||!dg[k]||!ds[k])return 2;
        /* Random fp4 nibbles; UE8M0 scales around 2^-7 (gate/up, 4096 inputs) and 2^-5
         * (down, 2048 inputs) keep the pre-activations of order 1, below the SwiGLU clamp,
         * so the outputs depend on every rounding step. */
        for(size_t j=0;j<w1;j++){wg[k][j]=(uint8_t)xs32();ug[k][j]=(uint8_t)xs32();}
        for(size_t j=0;j<w2;j++)dg[k][j]=(uint8_t)xs32();
        for(size_t j=0;j<s1;j++){ws[k][j]=(uint8_t)(119+xs32()%3);us[k][j]=(uint8_t)(119+xs32()%3);}
        for(size_t j=0;j<s2;j++)ds[k][j]=(uint8_t)(121+xs32()%3);
    }
    uint8_t *s1w=(uint8_t*)malloc(fw8),*s1s=(uint8_t*)malloc(fs8),*s2w=(uint8_t*)malloc(fw8),*s2s=(uint8_t*)malloc(fs8),*s3w=(uint8_t*)malloc(fw8),*s3s=(uint8_t*)malloc(fs8);
    if(!s1w||!s1s||!s2w||!s2s||!s3w||!s3s)return 2;
    /* Random e4m3 codes with exponent field <= 7 (|w| < 2, never NaN), block scales
     * near 2^-4 (gate/up) and 2^-3 (down). */
    for(size_t j=0;j<fw8;j++){s1w[j]=(uint8_t)(xs32()&0xBF);s2w[j]=(uint8_t)(xs32()&0xBF);s3w[j]=(uint8_t)(xs32()&0xBF);}
    for(size_t j=0;j<fs8;j++){s1s[j]=(uint8_t)(122+xs32()%3);s2s[j]=(uint8_t)(122+xs32()%3);s3s[j]=(uint8_t)(123+xs32()%3);}

    Dsv4CudaTensor *gate[6]={0},*up[6]={0},*down[6]={0},*sg=NULL,*su=NULL,*sd=NULL;
    for(int k=0;k<6;k++){
        if(upload4(&gate[k],wg[k%3],ws[k%3],I,H))return 3;
        if(upload4(&up[k],ug[k%3],us[k%3],I,H))return 3;
        if(upload4(&down[k],dg[k%3],ds[k%3],H,I))return 3;
    }
    if(!dsv4_cuda_upload_fp8(&sg,s1w,s1s,I,H,0))return 4;
    if(!dsv4_cuda_upload_fp8(&su,s2w,s2s,I,H,0))return 4;
    if(!dsv4_cuda_upload_fp8(&sd,s3w,s3s,H,I,0))return 4;

    static float x[H],yA[H],yB[H],wt[6];
    int bad=0,fin=0,total=0;
    double msA=0,msB=0,sumsq=0;
    /* 64 inputs and routing-weight sets, each run through the launch sequence
     * (DSV4_CUDA_MOE_FUSED6=0) and the cooperative kernel (=1) and compared bit for bit.
     * Inputs and weights have full 24-bit mantissas: with short ones every product would
     * be exact and a different rounding order could not show. Many outputs, because the
     * bf16 roundings between the stages hide most last-bit differences. The timings show
     * that the fused kernel actually ran: a refused cooperative launch falls back to the
     * launch sequence and would compare it with itself. */
    uint32_t seed=0x2545F491u;
    for(int round=0;round<64;round++){
        for(int j=0;j<H;j++){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;x[j]=((float)(seed>>8)/8388608.f-1.f)*(1.f+(round%3));}
        for(int k=0;k<6;k++){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;wt[k]=(round%4==2&&k==3)?-(float)(seed>>8)/16777216.f:(float)(seed>>8)/16777216.f;}
        set_fused6("0");
        if(!dsv4_cuda_moe(gate,up,down,wt,6,sg,su,sd,7.f,yA,x))return 5;
        set_fused6("1");
        if(!dsv4_cuda_moe(gate,up,down,wt,6,sg,su,sd,7.f,yB,x))return 6;
        for(int j=0;j<H;j++){float a=yA[j],b=yB[j];total++;sumsq+=(double)a*a;
            if(isfinite(a)&&isfinite(b))fin++;
            if(memcmp(&a,&b,sizeof(float))!=0)bad++;}
    }
    for(int pass=0;pass<2;pass++){
        set_fused6(pass?"1":"0");
        float *out=pass?yB:yA;
        if(!dsv4_cuda_moe(gate,up,down,wt,6,sg,su,sd,7.f,out,x))return 9;   /* warm-up */
        double t0=now_ms();
        for(int it=0;it<50;it++)if(!dsv4_cuda_moe(gate,up,down,wt,6,sg,su,sd,7.f,out,x))return 9;
        double dt=(now_ms()-t0)/50.0;
        if(pass)msB=dt;else msA=dt;
    }
    printf("fused6 per call (incl. upload/download/sync): launches %.1f us, fused %.1f us; output rms %.3g\n",
           msA*1000.0,msB*1000.0,sqrt(sumsq/(total>0?total:1)));
    dsv4_cuda_tensor_free(sg);dsv4_cuda_tensor_free(su);dsv4_cuda_tensor_free(sd);
    for(int k=0;k<6;k++){dsv4_cuda_tensor_free(gate[k]);dsv4_cuda_tensor_free(up[k]);dsv4_cuda_tensor_free(down[k]);}
    dsv4_cuda_shutdown();
    if(fin<total){printf("fused6 bitwise: non-finite outputs %d of %d\n",total-fin,total);return 7;}
    if(bad){printf("fused6 bitwise: %d/%d outputs differ\n",bad,total);return 8;}
    printf("fused6 bitwise: OK (%d outputs over 64 inputs)\n",total);
    return 0;
}
