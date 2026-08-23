#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../dsv4_mhc.h"

int main(void){
    enum { M=4,H=3,N=2*M+M*M };
    float r[M*H], fn[N*M*H], base[N], scale[3]={1,1,1};
    float post[M], comb[M*M], in[H]={0}, x[H]={2,-1,.5f}, out[M*H];
    for(int i=0;i<M*H;i++) r[i]=(float)(i-5);
    memset(fn,0,sizeof(fn)); memset(base,0,sizeof(base));
    dsv4_mhc_pre(r,fn,scale,base,M,H,1e-6f,0.f,1e-6f,2.f,20,post,comb,in);
    for(int i=0;i<M;i++) if(fabsf(post[i]-1.f)>1e-6f) return 1;
    for(int h=0;h<H;h++){
        float want=0; for(int i=0;i<M;i++) want+=.5f*r[i*H+h];
        if(fabsf(in[h]-want)>1e-5f) return 2;
    }
    for(int j=0;j<M;j++){ float col=0; for(int i=0;i<M;i++) col+=comb[i*M+j];
        if(fabsf(col-1.f)>1e-5f) return 3; }
    dsv4_mhc_post(x,r,post,comb,M,H,out);
    for(int j=0;j<M;j++) for(int h=0;h<H;h++){
        float want=x[h]; for(int i=0;i<M;i++) want+=comb[i*M+j]*r[i*H+h];
        if(fabsf(out[j*H+h]-want)>1e-5f) return 4;
    }
    puts("dsv4-mhc: OK"); return 0;
}
