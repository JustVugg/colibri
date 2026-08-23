#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

static Dsv4CudaTensor*fp8_zero(int O,int I){size_t wn=(size_t)O*I,sn=(size_t)((O+127)/128)*((I+127)/128);uint8_t*w=calloc(wn,1),*s=malloc(sn);Dsv4CudaTensor*t=NULL;if(!w||!s)return NULL;memset(s,127,sn);int ok=dsv4_cuda_upload_fp8(&t,w,s,O,I,0);free(s);free(w);return ok?t:NULL;}

int main(void){
    enum { T=2,H=4096,I=2048,E=256 };
    int dev=0;if(!dsv4_cuda_init(&dev,1))return 1;
    Dsv4CudaTensor*sg=fp8_zero(I,H),*su=fp8_zero(I,H),*sd=fp8_zero(H,I),*gate=NULL;
    float*gw_router=calloc((size_t)E*H,sizeof(float));if(!sg||!su||!sd||!gw_router||!dsv4_cuda_upload_f32(&gate,gw_router,E,H,dev))return 2;free(gw_router);
    Dsv4CudaExpertSet*bank=dsv4_cuda_expert_bank_create(E,H,I,dev,sg,su,sd);if(!bank)return 3;
    size_t w1=(size_t)I*H/2,s1=(size_t)I*H/32,w2=(size_t)H*I/2,s2=(size_t)H*I/32;uint8_t*w=calloc(w1,1),*s=malloc(s1),*dw=calloc(w2,1),*ds=malloc(s2);if(!w||!s||!dw||!ds)return 4;memset(s,127,s1);memset(ds,127,s2);
    for(int e=0;e<E;e++){Dsv4CudaTensor*g=NULL,*u=NULL,*d=NULL;if(!dsv4_cuda_expert_bank_upload(bank,e,w,s,w,s,dw,ds,&g,&u,&d))return 5;dsv4_cuda_tensor_free(g);dsv4_cuda_tensor_free(u);dsv4_cuda_tensor_free(d);}free(ds);free(dw);free(s);free(w);
    float*x=malloc((size_t)T*H*sizeof(float)),*got=malloc((size_t)T*H*sizeof(float));for(int i=0;i<T*H;i++)x[i]=(float)((i%11)-5)/16;
    Dsv4CudaActivation*in=dsv4_cuda_activation_create(dev,(long long)T*H),*out=dsv4_cuda_activation_create(dev,(long long)T*H);
    if(!x||!got||!in||!out||!dsv4_cuda_activation_upload(in,x,(long long)T*H)||!dsv4_cuda_route_moe_batch(in,gate,NULL,NULL,T,1.f,bank,7.f,out)||!dsv4_cuda_activation_download(got,out,(long long)T*H)||!dsv4_cuda_activation_sync(out))return 6;
    for(int i=0;i<T*H;i++)if(!isfinite(got[i])||fabsf(got[i])>1e-6f){fprintf(stderr,"bad output at %d: %g\n",i,got[i]);return 7;}
    dsv4_cuda_activation_free(out);dsv4_cuda_activation_free(in);free(got);free(x);dsv4_cuda_expert_set_free(bank);dsv4_cuda_tensor_free(gate);dsv4_cuda_tensor_free(sd);dsv4_cuda_tensor_free(su);dsv4_cuda_tensor_free(sg);dsv4_cuda_shutdown();puts("dsv4 MoE batch: OK");return 0;
}
