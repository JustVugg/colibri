#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

static Dsv4CudaTensor*fp8_diag(int O,int I){size_t wn=(size_t)O*I,sn=(size_t)((O+127)/128)*((I+127)/128);uint8_t*w=calloc(wn,1),*s=malloc(sn);Dsv4CudaTensor*t=NULL;if(!w||!s)return NULL;for(int o=0;o<O;o++)w[(size_t)o*I+o%I]=0x38;memset(s,127,sn);int ok=dsv4_cuda_upload_fp8(&t,w,s,O,I,0);free(s);free(w);return ok?t:NULL;}
static Dsv4CudaTensor*fp8_bf16_diag(int O,int I){size_t wn=(size_t)O*I,sn=(size_t)((O+127)/128)*((I+127)/128);uint8_t*w=calloc(wn,1),*s=malloc(sn);Dsv4CudaTensor*t=NULL;if(!w||!s)return NULL;for(int o=0;o<O;o++)w[(size_t)o*I+o%I]=0x38;memset(s,127,sn);int ok=dsv4_cuda_upload_fp8_bf16(&t,w,s,O,I,0);free(s);free(w);return ok?t:NULL;}

int main(void){
    enum { T=17,H=4096,R=1024,K=512,HEADS=64,Q=HEADS*K,MAX=32,ROPE=32 };
    int dev=0;if(!dsv4_cuda_init(&dev,1))return 1;
    float*one=malloc(H*sizeof(float)),*kvone=malloc(K*sizeof(float)),*sinkv=calloc(HEADS,sizeof(float));uint16_t*qnone=malloc(R*sizeof(uint16_t));
    float*rope_cos=malloc((size_t)MAX*ROPE*sizeof(float)),*rope_sin=calloc((size_t)MAX*ROPE,sizeof(float)),*x=malloc((size_t)T*H*sizeof(float)),*got=malloc((size_t)T*Q*sizeof(float));
    if(!one||!kvone||!sinkv||!qnone||!rope_cos||!rope_sin||!x||!got)return 2;
    for(int i=0;i<H;i++)one[i]=1;for(int i=0;i<K;i++)kvone[i]=1;for(int i=0;i<R;i++)qnone[i]=0x3f80;
    for(int i=0;i<MAX*ROPE;i++)rope_cos[i]=1;for(int i=0;i<T*H;i++)x[i]=(float)((i%7)-3)/8;
    Dsv4CudaTensor *an=NULL,*qkv=fp8_diag(R+K,H),*qn=NULL,*qb=fp8_diag(Q,R),*kvn=NULL,*sink=NULL;
    if(!qkv||!qb||!dsv4_cuda_upload_f32(&an,one,H,1,dev)||!dsv4_cuda_upload_bf16(&qn,qnone,R,1,dev)||
       !dsv4_cuda_upload_f32(&kvn,kvone,K,1,dev)||!dsv4_cuda_upload_f32(&sink,sinkv,HEADS,1,dev))return 3;
    Dsv4CudaKvCache*cache=dsv4_cuda_kv_create(dev,MAX,K,MAX,ROPE,rope_cos,rope_sin,rope_cos,rope_sin);
    Dsv4CudaActivation*in=dsv4_cuda_activation_create(dev,(long long)T*H),*out=dsv4_cuda_activation_create(dev,(long long)T*Q);
    if(!cache||!in||!out||!dsv4_cuda_activation_upload(in,x,(long long)T*H)||
       !dsv4_cuda_attention_sparse_batch(in,an,qkv,qn,qb,kvn,sink,HEADS,K,0,T,1e-6f,cache,out)||
       !dsv4_cuda_activation_download(got,out,(long long)T*Q)||!dsv4_cuda_activation_sync(out))return 4;
    for(int i=0;i<T*Q;i++)if(!isfinite(got[i])){fprintf(stderr,"bad output at %d: %g\n",i,got[i]);return 5;}
    Dsv4CudaKvCache*single_cache=dsv4_cuda_kv_create(dev,MAX,K,MAX,ROPE,rope_cos,rope_sin,rope_cos,rope_sin);Dsv4CudaActivation*single_in=dsv4_cuda_activation_create(dev,H),*single_out=dsv4_cuda_activation_create(dev,Q);float*single=malloc(Q*sizeof(float));
    if(!single_cache||!single_in||!single_out||!single)return 6;
    for(int t=0;t<T;t++){
        if(!dsv4_cuda_activation_upload(single_in,x+(long long)t*H,H)||
           !dsv4_cuda_attention_sparse_batch(single_in,an,qkv,qn,qb,kvn,sink,HEADS,K,t,1,1e-6f,single_cache,single_out)||
           !dsv4_cuda_activation_download(single,single_out,Q)||!dsv4_cuda_activation_sync(single_out))return 6;
        for(int i=0;i<Q;i++)if(fabsf(single[i]-got[(long long)t*Q+i])>1e-3f){fprintf(stderr,"causal mismatch token=%d at %d: single=%g batch=%g\n",t,i,single[i],got[(long long)t*Q+i]);return 7;}
    }
    free(single);dsv4_cuda_activation_free(single_out);dsv4_cuda_activation_free(single_in);dsv4_cuda_kv_free(single_cache);
    Dsv4CudaTensor*wa=fp8_bf16_diag(8192,4096),*wb=fp8_diag(4096,8192);Dsv4CudaActivation*projected=dsv4_cuda_activation_create(dev,(long long)T*H);float*projected_host=malloc((size_t)T*H*sizeof(float));
    if(!wa||!wb||!projected||!projected_host||!dsv4_cuda_attention_output_batch(out,wa,wb,8,T,projected)||
       !dsv4_cuda_activation_download(projected_host,projected,(long long)T*H)||!dsv4_cuda_activation_sync(projected))return 8;
    for(int i=0;i<T*H;i++)if(!isfinite(projected_host[i]))return 9;
    free(projected_host);dsv4_cuda_activation_free(projected);dsv4_cuda_tensor_free(wb);dsv4_cuda_tensor_free(wa);
    dsv4_cuda_activation_free(out);dsv4_cuda_activation_free(in);dsv4_cuda_kv_free(cache);dsv4_cuda_tensor_free(sink);dsv4_cuda_tensor_free(kvn);dsv4_cuda_tensor_free(qb);dsv4_cuda_tensor_free(qn);dsv4_cuda_tensor_free(qkv);dsv4_cuda_tensor_free(an);dsv4_cuda_shutdown();
    free(got);free(x);free(rope_sin);free(rope_cos);free(qnone);free(sinkv);free(kvone);free(one);puts("dsv4 attention batch: OK");return 0;
}
