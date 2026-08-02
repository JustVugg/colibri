#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../backend_cuda_dsv4.h"

int main(void){
    enum { T=17,I=4096,O=512 };
    int dev=0;if(!dsv4_cuda_init(&dev,1))return 1;
    uint8_t*w=malloc((size_t)O*I),*s=malloc((O/128)*(I/128));
    float*x=calloc((size_t)T*I,sizeof(float)),*got=malloc((size_t)T*O*sizeof(float)),*ref=malloc(O*sizeof(float));
    if(!w||!s||!x||!got||!ref)return 2;
    for(int o=0;o<O;o++)for(int i=0;i<I;i++)w[(size_t)o*I+i]=((o*3+i*5)&1)?0xb8:0x38;
    for(int i=0;i<(O/128)*(I/128);i++)s[i]=127;
    for(int t=0;t<T;t++)for(int i=0;i<64;i++)x[(size_t)t*I+i]=((t+i*7)%3)-1;
    Dsv4CudaTensor*weight=NULL;Dsv4CudaActivation*in=NULL,*out=NULL;
    if(!dsv4_cuda_upload_fp8(&weight,w,s,O,I,dev)||(in=dsv4_cuda_activation_create(dev,(long long)T*I))==NULL||
       (out=dsv4_cuda_activation_create(dev,(long long)T*O))==NULL||!dsv4_cuda_activation_upload(in,x,(long long)T*I)||
       !dsv4_cuda_matmul_batch(weight,in,T,out)||!dsv4_cuda_activation_download(got,out,(long long)T*O)||!dsv4_cuda_activation_sync(out))return 3;
    for(int t=0;t<T;t++){if(!dsv4_cuda_matvec(weight,ref,x+(size_t)t*I))return 4;for(int o=0;o<O;o++)if(fabsf(ref[o]-got[(size_t)t*O+o])>.01f){fprintf(stderr,"token=%d row=%d ref=%g got=%g\n",t,o,ref[o],got[(size_t)t*O+o]);return 5;}}
    dsv4_cuda_activation_free(out);dsv4_cuda_activation_free(in);dsv4_cuda_tensor_free(weight);dsv4_cuda_shutdown();free(ref);free(got);free(x);free(s);free(w);puts("dsv4 dense batch: OK");return 0;
}
