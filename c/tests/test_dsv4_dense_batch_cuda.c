#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../backend_cuda_dsv4.h"

/* fp8 tensors smaller than the DeepGEMM kernels take (I < 512, or fmt 9 with
 * O < 1024) must still upload and run on the generic kernels. DeepGEMM builds
 * packed their scales anyway: an empty launch grid failed the upload (I < 512)
 * and fmt 9 divided by O/1024 == 0 on the host. The reference is exact: +-1
 * weights at scale 1 times inputs in {-1,0,1}. */
static int small_shapes(int dev){
    static const struct { int O,I,fmt; } shapes[]={{128,128,8},{128,128,9},{512,256,8},{512,256,9},{256,384,9},{512,512,9}};
    for(size_t k=0;k<sizeof(shapes)/sizeof(*shapes);k++){int O=shapes[k].O,I=shapes[k].I,fmt=shapes[k].fmt,good=0;size_t sn=(size_t)(O/128)*(I/128);
        uint8_t*w=malloc((size_t)O*I),*s=malloc(sn);float*x=malloc((size_t)I*sizeof(float)),*y=malloc((size_t)O*sizeof(float));Dsv4CudaTensor*t=NULL;
        if(w&&s&&x&&y){
            for(int o=0;o<O;o++)for(int i=0;i<I;i++)w[(size_t)o*I+i]=((o*3+i*5)&1)?0xb8:0x38;
            for(size_t i=0;i<sn;i++)s[i]=127;
            for(int i=0;i<I;i++)x[i]=(float)((i*7+(int)k)%3)-1;
            if(!(fmt==9?dsv4_cuda_upload_fp8_bf16(&t,w,s,O,I,dev):dsv4_cuda_upload_fp8(&t,w,s,O,I,dev)))fprintf(stderr,"%dx%d fmt=%d: upload failed\n",O,I,fmt);
            else if(!dsv4_cuda_matvec(t,y,x))fprintf(stderr,"%dx%d fmt=%d: matvec failed\n",O,I,fmt);
            else{good=1;for(int o=0;o<O&&good;o++){float r=0;for(int i=0;i<I;i++)r+=(((o*3+i*5)&1)?-1.f:1.f)*x[i];
                if(fabsf(y[o]-r)>.01f){fprintf(stderr,"%dx%d fmt=%d row=%d ref=%g got=%g\n",O,I,fmt,o,r,y[o]);good=0;}}}
        }
        dsv4_cuda_tensor_free(t);free(y);free(x);free(s);free(w);if(!good)return 0;}
    return 1;
}

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
    dsv4_cuda_activation_free(out);dsv4_cuda_activation_free(in);dsv4_cuda_tensor_free(weight);free(ref);free(got);free(x);free(s);free(w);
    if(!small_shapes(dev))return 6;
    dsv4_cuda_shutdown();puts("dsv4 dense batch: OK");return 0;
}
