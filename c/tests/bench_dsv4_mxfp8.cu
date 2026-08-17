#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CUDA_OK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)
#define BLAS_OK(x) do{cublasStatus_t e=(x);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %d at %d\n",(int)e,__LINE__);return 2;}}while(0)
static float he4(uint8_t b){int s=b>>7,e=(b>>3)&15,m=b&7;float v=!e?std::ldexp((float)m,-9):std::ldexp(1.f+m*.125f,e-7);return s?-v:v;}
static float hround(float x){int best=0;float bd=INFINITY;for(int v=0;v<255;v++){float q=he4(v),d=std::fabs(q-x);if(d<bd||(d==bd&&!(v&1)&&(best&1))){bd=d;best=v;}}return he4(best);}
static const float hf4[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};

static __device__ int sf_off(int outer,int inner,int sf_inner_dim){
    int base=((inner/4)*4+(outer/128)*sf_inner_dim)*128;
    int o=outer&127,i=inner&3;
    return base+(o%32)*16+(o/32)*4+i;
}

__global__ void expand_weight(const uint8_t *q4,const uint8_t *src_scale,uint8_t *q8,uint8_t *dst_scale,int O,int I){
    static const uint8_t lut[16]={0x00,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c,0x80,0xb0,0xb8,0xbc,0xc0,0xc4,0xc8,0xcc};
    long long n=(long long)O*I;for(long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;p<n;p+=(long long)gridDim.x*blockDim.x){
        uint8_t b=q4[p>>1];q8[p]=lut[(p&1)?b>>4:b&15];
    }
    int blocks=O*(I/32);for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<blocks;p+=gridDim.x*blockDim.x){
        int o=p/(I/32),k=p%(I/32);dst_scale[sf_off(o,k,(I/32+3)&~3)]=src_scale[p];
    }
}

__device__ float e4(uint8_t b){int s=b>>7,e=(b>>3)&15,m=b&7;float v=!e?ldexpf((float)m,-9):ldexpf(1.f+m*.125f,e-7);return s?-v:v;}
__device__ uint8_t e4round(float x){int best=0;float bd=INFINITY;for(int v=0;v<255;v++){float q=e4(v);float d=fabsf(q-x);if(d<bd||(d==bd&&!(v&1)&&(best&1))){bd=d;best=v;}}return best;}
__global__ void quant_x(const float *x,uint8_t *q,uint8_t *scale,int I){
    int b=blockIdx.x,i=b*128+threadIdx.x;__shared__ float mx[128];float v=i<I?fabsf(x[i]):0;mx[threadIdx.x]=v;__syncthreads();
    for(int n=64;n;n>>=1){if(threadIdx.x<n&&mx[threadIdx.x+n]>mx[threadIdx.x])mx[threadIdx.x]=mx[threadIdx.x+n];__syncthreads();}
    float raw=fmaxf(mx[0],1e-4f)/448.f;int e;frexpf(raw,&e);float s=ldexpf(1.f,e-1);if(s<raw)s*=2;int se;frexpf(s,&se);int code=se-1+127;
    if(i<I)q[i]=e4round(fmaxf(-448.f,fminf(448.f,x[i]/s)));
    if(threadIdx.x<4){int k=b*4+threadIdx.x;scale[sf_off(0,k,(I/32+3)&~3)]=(uint8_t)code;}
}

int main(){const int O=2048,I=4096,iters=100;size_t wb=(size_t)O*I/2,sb=(size_t)O*I/32;
    size_t sf_inner=(I/32+3)&~3,asb=sf_inner*((O+127)/128)*128,bsb=sf_inner*128;
    std::vector<uint8_t> hw(wb),hs(sb,127);std::vector<float> hx(I);for(size_t i=0;i<wb;i++)hw[i]=(uint8_t)(i*29);for(int i=0;i<I;i++)hx[i]=(i%31-15)*.003f;
    uint8_t *w4,*ws,*w8,*as,*x8,*xs;float *x,*y;void *workspace;CUDA_OK(cudaMalloc(&w4,wb));CUDA_OK(cudaMalloc(&ws,sb));CUDA_OK(cudaMalloc(&w8,(size_t)O*I));CUDA_OK(cudaMalloc(&as,asb));CUDA_OK(cudaMalloc(&x,I*4));CUDA_OK(cudaMalloc(&x8,I));CUDA_OK(cudaMalloc(&xs,bsb));CUDA_OK(cudaMalloc(&y,O*4));CUDA_OK(cudaMalloc(&workspace,32<<20));
    CUDA_OK(cudaMemcpy(w4,hw.data(),wb,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(ws,hs.data(),sb,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(x,hx.data(),I*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemset(as,0,asb));CUDA_OK(cudaMemset(xs,0,bsb));
    cublasLtHandle_t lt;BLAS_OK(cublasLtCreate(&lt));cublasLtMatmulDesc_t op;BLAS_OK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;BLAS_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));BLAS_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    cublasLtMatmulMatrixScale_t mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;BLAS_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));BLAS_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));BLAS_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&as,sizeof(as)));BLAS_OK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&xs,sizeof(xs)));
    cublasLtMatrixLayout_t ad,bd,cd,dd;BLAS_OK(cublasLtMatrixLayoutCreate(&ad,CUDA_R_8F_E4M3,I,O,I));BLAS_OK(cublasLtMatrixLayoutCreate(&bd,CUDA_R_8F_E4M3,I,1,I));BLAS_OK(cublasLtMatrixLayoutCreate(&cd,CUDA_R_32F,O,1,O));BLAS_OK(cublasLtMatrixLayoutCreate(&dd,CUDA_R_32F,O,1,O));cublasLtMatmulPreference_t pref;BLAS_OK(cublasLtMatmulPreferenceCreate(&pref));size_t wsz=32<<20;BLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz)));cublasLtMatmulHeuristicResult_t algo{};int found=0;BLAS_OK(cublasLtMatmulAlgoGetHeuristic(lt,op,ad,bd,cd,dd,pref,1,&algo,&found));if(!found){fprintf(stderr,"no MXFP8 algo\n");return 3;}
    float alpha=1,beta=0;cudaEvent_t a,b;CUDA_OK(cudaEventCreate(&a));CUDA_OK(cudaEventCreate(&b));CUDA_OK(cudaEventRecord(a));for(int n=0;n<iters;n++){expand_weight<<<256,256>>>(w4,ws,w8,as,O,I);quant_x<<<I/128,128>>>(x,x8,xs,I);BLAS_OK(cublasLtMatmul(lt,op,&alpha,w8,ad,x8,bd,&beta,y,cd,y,dd,&algo.algo,workspace,wsz,0));}CUDA_OK(cudaEventRecord(b));CUDA_OK(cudaEventSynchronize(b));float ms;CUDA_OK(cudaEventElapsedTime(&ms,a,b));
    std::vector<float> hy(O),qx=hx;CUDA_OK(cudaMemcpy(hy.data(),y,O*4,cudaMemcpyDeviceToHost));for(int p=0;p<I;p+=128){float mx=1e-4f;for(int i=p;i<p+128;i++)mx=fmaxf(mx,fabsf(qx[i]));float raw=mx/448.f;int e;frexpf(raw,&e);float s=ldexpf(1.f,e-1);if(s<raw)s*=2;for(int i=p;i<p+128;i++)qx[i]=hround(fmaxf(-448.f,fminf(448.f,qx[i]/s)))*s;}
    float worst=0;for(int o=0;o<8;o++){float z=0;for(int i=0;i<I;i++){uint8_t p=hw[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;z+=qx[i]*hf4[q];}worst=fmaxf(worst,fabsf(z-hy[o]));}
    printf("mxfp8 expand+matvec O=%d I=%d %.4f ms/call worst8=%g\n",O,I,ms/iters,worst);return worst>1e-3f;
}
