#include "backend_cuda_dsv4.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef COLI_DSV4_NCCL
#include <nccl.h>
#endif
#ifdef COLI_DSV4_DEEPGEMM
#include <cuda.h>
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm120_split_k_reduce.cuh>
#include <deep_gemm/impls/sm120_bf16_gemm.cuh>
#endif

/* The cuBLASLt MXFP8 tensor-core path (tc_plan/tc_fp4_matvec, enabled at
 * runtime by DSV4_CUDA_TC=1, off by default) needs the cuBLASLt >= 12.8
 * block-scaling APIs (CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 / *_SCALE_MODE).
 * Older toolkits (e.g. Jetson JetPack with CUDA 12.6) lack them; build with
 * -DCOLI_DSV4_NO_TC to compile without that path (generic DLL/object only —
 * the DeepGEMM sm_120a flavour needs 12.8+ anyway). */
#if defined(COLI_DSV4_NO_TC)
#define COLI_DSV4_TC 0
#else
#define COLI_DSV4_TC 1
#endif
struct Dsv4CudaTensor { void *w,*bf16_cache; uint8_t *scale;int *dg_scale; int O,I,fmt,device,own_w,own_scale,own_dg_scale; long long bytes; };
struct Dsv4CudaActivation { float *data; long long elements; int device; };
struct Dsv4CudaGraph { cudaGraph_t graph;cudaGraphExec_t exec;int device; };
struct Dsv4CudaKvCache { float *kv,*compressed,*comp_value,*comp_score,*rope_cos,*rope_sin,*comp_cos,*comp_sin;int *compressed_len;int device,window,dim,max_tokens,max_compressed,rope_pairs; };
struct MvDesc { const uint8_t *w,*scale; const float *x; float *y; };
struct ExpertPtr { const uint8_t *w,*scale; };
struct Dsv4CudaExpertSet { ExpertPtr *table;int64_t *hash;Dsv4CudaTensor *sg,*su,*sd;uint8_t *bank_fc1,*bank_fc1_scale,*bank_fc2,*bank_fc2_scale;int *bank_fc1_dg_scale,*bank_fc2_dg_scale;int count,device,H,I,vocab,topk; };
struct TcPlan {int O,I,ok;cublasLtMatmulDesc_t op;cublasLtMatrixLayout_t a,b,c,d;cublasLtMatmulHeuristicResult_t algo;};
struct Dev { int id,sms; cudaStream_t stream,aux; cudaEvent_t ready,fork,join;float *dx,*dy,*p1,*p2,*p3,*p4,*aux1,*aux2,*aux3; size_t xcap,ycap,p1cap,p2cap,p3cap,p4cap,aux1cap,aux2cap,aux3cap; float *hx,*hy;
#ifdef COLI_DSV4_NCCL
    ncclComm_t ep_comm;
#endif
    int *decode_state;
    MvDesc *mvdesc;size_t mvdesccap;float *expert_weights;size_t expert_weightscap;int *expert_ids;size_t expert_idcap;
    cublasLtHandle_t lt;void *tc_workspace;uint8_t *tcw,*tcs,*tcx,*tcxs;size_t tcwcap,tcscap,tcxcap,tcxscap;TcPlan plans[2];
#ifdef COLI_DSV4_DEEPGEMM
    uint8_t *dga1,*dga2;int *dgsfa1,*dgsfa2,*dglayout,*dgrowmap;__nv_bfloat16 *dgmm1,*dgmm2;float *dgref,*dgdensews;size_t dga1cap,dga2cap,dgsfa1cap,dgsfa2cap,dglayoutcap,dgrowmapcap,dgmm1cap,dgmm2cap,dgrefcap,dgdensewscap;
#endif
};
static Dev g[16];static int ng;
static Dev *ctx(int d){for(int i=0;i<ng;i++)if(g[i].id==d)return g+i;return nullptr;}
static int ok(cudaError_t e,const char *s){if(e==cudaSuccess)return 1;fprintf(stderr,"[DSV4 CUDA] %s: %s\n",s,cudaGetErrorString(e));return 0;}
static int bok(cublasStatus_t e,const char*s){if(e==CUBLAS_STATUS_SUCCESS)return 1;fprintf(stderr,"[DSV4 CUDA] %s: cuBLAS status %d\n",s,(int)e);return 0;}
#ifdef COLI_DSV4_NCCL
static int nok(ncclResult_t e,const char*s){if(e==ncclSuccess)return 1;fprintf(stderr,"[DSV4 NCCL] %s: %s\n",s,ncclGetErrorString(e));return 0;}
#endif
static int buf(void **p,size_t *cap,size_t n){if(*cap>=n)return 1;if(*p)cudaFree(*p);*p=nullptr;*cap=0;if(!ok(cudaMalloc(p,n),"buffer allocation"))return 0;*cap=n;return 1;}


__constant__ float e8_table[256];
__global__ void set_decode_state(int*state,int token,int position){if(!threadIdx.x){state[0]=token;state[1]=position;}}
__device__ float e4m3(uint8_t b){int sign=b>>7,e=(b>>3)&15,m=b&7;float v;
    if(!e)v=ldexpf((float)m,-9);else if(e==15)v=m==7?NAN:ldexpf(1.f+m/8.f,8);else v=ldexpf(1.f+m/8.f,e-7);
    return sign?-v:v;}
__device__ float e8(uint8_t b){return e8_table[b];}
__device__ float f4(uint8_t q){const float a[8]={0,.5f,1,1.5f,2,3,4,6};return q&8?-a[q&7]:a[q];}
template<int F> __global__ void mv(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups){
    int o=blockIdx.x;if(o>=O)return;float sum=0;
    const float *xi=x+(long long)(o/(O/groups))*I;for(int i=threadIdx.x;i<I;i+=blockDim.x){float v;
        if constexpr(F==8||F==9){v=e4m3(w[(long long)o*I+i])*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));}
        else {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}
        sum+=xi[i]*v;
    }
    __shared__ float r[256];r[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)r[threadIdx.x]+=r[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[o]=r[0];
}
template<int F,int R> __global__ void mv_rows(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups){
    int base=blockIdx.x*R;float sum[R]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;if(o<O){const float *xi=x+(long long)(o/(O/groups))*I;float v;
            if constexpr(F==8||F==9){v=e4m3(w[(long long)o*I+i])*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));}
            else {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}sum[r]+=xi[i]*v;}}
    }
    __shared__ float s[R][256];
    #pragma unroll
    for(int r=0;r<R;r++)s[r][threadIdx.x]=sum[r];__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int r=0;r<R;r++)s[r][threadIdx.x]+=s[r][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int r=0;r<R;r++)if(base+r<O)y[base+r]=s[r][0];}
}
template<int R> __global__ void mv_fp4_grouped(const MvDesc *desc,int count,int O,int I){
    int matrix=blockIdx.y,base=blockIdx.x*R;if(matrix>=count)return;MvDesc d=desc[matrix];float sum[R]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;if(o<O){uint8_t p=d.w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;
            sum[r]+=d.x[i]*f4(q)*e8(d.scale[(long long)o*(I/32)+i/32]);}}
    }
    __shared__ float s[R][256];
    #pragma unroll
    for(int r=0;r<R;r++)s[r][threadIdx.x]=sum[r];__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int r=0;r<R;r++)s[r][threadIdx.x]+=s[r][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int r=0;r<R;r++)if(base+r<O)d.y[base+r]=s[r][0];}
}
template<int F> static void run_mv(const uint8_t*w,const uint8_t*sc,const float*x,float*y,int O,int I,int groups,cudaStream_t stream){
    static int rows=-1;if(rows<0){const char *p=getenv(F==4?"DSV4_CUDA_ROWS":"DSV4_CUDA_FP8_ROWS");rows=p?atoi(p):(F==4?4:1);}
    if(rows==32)mv_rows<F,32><<<(O+31)/32,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==16)mv_rows<F,16><<<(O+15)/16,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==8)mv_rows<F,8><<<(O+7)/8,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==4)mv_rows<F,4><<<(O+3)/4,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else mv<F><<<O,256,0,stream>>>(w,sc,x,y,O,I,groups);
}
static void run_fp4_grouped(const MvDesc *desc,int count,int O,int I,cudaStream_t stream){
    static int rows=-1;if(rows<0){const char *p=getenv("DSV4_CUDA_FP4_ROWS");rows=p?atoi(p):2;}
    if(rows==8)mv_fp4_grouped<8><<<dim3((O+7)/8,count),256,0,stream>>>(desc,count,O,I);
    else if(rows==4)mv_fp4_grouped<4><<<dim3((O+3)/4,count),256,0,stream>>>(desc,count,O,I);
    else if(rows==2)mv_fp4_grouped<2><<<dim3((O+1)/2,count),256,0,stream>>>(desc,count,O,I);
    else mv_fp4_grouped<1><<<dim3(O,count),256,0,stream>>>(desc,count,O,I);
}
/* Prefill batch GEMM without DeepGEMM: one block per output row and 16-token
 * tile, so each dequantized weight value is reused across the tile and the
 * weight matrix is read ceil(T/16) times per call instead of T times (what a
 * matvec-per-token loop would cost). Numerics match mv<F> exactly: same
 * e4m3/e8 dequant, same bf16 round-trip for fmt 9. x is tokens-major
 * [T][I], y is tokens-major [T][O] — the layout dg_dense_batch_dispatch and
 * the engine's coli_fp8_matmul_batch_ref both use. */
template<int F> __global__ void mm_batch(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int T){
    int o=blockIdx.x;if(o>=O)return;int tbase=blockIdx.y*16,tcount=T-tbase<16?T-tbase:16;
    float sum[16]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){float v;
        if constexpr(F==8||F==9){v=e4m3(w[(long long)o*I+i])*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));}
        else {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}
        #pragma unroll
        for(int t=0;t<16;t++)if(t<tcount)sum[t]+=x[(long long)(tbase+t)*I+i]*v;
    }
    __shared__ float s[16][256];
    #pragma unroll
    for(int t=0;t<16;t++)s[t][threadIdx.x]=sum[t];
    __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int t=0;t<16;t++)s[t][threadIdx.x]+=s[t][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x)for(int t=0;t<tcount;t++)y[(long long)(tbase+t)*O+o]=s[t][0];
}
static void run_mm_batch(Dsv4CudaTensor*t,const float*x,float*y,int T,cudaStream_t stream){
    dim3 grid(t->O,(T+15)/16);
    if(t->fmt==4)mm_batch<4><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
    else if(t->fmt==9)mm_batch<9><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
    else mm_batch<8><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
}
__device__ int tc_sf_off(int outer,int inner,int sf_inner_dim){int base=((inner/4)*4+(outer/128)*sf_inner_dim)*128,o=outer&127;return base+(o%32)*16+(o/32)*4+(inner&3);}
__device__ uint8_t e4m3_code(float x){int best=0;float bd=INFINITY;for(int v=0;v<255;v++){float z=e4m3((uint8_t)v),d=fabsf(z-x);if(d<bd||(d==bd&&!(v&1)&&(best&1))){bd=d;best=v;}}return (uint8_t)best;}
__global__ void tc_expand(const uint8_t*q4,const uint8_t*src,uint8_t*q8,uint8_t*dst,int O,int I){
    const uint8_t lut[16]={0x00,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c,0x80,0xb0,0xb8,0xbc,0xc0,0xc4,0xc8,0xcc};long long n=(long long)O*I;
    for(long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;p<n;p+=(long long)gridDim.x*blockDim.x){uint8_t b=q4[p>>1];q8[p]=lut[(p&1)?b>>4:b&15];}
    int blocks=O*(I/32),sid=(I/32+3)&~3;for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<blocks;p+=gridDim.x*blockDim.x){int o=p/(I/32),k=p%(I/32);dst[tc_sf_off(o,k,sid)]=src[p];}}
__global__ void tc_quant(const float*x,uint8_t*q,uint8_t*scale,int I){int b=blockIdx.x,i=b*128+threadIdx.x;__shared__ float mx[128];float v=i<I?fabsf(x[i]):0;mx[threadIdx.x]=v;__syncthreads();
    for(int n=64;n;n>>=1){if(threadIdx.x<n&&mx[threadIdx.x+n]>mx[threadIdx.x])mx[threadIdx.x]=mx[threadIdx.x+n];__syncthreads();}float raw=fmaxf(mx[0],1e-4f)/448.f;int e;frexpf(raw,&e);float s=ldexpf(1.f,e-1);if(s<raw)s*=2;int se;frexpf(s,&se);
    if(i<I)q[i]=e4m3_code(fmaxf(-448.f,fminf(448.f,x[i]/s)));if(threadIdx.x<4){int k=b*4+threadIdx.x;scale[tc_sf_off(0,k,(I/32+3)&~3)]=(uint8_t)(se-1+127);}}
__device__ __forceinline__ int compression_ready(const int *state,int ratio);
__global__ void mv_bf16(const __nv_bfloat16 *w,const float *x,float *y,int O,int I){int o=blockIdx.x;if(o>=O)return;float sum=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x)sum+=x[i]*__bfloat162float(w[(long long)o*I+i]);
    __shared__ float r[256];r[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)r[threadIdx.x]+=r[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[o]=r[0];}
__global__ void mv_f32(const float *w,const float *x,float *y,int O,int I){int o=blockIdx.x;if(o>=O)return;float sum=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x)sum+=x[i]*w[(long long)o*I+i];
    __shared__ float r[256];r[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)r[threadIdx.x]+=r[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[o]=r[0];}
__global__ void mhc_fn_split(const float*w,const float*x,float*part,int N,int I,int parts){int p=blockIdx.x,o=blockIdx.y;if(p>=parts||o>=N)return;int begin=(long long)I*p/parts,end=(long long)I*(p+1)/parts;float sum=0,ss=0;
    for(int i=begin+threadIdx.x;i<end;i+=blockDim.x){float v=x[i];sum+=w[(long long)o*I+i]*v;if(!o)ss+=v*v;}__shared__ float r[256],q[256];r[threadIdx.x]=sum;q[threadIdx.x]=ss;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){r[threadIdx.x]+=r[threadIdx.x+n];q[threadIdx.x]+=q[threadIdx.x+n];}__syncthreads();}if(!threadIdx.x){part[o*parts+p]=r[0];if(!o)part[N*parts+p]=q[0];}}
__global__ void mhc_fn_finish(float*out,const float*part,const float*scale,const float*base,int N,int M,int I,float eps,int parts){__shared__ float inv;if(!threadIdx.x){float ss=0;for(int p=0;p<parts;p++)ss+=part[N*parts+p];inv=rsqrtf(ss/I+eps);}__syncthreads();
    int n=threadIdx.x;if(n<N){float v=0;for(int p=0;p<parts;p++)v+=part[n*parts+p];int group=n<M?0:n<2*M?1:2;out[n]=v*inv*scale[group]+base[n];}}
__global__ void mhc_post_fn_split(const float*w,const float*x,const float*residual,const float*state,float*out,float*part,int N,int M,int H,int parts){
    int p=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;if(p>=parts)return;int I=M*H,begin=(long long)I*p/parts,end=(long long)I*(p+1)/parts;
    float acc[24]={},ss=0;const float*post=state,*comb=state+M;
    for(int i=begin+tid;i<end;i+=blockDim.x){int j=i/H,h=i-j*H;float v=post[j]*x[h];
        #pragma unroll
        for(int k=0;k<4;k++)v+=comb[k*M+j]*residual[(long long)k*H+h];
        v=__bfloat162float(__float2bfloat16(v));out[i]=v;ss+=v*v;
        #pragma unroll
        for(int n=0;n<24;n++)acc[n]+=w[(long long)n*I+i]*v;}
    __shared__ float warp_sum[25][8];
    #pragma unroll
    for(int n=0;n<24;n++){float v=acc[n];for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);if(!lane)warp_sum[n][warp]=v;}
    for(int d=16;d;d>>=1)ss+=__shfl_down_sync(0xffffffff,ss,d);if(!lane)warp_sum[24][warp]=ss;__syncthreads();
    if(!warp){if(lane<25){float v=0;for(int q=0;q<8;q++)v+=warp_sum[lane][q];if(lane<24)part[lane*parts+p]=v;else part[24*parts+p]=v;}}
}
__global__ void mhc_fn_split_batch(const float*w,const float*x,float*part,int tokens,int N,int I,int parts){int t=blockIdx.z,p=blockIdx.x,o=blockIdx.y;if(t>=tokens||p>=parts||o>=N)return;const float*xt=x+(long long)t*I;float*out=part+(long long)t*(N+1)*parts;int begin=(long long)I*p/parts,end=(long long)I*(p+1)/parts;float sum=0,ss=0;
    for(int i=begin+threadIdx.x;i<end;i+=blockDim.x){float v=xt[i];sum+=w[(long long)o*I+i]*v;if(!o)ss+=v*v;}__shared__ float r[256],q[256];r[threadIdx.x]=sum;q[threadIdx.x]=ss;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n){r[threadIdx.x]+=r[threadIdx.x+n];q[threadIdx.x]+=q[threadIdx.x+n];}__syncthreads();}if(!threadIdx.x){out[o*parts+p]=r[0];if(!o)out[N*parts+p]=q[0];}}
__global__ void mhc_fn_finish_batch(float*mix,const float*part,const float*scale,const float*base,int N,int I,float eps,int parts){int t=blockIdx.x;const float*src=part+(long long)t*(N+1)*parts;float*dst=mix+(long long)t*N;__shared__ float inv;if(!threadIdx.x){float ss=0;for(int p=0;p<parts;p++)ss+=src[N*parts+p];inv=rsqrtf(ss/I+eps);}__syncthreads();int n=threadIdx.x;if(n<N){float v=0;for(int p=0;p<parts;p++)v+=src[n*parts+p];dst[n]=v*inv*scale[n<4?0:n<8?1:2]+base[n];}}
__global__ void mhc_coeff_batch(float*post,float*comb,const float*mix){int t=blockIdx.x,lane=threadIdx.x&31;const float*m=mix+(long long)t*24;if(lane<4)post[(long long)t*4+lane]=(1.f/(1.f+expf(-m[4+lane])))*2.f;float v=lane<16?m[8+lane]:-INFINITY,mx=v;mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));v=lane<16?expf(v-mx):0.f;float sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);if(lane<16)v=v/sum+1e-6f;for(int it=0;it<20;it++){if(it){sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);if(lane<16)v/=sum+1e-6f;}sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,4);sum+=__shfl_xor_sync(0xffffffff,sum,8);if(lane<16)v/=sum+1e-6f;}if(lane<16)comb[(long long)t*16+lane]=v;}
__global__ void mhc_input_norm_batch(float*out,const float*residual,const float*mix,const float*w,int H){int t=blockIdx.x;const float*pre=mix+(long long)t*24,*r=residual+(long long)t*4*H;float*dst=out+(long long)t*H;__shared__ float ws[8],inv;float ss=0;int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int h=threadIdx.x;h<H;h+=blockDim.x){float v=0;for(int i=0;i<4;i++)v+=(1.f/(1.f+expf(-pre[i]))+1e-6f)*r[(long long)i*H+h];v=__bfloat162float(__float2bfloat16(v));dst[h]=v;ss+=v*v;}for(int d=16;d;d>>=1)ss+=__shfl_down_sync(0xffffffff,ss,d);if(!lane)ws[warp]=ss;__syncthreads();if(!warp){float v=lane<8?ws[lane]:0.f;for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);if(!lane)inv=rsqrtf(v/H+1e-6f);}__syncthreads();for(int h=threadIdx.x;h<H;h+=blockDim.x)dst[h]=__bfloat162float(__float2bfloat16(dst[h]*inv*w[h]));}
__global__ void mhc_input_batch(float*out,const float*residual,const float*mix,int H){int t=blockIdx.x;const float*pre=mix+(long long)t*24,*r=residual+(long long)t*4*H;float*dst=out+(long long)t*H;for(int h=threadIdx.x;h<H;h+=blockDim.x){float v=0;for(int i=0;i<4;i++)v+=(1.f/(1.f+expf(-pre[i]))+1e-6f)*r[(long long)i*H+h];dst[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void mhc_output_batch(float*out,const float*x,const float*residual,const float*post,const float*comb,int H){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;int t=p/(4LL*H),j=(p/H)&3,h=p%H;float v=post[(long long)t*4+j]*x[(long long)t*H+h];for(int k=0;k<4;k++)v+=comb[(long long)t*16+k*4+j]*residual[((long long)t*4+k)*H+h];out[p]=__bfloat162float(__float2bfloat16(v));}
__global__ void expert_act(float *a,const float *g,const float *u,float weight,float limit,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
    float gg=fminf(__bfloat162float(__float2bfloat16(g[i])),limit);
    float uu=fmaxf(-limit,fminf(__bfloat162float(__float2bfloat16(u[i])),limit));
    a[i]=__bfloat162float(__float2bfloat16(weight*(gg/(1.f+expf(-gg)))*uu));}}
__global__ void expert_act_grouped(float *a,const float *g,const float *u,const float *weight,float limit,int rows,int n){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<rows*n){int row=p/n;float gg=fminf(__bfloat162float(__float2bfloat16(g[p])),limit);
        float uu=fmaxf(-limit,fminf(__bfloat162float(__float2bfloat16(u[p])),limit));
        a[p]=__bfloat162float(__float2bfloat16(weight[row]*(gg/(1.f+expf(-gg)))*uu));}}
__device__ float e4m3_round(float x){int best=0;float bd=INFINITY;for(int v=0;v<255;v++){float q=e4m3((uint8_t)v);if(isnan(q))continue;
    float d=fabsf(q-x);if(d<bd||(d==bd&&!(v&1)&&(best&1))){bd=d;best=v;}}return e4m3((uint8_t)best);}
__global__ void fp8_sim(float *x,int rows,int n){int row=blockIdx.x/((n+127)/128),b=blockIdx.x%((n+127)/128),i=b*128+threadIdx.x;if(row>=rows)return;
    float v=i<n?fabsf(x[(long long)row*n+i]):0.f;__shared__ float mx[128];mx[threadIdx.x]=v;__syncthreads();
    for(int k=64;k;k>>=1){if(threadIdx.x<k&&mx[threadIdx.x+k]>mx[threadIdx.x])mx[threadIdx.x]=mx[threadIdx.x+k];__syncthreads();}
    if(i<n){float m=fmaxf(mx[0],1e-4f),raw=m/448.f;int e;frexpf(raw,&e);float scale=ldexpf(1.f,e-1);if(scale<raw)scale*=2.f;
        float q=fmaxf(-448.f,fminf(448.f,x[(long long)row*n+i]/scale));x[(long long)row*n+i]=__bfloat162float(__float2bfloat16(e4m3_round(q)*scale));}}
__global__ void fp8_sim64(float *x,int n){int i=blockIdx.x*64+threadIdx.x;__shared__ float mx[64];float v=i<n?fabsf(x[i]):0;mx[threadIdx.x]=v;__syncthreads();
    for(int k=32;k;k>>=1){if(threadIdx.x<k&&mx[threadIdx.x+k]>mx[threadIdx.x])mx[threadIdx.x]=mx[threadIdx.x+k];__syncthreads();}if(i<n){float m=fmaxf(mx[0],1e-4f),raw=m/448.f;int e;frexpf(raw,&e);float scale=ldexpf(1.f,e-1);uint8_t q=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[i]/scale)),__NV_SATFINITE,__NV_E4M3);x[i]=__bfloat162float(__float2bfloat16(e4m3(q)*scale));}}
/* Batched router (generic CUDA; used by both builds). */
__device__ __forceinline__ bool route_better(float,unsigned,float,unsigned);__device__ __forceinline__ float route_prob(float);
__global__ void route_logits_batch(const float*w,const float*x,float*y,int tokens,int E,int H){int e=blockIdx.x% E,t=blockIdx.x/E;float sum=0;const float*in=x+(long long)t*H;for(int h=threadIdx.x;h<H;h+=blockDim.x)sum+=w[(long long)e*H+h]*in[h];__shared__ float s[256];s[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)s[threadIdx.x]+=s[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[(long long)t*E+e]=s[0];(void)tokens;}
__global__ void route_top6_batch(int*ids,float*weights,const float*bias,const float*logits,int tokens,float scale){int t=blockIdx.x,lane=threadIdx.x;if(t>=tokens||lane>=32)return;const float*src=logits+(long long)t*256;float prob[8],score[8];for(int j=0;j<8;j++){int e=lane+j*32;prob[j]=route_prob(src[e]);score[j]=prob[j]+(bias?bias[e]:0.f);}float outp[6]={};unsigned outi[6]={};for(int k=0;k<6;k++){float bs=-INFINITY,bp=0;unsigned bi=UINT_MAX;for(int j=0;j<8;j++){unsigned e=lane+j*32;if(route_better(score[j],e,bs,bi)){bs=score[j];bp=prob[j];bi=e;}}for(int m=16;m;m>>=1){float os=__shfl_xor_sync(0xffffffff,bs,m),op=__shfl_xor_sync(0xffffffff,bp,m);unsigned oi=__shfl_xor_sync(0xffffffff,bi,m);if(route_better(os,oi,bs,bi)){bs=os;bp=op;bi=oi;}}for(int j=0;j<8;j++)if((unsigned)(lane+j*32)==bi)score[j]=-INFINITY;if(!lane){outp[k]=bp;outi[k]=bi;}}if(!lane){float sum=0;for(int k=0;k<6;k++)sum+=outp[k];sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++){ids[(long long)t*6+k]=(int)outi[k];weights[(long long)t*6+k]=outp[k]/sum*scale;}}}
#ifdef COLI_DSV4_DEEPGEMM
__global__ void dg_pack_weight_scale(int*out,const uint8_t*in,int expert,int N,int groups){
    int p=blockIdx.x*blockDim.x+threadIdx.x,total=N*(groups/4);if(p>=total)return;int n=p/(groups/4),kg=p%(groups/4);uint32_t v=0;
    for(int b=0;b<4;b++)v|=(uint32_t)in[(long long)n*groups+kg*4+b]<<(8*b);out[((long long)expert*(groups/4)+kg)*N+n]=(int)v;
}
__global__ void dg_pack_dense_scale(int*out,const uint8_t*in,int N,int groups){int p=blockIdx.x*blockDim.x+threadIdx.x,total=N*(groups/4);if(p>=total)return;
    int n=p/(groups/4),kg=p%(groups/4);uint32_t v=0;for(int b=0;b<4;b++)v|=(uint32_t)in[(long long)(n/128)*groups+kg*4+b]<<(8*b);out[(long long)kg*N+n]=(int)v;}
__global__ void dg_pack_batched_scale(int*out,const uint8_t*in,int N,int groups,int batches){int p=blockIdx.x*blockDim.x+threadIdx.x,per=N*(groups/4),total=batches*per;if(p>=total)return;
    int batch=p/per,r=p%per,n=r/(groups/4),kg=r%(groups/4);uint32_t v=0;for(int b=0;b<4;b++)v|=(uint32_t)in[(long long)(batch*N+n)/128*groups+kg*4+b]<<(8*b);out[((long long)batch*(groups/4)+kg)*N+n]=(int)v;}
__device__ __forceinline__ uint8_t dg_scale_code(float m){float raw=fmaxf(m,1e-4f)/448.f;int e;frexpf(raw,&e);float s=ldexpf(1.f,e-1);if(s<raw){s*=2.f;e++;}return (uint8_t)(e-1+127);}
__device__ __forceinline__ float dg_scale_value(uint8_t code){return ldexpf(1.f,(int)code-127);}
__global__ void dg_quant_dense(const float*x,uint8_t*q,int*sfa,int K){int pack=blockIdx.x;if(pack>=K/512)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[pack*4]=(int)codes;}
__global__ void dg_quant_dense_batch(const float*x,uint8_t*q,int*sfa,int rows,int aligned_rows,int K){int row=blockIdx.x/(K/512),pack=blockIdx.x%(K/512);if(row>=rows)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[(long long)row*K+k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[(long long)row*K+k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[(long long)pack*aligned_rows+row]=(int)codes;}
__global__ void dg_quant_batched8(const float*x,uint8_t*q,int*sfa,int K){int batch=blockIdx.x/(K/512),pack=blockIdx.x%(K/512);if(batch>=8)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[(long long)batch*K+k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)batch*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[(long long)batch*K+k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[((long long)batch*(K/512)+pack)*4]=(int)codes;}
__global__ void dg_bf16_to_float(float*out,const __nv_bfloat16*in,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__bfloat162float(in[i]);}
__global__ void dg_float_to_bf16(__nv_bfloat16*out,const float*in,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__float2bfloat16(in[i]);}
__global__ void dg_argmax_bf16(const __nv_bfloat16*x,int n,int*out_id,float*out_value){__shared__ float value[256];__shared__ int id[256];float best=-INFINITY;int bi=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float v=__bfloat162float(x[i]);if(v>best||(v==best&&i<bi)){best=v;bi=i;}}value[threadIdx.x]=best;id[threadIdx.x]=bi;__syncthreads();
    for(int k=128;k;k>>=1){if(threadIdx.x<k){float v=value[threadIdx.x+k];int i=id[threadIdx.x+k];if(v>value[threadIdx.x]||(v==value[threadIdx.x]&&i<id[threadIdx.x])){value[threadIdx.x]=v;id[threadIdx.x]=i;}}__syncthreads();}if(!threadIdx.x){*out_id=id[0];*out_value=value[0];}}
__global__ void dg_layout6(int*out,const int*ids,int rows,int align){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows)out[i]=ids[i/align];}
__global__ void dg_layout6_ep(int*out,const int*ids,int rows,int align,int base){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows){int id=ids[i/align]-base;out[i]=id>=0&&id<128?id:-1;}}
__global__ void dg_quant_replicate(const float*x,uint8_t*q,int*sfa,int K){int row=blockIdx.x;if(row>=768)return;__shared__ float mx[128];uint32_t pack=0;int groups=K/128;
    for(int g=0;g<groups;g++){int k=g*128+threadIdx.x;float v=fabsf(x[k]);mx[threadIdx.x]=v;__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[k]/scale)),__NV_SATFINITE,__NV_E4M3);
        if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*768+row]=(int)pack;pack=0;}}__syncthreads();}
}
__global__ void dg_quant_one(const float*x,uint8_t*q,int*sfa,int K,int rows){int pack=blockIdx.x;if(pack>=K/512)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[(long long)pack*rows]=(int)codes;}
__global__ void dg_replicate_rows(uint8_t*q,int*sfa,int K,int rows){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)rows*K;if(p<n){int row=p/K,k=p%K;q[p]=q[k];}
    int z=(int)p,total=rows*(K/512);if(z<total){int row=z/(K/512),pack=z%(K/512);sfa[(long long)pack*rows+row]=sfa[(long long)pack*rows];}}
__global__ void dg_swiglu_quant(const __nv_bfloat16*in,uint8_t*q,int*sfa,const float*weight,float limit){int row=blockIdx.x;if(row>=768)return;__shared__ float mx[128],value[128];uint32_t pack=0;int expert=row/128;
    for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));
        float v=__bfloat162float(__float2bfloat16(weight[expert]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();
        for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);
        q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);
        if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*768+row]=(int)pack;pack=0;}}__syncthreads();}
}
__global__ void dg_swiglu_quant_unique(const __nv_bfloat16*in,uint8_t*q,int*sfa,const float*weight,float limit,int rows,int align){int expert=blockIdx.x;if(expert>=6)return;int row=expert*align;__shared__ float mx[128],value[128];uint32_t pack=0;
    for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));
        float v=__bfloat162float(__float2bfloat16(weight[expert]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_swiglu_quant_unique_ep(const __nv_bfloat16*in,uint8_t*q,int*sfa,const int*ids,int base,float limit,int rows,int align){int slot=blockIdx.x;if(slot>=6)return;int id=ids[slot]-base;if(id<0||id>=128)return;int row=slot*align;__shared__ float mx[128],value[128];uint32_t pack=0;
    for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));
        float v=__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_swiglu_quant_unique_tp2(const __nv_bfloat16*in,uint8_t*q,int*sfa,float limit,int rows,int align){int slot=blockIdx.x;if(slot>=6)return;int row=slot*align;__shared__ float mx[128],value[128];uint32_t pack=0;
    for(int g=0;g<8;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*2048+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*2048+1024+k]),limit));
        float v=__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*1024+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_replicate_expert_rows(uint8_t*q,int*sfa,int K,int rows,int align){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)rows*K;if(p<n){int row=p/K,k=p%K,base=(row/align)*align;q[p]=q[(long long)base*K+k];}
    int z=(int)p,total=rows*(K/512);if(z<total){int row=z/(K/512),pack=z%(K/512),base=(row/align)*align;sfa[(long long)pack*rows+row]=sfa[(long long)pack*rows+base];}}
__global__ void dg_reduce6(float*out,const __nv_bfloat16*parts,int align){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<4096){float v=0;for(int e=0;e<6;e++)v+=__bfloat162float(parts[(long long)e*align*4096+h]);out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void dg_reduce6_ep(float*out,const __nv_bfloat16*parts,const int*ids,const float*weights,int base,int align){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<4096){float v=0;for(int e=0;e<6;e++){int id=ids[e]-base;if(id>=0&&id<128)v+=weights[e]*__bfloat162float(parts[(long long)e*align*4096+h]);}out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void dg_mask6(int*mask,const int*ids,int groups){int i=threadIdx.x;if(i<groups)mask[i]=0;__syncthreads();if(i<6&&ids[i]>=0)mask[ids[i]]=1;}
__global__ void dg_quant_masked(const float*x,uint8_t*q,int*sfa,const int*ids,int K,int maxm){int slot=blockIdx.x/(K/512),pack=blockIdx.x%(K/512),id=ids[slot];if(id<0)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[((long long)id*maxm)*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[((long long)id*(K/512)+pack)*maxm]=(int)codes;}
__global__ void dg_swiglu_quant_masked(const __nv_bfloat16*in,uint8_t*q,int*sfa,const int*ids,const float*weight,float limit,int maxm){int slot=blockIdx.x,id=ids[slot];if(id<0)return;long long row=(long long)id*maxm;__shared__ float mx[128],value[128];uint32_t pack=0;
    for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[row*4096+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[row*4096+2048+k]),limit));
        float v=__bfloat162float(__float2bfloat16(weight[slot]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[((long long)id*4+g/4)*maxm]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_reduce6_masked(float*out,const __nv_bfloat16*parts,const int*ids,int maxm){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<4096){float v=0;for(int e=0;e<6;e++)if(ids[e]>=0)v+=__bfloat162float(parts[((long long)ids[e]*maxm)*4096+h]);out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void dg_swiglu_quant_rows(const __nv_bfloat16*in,uint8_t*q,int*sfa,float limit,int rows){int row=blockIdx.x,warp=threadIdx.x>>5,lane=threadIdx.x&31;__shared__ uint8_t code[16];
    for(int phase=0;phase<2;phase++){int group=phase*8+warp,base=group*128;float value[4],mx=0;
        #pragma unroll
        for(int j=0;j<4;j++){int k=base+lane+j*32;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit),up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));value[j]=__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up));mx=fmaxf(mx,fabsf(value[j]));}
        for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));uint8_t c=dg_scale_code(mx),cc=__shfl_sync(0xffffffff,c,0);float scale=dg_scale_value(cc);if(!lane)code[group]=cc;
        #pragma unroll
        for(int j=0;j<4;j++){int k=base+lane+j*32;q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[j]/scale)),__NV_SATFINITE,__NV_E4M3);}}
    __syncthreads();if(threadIdx.x<4){int g=threadIdx.x*4;sfa[(long long)threadIdx.x*rows+row]=(int)((uint32_t)code[g]|((uint32_t)code[g+1]<<8)|((uint32_t)code[g+2]<<16)|((uint32_t)code[g+3]<<24));}}
__global__ void dg_reduce6_weighted(float*out,const __nv_bfloat16*parts,const float*weight,int align){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<4096){float v=0;for(int e=0;e<6;e++)v+=weight[e]*__bfloat162float(parts[(long long)e*align*4096+h]);out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void route_hash_batch(int*ids,const int64_t*map,const int*tokens,int rows){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<rows*6)ids[p]=(int)map[(long long)tokens[p/6]*6+p%6];}
__global__ void route_fixed_weights_batch(float*weights,const int*ids,const float*logits,int tokens,float scale){int t=blockIdx.x,k=threadIdx.x;__shared__ float p[6];if(t<tokens&&k<6)p[k]=route_prob(logits[(long long)t*256+ids[(long long)t*6+k]]);__syncthreads();if(t<tokens&&!k){float sum=0;for(int i=0;i<6;i++)sum+=p[i];sum=fmaxf(sum,6.103515625e-5f);for(int i=0;i<6;i++)weights[(long long)t*6+i]=p[i]/sum*scale;}}
__global__ void dg_layout_routes(int*out,const int*ids,int rows,int align){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows)out[i]=ids[i/align];}
__global__ void dg_quant_routes(const float*x,uint8_t*q,int*sfa,int tokens,int K,int rows,int align){int route=blockIdx.x/(K/512),pack=blockIdx.x%(K/512);if(route>=tokens*6)return;int token=route/6,row=route*align;__shared__ float mx[128];uint32_t codes=0;for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;float v=x[(long long)token*K+k];mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,v/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}if(!threadIdx.x)sfa[(long long)pack*rows+row]=(int)codes;}
__global__ void dg_swiglu_quant_routes(const __nv_bfloat16*in,uint8_t*q,int*sfa,const float*weight,float limit,int routes,int rows,int align){int route=blockIdx.x;if(route>=routes)return;int row=route*align;__shared__ float mx[128],value[128];uint32_t pack=0;for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit),up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));float v=__bfloat162float(__float2bfloat16(weight[route]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
/* EXPERT-GROUPED variants: routes are laid out contiguously per expert (host
 * builds rowmap[route] -> row and layout[row] -> expert, groups 64-aligned,
 * pad rows zeroed). Each route's row computes exactly what its old 64x
 * replicated row did, so results are bitwise the replicated layout with
 * ~4x fewer GEMM rows. */
__global__ void dg_quant_routes_map(const float*x,uint8_t*q,int*sfa,int tokens,int K,int rows,const int*rowmap){int route=blockIdx.x/(K/512),pack=blockIdx.x%(K/512);if(route>=tokens*6)return;int token=route/6,row=rowmap[route];__shared__ float mx[128];uint32_t codes=0;for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;float v=x[(long long)token*K+k];mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,v/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}if(!threadIdx.x)sfa[(long long)pack*rows+row]=(int)codes;}
__global__ void dg_swiglu_quant_routes_map(const __nv_bfloat16*in,uint8_t*q,int*sfa,const float*weight,float limit,int routes,int rows,const int*rowmap){int route=blockIdx.x;if(route>=routes)return;int row=rowmap[route];__shared__ float mx[128],value[128];uint32_t pack=0;for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit),up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));float v=__bfloat162float(__float2bfloat16(weight[route]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_reduce_routes_map(float*out,const __nv_bfloat16*parts,int tokens,const int*rowmap){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)tokens*4096;if(p<n){int t=p/4096,h=p%4096;float v=0;for(int k=0;k<6;k++)v+=__bfloat162float(parts[(long long)rowmap[t*6+k]*4096+h]);out[p]=__bfloat162float(__float2bfloat16(v));}}
__global__ void dg_reduce_routes(float*out,const __nv_bfloat16*parts,int tokens,int align){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)tokens*4096;if(p<n){int t=p/4096,h=p%4096;float v=0;for(int k=0;k<6;k++)v+=__bfloat162float(parts[((long long)(t*6+k)*align)*4096+h]);out[p]=__bfloat162float(__float2bfloat16(v));}}
__global__ void swiglu_rows(float*out,const float*gate,const float*up,float limit,int rows,int I){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)rows*I;if(p<n){float g=fminf(gate[p],limit),u=fmaxf(-limit,fminf(up[p],limit));out[p]=__bfloat162float(__float2bfloat16((g/(1.f+expf(-g)))*u));}}
__global__ void add_rows(float*out,const float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__bfloat162float(__float2bfloat16(out[i]+x[i]));}
__global__ void dg_compare(const float*ref,const float*got,float*stats,int n){__shared__ float ma[256],sa[256];float mx=0,sum=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float d=fabsf(ref[i]-got[i]);mx=fmaxf(mx,d);sum+=d;}ma[threadIdx.x]=mx;sa[threadIdx.x]=sum;__syncthreads();
    for(int k=128;k;k>>=1){if(threadIdx.x<k){ma[threadIdx.x]=fmaxf(ma[threadIdx.x],ma[threadIdx.x+k]);sa[threadIdx.x]+=sa[threadIdx.x+k];}__syncthreads();}
    if(!threadIdx.x){stats[0]=ma[0];stats[1]=sa[0]/n;}}
__global__ void dg_compare_fc1(const float*gate,const float*up,const __nv_bfloat16*got,float*stats){__shared__ float ma[256],sa[256];float mx=0,sum=0;
    for(int i=threadIdx.x;i<6*4096;i+=blockDim.x){int e=i/4096,n=i%4096;float ref=n<2048?gate[(long long)e*2048+n]:up[(long long)e*2048+n-2048];float d=fabsf(ref-__bfloat162float(got[(long long)e*128*4096+n]));mx=fmaxf(mx,d);sum+=d;}
    ma[threadIdx.x]=mx;sa[threadIdx.x]=sum;__syncthreads();for(int k=128;k;k>>=1){if(threadIdx.x<k){ma[threadIdx.x]=fmaxf(ma[threadIdx.x],ma[threadIdx.x+k]);sa[threadIdx.x]+=sa[threadIdx.x+k];}__syncthreads();}if(!threadIdx.x){stats[0]=ma[0];stats[1]=sa[0]/(6*4096);}}
__global__ void dg_compare_fc2(const float*ref,const __nv_bfloat16*got,float*stats){__shared__ float ma[256],sa[256];float mx=0,sum=0;
    for(int i=threadIdx.x;i<6*4096;i+=blockDim.x){int e=i/4096,n=i%4096;float d=fabsf(ref[i]-__bfloat162float(got[(long long)e*128*4096+n]));mx=fmaxf(mx,d);sum+=d;}
    ma[threadIdx.x]=mx;sa[threadIdx.x]=sum;__syncthreads();for(int k=128;k;k>>=1){if(threadIdx.x<k){ma[threadIdx.x]=fmaxf(ma[threadIdx.x],ma[threadIdx.x+k]);sa[threadIdx.x]+=sa[threadIdx.x+k];}__syncthreads();}if(!threadIdx.x){stats[0]=ma[0];stats[1]=sa[0]/(6*4096);}}

static int dg_driver(CUresult e,const char*what){if(e==CUDA_SUCCESS)return 1;const char*msg=nullptr;cuGetErrorString(e,&msg);fprintf(stderr,"[DSV4 DeepGEMM] %s: %s\n",what,msg?msg:"CUDA driver error");return 0;}
static int dg_map(CUtensorMap*map,CUtensorMapDataType type,void*ptr,unsigned long long inner,unsigned long long outer,unsigned long long stride,unsigned box_inner,unsigned box_outer,CUtensorMapSwizzle swizzle){
    const cuuint64_t dims[2]={inner,outer},strides[1]={stride};const cuuint32_t box[2]={box_inner,box_outer},elem[2]={1,1};
    CUresult e=cuTensorMapEncodeTiled(map,type,2,ptr,dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(e!=CUDA_SUCCESS)fprintf(stderr,"[DSV4 DeepGEMM] bad TMA type=%d ptr=%p dims=%llux%llu stride=%llu box=%ux%u swizzle=%d\n",(int)type,ptr,inner,outer,stride,box_inner,box_outer,(int)swizzle);
    return dg_driver(e,"encode TMA map");
}
static int dg_map_batched(CUtensorMap*map,CUtensorMapDataType type,void*ptr,unsigned long long inner,unsigned long long outer,unsigned long long batches,unsigned long long outer_stride,unsigned long long batch_stride,unsigned box_inner,unsigned box_outer,CUtensorMapSwizzle swizzle){
    const cuuint64_t dims[3]={inner,outer,batches},strides[2]={outer_stride,batch_stride};const cuuint32_t box[3]={box_inner,box_outer,1},elem[3]={1,1,1};
    return dg_driver(cuTensorMapEncodeTiled(map,type,3,ptr,dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"encode batched TMA map");
}
using DgOutput=cutlass::bfloat16_t;using DgEpilogue=deep_gemm::epilogue::transform::EpilogueIdentity;
static int dg_dense_2048x4096(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){constexpr int O=2048,I=4096;
    if(!t||t->O!=O||t->I!=I||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,I)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,(I/512)*4*sizeof(int))||
       !buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||!buf((void**)&c->dgdensews,&c->dgdensewscap,(size_t)4*O*sizeof(float)))return 0;
    dg_quant_dense<<<I/512,128,0,c->stream>>>(x,c->dga1,c->dgsfa1,I);CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,I,O,I,64,64,CU_TENSOR_MAP_SWIZZLE_64B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,I,1,I,64,16,CU_TENSOR_MAP_SWIZZLE_64B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,O,I/512,(unsigned long long)O*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,4,I/512,16,16,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,16,64,CU_TENSOR_MAP_SWIZZLE_NONE))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,64,16,64,64,64,0,16,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,64,4>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,64,16,64,64,64,0,16,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,64,4>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,88320),"dense DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,88320,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,c->dgdensews,O,1,I,1,O,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    auto reduce=deep_gemm::sm120_split_k_reduce_impl<DgOutput,4>;reduce<<<(O+255)/256,256,0,c->stream>>>((DgOutput*)c->dgmm1,c->dgdensews,O,1,1,O);
    dg_bf16_to_float<<<(O+255)/256,256,0,c->stream>>>(y,c->dgmm1,O);
    static int dumped;if(!dumped++&&getenv("DSV4_CUDA_DENSE_DUMP")){struct Item{const char*p;const void*d;size_t n;}items[]={{"/tmp/dsv4-dense-x.f32",x,I*4},{"/tmp/dsv4-dense-q.fp8",c->dga1,I},{"/tmp/dsv4-dense-sfa.bin",c->dgsfa1,(I/512)*4*sizeof(int)},{"/tmp/dsv4-dense-sfb.bin",t->dg_scale,(size_t)O*(I/512)*sizeof(int)},{"/tmp/dsv4-dense-out.f32",y,O*4}};
        if(!ok(cudaStreamSynchronize(c->stream),"dense dump sync"))return 0;for(unsigned z=0;z<sizeof(items)/sizeof(*items);z++){void*h=malloc(items[z].n);FILE*f;if(!h||!ok(cudaMemcpy(h,items[z].d,items[z].n,cudaMemcpyDeviceToHost),"dense dump copy")||!(f=fopen(items[z].p,"wb"))||fwrite(h,1,items[z].n,f)!=items[z].n){free(h);return 0;}fclose(f);free(h);}}
    return ok(cudaGetLastError(),"dense DeepGEMM launch");}
static int dg_dense_4096x2048(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){constexpr int O=4096,I=2048;
    if(!t||t->O!=O||t->I!=I||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,I)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,(I/512)*4*sizeof(int))||
       !buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||!buf((void**)&c->dgdensews,&c->dgdensewscap,(size_t)4*O*sizeof(float)))return 0;
    dg_quant_dense<<<I/512,128,0,c->stream>>>(x,c->dga1,c->dgsfa1,I);CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,I,O,I,64,64,CU_TENSOR_MAP_SWIZZLE_64B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,I,1,I,64,16,CU_TENSOR_MAP_SWIZZLE_64B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,O,I/512,(unsigned long long)O*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,4,I/512,16,16,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,16,64,CU_TENSOR_MAP_SWIZZLE_NONE))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,64,16,64,64,64,0,16,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,64,4>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,64,16,64,64,64,0,16,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,64,4>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,88320),"dense DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,88320,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,c->dgdensews,O,1,I,1,O,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    auto reduce=deep_gemm::sm120_split_k_reduce_impl<DgOutput,4>;reduce<<<(O+255)/256,256,0,c->stream>>>((DgOutput*)c->dgmm1,c->dgdensews,O,1,1,O);
    dg_bf16_to_float<<<(O+255)/256,256,0,c->stream>>>(y,c->dgmm1,O);return ok(cudaGetLastError(),"dense DeepGEMM launch");}
template<int O,int I,int BM,int BK,int SW,int STAGES,int EPI,int SPLIT,bool QUANT=true>
static int dg_dense_attention(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){
    constexpr int SMEM=SW==128?95872:98816;
    constexpr CUtensorMapSwizzle SWIZZLE=SW==128?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B;
    if(!t||t->O!=O||t->I!=I||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,I)||
       !buf((void**)&c->dgsfa1,&c->dgsfa1cap,(I/512)*4*sizeof(int))||
       !buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||
       (SPLIT>1&&!buf((void**)&c->dgdensews,&c->dgdensewscap,(size_t)SPLIT*O*sizeof(float)))){fprintf(stderr,"[DSV4 DG] attention buffer/shape failed %dx%d\n",O,I);return 0;}
    if constexpr(QUANT)dg_quant_dense<<<I/512,128,0,c->stream>>>(x,c->dga1,c->dgsfa1,I);CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,I,O,I,BK,BM,SWIZZLE)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,I,1,I,BK,16,SWIZZLE)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,O,I/512,(unsigned long long)O*4,BM,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,4,I/512,16,16,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,16,BM,CU_TENSOR_MAP_SWIZZLE_NONE)){fprintf(stderr,"[DSV4 DG] attention TMA map failed %dx%d\n",O,I);return 0;}
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,BM,16,BK,SW,SW,0,STAGES,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,EPI,SPLIT>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,BM,16,BK,SW,SW,0,STAGES,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,EPI,SPLIT>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM),"attention DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,SMEM,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,SPLIT>1?c->dgdensews:nullptr,O,1,I,1,O,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if constexpr(SPLIT>1){auto reduce=deep_gemm::sm120_split_k_reduce_impl<DgOutput,SPLIT>;reduce<<<(O+255)/256,256,0,c->stream>>>((DgOutput*)c->dgmm1,c->dgdensews,O,1,1,O);}
    dg_bf16_to_float<<<(O+255)/256,256,0,c->stream>>>(y,c->dgmm1,O);return ok(cudaGetLastError(),"attention DeepGEMM launch");
}
template<int O,int I,int BM,int BN,int BK,int SW,int STAGES,int EPI>
static int dg_dense_batch(Dev*c,Dsv4CudaTensor*t,const float*x,float*y,int tokens){
    constexpr int SMEM=88320;int aligned_tokens=(tokens+BN-1)/BN*BN;constexpr CUtensorMapSwizzle SWIZZLE=SW==128?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B;
    if(!t||tokens<1||t->O!=O||t->I!=I||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,(size_t)tokens*I)||
       !buf((void**)&c->dgsfa1,&c->dgsfa1cap,(size_t)aligned_tokens*(I/512)*sizeof(int))||
       !buf((void**)&c->dgmm1,&c->dgmm1cap,(size_t)tokens*O*sizeof(__nv_bfloat16)))return 0;
    dg_quant_dense_batch<<<tokens*(I/512),128,0,c->stream>>>(x,c->dga1,c->dgsfa1,tokens,aligned_tokens,I);CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,I,O,I,BK,BM,SWIZZLE)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,I,tokens,I,BK,BN,SWIZZLE)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,O,I/512,(unsigned long long)O*4,BM,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,aligned_tokens,I/512,(unsigned long long)aligned_tokens*4,BN,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,tokens,(unsigned long long)O*2,16,BM,CU_TENSOR_MAP_SWIZZLE_NONE))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,0,I,128,128,1,BM,BN,BK,SW,SW,0,STAGES,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,EPI,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,0,I,128,128,1,BM,BN,BK,SW,SW,0,STAGES,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,EPI,1>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM),"batched dense DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,SMEM,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,nullptr,O,tokens,I,1,O,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    dg_bf16_to_float<<<((long long)tokens*O+255)/256,256,0,c->stream>>>(y,c->dgmm1,tokens*O);return ok(cudaGetLastError(),"batched dense DeepGEMM launch");
}
static int dg_dense_batch_dispatch(Dev*c,Dsv4CudaTensor*t,const float*x,float*y,int tokens){
    if(t->O==1536&&t->I==4096)return dg_dense_batch<1536,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==1024&&t->I==4096)return dg_dense_batch<1024,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==512&&t->I==4096)return dg_dense_batch<512,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==2048&&t->I==4096)return dg_dense_batch<2048,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==4096&&t->I==4096)return dg_dense_batch<4096,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==4096&&t->I==8192)return dg_dense_batch<4096,8192,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==4096&&t->I==1024)return dg_dense_batch<4096,1024,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==4096&&t->I==2048)return dg_dense_batch<4096,2048,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==16384&&t->I==1024)return dg_dense_batch<16384,1024,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==32768&&t->I==1024)return dg_dense_batch<32768,1024,64,16,64,64,16,64>(c,t,x,y,tokens);
    return 0;
}
static int dg_attention_kv_reuse(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){(void)x;return t&&t->O==512&&t->I==4096&&dg_dense_attention<512,4096,64,128,128,9,64,4,false>(c,t,x,y);}
static int dg_attention_projection(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){
    int good=0;
    if(t->I==4096&&(t->O==1536||t->O==1024||t->O==512)){
        good=t->O==1536?dg_dense_attention<1536,4096,64,128,128,9,64,4>(c,t,x,y):
             t->O==1024?dg_dense_attention<1024,4096,64,128,128,9,64,4>(c,t,x,y):
                         dg_dense_attention<512,4096,64,128,128,9,64,4>(c,t,x,y);
    }else if(t->O==32768&&t->I==1024)good=dg_dense_attention<32768,1024,128,64,64,10,128,1>(c,t,x,y);
    else if(t->O==16384&&t->I==1024)good=dg_dense_attention<16384,1024,128,64,64,10,128,1>(c,t,x,y);
    else if(t->O==4096&&t->I==8192)good=dg_dense_attention<4096,8192,128,64,64,10,128,1>(c,t,x,y);
    else if(t->O==4096&&t->I==4096)good=dg_dense_attention<4096,4096,128,64,64,10,128,1>(c,t,x,y);
    else if(t->O==4096&&t->I==1024)good=dg_dense_attention<4096,1024,128,64,64,10,128,1>(c,t,x,y);
    if(!good)return 0;
    static int checks;
    if(checks<4&&getenv("DSV4_CUDA_ATTN_DG_AB")&&atoi(getenv("DSV4_CUDA_ATTN_DG_AB"))){
        checks++;float hs[2];
        if(!buf((void**)&c->dgref,&c->dgrefcap,(size_t)t->I*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(size_t)t->O*sizeof(float))||
           !ok(cudaMemcpyAsync(c->dgref,x,(size_t)t->I*sizeof(float),cudaMemcpyDeviceToDevice,c->stream),"attention A/B input copy"))return 0;
        fp8_sim<<<(t->I+127)/128,128,0,c->stream>>>(c->dgref,1,t->I);
        run_mv<8>((uint8_t*)t->w,t->scale,c->dgref,c->dy,t->O,t->I,1,c->stream);
        dg_compare<<<1,256,0,c->stream>>>(c->dy,y,c->dgref,t->O);
        if(!ok(cudaMemcpyAsync(hs,c->dgref,sizeof(hs),cudaMemcpyDeviceToHost,c->stream),"attention A/B stats")||
           !ok(cudaStreamSynchronize(c->stream),"attention A/B sync"))return 0;
        fprintf(stderr,"[DSV4 attention DeepGEMM A/B] %dx%d max=%.6g mae=%.6g\n",t->O,t->I,hs[0],hs[1]);
    }
    return 1;
}
template<int G> static int dg_attention_wo_a_grouped(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){constexpr int N=1024,K=4096,O=G*N;
    if(!t||t->fmt!=9||t->O!=O||t->I!=K||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,(size_t)G*K)||
       !buf((void**)&c->dgsfa1,&c->dgsfa1cap,(size_t)G*(K/512)*4*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,(size_t)O*sizeof(__nv_bfloat16)))return 0;
    dg_quant_batched8<<<G*(K/512),128,0,c->stream>>>(x,c->dga1,c->dgsfa1,K);CUtensorMap a,b,sa,sb,d;
    if(!dg_map_batched(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,K,N,G,K,(unsigned long long)K*N,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map_batched(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,K,1,G,K,K,128,16,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,N,G*(K/512),(unsigned long long)N*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,4,G*(K/512),16,16,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N,1,(unsigned long long)N*2,16,64,CU_TENSOR_MAP_SWIZZLE_NONE))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,K,128,128,G,64,16,128,128,128,0,9,128,256,170,deep_gemm::GemmType::Batched,false,DgOutput,DgEpilogue,false,false,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,K,128,128,G,64,16,128,128,128,0,9,128,256,84,deep_gemm::GemmType::Batched,false,DgOutput,DgEpilogue,false,false,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,95872),"wo_a DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,95872,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,nullptr,N,1,K,1,N,N,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    dg_bf16_to_float<<<(O+255)/256,256,0,c->stream>>>(y,c->dgmm1,O);return ok(cudaGetLastError(),"wo_a DeepGEMM launch");
}
static int dg_attention_wo_a(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){
    return t&&t->O==8192?dg_attention_wo_a_grouped<8>(c,t,x,y):t&&t->O==4096?dg_attention_wo_a_grouped<4>(c,t,x,y):0;
}
template<int G> static int dg_moe(Dev*c,Dsv4CudaExpertSet*set,const float*weight,float limit,float*out){constexpr int M=16,N=4096;
    size_t a1=(size_t)G*M*4096,a2=(size_t)G*M*2048,s1=(size_t)G*M*8*4,s2=(size_t)G*M*4*4,mm=(size_t)G*M*N*2;
    if(!buf((void**)&c->dga1,&c->dga1cap,a1)||!buf((void**)&c->dga2,&c->dga2cap,a2)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,s1)||!buf((void**)&c->dgsfa2,&c->dgsfa2cap,s2)||
       !buf((void**)&c->dglayout,&c->dglayoutcap,G*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,mm)||!buf((void**)&c->dgmm2,&c->dgmm2cap,mm))return 0;
    dg_mask6<<<1,256,0,c->stream>>>(c->dglayout,c->expert_ids,G);dg_quant_masked<<<6*8,128,0,c->stream>>>(c->dx,c->dga1,c->dgsfa1,c->expert_ids,4096,M);
    CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,4096,(unsigned long long)G*M,4096,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc1,4096,(unsigned long long)G*N,2048,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,M,(unsigned long long)G*8,(unsigned long long)M*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc1_dg_scale,N,(unsigned long long)G*32,16384,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N,(unsigned long long)G*M,(unsigned long long)N*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc1=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,128,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedMasked,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,128,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedMasked,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"DeepGEMM FC1 shared memory"))return 0;
    fc1<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)c->dga1,(__nv_fp8_e4m3*)set->bank_fc1,c->dglayout,nullptr,nullptr,M,N,4096,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    int debug=getenv("DSV4_CUDA_DG_PROFILE")&&atoi(getenv("DSV4_CUDA_DG_PROFILE"));if(debug&&!ok(cudaStreamSynchronize(c->stream),"DeepGEMM FC1 sync"))return 0;
    dg_swiglu_quant_masked<<<6,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,c->expert_ids,weight,limit,M);
    if(debug&&!ok(cudaStreamSynchronize(c->stream),"DeepGEMM SwiGLU sync"))return 0;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga2,2048,(unsigned long long)G*M,2048,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc2,2048,(unsigned long long)G*N,1024,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa2,M,(unsigned long long)G*4,(unsigned long long)M*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc2_dg_scale,N,(unsigned long long)G*16,16384,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm2,N,(unsigned long long)G*M,(unsigned long long)N*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc2=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,128,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedMasked,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,128,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedMasked,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"DeepGEMM FC2 shared memory"))return 0;
    fc2<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm2,nullptr,(__nv_fp8_e4m3*)c->dga2,(__nv_fp8_e4m3*)set->bank_fc2,c->dglayout,nullptr,nullptr,M,N,2048,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(debug&&!ok(cudaStreamSynchronize(c->stream),"DeepGEMM FC2 sync"))return 0;dg_reduce6_masked<<<16,256,0,c->stream>>>(out,c->dgmm2,c->expert_ids,M);if(debug&&!ok(cudaStreamSynchronize(c->stream),"DeepGEMM reduce sync"))return 0;return ok(cudaGetLastError(),"DeepGEMM routed MoE launch");
}
template<int G> static int dg_moe_contiguous_fast(Dev*c,Dsv4CudaExpertSet*set,const float*weight,float limit,float*out,int ep_base=-1){constexpr int M=384,N=4096,A=64;
    size_t a1=(size_t)M*4096,a2=(size_t)M*2048,s1=(size_t)M*8*4,s2=(size_t)M*4*4,mm=(size_t)M*N*2;
    if(!buf((void**)&c->dga1,&c->dga1cap,a1)||!buf((void**)&c->dga2,&c->dga2cap,a2)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,s1)||!buf((void**)&c->dgsfa2,&c->dgsfa2cap,s2)||
       !buf((void**)&c->dglayout,&c->dglayoutcap,M*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,mm)||!buf((void**)&c->dgmm2,&c->dgmm2cap,mm))return 0;
    if(ep_base>=0)dg_layout6_ep<<<3,256,0,c->stream>>>(c->dglayout,c->expert_ids,M,A,ep_base);else dg_layout6<<<3,256,0,c->stream>>>(c->dglayout,c->expert_ids,M,A);
    dg_quant_one<<<8,128,0,c->stream>>>(c->dx,c->dga1,c->dgsfa1,4096,M);dg_replicate_rows<<<((long long)M*4096+255)/256,256,0,c->stream>>>(c->dga1,c->dgsfa1,4096,M);
    CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,4096,M,4096,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc1,4096,(unsigned long long)G*N,2048,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,M,8,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc1_dg_scale,N,(unsigned long long)G*32,16384,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N,M,(unsigned long long)N*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc1=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,64,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,64,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"DeepGEMM FC1 shared memory"))return 0;
    fc1<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)c->dga1,(__nv_fp8_e4m3*)set->bank_fc1,c->dglayout,nullptr,nullptr,M,N,4096,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(ep_base>=0)dg_swiglu_quant_unique_ep<<<6,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,c->expert_ids,ep_base,limit,M,A);
    else dg_swiglu_quant_unique<<<6,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,weight,limit,M,A);
    dg_replicate_expert_rows<<<((long long)M*2048+255)/256,256,0,c->stream>>>(c->dga2,c->dgsfa2,2048,M,A);
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga2,2048,M,2048,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc2,2048,(unsigned long long)G*N,1024,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa2,M,4,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc2_dg_scale,N,(unsigned long long)G*16,16384,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm2,N,M,(unsigned long long)N*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc2=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,128,128,128,128,128,2,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,128,128,128,128,128,2,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,101376),"DeepGEMM FC2 shared memory"))return 0;
    fc2<<<(c->sms>=170?170:84),384,101376,c->stream>>>((DgOutput*)c->dgmm2,nullptr,(__nv_fp8_e4m3*)c->dga2,(__nv_fp8_e4m3*)set->bank_fc2,c->dglayout,nullptr,nullptr,M,N,2048,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(ep_base>=0)dg_reduce6_ep<<<16,256,0,c->stream>>>(out,c->dgmm2,c->expert_ids,weight,ep_base,A);else dg_reduce6<<<16,256,0,c->stream>>>(out,c->dgmm2,A);
    return ok(cudaGetLastError(),"DeepGEMM routed MoE fast launch");
}
/* ids/weight are DEVICE pointers (uploaded by the caller); host_ids is the
 * host copy used to build the expert-grouped layout (nullptr = legacy
 * replicated layout, also DSV4_CUDA_MOE_GROUPED=0). */
static int dg_moe_batch_contiguous(Dev*c,Dsv4CudaExpertSet*set,const float*input,const int*ids,const float*weight,int tokens,float limit,float*out,const int*host_ids=nullptr){constexpr int N=4096,A=64,G=256;int routes=tokens*6;
    static int grouped=-1;if(grouped<0){const char*e=getenv("DSV4_CUDA_MOE_GROUPED");grouped=!(e&&*e=='0');}
    int M=routes*A;
    std::vector<int> hrow,hlayout;
    if(host_ids&&grouped){
        /* Stable bucket by expert: rows per expert padded to A. */
        int cnt[G]={0};for(int r=0;r<routes;r++){int e=host_ids[r];if(e<0||e>=G)return 0;cnt[e]++;}
        int base[G];int m=0;for(int e=0;e<G;e++){base[e]=m;if(cnt[e])m+=(cnt[e]+A-1)/A*A;}
        M=m;hrow.resize(routes);hlayout.assign(M,0);
        int fill[G]={0};for(int r=0;r<routes;r++){int e=host_ids[r];hrow[r]=base[e]+fill[e]++;}
        for(int e=0;e<G;e++)if(cnt[e]){int g=(cnt[e]+A-1)/A*A;for(int i=0;i<g;i++)hlayout[base[e]+i]=e;}
    }
    size_t a1=(size_t)M*4096,a2=(size_t)M*2048,s1=(size_t)M*8*4,s2=(size_t)M*4*4,mm=(size_t)M*N*2;
    if(set->count!=G||set->H!=4096||set->I!=2048||!buf((void**)&c->dga1,&c->dga1cap,a1)||!buf((void**)&c->dga2,&c->dga2cap,a2)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,s1)||!buf((void**)&c->dgsfa2,&c->dgsfa2cap,s2)||!buf((void**)&c->dglayout,&c->dglayoutcap,(size_t)M*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,mm)||!buf((void**)&c->dgmm2,&c->dgmm2cap,mm))return 0;
    bool use_map=!hrow.empty();
    if(use_map){
        if(!buf((void**)&c->dgrowmap,&c->dgrowmapcap,(size_t)routes*sizeof(int))||
           !ok(cudaMemcpyAsync(c->dgrowmap,hrow.data(),(size_t)routes*sizeof(int),cudaMemcpyHostToDevice,c->stream),"grouped MoE rowmap upload")||
           !ok(cudaMemcpyAsync(c->dglayout,hlayout.data(),(size_t)M*sizeof(int),cudaMemcpyHostToDevice,c->stream),"grouped MoE layout upload")||
           !ok(cudaMemsetAsync(c->dga1,0,a1,c->stream),"grouped MoE A1 clear")||!ok(cudaMemsetAsync(c->dgsfa1,0,s1,c->stream),"grouped MoE S1 clear")||
           !ok(cudaMemsetAsync(c->dga2,0,a2,c->stream),"grouped MoE A2 clear")||!ok(cudaMemsetAsync(c->dgsfa2,0,s2,c->stream),"grouped MoE S2 clear"))return 0;
        dg_quant_routes_map<<<routes*8,128,0,c->stream>>>(input,c->dga1,c->dgsfa1,tokens,4096,M,c->dgrowmap);
    }else{
        dg_layout_routes<<<(M+255)/256,256,0,c->stream>>>(c->dglayout,ids,M,A);dg_quant_routes<<<routes*8,128,0,c->stream>>>(input,c->dga1,c->dgsfa1,tokens,4096,M,A);dg_replicate_expert_rows<<<((long long)M*4096+255)/256,256,0,c->stream>>>(c->dga1,c->dgsfa1,4096,M,A);
    }
    CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,4096,M,4096,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc1,4096,(unsigned long long)G*N,2048,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,M,8,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||!dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc1_dg_scale,N,(unsigned long long)G*32,16384,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||!dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N,M,(unsigned long long)N*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc1=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,64,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,64,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;if(!ok(cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"batched MoE FC1 shared memory"))return 0;fc1<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)c->dga1,(__nv_fp8_e4m3*)set->bank_fc1,c->dglayout,nullptr,nullptr,M,N,4096,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(use_map)dg_swiglu_quant_routes_map<<<routes,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,weight,limit,routes,M,c->dgrowmap);
    else{dg_swiglu_quant_routes<<<routes,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,weight,limit,routes,M,A);dg_replicate_expert_rows<<<((long long)M*2048+255)/256,256,0,c->stream>>>(c->dga2,c->dgsfa2,2048,M,A);}
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga2,2048,M,2048,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc2,2048,(unsigned long long)G*N,1024,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa2,M,4,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||!dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc2_dg_scale,N,(unsigned long long)G*16,16384,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||!dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm2,N,M,(unsigned long long)N*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc2=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,64,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,64,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;if(!ok(cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"batched MoE FC2 shared memory"))return 0;fc2<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm2,nullptr,(__nv_fp8_e4m3*)c->dga2,(__nv_fp8_e4m3*)set->bank_fc2,c->dglayout,nullptr,nullptr,M,N,2048,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);if(use_map)dg_reduce_routes_map<<<((long long)tokens*4096+255)/256,256,0,c->stream>>>(out,c->dgmm2,tokens,c->dgrowmap);else dg_reduce_routes<<<((long long)tokens*4096+255)/256,256,0,c->stream>>>(out,c->dgmm2,tokens,A);return ok(cudaGetLastError(),"batched contiguous MoE launch");
}
static int dg_moe_tp2(Dev*c,Dsv4CudaExpertSet*set,const float*weight,float limit,float*out){constexpr int G=256,M=384,A=64,H=4096,J=1024,N1=2048;
    int trace=getenv("DSV4_CUDA_TP2_TRACE")&&atoi(getenv("DSV4_CUDA_TP2_TRACE"));
    size_t a1=(size_t)M*H,a2=(size_t)M*J,s1=(size_t)M*(H/512)*4,s2=(size_t)M*(J/512)*4;
    if(!set||set->count!=G||set->H!=H||set->I!=J||!buf((void**)&c->dga1,&c->dga1cap,a1)||!buf((void**)&c->dga2,&c->dga2cap,a2)||
       !buf((void**)&c->dgsfa1,&c->dgsfa1cap,s1)||!buf((void**)&c->dgsfa2,&c->dgsfa2cap,s2)||
       !buf((void**)&c->dglayout,&c->dglayoutcap,M*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,(size_t)M*N1*2)||
       !buf((void**)&c->dgmm2,&c->dgmm2cap,(size_t)M*H*2))return 0;
    dg_layout6<<<3,256,0,c->stream>>>(c->dglayout,c->expert_ids,M,A);
    dg_quant_one<<<H/512,128,0,c->stream>>>(c->dx,c->dga1,c->dgsfa1,H,M);
    dg_replicate_rows<<<((long long)M*H+255)/256,256,0,c->stream>>>(c->dga1,c->dgsfa1,H,M);
    CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,H,M,H,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc1,H,(unsigned long long)G*N1,H/2,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,M,H/512,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc1_dg_scale,N1,(unsigned long long)G*(H/128),(unsigned long long)N1*4,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N1,M,(unsigned long long)N1*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc1=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N1,H,128,32,G,64,128,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N1,H,128,32,G,64,128,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"TP2 DeepGEMM FC1 shared memory"))return 0;
    fc1<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)c->dga1,(__nv_fp8_e4m3*)set->bank_fc1,c->dglayout,nullptr,nullptr,M,N1,H,N1,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(trace){fprintf(stderr,"[DSV4 TP2] device %d FC1 launched\n",c->id);fflush(stderr);if(!ok(cudaStreamSynchronize(c->stream),"TP2 FC1 trace sync"))return 0;fprintf(stderr,"[DSV4 TP2] device %d FC1 done\n",c->id);fflush(stderr);}
    dg_swiglu_quant_unique_tp2<<<6,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,limit,M,A);
    dg_replicate_expert_rows<<<((long long)M*J+255)/256,256,0,c->stream>>>(c->dga2,c->dgsfa2,J,M,A);
    if(trace){if(!ok(cudaStreamSynchronize(c->stream),"TP2 SwiGLU trace sync"))return 0;fprintf(stderr,"[DSV4 TP2] device %d SwiGLU done\n",c->id);fflush(stderr);}
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga2,J,M,J,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc2,J,(unsigned long long)G*H,J/2,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa2,M,J/512,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc2_dg_scale,H,(unsigned long long)G*(J/128),(unsigned long long)H*4,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm2,H,M,(unsigned long long)H*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc2=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,H,J,128,32,G,64,128,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,H,J,128,32,G,64,128,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"TP2 DeepGEMM FC2 shared memory"))return 0;
    fc2<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm2,nullptr,(__nv_fp8_e4m3*)c->dga2,(__nv_fp8_e4m3*)set->bank_fc2,c->dglayout,nullptr,nullptr,M,H,J,H,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(trace){fprintf(stderr,"[DSV4 TP2] device %d FC2 launched\n",c->id);fflush(stderr);if(!ok(cudaStreamSynchronize(c->stream),"TP2 FC2 trace sync"))return 0;fprintf(stderr,"[DSV4 TP2] device %d FC2 done\n",c->id);fflush(stderr);}
    dg_reduce6_weighted<<<16,256,0,c->stream>>>(out,c->dgmm2,weight,A);
    return ok(cudaGetLastError(),"TP2 DeepGEMM routed MoE launch");
}
static int dg_dump_once(Dev*c){static int dumped;if(dumped++||!(getenv("DSV4_CUDA_DG_DUMP")&&atoi(getenv("DSV4_CUDA_DG_DUMP"))))return 1;
    struct Item{const char*path;const void*device;size_t bytes;}items[]={
        {"/tmp/dsv4-dg-x.f32",c->dx,4096*4},{"/tmp/dsv4-dg-a1.fp8",c->dga1,(size_t)768*4096},
        {"/tmp/dsv4-dg-sfa1.bin",c->dgsfa1,(size_t)768*8*4},{"/tmp/dsv4-dg-layout.i32",c->dglayout,768*4},
        {"/tmp/dsv4-dg-fc1.bf16",c->dgmm1,(size_t)768*4096*2},{"/tmp/dsv4-dg-ids.i32",c->expert_ids,6*4},
        {"/tmp/dsv4-dg-weights.f32",c->expert_weights,6*4}};
    if(!ok(cudaStreamSynchronize(c->stream),"DeepGEMM dump sync"))return 0;
    for(unsigned i=0;i<sizeof(items)/sizeof(*items);i++){void*h=malloc(items[i].bytes);FILE*f=nullptr;if(!h||!ok(cudaMemcpy(h,items[i].device,items[i].bytes,cudaMemcpyDeviceToHost),"DeepGEMM dump copy")||!(f=fopen(items[i].path,"wb"))||fwrite(h,1,items[i].bytes,f)!=items[i].bytes){if(f)fclose(f);free(h);fprintf(stderr,"[DSV4 DeepGEMM] dump failed: %s\n",items[i].path);return 0;}fclose(f);free(h);}
    fprintf(stderr,"[DSV4 DeepGEMM] dumped first FC1 fixture to /tmp/dsv4-dg-*\n");return 1;}
#endif
__global__ void expert_reduce(float *y,const float *parts,int count,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=0;
    for(int e=0;e<count;e++)v+=__bfloat162float(__float2bfloat16(parts[(long long)e*n+i]));y[i]=v;}}
/* Batched siblings of expert_reduce / expert_act(weight=1) for the generic
 * batched MoE: parts hold `count` rows per token ([t*count+e][n]). */
__global__ void expert_reduce_rows(float *y,const float *parts,int tokens,int count,int n){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(p<(long long)tokens*n){int t=p/n,i=p%n;float v=0;for(int e=0;e<count;e++)v+=__bfloat162float(__float2bfloat16(parts[((long long)t*count+e)*n+i]));y[p]=v;}}
__global__ void expert_act_rows1(float *a,const float *g,const float *u,float limit,long long n){long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
    float gg=fminf(__bfloat162float(__float2bfloat16(g[i])),limit);
    float uu=fmaxf(-limit,fminf(__bfloat162float(__float2bfloat16(u[i])),limit));
    a[i]=__bfloat162float(__float2bfloat16((gg/(1.f+expf(-gg)))*uu));}}
__global__ void moe_combine(float *routed,const float *shared,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)
    routed[i]=__bfloat162float(__float2bfloat16(routed[i]+__bfloat162float(__float2bfloat16(shared[i]))));}
__global__ void ep2_localize(int *ids,float *weights,int base){int k=threadIdx.x;if(k<6){int id=ids[k]-base;if(id<0||id>=128){ids[k]=-1;weights[k]=0.f;}else ids[k]=id;}}
__global__ void sort_routes6(int*ids,float*weights){if(threadIdx.x||blockIdx.x)return;for(int i=1;i<6;i++){int id=ids[i],j=i-1;float w=weights[i];while(j>=0&&ids[j]>id){ids[j+1]=ids[j];weights[j+1]=weights[j];j--;}ids[j+1]=id;weights[j+1]=w;}}
__global__ void add_peer(float *out,const float *peer,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__bfloat162float(__float2bfloat16(out[i]+peer[i]));}
__global__ void bf16_round(float *x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=__bfloat162float(__float2bfloat16(x[i]));}
__device__ __forceinline__ float warp_sum_f32(float v){for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return v;}
__global__ void rmsnorm_seq(float *x,const __nv_bfloat16 *w,int n,float eps){
    if(blockDim.x==1){double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];float inv=rsqrtf((float)(ss/n)+eps);for(int i=0;i<n;i++)x[i]=__bfloat162float(__float2bfloat16(x[i]*inv*__bfloat162float(w[i])));return;}
    __shared__ float ws[8],inv;float ss=0;int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int i=threadIdx.x;i<n;i+=blockDim.x)ss+=x[i]*x[i];ss=warp_sum_f32(ss);if(!lane)ws[warp]=ss;__syncthreads();if(!warp){float v=lane<8?ws[lane]:0.f;v=warp_sum_f32(v);if(!lane)inv=rsqrtf(v/n+eps);}__syncthreads();for(int i=threadIdx.x;i<n;i+=blockDim.x)x[i]=__bfloat162float(__float2bfloat16(x[i]*inv*__bfloat162float(w[i])));
}
__global__ void rmsnorm_f32(float *out,const float *x,const float *w,int n,float eps){
    if(blockDim.x==1){double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];float inv=rsqrtf((float)(ss/n)+eps);for(int i=0;i<n;i++)out[i]=__bfloat162float(__float2bfloat16(x[i]*inv*w[i]));return;}
    __shared__ float ws[8],inv;float ss=0;int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int i=threadIdx.x;i<n;i+=blockDim.x)ss+=x[i]*x[i];ss=warp_sum_f32(ss);if(!lane)ws[warp]=ss;__syncthreads();if(!warp){float v=lane<8?ws[lane]:0.f;v=warp_sum_f32(v);if(!lane)inv=rsqrtf(v/n+eps);}__syncthreads();for(int i=threadIdx.x;i<n;i+=blockDim.x)out[i]=__bfloat162float(__float2bfloat16(x[i]*inv*w[i]));
}
__global__ void head_rmsnorm(float *q,int heads,int dim,float eps){int h=blockIdx.x;if(h>=heads)return;float *x=q+(long long)h*dim;__shared__ float sum[256],inv;float ss=0;
    for(int d=threadIdx.x;d<dim;d+=blockDim.x)ss+=x[d]*x[d];sum[threadIdx.x]=ss;__syncthreads();for(int n=blockDim.x/2;n;n>>=1){if(threadIdx.x<n)sum[threadIdx.x]+=sum[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)inv=rsqrtf(sum[0]/dim+eps);__syncthreads();for(int d=threadIdx.x;d<dim;d+=blockDim.x)x[d]=__bfloat162float(__float2bfloat16(x[d]*inv));}
__global__ void attention_one(float *context,const float *q,const float *kv,const float *sink,int heads,int dim){int h=blockIdx.x;if(h>=heads)return;__shared__ float a,den;
    if(!threadIdx.x){float score=0;for(int d=0;d<dim;d++)score+=q[(long long)h*dim+d]*kv[d];score/=sqrtf((float)dim);float mx=fmaxf(score,sink[h]);a=expf(score-mx);den=a+expf(sink[h]-mx);}__syncthreads();
    for(int d=threadIdx.x;d<dim;d+=blockDim.x)context[(long long)h*dim+d]=__bfloat162float(__float2bfloat16(a*kv[d]/den));}
__global__ void rope_heads(float *x,int heads,int dim,int rope,int pos,int inverse,const float *cs,const float *sn){int p=blockIdx.x*blockDim.x+threadIdx.x,pairs=rope/2;if(p<heads*pairs){int h=p/pairs,k=p%pairs,off=h*dim+dim-rope+2*k;float s=inverse?-sn[(long long)pos*pairs+k]:sn[(long long)pos*pairs+k],u=x[off],v=x[off+1];x[off]=__bfloat162float(__float2bfloat16(u*cs[(long long)pos*pairs+k]-v*s));x[off+1]=__bfloat162float(__float2bfloat16(u*s+v*cs[(long long)pos*pairs+k]));}}
__global__ void cache_store(float *cache,const float *kv,int slot,int dim){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<dim)cache[(long long)slot*dim+d]=kv[d];}
__global__ void fill_value(float *x,long long n,float v){long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=v;}
__global__ void compress_store(float *values,float *scores,const float *v,const float *g,const float *ape,int slot,int phase,int O){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<O){values[(long long)slot*O+d]=v[d];scores[(long long)slot*O+d]=g[d]+ape[(long long)phase*O+d];}}
__global__ void compress_pool(float *out,const float *values,const float *scores,int ratio,int overlap,int dim){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<dim){int O=(1+overlap)*dim,items=overlap?2*ratio:ratio;float mx=-INFINITY;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?dim+d:d;mx=fmaxf(mx,scores[(long long)j*O+col]);}float den=0,sum=0;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?dim+d:d,aoff=j*O+col;float a=expf(scores[aoff]-mx);den+=a;sum+=a*values[aoff];}out[d]=sum/den;}}
__global__ void attention_window_kernel(float *context,const float *q,const float *cache,const float *compressed,const float *sink,int heads,int dim,int window,int pos,int cn){
    int h=blockIdx.x;if(h>=heads)return;extern __shared__ float score[];__shared__ float mx,den;int start=pos-window+1;if(start<0)start=0;int nw=pos-start+1;
    if(!threadIdx.x){mx=sink[h];for(int t=start;t<=pos;t++){float z=0,*v=(float*)cache+(long long)(t%window)*dim;for(int d=0;d<dim;d++)z+=q[(long long)h*dim+d]*v[d];z/=sqrtf((float)dim);score[t-start]=z;mx=fmaxf(mx,z);}for(int i=0;i<cn;i++){float z=0,*v=(float*)compressed+(long long)i*dim;for(int d=0;d<dim;d++)z+=q[(long long)h*dim+d]*v[d];z/=sqrtf((float)dim);score[nw+i]=z;mx=fmaxf(mx,z);}den=expf(sink[h]-mx);for(int i=0;i<nw+cn;i++){float a=expf(score[i]-mx);score[i]=a;den+=a;}}
    __syncthreads();for(int d=threadIdx.x;d<dim;d+=blockDim.x){float sum=0;for(int t=start;t<=pos;t++)sum+=score[t-start]*cache[((long long)(t%window))*dim+d];for(int i=0;i<cn;i++)sum+=score[nw+i]*compressed[(long long)i*dim+d];context[(long long)h*dim+d]=__bfloat162float(__float2bfloat16(sum/den));}}
/* Prefill batched sparse window attention.  vals is a linear slab of KV rows:
 * [0, comp_base) window rows in position order, [comp_base, ...) the
 * compressed rows.  meta[3t] = first window row for token t, meta[3t+1] = its
 * window row count, meta[3t+2] = its visible compressed prefix.  Numerics
 * mirror coli_v4_sparse_attention_ref: fp32 scores, max over scores only
 * (sink joins the denominator, not the max), probabilities rounded to bf16
 * before the value sum, output rounded to bf16 after the division. */
/* ---- persistent per-layer KV cache for batched prefill attention ----
 * The engine rebuilt and re-uploaded the linear KV slab every chunk-layer
 * (~2 GB PCIe per 826-token prefill). Instead: a per-layer device RING of
 * window rows (absolute position % window) appended once per chunk, plus an
 * append-only compressed-prefix buffer. The engine tracks validity by exact
 * position continuity and reseeds on any mismatch, so no invalidation hooks
 * are needed here. */
#define DSV4_KV_CACHE_LAYERS 64
static float *g_kv_ring[DSV4_KV_CACHE_LAYERS];
static int g_kv_ring_rows[DSV4_KV_CACHE_LAYERS];
static float *g_kv_comp[DSV4_KV_CACHE_LAYERS];
static int g_kv_comp_cap[DSV4_KV_CACHE_LAYERS];

/* Which GPUs this build's kernels can run on. The DeepGEMM build carries
 * sm_120a-only block-scaled MMA kernels. The generic build is a portable fat
 * binary of plain CUDA kernels (fp32 compute from fp8/fp4 decode); the engine's
 * loader tries the DeepGEMM DLL first and falls back to the generic one when
 * this says no.
 *
 * The generic floor is sm_60, not sm_80. Nothing in these kernels needs Ampere:
 * bf16 is storage only (every use is a __bfloat162float / __float2bfloat16
 * convert, the arithmetic is fp32) and fp8 weights are decoded through a
 * constant lookup table rather than fp8 tensor cores. The warp primitives used
 * here (__shfl_*_sync, __syncwarp) are sm_30-era and impose no floor of their
 * own.
 *
 * sm_60 is chosen because Pascal is the oldest architecture this has actually
 * been validated on -- sm_61 on a GTX 1080 Ti, matching the CPU reference to
 * 2^-26. Maxwell would very likely work and is deliberately NOT claimed here,
 * because nobody has run it.
 *
 * This is a RUNTIME gate and is separate from what the build actually contains:
 * a binary compiled with CUDA_ARCH=portable still has no sm_61 or sm_75 code in
 * it. Use CUDA_ARCH=portable-pre-ampere for those cards. Saying yes here for a
 * card the build has no cubin for surfaces as a "no kernel image is available"
 * launch failure, not a wrong answer. */
extern "C" int dsv4_cuda_backend_arch_ok(int device){
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,device)!=cudaSuccess){cudaGetLastError();return 0;}
#ifdef COLI_DSV4_DEEPGEMM
    return prop.major==12;
#else
    return prop.major>=6;
#endif
}
extern "C" const char *dsv4_cuda_backend_name(void){
#ifdef COLI_DSV4_DEEPGEMM
    return "deepgemm-sm120";
#else
    return "generic";
#endif
}

extern "C" long long dsv4_cuda_mem_free_mb(int device){
    if(!ok(cudaSetDevice(device),"select meminfo device"))return -1;
    size_t free_b=0,total_b=0;
    if(!ok(cudaMemGetInfo(&free_b,&total_b),"device meminfo"))return -1;
    return (long long)(free_b>>20);
}

extern "C" int dsv4_cuda_kv_ring_append(int device,int layer,const float*rows,
        int start_pos,int count,int window,int dim){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!rows||start_pos<0||count<1||
       window<1||dim<1||!ok(cudaSetDevice(device),"select kv ring device"))return 0;
    if(g_kv_ring[layer]&&g_kv_ring_rows[layer]!=window){cudaFree(g_kv_ring[layer]);g_kv_ring[layer]=NULL;}
    if(!g_kv_ring[layer]){
        if(!ok(cudaMalloc((void**)&g_kv_ring[layer],(size_t)window*dim*4),"kv ring allocation"))return 0;
        g_kv_ring_rows[layer]=window;
    }
    int done=0;
    while(done<count){
        int slot=(start_pos+done)%window;
        int run=count-done;if(run>window-slot)run=window-slot;
        if(!ok(cudaMemcpyAsync(g_kv_ring[layer]+(size_t)slot*dim,rows+(size_t)done*dim,
                               (size_t)run*dim*4,cudaMemcpyHostToDevice,c->stream),"kv ring append"))return 0;
        done+=run;
    }
    return 1;
}

extern "C" int dsv4_cuda_kv_comp_append(int device,int layer,const float*rows,
        int start_idx,int count,int dim){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!rows||start_idx<0||count<1||dim<1||
       !ok(cudaSetDevice(device),"select kv comp device"))return 0;
    int need=start_idx+count;
    if(g_kv_comp_cap[layer]<need){
        int cap=g_kv_comp_cap[layer]?g_kv_comp_cap[layer]:512;
        while(cap<need)cap*=2;
        float*grown=NULL;
        if(!ok(cudaMalloc((void**)&grown,(size_t)cap*dim*4),"kv comp allocation"))return 0;
        if(g_kv_comp[layer]&&start_idx&&
           !ok(cudaMemcpyAsync(grown,g_kv_comp[layer],(size_t)start_idx*dim*4,
                               cudaMemcpyDeviceToDevice,c->stream),"kv comp grow copy")){cudaFree(grown);return 0;}
        if(g_kv_comp[layer]){cudaStreamSynchronize(c->stream);cudaFree(g_kv_comp[layer]);}
        g_kv_comp[layer]=grown;g_kv_comp_cap[layer]=cap;
    }
    return ok(cudaMemcpyAsync(g_kv_comp[layer]+(size_t)start_idx*dim,rows,
                              (size_t)count*dim*4,cudaMemcpyHostToDevice,c->stream),"kv comp append");
}

/* Window rows at absolute position >= chunk_start come from the chunk buffer,
 * not the ring: with the ring indexed position%window, appending the chunk
 * before attention would let its newest rows overwrite the oldest window rows
 * that early chunk tokens still need (only bites once start+batch > window).
 * The engine appends the chunk to the ring AFTER attention instead. */
__device__ __forceinline__ const float*cached_row(const float*win,const float*chunk,
        int ap,int chunk_start,int window,int dim){
    return ap>=chunk_start?chunk+(long long)(ap-chunk_start)*dim
                          :win+(long long)(ap%window)*dim;
}

__global__ void sparse_attn_batch_cached_kernel(float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,int abs_base,
        int chunk_start,int window,int heads,int dim,float scale){
    int t=blockIdx.y,h=blockIdx.x;
    int wo=meta[3*t],wn=meta[3*t+1],cn=meta[3*t+2],total=wn+cn;
    const float*qh=q+((long long)t*heads+h)*dim;
    extern __shared__ float score[];
    __shared__ float den;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int item=warp;item<total;item+=blockDim.x/32){
        const float*v=item<wn?cached_row(win,chunk,abs_base+wo+item,chunk_start,window,dim)
                              :comp+(long long)(item-wn)*dim;
        float z=0;for(int d=lane;d<dim;d+=32)z+=qh[d]*v[d];
        for(int n=16;n;n>>=1)z+=__shfl_down_sync(0xffffffff,z,n);
        if(!lane)score[item]=z*scale;
    }
    __syncthreads();
    if(!threadIdx.x){float m=-INFINITY;for(int i=0;i<total;i++)m=fmaxf(m,score[i]);
        float d0=expf(sink[h]-m);
        for(int i=0;i<total;i++){float a=expf(score[i]-m);d0+=a;score[i]=__bfloat162float(__float2bfloat16(a));}
        den=d0;}
    __syncthreads();
    for(int d=threadIdx.x;d<dim;d+=blockDim.x){
        float sum=0;
        for(int i=0;i<wn;i++)sum+=score[i]*cached_row(win,chunk,abs_base+wo+i,chunk_start,window,dim)[d];
        for(int i=0;i<cn;i++)sum+=score[wn+i]*comp[(long long)i*dim+d];
        out[((long long)t*heads+h)*dim+d]=__bfloat162float(__float2bfloat16(sum/den));
    }
}

extern "C" int dsv4_cuda_sparse_attn_batch_cached(int device,int layer,const float*q,
        const float*chunk,int chunk_start,const float*sinks,const int*meta,int abs_base,
        int comp_limit,int heads,int dim,int tokens,float scale,float*out){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!g_kv_ring[layer]||!q||!chunk||!sinks||
       !meta||!out||abs_base<0||chunk_start<0||heads<1||dim<1||tokens<1||!(scale>0.f))return 0;
    if(comp_limit>0&&(!g_kv_comp[layer]||g_kv_comp_cap[layer]<comp_limit))return 0;
    int window=g_kv_ring_rows[layer],max_total=0;
    for(int t=0;t<tokens;t++){
        int wn=meta[3*t+1],cn=meta[3*t+2],tt=wn+cn;
        if(meta[3*t]<0||wn<1||wn>window||cn<0||cn>comp_limit)return 0;
        if(abs_base+meta[3*t]+wn>chunk_start+tokens)return 0;
        if(tt>max_total)max_total=tt;
    }
    size_t qb=(size_t)tokens*heads*dim*4,cb=(size_t)tokens*dim*4,sb=(size_t)heads*4,
           mb=(size_t)tokens*3*sizeof(int),sm=(size_t)max_total*4;
    if(sm>96*1024)return 0;
    if(!ok(cudaSetDevice(device),"select cached sparse attention device")||
       !buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p4,&c->p4cap,qb)||
       !buf((void**)&c->p3,&c->p3cap,cb)||
       !buf((void**)&c->aux1,&c->aux1cap,sb)||!buf((void**)&c->aux2,&c->aux2cap,mb))return 0;
    if(!ok(cudaMemcpyAsync(c->p2,q,qb,cudaMemcpyHostToDevice,c->stream),"cached attention query upload")||
       !ok(cudaMemcpyAsync(c->p3,chunk,cb,cudaMemcpyHostToDevice,c->stream),"cached attention chunk upload")||
       !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"cached attention sink upload")||
       !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"cached attention meta upload"))return 0;
    sparse_attn_batch_cached_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,
        abs_base,chunk_start,window,heads,dim,scale);
    return ok(cudaGetLastError(),"cached sparse attention launch")&&
           ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"cached attention download")&&
           ok(cudaStreamSynchronize(c->stream),"cached sparse attention sync");
}

/* Indexed variant: past index_topk the indexer selects a strict subset of the
 * compressed prefix, so meta[3t+2] becomes the SELECTED count and sel holds
 * each token's compressed ordinals (score order — the same order the CPU
 * reference accumulates, keeping fp32 sums bitwise identical). */
__global__ void sparse_attn_batch_cached_idx_kernel(float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,const int*sel,
        int selstride,int abs_base,int chunk_start,int window,int heads,int dim,float scale){
    int t=blockIdx.y,h=blockIdx.x;
    int wo=meta[3*t],wn=meta[3*t+1],cn=meta[3*t+2],total=wn+cn;
    const int*ts=sel+(long long)t*selstride;
    const float*qh=q+((long long)t*heads+h)*dim;
    extern __shared__ float score[];
    __shared__ float den;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int item=warp;item<total;item+=blockDim.x/32){
        const float*v=item<wn?cached_row(win,chunk,abs_base+wo+item,chunk_start,window,dim)
                              :comp+(long long)ts[item-wn]*dim;
        float z=0;for(int d=lane;d<dim;d+=32)z+=qh[d]*v[d];
        for(int n=16;n;n>>=1)z+=__shfl_down_sync(0xffffffff,z,n);
        if(!lane)score[item]=z*scale;
    }
    __syncthreads();
    if(!threadIdx.x){float m=-INFINITY;for(int i=0;i<total;i++)m=fmaxf(m,score[i]);
        float d0=expf(sink[h]-m);
        for(int i=0;i<total;i++){float a=expf(score[i]-m);d0+=a;score[i]=__bfloat162float(__float2bfloat16(a));}
        den=d0;}
    __syncthreads();
    for(int d=threadIdx.x;d<dim;d+=blockDim.x){
        float sum=0;
        for(int i=0;i<wn;i++)sum+=score[i]*cached_row(win,chunk,abs_base+wo+i,chunk_start,window,dim)[d];
        for(int i=0;i<cn;i++)sum+=score[wn+i]*comp[(long long)ts[i]*dim+d];
        out[((long long)t*heads+h)*dim+d]=__bfloat162float(__float2bfloat16(sum/den));
    }
}

extern "C" int dsv4_cuda_sparse_attn_batch_cached_idx(int device,int layer,const float*q,
        const float*chunk,int chunk_start,const float*sinks,const int*meta,const int*sel,
        int selstride,int abs_base,int comp_limit,int heads,int dim,int tokens,
        float scale,float*out){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!g_kv_ring[layer]||!q||!chunk||!sinks||
       !meta||!sel||selstride<1||!out||abs_base<0||chunk_start<0||heads<1||dim<1||
       tokens<1||!(scale>0.f))return 0;
    if(comp_limit>0&&(!g_kv_comp[layer]||g_kv_comp_cap[layer]<comp_limit))return 0;
    int window=g_kv_ring_rows[layer],max_total=0;
    for(int t=0;t<tokens;t++){
        int wn=meta[3*t+1],cn=meta[3*t+2],tt=wn+cn;
        if(meta[3*t]<0||wn<1||wn>window||cn<0||cn>selstride||cn>comp_limit)return 0;
        if(abs_base+meta[3*t]+wn>chunk_start+tokens)return 0;
        for(int i=0;i<cn;i++){int o=sel[(long long)t*selstride+i];if(o<0||o>=comp_limit)return 0;}
        if(tt>max_total)max_total=tt;
    }
    size_t qb=(size_t)tokens*heads*dim*4,cb=(size_t)tokens*dim*4,sb=(size_t)heads*4,
           mb=(size_t)tokens*3*sizeof(int),ib=(size_t)tokens*selstride*sizeof(int),
           sm=(size_t)max_total*4;
    if(sm>96*1024)return 0;
    if(!ok(cudaSetDevice(device),"select cached idx sparse attention device")||
       !buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p4,&c->p4cap,qb)||
       !buf((void**)&c->p3,&c->p3cap,cb)||
       !buf((void**)&c->aux1,&c->aux1cap,sb)||!buf((void**)&c->aux2,&c->aux2cap,mb)||
       !buf((void**)&c->aux3,&c->aux3cap,ib))return 0;
    if(!ok(cudaMemcpyAsync(c->p2,q,qb,cudaMemcpyHostToDevice,c->stream),"cached idx attention query upload")||
       !ok(cudaMemcpyAsync(c->p3,chunk,cb,cudaMemcpyHostToDevice,c->stream),"cached idx attention chunk upload")||
       !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"cached idx attention sink upload")||
       !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"cached idx attention meta upload")||
       !ok(cudaMemcpyAsync(c->aux3,sel,ib,cudaMemcpyHostToDevice,c->stream),"cached idx attention index upload"))return 0;
    sparse_attn_batch_cached_idx_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,(int*)c->aux3,
        selstride,abs_base,chunk_start,window,heads,dim,scale);
    return ok(cudaGetLastError(),"cached idx sparse attention launch")&&
           ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"cached idx attention download")&&
           ok(cudaStreamSynchronize(c->stream),"cached idx sparse attention sync");
}

/* Batched indexer scoring for prefill. One thread per (token, candidate):
 * score = sum_h relu(q_h . k) * w_h with the SAME sequential fp32 order as the
 * CPU reference (dims inner, heads outer, separately rounded mul/add) so the
 * host-side top-k sort sees bitwise-identical scores. queries [tokens][heads*dim]
 * (already hadamard/fp4-rounded), keys [count][dim], head_w [tokens][heads],
 * counts[t] = candidates visible to token t. Candidates >= counts[t] get 0. */
__global__ void indexer_score_kernel(float*scores,const float*queries,const float*keys,
        const float*head_w,const int*counts,int heads,int dim,int count){
    int t=blockIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
    extern __shared__ float qs[];
    int qn=heads*dim;
    for(int i=threadIdx.x;i<qn;i+=blockDim.x)qs[i]=queries[(long long)t*qn+i];
    __syncthreads();
    if(c>=count)return;
    float score=0.f;
    if(c<counts[t]){
        const float*k=keys+(long long)c*dim;
        for(int h=0;h<heads;h++){
            const float*q=qs+h*dim;
            float dot=0.f;
            /* The CPU build (MinGW GCC, -O3 -march=native) does NOT contract
             * these into FMAs — measured — so keep mul and add separately
             * rounded here too. */
            for(int i=0;i<dim;i++)dot=__fadd_rn(dot,__fmul_rn(q[i],k[i]));
            score=__fadd_rn(score,__fmul_rn(fmaxf(dot,0.f),head_w[(long long)t*heads+h]));
        }
    }
    scores[(long long)t*count+c]=score;
}

extern "C" int dsv4_cuda_indexer_score_batch(int device,const float*queries,const float*keys,
        const float*head_w,const int*counts,int tokens,int heads,int dim,int count,
        float*scores){
    Dev*c=ctx(device);
    if(!c||!queries||!keys||!head_w||!counts||!scores||tokens<1||heads<1||dim<1||count<1)return 0;
    size_t qb=(size_t)tokens*heads*dim*4,kb=(size_t)count*dim*4,wb=(size_t)tokens*heads*4,
           cb=(size_t)tokens*sizeof(int),sb=(size_t)tokens*count*4,sm=(size_t)heads*dim*4;
    if(sm>96*1024)return 0;
    if(!ok(cudaSetDevice(device),"select indexer score device")||
       !buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p3,&c->p3cap,kb)||
       !buf((void**)&c->p4,&c->p4cap,sb)||!buf((void**)&c->aux1,&c->aux1cap,wb)||
       !buf((void**)&c->aux2,&c->aux2cap,cb))return 0;
    if(!ok(cudaMemcpyAsync(c->p2,queries,qb,cudaMemcpyHostToDevice,c->stream),"indexer query upload")||
       !ok(cudaMemcpyAsync(c->p3,keys,kb,cudaMemcpyHostToDevice,c->stream),"indexer key upload")||
       !ok(cudaMemcpyAsync(c->aux1,head_w,wb,cudaMemcpyHostToDevice,c->stream),"indexer head weight upload")||
       !ok(cudaMemcpyAsync(c->aux2,counts,cb,cudaMemcpyHostToDevice,c->stream),"indexer count upload"))return 0;
    if(sm>48*1024&&cudaFuncSetAttribute(indexer_score_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)!=cudaSuccess)return 0;
    indexer_score_kernel<<<dim3((count+127)/128,tokens),128,sm,c->stream>>>(
        c->p4,c->p2,c->p3,c->aux1,(int*)c->aux2,heads,dim,count);
    return ok(cudaGetLastError(),"indexer score launch")&&
           ok(cudaMemcpyAsync(scores,c->p4,sb,cudaMemcpyDeviceToHost,c->stream),"indexer score download")&&
           ok(cudaStreamSynchronize(c->stream),"indexer score sync");
}

/* Exact replica of the CPU reference matmul_fp8 (quant.h): per output row,
 * per 128-column block a sequential fp32 accumulate of e4m3(w)*x, then a
 * DOUBLE accumulate of block*scale across blocks, cast to float. Separately
 * rounded mul/add (the CPU build does not contract to FMA), so results are
 * bitwise the reference. x is the fp8-qdq'd activation (done on the host).
 * One thread per (row, token); tokens are the fast dimension so the 64
 * threads of a token block broadcast-read the same weight bytes. */
__device__ __forceinline__ float e4m3_dev(uint8_t b){
    int s=b>>7,e=(b>>3)&15,m=b&7;
    float v;
    if(e==15&&m==7)v=__int_as_float(0x7fc00000);          /* nan */
    else if(e==0)v=ldexpf((float)m/8.f,-6);
    else v=ldexpf(1.f+(float)m/8.f,e-7);
    return s?-v:v;
}
__global__ void fp8_ref_matmul_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x,int rows,int cols,int tokens){
    int t=threadIdx.x,o=blockIdx.x;
    if(t>=tokens||o>=rows)return;
    const uint8_t*wr=w+(long long)o*cols;
    const float*xs=x+(long long)t*cols;
    int nblk=(cols+127)/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    double a=0.0;
    for(int bi=0;bi*128<cols;bi++){
        int base=bi*128,blen=cols-base<128?cols-base:128;
        float acc=0.f;
        for(int i=base;i<base+blen;i++)acc=__fadd_rn(acc,__fmul_rn(e4m3_dev(wr[i]),xs[i]));
        a=__dadd_rn(a,__dmul_rn((double)acc,(double)scl[bi]));
    }
    y[(long long)t*rows+o]=__double2float_rn(a);
}

/* rows8-PACKED replica: the resident CPU layout stores 8 consecutive rows'
 * bytes per column ((tile*cols+col)*8 + r) and the AVX2 reference kernel
 * accumulates sum += (x*w)*scale in fp32 sequentially over ALL columns (no
 * per-block double). Same order here, so bitwise the AVX2 path. */
__global__ void fp8_ref_matmul_rows8_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x,int rows,int cols,int tokens){
    int t=threadIdx.x,o=blockIdx.x;
    if(t>=tokens||o>=rows)return;
    const float*xs=x+(long long)t*cols;
    int nblk=cols/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    long long tile=o>>3;int r=o&7;
    float sum=0.f;
    for(int base=0;base<cols;base+=128){
        float sc=scl[base/128];
        for(int i=base;i<base+128;i++){
            float v=e4m3_dev(w[((tile*cols)+i)*8+r]);
            sum=__fadd_rn(sum,__fmul_rn(__fmul_rn(xs[i],v),sc));
        }
    }
    y[(long long)t*rows+o]=sum;
}

extern "C" int dsv4_cuda_fp8_ref_matmul(int device,const uint8_t*w,const float*bscale,
        int rows,int cols,int packed_rows8,const float*x,int tokens,float*y){
    Dev*c=ctx(device);
    if(!c||!w||!bscale||!x||!y||rows<1||cols<1||cols%128||tokens<1||tokens>1024||
       (packed_rows8&&rows%8))return 0;
    size_t wb=(size_t)rows*cols,sb=(size_t)((rows+127)/128)*(cols/128)*4,
           xb=(size_t)tokens*cols*4,yb=(size_t)tokens*rows*4;
    if(!ok(cudaSetDevice(device),"select fp8 ref matmul device")||
       !buf((void**)&c->p1,&c->p1cap,wb)||!buf((void**)&c->aux1,&c->aux1cap,sb)||
       !buf((void**)&c->p2,&c->p2cap,xb)||!buf((void**)&c->p4,&c->p4cap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->p1,w,wb,cudaMemcpyHostToDevice,c->stream),"fp8 ref weight upload")||
       !ok(cudaMemcpyAsync(c->aux1,bscale,sb,cudaMemcpyHostToDevice,c->stream),"fp8 ref scale upload")||
       !ok(cudaMemcpyAsync(c->p2,x,xb,cudaMemcpyHostToDevice,c->stream),"fp8 ref activation upload"))return 0;
    int threads=tokens<32?32:((tokens+31)/32)*32;
    if(packed_rows8)
        fp8_ref_matmul_rows8_kernel<<<rows,threads,0,c->stream>>>(c->p4,(const uint8_t*)c->p1,c->aux1,c->p2,rows,cols,tokens);
    else
        fp8_ref_matmul_kernel<<<rows,threads,0,c->stream>>>(c->p4,(const uint8_t*)c->p1,c->aux1,c->p2,rows,cols,tokens);
    return ok(cudaGetLastError(),"fp8 ref matmul launch")&&
           ok(cudaMemcpyAsync(y,c->p4,yb,cudaMemcpyDeviceToHost,c->stream),"fp8 ref matmul download")&&
           ok(cudaStreamSynchronize(c->stream),"fp8 ref matmul sync");
}

__global__ void sparse_attn_batch_kernel(float*out,const float*q,const float*vals,const float*sink,
        const int*meta,int comp_base,int heads,int dim,float scale){
    int t=blockIdx.y,h=blockIdx.x;
    int wo=meta[3*t],wn=meta[3*t+1],cn=meta[3*t+2],total=wn+cn;
    const float*qh=q+((long long)t*heads+h)*dim;
    extern __shared__ float score[];
    __shared__ float den;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int item=warp;item<total;item+=blockDim.x/32){
        const float*v=vals+(long long)(item<wn?wo+item:comp_base+item-wn)*dim;
        float z=0;for(int d=lane;d<dim;d+=32)z+=qh[d]*v[d];
        for(int n=16;n;n>>=1)z+=__shfl_down_sync(0xffffffff,z,n);
        if(!lane)score[item]=z*scale;
    }
    __syncthreads();
    if(!threadIdx.x){float m=-INFINITY;for(int i=0;i<total;i++)m=fmaxf(m,score[i]);
        float d0=expf(sink[h]-m);
        for(int i=0;i<total;i++){float a=expf(score[i]-m);d0+=a;score[i]=__bfloat162float(__float2bfloat16(a));}
        den=d0;}
    __syncthreads();
    for(int d=threadIdx.x;d<dim;d+=blockDim.x){
        float sum=0;
        for(int i=0;i<wn;i++)sum+=score[i]*vals[(long long)(wo+i)*dim+d];
        for(int i=0;i<cn;i++)sum+=score[wn+i]*vals[(long long)(comp_base+i)*dim+d];
        out[((long long)t*heads+h)*dim+d]=__bfloat162float(__float2bfloat16(sum/den));
    }
}
extern "C" int dsv4_cuda_sparse_attn_batch(int device,const float*q,const float*vals,const float*sinks,
        const int*meta,int value_rows,int comp_base,int heads,int dim,int tokens,float scale,float*out){
    Dev*c=ctx(device);
    if(!c||!q||!vals||!sinks||!meta||!out||value_rows<1||comp_base<0||comp_base>value_rows||
       heads<1||dim<1||tokens<1||!(scale>0.f))return 0;
    int max_total=0;
    for(int t=0;t<tokens;t++){
        int wn=meta[3*t+1],cn=meta[3*t+2],tt=wn+cn;
        if(meta[3*t]<0||wn<1||cn<0||meta[3*t]+wn>comp_base||comp_base+cn>value_rows)return 0;
        if(tt>max_total)max_total=tt;
    }
    size_t qb=(size_t)tokens*heads*dim*4,vb=(size_t)value_rows*dim*4,sb=(size_t)heads*4,
           mb=(size_t)tokens*3*sizeof(int),sm=(size_t)max_total*4;
    if(sm>96*1024)return 0;
    if(!ok(cudaSetDevice(device),"select batched sparse attention device")||
       !buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p1,&c->p1cap,vb)||
       !buf((void**)&c->p4,&c->p4cap,qb)||!buf((void**)&c->aux1,&c->aux1cap,sb)||
       !buf((void**)&c->aux2,&c->aux2cap,mb))return 0;
    if(!ok(cudaMemcpyAsync(c->p2,q,qb,cudaMemcpyHostToDevice,c->stream),"batched attention query upload")||
       !ok(cudaMemcpyAsync(c->p1,vals,vb,cudaMemcpyHostToDevice,c->stream),"batched attention KV upload")||
       !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"batched attention sink upload")||
       !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"batched attention meta upload"))return 0;
    if(sm>48*1024&&!ok(cudaFuncSetAttribute(sparse_attn_batch_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm),"batched sparse attention shared memory"))return 0;
    sparse_attn_batch_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(c->p4,c->p2,c->p1,c->aux1,(int*)c->aux2,comp_base,heads,dim,scale);
    return ok(cudaGetLastError(),"batched sparse attention launch")&&
           ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"batched attention context download")&&
           ok(cudaStreamSynchronize(c->stream),"batched sparse attention sync");
}
/* Prefill batch GEMM over a resident bf16 matrix (compressor / indexer
 * projections).  Same tiling as mm_batch; numerics match the CPU reference:
 * bf16-decoded weights, fp32 accumulation, no activation quantization. */
__global__ void mm_bf16_batch(const __nv_bfloat16*w,const float*x,float*y,int O,int I,int T){
    int o=blockIdx.x;if(o>=O)return;int tbase=blockIdx.y*16,tcount=T-tbase<16?T-tbase:16;
    float sum[16]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){float v=__bfloat162float(w[(long long)o*I+i]);
        #pragma unroll
        for(int t=0;t<16;t++)if(t<tcount)sum[t]+=x[(long long)(tbase+t)*I+i]*v;}
    __shared__ float s[16][256];
    #pragma unroll
    for(int t=0;t<16;t++)s[t][threadIdx.x]=sum[t];
    __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int t=0;t<16;t++)s[t][threadIdx.x]+=s[t][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x)for(int t=0;t<tcount;t++)y[(long long)(tbase+t)*O+o]=s[t][0];
}
extern "C" int dsv4_cuda_matmul_bf16_batch(Dsv4CudaTensor*t,const float*x,int tokens,float*y){
    Dev*c=t?ctx(t->device):nullptr;
    if(!c||!x||!y||tokens<1||t->fmt!=16||!ok(cudaSetDevice(t->device),"select bf16 batch device"))return 0;
    size_t xb=(size_t)tokens*t->I*4,yb=(size_t)tokens*t->O*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"bf16 batch upload"))return 0;
    mm_bf16_batch<<<dim3(t->O,(tokens+15)/16),256,0,c->stream>>>((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I,tokens);
    return ok(cudaGetLastError(),"bf16 batch launch")&&
           ok(cudaMemcpyAsync(y,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"bf16 batch download")&&
           ok(cudaStreamSynchronize(c->stream),"bf16 batch sync");
}
__device__ __forceinline__ int decode_pos(const int *state){return state[1];}
__device__ __forceinline__ int compression_ready(const int *state,int ratio){return ratio&&decode_pos(state)%ratio==ratio-1;}
__global__ void rope_heads_dynamic(float*x,int heads,int dim,int rope,const int*state,int delta,int inverse,int predicate,const float*cs,const float*sn){if(predicate&&!compression_ready(state,predicate))return;int pos=decode_pos(state)+delta,p=blockIdx.x*blockDim.x+threadIdx.x,pairs=rope/2;if(p<heads*pairs){int h=p/pairs,k=p%pairs,off=h*dim+dim-rope+2*k;float s=inverse?-sn[(long long)pos*pairs+k]:sn[(long long)pos*pairs+k],u=x[off],v=x[off+1];x[off]=__bfloat162float(__float2bfloat16(u*cs[(long long)pos*pairs+k]-v*s));x[off+1]=__bfloat162float(__float2bfloat16(u*s+v*cs[(long long)pos*pairs+k]));}}
__global__ void cache_store_dynamic(float*cache,const float*kv,const int*state,int window,int dim){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<dim)cache[(long long)(decode_pos(state)%window)*dim+d]=kv[d];}
/* Unconditional, not FLASHINFER-gated: the DeepGEMM batched wo path
 * (dsv4_cuda_wo) launches these too, and a DEEPGEMM=1 FLASHINFER=0 build
 * otherwise fails to compile. */
__global__ void gather_group_rows(float*out,const float*in,int tokens,int groups,int width,int group){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)tokens*width;if(p<n){int row=p/width,col=p%width;out[p]=in[((long long)row*groups+group)*width+col];}}
__global__ void scatter_group_rows(float*out,const float*in,int tokens,int groups,int width,int group){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)tokens*width;if(p<n){int row=p/width,col=p%width;out[((long long)row*groups+group)*width+col]=in[p];}}
__global__ void compress_store_dynamic(float*values,float*scores,const float*v,const float*g,const float*ape,const int*state,int ratio,int overlap,int O){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<O){int phase=decode_pos(state)%ratio,slot=(overlap?ratio:0)+phase;values[(long long)slot*O+d]=v[d];scores[(long long)slot*O+d]=g[d]+ape[(long long)phase*O+d];}}
__global__ void compress_pool_dynamic(float*out,const float*values,const float*scores,const int*state,int ratio,int overlap,int dim){if(!compression_ready(state,ratio))return;int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<dim){int O=(1+overlap)*dim,items=overlap?2*ratio:ratio;float mx=-INFINITY;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?dim+d:d;mx=fmaxf(mx,scores[(long long)j*O+col]);}float den=0,sum=0;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?dim+d:d,aoff=j*O+col;float a=expf(scores[aoff]-mx);den+=a;sum+=a*values[aoff];}out[d]=sum/den;}}
__global__ void rmsnorm_dynamic(float*out,const float*x,const float*w,int n,float eps,const int*state,int ratio){if(!compression_ready(state,ratio))return;__shared__ float sum[256],inv;float ss=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)ss+=x[i]*x[i];sum[threadIdx.x]=ss;__syncthreads();for(int k=128;k;k>>=1){if(threadIdx.x<k)sum[threadIdx.x]+=sum[threadIdx.x+k];__syncthreads();}if(!threadIdx.x)inv=rsqrtf(sum[0]/n+eps);__syncthreads();for(int i=threadIdx.x;i<n;i+=blockDim.x)out[i]=__bfloat162float(__float2bfloat16(x[i]*inv*w[i]));}
__global__ void fp8_sim64_dynamic(float*x,int n,const int*state,int ratio){if(!compression_ready(state,ratio))return;int b=blockIdx.x,d=b*64+threadIdx.x;if(d>=n)return;float mx=0;for(int k=0;k<64&&b*64+k<n;k++)mx=fmaxf(mx,fabsf(x[b*64+k]));float sc=fmaxf(mx,1e-4f)/448.f;x[d]=e4m3(__nv_cvt_float_to_fp8(x[d]/sc,__NV_SATFINITE,__NV_E4M3))*sc;}
__global__ void compressed_store_dynamic(float*cache,const float*kv,int*count,const int*state,int ratio,int max_count,int dim){if(!compression_ready(state,ratio))return;int d=blockIdx.x*blockDim.x+threadIdx.x,slot=*count;if(slot<max_count&&d<dim)cache[(long long)slot*dim+d]=kv[d];}
__global__ void compression_commit(int*count,float*values,float*scores,const int*state,int ratio,int overlap,int O,int max_count){if(!compression_ready(state,ratio))return;int d=blockIdx.x*blockDim.x+threadIdx.x;if(overlap&&d<ratio*O){values[d]=values[(long long)ratio*O+d];scores[d]=scores[(long long)ratio*O+d];}if(!blockIdx.x&&!threadIdx.x&&*count<max_count)(*count)++;}
__global__ void attention_window_dynamic(float*context,const float*q,const float*cache,const float*compressed,const float*sink,int heads,int dim,int window,const int*state,const int*cnp){
    int pos=decode_pos(state),cn=*cnp,h=blockIdx.x;if(h>=heads)return;extern __shared__ float score[];__shared__ float mx,den;
    int start=pos-window+1;if(start<0)start=0;int nw=pos-start+1,total=nw+cn,warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int item=warp;item<total;item+=blockDim.x/32){
        const float*v=item<nw?cache+(long long)((start+item)%window)*dim:compressed+(long long)(item-nw)*dim;float z=0;
        for(int d=lane;d<dim;d+=32)z+=q[(long long)h*dim+d]*v[d];
        for(int n=16;n;n>>=1)z+=__shfl_down_sync(0xffffffff,z,n);if(!lane)score[item]=z*rsqrtf((float)dim);
    }
    __syncthreads();if(!threadIdx.x){mx=sink[h];for(int i=0;i<total;i++)mx=fmaxf(mx,score[i]);den=expf(sink[h]-mx);for(int i=0;i<total;i++){float a=expf(score[i]-mx);score[i]=a;den+=a;}}
    __syncthreads();for(int d=threadIdx.x;d<dim;d+=blockDim.x){float sum=0;for(int i=0;i<nw;i++)sum+=score[i]*cache[(long long)((start+i)%window)*dim+d];for(int i=0;i<cn;i++)sum+=score[nw+i]*compressed[(long long)i*dim+d];context[(long long)h*dim+d]=__bfloat162float(__float2bfloat16(sum/den));}
}
/* Adapted from ds4_cuda.cu (MIT, Copyright 2026 ds4.c authors). */
__device__ __forceinline__ bool route_better(float av,unsigned ai,float bv,unsigned bi){return av>bv||(av==bv&&ai<bi);}
__device__ __forceinline__ float route_prob(float x){return sqrtf(log1pf(expf(-fabsf(x)))+fmaxf(x,0.f));}
__global__ void route_top6(int *selected,float *weights,const float *bias,const float *logits,int fixed,float scale){unsigned lane=threadIdx.x;if(lane>=32)return;float prob[8],score[8];
    #pragma unroll
    for(unsigned j=0;j<8;j++){unsigned e=lane+j*32;prob[j]=route_prob(logits[e]);score[j]=prob[j]+(bias?bias[e]:0.f);}
    if(fixed){__shared__ float all[256];for(unsigned j=0;j<8;j++)all[lane+j*32]=prob[j];__syncwarp();if(!lane){float sum=0;for(int k=0;k<6;k++){weights[k]=all[selected[k]];sum+=weights[k];}sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++)weights[k]=weights[k]/sum*scale;}return;}
    float outp[6]={};unsigned outi[6]={};
    #pragma unroll
    for(unsigned k=0;k<6;k++){float bs=-INFINITY,bp=0;unsigned bi=UINT_MAX;
        #pragma unroll
        for(unsigned j=0;j<8;j++){unsigned e=lane+j*32;if(route_better(score[j],e,bs,bi)){bs=score[j];bp=prob[j];bi=e;}}
        #pragma unroll
        for(unsigned mask=16;mask;mask>>=1){float os=__shfl_xor_sync(0xffffffff,bs,mask),op=__shfl_xor_sync(0xffffffff,bp,mask);unsigned oi=__shfl_xor_sync(0xffffffff,bi,mask);if(route_better(os,oi,bs,bi)){bs=os;bp=op;bi=oi;}}
        #pragma unroll
        for(unsigned j=0;j<8;j++)if(lane+j*32==bi)score[j]=-INFINITY;if(!lane){outp[k]=bp;outi[k]=bi;}}
    if(!lane){float sum=0;for(int k=0;k<6;k++){selected[k]=(int)outi[k];weights[k]=outp[k];sum+=outp[k];}sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++)weights[k]=weights[k]/sum*scale;}}
__global__ void route_top6_parallel(int *selected,float *weights,const float *bias,const float *logits,int fixed,float scale){unsigned e=threadIdx.x;__shared__ float prob[256];if(e<256)prob[e]=route_prob(logits[e]);__syncthreads();if(e)return;if(!fixed){for(int k=0;k<6;k++)selected[k]=-1;for(int i=0;i<256;i++){float s=prob[i]+(bias?bias[i]:0.f);for(int k=0;k<6;k++)if(selected[k]<0||route_better(s,i,prob[selected[k]]+(bias?bias[selected[k]]:0.f),(unsigned)selected[k])){for(int j=5;j>k;j--)selected[j]=selected[j-1];selected[k]=i;break;}}}float sum=0;for(int k=0;k<6;k++){weights[k]=prob[selected[k]];sum+=weights[k];}sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++)weights[k]=weights[k]/sum*scale;}
__global__ void route_top6_serial(int *selected,float *weights,const float *bias,const float *logits,int fixed,float scale){if(threadIdx.x||blockIdx.x)return;if(!fixed){for(int k=0;k<6;k++){int best=-1;float bv=-INFINITY;for(int i=0;i<256;i++){bool used=false;for(int j=0;j<k;j++)used|=selected[j]==i;if(used)continue;float s=route_prob(logits[i])+(bias?bias[i]:0.f);if(s>bv){bv=s;best=i;}}selected[k]=best;}}float sum=0;for(int k=0;k<6;k++){weights[k]=route_prob(logits[selected[k]]);sum+=weights[k];}sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++)weights[k]=weights[k]/sum*scale;}
__global__ void route_fixed6(int *selected,float *weights,const float *logits,float scale){int k=threadIdx.x;__shared__ float prob[6];if(k<6)prob[k]=route_prob(logits[selected[k]]);__syncthreads();if(!k){float sum=0;for(int i=0;i<6;i++)sum+=prob[i];sum=fmaxf(sum,6.103515625e-5f);for(int i=0;i<6;i++)weights[i]=prob[i]/sum*scale;}}
__global__ void build_moe_desc(MvDesc *desc,const ExpertPtr *table,const int *ids,const float *x,
                               float *gate,float *up,float *act,float *down,int count,int experts,int H,int I){
    int k=threadIdx.x;if(k>=count)return;int e=ids[k];if(e<0||e>=experts)return;
    ExpertPtr g=table[e],u=table[experts+e],d=table[2*experts+e];
    desc[2*k]={g.w,g.scale,x,gate+(long long)k*I};desc[2*k+1]={u.w,u.scale,x,up+(long long)k*I};
    desc[2*count+k]={d.w,d.scale,act+(long long)k*I,down+(long long)k*H};
}
__global__ void load_hash_ids(int *ids,const int64_t *map,const int *state,int topk){int k=threadIdx.x;if(k<topk)ids[k]=(int)map[(long long)state[0]*topk+k];}
__global__ void mhc_mix_scale(float *mix,const float *residual,const float *scale,const float *base,int M,int H,float eps){
    __shared__ float sum[256],inv;int MH=M*H,N=2*M+M*M;float ss=0;for(int i=threadIdx.x;i<MH;i+=blockDim.x)ss+=residual[i]*residual[i];
    sum[threadIdx.x]=ss;__syncthreads();for(int k=128;k;k>>=1){if(threadIdx.x<k)sum[threadIdx.x]+=sum[threadIdx.x+k];__syncthreads();}
    if(!threadIdx.x)inv=rsqrtf(sum[0]/MH+eps);__syncthreads();for(int n=threadIdx.x;n<N;n+=blockDim.x){int group=n<M?0:n<2*M?1:2;mix[n]=mix[n]*inv*scale[group]+base[n];}
}
__global__ void mhc_coeff(float *state,const float *mix,int M,float pre_eps,float sink_eps,float post_mult,int sink_iters){
    if(blockIdx.x)return;int t=threadIdx.x;float *post=state,*comb=state+M,*pre=comb+M*M;
    if(M==4){int lane=t&31;if(lane<4){pre[lane]=1.f/(1.f+expf(-mix[lane]))+pre_eps;post[lane]=(1.f/(1.f+expf(-mix[4+lane])))*post_mult;}
        float v=lane<16?mix[8+lane]:-INFINITY,mx=v;mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
        v=lane<16?expf(v-mx):0.f;float sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);if(lane<16)v=v/sum+sink_eps;
        for(int it=0;it<sink_iters;it++){if(it){sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);if(lane<16)v/=sum+sink_eps;}
            sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,4);sum+=__shfl_xor_sync(0xffffffff,sum,8);if(lane<16)v/=sum+sink_eps;}
        if(lane<16)comb[lane]=v;return;}
    if(t<M){pre[t]=1.f/(1.f+expf(-mix[t]))+pre_eps;post[t]=(1.f/(1.f+expf(-mix[M+t])))*post_mult;
        float mx=-INFINITY,sum=0;for(int j=0;j<M;j++)mx=fmaxf(mx,mix[2*M+t*M+j]);
        for(int j=0;j<M;j++){float v=expf(mix[2*M+t*M+j]-mx);comb[t*M+j]=v;sum+=v;}
        for(int j=0;j<M;j++)comb[t*M+j]=comb[t*M+j]/sum+sink_eps;}
    __syncthreads();for(int it=0;it<sink_iters;it++){
        if(it&&t<M){float sum=0;for(int j=0;j<M;j++)sum+=comb[t*M+j];for(int j=0;j<M;j++)comb[t*M+j]/=sum+sink_eps;}
        __syncthreads();if(t<M){float sum=0;for(int i=0;i<M;i++)sum+=comb[i*M+t];for(int i=0;i<M;i++)comb[i*M+t]/=sum+sink_eps;}
        __syncthreads();
    }
}
__global__ void mhc_input(float *out,const float *residual,const float *state,int M,int H){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<H){
    const float *pre=state+M+M*M;float v=0;for(int i=0;i<M;i++)v+=pre[i]*residual[(long long)i*H+h];out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void mhc_input_norm(float*out,const float*residual,const float*state,const float*w,int M,int H,float eps){__shared__ float ws[8],inv;const float*pre=state+M+M*M;float ss=0;int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int h=threadIdx.x;h<H;h+=blockDim.x){float v=0;for(int i=0;i<M;i++)v+=pre[i]*residual[(long long)i*H+h];v=__bfloat162float(__float2bfloat16(v));out[h]=v;ss+=v*v;}ss=warp_sum_f32(ss);if(!lane)ws[warp]=ss;__syncthreads();if(!warp){float v=lane<8?ws[lane]:0.f;v=warp_sum_f32(v);if(!lane)inv=rsqrtf(v/H+eps);}__syncthreads();for(int h=threadIdx.x;h<H;h+=blockDim.x)out[h]=__bfloat162float(__float2bfloat16(out[h]*inv*w[h]));}
__global__ void mhc_output(float *out,const float *x,const float *residual,const float *state,int M,int H){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<M*H){
    int j=p/H,h=p%H;const float *post=state,*comb=state+M;float v=post[j]*x[h];for(int i=0;i<M;i++)v+=comb[i*M+j]*residual[(long long)i*H+h];out[p]=__bfloat162float(__float2bfloat16(v));}}
__global__ void final_mhc_coeff(float*pre,const float*res,const float*fn,const float*scale,const float*base,int M,int H,float eps,float pre_eps){__shared__ float sum[5][256];int MH=M*H;float v[5]={};
    for(int i=threadIdx.x;i<MH;i+=blockDim.x){float x=res[i];v[0]+=x*x;for(int m=0;m<M;m++)v[m+1]+=x*fn[(long long)m*MH+i];}for(int m=0;m<=M;m++)sum[m][threadIdx.x]=v[m];__syncthreads();
    for(int k=128;k;k>>=1){if(threadIdx.x<k)for(int m=0;m<=M;m++)sum[m][threadIdx.x]+=sum[m][threadIdx.x+k];__syncthreads();}if(threadIdx.x<M){float inv=rsqrtf(sum[0][0]/MH+eps);pre[threadIdx.x]=1.f/(1.f+expf(-(sum[threadIdx.x+1][0]*inv*scale[0]+base[threadIdx.x])))+pre_eps;}}
__global__ void final_mhc_mix(float*out,const float*res,const float*pre,int M,int H){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<H){float v=0;for(int m=0;m<M;m++)v+=pre[m]*res[(long long)m*H+h];out[h]=__bfloat162float(__float2bfloat16(v));}}

extern "C" int dsv4_cuda_init(const int *dev,int n){if(n<1||n>16)return 0;float scale[256];for(int b=0;b<256;b++)scale[b]=b?ldexpf(1.f,b-127):ldexpf(1.f,-127);ng=0;for(int i=0;i<n;i++){cudaDeviceProp p{};if(!ok(cudaSetDevice(dev[i]),"select device")||!ok(cudaMemcpyToSymbol(e8_table,scale,sizeof(scale)),"scale table upload")||!ok(cudaGetDeviceProperties(&p,dev[i]),"device properties")||!ok(cudaStreamCreateWithFlags(&g[i].stream,cudaStreamNonBlocking),"stream")||!ok(cudaStreamCreateWithFlags(&g[i].aux,cudaStreamNonBlocking),"aux stream")||!ok(cudaEventCreateWithFlags(&g[i].ready,cudaEventDisableTiming),"device event")||!ok(cudaEventCreateWithFlags(&g[i].fork,cudaEventDisableTiming),"fork event")||!ok(cudaEventCreateWithFlags(&g[i].join,cudaEventDisableTiming),"join event")||!bok(cublasLtCreate(&g[i].lt),"cuBLASLt create")||!ok(cudaMalloc(&g[i].tc_workspace,32<<20),"Tensor Core workspace")||!ok(cudaMalloc(&g[i].decode_state,2*sizeof(int)),"decode state allocation"))return 0;g[i].id=dev[i];g[i].sms=p.multiProcessorCount;ng++;fprintf(stderr,"[DSV4 CUDA] device %d: %s %.1f GB sm_%d%d\n",dev[i],p.name,p.totalGlobalMem/1e9,p.major,p.minor);}for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(i!=j){int can=0;if(!ok(cudaDeviceCanAccessPeer(&can,dev[i],dev[j]),"peer query"))return 0;if(can){if(!ok(cudaSetDevice(dev[i]),"select peer device"))return 0;cudaError_t e=cudaDeviceEnablePeerAccess(dev[j],0);if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)return ok(e,"enable peer access");}}
#ifdef COLI_DSV4_NCCL
    if(n==6)for(int pair=0;pair<3;pair++){int ids[2]={dev[2*pair],dev[2*pair+1]};ncclComm_t comm[2];if(!nok(ncclCommInitAll(comm,2,ids),"EP2 communicator init"))return 0;g[2*pair].ep_comm=comm[0];g[2*pair+1].ep_comm=comm[1];}
#endif
    return 1;}
static void plan_free(TcPlan*p){if(p->d)cublasLtMatrixLayoutDestroy(p->d);if(p->c)cublasLtMatrixLayoutDestroy(p->c);if(p->b)cublasLtMatrixLayoutDestroy(p->b);if(p->a)cublasLtMatrixLayoutDestroy(p->a);if(p->op)cublasLtMatmulDescDestroy(p->op);memset(p,0,sizeof(*p));}
extern "C" void dsv4_cuda_shutdown(void){
    for(int i=0;i<ng;i++){cudaSetDevice(g[i].id);if(g[i].decode_state)cudaFree(g[i].decode_state);
#ifdef COLI_DSV4_NCCL
        if(g[i].ep_comm)ncclCommDestroy(g[i].ep_comm);
#endif
    }
    for(int i=0;i<ng;i++){cudaSetDevice(g[i].id);if(g[i].dx)cudaFree(g[i].dx);if(g[i].dy)cudaFree(g[i].dy);if(g[i].p1)cudaFree(g[i].p1);if(g[i].p2)cudaFree(g[i].p2);if(g[i].p3)cudaFree(g[i].p3);if(g[i].p4)cudaFree(g[i].p4);if(g[i].aux1)cudaFree(g[i].aux1);if(g[i].aux2)cudaFree(g[i].aux2);if(g[i].aux3)cudaFree(g[i].aux3);if(g[i].mvdesc)cudaFree(g[i].mvdesc);if(g[i].expert_weights)cudaFree(g[i].expert_weights);if(g[i].expert_ids)cudaFree(g[i].expert_ids);if(g[i].tcw)cudaFree(g[i].tcw);if(g[i].tcs)cudaFree(g[i].tcs);if(g[i].tcx)cudaFree(g[i].tcx);if(g[i].tcxs)cudaFree(g[i].tcxs);if(g[i].tc_workspace)cudaFree(g[i].tc_workspace);
#ifdef COLI_DSV4_DEEPGEMM
        if(g[i].dga1)cudaFree(g[i].dga1);if(g[i].dga2)cudaFree(g[i].dga2);if(g[i].dgsfa1)cudaFree(g[i].dgsfa1);if(g[i].dgsfa2)cudaFree(g[i].dgsfa2);if(g[i].dglayout)cudaFree(g[i].dglayout);if(g[i].dgrowmap)cudaFree(g[i].dgrowmap);if(g[i].dgmm1)cudaFree(g[i].dgmm1);if(g[i].dgmm2)cudaFree(g[i].dgmm2);if(g[i].dgref)cudaFree(g[i].dgref);if(g[i].dgdensews)cudaFree(g[i].dgdensews);
#endif
        for(int p=0;p<2;p++)plan_free(&g[i].plans[p]);if(g[i].lt)cublasLtDestroy(g[i].lt);if(g[i].hx)cudaFreeHost(g[i].hx);if(g[i].hy)cudaFreeHost(g[i].hy);if(g[i].ready)cudaEventDestroy(g[i].ready);if(g[i].fork)cudaEventDestroy(g[i].fork);if(g[i].join)cudaEventDestroy(g[i].join);cudaStreamDestroy(g[i].aux);cudaStreamDestroy(g[i].stream);}ng=0;}
extern "C" Dsv4CudaActivation *dsv4_cuda_activation_create(int device,long long elements){Dev*c=ctx(device);if(!c||elements<1||!ok(cudaSetDevice(device),"select activation device"))return nullptr;Dsv4CudaActivation*a=(Dsv4CudaActivation*)calloc(1,sizeof(*a));if(!a)return nullptr;a->device=device;a->elements=elements;if(!ok(cudaMalloc(&a->data,(size_t)elements*sizeof(float)),"activation allocation")){free(a);return nullptr;}return a;}
extern "C" void dsv4_cuda_activation_free(Dsv4CudaActivation*a){if(!a)return;cudaSetDevice(a->device);cudaFree(a->data);free(a);}
extern "C" int dsv4_cuda_activation_upload(Dsv4CudaActivation*a,const float*x,long long n){Dev*c=a?ctx(a->device):nullptr;if(!c||!x||n<0||n>a->elements||!ok(cudaSetDevice(a->device),"select activation upload device"))return 0;return ok(cudaMemcpyAsync(a->data,x,(size_t)n*sizeof(float),cudaMemcpyHostToDevice,c->stream),"activation upload");}
extern "C" int dsv4_cuda_activation_download(float*x,const Dsv4CudaActivation*a,long long n){Dev*c=a?ctx(a->device):nullptr;if(!c||!x||n<0||n>a->elements||!ok(cudaSetDevice(a->device),"select activation download device"))return 0;return ok(cudaMemcpyAsync(x,a->data,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost,c->stream),"activation download");}
extern "C" int dsv4_cuda_activation_copy(Dsv4CudaActivation*dst,const Dsv4CudaActivation*src,long long n){Dev*c=dst?ctx(dst->device):nullptr,*s=src?ctx(src->device):nullptr;if(!c||!s||n<0||n>dst->elements||n>src->elements)return 0;
    if(dst->device==src->device)return ok(cudaSetDevice(dst->device),"select local activation copy")&&ok(cudaMemcpyAsync(dst->data,src->data,(size_t)n*sizeof(float),cudaMemcpyDeviceToDevice,c->stream),"local activation copy");
    if(!ok(cudaSetDevice(src->device),"select activation source")||!ok(cudaEventRecord(s->ready,s->stream),"record activation readiness")||!ok(cudaSetDevice(dst->device),"select activation copy device")||!ok(cudaStreamWaitEvent(c->stream,s->ready,0),"wait for activation"))return 0;
    return ok(cudaMemcpyPeerAsync(dst->data,dst->device,src->data,src->device,(size_t)n*sizeof(float),c->stream),"activation device copy");}
extern "C" int dsv4_cuda_activation_copy_range(Dsv4CudaActivation*dst,long long doff,const Dsv4CudaActivation*src,long long soff,long long n){Dev*c=dst?ctx(dst->device):nullptr,*s=src?ctx(src->device):nullptr;if(!c||!s||doff<0||soff<0||n<0||doff+n>dst->elements||soff+n>src->elements)return 0;
    if(dst->device==src->device)return ok(cudaSetDevice(dst->device),"select local activation range copy")&&ok(cudaMemcpyAsync(dst->data+doff,src->data+soff,(size_t)n*sizeof(float),cudaMemcpyDeviceToDevice,c->stream),"local activation range copy");
    if(!ok(cudaSetDevice(src->device),"select activation range source")||!ok(cudaEventRecord(s->ready,s->stream),"record activation range readiness")||!ok(cudaSetDevice(dst->device),"select activation range copy device")||!ok(cudaStreamWaitEvent(c->stream,s->ready,0),"wait for activation range"))return 0;
    return ok(cudaMemcpyPeerAsync(dst->data+doff,dst->device,src->data+soff,src->device,(size_t)n*sizeof(float),c->stream),"activation range device copy");}
extern "C" int dsv4_cuda_activation_sync(const Dsv4CudaActivation*a){Dev*c=a?ctx(a->device):nullptr;return c&&ok(cudaSetDevice(a->device),"select activation sync device")&&ok(cudaStreamSynchronize(c->stream),"activation sync");}
extern "C" int dsv4_cuda_activation_device(const Dsv4CudaActivation*a){return a?a->device:-1;}
extern "C" int dsv4_cuda_decode_state_set(int device,int token,int position){Dev*c=ctx(device);if(!c||token<0||position<0||!ok(cudaSetDevice(device),"select decode state device"))return 0;set_decode_state<<<1,1,0,c->stream>>>(c->decode_state,token,position);return ok(cudaGetLastError(),"decode state update");}
extern "C" void dsv4_cuda_profiler_start(void){cudaProfilerStart();}
extern "C" void dsv4_cuda_profiler_stop(void){for(int i=0;i<ng;i++){cudaSetDevice(g[i].id);cudaStreamSynchronize(g[i].stream);}cudaProfilerStop();}
extern "C" int dsv4_cuda_graph_begin(int device){Dev*c=ctx(device);return c&&ok(cudaSetDevice(device),"select graph device")&&ok(cudaStreamBeginCapture(c->stream,cudaStreamCaptureModeThreadLocal),"begin graph capture");}
extern "C" Dsv4CudaGraph *dsv4_cuda_graph_end(int device){Dev*c=ctx(device);if(!c||!ok(cudaSetDevice(device),"select graph device"))return nullptr;Dsv4CudaGraph*g=(Dsv4CudaGraph*)calloc(1,sizeof(*g));if(!g)return nullptr;g->device=device;if(!ok(cudaStreamEndCapture(c->stream,&g->graph),"end graph capture")||!ok(cudaGraphInstantiate(&g->exec,g->graph,0),"instantiate graph")){dsv4_cuda_graph_free(g);return nullptr;}return g;}
extern "C" int dsv4_cuda_graph_end_pair(int primary,int peer,Dsv4CudaGraph **pg,Dsv4CudaGraph **qg){
    Dev*p=ctx(primary),*q=ctx(peer);if(!p||!q||!pg||!qg||*pg||*qg)return 0;
    Dsv4CudaGraph*a=(Dsv4CudaGraph*)calloc(1,sizeof(*a)),*b=(Dsv4CudaGraph*)calloc(1,sizeof(*b));if(!a||!b){free(a);free(b);return 0;}
    a->device=primary;b->device=peer;
    if(!ok(cudaSetDevice(peer),"select peer graph device")||!ok(cudaStreamEndCapture(q->stream,&b->graph),"end peer graph capture")||
       !ok(cudaSetDevice(primary),"select primary graph device")||!ok(cudaStreamEndCapture(p->stream,&a->graph),"end primary graph capture")||
       !ok(cudaGraphInstantiate(&b->exec,b->graph,0),"instantiate peer graph")||!ok(cudaGraphInstantiate(&a->exec,a->graph,0),"instantiate primary graph")){
        dsv4_cuda_graph_free(a);dsv4_cuda_graph_free(b);return 0;}
    *pg=a;*qg=b;return 1;
}
extern "C" int dsv4_cuda_graph_launch(Dsv4CudaGraph*g){Dev*c=g?ctx(g->device):nullptr;return c&&ok(cudaSetDevice(g->device),"select graph device")&&ok(cudaGraphLaunch(g->exec,c->stream),"launch graph");}
extern "C" void dsv4_cuda_graph_free(Dsv4CudaGraph*g){if(!g)return;cudaSetDevice(g->device);if(g->exec)cudaGraphExecDestroy(g->exec);if(g->graph)cudaGraphDestroy(g->graph);free(g);}
static int upload(Dsv4CudaTensor **pt,const void *w,size_t wb,const uint8_t *sc,size_t sb,int O,int I,int d,int fmt){if(*pt)return 1;Dev *c=ctx(d);if(!c||!ok(cudaSetDevice(d),"select upload device"))return 0;Dsv4CudaTensor *t=(Dsv4CudaTensor*)calloc(1,sizeof(*t));t->O=O;t->I=I;t->device=d;t->fmt=fmt;t->bytes=wb+sb;
    if(!ok(cudaMalloc(&t->w,wb),"weight allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_w=1;if(!ok(cudaMemcpy(t->w,w,wb,cudaMemcpyHostToDevice),"weight upload")){dsv4_cuda_tensor_free(t);return 0;}
    if(sb){if(!ok(cudaMalloc(&t->scale,sb),"scale allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_scale=1;if(!ok(cudaMemcpy(t->scale,sc,sb,cudaMemcpyHostToDevice),"scale upload")){dsv4_cuda_tensor_free(t);return 0;}}
#ifdef COLI_DSV4_DEEPGEMM
    if((fmt==8||fmt==9)&&O%128==0&&I%128==0){size_t db=(size_t)O*(I/128/4)*sizeof(int);if(!ok(cudaMalloc(&t->dg_scale,db),"dense DeepGEMM scale allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_dg_scale=1;
        if(fmt==9){int groups=O/1024;dg_pack_batched_scale<<<(O*(I/128/4)+255)/256,256,0,c->stream>>>(t->dg_scale,t->scale,O/groups,I/128,groups);}
        else dg_pack_dense_scale<<<(O*(I/128/4)+255)/256,256,0,c->stream>>>(t->dg_scale,t->scale,O,I/128);
        if(!ok(cudaGetLastError(),"dense DeepGEMM scale packing")){dsv4_cuda_tensor_free(t);return 0;}t->bytes+=db;}
#endif
    *pt=t;return 1;}
extern "C" int dsv4_cuda_upload_fp8(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){return upload(t,w,(size_t)O*I,s,(size_t)((O+127)/128)*((I+127)/128),O,I,d,8);}
extern "C" int dsv4_cuda_upload_fp8_bf16(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){return upload(t,w,(size_t)O*I,s,(size_t)((O+127)/128)*((I+127)/128),O,I,d,9);}
extern "C" int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){if(!upload(t,w,(size_t)O*I/2,s,(size_t)O*I/32,O,I,d,4))return 0;
    return 1;}
/* In-place refill of an existing fp4 mirror with a same-shape expert: no
 * cudaFree/cudaMalloc churn, DMA from page-locked slabs when possible. The
 * copies are stream-ordered; sync=1 drains before returning (needed when the
 * host slab may be released before the next stream sync). */
static bool host_pinned(const void*p,size_t len);
extern "C" int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t,const uint8_t*w,const uint8_t*s,int O,int I,int sync){
    if(!t||!w||!s||t->fmt!=4||t->O!=O||t->I!=I||!t->own_w||!t->own_scale||!t->w||!t->scale)return 0;
    Dev*c=ctx(t->device);if(!c||!ok(cudaSetDevice(t->device),"select refill device"))return 0;
    size_t wb=(size_t)O*I/2,sb=(size_t)O*I/32;
    if(host_pinned(w,wb)&&host_pinned(s,sb)){
        if(!ok(cudaMemcpyAsync(t->w,w,wb,cudaMemcpyHostToDevice,c->stream),"refill weight upload")||
           !ok(cudaMemcpyAsync(t->scale,s,sb,cudaMemcpyHostToDevice,c->stream),"refill scale upload"))return 0;
        if(sync&&!ok(cudaStreamSynchronize(c->stream),"refill drain"))return 0;
    }else{
        if(!ok(cudaMemcpy(t->w,w,wb,cudaMemcpyHostToDevice),"refill weight upload")||
           !ok(cudaMemcpy(t->scale,s,sb,cudaMemcpyHostToDevice),"refill scale upload"))return 0;
    }
    return 1;
}
extern "C" int dsv4_cuda_upload_bf16(Dsv4CudaTensor **t,const uint16_t*w,int O,int I,int d){return upload(t,w,(size_t)O*I*2,nullptr,0,O,I,d,16);}
extern "C" int dsv4_cuda_upload_f32(Dsv4CudaTensor **t,const float*w,int O,int I,int d){return upload(t,w,(size_t)O*I*4,nullptr,0,O,I,d,32);}
extern "C" int dsv4_cuda_mhc_pre(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,int M,int H,float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,Dsv4CudaActivation*state,Dsv4CudaActivation*input){
    int N=2*M+M*M,MH=M*H;Dev*c=r?ctx(r->device):nullptr;if(!c||!fn||!scale||!base||!state||!input||fn->fmt!=32||scale->fmt!=32||base->fmt!=32||fn->device!=r->device||scale->device!=r->device||base->device!=r->device||state->device!=r->device||input->device!=r->device||r->elements<MH||fn->O!=N||fn->I!=MH||scale->O*scale->I<3||base->O*base->I<N||state->elements<M+M*M+M||input->elements<H||!ok(cudaSetDevice(r->device),"select mHC device")||!buf((void**)&c->p1,&c->p1cap,(size_t)N*4))return 0;
    if(N==24&&M==4){constexpr int parts=8;if(!buf((void**)&c->dy,&c->ycap,(size_t)(24+1)*parts*sizeof(float)))return 0;mhc_fn_split<<<dim3(parts,N),256,0,c->stream>>>((float*)fn->w,r->data,c->dy,N,MH,parts);mhc_fn_finish<<<1,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M,MH,rms_eps,parts);}
    else{mv_f32<<<N,256,0,c->stream>>>((float*)fn->w,r->data,c->p1,N,MH);mhc_mix_scale<<<1,256,0,c->stream>>>(c->p1,r->data,(float*)scale->w,(float*)base->w,M,H,rms_eps);}
    mhc_coeff<<<1,32,0,c->stream>>>(state->data,c->p1,M,pre_eps,sink_eps,post_mult,sink_iters);mhc_input<<<(H+255)/256,256,0,c->stream>>>(input->data,r->data,state->data,M,H);return ok(cudaGetLastError(),"mHC pre launch");}
extern "C" int dsv4_cuda_mhc_post(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,const Dsv4CudaActivation*state,int M,int H,Dsv4CudaActivation*out){
    Dev*c=x?ctx(x->device):nullptr;if(!c||!r||!state||!out||r->device!=x->device||state->device!=x->device||out->device!=x->device||x->elements<H||r->elements<M*H||state->elements<M+M*M+M||out->elements<M*H||!ok(cudaSetDevice(x->device),"select mHC device"))return 0;
    mhc_output<<<(M*H+255)/256,256,0,c->stream>>>(out->data,x->data,r->data,state->data,M,H);return ok(cudaGetLastError(),"mHC post launch");}
extern "C" int dsv4_cuda_attention_first(const Dsv4CudaActivation*input,Dsv4CudaTensor*an,Dsv4CudaTensor*qa,Dsv4CudaTensor*qn,Dsv4CudaTensor*qb,Dsv4CudaTensor*kv,Dsv4CudaTensor*kvn,Dsv4CudaTensor*sink,Dsv4CudaTensor*wa,Dsv4CudaTensor*wb,int heads,int dim,int qk_rope,int groups,float eps,Dsv4CudaActivation*output){
    Dev*c=input?ctx(input->device):nullptr;int H=qa?qa->I:0,R=qa?qa->O:0,Q=heads*dim,K=dim,WR=wa?wa->O:0;if(!c||!output||!an||!qa||!qn||!qb||!kv||!kvn||!sink||!wa||!wb||input->elements<H||output->elements<H||an->fmt!=32||qn->fmt!=16||kvn->fmt!=32||sink->fmt!=32||qa->fmt!=8||qb->fmt!=8||kv->fmt!=8||wa->fmt!=9||wb->fmt!=8||an->O*an->I<H||qn->O*qn->I<R||kvn->O*kvn->I<K||sink->O*sink->I<heads||qb->I!=R||qb->O!=Q||kv->I!=H||kv->O!=K||wa->I*groups!=Q||wb->I!=WR||wb->O!=H||heads%groups)return 0;
    Dsv4CudaTensor*tensors[]={an,qa,qn,qb,kv,kvn,sink,wa,wb};for(auto*t:tensors)if(t->device!=input->device)return 0;int P1=R>WR?R:WR;if(output->device!=input->device||!ok(cudaSetDevice(input->device),"select attention device")||!buf((void**)&c->dx,&c->xcap,(size_t)H*4)||!buf((void**)&c->p1,&c->p1cap,(size_t)P1*4)||!buf((void**)&c->p2,&c->p2cap,(size_t)Q*4)||!buf((void**)&c->p3,&c->p3cap,(size_t)K*4)||!buf((void**)&c->p4,&c->p4cap,(size_t)Q*4))return 0;
    int dense_dg=getenv("DSV4_CUDA_DENSE_DG")&&atoi(getenv("DSV4_CUDA_DENSE_DG"));
    rmsnorm_f32<<<1,256,0,c->stream>>>(c->dx,input->data,(float*)an->w,H,eps);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,qa,c->dx,c->p1)||!dg_attention_kv_reuse(c,kv,c->dx,c->p3))return 0;}else
#endif
    {fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);run_mv<8>((uint8_t*)qa->w,qa->scale,c->dx,c->p1,R,H,1,c->stream);}
    bf16_round<<<(R+255)/256,256,0,c->stream>>>(c->p1,R);rmsnorm_seq<<<1,256,0,c->stream>>>(c->p1,(__nv_bfloat16*)qn->w,R,eps);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,qb,c->p1,c->p2))return 0;}else
#endif
    {fp8_sim<<<(R+127)/128,128,0,c->stream>>>(c->p1,1,R);run_mv<8>((uint8_t*)qb->w,qb->scale,c->p1,c->p2,Q,R,1,c->stream);}
    bf16_round<<<(Q+255)/256,256,0,c->stream>>>(c->p2,Q);head_rmsnorm<<<heads,256,0,c->stream>>>(c->p2,heads,dim,eps);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){}else
#endif
    run_mv<8>((uint8_t*)kv->w,kv->scale,c->dx,c->p3,K,H,1,c->stream);
    bf16_round<<<(K+255)/256,256,0,c->stream>>>(c->p3,K);rmsnorm_f32<<<1,256,0,c->stream>>>(c->p3,c->p3,(float*)kvn->w,K,eps);if(K>qk_rope)fp8_sim64<<<(K-qk_rope+63)/64,64,0,c->stream>>>(c->p3,K-qk_rope);
    attention_one<<<heads,256,0,c->stream>>>(c->p4,c->p2,c->p3,(float*)sink->w,heads,dim);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_wo_a(c,wa,c->p4,c->p1))return 0;}else
#endif
    run_mv<9>((uint8_t*)wa->w,wa->scale,c->p4,c->p1,WR,wa->I,groups,c->stream);
    bf16_round<<<(WR+255)/256,256,0,c->stream>>>(c->p1,WR);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,wb,c->p1,output->data))return 0;}else
#endif
    {fp8_sim<<<(WR+127)/128,128,0,c->stream>>>(c->p1,1,WR);run_mv<8>((uint8_t*)wb->w,wb->scale,c->p1,output->data,H,WR,1,c->stream);}
    bf16_round<<<(H+255)/256,256,0,c->stream>>>(output->data,H);return ok(cudaGetLastError(),"attention first launch");}
extern "C" Dsv4CudaKvCache *dsv4_cuda_kv_create(int device, int window, int dim, int max_tokens, int rope_pairs,
                                                const float *cosv, const float *sinv, const float *ccos,
                                                const float *csin) {
    Dev *c = ctx(device);
    if (!c || window < 1 || dim < 1 || max_tokens < 1 || rope_pairs < 0 ||
        (rope_pairs && (!cosv || !sinv || !ccos || !csin)) || !ok(cudaSetDevice(device), "select KV device"))
        return nullptr;
    Dsv4CudaKvCache *k = (Dsv4CudaKvCache *)calloc(1, sizeof(*k));
    if (!k) return nullptr;
    k->device = device;
    k->window = window;
    k->dim = dim;
    k->max_tokens = max_tokens;
    k->max_compressed = (max_tokens + 3) / 4;
    k->rope_pairs = rope_pairs;
    size_t kb = (size_t)window * dim * 4, cb = (size_t)k->max_compressed * dim * 4, pb = (size_t)2 * window * dim * 4,
           rb = (size_t)max_tokens * rope_pairs * 4;
    if (!ok(cudaMalloc(&k->kv, kb), "KV allocation") || !ok(cudaMemset(k->kv, 0, kb), "KV clear") ||
        !ok(cudaMalloc(&k->compressed, cb), "compressed KV allocation") ||
        !ok(cudaMalloc(&k->compressed_len, sizeof(int)), "compressed length allocation") ||
        !ok(cudaMemset(k->compressed_len, 0, sizeof(int)), "compressed length clear") ||
        !ok(cudaMalloc(&k->comp_value, pb), "compress value allocation") ||
        !ok(cudaMalloc(&k->comp_score, pb), "compress score allocation") ||
        !ok(cudaMemset(k->compressed, 0, cb), "compressed KV clear") ||
        (rb && (!ok(cudaMalloc(&k->rope_cos, rb), "RoPE cos allocation") ||
                !ok(cudaMalloc(&k->rope_sin, rb), "RoPE sin allocation") ||
                !ok(cudaMalloc(&k->comp_cos, rb), "compress RoPE cos allocation") ||
                !ok(cudaMalloc(&k->comp_sin, rb), "compress RoPE sin allocation") ||
                !ok(cudaMemcpy(k->rope_cos, cosv, rb, cudaMemcpyHostToDevice), "RoPE cos upload") ||
                !ok(cudaMemcpy(k->rope_sin, sinv, rb, cudaMemcpyHostToDevice), "RoPE sin upload") ||
                !ok(cudaMemcpy(k->comp_cos, ccos, rb, cudaMemcpyHostToDevice), "compress RoPE cos upload") ||
                !ok(cudaMemcpy(k->comp_sin, csin, rb, cudaMemcpyHostToDevice), "compress RoPE sin upload")))) {
        dsv4_cuda_kv_free(k);
        return nullptr;
    }
    fill_value<<<(pb / 4 + 255) / 256, 256, 0, c->stream>>>(k->comp_score, pb / 4, -INFINITY);
    if (!ok(cudaGetLastError(), "compress score clear")) {
        dsv4_cuda_kv_free(k);
        return nullptr;
    }
    return k;
}
extern "C" int dsv4_cuda_mhc_pre_norm(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,int M,int H,float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,float norm_eps,Dsv4CudaActivation*state,Dsv4CudaActivation*input){
    int N=2*M+M*M,MH=M*H;Dev*c=r?ctx(r->device):nullptr;if(!c||!fn||!scale||!base||!norm||!state||!input||M!=4||fn->fmt!=32||scale->fmt!=32||base->fmt!=32||norm->fmt!=32||fn->device!=r->device||scale->device!=r->device||base->device!=r->device||norm->device!=r->device||state->device!=r->device||input->device!=r->device||r->elements<MH||fn->O!=N||fn->I!=MH||norm->O*norm->I<H||!ok(cudaSetDevice(r->device),"select normalized mHC device")||!buf((void**)&c->p1,&c->p1cap,(size_t)N*4)||!buf((void**)&c->dy,&c->ycap,(size_t)(N+1)*8*sizeof(float)))return 0;
    mhc_fn_split<<<dim3(8,N),256,0,c->stream>>>((float*)fn->w,r->data,c->dy,N,MH,8);mhc_fn_finish<<<1,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M,MH,rms_eps,8);mhc_coeff<<<1,32,0,c->stream>>>(state->data,c->p1,M,pre_eps,sink_eps,post_mult,sink_iters);mhc_input_norm<<<1,256,0,c->stream>>>(input->data,r->data,state->data,(float*)norm->w,M,H,norm_eps);return ok(cudaGetLastError(),"normalized mHC pre launch");
}
extern "C" int dsv4_cuda_mhc_pre_norm_batch(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,int tokens,int H,Dsv4CudaActivation*state,Dsv4CudaActivation*input){
    constexpr int M=4,N=24,parts=8;long long RH=(long long)tokens*M*H,XH=(long long)tokens*H;Dev*c=r?ctx(r->device):nullptr;
    if(!c||!fn||!scale||!base||!norm||!state||!input||tokens<1||tokens>65535||H!=4096||r->elements<RH||state->elements<(long long)tokens*20||input->elements<XH||
       fn->fmt!=32||scale->fmt!=32||base->fmt!=32||norm->fmt!=32||fn->device!=r->device||scale->device!=r->device||base->device!=r->device||norm->device!=r->device||state->device!=r->device||input->device!=r->device||
       fn->O!=N||fn->I!=M*H||scale->O*scale->I<3||base->O*base->I<N||norm->O*norm->I<H||!ok(cudaSetDevice(r->device),"select initial batched mHC device")||
       !buf((void**)&c->p1,&c->p1cap,(size_t)tokens*N*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(size_t)tokens*(N+1)*parts*sizeof(float)))return 0;
    mhc_fn_split_batch<<<dim3(parts,N,tokens),256,0,c->stream>>>((float*)fn->w,r->data,c->dy,tokens,N,M*H,parts);
    mhc_fn_finish_batch<<<tokens,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M*H,1e-6f,parts);
    float*post=state->data,*comb=post+(long long)tokens*M;mhc_coeff_batch<<<tokens,32,0,c->stream>>>(post,comb,c->p1);
    mhc_input_norm_batch<<<tokens,256,0,c->stream>>>(input->data,r->data,c->p1,(float*)norm->w,H);
    return ok(cudaGetLastError(),"initial batched mHC launch");
}
extern "C" int dsv4_cuda_mhc_pre_batch(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,int tokens,int H,Dsv4CudaActivation*state,Dsv4CudaActivation*input){
    constexpr int M=4,N=24,parts=8;long long RH=(long long)tokens*M*H,XH=(long long)tokens*H;Dev*c=r?ctx(r->device):nullptr;
    if(!c||!fn||!scale||!base||!state||!input||tokens<1||tokens>65535||H!=4096||r->elements<RH||state->elements<(long long)tokens*20||input->elements<XH||fn->fmt!=32||scale->fmt!=32||base->fmt!=32||fn->device!=r->device||scale->device!=r->device||base->device!=r->device||state->device!=r->device||input->device!=r->device||fn->O!=N||fn->I!=M*H||scale->O*scale->I<3||base->O*base->I<N||!ok(cudaSetDevice(r->device),"select initial batched mHC device")||!buf((void**)&c->p1,&c->p1cap,(size_t)tokens*N*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(size_t)tokens*(N+1)*parts*sizeof(float)))return 0;
    mhc_fn_split_batch<<<dim3(parts,N,tokens),256,0,c->stream>>>((float*)fn->w,r->data,c->dy,tokens,N,M*H,parts);mhc_fn_finish_batch<<<tokens,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M*H,1e-6f,parts);float*post=state->data,*comb=post+(long long)tokens*M;mhc_coeff_batch<<<tokens,32,0,c->stream>>>(post,comb,c->p1);mhc_input_batch<<<tokens,256,0,c->stream>>>(input->data,r->data,c->p1,H);return ok(cudaGetLastError(),"initial batched mHC launch");
}
extern "C" int dsv4_cuda_mhc_post_pre(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,Dsv4CudaActivation*state,int M,int H,Dsv4CudaActivation*out,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,Dsv4CudaActivation*input){
    constexpr int N=24,parts=8;int I=M*H;Dev*c=x?ctx(x->device):nullptr;if(!c||!r||!state||!out||!fn||!scale||!base||!input||M!=4||r->device!=x->device||state->device!=x->device||out->device!=x->device||fn->device!=x->device||scale->device!=x->device||base->device!=x->device||input->device!=x->device||x->elements<H||r->elements<I||state->elements<M+M*M+M||out->elements<I||fn->fmt!=32||fn->O!=N||fn->I!=I||!ok(cudaSetDevice(x->device),"select fused mHC device")||!buf((void**)&c->p1,&c->p1cap,N*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(N+1)*parts*sizeof(float)))return 0;
    mhc_post_fn_split<<<parts,256,0,c->stream>>>((float*)fn->w,x->data,r->data,state->data,out->data,c->dy,N,M,H,parts);
    mhc_fn_finish<<<1,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M,I,rms_eps,parts);
    mhc_coeff<<<1,32,0,c->stream>>>(state->data,c->p1,M,pre_eps,sink_eps,post_mult,sink_iters);
    mhc_input<<<(H+255)/256,256,0,c->stream>>>(input->data,out->data,state->data,M,H);
    return ok(cudaGetLastError(),"fused mHC post/pre launch");
}
extern "C" int dsv4_cuda_mhc_post_pre_norm(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,Dsv4CudaActivation*state,int M,int H,Dsv4CudaActivation*out,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,float norm_eps,Dsv4CudaActivation*input){
    constexpr int N=24,parts=8;int I=M*H;Dev*c=x?ctx(x->device):nullptr;if(!c||!r||!state||!out||!fn||!scale||!base||!norm||!input||M!=4||r->device!=x->device||state->device!=x->device||out->device!=x->device||fn->device!=x->device||scale->device!=x->device||base->device!=x->device||norm->device!=x->device||input->device!=x->device||x->elements<H||r->elements<I||state->elements<M+M*M+M||out->elements<I||fn->fmt!=32||norm->fmt!=32||fn->O!=N||fn->I!=I||norm->O*norm->I<H||!ok(cudaSetDevice(x->device),"select fused normalized mHC device")||!buf((void**)&c->p1,&c->p1cap,N*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(N+1)*parts*sizeof(float)))return 0;
    mhc_post_fn_split<<<parts,256,0,c->stream>>>((float*)fn->w,x->data,r->data,state->data,out->data,c->dy,N,M,H,parts);mhc_fn_finish<<<1,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M,I,rms_eps,parts);mhc_coeff<<<1,32,0,c->stream>>>(state->data,c->p1,M,pre_eps,sink_eps,post_mult,sink_iters);mhc_input_norm<<<1,256,0,c->stream>>>(input->data,out->data,state->data,(float*)norm->w,M,H,norm_eps);return ok(cudaGetLastError(),"fused normalized mHC post/pre launch");
}
extern "C" int dsv4_cuda_mhc_post_pre_norm_batch(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,Dsv4CudaActivation*state,int tokens,int H,Dsv4CudaActivation*out,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,Dsv4CudaActivation*input){
    (void)x;(void)r;(void)state;(void)tokens;(void)H;(void)out;(void)fn;(void)scale;(void)base;(void)norm;(void)input;return 0;
}
extern "C" int dsv4_cuda_mhc_post_batch(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,const Dsv4CudaActivation*state,int tokens,int H,Dsv4CudaActivation*out){
    long long RH=(long long)tokens*4*H,XH=(long long)tokens*H;Dev*c=x?ctx(x->device):nullptr;if(!c||!r||!state||!out||tokens<1||H!=4096||r->device!=x->device||state->device!=x->device||out->device!=x->device||x->elements<XH||r->elements<RH||state->elements<(long long)tokens*20||out->elements<RH||!ok(cudaSetDevice(x->device),"select batched mHC output device"))return 0;mhc_output_batch<<<(RH+255)/256,256,0,c->stream>>>(out->data,x->data,r->data,state->data,state->data+(long long)tokens*4,H);return ok(cudaGetLastError(),"batched mHC output launch");
}
extern "C" void dsv4_cuda_kv_free(Dsv4CudaKvCache *k) {
    if (!k) return;
    cudaSetDevice(k->device);
    if (k->kv) cudaFree(k->kv);
    if (k->compressed) cudaFree(k->compressed);
    if (k->compressed_len) cudaFree(k->compressed_len);
    if (k->comp_value) cudaFree(k->comp_value);
    if (k->comp_score) cudaFree(k->comp_score);
    if (k->rope_cos) cudaFree(k->rope_cos);
    if (k->rope_sin) cudaFree(k->rope_sin);
    if (k->comp_cos) cudaFree(k->comp_cos);
    if (k->comp_sin) cudaFree(k->comp_sin);
    free(k);
}
extern "C" int dsv4_cuda_attention_window(const Dsv4CudaActivation *input, Dsv4CudaTensor *an, Dsv4CudaTensor *qa,
                                          Dsv4CudaTensor *qn, Dsv4CudaTensor *qb, Dsv4CudaTensor *kv,
                                          Dsv4CudaTensor *kvn, Dsv4CudaTensor *sink, Dsv4CudaTensor *wa,
                                          Dsv4CudaTensor *wb, Dsv4CudaTensor *cwkv, Dsv4CudaTensor *cwg,
                                          Dsv4CudaTensor *ape, Dsv4CudaTensor *cnorm, int ratio, int heads, int dim,
                                          int qk_rope, int groups, int pos, float eps, Dsv4CudaKvCache *cache,
                                          Dsv4CudaActivation *output) {
    Dev *c = input ? ctx(input->device) : nullptr;
    int H = qa ? qa->I : 0, R = qb ? qb->I : 0, Q = heads * dim, K = dim, WR = wa ? wa->O : 0, overlap = ratio == 4,
        O = (1 + overlap) * dim;
    int fused_qkv = qa && qa->O == R + K;
    if (!c || !cache || !output || !an || !qa || !qn || !qb || !kv || !kvn || !sink || !wa || !wb || pos < 0 ||
        pos >= cache->max_tokens || cache->device != input->device || cache->window < 1 || cache->dim != dim ||
        cache->rope_pairs * 2 != qk_rope || input->elements < H || output->elements < H || an->fmt != 32 ||
        qn->fmt != 16 || kvn->fmt != 32 || sink->fmt != 32 || qa->fmt != 8 || qb->fmt != 8 || kv->fmt != 8 ||
        wa->fmt != 9 || wb->fmt != 8 || an->O * an->I < H || qn->O * qn->I < R || kvn->O * kvn->I < K ||
        sink->O * sink->I < heads || qa->O != R + (fused_qkv ? K : 0) || qb->O != Q || kv->I != H || kv->O != K || wa->I * groups != Q ||
        wb->I != WR || wb->O != H || heads % groups)
        return 0;
    if (ratio && (!cwkv || !cwg || !ape || !cnorm || cwkv->fmt != 16 || cwg->fmt != 16 || ape->fmt != 32 ||
                  cnorm->fmt != 32 || cwkv->O != O || cwkv->I != H || cwg->O != O || cwg->I != H ||
                  ape->O * ape->I != ratio * O || cnorm->O * cnorm->I < dim))
        return 0;
    Dsv4CudaTensor *tensors[] = {an, qa, qn, qb, kv, kvn, sink, wa, wb, cwkv, cwg, ape, cnorm};
    for (auto *t : tensors)
        if (t && t->device != input->device) return 0;
    int P1 = R + (fused_qkv ? K : 0); if (P1 < O) P1 = O;
    if (WR > P1) P1 = WR;
    if (output->device != input->device || !ok(cudaSetDevice(input->device), "select attention device") ||
        !buf((void **)&c->dx, &c->xcap, (size_t)H * 4) || !buf((void **)&c->p1, &c->p1cap, (size_t)P1 * 4) ||
        !buf((void **)&c->p2, &c->p2cap, (size_t)(Q > O ? Q : O) * 4) ||
        !buf((void **)&c->p3, &c->p3cap, (size_t)K * 4) || !buf((void **)&c->p4, &c->p4cap, (size_t)Q * 4) ||
        !buf((void **)&c->aux1, &c->aux1cap, (size_t)O * 4) ||
        !buf((void **)&c->aux2, &c->aux2cap, (size_t)O * 4) ||
        !buf((void **)&c->aux3, &c->aux3cap, (size_t)dim * 4))
        return 0;
    rmsnorm_f32<<<1, 256, 0, c->stream>>>(c->dx, input->data, (float *)an->w, H, eps);
    int async = ratio && getenv("DSV4_CUDA_ATTN_ASYNC") && atoi(getenv("DSV4_CUDA_ATTN_ASYNC"));
    cudaStream_t comp = async ? c->aux : c->stream;
    float *comp1 = async ? c->aux1 : c->p1, *comp2 = async ? c->aux2 : c->p2,
          *comp3 = async ? c->aux3 : c->p3;
    if (async && (!ok(cudaEventRecord(c->fork, c->stream), "attention fork") ||
                  !ok(cudaStreamWaitEvent(c->aux, c->fork, 0), "attention aux wait"))) return 0;
    if (ratio) {
        mv_bf16<<<O, 256, 0, comp>>>((__nv_bfloat16 *)cwkv->w, c->dx, comp1, O, H);
        mv_bf16<<<O, 256, 0, comp>>>((__nv_bfloat16 *)cwg->w, c->dx, comp2, O, H);
        compress_store_dynamic<<<(O + 255) / 256, 256, 0, comp>>>(cache->comp_value, cache->comp_score, comp1,
                                                                  comp2, (float *)ape->w, c->decode_state,
                                                                  ratio, overlap, O);
        compress_pool_dynamic<<<(dim + 255) / 256, 256, 0, comp>>>(comp3, cache->comp_value, cache->comp_score,
                                                                    c->decode_state, ratio, overlap, dim);
        rmsnorm_dynamic<<<1, 256, 0, comp>>>(comp3, comp3, (float *)cnorm->w, dim, eps, c->decode_state, ratio);
        if (qk_rope)
            rope_heads_dynamic<<<(qk_rope / 2 + 255) / 256, 256, 0, comp>>>(
                comp3, 1, dim, qk_rope, c->decode_state, 1 - ratio, 0, ratio, cache->comp_cos, cache->comp_sin);
        /* The FlashInfer cache packer performs the model's UE8M0 FP8
         * quantization.  Quantizing and dequantizing here first would apply
         * a second, differently-scaled FP8 round before the real cache
         * store.  Keep the simulation only for the float-cache fallback. */
        if (dim > qk_rope)
            fp8_sim64_dynamic<<<(dim - qk_rope + 63) / 64, 64, 0, comp>>>(comp3, dim - qk_rope,
                                                                          c->decode_state, ratio);
        compressed_store_dynamic<<<(dim + 255) / 256, 256, 0, comp>>>(
            cache->compressed, comp3, cache->compressed_len, c->decode_state, ratio, cache->max_compressed, dim);
        int commit_blocks = overlap ? (ratio * O + 255) / 256 : 1;
        compression_commit<<<commit_blocks, 256, 0, comp>>>(
            cache->compressed_len, cache->comp_value, cache->comp_score, c->decode_state, ratio, overlap, O,
            cache->max_compressed);
        if (async && !ok(cudaEventRecord(c->join, c->aux), "attention join record")) return 0;
    }
    int dense_dg=getenv("DSV4_CUDA_DENSE_DG")&&atoi(getenv("DSV4_CUDA_DENSE_DG"));
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,qa,c->dx,c->p1)||(!fused_qkv&&!dg_attention_kv_reuse(c,kv,c->dx,c->p3)))return 0;}else
#endif
    {fp8_sim<<<(H + 127) / 128, 128, 0, c->stream>>>(c->dx, 1, H);
    run_mv<8>((uint8_t *)qa->w, qa->scale, c->dx, c->p1, R, H, 1, c->stream);}
    bf16_round<<<(R + 255) / 256, 256, 0, c->stream>>>(c->p1, R);
    rmsnorm_seq<<<1, 256, 0, c->stream>>>(c->p1, (__nv_bfloat16 *)qn->w, R, eps);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,qb,c->p1,c->p2))return 0;}else
#endif
    {fp8_sim<<<(R + 127) / 128, 128, 0, c->stream>>>(c->p1, 1, R);
    run_mv<8>((uint8_t *)qb->w, qb->scale, c->p1, c->p2, Q, R, 1, c->stream);}
    bf16_round<<<(Q + 255) / 256, 256, 0, c->stream>>>(c->p2, Q);
    int sparse=0;
    if(!sparse)head_rmsnorm<<<heads, 256, 0, c->stream>>>(c->p2, heads, dim, eps);
    if (qk_rope&&!sparse)
        rope_heads_dynamic<<<(heads * (qk_rope / 2) + 255) / 256, 256, 0, c->stream>>>(
            c->p2, heads, dim, qk_rope, c->decode_state, 0, 0, 0, cache->rope_cos, cache->rope_sin);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){}else
#endif
    run_mv<8>((uint8_t *)kv->w, kv->scale, c->dx, c->p3, K, H, 1, c->stream);
    if (!fused_qkv) bf16_round<<<(K + 255) / 256, 256, 0, c->stream>>>(c->p3, K);
    rmsnorm_f32<<<1, 256, 0, c->stream>>>(c->p3, fused_qkv ? c->p1 + R : c->p3, (float *)kvn->w, K, eps);
    if (qk_rope&&!sparse)
        rope_heads_dynamic<<<(qk_rope / 2 + 255) / 256, 256, 0, c->stream>>>(
            c->p3, 1, dim, qk_rope, c->decode_state, 0, 0, 0, cache->rope_cos, cache->rope_sin);
    if(!sparse){if (K > qk_rope) fp8_sim64<<<(K - qk_rope + 63) / 64, 64, 0, c->stream>>>(c->p3, K - qk_rope);cache_store_dynamic<<<(K + 255) / 256, 256, 0, c->stream>>>(cache->kv, c->p3, c->decode_state, cache->window, K);}
    if (async && !ok(cudaStreamWaitEvent(c->stream, c->join, 0), "attention join wait")) return 0;
    attention_window_dynamic<<<heads, 256, (size_t)(cache->window + cache->max_compressed) * 4, c->stream>>>(
        c->p4, c->p2, cache->kv, cache->compressed, (float *)sink->w, heads, dim, cache->window, c->decode_state,
        cache->compressed_len);
    if (qk_rope)
        rope_heads_dynamic<<<(heads * (qk_rope / 2) + 255) / 256, 256, 0, c->stream>>>(
            c->p4, heads, dim, qk_rope, c->decode_state, 0, 1, 0, cache->rope_cos, cache->rope_sin);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_wo_a(c,wa,c->p4,c->p1))return 0;}else
#endif
    run_mv<9>((uint8_t *)wa->w, wa->scale, c->p4, c->p1, WR, wa->I, groups, c->stream);
    bf16_round<<<(WR + 255) / 256, 256, 0, c->stream>>>(c->p1, WR);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,wb,c->p1,output->data))return 0;}else
#endif
    {fp8_sim<<<(WR + 127) / 128, 128, 0, c->stream>>>(c->p1, 1, WR);
    run_mv<8>((uint8_t *)wb->w, wb->scale, c->p1, output->data, H, WR, 1, c->stream);}
    bf16_round<<<(H + 255) / 256, 256, 0, c->stream>>>(output->data, H);
    return ok(cudaGetLastError(), "attention window launch");
}

extern "C" int dsv4_cuda_attention_sparse_batch(const Dsv4CudaActivation*input,Dsv4CudaTensor*an,Dsv4CudaTensor*qkv,Dsv4CudaTensor*qn,Dsv4CudaTensor*qb,Dsv4CudaTensor*kvn,Dsv4CudaTensor*sink,int heads,int dim,int start,int tokens,float eps,Dsv4CudaKvCache*cache,Dsv4CudaActivation*context){
    (void)input;(void)an;(void)qkv;(void)qn;(void)qb;(void)kvn;(void)sink;(void)heads;(void)dim;(void)start;(void)tokens;(void)eps;(void)cache;(void)context;return 0;
}

extern "C" int dsv4_cuda_attention_output_batch(const Dsv4CudaActivation*context,Dsv4CudaTensor*wa,Dsv4CudaTensor*wb,int groups,int tokens,Dsv4CudaActivation*output){
#ifdef COLI_DSV4_DEEPGEMM
    Dev*c=context?ctx(context->device):nullptr;int K=wa?wa->I:0,N=wa&&groups?wa->O/groups:0,WR=wa?wa->O:0,H=wb?wb->O:0;
    if(!c||!wa||!wb||!output||groups<1||tokens<1||wa->fmt!=9||wa->O%groups||wb->fmt!=8||wb->I!=WR||
       wa->device!=context->device||wb->device!=context->device||output->device!=context->device||
       context->elements<(long long)tokens*groups*K||output->elements<(long long)tokens*H||N!=1024||K!=4096)return 0;
    size_t in=(size_t)tokens*K*sizeof(float),part=(size_t)tokens*N*sizeof(float),joined=(size_t)tokens*WR*sizeof(float);
    if(!ok(cudaSetDevice(context->device),"select batched attention output device")||!buf((void**)&c->aux1,&c->aux1cap,in)||
       !buf((void**)&c->aux2,&c->aux2cap,part)||!buf((void**)&c->p1,&c->p1cap,joined))return 0;
    for(int gidx=0;gidx<groups;gidx++){gather_group_rows<<<((long long)tokens*K+255)/256,256,0,c->stream>>>(c->aux1,context->data,tokens,groups,K,gidx);
        Dsv4CudaTensor view=*wa;view.O=N;view.w=(uint8_t*)wa->w+(long long)gidx*N*K;view.dg_scale=wa->dg_scale+(long long)gidx*(K/512)*N;
        if(!dg_dense_batch_dispatch(c,&view,c->aux1,c->aux2,tokens))return 0;
        scatter_group_rows<<<((long long)tokens*N+255)/256,256,0,c->stream>>>(c->p1,c->aux2,tokens,groups,N,gidx);}
    return dg_dense_batch_dispatch(c,wb,c->p1,output->data,tokens);
#else
    /* GENERIC grouped wo_a (fmt 9 view per group) + wo_b via the portable
     * batch matmul; same gather/scatter structure as the DeepGEMM path. */
    Dev*c=context?ctx(context->device):nullptr;int K=wa?wa->I:0,N=wa&&groups?wa->O/groups:0,WR=wa?wa->O:0,H=wb?wb->O:0;
    if(!c||!wa||!wb||!output||groups<1||tokens<1||wa->fmt!=9||wa->O%groups||N%128||K%128||wb->fmt!=8||wb->I!=WR||
       wa->device!=context->device||wb->device!=context->device||output->device!=context->device||
       context->elements<(long long)tokens*groups*K||output->elements<(long long)tokens*H)return 0;
    size_t in=(size_t)tokens*K*sizeof(float),part=(size_t)tokens*N*sizeof(float),joined=(size_t)tokens*WR*sizeof(float);
    if(!ok(cudaSetDevice(context->device),"select generic attention output device")||!buf((void**)&c->aux1,&c->aux1cap,in)||
       !buf((void**)&c->aux2,&c->aux2cap,part)||!buf((void**)&c->p1,&c->p1cap,joined))return 0;
    for(int gidx=0;gidx<groups;gidx++){gather_group_rows<<<((long long)tokens*K+255)/256,256,0,c->stream>>>(c->aux1,context->data,tokens,groups,K,gidx);
        Dsv4CudaTensor view=*wa;view.O=N;view.w=(uint8_t*)wa->w+(long long)gidx*N*K;view.scale=wa->scale+(long long)gidx*(N/128)*(K/128);
        run_mm_batch(&view,c->aux1,c->aux2,tokens,c->stream);
        scatter_group_rows<<<((long long)tokens*N+255)/256,256,0,c->stream>>>(c->p1,c->aux2,tokens,groups,N,gidx);}
    run_mm_batch(wb,c->p1,output->data,tokens,c->stream);
    return ok(cudaGetLastError(),"generic attention output launch");
#endif
}

extern "C" int dsv4_cuda_attention_window_tp2(const Dsv4CudaActivation *input,Dsv4CudaActivation *peer_input,
        const Dsv4CudaAttentionWeights *a,const Dsv4CudaAttentionWeights *b,int ratio,int heads,int dim,int qk_rope,
        int groups,int pos,float eps,Dsv4CudaKvCache *cache,Dsv4CudaKvCache *peer_cache,
        Dsv4CudaActivation *output,Dsv4CudaActivation *peer_output){
#ifndef COLI_DSV4_NCCL
    (void)input;(void)peer_input;(void)a;(void)b;(void)ratio;(void)heads;(void)dim;(void)qk_rope;(void)groups;
    (void)pos;(void)eps;(void)cache;(void)peer_cache;(void)output;(void)peer_output;return 0;
#else
    Dev *p=input?ctx(input->device):nullptr,*q=peer_input?ctx(peer_input->device):nullptr;int H=a&&a->q_a?a->q_a->I:0;
    if(!p||!q||!a||!b||!cache||!peer_cache||!output||!peer_output||!p->ep_comm||!q->ep_comm||heads%2||groups%2||
       peer_input->elements<H||output->elements<H||peer_output->elements<H)return 0;
    int replicated=getenv("DSV4_CUDA_REPLICATED_TP2")&&atoi(getenv("DSV4_CUDA_REPLICATED_TP2"));
#ifdef COLI_DSV4_DEEPGEMM
    const char *qkv_env=getenv("DSV4_CUDA_QKV_FUSED");int fuse_qkv=!qkv_env||atoi(qkv_env);
#else
    int fuse_qkv=0;
#endif
    if(!replicated&&(!ok(cudaSetDevice(p->id),"select TP2 attention primary")||!nok(ncclGroupStart(),"TP2 attention broadcast start")||
       !nok(ncclBroadcast(input->data,input->data,H,ncclFloat,0,p->ep_comm,p->stream),"TP2 attention input root")||
       !ok(cudaSetDevice(q->id),"select TP2 attention peer")||
       !nok(ncclBroadcast(peer_input->data,peer_input->data,H,ncclFloat,0,q->ep_comm,q->stream),"TP2 attention input peer")||
       !nok(ncclGroupEnd(),"TP2 attention broadcast end")))return 0;
#define ATTN_LOCAL(x,w,k,y) dsv4_cuda_attention_window((x),(w)->attn_norm,fuse_qkv?(w)->qkv:(w)->q_a,(w)->q_norm,(w)->q_b,(w)->wkv,\
        (w)->kv_norm,(w)->sink,(w)->wo_a,(w)->wo_b,(w)->compress_wkv,(w)->compress_wgate,(w)->compress_ape,\
        (w)->compress_norm,ratio,heads/2,dim,qk_rope,groups/2,pos,eps,(k),(y))
    if(!ATTN_LOCAL(input,a,cache,output)||!ATTN_LOCAL(peer_input,b,peer_cache,peer_output))return 0;
#undef ATTN_LOCAL
    if(!ok(cudaSetDevice(p->id),"select TP2 attention reduce primary")||!nok(ncclGroupStart(),"TP2 attention reduce start")||
       !nok(ncclAllReduce(output->data,output->data,H,ncclFloat,ncclSum,p->ep_comm,p->stream),"TP2 attention primary reduce")||
       !ok(cudaSetDevice(q->id),"select TP2 attention reduce peer")||
       !nok(ncclAllReduce(peer_output->data,peer_output->data,H,ncclFloat,ncclSum,q->ep_comm,q->stream),"TP2 attention peer reduce")||
       !nok(ncclGroupEnd(),"TP2 attention reduce end"))return 0;
    return ok(cudaSetDevice(p->id),"restore TP2 attention primary");
#endif
}
extern "C" int dsv4_cuda_route(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,const int*fixed,float scale,int ids[6],float weights[6]){
    Dev*c=input?ctx(input->device):nullptr;if(!c||!gate||!ids||!weights||gate->fmt!=32||gate->O!=256||gate->I>input->elements||gate->device!=input->device||(bias&&(bias->fmt!=32||bias->O*bias->I<256||bias->device!=input->device))||!ok(cudaSetDevice(input->device),"select router device")||!buf((void**)&c->p1,&c->p1cap,256*4)||!buf((void**)&c->expert_weights,&c->expert_weightscap,6*4)||!buf((void**)&c->expert_ids,&c->expert_idcap,6*sizeof(int)))return 0;
    mv_f32<<<256,256,0,c->stream>>>((float*)gate->w,input->data,c->p1,256,gate->I);if(!ok(cudaMemsetAsync(c->expert_ids,0,6*sizeof(int),c->stream),"router ids clear")||(fixed&&!ok(cudaMemcpyAsync(c->expert_ids,fixed,6*sizeof(int),cudaMemcpyHostToDevice,c->stream),"router fixed ids upload")))return 0;route_top6_serial<<<1,1,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->p1,fixed!=nullptr,scale);
    return ok(cudaGetLastError(),"router launch")&&ok(cudaMemcpyAsync(ids,c->expert_ids,6*sizeof(int),cudaMemcpyDeviceToHost,c->stream),"router ids download")&&ok(cudaMemcpyAsync(weights,c->expert_weights,6*sizeof(float),cudaMemcpyDeviceToHost,c->stream),"router weights download")&&ok(cudaStreamSynchronize(c->stream),"router sync");}
extern "C" int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t,float *y,const float*x,int groups){Dev*c=t?ctx(t->device):nullptr;if(!c||groups<1||t->O%groups||!ok(cudaSetDevice(t->device),"select matvec device"))return 0;size_t xb=(size_t)t->I*groups*4,yb=(size_t)t->O*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"activation upload"))return 0;
    if(t->fmt==8)run_mv<8>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==9)run_mv<9>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==4)run_mv<4>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==32)mv_f32<<<t->O,256,0,c->stream>>>((float*)t->w,c->dx,c->dy,t->O,t->I);
    else mv_bf16<<<t->O,256,0,c->stream>>>((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I);
    return ok(cudaGetLastError(),"matvec launch")&&ok(cudaMemcpyAsync(y,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"result download")&&ok(cudaStreamSynchronize(c->stream),"matvec sync");}
extern "C" int dsv4_cuda_matvec(Dsv4CudaTensor*t,float*y,const float*x){return dsv4_cuda_matvec_grouped(t,y,x,1);}
extern "C" int dsv4_cuda_matmul_batch(Dsv4CudaTensor*t,const Dsv4CudaActivation*input,int tokens,Dsv4CudaActivation*output){
    Dev*c=t?ctx(t->device):nullptr;if(!c||!input||!output||tokens<1||input->device!=t->device||output->device!=t->device||
       input->elements<(long long)tokens*t->I||output->elements<(long long)tokens*t->O||!ok(cudaSetDevice(t->device),"select batched dense device"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    return dg_dense_batch_dispatch(c,t,input->data,output->data,tokens);
#else
    if(t->fmt!=8&&t->fmt!=9&&t->fmt!=4)return 0;
    run_mm_batch(t,input->data,output->data,tokens,c->stream);
    return ok(cudaGetLastError(),"batched dense launch");
#endif
}
extern "C" int dsv4_cuda_head_argmax(Dsv4CudaTensor*t,const float*x,int*id,float*value){
#ifdef COLI_DSV4_DEEPGEMM
    constexpr int O=129280,I=4096;Dev*c=t?ctx(t->device):nullptr;if(!c||!x||!id||!value||t->fmt!=16||t->O!=O||t->I!=I||!ok(cudaSetDevice(t->device),"select output head device")||
       !buf((void**)&c->dx,&c->xcap,I*sizeof(float))||!buf((void**)&c->dga1,&c->dga1cap,I*sizeof(__nv_bfloat16))||!buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||
       !buf((void**)&c->expert_ids,&c->expert_idcap,sizeof(int))||!buf((void**)&c->expert_weights,&c->expert_weightscap,sizeof(float))||
       !ok(cudaMemcpyAsync(c->dx,x,I*sizeof(float),cudaMemcpyHostToDevice,c->stream),"output head input upload"))return 0;
    dg_float_to_bf16<<<(I+255)/256,256,0,c->stream>>>((__nv_bfloat16*)c->dga1,c->dx,I);CUtensorMap a,b,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dga1,I,1,(unsigned long long)I*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,t->w,I,O,(unsigned long long)I*2,64,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_bf16_gemm_impl<0,O,I,1,64,128,64,128,128,128,3,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,true,2>:deep_gemm::sm120_bf16_gemm_impl<0,O,I,1,64,128,64,128,128,128,3,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,true,2>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,90368),"output head DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,90368,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(DgOutput*)c->dga1,(DgOutput*)t->w,nullptr,nullptr,1,O,I,a,b,d);
    dg_argmax_bf16<<<1,256,0,c->stream>>>(c->dgmm1,O,c->expert_ids,c->expert_weights);
    return ok(cudaGetLastError(),"output head DeepGEMM launch")&&ok(cudaMemcpyAsync(id,c->expert_ids,sizeof(*id),cudaMemcpyDeviceToHost,c->stream),"output id download")&&
           ok(cudaMemcpyAsync(value,c->expert_weights,sizeof(*value),cudaMemcpyDeviceToHost,c->stream),"output logit download")&&ok(cudaStreamSynchronize(c->stream),"output head sync");
#else
    (void)t;(void)x;(void)id;(void)value;return 0;
#endif
}
extern "C" int dsv4_cuda_final_argmax(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,Dsv4CudaTensor*head,int M,int H,float eps,float pre_eps,int*id,float*value){
#ifdef COLI_DSV4_DEEPGEMM
    constexpr int O=129280,I=4096;Dev*c=r?ctx(r->device):nullptr;int MH=M*H;if(!c||!fn||!scale||!base||!norm||!head||!id||!value||M!=4||H!=I||r->elements<MH||fn->fmt!=32||fn->O!=M||fn->I!=MH||scale->fmt!=32||base->fmt!=32||norm->fmt!=32||head->fmt!=16||head->O!=O||head->I!=I||
       fn->device!=r->device||scale->device!=r->device||base->device!=r->device||norm->device!=r->device||head->device!=r->device||!ok(cudaSetDevice(r->device),"select final head device")||
       !buf((void**)&c->dx,&c->xcap,I*sizeof(float))||!buf((void**)&c->p1,&c->p1cap,M*sizeof(float))||!buf((void**)&c->dga1,&c->dga1cap,I*sizeof(__nv_bfloat16))||!buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||
       !buf((void**)&c->expert_ids,&c->expert_idcap,sizeof(int))||!buf((void**)&c->expert_weights,&c->expert_weightscap,sizeof(float)))return 0;
    final_mhc_coeff<<<1,256,0,c->stream>>>(c->p1,r->data,(float*)fn->w,(float*)scale->w,(float*)base->w,M,H,eps,pre_eps);final_mhc_mix<<<(H+255)/256,256,0,c->stream>>>(c->dx,r->data,c->p1,M,H);rmsnorm_f32<<<1,256,0,c->stream>>>(c->dx,c->dx,(float*)norm->w,H,eps);dg_float_to_bf16<<<(I+255)/256,256,0,c->stream>>>((__nv_bfloat16*)c->dga1,c->dx,I);CUtensorMap a,b,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dga1,I,1,(unsigned long long)I*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,head->w,I,O,(unsigned long long)I*2,64,128,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_bf16_gemm_impl<0,O,I,1,64,128,64,128,128,128,3,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,true,2>:deep_gemm::sm120_bf16_gemm_impl<0,O,I,1,64,128,64,128,128,128,3,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,true,2>;if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,90368),"final head DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,90368,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(DgOutput*)c->dga1,(DgOutput*)head->w,nullptr,nullptr,1,O,I,a,b,d);dg_argmax_bf16<<<1,256,0,c->stream>>>(c->dgmm1,O,c->expert_ids,c->expert_weights);
    return ok(cudaGetLastError(),"final head launch")&&ok(cudaMemcpyAsync(id,c->expert_ids,sizeof(*id),cudaMemcpyDeviceToHost,c->stream),"final id download")&&ok(cudaMemcpyAsync(value,c->expert_weights,sizeof(*value),cudaMemcpyDeviceToHost,c->stream),"final logit download")&&ok(cudaStreamSynchronize(c->stream),"final head sync");
#else
    (void)r;(void)fn;(void)scale;(void)base;(void)norm;(void)head;(void)M;(void)H;(void)eps;(void)pre_eps;(void)id;(void)value;return 0;
#endif
}
#if COLI_DSV4_TC
static TcPlan *tc_plan(Dev*c,int O,int I){int pi=O>I;TcPlan*p=&c->plans[pi];if(p->ok)return p;if(p->op)plan_free(p);p->O=O;p->I=I;
    cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;cublasLtMatmulMatrixScale_t mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if(!bok(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F),"TC operation")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)),"TC trans A")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)),"TC trans B")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)),"TC A scale mode")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)),"TC B scale mode")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&c->tcs,sizeof(c->tcs)),"TC initial A scale")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&c->tcxs,sizeof(c->tcxs)),"TC initial B scale")||
       !bok(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_8F_E4M3,I,O,I),"TC A layout")||
       !bok(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_8F_E4M3,I,1,I),"TC B layout")||
       !bok(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_32F,O,1,O),"TC C layout")||
       !bok(cublasLtMatrixLayoutCreate(&p->d,CUDA_R_32F,O,1,O),"TC D layout"))return NULL;
    cublasLtMatmulPreference_t pref=NULL;size_t wsz=32<<20;int found=0;if(!bok(cublasLtMatmulPreferenceCreate(&pref),"TC preference")||
       !bok(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz)),"TC workspace preference")||
       !bok(cublasLtMatmulAlgoGetHeuristic(c->lt,p->op,p->a,p->b,p->c,p->d,pref,1,&p->algo,&found),"TC heuristic")){if(pref)cublasLtMatmulPreferenceDestroy(pref);return NULL;}
    cublasLtMatmulPreferenceDestroy(pref);if(!found){fprintf(stderr,"[DSV4 CUDA] no MXFP8 Tensor Core plan O=%d I=%d\n",O,I);return NULL;}p->ok=1;return p;}
static int tc_prepare_x(Dev*c,const float*x,int I){size_t sb=(size_t)((I+127)/128)*4*128;if(!buf((void**)&c->tcx,&c->tcxcap,I)||!buf((void**)&c->tcxs,&c->tcxscap,sb))return 0;
    if(!ok(cudaMemsetAsync(c->tcxs,0,sb,c->stream),"TC activation scale clear"))return 0;tc_quant<<<(I+127)/128,128,0,c->stream>>>(x,c->tcx,c->tcxs,I);return ok(cudaGetLastError(),"TC activation quantize");}
static int tc_fp4_matvec(Dev*c,Dsv4CudaTensor*t,float*y){int O=t->O,I=t->I;size_t wb=(size_t)O*I,sb=(size_t)((I+127)/128)*4*((O+127)/128)*128;
    if(!buf((void**)&c->tcw,&c->tcwcap,wb)||!buf((void**)&c->tcs,&c->tcscap,sb))return 0;TcPlan*p=tc_plan(c,O,I);if(!p)return 0;
    tc_expand<<<256,256,0,c->stream>>>((uint8_t*)t->w,t->scale,c->tcw,c->tcs,O,I);if(!ok(cudaGetLastError(),"TC weight expansion"))return 0;
    void *as=c->tcs,*bs=c->tcxs;if(!bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&as,sizeof(as)),"TC A scale")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&bs,sizeof(bs)),"TC B scale"))return 0;float alpha=1,beta=0;
    return bok(cublasLtMatmul(c->lt,p->op,&alpha,c->tcw,p->a,c->tcx,p->b,&beta,y,p->c,y,p->d,&p->algo.algo,c->tc_workspace,32<<20,c->stream),"TC matvec");}
#endif /* COLI_DSV4_TC */
extern "C" int dsv4_cuda_expert_group(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,Dsv4CudaTensor *const *down,
        const float *weights,int count,float limit,float *y,const float *x){
    if(count<1||count>16||!gate||!up||!down)return 0;Dsv4CudaTensor *t=gate[0];Dev*c=t?ctx(t->device):nullptr;
    if(!c||t->fmt!=4||!ok(cudaSetDevice(t->device),"select expert-group device"))return 0;int H=t->I,I=t->O;
    for(int e=0;e<count;e++)if(!gate[e]||!up[e]||!down[e]||gate[e]->device!=t->device||up[e]->device!=t->device||down[e]->device!=t->device||gate[e]->fmt!=4||up[e]->fmt!=4||down[e]->fmt!=4||gate[e]->I!=H||gate[e]->O!=I||up[e]->I!=H||up[e]->O!=I||down[e]->I!=I||down[e]->O!=H)return 0;
    size_t xb=(size_t)H*4,ib=(size_t)count*I*4,hb=(size_t)count*H*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,(size_t)H*4)||
       !buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"expert-group activation upload"))return 0;
#if COLI_DSV4_TC
    static int tc=-1;if(tc<0)tc=getenv("DSV4_CUDA_TC")?atoi(getenv("DSV4_CUDA_TC")):0;
    if(tc){for(int e=0;e<count;e++){if(!tc_prepare_x(c,c->dx,H)||!tc_fp4_matvec(c,gate[e],c->p1+(long long)e*I)||!tc_fp4_matvec(c,up[e],c->p2+(long long)e*I))return 0;
            expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);
            if(!tc_prepare_x(c,c->p3+(long long)e*I,I)||!tc_fp4_matvec(c,down[e],c->p4+(long long)e*H))return 0;}
        expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,count,H);
        return ok(cudaGetLastError(),"TC expert-group launch")&&ok(cudaMemcpyAsync(y,c->dy,(size_t)H*4,cudaMemcpyDeviceToHost,c->stream),"TC expert-group result download")&&ok(cudaStreamSynchronize(c->stream),"TC expert-group sync");}
#endif /* COLI_DSV4_TC */
    static int batched=-1;if(batched<0)batched=getenv("DSV4_CUDA_BATCHED")?atoi(getenv("DSV4_CUDA_BATCHED")):1;
    if(batched){MvDesc hd[32];for(int e=0;e<count;e++){
            hd[2*e]={(uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I};
            hd[2*e+1]={(uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I};}
        if(!buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)2*count*sizeof(*hd))||
           !buf((void**)&c->expert_weights,&c->expert_weightscap,(size_t)count*sizeof(float))||
           !ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)2*count*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"expert descriptors upload")||
           !ok(cudaMemcpyAsync(c->expert_weights,weights,(size_t)count*sizeof(float),cudaMemcpyHostToDevice,c->stream),"expert weights upload"))return 0;
        mv_fp4_grouped<4><<<dim3((I+3)/4,2*count),256,0,c->stream>>>(c->mvdesc,2*count,I,H);
        expert_act_grouped<<<(count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,count,I);
        fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
        for(int e=0;e<count;e++)hd[e]={(uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H};
        if(!ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)count*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"down descriptors upload"))return 0;
        mv_fp4_grouped<4><<<dim3((H+3)/4,count),256,0,c->stream>>>(c->mvdesc,count,H,I);
    }else for(int e=0;e<count;e++){
        run_mv<4>((uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I,I,H,1,c->stream);
        run_mv<4>((uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I,I,H,1,c->stream);
        expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);
        fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3+(long long)e*I,1,I);
        run_mv<4>((uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H,H,I,1,c->stream);
    }
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,count,H);
    return ok(cudaGetLastError(),"expert-group launch")&&ok(cudaMemcpyAsync(y,c->dy,(size_t)H*4,cudaMemcpyDeviceToHost,c->stream),"expert-group result download")&&ok(cudaStreamSynchronize(c->stream),"expert-group sync");
}
extern "C" int dsv4_cuda_expert_fp8(Dsv4CudaTensor *gate,Dsv4CudaTensor *up,Dsv4CudaTensor *down,float limit,float *y,const float *x){
    Dev*c=gate?ctx(gate->device):nullptr;if(!c||!up||!down||gate->fmt!=8||up->fmt!=8||down->fmt!=8||
       up->device!=gate->device||down->device!=gate->device||!ok(cudaSetDevice(gate->device),"select shared-expert device"))return 0;
    int H=gate->I,I=gate->O;if(up->I!=H||up->O!=I||down->I!=I||down->O!=H)return 0;
    size_t hb=(size_t)H*4,ib=(size_t)I*4;if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,hb)||
       !buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,hb,cudaMemcpyHostToDevice,c->stream),"shared-expert activation upload"))return 0;
    run_mv<8>((uint8_t*)gate->w,gate->scale,c->dx,c->p1,I,H,1,c->stream);
    run_mv<8>((uint8_t*)up->w,up->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);
    fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);
    run_mv<8>((uint8_t*)down->w,down->scale,c->p3,c->p4,H,I,1,c->stream);
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,1,H);
    return ok(cudaGetLastError(),"shared-expert launch")&&ok(cudaMemcpyAsync(y,c->dy,hb,cudaMemcpyDeviceToHost,c->stream),"shared-expert result download")&&ok(cudaStreamSynchronize(c->stream),"shared-expert sync");
}
extern "C" int dsv4_cuda_moe(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,Dsv4CudaTensor *const *down,
        const float *weights,int count,Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd,float limit,float *y,const float *x){
    if(count<1||count>16||!gate||!up||!down||!sg||!su||!sd)return 0;Dsv4CudaTensor*t=gate[0];Dev*c=t?ctx(t->device):nullptr;
    if(!c||t->fmt!=4||sg->device!=t->device||su->device!=t->device||sd->device!=t->device||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||!ok(cudaSetDevice(t->device),"select MoE device"))return 0;
    int H=t->I,I=t->O;if(sg->I!=H||sg->O!=I||su->I!=H||su->O!=I||sd->I!=I||sd->O!=H)return 0;
    for(int e=0;e<count;e++)if(!gate[e]||!up[e]||!down[e]||gate[e]->device!=t->device||up[e]->device!=t->device||down[e]->device!=t->device||gate[e]->fmt!=4||up[e]->fmt!=4||down[e]->fmt!=4)return 0;
    size_t hb=(size_t)H*4,ib=(size_t)count*I*4,parts=(size_t)count*H*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,hb)||!buf((void**)&c->p1,&c->p1cap,ib)||
       !buf((void**)&c->p2,&c->p2cap,ib)||!buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,parts)||
       !ok(cudaMemcpyAsync(c->dx,x,hb,cudaMemcpyHostToDevice,c->stream),"MoE activation upload"))return 0;
    static int batched=-1;if(batched<0)batched=getenv("DSV4_CUDA_BATCHED")?atoi(getenv("DSV4_CUDA_BATCHED")):1;
    if(batched){MvDesc hd[32];for(int e=0;e<count;e++){hd[2*e]={(uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I};hd[2*e+1]={(uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I};}
        if(!buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)2*count*sizeof(*hd))||!buf((void**)&c->expert_weights,&c->expert_weightscap,(size_t)count*sizeof(float))||
           !ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)2*count*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"MoE descriptors upload")||!ok(cudaMemcpyAsync(c->expert_weights,weights,(size_t)count*sizeof(float),cudaMemcpyHostToDevice,c->stream),"MoE weights upload"))return 0;
        mv_fp4_grouped<4><<<dim3((I+3)/4,2*count),256,0,c->stream>>>(c->mvdesc,2*count,I,H);expert_act_grouped<<<(count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,count,I);fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
        for(int e=0;e<count;e++)hd[e]={(uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H};if(!ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)count*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"MoE down descriptors upload"))return 0;
        mv_fp4_grouped<4><<<dim3((H+3)/4,count),256,0,c->stream>>>(c->mvdesc,count,H,I);
    }else for(int e=0;e<count;e++){run_mv<4>((uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I,I,H,1,c->stream);run_mv<4>((uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I,I,H,1,c->stream);expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3+(long long)e*I,1,I);run_mv<4>((uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H,H,I,1,c->stream);}
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,count,H);
    run_mv<8>((uint8_t*)sg->w,sg->scale,c->dx,c->p1,I,H,1,c->stream);run_mv<8>((uint8_t*)su->w,su->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);run_mv<8>((uint8_t*)sd->w,sd->scale,c->p3,c->p4,H,I,1,c->stream);
    moe_combine<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,H);
    return ok(cudaGetLastError(),"MoE launch")&&ok(cudaMemcpyAsync(y,c->dy,hb,cudaMemcpyDeviceToHost,c->stream),"MoE result download")&&ok(cudaStreamSynchronize(c->stream),"MoE sync");
}
extern "C" int dsv4_cuda_rmsnorm(Dsv4CudaActivation*x,Dsv4CudaTensor*w,float eps,int elements){
    Dev*c=x?ctx(x->device):nullptr;
    if(!c||!w||w->device!=x->device||w->fmt!=32||elements<1||x->elements<elements||
       (long long)w->O*w->I<elements||!ok(cudaSetDevice(x->device),"select RMSNorm device"))return 0;
    rmsnorm_f32<<<1,256,0,c->stream>>>(x->data,x->data,(float*)w->w,elements,eps);
    return ok(cudaGetLastError(),"RMSNorm launch");
}
extern "C" int dsv4_cuda_moe_activation(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,Dsv4CudaTensor *const *down,
        const float *weights,int count,Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd,float limit,
        const Dsv4CudaActivation*input,Dsv4CudaActivation*output){
    if(count<1||count>16||!gate||!up||!down||!weights||!sg||!su||!sd||!input||!output)return 0;
    Dsv4CudaTensor*t=gate[0];Dev*c=t?ctx(t->device):nullptr;
    if(!c||input->device!=t->device||output->device!=t->device||t->fmt!=4||sg->device!=t->device||
       su->device!=t->device||sd->device!=t->device||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||
       !ok(cudaSetDevice(t->device),"select activation MoE device"))return 0;
    int H=t->I,I=t->O;
    if(input->elements<H||output->elements<H||sg->I!=H||sg->O!=I||su->I!=H||su->O!=I||sd->I!=I||sd->O!=H)return 0;
    for(int e=0;e<count;e++)if(!gate[e]||!up[e]||!down[e]||gate[e]->device!=t->device||up[e]->device!=t->device||
       down[e]->device!=t->device||gate[e]->fmt!=4||up[e]->fmt!=4||down[e]->fmt!=4)return 0;
    size_t hb=(size_t)H*4,ib=(size_t)count*I*4,parts=(size_t)count*H*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,parts))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,input->data,hb,cudaMemcpyDeviceToDevice,c->stream),"activation MoE input copy"))return 0;
    fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);
    static int batched=-1;if(batched<0)batched=getenv("DSV4_CUDA_BATCHED")?atoi(getenv("DSV4_CUDA_BATCHED")):1;
    if(batched){
        MvDesc hd[32];for(int e=0;e<count;e++){hd[2*e]={(uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I};hd[2*e+1]={(uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I};}
        if(!buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)2*count*sizeof(*hd))||
           !buf((void**)&c->expert_weights,&c->expert_weightscap,(size_t)count*sizeof(float))||
           !ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)2*count*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"activation MoE descriptors upload")||
           !ok(cudaMemcpyAsync(c->expert_weights,weights,(size_t)count*sizeof(float),cudaMemcpyHostToDevice,c->stream),"activation MoE weights upload"))return 0;
        mv_fp4_grouped<4><<<dim3((I+3)/4,2*count),256,0,c->stream>>>(c->mvdesc,2*count,I,H);
        expert_act_grouped<<<(count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,count,I);
        fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
        for(int e=0;e<count;e++)hd[e]={(uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H};
        if(!ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)count*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"activation MoE down descriptors upload"))return 0;
        mv_fp4_grouped<4><<<dim3((H+3)/4,count),256,0,c->stream>>>(c->mvdesc,count,H,I);
    }else for(int e=0;e<count;e++){
        run_mv<4>((uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I,I,H,1,c->stream);
        run_mv<4>((uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I,I,H,1,c->stream);
        expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);
        fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3+(long long)e*I,1,I);
        run_mv<4>((uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H,H,I,1,c->stream);
    }
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,count,H);
    run_mv<8>((uint8_t*)sg->w,sg->scale,c->dx,c->p1,I,H,1,c->stream);
    run_mv<8>((uint8_t*)su->w,su->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);
    fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);
    run_mv<8>((uint8_t*)sd->w,sd->scale,c->p3,c->p4,H,I,1,c->stream);
    moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
    return ok(cudaGetLastError(),"activation MoE launch");
}
extern "C" Dsv4CudaExpertSet *dsv4_cuda_expert_set_create(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,
        Dsv4CudaTensor *const *down,int count,Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd){
    if(!gate||!up||!down||count<1||!sg||!su||!sd||!gate[0])return nullptr;int device=gate[0]->device,H=gate[0]->I,I=gate[0]->O;
    if(sg->device!=device||su->device!=device||sd->device!=device||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||
       sg->I!=H||sg->O!=I||su->I!=H||su->O!=I||sd->I!=I||sd->O!=H||!ok(cudaSetDevice(device),"select expert table device"))return nullptr;
    ExpertPtr *host=(ExpertPtr*)malloc((size_t)3*count*sizeof(*host));if(!host)return nullptr;
    for(int e=0;e<count;e++){
        Dsv4CudaTensor*t[]={gate[e],up[e],down[e]};if(!t[0]||!t[1]||!t[2]||t[0]->device!=device||t[1]->device!=device||t[2]->device!=device||
           t[0]->fmt!=4||t[1]->fmt!=4||t[2]->fmt!=4||t[0]->I!=H||t[0]->O!=I||t[1]->I!=H||t[1]->O!=I||t[2]->I!=I||t[2]->O!=H){free(host);return nullptr;}
        host[e]={(uint8_t*)t[0]->w,t[0]->scale};host[count+e]={(uint8_t*)t[1]->w,t[1]->scale};host[2*count+e]={(uint8_t*)t[2]->w,t[2]->scale};
    }
    Dsv4CudaExpertSet *set=(Dsv4CudaExpertSet*)calloc(1,sizeof(*set));if(!set){free(host);return nullptr;}
    set->count=count;set->device=device;set->H=H;set->I=I;set->sg=sg;set->su=su;set->sd=sd;
    size_t bytes=(size_t)3*count*sizeof(*host);if(!ok(cudaMalloc(&set->table,bytes),"expert table allocation")||
       !ok(cudaMemcpy(set->table,host,bytes,cudaMemcpyHostToDevice),"expert table upload")){free(host);dsv4_cuda_expert_set_free(set);return nullptr;}
    free(host);return set;
}
static Dsv4CudaTensor *expert_view(uint8_t*w,uint8_t*scale,int O,int I,int device){
    Dsv4CudaTensor*t=(Dsv4CudaTensor*)calloc(1,sizeof(*t));if(!t)return nullptr;
    t->w=w;t->scale=scale;t->O=O;t->I=I;t->fmt=4;t->device=device;t->bytes=(long long)O*(I/2)+(long long)O*(I/32);return t;
}
extern "C" Dsv4CudaExpertSet *dsv4_cuda_expert_bank_create(int count,int H,int I,int device,
        Dsv4CudaTensor*sg,Dsv4CudaTensor*su,Dsv4CudaTensor*sd){
    int shared=sg||su||sd,J=sg?sg->O:0;
    if(count<1||H<1||I<1||(shared&&(!sg||!su||!sd))||(shared&&(sg->device!=device||su->device!=device||sd->device!=device||
       sg->fmt!=8||su->fmt!=8||sd->fmt!=8||sg->I!=H||su->I!=H||su->O!=J||sd->I!=J||sd->O!=H))||
       !ok(cudaSetDevice(device),"select expert bank device"))return nullptr;
    Dsv4CudaExpertSet*set=(Dsv4CudaExpertSet*)calloc(1,sizeof(*set));if(!set)return nullptr;
    set->count=count;set->device=device;set->H=H;set->I=I;set->sg=sg;set->su=su;set->sd=sd;
    size_t fc1_w=(size_t)count*I*H,fc1_s=(size_t)count*2*I*(H/32);
    size_t fc2_w=(size_t)count*H*(I/2),fc2_s=(size_t)count*H*(I/32);
    if(!ok(cudaMalloc(&set->bank_fc1,fc1_w),"expert FC1 bank allocation")||
       !ok(cudaMalloc(&set->bank_fc1_scale,fc1_s),"expert FC1 scale bank allocation")||
       !ok(cudaMalloc(&set->bank_fc2,fc2_w),"expert FC2 bank allocation")||
       !ok(cudaMalloc(&set->bank_fc2_scale,fc2_s),"expert FC2 scale bank allocation")||
#ifdef COLI_DSV4_DEEPGEMM
       !ok(cudaMalloc(&set->bank_fc1_dg_scale,fc1_s),"expert FC1 DeepGEMM scale allocation")||
       !ok(cudaMalloc(&set->bank_fc2_dg_scale,fc2_s),"expert FC2 DeepGEMM scale allocation")||
#endif
       !ok(cudaMalloc(&set->table,(size_t)3*count*sizeof(ExpertPtr)),"expert table allocation")){
        dsv4_cuda_expert_set_free(set);return nullptr;
    }
    return set;
}
/* ---- transparent host pinning for expert uploads ----
 * The engine's expert-cache slot slabs are stable for the process lifetime
 * (posix_memalign once per slot, freed only at store destroy), so the first
 * upload from a slab page-locks its whole allocation region (VirtualQuery
 * bounds) and every later upload from it is a pure DMA at PCIe speed instead
 * of a CPU staging copy (~4.7 GB/s measured while the refill's parallel
 * lookups compete for the cores). Failures fall back to pageable copies;
 * DSV4_CUDA_PIN_HOST=0 disables; total pinned bytes are capped. */
struct PinRegion{uintptr_t base;size_t size;};
static std::vector<PinRegion> g_pin_regions,g_pin_failed;
static size_t g_pin_bytes;
static int g_pin_enabled=-1;
static bool host_pinned(const void*p,size_t len){
    if(g_pin_enabled<0){const char*e=getenv("DSV4_CUDA_PIN_HOST");g_pin_enabled=!(e&&*e=='0');}
    if(!g_pin_enabled||!p||!len)return false;
    uintptr_t a=(uintptr_t)p;
    for(auto&r:g_pin_regions)if(a>=r.base&&a+len<=r.base+r.size)return true;
    for(auto&r:g_pin_failed)if(a>=r.base&&a+len<=r.base+r.size)return false;
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if(!VirtualQuery(p,&mbi,sizeof mbi)||mbi.State!=MEM_COMMIT)return false;
    uintptr_t base=(uintptr_t)mbi.BaseAddress;size_t size=mbi.RegionSize;
    if(a<base||a+len>base+size)return false;
    static const size_t cap=(size_t)20<<30;
    if(g_pin_bytes+size>cap){g_pin_failed.push_back({base,size});return false;}
    cudaError_t err=cudaHostRegister((void*)base,size,cudaHostRegisterPortable);
    if(err==cudaErrorHostMemoryAlreadyRegistered){cudaGetLastError();g_pin_regions.push_back({base,size});return true;}
    if(err!=cudaSuccess){cudaGetLastError();g_pin_failed.push_back({base,size});return false;}
    g_pin_bytes+=size;g_pin_regions.push_back({base,size});
    return true;
#else
    /* Linux/WSL: no VirtualQuery, but none is needed — cudaHostRegister pins
     * whole pages, so register the requested range rounded out to page
     * boundaries. The slot slabs are stable for the process lifetime (one
     * posix_memalign per slot, freed only at store destroy), exactly like the
     * Windows branch assumes. Overlapping ranges collapse into the
     * already-registered success case; any failure (WSL builds without
     * pinning support included) lands in g_pin_failed and every later upload
     * from that slab takes the pageable path, same as before this branch
     * existed. */
    long page=sysconf(_SC_PAGESIZE);if(page<=0)page=4096;
    uintptr_t base=a&~((uintptr_t)page-1);
    size_t size=((a+len+page-1)&~((uintptr_t)page-1))-base;
    static const size_t cap=(size_t)20<<30;
    if(g_pin_bytes+size>cap){g_pin_failed.push_back({base,size});return false;}
    cudaError_t err=cudaHostRegister((void*)base,size,cudaHostRegisterPortable);
    if(err==cudaErrorHostMemoryAlreadyRegistered){cudaGetLastError();g_pin_regions.push_back({base,size});return true;}
    if(err!=cudaSuccess){cudaGetLastError();g_pin_failed.push_back({base,size});return false;}
    g_pin_bytes+=size;g_pin_regions.push_back({base,size});
    return true;
#endif
}

/* Drain the device's expert stream: the hybrid decode split enqueues its
 * fill-set uploads asynchronously, computes the CPU subset while the DMA is
 * in flight, and closes the pipeline here before the GPU group runs. */
extern "C" int dsv4_cuda_stream_drain(int device){
    Dev*c=ctx(device);if(!c)return 0;
    if(!ok(cudaSetDevice(device),"select drain device"))return 0;
    return ok(cudaStreamSynchronize(c->stream),"hybrid drain")?1:0;
}

extern "C" int dsv4_cuda_expert_bank_upload(Dsv4CudaExpertSet*set,int e,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds,
        Dsv4CudaTensor**gate,Dsv4CudaTensor**up,Dsv4CudaTensor**down){
    if(!set||e<0||e>=set->count||!gw||!gs||!uw||!us||!dw||!ds||!gate||!up||!down||*gate||*up||*down||
       !ok(cudaSetDevice(set->device),"select expert bank upload device"))return 0;
    int H=set->H,I=set->I;size_t w1=(size_t)I*(H/2),s1=(size_t)I*(H/32),w2=(size_t)H*(I/2),s2=(size_t)H*(I/32);
    uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
    uint8_t*fc2=set->bank_fc2+(size_t)e*w2,*fc2s=set->bank_fc2_scale+(size_t)e*s2;
    Dev*c=ctx(set->device);
    bool pinned=c&&host_pinned(gw,w1)&&host_pinned(uw,w1)&&host_pinned(gs,s1)&&
                host_pinned(us,s1)&&host_pinned(dw,w2)&&host_pinned(ds,s2);
    if(pinned){
        /* Pure DMA on the device stream; the caller releases the host slab
         * right after we return, so drain before returning. */
        if(!ok(cudaMemcpyAsync(fc1,gw,w1,cudaMemcpyHostToDevice,c->stream),"expert gate upload")||
           !ok(cudaMemcpyAsync(fc1+w1,uw,w1,cudaMemcpyHostToDevice,c->stream),"expert up upload")||
           !ok(cudaMemcpyAsync(fc1s,gs,s1,cudaMemcpyHostToDevice,c->stream),"expert gate scale upload")||
           !ok(cudaMemcpyAsync(fc1s+s1,us,s1,cudaMemcpyHostToDevice,c->stream),"expert up scale upload")||
           !ok(cudaMemcpyAsync(fc2,dw,w2,cudaMemcpyHostToDevice,c->stream),"expert down upload")||
           !ok(cudaMemcpyAsync(fc2s,ds,s2,cudaMemcpyHostToDevice,c->stream),"expert down scale upload")||
           !ok(cudaStreamSynchronize(c->stream),"expert upload drain"))return 0;
    }else
    if(!ok(cudaMemcpy(fc1,gw,w1,cudaMemcpyHostToDevice),"expert gate upload")||
       !ok(cudaMemcpy(fc1+w1,uw,w1,cudaMemcpyHostToDevice),"expert up upload")||
       !ok(cudaMemcpy(fc1s,gs,s1,cudaMemcpyHostToDevice),"expert gate scale upload")||
       !ok(cudaMemcpy(fc1s+s1,us,s1,cudaMemcpyHostToDevice),"expert up scale upload")||
       !ok(cudaMemcpy(fc2,dw,w2,cudaMemcpyHostToDevice),"expert down upload")||
       !ok(cudaMemcpy(fc2s,ds,s2,cudaMemcpyHostToDevice),"expert down scale upload"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    dg_pack_weight_scale<<<(2*I*(H/32/4)+255)/256,256>>>(set->bank_fc1_dg_scale,fc1s,e,2*I,H/32);
    dg_pack_weight_scale<<<(H*(I/32/4)+255)/256,256>>>(set->bank_fc2_dg_scale,fc2s,e,H,I/32);
    if(!ok(cudaGetLastError(),"expert DeepGEMM scale packing"))return 0;
#endif
    Dsv4CudaTensor*g=expert_view(fc1,fc1s,I,H,set->device),*u=expert_view(fc1+w1,fc1s+s1,I,H,set->device),*d=expert_view(fc2,fc2s,H,I,set->device);
    if(!g||!u||!d){free(g);free(u);free(d);return 0;}
    ExpertPtr p[3]={{(uint8_t*)g->w,g->scale},{(uint8_t*)u->w,u->scale},{(uint8_t*)d->w,d->scale}};
    if(!ok(cudaMemcpy(set->table+e,p,sizeof(*p),cudaMemcpyHostToDevice),"expert gate table upload")||
       !ok(cudaMemcpy(set->table+set->count+e,p+1,sizeof(*p),cudaMemcpyHostToDevice),"expert up table upload")||
       !ok(cudaMemcpy(set->table+2*set->count+e,p+2,sizeof(*p),cudaMemcpyHostToDevice),"expert down table upload")){
        free(g);free(u);free(d);return 0;
    }
    *gate=g;*up=u;*down=d;return 1;
}
extern "C" int dsv4_cuda_expert_bank_upload_tp2(Dsv4CudaExpertSet*set,int e,int rank,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds){
    if(!set||e<0||e>=set->count||(rank!=0&&rank!=1)||!gw||!gs||!uw||!us||!dw||!ds||
       !ok(cudaSetDevice(set->device),"select TP2 expert bank upload device"))return 0;
    int H=set->H,J=set->I,I=2*J;size_t row_w=H/2,row_s=H/32;
    size_t w1=(size_t)J*row_w,s1=(size_t)J*row_s,w2=(size_t)H*(J/2),s2=(size_t)H*(J/32);
    uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
    uint8_t*fc2=set->bank_fc2+(size_t)e*w2,*fc2s=set->bank_fc2_scale+(size_t)e*s2;
    if(!ok(cudaMemcpy(fc1,gw+(size_t)rank*w1,w1,cudaMemcpyHostToDevice),"TP2 expert gate upload")||
       !ok(cudaMemcpy(fc1+w1,uw+(size_t)rank*w1,w1,cudaMemcpyHostToDevice),"TP2 expert up upload")||
       !ok(cudaMemcpy(fc1s,gs+(size_t)rank*s1,s1,cudaMemcpyHostToDevice),"TP2 expert gate scale upload")||
       !ok(cudaMemcpy(fc1s+s1,us+(size_t)rank*s1,s1,cudaMemcpyHostToDevice),"TP2 expert up scale upload")||
       !ok(cudaMemcpy2D(fc2,J/2,dw+(size_t)rank*(J/2),I/2,J/2,H,cudaMemcpyHostToDevice),"TP2 expert down upload")||
       !ok(cudaMemcpy2D(fc2s,J/32,ds+(size_t)rank*(J/32),I/32,J/32,H,cudaMemcpyHostToDevice),"TP2 expert down scale upload"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    dg_pack_weight_scale<<<(2*J*(H/32/4)+255)/256,256>>>(set->bank_fc1_dg_scale,fc1s,e,2*J,H/32);
    dg_pack_weight_scale<<<(H*(J/32/4)+255)/256,256>>>(set->bank_fc2_dg_scale,fc2s,e,H,J/32);
    if(!ok(cudaGetLastError(),"TP2 expert DeepGEMM scale packing"))return 0;
#endif
    return 1;
}
extern "C" void dsv4_cuda_expert_set_free(Dsv4CudaExpertSet*set){if(!set)return;cudaSetDevice(set->device);if(set->table)cudaFree(set->table);if(set->hash)cudaFree(set->hash);if(set->bank_fc1)cudaFree(set->bank_fc1);if(set->bank_fc1_scale)cudaFree(set->bank_fc1_scale);if(set->bank_fc2)cudaFree(set->bank_fc2);if(set->bank_fc2_scale)cudaFree(set->bank_fc2_scale);if(set->bank_fc1_dg_scale)cudaFree(set->bank_fc1_dg_scale);if(set->bank_fc2_dg_scale)cudaFree(set->bank_fc2_dg_scale);free(set);}
/* Rebind the shared-expert mirrors on a bank set. The bank is a per-layer
 * streaming buffer during batched prefill, but the shared experts differ per
 * layer and are already resident as the layer's fp8 mirrors — rebinding them
 * is free, re-creating the ~2 GiB bank per layer is not. Same shape contract
 * as dsv4_cuda_expert_bank_create. */
extern "C" int dsv4_cuda_expert_bank_set_shared(Dsv4CudaExpertSet*set,Dsv4CudaTensor*sg,Dsv4CudaTensor*su,Dsv4CudaTensor*sd){
    if(!set)return 0;
    if(!sg&&!su&&!sd){set->sg=set->su=set->sd=nullptr;return 1;}
    int H=set->H,J=sg?sg->O:0;
    if(!sg||!su||!sd||sg->device!=set->device||su->device!=set->device||sd->device!=set->device||
       sg->fmt!=8||su->fmt!=8||sd->fmt!=8||sg->I!=H||su->I!=H||su->O!=J||sd->I!=J||sd->O!=H)return 0;
    set->sg=sg;set->su=su;set->sd=sd;return 1;
}
extern "C" int dsv4_cuda_expert_set_upload_hash(Dsv4CudaExpertSet*set,const int64_t*map,int vocab,int topk){
    if(!set||!map||vocab<1||topk!=6||!ok(cudaSetDevice(set->device),"select hash router device"))return 0;
    size_t bytes=(size_t)vocab*topk*sizeof(*map);if(set->hash)cudaFree(set->hash);set->hash=nullptr;set->vocab=0;set->topk=0;
    if(!ok(cudaMalloc(&set->hash,bytes),"hash router allocation")||!ok(cudaMemcpy(set->hash,map,bytes,cudaMemcpyHostToDevice),"hash router upload"))return 0;
    set->vocab=vocab;set->topk=topk;return 1;
}
extern "C" int dsv4_cuda_route_moe(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,
        int token,float scale,Dsv4CudaExpertSet*set,float limit,Dsv4CudaActivation*output){
    Dev*c=input?ctx(input->device):nullptr;if(!c||!gate||!set||!output||set->device!=input->device||output->device!=input->device||
       gate->fmt!=32||gate->O!=set->count||gate->I!=set->H||gate->device!=input->device||input->elements<set->H||output->elements<set->H||
       (bias&&(bias->fmt!=32||bias->O*bias->I<set->count||bias->device!=input->device))||set->count!=256||token<0||
       (set->hash&&token>=set->vocab)||
       !ok(cudaSetDevice(input->device),"select routed MoE device"))return 0;
    int H=set->H,I=set->I,K=6;size_t hb=(size_t)H*4,ib=(size_t)K*I*4,parts=(size_t)K*H*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,(size_t)set->count*4)||!buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,parts)||
       !buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)3*K*sizeof(MvDesc))||
       !buf((void**)&c->expert_weights,&c->expert_weightscap,(size_t)K*sizeof(float))||
       !buf((void**)&c->expert_ids,&c->expert_idcap,(size_t)K*sizeof(int)))return 0;
    mv_f32<<<set->count,256,0,c->stream>>>((float*)gate->w,input->data,c->dy,set->count,H);
    if(!ok(cudaMemsetAsync(c->expert_ids,0,(size_t)K*sizeof(int),c->stream),"routed MoE ids clear"))return 0;
    if(set->hash)load_hash_ids<<<1,K,0,c->stream>>>(c->expert_ids,set->hash,c->decode_state,set->topk);
    if(set->hash)route_fixed6<<<1,K,0,c->stream>>>(c->expert_ids,c->expert_weights,c->dy,scale);
    else route_top6<<<1,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,0,scale);
    if(!ok(cudaMemcpyAsync(c->dx,input->data,hb,cudaMemcpyDeviceToDevice,c->stream),"routed MoE input copy"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    static int deepgemm=-1;if(deepgemm<0)deepgemm=getenv("DSV4_CUDA_DEEPGEMM")?atoi(getenv("DSV4_CUDA_DEEPGEMM")):1;
    if(deepgemm){
        static int announced;if(!announced++){fprintf(stderr,"[DSV4 CUDA] DeepGEMM routed experts active\n");}
        int ab=getenv("DSV4_CUDA_DG_AB")&&atoi(getenv("DSV4_CUDA_DG_AB"));
        if(ab){
            if(!buf((void**)&c->dgref,&c->dgrefcap,hb))return 0;
            fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);
            build_moe_desc<<<1,K,0,c->stream>>>(c->mvdesc,set->table,c->expert_ids,c->dx,c->p1,c->p2,c->p3,c->p4,K,set->count,H,I);
            run_fp4_grouped(c->mvdesc,2*K,I,H,c->stream);
            expert_act_grouped<<<(K*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,K,I);
            fp8_sim<<<K*((I+127)/128),128,0,c->stream>>>(c->p3,K,I);
            run_fp4_grouped(c->mvdesc+2*K,K,H,I,c->stream);
            expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dgref,c->p4,K,H);
            if(!ok(cudaMemcpyAsync(c->dx,input->data,hb,cudaMemcpyDeviceToDevice,c->stream),"DeepGEMM A/B input restore"))return 0;
        }
        cudaEvent_t dg_begin=nullptr,dg_end=nullptr,shared_end=nullptr;int profile=getenv("DSV4_CUDA_DG_PROFILE")&&atoi(getenv("DSV4_CUDA_DG_PROFILE"));
        if(profile){cudaEventCreate(&dg_begin);cudaEventCreate(&dg_end);cudaEventCreate(&shared_end);cudaEventRecord(dg_begin,c->stream);}
        static int masked=-1;if(masked<0)masked=getenv("DSV4_CUDA_MOE_MASKED")?atoi(getenv("DSV4_CUDA_MOE_MASKED")):0;
        int routed=set->count==256?
            (masked?dg_moe<256>(c,set,c->expert_weights,limit,output->data):dg_moe_contiguous_fast<256>(c,set,c->expert_weights,limit,output->data)):
            set->count==128?(masked?dg_moe<128>(c,set,c->expert_weights,limit,output->data):dg_moe_contiguous_fast<128>(c,set,c->expert_weights,limit,output->data)):0;
        if(!routed)return 0;
        if(!dg_dump_once(c))return 0;
        if(ab){float hs[6];
            dg_compare_fc1<<<1,256,0,c->stream>>>(c->p1,c->p2,c->dgmm1,c->dy);
            dg_compare_fc2<<<1,256,0,c->stream>>>(c->p4,c->dgmm2,c->dy+2);
            dg_compare<<<1,256,0,c->stream>>>(c->dgref,output->data,c->dy+4,H);
            if(!ok(cudaMemcpyAsync(hs,c->dy,sizeof(hs),cudaMemcpyDeviceToHost,c->stream),"DeepGEMM A/B stats")||!ok(cudaStreamSynchronize(c->stream),"DeepGEMM A/B sync"))return 0;
            fprintf(stderr,"[DSV4 DeepGEMM A/B] fc1 max=%.6g mae=%.6g fc2 max=%.6g mae=%.6g routed max=%.6g mae=%.6g\n",hs[0],hs[1],hs[2],hs[3],hs[4],hs[5]);}
        if(profile)cudaEventRecord(dg_end,c->stream);
        int dense_dg=getenv("DSV4_CUDA_DENSE_DG")&&atoi(getenv("DSV4_CUDA_DENSE_DG"));
        if(dense_dg){if(!dg_dense_2048x4096(c,set->sg,c->dx,c->p1)||!dg_dense_2048x4096(c,set->su,c->dx,c->p2))return 0;}
        else{fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);run_mv<8>((uint8_t*)set->sg->w,set->sg->scale,c->dx,c->p1,I,H,1,c->stream);run_mv<8>((uint8_t*)set->su->w,set->su->scale,c->dx,c->p2,I,H,1,c->stream);}
        expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);
        if(dense_dg){if(!dg_dense_4096x2048(c,set->sd,c->p3,c->p4))return 0;}else{fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);run_mv<8>((uint8_t*)set->sd->w,set->sd->scale,c->p3,c->p4,H,I,1,c->stream);}
        moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
        if(profile){float routed_ms=0,shared_ms=0;cudaEventRecord(shared_end,c->stream);cudaEventSynchronize(shared_end);cudaEventElapsedTime(&routed_ms,dg_begin,dg_end);cudaEventElapsedTime(&shared_ms,dg_end,shared_end);fprintf(stderr,"[DSV4 DeepGEMM] routed=%.3fms shared=%.3fms\n",routed_ms,shared_ms);cudaEventDestroy(dg_begin);cudaEventDestroy(dg_end);cudaEventDestroy(shared_end);}
        return ok(cudaGetLastError(),"DeepGEMM routed and shared MoE launch");
    }
#endif
    fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);
    build_moe_desc<<<1,K,0,c->stream>>>(c->mvdesc,set->table,c->expert_ids,c->dx,c->p1,c->p2,c->p3,c->p4,K,set->count,H,I);
    run_fp4_grouped(c->mvdesc,2*K,I,H,c->stream);
    expert_act_grouped<<<(K*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,K,I);
    fp8_sim<<<K*((I+127)/128),128,0,c->stream>>>(c->p3,K,I);
    run_fp4_grouped(c->mvdesc+2*K,K,H,I,c->stream);
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,K,H);
    run_mv<8>((uint8_t*)set->sg->w,set->sg->scale,c->dx,c->p1,I,H,1,c->stream);
    run_mv<8>((uint8_t*)set->su->w,set->su->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);
    run_mv<8>((uint8_t*)set->sd->w,set->sd->scale,c->p3,c->p4,H,I,1,c->stream);moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
    return ok(cudaGetLastError(),"routed MoE launch");
}
extern "C" int dsv4_cuda_route_moe_batch(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,const int*tokens,int count,float scale,Dsv4CudaExpertSet*set,float limit,Dsv4CudaActivation*output){
#ifdef COLI_DSV4_DEEPGEMM
    Dev*c=input?ctx(input->device):nullptr;if(!c||!gate||!set||!output||count<1||count>128||set->device!=input->device||output->device!=input->device||gate->fmt!=32||gate->O!=256||gate->I!=4096||gate->device!=input->device||set->count!=256||set->H!=4096||set->I!=2048||input->elements<(long long)count*4096||output->elements<(long long)count*4096||(bias&&(bias->fmt!=32||bias->O*bias->I<256||bias->device!=input->device))||(set->hash&&!tokens)||!ok(cudaSetDevice(input->device),"select batched MoE device"))return 0;
    int routes=count*6;size_t logits=(size_t)count*256*sizeof(float),ids=(size_t)routes*sizeof(int),weights=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->dy,&c->ycap,logits)||!buf((void**)&c->expert_ids,&c->expert_idcap,ids)||!buf((void**)&c->expert_weights,&c->expert_weightscap,weights)||
       (set->hash&&!buf((void**)&c->aux3,&c->aux3cap,(size_t)count*sizeof(int))))return 0;
    route_logits_batch<<<count*256,256,0,c->stream>>>((float*)gate->w,input->data,c->dy,count,256,4096);
    if(set->hash){if(!ok(cudaMemcpyAsync(c->aux3,tokens,(size_t)count*sizeof(int),cudaMemcpyHostToDevice,c->stream),"batch token upload"))return 0;route_hash_batch<<<(routes+255)/256,256,0,c->stream>>>(c->expert_ids,set->hash,(int*)c->aux3,count);route_fixed_weights_batch<<<count,6,0,c->stream>>>(c->expert_weights,c->expert_ids,c->dy,count,scale);}
    else route_top6_batch<<<count,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,count,scale);
    /* DSV4_CUDA_MOE_PROF=1: wall-time the three segments (forced syncs, so
     * only for diagnosis) and print the first few calls. */
    static int prof=-1;static int prof_calls;if(prof<0){const char*e=getenv("DSV4_CUDA_MOE_PROF");prof=e&&atoi(e);}
    double t_route=0,t_expert=0,t_shared=0;std::chrono::steady_clock::time_point ta,tb;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);ta=std::chrono::steady_clock::now();}
    if(!dg_moe_batch_contiguous(c,set,input->data,c->expert_ids,c->expert_weights,count,limit,output->data))return 0;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_expert=std::chrono::duration<double>(tb-ta).count();ta=tb;}
    Dsv4CudaTensor*sg=set->sg,*su=set->su,*sd=set->sd;if(!sg||!su||!sd||sg->O!=2048||su->O!=2048||sd->I!=2048)return 0;
    size_t mid=(size_t)count*2048*sizeof(float),hb=(size_t)count*4096*sizeof(float);if(!buf((void**)&c->p1,&c->p1cap,mid)||!buf((void**)&c->p2,&c->p2cap,mid)||!buf((void**)&c->p3,&c->p3cap,mid)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!dg_dense_batch_dispatch(c,sg,input->data,c->p1,count)||!dg_dense_batch_dispatch(c,su,input->data,c->p2,count))return 0;swiglu_rows<<<((long long)count*2048+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,limit,count,2048);if(!dg_dense_batch_dispatch(c,sd,c->p3,c->p4,count))return 0;add_rows<<<((long long)count*4096+255)/256,256,0,c->stream>>>(output->data,c->p4,count*4096);
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_shared=std::chrono::duration<double>(tb-ta).count();
        fprintf(stderr,"[DSV4 CUDA] moe-batch prof count=%d experts=%.3fs shared=%.3fs\n",count,t_expert,t_shared);prof_calls++;}
    (void)t_route;return ok(cudaGetLastError(),"batched MoE launch");
#else
    (void)input;(void)gate;(void)bias;(void)tokens;(void)count;(void)scale;(void)set;(void)limit;(void)output;return 0;
#endif
}
/* Batched top-6 routing only: logits from the f32 gate mirror, ids/weights
 * downloaded to the host so the caller can stage exactly the routed experts
 * before running the expert GEMMs. */
extern "C" int dsv4_cuda_route_top6_batch(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,int count,float scale,int*ids,float*weights){
    Dev*c=input?ctx(input->device):nullptr;
    if(!c||!gate||!ids||!weights||count<1||count>128||gate->fmt!=32||gate->O!=256||gate->I!=4096||gate->device!=input->device||
       input->elements<(long long)count*4096||(bias&&(bias->fmt!=32||bias->O*bias->I<256||bias->device!=input->device))||
       !ok(cudaSetDevice(input->device),"select batched route device"))return 0;
    int routes=count*6;size_t logits=(size_t)count*256*sizeof(float),ib=(size_t)routes*sizeof(int),wb=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->dy,&c->ycap,logits)||!buf((void**)&c->expert_ids,&c->expert_idcap,ib)||!buf((void**)&c->expert_weights,&c->expert_weightscap,wb))return 0;
    route_logits_batch<<<count*256,256,0,c->stream>>>((float*)gate->w,input->data,c->dy,count,256,4096);
    route_top6_batch<<<count,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,count,scale);
    return ok(cudaGetLastError(),"batched route launch")&&
           ok(cudaMemcpyAsync(ids,c->expert_ids,ib,cudaMemcpyDeviceToHost,c->stream),"batched route id download")&&
           ok(cudaMemcpyAsync(weights,c->expert_weights,wb,cudaMemcpyDeviceToHost,c->stream),"batched route weight download")&&
           ok(cudaStreamSynchronize(c->stream),"batched route sync");
}
/* Batched MoE over host-provided routes (ids/weights per token, 6 each).
 * The caller guarantees every id's bank slot holds that expert's weights;
 * routed sum and the shared expert land in output like route_moe_batch. */
extern "C" int dsv4_cuda_route_moe_ids_batch(const Dsv4CudaActivation*input,const int*ids,const float*weights,int count,Dsv4CudaExpertSet*set,float limit,Dsv4CudaActivation*output){
#ifdef COLI_DSV4_DEEPGEMM
    Dev*c=input?ctx(input->device):nullptr;
    if(!c||!ids||!weights||!set||!output||count<1||count>128||set->device!=input->device||output->device!=input->device||
       set->count!=256||set->H!=4096||set->I!=2048||!set->bank_fc1||
       input->elements<(long long)count*4096||output->elements<(long long)count*4096||
       !ok(cudaSetDevice(input->device),"select batched MoE device"))return 0;
    int routes=count*6;
    for(int r=0;r<routes;r++)if(ids[r]<0||ids[r]>=set->count)return 0;
    size_t ib=(size_t)routes*sizeof(int),wb=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->expert_ids,&c->expert_idcap,ib)||!buf((void**)&c->expert_weights,&c->expert_weightscap,wb))return 0;
    if(!ok(cudaMemcpyAsync(c->expert_ids,ids,ib,cudaMemcpyHostToDevice,c->stream),"batched MoE id upload")||
       !ok(cudaMemcpyAsync(c->expert_weights,weights,wb,cudaMemcpyHostToDevice,c->stream),"batched MoE weight upload"))return 0;
    static int prof=-1;static int prof_calls;if(prof<0){const char*e=getenv("DSV4_CUDA_MOE_PROF");prof=e&&atoi(e);}
    double t_expert=0,t_shared=0;std::chrono::steady_clock::time_point ta,tb;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);ta=std::chrono::steady_clock::now();}
    if(!dg_moe_batch_contiguous(c,set,input->data,c->expert_ids,c->expert_weights,count,limit,output->data,ids))return 0;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_expert=std::chrono::duration<double>(tb-ta).count();ta=tb;}
    Dsv4CudaTensor*sg=set->sg,*su=set->su,*sd=set->sd;if(!sg||!su||!sd||sg->O!=2048||su->O!=2048||sd->I!=2048)return 0;
    size_t mid=(size_t)count*2048*sizeof(float),hb=(size_t)count*4096*sizeof(float);
    if(!buf((void**)&c->p1,&c->p1cap,mid)||!buf((void**)&c->p2,&c->p2cap,mid)||!buf((void**)&c->p3,&c->p3cap,mid)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!dg_dense_batch_dispatch(c,sg,input->data,c->p1,count)||!dg_dense_batch_dispatch(c,su,input->data,c->p2,count))return 0;
    swiglu_rows<<<((long long)count*2048+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,limit,count,2048);
    if(!dg_dense_batch_dispatch(c,sd,c->p3,c->p4,count))return 0;
    add_rows<<<((long long)count*4096+255)/256,256,0,c->stream>>>(output->data,c->p4,count*4096);
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_shared=std::chrono::duration<double>(tb-ta).count();
        fprintf(stderr,"[DSV4 CUDA] moe-ids prof count=%d experts=%.3fs shared=%.3fs\n",count,t_expert,t_shared);prof_calls++;}
    return ok(cudaGetLastError(),"batched MoE ids launch");
#else
    /* GENERIC batched MoE (no DeepGEMM): the decode kernels applied to every
     * token of the chunk in one grouped launch each — fp32 accumulate over
     * fp4-decoded expert rows from the same VRAM bank, then the fp8 shared
     * experts through the generic batch matmul. Slower than the tensor-core
     * path (~10-20 ms per 128-token chunk-layer) but runs on any sm_80+ GPU,
     * so prefill stays GPU-side instead of falling back to the CPU union. */
    Dev*c=input?ctx(input->device):nullptr;
    if(!c||!ids||!weights||!set||!output||count<1||count>128||set->device!=input->device||output->device!=input->device||
       set->count!=256||set->H!=4096||set->I!=2048||!set->bank_fc1||
       input->elements<(long long)count*4096||output->elements<(long long)count*4096||
       !ok(cudaSetDevice(input->device),"select generic batched MoE device"))return 0;
    const int H=4096,I=2048,K=6;int routes=count*K;
    for(int r=0;r<routes;r++)if(ids[r]<0||ids[r]>=set->count)return 0;
    Dsv4CudaTensor*sg=set->sg,*su=set->su,*sd=set->sd;
    if(!sg||!su||!sd||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||sg->O!=I||su->O!=I||sd->I!=I||sg->I!=H||sd->O!=H)return 0;
    size_t w1=(size_t)I*(H/2),s1=(size_t)I*(H/32),w2=(size_t)H*(I/2),s2=(size_t)H*(I/32);
    size_t rb=(size_t)routes*I*sizeof(float),pb=(size_t)routes*H*sizeof(float),wb=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->p1,&c->p1cap,rb)||!buf((void**)&c->p2,&c->p2cap,rb)||!buf((void**)&c->p3,&c->p3cap,rb)||
       !buf((void**)&c->p4,&c->p4cap,pb)||!buf((void**)&c->expert_weights,&c->expert_weightscap,wb)||
       !buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)routes*2*sizeof(MvDesc)))return 0;
    if(!ok(cudaMemcpyAsync(c->expert_weights,weights,wb,cudaMemcpyHostToDevice,c->stream),"generic MoE weight upload"))return 0;
    std::vector<MvDesc> hd((size_t)routes*2);
    for(int r=0;r<routes;r++){int e=ids[r],t=r/K;
        const uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
        const float*x=input->data+(long long)t*H;
        hd[2*r]={fc1,fc1s,x,c->p1+(long long)r*I};
        hd[2*r+1]={fc1+w1,fc1s+s1,x,c->p2+(long long)r*I};}
    if(!ok(cudaMemcpyAsync(c->mvdesc,hd.data(),(size_t)routes*2*sizeof(MvDesc),cudaMemcpyHostToDevice,c->stream),"generic MoE fc1 descriptors"))return 0;
    mv_fp4_grouped<4><<<dim3((I+3)/4,routes*2),256,0,c->stream>>>(c->mvdesc,routes*2,I,H);
    expert_act_grouped<<<((long long)routes*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,routes,I);
    fp8_sim<<<routes*((I+127)/128),128,0,c->stream>>>(c->p3,routes,I);
    /* down: descriptors must not be overwritten while fc1 still reads them —
     * the copy is stream-ordered after the launches above, so reuse is safe. */
    for(int r=0;r<routes;r++){int e=ids[r];
        hd[r]={set->bank_fc2+(size_t)e*w2,set->bank_fc2_scale+(size_t)e*s2,c->p3+(long long)r*I,c->p4+(long long)r*H};}
    if(!ok(cudaMemcpyAsync(c->mvdesc,hd.data(),(size_t)routes*sizeof(MvDesc),cudaMemcpyHostToDevice,c->stream),"generic MoE down descriptors"))return 0;
    mv_fp4_grouped<4><<<dim3((H+3)/4,routes),256,0,c->stream>>>(c->mvdesc,routes,H,I);
    expert_reduce_rows<<<((long long)count*H+255)/256,256,0,c->stream>>>(output->data,c->p4,count,K,H);
    /* shared experts (fp8 dense) over the whole chunk, then the same combine
     * as decode: routed = bf16(routed + bf16(shared)). */
    size_t mid=(size_t)count*I*sizeof(float),hb=(size_t)count*H*sizeof(float);
    if(!buf((void**)&c->p1,&c->p1cap,mid)||!buf((void**)&c->p2,&c->p2cap,mid)||!buf((void**)&c->p3,&c->p3cap,mid)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    run_mm_batch(sg,input->data,c->p1,count,c->stream);run_mm_batch(su,input->data,c->p2,count,c->stream);
    expert_act_rows1<<<((long long)count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,limit,count*I);
    fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
    run_mm_batch(sd,c->p3,c->p4,count,c->stream);
    moe_combine<<<((long long)count*H+255)/256,256,0,c->stream>>>(output->data,c->p4,count*H);
    return ok(cudaGetLastError(),"generic batched MoE launch");
#endif
}
extern "C" int dsv4_cuda_route_moe_ep2(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,
        const Dsv4CudaActivation*peer_input,Dsv4CudaTensor*peer_gate,Dsv4CudaTensor*peer_bias,int token,float scale,
        Dsv4CudaExpertSet*local,Dsv4CudaExpertSet*peer,float limit,Dsv4CudaActivation*output,Dsv4CudaActivation*peer_output){
#ifndef COLI_DSV4_DEEPGEMM
    (void)input;(void)gate;(void)bias;(void)peer_input;(void)peer_gate;(void)peer_bias;(void)token;(void)scale;(void)local;(void)peer;(void)limit;(void)output;(void)peer_output;return 0;
#else
    Dev*c=input?ctx(input->device):nullptr,*p=peer?ctx(peer->device):nullptr;
    int replicated=getenv("DSV4_CUDA_REPLICATED_TP2")&&atoi(getenv("DSV4_CUDA_REPLICATED_TP2"));
    int moe_tp2=getenv("DSV4_CUDA_MOE_TP2")&&atoi(getenv("DSV4_CUDA_MOE_TP2"));
    if(!c||!p||!gate||!local||!peer||!output||!peer_output||
       (moe_tp2?(!replicated||local->count!=256||peer->count!=256||local->I!=1024):
                (local->count!=128||peer->count!=128))||local->H!=peer->H||local->I!=peer->I||
       gate->fmt!=32||gate->O!=256||gate->I!=local->H||gate->device!=input->device||output->device!=input->device||peer_output->device!=peer->device||
       (bias&&(bias->fmt!=32||bias->O*bias->I<256||bias->device!=input->device))||
       (replicated&&(!peer_input||!peer_gate||peer_input->device!=peer->device||peer_gate->device!=peer->device||peer_gate->fmt!=32||peer_gate->O!=256||peer_gate->I!=local->H||
       (peer_bias&&(peer_bias->device!=peer->device||peer_bias->fmt!=32||peer_bias->O*peer_bias->I<256))))||token<0||(local->hash&&token>=local->vocab))return 0;
    int H=local->H,I=local->I,K=6;size_t hb=(size_t)H*4,ib=(size_t)K*I*4;
    if(!ok(cudaSetDevice(c->id),"select EP2 primary")||!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,256*4)||
       !buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||!buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,hb)||
       !buf((void**)&c->expert_ids,&c->expert_idcap,K*sizeof(int))||!buf((void**)&c->expert_weights,&c->expert_weightscap,K*sizeof(float))||
       !buf((void**)&c->aux1,&c->aux1cap,hb))return 0;
    mv_f32<<<256,256,0,c->stream>>>((float*)gate->w,input->data,c->dy,256,H);
    if(!ok(cudaMemsetAsync(c->expert_ids,0,K*sizeof(int),c->stream),"EP2 ids clear"))return 0;
    if(local->hash)load_hash_ids<<<1,K,0,c->stream>>>(c->expert_ids,local->hash,c->decode_state,local->topk);
    if(local->hash)route_fixed6<<<1,K,0,c->stream>>>(c->expert_ids,c->expert_weights,c->dy,scale);
    else route_top6<<<1,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,0,scale);
    if(!ok(cudaMemcpyAsync(c->dx,input->data,hb,cudaMemcpyDeviceToDevice,c->stream),"EP2 primary input")||
       !ok(cudaSetDevice(p->id),"select EP2 peer")||!buf((void**)&p->dx,&p->xcap,hb)||!buf((void**)&p->dy,&p->ycap,256*4)||
       !buf((void**)&p->expert_ids,&p->expert_idcap,K*sizeof(int))||!buf((void**)&p->expert_weights,&p->expert_weightscap,K*sizeof(float))||
       !buf((void**)&p->p1,&p->p1cap,(size_t)I*4)||!buf((void**)&p->p2,&p->p2cap,(size_t)I*4)||
       !buf((void**)&p->p3,&p->p3cap,(size_t)I*4)||!buf((void**)&p->p4,&p->p4cap,hb))return 0;
    if(replicated){mv_f32<<<256,256,0,p->stream>>>((float*)peer_gate->w,peer_input->data,p->dy,256,H);
        if(!ok(cudaMemsetAsync(p->expert_ids,0,K*sizeof(int),p->stream),"EP2 peer ids clear"))return 0;
        if(peer->hash)load_hash_ids<<<1,K,0,p->stream>>>(p->expert_ids,peer->hash,p->decode_state,peer->topk);
        if(peer->hash)route_fixed6<<<1,K,0,p->stream>>>(p->expert_ids,p->expert_weights,p->dy,scale);
        else route_top6<<<1,32,0,p->stream>>>(p->expert_ids,p->expert_weights,peer_bias?(float*)peer_bias->w:nullptr,p->dy,0,scale);
        if(!ok(cudaMemcpyAsync(p->dx,peer_input->data,hb,cudaMemcpyDeviceToDevice,p->stream),"EP2 peer local input"))return 0;}
#ifdef COLI_DSV4_NCCL
    if(!replicated&&(!c->ep_comm||!p->ep_comm||!ok(cudaSetDevice(c->id),"select EP2 broadcast root")||!nok(ncclGroupStart(),"EP2 broadcast group start")||
       !nok(ncclBroadcast(c->dx,c->dx,H,ncclFloat,0,c->ep_comm,c->stream),"EP2 input root")||
       !nok(ncclBroadcast(c->expert_ids,c->expert_ids,K,ncclInt,0,c->ep_comm,c->stream),"EP2 ids root")||
       !nok(ncclBroadcast(c->expert_weights,c->expert_weights,K,ncclFloat,0,c->ep_comm,c->stream),"EP2 weights root")||
       !ok(cudaSetDevice(p->id),"select EP2 broadcast peer")||
       !nok(ncclBroadcast(p->dx,p->dx,H,ncclFloat,0,p->ep_comm,p->stream),"EP2 input peer")||
       !nok(ncclBroadcast(p->expert_ids,p->expert_ids,K,ncclInt,0,p->ep_comm,p->stream),"EP2 ids peer")||
       !nok(ncclBroadcast(p->expert_weights,p->expert_weights,K,ncclFloat,0,p->ep_comm,p->stream),"EP2 weights peer")||
       !nok(ncclGroupEnd(),"EP2 broadcast group end")))return 0;
#else
    if(!replicated&&(!ok(cudaEventRecord(c->ready,c->stream),"EP2 route ready")||!ok(cudaStreamWaitEvent(p->stream,c->ready,0),"EP2 peer route wait")||
       !ok(cudaMemcpyPeerAsync(p->dx,p->id,c->dx,c->id,hb,p->stream),"EP2 input transfer")||
       !ok(cudaMemcpyPeerAsync(p->expert_ids,p->id,c->expert_ids,c->id,K*sizeof(int),p->stream),"EP2 ids transfer")||
       !ok(cudaMemcpyPeerAsync(p->expert_weights,p->id,c->expert_weights,c->id,K*sizeof(float),p->stream),"EP2 weights transfer")))return 0;
#endif
    int contiguous=getenv("DSV4_CUDA_MOE_CONTIGUOUS")&&atoi(getenv("DSV4_CUDA_MOE_CONTIGUOUS"));
    if(moe_tp2){sort_routes6<<<1,1,0,p->stream>>>(p->expert_ids,p->expert_weights);if(!dg_moe_tp2(p,peer,p->expert_weights,limit,peer_output->data)){fprintf(stderr,"[DSV4 TP2] peer routed failed\n");return 0;}}
    else{ep2_localize<<<1,6,0,p->stream>>>(p->expert_ids,p->expert_weights,128);
        if(!(contiguous?dg_moe_contiguous_fast<128>(p,peer,p->expert_weights,limit,peer_output->data,0):dg_moe<128>(p,peer,p->expert_weights,limit,peer_output->data))){fprintf(stderr,"[DSV4 EP2] peer routed failed\n");return 0;}}
    int shared_tp=peer->sg!=nullptr;
    if(shared_tp){int PJ=peer->sg->O;if(!dg_attention_projection(p,peer->sg,p->dx,p->p1)||
       !dg_attention_projection(p,peer->su,p->dx,p->p2)){fprintf(stderr,"[DSV4 EP2] peer shared up failed\n");return 0;}
        expert_act<<<(PJ+255)/256,256,0,p->stream>>>(p->p3,p->p1,p->p2,1.f,limit,PJ);
        if(!dg_attention_projection(p,peer->sd,p->p3,p->p4)){fprintf(stderr,"[DSV4 EP2] peer shared down failed\n");return 0;}
        moe_combine<<<(H+255)/256,256,0,p->stream>>>(peer_output->data,p->p4,H);}
    if(!ok(cudaSetDevice(c->id),"restore EP2 primary"))return 0;
    if(moe_tp2){sort_routes6<<<1,1,0,c->stream>>>(c->expert_ids,c->expert_weights);if(!dg_moe_tp2(c,local,c->expert_weights,limit,output->data)){fprintf(stderr,"[DSV4 TP2] primary routed failed\n");return 0;}}
    else{ep2_localize<<<1,6,0,c->stream>>>(c->expert_ids,c->expert_weights,0);
        if(!(contiguous?dg_moe_contiguous_fast<128>(c,local,c->expert_weights,limit,output->data,0):dg_moe<128>(c,local,c->expert_weights,limit,output->data))){fprintf(stderr,"[DSV4 EP2] primary routed failed\n");return 0;}}
    int dense=getenv("DSV4_CUDA_DENSE_DG")&&atoi(getenv("DSV4_CUDA_DENSE_DG")),LJ=local->sg?local->sg->O:0;
    if(shared_tp){if(!LJ||!dg_attention_projection(c,local->sg,c->dx,c->p1)||!dg_attention_projection(c,local->su,c->dx,c->p2)){fprintf(stderr,"[DSV4 EP2] primary shared up failed\n");return 0;}}
    else if(dense){if(!dg_dense_2048x4096(c,local->sg,c->dx,c->p1)||!dg_dense_2048x4096(c,local->su,c->dx,c->p2))return 0;}
    else{fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);run_mv<8>((uint8_t*)local->sg->w,local->sg->scale,c->dx,c->p1,LJ,H,1,c->stream);run_mv<8>((uint8_t*)local->su->w,local->su->scale,c->dx,c->p2,LJ,H,1,c->stream);}
    expert_act<<<(LJ+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,LJ);
    if(shared_tp){if(!dg_attention_projection(c,local->sd,c->p3,c->p4)){fprintf(stderr,"[DSV4 EP2] primary shared down failed\n");return 0;}}
    else if(dense){if(!dg_dense_4096x2048(c,local->sd,c->p3,c->p4))return 0;}
    else{fp8_sim<<<(LJ+127)/128,128,0,c->stream>>>(c->p3,1,LJ);run_mv<8>((uint8_t*)local->sd->w,local->sd->scale,c->p3,c->p4,H,LJ,1,c->stream);}
    if(shared_tp)moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
#ifdef COLI_DSV4_NCCL
    if(!ok(cudaSetDevice(c->id),"select EP2 reduce root")||!nok(ncclGroupStart(),"EP2 reduce group start")||
       !nok(ncclAllReduce(output->data,output->data,H,ncclFloat,ncclSum,c->ep_comm,c->stream),"EP2 primary reduce")||
       !ok(cudaSetDevice(p->id),"select EP2 reduce peer")||
       !nok(ncclAllReduce(peer_output->data,peer_output->data,H,ncclFloat,ncclSum,p->ep_comm,p->stream),"EP2 peer reduce")||
       !nok(ncclGroupEnd(),"EP2 reduce group end"))return 0;
    if(!ok(cudaSetDevice(c->id),"restore EP2 combine device"))return 0;
    if(!shared_tp)moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
#else
    if(!ok(cudaEventRecord(p->ready,p->stream),"EP2 peer compute ready")||!ok(cudaStreamWaitEvent(c->stream,p->ready,0),"EP2 peer compute wait")||
       !ok(cudaMemcpyPeerAsync(c->aux1,c->id,peer_output->data,p->id,hb,c->stream),"EP2 result transfer"))return 0;
    add_peer<<<(H+255)/256,256,0,c->stream>>>(output->data,c->aux1,H);
    if(!shared_tp)moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
#endif
    return ok(cudaGetLastError(),"EP2 routed MoE launch");
#endif
}

extern "C" int dsv4_cuda_qkv(Dsv4CudaTensor *qa,Dsv4CudaTensor *qn,Dsv4CudaTensor *qb,Dsv4CudaTensor *kv,float eps,float *qo,float *ko,const float *x){
    Dev*c=qa?ctx(qa->device):nullptr;if(!c||!qn||!qb||!kv||qa->fmt!=8||qn->fmt!=16||qb->fmt!=8||kv->fmt!=8||
       qn->device!=qa->device||qb->device!=qa->device||kv->device!=qa->device||!ok(cudaSetDevice(qa->device),"select qkv device"))return 0;
    int H=qa->I,R=qa->O,Q=qb->O,K=kv->O;if(qb->I!=R||kv->I!=H||qn->I!=R)return 0;
    size_t hb=(size_t)H*4,rb=(size_t)R*4,qbbytes=(size_t)Q*4,kb=(size_t)K*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->p1,&c->p1cap,rb)||
       !buf((void**)&c->p2,&c->p2cap,qbbytes)||!buf((void**)&c->p3,&c->p3cap,kb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,hb,cudaMemcpyHostToDevice,c->stream),"qkv activation upload"))return 0;
    run_mv<8>((uint8_t*)qa->w,qa->scale,c->dx,c->p1,R,H,1,c->stream);bf16_round<<<(R+255)/256,256,0,c->stream>>>(c->p1,R);
    rmsnorm_seq<<<1,256,0,c->stream>>>(c->p1,(__nv_bfloat16*)qn->w,R,eps);fp8_sim<<<(R+127)/128,128,0,c->stream>>>(c->p1,1,R);
    run_mv<8>((uint8_t*)qb->w,qb->scale,c->p1,c->p2,Q,R,1,c->stream);bf16_round<<<(Q+255)/256,256,0,c->stream>>>(c->p2,Q);
    run_mv<8>((uint8_t*)kv->w,kv->scale,c->dx,c->p3,K,H,1,c->stream);bf16_round<<<(K+255)/256,256,0,c->stream>>>(c->p3,K);
    return ok(cudaGetLastError(),"qkv launch")&&ok(cudaMemcpyAsync(qo,c->p2,qbbytes,cudaMemcpyDeviceToHost,c->stream),"q download")&&
           ok(cudaMemcpyAsync(ko,c->p3,kb,cudaMemcpyDeviceToHost,c->stream),"kv download")&&ok(cudaStreamSynchronize(c->stream),"qkv sync");
}
extern "C" int dsv4_cuda_wo(Dsv4CudaTensor *wa,Dsv4CudaTensor *wb,int groups,float *out,const float *context){
    Dev*c=wa?ctx(wa->device):nullptr;if(!c||!wb||wa->fmt!=9||wb->fmt!=8||groups<1||wa->O%groups||
       wb->device!=wa->device||wb->I!=wa->O||!ok(cudaSetDevice(wa->device),"select wo device"))return 0;
    int C=wa->I*groups,R=wa->O,H=wb->O;size_t cb=(size_t)C*4,rb=(size_t)R*4,hb=(size_t)H*4;
    if(!buf((void**)&c->dx,&c->xcap,cb)||!buf((void**)&c->p1,&c->p1cap,rb)||!buf((void**)&c->p2,&c->p2cap,hb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,context,cb,cudaMemcpyHostToDevice,c->stream),"wo context upload"))return 0;
    run_mv<9>((uint8_t*)wa->w,wa->scale,c->dx,c->p1,R,wa->I,groups,c->stream);bf16_round<<<(R+255)/256,256,0,c->stream>>>(c->p1,R);
    fp8_sim<<<(R+127)/128,128,0,c->stream>>>(c->p1,1,R);run_mv<8>((uint8_t*)wb->w,wb->scale,c->p1,c->p2,H,R,1,c->stream);
    bf16_round<<<(H+255)/256,256,0,c->stream>>>(c->p2,H);
    return ok(cudaGetLastError(),"wo launch")&&ok(cudaMemcpyAsync(out,c->p2,hb,cudaMemcpyDeviceToHost,c->stream),"wo result download")&&ok(cudaStreamSynchronize(c->stream),"wo sync");
}
extern "C" void dsv4_cuda_tensor_free(Dsv4CudaTensor*t){if(!t)return;cudaSetDevice(t->device);if(t->own_w&&t->w)cudaFree(t->w);if(t->bf16_cache)cudaFree(t->bf16_cache);if(t->own_scale&&t->scale)cudaFree(t->scale);if(t->own_dg_scale&&t->dg_scale)cudaFree(t->dg_scale);free(t);}
extern "C" long long dsv4_cuda_tensor_bytes(const Dsv4CudaTensor*t){return t?t->bytes:0;}
extern "C" int dsv4_cuda_tensor_device(const Dsv4CudaTensor*t){return t?t->device:-1;}
