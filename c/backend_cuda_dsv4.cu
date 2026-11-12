#include "backend_cuda_dsv4.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "backend_cuda_dsv4_quant.cuh"
#if !defined(COLI_DSV4_NO_TC)
#include <cublasLt.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <algorithm>
#include <cfloat>
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
 * the DeepGEMM sm_120a flavour needs 12.8+ anyway). Without it nothing here
 * touches cuBLASLt, so the Windows DLLs the release ships (cuda-dsv4-dll
 * DSV4_TC=0) need no cublasLt64_12.dll beside them (#2047). */
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
#if COLI_DSV4_TC
struct TcPlan {int O,I,ok;cublasLtMatmulDesc_t op;cublasLtMatrixLayout_t a,b,c,d;cublasLtMatmulHeuristicResult_t algo;};
#endif
struct Dev { int id,sms,sm; cudaStream_t stream,aux; cudaEvent_t ready,fork,join;float *dx,*dy,*p1,*p2,*p3,*p4,*aux1,*aux2,*aux3; size_t xcap,ycap,p1cap,p2cap,p3cap,p4cap,aux1cap,aux2cap,aux3cap; float *hx,*hy; void *pjh; size_t pjhcap; float *ssc; size_t ssccap; void *chp; size_t chpcap; void *dchp; size_t dchpcap; void *lfh; size_t lfhcap; void *lfd; size_t lfdcap; void *lfhp; size_t lfhpcap;
#ifdef COLI_DSV4_NCCL
    ncclComm_t ep_comm;
#endif
    int *decode_state;
    MvDesc *mvdesc;size_t mvdesccap;float *expert_weights;size_t expert_weightscap;int *expert_ids;size_t expert_idcap;
    int *emaj;size_t emajcap;float *emv1,*emv2,*emv4;size_t emv1cap,emv2cap,emv4cap;unsigned long long *emcount; /* expert-major MoE */
    float *mmv;size_t mmvcap;unsigned long long *mmvcount; /* DSV4_MM_WARP_VERIFY */
    float *v4fact,*v4fparts;unsigned char *v4fbar; /* fused decode MoE (DSV4_CUDA_MOE_FUSED6) */
    float *sa_score,*sa_w,*sa_den,*sa_out;long long *sa_rows; /* V4_SPARSE_FAST */
    size_t sa_score_cap,sa_w_cap,sa_den_cap,sa_out_cap,sa_rows_cap;
#if COLI_DSV4_TC
    cublasLtHandle_t lt;TcPlan plans[2];
#endif
    void *tc_workspace;uint8_t *tcw,*tcs,*tcx,*tcxs;size_t tcwcap,tcscap,tcxcap,tcxscap;
#ifdef COLI_DSV4_DEEPGEMM
    uint8_t *dga1,*dga2;int *dgsfa1,*dgsfa2,*dglayout,*dgrowmap;__nv_bfloat16 *dgmm1,*dgmm2;float *dgref,*dgdensews;size_t dga1cap,dga2cap,dgsfa1cap,dgsfa2cap,dglayoutcap,dgrowmapcap,dgmm1cap,dgmm2cap,dgrefcap,dgdensewscap;
#endif
};
static Dev g[16];static int ng;
/* g_peer[a][b]: device a's kernels may read device b's memory (peer access
 * enabled by dsv4_cuda_init, e.g. over NVLink). g_peer_lost[d]: a peer-side
 * failure in the decode MoE latched device d off (dsv4_cuda_peer_healthy);
 * after the call that latched it (whose recompute on the primary may still
 * read its mirrors in place), no kernel reads its memory from another device
 * again. The latch outlives dsv4_cuda_shutdown on purpose: a sticky fault
 * stays in the device's context for the life of the process. */
static unsigned char g_peer[16][16],g_peer_lost[16];
static int g_peer_best_effort;   /* layer placement: P2P is optional */
static bool dev_lost(int d){return d>=0&&d<16&&g_peer_lost[d];}
static int g_cuda_lost;   /* see cuda_lost */
static bool dev_ok(int a,int b){return !g_cuda_lost&&(a==b||(a>=0&&a<16&&b>=0&&b<16&&g_peer[a][b]&&!g_peer_lost[a]&&!g_peer_lost[b]));}
/* A sticky error (a kernel fault: launch failure, illegal address, trap, ...)
 * leaves the CUDA context unusable for the rest of the process, and with peer
 * access between the GPUs a fault on one takes the others with it: measured
 * on two RTX 3090s with NVLink, a __trap() on the peer made device 0's next
 * calls fail too, whether device 0's stream or the host waited for the peer
 * (the F18 probe). g_cuda_lost latches on the first one: from then on ctx()
 * answers no device, so every entry point refuses without touching CUDA and
 * the engine takes its host paths (dsv4_cuda_lost) until the process ends. */
static bool sticky(cudaError_t e){switch(e){case cudaErrorIllegalAddress:case cudaErrorLaunchFailure:case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:case cudaErrorMisalignedAddress:case cudaErrorInvalidAddressSpace:case cudaErrorInvalidPc:
    case cudaErrorAssert:case cudaErrorLaunchTimeout:case cudaErrorECCUncorrectable:return true;default:return false;}}
static void cuda_lost(const char*s,cudaError_t e){if(g_cuda_lost)return;g_cuda_lost=1;
    fprintf(stderr,"[DSV4 CUDA] %s: %s is a sticky fault: CUDA is lost for this process, on every device; the GPU paths "
                   "stay off and the engine runs on the host until the server restarts\n",s,cudaGetErrorString(e));}
extern "C" int dsv4_cuda_lost(void){return g_cuda_lost;}
static Dev *ctx(int d){if(g_cuda_lost)return nullptr;for(int i=0;i<ng;i++)if(g[i].id==d)return g+i;return nullptr;}
static int ok(cudaError_t e,const char *s){if(e==cudaSuccess)return 1;fprintf(stderr,"[DSV4 CUDA] %s: %s\n",s,cudaGetErrorString(e));if(sticky(e))cuda_lost(s,e);return 0;}
#if COLI_DSV4_TC
static int bok(cublasStatus_t e,const char*s){if(e==CUBLAS_STATUS_SUCCESS)return 1;fprintf(stderr,"[DSV4 CUDA] %s: cuBLAS status %d\n",s,(int)e);return 0;}
#endif
#ifdef COLI_DSV4_NCCL
static int nok(ncclResult_t e,const char*s){if(e==ncclSuccess)return 1;fprintf(stderr,"[DSV4 NCCL] %s: %s\n",s,ncclGetErrorString(e));return 0;}
#endif
/* For cudaMalloc results: a failed allocation is recoverable (callers fall
 * back or skip work), but the runtime also keeps it as its last error, and the
 * next launch check (cudaGetLastError) would report that OOM against a kernel
 * that never touched the missing buffer. Read it here. */
static int alloc_ok(cudaError_t e,const char *s){if(e==cudaSuccess)return 1;(void)cudaGetLastError();return ok(e,s);}
static int buf(void **p,size_t *cap,size_t n){if(*cap>=n)return 1;if(*p)cudaFree(*p);*p=nullptr;*cap=0;if(!alloc_ok(cudaMalloc(p,n),"buffer allocation"))return 0;*cap=n;return 1;}
/* V4_LAYER_FUSED (0), the engine's switch for a decode token's attention on
 * the device: the kernels below that the existing entry points would also
 * pick up (the warp matvecs, the cached sparse attention's second form, the
 * coalesced indexer reads) default to it, so with it unset every existing
 * call runs the kernels it ran before. Each has its own variable, which
 * wins when set. */
static int attn_device_on(void){static int on=-1;if(on<0){const char*e=getenv("V4_LAYER_FUSED");on=e&&*e&&atoi(e)!=0;}return on;}


__constant__ float e8_table[256];
__global__ void set_decode_state(int*state,int token,int position){if(!threadIdx.x){state[0]=token;state[1]=position;}}
__device__ float e8(uint8_t b){return e8_table[b];}
__device__ __forceinline__ float e4m3_exact(unsigned b){
    unsigned sg=b>>7,e=(b>>3)&15u,m=b&7u;float v;
    if(e==0)v=(float)m*0.001953125f;
    else if(e==15u&&m==7u)v=NAN;
    else v=__uint_as_float(((e+120u)<<23)|(m<<20));
    return sg?-v:v;}
__global__ void mm_bf16_batch(const __nv_bfloat16*w,const float*x,float*y,int O,int I,int T);
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
/* DSV4_MV_WARP (1): the one-token fp8 matvec as one warp per row. mv<F>
 * has thread j of its 256-thread block sum partial j (i = j, j+256, ... in
 * order) and then fold the 256 partials as a tree (128, 64, ..., 1). Here
 * lane l holds partials 8l..8l+7: eight consecutive bytes of the row, one
 * 8-byte load per lane and 256 contiguous bytes per warp per step, x as two
 * float4 (mv<F> loads one byte per thread). The tree's 128/64/32/16/8 steps
 * are shuffles over lanes 16/8/4/2/1 apart, its 4/2/1 steps in-lane adds:
 * every output has mv<F>'s operands in mv<F>'s order (bit-identical;
 * DSV4_MV_WARP_VERIFY runs both and counts differing words). I % 256 == 0
 * and 8-byte aligned rows. Eight rows per 256-thread block; e4m3 through a
 * shared table of exactly e4m3()'s values. */
/* DSV4_MV_WARP_UNROLL (4): U chunks of 256 columns loaded before their
 * FMAs (the loop with a runtime trip count issued one chunk's loads per
 * step, a dependent load-then-add per 256 columns: wq_b's 1024-wide rows
 * at 355 GB/s where wo_a's 4096-wide ran 680). The partials' order is
 * unchanged (chunk c, then c+256, ...): bitwise. 1 = a chunk per step. */
__device__ __forceinline__ float bf16_round_dev(float v);
template<int F,int U,bool BITS> __global__ void __launch_bounds__(256) mv_warp(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups,int rnd){
    __shared__ float lut[256];
    if(!BITS){lut[threadIdx.x]=e4m3((uint8_t)threadIdx.x);__syncthreads();}
    int lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);if(o>=O)return;
    const float *xi=x+(long long)(o/(O/groups))*I;
    const uint8_t *wr=w+(long long)o*I;
    const uint8_t *scr=sc+(long long)(o/128)*((I+127)/128);
    float p[8];
    #pragma unroll
    for(int k=0;k<8;k++)p[k]=0.f;
    int c=0;
    if constexpr(U>1){
        for(;c+U*256<=I;c+=U*256){
            uint2 wv[U];float4 xa[U],xb[U];float es[U];
            #pragma unroll
            for(int u=0;u<U;u++){int i0=c+u*256+8*lane;
                wv[u]=*(const uint2*)(wr+i0);xa[u]=*(const float4*)(xi+i0);xb[u]=*(const float4*)(xi+i0+4);es[u]=e8(scr[i0/128]);}
            #pragma unroll
            for(int u=0;u<U;u++){
                unsigned b[8]={wv[u].x&255u,(wv[u].x>>8)&255u,(wv[u].x>>16)&255u,wv[u].x>>24,wv[u].y&255u,(wv[u].y>>8)&255u,(wv[u].y>>16)&255u,wv[u].y>>24};
                float xs[8]={xa[u].x,xa[u].y,xa[u].z,xa[u].w,xb[u].x,xb[u].y,xb[u].z,xb[u].w};
                #pragma unroll
                for(int k=0;k<8;k++){float v=(BITS?e4m3_exact(b[k]):lut[b[k]])*es[u];if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));p[k]+=xs[k]*v;}
            }
        }
    }
    for(;c<I;c+=256){
        int i0=c+8*lane;
        uint2 wv=*(const uint2*)(wr+i0);
        float4 xa=*(const float4*)(xi+i0),xb=*(const float4*)(xi+i0+4);
        float es=e8(scr[i0/128]);
        unsigned b[8]={wv.x&255u,(wv.x>>8)&255u,(wv.x>>16)&255u,wv.x>>24,wv.y&255u,(wv.y>>8)&255u,(wv.y>>16)&255u,wv.y>>24};
        float xs[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
        #pragma unroll
        for(int k=0;k<8;k++){float v=(BITS?e4m3_exact(b[k]):lut[b[k]])*es;if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));p[k]+=xs[k]*v;}
    }
    #pragma unroll
    for(int k=0;k<8;k++){
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],16);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],8);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],4);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],2);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],1);
    }
    if(!lane){float v0=p[0]+p[4],v1=p[1]+p[5],v2=p[2]+p[6],v3=p[3]+p[7];v0=v0+v2;v1=v1+v3;float r=v0+v1;y[o]=rnd?bf16_round_dev(r):r;}
}
/* The bf16 one (mm_bf16_batch with a batch of one, mv_bf16: the same
 * partials and tree): lane l's eight bf16 as one 16-byte load. */
template<int U> __global__ void __launch_bounds__(256) mv_bf16_warp(const __nv_bfloat16 *w,const float *x,float *y,int O,int I){
    int lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);if(o>=O)return;
    const __nv_bfloat16 *wr=w+(long long)o*I;
    float p[8];
    #pragma unroll
    for(int k=0;k<8;k++)p[k]=0.f;
    int c=0;
    if constexpr(U>1){
        for(;c+U*256<=I;c+=U*256){
            uint4 wv[U];float4 xa[U],xb[U];
            #pragma unroll
            for(int u=0;u<U;u++){int i0=c+u*256+8*lane;
                wv[u]=*(const uint4*)(wr+i0);xa[u]=*(const float4*)(x+i0);xb[u]=*(const float4*)(x+i0+4);}
            #pragma unroll
            for(int u=0;u<U;u++){
                unsigned uu[4]={wv[u].x,wv[u].y,wv[u].z,wv[u].w};
                float xs[8]={xa[u].x,xa[u].y,xa[u].z,xa[u].w,xb[u].x,xb[u].y,xb[u].z,xb[u].w};
                #pragma unroll
                for(int k=0;k<8;k++){float v=__uint_as_float((k&1)?(uu[k>>1]&0xffff0000u):(uu[k>>1]<<16));p[k]+=xs[k]*v;}
            }
        }
    }
    for(;c<I;c+=256){
        int i0=c+8*lane;
        uint4 wv=*(const uint4*)(wr+i0);
        float4 xa=*(const float4*)(x+i0),xb=*(const float4*)(x+i0+4);
        unsigned u[4]={wv.x,wv.y,wv.z,wv.w};
        float xs[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
        #pragma unroll
        for(int k=0;k<8;k++){float v=__uint_as_float((k&1)?(u[k>>1]&0xffff0000u):(u[k>>1]<<16));p[k]+=xs[k]*v;}
    }
    #pragma unroll
    for(int k=0;k<8;k++){
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],16);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],8);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],4);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],2);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],1);
    }
    if(!lane){float v0=p[0]+p[4],v1=p[1]+p[5],v2=p[2]+p[6],v3=p[3]+p[7];v0=v0+v2;v1=v1+v3;y[o]=v0+v1;}
}
static int mv_warp_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_MV_WARP");on=e&&*e?*e!='0':attn_device_on();}return on;}
static int mv_warp_verify(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_MV_WARP_VERIFY");on=e&&*e&&*e!='0';}return on;}
static int mv_warp_unroll(void){static int u=-1;if(u<0){const char*e=getenv("DSV4_MV_WARP_UNROLL");u=e&&*e?atoi(e):1;if(u!=1&&u!=2&&u!=4)u=4;}return u;}
static int mv_warp_lut(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_MV_WARP_LUT");on=!(e&&*e=='0');}return on;}   /* 1: the table; 0 = e4m3_exact, measured +0.025 ms per fused layer */
template<int F,int U> static void launch_mv_warp_u(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups,int rnd,cudaStream_t st){
    if(mv_warp_lut())mv_warp<F,U,false><<<(O+7)/8,256,0,st>>>(w,sc,x,y,O,I,groups,rnd);
    else mv_warp<F,U,true><<<(O+7)/8,256,0,st>>>(w,sc,x,y,O,I,groups,rnd);
}
static void launch_mv_bf16_warp(const __nv_bfloat16 *w,const float *x,float *y,int O,int I,cudaStream_t st){
    int u=mv_warp_unroll();
    if(u==4)mv_bf16_warp<4><<<(O+7)/8,256,0,st>>>(w,x,y,O,I);
    else if(u==2)mv_bf16_warp<2><<<(O+7)/8,256,0,st>>>(w,x,y,O,I);
    else mv_bf16_warp<1><<<(O+7)/8,256,0,st>>>(w,x,y,O,I);
}
template<int F> static void launch_mv_warp(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups,cudaStream_t st){
    int u=mv_warp_unroll();
    if(u==4)launch_mv_warp_u<F,4>(w,sc,x,y,O,I,groups,0,st);
    else if(u==2)launch_mv_warp_u<F,2>(w,sc,x,y,O,I,groups,0,st);
    else launch_mv_warp_u<F,1>(w,sc,x,y,O,I,groups,0,st);
}
static bool mv_warp_fits(const void*w,const float*x,int I,int walign){return I>0&&I%256==0&&!((uintptr_t)x&15)&&!((uintptr_t)w&(walign-1));}
__global__ void mv_warp_diff_kernel(const float*a,const float*b,int n,unsigned*cnt){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&__float_as_uint(a[i])!=__float_as_uint(b[i]))atomicAdd(cnt,1u);}
/* DSV4_MV_WARP_VERIFY: `got` (the warp kernel's output, O floats) against
 * `ref` (the block kernel's), counted on the device, reported per call and
 * in total; synchronous. */
static void mv_warp_report(const char*tag,const float*ref,const float*got,int O,int I,cudaStream_t st){
    static unsigned*dc=nullptr;static unsigned long long calls=0,bad_calls=0,words=0;
    if(!dc){if(cudaMalloc((void**)&dc,sizeof(unsigned))!=cudaSuccess){dc=nullptr;return;}}
    cudaMemsetAsync(dc,0,sizeof(unsigned),st);
    mv_warp_diff_kernel<<<(O+255)/256,256,0,st>>>(ref,got,O,dc);
    unsigned n=0;cudaMemcpyAsync(&n,dc,sizeof(n),cudaMemcpyDeviceToHost,st);cudaStreamSynchronize(st);
    calls++;words+=n;if(n)bad_calls++;
    fprintf(stderr,"mvwverify %s O=%d I=%d mismatches=%u (calls=%llu bad=%llu words=%llu)\n",tag,O,I,n,calls,bad_calls,words);
}
static float*mv_warp_scratch(size_t n){static float*p=nullptr;static size_t cap=0;if(cap<n){if(p)cudaFree(p);p=nullptr;cap=0;if(cudaMalloc((void**)&p,n*4)!=cudaSuccess){p=nullptr;return nullptr;}cap=n;}return p;}
/* bf16 rows for one token: the warp kernel when it fits, else
 * mm_bf16_batch with a batch of one (what the callers launched). */
static void launch_bf16_one(const __nv_bfloat16*w,const float*x,float*y,int O,int I,cudaStream_t st){
    if(mv_warp_on()&&mv_warp_fits(w,x,I,16)){
        if(mv_warp_verify()){float*t=mv_warp_scratch((size_t)O);if(t){
            mm_bf16_batch<<<dim3(O,1),256,0,st>>>(w,x,y,O,I,1);
            launch_mv_bf16_warp(w,x,t,O,I,st);
            mv_warp_report("bf16",y,t,O,I,st);return;}}
        launch_mv_bf16_warp(w,x,y,O,I,st);
    }else mm_bf16_batch<<<dim3(O,1),256,0,st>>>(w,x,y,O,I,1);
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
#ifndef COLI_DSV4_DEEPGEMM /* generic batched MoE only (DeepGEMM builds use dg_moe_batch_contiguous) */
/* Expert-major fp4 matvec for the generic batched MoE (DSV4_MOE_EXPERT_MAJOR).
 * mv_fp4_grouped runs one matvec per route, so an expert picked by k tokens of
 * the chunk has its weights streamed k times, and each 256-thread block spends
 * most of its time in the shared-memory tree. Here one WARP owns one row of
 * one bank expert and walks every route that picked it, TB at a time. Lane l
 * holds the tree's partials j = l + 32q (q = 0..7): partial j sums
 * i = j, j+256, ... in order, exactly like thread j of mv_fp4_grouped; the
 * tree's 128/64/32 steps pair partials inside a lane and 16..1 are shuffles,
 * so every output has the same operands in the same order: bit-identical
 * (DSV4_MOE_EXPERT_MAJOR_VERIFY checks it at run time). No shared memory, no
 * barriers. Rows below `split` go to y0, the rest to y1 (FC1 = gate, up). */
template<int TB> __global__ void __launch_bounds__(256) mv_fp4_expert_major(const uint8_t*bank_w,const uint8_t*bank_s,size_t wstride,size_t sstride,
        const int*experts,const int*offsets,const int*routes,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I){
    int slot=blockIdx.y,lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);
    if(o>=O)return;
    int e=experts[slot],begin=offsets[slot],end=offsets[slot+1];
    const uint8_t*w=bank_w+(size_t)e*wstride+(size_t)o*(I/2);const uint8_t*sc=bank_s+(size_t)e*sstride+(size_t)o*(I/32);
    for(int g=begin;g<end;g+=TB){
        int n=end-g<TB?end-g:TB;const float*xs[TB];float p[TB][8];
        #pragma unroll
        for(int t=0;t<TB;t++){xs[t]=t<n?x+(long long)(routes[g+t]/xdiv)*xstride:x;
            #pragma unroll
            for(int q=0;q<8;q++)p[t][q]=0.f;}
        for(int c0=0;c0<I/32;c0+=8){
            #pragma unroll
            for(int q=0;q<8;q++){int i=lane+32*(c0+q);
                float xv[TB];
                #pragma unroll
                for(int t=0;t<TB;t++)xv[t]=t<n?xs[t][i]:0.f;
                uint8_t pk=w[i>>1],qq=(i&1)?pk>>4:pk&15;float f=f4(qq),es=e8(sc[i/32]);
                #pragma unroll
                for(int t=0;t<TB;t++)if(t<n)p[t][q]+=xv[t]*f*es;
            }
        }
        #pragma unroll
        for(int t=0;t<TB;t++){
            float v0=p[t][0]+p[t][4],v1=p[t][1]+p[t][5],v2=p[t][2]+p[t][6],v3=p[t][3]+p[t][7];
            v0=v0+v2;v1=v1+v3;float v=v0+v1;
            #pragma unroll
            for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
            if(!lane&&t<n){long long route=routes[g+t];
                if(o<split)y0[route*ystride+o]=v;else y1[route*ystride+(o-split)]=v;}
        }
    }
}
static void run_fp4_expert_major(int tb,const uint8_t*w,const uint8_t*sc,size_t wstride,size_t sstride,
        const int*ex,const int*off,const int*rt,int nex,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I,cudaStream_t stream){
    dim3 grid((O+7)/8,nex);
    if(tb==3)mv_fp4_expert_major<3><<<grid,256,0,stream>>>(w,sc,wstride,sstride,ex,off,rt,x,xstride,xdiv,y0,y1,split,ystride,O,I);
    else if(tb==2)mv_fp4_expert_major<2><<<grid,256,0,stream>>>(w,sc,wstride,sstride,ex,off,rt,x,xstride,xdiv,y0,y1,split,ystride,O,I);
    else if(tb==6)mv_fp4_expert_major<6><<<grid,256,0,stream>>>(w,sc,wstride,sstride,ex,off,rt,x,xstride,xdiv,y0,y1,split,ystride,O,I);
    else mv_fp4_expert_major<4><<<grid,256,0,stream>>>(w,sc,wstride,sstride,ex,off,rt,x,xstride,xdiv,y0,y1,split,ystride,O,I);
}
__global__ void count_bit_mismatch(const float*a,const float*b,long long n,unsigned long long*count){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&__float_as_uint(a[i])!=__float_as_uint(b[i]))atomicAdd(count,1ULL);}
#endif
template<int F> static void run_mv(const uint8_t*w,const uint8_t*sc,const float*x,float*y,int O,int I,int groups,cudaStream_t stream){
    static int rows=-1;if(rows<0){const char *p=getenv(F==4?"DSV4_CUDA_ROWS":"DSV4_CUDA_FP8_ROWS");rows=p?atoi(p):(F==4?4:1);}
    if(rows==32)mv_rows<F,32><<<(O+31)/32,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==16)mv_rows<F,16><<<(O+15)/16,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==8)mv_rows<F,8><<<(O+7)/8,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==4)mv_rows<F,4><<<(O+3)/4,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else{
        if constexpr(F==8||F==9){
            if(mv_warp_on()&&mv_warp_fits(w,x,I,8)){
                if(mv_warp_verify()){float*t=mv_warp_scratch((size_t)O);if(t){
                    mv<F><<<O,256,0,stream>>>(w,sc,x,y,O,I,groups);
                    launch_mv_warp<F>(w,sc,x,t,O,I,groups,stream);
                    mv_warp_report(F==9?"fp8b":"fp8",y,t,O,I,stream);return;}}
                launch_mv_warp<F>(w,sc,x,y,O,I,groups,stream);return;
            }
        }
        mv<F><<<O,256,0,stream>>>(w,sc,x,y,O,I,groups);
    }
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
#ifndef COLI_DSV4_DEEPGEMM /* run_mm_batch's callers are all generic-build paths */
/* Warp-per-row mm_batch for fp8 / bf16-rounded fp8 weights (DSV4_MM_WARP).
 * One warp owns one output row and TB tokens; lane l holds the 256-wide
 * tree's partials j = l + 32q, each summing i = j, j+256, ... in mm_batch's
 * order, and the tree's 128/64/32 steps are in-lane adds, 16..1 shuffles:
 * every output has mm_batch's operands in mm_batch's order (bit-identical;
 * DSV4_MM_WARP_VERIFY counts mismatches). e4m3 decodes through a shared
 * 256-entry table holding exactly e4m3()'s values. */
template<int F,int TB> __global__ void __launch_bounds__(256) mm_batch_w(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int T){
    __shared__ float lut[256];
    lut[threadIdx.x]=e4m3((uint8_t)threadIdx.x);__syncthreads();
    int lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);if(o>=O)return;
    int tbase=blockIdx.y*TB,tcount=T-tbase<TB?T-tbase:TB;
    float p[TB][8];
    #pragma unroll
    for(int t=0;t<TB;t++)
        #pragma unroll
        for(int q=0;q<8;q++)p[t][q]=0.f;
    for(int c0=0;c0<I/32;c0+=8){
        #pragma unroll
        for(int q=0;q<8;q++){int i=lane+32*(c0+q);
            float v=lut[w[(long long)o*I+i]]*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));
            #pragma unroll
            for(int t=0;t<TB;t++)if(t<tcount)p[t][q]+=x[(long long)(tbase+t)*I+i]*v;
        }
    }
    #pragma unroll
    for(int t=0;t<TB;t++){
        float v0=p[t][0]+p[t][4],v1=p[t][1]+p[t][5],v2=p[t][2]+p[t][6],v3=p[t][3]+p[t][7];
        v0=v0+v2;v1=v1+v3;float v=v0+v1;
        #pragma unroll
        for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
        if(!lane&&t<tcount)y[(long long)(tbase+t)*O+o]=v;
    }
}
static void run_mm_batch(Dsv4CudaTensor*t,const float*x,float*y,int T,cudaStream_t stream){
    /* DSV4_MM_WARP, for fp8 shapes whose I is a multiple of 256: unset, empty
     * or "auto" = warp kernel for short rows only (I <= 1024, e.g. attn.wq_b,
     * where mm_batch spends most of each block in its tree); 1 (any non-zero
     * number) = all of them; 0 = never. Kernel time against mm_batch on sm_86:
     * ~0.4-0.55x at I = 1024, ~0.8x at I = 2048 (shared w2, 4096x2048), ~1.1x
     * at I = 4096, ~1.65x at I = 8192; auto stays at I <= 1024 until I = 2048
     * is checked in the engine.
     * DSV4_MM_WARP_VERIFY=1 also runs mm_batch, keeps ITS results and counts
     * mismatching bits; over DSV4_MM_WARP=0 it checks the auto shapes (said
     * once at startup). */
    static int mw=-1,mw_env=1,mwv=0;static unsigned long long vcalls=0,vbad=0;
    if(mw<0){const char*e=getenv("DSV4_MM_WARP");mw_env=!e||!*e||!strcmp(e,"auto")?1:atoi(e)!=0?2:0;mw=mw_env;
        e=getenv("DSV4_MM_WARP_VERIFY");mwv=e&&atoi(e)!=0;
        if(mwv&&!mw){mw=1;fprintf(stderr,"[DSV4 CUDA] DSV4_MM_WARP_VERIFY=1 overrides DSV4_MM_WARP=0: the warp kernel runs beside mm_batch on the auto shapes, mm_batch's results are kept\n");}}
    if(mw&&(t->fmt==8||t->fmt==9)&&t->I%256==0&&(mw==2||t->I<=1024)){
        size_t n=(size_t)T*t->O;Dev*c=mwv?ctx(t->device):nullptr;
        /* verify buffers are per device; without them verify switches itself
         * off for good and DSV4_MM_WARP's own setting applies again (an
         * overridden 0 sends this call to mm_batch below) */
        if(mwv&&(!c||!buf((void**)&c->mmv,&c->mmvcap,n*sizeof(float))||
                 (!c->mmvcount&&!alloc_ok(cudaMalloc(&c->mmvcount,sizeof(*c->mmvcount)),"mm-warp verify counter")))){
            mwv=0;mw=mw_env;fprintf(stderr,"[DSV4 CUDA] DSV4_MM_WARP_VERIFY off: no memory for its buffers\n");}
        if(mw){
            dim3 wg((t->O+7)/8,(T+7)/8);float*dst=mwv?c->mmv:y;
            if(t->fmt==9)mm_batch_w<9,8><<<wg,256,0,stream>>>((uint8_t*)t->w,t->scale,x,dst,t->O,t->I,T);
            else mm_batch_w<8,8><<<wg,256,0,stream>>>((uint8_t*)t->w,t->scale,x,dst,t->O,t->I,T);
            if(!mwv)return;
            dim3 grid(t->O,(T+15)/16);
            if(t->fmt==9)mm_batch<9><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
            else mm_batch<8><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
            unsigned long long bad=0;cudaMemsetAsync(c->mmvcount,0,sizeof(*c->mmvcount),stream);
            count_bit_mismatch<<<(unsigned)((n+255)/256),256,0,stream>>>(y,c->mmv,(long long)n,c->mmvcount);
            cudaMemcpyAsync(&bad,c->mmvcount,sizeof(bad),cudaMemcpyDeviceToHost,stream);cudaStreamSynchronize(stream);
            vcalls++;vbad+=bad;
            if(bad||vcalls%500==0)fprintf(stderr,"mm-warp verify calls=%llu mismatches=%llu (this call %llu of %zu, %dx%d T=%d fmt=%d)\n",
                                          vcalls,vbad,bad,n,t->O,t->I,T,t->fmt);
            return;
        }
    }
    dim3 grid(t->O,(T+15)/16);
    if(t->fmt==4)mm_batch<4><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
    else if(t->fmt==9)mm_batch<9><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
    else mm_batch<8><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
}
#endif
__device__ int tc_sf_off(int outer,int inner,int sf_inner_dim){int base=((inner/4)*4+(outer/128)*sf_inner_dim)*128,o=outer&127;return base+(o%32)*16+(o/32)*4+(inner&3);}
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
__global__ void dg_replicate_rows(uint8_t*q,int*sfa,int K,int rows){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)rows*K;if(p<n){int k=p%K;q[p]=q[k];}
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
static int g_kv_device[DSV4_KV_CACHE_LAYERS];

/* A KV ring/comp buffer belongs to one device; refuse a different owner
 * instead of writing through a pointer allocated on another device. */
static int kv_device_ok(int layer, int device) {
    if ((g_kv_ring[layer] || g_kv_comp[layer]) && g_kv_device[layer] != device) {
        fprintf(stderr, "[DSV4 CUDA] kv layer %d is on device %d, not %d; refusing append\n",
                layer, g_kv_device[layer], device);
        return 0;
    }
    return 1;
}

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
    if(g_cuda_lost)return -1;
    if(!ok(cudaSetDevice(device),"select meminfo device"))return -1;
    size_t free_b=0,total_b=0;
    if(!ok(cudaMemGetInfo(&free_b,&total_b),"device meminfo"))return -1;
    return (long long)(free_b>>20);
}

/* 1 when the device shares its memory with the host (GB10 / DGX Spark, Jetson,
 * any integrated part): cudaMemGetInfo then reports the system's free memory,
 * not headroom on a card, so a VRAM reserve compared against it says nothing
 * about whether one more mirror fits (#1538). */
extern "C" int dsv4_cuda_device_unified(int device){
    int integrated=0,host_tables=0;
    if(cudaDeviceGetAttribute(&integrated,cudaDevAttrIntegrated,device)!=cudaSuccess){cudaGetLastError();integrated=0;}
    if(cudaDeviceGetAttribute(&host_tables,cudaDevAttrPageableMemoryAccessUsesHostPageTables,device)!=cudaSuccess){cudaGetLastError();host_tables=0;}
    return (integrated||host_tables)?1:0;
}

extern "C" int dsv4_cuda_kv_ring_append(int device,int layer,const float*rows,
        int start_pos,int count,int window,int dim){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!rows||start_pos<0||count<1||
       window<1||dim<1||!ok(cudaSetDevice(device),"select kv ring device"))return 0;
    if(!kv_device_ok(layer,device))return 0;
    g_kv_device[layer]=device;
    if(g_kv_ring[layer]&&g_kv_ring_rows[layer]!=window){cudaFree(g_kv_ring[layer]);g_kv_ring[layer]=NULL;}
    if(!g_kv_ring[layer]){
        if(!alloc_ok(cudaMalloc((void**)&g_kv_ring[layer],(size_t)window*dim*4),"kv ring allocation"))return 0;
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
    if(!kv_device_ok(layer,device))return 0;
    g_kv_device[layer]=device;
    int need=start_idx+count;
    if(g_kv_comp_cap[layer]<need){
        int cap=g_kv_comp_cap[layer]?g_kv_comp_cap[layer]:512;
        while(cap<need)cap*=2;
        float*grown=NULL;
        if(!alloc_ok(cudaMalloc((void**)&grown,(size_t)cap*dim*4),"kv comp allocation"))return 0;
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

/* DSV4_SATTN_V2 (1): the cached sparse attention rewritten for latency —
 * the same arithmetic in the same order (each lane's per-row dot, the
 * warp shuffle, the denominator's serial sum, the row-ordered weighted
 * sum), only the loads overlapped: four rows per warp step for the
 * scores, the selection indices and eight rows prefetched into registers
 * for the weighted sum, the exponentials by every thread with the max as
 * a block reduction. IDX: rows past the window come from the indexer's
 * selection. Measured before: 0.39 ms per 640-row layer, one thread's
 * serial softmax and a dependent index load before each row load. */
template<int IDX,int HPB> __global__ void __launch_bounds__(512) sparse_attn_cached_v2(float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,const int*sel,
        int selstride,int abs_base,int chunk_start,int window,int heads,int dim,float scale){
    int t=blockIdx.y,h0=blockIdx.x*HPB,nh=heads-h0;if(nh>HPB)nh=HPB;
    int wo=meta[3*t],wn=meta[3*t+1],cn=meta[3*t+2],total=wn+cn;
    const int*ts=IDX?sel+(long long)t*selstride:nullptr;
    const float*qb=q+((long long)t*heads+h0)*dim;
    int qo[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++)qo[k]=(k<nh?k:0)*dim;
    extern __shared__ float score[];   /* [HPB][total] */
    __shared__ float red[HPB][32];
    __shared__ float den[HPB];
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,nw=blockDim.x>>5;
    for(int item=warp*4;item<total;item+=nw*4){
        int nr=total-item;if(nr>4)nr=4;
        const float*v[4];
        #pragma unroll
        for(int r=0;r<4;r++){int it=item+(r<nr?r:0);
            v[r]=it<wn?cached_row(win,chunk,abs_base+wo+it,chunk_start,window,dim)
                      :comp+(long long)(IDX?ts[it-wn]:(it-wn))*dim;}
        float z[HPB][4];
        #pragma unroll
        for(int k=0;k<HPB;k++){
            #pragma unroll
            for(int r=0;r<4;r++)z[k][r]=0.f;}
        for(int d=lane;d<dim;d+=32){float vd[4];
            #pragma unroll
            for(int r=0;r<4;r++)vd[r]=r<nr?v[r][d]:0.f;
            #pragma unroll
            for(int k=0;k<HPB;k++){float qd=qb[qo[k]+d];
                #pragma unroll
                for(int r=0;r<4;r++)if(r<nr)z[k][r]+=qd*vd[r];}}
        #pragma unroll
        for(int k=0;k<HPB;k++){
            #pragma unroll
            for(int r=0;r<4;r++){float zz=z[k][r];
                for(int n=16;n;n>>=1)zz+=__shfl_down_sync(0xffffffff,zz,n);
                if(!lane&&r<nr&&k<nh)score[k*total+item+r]=zz*scale;}}
    }
    __syncthreads();
    float m[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++){float lm=-INFINITY;
        if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)lm=fmaxf(lm,score[k*total+i]);
        for(int n=16;n;n>>=1)lm=fmaxf(lm,__shfl_xor_sync(0xffffffff,lm,n));
        if(!lane)red[k][warp]=lm;}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<HPB;k++){float mm=red[k][0];for(int w=1;w<nw;w++)mm=fmaxf(mm,red[k][w]);m[k]=mm;}
    #pragma unroll
    for(int k=0;k<HPB;k++)if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)score[k*total+i]=expf(score[k*total+i]-m[k]);
    __syncthreads();
    if(threadIdx.x<HPB){int k=threadIdx.x;if(k<nh){float d0=expf(sink[h0+k]-m[k]);const float*sc=score+k*total;for(int i=0;i<total;i++)d0+=sc[i];den[k]=d0;}}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<HPB;k++)if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)score[k*total+i]=__bfloat162float(__float2bfloat16(score[k*total+i]));
    __syncthreads();
    float dn[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++)dn[k]=den[k];
    for(int d=threadIdx.x;d<dim;d+=blockDim.x){
        float sum[HPB];
        #pragma unroll
        for(int k=0;k<HPB;k++)sum[k]=0.f;
        int i=0;
        for(;i+16<=wn;i+=16){float v[16];
            #pragma unroll
            for(int j=0;j<16;j++)v[j]=cached_row(win,chunk,abs_base+wo+i+j,chunk_start,window,dim)[d];
            #pragma unroll
            for(int k=0;k<HPB;k++){
                #pragma unroll
                for(int j=0;j<16;j++)sum[k]+=score[k*total+i+j]*v[j];}}
        for(;i<wn;i++){float vv=cached_row(win,chunk,abs_base+wo+i,chunk_start,window,dim)[d];
            #pragma unroll
            for(int k=0;k<HPB;k++)sum[k]+=score[k*total+i]*vv;}
        i=0;
        for(;i+16<=cn;i+=16){int ix[16];float v[16];
            #pragma unroll
            for(int j=0;j<16;j++)ix[j]=IDX?ts[i+j]:i+j;
            #pragma unroll
            for(int j=0;j<16;j++)v[j]=comp[(long long)ix[j]*dim+d];
            #pragma unroll
            for(int k=0;k<HPB;k++){
                #pragma unroll
                for(int j=0;j<16;j++)sum[k]+=score[k*total+wn+i+j]*v[j];}}
        for(;i<cn;i++){int ix=IDX?ts[i]:i;float vv=comp[(long long)ix*dim+d];
            #pragma unroll
            for(int k=0;k<HPB;k++)sum[k]+=score[k*total+wn+i]*vv;}
        #pragma unroll
        for(int k=0;k<HPB;k++)if(k<nh)out[((long long)t*heads+h0+k)*dim+d]=__bfloat162float(__float2bfloat16(sum[k]/dn[k]));
    }
}
static int sattn_v2_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_SATTN_V2");on=e&&*e?*e!='0':attn_device_on();}return on;}
/* DSV4_SATTN_SPLIT (1): the one-token cached sparse attention as two
 * launches over many more blocks than the v2 kernel's 16 (four heads per
 * block of 512 threads, so a 64-head layer runs on 16 of the card's SMs:
 * 0.09 ms per layer at ~390 rows, latency). The score kernel runs the v2
 * kernel's score phase (four rows per warp, each lane's dot over its
 * columns in order, the shuffle tree, times scale) over a grid of
 * (head quads x tiles of 64 rows) and writes the scores to a device
 * buffer; the value kernel, over (head quads x slices of DSV4_SATTN_SPLIT_DS
 * columns), reads its heads' scores into shared memory and runs the v2
 * kernel's softmax (the block max, the exponentials, the serial
 * denominator with the sink first, the bf16 round) and the weighted sum
 * over the rows in order for its columns. Every output is the v2 kernel's
 * operands in its order: bitwise. 0 = the v2 kernel. */
static int sattn_split_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_SATTN_SPLIT");on=!(e&&*e=='0');}return on;}
static int sattn_split_ds(void){static int n=-1;if(n<0){const char*e=getenv("DSV4_SATTN_SPLIT_DS");n=e&&*e?atoi(e):128;if(n<32)n=32;if(n>512)n=512;n&=~31;}return n;}
template<int IDX> __global__ void __launch_bounds__(512) sattn_split_score(float*gsc,const float*q,const float*win,
        const float*chunk,const float*comp,const int*meta,const int*sel,int abs_base,int chunk_start,int window,
        int heads,int dim,float scale,int dynab){
    const int HPB=4;
    int h0=blockIdx.x*HPB,nh=heads-h0;if(nh>HPB)nh=HPB;
    int wo=meta[0],wn=meta[1],cn=meta[2],total=wn+cn;
    if(dynab){abs_base=meta[4];chunk_start=meta[5];}
    const int*ts=IDX?sel:nullptr;
    const float*qb=q+(long long)h0*dim;
    int qo[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++)qo[k]=(k<nh?k:0)*dim;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,nw=blockDim.x>>5;
    int item=blockIdx.y*nw*4+warp*4;
    if(item>=total)return;
    int nr=total-item;if(nr>4)nr=4;
    const float*v[4];
    #pragma unroll
    for(int r=0;r<4;r++){int it=item+(r<nr?r:0);
        v[r]=it<wn?cached_row(win,chunk,abs_base+wo+it,chunk_start,window,dim)
                  :comp+(long long)(IDX?ts[it-wn]:(it-wn))*dim;}
    float z[HPB][4];
    #pragma unroll
    for(int k=0;k<HPB;k++){
        #pragma unroll
        for(int r=0;r<4;r++)z[k][r]=0.f;}
    for(int d=lane;d<dim;d+=32){float vd[4];
        #pragma unroll
        for(int r=0;r<4;r++)vd[r]=r<nr?v[r][d]:0.f;
        #pragma unroll
        for(int k=0;k<HPB;k++){float qd=qb[qo[k]+d];
            #pragma unroll
            for(int r=0;r<4;r++)if(r<nr)z[k][r]+=qd*vd[r];}}
    #pragma unroll
    for(int k=0;k<HPB;k++){
        #pragma unroll
        for(int r=0;r<4;r++){float zz=z[k][r];
            for(int n=16;n;n>>=1)zz+=__shfl_down_sync(0xffffffff,zz,n);
            if(!lane&&r<nr&&k<nh)gsc[(long long)(h0+k)*total+item+r]=zz*scale;}}
}
template<int IDX> __global__ void sattn_split_value(float*out,const float*gsc,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,const int*sel,int abs_base,int chunk_start,
        int window,int heads,int dim,int dynab){
    const int HPB=4;
    int h0=blockIdx.x*HPB,nh=heads-h0;if(nh>HPB)nh=HPB;
    int wo=meta[0],wn=meta[1],cn=meta[2],total=wn+cn;
    if(dynab){abs_base=meta[4];chunk_start=meta[5];}
    const int*ts=IDX?sel:nullptr;
    extern __shared__ float score[];   /* [HPB][total] */
    __shared__ float red[HPB][32];
    __shared__ float den[HPB];
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,nw=blockDim.x>>5;
    #pragma unroll
    for(int k=0;k<HPB;k++)if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)score[k*total+i]=gsc[(long long)(h0+k)*total+i];
    __syncthreads();
    float m[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++){float lm=-INFINITY;
        if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)lm=fmaxf(lm,score[k*total+i]);
        for(int n=16;n;n>>=1)lm=fmaxf(lm,__shfl_xor_sync(0xffffffff,lm,n));
        if(!lane)red[k][warp]=lm;}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<HPB;k++){float mm=red[k][0];for(int w=1;w<nw;w++)mm=fmaxf(mm,red[k][w]);m[k]=mm;}
    #pragma unroll
    for(int k=0;k<HPB;k++)if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)score[k*total+i]=expf(score[k*total+i]-m[k]);
    __syncthreads();
    if(threadIdx.x<HPB){int k=threadIdx.x;if(k<nh){float d0=expf(sink[h0+k]-m[k]);const float*sc=score+k*total;for(int i=0;i<total;i++)d0+=sc[i];den[k]=d0;}}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<HPB;k++)if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)score[k*total+i]=__bfloat162float(__float2bfloat16(score[k*total+i]));
    __syncthreads();
    float dn[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++)dn[k]=den[k];
    int d=blockIdx.y*blockDim.x+threadIdx.x;
    if(d<dim){
        float sum[HPB];
        #pragma unroll
        for(int k=0;k<HPB;k++)sum[k]=0.f;
        int i=0;
        for(;i+16<=wn;i+=16){float v[16];
            #pragma unroll
            for(int j=0;j<16;j++)v[j]=cached_row(win,chunk,abs_base+wo+i+j,chunk_start,window,dim)[d];
            #pragma unroll
            for(int k=0;k<HPB;k++){
                #pragma unroll
                for(int j=0;j<16;j++)sum[k]+=score[k*total+i+j]*v[j];}}
        for(;i<wn;i++){float vv=cached_row(win,chunk,abs_base+wo+i,chunk_start,window,dim)[d];
            #pragma unroll
            for(int k=0;k<HPB;k++)sum[k]+=score[k*total+i]*vv;}
        i=0;
        for(;i+16<=cn;i+=16){int ix[16];float v[16];
            #pragma unroll
            for(int j=0;j<16;j++)ix[j]=IDX?ts[i+j]:i+j;
            #pragma unroll
            for(int j=0;j<16;j++)v[j]=comp[(long long)ix[j]*dim+d];
            #pragma unroll
            for(int k=0;k<HPB;k++){
                #pragma unroll
                for(int j=0;j<16;j++)sum[k]+=score[k*total+wn+i+j]*v[j];}}
        for(;i<cn;i++){int ix=IDX?ts[i]:i;float vv=comp[(long long)ix*dim+d];
            #pragma unroll
            for(int k=0;k<HPB;k++)sum[k]+=score[k*total+wn+i]*vv;}
        #pragma unroll
        for(int k=0;k<HPB;k++)if(k<nh)out[(long long)(h0+k)*dim+d]=__bfloat162float(__float2bfloat16(sum[k]/dn[k]));
    }
}
template<int IDX> static void launch_sattn_split(float*gsc,int heads,size_t sm1,cudaStream_t st,float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,const int*sel,int abs_base,
        int chunk_start,int window,int dim,float scale,int total,int dynab){
    int ds=sattn_split_ds();
    sattn_split_score<IDX><<<dim3((heads+3)/4,(total+63)/64),512,0,st>>>(gsc,q,win,chunk,comp,meta,sel,abs_base,chunk_start,window,heads,dim,scale,dynab);
    sattn_split_value<IDX><<<dim3((heads+3)/4,(dim+ds-1)/ds),ds,sm1*4,st>>>(out,gsc,win,chunk,comp,sink,meta,sel,abs_base,chunk_start,window,heads,dim,dynab);
}
/* DSV4_SATTN_HPB (4): heads per block; sm1 = the one-head score row bytes
 * (max_total * 4). Four heads when their score rows fit the 48 KB default,
 * else one. */
static int sattn_hpb(void){static int n=-1;if(n<0){const char*e=getenv("DSV4_SATTN_HPB");n=e&&*e?atoi(e):4;if(n!=4)n=1;}return n;}
/* DSV4_SATTN_HPB_PF (16): heads per block for the multi-token launches
 * (prefill chunks), where the token count fills the card and the key rows
 * are shared across the heads; 16, 8 or 4 (else the one-token rule) */
static int sattn_hpb_pf(void){static int n=-1;if(n<0){const char*e=getenv("DSV4_SATTN_HPB_PF");n=e&&*e?atoi(e):16;if(n!=16&&n!=8)n=4;}return n;}
template<int IDX> static void launch_sattn_v2(int heads,int tokens,size_t sm1,cudaStream_t st,float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,const int*sel,int selstride,int abs_base,
        int chunk_start,int window,int dim,float scale){
    if(tokens>1&&sattn_hpb_pf()==16&&sm1*16<=48*1024)
        sparse_attn_cached_v2<IDX,16><<<dim3((heads+15)/16,tokens),512,sm1*16,st>>>(out,q,win,chunk,comp,sink,meta,sel,selstride,abs_base,chunk_start,window,heads,dim,scale);
    else if(tokens>1&&sattn_hpb_pf()>=8&&sm1*8<=48*1024)
        sparse_attn_cached_v2<IDX,8><<<dim3((heads+7)/8,tokens),512,sm1*8,st>>>(out,q,win,chunk,comp,sink,meta,sel,selstride,abs_base,chunk_start,window,heads,dim,scale);
    else if(sattn_hpb()==4&&sm1>=1024&&sm1*4<=48*1024)   /* four heads from 256 rows up: fewer blocks under-fill the card on short layers */
        sparse_attn_cached_v2<IDX,4><<<dim3((heads+3)/4,tokens),512,sm1*4,st>>>(out,q,win,chunk,comp,sink,meta,sel,selstride,abs_base,chunk_start,window,heads,dim,scale);
    else
        sparse_attn_cached_v2<IDX,1><<<dim3(heads,tokens),512,sm1,st>>>(out,q,win,chunk,comp,sink,meta,sel,selstride,abs_base,chunk_start,window,heads,dim,scale);
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

/* V4_SPARSE_FAST (decode, tokens == 1): defined after the indexed kernel below. */
struct SaOld{float*out;const float*q;const float*win;const float*chunk;const float*comp;
    const float*sink;const int*meta;const int*sel;int selstride,abs_base,chunk_start,window,heads,dim;float scale;size_t sm;};
static int sa_fast_decode(Dev*c,int layer,const int*dsel,const int*meta,int abs_base,int chunk_start,
        int window,int heads,int dim,float scale,float*out,size_t qb,
        void(*old_launch)(Dev*,void*),void*old_arg);
static void sa_old_plain(Dev*c,void*p);
static void sa_old_idx(Dev*c,void*p);

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
    if(tokens==1){
        SaOld a={c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,nullptr,0,
                 abs_base,chunk_start,window,heads,dim,scale,sm};
        int r=sa_fast_decode(c,layer,nullptr,meta,abs_base,chunk_start,window,heads,dim,scale,out,qb,sa_old_plain,&a);
        if(r)return r>0;
    }
    if(sattn_v2_on())launch_sattn_v2<0>(heads,tokens,sm,c->stream,
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,nullptr,0,
        abs_base,chunk_start,window,dim,scale);
    else sparse_attn_batch_cached_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(
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
    if(tokens==1){
        SaOld a={c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,(int*)c->aux3,selstride,
                 abs_base,chunk_start,window,heads,dim,scale,sm};
        int r=sa_fast_decode(c,layer,(int*)c->aux3,meta,abs_base,chunk_start,window,heads,dim,scale,out,qb,sa_old_idx,&a);
        if(r)return r>0;
    }
    if(sattn_v2_on())launch_sattn_v2<1>(heads,tokens,sm,c->stream,
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,(int*)c->aux3,
        selstride,abs_base,chunk_start,window,dim,scale);
    else sparse_attn_batch_cached_idx_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,(int*)c->aux3,
        selstride,abs_base,chunk_start,window,heads,dim,scale);
    return ok(cudaGetLastError(),"cached idx sparse attention launch")&&
           ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"cached idx attention download")&&
           ok(cudaStreamSynchronize(c->stream),"cached idx sparse attention sync");
}

/* ---- V4_SPARSE_FAST (opt-in): the decode step (tokens == 1) of both cached sparse
 * attention wrappers in three wider kernels, bitwise equal to the kernels above.
 * Those kernels, as nvcc compiles them by default (FMA contraction on), compute:
 *  - score of (head, item): lane l runs z = fma(q[d], v[d], z) over d = l, l+32, ...
 *    in order, then the __shfl_down tree (16, 8, 4, 2, 1) with plain adds, then z*scale;
 *  - softmax: the max (exact in any order); a_i = expf(score_i - m); the denominator
 *    starts at expf(sink - m) and adds a_0, a_1, ... in item order on one thread; the
 *    stored weight is bf16(a_i);
 *  - output dim d: sum = fma(w_i, v_i[d], sum) over the items in order (window items,
 *    then compressed items), then bf16(sum / den).
 * The replica spells each step with __fmaf_rn / __fadd_rn / __fmul_rn / __fdiv_rn in
 * the same operand order. The old kernels run one block per head, the softmax on one
 * thread of it and the output over every item per dim; these spread items and heads
 * over more blocks and compute each row's address once.
 * V4_SPARSE_VERIFY=1 runs both and byte-compares the outputs (the old one is returned;
 * mismatches are counted on stderr). Shapes other than dim 512 and heads % 8 == 0 keep
 * the old kernels. */
static int sa_env(int *cache,const char *name){
    if(*cache<0){const char*e=getenv(name);*cache=e&&*e&&atoi(e)!=0;}
    return *cache;
}
static int sa_fast_on(void){static int v=-1;return sa_env(&v,"V4_SPARSE_FAST");}
static int sa_verify_on(void){static int v=-1;return sa_env(&v,"V4_SPARSE_VERIFY");}
enum { SA_DJ = 16, SA_HG = 8, SA_ITEMS = 32, SA_HB = 8, SA_DT = 128, SA_TILE = 1024 };
static long long g_sa_checks,g_sa_bad;
/* Row of item i as an element offset: >= 0 is a window-ring row or (high bit set) a
 * compressed row; -1 - k is row k of this step's chunk. */
__device__ __forceinline__ long long sa_row(int i,int wo,int wn,const int*sel,int abs_base,
        int chunk_start,int window,int dim){
    if(i<wn){int ap=abs_base+wo+i;
        if(ap>=chunk_start)return -1-(long long)(ap-chunk_start);
        return (long long)(ap%window)*dim;}
    int o=sel?sel[i-wn]:i-wn;
    return ((long long)1<<62)|((long long)o*dim);
}
__device__ __forceinline__ float sa_load(long long r,int d,const float*win,
        const float*comp,const float*chunk,int dim){
    if(r<0)return chunk[(-1-r)*dim+d];
    if(r&((long long)1<<62))return comp[(r&~((long long)1<<62))+d];
    return win[r+d];
}
__global__ void sa_rows_kernel(long long*rows,const int*sel,int wo,int wn,int total,int abs_base,
        int chunk_start,int window,int dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<total)rows[i]=sa_row(i,wo,wn,sel,abs_base,chunk_start,window,dim);
}
/* grid (ceil(total/SA_ITEMS), heads/SA_HG), 256 threads; dim == 32*SA_DJ. */
__global__ void sa_fast_scores(float*score,const float*q,const long long*rows,
        const float*win,const float*comp,const float*chunk,int total,int dim,float scale){
    __shared__ float qs[SA_HG*32*SA_DJ];
    int h0=blockIdx.y*SA_HG;
    for(int i=threadIdx.x;i<SA_HG*dim;i+=blockDim.x)qs[i]=q[(long long)h0*dim+i];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int k=warp;k<SA_ITEMS;k+=blockDim.x/32){
        int item=blockIdx.x*SA_ITEMS+k;
        if(item>=total)break;
        long long r=rows[item];
        float v[SA_DJ];
#pragma unroll
        for(int j=0;j<SA_DJ;j++)v[j]=sa_load(r,lane+32*j,win,comp,chunk,dim);
#pragma unroll
        for(int hh=0;hh<SA_HG;hh++){
            const float*qh=qs+hh*dim;
            float z=0;
#pragma unroll
            for(int j=0;j<SA_DJ;j++)z=__fmaf_rn(qh[lane+32*j],v[j],z);
            for(int n=16;n;n>>=1)z=__fadd_rn(z,__shfl_down_sync(0xffffffff,z,n));
            if(!lane)score[(long long)(h0+hh)*total+item]=__fmul_rn(z,scale);
        }
    }
}
/* grid heads, 256 threads; dynamic smem total floats. Writes w = bf16(a) and den. */
__global__ void sa_fast_softmax(float*w,float*den,const float*score,const float*sink,int total){
    extern __shared__ float a[];
    __shared__ float red[256];
    int h=blockIdx.x;
    const float*s=score+(long long)h*total;
    float m=-INFINITY;
    for(int i=threadIdx.x;i<total;i+=blockDim.x)m=fmaxf(m,s[i]);
    red[threadIdx.x]=m;__syncthreads();
    for(int n=blockDim.x/2;n;n>>=1){if(threadIdx.x<n)red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+n]);__syncthreads();}
    m=red[0];
    for(int i=threadIdx.x;i<total;i+=blockDim.x){float e=expf(s[i]-m);a[i]=e;
        w[(long long)h*total+i]=__bfloat162float(__float2bfloat16(e));}
    __syncthreads();
    if(!threadIdx.x){float d0=expf(sink[h]-m);for(int i=0;i<total;i++)d0=__fadd_rn(d0,a[i]);den[h]=d0;}
}
/* grid (dim/SA_DT, heads/SA_HB), SA_DT threads; items in tiles of SA_TILE. */
__global__ void sa_fast_out(float*out,const float*w,const float*den,const long long*rows,
        const float*win,const float*comp,const float*chunk,int total,int dim){
    __shared__ float ws[SA_HB*SA_TILE];
    __shared__ long long rs[SA_TILE];
    int d=blockIdx.x*SA_DT+threadIdx.x,h0=blockIdx.y*SA_HB;
    float sum[SA_HB];
#pragma unroll
    for(int hh=0;hh<SA_HB;hh++)sum[hh]=0;
    for(int base=0;base<total;base+=SA_TILE){
        int n=total-base<SA_TILE?total-base:SA_TILE;
        __syncthreads();
        for(int i=threadIdx.x;i<n;i+=blockDim.x){
            rs[i]=rows[base+i];
#pragma unroll
            for(int hh=0;hh<SA_HB;hh++)ws[hh*SA_TILE+i]=w[(long long)(h0+hh)*total+base+i];
        }
        __syncthreads();
        if(d<dim){
            int i=0;
            for(;i+4<=n;i+=4){
                float v0=sa_load(rs[i],d,win,comp,chunk,dim),v1=sa_load(rs[i+1],d,win,comp,chunk,dim),
                      v2=sa_load(rs[i+2],d,win,comp,chunk,dim),v3=sa_load(rs[i+3],d,win,comp,chunk,dim);
#pragma unroll
                for(int hh=0;hh<SA_HB;hh++){
                    const float*wh=ws+hh*SA_TILE+i;
                    float acc=sum[hh];
                    acc=__fmaf_rn(wh[0],v0,acc);acc=__fmaf_rn(wh[1],v1,acc);
                    acc=__fmaf_rn(wh[2],v2,acc);acc=__fmaf_rn(wh[3],v3,acc);
                    sum[hh]=acc;
                }
            }
            for(;i<n;i++){
                float v=sa_load(rs[i],d,win,comp,chunk,dim);
#pragma unroll
                for(int hh=0;hh<SA_HB;hh++)sum[hh]=__fmaf_rn(ws[hh*SA_TILE+i],v,sum[hh]);
            }
        }
    }
    if(d<dim){
#pragma unroll
        for(int hh=0;hh<SA_HB;hh++)
            out[(long long)(h0+hh)*dim+d]=__bfloat162float(__float2bfloat16(__fdiv_rn(sum[hh],den[h0+hh])));
    }
}
/* The fast path on c->stream: device q, chunk, sinks and sel (or null) are already
 * uploaded; output into dout. Returns 0 for an unsupported shape (old kernel runs). */
static int sa_fast_launch(Dev*c,int layer,const float*dq,const float*dchunk,const float*dsinks,
        const int*dsel,int wo,int wn,int cn,int abs_base,
        int chunk_start,int window,int heads,int dim,float scale,float*dout){
    int total=wn+cn;
    if(dim!=32*SA_DJ||heads%SA_HG||heads%SA_HB||dim%SA_DT||total<1||(size_t)total*4>96*1024)return 0;
    size_t sb=(size_t)heads*total*4;
    if(!buf((void**)&c->sa_score,&c->sa_score_cap,sb)||!buf((void**)&c->sa_w,&c->sa_w_cap,sb)||
       !buf((void**)&c->sa_den,&c->sa_den_cap,(size_t)heads*4)||
       !buf((void**)&c->sa_rows,&c->sa_rows_cap,(size_t)total*sizeof(long long)))return 0;
    size_t sm=(size_t)total*4;
    if(sm>48*1024&&cudaFuncSetAttribute(sa_fast_softmax,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)!=cudaSuccess){
        cudaGetLastError();return 0;}
    sa_rows_kernel<<<(total+255)/256,256,0,c->stream>>>(c->sa_rows,dsel,wo,wn,total,abs_base,chunk_start,window,dim);
    sa_fast_scores<<<dim3((total+SA_ITEMS-1)/SA_ITEMS,heads/SA_HG),256,0,c->stream>>>(
        c->sa_score,dq,c->sa_rows,g_kv_ring[layer],g_kv_comp[layer],dchunk,total,dim,scale);
    sa_fast_softmax<<<heads,256,sm,c->stream>>>(c->sa_w,c->sa_den,c->sa_score,dsinks,total);
    sa_fast_out<<<dim3(dim/SA_DT,heads/SA_HB),SA_DT,0,c->stream>>>(
        dout,c->sa_w,c->sa_den,c->sa_rows,g_kv_ring[layer],g_kv_comp[layer],dchunk,total,dim);
    return ok(cudaGetLastError(),"fast sparse attention launch");
}
/* Fast path for both wrappers (tokens == 1, uploads queued). 1: the output is in `out`
 * and the stream synced; 0: run the old kernel; -1: a CUDA error. */
static int sa_fast_decode(Dev*c,int layer,const int*dsel,const int*meta,int abs_base,int chunk_start,
        int window,int heads,int dim,float scale,float*out,size_t qb,
        void(*old_launch)(Dev*,void*),void*old_arg){
    if(!sa_fast_on())return 0;
    int verify=sa_verify_on();
    if(verify&&!buf((void**)&c->sa_out,&c->sa_out_cap,qb))return 0;
    float*dout=verify?c->sa_out:c->p4;
    if(!sa_fast_launch(c,layer,c->p2,c->p3,c->aux1,dsel,meta[0],meta[1],meta[2],abs_base,chunk_start,
                       window,heads,dim,scale,dout))
        return 0;
    if(!verify)
        return ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"fast attention download")&&
               ok(cudaStreamSynchronize(c->stream),"fast sparse attention sync")?1:-1;
    old_launch(c,old_arg);
    float*hfast=(float*)malloc(qb);
    int okk=hfast&&ok(cudaGetLastError(),"verify old sparse launch")&&
            ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"verify old download")&&
            ok(cudaMemcpyAsync(hfast,c->sa_out,qb,cudaMemcpyDeviceToHost,c->stream),"verify fast download")&&
            ok(cudaStreamSynchronize(c->stream),"verify sparse sync");
    if(!okk){free(hfast);return -1;}
    g_sa_checks++;
    int same=memcmp(out,hfast,qb)==0;
    free(hfast);
    if(!same&&++g_sa_bad<=16)fprintf(stderr,"v4_sparse_verify MISMATCH layer=%d total=%d\n",layer,meta[1]+meta[2]);
    if(g_sa_checks%2048==0||(!same&&g_sa_bad<=1))
        fprintf(stderr,"v4_sparse_verify checks=%lld mismatches=%lld\n",g_sa_checks,g_sa_bad);
    return 1;
}
static void sa_old_plain(Dev*c,void*p){SaOld*a=(SaOld*)p;
    sparse_attn_batch_cached_kernel<<<dim3(a->heads,1),256,a->sm,c->stream>>>(a->out,a->q,a->win,a->chunk,a->comp,
        a->sink,a->meta,a->abs_base,a->chunk_start,a->window,a->heads,a->dim,a->scale);}
static void sa_old_idx(Dev*c,void*p){SaOld*a=(SaOld*)p;
    sparse_attn_batch_cached_idx_kernel<<<dim3(a->heads,1),256,a->sm,c->stream>>>(a->out,a->q,a->win,a->chunk,a->comp,
        a->sink,a->meta,a->sel,a->selstride,a->abs_base,a->chunk_start,a->window,a->heads,a->dim,a->scale);}

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

/* DSV4_IDX_COAL twin of indexer_score_kernel: keys_t [dim][kstride], a
 * warp's 32 candidates in one line per i. The same mul/add sequence. */
__global__ void indexer_score_coal_kernel(float*scores,const float*queries,const float*keys_t,int kstride,
        const float*head_w,const int*counts,int heads,int dim,int count){
    int t=blockIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
    extern __shared__ float qs[];
    int qn=heads*dim;
    for(int i=threadIdx.x;i<qn;i+=blockDim.x)qs[i]=queries[(long long)t*qn+i];
    __syncthreads();
    if(c>=count)return;
    float score=0.f;
    if(c<counts[t]){
        const float*k=keys_t+c;
        for(int h=0;h<heads;h++){
            const float*q=qs+h*dim;
            float dot=0.f;
            for(int i=0;i<dim;i++)dot=__fadd_rn(dot,__fmul_rn(q[i],k[(long long)i*kstride]));
            score=__fadd_rn(score,__fmul_rn(fmaxf(dot,0.f),head_w[(long long)t*heads+h]));
        }
    }
    scores[(long long)t*count+c]=score;
}

/* Naruto's indexer tiles keys transposed and splits work across heads.
 * Store individual head terms here, then fold them in the original order. */
__global__ void indexer_score_heads(float *scores,const float *queries,const float *keys,
        const float *head_w,const int *counts,int heads,int dim,int count){
    constexpr int tile=8;
    int candidate=blockIdx.x*tile+threadIdx.x%tile,head=threadIdx.x/tile,t=blockIdx.y;
    extern __shared__ float scratch[];
    float *ks=scratch,*partial=ks+dim*(tile+1);
    for(int i=threadIdx.x;i<dim*tile;i+=blockDim.x){
        int row=i/dim,col=i%dim,idx=blockIdx.x*tile+row;
        ks[col*(tile+1)+row]=idx<count?keys[(long long)idx*dim+col]:0.f;
    }
    __syncthreads();
    float dot=0.f;
    if(candidate<count&&candidate<counts[t]){
        const float *q=queries+((long long)t*heads+head)*dim;
        for(int d=0;d<dim;d++)dot=__fadd_rn(dot,__fmul_rn(q[d],ks[d*(tile+1)+threadIdx.x%tile]));
    }
    partial[head*tile+threadIdx.x%tile]=__fmul_rn(fmaxf(dot,0.f),head_w[(long long)t*heads+head]);
    __syncthreads();
    if(head==0&&candidate<count){
        float score=0.f;
        if(candidate<counts[t])for(int h=0;h<heads;h++)score=__fadd_rn(score,partial[h*tile+threadIdx.x%tile]);
        scores[(long long)t*count+candidate]=score;
    }
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
    const char *parallel=getenv("DSV4_CUDA_INDEXER_HEADS");
    if((!parallel||atoi(parallel))&&heads<=128&&dim<=512){
        size_t shared=(size_t)(dim*9+heads*8)*4;
        indexer_score_heads<<<dim3((count+7)/8,tokens),heads*8,shared,c->stream>>>(
            c->p4,c->p2,c->p3,c->aux1,(int*)c->aux2,heads,dim,count);
    }else{
        if(sm>48*1024&&cudaFuncSetAttribute(indexer_score_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)!=cudaSuccess)return 0;
        indexer_score_kernel<<<dim3((count+127)/128,tokens),128,sm,c->stream>>>(
            c->p4,c->p2,c->p3,c->aux1,(int*)c->aux2,heads,dim,count);
    }
    return ok(cudaGetLastError(),"indexer score launch")&&
           ok(cudaMemcpyAsync(scores,c->p4,sb,cudaMemcpyDeviceToHost,c->stream),"indexer score download")&&
           ok(cudaStreamSynchronize(c->stream),"indexer score sync");
}

static int g_idx_coal=-1;
static int idx_coal_on(void){if(g_idx_coal<0){const char*e=getenv("DSV4_IDX_COAL");g_idx_coal=e&&*e?*e!='0':attn_device_on();}return g_idx_coal;}
/* src [rows][cols] -> dst [cols][dstride] (dstride >= rows) through a padded tile */
__global__ void transpose_f32_kernel(float*dst,const float*src,int rows,int cols,int dstride){
    __shared__ float tile[32][33];
    int bx=blockIdx.x*32,by=blockIdx.y*32;
    int x=bx+threadIdx.x,y=by+threadIdx.y;
    for(int j=0;j<32;j+=8){int yy=y+j;if(x<cols&&yy<rows)tile[threadIdx.y+j][threadIdx.x]=src[(long long)yy*cols+x];}
    __syncthreads();
    x=by+threadIdx.x;y=bx+threadIdx.y;
    for(int j=0;j<32;j+=8){int yy=y+j;if(x<rows&&yy<cols)dst[(long long)yy*dstride+x]=tile[threadIdx.x][threadIdx.y+j];}
}
static void transpose_f32(float*dst,const float*src,int rows,int cols,int dstride,cudaStream_t st){
    transpose_f32_kernel<<<dim3((cols+31)/32,(rows+31)/32),dim3(32,8),0,st>>>(dst,src,rows,cols,dstride);
}

/* Exact replica of the CPU reference matmul_fp8 (quant.h): per output row,
 * per 128-column block a sequential fp32 accumulate of e4m3(w)*x, then a
 * DOUBLE accumulate of block*scale across blocks, cast to float. Separately
 * rounded mul/add (the CPU build does not contract to FMA), so results are
 * bitwise the reference. x is the fp8-qdq'd activation (done on the host).
 * One thread per (row, token); tokens are the fast dimension so the 64
 * threads of a token block broadcast-read the same weight bytes. */
__device__ __forceinline__ float e4m3_dev(uint8_t b){
    return e4m3(b);
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

/* DSV4_IDX_COAL twins of the two replicas: x_t [cols][tokens], a warp's
 * 32 tokens in one line per column. The same per-thread sequence. */
__global__ void fp8_ref_matmul_xt_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x_t,int rows,int cols,int tokens){
    int t=threadIdx.x,o=blockIdx.x;
    if(t>=tokens||o>=rows)return;
    const uint8_t*wr=w+(long long)o*cols;
    const float*xs=x_t+t;
    int nblk=(cols+127)/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    double a=0.0;
    for(int bi=0;bi*128<cols;bi++){
        int base=bi*128,blen=cols-base<128?cols-base:128;
        float acc=0.f;
        for(int i=base;i<base+blen;i++)acc=__fadd_rn(acc,__fmul_rn(e4m3_dev(wr[i]),xs[(long long)i*tokens]));
        a=__dadd_rn(a,__dmul_rn((double)acc,(double)scl[bi]));
    }
    y[(long long)t*rows+o]=__double2float_rn(a);
}
__global__ void fp8_ref_matmul_rows8_xt_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x_t,int rows,int cols,int tokens){
    int t=threadIdx.x,o=blockIdx.x;
    if(t>=tokens||o>=rows)return;
    const float*xs=x_t+t;
    int nblk=cols/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    long long tile=o>>3;int r=o&7;
    float sum=0.f;
    for(int base=0;base<cols;base+=128){
        float sc=scl[base/128];
        for(int i=base;i<base+128;i++){
            float v=e4m3_dev(w[((tile*cols)+i)*8+r]);
            sum=__fadd_rn(sum,__fmul_rn(__fmul_rn(xs[(long long)i*tokens],v),sc));
        }
    }
    y[(long long)t*rows+o]=sum;
}
/* DSV4_IDX_TILE (1): the packed replica a tile (8 rows) per block: one
 * 8-byte load per column gives the tile's weight bytes, the activation
 * load is shared by the 8 rows' sums. Each (row, token) sum in its order. */
__global__ void fp8_ref_matmul_rows8_tile_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x_t,int rows,int cols,int tokens){
    int t=threadIdx.x;long long tile=blockIdx.x;
    if(t>=tokens||tile*8>=rows)return;
    const float*xs=x_t+t;
    int nblk=cols/128;
    const float*scl=bscale+(long long)((tile*8)/128)*nblk;
    const unsigned long long*wt=(const unsigned long long*)(w+tile*cols*8);
    float sum[8];
    #pragma unroll
    for(int r=0;r<8;r++)sum[r]=0.f;
    for(int base=0;base<cols;base+=128){
        float sc=scl[base/128];
        for(int i=base;i<base+128;i++){
            unsigned long long w8=wt[i];
            float xv=xs[(long long)i*tokens];
            #pragma unroll
            for(int r=0;r<8;r++){
                float v=e4m3_dev((uint8_t)(w8>>(8*r)));
                sum[r]=__fadd_rn(sum[r],__fmul_rn(__fmul_rn(xv,v),sc));
            }
        }
    }
    #pragma unroll
    for(int r=0;r<8;r++)y[(long long)t*rows+tile*8+r]=sum[r];
}
static int idx_tile_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_IDX_TILE");on=e&&*e?*e!='0':attn_device_on();}return on;}
/* the replica over x [tokens][cols] on the stream: transposed into x_t when
 * DSV4_IDX_COAL (and the scratch) allows, the row-major kernels otherwise */
static int fp8_ref_matmul_launch(Dev*c,float*y,const uint8_t*w,const float*bscale,const float*x,float*x_t,
        int rows,int cols,int packed_rows8,int tokens){
    int threads=tokens<32?32:((tokens+31)/32)*32;
    if(x_t){
        transpose_f32(x_t,x,tokens,cols,tokens,c->stream);
        if(packed_rows8&&idx_tile_on()&&rows%8==0)
            fp8_ref_matmul_rows8_tile_kernel<<<rows/8,threads,0,c->stream>>>(y,w,bscale,x_t,rows,cols,tokens);
        else if(packed_rows8)
            fp8_ref_matmul_rows8_xt_kernel<<<rows,threads,0,c->stream>>>(y,w,bscale,x_t,rows,cols,tokens);
        else
            fp8_ref_matmul_xt_kernel<<<rows,threads,0,c->stream>>>(y,w,bscale,x_t,rows,cols,tokens);
    }else if(packed_rows8)
        fp8_ref_matmul_rows8_kernel<<<rows,threads,0,c->stream>>>(y,w,bscale,x,rows,cols,tokens);
    else
        fp8_ref_matmul_kernel<<<rows,threads,0,c->stream>>>(y,w,bscale,x,rows,cols,tokens);
    return ok(cudaGetLastError(),"fp8 ref matmul launch");
}

extern "C" int dsv4_cuda_fp8_ref_matmul_resident(Dsv4CudaTensor*t,const float*x,int tokens,float*y){
    Dev*c=t?ctx(t->device):nullptr;
    if(!c||!x||!y||tokens<1||tokens>1024||(t->fmt!=10&&t->fmt!=11))return 0;
    size_t xb=(size_t)tokens*t->I*sizeof(float),yb=(size_t)tokens*t->O*sizeof(float);
    if(!ok(cudaSetDevice(t->device),"select resident fp8 ref device")||
       !buf((void**)&c->p2,&c->p2cap,xb)||!buf((void**)&c->p4,&c->p4cap,yb)||
       !ok(cudaMemcpyAsync(c->p2,x,xb,cudaMemcpyHostToDevice,c->stream),"resident fp8 ref activation upload"))return 0;
    int threads=tokens<32?32:((tokens+31)/32)*32;
    if(t->fmt==11)
        fp8_ref_matmul_rows8_kernel<<<t->O,threads,0,c->stream>>>(c->p4,(const uint8_t*)t->w,(const float*)t->scale,c->p2,t->O,t->I,tokens);
    else
        fp8_ref_matmul_kernel<<<t->O,threads,0,c->stream>>>(c->p4,(const uint8_t*)t->w,(const float*)t->scale,c->p2,t->O,t->I,tokens);
    return ok(cudaGetLastError(),"resident fp8 ref launch")&&
           ok(cudaMemcpyAsync(y,c->p4,yb,cudaMemcpyDeviceToHost,c->stream),"resident fp8 ref download")&&
           ok(cudaStreamSynchronize(c->stream),"resident fp8 ref sync");
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
    float*xt=(idx_coal_on()&&buf((void**)&c->p3,&c->p3cap,xb))?c->p3:nullptr;
    return fp8_ref_matmul_launch(c,c->p4,(const uint8_t*)c->p1,c->aux1,c->p2,xt,rows,cols,packed_rows8,tokens)&&
           ok(cudaMemcpyAsync(y,c->p4,yb,cudaMemcpyDeviceToHost,c->stream),"fp8 ref matmul download")&&
           ok(cudaStreamSynchronize(c->stream),"fp8 ref matmul sync");
}

/* ---- Device-resident indexer selection (V4_IDX_FUSED) ----------------------
 * Per layer the card keeps the indexer's wq_b bytes and f32 block scales in
 * the CPU reference layout, the bf16 head-weight rows, and the compressed keys
 * (append-only; re-seeded when the owner changes or the count regresses). One
 * round trip per selection: projection replica -> query prep (bf16 round,
 * RoPE, hadamard, fp4 round trip) -> head weights -> scoring; the host keeps
 * the sort. The arithmetic mirrors the GCC build of the reference, which fuses
 * the head-weight accumulate and the cosine products of the RoPE into FMAs
 * (read off the binary) and nothing else, so the scores are bitwise the
 * CPU's and the top-k boundary cannot move. */
static uint8_t *g_idx_w[DSV4_KV_CACHE_LAYERS];
static float *g_idx_s[DSV4_KV_CACHE_LAYERS];
static uint16_t *g_idx_hw[DSV4_KV_CACHE_LAYERS];
static const void *g_idx_w_src[DSV4_KV_CACHE_LAYERS];
static int g_idx_rows[DSV4_KV_CACHE_LAYERS],g_idx_cols[DSV4_KV_CACHE_LAYERS],g_idx_packed[DSV4_KV_CACHE_LAYERS],
           g_idx_heads[DSV4_KV_CACHE_LAYERS],g_idx_hidden[DSV4_KV_CACHE_LAYERS],g_idx_dev[DSV4_KV_CACHE_LAYERS];
static float *g_idx_keys[DSV4_KV_CACHE_LAYERS];
static float *g_idx_keys_t[DSV4_KV_CACHE_LAYERS];   /* DSV4_IDX_COAL: [dim][cap] copy, refreshed on append */
static int g_idx_keys_t_n[DSV4_KV_CACHE_LAYERS];     /* rows the transposed copy holds (the fused layer appends without it) */
static int g_idx_keys_cap[DSV4_KV_CACHE_LAYERS],g_idx_keys_n[DSV4_KV_CACHE_LAYERS],g_idx_keys_dim[DSV4_KV_CACHE_LAYERS];
static unsigned long long g_idx_keys_owner[DSV4_KV_CACHE_LAYERS];

__device__ __forceinline__ float bf16_round_dev(float v){
    unsigned b=__float_as_uint(v);
    if((b&0x7f800000u)!=0x7f800000u){unsigned tie=(b>>16)&1u;b+=0x7fffu+tie;}
    return __uint_as_float(b&0xffff0000u);
}
__constant__ float c_idx_e2m1[16]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};

/* Query prep in place, grid (heads, tokens), block = dim threads, dim floats of
 * shared memory: bf16 round, RoPE on the last rope_dim (rope_fma 1 = the
 * cosine product fused, the GCC build's vfmaddsub; 2 = the sine product fused;
 * 0 = none), hadamard butterflies + scale + bf16 round, fp4 (e2m1 per 32,
 * e8m0 scale) round trip + bf16 round. Same operations in the same order as
 * indexer_prepare_queries. */
__global__ void indexer_prep_kernel(float*q,const float*cosv,const float*sinv,int heads,int dim,
        int rope_dim,float had_scale,int rope_fma){
    int h=blockIdx.x,t=blockIdx.y,i=threadIdx.x,pairs=rope_dim/2;
    float*row=q+((long long)t*heads+h)*dim;
    extern __shared__ float s[];
    __shared__ float bsc[32];
    s[i]=bf16_round_dev(row[i]);
    __syncthreads();
    if(i<pairs){
        int base=dim-rope_dim+2*i;
        float re=s[base],im=s[base+1];
        float c=cosv[(long long)t*pairs+i],sn=sinv[(long long)t*pairs+i];
        float a,b;
        if(rope_fma==1){a=fmaf(re,c,-__fmul_rn(im,sn));b=fmaf(im,c,__fmul_rn(re,sn));}
        else if(rope_fma==2){a=fmaf(-im,sn,__fmul_rn(re,c));b=fmaf(re,sn,__fmul_rn(im,c));}
        else{a=__fsub_rn(__fmul_rn(re,c),__fmul_rn(im,sn));b=__fadd_rn(__fmul_rn(re,sn),__fmul_rn(im,c));}
        s[base]=bf16_round_dev(a);s[base+1]=bf16_round_dev(b);
    }
    __syncthreads();
    for(int width=1;width<dim;width*=2){
        if(i<dim/2){
            int base=(i/width)*2*width,j=i-(i/width)*width;
            float l=s[base+j],r=s[base+width+j];
            s[base+j]=__fadd_rn(l,r);s[base+width+j]=__fsub_rn(l,r);
        }
        __syncthreads();
    }
    s[i]=bf16_round_dev(__fmul_rn(s[i],had_scale));
    __syncthreads();
    if(i<dim/32){
        float m=0.f;
        for(int k=0;k<32;k++)m=fmaxf(m,fabsf(s[i*32+k]));
        m=fmaxf(m,6.0f*ldexpf(1.0f,-126));
        int e;float fr=frexpf(__fdiv_rn(m,6.0f),&e);
        int ex=(fr==0.5f)?e-1:e;
        if(ex<-127)ex=-127;
        if(ex>127)ex=127;
        bsc[i]=ldexpf(1.0f,ex);
    }
    __syncthreads();
    float sc=bsc[i>>5];
    float v=fmaxf(-6.0f,fminf(6.0f,__fdiv_rn(s[i],sc)));
    int best=0;float d=fabsf(__fsub_rn(v,c_idx_e2m1[0]));
    for(int code=1;code<16;code++){float cd=fabsf(__fsub_rn(v,c_idx_e2m1[code]));if(cd<d){d=cd;best=code;}}
    row[i]=bf16_round_dev(__fmul_rn(c_idx_e2m1[best],sc));
}

/* head_w[t][h] = (sum_c bf16(w[h][c]) * x[t][c]) * wscale, sequential fp32;
 * hw_fma 1 = the accumulate fused (vfmadd132ss in the build), 0 = separate. */
__device__ __forceinline__ void headw_step8(float&sum,uint4 wv,float4 x0,float4 x1,int hw_fma){
    float w0=__uint_as_float(wv.x<<16),w1=__uint_as_float(wv.x&0xffff0000u),
          w2=__uint_as_float(wv.y<<16),w3=__uint_as_float(wv.y&0xffff0000u),
          w4=__uint_as_float(wv.z<<16),w5=__uint_as_float(wv.z&0xffff0000u),
          w6=__uint_as_float(wv.w<<16),w7=__uint_as_float(wv.w&0xffff0000u);
    if(hw_fma){
        sum=fmaf(w0,x0.x,sum);sum=fmaf(w1,x0.y,sum);sum=fmaf(w2,x0.z,sum);sum=fmaf(w3,x0.w,sum);
        sum=fmaf(w4,x1.x,sum);sum=fmaf(w5,x1.y,sum);sum=fmaf(w6,x1.z,sum);sum=fmaf(w7,x1.w,sum);
    }else{
        sum=__fadd_rn(sum,__fmul_rn(w0,x0.x));sum=__fadd_rn(sum,__fmul_rn(w1,x0.y));
        sum=__fadd_rn(sum,__fmul_rn(w2,x0.z));sum=__fadd_rn(sum,__fmul_rn(w3,x0.w));
        sum=__fadd_rn(sum,__fmul_rn(w4,x1.x));sum=__fadd_rn(sum,__fmul_rn(w5,x1.y));
        sum=__fadd_rn(sum,__fmul_rn(w6,x1.z));sum=__fadd_rn(sum,__fmul_rn(w7,x1.w));
    }
}
__global__ void indexer_headw_kernel(float*hw,const uint16_t*w,const float*x,int heads,int hidden,
        float wscale,int hw_fma){
    int t=blockIdx.x,h=threadIdx.x;
    if(h>=heads)return;
    const uint16_t*row=w+(long long)h*hidden;const float*xi=x+(long long)t*hidden;
    float sum=0.f;
    if((hidden&7)==0&&!((uintptr_t)row&15)&&!((uintptr_t)xi&15)){
        /* 8 columns per step through 16-byte loads (the chain is latency-bound
         * on the loads, not the FMAs); the same elements in the same order. */
        const uint4*rv=(const uint4*)row;const float4*xv=(const float4*)xi;
        for(int c=0;c<hidden/8;c++)headw_step8(sum,rv[c],xv[2*c],xv[2*c+1],hw_fma);
    }else if(hw_fma)for(int c=0;c<hidden;c++)sum=fmaf(__uint_as_float((unsigned)row[c]<<16),xi[c],sum);
    else for(int c=0;c<hidden;c++)sum=__fadd_rn(sum,__fmul_rn(__uint_as_float((unsigned)row[c]<<16),xi[c]));
    hw[(long long)t*heads+h]=__fmul_rn(sum,wscale);
}
/* One token's head weights with sixteen lanes per head: the lanes fetch the
 * head's next 128 columns (256 bytes, one coalesced load) into shared
 * memory and lane 0 runs the row's serial chain from there in the same
 * element order as the kernel above (bitwise). The direct kernel's one
 * thread per head is latency-bound on 512 strided 16-byte loads (0.115
 * ms per layer at 64 heads x 4096); here each chunk costs one load and
 * 128 steps of the chain. heads*16 threads per block (heads <= 64),
 * heads*272 bytes of shared memory. */
#define HEADW_LANES 16
#define HEADW_CHUNK (HEADW_LANES*8)
#define HEADW_PAD (HEADW_LANES*16+16)
__global__ void indexer_headw_lanes_kernel(float*hw,const uint16_t*w,const float*x,int heads,int hidden,
        float wscale,int hw_fma){
    extern __shared__ unsigned char headw_sh[];
    int t=blockIdx.x,h=threadIdx.x/HEADW_LANES,lane=threadIdx.x%HEADW_LANES;
    if(h>=heads)return;
    const uint16_t*row=w+(long long)h*hidden;const float4*xv=(const float4*)(x+(long long)t*hidden);
    uint4*my=(uint4*)(headw_sh+(size_t)h*HEADW_PAD);
    float sum=0.f;
    for(int c0=0;c0<hidden;c0+=HEADW_CHUNK){
        my[lane]=*(const uint4*)(row+c0+lane*8);
        __syncwarp();
        if(lane==0){
            #pragma unroll
            for(int part=0;part<HEADW_LANES;part++){
                int c=(c0>>3)+part;
                headw_step8(sum,my[part],xv[2*c],xv[2*c+1],hw_fma);
            }
        }
        __syncwarp();
    }
    if(lane==0)hw[(long long)t*heads+h]=__fmul_rn(sum,wscale);
}

/* DSV4_IDX_HEADW (2): one warp per head on its own block (the lanes kernel
 * above puts all heads' lanes in one block, on one SM), the row's 256-column
 * chunks (512 bytes, one coalesced load) double-buffered so the next
 * chunk's load overlaps the chain's 256 steps, and x staged in shared
 * memory once; lane 0 runs the chain in the same element order (bitwise:
 * DSV4_IDX_VERIFY counts its head weights against the CPU's). hidden a
 * multiple of 256. 1 = the lanes kernel, 0 = the direct one. */
#define HEADW_WCHUNK 256
__global__ void __launch_bounds__(32) indexer_headw_warp_kernel(float*hw,const uint16_t*w,const float*x,int heads,int hidden,
        float wscale,int hw_fma){
    extern __shared__ unsigned char headw_wsh[];
    int t=blockIdx.y,h=blockIdx.x,lane=threadIdx.x;
    if(h>=heads)return;
    uint4*buf=(uint4*)headw_wsh;                       /* [2][32] */
    float4*xs=(float4*)(headw_wsh+2*32*sizeof(uint4));  /* hidden/4 float4 */
    const uint16_t*row=w+(long long)h*hidden;const float4*xv=(const float4*)(x+(long long)t*hidden);
    for(int i=lane;i<hidden/4;i+=32)xs[i]=xv[i];
    buf[lane]=*(const uint4*)(row+lane*8);
    __syncwarp();
    float sum=0.f;
    int chunks=hidden/HEADW_WCHUNK;
    for(int it=0;it<chunks;it++){
        int c0=it*HEADW_WCHUNK;
        if(it+1<chunks)buf[((it+1)&1)*32+lane]=*(const uint4*)(row+c0+HEADW_WCHUNK+lane*8);
        if(lane==0){
            const uint4*my=buf+(it&1)*32;
            #pragma unroll
            for(int part=0;part<32;part++){
                int c=(c0>>3)+part;
                headw_step8(sum,my[part],xs[2*c],xs[2*c+1],hw_fma);
            }
        }
        __syncwarp();
    }
    if(lane==0)hw[(long long)t*heads+h]=__fmul_rn(sum,wscale);
}
static int idx_headw_mode(void){static int m=-1;if(m<0){const char*e=getenv("DSV4_IDX_HEADW");m=e&&*e?atoi(e):2;if(m<0)m=0;if(m>2)m=2;}return m;}

static int h2d(Dev*c,void*dst,const void*src,size_t n,const char*what);

/* Batch of one: one thread per output row (the per-token kernels above give a
 * block of 32 threads to each row and leave 31 idle), the same per-row
 * accumulation. */
__global__ void fp8_ref_matmul_rows8_t1_kernel(float*y,const uint8_t*w,const float*bscale,const float*x,int rows,int cols){
    __shared__ float tab[256];   /* e4m3_dev's values, one load per element instead of its ldexpf */
    for(int i=threadIdx.x;i<256;i+=blockDim.x)tab[i]=e4m3_dev((uint8_t)i);
    __syncthreads();
    int o=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=rows)return;
    int nblk=cols/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    long long tile=o>>3;int r=o&7;
    float sum=0.f;
    /* The chain is one thread's serial fadd (the reference order); what
     * hides its load latency is loads in flight: the inner 128 unrolled by
     * 16 lets the compiler issue the byte loads ahead of the chain
     * (DSV4_IDX_KPROF: the 8 MB layer's proj time, before 0.18 ms). */
    for(int base=0;base<cols;base+=128){
        float sc=scl[base/128];
        #pragma unroll 16
        for(int i=base;i<base+128;i++){
            float v=tab[w[((tile*cols)+i)*8+r]];
            sum=__fadd_rn(sum,__fmul_rn(__fmul_rn(x[i],v),sc));
        }
    }
    y[o]=sum;
}
/* DSV4_IDX_PROJ_WARP (0, tried): the rows8 projection as a warp per tile of eight
 * rows. Lane l loads the tile's eight bytes of column c+l (one uint2: 256
 * contiguous bytes per chunk of 32 columns, the next chunk's load in
 * flight), and lanes 0..7 run row r's chain over the columns in order with
 * each column's bytes shuffled from the lane that holds them: the same
 * (x*v)*scale, fadd sequence as the thread-per-row kernel above, whose
 * stride-8 byte loads and three warps per SM made the 8 MB layer 0.09 ms.
 * The decode is the bit form (e4m3_bits). */
__global__ void __launch_bounds__(32) fp8_ref_rows8_warp_kernel(float*y,const uint8_t*w,const float*bscale,const float*x,int rows,int cols){
    int tile=blockIdx.x,lane=threadIdx.x,r=lane&7;
    if(tile*8>=rows)return;
    int nblk=cols/128;
    const float*scl=bscale+(long long)((tile*8)/128)*nblk;
    const uint2*wt=(const uint2*)(w+(long long)tile*cols*8);
    uint2 cur=wt[lane];
    float sum=0.f;
    for(int c=0;c<cols;c+=32){
        uint2 nxt=make_uint2(0u,0u);
        if(c+32<cols)nxt=wt[c+32+lane];
        float sc=scl[c>>7];
        #pragma unroll 8
        for(int i=0;i<32;i++){
            unsigned lo=__shfl_sync(0xffffffffu,cur.x,i),hi=__shfl_sync(0xffffffffu,cur.y,i);
            unsigned word=r<4?lo:hi;unsigned b=(word>>((r&3)*8))&255u;
            float v=e4m3_exact(b);
            sum=__fadd_rn(sum,__fmul_rn(__fmul_rn(x[c+i],v),sc));
        }
        cur=nxt;
    }
    if(lane<8)y[tile*8+r]=sum;
}
static int idx_proj_warp(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_IDX_PROJ_WARP");on=e&&*e&&*e!='0';}return on;}   /* measured +0.011 ms per fused layer: off */
__global__ void fp8_ref_matmul_t1_kernel(float*y,const uint8_t*w,const float*bscale,const float*x,int rows,int cols){
    __shared__ float tab[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)tab[i]=e4m3_dev((uint8_t)i);
    __syncthreads();
    int o=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=rows)return;
    const uint8_t*wr=w+(long long)o*cols;
    int nblk=(cols+127)/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    double a=0.0;
    for(int bi=0;bi*128<cols;bi++){
        int base=bi*128,blen=cols-base<128?cols-base:128;
        float acc=0.f;
        #pragma unroll 16
        for(int i=base;i<base+blen;i++)acc=__fadd_rn(acc,__fmul_rn(tab[wr[i]],x[i]));
        a=__dadd_rn(a,__dmul_rn((double)acc,(double)scl[bi]));
    }
    y[o]=__double2float_rn(a);
}

/* Few tokens: a block of `heads` threads per (token, candidate). Thread h runs
 * head h's dot sequentially over dim, then thread 0 folds the heads in order:
 * the reference's arithmetic with the independent heads side by side instead
 * of one thread per candidate walking 820 strided key rows. Shared q is
 * padded to dim+1 so the per-head rows sit in distinct banks. */
__global__ void indexer_score_small_kernel(float*scores,const float*queries,const float*keys,
        const float*head_w,const int*counts,int heads,int dim,int count){
    int t=blockIdx.y,c=blockIdx.x,h=threadIdx.x;
    extern __shared__ float sm_[];
    float*qs=sm_;                      /* heads x (dim+1) */
    float*ks=sm_+(size_t)heads*(dim+1);  /* dim */
    float*dots=ks+dim;                 /* heads */
    int qn=heads*dim;
    for(int i=h;i<qn;i+=blockDim.x){int hh=i/dim,d=i-hh*dim;qs[hh*(dim+1)+d]=queries[(long long)t*qn+i];}
    for(int i=h;i<dim;i+=blockDim.x)ks[i]=keys[(long long)c*dim+i];
    __syncthreads();
    {
        const float*q=qs+h*(dim+1);
        float dot=0.f;
        for(int i=0;i<dim;i++)dot=__fadd_rn(dot,__fmul_rn(q[i],ks[i]));
        dots[h]=dot;
    }
    __syncthreads();
    if(!h){
        float score=0.f;
        if(c<counts[t])
            for(int hh=0;hh<heads;hh++)
                score=__fadd_rn(score,__fmul_rn(fmaxf(dots[hh],0.f),head_w[(long long)t*heads+hh]));
        scores[(long long)t*count+c]=score;
    }
}

/* DSV4_IDX_SCORE_K (8): the few-token scoring with a block of `heads`
 * threads walking K candidates in a row over queries staged in shared
 * memory once (the kernel above stages the token's 32 KB of queries for
 * every candidate: ~27 MB of L2 traffic per layer for 830 candidates,
 * 0.09 ms with the chains ~0.5 us); per candidate the key row, each
 * thread's sequential dot and thread 0's fold over the heads in order, as
 * above (bitwise: DSV4_IDX_VERIFY counts the scores). 1 = the kernel above. */
__global__ void indexer_score_multi_kernel(float*scores,const float*queries,const float*keys,
        const float*head_w,const int*counts,int heads,int dim,int count,int K){
    int t=blockIdx.y,c0=blockIdx.x*K,h=threadIdx.x;
    extern __shared__ float sm_[];
    float*qs=sm_;                      /* heads x (dim+1) */
    float*ks=sm_+(size_t)heads*(dim+1);  /* dim */
    float*dots=ks+dim;                 /* heads */
    int qn=heads*dim;
    for(int i=h;i<qn;i+=blockDim.x){int hh=i/dim,d=i-hh*dim;qs[hh*(dim+1)+d]=queries[(long long)t*qn+i];}
    const float*q=qs+h*(dim+1);
    int cn=counts[t];
    if(count>cn)count=cn;   /* a grid sized for the most candidates (the fused layer's graph) stops at the token's */
    for(int k=0;k<K;k++){
        int c=c0+k;if(c>=count)break;
        __syncthreads();
        for(int i=h;i<dim;i+=blockDim.x)ks[i]=keys[(long long)c*dim+i];
        __syncthreads();
        float dot=0.f;
        for(int i=0;i<dim;i++)dot=__fadd_rn(dot,__fmul_rn(q[i],ks[i]));
        dots[h]=dot;
        __syncthreads();
        if(!h){
            float score=0.f;
            if(c<cn)
                for(int hh=0;hh<heads;hh++)
                    score=__fadd_rn(score,__fmul_rn(fmaxf(dots[hh],0.f),head_w[(long long)t*heads+hh]));
            scores[(long long)t*count+c]=score;
        }
    }
}
static int idx_score_k(void){static int k=-1;if(k<0){const char*e=getenv("DSV4_IDX_SCORE_K");k=e&&*e?atoi(e):8;if(k<1)k=1;if(k>64)k=64;}return k;}


extern "C" int dsv4_cuda_indexer_resident(int device,int layer,const uint8_t*w,const float*bscale,int rows,int cols,
        int packed_rows8,const uint16_t*head_w,int heads,int hidden){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!w||!bscale||!head_w||rows<1||cols<1||cols%128||heads<1||hidden<1||
       (packed_rows8&&rows%8)||!ok(cudaSetDevice(device),"select indexer resident device"))return 0;
    if(g_idx_w[layer]&&g_idx_w_src[layer]==w&&g_idx_dev[layer]==device&&g_idx_rows[layer]==rows&&
       g_idx_cols[layer]==cols&&g_idx_heads[layer]==heads&&g_idx_hidden[layer]==hidden)return 1;
    if(g_idx_w[layer]){cudaFree(g_idx_w[layer]);g_idx_w[layer]=NULL;}
    if(g_idx_s[layer]){cudaFree(g_idx_s[layer]);g_idx_s[layer]=NULL;}
    if(g_idx_hw[layer]){cudaFree(g_idx_hw[layer]);g_idx_hw[layer]=NULL;}
    size_t wb=(size_t)rows*cols,sb=(size_t)((rows+127)/128)*(cols/128)*4,hb=(size_t)heads*hidden*2;
    if(!alloc_ok(cudaMalloc((void**)&g_idx_w[layer],wb),"indexer weight residency")||
       !alloc_ok(cudaMalloc((void**)&g_idx_s[layer],sb),"indexer scale residency")||
       !alloc_ok(cudaMalloc((void**)&g_idx_hw[layer],hb),"indexer head-weight residency")||
       !h2d(c,g_idx_w[layer],w,wb,"indexer weight upload")||!h2d(c,g_idx_s[layer],bscale,sb,"indexer scale upload")||
       !h2d(c,g_idx_hw[layer],head_w,hb,"indexer head-weight upload")){
        if(g_idx_w[layer])cudaFree(g_idx_w[layer]);
        if(g_idx_s[layer])cudaFree(g_idx_s[layer]);
        if(g_idx_hw[layer])cudaFree(g_idx_hw[layer]);
        g_idx_w[layer]=NULL;g_idx_s[layer]=NULL;g_idx_hw[layer]=NULL;return 0;}
    g_idx_w_src[layer]=w;g_idx_rows[layer]=rows;g_idx_cols[layer]=cols;g_idx_packed[layer]=packed_rows8;
    g_idx_heads[layer]=heads;g_idx_hidden[layer]=hidden;g_idx_dev[layer]=device;
    return 1;
}

/* Rows [n, count) of the host's append-only key cache go up; count 0, another
 * owner, another dim or a count below what is resident re-seed from row 0. */
extern "C" int dsv4_cuda_indexer_keys_sync(int device,int layer,unsigned long long owner,const float*keys,int count,int dim){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||count<0||dim<1||(count&&!keys)||
       !ok(cudaSetDevice(device),"select indexer key device"))return 0;
    if(!count||g_idx_keys_owner[layer]!=owner||g_idx_keys_dim[layer]!=dim||count<g_idx_keys_n[layer]||
       (g_idx_keys[layer]&&g_idx_dev[layer]!=device)){
        g_idx_keys_n[layer]=0;g_idx_keys_owner[layer]=owner;g_idx_keys_dim[layer]=dim;}
    if(!count)return 1;
    if(g_idx_keys_cap[layer]<count){
        int cap=g_idx_keys_cap[layer]?g_idx_keys_cap[layer]:512;
        while(cap<count)cap*=2;
        float*grown=NULL;
        if(!alloc_ok(cudaMalloc((void**)&grown,(size_t)cap*dim*4),"indexer key residency"))return 0;
        int have=g_idx_keys_n[layer];
        if(g_idx_keys[layer]&&have&&
           !ok(cudaMemcpyAsync(grown,g_idx_keys[layer],(size_t)have*dim*4,cudaMemcpyDeviceToDevice,c->stream),"indexer key grow copy")){
            cudaFree(grown);return 0;}
        if(g_idx_keys[layer]){cudaStreamSynchronize(c->stream);cudaFree(g_idx_keys[layer]);}
        g_idx_keys[layer]=grown;g_idx_keys_cap[layer]=cap;
        if(g_idx_keys_t[layer]){cudaFree(g_idx_keys_t[layer]);g_idx_keys_t[layer]=nullptr;}}
    int have=g_idx_keys_n[layer];
    if(count>have){
        if(!ok(cudaMemcpyAsync(g_idx_keys[layer]+(size_t)have*dim,keys+(size_t)have*dim,(size_t)(count-have)*dim*4,
                               cudaMemcpyHostToDevice,c->stream),"indexer key append"))return 0;
        g_idx_keys_n[layer]=count;
        /* DSV4_IDX_COAL: the [dim][cap] copy, the whole prefix again (an
         * append is a few rows; the copy is a few hundred KB) */
        if(idx_coal_on()){
            if(!g_idx_keys_t[layer]&&
               !alloc_ok(cudaMalloc((void**)&g_idx_keys_t[layer],(size_t)g_idx_keys_cap[layer]*dim*4),"indexer key transpose"))
                g_idx_keys_t[layer]=nullptr;
            if(g_idx_keys_t[layer]){transpose_f32(g_idx_keys_t[layer],g_idx_keys[layer],count,dim,g_idx_keys_cap[layer],c->stream);g_idx_keys_t_n[layer]=count;}
        }}
    else if(idx_coal_on()&&g_idx_keys_t[layer]&&g_idx_keys_t_n[layer]<count){
        transpose_f32(g_idx_keys_t[layer],g_idx_keys[layer],count,dim,g_idx_keys_cap[layer],c->stream);g_idx_keys_t_n[layer]=count;}
    return 1;
}

/* The fused selection's launches over device pointers (the resident weights
 * and keys of `layer`): the query projection into dq [tokens][rows], the prep
 * in place, the head weights into dhw [tokens][heads], the scores into dsc
 * [tokens][count]. ev (optional): the DSV4_IDX_KPROF events 1..3 between. */
static int idx_select_launch(Dev*c,int layer,float*dq,float*dhw,float*dsc,const float*dx,const float*din,const float*dcos,
        const float*dsin,const int*dcnt,int tokens,int heads,int dim,int rope_dim,int hidden,int count,float had_scale,
        float wscale,int rope_fma,int hw_fma,cudaEvent_t*ev){
    int cols=g_idx_cols[layer],rows=g_idx_rows[layer];size_t sm=(size_t)rows*4,xb=(size_t)tokens*cols*4;
    if(tokens==1){
        /* 64-thread blocks: 8192 rows as 128 blocks over the card's SMs
         * instead of 32 (each thread's serial chain is unchanged) */
        if(g_idx_packed[layer]&&idx_proj_warp()&&cols%128==0&&rows%8==0)
            fp8_ref_rows8_warp_kernel<<<rows/8,32,0,c->stream>>>(dq,g_idx_w[layer],g_idx_s[layer],dx,rows,cols);
        else if(g_idx_packed[layer])
            fp8_ref_matmul_rows8_t1_kernel<<<(rows+63)/64,64,0,c->stream>>>(dq,g_idx_w[layer],g_idx_s[layer],dx,rows,cols);
        else
            fp8_ref_matmul_t1_kernel<<<(rows+63)/64,64,0,c->stream>>>(dq,g_idx_w[layer],g_idx_s[layer],dx,rows,cols);
        if(!ok(cudaGetLastError(),"fused indexer projection launch"))return 0;
    }else{
        float*xt=(idx_coal_on()&&buf((void**)&c->p3,&c->p3cap,xb))?c->p3:nullptr;
        if(!fp8_ref_matmul_launch(c,dq,g_idx_w[layer],g_idx_s[layer],dx,xt,rows,cols,g_idx_packed[layer],tokens))return 0;
    }
    if(ev)cudaEventRecord(ev[1],c->stream);
    indexer_prep_kernel<<<dim3(heads,tokens),dim,(size_t)dim*4,c->stream>>>(dq,dcos,dsin,heads,dim,rope_dim,had_scale,rope_fma);
    if(!ok(cudaGetLastError(),"fused indexer prep launch"))return 0;
    if(ev)cudaEventRecord(ev[2],c->stream);
    size_t wsh=2*32*sizeof(uint4)+(size_t)hidden*sizeof(float);
    if(idx_headw_mode()==2&&hidden%HEADW_WCHUNK==0&&wsh<=48*1024&&!((uintptr_t)g_idx_hw[layer]&15)&&!((uintptr_t)din&15))
        indexer_headw_warp_kernel<<<dim3(heads,tokens),32,wsh,c->stream>>>(dhw,g_idx_hw[layer],din,heads,hidden,wscale,hw_fma);
    else if(idx_headw_mode()==1&&tokens==1&&hidden%HEADW_CHUNK==0&&heads*HEADW_LANES<=1024&&!((uintptr_t)g_idx_hw[layer]&15)&&!((uintptr_t)din&15))
        indexer_headw_lanes_kernel<<<1,heads*HEADW_LANES,(size_t)heads*HEADW_PAD,c->stream>>>(dhw,g_idx_hw[layer],din,heads,hidden,wscale,hw_fma);
    else
        indexer_headw_kernel<<<tokens,heads,0,c->stream>>>(dhw,g_idx_hw[layer],din,heads,hidden,wscale,hw_fma);
    if(!ok(cudaGetLastError(),"fused indexer head-weight launch"))return 0;
    if(ev)cudaEventRecord(ev[3],c->stream);
    size_t sm_small=((size_t)heads*(dim+1)+dim+heads)*4;
    if(tokens<=4&&sm_small<=48*1024){
        int K=idx_score_k();
        if(K>1)indexer_score_multi_kernel<<<dim3((count+K-1)/K,tokens),heads,sm_small,c->stream>>>(dsc,dq,g_idx_keys[layer],dhw,dcnt,heads,dim,count,K);
        else indexer_score_small_kernel<<<dim3(count,tokens),heads,sm_small,c->stream>>>(dsc,dq,g_idx_keys[layer],dhw,dcnt,heads,dim,count);
    }else{
        if(idx_coal_on()&&g_idx_keys_t[layer]){
            if(sm>48*1024&&cudaFuncSetAttribute(indexer_score_coal_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)!=cudaSuccess)return 0;
            indexer_score_coal_kernel<<<dim3((count+127)/128,tokens),128,sm,c->stream>>>(dsc,dq,g_idx_keys_t[layer],g_idx_keys_cap[layer],dhw,dcnt,heads,dim,count);
        }else{
            if(sm>48*1024&&cudaFuncSetAttribute(indexer_score_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)!=cudaSuccess)return 0;
            indexer_score_kernel<<<dim3((count+127)/128,tokens),128,sm,c->stream>>>(dsc,dq,g_idx_keys[layer],dhw,dcnt,heads,dim,count);
        }
    }
    return ok(cudaGetLastError(),"fused indexer score launch");
}


/* ---- Decode attention tail on the device (V4_WO_CHAIN) ---------------------
 * One token: the cached sparse attention (ring + compressed prefix, the
 * selected ordinals when the layer has an indexer) stays on the card for the
 * inverse RoPE + bf16 round, the grouped wo_a matvec (the decode's own mv
 * kernel: its result bitwise what the uploaded-activation call returns), the
 * bf16 round, and wo_b through the same mv kernel over its mirror (what the
 * engine's coli_fp8_matvec_ref dispatches to when the mirror exists), then
 * one download of the bf16-rounded hidden vector. Three syncs per layer
 * become one; the arithmetic is the separate calls' exactly. */
/* Per-head RoPE on the last rope_dim of each head's dim, bf16-rounded after:
 * the reference's coli_v4_rope_apply (sine * direction, direction +-1 exact)
 * + coli_bf16_round_array. fma_mode as the indexer prep. */
__global__ void attn_inv_rope_kernel(float*att,const float*cosv,const float*sinv,int dim,int rope_dim,int fma_mode,int inverse){
    int h=blockIdx.x,i=threadIdx.x,pairs=rope_dim/2;
    if(i>=pairs)return;
    float*v=att+(long long)h*dim+dim-rope_dim+2*i;
    float re=v[0],im=v[1],c=cosv[i],sn=inverse?-sinv[i]:sinv[i];
    float a,b;
    if(fma_mode==1){a=fmaf(re,c,-__fmul_rn(im,sn));b=fmaf(im,c,__fmul_rn(re,sn));}
    else if(fma_mode==2){a=fmaf(-im,sn,__fmul_rn(re,c));b=fmaf(re,sn,__fmul_rn(im,c));}
    else{a=__fsub_rn(__fmul_rn(re,c),__fmul_rn(im,sn));b=__fadd_rn(__fmul_rn(re,sn),__fmul_rn(im,c));}
    v[0]=bf16_round_dev(a);v[1]=bf16_round_dev(b);
}
__global__ void bf16_round_ref_kernel(float*x,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)x[i]=bf16_round_dev(x[i]);
}
/* The decode query's per-head rmsnorm, the reference's arithmetic: one
 * sequential fp32 mean-square (fma_on: the accumulate fused, as the GCC
 * build contracts it), scale = 1 / sqrtf(ms / dim + eps), bf16 round. */
__global__ void head_rmsnorm_ref_kernel(float*q,int dim,float eps,int fma_on){
    int h=blockIdx.x;
    float*v=q+(long long)h*dim;
    extern __shared__ float sv[];
    __shared__ float sc;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)sv[i]=v[i];
    __syncthreads();
    if(!threadIdx.x){
        float ms=0.f;
        if(fma_on)for(int i=0;i<dim;i++)ms=fmaf(sv[i],sv[i],ms);
        else for(int i=0;i<dim;i++)ms=__fadd_rn(ms,__fmul_rn(sv[i],sv[i]));
        sc=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(__fdiv_rn(ms,(float)dim),eps)));
    }
    __syncthreads();
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)v[i]=bf16_round_dev(__fmul_rn(sv[i],sc));
}
/* The one-token cached sparse attention launch over device pointers (the
 * split kernels, else v2, else the first kernels): out [heads][dim] from
 * q, the ring and comp cache of `layer`, chunk (the token's kv row), the
 * sinks, meta {wo, wn, cn}, sel (the selected ordinals, or NULL for the
 * first cn comp rows). */
static int attn_one_launch(Dev*c,int layer,size_t sm,float*out,const float*q,const float*chunk,const float*sink,const int*meta,
        const int*sel,int cn,int abs_base,int chunk_start,int window,int heads,int dim,float scale,int total,int dynab){
    int split=sattn_split_on()&&sm*4<=48*1024&&buf((void**)&c->ssc,&c->ssccap,(size_t)heads*sm);
    if(dynab&&!split)return 0;   /* the graph path: the split kernels only (their shapes are the token-free ones) */
    if(split&&sel)
        launch_sattn_split<1>(c->ssc,heads,sm,c->stream,out,q,g_kv_ring[layer],chunk,g_kv_comp[layer],sink,meta,sel,
            abs_base,chunk_start,window,dim,scale,total,dynab);
    else if(split)
        launch_sattn_split<0>(c->ssc,heads,sm,c->stream,out,q,g_kv_ring[layer],chunk,g_kv_comp[layer],sink,meta,nullptr,
            abs_base,chunk_start,window,dim,scale,total,dynab);
    else if(sel&&sattn_v2_on())
        launch_sattn_v2<1>(heads,1,sm,c->stream,out,q,g_kv_ring[layer],chunk,g_kv_comp[layer],sink,meta,sel,
            cn,abs_base,chunk_start,window,dim,scale);
    else if(sel)
        sparse_attn_batch_cached_idx_kernel<<<dim3(heads,1),256,sm,c->stream>>>(
            out,q,g_kv_ring[layer],chunk,g_kv_comp[layer],sink,meta,sel,cn,abs_base,chunk_start,window,heads,dim,scale);
    else if(sattn_v2_on())
        launch_sattn_v2<0>(heads,1,sm,c->stream,out,q,g_kv_ring[layer],chunk,g_kv_comp[layer],sink,meta,nullptr,0,
            abs_base,chunk_start,window,dim,scale);
    else
        sparse_attn_batch_cached_kernel<<<dim3(heads,1),256,sm,c->stream>>>(
            out,q,g_kv_ring[layer],chunk,g_kv_comp[layer],sink,meta,abs_base,chunk_start,window,heads,dim,scale);
    return ok(cudaGetLastError(),"decode chain attention launch");
}
extern "C" int dsv4_cuda_attn_decode_chain(int device,int layer,const float*q,const float*chunk,int chunk_start,
        const float*sinks,const int*meta,const int*sel,int selstride,int abs_base,int comp_limit,int heads,int dim,
        float scale,const float*icos,const float*isin,int rope_dim,int rope_fma,Dsv4CudaTensor*wa,int groups,
        Dsv4CudaTensor*wb,float*out,float*att_out,float*oa_out,
        const float*qa,int q_rank,Dsv4CudaTensor*wqb,float eps,int norm_fma,float*q_out){
    Dev*c=ctx(device);
    if(qa&&(!wqb||wqb->device!=device||(wqb->fmt!=8&&wqb->fmt!=9)||wqb->I!=q_rank||(long long)wqb->O!=(long long)heads*dim||q_rank<1))return 0;
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!g_kv_ring[layer]||(!q&&!qa)||!chunk||!sinks||!meta||!out||!icos||!isin||
       !wa||!wb||wa->device!=device||wb->device!=device||groups<1||wa->O%groups||(long long)wa->I*groups!=(long long)heads*dim||
       wb->I!=wa->O||(wb->fmt!=8&&wb->fmt!=9)||(wa->fmt!=8&&wa->fmt!=9)||
       abs_base<0||chunk_start<0||heads<1||dim<1||rope_dim<2||rope_dim>dim||(rope_dim&1)||!(scale>0.f))return 0;
    if(comp_limit>0&&(!g_kv_comp[layer]||g_kv_comp_cap[layer]<comp_limit))return 0;
    int window=g_kv_ring_rows[layer],wn=meta[1],cn=meta[2],total=wn+cn,pairs=rope_dim/2;
    if(meta[0]<0||wn<1||wn>window||cn<0||cn>comp_limit)return 0;
    if(abs_base+meta[0]+wn>chunk_start+1)return 0;
    if(sel){if(selstride<1||cn>selstride)return 0;for(int i=0;i<cn;i++){int o=sel[i];if(o<0||o>=comp_limit)return 0;}}
    size_t qb=(size_t)heads*dim*4,cb=(size_t)dim*4,sb=(size_t)heads*4,mb=3*sizeof(int),
           ib=sel?(size_t)cn*sizeof(int):0,rb=(size_t)pairs*4,sm=(size_t)total*4,
           ob=(size_t)wa->O*4,hb=(size_t)wb->O*4,qab=qa?(size_t)q_rank*4:0;
    if(sm>96*1024||!ok(cudaSetDevice(device),"select decode chain device"))return 0;
    if(!buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p4,&c->p4cap,qb)||!buf((void**)&c->p3,&c->p3cap,cb)||
       !buf((void**)&c->aux1,&c->aux1cap,sb)||!buf((void**)&c->aux2,&c->aux2cap,mb)||
       (ib&&!buf((void**)&c->aux3,&c->aux3cap,ib))||!buf((void**)&c->p1,&c->p1cap,2*rb+qab)||
       !buf((void**)&c->dy,&c->ycap,ob)||!buf((void**)&c->dx,&c->xcap,hb))return 0;
    /* DSV4_CHAIN_PACK (1): the call's inputs (cos, sin, qa or q, the kv
     * row, the sinks, meta, the selection) as one host block and one
     * upload. Each pageable cudaMemcpyAsync costs the host ~15 us to
     * stage, and the three before the wq_b launch left the stream idle
     * for 0.047 ms per layer (DSV4_WO_KPROF qup=; the four after it hid
     * behind the kernel). The device block's regions stand in for the
     * separate buffers; the kernels see the same bytes. 0 = a copy per
     * input. */
    static int pack=-1;
    if(pack<0){const char*e=getenv("DSV4_CHAIN_PACK");pack=!(e&&*e=='0');}
    int pk=pack;
    const float*dcos=c->p1,*dsin=(const float*)((char*)c->p1+rb),*dqa=qa?(const float*)((char*)c->p1+2*rb):nullptr,
               *dchunk=c->p3,*dsink=c->aux1,*dq=c->p2;
    const int*dmeta=(const int*)c->aux2,*dsel=sel?(const int*)c->aux3:nullptr;
    if(pk){
        const size_t A=256;
        size_t o=0,ocos=o;o+=(rb+A-1)/A*A;size_t osin=o;o+=(rb+A-1)/A*A;size_t oqa=o;o+=qa?(qab+A-1)/A*A:(qb+A-1)/A*A;
        size_t ochunk=o;o+=(cb+A-1)/A*A;size_t osink=o;o+=(sb+A-1)/A*A;size_t ometa=o;o+=(mb+A-1)/A*A;size_t osel=o;o+=(ib+A-1)/A*A;
        if(c->chpcap<o){free(c->chp);c->chp=malloc(o);c->chpcap=c->chp?o:0;}
        if(!c->chp||!buf((void**)&c->dchp,&c->dchpcap,o))pk=0;
        else{
            char*h=(char*)c->chp;
            memcpy(h+ocos,icos,rb);memcpy(h+osin,isin,rb);
            if(qa)memcpy(h+oqa,qa,qab);else memcpy(h+oqa,q,qb);
            memcpy(h+ochunk,chunk,cb);memcpy(h+osink,sinks,sb);memcpy(h+ometa,meta,mb);if(ib)memcpy(h+osel,sel,ib);
            if(!ok(cudaMemcpyAsync(c->dchp,h,o,cudaMemcpyHostToDevice,c->stream),"decode chain packed upload"))return 0;
            const char*d=(const char*)c->dchp;
            dcos=(const float*)(d+ocos);dsin=(const float*)(d+osin);
            if(qa)dqa=(const float*)(d+oqa);else dq=(const float*)(d+oqa);
            dchunk=(const float*)(d+ochunk);dsink=(const float*)(d+osink);dmeta=(const int*)(d+ometa);if(ib)dsel=(const int*)(d+osel);
        }
    }
    static int kprof=-1;
    if(kprof<0){const char*e=getenv("DSV4_WO_KPROF");kprof=e&&*e&&*e!='0';}
    cudaEvent_t ev[7],ex[5];
    if(kprof){for(int i=0;i<7;i++)cudaEventCreate(&ev[i]);for(int i=0;i<5;i++)cudaEventCreate(&ex[i]);cudaEventRecord(ex[0],c->stream);cudaEventRecord(ex[3],c->stream);cudaEventRecord(ex[4],c->stream);}
    if(qa){
        /* q from qa on the card: the wq_b mv kernel (what coli_fp8_matvec_ref
         * dispatches to over the mirror), bf16 round, per-head rmsnorm,
         * forward RoPE + bf16 round — the host's q, never crossing. */
        if(!pk&&(!ok(cudaMemcpyAsync((void*)dqa,qa,qab,cudaMemcpyHostToDevice,c->stream),"decode chain qa upload")||
           !ok(cudaMemcpyAsync(c->p1,icos,rb,cudaMemcpyHostToDevice,c->stream),"decode chain cos upload")||
           !ok(cudaMemcpyAsync((char*)c->p1+rb,isin,rb,cudaMemcpyHostToDevice,c->stream),"decode chain sin upload")))return 0;
        if(kprof)cudaEventRecord(ex[3],c->stream);
        {
        if(wqb->fmt==9)run_mv<9>((uint8_t*)wqb->w,wqb->scale,dqa,c->p2,wqb->O,wqb->I,1,c->stream);
        else run_mv<8>((uint8_t*)wqb->w,wqb->scale,dqa,c->p2,wqb->O,wqb->I,1,c->stream);}
        if(!ok(cudaGetLastError(),"decode chain wq_b launch"))return 0;
        if(kprof)cudaEventRecord(ex[4],c->stream);
        bf16_round_ref_kernel<<<(heads*dim+255)/256,256,0,c->stream>>>(c->p2,heads*dim);
        head_rmsnorm_ref_kernel<<<heads,256,(size_t)dim*4,c->stream>>>(c->p2,dim,eps,norm_fma);
        if(!ok(cudaGetLastError(),"decode chain q norm launch"))return 0;
        attn_inv_rope_kernel<<<heads,pairs,0,c->stream>>>(c->p2,dcos,dsin,dim,rope_dim,rope_fma,0);
        if(!ok(cudaGetLastError(),"decode chain q rope launch"))return 0;
        if(q_out&&!ok(cudaMemcpyAsync(q_out,c->p2,qb,cudaMemcpyDeviceToHost,c->stream),"decode chain q download"))return 0;
    }
    if(kprof)cudaEventRecord(ex[1],c->stream);
    if(!pk&&!qa&&!ok(cudaMemcpyAsync(c->p2,q,qb,cudaMemcpyHostToDevice,c->stream),"decode chain query upload"))return 0;
    if(!pk&&(
       !ok(cudaMemcpyAsync(c->p3,chunk,cb,cudaMemcpyHostToDevice,c->stream),"decode chain kv upload")||
       !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"decode chain sink upload")||
       !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"decode chain meta upload")||
       (ib&&!ok(cudaMemcpyAsync(c->aux3,sel,ib,cudaMemcpyHostToDevice,c->stream),"decode chain index upload"))||
       (!qa&&(!ok(cudaMemcpyAsync(c->p1,icos,rb,cudaMemcpyHostToDevice,c->stream),"decode chain cos upload")||
              !ok(cudaMemcpyAsync((char*)c->p1+rb,isin,rb,cudaMemcpyHostToDevice,c->stream),"decode chain sin upload")))))return 0;
    if(kprof)cudaEventRecord(ex[2],c->stream);
    if(!attn_one_launch(c,layer,sm,c->p4,dq,dchunk,dsink,dmeta,dsel,cn,abs_base,chunk_start,window,heads,dim,scale,total,0))return 0;
    if(kprof)cudaEventRecord(ev[0],c->stream);
    attn_inv_rope_kernel<<<heads,pairs,0,c->stream>>>(c->p4,dcos,dsin,dim,rope_dim,rope_fma,1);
    if(!ok(cudaGetLastError(),"decode chain rope launch"))return 0;
    if(kprof)cudaEventRecord(ev[1],c->stream);
    {
    if(wa->fmt==9)run_mv<9>((uint8_t*)wa->w,wa->scale,c->p4,c->dy,wa->O,wa->I,groups,c->stream);
    else run_mv<8>((uint8_t*)wa->w,wa->scale,c->p4,c->dy,wa->O,wa->I,groups,c->stream);}
    if(!ok(cudaGetLastError(),"decode chain wo_a launch"))return 0;
    if(kprof)cudaEventRecord(ev[2],c->stream);
    bf16_round_ref_kernel<<<(wa->O+255)/256,256,0,c->stream>>>(c->dy,wa->O);
    if(oa_out&&!ok(cudaMemcpyAsync(oa_out,c->dy,ob,cudaMemcpyDeviceToHost,c->stream),"decode chain oa download"))return 0;
    if(kprof)cudaEventRecord(ev[3],c->stream);
    {
    if(wb->fmt==9)run_mv<9>((uint8_t*)wb->w,wb->scale,c->dy,c->dx,wb->O,wb->I,1,c->stream);
    else run_mv<8>((uint8_t*)wb->w,wb->scale,c->dy,c->dx,wb->O,wb->I,1,c->stream);}
    if(!ok(cudaGetLastError(),"decode chain wo_b launch"))return 0;
    bf16_round_ref_kernel<<<(wb->O+255)/256,256,0,c->stream>>>(c->dx,wb->O);
    if(kprof)cudaEventRecord(ev[4],c->stream);
    if(!ok(cudaMemcpyAsync(out,c->dx,hb,cudaMemcpyDeviceToHost,c->stream),"decode chain output download"))return 0;
    if(att_out&&!ok(cudaMemcpyAsync(att_out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"decode chain attended download"))return 0;
    if(kprof)cudaEventRecord(ev[5],c->stream);
    int done=ok(cudaStreamSynchronize(c->stream),"decode chain sync");
    if(kprof){
        float ms[5],mq=0,mu=0,ma=0,mqu=0,mqm=0;
        for(int i=0;i<5;i++)cudaEventElapsedTime(&ms[i],ev[i],ev[i+1]);
        cudaEventElapsedTime(&mq,ex[0],ex[1]);cudaEventElapsedTime(&mu,ex[1],ex[2]);cudaEventElapsedTime(&ma,ex[2],ev[0]);
        if(qa){cudaEventElapsedTime(&mqu,ex[0],ex[3]);cudaEventElapsedTime(&mqm,ex[3],ex[4]);}
        fprintf(stderr,"wokprof layer=%d q=%.3f (qup=%.3f qmv=%.3f) up=%.3f attn=%.3f rope=%.3f wo_a=%.3f round=%.3f wo_b=%.3f down=%.3f ms rows=%d\n",layer,mq,mqu,mqm,mu,ma,ms[0],ms[1],ms[2],ms[3],ms[4],total);
        for(int i=0;i<7;i++)cudaEventDestroy(ev[i]);for(int i=0;i<5;i++)cudaEventDestroy(ex[i]);
    }
    return done;
}

/* ---- V4_LAYER_FUSED: a decode token's attention layer in one round trip ----
 * The three trips a layer took (the input's projections; the indexer's
 * selection; the chain) and the host work between them (the qa and kv
 * norms, the kv's RoPE and fp8 round trip, the query ranks' fp8 round
 * trip for the indexer, the selection's rank) run on the stream from one
 * upload (x, cos/sin, the sinks, meta, the two norm weights) to one
 * download (out, the kv row, qa and the compressor projections the host's
 * canonical state still needs). The token's kv row enters the ring on the
 * device after the attention. Every kernel is the separate path's, or the
 * host routine's arithmetic in its order: bitwise (V4_LAYER_FUSED_VERIFY).
 * Not taken on a token that emits a compressed row (the host's compressor
 * state produces it) nor on a layer's first token. */
__device__ __forceinline__ float e4m3fn_rt_dev(float v){
    /* coli_e4m3fn_decode(coli_e4m3fn_encode(v)): round to nearest even on
     * the binary32 bits, subnormals multiples of 2^-9, saturation at 448 */
    if(isnan(v))return NAN;
    unsigned b=__float_as_uint(v);int neg=b>>31;float m=fabsf(v),num;
    if(m==0.f)num=0.f;
    else if(m>=448.f)num=448.f;
    else if(m<0.015625f){float sc=__fmul_rn(m,512.f);unsigned r=(unsigned)sc;float fr=__fsub_rn(sc,(float)r);
        if(fr>0.5f||(fr==0.5f&&(r&1u)))r++;num=ldexpf((float)r,-9);}
    else{unsigned bits=__float_as_uint(m);int ex=(int)((bits>>23)&0xffu)-127;unsigned sig=0x800000u|(bits&0x7fffffu);
        unsigned r=sig>>20,rem=sig&0xfffffu;if(rem>0x80000u||(rem==0x80000u&&(r&1u)))r++;if(r==16u){r=8u;ex++;}
        num=ldexpf(1.0f+(float)(r-8u)*0.125f,ex);}
    return neg?-num:num;
}
/* coli_fp8_activation_qdq_ref: per block of `block` values one thread runs
 * the sequential max, the e8m0 scale (ceil log2 of max/448 by frexpf) and
 * each value's clamp, e4m3 round trip and rescale; round = the bf16 round
 * the kv and compressor paths apply after. out may alias in. */
__device__ __forceinline__ void fp8_qdq_block(float*out,const float*in,int n,int block,int b,int round){
    int base=b*block;if(base>=n)return;
    int count=n-base<block?n-base:block;
    float mx=0.f;for(int i=0;i<count;i++)mx=fmaxf(mx,fabsf(in[base+i]));
    mx=fmaxf(mx,1e-4f);
    int e;float fr=frexpf(__fdiv_rn(mx,448.f),&e);int ex=(fr==0.5f)?e-1:e;if(ex<-127)ex=-127;if(ex>127)ex=127;
    float sc=ldexpf(1.0f,ex);
    for(int i=0;i<count;i++){float nz=fmaxf(-448.f,fminf(448.f,__fdiv_rn(in[base+i],sc)));
        float q=__fmul_rn(e4m3fn_rt_dev(nz),sc);out[base+i]=round?bf16_round_dev(q):q;}
}
/* one block's rmsnorm of v (bf16 round, the sequential mean square, the
 * scale, (x * inverse_rms) * w, bf16 round), sv the block's shared copy */
__device__ __forceinline__ void norm_block(float*v,const uint16_t*w,int n,float eps,int fma_on,float*sv,float*sc){
    for(int i=threadIdx.x;i<n;i+=blockDim.x)sv[i]=bf16_round_dev(v[i]);
    __syncthreads();
    if(!threadIdx.x){float ms=0.f;
        if(fma_on)for(int i=0;i<n;i++)ms=fmaf(sv[i],sv[i],ms);
        else for(int i=0;i<n;i++)ms=__fadd_rn(ms,__fmul_rn(sv[i],sv[i]));
        *sc=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(__fdiv_rn(ms,(float)n),eps)));}
    __syncthreads();
    float s=*sc;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)
        v[i]=bf16_round_dev(__fmul_rn(__fmul_rn(sv[i],s),__uint_as_float((unsigned)w[i]<<16)));
}
/* pair i of the RoPE on the last rope_dim of v (attn_inv_rope_kernel's arithmetic) */
__device__ __forceinline__ void rope_pair(float*v,const float*cosv,const float*sinv,int dim,int rope_dim,int i,int fma_mode,int inverse){
    float*p=v+dim-rope_dim+2*i;
    float re=p[0],im=p[1],c=cosv[i],sn=inverse?-sinv[i]:sinv[i];
    float a,b;
    if(fma_mode==1){a=fmaf(re,c,-__fmul_rn(im,sn));b=fmaf(im,c,__fmul_rn(re,sn));}
    else if(fma_mode==2){a=fmaf(-im,sn,__fmul_rn(re,c));b=fmaf(re,sn,__fmul_rn(im,c));}
    else{a=__fsub_rn(__fmul_rn(re,c),__fmul_rn(im,sn));b=__fadd_rn(__fmul_rn(re,sn),__fmul_rn(im,c));}
    p[0]=bf16_round_dev(a);p[1]=bf16_round_dev(b);
}
/* The host's work between the projections and the chain in one launch:
 * block 0 the qa norm (then the query ranks' fp8 round trip into xq for
 * the indexer when xq_on), block 1 the kv norm, its RoPE, the fp8 round
 * trip of its first dim-rope_dim values with the bf16 round, and the row's
 * copy into its ring slot (the slot of position-window, which no row of
 * this token's window reads). */
struct LfCs{float*kv,*sc;const float*ape;const uint16_t*norm;const float*proj_kv,*proj_gate;float*cache;int rows,proj,dim,ratio,fp4;};
__global__ void lf_prep_kernel(float*qa,const uint16_t*qnw,int q_rank,float*xq,int xq_on,float*kv,const uint16_t*kvw,int dim,
        float eps,int fma_on,const float*cosv,const float*sinv,int rope_dim,int rope_fma,float*ring,const int*mp,LfCs c0,LfCs c1){
    extern __shared__ float sv[];__shared__ float sc;
    float*ring_row=ring+(size_t)mp[6]*dim;
    if(blockIdx.x>=2){
        /* the compressor's state row for this token: kv_row = the kv
         * projection, score_row = the gate projection + ape[slot]
         * (coli_v4_compressor_advance), slot = position % ratio */
        LfCs k=blockIdx.x==2?c0:c1;
        int slot=mp[9],row=k.ratio==4?k.ratio+slot:slot;
        for(int i=threadIdx.x;i<k.proj;i+=blockDim.x){
            k.kv[(size_t)row*k.proj+i]=k.proj_kv[i];
            k.sc[(size_t)row*k.proj+i]=__fadd_rn(k.proj_gate[i],k.ape[(size_t)slot*k.proj+i]);
        }
        return;
    }
    if(blockIdx.x==0){
        norm_block(qa,qnw,q_rank,eps,fma_on,sv,&sc);
        __syncthreads();
        if(xq_on)fp8_qdq_block(xq,qa,q_rank,128,threadIdx.x,0);
    }else{
        norm_block(kv,kvw,dim,eps,fma_on,sv,&sc);
        __syncthreads();
        if(threadIdx.x<rope_dim/2)rope_pair(kv,cosv,sinv,dim,rope_dim,threadIdx.x,rope_fma,0);
        __syncthreads();
        fp8_qdq_block(kv,kv,dim-rope_dim,64,threadIdx.x,1);
        __syncthreads();
        for(int i=threadIdx.x;i<dim;i+=blockDim.x)ring_row[i]=kv[i];
    }
}
/* The compressor's emit (compressor_pool_and_emit) for compressor k =
 * blockIdx.x: per column the max over the pool rows (rows past ratio read
 * column + dim, the comparison form), the exponentials' total and the
 * weighted sum in row order (pool_fma: the weighted accumulate fused, as
 * the build contracts it), weighted / total, bf16 round; the rmsnorm with
 * the compressor's norm, bf16; the RoPE at position + 1 - ratio (ecos,
 * esin), bf16; then the fp8 round trip of the first dim - rope_dim values
 * (block 64) or, rotate_fp4, the hadamard + fp4 round trip of the whole
 * row (indexer_prep_kernel's), bf16; the row into the cache at its index
 * (meta[7] the attention's, [8] the indexer's); and, ratio 4, rows 4..7
 * of the state down to 0..3. */
__global__ void lf_emit_kernel(LfCs c0,LfCs c1,float eps,int fma_on,int pool_fma,const float*ecos,const float*esin,int rope_dim,int rope_fma,const int*mp){
    LfCs k=blockIdx.x==0?c0:c1;
    extern __shared__ float s[];__shared__ float bsc[32];__shared__ float scl;
    int dim=k.dim,proj=k.proj,ratio=k.ratio,c=threadIdx.x,pool_rows=ratio==4?2*ratio:ratio;
    if(c<dim){
        float maximum=-INFINITY;
        for(int r=0;r<pool_rows;r++){int col=(ratio==4&&r>=ratio)?c+dim:c;float score=k.sc[(size_t)r*proj+col];if(score>maximum)maximum=score;}
        float total=0.f,weighted=0.f;
        for(int r=0;r<pool_rows;r++){int col=(ratio==4&&r>=ratio)?c+dim:c;
            float weight=expf(__fsub_rn(k.sc[(size_t)r*proj+col],maximum));
            total=__fadd_rn(total,weight);
            float v=k.kv[(size_t)r*proj+col];
            weighted=pool_fma?fmaf(v,weight,weighted):__fadd_rn(weighted,__fmul_rn(v,weight));}
        s[c]=bf16_round_dev(__fdiv_rn(weighted,total));
    }
    __syncthreads();
    if(!c){float ms=0.f;
        if(fma_on)for(int i=0;i<dim;i++)ms=fmaf(s[i],s[i],ms);
        else for(int i=0;i<dim;i++)ms=__fadd_rn(ms,__fmul_rn(s[i],s[i]));
        scl=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(__fdiv_rn(ms,(float)dim),eps)));}
    __syncthreads();
    if(c<dim)s[c]=bf16_round_dev(__fmul_rn(__fmul_rn(s[c],scl),__uint_as_float((unsigned)k.norm[c]<<16)));
    __syncthreads();
    if(c<rope_dim/2)rope_pair(s,ecos,esin,dim,rope_dim,c,rope_fma,0);
    __syncthreads();
    if(k.fp4){
        for(int width=1;width<dim;width*=2){
            if(c<dim/2){int base=(c/width)*2*width,j=c-(c/width)*width;float l=s[base+j],r=s[base+width+j];
                s[base+j]=__fadd_rn(l,r);s[base+width+j]=__fsub_rn(l,r);}
            __syncthreads();}
        float had_scale=__fdiv_rn(1.0f,__fsqrt_rn((float)dim));
        if(c<dim)s[c]=bf16_round_dev(__fmul_rn(s[c],had_scale));
        __syncthreads();
        if(c<dim/32){float m=0.f;for(int q=0;q<32;q++)m=fmaxf(m,fabsf(s[c*32+q]));
            m=fmaxf(m,6.0f*ldexpf(1.0f,-126));int e;float fr=frexpf(__fdiv_rn(m,6.0f),&e);int ex=(fr==0.5f)?e-1:e;
            if(ex<-127)ex=-127;if(ex>127)ex=127;bsc[c]=ldexpf(1.0f,ex);}
        __syncthreads();
        if(c<dim){float sc4=bsc[c>>5];float v=fmaxf(-6.0f,fminf(6.0f,__fdiv_rn(s[c],sc4)));
            int best=0;float d=fabsf(__fsub_rn(v,c_idx_e2m1[0]));
            for(int code=1;code<16;code++){float cd=fabsf(__fsub_rn(v,c_idx_e2m1[code]));if(cd<d){d=cd;best=code;}}
            s[c]=bf16_round_dev(__fmul_rn(c_idx_e2m1[best],sc4));}
    }else{
        fp8_qdq_block(s,s,dim-rope_dim,64,c,1);
    }
    __syncthreads();
    int index=blockIdx.x==0?mp[7]:mp[8];
    if(c<dim)k.cache[(size_t)index*dim+c]=s[c];
    if(ratio==4&&c<dim){
        for(int r=0;r<4;r++){
            k.kv[(size_t)r*proj+c]=k.kv[(size_t)(r+4)*proj+c];k.kv[(size_t)r*proj+c+dim]=k.kv[(size_t)(r+4)*proj+c+dim];
            k.sc[(size_t)r*proj+c]=k.sc[(size_t)(r+4)*proj+c];k.sc[(size_t)r*proj+c+dim]=k.sc[(size_t)(r+4)*proj+c+dim];}
    }
}
/* The query's per-head rmsnorm (head_rmsnorm_ref_kernel) and forward RoPE
 * in one launch; the bf16 round before them is the wq_b store's. */
__global__ void q_norm_rope_kernel(float*q,int dim,float eps,int fma_on,const float*cosv,const float*sinv,int rope_dim,int rope_fma){
    int h=blockIdx.x;float*v=q+(long long)h*dim;
    extern __shared__ float sv[];__shared__ float sc;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)sv[i]=v[i];
    __syncthreads();
    if(!threadIdx.x){float ms=0.f;
        if(fma_on)for(int i=0;i<dim;i++)ms=fmaf(sv[i],sv[i],ms);
        else for(int i=0;i<dim;i++)ms=__fadd_rn(ms,__fmul_rn(sv[i],sv[i]));
        sc=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(__fdiv_rn(ms,(float)dim),eps)));}
    __syncthreads();
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)v[i]=bf16_round_dev(__fmul_rn(sv[i],sc));
    __syncthreads();
    if(threadIdx.x<rope_dim/2)rope_pair(v,cosv,sinv,dim,rope_dim,threadIdx.x,rope_fma,0);
}
/* The decode matvec over an fp8 mirror with the output's bf16 round folded
 * into the store (the warp kernel's rnd); the block kernel plus a round
 * pass where the warp kernel does not fit. */
template<int F> static void run_mv_fused(const uint8_t*w,const uint8_t*sc,const float*x,float*y,int O,int I,int groups,int rnd,cudaStream_t st){
    if(mv_warp_on()&&mv_warp_fits(w,x,I,8)){launch_mv_warp_u<F,1>(w,sc,x,y,O,I,groups,rnd,st);return;}
    run_mv<F>(w,sc,x,y,O,I,groups,st);
    if(rnd)bf16_round_ref_kernel<<<(O+255)/256,256,0,st>>>(y,O);
}
static void run_mv_fused_fmt(Dsv4CudaTensor*t,const float*x,float*y,int groups,int rnd,cudaStream_t st){
    if(t->fmt==9)run_mv_fused<9>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,groups,rnd,st);
    else run_mv_fused<8>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,groups,rnd,st);
}
/* The selection's rank (score descending, index ascending: the host's
 * descending_score order, its radix sort) as a bitonic sort of composite
 * keys (the host's order-preserving score key above the index) in shared
 * memory, one block, n <= IDX_SORT_MAX; sel[k] = the k-th ranked index for
 * k < min(n, topk): what the host's ranked list gives element for element. */
#define IDX_SORT_MAX 4096
__global__ void __launch_bounds__(1024) idx_topk_kernel(int*sel,const float*scores,int n,int topk,const int*mp){
    __shared__ unsigned long long k[IDX_SORT_MAX];
    if(mp){n=mp[3];topk=mp[2];}
    int N=1;while(N<n)N<<=1;
    for(int i=threadIdx.x;i<N;i+=blockDim.x){
        unsigned long long v=~0ull;
        if(i<n){unsigned u=__float_as_uint(scores[i]);u=(u&0x80000000u)?~u:(u|0x80000000u);
            v=((unsigned long long)(~u)<<32)|(unsigned)i;}
        k[i]=v;}
    __syncthreads();
    for(int size=2;size<=N;size<<=1)for(int stride=size>>1;stride>0;stride>>=1){
        for(int i=threadIdx.x;i<N;i+=blockDim.x){int j=i^stride;
            if(j>i){bool up=((i&size)==0);unsigned long long a=k[i],b=k[j];if((a>b)==up){k[i]=b;k[j]=a;}}}
        __syncthreads();}
    int m=n<topk?n:topk;
    for(int i=threadIdx.x;i<m;i+=blockDim.x)sel[i]=(int)(k[i]&0xffffffffu);
}
static inline size_t lf_al(size_t v){return (v+255)/256*256;}

/* ---- V4_GPU_HEAD: the output head on the device, the CPU's arithmetic ----
 * head_bf16_dot sums one row's 4096 products (bf16 weight exact in fp32,
 * times the fp32 hidden, each product rounded) in column order, one
 * sequential fp32 chain. A warp per row: lane l holds the eight products
 * of columns c + 8l .. c + 8l + 7 of each 256-column chunk (one 16-byte
 * load; the next chunk's load in flight), lane 0 adds them in column order
 * as the lanes shuffle them over. The argmax is the first maximum in
 * vocabulary order (head_scores_argmax: strictly greater wins). */
__global__ void __launch_bounds__(256) head_dot_warp_kernel(float*scores,const __nv_bfloat16*w,const float*x,int vocab,int d){
    extern __shared__ float xs[];
    for(int i=threadIdx.x;i<d;i+=blockDim.x)xs[i]=x[i];
    __syncthreads();
    int lane=threadIdx.x&31,row=blockIdx.x*8+(threadIdx.x>>5);
    if(row>=vocab)return;
    const __nv_bfloat16*wr=w+(long long)row*d;
    float sum=0.f;
    uint4 cur=*(const uint4*)(wr+8*lane);
    for(int c=0;c<d;c+=256){
        uint4 nxt=make_uint4(0u,0u,0u,0u);
        if(c+256<d)nxt=*(const uint4*)(wr+c+256+8*lane);
        unsigned u[4]={cur.x,cur.y,cur.z,cur.w};
        float p[8];
        #pragma unroll
        for(int k=0;k<8;k++){float wv=__uint_as_float((k&1)?(u[k>>1]&0xffff0000u):(u[k>>1]<<16));p[k]=__fmul_rn(wv,xs[c+8*lane+k]);}
        #pragma unroll
        for(int j=0;j<32;j++){
            #pragma unroll
            for(int k=0;k<8;k++){float v=__shfl_sync(0xffffffffu,p[k],j);sum=__fadd_rn(sum,v);}
        }
        cur=nxt;
    }
    if(!lane)scores[row]=sum;
}
__global__ void __launch_bounds__(1024) head_argmax_first_kernel(const float*scores,int vocab,int*id,float*value){
    __shared__ float bv[1024];__shared__ int bi[1024];
    float m=-3.402823466e+38f;int mi=-1;
    for(int i=threadIdx.x;i<vocab;i+=blockDim.x){float v=scores[i];if(v>m){m=v;mi=i;}}
    bv[threadIdx.x]=m;bi[threadIdx.x]=mi;
    __syncthreads();
    for(int n=512;n;n>>=1){
        if(threadIdx.x<n){float a=bv[threadIdx.x],b=bv[threadIdx.x+n];int ia=bi[threadIdx.x],ib=bi[threadIdx.x+n];
            if(b>a||(b==a&&ib>=0&&(ia<0||ib<ia))){bv[threadIdx.x]=b;bi[threadIdx.x]=ib;}}
        __syncthreads();}
    if(!threadIdx.x){*id=bi[0];*value=bv[0];}
}
/* DSV4_LF_GRAPH (1): the fused layer's launches captured once per layer
 * (and per emit / no-emit) into a CUDA graph and replayed per token (one
 * submission where ~20 launches each paid the WDDM launch gap). Everything
 * the captured nodes bake in is token-free: the buffers (a per-device
 * block laid out at the layer's maxima, the pinned host block), the grids
 * (sized for the most rows: the kernels stop at the token's counts), the
 * weights, caches and state mirrors (their pointers); the token's scalars
 * travel in the uploaded meta block (meta[4] abs_base, [5] chunk_start,
 * [6] the ring slot, [2] cn, [3] the candidate count, [7]/[8] the emitted
 * rows' indices, [9] the compressor slot). A graph is recaptured when any
 * baked pointer or shape changed (a grown cache, a grown block). 0 = the
 * launches issued directly. */
static int lf_graph_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_LF_GRAPH");on=!(e&&*e=='0');}return on;}
static int lf_pool_fma(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_CS_POOL_FMA");on=!(e&&*e=='0');}return on;}
struct LfSig{const void*lfd,*lfh,*ssc,*ring,*comp,*keys,*wqa,*wkv,*wqb,*wa,*wb,*cwkv,*cwg,*iwkv,*iwg,*iw,*is,*ihw,*cs[8];
    int hidden,q_rank,dim,heads,groups,waO,idx_score,idx_heads,idx_dim,window,max_total,cn_max,rope_dim,ncomp,nidx,cO,gO,ikO,igO,
        vnorm_fma,rope_fma,idx_rope_fma,hw_fma,norm_fma,ds,K,headw,unroll,emit,ratio,pool_fma;float eps,scale,had_scale,wscale;};
struct LfGraph{cudaGraphExec_t exec;cudaGraph_t graph;int valid;LfSig sig;};
static LfGraph g_lf[16][DSV4_KV_CACHE_LAYERS][2];
/* the compressors' state mirrors: [0] the attention's, [1] the indexer's */
static float *g_cs_kv[2][DSV4_KV_CACHE_LAYERS],*g_cs_sc[2][DSV4_KV_CACHE_LAYERS],*g_cs_ape[2][DSV4_KV_CACHE_LAYERS];
static uint16_t *g_cs_norm[2][DSV4_KV_CACHE_LAYERS];
static size_t g_cs_cap[2][DSV4_KV_CACHE_LAYERS],g_cs_apecap[2][DSV4_KV_CACHE_LAYERS],g_cs_normcap[2][DSV4_KV_CACHE_LAYERS];
static int g_cs_upto[DSV4_KV_CACHE_LAYERS];   /* position + 1 of the last state row the device wrote; 0 none */
static int lf_comp_reserve(Dev*c,int layer,int have,int need,int dim){
    if(g_kv_comp_cap[layer]>=need)return 1;
    int cap=g_kv_comp_cap[layer]?g_kv_comp_cap[layer]:512;
    while(cap<need)cap*=2;
    float*grown=NULL;
    if(!alloc_ok(cudaMalloc((void**)&grown,(size_t)cap*dim*4),"kv comp allocation"))return 0;
    if(g_kv_comp[layer]&&have&&
       !ok(cudaMemcpyAsync(grown,g_kv_comp[layer],(size_t)have*dim*4,cudaMemcpyDeviceToDevice,c->stream),"kv comp grow copy")){cudaFree(grown);return 0;}
    if(g_kv_comp[layer]){cudaStreamSynchronize(c->stream);cudaFree(g_kv_comp[layer]);}
    g_kv_comp[layer]=grown;g_kv_comp_cap[layer]=cap;
    return 1;
}
static int lf_keys_reserve(Dev*c,int layer,int need,int dim){
    if(g_idx_keys_cap[layer]>=need)return 1;
    int cap=g_idx_keys_cap[layer]?g_idx_keys_cap[layer]:512;
    while(cap<need)cap*=2;
    float*grown=NULL;
    if(!alloc_ok(cudaMalloc((void**)&grown,(size_t)cap*dim*4),"indexer key residency"))return 0;
    int have=g_idx_keys_n[layer];
    if(g_idx_keys[layer]&&have&&
       !ok(cudaMemcpyAsync(grown,g_idx_keys[layer],(size_t)have*dim*4,cudaMemcpyDeviceToDevice,c->stream),"indexer key grow copy")){cudaFree(grown);return 0;}
    if(g_idx_keys[layer]){cudaStreamSynchronize(c->stream);cudaFree(g_idx_keys[layer]);}
    g_idx_keys[layer]=grown;g_idx_keys_cap[layer]=cap;
    if(g_idx_keys_t[layer]){cudaFree(g_idx_keys_t[layer]);g_idx_keys_t[layer]=nullptr;g_idx_keys_t_n[layer]=0;}
    return 1;
}
extern "C" int dsv4_cuda_attn_decode_fused(const Dsv4FusedArgs*a){
    if(!a)return 0;
    int device=a->device,layer=a->layer,hidden=a->hidden,q_rank=a->q_rank,dim=a->dim,rope_dim=a->rope_dim,position=a->position,
        first=a->first,comp_total=a->comp_total,heads=a->heads,groups=a->groups,idx_score=a->idx_score,idx_count=a->idx_count,
        idx_sel=a->idx_sel,idx_heads=a->idx_heads,idx_dim=a->idx_dim,vnorm_fma=a->vnorm_fma,rope_fma=a->rope_fma,norm_fma=a->norm_fma,
        idx_rope_fma=a->idx_rope_fma,hw_fma=a->hw_fma,emit=a->emit,ratio=a->ratio;
    float eps=a->eps,scale=a->scale,had_scale=a->had_scale,wscale=a->wscale;
    const float*x=a->x,*icos=a->icos,*isin=a->isin,*sinks=a->sinks;
    Dsv4CudaTensor*wqa=a->wqa,*wkv=a->wkv,*cwkv=a->cwkv,*cwg=a->cwg,*iwkv=a->iwkv,*iwg=a->iwg,*wqb=a->wqb,*wa=a->wa,*wb=a->wb;
    const uint16_t*qnw=a->qnw,*kvw=a->kvw;
    float*out=a->out,*kv_out=a->kv_out,*qa_out=a->qa_out,*ckv_out=a->ckv_out,*cg_out=a->cg_out,*ikv_out=a->ikv_out,*ig_out=a->ig_out,*q_out=a->q_out;
    int*sel_out=a->sel_out;
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!x||hidden<1||!wqa||!wkv||!qnw||!kvw||!icos||!isin||!sinks||!wqb||!wa||!wb||
       !out||!kv_out||!qa_out||q_rank<1||q_rank>1024||dim<1||dim>1024||rope_dim<2||rope_dim>dim||(rope_dim&1)||position<0||
       first<0||first>position||heads<1||groups<1||!(scale>0.f)||!g_kv_ring[layer])return 0;
    if(wqa->device!=device||wkv->device!=device||wqb->device!=device||wa->device!=device||wb->device!=device)return 0;
    if((wqa->fmt!=8&&wqa->fmt!=9)||wqa->I!=hidden||wqa->O!=q_rank||(wkv->fmt!=8&&wkv->fmt!=9)||wkv->I!=hidden||wkv->O!=dim)return 0;
    if((wqb->fmt!=8&&wqb->fmt!=9)||wqb->I!=q_rank||(long long)wqb->O!=(long long)heads*dim)return 0;
    if((wa->fmt!=8&&wa->fmt!=9)||wa->O%groups||(long long)wa->I*groups!=(long long)heads*dim||
       (wb->fmt!=8&&wb->fmt!=9)||wb->I!=wa->O||wb->O!=hidden)return 0;
    int ncomp=(cwkv&&cwg)?2:0,nidx=(iwkv&&iwg)?2:0;
    if(ncomp&&(!ckv_out||!cg_out||cwkv->fmt!=16||cwg->fmt!=16||cwkv->I!=hidden||cwg->I!=hidden||cwkv->O>1024||cwg->O>1024||
              cwkv->device!=device||cwg->device!=device))return 0;
    if(nidx&&(!ikv_out||!ig_out||iwkv->fmt!=16||iwg->fmt!=16||iwkv->I!=hidden||iwg->I!=hidden||iwkv->O>1024||iwg->O>1024||
             iwkv->device!=device||iwg->device!=device))return 0;
    /* the compressors on the device: the state mirrors (resynced from the
     * host's rows when this token does not follow the last one written
     * here), ape and norm resident */
    int ncs=0;
    if(ncomp){
        if(!a->cs_kv[0]||!a->cs_sc[0]||!a->cs_ape[0]||!a->cs_norm[0]||a->cs_dim[0]!=dim||a->cs_proj[0]!=cwkv->O||a->cs_proj[0]!=cwg->O||
           a->cs_rows[0]<1||ratio<1||(ratio==4&&a->cs_rows[0]!=8)||(ratio!=4&&a->cs_rows[0]!=ratio))return 0;
        ncs=1;
        if(nidx){if(!a->cs_kv[1]||!a->cs_sc[1]||!a->cs_ape[1]||!a->cs_norm[1]||a->cs_dim[1]!=idx_dim||a->cs_proj[1]!=iwkv->O||
                    a->cs_proj[1]!=iwg->O||ratio!=4||a->cs_rows[1]!=8)return 0;ncs=2;}
    }else if(emit)return 0;
    if(emit&&(!a->ecos||!a->esin))return 0;
    if(emit&&nidx&&!idx_score)return 0;   /* the emitted key must be scored (the selection runs on the device) */
    int window=g_kv_ring_rows[layer],wn=position-first+1;
    int comp_after=comp_total+(emit?1:0),idx_after=idx_count+(emit&&nidx?1:0);
    if(wn<1||wn>window||comp_total<0||(comp_total&&!g_kv_comp[layer])||(!emit&&comp_total&&g_kv_comp_cap[layer]<comp_total))return 0;
    int cn=idx_sel;
    if(cn<0||cn>comp_after)return 0;
    if(idx_score){
        if(!g_idx_w[layer]||(!emit&&!g_idx_keys[layer])||idx_after<1||idx_after>IDX_SORT_MAX||idx_count>g_idx_keys_n[layer]||
           g_idx_keys_dim[layer]!=idx_dim||g_idx_rows[layer]!=idx_heads*idx_dim||g_idx_heads[layer]!=idx_heads||
           g_idx_hidden[layer]!=hidden||g_idx_cols[layer]!=q_rank||idx_heads<1||idx_heads>1024||idx_dim<32||idx_dim>1024||
           (idx_dim&(idx_dim-1))||cn>idx_after||cn<1)return 0;
    }
    int pairs=rope_dim/2;
    int cn_max=nidx?(cn>512?cn:512):(g_kv_comp_cap[layer]>comp_after?g_kv_comp_cap[layer]:comp_after);
    if(cn_max<cn)cn_max=cn;
    int max_total=window+cn_max;size_t sm_max=(size_t)max_total*4;
    int K=idx_score_k();size_t sm_small=((size_t)idx_heads*(idx_dim+1)+idx_dim+idx_heads)*4;
    int graph=lf_graph_on()&&!q_out&&sm_max*4<=48*1024&&sattn_split_on()&&!mv_warp_verify()&&
              (!idx_score||(K>1&&sm_small<=48*1024&&idx_dim<=1024));
    if(sm_max>96*1024||!ok(cudaSetDevice(device),"select fused layer device"))return 0;
    static int kprof=-1;
    if(kprof<0){const char*e=getenv("DSV4_LF_KPROF");kprof=e&&*e&&*e!='0';}
    static cudaEvent_t kev[16][8];static int kev_made[16];
    int di=(int)(c-g);
    if(kprof&&!kev_made[di]){for(int i=0;i<8;i++)cudaEventCreate(&kev[di][i]);kev_made[di]=1;}
    if(kprof)graph=0;   /* event records captured into the graph changed the output (md5) and timed nothing: direct launches only */
    /* the caches the emit appends to, grown before anything is captured */
    if(emit&&!lf_comp_reserve(c,layer,comp_total,comp_after,dim))return 0;
    if(emit&&nidx&&!lf_keys_reserve(c,layer,idx_after,idx_dim))return 0;
    if(g_kv_comp_cap[layer]<comp_after)return 0;
    LfCs cs[2];memset(cs,0,sizeof cs);
    int resync=g_cs_upto[layer]!=position;
    for(int k=0;k<ncs;k++){
        size_t n=(size_t)a->cs_rows[k]*a->cs_proj[k],nb=n*4,ab=(size_t)ratio*a->cs_proj[k]*4,wb_=(size_t)a->cs_dim[k]*2;
        if(g_cs_cap[k][layer]<nb){
            if(g_cs_kv[k][layer])cudaFree(g_cs_kv[k][layer]);if(g_cs_sc[k][layer])cudaFree(g_cs_sc[k][layer]);
            g_cs_kv[k][layer]=g_cs_sc[k][layer]=nullptr;g_cs_cap[k][layer]=0;
            if(!alloc_ok(cudaMalloc((void**)&g_cs_kv[k][layer],nb),"compressor state mirror")||
               !alloc_ok(cudaMalloc((void**)&g_cs_sc[k][layer],nb),"compressor score mirror"))return 0;
            g_cs_cap[k][layer]=nb;resync=1;}
        if(g_cs_apecap[k][layer]<ab){if(g_cs_ape[k][layer])cudaFree(g_cs_ape[k][layer]);g_cs_ape[k][layer]=nullptr;g_cs_apecap[k][layer]=0;
            if(!alloc_ok(cudaMalloc((void**)&g_cs_ape[k][layer],ab),"compressor ape mirror"))return 0;g_cs_apecap[k][layer]=ab;resync=1;}
        if(g_cs_normcap[k][layer]<wb_){if(g_cs_norm[k][layer])cudaFree(g_cs_norm[k][layer]);g_cs_norm[k][layer]=nullptr;g_cs_normcap[k][layer]=0;
            if(!alloc_ok(cudaMalloc((void**)&g_cs_norm[k][layer],wb_),"compressor norm mirror"))return 0;g_cs_normcap[k][layer]=wb_;resync=1;}
        if(resync){
            if(!ok(cudaMemcpyAsync(g_cs_kv[k][layer],a->cs_kv[k],nb,cudaMemcpyHostToDevice,c->stream),"compressor state upload")||
               !ok(cudaMemcpyAsync(g_cs_sc[k][layer],a->cs_sc[k],nb,cudaMemcpyHostToDevice,c->stream),"compressor score upload")||
               !ok(cudaMemcpyAsync(g_cs_ape[k][layer],a->cs_ape[k],ab,cudaMemcpyHostToDevice,c->stream),"compressor ape upload")||
               !ok(cudaMemcpyAsync(g_cs_norm[k][layer],a->cs_norm[k],wb_,cudaMemcpyHostToDevice,c->stream),"compressor norm upload"))return 0;
        }
        cs[k].kv=g_cs_kv[k][layer];cs[k].sc=g_cs_sc[k][layer];cs[k].ape=g_cs_ape[k][layer];cs[k].norm=g_cs_norm[k][layer];
        cs[k].rows=a->cs_rows[k];cs[k].proj=a->cs_proj[k];cs[k].dim=a->cs_dim[k];cs[k].ratio=ratio;cs[k].fp4=a->cs_fp4[k];
        cs[k].cache=k==0?g_kv_comp[layer]:g_idx_keys[layer];
    }
    if(resync&&ncs&&!ok(cudaStreamSynchronize(c->stream),"compressor state resync"))return 0;
    /* layout: the upload block, the download block, the scratch */
    size_t o=0;
    size_t ox=o;o+=lf_al((size_t)hidden*4);size_t ocos=o;o+=lf_al((size_t)pairs*4);size_t osin=o;o+=lf_al((size_t)pairs*4);
    size_t oecos=o;o+=lf_al((size_t)pairs*4);size_t oesin=o;o+=lf_al((size_t)pairs*4);
    size_t osink=o;o+=lf_al((size_t)heads*4);size_t ometa=o;o+=lf_al(16*sizeof(int));
    size_t oqnw=o;o+=lf_al((size_t)q_rank*2);size_t okvw=o;o+=lf_al((size_t)dim*2);
    size_t upb=o;
    size_t oout=o;o+=lf_al((size_t)hidden*4);size_t okv=o;o+=lf_al((size_t)dim*4);size_t oqa=o;o+=lf_al((size_t)q_rank*4);
    size_t ockv=o;o+=lf_al(ncomp?(size_t)cwkv->O*4:0);size_t ocg=o;o+=lf_al(ncomp?(size_t)cwg->O*4:0);
    size_t oikv=o;o+=lf_al(nidx?(size_t)iwkv->O*4:0);size_t oig=o;o+=lf_al(nidx?(size_t)iwg->O*4:0);
    size_t osel=o;o+=lf_al(nidx?(size_t)cn_max*4:0);
    size_t dlb=o-upb;
    size_t oxq=o;o+=lf_al((size_t)q_rank*4);size_t oiq=o;o+=lf_al(nidx?(size_t)idx_heads*idx_dim*4:0);
    size_t oihw=o;o+=lf_al(nidx?(size_t)idx_heads*4:0);size_t osc=o;o+=lf_al(nidx?(size_t)IDX_SORT_MAX*4:0);
    size_t oq=o;o+=lf_al((size_t)heads*dim*4);size_t oatt=o;o+=lf_al((size_t)heads*dim*4);size_t ooa=o;o+=lf_al((size_t)wa->O*4);
    size_t tot=o,hostb=upb+dlb;
    char*h;
    if(graph){
        if(c->lfhpcap<hostb){if(c->lfhp)cudaFreeHost(c->lfhp);c->lfhp=nullptr;c->lfhpcap=0;
            void*pm=nullptr;if(!alloc_ok(cudaHostAlloc(&pm,hostb,cudaHostAllocPortable),"fused layer pinned block"))return 0;
            c->lfhp=pm;c->lfhpcap=hostb;}
        h=(char*)c->lfhp;
    }else{
        if(c->lfhcap<hostb){free(c->lfh);c->lfh=malloc(hostb);c->lfhcap=c->lfh?hostb:0;if(!c->lfh)return 0;}
        h=(char*)c->lfh;
    }
    if(!buf((void**)&c->lfd,&c->lfdcap,tot))return 0;
    if(!buf((void**)&c->ssc,&c->ssccap,(size_t)heads*sm_max))return 0;
    char*d=(char*)c->lfd;
    memcpy(h+ox,x,(size_t)hidden*4);memcpy(h+ocos,icos,(size_t)pairs*4);memcpy(h+osin,isin,(size_t)pairs*4);
    if(emit){memcpy(h+oecos,a->ecos,(size_t)pairs*4);memcpy(h+oesin,a->esin,(size_t)pairs*4);}
    memcpy(h+osink,sinks,(size_t)heads*4);
    int meta[16]={0,wn,cn,idx_after,first,position,position%window,comp_total,idx_count,ratio?position%ratio:0,0,0,0,0,0,0};
    memcpy(h+ometa,meta,sizeof(meta));
    memcpy(h+oqnw,qnw,(size_t)q_rank*2);memcpy(h+okvw,kvw,(size_t)dim*2);
    const float*dx=(const float*)(d+ox),*dcos=(const float*)(d+ocos),*dsin=(const float*)(d+osin),*dsink=(const float*)(d+osink);
    const float*decos=(const float*)(d+oecos),*desin=(const float*)(d+oesin);
    const int*dmeta=(const int*)(d+ometa);const uint16_t*dqnw=(const uint16_t*)(d+oqnw),*dkvw=(const uint16_t*)(d+okvw);
    float*dout=(float*)(d+oout),*dkv=(float*)(d+okv),*dqa=(float*)(d+oqa),*dckv=(float*)(d+ockv),*dcg=(float*)(d+ocg),
         *dikv=(float*)(d+oikv),*dig=(float*)(d+oig),*dxq=(float*)(d+oxq),*diq=(float*)(d+oiq),*dihw=(float*)(d+oihw),
         *dsc=(float*)(d+osc),*dq=(float*)(d+oq),*datt=(float*)(d+oatt),*doa=(float*)(d+ooa);
    int*dsel=(int*)(d+osel);
    cs[0].proj_kv=dckv;cs[0].proj_gate=dcg;cs[1].proj_kv=dikv;cs[1].proj_gate=dig;
    int pool_fma=lf_pool_fma();
    cudaEvent_t*ev=kev[di];
    /* the sequence (issued directly, or once into the capture) */
    auto enqueue=[&]()->int{
        if(kprof)cudaEventRecord(ev[0],c->stream);
        if(!ok(cudaMemcpyAsync(d,h,upb,cudaMemcpyHostToDevice,c->stream),"fused layer upload"))return 0;
        /* 1. the input's projections (the PROJ6 round trip's kernels) */
        if(kprof)cudaEventRecord(ev[1],c->stream);
        run_mv_fused_fmt(wqa,dx,dqa,1,0,c->stream);
        run_mv_fused_fmt(wkv,dx,dkv,1,0,c->stream);
        if(ncomp){launch_bf16_one((__nv_bfloat16*)cwkv->w,dx,dckv,cwkv->O,cwkv->I,c->stream);
                  launch_bf16_one((__nv_bfloat16*)cwg->w,dx,dcg,cwg->O,cwg->I,c->stream);}
        if(nidx){launch_bf16_one((__nv_bfloat16*)iwkv->w,dx,dikv,iwkv->O,iwkv->I,c->stream);
                 launch_bf16_one((__nv_bfloat16*)iwg->w,dx,dig,iwg->O,iwg->I,c->stream);}
        if(!ok(cudaGetLastError(),"fused layer projection launch"))return 0;
        /* 2. the host's qa norm (and the query ranks' fp8 round trip); the kv
         *    norm, RoPE, fp8 round trip and ring write; the compressors'
         *    state rows: one launch */
        if(kprof)cudaEventRecord(ev[2],c->stream);
        {int nt=q_rank>dim?q_rank:dim;for(int k=0;k<ncs;k++)if(cs[k].proj>nt)nt=cs[k].proj;if(nt>1024)nt=1024;
         lf_prep_kernel<<<2+ncs,nt,(size_t)(q_rank>dim?q_rank:dim)*4,c->stream>>>(dqa,dqnw,q_rank,dxq,idx_score,dkv,dkvw,dim,eps,vnorm_fma,
            dcos,dsin,rope_dim,rope_fma,g_kv_ring[layer],dmeta,cs[0],cs[1]);}
        if(!ok(cudaGetLastError(),"fused layer prep launch"))return 0;
        /* 2b. the emit: the compressors' rows into their caches */
        if(emit){
            int nt=cs[0].dim;if(ncs>1&&cs[1].dim>nt)nt=cs[1].dim;
            lf_emit_kernel<<<ncs,nt,(size_t)nt*4,c->stream>>>(cs[0],cs[1],eps,vnorm_fma,pool_fma,decos,desin,rope_dim,rope_fma,dmeta);
            if(!ok(cudaGetLastError(),"fused layer emit launch"))return 0;
        }
        /* 3. the indexer's selection: the fused selection's launches (the
         *    score grid for the most candidates), the rank */
        if(kprof)cudaEventRecord(ev[3],c->stream);
        if(idx_score){
            if(!idx_select_launch(c,layer,diq,dihw,dsc,dxq,dx,dcos,dsin,dmeta+3,1,idx_heads,idx_dim,rope_dim,hidden,
                                  graph?IDX_SORT_MAX:idx_after,had_scale,wscale,idx_rope_fma,hw_fma,nullptr))return 0;
            idx_topk_kernel<<<1,1024,0,c->stream>>>(dsel,dsc,idx_after,cn,graph?dmeta:nullptr);
            if(!ok(cudaGetLastError(),"fused layer rank launch"))return 0;
        }
        /* 4. the chain: q (the wq_b store rounded, the norm and RoPE in one
         *    launch), the attention, the tail with the rounds in the stores */
        if(kprof)cudaEventRecord(ev[4],c->stream);
        run_mv_fused_fmt(wqb,dqa,dq,1,1,c->stream);
        q_norm_rope_kernel<<<heads,256,(size_t)dim*4,c->stream>>>(dq,dim,eps,norm_fma,dcos,dsin,rope_dim,rope_fma);
        if(!ok(cudaGetLastError(),"fused layer q launch"))return 0;
        if(kprof)cudaEventRecord(ev[5],c->stream);
        if(!attn_one_launch(c,layer,graph?sm_max:(size_t)(wn+cn)*4,datt,dq,dkv,dsink,dmeta,idx_score?dsel:nullptr,cn,first,position,
                            window,heads,dim,scale,graph?max_total:wn+cn,graph))return 0;
        if(kprof)cudaEventRecord(ev[6],c->stream);
        attn_inv_rope_kernel<<<heads,pairs,0,c->stream>>>(datt,dcos,dsin,dim,rope_dim,rope_fma,1);
        run_mv_fused_fmt(wa,datt,doa,groups,1,c->stream);
        run_mv_fused_fmt(wb,doa,dout,1,1,c->stream);
        if(!ok(cudaGetLastError(),"fused layer tail launch"))return 0;
        /* 5. the download */
        if(kprof)cudaEventRecord(ev[7],c->stream);
        if(!ok(cudaMemcpyAsync(h+oout,d+oout,dlb,cudaMemcpyDeviceToHost,c->stream),"fused layer download"))return 0;
        if(q_out&&!ok(cudaMemcpyAsync(q_out,dq,(size_t)heads*dim*4,cudaMemcpyDeviceToHost,c->stream),"fused layer q download"))return 0;
        if(q_out&&emit)for(int k=0;k<ncs;k++)if(a->emit_out[k]&&
            !ok(cudaMemcpyAsync(a->emit_out[k],cs[k].cache+(size_t)(k==0?comp_total:idx_count)*cs[k].dim,(size_t)cs[k].dim*4,cudaMemcpyDeviceToHost,c->stream),"fused layer emit download"))return 0;
        return 1;
    };
    int done=0;
    if(graph){
        LfGraph*G=&g_lf[di][layer][emit?1:0];
        LfSig S;memset(&S,0,sizeof S);
        S.lfd=c->lfd;S.lfh=h;S.ssc=c->ssc;S.ring=g_kv_ring[layer];S.comp=g_kv_comp[layer];S.keys=g_idx_keys[layer];
        S.wqa=wqa->w;S.wkv=wkv->w;S.wqb=wqb->w;S.wa=wa->w;S.wb=wb->w;S.cwkv=cwkv?cwkv->w:nullptr;S.cwg=cwg?cwg->w:nullptr;
        S.iwkv=iwkv?iwkv->w:nullptr;S.iwg=iwg?iwg->w:nullptr;S.iw=g_idx_w[layer];S.is=g_idx_s[layer];S.ihw=g_idx_hw[layer];
        for(int k=0;k<2;k++){S.cs[k*4]=cs[k].kv;S.cs[k*4+1]=cs[k].sc;S.cs[k*4+2]=cs[k].ape;S.cs[k*4+3]=cs[k].norm;}
        S.hidden=hidden;S.q_rank=q_rank;S.dim=dim;S.heads=heads;S.groups=groups;S.waO=wa->O;S.idx_score=idx_score;
        S.idx_heads=idx_heads;S.idx_dim=idx_dim;S.window=window;S.max_total=max_total;S.cn_max=cn_max;S.rope_dim=rope_dim;
        S.ncomp=ncomp;S.nidx=nidx;S.cO=cwkv?cwkv->O:0;S.gO=cwg?cwg->O:0;S.ikO=iwkv?iwkv->O:0;S.igO=iwg?iwg->O:0;
        S.vnorm_fma=vnorm_fma;S.rope_fma=rope_fma;S.idx_rope_fma=idx_rope_fma;S.hw_fma=hw_fma;S.norm_fma=norm_fma;
        S.ds=sattn_split_ds();S.K=K;S.headw=idx_headw_mode();S.unroll=mv_warp_unroll();
        S.eps=eps;S.scale=scale;S.had_scale=had_scale;S.wscale=wscale;S.unroll+=(kprof?1000:0)+(mv_warp_lut()?2000:0)+(idx_proj_warp()?4000:0);
        S.emit=emit;S.ratio=ratio;S.pool_fma=pool_fma;
        if(!G->valid||memcmp(&G->sig,&S,sizeof S)!=0){
            if(G->valid){cudaGraphExecDestroy(G->exec);cudaGraphDestroy(G->graph);G->valid=0;}
            if(!ok(cudaStreamBeginCapture(c->stream,cudaStreamCaptureModeThreadLocal),"fused layer capture begin"))return 0;
            int okq=enqueue();
            cudaGraph_t gr=nullptr;
            int ended=ok(cudaStreamEndCapture(c->stream,&gr),"fused layer capture end");
            if(!okq||!ended||!gr){if(gr)cudaGraphDestroy(gr);return 0;}
            if(!ok(cudaGraphInstantiate(&G->exec,gr,0),"fused layer graph instantiate")){cudaGraphDestroy(gr);return 0;}
            G->graph=gr;G->sig=S;G->valid=1;
        }
        static int gprof=-1;static cudaEvent_t ge[2];static double gsum;static long long gn;
        if(gprof<0){const char*e=getenv("DSV4_LF_GPROF");gprof=e&&*e&&*e!='0';if(gprof){cudaEventCreate(&ge[0]);cudaEventCreate(&ge[1]);}}
        if(gprof)cudaEventRecord(ge[0],c->stream);
        done=ok(cudaGraphLaunch(G->exec,c->stream),"fused layer graph launch");
        if(gprof){cudaEventRecord(ge[1],c->stream);cudaEventSynchronize(ge[1]);float ms=0;cudaEventElapsedTime(&ms,ge[0],ge[1]);
            gsum+=ms;gn++;if(gn%1000==0)fprintf(stderr,"lfgprof calls=%lld device=%.3f ms per layer call\n",gn,gsum/gn);}
    }else done=enqueue();
    if(!done||!ok(cudaStreamSynchronize(c->stream),"fused layer sync"))return 0;
    g_cs_upto[layer]=position+1;
    if(emit&&nidx&&!q_out)g_idx_keys_n[layer]=idx_after;   /* the host's keys_sync finds the row here (not under verify: the host's row then overwrites) */
    if(kprof){float ms[7];for(int i=0;i<7;i++)cudaEventElapsedTime(&ms[i],ev[i],ev[i+1]);
        fprintf(stderr,"lfkprof layer=%d up=%.3f proj=%.3f prep=%.3f idx=%.3f q=%.3f attn=%.3f tail=%.3f ms cn=%d scored=%d graph=%d emit=%d\n",
                layer,ms[0],ms[1],ms[2],ms[3],ms[4],ms[5],ms[6],cn,idx_score,graph,emit);}
    memcpy(out,h+oout,(size_t)hidden*4);memcpy(kv_out,h+okv,(size_t)dim*4);memcpy(qa_out,h+oqa,(size_t)q_rank*4);
    if(ncomp){memcpy(ckv_out,h+ockv,(size_t)cwkv->O*4);memcpy(cg_out,h+ocg,(size_t)cwg->O*4);}
    if(nidx){memcpy(ikv_out,h+oikv,(size_t)iwkv->O*4);memcpy(ig_out,h+oig,(size_t)iwg->O*4);}
    if(idx_score&&sel_out)memcpy(sel_out,h+osel,(size_t)cn*4);
    return 1;
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
/* One lane owns one output row, preserving the decode reference's column
 * order and fused fp32 accumulation. Rows32 packing coalesces weight reads. */
__global__ void compressor_project_exact(const __nv_bfloat16 *kv,const __nv_bfloat16 *gate,
                                         const float *x,float *out,int O,int I){
    int row=blockIdx.x*32+threadIdx.x;
    const __nv_bfloat16 *w=blockIdx.y?gate:kv;
    float sum=0.f;
    for(int i=0;i<I;i++)
        sum=__fmaf_rn(__bfloat162float(w[((long long)blockIdx.x*I+i)*32+threadIdx.x]),x[i],sum);
    out[(long long)blockIdx.y*O+row]=sum;
}
__global__ void indexer_head_project(const __nv_bfloat16 *w,const float *x,float *out,int I,float scale){
    int row=blockIdx.x*32+threadIdx.x;float sum=0.f;
    /* Unlike the compressor pair, the reference indexer head projection
     * rounds the product before adding it to the sequential accumulator. */
    for(int i=0;i<I;i++)sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(w[((long long)blockIdx.x*I+i)*32+threadIdx.x]),x[i]));
    out[row]=__fmul_rn(sum,scale);
}
/* Keep the CPU butterfly order and its FP4 tie rule (first code wins). */
__global__ void indexer_hadamard_fp4(float *q,int dim,float norm){
    extern __shared__ float values[];
    int d=threadIdx.x;float v=q[(long long)blockIdx.x*dim+d];values[d]=v;
    __syncthreads();
    for(int width=1;width<dim;width*=2){
        float peer=values[d^width];v=values[d];
        v=(d&width)?__fsub_rn(peer,v):__fadd_rn(v,peer);
        __syncthreads();values[d]=v;__syncthreads();
    }
    v=__bfloat162float(__float2bfloat16(__fmul_rn(v,norm)));
    float mx=fabsf(v);
    for(int offset=16;offset;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
    mx=fmaxf(mx,6.f*0x1p-126f);
    int exponent;float fraction=frexpf(__fdiv_rn(mx,6.f),&exponent);
    if(fraction==.5f)--exponent;
    exponent=max(-127,min(127,exponent));float scale=ldexpf(1.f,exponent);
    float x=fmaxf(-6.f,fminf(6.f,__fdiv_rn(v,scale)));
    int best=0;float distance=fabsf(x);
    #pragma unroll
    for(int code=1;code<16;code++){float candidate=fabsf(__fsub_rn(x,f4(code)));if(candidate<distance){best=code;distance=candidate;}}
    q[(long long)blockIdx.x*dim+d]=__bfloat162float(__float2bfloat16(__fmul_rn(f4(best),scale)));
}
extern "C" int dsv4_cuda_indexer_prepare(Dsv4CudaTensor *weights,const float *input,
                                         float *queries,float *head_weights,int dim){
    Dev *c=weights?ctx(weights->device):nullptr;
    if(!c||!input||!queries||!head_weights||weights->fmt!=16||!weights->bf16_cache||
       weights->O%32||dim<32||dim>512||(dim&(dim-1))||
       !ok(cudaSetDevice(weights->device),"select indexer prepare device"))return 0;
    int heads=weights->O;size_t xb=(size_t)weights->I*4,qb=(size_t)heads*dim*4,wb=(size_t)heads*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,wb)||
       !buf((void**)&c->p1,&c->p1cap,qb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,input,xb,cudaMemcpyHostToDevice,c->stream),"indexer hidden upload")||
       !ok(cudaMemcpyAsync(c->p1,queries,qb,cudaMemcpyHostToDevice,c->stream),"indexer prepare upload"))return 0;
    indexer_head_project<<<heads/32,32,0,c->stream>>>((const __nv_bfloat16*)weights->bf16_cache,c->dx,c->dy,weights->I,1.f/sqrtf((float)(heads*dim)));
    indexer_hadamard_fp4<<<heads,dim,dim*4,c->stream>>>(c->p1,dim,1.f/sqrtf((float)dim));
    int good=ok(cudaGetLastError(),"indexer prepare launch")&&
        ok(cudaMemcpyAsync(queries,c->p1,qb,cudaMemcpyDeviceToHost,c->stream),"indexer prepare download")&&
        ok(cudaMemcpyAsync(head_weights,c->dy,wb,cudaMemcpyDeviceToHost,c->stream),"indexer head weights download");
    int synced=ok(cudaStreamSynchronize(c->stream),"indexer prepare sync");return good&&synced;
}
extern "C" int dsv4_cuda_compressor_project(Dsv4CudaTensor *kv,Dsv4CudaTensor *gate,
                                            const float *x,float *v,float *scores){
    Dev *c=kv?ctx(kv->device):nullptr;
    if(!c||!gate||!x||!v||!scores||kv->fmt!=16||gate->fmt!=16||
       kv->device!=gate->device||kv->O!=gate->O||kv->I!=gate->I||
       !kv->bf16_cache||!gate->bf16_cache||kv->O%32||
       !ok(cudaSetDevice(kv->device),"select compressor device"))return 0;
    size_t xb=(size_t)kv->I*4,yb=(size_t)kv->O*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,2*yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"compressor input upload"))return 0;
    compressor_project_exact<<<dim3(kv->O/32,2),32,0,c->stream>>>(
        (const __nv_bfloat16*)kv->bf16_cache,(const __nv_bfloat16*)gate->bf16_cache,c->dx,c->dy,kv->O,kv->I);
    int good=ok(cudaGetLastError(),"compressor projection launch")&&
        ok(cudaMemcpyAsync(v,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"compressor values download")&&
        ok(cudaMemcpyAsync(scores,c->dy+kv->O,yb,cudaMemcpyDeviceToHost,c->stream),"compressor scores download");
    /* Drain any issued download before the caller overwrites CPU state on fallback. */
    int synced=ok(cudaStreamSynchronize(c->stream),"compressor projection sync");
    return good&&synced;
}
extern "C" int dsv4_cuda_matmul_bf16_batch(Dsv4CudaTensor*t,const float*x,int tokens,float*y){
    Dev*c=t?ctx(t->device):nullptr;
    if(!c||!x||!y||tokens<1||t->fmt!=16||!ok(cudaSetDevice(t->device),"select bf16 batch device"))return 0;
    size_t xb=(size_t)tokens*t->I*4,yb=(size_t)tokens*t->O*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"bf16 batch upload"))return 0;
    if(tokens==1)launch_bf16_one((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I,c->stream);
    else mm_bf16_batch<<<dim3(t->O,(tokens+15)/16),256,0,c->stream>>>((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I,tokens);
    return ok(cudaGetLastError(),"bf16 batch launch")&&
           ok(cudaMemcpyAsync(y,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"bf16 batch download")&&
           ok(cudaStreamSynchronize(c->stream),"bf16 batch sync");
}
/* Several bf16 matmuls over the same x in one round trip: one upload, one
 * launch each, one download each, one sync. Same kernel as the single call. */
extern "C" int dsv4_cuda_matmul_bf16_multi(Dsv4CudaTensor*const*ts,int n,const float*x,int tokens,float*const*ys){
    if(!ts||!ys||n<1||n>8||!x||tokens<1)return 0;
    Dev*c=ctx(ts[0]->device);
    if(!c||!ok(cudaSetDevice(ts[0]->device),"select bf16 multi device"))return 0;
    size_t xb=(size_t)tokens*ts[0]->I*4,yb=0;
    for(int i=0;i<n;i++){
        if(!ts[i]||!ys[i]||ts[i]->fmt!=16||ts[i]->device!=ts[0]->device||ts[i]->I!=ts[0]->I)return 0;
        yb+=(size_t)tokens*ts[i]->O*4;
    }
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"bf16 multi upload"))return 0;
    size_t off=0;
    for(int i=0;i<n;i++){
        size_t ob=(size_t)tokens*ts[i]->O*4;
        float*dy=(float*)((char*)c->dy+off);
        if(tokens==1)launch_bf16_one((__nv_bfloat16*)ts[i]->w,c->dx,dy,ts[i]->O,ts[i]->I,c->stream);
        else mm_bf16_batch<<<dim3(ts[i]->O,(tokens+15)/16),256,0,c->stream>>>((__nv_bfloat16*)ts[i]->w,c->dx,dy,ts[i]->O,ts[i]->I,tokens);
        if(!ok(cudaGetLastError(),"bf16 multi launch")||
           !ok(cudaMemcpyAsync(ys[i],dy,ob,cudaMemcpyDeviceToHost,c->stream),"bf16 multi download"))return 0;
        off+=ob;
    }
    return ok(cudaStreamSynchronize(c->stream),"bf16 multi sync");
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

/* the cuBLASLt tensor-core path's handle and workspace (nothing without COLI_DSV4_TC) */
#if COLI_DSV4_TC
static int tc_dev_init(Dev*c){return bok(cublasLtCreate(&c->lt),"cuBLASLt create")&&ok(cudaMalloc(&c->tc_workspace,32<<20),"Tensor Core workspace");}
#else
static int tc_dev_init(Dev*){return 1;}
#endif
extern "C" int dsv4_cuda_peer_access(int a,int b){return a!=b&&dev_ok(a,b)&&dev_ok(b,a);}
/* #1766 layer placement does not need P2P; #2405's peer mirror pool does.
 * The engine sets this to 1 only around the multi-device layer-split init. */
extern "C" void dsv4_cuda_set_peer_best_effort(int on){g_peer_best_effort=on?1:0;}
extern "C" int dsv4_cuda_peer_healthy(int device){return !dev_lost(device);}
/* DSV4_CUDA_BATCHED (1): the fused decode MoE runs its routed experts as
 * grouped launches; 0 = one launch chain per expert, no peer compute. */
static bool moe_batched(){static int on=-1;if(on<0){const char*v=getenv("DSV4_CUDA_BATCHED");on=v?atoi(v)!=0:1;}return on;}
/* Peer compute (DSV4_CUDA_PEER_COMPUTE, default 1, batched kernels only):
 * the fused decode MoE on device a runs the experts mirrored on device b ON
 * b. b must be another initialised, healthy device, each must reach the
 * other's memory (b's down-projection writes a's reduce buffer), and both
 * must be the same SM version, so the two run the same machine code and the
 * result stays bit-identical. Otherwise a reads b's mirrors in place. */
static bool peer_runs(int a,int b){
    static int on=-1;if(on<0){const char*v=getenv("DSV4_CUDA_PEER_COMPUTE");on=v?atoi(v)!=0:1;}
    Dev*p=ctx(a),*q=ctx(b);return on&&moe_batched()&&a!=b&&p&&q&&p->sm==q->sm&&dev_ok(a,b)&&dev_ok(b,a);}
extern "C" int dsv4_cuda_peer_compute(int device,int peer){return peer_runs(device,peer);}
/* Asked BEFORE dsv4_cuda_init (DSV4_CUDA_PEER_DEVICE), so a peer that is
 * refused is never initialised: both ordinals visible, distinct and < 16 (the
 * size of g_peer), a peer not latched off by an earlier failure
 * (dsv4_cuda_peer_healthy), a compute capability this build accepts for each
 * (dsv4_cuda_backend_arch_ok), peer access in both directions, and a kernel
 * image in this binary for each (cudaFuncGetAttributes: "no kernel image" or
 * "invalid device function" otherwise -- the only errors the reason pairs
 * with a rebuild hint). The primary is checked first: one the build cannot
 * run fails its own init anyway, and its peer must not be left holding a
 * context for a CPU-only run. Only the kernel-image check makes a
 * context: the primary's is the one its init creates; the peer's is destroyed
 * again on refusal unless the backend already holds a device. 1 = usable;
 * 0 = refused, the reason in why. */
extern "C" int dsv4_cuda_peer_probe(int device,int peer,char *why,int why_size){
    char none[1];if(!why||why_size<1){why=none;why_size=1;}why[0]=0;
    int count=0;cudaError_t e=cudaGetDeviceCount(&count);
    if(e!=cudaSuccess){cudaGetLastError();snprintf(why,why_size,"no CUDA device (%s)",cudaGetErrorString(e));return 0;}
    if(peer==device){snprintf(why,why_size,"peer device %d is the primary device",peer);return 0;}
    for(int k=0;k<2;k++){int d=k?peer:device;
        if(d<0||d>=count){snprintf(why,why_size,"device %d is not a visible CUDA device (%d visible)",d,count);return 0;}
        if(d>=16){snprintf(why,why_size,"device %d: peer access is tracked for devices 0-15 only",d);return 0;}}
    if(dev_lost(peer)){snprintf(why,why_size,"peer device %d failed in the decode MoE earlier in this process and stays latched off",peer);return 0;}
    cudaDeviceProp p[2]{};const char *role[2]={"primary","peer"};
    for(int k=0;k<2;k++){int d=k?peer:device;
        if((e=cudaGetDeviceProperties(&p[k],d))!=cudaSuccess){cudaGetLastError();snprintf(why,why_size,"%s device %d: %s",role[k],d,cudaGetErrorString(e));return 0;}
        if(!dsv4_cuda_backend_arch_ok(d)){
#ifdef COLI_DSV4_DEEPGEMM
            snprintf(why,why_size,"%s device %d (%s, sm_%d%d): the %s backend runs on sm_120 only (pair other cards with the generic build: no DEEPGEMM)",role[k],d,p[k].name,p[k].major,p[k].minor,dsv4_cuda_backend_name());
#else
            snprintf(why,why_size,"%s device %d (%s, sm_%d%d): the generic kernels need sm_60 or newer",role[k],d,p[k].name,p[k].major,p[k].minor);
#endif
            return 0;}}
    int ab=0,ba=0;
    if((e=cudaDeviceCanAccessPeer(&ab,device,peer))!=cudaSuccess||(e=cudaDeviceCanAccessPeer(&ba,peer,device))!=cudaSuccess){cudaGetLastError();snprintf(why,why_size,"peer query %d<->%d: %s",device,peer,cudaGetErrorString(e));return 0;}
    if(!ab||!ba){snprintf(why,why_size,"no peer access %d%s%d (P2P needs NVLink or a P2P-capable PCIe path; nvidia-smi topo -p2p r shows it)",ab?peer:device,!ab&&!ba?"<->":"->",ab?device:peer);return 0;}
    for(int k=0;k<2;k++){int d=k?peer:device,set=(e=cudaSetDevice(d))==cudaSuccess;
        if(set){cudaFuncAttributes fa;e=cudaFuncGetAttributes(&fa,(const void*)mv_fp4_grouped<4>);}
        if(e==cudaSuccess)continue;
        cudaGetLastError();if(k&&!ng&&cudaSetDevice(peer)==cudaSuccess)cudaDeviceReset();cudaGetLastError();
        /* only these two mean "no code for this device"; a GPU that is full,
         * held by an exclusive-process job or faulting gets no rebuild advice */
        if(e!=cudaErrorNoKernelImageForDevice&&e!=cudaErrorInvalidDeviceFunction){
            snprintf(why,why_size,"%s device %d (%s, sm_%d%d): cannot be used: %s: %s",role[k],d,p[k].name,p[k].major,p[k].minor,set?"kernel-image check":"cudaSetDevice",cudaGetErrorString(e));return 0;}
#ifdef COLI_DSV4_DEEPGEMM
        snprintf(why,why_size,"%s device %d (%s, sm_%d%d): %s (this %s build carries sm_120a code only; pair other cards with the generic build: no DEEPGEMM)",role[k],d,p[k].name,p[k].major,p[k].minor,cudaGetErrorString(e),dsv4_cuda_backend_name());
#else
        snprintf(why,why_size,"%s device %d (%s, sm_%d%d): %s (this build has no code for it; rebuild with a CUDA_ARCH that covers sm_%d%d, e.g. CUDA_ARCH=%s)",role[k],d,p[k].name,p[k].major,p[k].minor,cudaGetErrorString(e),p[k].major,p[k].minor,p[k].major<8?"portable-pre-ampere NO_TC=1":"portable");
#endif
        return 0;}
    return 1;}
#if COLI_DSV4_TC
static void plan_free(TcPlan*p){if(p->d)cublasLtMatrixLayoutDestroy(p->d);if(p->c)cublasLtMatrixLayoutDestroy(p->c);if(p->b)cublasLtMatrixLayoutDestroy(p->b);if(p->a)cublasLtMatrixLayoutDestroy(p->a);if(p->op)cublasLtMatmulDescDestroy(p->op);memset(p,0,sizeof(*p));}
static void tc_dev_free(Dev*c){for(int p=0;p<2;p++)plan_free(&c->plans[p]);if(c->lt)cublasLtDestroy(c->lt);}
#else
static void tc_dev_free(Dev*){}
#endif
/* Frees what g[0..n) holds -- a live backend, or whatever a failed init had
 * set up when it stopped (a stream or handle it never created is 0) -- and
 * zeroes the slots. */
static void dev_release(int n){
    for(int i=0;i<n;i++){cudaSetDevice(g[i].id);if(g[i].decode_state)cudaFree(g[i].decode_state);
#ifdef COLI_DSV4_NCCL
        if(g[i].ep_comm)ncclCommDestroy(g[i].ep_comm);
#endif
    }
    for(int i=0;i<n;i++){cudaSetDevice(g[i].id);if(g[i].dx)cudaFree(g[i].dx);if(g[i].dy)cudaFree(g[i].dy);if(g[i].p1)cudaFree(g[i].p1);if(g[i].p2)cudaFree(g[i].p2);if(g[i].p3)cudaFree(g[i].p3);if(g[i].p4)cudaFree(g[i].p4);if(g[i].aux1)cudaFree(g[i].aux1);if(g[i].aux2)cudaFree(g[i].aux2);if(g[i].aux3)cudaFree(g[i].aux3);if(g[i].mvdesc)cudaFree(g[i].mvdesc);if(g[i].expert_weights)cudaFree(g[i].expert_weights);if(g[i].expert_ids)cudaFree(g[i].expert_ids);if(g[i].tcw)cudaFree(g[i].tcw);if(g[i].tcs)cudaFree(g[i].tcs);if(g[i].tcx)cudaFree(g[i].tcx);if(g[i].tcxs)cudaFree(g[i].tcxs);if(g[i].tc_workspace)cudaFree(g[i].tc_workspace);
        if(g[i].emaj)cudaFree(g[i].emaj);if(g[i].emv1)cudaFree(g[i].emv1);if(g[i].emv2)cudaFree(g[i].emv2);if(g[i].emv4)cudaFree(g[i].emv4);if(g[i].emcount)cudaFree(g[i].emcount);if(g[i].mmv)cudaFree(g[i].mmv);if(g[i].mmvcount)cudaFree(g[i].mmvcount);if(g[i].v4fact)cudaFree(g[i].v4fact);if(g[i].v4fparts)cudaFree(g[i].v4fparts);if(g[i].v4fbar)cudaFree(g[i].v4fbar);
        if(g[i].sa_score)cudaFree(g[i].sa_score);if(g[i].sa_w)cudaFree(g[i].sa_w);if(g[i].sa_den)cudaFree(g[i].sa_den);if(g[i].sa_out)cudaFree(g[i].sa_out);if(g[i].sa_rows)cudaFree(g[i].sa_rows);
#ifdef COLI_DSV4_DEEPGEMM
        if(g[i].dga1)cudaFree(g[i].dga1);if(g[i].dga2)cudaFree(g[i].dga2);if(g[i].dgsfa1)cudaFree(g[i].dgsfa1);if(g[i].dgsfa2)cudaFree(g[i].dgsfa2);if(g[i].dglayout)cudaFree(g[i].dglayout);if(g[i].dgrowmap)cudaFree(g[i].dgrowmap);if(g[i].dgmm1)cudaFree(g[i].dgmm1);if(g[i].dgmm2)cudaFree(g[i].dgmm2);if(g[i].dgref)cudaFree(g[i].dgref);if(g[i].dgdensews)cudaFree(g[i].dgdensews);
#endif
        tc_dev_free(&g[i]);if(g[i].hx)cudaFreeHost(g[i].hx);if(g[i].hy)cudaFreeHost(g[i].hy);if(g[i].ready)cudaEventDestroy(g[i].ready);if(g[i].fork)cudaEventDestroy(g[i].fork);if(g[i].join)cudaEventDestroy(g[i].join);if(g[i].aux)cudaStreamDestroy(g[i].aux);if(g[i].stream)cudaStreamDestroy(g[i].stream);
        /* zero the slot: every buffer above is grown lazily through buf(), which
         * would cudaFree a stale pointer (and trust its cap) after a re-init */
        memset(&g[i],0,sizeof g[i]);}ng=0;
    /* peer access ends with the contexts; a re-init records it afresh */
    memset(g_peer,0,sizeof g_peer);}
/* A failed init releases every device it touched (streams, events, cuBLASLt
 * handles, buffers, peer records) and clears the runtime's last error, so a
 * retry with fewer devices -- the engine's two-GPU fallback -- starts clean
 * instead of leaking the first attempt and failing on its stale error. With
 * several devices it names the one that failed (b < 0) or the link a -> b,
 * and it also destroys the contexts of dev[1..n) (made by this init or by
 * dsv4_cuda_peer_probe, holding nothing now): the fallback runs on dev[0]
 * alone, or on the CPU, so those GPUs would otherwise keep ~0.3 GiB each for
 * the whole run. dev[0]'s context stays for the retry. */
static int init_fail(const int *dev,int n,int touched,int a,int b){
    if(n>1&&a>=0){if(b<0)fprintf(stderr,"[DSV4 CUDA] device %d did not initialise\n",a);
        else fprintf(stderr,"[DSV4 CUDA] peer access %d -> %d failed\n",a,b);}
    dev_release(touched);
    for(int i=1;i<n;i++)if(dev[i]!=dev[0]&&cudaSetDevice(dev[i])==cudaSuccess)cudaDeviceReset();
    cudaGetLastError();return 0;}
extern "C" int dsv4_cuda_init(const int *dev,int n){if(!dev||n<1||n>16||g_cuda_lost||ng)return 0;
    for(int i=0;i<n;i++){
        cudaDeviceProp p{};
        /* name the refused device, and leave no stale error for the caller's next call */
        if(!ok(cudaGetDeviceProperties(&p,dev[i]),"validate device")){
            if(n>1)fprintf(stderr,"[DSV4 CUDA] device %d did not initialise\n",dev[i]);
            cudaGetLastError();return 0;}
        for(int j=0;j<i;j++)if(dev[j]==dev[i])return 0;
    }
    float scale[256];for(int b=0;b<256;b++)scale[b]=b?ldexpf(1.f,b-127):ldexpf(1.f,-127);ng=0;for(int i=0;i<n;i++){cudaDeviceProp p{};g[i].id=dev[i];if(!ok(cudaSetDevice(dev[i]),"select device")||!ok(cudaMemcpyToSymbol(e8_table,scale,sizeof(scale)),"scale table upload")||!ok(cudaGetDeviceProperties(&p,dev[i]),"device properties")||!ok(cudaStreamCreateWithFlags(&g[i].stream,cudaStreamNonBlocking),"stream")||!ok(cudaStreamCreateWithFlags(&g[i].aux,cudaStreamNonBlocking),"aux stream")||!ok(cudaEventCreateWithFlags(&g[i].ready,cudaEventDisableTiming),"device event")||!ok(cudaEventCreateWithFlags(&g[i].fork,cudaEventDisableTiming),"fork event")||!ok(cudaEventCreateWithFlags(&g[i].join,cudaEventDisableTiming),"join event")||!tc_dev_init(&g[i])||!ok(cudaMalloc(&g[i].decode_state,2*sizeof(int)),"decode state allocation"))return init_fail(dev,n,i+1,dev[i],-1);g[i].sms=p.multiProcessorCount;g[i].sm=p.major*10+p.minor;ng++;fprintf(stderr,"[DSV4 CUDA] device %d: %s %.1f GB sm_%d%d\n",dev[i],p.name,p.totalGlobalMem/1e9,p.major,p.minor);}for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(i!=j){
    int can=0;cudaError_t q=cudaDeviceCanAccessPeer(&can,dev[i],dev[j]);
    if(q!=cudaSuccess){if(!g_peer_best_effort)return init_fail(dev,n,n,dev[i],dev[j]);const char*why=cudaGetErrorString(q);cudaGetLastError();fprintf(stderr,"[DSV4 CUDA] peer query %d -> %d unavailable (%s); continuing without peer access\n",dev[i],dev[j],why);continue;}
    if(!can)continue;
    if(!ok(cudaSetDevice(dev[i]),"select peer device"))return init_fail(dev,n,n,dev[i],dev[j]);
    cudaError_t e=cudaDeviceEnablePeerAccess(dev[j],0);
    if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled){if(!g_peer_best_effort)return init_fail(dev,n,n,dev[i],dev[j]);const char*why=cudaGetErrorString(e);cudaGetLastError();fprintf(stderr,"[DSV4 CUDA] peer access %d -> %d unavailable (%s); continuing without it\n",dev[i],dev[j],why);}
    else {if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();if(dev[i]>=0&&dev[i]<16&&dev[j]>=0&&dev[j]<16)g_peer[dev[i]][dev[j]]=1;}
}
#ifdef COLI_DSV4_NCCL
    if(n==6)for(int pair=0;pair<3;pair++){int ids[2]={dev[2*pair],dev[2*pair+1]};ncclComm_t comm[2];if(!nok(ncclCommInitAll(comm,2,ids),"EP2 communicator init"))return init_fail(dev,n,n,-1,-1);g[2*pair].ep_comm=comm[0];g[2*pair+1].ep_comm=comm[1];}
#endif
    return 1;}
extern "C" void dsv4_cuda_shutdown(void){
    for(int layer=0;layer<DSV4_KV_CACHE_LAYERS;layer++){
        if(!g_kv_ring[layer]&&!g_kv_comp[layer])continue;
        cudaSetDevice(g_kv_device[layer]);
        if(g_kv_ring[layer])cudaFree(g_kv_ring[layer]);
        if(g_kv_comp[layer])cudaFree(g_kv_comp[layer]);
        g_kv_ring[layer]=g_kv_comp[layer]=nullptr;
        g_kv_ring_rows[layer]=g_kv_comp_cap[layer]=0;
    }
    dev_release(ng);}
extern "C" Dsv4CudaActivation *dsv4_cuda_activation_create(int device,long long elements){Dev*c=ctx(device);if(!c||elements<1||!ok(cudaSetDevice(device),"select activation device"))return nullptr;Dsv4CudaActivation*a=(Dsv4CudaActivation*)calloc(1,sizeof(*a));if(!a)return nullptr;a->device=device;a->elements=elements;if(!alloc_ok(cudaMalloc(&a->data,(size_t)elements*sizeof(float)),"activation allocation")){free(a);return nullptr;}return a;}
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
/* Host->device copy ordered on the device's compute stream, then drained.
 * A plain cudaMemcpy runs on the legacy stream, which the non-blocking
 * c->stream does not wait for, and from pageable memory it may return before
 * the DMA has landed: a kernel queued on c->stream right after could read a
 * half-written decode mirror, and decode output then drifted with the mirror
 * cache's recycling timing (expert slabs past DSV4_CUDA_PIN_CAP_GB, or all of
 * them with DSV4_CUDA_PIN_HOST=0, are pageable). Same blocking contract for
 * the caller as cudaMemcpy. */
static int h2d(Dev*c,void*dst,const void*src,size_t n,const char*what){
    return ok(cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice,c->stream),what)&&ok(cudaStreamSynchronize(c->stream),what);}
static int upload(Dsv4CudaTensor **pt,const void *w,size_t wb,const uint8_t *sc,size_t sb,int O,int I,int d,int fmt){if(*pt)return 1;Dev *c=ctx(d);if(!c||!ok(cudaSetDevice(d),"select upload device"))return 0;Dsv4CudaTensor *t=(Dsv4CudaTensor*)calloc(1,sizeof(*t));t->O=O;t->I=I;t->device=d;t->fmt=fmt;t->bytes=wb+sb;
    if(!alloc_ok(cudaMalloc(&t->w,wb),"weight allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_w=1;if(!h2d(c,t->w,w,wb,"weight upload")){dsv4_cuda_tensor_free(t);return 0;}
    if(sb){if(!alloc_ok(cudaMalloc(&t->scale,sb),"scale allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_scale=1;if(!h2d(c,t->scale,sc,sb,"scale upload")){dsv4_cuda_tensor_free(t);return 0;}}
#ifdef COLI_DSV4_DEEPGEMM
    if((fmt==8||fmt==9)&&O%128==0&&I%128==0){size_t db=(size_t)O*(I/128/4)*sizeof(int);if(!alloc_ok(cudaMalloc(&t->dg_scale,db),"dense DeepGEMM scale allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_dg_scale=1;
        if(fmt==9){int groups=O/1024;dg_pack_batched_scale<<<(O*(I/128/4)+255)/256,256,0,c->stream>>>(t->dg_scale,t->scale,O/groups,I/128,groups);}
        else dg_pack_dense_scale<<<(O*(I/128/4)+255)/256,256,0,c->stream>>>(t->dg_scale,t->scale,O,I/128);
        if(!ok(cudaGetLastError(),"dense DeepGEMM scale packing")){dsv4_cuda_tensor_free(t);return 0;}t->bytes+=db;}
#endif
    *pt=t;return 1;}
extern "C" int dsv4_cuda_upload_fp8(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){return upload(t,w,(size_t)O*I,s,(size_t)((O+127)/128)*((I+127)/128),O,I,d,8);}
extern "C" int dsv4_cuda_upload_fp8_ref(Dsv4CudaTensor **t,const uint8_t*w,const float*s,int O,int I,int packed,int d){
    if(!t||!w||!s||O<1||I<1||I%128||(packed!=0&&packed!=1)||(packed&&O%8))return 0;
    return upload(t,w,(size_t)O*I,(const uint8_t*)s,(size_t)((O+127)/128)*(I/128)*sizeof(float),O,I,d,packed?11:10);
}
extern "C" int dsv4_cuda_upload_fp8_bf16(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){return upload(t,w,(size_t)O*I,s,(size_t)((O+127)/128)*((I+127)/128),O,I,d,9);}
extern "C" int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){if(!upload(t,w,(size_t)O*I/2,s,(size_t)O*I/32,O,I,d,4))return 0;
    return 1;}
/* In-place refill of an existing fp4 mirror with a same-shape expert: no
 * cudaFree/cudaMalloc churn, DMA from page-locked slabs when possible. The
 * copies are ordered on c->stream from pinned AND pageable slabs (a pageable
 * cudaMemcpyAsync stages the source before it returns, so the slab may be
 * released at once); sync=1 drains before returning. A legacy-stream
 * cudaMemcpy here was not ordered against the kernels on c->stream (see h2d). */
static bool host_pinned(const void*p,size_t len);
extern "C" int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t,const uint8_t*w,const uint8_t*s,int O,int I,int sync){
    if(!t||!w||!s||t->fmt!=4||t->O!=O||t->I!=I||!t->own_w||!t->own_scale||!t->w||!t->scale)return 0;
    Dev*c=ctx(t->device);if(!c||!ok(cudaSetDevice(t->device),"select refill device"))return 0;
    size_t wb=(size_t)O*I/2,sb=(size_t)O*I/32;
    if(host_pinned(w,wb))host_pinned(s,sb);   /* page-lock the slab when the cap allows: pure DMA then */
    if(!ok(cudaMemcpyAsync(t->w,w,wb,cudaMemcpyHostToDevice,c->stream),"refill weight upload")||
       !ok(cudaMemcpyAsync(t->scale,s,sb,cudaMemcpyHostToDevice,c->stream),"refill scale upload"))return 0;
    if(sync&&!ok(cudaStreamSynchronize(c->stream),"refill drain"))return 0;
    return 1;
}
extern "C" int dsv4_cuda_upload_bf16(Dsv4CudaTensor **t,const uint16_t*w,int O,int I,int d){return upload(t,w,(size_t)O*I*2,nullptr,0,O,I,d,16);}
__global__ void pack_bf16_rows32(uint16_t *out,const uint16_t *in,int O,int I){
    long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(p<(long long)O*I){int lane=p%32,col=(p/32)%I,row=(p/(32LL*I))*32+lane;out[p]=in[(long long)row*I+col];}
}
extern "C" int dsv4_cuda_upload_compressor(Dsv4CudaTensor **pt,const uint16_t *w,int O,int I,int device){
    if(!pt||!w||O<1||I<1||O%32)return 0;
    if(*pt)return (*pt)->fmt==16&&(*pt)->bf16_cache&&(*pt)->O==O&&(*pt)->I==I&&(*pt)->device==device;
    Dsv4CudaTensor *t=nullptr;
    if(!dsv4_cuda_upload_bf16(&t,w,O,I,device))return 0;
    Dev *c=ctx(device);size_t bytes=(size_t)O*I*2;
    /* upload() uses the default stream; packing runs on a nonblocking stream. */
    if(!ok(cudaStreamSynchronize(nullptr),"compressor source upload sync")||
       !ok(cudaMalloc(&t->bf16_cache,bytes),"compressor packed allocation")){dsv4_cuda_tensor_free(t);return 0;}
    pack_bf16_rows32<<<((long long)O*I+255)/256,256,0,c->stream>>>((uint16_t*)t->bf16_cache,(const uint16_t*)t->w,O,I);
    if(!ok(cudaGetLastError(),"compressor packing launch")||!ok(cudaStreamSynchronize(c->stream),"compressor packing sync")){dsv4_cuda_tensor_free(t);return 0;}
    const char *verify=getenv("DSV4_IDX_VERIFY");
    if(verify&&*verify&&*verify!='0'){
        std::vector<uint16_t> packed((size_t)O*I);
        if(!ok(cudaMemcpy(packed.data(),t->bf16_cache,bytes,cudaMemcpyDeviceToHost),"packed verification download")){dsv4_cuda_tensor_free(t);return 0;}
        for(int row=0;row<O;row++)for(int col=0;col<I;col++)
            if(packed[((size_t)(row/32)*I+col)*32+row%32]!=w[(size_t)row*I+col]){
                fprintf(stderr,"packed verification mismatch device=%d O=%d I=%d row=%d col=%d\n",device,O,I,row,col);
                dsv4_cuda_tensor_free(t);return 0;
            }
    }
    t->bytes+=bytes;*pt=t;return 1;
}

extern "C" int dsv4_cuda_upload_head_exact(Dsv4CudaTensor **pt,const uint16_t *w,int O,int I,int device){
    if(!pt||*pt||I<1||I%8)return 0;
    Dsv4CudaTensor *t=nullptr;
    if(!dsv4_cuda_upload_compressor(&t,w,O,I,device))return 0;
    // The head only consumes the packed copy; discard the upload source.
    if(!ok(cudaFree(t->w),"head source release")){dsv4_cuda_tensor_free(t);return 0;}
    t->w=t->bf16_cache;t->bf16_cache=nullptr;t->fmt=12;t->bytes=(long long)O*I*2;
    *pt=t;return 1;
}
__global__ void head_scores_exact(const __nv_bfloat16 *w,const float *x,float *out,int O,int I){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=O)return;
    float sum=0.f;long long base=(long long)(row/32)*I*32+row%32;
    for(int i=0;i<I;i++)sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(w[base+(long long)i*32]),x[i]));
    out[row]=sum;
}
extern "C" int dsv4_cuda_head_scores_exact(Dsv4CudaTensor *t,const float *input,float *scores){
    Dev *c=t?ctx(t->device):nullptr;
    if(!c||t->fmt!=12||!input||!scores||!ok(cudaSetDevice(t->device),"select head device"))return 0;
    size_t xb=(size_t)t->I*sizeof(float),yb=(size_t)t->O*sizeof(float);
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    int success=ok(cudaMemcpyAsync(c->dx,input,xb,cudaMemcpyHostToDevice,c->stream),"head input upload");
    if(success){
        head_scores_exact<<<(t->O+127)/128,128,0,c->stream>>>((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I);
        success=ok(cudaGetLastError(),"head scores launch")&&
            ok(cudaMemcpyAsync(scores,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"head scores download");
    }
    int drained=ok(cudaStreamSynchronize(c->stream),"head scores sync");
    return success&&drained;
}

// Four independent sums reuse each weight without changing column order.
__global__ void head_scores_batch_exact(const __nv_bfloat16 *w,const float *x,float *out,int O,int I,int batch){
    int row=blockIdx.x*blockDim.x+threadIdx.x,first=blockIdx.y*4;
    if(row>=O)return;
    float sums[4]={0.f,0.f,0.f,0.f};
    long long base=(long long)(row/32)*I*32+row%32;
    for(int i=0;i<I;i++){
        float weight=__bfloat162float(w[base+(long long)i*32]);
        #pragma unroll
        for(int j=0;j<4;j++)if(first+j<batch)
            sums[j]=__fadd_rn(sums[j],__fmul_rn(weight,x[(size_t)(first+j)*I+i]));
    }
    #pragma unroll
    for(int j=0;j<4;j++)if(first+j<batch)out[(size_t)(first+j)*O+row]=sums[j];
}

__global__ void head_first_argmax(const float *scores,int vocab,int *ids,float *values){
    __shared__ float maxima[256];
    __shared__ int winners[256];
    int lane=threadIdx.x,winner=-1;
    float maximum=-FLT_MAX;
    for(int row=lane;row<vocab;row+=blockDim.x){
        float value=scores[(size_t)blockIdx.x*vocab+row];
        if(value>maximum){maximum=value;winner=row;}
    }
    maxima[lane]=maximum;winners[lane]=winner;
    __syncthreads();
    for(int stride=128;stride;stride/=2){
        if(lane<stride){
            float other=maxima[lane+stride];int id=winners[lane+stride];
            if(id>=0&&(other>maxima[lane]||
               (other==maxima[lane]&&(winners[lane]<0||id<winners[lane])))){
                maxima[lane]=other;winners[lane]=id;
            }
        }
        __syncthreads();
    }
    if(!lane){ids[blockIdx.x]=winners[0];values[blockIdx.x]=maxima[0];}
}

static int head_batch_exact(Dsv4CudaTensor *t,const float *input,int batch,
                            float *scores,int *ids,float *values){
    if(!t||t->fmt!=12||!input||batch<1||batch>128||
       (!scores&&(!ids||!values)))return 0;
    Dev *c=ctx(t->device);
    if(!c||!ok(cudaSetDevice(t->device),"select batched head device"))return 0;
    size_t xb=(size_t)batch*t->I*sizeof(float),yb=(size_t)batch*t->O*sizeof(float);
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!scores&&(!buf((void**)&c->p1,&c->p1cap,(size_t)batch*sizeof(int))||
                !buf((void**)&c->p2,&c->p2cap,(size_t)batch*sizeof(float))))return 0;
    int success=ok(cudaMemcpyAsync(c->dx,input,xb,cudaMemcpyHostToDevice,c->stream),"batch head input upload");
    if(success){
        head_scores_batch_exact<<<dim3((t->O+127)/128,(batch+3)/4),128,0,c->stream>>>(
            (const __nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I,batch);
        success=ok(cudaGetLastError(),"batch head scores launch");
    }
    if(success&&scores)
        success=ok(cudaMemcpyAsync(scores,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"batch head scores download");
    else if(success){
        head_first_argmax<<<batch,256,0,c->stream>>>(c->dy,t->O,(int*)c->p1,c->p2);
        success=ok(cudaGetLastError(),"batch head argmax launch")&&
            ok(cudaMemcpyAsync(ids,c->p1,(size_t)batch*sizeof(int),cudaMemcpyDeviceToHost,c->stream),"batch head ids download")&&
            ok(cudaMemcpyAsync(values,c->p2,(size_t)batch*sizeof(float),cudaMemcpyDeviceToHost,c->stream),"batch head values download");
    }
    int drained=ok(cudaStreamSynchronize(c->stream),"batch head drain");
    return success&&drained;
}
extern "C" int dsv4_cuda_head_scores_batch_exact(Dsv4CudaTensor *t,const float *input,int batch,float *scores){
    return scores&&head_batch_exact(t,input,batch,scores,nullptr,nullptr);
}
extern "C" int dsv4_cuda_head_argmax_batch_exact(Dsv4CudaTensor *t,const float *input,int batch,int *ids,float *values){
    return head_batch_exact(t,input,batch,nullptr,ids,values);
}

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
                !h2d(c, k->rope_cos, cosv, rb, "RoPE cos upload") ||
                !h2d(c, k->rope_sin, sinv, rb, "RoPE sin upload") ||
                !h2d(c, k->comp_cos, ccos, rb, "compress RoPE cos upload") ||
                !h2d(c, k->comp_sin, csin, rb, "compress RoPE sin upload")))) {
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
/* A tensor on a latched-off peer is refused before any CUDA call (the CPU
 * reference takes over), so no work lands on that device and no error is
 * left behind for the next call on the primary. */
extern "C" int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t,float *y,const float*x,int groups){Dev*c=t?ctx(t->device):nullptr;if(!c||dev_lost(t->device)||groups<1||t->O%groups||!ok(cudaSetDevice(t->device),"select matvec device"))return 0;size_t xb=(size_t)t->I*groups*4,yb=(size_t)t->O*4;
    if(t->fmt!=4&&t->fmt!=8&&t->fmt!=9&&t->fmt!=16&&t->fmt!=32)return 0;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"activation upload"))return 0;
    if(t->fmt==8)run_mv<8>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==9)run_mv<9>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==4)run_mv<4>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==32)mv_f32<<<t->O,256,0,c->stream>>>((float*)t->w,c->dx,c->dy,t->O,t->I);
    else launch_bf16_one((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I,c->stream);
    return ok(cudaGetLastError(),"matvec launch")&&ok(cudaMemcpyAsync(y,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"result download")&&ok(cudaStreamSynchronize(c->stream),"matvec sync");}
extern "C" int dsv4_cuda_wo_decode(Dsv4CudaTensor *wa,Dsv4CudaTensor *wb,int groups,float *out,const float *context){
    Dev *c=wa?ctx(wa->device):nullptr;
    if(!c||!wb||!out||!context||groups<1||wa->O<1||wa->I<1||wb->O<1||wa->O%groups||
       (wa->fmt!=8&&wa->fmt!=9)||wb->fmt!=8||wb->device!=wa->device||wb->I!=wa->O||
       !ok(cudaSetDevice(wa->device),"select decode output device"))return 0;
    size_t cb=(size_t)wa->I*groups*sizeof(float),rb=(size_t)wa->O*sizeof(float),hb=(size_t)wb->O*sizeof(float);
    if(!buf((void**)&c->dx,&c->xcap,cb)||!buf((void**)&c->p1,&c->p1cap,rb)||
       !buf((void**)&c->dy,&c->ycap,hb))return 0;
    int success=ok(cudaMemcpyAsync(c->dx,context,cb,cudaMemcpyHostToDevice,c->stream),"decode output upload");
    if(success){
        if(wa->fmt==9)run_mv<9>((uint8_t*)wa->w,wa->scale,c->dx,c->p1,wa->O,wa->I,groups,c->stream);
        else run_mv<8>((uint8_t*)wa->w,wa->scale,c->dx,c->p1,wa->O,wa->I,groups,c->stream);
        bf16_round<<<(wa->O+255)/256,256,0,c->stream>>>(c->p1,wa->O);
        run_mv<8>((uint8_t*)wb->w,wb->scale,c->p1,c->dy,wb->O,wb->I,1,c->stream);
        bf16_round<<<(wb->O+255)/256,256,0,c->stream>>>(c->dy,wb->O);
        success=ok(cudaGetLastError(),"decode output launch")&&
            ok(cudaMemcpyAsync(out,c->dy,hb,cudaMemcpyDeviceToHost,c->stream),"decode output download");
    }
    // Drain even on failure before the caller reuses host input/output in fallback.
    int drained=ok(cudaStreamSynchronize(c->stream),"decode output sync");
    return success&&drained;
}
extern "C" int dsv4_cuda_matvec(Dsv4CudaTensor*t,float*y,const float*x){return dsv4_cuda_matvec_grouped(t,y,x,1);}
/* One token's projections of one input through several mirrors (bf16, fp8
 * or fp8-bf16) in one round trip: one upload, a launch per mirror (the
 * kernel each mirror's own call runs: mm_bf16_batch with a batch of one,
 * run_mv for fp8), the downloads, one sync. The outputs are what the
 * separate calls return, bitwise. */
extern "C" int dsv4_cuda_decode_proj_multi(Dsv4CudaTensor*const*ts,int n,const float*x,float*const*ys){
    if(!ts||!ys||n<1||n>8||!x||!ts[0])return 0;
    Dev*c=ctx(ts[0]->device);
    if(!c||!ok(cudaSetDevice(ts[0]->device),"select decode proj device"))return 0;
    size_t xb=(size_t)ts[0]->I*4,yb=0;
    for(int i=0;i<n;i++){
        if(!ts[i]||!ys[i]||ts[i]->device!=ts[0]->device||ts[i]->I!=ts[0]->I)return 0;
        int f=ts[i]->fmt;if(f!=16&&f!=8&&f!=9)return 0;
        yb+=(size_t)ts[i]->O*4;
    }
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"decode proj upload"))return 0;
    /* DSV4_PROJ_ONEDL (1): the outputs come back as one download of the
     * contiguous dy into a host block and are scattered after the sync; a
     * pageable download per output waits for its kernel on the host each
     * time (n stalls in a row where one serves). 0 = a download per output. */
    static int onedl=-1;
    if(onedl<0){const char*e=getenv("DSV4_PROJ_ONEDL");onedl=!(e&&*e=='0');}
    if(onedl&&c->pjhcap<yb){free(c->pjh);c->pjh=malloc(yb);c->pjhcap=c->pjh?yb:0;if(!c->pjh)onedl=0;}
    static int kprof=-1;
    if(kprof<0){const char*e=getenv("DSV4_PROJ_KPROF");kprof=e&&*e&&*e!='0';}
    cudaEvent_t ev[9];
    if(kprof){for(int i=0;i<=n;i++)cudaEventCreate(&ev[i]);cudaEventRecord(ev[0],c->stream);}
    size_t off=0;
    for(int i=0;i<n;i++){
        size_t ob=(size_t)ts[i]->O*4;float*dy=(float*)((char*)c->dy+off);
        if(ts[i]->fmt==16)launch_bf16_one((__nv_bfloat16*)ts[i]->w,c->dx,dy,ts[i]->O,ts[i]->I,c->stream);
        else if(ts[i]->fmt==8)run_mv<8>((uint8_t*)ts[i]->w,ts[i]->scale,c->dx,dy,ts[i]->O,ts[i]->I,1,c->stream);
        else run_mv<9>((uint8_t*)ts[i]->w,ts[i]->scale,c->dx,dy,ts[i]->O,ts[i]->I,1,c->stream);
        if(kprof)cudaEventRecord(ev[i+1],c->stream);
        if(!ok(cudaGetLastError(),"decode proj launch")||
           (!onedl&&!ok(cudaMemcpyAsync(ys[i],dy,ob,cudaMemcpyDeviceToHost,c->stream),"decode proj download")))return 0;
        off+=ob;
    }
    if(onedl&&!ok(cudaMemcpyAsync(c->pjh,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"decode proj download"))return 0;
    int done=ok(cudaStreamSynchronize(c->stream),"decode proj sync");
    if(done&&onedl){size_t o=0;for(int i=0;i<n;i++){size_t ob=(size_t)ts[i]->O*4;memcpy(ys[i],(char*)c->pjh+o,ob);o+=ob;}}
    if(kprof){
        fprintf(stderr,"projkprof n=%d",n);
        for(int i=0;i<n;i++){float ms=0;cudaEventElapsedTime(&ms,ev[i],ev[i+1]);fprintf(stderr," [%d]%s%dx%d=%.3f",i,ts[i]->fmt==16?"bf16":"fp8",ts[i]->O,ts[i]->I,ms);}
        fprintf(stderr," ms\n");
        for(int i=0;i<=n;i++)cudaEventDestroy(ev[i]);
    }
    return done;
}
/* V4_ATTN_BATCH_PROJ (optional export): up to 6 decode fp8 matvecs over the same input x
 * in ONE device round trip: one x upload, one run_mv<8> per tensor (the kernel
 * dsv4_cuda_matvec uses, groups=1), one D2H of all outputs into stage (output i at
 * the sum of rows[0..i-1]), one sync. Each output is therefore bit-identical to the
 * per-call dsv4_cuda_matvec. Refuses (0) anything else; the caller then runs the
 * per-call path. Output buffers are per device (index = position in g). */
static float *pj_out[16];static size_t pj_cap[16];
extern "C" int dsv4_cuda_proj_batch(int n,Dsv4CudaTensor*const*t,const int*rows,const float*x,float*stage){
    if(n<1||n>6||!t||!rows||!x||!stage||!t[0])return 0;
    int d=t[0]->device,I=t[0]->I;size_t total=0;
    if(I<1)return 0;
    for(int i=0;i<n;i++){
        if(!t[i]||t[i]->device!=d||t[i]->I!=I||t[i]->O!=rows[i]||rows[i]<1||t[i]->fmt!=8)return 0;
        total+=(size_t)rows[i];
    }
    Dev*c=ctx(d);if(!c||!ok(cudaSetDevice(d),"select proj batch device"))return 0;
    int gi=(int)(c-g);
    size_t need=total*sizeof(float);
    if(pj_cap[gi]<need){
        if(pj_out[gi]&&!ok(cudaFree(pj_out[gi]),"proj batch free"))return 0;
        pj_out[gi]=nullptr;pj_cap[gi]=0;
        if(!alloc_ok(cudaMalloc((void**)&pj_out[gi],need),"proj batch buffer"))return 0;
        pj_cap[gi]=need;
    }
    float *out=pj_out[gi];
    if(!buf((void**)&c->dx,&c->xcap,(size_t)I*sizeof(float)))return 0;
    int good=ok(cudaMemcpyAsync(c->dx,x,(size_t)I*sizeof(float),cudaMemcpyHostToDevice,c->stream),"proj batch upload");
    size_t off=0;
    for(int i=0;good&&i<n;i++){
        run_mv<8>((uint8_t*)t[i]->w,t[i]->scale,c->dx,out+off,t[i]->O,I,1,c->stream);
        off+=(size_t)rows[i];
    }
    good=good&&ok(cudaGetLastError(),"proj batch launch")&&
         ok(cudaMemcpyAsync(stage,out,need,cudaMemcpyDeviceToHost,c->stream),"proj batch download");
    int drained=ok(cudaStreamSynchronize(c->stream),"proj batch sync");
    return good&&drained;
}
/* Engine close: frees the per-device proj batch output buffers (each on its own device). */
extern "C" void dsv4_cuda_proj_batch_release(void){
    for(int i=0;i<16;i++){
        if(!pj_out[i])continue;
        cudaSetDevice(g[i].id);cudaFree(pj_out[i]);
        pj_out[i]=nullptr;pj_cap[i]=0;
    }
}
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
    if(!c||dev_lost(t->device)||t->fmt!=4||!ok(cudaSetDevice(t->device),"select expert-group device"))return 0;int H=t->I,I=t->O;
    for(int e=0;e<count;e++)if(!gate[e]||!up[e]||!down[e]||!dev_ok(t->device,gate[e]->device)||!dev_ok(t->device,up[e]->device)||!dev_ok(t->device,down[e]->device)||gate[e]->fmt!=4||up[e]->fmt!=4||down[e]->fmt!=4||gate[e]->I!=H||gate[e]->O!=I||up[e]->I!=H||up[e]->O!=I||down[e]->I!=I||down[e]->O!=H)return 0;
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
/* Fused decode MoE (DSV4_CUDA_MOE_FUSED6=1, opt-in): when the six routed FP4 experts
 * and the FP8 shared expert of a layer are device mirrors of the decode shape
 * (6 experts, H=4096, I=2048), one cooperative launch over every SM does gate/up,
 * SwiGLU with the clamp, the fp8 activation round trip, down and the combine, with
 * two grid barriers between the phases, instead of dsv4_cuda_moe's launch sequence.
 * Each output keeps the launch sequence's arithmetic and order (tests/
 * test_dsv4_fused6_cuda.cu compares the two bit for bit). Any other shape, or a
 * device without cooperative launch, takes the launch sequence. */
struct V4FBar { unsigned arrive; unsigned pad0[31]; unsigned gen; unsigned pad1[31]; };
/* Grid barrier for a cooperative launch: the last block to arrive bumps the generation. */
__device__ __forceinline__ void v4f_bar_sync(V4FBar* b,unsigned n){
    __syncthreads();
    if(threadIdx.x==0){ volatile unsigned* vg=&b->gen; unsigned g=*vg; __threadfence();
        if(atomicAdd(&b->arrive,1u)==n-1u){ atomicExch(&b->arrive,0u); __threadfence(); atomicAdd(&b->gen,1u); }
        else { while(*vg==g){} }
        __threadfence(); }
    __syncthreads();
}
__device__ __forceinline__ float v4f_bfr(float v){ return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float v4f_act_fn(float g,float u,float w,float limit){ float gg=fminf(v4f_bfr(g),limit); float uu=fmaxf(-limit,fminf(v4f_bfr(u),limit));
    return v4f_bfr(w*(gg/(1.f+expf(-gg)))*uu); }
__device__ __forceinline__ float v4f_e8f(unsigned b){ return b? __uint_as_float(b<<23) : __uint_as_float(0x00400000u); }
/* A warp's row dot in the order of the launch sequence's 256-thread matvecs
 * (mv_fp4_grouped for the routed experts, mv<8> for the shared one): partial j sums
 * i = j, j+256, ... in order, then the shared-memory tree s[j] += s[j+n] for
 * n = 128, 64, ..., 1. Lane l holds the partials j = 8l .. 8l+7, so each step m reads
 * the eight contiguous weights m + 8l .. m + 8l + 7 with one vector load. The tree's
 * steps 128..8 pair the same register of lanes l and l + n/8 (shuffles), 4, 2, 1 pair
 * registers inside lane 0. Every output gets the same operands in the same order. */
__device__ __forceinline__ float v4f_tree(float p[8]){
    #pragma unroll
    for(int k=16;k;k>>=1){
        #pragma unroll
        for(int r=0;r<8;r++)p[r]=__fadd_rn(p[r],__shfl_down_sync(0xffffffffu,p[r],k));}
    p[0]=__fadd_rn(p[0],p[4]);p[1]=__fadd_rn(p[1],p[5]);p[2]=__fadd_rn(p[2],p[6]);p[3]=__fadd_rn(p[3],p[7]);
    p[0]=__fadd_rn(p[0],p[2]);p[1]=__fadd_rn(p[1],p[3]);
    return __fadd_rn(p[0],p[1]);}
/* The decodes give the same values as e4m3() and e8_table: the hardware e4m3
 * conversion is exact, and the scale is 2^(b-127) (2^-127 for b = 0) from its bits. */
__device__ __forceinline__ float v4f_e4m3(unsigned b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(b&0xFFu),__NV_E4M3);return __half2float(__half(r));}
/* fp8 e4m3 row, 128x128 UE8M0 blocks (srow: the row's block-row of scales): mv<8>'s
 * sum += x * (e4m3 * e8), which nvcc contracts to one fma. */
__device__ __forceinline__ float v4f_dot_fp8(const uint8_t* __restrict__ w,const uint8_t* __restrict__ srow,const float* xs,int I,int lane){
    float p[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for(int m=0;m<I;m+=256){
        int i=m+8*lane;
        uint2 u=__ldcs((const uint2*)(w+i));
        float sc=v4f_e8f(__ldg(srow+(i>>7)));
        const float4 *xp=(const float4*)(xs+i); float4 xa=xp[0],xb=xp[1];
        const float xv[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
        #pragma unroll
        for(int r=0;r<8;r++){unsigned byte=(r<4?u.x:u.y)>>(8*(r&3));
            p[r]=__fmaf_rn(xv[r],__fmul_rn(v4f_e4m3(byte),sc),p[r]);}}
    return v4f_tree(p);}
/* fp4 row, a UE8M0 scale per 32 weights (srow: the row's scales): mv_fp4_grouped's
 * sum += x * f4 * e8, i.e. (x * f4) rounded, then times the power of two (exact).
 * lut holds f4()'s sixteen values (code 8 is -0) in shared memory. */
__device__ __forceinline__ float v4f_dot_fp4(const uint8_t* __restrict__ w,const uint8_t* __restrict__ srow,const float* xs,const float* lut,int I,int lane){
    float p[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for(int m=0;m<I;m+=256){
        int i=m+8*lane;
        unsigned u=__ldcs((const unsigned*)(w+(i>>1)));
        float sc=v4f_e8f(__ldg(srow+(i>>5)));
        const float4 *xp=(const float4*)(xs+i); float4 xa=xp[0],xb=xp[1];
        const float xv[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
        #pragma unroll
        for(int r=0;r<8;r++)p[r]=__fadd_rn(p[r],__fmul_rn(__fmul_rn(xv[r],lut[(u>>(4*r))&15]),sc));}
    return v4f_tree(p);}
/* one block's quantization of a 2048-row like fp8_sim (RNE e4m3 incl. ties, == e4m3_round) */
__device__ void v4f_block_fp8_sim(const float* src,float* dst,float* scl){
    const int lane=threadIdx.x&31, wid=threadIdx.x>>5;
    for(int i=threadIdx.x;i<2048;i+=blockDim.x) dst[i]=__ldcg(src+i);
    __syncthreads();
    if(wid<16){ float m=0.f; for(int j=lane;j<128;j+=32) m=fmaxf(m,fabsf(dst[wid*128+j]));
        #pragma unroll
        for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,o));
        if(!lane){ m=fmaxf(m,1e-4f); float raw=m/448.f; int e; frexpf(raw,&e); float sc=ldexpf(1.f,e-1); if(sc<raw) sc*=2.f; scl[wid]=sc; } }
    __syncthreads();
    for(int i=threadIdx.x;i<2048;i+=blockDim.x){ float sc=scl[i>>7]; float q=fmaxf(-448.f,fminf(448.f,dst[i]/sc));
        __nv_fp8_storage_t c=__nv_cvt_float_to_fp8(q,__NV_SATFINITE,__NV_E4M3); __half_raw hr=__nv_cvt_fp8_to_halfraw(c,__NV_E4M3); __half h(hr);
        dst[i]=v4f_bfr(__half2float(h)*sc); }
    __syncthreads();
}
struct V4FMoeExp { const uint8_t *w1,*s1,*w3,*s3,*w2,*s2; float weight; int fp8; };
struct V4FMoeArgs { V4FMoeExp e[7]; const float* x; float* act; float* parts; float* y; V4FBar* bar; float limit; };
/* e[0..5]: routed FP4 experts, e[6]: the FP8 shared expert. Work is split evenly over
 * every warp of the grid; a shared-expert row counts twice (an fp8 row reads twice the
 * bytes of an fp4 row), so the odd units of its range are skipped. */
__global__ void __launch_bounds__(1024,1) v4f_moe_kernel(V4FMoeArgs a){
    __shared__ __align__(16) float xs[4096]; __shared__ __align__(16) float aq[2][2048]; __shared__ float lut[16]; __shared__ float scl[16];
    const int lane=threadIdx.x&31, wid=threadIdx.x>>5, nw=1024/32; const int nb=gridDim.x, b=blockIdx.x;
    if(threadIdx.x<16){ const float t[16]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6}; lut[threadIdx.x]=t[threadIdx.x]; }
    for(int i=threadIdx.x*4;i<4096;i+=1024*4) *(float4*)(xs+i)=__ldg((const float4*)(a.x+i));
    __syncthreads();
    auto gate_up=[&](int k,int o){ const V4FMoeExp& E=a.e[k]; float g,u;
        if(E.fp8){ g=v4f_dot_fp8(E.w1+(size_t)o*4096,E.s1+(o/128)*32,xs,4096,lane); u=v4f_dot_fp8(E.w3+(size_t)o*4096,E.s3+(o/128)*32,xs,4096,lane); }
        else { g=v4f_dot_fp4(E.w1+(size_t)o*2048,E.s1+(size_t)o*128,xs,lut,4096,lane); u=v4f_dot_fp4(E.w3+(size_t)o*2048,E.s3+(size_t)o*128,xs,lut,4096,lane); }
        if(!lane) a.act[k*2048+o]=v4f_act_fn(g,u,E.weight,a.limit); };
    auto down=[&](int k,int o,const float* aqk){ const V4FMoeExp& E=a.e[k]; float p;
        if(E.fp8) p=v4f_dot_fp8(E.w2+(size_t)o*2048,E.s2+(o/128)*16,aqk,2048,lane);
        else p=v4f_dot_fp4(E.w2+(size_t)o*1024,E.s2+(size_t)o*64,aqk,lut,2048,lane);
        if(!lane) a.parts[k*4096+o]=p; };
    const int W=nb*nw, gw=b*nw+wid;
    { const int C=8*2048; const int u0=(int)((long long)C*gw/W), u1=(int)((long long)C*(gw+1)/W);
      for(int u=u0;u<u1;u++){ if(u<12288) gate_up(u>>11,u&2047); else { int v=u-12288; if(!(v&1)) gate_up(6,v>>1); } } }
    v4f_bar_sync(a.bar,(unsigned)nb);
    /* A block's warps span at most two experts' down rows: quantize those activations once. */
    { const int C=8*4096; const int bu0=(int)((long long)C*(b*nw)/W), bu1=(int)((long long)C*((b+1)*nw)/W);
      auto kof=[](int u){ return u<24576? (u>>12) : 6; };
      const int k0=kof(bu0), k1=kof(bu1>bu0? bu1-1 : bu0);
      v4f_block_fp8_sim(a.act+k0*2048,aq[0],scl); if(k1!=k0) v4f_block_fp8_sim(a.act+k1*2048,aq[1],scl);
      const int u0=(int)((long long)C*gw/W), u1=(int)((long long)C*(gw+1)/W);
      for(int u=u0;u<u1;u++){ int k,o; if(u<24576){ k=u>>12; o=u&4095; } else { int v=u-24576; if(v&1) continue; o=v>>1; k=6; }
        down(k,o,k==k0? aq[0] : aq[1]); } }
    v4f_bar_sync(a.bar,(unsigned)nb);
    for(int o=b*1024+threadIdx.x;o<4096;o+=nb*1024){ float v=0.f;
        #pragma unroll
        for(int k=0;k<6;k++) v+=v4f_bfr(__ldcg(a.parts+k*4096+o));
        a.y[o]=v4f_bfr(v+v4f_bfr(__ldcg(a.parts+6*4096+o))); }
}

/* Peer-compute failures. pstep keeps a peer step's first error; peer_fail
 * drains the peer and peer_latch reports the error once, with the device, as
 * it latches the peer off. peer_drain waits out what was queued on the peer,
 * so no peer kernel still writes this device's p4 when it is reused, and
 * clears the error the step left. */
static bool pstep(cudaError_t e,const char*s,const char**why,cudaError_t*err){if(e==cudaSuccess)return true;if(!*why){*why=s;*err=e;}return false;}
static bool peer_drain(Dev*pc,int home){bool fine=cudaSetDevice(pc->id)==cudaSuccess&&cudaStreamSynchronize(pc->stream)==cudaSuccess;cudaSetDevice(home);cudaGetLastError();return fine;}
/* DSV4_CUDA_PEER_FAULT=n[:lost|:sticky] (tests): the n-th peer-compute pass
 * fails at its join, after the peer's kernels ran. By default with an error
 * the devices survive (a launch out of resources), so the peer is latched off
 * and the call recomputes on the primary; with :lost the peer also counts as
 * unreadable; with :sticky the error is a kernel fault's (a launch failure),
 * which latches CUDA off for the whole process (cuda_lost), as a real one
 * does, without a fault on the device. One stderr line says when it fired. */
static bool peer_fault(bool*lost,cudaError_t*err){
    static long at=-1,n;static bool gone,stk;
    if(at<0){const char*e=getenv("DSV4_CUDA_PEER_FAULT");at=e&&*e?atol(e):0;gone=e&&strstr(e,":lost");stk=e&&strstr(e,":sticky");}
    if(at<=0||++n!=at)return false;*lost=gone||stk;*err=stk?cudaErrorLaunchFailure:cudaErrorLaunchOutOfResources;
    fprintf(stderr,"[DSV4 CUDA] DSV4_CUDA_PEER_FAULT: peer-compute pass #%ld fails%s\n",at,stk?", a sticky fault":gone?", device lost":"");return true;}
/* Latch the peer off; peer_latch is for a peer already drained (alive: it
 * drained). -1: it still answers (an allocation or launch failure), so its
 * mirrors are readable and the caller computes their experts here, reading
 * them in place; 0: a sticky fault, its memory is not read again and the
 * caller must fall back to host copies. */
static int peer_latch(Dev*pc,int home,const char*why,cudaError_t err,bool alive){int d=pc->id;
    if(d>=0&&d<16&&!g_peer_lost[d]){g_peer_lost[d]=1;
        if(sticky(err))
            fprintf(stderr,"[DSV4 CUDA] peer device %d failed in the decode MoE (%s: %s); latched off. The fault is sticky, "
                           "so device %d's context is lost with it: see the next line\n",d,why?why:"peer step",cudaGetErrorString(err),home);
        else
            fprintf(stderr,"[DSV4 CUDA] peer device %d failed in the decode MoE (%s: %s); latched off, decode continues on device %d%s\n",
                    d,why?why:"peer step",cudaGetErrorString(err),home,alive?"":" (its memory is no longer read)");}
    if(sticky(err))cuda_lost(why?why:"peer step",err);
    return alive?-1:0;}
static int peer_fail(Dev*pc,int home,const char*why,cudaError_t err,bool lost){return peer_latch(pc,home,why,err,peer_drain(pc,home)&&!lost);}
/* One pass of dsv4_cuda_moe on c. With peer, experts mirrored on a device
 * peer_runs() accepts run there. 1 = done, 0 = failed, -1 = the peer failed
 * but still answers: run again without peer, which computes its experts here
 * from the same mirrors (bit-identical). Every exit after peer work was
 * queued drains the peer first. */

static int moe_once(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,Dsv4CudaTensor *const *down,const float *weights,int count,
        Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd,float limit,float *y,const float *x,Dev*c,int H,int I,bool peer){
    size_t hb=(size_t)H*4,ib=(size_t)count*I*4,parts=(size_t)count*H*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,hb)||!buf((void**)&c->p1,&c->p1cap,ib)||
       !buf((void**)&c->p2,&c->p2cap,ib)||!buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,parts)||
       !ok(cudaMemcpyAsync(c->dx,x,hb,cudaMemcpyHostToDevice,c->stream),"MoE activation upload"))return 0;
    bool all_local=true;
    for(int e=0;e<count;e++)
        if(gate[e]->device!=sg->device||up[e]->device!=sg->device||down[e]->device!=sg->device){all_local=false;break;}
    /* DSV4_CUDA_MOE_FUSED6=1 (opt-in, read on every call): the decode shape as one
     * cooperative v4f_moe_kernel launch. fnb = co-resident blocks per SM, probed once;
     * 0 once the device has no cooperative launch, a buffer failed or a launch was
     * refused: from then on the launch sequence below. The grid needs >= 8 blocks so
     * that one block's down rows span at most two experts (the kernel's aq[2]). */
    static int fnb=-1;
    const char*fused_env=getenv("DSV4_CUDA_MOE_FUSED6");
    if(fused_env&&atoi(fused_env)&&fnb<0){int coop=0;cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,sg->device);
        if(!coop){fnb=0;fprintf(stderr,"[DSV4 CUDA] cooperative launch unsupported: fused MoE stays off\n");}
        else{int nbb=0;if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbb,v4f_moe_kernel,1024,0)!=cudaSuccess){cudaGetLastError();nbb=0;}fnb=nbb;}}
    if(fused_env&&atoi(fused_env)&&all_local&&count==6&&H==4096&&I==2048&&fnb>0&&c->sms*fnb>=8){
        if((!c->v4fact&&!alloc_ok(cudaMalloc(&c->v4fact,7*2048*sizeof(float)),"fused act"))||
           (!c->v4fparts&&!alloc_ok(cudaMalloc(&c->v4fparts,7*4096*sizeof(float)),"fused parts"))||
           (!c->v4fbar&&!alloc_ok(cudaMalloc(&c->v4fbar,sizeof(V4FBar)),"fused barrier"))){
            fnb=0;fprintf(stderr,"[DSV4 CUDA] fused MoE buffers unavailable: launch sequence from now on\n");
        }else{
            V4FMoeArgs ma{};
            for(int k=0;k<6;k++)ma.e[k]={(uint8_t*)gate[k]->w,gate[k]->scale,(uint8_t*)up[k]->w,up[k]->scale,
                                          (uint8_t*)down[k]->w,down[k]->scale,weights[k],0};
            ma.e[6]={(uint8_t*)sg->w,sg->scale,(uint8_t*)su->w,su->scale,(uint8_t*)sd->w,sd->scale,1.f,1};
            ma.x=c->dx;ma.act=c->v4fact;ma.parts=c->v4fparts;ma.y=c->dy;ma.bar=(V4FBar*)c->v4fbar;ma.limit=limit;
            if(!ok(cudaMemsetAsync(c->v4fbar,0,sizeof(V4FBar),c->stream),"fused barrier clear"))return 0;
            void* kargs[]={&ma};
            cudaError_t rco=cudaLaunchCooperativeKernel((void*)v4f_moe_kernel,dim3(c->sms*fnb),dim3(1024),kargs,0,c->stream);
            if(rco!=cudaSuccess){cudaGetLastError();fnb=0;
                fprintf(stderr,"[DSV4 CUDA] fused MoE cooperative launch refused (code %d): launch sequence from now on\n",(int)rco);}
            else return ok(cudaGetLastError(),"fused MoE")&&
                   ok(cudaMemcpyAsync(y,c->dy,hb,cudaMemcpyDeviceToHost,c->stream),"MoE result download")&&
                   ok(cudaStreamSynchronize(c->stream),"MoE sync");
        }
    }

    Dev*pc=nullptr;
    if(moe_batched()){
        /* Peer compute (peer_runs): routed experts whose mirrors live on a
         * peer device run THERE on local weights, concurrently with this
         * device's experts. The token's input goes over P2P and each
         * expert's down-projection row lands straight in this device's p4
         * slot, so the reduce below is unchanged. Same kernels, same slots,
         * same order: bit-identical to computing every expert here. */
        int pdev=-1;unsigned char on_peer[16]={0};
        for(int e=0;e<count;e++){int d=gate[e]->device;
            if(peer&&d!=sg->device&&up[e]->device==d&&down[e]->device==d&&(pdev<0||pdev==d)&&peer_runs(sg->device,d)){pdev=d;on_peer[e]=1;}}
        pc=pdev>=0?ctx(pdev):nullptr;
        if(!buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)2*count*sizeof(MvDesc))||!buf((void**)&c->expert_weights,&c->expert_weightscap,(size_t)count*sizeof(float))||
           !ok(cudaMemcpyAsync(c->expert_weights,weights,(size_t)count*sizeof(float),cudaMemcpyHostToDevice,c->stream),"MoE weights upload"))return 0;
        if(pc){
            /* A stale error must not be blamed on the peer's launches below.
             * The fork is recorded on this device: its failure is ours. */
            cudaGetLastError();
            if(!ok(cudaEventRecord(c->fork,c->stream),"MoE fork record"))return 0;
            MvDesc pd[32];int k=0;const char*why=nullptr;cudaError_t err=cudaSuccess;
            bool fine=pstep(cudaSetDevice(pdev),"device select",&why,&err)&&
               pstep(buf((void**)&pc->dx,&pc->xcap,hb)&&buf((void**)&pc->p1,&pc->p1cap,ib)&&buf((void**)&pc->p2,&pc->p2cap,ib)&&buf((void**)&pc->p3,&pc->p3cap,ib)&&
                     buf((void**)&pc->mvdesc,&pc->mvdesccap,(size_t)2*count*sizeof(MvDesc))&&buf((void**)&pc->expert_weights,&pc->expert_weightscap,(size_t)count*sizeof(float))
                     ?cudaSuccess:cudaErrorMemoryAllocation,"buffer allocation",&why,&err)&&
               pstep(cudaStreamWaitEvent(pc->stream,c->fork,0),"fork wait",&why,&err)&&
               pstep(cudaMemcpyPeerAsync(pc->dx,pdev,c->dx,sg->device,hb,pc->stream),"activation copy",&why,&err)&&
               pstep(cudaMemcpyAsync(pc->expert_weights,weights,(size_t)count*sizeof(float),cudaMemcpyHostToDevice,pc->stream),"weights upload",&why,&err);
            if(fine){for(int e=0;e<count;e++)if(on_peer[e]){pd[k++]={(uint8_t*)gate[e]->w,gate[e]->scale,pc->dx,pc->p1+(long long)e*I};pd[k++]={(uint8_t*)up[e]->w,up[e]->scale,pc->dx,pc->p2+(long long)e*I};}
                fine=pstep(cudaMemcpyAsync(pc->mvdesc,pd,(size_t)k*sizeof(MvDesc),cudaMemcpyHostToDevice,pc->stream),"descriptors upload",&why,&err);}
            if(fine){mv_fp4_grouped<4><<<dim3((I+3)/4,k),256,0,pc->stream>>>(pc->mvdesc,k,I,H);
                expert_act_grouped<<<(count*I+255)/256,256,0,pc->stream>>>(pc->p3,pc->p1,pc->p2,pc->expert_weights,limit,count,I);fp8_sim<<<count*((I+127)/128),128,0,pc->stream>>>(pc->p3,count,I);
                k=0;for(int e=0;e<count;e++)if(on_peer[e])pd[k++]={(uint8_t*)down[e]->w,down[e]->scale,pc->p3+(long long)e*I,c->p4+(long long)e*H};
                fine=pstep(cudaMemcpyAsync(pc->mvdesc,pd,(size_t)k*sizeof(MvDesc),cudaMemcpyHostToDevice,pc->stream),"down descriptors upload",&why,&err);}
            if(fine){mv_fp4_grouped<4><<<dim3((H+3)/4,k),256,0,pc->stream>>>(pc->mvdesc,k,H,I);
                fine=pstep(cudaGetLastError(),"launch",&why,&err)&&pstep(cudaEventRecord(pc->join,pc->stream),"join record",&why,&err);}
            if(!fine)return peer_fail(pc,sg->device,why,err,false);
            if(!ok(cudaSetDevice(sg->device),"select MoE device")){peer_drain(pc,sg->device);return 0;}
        }
        MvDesc hd[32];int k=0;
        for(int e=0;e<count;e++)if(!on_peer[e]){hd[k++]={(uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I};hd[k++]={(uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I};}
        if(k){if(!ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)k*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"MoE descriptors upload")){if(pc)peer_drain(pc,sg->device);return 0;}
            mv_fp4_grouped<4><<<dim3((I+3)/4,k),256,0,c->stream>>>(c->mvdesc,k,I,H);}
        expert_act_grouped<<<(count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,count,I);fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
        k=0;for(int e=0;e<count;e++)if(!on_peer[e])hd[k++]={(uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H};
        if(k){if(!ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)k*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"MoE down descriptors upload")){if(pc)peer_drain(pc,sg->device);return 0;}
            mv_fp4_grouped<4><<<dim3((H+3)/4,k),256,0,c->stream>>>(c->mvdesc,k,H,I);}
        /* The join wait is queued on this device: the peer is to blame only
         * if it does not drain. */
        if(pc){cudaError_t e=cudaStreamWaitEvent(c->stream,pc->join,0);
            if(e!=cudaSuccess){if(!peer_drain(pc,sg->device))return peer_latch(pc,sg->device,"join wait",e,false);ok(e,"MoE join wait");return 0;}}
    }else for(int e=0;e<count;e++){run_mv<4>((uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I,I,H,1,c->stream);run_mv<4>((uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I,I,H,1,c->stream);expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3+(long long)e*I,1,I);run_mv<4>((uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H,H,I,1,c->stream);}
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,count,H);
    run_mv<8>((uint8_t*)sg->w,sg->scale,c->dx,c->p1,I,H,1,c->stream);run_mv<8>((uint8_t*)su->w,su->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);run_mv<8>((uint8_t*)sd->w,sd->scale,c->p3,c->p4,H,I,1,c->stream);
    moe_combine<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,H);
    int r=ok(cudaGetLastError(),"MoE launch")&&ok(cudaMemcpyAsync(y,c->dy,hb,cudaMemcpyDeviceToHost,c->stream),"MoE result download")&&ok(cudaStreamSynchronize(c->stream),"MoE sync");
    if(pc){
        /* This stream waited on the join, so the peer's work is done, unless
         * the peer faulted after it was queued: an asynchronous error there
         * does not show on this device, and y could hold stale peer rows.
         * Not covered: a faulted peer that never releases this stream's wait
         * on the join leaves the MoE sync above blocked. */
        bool lost=false;cudaError_t e=cudaEventSynchronize(pc->join);
        if(e==cudaSuccess)peer_fault(&lost,&e);
        if(e!=cudaSuccess)return peer_fail(pc,sg->device,"join",e,lost);
    }
    return r;
}
extern "C" int dsv4_cuda_moe(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,Dsv4CudaTensor *const *down,
        const float *weights,int count,Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd,float limit,float *y,const float *x){
    /* Runs on the shared expert's device. Routed experts may live on a peer
     * device (peer mirror pool): they run there (peer_runs), or this device
     * reads them in place over P2P/NVLink with DSV4_CUDA_PEER_COMPUTE=0,
     * DSV4_CUDA_BATCHED=0, a peer of another SM version, and for the rest of
     * a call whose peer failed. A latched-off peer's tensors are refused. */
    if(count<1||count>16||!gate||!up||!down||!sg||!su||!sd)return 0;Dsv4CudaTensor*t=gate[0];Dev*c=ctx(sg->device);
    if(!c||!t||dev_lost(sg->device)||t->fmt!=4||su->device!=sg->device||sd->device!=sg->device||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||!ok(cudaSetDevice(sg->device),"select MoE device"))return 0;
    int H=t->I,I=t->O;if(sg->I!=H||sg->O!=I||su->I!=H||su->O!=I||sd->I!=I||sd->O!=H)return 0;
    for(int e=0;e<count;e++)if(!gate[e]||!up[e]||!down[e]||!dev_ok(sg->device,gate[e]->device)||!dev_ok(sg->device,up[e]->device)||!dev_ok(sg->device,down[e]->device)||gate[e]->fmt!=4||up[e]->fmt!=4||down[e]->fmt!=4)return 0;
    int r=moe_once(gate,up,down,weights,count,sg,su,sd,limit,y,x,c,H,I,true);
    return r<0?moe_once(gate,up,down,weights,count,sg,su,sd,limit,y,x,c,H,I,false):r;
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
    size_t bytes=(size_t)3*count*sizeof(*host);Dev*tc=ctx(device);if(!tc||!ok(cudaMalloc(&set->table,bytes),"expert table allocation")||
       !h2d(tc,set->table,host,bytes,"expert table upload")){free(host);dsv4_cuda_expert_set_free(set);return nullptr;}
    free(host);return set;
}
static Dsv4CudaTensor *expert_view(uint8_t*w,uint8_t*scale,int O,int I,int device){
    Dsv4CudaTensor*t=(Dsv4CudaTensor*)calloc(1,sizeof(*t));if(!t)return nullptr;
    t->w=w;t->scale=scale;t->O=O;t->I=I;t->fmt=4;t->device=device;t->bytes=(long long)O*(I/2)+(long long)O*(I/32);return t;
}
/* Slots the batched-MoE bank needs for a model with `experts` routed experts:
 * the DeepGEMM kernels are compiled for a 256-expert grouping, the generic
 * kernels take the model's own count (smaller bank for pruned checkpoints). */
extern "C" int dsv4_cuda_expert_bank_slots(int experts){
#ifdef COLI_DSV4_DEEPGEMM
    return experts>=1&&experts<=256?256:0;
#else
    return experts>=1&&experts<=256?experts:0;
#endif
}
/* Device bytes per slot of dsv4_cuda_expert_bank_create(count,H,I,...) below
 * (FC1/FC2 fp4 weights and scales, DeepGEMM's scale copies, the pointer
 * table): the engine sizes the VRAM decode leaves free for a bank before one
 * exists. 12.75 MiB generic, 13.5 MiB DeepGEMM at H=4096, I=2048. */
extern "C" long long dsv4_cuda_expert_bank_slot_bytes(int H,int I){
    if(H<1||I<1)return 0;long long fc1_s=2LL*I*(H/32),fc2_s=(long long)H*(I/32);
    long long bytes=(long long)I*H+fc1_s+(long long)H*(I/2)+fc2_s+3*(long long)sizeof(ExpertPtr);
#ifdef COLI_DSV4_DEEPGEMM
    bytes+=fc1_s+fc2_s;
#endif
    return bytes;
}
extern "C" Dsv4CudaExpertSet *dsv4_cuda_expert_bank_create(int count,int H,int I,int device,
        Dsv4CudaTensor*sg,Dsv4CudaTensor*su,Dsv4CudaTensor*sd){
    int shared=sg||su||sd,J=sg?sg->O:0;
    if(count<1||H<1||I<1||(shared&&(!sg||!su||!sd))||(shared&&(sg->device!=device||su->device!=device||sd->device!=device||
       sg->fmt!=8||su->fmt!=8||sd->fmt!=8||sg->I!=H||su->I!=H||su->O!=J||sd->I!=J||sd->O!=H))||
       g_cuda_lost||!ok(cudaSetDevice(device),"select expert bank device"))return nullptr;
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
        /* A failed cudaMalloc is also the runtime's last error: clear it, or
         * the next launch check (cudaGetLastError) reports this OOM and fails
         * a chunk that never touched the missing bank. */
        (void)cudaGetLastError();
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
/* DSV4_CUDA_PIN_CAP_GB (20): total page-locked expert-slab bytes, in GiB. The
 * RAM expert cache can exceed 20 GB (--ram 40 -> ~32 GB of slabs); slabs past
 * the cap upload through pageable staging at roughly half the PCIe rate.
 * Unset, empty, unparsable or NaN: 20; negative: 0 (nothing pinned); inf or
 * anything above 1 PiB: 1 PiB, i.e. no cap (the size_t conversion of a larger
 * double is undefined). */
static size_t pin_cap_bytes(){
    const char*e=getenv("DSV4_CUDA_PIN_CAP_GB");char*end=nullptr;double gb=e&&*e?strtod(e,&end):20.0;
    if(e&&*e&&(end==e||gb!=gb))gb=20.0;if(gb<0)gb=0;if(gb>1048576.0)gb=1048576.0;
    return (size_t)(gb*(double)(1ULL<<30));}
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
    static const size_t cap=pin_cap_bytes();
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
    static const size_t cap=pin_cap_bytes();
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

static int bank_upload_stream(Dsv4CudaExpertSet*set,int e,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds,
        Dsv4CudaTensor**gate,Dsv4CudaTensor**up,Dsv4CudaTensor**down,int use_aux){
    if(!set||e<0||e>=set->count||!gw||!gs||!uw||!us||!dw||!ds||!gate||!up||!down||*gate||*up||*down||
       !ok(cudaSetDevice(set->device),"select expert bank upload device"))return 0;
    int H=set->H,I=set->I;size_t w1=(size_t)I*(H/2),s1=(size_t)I*(H/32),w2=(size_t)H*(I/2),s2=(size_t)H*(I/32);
    uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
    uint8_t*fc2=set->bank_fc2+(size_t)e*w2,*fc2s=set->bank_fc2_scale+(size_t)e*s2;
    Dev*c=ctx(set->device);if(!c)return 0;
    /* use_aux: the double-buffer prefetch worker uploads layer L+1 on the
     * aux stream WHILE the compute stream chews layer L. Everything —
     * copies, DeepGEMM scale packing, the pointer table — rides the chosen
     * stream and drains it before returning, so the caller may release the
     * host slab immediately and the compute stream never stalls. Pageable
     * sources are safe under cudaMemcpyAsync (the copy is staged before it
     * returns with respect to the host buffer). */
    cudaStream_t s=use_aux?c->aux:c->stream;
    if(!ok(cudaMemcpyAsync(fc1,gw,w1,cudaMemcpyHostToDevice,s),"expert gate upload")||
       !ok(cudaMemcpyAsync(fc1+w1,uw,w1,cudaMemcpyHostToDevice,s),"expert up upload")||
       !ok(cudaMemcpyAsync(fc1s,gs,s1,cudaMemcpyHostToDevice,s),"expert gate scale upload")||
       !ok(cudaMemcpyAsync(fc1s+s1,us,s1,cudaMemcpyHostToDevice,s),"expert up scale upload")||
       !ok(cudaMemcpyAsync(fc2,dw,w2,cudaMemcpyHostToDevice,s),"expert down upload")||
       !ok(cudaMemcpyAsync(fc2s,ds,s2,cudaMemcpyHostToDevice,s),"expert down scale upload"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    dg_pack_weight_scale<<<(2*I*(H/32/4)+255)/256,256,0,s>>>(set->bank_fc1_dg_scale,fc1s,e,2*I,H/32);
    dg_pack_weight_scale<<<(H*(I/32/4)+255)/256,256,0,s>>>(set->bank_fc2_dg_scale,fc2s,e,H,I/32);
    if(!ok(cudaGetLastError(),"expert DeepGEMM scale packing"))return 0;
#endif
    Dsv4CudaTensor*g=expert_view(fc1,fc1s,I,H,set->device),*u=expert_view(fc1+w1,fc1s+s1,I,H,set->device),*d=expert_view(fc2,fc2s,H,I,set->device);
    if(!g||!u||!d){free(g);free(u);free(d);return 0;}
    ExpertPtr p[3]={{(uint8_t*)g->w,g->scale},{(uint8_t*)u->w,u->scale},{(uint8_t*)d->w,d->scale}};
    if(!ok(cudaMemcpyAsync(set->table+e,p,sizeof(*p),cudaMemcpyHostToDevice,s),"expert gate table upload")||
       !ok(cudaMemcpyAsync(set->table+set->count+e,p+1,sizeof(*p),cudaMemcpyHostToDevice,s),"expert up table upload")||
       !ok(cudaMemcpyAsync(set->table+2*set->count+e,p+2,sizeof(*p),cudaMemcpyHostToDevice,s),"expert down table upload")||
       !ok(cudaStreamSynchronize(s),"expert upload drain")){
        free(g);free(u);free(d);return 0;
    }
    *gate=g;*up=u;*down=d;return 1;
}

extern "C" int dsv4_cuda_expert_bank_upload(Dsv4CudaExpertSet*set,int e,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds,
        Dsv4CudaTensor**gate,Dsv4CudaTensor**up,Dsv4CudaTensor**down){
    return bank_upload_stream(set,e,gw,gs,uw,us,dw,ds,gate,up,down,0);
}

/* Double-buffer prefetch: same upload, aux stream (overlaps the compute
 * stream), drained before return. */
extern "C" int dsv4_cuda_expert_bank_upload_aux(Dsv4CudaExpertSet*set,int e,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds,
        Dsv4CudaTensor**gate,Dsv4CudaTensor**up,Dsv4CudaTensor**down){
    return bank_upload_stream(set,e,gw,gs,uw,us,dw,ds,gate,up,down,1);
}
extern "C" int dsv4_cuda_expert_bank_upload_tp2(Dsv4CudaExpertSet*set,int e,int rank,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds){
    if(!set||e<0||e>=set->count||(rank!=0&&rank!=1)||!gw||!gs||!uw||!us||!dw||!ds||
       !ok(cudaSetDevice(set->device),"select TP2 expert bank upload device"))return 0;
    int H=set->H,J=set->I,I=2*J;size_t row_w=H/2,row_s=H/32;
    size_t w1=(size_t)J*row_w,s1=(size_t)J*row_s,w2=(size_t)H*(J/2),s2=(size_t)H*(J/32);
    uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
    uint8_t*fc2=set->bank_fc2+(size_t)e*w2,*fc2s=set->bank_fc2_scale+(size_t)e*s2;
    Dev*tc=ctx(set->device);if(!tc)return 0;cudaStream_t ts=tc->stream;   /* ordered on the compute stream, see h2d */
    if(!ok(cudaMemcpyAsync(fc1,gw+(size_t)rank*w1,w1,cudaMemcpyHostToDevice,ts),"TP2 expert gate upload")||
       !ok(cudaMemcpyAsync(fc1+w1,uw+(size_t)rank*w1,w1,cudaMemcpyHostToDevice,ts),"TP2 expert up upload")||
       !ok(cudaMemcpyAsync(fc1s,gs+(size_t)rank*s1,s1,cudaMemcpyHostToDevice,ts),"TP2 expert gate scale upload")||
       !ok(cudaMemcpyAsync(fc1s+s1,us+(size_t)rank*s1,s1,cudaMemcpyHostToDevice,ts),"TP2 expert up scale upload")||
       !ok(cudaMemcpy2DAsync(fc2,J/2,dw+(size_t)rank*(J/2),I/2,J/2,H,cudaMemcpyHostToDevice,ts),"TP2 expert down upload")||
       !ok(cudaMemcpy2DAsync(fc2s,J/32,ds+(size_t)rank*(J/32),I/32,J/32,H,cudaMemcpyHostToDevice,ts),"TP2 expert down scale upload")||
       !ok(cudaStreamSynchronize(ts),"TP2 expert upload drain"))return 0;
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
    Dev*hc=ctx(set->device);if(!hc||!ok(cudaMalloc(&set->hash,bytes),"hash router allocation")||!h2d(hc,set->hash,map,bytes,"hash router upload"))return 0;
    set->vocab=vocab;set->topk=topk;return 1;
}
/* The host-routed resident path's exact operation/rounding order, with routes
 * and descriptors retained on device. One activation upload and one download. */
extern "C" int dsv4_cuda_resident_route_moe(Dsv4CudaExpertSet *set,
        Dsv4CudaTensor *gate,Dsv4CudaTensor *bias,const int *fixed,
        float scale,float limit,float *out,const float *in){
    Dev *c=set?ctx(set->device):nullptr;
    if(!c||!gate||!in||!out||set->count!=256||!set->table||!set->sg||!set->su||!set->sd||
       gate->device!=set->device||gate->fmt!=32||gate->O!=256||gate->I!=set->H||
       (bias&&(bias->device!=set->device||bias->fmt!=32||bias->O*bias->I<256)))return 0;
    if(fixed)for(int k=0;k<6;k++)if(fixed[k]<0||fixed[k]>=256)return 0;
    if(!ok(cudaSetDevice(set->device),"select resident route device"))return 0;
    int H=set->H,I=set->I,K=6;size_t hb=(size_t)H*4,ib=(size_t)K*I*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,H>256?hb:256*4)||
       !buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,(size_t)K*H*4)||
       !buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)3*K*sizeof(MvDesc))||
       !buf((void**)&c->expert_weights,&c->expert_weightscap,K*sizeof(float))||
       !buf((void**)&c->expert_ids,&c->expert_idcap,K*sizeof(int))||
       !ok(cudaMemcpyAsync(c->dx,in,hb,cudaMemcpyHostToDevice,c->stream),"resident MoE input"))return 0;
    mv_f32<<<256,256,0,c->stream>>>((float*)gate->w,c->dx,c->dy,256,H);
    if(fixed&&!ok(cudaMemcpyAsync(c->expert_ids,fixed,K*sizeof(int),cudaMemcpyHostToDevice,c->stream),"resident fixed routes"))return 0;
    route_top6<<<1,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,fixed!=nullptr,scale);
    sort_routes6<<<1,1,0,c->stream>>>(c->expert_ids,c->expert_weights);
    build_moe_desc<<<1,K,0,c->stream>>>(c->mvdesc,set->table,c->expert_ids,c->dx,c->p1,c->p2,c->p3,c->p4,K,256,H,I);
    mv_fp4_grouped<4><<<dim3((I+3)/4,2*K),256,0,c->stream>>>(c->mvdesc,2*K,I,H);
    expert_act_grouped<<<(K*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,K,I);
    fp8_sim<<<K*((I+127)/128),128,0,c->stream>>>(c->p3,K,I);
    mv_fp4_grouped<4><<<dim3((H+3)/4,K),256,0,c->stream>>>(c->mvdesc+2*K,K,H,I);
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,K,H);
    run_mv<8>((uint8_t*)set->sg->w,set->sg->scale,c->dx,c->p1,I,H,1,c->stream);
    run_mv<8>((uint8_t*)set->su->w,set->su->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);
    fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);
    run_mv<8>((uint8_t*)set->sd->w,set->sd->scale,c->p3,c->p4,H,I,1,c->stream);
    moe_combine<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,H);
    return ok(cudaGetLastError(),"resident MoE launch")&&
        ok(cudaMemcpyAsync(out,c->dy,hb,cudaMemcpyDeviceToHost,c->stream),"resident MoE output")&&
        ok(cudaStreamSynchronize(c->stream),"resident MoE sync");
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
    double t_expert=0,t_shared=0;std::chrono::steady_clock::time_point ta,tb;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);ta=std::chrono::steady_clock::now();}
    if(!dg_moe_batch_contiguous(c,set,input->data,c->expert_ids,c->expert_weights,count,limit,output->data))return 0;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_expert=std::chrono::duration<double>(tb-ta).count();ta=tb;}
    Dsv4CudaTensor*sg=set->sg,*su=set->su,*sd=set->sd;if(!sg||!su||!sd||sg->O!=2048||su->O!=2048||sd->I!=2048)return 0;
    size_t mid=(size_t)count*2048*sizeof(float),hb=(size_t)count*4096*sizeof(float);if(!buf((void**)&c->p1,&c->p1cap,mid)||!buf((void**)&c->p2,&c->p2cap,mid)||!buf((void**)&c->p3,&c->p3cap,mid)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!dg_dense_batch_dispatch(c,sg,input->data,c->p1,count)||!dg_dense_batch_dispatch(c,su,input->data,c->p2,count))return 0;swiglu_rows<<<((long long)count*2048+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,limit,count,2048);if(!dg_dense_batch_dispatch(c,sd,c->p3,c->p4,count))return 0;add_rows<<<((long long)count*4096+255)/256,256,0,c->stream>>>(output->data,c->p4,count*4096);
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_shared=std::chrono::duration<double>(tb-ta).count();
        fprintf(stderr,"[DSV4 CUDA] moe-batch prof count=%d experts=%.3fs shared=%.3fs\n",count,t_expert,t_shared);prof_calls++;}
    return ok(cudaGetLastError(),"batched MoE launch");
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
    /* Any bank of up to 256 slots: the grouped kernels index slots by expert
     * id, so expert-pruned checkpoints use a bank sized to their count. */
    if(!c||!ids||!weights||!set||!output||count<1||count>128||set->device!=input->device||output->device!=input->device||
       set->count<1||set->count>256||set->H!=4096||set->I!=2048||!set->bank_fc1||
       input->elements<(long long)count*4096||output->elements<(long long)count*4096||
       !ok(cudaSetDevice(input->device),"select generic batched MoE device"))return 0;
    const int H=4096,I=2048,K=6;int routes=count*K;
    for(int r=0;r<routes;r++)if(ids[r]<0||ids[r]>=set->count)return 0;
    Dsv4CudaTensor*sg=set->sg,*su=set->su,*sd=set->sd;
    if(!sg||!su||!sd||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||sg->O!=I||su->O!=I||sd->I!=I||sg->I!=H||sd->O!=H)return 0;
    size_t w1=(size_t)I*(H/2),s1=(size_t)I*(H/32),w2=(size_t)H*(I/2),s2=(size_t)H*(I/32);
    size_t rb=(size_t)routes*I*sizeof(float),pb=(size_t)routes*H*sizeof(float),wb=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->p1,&c->p1cap,rb)||!buf((void**)&c->p2,&c->p2cap,rb)||!buf((void**)&c->p3,&c->p3cap,rb)||
       !buf((void**)&c->p4,&c->p4cap,pb)||!buf((void**)&c->expert_weights,&c->expert_weightscap,wb))return 0;
    if(!ok(cudaMemcpyAsync(c->expert_weights,weights,wb,cudaMemcpyHostToDevice,c->stream),"generic MoE weight upload"))return 0;
    /* Default: stream each routed expert once per chunk (mv_fp4_expert_major,
     * bit-identical); DSV4_MOE_EXPERT_MAJOR=0 keeps one matvec per route.
     * DSV4_MOE_EXPERT_MAJOR_VERIFY=1 runs both and counts bit mismatches (the
     * per-route results are kept, so it also runs over an explicit =0, which
     * it says once). Routes per pass (TB): 2 while the chunk averages fewer
     * than 4 routes per routed expert (small prompts, tail chunks), else 4.
     * On sm_86 TB=4 took ~1.4x TB=2's time at 1-2 routes per expert and
     * pulled ahead from ~3-4.5 (more skewed routing = later); every TB is
     * bit-identical. DSV4_MOE_EM_TB=2|3|4|6 forces one. */
    static int em_mode=-1,em_mode_env=1,em_verify=0,em_tb=0;static unsigned long long em_calls=0,em_bad1=0,em_bad2=0;
    if(em_mode<0){const char*v=getenv("DSV4_MOE_EXPERT_MAJOR");em_mode_env=v?atoi(v)!=0:1;em_mode=em_mode_env;
        v=getenv("DSV4_MOE_EM_TB");if(v&&*v){em_tb=atoi(v);
            if(em_tb!=2&&em_tb!=3&&em_tb!=4&&em_tb!=6){fprintf(stderr,"[DSV4 CUDA] DSV4_MOE_EM_TB=%s ignored (2, 3, 4 or 6): TB by route density\n",v);em_tb=0;}}
        v=getenv("DSV4_MOE_EXPERT_MAJOR_VERIFY");em_verify=v&&atoi(v)!=0;
        if(em_verify&&!em_mode){em_mode=1;fprintf(stderr,"[DSV4 CUDA] DSV4_MOE_EXPERT_MAJOR_VERIFY=1 overrides DSV4_MOE_EXPERT_MAJOR=0: the expert-major kernel runs beside the per-route one, whose results are kept\n");}}
    int nex=0;const int*em_ex=nullptr,*em_off=nullptr,*em_rt=nullptr;
    if(em_mode){
        int cnt[256]={0},slot_of[256];for(int r=0;r<routes;r++)cnt[ids[r]]++;
        std::vector<int> hm((size_t)set->count+1+set->count+routes);int*hex=hm.data(),*hoff=hex+set->count,*hrt=hoff+set->count+1;
        hoff[0]=0;for(int e=0;e<set->count;e++)if(cnt[e]){slot_of[e]=nex;hex[nex]=e;hoff[nex+1]=hoff[nex]+cnt[e];nex++;}
        std::vector<int> fill(hoff,hoff+nex);for(int r=0;r<routes;r++)hrt[fill[slot_of[ids[r]]]++]=r;
        /* pack as [experts nex][offsets nex+1][routes] */
        std::vector<int> packed((size_t)nex+nex+1+routes);std::copy(hex,hex+nex,packed.begin());std::copy(hoff,hoff+nex+1,packed.begin()+nex);
        std::copy(hrt,hrt+routes,packed.begin()+2*nex+1);
        if(!buf((void**)&c->emaj,&c->emajcap,packed.size()*sizeof(int))||
           !ok(cudaMemcpyAsync(c->emaj,packed.data(),packed.size()*sizeof(int),cudaMemcpyHostToDevice,c->stream),"expert-major route lists"))return 0;
        em_ex=c->emaj;em_off=c->emaj+nex;em_rt=c->emaj+2*nex+1;
        /* verify is a diagnostic: without memory for its buffers it switches
         * itself off for good and DSV4_MOE_EXPERT_MAJOR's own setting applies
         * again (an overridden 0 sends this call down the per-route path) */
        if(em_verify&&(!buf((void**)&c->emv1,&c->emv1cap,rb)||!buf((void**)&c->emv2,&c->emv2cap,rb)||!buf((void**)&c->emv4,&c->emv4cap,pb)||
           (!c->emcount&&!alloc_ok(cudaMalloc(&c->emcount,2*sizeof(unsigned long long)),"expert-major verify counter")))){
            em_verify=0;em_mode=em_mode_env;fprintf(stderr,"[DSV4 CUDA] DSV4_MOE_EXPERT_MAJOR_VERIFY off: no memory for its buffers\n");}
    }
    const int tb=em_tb?em_tb:routes<4*nex?2:4;
    /* the per-route kernel's descriptor tables: per-route mode and verify only */
    const bool per_route=!em_mode||em_verify;std::vector<MvDesc> hd;
    if(per_route){
        if(!buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)routes*2*sizeof(MvDesc)))return 0;
        hd.resize((size_t)routes*2);
        for(int r=0;r<routes;r++){int e=ids[r],t=r/K;
            const uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
            const float*x=input->data+(long long)t*H;
            hd[2*r]={fc1,fc1s,x,c->p1+(long long)r*I};
            hd[2*r+1]={fc1+w1,fc1s+s1,x,c->p2+(long long)r*I};}
        if(!ok(cudaMemcpyAsync(c->mvdesc,hd.data(),(size_t)routes*2*sizeof(MvDesc),cudaMemcpyHostToDevice,c->stream),"generic MoE fc1 descriptors"))return 0;
        mv_fp4_grouped<4><<<dim3((I+3)/4,routes*2),256,0,c->stream>>>(c->mvdesc,routes*2,I,H);
    }
    if(em_mode){
        float*y0=em_verify?c->emv1:c->p1,*y1=em_verify?c->emv2:c->p2;
        run_fp4_expert_major(tb,set->bank_fc1,set->bank_fc1_scale,2*w1,2*s1,
            em_ex,em_off,em_rt,nex,input->data,H,K,y0,y1,I,I,2*I,H,c->stream);
        if(em_verify){
            if(!ok(cudaMemsetAsync(c->emcount,0,2*sizeof(unsigned long long),c->stream),"verify clear"))return 0;
            count_bit_mismatch<<<((long long)routes*I+255)/256,256,0,c->stream>>>(c->p1,c->emv1,(long long)routes*I,c->emcount);
            count_bit_mismatch<<<((long long)routes*I+255)/256,256,0,c->stream>>>(c->p2,c->emv2,(long long)routes*I,c->emcount);
        }
    }
    expert_act_grouped<<<((long long)routes*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,routes,I);
    fp8_sim<<<routes*((I+127)/128),128,0,c->stream>>>(c->p3,routes,I);
    if(per_route){
        /* down: descriptors must not be overwritten while fc1 still reads them —
         * the copy is stream-ordered after the launches above, so reuse is safe. */
        for(int r=0;r<routes;r++){int e=ids[r];
            hd[r]={set->bank_fc2+(size_t)e*w2,set->bank_fc2_scale+(size_t)e*s2,c->p3+(long long)r*I,c->p4+(long long)r*H};}
        if(!ok(cudaMemcpyAsync(c->mvdesc,hd.data(),(size_t)routes*sizeof(MvDesc),cudaMemcpyHostToDevice,c->stream),"generic MoE down descriptors"))return 0;
        mv_fp4_grouped<4><<<dim3((H+3)/4,routes),256,0,c->stream>>>(c->mvdesc,routes,H,I);
    }
    if(em_mode){
        float*y=em_verify?c->emv4:c->p4;
        run_fp4_expert_major(tb,set->bank_fc2,set->bank_fc2_scale,w2,s2,
            em_ex,em_off,em_rt,nex,c->p3,I,1,y,y,H,H,H,I,c->stream);
        if(em_verify){
            count_bit_mismatch<<<((long long)routes*H+255)/256,256,0,c->stream>>>(c->p4,c->emv4,(long long)routes*H,c->emcount+1);
            unsigned long long bad[2]={0,0};
            if(!ok(cudaMemcpyAsync(bad,c->emcount,sizeof(bad),cudaMemcpyDeviceToHost,c->stream),"verify read")||
               !ok(cudaStreamSynchronize(c->stream),"verify sync"))return 0;
            em_calls++;em_bad1+=bad[0];em_bad2+=bad[1];
            if(bad[0]||bad[1]||em_calls%100==0)
                fprintf(stderr,"moe-expert-major verify calls=%llu fc1_mismatch=%llu fc2_mismatch=%llu (this call %llu/%llu of %lld/%lld)\n",
                        em_calls,em_bad1,em_bad2,bad[0],bad[1],(long long)routes*I*2,(long long)routes*H);
        }
    }
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
/* A latched-off peer's context may refuse the frees (sticky fault): read the
 * error so the next launch check does not report it. */
extern "C" void dsv4_cuda_tensor_free(Dsv4CudaTensor*t){if(!t)return;cudaSetDevice(t->device);if(t->own_w&&t->w)cudaFree(t->w);if(t->bf16_cache)cudaFree(t->bf16_cache);if(t->own_scale&&t->scale)cudaFree(t->scale);if(t->own_dg_scale&&t->dg_scale)cudaFree(t->dg_scale);if(dev_lost(t->device))cudaGetLastError();free(t);}
extern "C" long long dsv4_cuda_tensor_bytes(const Dsv4CudaTensor*t){return t?t->bytes:0;}
extern "C" int dsv4_cuda_tensor_device(const Dsv4CudaTensor*t){return t?t->device:-1;}
