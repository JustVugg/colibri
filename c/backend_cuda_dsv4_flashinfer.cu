#include "backend_cuda_dsv4_flashinfer.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <flashinfer/gemm/group_gemm_mxfp4_groupwise_sm120.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <cstdio>
#include <cstdlib>

namespace flashinfer::group_gemm {
template <bool SwapAB,int ScaleGranularity,typename ScaleConfig,typename ElementA,typename ElementB,
          typename ElementSFA,typename ElementSFB,typename ElementD,typename ProblemShape,typename StrideA,
          typename StrideB,typename StrideD,typename LayoutSFA,typename LayoutSFB>
__global__ void compute_pointer_group_gemm_args(
    ElementA *packed_a,ElementB *B,ElementSFA *packed_sfa,ElementSFB *SFB,ElementD *D,int *m_indptr,
    int n,int k,int num_groups,ProblemShape *problem_sizes,const ElementA **A_ptr,const ElementB **B_ptr,
    const ElementSFA **SFA_ptr,const ElementSFB **SFB_ptr,ElementD **D_ptr,StrideA *stride_A,
    StrideB *stride_B,StrideD *stride_D,LayoutSFA *layout_SFA,LayoutSFB *layout_SFB){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=num_groups)return;
    auto A_list=reinterpret_cast<const ElementA *const *>(packed_a);
    auto SFA_list=reinterpret_cast<const ElementSFA *const *>(packed_sfa);
    constexpr size_t mn_align=128,k_align=(size_t)ScaleGranularity*4;
    size_t sf_n=((size_t)n+mn_align-1)/mn_align*mn_align,swk=((size_t)k+k_align-1)/k_align*k_align,sf_k=swk/ScaleGranularity;
    asm volatile("griddepcontrol.wait;");asm volatile("griddepcontrol.launch_dependents;");
    int mo=m_indptr[i],mn=m_indptr[i+1];size_t m=(size_t)(mn-mo),sf_mo=((size_t)mo+(size_t)i*(mn_align-1))/mn_align*mn_align;
    static_assert(SwapAB,"pointer grouped GEMM is decode-only");
    problem_sizes[i]=ProblemShape(n,m,k);stride_A[i]=cutlass::make_cute_packed_stride(StrideA{},{n,k,1});
    stride_B[i]=cutlass::make_cute_packed_stride(StrideB{},{m,k,1});stride_D[i]=cutlass::make_cute_packed_stride(StrideD{},{n,m,1});
    A_ptr[i]=A_list[i];B_ptr[i]=B+(size_t)mo*k;D_ptr[i]=D+(size_t)mo*n;
    layout_SFA[i]=ScaleConfig::tile_atom_to_shape_SFA(make_shape((int)sf_n,(int)m,(int)swk,1));SFA_ptr[i]=SFA_list[i];
    layout_SFB[i]=ScaleConfig::tile_atom_to_shape_SFB(make_shape((int)sf_n,(int)m,(int)swk,1));SFB_ptr[i]=SFB+sf_mo*sf_k;
}
#define compute_sm120_cutlass_group_gemm_args compute_pointer_group_gemm_args
INSTANTIATE_GROUP_GEMM_MXFP4_GROUPWISE_SM120(
    128,32,128,true,cutlass::float_e4m3_t,cutlass::float_e2m1_t,
    cutlass::float_ue8m0_t,cutlass::float_ue8m0_t,cutlass::bfloat16_t,
    fp8,fp4,ue8m0,ue8m0,bf16)
#undef compute_sm120_cutlass_group_gemm_args
}
using namespace flashinfer::group_gemm;

namespace flashinfer::sparse_mla_sm120 {
bool launch_sparse_mla_decode_dsv4(ModelType,int,int,int,int,int,const bf16*,const uint8_t*,
                                   const int32_t*,bf16*,float*,bf16*,float*,const int*,
                                   const float*,const uint8_t*,const int32_t*,const int*,int,
                                   int,size_t,int,float,size_t,cudaStream_t);
}

struct Desc { const uint8_t *w,*scale;const float *x;float *y; };
struct FiCtx { int device=-1;const uint8_t **wp=nullptr,**sp=nullptr;uint8_t *ws=nullptr,*a=nullptr,*as=nullptr;__nv_bfloat16 *out=nullptr;int *indptr=nullptr;void *iw=nullptr,*fw=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;size_t wpc=0,spc=0,wsc=0,ac=0,asc=0,oc=0,ic=0; };
static FiCtx fc[16][2];
static int grow(void **p,size_t *cap,size_t n){if(*cap>=n)return 1;if(*p)cudaFree(*p);*p=nullptr;*cap=0;if(cudaMalloc(p,n)!=cudaSuccess)return 0;*cap=n;return 1;}
__device__ __forceinline__ int sf_off(int row,int col,int cols){return (col&3)+(col/4)*512+(row&31)*16+((row&127)/32)*4+(row/128)*128*((cols+3)&~3);}
__global__ void stage_pointer(const Desc*d,const uint8_t**wp,const uint8_t**sp,int count){int t=threadIdx.x;if(t<count){wp[t]=d[t].w;sp[t]=d[t].scale;}}
__global__ void quant_activation(const Desc*d,uint8_t*a,uint8_t*s,int count,int I){int g=blockIdx.x/(I/32),b=blockIdx.x%(I/32),i=b*32+threadIdx.x;float v=d[g].x[i],mx=fabsf(v);for(int m=16;m;m>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,m));float raw=fmaxf(mx,1e-4f)/448.f;int e;frexpf(raw,&e);float sc=ldexpf(1.f,e-1);if(sc<raw)sc*=2;int se;frexpf(sc,&se);uint8_t q=__nv_cvt_float_to_fp8(v/sc,__NV_SATFINITE,__NV_E4M3);for(int r=0;r<4;r++){a[((long long)g*4+r)*I+i]=q;if(!threadIdx.x)s[(long long)g*128*((I/32+3)&~3)+sf_off(r,b,I/32)]=(uint8_t)(se-1+127);}}
__global__ void make_indptr(int*p,int count){int i=threadIdx.x;if(i<=count)p[i]=i*4;}
__global__ void extract(const Desc*d,const __nv_bfloat16*out,int count,int O){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count*O){int g=p/O,o=p%O;d[g].y[o]=__bfloat162float(out[(long long)g*4*O+o]);}}

extern "C" int dsv4_flashinfer_grouped(const void *descriptors,int count,int O,int I,int device,cudaStream_t stream){if(!descriptors||count<1||count>16||O%128||I%128||device<0||device>=16)return 0;if(cudaSetDevice(device)!=cudaSuccess)return 0;FiCtx&c=fc[device][O>I];c.device=device;size_t sb=(size_t)count*O*(I/32),ab=(size_t)count*4*I,asb=(size_t)count*128*((I/32+3)&~3),ob=(size_t)count*4*O*2;
    if(!grow((void**)&c.wp,&c.wpc,count*sizeof(*c.wp))||!grow((void**)&c.sp,&c.spc,count*sizeof(*c.sp))||!grow((void**)&c.ws,&c.wsc,sb)||!grow((void**)&c.a,&c.ac,ab)||!grow((void**)&c.as,&c.asc,asb)||!grow((void**)&c.out,&c.oc,ob)||!grow((void**)&c.indptr,&c.ic,(count+1)*4)||(!c.iw&&cudaMalloc(&c.iw,8<<20)!=cudaSuccess)||(!c.fw&&cudaMalloc(&c.fw,64<<20)!=cudaSuccess))return 0;
    if(c.exec)return cudaGraphLaunch(c.exec,stream)==cudaSuccess;
    auto launch=[&]()->int{cudaMemsetAsync(c.as,0,asb,stream);stage_pointer<<<1,32,0,stream>>>((const Desc*)descriptors,c.wp,c.sp,count);quant_activation<<<count*(I/32),32,0,stream>>>((const Desc*)descriptors,c.a,c.as,count,I);make_indptr<<<1,32,0,stream>>>(c.indptr,count);cudaError_t e=CutlassMXFP4GroupwiseScaledGroupGEMMSM120<128,32,128,true,cutlass::float_e4m3_t,cutlass::float_e2m1_t,cutlass::float_ue8m0_t,cutlass::float_ue8m0_t,cutlass::bfloat16_t>(c.iw,8<<20,c.fw,64<<20,(cutlass::float_e4m3_t*)c.a,(cutlass::float_e2m1_t*)c.wp,(cutlass::float_ue8m0_t*)c.as,(cutlass::float_ue8m0_t*)c.sp,(cutlass::bfloat16_t*)c.out,c.indptr,O,I,count,stream,device);if(e!=cudaSuccess)return 0;extract<<<(count*O+255)/256,256,0,stream>>>((const Desc*)descriptors,c.out,count,O);return cudaGetLastError()==cudaSuccess;};
    static int graph=-1;if(graph<0){const char*p=getenv("DSV4_CUDA_FLASHINFER_GRAPH");graph=p&&atoi(p)!=0;}if(!graph)return launch();if(c.exec)return cudaGraphLaunch(c.exec,stream)==cudaSuccess;if(!launch()||cudaStreamSynchronize(stream)!=cudaSuccess||cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)!=cudaSuccess||!launch()||cudaStreamEndCapture(stream,&c.graph)!=cudaSuccess||cudaGraphInstantiate(&c.exec,c.graph,0)!=cudaSuccess)return 0;return 1;}
extern "C" int dsv4_flashinfer_sparse_mla(const void*q,const void*kv,const int*indices,void*mid,float*mid_lse,void*out,float*out_lse,const int*length,const float*sink,const void*extra_kv,const int*extra_indices,const int*extra_length,int extra_topk,int pbs_extra,size_t extra_stride,int heads,int topk,int tokens,int splits,int chunks,float scale,size_t stride,cudaStream_t stream){
    using namespace flashinfer::sparse_mla_sm120;
    if(!q||!kv||!indices||!mid||!mid_lse||!out||!length||heads<1||tokens<1||splits<1)return 0;
    return launch_sparse_mla_decode_dsv4(ModelType::DSV4,heads,topk,64,tokens,splits,
        (const bf16*)q,(const uint8_t*)kv,(const int32_t*)indices,(bf16*)mid,mid_lse,
        (bf16*)out,out_lse,length,sink,(const uint8_t*)extra_kv,(const int32_t*)extra_indices,
        extra_length,extra_topk,pbs_extra,extra_stride,chunks,scale,stride,stream);
}
extern "C" void dsv4_flashinfer_shutdown(void){for(auto&dev:fc)for(auto&c:dev)if(c.device>=0){cudaSetDevice(c.device);if(c.exec)cudaGraphExecDestroy(c.exec);if(c.graph)cudaGraphDestroy(c.graph);cudaFree(c.wp);cudaFree(c.sp);cudaFree(c.ws);cudaFree(c.a);cudaFree(c.as);cudaFree(c.out);cudaFree(c.indptr);cudaFree(c.iw);cudaFree(c.fw);c=FiCtx{};}}
