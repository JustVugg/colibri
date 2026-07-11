// Apple-GPU (Metal) backend for colibrì. Runtime-compiled shader (no Xcode needed),
// zero-copy over unified memory. See backend_metal.h and docs/plans/2026-07-10-*.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backend_metal.h"
#include <cstring>
#include <vector>
#include <mutex>

// ---- shader: general quantized GEMV, one threadgroup per output element (o,si) ----
// y[si,o] = (sum_i dequant(W[o,i]) * x[si,i]) * scale[o]. fmt: 0=f32 1=i8 2=i4 3=i2.
static const char *SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void mm_gemv(device const uchar* w      [[buffer(0)]],   // raw weight bytes
                    device const float* scale  [[buffer(1)]],   // [O]
                    device const float* x      [[buffer(2)]],   // [S,I]
                    device float*       y      [[buffer(3)]],   // [S,O]
                    constant int& S [[buffer(4)]], constant int& I [[buffer(5)]],
                    constant int& O [[buffer(6)]], constant int& fmt [[buffer(7)]],
                    uint tg   [[threadgroup_position_in_grid]],
                    uint lid  [[thread_position_in_threadgroup]],
                    uint tgsz [[threads_per_threadgroup]],
                    uint slane[[thread_index_in_simdgroup]],
                    uint sgid [[simdgroup_index_in_threadgroup]]) {
  int o  = tg % O;          // output row
  int si = tg / O;          // sequence position
  device const float* xr = x + (long)si * I;
  int I4 = (I & 3) ? 0 : (I / 4);   // vector path only when 4-aligned; else scalar tail covers all
  float acc = 0.0f;

  if (fmt == 1) {                                   // int8
    device const char* wr = (device const char*)(w) + (long)o * I;
    device const char4* w4 = (device const char4*)wr;
    device const float4* x4 = (device const float4*)xr;
    for (int c = lid; c < I4; c += tgsz) acc += dot(float4(w4[c]), x4[c]);
    for (int i = I4*4 + lid; i < I; i += tgsz) acc += float(wr[i]) * xr[i];
  } else if (fmt == 2) {                            // int4 packed, rb=(I+1)/2
    int rb = (I+1)/2;
    device const uchar* wr = w + (long)o * rb;
    device const uchar2* w2 = (device const uchar2*)wr;   // 2 bytes = 4 nibbles
    device const float4* x4 = (device const float4*)xr;
    for (int c = lid; c < I4; c += tgsz) {
      uchar2 bb = w2[c];
      float4 wv = float4(float(int(bb.x & 0xF)-8), float(int(bb.x >> 4)-8),
                         float(int(bb.y & 0xF)-8), float(int(bb.y >> 4)-8));
      acc += dot(wv, x4[c]);
    }
    for (int i = I4*4 + lid; i < I; i += tgsz) {
      uchar b = wr[i>>1]; int v = (i&1) ? (b>>4) : (b&0xF); acc += float(v-8) * xr[i];
    }
  } else if (fmt == 3) {                            // int2 packed, rb=(I+3)/4
    int rb = (I+3)/4;
    device const uchar* wr = w + (long)o * rb;
    for (int i = lid; i < I; i += tgsz) {
      uchar b = wr[i>>2]; int v = (b >> (2*(i&3))) & 0x3; acc += float(v-2) * xr[i];
    }
  } else {                                          // f32
    device const float* wr = (device const float*)(w) + (long)o * I;
    device const float4* w4 = (device const float4*)wr;
    device const float4* x4 = (device const float4*)xr;
    for (int c = lid; c < I4; c += tgsz) acc += dot(w4[c], x4[c]);
    for (int i = I4*4 + lid; i < I; i += tgsz) acc += wr[i] * xr[i];
  }

  acc = simd_sum(acc);
  threadgroup float sh[32];
  if (slane == 0) sh[sgid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    uint nsg = (tgsz + 31) / 32; float t = 0.0f;
    for (uint k = 0; k < nsg; k++) t += sh[k];
    y[(long)si * O + o] = t * scale[o];
  }
}

// Batched bindless expert GEMV: each row gr belongs to expert erow[gr], whose weight and
// scale live at gpuAddresses waddr[e]/saddr[e] (zero-copy in the RAM slab). fmt 1=i8, 2=i4.
kernel void moe_gemv(device const ulong* waddr [[buffer(0)]], device const ulong* saddr [[buffer(1)]],
                     device const int* erow [[buffer(2)]], device const float* xin [[buffer(3)]],
                     device float* yout [[buffer(4)]],
                     constant int& O [[buffer(5)]], constant int& K [[buffer(6)]],
                     constant int& Kin [[buffer(7)]], constant int& fmt [[buffer(8)]],
                     uint tg [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                     uint tgsz [[threads_per_threadgroup]], uint slane [[thread_index_in_simdgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]]) {
  int gr = tg / O, o = tg % O; int e = erow[gr]; int K4 = (K & 3) ? 0 : (K/4);
  device const float* xr = xin + (long)gr * Kin;
  device const float* sc = (device const float*)(saddr[e]);
  device const float4* x4 = (device const float4*)xr;
  float acc = 0.0f;
  if (fmt == 2) { int rb=(K+1)/2; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const uchar2* w2=(device const uchar2*)w;
    for(int c=lid;c<K4;c+=tgsz){ uchar2 bb=w2[c];
      float4 wv=float4(float(int(bb.x&0xF)-8),float(int(bb.x>>4)-8),float(int(bb.y&0xF)-8),float(int(bb.y>>4)-8));
      acc+=dot(wv,x4[c]); }
    for(int i=K4*4+lid;i<K;i+=tgsz){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]; }
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
    device const char4* w4=(device const char4*)w;
    for(int c=lid;c<K4;c+=tgsz) acc+=dot(float4(w4[c]),x4[c]);
    for(int i=K4*4+lid;i<K;i+=tgsz) acc+=float(w[i])*xr[i];
  }
  acc=simd_sum(acc); threadgroup float sh[32];
  if(slane==0) sh[sgid]=acc; threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lid==0){ uint n=(tgsz+31)/32; float t=0; for(uint k=0;k<n;k++) t+=sh[k]; yout[(long)gr*O+o]=t*sc[o]; }
}
kernel void moe_silu(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) { float v=g[i]; g[i]=(v/(1.0f+exp(-v)))*u[i]; }

// ===== Fused decode attention (GLM-5.2 dims, S=1) =====
constant int A_HID=6144, A_H=64, A_QLORA=2048, A_KVL=512, A_NOPE=192, A_ROPE=64, A_VH=256;
constant int A_QH=256 /*nope+rope*/, A_ROWSH=448 /*nope+vh*/;
// in-place RMSNorm over n elems: out[i]=x[i]*rsqrt(mean(x^2)+eps)*w[i]. One threadgroup.
kernel void a_rmsnorm(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      constant int& n [[buffer(2)]], constant float& eps [[buffer(3)]],
                      uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]) {
  threadgroup float red[256]; float s=0; for(int i=lid;i<n;i+=tgsz) s+=x[i]*x[i];
  red[lid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float r=rsqrt(red[0]/n+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=lid;i<n;i+=tgsz) x[i]=x[i]*r*w[i];
}
// interleaved partial RoPE on a qk_rope vector at [base]; half=ROPE/2. grid = count*half.
kernel void a_rope(device float* v [[buffer(0)]], constant int& base [[buffer(1)]],
                   constant int& stride [[buffer(2)]], constant int& pos [[buffer(3)]],
                   constant float& theta [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  int hlf=A_ROPE/2; int u=gid/hlf, j=gid%hlf; device float* vv=v+(long)base+(long)u*stride;
  float inv=pow(theta, -2.0f*j/A_ROPE); float ang=pos*inv, cs=cos(ang), sn=sin(ang);
  float a=vv[2*j], b=vv[2*j+1]; vv[j]=a*cs-b*sn; vv[hlf+j]=b*cs+a*sn;
}
// copy src[off..off+n) -> dst[0..n)
kernel void a_copy(device const float* src [[buffer(0)]], constant int& off [[buffer(1)]],
                   device float* dst [[buffer(2)]], uint i [[thread_position_in_grid]]) { dst[i]=src[off+i]; }
// ---- absorption core (validated) ----
inline float a_deqrow(device const uchar* base, int row, int i, device const float* sc){
  device const uchar* w=base+(long)row*((A_KVL+1)/2); uchar b=w[i>>1]; int val=(i&1)?(b>>4):(b&0xF); return float(val-8)*sc[row]; }
kernel void a_qabs(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                   device const float* q [[buffer(2)]], device float* qabs [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  int h=gid/A_KVL, i=gid%A_KVL; int rbase=h*A_ROWSH; device const float* qp=q+(long)h*A_QH;
  float a=0; for(int d=0;d<A_NOPE;d++) a+=qp[d]*a_deqrow(kvb,rbase+d,i,sc); qabs[(long)h*A_KVL+i]=a;
}
kernel void a_score(device const float* qabs [[buffer(0)]], device const float* Lc [[buffer(1)]],
                    device const float* Rc [[buffer(2)]], device const float* q [[buffer(3)]],
                    device float* sc [[buffer(4)]], constant int& T [[buffer(5)]], constant float& ascale [[buffer(6)]],
                    uint gid [[thread_position_in_grid]]) {
  int h=gid/T, t=gid%T; device const float* qa=qabs+(long)h*A_KVL; device const float* Lt=Lc+(long)t*A_KVL;
  device const float* qr=q+(long)h*A_QH+A_NOPE; device const float* Rt=Rc+(long)t*A_ROPE;
  float a=0; for(int i=0;i<A_KVL;i++) a+=qa[i]*Lt[i]; for(int d=0;d<A_ROPE;d++) a+=qr[d]*Rt[d]; sc[(long)h*T+t]=a*ascale;
}
kernel void a_smax(device float* sc [[buffer(0)]], constant int& T [[buffer(1)]],
                   uint h [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]) {
  device float* s=sc+(long)h*T; threadgroup float red[256];
  float m=-1e30f; for(int t=lid;t<T;t+=tgsz) m=max(m,s[t]); red[lid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]=max(red[lid],red[lid+k]); threadgroup_barrier(mem_flags::mem_threadgroup);}
  float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum=0; for(int t=lid;t<T;t+=tgsz){ float e=exp(s[t]-mx); s[t]=e; sum+=e; } red[lid]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup);}
  float tot=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup); for(int t=lid;t<T;t+=tgsz) s[t]/=tot;
}
kernel void a_clat(device const float* sc [[buffer(0)]], device const float* Lc [[buffer(1)]],
                   device float* clat [[buffer(2)]], constant int& T [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  int h=gid/A_KVL, i=gid%A_KVL; device const float* s=sc+(long)h*T; float a=0;
  for(int t=0;t<T;t++) a+=s[t]*Lc[(long)t*A_KVL+i]; clat[(long)h*A_KVL+i]=a;
}
kernel void a_ctx(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                  device const float* clat [[buffer(2)]], device float* ctx [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  int h=gid/A_VH, j=gid%A_VH; int row=h*A_ROWSH+A_NOPE+j; device const float* cl=clat+(long)h*A_KVL;
  float a=0; for(int i=0;i<A_KVL;i++) a+=cl[i]*a_deqrow(kvb,row,i,sc); ctx[(long)h*A_VH+j]=a;
}
)METAL";

struct ColiMetalTensor {
  id<MTLBuffer> w;      // weights (wrapped, zero-copy when page-aligned)
  id<MTLBuffer> s;      // scales
  int fmt, I, O; size_t wbytes;
};

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_gemv, g_moe_gemv, g_moe_silu;
static id<MTLComputePipelineState> g_a_rms, g_a_rope, g_a_copy, g_a_qabs, g_a_score, g_a_smax, g_a_clat, g_a_ctx;
static size_t g_tensor_count, g_tensor_bytes;
static uint64_t g_moe_ok, g_moe_fb, g_moe_experts;   // GPU blocks / CPU-fallback blocks / experts on GPU
static double g_t_setup, g_t_gpu, g_t_scatter, g_t_kernel;       // per-block time breakdown (seconds)
static const int TG = 128;
#include <mach/mach_time.h>
static double mnow(){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }

extern "C" void coli_metal_moe_counts(uint64_t *ok, uint64_t *fb, uint64_t *ex) {
  if(ok)*ok=g_moe_ok; if(fb)*fb=g_moe_fb; if(ex)*ex=g_moe_experts;
}
extern "C" void coli_metal_moe_times(double *setup, double *gpu, double *scatter) {
  if(setup)*setup=g_t_setup; if(gpu)*gpu=g_t_gpu; if(scatter)*scatter=g_t_scatter;
}
extern "C" double coli_metal_moe_kernel_time(void){ return g_t_kernel; }
static uint64_t g_attn_ok; static double g_attn_wall, g_attn_kernel;
extern "C" void coli_metal_attn_counts(uint64_t *ok, double *wall, double *kernel){
  if(ok)*ok=g_attn_ok; if(wall)*wall=g_attn_wall; if(kernel)*kernel=g_attn_kernel; }

// Registry of page-aligned host slabs wrapped zero-copy for the batched MoE path.
struct Slab { void *base; size_t len; id<MTLBuffer> buf; };
static std::vector<Slab> g_slabs;
static std::mutex g_slab_mtx;   // expert_load registers slabs from parallel OpenMP threads
// Persistent scratch buffers (grow-only) for the MoE pipeline.
static id<MTLBuffer> g_gg, g_uu, g_hh, g_xg; static size_t g_gg_cap, g_uu_cap, g_hh_cap, g_xg_cap;
static id<MTLBuffer> ensure(id<MTLBuffer> b, size_t *cap, size_t need) {
  if (b && *cap >= need) return b;
  *cap = need; return [g_dev newBufferWithLength:need options:MTLResourceStorageModeShared];
}

static size_t fmt_bytes(int fmt, int I, int O) {
  if (fmt == 1) return (size_t)O * I;
  if (fmt == 2) return (size_t)O * ((I+1)/2);
  if (fmt == 3) return (size_t)O * ((I+3)/4);
  return (size_t)O * I * sizeof(float);
}

// Wrap host memory zero-copy if page-aligned, else copy into a shared buffer.
static id<MTLBuffer> wrap(const void *p, size_t n) {
  size_t pg = 16384; // Apple Silicon page
  if (((uintptr_t)p % pg) == 0 && (n % pg) == 0)
    return [g_dev newBufferWithBytesNoCopy:(void*)p length:n options:MTLResourceStorageModeShared deallocator:nil];
  return [g_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
}

extern "C" int coli_metal_init(void) {
  if (g_dev) return 1;
  @autoreleasepool {
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) return 0;
    g_queue = [g_dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:SHADER]
                                             options:nil error:&err];
    if (!lib) { fprintf(stderr, "[metal] shader compile failed: %s\n",
                        err ? [[err localizedDescription] UTF8String] : "?"); g_dev = nil; return 0; }
    g_gemv     = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_gemv"]   error:&err];
    g_moe_gemv = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_gemv"] error:&err];
    g_moe_silu = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_silu"] error:&err];
    auto P=[&](const char*n){ return [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:&err]; };
    g_a_rms=P("a_rmsnorm"); g_a_rope=P("a_rope"); g_a_copy=P("a_copy");
    g_a_qabs=P("a_qabs"); g_a_score=P("a_score"); g_a_smax=P("a_smax"); g_a_clat=P("a_clat"); g_a_ctx=P("a_ctx");
    if (!g_gemv || !g_moe_gemv || !g_moe_silu || !g_a_rms || !g_a_rope || !g_a_copy ||
        !g_a_qabs || !g_a_score || !g_a_smax || !g_a_clat || !g_a_ctx) {
      fprintf(stderr, "[metal] pipeline failed\n"); g_dev = nil; return 0; }
  }
  return 1;
}

extern "C" void coli_metal_register(void *base, size_t len) {
  if (!g_dev || !base) return;
  id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:base length:len
                              options:MTLResourceStorageModeShared deallocator:nil];
  if (!b) return;
  std::lock_guard<std::mutex> lk(g_slab_mtx);   // called from parallel expert_load threads
  for (auto &s : g_slabs) if (s.base == base) { s.len = len; s.buf = b; return; }
  g_slabs.push_back({base, len, b});
}
extern "C" void coli_metal_unregister(void *base) {
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  for (size_t i=0;i<g_slabs.size();i++) if (g_slabs[i].base==base) { g_slabs[i].buf=nil; g_slabs.erase(g_slabs.begin()+i); return; }
}
// Resolve a host pointer inside a registered slab to (buffer, gpuAddress). Returns nil if unknown.
static id<MTLBuffer> resolve(const void *p, uint64_t *addr) {
  std::lock_guard<std::mutex> lk(g_slab_mtx);
  uintptr_t u=(uintptr_t)p;
  for (auto &s : g_slabs) { uintptr_t b=(uintptr_t)s.base;
    if (u>=b && u<b+s.len) { *addr = (uint64_t)[s.buf gpuAddress] + (u-b); return s.buf; } }
  return nil;
}

extern "C" void coli_metal_shutdown(void) { g_gemv=nil; g_queue=nil; g_dev=nil; g_tensor_count=g_tensor_bytes=0; }
extern "C" int  coli_metal_available(void) { return g_dev != nil; }
extern "C" void coli_metal_stats(size_t *c, size_t *b) { if(c)*c=g_tensor_count; if(b)*b=g_tensor_bytes; }
extern "C" int  coli_metal_mem_info(size_t *used, size_t *total) {
  if (!g_dev) return 0;
  if (used) *used = (size_t)[g_dev currentAllocatedSize];
  if (total) *total = (size_t)[g_dev recommendedMaxWorkingSetSize];
  return 1;
}

extern "C" int coli_metal_matmul(ColiMetalTensor **tp, float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O) {
  if (!g_dev || fmt < 0 || fmt > 3) return 0;
  @autoreleasepool {
    ColiMetalTensor *t = *tp;
    if (!t) {
      t = new ColiMetalTensor();
      t->fmt = fmt; t->I = I; t->O = O; t->wbytes = fmt_bytes(fmt, I, O);
      t->w = wrap(weights, t->wbytes);
      t->s = wrap(scales, (size_t)O * sizeof(float));
      *tp = t;
      g_tensor_count++; g_tensor_bytes += t->wbytes;
    }
    id<MTLBuffer> bx = [g_dev newBufferWithBytes:x length:(size_t)S*I*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> by = [g_dev newBufferWithLength:(size_t)S*O*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_gemv];
    [e setBuffer:t->w offset:0 atIndex:0]; [e setBuffer:t->s offset:0 atIndex:1];
    [e setBuffer:bx offset:0 atIndex:2];   [e setBuffer:by offset:0 atIndex:3];
    [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5];
    [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake((size_t)O*S,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(y, [by contents], (size_t)S*O*sizeof(float));
  }
  return 1;
}

// ---- fused decode attention scratch (GLM-5.2 dims) ----
enum { AH=6144, AHEADS=64, AQLORA=2048, AKVL=512, AROPE=64, AVH=256, AQH=256, ANOPE=192, AROWSH=448, AHQH=AHEADS*AQH, AHVH=AHEADS*AVH };
static id<MTLBuffer> ax_,aqr_,aqf_,acomp_,aqabs_,ascore_,aclat_,actx_,aout_,aqaln_,akvaln_; static size_t ascore_cap;
static void attn_scratch_init(){
  if(ax_) return;
  auto L=[&](size_t n){ return [g_dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
  ax_=L(AH*4); aqr_=L(AQLORA*4); aqf_=L(AHQH*4); acomp_=L((AKVL+AROPE)*4);
  aqabs_=L((size_t)AHEADS*AKVL*4); aclat_=L((size_t)AHEADS*AKVL*4); actx_=L(AHVH*4); aout_=L(AH*4);
  aqaln_=L(AQLORA*4); akvaln_=L(AKVL*4);
}
// Cache of uploaded resident weights/scales (fixed dense tensors reused every token).
// Attention runs serially (not under OpenMP), so no lock needed.
static std::vector<std::pair<const void*,id<MTLBuffer>>> g_wcache;
static id<MTLBuffer> wcache(const void* p, size_t bytes){
  for(auto&kv:g_wcache) if(kv.first==p) return kv.second;
  id<MTLBuffer> b=[g_dev newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
  g_wcache.push_back({p,b}); return b;
}
// y[O] = quantized-weight(w) applied to xin[I], S=1. Uploads+caches w and its scales.
static void bind_gemv(id<MTLComputeCommandEncoder> e, const void* w, const float* s, int fmt, int I, int O,
                      id<MTLBuffer> xin, id<MTLBuffer> yout){
  id<MTLBuffer> wb=wcache(w,fmt_bytes(fmt,I,O)); id<MTLBuffer> sb=wcache(s,(size_t)O*4);
  int S=1;
  [e setComputePipelineState:g_gemv];
  [e setBuffer:wb offset:0 atIndex:0]; [e setBuffer:sb offset:0 atIndex:1];
  [e setBuffer:xin offset:0 atIndex:2]; [e setBuffer:yout offset:0 atIndex:3];
  [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5]; [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
  [e dispatchThreadgroups:MTLSizeMake((size_t)O,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)];
}

extern "C" int coli_metal_attn_decode(const float* x,
    const void* qa_w,const float* qa_s,int qa_fmt,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,
    const void* kva_w,const float* kva_s,int kva_fmt,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,
    const void* o_w,const float* o_s,int o_fmt,
    float* Lc,float* Rc,int pos,int st0,float eps,float theta,float ascale,float* out){
  if(!g_dev) return 0;
  if(st0!=0) return 0;                       // partial-KV (MTP) not handled here -> CPU
  int T=pos+1;
  @autoreleasepool {
    attn_scratch_init();
    // Lc/Rc caches are registered (page-aligned) -> zero-copy resolve. kv_b uploaded+cached.
    uint64_t la=0,ra=0; id<MTLBuffer> Lb=resolve(Lc,&la), Rb=resolve(Rc,&ra);
    if(!Lb||!Rb) return 0;
    size_t loff=la-(uint64_t)[Lb gpuAddress], roff=ra-(uint64_t)[Rb gpuAddress];
    id<MTLBuffer> kvbW=wcache(kvb_w,fmt_bytes(kvb_fmt,AKVL,AHEADS*AROWSH)); size_t kvbwoff=0;
    id<MTLBuffer> kvbS=wcache(kvb_s,(size_t)AHEADS*AROWSH*4); size_t kvbsoff=0;
    ascore_=ensure(ascore_,&ascore_cap,(size_t)AHEADS*T*4);
    memcpy([ax_ contents],x,AH*4); memcpy([aqaln_ contents],qa_ln,AQLORA*4); memcpy([akvaln_ contents],kva_ln,AKVL*4);

    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    // keep resolved resources resident (bindless-addressed cache/weights)
    [e useResource:Lb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rb usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    auto BAR=[&]{ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    // q path: q_a -> rmsnorm -> q_b -> rope
    bind_gemv(e,qa_w,qa_s,qa_fmt,AH,AQLORA,ax_,aqr_); BAR();
    { int n=AQLORA; [e setComputePipelineState:g_a_rms]; [e setBuffer:aqr_ offset:0 atIndex:0]; [e setBuffer:aqaln_ offset:0 atIndex:1];
      [e setBytes:&n length:4 atIndex:2]; [e setBytes:&eps length:4 atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; } BAR();
    bind_gemv(e,qb_w,qb_s,qb_fmt,AQLORA,AHQH,aqr_,aqf_); BAR();
    { int base=ANOPE, stride=AQH; [e setComputePipelineState:g_a_rope]; [e setBuffer:aqf_ offset:0 atIndex:0];
      [e setBytes:&base length:4 atIndex:1]; [e setBytes:&stride length:4 atIndex:2]; [e setBytes:&pos length:4 atIndex:3]; [e setBytes:&theta length:4 atIndex:4];
      [e dispatchThreads:MTLSizeMake((size_t)AHEADS*(AROPE/2),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; } BAR();
    // kv path: kv_a -> split -> latent rmsnorm@pos + krot rope@pos
    bind_gemv(e,kva_w,kva_s,kva_fmt,AH,AKVL+AROPE,ax_,acomp_); BAR();
    { int off=0; [e setComputePipelineState:g_a_copy]; [e setBuffer:acomp_ offset:0 atIndex:0]; [e setBytes:&off length:4 atIndex:1];
      [e setBuffer:Lb offset:loff+(size_t)pos*AKVL*4 atIndex:2]; [e dispatchThreads:MTLSizeMake(AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; }
    { int off=AKVL; [e setComputePipelineState:g_a_copy]; [e setBuffer:acomp_ offset:0 atIndex:0]; [e setBytes:&off length:4 atIndex:1];
      [e setBuffer:Rb offset:roff+(size_t)pos*AROPE*4 atIndex:2]; [e dispatchThreads:MTLSizeMake(AROPE,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; } BAR();
    { int n=AKVL; [e setComputePipelineState:g_a_rms]; [e setBuffer:Lb offset:loff+(size_t)pos*AKVL*4 atIndex:0]; [e setBuffer:akvaln_ offset:0 atIndex:1];
      [e setBytes:&n length:4 atIndex:2]; [e setBytes:&eps length:4 atIndex:3]; [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; }
    { int base=0, stride=AROPE; [e setComputePipelineState:g_a_rope]; [e setBuffer:Rb offset:roff+(size_t)pos*AROPE*4 atIndex:0];
      [e setBytes:&base length:4 atIndex:1]; [e setBytes:&stride length:4 atIndex:2]; [e setBytes:&pos length:4 atIndex:3]; [e setBytes:&theta length:4 atIndex:4];
      [e dispatchThreads:MTLSizeMake(AROPE/2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; } BAR();
    // absorption core
    [e setComputePipelineState:g_a_qabs]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aqf_ offset:0 atIndex:2]; [e setBuffer:aqabs_ offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_score]; [e setBuffer:aqabs_ offset:0 atIndex:0]; [e setBuffer:Lb offset:loff atIndex:1]; [e setBuffer:Rb offset:roff atIndex:2]; [e setBuffer:aqf_ offset:0 atIndex:3]; [e setBuffer:ascore_ offset:0 atIndex:4];
    [e setBytes:&T length:4 atIndex:5]; [e setBytes:&ascale length:4 atIndex:6];
    [e dispatchThreads:MTLSizeMake((size_t)AHEADS*T,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_smax]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBytes:&T length:4 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(AHEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_clat]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBuffer:Lb offset:loff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBytes:&T length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_ctx]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBuffer:actx_ offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)AHEADS*AVH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    // o_proj
    bind_gemv(e,o_w,o_s,o_fmt,AHVH,AH,actx_,aout_);
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] attn cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    memcpy(out,[aout_ contents],AH*4);
  }
  return 1;
}

extern "C" void coli_metal_tensor_free(ColiMetalTensor *t) {
  if (!t) return;
  g_tensor_count--; g_tensor_bytes -= t->wbytes;
  t->w = nil; t->s = nil; delete t;
}
extern "C" size_t coli_metal_tensor_bytes(const ColiMetalTensor *t) { return t ? t->wbytes : 0; }

// Batched routed-expert SwiGLU for one block in ONE command buffer. Returns 0 (CPU fallback)
// if Metal is off or any expert pointer is not in a registered slab.
extern "C" int coli_metal_moe_block(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw, float *out, int S) {
  if (!g_dev || (fmt != 1 && fmt != 2)) return 0;
  (void)S;
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0) return 1;
    double ts_start = mnow();
    // address + erow tables
    std::vector<uint64_t> ag(nb),au(nb),ad(nb),sgv(nb),suv(nb),sdv(nb);
    std::vector<id<MTLBuffer>> use; use.reserve(nb*2);
    auto add_use=[&](id<MTLBuffer> b){ for(auto&x:use) if(x==b) return; use.push_back(b); };
    for (int e=0;e<nb;e++) {
      id<MTLBuffer> b;
      if(!(b=resolve(g[e],&ag[e]))) {g_moe_fb++; return 0;} add_use(b);
      if(!(b=resolve(u[e],&au[e]))) {g_moe_fb++; return 0;} add_use(b);
      if(!(b=resolve(d[e],&ad[e]))) {g_moe_fb++; return 0;} add_use(b);
      if(!(b=resolve(gs[e],&sgv[e]))) {g_moe_fb++; return 0;} add_use(b);
      if(!(b=resolve(us[e],&suv[e]))) {g_moe_fb++; return 0;} add_use(b);
      if(!(b=resolve(ds[e],&sdv[e]))) {g_moe_fb++; return 0;} add_use(b);
    }
    static int dbg=-1; if(dbg<0) dbg = getenv("COLI_METAL_DEBUG")?atoi(getenv("COLI_METAL_DEBUG")):0;
    if(dbg){ dbg=0; fprintf(stderr,"[metal dbg] moe_block nb=%d R=%d D=%d Iinter=%d fmt=%d | e0: wg=%p ag=0x%llx sg=0x%llx slabs=%zu\n",
             nb,R,D,Iinter,fmt,g[0],(unsigned long long)ag[0],(unsigned long long)sgv[0],g_slabs.size()); }
    std::vector<int> erow(R); for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++) erow[xoff[e]+r]=e;
    auto shb=[&](const void*p,size_t n){ return [g_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared]; };
    id<MTLBuffer> bag=shb(ag.data(),nb*8), bau=shb(au.data(),nb*8), bad=shb(ad.data(),nb*8);
    id<MTLBuffer> bsg=shb(sgv.data(),nb*8), bsu=shb(suv.data(),nb*8), bsd=shb(sdv.data(),nb*8);
    id<MTLBuffer> berow=shb(erow.data(),R*4);
    g_xg = ensure(g_xg,&g_xg_cap,(size_t)R*D*4);       memcpy([g_xg contents], xg, (size_t)R*D*4);
    g_gg = ensure(g_gg,&g_gg_cap,(size_t)R*Iinter*4);
    g_uu = ensure(g_uu,&g_uu_cap,(size_t)R*Iinter*4);
    g_hh = ensure(g_hh,&g_hh_cap,(size_t)R*D*4);

    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];
    auto gemv=[&](id<MTLBuffer> wa,id<MTLBuffer> sa,id<MTLBuffer> xin,id<MTLBuffer> y,int O,int K,int Kin){
      [e setComputePipelineState:g_moe_gemv];
      [e setBuffer:wa offset:0 atIndex:0];[e setBuffer:sa offset:0 atIndex:1];[e setBuffer:berow offset:0 atIndex:2];
      [e setBuffer:xin offset:0 atIndex:3];[e setBuffer:y offset:0 atIndex:4];
      [e setBytes:&O length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&Kin length:4 atIndex:7];[e setBytes:&fmt length:4 atIndex:8];
      [e dispatchThreadgroups:MTLSizeMake((size_t)R*O,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)]; };
    gemv(bag,bsg,g_xg,g_gg,Iinter,D,D);                       // gate
    gemv(bau,bsu,g_xg,g_uu,Iinter,D,D);                       // up
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [e setComputePipelineState:g_moe_silu];
    [e setBuffer:g_gg offset:0 atIndex:0];[e setBuffer:g_uu offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)R*Iinter,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    gemv(bad,bsd,g_gg,g_hh,D,Iinter,Iinter);                  // down
    double ts_commit = mnow(); g_t_setup += ts_commit - ts_start;
    [e endEncoding];[cb commit];[cb waitUntilCompleted];
    double ts_gpu = mnow(); g_t_gpu += ts_gpu - ts_commit;
    g_t_kernel += [cb GPUEndTime] - [cb GPUStartTime];   // actual on-GPU execution window
    if (cb.status == MTLCommandBufferStatusError) {           // GPU fault -> fall back to CPU for this block
      fprintf(stderr, "[metal] moe_block cmdbuf error (nb=%d R=%d): %s\n", nb, R,
              cb.error ? [[cb.error localizedDescription] UTF8String] : "?");
      g_moe_fb++; return 0;
    }

    // scatter-add: out[rows[gr]] += rw[gr] * hh[gr]
    const float *hh=(const float*)[g_hh contents];
    for(int gr=0;gr<R;gr++){ float *os=out+(size_t)rows[gr]*D, w=rw[gr]; const float *hr=hh+(size_t)gr*D;
      for(int dd=0;dd<D;dd++) os[dd]+=w*hr[dd]; }
    g_t_scatter += mnow() - ts_gpu;
    g_moe_ok++; g_moe_experts += nb;
  }
  return 1;
}
