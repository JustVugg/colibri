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
// per-row in-place RMSNorm: row = threadgroup index, x[row*n + i]. grid = nrows threadgroups.
kernel void a_rmsnorm(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      constant int& n [[buffer(2)]], constant float& eps [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]) {
  device float* xr=x+(long)row*n; threadgroup float red[256];
  float s=0; for(int i=lid;i<n;i+=tgsz) s+=xr[i]*xr[i];
  red[lid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float r=rsqrt(red[0]/n+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=lid;i<n;i+=tgsz) xr[i]=xr[i]*r*w[i];
}
// interleaved partial RoPE. vv = v + base + s*rowstride + h*headstride, pos = PB+s. grid = S*nheads*(ROPE/2).
kernel void a_rope(device float* v [[buffer(0)]], constant int& base [[buffer(1)]],
                   constant int& rowstride [[buffer(2)]], constant int& headstride [[buffer(3)]],
                   constant int& nheads [[buffer(4)]], constant int& PB [[buffer(5)]],
                   constant float& theta [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int hlf=A_ROPE/2; int idx=gid/hlf, j=gid%hlf; int s=idx/nheads, h=idx%nheads; int pos=PB+s;
  device float* vv=v+(long)base+(long)s*rowstride+(long)h*headstride;
  float inv=pow(theta, -2.0f*j/A_ROPE); float ang=pos*inv, cs=cos(ang), sn=sin(ang);
  float a=vv[2*j], b=vv[2*j+1]; vv[j]=a*cs-b*sn; vv[hlf+j]=b*cs+a*sn;
}
// per-row copy: dst[s*dststride + i] = src[s*srcstride + off + i]. grid = S*n.
kernel void a_copy(device const float* src [[buffer(0)]], constant int& off [[buffer(1)]], constant int& srcstride [[buffer(2)]],
                   device float* dst [[buffer(3)]], constant int& dststride [[buffer(4)]], constant int& n [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) { int s=gid/n, i=gid%n; dst[(long)s*dststride+i]=src[(long)s*srcstride+off+i]; }
// ---- absorption core (S query rows, per-row causal). q:[S,H*QH]; qabs/clat:[S*H,KVL];
//      sc:[S*H,T]; ctx:[S*H,VH]. Query row s (abs pos PB+s) attends keys [0, PB+s]. ----
constant int A_QHH=A_H*A_QH;
inline float a_deqrow(device const uchar* base, int row, int i, device const float* sc){
  device const uchar* w=base+(long)row*((A_KVL+1)/2); uchar b=w[i>>1]; int val=(i&1)?(b>>4):(b&0xF); return float(val-8)*sc[row]; }
kernel void a_qabs(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                   device const float* q [[buffer(2)]], device float* qabs [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  int s=gid/(A_H*A_KVL), r=gid%(A_H*A_KVL), h=r/A_KVL, i=r%A_KVL; int rbase=h*A_ROWSH;
  device const float* qp=q+(long)s*A_QHH+(long)h*A_QH;
  float a=0; for(int d=0;d<A_NOPE;d++) a+=qp[d]*a_deqrow(kvb,rbase+d,i,sc); qabs[(long)(s*A_H+h)*A_KVL+i]=a;
}
kernel void a_score(device const float* qabs [[buffer(0)]], device const float* Lc [[buffer(1)]],
                    device const float* Rc [[buffer(2)]], device const float* q [[buffer(3)]],
                    device float* sc [[buffer(4)]], constant int& T [[buffer(5)]], constant float& ascale [[buffer(6)]],
                    constant int& PB [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  int s=gid/(A_H*T), r=gid%(A_H*T), h=r/T, t=r%T; long o=(long)(s*A_H+h)*T+t;
  if(t > PB+s){ sc[o]=-1e30f; return; }                                 // causal mask
  device const float* qa=qabs+(long)(s*A_H+h)*A_KVL; device const float* Lt=Lc+(long)t*A_KVL;
  device const float* qr=q+(long)s*A_QHH+(long)h*A_QH+A_NOPE; device const float* Rt=Rc+(long)t*A_ROPE;
  float a=0; for(int i=0;i<A_KVL;i++) a+=qa[i]*Lt[i]; for(int d=0;d<A_ROPE;d++) a+=qr[d]*Rt[d]; sc[o]=a*ascale;
}
kernel void a_smax(device float* sc [[buffer(0)]], constant int& T [[buffer(1)]],
                   uint sh [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]) {
  device float* s=sc+(long)sh*T; threadgroup float red[256];
  float m=-1e30f; for(int t=lid;t<T;t+=tgsz) m=max(m,s[t]); red[lid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]=max(red[lid],red[lid+k]); threadgroup_barrier(mem_flags::mem_threadgroup);}
  float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum=0; for(int t=lid;t<T;t+=tgsz){ float e=exp(s[t]-mx); s[t]=e; sum+=e; } red[lid]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup);}
  float tot=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup); for(int t=lid;t<T;t+=tgsz) s[t]/=tot;
}
kernel void a_clat(device const float* sc [[buffer(0)]], device const float* Lc [[buffer(1)]],
                   device float* clat [[buffer(2)]], constant int& T [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  int sh=gid/A_KVL, i=gid%A_KVL; device const float* s=sc+(long)sh*T; float a=0;
  for(int t=0;t<T;t++) a+=s[t]*Lc[(long)t*A_KVL+i]; clat[(long)sh*A_KVL+i]=a;
}
kernel void a_ctx(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                  device const float* clat [[buffer(2)]], device float* ctx [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  int sh=gid/A_VH, j=gid%A_VH, h=sh%A_H; int row=h*A_ROWSH+A_NOPE+j; device const float* cl=clat+(long)sh*A_KVL;
  float a=0; for(int i=0;i<A_KVL;i++) a+=cl[i]*a_deqrow(kvb,row,i,sc); ctx[(long)sh*A_VH+j]=a;
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
static uint64_t g_attn_ok; static double g_attn_wall, g_attn_kernel, g_attn_sched, g_attn_ksched;
extern "C" void coli_metal_attn_counts(uint64_t *ok, double *wall, double *kernel){
  if(ok)*ok=g_attn_ok; if(wall)*wall=g_attn_wall; if(kernel)*kernel=g_attn_kernel; }
extern "C" void coli_metal_attn_lat(double *ksched, double *gsched){
  if(ksched)*ksched=g_attn_ksched; if(gsched)*gsched=g_attn_sched; }

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

// Keep-alive spinner (COLI_METAL_SPIN=1): keeps trivial GPU work in flight so the GPU
// doesn't ramp its clock down between the engine's short per-layer bursts. Experiment to
// quantify how much of the observed submit latency is clock ramp-down.
#include <thread>
#include <atomic>
static std::atomic<bool> g_spin_run{false};
static std::thread g_spin_thr;
extern "C" void coli_metal_spin_start(void) {
  if (!g_dev || g_spin_run.exchange(true)) return;
  g_spin_thr = std::thread([]{
    id<MTLCommandQueue> q = [g_dev newCommandQueue];       // own queue: never blocks real work
    id<MTLBuffer> b = [g_dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    while (g_spin_run.load()) {
      @autoreleasepool {
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:g_moe_silu];
        [e setBuffer:b offset:0 atIndex:0]; [e setBuffer:b offset:0 atIndex:1];
        [e dispatchThreads:MTLSizeMake(1024,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
    }
  });
  g_spin_thr.detach();               // never joinable at exit (joinable global -> std::terminate)
}
extern "C" void coli_metal_spin_stop(void) { g_spin_run.store(false); }

extern "C" void coli_metal_shutdown(void) { coli_metal_spin_stop(); g_gemv=nil; g_queue=nil; g_dev=nil; g_tensor_count=g_tensor_bytes=0; }
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
enum { AH=6144, AHEADS=64, AQLORA=2048, AKVL=512, AROPE=64, AVH=256, AQH=256, ANOPE=192, AROWSH=448, AHQH=AHEADS*AQH, AHVH=AHEADS*AVH, AMAXS=4 };
static id<MTLBuffer> ax_,aqr_,aqf_,acomp_,aqabs_,ascore_,aclat_,actx_,aout_,aqaln_,akvaln_; static size_t ascore_cap;
static void attn_scratch_init(){
  if(ax_) return;
  auto L=[&](size_t n){ return [g_dev newBufferWithLength:n*AMAXS options:MTLResourceStorageModeShared]; };
  ax_=L(AH*4); aqr_=L(AQLORA*4); aqf_=L(AHQH*4); acomp_=L((AKVL+AROPE)*4);
  aqabs_=L((size_t)AHEADS*AKVL*4); aclat_=L((size_t)AHEADS*AKVL*4); actx_=L(AHVH*4); aout_=L(AH*4);
  aqaln_=L(AQLORA*4/AMAXS); akvaln_=L(AKVL*4/AMAXS);   // norm weights are per-tensor, not per-row
}
// y[S,O] = quantized-weight(w) applied to xin[S,I]. Weights are registered (page-aligned,
// zero-copy) at model load; resolve to (buffer,offset). Returns false to fall back to CPU.
static bool bind_gemv(id<MTLComputeCommandEncoder> e, const void* w, const float* s, int fmt, int I, int O,
                      id<MTLBuffer> xin, id<MTLBuffer> yout, int S){
  uint64_t wa=0,sa=0; id<MTLBuffer> wb=resolve(w,&wa); id<MTLBuffer> sb=resolve(s,&sa);
  if(!wb||!sb) return false;
  size_t woff=wa-(uint64_t)[wb gpuAddress], soff=sa-(uint64_t)[sb gpuAddress];
  [e useResource:wb usage:MTLResourceUsageRead]; [e useResource:sb usage:MTLResourceUsageRead];
  [e setComputePipelineState:g_gemv];
  [e setBuffer:wb offset:woff atIndex:0]; [e setBuffer:sb offset:soff atIndex:1];
  [e setBuffer:xin offset:0 atIndex:2]; [e setBuffer:yout offset:0 atIndex:3];
  [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5]; [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
  [e dispatchThreadgroups:MTLSizeMake((size_t)O*S,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)];
  return true;
}

extern "C" int coli_metal_attn_decode(const float* x,
    const void* qa_w,const float* qa_s,int qa_fmt,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,
    const void* kva_w,const float* kva_s,int kva_fmt,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,
    const void* o_w,const float* o_s,int o_fmt,
    float* Lc,float* Rc,int S,int pos_base,int st0,float eps,float theta,float ascale,float* out){
  if(!g_dev) return 0;
  if(st0!=0 || S<1 || S>AMAXS) return 0;     // partial-KV / S>4 -> CPU
  int T=pos_base+S;
  @autoreleasepool {
    attn_scratch_init();
    // Everything zero-copy: Lc/Rc caches and all weights are registered (page-aligned).
    uint64_t la=0,ra=0,kva=0,ksa=0; id<MTLBuffer> Lb=resolve(Lc,&la), Rb=resolve(Rc,&ra);
    id<MTLBuffer> kvbW=resolve(kvb_w,&kva), kvbS=resolve(kvb_s,&ksa);
    if(!Lb||!Rb||!kvbW||!kvbS) return 0;
    // pre-check the projection weights resolve (they always do; guards mid-encode fallback)
    { uint64_t d; const void* ws[]={qa_w,qa_s,qb_w,qb_s,kva_w,kva_s,o_w,o_s};
      for(auto p:ws) if(!resolve(p,&d)) return 0; }
    size_t loff=la-(uint64_t)[Lb gpuAddress], roff=ra-(uint64_t)[Rb gpuAddress];
    size_t kvbwoff=kva-(uint64_t)[kvbW gpuAddress], kvbsoff=ksa-(uint64_t)[kvbS gpuAddress];
    ascore_=ensure(ascore_,&ascore_cap,(size_t)S*AHEADS*T*4);
    memcpy([ax_ contents],x,(size_t)S*AH*4); memcpy([aqaln_ contents],qa_ln,AQLORA*4); memcpy([akvaln_ contents],kva_ln,AKVL*4);
    size_t Loff=loff+(size_t)pos_base*AKVL*4, Roff=roff+(size_t)pos_base*AROPE*4;   // new-token region

    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:Lb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rb usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    auto BAR=[&]{ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    int zero=0, one=1;
    // Projections interleave the independent q-path and kv-path: 4 barriers instead of 7,
    // so the GPU overlaps them. q: q_a->rmsnorm->q_b->rope ; kv: kv_a->copy->{rmsnorm,rope}.
    auto rms=[&](id<MTLBuffer> b,size_t off,id<MTLBuffer> w,int n,int nrows){ [e setComputePipelineState:g_a_rms];
      [e setBuffer:b offset:off atIndex:0]; [e setBuffer:w offset:0 atIndex:1]; [e setBytes:&n length:4 atIndex:2]; [e setBytes:&eps length:4 atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(nrows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
    auto rope=[&](id<MTLBuffer> b,size_t off,int base,int rs,int hs,int nh){ [e setComputePipelineState:g_a_rope]; [e setBuffer:b offset:off atIndex:0];
      [e setBytes:&base length:4 atIndex:1]; [e setBytes:&rs length:4 atIndex:2]; [e setBytes:&hs length:4 atIndex:3]; [e setBytes:&nh length:4 atIndex:4]; [e setBytes:&pos_base length:4 atIndex:5]; [e setBytes:&theta length:4 atIndex:6];
      [e dispatchThreads:MTLSizeMake((size_t)S*nh*(AROPE/2),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; };
    auto cpy=[&](int off,id<MTLBuffer> dst,size_t doff,int n){ int ss=AKVL+AROPE; [e setComputePipelineState:g_a_copy];
      [e setBuffer:acomp_ offset:0 atIndex:0]; [e setBytes:&off length:4 atIndex:1]; [e setBytes:&ss length:4 atIndex:2];
      [e setBuffer:dst offset:doff atIndex:3]; [e setBytes:&n length:4 atIndex:4]; [e setBytes:&n length:4 atIndex:5];
      [e dispatchThreads:MTLSizeMake((size_t)S*n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; };
    // A: q_a, kv_a (both read ax_)
    bind_gemv(e,qa_w,qa_s,qa_fmt,AH,AQLORA,ax_,aqr_,S);
    bind_gemv(e,kva_w,kva_s,kva_fmt,AH,AKVL+AROPE,ax_,acomp_,S); BAR();
    // B: rmsnorm(q), copy latent+krot
    rms(aqr_,0,aqaln_,AQLORA,S); cpy(0,Lb,Loff,AKVL); cpy(AKVL,Rb,Roff,AROPE); BAR();
    // C: q_b, latent rmsnorm, krot rope
    bind_gemv(e,qb_w,qb_s,qb_fmt,AQLORA,AHQH,aqr_,aqf_,S); rms(Lb,Loff,akvaln_,AKVL,S); rope(Rb,Roff,0,AROPE,0,1); BAR();
    // D: rope(q)
    rope(aqf_,0,ANOPE,AHQH,AQH,AHEADS); BAR();
    (void)zero;(void)one;
    // absorption core (S query rows)
    [e setComputePipelineState:g_a_qabs]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aqf_ offset:0 atIndex:2]; [e setBuffer:aqabs_ offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_score]; [e setBuffer:aqabs_ offset:0 atIndex:0]; [e setBuffer:Lb offset:loff atIndex:1]; [e setBuffer:Rb offset:roff atIndex:2]; [e setBuffer:aqf_ offset:0 atIndex:3]; [e setBuffer:ascore_ offset:0 atIndex:4];
    [e setBytes:&T length:4 atIndex:5]; [e setBytes:&ascale length:4 atIndex:6]; [e setBytes:&pos_base length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*T,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_smax]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBytes:&T length:4 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((size_t)S*AHEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_clat]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBuffer:Lb offset:loff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBytes:&T length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_ctx]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBuffer:actx_ offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AVH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    // o_proj (S rows)
    bind_gemv(e,o_w,o_s,o_fmt,AHVH,AH,actx_,aout_,S);
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] attn cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    g_attn_sched += [cb GPUStartTime]-[cb kernelStartTime]; g_attn_ksched += [cb kernelStartTime]-tc;
    memcpy(out,[aout_ contents],(size_t)S*AH*4);
  }
  return 1;
}

// Sync GEMM for large row-batches (prefill): y[S,O] = x[S,I] @ W^T * scale. Weights must be
// registered (zero-copy); x/y go through grow-only shared scratch. Returns 0 -> CPU fallback.
static id<MTLBuffer> g_gx, g_gy; static size_t g_gx_cap, g_gy_cap;
extern "C" int coli_metal_gemm(float *y, const float *x, const void *wp, const float *sp,
                               int fmt, int S, int I, int O) {
  if (!g_dev || (fmt!=1 && fmt!=2)) return 0;
  @autoreleasepool {
    uint64_t wa=0,sa=0; id<MTLBuffer> wb=resolve(wp,&wa), sb=resolve(sp,&sa);
    if(!wb||!sb) return 0;
    size_t woff=wa-(uint64_t)[wb gpuAddress], soff=sa-(uint64_t)[sb gpuAddress];
    g_gx=ensure(g_gx,&g_gx_cap,(size_t)S*I*4); g_gy=ensure(g_gy,&g_gy_cap,(size_t)S*O*4);
    memcpy([g_gx contents],x,(size_t)S*I*4);
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:wb usage:MTLResourceUsageRead]; [e useResource:sb usage:MTLResourceUsageRead];
    [e setComputePipelineState:g_gemv];
    [e setBuffer:wb offset:woff atIndex:0]; [e setBuffer:sb offset:soff atIndex:1];
    [e setBuffer:g_gx offset:0 atIndex:2]; [e setBuffer:g_gy offset:0 atIndex:3];
    [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5];
    [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake((size_t)O*S,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] gemm cmdbuf error (S=%d O=%d)\n",S,O); return 0; }
    memcpy(y,[g_gy contents],(size_t)S*O*4);
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
// Encode + commit a MoE block (no wait). Writes hh[R,D] into hh_buf. Returns nil on
// unresolved slab / bad fmt (caller falls back to CPU).
static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr, int R,
                         id<MTLBuffer> xg_buf, id<MTLBuffer> gg_buf, id<MTLBuffer> uu_buf, id<MTLBuffer> hh_buf) {
  if (!g_dev || (fmt != 1 && fmt != 2)) return nil;
  double ts_start = mnow();
  std::vector<uint64_t> ag(nb),au(nb),ad(nb),sgv(nb),suv(nb),sdv(nb);
  std::vector<id<MTLBuffer>> use; use.reserve(nb*2);
  auto add_use=[&](id<MTLBuffer> b){ for(auto&x:use) if(x==b) return; use.push_back(b); };
  for (int e=0;e<nb;e++) {
    id<MTLBuffer> b;
    if(!(b=resolve(g[e],&ag[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(u[e],&au[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(d[e],&ad[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(gs[e],&sgv[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(us[e],&suv[e]))) {g_moe_fb++; return nil;} add_use(b);
    if(!(b=resolve(ds[e],&sdv[e]))) {g_moe_fb++; return nil;} add_use(b);
  }
  std::vector<int> erow(R); for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++) erow[xoff[e]+r]=e;
  auto shb=[&](const void*p,size_t n){ return [g_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared]; };
  id<MTLBuffer> bag=shb(ag.data(),nb*8), bau=shb(au.data(),nb*8), bad=shb(ad.data(),nb*8);
  id<MTLBuffer> bsg=shb(sgv.data(),nb*8), bsu=shb(suv.data(),nb*8), bsd=shb(sdv.data(),nb*8);
  id<MTLBuffer> berow=shb(erow.data(),R*4);
  memcpy([xg_buf contents], xg, (size_t)R*D*4);

  id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];
  auto gemv=[&](id<MTLBuffer> wa,id<MTLBuffer> sa,id<MTLBuffer> xin,id<MTLBuffer> y,int O,int K,int Kin){
    [e setComputePipelineState:g_moe_gemv];
    [e setBuffer:wa offset:0 atIndex:0];[e setBuffer:sa offset:0 atIndex:1];[e setBuffer:berow offset:0 atIndex:2];
    [e setBuffer:xin offset:0 atIndex:3];[e setBuffer:y offset:0 atIndex:4];
    [e setBytes:&O length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&Kin length:4 atIndex:7];[e setBytes:&fmt length:4 atIndex:8];
    [e dispatchThreadgroups:MTLSizeMake((size_t)R*O,1,1) threadsPerThreadgroup:MTLSizeMake(TG,1,1)]; };
  gemv(bag,bsg,xg_buf,gg_buf,Iinter,D,D);                     // gate
  gemv(bau,bsu,xg_buf,uu_buf,Iinter,D,D);                     // up
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [e setComputePipelineState:g_moe_silu];
  [e setBuffer:gg_buf offset:0 atIndex:0];[e setBuffer:uu_buf offset:0 atIndex:1];
  [e dispatchThreads:MTLSizeMake((size_t)R*Iinter,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  gemv(bad,bsd,gg_buf,hh_buf,D,Iinter,Iinter);                // down
  g_t_setup += mnow() - ts_start;
  [e endEncoding];[cb commit];
  return cb;
}

// Wait + error-check + scatter-add hh into out. Returns 0 on GPU fault.
static int moe_finish(id<MTLCommandBuffer> cb, id<MTLBuffer> hh_buf, int nb, int R, int D,
                      const int *rows, const float *rw, float *out) {
  double t0 = mnow();
  [cb waitUntilCompleted];
  double ts_gpu = mnow(); g_t_gpu += ts_gpu - t0;
  g_t_kernel += [cb GPUEndTime] - [cb GPUStartTime];
  if (cb.status == MTLCommandBufferStatusError) {
    fprintf(stderr, "[metal] moe_block cmdbuf error (nb=%d R=%d): %s\n", nb, R,
            cb.error ? [[cb.error localizedDescription] UTF8String] : "?");
    g_moe_fb++; return 0;
  }
  const float *hh=(const float*)[hh_buf contents];
  for(int gr=0;gr<R;gr++){ float *os=out+(size_t)rows[gr]*D, w=rw[gr]; const float *hr=hh+(size_t)gr*D;
    for(int dd=0;dd<D;dd++) os[dd]+=w*hr[dd]; }
  g_t_scatter += mnow() - ts_gpu;
  g_moe_ok++; g_moe_experts += nb;
  return 1;
}

extern "C" int coli_metal_moe_block(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw, float *out, int S) {
  (void)S;
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0) return 1;
    g_xg = ensure(g_xg,&g_xg_cap,(size_t)R*D*4);
    g_gg = ensure(g_gg,&g_gg_cap,(size_t)R*Iinter*4);
    g_uu = ensure(g_uu,&g_uu_cap,(size_t)R*Iinter*4);
    g_hh = ensure(g_hh,&g_hh_cap,(size_t)R*D*4);
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);
    if (!cb) return 0;
    return moe_finish(cb,g_hh,nb,R,D,rows,rw,out);
  }
}

// Async two-phase API: begin submits the block (own scratch, no wait) so the CPU can
// overlap disk loads with GPU compute; end waits + scatters. Handle owns everything.
struct ColiMetalMoeHandle {
  id<MTLCommandBuffer> cb; id<MTLBuffer> hh;
  std::vector<int> rows; std::vector<float> rwv;
  int nb, R, D;
};
extern "C" ColiMetalMoeHandle* coli_metal_moe_block_begin(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw) {
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0 || !g_dev) return nullptr;
    id<MTLBuffer> bxg=[g_dev newBufferWithLength:(size_t)R*D*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bgg=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buu=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bhh=[g_dev newBufferWithLength:(size_t)R*D*4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,g,u,d,gs,us,ds,xg,xoff,nr,R,bxg,bgg,buu,bhh);
    if (!cb) return nullptr;
    ColiMetalMoeHandle *h = new ColiMetalMoeHandle();
    h->cb=cb; h->hh=bhh; h->rows.assign(rows,rows+R); h->rwv.assign(rw,rw+R);
    h->nb=nb; h->R=R; h->D=D;
    return h;
  }
}
extern "C" int coli_metal_moe_block_end(ColiMetalMoeHandle *h, float *out) {
  if (!h) return 0;
  int ok;
  @autoreleasepool { ok = moe_finish(h->cb,h->hh,h->nb,h->R,h->D,h->rows.data(),h->rwv.data(),out); }
  h->cb=nil; h->hh=nil; delete h;
  return ok;
}
