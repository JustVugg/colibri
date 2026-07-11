// Apple-GPU (Metal) backend for colibrì. Runtime-compiled shader (no Xcode needed),
// zero-copy over unified memory. See backend_metal.h and docs/plans/2026-07-10-*.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backend_metal.h"
#include <cstring>
#include <vector>

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
)METAL";

struct ColiMetalTensor {
  id<MTLBuffer> w;      // weights (wrapped, zero-copy when page-aligned)
  id<MTLBuffer> s;      // scales
  int fmt, I, O; size_t wbytes;
};

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_gemv, g_moe_gemv, g_moe_silu;
static size_t g_tensor_count, g_tensor_bytes;
static const int TG = 128;

// Registry of page-aligned host slabs wrapped zero-copy for the batched MoE path.
struct Slab { void *base; size_t len; id<MTLBuffer> buf; };
static std::vector<Slab> g_slabs;
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
    if (!g_gemv || !g_moe_gemv || !g_moe_silu) { fprintf(stderr, "[metal] pipeline failed\n"); g_dev = nil; return 0; }
  }
  return 1;
}

extern "C" void coli_metal_register(void *base, size_t len) {
  if (!g_dev || !base) return;
  for (auto &s : g_slabs) if (s.base == base) { s.len = len; return; }  // already registered
  id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:base length:len
                              options:MTLResourceStorageModeShared deallocator:nil];
  if (b) g_slabs.push_back({base, len, b});
}
extern "C" void coli_metal_unregister(void *base) {
  for (size_t i=0;i<g_slabs.size();i++) if (g_slabs[i].base==base) { g_slabs[i].buf=nil; g_slabs.erase(g_slabs.begin()+i); return; }
}
// Resolve a host pointer inside a registered slab to (buffer, gpuAddress). Returns nil if unknown.
static id<MTLBuffer> resolve(const void *p, uint64_t *addr) {
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
    // address + erow tables
    std::vector<uint64_t> ag(nb),au(nb),ad(nb),sgv(nb),suv(nb),sdv(nb);
    std::vector<id<MTLBuffer>> use; use.reserve(nb*2);
    auto add_use=[&](id<MTLBuffer> b){ for(auto&x:use) if(x==b) return; use.push_back(b); };
    for (int e=0;e<nb;e++) {
      id<MTLBuffer> b;
      if(!(b=resolve(g[e],&ag[e]))) return 0; add_use(b);
      if(!(b=resolve(u[e],&au[e]))) return 0; add_use(b);
      if(!(b=resolve(d[e],&ad[e]))) return 0; add_use(b);
      if(!(b=resolve(gs[e],&sgv[e]))) return 0; add_use(b);
      if(!(b=resolve(us[e],&suv[e]))) return 0; add_use(b);
      if(!(b=resolve(ds[e],&sdv[e]))) return 0; add_use(b);
    }
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
    [e endEncoding];[cb commit];[cb waitUntilCompleted];

    // scatter-add: out[rows[gr]] += rw[gr] * hh[gr]
    const float *hh=(const float*)[g_hh contents];
    for(int gr=0;gr<R;gr++){ float *os=out+(size_t)rows[gr]*D, w=rw[gr]; const float *hr=hh+(size_t)gr*D;
      for(int dd=0;dd<D;dd++) os[dd]+=w*hr[dd]; }
  }
  return 1;
}
