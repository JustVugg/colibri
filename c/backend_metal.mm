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
)METAL";

struct ColiMetalTensor {
  id<MTLBuffer> w;      // weights (wrapped, zero-copy when page-aligned)
  id<MTLBuffer> s;      // scales
  int fmt, I, O; size_t wbytes;
};

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_gemv;
static size_t g_tensor_count, g_tensor_bytes;
static const int TG = 128;

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
    g_gemv = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_gemv"] error:&err];
    if (!g_gemv) { fprintf(stderr, "[metal] pipeline failed\n"); g_dev = nil; return 0; }
  }
  return 1;
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

// Milestone 2: batched MoE block. Returns 0 so glm.c keeps the CPU path for now.
extern "C" int coli_metal_moe_block(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw, float *out, int S) {
  (void)nb;(void)D;(void)Iinter;(void)fmt;(void)g;(void)u;(void)d;(void)gs;(void)us;(void)ds;
  (void)xg;(void)xoff;(void)nr;(void)rows;(void)rw;(void)out;(void)S;
  return 0;
}
