// Apple-GPU (Metal) backend for colibrì. Runtime-compiled shader (no Xcode needed),
// zero-copy over unified memory. See backend_metal.h and docs/plans/2026-07-10-*.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backend_metal.h"
#include <cstring>
#include <vector>
#include <mutex>
#include <unordered_map>

// ---- shader: general quantized GEMV, one threadgroup per output element (o,si) ----
// y[si,o] = (sum_i dequant(W[o,i]) * x[si,i]) * scale[o]. fmt: 0=f32 1=i8 2=i4 3=i2.
// fmt=4 (grouped int4, dev e9b3614 matmul_i4_grouped, #298/#451 CUDA twin): same packed
// nibble layout as fmt=2, but scale is PER-GROUP (gsz elements of I), buffer(1) holds
// [O, ceil(I/gsz)] floats indexed scale[o*ng+g] -- so unlike fmt 1-3, the group scale is
// folded into the accumulation itself (see the fmt==4 branch), not applied once at the end.
// fmt=8 (native FP8-e4m3 passthrough -- see colibri.c): raw byte
// layout identical to fmt=1 (one e4m3 byte per element, no packing), but the scale is
// per-128x128 BLOCK of [O,I] -- buffer(1) holds [ceil(O/128),ceil(I/128)] floats indexed
// scale[(o/128)*nblkI+i/128]. Folded into acc like fmt=4's per-group scale, for the
// same reason: it is not constant across one output row. e4m3->float is bit manipulation
// in-kernel (sign/exp/mantissa, OCP E4M3-FN: exp==0 subnormal, exp==0xF&&mant==0x7 is the
// only NaN code) -- BW-bound kernel, decode ALU is free, no LUT texture. Must byte-match
// quant.h's E4M3_LUT/e4m3_decode (the CPU reference) including the NaN policy, which is
// why the metal-test suite runs all 256 byte codes through this exact kernel path (not
// just spot values) -- see run_fp8_lut().
static const char *SHADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

// fmt=6 E8/IQ3 magnitude grid — generated from quant.h e8_grid (must stay identical;
// the metal-test oracle compares against the CPU decoder, so drift fails the build).
constant uchar4 E8G[256] = {
  uchar4(4,4,4,4),uchar4(20,4,4,4),uchar4(36,4,4,4),uchar4(12,12,4,4),uchar4(28,12,4,4),uchar4(62,12,4,4),uchar4(4,20,4,4),uchar4(20,20,4,4),
  uchar4(12,28,4,4),uchar4(20,36,4,4),uchar4(28,62,4,4),uchar4(44,62,4,4),uchar4(12,4,12,4),uchar4(28,4,12,4),uchar4(4,12,12,4),uchar4(20,12,12,4),
  uchar4(12,20,12,4),uchar4(44,20,12,4),uchar4(4,28,12,4),uchar4(20,28,12,4),uchar4(12,36,12,4),uchar4(36,44,12,4),uchar4(4,62,12,4),uchar4(4,4,20,4),
  uchar4(20,4,20,4),uchar4(36,4,20,4),uchar4(12,12,20,4),uchar4(4,20,20,4),uchar4(20,20,20,4),uchar4(12,28,20,4),uchar4(28,28,20,4),uchar4(62,28,20,4),
  uchar4(12,44,20,4),uchar4(62,44,20,4),uchar4(44,62,20,4),uchar4(12,4,28,4),uchar4(62,4,28,4),uchar4(4,12,28,4),uchar4(20,12,28,4),uchar4(44,20,28,4),
  uchar4(4,62,28,4),uchar4(28,12,36,4),uchar4(62,28,36,4),uchar4(36,36,36,4),uchar4(62,44,36,4),uchar4(28,62,36,4),uchar4(44,62,36,4),uchar4(12,4,44,4),
  uchar4(62,4,44,4),uchar4(20,28,44,4),uchar4(20,44,44,4),uchar4(44,28,52,4),uchar4(36,52,52,4),uchar4(4,12,62,4),uchar4(36,12,62,4),uchar4(52,12,62,4),
  uchar4(28,36,62,4),uchar4(12,52,62,4),uchar4(12,4,4,12),uchar4(28,4,4,12),uchar4(4,12,4,12),uchar4(20,12,4,12),uchar4(12,20,4,12),uchar4(28,20,4,12),
  uchar4(4,28,4,12),uchar4(20,28,4,12),uchar4(36,28,4,12),uchar4(62,36,4,12),uchar4(4,44,4,12),uchar4(4,4,12,12),uchar4(20,4,12,12),uchar4(12,12,12,12),
  uchar4(4,20,12,12),uchar4(20,20,12,12),uchar4(12,4,20,12),uchar4(28,4,20,12),uchar4(4,12,20,12),uchar4(20,12,20,12),uchar4(12,20,20,12),uchar4(4,28,20,12),
  uchar4(20,62,20,12),uchar4(4,4,28,12),uchar4(20,4,28,12),uchar4(4,20,28,12),uchar4(12,28,28,12),uchar4(52,36,28,12),uchar4(52,52,28,12),uchar4(12,4,36,12),
  uchar4(44,4,36,12),uchar4(4,44,36,12),uchar4(4,20,44,12),uchar4(36,20,44,12),uchar4(52,36,44,12),uchar4(12,62,44,12),uchar4(44,4,52,12),uchar4(20,20,62,12),
  uchar4(4,36,62,12),uchar4(4,4,4,20),uchar4(20,4,4,20),uchar4(12,12,4,20),uchar4(28,12,4,20),uchar4(4,20,4,20),uchar4(20,20,4,20),uchar4(52,20,4,20),
  uchar4(12,28,4,20),uchar4(20,36,4,20),uchar4(12,4,12,20),uchar4(28,4,12,20),uchar4(44,4,12,20),uchar4(4,12,12,20),uchar4(20,12,12,20),uchar4(12,20,12,20),
  uchar4(4,28,12,20),uchar4(28,52,12,20),uchar4(62,52,12,20),uchar4(4,62,12,20),uchar4(4,4,20,20),uchar4(20,4,20,20),uchar4(12,12,20,20),uchar4(62,12,20,20),
  uchar4(4,20,20,20),uchar4(20,20,20,20),uchar4(62,28,20,20),uchar4(4,36,20,20),uchar4(44,44,20,20),uchar4(12,4,28,20),uchar4(4,12,28,20),uchar4(36,12,28,20),
  uchar4(4,62,28,20),uchar4(36,62,28,20),uchar4(44,28,36,20),uchar4(28,44,36,20),uchar4(28,4,44,20),uchar4(62,20,44,20),uchar4(12,36,44,20),uchar4(36,62,44,20),
  uchar4(12,4,62,20),uchar4(28,4,62,20),uchar4(52,12,62,20),uchar4(44,36,62,20),uchar4(12,4,4,28),uchar4(4,12,4,28),uchar4(20,12,4,28),uchar4(12,20,4,28),
  uchar4(28,20,4,28),uchar4(4,44,4,28),uchar4(44,52,4,28),uchar4(20,62,4,28),uchar4(4,4,12,28),uchar4(20,4,12,28),uchar4(4,20,12,28),uchar4(12,28,12,28),
  uchar4(36,36,12,28),uchar4(52,36,12,28),uchar4(12,4,20,28),uchar4(28,4,20,28),uchar4(4,12,20,28),uchar4(44,20,20,28),uchar4(20,44,20,28),uchar4(20,62,20,28),
  uchar4(12,12,28,28),uchar4(28,28,28,28),uchar4(4,28,36,28),uchar4(62,36,36,28),uchar4(20,62,36,28),uchar4(4,4,44,28),uchar4(52,4,44,28),uchar4(20,20,44,28),
  uchar4(44,44,44,28),uchar4(36,12,52,28),uchar4(52,28,52,28),uchar4(28,52,52,28),uchar4(28,28,62,28),uchar4(4,52,62,28),uchar4(36,4,4,36),uchar4(62,12,4,36),
  uchar4(44,28,4,36),uchar4(62,28,4,36),uchar4(28,44,4,36),uchar4(62,44,4,36),uchar4(36,62,12,36),uchar4(4,20,20,36),uchar4(62,28,20,36),uchar4(4,36,20,36),
  uchar4(4,52,20,36),uchar4(52,52,20,36),uchar4(62,4,28,36),uchar4(44,36,28,36),uchar4(36,4,36,36),uchar4(12,44,36,36),uchar4(36,52,36,36),uchar4(44,20,44,36),
  uchar4(28,36,44,36),uchar4(4,62,44,36),uchar4(44,4,62,36),uchar4(4,12,62,36),uchar4(20,12,62,36),uchar4(4,28,62,36),uchar4(20,12,4,44),uchar4(12,36,4,44),
  uchar4(4,62,4,44),uchar4(4,4,12,44),uchar4(52,4,12,44),uchar4(52,20,12,44),uchar4(44,44,12,44),uchar4(36,12,20,44),uchar4(20,28,20,44),uchar4(20,62,20,44),
  uchar4(20,4,28,44),uchar4(28,44,28,44),uchar4(4,12,36,44),uchar4(28,20,36,44),uchar4(62,20,36,44),uchar4(20,62,36,44),uchar4(20,4,44,44),uchar4(12,28,44,44),
  uchar4(4,44,52,44),uchar4(36,20,62,44),uchar4(20,36,62,44),uchar4(36,20,4,52),uchar4(36,36,4,52),uchar4(52,36,4,52),uchar4(36,52,4,52),uchar4(12,20,12,52),
  uchar4(12,52,12,52),uchar4(62,12,20,52),uchar4(36,52,20,52),uchar4(4,28,28,52),uchar4(52,28,28,52),uchar4(36,36,36,52),uchar4(44,4,44,52),uchar4(20,44,44,52),
  uchar4(28,28,52,52),uchar4(28,4,62,52),uchar4(12,20,62,52),uchar4(28,4,4,62),uchar4(44,4,4,62),uchar4(62,4,4,62),uchar4(4,12,4,62),uchar4(20,28,4,62),
  uchar4(20,44,4,62),uchar4(52,20,12,62),uchar4(4,36,12,62),uchar4(20,12,20,62),uchar4(44,36,20,62),uchar4(20,44,20,62),uchar4(4,4,28,62),uchar4(44,12,28,62),
  uchar4(28,28,28,62),uchar4(4,52,28,62),uchar4(12,20,36,62),uchar4(12,36,36,62),uchar4(4,4,44,62),uchar4(20,4,44,62),uchar4(36,20,44,62),uchar4(4,28,52,62)
};

kernel void mm_gemv(device const uchar* w      [[buffer(0)]],   // raw weight bytes
                    device const float* scale  [[buffer(1)]],   // [O] (fmt<4) or [O,ceil(I/gsz)] (fmt==4)
                    device const float* x      [[buffer(2)]],   // [S,I]
                    device float*       y      [[buffer(3)]],   // [S,O]
                    constant int& S [[buffer(4)]], constant int& I [[buffer(5)]],
                    constant int& O [[buffer(6)]], constant int& fmt [[buffer(7)]],
                    constant int& NT [[buffer(8)]],
                    constant int& gsz [[buffer(9)]],             // fmt==4 group size (ignored otherwise)
                    uint tg [[threadgroup_position_in_grid]],
                    uint slane [[thread_index_in_simdgroup]],
                    uint sgid [[simdgroup_index_in_threadgroup]]) {
  // one SIMDGROUP per output element, 4 per threadgroup, 8-value loads (see moe_gemv)
  long row = (long)tg*4 + sgid; if (row >= NT) return;
  int o = row % O, si = row / O;
  device const float* xr = x + (long)si * I;
  device const float4* x4 = (device const float4*)xr;
  int I8 = (I & 7) ? 0 : (I/8);
  float acc = 0.0f;
  if (fmt == 1) {                                   // int8
    device const char* wr = (device const char*)(w) + (long)o * I;
    device const char4* w4 = (device const char4*)wr;
    for (int c = slane; c < I8; c += 32) acc += dot(float4(w4[2*c]),x4[2*c]) + dot(float4(w4[2*c+1]),x4[2*c+1]);
    for (int i = I8*8 + slane; i < I; i += 32) acc += float(wr[i]) * xr[i];
  } else if (fmt == 2) {                            // int4 packed, rb=(I+1)/2
    int rb = (I+1)/2;
    device const uchar* wr = w + (long)o * rb;
    device const uchar4* w4 = (device const uchar4*)wr;
    for (int c = slane; c < I8; c += 32) { uchar4 b = w4[c];
      float4 w0 = float4(float(int(b.x&0xF)-8), float(int(b.x>>4)-8), float(int(b.y&0xF)-8), float(int(b.y>>4)-8));
      float4 w1 = float4(float(int(b.z&0xF)-8), float(int(b.z>>4)-8), float(int(b.w&0xF)-8), float(int(b.w>>4)-8));
      acc += dot(w0,x4[2*c]) + dot(w1,x4[2*c+1]);
    }
    for (int i = I8*8 + slane; i < I; i += 32) {
      uchar b = wr[i>>1]; int v = (i&1) ? (b>>4) : (b&0xF); acc += float(v-8) * xr[i];
    }
  } else if (fmt == 3) {                            // int2 packed, rb=(I+3)/4
    int rb = (I+3)/4;
    device const uchar* wr = w + (long)o * rb;
    for (int i = slane; i < I; i += 32) {
      uchar b = wr[i>>2]; int v = (b >> (2*(i&3))) & 0x3; acc += float(v-2) * xr[i];
    }
  } else if (fmt == 4) {                            // int4 GROUPED: same nibble packing as fmt=2,
                                                     // one f32 scale per gsz-element group along I.
                                                     // Each lane owns one packed byte (2 elements) per
                                                     // 64-lane stride -> memory access stays coalesced,
                                                     // and a group never splits a byte (gsz is even).
    int rb = (I+1)/2; int ng = (I+gsz-1)/gsz;
    device const uchar* wr = w + (long)o * rb;
    device const float* scl = scale + (long)o * ng;
    for (int i = slane*2; i < I; i += 64) {
      uchar b = wr[i>>1];
      int g0 = i / gsz; float sc0 = scl[g0];
      acc += float(int(b&0xF)-8) * xr[i] * sc0;
      if (i+1 < I) {
        int g1 = (i+1) / gsz; float sc1 = (g1==g0) ? sc0 : scl[g1];
        acc += float(int(b>>4)-8) * xr[i+1] * sc1;
      }
    }
  } else if (fmt == 8) {                            // fp8 e4m3 passthrough: one raw byte per
                                                     // element (like fmt=1), scale per 128x128
                                                     // block folded into acc (like a grouped fmt).
    int nblkI = (I + 127) / 128;
    device const uchar* wr = w + (long)o * I;
    device const float* scl = scale + (long)(o/128) * nblkI;
    for (int i = slane; i < I; i += 32) {
      uchar b = wr[i];
      uint sign = b >> 7, exp = (b >> 3) & 0xF, mant = b & 0x7;
      float wv;
      if (exp == 0xF && mant == 0x7) {
        wv = as_type<float>(0x7fc00000u);           // qNaN -- matches quant.h's e4m3_decode
      } else {
        float mag = (exp == 0) ? (float(mant) * 0.001953125f)                 // subnormal: mant*2^-9
                                : (1.0f + float(mant)*0.125f) * exp2(float(int(exp) - 7));
        wv = sign ? -mag : mag;
      }
      acc += wv * xr[i] * scl[i/128];
    }
  } else {                                          // f32
    device const float* wr = (device const float*)(w) + (long)o * I;
    device const float4* w4 = (device const float4*)wr;
    for (int c = slane; c < I8; c += 32) acc += dot(w4[2*c],x4[2*c]) + dot(w4[2*c+1],x4[2*c+1]);
    for (int i = I8*8 + slane; i < I; i += 32) acc += wr[i] * xr[i];
  }
  acc = simd_sum(acc);
  // fmt==4 (per-group) and fmt==8 (per-block) already folded their scale into acc
  // above -- do not scale again.
  if (slane == 0) y[row] = (fmt == 4 || fmt == 8) ? acc : acc * scale[o];
}

// Batched bindless expert GEMV: each row gr belongs to expert erow[gr], whose weight and
// scale live at gpuAddresses waddr[e]/saddr[e] (zero-copy in the RAM slab). fmt 1=i8, 2=i4
// per-row, 4=i4 grouped (scale layout [O][ng], ng=ceil(K/qgs) -- same convention as mm_gemv
// fmt=4 above, but folded into the vectorized uchar4/float4 dot-product loop this kernel's
// fmt=2 branch already uses, since moe_gemv has no scalar strided branch to reuse).
// One SIMDGROUP per output row, 4 rows/threadgroup, 8-value loads: measured 1.5-2.1x over
// one-threadgroup-per-row with uchar2 loads (358-389 GB/s on engine-like block shapes).
kernel void moe_gemv(device const ulong* waddr [[buffer(0)]], device const ulong* saddr [[buffer(1)]],
                     device const int* erow [[buffer(2)]], device const float* xin [[buffer(3)]],
                     device float* yout [[buffer(4)]],
                     constant int& O [[buffer(5)]], constant int& K [[buffer(6)]],
                     constant int& Kin [[buffer(7)]], constant int& fmt [[buffer(8)]],
                     constant int& NT [[buffer(9)]], constant int& qgs [[buffer(10)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint slane [[thread_index_in_simdgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row = (long)tg*4 + sgid; if (row >= NT) return;
  int gr = row / O, o = row % O; int e = erow[gr]; int K8 = (K & 7) ? 0 : (K/8);
  device const float* xr = xin + (long)gr * Kin;
  device const float* sc = (device const float*)(saddr[e]);
  device const float4* x4 = (device const float4*)xr;
  float acc = 0.0f;
  if (fmt == 2) { int rb=(K+1)/2; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      acc+=dot(w0,x4[2*c])+dot(w1,x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]; }
  } else if (fmt == 6) {                            // E8/IQ3: 98B per 256 weights, scales in-block
    long rb=((long)(K+255)/256)*98;                 // host guards K%256==0 (GLM dims are)
    device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    int nsub=K/32;                                  // one 32-weight sub-block per lane step
    for(int s6=slane;s6<nsub;s6+=32){
      int b=s6>>3, ib=s6&7; device const uchar* blk=w+(long)b*98;
      uint word = uint(blk[64+ib*4]) | (uint(blk[65+ib*4])<<8)
                | (uint(blk[66+ib*4])<<16) | (uint(blk[67+ib*4])<<24);
      ushort dh = ushort(blk[96]) | (ushort(blk[97])<<8);
      float db = float(as_type<half>(dh)) * (0.5f + float((word>>28)&0xFu)) * 0.5f;
      device const uchar* idx = blk + ib*8;
      device const float4* xs = (device const float4*)(xr + s6*32);
      for(int l=0;l<4;l++){
        uint sv=(word>>(7*l))&0x7Fu;
        float4 m0=float4(E8G[idx[l*2+0]]), m1=float4(E8G[idx[l*2+1]]);
        float4 sA=float4((sv&1u)?-1.0f:1.0f,(sv&2u)?-1.0f:1.0f,(sv&4u)?-1.0f:1.0f,(sv&8u)?-1.0f:1.0f);
        float4 sB=float4((sv&16u)?-1.0f:1.0f,(sv&32u)?-1.0f:1.0f,(sv&64u)?-1.0f:1.0f,
                         (popcount(sv)&1u)?-1.0f:1.0f);   // j=7 closes by odd parity
        acc += (dot(m0*sA,xs[l*2]) + dot(m1*sB,xs[l*2+1])) * (0.5f*db);
      }
    }
  } else if (fmt == 5) {                            // bf16 raw rows: no scales, upper half of f32
    device const ushort* w=(device const ushort*)(waddr[e])+(long)o*K;
    device const ushort4* w4=(device const ushort4*)w;
    for(int c=slane;c<K8;c+=32){ ushort4 a=w4[2*c], b=w4[2*c+1];
      float4 w0=float4(as_type<float>(uint(a.x)<<16), as_type<float>(uint(a.y)<<16),
                       as_type<float>(uint(a.z)<<16), as_type<float>(uint(a.w)<<16));
      float4 w1=float4(as_type<float>(uint(b.x)<<16), as_type<float>(uint(b.y)<<16),
                       as_type<float>(uint(b.z)<<16), as_type<float>(uint(b.w)<<16));
      acc+=dot(w0,x4[2*c])+dot(w1,x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32) acc += as_type<float>(uint(w[i])<<16)*xr[i];
  } else if (fmt == 4) {                            // grouped int4: per-expert scale [O][ng]
    int rb=(K+1)/2, ng=(K+qgs-1)/qgs; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const float* sr=sc+(long)o*ng;           // grouped scales for this output row
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      int g0=(8*c+0)/qgs,g1=(8*c+1)/qgs,g2=(8*c+2)/qgs,g3=(8*c+3)/qgs;
      int g4=(8*c+4)/qgs,g5=(8*c+5)/qgs,g6=(8*c+6)/qgs,g7=(8*c+7)/qgs;
      acc+=dot(w0*float4(sr[g0],sr[g1],sr[g2],sr[g3]),x4[2*c])
          +dot(w1*float4(sr[g4],sr[g5],sr[g6],sr[g7]),x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]*sr[i/qgs]; }
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
    device const char4* w4=(device const char4*)w;
    for(int c=slane;c<K8;c+=32) acc+=dot(float4(w4[2*c]),x4[2*c])+dot(float4(w4[2*c+1]),x4[2*c+1]);
    for(int i=K8*8+slane;i<K;i+=32) acc+=float(w[i])*xr[i];
  }
  acc=simd_sum(acc);
  if(slane==0) yout[row] = (fmt==4 || fmt==5 || fmt==6) ? acc : acc*sc[o];   // fmt4 grouped / fmt6 in-block scales folded; fmt5 none
}

// fmt=6 activation rotation for the GPU-resident down-projection input: one FWHT
// tile per dispatch (block-diagonal tiling and sign stream match quant.h e8_rot_rows;
// signs are regenerated host-side with e8_signs and passed in). One threadgroup per
// row; tile fits threadgroup memory (n <= 4096).
kernel void moe_fwht(device float* v [[buffer(0)]], device const uchar* signs [[buffer(1)]],
                     constant int& dim [[buffer(2)]], constant int& off [[buffer(3)]],
                     constant int& n [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
  threadgroup float sh[4096];
  device float* row = v + (long)tg*dim + off;
  for (int i=int(t);i<n;i+=int(nt)){ float x=row[i]; if((signs[i>>3]>>(i&7))&1u) x=-x; sh[i]=x; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int len=1;len<n;len<<=1){
    for (int j=int(t);j<n/2;j+=int(nt)){
      int blk=j/len, k=j%len, i0=blk*(len<<1)+k;
      float a=sh[i0], b=sh[i0+len]; sh[i0]=a+b; sh[i0+len]=a-b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float s=rsqrt(float(n));
  for (int i=int(t);i<n;i+=int(nt)) row[i]=sh[i]*s;
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
// kv_b inline dequant of column i of output row `row`. fmt=2 -> one scale per row;
// fmt=4 -> grouped int4, one scale per gs-wide group along the A_KVL input dim
// (scale layout [O][ng], ng=ceil(A_KVL/gs)), matching QT fmt=4 / mm_gemv above.
inline float a_deqrow(device const uchar* base, int row, int i, device const float* sc, int fmt, int gs){
  device const uchar* w=base+(long)row*((A_KVL+1)/2); uchar b=w[i>>1]; int val=(i&1)?(b>>4):(b&0xF);
  float s = (fmt==4) ? sc[(long)row*((A_KVL+gs-1)/gs) + i/gs] : sc[row];
  return float(val-8)*s; }
kernel void a_qabs(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                   device const float* q [[buffer(2)]], device float* qabs [[buffer(3)]],
                   constant int& fmt [[buffer(4)]], constant int& gs [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) {
  int s=gid/(A_H*A_KVL), r=gid%(A_H*A_KVL), h=r/A_KVL, i=r%A_KVL; int rbase=h*A_ROWSH;
  device const float* qp=q+(long)s*A_QHH+(long)h*A_QH;
  float a=0; for(int d=0;d<A_NOPE;d++) a+=qp[d]*a_deqrow(kvb,rbase+d,i,sc,fmt,gs); qabs[(long)(s*A_H+h)*A_KVL+i]=a;
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
                  device const float* clat [[buffer(2)]], device float* ctx [[buffer(3)]],
                  constant int& fmt [[buffer(4)]], constant int& gs [[buffer(5)]],
                  uint gid [[thread_position_in_grid]]) {
  int sh=gid/A_VH, j=gid%A_VH, h=sh%A_H; int row=h*A_ROWSH+A_NOPE+j; device const float* cl=clat+(long)sh*A_KVL;
  float a=0; for(int i=0;i<A_KVL;i++) a+=cl[i]*a_deqrow(kvb,row,i,sc,fmt,gs); ctx[(long)sh*A_VH+j]=a;
}

// ===== full-layer tail kernels =====
// y[i] += a[i]  (residual add), grid = n
kernel void a_add(device float* y [[buffer(0)]], device const float* a [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { y[i] += a[i]; }
// router: logit[s][e] = x[s].w_e (f32 rows [E,D]) -> sig=1/(1+exp(-logit)). One simdgroup/row.
kernel void r_router(device const float* rw [[buffer(0)]], device const float* x [[buffer(1)]],
                     device float* sig [[buffer(2)]], constant int& E [[buffer(3)]],
                     constant int& D [[buffer(4)]], constant int& NT [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint slane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row=(long)tg*4+sgid; if(row>=NT) return;
  int e=row%E, s=row/E;
  device const float4* w4=(device const float4*)(rw+(long)e*D);
  device const float4* x4=(device const float4*)(x+(long)s*D);
  float acc=0; int D4=D/4;
  for(int c=slane;c<D4;c+=32) acc+=dot(w4[c],x4[c]);
  acc=simd_sum(acc);
  if(slane==0) sig[row]=1.0f/(1.0f+exp(-acc));
}
// exact replica of glm.c phase-A selection per row s (serial, deterministic ties):
// choice=sig+bias; greedy top-Ksel by choice; w=sig[best]; optional topp truncation
// (insertion-sort desc + cumulative); optional norm_topk; * routed_scale.
kernel void r_top8(device const float* sig [[buffer(0)]], device const float* bias [[buffer(1)]],
                   device int* idx [[buffer(2)]], device float* w [[buffer(3)]],
                   device int* keff [[buffer(4)]], constant int& E [[buffer(5)]],
                   constant int& K [[buffer(6)]], constant int& Ksel [[buffer(7)]],
                   constant float& topp [[buffer(8)]], constant int& normk [[buffer(9)]],
                   constant float& rscale [[buffer(10)]],
                   uint s [[thread_position_in_grid]]) {
  device const float* sg=sig+(long)s*E;
  device int* id_=idx+(long)s*K; device float* ww=w+(long)s*K;
  for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
    for(int e=0;e<E;e++){ bool tk=false; for(int j=0;j<kk;j++) if(id_[j]==e){tk=true;break;}
      float ch=sg[e]+bias[e];
      if(!tk && ch>bv){bv=ch;best=e;} }
    id_[kk]=best; ww[kk]=sg[best];
  }
  int Ke=Ksel;
  if(topp>0.0f && topp<1.0f){
    for(int a=1;a<Ksel;a++){ int ii=id_[a]; float wv=ww[a]; int b=a-1;
      while(b>=0 && ww[b]<wv){ ww[b+1]=ww[b]; id_[b+1]=id_[b]; b--; } ww[b+1]=wv; id_[b+1]=ii; }
    float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=ww[kk];
    float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=ww[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
  }
  keff[s]=Ke;
  if(normk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=ww[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) ww[kk]/=sm; }
  for(int kk=0;kk<Ke;kk++) ww[kk]*=rscale;
}
// parallel replica of r_top8's selection on ONE SIMDGROUP per row instead of one serial
// thread (bench/kernels @ 27bfe83: serial r_top8 measured 0.465 ms/layer, ~55% of the
// layer CB; this replica ~93x faster with exactly matching output). EXACT-MATCH is the
// contract: each lane owns ceil(E/32) contiguous experts (blocked) and keeps a taken
// bitmask; per selection step: lane-local strict-'>' ascending max (lowest index wins
// within a lane, matching the serial ascending scan), then a shuffle-down argmax
// reduction where ties prefer the LOWER index — together exactly the serial kernel's
// first-max-wins order. The topp/normk/rscale tail is the serial code verbatim on lane 0
// (same ops, same order => bitwise-identical results; metal-test enforces this with
// memcmp). Contract: E<=256 (ch[8]/taken mask sizing: ceil(E/32)<=8) — the defensive
// return below makes an out-of-contract dispatch a visible no-op (idx/w/keff untouched),
// never an OOB write; both call sites (coli_metal_layer_decode's dispatch and the
// standalone coli_metal_rtop8 runner) additionally gate on E<=256 in host code before
// selecting this pipeline at all, so the return here is defense-in-depth, not the only
// guard. Sentinel-per-lane design (ch[j]=-1e30f for e>=E) makes non-multiple-of-32 E
// and small E correct without special-casing — validated for E=24, E=168 (REAP
// expert-pruned packages, see the upstream feature-request thread) and E=256 by metal-test.
// ASSUMES SIMD width 32 (shuffle offsets 16..1, 32-thread threadgroup per row): enforced
// at init — coli_metal_init clears g_rtop8_width_ok (and therefore both call sites' use
// of this pipeline) if threadExecutionWidth != 32.
kernel void r_top8_par(device const float* sig [[buffer(0)]], device const float* bias [[buffer(1)]],
                       device int* idx [[buffer(2)]], device float* w [[buffer(3)]],
                       device int* keff [[buffer(4)]], constant int& E [[buffer(5)]],
                       constant int& K [[buffer(6)]], constant int& Ksel [[buffer(7)]],
                       constant float& topp [[buffer(8)]], constant int& normk [[buffer(9)]],
                       constant float& rscale [[buffer(10)]],
                       uint s [[threadgroup_position_in_grid]],
                       uint slane [[thread_index_in_simdgroup]]) {
  if(E>256) return;
  device const float* sg=sig+(long)s*E;
  device int* id_=idx+(long)s*K; device float* ww=w+(long)s*K;
  int per=(E+31)/32, base=(int)slane*per;
  float ch[8]; uint taken=0u;
  for(int j=0;j<per;j++){ int e=base+j; ch[j]=(e<E)?sg[e]+bias[e]:-1e30f; }
  for(int kk=0;kk<Ksel;kk++){
    float bv=-1e30f; int bi=0x7FFFFFFF;
    for(int j=0;j<per;j++) if(!(taken&(1u<<j)) && ch[j]>bv){ bv=ch[j]; bi=base+j; }
    for(uint off=16;off>0;off>>=1){
      float ov=simd_shuffle_down(bv,off); int oi=simd_shuffle_down(bi,off);
      if(ov>bv || (ov==bv && oi<bi)){ bv=ov; bi=oi; }
    }
    bv=simd_broadcast(bv,0); bi=simd_broadcast(bi,0);
    if(bi>=base && bi<base+per) taken|=1u<<(bi-base);
    if(slane==0){ id_[kk]=bi; ww[kk]=sg[bi]; }
  }
  if(slane!=0) return;
  int Ke=Ksel;
  if(topp>0.0f && topp<1.0f){
    for(int a=1;a<Ksel;a++){ int ii=id_[a]; float wv=ww[a]; int b=a-1;
      while(b>=0 && ww[b]<wv){ ww[b+1]=ww[b]; id_[b+1]=id_[b]; b--; } ww[b+1]=wv; id_[b+1]=ii; }
    float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=ww[kk];
    float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=ww[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
  }
  keff[s]=Ke;
  if(normk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=ww[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) ww[kk]/=sm; }
  for(int kk=0;kk<Ke;kk++) ww[kk]*=rscale;
}

// ===== KV cache kernels =====
// per-row rope_interleave on a single qk_rope vector (no heads). grid = S*(qk_rope/2).
kernel void kv_rope(device float* R [[buffer(0)]], constant int& pos_base [[buffer(1)]],
                    constant int& qk_rope [[buffer(2)]], constant float& theta [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
  int S=gid/(qk_rope/2); int j=gid%(qk_rope/2); int s=S; int hl=qk_rope/2;
  int pos=pos_base+s;
  device float* rv=R+(long)s*qk_rope;
  float inv=pow(theta, -2.0f*j/(float)qk_rope); float ang=pos*inv, cs=cos(ang), sn=sin(ang);
  // Read original pair BEFORE any write (threadgroup-local copy to avoid hazard)
  thread float a=rv[2*j], b=rv[2*j+1];
  // Write to both halves — may read from positions we'll also write, but we already have a/b
  rv[j]     = a*cs - b*sn;
  rv[hl+j]  = b*cs + a*sn;
}
// clear range: zero rows [from, to) [from,to) of Lc/Rc. grid = 2*nrows*maxD.
kernel void kv_clear(device float* Lc [[buffer(0)]], device float* Rc [[buffer(1)]],
                     constant int& from [[buffer(2)]], constant int& to [[buffer(3)]],
                     constant int& kv_lora [[buffer(4)]], constant int& qk_rope [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
  int nr=to-from; int sel=gid%(2*nr), row=gid/(2*nr);
   int s= row; if(s>=nr) return;
   if(sel < nr){ Lc[(long)(from+s)*kv_lora+gid%(kv_lora)]=0.f; }
   else        { Rc[(long)(from+s)*qk_rope+gid%(qk_rope)]=0.f; }
}

// ===== KDA Conv1d + SiLU kernel =====
// Each thread (0..P-1) handles one dimension of one projection (q, k, or v).
//  conv_win[d*K]   — stateful shift register [P*K], oldest first, newest at K-1
//  vec[d]          — raw projected value (saved to window[K-1], overwritten with SiLU result)
//  taps[d*K]       — constant convolution taps [P*K]
kernel void kda_conv_silu(
    device float* conv_win  [[buffer(0)]],
    device float* vec       [[buffer(1)]],
    device const float* taps [[buffer(2)]],
    constant int& P         [[buffer(3)]],
    constant int& K         [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  int d = (int)gid;
  if (d >= P) return;
  long base = (long)d * K;

  // Shift window left by 1 and store new value at position K-1
  for (int j = 0; j < K - 1; j++) {
    conv_win[base + j] = conv_win[base + j + 1];
  }
  conv_win[base + K - 1] = vec[d];

  // Compute dot product of taps and window
  float acc = 0.f;
  for (int j = 0; j < K; j++) {
    acc += taps[base + j] * conv_win[base + j];
  }

  // Apply SiLU: x / (1 + exp(-x))
  vec[d] = acc / (1.f + exp(-acc));
}

// ===== KDA L2 normalization kernel =====
// Each thread (0..H-1) normalizes one head of Q and K in-place.
//  q[h*hd:(h+1)*hd] *= 1/sqrt(sum(q^2) + eps) * qscale
//  k[h*hd:(h+1)*hd] *= 1/sqrt(sum(k^2) + eps)
kernel void kda_l2_norm(
    device float* q       [[buffer(0)]],
    device float* k       [[buffer(1)]],
    constant int& H       [[buffer(2)]],
    constant int& hd      [[buffer(3)]],
    constant float& qscale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  int h = (int)gid;
  if (h >= H) return;

  long base = (long)h * hd;
  float sq = 0.f, sk = 0.f;
  for (int i = 0; i < hd; i++) {
    sq += q[base + i] * q[base + i];
    sk += k[base + i] * k[base + i];
  }
  sq = 1.f / sqrt(sq + 1e-6f);
  sk = 1.f / sqrt(sk + 1e-6f);
  float qs = sq * qscale;
  for (int i = 0; i < hd; i++) {
    q[base + i] *= qs;
    k[base + i] *= sk;
  }
}

// ===== KDA state recurrence kernel =====
// Each thread handles one (h, i) element of output space [H, hd].
// Two-pass sweep: (1) decay S by alpha, accumulate kS = S^T * kn,
//   then vt = (vh - kS) * beta; (2) expand S += kn[kk]*vt[vv],
//   merge oh[i] += qn[kk] * S[kk][i].
kernel void kda_state(
    device float* S      [[buffer(0)]],     // [H, hd, hd] in-place state
    device const float* qn     [[buffer(1)]],   // [H, hd] L2-normalized, gated Q
    device const float* kn     [[buffer(2)]],   // [H, hd] L2-normalized K
    device const float* vh     [[buffer(3)]],   // [H, hd] gated V
    device const float* alpha  [[buffer(4)]],   // [H, hd] per-head decay factors
    device const float* beta   [[buffer(5)]],   // [H] mixing factor
    device float* oh     [[buffer(6)]],     // [H, hd] output accumulator
    constant int& H      [[buffer(7)]],
    constant int& hd     [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  int th = gid / hd;
  int i = gid % hd;
  if (th >= H) return;
  long hS = (long)th * hd * hd;
  long hq = (long)th * hd;
  const device float* qn_h = qn + hq;
  const device float* kn_h = kn + hq;
  const device float* vh_h = vh + hq;
  const device float* alpha_h = alpha + hq;
  // Pass 1: decay + kS accumulate for this output column
  float kiS = 0.f;
  for (int kk = 0; kk < hd; kk++) {
    long row = hS + (long)kk * hd;
    float al = alpha_h[kk];
    S[row + i] *= al;
    kiS += kn_h[kk] * S[row + i];
  }
  float vi = (vh_h[i] - kiS) * beta[th];
  // Pass 2: expand + merge
  float oi = 0.f;
  for (int kk = 0; kk < hd; kk++) {
    long row = hS + (long)kk * hd;
    float kv = kn_h[kk];
    S[row + i] += kv * vi;
    oi += qn_h[kk] * S[row + i];
  }
  oh[hq + i] = oi;
}
)METAL";

struct ColiMetalTensor {
  id<MTLBuffer> w;      // weights (wrapped, zero-copy when page-aligned)
  id<MTLBuffer> s;      // scales
  int fmt, I, O; size_t wbytes;
};

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_gemv, g_moe_gemv, g_moe_silu, g_moe_fwht;

// fmt=6: sign-bit buffers for the GPU FWHT, one per tile size, cached forever (a
// handful of sizes). The xorshift64* draw replicates quant.h e8_signs exactly —
// the metal-test fmt=6 oracle compares end-to-end against quant.h, so any drift
// between the two copies fails the build.
static void e8_signs_local(uint8_t *bits, int n) {
  uint64_t s = 417u + (uint64_t)n;
  for (int i = 0; i < (n+7)/8; i++) {
    s ^= s>>12; s ^= s<<25; s ^= s>>27;
    bits[i] = (uint8_t)((s*2685821657736338717ULL)>>56);
  }
}
static id<MTLBuffer> fwht_signs(int n) {
  static std::mutex mtx; static std::vector<std::pair<int, id<MTLBuffer>>> cache;
  std::lock_guard<std::mutex> lk(mtx);
  for (auto &p : cache) if (p.first == n) return p.second;
  std::vector<uint8_t> bits((n+7)/8);
  e8_signs_local(bits.data(), n);
  id<MTLBuffer> b = [g_dev newBufferWithBytes:bits.data() length:bits.size()
                                      options:MTLResourceStorageModeShared];
  cache.push_back({n, b});
  return b;
}
static id<MTLComputePipelineState> g_a_rms, g_a_rope, g_a_copy, g_a_qabs, g_a_score, g_a_smax, g_a_clat, g_a_ctx;
static id<MTLComputePipelineState> g_a_add, g_r_router, g_r_top8, g_r_top8p;
static id<MTLComputePipelineState> g_kv_rope, g_kv_clear;
static id<MTLComputePipelineState> g_kda_state;
static id<MTLComputePipelineState> g_kda_conv_silu, g_kda_l2_norm;
static int g_rtop8_par = 1;      // COLI_RTOP8 (default ON); COLI_RTOP8=0 opts out to the
                                  // serial kernel — see coli_metal_init.
static int g_rtop8_width_ok = 1; // hardware fact, independent of the policy gate above:
                                  // false if this device's threadExecutionWidth != 32.
                                  // Consulted by BOTH the engine dispatch site and the
                                  // standalone coli_metal_rtop8 runner, so no caller can
                                  // reach r_top8_par's 32-lane reduction on an unsafe
                                  // device even by explicitly requesting par=1.
static size_t g_tensor_count, g_tensor_bytes;
static uint64_t g_moe_ok, g_moe_fb, g_moe_experts;   // GPU blocks / CPU-fallback blocks / experts on GPU
static double g_t_setup, g_t_gpu, g_t_scatter, g_t_kernel;       // per-block time breakdown (seconds)
static const int TG = 128;
static MTLResourceOptions g_res_opts = MTLResourceStorageModeShared;   // COLI_METAL_UNTRACKED=1 adds HazardTrackingModeUntracked
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

// ---- E5 experiment: COLI_METAL_RESSET=1 -- one persistent MTLResidencySet attached to
// g_queue (macOS 15+) replaces moe_submit's per-command-buffer useResource: loop over
// resolved expert weight/scale slabs. Allocation is untouched (same newBufferWithBytesNoCopy
// wrap as stock); only residency bookkeeping moves off the dispatch hot path -- see
// docs/experiments/e5-metal-residency-set.md for why skipping useResource: there is safe (read-only, indirectly-referenced
// buffers only; residency sets don't do hazard tracking, but nothing here relied on it).
// g_resset_obj is a bare `id` (holds id<MTLResidencySet>) so the global's declared type
// carries no availability annotation -- the protocol name only appears inside
// @available(macOS 15.0, *) guards below, keeping -Wunguarded-availability clean.
// @available is a runtime guard only: the compiler still resolves MTLResidencySet and its
// selectors against the SDK headers, which don't declare them before macOS 15 (#596).
// COLI_HAS_RESSET is the compile-time floor (helpers become no-ops below it); the @available
// guards stay on top so a macOS 15 SDK build still runs right on macOS 14. -DCOLI_HAS_RESSET=0
// forces the fallback branch on a modern SDK.
#ifndef COLI_HAS_RESSET
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
#define COLI_HAS_RESSET 1
#else
#define COLI_HAS_RESSET 0
#endif
#endif
static id g_resset_obj;
static bool g_resset_enabled;   // COLI_METAL_RESSET=1, macOS 15 SDK+OS, and creation succeeded
static bool g_resset_dirty;     // addAllocation: calls pending commit; g_resset_mtx-guarded
// Set mutations + dirty flag get their OWN mutex, never held together with g_slab_mtx: no
// live Metal call may run under the slab lock the parallel OMP loader threads contend on
// (E4's audit round 2 found exactly that shape -- mutex over a live Metal call -- as the
// leading suspect for its +12s expert-disk regression). g_slab_mtx keeps guarding g_slabs
// bookkeeping only, exactly as on stock.
static std::mutex g_resset_mtx;
static double g_t_resset_flush;   // sec committing pending adds in moe_submit (gate on only)

// Add a just-wrapped buffer to the set; commit deferred (an OMP loader burst batches into
// one commit at the next moe_submit instead of one per slab). Called by coli_metal_register
// after it drops g_slab_mtx but before it returns -- and the engine cannot dispatch an
// expert before the load that registers its slab returns, so any slab a given moe_submit
// can resolve() was added (and marked dirty) under g_resset_mtx strictly before that
// moe_submit's resset_flush() acquired the same mutex: the flush covers it. The slab-table
// ordering itself (register-before-resolve) is unchanged and stays under g_slab_mtx.
// Cost lands in the caller's existing expert-load accounting (t_ewait window in colibri.c);
// no separate counter for the add/remove side.
static void resset_add(id<MTLBuffer> b) {
  if (!g_resset_enabled) return;
#if COLI_HAS_RESSET
  std::lock_guard<std::mutex> lk(g_resset_mtx);
  if (@available(macOS 15.0, *)) { [(id<MTLResidencySet>)g_resset_obj addAllocation:b]; g_resset_dirty = true; }
#else
  (void)b;
#endif
}
// Remove + commit immediately, NOT deferred: the caller frees the underlying host memory
// right after coli_metal_unregister returns, so the removal must be applied before that --
// an uncommitted-but-still-resident allocation pointing at freed memory is a use-after-free
// risk the GPU could act on. Also runs outside g_slab_mtx (see g_resset_mtx above).
static void resset_remove(id<MTLBuffer> b) {
  if (!g_resset_enabled) return;
#if COLI_HAS_RESSET
  std::lock_guard<std::mutex> lk(g_resset_mtx);
  if (@available(macOS 15.0, *)) {
    id<MTLResidencySet> rs = (id<MTLResidencySet>)g_resset_obj;
    [rs removeAllocation:b]; [rs commit];
  }
  g_resset_dirty = false;   // commit above also flushes any pending adds
#else
  (void)b;
#endif
}
// Flush pending adds before moe_submit relies on the set alone for residency -- the only
// caller that skips per-buffer useResource: (see moe_submit below). Takes g_resset_mtx
// only, never g_slab_mtx; the happens-before argument lives at resset_add above.
static void resset_flush() {
  if (!g_resset_enabled) return;
#if COLI_HAS_RESSET
  std::lock_guard<std::mutex> lk(g_resset_mtx);
  if (!g_resset_dirty) return;
  if (@available(macOS 15.0, *)) { [(id<MTLResidencySet>)g_resset_obj commit]; }
  g_resset_dirty = false;
#endif
}
// Harness visibility for the flush cost, which sits OUTSIDE the moe_times setup/gpu
// breakdown (timed around resset_flush in moe_submit, before ts_start). Returns whether
// the set is active so colibri.c prints the METAL-RESSET line only when the gate is on --
// stock output stays byte-identical.
extern "C" int coli_metal_resset_stats(double *flush_s) {
  if (flush_s) *flush_s = g_t_resset_flush;
  return g_resset_enabled ? 1 : 0;
}

// Persistent scratch buffers (grow-only) for the MoE pipeline.
static id<MTLBuffer> g_gg, g_uu, g_hh, g_xg; static size_t g_gg_cap, g_uu_cap, g_hh_cap, g_xg_cap;
static id<MTLBuffer> ensure(id<MTLBuffer> b, size_t *cap, size_t need) {
  if (b && *cap >= need) return b;
  *cap = need; return [g_dev newBufferWithLength:need options:g_res_opts];
}

static size_t fmt_bytes(int fmt, int I, int O) {
  if (fmt == 1) return (size_t)O * I;
  if (fmt == 2) return (size_t)O * ((I+1)/2);
  if (fmt == 3) return (size_t)O * ((I+3)/4);
  if (fmt == 4) return (size_t)O * ((I+1)/2);   // grouped int4: identical packed-nibble layout to fmt=2
  if (fmt == 8) return (size_t)O * I;           // fp8 e4m3: one raw byte/element, same as fmt=1
  return (size_t)O * I * sizeof(float);
}
// Grouped-int4 (fmt=4) scale-array size: one f32 per gsz-element group, per row -> O*ceil(I/gsz).
// fp8 (fmt=8) scale-array size: one f32 per 128x128 BLOCK -> ceil(O/128)*ceil(I/128) (2D,
// not per-row -- quant.h isn't included here, so the ceil-div is inlined rather than sharing
// colibri.c's qt_scale_bytes/quant.h's fp8_nblk). The block is a fixed 128x128, so gs is
// ignored for fmt==8. f32 is this build's implemented scale
// ENCODING for fmt=8 (see quant.h/colibri.c) -- this file has no reason to know that a
// UE8M0 encoding exists at all: qt_resolve_fmt refuses it on the CPU read path before any
// tensor in that encoding could ever reach this Metal-side sizing helper.
static size_t fmt_scale_bytes(int fmt, int I, int O, int gs) {
  if (fmt == 4) return (size_t)O * ((I + gs - 1) / gs) * sizeof(float);
  if (fmt == 6) return (size_t)O * ((I + gs - 1) / gs) * 2; // e8: 2-byte group scales
  if (fmt == 8) return (size_t)((O + 127) / 128) * (size_t)((I + 127) / 128) * sizeof(float); // fp8 block scales
  return (size_t)O * sizeof(float); // per-row scale, fmt 0/1/2/3 (dev catch-all; #790 rebase must keep this)
}

// Wrap host memory zero-copy if page-aligned, else copy into a shared buffer.
static id<MTLBuffer> wrap(const void *p, size_t n) {
  size_t pg = 16384; // Apple Silicon page
  if (((uintptr_t)p % pg) == 0 && (n % pg) == 0)
    return [g_dev newBufferWithBytesNoCopy:(void*)p length:n options:MTLResourceStorageModeShared deallocator:nil];
  return [g_dev newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
}

// Wrap-once cache for host buffers that live for the whole model (KDA state, conv windows,
// conv taps). These are re-used every token, so wrapping them on each call churns thousands
// of MTLBuffer objects. Keyed by host pointer, which is safe ONLY because these allocations
// are never freed or resized during inference. Do NOT use for per-forward transient buffers
// (qt/kt/tv/alpha/beta/oh) — their pointers get freed and reused.
static std::unordered_map<const void*, id<MTLBuffer>> g_wrap_cache;
static std::mutex g_wrap_cache_mu;
static id<MTLBuffer> wrap_persistent(const void *p, size_t n) {
  std::lock_guard<std::mutex> lk(g_wrap_cache_mu);
  auto it = g_wrap_cache.find(p);
  if (it != g_wrap_cache.end()) return it->second;
  id<MTLBuffer> b = wrap(p, n);
  if (b) g_wrap_cache[p] = b;
  return b;
}

extern "C" int coli_metal_init(void) {
  if (g_dev) return 1;
  if (getenv("COLI_METAL_UNTRACKED") && atoi(getenv("COLI_METAL_UNTRACKED")))
    g_res_opts = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
  { const char *e = getenv("COLI_RTOP8");           // default ON; COLI_RTOP8=0 opts out
    if (e && atoi(e) == 0) g_rtop8_par = 0; }
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
    g_moe_fwht = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_fwht"] error:&err];
    auto P=[&](const char*n){ return [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:&err]; };
    g_a_rms=P("a_rmsnorm"); g_a_rope=P("a_rope"); g_a_copy=P("a_copy");
    g_a_qabs=P("a_qabs"); g_a_score=P("a_score"); g_a_smax=P("a_smax"); g_a_clat=P("a_clat"); g_a_ctx=P("a_ctx");
    g_a_add=P("a_add"); g_r_router=P("r_router"); g_r_top8=P("r_top8"); g_r_top8p=P("r_top8_par");
    g_kv_rope=P("kv_rope"); g_kv_clear=P("kv_clear");
    g_kda_state=P("kda_state");
    g_kda_conv_silu=P("kda_conv_silu"); g_kda_l2_norm=P("kda_l2_norm");
    if(!g_a_add||!g_r_router||!g_r_top8||!g_r_top8p||!g_kv_rope||!g_kv_clear||!g_kda_state||!g_kda_conv_silu||!g_kda_l2_norm){ fprintf(stderr,"[metal] tail pipelines failed\n"); g_dev=nil; return 0; }
    // r_top8_par's reduction hardcodes SIMD width 32 (shuffle-down offsets 16..1, one
    // 32-thread threadgroup per row). True on all Apple Silicon shipped to date, but a
    // non-32-width device would reduce wrongly AND race multiple lane-0 writers, so this
    // is a hard safety fact (g_rtop8_width_ok), not just a policy default: it gates BOTH
    // the engine dispatch site and the standalone coli_metal_rtop8 runner (degrade-to-safe,
    // same pattern as the pool/ring fallbacks elsewhere) — no caller can opt back into an
    // unsafe reduction on such a device, even by explicitly requesting par=1.
    if ([g_r_top8p threadExecutionWidth] != 32) {
      g_rtop8_width_ok = 0;
      if (g_rtop8_par)
        fprintf(stderr, "[metal] COLI_RTOP8 parallel top-8 disabled: threadExecutionWidth=%lu "
                        "!= 32 (r_top8_par's reduction assumes 32-lane simdgroups) — serial "
                        "r_top8 in use\n", (unsigned long)[g_r_top8p threadExecutionWidth]);
      g_rtop8_par = 0;
    }
    if (!g_gemv || !g_moe_gemv || !g_moe_silu || !g_moe_fwht || !g_a_rms || !g_a_rope || !g_a_copy ||
        !g_a_qabs || !g_a_score || !g_a_smax || !g_a_clat || !g_a_ctx) {
      fprintf(stderr, "[metal] pipeline failed\n"); g_dev = nil; return 0; }
    // E5 experiment: COLI_METAL_RESSET=1 -- see g_resset_obj comment above.
    if (getenv("COLI_METAL_RESSET") && atoi(getenv("COLI_METAL_RESSET"))) {
#if COLI_HAS_RESSET
      if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 4096;   // hint only (internal array presize), not a hard limit
        NSError *rerr = nil;
        id<MTLResidencySet> rs = [g_dev newResidencySetWithDescriptor:rd error:&rerr];
        if (rs) {
          [g_queue addResidencySet:rs];
          g_resset_obj = rs; g_resset_enabled = true;
          fprintf(stderr, "[METAL] residency-set: on (macOS 15+, moe_submit skips per-buffer useResource:)\n");
        } else {
          fprintf(stderr, "[METAL] residency-set create failed: %s -- stock per-CB residency path\n",
                  rerr ? [[rerr localizedDescription] UTF8String] : "?");
        }
      } else {
        fprintf(stderr, "[METAL] COLI_METAL_RESSET=1 requested but OS < macOS 15 -- stock per-CB residency path\n");
      }
#else
      fprintf(stderr, "[METAL] COLI_METAL_RESSET=1 requested but this binary was built against a "
                      "macOS SDK < 15 (MTLResidencySet unavailable) -- stock per-CB residency path\n");
#endif
    }
  }
  return 1;
}

extern "C" void coli_metal_register(void *base, size_t len) {
  if (!g_dev || !base) return;
  id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:base length:len
                              options:g_res_opts deallocator:nil];
  if (!b) return;
  id<MTLBuffer> old = nil;   // E5: replaced wrapper on re-register of a live base (defensive)
  {
    std::lock_guard<std::mutex> lk(g_slab_mtx);   // called from parallel expert_load threads
    bool found = false;
    for (auto &s : g_slabs) if (s.base == base) { old = s.buf; s.len = len; s.buf = b; found = true; break; }
    if (!found) g_slabs.push_back({base, len, b});
  }
  // E5, outside g_slab_mtx (no Metal call under the slab lock), before returning. Invariant
  // defended: set membership mirrors g_slabs exactly -- a re-register of a live base must
  // drop the replaced wrapper from the set (ARC releases our reference, but the set retains
  // it and keeps its pages resident forever) before adding the new one. No in-tree caller
  // re-registers a live base today; defensive.
  if (old && old != b) resset_remove(old);
  if (old != b) resset_add(b);
}
extern "C" void coli_metal_unregister(void *base) {
  id<MTLBuffer> b = nil;
  {
    std::lock_guard<std::mutex> lk(g_slab_mtx);
    for (size_t i=0;i<g_slabs.size();i++) if (g_slabs[i].base==base) {
      b = g_slabs[i].buf; g_slabs[i].buf=nil; g_slabs.erase(g_slabs.begin()+i); break; }
  }
  if (b) resset_remove(b);   // E5: outside g_slab_mtx; commits before the caller frees base
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

extern "C" void coli_metal_shutdown(void) {
  coli_metal_spin_stop();
#if COLI_HAS_RESSET
  if (g_resset_enabled) {
    if (@available(macOS 15.0, *)) { [g_queue removeResidencySet:(id<MTLResidencySet>)g_resset_obj]; }
  }
#endif
  g_resset_obj=nil; g_resset_enabled=false; g_resset_dirty=false;
  g_gemv=nil; g_queue=nil; g_dev=nil; g_tensor_count=g_tensor_bytes=0;
}
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
                                 int fmt, int S, int I, int O, int gs) {
  /* fmt==8 (fp8 passthrough) is an explicit allow-list entry, not folded into the 0..4
   * contiguous range check below: it is not adjacent to it. */
  if (!g_dev || fmt < 0 || (fmt > 4 && fmt != 8)) return 0;
  if(!weights||!x||!y) return 0;
  @autoreleasepool {
    ColiMetalTensor *t = *tp;
    if (!t) {
      t = new ColiMetalTensor();
      t->fmt = fmt; t->I = I; t->O = O; t->wbytes = fmt_bytes(fmt, I, O);
      t->w = wrap(weights, t->wbytes);
      { size_t sb=fmt_scale_bytes(fmt, I, O, gs); t->s=sb?wrap(scales,sb):0; }
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
    int NT=S*O;
    [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5];
    [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
    [e setBytes:&NT length:4 atIndex:8]; [e setBytes:&gs length:4 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(y, [by contents], (size_t)S*O*sizeof(float));
  }
  return 1;
}

// ---- fused decode attention scratch (GLM-5.2 dims) ----
enum { AH=6144, AHEADS=64, AQLORA=2048, AKVL=512, AROPE=64, AVH=256, AQH=256, ANOPE=192, AROWSH=448, AHQH=AHEADS*AQH, AHVH=AHEADS*AVH, AMAXS=4 };
static id<MTLBuffer> ax_,aqr_,aqf_,acomp_,aqabs_,ascore_,aclat_,actx_,aout_,aqaln_,akvaln_; static size_t ascore_cap;
static size_t ax_cap,aqr_cap,aqf_cap,acomp_cap,aqabs_cap,aclat_cap,actx_cap,aout_cap;
static id<MTLBuffer> axr_,anrm_,ash1_,ash2_,ashout_,asig_,aidx_,aw_,akeff_;   // full-layer tail (AMAXS-sized)
static void attn_scratch_init(){
  if(ax_) return;
  auto L=[&](size_t n){ return [g_dev newBufferWithLength:n*AMAXS options:g_res_opts]; };
  axr_=L(AH*4); anrm_=L(AH*4); ash1_=L(2048*4); ash2_=L(2048*4); ashout_=L(AH*4);
  asig_=L(256*4); aidx_=L(8*4); aw_=L(8*4); akeff_=L(4);
  aqaln_=[g_dev newBufferWithLength:AQLORA*4 options:g_res_opts];
  akvaln_=[g_dev newBufferWithLength:AKVL*4 options:g_res_opts];
}
static void attn_scratch_reserve(int S, int T){
  attn_scratch_init();
  ax_=ensure(ax_,&ax_cap,(size_t)S*AH*4);
  aqr_=ensure(aqr_,&aqr_cap,(size_t)S*AQLORA*4);
  aqf_=ensure(aqf_,&aqf_cap,(size_t)S*AHQH*4);
  acomp_=ensure(acomp_,&acomp_cap,(size_t)S*(AKVL+AROPE)*4);
  aqabs_=ensure(aqabs_,&aqabs_cap,(size_t)S*AHEADS*AKVL*4);
  ascore_=ensure(ascore_,&ascore_cap,(size_t)S*AHEADS*T*4);
  aclat_=ensure(aclat_,&aclat_cap,(size_t)S*AHEADS*AKVL*4);
  actx_=ensure(actx_,&actx_cap,(size_t)S*AHVH*4);
  aout_=ensure(aout_,&aout_cap,(size_t)S*AH*4);
}
// y[S,O] = quantized-weight(w) applied to xin[S,I]. Weights are registered (page-aligned,
// zero-copy) at model load; resolve to (buffer,offset). Returns false to fall back to CPU.
// gs: fmt=4 group size (ignored for fmt!=4; callers pass QT.gs, which is 0 for non-grouped
// tensors -- harmless, since the shader only reads it when fmt==4).
static bool bind_gemv(id<MTLComputeCommandEncoder> e, const void* w, const float* s, int fmt, int gs, int I, int O,
                      id<MTLBuffer> xin, id<MTLBuffer> yout, int S){
  uint64_t wa=0,sa=0; id<MTLBuffer> wb=resolve(w,&wa); id<MTLBuffer> sb=resolve(s,&sa);
  if(!wb||!sb) return false;
  size_t woff=wa-(uint64_t)[wb gpuAddress], soff=sa-(uint64_t)[sb gpuAddress];
  [e useResource:wb usage:MTLResourceUsageRead]; [e useResource:sb usage:MTLResourceUsageRead];
  [e setComputePipelineState:g_gemv];
  [e setBuffer:wb offset:woff atIndex:0]; [e setBuffer:sb offset:soff atIndex:1];
  [e setBuffer:xin offset:0 atIndex:2]; [e setBuffer:yout offset:0 atIndex:3];
  int NT=S*O;
  [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5]; [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
  [e setBytes:&NT length:4 atIndex:8]; [e setBytes:&gs length:4 atIndex:9];
  [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  return true;
}

// Weight-pointer bundle for one layer's attention (+optional layer tail). All pointers
// must be inside registered allocations. *_gs: fmt=4 group size for the corresponding
// weight (0 if that weight isn't grouped). kv_b never flows through bind_gemv -- a_qabs/
// a_ctx dequantize it inline via a_deqrow, which is fmt/gs-aware (fmt=2 per-row, fmt=4
// grouped along A_KVL); kvb_gs is that group size (0 for fmt=2).
typedef struct {
  const void *qa_w; const float *qa_s; int qa_fmt; int qa_gs; const float *qa_ln;
  const void *qb_w; const float *qb_s; int qb_fmt; int qb_gs;
  const void *kva_w; const float *kva_s; int kva_fmt; int kva_gs; const float *kva_ln;
  const void *kvb_w; const float *kvb_s; int kvb_fmt; int kvb_gs;
  const void *o_w;  const float *o_s;  int o_fmt; int o_gs;
} AttnW;

// Phase 1: projections (qa, kva, qb, RMS, RoPE, qabs) for all S rows.
// Reads ax_[S*AH], writes aqr_[S*AQLORA], acomp_[S*(AKVL+AROPE)], aqf_[S*AHQH], aqabs_[S*AHEADS*AKVL].
// Also writes Lc (keys) and Rc (rope keys) into the KV cache at pos_base.
static bool encode_attn_projections(id<MTLComputeCommandEncoder> e, const AttnW *W,
                             id<MTLBuffer> Lb, size_t loff, id<MTLBuffer> Rb, size_t roff,
                             id<MTLBuffer> kvbW, size_t kvbwoff, id<MTLBuffer> kvbS, size_t kvbsoff,
                             int S, int pos_base, float eps, float theta) {
    memcpy([aqaln_ contents],W->qa_ln,AQLORA*4); memcpy([akvaln_ contents],W->kva_ln,AKVL*4);
    size_t Loff=loff+(size_t)pos_base*AKVL*4, Roff=roff+(size_t)pos_base*AROPE*4;
    auto BAR=[&]{ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
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
    bind_gemv(e,W->qa_w,W->qa_s,W->qa_fmt,W->qa_gs,AH,AQLORA,ax_,aqr_,S);
    bind_gemv(e,W->kva_w,W->kva_s,W->kva_fmt,W->kva_gs,AH,AKVL+AROPE,ax_,acomp_,S); BAR();
    rms(aqr_,0,aqaln_,AQLORA,S); cpy(0,Lb,Loff,AKVL); cpy(AKVL,Rb,Roff,AROPE); BAR();
    bind_gemv(e,W->qb_w,W->qb_s,W->qb_fmt,W->qb_gs,AQLORA,AHQH,aqr_,aqf_,S); rms(Lb,Loff,akvaln_,AKVL,S); rope(Rb,Roff,0,AROPE,0,1); BAR();
    rope(aqf_,0,ANOPE,AHQH,AQH,AHEADS); BAR();
    [e setComputePipelineState:g_a_qabs]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aqf_ offset:0 atIndex:2]; [e setBuffer:aqabs_ offset:0 atIndex:3];
    [e setBytes:&W->kvb_fmt length:4 atIndex:4]; [e setBytes:&W->kvb_gs length:4 atIndex:5];
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    return true;
}
// Phase 2: chunked attention core for one chunk of ch rows (starting at r0 within the S-row batch).
// Reads aqabs_[r0*AHEADS*AKVL], aqf_[r0*AHQH], Lb, Rb, kvbW, kvbS.
// Writes actx_[r0*AHVH] (accumulated into the S-row ctx buffer).
// Intermediate: ascore_[ch*AHEADS*T], aclat_[ch*AHEADS*AKVL] (per chunk, ephemeral).
// T = total keys in the KV cache (pos_base_global + S_total). pos_base here = pos_base_global + r0
// so that the score kernel's per-row causal length (pos - t + 1) is correct for this chunk.
static bool encode_attn_core_chunk(id<MTLComputeCommandEncoder> e,
                             id<MTLBuffer> Lb, size_t loff, id<MTLBuffer> Rb, size_t roff,
                             id<MTLBuffer> kvbW, size_t kvbwoff, id<MTLBuffer> kvbS, size_t kvbsoff,
                             int r0, int ch, int T, int pos_base, float ascale,
                             int kvb_fmt, int kvb_gs) {
    size_t qabs_off=(size_t)r0*AHEADS*AKVL*4, qf_off=(size_t)r0*AHQH*4, ctx_off=(size_t)r0*AHVH*4;
    int PB=pos_base;
    auto BAR=[&]{ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    [e setComputePipelineState:g_a_score]; [e setBuffer:aqabs_ offset:qabs_off atIndex:0];
    [e setBuffer:Lb offset:loff atIndex:1]; [e setBuffer:Rb offset:roff atIndex:2]; [e setBuffer:aqf_ offset:qf_off atIndex:3];
    [e setBuffer:ascore_ offset:0 atIndex:4]; [e setBytes:&T length:4 atIndex:5]; [e setBytes:&ascale length:4 atIndex:6]; [e setBytes:&PB length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake((size_t)ch*AHEADS*T,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_smax]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBytes:&T length:4 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((size_t)ch*AHEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_clat]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBuffer:Lb offset:loff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBytes:&T length:4 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)ch*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_ctx]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBuffer:actx_ offset:ctx_off atIndex:3];
    [e setBytes:&kvb_fmt length:4 atIndex:4]; [e setBytes:&kvb_gs length:4 atIndex:5];
    [e dispatchThreads:MTLSizeMake((size_t)ch*AHEADS*AVH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    return true;
}
// Old monolithic encode_attention: projections + core + output GEMV in one call.
// Used by the S<=4 decode path and as a building block for larger S.
static bool encode_attention(id<MTLComputeCommandEncoder> e, const AttnW *W,
                             id<MTLBuffer> Lb, size_t loff, id<MTLBuffer> Rb, size_t roff,
                             id<MTLBuffer> kvbW, size_t kvbwoff, id<MTLBuffer> kvbS, size_t kvbsoff,
                             int S, int T, int pos_base, float eps, float theta, float ascale) {
    if(!encode_attn_projections(e,W,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,S,pos_base,eps,theta)) return false;
    if(!encode_attn_core_chunk(e,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,0,S,T,pos_base,ascale,W->kvb_fmt,W->kvb_gs)) return false;
    bind_gemv(e,W->o_w,W->o_s,W->o_fmt,W->o_gs,AHVH,AH,actx_,aout_,S);
    return true;
}
// Resolve Lc/Rc + kv_b (+pre-check the projection weights). Returns false -> CPU fallback.
static bool resolve_attn(const AttnW *W, float *Lc, float *Rc,
                         id<MTLBuffer> *Lb, size_t *loff, id<MTLBuffer> *Rb, size_t *roff,
                         id<MTLBuffer> *kvbW, size_t *kvbwoff, id<MTLBuffer> *kvbS, size_t *kvbsoff) {
    uint64_t la=0,ra=0,kva=0,ksa=0;
    *Lb=resolve(Lc,&la); *Rb=resolve(Rc,&ra); *kvbW=resolve(W->kvb_w,&kva); *kvbS=resolve(W->kvb_s,&ksa);
    if(!*Lb||!*Rb||!*kvbW||!*kvbS) return false;
    uint64_t d; const void* ws[]={W->qa_w,W->qa_s,W->qb_w,W->qb_s,W->kva_w,W->kva_s,W->o_w,W->o_s};
    for(auto p:ws) if(!resolve(p,&d)) return false;
    *loff=la-(uint64_t)[*Lb gpuAddress]; *roff=ra-(uint64_t)[*Rb gpuAddress];
    *kvbwoff=kva-(uint64_t)[*kvbW gpuAddress]; *kvbsoff=ksa-(uint64_t)[*kvbS gpuAddress];
    return true;
}

extern "C" int coli_metal_attn_decode(const float* x,
    const void* qa_w,const float* qa_s,int qa_fmt,int qa_gs,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,int qb_gs,
    const void* kva_w,const float* kva_s,int kva_fmt,int kva_gs,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,int kvb_gs,
    const void* o_w,const float* o_s,int o_fmt,int o_gs,
    float* Lc,float* Rc,int S,int pos_base,int st0,float eps,float theta,float ascale,float* out){
  if(!g_dev) return 0;
  if(st0!=0 || S<1) return 0;     // partial-KV -> CPU (S no longer capped)
  int T=pos_base+S;
  @autoreleasepool {
    attn_scratch_init();
    AttnW W={qa_w,qa_s,qa_fmt,qa_gs,qa_ln,qb_w,qb_s,qb_fmt,qb_gs,kva_w,kva_s,kva_fmt,kva_gs,kva_ln,kvb_w,kvb_s,kvb_fmt,kvb_gs,o_w,o_s,o_fmt,o_gs};
    id<MTLBuffer> Lb,Rb,kvbW,kvbS; size_t loff,roff,kvbwoff,kvbsoff;
    if(!resolve_attn(&W,Lc,Rc,&Lb,&loff,&Rb,&roff,&kvbW,&kvbwoff,&kvbS,&kvbsoff)) return 0;

    // One command buffer: projections + attention core + output GEMV in a single encoder,
    // ordered by memory barriers — for both the S<=4 decode path and S>4 prefill. The earlier
    // three-command-buffer split (projections/core/output as separate commit+wait buffers)
    // corrupted cross-buffer state and forked greedy output from the first prefill token; doing
    // it in one encoder is token-exact vs the CPU absorbed path. Guard: a single a_score dispatch
    // is S*AHEADS*T threads — cap it under ~2^30 and fall back to CPU for giant prompts
    // (in-encoder row-chunking to restore GPU coverage above the cap is a follow-up).
    if((int64_t)S*AHEADS*T >= (1LL<<30)) return 0;   // too large for one dispatch -> CPU
    attn_scratch_reserve(S,T);
    memcpy([ax_ contents],x,(size_t)S*AH*4);
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:Lb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rb usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    if(!encode_attention(e,&W,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,S,T,pos_base,eps,theta,ascale)) return 0;
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] attn cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    g_attn_sched += [cb GPUStartTime]-[cb kernelStartTime]; g_attn_ksched += [cb kernelStartTime]-tc;
    memcpy(out,[aout_ contents],(size_t)S*AH*4);
  }
  return 1;
}

// Full decode layer on the GPU in ONE command buffer:
//   in_ln rmsnorm -> fused attention -> residual add (x updated) -> post_ln rmsnorm ->
//   shared expert (gate/up/silu/down) -> router (f32 matvec+sigmoid) -> exact top-K select.
// CPU keeps: expert resolve/disk loads + expert CBs + scatter (unchanged). Outputs:
// x (updated in place), nrm=post_ln(x) (expert input), sh_out (shared-expert output),
// idx/w/keff (routing). Returns 0 -> CPU fallback (whole layer falls back).
extern "C" int coli_metal_layer_decode(float *x,
    const float *in_ln, const float *post_ln,
    const void* qa_w,const float* qa_s,int qa_fmt,int qa_gs,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,int qb_gs,
    const void* kva_w,const float* kva_s,int kva_fmt,int kva_gs,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,int kvb_gs,
    const void* o_w,const float* o_s,int o_fmt,int o_gs,
    const void* shg_w,const float* shg_s,int shg_fmt,int shg_gs,
    const void* shu_w,const float* shu_s,int shu_fmt,int shu_gs,
    const void* shd_w,const float* shd_s,int shd_fmt,int shd_gs,
    const float *router_w, const float *router_bias,
    int E, int K, int Ksel, float topp, int normk, float rscale,
    float *Lc, float *Rc, int S, int pos_base, int st0,
    float eps, float theta, float ascale,
    float *inrm_out, float *nrm_out, float *sh_out, int *idx_out, float *w_out, int *keff_out) {
  if(!g_dev) return 0;
  if(st0!=0 || S<1 || S>AMAXS || E!=256 || K!=8) return 0;
  int T=pos_base+S; const int SI=2048;
  @autoreleasepool {
    attn_scratch_init();
    AttnW W={qa_w,qa_s,qa_fmt,qa_gs,qa_ln,qb_w,qb_s,qb_fmt,qb_gs,kva_w,kva_s,kva_fmt,kva_gs,kva_ln,kvb_w,kvb_s,kvb_fmt,kvb_gs,o_w,o_s,o_fmt,o_gs};
    id<MTLBuffer> Lb,Rb,kvbW,kvbS; size_t loff,roff,kvbwoff,kvbsoff;
    if(!resolve_attn(&W,Lc,Rc,&Lb,&loff,&Rb,&roff,&kvbW,&kvbwoff,&kvbS,&kvbsoff)) return 0;
    uint64_t ina=0,pna=0,rwa=0,rba=0,d;
    id<MTLBuffer> inB=resolve(in_ln,&ina), pnB=resolve(post_ln,&pna);
    id<MTLBuffer> rwB=resolve(router_w,&rwa), rbB=resolve(router_bias,&rba);
    if(!inB||!pnB||!rwB||!rbB) return 0;
    { const void* ws[]={shg_w,shg_s,shu_w,shu_s,shd_w,shd_s};
      for(auto p:ws) if(!resolve(p,&d)) return 0; }
    size_t inoff=ina-(uint64_t)[inB gpuAddress], pnoff=pna-(uint64_t)[pnB gpuAddress];
    size_t rwoff=rwa-(uint64_t)[rwB gpuAddress], rboff=rba-(uint64_t)[rbB gpuAddress];
    ascore_=ensure(ascore_,&ascore_cap,(size_t)S*AHEADS*T*4);
    memcpy([axr_ contents],x,(size_t)S*AH*4);

    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:Lb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rb usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    [e useResource:inB usage:MTLResourceUsageRead]; [e useResource:pnB usage:MTLResourceUsageRead];
    [e useResource:rwB usage:MTLResourceUsageRead]; [e useResource:rbB usage:MTLResourceUsageRead];
    auto BAR=[&]{ [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; };
    auto rmsw=[&](id<MTLBuffer> b,id<MTLBuffer> wb,size_t woff,int n,int nrows){ [e setComputePipelineState:g_a_rms];
      [e setBuffer:b offset:0 atIndex:0]; [e setBuffer:wb offset:woff atIndex:1]; [e setBytes:&n length:4 atIndex:2]; [e setBytes:&eps length:4 atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake(nrows,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
    auto copyrow=[&](id<MTLBuffer> src,id<MTLBuffer> dst,int n){ int off=0,ss=n; [e setComputePipelineState:g_a_copy];
      [e setBuffer:src offset:0 atIndex:0]; [e setBytes:&off length:4 atIndex:1]; [e setBytes:&ss length:4 atIndex:2];
      [e setBuffer:dst offset:0 atIndex:3]; [e setBytes:&n length:4 atIndex:4]; [e setBytes:&n length:4 atIndex:5];
      [e dispatchThreads:MTLSizeMake((size_t)S*n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
    // 1) in_ln: ax_ = rmsnorm(x)
    copyrow(axr_,ax_,AH); BAR(); rmsw(ax_,inB,inoff,AH,S); BAR();
    // 2) attention (ax_ -> aout_)
    if(!encode_attention(e,&W,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,S,T,pos_base,eps,theta,ascale)) return 0;
    BAR();
    // 3) residual: axr_ += aout_ ; then nrm = post_ln(x_new)
    [e setComputePipelineState:g_a_add]; [e setBuffer:axr_ offset:0 atIndex:0]; [e setBuffer:aout_ offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)S*AH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    copyrow(axr_,anrm_,AH); BAR(); rmsw(anrm_,pnB,pnoff,AH,S); BAR();
    // 4) shared expert gate/up + router (all read anrm_, independent)
    bind_gemv(e,shg_w,shg_s,shg_fmt,shg_gs,AH,SI,anrm_,ash1_,S);
    bind_gemv(e,shu_w,shu_s,shu_fmt,shu_gs,AH,SI,anrm_,ash2_,S);
    { int NT=S*E, D=AH; [e setComputePipelineState:g_r_router];
      [e setBuffer:rwB offset:rwoff atIndex:0]; [e setBuffer:anrm_ offset:0 atIndex:1]; [e setBuffer:asig_ offset:0 atIndex:2];
      [e setBytes:&E length:4 atIndex:3]; [e setBytes:&D length:4 atIndex:4]; [e setBytes:&NT length:4 atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; }
    BAR();
    // 5) silu(gate)*up + exact top-K select
    [e setComputePipelineState:g_moe_silu]; [e setBuffer:ash1_ offset:0 atIndex:0]; [e setBuffer:ash2_ offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)S*SI,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    { // COLI_RTOP8 (default ON) swaps the serial 1-thread-per-row select for the exact-
      // match 1-simdgroup-per-row replica (same buffers/args; only pipeline+grid change).
      // E<=256 is required by r_top8_par's ch[8]/32-lane blocking contract; this call
      // site's E is always 256 today (layer_forward_rows' own architecture-shape gate in
      // colibri.c requires c->n_experts==256 to reach coli_metal_layer_decode at all —
      // see PR body "Scope statement") but the check is kept here too, defense-in-depth,
      // so a future relaxation of that gate (e.g. to admit REAP-pruned E=168 models into
      // the fused path) degrades safely to the serial kernel instead of mis-dispatching.
      int use_par = g_rtop8_par && g_rtop8_width_ok && E<=256;
      [e setComputePipelineState:use_par?g_r_top8p:g_r_top8];
      [e setBuffer:asig_ offset:0 atIndex:0]; [e setBuffer:rbB offset:rboff atIndex:1];
      [e setBuffer:aidx_ offset:0 atIndex:2]; [e setBuffer:aw_ offset:0 atIndex:3]; [e setBuffer:akeff_ offset:0 atIndex:4];
      [e setBytes:&E length:4 atIndex:5]; [e setBytes:&K length:4 atIndex:6]; [e setBytes:&Ksel length:4 atIndex:7];
      [e setBytes:&topp length:4 atIndex:8]; [e setBytes:&normk length:4 atIndex:9]; [e setBytes:&rscale length:4 atIndex:10];
      if(use_par) [e dispatchThreadgroups:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
      else        [e dispatchThreads:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(S,1,1)]; }
    BAR();
    // 6) shared down
    bind_gemv(e,shd_w,shd_s,shd_fmt,shd_gs,SI,AH,ash1_,ashout_,S);
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] layer cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    g_attn_sched += [cb GPUStartTime]-[cb kernelStartTime]; g_attn_ksched += [cb kernelStartTime]-tc;
    memcpy(x,[axr_ contents],(size_t)S*AH*4);
    memcpy(inrm_out,[ax_ contents],(size_t)S*AH*4);
    memcpy(nrm_out,[anrm_ contents],(size_t)S*AH*4);
    memcpy(sh_out,[ashout_ contents],(size_t)S*AH*4);
    memcpy(idx_out,[aidx_ contents],(size_t)S*K*4);
    memcpy(w_out,[aw_ contents],(size_t)S*K*4);
    memcpy(keff_out,[akeff_ contents],(size_t)S*4);
  }
  return 1;
}

// Sync GEMM for large row-batches (prefill): y[S,O] = x[S,I] @ W^T * scale. Weights must be
// registered (zero-copy); x/y go through grow-only shared scratch. Returns 0 -> CPU fallback.
static id<MTLBuffer> g_gx, g_gy; static size_t g_gx_cap, g_gy_cap;
extern "C" int coli_metal_gemm(float *y, const float *x, const void *wp, const float *sp,
                               int fmt, int S, int I, int O, int gs) {
  if (!g_dev || (fmt!=1 && fmt!=2 && fmt!=4)) return 0;
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
    [e setBytes:&I length:4 atIndex:5];
    [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
    [e setBytes:&gs length:4 atIndex:9];   /* loop-invariant, like I/O/fmt: group size for fmt=4 (0 = per-row) */
    // Grid-size cap. One dispatch of NT=S*O output elements launches (NT+3)/4 threadgroups; past
    // a device grid limit (observed on M-series: kv_b grid ~3.1e7 tg clean at S=4376, ~5.4e7 tg
    // CORRUPT at S=7478) rows beyond the limit are silently never computed and the output keeps
    // its prior contents -- fresh-zero standalone (nerr=1.0), stale scratch in-engine (the
    // nondeterministic long-context corruption). Chunk over rows so each dispatch stays <=2^25
    // elements (grid <=2^23 tg, ~4x under the observed-clean bound). Chunks write disjoint g_gy
    // ranges. Offset alignment is STRUCTURAL, not shape-dependent (holds for any I,O -- a tiny
    // oracle checkpoint or a future container with odd dims, not just GLM's shapes): g_gy is only
    // ever written scalar (y[row]), so r0*O*4 needs just 4B; and the 16B float4 loads on g_gx
    // execute only inside loops gated by I8=(I&7)?0:(I/8), i.e. only when I%8==0, where r0*I*4 is
    // a multiple of 32. With I%8!=0, I8=0, x4 is never dereferenced and x is read scalar via xr[i].
    // COLI_GEMM_CHUNK=0 disables chunking (one full dispatch = the buggy pre-fix behavior) so the
    // fix can be A/B'd on a single binary. Default on.
    static int chunk_on=-1;
    if(chunk_on<0){ const char*e=getenv("COLI_GEMM_CHUNK"); chunk_on=(e&&e[0]=='0'&&!e[1])?0:1; }
    const int64_t NT_MAX = chunk_on ? ((int64_t)1<<25) : ((int64_t)1<<62);
    // Clamp in 64-bit BEFORE narrowing: with chunking off NT_MAX/O is ~1.6e14 for kv_b (O=28672),
    // far past INT_MAX, so casting first is implementation-defined. Were it to truncate negative
    // on some toolchain, CH=1 would make the disable path dispatch one row at a time and silently
    // STOP reproducing the bug it exists to demonstrate. After the clamp CH <= S <= INT_MAX.
    int64_t ch64=NT_MAX/O; if(ch64<1) ch64=1; if(ch64>S) ch64=S; int CH=(int)ch64;
    for(int r0=0;r0<S;r0+=CH){
      int ch=(S-r0<CH)?(S-r0):CH; int NT=ch*O;
      [e setBuffer:g_gx offset:(size_t)r0*I*4 atIndex:2];
      [e setBuffer:g_gy offset:(size_t)r0*O*4 atIndex:3];
      [e setBytes:&ch length:4 atIndex:4];
      [e setBytes:&NT length:4 atIndex:8];
      [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] gemm cmdbuf error (S=%d O=%d)\n",S,O); return 0; }
    memcpy(y,[g_gy contents],(size_t)S*O*4);
  }
  return 1;
}

// Standalone single-kernel runner for the top-8 select (see backend_metal.h). Fresh
// shared buffers per call (a test/probe path, not a hot path); grids exactly as the
// engine dispatch site: serial = S threads of one S-wide threadgroup, parallel = S
// threadgroups x 32 (one simdgroup per row). "par" is a REQUEST, not a guarantee: same
// E<=256 and SIMD-width-32 host-side checks as the engine dispatch site gate the actual
// pipeline choice, so a caller (including metal-test itself) can never reach the parallel
// kernel out of contract by asking for it — par=1 with E>256, or on a non-32-wide device,
// transparently runs the serial kernel instead and still returns 1 (success).
extern "C" int coli_metal_rtop8(int par, const float *sig, const float *bias, int S, int E, int K,
                                int Ksel, float topp, int normk, float rscale,
                                int *idx, float *w, int *keff) {
  if (!g_dev || S < 1 || E < 1 || K < 1 || Ksel < 1 || Ksel > K) return 0;
  int use_par = par && g_r_top8p && g_rtop8_width_ok && E<=256;
  @autoreleasepool {
    id<MTLBuffer> bs=[g_dev newBufferWithBytes:sig  length:(size_t)S*E*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb=[g_dev newBufferWithBytes:bias length:(size_t)E*4   options:MTLResourceStorageModeShared];
    id<MTLBuffer> bi=[g_dev newBufferWithLength:(size_t)S*K*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw=[g_dev newBufferWithLength:(size_t)S*K*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bk=[g_dev newBufferWithLength:(size_t)S*4   options:MTLResourceStorageModeShared];
    if(!bs||!bb||!bi||!bw||!bk) return 0;
    memset(bi.contents,0xFF,(size_t)S*K*4);           // poison: untouched slots stay visible
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:use_par?g_r_top8p:g_r_top8];
    [e setBuffer:bs offset:0 atIndex:0]; [e setBuffer:bb offset:0 atIndex:1];
    [e setBuffer:bi offset:0 atIndex:2]; [e setBuffer:bw offset:0 atIndex:3]; [e setBuffer:bk offset:0 atIndex:4];
    [e setBytes:&E length:4 atIndex:5]; [e setBytes:&K length:4 atIndex:6]; [e setBytes:&Ksel length:4 atIndex:7];
    [e setBytes:&topp length:4 atIndex:8]; [e setBytes:&normk length:4 atIndex:9]; [e setBytes:&rscale length:4 atIndex:10];
    if(use_par) [e dispatchThreadgroups:MTLSizeMake((NSUInteger)S,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    else        [e dispatchThreads:MTLSizeMake((NSUInteger)S,1,1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)S,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] rtop8 cmdbuf error\n"); return 0; }
    memcpy(idx,bi.contents,(size_t)S*K*4);
    memcpy(w,bw.contents,(size_t)S*K*4);
    memcpy(keff,bk.contents,(size_t)S*4);
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
static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr, int R,
                         id<MTLBuffer> xg_buf, id<MTLBuffer> gg_buf, id<MTLBuffer> uu_buf, id<MTLBuffer> hh_buf) {
  if (!g_dev || (fmt != 1 && fmt != 2 && fmt != 4 && fmt != 5 && fmt != 6)) return nil;
  if (fmt == 6) {   /* e8 kernel assumes clean block tiling, and every FWHT tile of the
                     * down input (CPU tiling rule, e8_rot_rows) must fit threadgroup mem */
    if ((D & 255) || (Iinter & 31)) return nil;
    for (int off = 0; off < Iinter; ) {
      int rem = Iinter - off, n = rem & (-rem);
      while (n > 32768) n >>= 1;
      if (n > 4096) return nil;
      off += n;
    }
  }
  // COLI_METAL_MOE_EXACT=1: route fmt=4 routed experts to the CPU reference path (bit-exact,
  // matches matmul_i4_grouped) instead of the fast batched float4 kernel. Opt-in; default is
  // the fast GPU path. Returning nil makes moe() fall back to CPU for fmt=4 experts, exactly
  // like an unresolved slab -- fmt=1/2 stay on the GPU, attention/dense are untouched.
  // (PR #587 gate-2: token-exact mode for trajectories whose gap dips under the drift tail.)
  { static int g_moe_exact = -1;
    if (g_moe_exact < 0) { const char *e = getenv("COLI_METAL_MOE_EXACT"); g_moe_exact = (e && e[0] && e[0] != '0'); }
    if (g_moe_exact) return nil; }  /* exact mode is path-scoped, not fmt-scoped: the resident
                                     * tier (fmt=1 on mixed containers) carries the same
                                     * accumulation-order drift, so ALL routed experts fall to
                                     * CPU under the flag (measured: 4/5 prompt flips -> 0/5,
                                     * real g64 744B container, #587) */
  if (g_resset_enabled) {   // E5: commit any pending slab adds before we may skip useResource:
    double t0 = mnow(); resset_flush(); g_t_resset_flush += mnow() - t0;   // METAL-RESSET line
  }
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
  // E5 (COLI_METAL_RESSET=1): the queue-attached MTLResidencySet already guarantees these
  // buffers are resident, so skip the per-buffer declaration whose count scales with LRU
  // cache size (mechanism history v5). Residency sets don't do hazard tracking (Apple docs),
  // but none was load-bearing here: every buffer in `use` is MTLResourceUsageRead-only and
  // referenced only indirectly (moe_gemv dereferences waddr[]/saddr[] baked into bag/bsg's
  // contents), so there's no GPU-side write to serialize against; the one real hazard -- a
  // slab unregistered+freed+reused while an async in-flight CB still reads it -- is a
  // CPU-write race outside Metal's hazard tracking either way, held by the engine's own slot
  // lifecycle, not by useResource:. See docs/experiments/e5-metal-residency-set.md UNCERTAINTIES.
  if (!g_resset_enabled) {
    for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];
  }
  auto gemv=[&](id<MTLBuffer> wa,id<MTLBuffer> sa,id<MTLBuffer> xin,id<MTLBuffer> y,int O,int K,int Kin){
    int NT=R*O;
    [e setComputePipelineState:g_moe_gemv];
    [e setBuffer:wa offset:0 atIndex:0];[e setBuffer:sa offset:0 atIndex:1];[e setBuffer:berow offset:0 atIndex:2];
    [e setBuffer:xin offset:0 atIndex:3];[e setBuffer:y offset:0 atIndex:4];
    [e setBytes:&O length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&Kin length:4 atIndex:7];[e setBytes:&fmt length:4 atIndex:8];
    [e setBytes:&NT length:4 atIndex:9];[e setBytes:&qgs length:4 atIndex:10];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; };
  gemv(bag,bsg,xg_buf,gg_buf,Iinter,D,D);                     // gate
  gemv(bau,bsu,xg_buf,uu_buf,Iinter,D,D);                     // up
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [e setComputePipelineState:g_moe_silu];
  [e setBuffer:gg_buf offset:0 atIndex:0];[e setBuffer:uu_buf offset:0 atIndex:1];
  [e dispatchThreads:MTLSizeMake((size_t)R*Iinter,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  if (fmt == 6) {   /* rotate the down-projection input in place: same block-diagonal
                     * tiling as quant.h e8_rot_rows (largest power of two dividing the
                     * remainder, capped at 4096 for threadgroup memory) */
    int off = 0;
    while (off < Iinter) {
      int rem = Iinter - off, n = rem & (-rem);
      while (n > 32768) n >>= 1;   /* CPU tiling rule (e8_rot_rows); sizes pre-validated above */
      id<MTLBuffer> sb = fwht_signs(n);
      [e setComputePipelineState:g_moe_fwht];
      [e setBuffer:gg_buf offset:0 atIndex:0];[e setBuffer:sb offset:0 atIndex:1];
      [e setBytes:&Iinter length:4 atIndex:2];[e setBytes:&off length:4 atIndex:3];
      [e setBytes:&n length:4 atIndex:4];
      [e dispatchThreadgroups:MTLSizeMake((size_t)R,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      off += n;
    }
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
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

extern "C" int coli_metal_moe_block(int nb, int D, int Iinter, int fmt, int qgs,
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
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,gs,us,ds,xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);
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
extern "C" ColiMetalMoeHandle* coli_metal_moe_block_begin(int nb, int D, int Iinter, int fmt, int qgs,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr,
                         const int *rows, const float *rw) {
  @autoreleasepool {
    int R = 0; for (int e=0;e<nb;e++) R += nr[e];
    if (R == 0 || !g_dev) return nullptr;
    id<MTLBuffer> bxg=[g_dev newBufferWithLength:(size_t)R*D*4 options:g_res_opts];
    id<MTLBuffer> bgg=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:g_res_opts];
    id<MTLBuffer> buu=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:g_res_opts];
    id<MTLBuffer> bhh=[g_dev newBufferWithLength:(size_t)R*D*4 options:g_res_opts];
    id<MTLCommandBuffer> cb = moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,gs,us,ds,xg,xoff,nr,R,bxg,bgg,buu,bhh);
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

/* ---- Standalone elementwise / normalization ops --------------------------------

 These are thin wrappers around the already-compiled g_a_rms / g_a_add / g_moe_silu
 pipelines.  Each helper allocates a Managed (私密) buffer so the initial CPU-side
 floating-point data survives for GPU reads, then uses `blitSynchronizeResource` +
 `blitFromResource:toBuffer` to push back the GPU result into caller's ptr.  */

// Push initial bytes from a CPU pointer into a Managed buffer and return the buffer.
static id<MTLBuffer> cpubuf(id<MTLDevice> dev, const void *cpu, size_t len) {
  id<MTLBuffer> b = [dev newBufferWithLength:len options:MTLResourceStorageModeManaged];
  if (!b) return nil;
  memcpy(b.contents, cpu, len);
  [b didModifyRange:NSMakeRange(0, (NSUInteger)len)];
  return b;
}

extern "C" int coli_metal_rmsnorm(float *x, const float *w, int nrows, int D, float eps) {
  if (!g_dev) return 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBuffer> xb = cpubuf(g_dev, x, (size_t)nrows*D*4);
    id<MTLBuffer> wb = cpubuf(g_dev, w, (size_t)D*4);
    if (!xb || !wb) return 0;
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    int tgsz = 256;
    [e setComputePipelineState:g_a_rms];
    [e setBuffer:xb offset:0 atIndex:0]; [e setBuffer:wb offset:0 atIndex:1];
    [e setBytes:&D length:4 atIndex:2]; [e setBytes:&eps length:4 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(nrows,1,1) threadsPerThreadgroup:MTLSizeMake(tgsz,1,1)];
    [e endEncoding];
    { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:xb]; [bl endEncoding]; }
    [cb commit]; [cb waitUntilCompleted];
    memcpy(x, xb.contents, (size_t)nrows * D * 4);
    return cb.status != MTLCommandBufferStatusError;
  }
}

extern "C" int coli_metal_add(float *y, const float *a, size_t n) {
  if (!g_dev || n == 0) return n == 0 ? 1 : 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBuffer> yb = cpubuf(g_dev, y, n*4);
    id<MTLBuffer> ab = cpubuf(g_dev, a, n*4);
    if (!yb || !ab) return 0;
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    int tgsz = 256;
    [e setComputePipelineState:g_a_add];
    [e setBuffer:yb offset:0 atIndex:0]; [e setBuffer:ab offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((uint32_t)n,1,1) threadsPerThreadgroup:MTLSizeMake(tgsz,1,1)];
    [e endEncoding];
    { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:yb]; [bl endEncoding]; }
    [cb commit]; [cb waitUntilCompleted];
    memcpy(y, yb.contents, n*4);
    return cb.status != MTLCommandBufferStatusError;
  }
}

extern "C" int coli_metal_silu_mul(float *g, const float *u, size_t n) {
  if (!g_dev || n == 0) return n == 0 ? 1 : 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBuffer> gb = cpubuf(g_dev, g, n*4);
    id<MTLBuffer> ub = cpubuf(g_dev, u, n*4);
    if (!gb || !ub) return 0;
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    int tgsz = 256;
    [e setComputePipelineState:g_moe_silu];
    [e setBuffer:gb offset:0 atIndex:0]; [e setBuffer:ub offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((uint32_t)n,1,1) threadsPerThreadgroup:MTLSizeMake(tgsz,1,1)];
    [e endEncoding];
    { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:gb]; [bl endEncoding]; }
    [cb commit]; [cb waitUntilCompleted];
    memcpy(g, gb.contents, n*4);
    return cb.status != MTLCommandBufferStatusError;
  }
}

// — KV cache row write & clear —

static int cpy(id<MTLCommandBuffer> cb, id<MTLBuffer> dst, ptrdiff_t doff,
               id<MTLBuffer> src, ptrdiff_t soff, uint32_t n) {
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:g_a_copy];
  [e setBuffer:dst offset:0 atIndex:0]; [e setBuffer:src offset:0 atIndex:1];
  [e setBytes:&doff length:4 atIndex:2]; [e setBytes:&soff length:4 atIndex:3];
  uint32_t tg=128, nt=((uint32_t)n+tg-1)/tg;
  [e dispatchThreads:MTLSizeMake(nt*tg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
  [e endEncoding];
  return cb.status != MTLCommandBufferStatusError;
}

extern "C" int coli_metal_kv_write(float *Lc, float *Rc, const float *L_raw, const float *R_raw,
    const float *kv_a_ln, int S, int pos_base, int kv_lora, int qk_rope, float eps, float theta) {
  if (!g_dev || S <= 0) return S <= 0 ? 1 : 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];

    id<MTLBuffer> bLc = wrap(Lc,  (size_t)S * kv_lora * 4);
    id<MTLBuffer> bRc = wrap(Rc,  (size_t)S * qk_rope * 4);
    id<MTLBuffer> bLu = wrap(L_raw, (size_t)S * kv_lora * 4);
    id<MTLBuffer> bRu = wrap(R_raw, (size_t)S * qk_rope * 4);
    id<MTLBuffer> bWa = wrap(kv_a_ln, kv_lora * 4);

    if (!bLc || !bRc || !bLu || !bRu || !bWa) return 0;

    // 1) Copy raw L → Lc rows
    int rL = cpy(cb, bLc, 0, bLu, 0, (uint32_t)(S * kv_lora));
    // 2) Copy raw R → Rc rows
    int rR = cpy(cb, bRc, 0, bRu, 0, (uint32_t)(S * qk_rope));
    if (!rL || !rR) return 0; // will still commit below — a_copy sets both buffers

    auto rms = [&](id<MTLCommandBuffer> x, id<MTLBuffer> b, size_t off, id<MTLBuffer> Qw, int n) {
      int QV = n; uint32_t tg = 128, nt = ((uint32_t)QV + tg - 1) / tg;
      id<MTLComputeCommandEncoder> e = [x computeCommandEncoder];
      [e setComputePipelineState:g_a_rms];
      [e setBuffer:b  offset:off atIndex:0]; [e setBuffer:Qw offset:0 atIndex:1];
      float ep[2] = {eps, 0}; [e setBytes:ep length:8 atIndex:2];
      [e dispatchThreads:MTLSizeMake(nt*tg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [e endEncoding];
    };

    auto rope = [&](id<MTLCommandBuffer> x, id<MTLBuffer> b, size_t off,
                    int base, int rs, int hs, int nh, float theta) {
      uint32_t ttgq = 256; uint32_t threads = (uint32_t)nh * (uint32_t)(rs / 2);
      uint32_t ntg = (threads + ttgq - 1) / ttgq;
      id<MTLComputeCommandEncoder> e = [x computeCommandEncoder];
      [e setComputePipelineState:g_a_rope];
      [e setBuffer:b    offset:off atIndex:0];
      [e setBytes:&pos_base length:4 atIndex:1];
      int v0[4] = {base, rs, hs, 0};
      [e setBytes:v0 length:16 atIndex:2];
      float th[2] = {theta, 0}; [e setBytes:th length:8 atIndex:3];
      [e dispatchThreads:MTLSizeMake(ntg*ttgq,1,1) threadsPerThreadgroup:MTLSizeMake(ttgq,1,1)];
      [e endEncoding];
    };

    // 3) rmsnorm on L rows
    for (int i = 0; i < S; i++)
      rms(cb, bLc, (size_t)i * kv_lora * 4, bWa, kv_lora);

    // 4) rope_interleave on R rows
    rope(cb, bRc, 0, 0, qk_rope, 0, S, theta);

    [cb commit]; [cb waitUntilCompleted];
    return cb.status != MTLCommandBufferStatusError;
  }
}

extern "C" int coli_metal_kv_clear(float *Lc, float *Rc, int from, int to, int kv_lora, int qk_rope) {
  if (!g_dev) return 0;
  int nr = to - from;
  if (nr <= 0) return 1;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBuffer> bLc = wrap(Lc, (size_t)nr * kv_lora * 4);
    id<MTLBuffer> bRc = wrap(Rc, (size_t)nr * qk_rope * 4);
    if (!bLc || !bRc) return 0;

    uint32_t maxD = (uint32_t)(kv_lora > qk_rope ? kv_lora : qk_rope);
    uint32_t total = 2 * (uint32_t)nr * maxD;
    uint32_t tg = 256, ntg = (total + tg - 1) / tg;

    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_kv_clear];
    [e setBuffer:bLc offset:(size_t)from * kv_lora * 4 atIndex:0];
    [e setBuffer:bRc offset:(size_t)from * qk_rope * 4 atIndex:1];
    int v[4] = {from, to, kv_lora, qk_rope};
    [e setBytes:v length:16 atIndex:2];
    [e dispatchThreads:MTLSizeMake(ntg * tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [e endEncoding];

    [cb commit]; [cb waitUntilCompleted];
    return cb.status != MTLCommandBufferStatusError;
  }
}

// — KDA state recurrence —

extern "C" int coli_metal_kda_state(float *S, const float *qn, const float *kn, const float *vh,
                                     const float *alpha, const float *beta, float *oh,
                                     int H, int hd) {
  if (!g_dev || H <= 0) return H <= 0 ? 1 : 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBuffer> bS = wrap(S,        (size_t)H * hd * hd * sizeof(float));
    id<MTLBuffer> bqn = wrap(qn,      (size_t)H * hd * sizeof(float));
    id<MTLBuffer> bkn = wrap(kn,      (size_t)H * hd * sizeof(float));
    id<MTLBuffer> bvh = wrap(vh,      (size_t)H * hd * sizeof(float));
    id<MTLBuffer> balph = wrap(alpha, (size_t)H * hd * sizeof(float));
    id<MTLBuffer> bbet = wrap(beta,   (size_t)H * sizeof(float));
    id<MTLBuffer> boh = wrap(oh,      (size_t)H * hd * sizeof(float));
    if (!bS || !bqn || !bkn || !bvh || !balph || !bbet || !boh) return 0;
    uint32_t total = (uint32_t)H * (uint32_t)hd;
    uint32_t tg = 256, ntg = (total + tg - 1) / tg;
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_kda_state];
    [e setBuffer:bS    offset:0 atIndex:0];
    [e setBuffer:bqn   offset:0 atIndex:1];
    [e setBuffer:bkn   offset:0 atIndex:2];
    [e setBuffer:bvh   offset:0 atIndex:3];
    [e setBuffer:balph offset:0 atIndex:4];
    [e setBuffer:bbet  offset:0 atIndex:5];
    [e setBuffer:boh   offset:0 atIndex:6];
    [e setBytes:&H length:4 atIndex:7];
    [e setBytes:&hd length:4 atIndex:8];
    [e dispatchThreads:MTLSizeMake(ntg * tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    { memcpy(oh, [boh contents], (size_t)H * hd * sizeof(float)); }
    return cb.status != MTLCommandBufferStatusError;
  }
}

// — KDA Conv1d + SiLU (one projection) —

extern "C" int coli_metal_kda_conv_silu(float *conv_win, float *vec, const float *taps, int P, int K) {
  if (!g_dev || P <= 0) return P <= 0 ? 1 : 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBuffer> bwin = wrap(conv_win, (size_t)P * K * sizeof(float));
    id<MTLBuffer> bvec = wrap(vec,      (size_t)P * sizeof(float));
    id<MTLBuffer> btap = wrap(taps,     (size_t)P * K * sizeof(float));
    if (!bwin || !bvec || !btap) return 0;

    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_kda_conv_silu];
    [e setBuffer:bwin offset:0 atIndex:0];
    [e setBuffer:bvec offset:0 atIndex:1];
    [e setBuffer:btap offset:0 atIndex:2];

    int Pv = P; int Kv = K;
    [e setBytes:&Pv length:4 atIndex:3];
    [e setBytes:&Kv length:4 atIndex:4];

    uint32_t tg = 256, ntg = (P + tg - 1) / tg;
    [e dispatchThreads:MTLSizeMake(ntg * tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    return cb.status != MTLCommandBufferStatusError;
  }
}

// — KDA L2 normalization (q and k, in-place) —

extern "C" int coli_metal_kda_l2_norm(float *q, float *k, int H, int hd, float qscale) {
  if (!g_dev || H <= 0) return H <= 0 ? 1 : 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBuffer> bq = wrap(q, (size_t)H * hd * sizeof(float));
    id<MTLBuffer> bk = wrap(k, (size_t)H * hd * sizeof(float));
    if (!bq || !bk) return 0;

    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:g_kda_l2_norm];
    [e setBuffer:bq offset:0 atIndex:0];
    [e setBuffer:bk offset:0 atIndex:1];

    int Hv = H; int hdv = hd;
    [e setBytes:&Hv length:4 atIndex:2];
    [e setBytes:&hdv length:4 atIndex:3];
    [e setBytes:&qscale length:sizeof(float) atIndex:4];

    uint32_t tg = 256, ntg = (H + tg - 1) / tg;
    [e dispatchThreads:MTLSizeMake(ntg * tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [e endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    return cb.status != MTLCommandBufferStatusError;
  }
}

/* ---------- KDA fused token step ----------
 * Single command buffer: conv_silu(q,k,v) → l2_norm(q,k) → state recurrence.
 * Alpha and beta are precomputed on CPU (graw+dt → alpha, braw → beta).
 * oh[] is written by state kernel; caller must zero before call.
 * After return, CPU performs per-head RMSNorm + sigmoid gate from oh → on. */
extern "C" int coli_metal_kda_fused_token(
    float *win_q, float *qt,
    float *win_k, float *kt,
    float *win_v, float *tv,
    const float *taps_q, const float *taps_k, const float *taps_v,
    float *S,
    const float *alpha, const float *beta,
    float *oh,
    int P, int K, int H, int hd) {
  if (!g_dev || P <= 0 || H <= 0) return 0;
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];

    // Pre-wrap all buffers
    size_t P_K_sz = (size_t)P * K * sizeof(float);
    size_t P_sz   = (size_t)P * sizeof(float);
    size_t S_sz   = (size_t)H * hd * hd * sizeof(float);
    size_t a_sz   = (size_t)H * hd * sizeof(float);
    size_t b_sz   = (size_t)H * sizeof(float);

    // Persistent (model-lifetime) buffers: wrap once and reuse. win_* and S are 16 KB-aligned
    // (afcalloc) so these are zero-copy — GPU writes to conv windows and state land directly
    // in host memory and persist to the next token. taps are read-only constants.
    id<MTLBuffer> bwin_q = wrap_persistent(win_q, P_K_sz);
    id<MTLBuffer> bwin_k = wrap_persistent(win_k, P_K_sz);
    id<MTLBuffer> bwin_v = wrap_persistent(win_v, P_K_sz);
    id<MTLBuffer> btap_q = wrap_persistent(taps_q, P_K_sz);
    id<MTLBuffer> btap_k = wrap_persistent(taps_k, P_K_sz);
    id<MTLBuffer> btap_v = wrap_persistent(taps_v, P_K_sz);
    id<MTLBuffer> bS     = wrap_persistent(S, S_sz);
    // Transient (per-forward) buffers: pointers change each call, so wrap fresh every time.
    id<MTLBuffer> bqt    = wrap(qt, P_sz);
    id<MTLBuffer> bkt    = wrap(kt, P_sz);
    id<MTLBuffer> btv    = wrap(tv, P_sz);
    id<MTLBuffer> balph  = wrap(alpha, a_sz);
    id<MTLBuffer> bbet   = wrap(beta, b_sz);
    id<MTLBuffer> boh    = wrap(oh, a_sz);

    #define _BT(b) ((b)?(void*)1:(void*)0)
    if(!bwin_q||!bwin_k||!bwin_v||!bqt||!bkt||!btv||
       !btap_q||!btap_k||!btap_v||!bS||!balph||!bbet||!boh){
      fprintf(stderr,"[FUSED] wrap nil: q=%p k=%p v=%p qt=%p kt=%p tv=%p tq=%p tk=%p tv2=%p S=%p a=%p b=%p oh=%p\n",
              _BT(bwin_q),_BT(bwin_k),_BT(bwin_v),_BT(bqt),_BT(bkt),_BT(btv),
              _BT(btap_q),_BT(btap_k),_BT(btap_v),_BT(bS),_BT(balph),_BT(bbet),_BT(boh));
      return 0;
    }
    #undef _BT

    /* -- Dispatch 1-3: conv_silu for q, k, v -- */
    { // conv q
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:g_kda_conv_silu];
      [e setBuffer:bwin_q offset:0 atIndex:0];
      [e setBuffer:bqt    offset:0 atIndex:1];
      [e setBuffer:btap_q offset:0 atIndex:2];
      int Pv=P, Kv=K;
      [e setBytes:&Pv length:4 atIndex:3];
      [e setBytes:&Kv length:4 atIndex:4];
      uint32_t tg=256, ntg=(P+tg-1)/tg;
      [e dispatchThreads:MTLSizeMake(ntg*tg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [e endEncoding];
    }
    { // conv k
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:g_kda_conv_silu];
      [e setBuffer:bwin_k offset:0 atIndex:0];
      [e setBuffer:bkt    offset:0 atIndex:1];
      [e setBuffer:btap_k offset:0 atIndex:2];
      int Pv=P, Kv=K;
      [e setBytes:&Pv length:4 atIndex:3];
      [e setBytes:&Kv length:4 atIndex:4];
      uint32_t tg=256, ntg=(P+tg-1)/tg;
      [e dispatchThreads:MTLSizeMake(ntg*tg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [e endEncoding];
    }
    { // conv v
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:g_kda_conv_silu];
      [e setBuffer:bwin_v offset:0 atIndex:0];
      [e setBuffer:btv    offset:0 atIndex:1];
      [e setBuffer:btap_v offset:0 atIndex:2];
      int Pv=P, Kv=K;
      [e setBytes:&Pv length:4 atIndex:3];
      [e setBytes:&Kv length:4 atIndex:4];
      uint32_t tg=256, ntg=(P+tg-1)/tg;
      [e dispatchThreads:MTLSizeMake(ntg*tg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [e endEncoding];
    }

    /* -- Dispatch 4: l2_norm on q, k -- */
    {
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:g_kda_l2_norm];
      [e setBuffer:bqt offset:0 atIndex:0];
      [e setBuffer:bkt offset:0 atIndex:1];
      int Hv=H, hdv=hd;
      float qs = 1.f / sqrtf((float)hd);
      [e setBytes:&Hv length:4 atIndex:2];
      [e setBytes:&hdv length:4 atIndex:3];
      [e setBytes:&qs length:sizeof(float) atIndex:4];
      uint32_t tg=256, ntg=(H+tg-1)/tg;
      [e dispatchThreads:MTLSizeMake(ntg*tg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [e endEncoding];
    }

    /* -- Dispatch 5: state recurrence -- */
    {
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:g_kda_state];
      [e setBuffer:bS    offset:0 atIndex:0];
      [e setBuffer:bqt   offset:0 atIndex:1];
      [e setBuffer:bkt   offset:0 atIndex:2];
      [e setBuffer:btv   offset:0 atIndex:3];
      [e setBuffer:balph offset:0 atIndex:4];
      [e setBuffer:bbet  offset:0 atIndex:5];
      [e setBuffer:boh   offset:0 atIndex:6];
      [e setBytes:&H length:4 atIndex:7];
      [e setBytes:&hd length:4 atIndex:8];
      uint32_t total=(uint32_t)H*(uint32_t)hd;
      uint32_t tg=256, ntg=(total+tg-1)/tg;
      [e dispatchThreads:MTLSizeMake(ntg*tg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
      [e endEncoding];
    }

    [cb commit];
    [cb waitUntilCompleted];
    memcpy(oh, [boh contents], (size_t)H * hd * sizeof(float));
    memcpy(S, [bS contents], S_sz);
    memcpy(win_q, [bwin_q contents], P_K_sz);
    memcpy(win_k, [bwin_k contents], P_K_sz);
    memcpy(win_v, [bwin_v contents], P_K_sz);
    return cb.status != MTLCommandBufferStatusError;
  }
}
