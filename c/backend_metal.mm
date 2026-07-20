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
                    constant int& NT [[buffer(8)]],
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
  } else {                                          // f32
    device const float* wr = (device const float*)(w) + (long)o * I;
    device const float4* w4 = (device const float4*)wr;
    for (int c = slane; c < I8; c += 32) acc += dot(w4[2*c],x4[2*c]) + dot(w4[2*c+1],x4[2*c+1]);
    for (int i = I8*8 + slane; i < I; i += 32) acc += wr[i] * xr[i];
  }
  acc = simd_sum(acc);
  if (slane == 0) y[row] = acc * scale[o];
}

// Batched bindless expert GEMV: each row gr belongs to expert erow[gr], whose weight and
// scale live at gpuAddresses waddr[e]/saddr[e] (zero-copy in the RAM slab). fmt 1=i8, 2=i4.
// One SIMDGROUP per output row, 4 rows/threadgroup, 8-value loads: measured 1.5-2.1x over
// one-threadgroup-per-row with uchar2 loads (358-389 GB/s on engine-like block shapes).
kernel void moe_gemv(device const ulong* waddr [[buffer(0)]], device const ulong* saddr [[buffer(1)]],
                     device const int* erow [[buffer(2)]], device const float* xin [[buffer(3)]],
                     device float* yout [[buffer(4)]],
                     constant int& O [[buffer(5)]], constant int& K [[buffer(6)]],
                     constant int& Kin [[buffer(7)]], constant int& fmt [[buffer(8)]],
                     constant int& NT [[buffer(9)]],
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
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
    device const char4* w4=(device const char4*)w;
    for(int c=slane;c<K8;c+=32) acc+=dot(float4(w4[2*c]),x4[2*c])+dot(float4(w4[2*c+1]),x4[2*c+1]);
    for(int i=K8*8+slane;i<K;i+=32) acc+=float(w[i])*xr[i];
  }
  acc=simd_sum(acc);
  if(slane==0) yout[row]=acc*sc[o];
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

// ===== KV8: fp8 e4m3 latent KV cache (bytes + per-row f32 scale) =====
// exact twin of coli_fp8_enc (kv_fp8.h): float -> e4m3, round-to-nearest-even, sat +-448.
inline uchar fp8_enc(float f){
  uint u=as_type<uint>(f); uchar s=(uchar)((u>>24)&0x80u); float a=fabs(f);
  if(!(a<=448.0f)) return (a!=a)?(uchar)0:(uchar)(s|0x7e);        // NaN->0, inf/overflow->+-448
  if(a<0x1p-6f){ int k=(int)rint(a*0x1p9f); return (uchar)(s|(uchar)k); }   // denormal grid, step 2^-9
  int e; frexp(a,e); int E=e+6; int k=(int)rint(ldexp(a,3-(e-1)));         // mantissa RNE in [8,16]
  if(k==16){ k=8; E++; } if(E>15||(E==15&&k>14)) return (uchar)(s|0x7e);
  return (uchar)(s|(uchar)(E<<3)|(uchar)(k-8));
}
// per-row fp8 quantize: one threadgroup per row. src[nrows,n] contiguous scratch -> dst rows at (pb+row).
// matches coli_kv8_quant_row: amax/448 scale; zero/subnormal/NaN row -> bytes 0, scale 1.
kernel void a_fp8enc(device const float* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                     device float* scl [[buffer(2)]], constant int& n [[buffer(3)]], constant int& pb [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]){
  device const float* xr=src+(long)row*n; threadgroup float red[256];
  float m=0; for(int i=lid;i<n;i+=tgsz) m=max(m,fabs(xr[i])); red[lid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]=max(red[lid],red[lid+k]); threadgroup_barrier(mem_flags::mem_threadgroup); }
  float amax=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
  device uchar* dr=dst+(long)(pb+(int)row)*n;
  if(!(amax>1e-35f) || amax>3.4e38f){ for(int i=lid;i<n;i+=tgsz) dr[i]=0; if(lid==0) scl[pb+(int)row]=1.0f; return; }
  float inv=448.0f/amax; for(int i=lid;i<n;i+=tgsz) dr[i]=fp8_enc(xr[i]*inv);
  if(lid==0) scl[pb+(int)row]=amax/448.0f;
}
// score with fp8 L/R rows + per-row scale: a = Lsc*sum(qabs*lut[L]) + Rsc*sum(qr*lut[R]).
kernel void a_score8(device const float* qabs [[buffer(0)]], device const uchar* Lc [[buffer(1)]],
                     device const float* Lsc [[buffer(2)]], device const uchar* Rc [[buffer(3)]],
                     device const float* Rsc [[buffer(4)]], device const float* q [[buffer(5)]],
                     device const float* lut [[buffer(6)]], device float* sc [[buffer(7)]],
                     constant int& T [[buffer(8)]], constant float& ascale [[buffer(9)]],
                     constant int& PB [[buffer(10)]], uint gid [[thread_position_in_grid]]){
  int s=gid/(A_H*T), r=gid%(A_H*T), h=r/T, t=r%T; long o=(long)(s*A_H+h)*T+t;
  if(t > PB+s){ sc[o]=-1e30f; return; }
  device const float* qa=qabs+(long)(s*A_H+h)*A_KVL; device const uchar* Lt=Lc+(long)t*A_KVL;
  device const float* qr=q+(long)s*A_QHH+(long)h*A_QH+A_NOPE; device const uchar* Rt=Rc+(long)t*A_ROPE;
  float al=0,ar=0; for(int i=0;i<A_KVL;i++) al+=qa[i]*lut[Lt[i]]; for(int d=0;d<A_ROPE;d++) ar+=qr[d]*lut[Rt[d]];
  sc[o]=(al*Lsc[t]+ar*Rsc[t])*ascale;
}
// context accumulation in latent space with fp8 L rows + scale: clat[i] = sum_t sc[t]*Lsc[t]*lut[L[t][i]].
kernel void a_clat8(device const float* sc [[buffer(0)]], device const uchar* Lc [[buffer(1)]],
                    device const float* Lsc [[buffer(2)]], device const float* lut [[buffer(3)]],
                    device float* clat [[buffer(4)]], constant int& T [[buffer(5)]], uint gid [[thread_position_in_grid]]){
  int sh=gid/A_KVL, i=gid%A_KVL; device const float* s=sc+(long)sh*T; float a=0;
  for(int t=0;t<T;t++) a+=s[t]*Lsc[t]*lut[Lc[(long)t*A_KVL+i]]; clat[(long)sh*A_KVL+i]=a;
}

// ===== KV_TQ: PolarQuant latent KV (randomized Hadamard rotation + recursive polar) =====
// Faithful device twin of kv_tq.h: one thread per row (serial), n<=512, bits in {2..6}. The
// radius (row L2 norm) rides the per-row scale slot; only angles are packed. Decode reconstructs
// each row to f32 (a_tq_deq); the existing f32 a_score/a_clat then run on the staged rows.
constant uint TQ_SEED = 0x9E3779B9u;
constant float TQ_PI = 3.14159265358979323846f;
// rotated-int4 Lloyd-Max 16-level codebook for N(0,1) (twin of coli_q4_lev).
constant float TQ_Q4LEV[16] = { -2.733f,-2.069f,-1.618f,-1.256f,-0.942f,-0.657f,-0.388f,-0.128f,
                                 0.128f, 0.388f, 0.657f, 0.942f, 1.256f, 1.618f, 2.069f, 2.733f };
inline int tq_q4enc(float v){ int best=0; float bd=1e30f;
  for(int k=0;k<16;k++){ float d=fabs(v-TQ_Q4LEV[k]); if(d<bd){bd=d;best=k;} } return best; }
inline uint  tq_hash(uint x){ x+=0x9E3779B9u; x=(x^(x>>16))*0x85EBCA6Bu; x=(x^(x>>13))*0xC2B2AE35u; return x^(x>>16); }
inline float tq_sign(int i){ return (tq_hash(TQ_SEED ^ (uint)i)&1u)?-1.0f:1.0f; }
inline void  tq_fwht(thread float* a, int n){
  for(int len=1; len<n; len<<=1) for(int i=0;i<n;i+=len<<1) for(int j=i;j<i+len;j++){ float u=a[j],v=a[j+len]; a[j]=u+v; a[j+len]=u-v; }
  float inv=1.0f/sqrt((float)n); for(int i=0;i<n;i++) a[i]*=inv;
}
inline int   tq_b1(int bits){ return bits; }
inline int   tq_b2(int bits){ return bits>1?bits-1:1; }
inline uint  tq_enc1(float ang,int b){ float u=(ang+TQ_PI)*(0.5f/TQ_PI); int m=(1<<b)-1; int q=(int)floor(u*(float)(1<<b)); return (uint)clamp(q,0,m); }
inline float tq_dec1(uint q,int b){ return -TQ_PI + ((float)q+0.5f)*(2.0f*TQ_PI)/(float)(1<<b); }
inline float tq_hw(int level){ float s=(float)(1u<<(level-1)); float w=1.5f/sqrt(s); float cap=0.25f*TQ_PI; return w<cap?w:cap; }
inline uint  tq_enc2(float ang,int b,int level){ float w=tq_hw(level); float lo=0.25f*TQ_PI-w; float u=(ang-lo)/(2.0f*w); int m=(1<<b)-1; int q=(int)floor(u*(float)(1<<b)); return (uint)clamp(q,0,m); }
inline float tq_dec2(uint q,int b,int level){ float w=tq_hw(level); float lo=0.25f*TQ_PI-w; return lo+((float)q+0.5f)*(2.0f*w)/(float)(1<<b); }
inline void  tq_wbits(device uchar* buf,thread int* pos,uint val,int w){ for(int k=0;k<w;k++){ int bit=(*pos)+k; if((val>>k)&1u) buf[bit>>3]|=(uchar)(1u<<(bit&7)); } *pos+=w; }
inline uint  tq_rbits(device const uchar* buf,thread int* pos,int w){ uint v=0; for(int k=0;k<w;k++){ int bit=(*pos)+k; if((buf[bit>>3]>>(bit&7))&1u) v|=(1u<<k); } *pos+=w; return v; }
// producer: quantize the S new rows (contiguous f32 scratch) into the byte cache at pos_base + radius.
kernel void a_tq_enc(device const float* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                     device float* rad [[buffer(2)]], constant int& n [[buffer(3)]], constant int& bits [[buffer(4)]],
                     constant int& pb [[buffer(5)]], constant int& rowbytes [[buffer(6)]], constant int& codec [[buffer(7)]],
                     uint row [[thread_position_in_grid]]){
  device const float* xr=src+(long)row*n; device uchar* dr=dst+(long)(pb+(int)row)*rowbytes;
  for(int i=0;i<rowbytes;i++) dr[i]=0;
  float ss=0; for(int i=0;i<n;i++) ss+=xr[i]*xr[i]; float norm=sqrt(ss);
  if(!(norm>1e-35f) || norm>3.4e38f){ rad[pb+(int)row]=0.0f; return; }
  thread float r[512]; for(int i=0;i<n;i++) r[i]=xr[i]*tq_sign(i); tq_fwht(r,n);   // rotate
  if(codec==1){                                     // rotated int4 + Lloyd codebook
    float invstd=sqrt((float)n)/norm;
    for(int i=0;i<n;i++){ int c=tq_q4enc(r[i]*invstd); dr[i>>1]|=(uchar)(c<<((i&1)*4)); }
    rad[pb+(int)row]=norm; return;
  }
  int b1=tq_b1(bits), b2=tq_b2(bits), pos=0;
  for(int level=1, rlen=n; rlen>1; level++){ int hlf=rlen/2, b=(level==1)?b1:b2;   // 'half' is an MSL type
    for(int j=0;j<hlf;j++){ float a=r[2*j], c=r[2*j+1]; float ang=atan2(c,a);
      uint code=(level==1)?tq_enc1(ang,b):tq_enc2(ang,b,level); tq_wbits(dr,&pos,code,b); r[j]=sqrt(a*a+c*c); }
    rlen=hlf; }
  rad[pb+(int)row]=norm;
}
// consumer: reconstruct T rows from byte cache + radius into a contiguous f32 stage [T,n].
kernel void a_tq_deq(device const uchar* src [[buffer(0)]], device const float* rad [[buffer(1)]],
                     device float* dst [[buffer(2)]], constant int& n [[buffer(3)]], constant int& bits [[buffer(4)]],
                     constant int& rowbytes [[buffer(5)]], constant int& codec [[buffer(6)]], uint row [[thread_position_in_grid]]){
  device const uchar* sr=src+(long)row*rowbytes; device float* dr=dst+(long)row*n;
  float radius=rad[row];
  if(!(radius>0.0f) || radius>3.4e38f){ for(int i=0;i<n;i++) dr[i]=0.0f; return; }
  if(codec==1){                                     // rotated int4 + Lloyd codebook
    float st=radius/sqrt((float)n); thread float y[512];
    for(int i=0;i<n;i++){ int c=(sr[i>>1]>>((i&1)*4))&0xF; y[i]=TQ_Q4LEV[c]*st; }
    tq_fwht(y,n); for(int i=0;i<n;i++) dr[i]=y[i]*tq_sign(i); return;   // unrotate
  }
  int b1=tq_b1(bits), b2=tq_b2(bits), pos=0; thread float ang[512];
  for(int level=1, len=n/2, off=0; len>=1; level++){ int b=(level==1)?b1:b2;
    for(int j=0;j<len;j++){ uint code=tq_rbits(sr,&pos,b); ang[off+j]=(level==1)?tq_dec1(code,b):tq_dec2(code,b,level); }
    off+=len; if(len==1) break; len>>=1; }
  thread float cur[512], nxt[512]; cur[0]=radius;
  for(int clen=1; clen<n; clen*=2){ int offb=n-2*clen;
    for(int j=0;j<clen;j++){ float aa=ang[offb+j]; nxt[2*j]=cur[j]*cos(aa); nxt[2*j+1]=cur[j]*sin(aa); }
    for(int j=0;j<2*clen;j++) cur[j]=nxt[j]; }
  tq_fwht(cur,n); for(int i=0;i<n;i++) dr[i]=cur[i]*tq_sign(i);   // unrotate
}

// ===== KV_TQ codec-1 NATIVE fused attention (no f32 restage) =====================
// The rotated-int4 codec reconstructs x_hat = unrotate(c), c_i = TQ_Q4LEV[code_i]*std,
// std = radius/sqrt(n). Because the randomized-Hadamard rotate/unrotate are ADJOINT
// orthonormal maps, score and context need NO per-row dequant:
//   score:   q . x_hat            == rotate(q) . c
//   context: sum_t w_t . x_hat_t  == unrotate(sum_t w_t . c_t)
// So we rotate the query ONCE per head (a_tq_qrot), dot it against the packed nibbles
// through TQ_Q4LEV (a_score_tq), accumulate weighted codebook values (a_clat_tq), and
// unrotate the accumulator ONCE (a_tq_unrot) — the KV8 a_score8/a_clat8 shape, at 4 bits.
// This replaces the 2*T serial per-row FWHTs of a_tq_deq with 2 FWHTs per head per token.
// (Proven by test_kv_tq.c's score/context orthogonality identity; codec-0 keeps a_tq_deq.)

// threadgroup-parallel FWHT of one power-of-two row already resident in shared memory.
inline void tq_fwht_tg(threadgroup float* sh, int n, uint lid, uint tgsz){
  for(int len=1; len<n; len<<=1){
    for(int k=(int)lid; k<(n>>1); k+=(int)tgsz){ int off=k%len, blk=k/len, j=blk*(len<<1)+off;
      float u=sh[j], v=sh[j+len]; sh[j]=u+v; sh[j+len]=u-v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inv=1.0f/sqrt((float)n); for(int i=(int)lid;i<n;i+=(int)tgsz) sh[i]*=inv;   // normalized FWHT
}
// rotate one query row per threadgroup: qt = fwht(sign .* q). Source row `row` starts at
// src + row*srcstride + srcbase (n contiguous); dst row is dst + row*n (compact).
kernel void a_tq_qrot(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                      constant int& n [[buffer(2)]], constant int& srcstride [[buffer(3)]], constant int& srcbase [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]){
  threadgroup float sh[512];
  device const float* q=src+(long)row*srcstride+srcbase;
  for(int i=(int)lid;i<n;i+=(int)tgsz) sh[i]=q[i]*tq_sign(i);            // sign flip (D)
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_fwht_tg(sh,n,lid,tgsz);                                            // H
  for(int i=(int)lid;i<n;i+=(int)tgsz) dst[(long)row*n+i]=sh[i];
}
// score with packed-nibble L/R rows + per-row radius: a = stdL*sum(qtl*lev[L]) + stdR*sum(qtr*lev[R]).
// qtl/qtr are the pre-rotated queries; lrb/rrb are the packed row-byte strides (codec-1: n/2).
kernel void a_score_tq(device const float* qtl [[buffer(0)]], device const uchar* Lc [[buffer(1)]],
                       device const float* Lrad [[buffer(2)]], device const uchar* Rc [[buffer(3)]],
                       device const float* Rrad [[buffer(4)]], device const float* qtr [[buffer(5)]],
                       device float* sc [[buffer(6)]], constant int& T [[buffer(7)]], constant float& ascale [[buffer(8)]],
                       constant int& PB [[buffer(9)]], constant int& lrb [[buffer(10)]], constant int& rrb [[buffer(11)]],
                       uint gid [[thread_position_in_grid]]){
  int s=gid/(A_H*T), r=gid%(A_H*T), h=r/T, t=r%T; long o=(long)(s*A_H+h)*T+t;
  if(t > PB+s){ sc[o]=-1e30f; return; }
  device const float* ql=qtl+(long)(s*A_H+h)*A_KVL; device const uchar* Lt=Lc+(long)t*lrb;
  device const float* qr=qtr+(long)(s*A_H+h)*A_ROPE; device const uchar* Rt=Rc+(long)t*rrb;
  float al=0,ar=0;
  for(int i=0;i<A_KVL;i++){ int c=(Lt[i>>1]>>((i&1)*4))&0xF; al+=ql[i]*TQ_Q4LEV[c]; }
  for(int d=0;d<A_ROPE;d++){ int c=(Rt[d>>1]>>((d&1)*4))&0xF; ar+=qr[d]*TQ_Q4LEV[c]; }
  float stdL=Lrad[t]*(1.0f/sqrt((float)A_KVL)), stdR=Rrad[t]*(1.0f/sqrt((float)A_ROPE));
  sc[o]=(al*stdL+ar*stdR)*ascale;
}
// context accumulation (pre-unrotate): acc[i] = sum_t sc[t]*stdL[t]*lev[L[t][i]]. a_tq_unrot
// then turns acc into clat = unrotate(acc), reusing the aclat_ buffer in place.
kernel void a_clat_tq(device const float* sc [[buffer(0)]], device const uchar* Lc [[buffer(1)]],
                      device const float* Lrad [[buffer(2)]], device float* acc [[buffer(3)]],
                      constant int& T [[buffer(4)]], constant int& lrb [[buffer(5)]], uint gid [[thread_position_in_grid]]){
  int sh=gid/A_KVL, i=gid%A_KVL; device const float* s=sc+(long)sh*T; float a=0; float invsn=1.0f/sqrt((float)A_KVL);
  for(int t=0;t<T;t++){ int c=(Lc[(long)t*lrb+(i>>1)]>>((i&1)*4))&0xF; a+=s[t]*(Lrad[t]*invsn)*TQ_Q4LEV[c]; }
  acc[(long)sh*A_KVL+i]=a;
}
// unrotate the accumulator in place: clat = D (H acc). One threadgroup per (s,head) row.
kernel void a_tq_unrot(device float* clat [[buffer(0)]], constant int& n [[buffer(1)]],
                       uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]){
  threadgroup float sh[512];
  device float* c=clat+(long)row*n;
  for(int i=(int)lid;i<n;i+=(int)tgsz) sh[i]=c[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_fwht_tg(sh,n,lid,tgsz);                                            // H
  for(int i=(int)lid;i<n;i+=(int)tgsz) c[i]=sh[i]*tq_sign(i);           // D
}
// codec-1 producer, threadgroup-per-row (replaces the serial a_tq_enc at decode): parallel
// norm reduction + FWHT, each thread packs one byte (two nibbles). Twin of coli_q4_quant_row.
kernel void a_tq_enc1(device const float* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                      device float* rad [[buffer(2)]], constant int& n [[buffer(3)]], constant int& pb [[buffer(4)]], constant int& rowbytes [[buffer(5)]],
                      uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]){
  threadgroup float sh[512]; threadgroup float red[256];
  device const float* xr=src+(long)row*n;
  float ss=0; for(int i=(int)lid;i<n;i+=(int)tgsz){ float v=xr[i]; ss+=v*v; }
  red[lid]=ss; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float norm=sqrt(red[0]); device uchar* dr=dst+(long)(pb+(int)row)*rowbytes;
  if(!(norm>1e-35f) || norm>3.4e38f){ for(int i=(int)lid;i<rowbytes;i+=(int)tgsz) dr[i]=0; if(lid==0) rad[pb+(int)row]=0.0f; return; }
  for(int i=(int)lid;i<n;i+=(int)tgsz) sh[i]=xr[i]*tq_sign(i);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tq_fwht_tg(sh,n,lid,tgsz);
  float invstd=sqrt((float)n)/norm;
  for(int b=(int)lid;b<(n>>1);b+=(int)tgsz){ int c0=tq_q4enc(sh[2*b]*invstd), c1=tq_q4enc(sh[2*b+1]*invstd); dr[b]=(uchar)(c0|(c1<<4)); }
  if(lid==0) rad[pb+(int)row]=norm;
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
static id<MTLComputePipelineState> g_a_add, g_r_router, g_r_top8, g_r_top8p;
static int g_rtop8_par = 1;      // COLI_RTOP8 (default ON); COLI_RTOP8=0 opts out to the
                                  // serial kernel — see coli_metal_init.
static int g_rtop8_width_ok = 1; // hardware fact, independent of the policy gate above:
                                  // false if this device's threadExecutionWidth != 32.
                                  // Consulted by BOTH the engine dispatch site and the
                                  // standalone coli_metal_rtop8 runner, so no caller can
                                  // reach r_top8_par's 32-lane reduction on an unsafe
                                  // device even by explicitly requesting par=1.
static id<MTLComputePipelineState> g_a_fp8enc, g_a_score8, g_a_clat8;   // KV8 fp8 attention
static id<MTLComputePipelineState> g_a_tq_enc, g_a_tq_deq;              // KV_TQ codec-0 (PolarQuant): serial enc + staging deq
static id<MTLComputePipelineState> g_a_tq_enc1, g_a_tq_qrot, g_a_score_tq, g_a_clat_tq, g_a_tq_unrot; // KV_TQ codec-1 native fused
static id<MTLBuffer> g_fp8lut;                                          // 256-entry e4m3 decode LUT (twin of coli_fp8_lut)
// PolarQuant packed-row byte count (host twin of coli_tq_row_bytes): (n/2)*b1 + (n/2-1)*b2 bits.
static inline int tq_row_bytes_h(int n,int bits,int codec){ if(codec) return n/2;   /* rotated int4 */
  int b2=bits>1?bits-1:1; long t=(long)(n/2)*bits+(long)(n/2-1)*b2; return (int)((t+7)/8); }
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
// SUMMARY.md for why skipping useResource: there is safe (read-only, indirectly-referenced
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
  if (getenv("COLI_METAL_UNTRACKED") && atoi(getenv("COLI_METAL_UNTRACKED")))
    g_res_opts = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
  { const char *e = getenv("COLI_RTOP8");           // default ON; COLI_RTOP8=0 opts out
    if (e && atoi(e) == 0) g_rtop8_par = 0; }
  @autoreleasepool {
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) return 0;
    g_queue = [g_dev newCommandQueue];
    NSError *err = nil;
    // TWO libraries from the same source. The KV quantizers (a_fp8enc RNE e4m3; the TQ
    // codec encoders) must round bit-for-bit like their CPU twins, so they compile with
    // fast-math OFF (no rint/frexp reassociation). EVERYTHING else — the f32/MoE kernels and
    // all quantized CONSUMERS — compiles fast-math ON, exactly as the merge-base did: a
    // whole-library MTLMathModeSafe flip silently changed f32 numerics (mm_gemv alone drifts
    // ~4e-6, compounding to flip greedy argmax at the first decode token vs merge-base).
    MTLCompileOptions *fastOpts = [MTLCompileOptions new];   // fast-math ON (== merge-base options:nil)
    MTLCompileOptions *safeOpts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) safeOpts.mathMode = MTLMathModeSafe;   // exact rint/frexp for the encoders
    else safeOpts.fastMathEnabled = NO;
    NSString *src = [NSString stringWithUTF8String:SHADER];
    id<MTLLibrary> libFast = [g_dev newLibraryWithSource:src options:fastOpts error:&err];
    if (!libFast) { fprintf(stderr, "[metal] shader compile failed (fast): %s\n",
                        err ? [[err localizedDescription] UTF8String] : "?"); g_dev = nil; return 0; }
    id<MTLLibrary> libSafe = [g_dev newLibraryWithSource:src options:safeOpts error:&err];
    if (!libSafe) { fprintf(stderr, "[metal] shader compile failed (safe): %s\n",
                        err ? [[err localizedDescription] UTF8String] : "?"); g_dev = nil; return 0; }
    auto P=[&](id<MTLLibrary> lib,const char*n){ return [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:&err]; };
    g_gemv=P(libFast,"mm_gemv"); g_moe_gemv=P(libFast,"moe_gemv"); g_moe_silu=P(libFast,"moe_silu");
    g_a_rms=P(libFast,"a_rmsnorm"); g_a_rope=P(libFast,"a_rope"); g_a_copy=P(libFast,"a_copy");
    g_a_qabs=P(libFast,"a_qabs"); g_a_score=P(libFast,"a_score"); g_a_smax=P(libFast,"a_smax"); g_a_clat=P(libFast,"a_clat"); g_a_ctx=P(libFast,"a_ctx");
    g_a_add=P(libFast,"a_add"); g_r_router=P(libFast,"r_router"); g_r_top8=P(libFast,"r_top8"); g_r_top8p=P(libFast,"r_top8_par");
    g_a_score8=P(libFast,"a_score8"); g_a_clat8=P(libFast,"a_clat8");
    g_a_tq_deq=P(libFast,"a_tq_deq"); g_a_tq_qrot=P(libFast,"a_tq_qrot");
    g_a_score_tq=P(libFast,"a_score_tq"); g_a_clat_tq=P(libFast,"a_clat_tq"); g_a_tq_unrot=P(libFast,"a_tq_unrot");
    g_a_fp8enc=P(libSafe,"a_fp8enc");
    g_a_tq_enc=P(libSafe,"a_tq_enc"); g_a_tq_enc1=P(libSafe,"a_tq_enc1");
    if(!g_a_add||!g_r_router||!g_r_top8||!g_r_top8p){ fprintf(stderr,"[metal] tail pipelines failed\n"); g_dev=nil; return 0; }
    if(!g_a_fp8enc||!g_a_score8||!g_a_clat8){ fprintf(stderr,"[metal] fp8 attn pipelines failed\n"); g_dev=nil; return 0; }
    if(!g_a_tq_enc||!g_a_tq_deq||!g_a_tq_enc1||!g_a_tq_qrot||!g_a_score_tq||!g_a_clat_tq||!g_a_tq_unrot){ fprintf(stderr,"[metal] tq attn pipelines failed\n"); g_dev=nil; return 0; }
    if ([g_r_top8p threadExecutionWidth] != 32) {
      g_rtop8_width_ok = 0;
      if (g_rtop8_par)
        fprintf(stderr, "[metal] COLI_RTOP8 parallel top-8 disabled: threadExecutionWidth=%lu != 32 "
                        "(r_top8_par's reduction assumes 32-lane simdgroups) - serial r_top8 in use\n",
                        (unsigned long)[g_r_top8p threadExecutionWidth]);
      g_rtop8_par = 0;
    }
    if (!g_gemv || !g_moe_gemv || !g_moe_silu || !g_a_rms || !g_a_rope || !g_a_copy ||
        !g_a_qabs || !g_a_score || !g_a_smax || !g_a_clat || !g_a_ctx) {
      fprintf(stderr, "[metal] pipeline failed\n"); g_dev = nil; return 0; }
    if (getenv("COLI_METAL_RESSET") && atoi(getenv("COLI_METAL_RESSET"))) {
#if COLI_HAS_RESSET
      if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.initialCapacity = 4096;
        NSError *rerr = nil;
        id<MTLResidencySet> rs = [g_dev newResidencySetWithDescriptor:rd error:&rerr];
        if (rs) {
          [g_queue addResidencySet:rs];
          g_resset_obj = rs; g_resset_enabled = true;
          fprintf(stderr, "[METAL] residency-set: on (macOS 15+, moe_submit skips per-buffer useResource:)\n");
        } else {
          fprintf(stderr, "[METAL] residency-set create failed: %s - stock per-CB residency path\n",
                  rerr ? [[rerr localizedDescription] UTF8String] : "?");
        }
      } else {
        fprintf(stderr, "[METAL] COLI_METAL_RESSET=1 requested but OS < macOS 15 - stock per-CB residency path\n");
      }
#else
      fprintf(stderr, "[METAL] COLI_METAL_RESSET=1 requested but this binary was built against a "
                      "macOS SDK < 15 (MTLResidencySet unavailable) -- stock per-CB residency path\n");
#endif
    }
    // e4m3 decode LUT, byte-identical to coli_fp8_lut_init (kv_fp8.h): lut[b] = value of code b.
    g_fp8lut = [g_dev newBufferWithLength:256*4 options:MTLResourceStorageModeShared];
    { float *L=(float*)[g_fp8lut contents];
      for(int b=0;b<256;b++){ int E=(b>>3)&0xF, M=b&7;
        float v = E ? ldexpf(1.f+(float)M/8.f, E-7) : ldexpf((float)M,-9);
        if(E==15&&M==7) v=0.f; L[b]=(b&0x80)?-v:v; } }
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
    int NT=S*O;
    [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5];
    [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
    [e setBytes:&NT length:4 atIndex:8];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(y, [by contents], (size_t)S*O*sizeof(float));
  }
  return 1;
}

// ---- fused decode attention scratch (GLM-5.2 dims) ----
enum { AH=6144, AHEADS=64, AQLORA=2048, AKVL=512, AROPE=64, AVH=256, AQH=256, ANOPE=192, AROWSH=448, AHQH=AHEADS*AQH, AHVH=AHEADS*AVH, AMAXS=4 };
static id<MTLBuffer> ax_,aqr_,aqf_,acomp_,aqabs_,ascore_,aclat_,actx_,aout_,aqaln_,akvaln_; static size_t ascore_cap;
static id<MTLBuffer> axr_,anrm_,ash1_,ash2_,ashout_,asig_,aidx_,aw_,akeff_;   // full-layer tail
static id<MTLBuffer> aLf_,aRf_;                                               // KV8/TQ producer f32 scratch (S rows)
static id<MTLBuffer> aLstg_,aRstg_; static size_t aLstg_cap,aRstg_cap;        // KV_TQ codec-0 consumer f32 stage (T rows)
static id<MTLBuffer> aqtl_,aqtr_;                                             // KV_TQ codec-1 rotated query (latent/rope)
static int g_tq_force_stage=0;                                                // test/bench hook: force codec-1 down the staging path
extern "C" void coli_metal_tq_force_stage(int on){ g_tq_force_stage=on?1:0; } // native-vs-staging A/B in one process
static void attn_scratch_init(){
  if(ax_) return;
  auto L=[&](size_t n){ return [g_dev newBufferWithLength:n*AMAXS options:g_res_opts]; };
  ax_=L(AH*4); aqr_=L(AQLORA*4); aqf_=L(AHQH*4); acomp_=L((AKVL+AROPE)*4);
  aLf_=L(AKVL*4); aRf_=L(AROPE*4);
  aqtl_=L((size_t)AHEADS*AKVL*4); aqtr_=L((size_t)AHEADS*AROPE*4);
  aqabs_=L((size_t)AHEADS*AKVL*4); aclat_=L((size_t)AHEADS*AKVL*4); actx_=L(AHVH*4); aout_=L(AH*4);
  aqaln_=L(AQLORA*4/AMAXS); akvaln_=L(AKVL*4/AMAXS);   // norm weights are per-tensor, not per-row
  axr_=L(AH*4); anrm_=L(AH*4); ash1_=L(2048*4); ash2_=L(2048*4); ashout_=L(AH*4);
  asig_=L(256*4); aidx_=L(8*4); aw_=L(8*4); akeff_=L(4);
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
  int NT=S*O;
  [e setBytes:&S length:4 atIndex:4]; [e setBytes:&I length:4 atIndex:5]; [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
  [e setBytes:&NT length:4 atIndex:8];
  [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  return true;
}

// Weight-pointer bundle for one layer's attention (+optional layer tail). All pointers
// must be inside registered allocations.
typedef struct {
  const void *qa_w; const float *qa_s; int qa_fmt; const float *qa_ln;
  const void *qb_w; const float *qb_s; int qb_fmt;
  const void *kva_w; const float *kva_s; int kva_fmt; const float *kva_ln;
  const void *kvb_w; const float *kvb_s; int kvb_fmt;
  const void *o_w;  const float *o_s;  int o_fmt;
} AttnW;

// Encode the fused attention chain into encoder e. Input: ax_ holds the NORMED x [S,AH].
// Output: aout_ holds attention output [S,AH]. Returns false on unresolved weights.
static bool encode_attention(id<MTLComputeCommandEncoder> e, const AttnW *W,
                             id<MTLBuffer> Lb, size_t loff, id<MTLBuffer> Rb, size_t roff,
                             id<MTLBuffer> kvbW, size_t kvbwoff, id<MTLBuffer> kvbS, size_t kvbsoff,
                             int S, int pos_base, float eps, float theta, float ascale,
                             int qmode, int bits, int codec, id<MTLBuffer> L8, size_t l8off, id<MTLBuffer> R8, size_t r8off,
                             id<MTLBuffer> Ls, size_t lsoff, id<MTLBuffer> Rs, size_t rsoff,
                             id<MTLBuffer> Lstg, id<MTLBuffer> Rstg) {
    /* qmode: 0=f32 cache (Lb/Rb), 1=KV8 fp8 (L8/R8 bytes + Ls/Rs scale), 2=KV_TQ PolarQuant
     * (L8/R8 packed bytes + Ls/Rs radius; decoded to the f32 stage Lstg/Rstg then f32 score/clat). */
    int T=pos_base+S;
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
    bind_gemv(e,W->qa_w,W->qa_s,W->qa_fmt,AH,AQLORA,ax_,aqr_,S);
    bind_gemv(e,W->kva_w,W->kva_s,W->kva_fmt,AH,AKVL+AROPE,ax_,acomp_,S); BAR();
    // producer: normed latent L + roped k_rot. f32 -> straight into the cache; KV8 -> a f32
    // scratch (aLf_/aRf_), then fp8-quantized (per-row amax/448 scale) into the byte cache at pos_base.
    bool quant = (qmode!=0);
    bool tq_native = (qmode==2 && codec==1 && !g_tq_force_stage);   // rotated-int4: native fused score/clat, no restage
    id<MTLBuffer> Lp = quant?aLf_:Lb; size_t Lpo = quant?0:Loff;
    id<MTLBuffer> Rp = quant?aRf_:Rb; size_t Rpo = quant?0:Roff;
    rms(aqr_,0,aqaln_,AQLORA,S); cpy(0,Lp,Lpo,AKVL); cpy(AKVL,Rp,Rpo,AROPE); BAR();
    bind_gemv(e,W->qb_w,W->qb_s,W->qb_fmt,AQLORA,AHQH,aqr_,aqf_,S); rms(Lp,Lpo,akvaln_,AKVL,S); rope(Rp,Rpo,0,AROPE,0,1); BAR();
    if(qmode==1){
      auto enc=[&](id<MTLBuffer> src, id<MTLBuffer> dstB, size_t dstoff, id<MTLBuffer> sclB, size_t scloff, int n){
        [e setComputePipelineState:g_a_fp8enc];
        [e setBuffer:src offset:0 atIndex:0]; [e setBuffer:dstB offset:dstoff atIndex:1]; [e setBuffer:sclB offset:scloff atIndex:2];
        [e setBytes:&n length:4 atIndex:3]; [e setBytes:&pos_base length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
      enc(aLf_,L8,l8off,Ls,lsoff,AKVL); enc(aRf_,R8,r8off,Rs,rsoff,AROPE); BAR();
    } else if(tq_native){
      // codec-1 producer: one threadgroup per new row (parallel norm+FWHT+pack) — not the
      // single-thread serial a_tq_enc, which was a per-token latency floor at decode.
      auto tqenc1=[&](id<MTLBuffer> src, id<MTLBuffer> dstB, size_t dstoff, id<MTLBuffer> radB, size_t radoff, int n){
        int rbw=tq_row_bytes_h(n,bits,codec); [e setComputePipelineState:g_a_tq_enc1];
        [e setBuffer:src offset:0 atIndex:0]; [e setBuffer:dstB offset:dstoff atIndex:1]; [e setBuffer:radB offset:radoff atIndex:2];
        [e setBytes:&n length:4 atIndex:3]; [e setBytes:&pos_base length:4 atIndex:4]; [e setBytes:&rbw length:4 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
      tqenc1(aLf_,L8,l8off,Ls,lsoff,AKVL); tqenc1(aRf_,R8,r8off,Rs,rsoff,AROPE); BAR();
    } else if(qmode==2){
      auto tqenc=[&](id<MTLBuffer> src, id<MTLBuffer> dstB, size_t dstoff, id<MTLBuffer> radB, size_t radoff, int n){
        int rbw=tq_row_bytes_h(n,bits,codec); [e setComputePipelineState:g_a_tq_enc];
        [e setBuffer:src offset:0 atIndex:0]; [e setBuffer:dstB offset:dstoff atIndex:1]; [e setBuffer:radB offset:radoff atIndex:2];
        [e setBytes:&n length:4 atIndex:3]; [e setBytes:&bits length:4 atIndex:4]; [e setBytes:&pos_base length:4 atIndex:5]; [e setBytes:&rbw length:4 atIndex:6]; [e setBytes:&codec length:4 atIndex:7];
        [e dispatchThreads:MTLSizeMake(S,1,1) threadsPerThreadgroup:MTLSizeMake(S,1,1)]; };
      tqenc(aLf_,L8,l8off,Ls,lsoff,AKVL); tqenc(aRf_,R8,r8off,Rs,rsoff,AROPE); BAR();
    }
    rope(aqf_,0,ANOPE,AHQH,AQH,AHEADS); BAR();
    [e setComputePipelineState:g_a_qabs]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aqf_ offset:0 atIndex:2]; [e setBuffer:aqabs_ offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    // KV_TQ codec-0 (PolarQuant): reconstruct ALL T rows to the f32 stage, then f32 score/clat.
    // codec-1 (rotated int4): NO restage — rotate the query once, then dot packed nibbles natively.
    if(qmode==2 && !tq_native){
      auto tqdeq=[&](id<MTLBuffer> srcB, size_t srcoff, id<MTLBuffer> radB, size_t radoff, id<MTLBuffer> stg, int n){
        int rbw=tq_row_bytes_h(n,bits,codec); [e setComputePipelineState:g_a_tq_deq];
        [e setBuffer:srcB offset:srcoff atIndex:0]; [e setBuffer:radB offset:radoff atIndex:1]; [e setBuffer:stg offset:0 atIndex:2];
        [e setBytes:&n length:4 atIndex:3]; [e setBytes:&bits length:4 atIndex:4]; [e setBytes:&rbw length:4 atIndex:5]; [e setBytes:&codec length:4 atIndex:6];
        [e dispatchThreads:MTLSizeMake(T,1,1) threadsPerThreadgroup:MTLSizeMake(T<64?T:64,1,1)]; };
      tqdeq(L8,l8off,Ls,lsoff,Lstg,AKVL); tqdeq(R8,r8off,Rs,rsoff,Rstg,AROPE); BAR();
    } else if(tq_native){
      // q~ = rotate(q): latent from aqabs_ (contiguous, stride AKVL), rope from aqf_ (stride AQH, base ANOPE).
      auto qrot=[&](id<MTLBuffer> srcB, id<MTLBuffer> dstB, int n, int stride, int base){
        [e setComputePipelineState:g_a_tq_qrot];
        [e setBuffer:srcB offset:0 atIndex:0]; [e setBuffer:dstB offset:0 atIndex:1];
        [e setBytes:&n length:4 atIndex:2]; [e setBytes:&stride length:4 atIndex:3]; [e setBytes:&base length:4 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((size_t)S*AHEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
      qrot(aqabs_,aqtl_,AKVL,AKVL,0); qrot(aqf_,aqtr_,AROPE,AQH,ANOPE); BAR();
    }
    id<MTLBuffer> CL=(qmode==2)?Lstg:Lb; size_t CLo=(qmode==2)?0:loff;
    id<MTLBuffer> CR=(qmode==2)?Rstg:Rb; size_t CRo=(qmode==2)?0:roff;
    int lrb=tq_row_bytes_h(AKVL,bits,codec), rrb=tq_row_bytes_h(AROPE,bits,codec);
    if(tq_native){
      [e setComputePipelineState:g_a_score_tq]; [e setBuffer:aqtl_ offset:0 atIndex:0];
      [e setBuffer:L8 offset:l8off atIndex:1]; [e setBuffer:Ls offset:lsoff atIndex:2];
      [e setBuffer:R8 offset:r8off atIndex:3]; [e setBuffer:Rs offset:rsoff atIndex:4];
      [e setBuffer:aqtr_ offset:0 atIndex:5]; [e setBuffer:ascore_ offset:0 atIndex:6];
      [e setBytes:&T length:4 atIndex:7]; [e setBytes:&ascale length:4 atIndex:8]; [e setBytes:&pos_base length:4 atIndex:9];
      [e setBytes:&lrb length:4 atIndex:10]; [e setBytes:&rrb length:4 atIndex:11];
    } else if(qmode==1){
      [e setComputePipelineState:g_a_score8]; [e setBuffer:aqabs_ offset:0 atIndex:0];
      [e setBuffer:L8 offset:l8off atIndex:1]; [e setBuffer:Ls offset:lsoff atIndex:2];
      [e setBuffer:R8 offset:r8off atIndex:3]; [e setBuffer:Rs offset:rsoff atIndex:4];
      [e setBuffer:aqf_ offset:0 atIndex:5]; [e setBuffer:g_fp8lut offset:0 atIndex:6]; [e setBuffer:ascore_ offset:0 atIndex:7];
      [e setBytes:&T length:4 atIndex:8]; [e setBytes:&ascale length:4 atIndex:9]; [e setBytes:&pos_base length:4 atIndex:10];
    } else {
      [e setComputePipelineState:g_a_score]; [e setBuffer:aqabs_ offset:0 atIndex:0]; [e setBuffer:CL offset:CLo atIndex:1]; [e setBuffer:CR offset:CRo atIndex:2]; [e setBuffer:aqf_ offset:0 atIndex:3]; [e setBuffer:ascore_ offset:0 atIndex:4];
      [e setBytes:&T length:4 atIndex:5]; [e setBytes:&ascale length:4 atIndex:6]; [e setBytes:&pos_base length:4 atIndex:7];
    }
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*T,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    [e setComputePipelineState:g_a_smax]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBytes:&T length:4 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake((size_t)S*AHEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    if(tq_native){
      // acc = sum_t w_t*std_t*lev[L[t]] into aclat_, then unrotate aclat_ in place -> clat.
      [e setComputePipelineState:g_a_clat_tq]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBuffer:L8 offset:l8off atIndex:1]; [e setBuffer:Ls offset:lsoff atIndex:2]; [e setBuffer:aclat_ offset:0 atIndex:3]; [e setBytes:&T length:4 atIndex:4]; [e setBytes:&lrb length:4 atIndex:5];
      [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
      int nkvl=AKVL; [e setComputePipelineState:g_a_tq_unrot]; [e setBuffer:aclat_ offset:0 atIndex:0]; [e setBytes:&nkvl length:4 atIndex:1];
      [e dispatchThreadgroups:MTLSizeMake((size_t)S*AHEADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    } else {
      if(qmode==1){
        [e setComputePipelineState:g_a_clat8]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBuffer:L8 offset:l8off atIndex:1]; [e setBuffer:Ls offset:lsoff atIndex:2]; [e setBuffer:g_fp8lut offset:0 atIndex:3]; [e setBuffer:aclat_ offset:0 atIndex:4]; [e setBytes:&T length:4 atIndex:5];
      } else {
        [e setComputePipelineState:g_a_clat]; [e setBuffer:ascore_ offset:0 atIndex:0]; [e setBuffer:CL offset:CLo atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBytes:&T length:4 atIndex:3];
      }
      [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AKVL,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    }
    [e setComputePipelineState:g_a_ctx]; [e setBuffer:kvbW offset:kvbwoff atIndex:0]; [e setBuffer:kvbS offset:kvbsoff atIndex:1]; [e setBuffer:aclat_ offset:0 atIndex:2]; [e setBuffer:actx_ offset:0 atIndex:3];
    [e dispatchThreads:MTLSizeMake((size_t)S*AHEADS*AVH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    bind_gemv(e,W->o_w,W->o_s,W->o_fmt,AHVH,AH,actx_,aout_,S);
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
    AttnW W={qa_w,qa_s,qa_fmt,qa_ln,qb_w,qb_s,qb_fmt,kva_w,kva_s,kva_fmt,kva_ln,kvb_w,kvb_s,kvb_fmt,o_w,o_s,o_fmt};
    id<MTLBuffer> Lb,Rb,kvbW,kvbS; size_t loff,roff,kvbwoff,kvbsoff;
    if(!resolve_attn(&W,Lc,Rc,&Lb,&loff,&Rb,&roff,&kvbW,&kvbwoff,&kvbS,&kvbsoff)) return 0;
    ascore_=ensure(ascore_,&ascore_cap,(size_t)S*AHEADS*T*4);
    memcpy([ax_ contents],x,(size_t)S*AH*4);
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:Lb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rb usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    if(!encode_attention(e,&W,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,S,pos_base,eps,theta,ascale,
                         0,0,0,nil,0,nil,0,nil,0,nil,0,nil,nil)) return 0;
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] attn cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    g_attn_sched += [cb GPUStartTime]-[cb kernelStartTime]; g_attn_ksched += [cb kernelStartTime]-tc;
    memcpy(out,[aout_ contents],(size_t)S*AH*4);
  }
  return 1;
}

// KV8 twin of coli_metal_attn_decode: latent KV cache is fp8 bytes (Lc8/Rc8) + per-row
// f32 scale (Lsc/Rsc). The producer fp8-quantizes the new rows on the GPU; the consumer
// decodes via the e4m3 LUT + scale. All four cache arrays must be registered (page-aligned).
extern "C" int coli_metal_attn_decode8(const float* x,
    const void* qa_w,const float* qa_s,int qa_fmt,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,
    const void* kva_w,const float* kva_s,int kva_fmt,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,
    const void* o_w,const float* o_s,int o_fmt,
    const unsigned char* Lc8,const float* Lsc,const unsigned char* Rc8,const float* Rsc,
    int S,int pos_base,int st0,float eps,float theta,float ascale,float* out){
  if(!g_dev) return 0;
  if(st0!=0 || S<1 || S>AMAXS) return 0;     // partial-KV / S>4 -> CPU
  int T=pos_base+S;
  @autoreleasepool {
    attn_scratch_init();
    AttnW W={qa_w,qa_s,qa_fmt,qa_ln,qb_w,qb_s,qb_fmt,kva_w,kva_s,kva_fmt,kva_ln,kvb_w,kvb_s,kvb_fmt,o_w,o_s,o_fmt};
    uint64_t la=0,ra=0,lsa=0,rsa=0,kwa=0,ksa=0;
    id<MTLBuffer> L8=resolve(Lc8,&la), R8=resolve(Rc8,&ra), Ls=resolve(Lsc,&lsa), Rs=resolve(Rsc,&rsa);
    id<MTLBuffer> kvbW=resolve(kvb_w,&kwa), kvbS=resolve(kvb_s,&ksa);
    if(!L8||!R8||!Ls||!Rs||!kvbW||!kvbS) return 0;
    uint64_t d; const void* ws[]={qa_w,qa_s,qb_w,qb_s,kva_w,kva_s,o_w,o_s};
    for(auto p:ws) if(!resolve(p,&d)) return 0;
    size_t l8off=la-(uint64_t)[L8 gpuAddress], r8off=ra-(uint64_t)[R8 gpuAddress];
    size_t lsoff=lsa-(uint64_t)[Ls gpuAddress], rsoff=rsa-(uint64_t)[Rs gpuAddress];
    size_t kvbwoff=kwa-(uint64_t)[kvbW gpuAddress], kvbsoff=ksa-(uint64_t)[kvbS gpuAddress];
    ascore_=ensure(ascore_,&ascore_cap,(size_t)S*AHEADS*T*4);
    memcpy([ax_ contents],x,(size_t)S*AH*4);
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:L8 usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:R8 usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:Ls usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rs usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    if(!encode_attention(e,&W,nil,0,nil,0,kvbW,kvbwoff,kvbS,kvbsoff,S,pos_base,eps,theta,ascale,
                         1,0,0,L8,l8off,R8,r8off,Ls,lsoff,Rs,rsoff,nil,nil)) return 0;
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] attn8 cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    g_attn_sched += [cb GPUStartTime]-[cb kernelStartTime]; g_attn_ksched += [cb kernelStartTime]-tc;
    memcpy(out,[aout_ contents],(size_t)S*AH*4);
  }
  return 1;
}

// KV_TQ (PolarQuant) twin of the fused decode: latent KV is packed polar bytes (Lc8/Rc8) +
// per-row radius (Lsc/Rsc). The GPU rotate+polar-encodes the new rows, reconstructs all T rows
// to an f32 stage, then runs the f32 score/clat. All four cache arrays must be registered.
extern "C" int coli_metal_attn_decode_tq(const float* x,
    const void* qa_w,const float* qa_s,int qa_fmt,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,
    const void* kva_w,const float* kva_s,int kva_fmt,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,
    const void* o_w,const float* o_s,int o_fmt,
    const unsigned char* Lc8,const float* Lsc,const unsigned char* Rc8,const float* Rsc,int bits,int codec,
    int S,int pos_base,int st0,float eps,float theta,float ascale,float* out){
  if(!g_dev) return 0;
  if(st0!=0 || S<1 || S>AMAXS) return 0;
  int T=pos_base+S;
  @autoreleasepool {
    attn_scratch_init();
    AttnW W={qa_w,qa_s,qa_fmt,qa_ln,qb_w,qb_s,qb_fmt,kva_w,kva_s,kva_fmt,kva_ln,kvb_w,kvb_s,kvb_fmt,o_w,o_s,o_fmt};
    uint64_t la=0,ra=0,lsa=0,rsa=0,kwa=0,ksa=0;
    id<MTLBuffer> L8=resolve(Lc8,&la), R8=resolve(Rc8,&ra), Ls=resolve(Lsc,&lsa), Rs=resolve(Rsc,&rsa);
    id<MTLBuffer> kvbW=resolve(kvb_w,&kwa), kvbS=resolve(kvb_s,&ksa);
    if(!L8||!R8||!Ls||!Rs||!kvbW||!kvbS) return 0;
    uint64_t d; const void* ws[]={qa_w,qa_s,qb_w,qb_s,kva_w,kva_s,o_w,o_s};
    for(auto p:ws) if(!resolve(p,&d)) return 0;
    size_t l8off=la-(uint64_t)[L8 gpuAddress], r8off=ra-(uint64_t)[R8 gpuAddress];
    size_t lsoff=lsa-(uint64_t)[Ls gpuAddress], rsoff=rsa-(uint64_t)[Rs gpuAddress];
    size_t kvbwoff=kwa-(uint64_t)[kvbW gpuAddress], kvbsoff=ksa-(uint64_t)[kvbS gpuAddress];
    ascore_=ensure(ascore_,&ascore_cap,(size_t)S*AHEADS*T*4);
    aLstg_=ensure(aLstg_,&aLstg_cap,(size_t)T*AKVL*4);
    aRstg_=ensure(aRstg_,&aRstg_cap,(size_t)T*AROPE*4);
    memcpy([ax_ contents],x,(size_t)S*AH*4);
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e useResource:L8 usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:R8 usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:Ls usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rs usage:MTLResourceUsageRead|MTLResourceUsageWrite];
    [e useResource:kvbW usage:MTLResourceUsageRead]; [e useResource:kvbS usage:MTLResourceUsageRead];
    if(!encode_attention(e,&W,nil,0,nil,0,kvbW,kvbwoff,kvbS,kvbsoff,S,pos_base,eps,theta,ascale,
                         2,bits,codec,L8,l8off,R8,r8off,Ls,lsoff,Rs,rsoff,aLstg_,aRstg_)) return 0;
    double tc=mnow();
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError){ fprintf(stderr,"[metal] attn_tq cmdbuf error: %s\n", cb.error?[[cb.error localizedDescription]UTF8String]:"?"); return 0; }
    g_attn_ok++; g_attn_wall += mnow()-tc; g_attn_kernel += [cb GPUEndTime]-[cb GPUStartTime];
    g_attn_sched += [cb GPUStartTime]-[cb kernelStartTime]; g_attn_ksched += [cb kernelStartTime]-tc;
    memcpy(out,[aout_ contents],(size_t)S*AH*4);
  }
  return 1;
}

// Test hook: run the GPU fp8 per-row quantizer (a_fp8enc) on one host row and return
// bytes + scale, so metal-test can check byte-for-byte parity with coli_kv8_quant_row
// in isolation from the fused chain (whose pre-quant latent diverges ~1e-5 CPU vs GPU).
extern "C" int coli_metal_fp8_quant_row(const float* src, unsigned char* dst, float* scale, int n){
  if(!g_dev || n<1) return 0;
  @autoreleasepool {
    id<MTLBuffer> sb=[g_dev newBufferWithBytes:src length:(size_t)n*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> db=[g_dev newBufferWithLength:(size_t)n options:MTLResourceStorageModeShared];
    id<MTLBuffer> cb2=[g_dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    int pb=0; [e setComputePipelineState:g_a_fp8enc];
    [e setBuffer:sb offset:0 atIndex:0]; [e setBuffer:db offset:0 atIndex:1]; [e setBuffer:cb2 offset:0 atIndex:2];
    [e setBytes:&n length:4 atIndex:3]; [e setBytes:&pb length:4 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError) return 0;
    memcpy(dst,[db contents],(size_t)n); *scale=*(float*)[cb2 contents];
  }
  return 1;
}

// Test hook: GPU PolarQuant round-trip (a_tq_enc then a_tq_deq) for one row, so metal-test
// can check the MSL FWHT/polar math against kv_tq.h's coli_tq_quant_row/dequant_row.
extern "C" int coli_metal_tq_roundtrip(const float* src, float* out, int n, int bits, int codec){
  if(!g_dev || n<2) return 0;
  @autoreleasepool {
    int rbw=tq_row_bytes_h(n,bits,codec), pb=0;
    id<MTLBuffer> sb=[g_dev newBufferWithBytes:src length:(size_t)n*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> db=[g_dev newBufferWithLength:(size_t)rbw options:MTLResourceStorageModeShared];
    id<MTLBuffer> rb=[g_dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob=[g_dev newBufferWithLength:(size_t)n*4 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb=[g_queue commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:g_a_tq_enc];
    [e setBuffer:sb offset:0 atIndex:0]; [e setBuffer:db offset:0 atIndex:1]; [e setBuffer:rb offset:0 atIndex:2];
    [e setBytes:&n length:4 atIndex:3]; [e setBytes:&bits length:4 atIndex:4]; [e setBytes:&pb length:4 atIndex:5]; [e setBytes:&rbw length:4 atIndex:6]; [e setBytes:&codec length:4 atIndex:7];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [e setComputePipelineState:g_a_tq_deq];
    [e setBuffer:db offset:0 atIndex:0]; [e setBuffer:rb offset:0 atIndex:1]; [e setBuffer:ob offset:0 atIndex:2];
    [e setBytes:&n length:4 atIndex:3]; [e setBytes:&bits length:4 atIndex:4]; [e setBytes:&rbw length:4 atIndex:5]; [e setBytes:&codec length:4 atIndex:6];
    [e dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError) return 0;
    memcpy(out,[ob contents],(size_t)n*4);
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
    const void* qa_w,const float* qa_s,int qa_fmt,const float* qa_ln,
    const void* qb_w,const float* qb_s,int qb_fmt,
    const void* kva_w,const float* kva_s,int kva_fmt,const float* kva_ln,
    const void* kvb_w,const float* kvb_s,int kvb_fmt,
    const void* o_w,const float* o_s,int o_fmt,
    const void* shg_w,const float* shg_s,int shg_fmt,
    const void* shu_w,const float* shu_s,int shu_fmt,
    const void* shd_w,const float* shd_s,int shd_fmt,
    const float *router_w, const float *router_bias,
    int E, int K, int Ksel, float topp, int normk, float rscale,
    float *Lc, float *Rc, int qmode, int bits, int codec,
    const unsigned char* Lc8, const float* Lsc, const unsigned char* Rc8, const float* Rsc,
    int S, int pos_base, int st0,
    float eps, float theta, float ascale,
    float *inrm_out, float *nrm_out, float *sh_out, int *idx_out, float *w_out, int *keff_out) {
  if(!g_dev) return 0;
  if(st0!=0 || S<1 || S>AMAXS || E!=256 || K!=8) return 0;
  int T=pos_base+S; const int SI=2048;
  @autoreleasepool {
    attn_scratch_init();
    AttnW W={qa_w,qa_s,qa_fmt,qa_ln,qb_w,qb_s,qb_fmt,kva_w,kva_s,kva_fmt,kva_ln,kvb_w,kvb_s,kvb_fmt,o_w,o_s,o_fmt};
    // qmode 0: f32 cache (Lb/Rb via resolve_attn). qmode 1/2: byte cache + scale/radius (KV8/KV_TQ).
    id<MTLBuffer> Lb=nil,Rb=nil,kvbW,kvbS; size_t loff=0,roff=0,kvbwoff,kvbsoff;
    id<MTLBuffer> L8=nil,R8=nil,Ls=nil,Rs=nil; size_t l8off=0,r8off=0,lsoff=0,rsoff=0;
    if(qmode==0){
      if(!resolve_attn(&W,Lc,Rc,&Lb,&loff,&Rb,&roff,&kvbW,&kvbwoff,&kvbS,&kvbsoff)) return 0;
    } else {
      uint64_t la=0,ra=0,lsa=0,rsa=0,kwa=0,ksa=0;
      L8=resolve(Lc8,&la); R8=resolve(Rc8,&ra); Ls=resolve(Lsc,&lsa); Rs=resolve(Rsc,&rsa);
      kvbW=resolve(W.kvb_w,&kwa); kvbS=resolve(W.kvb_s,&ksa);
      if(!L8||!R8||!Ls||!Rs||!kvbW||!kvbS) return 0;
      uint64_t dd; const void* ws[]={W.qa_w,W.qa_s,W.qb_w,W.qb_s,W.kva_w,W.kva_s,W.o_w,W.o_s};
      for(auto p:ws) if(!resolve(p,&dd)) return 0;
      l8off=la-(uint64_t)[L8 gpuAddress]; r8off=ra-(uint64_t)[R8 gpuAddress];
      lsoff=lsa-(uint64_t)[Ls gpuAddress]; rsoff=rsa-(uint64_t)[Rs gpuAddress];
      kvbwoff=kwa-(uint64_t)[kvbW gpuAddress]; kvbsoff=ksa-(uint64_t)[kvbS gpuAddress];
      if(qmode==2){ aLstg_=ensure(aLstg_,&aLstg_cap,(size_t)T*AKVL*4); aRstg_=ensure(aRstg_,&aRstg_cap,(size_t)T*AROPE*4); }
    }
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
    if(qmode==0){ [e useResource:Lb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rb usage:MTLResourceUsageRead|MTLResourceUsageWrite]; }
    else { [e useResource:L8 usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:R8 usage:MTLResourceUsageRead|MTLResourceUsageWrite];
           [e useResource:Ls usage:MTLResourceUsageRead|MTLResourceUsageWrite]; [e useResource:Rs usage:MTLResourceUsageRead|MTLResourceUsageWrite]; }
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
    // 2) attention (ax_ -> aout_) — f32 / KV8 fp8 / KV_TQ per qmode
    if(!encode_attention(e,&W,Lb,loff,Rb,roff,kvbW,kvbwoff,kvbS,kvbsoff,S,pos_base,eps,theta,ascale,
                         qmode,bits,codec,L8,l8off,R8,r8off,Ls,lsoff,Rs,rsoff,aLstg_,aRstg_)) return 0;
    BAR();
    // 3) residual: axr_ += aout_ ; then nrm = post_ln(x_new)
    [e setComputePipelineState:g_a_add]; [e setBuffer:axr_ offset:0 atIndex:0]; [e setBuffer:aout_ offset:0 atIndex:1];
    [e dispatchThreads:MTLSizeMake((size_t)S*AH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; BAR();
    copyrow(axr_,anrm_,AH); BAR(); rmsw(anrm_,pnB,pnoff,AH,S); BAR();
    // 4) shared expert gate/up + router (all read anrm_, independent)
    bind_gemv(e,shg_w,shg_s,shg_fmt,AH,SI,anrm_,ash1_,S);
    bind_gemv(e,shu_w,shu_s,shu_fmt,AH,SI,anrm_,ash2_,S);
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
    bind_gemv(e,shd_w,shd_s,shd_fmt,SI,AH,ash1_,ashout_,S);
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
    [e setBytes:&I length:4 atIndex:5];
    [e setBytes:&O length:4 atIndex:6]; [e setBytes:&fmt length:4 atIndex:7];
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
static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt,
                         const void *const *g, const void *const *u, const void *const *d,
                         const float *const *gs, const float *const *us, const float *const *ds,
                         const float *xg, const int *xoff, const int *nr, int R,
                         id<MTLBuffer> xg_buf, id<MTLBuffer> gg_buf, id<MTLBuffer> uu_buf, id<MTLBuffer> hh_buf) {
  if (!g_dev || (fmt != 1 && fmt != 2)) return nil;
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
  // lifecycle, not by useResource:. See SUMMARY.md UNCERTAINTIES.
  if (!g_resset_enabled) {
    for(auto&b:use) [e useResource:b usage:MTLResourceUsageRead];
  }
  auto gemv=[&](id<MTLBuffer> wa,id<MTLBuffer> sa,id<MTLBuffer> xin,id<MTLBuffer> y,int O,int K,int Kin){
    int NT=R*O;
    [e setComputePipelineState:g_moe_gemv];
    [e setBuffer:wa offset:0 atIndex:0];[e setBuffer:sa offset:0 atIndex:1];[e setBuffer:berow offset:0 atIndex:2];
    [e setBuffer:xin offset:0 atIndex:3];[e setBuffer:y offset:0 atIndex:4];
    [e setBytes:&O length:4 atIndex:5];[e setBytes:&K length:4 atIndex:6];[e setBytes:&Kin length:4 atIndex:7];[e setBytes:&fmt length:4 atIndex:8];
    [e setBytes:&NT length:4 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(((size_t)NT+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; };
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
    id<MTLBuffer> bxg=[g_dev newBufferWithLength:(size_t)R*D*4 options:g_res_opts];
    id<MTLBuffer> bgg=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:g_res_opts];
    id<MTLBuffer> buu=[g_dev newBufferWithLength:(size_t)R*Iinter*4 options:g_res_opts];
    id<MTLBuffer> bhh=[g_dev newBufferWithLength:(size_t)R*D*4 options:g_res_opts];
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
