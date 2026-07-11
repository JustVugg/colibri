// Kernel-correctness test for the Metal backend: coli_metal_matmul vs CPU reference
// (dequant->f32 MAC * per-row scale) for f32/int8/int4/int2 across real GLM shapes.
#include "../backend_metal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

enum { F32=0, I8=1, I4=2, I2=3 };

static void cpu_ref(int fmt, const void *W, const float *s, const float *x,
                    float *y, int S, int I, int O) {
  const int8_t *q8 = (const int8_t*)W; const uint8_t *q4 = (const uint8_t*)W;
  const float *qf = (const float*)W;
  int rb4=(I+1)/2, rb2=(I+3)/4;
  for (int o=0;o<O;o++) for (int si=0;si<S;si++){
    const float *xr = x + (size_t)si*I; float acc=0;
    for (int i=0;i<I;i++){
      float w;
      if (fmt==I8) w=(float)q8[(size_t)o*I+i];
      else if (fmt==I4){ uint8_t b=q4[(size_t)o*rb4+(i>>1)]; int v=(i&1)?(b>>4):(b&0xF); w=(float)(v-8); }
      else if (fmt==I2){ uint8_t b=q4[(size_t)o*rb2+(i>>2)]; int v=(b>>(2*(i&3)))&0x3; w=(float)(v-2); }
      else w=qf[(size_t)o*I+i];
      acc += w*xr[i];
    }
    y[(size_t)si*O+o]=acc*s[o];
  }
}

static int run(int fmt, int O, int I, int S, const char *name) {
  int rb4=(I+1)/2, rb2=(I+3)/4;
  size_t wn = (fmt==I8)?(size_t)O*I : (fmt==I4)?(size_t)O*rb4 : (fmt==I2)?(size_t)O*rb2 : (size_t)O*I*sizeof(float);
  std::vector<uint8_t> W(wn); std::vector<float> Wf;
  srand(99);
  if (fmt==F32){ Wf.resize((size_t)O*I); for(auto&v:Wf) v=((rand()%2000)-1000)/1000.f; }
  else for(auto&b:W) b=(uint8_t)((fmt==I8)?((rand()%255)-127):(rand()&0xFF));
  const void *Wp = (fmt==F32)?(const void*)Wf.data():(const void*)W.data();
  std::vector<float> s(O), x((size_t)S*I), yr((size_t)S*O), yg((size_t)S*O);
  for(auto&v:s) v=(fmt==F32)?1.0f:(0.01f+(rand()%100)/10000.f);
  for(auto&v:x) v=((rand()%2000)-1000)/1000.f;
  cpu_ref(fmt, Wp, s.data(), x.data(), yr.data(), S, I, O);
  ColiMetalTensor *t=nullptr;
  if (!coli_metal_matmul(&t, yg.data(), x.data(), Wp, s.data(), fmt, S, I, O)) {
    printf("  %-22s FAIL (matmul returned 0)\n", name); return 1; }
  double maxabs=0, ymax=0;
  for(size_t i=0;i<(size_t)S*O;i++){ maxabs=fmax(maxabs,fabs(yg[i]-yr[i])); ymax=fmax(ymax,fabs(yr[i])); }
  double nerr=maxabs/(ymax+1e-9);
  int ok = nerr < 1e-4;
  printf("  %-22s nerr=%.2e  %s\n", name, nerr, ok?"ok":"*** MISMATCH");
  coli_metal_tensor_free(t);
  return ok?0:1;
}

static float deq4(const uint8_t* w,int i){ uint8_t b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); return (float)(v-8); }
static size_t roundpg(size_t n){ size_t p=16384; return ((n+p-1)/p)*p; }

// Validate coli_metal_moe_block against a CPU reference (gate/up/silu/down + weighted scatter-add).
static int run_moe(const std::vector<int>& nrv, const char* name) {
  const int D=6144, I=2048, fmt=2; int rbG=(D+1)/2, rbD=(I+1)/2, nb=(int)nrv.size();
  int R=0; std::vector<int> xoff(nb),nr(nrv); for(int e=0;e<nb;e++){ xoff[e]=R; R+=nrv[e]; }
  srand(2024+nb);
  // per-expert page-aligned slab [Wg|Wu|Wd] and fslab [Sg|Su|Sd]; register both.
  std::vector<void*> slab(nb), fslab(nb);
  std::vector<const void*> g(nb),u(nb),d(nb); std::vector<const float*> gs(nb),us(nb),ds(nb);
  size_t wlen=roundpg((size_t)I*rbG*2 + (size_t)D*rbD), flen=roundpg(((size_t)I*2+D)*sizeof(float));
  for(int e=0;e<nb;e++){
    posix_memalign(&slab[e],16384,wlen); posix_memalign(&fslab[e],16384,flen);
    uint8_t* sp=(uint8_t*)slab[e]; for(size_t i=0;i<(size_t)I*rbG*2+(size_t)D*rbD;i++) sp[i]=(uint8_t)(rand()&0xFF);
    float* fp=(float*)fslab[e]; for(size_t i=0;i<(size_t)I*2+D;i++) fp[i]=0.01f+(rand()%50)/50000.f;
    g[e]=sp; u[e]=sp+(size_t)I*rbG; d[e]=sp+(size_t)I*rbG*2;
    gs[e]=fp; us[e]=fp+I; ds[e]=fp+2*I;
    coli_metal_register(slab[e],wlen); coli_metal_register(fslab[e],flen);
  }
  std::vector<float> xg((size_t)R*D); for(auto&v:xg) v=((rand()%2000)-1000)/1000.f;
  std::vector<int> rows(R); std::vector<float> rw(R);
  for(int gr=0;gr<R;gr++){ rows[gr]=0; rw[gr]=0.1f+(rand()%100)/100.f; }   // decode: all -> position 0
  int S=1;
  // CPU reference
  std::vector<float> refout((size_t)S*D,0.f), gg(I),uu(I),hh(D);
  for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++){ int gr=xoff[e]+r; const float* xr=&xg[(size_t)gr*D];
    const uint8_t* wg=(const uint8_t*)g[e]; const uint8_t* wu=(const uint8_t*)u[e]; const uint8_t* wd=(const uint8_t*)d[e];
    for(int o=0;o<I;o++){ float a=0; for(int k=0;k<D;k++) a+=deq4(wg+(size_t)o*rbG,k)*xr[k]; gg[o]=a*gs[e][o]; }
    for(int o=0;o<I;o++){ float a=0; for(int k=0;k<D;k++) a+=deq4(wu+(size_t)o*rbG,k)*xr[k]; uu[o]=a*us[e][o]; }
    for(int o=0;o<I;o++){ float v=gg[o]; gg[o]=(v/(1.f+expf(-v)))*uu[o]; }
    for(int o=0;o<D;o++){ float a=0; for(int k=0;k<I;k++) a+=deq4(wd+(size_t)o*rbD,k)*gg[k]; hh[o]=a*ds[e][o]; }
    float* os=&refout[(size_t)rows[gr]*D]; for(int o=0;o<D;o++) os[o]+=rw[gr]*hh[o];
  }
  std::vector<float> gout((size_t)S*D,0.f);
  int ok = coli_metal_moe_block(nb,D,I,fmt,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                                xg.data(),xoff.data(),nr.data(),rows.data(),rw.data(),gout.data(),S);
  double maxabs=0,ymax=0; for(size_t i=0;i<gout.size();i++){ maxabs=fmax(maxabs,fabs(gout[i]-refout[i])); ymax=fmax(ymax,fabs(refout[i])); }
  double nerr=maxabs/(ymax+1e-9); int pass = ok && nerr<1e-4;
  printf("  %-22s R=%d nerr=%.2e  %s\n", name, R, nerr, pass?"ok":"*** MISMATCH");
  for(int e=0;e<nb;e++){ coli_metal_unregister(slab[e]); coli_metal_unregister(fslab[e]); free(slab[e]); free(fslab[e]); }
  return pass?0:1;
}

// ---- fused decode attention vs a CPU reference replicating glm.c's exact math ----
// GLM-5.2 dims (hardcoded in the backend): hidden=6144 H=64 q_lora=2048 kv_lora=512
// nope=192 rope=64 vh=256; theta=10000 ascale=1/16 eps=1e-5.
enum { TH=6144, THH=64, TQL=2048, TKVL=512, TNOPE=192, TROPE=64, TVH=256, TQH=256, TROWSH=448 };
static void t_rms(float*o,const float*x,const float*w,int n,float eps){ double ms=0; for(int i=0;i<n;i++) ms+=(double)x[i]*x[i];
  float r=1.f/sqrtf((float)(ms/n)+eps); for(int i=0;i<n;i++) o[i]=x[i]*r*w[i]; }
static void t_rope(float*v,int pos,float th){ int hl=TROPE/2; float in[TROPE]; memcpy(in,v,sizeof(in));
  for(int j=0;j<hl;j++){ float inv=powf(th,-2.f*j/TROPE), a=in[2*j], b=in[2*j+1], cs=cosf(pos*inv), sn=sinf(pos*inv);
    v[j]=a*cs-b*sn; v[hl+j]=b*cs+a*sn; } }
static void t_gemv4(float*y,const float*x,const uint8_t*w,const float*sc,int O,int I){ int rb=(I+1)/2;
  for(int o=0;o<O;o++){ const uint8_t*r=w+(size_t)o*rb; float a=0;
    for(int i=0;i<I;i++){ uint8_t b=r[i>>1]; int v=(i&1)?(b>>4):(b&0xF); a+=(float)(v-8)*x[i]; } y[o]=a*sc[o]; } }
struct TW { uint8_t*w; float*s; size_t wb, sb; };
static TW t_mkw(int O,int I){ TW t; int rb=(I+1)/2;
  t.wb=((size_t)O*rb+16383)&~(size_t)16383; t.sb=((size_t)O*4+16383)&~(size_t)16383;
  posix_memalign((void**)&t.w,16384,t.wb); posix_memalign((void**)&t.s,16384,t.sb);
  for(size_t i=0;i<(size_t)O*rb;i++) t.w[i]=(uint8_t)(rand()&0xFF);
  for(int i=0;i<O;i++) t.s[i]=0.01f+(rand()%40)/40000.f;
  coli_metal_register(t.w,t.wb); coli_metal_register(t.s,t.sb); return t; }
static int run_attn(int S, int pos_base, const char* name){
  const float eps=1e-5f, theta=10000.f, ascale=1.f/16.f;
  srand(4242+S+pos_base);
  TW qa=t_mkw(TQL,TH), qb=t_mkw(THH*TQH,TQL), kva=t_mkw(TKVL+TROPE,TH), kvb=t_mkw(THH*TROWSH,TKVL), o=t_mkw(TH,THH*TVH);
  std::vector<float> qaln(TQL), kvaln(TKVL);
  for(auto&v:qaln) v=0.5f+(rand()%1000)/1000.f; for(auto&v:kvaln) v=0.5f+(rand()%1000)/1000.f;
  int T=pos_base+S; size_t lcb=(((size_t)T*TKVL*4)+16383)&~(size_t)16383, rcb=(((size_t)T*TROPE*4)+16383)&~(size_t)16383;
  float *Lc,*Rc; posix_memalign((void**)&Lc,16384,lcb); posix_memalign((void**)&Rc,16384,rcb);
  coli_metal_register(Lc,lcb); coli_metal_register(Rc,rcb);
  // pre-existing cache history [0,pos_base): random normed latents + roped krot
  for(int t=0;t<pos_base;t++){ for(int i=0;i<TKVL;i++) Lc[(size_t)t*TKVL+i]=((rand()%2000)-1000)/1500.f;
    for(int i=0;i<TROPE;i++) Rc[(size_t)t*TROPE+i]=((rand()%2000)-1000)/1500.f; }
  std::vector<float> x((size_t)S*TH); for(auto&v:x) v=((rand()%2000)-1000)/1000.f;
  std::vector<float> Lr((size_t)T*TKVL), Rr((size_t)T*TROPE);   // reference cache copies
  memcpy(Lr.data(),Lc,(size_t)pos_base*TKVL*4); memcpy(Rr.data(),Rc,(size_t)pos_base*TROPE*4);
  // CPU reference: mirrors glm.c attention() absorb branch (per new token, then per head)
  std::vector<float> Q((size_t)S*THH*TQH), ref((size_t)S*TH);
  for(int s=0;s<S;s++){ int pos=pos_base+s;
    std::vector<float> qr(TQL), comp(TKVL+TROPE);
    t_gemv4(qr.data(),&x[(size_t)s*TH],qa.w,qa.s,TQL,TH); t_rms(qr.data(),qr.data(),qaln.data(),TQL,eps);
    t_gemv4(&Q[(size_t)s*THH*TQH],qr.data(),qb.w,qb.s,THH*TQH,TQL);
    for(int h=0;h<THH;h++) t_rope(&Q[(size_t)s*THH*TQH+(size_t)h*TQH+TNOPE],pos,theta);
    t_gemv4(comp.data(),&x[(size_t)s*TH],kva.w,kva.s,TKVL+TROPE,TH);
    t_rms(&Lr[(size_t)pos*TKVL],comp.data(),kvaln.data(),TKVL,eps);
    memcpy(&Rr[(size_t)pos*TROPE],&comp[TKVL],TROPE*4); t_rope(&Rr[(size_t)pos*TROPE],pos,theta);
  }
  int rb=(TKVL+1)/2;
  for(int s=0;s<S;s++){ int pos=pos_base+s; std::vector<float> ctx((size_t)THH*TVH);
    for(int h=0;h<THH;h++){ int rbase=h*TROWSH;
      const float* qp=&Q[(size_t)s*THH*TQH+(size_t)h*TQH]; const float* qro=qp+TNOPE;
      std::vector<float> qabs(TKVL,0);
      for(int d=0;d<TNOPE;d++){ const uint8_t*r=kvb.w+(size_t)(rbase+d)*rb; float sc=kvb.s[rbase+d];
        for(int i=0;i<TKVL;i++){ uint8_t b=r[i>>1]; int v=(i&1)?(b>>4):(b&0xF); qabs[i]+=qp[d]*(float)(v-8)*sc; } }
      std::vector<float> a(pos+1);
      for(int t=0;t<=pos;t++){ const float*Lt=&Lr[(size_t)t*TKVL]; const float*Rt=&Rr[(size_t)t*TROPE];
        float v=0; for(int i=0;i<TKVL;i++) v+=qabs[i]*Lt[i]; for(int d=0;d<TROPE;d++) v+=qro[d]*Rt[d]; a[t]=v*ascale; }
      float mx=-1e30f; for(float v:a) mx=fmaxf(mx,v); float sum=0; for(float&v:a){ v=expf(v-mx); sum+=v; } for(float&v:a) v/=sum;
      std::vector<float> cl(TKVL,0);
      for(int t=0;t<=pos;t++){ const float*Lt=&Lr[(size_t)t*TKVL]; for(int i=0;i<TKVL;i++) cl[i]+=a[t]*Lt[i]; }
      for(int j=0;j<TVH;j++){ const uint8_t*r=kvb.w+(size_t)(rbase+TNOPE+j)*rb; float sc=kvb.s[rbase+TNOPE+j];
        float v=0; for(int i=0;i<TKVL;i++){ uint8_t b=r[i>>1]; int vv=(i&1)?(b>>4):(b&0xF); v+=cl[i]*(float)(vv-8)*sc; }
        ctx[(size_t)h*TVH+j]=v; } }
    t_gemv4(&ref[(size_t)s*TH],ctx.data(),o.w,o.s,TH,THH*TVH);
  }
  std::vector<float> got((size_t)S*TH);
  int ok=coli_metal_attn_decode(x.data(), qa.w,qa.s,2,qaln.data(), qb.w,qb.s,2,
        kva.w,kva.s,2,kvaln.data(), kvb.w,kvb.s,2, o.w,o.s,2,
        Lc,Rc,S,pos_base,0,eps,theta,ascale,got.data());
  double ma=0,ym=0; for(size_t i=0;i<ref.size();i++){ ma=fmax(ma,fabs(got[i]-ref[i])); ym=fmax(ym,fabs(ref[i])); }
  // also verify the cache write-back (Lc/Rc for the new positions)
  double mc=0; for(int s=0;s<S;s++){ int pos=pos_base+s;
    for(int i=0;i<TKVL;i++) mc=fmax(mc,fabs(Lc[(size_t)pos*TKVL+i]-Lr[(size_t)pos*TKVL+i]));
    for(int i=0;i<TROPE;i++) mc=fmax(mc,fabs(Rc[(size_t)pos*TROPE+i]-Rr[(size_t)pos*TROPE+i])); }
  double nerr=ma/(ym+1e-9);
  int pass = ok && nerr<2e-4 && mc<1e-4;
  printf("  %-24s nerr=%.2e cache=%.2e  %s\n", name, nerr, mc, pass?"ok":"*** MISMATCH");
  auto freew=[&](TW&t){ coli_metal_unregister(t.w); coli_metal_unregister(t.s); free(t.w); free(t.s); };
  freew(qa); freew(qb); freew(kva); freew(kvb); freew(o);
  coli_metal_unregister(Lc); coli_metal_unregister(Rc); free(Lc); free(Rc);
  return pass?0:1;
}

int main(void) {
  if (!coli_metal_init()) { printf("Metal unavailable (skipping)\n"); return 0; }
  printf("Metal backend kernel tests:\n");
  int fail=0;
  fail |= run(I8, 2048,6144,1, "int8 gate/up S=1");
  fail |= run(I4, 2048,6144,1, "int4 gate/up S=1");
  fail |= run(I4, 6144,2048,1, "int4 down S=1");
  fail |= run(I2, 2048,6144,1, "int2 gate/up S=1");
  fail |= run(F32,1024,6144,1, "f32  S=1");
  fail |= run(I8, 2048,6144,4, "int8 gate/up S=4");
  fail |= run(I4, 2048,6144,7, "int4 gate/up S=7 (odd)");
  fail |= run(I4, 2050,6146,3, "int4 non-mult-4 dims");
  printf("Metal batched moe_block tests:\n");
  fail |= run_moe({1,1,1,1,1,1,1,1}, "moe decode nb=8");
  fail |= run_moe({3,1,4,2,1,5},     "moe ragged nb=6");
  printf("Metal large-batch gemm test:\n");
  { // registered int4 weights, S=64: coli_metal_gemm vs cpu_ref
    srand(77); int O=2048,I=6144,S=64,rb=(I+1)/2;
    size_t wb=(((size_t)O*rb)+16383)&~(size_t)16383, sb2=(((size_t)O*4)+16383)&~(size_t)16383;
    uint8_t*W; float*Sc; posix_memalign((void**)&W,16384,wb); posix_memalign((void**)&Sc,16384,sb2);
    for(size_t i=0;i<(size_t)O*rb;i++) W[i]=(uint8_t)(rand()&0xFF);
    for(int i=0;i<O;i++) Sc[i]=0.01f+(rand()%50)/50000.f;
    coli_metal_register(W,wb); coli_metal_register(Sc,sb2);
    std::vector<float> x((size_t)S*I), yr((size_t)S*O), yg((size_t)S*O);
    for(auto&v:x) v=((rand()%2000)-1000)/1000.f;
    cpu_ref(I4,W,Sc,x.data(),yr.data(),S,I,O);
    int ok=coli_metal_gemm(yg.data(),x.data(),W,Sc,2,S,I,O);
    double ma=0,ym=0; for(size_t i=0;i<yr.size();i++){ ma=fmax(ma,fabs(yg[i]-yr[i])); ym=fmax(ym,fabs(yr[i])); }
    int pass = ok && ma/(ym+1e-9)<1e-4;
    printf("  gemm S=64 int4          nerr=%.2e  %s\n", ma/(ym+1e-9), pass?"ok":"*** MISMATCH");
    fail |= !pass;
    coli_metal_unregister(W); coli_metal_unregister(Sc); free(W); free(Sc);
  }
  printf("Metal fused attention tests:\n");
  fail |= run_attn(1, 0,   "attn S=1 pos=0");
  fail |= run_attn(1, 37,  "attn S=1 pos=37");
  fail |= run_attn(4, 12,  "attn S=4 pos=12 (MTP)");
  fail |= run_attn(3, 0,   "attn S=3 pos=0");
  printf(fail? "metal backend tests: FAILED\n" : "metal backend tests: ok\n");
  coli_metal_shutdown();
  return fail;
}
