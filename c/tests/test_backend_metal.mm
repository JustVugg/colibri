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
  printf(fail? "metal backend tests: FAILED\n" : "metal backend tests: ok\n");
  coli_metal_shutdown();
  return fail;
}
