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
  printf(fail? "metal backend tests: FAILED\n" : "metal backend tests: ok\n");
  coli_metal_shutdown();
  return fail;
}
