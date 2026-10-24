// MPS vs built-in kernel speed for coli_metal_gemm on real GLM-5.x matmul shapes.
// No model needed: random int4 weights (grouped g64, the GLM-5.3 container format) in
// registered page-aligned slabs, exactly how the engine hands them to the GPU.
//
// For each shape and prompt length S it times the same GEMM with MPS on and off, checks
// the two results agree, and prints the speedup. Build/run:  make mps-bench
#include "../backend_metal.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Median of `reps` timed calls after one warm-up call.
static double time_gemm(float *y, const float *x, const void *W, const float *Sc,
                        int S, int I, int O, int gs, int reps, int *ok) {
  *ok = coli_metal_gemm(y, x, W, Sc, 4, S, I, O, gs);
  std::vector<double> t;
  for (int r = 0; r < reps && *ok; r++) {
    double t0 = now_s();
    *ok = coli_metal_gemm(y, x, W, Sc, 4, S, I, O, gs);
    t.push_back(now_s() - t0);
  }
  if (t.empty()) return 0;
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(void) {
  if (!coli_metal_init()) { printf("Metal unavailable\n"); return 1; }
  if (!coli_metal_mps_available()) { printf("MPS unavailable on this device/build\n"); return 1; }
  const int gs = 64;
  struct { int O, I; const char *n; } shapes[] = {
    { 2048, 6144,  "expert gate/up  O2048  I6144" },
    { 6144, 2048,  "expert down     O6144  I2048" },
    { 28672, 512,  "attn kv_b       O28672 I512 " },
    { 6144, 16384, "attn o_proj     O6144  I16384" },
  };
  int Ss[] = { 16, 64, 256, 1024 };
  printf("coli_metal_gemm, int4 grouped g64: built-in kernel vs MPS (median of 5)\n");
  printf("  %-30s %6s %11s %11s %8s %9s\n", "shape", "S", "kernel ms", "MPS ms", "speedup", "max diff");
  int fail = 0;
  for (auto &sh : shapes) {
    int O = sh.O, I = sh.I, rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    size_t wb = (((size_t)O * rb) + 16383) & ~(size_t)16383;
    size_t sb = (((size_t)O * ng * 4) + 16383) & ~(size_t)16383;
    uint8_t *W; float *Sc;
    if (posix_memalign((void **)&W, 16384, wb) || posix_memalign((void **)&Sc, 16384, sb)) return 1;
    srand(O + I);
    for (size_t i = 0; i < (size_t)O * rb; i++) W[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < O * ng; i++) Sc[i] = (0.001f + (rand() % 1000) / 1000.f) * ((rand() & 1) ? 1.f : -1.f) * 0.05f;
    coli_metal_register(W, wb); coli_metal_register(Sc, sb);
    for (int S : Ss) {
      std::vector<float> x((size_t)S * I), yk((size_t)S * O), ym((size_t)S * O);
      for (auto &v : x) v = ((rand() % 2000) - 1000) / 1000.f;
      int okk, okm;
      coli_metal_mps_config(0, 0, 0);
      double tk = time_gemm(yk.data(), x.data(), W, Sc, S, I, O, gs, 5, &okk);
      coli_metal_mps_config(1, 1, 0);
      double tm = time_gemm(ym.data(), x.data(), W, Sc, S, I, O, gs, 5, &okm);
      double md = 0, ymax = 0;
      for (size_t i = 0; i < ym.size(); i++) { md = fmax(md, fabs(ym[i] - yk[i])); ymax = fmax(ymax, fabs(yk[i])); }
      double rel = md / (ymax + 1e-9);
      int bad = !okk || !okm || rel > 1e-4;
      fail |= bad;
      printf("  %-30s %6d %11.2f %11.2f %7.2fx %9.1e%s\n", sh.n, S, tk * 1e3, tm * 1e3,
             tm > 0 ? tk / tm : 0.0, rel, bad ? "  *** FAIL" : "");
    }
    coli_metal_unregister(W); coli_metal_unregister(Sc); free(W); free(Sc);
  }
  coli_metal_mps_config(1, 8, 0);
  uint64_t c = 0, f = 0; coli_metal_mps_counts(&c, &f, nullptr);
  printf("MPS calls %llu, MPS fallbacks %llu\n", (unsigned long long)c, (unsigned long long)f);
  coli_metal_shutdown();
  return fail || f;
}
