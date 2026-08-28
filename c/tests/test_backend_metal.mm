// Kernel-correctness test for the Metal backend: coli_metal_matmul vs CPU reference
// (dequant->f32 MAC * per-row scale) for f32/int8/int4/int2 across real GLM shapes.
// fmt=4 (grouped int4, dev e9b3614 matmul_i4_grouped / CUDA #298 twin) gets its own
// reference (cpu_ref_grouped) and harness (run_grouped) below -- unlike fmt 1-3 the
// group scale is per-GROUP, not per-row, so it can't share cpu_ref's [O] scale layout.
#include "../backend_metal.h"
#include <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

enum { F32=0, I8=1, I4=2, I2=3, I4G=4, FP8=8 };

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
  if (!coli_metal_matmul(&t, yg.data(), x.data(), Wp, s.data(), fmt, S, I, O, 0)) {
    printf("  %-22s FAIL (matmul returned 0)\n", name); return 1; }
  double maxabs=0, ymax=0;
  for(size_t i=0;i<(size_t)S*O;i++){ maxabs=fmax(maxabs,fabs(yg[i]-yr[i])); ymax=fmax(ymax,fabs(yr[i])); }
  double nerr=maxabs/(ymax+1e-9);
  int ok = nerr < 1e-4;
  printf("  %-22s nerr=%.2e  %s\n", name, nerr, ok?"ok":"*** MISMATCH");
  coli_metal_tensor_free(t);
  return ok?0:1;
}

// ---- fmt=4 (grouped int4): own CPU reference + harness -----------------------------
// cpu_ref's [O] per-row scale layout can't express fmt=4's [O,ceil(I/gs)] per-group
// scale, so this is a separate reference rather than a branch of cpu_ref. Mirrors
// quant.h's matmul_i4_grouped exactly (same nibble decode, same group indexing i/gs)
// and accumulates in DOUBLE, matching the convention tests/test_i4_grouped.c already
// established as the fmt=4 oracle for the CUDA port (#298) -- reusing it here means the
// Metal kernel is checked against the same "known exact" reference, not a third one.
static void cpu_ref_grouped(const uint8_t *q4, const float *scale, const float *x,
                            double *y, double *mag, int S, int I, int O, int gs) {
  int rb=(I+1)/2, ng=(I+gs-1)/gs;
  for (int o=0;o<O;o++){
    const uint8_t *w = q4 + (size_t)o*rb;
    const float *scl = scale + (size_t)o*ng;
    for (int s=0;s<S;s++){
      const float *xs = x + (size_t)s*I; double a=0, m=0;
      for (int i=0;i<I;i++){
        uint8_t b = w[i>>1]; int nib = (i&1) ? (int)(b>>4) : (int)(b&0xF);
        double term = (double)xs[i] * (double)(nib-8) * (double)scl[i/gs];
        a += term; m += fabs(term);
      }
      y[(size_t)s*O+o] = a; mag[(size_t)s*O+o] = m;
    }
  }
}

// ---- fmt=8 (native FP8-e4m3 passthrough -- see colibri.c): own CPU reference +
// harness --------------------------------------------------------------------------
// Independent reference decode (bit manipulation, NOT quant.h's E4M3_LUT -- this file
// stays self-contained like the rest of its CPU references) for OCP E4M3-FN: exp==0 is
// subnormal, exp==0xF&&mant==0x7 is the only NaN code (both signs), else normal with
// bias 7. Must match quant.h's e4m3_decode AND the mm_gemv fmt==8 branch's in-kernel
// bit manipulation exactly -- see run_fp8_lut() below for the exhaustive 256-code check
// that proves the GPU kernel agrees with this reference (and by transitivity with the
// CPU LUT, which was cross-checked byte-for-byte against torch.float8_e4m3fn offline).
static float ref_e4m3(uint8_t b) {
  unsigned sign=(b>>7)&1, exp=(b>>3)&0xF, mant=b&0x7;
  if (exp==0xF && mant==0x7) return NAN;
  float val = (exp==0) ? (float)mant*(1.0f/8.0f)*powf(2.0f,1.0f-7.0f)
                       : (1.0f+(float)mant*(1.0f/8.0f))*powf(2.0f,(float)exp-7.0f);
  return sign ? -val : val;
}
/* Renamed from fp8_nblk: quant.h, included further down this same
 * translation unit, now declares `static inline int64_t fp8_nblk`.
 * Matching the return type alone still collides (inline vs non-inline),
 * and this file keeps its own reference implementations on purpose --
 * see the E4M3 note above -- so it takes the ref_ prefix the other
 * independent helpers here already use. (#838) */
static int64_t ref_fp8_nblk(int n){ return (n+127)/128; }

static void cpu_ref_fp8(const uint8_t *q8, const float *bscale, const float *x,
                        double *y, double *mag, int S, int I, int O) {
  int nblkI = ref_fp8_nblk(I);
  for (int o=0;o<O;o++){
    const uint8_t *w = q8 + (size_t)o*I;
    const float *scl = bscale + (size_t)(o/128)*nblkI;
    for (int s=0;s<S;s++){
      const float *xs = x + (size_t)s*I; double a=0, m=0;
      for (int i=0;i<I;i++){
        double term = (double)ref_e4m3(w[i]) * (double)xs[i] * (double)scl[i/128];
        a += term; m += fabs(term);
      }
      y[(size_t)s*O+o] = a; mag[(size_t)s*O+o] = m;
    }
  }
}

// Tolerance: magnitude-relative (error / sum of |terms|), NOT the ymax-relative-max-
// abs-error convention run() above uses. Reason: a dot product over many groups can
// land near zero from signed-term cancellation, and a fixed-point-ish absolute GPU/CPU
// rounding difference then reads as a huge relative error against the (near-zero)
// RESULT -- that is the accumulator's precision, not a kernel defect. Comparing against
// the sum of |terms| instead means a wrong scale index or wrong group boundary is still
// caught as an O(1) relative error (it shifts the result by a fraction of the terms),
// while cancellation noise is not misreported as a bug. This is the exact convention
// tests/test_i4_grouped.c uses for the same kernel's CPU-side oracle (see its
// `ref_grouped`/`mag` comment) -- reused here for a GPU oracle of the same format.
// 1e-4 (vs. test_i4_grouped.c's 1e-6) because the GPU sums in a different order --
// byte-pair-strided across 32 SIMD lanes then simd_sum -- than the scalar double
// reference; that's the same 1e-4 slack run()'s f32-vs-f32 comparisons already carry
// for fmt 1-3 above, just applied to a magnitude-relative rather than max-relative base.
static int run_grouped(int O, int I, int gs, int S, int outlier, const char *name) {
  int rb=(I+1)/2, ng=(I+gs-1)/gs;
  std::vector<uint8_t> W((size_t)O*rb);
  std::vector<float> scale((size_t)O*ng), x((size_t)S*I), yg((size_t)S*O);
  std::vector<double> yr((size_t)S*O), mag((size_t)S*O);
  srand(4110 + I*7 + O*3 + gs + S*13 + outlier*97);
  for (auto &b : W) b = (uint8_t)(rand()&0xFF);
  // scales span orders of magnitude -- a wrong group index shows up big, not as noise.
  for (auto &v : scale) v = (0.001f + (rand()%1000)/1000.0f) * ((rand()&1) ? 1.f : -1.f);
  for (auto &v : x) v = ((rand()%2000)-1000)/1000.0f;
  if (outlier) {
    // Tail-clipping regime grouped scales exist for: one huge activation per row,
    // confined to group 0. A per-row quantizer's single scale would have to stretch to
    // cover this outlier, crushing every other group's precision; per-group scaling
    // contains the damage to group 0's scale and leaves the rest of the row exact. This
    // isn't a property of matmul_i4_grouped's weights (which are the same random bytes
    // as the non-outlier case) -- it's the activation-side stress the format is FOR, and
    // it exercises the SAME code path (still just per-group scale lookup + MAC), so a
    // reconstruction win here is really a check that group boundaries/indices are right
    // under the shape that would most expose a wrong one.
    for (int s=0; s<S; s++) x[(size_t)s*I+0] = 50.0f;
  }
  cpu_ref_grouped(W.data(), scale.data(), x.data(), yr.data(), mag.data(), S, I, O, gs);
  ColiMetalTensor *t=nullptr;
  if (!coli_metal_matmul(&t, yg.data(), x.data(), W.data(), scale.data(), I4G, S, I, O, gs)) {
    printf("  %-34s FAIL (matmul returned 0)\n", name); return 1; }
  double worst=0; int bad=0;
  for (size_t i=0; i<(size_t)S*O; i++) {
    double d = fabs((double)yg[i] - yr[i]);
    double rel = mag[i] > 1e-30 ? d/mag[i] : d;
    if (rel > worst) worst = rel;
    if (rel > 1e-4) bad++;
  }
  int ok = (bad == 0);
  printf("  %-34s worst_rel=%.2e (I=%d gs=%d ng=%d S=%d)  %s\n", name, worst, I, gs, ng, S, ok?"ok":"*** MISMATCH");
  coli_metal_tensor_free(t);
  return ok?0:1;
}

// Tolerance/oracle convention identical to run_grouped() above (magnitude-relative for
// the block-scale accumulation, avoiding cancellation-noise false positives on
// near-zero results).
static int run_fp8(int O, int I, int S, const char *name) {
  int nblkO=ref_fp8_nblk(O), nblkI=ref_fp8_nblk(I), nblk=nblkO*nblkI;
  std::vector<uint8_t> W((size_t)O*I);
  std::vector<float> scale((size_t)nblk), x((size_t)S*I), yg((size_t)S*O);
  std::vector<double> yr((size_t)S*O), mag((size_t)S*O);
  srand(5150 + I*7 + O*3 + S*13);
  for (auto &b : W) { do { b = (uint8_t)(rand()&0xFF); } while (b==0x7F || b==0xFF); }  // exclude NaN codes
  // distinct, easily-distinguishable scale per block (stride-audit idiom, same as
  // test_fp8_passthrough.c's CPU version): a swapped o/128 vs i/128 stride, or a wrong
  // nblkI in the shader's indexing, lands on a DIFFERENT block's scale and shows up as
  // a loud numeric mismatch rather than passing by luck.
  for (int b=0;b<nblk;b++) scale[b] = (float)(b+1)*0.01f*((b&1)?-1.f:1.f);
  for (auto &v : x) v = ((rand()%2000)-1000)/1000.0f;
  cpu_ref_fp8(W.data(), scale.data(), x.data(), yr.data(), mag.data(), S, I, O);
  ColiMetalTensor *t=nullptr;
  if (!coli_metal_matmul(&t, yg.data(), x.data(), W.data(), scale.data(), FP8, S, I, O, 0)) {
    printf("  %-42s FAIL (matmul returned 0)\n", name); return 1; }
  double worst=0; int bad=0;
  for (size_t i=0; i<(size_t)S*O; i++) {
    double d = fabs((double)yg[i] - yr[i]);
    double rel = mag[i] > 1e-30 ? d/mag[i] : d;
    if (rel > worst) worst = rel;
    if (rel > 1e-4) bad++;
  }
  int ok = (bad == 0);
  printf("  %-42s worst_rel=%.2e (I=%d O=%d nblkO=%d nblkI=%d S=%d)  %s\n",
         name, worst, I, O, nblkO, nblkI, S, ok?"ok":"*** MISMATCH");
  coli_metal_tensor_free(t);
  return ok?0:1;
}

// Exhaustive LUT-exactness check run THROUGH the actual GPU kernel: O=256,I=1 means
// each of the 256 output rows has exactly ONE weight byte, set to that row's own index
// (0..255) -- so row o's dequant is exactly e4m3_decode(o). x=[1.0] and both blocks'
// scale=1.0 isolate the decode from the block-scale/accumulation logic entirely. This
// mirrors test_fp8_passthrough.c's CPU LUT-exactness test (test_lut), but exercised
// through mm_gemv's in-kernel bit manipulation instead of quant.h's table -- proving
// the two independently-written decoders agree on all 256 codes, including both NaN
// codes and the sign of zero.
static int run_fp8_lut(const char *name) {
  enum { O=256, I=1 };
  std::vector<uint8_t> W(O*I); for (int b=0;b<O;b++) W[b]=(uint8_t)b;
  std::vector<float> scale(ref_fp8_nblk(O)*ref_fp8_nblk(I), 1.0f);   // nblkO=2,nblkI=1 -> both blocks scale=1
  std::vector<float> x(I, 1.0f), yg(O);
  ColiMetalTensor *t=nullptr;
  if (!coli_metal_matmul(&t, yg.data(), x.data(), W.data(), scale.data(), FP8, 1, I, O, 0)) {
    printf("  %-42s FAIL (matmul returned 0)\n", name); return 1; }
  int bad=0;
  for (int b=0;b<O;b++) {
    float ref = ref_e4m3((uint8_t)b);
    if (b==0x7F || b==0xFF) { if (!std::isnan(yg[b])) bad++; continue; }
    if (yg[b] != ref) bad++;
  }
  int ok = (bad==0);
  printf("  %-42s %d/256 mismatches  %s\n", name, bad, ok?"ok":"*** MISMATCH");
  coli_metal_tensor_free(t);
  return ok?0:1;
}

// Confirms the documented "GEMM path optional this build" claim is actually true, not
// just asserted in a comment: coli_metal_gemm must refuse fmt=8 (return 0, the CPU-
// fallback signal) so matmul_qt_ex's caller-side allowlist exclusion is backed by a
// second, independent gate inside the Metal backend itself.
static int run_fp8_gemm_gate(const char *name) {
  uint8_t w[8]={0}; float s[1]={1.0f}, x[8]={0}, y[1]={0};
  int rc = coli_metal_gemm(y, x, w, s, FP8, 1, 8, 1, 0);
  int ok = (rc == 0);
  printf("  %-42s rc=%d (expect 0/CPU-fallback)  %s\n", name, rc, ok?"ok":"*** MISMATCH (should have refused)");
  return ok?0:1;
}

// colibri.c's moe() has a THIRD fmt=8-adjacent Metal entry point besides the two
// guarded above (bind_gemv's attn/layer-decode shaders and coli_metal_gemm) -- the
// batched routed-expert dispatch via coli_metal_moe_block[_begin] (colibri.c's
// MB_BUILD macro). MB_BUILD's own pointer-selection ternary does NOT special-case
// fmt=8: if a layer's shared expert is fmt=8 and MB_BUILD's TRY_SH path picks it
// up with no routed expert already fixing `mfmt`, it would submit the WRONG pointer
// (q4, NULL/stale for an fmt=8 tensor whose weights live in q8) tagged as fmt=8.
// This is safe ANYWAY, but only incidentally: moe_submit() (backend_metal.mm) gates
// on its fmt allowlist ({1,2,4,5,6}) as its very FIRST statement, before any of
// g/u/d/gs/us/ds is dereferenced or even resolve()'d -- so an fmt=8 submission is
// refused before the bad pointer would ever be read, no matter what garbage
// MB_BUILD packed into it. This
// test uses deliberately-invalid weight/scale pointers (never dereferenced if the gate
// holds) to prove the fence BY TEST rather than leaving it an artifact of moe_submit's
// fmt allowlist happening not to include 8 (yet) -- same discipline
// run_fp8_gemm_gate above applies to coli_metal_gemm's analogous exclusion.
static int run_fp8_moe_gate(const char *name) {
  const void *bad = (const void*)(uintptr_t)0xdeadbeef;   // must NEVER be dereferenced
  const void *g[1] = {bad}, *u[1] = {bad}, *d[1] = {bad};
  const float *gs[1] = {(const float*)bad}, *us[1] = {(const float*)bad}, *ds[1] = {(const float*)bad};
  float xg[8]={0}, out[8]={0}, rw[1]={1.0f};
  int xoff[1]={0}, nr[1]={1}, rows[1]={0};
  int rc = coli_metal_moe_block(1, 8, 8, FP8, 0, g, u, d, gs, us, ds, xg, xoff, nr, rows, rw, out, 1);
  int ok = (rc == 0);
  printf("  %-42s rc=%d (expect 0/CPU-fallback)  %s\n", name, rc, ok?"ok":"*** MISMATCH (should have refused)");
  return ok?0:1;
}

static float deq4(const uint8_t* w,int i){ uint8_t b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); return (float)(v-8); }
static size_t roundpg(size_t n){ size_t p=16384; return ((n+p-1)/p)*p; }

// Validate coli_metal_moe_block against a CPU reference (gate/up/silu/down + weighted scatter-add).
// qgs==0 -> fmt=2 (per-row scale). qgs>0 -> fmt=4 grouped int4: per-expert scale slab is
// [O][ng] (ng=ceil(K/qgs)) for each of gate/up (K=D) and down (K=Iinter).
static int run_moe(const std::vector<int>& nrv, int qgs, const char* name) {
  const int D=6144, I=2048; int fmt = qgs>0 ? 4 : 2;
  int rbG=(D+1)/2, rbD=(I+1)/2, nb=(int)nrv.size();
  int ngG = qgs>0 ? (D+qgs-1)/qgs : 1, ngD = qgs>0 ? (I+qgs-1)/qgs : 1;  // scales/row for gate-up / down
  int R=0; std::vector<int> xoff(nb),nr(nrv); for(int e=0;e<nb;e++){ xoff[e]=R; R+=nrv[e]; }
  srand(2024+nb+qgs);
  // per-column scale accessor: grouped picks s[o*ng + k/qgs], per-row picks s[o].
  auto scaG=[&](const float* s,int o,int k){ return qgs>0 ? s[(size_t)o*ngG + k/qgs] : s[o]; };
  auto scaD=[&](const float* s,int o,int k){ return qgs>0 ? s[(size_t)o*ngD + k/qgs] : s[o]; };
  // per-expert page-aligned slab [Wg|Wu|Wd] and fslab [Sg|Su|Sd]; register both.
  std::vector<void*> slab(nb), fslab(nb);
  std::vector<const void*> g(nb),u(nb),d(nb); std::vector<const float*> gs(nb),us(nb),ds(nb);
  size_t nsc=(size_t)I*ngG*2 + (size_t)D*ngD;   // gate + up + down scale counts
  size_t wlen=roundpg((size_t)I*rbG*2 + (size_t)D*rbD), flen=roundpg(nsc*sizeof(float));
  for(int e=0;e<nb;e++){
    posix_memalign(&slab[e],16384,wlen); posix_memalign(&fslab[e],16384,flen);
    uint8_t* sp=(uint8_t*)slab[e]; for(size_t i=0;i<(size_t)I*rbG*2+(size_t)D*rbD;i++) sp[i]=(uint8_t)(rand()&0xFF);
    float* fp=(float*)fslab[e]; for(size_t i=0;i<nsc;i++) fp[i]=0.01f+(rand()%50)/50000.f;
    g[e]=sp; u[e]=sp+(size_t)I*rbG; d[e]=sp+(size_t)I*rbG*2;
    gs[e]=fp; us[e]=fp+(size_t)I*ngG; ds[e]=fp+(size_t)I*ngG*2;
    coli_metal_register(slab[e],wlen); coli_metal_register(fslab[e],flen);
  }
  std::vector<float> xg((size_t)R*D); for(auto&v:xg) v=((rand()%2000)-1000)/1000.f;
  std::vector<int> rows(R); std::vector<float> rw(R);
  for(int gr=0;gr<R;gr++){ rows[gr]=0; rw[gr]=0.1f+(rand()%100)/100.f; }   // decode: all -> position 0
  int S=1;
  // CPU reference (grouped scale folded per-term; for fmt=2 that reduces to a*s[o])
  std::vector<float> refout((size_t)S*D,0.f), gg(I),uu(I),hh(D);
  for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++){ int gr=xoff[e]+r; const float* xr=&xg[(size_t)gr*D];
    const uint8_t* wg=(const uint8_t*)g[e]; const uint8_t* wu=(const uint8_t*)u[e]; const uint8_t* wd=(const uint8_t*)d[e];
    for(int o=0;o<I;o++){ float a=0; for(int k=0;k<D;k++) a+=deq4(wg+(size_t)o*rbG,k)*xr[k]*scaG(gs[e],o,k); gg[o]=a; }
    for(int o=0;o<I;o++){ float a=0; for(int k=0;k<D;k++) a+=deq4(wu+(size_t)o*rbG,k)*xr[k]*scaG(us[e],o,k); uu[o]=a; }
    for(int o=0;o<I;o++){ float v=gg[o]; gg[o]=(v/(1.f+expf(-v)))*uu[o]; }
    for(int o=0;o<D;o++){ float a=0; for(int k=0;k<I;k++) a+=deq4(wd+(size_t)o*rbD,k)*gg[k]*scaD(ds[e],o,k); hh[o]=a; }
    float* os=&refout[(size_t)rows[gr]*D]; for(int o=0;o<D;o++) os[o]+=rw[gr]*hh[o];
  }
  std::vector<float> gout((size_t)S*D,0.f);
  int ok = coli_metal_moe_block(nb,D,I,fmt,qgs,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                                xg.data(),xoff.data(),nr.data(),rows.data(),rw.data(),gout.data(),S);
  double maxabs=0,ymax=0; for(size_t i=0;i<gout.size();i++){ maxabs=fmax(maxabs,fabs(gout[i]-refout[i])); ymax=fmax(ymax,fabs(refout[i])); }
  double nerr=maxabs/(ymax+1e-9); int pass = ok && nerr<1e-4;
  printf("  %-30s R=%d nerr=%.2e  %s\n", name, R, nerr, pass?"ok":"*** MISMATCH");
  for(int e=0;e<nb;e++){ coli_metal_unregister(slab[e]); coli_metal_unregister(fslab[e]); free(slab[e]); free(fslab[e]); }
  return pass?0:1;
}

// ---- fmt=6 (E8/IQ3) moe_block vs the engine's own scalar decoder ----
// Mirrors the engine split exactly: the caller pre-rotates the staged gate/up input
// (colibri.c metal_stage_rot_e8), the GPU rotates the down input (moe_fwht), and
// matmul_e8/e8_rot_rows from quant.h are the reference for both.
#define COLI_QUANT_TEST_ONLY 1
#include "../quant.h"
static int run_moe_e8(const std::vector<int>& nrv, const char* name) {
  const int D=6144, I=1536, fmt=6;                     // I=1536 exercises the 512+1024 FWHT tiles
  int64_t rbG=e8_rowbytes(D), rbD=e8_rowbytes(I); int nb=(int)nrv.size();
  int R=0; std::vector<int> xoff(nb),nr(nrv); for(int e=0;e<nb;e++){ xoff[e]=R; R+=nrv[e]; }
  srand(6666+nb);
  std::vector<void*> slab(nb), fslab(nb);
  std::vector<const void*> g(nb),u(nb),d(nb); std::vector<const float*> gs(nb),us(nb),ds(nb);
  size_t wlen=roundpg((size_t)I*rbG*2 + (size_t)D*rbD), flen=roundpg(3*sizeof(float));
  for(int e=0;e<nb;e++){
    posix_memalign(&slab[e],16384,wlen); posix_memalign(&fslab[e],16384,flen);
    uint8_t* sp=(uint8_t*)slab[e];
    for(size_t i=0;i<(size_t)I*rbG*2+(size_t)D*rbD;i++) sp[i]=(uint8_t)(rand()&0xFF);
    // clamp every block scale to a sane positive fp16 (random fp16 can be inf/nan)
    for(size_t off=0; off+98<=(size_t)I*rbG*2+(size_t)D*rbD; off+=98){
      uint16_t dh=(uint16_t)(0x2C00 | (rand()&0x3FF)); memcpy(sp+off+96,&dh,2); }
    float* fp=(float*)fslab[e]; fp[0]=fp[1]=fp[2]=1.f;  // .qs tags: resolvable, unused
    g[e]=sp; u[e]=sp+(size_t)I*rbG; d[e]=sp+(size_t)I*rbG*2;
    gs[e]=fp; us[e]=fp+1; ds[e]=fp+2;
    coli_metal_register(slab[e],wlen); coli_metal_register(fslab[e],flen);
  }
  std::vector<float> xg((size_t)R*D); for(auto&v:xg) v=((rand()%2000)-1000)/1000.f;
  std::vector<int> rows(R); std::vector<float> rw(R);
  for(int gr=0;gr<R;gr++){ rows[gr]=0; rw[gr]=0.1f+(rand()%100)/100.f; }
  int S=1;
  // CPU reference from the unrotated input, engine semantics
  std::vector<float> refout((size_t)S*D,0.f), xrot(D), gg(I),uu(I),hh(D);
  for(int e=0;e<nb;e++) for(int r=0;r<nr[e];r++){ int gr=xoff[e]+r;
    memcpy(xrot.data(), &xg[(size_t)gr*D], D*sizeof(float)); e8_rot_rows(xrot.data(),1,D);
    matmul_e8(gg.data(),xrot.data(),(const uint8_t*)g[e],NULL,1,D,I);
    matmul_e8(uu.data(),xrot.data(),(const uint8_t*)u[e],NULL,1,D,I);
    for(int o=0;o<I;o++){ float v=gg[o]; gg[o]=(v/(1.f+expf(-v)))*uu[o]; }
    e8_rot_rows(gg.data(),1,I);
    matmul_e8(hh.data(),gg.data(),(const uint8_t*)d[e],NULL,1,I,D);
    float* os=&refout[(size_t)rows[gr]*D]; for(int o=0;o<D;o++) os[o]+=rw[gr]*hh[o];
  }
  // GPU input: pre-rotated, as colibri.c stages it
  std::vector<float> xg_gpu(xg);
  for(int gr=0;gr<R;gr++) e8_rot_rows(&xg_gpu[(size_t)gr*D],1,D);
  std::vector<float> gout((size_t)S*D,0.f);
  int ok = coli_metal_moe_block(nb,D,I,fmt,0,g.data(),u.data(),d.data(),gs.data(),us.data(),ds.data(),
                                xg_gpu.data(),xoff.data(),nr.data(),rows.data(),rw.data(),gout.data(),S);
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
// Grouped-int4 (fmt=4) variant of t_mkw: same packed-nibble weight layout, but the
// scale slab holds O*ceil(I/gs) floats. Used to prove the bind_gemv/AttnW/
// coli_metal_attn_decode plumbing (not just the standalone GEMV) threads gs correctly.
static TW t_mkw_g(int O,int I,int gs){ TW t; int rb=(I+1)/2, ng=(I+gs-1)/gs;
  t.wb=((size_t)O*rb+16383)&~(size_t)16383; t.sb=((size_t)O*(size_t)ng*4+16383)&~(size_t)16383;
  posix_memalign((void**)&t.w,16384,t.wb); posix_memalign((void**)&t.s,16384,t.sb);
  for(size_t i=0;i<(size_t)O*rb;i++) t.w[i]=(uint8_t)(rand()&0xFF);
  for(int i=0;i<O*ng;i++) t.s[i]=(0.001f+(rand()%1000)/1000.f)*((rand()&1)?1.f:-1.f);
  coli_metal_register(t.w,t.wb); coli_metal_register(t.s,t.sb); return t; }
// Grouped-int4 CPU reference GEMV, mirroring t_gemv4 but with a per-group scale lookup
// (same semantics as quant.h's matmul_i4_grouped / cpu_ref_grouped above).
static void t_gemv4g(float*y,const float*x,const uint8_t*w,const float*sc,int O,int I,int gs){
  int rb=(I+1)/2, ng=(I+gs-1)/gs;
  for(int o=0;o<O;o++){ const uint8_t*r=w+(size_t)o*rb; const float* scl=sc+(size_t)o*ng; float a=0;
    for(int i=0;i<I;i++){ uint8_t b=r[i>>1]; int v=(i&1)?(b>>4):(b&0xF); a+=(float)(v-8)*x[i]*scl[i/gs]; }
    y[o]=a; } }
// kvb_gs==0 -> kv_b as fmt=2 (per-row scale); kvb_gs>0 -> kv_b as fmt=4 grouped int4
// (exercises a_deqrow's grouped-scale path in a_qabs/a_ctx, #587's kv_b addition).
static int run_attn(int S, int pos_base, int kvb_gs, const char* name){
  const float eps=1e-5f, theta=10000.f, ascale=1.f/16.f;
  srand(4242+S+pos_base);
  int kvb_fmt = kvb_gs>0 ? 4 : 2, kvng = kvb_gs>0 ? (TKVL+kvb_gs-1)/kvb_gs : 1;
  TW qa=t_mkw(TQL,TH), qb=t_mkw(THH*TQH,TQL), kva=t_mkw(TKVL+TROPE,TH);
  TW kvb = kvb_gs>0 ? t_mkw_g(THH*TROWSH,TKVL,kvb_gs) : t_mkw(THH*TROWSH,TKVL);
  TW o=t_mkw(TH,THH*TVH);
  // per-column kv_b scale: grouped (fmt=4) picks scale[row*ng + i/gs], else per-row.
  auto kvb_sc=[&](int row,int i)->float{ return kvb_gs>0 ? kvb.s[(size_t)row*kvng + i/kvb_gs] : kvb.s[row]; };
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
      for(int d=0;d<TNOPE;d++){ const uint8_t*r=kvb.w+(size_t)(rbase+d)*rb;
        for(int i=0;i<TKVL;i++){ uint8_t b=r[i>>1]; int v=(i&1)?(b>>4):(b&0xF); qabs[i]+=qp[d]*(float)(v-8)*kvb_sc(rbase+d,i); } }
      std::vector<float> a(pos+1);
      for(int t=0;t<=pos;t++){ const float*Lt=&Lr[(size_t)t*TKVL]; const float*Rt=&Rr[(size_t)t*TROPE];
        float v=0; for(int i=0;i<TKVL;i++) v+=qabs[i]*Lt[i]; for(int d=0;d<TROPE;d++) v+=qro[d]*Rt[d]; a[t]=v*ascale; }
      float mx=-1e30f; for(float v:a) mx=fmaxf(mx,v); float sum=0; for(float&v:a){ v=expf(v-mx); sum+=v; } for(float&v:a) v/=sum;
      std::vector<float> cl(TKVL,0);
      for(int t=0;t<=pos;t++){ const float*Lt=&Lr[(size_t)t*TKVL]; for(int i=0;i<TKVL;i++) cl[i]+=a[t]*Lt[i]; }
      for(int j=0;j<TVH;j++){ const uint8_t*r=kvb.w+(size_t)(rbase+TNOPE+j)*rb;
        float v=0; for(int i=0;i<TKVL;i++){ uint8_t b=r[i>>1]; int vv=(i&1)?(b>>4):(b&0xF); v+=cl[i]*(float)(vv-8)*kvb_sc(rbase+TNOPE+j,i); }
        ctx[(size_t)h*TVH+j]=v; } }
    t_gemv4(&ref[(size_t)s*TH],ctx.data(),o.w,o.s,TH,THH*TVH);
  }
  std::vector<float> got((size_t)S*TH);
  int ok=coli_metal_attn_decode(x.data(), qa.w,qa.s,2,0,qaln.data(), qb.w,qb.s,2,0,
        kva.w,kva.s,2,0,kvaln.data(), kvb.w,kvb.s,kvb_fmt,kvb_gs, o.w,o.s,2,0,
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

// serial r_top8 vs parallel r_top8_par on the ENGINE build's own compiled shaders — the
// exact-match contract (same indices, same order, same weights bitwise, same keff)
// enforced with memcmp, per adversarial input family. `mode` selects the input
// construction; see the inventory at the call sites in main(). E is a parameter (not
// hardcoded 256) so the same helper drives both the original E=256 fuzz and the
// expert-count-generality cases (E=24 <32-lane-width, E=168 REAP-pruned, E=200
// lane-straddling boundary, E=257 out-of-contract auto-serial-fallback proof).
static int run_rtop8(int mode, int S, int E, float topp, int normk, float rscale, const char *name) {
  const int K=8, Ksel=8;
  std::vector<float> sig((size_t)S*E), bias(E);
  srand(4242+mode*17+S+E);
  for (int e=0;e<E;e++) bias[e]=((rand()%2001)-1000)/1000.f;
  for (int s=0;s<S;s++) for (int e=0;e<E;e++) {
    float *v=&sig[(size_t)s*E+e];
    switch (mode) {
      case 0: *v=(float)(rand()%10000)/10000.f; break;                  // generic sigmoid-like
      case 1: *v=0.5f; break;                                           // ALL EQUAL: pure tie-break test
      case 2: *v=(float)((e/2)%8)/8.f; break;                           // massed duplicates (paired+cyclic ties)
      case 3: *v=(e%2)?1e-40f:2e-40f; break;                            // denormal logits (flush behavior must match)
      case 4: *v=(float)(rand()%3)/2.f; break;                          // 3-level ties across the whole row
      // boundary-forcing: elevate the LAST 4 valid experts (E-4..E-1) to near-max choice
      // so they are guaranteed in the top-8. For an E whose per-lane block size doesn't
      // divide E evenly, E-1's lane straddles the E boundary (real indices below E,
      // sentinel -1e30f at/above E in the SAME ch[] block) -- e.g. E=200: per=ceil(200/
      // 32)=7, lane 28 owns indices 196..202, of which 196-199 are real and 200-202 are
      // sentinel. Forcing selection onto 196-199 exercises exactly that lane's per-index
      // e<E boundary check, rather than hoping random data happens to land there.
      case 5: *v=(e>=E-4)?1.0f:(float)(rand()%10000)/10000.f; break;
      default: *v=(float)(rand()%10000)/10000.f; break;
    }
  }
  if (mode==1) for (int e=0;e<E;e++) bias[e]=0.25f;                     // choice fully tied too
  if (mode==3) for (int e=0;e<E;e++) bias[e]=(e%3)?3e-40f:-3e-40f;      // denormal bias as well
  if (mode==5) for (int e=E-4;e<E;e++) bias[e]=1.0f;                    // combined choice = 2.0, max possible
  std::vector<int> is((size_t)S*K), ip((size_t)S*K); std::vector<float> ws((size_t)S*K), wp((size_t)S*K);
  std::vector<int> ks(S), kp(S);
  if (!coli_metal_rtop8(0,sig.data(),bias.data(),S,E,K,Ksel,topp,normk,rscale,is.data(),ws.data(),ks.data()) ||
      !coli_metal_rtop8(1,sig.data(),bias.data(),S,E,K,Ksel,topp,normk,rscale,ip.data(),wp.data(),kp.data())) {
    printf("  %-34s FAIL (rtop8 runner returned 0)\n", name); return 1; }
  int ok = memcmp(is.data(),ip.data(),(size_t)S*K*4)==0 &&
           memcmp(ws.data(),wp.data(),(size_t)S*K*4)==0 &&              // bitwise: same ops, same order
           memcmp(ks.data(),kp.data(),(size_t)S*4)==0;
  if (mode==5 && ok) {
    // Don't just trust the input design -- confirm the straddling lane's valid segment
    // (E-4..E-1) was actually selected, in EVERY row, so this case can't silently
    // degrade into an unrelated pass if the input construction above ever changes.
    for (int s=0;s<S;s++) { int seen=0;
      for (int k=0;k<K;k++) if (ip[(size_t)s*K+k]>=E-4 && ip[(size_t)s*K+k]<E) seen++;
      if (seen<4) { printf("  %-34s *** boundary segment not exercised (row %d saw %d/4) -- test setup bug\n", name, s, seen); return 1; }
    }
  }
  if (!ok) {
    printf("  %-34s *** MISMATCH\n", name);
    for (int s=0;s<S;s++){ printf("    row %d keff %d/%d:",s,ks[s],kp[s]);
      for(int k=0;k<K;k++) printf(" [%d]%d/%d %.6g/%.6g",k,is[s*K+k],ip[s*K+k],ws[s*K+k],wp[s*K+k]);
      printf("\n"); }
    return 1;
  }
  printf("  %-34s ok (serial==parallel bitwise, S=%d E=%d)\n", name, S, E);
  return 0;
}

// Same fused-attention pipeline as run_attn, but q_a is fmt=4 (grouped int4, gs) while
// q_b/kv_a/kv_b/o stay fmt=2 -- proves the bind_gemv/AttnW/coli_metal_attn_decode
// plumbing (qa_gs threaded through encode_attention) is wired correctly end-to-end
// through the SAME fused command buffer real decode uses (attention_rows in colibri.c),
// not just the standalone coli_metal_matmul entry point run_grouped() above exercises.
// kv_b is deliberately left fmt=2 HERE: this test isolates the qa_gs plumbing through
// bind_gemv specifically. kv_b never flows through bind_gemv at all (a_qabs/a_ctx
// dequantize it inline via a_deqrow, which is itself fmt/gs-aware) -- its own grouped
// (fmt=4) coverage lives in run_attn's kvb_gs>0 cases below, not here.
//
// Two blind spots closed here (review round 1 -- see PR_BODY.md sec 10):
//  (a) pos_base must be >0 for any S=1 case. At pos_base=0, T=pos_base+S=1: softmax
//      over a SINGLE key is identically 1.0 regardless of the score's value, so the
//      final output is provably independent of q_a's kernel output entirely -- an S=1
//      pos=0 case cannot catch ANY defect in q_a, not just scale bugs. Every S=1 call
//      below now uses pos_base>0 (mirroring run_attn's own S=1 pos=37 case, which exists
//      for the same reason on the non-grouped kernels).
//  (b) RMSNorm(c*v) == RMSNorm(v) for any positive scalar c -- it divides out any
//      UNIFORM rescale of its input before qb/RoPE/attention ever see it. So even with
//      T>1, a whole-tensor scale-calibration bug in q_a's grouped-int4 kernel (every
//      group's scale off by the same factor) would be invisible in `got`/`ref` below no
//      matter how the rest of the pipeline is shaped -- fixing (a) alone does not fix
//      this. Closed by comparing q_a's RAW GEMV output (before RMSNorm swallows it)
//      directly against the CPU oracle, via the standalone coli_metal_matmul entry point
//      on the EXACT weight/scale/x data this attention test generated (not a re-run of
//      run_grouped()'s own separate random data) -- see the qraw block below.
static int run_attn_grouped(int S, int pos_base, int gs, const char* name){
  const float eps=1e-5f, theta=10000.f, ascale=1.f/16.f;
  srand(5150+S+pos_base+gs);
  TW qa=t_mkw_g(TQL,TH,gs), qb=t_mkw(THH*TQH,TQL), kva=t_mkw(TKVL+TROPE,TH), kvb=t_mkw(THH*TROWSH,TKVL), o=t_mkw(TH,THH*TVH);
  std::vector<float> qaln(TQL), kvaln(TKVL);
  for(auto&v:qaln) v=0.5f+(rand()%1000)/1000.f; for(auto&v:kvaln) v=0.5f+(rand()%1000)/1000.f;
  int T=pos_base+S; size_t lcb=(((size_t)T*TKVL*4)+16383)&~(size_t)16383, rcb=(((size_t)T*TROPE*4)+16383)&~(size_t)16383;
  float *Lc,*Rc; posix_memalign((void**)&Lc,16384,lcb); posix_memalign((void**)&Rc,16384,rcb);
  coli_metal_register(Lc,lcb); coli_metal_register(Rc,rcb);
  for(int t=0;t<pos_base;t++){ for(int i=0;i<TKVL;i++) Lc[(size_t)t*TKVL+i]=((rand()%2000)-1000)/1500.f;
    for(int i=0;i<TROPE;i++) Rc[(size_t)t*TROPE+i]=((rand()%2000)-1000)/1500.f; }
  std::vector<float> x((size_t)S*TH); for(auto&v:x) v=((rand()%2000)-1000)/1000.f;
  std::vector<float> Lr((size_t)T*TKVL), Rr((size_t)T*TROPE);
  memcpy(Lr.data(),Lc,(size_t)pos_base*TKVL*4); memcpy(Rr.data(),Rc,(size_t)pos_base*TROPE*4);
  std::vector<float> Q((size_t)S*THH*TQH), ref((size_t)S*TH);
  ColiMetalTensor *traw=nullptr;         // persistent handle for the raw-qa GPU probe (blind spot (b))
  double qraw_worst=0; int qraw_bad=0;
  for(int s=0;s<S;s++){ int pos=pos_base+s;
    std::vector<float> qr(TQL), comp(TKVL+TROPE);
    t_gemv4g(qr.data(),&x[(size_t)s*TH],qa.w,qa.s,TQL,TH,gs);                          // <- grouped, RAW (pre-RMSNorm)
    { // Blind spot (b): compare this RAW q_a output against the CPU oracle BEFORE
      // t_rms below overwrites qr in place -- RMSNorm is what makes the post-norm
      // comparison blind to a uniform scale error, so the check has to happen here,
      // not on `got`/`ref`. Same magnitude-relative construction as run_grouped().
      std::vector<double> qrd(TQL), qrmag(TQL);
      cpu_ref_grouped(qa.w,qa.s,&x[(size_t)s*TH],qrd.data(),qrmag.data(),1,TH,TQL,gs);
      std::vector<float> qraw(TQL);
      int qok=coli_metal_matmul(&traw,qraw.data(),&x[(size_t)s*TH],qa.w,qa.s,4,1,TH,TQL,gs);
      if(!qok) qraw_bad++;
      for(int i=0;i<TQL;i++){ double d=fabs((double)qraw[i]-qrd[i]); double rel=qrmag[i]>1e-30?d/qrmag[i]:d;
        if(rel>qraw_worst) qraw_worst=rel; if(rel>1e-4) qraw_bad++; }
    }
    t_rms(qr.data(),qr.data(),qaln.data(),TQL,eps);   // <- grouped
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
  int ok=coli_metal_attn_decode(x.data(), qa.w,qa.s,4,gs,qaln.data(), qb.w,qb.s,2,0,      // <- qa fmt=4/gs
        kva.w,kva.s,2,0,kvaln.data(), kvb.w,kvb.s,2,0, o.w,o.s,2,0,
        Lc,Rc,S,pos_base,0,eps,theta,ascale,got.data());
  double ma=0,ym=0; for(size_t i=0;i<ref.size();i++){ ma=fmax(ma,fabs(got[i]-ref[i])); ym=fmax(ym,fabs(ref[i])); }
  double mc=0; for(int s=0;s<S;s++){ int pos=pos_base+s;
    for(int i=0;i<TKVL;i++) mc=fmax(mc,fabs(Lc[(size_t)pos*TKVL+i]-Lr[(size_t)pos*TKVL+i]));
    for(int i=0;i<TROPE;i++) mc=fmax(mc,fabs(Rc[(size_t)pos*TROPE+i]-Rr[(size_t)pos*TROPE+i])); }
  double nerr=ma/(ym+1e-9);
  int pass = ok && nerr<2e-4 && mc<1e-4 && qraw_bad==0;
  printf("  %-24s nerr=%.2e cache=%.2e qraw=%.2e  %s\n", name, nerr, mc, qraw_worst, pass?"ok":"*** MISMATCH");
  coli_metal_tensor_free(traw);
  auto freew=[&](TW&t){ coli_metal_unregister(t.w); coli_metal_unregister(t.s); free(t.w); free(t.s); };
  freew(qa); freew(qb); freew(kva); freew(kvb); freew(o);
  coli_metal_unregister(Lc); coli_metal_unregister(Rc); free(Lc); free(Rc);
  return pass?0:1;
}

// ---- Shared error-reporting helper ------------------------------------------------
// Computes maxAbsErr and meanAbsErr between GPU and CPU arrays, prints standardized
// report line. `tol` is the max-abs-error tolerance (absolute, not normalized).
// Returns 0 on pass, 1 on failure.
static int report_err(const float *cpu, const float *gpu, size_t n, const char *name, double tol) {
  double maxabs=0, sumabs=0, ymax=0;
  for (size_t i=0; i<n; i++) {
    double d = fabs((double)cpu[i] - (double)gpu[i]);
    double v = fabs((double)cpu[i]);
    maxabs = fmax(maxabs, d);
    sumabs += d;
    ymax = fmax(ymax, v);
  }
  double mae = sumabs / (double)n;
  double nerr = maxabs / (ymax + 1e-9);
  int ok = maxabs < tol;
  printf("  %-30s maxAbs=%.2e mae=%.2e nerr=%.2e  %s\n", name, maxabs, mae, nerr, ok ? "ok" : "*** MISMATCH");
  return ok ? 0 : 1;
}

// ---- RMSNorm CPU reference -------------------------------------------------------
static void cpu_rmsnorm(float *out, const float *x, const float *w, int D, float eps) {
  double ms = 0; for (int i=0; i<D; i++) ms += (double)x[i] * (double)x[i];
  float r = 1.f / sqrtf((float)(ms / D) + eps);
  for (int i=0; i<D; i++) out[i] = x[i] * r * w[i];
}

static int run_rmsnorm(int nrows, int D, const char *name) {
  std::vector<float> x((size_t)nrows * D), w(D), ref((size_t)nrows * D), gpu((size_t)nrows * D);
  srand(5050 + D + nrows);
  for (auto &v : x) v = ((rand() % 2000) - 1000) / 1000.f;
  for (auto &v : w) v = 0.5f + (rand() % 1000) / 1000.f;
  float eps = 1e-5f;
  for (int r = 0; r < nrows; r++) {
    cpu_rmsnorm(&ref[(size_t)r * D], &x[(size_t)r * D], w.data(), D, eps);
  }
  memcpy(gpu.data(), x.data(), (size_t)nrows * D * 4);
  if (!coli_metal_rmsnorm(gpu.data(), w.data(), nrows, D, eps)) {
    printf("  %-30s FAIL (rmsnorm returned 0)\n", name); return 1;
  }
  return report_err(ref.data(), gpu.data(), (size_t)nrows * D, name, 1e-4);
}

// ---- Residual add CPU reference --------------------------------------------------
static void cpu_add(float *y, const float *a, size_t n) {
  for (size_t i = 0; i < n; i++) y[i] += a[i];
}

static int run_add(size_t n, const char *name) {
  std::vector<float> y_cpu(n), y_gpu(n), a(n);
  srand(6060 + (int)(n % 10000));
  for (auto &v : y_cpu) v = ((rand() % 2000) - 1000) / 1000.f;
  for (auto &v : a)     v = ((rand() % 2000) - 1000) / 1000.f;
  memcpy(y_gpu.data(), y_cpu.data(), n * 4);
  cpu_add(y_cpu.data(), a.data(), n);
  if (!coli_metal_add(y_gpu.data(), a.data(), n)) {
    printf("  %-30s FAIL (add returned 0)\n", name); return 1;
  }
  return report_err(y_cpu.data(), y_gpu.data(), n, name, 1e-6);
}

// ---- SiLU-multiply CPU reference -------------------------------------------------
static float cpu_siluf(float x) { return x / (1.f + expf(-x)); }

static void cpu_silu_mul(float *g, const float *u, size_t n) {
  for (size_t i = 0; i < n; i++) g[i] = cpu_siluf(g[i]) * u[i];
}

static int run_silu_mul(size_t n, const char *name) {
  std::vector<float> g_cpu(n), g_gpu(n), u(n);
  srand(7070 + (int)(n % 10000));
  for (auto &v : g_cpu) v = ((rand() % 4000) - 2000) / 1000.f;     // wider range to exercise sigmoid saturation
  for (auto &v : u)     v = ((rand() % 2000) - 1000) / 1000.f;
  memcpy(g_gpu.data(), g_cpu.data(), n * 4);
  cpu_silu_mul(g_cpu.data(), u.data(), n);
  if (!coli_metal_silu_mul(g_gpu.data(), u.data(), n)) {
    printf("  %-30s FAIL (silu_mul returned 0)\n", name); return 1;
  }
  return report_err(g_cpu.data(), g_gpu.data(), n, name, 1e-4);
}

int main(void) {
  if (!coli_metal_init()) { printf("Metal unavailable (skipping)\n"); return 0; }
  /* GitHub Actions Apple Silicon runners expose Metal through the Apple
   * Paravirtual device, where Metal submissions never complete (hangs -- the
   * #947 CI observation). The shader compile above (coli_metal_init) still
   * runs, keeping this suite's compile coverage -- the class of bug #940
   * shipped -- while GPU execution is honestly skipped here. (#947 review) */
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (dev && [[dev name] containsString:@"Apple Paravirtual device"]) {
    printf("SKIPPED: paravirtual device (%s) -- compile-only\n",
           [[dev name] UTF8String]);
    return 0;
  }
  printf("Metal standalone op tests (MAE / maxAbs error vs CPU reference):\n");
  int fail=0;
  // RMSNorm: single-row and multi-row, real model dims, small D
  fail |= run_rmsnorm(1, 6144, "rmsnorm S=1 D=6144");
  fail |= run_rmsnorm(1, 256,  "rmsnorm S=1 D=256");
  fail |= run_rmsnorm(1, 4096, "rmsnorm S=1 D=4096");
  fail |= run_rmsnorm(1, 73,   "rmsnorm S=1 D=73 (odd)");
  fail |= run_rmsnorm(4, 2048, "rmsnorm S=4 D=2048");
  fail |= run_rmsnorm(8, 4096, "rmsnorm S=8 D=4096");
  // Residual add: small, mid, large
  fail |= run_add(256,    "add n=256");
  fail |= run_add(6144,   "add n=6144");
  fail |= run_add(32768,  "add n=32768");
  fail |= run_add(1,      "add n=1 (degenerate)");
  // SiLU-multiply: small, mid, large
  fail |= run_silu_mul(256,    "silu_mul n=256");
  fail |= run_silu_mul(2048,   "silu_mul n=2048");
  fail |= run_silu_mul(32768,  "silu_mul n=32768");
  fail |= run_silu_mul(13,     "silu_mul n=13 (odd)");
  fail |= run_silu_mul(1,      "silu_mul n=1 (degenerate)");
  // ---- Phase 5: expanded standalone op battery (edge cases + real model shapes) ----
  printf("  Phase 5: expanded standalone op battery:\n");
  // RMSNorm: simd boundary, extreme D, large batch, pathological inputs
  fail |= run_rmsnorm(1, 1,      "rmsnorm S=1 D=1 (degenerate)");
  fail |= run_rmsnorm(1, 2,      "rmsnorm S=1 D=2 (minimum)");
  fail |= run_rmsnorm(1, 16,     "rmsnorm S=1 D=16 (sub-simd)");
  fail |= run_rmsnorm(1, 32,     "rmsnorm S=1 D=32 (simd_width)");
  fail |= run_rmsnorm(1, 128,    "rmsnorm S=1 D=128");
  fail |= run_rmsnorm(1, 16384,  "rmsnorm S=1 D=16384 (larger)");
  fail |= run_rmsnorm(16, 6144,  "rmsnorm S=16 D=6144 (batch)");
  fail |= run_rmsnorm(32, 2048,  "rmsnorm S=32 D=2048 (wide batch)");
  // Residual add: simd boundary, very large, non-power-of-2
  fail |= run_add(16,       "add n=16 (sub-simd)");
  fail |= run_add(32,       "add n=32 (simd_width)");
  fail |= run_add(131072,   "add n=131072 (large)");
  fail |= run_add(262145,   "add n=262145 (large odd)");
  fail |= run_add(65537,    "add n=65537 (prime-ish)");
  // SiLU-multiply: simd boundary, large, odd
  fail |= run_silu_mul(32,     "silu_mul n=32 (simd_width)");
  fail |= run_silu_mul(128,    "silu_mul n=128");
  fail |= run_silu_mul(65536,  "silu_mul n=65536 (large)");
  fail |= run_silu_mul(131073, "silu_mul n=131073 (large odd)");
  fail |= run_silu_mul(7,      "silu_mul n=7 (small odd)");
  fail |= run_silu_mul(2,      "silu_mul n=2 (minimum)");
  fail |= run_silu_mul(16,     "silu_mul n=16 (sub-simd)");
  /* TODO: KV-cache tests — run_kv_* not implemented yet
  printf("Metal KV cache tests (MAE / maxAbs error vs CPU reference):\n");
  fail |= run_kv_write(1, 0,   "kv_write S=1 pos=0");
  fail |= run_kv_write(4, 12,  "kv_write S=4 pos=12");
  fail |= run_kv_write(1, 100, "kv_write S=1 pos=100");
  fail |= run_kv_write(8, 0,   "kv_write S=8 pos=0");
  fail |= run_kv_clear(1, 10,  "kv_clear [0,10)");
  fail |= run_kv_clear(5, 20,  "kv_clear [5,20)");
  fail |= run_kv_clear(0, 0,   "kv_clear empty range");
  fail |= run_kv_prefill(10,   "kv_prefill 10 tokens");
  fail |= run_kv_decode(15,    "kv_decode pos=15");
  fail |= run_kv_long_decode(100, "kv_long_decode pos=100");
  fail |= run_kv_reset(12,     "kv_reset write-clear-reuse");
  fail |= run_kv_multiseq(3, 8, "kv_multiseq 3 seqs len=8");
  */
  printf("Metal quantized GEMV tests:\n");
  fail |= run(I8, 2048,6144,1, "int8 gate/up S=1");
  fail |= run(I4, 2048,6144,1, "int4 gate/up S=1");
  fail |= run(I4, 6144,2048,1, "int4 down S=1");
  fail |= run(I2, 2048,6144,1, "int2 gate/up S=1");
  fail |= run(F32,1024,6144,1, "f32  S=1");
  fail |= run(I8, 2048,6144,4, "int8 gate/up S=4");
  fail |= run(I4, 2048,6144,7, "int4 gate/up S=7 (odd)");
  fail |= run(I4, 2050,6146,3, "int4 non-mult-4 dims");
  printf("Metal fmt=4 grouped-int4 tests (coli_metal_matmul vs matmul_i4_grouped semantics):\n");
  // I multiple of gs=64, S=1 and S>1 (real g64-checkpoint shapes: gate/up I=6144, down I=2048)
  fail |= run_grouped(2048,6144,64,1,0, "grouped gate/up I=6144(mult64) S=1");
  fail |= run_grouped(6144,2048,64,1,0, "grouped down I=2048(mult64) S=1");
  fail |= run_grouped(2048,6144,64,4,0, "grouped gate/up I=6144(mult64) S=4");
  // I NOT a multiple of gs=64 (partial last group, the glen-clamp off-by-one)
  fail |= run_grouped(6,   200, 64,2,0, "grouped I=200 (non-mult-64) S=2");
  fail |= run_grouped(5,   201, 64,3,0, "grouped I=201 (odd, non-mult-64) S=3");
  // I<=64 degenerate: single group (ng=1), including I<gs
  fail |= run_grouped(3,   64,  64,1,0, "grouped I=64 (degenerate, ng=1) S=1");
  fail |= run_grouped(3,   40,  64,1,0, "grouped I=40 (<gs, single partial group) S=1");
  // random + outlier-heavy rows: one large-magnitude activation per row (the
  // tail-clipping regime grouped scales exist for), verified end-to-end through the GPU.
  fail |= run_grouped(5,   512, 64,3,1, "grouped I=512(mult64) outlier-heavy S=3");
  fail |= run_grouped(5,   201, 64,2,1, "grouped I=201(non-mult-64) outlier-heavy S=2");
  printf("Metal fmt=8 native FP8-e4m3 passthrough tests:\n");
  fail |= run_fp8_lut("fp8 LUT exactness (256/256 codes via GPU kernel)");
  fail |= run_fp8(2048,6144,1, "fp8 gate/up-shaped O=2048 I=6144 (spec example) S=1");
  fail |= run_fp8(6144,2048,1, "fp8 down-shaped O=6144 I=2048 S=1");
  fail |= run_fp8(2048,6144,4, "fp8 gate/up-shaped O=2048 I=6144 S=4");
  // block edges: O,I not multiples of 128 (the partial-tile clamp)
  fail |= run_fp8(130, 200, 2, "fp8 block edges: O,I both non-mult-128");
  fail |= run_fp8(129, 128, 1, "fp8 block edges: O just over 128, I exact");
  fail |= run_fp8(128, 129, 1, "fp8 block edges: O exact, I just over 128");
  fail |= run_fp8(1,   1,   1, "fp8 degenerate 1x1 (single sub-block)");
  // non-square block grid (nblkO=3 != nblkI=48) with a distinct scale per block: a
  // swapped o/128 vs i/128 index, or a wrong nblkI stride, lands on the wrong block's
  // scale and this shape/scale choice makes that numerically loud.
  fail |= run_fp8(384, 6144, 3, "fp8 non-square block grid nblkO=3 nblkI=48 (stride audit)");
  fail |= run_fp8_gemm_gate("fp8 GEMM entry explicitly gated off (coli_metal_gemm refuses)");
  fail |= run_fp8_moe_gate("fp8 MB_BUILD/moe_submit entry gated off (shared-expert fmt=8 hazard)");
  printf("Metal batched moe_block tests:\n");
  fail |= run_moe({1,1,1,1,1,1,1,1}, 0,   "moe decode nb=8");
  fail |= run_moe({3,1,4,2,1,5},     0,   "moe ragged nb=6");
  fail |= run_moe({1,1,1,1,1,1,1,1}, 128, "moe decode nb=8  fmt4-g128");
  fail |= run_moe({3,1,4,2,1,5},     128, "moe ragged nb=6  fmt4-g128");
  fail |= run_moe({3,1,4,2,1,5},     64,  "moe ragged nb=6  fmt4-g64");
  printf("Metal fmt=6 (E8/IQ3) moe_block tests:\n");
  fail |= run_moe_e8({1,1,1,1,1,1,1,1}, "e8 decode nb=8");
  fail |= run_moe_e8({3,1,4,2,1,5},     "e8 ragged nb=6");
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
    int ok=coli_metal_gemm(yg.data(),x.data(),W,Sc,2,S,I,O,0);
    double ma=0,ym=0; for(size_t i=0;i<yr.size();i++){ ma=fmax(ma,fabs(yg[i]-yr[i])); ym=fmax(ym,fabs(yr[i])); }
    int pass = ok && ma/(ym+1e-9)<1e-4;
    printf("  gemm S=64 int4          nerr=%.2e  %s\n", ma/(ym+1e-9), pass?"ok":"*** MISMATCH");
    fail |= !pass;
    coli_metal_unregister(W); coli_metal_unregister(Sc); free(W); free(Sc);
  }
  { // registered GROUPED int4 (fmt=4) weights, S=64: coli_metal_gemm vs cpu_ref_grouped
    // -- covers the matmul_qt_ex prefill dispatch path (colibri.c), the second of the
    // two directly-named "gemm-test pattern" entry points mm_gemv is reached through.
    srand(881); int O=2048,I=6144,S=64,gs=64,rb=(I+1)/2,ng=(I+gs-1)/gs;
    size_t wb=(((size_t)O*rb)+16383)&~(size_t)16383, sbg=(((size_t)O*(size_t)ng*4)+16383)&~(size_t)16383;
    uint8_t*W; float*Sc; posix_memalign((void**)&W,16384,wb); posix_memalign((void**)&Sc,16384,sbg);
    for(size_t i=0;i<(size_t)O*rb;i++) W[i]=(uint8_t)(rand()&0xFF);
    for(int i=0;i<O*ng;i++) Sc[i]=(0.001f+(rand()%1000)/1000.f)*((rand()&1)?1.f:-1.f);
    coli_metal_register(W,wb); coli_metal_register(Sc,sbg);
    std::vector<float> x((size_t)S*I), yg((size_t)S*O);
    std::vector<double> yr((size_t)S*O), mag((size_t)S*O);
    for(auto&v:x) v=((rand()%2000)-1000)/1000.f;
    cpu_ref_grouped(W,Sc,x.data(),yr.data(),mag.data(),S,I,O,gs);
    int ok=coli_metal_gemm(yg.data(),x.data(),W,Sc,4,S,I,O,gs);
    double worst=0; for(size_t i=0;i<yg.size();i++){ double d=fabs((double)yg[i]-yr[i]); double rel=mag[i]>1e-30?d/mag[i]:d; if(rel>worst) worst=rel; }
    int pass = ok && worst<1e-4;
    printf("  gemm S=64 int4-grouped   worst_rel=%.2e  %s\n", worst, pass?"ok":"*** MISMATCH");
    fail |= !pass;
    coli_metal_unregister(W); coli_metal_unregister(Sc); free(W); free(Sc);
  }
  printf("Metal fused attention tests:\n");
  fail |= run_attn(1, 0,   0,   "attn S=1 pos=0");
  fail |= run_attn(1, 37,  0,   "attn S=1 pos=37");
  fail |= run_attn(4, 12,  0,   "attn S=4 pos=12 (MTP)");
  fail |= run_attn(3, 0,   0,   "attn S=3 pos=0");
  // fmt=4 grouped-int4 kv_b (g128/g64): exercises a_deqrow's grouped-scale path in
  // a_qabs/a_ctx (#587's kv_b addition). S=3/S=4 batches carry rows with T>1 (non-
  // degenerate softmax), so the qabs (NOPE-side) dequant is exercised too, not just
  // the always-live a_ctx (VH-side) dequant that a T=1 row alone would cover.
  fail |= run_attn(1, 0,   128, "attn S=1 pos=0   kvb-fmt4-g128");
  fail |= run_attn(1, 37,  128, "attn S=1 pos=37  kvb-fmt4-g128");
  fail |= run_attn(4, 12,  128, "attn S=4 pos=12  kvb-fmt4-g128 (MTP)");
  fail |= run_attn(3, 0,   64,  "attn S=3 pos=0   kvb-fmt4-g64");
  printf("Metal fused attention tests (fmt=4 grouped q_a, proves bind_gemv gs plumbing):\n");
  // pos_base=37 (not 0): at T=1 softmax is identically 1.0 regardless of q_a's output,
  // so an S=1 pos=0 case cannot catch ANY q_a defect (review round 1, auditor -- see
  // PR_BODY.md sec 10). Both cases also carry the raw pre-RMSNorm qraw check internally.
  fail |= run_attn_grouped(1, 37, 64, "attn grouped-qa S=1 pos=37");
  fail |= run_attn_grouped(4, 12, 64, "attn grouped-qa S=4 pos=12 (MTP)");
  printf("Metal negative control: fmt 1/2/3 unaffected by the fmt=4 shader branch --\n"
         "  see the runs above (int8/int4/int2/f32/moe/gemm/attn cases): all still ok.\n");
  printf("Metal top-8 select serial-vs-parallel tests (exact-match contract, E=256):\n");
  fail |= run_rtop8(0, 1, 256, 0.0f,  1, 1.0f,   "top8 generic S=1");
  fail |= run_rtop8(0, 4, 256, 0.0f,  1, 1.0f,   "top8 generic S=4");
  fail |= run_rtop8(1, 1, 256, 0.0f,  1, 1.0f,   "top8 ALL-EQUAL ties");
  fail |= run_rtop8(2, 4, 256, 0.0f,  1, 1.0f,   "top8 massed dup ties S=4");
  fail |= run_rtop8(4, 2, 256, 0.0f,  0, 2.5f,   "top8 3-level ties rscale");
  fail |= run_rtop8(3, 1, 256, 0.0f,  1, 1.0f,   "top8 denormal logits");
  fail |= run_rtop8(0, 1, 256, 0.01f, 1, 1.0f,   "top8 topp=0.01 (Ke=1 edge)");
  fail |= run_rtop8(2, 1, 256, 0.6f,  1, 1.0f,   "top8 topp=0.6 tied weights");
  fail |= run_rtop8(0, 4, 256, 0.999f,1, 1.75f,  "top8 topp=0.999 S=4");
  fail |= run_rtop8(1, 2, 256, 0.5f,  0, 1.0f,   "top8 topp on ALL-EQUAL");
  printf("Metal top-8 select expert-count-generality tests (E!=256, REAP/#428 motivated):\n");
  fail |= run_rtop8(0, 1, 168, 0.0f,  1, 1.0f,   "top8 E=168 (REAP) generic S=1");
  fail |= run_rtop8(2, 4, 168, 0.0f,  1, 1.0f,   "top8 E=168 (REAP) massed dup ties S=4");
  fail |= run_rtop8(0, 1, 24,  0.0f,  1, 1.0f,   "top8 E=24 (<32 lane width) generic");
  fail |= run_rtop8(1, 1, 24,  0.0f,  1, 1.0f,   "top8 E=24 (<32 lane width) ALL-EQUAL ties");
  // E=200: per-lane block size ceil(200/32)=7, and 200 is NOT a multiple of 7, so lane 28
  // (indices 196..202) straddles the boundary -- 196-199 real, 200-202 sentinel -1e30f in
  // the SAME ch[] block. E=24 and E=168 above both happen to divide evenly by their own
  // per (24/1, 168/6), so no case before this one exercised a lane whose ch[] mixes real
  // and sentinel indices. mode 5 deterministically forces indices 196-199 into the top-8
  // (see run_rtop8) and asserts they were actually selected, rather than hoping random
  // data lands there -- proving by TEST what the per-index `e<E` check was proven by
  // reading (both kernels agree bitwise on a selection that requires that check to fire).
  fail |= run_rtop8(5, 4, 200, 0.0f,  1, 1.0f,   "top8 E=200 (lane straddles E boundary)");
  fail |= run_rtop8(0, 1, 257, 0.0f,  1, 1.0f,   "top8 E=257 (>256, auto-serial-fallback)");
  printf(fail? "metal backend tests: FAILED\n" : "metal backend tests: ok\n");
  coli_metal_shutdown();
  return fail;
}
