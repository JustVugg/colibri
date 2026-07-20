/* Regression for non-finite-logit poisoning of sampling.
 *
 * A single NaN or +Inf in the logits (a bad streamed expert tile, or an fp
 * overflow in the matmul at a low-RAM eviction boundary) used to make softmax
 * produce an all-NaN g_pbuf; dist_sample then never satisfied cum>=u and fell
 * through to return token 0 — so the engine silently emitted an unbroken run of
 * token 0 with no error, on the DEFAULT serve path (TEMP>0, 0<NUCLEUS<1).
 *
 * Fix under test: argmax_v() skips NaN (picks the max finite/+Inf entry instead
 * of being pinned to index 0), and dist_build() detects a non-finite softmax sum
 * and collapses to a one-hot over the finite argmax (warning once) rather than
 * dividing every entry into NaN. Degrade + diagnose, never silently corrupt.
 *
 * No model file needed: exercises argmax_v / dist_build / dist_sample directly. */
#include <assert.h>
#include <math.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int approx1(double x){ return x > 0.999 && x < 1.001; }

int main(void){
    /* --- argmax_v must skip NaN (greedy decode + speculative-verify paths) --- */
    { float lo[8]={NAN,1.f,5.f,2.f,NAN,-3.f,4.f,0.f};
      assert(argmax_v(lo,8)==2 && "pick max finite (idx2=5.0), not NaN-pinned idx0"); }
    { float lo[8]={3.f,INFINITY,1.f,2.f,0.f,-1.f,2.5f,1.5f};
      assert(argmax_v(lo,8)==1 && "pick the +Inf position"); }
    { float lo[8]; for(int i=0;i<8;i++) lo[i]=NAN;
      assert(argmax_v(lo,8)==0 && "all-NaN: no crash, defined fallback"); }

    g_temp=0.7f; g_nuc=0.9f;   /* the default serve/chat sampling path */

    /* --- dist_build: a NaN logit must yield a finite one-hot, not all-NaN --- */
    { float lo[8]={0.5f,1.f,NAN,8.f,0.2f,-1.f,0.f,0.3f};   /* max finite = idx3 (8.0) */
      dist_build(lo,8);
      double sum=0; int nan=0;
      for(int i=0;i<8;i++){ if(!(g_pbuf[i]==g_pbuf[i])) nan=1; sum+=g_pbuf[i]; }
      assert(!nan && "g_pbuf must be finite after a NaN logit");
      assert(approx1(sum) && "g_pbuf must normalize to 1");
      assert(approx1(g_pbuf[3]) && "mass must land on the max finite logit (idx3)");
      assert(dist_sample(8,-1)==3 && "sampler emits the finite argmax, not token 0"); }

    /* --- NaN at index 0: poisons the max scan itself (the old mx=lo[0] seed),
     *     the failure mode that starts before the sum (review note on #369) --- */
    { float lo[8]={NAN,1.f,0.5f,6.f,0.2f,-1.f,0.f,0.3f};   /* max finite = idx3 (6.0) */
      dist_build(lo,8);
      double sum=0; int nan=0;
      for(int i=0;i<8;i++){ if(!(g_pbuf[i]==g_pbuf[i])) nan=1; sum+=g_pbuf[i]; }
      assert(!nan && "NaN at lo[0] must not poison via the mx seed");
      assert(approx1(sum) && "still normalizes to 1");
      assert(dist_sample(8,-1)==3 && "emits the max finite logit, not token 0"); }

    /* --- all-NaN logits: worst case — must stay finite, no crash --- */
    { float lo[8]; for(int i=0;i<8;i++) lo[i]=NAN;
      dist_build(lo,8);
      for(int i=0;i<8;i++) assert(g_pbuf[i]==g_pbuf[i] && "all-NaN: g_pbuf stays finite"); }

    /* --- regression: clean logits still produce a valid distribution --- */
    { float lo[8]={0.1f,0.2f,3.0f,0.4f,0.5f,0.6f,0.7f,0.8f};   /* peak = idx2 */
      dist_build(lo,8);
      double sum=0; int nan=0;
      for(int i=0;i<8;i++){ if(!(g_pbuf[i]==g_pbuf[i])) nan=1; sum+=g_pbuf[i]; }
      assert(!nan && "clean softmax stays finite");
      assert(approx1(sum) && "clean softmax must sum to 1");
      assert(g_pbuf[2]>=g_pbuf[0] && "peak token keeps the most mass"); }

    printf("OK test_logit_nan: argmax_v NaN-skip + dist_build finite-collapse\n");
    return 0;
}
