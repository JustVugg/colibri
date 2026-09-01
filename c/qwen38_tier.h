/* qwen38_tier.h -- optional CUDA VRAM expert tier for the qwen38 engine.
 *
 * WHY THIS EXISTS
 *
 * `COLI_TIMERS=1` on an RTX 3090 / i9-14900K, 18-token prompt, warm, per
 * forward:
 *
 *     routed-expert   265.3 ms      <- this file
 *     resident-mm     191.9 ms
 *     deltanet        150.4 ms
 *     shared-expert    26.7 ms
 *     lm-head          22.8 ms
 *     qsa-attn         11.8 ms
 *     ple               2.2 ms
 *     qsa-index         1.4 ms
 *     expert-read       0.000 ms    <- no disk wait at all
 *     fp8-expand        0.000 ms
 *
 * Two things follow. The routed experts are the single largest cost, and the
 * engine is COMPUTE-bound rather than disk-bound: at cache 424/layer nothing is
 * fetched from disk during decode. That is the opposite of the GLM engine,
 * where expert I/O is ~48% of the run, and it is why this tier targets the
 * matmul rather than the read.
 *
 * WHAT IT DOES NOT DO
 *
 * Nothing else moves to the GPU in this phase - resident matmuls, DeltaNet,
 * QSA, PLE and the shared expert all stay on the CPU. Same deliberately small
 * scope as backend_cuda_ink.h.
 *
 * THE WEIGHTS NEED NO NEW KERNEL
 *
 * qwen38's routed experts are official block-FP8: raw e4m3 bytes plus one f32
 * scale per 128x128 block, `fp8_nblk(rows) * fp8_nblk(cols)` of them
 * (quant.h:517). backend_cuda.cu's fmt=8 decodes exactly that layout -
 * `scales + (o>>7)*((I+127)>>7)`, then `scl[i>>7]` (backend_cuda.cu:534). The
 * formats are byte-identical, so this tier is wiring over the existing
 * expert-group API and adds no numerics of its own.
 *
 * GEOMETRY on Qwen3.8-Flash-Next: hidden 2560, moe_intermediate 640, 512
 * experts across 48 layers, top-k 10. Each expert is
 * gate[640,2560] + up[640,2560] + down[2560,640] = 4.69 MiB in fp8, so the full
 * set is ~112.6 GiB and a 24 GB card holds roughly a fifth of it. Residency is
 * therefore a routing-skew bet, exactly as it is for GLM: hot experts earn
 * VRAM, everything else stays on the CPU path.
 *
 * DEFAULT OFF, AND THAT IS DELIBERATE. The Makefile calls the qwen38 target
 * "intentionally CPU-only". Built without -DCOLI_CUDA the inline stubs below
 * compile to nothing and the engine is byte-for-byte what it is today; built
 * with it, the tier still does nothing until COLI_CUDA=1 is set at runtime.
 * This adds a capability that was scoped out. It does not change the existing
 * path.
 *
 * Enable with COLI_CUDA=1 [COLI_GPUS=0,1] [CUDA_EXPERT_GB=<G>|auto].
 */
#ifndef QWEN38_TIER_H
#define QWEN38_TIER_H
#include <stdint.h>

#ifdef COLI_CUDA

/* Init after model load, before the first token. Returns 1 when the tier is
 * live. A zero return is not an error: it means CPU-only, and every call below
 * then behaves as a miss. */
int  q38t_init(int n_layers, int n_experts, int hidden, int inter, int topk);
int  q38t_ready(void);
void q38t_shutdown(void);

/* True when this expert's three projections are resident in some device's VRAM
 * and can be issued. */
int  q38t_is_resident(int layer, int eid);

/* Offer one routed expert to the tier: updates its heat and may enqueue a
 * background upload. Pointers are into the engine's own Slot, which the tier
 * must not free and must not outlive.
 *
 * fp8 weights are raw e4m3 bytes; scales are the 128x128 block table. Passing a
 * non-FP8 expert is safe and is simply ignored - a checkpoint can legitimately
 * carry bf16 experts and those stay on the CPU path. */
void q38t_note(int layer, int eid,
               const void *g, const float *gs,
               const void *u, const float *us,
               const void *d, const float *ds);

/* Launch the GPU groups for whichever of the K selected experts are resident.
 * Returns a bitmask of the k handled on the GPU; compute the rest on the CPU
 * and then call q38t_take() to fold in the GPU half. Asynchronous, so the CPU
 * misses overlap with the in-flight groups rather than waiting behind them. */
uint32_t q38t_issue(int layer, const int *eids, int K, const float *x);

/* Accumulate route_gates[k] * y_k into out[hidden] for the k in `mask`. */
void q38t_take(uint32_t mask, const float *route_gates, int K, float *out);

/* One telemetry block on stderr: residency, hits, misses, uploads per device.
 * Silent unless the tier ran, so a CPU-only run prints nothing new. */
void q38t_stats(void);

#else /* !COLI_CUDA: inline stubs, engine stays exactly as it is today */

static inline int  q38t_init(int a,int b,int c,int d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int  q38t_ready(void){return 0;}
static inline void q38t_shutdown(void){}
static inline int  q38t_is_resident(int a,int b){(void)a;(void)b;return 0;}
static inline void q38t_note(int a,int b,const void*c,const float*d,const void*e,const float*f,const void*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline uint32_t q38t_issue(int a,const int*b,int c,const float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline void q38t_take(uint32_t a,const float*b,int c,float*d){(void)a;(void)b;(void)c;(void)d;}
static inline void q38t_stats(void){}

#endif /* COLI_CUDA */
#endif /* QWEN38_TIER_H */
