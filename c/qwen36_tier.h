/* qwen36_tier.h -- optional CUDA VRAM expert tier for the qwen36 engine.
 *
 * Applies colibri's placement concept ("route -> place -> overlap -> learn")
 * one level up from the GLM disk tier: experts live in RAM, the *hot* ones
 * are promoted into DEVICE_LOCAL VRAM across one or more GPUs and computed
 * there via the existing CUDA backend (backend_cuda.cu, expert-group API).
 *
 *  - Every expert has one home device (eid % n_gpus), no duplicates.
 *  - Routing heat decides who earns VRAM (LFRU semantics from tier.h, with
 *    hysteresis); a warmstart pre-fills the budget before the first token,
 *    ordered by a persisted heat table (HEAT_FILE) when available.
 *  - Uploads run on a background thread through staging copies; decode never
 *    blocks on placement. A VRAM miss falls back to the CPU int8 path and
 *    overlaps with the in-flight GPU groups.
 *
 * Enable with COLI_CUDA=1 [COLI_GPUS=0,1] [CUDA_EXPERT_GB=<G>|auto]
 * [HEAT_FILE=<path>] [QT_NO_WARMSTART=1]. Compiled only when the build sets
 * -DCOLI_CUDA (CUDA=1); otherwise the inline stubs below keep the engine
 * CPU-only with zero overhead. */
#ifndef QWEN36_TIER_H
#define QWEN36_TIER_H
#include <stdint.h>

#ifdef COLI_CUDA

/* Init after model load. Returns 1 when the tier is active.
 * cap_experts_per_layer must equal n_experts (full RAM residency): the tier
 * stores raw pointers into the expert slots, which must never be evicted. */
/* expert_is_int4: 1 = pesi int4 impacchettati (fmt=4), 0 = int8 (fmt=1). Il
 * chiamante lo determina dalla TAGLIA SU DISCO, non da meta.ebits, che su
 * qualche container mente (cfr. il rilevamento in qwen36.c). */
/* R4 role split: park the dense-i8 lm_head on its own CUDA device
 * (COLI_LMHEAD_GPU=<dev>). One GEMV per token, at token end — outside the
 * per-layer latency chain — so a slower second card can host it without
 * pacing the expert stream. qt_init places no experts on that device. */
int  qt_lmhead_init(const int8_t *q, const float *sc, int I, int O);
int  qt_lmhead_matmul(float *y, const float *x, int I, int O);

/* ---- placement table (R4) ------------------------------------------------
 * Every movable piece of the forward pass can be pinned to the CPU or to a
 * specific CUDA device, so configurations can be A/B'd instead of argued
 * about. One variable, not one per component:
 *
 *   COLI_PLACE="experts=0,lmhead=0,dnproj=1,dnout=1,attnproj=cpu"
 *
 * Target is `cpu` or a CUDA ordinal. A component may also be split across
 * cards by layer count, joined with '+' so it cannot be confused with the
 * component separator:
 *
 *   COLI_PLACE="dnproj=0:15+1:15"   first 15 DeltaNet layers on dev 0, rest on 1
 *
 * Unnamed components keep their default (CPU; experts keep following
 * COLI_GPUS). Splitting matters because the DeltaNet projections hang
 * serially in the layer chain anyway -- a slower card delays only its own
 * layers, never the whole stream, which is what sank asymmetric EXPERT
 * placement. `layer` is the model layer index; pass 0 for whole-model
 * components like lmhead. */
#define QT_PLACE_CPU (-1)
int  qt_place_of(const char *component, int layer);

/* DeltaNet input projections, qkv ++ z fused into one resident tensor per
 * layer: one GEMV instead of two, and the engine's qkv/z buffers are laid out
 * contiguously so the result needs no split copy. */
int  qt_dnproj_init(int layer, const int8_t *q, const float *sc,
                    int I, int O, int device);
int  qt_dnproj_matmul(int layer, float *y, const float *x, int I, int O);

int  qt_init(int n_layers, int n_experts, int hidden, int inter,
             int cap_experts_per_layer, int topk, int expert_gs,
             int expert_is_int4);
int  qt_ready(void);
int  qt_is_resident(int layer, int eid);
void qt_shutdown(void);

/* Call once per routed expert per token (pointers to the RAM slot: packed
 * int4 + per-row scales). Updates heat and may enqueue a background upload. */
void qt_note(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);

/* Launch the GPU groups for the resident subset of the K selected experts
 * (async, all devices in parallel). Returns a bitmask of the k handled by
 * the GPU. Compute the misses on the CPU, then call qt_take(). */
uint32_t qt_issue(int layer, const int *eids, int K, const float *x);

/* Collect the GPU results and accumulate val[k]*y_k into out[hidden]. */
void qt_take(uint32_t mask, const float *val, int K, float *out);

/* Warmstart: plan the full fill set (heat order, budget reserved), then any
 * number of loader threads may call qt_note_planned per planned expert. */
int  qt_plan_fill(int *layers, int *eids, int max);
void qt_note_planned(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);
int  qt_fill_next(int *layer, int *eid);
void qt_note_block(int layer, int eid,
             const uint8_t *g4, const uint8_t *u4, const uint8_t *d4,
             const float *gs, const float *us, const float *ds);
void qt_fill_wait(void);   /* blocks until the upload queue is drained */

/* One telemetry block on stderr: residency, hits/misses, uploads per device. */
void qt_stats(void);

#else /* !COLI_CUDA: inline stubs, engine stays CPU-only */

static inline int  qt_init(int a,int b,int c,int d,int e,int f,int g,int h){(void)h;(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
static inline int  qt_lmhead_init(const int8_t*a,const float*b,int c,int d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline int  qt_lmhead_matmul(float*a,const float*b,int c,int d){(void)a;(void)b;(void)c;(void)d;return 0;}
#define QT_PLACE_CPU (-1)
static inline int  qt_place_of(const char*a,int b){(void)a;(void)b;return QT_PLACE_CPU;}
static inline int  qt_dnproj_init(int a,const int8_t*b,const float*c,int d,int e,int f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return 0;}
static inline int  qt_dnproj_matmul(int a,float*b,const float*c,int d,int e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline int  qt_ready(void){return 0;}
static inline int  qt_is_resident(int a,int b){(void)a;(void)b;return 0;}
static inline void qt_shutdown(void){}
static inline void qt_note(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline uint32_t qt_issue(int a,const int*b,int c,const float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline void qt_take(uint32_t a,const float*b,int c,float*d){(void)a;(void)b;(void)c;(void)d;}
static inline int  qt_plan_fill(int*a,int*b,int c){(void)a;(void)b;(void)c;return 0;}
static inline void qt_note_planned(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline int  qt_fill_next(int*a,int*b){(void)a;(void)b;return 0;}
static inline void qt_note_block(int a,int b,const uint8_t*c,const uint8_t*d,const uint8_t*e,const float*f,const float*g,const float*h){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;}
static inline void qt_fill_wait(void){}
static inline void qt_stats(void){}

#endif /* COLI_CUDA */
#endif /* QWEN36_TIER_H */
