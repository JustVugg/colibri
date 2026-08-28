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
int  qt_init(int n_layers, int n_experts, int hidden, int inter,
             int cap_experts_per_layer, int topk, int expert_gs);
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

static inline int  qt_init(int a,int b,int c,int d,int e,int f,int g){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;return 0;}
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
