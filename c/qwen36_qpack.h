/* qwen36_qpack.h -- optional MLX-affine routed-expert path for the qwen36
 * engine, fed by a Swiftlet qpack v1 expert container: parity with the CPU
 * affine reference from a bounded pool of expert slots.
 *
 * When a container is opened, moe() computes every routed expert through
 * checked ColiAffineQuantizedView descriptors into a BOUNDED pool of
 * refillable expert slots: each slot is one page-aligned expert_stride blob,
 * registered with the Metal backend exactly once as a whole-slot buffer and
 * addressed by byte offset at dispatch (coli_metal_matmul_affine_slot) on
 * Apple builds, or run through the portable CPU reference everywhere else.
 * A miss refills the least-recently-used slot in place and bumps its
 * generation; dispatch re-checks the generation, so a stale reference is
 * refused instead of consuming another expert's bytes.  Router, SwiGLU
 * combine, shared expert, attention, and DeltaNet stay on the CPU -- this is
 * NOT a whole-GPU forward pass.
 *
 * Compiled only when the build sets -DQWEN36_QPACK (Apple METAL=1 engine
 * builds and the qpack parity tests); otherwise the inline stubs below keep
 * the engine on its QT int8/int4 expert path with zero overhead -- the same
 * arrangement as the CUDA tier in qwen36_tier.h. */
#ifndef QWEN36_QPACK_H
#define QWEN36_QPACK_H
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Handle to one slot fill: `slot` names the pool entry, `generation` the
 * fill it was captured against.  qq_ref_forward re-checks the generation on
 * every dispatch, so a ref held across an eviction/refill is refused rather
 * than silently reading the expert that replaced it. */
typedef struct {
    int32_t  slot;
    uint32_t generation;
} QqExpertRef;

#ifdef QWEN36_QPACK

/* Open a qpack container for the loaded model's routed experts and validate
 * its geometry: layerCount/expertCount against the config, and expert (0,0)'s
 * gate/up/down projections against [inter,hidden]/[inter,hidden]/[hidden,inter].
 * Returns 1 when the store is active, 0 with a message in `error` otherwise.
 * Single-threaded by contract: only the engine's MoE loop may touch the store
 * (the PILOT prefetch worker never does). */
int  qq_open(const char *container_dir, int n_layers, int n_experts,
             int hidden, int inter, char *error, size_t error_capacity);
int  qq_active(void);

/* Bound the slot pool for the NEXT qq_open; n <= 0 restores the default (96
 * slots).  The pool is always clamped to the container's total expert count,
 * so the default keeps the tiny test fixtures fully resident with headroom
 * while bounding memory on real containers.  Engine builds expose this as
 * QWEN36_QPACK_SLOTS=<n>. */
void qq_set_slot_count(int n_slots);

/* Pool observability: slot count, evictions (a live expert displaced), and
 * fills (every expert read into a slot, first fills included).  Tests use
 * this to PROVE eviction/refill happened during a parity run instead of
 * trusting that an undersized pool was exercised. */
void qq_slot_stats(int *n_slots, uint64_t *evictions, uint64_t *fills);

/* 1 = force every projection through the CPU affine reference even when the
 * Metal affine pipelines are ready.  This is the parity oracle switch: the
 * one-layer and logit parity gates run the same engine code twice and compare
 * the two dispatch modes. */
void qq_force_cpu(int force);

/* Split forward.  qq_expert_acquire loads (layer,eid) into a slot -- a hit
 * touches LRU state, a miss refills the least-recently-used slot in place
 * (one pread) under a NEW generation -- and captures a generation-checked
 * ref.  qq_ref_forward computes out[hidden] += weight * expert(x) against
 * that exact fill and refuses a stale ref (returns 0 without touching out).
 * The split keeps the eviction contract testable and gives future async
 * begin/end orchestration a checked handle to carry across the overlap;
 * today's dispatch is synchronous, so a refused ref can only mean a caller
 * held it across another acquire. */
int  qq_expert_acquire(int layer, int eid, QqExpertRef *ref);
int  qq_ref_forward(QqExpertRef ref, const float *x, float weight,
                    float *out);

/* out[hidden] += weight * down(silu(gate(x)) * up(x)) for one routed expert:
 * acquire + ref_forward in one call.  Returns 1 on success, 0 on
 * read/descriptor/compute failure (the caller must treat 0 as fatal -- there
 * is no other expert source to fall back to once the container owns the
 * routed experts). */
int  qq_expert_forward(int layer, int eid, const float *x, float weight,
                       float *out);

/* Cumulative projection dispatch counts, so tests can PROVE which path ran
 * instead of trusting a silent fallback: coli_metal_matmul_affine_slot
 * returns 0 for CPU fallback by contract, and this is where that becomes
 * visible. */
void qq_counts(uint64_t *metal_projections, uint64_t *cpu_projections);
void qq_close(void);

#else /* !QWEN36_QPACK: inline stubs, engine keeps its QT expert path */

static inline int qq_open(const char *a, int b, int c, int d, int e,
                          char *error, size_t error_capacity) {
    (void)a;(void)b;(void)c;(void)d;(void)e;
    /* Loud, not silent: QWEN36_QPACK=<dir> on a build without the path is a
     * refusal, never an engine that quietly computes different experts. */
    if (error && error_capacity)
        snprintf(error, error_capacity,
                 "engine built without qpack affine support (QWEN36_QPACK)");
    return 0;
}
static inline int  qq_active(void){return 0;}
static inline void qq_set_slot_count(int a){(void)a;}
static inline void qq_slot_stats(int *a,uint64_t *b,uint64_t *c){if(a)*a=0;if(b)*b=0;if(c)*c=0;}
static inline void qq_force_cpu(int a){(void)a;}
static inline int  qq_expert_acquire(int a,int b,QqExpertRef *c){(void)a;(void)b;(void)c;return 0;}
static inline int  qq_ref_forward(QqExpertRef a,const float*b,float c,float*d){(void)a;(void)b;(void)c;(void)d;return 0;}
static inline int  qq_expert_forward(int a,int b,const float*c,float d,float*e){(void)a;(void)b;(void)c;(void)d;(void)e;return 0;}
static inline void qq_counts(uint64_t *a,uint64_t *b){if(a)*a=0;if(b)*b=0;}
static inline void qq_close(void){}

#endif /* QWEN36_QPACK */
#endif /* QWEN36_QPACK_H */
