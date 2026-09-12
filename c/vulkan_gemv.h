/* vulkan_gemv.h -- optional Vulkan compute backend for qwen36 MoE experts.
 *
 * This module accelerates the routed-expert GEMVs (matmul_q hot path) with a
 * Vulkan compute shader. It is transparent: vg_init() returns -1 if no usable
 * Vulkan device exists, and the engine keeps using its CPU path in that case.
 *
 * Design notes (why it is shaped this way):
 *  - Weights are NOT re-uploaded every token. They are mirrored into a
 *    host-visible *coherent* GPU buffer (one slot per CPU LRU slot) on first
 *    load, keyed by (layer, li) so it stays in lock-step with the CPU cache.
 *    For integrated GPUs this is just system RAM, so "upload" is a memcpy.
 *  - A whole (token, layer) of experts is computed in exactly TWO dispatches
 *    (phase1: gate+up batched; phase2: down batched), amortizing submit/stall
 *    overhead that would otherwise dominate tiny per-expert GEMVs.
 *  - Int8 weights are stored as raw bytes packed 4-per-uint32 and unpacked in
 *    the shader, so no 8-bit-storage extension is required and memory is tiny.
 */
#ifndef VULKAN_GEMV_H
#define VULKAN_GEMV_H

#include <stdint.h>

typedef struct {
    int n_layers;
    int hidden;   /* D  */
    int inter;    /* Ih */
    int cap;      /* experts cached per layer (mirror of CPU LRU cap) */
    int topk;     /* K  */
    int weight_bits; /* 4 = int4-packed weights (float-compute GEMV),
                        8 (or 0) = int8/int4-container stored as int8, float-compute.
                        int4 uses ~half the GPU weight memory and the safe
                        Shader-only shader path (no int8/dot-product). */
} vg_cfg;

/* Returns 0 on success (GPU usable), -1 if no device / init failed.
 * On -1 the engine must keep using the CPU path. */
int  vg_init(const vg_cfg *cfg);

/* 1 if a working GPU device is present and initialized. */
int  vg_ready(void);

/* 1 if the int4 (packed, Shader-only) GPU path is active. The engine uses this
 * to decide whether to retain packed int4 expert weights for the GPU shader. */
int  vg_use_int4(void);

void vg_shutdown(void);

/* Called by the engine's load_expert_merged() for every expert load, using the
 * SAME local slot index `li` the CPU LRU assigned. Uploads the expert's int8
 * weights + per-row scales into the matching GPU slot (no-op if !vg_ready()). */
void vg_expert_loaded(int layer, int eid, int li,
                      const int8_t *g, const int8_t *u, const int8_t *d,
                      const float *gs, const float *us, const float *ds);

/* Lightweight per-token gate: mirror of vg_expert_loaded but skips the upload
 * when the GPU slot already holds this eid (keeps GPU LRU in lock-step with the
 * CPU LRU). Called once per selected expert before vg_moe_run(). */
void vg_expert_ensure(int layer, int li, int eid,
                      const int8_t *g, const int8_t *u, const int8_t *d,
                      const float *gs, const float *us, const float *ds);

/* int4 variants: weights are packed 2 nibbles/byte (8 int4 per uint32). Same
 * scales/activation semantics as the int8 path. Used when cfg.weight_bits==4.
 * The int4 GPU shader only needs OpCapability Shader, so it runs on drivers
 * whose int8/dot-product compiler path is broken (AMD Radeon 780M 0x800184). */
void vg_expert_loaded_int4(int layer, int eid, int li,
                           const uint8_t *g, const uint8_t *u, const uint8_t *d,
                           const float *gs, const float *us, const float *ds);
void vg_expert_ensure_int4(int layer, int li, int eid,
                           const uint8_t *g, const uint8_t *u, const uint8_t *d,
                           const float *gs, const float *us, const float *ds);

/* Run one (token, layer) of the routed-expert forward on the GPU.
 *   handles[k] = global GPU slot index (layer*cap + li) for selected expert k
 *   val[k]     = router weight (already renormalized)
 *   xs         = input activation [hidden]
 *   out        = output row [hidden]; routed-expert results are ADDED into it
 *                (the caller still adds the shared expert on CPU afterwards). */
void vg_moe_run(int layer, int K, const int *handles, const float *val,
                const float *xs, float *out);

#endif /* VULKAN_GEMV_H */
