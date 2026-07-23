#ifndef COLIBRI_BACKEND_VULKAN_H
#define COLIBRI_BACKEND_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque persistent device copy of one resident quantized tensor,
 * mirroring backend_cuda.h. On Strix Halo the "upload" writes into
 * HOST_VISIBLE|DEVICE_LOCAL memory — same physical RAM the iGPU reads,
 * so there is no PCIe copy, unlike the discrete-CUDA path. */
typedef struct ColiVkTensor ColiVkTensor;

/* Bring up instance/device/queue/pipeline. Returns 1 on success.
 * spv_path points at the compiled qmatmul.spv. */
int  coli_vk_init(const char *spv_path);
void coli_vk_shutdown(void);
int  coli_vk_available(void);
void coli_vk_mem_info(size_t *used_bytes, size_t *tensor_count);

/* VRAM pressure-proofing (both no-ops when the extension is absent):
 * alloc_priority sets the eviction-priority class of SUBSEQUENT weight uploads
 * (VK_EXT_memory_priority; scratches and the KV mirror pin themselves at 1.0) —
 * the engine brackets the bulk expert-tier fill at 0.4 so an oversubscribed heap
 * evicts cold experts, never the per-token attention working set.
 * mem_budget reports device-local usage/budget in GB (VK_EXT_memory_budget);
 * returns 0 if unavailable. */
void coli_vk_alloc_priority(float p);
/* Fused gate+up activation (model-global): 0 = silu(gate)*up (GLM), 1 = swigluoai (M3). */
void coli_vk_set_activation(int act, float alpha, float limit);
int  coli_vk_mem_budget(double *used_gb, double *budget_gb);

/* y[S,O] = (x[S,I] @ dequant(W[O,I])^T) * scale[O].
 * fmt matches QT in glm.c: 1=int8, 2=int4. (0=f32,3=int2 fall back to CPU.)
 * First call uploads W+scales; later calls reuse the resident copy.
 * Returns 1 on success, 0 if unavailable / unsupported fmt. */
int  coli_vk_matmul(ColiVkTensor **tensor,
                    float *y, const float *x,
                    const void *weights, const float *scales,
                    int fmt, int S, int I, int O, int gs);

/* Fused first half of the expert MLP in ONE dispatch (VK equivalent of
 * grouped_hidden_w4_dual): hidden[s,o] = silu(gate(x)) * up(x), reading x once for both
 * projections. D = input (hidden) dim, I = moe_inter. gate/up upload on first call.
 * Returns 0 if unavailable (no gate_up shader) / unsupported fmt so the caller falls back. */
int  coli_vk_gate_up(ColiVkTensor **gate, ColiVkTensor **up,
                     float *hidden, const float *x,
                     const void *gw, const float *gs,
                     const void *uw, const float *us,
                     int fmt, int S, int D, int I, int grp);

/* Full batched expert MLP for `count` experts in ONE submit, hidden staying on-device:
 * for each c, y_c = down_c(silu(gate_c(x_c)) * up_c(x_c)). x/y packed [sum(rows)*D];
 * experts are resident (gate/up: D->I, down: I->D). Mirrors coli_cuda_expert_group.
 * Returns 0 -> caller falls back to CPU. */
int  coli_vk_expert_group(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                          ColiVkTensor *const *downs, const int *rows, int count,
                          float *y, const float *x);
/* Async form: _issue submits the group and returns immediately (one in flight max);
 * the caller computes its CPU share, then _take joins and reads back the packed y.
 * Both return 0 on failure (caller computes those experts on the CPU instead). */
int  coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                ColiVkTensor *const *downs, const int *rows, int count,
                                const float *x);
int  coli_vk_expert_group_take(float *y);

/* Upload a resident tensor without computing (expert tier: gate/up/down uploaded once,
 * then driven by coli_vk_expert_group). Returns 0 on failure/unsupported fmt. */
int  coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);

/* SECOND DEVICE (COLI_VK_DEV2): a self-contained context on another Vulkan GPU that
 * hosts ONLY tier experts and runs ONLY the async expert-group path. devidx: -1 =
 * auto (best real GPU that is not device 0), >=0 = enumeration index (the same
 * physical device is allowed with a warning — pre-hardware test mode). Its group
 * may be in flight simultaneously with device 0's. Tensors remember their device
 * (coli_vk_tensor_dev); free/bytes work on either. */
int  coli_vk_init_dev2(const char *spv_path, int devidx);
int  coli_vk_dev2_available(void);
int  coli_vk_tensor_dev(const ColiVkTensor *t);
int  coli_vk_mem_budget2(double *used_gb, double *budget_gb);
int  coli_vk_tensor_ensure2(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);
int  coli_vk_expert_group_issue2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                 ColiVkTensor *const *downs, const int *rows, int count,
                                 const float *x);
int  coli_vk_expert_group_take2(float *y);
int  coli_vk_expert_group2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                           ColiVkTensor *const *downs, const int *rows, int count,
                           float *y, const float *x);

/* MLA absorb attention core (decode). The KV latent/rope caches live in persistent
 * per-layer device buffers: _ensure allocates a layer's cache at max_rows (once; resize
 * via _reset), _row mirrors one host row (absolute position), _reset drops all layers.
 * The caller keeps a valid-watermark and re-mirrors rows after any invalidation.
 * absorb runs S causal query rows over cache rows [st0, T) in one submit:
 * q [S,H*(Q+R)] roped, kv_b [H*(Q+V), K] uploads once (fmt 1=int8/2=int4),
 * ctx out [S,H*V]. Returns 0 -> caller falls back to CPU. */
int  coli_vk_kv_ensure(int layer, int max_rows, int K, int Rd);
int  coli_vk_kv_row(int layer, int pos, const float *L, const float *R);
void coli_vk_kv_reset(void);
int  coli_vk_attention_absorb(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                              float *ctx, const float *q, int layer, int S, int H,
                              int Q, int R, int V, int K, int st0, int T, float scale);
/* Two resident matmuls sharing one input x in ONE submit (q_a + kv_a prologue pair).
 * Returns 0 -> caller falls back to single-matmul calls. */
/* q-prep chain: [q_a+kv_a pair] -> rmsnorm(q latent) -> q_b in ONE submit (needs
 * rmsnorm.spv next to the main shader; returns 0 without it -> 3-submit path).
 * lnw = the q-latent RMS-norm weights [Oqa], resident per layer after first call. */
int  coli_vk_attn_qprep(int layer,
                        ColiVkTensor **qa,  const void *wqa,  const float *sqa,  int Oqa,
                        ColiVkTensor **kva, const void *wkva, const float *skva, int Okva,
                        ColiVkTensor **qb,  const void *wqb,  const float *sqb,  int Oqb,
                        int fmt, int grp, const float *lnw, float eps,
                        const float *x, int S, int I, float *q_out, float *kv_out,
                        float *lat_out /* normed q latent [S,Oqa], NULLable — DSA indexer input */);
int  coli_vk_matmul_pair(ColiVkTensor **t1p, float *y1, const void *w1, const float *s1, int O1,
                         ColiVkTensor **t2p, float *y2, const void *w2, const float *s2, int O2,
                         int fmt, const float *x, int S, int I, int grp);

/* Fused variant: absorb + resident o-projection ([Dout, H*V]) in one submit; ctx stays
 * on-device, only out [S,Dout] is read back. Falls back like absorb (returns 0). */
int  coli_vk_attention_absorb_project(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                              ColiVkTensor **ot, const void *ow, const float *osc, int ofmt, int ogrp,
                              float *out, const float *q, int layer, int S, int H,
                              int Q, int R, int V, int K, int st0, int T, float scale, int Dout);

void   coli_vk_tensor_free(ColiVkTensor *t);
size_t coli_vk_tensor_bytes(const ColiVkTensor *t);

#ifdef __cplusplus
}
#endif

#endif
