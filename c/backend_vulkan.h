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
/* GLM-5.3 SwiGLU clamp for the fused gate_up kernel. 0 disables (GLM-5.2). */
void coli_vk_set_swiglu_limit(float limit);

/* VRAM pressure-proofing (both no-ops when the extension is absent):
 * alloc_priority sets the eviction-priority class of SUBSEQUENT weight uploads
 * (VK_EXT_memory_priority; scratches and the KV mirror pin themselves at 1.0) —
 * the engine brackets the bulk expert-tier fill at 0.4 so an oversubscribed heap
 * evicts cold experts, never the per-token attention working set.
 * mem_budget reports device-local usage/budget in GB (VK_EXT_memory_budget);
 * returns 0 if unavailable. */
void coli_vk_alloc_priority(float p);
int  coli_vk_mem_budget(double *used_gb, double *budget_gb);

/* Device memory books (docs/vulkan.md, "A partial chain"). Every device memory allocation
 * of the backend and of the chain is counted per device when its heap is device-local.
 * COLI_VK_DEVICE_CAP_MB=n (tests; n may be a fraction) makes the device hold at most n MiB:
 * an allocation past it fails as out of device memory, and coli_vk_mem_budget (and
 * _budget2, coli_vk_free_bytes, coli_vk_device_local_bytes) report the cap and this
 * process's bytes, so a CPU device behaves like a small card.
 *   coli_vk_device_cap      the cap in bytes (0: none)
 *   coli_vk_device_used     device-local bytes this process holds on the device
 *   coli_vk_free_bytes      free device memory now: the cap less what is held, else the
 *                           budget less its usage (VK_EXT_memory_budget), else the largest
 *                           device-local heap less what is held
 *   coli_vk_block_bytes     the block size a memory pool of `def` bytes takes (smaller
 *                           under the cap, at most about 4096 blocks to fill it)
 *   coli_vk_mem_alloc/free  vkAllocateMemory / vkFreeMemory through the books (vk_chain.c):
 *                           info is a VkMemoryAllocateInfo *, memory a VkDeviceMemory *;
 *                           the result is a VkResult */
size_t coli_vk_device_cap(void);
size_t coli_vk_device_used(void);
size_t coli_vk_free_bytes(void);
size_t coli_vk_block_bytes(size_t def);
size_t coli_vk_block_bytes_dev(int dev, size_t def);   /* the same for device 0 or COLI_VK_DEV2's (1) */
int    coli_vk_mem_alloc(void *device, const void *info, void *memory);
void   coli_vk_mem_free(void *device, const void *memory);
/* COLI_VK_STAGED_FAULT (tests): whether a point is set, and how many times it was reached. */
int    coli_vk_fault_set(void);
unsigned long long coli_vk_fault_reached(void);

/* y[S,O] = (x[S,I] @ dequant(W[O,I])^T) * scale[O].
 * fmt matches QT in glm.c: 1=int8, 2=int4. (0=f32,3=int2 fall back to CPU.)
 * fmt 10 = plain f32 weights and 11 = bf16 weights (low half = even column): no
 * scales, pass NULL. Numbered apart from QT's 0 (f32), which keeps falling back.
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
 * then driven by coli_vk_expert_group). Returns 0 on failure/unsupported fmt.
 * fmt 14 = f16 weights (low half = even column), no scales, like 10 and 11. */
int  coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);
/* The same tensor with its rows read where the host keeps them (VK_EXT_external_memory_host),
 * no device copy: for a device that shares the CPU's RAM, where a copy would hold the
 * matrix twice. weights must be aligned to coli_vk_import_alignment() and its allocation
 * (alloc_bytes) must cover the rows rounded up to that alignment; the rows must need no
 * padding (cpu row bytes a multiple of 4). The host memory must outlive the tensor
 * (coli_vk_tensor_free first). 0 = not possible here: the caller uploads a copy instead.
 * coli_vk_import_alignment() is 0 where imports are not possible (no extension, staged
 * uploads); coli_vk_imported_bytes() the rows read in place now. */
int    coli_vk_tensor_import(ColiVkTensor **tensor, const void *weights, size_t alloc_bytes, const float *scales,
                             int fmt, int I, int O, int gs);
size_t coli_vk_import_alignment(void);
/* Host memory a shader reads in place (VK_EXT_external_memory_host, where
 * coli_vk_import_alignment() is not 0): the pages around [ptr, ptr + bytes) as a storage
 * buffer of the primary device (VkBuffer, VkDeviceMemory as void *), *off the byte
 * offset of ptr in it. 0 when the device refuses. The pages must stay mapped until
 * coli_vk_host_buffer_free (bytes: the buffer's, ptr's page-rounded range), after the
 * last submission that read them. */
int    coli_vk_host_buffer(const void *ptr, size_t bytes, void **buf, void **mem, size_t *off);
void   coli_vk_host_buffer_free(void *buf, void *mem, size_t bytes);
size_t coli_vk_imported_bytes(void);

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

/* Frees the tensor and gives its device memory back to its pool (the next upload
 * reuses it). Safe from any thread; a free while async work is in flight on the
 * tensor's device takes effect when that work has been joined. */
void   coli_vk_tensor_free(ColiVkTensor *t);
size_t coli_vk_tensor_bytes(const ColiVkTensor *t);

/* ---- weight memory -----------------------------------------------------------
 * Resident tensors live in a few big device-memory blocks per pool, handed out by
 * an offset allocator (vk_alloc.h) that takes freed ranges back. Pool 0 holds every
 * engine's resident weights (what coli_vk_mem_info counts), pool 1 the routed-expert
 * tier's experts, pool 2 COLI_VK_DEV2's registry (colibri.c), pool 3 the tier's experts
 * on COLI_VK_DEV2's device. */
typedef struct {
    int blocks, live;                 /* device-memory blocks; ranges handed out */
    size_t total, used, free;         /* block bytes; in ranges; between them */
    size_t largest_free, peak_used;
    size_t limit;                     /* the pool's byte limit (0 = none) */
    size_t tensors, payload;          /* live tensors and their row + scale bytes */
    unsigned long long allocs, frees, refusals;   /* refusals: a block over the limit */
    double frag;                      /* 1 - largest free extent / free bytes */
} ColiVkPoolStats;
void coli_vk_pool_stats(int pool, ColiVkPoolStats *st);
/* The expert tier's budget: its pool never holds more block bytes than this. */
void coli_vk_tier_pool_limit(size_t bytes);
void coli_vk_tier_pool_limit_dev(int dev, size_t bytes);   /* dev 1: COLI_VK_DEV2's pool (3) */
/* The tier's extra layers (an MTP head's experts, another size than the main ones'):
 * a pool of their own on the primary device (pool 4), its budget, and a tensor in it
 * (as coli_vk_tier_tensor). */
void coli_vk_tier_extra_pool_limit(size_t bytes);
int  coli_vk_tier_tensor_extra(ColiVkTensor **t, int fmt, int I, int O, int gs,
                               uint8_t **rows, size_t *stride, float **scales);
/* A tensor in the tier's pool, to fill in place: O rows of coli_vk_tensor_row_bytes
 * at *stride apart (padding zeroed) and coli_vk_tensor_scale_count floats of scales
 * (fmt 10/11: one, set it to 1), then coli_vk_tensor_commit. Thread-safe. Returns 0
 * at the budget or when the device is out of memory. */
int    coli_vk_tier_tensor(ColiVkTensor **t, int fmt, int I, int O, int gs,
                           uint8_t **rows, size_t *stride, float **scales);
/* Staged uploads (a discrete card without Resizable BAR, or COLI_VK_STAGED=1; see
 * docs/vulkan.md, "Memory placement without Resizable BAR"): the tier's tensors live in
 * device memory the host does not map, so the rows and scales coli_vk_tier_tensor hands
 * out are a host image, which this copies to the device (all n before it returns) and
 * frees. With mapped memory it does nothing. Thread-safe; the n tensors on one device.
 * 0 = the copy failed (the device is lost): free the tensors. */
int    coli_vk_tensor_commit(ColiVkTensor *const *t, int n);
int    coli_vk_staged(void);   /* 1 = resident data goes to the device through staged uploads */
size_t coli_vk_tensor_row_bytes(int fmt, int I);
/* What coli_vk_tensor_ensure(fmt, I, O, gs) adds to coli_vk_mem_info's bytes: the rows
 * at their padded stride and the scales (0 for a grouped format without a group size). */
size_t coli_vk_tensor_payload(int fmt, int I, int O, int gs);
size_t coli_vk_buffer_alignment(void);   /* where a weight range may start (bytes) */
size_t coli_vk_tensor_scale_count(int fmt, int I, int O, int gs);

/* ---- async expert batch (the routed-expert tier) ------------------------------
 * One layer step of resident experts in one submit on the tier queue (a second
 * queue when the device has one, so the synchronous dense matmuls do not wait
 * behind it). Every expert: hidden = act(gate(x), up(x)), y = down(hidden), for
 * each of its rows. act: COLI_VK_ACT_SWIGLU (silu(g)*u, limit > 0 clamps the gate
 * from above and up to [-limit, limit]) or COLI_VK_ACT_SITU (a*tanh(g/a)*sigmoid(g) *
 * b*tanh(u/b)). One geometry per process: hidden D, intermediate I.
 * Threading: engine thread only, except where noted. */
#define COLI_VK_ACT_SWIGLU 0
#define COLI_VK_ACT_SITU   1
/* DeepSeek V4's expert, with the roundings its CPU kernel makes: gate and up rounded
 * to bf16, the clamped SwiGLU, times the row's route weight (coli_vk_xb_issue_w) and
 * rounded to bf16, then quantized to E4M3 and back with one power-of-two scale per
 * 128 inputs before down (expert_act_v4.spv beside the main shader). The rows given
 * are already E4M3-rounded by the caller, as the CPU kernel rounds x; down's output
 * comes back in f32 for the caller's own bf16 rounding. */
#define COLI_VK_ACT_SWIGLU_V4 2
typedef struct ColiVkExpert ColiVkExpert;
int  coli_vk_xb_init(int D, int I, int act, float limit, float a, float b);   /* again: same D, I, new act */
int  coli_vk_xb_ready(void);
int  coli_vk_xb_queue_shared(void);   /* 1 = the batch shares the main queue */
/* A resident expert from three tensors (gate/up [I x D] in one format, down [D x I]
 * in any): writes its descriptor sets once. NULL when the shapes do not match the
 * geometry. Not while a batch is in flight. */
ColiVkExpert *coli_vk_xb_expert(ColiVkTensor *gate, ColiVkTensor *up, ColiVkTensor *down);
/* Its sets and its three tensors. Not while a batch is in flight. */
void coli_vk_xb_expert_free(ColiVkExpert *e);
/* Submit and return: count experts, rows[c] activation rows each, the rows given as
 * sum(rows) pointers to D floats, expert by expert. 0 = nothing was submitted (the
 * caller computes those experts itself). One batch in flight at a time. */
int  coli_vk_xb_issue(ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows);
/* The same with one weight per input row (same order as xrows, NULL = 1), which
 * COLI_VK_ACT_SWIGLU_V4 applies before down; the other activations ignore it. */
int  coli_vk_xb_issue_w(ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                        const float *wrows);
/* Wait for it: yrows[j] points at the D outputs of input row j (same order), valid
 * until the next issue; *device_ms is its device time when timestamps exist (else 0).
 * 0 = the batch failed (device lost): the caller computes those rows itself. */
int  coli_vk_xb_join(const float **yrows, double *device_ms);
typedef struct {
    unsigned long long batches, experts, rows, gemm_experts;
    unsigned long long grouped_batches;   /* batches that took the grouped GEMM (qmatmul_grp.comp) */
    unsigned long long gemv_batches;      /* batches that took the grouped GEMV (qmatmul_grp_gemv.comp) */
    double device_ms;                 /* summed batch device time (timestamps) */
    int timestamps, queue_shared, gemm_rows;
    size_t scratch_bytes;
    unsigned long long sub_batches, sub_experts, sub_rows, sub_gemm;   /* coli_vk_xb_sub_* */
    double sub_ms;
    unsigned long long cooperative_matmuls; /* all projections, including sub-batches */
} ColiVkXbStats;
void coli_vk_xb_stats(ColiVkXbStats *st);
/* A big step's whole-step buffers on the primary device (the grouped GEMM): the step's
 * S token rows x[S*D] uploaded once; returns its S*K output rows (s*K + k, host-readable)
 * or NULL when the device cannot (the tier then packs and copies as before). Sub-batches
 * of the step go through coli_vk_xb_sub_issue_step (assign[j]: packed row j's s*K + k)
 * and land in those rows; coli_vk_xb_step_sum then adds a token's used rows in rank order
 * (w[i] * row i as an fma chain from 0) on the device and returns S*D rows, valid until
 * the next step. coli_vk_xb_step_end closes the step either way. */
float *coli_vk_xb_step_begin(int S, int K, const float *x);
int   coli_vk_xb_sub_issue_step(int h, ColiVkExpert *const *ex, const int *rows, int count, const int *assign,
                                const float *wrows);   /* wrows: as coli_vk_xb_sub_issue's (a batch off the grouped route) */
const float *coli_vk_xb_step_sum(const float *w, const uint8_t *use, double *device_ms);
void  coli_vk_xb_step_end(void);
/* An expert's entries for qmatmul_grp_gemv.comp (gate|up, down; rows and packed row
 * left 0), for a caller that routes on the device; 0 when the grouped GEMV cannot take
 * it. The addresses stay valid while the expert does. */
int   coli_vk_xb_expert_entries(const ColiVkExpert *e, uint32_t gu[16], uint32_t dn[16]);
/* Sub-batches, for a prefill step too big for one batch (vk_tier.c's streaming): batch k
 * of a step goes to half k % 2 of the scratch, so two run at once and the host fills one
 * while the device computes the other. _reserve sizes each half for a batch of up to
 * `rows` rows over at most `experts` experts (not while any batch is in flight; it may
 * move the scratch). _issue submits one into half h and returns (0: nothing submitted);
 * _join waits for it and copies the D outputs of its row j to yout[j], same order as
 * xrows; *device_ms its device time when timestamps exist. The single batch above and
 * the sub-batches do not overlap: neither starts while the other is in flight. */
/* Largest sub-batch up to rows that can grow all five buffers within extra_budget
 * and half each heap's free budget; does not allocate. Zero means no row fits. */
int  coli_vk_xb_sub_fit(int rows, int experts, size_t extra_budget);
int  coli_vk_xb_sub_reserve(int rows, int experts);
int  coli_vk_xb_sub_issue(int h, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                          const float *wrows);
int  coli_vk_xb_sub_join(int h, float *const *yout, double *device_ms);
int  coli_vk_xb_sub_busy(int h);
/* A tier tensor (coli_vk_tier_tensor) to fill again in place: rows and scales as
 * coli_vk_tier_tensor hands them out, then coli_vk_tensor_commit. 0 = no memory. Not
 * while a batch that reads it is in flight. */
int  coli_vk_tensor_refill(ColiVkTensor *t, uint8_t **rows, size_t *stride, float **scales);

/* ---- the batch on the second device (COLI_VK_DEV2) --------------------------------
 * The routed-expert tier can hold experts on a second device too (vk_tier.c). dev 0
 * is the device the calls above use, dev 1 COLI_VK_DEV2's: there the batch has its own
 * context (pipelines, scratch, the device's one queue), the GEMV and fp32 GEMM routes
 * and no cooperative-matrix one, and no sub-batches. A step can have a batch in flight
 * on each device at once. An expert belongs to the device of its tensors
 * (coli_vk_xb_expert); a batch takes experts of its device only.
 *   coli_vk_dev2_open_env  COLI_VK_DEV2=auto|<index>: the second device up (1), with the
 *                          shaders device 0 was opened with; 0 when unset or unusable
 *   coli_vk_dev_info       what the tier sizes itself by, for either device
 *   coli_vk_tier_tensor_dev  coli_vk_tier_tensor in that device's tier pool
 *   coli_vk_xb_*_dev       coli_vk_xb_init/_ready/_issue_w/_join/_stats on that device;
 *                          _busy_dev: a batch in flight there */
typedef struct {
    const char *name;
    int integrated, shares_ram;       /* shares_ram: an integrated GPU or a CPU device */
    size_t local_bytes;               /* the largest device-local heap (the cap under COLI_VK_DEVICE_CAP_MB) */
    int has_budget;                   /* free_bytes is the cap or VK_EXT_memory_budget's */
    size_t free_bytes;                /* free now; else the heap less what this process holds */
    size_t buf_align;                 /* where a weight range starts */
} ColiVkDevInfo;
int  coli_vk_dev2_open_env(void);
int  coli_vk_dev_info(int dev, ColiVkDevInfo *out);
int  coli_vk_tier_tensor_dev(int dev, ColiVkTensor **t, int fmt, int I, int O, int gs,
                             uint8_t **rows, size_t *stride, float **scales);
int  coli_vk_xb_init_dev(int dev, int D, int I, int act, float limit, float a, float b);
int  coli_vk_xb_ready_dev(int dev);
int  coli_vk_xb_issue_dev(int dev, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                          const float *wrows);
int  coli_vk_xb_join_dev(int dev, const float **yrows, double *device_ms);
int  coli_vk_xb_busy_dev(int dev);
void coli_vk_xb_stats_dev(int dev, ColiVkXbStats *st);

/* 1 if the selected device is an integrated GPU (shares physical memory with
 * the host), 0 otherwise or when no device is selected. */
int coli_vk_device_integrated(void);
/* 1 for an integrated GPU or a CPU device (Lavapipe): device memory is host RAM. */
int coli_vk_device_shares_ram(void);
/* The largest DEVICE_LOCAL heap in bytes (for a budget without VK_EXT_memory_budget). */
size_t coli_vk_device_local_bytes(void);
const char *coli_vk_device_name(void);

/* For engines: COLI_VULKAN=1 opens the device with the shaders found by
 * coli_vk_shader_path() and prints one line naming the engine and where its dense
 * matrices go (coli_vk_dense_decide below); 0 (and one line) when Vulkan is not asked
 * for or no device is usable, so the engine stays on the CPU. tier_on: the engine is
 * about to start the routed-expert tier (vk_tier.c); coli_vk_init_env(engine) is
 * coli_vk_init_env_tier(engine, 0). */
int coli_vk_init_env(const char *engine);
int coli_vk_init_env_tier(const char *engine, int tier_on);
/* Where an engine's dense (resident, non-expert) matrices run, the one rule every
 * engine follows. COLI_VK_DENSE set and non-empty: 0 keeps them on the CPU, any other
 * number puts them on the device. Unset: the engine's default `def`, except that with
 * the routed-expert tier on (tier_on) a device that shares the CPU's RAM (an
 * integrated GPU, or a CPU device such as Lavapipe) keeps them on the CPU: there the
 * dense matmuls, one synchronous call each, cost more than the tier gains (measured
 * on a Radeon 780M, docs/vulkan.md). The decision is kept for coli_vk_dense(); with
 * an engine name it is printed as a [VK] line when it changes (NULL: silent). */
int coli_vk_dense_decide(const char *engine, int tier_on, int def);
int coli_vk_dense(void);   /* the last decision; 1 before any */
/* Whether the dense weights the device holds keep a host copy (docs/vulkan.md, "Dense
 * weights on the device only"). With the dense part on the device (dense_on_device: the
 * chain on, or the per-matrix path on), an engine in this mode uploads each resident
 * dense matrix once, drops its host copy, and reads it back from disk only when the CPU
 * needs it (a lost device, a step the device declines). COLI_VK_DENSE_HOST set and
 * non-empty: 0 device only, any other number keeps the host copies. Unset: device only
 * on an integrated GPU (its memory is the same RAM) and on a discrete GPU whose free
 * memory, less 1 GiB, holds dense_bytes; host copies kept on a CPU device (Lavapipe)
 * and when the dense part runs on the CPU. Printed as a [VK] line with an engine name;
 * with the dense part on the device, one more at exit: in the mode, what was dropped and
 * read back, and the resident set either way. Returns 1 for device only. */
int  coli_vk_dense_host_decide(const char *engine, int dense_on_device, size_t dense_bytes);
int  coli_vk_dense_device_only(void);          /* the last decision; 0 before any */
void coli_vk_dense_host_dropped(size_t bytes); /* a host copy given back after its upload */
void coli_vk_dense_host_reloaded(size_t bytes);/* a host copy read back from disk for the CPU */
/* One [VK] line once the engine has placed its dense matrices: how many are on the device
 * only, the RAM given back, and what stays on the host (kept: NULL or a short note). */
void coli_vk_dense_host_placed(const char *engine, const char *kept);
unsigned long long coli_vk_dense_host_dropped_bytes(void);
/* A partial chain (vk_chain.h, vkc_fit): only the first `on_device` of `layers` layers drop
 * their host copies; the placed and exit lines then say so (nothing changes when
 * on_device == layers, or before any call). Call it before coli_vk_dense_host_placed. The
 * bytes the engine passes coli_vk_dense_host_decide are those layers' only. */
void coli_vk_dense_host_layers(int on_device, int layers);
/* COLI_VK_SHADERS (the .spv or its directory), else shaders/ next to the binary, else
 * shaders/ in the working directory. buf holds the result when it is not a literal. */
const char *coli_vk_shader_path(char *buf, size_t n);
/* How many coli_vk_matmul calls ran on the device: a check that a path is really used. */
unsigned long long coli_vk_matmul_calls(void);

/* ---- the dense chain (vk_chain.h) ----------------------------------------------
 * Whether an engine runs its layers' dense chain on the device (COLI_VK_CHAIN): set,
 * 0 off, 2 prompts only, else on; unset, on for a discrete GPU, `igpu` (the engine's
 * measured choice) on an integrated GPU with the expert tier on, off otherwise. With an
 * engine name the decision is printed as a [VK] line. */
#define COLI_VK_CHAIN_OFF     0
#define COLI_VK_CHAIN_ON      1
#define COLI_VK_CHAIN_PREFILL 2   /* forwards of more than two rows only */
#define COLI_VK_CHAIN_UNMEASURED 3 /* as `igpu`: not measured on an integrated GPU, so off there */
int coli_vk_chain_decide(const char *engine, int tier_on, int igpu);
/* The device as the chain sees it: Vulkan handles as void * (VkInstance,
 * VkPhysicalDevice, VkDevice, VkQueue), the memory types the backend picked, the
 * shader directory's qmatmul.spv and the fp32 GEMM's tiles. 0 before coli_vk_init. */
typedef struct {
    void *instance, *phys, *device, *queue;
    uint32_t qfam, memtype_host, memtype_cached, memtype_dev;
    size_t ssbo_align, ssbo_range;
    const char *spv_path;
    int gemm_tiles, gemm_tile[4][6];      /* bm, bn, bk, tm, tn, pf */
    int gemm_min_s, gemm_min_so;
    int has_prio, integrated, shares_ram;
    int coop_sg;                          /* subgroup size the cooperative-matrix shaders run at; 0 = none */
} ColiVkCore;
int  coli_vk_core(ColiVkCore *out);
/* A resident tensor's buffers (VkBuffer as void *) and layout; 0 for a COLI_VK_DEV2 one. */
typedef struct { void *wbuf, *sbuf; int fmt, I, O, rowWords, gs; } ColiVkTensorInfo;
int  coli_vk_tensor_info(const ColiVkTensor *t, ColiVkTensorInfo *out);
/* The chain's fence wait failed: the device is lost, the backend stops. */
void coli_vk_mark_lost(void);
/* vkQueueSubmit(queue, 1, submit_info, fence) as the backend submits (a VkResult): the
 * staged uploader may share the main queue from its own thread. */
int  coli_vk_queue_submit(void *queue, const void *submit_info, void *fence);
/* The same for either device: d = 0 the primary (as the calls above), d = 1 COLI_VK_DEV2's,
 * whose context the chain opens for the layers it places there (vk_chain.c). The core of
 * device 1 carries the primary's shader path and GEMM tiles. coli_vk_tensor_info_dev
 * reports a tensor on either device and which (coli_vk_tensor_info: device 0's only). */
int    coli_vk_core_dev(int d, ColiVkCore *out);
int    coli_vk_available_dev(int d);
void   coli_vk_mark_lost_dev(int d);
int    coli_vk_queue_submit_dev(int d, void *queue, const void *submit_info, void *fence);
int    coli_vk_tensor_info_dev(const ColiVkTensor *t, ColiVkTensorInfo *out, int *dev);
int    coli_vk_mem_budget_dev(int d, double *used_gb, double *budget_gb);
size_t coli_vk_device_used_dev(int d);
size_t coli_vk_device_local_bytes_dev(int d);
size_t coli_vk_free_bytes_dev(int d);
size_t coli_vk_buffer_alignment_dev(int d);
void   coli_vk_mem_info_dev(int d, size_t *used_bytes, size_t *tensor_count);   /* dense weights on device d */

#ifdef __cplusplus
}
#endif

#endif
