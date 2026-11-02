/* cu_tier_ops.h -- the routed-expert tier (vk_tier.c) on a CUDA device.
 *
 * vk_tier.c holds the tier's logic: which experts are resident, the warm start, the
 * promotions and evictions, the host pool, the hybrid steps and the report. It reaches
 * the device through backend_vulkan.c's calls only (the tier's tensors and pools, the
 * expert batch). cu_tier.c compiles the same file a second time with those calls going
 * to a CUDA backend (cu_xb.cu) through the table below, and with the tier's own names
 * prefixed cut_ in place of vkt_ (cu_tier.h), so a build with both backends holds both
 * tiers. The table is one symbol, coli_cuda_tier_ops, so the Windows DLL loader
 * (backend_loader.c) resolves it alone.
 *
 * The calls keep backend_vulkan.h's contracts and its format numbers (fmt 1 int8 a row,
 * 13 int8 per group, 2 and 4 int4 pairs, 5 int3-g64, 7 MXFP4 with f32 group scales,
 * 12 FP8 per group, 10 f32, 11 bf16, 100 + t a ggml block type), so the tier's
 * conversions and sizes are the same. What the CUDA backend does not do it declines the
 * way the Vulkan one declines on a device without it, and the tier then does without:
 * no dense chain entries (xb_expert_entry 0), no second device (dev2_open_env 0), no
 * streaming (gemm_rows 0 in xb_stats), no grouped step (xb_step_begin NULL). */
#ifndef COLI_CU_TIER_OPS_H
#define COLI_CU_TIER_OPS_H
#include <stddef.h>
#include <stdint.h>
#include "backend_vulkan.h"   /* the types: ColiVkTensor, ColiVkExpert, the stats and infos */
#include "backend_cuda.h"     /* COLI_CUDA_DLLEXPORT */

#ifdef __cplusplus
extern "C" {
#endif

#define CU_TIER_OPS_VERSION 1

typedef struct {
    int version;                 /* CU_TIER_OPS_VERSION */
    /* the device */
    int    (*available)(void);
    const char *(*device_name)(void);
    int    (*device_shares_ram)(void);
    size_t (*device_local_bytes)(void);
    int    (*mem_budget)(double *used_gb, double *budget_gb);
    int    (*dev_info)(int dev, ColiVkDevInfo *out);
    int    (*dev2_open_env)(void);
    /* the pools */
    size_t (*block_bytes)(size_t def);
    size_t (*block_bytes_dev)(int dev, size_t def);
    size_t (*buffer_alignment)(void);
    void   (*pool_stats)(int pool, ColiVkPoolStats *st);
    void   (*tier_pool_limit)(size_t bytes);
    void   (*tier_pool_limit_dev)(int dev, size_t bytes);
    void   (*tier_extra_pool_limit)(size_t bytes);
    int    (*tier_host_pool)(void);
    /* the tier's tensors */
    int    (*block_elems)(int fmt);
    size_t (*tensor_row_bytes)(int fmt, int I);
    size_t (*tensor_scale_count)(int fmt, int I, int O, int gs);
    size_t (*tensor_payload)(int fmt, int I, int O, int gs);
    size_t (*tensor_alloc_bytes)(int fmt, int I, int O, int gs, size_t *scale_bytes);
    int    (*tier_tensor_pad)(int dev, ColiVkTensor **t, int fmt, int I, int O, int gs, size_t pad_w, size_t pad_s,
                              uint8_t **rows, size_t *stride, float **scales);
    int    (*tier_tensor_extra)(ColiVkTensor **t, int fmt, int I, int O, int gs, uint8_t **rows, size_t *stride, float **scales);
    int    (*tier_tensor_like)(int dev, ColiVkTensor **t, const ColiVkTensor *src, size_t pad_w, size_t pad_s);
    int    (*tensor_commit)(ColiVkTensor *const *t, int n);
    int    (*tensor_refill)(ColiVkTensor *t, uint8_t **rows, size_t *stride, float **scales);
    int    (*tensor_refill_as)(ColiVkTensor *t, int fmt, int I, int O, int gs, uint8_t **rows, size_t *stride, float **scales);
    int    (*tensor_copy)(ColiVkTensor *const *dst, ColiVkTensor *const *src, int n);
    void   (*tensor_free)(ColiVkTensor *t);
    size_t (*tensor_bytes)(const ColiVkTensor *t);
    int    (*tensor_info)(const ColiVkTensor *t, ColiVkTensorInfo *out);
    const void *(*tensor_host_rows)(const ColiVkTensor *t, const void **scales);
    /* the expert batch */
    int    (*xb_init)(int D, int I, int act, float limit, float a, float b);
    int    (*xb_init_dev)(int dev, int D, int I, int act, float limit, float a, float b);
    int    (*xb_ready)(void);
    int    (*xb_ready_dev)(int dev);
    int    (*xb_queue_shared)(void);
    ColiVkExpert *(*xb_expert)(ColiVkTensor *gate, ColiVkTensor *up, ColiVkTensor *down);
    void   (*xb_expert_free)(ColiVkExpert *e);
    void   (*xb_expert_tensors)(const ColiVkExpert *e, ColiVkTensor **g, ColiVkTensor **u, ColiVkTensor **d);
    int    (*xb_expert_entry)(const ColiVkExpert *e, uint32_t *entry);
    int    (*xb_issue_w)(ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows, const float *wrows);
    int    (*xb_issue_dev)(int dev, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                           const float *wrows);
    int    (*xb_join)(const float **yrows, double *device_ms);
    int    (*xb_join_dev)(int dev, const float **yrows, double *device_ms);
    int    (*xb_busy_dev)(int dev);
    void   (*xb_stats)(ColiVkXbStats *st);
    /* streaming's sub-batches and the grouped step: declined while gemm_rows is 0 */
    float *(*xb_step_begin)(int S, int K, const float *x);
    const float *(*xb_step_sum)(const float *w, const uint8_t *use, double *device_ms);
    void   (*xb_step_end)(void);
    int    (*xb_sub_fit)(int rows, int experts, size_t extra_budget);
    int    (*xb_sub_reserve)(int rows, int experts);
    int    (*xb_sub_issue)(int h, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                           const float *wrows);
    int    (*xb_sub_issue_step)(int h, ColiVkExpert *const *ex, const int *rows, int count, const int *assign,
                                const float *wrows);
    int    (*xb_sub_join)(int h, float *const *yout, double *device_ms);
    int    (*xb_sub_busy)(int h);
    int    (*xb_sub_copy)(int h, ColiVkTensor *const *dst, ColiVkTensor *const *src, int n);
    void   (*xb_sub_copy_drop)(void);
    /* the decode group's tail sum (COLI_VK_TIER_CS_DEVSUM, backend_vulkan.c's
     * coli_vk_xb_sum_arm/join_sum): declined here (NULL) -- the CS arm is the Vulkan
     * tier's own, so a CUDA tier never arms the tail. Appended: the CUDA table's
     * positional initializer zeroes the tail. */
    int    (*xb_sum_arm)(int n);
    int    (*xb_join_sum)(const float **yrows, const float **sum, double *device_ms);
} CuTierOps;

/* The table. Its calls need the CUDA device open (coli_cuda_init first): available() says
 * whether it is; without it the others decline. NULL from a DLL that predates it. */
COLI_CUDA_DLLEXPORT const CuTierOps *coli_cuda_tier_ops(void);

#ifdef __cplusplus
}
#endif
#endif
