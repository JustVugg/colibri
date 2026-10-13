/* olmoe_cuda.h -- ABI of the OLMoE GPU-resident decode library (olmoe_cuda.cu).
 *
 * Same split as backend_cuda_dsv4.h: on Linux olmoe_cuda.o is linked into olmoe directly (OLMOE_CUDA=1); on Windows the
 * kernels are built with nvcc + MSVC into coli_cuda_olmoe.dll (symbols exported through olmoe_cuda.def), and
 * olmoe_cuda_loader.c resolves them at run time (OLMOE_CUDA_DLL=1), so the MinGW-built engine never links cudart.
 * With neither switch nothing here is compiled into the engine.
 *
 * Every entry is a plain C call over ints, floats and pointers (no structs cross the boundary), so the MSVC- and
 * MinGW-built halves agree on the ABI. Functions returning int return > 0 on success and <= 0 on failure; the engine
 * treats any failure as a breaker trip and continues on the CPU. */
#ifndef COLIBRI_OLMOE_CUDA_H
#define COLIBRI_OLMOE_CUDA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define OLMOE_CUDA_API

/* 1 if the library is usable here: a CUDA device is visible (and, on Windows, the DLL loaded). */
OLMOE_CUDA_API int olmoe_cuda_available(void);

/* Allocate the per-model buffers. Refuses (returns -2) shapes the kernels do not implement: head_dim != 128,
 * topk > 16, more than 1024 experts, hidden % 256, inter % 16. max_t = positions of device KV per layer. */
OLMOE_CUDA_API int olmoe_cuda_init(int device, int layers, int hidden, int heads, int head_dim, int inter, int nexp,
                                   int topk, int vocab, int max_t, float eps, float theta, int norm_topk);
/* Dense weights of layer l: q/k/v/o as IEEE half [D][D]; norms and the router as float. */
OLMOE_CUDA_API int olmoe_cuda_set_layer(int l, const void *q, const void *k, const void *v, const void *o,
                                        const float *in_ln, const float *post_ln, const float *qn, const float *kn,
                                        const float *gate);
OLMOE_CUDA_API int olmoe_cuda_set_head(const void *lm_head_half, const float *final_norm);
OLMOE_CUDA_API void olmoe_cuda_mem_info(size_t *free_bytes, size_t *total_bytes);

/* Expert slots in VRAM; returns how many were allocated (<= n). One expert = 3*inter*hidden int8 + (2*inter+hidden)
 * float row scales. olmoe_cuda_stage_ptr() is a pinned buffer holding the blob followed by the scales, for the host
 * to fill before olmoe_cuda_upload_expert(); the upload also publishes (layer, eid) -> slot in the device table. */
OLMOE_CUDA_API int olmoe_cuda_alloc_slots(int n);
OLMOE_CUDA_API void *olmoe_cuda_stage_ptr(void);
OLMOE_CUDA_API int olmoe_cuda_upload_expert(int layer, int eid, int slot, const int8_t *w, const float *scales);
OLMOE_CUDA_API int olmoe_cuda_unmap_expert(int layer, int eid);

/* Device KV rows [from,to) of one layer from a host cache laid out [heads][host_max_t][head_dim]. */
OLMOE_CUDA_API int olmoe_cuda_kv_upload(int layer, const float *K, const float *V, int host_max_t, int from, int to);

/* One decode token: begin_token(embedding row, position), then per layer attn, route, [route_fetch], experts; finish
 * returns the logits and this token's K/V rows ([layers][2][hidden], K then V) for the host mirror. With every expert
 * resident, layer_experts(NULL) uses the slots the router resolved on the device and the token needs no host sync;
 * otherwise route_fetch returns the routed ids so the host can make them resident and pass explicit slots. */
OLMOE_CUDA_API int olmoe_cuda_begin_token(const float *x_embed, int pos);
OLMOE_CUDA_API int olmoe_cuda_layer_attn(int l);
OLMOE_CUDA_API int olmoe_cuda_layer_route(int l);
OLMOE_CUDA_API int olmoe_cuda_route_fetch(int *ids, float *w, int *miss);
OLMOE_CUDA_API int olmoe_cuda_layer_experts(const int *slots);
OLMOE_CUDA_API int olmoe_cuda_finish(float *logits_out, float *kv_stage_out);
OLMOE_CUDA_API void olmoe_cuda_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif
