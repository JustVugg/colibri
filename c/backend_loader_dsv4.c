/* backend_loader_dsv4.c — Windows runtime loader for the DeepSeek V4 CUDA backend DLL.
 *
 * Why this exists: deepseek-v4.exe is built with MinGW-w64 (gcc), but the
 * DeepSeek V4 CUDA kernels in backend_cuda_dsv4.cu must be compiled with
 * MSVC + nvcc. The MSVC/MinGW ABI split means we cannot link a CUDA .o into a
 * gcc binary, so the kernels are built into a standalone coli_cuda_dsv4.dll
 * (nvcc + MSVC, symbols exported via dsv4.def) and resolved here at runtime
 * through LoadLibrary/GetProcAddress. The host never links cudart directly.
 *
 * This mirrors backend_loader.c, which does the same for the GLM engine's
 * coli_cuda.dll. The two loaders are separate files on purpose: the dsv4 ABI
 * (dsv4_cuda_*) is entirely different from the GLM ABI (coli_cuda_*), and the
 * dsv4 host is a different executable that must be able to load the V4 backend
 * even when the GLM backend is absent.
 *
 * On non-Windows builds this file is not compiled (the Makefile.deepseek-v4
 * links backend_cuda_dsv4.o directly). If the DLL is absent, every wrapper
 * safely returns the "not initialized" sentinel and the engine falls back to
 * CPU — the same contract the raw backend_cuda_dsv4.h functions have when no
 * CUDA device is present.
 *
 * ABI note: Dsv4CudaTensor/Dsv4CudaActivation/Dsv4CudaKvCache/Dsv4CudaExpertSet/
 * Dsv4CudaGraph are opaque to the host (it stores the pointer, never dereferences
 * it), so the MSVC-allocated structs are safe to pass across the boundary. All
 * scalar types (int, float, pointers, long long) agree between MSVC and
 * MinGW-w64 on x86-64.
 */
#ifdef _WIN32

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <windows.h>

#include "backend_cuda_dsv4.h"

/* Two builds of the same source ship side by side: coli_cuda_dsv4_dg.dll
 * (DeepGEMM, sm_120a tensor-core paths) and coli_cuda_dsv4.dll (portable
 * generic kernels, sm_80+). The loader tries the DeepGEMM one first and asks
 * it whether the device fits (dsv4_cuda_backend_arch_ok); on refusal it
 * unloads and takes the generic one, so one install adapts to whatever GPU
 * is present. COLI_DSV4_DLL=<file name> forces a specific DLL. */
#define DSV4_BACKEND_DLL "coli_cuda_dsv4.dll"
#define DSV4_BACKEND_DLL_DG "coli_cuda_dsv4_dg.dll"
#define DSV4_VENDOR_TAG  "[DSV4 CUDA]"

/* Function-pointer typedefs matching each exported symbol in dsv4.def. */
typedef int             (*fn_init)(const int *devices, int count);
typedef void            (*fn_shutdown)(void);
typedef Dsv4CudaActivation *(*fn_activation_create)(int device, long long elements);
typedef void            (*fn_activation_free)(Dsv4CudaActivation *a);
typedef int             (*fn_activation_upload)(Dsv4CudaActivation *a, const float *x, long long elements);
typedef int             (*fn_activation_download)(float *x, const Dsv4CudaActivation *a, long long elements);
typedef int             (*fn_activation_copy)(Dsv4CudaActivation *dst, const Dsv4CudaActivation *src, long long elements);
typedef int             (*fn_activation_copy_range)(Dsv4CudaActivation *dst, long long dst_offset,
                                                    const Dsv4CudaActivation *src, long long src_offset,
                                                    long long elements);
typedef int             (*fn_activation_sync)(const Dsv4CudaActivation *a);
typedef int             (*fn_activation_device)(const Dsv4CudaActivation *a);
typedef int             (*fn_decode_state_set)(int device, int token, int position);
typedef void            (*fn_profiler_start)(void);
typedef void            (*fn_profiler_stop)(void);
typedef int             (*fn_graph_begin)(int device);
typedef Dsv4CudaGraph   *(*fn_graph_end)(int device);
typedef int             (*fn_graph_end_pair)(int primary, int peer, Dsv4CudaGraph **primary_graph,
                                            Dsv4CudaGraph **peer_graph);
typedef int             (*fn_graph_launch)(Dsv4CudaGraph *graph);
typedef void            (*fn_graph_free)(Dsv4CudaGraph *graph);
typedef int             (*fn_upload_fp8)(Dsv4CudaTensor **t, const uint8_t *w, const uint8_t *scale,
                                        int O, int I, int device);
typedef int             (*fn_upload_fp8_bf16)(Dsv4CudaTensor **t, const uint8_t *w, const uint8_t *scale,
                                              int O, int I, int device);
typedef int             (*fn_upload_fp4)(Dsv4CudaTensor **t, const uint8_t *w, const uint8_t *scale,
                                        int O, int I, int device);
typedef int             (*fn_upload_bf16)(Dsv4CudaTensor **t, const uint16_t *w, int O, int I, int device);
typedef int             (*fn_upload_f32)(Dsv4CudaTensor **t, const float *w, int O, int I, int device);
typedef int             (*fn_mhc_pre)(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn,
                                      Dsv4CudaTensor *scale, Dsv4CudaTensor *base, int M, int H,
                                      float rms_eps, float pre_eps, float sink_eps, float post_mult,
                                      int sink_iters, Dsv4CudaActivation *state, Dsv4CudaActivation *input);
typedef int             (*fn_mhc_post)(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                                       const Dsv4CudaActivation *state, int M, int H, Dsv4CudaActivation *out);
typedef int             (*fn_attention_first)(const Dsv4CudaActivation *input, Dsv4CudaTensor *attn_norm,
                                              Dsv4CudaTensor *q_a, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
                                              Dsv4CudaTensor *wkv, Dsv4CudaTensor *kv_norm, Dsv4CudaTensor *sink,
                                              Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b, int heads, int head_dim,
                                              int qk_rope, int groups, float eps, Dsv4CudaActivation *output);
typedef Dsv4CudaKvCache *(*fn_kv_create)(int device, int window, int head_dim, int max_tokens,
                                         int rope_pairs, const float *rope_cos, const float *rope_sin,
                                         const float *compress_cos, const float *compress_sin);
typedef int             (*fn_mhc_pre_norm)(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn,
                                           Dsv4CudaTensor *scale, Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                           int M, int H, float rms_eps, float pre_eps, float sink_eps,
                                           float post_mult, int sink_iters, float norm_eps,
                                           Dsv4CudaActivation *state, Dsv4CudaActivation *input);
typedef int             (*fn_mhc_pre_norm_batch)(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn,
                                                 Dsv4CudaTensor *scale, Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                                 int tokens, int H, Dsv4CudaActivation *state, Dsv4CudaActivation *input);
typedef int             (*fn_mhc_pre_batch)(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn,
                                            Dsv4CudaTensor *scale, Dsv4CudaTensor *base, int tokens, int H,
                                            Dsv4CudaActivation *state, Dsv4CudaActivation *input);
typedef int             (*fn_mhc_post_pre)(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                                           Dsv4CudaActivation *state, int M, int H, Dsv4CudaActivation *out,
                                           Dsv4CudaTensor *fn, Dsv4CudaTensor *scale, Dsv4CudaTensor *base,
                                           float rms_eps, float pre_eps, float sink_eps, float post_mult,
                                           int sink_iters, Dsv4CudaActivation *input);
typedef int             (*fn_mhc_post_pre_norm)(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                                                Dsv4CudaActivation *state, int M, int H, Dsv4CudaActivation *out,
                                                Dsv4CudaTensor *fn, Dsv4CudaTensor *scale, Dsv4CudaTensor *base,
                                                Dsv4CudaTensor *norm, float rms_eps, float pre_eps, float sink_eps,
                                                float post_mult, int sink_iters, float norm_eps,
                                                Dsv4CudaActivation *input);
typedef int             (*fn_mhc_post_pre_norm_batch)(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                                                      Dsv4CudaActivation *state, int tokens, int H, Dsv4CudaActivation *out,
                                                      Dsv4CudaTensor *fn, Dsv4CudaTensor *scale, Dsv4CudaTensor *base,
                                                      Dsv4CudaTensor *norm, Dsv4CudaActivation *input);
typedef int             (*fn_mhc_post_batch)(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                                             const Dsv4CudaActivation *state, int tokens, int H,
                                             Dsv4CudaActivation *out);
typedef void            (*fn_kv_free)(Dsv4CudaKvCache *cache);
typedef int             (*fn_attention_window)(const Dsv4CudaActivation *input, Dsv4CudaTensor *attn_norm,
                                               Dsv4CudaTensor *q_a, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
                                               Dsv4CudaTensor *wkv, Dsv4CudaTensor *kv_norm, Dsv4CudaTensor *sink,
                                               Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b,
                                               Dsv4CudaTensor *compress_wkv, Dsv4CudaTensor *compress_wgate,
                                               Dsv4CudaTensor *compress_ape, Dsv4CudaTensor *compress_norm,
                                               int compress_ratio, int heads, int head_dim, int qk_rope,
                                               int groups, int pos, float eps, Dsv4CudaKvCache *cache,
                                               Dsv4CudaActivation *output);
typedef int             (*fn_attention_sparse_batch)(const Dsv4CudaActivation *input, Dsv4CudaTensor *attn_norm,
                                                     Dsv4CudaTensor *qkv, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
                                                     Dsv4CudaTensor *kv_norm, Dsv4CudaTensor *sink, int heads,
                                                     int head_dim, int start_pos, int tokens, float eps,
                                                     Dsv4CudaKvCache *cache, Dsv4CudaActivation *context);
typedef int             (*fn_attention_output_batch)(const Dsv4CudaActivation *context, Dsv4CudaTensor *wo_a,
                                                     Dsv4CudaTensor *wo_b, int groups, int tokens,
                                                     Dsv4CudaActivation *output);
typedef int             (*fn_attention_window_tp2)(const Dsv4CudaActivation *input, Dsv4CudaActivation *peer_input,
                                                   const Dsv4CudaAttentionWeights *primary,
                                                   const Dsv4CudaAttentionWeights *peer, int compress_ratio,
                                                   int heads, int head_dim, int qk_rope, int groups, int pos,
                                                   float eps, Dsv4CudaKvCache *cache, Dsv4CudaKvCache *peer_cache,
                                                   Dsv4CudaActivation *output, Dsv4CudaActivation *peer_output);
typedef int             (*fn_route)(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                                    const int *fixed_ids, float routed_scale, int ids[6], float weights[6]);
typedef int             (*fn_matvec_grouped)(Dsv4CudaTensor *t, float *y, const float *x, int groups);
typedef int             (*fn_matvec)(Dsv4CudaTensor *t, float *y, const float *x);
typedef int             (*fn_matmul_batch)(Dsv4CudaTensor *t, const Dsv4CudaActivation *input, int tokens,
                                           Dsv4CudaActivation *output);
typedef int             (*fn_matmul_bf16_batch)(Dsv4CudaTensor *t, const float *x, int tokens, float *y);
typedef int             (*fn_sparse_attn_batch_cached)(int device, int layer, const float *q,
                                                        const float *chunk, int chunk_start,
                                                        const float *sinks, const int *meta,
                                                        int abs_base, int comp_limit, int heads,
                                                        int dim, int tokens, float scale, float *out);
typedef int             (*fn_sparse_attn_batch_cached_idx)(int device, int layer, const float *q,
                                                            const float *chunk, int chunk_start,
                                                            const float *sinks, const int *meta,
                                                            const int *sel, int selstride,
                                                            int abs_base, int comp_limit, int heads,
                                                            int dim, int tokens, float scale, float *out);
typedef int             (*fn_indexer_score_batch)(int device, const float *queries, const float *keys,
                                                  const float *head_w, const int *counts, int tokens,
                                                  int heads, int dim, int count, float *scores);
typedef int             (*fn_fp8_ref_matmul)(int device, const uint8_t *w, const float *bscale,
                                             int rows, int cols, int packed_rows8, const float *x,
                                             int tokens, float *y);
typedef int             (*fn_tensor_refill_fp4)(Dsv4CudaTensor *t, const uint8_t *w, const uint8_t *scale,
                                                int O, int I, int sync);
typedef int             (*fn_backend_arch_ok)(int device);
typedef const char     *(*fn_backend_name)(void);
typedef long long       (*fn_mem_free_mb)(int device);
typedef int             (*fn_stream_drain)(int device);
typedef int             (*fn_kv_ring_append)(int device, int layer, const float *rows,
                                             int start_pos, int count, int window, int dim);
typedef int             (*fn_kv_comp_append)(int device, int layer, const float *rows,
                                             int start_idx, int count, int dim);
typedef int             (*fn_sparse_attn_batch)(int device, const float *q, const float *vals,
                                                const float *sinks, const int *meta, int value_rows,
                                                int comp_base, int heads, int dim, int tokens,
                                                float scale, float *out);
typedef int             (*fn_head_argmax)(Dsv4CudaTensor *t, const float *x, int *id, float *value);
typedef int             (*fn_final_argmax)(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn,
                                           Dsv4CudaTensor *scale, Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                           Dsv4CudaTensor *head, int M, int H, float eps, float pre_eps,
                                           int *id, float *value);
typedef int             (*fn_expert_group)(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                                           Dsv4CudaTensor *const *down, const float *weights, int count,
                                           float limit, float *y, const float *x);
typedef int             (*fn_expert_fp8)(Dsv4CudaTensor *gate, Dsv4CudaTensor *up, Dsv4CudaTensor *down,
                                         float limit, float *y, const float *x);
typedef int             (*fn_moe)(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                                  Dsv4CudaTensor *const *down, const float *weights, int count,
                                  Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up,
                                  Dsv4CudaTensor *shared_down, float limit, float *y, const float *x);
typedef int             (*fn_rmsnorm)(Dsv4CudaActivation *x, Dsv4CudaTensor *weight, float eps, int elements);
typedef int             (*fn_moe_activation)(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                                             Dsv4CudaTensor *const *down, const float *weights, int count,
                                             Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up,
                                             Dsv4CudaTensor *shared_down, float limit,
                                             const Dsv4CudaActivation *input, Dsv4CudaActivation *output);
typedef Dsv4CudaExpertSet *(*fn_expert_set_create)(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                                                   Dsv4CudaTensor *const *down, int count,
                                                   Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up,
                                                   Dsv4CudaTensor *shared_down);
typedef Dsv4CudaExpertSet *(*fn_expert_bank_create)(int count, int hidden, int intermediate, int device,
                                                    Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up,
                                                    Dsv4CudaTensor *shared_down);
typedef int             (*fn_expert_bank_upload)(Dsv4CudaExpertSet *set, int expert,
                                                 const uint8_t *gate_weight, const uint8_t *gate_scale,
                                                 const uint8_t *up_weight, const uint8_t *up_scale,
                                                 const uint8_t *down_weight, const uint8_t *down_scale,
                                                 Dsv4CudaTensor **gate, Dsv4CudaTensor **up, Dsv4CudaTensor **down);
typedef int             (*fn_expert_bank_set_shared)(Dsv4CudaExpertSet *set, Dsv4CudaTensor *sg,
                                                     Dsv4CudaTensor *su, Dsv4CudaTensor *sd);
typedef int             (*fn_expert_bank_upload_aux)(Dsv4CudaExpertSet *set, int expert,
                            const uint8_t *gw, const uint8_t *gs, const uint8_t *uw, const uint8_t *us,
                            const uint8_t *dw, const uint8_t *ds,
                            Dsv4CudaTensor **gate, Dsv4CudaTensor **up, Dsv4CudaTensor **down);
typedef int             (*fn_expert_bank_upload_tp2)(Dsv4CudaExpertSet *set, int expert, int rank,
                                                     const uint8_t *gate_weight, const uint8_t *gate_scale,
                                                     const uint8_t *up_weight, const uint8_t *up_scale,
                                                     const uint8_t *down_weight, const uint8_t *down_scale);
typedef void            (*fn_expert_set_free)(Dsv4CudaExpertSet *set);
typedef int             (*fn_expert_set_upload_hash)(Dsv4CudaExpertSet *set, const int64_t *map,
                                                    int vocab, int topk);
typedef int             (*fn_route_moe)(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                                        int token, float routed_scale, Dsv4CudaExpertSet *experts,
                                        float limit, Dsv4CudaActivation *output);
typedef int             (*fn_route_moe_batch)(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate,
                                              Dsv4CudaTensor *bias, const int *tokens, int count,
                                              float routed_scale, Dsv4CudaExpertSet *experts, float limit,
                                              Dsv4CudaActivation *output);
typedef int             (*fn_route_top6_batch)(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate,
                                               Dsv4CudaTensor *bias, int count, float routed_scale,
                                               int *ids, float *weights);
typedef int             (*fn_route_moe_ids_batch)(const Dsv4CudaActivation *input, const int *ids,
                                                  const float *weights, int count,
                                                  Dsv4CudaExpertSet *experts, float limit,
                                                  Dsv4CudaActivation *output);
typedef int             (*fn_route_moe_ep2)(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate,
                                            Dsv4CudaTensor *bias, const Dsv4CudaActivation *peer_input,
                                            Dsv4CudaTensor *peer_gate, Dsv4CudaTensor *peer_bias, int token,
                                            float routed_scale, Dsv4CudaExpertSet *local, Dsv4CudaExpertSet *peer,
                                            float limit, Dsv4CudaActivation *output, Dsv4CudaActivation *peer_output);
typedef int             (*fn_qkv)(Dsv4CudaTensor *q_a, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
                                  Dsv4CudaTensor *kv, float eps, float *q_out, float *kv_out, const float *x);
typedef int             (*fn_wo)(Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b, int groups,
                                 float *out, const float *context);
typedef void            (*fn_tensor_free)(Dsv4CudaTensor *t);
typedef long long       (*fn_tensor_bytes)(const Dsv4CudaTensor *t);
typedef int             (*fn_tensor_device)(const Dsv4CudaTensor *t);

/* Resolved pointers, plus a flag so we attempt the load at most once. */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_init            init;
    fn_shutdown        shutdown;
    fn_activation_create activation_create;
    fn_activation_free activation_free;
    fn_activation_upload activation_upload;
    fn_activation_download activation_download;
    fn_activation_copy activation_copy;
    fn_activation_copy_range activation_copy_range;
    fn_activation_sync activation_sync;
    fn_activation_device activation_device;
    fn_decode_state_set decode_state_set;
    fn_profiler_start  profiler_start;
    fn_profiler_stop   profiler_stop;
    fn_graph_begin     graph_begin;
    fn_graph_end       graph_end;
    fn_graph_end_pair  graph_end_pair;
    fn_graph_launch    graph_launch;
    fn_graph_free      graph_free;
    fn_upload_fp8      upload_fp8;
    fn_upload_fp8_bf16 upload_fp8_bf16;
    fn_upload_fp4      upload_fp4;
    fn_upload_bf16     upload_bf16;
    fn_upload_f32      upload_f32;
    fn_mhc_pre         mhc_pre;
    fn_mhc_post        mhc_post;
    fn_attention_first attention_first;
    fn_kv_create       kv_create;
    fn_mhc_pre_norm    mhc_pre_norm;
    fn_mhc_pre_norm_batch mhc_pre_norm_batch;
    fn_mhc_pre_batch   mhc_pre_batch;
    fn_mhc_post_pre    mhc_post_pre;
    fn_mhc_post_pre_norm mhc_post_pre_norm;
    fn_mhc_post_pre_norm_batch mhc_post_pre_norm_batch;
    fn_mhc_post_batch  mhc_post_batch;
    fn_kv_free         kv_free;
    fn_attention_window attention_window;
    fn_attention_sparse_batch attention_sparse_batch;
    fn_attention_output_batch attention_output_batch;
    fn_attention_window_tp2 attention_window_tp2;
    fn_route            route;
    fn_matvec_grouped  matvec_grouped;
    fn_matvec          matvec;
    fn_matmul_batch    matmul_batch;
    fn_matmul_bf16_batch matmul_bf16_batch;
    fn_sparse_attn_batch sparse_attn_batch;
    fn_sparse_attn_batch_cached sparse_attn_batch_cached;
    fn_sparse_attn_batch_cached_idx sparse_attn_batch_cached_idx;
    fn_indexer_score_batch indexer_score_batch;
    fn_fp8_ref_matmul fp8_ref_matmul;
    fn_tensor_refill_fp4 tensor_refill_fp4;
    fn_backend_arch_ok backend_arch_ok;   /* optional */
    fn_backend_name backend_name;         /* optional */
    char loaded_name[64];
    fn_mem_free_mb mem_free_mb;
    fn_stream_drain stream_drain;                /* optional (older DLLs) */
    fn_kv_ring_append kv_ring_append;
    fn_kv_comp_append kv_comp_append;
    fn_head_argmax     head_argmax;
    fn_final_argmax    final_argmax;
    fn_expert_group    expert_group;
    fn_expert_fp8      expert_fp8;
    fn_moe             moe;
    fn_rmsnorm         rmsnorm;
    fn_moe_activation  moe_activation;
    fn_expert_set_create expert_set_create;
    fn_expert_bank_create expert_bank_create;
    fn_expert_bank_upload expert_bank_upload;
    fn_expert_bank_set_shared expert_bank_set_shared;
    fn_expert_bank_upload_tp2 expert_bank_upload_tp2;
    fn_expert_bank_upload_aux expert_bank_upload_aux; /* optional (older DLLs) */
    fn_expert_set_free expert_set_free;
    fn_expert_set_upload_hash expert_set_upload_hash;
    fn_route_moe       route_moe;
    fn_route_moe_batch route_moe_batch;
    fn_route_top6_batch route_top6_batch;
    fn_route_moe_ids_batch route_moe_ids_batch;
    fn_route_moe_ep2   route_moe_ep2;
    fn_qkv             qkv;
    fn_wo              wo;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;
} g_dsv4;

/* One-shot loader: mirror of coli_cuda_load(). The DLL is loaded from the
 * engine's OWN directory by absolute path — never a bare name — so the
 * dependency search anchors to the trusted install directory and an attacker
 * cannot plant a same-named DLL in the CWD (classic DLL hijacking). */
static HMODULE dsv4_load_one(const char *name){
    char dllpath[MAX_PATH];
    HMODULE dll = NULL;
    DWORD mn = GetModuleFileNameA(NULL, dllpath, (DWORD)sizeof(dllpath));
    if(mn > 0 && mn < sizeof(dllpath)){
        char *slash = strrchr(dllpath, '\\');
        if(slash && (size_t)(slash + 1 - dllpath) + strlen(name) + 1 <= sizeof(dllpath)){
            strcpy(slash + 1, name);
            dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    if(!dll)
        /* Fallback by name, restricted to the app dir and System32 — never CWD. */
        dll = LoadLibraryExA(name, NULL,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    return dll;
}

static int dsv4_cuda_resolve(const char *dllname);

static int dsv4_cuda_load(void){
    if(g_dsv4.loaded) return g_dsv4.available;
    g_dsv4.loaded = 1;
    const char *forced = getenv("COLI_DSV4_DLL");
    const char *candidates[3];
    int n = 0;
    if(forced && *forced) candidates[n++] = forced;
    else { candidates[n++] = DSV4_BACKEND_DLL_DG; candidates[n++] = DSV4_BACKEND_DLL; }
    for(int i = 0; i < n; i++){
        g_dsv4.dll = dsv4_load_one(candidates[i]);
        if(!g_dsv4.dll) continue;
        if(!dsv4_cuda_resolve(candidates[i])) continue;      /* unloaded inside */
        /* Optional compatibility probe: a build whose kernels cannot run on
         * device 0 says so, and the next candidate is tried. */
        if(g_dsv4.backend_arch_ok && !g_dsv4.backend_arch_ok(0)){
            fprintf(stderr, DSV4_VENDOR_TAG " %s does not fit this GPU; trying the next backend\n",
                    candidates[i]);
            FreeLibrary(g_dsv4.dll); g_dsv4.dll = NULL;
            memset(&g_dsv4, 0, sizeof(g_dsv4)); g_dsv4.loaded = 1;
            continue;
        }
        snprintf(g_dsv4.loaded_name, sizeof(g_dsv4.loaded_name), "%s", candidates[i]);
        fprintf(stderr, DSV4_VENDOR_TAG " backend=%s (%s)\n", candidates[i],
                g_dsv4.backend_name ? g_dsv4.backend_name() : "unknown");
        g_dsv4.available = 1;
        return 1;
    }
    fprintf(stderr, DSV4_VENDOR_TAG " no usable backend DLL (" DSV4_BACKEND_DLL_DG " / "
                    DSV4_BACKEND_DLL "); GPU tier disabled (CPU path remains active).\n");
    return 0;
}

static int dsv4_cuda_resolve(const char *dllname){
    /* Mandatory symbol: missing backend means the DLL is unusable, so unload. */
    #define RESOLVE(name, type) \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_dsv4.name = (type)GetProcAddress(g_dsv4.dll, "dsv4_cuda_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_dsv4.name){ \
            fprintf(stderr, DSV4_VENDOR_TAG " %s missing symbol dsv4_cuda_" #name "\n", dllname); \
            FreeLibrary(g_dsv4.dll); g_dsv4.dll = NULL; \
            return 0; }

    RESOLVE(init, fn_init);
    RESOLVE(shutdown, fn_shutdown);
    RESOLVE(activation_create, fn_activation_create);
    RESOLVE(activation_free, fn_activation_free);
    RESOLVE(activation_upload, fn_activation_upload);
    RESOLVE(activation_download, fn_activation_download);
    RESOLVE(activation_copy, fn_activation_copy);
    RESOLVE(activation_copy_range, fn_activation_copy_range);
    RESOLVE(activation_sync, fn_activation_sync);
    RESOLVE(activation_device, fn_activation_device);
    RESOLVE(decode_state_set, fn_decode_state_set);
    RESOLVE(profiler_start, fn_profiler_start);
    RESOLVE(profiler_stop, fn_profiler_stop);
    RESOLVE(graph_begin, fn_graph_begin);
    RESOLVE(graph_end, fn_graph_end);
    RESOLVE(graph_end_pair, fn_graph_end_pair);
    RESOLVE(graph_launch, fn_graph_launch);
    RESOLVE(graph_free, fn_graph_free);
    RESOLVE(upload_fp8, fn_upload_fp8);
    RESOLVE(upload_fp8_bf16, fn_upload_fp8_bf16);
    RESOLVE(upload_fp4, fn_upload_fp4);
    RESOLVE(upload_bf16, fn_upload_bf16);
    RESOLVE(upload_f32, fn_upload_f32);
    RESOLVE(mhc_pre, fn_mhc_pre);
    RESOLVE(mhc_post, fn_mhc_post);
    RESOLVE(attention_first, fn_attention_first);
    RESOLVE(kv_create, fn_kv_create);
    RESOLVE(mhc_pre_norm, fn_mhc_pre_norm);
    RESOLVE(mhc_pre_norm_batch, fn_mhc_pre_norm_batch);
    RESOLVE(mhc_pre_batch, fn_mhc_pre_batch);
    RESOLVE(mhc_post_pre, fn_mhc_post_pre);
    RESOLVE(mhc_post_pre_norm, fn_mhc_post_pre_norm);
    RESOLVE(mhc_post_pre_norm_batch, fn_mhc_post_pre_norm_batch);
    RESOLVE(mhc_post_batch, fn_mhc_post_batch);
    RESOLVE(kv_free, fn_kv_free);
    RESOLVE(attention_window, fn_attention_window);
    RESOLVE(attention_sparse_batch, fn_attention_sparse_batch);
    RESOLVE(attention_output_batch, fn_attention_output_batch);
    RESOLVE(attention_window_tp2, fn_attention_window_tp2);
    RESOLVE(route, fn_route);
    RESOLVE(matvec_grouped, fn_matvec_grouped);
    RESOLVE(matvec, fn_matvec);
    RESOLVE(matmul_batch, fn_matmul_batch);
    RESOLVE(matmul_bf16_batch, fn_matmul_bf16_batch);
    RESOLVE(sparse_attn_batch, fn_sparse_attn_batch);
    RESOLVE(sparse_attn_batch_cached, fn_sparse_attn_batch_cached);
    RESOLVE(sparse_attn_batch_cached_idx, fn_sparse_attn_batch_cached_idx);
    RESOLVE(indexer_score_batch, fn_indexer_score_batch);
    RESOLVE(fp8_ref_matmul, fn_fp8_ref_matmul);
    RESOLVE(tensor_refill_fp4, fn_tensor_refill_fp4);
    RESOLVE(mem_free_mb, fn_mem_free_mb);
    RESOLVE(kv_ring_append, fn_kv_ring_append);
    RESOLVE(kv_comp_append, fn_kv_comp_append);
    RESOLVE(head_argmax, fn_head_argmax);
    RESOLVE(final_argmax, fn_final_argmax);
    RESOLVE(expert_group, fn_expert_group);
    RESOLVE(expert_fp8, fn_expert_fp8);
    RESOLVE(moe, fn_moe);
    RESOLVE(rmsnorm, fn_rmsnorm);
    RESOLVE(moe_activation, fn_moe_activation);
    RESOLVE(expert_set_create, fn_expert_set_create);
    RESOLVE(expert_bank_create, fn_expert_bank_create);
    RESOLVE(expert_bank_upload, fn_expert_bank_upload);
    RESOLVE(expert_bank_set_shared, fn_expert_bank_set_shared);
    RESOLVE(expert_bank_upload_tp2, fn_expert_bank_upload_tp2);
    RESOLVE(expert_set_free, fn_expert_set_free);
    RESOLVE(expert_set_upload_hash, fn_expert_set_upload_hash);
    RESOLVE(route_moe, fn_route_moe);
    RESOLVE(route_moe_batch, fn_route_moe_batch);
    RESOLVE(route_top6_batch, fn_route_top6_batch);
    RESOLVE(route_moe_ids_batch, fn_route_moe_ids_batch);
    RESOLVE(route_moe_ep2, fn_route_moe_ep2);
    RESOLVE(qkv, fn_qkv);
    RESOLVE(wo, fn_wo);
    RESOLVE(tensor_free, fn_tensor_free);
    RESOLVE(tensor_bytes, fn_tensor_bytes);
    RESOLVE(tensor_device, fn_tensor_device);
    #undef RESOLVE
    /* Optional probes (older DLLs lack them). */
    _Pragma("GCC diagnostic push")
    _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"")
    g_dsv4.backend_arch_ok = (fn_backend_arch_ok)GetProcAddress(g_dsv4.dll, "dsv4_cuda_backend_arch_ok");
    /* optional: an older DLL without the export keeps the whole tier alive;
     * the engine probes this and simply stays on synchronous attaches. */
    g_dsv4.stream_drain = (fn_stream_drain)GetProcAddress(g_dsv4.dll, "dsv4_cuda_stream_drain");
    g_dsv4.expert_bank_upload_aux = (fn_expert_bank_upload_aux)GetProcAddress(g_dsv4.dll, "dsv4_cuda_expert_bank_upload_aux");
    g_dsv4.backend_name = (fn_backend_name)GetProcAddress(g_dsv4.dll, "dsv4_cuda_backend_name");
    _Pragma("GCC diagnostic pop")
    return 1;
}

/* ---- Public wrappers: match backend_cuda_dsv4.h signatures exactly.
 * Each forwards to the resolved pointer; if the DLL never loaded, return the
 * "not initialized" result the engine already handles (0 / NULL), so the
 * engine falls back to CPU. ---- */

int dsv4_cuda_init(const int *devices, int count){
    if(!dsv4_cuda_load()) return 0;
    return g_dsv4.init(devices, count);
}

void dsv4_cuda_shutdown(void){
    if(g_dsv4.available && g_dsv4.shutdown) g_dsv4.shutdown();
    if(g_dsv4.dll){ FreeLibrary(g_dsv4.dll); g_dsv4.dll = NULL; }
    g_dsv4.available = 0;
}

Dsv4CudaActivation *dsv4_cuda_activation_create(int device, long long elements){
    if(!g_dsv4.available) return NULL;
    return g_dsv4.activation_create(device, elements);
}

void dsv4_cuda_activation_free(Dsv4CudaActivation *a){
    if(g_dsv4.available && g_dsv4.activation_free) g_dsv4.activation_free(a);
}

int dsv4_cuda_activation_upload(Dsv4CudaActivation *a, const float *x, long long elements){
    if(!g_dsv4.available) return 0;
    return g_dsv4.activation_upload(a, x, elements);
}

int dsv4_cuda_activation_download(float *x, const Dsv4CudaActivation *a, long long elements){
    if(!g_dsv4.available) return 0;
    return g_dsv4.activation_download(x, a, elements);
}

int dsv4_cuda_activation_copy(Dsv4CudaActivation *dst, const Dsv4CudaActivation *src, long long elements){
    if(!g_dsv4.available) return 0;
    return g_dsv4.activation_copy(dst, src, elements);
}

int dsv4_cuda_activation_copy_range(Dsv4CudaActivation *dst, long long dst_offset,
                                    const Dsv4CudaActivation *src, long long src_offset,
                                    long long elements){
    if(!g_dsv4.available) return 0;
    return g_dsv4.activation_copy_range(dst, dst_offset, src, src_offset, elements);
}

int dsv4_cuda_activation_sync(const Dsv4CudaActivation *a){
    if(!g_dsv4.available) return 0;
    return g_dsv4.activation_sync(a);
}

int dsv4_cuda_activation_device(const Dsv4CudaActivation *a){
    if(!g_dsv4.available) return -1;
    return g_dsv4.activation_device(a);
}

int dsv4_cuda_decode_state_set(int device, int token, int position){
    if(!g_dsv4.available) return 0;
    return g_dsv4.decode_state_set(device, token, position);
}

void dsv4_cuda_profiler_start(void){
    if(g_dsv4.available && g_dsv4.profiler_start) g_dsv4.profiler_start();
}

void dsv4_cuda_profiler_stop(void){
    if(g_dsv4.available && g_dsv4.profiler_stop) g_dsv4.profiler_stop();
}

int dsv4_cuda_graph_begin(int device){
    if(!g_dsv4.available) return 0;
    return g_dsv4.graph_begin(device);
}

Dsv4CudaGraph *dsv4_cuda_graph_end(int device){
    if(!g_dsv4.available) return NULL;
    return g_dsv4.graph_end(device);
}

int dsv4_cuda_graph_end_pair(int primary, int peer, Dsv4CudaGraph **primary_graph, Dsv4CudaGraph **peer_graph){
    if(!g_dsv4.available) return 0;
    return g_dsv4.graph_end_pair(primary, peer, primary_graph, peer_graph);
}

int dsv4_cuda_graph_launch(Dsv4CudaGraph *graph){
    if(!g_dsv4.available) return 0;
    return g_dsv4.graph_launch(graph);
}

void dsv4_cuda_graph_free(Dsv4CudaGraph *graph){
    if(g_dsv4.available && g_dsv4.graph_free) g_dsv4.graph_free(graph);
}

int dsv4_cuda_upload_fp8(Dsv4CudaTensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device){
    if(!g_dsv4.available) return 0;
    return g_dsv4.upload_fp8(t, w, scale, O, I, device);
}

int dsv4_cuda_upload_fp8_bf16(Dsv4CudaTensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device){
    if(!g_dsv4.available) return 0;
    return g_dsv4.upload_fp8_bf16(t, w, scale, O, I, device);
}

int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t, const uint8_t *w, const uint8_t *scale, int O, int I, int device){
    if(!g_dsv4.available) return 0;
    return g_dsv4.upload_fp4(t, w, scale, O, I, device);
}

int dsv4_cuda_upload_bf16(Dsv4CudaTensor **t, const uint16_t *w, int O, int I, int device){
    if(!g_dsv4.available) return 0;
    return g_dsv4.upload_bf16(t, w, O, I, device);
}

int dsv4_cuda_upload_f32(Dsv4CudaTensor **t, const float *w, int O, int I, int device){
    if(!g_dsv4.available) return 0;
    return g_dsv4.upload_f32(t, w, O, I, device);
}

int dsv4_cuda_mhc_pre(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                      Dsv4CudaTensor *base, int M, int H, float rms_eps, float pre_eps, float sink_eps,
                      float post_mult, int sink_iters, Dsv4CudaActivation *state, Dsv4CudaActivation *input){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_pre(residual, fn, scale, base, M, H, rms_eps, pre_eps, sink_eps, post_mult,
                          sink_iters, state, input);
}

int dsv4_cuda_mhc_post(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                       const Dsv4CudaActivation *state, int M, int H, Dsv4CudaActivation *out){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_post(x, residual, state, M, H, out);
}

int dsv4_cuda_attention_first(const Dsv4CudaActivation *input, Dsv4CudaTensor *attn_norm,
                              Dsv4CudaTensor *q_a, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
                              Dsv4CudaTensor *wkv, Dsv4CudaTensor *kv_norm, Dsv4CudaTensor *sink,
                              Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b, int heads, int head_dim,
                              int qk_rope, int groups, float eps, Dsv4CudaActivation *output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.attention_first(input, attn_norm, q_a, q_norm, q_b, wkv, kv_norm, sink, wo_a, wo_b,
                                  heads, head_dim, qk_rope, groups, eps, output);
}

Dsv4CudaKvCache *dsv4_cuda_kv_create(int device, int window, int head_dim, int max_tokens,
                                     int rope_pairs, const float *rope_cos, const float *rope_sin,
                                     const float *compress_cos, const float *compress_sin){
    if(!g_dsv4.available) return NULL;
    return g_dsv4.kv_create(device, window, head_dim, max_tokens, rope_pairs,
                            rope_cos, rope_sin, compress_cos, compress_sin);
}

int dsv4_cuda_mhc_pre_norm(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                           Dsv4CudaTensor *base, Dsv4CudaTensor *norm, int M, int H, float rms_eps,
                           float pre_eps, float sink_eps, float post_mult, int sink_iters, float norm_eps,
                           Dsv4CudaActivation *state, Dsv4CudaActivation *input){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_pre_norm(residual, fn, scale, base, norm, M, H, rms_eps, pre_eps, sink_eps,
                               post_mult, sink_iters, norm_eps, state, input);
}

int dsv4_cuda_mhc_pre_norm_batch(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn,
                                 Dsv4CudaTensor *scale, Dsv4CudaTensor *base, Dsv4CudaTensor *norm,
                                 int tokens, int H, Dsv4CudaActivation *state, Dsv4CudaActivation *input){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_pre_norm_batch(residual, fn, scale, base, norm, tokens, H, state, input);
}

int dsv4_cuda_mhc_pre_batch(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                            Dsv4CudaTensor *base, int tokens, int H, Dsv4CudaActivation *state,
                            Dsv4CudaActivation *input){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_pre_batch(residual, fn, scale, base, tokens, H, state, input);
}

int dsv4_cuda_mhc_post_pre(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                           Dsv4CudaActivation *state, int M, int H, Dsv4CudaActivation *out,
                           Dsv4CudaTensor *fn, Dsv4CudaTensor *scale, Dsv4CudaTensor *base,
                           float rms_eps, float pre_eps, float sink_eps, float post_mult, int sink_iters,
                           Dsv4CudaActivation *input){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_post_pre(x, residual, state, M, H, out, fn, scale, base,
                               rms_eps, pre_eps, sink_eps, post_mult, sink_iters, input);
}

int dsv4_cuda_mhc_post_pre_norm(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                                Dsv4CudaActivation *state, int M, int H, Dsv4CudaActivation *out,
                                Dsv4CudaTensor *fn, Dsv4CudaTensor *scale, Dsv4CudaTensor *base,
                                Dsv4CudaTensor *norm, float rms_eps, float pre_eps, float sink_eps,
                                float post_mult, int sink_iters, float norm_eps, Dsv4CudaActivation *input){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_post_pre_norm(x, residual, state, M, H, out, fn, scale, base, norm,
                                    rms_eps, pre_eps, sink_eps, post_mult, sink_iters, norm_eps, input);
}

int dsv4_cuda_mhc_post_pre_norm_batch(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                                      Dsv4CudaActivation *state, int tokens, int H, Dsv4CudaActivation *out,
                                      Dsv4CudaTensor *fn, Dsv4CudaTensor *scale, Dsv4CudaTensor *base,
                                      Dsv4CudaTensor *norm, Dsv4CudaActivation *input){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_post_pre_norm_batch(x, residual, state, tokens, H, out, fn, scale, base, norm, input);
}

int dsv4_cuda_mhc_post_batch(const Dsv4CudaActivation *x, const Dsv4CudaActivation *residual,
                             const Dsv4CudaActivation *state, int tokens, int H, Dsv4CudaActivation *out){
    if(!g_dsv4.available) return 0;
    return g_dsv4.mhc_post_batch(x, residual, state, tokens, H, out);
}

void dsv4_cuda_kv_free(Dsv4CudaKvCache *cache){
    if(g_dsv4.available && g_dsv4.kv_free) g_dsv4.kv_free(cache);
}

int dsv4_cuda_attention_window(const Dsv4CudaActivation *input, Dsv4CudaTensor *attn_norm,
                               Dsv4CudaTensor *q_a, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
                               Dsv4CudaTensor *wkv, Dsv4CudaTensor *kv_norm, Dsv4CudaTensor *sink,
                               Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b,
                               Dsv4CudaTensor *compress_wkv, Dsv4CudaTensor *compress_wgate,
                               Dsv4CudaTensor *compress_ape, Dsv4CudaTensor *compress_norm,
                               int compress_ratio, int heads, int head_dim, int qk_rope, int groups,
                               int pos, float eps, Dsv4CudaKvCache *cache, Dsv4CudaActivation *output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.attention_window(input, attn_norm, q_a, q_norm, q_b, wkv, kv_norm, sink, wo_a, wo_b,
                                   compress_wkv, compress_wgate, compress_ape, compress_norm,
                                   compress_ratio, heads, head_dim, qk_rope, groups, pos, eps, cache, output);
}

int dsv4_cuda_attention_sparse_batch(const Dsv4CudaActivation *input, Dsv4CudaTensor *attn_norm,
                                     Dsv4CudaTensor *qkv, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b,
                                     Dsv4CudaTensor *kv_norm, Dsv4CudaTensor *sink, int heads,
                                     int head_dim, int start_pos, int tokens, float eps,
                                     Dsv4CudaKvCache *cache, Dsv4CudaActivation *context){
    if(!g_dsv4.available) return 0;
    return g_dsv4.attention_sparse_batch(input, attn_norm, qkv, q_norm, q_b, kv_norm, sink, heads,
                                         head_dim, start_pos, tokens, eps, cache, context);
}

int dsv4_cuda_attention_output_batch(const Dsv4CudaActivation *context, Dsv4CudaTensor *wo_a,
                                     Dsv4CudaTensor *wo_b, int groups, int tokens, Dsv4CudaActivation *output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.attention_output_batch(context, wo_a, wo_b, groups, tokens, output);
}

int dsv4_cuda_attention_window_tp2(const Dsv4CudaActivation *input, Dsv4CudaActivation *peer_input,
                                   const Dsv4CudaAttentionWeights *primary, const Dsv4CudaAttentionWeights *peer,
                                   int compress_ratio, int heads, int head_dim, int qk_rope, int groups,
                                   int pos, float eps, Dsv4CudaKvCache *cache, Dsv4CudaKvCache *peer_cache,
                                   Dsv4CudaActivation *output, Dsv4CudaActivation *peer_output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.attention_window_tp2(input, peer_input, primary, peer, compress_ratio, heads, head_dim,
                                       qk_rope, groups, pos, eps, cache, peer_cache, output, peer_output);
}

int dsv4_cuda_route(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                    const int *fixed_ids, float routed_scale, int ids[6], float weights[6]){
    if(!g_dsv4.available) return 0;
    return g_dsv4.route(input, gate, bias, fixed_ids, routed_scale, ids, weights);
}

int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t, float *y, const float *x, int groups){
    if(!g_dsv4.available) return 0;
    return g_dsv4.matvec_grouped(t, y, x, groups);
}

int dsv4_cuda_matvec(Dsv4CudaTensor *t, float *y, const float *x){
    if(!g_dsv4.available) return 0;
    return g_dsv4.matvec(t, y, x);
}

int dsv4_cuda_matmul_batch(Dsv4CudaTensor *t, const Dsv4CudaActivation *input, int tokens,
                           Dsv4CudaActivation *output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.matmul_batch(t, input, tokens, output);
}

int dsv4_cuda_matmul_bf16_batch(Dsv4CudaTensor *t, const float *x, int tokens, float *y){
    if(!g_dsv4.available) return 0;
    return g_dsv4.matmul_bf16_batch(t, x, tokens, y);
}

int dsv4_cuda_sparse_attn_batch(int device, const float *q, const float *vals, const float *sinks,
                                const int *meta, int value_rows, int comp_base, int heads, int dim,
                                int tokens, float scale, float *out){
    if(!g_dsv4.available) return 0;
    return g_dsv4.sparse_attn_batch(device, q, vals, sinks, meta, value_rows, comp_base,
                                    heads, dim, tokens, scale, out);
}

int dsv4_cuda_sparse_attn_batch_cached(int device, int layer, const float *q, const float *chunk,
                                       int chunk_start, const float *sinks,
                                       const int *meta, int abs_base, int comp_limit, int heads,
                                       int dim, int tokens, float scale, float *out){
    if(!g_dsv4.available) return 0;
    return g_dsv4.sparse_attn_batch_cached(device, layer, q, chunk, chunk_start, sinks, meta,
                                           abs_base, comp_limit, heads, dim, tokens, scale, out);
}

int dsv4_cuda_sparse_attn_batch_cached_idx(int device, int layer, const float *q, const float *chunk,
                                           int chunk_start, const float *sinks,
                                           const int *meta, const int *sel, int selstride,
                                           int abs_base, int comp_limit, int heads,
                                           int dim, int tokens, float scale, float *out){
    if(!g_dsv4.available) return 0;
    return g_dsv4.sparse_attn_batch_cached_idx(device, layer, q, chunk, chunk_start, sinks, meta,
                                               sel, selstride, abs_base, comp_limit, heads, dim,
                                               tokens, scale, out);
}

int dsv4_cuda_indexer_score_batch(int device, const float *queries, const float *keys,
                                  const float *head_w, const int *counts, int tokens,
                                  int heads, int dim, int count, float *scores){
    if(!g_dsv4.available) return 0;
    return g_dsv4.indexer_score_batch(device, queries, keys, head_w, counts, tokens,
                                      heads, dim, count, scores);
}

int dsv4_cuda_fp8_ref_matmul(int device, const uint8_t *w, const float *bscale,
                             int rows, int cols, int packed_rows8, const float *x,
                             int tokens, float *y){
    if(!g_dsv4.available) return 0;
    return g_dsv4.fp8_ref_matmul(device, w, bscale, rows, cols, packed_rows8, x, tokens, y);
}

int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t, const uint8_t *w, const uint8_t *scale, int O, int I, int sync){
    if(!g_dsv4.available) return 0;
    return g_dsv4.tensor_refill_fp4(t, w, scale, O, I, sync);
}

int dsv4_cuda_backend_arch_ok(int device){
    if(!g_dsv4.available) return 0;
    return g_dsv4.backend_arch_ok ? g_dsv4.backend_arch_ok(device) : 1;
}

const char *dsv4_cuda_backend_name(void){
    if(!g_dsv4.available) return "none";
    return g_dsv4.backend_name ? g_dsv4.backend_name() : "unknown";
}

long long dsv4_cuda_mem_free_mb(int device){
    if(!g_dsv4.available) return -1;
    return g_dsv4.mem_free_mb(device);
}

int dsv4_cuda_stream_drain(int device){
    /* Without the DLL there is never in-flight DMA to wait for, so a drain
     * trivially succeeds. With an older DLL that lacks the export, report
     * failure instead: the engine probes this once and keeps every attach
     * synchronous, so nothing is ever enqueued that could not be drained. */
    if(!g_dsv4.available) return 1;
    if(!g_dsv4.stream_drain) return 0;
    return g_dsv4.stream_drain(device);
}

int dsv4_cuda_kv_ring_append(int device, int layer, const float *rows, int start_pos,
                             int count, int window, int dim){
    if(!g_dsv4.available) return 0;
    return g_dsv4.kv_ring_append(device, layer, rows, start_pos, count, window, dim);
}

int dsv4_cuda_kv_comp_append(int device, int layer, const float *rows, int start_idx,
                             int count, int dim){
    if(!g_dsv4.available) return 0;
    return g_dsv4.kv_comp_append(device, layer, rows, start_idx, count, dim);
}

int dsv4_cuda_head_argmax(Dsv4CudaTensor *t, const float *x, int *id, float *value){
    if(!g_dsv4.available) return 0;
    return g_dsv4.head_argmax(t, x, id, value);
}

int dsv4_cuda_final_argmax(const Dsv4CudaActivation *residual, Dsv4CudaTensor *fn, Dsv4CudaTensor *scale,
                           Dsv4CudaTensor *base, Dsv4CudaTensor *norm, Dsv4CudaTensor *head, int M, int H,
                           float eps, float pre_eps, int *id, float *value){
    if(!g_dsv4.available) return 0;
    return g_dsv4.final_argmax(residual, fn, scale, base, norm, head, M, H, eps, pre_eps, id, value);
}

int dsv4_cuda_expert_group(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                           Dsv4CudaTensor *const *down, const float *weights, int count, float limit,
                           float *y, const float *x){
    if(!g_dsv4.available) return 0;
    return g_dsv4.expert_group(gate, up, down, weights, count, limit, y, x);
}

int dsv4_cuda_expert_fp8(Dsv4CudaTensor *gate, Dsv4CudaTensor *up, Dsv4CudaTensor *down, float limit,
                         float *y, const float *x){
    if(!g_dsv4.available) return 0;
    return g_dsv4.expert_fp8(gate, up, down, limit, y, x);
}

int dsv4_cuda_moe(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                  Dsv4CudaTensor *const *down, const float *weights, int count,
                  Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up, Dsv4CudaTensor *shared_down,
                  float limit, float *y, const float *x){
    if(!g_dsv4.available) return 0;
    return g_dsv4.moe(gate, up, down, weights, count, shared_gate, shared_up, shared_down, limit, y, x);
}

int dsv4_cuda_rmsnorm(Dsv4CudaActivation *x, Dsv4CudaTensor *weight, float eps, int elements){
    if(!g_dsv4.available) return 0;
    return g_dsv4.rmsnorm(x, weight, eps, elements);
}

int dsv4_cuda_moe_activation(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                             Dsv4CudaTensor *const *down, const float *weights, int count,
                             Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up,
                             Dsv4CudaTensor *shared_down, float limit, const Dsv4CudaActivation *input,
                             Dsv4CudaActivation *output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.moe_activation(gate, up, down, weights, count, shared_gate, shared_up, shared_down,
                                 limit, input, output);
}

Dsv4CudaExpertSet *dsv4_cuda_expert_set_create(Dsv4CudaTensor *const *gate, Dsv4CudaTensor *const *up,
                                               Dsv4CudaTensor *const *down, int count,
                                               Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up,
                                               Dsv4CudaTensor *shared_down){
    if(!g_dsv4.available) return NULL;
    return g_dsv4.expert_set_create(gate, up, down, count, shared_gate, shared_up, shared_down);
}

Dsv4CudaExpertSet *dsv4_cuda_expert_bank_create(int count, int hidden, int intermediate, int device,
                                                Dsv4CudaTensor *shared_gate, Dsv4CudaTensor *shared_up,
                                                Dsv4CudaTensor *shared_down){
    if(!g_dsv4.available) return NULL;
    return g_dsv4.expert_bank_create(count, hidden, intermediate, device, shared_gate, shared_up, shared_down);
}

int dsv4_cuda_expert_bank_upload(Dsv4CudaExpertSet *set, int expert,
                                 const uint8_t *gate_weight, const uint8_t *gate_scale,
                                 const uint8_t *up_weight, const uint8_t *up_scale,
                                 const uint8_t *down_weight, const uint8_t *down_scale,
                                 Dsv4CudaTensor **gate, Dsv4CudaTensor **up, Dsv4CudaTensor **down){
    if(!g_dsv4.available) return 0;
    return g_dsv4.expert_bank_upload(set, expert, gate_weight, gate_scale, up_weight, up_scale,
                                     down_weight, down_scale, gate, up, down);
}

int dsv4_cuda_expert_bank_upload_aux(Dsv4CudaExpertSet *set, int expert,
                                     const uint8_t *gate_weight, const uint8_t *gate_scale,
                                     const uint8_t *up_weight, const uint8_t *up_scale,
                                     const uint8_t *down_weight, const uint8_t *down_scale,
                                     Dsv4CudaTensor **gate, Dsv4CudaTensor **up, Dsv4CudaTensor **down){
    /* Optional export: an older DLL simply fails the first prefetch upload,
     * and the engine falls back to the single-bank path for good. */
    if(!g_dsv4.available || !g_dsv4.expert_bank_upload_aux) return 0;
    return g_dsv4.expert_bank_upload_aux(set, expert, gate_weight, gate_scale, up_weight, up_scale,
                                         down_weight, down_scale, gate, up, down);
}

int dsv4_cuda_expert_bank_set_shared(Dsv4CudaExpertSet *set, Dsv4CudaTensor *sg,
                                     Dsv4CudaTensor *su, Dsv4CudaTensor *sd){
    if(!g_dsv4.available) return 0;
    return g_dsv4.expert_bank_set_shared(set, sg, su, sd);
}

int dsv4_cuda_expert_bank_upload_tp2(Dsv4CudaExpertSet *set, int expert, int rank,
                                     const uint8_t *gate_weight, const uint8_t *gate_scale,
                                     const uint8_t *up_weight, const uint8_t *up_scale,
                                     const uint8_t *down_weight, const uint8_t *down_scale){
    if(!g_dsv4.available) return 0;
    return g_dsv4.expert_bank_upload_tp2(set, expert, rank, gate_weight, gate_scale, up_weight, up_scale,
                                         down_weight, down_scale);
}

void dsv4_cuda_expert_set_free(Dsv4CudaExpertSet *set){
    if(g_dsv4.available && g_dsv4.expert_set_free) g_dsv4.expert_set_free(set);
}

int dsv4_cuda_expert_set_upload_hash(Dsv4CudaExpertSet *set, const int64_t *map, int vocab, int topk){
    if(!g_dsv4.available) return 0;
    return g_dsv4.expert_set_upload_hash(set, map, vocab, topk);
}

int dsv4_cuda_route_moe(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                        int token, float routed_scale, Dsv4CudaExpertSet *experts, float limit,
                        Dsv4CudaActivation *output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.route_moe(input, gate, bias, token, routed_scale, experts, limit, output);
}

int dsv4_cuda_route_moe_batch(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                              const int *tokens, int count, float routed_scale, Dsv4CudaExpertSet *experts,
                              float limit, Dsv4CudaActivation *output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.route_moe_batch(input, gate, bias, tokens, count, routed_scale, experts, limit, output);
}

int dsv4_cuda_route_top6_batch(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                               int count, float routed_scale, int *ids, float *weights){
    if(!g_dsv4.available) return 0;
    return g_dsv4.route_top6_batch(input, gate, bias, count, routed_scale, ids, weights);
}

int dsv4_cuda_route_moe_ids_batch(const Dsv4CudaActivation *input, const int *ids, const float *weights,
                                  int count, Dsv4CudaExpertSet *experts, float limit,
                                  Dsv4CudaActivation *output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.route_moe_ids_batch(input, ids, weights, count, experts, limit, output);
}

int dsv4_cuda_route_moe_ep2(const Dsv4CudaActivation *input, Dsv4CudaTensor *gate, Dsv4CudaTensor *bias,
                            const Dsv4CudaActivation *peer_input, Dsv4CudaTensor *peer_gate,
                            Dsv4CudaTensor *peer_bias, int token, float routed_scale,
                            Dsv4CudaExpertSet *local, Dsv4CudaExpertSet *peer, float limit,
                            Dsv4CudaActivation *output, Dsv4CudaActivation *peer_output){
    if(!g_dsv4.available) return 0;
    return g_dsv4.route_moe_ep2(input, gate, bias, peer_input, peer_gate, peer_bias, token, routed_scale,
                                local, peer, limit, output, peer_output);
}

int dsv4_cuda_qkv(Dsv4CudaTensor *q_a, Dsv4CudaTensor *q_norm, Dsv4CudaTensor *q_b, Dsv4CudaTensor *kv,
                  float eps, float *q_out, float *kv_out, const float *x){
    if(!g_dsv4.available) return 0;
    return g_dsv4.qkv(q_a, q_norm, q_b, kv, eps, q_out, kv_out, x);
}

int dsv4_cuda_wo(Dsv4CudaTensor *wo_a, Dsv4CudaTensor *wo_b, int groups, float *out, const float *context){
    if(!g_dsv4.available) return 0;
    return g_dsv4.wo(wo_a, wo_b, groups, out, context);
}

void dsv4_cuda_tensor_free(Dsv4CudaTensor *t){
    if(g_dsv4.available && g_dsv4.tensor_free) g_dsv4.tensor_free(t);
}

long long dsv4_cuda_tensor_bytes(const Dsv4CudaTensor *t){
    if(!g_dsv4.available) return 0;
    return g_dsv4.tensor_bytes(t);
}

int dsv4_cuda_tensor_device(const Dsv4CudaTensor *t){
    if(!g_dsv4.available) return -1;
    return g_dsv4.tensor_device(t);
}

#endif /* _WIN32 */
