#ifndef COLIBRI_BACKEND_CUDA_DSV4_H
#define COLIBRI_BACKEND_CUDA_DSV4_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct Dsv4CudaTensor Dsv4CudaTensor;
typedef struct Dsv4CudaActivation Dsv4CudaActivation;
typedef struct Dsv4CudaKvCache Dsv4CudaKvCache;
typedef struct Dsv4CudaExpertSet Dsv4CudaExpertSet;
typedef struct Dsv4CudaGraph Dsv4CudaGraph;
typedef struct {
    Dsv4CudaTensor *attn_norm,*q_a,*qkv,*q_norm,*q_b,*wkv,*kv_norm,*sink,*wo_a,*wo_b;
    Dsv4CudaTensor *compress_wkv,*compress_wgate,*compress_ape,*compress_norm;
} Dsv4CudaAttentionWeights;
int dsv4_cuda_init(const int *devices,int count);
void dsv4_cuda_shutdown(void);
int dsv4_cuda_upload_fp8(Dsv4CudaTensor **t,const uint8_t *w,const uint8_t *scale,int O,int I,int device);
int dsv4_cuda_upload_fp8_bf16(Dsv4CudaTensor **t,const uint8_t *w,const uint8_t *scale,int O,int I,int device);
int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t,const uint8_t *w,const uint8_t *scale,int O,int I,int device);
int dsv4_cuda_upload_bf16(Dsv4CudaTensor **t,const uint16_t *w,int O,int I,int device);
/* Same-shape in-place refill of an fp4 mirror (no realloc); sync=1 drains. */
int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t,const uint8_t *w,const uint8_t *scale,int O,int I,int sync);
int dsv4_cuda_upload_f32(Dsv4CudaTensor **t,const float *w,int O,int I,int device);
int dsv4_cuda_matvec(Dsv4CudaTensor *t,float *y,const float *x);
int dsv4_cuda_matmul_batch(Dsv4CudaTensor *t,const Dsv4CudaActivation *input,
                           int tokens,Dsv4CudaActivation *output);
int dsv4_cuda_matmul_bf16_batch(Dsv4CudaTensor *t,const float *x,int tokens,float *y);
/* Host-pointer batched sparse window attention (prefill). vals holds
 * [0,comp_base) window rows in position order then the compressed rows;
 * meta[3t]=window row offset, meta[3t+1]=window rows, meta[3t+2]=compressed
 * prefix visible to token t. out receives [tokens][heads*dim]. */
int dsv4_cuda_sparse_attn_batch(int device,const float *q,const float *vals,
                                const float *sinks,const int *meta,int value_rows,
                                int comp_base,int heads,int dim,int tokens,
                                float scale,float *out);
/* chunk holds the current chunk's kv rows (tokens x dim) starting at absolute
 * position chunk_start; the engine appends them to the ring only AFTER the
 * call (ring aliasing past window positions). */
int dsv4_cuda_sparse_attn_batch_cached(int device,int layer,const float *q,
                                       const float *chunk,int chunk_start,
                                       const float *sinks,const int *meta,int abs_base,
                                       int comp_limit,int heads,int dim,int tokens,
                                       float scale,float *out);
/* Indexed cached variant: meta[3t+2] is the SELECTED count and sel holds each
 * token's compressed ordinals (tokens x selstride, score order). */
int dsv4_cuda_sparse_attn_batch_cached_idx(int device,int layer,const float *q,
                                           const float *chunk,int chunk_start,
                                           const float *sinks,const int *meta,
                                           const int *sel,int selstride,int abs_base,
                                           int comp_limit,int heads,int dim,int tokens,
                                           float scale,float *out);
/* Batched indexer scoring: scores[t][c] = sum_h relu(q_th . k_c) * w_th for
 * c < counts[t] (else 0), sequential fp32 order identical to the CPU loop. */
int dsv4_cuda_indexer_score_batch(int device,const float *queries,const float *keys,
                                  const float *head_w,const int *counts,int tokens,
                                  int heads,int dim,int count,float *scores);
/* Bitwise replica of the CPU reference fp8 matmul (row-major e4m3 weights,
 * 128x128 block f32 scales, host-quantized activations). y[tokens][rows]. */
int dsv4_cuda_fp8_ref_matmul(int device,const uint8_t *w,const float *bscale,
                             int rows,int cols,int packed_rows8,const float *x,
                             int tokens,float *y);
/* Build/GPU compatibility (loader DLL selection). */
int dsv4_cuda_backend_arch_ok(int device);
const char *dsv4_cuda_backend_name(void);
long long dsv4_cuda_mem_free_mb(int device);
/* Drain the device's expert stream (hybrid split: async fill DMA is enqueued,
 * the CPU subset computes meanwhile, this closes the pipeline). Returns 1 on
 * success, 0 when the backend is unavailable. */
int dsv4_cuda_stream_drain(int device);
int dsv4_cuda_kv_ring_append(int device,int layer,const float *rows,int start_pos,
                             int count,int window,int dim);
int dsv4_cuda_kv_comp_append(int device,int layer,const float *rows,int start_idx,
                             int count,int dim);
int dsv4_cuda_head_argmax(Dsv4CudaTensor *t,const float *x,int *id,float *value);
int dsv4_cuda_final_argmax(const Dsv4CudaActivation *residual,Dsv4CudaTensor *fn,Dsv4CudaTensor *scale,
                           Dsv4CudaTensor *base,Dsv4CudaTensor *norm,Dsv4CudaTensor *head,
                           int M,int H,float eps,float pre_eps,int *id,float *value);
int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t,float *y,const float *x,int groups);
int dsv4_cuda_expert_group(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,
                           Dsv4CudaTensor *const *down,const float *weights,int count,
                           float limit,float *y,const float *x);
int dsv4_cuda_expert_fp8(Dsv4CudaTensor *gate,Dsv4CudaTensor *up,Dsv4CudaTensor *down,
                         float limit,float *y,const float *x);
int dsv4_cuda_moe(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,
                  Dsv4CudaTensor *const *down,const float *weights,int count,
                  Dsv4CudaTensor *shared_gate,Dsv4CudaTensor *shared_up,
                  Dsv4CudaTensor *shared_down,float limit,float *y,const float *x);
int dsv4_cuda_qkv(Dsv4CudaTensor *q_a,Dsv4CudaTensor *q_norm,Dsv4CudaTensor *q_b,
                  Dsv4CudaTensor *kv,float eps,float *q_out,float *kv_out,const float *x);
int dsv4_cuda_wo(Dsv4CudaTensor *wo_a,Dsv4CudaTensor *wo_b,int groups,
                 float *out,const float *context);
void dsv4_cuda_tensor_free(Dsv4CudaTensor *t);
long long dsv4_cuda_tensor_bytes(const Dsv4CudaTensor *t);
int dsv4_cuda_tensor_device(const Dsv4CudaTensor *t);
Dsv4CudaActivation *dsv4_cuda_activation_create(int device,long long elements);
void dsv4_cuda_activation_free(Dsv4CudaActivation *a);
int dsv4_cuda_activation_upload(Dsv4CudaActivation *a,const float *x,long long elements);
int dsv4_cuda_activation_download(float *x,const Dsv4CudaActivation *a,long long elements);
int dsv4_cuda_activation_copy(Dsv4CudaActivation *dst,const Dsv4CudaActivation *src,long long elements);
int dsv4_cuda_activation_copy_range(Dsv4CudaActivation *dst,long long dst_offset,
                                    const Dsv4CudaActivation *src,long long src_offset,
                                    long long elements);
int dsv4_cuda_activation_sync(const Dsv4CudaActivation *a);
int dsv4_cuda_activation_device(const Dsv4CudaActivation *a);
int dsv4_cuda_decode_state_set(int device,int token,int position);
void dsv4_cuda_profiler_start(void);
void dsv4_cuda_profiler_stop(void);
int dsv4_cuda_graph_begin(int device);
Dsv4CudaGraph *dsv4_cuda_graph_end(int device);
int dsv4_cuda_graph_end_pair(int primary,int peer,Dsv4CudaGraph **primary_graph,Dsv4CudaGraph **peer_graph);
int dsv4_cuda_graph_launch(Dsv4CudaGraph *graph);
void dsv4_cuda_graph_free(Dsv4CudaGraph *graph);
int dsv4_cuda_mhc_pre(const Dsv4CudaActivation *residual,Dsv4CudaTensor *fn,
                      Dsv4CudaTensor *scale,Dsv4CudaTensor *base,int M,int H,
                      float rms_eps,float pre_eps,float sink_eps,float post_mult,
                      int sink_iters,Dsv4CudaActivation *state,Dsv4CudaActivation *input);
int dsv4_cuda_mhc_pre_norm(const Dsv4CudaActivation *residual,Dsv4CudaTensor *fn,Dsv4CudaTensor *scale,
                           Dsv4CudaTensor *base,Dsv4CudaTensor *norm,int M,int H,float rms_eps,float pre_eps,
                           float sink_eps,float post_mult,int sink_iters,float norm_eps,
                           Dsv4CudaActivation *state,Dsv4CudaActivation *input);
int dsv4_cuda_mhc_pre_norm_batch(const Dsv4CudaActivation *residual,Dsv4CudaTensor *fn,
                                 Dsv4CudaTensor *scale,Dsv4CudaTensor *base,Dsv4CudaTensor *norm,
                                 int tokens,int H,Dsv4CudaActivation *state,Dsv4CudaActivation *input);
int dsv4_cuda_mhc_pre_batch(const Dsv4CudaActivation *residual,Dsv4CudaTensor *fn,
                            Dsv4CudaTensor *scale,Dsv4CudaTensor *base,int tokens,int H,
                            Dsv4CudaActivation *state,Dsv4CudaActivation *input);
int dsv4_cuda_mhc_post(const Dsv4CudaActivation *x,const Dsv4CudaActivation *residual,
                       const Dsv4CudaActivation *state,int M,int H,Dsv4CudaActivation *out);
int dsv4_cuda_mhc_post_pre(const Dsv4CudaActivation *x,const Dsv4CudaActivation *residual,
                           Dsv4CudaActivation *state,int M,int H,Dsv4CudaActivation *out,
                           Dsv4CudaTensor *fn,Dsv4CudaTensor *scale,Dsv4CudaTensor *base,
                           float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,
                           Dsv4CudaActivation *input);
int dsv4_cuda_mhc_post_pre_norm(const Dsv4CudaActivation *x,const Dsv4CudaActivation *residual,
                                Dsv4CudaActivation *state,int M,int H,Dsv4CudaActivation *out,
                                Dsv4CudaTensor *fn,Dsv4CudaTensor *scale,Dsv4CudaTensor *base,Dsv4CudaTensor *norm,
                                float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,float norm_eps,
                                Dsv4CudaActivation *input);
/* Batched state is stored as post_mix[tokens][4], then comb_mix[tokens][4][4]. */
int dsv4_cuda_mhc_post_pre_norm_batch(const Dsv4CudaActivation *x,const Dsv4CudaActivation *residual,
                                      Dsv4CudaActivation *state,int tokens,int H,Dsv4CudaActivation *out,
                                      Dsv4CudaTensor *fn,Dsv4CudaTensor *scale,Dsv4CudaTensor *base,
                                      Dsv4CudaTensor *norm,Dsv4CudaActivation *input);
int dsv4_cuda_mhc_post_batch(const Dsv4CudaActivation *x,const Dsv4CudaActivation *residual,
                             const Dsv4CudaActivation *state,int tokens,int H,
                             Dsv4CudaActivation *out);
int dsv4_cuda_attention_first(const Dsv4CudaActivation *input,Dsv4CudaTensor *attn_norm,
                              Dsv4CudaTensor *q_a,Dsv4CudaTensor *q_norm,Dsv4CudaTensor *q_b,
                              Dsv4CudaTensor *wkv,Dsv4CudaTensor *kv_norm,Dsv4CudaTensor *sink,
                              Dsv4CudaTensor *wo_a,Dsv4CudaTensor *wo_b,int heads,int head_dim,
                              int qk_rope,int groups,float eps,Dsv4CudaActivation *output);
Dsv4CudaKvCache *dsv4_cuda_kv_create(int device,int window,int head_dim,int max_tokens,
                                     int rope_pairs,const float *rope_cos,const float *rope_sin,
                                     const float *compress_cos,const float *compress_sin);
void dsv4_cuda_kv_free(Dsv4CudaKvCache *cache);
int dsv4_cuda_attention_window(const Dsv4CudaActivation *input,Dsv4CudaTensor *attn_norm,
                               Dsv4CudaTensor *q_a,Dsv4CudaTensor *q_norm,Dsv4CudaTensor *q_b,
                               Dsv4CudaTensor *wkv,Dsv4CudaTensor *kv_norm,Dsv4CudaTensor *sink,
                               Dsv4CudaTensor *wo_a,Dsv4CudaTensor *wo_b,
                               Dsv4CudaTensor *compress_wkv,Dsv4CudaTensor *compress_wgate,
                               Dsv4CudaTensor *compress_ape,Dsv4CudaTensor *compress_norm,
                               int compress_ratio,int heads,int head_dim,int qk_rope,int groups,int pos,float eps,Dsv4CudaKvCache *cache,
                               Dsv4CudaActivation *output);
int dsv4_cuda_attention_sparse_batch(const Dsv4CudaActivation *input,Dsv4CudaTensor *attn_norm,
                                     Dsv4CudaTensor *qkv,Dsv4CudaTensor *q_norm,Dsv4CudaTensor *q_b,
                                     Dsv4CudaTensor *kv_norm,Dsv4CudaTensor *sink,int heads,int head_dim,
                                     int start_pos,int tokens,float eps,Dsv4CudaKvCache *cache,
                                     Dsv4CudaActivation *context);
int dsv4_cuda_attention_output_batch(const Dsv4CudaActivation *context,Dsv4CudaTensor *wo_a,
                                     Dsv4CudaTensor *wo_b,int groups,int tokens,
                                     Dsv4CudaActivation *output);
int dsv4_cuda_attention_window_tp2(const Dsv4CudaActivation *input,Dsv4CudaActivation *peer_input,
                                   const Dsv4CudaAttentionWeights *primary,const Dsv4CudaAttentionWeights *peer,
                                   int compress_ratio,int heads,int head_dim,int qk_rope,int groups,int pos,float eps,
                                   Dsv4CudaKvCache *cache,Dsv4CudaKvCache *peer_cache,
                                   Dsv4CudaActivation *output,Dsv4CudaActivation *peer_output);
int dsv4_cuda_route(const Dsv4CudaActivation *input,Dsv4CudaTensor *gate,Dsv4CudaTensor *bias,
                    const int *fixed_ids,float routed_scale,int ids[6],float weights[6]);
int dsv4_cuda_rmsnorm(Dsv4CudaActivation *x,Dsv4CudaTensor *weight,float eps,int elements);
int dsv4_cuda_moe_activation(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,
                              Dsv4CudaTensor *const *down,const float *weights,int count,
                              Dsv4CudaTensor *shared_gate,Dsv4CudaTensor *shared_up,
                              Dsv4CudaTensor *shared_down,float limit,
                              const Dsv4CudaActivation *input,Dsv4CudaActivation *output);
Dsv4CudaExpertSet *dsv4_cuda_expert_set_create(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,
                                                Dsv4CudaTensor *const *down,int count,
                                                Dsv4CudaTensor *shared_gate,Dsv4CudaTensor *shared_up,
                                                Dsv4CudaTensor *shared_down);
Dsv4CudaExpertSet *dsv4_cuda_expert_bank_create(int count,int hidden,int intermediate,int device,
                                                 Dsv4CudaTensor *shared_gate,Dsv4CudaTensor *shared_up,
                                                 Dsv4CudaTensor *shared_down);
int dsv4_cuda_expert_bank_upload(Dsv4CudaExpertSet *set,int expert,
                                 const uint8_t *gate_weight,const uint8_t *gate_scale,
                                 const uint8_t *up_weight,const uint8_t *up_scale,
                                 const uint8_t *down_weight,const uint8_t *down_scale,
                                 Dsv4CudaTensor **gate,Dsv4CudaTensor **up,Dsv4CudaTensor **down);
int dsv4_cuda_expert_bank_set_shared(Dsv4CudaExpertSet *set,Dsv4CudaTensor *sg,
                                     Dsv4CudaTensor *su,Dsv4CudaTensor *sd);
int dsv4_cuda_expert_bank_upload_tp2(Dsv4CudaExpertSet *set,int expert,int rank,
                                     const uint8_t *gate_weight,const uint8_t *gate_scale,
                                     const uint8_t *up_weight,const uint8_t *up_scale,
                                     const uint8_t *down_weight,const uint8_t *down_scale);
void dsv4_cuda_expert_set_free(Dsv4CudaExpertSet *set);
int dsv4_cuda_expert_set_upload_hash(Dsv4CudaExpertSet *set,const int64_t *map,int vocab,int topk);
int dsv4_cuda_route_moe(const Dsv4CudaActivation *input,Dsv4CudaTensor *gate,Dsv4CudaTensor *bias,
                        int token,float routed_scale,Dsv4CudaExpertSet *experts,
                        float limit,Dsv4CudaActivation *output);
int dsv4_cuda_route_moe_batch(const Dsv4CudaActivation *input,Dsv4CudaTensor *gate,
                              Dsv4CudaTensor *bias,const int *tokens,int count,float routed_scale,
                              Dsv4CudaExpertSet *experts,float limit,Dsv4CudaActivation *output);
/* Batched top-6 routing only; ids[count*6] / weights[count*6] land on the host. */
int dsv4_cuda_route_top6_batch(const Dsv4CudaActivation *input,Dsv4CudaTensor *gate,
                               Dsv4CudaTensor *bias,int count,float routed_scale,
                               int *ids,float *weights);
/* Batched MoE over host-provided routes; every id's bank slot must already
 * hold that expert (route-aware refill). */
int dsv4_cuda_route_moe_ids_batch(const Dsv4CudaActivation *input,const int *ids,
                                  const float *weights,int count,Dsv4CudaExpertSet *experts,
                                  float limit,Dsv4CudaActivation *output);
int dsv4_cuda_route_moe_ep2(const Dsv4CudaActivation *input,Dsv4CudaTensor *gate,Dsv4CudaTensor *bias,
                            const Dsv4CudaActivation *peer_input,Dsv4CudaTensor *peer_gate,Dsv4CudaTensor *peer_bias,
                            int token,float routed_scale,Dsv4CudaExpertSet *local,Dsv4CudaExpertSet *peer,
                            float limit,Dsv4CudaActivation *output,Dsv4CudaActivation *peer_output);
#ifdef __cplusplus
}
#endif
#endif
