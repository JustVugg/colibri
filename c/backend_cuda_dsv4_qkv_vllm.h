#ifndef COLIBRI_BACKEND_CUDA_DSV4_QKV_VLLM_H
#define COLIBRI_BACKEND_CUDA_DSV4_QKV_VLLM_H
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int dsv4_vllm_qnorm_rope_kv_insert(const void *q_in,void *q_out,const void *kv_in,
                                    uint8_t *kv_cache,const int64_t *slot_mapping,
                                    const int64_t *position_ids,const float *cos_sin_cache,
                                    float eps,int heads,int cache_block_size,
                                    int kv_block_stride,cudaStream_t stream);
int dsv4_vllm_qnorm_rope_kv_insert_batch(const void *q_in,void *q_out,const void *kv_in,
                                          uint8_t *kv_cache,const int64_t *slot_mapping,
                                          const int64_t *position_ids,const float *cos_sin_cache,
                                          float eps,int tokens,int insert_tokens,int heads,
                                          int padded_heads,int cache_block_size,
                                          int kv_block_stride,cudaStream_t stream);
int dsv4_vllm_pack_compressed(const float *kv,uint8_t *cache,const int *decode_state,
                              int ratio,
                              int cache_block_size,int kv_block_stride,
                              cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
