#ifndef COLIBRI_BACKEND_CUDA_DSV4_FLASHINFER_H
#define COLIBRI_BACKEND_CUDA_DSV4_FLASHINFER_H
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
int dsv4_flashinfer_grouped(const void *descriptors,int count,int O,int I,int device,cudaStream_t stream);
int dsv4_flashinfer_sparse_mla(const void *q,const void *kv_cache,const int *indices,
                               void *mid_out,float *mid_lse,void *output,float *out_lse,
                               const int *topk_length,const float *attn_sink,
                               const void *extra_kv_cache,const int *extra_indices,
                               const int *extra_topk_length,int extra_topk,int pbs_extra,
                               size_t stride_extra_kv_block,int heads,int topk,int tokens,
                               int splits,int chunks_per_block,float sm_scale,
                               size_t stride_kv_block,cudaStream_t stream);
void dsv4_flashinfer_shutdown(void);
#ifdef __cplusplus
}
#endif
#endif
