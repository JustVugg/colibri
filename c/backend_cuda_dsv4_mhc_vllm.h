#ifndef COLIBRI_BACKEND_CUDA_DSV4_MHC_VLLM_H
#define COLIBRI_BACKEND_CUDA_DSV4_MHC_VLLM_H

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t dsv4_vllm_mhc_post_pre_norm(
    cudaStream_t stream, const float *comb_mix, const void *residual_in,
    const float *post_mix, const void *x_in, const float *weight_t,
    float *gemm_out_mul, float *gemm_out_sqrsum, void *residual_out,
    const float *hc_scale, const float *hc_base, void *layer_input,
    const void *norm_weight);
cudaError_t dsv4_vllm_mhc_post_pre_norm_batch(
    cudaStream_t stream, const float *comb_mix, const void *residual_in,
    const float *post_mix, const void *x_in, const float *weight_t,
    float *gemm_out_mul, float *gemm_out_sqrsum, void *residual_out,
    const float *hc_scale, const float *hc_base, void *layer_input,
    const void *norm_weight, int num_tokens);

#ifdef __cplusplus
}
#endif

#endif
