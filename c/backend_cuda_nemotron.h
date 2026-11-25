/* backend_cuda_nemotron.h -- Self-contained CUDA backend for Nemotron-H /
 * Nemotron-3.5-Super-VL-120B-A12B (Mamba-2 + NoPE GQA Attention + LatentMoE
 * NVFP4/FP8/BF16/F32).
 *
 * On Linux with CUDA=1, backend_cuda_nemotron.o links directly into nemotron.
 * On Windows, nvcc+MSVC builds coli_cuda_nemotron.dll exporting
 * coli_cuda_nemotron_ops(); nemotron.exe loads it at runtime via LoadLibraryA
 * and falls back cleanly to CPU when the DLL or a CUDA GPU is absent.
 */
#ifndef BACKEND_CUDA_NEMOTRON_H
#define BACKEND_CUDA_NEMOTRON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    (*init)(int device);
    void   (*shutdown)(void);
    void   (*mem_info)(size_t *free_bytes, size_t *total_bytes);
    void  *(*alloc)(size_t nbytes);
    void  *(*upload)(const void *host, size_t nbytes);
    void   (*free_dev)(void *dev);
    int    (*h2d)(void *dst_dev, const void *src_host, size_t nbytes);
    int    (*d2h)(void *dst_host, const void *src_dev, size_t nbytes);
    int    (*memset_zero)(void *dst_dev, size_t nbytes);
    int    (*sync)(void);

    void   (*rmsnorm)(float *d_out, const float *d_in, const float *d_w, int n, float eps);
    void   (*add_inplace)(float *d_a, const float *d_b, int n);
    /* fmt: 0=F32, 1=BF16, 2=FP8_E4M3, 3=NVFP4 (U8 nibbles + E4M3 per-16 scales + scale_scalar)
     * act_relu2: if non-zero, apply max(0, v)^2 before store.
     * accum_scale: if != 0.0f, do d_y[r] += accum_scale * v instead of d_y[r] = v. */
    void   (*matvec)(float *d_y, const void *d_w, const uint8_t *d_sc, float scale_scalar,
                     int fmt, int O, int I, const float *d_x, int act_relu2, float accum_scale);

    /* Mamba-2 fused GPU kernels */
    void   (*mamba_conv_silu)(float *d_xbc_out, float *d_conv_state, const float *d_xbc_in,
                              const float *d_conv_w, const float *d_conv_b, int C_dim, int K);
    void   (*mamba_ssm_step)(float *d_y, float *d_ssm_state, const float *d_xbc,
                             const float *d_dt_raw, const float *d_dt_bias,
                             const float *d_A_neg_exp, const float *d_D,
                             int H, int P, int N, int G, float dt_min, float dt_max);
    void   (*mamba_gated_rmsnorm)(float *d_y, const float *d_z, const float *d_norm_w,
                                  int D_in, int G, float eps);

    /* NoPE GQA Attention GPU kernel (appends k,v at pos and attends 0..pos) */
    void   (*attn_nope_gqa)(float *d_out, float *d_k_cache, float *d_v_cache,
                            const float *d_q, const float *d_k, const float *d_v,
                            int H, int KVH, int hd, int pos);
} ColiNemoCudaOps;

#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
__declspec(dllexport)
#endif
const ColiNemoCudaOps *coli_cuda_nemotron_ops(void);

#ifdef __cplusplus
}
#endif
#endif
