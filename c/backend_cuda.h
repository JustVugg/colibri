#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* COLI_CUDA_DLLEXPORT marks functions exported from coli_cuda.dll on Windows.
 * Define COLI_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
#define COLI_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define COLI_CUDA_DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
COLI_CUDA_DLLEXPORT int coli_cuda_init(const int *devices, int count);
COLI_CUDA_DLLEXPORT void coli_cuda_shutdown(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_at(int index);
COLI_CUDA_DLLEXPORT int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* device < 0 returns aggregate statistics for all configured devices. */
COLI_CUDA_DLLEXPORT void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* Upload without executing, so capacity failures happen during model startup. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2.
 * The first successful call uploads W and its row scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
COLI_CUDA_DLLEXPORT int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

COLI_CUDA_DLLEXPORT void coli_cuda_tensor_free(ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
