#ifndef COLIBRI_NATIVE_QUANT_H
#define COLIBRI_NATIVE_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

float coli_e8m0_decode(uint8_t value);
/* Process-wide immutable decode table shared by split native-quant units. */
const float *coli_e8m0_table(void);
float coli_e2m1_decode(uint8_t nibble);
float coli_e4m3fn_decode(uint8_t value);
uint8_t coli_e4m3fn_encode(float value);
float coli_bf16_round(float value);
float coli_bf16_decode(uint16_t value);
void coli_bf16_round_array(float *values, size_t count);

/* Simulates the official dynamic E4M3 activation quantization with one E8M0
 * power-of-two scale per block. Output contains the dequantized FP32 values. */
int coli_fp8_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size);
int coli_fp4_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size);
int coli_hadamard_bf16_ref(float *values, size_t length);

/* Correctness-first FP4 matvec. The input is dynamically quantized to E4M3 in
 * blocks of 128, weights are native E2M1 with one E8M0 scale per 32 K, and
 * accumulation is FP32. */
int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input);

/* #1136 convergence core: row-major fp4 matvec accumulating in the rows16
 * kernels' per-row order — (x*w)*scale folded straight into the row
 * accumulator, column by column, product rounded before the add — so a
 * rows16-packed and a row-major copy of the same matrix produce identical
 * bits. Requires I%32==0 and O%16==0 (the only shapes rows16 can pack);
 * `x` is the already-qdq'd activation. */
void coli_fp4_matvec_rows16_order(float *y, const uint8_t *q4,
                                  const uint8_t *e8s, const float *x,
                                  int I, int O);
/* Batch-major companion: independent scalar-order accumulators share each
 * decoded weight tile, so the matrix is streamed once for the whole batch. */
void coli_fp4_matmul_batch_rows16_order(float *y, const uint8_t *q4,
                                        const uint8_t *e8s, const float *x,
                                        int batch, int I, int O);

/* Correctness-first FP8 matvec for native 128x128 E4M3 weight blocks with
 * UE8M0 scales and dynamically quantized E4M3 activations. */
int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input);

/* Growable thread-local scratch for the qdq activation buffers shared by the
 * *_ref/_pre matvec and matmul entries (defined in the NATIVE_QUANT unit).
 * Replaces the historical malloc/free pair per call; buffers are never
 * returned. Values computed from them are bit-for-bit unchanged. */
int coli_v4_qdq_scratch(size_t activation_count, size_t scales_count,
                        float **activation, uint8_t **scales);

/* Hoisted-qdq FP8 matvec: `activation` is `input` already passed through
 * coli_fp8_activation_qdq_ref once by the caller (wq_a and wkv consume the
 * same vector); `input` stays raw for the GPU path, exactly as in _ref.
 * Same checks, same compute: bit-identical to _ref. */
int coli_fp8_matvec_pre(float *output, const ColiTensorView *weight,
                        const float *input, const float *activation);

/* Optional CUDA tier (Windows engine build only, COLI_V4_GPU_TIER). The
 * engine-side wrappers in deepseek_v4.c resolve coli_cuda_dsv4.dll through
 * backend_loader_dsv4.c; the matvec_ref implementations dispatch to them when
 * the resident weight mirror (ColiTensorView.gpu) is non-NULL. */
#ifdef COLI_V4_GPU_TIER
int coli_v4_gpu_fp8_matvec(const ColiTensorView *w, float *output,
                           const float *input);
int coli_v4_gpu_matvec_grouped(const ColiTensorView *w, float *output,
                               const float *input, int groups);
int coli_v4_gpu_fp8_matmul_batch(const ColiTensorView *w, float *outputs,
                                 const float *inputs, int batch);
#endif

/* Optional Vulkan path (Makefile.deepseek-v4 VK=1). A pointer, not a function:
 * the engine binary sets it when COLI_VULKAN=1 opened a device, and every other
 * link of these units (the parent Makefile's tests among them, which carry no
 * backend) sees NULL and stays on the CPU. It computes
 * output[batch, rows] = input[batch, columns] W^T for a weight the engine keeps
 * resident, in the device's own formats: fmt 12 is E4M3 with the f32 128x128
 * block scales (rows8 = the AVX2 8-row tile layout), fmt 11 is bf16 with no
 * scales. 0 = done on the device, anything else = run the CPU path. */
#ifdef COLI_VULKAN
typedef int (*ColiV4VkMatmul)(int fmt, const void *data, const float *scales,
                              int rows8, int rows, int columns, float *output,
                              const float *input, int batch);
extern ColiV4VkMatmul coli_v4_vk_matmul;

/* With the dense weights on the device only (COLI_VK_DENSE_HOST), a matrix may have no
 * host copy: the CPU paths call this before they read one, and it is read back from disk
 * at its own address (deepseek_v4_internal.h, coli_v4_layer_host_restore). NULL when no
 * device holds a matrix alone. */
typedef int (*ColiV4VkHost)(const void *data);
extern ColiV4VkHost coli_v4_vk_host;
static inline void coli_v4_vk_host_ensure(const void *data) {
    if (coli_v4_vk_host && data) coli_v4_vk_host(data);
}

/* An fp8 view on the device, fed the activation the CPU kernel would read
 * (already rounded to E4M3 per 128 by coli_fp8_activation_qdq_ref). */
static inline int coli_v4_vk_fp8(float *output, const ColiTensorView *weight,
                                 const float *activation, int batch) {
    return coli_v4_vk_matmul &&
           coli_v4_vk_matmul(12, weight->data, (const float *)weight->scales,
                             weight->block_rows == 8, (int)weight->rows,
                             (int)weight->columns, output, activation, batch) == 0;
}
#endif

#ifdef __cplusplus
}
#endif

#endif
