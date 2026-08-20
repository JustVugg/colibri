#ifndef COLIBRI_NATIVE_QUANT_H
#define COLIBRI_NATIVE_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

float coli_e8m0_decode(uint8_t value);
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

#ifdef __cplusplus
}
#endif

#endif
