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

/* Correctness-first FP8 matvec for native 128x128 E4M3 weight blocks with
 * UE8M0 scales and dynamically quantized E4M3 activations. */
int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input);

#ifdef __cplusplus
}
#endif

#endif
