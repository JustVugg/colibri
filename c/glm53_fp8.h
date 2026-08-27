#ifndef COLIBRI_GLM53_FP8_H
#define COLIBRI_GLM53_FP8_H

#include <stdint.h>

/* GLM-5.3 fine-grained FP8 matvec. Weights use row-major E4M3FN with one
 * float32 dequant scale per 128x128 tile. Activations are dynamically
 * quantized to E4M3FN independently for every 128-column block. */
int coli_glm53_fp8_matvec(float *output, const uint8_t *weight, const float *weight_scale_inv, const float *input,
                          int rows, int columns);
int coli_glm53_fp8_quantize_activation(float *output, const float *input, int columns);
void coli_glm53_fp8_decode_table(float output[256]);

#endif
