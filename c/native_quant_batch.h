#ifndef COLIBRI_NATIVE_QUANT_BATCH_H
#define COLIBRI_NATIVE_QUANT_BATCH_H

#include "tensor.h"

/* inputs and outputs are batch-major. Every batch element preserves the exact
 * scalar column accumulation order of the single-token reference kernels. */
int coli_fp8_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch);
int coli_fp4_matmul_batch_ref(float *outputs, const ColiTensorView *weight,
                              const float *inputs, int batch);

#endif
