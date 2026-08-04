#ifndef COLIBRI_NATIVE_QUANT_DUAL_H
#define COLIBRI_NATIVE_QUANT_DUAL_H

#include "tensor.h"

int coli_fp4_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *weight_a,
                             const ColiTensorView *weight_b,
                             const float *input);
int coli_fp8_dual_matvec_ref(float *output_a, float *output_b,
                             const ColiTensorView *weight_a,
                             const ColiTensorView *weight_b,
                             const float *input);

#endif
