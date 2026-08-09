#ifndef COLIBRI_NATIVE_QUANT_FP4_ROWS16_H
#define COLIBRI_NATIVE_QUANT_FP4_ROWS16_H

#include "native_quant.h"

/* Targets with a rows16 fast kernel: the matvec entry points below return 0
 * and the engine may pack hot experts into the 16-row interleaved layout.
 * Elsewhere they return -1 and callers keep the flat reference path. */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__aarch64__)
#define COLI_FP4_ROWS16_KERNEL 1
#endif

int coli_fp4_pack_rows16_v10(unsigned char *packed_data,
                             unsigned char *packed_scales,
                             const ColiTensorView *source);

int coli_fp4_matvec_rows16_v10(float *output,
                               const ColiTensorView *weight,
                               const float *input);

int coli_fp4_dual_matvec_rows16_v10(float *output_a, float *output_b,
                                    const ColiTensorView *a,
                                    const ColiTensorView *b,
                                    const float *input);

#endif
