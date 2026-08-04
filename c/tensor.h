#ifndef COLIBRI_TENSOR_H
#define COLIBRI_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Storage/execution formats understood by a model backend.  A view describes
 * bytes owned by a model or ExpertStore; it never owns or frees those bytes. */
typedef enum {
    COLI_TENSOR_F32 = 0,
    COLI_TENSOR_Q8_ROW,
    COLI_TENSOR_Q4_ROW,
    COLI_TENSOR_Q2_ROW,
    COLI_TENSOR_FP8_E4M3_BLOCK,
    COLI_TENSOR_FP4_NATIVE_BLOCK,
    COLI_TENSOR_INT8_BLOCK,
    COLI_TENSOR_INT4_BLOCK
} ColiTensorFormat;

typedef enum {
    COLI_SCALE_NONE = 0,
    COLI_SCALE_F32,
    COLI_SCALE_UE8M0
} ColiScaleFormat;

typedef struct {
    ColiTensorFormat format;
    ColiScaleFormat scale_format;
    const void *data;
    const void *scales;
    size_t data_bytes;
    size_t scale_bytes;
    int64_t rows;
    int64_t columns;
    uint32_t block_rows;
    uint32_t block_columns;
} ColiTensorView;

#ifdef __cplusplus
}
#endif

#endif
