#ifndef COLI_GEMMA4_CUDA_H
#define COLI_GEMMA4_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int coli_gemma4_cuda_create(void **context, int device,
                            char *error, size_t error_capacity);
void coli_gemma4_cuda_destroy(void *context);
int coli_gemma4_cuda_run(void *context, const uint8_t *payload,
                         size_t payload_bytes, size_t gate_bytes,
                         size_t up_bytes, uint32_t model_width,
                         uint32_t expert_width, const float *input,
                         float scale, float *output,
                         char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
