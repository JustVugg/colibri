#include "gemma4_cuda.h"

#include <stdio.h>

int coli_gemma4_cuda_create(void **context, int device,
                            char *error, size_t error_capacity) {
    (void)device;
    if (context) *context = NULL;
    if (error && error_capacity)
        snprintf(error, error_capacity, "Gemma CUDA backend was not built");
    return -1;
}

void coli_gemma4_cuda_destroy(void *context) { (void)context; }

int coli_gemma4_cuda_run(void *context, const uint8_t *payload,
                         size_t payload_bytes, size_t gate_bytes,
                         size_t up_bytes, uint32_t model_width,
                         uint32_t expert_width, const float *input,
                         float scale, float *output,
                         char *error, size_t error_capacity) {
    (void)context; (void)payload; (void)payload_bytes; (void)gate_bytes;
    (void)up_bytes; (void)model_width; (void)expert_width; (void)input;
    (void)scale; (void)output;
    if (error && error_capacity)
        snprintf(error, error_capacity, "Gemma CUDA backend was not built");
    return -1;
}
