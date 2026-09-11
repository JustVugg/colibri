#ifndef COLI_GEMMA4_SAMPLING_H
#define COLI_GEMMA4_SAMPLING_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float temperature;
    float top_p;
    uint32_t top_k;
    uint64_t state;
} coli_gemma4_sampler;

int coli_gemma4_sampler_init(coli_gemma4_sampler *sampler,
                             float temperature, uint32_t top_k,
                             float top_p, uint64_t seed);
int coli_gemma4_sample(coli_gemma4_sampler *sampler, const float *logits,
                       size_t count, uint32_t *token, float *probability);

#endif
