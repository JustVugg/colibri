#include "gemma4_sampling.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    const float logits[] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
    coli_gemma4_sampler left, right;
    uint32_t token, other, index;
    float probability;
    if (coli_gemma4_sampler_init(&left, 0.0F, 0, 1.0F, 7) != 0 ||
        coli_gemma4_sample(&left, logits, 5, &token, &probability) != 0 ||
        token != 4 || probability != 1.0F) return 1;
    if (coli_gemma4_sampler_init(&left, 1.0F, 1, 1.0F, 7) != 0 ||
        coli_gemma4_sample(&left, logits, 5, &token, &probability) != 0 ||
        token != 4 || probability != 1.0F) return 1;
    if (coli_gemma4_sampler_init(&left, 1.0F, 0, 0.1F, 7) != 0 ||
        coli_gemma4_sample(&left, logits, 5, &token, &probability) != 0 ||
        token != 4 || probability != 1.0F) return 1;
    if (coli_gemma4_sampler_init(&left, 0.8F, 2, 0.95F, 42) != 0 ||
        coli_gemma4_sampler_init(&right, 0.8F, 2, 0.95F, 42) != 0) return 1;
    for (index = 0; index < 64; ++index) {
        if (coli_gemma4_sample(&left, logits, 5, &token, NULL) != 0 ||
            coli_gemma4_sample(&right, logits, 5, &other, NULL) != 0 ||
            token != other || token < 3) return 1;
    }
    if (coli_gemma4_sampler_init(&left, -1.0F, 0, 1.0F, 1) == 0 ||
        coli_gemma4_sampler_init(&left, 1.0F, 0, NAN, 1) == 0) return 1;
    puts("Gemma 4 sampling tests passed");
    return 0;
}
