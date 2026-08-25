#include "gemma4_sampling.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t token;
    float logit;
    double weight;
} sampling_candidate;

static int compare_candidates(const void *left, const void *right) {
    const sampling_candidate *a = (const sampling_candidate *)left;
    const sampling_candidate *b = (const sampling_candidate *)right;
    if (a->logit < b->logit) return 1;
    if (a->logit > b->logit) return -1;
    if (a->token > b->token) return 1;
    if (a->token < b->token) return -1;
    return 0;
}

static uint64_t sampler_random(coli_gemma4_sampler *sampler) {
    uint64_t value = sampler->state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    sampler->state = value;
    return value * UINT64_C(2685821657736338717);
}

static double sampler_uniform(coli_gemma4_sampler *sampler) {
    return (double)(sampler_random(sampler) >> 11) *
           (1.0 / 9007199254740992.0);
}

int coli_gemma4_sampler_init(coli_gemma4_sampler *sampler,
                             float temperature, uint32_t top_k,
                             float top_p, uint64_t seed) {
    if (!sampler || !isfinite(temperature) || temperature < 0.0F ||
        !isfinite(top_p) || top_p <= 0.0F || top_p > 1.0F) return -1;
    memset(sampler, 0, sizeof(*sampler));
    sampler->temperature = temperature;
    sampler->top_k = top_k;
    sampler->top_p = top_p;
    sampler->state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
    return 0;
}

int coli_gemma4_sample(coli_gemma4_sampler *sampler, const float *logits,
                       size_t count, uint32_t *token, float *probability) {
    sampling_candidate *candidates;
    size_t index, candidate_count = 0, retained;
    double total = 0.0, threshold, cumulative = 0.0;
    if (!sampler || !logits || !token || !count || count > UINT32_MAX)
        return -1;
    if (sampler->temperature == 0.0F) {
        uint32_t best = UINT32_MAX;
        float best_logit = 0.0F;
        for (index = 0; index < count; ++index) {
            if (isfinite(logits[index]) &&
                (best == UINT32_MAX || logits[index] > best_logit)) {
                best = (uint32_t)index;
                best_logit = logits[index];
            }
        }
        if (best == UINT32_MAX) return -1;
        *token = best;
        if (probability) *probability = 1.0F;
        return 0;
    }
    if (count > SIZE_MAX / sizeof(*candidates)) return -1;
    candidates = (sampling_candidate *)malloc(count * sizeof(*candidates));
    if (!candidates) return -1;
    for (index = 0; index < count; ++index) {
        if (isfinite(logits[index])) {
            candidates[candidate_count].token = (uint32_t)index;
            candidates[candidate_count].logit = logits[index];
            candidates[candidate_count].weight = 0.0;
            ++candidate_count;
        }
    }
    if (!candidate_count) {
        free(candidates);
        return -1;
    }
    qsort(candidates, candidate_count, sizeof(*candidates), compare_candidates);
    retained = sampler->top_k && sampler->top_k < candidate_count ?
        sampler->top_k : candidate_count;
    for (index = 0; index < retained; ++index) {
        candidates[index].weight = exp(
            ((double)candidates[index].logit - candidates[0].logit) /
            sampler->temperature);
        total += candidates[index].weight;
    }
    if (!isfinite(total) || total <= 0.0) {
        free(candidates);
        return -1;
    }
    cumulative = 0.0;
    for (index = 0; index < retained; ++index) {
        cumulative += candidates[index].weight / total;
        if (cumulative >= sampler->top_p) {
            retained = index + 1;
            break;
        }
    }
    total = 0.0;
    for (index = 0; index < retained; ++index)
        total += candidates[index].weight;
    threshold = sampler_uniform(sampler) * total;
    cumulative = 0.0;
    for (index = 0; index < retained; ++index) {
        cumulative += candidates[index].weight;
        if (threshold < cumulative || index + 1 == retained) {
            *token = candidates[index].token;
            if (probability)
                *probability = (float)(candidates[index].weight / total);
            free(candidates);
            return 0;
        }
    }
    free(candidates);
    return -1;
}
