#include "glm53_indexer.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

int coli_glm53_index_select(int *output, const float *queries,
                            const float *keys, const float *gate_scores,
                            const float *head_weights, const float *ape,
                            const unsigned char *valid, int sequence,
                            int heads, int dimension, int pool, int topk) {
    if (!output || !queries || !keys || !gate_scores || !head_weights ||
        !ape || !valid || sequence < 1 || heads < 1 || dimension < 1 ||
        pool < 1 || topk < pool || topk % pool) return -1;
    int width = topk + pool - 1;
    int pools = (sequence + pool - 1) / pool;
    int selected_pools = topk / pool;
    float *pooled = calloc((size_t)pools * dimension, sizeof(*pooled));
    float *scores = malloc((size_t)pools * sizeof(*scores));
    unsigned char *complete = calloc((size_t)pools, 1);
    unsigned char *picked = calloc((size_t)pools, 1);
    if (!pooled || !scores || !complete || !picked) {
        free(pooled); free(scores); free(complete); free(picked);
        return -1;
    }
    int first = 0;
    while (first < sequence && !valid[first]) first++;
    for (int p = 0; p < pools; p++) {
        int start = first + p * pool;
        complete[p] = start + pool <= sequence;
        for (int j = 0; complete[p] && j < pool; j++)
            if (!valid[start + j]) complete[p] = 0;
        if (!complete[p]) continue;
        for (int d = 0; d < dimension; d++) {
            float maximum = -FLT_MAX;
            for (int j = 0; j < pool; j++) {
                float value = gate_scores[(size_t)(start + j) * dimension + d] +
                              ape[(size_t)j * dimension + d];
                if (value > maximum) maximum = value;
            }
            float total = 0.0f;
            for (int j = 0; j < pool; j++)
                total += expf(gate_scores[(size_t)(start + j) * dimension + d] +
                              ape[(size_t)j * dimension + d] - maximum);
            for (int j = 0; j < pool; j++) {
                float probability = expf(
                    gate_scores[(size_t)(start + j) * dimension + d] +
                    ape[(size_t)j * dimension + d] - maximum) / total;
                pooled[(size_t)p * dimension + d] +=
                    probability * keys[(size_t)(start + j) * dimension + d];
            }
        }
    }
    float query_scale = 1.0f / sqrtf((float)dimension);
    for (int q = 0; q < sequence; q++) {
        int *row = output + (size_t)q * width;
        for (int i = 0; i < width; i++) row[i] = -1;
        if (!valid[q]) continue;
        for (int p = 0; p < pools; p++) {
            int end = first + (p + 1) * pool - 1;
            if (!complete[p] || end > q) {
                scores[p] = -FLT_MAX;
                continue;
            }
            float score = 0.0f;
            for (int h = 0; h < heads; h++) {
                float dot = 0.0f;
                const float *query = queries + ((size_t)q * heads + h) * dimension;
                for (int d = 0; d < dimension; d++)
                    dot += query[d] * pooled[(size_t)p * dimension + d];
                if (dot > 0.0f)
                    score += head_weights[(size_t)q * heads + h] * dot * query_scale;
            }
            scores[p] = score;
        }
        for (int rank = 0; rank < selected_pools; rank++) {
            int best = -1;
            for (int p = 0; p < pools; p++)
                if (!picked[p] && scores[p] > -FLT_MAX &&
                    (best < 0 || scores[p] > scores[best])) best = p;
            if (best < 0) break;
            picked[best] = 1;
            for (int j = 0; j < pool; j++) row[rank * pool + j] = first + best * pool + j;
        }
        for (int p = 0; p < pools; p++) picked[p] = 0;
        int visible = 0;
        for (int i = first; i <= q; i++) if (valid[i]) visible++;
        int tail = visible % pool;
        int tail_start = first + visible - tail;
        for (int j = 0; j < tail && j < pool - 1; j++)
            if (tail_start + j <= q && valid[tail_start + j])
                row[topk + j] = tail_start + j;
    }
    free(picked); free(complete); free(scores); free(pooled);
    return 0;
}
