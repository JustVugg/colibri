#include "glm53_sparse_attention.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

int coli_glm53_sparse_attention(float *output, const float *queries,
                                const float *keys, const float *values,
                                const int *indices, int sequence, int width,
                                int heads, int key_dim, int value_dim) {
    if (!output || !queries || !keys || !values || !indices || sequence < 1 ||
        width < 1 || heads < 1 || key_dim < 1 || value_dim < 1) return -1;
    float *scores = malloc((size_t)width * sizeof(*scores));
    if (!scores) return -1;
    float scale = 1.0f / sqrtf((float)key_dim);
    for (int query_position = 0; query_position < sequence; query_position++) {
        const int *selected = indices + (size_t)query_position * width;
        for (int head = 0; head < heads; head++) {
            const float *query = queries +
                ((size_t)query_position * heads + head) * key_dim;
            float maximum = -FLT_MAX;
            for (int slot = 0; slot < width; slot++) {
                int key_position = selected[slot];
                if (key_position < 0 || key_position >= sequence) {
                    scores[slot] = -FLT_MAX;
                    continue;
                }
                const float *key = keys +
                    ((size_t)key_position * heads + head) * key_dim;
                float score = 0.0f;
                for (int dim = 0; dim < key_dim; dim++) score += query[dim] * key[dim];
                scores[slot] = score * scale;
                if (scores[slot] > maximum) maximum = scores[slot];
            }
            float total = 0.0f;
            for (int slot = 0; slot < width; slot++) {
                if (scores[slot] == -FLT_MAX) continue;
                scores[slot] = expf(scores[slot] - maximum);
                total += scores[slot];
            }
            float *result = output +
                ((size_t)query_position * heads + head) * value_dim;
            for (int dim = 0; dim < value_dim; dim++) result[dim] = 0.0f;
            if (total == 0.0f) continue;
            for (int slot = 0; slot < width; slot++) {
                int value_position = selected[slot];
                if (value_position < 0 || value_position >= sequence) continue;
                float probability = scores[slot] / total;
                const float *value = values +
                    ((size_t)value_position * heads + head) * value_dim;
                for (int dim = 0; dim < value_dim; dim++)
                    result[dim] += probability * value[dim];
            }
        }
    }
    free(scores);
    return 0;
}
