#include "glm53_kda.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static float silu(float value) { return value / (1.0f + expf(-value)); }

int coli_glm53_kda_step(float *output, float *state, float *conv_window,
                        const float *qkv, const float *conv_weight,
                        const float *log_decay, const float *beta,
                        int heads, int dim, int kernel) {
    if (!output || !state || !conv_window || !qkv || !conv_weight ||
        !log_decay || !beta || heads < 1 || dim < 1 || kernel < 2)
        return -1;
    size_t projection = (size_t)heads * (size_t)dim;
    float *mixed = malloc(3 * projection * sizeof(*mixed));
    float *key_memory = malloc((size_t)dim * sizeof(*key_memory));
    if (!mixed || !key_memory) {
        free(mixed);
        free(key_memory);
        return -1;
    }

    for (size_t channel = 0; channel < 3 * projection; channel++) {
        float *window = conv_window + channel * (size_t)kernel;
        memmove(window, window + 1, (size_t)(kernel - 1) * sizeof(*window));
        window[kernel - 1] = qkv[channel];
        const float *weight = conv_weight + channel * (size_t)kernel;
        float sum = 0.0f;
        for (int tap = 0; tap < kernel; tap++) sum += weight[tap] * window[tap];
        mixed[channel] = silu(sum);
    }

    const float scale = 1.0f / sqrtf((float)dim);
    for (int head = 0; head < heads; head++) {
        float *matrix = state + (size_t)head * dim * dim;
        const float *query = mixed + (size_t)head * dim;
        const float *key = mixed + projection + (size_t)head * dim;
        const float *value = mixed + 2 * projection + (size_t)head * dim;
        float *head_output = output + (size_t)head * dim;
        float qnorm = 1.0e-6f, knorm = 1.0e-6f;
        for (int i = 0; i < dim; i++) {
            qnorm += query[i] * query[i];
            knorm += key[i] * key[i];
        }
        qnorm = scale / sqrtf(qnorm);
        knorm = 1.0f / sqrtf(knorm);
        memset(key_memory, 0, (size_t)dim * sizeof(*key_memory));
        for (int k = 0; k < dim; k++) {
            float *row = matrix + (size_t)k * dim;
            float decay = expf(log_decay[(size_t)head * dim + k]);
            float normalized_key = key[k] * knorm;
            for (int v = 0; v < dim; v++) {
                row[v] *= decay;
                key_memory[v] += normalized_key * row[v];
            }
        }
        memset(head_output, 0, (size_t)dim * sizeof(*head_output));
        for (int k = 0; k < dim; k++) {
            float *row = matrix + (size_t)k * dim;
            float normalized_key = key[k] * knorm;
            float normalized_query = query[k] * qnorm;
            for (int v = 0; v < dim; v++) {
                float delta = (value[v] - key_memory[v]) * beta[head];
                row[v] += normalized_key * delta;
                head_output[v] += normalized_query * row[v];
            }
        }
    }
    free(key_memory);
    free(mixed);
    return 0;
}
