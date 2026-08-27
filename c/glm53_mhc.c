#include "glm53_mhc.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static float sigmoid_stable(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    float growth = expf(value);
    return growth / (1.0f + growth);
}

int coli_glm53_mhc_pre(float *collapsed, float *post, float *comb,
                       const float *streams, const float *function,
                       const float scale[3], const float *base,
                       int copies, int dimension, int iterations,
                       float norm_eps, float hc_eps) {
    if (!collapsed || !post || !comb || !streams || !function || !scale ||
        !base || copies < 1 || dimension < 1 || iterations < 1) return -1;
    int flattened = copies * dimension;
    int count = (2 + copies) * copies;
    float *mix = malloc((size_t)count * sizeof(*mix));
    float *pre = malloc((size_t)copies * sizeof(*pre));
    if (!mix || !pre) { free(mix); free(pre); return -1; }
    float square = 0.0f;
    for (int i = 0; i < flattened; i++) square += streams[i] * streams[i];
    float inverse = 1.0f / sqrtf(square / flattened + norm_eps);
    for (int row = 0; row < count; row++) {
        float sum = 0.0f;
        for (int column = 0; column < flattened; column++)
            sum += function[(size_t)row * flattened + column] * streams[column];
        mix[row] = sum * inverse;
    }
    for (int i = 0; i < copies; i++) {
        pre[i] = sigmoid_stable(mix[i] * scale[0] + base[i]) + hc_eps;
        post[i] = 2.0f * sigmoid_stable(
            mix[copies + i] * scale[1] + base[copies + i]);
    }
    int matrix = 2 * copies;
    for (int row = 0; row < copies; row++) {
        float maximum = -INFINITY, total = 0.0f;
        for (int column = 0; column < copies; column++) {
            int index = matrix + row * copies + column;
            comb[row * copies + column] = mix[index] * scale[2] + base[index];
            if (comb[row * copies + column] > maximum)
                maximum = comb[row * copies + column];
        }
        for (int column = 0; column < copies; column++) {
            float value = expf(comb[row * copies + column] - maximum);
            comb[row * copies + column] = value;
            total += value;
        }
        for (int column = 0; column < copies; column++)
            comb[row * copies + column] =
                comb[row * copies + column] / total + hc_eps;
    }
    for (int iteration = 0; iteration < iterations; iteration++) {
        if (iteration) for (int row = 0; row < copies; row++) {
            float total = 0.0f;
            for (int column = 0; column < copies; column++)
                total += comb[row * copies + column];
            for (int column = 0; column < copies; column++)
                comb[row * copies + column] /= total + hc_eps;
        }
        for (int column = 0; column < copies; column++) {
            float total = 0.0f;
            for (int row = 0; row < copies; row++)
                total += comb[row * copies + column];
            for (int row = 0; row < copies; row++)
                comb[row * copies + column] /= total + hc_eps;
        }
    }
    for (int column = 0; column < dimension; column++) {
        float sum = 0.0f;
        for (int copy = 0; copy < copies; copy++)
            sum += pre[copy] * streams[copy * dimension + column];
        collapsed[column] = sum;
    }
    free(pre); free(mix);
    return 0;
}

int coli_glm53_mhc_post(float *output, const float *branch,
                        const float *streams, const float *post,
                        const float *comb, int copies, int dimension) {
    if (!output || !branch || !streams || !post || !comb ||
        copies < 1 || dimension < 1) return -1;
    for (int destination = 0; destination < copies; destination++)
        for (int column = 0; column < dimension; column++) {
            float value = post[destination] * branch[column];
            for (int source = 0; source < copies; source++)
                value += comb[source * copies + destination] *
                         streams[source * dimension + column];
            output[destination * dimension + column] = value;
        }
    return 0;
}
