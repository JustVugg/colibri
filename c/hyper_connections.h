#ifndef COLIBRI_HYPER_CONNECTIONS_H
#define COLIBRI_HYPER_CONNECTIONS_H

/*
 * Manifold-Constrained Hyper-Connections (mHC), shared by every engine whose
 * checkpoint carries them.
 *
 * mHC replaces the plain residual with `hc_mult` parallel residual streams. At
 * each site a learned map turns the incoming streams into three things:
 *
 *   pre   [H]      how to collapse the H streams into the block's input
 *   post  [H]      where in the streams the block's output is written back
 *   comb  [H,H]    how the streams mix with each other
 *
 * `comb` is projected onto the doubly-stochastic manifold by Sinkhorn
 * normalization, which is the "manifold-constrained" part: the mixing matrix
 * keeps rows and columns summing to one, so the residual pathway cannot
 * silently gain or lose scale layer after layer.
 *
 * The functions here are the arithmetic only. Weight layout, per-layer wiring
 * and any batched or accelerated variant stay with the engine.
 *
 * Two checkpoints use this today and agree down to the config key names
 * (hc_mult, hc_eps, hc_sinkhorn_iters): DeepSeek V4 and GLM-5.3-Flash. This
 * header exists so they share the arithmetic rather than each carrying a copy —
 * copies of shared mechanisms drifting apart is the defect class that keeps
 * recurring in this tree (#720/#748 being the clearest example).
 */

#include <math.h>
#include <stdlib.h>

static inline float coli_hc_sigmoid(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

/* Split the mix logits into pre/post/comb and project comb onto the
 * doubly-stochastic manifold. `mixes` is [(2+hc)*hc], `scale` is the three
 * learned scales, `base` the matching biases. Returns 0, or -1 on bad
 * arguments or allocation failure. */
static inline int coli_hc_split_sinkhorn(float *pre, float *post, float *comb,
                                         const float *mixes, const float scale[3],
                                         const float *base, int hc, int iterations,
                                         float eps) {
    if (!pre || !post || !comb || !mixes || !scale || !base ||
        hc < 1 || iterations < 1 || eps < 0.0f)
        return -1;
    for (int index = 0; index < hc; index++) {
        pre[index] = coli_hc_sigmoid(
            mixes[index] * scale[0] + base[index]) + eps;
        post[index] = 2.0f * coli_hc_sigmoid(
            mixes[hc + index] * scale[1] + base[hc + index]);
    }
    int matrix_offset = 2 * hc;
    for (int row = 0; row < hc; row++) {
        float maximum = -INFINITY;
        for (int column = 0; column < hc; column++) {
            int index = matrix_offset + row * hc + column;
            float value = mixes[index] * scale[2] + base[index];
            comb[row * hc + column] = value;
            if (value > maximum) maximum = value;
        }
        float sum = 0.0f;
        for (int column = 0; column < hc; column++) {
            float value = expf(comb[row * hc + column] - maximum);
            comb[row * hc + column] = value;
            sum += value;
        }
        for (int column = 0; column < hc; column++)
            comb[row * hc + column] = comb[row * hc + column] / sum + eps;
    }
    float *sums = malloc((size_t)hc * sizeof(*sums));
    if (!sums) return -1;
    for (int column = 0; column < hc; column++) {
        float sum = 0.0f;
        for (int row = 0; row < hc; row++)
            sum += comb[row * hc + column];
        sums[column] = sum;
    }
    for (int row = 0; row < hc; row++)
        for (int column = 0; column < hc; column++)
            comb[row * hc + column] /= sums[column] + eps;

    for (int iteration = 1; iteration < iterations; iteration++) {
        for (int row = 0; row < hc; row++) {
            float sum = 0.0f;
            for (int column = 0; column < hc; column++)
                sum += comb[row * hc + column];
            sums[row] = sum;
        }
        for (int row = 0; row < hc; row++)
            for (int column = 0; column < hc; column++)
                comb[row * hc + column] /= sums[row] + eps;
        for (int column = 0; column < hc; column++) {
            float sum = 0.0f;
            for (int row = 0; row < hc; row++)
                sum += comb[row * hc + column];
            sums[column] = sum;
        }
        for (int row = 0; row < hc; row++)
            for (int column = 0; column < hc; column++)
                comb[row * hc + column] /= sums[column] + eps;
    }
    free(sums);
    return 0;
}

/* Entering a block: RMS-rescale the flattened streams, map them to the mix
 * logits through `hc_fn` [(2+hc)*hc, hc*dimension], split, and collapse the
 * streams into `output` [dimension] with the pre weights. `post` and `comb`
 * are handed back for the matching coli_hc_post(). */
static inline int coli_hc_pre(float *output, float *post, float *comb,
                              const float *input, const float *hc_fn,
                              const float scale[3], const float *base,
                              int hc, int dimension, int iterations,
                              float norm_eps, float hc_eps) {
    if (!output || !post || !comb || !input || !hc_fn || !scale || !base ||
        hc < 1 || dimension < 1 || norm_eps < 0.0f)
        return -1;
    int flattened = hc * dimension;
    int mix_count = (2 + hc) * hc;
    float mean_square = 0.0f;
    for (int index = 0; index < flattened; index++)
        mean_square += input[index] * input[index];
    float inverse_rms = 1.0f / sqrtf(mean_square / flattened + norm_eps);
    float *mixes = malloc((size_t)mix_count * sizeof(*mixes));
    float *pre = malloc((size_t)hc * sizeof(*pre));
    if (!mixes || !pre) {
        free(mixes);
        free(pre);
        return -1;
    }
    for (int row = 0; row < mix_count; row++) {
        float sum = 0.0f;
        for (int column = 0; column < flattened; column++)
            sum += hc_fn[(size_t)row * flattened + column] * input[column];
        mixes[row] = sum * inverse_rms;
    }
    if (coli_hc_split_sinkhorn(pre, post, comb, mixes, scale, base,
                               hc, iterations, hc_eps) != 0) {
        free(pre);
        free(mixes);
        return -1;
    }
    for (int column = 0; column < dimension; column++) {
        float sum = 0.0f;
        for (int copy = 0; copy < hc; copy++)
            sum += pre[copy] * input[copy * dimension + column];
        output[column] = sum;
    }
    free(pre);
    free(mixes);
    return 0;
}

/* Leaving a block: mix the incoming streams through `comb` and add the block's
 * output where `post` says it goes. `output` and `residual` are [hc*dimension]
 * and may not overlap. */
static inline int coli_hc_post(float *output, const float *branch,
                               const float *residual, const float *post,
                               const float *comb, int hc, int dimension) {
    if (!output || !branch || !residual || !post || !comb ||
        hc < 1 || dimension < 1)
        return -1;
    for (int destination = 0; destination < hc; destination++) {
        for (int column = 0; column < dimension; column++) {
            float value = 0.0f;
            for (int source = 0; source < hc; source++)
                value += comb[source * hc + destination] *
                         residual[source * dimension + column];
            value += post[destination] * branch[column];
            output[destination * dimension + column] = value;
        }
    }
    return 0;
}

#endif /* COLIBRI_HYPER_CONNECTIONS_H */
