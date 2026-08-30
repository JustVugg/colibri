#ifndef COLIBRI_SPARSE_INDEX_H
#define COLIBRI_SPARSE_INDEX_H

/*
 * DeepSeek-style sparse attention with k-pooling, as GLM-5.3-Flash runs it.
 *
 * The plain lightning indexer scores every key and keeps the top ones. With
 * k-pooling the keys are first grouped into pools of `pool` tokens; the pools
 * are scored and selected, and the winners expand back into their token
 * indices. Selecting 512 pools of 4 instead of 2048 tokens is the same budget
 * with a quarter of the scoring, and the pooled key is a per-channel softmax
 * mixture of the pool's keys rather than an average, so a pool is not forced to
 * describe itself by its mean.
 *
 * Three rules decide which pools may be chosen, and all three matter:
 *
 *   - a pool is selectable only if it is COMPLETE (every member present) and
 *     its last token is causally visible to the query;
 *   - the incomplete tail - the tokens after the last complete pool - is
 *     appended unconditionally when the model asks for it, which is why the
 *     output is `topk + pool - 1` wide rather than `topk`;
 *   - ties go to the lower pool index, so selection is deterministic.
 *
 * The scores carry a ReLU. It is written here as `if (dot > 0)` before scaling,
 * which is the same function as the reference's relu(dot * scale): ReLU
 * commutes with a positive scale.
 *
 * Unselected slots are -1, and the attention below skips them. Callers must not
 * treat the width as a count.
 */

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Width of one query's index row: topk expanded tokens plus the tail. */
static inline int coli_sparse_index_width(int topk, int pool, int with_tail) {
    return with_tail ? topk + pool - 1 : topk;
}

/* Select the key blocks each query may attend.
 *
 * In the range form, `keys`, `gates` and `valid` cover the whole sequence,
 * because a query may attend anywhere behind it. `queries`, `head_w` and `out`
 * cover only [q_from, q_to) and are indexed from zero: during decode the whole
 * prefix is already in the cache and only the new token has a query, so making
 * the caller allocate prefix-length query arrays would be asking it to build
 * what it deliberately no longer computes.
 *
 *   out       [sequence * width]  token indices, -1 where unused
 *   queries   [sequence * heads * dim]
 *   keys      [sequence * dim]     one indexer key per token (shared by heads)
 *   gates     [sequence * dim]     pooling logits, per channel
 *   head_w    [sequence * heads]   already scaled by heads^-0.5 by the caller
 *   ape       [pool * dim]         per-position bias inside a pool
 *   valid     [sequence]           0 marks padding
 *
 * Returns 0, or -1 on bad arguments or allocation failure. */
static inline int coli_sparse_index_select_range(int *out, const float *queries,
                                                 const float *keys, const float *gates,
                                                 const float *head_w, const float *ape,
                                                 const unsigned char *valid,
                                                 int sequence, int heads, int dim,
                                                 int pool, int topk, int with_tail,
                                                 int q_from, int q_to) {
    if (!out || !queries || !keys || !gates || !head_w || !ape || !valid ||
        q_from < 0 || q_to > sequence || q_from > q_to ||
        sequence < 1 || heads < 1 || dim < 1 || pool < 1 || topk < pool || topk % pool)
        return -1;

    const int width = coli_sparse_index_width(topk, pool, with_tail);
    const int pools = (sequence + pool - 1) / pool;
    const int wanted = topk / pool;

    float *pooled = calloc((size_t)pools * dim, sizeof(*pooled));
    float *scores = malloc((size_t)pools * sizeof(*scores));
    unsigned char *complete = calloc((size_t)pools, 1);
    unsigned char *taken = calloc((size_t)pools, 1);
    if (!pooled || !scores || !complete || !taken) {
        free(taken); free(complete); free(scores); free(pooled);
        return -1;
    }

    int first = 0;
    while (first < sequence && !valid[first]) first++;

    /* Pool the keys: per channel, a softmax over the pool's members using the
     * gate logits plus the positional bias. */
    for (int p = 0; p < pools; p++) {
        const int start = first + p * pool;
        complete[p] = start + pool <= sequence;
        for (int j = 0; complete[p] && j < pool; j++)
            if (!valid[start + j]) complete[p] = 0;
        if (!complete[p]) continue;
        for (int d = 0; d < dim; d++) {
            float maximum = -FLT_MAX;
            for (int j = 0; j < pool; j++) {
                const float logit = gates[(size_t)(start + j) * dim + d] +
                                    ape[(size_t)j * dim + d];
                if (logit > maximum) maximum = logit;
            }
            float total = 0.0f;
            for (int j = 0; j < pool; j++)
                total += expf(gates[(size_t)(start + j) * dim + d] +
                              ape[(size_t)j * dim + d] - maximum);
            float mixed = 0.0f;
            for (int j = 0; j < pool; j++) {
                const float weight = expf(gates[(size_t)(start + j) * dim + d] +
                                          ape[(size_t)j * dim + d] - maximum) / total;
                mixed += weight * keys[(size_t)(start + j) * dim + d];
            }
            pooled[(size_t)p * dim + d] = mixed;
        }
    }

    const float scale = 1.0f / sqrtf((float)dim);
    for (int q = q_from; q < q_to; q++) {
        int *row = out + (size_t)(q - q_from) * width;
        for (int i = 0; i < width; i++) row[i] = -1;
        if (!valid[q]) continue;

        for (int p = 0; p < pools; p++) {
            const int last = first + (p + 1) * pool - 1;
            if (!complete[p] || last > q) { scores[p] = -FLT_MAX; continue; }
            float score = 0.0f;
            for (int h = 0; h < heads; h++) {
                const float *query = queries + ((size_t)(q - q_from) * heads + h) * dim;
                float dot = 0.0f;
                for (int d = 0; d < dim; d++) dot += query[d] * pooled[(size_t)p * dim + d];
                if (dot > 0.0f)                                  /* ReLU */
                    score += head_w[(size_t)(q - q_from) * heads + h] * dot * scale;
            }
            scores[p] = score;
        }

        for (int rank = 0; rank < wanted; rank++) {
            int best = -1;
            for (int p = 0; p < pools; p++)
                if (!taken[p] && scores[p] > -FLT_MAX &&
                    (best < 0 || scores[p] > scores[best])) best = p;
            if (best < 0) break;
            taken[best] = 1;
            for (int j = 0; j < pool; j++) row[rank * pool + j] = first + best * pool + j;
        }
        memset(taken, 0, (size_t)pools);

        if (with_tail) {
            int visible = 0;
            for (int i = first; i <= q; i++) if (valid[i]) visible++;
            const int tail = visible % pool;
            const int tail_start = first + visible - tail;
            for (int j = 0; j < tail && j < pool - 1; j++)
                if (tail_start + j <= q && valid[tail_start + j])
                    row[topk + j] = tail_start + j;
        }
    }

    free(taken);
    free(complete);
    free(scores);
    free(pooled);
    return 0;
}

/* Tutte le query, che e' il caso del prefill. */
static inline int coli_sparse_index_select(int *out, const float *queries,
                                           const float *keys, const float *gates,
                                           const float *head_w, const float *ape,
                                           const unsigned char *valid,
                                           int sequence, int heads, int dim,
                                           int pool, int topk, int with_tail) {
    return coli_sparse_index_select_range(out, queries, keys, gates, head_w, ape,
                                          valid, sequence, heads, dim, pool, topk,
                                          with_tail, 0, sequence);
}

/* Softmax attention restricted to the selected indices.
 *
 *   out      [sequence * heads * value_dim]
 *   indices  [sequence * width], -1 entries skipped
 *
 * Queries, keys and values are laid out [position][head][dim]. */
static inline int coli_sparse_attention_range(float *out, const float *queries,
                                             const float *keys, const float *values,
                                             const int *indices, int sequence, int width,
                                             int heads, int key_dim, int value_dim,
                                             int q_from, int q_to) {
    if (!out || !queries || !keys || !values || !indices || sequence < 1 ||
        width < 1 || heads < 1 || key_dim < 1 || value_dim < 1 ||
        q_from < 0 || q_to > sequence || q_from > q_to) return -1;
    float *scores = malloc((size_t)width * sizeof(*scores));
    if (!scores) return -1;
    const float scale = 1.0f / sqrtf((float)key_dim);
    for (int q = q_from; q < q_to; q++) {
        const int *selected = indices + (size_t)(q - q_from) * width;
        for (int h = 0; h < heads; h++) {
            const float *query = queries + ((size_t)(q - q_from) * heads + h) * key_dim;
            float maximum = -FLT_MAX;
            int used = 0;
            for (int slot = 0; slot < width; slot++) {
                const int key_position = selected[slot];
                if (key_position < 0 || key_position >= sequence) continue;
                const float *key = keys + ((size_t)key_position * heads + h) * key_dim;
                float dot = 0.0f;
                for (int d = 0; d < key_dim; d++) dot += query[d] * key[d];
                scores[used] = dot * scale;
                if (scores[used] > maximum) maximum = scores[used];
                used++;
            }
            float *result = out + ((size_t)(q - q_from) * heads + h) * value_dim;
            memset(result, 0, (size_t)value_dim * sizeof(*result));
            if (!used) continue;
            float total = 0.0f;
            for (int i = 0; i < used; i++) {
                scores[i] = expf(scores[i] - maximum);
                total += scores[i];
            }
            int index = 0;
            for (int slot = 0; slot < width; slot++) {
                const int key_position = selected[slot];
                if (key_position < 0 || key_position >= sequence) continue;
                const float weight = scores[index++] / total;
                const float *value = values + ((size_t)key_position * heads + h) * value_dim;
                for (int d = 0; d < value_dim; d++) result[d] += weight * value[d];
            }
        }
    }
    free(scores);
    return 0;
}

/* Tutte le query, che e' il caso del prefill. */
static inline int coli_sparse_attention(float *out, const float *queries,
                                        const float *keys, const float *values,
                                        const int *indices, int sequence, int width,
                                        int heads, int key_dim, int value_dim) {
    return coli_sparse_attention_range(out, queries, keys, values, indices, sequence,
                                       width, heads, key_dim, value_dim, 0, sequence);
}

#endif /* COLIBRI_SPARSE_INDEX_H */
