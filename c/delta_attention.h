#ifndef COLIBRI_DELTA_ATTENTION_H
#define COLIBRI_DELTA_ATTENTION_H

/*
 * Kimi Delta Attention (KDA): the gated delta rule that the linear-attention
 * layers of Kimi K3, Qwen3.6 and GLM-5.3-Flash all run, differing only in the
 * dimensions and in how the caller produces the gate.
 *
 * Per token and per head, with S the [k_dim x v_dim] recurrent state:
 *
 *     q,k,v  = SiLU(ShortConv(W{q,k,v} x))     short causal convolution first
 *     q,k    = q/sqrt(|q|^2 + eps)             L2 norm with eps INSIDE the sqrt
 *     q     *= k_dim^-0.5
 *     S      = Diag(alpha) S                   alpha = exp(gate), per k channel
 *     delta  = (v - S^T k) * beta              what the memory got wrong
 *     S      = S + k delta^T                   write the correction back
 *     out    = S^T q
 *
 * The two details worth stating, because both are silently wrong-looking-right
 * if you get them backwards: the eps sits inside the square root (FLA's
 * convention, which the reference notes is deliberately unlike F.normalize),
 * and `delta` is computed against the state AFTER the decay but BEFORE the
 * write, so the decay cannot be folded into the second pass.
 *
 * `gate` and `beta` arrive already resolved. Every model builds them
 * differently - GLM-5.3 uses gate_lower_bound * sigmoid(exp(A_log) * (W_fb W_fa
 * x + dt_bias)) - and that belongs to the engine, not here.
 *
 * No allocation: the caller owns the scratch. A per-token malloc costs more
 * than the arithmetic at these sizes, which is the lesson #1179 paid for on the
 * DeepSeek V4 indexer.
 */

#include <math.h>
#include <string.h>

/* Scratch needed by coli_kda_step, in floats. */
static inline int coli_kda_scratch_floats(int heads, int k_dim, int v_dim) {
    int mixed = 3 * heads * (k_dim > v_dim ? k_dim : v_dim);
    return mixed + v_dim;
}

static inline float coli_kda_silu(float value) {
    return value / (1.0f + expf(-value));
}

/* One token through one KDA layer.
 *
 *   out     [heads * v_dim]        written
 *   state   [heads * k_dim * v_dim] read and updated in place
 *   window  [3 * heads * k_dim * kernel] convolution history, updated in place
 *   qkv     [3 * heads * k_dim]    this token's q, k and v projections
 *   conv_w  [3 * heads * k_dim * kernel] depthwise taps, oldest first
 *   gate    [heads * k_dim]        log-decay, already gated by the engine
 *   beta    [heads]                already through its sigmoid
 *   scratch [coli_kda_scratch_floats()]
 *
 * k_dim and v_dim are equal in every model shipping today; they are separate
 * arguments because the recurrence does not require them to be. */
static inline int coli_kda_step(float *out, float *state, float *window,
                                const float *qkv, const float *conv_w,
                                const float *gate, const float *beta,
                                int heads, int k_dim, int v_dim, int kernel,
                                float norm_eps, float *scratch) {
    if (!out || !state || !window || !qkv || !conv_w || !gate || !beta ||
        !scratch || heads < 1 || k_dim < 1 || v_dim < 1 || kernel < 1)
        return -1;

    const int width = heads * k_dim;
    float *mixed = scratch;                    /* 3 * width */
    float *memory = scratch + 3 * width;       /* v_dim */

    /* Short causal convolution over each channel's own history, then SiLU. */
    for (int channel = 0; channel < 3 * width; channel++) {
        float *history = window + (size_t)channel * kernel;
        memmove(history, history + 1, (size_t)(kernel - 1) * sizeof(*history));
        history[kernel - 1] = qkv[channel];
        const float *taps = conv_w + (size_t)channel * kernel;
        float sum = 0.0f;
        for (int tap = 0; tap < kernel; tap++) sum += taps[tap] * history[tap];
        mixed[channel] = coli_kda_silu(sum);
    }

    const float query_scale = 1.0f / sqrtf((float)k_dim);
    for (int head = 0; head < heads; head++) {
        float *matrix = state + (size_t)head * k_dim * v_dim;
        const float *query = mixed + (size_t)head * k_dim;
        const float *key = mixed + width + (size_t)head * k_dim;
        const float *value = mixed + 2 * width + (size_t)head * v_dim;
        const float *decay = gate + (size_t)head * k_dim;
        float *result = out + (size_t)head * v_dim;

        float query_square = norm_eps, key_square = norm_eps;
        for (int i = 0; i < k_dim; i++) {
            query_square += query[i] * query[i];
            key_square += key[i] * key[i];
        }
        const float query_norm = query_scale / sqrtf(query_square);
        const float key_norm = 1.0f / sqrtf(key_square);

        /* Pass one: decay the state and read what it currently predicts for
         * this key. Both must happen before anything is written back. */
        memset(memory, 0, (size_t)v_dim * sizeof(*memory));
        for (int k = 0; k < k_dim; k++) {
            float *row = matrix + (size_t)k * v_dim;
            const float alpha = expf(decay[k]);
            const float scaled_key = key[k] * key_norm;
            for (int v = 0; v < v_dim; v++) {
                row[v] *= alpha;
                memory[v] += scaled_key * row[v];
            }
        }
        /* Pass two: write the correction and read the answer out. */
        memset(result, 0, (size_t)v_dim * sizeof(*result));
        for (int k = 0; k < k_dim; k++) {
            float *row = matrix + (size_t)k * v_dim;
            const float scaled_key = key[k] * key_norm;
            const float scaled_query = query[k] * query_norm;
            for (int v = 0; v < v_dim; v++) {
                row[v] += scaled_key * (value[v] - memory[v]) * beta[head];
                result[v] += scaled_query * row[v];
            }
        }
    }
    return 0;
}

#endif /* COLIBRI_DELTA_ATTENTION_H */
