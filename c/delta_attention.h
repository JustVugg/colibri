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
#ifdef _OPENMP
#include <omp.h>
#endif

/* Scratch needed by coli_kda_step, in floats.
 *
 * The `memory` accumulator is per-thread: the head loop below runs in
 * parallel and each head needs its own, so the tail is v_dim per thread
 * rather than one v_dim. Sized from omp_get_max_threads() at call time,
 * which is what the head loop's omp_get_thread_num() indexes into. */
static inline int coli_kda_threads(void) {
#ifdef _OPENMP
    int t = omp_get_max_threads();
    return t > 0 ? t : 1;
#else
    return 1;
#endif
}

static inline int coli_kda_scratch_floats(int heads, int k_dim, int v_dim) {
    int mixed = 3 * heads * (k_dim > v_dim ? k_dim : v_dim);
    return mixed + coli_kda_threads() * v_dim;
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
 * arguments because the recurrence does not require them to be.
 *
 * Two conditions on `scratch`, both satisfied by the one caller today
 * (glm53.c, which sizes it at session open and calls from serial code):
 * it must have been sized by coli_kda_scratch_floats() under the same
 * OpenMP thread count this call will see, and this function must NOT be
 * called from inside a parallel region -- the head loop indexes the scratch
 * tail by omp_get_thread_num(), so an outer team's larger ids would run off
 * the end of the buffer. */
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
    float *memory_pool = scratch + 3 * width;  /* v_dim PER THREAD */

    /* Short causal convolution over each channel's own history, then SiLU.
     * Channels are independent: each owns its slice of `window` and one
     * element of `mixed`, and reads `qkv`/`conv_w` without writing them.
     * This loop carries 3*width SiLU calls -- 24,576 expf per call at GLM's
     * shape -- which is why it is worth a team of its own: parallelising only
     * the head loop below leaves coli_kda_step flat at 1.03 ms/call, because
     * this loop is then the serial half. Both need a team. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
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
    /* Heads are independent: each owns its own slice of `state` and of `out`,
     * and reads `mixed` without writing it. The one thing they would share is
     * the `memory` accumulator, so each thread takes its own slice of the
     * pool -- sharing it corrupts every head's read-back while still emitting
     * plausible tokens -- a teacher-forcing oracle catches it, reading the
     * output does not. Measured before parallelising: 2.25 ms/call on one of
     * eight cores. No num_threads() cap here: capping a single kernel's team
     * inside a larger pool was measured on another engine and rejected, since
     * the idle pool threads spin on the sibling hyperthreads. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int head = 0; head < heads; head++) {
#ifdef _OPENMP
        float *memory = memory_pool + (size_t)omp_get_thread_num() * v_dim;
#else
        float *memory = memory_pool;
#endif
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
