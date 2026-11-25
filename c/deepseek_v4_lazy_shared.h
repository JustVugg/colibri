/* deepseek_v4_lazy_shared.h — V4_LAZY_SHARED deferral policy and ensure helper.
 *
 * When V4_LAZY_SHARED=1 (opt-in, default off) and the fused GPU MoE kernel
 * (dsv4_cuda_moe) can compute the shared expert directly from GPU-resident
 * sg/su/sd, the host shared-expert forward is deferred. Every fallback or
 * CPU path calls ensure before reading shared_output, so the host forward
 * still runs whenever GPU MoE did not produce the shared expert. */
#ifndef DEEPSEEK_V4_LAZY_SHARED_H
#define DEEPSEEK_V4_LAZY_SHARED_H

#include <stdlib.h>

static inline int coli_v4_lazy_shared_enabled(void) {
    const char *setting = getenv("V4_LAZY_SHARED");
    return setting && *setting && atoi(setting) != 0;
}

static inline int coli_v4_lazy_shared_should_defer(
    int lazy_enabled, int has_gpu, int hybrid_enabled, int hot_split,
    int has_gpu_w1, int has_gpu_w3, int has_gpu_w2) {
    return lazy_enabled && has_gpu && !hybrid_enabled && !hot_split &&
           has_gpu_w1 && has_gpu_w3 && has_gpu_w2;
}

typedef int (*coli_v4_shared_forward_fn)(void *ctx, float *shared_output);

typedef struct ColiV4LazyShared {
    int shared_here;
    int deferred;
    int ready;
    int host_calls;
} ColiV4LazyShared;

static inline void coli_v4_lazy_shared_init(
    ColiV4LazyShared *state, int shared_here, int deferred) {
    state->shared_here = shared_here;
    state->deferred = deferred;
    state->ready = 0;
    state->host_calls = 0;
}

static inline int coli_v4_lazy_shared_ensure(
    ColiV4LazyShared *state, int *result, float *shared_output,
    coli_v4_shared_forward_fn fn, void *ctx) {
    if (!*result && state->shared_here && !state->ready) {
        *result = fn(ctx, shared_output);
        state->ready = 1;
        state->host_calls++;
    }
    return *result;
}

#endif
