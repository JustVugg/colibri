#ifndef COLI_MODEL_H
#define COLI_MODEL_H

#include <stddef.h>
#include <stdint.h>

/*
 * Model graphs decide which experts run. Placement backends decide where the
 * encoded records live and how their math is executed. Keeping this boundary
 * explicit lets Gemma use Colibri's cache/prefetch policies without borrowing
 * GLM's attention, tokenizer, or routing semantics.
 */
typedef struct {
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_ff;
    uint32_t n_vocab;
    uint32_t sliding_window;
} coli_model_config;

typedef struct {
    void *ctx;
    int (*prepare_layer)(void *ctx, uint32_t layer,
                         const uint32_t *expert_ids, uint32_t count);
    int (*run_experts)(void *ctx, uint32_t layer,
                       const uint32_t *expert_ids, const float *weights,
                       uint32_t count, const float *input, float *output);
    void (*release_layer)(void *ctx, uint32_t layer);
    int (*prefetch_layer)(void *ctx, uint32_t layer,
                          const uint32_t *expert_ids, uint32_t count);
} coli_expert_backend;

typedef struct {
    int (*prefill)(void *model, const int32_t *tokens, uint32_t count);
    int (*decode)(void *model, int32_t token, float *logits);
} coli_model_ops;

#endif
