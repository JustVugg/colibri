#ifndef COLI_GEMMA4_MODEL_H
#define COLI_GEMMA4_MODEL_H

#include <stdint.h>

#include "gemma4_gguf.h"

typedef struct {
    const coli_gemma4_gguf *gguf;
    uint32_t layer;
    uint32_t width;
    uint32_t expert_count;
    uint32_t selected_count;
    float *input_scale;
    float *projection;
    float *expert_scale;
    char last_error[COLI_GEMMA4_GGUF_ERROR_MAX];
} coli_gemma4_router;

typedef struct {
    const coli_gemma4_gguf *gguf;
    const coli_gemma4_tensor *tensor;
    uint8_t *data;
    uint32_t input_width;
    uint32_t output_width;
} coli_gemma4_matrix;

typedef struct {
    const coli_gemma4_gguf *gguf;
    uint32_t layer;
    uint32_t model_width;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    int sliding;
    int key_equals_value;
    coli_gemma4_matrix query;
    coli_gemma4_matrix key;
    coli_gemma4_matrix value;
    coli_gemma4_matrix output;
    float *input_norm;
    float *query_norm;
    float *key_norm;
    float *rope_factors;
    uint32_t rope_factor_count;
    char last_error[COLI_GEMMA4_GGUF_ERROR_MAX];
} coli_gemma4_attention;

typedef struct {
    uint32_t capacity;
    uint32_t count;
    uint32_t kv_width;
    int sliding;
    uint64_t *positions;
    float *keys;
    float *values;
} coli_gemma4_kv_cache;

typedef struct {
    const coli_gemma4_gguf *gguf;
    uint32_t layer;
    uint32_t model_width;
    uint32_t intermediate_width;
    coli_gemma4_matrix gate;
    coli_gemma4_matrix up;
    coli_gemma4_matrix down;
    char last_error[COLI_GEMMA4_GGUF_ERROR_MAX];
} coli_gemma4_dense_mlp;

typedef struct {
    const coli_gemma4_gguf *gguf;
    uint32_t layer;
    uint32_t model_width;
    coli_gemma4_attention attention;
    coli_gemma4_router router;
    coli_gemma4_dense_mlp dense_mlp;
    float *post_attention_norm;
    float *ffn_norm;
    float *post_ffw_norm_1;
    float *pre_ffw_norm_2;
    float *post_ffw_norm_2;
    float *post_ffw_norm;
    float layer_output_scale;
    char last_error[COLI_GEMMA4_GGUF_ERROR_MAX];
} coli_gemma4_decoder_layer;

typedef struct {
    const coli_gemma4_gguf *gguf;
    uint32_t model_width;
    uint32_t vocab_size;
    coli_gemma4_matrix embedding;
    coli_gemma4_matrix output;
    int tied_output;
    float *output_norm;
    float embedding_scale;
    float logit_softcap;
    char last_error[COLI_GEMMA4_GGUF_ERROR_MAX];
} coli_gemma4_model_io;

typedef struct {
    const coli_gemma4_gguf *gguf;
    coli_gemma4_model_io io;
    coli_gemma4_decoder_layer *layers;
    coli_gemma4_kv_cache *caches;
    coli_expert_backend experts;
    uint32_t layer_count;
    uint32_t maximum_tokens;
    uint64_t next_position;
    char last_error[COLI_GEMMA4_GGUF_ERROR_MAX];
} coli_gemma4_model;

int coli_gemma4_matrix_open(coli_gemma4_matrix *matrix,
                            const coli_gemma4_gguf *gguf, const char *name);
void coli_gemma4_matrix_close(coli_gemma4_matrix *matrix);
int coli_gemma4_matrix_matvec(const coli_gemma4_matrix *matrix,
                              const float *input, float *output);
int coli_gemma4_matrix_row(const coli_gemma4_matrix *matrix, uint32_t row,
                           float *output);

int coli_gemma4_model_io_open(coli_gemma4_model_io *io,
                              const coli_gemma4_gguf *gguf);
void coli_gemma4_model_io_close(coli_gemma4_model_io *io);
const char *coli_gemma4_model_io_last_error(const coli_gemma4_model_io *io);
int coli_gemma4_model_embed(const coli_gemma4_model_io *io, uint32_t token,
                            float *embedding);
int coli_gemma4_model_logits(const coli_gemma4_model_io *io,
                             const float *residual, float *normalized,
                             float *logits);
int coli_gemma4_model_open(coli_gemma4_model *model,
                           const coli_gemma4_gguf *gguf,
                           const coli_expert_backend *experts,
                           uint32_t maximum_tokens);
void coli_gemma4_model_close(coli_gemma4_model *model);
const char *coli_gemma4_model_last_error(const coli_gemma4_model *model);
int coli_gemma4_model_step(coli_gemma4_model *model, uint32_t token,
                           uint64_t position, float *final_residual,
                           float *logits);
int coli_gemma4_model_step_cancelable(coli_gemma4_model *model,
                                      uint32_t token, uint64_t position,
                                      float *final_residual, float *logits,
                                      int (*cancelled)(void *), void *opaque);
int coli_gemma4_model_image_embeddings(coli_gemma4_model *model,
                                       const float *embeddings,
                                       uint32_t token_count,
                                       uint64_t start_position,
                                       float *final_residuals);

int coli_gemma4_attention_open(coli_gemma4_attention *attention,
                               const coli_gemma4_gguf *gguf, uint32_t layer);
void coli_gemma4_attention_close(coli_gemma4_attention *attention);
const char *coli_gemma4_attention_last_error(
    const coli_gemma4_attention *attention);
int coli_gemma4_attention_project(const coli_gemma4_attention *attention,
                                  const float *residual,
                                  float *query, float *key, float *value);
int coli_gemma4_attention_apply_rope(const coli_gemma4_attention *attention,
                                     uint64_t position, float *query, float *key);
int coli_gemma4_attention_step(const coli_gemma4_attention *attention,
                               coli_gemma4_kv_cache *cache,
                               uint64_t position, const float *residual,
                               float *output);
int coli_gemma4_attention_noncausal(
    const coli_gemma4_attention *attention, coli_gemma4_kv_cache *cache,
    uint64_t start_position, const float *residuals, uint32_t token_count,
    float *outputs);

int coli_gemma4_kv_cache_init(coli_gemma4_kv_cache *cache,
                              const coli_gemma4_attention *attention,
                              uint32_t maximum_tokens);
void coli_gemma4_kv_cache_close(coli_gemma4_kv_cache *cache);
int coli_gemma4_kv_cache_store(coli_gemma4_kv_cache *cache, uint64_t position,
                               const float *key, const float *value);
int coli_gemma4_kv_cache_find(const coli_gemma4_kv_cache *cache,
                              uint64_t position,
                              const float **key, const float **value);

int coli_gemma4_dense_mlp_open(coli_gemma4_dense_mlp *mlp,
                               const coli_gemma4_gguf *gguf, uint32_t layer);
void coli_gemma4_dense_mlp_close(coli_gemma4_dense_mlp *mlp);
const char *coli_gemma4_dense_mlp_last_error(
    const coli_gemma4_dense_mlp *mlp);
int coli_gemma4_dense_mlp_run(const coli_gemma4_dense_mlp *mlp,
                              const float *input, float *output);

int coli_gemma4_decoder_layer_open(coli_gemma4_decoder_layer *decoder,
                                   const coli_gemma4_gguf *gguf,
                                   uint32_t layer);
void coli_gemma4_decoder_layer_close(coli_gemma4_decoder_layer *decoder);
const char *coli_gemma4_decoder_layer_last_error(
    const coli_gemma4_decoder_layer *decoder);
int coli_gemma4_decoder_layer_step(
    const coli_gemma4_decoder_layer *decoder, coli_gemma4_kv_cache *cache,
    const coli_expert_backend *experts, uint64_t position,
    const float *residual, float *output);
int coli_gemma4_decoder_layer_noncausal(
    const coli_gemma4_decoder_layer *decoder, coli_gemma4_kv_cache *cache,
    const coli_expert_backend *experts, uint64_t start_position,
    const float *residuals, uint32_t token_count, float *outputs);

int coli_gemma4_router_open(coli_gemma4_router *router,
                            const coli_gemma4_gguf *gguf, uint32_t layer);
void coli_gemma4_router_close(coli_gemma4_router *router);
const char *coli_gemma4_router_last_error(const coli_gemma4_router *router);

/*
 * probabilities receives the full softmax when non-NULL. selected_weights are
 * normalized top-k probabilities before per-expert scaling, which is the form
 * expected by coli_gemma4_backend (that backend applies the expert scale while
 * executing each record). effective_weights optionally receives the equivalent
 * reference-graph weights after per-expert scaling.
 */
int coli_gemma4_router_route(const coli_gemma4_router *router,
                             const float *hidden_state,
                             float *probabilities,
                             uint32_t *selected_ids,
                             float *selected_weights,
                             float *effective_weights);

int coli_gemma4_rmsnorm(const float *input, const float *weight,
                        uint32_t width, float epsilon, float *output);

#endif
