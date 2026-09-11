#define _CRT_SECURE_NO_WARNINGS

#include "gemma4_model.h"
#include "gemma4_backend.h"

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void router_error(coli_gemma4_router *router, const char *format, ...) {
    va_list arguments;
    if (!router) return;
    va_start(arguments, format);
    vsnprintf(router->last_error, sizeof(router->last_error), format, arguments);
    va_end(arguments);
}

static int tensor_is(const coli_gemma4_tensor *tensor, uint32_t type,
                     uint32_t n_dims, uint64_t dim0, uint64_t dim1) {
    return tensor && tensor->type == type && tensor->n_dims == n_dims &&
           tensor->dims[0] == dim0 && (n_dims < 2 || tensor->dims[1] == dim1);
}

static int read_f32_tensor(coli_gemma4_router *router,
                           const coli_gemma4_tensor *tensor, float **result) {
    float *values;
    if (!tensor || tensor->nbytes > SIZE_MAX) {
        router_error(router, "router tensor is missing or too large");
        return -1;
    }
    values = (float *)malloc((size_t)tensor->nbytes);
    if (!values) {
        router_error(router, "out of memory loading router tensor");
        return -1;
    }
    if (coli_gemma4_gguf_read(router->gguf, tensor, values,
                              (size_t)tensor->nbytes) != 0) {
        free(values);
        router_error(router, "failed reading router tensor %s", tensor->name);
        return -1;
    }
    *result = values;
    return 0;
}

int coli_gemma4_matrix_open(coli_gemma4_matrix *matrix,
                            const coli_gemma4_gguf *gguf, const char *name) {
    const coli_gemma4_tensor *tensor;
    if (!matrix || !gguf || !name) return -1;
    memset(matrix, 0, sizeof(*matrix));
    tensor = coli_gemma4_gguf_find(gguf, name);
    if (!tensor || (tensor->type != COLI_GGML_TYPE_Q4_0 &&
                    tensor->type != COLI_GGML_TYPE_Q6_K) ||
        tensor->n_dims != 2 || !tensor->dims[0] || !tensor->dims[1] ||
        (tensor->type == COLI_GGML_TYPE_Q4_0 && tensor->dims[0] % 32 != 0) ||
        (tensor->type == COLI_GGML_TYPE_Q6_K && tensor->dims[0] % 256 != 0) ||
        tensor->dims[0] > UINT32_MAX || tensor->dims[1] > UINT32_MAX ||
        tensor->nbytes > SIZE_MAX) return -1;
    matrix->data = (uint8_t *)malloc((size_t)tensor->nbytes);
    if (!matrix->data) return -1;
    if (coli_gemma4_gguf_read(gguf, tensor, matrix->data,
                              (size_t)tensor->nbytes) != 0) {
        free(matrix->data);
        memset(matrix, 0, sizeof(*matrix));
        return -1;
    }
    matrix->gguf = gguf;
    matrix->tensor = tensor;
    matrix->input_width = (uint32_t)tensor->dims[0];
    matrix->output_width = (uint32_t)tensor->dims[1];
    return 0;
}

void coli_gemma4_matrix_close(coli_gemma4_matrix *matrix) {
    if (!matrix) return;
    free(matrix->data);
    memset(matrix, 0, sizeof(*matrix));
}

int coli_gemma4_matrix_matvec(const coli_gemma4_matrix *matrix,
                              const float *input, float *output) {
    if (!matrix || !matrix->data || !input || !output) return -1;
    if (matrix->tensor->type == COLI_GGML_TYPE_Q4_0)
        return coli_gemma4_q4_0_matvec(matrix->data, matrix->output_width,
                                       matrix->input_width, input, output);
    if (matrix->tensor->type == COLI_GGML_TYPE_Q6_K)
        return coli_gemma4_q6_k_matvec(matrix->data, matrix->output_width,
                                       matrix->input_width, input, output);
    return -1;
}

int coli_gemma4_matrix_row(const coli_gemma4_matrix *matrix, uint32_t row,
                           float *output) {
    if (!matrix || !matrix->data || !output ||
        matrix->tensor->type != COLI_GGML_TYPE_Q6_K) return -1;
    return coli_gemma4_q6_k_row(matrix->data, matrix->output_width,
                                matrix->input_width, row, output);
}

static void model_io_error(coli_gemma4_model_io *io,
                           const char *format, ...) {
    va_list arguments;
    if (!io) return;
    va_start(arguments, format);
    vsnprintf(io->last_error, sizeof(io->last_error), format, arguments);
    va_end(arguments);
}

int coli_gemma4_model_io_open(coli_gemma4_model_io *io,
                              const coli_gemma4_gguf *gguf) {
    const coli_gemma4_tensor *norm;
    if (!io || !gguf) return -1;
    memset(io, 0, sizeof(*io));
    io->gguf = gguf;
    io->model_width = gguf->config.n_embd;
    io->vocab_size = gguf->config.n_vocab;
    io->embedding_scale = sqrtf((float)io->model_width);
    io->logit_softcap = gguf->final_logit_softcap;
    if (!io->model_width || !io->vocab_size ||
        !isfinite(io->embedding_scale) ||
        !isfinite(io->logit_softcap) || io->logit_softcap < 0.0F) {
        model_io_error(io, "invalid Gemma 4 model I/O configuration");
        return -1;
    }
    if (coli_gemma4_matrix_open(&io->embedding, gguf,
                                "token_embd.weight") != 0 ||
        io->embedding.input_width != io->model_width ||
        io->embedding.output_width != io->vocab_size) {
        model_io_error(io, "invalid token_embd.weight tensor");
        goto fail;
    }
    if (coli_gemma4_gguf_find(gguf, "output.weight")) {
        if (coli_gemma4_matrix_open(&io->output, gguf, "output.weight") != 0 ||
            io->output.input_width != io->model_width ||
            io->output.output_width != io->vocab_size) {
            model_io_error(io, "invalid output.weight tensor");
            goto fail;
        }
    } else {
        io->tied_output = 1;
    }
    norm = coli_gemma4_gguf_find(gguf, "output_norm.weight");
    if (!tensor_is(norm, COLI_GGML_TYPE_F32, 1, io->model_width, 0) ||
        norm->nbytes != (uint64_t)io->model_width * sizeof(float) ||
        norm->nbytes > SIZE_MAX) {
        model_io_error(io, "invalid output_norm.weight tensor");
        goto fail;
    }
    io->output_norm = (float *)malloc((size_t)norm->nbytes);
    if (!io->output_norm ||
        coli_gemma4_gguf_read(gguf, norm, io->output_norm,
                              (size_t)norm->nbytes) != 0) {
        model_io_error(io, "cannot read output_norm.weight tensor");
        goto fail;
    }
    return 0;

fail:
    {
        char saved[COLI_GEMMA4_GGUF_ERROR_MAX];
        memcpy(saved, io->last_error, sizeof(saved));
        coli_gemma4_model_io_close(io);
        memcpy(io->last_error, saved, sizeof(saved));
    }
    return -1;
}

void coli_gemma4_model_io_close(coli_gemma4_model_io *io) {
    if (!io) return;
    coli_gemma4_matrix_close(&io->embedding);
    coli_gemma4_matrix_close(&io->output);
    free(io->output_norm);
    memset(io, 0, sizeof(*io));
}

const char *coli_gemma4_model_io_last_error(const coli_gemma4_model_io *io) {
    return io ? io->last_error : "invalid Gemma 4 model I/O";
}

int coli_gemma4_model_embed(const coli_gemma4_model_io *io, uint32_t token,
                            float *embedding) {
    uint32_t index;
    if (!io || !embedding || token >= io->vocab_size ||
        coli_gemma4_matrix_row(&io->embedding, token, embedding) != 0)
        return -1;
    for (index = 0; index < io->model_width; ++index)
        embedding[index] *= io->embedding_scale;
    return 0;
}

int coli_gemma4_model_logits(const coli_gemma4_model_io *io,
                             const float *residual, float *normalized,
                             float *logits) {
    const coli_gemma4_matrix *output;
    uint32_t token;
    if (!io || !residual || !normalized || !logits || !io->output_norm)
        return -1;
    output = io->tied_output ? &io->embedding : &io->output;
    if (coli_gemma4_rmsnorm(residual, io->output_norm, io->model_width,
                            io->gguf->rms_epsilon, normalized) != 0 ||
        coli_gemma4_matrix_matvec(output, normalized, logits) != 0)
        return -1;
    if (io->logit_softcap > 0.0F) {
        for (token = 0; token < io->vocab_size; ++token)
            logits[token] = io->logit_softcap *
                tanhf(logits[token] / io->logit_softcap);
    }
    return 0;
}

static void model_error(coli_gemma4_model *model, const char *format, ...) {
    va_list arguments;
    if (!model) return;
    va_start(arguments, format);
    vsnprintf(model->last_error, sizeof(model->last_error), format, arguments);
    va_end(arguments);
}

int coli_gemma4_model_open(coli_gemma4_model *model,
                           const coli_gemma4_gguf *gguf,
                           const coli_expert_backend *experts,
                           uint32_t maximum_tokens) {
    uint32_t layer;
    if (!model || !gguf || !experts || !maximum_tokens) return -1;
    memset(model, 0, sizeof(*model));
    model->gguf = gguf;
    model->experts = *experts;
    model->layer_count = gguf->config.n_layer;
    model->maximum_tokens = maximum_tokens;
    if (!model->layer_count || !experts->prepare_layer ||
        !experts->run_experts || !experts->release_layer) {
        model_error(model, "invalid Gemma 4 model runner configuration");
        return -1;
    }
    if (coli_gemma4_model_io_open(&model->io, gguf) != 0) {
        model_error(model, "%s", coli_gemma4_model_io_last_error(&model->io));
        goto fail;
    }
    model->layers = (coli_gemma4_decoder_layer *)calloc(
        model->layer_count, sizeof(*model->layers));
    model->caches = (coli_gemma4_kv_cache *)calloc(
        model->layer_count, sizeof(*model->caches));
    if (!model->layers || !model->caches) {
        model_error(model, "out of memory allocating Gemma 4 layers");
        goto fail;
    }
    for (layer = 0; layer < model->layer_count; ++layer) {
        if (coli_gemma4_decoder_layer_open(&model->layers[layer], gguf,
                                           layer) != 0) {
            model_error(model, "layer %u: %s", layer,
                        coli_gemma4_decoder_layer_last_error(
                            &model->layers[layer]));
            goto fail;
        }
        if (coli_gemma4_kv_cache_init(&model->caches[layer],
                                      &model->layers[layer].attention,
                                      maximum_tokens) != 0) {
            model_error(model, "layer %u: cannot allocate KV cache", layer);
            goto fail;
        }
    }
    return 0;

fail:
    {
        char saved[COLI_GEMMA4_GGUF_ERROR_MAX];
        memcpy(saved, model->last_error, sizeof(saved));
        coli_gemma4_model_close(model);
        memcpy(model->last_error, saved, sizeof(saved));
    }
    return -1;
}

void coli_gemma4_model_close(coli_gemma4_model *model) {
    uint32_t layer;
    if (!model) return;
    for (layer = 0; layer < model->layer_count; ++layer) {
        if (model->caches) coli_gemma4_kv_cache_close(&model->caches[layer]);
        if (model->layers)
            coli_gemma4_decoder_layer_close(&model->layers[layer]);
    }
    free(model->caches);
    free(model->layers);
    coli_gemma4_model_io_close(&model->io);
    memset(model, 0, sizeof(*model));
}

const char *coli_gemma4_model_last_error(const coli_gemma4_model *model) {
    return model ? model->last_error : "invalid Gemma 4 model runner";
}

int coli_gemma4_model_step_cancelable(coli_gemma4_model *model,
                                      uint32_t token, uint64_t position,
                                      float *final_residual, float *logits,
                                      int (*cancelled)(void *), void *opaque) {
    float *current = NULL, *next = NULL, *normalized = NULL;
    float *lookahead_weights = NULL;
    uint32_t *lookahead_ids = NULL;
    uint32_t layer;
    size_t vector_bytes;
    int status = -1;
    if (!model || !model->layers || !model->caches ||
        position != model->next_position || position >= model->maximum_tokens)
        return -1;
    vector_bytes = (size_t)model->io.model_width * sizeof(float);
    current = (float *)malloc(vector_bytes);
    next = (float *)malloc(vector_bytes);
    if (logits) normalized = (float *)malloc(vector_bytes);
    if (model->experts.prefetch_layer) {
        lookahead_ids = (uint32_t *)malloc(
            (size_t)model->gguf->config.n_expert_used * sizeof(*lookahead_ids));
        lookahead_weights = (float *)malloc(
            (size_t)model->gguf->config.n_expert_used *
            sizeof(*lookahead_weights));
    }
    if (!current || !next || (logits && !normalized) ||
        (model->experts.prefetch_layer &&
         (!lookahead_ids || !lookahead_weights))) {
        model_error(model, "out of memory allocating token workspace");
        goto cleanup;
    }
    if (coli_gemma4_model_embed(&model->io, token, current) != 0) {
        model_error(model, "cannot decode embedding for token %u", token);
        goto cleanup;
    }
    for (layer = 0; layer < model->layer_count; ++layer) {
        float *swap;
        if (cancelled && cancelled(opaque)) {
            status = 1;
            goto cleanup;
        }
        if (coli_gemma4_decoder_layer_step(
                &model->layers[layer], &model->caches[layer], &model->experts,
                position, current, next) != 0) {
            model_error(model, "decoder layer %u failed at position %llu",
                        layer, (unsigned long long)position);
            goto cleanup;
        }
        swap = current;
        current = next;
        next = swap;
        if (layer + 1 < model->layer_count &&
            model->experts.prefetch_layer &&
            (coli_gemma4_router_route(
                 &model->layers[layer + 1].router, current, NULL,
                 lookahead_ids, lookahead_weights, NULL) != 0 ||
             model->experts.prefetch_layer(
                 model->experts.ctx, layer + 1, lookahead_ids,
                 model->layers[layer + 1].router.selected_count) != 0)) {
            model_error(model,
                        "next-layer expert lookahead failed after layer %u",
                        layer);
            goto cleanup;
        }
    }
    if (final_residual) memcpy(final_residual, current, vector_bytes);
    if (logits && coli_gemma4_model_logits(&model->io, current, normalized,
                                           logits) != 0) {
        model_error(model, "LM head failed at position %llu",
                    (unsigned long long)position);
        goto cleanup;
    }
    ++model->next_position;
    status = 0;

cleanup:
    free(current);
    free(next);
    free(normalized);
    free(lookahead_ids);
    free(lookahead_weights);
    return status;
}

int coli_gemma4_model_step(coli_gemma4_model *model, uint32_t token,
                           uint64_t position, float *final_residual,
                           float *logits) {
    return coli_gemma4_model_step_cancelable(
        model, token, position, final_residual, logits, NULL, NULL);
}

int coli_gemma4_model_image_embeddings(coli_gemma4_model *model,
                                       const float *embeddings,
                                       uint32_t token_count,
                                       uint64_t start_position,
                                       float *final_residuals) {
    float *current = NULL, *next = NULL;
    uint32_t layer;
    uint64_t value_count;
    size_t buffer_bytes;
    int status = -1;
    if (!model || !model->layers || !model->caches || !embeddings ||
        !token_count || start_position != model->next_position ||
        start_position >= model->maximum_tokens ||
        token_count > model->maximum_tokens - start_position)
        return -1;
    value_count = (uint64_t)token_count * model->io.model_width;
    if (value_count > SIZE_MAX / sizeof(float)) return -1;
    buffer_bytes = (size_t)value_count * sizeof(float);
    current = (float *)malloc(buffer_bytes);
    next = (float *)malloc(buffer_bytes);
    if (!current || !next) {
        model_error(model, "out of memory allocating image-token workspace");
        goto cleanup;
    }
    memcpy(current, embeddings, buffer_bytes);
    for (layer = 0; layer < model->layer_count; ++layer) {
        float *swap;
        if (coli_gemma4_decoder_layer_noncausal(
                &model->layers[layer], &model->caches[layer], &model->experts,
                start_position, current, token_count, next) != 0) {
            model_error(model,
                        "decoder layer %u failed for %u image tokens at position %llu",
                        layer, token_count, (unsigned long long)start_position);
            goto cleanup;
        }
        swap = current;
        current = next;
        next = swap;
    }
    if (final_residuals) memcpy(final_residuals, current, buffer_bytes);
    model->next_position += token_count;
    status = 0;
cleanup:
    free(current);
    free(next);
    return status;
}

static void attention_error(coli_gemma4_attention *attention,
                            const char *format, ...) {
    va_list arguments;
    if (!attention) return;
    va_start(arguments, format);
    vsnprintf(attention->last_error, sizeof(attention->last_error),
              format, arguments);
    va_end(arguments);
}

static int attention_name(char *name, size_t capacity, uint32_t layer,
                          const char *suffix) {
    int length = snprintf(name, capacity, "blk.%u.%s", layer, suffix);
    return length >= 0 && (size_t)length < capacity ? 0 : -1;
}

static int attention_read_norm(coli_gemma4_attention *attention,
                               const char *suffix, uint32_t width,
                               float **result) {
    char name[128];
    const coli_gemma4_tensor *tensor;
    float *values;
    if (attention_name(name, sizeof(name), attention->layer, suffix) != 0)
        return -1;
    tensor = coli_gemma4_gguf_find(attention->gguf, name);
    if (!tensor_is(tensor, COLI_GGML_TYPE_F32, 1, width, 0) ||
        tensor->nbytes != (uint64_t)width * sizeof(float) ||
        tensor->nbytes > SIZE_MAX) {
        attention_error(attention, "invalid attention norm tensor %s", name);
        return -1;
    }
    values = (float *)malloc((size_t)tensor->nbytes);
    if (!values || coli_gemma4_gguf_read(attention->gguf, tensor, values,
                                         (size_t)tensor->nbytes) != 0) {
        free(values);
        attention_error(attention, "cannot read attention norm tensor %s", name);
        return -1;
    }
    *result = values;
    return 0;
}

static int attention_open_matrix(coli_gemma4_attention *attention,
                                 const char *suffix,
                                 coli_gemma4_matrix *matrix,
                                 uint32_t input_width,
                                 uint32_t output_width) {
    char name[128];
    if (attention_name(name, sizeof(name), attention->layer, suffix) != 0 ||
        coli_gemma4_matrix_open(matrix, attention->gguf, name) != 0 ||
        matrix->input_width != input_width ||
        matrix->output_width != output_width) {
        coli_gemma4_matrix_close(matrix);
        attention_error(attention, "invalid attention projection %s", name);
        return -1;
    }
    return 0;
}

int coli_gemma4_attention_open(coli_gemma4_attention *attention,
                               const coli_gemma4_gguf *gguf, uint32_t layer) {
    uint32_t query_width, kv_width;
    if (!attention || !gguf) return -1;
    memset(attention, 0, sizeof(*attention));
    attention->gguf = gguf;
    attention->layer = layer;
    if (layer >= gguf->config.n_layer) {
        attention_error(attention, "attention layer %u is outside the model", layer);
        return -1;
    }
    attention->model_width = gguf->config.n_embd;
    attention->query_heads = gguf->attention_heads;
    attention->kv_heads = gguf->head_count_kv[layer];
    attention->sliding = gguf->sliding_window_pattern[layer] != 0;
    attention->head_dim = attention->sliding ?
        gguf->key_length_swa : gguf->key_length;
    if (!attention->model_width || !attention->query_heads ||
        !attention->kv_heads || !attention->head_dim ||
        attention->query_heads > UINT32_MAX / attention->head_dim ||
        attention->kv_heads > UINT32_MAX / attention->head_dim) {
        attention_error(attention, "invalid attention geometry for layer %u", layer);
        return -1;
    }
    query_width = attention->query_heads * attention->head_dim;
    kv_width = attention->kv_heads * attention->head_dim;
    if (attention_open_matrix(attention, "attn_q.weight", &attention->query,
                              attention->model_width, query_width) != 0 ||
        attention_open_matrix(attention, "attn_k.weight", &attention->key,
                              attention->model_width, kv_width) != 0 ||
        attention_open_matrix(attention, "attn_output.weight", &attention->output,
                              query_width, attention->model_width) != 0)
        goto fail;
    {
        char value_name[128];
        const coli_gemma4_tensor *value_tensor;
        if (attention_name(value_name, sizeof(value_name), layer,
                           "attn_v.weight") != 0) goto fail;
        value_tensor = coli_gemma4_gguf_find(gguf, value_name);
        if (value_tensor) {
            if (attention_open_matrix(attention, "attn_v.weight", &attention->value,
                                      attention->model_width, kv_width) != 0)
                goto fail;
        } else {
            attention->key_equals_value = 1;
        }
    }
    if (attention_read_norm(attention, "attn_norm.weight",
                            attention->model_width, &attention->input_norm) != 0 ||
        attention_read_norm(attention, "attn_q_norm.weight",
                            attention->head_dim, &attention->query_norm) != 0 ||
        attention_read_norm(attention, "attn_k_norm.weight",
                            attention->head_dim, &attention->key_norm) != 0)
        goto fail;
    if (!attention->sliding) {
        const coli_gemma4_tensor *factors =
            coli_gemma4_gguf_find(gguf, "rope_freqs.weight");
        uint32_t factor_count = attention->head_dim / 2;
        if (!tensor_is(factors, COLI_GGML_TYPE_F32, 1, factor_count, 0) ||
            factors->nbytes != (uint64_t)factor_count * sizeof(float) ||
            factors->nbytes > SIZE_MAX) {
            attention_error(attention, "invalid global RoPE frequency factors");
            goto fail;
        }
        attention->rope_factors = (float *)malloc((size_t)factors->nbytes);
        if (!attention->rope_factors ||
            coli_gemma4_gguf_read(gguf, factors, attention->rope_factors,
                                  (size_t)factors->nbytes) != 0) {
            attention_error(attention, "cannot read global RoPE frequency factors");
            goto fail;
        }
        attention->rope_factor_count = factor_count;
    }
    return 0;

fail:
    {
        char saved[COLI_GEMMA4_GGUF_ERROR_MAX];
        memcpy(saved, attention->last_error, sizeof(saved));
        coli_gemma4_attention_close(attention);
        memcpy(attention->last_error, saved, sizeof(saved));
    }
    return -1;
}

void coli_gemma4_attention_close(coli_gemma4_attention *attention) {
    if (!attention) return;
    coli_gemma4_matrix_close(&attention->query);
    coli_gemma4_matrix_close(&attention->key);
    coli_gemma4_matrix_close(&attention->value);
    coli_gemma4_matrix_close(&attention->output);
    free(attention->input_norm);
    free(attention->query_norm);
    free(attention->key_norm);
    free(attention->rope_factors);
    memset(attention, 0, sizeof(*attention));
}

const char *coli_gemma4_attention_last_error(
    const coli_gemma4_attention *attention) {
    return attention ? attention->last_error : "invalid Gemma attention";
}

static int rmsnorm_heads(float *values, uint32_t heads, uint32_t head_dim,
                         const float *weight, float epsilon) {
    uint32_t head;
    for (head = 0; head < heads; ++head) {
        float *row = values + (size_t)head * head_dim;
        float sum_squares = 0.0F, inverse_rms;
        uint32_t i;
        for (i = 0; i < head_dim; ++i) {
            if (!isfinite(row[i]) || (weight && !isfinite(weight[i]))) return -1;
            sum_squares += row[i] * row[i];
        }
        inverse_rms = 1.0F / sqrtf(sum_squares / (float)head_dim + epsilon);
        for (i = 0; i < head_dim; ++i)
            row[i] *= inverse_rms * (weight ? weight[i] : 1.0F);
    }
    return 0;
}

int coli_gemma4_attention_project(const coli_gemma4_attention *attention,
                                  const float *residual,
                                  float *query, float *key, float *value) {
    float *normalized;
    uint32_t kv_width;
    if (!attention || !residual || !query || !key || !value ||
        !attention->input_norm || !attention->query_norm || !attention->key_norm)
        return -1;
    normalized = (float *)malloc((size_t)attention->model_width * sizeof(float));
    if (!normalized) return -1;
    if (coli_gemma4_rmsnorm(residual, attention->input_norm,
                            attention->model_width, attention->gguf->rms_epsilon,
                            normalized) != 0 ||
        coli_gemma4_matrix_matvec(&attention->query, normalized, query) != 0 ||
        coli_gemma4_matrix_matvec(&attention->key, normalized, key) != 0) {
        free(normalized);
        return -1;
    }
    kv_width = attention->kv_heads * attention->head_dim;
    if (attention->key_equals_value) memcpy(value, key, (size_t)kv_width * sizeof(float));
    else if (coli_gemma4_matrix_matvec(&attention->value, normalized, value) != 0) {
        free(normalized);
        return -1;
    }
    free(normalized);
    if (rmsnorm_heads(query, attention->query_heads, attention->head_dim,
                      attention->query_norm, attention->gguf->rms_epsilon) != 0 ||
        rmsnorm_heads(key, attention->kv_heads, attention->head_dim,
                      attention->key_norm, attention->gguf->rms_epsilon) != 0 ||
        rmsnorm_heads(value, attention->kv_heads, attention->head_dim,
                      NULL, attention->gguf->rms_epsilon) != 0)
        return -1;
    return 0;
}

static int apply_rope_heads(float *values, uint32_t heads, uint32_t head_dim,
                            uint64_t position, float base,
                            const float *factors, uint32_t factor_count) {
    uint32_t head, index, half = head_dim / 2;
    float position_f = (float)position;
    if (!values || !heads || !head_dim || head_dim % 2 ||
        !isfinite(base) || base <= 0.0F ||
        (factors && factor_count != half)) return -1;
    for (index = 0; index < half; ++index) {
        float factor = factors ? factors[index] : 1.0F;
        float frequency, angle, cosine, sine;
        if (!isfinite(factor) || factor <= 0.0F) return -1;
        frequency = powf(base, -2.0F * (float)index / (float)head_dim) / factor;
        angle = position_f * frequency;
        cosine = cosf(angle);
        sine = sinf(angle);
        for (head = 0; head < heads; ++head) {
            float *row = values + (size_t)head * head_dim;
            float first = row[index], second = row[index + half];
            row[index] = first * cosine - second * sine;
            row[index + half] = second * cosine + first * sine;
        }
    }
    return 0;
}

int coli_gemma4_attention_apply_rope(const coli_gemma4_attention *attention,
                                     uint64_t position, float *query, float *key) {
    float base;
    if (!attention || !attention->gguf || !query || !key) return -1;
    base = attention->sliding ? attention->gguf->rope_freq_base_swa :
                                attention->gguf->rope_freq_base;
    if (apply_rope_heads(query, attention->query_heads, attention->head_dim,
                         position, base, attention->rope_factors,
                         attention->rope_factor_count) != 0 ||
        apply_rope_heads(key, attention->kv_heads, attention->head_dim,
                         position, base, attention->rope_factors,
                         attention->rope_factor_count) != 0)
        return -1;
    return 0;
}

int coli_gemma4_kv_cache_init(coli_gemma4_kv_cache *cache,
                              const coli_gemma4_attention *attention,
                              uint32_t maximum_tokens) {
    uint32_t capacity, i;
    uint64_t value_count;
    if (!cache || !attention || !maximum_tokens) return -1;
    memset(cache, 0, sizeof(*cache));
    capacity = maximum_tokens;
    if (attention->sliding && attention->gguf->config.sliding_window < capacity)
        capacity = attention->gguf->config.sliding_window;
    if (!capacity || attention->kv_heads > UINT32_MAX / attention->head_dim)
        return -1;
    cache->kv_width = attention->kv_heads * attention->head_dim;
    value_count = (uint64_t)capacity * cache->kv_width;
    if (value_count > SIZE_MAX / sizeof(float)) return -1;
    cache->positions = (uint64_t *)malloc((size_t)capacity * sizeof(uint64_t));
    cache->keys = (float *)malloc((size_t)value_count * sizeof(float));
    cache->values = (float *)malloc((size_t)value_count * sizeof(float));
    if (!cache->positions || !cache->keys || !cache->values) {
        coli_gemma4_kv_cache_close(cache);
        return -1;
    }
    cache->capacity = capacity;
    cache->sliding = attention->sliding;
    for (i = 0; i < capacity; ++i) cache->positions[i] = UINT64_MAX;
    return 0;
}

void coli_gemma4_kv_cache_close(coli_gemma4_kv_cache *cache) {
    if (!cache) return;
    free(cache->positions);
    free(cache->keys);
    free(cache->values);
    memset(cache, 0, sizeof(*cache));
}

int coli_gemma4_kv_cache_store(coli_gemma4_kv_cache *cache, uint64_t position,
                               const float *key, const float *value) {
    uint64_t slot;
    if (!cache || !cache->capacity || !key || !value) return -1;
    if (cache->sliding) slot = position % cache->capacity;
    else {
        if (position >= cache->capacity) return -1;
        slot = position;
    }
    memcpy(cache->keys + (size_t)slot * cache->kv_width, key,
           (size_t)cache->kv_width * sizeof(float));
    memcpy(cache->values + (size_t)slot * cache->kv_width, value,
           (size_t)cache->kv_width * sizeof(float));
    if (cache->positions[slot] == UINT64_MAX && cache->count < cache->capacity)
        ++cache->count;
    cache->positions[slot] = position;
    return 0;
}

int coli_gemma4_kv_cache_find(const coli_gemma4_kv_cache *cache,
                              uint64_t position,
                              const float **key, const float **value) {
    uint64_t slot;
    if (!cache || !cache->capacity || !key || !value) return -1;
    if (cache->sliding) slot = position % cache->capacity;
    else {
        if (position >= cache->capacity) return -1;
        slot = position;
    }
    if (cache->positions[slot] != position) return -1;
    *key = cache->keys + (size_t)slot * cache->kv_width;
    *value = cache->values + (size_t)slot * cache->kv_width;
    return 0;
}

int coli_gemma4_attention_step(const coli_gemma4_attention *attention,
                               coli_gemma4_kv_cache *cache,
                               uint64_t position, const float *residual,
                               float *output) {
    float *query = NULL, *key = NULL, *value = NULL;
    float *context = NULL, *scores = NULL;
    uint32_t query_width, kv_width, queries_per_kv, head;
    uint64_t first_position, history_count, history_index;
    int status = -1;
    if (!attention || !cache || !residual || !output ||
        !cache->capacity || cache->sliding != attention->sliding ||
        attention->query_heads % attention->kv_heads != 0 ||
        attention->query_heads > UINT32_MAX / attention->head_dim ||
        attention->kv_heads > UINT32_MAX / attention->head_dim)
        return -1;
    query_width = attention->query_heads * attention->head_dim;
    kv_width = attention->kv_heads * attention->head_dim;
    if (cache->kv_width != kv_width) return -1;
    query = (float *)malloc((size_t)query_width * sizeof(float));
    key = (float *)malloc((size_t)kv_width * sizeof(float));
    value = (float *)malloc((size_t)kv_width * sizeof(float));
    context = (float *)calloc(query_width, sizeof(float));
    scores = (float *)malloc((size_t)cache->capacity * sizeof(float));
    if (!query || !key || !value || !context || !scores) goto cleanup;
    if (coli_gemma4_attention_project(attention, residual, query, key, value) != 0 ||
        coli_gemma4_attention_apply_rope(attention, position, query, key) != 0 ||
        coli_gemma4_kv_cache_store(cache, position, key, value) != 0)
        goto cleanup;

    first_position = 0;
    if (cache->sliding && position >= cache->capacity)
        first_position = position - cache->capacity + 1;
    history_count = position - first_position + 1;
    if (!history_count || history_count > cache->capacity) goto cleanup;
    queries_per_kv = attention->query_heads / attention->kv_heads;
    for (head = 0; head < attention->query_heads; ++head) {
        const float *query_head = query + (size_t)head * attention->head_dim;
        float *context_head = context + (size_t)head * attention->head_dim;
        uint32_t kv_head = head / queries_per_kv;
        float maximum = -FLT_MAX, denominator = 0.0F;
        for (history_index = 0; history_index < history_count; ++history_index) {
            const float *cached_key, *cached_value;
            const float *key_head;
            float score = 0.0F;
            uint32_t dimension;
            uint64_t cached_position = first_position + history_index;
            if (coli_gemma4_kv_cache_find(cache, cached_position,
                                          &cached_key, &cached_value) != 0)
                goto cleanup;
            (void)cached_value;
            key_head = cached_key + (size_t)kv_head * attention->head_dim;
            for (dimension = 0; dimension < attention->head_dim; ++dimension)
                score += query_head[dimension] * key_head[dimension];
            if (!isfinite(score)) goto cleanup;
            scores[history_index] = score;
            if (score > maximum) maximum = score;
        }
        for (history_index = 0; history_index < history_count; ++history_index) {
            scores[history_index] = expf(scores[history_index] - maximum);
            denominator += scores[history_index];
        }
        if (!isfinite(denominator) || denominator <= 0.0F) goto cleanup;
        for (history_index = 0; history_index < history_count; ++history_index) {
            const float *cached_key, *cached_value;
            const float *value_head;
            float probability = scores[history_index] / denominator;
            uint32_t dimension;
            uint64_t cached_position = first_position + history_index;
            if (coli_gemma4_kv_cache_find(cache, cached_position,
                                          &cached_key, &cached_value) != 0)
                goto cleanup;
            (void)cached_key;
            value_head = cached_value + (size_t)kv_head * attention->head_dim;
            for (dimension = 0; dimension < attention->head_dim; ++dimension)
                context_head[dimension] += probability * value_head[dimension];
        }
    }
    if (coli_gemma4_matrix_matvec(&attention->output, context, output) != 0)
        goto cleanup;
    status = 0;

cleanup:
    free(query); free(key); free(value); free(context); free(scores);
    return status;
}

int coli_gemma4_attention_noncausal(
    const coli_gemma4_attention *attention, coli_gemma4_kv_cache *cache,
    uint64_t start_position, const float *residuals, uint32_t token_count,
    float *outputs) {
    float *queries = NULL, *keys = NULL, *values = NULL;
    float *context = NULL, *scores = NULL;
    uint32_t query_width, kv_width, queries_per_kv, token, head;
    uint64_t end_position, first_position, history_count, history_index;
    uint64_t query_values, kv_values;
    int status = -1;
    if (!attention || !cache || !residuals || !outputs || !token_count ||
        !cache->capacity || cache->sliding != attention->sliding ||
        attention->query_heads % attention->kv_heads != 0 ||
        attention->query_heads > UINT32_MAX / attention->head_dim ||
        attention->kv_heads > UINT32_MAX / attention->head_dim ||
        start_position > UINT64_MAX - token_count)
        return -1;
    query_width = attention->query_heads * attention->head_dim;
    kv_width = attention->kv_heads * attention->head_dim;
    if (cache->kv_width != kv_width || token_count > cache->capacity)
        return -1;
    query_values = (uint64_t)token_count * query_width;
    kv_values = (uint64_t)token_count * kv_width;
    if (query_values > SIZE_MAX / sizeof(float) ||
        kv_values > SIZE_MAX / sizeof(float)) return -1;
    queries = (float *)malloc((size_t)query_values * sizeof(float));
    keys = (float *)malloc((size_t)kv_values * sizeof(float));
    values = (float *)malloc((size_t)kv_values * sizeof(float));
    context = (float *)malloc((size_t)query_width * sizeof(float));
    scores = (float *)malloc((size_t)cache->capacity * sizeof(float));
    if (!queries || !keys || !values || !context || !scores) goto cleanup;
    for (token = 0; token < token_count; ++token) {
        uint64_t position = start_position + token;
        const float *residual = residuals +
            (size_t)token * attention->model_width;
        float *query = queries + (size_t)token * query_width;
        float *key = keys + (size_t)token * kv_width;
        float *value = values + (size_t)token * kv_width;
        if (coli_gemma4_attention_project(
                attention, residual, query, key, value) != 0 ||
            coli_gemma4_attention_apply_rope(
                attention, position, query, key) != 0 ||
            coli_gemma4_kv_cache_store(cache, position, key, value) != 0)
            goto cleanup;
    }
    end_position = start_position + token_count - 1;
    first_position = 0;
    if (cache->sliding && end_position >= cache->capacity)
        first_position = end_position - cache->capacity + 1;
    history_count = end_position - first_position + 1;
    if (!history_count || history_count > cache->capacity) goto cleanup;
    queries_per_kv = attention->query_heads / attention->kv_heads;
    for (token = 0; token < token_count; ++token) {
        const float *query = queries + (size_t)token * query_width;
        float *output = outputs + (size_t)token * attention->model_width;
        memset(context, 0, (size_t)query_width * sizeof(float));
        for (head = 0; head < attention->query_heads; ++head) {
            const float *query_head = query + (size_t)head * attention->head_dim;
            float *context_head = context + (size_t)head * attention->head_dim;
            uint32_t kv_head = head / queries_per_kv;
            float maximum = -FLT_MAX, denominator = 0.0F;
            for (history_index = 0; history_index < history_count;
                 ++history_index) {
                const float *cached_key, *cached_value, *key_head;
                float score = 0.0F;
                uint32_t dimension;
                uint64_t cached_position = first_position + history_index;
                if (coli_gemma4_kv_cache_find(
                        cache, cached_position,
                        &cached_key, &cached_value) != 0)
                    goto cleanup;
                (void)cached_value;
                key_head = cached_key + (size_t)kv_head * attention->head_dim;
                for (dimension = 0; dimension < attention->head_dim; ++dimension)
                    score += query_head[dimension] * key_head[dimension];
                if (!isfinite(score)) goto cleanup;
                scores[history_index] = score;
                if (score > maximum) maximum = score;
            }
            for (history_index = 0; history_index < history_count;
                 ++history_index) {
                scores[history_index] = expf(scores[history_index] - maximum);
                denominator += scores[history_index];
            }
            if (!isfinite(denominator) || denominator <= 0.0F) goto cleanup;
            for (history_index = 0; history_index < history_count;
                 ++history_index) {
                const float *cached_key, *cached_value, *value_head;
                float probability = scores[history_index] / denominator;
                uint32_t dimension;
                uint64_t cached_position = first_position + history_index;
                if (coli_gemma4_kv_cache_find(
                        cache, cached_position,
                        &cached_key, &cached_value) != 0)
                    goto cleanup;
                (void)cached_key;
                value_head = cached_value +
                    (size_t)kv_head * attention->head_dim;
                for (dimension = 0; dimension < attention->head_dim; ++dimension)
                    context_head[dimension] +=
                        probability * value_head[dimension];
            }
        }
        if (coli_gemma4_matrix_matvec(
                &attention->output, context, output) != 0)
            goto cleanup;
    }
    status = 0;
cleanup:
    free(queries); free(keys); free(values); free(context); free(scores);
    return status;
}

static void dense_mlp_error(coli_gemma4_dense_mlp *mlp,
                            const char *format, ...) {
    va_list arguments;
    if (!mlp) return;
    va_start(arguments, format);
    vsnprintf(mlp->last_error, sizeof(mlp->last_error), format, arguments);
    va_end(arguments);
}

int coli_gemma4_dense_mlp_open(coli_gemma4_dense_mlp *mlp,
                               const coli_gemma4_gguf *gguf, uint32_t layer) {
    char name[128];
    if (!mlp || !gguf) return -1;
    memset(mlp, 0, sizeof(*mlp));
    mlp->gguf = gguf;
    mlp->layer = layer;
    mlp->model_width = gguf->config.n_embd;
    if (layer >= gguf->config.n_layer || !mlp->model_width) {
        dense_mlp_error(mlp, "dense MLP layer %u is outside the model", layer);
        return -1;
    }
    if (attention_name(name, sizeof(name), layer, "ffn_gate.weight") != 0 ||
        coli_gemma4_matrix_open(&mlp->gate, gguf, name) != 0 ||
        mlp->gate.input_width != mlp->model_width ||
        !mlp->gate.output_width) {
        dense_mlp_error(mlp, "invalid dense MLP gate projection");
        goto fail;
    }
    mlp->intermediate_width = mlp->gate.output_width;
    if (attention_name(name, sizeof(name), layer, "ffn_up.weight") != 0 ||
        coli_gemma4_matrix_open(&mlp->up, gguf, name) != 0 ||
        mlp->up.input_width != mlp->model_width ||
        mlp->up.output_width != mlp->intermediate_width) {
        dense_mlp_error(mlp, "invalid dense MLP up projection");
        goto fail;
    }
    if (attention_name(name, sizeof(name), layer, "ffn_down.weight") != 0 ||
        coli_gemma4_matrix_open(&mlp->down, gguf, name) != 0 ||
        mlp->down.input_width != mlp->intermediate_width ||
        mlp->down.output_width != mlp->model_width) {
        dense_mlp_error(mlp, "invalid dense MLP down projection");
        goto fail;
    }
    return 0;

fail:
    {
        char saved[COLI_GEMMA4_GGUF_ERROR_MAX];
        memcpy(saved, mlp->last_error, sizeof(saved));
        coli_gemma4_dense_mlp_close(mlp);
        memcpy(mlp->last_error, saved, sizeof(saved));
    }
    return -1;
}

void coli_gemma4_dense_mlp_close(coli_gemma4_dense_mlp *mlp) {
    if (!mlp) return;
    coli_gemma4_matrix_close(&mlp->gate);
    coli_gemma4_matrix_close(&mlp->up);
    coli_gemma4_matrix_close(&mlp->down);
    memset(mlp, 0, sizeof(*mlp));
}

const char *coli_gemma4_dense_mlp_last_error(
    const coli_gemma4_dense_mlp *mlp) {
    return mlp ? mlp->last_error : "invalid Gemma dense MLP";
}

int coli_gemma4_dense_mlp_run(const coli_gemma4_dense_mlp *mlp,
                              const float *input, float *output) {
    float *gate = NULL, *up = NULL;
    uint32_t index;
    int status = -1;
    if (!mlp || !input || !output || !mlp->intermediate_width) return -1;
    gate = (float *)malloc((size_t)mlp->intermediate_width * sizeof(float));
    up = (float *)malloc((size_t)mlp->intermediate_width * sizeof(float));
    if (!gate || !up) goto cleanup;
    if (coli_gemma4_matrix_matvec(&mlp->gate, input, gate) != 0 ||
        coli_gemma4_matrix_matvec(&mlp->up, input, up) != 0)
        goto cleanup;
    for (index = 0; index < mlp->intermediate_width; ++index)
        gate[index] = coli_gemma4_gelu_tanh(gate[index]) * up[index];
    if (coli_gemma4_matrix_matvec(&mlp->down, gate, output) != 0) goto cleanup;
    status = 0;

cleanup:
    free(gate); free(up);
    return status;
}

static void decoder_error(coli_gemma4_decoder_layer *decoder,
                          const char *format, ...) {
    va_list arguments;
    if (!decoder) return;
    va_start(arguments, format);
    vsnprintf(decoder->last_error, sizeof(decoder->last_error),
              format, arguments);
    va_end(arguments);
}

static int decoder_read_norm(coli_gemma4_decoder_layer *decoder,
                             const char *suffix, float **result) {
    char name[128];
    const coli_gemma4_tensor *tensor;
    float *values;
    if (attention_name(name, sizeof(name), decoder->layer, suffix) != 0)
        return -1;
    tensor = coli_gemma4_gguf_find(decoder->gguf, name);
    if (!tensor_is(tensor, COLI_GGML_TYPE_F32, 1,
                   decoder->model_width, 0) ||
        tensor->nbytes != (uint64_t)decoder->model_width * sizeof(float) ||
        tensor->nbytes > SIZE_MAX) {
        decoder_error(decoder, "invalid decoder norm tensor %s", name);
        return -1;
    }
    values = (float *)malloc((size_t)tensor->nbytes);
    if (!values || coli_gemma4_gguf_read(decoder->gguf, tensor, values,
                                         (size_t)tensor->nbytes) != 0) {
        free(values);
        decoder_error(decoder, "cannot read decoder norm tensor %s", name);
        return -1;
    }
    *result = values;
    return 0;
}

int coli_gemma4_decoder_layer_open(coli_gemma4_decoder_layer *decoder,
                                   const coli_gemma4_gguf *gguf,
                                   uint32_t layer) {
    char name[128];
    const coli_gemma4_tensor *scale;
    if (!decoder || !gguf) return -1;
    memset(decoder, 0, sizeof(*decoder));
    decoder->gguf = gguf;
    decoder->layer = layer;
    decoder->model_width = gguf->config.n_embd;
    if (layer >= gguf->config.n_layer || !decoder->model_width) {
        decoder_error(decoder, "decoder layer %u is outside the model", layer);
        return -1;
    }
    if (coli_gemma4_attention_open(&decoder->attention, gguf, layer) != 0) {
        decoder_error(decoder, "%s",
                      coli_gemma4_attention_last_error(&decoder->attention));
        goto fail;
    }
    if (coli_gemma4_router_open(&decoder->router, gguf, layer) != 0) {
        decoder_error(decoder, "%s",
                      coli_gemma4_router_last_error(&decoder->router));
        goto fail;
    }
    if (coli_gemma4_dense_mlp_open(&decoder->dense_mlp, gguf, layer) != 0) {
        decoder_error(decoder, "%s",
                      coli_gemma4_dense_mlp_last_error(&decoder->dense_mlp));
        goto fail;
    }
    if (decoder_read_norm(decoder, "post_attention_norm.weight",
                          &decoder->post_attention_norm) != 0 ||
        decoder_read_norm(decoder, "ffn_norm.weight", &decoder->ffn_norm) != 0 ||
        decoder_read_norm(decoder, "post_ffw_norm_1.weight",
                          &decoder->post_ffw_norm_1) != 0 ||
        decoder_read_norm(decoder, "pre_ffw_norm_2.weight",
                          &decoder->pre_ffw_norm_2) != 0 ||
        decoder_read_norm(decoder, "post_ffw_norm_2.weight",
                          &decoder->post_ffw_norm_2) != 0 ||
        decoder_read_norm(decoder, "post_ffw_norm.weight",
                          &decoder->post_ffw_norm) != 0)
        goto fail;
    if (attention_name(name, sizeof(name), layer,
                       "layer_output_scale.weight") != 0)
        goto fail;
    scale = coli_gemma4_gguf_find(gguf, name);
    if (!tensor_is(scale, COLI_GGML_TYPE_F32, 1, 1, 0) ||
        scale->nbytes != sizeof(float) ||
        coli_gemma4_gguf_read(gguf, scale, &decoder->layer_output_scale,
                              sizeof(float)) != 0 ||
        !isfinite(decoder->layer_output_scale)) {
        decoder_error(decoder, "invalid decoder layer output scale %s", name);
        goto fail;
    }
    return 0;

fail:
    {
        char saved[COLI_GEMMA4_GGUF_ERROR_MAX];
        memcpy(saved, decoder->last_error, sizeof(saved));
        coli_gemma4_decoder_layer_close(decoder);
        memcpy(decoder->last_error, saved, sizeof(saved));
    }
    return -1;
}

void coli_gemma4_decoder_layer_close(coli_gemma4_decoder_layer *decoder) {
    if (!decoder) return;
    coli_gemma4_attention_close(&decoder->attention);
    coli_gemma4_router_close(&decoder->router);
    coli_gemma4_dense_mlp_close(&decoder->dense_mlp);
    free(decoder->post_attention_norm);
    free(decoder->ffn_norm);
    free(decoder->post_ffw_norm_1);
    free(decoder->pre_ffw_norm_2);
    free(decoder->post_ffw_norm_2);
    free(decoder->post_ffw_norm);
    memset(decoder, 0, sizeof(*decoder));
}

const char *coli_gemma4_decoder_layer_last_error(
    const coli_gemma4_decoder_layer *decoder) {
    return decoder ? decoder->last_error : "invalid Gemma decoder layer";
}

static int decoder_layer_complete(
    const coli_gemma4_decoder_layer *decoder,
    const coli_expert_backend *experts, const float *residual,
    const float *attention_output, float *output) {
    float *after_attention = NULL;
    float *dense_input = NULL, *dense_output = NULL, *dense_normalized = NULL;
    float *expert_input = NULL, *expert_output = NULL, *expert_normalized = NULL;
    float *combined = NULL;
    uint32_t *ids = NULL, index;
    float *weights = NULL;
    size_t vector_bytes;
    int prepared = 0, status = -1;
    if (!decoder || !experts || !residual || !attention_output || !output ||
        !experts->prepare_layer || !experts->run_experts ||
        !experts->release_layer || !decoder->model_width)
        return -1;
    vector_bytes = (size_t)decoder->model_width * sizeof(float);
    after_attention = (float *)malloc(vector_bytes);
    dense_input = (float *)malloc(vector_bytes);
    dense_output = (float *)malloc(vector_bytes);
    dense_normalized = (float *)malloc(vector_bytes);
    expert_input = (float *)malloc(vector_bytes);
    expert_output = (float *)malloc(vector_bytes);
    expert_normalized = (float *)malloc(vector_bytes);
    combined = (float *)calloc((size_t)decoder->model_width, sizeof(*combined));
    ids = (uint32_t *)malloc((size_t)decoder->router.selected_count * sizeof(uint32_t));
    weights = (float *)malloc((size_t)decoder->router.selected_count * sizeof(float));
    if (!after_attention || !dense_input || !dense_output ||
        !dense_normalized || !expert_input || !expert_output ||
        !expert_normalized || !combined || !ids || !weights)
        goto cleanup;
    if (coli_gemma4_rmsnorm(attention_output, decoder->post_attention_norm,
                            decoder->model_width, decoder->gguf->rms_epsilon,
                            after_attention) != 0)
        goto cleanup;
    for (index = 0; index < decoder->model_width; ++index)
        after_attention[index] += residual[index];
    if (coli_gemma4_rmsnorm(after_attention, decoder->ffn_norm,
                            decoder->model_width, decoder->gguf->rms_epsilon,
                            dense_input) != 0 ||
        coli_gemma4_router_route(&decoder->router, after_attention, NULL,
                                 ids, weights, NULL) != 0 ||
        coli_gemma4_rmsnorm(after_attention, decoder->pre_ffw_norm_2,
                            decoder->model_width, decoder->gguf->rms_epsilon,
                            expert_input) != 0)
        goto cleanup;
    if (experts->prepare_layer(experts->ctx, decoder->layer, ids,
                               decoder->router.selected_count) != 0)
        goto cleanup;
    prepared = 1;
    if (coli_gemma4_dense_mlp_run(&decoder->dense_mlp, dense_input,
                                  dense_output) != 0 ||
        coli_gemma4_rmsnorm(dense_output, decoder->post_ffw_norm_1,
                            decoder->model_width, decoder->gguf->rms_epsilon,
                            dense_normalized) != 0)
        goto cleanup;
    if (experts->run_experts(experts->ctx, decoder->layer, ids, weights,
                             decoder->router.selected_count, expert_input,
                             expert_output) != 0)
        goto cleanup;
    experts->release_layer(experts->ctx, decoder->layer);
    prepared = 0;
    if (coli_gemma4_rmsnorm(expert_output, decoder->post_ffw_norm_2,
                            decoder->model_width, decoder->gguf->rms_epsilon,
                            expert_normalized) != 0)
        goto cleanup;
    for (index = 0; index < decoder->model_width; ++index)
        combined[index] = dense_normalized[index] + expert_normalized[index];
    if (coli_gemma4_rmsnorm(combined, decoder->post_ffw_norm,
                            decoder->model_width, decoder->gguf->rms_epsilon,
                            output) != 0)
        goto cleanup;
    for (index = 0; index < decoder->model_width; ++index)
        output[index] = (after_attention[index] + output[index]) *
                        decoder->layer_output_scale;
    status = 0;

cleanup:
    if (prepared) experts->release_layer(experts->ctx, decoder->layer);
    free(after_attention);
    free(dense_input); free(dense_output); free(dense_normalized);
    free(expert_input); free(expert_output); free(expert_normalized);
    free(combined); free(ids); free(weights);
    return status;
}

int coli_gemma4_decoder_layer_step(
    const coli_gemma4_decoder_layer *decoder, coli_gemma4_kv_cache *cache,
    const coli_expert_backend *experts, uint64_t position,
    const float *residual, float *output) {
    float *attention_output;
    size_t vector_bytes;
    int status;
    if (!decoder || !cache || !experts || !residual || !output ||
        !decoder->model_width) return -1;
    vector_bytes = (size_t)decoder->model_width * sizeof(float);
    attention_output = (float *)malloc(vector_bytes);
    if (!attention_output) return -1;
    status = coli_gemma4_attention_step(
        &decoder->attention, cache, position, residual, attention_output);
    if (status == 0)
        status = decoder_layer_complete(
            decoder, experts, residual, attention_output, output);
    free(attention_output);
    return status;
}

int coli_gemma4_decoder_layer_noncausal(
    const coli_gemma4_decoder_layer *decoder, coli_gemma4_kv_cache *cache,
    const coli_expert_backend *experts, uint64_t start_position,
    const float *residuals, uint32_t token_count, float *outputs) {
    float *attention_outputs = NULL;
    uint64_t value_count;
    uint32_t token;
    int status = -1;
    if (!decoder || !cache || !experts || !residuals || !outputs ||
        !decoder->model_width || !token_count) return -1;
    value_count = (uint64_t)token_count * decoder->model_width;
    if (value_count > SIZE_MAX / sizeof(float)) return -1;
    attention_outputs = (float *)malloc(
        (size_t)value_count * sizeof(float));
    if (!attention_outputs) return -1;
    if (coli_gemma4_attention_noncausal(
            &decoder->attention, cache, start_position, residuals,
            token_count, attention_outputs) != 0)
        goto cleanup;
    for (token = 0; token < token_count; ++token) {
        size_t offset = (size_t)token * decoder->model_width;
        if (decoder_layer_complete(
                decoder, experts, residuals + offset,
                attention_outputs + offset, outputs + offset) != 0)
            goto cleanup;
    }
    status = 0;
cleanup:
    free(attention_outputs);
    return status;
}

int coli_gemma4_router_open(coli_gemma4_router *router,
                            const coli_gemma4_gguf *gguf, uint32_t layer) {
    const coli_gemma4_tensor *input_scale, *projection, *expert_scale;
    char name[128];
    int length;
    if (!router || !gguf) return -1;
    memset(router, 0, sizeof(*router));
    router->gguf = gguf;
    router->layer = layer;
    router->width = gguf->config.n_embd;
    router->expert_count = gguf->config.n_expert;
    router->selected_count = gguf->config.n_expert_used;
    if (layer >= gguf->config.n_layer) {
        router_error(router, "router layer %u is outside the model", layer);
        return -1;
    }
    length = snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.scale", layer);
    if (length < 0 || (size_t)length >= sizeof(name)) return -1;
    input_scale = coli_gemma4_gguf_find(gguf, name);
    length = snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", layer);
    if (length < 0 || (size_t)length >= sizeof(name)) return -1;
    projection = coli_gemma4_gguf_find(gguf, name);
    length = snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.scale", layer);
    if (length < 0 || (size_t)length >= sizeof(name)) return -1;
    expert_scale = coli_gemma4_gguf_find(gguf, name);
    if (!tensor_is(input_scale, COLI_GGML_TYPE_F32, 1, router->width, 0) ||
        !tensor_is(projection, COLI_GGML_TYPE_F32, 2,
                   router->width, router->expert_count) ||
        !tensor_is(expert_scale, COLI_GGML_TYPE_F32, 1,
                   router->expert_count, 0)) {
        router_error(router, "layer %u router tensor types or dimensions disagree with metadata",
                     layer);
        return -1;
    }
    if (read_f32_tensor(router, input_scale, &router->input_scale) != 0 ||
        read_f32_tensor(router, projection, &router->projection) != 0 ||
        read_f32_tensor(router, expert_scale, &router->expert_scale) != 0) {
        char saved[COLI_GEMMA4_GGUF_ERROR_MAX];
        memcpy(saved, router->last_error, sizeof(saved));
        coli_gemma4_router_close(router);
        memcpy(router->last_error, saved, sizeof(saved));
        return -1;
    }
    return 0;
}

void coli_gemma4_router_close(coli_gemma4_router *router) {
    if (!router) return;
    free(router->input_scale);
    free(router->projection);
    free(router->expert_scale);
    memset(router, 0, sizeof(*router));
}

const char *coli_gemma4_router_last_error(const coli_gemma4_router *router) {
    return router ? router->last_error : "invalid Gemma router";
}

int coli_gemma4_rmsnorm(const float *input, const float *weight,
                        uint32_t width, float epsilon, float *output) {
    float sum_squares = 0.0F, inverse_rms;
    uint32_t i;
    if (!input || !weight || !output || !width ||
        !isfinite(epsilon) || epsilon <= 0.0F) return -1;
    for (i = 0; i < width; ++i) {
        if (!isfinite(input[i]) || !isfinite(weight[i])) return -1;
        sum_squares += input[i] * input[i];
    }
    inverse_rms = 1.0F / sqrtf(sum_squares / (float)width + epsilon);
    for (i = 0; i < width; ++i) output[i] = input[i] * inverse_rms * weight[i];
    return 0;
}

int coli_gemma4_router_route(const coli_gemma4_router *router,
                             const float *hidden_state,
                             float *probabilities,
                             uint32_t *selected_ids,
                             float *selected_weights,
                             float *effective_weights) {
    float *normalized = NULL, *scores = NULL;
    float sum_squares = 0.0F, inverse_rms, root_scale, maximum, probability_sum;
    float selected_sum = 0.0F;
    uint32_t width, experts, selected, i, e;
    if (!router || !hidden_state || !selected_ids || !selected_weights ||
        !router->input_scale || !router->projection || !router->expert_scale)
        return -1;
    width = router->width;
    experts = router->expert_count;
    selected = router->selected_count;
    normalized = (float *)malloc((size_t)width * sizeof(float));
    scores = (float *)malloc((size_t)experts * sizeof(float));
    if (!normalized || !scores) {
        free(normalized); free(scores);
        return -1;
    }
    for (i = 0; i < width; ++i) {
        if (!isfinite(hidden_state[i]) || !isfinite(router->input_scale[i])) {
            free(normalized); free(scores);
            return -1;
        }
        sum_squares += hidden_state[i] * hidden_state[i];
    }
    inverse_rms = 1.0F / sqrtf(sum_squares / (float)width + router->gguf->rms_epsilon);
    root_scale = 1.0F / sqrtf((float)width);
    for (i = 0; i < width; ++i)
        normalized[i] = hidden_state[i] * inverse_rms *
                        router->input_scale[i] * root_scale;
    for (e = 0; e < experts; ++e) {
        const float *row = router->projection + (size_t)e * width;
        float score = 0.0F;
        for (i = 0; i < width; ++i) score += row[i] * normalized[i];
        scores[e] = score;
    }
    maximum = scores[0];
    for (e = 1; e < experts; ++e) if (scores[e] > maximum) maximum = scores[e];
    probability_sum = 0.0F;
    for (e = 0; e < experts; ++e) {
        scores[e] = expf(scores[e] - maximum);
        probability_sum += scores[e];
    }
    if (!isfinite(probability_sum) || probability_sum <= 0.0F) {
        free(normalized); free(scores);
        return -1;
    }
    for (e = 0; e < experts; ++e) {
        scores[e] /= probability_sum;
        if (probabilities) probabilities[e] = scores[e];
    }
    for (i = 0; i < selected; ++i) {
        uint32_t best = UINT32_MAX;
        float best_value = -FLT_MAX;
        for (e = 0; e < experts; ++e) {
            uint32_t prior;
            int already_selected = 0;
            for (prior = 0; prior < i; ++prior)
                if (selected_ids[prior] == e) already_selected = 1;
            if (!already_selected &&
                (scores[e] > best_value ||
                 (scores[e] == best_value && (best == UINT32_MAX || e < best)))) {
                best = e;
                best_value = scores[e];
            }
        }
        if (best == UINT32_MAX) {
            free(normalized); free(scores);
            return -1;
        }
        selected_ids[i] = best;
        selected_weights[i] = best_value;
        selected_sum += best_value;
    }
    if (!isfinite(selected_sum) || selected_sum <= 0.0F) {
        free(normalized); free(scores);
        return -1;
    }
    for (i = 0; i < selected; ++i) {
        selected_weights[i] /= selected_sum;
        if (effective_weights)
            effective_weights[i] = selected_weights[i] *
                                   router->expert_scale[selected_ids[i]];
    }
    free(normalized);
    free(scores);
    return 0;
}
