#include "glm53_gpu.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#ifdef __cplusplus
extern "C" int coli_gpu_context_advertise_pipeline(ColiGpuContext *ctx);
#else
extern int coli_gpu_context_advertise_pipeline(ColiGpuContext *ctx);
#endif
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void set_err(char *err, size_t err_cap, const char *text) {
    if (!err || err_cap == 0) return;
    snprintf(err, err_cap, "%s", text ? text : "");
}

int coli_glm53_gpu_mode_parse(const char *text, ColiGlm53GpuMode *out) {
    if (!out) return 0;
    if (!text || !*text || !strcmp(text, "auto")) {
        *out = COLI_GLM53_GPU_MODE_AUTO;
        return 1;
    }
    if (!strcmp(text, "gpu")) {
        *out = COLI_GLM53_GPU_MODE_GPU;
        return 1;
    }
    if (!strcmp(text, "cpu")) {
        *out = COLI_GLM53_GPU_MODE_CPU;
        return 1;
    }
    return 0;
}

const char *coli_glm53_gpu_mode_name(ColiGlm53GpuMode mode) {
    switch (mode) {
    case COLI_GLM53_GPU_MODE_AUTO: return "auto";
    case COLI_GLM53_GPU_MODE_GPU: return "gpu";
    case COLI_GLM53_GPU_MODE_CPU: return "cpu";
    }
    return "unknown";
}

const char *coli_glm53_backend_name(ColiGlm53BackendKind backend) {
    switch (backend) {
    case COLI_GLM53_BACKEND_UNSELECTED: return "unselected";
    case COLI_GLM53_BACKEND_CPU: return "cpu";
    case COLI_GLM53_BACKEND_GPU: return "gpu";
    }
    return "unknown";
}

void coli_glm53_gpu_backend_init(ColiGlm53GpuBackend *backend) {
    if (!backend) return;
    memset(backend, 0, sizeof(*backend));
    backend->requested = COLI_GLM53_GPU_MODE_AUTO;
    backend->selected = COLI_GLM53_BACKEND_UNSELECTED;
    backend->device = -1;
}

int coli_glm53_gpu_backend_select(ColiGlm53GpuBackend *backend,
                                  ColiGlm53GpuMode mode,
                                  ColiGlm53GpuOps ops,
                                  int device,
                                  uint64_t required_caps,
                                  char *err,
                                  size_t err_cap) {
    ColiGpuContext *ctx = NULL;
    if (!backend) {
        set_err(err, err_cap, "missing backend state");
        return 0;
    }
    if (backend->active_requests > 0) {
        set_err(err, err_cap, "backend selection locked during active request");
        return 0;
    }
    if (backend->selected != COLI_GLM53_BACKEND_UNSELECTED) {
        set_err(err, err_cap, "backend selection already completed");
        return 0;
    }
    backend->requested = mode;
    if (mode == COLI_GLM53_GPU_MODE_CPU) {
        backend->selected = COLI_GLM53_BACKEND_CPU;
        backend->device = -1;
        set_err(err, err_cap, "");
        return 1;
    }
    if (!ops.context_create || !ops.context_probe || !ops.context_destroy) {
        if (mode == COLI_GLM53_GPU_MODE_AUTO) {
            backend->selected = COLI_GLM53_BACKEND_CPU;
            set_err(err, err_cap, "gpu context operations unavailable");
            return 1;
        }
        set_err(err, err_cap, "gpu context operations unavailable");
        return 0;
    }
    if (!ops.context_create(&ctx, device) || !ctx) {
        if (mode == COLI_GLM53_GPU_MODE_AUTO) {
            backend->selected = COLI_GLM53_BACKEND_CPU;
            set_err(err, err_cap, "gpu context create failed");
            return 1;
        }
        set_err(err, err_cap, "gpu context create failed");
        return 0;
    }
    if (!ops.context_probe(ctx, required_caps)) {
        ops.context_destroy(ctx);
        if (mode == COLI_GLM53_GPU_MODE_AUTO) {
            backend->selected = COLI_GLM53_BACKEND_CPU;
            set_err(err, err_cap, "gpu probe failed");
            return 1;
        }
        set_err(err, err_cap, "gpu probe failed");
        return 0;
    }
    backend->ctx = ctx;
    backend->device = device;
    backend->selected = COLI_GLM53_BACKEND_GPU;
    set_err(err, err_cap, "");
    return 1;
}

ColiGlm53BackendKind coli_glm53_gpu_backend_selected(const ColiGlm53GpuBackend *backend) {
    return backend ? backend->selected : COLI_GLM53_BACKEND_UNSELECTED;
}

ColiGpuContext *coli_glm53_gpu_backend_context(const ColiGlm53GpuBackend *backend) {
    return backend ? backend->ctx : NULL;
}

int coli_glm53_gpu_backend_begin_request(ColiGlm53GpuBackend *backend) {
    if (!backend || backend->selected == COLI_GLM53_BACKEND_UNSELECTED) return 0;
    backend->active_requests++;
    return 1;
}

int coli_glm53_gpu_backend_fail_request(ColiGlm53GpuBackend *backend) {
    if (!backend || backend->selected != COLI_GLM53_BACKEND_GPU)
        return 0;
    backend->selected = COLI_GLM53_BACKEND_CPU;
    backend->device = -1;
    return 1;
}

void coli_glm53_gpu_backend_end_request(ColiGlm53GpuBackend *backend) {
    if (!backend || backend->active_requests <= 0) return;
    backend->active_requests--;
}

void coli_glm53_gpu_backend_destroy(ColiGlm53GpuBackend *backend,
                                    ColiGlm53GpuOps ops) {
    if (!backend) return;
    if (backend->ctx && ops.context_destroy) ops.context_destroy(backend->ctx);
    coli_glm53_gpu_backend_init(backend);
}

#ifdef COLI_CUDA

typedef struct {
    ColiGpuTensor *fn;
    ColiGpuTensor *base;
    ColiGpuTensor *scale;
    ColiGpuTensor *norm_weight;
} ColiGlm53GpuSite;

typedef struct {
    ColiGpuKdaWeights weights;
} ColiGlm53GpuKdaLayer;

typedef struct {
    ColiGpuMlaWeights weights;
} ColiGlm53GpuMlaLayer;

typedef struct {
    ColiGpuTensor *router;
    ColiGpuTensor *correction_bias;
    ColiGpuTensor *shared_gate;
    ColiGpuTensor *shared_up;
    ColiGpuTensor *shared_down;
    ColiGpuMoeSharedWeights shared;
    ColiGpuExpertCache *cache;
} ColiGlm53GpuMoeLayer;

typedef struct {
    ColiGpuTensor *gate;
    ColiGpuTensor *up;
    ColiGpuTensor *down;
} ColiGlm53GpuDenseLayer;

struct ColiGlm53GpuModel {
    ColiGpuContext *ctx;
    int hidden;
    int streams;
    int vocab;
    int max_rows;
    int max_context;
    float norm_eps;
    float hc_eps;
    float swiglu_limit;
    ColiGpuTensor *embedding;
    ColiGpuTensor *final_norm;
    ColiGpuTensor *lm_head;
    ColiGlm53GpuSite *sites;
    int site_count;
    ColiGpuKdaConfig kda_config;
    ColiGlm53GpuKdaLayer *kda_layers;
    int kda_layer_count;
    ColiGpuMlaConfig mla_config;
    ColiGlm53GpuMlaLayer *mla_layers;
    int mla_layer_count;
    ColiGlm53GpuDenseLayer *dense_layers;
    int dense_layer_count;
    int dense_intermediate;
    ColiGpuRouteConfig route_config;
    ColiGpuExpertCacheConfig expert_config;
    ColiGlm53GpuMoeLayer *moe_layers;
    int moe_layer_count;
    ColiGlm53GpuLayerDesc *layers;
    int layer_count;
    ColiGlm53GpuExpertLoader expert_loader;
    void *expert_loader_user;
};

struct ColiGlm53GpuSession {
    ColiGlm53GpuModel *model;
    ColiGpuArena *arena;
    int max_rows;
    int max_context;
    int filled;
    int failed;
    int resident_rows;
    size_t token_ids;
    size_t streams_a;
    size_t streams_b;
    size_t current_streams;
    size_t next_streams;
    size_t collapsed;
    size_t normed;
    size_t branch;
    size_t post;
    size_t comb;
    size_t logits;
    size_t dense_gate;
    size_t dense_up;
    size_t kda_scratch;
    ColiGpuKdaState **kda_states;
    size_t mla_scratch;
    size_t mla_selected;
    ColiGpuMlaState **mla_states;
    ColiGpuRouter *router;
    size_t moe_input;
    size_t moe_output;
    size_t moe_scratch;
    int moe_route_layer;
    ColiGlm53GpuQualityCapture quality;
};

static size_t align256(size_t value) {
    return (value + 255u) & ~(size_t)255u;
}

static int create_desc_tensor(ColiGpuTensor **out, ColiGpuContext *ctx,
                              const ColiGlm53GpuWeightDesc *desc,
                              int rows, int columns);

static int create_tensor(ColiGpuTensor **out, ColiGpuContext *ctx,
                         const void *data, const float *scales, int format,
                         int rows, int columns, int group_size) {
    ColiGpuTensorDesc tensor = {
        data, scales, format, rows, columns, group_size
    };
    return coli_gpu_tensor_create(out, ctx, &tensor);
}

static int create_weight_or_f32(
    ColiGpuTensor **out, ColiGpuContext *ctx,
    const ColiGlm53GpuWeightDesc *weight, const float *f32,
    int rows, int columns) {
    if (weight && weight->data)
        return create_desc_tensor(out, ctx, weight, rows, columns);
    if (!f32) return 0;
    return create_tensor(out, ctx, f32, NULL, 0, rows, columns, 0);
}

static void destroy_kda_layer(ColiGlm53GpuKdaLayer *layer) {
    if (!layer) return;
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.o_norm);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.a_log);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.dt_bias);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.conv);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.beta_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.decay_b_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.decay_a_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.gate_b_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.gate_a_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.o_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.v_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.k_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.q_proj);
    memset(layer, 0, sizeof(*layer));
}

static int create_kda_layer(ColiGlm53GpuKdaLayer *owned,
                            ColiGpuContext *ctx,
                            const ColiGlm53GpuKdaLayerDesc *layer,
                            int hidden, int heads, int dim, int kernel) {
    const int projection = heads * dim;
    if (!owned || !layer ||
        (!layer->q_proj && !layer->q_proj_weight.data) ||
        (!layer->k_proj && !layer->k_proj_weight.data) ||
        (!layer->v_proj && !layer->v_proj_weight.data) ||
        (!layer->o_proj && !layer->o_proj_weight.data) ||
        (!layer->gate_a_proj && !layer->gate_a_proj_weight.data) ||
        (!layer->gate_b_proj && !layer->gate_b_proj_weight.data) ||
        (!layer->decay_a_proj && !layer->decay_a_proj_weight.data) ||
        (!layer->decay_b_proj && !layer->decay_b_proj_weight.data) ||
        (!layer->beta_proj && !layer->beta_proj_weight.data) || !layer->conv ||
        !layer->dt_bias || !layer->a_log || !layer->o_norm)
        return 0;
#define CREATE_KDA_WEIGHT(field, data, weight, rows, columns) do {          \
        ColiGpuTensor *tensor_ = NULL;                                      \
        if (!create_weight_or_f32(                                           \
                &tensor_, ctx, &(weight), data, rows, columns))             \
            goto fail;                                                      \
        owned->weights.field = tensor_;                                     \
    } while (0)
    CREATE_KDA_WEIGHT(q_proj, layer->q_proj, layer->q_proj_weight,
                      projection, hidden);
    CREATE_KDA_WEIGHT(k_proj, layer->k_proj, layer->k_proj_weight,
                      projection, hidden);
    CREATE_KDA_WEIGHT(v_proj, layer->v_proj, layer->v_proj_weight,
                      projection, hidden);
    CREATE_KDA_WEIGHT(o_proj, layer->o_proj, layer->o_proj_weight,
                      hidden, projection);
    CREATE_KDA_WEIGHT(gate_a_proj, layer->gate_a_proj,
                      layer->gate_a_proj_weight, dim, hidden);
    CREATE_KDA_WEIGHT(gate_b_proj, layer->gate_b_proj,
                      layer->gate_b_proj_weight, projection, dim);
    CREATE_KDA_WEIGHT(decay_a_proj, layer->decay_a_proj,
                      layer->decay_a_proj_weight, dim, hidden);
    CREATE_KDA_WEIGHT(decay_b_proj, layer->decay_b_proj,
                      layer->decay_b_proj_weight, projection, dim);
    CREATE_KDA_WEIGHT(beta_proj, layer->beta_proj,
                      layer->beta_proj_weight, heads, hidden);
    if (!create_tensor((ColiGpuTensor **)&owned->weights.conv, ctx,
                       layer->conv, NULL, 0, 3 * projection, kernel, 0) ||
        !create_tensor((ColiGpuTensor **)&owned->weights.dt_bias, ctx,
                       layer->dt_bias, NULL, 0, 1, projection, 0) ||
        !create_tensor((ColiGpuTensor **)&owned->weights.a_log, ctx,
                       layer->a_log, NULL, 0, 1, heads, 0) ||
        !create_tensor((ColiGpuTensor **)&owned->weights.o_norm, ctx,
                       layer->o_norm, NULL, 0, 1, dim, 0))
        goto fail;
#undef CREATE_KDA_WEIGHT
    return 1;
fail:
#undef CREATE_KDA_WEIGHT
    destroy_kda_layer(owned);
    return 0;
}

static void destroy_mla_layer(ColiGlm53GpuMlaLayer *layer) {
    if (!layer) return;
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.index_pool_gate);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.index_pool_ape);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.index_key_bias);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.index_key_norm);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.index_weight_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.index_k_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.index_q_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.o_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.kv_b_value);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.kv_b_key);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.kv_a_norm);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.kv_a_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.q_b_proj);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.q_a_norm);
    coli_gpu_tensor_destroy((ColiGpuTensor *)layer->weights.q_a_proj);
    memset(layer, 0, sizeof(*layer));
}

static int create_mla_layer(ColiGlm53GpuMlaLayer *owned,
                            ColiGpuContext *ctx,
                            const ColiGlm53GpuMlaLayerDesc *layer,
                            const ColiGpuMlaConfig *config) {
    if (!owned || !layer || !config ||
        (!layer->q_a_proj && !layer->q_a_proj_weight.data) ||
        !layer->q_a_norm ||
        (!layer->q_b_proj && !layer->q_b_proj_weight.data) ||
        (!layer->kv_a_proj && !layer->kv_a_proj_weight.data) ||
        !layer->kv_a_norm ||
        (!layer->kv_b_key && !layer->kv_b_key_weight.data) ||
        (!layer->kv_b_value && !layer->kv_b_value_weight.data) ||
        (!layer->o_proj && !layer->o_proj_weight.data) ||
        (!layer->index_q_proj && !layer->index_q_proj_weight.data) ||
        (!layer->index_k_proj && !layer->index_k_proj_weight.data) ||
        (!layer->index_weight_proj &&
         !layer->index_weight_proj_weight.data) ||
        !layer->index_key_norm ||
        !layer->index_key_bias || !layer->index_pool_ape ||
        (!layer->index_pool_gate && !layer->index_pool_gate_weight.data))
        return 0;
#define CREATE_MLA_WEIGHT(field, data, weight, rows, columns) do {          \
        ColiGpuTensor *tensor_ = NULL;                                      \
        if (!create_weight_or_f32(                                           \
                &tensor_, ctx, &(weight), data, rows, columns))             \
            goto fail;                                                      \
        owned->weights.field = tensor_;                                     \
    } while (0)
    CREATE_MLA_WEIGHT(q_a_proj, layer->q_a_proj, layer->q_a_proj_weight,
                      config->q_lora, config->hidden);
    CREATE_MLA_WEIGHT(q_b_proj, layer->q_b_proj, layer->q_b_proj_weight,
                      config->heads * config->qk_nope, config->q_lora);
    CREATE_MLA_WEIGHT(kv_a_proj, layer->kv_a_proj, layer->kv_a_proj_weight,
                      config->kv_lora, config->hidden);
    CREATE_MLA_WEIGHT(kv_b_key, layer->kv_b_key, layer->kv_b_key_weight,
                      config->heads * config->kv_lora, config->qk_nope);
    CREATE_MLA_WEIGHT(kv_b_value, layer->kv_b_value,
                      layer->kv_b_value_weight,
                      config->heads * config->value_dim, config->kv_lora);
    CREATE_MLA_WEIGHT(o_proj, layer->o_proj, layer->o_proj_weight,
                      config->hidden, config->heads * config->value_dim);
    CREATE_MLA_WEIGHT(index_q_proj, layer->index_q_proj,
                      layer->index_q_proj_weight,
                      config->index_heads * config->index_dim,
                      config->q_lora);
    CREATE_MLA_WEIGHT(index_k_proj, layer->index_k_proj,
                      layer->index_k_proj_weight,
                      config->index_dim, config->hidden);
    CREATE_MLA_WEIGHT(index_weight_proj, layer->index_weight_proj,
                      layer->index_weight_proj_weight,
                      config->index_heads, config->hidden);
    CREATE_MLA_WEIGHT(index_pool_gate, layer->index_pool_gate,
                      layer->index_pool_gate_weight,
                      config->index_dim, config->hidden);
    if (!create_tensor((ColiGpuTensor **)&owned->weights.q_a_norm, ctx,
                       layer->q_a_norm, NULL, 0, 1, config->q_lora, 0) ||
        !create_tensor((ColiGpuTensor **)&owned->weights.kv_a_norm, ctx,
                       layer->kv_a_norm, NULL, 0, 1, config->kv_lora, 0) ||
        !create_tensor((ColiGpuTensor **)&owned->weights.index_key_norm, ctx,
                       layer->index_key_norm, NULL, 0, 1,
                       config->index_dim, 0) ||
        !create_tensor((ColiGpuTensor **)&owned->weights.index_key_bias, ctx,
                       layer->index_key_bias, NULL, 0, 1,
                       config->index_dim, 0) ||
        !create_tensor((ColiGpuTensor **)&owned->weights.index_pool_ape, ctx,
                       layer->index_pool_ape, NULL, 0, config->index_pool,
                       config->index_dim, 0))
        goto fail;
#undef CREATE_MLA_WEIGHT
    return 1;
fail:
#undef CREATE_MLA_WEIGHT
    destroy_mla_layer(owned);
    return 0;
}

static int create_desc_tensor(ColiGpuTensor **out, ColiGpuContext *ctx,
                              const ColiGlm53GpuWeightDesc *desc,
                              int rows, int columns) {
    if (!desc || !desc->data || desc->rows != rows ||
        desc->columns != columns)
        return 0;
    return create_tensor(out, ctx, desc->data, desc->scales, desc->format,
                         desc->rows, desc->columns, desc->group_size);
}

static void destroy_moe_layer(ColiGlm53GpuMoeLayer *layer) {
    if (!layer) return;
    coli_gpu_expert_cache_destroy(layer->cache);
    coli_gpu_tensor_destroy(layer->shared_down);
    coli_gpu_tensor_destroy(layer->shared_up);
    coli_gpu_tensor_destroy(layer->shared_gate);
    coli_gpu_tensor_destroy(layer->correction_bias);
    coli_gpu_tensor_destroy(layer->router);
    memset(layer, 0, sizeof(*layer));
}

static int create_moe_layer(ColiGlm53GpuMoeLayer *owned,
                            ColiGpuContext *ctx,
                            const ColiGlm53GpuMoeLayerDesc *layer,
                            const ColiGpuExpertCacheConfig *config) {
    if (!owned || !layer || !config || !layer->router) return 0;
    if (!create_tensor(&owned->router, ctx, layer->router, NULL, 0,
                       config->experts, config->hidden, 0))
        goto fail;
    if (layer->correction_bias &&
        !create_tensor(&owned->correction_bias, ctx, layer->correction_bias,
                       NULL, 0, 1, config->experts, 0))
        goto fail;
    if (!create_desc_tensor(&owned->shared_gate, ctx, &layer->shared_gate,
                            config->intermediate, config->hidden) ||
        !create_desc_tensor(&owned->shared_up, ctx, &layer->shared_up,
                            config->intermediate, config->hidden) ||
        !create_desc_tensor(&owned->shared_down, ctx, &layer->shared_down,
                            config->hidden, config->intermediate) ||
        !coli_gpu_expert_cache_create(&owned->cache, ctx, config))
        goto fail;
    owned->shared.gate = owned->shared_gate;
    owned->shared.up = owned->shared_up;
    owned->shared.down = owned->shared_down;
    return 1;
fail:
    destroy_moe_layer(owned);
    return 0;
}

static void destroy_dense_layer(ColiGlm53GpuDenseLayer *layer) {
    if (!layer) return;
    coli_gpu_tensor_destroy(layer->down);
    coli_gpu_tensor_destroy(layer->up);
    coli_gpu_tensor_destroy(layer->gate);
    memset(layer, 0, sizeof(*layer));
}

static int create_dense_layer(
    ColiGlm53GpuDenseLayer *owned, ColiGpuContext *ctx,
    const ColiGlm53GpuDenseLayerDesc *layer,
    int hidden, int intermediate) {
    if (!owned || !layer ||
        !create_desc_tensor(&owned->gate, ctx, &layer->gate,
                            intermediate, hidden) ||
        !create_desc_tensor(&owned->up, ctx, &layer->up,
                            intermediate, hidden) ||
        !create_desc_tensor(&owned->down, ctx, &layer->down,
                            hidden, intermediate)) {
        destroy_dense_layer(owned);
        return 0;
    }
    return 1;
}

void coli_glm53_gpu_model_destroy(ColiGlm53GpuModel *model) {
    if (!model) return;
    if (model->moe_layers)
        for (int i = 0; i < model->moe_layer_count; ++i)
            destroy_moe_layer(&model->moe_layers[i]);
    free(model->moe_layers);
    if (model->dense_layers)
        for (int i = 0; i < model->dense_layer_count; ++i)
            destroy_dense_layer(&model->dense_layers[i]);
    free(model->dense_layers);
    if (model->mla_layers)
        for (int i = 0; i < model->mla_layer_count; ++i)
            destroy_mla_layer(&model->mla_layers[i]);
    free(model->mla_layers);
    if (model->kda_layers)
        for (int i = 0; i < model->kda_layer_count; ++i)
            destroy_kda_layer(&model->kda_layers[i]);
    free(model->kda_layers);
    if (model->sites) {
        for (int i = 0; i < model->site_count; ++i) {
            coli_gpu_tensor_destroy(model->sites[i].norm_weight);
            coli_gpu_tensor_destroy(model->sites[i].scale);
            coli_gpu_tensor_destroy(model->sites[i].base);
            coli_gpu_tensor_destroy(model->sites[i].fn);
        }
    }
    free(model->sites);
    free(model->layers);
    coli_gpu_tensor_destroy(model->lm_head);
    coli_gpu_tensor_destroy(model->final_norm);
    coli_gpu_tensor_destroy(model->embedding);
    free(model);
}

int coli_glm53_gpu_model_create(ColiGlm53GpuModel **out,
                                ColiGpuContext *ctx,
                                const ColiGlm53GpuModelDesc *desc) {
    if (!out) return 0;
    *out = NULL;
    if (!ctx || !coli_gpu_context_healthy(ctx) || !desc ||
        desc->hidden_size < 1 ||
        desc->stream_count < 1 || desc->stream_count > 8 ||
        desc->vocab_size < 1 || desc->max_prefill_rows < 1 ||
        desc->max_context_tokens < 0 ||
        desc->norm_eps < 0.0f || desc->hc_eps < 0.0f ||
        !desc->embedding || !desc->final_norm || !desc->lm_head ||
        desc->site_count < 0 ||
        (desc->site_count && !desc->sites) ||
        desc->kda_layer_count < 0 ||
        (desc->kda_layer_count &&
         (!desc->kda_layers || desc->kda_heads < 1 ||
          desc->kda_head_dim < 1 || desc->kda_head_dim > 512 ||
          desc->kda_kernel < 1 || desc->kda_kernel > 8)) ||
        desc->mla_layer_count < 0 ||
        (desc->mla_layer_count && !desc->mla_layers) ||
        desc->dense_layer_count < 0 ||
        (desc->dense_layer_count &&
         (!desc->dense_layers || desc->dense_intermediate < 1 ||
          !isfinite(desc->swiglu_limit) || desc->swiglu_limit <= 0.0f)) ||
        desc->moe_layer_count < 0 ||
        (desc->moe_layer_count &&
         (!desc->moe_layers || desc->moe_experts < 1 ||
          desc->moe_experts > 4096 || desc->moe_topk < 1 ||
          desc->moe_topk > desc->moe_experts ||
          desc->moe_intermediate < 1 || desc->moe_cache_slots < 1 ||
          desc->moe_cache_slots < desc->moe_topk ||
          desc->moe_cache_slots > desc->moe_experts ||
          desc->moe_group_size != 64 ||
          (desc->moe_normalize_topk != 0 &&
           desc->moe_normalize_topk != 1) ||
          !isfinite(desc->moe_routed_scale) ||
          !isfinite(desc->moe_swiglu_limit) ||
          desc->moe_swiglu_limit <= 0.0f)) ||
        desc->layer_count < 0 ||
        (desc->layer_count && !desc->layers))
        return 0;
    for (int i = 0; i < desc->layer_count; ++i) {
        const ColiGlm53GpuLayerDesc *layer = &desc->layers[i];
        if (layer->attention_site < 0 ||
            layer->attention_site >= desc->site_count ||
            layer->ffn_site < 0 || layer->ffn_site >= desc->site_count ||
            (layer->attention_kind == COLI_GLM53_GPU_ATTN_KDA
                 ? layer->attention_index < 0 ||
                       layer->attention_index >= desc->kda_layer_count
                 : layer->attention_kind == COLI_GLM53_GPU_ATTN_MLA
                     ? layer->attention_index < 0 ||
                           layer->attention_index >= desc->mla_layer_count
                     : 1) ||
            (layer->ffn_kind == COLI_GLM53_GPU_FFN_DENSE
                 ? layer->ffn_index < 0 ||
                       layer->ffn_index >= desc->dense_layer_count
                 : layer->ffn_kind == COLI_GLM53_GPU_FFN_MOE
                     ? layer->ffn_index < 0 ||
                           layer->ffn_index >= desc->moe_layer_count
                     : 1))
            return 0;
    }
    if (coli_gpu_context_consume_fault(
            ctx, COLI_GPU_FAULT_MODEL_ALLOCATION))
        return 0;
    ColiGpuKdaConfig kda_config;
    memset(&kda_config, 0, sizeof(kda_config));
    if (desc->kda_layer_count) {
        kda_config.heads = desc->kda_heads;
        kda_config.head_dim = desc->kda_head_dim;
        kda_config.kernel = desc->kda_kernel;
        kda_config.max_rows = desc->max_prefill_rows;
        kda_config.max_context = desc->max_context_tokens > 0
            ? desc->max_context_tokens : desc->max_prefill_rows;
        kda_config.recurrent_norm_eps = 1e-6f;
        kda_config.output_norm_eps = desc->norm_eps;
        kda_config.gate_lower_bound = desc->kda_gate_lower_bound;
        if (!coli_gpu_kda_state_bytes(&kda_config) ||
            !coli_gpu_kda_window_bytes(&kda_config) ||
            !coli_gpu_kda_scratch_bytes(
                &kda_config, desc->max_prefill_rows, desc->hidden_size))
            return 0;
    }
    ColiGpuMlaConfig mla_config;
    memset(&mla_config, 0, sizeof(mla_config));
    if (desc->mla_layer_count) {
        mla_config.hidden = desc->hidden_size;
        mla_config.heads = desc->mla_heads;
        mla_config.q_lora = desc->mla_q_lora;
        mla_config.kv_lora = desc->mla_kv_lora;
        mla_config.qk_nope = desc->mla_qk_nope;
        mla_config.qk_rope = desc->mla_qk_rope;
        mla_config.value_dim = desc->mla_value_dim;
        mla_config.index_heads = desc->mla_index_heads;
        mla_config.index_dim = desc->mla_index_dim;
        mla_config.index_pool = desc->mla_index_pool;
        mla_config.index_topk = desc->mla_index_topk;
        mla_config.index_select_tail = desc->mla_index_select_tail;
        mla_config.max_rows = desc->max_prefill_rows;
        mla_config.max_context = desc->max_context_tokens > 0
            ? desc->max_context_tokens : desc->max_prefill_rows;
        mla_config.page_tokens = desc->mla_page_tokens;
        mla_config.rms_norm_eps = desc->norm_eps;
        mla_config.index_norm_eps = 1e-5f;
        if (!coli_gpu_mla_scratch_bytes(&mla_config) ||
            !coli_gpu_mla_selected_bytes(
                &mla_config, desc->max_prefill_rows))
            return 0;
    }
    ColiGlm53GpuModel *model =
        (ColiGlm53GpuModel *)calloc(1, sizeof(*model));
    if (!model) return 0;
    model->ctx = ctx;
    model->hidden = desc->hidden_size;
    model->streams = desc->stream_count;
    model->vocab = desc->vocab_size;
    model->max_rows = desc->max_prefill_rows;
    model->max_context = desc->max_context_tokens > 0
        ? desc->max_context_tokens : desc->max_prefill_rows;
    model->norm_eps = desc->norm_eps;
    model->hc_eps = desc->hc_eps;
    model->swiglu_limit = desc->swiglu_limit;
    model->site_count = desc->site_count;
    model->kda_layer_count = desc->kda_layer_count;
    model->kda_config = kda_config;
    model->mla_layer_count = desc->mla_layer_count;
    model->mla_config = mla_config;
    model->dense_layer_count = desc->dense_layer_count;
    model->dense_intermediate = desc->dense_intermediate;
    model->moe_layer_count = desc->moe_layer_count;
    model->layer_count = desc->layer_count;
    model->expert_loader = desc->expert_loader;
    model->expert_loader_user = desc->expert_loader_user;
    if (model->layer_count) {
        model->layers = (ColiGlm53GpuLayerDesc *)malloc(
            (size_t)model->layer_count * sizeof(*model->layers));
        if (!model->layers) {
            coli_glm53_gpu_model_destroy(model);
            return 0;
        }
        memcpy(model->layers, desc->layers,
               (size_t)model->layer_count * sizeof(*model->layers));
    }
    if (model->moe_layer_count) {
        model->route_config.hidden = model->hidden;
        model->route_config.experts = desc->moe_experts;
        model->route_config.topk = desc->moe_topk;
        model->route_config.normalize_topk = desc->moe_normalize_topk;
        model->route_config.routed_scale = desc->moe_routed_scale;
        model->expert_config.experts = desc->moe_experts;
        model->expert_config.slots = desc->moe_cache_slots;
        model->expert_config.hidden = model->hidden;
        model->expert_config.intermediate = desc->moe_intermediate;
        model->expert_config.group_size = desc->moe_group_size;
        model->expert_config.max_rows = model->max_rows;
        model->expert_config.swiglu_limit = desc->moe_swiglu_limit;
    }
    if (!create_tensor(&model->embedding, ctx, desc->embedding, NULL, 0,
                       model->vocab, model->hidden, 0) ||
        !create_tensor(&model->final_norm, ctx, desc->final_norm, NULL, 0,
                       1, model->hidden, 0) ||
        !create_tensor(&model->lm_head, ctx, desc->lm_head,
                       desc->lm_head_scales, desc->lm_head_format,
                       model->vocab, model->hidden,
                       desc->lm_head_group_size)) {
        coli_glm53_gpu_model_destroy(model);
        return 0;
    }
    if (model->site_count) {
        model->sites =
            (ColiGlm53GpuSite *)calloc((size_t)model->site_count,
                                       sizeof(*model->sites));
        if (!model->sites) {
            coli_glm53_gpu_model_destroy(model);
            return 0;
        }
    }
    const int mix_count = (2 + model->streams) * model->streams;
    for (int i = 0; i < model->site_count; ++i) {
        const ColiGlm53GpuMhcSiteDesc *site = &desc->sites[i];
        ColiGlm53GpuSite *owned = &model->sites[i];
        if (!site->fn || !site->base || !site->scale || !site->norm_weight ||
            !create_tensor(&owned->fn, ctx, site->fn, NULL, 0, mix_count,
                           model->streams * model->hidden, 0) ||
            !create_tensor(&owned->base, ctx, site->base, NULL, 0, 1,
                           mix_count, 0) ||
            !create_tensor(&owned->scale, ctx, site->scale, NULL, 0, 1, 3, 0) ||
            !create_tensor(&owned->norm_weight, ctx, site->norm_weight, NULL, 0,
                           1, model->hidden, 0)) {
            coli_glm53_gpu_model_destroy(model);
            return 0;
        }
    }
    if (model->kda_layer_count) {
        model->kda_layers = (ColiGlm53GpuKdaLayer *)calloc(
            (size_t)model->kda_layer_count, sizeof(*model->kda_layers));
        if (!model->kda_layers) {
            coli_glm53_gpu_model_destroy(model);
            return 0;
        }
        for (int i = 0; i < model->kda_layer_count; ++i)
            if (!create_kda_layer(
                    &model->kda_layers[i], ctx, &desc->kda_layers[i],
                    model->hidden, model->kda_config.heads,
                    model->kda_config.head_dim, model->kda_config.kernel)) {
                coli_glm53_gpu_model_destroy(model);
                return 0;
            }
    }
    if (model->mla_layer_count) {
        model->mla_layers = (ColiGlm53GpuMlaLayer *)calloc(
            (size_t)model->mla_layer_count, sizeof(*model->mla_layers));
        if (!model->mla_layers) {
            coli_glm53_gpu_model_destroy(model);
            return 0;
        }
        for (int i = 0; i < model->mla_layer_count; ++i)
            if (!create_mla_layer(
                    &model->mla_layers[i], ctx, &desc->mla_layers[i],
                    &model->mla_config)) {
                coli_glm53_gpu_model_destroy(model);
                return 0;
            }
    }
    if (model->dense_layer_count) {
        model->dense_layers = (ColiGlm53GpuDenseLayer *)calloc(
            (size_t)model->dense_layer_count, sizeof(*model->dense_layers));
        if (!model->dense_layers) {
            coli_glm53_gpu_model_destroy(model);
            return 0;
        }
        for (int i = 0; i < model->dense_layer_count; ++i)
            if (!create_dense_layer(
                    &model->dense_layers[i], ctx, &desc->dense_layers[i],
                    model->hidden, model->dense_intermediate)) {
                coli_glm53_gpu_model_destroy(model);
                return 0;
            }
    }
    if (model->moe_layer_count) {
        model->moe_layers = (ColiGlm53GpuMoeLayer *)calloc(
            (size_t)model->moe_layer_count, sizeof(*model->moe_layers));
        if (!model->moe_layers) {
            coli_glm53_gpu_model_destroy(model);
            return 0;
        }
        for (int i = 0; i < model->moe_layer_count; ++i)
            if (!create_moe_layer(
                    &model->moe_layers[i], ctx, &desc->moe_layers[i],
                    &model->expert_config)) {
                coli_glm53_gpu_model_destroy(model);
                return 0;
            }
    }
    *out = model;
    return 1;
}

void coli_glm53_gpu_session_destroy(ColiGlm53GpuSession *session) {
    if (!session) return;
    if (session->mla_states)
        for (int i = 0; i < session->model->mla_layer_count; ++i)
            coli_gpu_mla_state_destroy(session->mla_states[i]);
    free(session->mla_states);
    if (session->kda_states)
        for (int i = 0; i < session->model->kda_layer_count; ++i)
            coli_gpu_kda_state_destroy(session->kda_states[i]);
    free(session->kda_states);
    coli_gpu_router_destroy(session->router);
    coli_gpu_arena_destroy(session->arena);
    free(session);
}

int coli_glm53_gpu_session_create(ColiGlm53GpuSession **out,
                                  ColiGlm53GpuModel *model,
                                  int max_context) {
    if (!out) return 0;
    *out = NULL;
    if (!model || !coli_gpu_context_healthy(model->ctx) ||
        max_context < 1 || max_context > model->max_context ||
        coli_gpu_context_consume_fault(
            model->ctx, COLI_GPU_FAULT_SESSION_ALLOCATION))
        return 0;
    ColiGlm53GpuSession *session =
        (ColiGlm53GpuSession *)calloc(1, sizeof(*session));
    if (!session) return 0;
    session->model = model;
    session->max_rows = model->max_rows;
    session->max_context = max_context;
    const int max_rows = session->max_rows;
    size_t row_values = (size_t)max_rows * model->hidden;
    size_t stream_values = row_values * model->streams;
    session->token_ids = 0;
    session->streams_a =
        align256((size_t)max_rows * sizeof(int32_t));
    session->streams_b =
        session->streams_a + align256(stream_values * sizeof(float));
    session->collapsed =
        session->streams_b + align256(stream_values * sizeof(float));
    session->normed =
        session->collapsed + align256(row_values * sizeof(float));
    session->branch =
        session->normed + align256(row_values * sizeof(float));
    session->post =
        session->branch + align256(row_values * sizeof(float));
    session->comb =
        session->post + align256((size_t)max_rows * model->streams *
                                 sizeof(float));
    session->logits =
        session->comb + align256((size_t)max_rows * model->streams *
                                 model->streams * sizeof(float));
    size_t capacity = session->logits +
        (size_t)max_rows * model->vocab * sizeof(float);
    if (model->dense_layer_count) {
        session->dense_gate = align256(capacity);
        session->dense_up = session->dense_gate +
            align256((size_t)max_rows * model->dense_intermediate *
                     sizeof(float));
        capacity = session->dense_up +
            align256((size_t)max_rows * model->dense_intermediate *
                     sizeof(float));
    }
    if (model->kda_layer_count) {
        session->kda_scratch = align256(capacity);
        size_t scratch_bytes = coli_gpu_kda_scratch_bytes(
            &model->kda_config, max_rows, model->hidden);
        size_t state_bytes = coli_gpu_kda_state_bytes(&model->kda_config);
        size_t window_bytes = coli_gpu_kda_window_bytes(&model->kda_config);
        if (!scratch_bytes || !state_bytes || !window_bytes) {
            free(session);
            return 0;
        }
        capacity = session->kda_scratch + align256(scratch_bytes);
        for (int i = 0; i < model->kda_layer_count; ++i) {
            capacity += align256(state_bytes);
            capacity += align256(window_bytes);
        }
    }
    if (model->mla_layer_count) {
        session->mla_scratch = align256(capacity);
        capacity = session->mla_scratch +
            align256(coli_gpu_mla_scratch_bytes(&model->mla_config));
        session->mla_selected = capacity;
        capacity += coli_gpu_mla_selected_bytes(
            &model->mla_config, max_rows);
    }
    if (model->moe_layer_count) {
        session->moe_input = align256(capacity);
        session->moe_output = session->moe_input +
            align256((size_t)max_rows * model->hidden * sizeof(float));
        session->moe_scratch = session->moe_output +
            align256((size_t)max_rows * model->hidden * sizeof(float));
        capacity = session->moe_scratch +
            align256(coli_gpu_moe_scratch_bytes(&model->expert_config));
    }
    if (capacity < session->logits ||
        !coli_gpu_arena_create(&session->arena, model->ctx, capacity)) {
        free(session);
        return 0;
    }
    if (model->kda_layer_count) {
        session->kda_states = (ColiGpuKdaState **)calloc(
            (size_t)model->kda_layer_count, sizeof(*session->kda_states));
        if (!session->kda_states) {
            coli_glm53_gpu_session_destroy(session);
            return 0;
        }
        size_t cursor = session->kda_scratch + align256(
            coli_gpu_kda_scratch_bytes(&model->kda_config, max_rows,
                                       model->hidden));
        size_t state_bytes = coli_gpu_kda_state_bytes(&model->kda_config);
        size_t window_bytes = coli_gpu_kda_window_bytes(&model->kda_config);
        for (int i = 0; i < model->kda_layer_count; ++i) {
            size_t state_offset = cursor;
            cursor += align256(state_bytes);
            size_t window_offset = cursor;
            cursor += align256(window_bytes);
            if (!coli_gpu_kda_state_create(
                    &session->kda_states[i], model->ctx, session->arena,
                    state_offset, window_offset, &model->kda_config) ||
                !coli_gpu_kda_state_reset(session->kda_states[i])) {
                coli_glm53_gpu_session_destroy(session);
                return 0;
            }
        }
    }
    if (model->mla_layer_count) {
        session->mla_states = (ColiGpuMlaState **)calloc(
            (size_t)model->mla_layer_count, sizeof(*session->mla_states));
        if (!session->mla_states) {
            coli_glm53_gpu_session_destroy(session);
            return 0;
        }
        for (int i = 0; i < model->mla_layer_count; ++i)
            if (!coli_gpu_mla_state_create(
                    &session->mla_states[i], model->ctx,
                    &model->mla_config) ||
                (model->layer_count &&
                 !coli_gpu_mla_state_reserve(
                     session->mla_states[i], max_context))) {
                coli_glm53_gpu_session_destroy(session);
                return 0;
            }
    }
    if (model->moe_layer_count &&
        !coli_gpu_router_create(&session->router, model->ctx,
                                &model->route_config, max_rows)) {
        coli_glm53_gpu_session_destroy(session);
        return 0;
    }
    session->moe_route_layer = -1;
    session->current_streams = session->streams_a;
    session->next_streams = session->streams_b;
    *out = session;
    return 1;
}

int coli_glm53_gpu_session_embed(ColiGlm53GpuSession *session,
                                 const int32_t *token_ids, int rows) {
    if (!session || !token_ids || rows < 1 || rows > session->max_rows)
        return 0;
    for (int row = 0; row < rows; ++row)
        if (token_ids[row] < 0 || token_ids[row] >= session->model->vocab)
            return 0;
    session->current_streams = session->streams_a;
    session->next_streams = session->streams_b;
    if (!coli_gpu_arena_upload(session->arena, session->token_ids, token_ids,
                               (size_t)rows * sizeof(*token_ids)) ||
        !coli_gpu_embedding(session->arena, session->current_streams,
                            session->token_ids, session->model->embedding,
                            rows, session->model->streams,
                            session->model->hidden))
        return 0;
    session->resident_rows = rows;
    return 1;
}

int coli_glm53_gpu_session_core_site(ColiGlm53GpuSession *session,
                                     int site_index, int rows) {
    if (!session || rows < 1 || rows > session->max_rows ||
        rows != session->resident_rows || site_index < 0 ||
        site_index >= session->model->site_count)
        return 0;
    ColiGlm53GpuModel *model = session->model;
    ColiGlm53GpuSite *site = &model->sites[site_index];
    /* The standalone resident-core oracle uses the normalized pre-projection
     * as its deterministic branch. KDA/MLA/MoE branch production belongs to
     * later tasks; the mHC pre/post operation itself is complete here. */
    if (!coli_gpu_mhc_site(
            session->arena, session->next_streams, session->collapsed,
            session->normed, session->post, session->comb,
            session->current_streams, session->normed,
            site->fn, site->scale, site->base, site->norm_weight,
            rows, model->streams, model->hidden,
            model->norm_eps, model->hc_eps))
        return 0;
    size_t swap = session->current_streams;
    session->current_streams = session->next_streams;
    session->next_streams = swap;
    return 1;
}

int coli_glm53_gpu_session_kda_upload_input(ColiGlm53GpuSession *session,
                                            const float *input, int rows) {
    if (!session || !input || rows < 1 || rows > session->max_rows ||
        session->model->kda_layer_count < 1)
        return 0;
    size_t bytes = (size_t)rows * session->model->hidden * sizeof(float);
    if (!coli_gpu_arena_upload_activation(
            session->arena, session->normed, input, bytes))
        return 0;
    session->resident_rows = rows;
    return 1;
}

int coli_glm53_gpu_session_kda_site(ColiGlm53GpuSession *session,
                                    int layer_index, int rows,
                                    int start_position) {
    if (!session || layer_index < 0 ||
        layer_index >= session->model->kda_layer_count ||
        rows < 1 || rows > session->max_rows ||
        rows != session->resident_rows)
        return 0;
    ColiGlm53GpuModel *model = session->model;
    return coli_gpu_kda_site(
        session->arena, session->collapsed, session->normed,
        session->kda_scratch, session->kda_states[layer_index],
        &model->kda_layers[layer_index].weights, rows, start_position,
        model->hidden);
}

int coli_glm53_gpu_session_kda_download_output(
    ColiGlm53GpuSession *session, float *output, int rows) {
    if (!session || !output || rows < 1 || rows > session->max_rows ||
        rows != session->resident_rows)
        return 0;
    return coli_gpu_arena_download_activation(
        session->arena, session->collapsed, output,
        (size_t)rows * session->model->hidden * sizeof(float));
}

int coli_glm53_gpu_session_kda_state_download(
    ColiGlm53GpuSession *session, int layer_index,
    float *matrix, size_t matrix_floats,
    float *window, size_t window_floats) {
    if (!session || layer_index < 0 ||
        layer_index >= session->model->kda_layer_count)
        return 0;
    return coli_gpu_kda_state_download(
        session->kda_states[layer_index], matrix, matrix_floats,
        window, window_floats);
}

int coli_glm53_gpu_session_mla_upload_input(
    ColiGlm53GpuSession *session, const float *input, int rows) {
    if (!session || !input || rows < 1 || rows > session->max_rows ||
        session->model->mla_layer_count < 1)
        return 0;
    size_t values = (size_t)rows * session->model->hidden;
    for (size_t i = 0; i < values; ++i)
        if (!isfinite(input[i])) return 0;
    size_t bytes = (size_t)rows * session->model->hidden * sizeof(float);
    if (!coli_gpu_arena_upload_activation(
            session->arena, session->normed, input, bytes))
        return 0;
    session->resident_rows = rows;
    return 1;
}

int coli_glm53_gpu_session_mla_site(
    ColiGlm53GpuSession *session, int layer_index, int rows,
    int start_position) {
    if (!session || layer_index < 0 ||
        layer_index >= session->model->mla_layer_count ||
        rows < 1 || rows > session->max_rows ||
        rows != session->resident_rows)
        return 0;
    return coli_gpu_mla_site(
        session->arena, session->collapsed, session->normed,
        session->mla_scratch, session->mla_selected,
        session->mla_states[layer_index],
        &session->model->mla_layers[layer_index].weights,
        rows, start_position);
}

int coli_glm53_gpu_session_mla_download_output(
    ColiGlm53GpuSession *session, float *output,
    int *selected, size_t selected_count, int rows) {
    if (!session || !output || !selected ||
        rows < 1 || rows > session->max_rows ||
        rows != session->resident_rows)
        return 0;
    size_t output_bytes =
        (size_t)rows * session->model->hidden * sizeof(float);
    size_t selected_bytes =
        coli_gpu_mla_selected_bytes(&session->model->mla_config, rows);
    if (!selected_bytes || selected_count != selected_bytes / sizeof(int))
        return 0;
    return coli_gpu_arena_download_activation(
               session->arena, session->collapsed, output, output_bytes) &&
           coli_gpu_arena_download(
               session->arena, session->mla_selected,
               selected, selected_bytes);
}

int coli_glm53_gpu_session_mla_state_info(
    ColiGlm53GpuSession *session, int layer_index,
    int *logical_length, int *capacity) {
    if (!session || !logical_length || !capacity ||
        layer_index < 0 || layer_index >= session->model->mla_layer_count)
        return 0;
    *logical_length =
        coli_gpu_mla_state_length(session->mla_states[layer_index]);
    *capacity =
        coli_gpu_mla_state_capacity(session->mla_states[layer_index]);
    return 1;
}

int coli_glm53_gpu_session_mla_state_download(
    ColiGlm53GpuSession *session, int layer_index,
    float *latent, size_t latent_floats,
    float *index_keys, size_t index_key_floats,
    float *index_gates, size_t index_gate_floats) {
    if (!session || layer_index < 0 ||
        layer_index >= session->model->mla_layer_count)
        return 0;
    return coli_gpu_mla_state_download(
        session->mla_states[layer_index],
        latent, latent_floats,
        index_keys, index_key_floats,
        index_gates, index_gate_floats);
}

int coli_glm53_gpu_session_mla_reset_layer(
    ColiGlm53GpuSession *session, int layer_index) {
    if (!session || layer_index < 0 ||
        layer_index >= session->model->mla_layer_count)
        return 0;
    return coli_gpu_mla_state_reset(session->mla_states[layer_index]);
}

int coli_glm53_gpu_model_moe_upload_expert(
    ColiGlm53GpuModel *model, int layer_index, int expert_id, int slot,
    const ColiGlm53GpuExpertDesc *expert,
    ColiGlm53GpuExpertHandle *handle) {
    if (handle) memset(handle, 0, sizeof(*handle));
    if (!model || !expert || !handle || layer_index < 0 ||
        layer_index >= model->moe_layer_count)
        return 0;
    ColiGpuExpertSource source;
    source.gate = (ColiGpuTensorDesc){
        expert->gate.data, expert->gate.scales, expert->gate.format,
        expert->gate.rows, expert->gate.columns, expert->gate.group_size
    };
    source.up = (ColiGpuTensorDesc){
        expert->up.data, expert->up.scales, expert->up.format,
        expert->up.rows, expert->up.columns, expert->up.group_size
    };
    source.down = (ColiGpuTensorDesc){
        expert->down.data, expert->down.scales, expert->down.format,
        expert->down.rows, expert->down.columns, expert->down.group_size
    };
    ColiGpuExpertHandle backend_handle;
    if (!coli_gpu_expert_cache_upload(
            model->moe_layers[layer_index].cache, expert_id, slot,
            &source, &backend_handle))
        return 0;
    handle->expert_id = backend_handle.expert_id;
    handle->slot = backend_handle.slot;
    handle->generation = backend_handle.generation;
    return 1;
}

int coli_glm53_gpu_session_moe_upload_input(
    ColiGlm53GpuSession *session, const float *input, int rows) {
    if (!session || !input || rows < 1 || rows > session->max_rows ||
        session->model->moe_layer_count < 1)
        return 0;
    size_t bytes = (size_t)rows * session->model->hidden * sizeof(float);
    if (!coli_gpu_arena_upload_activation(
            session->arena, session->moe_input, input, bytes))
        return 0;
    session->resident_rows = rows;
    session->moe_route_layer = -1;
    return 1;
}

int coli_glm53_gpu_session_moe_route(
    ColiGlm53GpuSession *session, int layer_index, int rows) {
    if (!session || rows < 1 || rows != session->resident_rows ||
        layer_index < 0 || layer_index >= session->model->moe_layer_count)
        return 0;
    ColiGlm53GpuMoeLayer *layer =
        &session->model->moe_layers[layer_index];
    if (!coli_gpu_router_run(
            session->router, session->arena, session->moe_input,
            layer->router, layer->correction_bias, rows))
        return 0;
    session->moe_route_layer = layer_index;
    return 1;
}

int coli_glm53_gpu_session_moe_selected(
    ColiGlm53GpuSession *session, int *selected_ids, float *routing_weights,
    size_t selected_count, int rows) {
    if (!session || session->moe_route_layer < 0 ||
        rows != session->resident_rows)
        return 0;
    return coli_gpu_router_download(
        session->router, selected_ids, routing_weights, selected_count, rows);
}

int coli_glm53_gpu_session_moe_site(
    ColiGlm53GpuSession *session, int layer_index,
    const ColiGlm53GpuExpertHandle *handles, size_t handle_count, int rows) {
    if (!session || layer_index < 0 ||
        layer_index >= session->model->moe_layer_count ||
        layer_index != session->moe_route_layer ||
        rows != session->resident_rows)
        return 0;
    ColiGlm53GpuMoeLayer *layer =
        &session->model->moe_layers[layer_index];
    return coli_gpu_moe_site(
        session->arena, session->moe_output, session->moe_input,
        session->moe_scratch, session->router, layer->cache, &layer->shared,
        (const ColiGpuExpertHandle *)handles, handle_count, rows);
}

int coli_glm53_gpu_session_moe_download_output(
    ColiGlm53GpuSession *session, float *output, int rows) {
    if (!session || !output || rows < 1 || rows != session->resident_rows ||
        session->moe_route_layer < 0)
        return 0;
    return coli_gpu_arena_download_activation(
        session->arena, session->moe_output, output,
        (size_t)rows * session->model->hidden * sizeof(float));
}

int coli_glm53_gpu_session_reset(ColiGlm53GpuSession *session) {
    if (!session || !coli_gpu_context_healthy(session->model->ctx)) return 0;
    for (int i = 0; i < session->model->mla_layer_count; ++i)
        if (!coli_gpu_mla_state_reset(session->mla_states[i])) return 0;
    for (int i = 0; i < session->model->kda_layer_count; ++i)
        if (!coli_gpu_kda_state_reset(session->kda_states[i])) return 0;
    session->resident_rows = 0;
    session->filled = 0;
    session->failed = 0;
    session->moe_route_layer = -1;
    session->current_streams = session->streams_a;
    session->next_streams = session->streams_b;
    return 1;
}

int coli_glm53_gpu_session_set_quality_capture(
    ColiGlm53GpuSession *session,
    const ColiGlm53GpuQualityCapture *capture) {
    if (!session) return 0;
    memset(&session->quality, 0, sizeof(session->quality));
    if (!capture) return 1;
    if (capture->output_floats != (size_t)session->model->hidden)
        return 0;
    session->quality = *capture;
    return 1;
}

static int pipeline_quality_download(
    ColiGlm53GpuSession *session, int layer, int attention,
    int rows) {
    ColiGlm53GpuQualityCapture *q = &session->quality;
    float *out = NULL;
    if (attention && layer == q->kda_layer) out = q->kda_output;
    if (attention && layer == q->mla_layer) out = q->mla_output;
    if (!attention && layer == q->dense_layer) out = q->dense_output;
    if (!attention && layer == q->moe_layer) out = q->moe_output;
    if (!out) return 1;
    return coli_gpu_arena_download(
        session->arena,
        session->branch +
            (size_t)(rows - 1) * session->model->hidden * sizeof(float),
        out, (size_t)session->model->hidden * sizeof(float));
}

int coli_glm53_gpu_session_output(ColiGlm53GpuSession *session,
                                  float *logits, int rows) {
    if (!session || !logits || rows < 1 || rows > session->max_rows ||
        rows != session->resident_rows)
        return 0;
    ColiGlm53GpuModel *model = session->model;
    size_t bytes = (size_t)rows * model->vocab * sizeof(float);
    return coli_gpu_collapse_streams(
               session->arena, session->collapsed, session->current_streams,
               rows, model->streams, model->hidden) &&
           coli_gpu_rmsnorm(
               session->arena, session->normed, session->collapsed,
               model->final_norm, rows, model->hidden, model->norm_eps) &&
           coli_gpu_projection(
               session->arena, session->logits, session->normed,
               model->lm_head, rows, model->hidden, model->vocab) &&
           coli_gpu_arena_download_activation(
               session->arena, session->logits, logits, bytes);
}

static int pipeline_fail_at(
    ColiGlm53GpuSession *session, const char *stage) {
    if (session) {
        session->failed = 1;
        coli_gpu_context_mark_unhealthy(session->model->ctx);
    }
    fprintf(stderr, "GLM53 GPU request failed at %s\n",
            stage ? stage : "unknown stage");
    return 0;
}

static int pipeline_kernel_fault(ColiGlm53GpuSession *session) {
    return coli_gpu_context_consume_fault(
        session->model->ctx, COLI_GPU_FAULT_KERNEL_LAUNCH);
}

static int pipeline_mhc_pre(
    ColiGlm53GpuSession *session, int site_index, int rows) {
    ColiGlm53GpuModel *model = session->model;
    ColiGlm53GpuSite *site = &model->sites[site_index];
    return !pipeline_kernel_fault(session) &&
           coli_gpu_mhc_pre_norm(
               session->arena, session->collapsed, session->normed,
               session->post, session->comb, session->current_streams,
               site->fn, site->scale, site->base, site->norm_weight,
               rows, model->streams, model->hidden,
               model->norm_eps, model->hc_eps);
}

static int pipeline_mhc_post(ColiGlm53GpuSession *session, int rows) {
    ColiGlm53GpuModel *model = session->model;
    if (pipeline_kernel_fault(session) ||
        !coli_gpu_mhc_post(
            session->arena, session->next_streams, session->branch,
            session->current_streams, session->post, session->comb,
            rows, model->streams, model->hidden))
        return 0;
    size_t swap = session->current_streams;
    session->current_streams = session->next_streams;
    session->next_streams = swap;
    return 1;
}

static int pipeline_moe(
    ColiGlm53GpuSession *session, int pipeline_layer,
    int moe_index, int rows) {
    ColiGlm53GpuModel *model = session->model;
    ColiGlm53GpuMoeLayer *layer = &model->moe_layers[moe_index];
    if (pipeline_kernel_fault(session) ||
        !coli_gpu_router_run(session->router, session->arena, session->normed,
                             layer->router, layer->correction_bias, rows))
        return 0;
    size_t selected_count =
        (size_t)rows * (size_t)model->route_config.topk;
    int *selected = (int *)malloc(selected_count * sizeof(*selected));
    float *weights = (float *)malloc(selected_count * sizeof(*weights));
    ColiGpuExpertHandle *handles = (ColiGpuExpertHandle *)malloc(
        selected_count * sizeof(*handles));
    if (!selected || !weights || !handles ||
        !coli_gpu_router_download(session->router, selected, weights,
                                  selected_count, rows)) {
        free(handles);
        free(weights);
        free(selected);
        return 0;
    }
    int ok = 1;
    for (size_t i = 0; ok && i < selected_count; ++i) {
        int expert_id = selected[i];
        ColiGlm53GpuExpertDesc source;
        int slot = -1;
        memset(&source, 0, sizeof(source));
        if (!model->expert_loader ||
            !model->expert_loader(model->expert_loader_user,
                                  pipeline_layer, expert_id,
                                  &slot, &source))
            ok = 0;
        else if (coli_gpu_expert_cache_lookup(
                     layer->cache, expert_id, &handles[i])) {
            if (handles[i].slot != slot) ok = 0;
        } else if (
            coli_gpu_context_consume_fault(
                model->ctx, COLI_GPU_FAULT_EXPERT_PUBLICATION) ||
            !coli_glm53_gpu_model_moe_upload_expert(
                model, moe_index, expert_id, slot, &source,
                (ColiGlm53GpuExpertHandle *)&handles[i]))
            ok = 0;
    }
    if (ok && !pipeline_kernel_fault(session))
        ok = coli_gpu_moe_site(
            session->arena, session->branch, session->normed,
            session->moe_scratch, session->router, layer->cache,
            &layer->shared, handles, selected_count, rows);
    free(handles);
    free(weights);
    free(selected);
    return ok;
}

int coli_glm53_gpu_forward(ColiGlm53GpuSession *session,
                           const int *token_ids, int rows,
                           float *last_logits_host) {
    if (!session || !token_ids || !last_logits_host || rows < 1 ||
        rows > session->max_rows || session->failed ||
        session->filled > session->max_context - rows ||
        !coli_gpu_context_healthy(session->model->ctx))
        return 0;
    ColiGlm53GpuModel *model = session->model;
    if (coli_gpu_context_consume_fault(
            model->ctx, COLI_GPU_FAULT_TENSOR_UPLOAD) ||
        !coli_glm53_gpu_session_embed(
            session, (const int32_t *)token_ids, rows))
        return pipeline_fail_at(session, "token upload/embedding");

    for (int i = 0; i < model->layer_count; ++i) {
        const ColiGlm53GpuLayerDesc *layer = &model->layers[i];
        if (!pipeline_mhc_pre(session, layer->attention_site, rows))
            return pipeline_fail_at(session, "attention mHC pre");
        if (pipeline_kernel_fault(session))
            return pipeline_fail_at(session, "attention launch");
        int attention_ok =
            layer->attention_kind == COLI_GLM53_GPU_ATTN_KDA
                ? coli_gpu_kda_site(
                      session->arena, session->branch, session->normed,
                      session->kda_scratch,
                      session->kda_states[layer->attention_index],
                      &model->kda_layers[layer->attention_index].weights,
                      rows, session->filled, model->hidden)
                : coli_gpu_mla_site(
                      session->arena, session->branch, session->normed,
                      session->mla_scratch, session->mla_selected,
                      session->mla_states[layer->attention_index],
                      &model->mla_layers[layer->attention_index].weights,
                      rows, session->filled);
        if (!attention_ok ||
            !pipeline_quality_download(session, i, 1, rows) ||
            !pipeline_mhc_post(session, rows) ||
            !pipeline_mhc_pre(session, layer->ffn_site, rows))
            return pipeline_fail_at(session, "attention/mHC post");

        int ffn_ok;
        if (layer->ffn_kind == COLI_GLM53_GPU_FFN_DENSE) {
            ColiGlm53GpuDenseLayer *dense =
                &model->dense_layers[layer->ffn_index];
            ffn_ok = !pipeline_kernel_fault(session) &&
                coli_gpu_dense_mlp(
                    session->arena, session->branch, session->normed,
                    session->dense_gate, session->dense_up,
                    dense->gate, dense->up, dense->down,
                    rows, model->hidden, model->dense_intermediate,
                    model->swiglu_limit);
        } else {
            ffn_ok = pipeline_moe(
                session, i, layer->ffn_index, rows);
        }
        if (!ffn_ok) {
            fprintf(stderr, "GLM53 GPU FFN failed at layer %d kind %d\n",
                    i, (int)layer->ffn_kind);
            return pipeline_fail_at(session, "FFN");
        }
        if (!pipeline_quality_download(session, i, 0, rows) ||
            !pipeline_mhc_post(session, rows))
            return pipeline_fail_at(session, "FFN/mHC post");
    }

    size_t bytes = (size_t)model->vocab * sizeof(float);
    float *complete = (float *)malloc(bytes);
    if (!complete || pipeline_kernel_fault(session) ||
        !coli_gpu_collapse_streams(
            session->arena, session->collapsed, session->current_streams,
            rows, model->streams, model->hidden) ||
        pipeline_kernel_fault(session) ||
        !coli_gpu_rmsnorm(
            session->arena, session->normed, session->collapsed,
            model->final_norm, rows, model->hidden, model->norm_eps) ||
        (session->quality.final_norm &&
         !coli_gpu_arena_download(
             session->arena,
             session->normed +
                 (size_t)(rows - 1) * model->hidden * sizeof(float),
             session->quality.final_norm,
             (size_t)model->hidden * sizeof(float))) ||
        pipeline_kernel_fault(session) ||
        !coli_gpu_projection(
            session->arena, session->logits, session->normed,
            model->lm_head, rows, model->hidden, model->vocab) ||
        coli_gpu_context_consume_fault(
            model->ctx, COLI_GPU_FAULT_DEVICE_STATUS) ||
        !coli_gpu_context_sync(model->ctx) ||
        !coli_gpu_arena_download_activation(
            session->arena,
            session->logits +
                (size_t)(rows - 1) * model->vocab * sizeof(float),
            complete, bytes)) {
        free(complete);
        return pipeline_fail_at(session, "final output/synchronization");
    }
    for (int token = 0; token < model->vocab; ++token) {
        if (!isfinite(complete[token])) {
            free(complete);
            return pipeline_fail_at(session, "non-finite final logits");
        }
    }
    memcpy(last_logits_host, complete, bytes);
    free(complete);
    session->filled += rows;
    return 1;
}

static int probe_expert_loader(
    void *user, int pipeline_layer, int expert_id, int *slot,
    ColiGlm53GpuExpertDesc *expert) {
    static const unsigned char zero[2] = {0x88, 0x88};
    static const float scales[2] = {1.0f, 1.0f};
    (void)user;
    (void)pipeline_layer;
    if (!slot || !expert || expert_id != 0) return 0;
    memset(expert, 0, sizeof(*expert));
    expert->gate = (ColiGlm53GpuWeightDesc){zero, scales, 4, 2, 2, 64};
    expert->up = (ColiGlm53GpuWeightDesc){zero, scales, 4, 2, 2, 64};
    expert->down = (ColiGlm53GpuWeightDesc){zero, scales, 4, 2, 2, 64};
    *slot = 0;
    return 1;
}

int coli_glm53_gpu_pipeline_probe(ColiGpuContext *ctx) {
    if (!ctx || !coli_gpu_context_healthy(ctx)) return 0;
    if (coli_gpu_context_probe(ctx, COLI_GPU_CAP_PIPELINE)) return 1;
    if (coli_gpu_context_consume_fault(
            ctx, COLI_GPU_FAULT_BASE_ALLOCATION))
        return 0;

    static const float zero4[8] = {0};
    static const float zero6[8] = {0};
    static const float zero_mhc[192] = {0};
    static const float zero_base[24] = {0};
    static const float one2[2] = {1.0f, 1.0f};
    static const float scale3[3] = {0.0f, 0.0f, 0.0f};
    static const float embedding[4] = {
        1.0f, -0.5f, 0.25f, 0.75f
    };
    static const float head[4] = {
        1.0f, 0.0f, 0.0f, 1.0f
    };

    ColiGlm53GpuMhcSiteDesc sites[4];
    for (int i = 0; i < 4; ++i)
        sites[i] = (ColiGlm53GpuMhcSiteDesc){
            zero_mhc, zero_base, scale3, one2
        };
    ColiGlm53GpuKdaLayerDesc kda;
    memset(&kda, 0, sizeof(kda));
    kda.q_proj = zero4;
    kda.k_proj = zero4;
    kda.v_proj = zero4;
    kda.o_proj = zero4;
    kda.gate_a_proj = zero4;
    kda.gate_b_proj = zero4;
    kda.decay_a_proj = zero4;
    kda.decay_b_proj = zero4;
    kda.beta_proj = zero4;
    kda.conv = zero6;
    kda.dt_bias = zero4;
    kda.a_log = zero4;
    kda.o_norm = one2;
    ColiGlm53GpuMlaLayerDesc mla;
    memset(&mla, 0, sizeof(mla));
    mla.q_a_proj = zero4;
    mla.q_a_norm = one2;
    mla.q_b_proj = zero4;
    mla.kv_a_proj = zero4;
    mla.kv_a_norm = one2;
    mla.kv_b_key = zero4;
    mla.kv_b_value = zero4;
    mla.o_proj = zero4;
    mla.index_q_proj = zero4;
    mla.index_k_proj = zero4;
    mla.index_weight_proj = zero4;
    mla.index_key_norm = one2;
    mla.index_key_bias = zero4;
    mla.index_pool_ape = zero4;
    mla.index_pool_gate = zero4;
    ColiGlm53GpuDenseLayerDesc dense = {
        {zero4, NULL, 0, 2, 2, 0},
        {zero4, NULL, 0, 2, 2, 0},
        {zero4, NULL, 0, 2, 2, 0}
    };
    ColiGlm53GpuMoeLayerDesc moe;
    memset(&moe, 0, sizeof(moe));
    moe.router = zero4;
    moe.shared_gate = (ColiGlm53GpuWeightDesc){
        zero4, NULL, 0, 2, 2, 0
    };
    moe.shared_up = moe.shared_gate;
    moe.shared_down = moe.shared_gate;
    const ColiGlm53GpuLayerDesc layers[2] = {
        {COLI_GLM53_GPU_ATTN_KDA, 0, COLI_GLM53_GPU_FFN_DENSE, 0, 0, 1},
        {COLI_GLM53_GPU_ATTN_MLA, 0, COLI_GLM53_GPU_FFN_MOE, 0, 2, 3}
    };
    ColiGlm53GpuModelDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.hidden_size = 2;
    desc.stream_count = 4;
    desc.vocab_size = 2;
    desc.max_prefill_rows = 2;
    desc.max_context_tokens = 4;
    desc.norm_eps = 1e-6f;
    desc.hc_eps = 1e-6f;
    desc.embedding = embedding;
    desc.final_norm = one2;
    desc.lm_head = head;
    desc.sites = sites;
    desc.site_count = 4;
    desc.kda_heads = 1;
    desc.kda_head_dim = 2;
    desc.kda_kernel = 1;
    desc.kda_gate_lower_bound = -20.0f;
    desc.kda_layers = &kda;
    desc.kda_layer_count = 1;
    desc.mla_heads = 1;
    desc.mla_q_lora = 2;
    desc.mla_kv_lora = 2;
    desc.mla_qk_nope = 2;
    desc.mla_value_dim = 2;
    desc.mla_index_heads = 1;
    desc.mla_index_dim = 2;
    desc.mla_index_pool = 1;
    desc.mla_index_topk = 1;
    desc.mla_index_select_tail = 0;
    desc.mla_page_tokens = 1;
    desc.mla_layers = &mla;
    desc.mla_layer_count = 1;
    desc.dense_intermediate = 2;
    desc.swiglu_limit = 1.0f;
    desc.dense_layers = &dense;
    desc.dense_layer_count = 1;
    desc.moe_experts = 1;
    desc.moe_topk = 1;
    desc.moe_intermediate = 2;
    desc.moe_cache_slots = 1;
    desc.moe_group_size = 64;
    desc.moe_normalize_topk = 1;
    desc.moe_routed_scale = 1.0f;
    desc.moe_swiglu_limit = 1.0f;
    desc.moe_layers = &moe;
    desc.moe_layer_count = 1;
    desc.layers = layers;
    desc.layer_count = 2;
    desc.expert_loader = probe_expert_loader;

    ColiGlm53GpuModel *model = NULL;
    ColiGlm53GpuSession *session = NULL;
    const int token = 0;
    float logits[2];
    int ok = coli_glm53_gpu_model_create(&model, ctx, &desc);
    if (!ok) fprintf(stderr, "GLM53 pipeline probe: model allocation failed\n");
    if (ok) {
        ok = coli_glm53_gpu_session_create(&session, model, 4);
        if (!ok)
            fprintf(stderr, "GLM53 pipeline probe: session allocation failed\n");
    }
    if (ok) {
        ok = coli_glm53_gpu_forward(session, &token, 1, logits);
        if (!ok) fprintf(stderr, "GLM53 pipeline probe: forward failed\n");
    }
    if (ok) {
        ok = isfinite(logits[0]) && isfinite(logits[1]);
        if (!ok) fprintf(stderr, "GLM53 pipeline probe: non-finite logits\n");
    }
    coli_glm53_gpu_session_destroy(session);
    coli_glm53_gpu_model_destroy(model);
    return ok && coli_gpu_context_healthy(ctx) &&
           coli_gpu_context_advertise_pipeline(ctx);
}

#else

int coli_glm53_gpu_model_create(ColiGlm53GpuModel **out,
                                ColiGpuContext *ctx,
                                const ColiGlm53GpuModelDesc *desc) {
    (void)ctx;
    (void)desc;
    if (out) *out = NULL;
    return 0;
}

void coli_glm53_gpu_model_destroy(ColiGlm53GpuModel *model) {
    (void)model;
}

int coli_glm53_gpu_session_create(ColiGlm53GpuSession **out,
                                  ColiGlm53GpuModel *model,
                                  int max_context) {
    (void)model;
    (void)max_context;
    if (out) *out = NULL;
    return 0;
}

void coli_glm53_gpu_session_destroy(ColiGlm53GpuSession *session) {
    (void)session;
}

int coli_glm53_gpu_session_embed(ColiGlm53GpuSession *session,
                                 const int32_t *token_ids, int rows) {
    (void)session;
    (void)token_ids;
    (void)rows;
    return 0;
}

int coli_glm53_gpu_session_core_site(ColiGlm53GpuSession *session,
                                     int site_index, int rows) {
    (void)session;
    (void)site_index;
    (void)rows;
    return 0;
}

int coli_glm53_gpu_session_kda_upload_input(ColiGlm53GpuSession *session,
                                            const float *input, int rows) {
    (void)session;
    (void)input;
    (void)rows;
    return 0;
}

int coli_glm53_gpu_session_kda_site(ColiGlm53GpuSession *session,
                                    int layer_index, int rows,
                                    int start_position) {
    (void)session;
    (void)layer_index;
    (void)rows;
    (void)start_position;
    return 0;
}

int coli_glm53_gpu_session_kda_download_output(
    ColiGlm53GpuSession *session, float *output, int rows) {
    (void)session;
    (void)output;
    (void)rows;
    return 0;
}

int coli_glm53_gpu_session_kda_state_download(
    ColiGlm53GpuSession *session, int layer_index,
    float *matrix, size_t matrix_floats,
    float *window, size_t window_floats) {
    (void)session;
    (void)layer_index;
    (void)matrix;
    (void)matrix_floats;
    (void)window;
    (void)window_floats;
    return 0;
}

int coli_glm53_gpu_session_mla_upload_input(
    ColiGlm53GpuSession *session, const float *input, int rows) {
    (void)session;
    (void)input;
    (void)rows;
    return 0;
}

int coli_glm53_gpu_session_mla_site(
    ColiGlm53GpuSession *session, int layer_index, int rows,
    int start_position) {
    (void)session;
    (void)layer_index;
    (void)rows;
    (void)start_position;
    return 0;
}

int coli_glm53_gpu_session_mla_download_output(
    ColiGlm53GpuSession *session, float *output,
    int *selected, size_t selected_count, int rows) {
    (void)session;
    (void)output;
    (void)selected;
    (void)selected_count;
    (void)rows;
    return 0;
}

int coli_glm53_gpu_session_mla_state_info(
    ColiGlm53GpuSession *session, int layer_index,
    int *logical_length, int *capacity) {
    (void)session;
    (void)layer_index;
    (void)logical_length;
    (void)capacity;
    return 0;
}

int coli_glm53_gpu_session_mla_state_download(
    ColiGlm53GpuSession *session, int layer_index,
    float *latent, size_t latent_floats,
    float *index_keys, size_t index_key_floats,
    float *index_gates, size_t index_gate_floats) {
    (void)session;
    (void)layer_index;
    (void)latent;
    (void)latent_floats;
    (void)index_keys;
    (void)index_key_floats;
    (void)index_gates;
    (void)index_gate_floats;
    return 0;
}

int coli_glm53_gpu_session_mla_reset_layer(
    ColiGlm53GpuSession *session, int layer_index) {
    (void)session;
    (void)layer_index;
    return 0;
}

int coli_glm53_gpu_model_moe_upload_expert(
    ColiGlm53GpuModel *model, int layer_index, int expert_id, int slot,
    const ColiGlm53GpuExpertDesc *expert,
    ColiGlm53GpuExpertHandle *handle) {
    (void)model; (void)layer_index; (void)expert_id; (void)slot;
    (void)expert; (void)handle;
    return 0;
}

int coli_glm53_gpu_session_moe_upload_input(
    ColiGlm53GpuSession *session, const float *input, int rows) {
    (void)session; (void)input; (void)rows;
    return 0;
}

int coli_glm53_gpu_session_moe_route(
    ColiGlm53GpuSession *session, int layer_index, int rows) {
    (void)session; (void)layer_index; (void)rows;
    return 0;
}

int coli_glm53_gpu_session_moe_selected(
    ColiGlm53GpuSession *session, int *selected_ids, float *routing_weights,
    size_t selected_count, int rows) {
    (void)session; (void)selected_ids; (void)routing_weights;
    (void)selected_count; (void)rows;
    return 0;
}

int coli_glm53_gpu_session_moe_site(
    ColiGlm53GpuSession *session, int layer_index,
    const ColiGlm53GpuExpertHandle *handles, size_t handle_count, int rows) {
    (void)session; (void)layer_index; (void)handles;
    (void)handle_count; (void)rows;
    return 0;
}

int coli_glm53_gpu_session_moe_download_output(
    ColiGlm53GpuSession *session, float *output, int rows) {
    (void)session; (void)output; (void)rows;
    return 0;
}

int coli_glm53_gpu_session_reset(ColiGlm53GpuSession *session) {
    (void)session;
    return 0;
}

int coli_glm53_gpu_session_set_quality_capture(
    ColiGlm53GpuSession *session,
    const ColiGlm53GpuQualityCapture *capture) {
    (void)session;
    (void)capture;
    return 0;
}

int coli_glm53_gpu_session_output(ColiGlm53GpuSession *session,
                                  float *logits, int rows) {
    (void)session;
    (void)logits;
    (void)rows;
    return 0;
}

int coli_glm53_gpu_forward(ColiGlm53GpuSession *session,
                           const int *token_ids, int rows,
                           float *last_logits_host) {
    (void)session;
    (void)token_ids;
    (void)rows;
    (void)last_logits_host;
    return 0;
}

int coli_glm53_gpu_pipeline_probe(ColiGpuContext *ctx) {
    (void)ctx;
    return 0;
}

#endif
