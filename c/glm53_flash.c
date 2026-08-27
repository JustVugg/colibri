#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "st.h"
#include "glm53_indexer.h"
#include "glm53_kda.h"
#include "glm53_mhc.h"
#include "glm53_sparse_attention.h"
#include "glm53_fp8.h"
#include "tensor.h"

typedef struct {
    ColiTensorView view;
    void *data;
    float *scales;
    shards *source;
    char *name;
    int streamed;
} Matrix;

typedef struct {
    int hidden, layers, vocab, dense_inter, moe_inter, experts, topk, shared;
    int heads, q_rank, kv_rank, key_dim, value_dim;
    int kda_heads, kda_dim, conv_kernel;
    int index_heads, index_dim, index_topk, index_pool;
    int hc, hc_iters;
    unsigned char *layer_kda, *layer_sparse;
    float eps, hc_eps, route_scale, swiglu_limit, gate_lower_bound;
} Config;

typedef struct {
    float *norm1, *norm2;
    float *ah_fn, *ah_base, *ah_scale, *fh_fn, *fh_base, *fh_scale;
    int kda, sparse;
    float *q, *k, *v, *conv, *fa, *fb, *dt, *alog, *beta, *ga, *gb, *onorm, *op;
    Matrix dqa, dqb, dkva, dop;
    float *qan, *kvan, *kvb;
    float *iwq, *iwk, *iknw, *iknb, *igate, *iweight, *iape;
    Matrix fg, fu, fd;
    float *router, *router_bias;
    Matrix sg, su, sd;
    Matrix *eg, *eu, *ed;
} Layer;

typedef struct {
    Config c;
    shards tensors;
    float *embed, *norm, *head;
    Layer *layer;
} Model;

typedef struct {
    float *kda_state, *kda_window;
    float *keys, *values, *index_keys, *index_gates;
    int capacity;
} LayerCache;

typedef struct {
    LayerCache *layers;
    int length;
} RuntimeCache;

static void matmul(float *out, const float *input, const float *weight, int rows, int in, int columns);

static void die(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}
static float *alloc_floats(size_t count) {
    float *result = calloc(count, sizeof(*result));
    if (!result) die("glm53: out of memory");
    return result;
}
static double required_number(jval *object, const char *key) {
    jval *value = json_get(object, key);
    if (!value || value->t != J_NUM) {
        fprintf(stderr, "glm53 config: missing numeric %s\n", key);
        exit(1);
    }
    return value->num;
}
static jval *required_object(jval *object, const char *key) {
    jval *value = json_get(object, key);
    if (!value || value->t != J_OBJ) {
        fprintf(stderr, "glm53 config: missing object %s\n", key);
        exit(1);
    }
    return value;
}
static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror(path);
        exit(1);
    }
    if (fseek(file, 0, SEEK_END)) die("glm53: config seek failed");
    long size = ftell(file);
    if (size < 0 || size > (64L << 20)) die("glm53: invalid config size");
    rewind(file);
    char *data = malloc((size_t)size + 1);
    if (!data || fread(data, 1, (size_t)size, file) != (size_t)size) die("glm53: config read failed");
    data[size] = 0;
    fclose(file);
    return data;
}
static void load_config(Config *c, const char *directory) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", directory);
    char *data = read_file(path), *arena = NULL;
    jval *root = json_parse(data, &arena);
    jval *text = required_object(root, "text_config");
    jval *linear = required_object(text, "linear_attn_config");
    c->hidden = (int)required_number(text, "hidden_size");
    c->layers = (int)required_number(text, "num_hidden_layers");
    c->vocab = (int)required_number(text, "vocab_size");
    c->dense_inter = (int)required_number(text, "intermediate_size");
    c->moe_inter = (int)required_number(text, "moe_intermediate_size");
    c->experts = (int)required_number(text, "n_routed_experts");
    c->topk = (int)required_number(text, "num_experts_per_tok");
    c->shared = (int)required_number(text, "n_shared_experts");
    c->heads = (int)required_number(text, "num_attention_heads");
    c->q_rank = (int)required_number(text, "q_lora_rank");
    c->kv_rank = (int)required_number(text, "kv_lora_rank");
    c->key_dim = (int)required_number(text, "qk_nope_head_dim");
    c->value_dim = (int)required_number(text, "v_head_dim");
    c->index_heads = (int)required_number(text, "index_n_heads");
    c->index_dim = (int)required_number(text, "index_head_dim");
    c->index_topk = (int)required_number(text, "index_topk");
    c->index_pool = (int)required_number(text, "index_kpool");
    c->hc = (int)required_number(text, "hc_mult");
    c->hc_iters = (int)required_number(text, "hc_sinkhorn_iters");
    c->kda_heads = (int)required_number(linear, "num_heads");
    c->kda_dim = (int)required_number(linear, "head_dim");
    c->conv_kernel = (int)required_number(linear, "short_conv_kernel_size");
    c->eps = (float)required_number(text, "rms_norm_eps");
    c->hc_eps = (float)required_number(text, "hc_eps");
    c->route_scale = (float)required_number(text, "routed_scaling_factor");
    c->swiglu_limit = (float)required_number(text, "swiglu_limit");
    c->gate_lower_bound = (float)required_number(linear, "gate_lower_bound");
    jval *layer_types = json_get(text, "layer_types");
    jval *mlp_types = json_get(text, "mlp_layer_types");
    if (!layer_types || layer_types->t != J_ARR || layer_types->len != c->layers || !mlp_types ||
        mlp_types->t != J_ARR || mlp_types->len != c->layers)
        die("glm53 config: invalid layer type arrays");
    c->layer_kda = calloc((size_t)c->layers, 1);
    c->layer_sparse = calloc((size_t)c->layers, 1);
    if (!c->layer_kda || !c->layer_sparse) die("glm53 config: layer type allocation failed");
    for (int i = 0; i < c->layers; i++) {
        jval *attention = layer_types->kids[i];
        jval *mlp = mlp_types->kids[i];
        if (!attention || attention->t != J_STR ||
            (strcmp(attention->str, "linear_attention") && strcmp(attention->str, "deepseek_sparse_attention")))
            die("glm53 config: unsupported layer_types value");
        if (!mlp || mlp->t != J_STR || (strcmp(mlp->str, "dense") && strcmp(mlp->str, "sparse")))
            die("glm53 config: unsupported mlp_layer_types value");
        c->layer_kda[i] = !strcmp(attention->str, "linear_attention");
        c->layer_sparse[i] = !strcmp(mlp->str, "sparse");
    }
    if (c->hidden < 1 || c->layers < 1 || c->layers > 128 || c->heads < 1 || c->experts < 1 || c->topk < 1 ||
        c->topk > c->experts || c->kda_heads * c->kda_dim <= 0 || c->hc < 1 || c->hc > 16)
        die("glm53 config: dimension out of range");
    json_free(root);
    free(arena);
    free(data);
}

static float *load_tensor(Model *model, const char *name) {
    int64_t count = st_numel(&model->tensors, name);
    if (count < 0) {
        fprintf(stderr, "glm53: missing tensor %s\n", name);
        exit(1);
    }
    float *data = alloc_floats((size_t)count);
    st_read_f32_cap(&model->tensors, name, data, count, 0);
    return data;
}
static float *load_named(Model *m, int layer, const char *suffix) {
    char name[512];
    snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", layer, suffix);
    return load_tensor(m, name);
}
static Matrix load_matrix_mode(Model *model, const char *name, int streamed) {
    st_tensor *tensor = st_find(&model->tensors, name);
    if (!tensor || tensor->rank != 2 || tensor->shape[0] < 1 || tensor->shape[1] < 1)
        die("glm53: invalid matrix tensor");
    Matrix matrix = {0};
    matrix.view.rows = tensor->shape[0];
    matrix.view.columns = tensor->shape[1];
    matrix.source = &model->tensors;
    matrix.streamed = streamed;
    matrix.name = strdup(name);
    if (!matrix.name) die("glm53: matrix name allocation failed");
    if (tensor->dtype <= 2) {
        matrix.view.format = COLI_TENSOR_F32;
        matrix.view.data_bytes = (size_t)tensor->numel * sizeof(float);
        if (!streamed) {
            matrix.data = alloc_floats((size_t)tensor->numel);
            st_read_f32_cap(&model->tensors, name, matrix.data, tensor->numel, 0);
            matrix.view.data = matrix.data;
        }
        return matrix;
    }
    if (tensor->dtype != 4 || tensor->nbytes != tensor->numel) die("glm53: matrix must be float or F8_E4M3");
    char scale_name[640];
    snprintf(scale_name, sizeof(scale_name), "%s_scale_inv", name);
    st_tensor *scale = st_find(&model->tensors, scale_name);
    int64_t scale_rows = (tensor->shape[0] + 127) / 128;
    int64_t scale_columns = (tensor->shape[1] + 127) / 128;
    if (!scale || scale->dtype > 2 || scale->rank != 2 || scale->shape[0] != scale_rows ||
        scale->shape[1] != scale_columns)
        die("glm53: invalid FP8 weight_scale_inv tensor");
    matrix.view.format = COLI_TENSOR_FP8_E4M3_BLOCK;
    matrix.view.scale_format = COLI_SCALE_F32;
    matrix.view.data_bytes = (size_t)tensor->nbytes;
    matrix.view.scale_bytes = (size_t)scale->numel * sizeof(float);
    matrix.view.block_rows = 128;
    matrix.view.block_columns = 128;
    if (!streamed) {
        matrix.data = malloc((size_t)tensor->nbytes);
        matrix.scales = alloc_floats((size_t)scale->numel);
        if (!matrix.data) die("glm53: FP8 matrix allocation failed");
        st_read_raw_cap(&model->tensors, name, matrix.data, tensor->nbytes, 0);
        st_read_f32_cap(&model->tensors, scale_name, matrix.scales, scale->numel, 0);
        matrix.view.data = matrix.data;
        matrix.view.scales = matrix.scales;
    }
    return matrix;
}
static Matrix load_matrix(Model *model, const char *name) { return load_matrix_mode(model, name, 0); }
static Matrix load_named_matrix(Model *model, int layer, const char *suffix) {
    char name[512];
    snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", layer, suffix);
    return load_matrix(model, name);
}
static Matrix load_named_streamed_matrix(Model *model, int layer, const char *suffix) {
    char name[512];
    snprintf(name, sizeof(name), "model.language_model.layers.%d.%s", layer, suffix);
    return load_matrix_mode(model, name, 1);
}
static void matrix_free(Matrix *matrix) {
    free(matrix->data);
    free(matrix->scales);
    free(matrix->name);
    memset(matrix, 0, sizeof(*matrix));
}
static void matrix_multiply(float *output, const float *input, const Matrix *weight, int rows) {
    if (weight->streamed) {
        Matrix loaded = *weight;
        loaded.streamed = 0;
        loaded.name = NULL;
        loaded.data = malloc(weight->view.data_bytes);
        if (!loaded.data) die("glm53: streamed matrix allocation failed");
        loaded.view.data = loaded.data;
        if (weight->view.format == COLI_TENSOR_F32) {
            st_read_f32_cap(weight->source, weight->name, loaded.data,
                            (int64_t)(weight->view.data_bytes / sizeof(float)), 1);
        } else {
            loaded.scales = alloc_floats(weight->view.scale_bytes / sizeof(float));
            loaded.view.scales = loaded.scales;
            st_read_raw_cap(weight->source, weight->name, loaded.data, (int64_t)weight->view.data_bytes, 1);
            char scale_name[640];
            snprintf(scale_name, sizeof(scale_name), "%s_scale_inv", weight->name);
            st_read_f32_cap(weight->source, scale_name, loaded.scales,
                            (int64_t)(weight->view.scale_bytes / sizeof(float)), 1);
        }
        matrix_multiply(output, input, &loaded, rows);
        free(loaded.scales);
        free(loaded.data);
        return;
    }
    if (weight->view.format == COLI_TENSOR_F32) {
        matmul(output, input, weight->data, rows, (int)weight->view.columns, (int)weight->view.rows);
        return;
    }
    for (int row = 0; row < rows; row++)
        if (coli_glm53_fp8_matvec(output + (size_t)row * weight->view.rows, weight->view.data, weight->view.scales,
                                  input + (size_t)row * weight->view.columns, (int)weight->view.rows,
                                  (int)weight->view.columns))
            die("glm53: FP8 matrix multiply failed");
}
static float *load_conv(Model *model, int layer) {
    const char *parts[] = {"q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"};
    int64_t counts[3], total = 0;
    char name[512];
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.%s", layer, parts[i]);
        counts[i] = st_numel(&model->tensors, name);
        if (counts[i] < 0 || (i && counts[i] != counts[0])) die("glm53: invalid split q/k/v conv tensors");
        total += counts[i];
    }
    float *result = alloc_floats((size_t)total);
    int64_t offset = 0;
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.%s", layer, parts[i]);
        st_read_f32_cap(&model->tensors, name, result + offset, counts[i], 0);
        offset += counts[i];
    }
    return result;
}
static void model_load(Model *m, const char *directory) {
    memset(m, 0, sizeof(*m));
    load_config(&m->c, directory);
    st_init(&m->tensors, directory);
    Config *c = &m->c;
    m->embed = load_tensor(m, "model.language_model.embed_tokens.weight");
    m->norm = load_tensor(m, "model.language_model.norm.weight");
    m->head = load_tensor(m, "lm_head.weight");
    m->layer = calloc((size_t)c->layers, sizeof(*m->layer));
    if (!m->layer) die("glm53: layer allocation failed");
    for (int i = 0; i < c->layers; i++) {
        Layer *l = &m->layer[i];
        l->kda = c->layer_kda[i];
        l->sparse = c->layer_sparse[i];
        l->norm1 = load_named(m, i, "input_layernorm.weight");
        l->norm2 = load_named(m, i, "post_attention_layernorm.weight");
        l->ah_fn = load_named(m, i, "hc_attn_fn");
        l->ah_base = load_named(m, i, "hc_attn_base");
        l->ah_scale = load_named(m, i, "hc_attn_scale");
        l->fh_fn = load_named(m, i, "hc_ffn_fn");
        l->fh_base = load_named(m, i, "hc_ffn_base");
        l->fh_scale = load_named(m, i, "hc_ffn_scale");
        if (l->kda) {
            l->q = load_named(m, i, "self_attn.q_proj.weight");
            l->k = load_named(m, i, "self_attn.k_proj.weight");
            l->v = load_named(m, i, "self_attn.v_proj.weight");
            l->conv = load_conv(m, i);
            l->fa = load_named(m, i, "self_attn.f_a_proj.weight");
            l->fb = load_named(m, i, "self_attn.f_b_proj.weight");
            l->dt = load_named(m, i, "self_attn.dt_bias");
            l->alog = load_named(m, i, "self_attn.A_log");
            l->beta = load_named(m, i, "self_attn.b_proj.weight");
            l->ga = load_named(m, i, "self_attn.g_a_proj.weight");
            l->gb = load_named(m, i, "self_attn.g_b_proj.weight");
            l->onorm = load_named(m, i, "self_attn.o_norm.weight");
            l->op = load_named(m, i, "self_attn.o_proj.weight");
        } else {
            l->dqa = load_named_matrix(m, i, "self_attn.q_a_proj.weight");
            l->qan = load_named(m, i, "self_attn.q_a_layernorm.weight");
            l->dqb = load_named_matrix(m, i, "self_attn.q_b_proj.weight");
            l->dkva = load_named_matrix(m, i, "self_attn.kv_a_proj_with_mqa.weight");
            l->kvan = load_named(m, i, "self_attn.kv_a_layernorm.weight");
            l->kvb = load_named(m, i, "self_attn.kv_b_proj.weight");
            l->dop = load_named_matrix(m, i, "self_attn.o_proj.weight");
            l->iwq = load_named(m, i, "self_attn.indexer.wq_b.weight");
            l->iwk = load_named(m, i, "self_attn.indexer.wk.weight");
            l->iknw = load_named(m, i, "self_attn.indexer.k_norm.weight");
            l->iknb = load_named(m, i, "self_attn.indexer.k_norm.bias");
            l->igate = load_named(m, i, "self_attn.indexer.index_kpool_compress_gate");
            l->iweight = load_named(m, i, "self_attn.indexer.weights_proj.weight");
            l->iape = load_named(m, i, "self_attn.indexer.index_kpool_compress_ape");
        }
        if (!l->sparse) {
            l->fg = load_named_matrix(m, i, "mlp.gate_proj.weight");
            l->fu = load_named_matrix(m, i, "mlp.up_proj.weight");
            l->fd = load_named_matrix(m, i, "mlp.down_proj.weight");
        } else {
            l->router = load_named(m, i, "mlp.gate.weight");
            l->router_bias = load_named(m, i, "mlp.gate.e_score_correction_bias");
            l->sg = load_named_matrix(m, i, "mlp.shared_experts.gate_proj.weight");
            l->su = load_named_matrix(m, i, "mlp.shared_experts.up_proj.weight");
            l->sd = load_named_matrix(m, i, "mlp.shared_experts.down_proj.weight");
            l->eg = calloc((size_t)c->experts, sizeof(Matrix));
            l->eu = calloc((size_t)c->experts, sizeof(Matrix));
            l->ed = calloc((size_t)c->experts, sizeof(Matrix));
            for (int e = 0; e < c->experts; e++) {
                char name[128];
                snprintf(name, sizeof(name), "mlp.experts.%d.gate_proj.weight", e);
                l->eg[e] = load_named_streamed_matrix(m, i, name);
                snprintf(name, sizeof(name), "mlp.experts.%d.up_proj.weight", e);
                l->eu[e] = load_named_streamed_matrix(m, i, name);
                snprintf(name, sizeof(name), "mlp.experts.%d.down_proj.weight", e);
                l->ed[e] = load_named_streamed_matrix(m, i, name);
            }
        }
    }
}

static void layer_free(Layer *layer, int experts) {
    free(layer->norm1);
    free(layer->norm2);
    free(layer->ah_fn);
    free(layer->ah_base);
    free(layer->ah_scale);
    free(layer->fh_fn);
    free(layer->fh_base);
    free(layer->fh_scale);
    free(layer->q);
    free(layer->k);
    free(layer->v);
    free(layer->conv);
    free(layer->fa);
    free(layer->fb);
    free(layer->dt);
    free(layer->alog);
    free(layer->beta);
    free(layer->ga);
    free(layer->gb);
    free(layer->onorm);
    free(layer->op);
    matrix_free(&layer->dqa);
    free(layer->qan);
    matrix_free(&layer->dqb);
    matrix_free(&layer->dkva);
    free(layer->kvan);
    free(layer->kvb);
    matrix_free(&layer->dop);
    free(layer->iwq);
    free(layer->iwk);
    free(layer->iknw);
    free(layer->iknb);
    free(layer->igate);
    free(layer->iweight);
    free(layer->iape);
    matrix_free(&layer->fg);
    matrix_free(&layer->fu);
    matrix_free(&layer->fd);
    free(layer->router);
    free(layer->router_bias);
    matrix_free(&layer->sg);
    matrix_free(&layer->su);
    matrix_free(&layer->sd);
    for (int expert = 0; expert < experts; expert++) {
        if (layer->eg) matrix_free(&layer->eg[expert]);
        if (layer->eu) matrix_free(&layer->eu[expert]);
        if (layer->ed) matrix_free(&layer->ed[expert]);
    }
    free(layer->eg);
    free(layer->eu);
    free(layer->ed);
}

static void model_free(Model *model) {
    for (int layer = 0; layer < model->c.layers; layer++) layer_free(&model->layer[layer], model->c.experts);
    free(model->layer);
    free(model->embed);
    free(model->norm);
    free(model->head);
    free(model->c.layer_kda);
    free(model->c.layer_sparse);
    st_destroy(&model->tensors);
}

static void matmul(float *out, const float *input, const float *weight, int rows, int in, int columns) {
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < columns; o++) {
            float sum = 0.0f;
            const float *w = weight + (size_t)o * in, *x = input + (size_t)r * in;
            for (int i = 0; i < in; i++) sum += x[i] * w[i];
            out[(size_t)r * columns + o] = sum;
        }
}
static void rmsnorm(float *out, const float *input, const float *weight, int rows, int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *x = input + (size_t)r * dim;
        for (int i = 0; i < dim; i++) sum += x[i] * x[i];
        float scale = 1.0f / sqrtf(sum / dim + eps);
        for (int i = 0; i < dim; i++) out[(size_t)r * dim + i] = x[i] * scale * weight[i];
    }
}
static void layernorm(float *out, const float *input, const float *w, const float *b, int rows, int dim) {
    for (int r = 0; r < rows; r++) {
        const float *x = input + (size_t)r * dim;
        float mean = 0, var = 0;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        for (int i = 0; i < dim; i++) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= dim;
        float inv = 1.0f / sqrtf(var + 1e-6f);
        for (int i = 0; i < dim; i++) out[(size_t)r * dim + i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}
static float sigmoid(float x) { return x >= 0 ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x)); }
static float swiglu(float gate, float up, float limit) {
    if (gate > limit) gate = limit;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;
    return gate * sigmoid(gate) * up;
}
static void trace_state(const char *name, const float *data, size_t count) {
    if (!getenv("GLM53_TRACE")) return;
    double sum = 0, square = 0;
    for (size_t i = 0; i < count; i++) {
        sum += data[i];
        square += (double)data[i] * data[i];
    }
    fprintf(stderr, "[GLM53-TRACE] %s sum=%.12g square=%.12g\n", name, sum, square);
}

static void kda_forward(const Config *c, const Layer *l, const float *x, int n, float *out) {
    int p = c->kda_heads * c->kda_dim;
    float *qkv = alloc_floats((size_t)n * 3 * p), *tmp = alloc_floats((size_t)n * c->kda_dim);
    float *decay = alloc_floats((size_t)n * p), *betas = alloc_floats((size_t)n * c->kda_heads),
          *gate = alloc_floats((size_t)n * p);
    matmul(qkv, x, l->q, n, c->hidden, p);
    matmul(qkv + (size_t)n * p, x, l->k, n, c->hidden, p);
    matmul(qkv + (size_t)n * 2 * p, x, l->v, n, c->hidden, p);
    /* Rearrange projection-major batches into token-major q/k/v. */
    float *packed = alloc_floats((size_t)n * 3 * p);
    for (int t = 0; t < n; t++)
        for (int z = 0; z < 3; z++)
            memcpy(packed + (size_t)t * 3 * p + z * p, qkv + (size_t)z * n * p + (size_t)t * p,
                   (size_t)p * sizeof(float));
    matmul(tmp, x, l->fa, n, c->hidden, c->kda_dim);
    matmul(decay, tmp, l->fb, n, c->kda_dim, p);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < c->kda_heads; h++)
            for (int d = 0; d < c->kda_dim; d++) {
                int i = h * c->kda_dim + d;
                decay[(size_t)t * p + i] =
                    c->gate_lower_bound * sigmoid(expf(l->alog[h]) * (decay[(size_t)t * p + i] + l->dt[i]));
            }
    matmul(betas, x, l->beta, n, c->hidden, c->kda_heads);
    for (int i = 0; i < n * c->kda_heads; i++) betas[i] = sigmoid(betas[i]);
    matmul(tmp, x, l->ga, n, c->hidden, c->kda_dim);
    matmul(gate, tmp, l->gb, n, c->kda_dim, p);
    float *state = alloc_floats((size_t)c->kda_heads * c->kda_dim * c->kda_dim),
          *window = alloc_floats((size_t)3 * p * c->conv_kernel), *core = alloc_floats((size_t)n * p),
          *normed = alloc_floats((size_t)n * p);
    for (int t = 0; t < n; t++)
        coli_glm53_kda_step(core + (size_t)t * p, state, window, packed + (size_t)t * 3 * p, l->conv,
                            decay + (size_t)t * p, betas + (size_t)t * c->kda_heads, c->kda_heads, c->kda_dim,
                            c->conv_kernel);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < c->kda_heads; h++) {
            float sum = 0;
            float *src = core + (size_t)t * p + h * c->kda_dim, *dst = normed + (size_t)t * p + h * c->kda_dim;
            for (int d = 0; d < c->kda_dim; d++) sum += src[d] * src[d];
            float inv = 1.0f / sqrtf(sum / c->kda_dim + c->eps);
            for (int d = 0; d < c->kda_dim; d++)
                dst[d] = src[d] * inv * l->onorm[d] * sigmoid(gate[(size_t)t * p + h * c->kda_dim + d]);
        }
    matmul(out, normed, l->op, n, p, c->hidden);
    free(normed);
    free(core);
    free(window);
    free(state);
    free(gate);
    free(betas);
    free(decay);
    free(tmp);
    free(packed);
    free(qkv);
}

static void dsa_forward(const Config *c, const Layer *l, const float *x, int n, float *out) {
    int qwidth = c->heads * c->key_dim, kvwidth = c->heads * (c->key_dim + c->value_dim),
        iwidth = c->index_topk + c->index_pool - 1;
    float *qr = alloc_floats((size_t)n * c->q_rank), *qn = alloc_floats((size_t)n * c->q_rank),
          *queries = alloc_floats((size_t)n * qwidth);
    float *latent = alloc_floats((size_t)n * c->kv_rank), *ln = alloc_floats((size_t)n * c->kv_rank),
          *expanded = alloc_floats((size_t)n * kvwidth);
    matrix_multiply(qr, x, &l->dqa, n);
    rmsnorm(qn, qr, l->qan, n, c->q_rank, c->eps);
    matrix_multiply(queries, qn, &l->dqb, n);
    matrix_multiply(latent, x, &l->dkva, n);
    rmsnorm(ln, latent, l->kvan, n, c->kv_rank, c->eps);
    matmul(expanded, ln, l->kvb, n, c->kv_rank, kvwidth);
    float *keys = alloc_floats((size_t)n * c->heads * c->key_dim),
          *values = alloc_floats((size_t)n * c->heads * c->value_dim);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < c->heads; h++) {
            const float *src = expanded + (size_t)t * kvwidth + h * (c->key_dim + c->value_dim);
            memcpy(keys + ((size_t)t * c->heads + h) * c->key_dim, src, (size_t)c->key_dim * sizeof(float));
            memcpy(values + ((size_t)t * c->heads + h) * c->value_dim, src + c->key_dim,
                   (size_t)c->value_dim * sizeof(float));
        }
    float *iq = alloc_floats((size_t)n * c->index_heads * c->index_dim),
          *ikraw = alloc_floats((size_t)n * c->index_dim), *ik = alloc_floats((size_t)n * c->index_dim),
          *gates = alloc_floats((size_t)n * c->index_dim), *weights = alloc_floats((size_t)n * c->index_heads);
    matmul(iq, qn, l->iwq, n, c->q_rank, c->index_heads * c->index_dim);
    matmul(ikraw, x, l->iwk, n, c->hidden, c->index_dim);
    layernorm(ik, ikraw, l->iknw, l->iknb, n, c->index_dim);
    matmul(gates, x, l->igate, n, c->hidden, c->index_dim);
    matmul(weights, x, l->iweight, n, c->hidden, c->index_heads);
    for (int i = 0; i < n * c->index_heads; i++) weights[i] /= sqrtf((float)c->index_heads);
    unsigned char *valid = malloc((size_t)n);
    memset(valid, 1, (size_t)n);
    int *indices = malloc((size_t)n * iwidth * sizeof(int));
    coli_glm53_index_select(indices, iq, ik, gates, weights, l->iape, valid, n, c->index_heads, c->index_dim,
                            c->index_pool, c->index_topk);
    if (getenv("GLM53_TRACE")) {
        fprintf(stderr, "[GLM53-TRACE] indices");
        for (int i = 0; i < n * iwidth; i++) fprintf(stderr, " %d", indices[i]);
        fprintf(stderr, "\n");
    }
    float *context = alloc_floats((size_t)n * c->heads * c->value_dim);
    coli_glm53_sparse_attention(context, queries, keys, values, indices, n, iwidth, c->heads, c->key_dim, c->value_dim);
    matrix_multiply(out, context, &l->dop, n);
    free(context);
    free(indices);
    free(valid);
    free(weights);
    free(gates);
    free(ik);
    free(ikraw);
    free(iq);
    free(values);
    free(keys);
    free(expanded);
    free(ln);
    free(latent);
    free(queries);
    free(qn);
    free(qr);
}

static void mlp3(float *out, const float *x, const Matrix *wg, const Matrix *wu, const Matrix *wd, int n, int inter,
                 float limit) {
    float *g = alloc_floats((size_t)n * inter), *u = alloc_floats((size_t)n * inter),
          *a = alloc_floats((size_t)n * inter);
    matrix_multiply(g, x, wg, n);
    matrix_multiply(u, x, wu, n);
    for (int i = 0; i < n * inter; i++) a[i] = swiglu(g[i], u[i], limit);
    matrix_multiply(out, a, wd, n);
    free(a);
    free(u);
    free(g);
}
static void ffn_forward(const Config *c, const Layer *l, const float *x, int n, float *out) {
    if (!l->sparse) {
        mlp3(out, x, &l->fg, &l->fu, &l->fd, n, c->dense_inter, c->swiglu_limit);
        return;
    }
    mlp3(out, x, &l->sg, &l->su, &l->sd, n, c->moe_inter * c->shared, c->swiglu_limit);
    float *score = alloc_floats((size_t)c->experts), *temp = alloc_floats((size_t)c->hidden);
    for (int t = 0; t < n; t++) {
        const float *row = x + (size_t)t * c->hidden;
        for (int e = 0; e < c->experts; e++) {
            float s = 0;
            for (int d = 0; d < c->hidden; d++) s += row[d] * l->router[(size_t)e * c->hidden + d];
            score[e] = sigmoid(s);
        }
        int ids[64];
        float weights[64], total = 0;
        for (int k = 0; k < c->topk; k++) {
            int best = -1;
            float bv = -INFINITY;
            for (int e = 0; e < c->experts; e++) {
                int used = 0;
                for (int j = 0; j < k; j++)
                    if (ids[j] == e) used = 1;
                float choice = score[e] + l->router_bias[e];
                if (!used && choice > bv) {
                    bv = choice;
                    best = e;
                }
            }
            ids[k] = best;
            weights[k] = score[best];
            total += weights[k];
        }
        for (int k = 0; k < c->topk; k++) {
            weights[k] = weights[k] / (total + 1e-20f) * c->route_scale;
            mlp3(temp, row, &l->eg[ids[k]], &l->eu[ids[k]], &l->ed[ids[k]], 1, c->moe_inter, c->swiglu_limit);
            for (int d = 0; d < c->hidden; d++) out[(size_t)t * c->hidden + d] += weights[k] * temp[d];
        }
    }
    free(temp);
    free(score);
}

static void cache_init(RuntimeCache *cache, const Config *config) {
    memset(cache, 0, sizeof(*cache));
    cache->layers = calloc((size_t)config->layers, sizeof(*cache->layers));
    if (!cache->layers) die("glm53: cache allocation failed");
    for (int i = 0; i < config->layers; i++) {
        if (!config->layer_kda[i]) continue;
        cache->layers[i].kda_state = alloc_floats((size_t)config->kda_heads * config->kda_dim * config->kda_dim);
        cache->layers[i].kda_window =
            alloc_floats((size_t)3 * config->kda_heads * config->kda_dim * config->conv_kernel);
    }
}

static void cache_free(RuntimeCache *cache, const Config *config) {
    for (int i = 0; i < config->layers; i++) {
        LayerCache *layer = &cache->layers[i];
        free(layer->kda_state);
        free(layer->kda_window);
        free(layer->keys);
        free(layer->values);
        free(layer->index_keys);
        free(layer->index_gates);
    }
    free(cache->layers);
}

static void kda_step_forward(const Config *c, const Layer *l, LayerCache *cache, const float *x, float *out) {
    int width = c->kda_heads * c->kda_dim;
    float *qkv = alloc_floats((size_t)3 * width);
    float *tmp = alloc_floats((size_t)c->kda_dim);
    float *decay = alloc_floats((size_t)width);
    float *beta = alloc_floats((size_t)c->kda_heads);
    float *gate = alloc_floats((size_t)width);
    float *core = alloc_floats((size_t)width);
    float *normed = alloc_floats((size_t)width);

    matmul(qkv, x, l->q, 1, c->hidden, width);
    matmul(qkv + width, x, l->k, 1, c->hidden, width);
    matmul(qkv + 2 * width, x, l->v, 1, c->hidden, width);
    matmul(tmp, x, l->fa, 1, c->hidden, c->kda_dim);
    matmul(decay, tmp, l->fb, 1, c->kda_dim, width);
    for (int h = 0; h < c->kda_heads; h++)
        for (int d = 0; d < c->kda_dim; d++) {
            int i = h * c->kda_dim + d;
            decay[i] = c->gate_lower_bound * sigmoid(expf(l->alog[h]) * (decay[i] + l->dt[i]));
        }
    matmul(beta, x, l->beta, 1, c->hidden, c->kda_heads);
    for (int h = 0; h < c->kda_heads; h++) beta[h] = sigmoid(beta[h]);
    matmul(tmp, x, l->ga, 1, c->hidden, c->kda_dim);
    matmul(gate, tmp, l->gb, 1, c->kda_dim, width);
    coli_glm53_kda_step(core, cache->kda_state, cache->kda_window, qkv, l->conv, decay, beta, c->kda_heads, c->kda_dim,
                        c->conv_kernel);
    for (int h = 0; h < c->kda_heads; h++) {
        float sum = 0.0f;
        for (int d = 0; d < c->kda_dim; d++) sum += core[h * c->kda_dim + d] * core[h * c->kda_dim + d];
        float inv = 1.0f / sqrtf(sum / c->kda_dim + c->eps);
        for (int d = 0; d < c->kda_dim; d++) {
            int i = h * c->kda_dim + d;
            normed[i] = core[i] * inv * l->onorm[d] * sigmoid(gate[i]);
        }
    }
    matmul(out, normed, l->op, 1, width, c->hidden);
    free(normed);
    free(core);
    free(gate);
    free(beta);
    free(decay);
    free(tmp);
    free(qkv);
}

static void dsa_cache_reserve(const Config *c, LayerCache *cache, int needed) {
    if (needed <= cache->capacity) return;
    int capacity = cache->capacity ? cache->capacity * 2 : 16;
    while (capacity < needed) capacity *= 2;
    cache->keys = realloc(cache->keys, (size_t)capacity * c->heads * c->key_dim * sizeof(float));
    cache->values = realloc(cache->values, (size_t)capacity * c->heads * c->value_dim * sizeof(float));
    cache->index_keys = realloc(cache->index_keys, (size_t)capacity * c->index_dim * sizeof(float));
    cache->index_gates = realloc(cache->index_gates, (size_t)capacity * c->index_dim * sizeof(float));
    if (!cache->keys || !cache->values || !cache->index_keys || !cache->index_gates)
        die("glm53: DSA cache allocation failed");
    cache->capacity = capacity;
}

static void dsa_step_forward(const Config *c, const Layer *l, LayerCache *cache, int position, const float *x,
                             float *out) {
    int length = position + 1;
    int qwidth = c->heads * c->key_dim;
    int kvwidth = c->heads * (c->key_dim + c->value_dim);
    int index_width = c->index_topk + c->index_pool - 1;
    float *q_resid = alloc_floats((size_t)c->q_rank);
    float *q_norm = alloc_floats((size_t)c->q_rank);
    float *query = alloc_floats((size_t)qwidth);
    float *latent = alloc_floats((size_t)c->kv_rank);
    float *latent_norm = alloc_floats((size_t)c->kv_rank);
    float *expanded = alloc_floats((size_t)kvwidth);
    float *index_query = alloc_floats((size_t)c->index_heads * c->index_dim);
    float *index_raw = alloc_floats((size_t)c->index_dim);
    float *head_weight = alloc_floats((size_t)c->index_heads);

    matrix_multiply(q_resid, x, &l->dqa, 1);
    rmsnorm(q_norm, q_resid, l->qan, 1, c->q_rank, c->eps);
    matrix_multiply(query, q_norm, &l->dqb, 1);
    matrix_multiply(latent, x, &l->dkva, 1);
    rmsnorm(latent_norm, latent, l->kvan, 1, c->kv_rank, c->eps);
    matmul(expanded, latent_norm, l->kvb, 1, c->kv_rank, kvwidth);
    matmul(index_query, q_norm, l->iwq, 1, c->q_rank, c->index_heads * c->index_dim);
    matmul(index_raw, x, l->iwk, 1, c->hidden, c->index_dim);
    matmul(head_weight, x, l->iweight, 1, c->hidden, c->index_heads);
    for (int h = 0; h < c->index_heads; h++) head_weight[h] /= sqrtf((float)c->index_heads);

    dsa_cache_reserve(c, cache, length);
    float *key_row = cache->keys + (size_t)position * c->heads * c->key_dim;
    float *value_row = cache->values + (size_t)position * c->heads * c->value_dim;
    for (int h = 0; h < c->heads; h++) {
        const float *source = expanded + h * (c->key_dim + c->value_dim);
        memcpy(key_row + h * c->key_dim, source, (size_t)c->key_dim * sizeof(float));
        memcpy(value_row + h * c->value_dim, source + c->key_dim, (size_t)c->value_dim * sizeof(float));
    }
    layernorm(cache->index_keys + (size_t)position * c->index_dim, index_raw, l->iknw, l->iknb, 1, c->index_dim);
    matmul(cache->index_gates + (size_t)position * c->index_dim, x, l->igate, 1, c->hidden, c->index_dim);

    float *queries = alloc_floats((size_t)length * qwidth);
    float *index_queries = alloc_floats((size_t)length * c->index_heads * c->index_dim);
    float *head_weights = alloc_floats((size_t)length * c->index_heads);
    unsigned char *valid = malloc((size_t)length);
    int *indices = malloc((size_t)length * index_width * sizeof(int));
    float *context = alloc_floats((size_t)length * c->heads * c->value_dim);
    if (!valid || !indices) die("glm53: DSA scratch allocation failed");
    memcpy(queries + (size_t)position * qwidth, query, (size_t)qwidth * sizeof(float));
    memcpy(index_queries + (size_t)position * c->index_heads * c->index_dim, index_query,
           (size_t)c->index_heads * c->index_dim * sizeof(float));
    memcpy(head_weights + (size_t)position * c->index_heads, head_weight, (size_t)c->index_heads * sizeof(float));
    memset(valid, 1, (size_t)length);
    coli_glm53_index_select(indices, index_queries, cache->index_keys, cache->index_gates, head_weights, l->iape, valid,
                            length, c->index_heads, c->index_dim, c->index_pool, c->index_topk);
    coli_glm53_sparse_attention(context, queries, cache->keys, cache->values, indices, length, index_width, c->heads,
                                c->key_dim, c->value_dim);
    matrix_multiply(out, context + (size_t)position * c->heads * c->value_dim, &l->dop, 1);
    free(context);
    free(indices);
    free(valid);
    free(head_weights);
    free(index_queries);
    free(queries);
    free(head_weight);
    free(index_raw);
    free(index_query);
    free(expanded);
    free(latent_norm);
    free(latent);
    free(query);
    free(q_norm);
    free(q_resid);
}

static float *forward_cached_step(Model *model, RuntimeCache *cache, int token) {
    Config *c = &model->c;
    float *streams = alloc_floats((size_t)c->hc * c->hidden);
    float *next = alloc_floats((size_t)c->hc * c->hidden);
    float *collapsed = alloc_floats((size_t)c->hidden);
    float *normed = alloc_floats((size_t)c->hidden);
    float *branch = alloc_floats((size_t)c->hidden);
    float *post = alloc_floats((size_t)c->hc);
    float *comb = alloc_floats((size_t)c->hc * c->hc);
    for (int h = 0; h < c->hc; h++)
        memcpy(streams + (size_t)h * c->hidden, model->embed + (size_t)token * c->hidden,
               (size_t)c->hidden * sizeof(float));
    for (int i = 0; i < c->layers; i++) {
        Layer *layer = &model->layer[i];
        coli_glm53_mhc_pre(collapsed, post, comb, streams, layer->ah_fn, layer->ah_scale, layer->ah_base, c->hc,
                           c->hidden, c->hc_iters, c->eps, c->hc_eps);
        rmsnorm(normed, collapsed, layer->norm1, 1, c->hidden, c->eps);
        if (layer->kda) kda_step_forward(c, layer, &cache->layers[i], normed, branch);
        else dsa_step_forward(c, layer, &cache->layers[i], cache->length, normed, branch);
        coli_glm53_mhc_post(next, branch, streams, post, comb, c->hc, c->hidden);
        float *swap = streams;
        streams = next;
        next = swap;
        coli_glm53_mhc_pre(collapsed, post, comb, streams, layer->fh_fn, layer->fh_scale, layer->fh_base, c->hc,
                           c->hidden, c->hc_iters, c->eps, c->hc_eps);
        rmsnorm(normed, collapsed, layer->norm2, 1, c->hidden, c->eps);
        memset(branch, 0, (size_t)c->hidden * sizeof(float));
        ffn_forward(c, layer, normed, 1, branch);
        coli_glm53_mhc_post(next, branch, streams, post, comb, c->hc, c->hidden);
        swap = streams;
        streams = next;
        next = swap;
    }
    for (int d = 0; d < c->hidden; d++) {
        float sum = 0.0f;
        for (int h = 0; h < c->hc; h++) sum += streams[(size_t)h * c->hidden + d];
        collapsed[d] = sum / c->hc;
    }
    rmsnorm(normed, collapsed, model->norm, 1, c->hidden, c->eps);
    float *logits = alloc_floats((size_t)c->vocab);
    matmul(logits, normed, model->head, 1, c->hidden, c->vocab);
    cache->length++;
    free(comb);
    free(post);
    free(branch);
    free(normed);
    free(collapsed);
    free(next);
    free(streams);
    return logits;
}

static float *forward(Model *m, const int *tokens, int n) {
    Config *c = &m->c;
    size_t streams_count = (size_t)n * c->hc * c->hidden;
    float *streams = alloc_floats(streams_count);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < c->hc; h++)
            memcpy(streams + ((size_t)t * c->hc + h) * c->hidden, m->embed + (size_t)tokens[t] * c->hidden,
                   (size_t)c->hidden * sizeof(float));
    trace_state("embed", streams, streams_count);
    float *collapsed = alloc_floats((size_t)n * c->hidden), *normed = alloc_floats((size_t)n * c->hidden),
          *branch = alloc_floats((size_t)n * c->hidden), *next = alloc_floats(streams_count),
          *post = alloc_floats((size_t)n * c->hc), *comb = alloc_floats((size_t)n * c->hc * c->hc);
    for (int li = 0; li < c->layers; li++) {
        Layer *l = &m->layer[li];
        for (int t = 0; t < n; t++)
            coli_glm53_mhc_pre(collapsed + (size_t)t * c->hidden, post + (size_t)t * c->hc,
                               comb + (size_t)t * c->hc * c->hc, streams + (size_t)t * c->hc * c->hidden, l->ah_fn,
                               l->ah_scale, l->ah_base, c->hc, c->hidden, c->hc_iters, c->eps, c->hc_eps);
        rmsnorm(normed, collapsed, l->norm1, n, c->hidden, c->eps);
        if (l->kda) kda_forward(c, l, normed, n, branch);
        else dsa_forward(c, l, normed, n, branch);
        if (!l->kda) trace_state("layer.3.attn_branch", branch, (size_t)n * c->hidden);
        for (int t = 0; t < n; t++)
            coli_glm53_mhc_post(next + (size_t)t * c->hc * c->hidden, branch + (size_t)t * c->hidden,
                                streams + (size_t)t * c->hc * c->hidden, post + (size_t)t * c->hc,
                                comb + (size_t)t * c->hc * c->hc, c->hc, c->hidden);
        float *swap = streams;
        streams = next;
        next = swap;
        if (!l->kda) trace_state("layer.3.attn_streams", streams, streams_count);
        for (int t = 0; t < n; t++)
            coli_glm53_mhc_pre(collapsed + (size_t)t * c->hidden, post + (size_t)t * c->hc,
                               comb + (size_t)t * c->hc * c->hc, streams + (size_t)t * c->hc * c->hidden, l->fh_fn,
                               l->fh_scale, l->fh_base, c->hc, c->hidden, c->hc_iters, c->eps, c->hc_eps);
        rmsnorm(normed, collapsed, l->norm2, n, c->hidden, c->eps);
        if (!l->kda) trace_state("layer.3.ffn_norm", normed, (size_t)n * c->hidden);
        ffn_forward(c, l, normed, n, branch);
        if (!l->kda) trace_state("layer.3.ffn_branch", branch, (size_t)n * c->hidden);
        for (int t = 0; t < n; t++)
            coli_glm53_mhc_post(next + (size_t)t * c->hc * c->hidden, branch + (size_t)t * c->hidden,
                                streams + (size_t)t * c->hc * c->hidden, post + (size_t)t * c->hc,
                                comb + (size_t)t * c->hc * c->hc, c->hc, c->hidden);
        swap = streams;
        streams = next;
        next = swap;
        char label[32];
        snprintf(label, sizeof(label), "layer.%d", li);
        trace_state(label, streams, streams_count);
    }
    for (int t = 0; t < n; t++)
        for (int d = 0; d < c->hidden; d++) {
            float sum = 0;
            for (int h = 0; h < c->hc; h++) sum += streams[((size_t)t * c->hc + h) * c->hidden + d];
            collapsed[(size_t)t * c->hidden + d] = sum / c->hc;
        }
    rmsnorm(normed, collapsed, m->norm, n, c->hidden, c->eps);
    trace_state("final", normed, (size_t)n * c->hidden);
    float *logits = alloc_floats((size_t)n * c->vocab);
    matmul(logits, normed, m->head, n, c->hidden, c->vocab);
    free(comb);
    free(post);
    free(next);
    free(branch);
    free(normed);
    free(collapsed);
    free(streams);
    return logits;
}

static int parse_ids(const char *text, int **output) {
    char *copy = strdup(text);
    int cap = 16, n = 0;
    int *ids = malloc((size_t)cap * sizeof(int));
    for (char *p = strtok(copy, ","); p; p = strtok(NULL, ",")) {
        if (n == cap) {
            cap *= 2;
            ids = realloc(ids, (size_t)cap * sizeof(int));
        }
        ids[n++] = atoi(p);
    }
    free(copy);
    *output = ids;
    return n;
}
int main(int argc, char **argv) {
    if (argc < 4 || strcmp(argv[2], "--ids")) {
        fprintf(stderr, "usage: %s MODEL --ids 1,2,3 [--greedy N] [--cached]\n", argv[0]);
        return 2;
    }
    Model model;
    model_load(&model, argv[1]);
    int *ids = NULL, n = parse_ids(argv[3], &ids);
    int greedy = 0, cached = 0;
    for (int arg = 4; arg < argc; arg++) {
        if (!strcmp(argv[arg], "--cached")) cached = 1;
        else if (!strcmp(argv[arg], "--greedy") && arg + 1 < argc) greedy = atoi(argv[++arg]);
        else {
            fprintf(stderr, "glm53: unknown argument %s\n", argv[arg]);
            return 2;
        }
    }
    if (cached) {
        RuntimeCache cache;
        cache_init(&cache, &model.c);
        float *logits = NULL;
        printf("teacher");
        for (int i = 0; i < n; i++) {
            free(logits);
            logits = forward_cached_step(&model, &cache, ids[i]);
            int best = 0;
            for (int v = 1; v < model.c.vocab; v++)
                if (logits[v] > logits[best]) best = v;
            printf(" %d", best);
        }
        printf("\nlast_logits");
        for (int v = 0; v < model.c.vocab; v++) printf(" %.9g", logits[v]);
        printf("\n");
        for (int step = 0; step < greedy; step++) {
            int best = 0;
            for (int v = 1; v < model.c.vocab; v++)
                if (logits[v] > logits[best]) best = v;
            printf("greedy %d\n", best);
            free(logits);
            logits = forward_cached_step(&model, &cache, best);
        }
        free(logits);
        cache_free(&cache, &model.c);
        free(ids);
        model_free(&model);
        return 0;
    }
    for (int step = 0; step <= greedy; step++) {
        float *logits = forward(&model, ids, n);
        if (!step) {
            printf("teacher");
            for (int t = 0; t < n; t++) {
                int best = 0;
                for (int v = 1; v < model.c.vocab; v++)
                    if (logits[(size_t)t * model.c.vocab + v] > logits[(size_t)t * model.c.vocab + best]) best = v;
                printf(" %d", best);
            }
            printf("\nlast_logits");
            float *last = logits + (size_t)(n - 1) * model.c.vocab;
            for (int v = 0; v < model.c.vocab; v++) printf(" %.9g", last[v]);
            printf("\n");
        }
        if (step < greedy) {
            int best = 0;
            float *last = logits + (size_t)(n - 1) * model.c.vocab;
            for (int v = 1; v < model.c.vocab; v++)
                if (last[v] > last[best]) best = v;
            ids = realloc(ids, (size_t)(n + 1) * sizeof(int));
            ids[n++] = best;
            printf("greedy %d\n", best);
        }
        free(logits);
    }
    free(ids);
    model_free(&model);
    return 0;
}
