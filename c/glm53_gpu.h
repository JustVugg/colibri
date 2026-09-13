#ifndef COLIBRI_GLM53_GPU_H
#define COLIBRI_GLM53_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct ColiGpuContext ColiGpuContext;
typedef struct ColiGlm53GpuModel ColiGlm53GpuModel;
typedef struct ColiGlm53GpuSession ColiGlm53GpuSession;

#ifdef __cplusplus
extern "C" {
#endif

enum {
    COLI_GLM53_GPU_REQUIRED_CAPS =
        (1ull << 0) | (1ull << 1) | (1ull << 2)
};

typedef enum {
    COLI_GLM53_GPU_MODE_AUTO = 0,
    COLI_GLM53_GPU_MODE_GPU,
    COLI_GLM53_GPU_MODE_CPU
} ColiGlm53GpuMode;

typedef enum {
    COLI_GLM53_BACKEND_UNSELECTED = 0,
    COLI_GLM53_BACKEND_CPU,
    COLI_GLM53_BACKEND_GPU
} ColiGlm53BackendKind;

typedef struct {
    int (*context_create)(ColiGpuContext **out, int device);
    int (*context_probe)(ColiGpuContext *ctx, uint64_t required_caps);
    void (*context_destroy)(ColiGpuContext *ctx);
} ColiGlm53GpuOps;

typedef struct {
    ColiGlm53GpuMode requested;
    ColiGlm53BackendKind selected;
    ColiGpuContext *ctx;
    int device;
    int active_requests;
} ColiGlm53GpuBackend;

typedef struct {
    const void *data;
    const float *scales;
    int format;
    int rows;
    int columns;
    int group_size;
} ColiGlm53GpuWeightDesc;

typedef struct {
    const float *fn;          /* [(2 + stream_count) * stream_count,
                                  stream_count * hidden_size] */
    const float *base;        /* [(2 + stream_count) * stream_count] */
    const float *scale;       /* [3] */
    const float *norm_weight; /* [hidden_size] */
} ColiGlm53GpuMhcSiteDesc;

typedef struct {
    const float *q_proj;       /* [kda_heads * kda_head_dim, hidden_size] */
    const float *k_proj;       /* [kda_heads * kda_head_dim, hidden_size] */
    const float *v_proj;       /* [kda_heads * kda_head_dim, hidden_size] */
    const float *o_proj;       /* [hidden_size, kda_heads * kda_head_dim] */
    const float *gate_a_proj;  /* [kda_head_dim, hidden_size] */
    const float *gate_b_proj;  /* [kda_heads * kda_head_dim, kda_head_dim] */
    const float *decay_a_proj; /* [kda_head_dim, hidden_size] */
    const float *decay_b_proj; /* [kda_heads * kda_head_dim, kda_head_dim] */
    const float *beta_proj;    /* [kda_heads, hidden_size] */
    const float *conv;         /* [3 * kda_heads * kda_head_dim, kda_kernel] */
    const float *dt_bias;      /* [kda_heads * kda_head_dim] */
    const float *a_log;        /* [kda_heads] */
    const float *o_norm;       /* [kda_head_dim] */
    /* Production loader path.  A non-NULL data pointer overrides the
     * corresponding fp32 compatibility pointer above. */
    ColiGlm53GpuWeightDesc q_proj_weight;
    ColiGlm53GpuWeightDesc k_proj_weight;
    ColiGlm53GpuWeightDesc v_proj_weight;
    ColiGlm53GpuWeightDesc o_proj_weight;
    ColiGlm53GpuWeightDesc gate_a_proj_weight;
    ColiGlm53GpuWeightDesc gate_b_proj_weight;
    ColiGlm53GpuWeightDesc decay_a_proj_weight;
    ColiGlm53GpuWeightDesc decay_b_proj_weight;
    ColiGlm53GpuWeightDesc beta_proj_weight;
} ColiGlm53GpuKdaLayerDesc;

typedef struct {
    const float *q_a_proj;
    const float *q_a_norm;
    const float *q_b_proj;
    const float *kv_a_proj;
    const float *kv_a_norm;
    const float *kv_b_key;
    const float *kv_b_value;
    const float *o_proj;
    const float *index_q_proj;
    const float *index_k_proj;
    const float *index_weight_proj;
    const float *index_key_norm;
    const float *index_key_bias;
    const float *index_pool_ape;
    const float *index_pool_gate;
    ColiGlm53GpuWeightDesc q_a_proj_weight;
    ColiGlm53GpuWeightDesc q_b_proj_weight;
    ColiGlm53GpuWeightDesc kv_a_proj_weight;
    ColiGlm53GpuWeightDesc kv_b_key_weight;
    ColiGlm53GpuWeightDesc kv_b_value_weight;
    ColiGlm53GpuWeightDesc o_proj_weight;
    ColiGlm53GpuWeightDesc index_q_proj_weight;
    ColiGlm53GpuWeightDesc index_k_proj_weight;
    ColiGlm53GpuWeightDesc index_weight_proj_weight;
    ColiGlm53GpuWeightDesc index_pool_gate_weight;
} ColiGlm53GpuMlaLayerDesc;

typedef struct {
    const float *router;           /* [moe_experts, hidden_size] */
    const float *correction_bias;  /* [moe_experts], optional */
    ColiGlm53GpuWeightDesc shared_gate;
    ColiGlm53GpuWeightDesc shared_up;
    ColiGlm53GpuWeightDesc shared_down;
} ColiGlm53GpuMoeLayerDesc;

typedef struct {
    ColiGlm53GpuWeightDesc gate;
    ColiGlm53GpuWeightDesc up;
    ColiGlm53GpuWeightDesc down;
} ColiGlm53GpuExpertDesc;

typedef struct {
    ColiGlm53GpuWeightDesc gate;
    ColiGlm53GpuWeightDesc up;
    ColiGlm53GpuWeightDesc down;
} ColiGlm53GpuDenseLayerDesc;

typedef enum {
    COLI_GLM53_GPU_ATTN_KDA = 0,
    COLI_GLM53_GPU_ATTN_MLA = 1
} ColiGlm53GpuAttentionKind;

typedef enum {
    COLI_GLM53_GPU_FFN_DENSE = 0,
    COLI_GLM53_GPU_FFN_MOE = 1
} ColiGlm53GpuFfnKind;

typedef struct {
    ColiGlm53GpuAttentionKind attention_kind;
    int attention_index;
    ColiGlm53GpuFfnKind ffn_kind;
    int ffn_index;
    int attention_site;
    int ffn_site;
} ColiGlm53GpuLayerDesc;

typedef int (*ColiGlm53GpuExpertLoader)(
    void *user, int pipeline_layer, int expert_id, int *slot,
    ColiGlm53GpuExpertDesc *expert);

typedef struct {
    int expert_id;
    int slot;
    uint64_t generation;
} ColiGlm53GpuExpertHandle;

typedef struct {
    int hidden_size;
    int stream_count;
    int vocab_size;
    int max_prefill_rows;
    int max_context_tokens;        /* zero defaults to max_prefill_rows */
    float norm_eps;
    float hc_eps;
    const float *embedding;       /* [vocab_size, hidden_size], fp32 */
    const float *final_norm;      /* [hidden_size], fp32 */
    const void *lm_head;          /* [vocab_size, hidden_size] */
    const float *lm_head_scales;  /* fmt 1/2: [vocab_size]; fmt 4:
                                     [vocab_size,
                                      ceil(hidden_size/lm_head_group_size)] */
    int lm_head_format;           /* 0=f32, 1=int8, 2=int4, 4=int4-gs */
    int lm_head_group_size;
    const ColiGlm53GpuMhcSiteDesc *sites;
    int site_count;
    int kda_heads;
    int kda_head_dim;
    int kda_kernel;
    float kda_gate_lower_bound;
    const ColiGlm53GpuKdaLayerDesc *kda_layers;
    int kda_layer_count;
    int mla_heads;
    int mla_q_lora;
    int mla_kv_lora;
    int mla_qk_nope;
    int mla_qk_rope;
    int mla_value_dim;
    int mla_index_heads;
    int mla_index_dim;
    int mla_index_pool;
    int mla_index_topk;
    int mla_index_select_tail;
    int mla_page_tokens;
    const ColiGlm53GpuMlaLayerDesc *mla_layers;
    int mla_layer_count;
    int dense_intermediate;
    float swiglu_limit;
    const ColiGlm53GpuDenseLayerDesc *dense_layers;
    int dense_layer_count;
    int moe_experts;
    int moe_topk;
    int moe_intermediate;
    int moe_cache_slots;
    int moe_group_size;
    int moe_normalize_topk;
    float moe_routed_scale;
    float moe_swiglu_limit;
    const ColiGlm53GpuMoeLayerDesc *moe_layers;
    int moe_layer_count;
    const ColiGlm53GpuLayerDesc *layers;
    int layer_count;
    ColiGlm53GpuExpertLoader expert_loader;
    void *expert_loader_user;
} ColiGlm53GpuModelDesc;

typedef struct {
    int kda_layer;
    int mla_layer;
    int dense_layer;
    int moe_layer;
    float *kda_output;
    float *mla_output;
    float *dense_output;
    float *moe_output;
    float *final_norm;
    size_t output_floats;
} ColiGlm53GpuQualityCapture;

int coli_glm53_gpu_mode_parse(const char *text, ColiGlm53GpuMode *out);
const char *coli_glm53_gpu_mode_name(ColiGlm53GpuMode mode);
const char *coli_glm53_backend_name(ColiGlm53BackendKind backend);
void coli_glm53_gpu_backend_init(ColiGlm53GpuBackend *backend);
int coli_glm53_gpu_backend_select(ColiGlm53GpuBackend *backend,
                                  ColiGlm53GpuMode mode,
                                  ColiGlm53GpuOps ops,
                                  int device,
                                  uint64_t required_caps,
                                  char *err,
                                  size_t err_cap);
ColiGlm53BackendKind coli_glm53_gpu_backend_selected(const ColiGlm53GpuBackend *backend);
ColiGpuContext *coli_glm53_gpu_backend_context(const ColiGlm53GpuBackend *backend);
int coli_glm53_gpu_backend_begin_request(ColiGlm53GpuBackend *backend);
int coli_glm53_gpu_backend_fail_request(ColiGlm53GpuBackend *backend);
void coli_glm53_gpu_backend_end_request(ColiGlm53GpuBackend *backend);
void coli_glm53_gpu_backend_destroy(ColiGlm53GpuBackend *backend,
                                    ColiGlm53GpuOps ops);

int coli_glm53_gpu_model_create(ColiGlm53GpuModel **out,
                                ColiGpuContext *ctx,
                                const ColiGlm53GpuModelDesc *desc);
void coli_glm53_gpu_model_destroy(ColiGlm53GpuModel *model);
int coli_glm53_gpu_session_create(ColiGlm53GpuSession **out,
                                  ColiGlm53GpuModel *model,
                                  int max_context);
void coli_glm53_gpu_session_destroy(ColiGlm53GpuSession *session);
int coli_glm53_gpu_session_embed(ColiGlm53GpuSession *session,
                                 const int32_t *token_ids, int rows);
int coli_glm53_gpu_session_core_site(ColiGlm53GpuSession *session,
                                     int site_index, int rows);
int coli_glm53_gpu_session_kda_upload_input(ColiGlm53GpuSession *session,
                                            const float *input, int rows);
int coli_glm53_gpu_session_kda_site(ColiGlm53GpuSession *session,
                                    int layer_index, int rows,
                                    int start_position);
int coli_glm53_gpu_session_kda_download_output(
    ColiGlm53GpuSession *session, float *output, int rows);
int coli_glm53_gpu_session_kda_state_download(
    ColiGlm53GpuSession *session, int layer_index,
    float *matrix, size_t matrix_floats,
    float *window, size_t window_floats);
int coli_glm53_gpu_session_mla_upload_input(
    ColiGlm53GpuSession *session, const float *input, int rows);
int coli_glm53_gpu_session_mla_site(
    ColiGlm53GpuSession *session, int layer_index, int rows,
    int start_position);
int coli_glm53_gpu_session_mla_download_output(
    ColiGlm53GpuSession *session, float *output,
    int *selected, size_t selected_count, int rows);
int coli_glm53_gpu_session_mla_state_info(
    ColiGlm53GpuSession *session, int layer_index,
    int *logical_length, int *capacity);
int coli_glm53_gpu_session_mla_state_download(
    ColiGlm53GpuSession *session, int layer_index,
    float *latent, size_t latent_floats,
    float *index_keys, size_t index_key_floats,
    float *index_gates, size_t index_gate_floats);
int coli_glm53_gpu_session_mla_reset_layer(
    ColiGlm53GpuSession *session, int layer_index);
int coli_glm53_gpu_model_moe_upload_expert(
    ColiGlm53GpuModel *model, int layer_index, int expert_id, int slot,
    const ColiGlm53GpuExpertDesc *expert,
    ColiGlm53GpuExpertHandle *handle);
int coli_glm53_gpu_session_moe_upload_input(
    ColiGlm53GpuSession *session, const float *input, int rows);
int coli_glm53_gpu_session_moe_route(
    ColiGlm53GpuSession *session, int layer_index, int rows);
int coli_glm53_gpu_session_moe_selected(
    ColiGlm53GpuSession *session, int *selected_ids, float *routing_weights,
    size_t selected_count, int rows);
int coli_glm53_gpu_session_moe_site(
    ColiGlm53GpuSession *session, int layer_index,
    const ColiGlm53GpuExpertHandle *handles, size_t handle_count, int rows);
int coli_glm53_gpu_session_moe_download_output(
    ColiGlm53GpuSession *session, float *output, int rows);
int coli_glm53_gpu_session_reset(ColiGlm53GpuSession *session);
int coli_glm53_gpu_session_set_quality_capture(
    ColiGlm53GpuSession *session,
    const ColiGlm53GpuQualityCapture *capture);
int coli_glm53_gpu_session_output(ColiGlm53GpuSession *session,
                                  float *logits, int rows);
int coli_glm53_gpu_forward(ColiGlm53GpuSession *session,
                           const int *token_ids, int rows,
                           float *last_logits_host);
int coli_glm53_gpu_pipeline_probe(ColiGpuContext *ctx);

#ifdef __cplusplus
}
#endif

#endif
