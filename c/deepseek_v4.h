#ifndef COLIBRI_DEEPSEEK_V4_H
#define COLIBRI_DEEPSEEK_V4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- config ---- */

#define COLI_V4_MAX_LAYERS 128

typedef struct {
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int head_dim;
    int q_lora_rank;
    int qk_rope_head_dim;
    int o_groups;
    int o_lora_rank;
    int sliding_window;
    int index_n_heads;
    int index_head_dim;
    int index_topk;
    int n_routed_experts;
    int num_experts_per_tok;
    int n_shared_experts;
    int moe_intermediate_size;
    int num_hash_layers;
    int num_nextn_predict_layers;
    int dspark_block_size;
    int dspark_noise_token_id;
    int dspark_markov_rank;
    int hc_mult;
    int hc_sinkhorn_iters;
    int vocab_size;
    int max_position_embeddings;
    int original_max_position_embeddings;
    int compress_ratio_count;
    int compress_ratios[COLI_V4_MAX_LAYERS];
    float rms_norm_eps;
    float hc_eps;
    float routed_scaling_factor;
    float swiglu_limit;
    float rope_theta;
    float rope_factor;
    float compress_rope_theta;
    int rope_beta_fast;
    int rope_beta_slow;
} ColiDeepSeekV4Config;

int coli_v4_config_parse(ColiDeepSeekV4Config *config, const char *json,
                         char *error, size_t error_size);
int coli_v4_config_load(ColiDeepSeekV4Config *config, const char *model_dir,
                        char *error, size_t error_size);

/* ---- prompt ---- */

typedef enum {
    COLI_V4_PROMPT_CHAT,
    COLI_V4_PROMPT_THINKING,
    COLI_V4_PROMPT_RAW,
} ColiDeepSeekV4PromptMode;

int coli_v4_prompt_build(char **output, size_t *output_length,
                         const char *user_message, const char *system_message,
                         ColiDeepSeekV4PromptMode mode);

/* ---- experimental public engine API ---- */
/*
 * The API is scoped to the DeepSeek V4 engine and may change while
 * the implementation remains experimental.
 */

typedef struct ColiV4Engine ColiV4Engine;

typedef struct {
    /* Copied by coli_v4_engine_open; caller strings need not outlive the engine. */
    const char *target_model_dir;   /* required */
    uint64_t memory_limit_bytes;    /* 0 => use OS available memory */
    int context_tokens;             /* 0 => 4096 */
    int pin_slots_per_layer;        /* -1 => auto */
    uint64_t repin_interval;        /* 0 => auto */
    int no_dspark;                  /* disable speculative draft/verification */
} ColiV4EngineOpenOptions;

typedef struct {
    uint64_t projected_bytes;
    uint64_t expert_cache_bytes;
    int slots_per_layer;
    int dense_resident;
    int head_resident;
} ColiV4EngineMemorySummary;

int coli_v4_engine_open(ColiV4Engine **engine,
                        const ColiV4EngineOpenOptions *options,
                        char *error, size_t error_size);
/* Undefined if any ColiV4Session created from this engine is still alive.
 * Destroy every session before destroying the engine. */
void coli_v4_engine_destroy(ColiV4Engine *engine);

const ColiDeepSeekV4Config *coli_v4_engine_config(const ColiV4Engine *engine);
void coli_v4_engine_memory_summary(const ColiV4Engine *engine,
                                   ColiV4EngineMemorySummary *summary);

const char *coli_v4_engine_target_model_dir(const ColiV4Engine *engine);

/* ---- experimental public session API ----
 * Session borrows the engine; destroy all sessions before coli_v4_engine_destroy.
 */

typedef struct ColiV4Session ColiV4Session;

typedef struct {
    int max_prompt_tokens;   /* 0 => 512 */
    int max_new_tokens_cap;  /* 0 => 512 */
} ColiV4SessionCreateOptions;

typedef struct {
    int max_new_tokens;      /* required; clamped by session cap */
    int stop_at_sentence;
    int no_dspark;           /* disable speculative draft/verification */
} ColiV4SessionGenerateOptions;

typedef struct {
    int prompt_tokens;
    int generated_tokens;
    int eos_stopped;
    double time_to_first_token_sec;
    double decode_sec;
    uint64_t speculative_drafted;
    uint64_t speculative_accepted;
} ColiV4SessionGenerateStats;

/* Return non-zero to stop generation. */
typedef int (*ColiV4SessionTokenFn)(void *user_data, int token, float logit,
                                    int position, int ordinal);

int coli_v4_session_create(ColiV4Session **session, ColiV4Engine *engine,
                           const ColiV4SessionCreateOptions *options,
                           char *error, size_t error_size);
void coli_v4_session_destroy(ColiV4Session *session);

int coli_v4_session_generate(ColiV4Session *session,
                             const char *prompt, size_t prompt_length,
                             const ColiV4SessionGenerateOptions *options,
                             ColiV4SessionTokenFn on_token, void *user_data,
                             ColiV4SessionGenerateStats *stats,
                             char *error, size_t error_size);

int coli_v4_session_generated_text(const ColiV4Session *session,
                                   char *buffer, size_t buffer_size,
                                   size_t *out_length);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_DEEPSEEK_V4_H */
