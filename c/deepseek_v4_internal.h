#ifndef COLIBRI_DEEPSEEK_V4_INTERNAL_H
#define COLIBRI_DEEPSEEK_V4_INTERNAL_H

/*
 * Internal DeepSeek-V4 API. Not a stability commitment.
 * External callers should use deepseek_v4.h (engine / config / prompt).
 */
#include "deepseek_v4.h"

#include "tensor.h"
#include "expert_store.h"
#include "native_quant.h"
#include "native_quant_batch.h"
#include "native_quant_dual.h"
#include "native_quant_fp4_rows16.h"
#include "st.h"

/* Worker count for the persistent expert-loader pool in the block pipeline
 * (deepseek_v4_block_pipeline.c). Shared here so the CLI can size the OpenMP
 * team around the loaders instead of scheduling compute onto their CPUs. */
#ifndef COLI_V4_EXPERT_LOADER_COUNT
#define COLI_V4_EXPERT_LOADER_COUNT 3
#endif

#define COLI_ST_MAX_RANK ST_MAX_RANK
#define COLI_ST_BF16 0
#define COLI_ST_F16 1
#define COLI_ST_F32 2
#define COLI_ST_U8 3
#define COLI_ST_I8 3
#define COLI_ST_F8_E4M3 4
#define COLI_ST_F8_E8M0 5
#define COLI_ST_I64 6

typedef int ColiSafetensorsDType;
typedef st_tensor ColiSafetensorsTensor;
typedef shards ColiSafetensorsIndex;

typedef struct {
    ColiTensorView view;
    void *data_allocation;
    void *scale_allocation;
} ColiOwnedTensor;

typedef struct {
    float *data;
    uint64_t count;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
} ColiFloatTensor;

int coli_st_index_open(ColiSafetensorsIndex **out, const char *directory,
                       char *error, size_t error_size);
void coli_st_index_close(ColiSafetensorsIndex *index);
size_t coli_st_tensor_count(const ColiSafetensorsIndex *index);
size_t coli_st_shard_count(const ColiSafetensorsIndex *index);
const char *coli_st_shard_path(const ColiSafetensorsIndex *index, int shard);
const ColiSafetensorsTensor *coli_st_find(const ColiSafetensorsIndex *index,
                                         const char *name);
int coli_st_tensor_shard(const ColiSafetensorsIndex *index,
                         const ColiSafetensorsTensor *tensor);
int coli_st_read_tensor(const ColiSafetensorsIndex *index,
                        const ColiSafetensorsTensor *tensor, void *destination);
int coli_st_read_at(const ColiSafetensorsIndex *index, int shard,
                    uint64_t offset, size_t length, void *destination);
/* Large, transient SSD read: prefer the index's O_DIRECT twin and use an
 * aligned bounce buffer, falling back to the ordinary buffered path. */
int coli_st_read_at_streaming(const ColiSafetensorsIndex *index, int shard,
                              uint64_t offset, size_t length,
                              void *destination);
int coli_st_streaming_direct_available(const ColiSafetensorsIndex *index,
                                       int shard);
int coli_st_prefetch_at(const ColiSafetensorsIndex *index, int shard,
                        uint64_t offset, size_t length);
const char *coli_st_dtype_name(ColiSafetensorsDType dtype);

int coli_tensor_load_fp8(ColiOwnedTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *prefix, char *error, size_t error_size);
void coli_owned_tensor_free(ColiOwnedTensor *tensor);
int coli_tensor_load_f32(ColiFloatTensor *output,
                         const ColiSafetensorsIndex *index,
                         const char *name, char *error, size_t error_size);
void coli_float_tensor_free(ColiFloatTensor *tensor);

typedef struct ColiV4Engine ColiV4Engine;

/* Runtime-selected full DSpark profile, shared with the separately compiled
 * generation unit. */
extern int coli_v4_full_dspark_wanted;
double coli_v4_dspark_cache_gb(void);

/* ==== begin deepseek_v4_math.h ==== */

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_hc_split_sinkhorn(float *pre, float *post, float *comb,
                              const float *mixes, const float scale[3],
                              const float *base, int hc, int iterations,
                              float eps);

int coli_v4_hc_pre(float *output, float *post, float *comb,
                   const float *input, const float *hc_fn,
                   const float scale[3], const float *base,
                   int hc, int dimension, int iterations,
                   float norm_eps, float hc_eps);

int coli_v4_hc_post(float *output, const float *branch,
                    const float *residual, const float *post,
                    const float *comb, int hc, int dimension);

int coli_v4_rmsnorm(float *output, const float *input, const float *weight,
                    int dimension, float eps);

int coli_v4_rope_precompute(float *cosines, float *sines,
                            int dimension, int sequence_length,
                            int original_sequence_length, float base,
                            float factor, int beta_fast, int beta_slow);

int coli_v4_rope_precompute_range(float *cosines, float *sines,
                                  int dimension, int start_position,
                                  int sequence_length,
                                  int original_sequence_length, float base,
                                  float factor, int beta_fast, int beta_slow);

int coli_v4_rope_position(float *cosines, float *sines,
                          int dimension, int position,
                          int original_sequence_length, float base,
                          float factor, int beta_fast, int beta_slow);

int coli_v4_rope_apply(float *vectors, int vector_count, int dimension,
                       const float *cosines, const float *sines, int inverse);

int coli_v4_route(float *weights, int *indices, const float *hidden,
                  const float *gate, const float *bias,
                  const int *forced_indices, int experts, int dimension,
                  int topk, float route_scale);

int coli_v4_swiglu(float *output, const float *gate, const float *up,
                   int dimension, float limit);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_math.h ==== */

/* ==== begin deepseek_v4_layer.h ==== */

#include <stddef.h>
#include <stdint.h>

/* amalgamated: deepseek_v4_config.h */

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_V4_MAX_LAYER_TENSORS 48
#define COLI_V4_MAX_TENSOR_NAME 160

typedef struct {
    char name[COLI_V4_MAX_TENSOR_NAME];
    ColiSafetensorsDType dtype;
    int rank;
    int64_t shape[COLI_ST_MAX_RANK];
    /* Dense FP8 weights may be transposed inside each 8-row tile after load.
     * This is an in-memory execution layout only; checkpoint bytes and scales
     * remain unchanged. */
    int packed_rows8;
} ColiDeepSeekV4TensorSpec;

typedef struct {
    int layer;
    int compression_ratio;
    int uses_hash_router;
    int has_compressor;
    int has_indexer;
    size_t tensor_count;
    ColiDeepSeekV4TensorSpec tensors[COLI_V4_MAX_LAYER_TENSORS];
} ColiDeepSeekV4LayerPlan;

typedef struct {
    size_t tensor_count;
    uint64_t total_bytes;
    uint64_t bf16_bytes;
    uint64_t f32_bytes;
    uint64_t fp8_weight_bytes;
    uint64_t fp8_scale_bytes;
    uint64_t i64_bytes;
} ColiDeepSeekV4LayerStats;

typedef struct {
    ColiDeepSeekV4LayerPlan plan;
    ColiDeepSeekV4LayerStats stats;
    void *data[COLI_V4_MAX_LAYER_TENSORS];
} ColiDeepSeekV4LayerWeights;

int coli_v4_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                       const ColiDeepSeekV4Config *config, int layer,
                       char *error, size_t error_size);
int coli_v4_layer_validate(const ColiDeepSeekV4LayerPlan *plan,
                           const ColiSafetensorsIndex *index,
                           ColiDeepSeekV4LayerStats *stats,
                           char *error, size_t error_size);
int coli_v4_layer_load(ColiV4Engine *engine,
                       ColiDeepSeekV4LayerWeights *weights,
                       const ColiDeepSeekV4Config *config,
                       const ColiSafetensorsIndex *index, int layer,
                       char *error, size_t error_size);
void coli_v4_layer_free(ColiV4Engine *engine,
                        ColiDeepSeekV4LayerWeights *weights);
const void *coli_v4_layer_data(const ColiDeepSeekV4LayerWeights *weights,
                               const char *name,
                               const ColiDeepSeekV4TensorSpec **spec);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_layer.h ==== */

/* ==== begin deepseek_v4_sparse_attention.h ==== */

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_sparse_attention_ref(float *output, const float *queries,
                                 const float *kv, const float *sinks,
                                 const int *indices, int heads,
                                 int head_dimension, int kv_count, int topk,
                                 float softmax_scale);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_sparse_attention.h ==== */

/* ==== begin deepseek_v4_kv_cache.h ==== */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4KVCache ColiDeepSeekV4KVCache;

int coli_v4_kv_cache_create(ColiDeepSeekV4KVCache **cache,
                            int window_size, int compression_ratio,
                            int head_dimension, int max_context);
void coli_v4_kv_cache_reset(ColiDeepSeekV4KVCache *cache);
void coli_v4_kv_cache_destroy(ColiDeepSeekV4KVCache *cache);
int coli_v4_kv_cache_put_window(ColiDeepSeekV4KVCache *cache,
                                int position, const float *kv);
int coli_v4_kv_cache_put_compressed(ColiDeepSeekV4KVCache *cache,
                                    int position, const float *kv);
int coli_v4_kv_cache_indices(const ColiDeepSeekV4KVCache *cache,
                             int position, int *indices, size_t capacity);
const float *coli_v4_kv_cache_values(const ColiDeepSeekV4KVCache *cache);
int coli_v4_kv_cache_value_count(const ColiDeepSeekV4KVCache *cache);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_kv_cache.h ==== */

/* ==== begin deepseek_v4_attention_cache.h ==== */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4AttentionCache ColiDeepSeekV4AttentionCache;

int coli_v4_attention_cache_create(ColiDeepSeekV4AttentionCache **cache,
                                   int window_size, int compression_ratio,
                                   int head_dimension, int max_context);
void coli_v4_attention_cache_reset(ColiDeepSeekV4AttentionCache *cache);
void coli_v4_attention_cache_destroy(ColiDeepSeekV4AttentionCache *cache);

/* query is [heads, head_dimension]. window_kv and compressed_kv have one
 * head_dimension vector each. compressed_kv is required at ratio boundaries. */
int coli_v4_attention_cache_step(ColiDeepSeekV4AttentionCache *cache,
                                 float *output, const float *query,
                                 const float *window_kv,
                                 const float *compressed_kv,
                                 const float *sinks, int heads,
                                 int position, float softmax_scale);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_attention_cache.h ==== */

/* ==== begin deepseek_v4_attention.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4WindowAttentionState
    ColiDeepSeekV4WindowAttentionState;

int coli_v4_window_attention_create(ColiDeepSeekV4WindowAttentionState **state,
                                    const ColiDeepSeekV4Config *config);
void coli_v4_window_attention_reset(ColiDeepSeekV4WindowAttentionState *state);
void coli_v4_window_attention_destroy(ColiDeepSeekV4WindowAttentionState *state);

/* Correctness-first single-KV attention. Compressed layers may use this at
 * position zero, before any compressed KV/indexer candidate exists. */
int coli_v4_attention_token_ref(float *output,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size);
int coli_v4_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_attention.h ==== */

/* ==== begin deepseek_v4_attention_batch.h ==== */

/* amalgamated: deepseek_v4_attention.h */

int coli_v4_attention_window_batch_ref(
    float *outputs, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size);
/* ==== end deepseek_v4_attention_batch.h ==== */

/* ==== begin deepseek_v4_attention_transaction.h ==== */

/* amalgamated: deepseek_v4_attention.h */

typedef struct ColiV4AttentionSnapshot ColiV4AttentionSnapshot;

int coli_v4_attention_snapshot_create(
    const ColiDeepSeekV4WindowAttentionState *state,
    ColiV4AttentionSnapshot **output);
int coli_v4_attention_snapshot_restore(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot);
void coli_v4_attention_snapshot_destroy(ColiV4AttentionSnapshot *snapshot);
/* ==== end deepseek_v4_attention_transaction.h ==== */

/* ==== begin deepseek_v4_compressor.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4CompressorState ColiDeepSeekV4CompressorState;

typedef struct {
    const char *prefix;
    int head_dimension;
    int rotate_fp4;
} ColiDeepSeekV4CompressorOptions;

int coli_v4_compressor_create(ColiDeepSeekV4CompressorState **state,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              char *error, size_t error_size);
int coli_v4_compressor_create_with_options(
    ColiDeepSeekV4CompressorState **state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size);
void coli_v4_compressor_reset(ColiDeepSeekV4CompressorState *state);
int coli_v4_compressor_bind_weights(ColiDeepSeekV4CompressorState *state,
                                    const ColiDeepSeekV4LayerWeights *weights,
                                    char *error, size_t error_size);
void coli_v4_compressor_destroy(ColiDeepSeekV4CompressorState *state);

/* Processes one decode token. produced is set to one only when a complete
 * compression window emits a KV vector. output may be NULL on other steps. */
int coli_v4_compressor_step(ColiDeepSeekV4CompressorState *state,
                            float *output, int *produced,
                            const float *input, int position,
                            char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_compressor.h ==== */

/* ==== begin deepseek_v4_compressor_snapshot.h ==== */

/* amalgamated: deepseek_v4_compressor.h */

typedef struct ColiV4CompressorSnapshot ColiV4CompressorSnapshot;

int coli_v4_compressor_snapshot_create(
    const ColiDeepSeekV4CompressorState *state,
    ColiV4CompressorSnapshot **output);
int coli_v4_compressor_snapshot_restore(
    ColiDeepSeekV4CompressorState *state,
    const ColiV4CompressorSnapshot *snapshot);
void coli_v4_compressor_snapshot_destroy(ColiV4CompressorSnapshot *snapshot);
/* ==== end deepseek_v4_compressor_snapshot.h ==== */

/* ==== begin deepseek_v4_indexer.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */

typedef struct ColiDeepSeekV4Indexer ColiDeepSeekV4Indexer;

int coli_v4_indexer_create(ColiDeepSeekV4Indexer **state,
                           const ColiDeepSeekV4LayerWeights *weights,
                           const ColiDeepSeekV4Config *config,
                           int max_context, char *error, size_t error_size);
int coli_v4_indexer_bind_weights(ColiDeepSeekV4Indexer *state,
                                 const ColiDeepSeekV4LayerWeights *weights,
                                 char *error, size_t error_size);
void coli_v4_indexer_reset(ColiDeepSeekV4Indexer *state);
void coli_v4_indexer_destroy(ColiDeepSeekV4Indexer *state);

/* Updates the overlap compressor, then returns compressed-cache ordinals in
 * descending index score order. query_rank is the normalized q_lora vector. */
int coli_v4_indexer_step(ColiDeepSeekV4Indexer *state, int *indices,
                         int index_capacity, const float *query_rank,
                         const float *input, int position,
                         char *error, size_t error_size);
const float *coli_v4_indexer_compressed_values(
    const ColiDeepSeekV4Indexer *state);
int coli_v4_indexer_compressed_count(const ColiDeepSeekV4Indexer *state);
/* ==== end deepseek_v4_indexer.h ==== */

/* ==== begin deepseek_v4_indexer_snapshot.h ==== */

/* amalgamated: deepseek_v4_indexer.h */

typedef struct ColiV4IndexerSnapshot ColiV4IndexerSnapshot;

int coli_v4_indexer_snapshot_create(const ColiDeepSeekV4Indexer *state,
                                    ColiV4IndexerSnapshot **output);
int coli_v4_indexer_snapshot_restore(ColiDeepSeekV4Indexer *state,
                                     const ColiV4IndexerSnapshot *snapshot);
void coli_v4_indexer_snapshot_destroy(ColiV4IndexerSnapshot *snapshot);
/* ==== end deepseek_v4_indexer_snapshot.h ==== */

/* ==== begin deepseek_v4_expert.h ==== */

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit);

int coli_v4_shared_expert_forward_ref(float *output,
                                      const ColiTensorView *gate,
                                      const ColiTensorView *down,
                                      const ColiTensorView *up,
                                      const float *input,
                                      float swiglu_limit);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_expert.h ==== */

/* ==== begin deepseek_v4_expert_store.h ==== */

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *model_dir;
    int layers;
    int experts_per_layer;
    uint64_t cache_bytes;
    /* Optional hot-pin policy (-1 / 0 => implementation default). */
    int pin_slots_per_layer;
    uint64_t repin_interval;
} ColiDeepSeekV4ExpertStoreOptions;

int coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_expert_store.h ==== */

/* ==== begin deepseek_v4_block.h ==== */

#include <stddef.h>

/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_attention.h */
/* amalgamated: deepseek_v4_layer.h */
#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size);
int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
/* ==== end deepseek_v4_block.h ==== */

/* ==== begin deepseek_v4_block_batch.h ==== */

/* amalgamated: deepseek_v4_attention.h */
/* amalgamated: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_layer.h */
#include "expert_store.h"

int coli_v4_block_window_batch_ref(
    float *outputs_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size);
/* ==== end deepseek_v4_block_batch.h ==== */

/* ==== begin deepseek_v4_resource_plan.h ==== */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t available_bytes;
    uint64_t user_limit_bytes;
    uint64_t maximum_layer_bytes;
    uint64_t runtime_other_bytes;
    uint64_t expert_record_bytes;
    int sparse_layers;
    int routed_topk;
    int experts_per_layer;
} ColiDeepSeekV4ResourceInputs;

typedef struct {
    uint64_t os_available_bytes;
    uint64_t planner_available_bytes;
    uint64_t system_reserve_bytes;
    uint64_t runtime_reserve_bytes;
    uint64_t minimum_expert_bytes;
    uint64_t expert_cache_bytes;
    uint64_t projected_bytes;
    int slots_per_layer;
} ColiDeepSeekV4ResourcePlan;

typedef struct {
    uint64_t available_bytes;
    uint64_t fixed_bytes;
    uint64_t dense_bytes;
    uint64_t minimum_expert_bytes;
} ColiDeepSeekV4ResidentTierInputs;

typedef struct {
    uint64_t dense_bytes;
    int dense_resident;
} ColiDeepSeekV4ResidentTierPlan;

uint64_t coli_v4_os_available_memory(void);
int coli_v4_resource_plan_compute(
    ColiDeepSeekV4ResourcePlan *plan,
    const ColiDeepSeekV4ResourceInputs *inputs,
    char *error, size_t error_size);
int coli_v4_resident_tier_plan(
    ColiDeepSeekV4ResidentTierPlan *plan,
    const ColiDeepSeekV4ResidentTierInputs *inputs,
    char *error, size_t error_size);
/* ==== end deepseek_v4_resource_plan.h ==== */

/* ==== begin deepseek_v4_head_cache.h ==== */

#include <stddef.h>
#include <stdint.h>

int coli_v4_head_cache_probe(const char *model_dir, uint64_t *bytes,
                             char *error, size_t error_size);
int coli_v4_head_cache_load(ColiV4Engine *engine, const char *model_dir,
                            char *error, size_t error_size);
uint64_t coli_v4_head_cache_bytes(const ColiV4Engine *engine);
const void *coli_v4_head_cache_data(const ColiV4Engine *engine,
                                    int shard, uint64_t offset, size_t length);
/* ==== end deepseek_v4_head_cache.h ==== */


/* Runtime options live on ColiV4Engine. */
typedef struct {
    const char *target_model_dir;
    uint64_t memory_limit_bytes;
    int context_tokens;
    int dense_resident;
    uint64_t target_expert_cache_bytes;
    int pin_slots_per_layer;
    uint64_t repin_interval;
    uint64_t dspark_reserve_bytes;
} ColiDeepSeekV4RuntimeOptions;

enum { COLI_V4_RESIDENT_MAX_LAYERS = 128 };

struct ColiV4Engine {
    ColiDeepSeekV4Config config;
    ColiDeepSeekV4RuntimeOptions runtime;
    ColiSafetensorsIndex *target_index;
    ColiExpertStore *experts;
    ColiV4EngineMemorySummary summary;
    struct {
        unsigned char *data;
        uint64_t bytes;
        uint64_t offset;
        int shard;
    } head_cache;
    struct {
        ColiDeepSeekV4LayerWeights layers[COLI_V4_RESIDENT_MAX_LAYERS];
        unsigned char ready[COLI_V4_RESIDENT_MAX_LAYERS];
        const ColiSafetensorsIndex *index;
        uint64_t total_bytes;
    } dense_resident;
    struct {
        uint16_t *markov_w1;
        uint16_t *markov_w2;
        uint64_t bytes;
        int rank;
        int block_size;
        int stage;
        int enabled;
    } dspark;
    char *owned_target_model_dir;
    int owns_experts;
    int owns_index;
    int active_sessions; /* sessions created against this engine */
};

/* Session ownership helpers shared by production session code and tests. */
void coli_v4_engine_attach_session(ColiV4Engine *engine);
void coli_v4_engine_detach_session(ColiV4Engine *engine);

#include "tok.h"
#include "kv_prefix.h"

struct ColiV4Session {
    ColiV4Engine *engine;
    ColiDeepSeekV4Config config;
    ColiDeepSeekV4WindowAttentionState **attention;
    float *state;
    float *next;
    float *hidden;
    int *prompt_ids;
    int *generated;
    int max_prompt_tokens;
    int max_new_tokens_cap;
    int prompt_count;
    int generated_count;
    Tok tokenizer;
    int tokenizer_ready;
    char *text;
    int text_length;
    /* Token ids this session's attention state already holds, prompt and
     * generated alike, in the shared format colibri.c/inkling.c/kimi_k3.c use.
     * A follow-up request whose prompt starts with exactly these ids continues
     * from that position instead of re-prefilling it. */
    kv_prefix fed;
    int prefix_reused;   /* reuse length of the request in flight, for stats */
    uint64_t spec_attempts;
    uint64_t spec_drafted;
    uint64_t spec_accepted;
    int spec_disabled;
};

/* RAM-tiered expert open used by coli_v4_engine_open (replaces ld --wrap). */
int coli_v4_expert_store_open_planned(
    ColiV4Engine *engine,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size);

/* Internal accessors — not part of the experimental public API. */
ColiSafetensorsIndex *coli_v4_engine_target_index(ColiV4Engine *engine);
ColiExpertStore *coli_v4_engine_expert_store(ColiV4Engine *engine);

/* Head-cache aware safetensors read (engine NULL => plain coli_st_read_at). */
int coli_st_read_at_engine(ColiV4Engine *engine,
                           const ColiSafetensorsIndex *index, int shard,
                           uint64_t offset, size_t length, void *destination);

#ifdef COLI_V4_TEST_HOOKS
/*
 * Fault-injection / counters for ownership tests only.
 * Compile ownership objects with -DCOLI_V4_TEST_HOOKS; production objects omit this.
 */
extern int coli_v4_test_fail_expert_store_open;
extern int coli_v4_test_skip_expert_store_open;
extern int coli_v4_test_closed_owned_index;

ColiV4Session *coli_v4_test_session_bare_create(ColiV4Engine *engine);
void coli_v4_test_session_bare_destroy(ColiV4Session *session);
#endif /* COLI_V4_TEST_HOOKS */

#endif /* COLIBRI_DEEPSEEK_V4_INTERNAL_H */
