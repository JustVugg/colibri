#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* COLI_CUDA_DLLEXPORT marks functions exported from coli_cuda.dll on Windows.
 * Define COLI_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
#define COLI_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define COLI_CUDA_DLLEXPORT
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Weight formats the generic per-element device decoder (weight_at,
 * backend_cuda.cu) can actually decode: f32, int8-row, int4 nibbles (fmt=2 and
 * the grouped fmt=4, same packing), and int2. Nothing else.
 *
 * WHY THIS IS A PREDICATE AND NOT A COMMENT. weight_at used to END in the int2
 * decode as an unguarded fall-through, so ANY other format handed to it -- a
 * format with a different element width, a different scale geometry, or no
 * device decoder at all -- was silently read as 2-bit values and returned
 * plausible-looking numbers. The CPU twins refuse the same input loudly:
 * qt_addrow and qt_matvec_rows (colibri.c) both exit(1) naming the function and
 * the fmt. Two backends given identical unsupported input, one refusing and one
 * fabricating, is the defect -- not the missing decoder.
 *
 * Defined here, in the host header, rather than inside the .cu: the host gates
 * that keep unsupported formats off the device (absorb_fmt_ok and friends) and
 * the device-side backstop must agree by construction rather than by two people
 * writing the same list twice, and a CPU test build can then unit-test the truth
 * table without a GPU or a CUDA toolchain (tests/test_cuda_fmt_guard.c) -- the
 * same arrangement colibri.c uses for metal_fused_fmt_ok.
 *
 * NOT a statement about which formats the CUDA BACKEND supports: quant_matmul
 * has its own explicit branches for fmt=6 (E8/IQ3), fmt=7 (MXFP4) and fmt=8
 * (fp8-e4m3) that never route through weight_at. This predicate is scoped to
 * weight_at's own dispatch, which is what the absorb and grouped-expert kernels
 * decode through. */
static inline int coli_cuda_weight_at_supported(int fmt) {
    return fmt == 0 || fmt == 1 || fmt == 2 || fmt == 3 || fmt == 4;
}

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Opaque model-agnostic GPU execution context for startup probes and later
 * device-resident pipelines. */
typedef struct ColiGpuContext ColiGpuContext;
typedef struct ColiGpuTensor ColiGpuTensor;
typedef struct ColiGpuArena ColiGpuArena;
typedef struct ColiGpuKdaState ColiGpuKdaState;
typedef struct ColiGpuMlaState ColiGpuMlaState;
typedef struct ColiGpuRouter ColiGpuRouter;
typedef struct ColiGpuExpertCache ColiGpuExpertCache;

typedef struct {
    const void *data;
    const float *scales; /* fmt 1/2: [rows]; fmt 4:
                            [rows, ceil(columns / group_size)] */
    int format;       /* 0=f32, 1=int8 row, 2=int4 row, 4=int4 grouped */
    int rows;
    int columns;
    int group_size;   /* required for format 4, zero otherwise */
} ColiGpuTensorDesc;

typedef struct {
    uint64_t device_allocations;
    uint64_t h2d_copies;
    uint64_t h2d_bytes;
    uint64_t d2h_copies;
    uint64_t d2h_bytes;
    uint64_t route_launches;
    uint64_t expert_upload_bytes;
    uint64_t expert_upload_events;
    uint64_t expert_cache_hits;
    uint64_t expert_cache_misses;
    uint64_t expert_cache_evictions;
    uint64_t expert_publications;
    uint64_t stale_generation_rejections;
    uint64_t wrong_expert_rejections;
    uint64_t unpublished_slot_rejections;
    uint64_t expert_handle_range_rejections;
    uint64_t generation_exhaustions;
    uint64_t expert_event_wait_failures;
    uint64_t expert_event_wait_calls;
    uint64_t unhealthy_cache_rejections;
    uint64_t moe_compute_launches;
    uint64_t selected_expert_count;
    uint64_t host_activation_h2d_copies;
    uint64_t host_activation_d2h_copies;
} ColiGpuTelemetry;

typedef enum {
    COLI_GPU_FAULT_NONE = 0,
    COLI_GPU_FAULT_BASE_ALLOCATION,
    COLI_GPU_FAULT_MODEL_ALLOCATION,
    COLI_GPU_FAULT_SESSION_ALLOCATION,
    COLI_GPU_FAULT_TENSOR_UPLOAD,
    COLI_GPU_FAULT_EXPERT_PUBLICATION,
    COLI_GPU_FAULT_KERNEL_LAUNCH,
    COLI_GPU_FAULT_DEVICE_STATUS,
    COLI_GPU_FAULT_STREAM_SYNC
} ColiGpuFaultPoint;

typedef struct {
    int hidden;
    int experts;
    int topk;
    int normalize_topk;
    float routed_scale;
} ColiGpuRouteConfig;

typedef struct {
    int experts;
    int slots;
    int hidden;
    int intermediate;
    int group_size;
    int max_rows;
    float swiglu_limit;
} ColiGpuExpertCacheConfig;

typedef struct {
    ColiGpuTensorDesc gate;
    ColiGpuTensorDesc up;
    ColiGpuTensorDesc down;
} ColiGpuExpertSource;

typedef struct {
    int expert_id;
    int slot;
    uint64_t generation;
} ColiGpuExpertHandle;

typedef struct {
    int expert_id;
    int slot;
    uint64_t generation;
    int published;
} ColiGpuExpertSlotInfo;

typedef enum {
    COLI_GPU_EXPERT_FAULT_NONE = 0,
    COLI_GPU_EXPERT_FAULT_ALLOCATION,
    COLI_GPU_EXPERT_FAULT_UPLOAD,
    COLI_GPU_EXPERT_FAULT_EVENT_RECORD,
    COLI_GPU_EXPERT_FAULT_EVENT_WAIT,
    COLI_GPU_EXPERT_FAULT_LAUNCH
} ColiGpuExpertFaultPoint;

typedef struct {
    const ColiGpuTensor *gate;
    const ColiGpuTensor *up;
    const ColiGpuTensor *down;
} ColiGpuMoeSharedWeights;

typedef struct {
    int heads;
    int head_dim;
    int kernel;
    int max_rows;
    int max_context;
    float recurrent_norm_eps;
    float output_norm_eps;
    float gate_lower_bound;
} ColiGpuKdaConfig;

typedef struct {
    const ColiGpuTensor *q_proj;
    const ColiGpuTensor *k_proj;
    const ColiGpuTensor *v_proj;
    const ColiGpuTensor *o_proj;
    const ColiGpuTensor *gate_a_proj;
    const ColiGpuTensor *gate_b_proj;
    const ColiGpuTensor *decay_a_proj;
    const ColiGpuTensor *decay_b_proj;
    const ColiGpuTensor *beta_proj;
    const ColiGpuTensor *conv;
    const ColiGpuTensor *dt_bias;
    const ColiGpuTensor *a_log;
    const ColiGpuTensor *o_norm;
} ColiGpuKdaWeights;

typedef struct {
    int hidden;
    int heads;
    int q_lora;
    int kv_lora;
    int qk_nope;
    int qk_rope; /* GLM-5.3 currently requires zero (NoPE). */
    int value_dim;
    int index_heads;
    int index_dim;
    int index_pool;
    int index_topk;
    int index_select_tail;
    int max_rows;
    int max_context;
    int page_tokens;
    float rms_norm_eps;
    float index_norm_eps;
} ColiGpuMlaConfig;

typedef struct {
    const ColiGpuTensor *q_a_proj;          /* [q_lora, hidden] */
    const ColiGpuTensor *q_a_norm;          /* [1, q_lora] */
    const ColiGpuTensor *q_b_proj;          /* [heads*qk_nope, q_lora] */
    const ColiGpuTensor *kv_a_proj;         /* [kv_lora, hidden] */
    const ColiGpuTensor *kv_a_norm;         /* [1, kv_lora] */
    const ColiGpuTensor *kv_b_key;          /* [heads*kv_lora, qk_nope] */
    const ColiGpuTensor *kv_b_value;        /* [heads*value_dim, kv_lora] */
    const ColiGpuTensor *o_proj;             /* [hidden, heads*value_dim] */
    const ColiGpuTensor *index_q_proj;       /* [index_heads*index_dim, q_lora] */
    const ColiGpuTensor *index_k_proj;       /* [index_dim, hidden] */
    const ColiGpuTensor *index_weight_proj;  /* [index_heads, hidden] */
    const ColiGpuTensor *index_key_norm;     /* [1, index_dim] */
    const ColiGpuTensor *index_key_bias;     /* [1, index_dim] */
    const ColiGpuTensor *index_pool_ape;     /* [index_pool, index_dim] */
    const ColiGpuTensor *index_pool_gate;    /* [index_dim, hidden] */
} ColiGpuMlaWeights;

typedef enum {
    COLI_GPU_MLA_FAULT_NONE = 0,
    COLI_GPU_MLA_FAULT_TABLE_ALLOC,
    COLI_GPU_MLA_FAULT_TABLE_COPY,
    COLI_GPU_MLA_FAULT_PAGE_ALLOC,
    COLI_GPU_MLA_FAULT_PAGE_PUBLISH,
    COLI_GPU_MLA_FAULT_TABLE_PUBLISH
} ColiGpuMlaFaultPoint;

typedef struct {
    int logical_length;
    int capacity;
    int page_count;
    int page_table_capacity;
    uint64_t payload_copy_bytes;
} ColiGpuMlaCacheInfo;

typedef enum {
    COLI_GPU_MLA_STATUS_OK = 0,
    COLI_GPU_MLA_STATUS_NONFINITE_INPUT,
    COLI_GPU_MLA_STATUS_NONFINITE_CACHE,
    COLI_GPU_MLA_STATUS_NONFINITE_RESULT
} ColiGpuMlaStatus;

typedef enum {
    COLI_GPU_MLA_CACHE_LATENT = 0,
    COLI_GPU_MLA_CACHE_INDEX_KEY,
    COLI_GPU_MLA_CACHE_INDEX_GATE
} ColiGpuMlaCacheKind;

typedef struct {
    uint64_t kernel_launches;
    uint64_t total_grid_blocks;
    int max_grid_blocks;
    int max_block_threads;
} ColiGpuMlaLaunchInfo;

enum {
    COLI_GPU_CAP_STREAM_ORDERED = 1ull << 0,
    COLI_GPU_CAP_INT4_GS64      = 1ull << 1,
    COLI_GPU_CAP_PIPELINE       = 1ull << 2
};

COLI_CUDA_DLLEXPORT int coli_gpu_context_create(ColiGpuContext **out, int device);
COLI_CUDA_DLLEXPORT int coli_gpu_context_probe(ColiGpuContext *ctx, uint64_t required_caps);
COLI_CUDA_DLLEXPORT int coli_gpu_context_healthy(const ColiGpuContext *ctx);
COLI_CUDA_DLLEXPORT int coli_gpu_context_memory_info(
    ColiGpuContext *ctx, size_t *free_bytes, size_t *total_bytes);
COLI_CUDA_DLLEXPORT int coli_gpu_context_sync(ColiGpuContext *ctx);
COLI_CUDA_DLLEXPORT void coli_gpu_context_mark_unhealthy(ColiGpuContext *ctx);
COLI_CUDA_DLLEXPORT int coli_gpu_context_inject_fault(
    ColiGpuContext *ctx, ColiGpuFaultPoint point, int occurrence);
COLI_CUDA_DLLEXPORT int coli_gpu_context_consume_fault(
    ColiGpuContext *ctx, ColiGpuFaultPoint point);
COLI_CUDA_DLLEXPORT void coli_gpu_context_destroy(ColiGpuContext *ctx);
COLI_CUDA_DLLEXPORT void coli_gpu_context_telemetry(const ColiGpuContext *ctx,
                                                    ColiGpuTelemetry *out);

/* Context-owned-stream resident storage. Host transfers synchronize only this
 * explicit non-blocking stream and are therefore API boundaries. */
COLI_CUDA_DLLEXPORT int coli_gpu_tensor_create(ColiGpuTensor **out,
                                               ColiGpuContext *ctx,
                                               const ColiGpuTensorDesc *desc);
COLI_CUDA_DLLEXPORT void coli_gpu_tensor_destroy(ColiGpuTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_gpu_arena_create(ColiGpuArena **out,
                                              ColiGpuContext *ctx,
                                              size_t capacity);
COLI_CUDA_DLLEXPORT void coli_gpu_arena_destroy(ColiGpuArena *arena);
COLI_CUDA_DLLEXPORT size_t coli_gpu_arena_capacity(const ColiGpuArena *arena);
COLI_CUDA_DLLEXPORT int coli_gpu_arena_upload(ColiGpuArena *arena, size_t offset,
                                              const void *src, size_t bytes);
COLI_CUDA_DLLEXPORT int coli_gpu_arena_download(ColiGpuArena *arena, size_t offset,
                                                void *dst, size_t bytes);
COLI_CUDA_DLLEXPORT int coli_gpu_arena_upload_activation(
    ColiGpuArena *arena, size_t offset, const void *src, size_t bytes);
COLI_CUDA_DLLEXPORT int coli_gpu_arena_download_activation(
    ColiGpuArena *arena, size_t offset, void *dst, size_t bytes);

COLI_CUDA_DLLEXPORT int coli_gpu_embedding(ColiGpuArena *arena,
                                           size_t streams_offset,
                                           size_t token_ids_offset,
                                           const ColiGpuTensor *embedding,
                                           int rows, int streams, int hidden);
COLI_CUDA_DLLEXPORT int coli_gpu_rmsnorm(ColiGpuArena *arena, size_t output_offset,
                                         size_t input_offset,
                                         const ColiGpuTensor *weight,
                                         int rows, int hidden, float eps);
COLI_CUDA_DLLEXPORT int coli_gpu_layernorm(ColiGpuArena *arena, size_t output_offset,
                                           size_t input_offset,
                                           const ColiGpuTensor *weight,
                                           const ColiGpuTensor *bias,
                                           int rows, int hidden, float eps);
COLI_CUDA_DLLEXPORT int coli_gpu_mhc_pre_norm(
    ColiGpuArena *arena, size_t collapsed_offset, size_t normed_offset,
    size_t post_offset, size_t comb_offset, size_t residual_offset,
    const ColiGpuTensor *fn, const ColiGpuTensor *scale,
    const ColiGpuTensor *base, const ColiGpuTensor *norm_weight,
    int rows, int streams, int hidden, float norm_eps, float hc_eps);
COLI_CUDA_DLLEXPORT int coli_gpu_mhc_post(
    ColiGpuArena *arena, size_t output_offset, size_t branch_offset,
    size_t residual_offset, size_t post_offset, size_t comb_offset,
    int rows, int streams, int hidden);
COLI_CUDA_DLLEXPORT int coli_gpu_mhc_site(
    ColiGpuArena *arena, size_t output_offset, size_t collapsed_offset,
    size_t normed_offset, size_t post_offset, size_t comb_offset,
    size_t residual_offset, size_t branch_offset,
    const ColiGpuTensor *fn, const ColiGpuTensor *scale,
    const ColiGpuTensor *base, const ColiGpuTensor *norm_weight,
    int rows, int streams, int hidden, float norm_eps, float hc_eps);
COLI_CUDA_DLLEXPORT int coli_gpu_collapse_streams(
    ColiGpuArena *arena, size_t output_offset, size_t streams_offset,
    int rows, int streams, int hidden);
COLI_CUDA_DLLEXPORT int coli_gpu_projection(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    const ColiGpuTensor *weight, int rows, int input_size, int output_size);
COLI_CUDA_DLLEXPORT int coli_gpu_dense_mlp(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t gate_offset, size_t up_offset,
    const ColiGpuTensor *gate, const ColiGpuTensor *up,
    const ColiGpuTensor *down, int rows, int hidden, int intermediate,
    float swiglu_limit);

/* GLM-5.3 routing remains device-resident. Download is a metadata-only
 * loader/test boundary; no activation is copied by route or MoE compute. */
COLI_CUDA_DLLEXPORT int coli_gpu_router_create(
    ColiGpuRouter **out, ColiGpuContext *ctx,
    const ColiGpuRouteConfig *config, int max_rows);
COLI_CUDA_DLLEXPORT void coli_gpu_router_destroy(ColiGpuRouter *router);
COLI_CUDA_DLLEXPORT int coli_gpu_router_run(
    ColiGpuRouter *router, ColiGpuArena *arena, size_t input_offset,
    const ColiGpuTensor *weight, const ColiGpuTensor *correction_bias,
    int rows);
COLI_CUDA_DLLEXPORT int coli_gpu_router_download(
    ColiGpuRouter *router, int *selected_ids, float *routing_weights,
    size_t selected_count, int rows);

/* The caller owns expert bytes and victim selection. A successful upload
 * records a stream event and returns a generation-bound handle; compute waits
 * for that event. Failed uploads leave the previous slot mapping publishable. */
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_create(
    ColiGpuExpertCache **out, ColiGpuContext *ctx,
    const ColiGpuExpertCacheConfig *config);
COLI_CUDA_DLLEXPORT void coli_gpu_expert_cache_destroy(
    ColiGpuExpertCache *cache);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_upload(
    ColiGpuExpertCache *cache, int expert_id, int slot,
    const ColiGpuExpertSource *source, ColiGpuExpertHandle *out);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_lookup(
    ColiGpuExpertCache *cache, int expert_id, ColiGpuExpertHandle *out);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_validate(
    ColiGpuExpertCache *cache, const ColiGpuExpertHandle *handle);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_slot_info(
    const ColiGpuExpertCache *cache, int slot, ColiGpuExpertSlotInfo *out);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_inject_fault(
    ColiGpuExpertCache *cache, ColiGpuExpertFaultPoint point);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_inject_fault_at(
    ColiGpuExpertCache *cache, ColiGpuExpertFaultPoint point, int occurrence);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_healthy(
    const ColiGpuExpertCache *cache);
/* Deterministic rollover-boundary test hook; requires a published slot. */
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_test_set_generation(
    ColiGpuExpertCache *cache, int slot, uint64_t generation);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_test_delay_upload(
    ColiGpuExpertCache *cache, unsigned milliseconds);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_test_hold_snapshot(
    ColiGpuExpertCache *cache, int hold);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_cache_test_snapshot_entered(
    const ColiGpuExpertCache *cache);
COLI_CUDA_DLLEXPORT size_t coli_gpu_moe_scratch_bytes(
    const ColiGpuExpertCacheConfig *config);
COLI_CUDA_DLLEXPORT int coli_gpu_expert_primitive(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t scratch_offset, ColiGpuExpertCache *cache,
    const ColiGpuMoeSharedWeights *weights, int rows);
/* shared or router may be NULL for routed-only/shared-only execution.
 * Routed handles are row-major [rows, topk] and must match the router result. */
COLI_CUDA_DLLEXPORT int coli_gpu_moe_site(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t scratch_offset, ColiGpuRouter *router,
    ColiGpuExpertCache *cache, const ColiGpuMoeSharedWeights *shared,
    const ColiGpuExpertHandle *handles, size_t handle_count, int rows);

/* Stateful KDA storage is bound to caller-reserved persistent arena ranges.
 * Creation allocates only the opaque host handle; reset and hot-path calls
 * enqueue work on ctx's non-blocking stream without device allocation or
 * host transfer. */
COLI_CUDA_DLLEXPORT size_t coli_gpu_kda_state_bytes(
    const ColiGpuKdaConfig *config);
COLI_CUDA_DLLEXPORT size_t coli_gpu_kda_window_bytes(
    const ColiGpuKdaConfig *config);
COLI_CUDA_DLLEXPORT size_t coli_gpu_kda_scratch_bytes(
    const ColiGpuKdaConfig *config, int rows, int hidden);
COLI_CUDA_DLLEXPORT int coli_gpu_kda_state_create(
    ColiGpuKdaState **out, ColiGpuContext *ctx, ColiGpuArena *arena,
    size_t state_offset, size_t window_offset,
    const ColiGpuKdaConfig *config);
COLI_CUDA_DLLEXPORT void coli_gpu_kda_state_destroy(ColiGpuKdaState *state);
COLI_CUDA_DLLEXPORT int coli_gpu_kda_state_reset(ColiGpuKdaState *state);
COLI_CUDA_DLLEXPORT int coli_gpu_kda_state_download(
    ColiGpuKdaState *state, float *matrix, size_t matrix_floats,
    float *window, size_t window_floats);
/* qkv is [3, rows, heads * head_dim], decay is
 * [rows, heads * head_dim], beta is [rows, heads], and output is
 * [rows, heads * head_dim]. The call advances state from start_position.
 * All four arena ranges and the state's matrix/window ranges must be mutually
 * non-overlapping; aliases are rejected before any launch. */
COLI_CUDA_DLLEXPORT int coli_gpu_kda_recurrent(
    ColiGpuArena *arena, size_t output_offset, size_t qkv_offset,
    size_t decay_offset, size_t beta_offset, ColiGpuKdaState *state,
    const ColiGpuTensor *conv, int rows, int start_position);
/* input, output, scratch, and the state's matrix/window ranges must be
 * mutually non-overlapping. Any overlap is rejected before state mutation. */
COLI_CUDA_DLLEXPORT int coli_gpu_kda_site(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t scratch_offset, ColiGpuKdaState *state,
    const ColiGpuKdaWeights *weights, int rows, int start_position,
    int hidden);

/* Device-resident GLM-5.3 MLA/DSA state. Caches are contiguous geometric
 * page-capacity arrays with layouts [position, kv_lora] and
 * [position, index_dim]. Growth preserves the prefix and only synchronizes the
 * owning context stream before retiring old storage. */
COLI_CUDA_DLLEXPORT size_t coli_gpu_mla_scratch_bytes(
    const ColiGpuMlaConfig *config);
COLI_CUDA_DLLEXPORT size_t coli_gpu_mla_selected_bytes(
    const ColiGpuMlaConfig *config, int rows);
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_create(
    ColiGpuMlaState **out, ColiGpuContext *ctx,
    const ColiGpuMlaConfig *config);
/* Persistent layout: each independently allocated page is
 * [page_tokens, kv_lora] latent, then [page_tokens, index_dim] index keys,
 * then [page_tokens, index_dim] index gates. A stream-published device page
 * table addresses pages; geometric growth replaces only this metadata table
 * and allocates new pages, never copies an existing cache payload. */
COLI_CUDA_DLLEXPORT void coli_gpu_mla_state_destroy(ColiGpuMlaState *state);
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_reset(ColiGpuMlaState *state);
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_length(
    const ColiGpuMlaState *state);
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_capacity(
    const ColiGpuMlaState *state);
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_reserve(
    ColiGpuMlaState *state, int required);
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_cache_info(
    const ColiGpuMlaState *state, ColiGpuMlaCacheInfo *out);
/* Deterministic test hook. occurrence is zero-based among boundaries of the
 * selected type in the next growth attempt. The injected failure is one-shot. */
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_inject_growth_fault(
    ColiGpuMlaState *state, ColiGpuMlaFaultPoint point, int occurrence);
COLI_CUDA_DLLEXPORT ColiGpuMlaStatus coli_gpu_mla_state_status(
    const ColiGpuMlaState *state);
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_launch_info(
    const ColiGpuMlaState *state, ColiGpuMlaLaunchInfo *out);
/* Test-only corruption hook used to prove persistent-cache validation. */
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_test_corrupt_cache(
    ColiGpuMlaState *state, ColiGpuMlaCacheKind kind,
    int position, int channel, float value);
COLI_CUDA_DLLEXPORT int coli_gpu_mla_state_download(
    ColiGpuMlaState *state,
    float *latent, size_t latent_floats,
    float *index_keys, size_t index_key_floats,
    float *index_gates, size_t index_gate_floats);
/* input/output are [rows, hidden], selected is
 * [rows, index_topk + (tail ? index_pool - 1 : 0)]. start_position must equal
 * the state's logical length. All arena ranges must be mutually disjoint. */
COLI_CUDA_DLLEXPORT int coli_gpu_mla_site(
    ColiGpuArena *arena, size_t output_offset, size_t input_offset,
    size_t scratch_offset, size_t selected_offset,
    ColiGpuMlaState *state, const ColiGpuMlaWeights *weights,
    int rows, int start_position);

/* Devices are CUDA ordinals, not positions in the input list. */
COLI_CUDA_DLLEXPORT int coli_cuda_init(const int *devices, int count);
COLI_CUDA_DLLEXPORT void coli_cuda_shutdown(void);
/* Number of CUDA devices visible to this process, before a device list is
 * selected. Returns 0 when the runtime cannot discover any device. */
COLI_CUDA_DLLEXPORT int coli_cuda_available_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_at(int index);
COLI_CUDA_DLLEXPORT int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_device_integrated(int device);
/* device < 0 returns aggregate statistics for all configured devices. */
COLI_CUDA_DLLEXPORT void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);
/* Per-device form of coli_cuda_group_stats; unknown devices return zeros. */
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats_device(
    int device, uint64_t *calls, uint64_t *experts, uint64_t *rows,
    double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* Publish the E8 codebook (quant.h's e8_grid, 256x4 bytes) to every configured
 * device. Must be called after coli_cuda_init and before any fmt=6 upload; the
 * backend keeps no copy of the table so it cannot drift from the CPU decoder. */
COLI_CUDA_DLLEXPORT int coli_cuda_e8_set_grid(const void *grid);

/* Publish the fmt=8 e4m3 decode table (quant.h's E4M3_LUT, 256 f32) the same
 * way. Must be called after coli_cuda_init; fmt=8 uploads are refused until it
 * succeeds, because kernels would decode against a zero-initialized table. */
COLI_CUDA_DLLEXPORT int coli_cuda_fp8_set_lut(const float *lut);

/* Upload without executing, so capacity failures happen during model startup. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
        const void *weights, const float *scales,
        int fmt, int I, int O, int device, int gs);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
#ifdef COLI_ANS
/* Experimental Linux-only GPU-resident entropy tier. The archive remains in
 * VRAM and is decoded into per-device scratch immediately before a grouped
 * expert launch. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_compressed(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
#endif

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2, 4=grouped int4.
 * gs is the group size for fmt=4 (0 for all other formats).
 * The first successful call uploads W and its scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
/* y[S,O] = x[S,I] @ dequant_mxfp4(W[O,I])^T, fmt=7 (OCP microscaling FP4).
 * q4 is [O, ceil(I/2)] e2m1 nibbles, e8s is [O, ceil(I/32)] ue8m0 exponents --
 * BYTES, not floats, which is why this is not folded into coli_cuda_matmul.
 * Stateless: weights are uploaded per call, matching the streaming expert tier
 * Kimi K3 uses. Returns 1 on success, 0 (y untouched) to fall back to CPU. */
COLI_CUDA_DLLEXPORT int coli_cuda_matmul_mxfp4(float *y, const float *x,
                                               const unsigned char *q4,
                                               const unsigned char *e8s,
                                               int S, int I, int O);

COLI_CUDA_DLLEXPORT int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Prefill-oriented shared expert path.  INT4 weights stay packed in global
 * memory, activations are converted to FP16 per tile, and Tensor Cores
 * accumulate into FP32.  Unlike COLI_CUDA_TC_INT4 this does not quantize the
 * activation to INT4. */
COLI_CUDA_DLLEXPORT int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up,
                               ColiCudaTensor *down, float *y,
                               const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
/* Async issue/take split of the group call below (Inc.4): issue launches on the
 * device stream and returns; take syncs and returns the pinned result rows (valid
 * until the next issue on that device). Small totals only (<=8 rows); one
 * outstanding issue per device. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                               ColiCudaTensor *const *ups,
                               ColiCudaTensor *const *downs,
                               const int *rows, int count, const float *x);
COLI_CUDA_DLLEXPORT const float *coli_cuda_expert_group_take(int device);

COLI_CUDA_DLLEXPORT int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);
/* Same operation, but force the small-batch grouped kernel family when
 * pin_small_batch is nonzero.  Speculative verification uses this to keep
 * CUDA on the same numeric family as S=1 regardless of accepted draft depth. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_pinned(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x, int pin_small_batch);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

/* Causal MLA absorption for S contiguous rows from one sequence.  The KV
 * arrays contain T rows ending at the final query; query s attends T-S+s+1
 * rows.  One transfer and one launch replace S host round-trips. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q,
                                     const float *latent,const float *rope,int S,
                                     int H,int Q,int R,int V,int K,int T,
                                     float attention_scale);

/* Same attention batch followed immediately by resident o_proj on the same
 * device.  Only the final [S,D] tensor crosses back to the host. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
                                      float *out,const float *q,const float *latent,
                                      const float *rope,int S,int H,int Q,int R,
                                      int V,int K,int T,float attention_scale);

COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);

COLI_CUDA_DLLEXPORT void coli_cuda_tensor_free(ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_alloc_footprint(size_t bytes);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_vram(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

/* Replace a resident tensor's contents without reallocating its device slot. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                            const void *weights, const float *scales);

/* ---- resident-pipeline primitives (Inc.0): device-pointer entry points ---- */
COLI_CUDA_DLLEXPORT float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes);
COLI_CUDA_DLLEXPORT void *coli_cuda_pipe_alloc(int device,size_t bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_pipe_free(int device,void *p);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                           const float *w_dev,int S,int D,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows,
                        int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                            const int *rows_dev,int nrows,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                             const float *w_dev,int S,int D,float eps,
                             int xstride,int ystride);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                             int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_take(int home_device,const int *devices,
        int n_issued,float *slots_dev,float *acc_dev,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                          int spitch,int width,int height);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                             const float *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_sync(int device);

#ifdef __cplusplus
}
#endif

#endif
