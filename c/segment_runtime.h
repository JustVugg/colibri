#ifndef COLIBRI_SEGMENT_RUNTIME_H
#define COLIBRI_SEGMENT_RUNTIME_H

/*
 * Public, engine-neutral layer-segment runtime.
 *
 * Colibri owns model math, weights, accelerators and sequence state.  A
 * distributed caller owns transport, placement, leases and network session
 * identifiers.  This API is the boundary between the two: adapters keep all
 * engine-specific objects opaque while callers can open a layer range, create
 * isolated sessions, run activations through it and stream snapshots.
 *
 * Version 1 is additive and unused by the existing CLI/serve paths.  A
 * consumer registers adapters explicitly during process initialization and
 * before the first lookup, after which the registry is read-only and safe for
 * concurrent lookup.  No compiler-specific constructor mechanism is needed.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_SEGMENT_ABI_VERSION 1u
#define COLI_SEGMENT_ENGINE_ID_CAP 64u
#define COLI_SEGMENT_STATE_SCHEMA_CAP 128u
#define COLI_SEGMENT_NUMERIC_CLASS_CAP 96u

typedef struct ColiSegmentEngine ColiSegmentEngine;
typedef struct ColiSegmentSession ColiSegmentSession;

typedef enum {
    COLI_SEGMENT_DTYPE_INVALID = 0,
    COLI_SEGMENT_DTYPE_F32 = 1,
    COLI_SEGMENT_DTYPE_F16 = 2,
    COLI_SEGMENT_DTYPE_BF16 = 3,
} ColiSegmentDType;

enum {
    COLI_SEGMENT_CAP_TOKEN_IDS = UINT64_C(1) << 0,
    COLI_SEGMENT_CAP_SNAPSHOT = UINT64_C(1) << 1,
    COLI_SEGMENT_CAP_RANGE_NATIVE = UINT64_C(1) << 2,
    COLI_SEGMENT_CAP_MULTI_SESSION = UINT64_C(1) << 3,
    COLI_SEGMENT_CAP_CPU = UINT64_C(1) << 8,
    COLI_SEGMENT_CAP_CUDA = UINT64_C(1) << 9,
    COLI_SEGMENT_CAP_HIP = UINT64_C(1) << 10,
    COLI_SEGMENT_CAP_METAL = UINT64_C(1) << 11,
    COLI_SEGMENT_CAP_VULKAN = UINT64_C(1) << 12,
};

#define COLI_SEGMENT_CAP_BACKEND_MASK                                      \
    (COLI_SEGMENT_CAP_CPU | COLI_SEGMENT_CAP_CUDA | COLI_SEGMENT_CAP_HIP | \
     COLI_SEGMENT_CAP_METAL | COLI_SEGMENT_CAP_VULKAN)

/* Filled by the adapter during engine_open, after it has inspected the model.
 * Fixed-size identity fields make the record safe to copy across a local ABI
 * boundary. state_width is the activation width on the segment wire; it need
 * not equal the model hidden size (DeepSeek V4 mHC is one example). */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    char engine_id[COLI_SEGMENT_ENGINE_ID_CAP];
    char state_schema[COLI_SEGMENT_STATE_SCHEMA_CAP];
    char numeric_class[COLI_SEGMENT_NUMERIC_CLASS_CAP];
    uint32_t state_dtype;
    uint32_t state_width;
    uint32_t max_batch_rows;
    uint32_t max_context_tokens;
    uint32_t num_layers;
    uint32_t reserved_u32[7];
    uint64_t reserved_u64[4];
} ColiSegmentCapabilities;

/* layer_begin is inclusive and layer_end is exclusive.  backend_mask == 0
 * lets the adapter select its normal local policy.  memory_limit_bytes == 0
 * uses the adapter's ordinary automatic budget. */
typedef struct {
    uint32_t struct_size;
    const char *model_dir;
    uint32_t layer_begin;
    uint32_t layer_end;
    uint32_t context_tokens;
    uint32_t reserved_u32;
    uint64_t memory_limit_bytes;
    uint64_t backend_mask;
    const void *resource_plan;
    size_t resource_plan_size;
} ColiSegmentEngineOptions;

typedef struct {
    uint32_t struct_size;
    uint32_t context_tokens;
    uint64_t memory_limit_bytes;
    uint64_t reserved_u64[4];
} ColiSegmentSessionOptions;

typedef int (*ColiSegmentCancelFn)(void *user_data);

/* input and output each hold rows * state_width values of the advertised
 * dtype.  token_ids is either NULL/0 or exactly rows entries.  Engines whose
 * capabilities include COLI_SEGMENT_CAP_TOKEN_IDS require the latter. */
typedef struct {
    uint32_t struct_size;
    uint32_t rows;
    uint64_t position;
    const int32_t *token_ids;
    size_t token_count;
    const void *input;
    size_t input_bytes;
    void *output;
    size_t output_bytes;
    ColiSegmentCancelFn should_cancel;
    void *cancel_user_data;
    uint64_t reserved_u64[4];
} ColiSegmentRunRequest;

/* Snapshot formats are private to the adapter and named by state_schema.
 * Streaming callbacks avoid a second full-state allocation in Colibri or in
 * a network daemon.  A callback returns zero on success, non-zero to abort. */
typedef int (*ColiSegmentWriteFn)(void *user_data, const void *data, size_t size);
typedef int (*ColiSegmentReadFn)(void *user_data, void *data, size_t size);

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *engine_id;

    int (*engine_open)(void **engine_impl,
                       ColiSegmentCapabilities *capabilities,
                       const ColiSegmentEngineOptions *options,
                       char *error, size_t error_size);
    void (*engine_destroy)(void *engine_impl);
    int (*session_create)(void *engine_impl, void **session_impl,
                          const ColiSegmentSessionOptions *options,
                          char *error, size_t error_size);
    void (*session_destroy)(void *session_impl);
    int (*session_run)(void *session_impl,
                       const ColiSegmentRunRequest *request,
                       char *error, size_t error_size);
    int (*session_snapshot)(void *session_impl, ColiSegmentWriteFn write_fn,
                            void *write_user_data,
                            char *error, size_t error_size);
    int (*session_restore)(void *session_impl, ColiSegmentReadFn read_fn,
                           void *read_user_data,
                           char *error, size_t error_size);

    void (*reserved_fn[8])(void);
} ColiSegmentAdapter;

/* Registering is an initialization-time operation. Duplicate engine IDs and
 * malformed/currently incompatible adapters are rejected. Complete all
 * registration before lookups or execution begin. */
int coli_segment_adapter_register(const ColiSegmentAdapter *adapter);
const ColiSegmentAdapter *coli_segment_adapter_lookup(const char *engine_id);
int coli_segment_adapter_count(void);

int coli_segment_engine_open(const char *engine_id,
                             const ColiSegmentEngineOptions *options,
                             ColiSegmentEngine **engine,
                             char *error, size_t error_size);
/* The caller sets struct_size to its allocation size. Bytes beyond the ABI
 * version known by this runtime are zeroed for forward-compatible callers. */
int coli_segment_engine_capabilities(const ColiSegmentEngine *engine,
                                     ColiSegmentCapabilities *capabilities,
                                     char *error, size_t error_size);
/* Refuses to close an engine while sessions are alive. */
int coli_segment_engine_close(ColiSegmentEngine *engine,
                              char *error, size_t error_size);

int coli_segment_session_create(ColiSegmentEngine *engine,
                                const ColiSegmentSessionOptions *options,
                                ColiSegmentSession **session,
                                char *error, size_t error_size);
void coli_segment_session_destroy(ColiSegmentSession *session);
int coli_segment_run(ColiSegmentSession *session,
                     const ColiSegmentRunRequest *request,
                     char *error, size_t error_size);
int coli_segment_snapshot(ColiSegmentSession *session,
                          ColiSegmentWriteFn write_fn, void *write_user_data,
                          char *error, size_t error_size);
int coli_segment_restore(ColiSegmentSession *session,
                         ColiSegmentReadFn read_fn, void *read_user_data,
                         char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_SEGMENT_RUNTIME_H */
