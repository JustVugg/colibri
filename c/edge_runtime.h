#ifndef COLIBRI_EDGE_RUNTIME_H
#define COLIBRI_EDGE_RUNTIME_H

/*
 * Public, engine-neutral model-edge runtime.
 *
 * A distributed inference client keeps only the input/output boundary of a
 * model locally: tokenizer, token embedding, final normalization and output
 * head. Layer ranges remain owned by the Segment runtime. Keeping this ABI
 * separate prevents a network client from reaching into an engine's private
 * Model structure or loading the complete transformer just to start a chat.
 *
 * Version 2 additionally exposes final-head logits. Sampling policy and RNG
 * remain with the serving caller, while every model-specific final transform
 * and output-head implementation stays owned by Colibri. Registration is
 * explicit; ordinary Colibri executables do not use this API.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_EDGE_ABI_VERSION 2u
#define COLI_EDGE_ENGINE_ID_CAP 64u
#define COLI_EDGE_STATE_SCHEMA_CAP 128u
#define COLI_EDGE_NUMERIC_CLASS_CAP 96u
#define COLI_EDGE_TOKENIZER_CLASS_CAP 64u

typedef struct ColiEdgeEngine ColiEdgeEngine;

typedef enum {
    COLI_EDGE_DTYPE_INVALID = 0,
    COLI_EDGE_DTYPE_F32 = 1,
    COLI_EDGE_DTYPE_F16 = 2,
    COLI_EDGE_DTYPE_BF16 = 3,
} ColiEdgeDType;

enum {
    COLI_EDGE_CAP_TOKENIZE = UINT64_C(1) << 0,
    COLI_EDGE_CAP_DETOKENIZE = UINT64_C(1) << 1,
    COLI_EDGE_CAP_GREEDY = UINT64_C(1) << 2,
    COLI_EDGE_CAP_LOGITS = UINT64_C(1) << 3,
    COLI_EDGE_CAP_CPU = UINT64_C(1) << 8,
    COLI_EDGE_CAP_CUDA = UINT64_C(1) << 9,
    COLI_EDGE_CAP_HIP = UINT64_C(1) << 10,
    COLI_EDGE_CAP_METAL = UINT64_C(1) << 11,
    COLI_EDGE_CAP_VULKAN = UINT64_C(1) << 12,
};

#define COLI_EDGE_CAP_BACKEND_MASK                                      \
    (COLI_EDGE_CAP_CPU | COLI_EDGE_CAP_CUDA | COLI_EDGE_CAP_HIP |       \
     COLI_EDGE_CAP_METAL | COLI_EDGE_CAP_VULKAN)

/* state_schema, numeric_class, dtype and width must match the first/last
 * Segment peers selected by the distributed caller. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    char engine_id[COLI_EDGE_ENGINE_ID_CAP];
    char state_schema[COLI_EDGE_STATE_SCHEMA_CAP];
    char numeric_class[COLI_EDGE_NUMERIC_CLASS_CAP];
    char tokenizer_class[COLI_EDGE_TOKENIZER_CLASS_CAP];
    uint32_t state_dtype;
    uint32_t state_width;
    uint32_t vocab_size;
    uint32_t max_batch_rows;
    uint32_t max_context_tokens;
    uint32_t num_layers;
    int32_t bos_token_id;
    int32_t eos_token_id;
    uint32_t reserved_u32[4];
    uint64_t resident_bytes;
    uint64_t reserved_u64[3];
} ColiEdgeCapabilities;

typedef struct {
    uint32_t struct_size;
    const char *model_dir;
    uint64_t memory_limit_bytes;
    uint64_t backend_mask;
    const void *resource_plan;
    size_t resource_plan_size;
    uint64_t reserved_u64[3];
} ColiEdgeEngineOptions;

typedef int (*ColiEdgeCancelFn)(void *user_data);

typedef struct {
    uint32_t struct_size;
    uint32_t rows;
    const int32_t *token_ids;
    size_t token_count;
    void *output;
    size_t output_bytes;
    ColiEdgeCancelFn should_cancel;
    void *cancel_user_data;
    uint64_t reserved_u64[3];
} ColiEdgeEmbedRequest;

/* Greedy selection applies the engine's exact final-state transform and LM
 * head to every input row. Scores may be NULL; token_ids must hold rows IDs. */
typedef struct {
    uint32_t struct_size;
    uint32_t rows;
    const void *input;
    size_t input_bytes;
    int32_t *token_ids;
    size_t token_capacity;
    float *scores;
    size_t score_capacity;
    ColiEdgeCancelFn should_cancel;
    void *cancel_user_data;
    uint64_t reserved_u64[3];
} ColiEdgeSelectRequest;

/* Applies the exact model-specific final transform and LM head, returning
 * rows*vocab_size float logits in row-major order. This deliberately does
 * not prescribe temperature, top-p or an RNG: those are serving policy, not
 * model math. */
typedef struct {
    uint32_t struct_size;
    uint32_t rows;
    const void *input;
    size_t input_bytes;
    float *logits;
    size_t logits_capacity;
    ColiEdgeCancelFn should_cancel;
    void *cancel_user_data;
    uint64_t reserved_u64[3];
} ColiEdgeLogitsRequest;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *engine_id;

    int (*engine_open)(void **engine_impl,
                       ColiEdgeCapabilities *capabilities,
                       const ColiEdgeEngineOptions *options,
                       char *error, size_t error_size);
    void (*engine_destroy)(void *engine_impl);
    int (*tokenize)(void *engine_impl, const char *text, size_t text_bytes,
                    int32_t *token_ids, size_t token_capacity,
                    size_t *token_count, char *error, size_t error_size);
    int (*detokenize)(void *engine_impl, const int32_t *token_ids,
                      size_t token_count, char *text, size_t text_capacity,
                      size_t *text_bytes, char *error, size_t error_size);
    int (*embed)(void *engine_impl, const ColiEdgeEmbedRequest *request,
                 char *error, size_t error_size);
    int (*select)(void *engine_impl, const ColiEdgeSelectRequest *request,
                  char *error, size_t error_size);
    int (*logits)(void *engine_impl, const ColiEdgeLogitsRequest *request,
                  char *error, size_t error_size);

    void (*reserved_fn[7])(void);
} ColiEdgeAdapter;

int coli_edge_adapter_register(const ColiEdgeAdapter *adapter);
const ColiEdgeAdapter *coli_edge_adapter_lookup(const char *engine_id);
int coli_edge_adapter_count(void);

int coli_edge_engine_open(const char *engine_id,
                          const ColiEdgeEngineOptions *options,
                          ColiEdgeEngine **engine,
                          char *error, size_t error_size);
int coli_edge_engine_capabilities(const ColiEdgeEngine *engine,
                                  ColiEdgeCapabilities *capabilities,
                                  char *error, size_t error_size);
void coli_edge_engine_close(ColiEdgeEngine *engine);

/* Tokenizer calls support a sizing pass: output may be NULL only when its
 * capacity is zero. The required count/size is always returned on success. */
int coli_edge_tokenize(ColiEdgeEngine *engine,
                       const char *text, size_t text_bytes,
                       int32_t *token_ids, size_t token_capacity,
                       size_t *token_count,
                       char *error, size_t error_size);
int coli_edge_detokenize(ColiEdgeEngine *engine,
                         const int32_t *token_ids, size_t token_count,
                         char *text, size_t text_capacity, size_t *text_bytes,
                         char *error, size_t error_size);
int coli_edge_embed(ColiEdgeEngine *engine,
                    const ColiEdgeEmbedRequest *request,
                    char *error, size_t error_size);
int coli_edge_select(ColiEdgeEngine *engine,
                     const ColiEdgeSelectRequest *request,
                     char *error, size_t error_size);
int coli_edge_logits(ColiEdgeEngine *engine,
                     const ColiEdgeLogitsRequest *request,
                     char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EDGE_RUNTIME_H */
