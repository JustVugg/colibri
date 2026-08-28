#include "edge_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLI_EDGE_MAX_ADAPTERS 16

struct ColiEdgeEngine {
    const ColiEdgeAdapter *adapter;
    void *impl;
    ColiEdgeCapabilities capabilities;
};

static const ColiEdgeAdapter *g_adapters[COLI_EDGE_MAX_ADAPTERS];
static int g_adapter_count;

static int set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return -1;
}

static int id_valid(const char *id) {
    if (!id || !*id) return 0;
    size_t length = strlen(id);
    if (length >= COLI_EDGE_ENGINE_ID_CAP) return 0;
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '-')) return 0;
    }
    return 1;
}

static int fixed_string_valid(const char *value, size_t capacity) {
    return value && value[0] && memchr(value, '\0', capacity) != NULL;
}

static size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
        case COLI_EDGE_DTYPE_F32: return 4;
        case COLI_EDGE_DTYPE_F16:
        case COLI_EDGE_DTYPE_BF16: return 2;
        default: return 0;
    }
}

static int capabilities_valid(const ColiEdgeCapabilities *capabilities,
                              const ColiEdgeAdapter *adapter) {
    int tokenizer = adapter->tokenize && adapter->detokenize;
    return capabilities->struct_size == sizeof(*capabilities) &&
           capabilities->abi_version == COLI_EDGE_ABI_VERSION &&
           fixed_string_valid(capabilities->engine_id,
                              sizeof(capabilities->engine_id)) &&
           strcmp(capabilities->engine_id, adapter->engine_id) == 0 &&
           fixed_string_valid(capabilities->state_schema,
                              sizeof(capabilities->state_schema)) &&
           fixed_string_valid(capabilities->numeric_class,
                              sizeof(capabilities->numeric_class)) &&
           fixed_string_valid(capabilities->tokenizer_class,
                              sizeof(capabilities->tokenizer_class)) &&
           dtype_size(capabilities->state_dtype) != 0 &&
           capabilities->state_width && capabilities->vocab_size &&
           capabilities->num_layers &&
           !!(capabilities->flags & COLI_EDGE_CAP_TOKENIZE) == tokenizer &&
           !!(capabilities->flags & COLI_EDGE_CAP_DETOKENIZE) == tokenizer &&
           !!(capabilities->flags & COLI_EDGE_CAP_GREEDY) ==
               !!adapter->select &&
           !!(capabilities->flags & COLI_EDGE_CAP_LOGITS) ==
               !!adapter->logits;
}

int coli_edge_adapter_register(const ColiEdgeAdapter *adapter) {
    if (!adapter || adapter->struct_size < sizeof(*adapter) ||
        adapter->abi_version != COLI_EDGE_ABI_VERSION ||
        !id_valid(adapter->engine_id) || !adapter->engine_open ||
        !adapter->engine_destroy || !adapter->embed ||
        !!adapter->tokenize != !!adapter->detokenize)
        return -1;
    for (int i = 0; i < g_adapter_count; i++)
        if (strcmp(g_adapters[i]->engine_id, adapter->engine_id) == 0)
            return -1;
    if (g_adapter_count >= COLI_EDGE_MAX_ADAPTERS) return -1;
    g_adapters[g_adapter_count++] = adapter;
    return 0;
}

const ColiEdgeAdapter *coli_edge_adapter_lookup(const char *engine_id) {
    if (!engine_id) return NULL;
    for (int i = 0; i < g_adapter_count; i++)
        if (strcmp(g_adapters[i]->engine_id, engine_id) == 0)
            return g_adapters[i];
    return NULL;
}

int coli_edge_adapter_count(void) { return g_adapter_count; }

int coli_edge_engine_open(const char *engine_id,
                          const ColiEdgeEngineOptions *options,
                          ColiEdgeEngine **engine,
                          char *error, size_t error_size) {
    if (!engine) return set_error(error, error_size,
                                  "missing edge engine output");
    *engine = NULL;
    if (!options || options->struct_size < sizeof(*options) ||
        !options->model_dir || !*options->model_dir)
        return set_error(error, error_size, "invalid edge engine options");
    const ColiEdgeAdapter *adapter = coli_edge_adapter_lookup(engine_id);
    if (!adapter)
        return set_error(error, error_size, "edge engine is not registered");

    ColiEdgeEngine *created = calloc(1, sizeof(*created));
    if (!created)
        return set_error(error, error_size, "out of memory opening edge engine");
    created->adapter = adapter;
    created->capabilities.struct_size = sizeof(created->capabilities);
    created->capabilities.abi_version = COLI_EDGE_ABI_VERSION;
    if (adapter->engine_open(&created->impl, &created->capabilities, options,
                             error, error_size) || !created->impl) {
        if (created->impl) adapter->engine_destroy(created->impl);
        free(created);
        return -1;
    }
    if (!capabilities_valid(&created->capabilities, adapter)) {
        adapter->engine_destroy(created->impl);
        free(created);
        return set_error(error, error_size,
                         "edge adapter returned incompatible capabilities");
    }
    *engine = created;
    return 0;
}

int coli_edge_engine_capabilities(const ColiEdgeEngine *engine,
                                  ColiEdgeCapabilities *capabilities,
                                  char *error, size_t error_size) {
    if (!engine || !capabilities ||
        capabilities->struct_size < sizeof(*capabilities))
        return set_error(error, error_size, "invalid edge capabilities buffer");
    size_t caller_size = capabilities->struct_size;
    memset(capabilities, 0, caller_size);
    memcpy(capabilities, &engine->capabilities, sizeof(*capabilities));
    return 0;
}

void coli_edge_engine_close(ColiEdgeEngine *engine) {
    if (!engine) return;
    engine->adapter->engine_destroy(engine->impl);
    free(engine);
}

int coli_edge_tokenize(ColiEdgeEngine *engine,
                       const char *text, size_t text_bytes,
                       int32_t *token_ids, size_t token_capacity,
                       size_t *token_count,
                       char *error, size_t error_size) {
    if (!engine || !text || !token_count ||
        (!!token_ids != !!token_capacity))
        return set_error(error, error_size, "invalid edge tokenize request");
    if (!engine->adapter->tokenize)
        return set_error(error, error_size, "edge tokenizer is not supported");
    return engine->adapter->tokenize(engine->impl, text, text_bytes,
                                     token_ids, token_capacity, token_count,
                                     error, error_size);
}

int coli_edge_detokenize(ColiEdgeEngine *engine,
                         const int32_t *token_ids, size_t token_count,
                         char *text, size_t text_capacity, size_t *text_bytes,
                         char *error, size_t error_size) {
    if (!engine || !token_ids || !token_count || !text_bytes ||
        (!!text != !!text_capacity))
        return set_error(error, error_size, "invalid edge detokenize request");
    if (!engine->adapter->detokenize)
        return set_error(error, error_size,
                         "edge detokenizer is not supported");
    return engine->adapter->detokenize(engine->impl, token_ids, token_count,
                                       text, text_capacity, text_bytes,
                                       error, error_size);
}

static int activation_bytes(const ColiEdgeEngine *engine, uint32_t rows,
                            size_t *bytes) {
    size_t element_size = dtype_size(engine->capabilities.state_dtype);
    if (!rows || !element_size ||
        engine->capabilities.state_width > SIZE_MAX / rows ||
        (size_t)rows * engine->capabilities.state_width >
            SIZE_MAX / element_size)
        return -1;
    *bytes = (size_t)rows * engine->capabilities.state_width * element_size;
    return 0;
}

int coli_edge_embed(ColiEdgeEngine *engine,
                    const ColiEdgeEmbedRequest *request,
                    char *error, size_t error_size) {
    size_t expected = 0;
    if (!engine || !request || request->struct_size < sizeof(*request) ||
        !request->rows || !request->token_ids ||
        request->token_count != request->rows || !request->output ||
        activation_bytes(engine, request->rows, &expected) ||
        request->output_bytes != expected)
        return set_error(error, error_size, "invalid edge embed request");
    if (engine->capabilities.max_batch_rows &&
        request->rows > engine->capabilities.max_batch_rows)
        return set_error(error, error_size, "edge batch exceeds capabilities");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return set_error(error, error_size, "edge embedding cancelled");
    return engine->adapter->embed(engine->impl, request, error, error_size);
}

int coli_edge_select(ColiEdgeEngine *engine,
                     const ColiEdgeSelectRequest *request,
                     char *error, size_t error_size) {
    size_t expected = 0;
    if (!engine || !request || request->struct_size < sizeof(*request) ||
        !request->rows || !request->input ||
        activation_bytes(engine, request->rows, &expected) ||
        request->input_bytes != expected || !request->token_ids ||
        request->token_capacity < request->rows ||
        (!!request->scores != !!request->score_capacity) ||
        (request->scores && request->score_capacity < request->rows))
        return set_error(error, error_size, "invalid edge selection request");
    if (!engine->adapter->select)
        return set_error(error, error_size, "edge greedy head is not supported");
    if (engine->capabilities.max_batch_rows &&
        request->rows > engine->capabilities.max_batch_rows)
        return set_error(error, error_size, "edge batch exceeds capabilities");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return set_error(error, error_size, "edge selection cancelled");
    return engine->adapter->select(engine->impl, request, error, error_size);
}

int coli_edge_logits(ColiEdgeEngine *engine,
                     const ColiEdgeLogitsRequest *request,
                     char *error, size_t error_size) {
    size_t expected = 0;
    if (!engine || !request || request->struct_size < sizeof(*request) ||
        !request->rows || !request->input ||
        activation_bytes(engine, request->rows, &expected) ||
        request->input_bytes != expected || !request->logits ||
        engine->capabilities.vocab_size > SIZE_MAX / request->rows ||
        request->logits_capacity <
            (size_t)request->rows * engine->capabilities.vocab_size)
        return set_error(error, error_size, "invalid edge logits request");
    if (!engine->adapter->logits)
        return set_error(error, error_size, "edge logits are not supported");
    if (engine->capabilities.max_batch_rows &&
        request->rows > engine->capabilities.max_batch_rows)
        return set_error(error, error_size, "edge batch exceeds capabilities");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return set_error(error, error_size, "edge logits cancelled");
    return engine->adapter->logits(engine->impl, request, error, error_size);
}
