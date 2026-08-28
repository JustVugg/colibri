#include "segment_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLI_SEGMENT_MAX_ADAPTERS 16

struct ColiSegmentEngine {
    const ColiSegmentAdapter *adapter;
    void *impl;
    ColiSegmentCapabilities capabilities;
    uint32_t context_tokens;
    unsigned active_sessions;
};

struct ColiSegmentSession {
    ColiSegmentEngine *engine;
    void *impl;
    uint32_t context_tokens;
};

static const ColiSegmentAdapter *g_adapters[COLI_SEGMENT_MAX_ADAPTERS];
static int g_adapter_count;

static int set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return -1;
}

static int id_valid(const char *id) {
    if (!id || !*id) return 0;
    size_t length = strlen(id);
    if (length >= COLI_SEGMENT_ENGINE_ID_CAP) return 0;
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '-'))
            return 0;
    }
    return 1;
}

static int fixed_string_valid(const char *value, size_t capacity) {
    return value && value[0] && memchr(value, '\0', capacity) != NULL;
}

static size_t dtype_size(uint32_t dtype) {
    switch (dtype) {
        case COLI_SEGMENT_DTYPE_F32: return 4;
        case COLI_SEGMENT_DTYPE_F16:
        case COLI_SEGMENT_DTYPE_BF16: return 2;
        default: return 0;
    }
}

static int capabilities_valid(const ColiSegmentCapabilities *capabilities,
                              const ColiSegmentAdapter *adapter) {
    int has_snapshot = adapter->session_snapshot && adapter->session_restore;
    return capabilities->struct_size == sizeof(*capabilities) &&
           capabilities->abi_version == COLI_SEGMENT_ABI_VERSION &&
           fixed_string_valid(capabilities->engine_id,
                              sizeof(capabilities->engine_id)) &&
           strcmp(capabilities->engine_id, adapter->engine_id) == 0 &&
           fixed_string_valid(capabilities->state_schema,
                              sizeof(capabilities->state_schema)) &&
           fixed_string_valid(capabilities->numeric_class,
                              sizeof(capabilities->numeric_class)) &&
           dtype_size(capabilities->state_dtype) != 0 &&
           capabilities->state_width && capabilities->num_layers &&
           !!(capabilities->flags & COLI_SEGMENT_CAP_SNAPSHOT) == has_snapshot;
}

int coli_segment_adapter_register(const ColiSegmentAdapter *adapter) {
    if (!adapter || adapter->struct_size < sizeof(*adapter) ||
        adapter->abi_version != COLI_SEGMENT_ABI_VERSION ||
        !id_valid(adapter->engine_id) || !adapter->engine_open ||
        !adapter->engine_destroy ||
        !adapter->session_create || !adapter->session_destroy ||
        !adapter->session_run ||
        !!adapter->session_snapshot != !!adapter->session_restore)
        return -1;
    for (int i = 0; i < g_adapter_count; i++)
        if (strcmp(g_adapters[i]->engine_id, adapter->engine_id) == 0)
            return -1;
    if (g_adapter_count >= COLI_SEGMENT_MAX_ADAPTERS) return -1;
    g_adapters[g_adapter_count++] = adapter;
    return 0;
}

const ColiSegmentAdapter *coli_segment_adapter_lookup(const char *engine_id) {
    if (!engine_id) return NULL;
    for (int i = 0; i < g_adapter_count; i++)
        if (strcmp(g_adapters[i]->engine_id, engine_id) == 0)
            return g_adapters[i];
    return NULL;
}

int coli_segment_adapter_count(void) { return g_adapter_count; }

int coli_segment_engine_open(const char *engine_id,
                             const ColiSegmentEngineOptions *options,
                             ColiSegmentEngine **engine,
                             char *error, size_t error_size) {
    if (!engine) return set_error(error, error_size, "missing segment engine output");
    *engine = NULL;
    if (!options || options->struct_size < sizeof(*options) ||
        !options->model_dir || !*options->model_dir ||
        options->layer_begin >= options->layer_end || !options->context_tokens)
        return set_error(error, error_size, "invalid segment engine options");
    const ColiSegmentAdapter *adapter = coli_segment_adapter_lookup(engine_id);
    if (!adapter)
        return set_error(error, error_size, "segment engine is not registered");

    ColiSegmentEngine *created = (ColiSegmentEngine *)calloc(1, sizeof(*created));
    if (!created) return set_error(error, error_size, "out of memory opening segment engine");
    created->adapter = adapter;
    created->capabilities.struct_size = sizeof(created->capabilities);
    created->capabilities.abi_version = COLI_SEGMENT_ABI_VERSION;
    if (adapter->engine_open(&created->impl, &created->capabilities, options,
                             error, error_size) ||
        !created->impl) {
        if (created->impl) adapter->engine_destroy(created->impl);
        free(created);
        return -1;
    }
    if (!capabilities_valid(&created->capabilities, adapter) ||
        options->layer_end > created->capabilities.num_layers ||
        (created->capabilities.max_context_tokens &&
         options->context_tokens > created->capabilities.max_context_tokens)) {
        adapter->engine_destroy(created->impl);
        free(created);
        return set_error(error, error_size,
                         "segment adapter returned incompatible capabilities");
    }
    created->context_tokens = options->context_tokens;
    *engine = created;
    return 0;
}

int coli_segment_engine_capabilities(const ColiSegmentEngine *engine,
                                     ColiSegmentCapabilities *capabilities,
                                     char *error, size_t error_size) {
    if (!engine || !capabilities ||
        capabilities->struct_size < sizeof(*capabilities))
        return set_error(error, error_size,
                         "invalid segment capabilities buffer");
    size_t caller_size = capabilities->struct_size;
    memset(capabilities, 0, caller_size);
    memcpy(capabilities, &engine->capabilities, sizeof(*capabilities));
    return 0;
}

int coli_segment_engine_close(ColiSegmentEngine *engine,
                              char *error, size_t error_size) {
    if (!engine) return 0;
    if (engine->active_sessions)
        return set_error(error, error_size,
                         "cannot close segment engine with live sessions");
    engine->adapter->engine_destroy(engine->impl);
    free(engine);
    return 0;
}

int coli_segment_session_create(ColiSegmentEngine *engine,
                                const ColiSegmentSessionOptions *options,
                                ColiSegmentSession **session,
                                char *error, size_t error_size) {
    if (!session) return set_error(error, error_size, "missing segment session output");
    *session = NULL;
    if (!engine || !options || options->struct_size < sizeof(*options) ||
        !options->context_tokens ||
        options->context_tokens > engine->context_tokens)
        return set_error(error, error_size, "invalid segment session options");
    ColiSegmentSession *created =
        (ColiSegmentSession *)calloc(1, sizeof(*created));
    if (!created)
        return set_error(error, error_size, "out of memory creating segment session");
    created->engine = engine;
    if (engine->adapter->session_create(engine->impl, &created->impl, options,
                                        error, error_size) || !created->impl) {
        if (created->impl) engine->adapter->session_destroy(created->impl);
        free(created);
        return -1;
    }
    created->context_tokens = options->context_tokens;
    engine->active_sessions++;
    *session = created;
    return 0;
}

void coli_segment_session_destroy(ColiSegmentSession *session) {
    if (!session) return;
    ColiSegmentEngine *engine = session->engine;
    engine->adapter->session_destroy(session->impl);
    if (engine->active_sessions) engine->active_sessions--;
    free(session);
}

int coli_segment_run(ColiSegmentSession *session,
                     const ColiSegmentRunRequest *request,
                     char *error, size_t error_size) {
    if (!session || !request || request->struct_size < sizeof(*request) ||
        !request->rows || !request->input || !request->output ||
        !request->input_bytes || !request->output_bytes)
        return set_error(error, error_size, "invalid segment run request");

    const ColiSegmentCapabilities *capabilities = &session->engine->capabilities;
    if (capabilities->max_batch_rows && request->rows > capabilities->max_batch_rows)
        return set_error(error, error_size, "segment batch exceeds capabilities");
    if (!!request->token_ids != !!request->token_count)
        return set_error(error, error_size,
                         "segment token IDs and count must both be present");
    if ((capabilities->flags & COLI_SEGMENT_CAP_TOKEN_IDS) &&
        request->token_count != request->rows)
        return set_error(error, error_size, "segment engine requires one token ID per row");
    if (request->token_count && request->token_count != request->rows)
        return set_error(error, error_size, "segment token count does not match rows");
    if (request->position > session->context_tokens ||
        request->rows > session->context_tokens - request->position)
        return set_error(error, error_size, "segment run exceeds session context");
    size_t element_size = dtype_size(capabilities->state_dtype);
    if (capabilities->state_width > SIZE_MAX / request->rows ||
        (size_t)request->rows * capabilities->state_width >
            SIZE_MAX / element_size)
        return set_error(error, error_size, "segment activation size overflows");
    size_t expected_bytes =
        (size_t)request->rows * capabilities->state_width * element_size;
    if (request->input_bytes != expected_bytes ||
        request->output_bytes != expected_bytes)
        return set_error(error, error_size,
                         "segment activation buffer has the wrong size");
    if (request->should_cancel && request->should_cancel(request->cancel_user_data))
        return set_error(error, error_size, "segment run cancelled");
    return session->engine->adapter->session_run(session->impl, request,
                                                  error, error_size);
}

int coli_segment_snapshot(ColiSegmentSession *session,
                          ColiSegmentWriteFn write_fn, void *write_user_data,
                          char *error, size_t error_size) {
    if (!session || !write_fn)
        return set_error(error, error_size, "invalid segment snapshot request");
    if (!session->engine->adapter->session_snapshot)
        return set_error(error, error_size, "segment snapshots are not supported");
    return session->engine->adapter->session_snapshot(
        session->impl, write_fn, write_user_data, error, error_size);
}

int coli_segment_restore(ColiSegmentSession *session,
                         ColiSegmentReadFn read_fn, void *read_user_data,
                         char *error, size_t error_size) {
    if (!session || !read_fn)
        return set_error(error, error_size, "invalid segment restore request");
    if (!session->engine->adapter->session_restore)
        return set_error(error, error_size, "segment restore is not supported");
    return session->engine->adapter->session_restore(
        session->impl, read_fn, read_user_data, error, error_size);
}
