#include "../segment_runtime.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t begin, end, context;
} FakeEngine;

typedef struct {
    FakeEngine *engine;
    uint64_t position;
} FakeSession;

typedef struct {
    unsigned char bytes[64];
    size_t length, offset;
} MemoryStream;

static void fake_capabilities(ColiSegmentCapabilities *capabilities) {
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_TOKEN_IDS |
                          COLI_SEGMENT_CAP_SNAPSHOT |
                          COLI_SEGMENT_CAP_RANGE_NATIVE |
                          COLI_SEGMENT_CAP_MULTI_SESSION |
                          COLI_SEGMENT_CAP_CPU;
    snprintf(capabilities->engine_id, sizeof(capabilities->engine_id), "fake");
    snprintf(capabilities->state_schema, sizeof(capabilities->state_schema),
             "fake-state-v1");
    snprintf(capabilities->numeric_class, sizeof(capabilities->numeric_class),
             "fake-f32-strict-v1");
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = 4;
    capabilities->max_batch_rows = 8;
    capabilities->max_context_tokens = 32;
    capabilities->num_layers = 6;
}

static int fake_engine_open(void **engine_impl,
                            ColiSegmentCapabilities *capabilities,
                            const ColiSegmentEngineOptions *options,
                            char *error, size_t error_size) {
    (void)error; (void)error_size;
    FakeEngine *engine = (FakeEngine *)calloc(1, sizeof(*engine));
    if (!engine) return -1;
    engine->begin = options->layer_begin;
    engine->end = options->layer_end;
    engine->context = options->context_tokens;
    fake_capabilities(capabilities);
    *engine_impl = engine;
    return 0;
}

static void fake_engine_destroy(void *engine_impl) { free(engine_impl); }

static int fake_session_create(void *engine_impl, void **session_impl,
                               const ColiSegmentSessionOptions *options,
                               char *error, size_t error_size) {
    FakeEngine *engine = (FakeEngine *)engine_impl;
    if (options->context_tokens > engine->context) {
        if (error && error_size) snprintf(error, error_size, "context too large");
        return -1;
    }
    FakeSession *session = (FakeSession *)calloc(1, sizeof(*session));
    if (!session) return -1;
    session->engine = engine;
    *session_impl = session;
    return 0;
}

static void fake_session_destroy(void *session_impl) { free(session_impl); }

static int fake_session_run(void *session_impl,
                            const ColiSegmentRunRequest *request,
                            char *error, size_t error_size) {
    FakeSession *session = (FakeSession *)session_impl;
    size_t count = (size_t)request->rows * 4;
    if (request->position != session->position ||
        request->input_bytes != count * sizeof(float) ||
        request->output_bytes != count * sizeof(float)) {
        if (error && error_size) snprintf(error, error_size, "bad fake run");
        return -1;
    }
    const float *input = (const float *)request->input;
    float *output = (float *)request->output;
    float add = (float)(session->engine->end - session->engine->begin);
    for (size_t i = 0; i < count; i++) output[i] = input[i] + add;
    session->position += request->rows;
    return 0;
}

static int memory_write(void *user_data, const void *data, size_t size) {
    MemoryStream *stream = (MemoryStream *)user_data;
    if (size > sizeof(stream->bytes) - stream->length) return -1;
    memcpy(stream->bytes + stream->length, data, size);
    stream->length += size;
    return 0;
}

static int memory_read(void *user_data, void *data, size_t size) {
    MemoryStream *stream = (MemoryStream *)user_data;
    if (stream->offset > stream->length) return -1;
    if (size > stream->length - stream->offset) return -1;
    memcpy(data, stream->bytes + stream->offset, size);
    stream->offset += size;
    return 0;
}

static int fake_snapshot(void *session_impl, ColiSegmentWriteFn write_fn,
                         void *write_user_data,
                         char *error, size_t error_size) {
    (void)error; (void)error_size;
    FakeSession *session = (FakeSession *)session_impl;
    return write_fn(write_user_data, &session->position, sizeof(session->position));
}

static int fake_restore(void *session_impl, ColiSegmentReadFn read_fn,
                        void *read_user_data,
                        char *error, size_t error_size) {
    (void)error; (void)error_size;
    FakeSession *session = (FakeSession *)session_impl;
    return read_fn(read_user_data, &session->position, sizeof(session->position));
}

static const ColiSegmentAdapter fake_adapter = {
    .struct_size = sizeof(fake_adapter),
    .abi_version = COLI_SEGMENT_ABI_VERSION,
    .engine_id = "fake",
    .engine_open = fake_engine_open,
    .engine_destroy = fake_engine_destroy,
    .session_create = fake_session_create,
    .session_destroy = fake_session_destroy,
    .session_run = fake_session_run,
    .session_snapshot = fake_snapshot,
    .session_restore = fake_restore,
};

static const ColiSegmentAdapter incomplete_snapshot_adapter = {
    .struct_size = sizeof(incomplete_snapshot_adapter),
    .abi_version = COLI_SEGMENT_ABI_VERSION,
    .engine_id = "bad-snapshot",
    .engine_open = fake_engine_open,
    .engine_destroy = fake_engine_destroy,
    .session_create = fake_session_create,
    .session_destroy = fake_session_destroy,
    .session_run = fake_session_run,
    .session_snapshot = fake_snapshot,
};

static int never_cancel(void *unused) { (void)unused; return 0; }
static int always_cancel(void *unused) { (void)unused; return 1; }

int main(void) {
    char error[160] = "";
    assert(coli_segment_adapter_count() == 0);
    assert(coli_segment_adapter_register(NULL) != 0);
    assert(coli_segment_adapter_register(&incomplete_snapshot_adapter) != 0);
    assert(coli_segment_adapter_register(&fake_adapter) == 0);
    assert(coli_segment_adapter_register(&fake_adapter) != 0);
    assert(coli_segment_adapter_count() == 1);
    assert(coli_segment_adapter_lookup("fake") == &fake_adapter);
    assert(coli_segment_adapter_lookup("missing") == NULL);

    ColiSegmentEngineOptions open_options = {
        .struct_size = sizeof(open_options),
        .model_dir = "unused-fixture",
        .layer_begin = 1,
        .layer_end = 4,
        .context_tokens = 16,
    };
    ColiSegmentEngine *engine = NULL;
    assert(coli_segment_engine_open("fake", &open_options, &engine,
                                    error, sizeof(error)) == 0);
    assert(engine);

    ColiSegmentCapabilities capabilities = {.struct_size = sizeof(capabilities)};
    assert(coli_segment_engine_capabilities(engine, &capabilities,
                                            error, sizeof(error)) == 0);
    assert(capabilities.state_width == 4 && capabilities.num_layers == 6);
    assert(capabilities.flags & COLI_SEGMENT_CAP_RANGE_NATIVE);

    struct {
        ColiSegmentCapabilities capabilities;
        uint64_t future_fields[2];
    } future_capabilities;
    memset(&future_capabilities, 0xa5, sizeof(future_capabilities));
    future_capabilities.capabilities.struct_size = sizeof(future_capabilities);
    assert(coli_segment_engine_capabilities(
               engine, &future_capabilities.capabilities,
               error, sizeof(error)) == 0);
    assert(future_capabilities.capabilities.state_width == 4);
    assert(future_capabilities.future_fields[0] == 0);
    assert(future_capabilities.future_fields[1] == 0);

    capabilities.struct_size = 0;
    assert(coli_segment_engine_capabilities(engine, &capabilities,
                                            error, sizeof(error)) != 0);

    ColiSegmentSessionOptions session_options = {
        .struct_size = sizeof(session_options),
        .context_tokens = 16,
    };
    ColiSegmentSession *session = NULL, *second = NULL;
    assert(coli_segment_session_create(engine, &session_options, &session,
                                       error, sizeof(error)) == 0);
    assert(coli_segment_session_create(engine, &session_options, &second,
                                       error, sizeof(error)) == 0);
    assert(coli_segment_engine_close(engine, error, sizeof(error)) != 0);

    float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float output[8] = {0};
    int32_t tokens[2] = {7, 9};
    ColiSegmentRunRequest run = {
        .struct_size = sizeof(run),
        .rows = 2,
        .position = 0,
        .token_ids = tokens,
        .token_count = 2,
        .input = input,
        .input_bytes = sizeof(input),
        .output = output,
        .output_bytes = sizeof(output),
        .should_cancel = never_cancel,
    };
    assert(coli_segment_run(session, &run, error, sizeof(error)) == 0);
    for (size_t i = 0; i < 8; i++) assert(output[i] == input[i] + 3.0f);

    MemoryStream stream = {0};
    assert(coli_segment_snapshot(session, memory_write, &stream,
                                 error, sizeof(error)) == 0);
    assert(stream.length == sizeof(uint64_t));
    stream.offset = 0;
    assert(coli_segment_restore(second, memory_read, &stream,
                                error, sizeof(error)) == 0);

    memset(output, 0, sizeof(output));
    run.position = 2;
    assert(coli_segment_run(second, &run, error, sizeof(error)) == 0);
    for (size_t i = 0; i < 8; i++) assert(output[i] == input[i] + 3.0f);

    run.position = 4;
    run.should_cancel = always_cancel;
    assert(coli_segment_run(second, &run, error, sizeof(error)) != 0);
    run.should_cancel = never_cancel;
    run.token_count = 0;
    run.token_ids = NULL;
    assert(coli_segment_run(second, &run, error, sizeof(error)) != 0);
    run.token_count = 2;
    assert(coli_segment_run(second, &run, error, sizeof(error)) != 0);
    run.token_ids = tokens;
    run.position = 15;
    assert(coli_segment_run(second, &run, error, sizeof(error)) != 0);
    run.position = 4;
    run.input_bytes--;
    assert(coli_segment_run(second, &run, error, sizeof(error)) != 0);

    coli_segment_session_destroy(second);
    coli_segment_session_destroy(session);
    assert(coli_segment_engine_close(engine, error, sizeof(error)) == 0);
    puts("segment runtime tests: ok");
    return 0;
}
