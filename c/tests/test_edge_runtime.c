#include "../edge_runtime.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int unused; } FakeEdge;

static int fake_open(void **impl, ColiEdgeCapabilities *capabilities,
                     const ColiEdgeEngineOptions *options,
                     char *error, size_t error_size) {
    (void)options; (void)error; (void)error_size;
    FakeEdge *created = calloc(1, sizeof(*created));
    if (!created) return -1;
    capabilities->abi_version = COLI_EDGE_ABI_VERSION;
    capabilities->flags = COLI_EDGE_CAP_TOKENIZE |
                          COLI_EDGE_CAP_DETOKENIZE |
                          COLI_EDGE_CAP_GREEDY | COLI_EDGE_CAP_LOGITS |
                          COLI_EDGE_CAP_CPU;
    snprintf(capabilities->engine_id, sizeof(capabilities->engine_id), "fake");
    snprintf(capabilities->state_schema, sizeof(capabilities->state_schema),
             "fake/f32-v1");
    snprintf(capabilities->numeric_class, sizeof(capabilities->numeric_class),
             "fake/strict-v1");
    snprintf(capabilities->tokenizer_class,
             sizeof(capabilities->tokenizer_class), "fake/bytes-v1");
    capabilities->state_dtype = COLI_EDGE_DTYPE_F32;
    capabilities->state_width = 4;
    capabilities->vocab_size = 256;
    capabilities->max_batch_rows = 8;
    capabilities->max_context_tokens = 32;
    capabilities->num_layers = 6;
    capabilities->bos_token_id = 2;
    capabilities->eos_token_id = 3;
    capabilities->resident_bytes = 64;
    *impl = created;
    return 0;
}

static void fake_destroy(void *impl) { free(impl); }

static int fake_tokenize(void *impl, const char *text, size_t text_bytes,
                         int32_t *ids, size_t capacity, size_t *count,
                         char *error, size_t error_size) {
    (void)impl;
    *count = text_bytes;
    if (ids && capacity < text_bytes) {
        if (error && error_size) snprintf(error, error_size, "small token buffer");
        return -1;
    }
    if (ids)
        for (size_t item = 0; item < text_bytes; item++)
            ids[item] = (unsigned char)text[item];
    return 0;
}

static int fake_detokenize(void *impl, const int32_t *ids, size_t count,
                           char *text, size_t capacity, size_t *bytes,
                           char *error, size_t error_size) {
    (void)impl;
    *bytes = count;
    if (text && capacity < count + 1) {
        if (error && error_size) snprintf(error, error_size, "small text buffer");
        return -1;
    }
    if (text) {
        for (size_t item = 0; item < count; item++) text[item] = (char)ids[item];
        text[count] = '\0';
    }
    return 0;
}

static int fake_embed(void *impl, const ColiEdgeEmbedRequest *request,
                      char *error, size_t error_size) {
    (void)impl; (void)error; (void)error_size;
    float *output = request->output;
    for (uint32_t row = 0; row < request->rows; row++)
        for (uint32_t column = 0; column < 4; column++)
            output[(size_t)row * 4 + column] =
                (float)(request->token_ids[row] + (int32_t)column);
    return 0;
}

static int fake_select(void *impl, const ColiEdgeSelectRequest *request,
                       char *error, size_t error_size) {
    (void)impl; (void)error; (void)error_size;
    const float *input = request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        float sum = 0.0f;
        for (uint32_t column = 0; column < 4; column++)
            sum += input[(size_t)row * 4 + column];
        request->token_ids[row] = (int32_t)sum & 255;
        if (request->scores) request->scores[row] = sum;
    }
    return 0;
}

static int fake_logits(void *impl, const ColiEdgeLogitsRequest *request,
                       char *error, size_t error_size) {
    (void)impl; (void)error; (void)error_size;
    const float *input = request->input;
    for (uint32_t row = 0; row < request->rows; row++) {
        float sum = 0.0f;
        for (uint32_t column = 0; column < 4; column++)
            sum += input[(size_t)row * 4 + column];
        float *logits = request->logits + (size_t)row * 256;
        for (uint32_t token = 0; token < 256; token++) logits[token] = -1.0f;
        logits[(uint32_t)((int32_t)sum & 255)] = sum;
    }
    return 0;
}

static const ColiEdgeAdapter fake_adapter = {
    sizeof(ColiEdgeAdapter), COLI_EDGE_ABI_VERSION, "fake",
    fake_open, fake_destroy, fake_tokenize, fake_detokenize,
    fake_embed, fake_select, fake_logits, {0}
};

static const ColiEdgeAdapter incomplete_tokenizer = {
    .struct_size = sizeof(ColiEdgeAdapter),
    .abi_version = COLI_EDGE_ABI_VERSION,
    .engine_id = "bad-tokenizer",
    .engine_open = fake_open,
    .engine_destroy = fake_destroy,
    .tokenize = fake_tokenize,
    .embed = fake_embed,
    .select = fake_select,
    .logits = fake_logits,
};

static int never_cancel(void *unused) { (void)unused; return 0; }
static int always_cancel(void *unused) { (void)unused; return 1; }

int main(void) {
    char error[160] = {0};
    assert(coli_edge_adapter_count() == 0);
    assert(coli_edge_adapter_register(NULL) != 0);
    assert(coli_edge_adapter_register(&incomplete_tokenizer) != 0);
    assert(coli_edge_adapter_register(&fake_adapter) == 0);
    assert(coli_edge_adapter_register(&fake_adapter) != 0);
    assert(coli_edge_adapter_lookup("fake") == &fake_adapter);
    assert(coli_edge_adapter_lookup("missing") == NULL);

    ColiEdgeEngineOptions options = {
        .struct_size = sizeof(options), .model_dir = "unused-fixture",
    };
    ColiEdgeEngine *engine = NULL;
    assert(coli_edge_engine_open("fake", &options, &engine,
                                 error, sizeof(error)) == 0);
    assert(engine);

    struct {
        ColiEdgeCapabilities capabilities;
        uint64_t future[2];
    } capabilities;
    memset(&capabilities, 0xa5, sizeof(capabilities));
    capabilities.capabilities.struct_size = sizeof(capabilities);
    assert(coli_edge_engine_capabilities(
               engine, &capabilities.capabilities,
               error, sizeof(error)) == 0);
    assert(capabilities.capabilities.state_width == 4);
    assert(capabilities.future[0] == 0 && capabilities.future[1] == 0);

    size_t count = 0;
    assert(coli_edge_tokenize(engine, "ab", 2, NULL, 0, &count,
                              error, sizeof(error)) == 0 && count == 2);
    int32_t ids[2] = {0};
    assert(coli_edge_tokenize(engine, "ab", 2, ids, 1, &count,
                              error, sizeof(error)) != 0);
    assert(coli_edge_tokenize(engine, "ab", 2, ids, 2, &count,
                              error, sizeof(error)) == 0);
    assert(ids[0] == 'a' && ids[1] == 'b');

    size_t bytes = 0;
    assert(coli_edge_detokenize(engine, ids, 2, NULL, 0, &bytes,
                                error, sizeof(error)) == 0 && bytes == 2);
    char text[3];
    assert(coli_edge_detokenize(engine, ids, 2, text, sizeof(text), &bytes,
                                error, sizeof(error)) == 0);
    assert(strcmp(text, "ab") == 0);

    float states[8] = {0};
    ColiEdgeEmbedRequest embed = {
        .struct_size = sizeof(embed), .rows = 2,
        .token_ids = ids, .token_count = 2,
        .output = states, .output_bytes = sizeof(states),
        .should_cancel = never_cancel,
    };
    assert(coli_edge_embed(engine, &embed, error, sizeof(error)) == 0);
    assert(states[0] == 97.0f && states[7] == 101.0f);
    embed.output_bytes--;
    assert(coli_edge_embed(engine, &embed, error, sizeof(error)) != 0);
    embed.output_bytes = sizeof(states);
    embed.should_cancel = always_cancel;
    assert(coli_edge_embed(engine, &embed, error, sizeof(error)) != 0);

    int32_t selected[2] = {0};
    float scores[2] = {0};
    ColiEdgeSelectRequest select = {
        .struct_size = sizeof(select), .rows = 2,
        .input = states, .input_bytes = sizeof(states),
        .token_ids = selected, .token_capacity = 2,
        .scores = scores, .score_capacity = 2,
        .should_cancel = never_cancel,
    };
    assert(coli_edge_select(engine, &select, error, sizeof(error)) == 0);
    assert(selected[0] == ((97 + 98 + 99 + 100) & 255));
    assert(scores[1] == 98.0f + 99.0f + 100.0f + 101.0f);
    select.token_capacity = 1;
    assert(coli_edge_select(engine, &select, error, sizeof(error)) != 0);
    select.token_capacity = 2; select.should_cancel = always_cancel;
    assert(coli_edge_select(engine, &select, error, sizeof(error)) != 0);

    float logits[2 * 256];
    ColiEdgeLogitsRequest logits_request = {
        .struct_size = sizeof(logits_request), .rows = 2,
        .input = states, .input_bytes = sizeof(states),
        .logits = logits, .logits_capacity = 2 * 256,
        .should_cancel = never_cancel,
    };
    assert(coli_edge_logits(engine, &logits_request,
                            error, sizeof(error)) == 0);
    assert(logits[(97 + 98 + 99 + 100) & 255] ==
           97.0f + 98.0f + 99.0f + 100.0f);
    logits_request.logits_capacity--;
    assert(coli_edge_logits(engine, &logits_request,
                            error, sizeof(error)) != 0);
    logits_request.logits_capacity = 2 * 256;
    logits_request.should_cancel = always_cancel;
    assert(coli_edge_logits(engine, &logits_request,
                            error, sizeof(error)) != 0);

    coli_edge_engine_close(engine);
    puts("edge runtime tests: ok");
    return 0;
}
