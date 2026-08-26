#include "../segment_adapters.h"
#include "../segment_runtime.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char *data;
    size_t size, capacity, offset;
} MemoryStream;

static int memory_write(void *user_data, const void *data, size_t size) {
    MemoryStream *stream = user_data;
    if (size > SIZE_MAX - stream->size) return -1;
    size_t needed = stream->size + size;
    if (needed > stream->capacity) {
        size_t capacity = stream->capacity ? stream->capacity : 1024;
        while (capacity < needed) {
            if (capacity > SIZE_MAX / 2) return -1;
            capacity *= 2;
        }
        unsigned char *grown = realloc(stream->data, capacity);
        if (!grown) return -1;
        stream->data = grown;
        stream->capacity = capacity;
    }
    memcpy(stream->data + stream->size, data, size);
    stream->size = needed;
    return 0;
}

static int memory_read(void *user_data, void *data, size_t size) {
    MemoryStream *stream = user_data;
    if (size > stream->size - stream->offset) return -1;
    memcpy(data, stream->data + stream->offset, size);
    stream->offset += size;
    return 0;
}

static int register_all(void) {
    return coli_glm_segment_adapter_register() ||
           coli_inkling_segment_adapter_register() ||
           coli_kimi_segment_adapter_register() ||
           coli_olmoe_segment_adapter_register() ||
           coli_qwen36_segment_adapter_register() ||
           coli_deepseek_v4_segment_adapter_register();
}

static int close_enough(const float *left, const float *right, size_t count,
                        const char *label) {
    /* A segment boundary may change the engine's reduction schedule and the
     * point at which an activation is rounded/stored.  Require a tight f32
     * numerical match instead of bit identity; token-exact standalone oracles
     * remain the stricter end-to-end gate. */
    for (size_t item = 0; item < count; item++) {
        float difference = fabsf(left[item] - right[item]);
        float scale = fmaxf(1.0f, fmaxf(fabsf(left[item]), fabsf(right[item])));
        if (!isfinite(left[item]) || !isfinite(right[item]) ||
            difference > 2e-5f * scale) {
            fprintf(stderr,
                    "%s differs at %zu: %.9g versus %.9g (difference %.9g)\n",
                    label, item, left[item], right[item], difference);
            return -1;
        }
    }
    return 0;
}

static int open_engine(const char *family, const char *model_dir,
                       uint32_t begin, uint32_t end, uint32_t context,
                       ColiSegmentEngine **engine,
                       ColiSegmentCapabilities *capabilities,
                       char *error, size_t error_size) {
    ColiSegmentEngineOptions options = {
        .struct_size = sizeof(options),
        .model_dir = model_dir,
        .layer_begin = begin,
        .layer_end = end,
        .context_tokens = context,
        .backend_mask = COLI_SEGMENT_CAP_CPU,
    };
    if (coli_segment_engine_open(family, &options, engine, error, error_size))
        return -1;
    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    return coli_segment_engine_capabilities(*engine, capabilities,
                                             error, error_size);
}

static int create_session(ColiSegmentEngine *engine, uint32_t context,
                          ColiSegmentSession **session,
                          char *error, size_t error_size) {
    ColiSegmentSessionOptions options = {
        .struct_size = sizeof(options), .context_tokens = context
    };
    return coli_segment_session_create(engine, &options, session,
                                        error, error_size);
}

static int run_rows(ColiSegmentSession *session, uint64_t position,
                    uint32_t rows, uint32_t width, const float *input,
                    float *output, const int32_t *tokens,
                    char *error, size_t error_size) {
    size_t bytes = (size_t)rows * width * sizeof(float);
    ColiSegmentRunRequest request = {
        .struct_size = sizeof(request), .rows = rows, .position = position,
        .token_ids = tokens, .token_count = rows,
        .input = input, .input_bytes = bytes,
        .output = output, .output_bytes = bytes,
    };
    return coli_segment_run(session, &request, error, error_size);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s FAMILY MODEL_DIR BEGIN SPLIT END CONTEXT\n", argv[0]);
        return 2;
    }
    const char *family = argv[1], *model_dir = argv[2];
    uint32_t begin = (uint32_t)strtoul(argv[3], NULL, 10);
    uint32_t split = (uint32_t)strtoul(argv[4], NULL, 10);
    uint32_t end = (uint32_t)strtoul(argv[5], NULL, 10);
    uint32_t context = (uint32_t)strtoul(argv[6], NULL, 10);
    if (!(begin < split && split < end) || context < 4 || register_all()) {
        fprintf(stderr, "invalid real-adapter test setup\n");
        return 2;
    }

    char error[512] = {0};
    ColiSegmentEngine *full_engine = NULL, *left_engine = NULL,
                      *right_engine = NULL;
    ColiSegmentSession *full = NULL, *restored = NULL, *left = NULL,
                       *right = NULL;
    ColiSegmentCapabilities full_caps, left_caps, right_caps;
    int result = 1;
    if (open_engine(family, model_dir, begin, end, context,
                    &full_engine, &full_caps, error, sizeof(error)) ||
        open_engine(family, model_dir, begin, split, context,
                    &left_engine, &left_caps, error, sizeof(error)) ||
        open_engine(family, model_dir, split, end, context,
                    &right_engine, &right_caps, error, sizeof(error)))
        goto cleanup;
    if (full_caps.state_dtype != COLI_SEGMENT_DTYPE_F32 ||
        full_caps.state_width != left_caps.state_width ||
        full_caps.state_width != right_caps.state_width ||
        !(full_caps.flags & COLI_SEGMENT_CAP_RANGE_NATIVE) ||
        !(full_caps.flags & COLI_SEGMENT_CAP_SNAPSHOT)) {
        snprintf(error, sizeof(error), "incompatible real-adapter capabilities");
        goto cleanup;
    }
    if (create_session(full_engine, context, &full, error, sizeof(error)) ||
        create_session(full_engine, context, &restored, error, sizeof(error)) ||
        create_session(left_engine, context, &left, error, sizeof(error)) ||
        create_session(right_engine, context, &right, error, sizeof(error)))
        goto cleanup;

    uint32_t width = full_caps.state_width;
    const uint32_t first_rows = 2, next_rows = 1;
    size_t first_cells = (size_t)first_rows * width;
    float *first_input = malloc(first_cells * sizeof(*first_input));
    float *full_first = malloc(first_cells * sizeof(*full_first));
    float *left_first = malloc(first_cells * sizeof(*left_first));
    float *chain_first = malloc(first_cells * sizeof(*chain_first));
    float *next_input = malloc((size_t)width * sizeof(*next_input));
    float *full_next = malloc((size_t)width * sizeof(*full_next));
    float *restored_next = malloc((size_t)width * sizeof(*restored_next));
    float *left_next = malloc((size_t)width * sizeof(*left_next));
    float *chain_next = malloc((size_t)width * sizeof(*chain_next));
    if (!first_input || !full_first || !left_first || !chain_first ||
        !next_input || !full_next || !restored_next || !left_next ||
        !chain_next) {
        snprintf(error, sizeof(error), "out of memory allocating activations");
        free(chain_next); free(left_next); free(restored_next); free(full_next);
        free(next_input); free(chain_first); free(left_first); free(full_first);
        free(first_input);
        goto cleanup;
    }
    for (size_t item = 0; item < first_cells; item++)
        first_input[item] = 0.025f * sinf((float)item * 0.071f) +
                            0.003f * cosf((float)item * 0.013f);
    for (uint32_t item = 0; item < width; item++)
        next_input[item] = 0.021f * cosf((float)item * 0.047f) - 0.002f;
    const int32_t first_tokens[2] = {7, 23};
    const int32_t next_token[1] = {41};

    if (run_rows(full, 0, first_rows, width, first_input, full_first,
                 first_tokens, error, sizeof(error)) ||
        run_rows(left, 0, first_rows, width, first_input, left_first,
                 first_tokens, error, sizeof(error)) ||
        run_rows(right, 0, first_rows, width, left_first, chain_first,
                 first_tokens, error, sizeof(error)) ||
        close_enough(full_first, chain_first, first_cells,
                     "full versus chained first chunk"))
        goto free_buffers;

    MemoryStream checkpoint = {0};
    if (coli_segment_snapshot(full, memory_write, &checkpoint,
                              error, sizeof(error)))
        goto free_buffers;
    checkpoint.offset = 0;
    if (coli_segment_restore(restored, memory_read, &checkpoint,
                             error, sizeof(error))) {
        free(checkpoint.data);
        goto free_buffers;
    }
    if (run_rows(full, first_rows, next_rows, width, next_input, full_next,
                 next_token, error, sizeof(error)) ||
        run_rows(restored, first_rows, next_rows, width, next_input,
                 restored_next, next_token, error, sizeof(error)) ||
        close_enough(full_next, restored_next, width,
                     "snapshot/replay continuation") ||
        run_rows(left, first_rows, next_rows, width, next_input, left_next,
                 next_token, error, sizeof(error)) ||
        run_rows(right, first_rows, next_rows, width, left_next, chain_next,
                 next_token, error, sizeof(error)) ||
        close_enough(full_next, chain_next, width,
                     "full versus chained continuation")) {
        free(checkpoint.data);
        goto free_buffers;
    }

    MemoryStream before_bad = {0}, after_bad = {0};
    if (coli_segment_snapshot(restored, memory_write, &before_bad,
                              error, sizeof(error))) {
        free(checkpoint.data);
        goto free_buffers;
    }
    if (checkpoint.size) checkpoint.data[checkpoint.size - 1] ^= 0x5a;
    checkpoint.offset = 0;
    if (coli_segment_restore(restored, memory_read, &checkpoint,
                             error, sizeof(error)) == 0) {
        snprintf(error, sizeof(error), "corrupt snapshot was accepted");
        free(before_bad.data); free(checkpoint.data);
        goto free_buffers;
    }
    error[0] = '\0';
    if (coli_segment_snapshot(restored, memory_write, &after_bad,
                              error, sizeof(error)) ||
        before_bad.size != after_bad.size ||
        memcmp(before_bad.data, after_bad.data, before_bad.size)) {
        snprintf(error, sizeof(error), "failed restore mutated live state");
        free(after_bad.data); free(before_bad.data); free(checkpoint.data);
        goto free_buffers;
    }
    free(after_bad.data); free(before_bad.data); free(checkpoint.data);
    result = 0;

free_buffers:
    free(chain_next); free(left_next); free(restored_next); free(full_next);
    free(next_input); free(chain_first); free(left_first); free(full_first);
    free(first_input);
cleanup:
    if (right) coli_segment_session_destroy(right);
    if (left) coli_segment_session_destroy(left);
    if (restored) coli_segment_session_destroy(restored);
    if (full) coli_segment_session_destroy(full);
    if (right_engine) (void)coli_segment_engine_close(right_engine, NULL, 0);
    if (left_engine) (void)coli_segment_engine_close(left_engine, NULL, 0);
    if (full_engine) (void)coli_segment_engine_close(full_engine, NULL, 0);
    if (result) {
        fprintf(stderr, "%s real Segment adapter failed: %s\n", family,
                error[0] ? error : "numerical mismatch");
        return 1;
    }
    printf("%s real Segment range/chaining/snapshot: ok\n", family);
    return 0;
}
