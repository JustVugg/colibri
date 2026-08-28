#include "../edge_adapters.h"
#include "../edge_runtime.h"
#include "../json.h"
#include "../segment_adapters.h"
#include "../segment_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "%s: %s\n", family, message); goto fail; } \
} while (0)

static char *read_file(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (!stream) return NULL;
    if (fseek(stream, 0, SEEK_END)) { fclose(stream); return NULL; }
    long length = ftell(stream);
    if (length < 0 || fseek(stream, 0, SEEK_SET)) { fclose(stream); return NULL; }
    char *data = malloc((size_t)length + 1u);
    if (!data || fread(data, 1, (size_t)length, stream) != (size_t)length) {
        free(data); fclose(stream); return NULL;
    }
    data[length] = '\0'; fclose(stream);
    return data;
}

static int32_t *read_ids(jval *object, const char *key, int *count) {
    jval *array = json_get(object, key);
    if (!array || array->t != J_ARR || array->len < 1) return NULL;
    int32_t *ids = malloc((size_t)array->len * sizeof(*ids));
    if (!ids) return NULL;
    for (int item = 0; item < array->len; item++) {
        if (!array->kids[item] || array->kids[item]->t != J_NUM ||
            array->kids[item]->num < INT32_MIN ||
            array->kids[item]->num > INT32_MAX ||
            array->kids[item]->num !=
                (double)(int32_t)array->kids[item]->num) {
            free(ids); return NULL;
        }
        ids[item] = (int32_t)array->kids[item]->num;
    }
    *count = array->len;
    return ids;
}

static int register_all(void) {
    return coli_glm_segment_adapter_register() ||
           coli_glm53_segment_adapter_register() ||
           coli_inkling_segment_adapter_register() ||
           coli_kimi_segment_adapter_register() ||
           coli_olmoe_segment_adapter_register() ||
           coli_qwen36_segment_adapter_register() ||
           coli_deepseek_v4_segment_adapter_register() ||
           coli_glm_edge_adapter_register() ||
           coli_glm53_edge_adapter_register() ||
           coli_inkling_edge_adapter_register() ||
           coli_kimi_edge_adapter_register() ||
           coli_olmoe_edge_adapter_register() ||
           coli_qwen36_edge_adapter_register() ||
           coli_deepseek_v4_edge_adapter_register();
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s FAMILY MODEL_DIR REF_JSON [MAX_NEW]\n", argv[0]);
        return 2;
    }
    const char *family = argv[1], *model_dir = argv[2], *ref_path = argv[3];
    int max_new = argc == 5 ? atoi(argv[4]) : 3;
    if (max_new < 1) return 2;
    char error[512] = {0};
    char *json_text = NULL, *arena = NULL;
    jval *root = NULL;
    int32_t *prompt = NULL, *full = NULL;
    int prompt_count = 0, full_count = 0;
    ColiEdgeEngine *edge = NULL;
    ColiSegmentEngine *segment = NULL;
    ColiSegmentSession *session = NULL;
    float *input = NULL, *output = NULL, *logits = NULL;
    int32_t *probe_ids = NULL;
    char *probe_text = NULL;
    int status = 1;

    REQUIRE(register_all() == 0, "cannot register all adapters");
    json_text = read_file(ref_path);
    REQUIRE(json_text != NULL, "cannot read reference JSON");
    root = json_parse(json_text, &arena);
    REQUIRE(root && root->t == J_OBJ, "invalid reference JSON");
    jval *case_object = root;
    const char *full_key = "full_ids";
    if (!strcmp(family, "kimi") || !strcmp(family, "deepseek_v4")) {
        jval *cases = json_get(root, "cases");
        case_object = cases ? json_get(cases, "short") : NULL;
        full_key = "greedy_full_ids";
    }
    REQUIRE(case_object && case_object->t == J_OBJ,
            "reference has no selected case");
    prompt = read_ids(case_object, "prompt_ids", &prompt_count);
    full = read_ids(case_object, full_key, &full_count);
    REQUIRE(prompt && full && full_count > prompt_count,
            "reference token arrays are missing");
    if (max_new > full_count - prompt_count) max_new = full_count - prompt_count;

    ColiEdgeEngineOptions edge_options = {
        .struct_size = sizeof(edge_options), .model_dir = model_dir,
    };
    REQUIRE(coli_edge_engine_open(family, &edge_options, &edge,
                                  error, sizeof(error)) == 0, error);
    ColiEdgeCapabilities edge_cap = {.struct_size = sizeof(edge_cap)};
    REQUIRE(coli_edge_engine_capabilities(edge, &edge_cap,
                                          error, sizeof(error)) == 0, error);
    REQUIRE(edge_cap.flags & COLI_EDGE_CAP_TOKENIZE,
            "real adapter does not expose tokenization");
    REQUIRE(edge_cap.flags & COLI_EDGE_CAP_GREEDY,
            "real adapter does not expose the model head");
    REQUIRE(edge_cap.flags & COLI_EDGE_CAP_LOGITS,
            "real adapter does not expose final-head logits");

    static const char tokenizer_probe[] = "edge";
    size_t probe_count = 0, probe_bytes = 0;
    REQUIRE(coli_edge_tokenize(edge, tokenizer_probe,
                               sizeof(tokenizer_probe) - 1u,
                               NULL, 0, &probe_count,
                               error, sizeof(error)) == 0 && probe_count,
            "tokenizer sizing pass failed");
    probe_ids = malloc(probe_count * sizeof(*probe_ids));
    REQUIRE(probe_ids != NULL, "out of memory for tokenizer probe");
    REQUIRE(coli_edge_tokenize(edge, tokenizer_probe,
                               sizeof(tokenizer_probe) - 1u,
                               probe_ids, probe_count, &probe_count,
                               error, sizeof(error)) == 0,
            "tokenizer encode failed");
    REQUIRE(coli_edge_detokenize(edge, probe_ids, probe_count,
                                 NULL, 0, &probe_bytes,
                                 error, sizeof(error)) == 0,
            "detokenizer sizing pass failed");
    probe_text = malloc(probe_bytes + 1u);
    REQUIRE(probe_text != NULL, "out of memory for detokenizer probe");
    REQUIRE(coli_edge_detokenize(edge, probe_ids, probe_count,
                                 probe_text, probe_bytes + 1u, &probe_bytes,
                                 error, sizeof(error)) == 0,
            "detokenizer decode failed");
    REQUIRE(probe_bytes == sizeof(tokenizer_probe) - 1u &&
            !memcmp(probe_text, tokenizer_probe, probe_bytes),
            "tokenizer round-trip differs");
    free(probe_text); probe_text = NULL;
    free(probe_ids); probe_ids = NULL;

    uint32_t context = (uint32_t)(prompt_count + max_new + 2);
    ColiSegmentEngineOptions segment_options = {
        .struct_size = sizeof(segment_options), .model_dir = model_dir,
        .layer_begin = 0, .layer_end = edge_cap.num_layers,
        .context_tokens = context,
    };
    REQUIRE(coli_segment_engine_open(family, &segment_options, &segment,
                                     error, sizeof(error)) == 0, error);
    ColiSegmentCapabilities segment_cap = {
        .struct_size = sizeof(segment_cap)
    };
    REQUIRE(coli_segment_engine_capabilities(segment, &segment_cap,
                                             error, sizeof(error)) == 0, error);
    REQUIRE(segment_cap.state_dtype == edge_cap.state_dtype &&
            segment_cap.state_width == edge_cap.state_width &&
            !strcmp(segment_cap.state_schema, edge_cap.state_schema) &&
            !strcmp(segment_cap.numeric_class, edge_cap.numeric_class),
            "Edge/Segment activation identity differs");
    ColiSegmentSessionOptions session_options = {
        .struct_size = sizeof(session_options), .context_tokens = context,
    };
    REQUIRE(coli_segment_session_create(segment, &session_options, &session,
                                        error, sizeof(error)) == 0, error);

    size_t row_bytes = (size_t)edge_cap.state_width * sizeof(float);
    REQUIRE((size_t)prompt_count <= SIZE_MAX / row_bytes,
            "prompt activation size overflows");
    input = malloc((size_t)prompt_count * row_bytes);
    output = malloc((size_t)prompt_count * row_bytes);
    REQUIRE(input && output, "out of memory for prompt activations");
    ColiEdgeEmbedRequest embed = {
        .struct_size = sizeof(embed), .rows = (uint32_t)prompt_count,
        .token_ids = prompt,
        .token_count = (size_t)prompt_count,
        .output = input, .output_bytes = (size_t)prompt_count * row_bytes,
    };
    REQUIRE(coli_edge_embed(edge, &embed, error, sizeof(error)) == 0, error);
    ColiSegmentRunRequest run = {
        .struct_size = sizeof(run), .rows = (uint32_t)prompt_count,
        .position = 0, .token_ids = prompt,
        .token_count = (size_t)prompt_count,
        .input = input, .input_bytes = (size_t)prompt_count * row_bytes,
        .output = output, .output_bytes = (size_t)prompt_count * row_bytes,
    };
    REQUIRE(coli_segment_run(session, &run, error, sizeof(error)) == 0, error);

    int32_t predicted = -1;
    float score = 0.0f;
    ColiEdgeSelectRequest select = {
        .struct_size = sizeof(select), .rows = 1,
        .input = output + (size_t)(prompt_count - 1) * edge_cap.state_width,
        .input_bytes = row_bytes, .token_ids = &predicted,
        .token_capacity = 1, .scores = &score, .score_capacity = 1,
    };
    REQUIRE(coli_edge_select(edge, &select, error, sizeof(error)) == 0, error);
    REQUIRE(predicted == full[prompt_count],
            "first generated token differs from the independent oracle");
    logits = malloc((size_t)edge_cap.vocab_size * sizeof(*logits));
    REQUIRE(logits != NULL, "out of memory for Edge logits");
    ColiEdgeLogitsRequest logits_request = {
        .struct_size = sizeof(logits_request), .rows = 1,
        .input = select.input, .input_bytes = row_bytes,
        .logits = logits, .logits_capacity = edge_cap.vocab_size,
    };
    REQUIRE(coli_edge_logits(edge, &logits_request,
                             error, sizeof(error)) == 0, error);
    int32_t logits_winner = 0;
    for (uint32_t token = 1; token < edge_cap.vocab_size; token++)
        if (logits[token] > logits[logits_winner])
            logits_winner = (int32_t)token;
    REQUIRE(logits_winner == predicted,
            "argmax(logits) differs from deterministic Edge selection");
    free(logits); logits = NULL;
    size_t generated_bytes = 0;
    REQUIRE(coli_edge_detokenize(edge, &predicted, 1, NULL, 0,
                                 &generated_bytes, error, sizeof(error)) == 0 &&
            generated_bytes > 0,
            "generated token cannot be detokenized");

    free(input); free(output);
    input = malloc(row_bytes); output = malloc(row_bytes);
    REQUIRE(input && output, "out of memory for decode activations");
    for (int generated = 1; generated < max_new; generated++) {
        int32_t token = predicted;
        embed.rows = 1; embed.token_ids = &token; embed.token_count = 1;
        embed.output = input; embed.output_bytes = row_bytes;
        REQUIRE(coli_edge_embed(edge, &embed, error, sizeof(error)) == 0, error);
        run.rows = 1;
        run.position = (uint64_t)prompt_count + generated - 1u;
        run.token_ids = &token; run.token_count = 1;
        run.input = input; run.input_bytes = row_bytes;
        run.output = output; run.output_bytes = row_bytes;
        REQUIRE(coli_segment_run(session, &run, error, sizeof(error)) == 0, error);
        select.input = output;
        REQUIRE(coli_edge_select(edge, &select, error, sizeof(error)) == 0, error);
        REQUIRE(predicted == full[prompt_count + generated],
                "decode token differs from the independent oracle");
    }
    printf("%s Edge -> full Segment -> Edge: %d oracle tokens ok\n",
           family, max_new);
    status = 0;

fail:
    if (status && error[0]) fprintf(stderr, "%s: %s\n", family, error);
    free(probe_text); free(probe_ids);
    free(logits); free(output); free(input);
    coli_segment_session_destroy(session);
    if (segment) (void)coli_segment_engine_close(segment, NULL, 0);
    coli_edge_engine_close(edge);
    free(full); free(prompt);
    json_free(root); free(arena); free(json_text);
    return status;
}
