#define _CRT_SECURE_NO_WARNINGS

#include "gemma4_backend.h"
#include "gemma4_model.h"
#include "gemma4_sampling.h"
#include "gemma4_tokenizer.h"
#include "gemma4_tools.h"
#include "gemma4_vision.h"
#include "grammar.h"
#include "schema_gbnf.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <sys/select.h>
#include <unistd.h>
#endif

static void usage(const char *program) {
    fprintf(stderr,
        "Colibri Gemma 4 expert bridge\n\n"
        "usage:\n"
        "  %s --version\n"
        "  %s model-info MODEL.gguf\n"
        "  %s vision-info MMPROJ.gguf [SOURCE_WIDTH SOURCE_HEIGHT]\n"
        "  %s vision-image-info MMPROJ.gguf IMAGE\n"
        "  %s vision-patch-probe MMPROJ.gguf SOURCE_WIDTH SOURCE_HEIGHT\n"
        "      [--output-f32 FILE]\n"
        "  %s vision-encode-probe MMPROJ.gguf SOURCE_WIDTH SOURCE_HEIGHT\n"
        "      [--layers N] [--start-layer N --input-f32 FILE]\n"
        "      [--output-f32 FILE] [--prepared-f32 FILE]\n"
        "      [--trace-layer N --trace-dir DIRECTORY]\n"
        "  %s tokenize MODEL.gguf TEXT [--no-bos]\n"
        "  %s tokenize-file MODEL.gguf UTF8_FILE [--no-bos]\n"
        "  %s chat-user MODEL.gguf TEXT [--system TEXT|--system-file FILE]\n"
        "      [--tools-file FILE]\n"
        "  %s chat-user-file MODEL.gguf UTF8_FILE [--system TEXT|--system-file FILE]\n"
        "      [--tools-file FILE]\n"
        "  %s image-chat-user MODEL.gguf TEXT [--system TEXT|--system-file FILE]\n"
        "      [--tools-file FILE]\n"
        "  %s embed MODEL.gguf TOKEN [--output-f32 FILE]\n"
        "  %s lm-head MODEL.gguf --input-f32 FILE [--top N]\n"
        "      [--normalized-f32 FILE] [--logits-f32 FILE]\n"
        "  %s next-token MODEL.gguf PACKED_DIR TEXT [--chat] [--top N]\n"
        "      [--system TEXT|--system-file FILE]\n"
        "      [--tools-file FILE]\n"
        "      [--image IMAGE --mmproj MMPROJ.gguf]\n"
        "      [--image-embeddings-f32 FILE]\n"
        "      [--residual-f32 FILE] [--logits-f32 FILE] [--expert-cache N]\n"
        "      [--expert-pins N] [--expert-prefetch] [--expert-lookahead]\n"
        "      [--cuda-device N]\n"
        "      [--expert-usage FILE]\n"
        "  %s next-token-file MODEL.gguf PACKED_DIR UTF8_FILE [--chat] [--top N]\n"
        "      [--system TEXT|--system-file FILE] [--tools-file FILE]\n"
        "  %s generate MODEL.gguf PACKED_DIR TEXT [--chat] [--max-new N]\n"
        "      [--system TEXT|--system-file FILE]\n"
        "      [--tools-file FILE]\n"
        "      [--image IMAGE --mmproj MMPROJ.gguf]\n"
        "      [--temperature F] [--top-k N] [--top-p F] [--seed N]\n"
        "      [--show-special] [--expert-cache N] [--expert-pins N]\n"
        "      [--expert-prefetch] [--expert-lookahead]\n"
        "      [--cuda-device N]\n"
        "      [--expert-usage FILE]\n"
        "  %s generate-file MODEL.gguf PACKED_DIR UTF8_FILE [--chat]\n"
        "      [--system TEXT|--system-file FILE]\n"
        "      [--tools-file FILE]\n"
        "      [--max-new N] [--show-special] [--expert-cache N]\n"
        "      [--expert-pins N] [--expert-prefetch]\n"
        "      [--expert-usage FILE]\n"
        "  %s chat MODEL.gguf PACKED_DIR [--max-context N] [--max-new N]\n"
        "      [--system TEXT|--system-file FILE]\n"
        "      [--tools-file FILE] [--show-special]\n"
        "      [--temperature F] [--top-k N] [--top-p F] [--seed N]\n"
        "      [--expert-cache N] [--expert-pins N] [--expert-prefetch]\n"
        "      [--expert-lookahead]\n"
        "      [--cuda-device N]\n"
        "      [--expert-usage FILE]\n"
        "  %s route MODEL.gguf --layer N [--input-f32 FILE] [--seed N]\n"
        "      [--probabilities-f32 FILE]\n"
        "  %s attention-proj MODEL.gguf --layer N [--position N] [--input-f32 FILE]\n"
        "      [--query-f32 FILE] [--key-f32 FILE] [--value-f32 FILE] [--seed N]\n"
        "  %s attention-seq MODEL.gguf --layer N [--tokens N] [--input-f32 FILE]\n"
        "      [--output-f32 FILE] [--seed N]\n"
        "  %s dense-mlp MODEL.gguf --layer N [--input-f32 FILE]\n"
        "      [--output-f32 FILE] [--seed N]\n"
        "  %s routed-mlp MODEL.gguf PACKED_DIR --layer N [--input-f32 FILE]\n"
        "      [--output-f32 FILE] [--seed N]\n"
        "  %s layer-seq MODEL.gguf PACKED_DIR --layer N [--tokens N]\n"
        "      [--input-f32 FILE] [--output-f32 FILE] [--seed N]\n"
        "  %s info PACKED_DIR\n"
        "  %s expert PACKED_DIR --layer N --expert N [--input-f32 FILE]\n"
        "      [--output-f32 FILE] [--seed N] [--repeat N] [--cuda-device N]\n\n"
        "model-info and route operate directly on the GGUF. PACKED_DIR is\n"
        "produced by g4lab pack and may fall back to that GGUF for unpacked\n"
        "experts. Generation is still in progress.\n",
        program, program, program, program, program, program, program, program,
        program,
        program,
        program,
        program,
        program, program, program, program, program, program, program, program,
        program,
        program, program, program, program, program);
}

static int parse_u32(const char *text, uint32_t *result) {
    char *end = NULL;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno || !end || *end || value > UINT32_MAX) return -1;
    *result = (uint32_t)value;
    return 0;
}

static int option_value(int argc, char **argv, const char *name,
                        const char **result) {
    int i;
    for (i = 0; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            if (i + 1 >= argc) return -1;
            *result = argv[i + 1];
            return 1;
        }
    }
    return 0;
}

static int read_f32(const char *path, float *values, size_t count) {
    FILE *file = fopen(path, "rb");
    long extra;
    if (!file) {
        fprintf(stderr, "cannot open input %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fread(values, sizeof(float), count, file) != count) {
        fclose(file);
        fprintf(stderr, "input must contain exactly %zu float32 values\n", count);
        return -1;
    }
    extra = fgetc(file);
    fclose(file);
    if (extra != EOF) {
        fprintf(stderr, "input must contain exactly %zu float32 values\n", count);
        return -1;
    }
    return 0;
}

static int write_f32(const char *path, const float *values, size_t count) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "cannot create output %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fwrite(values, sizeof(float), count, file) != count || fclose(file) != 0) {
        fprintf(stderr, "failed writing output %s\n", path);
        return -1;
    }
    return 0;
}

static uint64_t next_random(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static void make_input(float *values, size_t count, uint32_t seed) {
    uint64_t state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
    size_t i;
    for (i = 0; i < count; ++i) {
        uint32_t bits = (uint32_t)(next_random(&state) >> 40);
        values[i] = ((float)bits / 16777215.0F - 0.5F) * 0.1F;
    }
}

static int command_info(const char *directory) {
    coli_gemma4_backend gemma;
    uint32_t i;
    if (coli_gemma4_open_packed(&gemma, directory) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        return 1;
    }
    printf("architecture:       gemma4\n");
    printf("routed layers:      %u\n", gemma.config.n_layer);
    printf("experts per layer:  %u\n", gemma.config.n_expert);
    printf("experts selected:   %u\n", gemma.config.n_expert_used);
    printf("expert shape:       %u -> %u -> %u\n",
           gemma.config.n_embd, gemma.config.n_expert_ff, gemma.config.n_embd);
    printf("source GGUF:        %s\n", gemma.source);
    for (i = 0; i < gemma.layer_count; ++i) {
        const coli_gemma4_layer *layer = &gemma.layers[i];
        printf("layer %2u: %s, %" PRIu64 " bytes/record%s, %s\n",
               layer->layer, layer->packed_file, layer->record_stride,
               layer->has_scale ? ", learned scale" : ", unit scale",
               coli_gemma4_has_packed_layer(&gemma, layer) ?
                   "packed" : "direct GGUF fallback");
    }
    coli_gemma4_close(&gemma);
    return 0;
}

static int parse_u64(const char *text, uint64_t *result) {
    char *end = NULL;
    unsigned long long value;
    if (!text || text[0] == '-') return -1;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno || !end || *end) return -1;
    *result = (uint64_t)value;
    return 0;
}

static int read_utf8_text(const char *path, char **text, size_t *text_bytes) {
    FILE *file = fopen(path, "rb");
    char *storage;
    long length;
    if (!file || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file) fclose(file);
        fprintf(stderr, "cannot read UTF-8 input %s\n", path);
        return -1;
    }
    if ((uint64_t)length > SIZE_MAX - 1) {
        fclose(file);
        fprintf(stderr, "UTF-8 input is too large\n");
        return -1;
    }
    storage = (char *)malloc((size_t)length + 1);
    if (!storage) {
        fclose(file);
        fprintf(stderr, "out of memory reading UTF-8 input\n");
        return -1;
    }
    {
        int read_failed = length &&
            fread(storage, 1, (size_t)length, file) != (size_t)length;
        int close_failed = fclose(file) != 0;
        if (read_failed || close_failed) {
            free(storage);
            fprintf(stderr, "cannot read UTF-8 input %s\n", path);
            return -1;
        }
    }
    storage[(size_t)length] = '\0';
    *text = storage;
    *text_bytes = (size_t)length;
    return 0;
}

static int read_tool_declarations(const char *path, char **rendered,
                                  size_t *rendered_bytes) {
    char *json = NULL;
    size_t json_bytes = 0;
    char error[COLI_GEMMA4_TOOLS_ERROR_MAX];
    int result;
    if (read_utf8_text(path, &json, &json_bytes) != 0) return -1;
    result = coli_gemma4_render_tools(
        json, json_bytes, rendered, rendered_bytes, error, sizeof(error));
    free(json);
    if (result != 0) {
        fprintf(stderr, "invalid tools file %s: %s\n", path, error);
        return -1;
    }
    return 0;
}

static int parse_f32(const char *text, float *result) {
    char *end = NULL;
    float value;
    errno = 0;
    value = strtof(text, &end);
    if (errno || !end || *end || !isfinite(value)) return -1;
    *result = value;
    return 0;
}

static int command_tokenize(int argc, char **argv, const char *model_path,
                            const char *text, size_t text_bytes, int user_chat,
                            int image_chat) {
    coli_gemma4_gguf gguf;
    coli_gemma4_tokenizer tokenizer;
    const char *system_text = NULL, *system_path = NULL;
    const char *tools_path = NULL;
    char *system_storage = NULL, *rendered_tools = NULL;
    uint32_t *tokens = NULL, *suffix_tokens = NULL;
    size_t system_bytes = 0, rendered_tools_bytes = 0, total_bytes;
    size_t capacity, count = 0, suffix_count = 0, index;
    int add_bos = 1, status = 1, argument;
    memset(&gguf, 0, sizeof(gguf));
    memset(&tokenizer, 0, sizeof(tokenizer));
    for (argument = 0; argument < argc; ++argument) {
        if (!user_chat && strcmp(argv[argument], "--no-bos") == 0) add_bos = 0;
        else if (user_chat &&
                 (strcmp(argv[argument], "--system") == 0 ||
                  strcmp(argv[argument], "--system-file") == 0 ||
                  strcmp(argv[argument], "--tools-file") == 0)) {
            if (++argument >= argc) {
                fprintf(stderr, "missing chat option value\n");
                return 2;
            }
        }
        else {
            fprintf(stderr, "unknown tokenize option: %s\n", argv[argument]);
            return 2;
        }
    }
    if (user_chat &&
        (option_value(argc, argv, "--system", &system_text) < 0 ||
         option_value(argc, argv, "--system-file", &system_path) < 0 ||
         option_value(argc, argv, "--tools-file", &tools_path) < 0 ||
         (system_text && system_path))) {
        fprintf(stderr, "use exactly one of --system and --system-file\n");
        return 2;
    }
    if (system_path) {
        if (read_utf8_text(system_path, &system_storage, &system_bytes) != 0)
            return 1;
        system_text = system_storage;
    } else if (system_text) {
        system_bytes = strlen(system_text);
    }
    if (tools_path && read_tool_declarations(
            tools_path, &rendered_tools, &rendered_tools_bytes) != 0) {
        free(system_storage);
        return 1;
    }
    if (system_bytes > SIZE_MAX - text_bytes ||
        rendered_tools_bytes > SIZE_MAX - text_bytes - system_bytes) {
        fprintf(stderr, "text is too large to tokenize\n");
        free(system_storage);
        free(rendered_tools);
        return 1;
    }
    total_bytes = text_bytes + system_bytes + rendered_tools_bytes;
    if (total_bytes > (SIZE_MAX - 256) / 3) {
        fprintf(stderr, "text is too large to tokenize\n");
        free(system_storage);
        free(rendered_tools);
        return 1;
    }
    capacity = total_bytes * 3 + 256;
    if (capacity > SIZE_MAX / sizeof(*tokens)) {
        fprintf(stderr, "text is too large to tokenize\n");
        free(system_storage);
        free(rendered_tools);
        return 1;
    }
    tokens = (uint32_t *)malloc(capacity * sizeof(*tokens));
    if (image_chat)
        suffix_tokens = (uint32_t *)malloc(capacity * sizeof(*suffix_tokens));
    if (!tokens || (image_chat && !suffix_tokens)) {
        fprintf(stderr, "out of memory allocating token buffer\n");
        free(system_storage);
        free(rendered_tools);
        free(tokens);
        free(suffix_tokens);
        return 1;
    }
    if (coli_gemma4_gguf_open(&gguf, model_path) != 0) {
        fprintf(stderr, "%s\n", coli_gemma4_gguf_last_error(&gguf));
        goto cleanup;
    }
    if (coli_gemma4_tokenizer_init(&tokenizer, &gguf) != 0) {
        fprintf(stderr, "%s\n", coli_gemma4_tokenizer_last_error(&tokenizer));
        goto cleanup;
    }
    if ((image_chat ?
         coli_gemma4_tokenize_image_chat_turn_with_tools(
             &tokenizer, system_text, system_bytes,
             rendered_tools, rendered_tools_bytes,
             text, text_bytes, 1, tokens, capacity, &count,
             suffix_tokens, capacity, &suffix_count) : user_chat ?
         coli_gemma4_tokenize_chat_turn_with_tools(
             &tokenizer, system_text, system_bytes,
             rendered_tools, rendered_tools_bytes,
             text, text_bytes, 1, tokens, capacity, &count) :
         coli_gemma4_tokenize(&tokenizer, text, text_bytes, add_bos,
                              tokens, capacity, &count)) != 0) {
        fprintf(stderr, "Gemma 4 tokenization failed\n");
        goto cleanup;
    }
    if (image_chat) {
        printf("prefix tokens: %zu\nprefix ids:", count);
        for (index = 0; index < count; ++index) printf(" %u", tokens[index]);
        printf("\nimage embeddings: insert N x 2816 float32 values here (non-causal chunk)\n");
        printf("suffix tokens: %zu\nsuffix ids:", suffix_count);
        for (index = 0; index < suffix_count; ++index)
            printf(" %u", suffix_tokens[index]);
        putchar('\n');
    } else {
        printf("tokens: %zu\nids:", count);
        for (index = 0; index < count; ++index) printf(" %u", tokens[index]);
        putchar('\n');
    }
    status = 0;
cleanup:
    coli_gemma4_tokenizer_close(&tokenizer);
    coli_gemma4_gguf_close(&gguf);
    free(system_storage);
    free(rendered_tools);
    free(tokens);
    free(suffix_tokens);
    return status;
}

static int command_tokenize_file(int argc, char **argv,
                                 const char *model_path, const char *path,
                                 int user_chat) {
    FILE *file = fopen(path, "rb");
    char *text = NULL;
    long length;
    int status;
    if (!file || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file) fclose(file);
        fprintf(stderr, "cannot read UTF-8 input %s\n", path);
        return 1;
    }
    text = (char *)malloc((size_t)length + 1);
    if (!text) {
        fclose(file);
        fprintf(stderr, "out of memory reading UTF-8 input %s\n", path);
        return 1;
    }
    {
        int read_failed = length &&
            fread(text, 1, (size_t)length, file) != (size_t)length;
        int close_failed = fclose(file) != 0;
        if (read_failed || close_failed) {
            free(text);
            fprintf(stderr, "cannot read UTF-8 input %s\n", path);
            return 1;
        }
    }
    text[length] = '\0';
    status = command_tokenize(argc, argv, model_path, text, (size_t)length,
                              user_chat, 0);
    free(text);
    return status;
}

static int command_embedding(int argc, char **argv, const char *model_path,
                             const char *token_text) {
    coli_gemma4_gguf gguf;
    coli_gemma4_model_io io;
    const char *output_path = NULL;
    uint32_t token;
    float *embedding = NULL;
    int status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&io, 0, sizeof(io));
    if (parse_u32(token_text, &token) != 0 ||
        option_value(argc, argv, "--output-f32", &output_path) < 0) {
        fprintf(stderr, "invalid token or output option\n");
        return 2;
    }
    if (coli_gemma4_gguf_open(&gguf, model_path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    if (coli_gemma4_model_io_open(&io, &gguf) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_model_io_last_error(&io));
        goto cleanup;
    }
    embedding = (float *)malloc((size_t)io.model_width * sizeof(float));
    if (!embedding || coli_gemma4_model_embed(&io, token, embedding) != 0) {
        fprintf(stderr, "error: cannot decode embedding for token %u\n", token);
        goto cleanup;
    }
    if (output_path && write_f32(output_path, embedding, io.model_width) != 0)
        goto cleanup;
    printf("token:             %u\n", token);
    printf("embedding width:   %u\n", io.model_width);
    printf("embedding scale:   %.9g\n", io.embedding_scale);
    printf("embedding hash:    0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(embedding, io.model_width));
    if (output_path) printf("wrote embedding:   %s\n", output_path);
    status = 0;

cleanup:
    free(embedding);
    coli_gemma4_model_io_close(&io);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_lm_head(int argc, char **argv, const char *model_path) {
    coli_gemma4_gguf gguf;
    coli_gemma4_model_io io;
    const char *input_path = NULL, *normalized_path = NULL;
    const char *logits_path = NULL, *top_text = NULL;
    uint32_t top_count = 5, rank;
    float *input = NULL, *normalized = NULL, *logits = NULL;
    uint8_t *selected = NULL;
    int status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&io, 0, sizeof(io));
    if (option_value(argc, argv, "--input-f32", &input_path) <= 0 ||
        option_value(argc, argv, "--normalized-f32", &normalized_path) < 0 ||
        option_value(argc, argv, "--logits-f32", &logits_path) < 0 ||
        option_value(argc, argv, "--top", &top_text) < 0 ||
        (top_text && parse_u32(top_text, &top_count) != 0) ||
        !top_count || top_count > 100) {
        fprintf(stderr, "lm-head requires --input-f32 and --top must be 1..100\n");
        return 2;
    }
    if (coli_gemma4_gguf_open(&gguf, model_path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    if (coli_gemma4_model_io_open(&io, &gguf) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_model_io_last_error(&io));
        goto cleanup;
    }
    input = (float *)malloc((size_t)io.model_width * sizeof(float));
    normalized = (float *)malloc((size_t)io.model_width * sizeof(float));
    logits = (float *)malloc((size_t)io.vocab_size * sizeof(float));
    selected = (uint8_t *)calloc(io.vocab_size, 1);
    if (!input || !normalized || !logits || !selected) {
        fprintf(stderr, "error: out of memory allocating LM-head buffers\n");
        goto cleanup;
    }
    if (read_f32(input_path, input, io.model_width) != 0 ||
        coli_gemma4_model_logits(&io, input, normalized, logits) != 0) {
        fprintf(stderr, "error: Gemma 4 LM head failed\n");
        goto cleanup;
    }
    if ((normalized_path && write_f32(normalized_path, normalized,
                                      io.model_width) != 0) ||
        (logits_path && write_f32(logits_path, logits, io.vocab_size) != 0))
        goto cleanup;
    printf("vocabulary:        %u\n", io.vocab_size);
    printf("logit softcap:     %.9g\n", io.logit_softcap);
    printf("top tokens:\n");
    for (rank = 0; rank < top_count; ++rank) {
        uint32_t token, best = UINT32_MAX;
        float best_value = -INFINITY;
        for (token = 0; token < io.vocab_size; ++token) {
            if (!selected[token] && isfinite(logits[token]) &&
                (best == UINT32_MAX || logits[token] > best_value)) {
                best = token;
                best_value = logits[token];
            }
        }
        if (best == UINT32_MAX) break;
        selected[best] = 1;
        printf("  %u: token=%u logit=%.9g\n", rank + 1, best, best_value);
    }
    if (normalized_path) printf("wrote normalized:  %s\n", normalized_path);
    if (logits_path) printf("wrote logits:      %s\n", logits_path);
    status = 0;

cleanup:
    free(input); free(normalized); free(logits); free(selected);
    coli_gemma4_model_io_close(&io);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int emit_token_piece(const coli_gemma4_tokenizer *tokenizer,
                            uint32_t token, int show_special) {
    size_t piece_bytes = 0;
    char *piece;
    if (coli_gemma4_decode_token(tokenizer, token, NULL, 0,
                                 &piece_bytes) != 0 ||
        piece_bytes == SIZE_MAX) return -1;
    piece = (char *)malloc(piece_bytes + 1);
    if (!piece || coli_gemma4_decode_token(
            tokenizer, token, piece, piece_bytes + 1, &piece_bytes) != 0) {
        free(piece);
        return -1;
    }
    if (show_special || !coli_gemma4_token_is_control(tokenizer, token))
        fwrite(piece, 1, piece_bytes, stdout);
    free(piece);
    return ferror(stdout) ? -1 : 0;
}

static int append_dynamic_text(char **buffer, size_t *length,
                               size_t *capacity, const char *text,
                               size_t text_bytes) {
    size_t needed, grown_capacity;
    char *grown;
    if (text_bytes > SIZE_MAX - *length - 1) return -1;
    needed = *length + text_bytes + 1;
    if (needed > *capacity) {
        grown_capacity = *capacity ? *capacity : 256;
        while (grown_capacity < needed) {
            if (grown_capacity > SIZE_MAX / 2) {
                grown_capacity = needed;
                break;
            }
            grown_capacity *= 2;
        }
        grown = (char *)realloc(*buffer, grown_capacity);
        if (!grown) return -1;
        *buffer = grown;
        *capacity = grown_capacity;
    }
    if (text_bytes) memcpy(*buffer + *length, text, text_bytes);
    *length += text_bytes;
    (*buffer)[*length] = '\0';
    return 0;
}

static int capture_token_piece(const coli_gemma4_tokenizer *tokenizer,
                               uint32_t token, char **buffer,
                               size_t *length, size_t *capacity) {
    size_t piece_bytes = 0, needed;
    char *grown;
    if (coli_gemma4_decode_token(tokenizer, token, NULL, 0,
                                 &piece_bytes) != 0 ||
        piece_bytes > SIZE_MAX - *length - 1) return -1;
    needed = *length + piece_bytes + 1;
    if (needed > *capacity) {
        size_t grown_capacity = *capacity ? *capacity : 256;
        while (grown_capacity < needed) {
            if (grown_capacity > SIZE_MAX / 2) {
                grown_capacity = needed;
                break;
            }
            grown_capacity *= 2;
        }
        grown = (char *)realloc(*buffer, grown_capacity);
        if (!grown) return -1;
        *buffer = grown;
        *capacity = grown_capacity;
    }
    if (coli_gemma4_decode_token(
            tokenizer, token, *buffer + *length, *capacity - *length,
            &piece_bytes) != 0) return -1;
    *length += piece_bytes;
    (*buffer)[*length] = '\0';
    return 0;
}

static void report_expert_cache(const coli_gemma4_backend *backend) {
    coli_gemma4_cache_stats stats;
    coli_gemma4_cache_get_stats(backend, &stats);
    if (!stats.capacity) return;
    fprintf(stderr,
            "expert cache: slots=%u resident=%u pins=%u hits=%llu "
            "pin-hits=%llu misses=%llu preloads=%llu prefetch=%llu/%llu "
            "lookahead=%llu/%llu match=%llu/%llu "
            "evictions=%llu promotions=%llu "
            "loaded=%.1f MiB\n",
            stats.capacity, stats.resident, stats.pinned_capacity,
            (unsigned long long)stats.hits,
            (unsigned long long)stats.pinned_hits,
            (unsigned long long)stats.misses,
            (unsigned long long)stats.preloads,
            (unsigned long long)stats.prefetch_launches,
            (unsigned long long)stats.prefetched_records,
            (unsigned long long)stats.lookahead_launches,
            (unsigned long long)stats.lookahead_records,
            (unsigned long long)stats.lookahead_matches,
            (unsigned long long)stats.lookahead_selected,
            (unsigned long long)stats.evictions,
            (unsigned long long)stats.promotions,
            (double)stats.bytes_loaded / (1024.0 * 1024.0));
}

static int persist_expert_usage(coli_gemma4_backend *backend,
                                const char *path, uint64_t *selections) {
    if (coli_gemma4_usage_save(backend, path, selections) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(backend));
        return -1;
    }
    return 0;
}

static int command_next_token(int argc, char **argv, const char *model_path,
                              const char *packed_dir, const char *text,
                              size_t text_bytes, int generate) {
    coli_gemma4_gguf gguf;
    coli_gemma4_tokenizer tokenizer;
    coli_gemma4_backend gemma;
    coli_expert_backend experts;
    coli_gemma4_model model;
    coli_gemma4_sampler sampler;
    coli_gemma4_vision vision;
    coli_gemma4_vision_image prepared_image;
    const char *top_text = NULL, *max_new_text = NULL;
    const char *temperature_text = NULL, *sample_top_k_text = NULL;
    const char *sample_top_p_text = NULL, *seed_text = NULL;
    const char *expert_cache_text = NULL, *expert_pins_text = NULL;
    const char *expert_usage_path = NULL;
    const char *cuda_device_text = NULL;
    const char *system_text = NULL, *system_path = NULL;
    const char *tools_path = NULL;
    const char *image_path = NULL, *mmproj_path = NULL;
    const char *image_embeddings_path = NULL;
    const char *residual_path = NULL, *logits_path = NULL;
    char *system_storage = NULL, *rendered_tools = NULL;
    uint32_t *tokens = NULL, *suffix_tokens = NULL;
    uint32_t top_count = 5, max_new = generate ? 16U : 0U;
    uint32_t position, rank, maximum_tokens, sample_top_k = 0;
    uint32_t image_token_count = 0, source_width = 0, source_height = 0;
    uint32_t expert_cache = 0, expert_pins = 0;
    uint32_t cuda_device = 0;
    uint64_t seed = 1;
    uint64_t usage_total = 0;
    float temperature = 0.0F, sample_top_p = 1.0F;
    uint8_t *selected = NULL, *source_rgb = NULL;
    float *residual = NULL, *logits = NULL, *image_embeddings = NULL;
    size_t capacity, count = 0, system_bytes = 0;
    size_t suffix_count = 0, rendered_tools_bytes = 0, total_bytes;
    uint32_t prompt_positions = 0;
    char image_error[COLI_GEMMA4_VISION_ERROR_MAX];
    int chat = 0, show_special = 0, expert_prefetch = 0;
    int expert_lookahead = 0;
    int argument, status = 1, usage_active = 0;
    memset(&gguf, 0, sizeof(gguf));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(&gemma, 0, sizeof(gemma));
    memset(&model, 0, sizeof(model));
    memset(&sampler, 0, sizeof(sampler));
    memset(&vision, 0, sizeof(vision));
    memset(&prepared_image, 0, sizeof(prepared_image));
    for (argument = 0; argument < argc; ++argument)
        if (strcmp(argv[argument], "--chat") == 0) chat = 1;
        else if (strcmp(argv[argument], "--show-special") == 0) show_special = 1;
        else if (strcmp(argv[argument], "--expert-prefetch") == 0)
            expert_prefetch = 1;
        else if (strcmp(argv[argument], "--expert-lookahead") == 0)
            expert_lookahead = 1;
    if (option_value(argc, argv, "--top", &top_text) < 0 ||
        option_value(argc, argv, "--max-new", &max_new_text) < 0 ||
        option_value(argc, argv, "--temperature", &temperature_text) < 0 ||
        option_value(argc, argv, "--top-k", &sample_top_k_text) < 0 ||
        option_value(argc, argv, "--top-p", &sample_top_p_text) < 0 ||
        option_value(argc, argv, "--seed", &seed_text) < 0 ||
        option_value(argc, argv, "--expert-cache", &expert_cache_text) < 0 ||
        option_value(argc, argv, "--expert-pins", &expert_pins_text) < 0 ||
        option_value(argc, argv, "--expert-usage", &expert_usage_path) < 0 ||
        option_value(argc, argv, "--cuda-device", &cuda_device_text) < 0 ||
        option_value(argc, argv, "--system", &system_text) < 0 ||
        option_value(argc, argv, "--system-file", &system_path) < 0 ||
        option_value(argc, argv, "--tools-file", &tools_path) < 0 ||
        option_value(argc, argv, "--image", &image_path) < 0 ||
        option_value(argc, argv, "--mmproj", &mmproj_path) < 0 ||
        option_value(argc, argv, "--image-embeddings-f32",
                     &image_embeddings_path) < 0 ||
        option_value(argc, argv, "--residual-f32", &residual_path) < 0 ||
        option_value(argc, argv, "--logits-f32", &logits_path) < 0 ||
        (top_text && parse_u32(top_text, &top_count) != 0) ||
        (max_new_text && parse_u32(max_new_text, &max_new) != 0) ||
        (temperature_text && parse_f32(temperature_text, &temperature) != 0) ||
        (sample_top_k_text && parse_u32(sample_top_k_text, &sample_top_k) != 0) ||
        (sample_top_p_text && parse_f32(sample_top_p_text, &sample_top_p) != 0) ||
        (seed_text && parse_u64(seed_text, &seed) != 0) ||
        (expert_cache_text && parse_u32(expert_cache_text, &expert_cache) != 0) ||
        (expert_pins_text && parse_u32(expert_pins_text, &expert_pins) != 0) ||
        (cuda_device_text && parse_u32(cuda_device_text, &cuda_device) != 0) ||
        (expert_usage_path && !expert_cache) ||
        (expert_prefetch && !expert_cache) ||
        (expert_lookahead && !expert_prefetch) ||
        (expert_pins && expert_pins >= expert_cache) ||
        ((system_text || system_path || tools_path) && !chat) ||
        ((image_path || mmproj_path) && (!image_path || !mmproj_path || !chat)) ||
        (image_embeddings_path && !image_path) ||
        (system_text && system_path) ||
        !top_count || top_count > 100 || (generate && !max_new) ||
        (!generate && (max_new_text || temperature_text || sample_top_k_text ||
                       sample_top_p_text || seed_text))) {
        fprintf(stderr, "invalid generation or ranking option\n");
        return 2;
    }
    if (system_path) {
        if (read_utf8_text(system_path, &system_storage, &system_bytes) != 0)
            return 1;
        system_text = system_storage;
    } else if (system_text) {
        system_bytes = strlen(system_text);
    }
    if (tools_path && read_tool_declarations(
            tools_path, &rendered_tools, &rendered_tools_bytes) != 0) {
        free(system_storage);
        return 1;
    }
    if (system_bytes > SIZE_MAX - text_bytes ||
        rendered_tools_bytes > SIZE_MAX - text_bytes - system_bytes) {
        fprintf(stderr, "prompt is too large to tokenize\n");
        free(system_storage);
        free(rendered_tools);
        return 2;
    }
    total_bytes = text_bytes + system_bytes + rendered_tools_bytes;
    if (total_bytes > (SIZE_MAX - 256) / 3) {
        fprintf(stderr, "prompt is too large to tokenize\n");
        free(system_storage);
        free(rendered_tools);
        return 2;
    }
    capacity = total_bytes * 3 + 256;
    if (capacity > SIZE_MAX / sizeof(*tokens)) {
        free(system_storage);
        free(rendered_tools);
        return 2;
    }
    tokens = (uint32_t *)malloc(capacity * sizeof(*tokens));
    if (image_path)
        suffix_tokens = (uint32_t *)malloc(capacity * sizeof(*suffix_tokens));
    if (!tokens || (image_path && !suffix_tokens)) {
        free(system_storage);
        free(rendered_tools);
        return 1;
    }
    if (coli_gemma4_gguf_open(&gguf, model_path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        goto cleanup;
    }
    if (coli_gemma4_tokenizer_init(&tokenizer, &gguf) != 0) {
        fprintf(stderr, "error: %s\n",
                coli_gemma4_tokenizer_last_error(&tokenizer));
        goto cleanup;
    }
    if (!sample_top_k_text) sample_top_k = gguf.sampling_top_k;
    if (!sample_top_p_text) sample_top_p = gguf.sampling_top_p;
    if (coli_gemma4_sampler_init(&sampler, temperature, sample_top_k,
                                 sample_top_p, seed) != 0) {
        fprintf(stderr, "error: invalid sampling configuration\n");
        goto cleanup;
    }
    if ((image_path ? coli_gemma4_tokenize_image_chat_turn_with_tools(
                    &tokenizer, system_text, system_bytes,
                    rendered_tools, rendered_tools_bytes,
                    text, text_bytes, 1, tokens, capacity, &count,
                    suffix_tokens, capacity, &suffix_count) :
                chat ? coli_gemma4_tokenize_chat_turn_with_tools(
                    &tokenizer, system_text, system_bytes,
                    rendered_tools, rendered_tools_bytes,
                    text, text_bytes, 1, tokens, capacity, &count) :
                coli_gemma4_tokenize(&tokenizer, text, text_bytes, 1,
                                     tokens, capacity, &count)) != 0 ||
        !count || count > UINT32_MAX) {
        fprintf(stderr, "error: prompt tokenization failed\n");
        goto cleanup;
    }
    if (image_path) {
        if (coli_gemma4_vision_load_image(
                image_path, &source_rgb, &source_width, &source_height,
                image_error, sizeof(image_error)) != 0) {
            fprintf(stderr, "error: %s\n", image_error);
            goto cleanup;
        }
        if (coli_gemma4_vision_open(&vision, mmproj_path) != 0) {
            fprintf(stderr, "error: %s\n",
                    coli_gemma4_vision_last_error(&vision));
            goto cleanup;
        }
        if (coli_gemma4_vision_prepare_rgb(
                &vision, source_rgb, source_width, source_height,
                &prepared_image) != 0 ||
            coli_gemma4_vision_encode(
                &vision, &prepared_image, &image_embeddings,
                &image_token_count) != 0) {
            fprintf(stderr, "error: %s\n",
                    coli_gemma4_vision_last_error(&vision));
            goto cleanup;
        }
        if (image_embeddings_path && read_f32(
                image_embeddings_path, image_embeddings,
                (size_t)image_token_count * gguf.config.n_embd) != 0)
            goto cleanup;
    }
    if (count > UINT32_MAX - image_token_count ||
        suffix_count > UINT32_MAX - count - image_token_count) {
        fprintf(stderr, "error: image prompt is too large\n");
        goto cleanup;
    }
    prompt_positions = (uint32_t)count + image_token_count +
                       (uint32_t)suffix_count;
    if (prompt_positions > UINT32_MAX - max_new) {
        fprintf(stderr, "error: prompt plus generation length is too large\n");
        goto cleanup;
    }
    maximum_tokens = prompt_positions + max_new;
    if (coli_gemma4_open_packed(&gemma, packed_dir) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (strlen(gguf.path) + 1 > sizeof(gemma.source)) {
        fprintf(stderr, "error: model path is too long for expert backend\n");
        goto cleanup;
    }
    memcpy(gemma.source, gguf.path, strlen(gguf.path) + 1);
    if (coli_gemma4_cache_configure(&gemma, expert_cache) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (expert_pins &&
        coli_gemma4_cache_set_pinned(&gemma, expert_pins) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (expert_usage_path) {
        if (coli_gemma4_usage_load(&gemma, expert_usage_path,
                                   &usage_total) != 0) {
            fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
            goto cleanup;
        }
        usage_active = 1;
        fprintf(stderr, "expert usage: loaded=%llu path=%s\n",
                (unsigned long long)usage_total, expert_usage_path);
    }
    if (expert_prefetch &&
        coli_gemma4_prefetch_configure(&gemma, 1) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (expert_lookahead &&
        coli_gemma4_lookahead_configure(&gemma, 1) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (cuda_device_text && coli_gemma4_cuda_configure(
            &gemma, (int)cuda_device) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (gemma.config.n_layer != gguf.config.n_layer ||
        gemma.config.n_embd != gguf.config.n_embd ||
        gemma.config.n_expert != gguf.config.n_expert ||
        gemma.config.n_expert_used != gguf.config.n_expert_used) {
        fprintf(stderr, "error: packed manifest and GGUF configuration disagree\n");
        goto cleanup;
    }
    residual = (float *)malloc((size_t)gguf.config.n_embd * sizeof(float));
    logits = (float *)malloc((size_t)gguf.config.n_vocab * sizeof(float));
    selected = (uint8_t *)calloc(gguf.config.n_vocab, 1);
    if (!residual || !logits || !selected) {
        fprintf(stderr, "error: out of memory allocating full-model output\n");
        goto cleanup;
    }
    experts = coli_gemma4_expert_backend(&gemma);
    if (coli_gemma4_model_open(&model, &gguf, &experts,
                               maximum_tokens) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_model_last_error(&model));
        goto cleanup;
    }
    printf("prompt tokens:     %zu\n", count + suffix_count);
    if (image_path)
        printf("image:             %ux%u -> %u embeddings\n",
               source_width, source_height, image_token_count);
    for (position = 0; position < (uint32_t)count; ++position) {
        int last = !image_path && position + 1 == (uint32_t)count;
        if (coli_gemma4_model_step(&model, tokens[position], position,
                                   last ? residual : NULL,
                                   last ? logits : NULL) != 0) {
            fprintf(stderr, "error: %s\n", coli_gemma4_model_last_error(&model));
            goto cleanup;
        }
        printf("evaluated:         %u/%u (token %u)\n",
               position + 1, prompt_positions, tokens[position]);
    }
    if (image_path) {
        size_t suffix_index;
        if (coli_gemma4_model_image_embeddings(
                &model, image_embeddings, image_token_count,
                (uint64_t)count, NULL) != 0) {
            fprintf(stderr, "error: %s\n", coli_gemma4_model_last_error(&model));
            goto cleanup;
        }
        printf("evaluated:         %u/%u (image embeddings)\n",
               (uint32_t)count + image_token_count, prompt_positions);
        for (suffix_index = 0; suffix_index < suffix_count; ++suffix_index) {
            uint32_t suffix_position = (uint32_t)count + image_token_count +
                                       (uint32_t)suffix_index;
            int last = suffix_index + 1 == suffix_count;
            if (coli_gemma4_model_step(
                    &model, suffix_tokens[suffix_index], suffix_position,
                    last ? residual : NULL, last ? logits : NULL) != 0) {
                fprintf(stderr, "error: %s\n",
                        coli_gemma4_model_last_error(&model));
                goto cleanup;
            }
            printf("evaluated:         %u/%u (token %u)\n",
                   suffix_position + 1, prompt_positions,
                   suffix_tokens[suffix_index]);
        }
    }
    if ((residual_path && write_f32(residual_path, residual,
                                    gguf.config.n_embd) != 0) ||
        (logits_path && write_f32(logits_path, logits,
                                  gguf.config.n_vocab) != 0)) goto cleanup;
    if (generate) {
        uint32_t generated;
        printf("sampling:          temperature=%.7g top-k=%u top-p=%.7g seed=%llu\n",
               temperature, sample_top_k, sample_top_p,
               (unsigned long long)seed);
        printf("generated:\n");
        for (generated = 0; generated < max_new; ++generated) {
            uint32_t best;
            if (coli_gemma4_sample(&sampler, logits, gguf.config.n_vocab,
                                   &best, NULL) != 0) {
                fprintf(stderr, "error: sampling failed\n");
                goto cleanup;
            }
            if (emit_token_piece(&tokenizer, best, show_special) != 0)
                goto cleanup;
            fflush(stdout);
            if (coli_gemma4_token_is_eog(&tokenizer, best)) break;
            if (generated + 1 < max_new &&
                coli_gemma4_model_step(&model, best,
                                       (uint64_t)prompt_positions + generated,
                                       NULL, logits) != 0) {
                fprintf(stderr, "error: %s\n",
                        coli_gemma4_model_last_error(&model));
                goto cleanup;
            }
        }
        fputc('\n', stdout);
    } else {
        printf("top tokens:\n");
        for (rank = 0; rank < top_count; ++rank) {
            uint32_t token, best = UINT32_MAX;
            float best_value = -INFINITY;
            for (token = 0; token < gguf.config.n_vocab; ++token) {
                if (!selected[token] && isfinite(logits[token]) &&
                    (best == UINT32_MAX || logits[token] > best_value)) {
                    best = token;
                    best_value = logits[token];
                }
            }
            if (best == UINT32_MAX) break;
            selected[best] = 1;
            printf("  %u: token=%u logit=%.9g\n", rank + 1, best, best_value);
        }
    }
    if (residual_path) printf("wrote residual:    %s\n", residual_path);
    if (logits_path) printf("wrote logits:      %s\n", logits_path);
    status = 0;

cleanup:
    if (usage_active) {
        if (persist_expert_usage(&gemma, expert_usage_path,
                                 &usage_total) != 0) status = 1;
        else fprintf(stderr, "expert usage: saved=%llu path=%s\n",
                     (unsigned long long)usage_total, expert_usage_path);
    }
    report_expert_cache(&gemma);
    free(system_storage);
    free(rendered_tools);
    free(tokens); free(suffix_tokens); free(residual); free(logits);
    free(selected); free(source_rgb); free(image_embeddings);
    coli_gemma4_vision_image_close(&prepared_image);
    coli_gemma4_vision_close(&vision);
    coli_gemma4_model_close(&model);
    coli_gemma4_close(&gemma);
    coli_gemma4_tokenizer_close(&tokenizer);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_next_token_file(int argc, char **argv,
                                   const char *model_path,
                                   const char *packed_dir,
                                   const char *path, int generate) {
    FILE *file = fopen(path, "rb");
    char *text = NULL;
    long length;
    int status;
    if (!file || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file) fclose(file);
        fprintf(stderr, "cannot read UTF-8 input %s\n", path);
        return 1;
    }
    if ((uint64_t)length > SIZE_MAX - 1) {
        fclose(file);
        fprintf(stderr, "UTF-8 input is too large\n");
        return 1;
    }
    text = (char *)malloc((size_t)length + 1);
    if (!text) {
        fclose(file);
        return 1;
    }
    {
        int read_failed = length &&
            fread(text, 1, (size_t)length, file) != (size_t)length;
        int close_failed = fclose(file) != 0;
        if (read_failed || close_failed) {
            free(text);
            fprintf(stderr, "cannot read UTF-8 input %s\n", path);
            return 1;
        }
    }
    text[(size_t)length] = '\0';
    status = command_next_token(argc, argv, model_path, packed_dir,
                                text, (size_t)length, generate);
    free(text);
    return status;
}

static int inject_chat_tool_responses(
    const coli_gemma4_tokenizer *tokenizer, coli_gemma4_model *model,
    const char *generated, size_t generated_bytes,
    uint32_t *tokens, uint32_t max_context, uint32_t *position,
    float *logits, char *line, size_t line_capacity, int show_special,
    int *exit_requested) {
    enum { MAX_TOOL_CALLS = 16 };
    coli_gemma4_tool_call calls[MAX_TOOL_CALLS];
    char error[COLI_GEMMA4_TOOLS_ERROR_MAX];
    char *response_prompt = NULL;
    size_t response_bytes = 0, response_capacity = 0;
    size_t call_count = 0, call_index, response_token_count = 0, token_index;
    int result = -1;
    *exit_requested = 0;
    if (coli_gemma4_parse_tool_calls(
            generated, generated_bytes, calls, MAX_TOOL_CALLS, &call_count,
            error, sizeof(error)) != 0) {
        fprintf(stderr, "error: cannot parse Gemma tool call: %s\n", error);
        return -1;
    }
    for (call_index = 0; call_index < call_count; ++call_index) {
        char *rendered = NULL;
        size_t rendered_bytes = 0;
        for (;;) {
            size_t line_bytes;
            printf("\ntool %s arguments: ", calls[call_index].name);
            fwrite(generated + calls[call_index].arguments_offset, 1,
                   calls[call_index].arguments_bytes, stdout);
            fputs("\nresult JSON> ", stdout);
            fflush(stdout);
            if (!fgets(line, (int)line_capacity, stdin)) {
                if (ferror(stdin)) fprintf(stderr, "error reading tool result\n");
                else *exit_requested = 1;
                goto cleanup;
            }
            line_bytes = strlen(line);
            if (line_bytes && line[line_bytes - 1] != '\n' && !feof(stdin)) {
                int character;
                while ((character = fgetc(stdin)) != '\n' && character != EOF) {}
                fprintf(stderr, "tool result exceeds %zu bytes\n",
                        line_capacity - 1);
                continue;
            }
            while (line_bytes &&
                   (line[line_bytes - 1] == '\r' ||
                    line[line_bytes - 1] == '\n'))
                line[--line_bytes] = '\0';
            if (!strcmp(line, "/exit")) {
                *exit_requested = 1;
                goto cleanup;
            }
            if (coli_gemma4_render_tool_response(
                    calls[call_index].name, line, line_bytes,
                    call_index != 0, &rendered, &rendered_bytes,
                    error, sizeof(error)) != 0) {
                fprintf(stderr, "invalid tool result: %s\n", error);
                continue;
            }
            break;
        }
        if (append_dynamic_text(
                &response_prompt, &response_bytes, &response_capacity,
                rendered, rendered_bytes) != 0) {
            free(rendered);
            fprintf(stderr, "error: out of memory collecting tool results\n");
            goto cleanup;
        }
        free(rendered);
    }
    if (!response_bytes || *position >= max_context ||
        coli_gemma4_tokenize_ex(
            tokenizer, response_prompt, response_bytes, 0, 1,
            tokens, max_context - *position, &response_token_count) != 0 ||
        !response_token_count ||
        response_token_count >= max_context - *position) {
        fprintf(stderr, "error: tool response exhausts the chat context\n");
        goto cleanup;
    }
    if (show_special) fwrite(response_prompt, 1, response_bytes, stdout);
    for (token_index = 0; token_index < response_token_count; ++token_index) {
        int last = token_index + 1 == response_token_count;
        if (coli_gemma4_model_step(
                model, tokens[token_index], (*position)++, NULL,
                last ? logits : NULL) != 0) {
            fprintf(stderr, "error: %s\n", coli_gemma4_model_last_error(model));
            goto cleanup;
        }
    }
    fputs("\ngemma> ", stdout);
    fflush(stdout);
    result = 0;
cleanup:
    free(response_prompt);
    return result;
}

static int command_chat(int argc, char **argv, const char *model_path,
                        const char *packed_dir) {
    coli_gemma4_gguf gguf;
    coli_gemma4_tokenizer tokenizer;
    coli_gemma4_backend gemma;
    coli_expert_backend experts;
    coli_gemma4_model model;
    coli_gemma4_sampler sampler;
    const char *max_context_text = NULL, *max_new_text = NULL;
    const char *temperature_text = NULL, *top_k_text = NULL;
    const char *top_p_text = NULL, *seed_text = NULL;
    const char *expert_cache_text = NULL, *expert_pins_text = NULL;
    const char *expert_usage_path = NULL;
    const char *cuda_device_text = NULL;
    const char *system_text = NULL, *system_path = NULL;
    const char *tools_path = NULL;
    char *system_storage = NULL, *rendered_tools = NULL;
    size_t system_bytes = 0, rendered_tools_bytes = 0;
    uint32_t max_context = 512, max_new = 64, top_k = 0;
    uint32_t expert_cache = 0, expert_pins = 0;
    uint32_t cuda_device = 0;
    uint32_t *tokens = NULL, position = 0;
    uint64_t seed = 1;
    uint64_t usage_total = 0;
    float temperature = 0.0F, top_p = 1.0F;
    float *logits = NULL;
    char *line = NULL, *generated_text = NULL;
    size_t generated_text_bytes = 0, generated_text_capacity = 0;
    int show_special = 0, expert_prefetch = 0, expert_lookahead = 0;
    int argument, status = 1, usage_active = 0;
    memset(&gguf, 0, sizeof(gguf));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(&gemma, 0, sizeof(gemma));
    memset(&model, 0, sizeof(model));
    memset(&sampler, 0, sizeof(sampler));
    for (argument = 0; argument < argc; ++argument)
        if (strcmp(argv[argument], "--show-special") == 0) show_special = 1;
        else if (strcmp(argv[argument], "--expert-prefetch") == 0)
            expert_prefetch = 1;
        else if (strcmp(argv[argument], "--expert-lookahead") == 0)
            expert_lookahead = 1;
    if (option_value(argc, argv, "--max-context", &max_context_text) < 0 ||
        option_value(argc, argv, "--max-new", &max_new_text) < 0 ||
        option_value(argc, argv, "--temperature", &temperature_text) < 0 ||
        option_value(argc, argv, "--top-k", &top_k_text) < 0 ||
        option_value(argc, argv, "--top-p", &top_p_text) < 0 ||
        option_value(argc, argv, "--seed", &seed_text) < 0 ||
        option_value(argc, argv, "--expert-cache", &expert_cache_text) < 0 ||
        option_value(argc, argv, "--expert-pins", &expert_pins_text) < 0 ||
        option_value(argc, argv, "--expert-usage", &expert_usage_path) < 0 ||
        option_value(argc, argv, "--cuda-device", &cuda_device_text) < 0 ||
        option_value(argc, argv, "--system", &system_text) < 0 ||
        option_value(argc, argv, "--system-file", &system_path) < 0 ||
        option_value(argc, argv, "--tools-file", &tools_path) < 0 ||
        (max_context_text && parse_u32(max_context_text, &max_context) != 0) ||
        (max_new_text && parse_u32(max_new_text, &max_new) != 0) ||
        (temperature_text && parse_f32(temperature_text, &temperature) != 0) ||
        (top_k_text && parse_u32(top_k_text, &top_k) != 0) ||
        (top_p_text && parse_f32(top_p_text, &top_p) != 0) ||
        (seed_text && parse_u64(seed_text, &seed) != 0) ||
        (expert_cache_text && parse_u32(expert_cache_text, &expert_cache) != 0) ||
        (expert_pins_text && parse_u32(expert_pins_text, &expert_pins) != 0) ||
        (cuda_device_text && parse_u32(cuda_device_text, &cuda_device) != 0) ||
        (expert_usage_path && !expert_cache) ||
        (expert_prefetch && !expert_cache) ||
        (expert_lookahead && !expert_prefetch) ||
        (expert_pins && expert_pins >= expert_cache) ||
        (system_text && system_path) ||
        !max_context || !max_new || max_new >= max_context) {
        fprintf(stderr, "invalid chat-session option\n");
        return 2;
    }
    if (system_path) {
        if (read_utf8_text(system_path, &system_storage, &system_bytes) != 0)
            return 1;
        system_text = system_storage;
    } else if (system_text) {
        system_bytes = strlen(system_text);
    }
    if (tools_path && read_tool_declarations(
            tools_path, &rendered_tools, &rendered_tools_bytes) != 0) {
        free(system_storage);
        return 1;
    }
    if (coli_gemma4_gguf_open(&gguf, model_path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        goto cleanup;
    }
    if (coli_gemma4_tokenizer_init(&tokenizer, &gguf) != 0) {
        fprintf(stderr, "error: %s\n",
                coli_gemma4_tokenizer_last_error(&tokenizer));
        goto cleanup;
    }
    if (!top_k_text) top_k = gguf.sampling_top_k;
    if (!top_p_text) top_p = gguf.sampling_top_p;
    if (coli_gemma4_sampler_init(&sampler, temperature, top_k,
                                 top_p, seed) != 0) {
        fprintf(stderr, "error: invalid sampling configuration\n");
        goto cleanup;
    }
    if (coli_gemma4_open_packed(&gemma, packed_dir) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (strlen(gguf.path) + 1 > sizeof(gemma.source)) {
        fprintf(stderr, "error: model path is too long for expert backend\n");
        goto cleanup;
    }
    memcpy(gemma.source, gguf.path, strlen(gguf.path) + 1);
    if (coli_gemma4_cache_configure(&gemma, expert_cache) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (expert_pins &&
        coli_gemma4_cache_set_pinned(&gemma, expert_pins) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (expert_usage_path) {
        if (coli_gemma4_usage_load(&gemma, expert_usage_path,
                                   &usage_total) != 0) {
            fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
            goto cleanup;
        }
        usage_active = 1;
        fprintf(stderr, "expert usage: loaded=%llu path=%s\n",
                (unsigned long long)usage_total, expert_usage_path);
    }
    if (expert_prefetch &&
        coli_gemma4_prefetch_configure(&gemma, 1) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (expert_lookahead &&
        coli_gemma4_lookahead_configure(&gemma, 1) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (cuda_device_text && coli_gemma4_cuda_configure(
            &gemma, (int)cuda_device) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (gemma.config.n_layer != gguf.config.n_layer ||
        gemma.config.n_embd != gguf.config.n_embd ||
        gemma.config.n_expert != gguf.config.n_expert ||
        gemma.config.n_expert_used != gguf.config.n_expert_used) {
        fprintf(stderr, "error: packed manifest and GGUF configuration disagree\n");
        goto cleanup;
    }
#if SIZE_MAX < UINT32_MAX
    if ((size_t)max_context > SIZE_MAX / sizeof(*tokens)) goto cleanup;
#endif
    tokens = (uint32_t *)malloc((size_t)max_context * sizeof(*tokens));
    logits = (float *)malloc((size_t)gguf.config.n_vocab * sizeof(float));
    line = (char *)malloc(65536);
    if (!tokens || !logits || !line) {
        fprintf(stderr, "error: out of memory allocating chat session\n");
        goto cleanup;
    }
    experts = coli_gemma4_expert_backend(&gemma);
    if (coli_gemma4_model_open(&model, &gguf, &experts, max_context) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_model_last_error(&model));
        goto cleanup;
    }
    printf("Gemma 4 chat: context=%u max-new=%u temperature=%.7g "
           "top-k=%u top-p=%.7g seed=%llu\n",
           max_context, max_new, temperature, top_k, top_p,
           (unsigned long long)seed);
    puts("Enter /exit to quit.");
    while (position < max_context) {
        size_t line_bytes, token_count = 0, index;
        int initial_turn = position == 0;
        fputs("you> ", stdout);
        fflush(stdout);
        if (!fgets(line, 65536, stdin)) {
            if (ferror(stdin)) fprintf(stderr, "error reading chat input\n");
            else status = 0;
            break;
        }
        line_bytes = strlen(line);
        if (line_bytes && line[line_bytes - 1] != '\n' && !feof(stdin)) {
            int character;
            while ((character = fgetc(stdin)) != '\n' && character != EOF) {}
            fprintf(stderr, "input line exceeds 65535 bytes\n");
            continue;
        }
        while (line_bytes &&
               (line[line_bytes - 1] == '\r' || line[line_bytes - 1] == '\n'))
            line[--line_bytes] = '\0';
        if (strcmp(line, "/exit") == 0) {
            status = 0;
            break;
        }
        if (!line_bytes) continue;
        if (coli_gemma4_tokenize_chat_turn_with_tools(
                &tokenizer, initial_turn ? system_text : NULL,
                initial_turn ? system_bytes : 0,
                initial_turn ? rendered_tools : NULL,
                initial_turn ? rendered_tools_bytes : 0,
                line, line_bytes, initial_turn, tokens,
                max_context - position, &token_count) != 0 || !token_count) {
            fprintf(stderr, "error: cannot tokenize user turn\n");
            goto cleanup;
        }
        if (token_count > max_context - position ||
            max_new + 1 > max_context - position - token_count) {
            fprintf(stderr, "context capacity exhausted; start a new session\n");
            status = 0;
            break;
        }
        for (index = 0; index < token_count; ++index) {
            int last = index + 1 == token_count;
            if (coli_gemma4_model_step(&model, tokens[index], position++,
                                       NULL, last ? logits : NULL) != 0) {
                fprintf(stderr, "error: %s\n",
                        coli_gemma4_model_last_error(&model));
                goto cleanup;
            }
        }
        generated_text_bytes = 0;
        if (generated_text) generated_text[0] = '\0';
        fputs("gemma> ", stdout);
        fflush(stdout);
        {
            uint32_t generated;
            int ended = 0;
            for (generated = 0; generated < max_new; ++generated) {
                uint32_t selected;
                int eog, tool_handoff;
                if (coli_gemma4_sample(&sampler, logits, gguf.config.n_vocab,
                                       &selected, NULL) != 0 ||
                    capture_token_piece(
                        &tokenizer, selected, &generated_text,
                        &generated_text_bytes, &generated_text_capacity) != 0 ||
                    emit_token_piece(&tokenizer, selected, show_special) != 0) {
                    fprintf(stderr, "error: chat sampling or decoding failed\n");
                    goto cleanup;
                }
                eog = coli_gemma4_token_is_eog(&tokenizer, selected);
                tool_handoff = coli_gemma4_token_is_tool_response(
                    &tokenizer, selected);
                if (position >= max_context ||
                    coli_gemma4_model_step(
                        &model, selected, position++, NULL,
                        !eog && generated + 1 < max_new ? logits : NULL) != 0) {
                    fprintf(stderr, "error: %s\n",
                            coli_gemma4_model_last_error(&model));
                    goto cleanup;
                }
                fflush(stdout);
                if (tool_handoff && rendered_tools_bytes) {
                    int exit_requested = 0;
                    int inject_result = inject_chat_tool_responses(
                        &tokenizer, &model,
                        generated_text, generated_text_bytes,
                        tokens, max_context, &position, logits,
                        line, 65536, show_special, &exit_requested);
                    if (exit_requested) {
                        fputc('\n', stdout);
                        status = 0;
                        goto cleanup;
                    }
                    if (inject_result != 0) goto cleanup;
                    generated_text_bytes = 0;
                    generated_text[0] = '\0';
                    continue;
                }
                if (eog) {
                    ended = 1;
                    break;
                }
            }
            if (!ended) {
                size_t close_count = 0;
                static const char close_turn[] = "<turn|>";
                if (position >= max_context ||
                    coli_gemma4_tokenize_ex(
                        &tokenizer, close_turn, sizeof(close_turn) - 1,
                        0, 1, tokens, max_context - position,
                        &close_count) != 0 || close_count != 1 ||
                    coli_gemma4_model_step(&model, tokens[0], position++,
                                           NULL, NULL) != 0) {
                    fprintf(stderr, "error: cannot close truncated model turn\n");
                    goto cleanup;
                }
                if (show_special) fputs("<turn|>", stdout);
            }
        }
        fputc('\n', stdout);
        if (usage_active && persist_expert_usage(
                &gemma, expert_usage_path, &usage_total) != 0) {
            usage_active = 0;
            goto cleanup;
        }
    }
    if (position >= max_context) status = 0;

cleanup:
    if (usage_active) {
        if (persist_expert_usage(&gemma, expert_usage_path,
                                 &usage_total) != 0) status = 1;
        else fprintf(stderr, "expert usage: saved=%llu path=%s\n",
                     (unsigned long long)usage_total, expert_usage_path);
    }
    report_expert_cache(&gemma);
    free(system_storage);
    free(rendered_tools);
    free(tokens); free(logits); free(line); free(generated_text);
    coli_gemma4_model_close(&model);
    coli_gemma4_close(&gemma);
    coli_gemma4_tokenizer_close(&tokenizer);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_model_info(const char *path) {
    coli_gemma4_gguf gguf;
    uint32_t layer;
    if (coli_gemma4_gguf_open(&gguf, path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    printf("architecture:       %s\n", gguf.architecture);
    printf("GGUF version:       %u\n", gguf.version);
    printf("tensor count:       %" PRIu64 "\n", gguf.tensor_count);
    printf("tensor data offset: %" PRIu64 "\n", gguf.data_offset);
    printf("layers/width/vocab: %u / %u / %u\n",
           gguf.config.n_layer, gguf.config.n_embd, gguf.config.n_vocab);
    printf("experts/top-k/ff:   %u / %u / %u\n",
           gguf.config.n_expert, gguf.config.n_expert_used,
           gguf.config.n_expert_ff);
    printf("attention heads:    %u\n", gguf.attention_heads);
    printf("sliding window:     %u\n", gguf.config.sliding_window);
    printf("RMS epsilon:        %.9g\n", gguf.rms_epsilon);
    printf("sampling defaults: temp=%.7g top-k=%u top-p=%.7g\n",
           gguf.sampling_temperature, gguf.sampling_top_k,
           gguf.sampling_top_p);
    printf("attention schedule:\n");
    for (layer = 0; layer < gguf.config.n_layer; ++layer) {
        int sliding = gguf.sliding_window_pattern[layer] != 0;
        printf("  layer %2u: %-7s kv-heads=%u head-dim=%u rope-base=%.7g\n",
               layer, sliding ? "sliding" : "global",
               gguf.head_count_kv[layer],
               sliding ? gguf.key_length_swa : gguf.key_length,
               sliding ? gguf.rope_freq_base_swa : gguf.rope_freq_base);
    }
    coli_gemma4_gguf_close(&gguf);
    return 0;
}

static int command_vision_info(int argc, char **argv) {
    coli_gemma4_vision vision;
    uint32_t width = 0, height = 0, tokens = 0;
    if (argc != 1 && argc != 3) {
        fprintf(stderr, "vision-info expects MMPROJ.gguf and optional width height\n");
        return 2;
    }
    if (argc == 3 &&
        (parse_u32(argv[1], &width) != 0 || parse_u32(argv[2], &height) != 0 ||
         !width || !height)) {
        fprintf(stderr, "invalid source image dimensions\n");
        return 2;
    }
    memset(&vision, 0, sizeof(vision));
    if (coli_gemma4_vision_open(&vision, argv[0]) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_vision_last_error(&vision));
        return 1;
    }
    printf("architecture:       %s/%s\n",
           vision.gguf.architecture, vision.gguf.projector_type);
    printf("GGUF/tensors:       v%u / %" PRIu64 "\n",
           vision.gguf.version, vision.gguf.tensor_count);
    printf("image/patch/merge:  %u / %u / %u\n",
           vision.gguf.vision_image_size, vision.gguf.vision_patch_size,
           vision.merge_size);
    printf("layers/width/ff:    %u / %u / %u\n",
           vision.gguf.vision_block_count,
           vision.gguf.vision_embedding_length,
           vision.gguf.vision_feed_forward_length);
    printf("heads/epsilon:      %u / %.9g\n",
           vision.gguf.vision_head_count, vision.gguf.vision_epsilon);
    printf("decoder projection: %u\n", vision.gguf.vision_projection_dim);
    printf("image token range:  %u..%u\n",
           vision.minimum_tokens, vision.maximum_tokens);
    printf("image mean:         %.7g %.7g %.7g\n",
           vision.gguf.vision_image_mean[0],
           vision.gguf.vision_image_mean[1],
           vision.gguf.vision_image_mean[2]);
    printf("image std:          %.7g %.7g %.7g\n",
           vision.gguf.vision_image_std[0],
           vision.gguf.vision_image_std[1],
           vision.gguf.vision_image_std[2]);
    if (argc == 3) {
        uint32_t target_width, target_height;
        if (coli_gemma4_vision_target_size(
                &vision, width, height, &target_width, &target_height,
                &tokens) != 0) {
            fprintf(stderr, "cannot calculate target image dimensions\n");
            coli_gemma4_vision_close(&vision);
            return 1;
        }
        printf("source/target:      %ux%u -> %ux%u\n",
               width, height, target_width, target_height);
        printf("projected tokens:   %u (%ux%u)\n", tokens,
               target_width / (vision.gguf.vision_patch_size * vision.merge_size),
               target_height / (vision.gguf.vision_patch_size * vision.merge_size));
    }
    coli_gemma4_vision_close(&vision);
    return 0;
}

static int command_vision_image_info(const char *mmproj_path,
                                     const char *image_path) {
    coli_gemma4_vision vision;
    uint8_t *rgb = NULL;
    uint32_t source_width = 0, source_height = 0;
    uint32_t target_width = 0, target_height = 0, token_count = 0;
    char error[COLI_GEMMA4_VISION_ERROR_MAX];
    int status = 1;
    memset(&vision, 0, sizeof(vision));
    if (coli_gemma4_vision_load_image(
            image_path, &rgb, &source_width, &source_height,
            error, sizeof(error)) != 0) {
        fprintf(stderr, "error: %s\n", error);
        goto cleanup;
    }
    if (coli_gemma4_vision_open(&vision, mmproj_path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_vision_last_error(&vision));
        goto cleanup;
    }
    if (coli_gemma4_vision_target_size(
            &vision, source_width, source_height,
            &target_width, &target_height, &token_count) != 0) {
        fprintf(stderr, "error: cannot calculate image geometry\n");
        goto cleanup;
    }
    printf("image:              %s\n", image_path);
    printf("source geometry:    %u x %u\n", source_width, source_height);
    printf("prepared geometry:  %u x %u\n", target_width, target_height);
    printf("projected tokens:   %u\n", token_count);
    status = 0;
cleanup:
    free(rgb);
    coli_gemma4_vision_close(&vision);
    return status;
}

static int command_vision_patch_probe(const char *path,
                                      const char *width_text,
                                      const char *height_text,
                                      const char *output_path) {
    coli_gemma4_vision vision;
    coli_gemma4_vision_image image;
    uint8_t *rgb = NULL;
    float *embeddings = NULL;
    uint32_t source_width, source_height, patch_count = 0;
    uint64_t pixel_values, index;
    clock_t started, stopped;
    int status = 1;
    if (parse_u32(width_text, &source_width) != 0 ||
        parse_u32(height_text, &source_height) != 0 ||
        !source_width || !source_height ||
        source_width > SIZE_MAX / 3U / source_height) {
        fprintf(stderr, "invalid source image dimensions\n");
        return 2;
    }
    memset(&vision, 0, sizeof(vision));
    memset(&image, 0, sizeof(image));
    pixel_values = (uint64_t)source_width * source_height * 3U;
    if (pixel_values > SIZE_MAX) {
        fprintf(stderr, "source image is too large\n");
        return 2;
    }
    rgb = (uint8_t *)malloc((size_t)pixel_values);
    if (!rgb) {
        fprintf(stderr, "out of memory allocating probe image\n");
        return 1;
    }
    for (index = 0; index < pixel_values; ++index)
        rgb[index] = (uint8_t)((index * UINT64_C(37) + UINT64_C(11)) & 255U);
    if (coli_gemma4_vision_open(&vision, path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_vision_last_error(&vision));
        goto cleanup;
    }
    started = clock();
    if (coli_gemma4_vision_prepare_rgb(
            &vision, rgb, source_width, source_height, &image) != 0 ||
        coli_gemma4_vision_patch_embeddings(
            &vision, &image, &embeddings, &patch_count) != 0) {
        fprintf(stderr, "Gemma 4 patch embedding probe failed\n");
        goto cleanup;
    }
    stopped = clock();
    if (output_path && write_f32(
            output_path, embeddings,
            (size_t)patch_count * vision.gguf.vision_embedding_length) != 0)
        goto cleanup;
    printf("source/prepared:    %ux%u -> %ux%u\n",
           source_width, source_height, image.width, image.height);
    printf("patches/width:      %u / %u\n",
           patch_count, vision.gguf.vision_embedding_length);
    printf("embedding hash:    0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(
               embeddings,
               (size_t)patch_count * vision.gguf.vision_embedding_length));
    printf("elapsed:           %.3f ms\n",
           1000.0 * (double)(stopped - started) / (double)CLOCKS_PER_SEC);
    if (output_path) printf("wrote embeddings:  %s\n", output_path);
    status = 0;
cleanup:
    free(rgb);
    free(embeddings);
    coli_gemma4_vision_image_close(&image);
    coli_gemma4_vision_close(&vision);
    return status;
}

static int command_vision_encode_probe(int argc, char **argv) {
    coli_gemma4_vision vision;
    coli_gemma4_vision_image image;
    uint8_t *rgb = NULL;
    float *embeddings = NULL;
    const char *input_path = NULL, *output_path = NULL, *layer_text = NULL;
    const char *prepared_path = NULL;
    const char *start_layer_text = NULL;
    const char *trace_layer_text = NULL, *trace_directory = NULL;
    uint32_t source_width, source_height, count = 0, layers = 0;
    uint32_t start_layer = 0, trace_layer = 0;
    uint32_t output_width = 0;
    uint64_t pixel_values, index;
    clock_t started, stopped;
    int partial, status = 1;
    if (argc < 3 ||
        parse_u32(argv[1], &source_width) != 0 ||
        parse_u32(argv[2], &source_height) != 0 ||
        !source_width || !source_height ||
        source_width > SIZE_MAX / 3U / source_height ||
        option_value(argc - 3, argv + 3, "--input-f32", &input_path) < 0 ||
        option_value(argc - 3, argv + 3, "--output-f32", &output_path) < 0 ||
        option_value(argc - 3, argv + 3, "--prepared-f32", &prepared_path) < 0 ||
        option_value(argc - 3, argv + 3, "--layers", &layer_text) < 0 ||
        option_value(argc - 3, argv + 3, "--start-layer",
                     &start_layer_text) < 0 ||
        option_value(argc - 3, argv + 3, "--trace-layer",
                     &trace_layer_text) < 0 ||
        option_value(argc - 3, argv + 3, "--trace-dir",
                     &trace_directory) < 0 ||
        (layer_text && parse_u32(layer_text, &layers) != 0) ||
        (start_layer_text && parse_u32(start_layer_text, &start_layer) != 0) ||
        (trace_layer_text && parse_u32(trace_layer_text, &trace_layer) != 0) ||
        ((trace_layer_text != NULL) != (trace_directory != NULL)) ||
        (input_path && !layer_text) ||
        (start_layer_text && !input_path)) {
        fprintf(stderr, "invalid vision encode probe arguments\n");
        return 2;
    }
    partial = layer_text != NULL;
    memset(&vision, 0, sizeof(vision));
    memset(&image, 0, sizeof(image));
    pixel_values = (uint64_t)source_width * source_height * 3U;
    if (pixel_values > SIZE_MAX) {
        fprintf(stderr, "source image is too large\n");
        return 2;
    }
    rgb = (uint8_t *)malloc((size_t)pixel_values);
    if (!rgb) {
        fprintf(stderr, "out of memory allocating probe image\n");
        return 1;
    }
    for (index = 0; index < pixel_values; ++index)
        rgb[index] = (uint8_t)((index * UINT64_C(37) + UINT64_C(11)) & 255U);
    if (coli_gemma4_vision_open(&vision, argv[0]) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_vision_last_error(&vision));
        goto cleanup;
    }
    if (partial && (layers > vision.gguf.vision_block_count ||
                    start_layer > layers)) {
        fprintf(stderr, "--layers exceeds the projector's %u blocks\n",
                vision.gguf.vision_block_count);
        status = 2;
        goto cleanup;
    }
    if (trace_layer_text &&
        (trace_layer >= vision.gguf.vision_block_count ||
         (partial && (trace_layer < start_layer || trace_layer >= layers)))) {
        fprintf(stderr, "--trace-layer is outside the evaluated block range\n");
        status = 2;
        goto cleanup;
    }
    if (trace_layer_text)
        coli_gemma4_vision_trace_layer(&vision, trace_layer, trace_directory);
    started = clock();
    if (coli_gemma4_vision_prepare_rgb(
            &vision, rgb, source_width, source_height, &image) != 0)
        goto graph_failure;
    if (prepared_path) {
        size_t plane = (size_t)image.width * image.height;
        float *prepared = (float *)malloc(image.value_count * sizeof(*prepared));
        uint32_t py, px, channel;
        if (!prepared) goto graph_failure;
        for (channel = 0; channel < 3; ++channel)
            for (py = 0; py < image.height; ++py)
                for (px = 0; px < image.width; ++px)
                    prepared[(size_t)channel * plane +
                             (size_t)py * image.width + px] =
                        2.0F * image.pixels[
                            ((size_t)py * image.width + px) * 3U + channel] -
                        1.0F;
        if (write_f32(prepared_path, prepared, image.value_count) != 0) {
            free(prepared);
            goto cleanup;
        }
        free(prepared);
    }
    if (partial) {
        if (coli_gemma4_vision_patch_embeddings(
                &vision, &image, &embeddings, &count) != 0 ||
            (input_path && read_f32(
                input_path, embeddings,
                (size_t)count * vision.gguf.vision_embedding_length) != 0) ||
            coli_gemma4_vision_transform_range(
                &vision, embeddings, count, image.patch_columns,
                start_layer, layers) != 0)
            goto graph_failure;
        output_width = vision.gguf.vision_embedding_length;
    } else {
        if (coli_gemma4_vision_encode(
                &vision, &image, &embeddings, &count) != 0)
            goto graph_failure;
        output_width = vision.gguf.vision_projection_dim;
    }
    stopped = clock();
    if (output_path && write_f32(output_path, embeddings,
            (size_t)count * output_width) != 0)
        goto cleanup;
    printf("source/prepared:    %ux%u -> %ux%u\n",
           source_width, source_height, image.width, image.height);
    if (partial)
        printf("transform blocks:   %u..%u / %u\n", start_layer, layers,
               vision.gguf.vision_block_count);
    printf("vectors/width:      %u / %u\n", count, output_width);
    printf("embedding hash:    0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(
               embeddings, (size_t)count * output_width));
    printf("elapsed:           %.3f ms\n",
           1000.0 * (double)(stopped - started) / (double)CLOCKS_PER_SEC);
    if (output_path) printf("wrote embeddings:  %s\n", output_path);
    status = 0;
    goto cleanup;
graph_failure:
    fprintf(stderr, "Gemma 4 vision encode probe failed: %s\n",
            coli_gemma4_vision_last_error(&vision));
cleanup:
    free(rgb);
    free(embeddings);
    coli_gemma4_vision_image_close(&image);
    coli_gemma4_vision_close(&vision);
    return status;
}

static int command_route(int argc, char **argv, const char *path) {
    coli_gemma4_gguf gguf;
    coli_gemma4_router router;
    const char *value = NULL, *input_path = NULL, *probability_path = NULL;
    uint32_t layer = 0, seed = 1, i;
    float *input = NULL, *probabilities = NULL, *weights = NULL, *effective = NULL;
    uint32_t *ids = NULL;
    int status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&router, 0, sizeof(router));
    if (option_value(argc, argv, "--layer", &value) < 0 ||
        (value && parse_u32(value, &layer) != 0)) {
        fprintf(stderr, "invalid --layer value\n");
        return 2;
    }
    value = NULL;
    if (option_value(argc, argv, "--seed", &value) < 0 ||
        (value && parse_u32(value, &seed) != 0) ||
        option_value(argc, argv, "--input-f32", &input_path) < 0 ||
        option_value(argc, argv, "--probabilities-f32", &probability_path) < 0) {
        fprintf(stderr, "invalid or missing option value\n");
        return 2;
    }
    if (coli_gemma4_gguf_open(&gguf, path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    if (coli_gemma4_router_open(&router, &gguf, layer) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_router_last_error(&router));
        goto cleanup;
    }
    input = (float *)malloc((size_t)router.width * sizeof(float));
    probabilities = (float *)malloc((size_t)router.expert_count * sizeof(float));
    ids = (uint32_t *)malloc((size_t)router.selected_count * sizeof(uint32_t));
    weights = (float *)malloc((size_t)router.selected_count * sizeof(float));
    effective = (float *)malloc((size_t)router.selected_count * sizeof(float));
    if (!input || !probabilities || !ids || !weights || !effective) {
        fprintf(stderr, "error: out of memory allocating router buffers\n");
        goto cleanup;
    }
    if (input_path) {
        if (read_f32(input_path, input, router.width) != 0) goto cleanup;
    } else {
        make_input(input, router.width, seed);
    }
    if (coli_gemma4_router_route(&router, input, probabilities, ids,
                                 weights, effective) != 0) {
        fprintf(stderr, "error: router evaluation failed\n");
        goto cleanup;
    }
    if (probability_path &&
        write_f32(probability_path, probabilities, router.expert_count) != 0)
        goto cleanup;
    printf("layer:             %u\n", layer);
    printf("input:             %s\n",
           input_path ? input_path : "deterministic pseudorandom");
    printf("probability hash:  0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(probabilities, router.expert_count));
    printf("selected experts:\n");
    for (i = 0; i < router.selected_count; ++i)
        printf("  %2u: expert %3u  normalized=%.9g  effective=%.9g\n",
               i, ids[i], weights[i], effective[i]);
    if (probability_path) printf("wrote probabilities: %s\n", probability_path);
    status = 0;

cleanup:
    free(input); free(probabilities); free(ids); free(weights); free(effective);
    coli_gemma4_router_close(&router);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_attention_projection(int argc, char **argv, const char *path) {
    coli_gemma4_gguf gguf;
    coli_gemma4_attention attention;
    const char *option = NULL, *input_path = NULL;
    const char *query_path = NULL, *key_path = NULL, *value_path = NULL;
    uint32_t layer = 0, position = 0, seed = 1;
    uint32_t query_width, kv_width;
    float *input = NULL, *query = NULL, *key = NULL, *value = NULL;
    int status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&attention, 0, sizeof(attention));
    if (option_value(argc, argv, "--layer", &option) < 0 ||
        (option && parse_u32(option, &layer) != 0)) {
        fprintf(stderr, "invalid --layer value\n");
        return 2;
    }
    option = NULL;
    if (option_value(argc, argv, "--position", &option) < 0 ||
        (option && parse_u32(option, &position) != 0)) {
        fprintf(stderr, "invalid --position value\n");
        return 2;
    }
    option = NULL;
    if (option_value(argc, argv, "--seed", &option) < 0 ||
        (option && parse_u32(option, &seed) != 0) ||
        option_value(argc, argv, "--input-f32", &input_path) < 0 ||
        option_value(argc, argv, "--query-f32", &query_path) < 0 ||
        option_value(argc, argv, "--key-f32", &key_path) < 0 ||
        option_value(argc, argv, "--value-f32", &value_path) < 0) {
        fprintf(stderr, "invalid or missing option value\n");
        return 2;
    }
    if (coli_gemma4_gguf_open(&gguf, path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    if (coli_gemma4_attention_open(&attention, &gguf, layer) != 0) {
        fprintf(stderr, "error: %s\n",
                coli_gemma4_attention_last_error(&attention));
        goto cleanup;
    }
    query_width = attention.query_heads * attention.head_dim;
    kv_width = attention.kv_heads * attention.head_dim;
    input = (float *)malloc((size_t)attention.model_width * sizeof(float));
    query = (float *)malloc((size_t)query_width * sizeof(float));
    key = (float *)malloc((size_t)kv_width * sizeof(float));
    value = (float *)malloc((size_t)kv_width * sizeof(float));
    if (!input || !query || !key || !value) {
        fprintf(stderr, "error: out of memory allocating attention vectors\n");
        goto cleanup;
    }
    if (input_path) {
        if (read_f32(input_path, input, attention.model_width) != 0) goto cleanup;
    } else make_input(input, attention.model_width, seed);
    if (coli_gemma4_attention_project(&attention, input, query, key, value) != 0 ||
        coli_gemma4_attention_apply_rope(&attention, position, query, key) != 0) {
        fprintf(stderr, "error: attention projection failed\n");
        goto cleanup;
    }
    if ((query_path && write_f32(query_path, query, query_width) != 0) ||
        (key_path && write_f32(key_path, key, kv_width) != 0) ||
        (value_path && write_f32(value_path, value, kv_width) != 0))
        goto cleanup;
    printf("layer:             %u (%s)\n", layer,
           attention.sliding ? "sliding" : "global");
    printf("position:          %u\n", position);
    printf("heads/dimension:   q=%u kv=%u dim=%u\n",
           attention.query_heads, attention.kv_heads, attention.head_dim);
    printf("value projection:  %s\n",
           attention.key_equals_value ? "derived from key" : "independent");
    printf("query hash:        0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(query, query_width));
    printf("key hash:          0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(key, kv_width));
    printf("value hash:        0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(value, kv_width));
    status = 0;

cleanup:
    free(input); free(query); free(key); free(value);
    coli_gemma4_attention_close(&attention);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_attention_sequence(int argc, char **argv, const char *path) {
    coli_gemma4_gguf gguf;
    coli_gemma4_attention attention;
    coli_gemma4_kv_cache cache;
    const char *option = NULL, *input_path = NULL, *output_path = NULL;
    uint32_t layer = 0, tokens = 1, seed = 1, token;
    uint64_t value_count;
    float *input = NULL, *output = NULL;
    int status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&attention, 0, sizeof(attention));
    memset(&cache, 0, sizeof(cache));
    if (option_value(argc, argv, "--layer", &option) < 0 ||
        (option && parse_u32(option, &layer) != 0)) {
        fprintf(stderr, "invalid --layer value\n");
        return 2;
    }
    option = NULL;
    if (option_value(argc, argv, "--tokens", &option) < 0 ||
        (option && parse_u32(option, &tokens) != 0) || !tokens) {
        fprintf(stderr, "invalid --tokens value\n");
        return 2;
    }
    option = NULL;
    if (option_value(argc, argv, "--seed", &option) < 0 ||
        (option && parse_u32(option, &seed) != 0) ||
        option_value(argc, argv, "--input-f32", &input_path) < 0 ||
        option_value(argc, argv, "--output-f32", &output_path) < 0) {
        fprintf(stderr, "invalid or missing option value\n");
        return 2;
    }
    if (coli_gemma4_gguf_open(&gguf, path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    if (coli_gemma4_attention_open(&attention, &gguf, layer) != 0) {
        fprintf(stderr, "error: %s\n",
                coli_gemma4_attention_last_error(&attention));
        goto cleanup;
    }
    value_count = (uint64_t)tokens * attention.model_width;
    if (value_count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "error: attention sequence is too large\n");
        goto cleanup;
    }
    input = (float *)malloc((size_t)value_count * sizeof(float));
    output = (float *)malloc((size_t)value_count * sizeof(float));
    if (!input || !output ||
        coli_gemma4_kv_cache_init(&cache, &attention, tokens) != 0) {
        fprintf(stderr, "error: out of memory allocating attention sequence\n");
        goto cleanup;
    }
    if (input_path) {
        if (read_f32(input_path, input, (size_t)value_count) != 0) goto cleanup;
    } else make_input(input, (size_t)value_count, seed);
    for (token = 0; token < tokens; ++token) {
        if (coli_gemma4_attention_step(
                &attention, &cache, token,
                input + (size_t)token * attention.model_width,
                output + (size_t)token * attention.model_width) != 0) {
            fprintf(stderr, "error: causal attention failed at token %u\n", token);
            goto cleanup;
        }
    }
    if (output_path && write_f32(output_path, output, (size_t)value_count) != 0)
        goto cleanup;
    printf("layer:             %u (%s)\n", layer,
           attention.sliding ? "sliding" : "global");
    printf("tokens:            %u\n", tokens);
    printf("heads/dimension:   q=%u kv=%u dim=%u\n",
           attention.query_heads, attention.kv_heads, attention.head_dim);
    printf("cache:             %s, capacity=%u, retained=%u\n",
           attention.sliding ? "sliding ring" : "global append-only",
           cache.capacity, cache.count);
    printf("output hash:       0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(output, (size_t)value_count));
    if (output_path) printf("wrote output:      %s\n", output_path);
    status = 0;

cleanup:
    free(input); free(output);
    coli_gemma4_kv_cache_close(&cache);
    coli_gemma4_attention_close(&attention);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_dense_mlp(int argc, char **argv, const char *path) {
    coli_gemma4_gguf gguf;
    coli_gemma4_dense_mlp mlp;
    const char *option = NULL, *input_path = NULL, *output_path = NULL;
    uint32_t layer = 0, seed = 1;
    float *input = NULL, *output = NULL;
    int status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&mlp, 0, sizeof(mlp));
    if (option_value(argc, argv, "--layer", &option) < 0 ||
        (option && parse_u32(option, &layer) != 0)) {
        fprintf(stderr, "invalid --layer value\n");
        return 2;
    }
    option = NULL;
    if (option_value(argc, argv, "--seed", &option) < 0 ||
        (option && parse_u32(option, &seed) != 0) ||
        option_value(argc, argv, "--input-f32", &input_path) < 0 ||
        option_value(argc, argv, "--output-f32", &output_path) < 0) {
        fprintf(stderr, "invalid or missing option value\n");
        return 2;
    }
    if (coli_gemma4_gguf_open(&gguf, path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    if (coli_gemma4_dense_mlp_open(&mlp, &gguf, layer) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_dense_mlp_last_error(&mlp));
        goto cleanup;
    }
    input = (float *)malloc((size_t)mlp.model_width * sizeof(float));
    output = (float *)malloc((size_t)mlp.model_width * sizeof(float));
    if (!input || !output) {
        fprintf(stderr, "error: out of memory allocating dense MLP vectors\n");
        goto cleanup;
    }
    if (input_path) {
        if (read_f32(input_path, input, mlp.model_width) != 0) goto cleanup;
    } else make_input(input, mlp.model_width, seed);
    if (coli_gemma4_dense_mlp_run(&mlp, input, output) != 0) {
        fprintf(stderr, "error: dense MLP evaluation failed\n");
        goto cleanup;
    }
    if (output_path && write_f32(output_path, output, mlp.model_width) != 0)
        goto cleanup;
    printf("layer:             %u\n", layer);
    printf("shape:             %u -> %u -> %u\n",
           mlp.model_width, mlp.intermediate_width, mlp.model_width);
    printf("output hash:       0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(output, mlp.model_width));
    if (output_path) printf("wrote output:      %s\n", output_path);
    status = 0;

cleanup:
    free(input); free(output);
    coli_gemma4_dense_mlp_close(&mlp);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_layer_sequence(int argc, char **argv,
                                  const char *model_path,
                                  const char *packed_dir) {
    coli_gemma4_gguf gguf;
    coli_gemma4_decoder_layer decoder;
    coli_gemma4_kv_cache cache;
    coli_gemma4_backend gemma;
    coli_expert_backend experts;
    const char *option = NULL, *input_path = NULL, *output_path = NULL;
    uint32_t layer = 0, tokens = 1, seed = 1, token;
    uint64_t value_count;
    float *input = NULL, *output = NULL;
    int status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&decoder, 0, sizeof(decoder));
    memset(&cache, 0, sizeof(cache));
    memset(&gemma, 0, sizeof(gemma));
    if (option_value(argc, argv, "--layer", &option) < 0 ||
        (option && parse_u32(option, &layer) != 0)) {
        fprintf(stderr, "invalid --layer value\n");
        return 2;
    }
    option = NULL;
    if (option_value(argc, argv, "--tokens", &option) < 0 ||
        (option && parse_u32(option, &tokens) != 0) || !tokens) {
        fprintf(stderr, "invalid --tokens value\n");
        return 2;
    }
    option = NULL;
    if (option_value(argc, argv, "--seed", &option) < 0 ||
        (option && parse_u32(option, &seed) != 0) ||
        option_value(argc, argv, "--input-f32", &input_path) < 0 ||
        option_value(argc, argv, "--output-f32", &output_path) < 0) {
        fprintf(stderr, "invalid or missing option value\n");
        return 2;
    }
    if (coli_gemma4_gguf_open(&gguf, model_path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    if (coli_gemma4_decoder_layer_open(&decoder, &gguf, layer) != 0) {
        fprintf(stderr, "error: %s\n",
                coli_gemma4_decoder_layer_last_error(&decoder));
        goto cleanup;
    }
    if (coli_gemma4_open_packed(&gemma, packed_dir) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (strlen(gguf.path) + 1 > sizeof(gemma.source)) {
        fprintf(stderr, "error: model path is too long for expert backend\n");
        goto cleanup;
    }
    memcpy(gemma.source, gguf.path, strlen(gguf.path) + 1);
    if (gemma.config.n_embd != gguf.config.n_embd ||
        gemma.config.n_expert != gguf.config.n_expert ||
        gemma.config.n_expert_used != gguf.config.n_expert_used ||
        !coli_gemma4_find_layer(&gemma, layer)) {
        fprintf(stderr, "error: packed manifest and GGUF configuration disagree\n");
        goto cleanup;
    }
    value_count = (uint64_t)tokens * decoder.model_width;
    if (value_count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "error: decoder sequence is too large\n");
        goto cleanup;
    }
    input = (float *)malloc((size_t)value_count * sizeof(float));
    output = (float *)malloc((size_t)value_count * sizeof(float));
    if (!input || !output ||
        coli_gemma4_kv_cache_init(&cache, &decoder.attention, tokens) != 0) {
        fprintf(stderr, "error: out of memory allocating decoder sequence\n");
        goto cleanup;
    }
    if (input_path) {
        if (read_f32(input_path, input, (size_t)value_count) != 0) goto cleanup;
    } else make_input(input, (size_t)value_count, seed);
    experts = coli_gemma4_expert_backend(&gemma);
    for (token = 0; token < tokens; ++token) {
        if (coli_gemma4_decoder_layer_step(
                &decoder, &cache, &experts, token,
                input + (size_t)token * decoder.model_width,
                output + (size_t)token * decoder.model_width) != 0) {
            fprintf(stderr, "error: decoder layer failed at token %u: %s\n",
                    token, coli_gemma4_last_error(&gemma));
            goto cleanup;
        }
    }
    if (output_path && write_f32(output_path, output, (size_t)value_count) != 0)
        goto cleanup;
    printf("layer:             %u (%s)\n", layer,
           decoder.attention.sliding ? "sliding" : "global");
    printf("tokens:            %u\n", tokens);
    printf("dense MLP:         %u -> %u -> %u\n", decoder.model_width,
           decoder.dense_mlp.intermediate_width, decoder.model_width);
    printf("routed experts:    top-%u of %u\n",
           decoder.router.selected_count, decoder.router.expert_count);
    printf("cache:             %s, capacity=%u, retained=%u\n",
           decoder.attention.sliding ? "sliding ring" : "global append-only",
           cache.capacity, cache.count);
    printf("output hash:       0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(output, (size_t)value_count));
    if (output_path) printf("wrote output:      %s\n", output_path);
    status = 0;

cleanup:
    free(input); free(output);
    coli_gemma4_kv_cache_close(&cache);
    coli_gemma4_close(&gemma);
    coli_gemma4_decoder_layer_close(&decoder);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_routed_mlp(int argc, char **argv,
                              const char *model_path, const char *packed_dir) {
    coli_gemma4_gguf gguf;
    coli_gemma4_router router;
    coli_gemma4_backend gemma;
    coli_expert_backend experts;
    const coli_gemma4_tensor *norm_tensor;
    const char *value = NULL, *input_path = NULL, *output_path = NULL;
    char norm_name[128];
    uint32_t layer = 0, seed = 1, i;
    uint32_t *ids = NULL;
    float *input = NULL, *normalized = NULL, *norm_weight = NULL;
    float *weights = NULL, *effective = NULL, *output = NULL;
    int length, status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&router, 0, sizeof(router));
    memset(&gemma, 0, sizeof(gemma));
    if (option_value(argc, argv, "--layer", &value) < 0 ||
        (value && parse_u32(value, &layer) != 0)) {
        fprintf(stderr, "invalid --layer value\n");
        return 2;
    }
    value = NULL;
    if (option_value(argc, argv, "--seed", &value) < 0 ||
        (value && parse_u32(value, &seed) != 0) ||
        option_value(argc, argv, "--input-f32", &input_path) < 0 ||
        option_value(argc, argv, "--output-f32", &output_path) < 0) {
        fprintf(stderr, "invalid or missing option value\n");
        return 2;
    }
    if (coli_gemma4_gguf_open(&gguf, model_path) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_gguf_last_error(&gguf));
        return 1;
    }
    if (coli_gemma4_router_open(&router, &gguf, layer) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_router_last_error(&router));
        goto cleanup;
    }
    if (coli_gemma4_open_packed(&gemma, packed_dir) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (strlen(gguf.path) + 1 > sizeof(gemma.source)) {
        fprintf(stderr, "error: model path is too long for expert backend\n");
        goto cleanup;
    }
    /* The explicit model argument is authoritative and also makes old
       manifests with cwd-relative source paths portable. */
    memcpy(gemma.source, gguf.path, strlen(gguf.path) + 1);
    if (gemma.config.n_embd != gguf.config.n_embd ||
        gemma.config.n_expert != gguf.config.n_expert ||
        gemma.config.n_expert_used != gguf.config.n_expert_used ||
        !coli_gemma4_find_layer(&gemma, layer)) {
        fprintf(stderr, "error: packed manifest and GGUF configuration disagree\n");
        goto cleanup;
    }
    length = snprintf(norm_name, sizeof(norm_name),
                      "blk.%u.pre_ffw_norm_2.weight", layer);
    norm_tensor = length >= 0 && (size_t)length < sizeof(norm_name) ?
        coli_gemma4_gguf_find(&gguf, norm_name) : NULL;
    if (!norm_tensor || norm_tensor->type != COLI_GGML_TYPE_F32 ||
        norm_tensor->n_dims != 1 || norm_tensor->dims[0] != router.width ||
        norm_tensor->nbytes != (uint64_t)router.width * sizeof(float)) {
        fprintf(stderr, "error: invalid or missing %s\n", norm_name);
        goto cleanup;
    }
    input = (float *)malloc((size_t)router.width * sizeof(float));
    normalized = (float *)malloc((size_t)router.width * sizeof(float));
    norm_weight = (float *)malloc((size_t)norm_tensor->nbytes);
    output = (float *)malloc((size_t)router.width * sizeof(float));
    ids = (uint32_t *)malloc((size_t)router.selected_count * sizeof(uint32_t));
    weights = (float *)malloc((size_t)router.selected_count * sizeof(float));
    effective = (float *)malloc((size_t)router.selected_count * sizeof(float));
    if (!input || !normalized || !norm_weight || !output ||
        !ids || !weights || !effective) {
        fprintf(stderr, "error: out of memory allocating routed-MLP buffers\n");
        goto cleanup;
    }
    if (input_path) {
        if (read_f32(input_path, input, router.width) != 0) goto cleanup;
    } else make_input(input, router.width, seed);
    if (coli_gemma4_gguf_read(&gguf, norm_tensor, norm_weight,
                              (size_t)norm_tensor->nbytes) != 0 ||
        coli_gemma4_rmsnorm(input, norm_weight, router.width,
                            gguf.rms_epsilon, normalized) != 0 ||
        coli_gemma4_router_route(&router, input, NULL, ids,
                                 weights, effective) != 0) {
        fprintf(stderr, "error: routed-MLP normalization or routing failed\n");
        goto cleanup;
    }
    experts = coli_gemma4_expert_backend(&gemma);
    if (experts.prepare_layer(experts.ctx, layer, ids,
                              router.selected_count) != 0 ||
        experts.run_experts(experts.ctx, layer, ids, weights,
                            router.selected_count, normalized, output) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    experts.release_layer(experts.ctx, layer);
    if (output_path && write_f32(output_path, output, router.width) != 0)
        goto cleanup;
    printf("layer:             %u\n", layer);
    printf("input:             %s\n",
           input_path ? input_path : "deterministic pseudorandom");
    printf("selected experts:  ");
    for (i = 0; i < router.selected_count; ++i)
        printf("%s%u", i ? "," : "", ids[i]);
    printf("\n");
    printf("output hash:       0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(output, router.width));
    if (output_path) printf("wrote output:      %s\n", output_path);
    status = 0;

cleanup:
    free(input); free(normalized); free(norm_weight); free(output);
    free(ids); free(weights); free(effective);
    coli_gemma4_close(&gemma);
    coli_gemma4_router_close(&router);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

static int command_expert(int argc, char **argv, const char *directory) {
    coli_gemma4_backend gemma;
    coli_expert_backend expert_backend;
    const coli_gemma4_layer *layer;
    const char *value = NULL;
    const char *input_path = NULL;
    const char *output_path = NULL;
    const char *cuda_device_text = NULL;
    const char *repeat_text = NULL;
    uint32_t layer_number = 0, expert_number = 0, seed = 1, repeat = 1;
    uint32_t cuda_device = 0;
    uint32_t ids[1];
    float weights[1] = {1.0F};
    float *input = NULL, *output = NULL;
    clock_t started, stopped;
    int status = 1;
    size_t preview, i;

    if (option_value(argc, argv, "--layer", &value) < 0 ||
        (value && parse_u32(value, &layer_number) != 0)) {
        fprintf(stderr, "invalid --layer value\n");
        return 2;
    }
    value = NULL;
    if (option_value(argc, argv, "--expert", &value) < 0 ||
        (value && parse_u32(value, &expert_number) != 0)) {
        fprintf(stderr, "invalid --expert value\n");
        return 2;
    }
    value = NULL;
    if (option_value(argc, argv, "--seed", &value) < 0 ||
        (value && parse_u32(value, &seed) != 0)) {
        fprintf(stderr, "invalid --seed value\n");
        return 2;
    }
    if (option_value(argc, argv, "--input-f32", &input_path) < 0 ||
        option_value(argc, argv, "--output-f32", &output_path) < 0 ||
        option_value(argc, argv, "--cuda-device", &cuda_device_text) < 0 ||
        option_value(argc, argv, "--repeat", &repeat_text) < 0 ||
        (cuda_device_text && parse_u32(cuda_device_text, &cuda_device) != 0) ||
        (repeat_text && (parse_u32(repeat_text, &repeat) != 0 || !repeat))) {
        fprintf(stderr, "missing option value\n");
        return 2;
    }

    if (coli_gemma4_open_packed(&gemma, directory) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        return 1;
    }
    if (cuda_device_text && coli_gemma4_cuda_configure(
            &gemma, (int)cuda_device) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (coli_gemma4_cache_configure(&gemma, 1) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    layer = coli_gemma4_find_layer(&gemma, layer_number);
    if (!layer || expert_number >= layer->expert_count) {
        fprintf(stderr, "error: layer or expert index is outside the model\n");
        goto cleanup;
    }
    input = (float *)malloc((size_t)layer->model_width * sizeof(float));
    output = (float *)malloc((size_t)layer->model_width * sizeof(float));
    if (!input || !output) {
        fprintf(stderr, "error: out of memory allocating expert vectors\n");
        goto cleanup;
    }
    if (input_path) {
        if (read_f32(input_path, input, layer->model_width) != 0) goto cleanup;
    } else {
        make_input(input, layer->model_width, seed);
    }

    ids[0] = expert_number;
    expert_backend = coli_gemma4_expert_backend(&gemma);
    if (expert_backend.prepare_layer(
            expert_backend.ctx, layer_number, ids, 1) != 0) {
        fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    started = clock();
    for (i = 0; i < repeat; ++i)
        if (expert_backend.run_experts(
                expert_backend.ctx, layer_number, ids, weights,
                1, input, output) != 0) {
            fprintf(stderr, "error: %s\n", coli_gemma4_last_error(&gemma));
            goto cleanup;
        }
    stopped = clock();
    expert_backend.release_layer(expert_backend.ctx, layer_number);

    if (output_path && write_f32(output_path, output, layer->model_width) != 0)
        goto cleanup;
    printf("layer/expert: %u/%u\n", layer_number, expert_number);
    printf("shape:        %u -> %u -> %u\n",
           layer->model_width, layer->expert_width, layer->model_width);
    printf("input:        %s\n", input_path ? input_path : "deterministic pseudorandom");
    printf("compute:      %s\n", cuda_device_text ? "CUDA" : "CPU");
    printf("payload:      %" PRIu64 " bytes\n", layer->payload_bytes);
    printf("elapsed:      %.3f ms\n",
           1000.0 * (double)(stopped - started) / (double)CLOCKS_PER_SEC);
    printf("average:      %.3f ms/expert (repeat %u)\n",
           1000.0 * (double)(stopped - started) /
               ((double)CLOCKS_PER_SEC * repeat), repeat);
    printf("checksum:     0x%016" PRIx64 "\n",
           coli_gemma4_checksum_f32(output, layer->model_width));
    if (output_path) printf("wrote output: %s\n", output_path);
    preview = layer->model_width < 8 ? layer->model_width : 8;
    printf("first:        ");
    for (i = 0; i < preview; ++i)
        printf("%s%.7g", i ? ", " : "", output[i]);
    printf("\n");
    status = 0;

cleanup:
    free(input);
    free(output);
    coli_gemma4_close(&gemma);
    return status;
}

static int serve_data(uint64_t request_id, const char *bytes, size_t count) {
    if (!count) return 0;
    if (printf("DATA %llu %zu\n", (unsigned long long)request_id, count) < 0 ||
        fwrite(bytes, 1, count, stdout) != count || fputc('\n', stdout) == EOF ||
        fflush(stdout) != 0) return -1;
    return 0;
}

typedef struct {
    coli_gemma4_model model;
    uint32_t *tokens;
    size_t token_count;
    int opened;
} gemma4_serve_slot;

typedef struct {
    Grammar grammar;
    GrState state;
    char *compiled;
    int active;
} gemma4_serve_grammar;

static void serve_grammar_close(gemma4_serve_grammar *grammar) {
    if (!grammar) return;
    if (grammar->active) gr_free(&grammar->grammar);
    free(grammar->compiled);
    memset(grammar, 0, sizeof(*grammar));
}

static int serve_grammar_open(gemma4_serve_grammar *grammar,
                              const char *source) {
    const char *cursor = source;
    char error[160];
    if (!grammar || !source) return -1;
    memset(grammar, 0, sizeof(*grammar));
    while (*cursor == ' ' || *cursor == '\t' ||
           *cursor == '\r' || *cursor == '\n') ++cursor;
    if (*cursor == '{') {
        grammar->compiled = schema_to_gbnf(source, error, sizeof(error));
        if (!grammar->compiled) return -1;
        source = grammar->compiled;
    }
    if (gr_parse(&grammar->grammar, source) != 0) {
        serve_grammar_close(grammar);
        return -1;
    }
    grammar->active = 1;
    gr_state_init(&grammar->state, &grammar->grammar);
    if (!grammar->state.alive) {
        serve_grammar_close(grammar);
        return -1;
    }
    return 0;
}

static int serve_grammar_filter(const coli_gemma4_tokenizer *tokenizer,
                                gemma4_serve_grammar *grammar,
                                float *logits, uint32_t vocabulary) {
    unsigned char first_mask[32];
    uint32_t token;
    int can_end = 0, valid = 0;
    if (!grammar || !grammar->active) return 0;
    gr_admissible(&grammar->state, first_mask, &can_end);
    for (token = 0; token < vocabulary; ++token) {
        size_t bytes = 0, index;
        char *piece = NULL;
        GrState candidate;
        if (!isfinite(logits[token])) continue;
        if (coli_gemma4_token_is_control(tokenizer, token)) {
            if (coli_gemma4_token_is_eog(tokenizer, token) && !can_end)
                logits[token] = -INFINITY;
            else
                ++valid;
            continue;
        }
        if (coli_gemma4_decode_token(
                tokenizer, token, NULL, 0, &bytes) != 0 || !bytes) {
            logits[token] = -INFINITY;
            continue;
        }
        piece = (char *)malloc(bytes + 1);
        if (!piece || coli_gemma4_decode_token(
                tokenizer, token, piece, bytes + 1, &bytes) != 0) {
            free(piece);
            return -1;
        }
        if (!(first_mask[(unsigned char)piece[0] >> 3] &
              (1U << ((unsigned char)piece[0] & 7)))) {
            logits[token] = -INFINITY;
            free(piece);
            continue;
        }
        candidate = grammar->state;
        for (index = 0; index < bytes; ++index)
            if (gr_accept(&candidate, (unsigned char)piece[index]) != 1)
                break;
        if (index != bytes)
            logits[token] = -INFINITY;
        else
            ++valid;
        free(piece);
    }
    return valid ? 0 : -1;
}

static int serve_grammar_accept_token(
    const coli_gemma4_tokenizer *tokenizer, gemma4_serve_grammar *grammar,
    uint32_t token) {
    size_t bytes = 0, index;
    char *piece;
    if (!grammar || !grammar->active ||
        coli_gemma4_token_is_control(tokenizer, token)) return 0;
    if (coli_gemma4_decode_token(
            tokenizer, token, NULL, 0, &bytes) != 0 || !bytes) return -1;
    piece = (char *)malloc(bytes + 1);
    if (!piece || coli_gemma4_decode_token(
            tokenizer, token, piece, bytes + 1, &bytes) != 0) {
        free(piece);
        return -1;
    }
    for (index = 0; index < bytes; ++index)
        if (gr_accept(&grammar->state, (unsigned char)piece[index]) != 1)
            break;
    free(piece);
    return index == bytes ? 0 : -1;
}

static size_t serve_prefix_length(const gemma4_serve_slot *slot,
                                  const uint32_t *tokens, size_t count) {
    size_t prefix = 0, available;
    if (!slot || !slot->opened || !slot->tokens || !tokens) return 0;
    available = slot->token_count < count ? slot->token_count : count;
    while (prefix < available && slot->tokens[prefix] == tokens[prefix])
        ++prefix;
    if (prefix == count && prefix) --prefix;
    return prefix;
}

typedef struct {
    uint64_t request_id;
    int action; /* 0 = run, 1 = graceful stop, 2 = client cancellation */
} gemma4_serve_control;

static int serve_stdin_ready(void) {
#ifdef _WIN32
    HANDLE input = (HANDLE)_get_osfhandle(_fileno(stdin));
    DWORD available = 0;
    return input != INVALID_HANDLE_VALUE &&
           PeekNamedPipe(input, NULL, 0, NULL, &available, NULL) && available;
#else
    fd_set readers;
    struct timeval timeout = {0, 0};
    int descriptor = fileno(stdin);
    FD_ZERO(&readers);
    FD_SET(descriptor, &readers);
    return select(descriptor + 1, &readers, NULL, NULL, &timeout) > 0;
#endif
}

static int serve_discard_bytes(uint64_t count) {
    unsigned char buffer[4096];
    while (count) {
        size_t chunk = count < sizeof(buffer) ? (size_t)count : sizeof(buffer);
        if (fread(buffer, 1, chunk, stdin) != chunk) return -1;
        count -= chunk;
    }
    return 0;
}

static int serve_control_poll(gemma4_serve_control *control) {
    while (control && !control->action && serve_stdin_ready()) {
        char header[512], kind[16];
        unsigned long long request_id = 0, payload = 0, grammar = 0;
        unsigned slot = 0, requested = 0;
        float temperature = 0.0F, top_p = 1.0F;
        int fields;
        if (!fgets(header, sizeof(header), stdin)) return -1;
        fields = sscanf(header, "%15s %llu %u %llu %u %f %f %llu",
                        kind, &request_id, &slot, &payload, &requested,
                        &temperature, &top_p, &grammar);
        if (fields >= 2 && (!strcmp(kind, "STOP") ||
                           !strcmp(kind, "CANCEL"))) {
            if ((uint64_t)request_id == control->request_id)
                control->action = !strcmp(kind, "CANCEL") ? 2 : 1;
            else {
                printf("ERROR %llu NOT_FOUND\n", request_id);
                fflush(stdout);
            }
            continue;
        }
        /* The native engine is intentionally single-flight.  Drain a
           pipelined submission so its payload cannot desynchronize framing. */
        if (fields >= 7 && !strcmp(kind, "SUBMIT") &&
            payload <= UINT64_MAX - grammar &&
            serve_discard_bytes((uint64_t)payload + grammar) == 0 &&
            fgetc(stdin) == '\n') {
            printf("ERROR %llu BUSY\n", request_id);
            fflush(stdout);
            continue;
        }
        control->action = 3;
        return -1;
    }
    return control && control->action ? 1 : 0;
}

static int serve_model_cancelled(void *opaque) {
    return serve_control_poll((gemma4_serve_control *)opaque) != 0;
}

static int command_serve(const char *cap_text) {
    coli_gemma4_gguf gguf;
    coli_gemma4_tokenizer tokenizer;
    coli_gemma4_backend gemma;
    const char *model_path = getenv("SNAP");
    const char *packed_dir = getenv("GEMMA4_PACKED");
    const char *context_text = getenv("CTX");
    const char *cuda_text = getenv("GEMMA4_CUDA_DEVICE");
    const char *kv_slots_text = getenv("KV_SLOTS");
    uint32_t cache_slots = 0, maximum_context = 4096, cuda_device = 0;
    uint32_t kv_slot_count = 1, slot_index;
    gemma4_serve_slot *slots = NULL;
    int status = 1;
    memset(&gguf, 0, sizeof(gguf));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(&gemma, 0, sizeof(gemma));
    if (!model_path || !packed_dir ||
        parse_u32(cap_text, &cache_slots) != 0 ||
        (context_text && (parse_u32(context_text, &maximum_context) != 0 ||
                          !maximum_context || maximum_context == UINT32_MAX)) ||
        (kv_slots_text && (parse_u32(kv_slots_text, &kv_slot_count) != 0 ||
                           !kv_slot_count || kv_slot_count > 16)) ||
        (cuda_text && parse_u32(cuda_text, &cuda_device) != 0)) {
        fprintf(stderr, "Gemma serve requires SNAP, GEMMA4_PACKED, and valid settings\n");
        return 2;
    }
    if (!cache_slots) cache_slots = 256;
    slots = (gemma4_serve_slot *)calloc(kv_slot_count, sizeof(*slots));
    if (!slots) {
        fprintf(stderr, "cannot allocate Gemma KV slots\n");
        return 1;
    }
    if (coli_gemma4_gguf_open(&gguf, model_path) != 0 ||
        coli_gemma4_tokenizer_init(&tokenizer, &gguf) != 0 ||
        coli_gemma4_open_packed(&gemma, packed_dir) != 0) {
        fprintf(stderr, "cannot initialize Gemma serve boundary: %s\n",
                gguf.last_error[0] ? gguf.last_error :
                tokenizer.last_error[0] ? tokenizer.last_error :
                coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
    if (strlen(gguf.path) + 1 > sizeof(gemma.source)) {
        fprintf(stderr, "Gemma model path is too long for serving\n");
        goto cleanup;
    }
    memcpy(gemma.source, gguf.path, strlen(gguf.path) + 1);
    if (coli_gemma4_cache_configure(&gemma, cache_slots) != 0 ||
        (cuda_text && coli_gemma4_cuda_configure(
             &gemma, (int)cuda_device) != 0)) {
        fprintf(stderr, "cannot configure Gemma serve boundary: %s\n",
                coli_gemma4_last_error(&gemma));
        goto cleanup;
    }
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("\x01\x01READY\x01\x01\nSTAT 0 0.00 0.0 0.00\n");
    fflush(stdout);
    for (;;) {
        char header[512];
        char kind[16];
        unsigned long long request_id = 0, payload_bytes_ull = 0;
        unsigned long long grammar_bytes_ull = 0;
        unsigned slot = 0, requested = 0;
        float temperature = 0.0F, top_p = 1.0F;
        int fields;
        char *prompt = NULL, *grammar = NULL;
        uint32_t *tokens = NULL;
        float *logits = NULL;
        size_t token_count = 0;
        coli_expert_backend experts;
        gemma4_serve_slot *serve_slot = NULL;
        coli_gemma4_model *model = NULL;
        coli_gemma4_sampler sampler;
        gemma4_serve_grammar request_grammar;
        uint32_t position, emitted = 0, generation_limit;
        size_t reused = 0;
        int length_limited = 0, accepted = 0, channel_header = 0;
        int step_status;
        gemma4_serve_control control;
        clock_t started;
        memset(&sampler, 0, sizeof(sampler));
        memset(&request_grammar, 0, sizeof(request_grammar));
        memset(&control, 0, sizeof(control));
        if (!fgets(header, sizeof(header), stdin)) {
            status = feof(stdin) ? 0 : 1;
            break;
        }
        fields = sscanf(header, "%15s %llu %u %llu %u %f %f %llu",
                        kind, &request_id, &slot, &payload_bytes_ull,
                        &requested, &temperature, &top_p, &grammar_bytes_ull);
        if (fields >= 2 && (!strcmp(kind, "STOP") ||
                           !strcmp(kind, "CANCEL"))) {
            printf("ERROR %llu NOT_FOUND\n", request_id);
            fflush(stdout);
            continue;
        }
        if (fields < 7 || strcmp(kind, "SUBMIT") || slot >= kv_slot_count ||
            !payload_bytes_ull || payload_bytes_ull > SIZE_MAX - 1 ||
            grammar_bytes_ull > SIZE_MAX - 1) {
            printf("ERROR %llu BAD_REQUEST\n", request_id);
            fflush(stdout);
            continue;
        }
        prompt = (char *)malloc((size_t)payload_bytes_ull + 1);
        if (grammar_bytes_ull)
            grammar = (char *)malloc((size_t)grammar_bytes_ull + 1);
        if (!prompt || (grammar_bytes_ull && !grammar) ||
            fread(prompt, 1, (size_t)payload_bytes_ull, stdin) !=
                (size_t)payload_bytes_ull ||
            (grammar_bytes_ull && fread(
                grammar, 1, (size_t)grammar_bytes_ull, stdin) !=
                (size_t)grammar_bytes_ull) || fgetc(stdin) != '\n') {
            free(prompt); free(grammar);
            status = 1;
            break;
        }
        prompt[(size_t)payload_bytes_ull] = '\0';
        if (grammar) grammar[(size_t)grammar_bytes_ull] = '\0';
        if (memchr(prompt, 0, (size_t)payload_bytes_ull) ||
            (grammar_bytes_ull && (memchr(grammar, 0,
                 (size_t)grammar_bytes_ull) ||
             serve_grammar_open(&request_grammar, grammar) != 0))) {
            printf("ERROR %llu %s\n", request_id,
                   grammar_bytes_ull ? "INVALID_GRAMMAR" : "BAD_REQUEST");
            fflush(stdout);
            goto request_cleanup;
        }
        tokens = (uint32_t *)malloc(
            ((size_t)maximum_context + 1) * sizeof(*tokens));
        logits = (float *)malloc((size_t)gguf.config.n_vocab * sizeof(*logits));
        if (!tokens || !logits || coli_gemma4_tokenize_ex(
                &tokenizer, prompt, (size_t)payload_bytes_ull, 0, 1,
                tokens, maximum_context + 1, &token_count) != 0 ||
            !token_count || token_count >= maximum_context) {
            printf("ERROR %llu CONTEXT_EXCEEDED %zu %u\n", request_id,
                   token_count, maximum_context - 1);
            fflush(stdout);
            goto request_cleanup;
        }
        generation_limit = requested;
        if (generation_limit > maximum_context - (uint32_t)token_count) {
            generation_limit = maximum_context - (uint32_t)token_count;
            length_limited = 1;
        }
        experts = coli_gemma4_expert_backend(&gemma);
        serve_slot = &slots[slot];
        model = &serve_slot->model;
        if (!serve_slot->opened) {
            serve_slot->tokens = (uint32_t *)malloc(
                (size_t)maximum_context * sizeof(*serve_slot->tokens));
            if (!serve_slot->tokens || coli_gemma4_model_open(
                    model, &gguf, &experts, maximum_context) != 0) {
                printf("ERROR %llu ENGINE_INIT %s\n", request_id,
                       coli_gemma4_model_last_error(model));
                fflush(stdout);
                goto request_cleanup;
            }
            serve_slot->opened = 1;
        }
        reused = serve_prefix_length(serve_slot, tokens, token_count);
        model->next_position = reused;
        serve_slot->token_count = reused;
        if (coli_gemma4_sampler_init(
                &sampler, temperature, gguf.sampling_top_k, top_p,
                request_id ? request_id : 1) != 0) {
            printf("ERROR %llu ENGINE_INIT %s\n", request_id,
                   coli_gemma4_model_last_error(model));
            fflush(stdout);
            goto request_cleanup;
        }
        printf("ACCEPT %llu %zu\n", request_id, token_count);
        fflush(stdout);
        accepted = 1;
        control.request_id = request_id;
        started = clock();
        for (position = (uint32_t)reused;
             position < (uint32_t)token_count; ++position) {
            int last = position + 1 == (uint32_t)token_count;
            step_status = coli_gemma4_model_step_cancelable(
                model, tokens[position], position, NULL,
                last ? logits : NULL, serve_model_cancelled, &control);
            if (step_status > 0 || control.action) goto request_interrupted;
            if (step_status < 0) goto engine_error;
            serve_slot->tokens[position] = tokens[position];
            serve_slot->token_count = (size_t)position + 1;
        }
        while (emitted < generation_limit) {
            uint32_t selected;
            size_t piece_bytes = 0;
            char *piece = NULL;
            if (serve_control_poll(&control) != 0) goto request_interrupted;
            if (serve_grammar_filter(
                    &tokenizer, &request_grammar, logits,
                    gguf.config.n_vocab) != 0 ||
                coli_gemma4_sample(
                    &sampler, logits, gguf.config.n_vocab,
                    &selected, NULL) != 0) goto engine_error;
            if (serve_grammar_accept_token(
                    &tokenizer, &request_grammar, selected) != 0)
                goto engine_error;
            if (coli_gemma4_token_is_eog(&tokenizer, selected)) break;
            if (coli_gemma4_decode_token(
                    &tokenizer, selected, NULL, 0, &piece_bytes) != 0)
                goto engine_error;
            if (piece_bytes) {
                piece = (char *)malloc(piece_bytes + 1);
                if (!piece || coli_gemma4_decode_token(
                        &tokenizer, selected, piece, piece_bytes + 1,
                        &piece_bytes) != 0) {
                    free(piece);
                    goto engine_error;
                }
            }
            if (piece && !strcmp(piece, "<|channel>"))
                channel_header = 1;
            else if (piece && !strcmp(piece, "<channel|>"))
                channel_header = 0;
            if (((!coli_gemma4_token_is_control(&tokenizer, selected) &&
                  !channel_header) ||
                 (piece && (!strcmp(piece, "<|tool_call>") ||
                            !strcmp(piece, "<tool_call|>")))) &&
                serve_data(request_id, piece, piece_bytes) != 0) {
                free(piece);
                goto engine_error;
            }
            free(piece);
            ++emitted;
            if (emitted < generation_limit) {
                uint64_t generated_position =
                    (uint64_t)token_count + emitted - 1;
                step_status = coli_gemma4_model_step_cancelable(
                    model, selected, generated_position, NULL, logits,
                    serve_model_cancelled, &control);
                if (step_status > 0 || control.action) goto request_interrupted;
                if (step_status < 0) goto engine_error;
                serve_slot->tokens[generated_position] = selected;
                serve_slot->token_count = (size_t)generated_position + 1;
            }
        }
        if (emitted == generation_limit && generation_limit == requested)
            length_limited = 1;
generation_done:
        {
            double seconds = (double)(clock() - started) / CLOCKS_PER_SEC;
            double reuse_percent = token_count ?
                100.0 * (double)reused / (double)token_count : 0.0;
            if (seconds < 0.001) seconds = 0.001;
            printf("DONE %llu STAT %u %.3f %.1f 0.00 %zu %d\n",
                   request_id, emitted, emitted / seconds, reuse_percent,
                   token_count, length_limited);
            fflush(stdout);
        }
        goto request_cleanup;

request_interrupted:
        if (control.action == 3) goto engine_error;
        if (control.action == 2) {
            printf("ERROR %llu CANCELLED\n", request_id);
            fflush(stdout);
            goto request_cleanup;
        }
        length_limited = 0;
        goto generation_done;

engine_error:
        if (accepted) {
            printf("ERROR %llu ENGINE_ERROR %s\n", request_id,
                   coli_gemma4_model_last_error(model));
            fflush(stdout);
        }
        if (serve_slot) serve_slot->token_count = 0;
request_cleanup:
        free(prompt); free(grammar); free(tokens); free(logits);
        serve_grammar_close(&request_grammar);
    }
cleanup:
    if (slots) for (slot_index = 0; slot_index < kv_slot_count; ++slot_index) {
        coli_gemma4_model_close(&slots[slot_index].model);
        free(slots[slot_index].tokens);
    }
    free(slots);
    coli_gemma4_close(&gemma);
    coli_gemma4_tokenizer_close(&tokenizer);
    coli_gemma4_gguf_close(&gguf);
    return status;
}

int main(int argc, char **argv) {
    if (argc == 2 && getenv("SERVE") && atoi(getenv("SERVE")) != 0)
        return command_serve(argv[1]);
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        puts("0.29.0-colibri-gemma4");
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "model-info") == 0)
        return command_model_info(argv[2]);
    if (argc >= 3 && strcmp(argv[1], "vision-info") == 0)
        return command_vision_info(argc - 2, argv + 2);
    if (argc == 4 && strcmp(argv[1], "vision-image-info") == 0)
        return command_vision_image_info(argv[2], argv[3]);
    if ((argc == 5 || argc == 7) &&
        strcmp(argv[1], "vision-patch-probe") == 0 &&
        (argc == 5 || strcmp(argv[5], "--output-f32") == 0))
        return command_vision_patch_probe(
            argv[2], argv[3], argv[4], argc == 7 ? argv[6] : NULL);
    if (argc >= 5 && strcmp(argv[1], "vision-encode-probe") == 0)
        return command_vision_encode_probe(argc - 2, argv + 2);
    if (argc >= 4 && strcmp(argv[1], "tokenize") == 0)
        return command_tokenize(argc - 4, argv + 4, argv[2], argv[3],
                                strlen(argv[3]), 0, 0);
    if (argc >= 4 && strcmp(argv[1], "tokenize-file") == 0)
        return command_tokenize_file(argc - 4, argv + 4, argv[2], argv[3], 0);
    if (argc >= 4 && strcmp(argv[1], "chat-user") == 0)
        return command_tokenize(argc - 4, argv + 4, argv[2], argv[3],
                                strlen(argv[3]), 1, 0);
    if (argc >= 4 && strcmp(argv[1], "chat-user-file") == 0)
        return command_tokenize_file(argc - 4, argv + 4,
                                     argv[2], argv[3], 1);
    if (argc >= 4 && strcmp(argv[1], "image-chat-user") == 0)
        return command_tokenize(argc - 4, argv + 4, argv[2], argv[3],
                                strlen(argv[3]), 1, 1);
    if (argc >= 4 && strcmp(argv[1], "embed") == 0)
        return command_embedding(argc - 4, argv + 4, argv[2], argv[3]);
    if (argc >= 3 && strcmp(argv[1], "lm-head") == 0)
        return command_lm_head(argc - 3, argv + 3, argv[2]);
    if (argc >= 5 && strcmp(argv[1], "next-token") == 0)
        return command_next_token(argc - 5, argv + 5, argv[2], argv[3],
                                  argv[4], strlen(argv[4]), 0);
    if (argc >= 5 && strcmp(argv[1], "next-token-file") == 0)
        return command_next_token_file(argc - 5, argv + 5, argv[2], argv[3],
                                       argv[4], 0);
    if (argc >= 5 && strcmp(argv[1], "generate") == 0)
        return command_next_token(argc - 5, argv + 5, argv[2], argv[3],
                                  argv[4], strlen(argv[4]), 1);
    if (argc >= 5 && strcmp(argv[1], "generate-file") == 0)
        return command_next_token_file(argc - 5, argv + 5, argv[2], argv[3],
                                       argv[4], 1);
    if (argc >= 4 && strcmp(argv[1], "chat") == 0)
        return command_chat(argc - 4, argv + 4, argv[2], argv[3]);
    if (argc >= 3 && strcmp(argv[1], "route") == 0)
        return command_route(argc - 3, argv + 3, argv[2]);
    if (argc >= 3 && strcmp(argv[1], "attention-proj") == 0)
        return command_attention_projection(argc - 3, argv + 3, argv[2]);
    if (argc >= 3 && strcmp(argv[1], "attention-seq") == 0)
        return command_attention_sequence(argc - 3, argv + 3, argv[2]);
    if (argc >= 3 && strcmp(argv[1], "dense-mlp") == 0)
        return command_dense_mlp(argc - 3, argv + 3, argv[2]);
    if (argc >= 4 && strcmp(argv[1], "routed-mlp") == 0)
        return command_routed_mlp(argc - 4, argv + 4, argv[2], argv[3]);
    if (argc >= 4 && strcmp(argv[1], "layer-seq") == 0)
        return command_layer_sequence(argc - 4, argv + 4, argv[2], argv[3]);
    if (argc >= 3 && strcmp(argv[1], "info") == 0)
        return command_info(argv[2]);
    if (argc >= 3 && strcmp(argv[1], "expert") == 0)
        return command_expert(argc - 3, argv + 3, argv[2]);
    usage(argv[0]);
    return 2;
}
