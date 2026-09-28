#include "gemma4_tokenizer.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct coli_gemma4_vocab_entry {
    const char *text;
    size_t length;
    uint32_t token;
    uint8_t used;
};

struct coli_gemma4_merge_entry {
    const char *left;
    const char *right;
    size_t left_length;
    size_t right_length;
    uint32_t rank;
    uint8_t used;
};

struct coli_gemma4_special_entry {
    const char *text;
    size_t length;
    uint32_t token;
};

static void tokenizer_error(coli_gemma4_tokenizer *tokenizer,
                            const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(tokenizer->last_error, sizeof(tokenizer->last_error),
              format, arguments);
    va_end(arguments);
}

static uint64_t hash_bytes(uint64_t hash, const char *text, size_t length) {
    size_t index;
    for (index = 0; index < length; ++index) {
        hash ^= (uint8_t)text[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t vocab_hash(const char *text, size_t length) {
    return hash_bytes(UINT64_C(1469598103934665603), text, length);
}

static uint64_t merge_hash(const char *left, size_t left_length,
                           const char *right, size_t right_length) {
    uint64_t hash = vocab_hash(left, left_length);
    const char separator = '\0';
    hash = hash_bytes(hash, &separator, 1);
    return hash_bytes(hash, right, right_length);
}

static size_t map_capacity(uint64_t count) {
    size_t capacity = 1;
    if (count > SIZE_MAX / 2) return 0;
    while (capacity < (size_t)count * 2) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity *= 2;
    }
    return capacity;
}

static void vocab_insert(coli_gemma4_tokenizer *tokenizer,
                         const char *text, size_t length, uint32_t token) {
    size_t slot = (size_t)(vocab_hash(text, length) &
                           (tokenizer->vocab_capacity - 1));
    while (tokenizer->vocab[slot].used)
        slot = (slot + 1) & (tokenizer->vocab_capacity - 1);
    tokenizer->vocab[slot].text = text;
    tokenizer->vocab[slot].length = length;
    tokenizer->vocab[slot].token = token;
    tokenizer->vocab[slot].used = 1;
}

static int vocab_find(const coli_gemma4_tokenizer *tokenizer,
                      const char *text, size_t length, uint32_t *token) {
    size_t slot = (size_t)(vocab_hash(text, length) &
                           (tokenizer->vocab_capacity - 1));
    while (tokenizer->vocab[slot].used) {
        const coli_gemma4_vocab_entry *entry = &tokenizer->vocab[slot];
        if (entry->length == length && !memcmp(entry->text, text, length)) {
            if (token) *token = entry->token;
            return 1;
        }
        slot = (slot + 1) & (tokenizer->vocab_capacity - 1);
    }
    return 0;
}

static void merge_insert(coli_gemma4_tokenizer *tokenizer,
                         const char *left, size_t left_length,
                         const char *right, size_t right_length, uint32_t rank) {
    size_t slot = (size_t)(merge_hash(left, left_length, right, right_length) &
                           (tokenizer->merge_capacity - 1));
    while (tokenizer->merges[slot].used)
        slot = (slot + 1) & (tokenizer->merge_capacity - 1);
    tokenizer->merges[slot].left = left;
    tokenizer->merges[slot].right = right;
    tokenizer->merges[slot].left_length = left_length;
    tokenizer->merges[slot].right_length = right_length;
    tokenizer->merges[slot].rank = rank;
    tokenizer->merges[slot].used = 1;
}

static int merge_find(const coli_gemma4_tokenizer *tokenizer,
                      const char *left, size_t left_length,
                      const char *right, size_t right_length, uint32_t *rank) {
    size_t slot = (size_t)(merge_hash(left, left_length, right, right_length) &
                           (tokenizer->merge_capacity - 1));
    while (tokenizer->merges[slot].used) {
        const coli_gemma4_merge_entry *entry = &tokenizer->merges[slot];
        if (entry->left_length == left_length &&
            entry->right_length == right_length &&
            !memcmp(entry->left, left, left_length) &&
            !memcmp(entry->right, right, right_length)) {
            *rank = entry->rank;
            return 1;
        }
        slot = (slot + 1) & (tokenizer->merge_capacity - 1);
    }
    return 0;
}

static int compare_specials(const void *left, const void *right) {
    const coli_gemma4_special_entry *a =
        (const coli_gemma4_special_entry *)left;
    const coli_gemma4_special_entry *b =
        (const coli_gemma4_special_entry *)right;
    if (a->length < b->length) return 1;
    if (a->length > b->length) return -1;
    return 0;
}

int coli_gemma4_tokenizer_init(coli_gemma4_tokenizer *tokenizer,
                               const coli_gemma4_gguf *gguf) {
    uint64_t index;
    if (!tokenizer || !gguf) return -1;
    memset(tokenizer, 0, sizeof(*tokenizer));
    tokenizer->gguf = gguf;
    if (strcmp(gguf->tokenizer_model, "gemma4") != 0 ||
        !gguf->tokenizer_tokens || !gguf->tokenizer_merges ||
        gguf->tokenizer_token_count != gguf->config.n_vocab ||
        gguf->tokenizer_bos_id >= gguf->tokenizer_token_count) {
        tokenizer_error(tokenizer, "GGUF is missing a valid Gemma 4 tokenizer");
        return -1;
    }
    tokenizer->vocab_capacity = map_capacity(gguf->tokenizer_token_count);
    tokenizer->merge_capacity = map_capacity(gguf->tokenizer_merge_count);
    if (!tokenizer->vocab_capacity || !tokenizer->merge_capacity) {
        tokenizer_error(tokenizer, "tokenizer tables are too large");
        return -1;
    }
    tokenizer->vocab = (coli_gemma4_vocab_entry *)calloc(
        tokenizer->vocab_capacity, sizeof(*tokenizer->vocab));
    tokenizer->merges = (coli_gemma4_merge_entry *)calloc(
        tokenizer->merge_capacity, sizeof(*tokenizer->merges));
    if (!tokenizer->vocab || !tokenizer->merges) {
        tokenizer_error(tokenizer, "out of memory building tokenizer tables");
        coli_gemma4_tokenizer_close(tokenizer);
        return -1;
    }
    for (index = 0; index < gguf->tokenizer_token_count; ++index) {
        const char *text = gguf->tokenizer_tokens[index];
        vocab_insert(tokenizer, text, strlen(text), (uint32_t)index);
    }
    for (index = 0; index < gguf->tokenizer_merge_count; ++index) {
        const char *merge = gguf->tokenizer_merges[index];
        const char *separator = strchr(merge + (merge[0] ? 1 : 0), ' ');
        if (!separator || !separator[1]) {
            tokenizer_error(tokenizer, "invalid tokenizer merge at index %llu",
                            (unsigned long long)index);
            coli_gemma4_tokenizer_close(tokenizer);
            return -1;
        }
        merge_insert(tokenizer, merge, (size_t)(separator - merge),
                     separator + 1, strlen(separator + 1), (uint32_t)index);
    }
    if (gguf->tokenizer_token_types &&
        gguf->tokenizer_token_type_count == gguf->tokenizer_token_count) {
        for (index = 0; index < gguf->tokenizer_token_type_count; ++index)
            if (gguf->tokenizer_token_types[index] == 3 ||
                gguf->tokenizer_token_types[index] == 4)
                ++tokenizer->special_count;
        tokenizer->specials = (coli_gemma4_special_entry *)calloc(
            tokenizer->special_count, sizeof(*tokenizer->specials));
        if (tokenizer->special_count && !tokenizer->specials) {
            tokenizer_error(tokenizer, "out of memory building special-token table");
            coli_gemma4_tokenizer_close(tokenizer);
            return -1;
        }
        tokenizer->special_count = 0;
        for (index = 0; index < gguf->tokenizer_token_type_count; ++index) {
            if (gguf->tokenizer_token_types[index] == 3 ||
                gguf->tokenizer_token_types[index] == 4) {
                coli_gemma4_special_entry *entry =
                    &tokenizer->specials[tokenizer->special_count++];
                entry->text = gguf->tokenizer_tokens[index];
                entry->length = strlen(entry->text);
                entry->token = (uint32_t)index;
            }
        }
        qsort(tokenizer->specials, tokenizer->special_count,
              sizeof(*tokenizer->specials), compare_specials);
    }
    return 0;
}

void coli_gemma4_tokenizer_close(coli_gemma4_tokenizer *tokenizer) {
    char error[COLI_GEMMA4_GGUF_ERROR_MAX];
    if (!tokenizer) return;
    memcpy(error, tokenizer->last_error, sizeof(error));
    free(tokenizer->vocab);
    free(tokenizer->merges);
    free(tokenizer->specials);
    memset(tokenizer, 0, sizeof(*tokenizer));
    memcpy(tokenizer->last_error, error, sizeof(error));
}

const char *coli_gemma4_tokenizer_last_error(
    const coli_gemma4_tokenizer *tokenizer) {
    return tokenizer ? tokenizer->last_error : "invalid tokenizer";
}

static size_t utf8_character_bytes(const char *text, size_t remaining) {
    uint8_t first = (uint8_t)text[0];
    if (first < 0x80) return 1;
    if ((first & 0xe0) == 0xc0 && remaining >= 2) return 2;
    if ((first & 0xf0) == 0xe0 && remaining >= 3) return 3;
    if ((first & 0xf8) == 0xf0 && remaining >= 4) return 4;
    return 1;
}

static int append_token(uint32_t token, uint32_t *tokens, size_t capacity,
                        size_t *count) {
    if (*count >= capacity) return -1;
    tokens[(*count)++] = token;
    return 0;
}

static int tokenize_segment(const coli_gemma4_tokenizer *tokenizer,
                            const char *text, size_t length, int newline_only,
                            uint32_t *tokens, size_t capacity, size_t *count) {
    size_t *offsets = NULL, *lengths = NULL;
    size_t symbols = 0, offset, index;
    uint32_t token;
    int result = -1;
    if (newline_only && vocab_find(tokenizer, text, length, &token))
        return append_token(token, tokens, capacity, count);
    offsets = (size_t *)malloc((length + 1) * sizeof(*offsets));
    lengths = (size_t *)malloc((length + 1) * sizeof(*lengths));
    if (!offsets || !lengths) goto cleanup;
    for (offset = 0; offset < length;) {
        size_t bytes = utf8_character_bytes(text + offset, length - offset);
        offsets[symbols] = offset;
        lengths[symbols++] = bytes;
        offset += bytes;
    }
    for (;;) {
        uint32_t best_rank = UINT32_MAX;
        size_t best = SIZE_MAX;
        for (index = 0; index + 1 < symbols; ++index) {
            uint32_t rank;
            if (merge_find(tokenizer, text + offsets[index], lengths[index],
                           text + offsets[index + 1], lengths[index + 1], &rank) &&
                rank < best_rank) {
                best_rank = rank;
                best = index;
            }
        }
        if (best == SIZE_MAX) break;
        lengths[best] += lengths[best + 1];
        for (index = best + 1; index + 1 < symbols; ++index) {
            offsets[index] = offsets[index + 1];
            lengths[index] = lengths[index + 1];
        }
        --symbols;
    }
    for (index = 0; index < symbols; ++index) {
        if (vocab_find(tokenizer, text + offsets[index], lengths[index], &token)) {
            if (append_token(token, tokens, capacity, count) != 0) goto cleanup;
        } else {
            size_t byte;
            static const char hex[] = "0123456789ABCDEF";
            for (byte = 0; byte < lengths[index]; ++byte) {
                uint8_t value = (uint8_t)text[offsets[index] + byte];
                char fallback[6] = {'<', '0', 'x', hex[value >> 4],
                                    hex[value & 15], '>'};
                if (!vocab_find(tokenizer, fallback, sizeof(fallback), &token) ||
                    append_token(token, tokens, capacity, count) != 0) goto cleanup;
            }
        }
    }
    result = 0;
cleanup:
    free(offsets);
    free(lengths);
    return result;
}

static int tokenize_text(const coli_gemma4_tokenizer *tokenizer,
                         const char *text, size_t text_bytes,
                         uint32_t *tokens, size_t capacity, size_t *count) {
    char *normalized;
    size_t input, output = 0, start;
    int result = -1;
    static const char escaped_space[] = "\xe2\x96\x81";
    if (text_bytes > (SIZE_MAX - 1) / 3) return -1;
    normalized = (char *)malloc(text_bytes * 3 + 1);
    if (!normalized) return -1;
    for (input = 0; input < text_bytes; ++input) {
        if (text[input] == ' ') {
            memcpy(normalized + output, escaped_space, 3);
            output += 3;
        } else {
            normalized[output++] = text[input];
        }
    }
    normalized[output] = '\0';
    for (start = 0; start < output;) {
        size_t end = start;
        int newline_only = normalized[start] == '\n';
        if (newline_only) {
            while (end < output && normalized[end] == '\n') ++end;
        } else {
            while (end < output && normalized[end] != '\n') ++end;
        }
        if (tokenize_segment(tokenizer, normalized + start, end - start,
                             newline_only, tokens, capacity, count) != 0)
            goto cleanup;
        start = end;
    }
    result = 0;
cleanup:
    free(normalized);
    return result;
}

int coli_gemma4_tokenize_ex(const coli_gemma4_tokenizer *tokenizer,
                            const char *text, size_t text_bytes, int add_bos,
                            int parse_special, uint32_t *tokens,
                            size_t capacity, size_t *count) {
    size_t position = 0, raw_start = 0;
    if (!tokenizer || !tokenizer->gguf || (!text && text_bytes) ||
        !tokens || !count) return -1;
    *count = 0;
    if (add_bos && tokenizer->gguf->tokenizer_add_bos &&
        append_token(tokenizer->gguf->tokenizer_bos_id,
                     tokens, capacity, count) != 0) return -1;
    if (!parse_special || !tokenizer->special_count)
        return tokenize_text(tokenizer, text, text_bytes,
                             tokens, capacity, count);
    while (position < text_bytes) {
        size_t special_index;
        const coli_gemma4_special_entry *match = NULL;
        for (special_index = 0; special_index < tokenizer->special_count;
             ++special_index) {
            const coli_gemma4_special_entry *candidate =
                &tokenizer->specials[special_index];
            if (candidate->length <= text_bytes - position &&
                !memcmp(text + position, candidate->text, candidate->length)) {
                match = candidate;
                break;
            }
        }
        if (!match) {
            ++position;
            continue;
        }
        if (position > raw_start &&
            tokenize_text(tokenizer, text + raw_start, position - raw_start,
                          tokens, capacity, count) != 0) return -1;
        if (append_token(match->token, tokens, capacity, count) != 0) return -1;
        position += match->length;
        raw_start = position;
    }
    return raw_start < text_bytes ?
        tokenize_text(tokenizer, text + raw_start, text_bytes - raw_start,
                      tokens, capacity, count) : 0;
}

int coli_gemma4_tokenize(const coli_gemma4_tokenizer *tokenizer,
                         const char *text, size_t text_bytes, int add_bos,
                         uint32_t *tokens, size_t capacity, size_t *count) {
    return coli_gemma4_tokenize_ex(tokenizer, text, text_bytes, add_bos, 0,
                                   tokens, capacity, count);
}

static void trim_chat_text(const char *text, size_t text_bytes,
                           size_t *begin, size_t *end) {
    *begin = 0;
    *end = text_bytes;
    while (*begin < *end &&
           (text[*begin] == ' ' || text[*begin] == '\t' ||
            text[*begin] == '\r' || text[*begin] == '\n'))
        ++*begin;
    while (*end > *begin &&
           (text[*end - 1] == ' ' || text[*end - 1] == '\t' ||
            text[*end - 1] == '\r' || text[*end - 1] == '\n'))
        --*end;
}

int coli_gemma4_tokenize_chat_turn_with_tools(
    const coli_gemma4_tokenizer *tokenizer,
    const char *system_text, size_t system_bytes,
    const char *rendered_tools, size_t rendered_tools_bytes,
    const char *user_text, size_t text_bytes, int initial_turn,
    uint32_t *tokens, size_t capacity, size_t *count) {
    static const char initial_prefix[] = "<bos><|turn>user\n";
    static const char system_prefix[] = "<bos><|turn>system\n";
    static const char user_after_system[] = "<turn|>\n<|turn>user\n";
    static const char continuation_prefix[] = "\n<|turn>user\n";
    static const char suffix[] =
        "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    const int has_system = system_text != NULL;
    const int has_tools = rendered_tools_bytes != 0;
    const char *prefix;
    size_t prefix_bytes;
    char *rendered;
    size_t begin, end, system_begin = 0, system_end = 0;
    size_t rendered_bytes, offset = 0, overhead;
    if (!tokenizer || !tokenizer->gguf || !tokens || !count) return -1;
    if (!user_text) {
        if (text_bytes) return -1;
        user_text = "";
    }
    if ((has_system || has_tools) && !initial_turn) return -1;
    if (!system_text && system_bytes) return -1;
    if (!rendered_tools && rendered_tools_bytes) return -1;
    trim_chat_text(user_text, text_bytes, &begin, &end);
    if (has_system || has_tools) {
        if (has_system)
            trim_chat_text(system_text, system_bytes,
                           &system_begin, &system_end);
        prefix = system_prefix;
        prefix_bytes = sizeof(system_prefix) - 1;
        overhead = prefix_bytes + (sizeof(user_after_system) - 1) +
                   (sizeof(suffix) - 1);
        if (system_end - system_begin > SIZE_MAX - overhead) return -1;
        overhead += system_end - system_begin;
        if (rendered_tools_bytes > SIZE_MAX - overhead) return -1;
        overhead += rendered_tools_bytes;
    } else {
        prefix = initial_turn ? initial_prefix : continuation_prefix;
        prefix_bytes = initial_turn ? sizeof(initial_prefix) - 1 :
                                     sizeof(continuation_prefix) - 1;
        overhead = prefix_bytes + (sizeof(suffix) - 1);
    }
    if (overhead >= SIZE_MAX ||
        end - begin > SIZE_MAX - overhead - 1) return -1;
    rendered_bytes = overhead + (end - begin);
    rendered = (char *)malloc(rendered_bytes + 1);
    if (!rendered) return -1;
    memcpy(rendered + offset, prefix, prefix_bytes);
    offset += prefix_bytes;
    if (has_system || has_tools) {
        if (has_system) {
            memcpy(rendered + offset, system_text + system_begin,
                   system_end - system_begin);
            offset += system_end - system_begin;
        }
        if (has_tools) {
            memcpy(rendered + offset, rendered_tools, rendered_tools_bytes);
            offset += rendered_tools_bytes;
        }
        memcpy(rendered + offset, user_after_system,
               sizeof(user_after_system) - 1);
        offset += sizeof(user_after_system) - 1;
    }
    memcpy(rendered + offset, user_text + begin, end - begin);
    offset += end - begin;
    memcpy(rendered + offset, suffix, sizeof(suffix) - 1);
    rendered[rendered_bytes] = '\0';
    {
        int result = coli_gemma4_tokenize_ex(
            tokenizer, rendered, rendered_bytes, 0, 1,
            tokens, capacity, count);
        free(rendered);
        return result;
    }
}

int coli_gemma4_tokenize_chat_turn(const coli_gemma4_tokenizer *tokenizer,
                                   const char *system_text,
                                   size_t system_bytes,
                                   const char *user_text, size_t text_bytes,
                                   int initial_turn, uint32_t *tokens,
                                   size_t capacity, size_t *count) {
    return coli_gemma4_tokenize_chat_turn_with_tools(
        tokenizer, system_text, system_bytes, NULL, 0,
        user_text, text_bytes, initial_turn, tokens, capacity, count);
}

int coli_gemma4_tokenize_image_chat_turn_with_tools(
    const coli_gemma4_tokenizer *tokenizer,
    const char *system_text, size_t system_bytes,
    const char *rendered_tools, size_t rendered_tools_bytes,
    const char *user_text, size_t text_bytes, int initial_turn,
    uint32_t *prefix_tokens, size_t prefix_capacity, size_t *prefix_count,
    uint32_t *suffix_tokens, size_t suffix_capacity, size_t *suffix_count) {
    static const char image_open[] = "<|image>";
    static const char image_close[] = "<image|>";
    char *image_user = NULL;
    uint32_t marker_tokens[2], *all_tokens = NULL;
    size_t marker_count = 0, all_count = 0, capacity, index;
    size_t image_user_bytes;
    const size_t marker_bytes = sizeof(image_open) + sizeof(image_close) - 2;
    int result = -1;
    if (!tokenizer || !prefix_tokens || !prefix_count ||
        !suffix_tokens || !suffix_count ||
        (!user_text && text_bytes) ||
        prefix_capacity > SIZE_MAX - suffix_capacity ||
        text_bytes > SIZE_MAX - marker_bytes - 1)
        return -1;
    *prefix_count = 0;
    *suffix_count = 0;
    if (!user_text) user_text = "";
    image_user_bytes = marker_bytes + text_bytes;
    image_user = (char *)malloc(image_user_bytes + 1);
    if (!image_user) goto cleanup;
    memcpy(image_user, image_open, sizeof(image_open) - 1);
    memcpy(image_user + sizeof(image_open) - 1,
           image_close, sizeof(image_close) - 1);
    memcpy(image_user + sizeof(image_open) + sizeof(image_close) - 2,
           user_text, text_bytes);
    image_user[image_user_bytes] = '\0';
    if (coli_gemma4_tokenize_ex(
            tokenizer, image_open, sizeof(image_open) - 1, 0, 1,
            marker_tokens, 2, &marker_count) != 0 || marker_count != 1 ||
        coli_gemma4_tokenize_ex(
            tokenizer, image_close, sizeof(image_close) - 1, 0, 1,
            marker_tokens + 1, 1, &marker_count) != 0 || marker_count != 1)
        goto cleanup;
    capacity = prefix_capacity + suffix_capacity;
    if (!capacity || capacity > SIZE_MAX / sizeof(*all_tokens)) goto cleanup;
    all_tokens = (uint32_t *)malloc(capacity * sizeof(*all_tokens));
    if (!all_tokens) goto cleanup;
    if (coli_gemma4_tokenize_chat_turn_with_tools(
            tokenizer, system_text, system_bytes,
            rendered_tools, rendered_tools_bytes,
            image_user, image_user_bytes, initial_turn,
            all_tokens, capacity, &all_count) != 0)
        goto cleanup;
    for (index = 0; index + 1 < all_count; ++index) {
        size_t before_count, after_count;
        if (all_tokens[index] != marker_tokens[0] ||
            all_tokens[index + 1] != marker_tokens[1]) continue;
        before_count = index + 1;
        after_count = all_count - before_count;
        if (before_count > prefix_capacity || after_count > suffix_capacity)
            goto cleanup;
        memcpy(prefix_tokens, all_tokens,
               before_count * sizeof(*prefix_tokens));
        memcpy(suffix_tokens, all_tokens + before_count,
               after_count * sizeof(*suffix_tokens));
        *prefix_count = before_count;
        *suffix_count = after_count;
        result = 0;
        break;
    }
cleanup:
    free(all_tokens);
    free(image_user);
    return result;
}

int coli_gemma4_tokenize_user_turn(const coli_gemma4_tokenizer *tokenizer,
                                   const char *user_text, size_t text_bytes,
                                   int initial_turn, uint32_t *tokens,
                                   size_t capacity, size_t *count) {
    return coli_gemma4_tokenize_chat_turn(
        tokenizer, NULL, 0, user_text, text_bytes, initial_turn,
        tokens, capacity, count);
}

int coli_gemma4_tokenize_user_chat(const coli_gemma4_tokenizer *tokenizer,
                                   const char *user_text, size_t text_bytes,
                                   uint32_t *tokens, size_t capacity,
                                   size_t *count) {
    return coli_gemma4_tokenize_user_turn(
        tokenizer, user_text, text_bytes, 1, tokens, capacity, count);
}

static int hex_digit(unsigned char value) {
    if (value >= '0' && value <= '9') return (int)(value - '0');
    if (value >= 'A' && value <= 'F') return (int)(value - 'A') + 10;
    if (value >= 'a' && value <= 'f') return (int)(value - 'a') + 10;
    return -1;
}

int coli_gemma4_decode_token(const coli_gemma4_tokenizer *tokenizer,
                             uint32_t token, char *text, size_t capacity,
                             size_t *text_bytes) {
    const char *piece;
    size_t input = 0, output = 0;
    if (!tokenizer || !tokenizer->gguf || !text_bytes ||
        token >= tokenizer->gguf->tokenizer_token_count) return -1;
    piece = tokenizer->gguf->tokenizer_tokens[token];
    if (!piece) return -1;
    if (strlen(piece) == 6 && !memcmp(piece, "<0x", 3) && piece[5] == '>') {
        int high = hex_digit((unsigned char)piece[3]);
        int low = hex_digit((unsigned char)piece[4]);
        if (high >= 0 && low >= 0) {
            *text_bytes = 1;
            if (!text) return 0;
            if (capacity < 2) return -1;
            text[0] = (char)((high << 4) | low);
            text[1] = '\0';
            return 0;
        }
    }
    while (piece[input]) {
        if (piece[input + 1] && piece[input + 2] &&
            (unsigned char)piece[input] == 0xe2U &&
            (unsigned char)piece[input + 1] == 0x96U &&
            (unsigned char)piece[input + 2] == 0x81U) {
            ++output;
            input += 3;
        } else {
            ++output;
            ++input;
        }
    }
    *text_bytes = output;
    if (!text) return 0;
    if (capacity <= output) return -1;
    input = 0;
    output = 0;
    while (piece[input]) {
        if (piece[input + 1] && piece[input + 2] &&
            (unsigned char)piece[input] == 0xe2U &&
            (unsigned char)piece[input + 1] == 0x96U &&
            (unsigned char)piece[input + 2] == 0x81U) {
            text[output++] = ' ';
            input += 3;
        } else {
            text[output++] = piece[input++];
        }
    }
    text[output] = '\0';
    return 0;
}

int coli_gemma4_token_is_control(const coli_gemma4_tokenizer *tokenizer,
                                 uint32_t token) {
    uint32_t type;
    if (!tokenizer || !tokenizer->gguf ||
        token >= tokenizer->gguf->tokenizer_token_type_count ||
        !tokenizer->gguf->tokenizer_token_types) return 0;
    type = tokenizer->gguf->tokenizer_token_types[token];
    return type == 3 || type == 4;
}

int coli_gemma4_token_is_eog(const coli_gemma4_tokenizer *tokenizer,
                             uint32_t token) {
    const char *piece;
    if (!tokenizer || !tokenizer->gguf ||
        token >= tokenizer->gguf->tokenizer_token_count) return 0;
    if (token == tokenizer->gguf->tokenizer_eos_id) return 1;
    piece = tokenizer->gguf->tokenizer_tokens[token];
    return piece && (!strcmp(piece, "<turn|>") ||
                     !strcmp(piece, "<|tool_response>"));
}

int coli_gemma4_token_is_tool_response(const coli_gemma4_tokenizer *tokenizer,
                                       uint32_t token) {
    const char *piece;
    if (!tokenizer || !tokenizer->gguf ||
        token >= tokenizer->gguf->tokenizer_token_count) return 0;
    piece = tokenizer->gguf->tokenizer_tokens[token];
    return piece && !strcmp(piece, "<|tool_response>");
}
