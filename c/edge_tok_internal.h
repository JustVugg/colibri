#ifndef COLIBRI_EDGE_TOK_INTERNAL_H
#define COLIBRI_EDGE_TOK_INTERNAL_H

/* Include after tok.h. Kept separate because Qwen3.6 owns a different
 * production tokenizer whose internal BPE symbol names intentionally overlap
 * with tok.h. */

#include "edge_adapter_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Byte-level BPE cannot emit more IDs than input bytes: every merge reduces
 * the count and an added token replaces at least one byte. A private temporary
 * makes the public two-pass API exact without depending on tok_encode's
 * caller-capacity truncation semantics. */
static int coli_edge_tok_tokenize(
    Tok *tokenizer, const char *text, size_t text_bytes,
    int32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error, size_t error_size) {
    if (!tokenizer || !text || !token_count || text_bytes > INT_MAX)
        return coli_edge_adapter_error(error, error_size,
                                       "tokenizer input is too large");
    size_t capacity = text_bytes ? text_bytes : 1;
    if (capacity > INT_MAX)
        return coli_edge_adapter_error(error, error_size,
                                       "tokenizer output is too large");
    int *temporary = malloc(capacity * sizeof(*temporary));
    if (!temporary)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory tokenizing text");
    int count = tok_encode(tokenizer, text, (int)text_bytes,
                           temporary, (int)capacity);
    if (count < 0 || (size_t)count > capacity) {
        free(temporary);
        return coli_edge_adapter_error(error, error_size,
                                       "tokenizer returned an invalid count");
    }
    *token_count = (size_t)count;
    if (token_ids && token_capacity < (size_t)count) {
        free(temporary);
        return coli_edge_adapter_error(error, error_size,
                                       "token output buffer is too small");
    }
    if (token_ids)
        for (int item = 0; item < count; item++) token_ids[item] = temporary[item];
    free(temporary);
    return 0;
}

static int coli_edge_tok_detokenize(
    Tok *tokenizer, const int32_t *token_ids, size_t token_count,
    char *text, size_t text_capacity, size_t *text_bytes,
    char *error, size_t error_size) {
    if (!tokenizer || !token_ids || !token_count || !text_bytes ||
        token_count > INT_MAX)
        return coli_edge_adapter_error(error, error_size,
                                       "invalid detokenizer input");
    int *ids = malloc(token_count * sizeof(*ids));
    if (!ids)
        return coli_edge_adapter_error(error, error_size,
                                       "out of memory detokenizing tokens");
    for (size_t item = 0; item < token_count; item++) ids[item] = token_ids[item];

    size_t capacity = token_count < 16 ? 256 :
        (token_count > (size_t)INT_MAX / 16u ? (size_t)INT_MAX :
         token_count * 16u);
    char *temporary = NULL;
    int count = 0;
    for (;;) {
        temporary = malloc(capacity + 1);
        if (!temporary) {
            free(ids);
            return coli_edge_adapter_error(error, error_size,
                                           "out of memory detokenizing tokens");
        }
        count = tok_decode(tokenizer, ids, (int)token_count,
                           temporary, (int)capacity);
        if (count < (int)capacity || capacity == INT_MAX) break;
        free(temporary); temporary = NULL;
        if (capacity > (size_t)INT_MAX / 2u) capacity = INT_MAX;
        else capacity *= 2u;
    }
    free(ids);
    if (count < 0 || count == INT_MAX) {
        free(temporary);
        return coli_edge_adapter_error(error, error_size,
                                       "detokenized text is too large");
    }
    *text_bytes = (size_t)count;
    if (text && text_capacity < (size_t)count + 1u) {
        free(temporary);
        return coli_edge_adapter_error(error, error_size,
                                       "text output buffer is too small");
    }
    if (text) {
        memcpy(text, temporary, (size_t)count);
        text[count] = '\0';
    }
    free(temporary);
    return 0;
}

#endif /* COLIBRI_EDGE_TOK_INTERNAL_H */
