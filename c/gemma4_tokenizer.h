#ifndef COLI_GEMMA4_TOKENIZER_H
#define COLI_GEMMA4_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "gemma4_gguf.h"

typedef struct coli_gemma4_vocab_entry coli_gemma4_vocab_entry;
typedef struct coli_gemma4_merge_entry coli_gemma4_merge_entry;
typedef struct coli_gemma4_special_entry coli_gemma4_special_entry;

typedef struct {
    const coli_gemma4_gguf *gguf;
    coli_gemma4_vocab_entry *vocab;
    coli_gemma4_merge_entry *merges;
    coli_gemma4_special_entry *specials;
    size_t vocab_capacity;
    size_t merge_capacity;
    size_t special_count;
    char last_error[COLI_GEMMA4_GGUF_ERROR_MAX];
} coli_gemma4_tokenizer;

int coli_gemma4_tokenizer_init(coli_gemma4_tokenizer *tokenizer,
                               const coli_gemma4_gguf *gguf);
void coli_gemma4_tokenizer_close(coli_gemma4_tokenizer *tokenizer);
const char *coli_gemma4_tokenizer_last_error(
    const coli_gemma4_tokenizer *tokenizer);
int coli_gemma4_tokenize(const coli_gemma4_tokenizer *tokenizer,
                         const char *text, size_t text_bytes, int add_bos,
                         uint32_t *tokens, size_t capacity, size_t *count);
int coli_gemma4_tokenize_ex(const coli_gemma4_tokenizer *tokenizer,
                            const char *text, size_t text_bytes, int add_bos,
                            int parse_special, uint32_t *tokens,
                            size_t capacity, size_t *count);
int coli_gemma4_tokenize_user_chat(const coli_gemma4_tokenizer *tokenizer,
                                   const char *user_text, size_t text_bytes,
                                   uint32_t *tokens, size_t capacity,
                                   size_t *count);
int coli_gemma4_tokenize_user_turn(const coli_gemma4_tokenizer *tokenizer,
                                   const char *user_text, size_t text_bytes,
                                   int initial_turn, uint32_t *tokens,
                                   size_t capacity, size_t *count);
int coli_gemma4_tokenize_chat_turn(const coli_gemma4_tokenizer *tokenizer,
                                   const char *system_text,
                                   size_t system_bytes,
                                   const char *user_text, size_t text_bytes,
                                   int initial_turn, uint32_t *tokens,
                                   size_t capacity, size_t *count);
int coli_gemma4_tokenize_chat_turn_with_tools(
    const coli_gemma4_tokenizer *tokenizer,
    const char *system_text, size_t system_bytes,
    const char *rendered_tools, size_t rendered_tools_bytes,
    const char *user_text, size_t text_bytes, int initial_turn,
    uint32_t *tokens, size_t capacity, size_t *count);
int coli_gemma4_tokenize_image_chat_turn_with_tools(
    const coli_gemma4_tokenizer *tokenizer,
    const char *system_text, size_t system_bytes,
    const char *rendered_tools, size_t rendered_tools_bytes,
    const char *user_text, size_t text_bytes, int initial_turn,
    uint32_t *prefix_tokens, size_t prefix_capacity, size_t *prefix_count,
    uint32_t *suffix_tokens, size_t suffix_capacity, size_t *suffix_count);
int coli_gemma4_decode_token(const coli_gemma4_tokenizer *tokenizer,
                             uint32_t token, char *text, size_t capacity,
                             size_t *text_bytes);
int coli_gemma4_token_is_control(const coli_gemma4_tokenizer *tokenizer,
                                 uint32_t token);
int coli_gemma4_token_is_eog(const coli_gemma4_tokenizer *tokenizer,
                             uint32_t token);
int coli_gemma4_token_is_tool_response(const coli_gemma4_tokenizer *tokenizer,
                                       uint32_t token);

#endif
