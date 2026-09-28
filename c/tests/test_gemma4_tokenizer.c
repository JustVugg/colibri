#include "gemma4_tokenizer.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char *tokens[] = {
        "<pad>", "<eos>", "<bos>", "<unk>", "h", "e", "l", "o",
        "\xe2\x96\x81", "he", "hel", "hell", "hello", "\n\n",
        "<0xF0>", "<0x9F>", "<0x99>", "<0x82>",
        "<|turn>", "<turn|>", "<|channel>", "<channel|>",
        "s", "y", "t", "m", "u", "r", "d", "g", "\n",
        "<|image>", "<image|>"
    };
    char *merges[] = {"h e", "he l", "hel l", "hell o"};
    uint32_t token_types[] = {
        3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3
    };
    const uint32_t expected[] = {2, 12, 8, 12, 13, 14, 15, 16, 17};
    uint32_t actual[128];
    size_t count = 0, index;
    coli_gemma4_gguf gguf;
    coli_gemma4_tokenizer tokenizer;
    int failed = 0;
    memset(&gguf, 0, sizeof(gguf));
    memcpy(gguf.tokenizer_model, "gemma4", sizeof("gemma4"));
    gguf.config.n_vocab = (uint32_t)(sizeof(tokens) / sizeof(tokens[0]));
    gguf.tokenizer_tokens = tokens;
    gguf.tokenizer_token_count = sizeof(tokens) / sizeof(tokens[0]);
    gguf.tokenizer_merges = merges;
    gguf.tokenizer_merge_count = sizeof(merges) / sizeof(merges[0]);
    gguf.tokenizer_token_types = token_types;
    gguf.tokenizer_token_type_count = sizeof(token_types) / sizeof(token_types[0]);
    gguf.tokenizer_bos_id = 2;
    gguf.tokenizer_add_bos = 1;
    if (coli_gemma4_tokenizer_init(&tokenizer, &gguf) != 0) {
        fprintf(stderr, "%s\n", coli_gemma4_tokenizer_last_error(&tokenizer));
        return 1;
    }
    if (coli_gemma4_tokenize(&tokenizer, "hello hello\n\n\xf0\x9f\x99\x82",
                             strlen("hello hello\n\n\xf0\x9f\x99\x82"),
                             1, actual, 32, &count) != 0 ||
        count != sizeof(expected) / sizeof(expected[0])) {
        failed = 1;
    }
    for (index = 0; !failed && index < count; ++index)
        if (actual[index] != expected[index]) failed = 1;
    if (!failed) {
        const uint32_t special_expected[] = {2, 18, 19, 20, 21};
        const char *special_text =
            "<bos><|turn><turn|><|channel><channel|>";
        if (coli_gemma4_tokenize_ex(&tokenizer, special_text,
                                    strlen(special_text), 0, 1,
                                    actual, 32, &count) != 0 ||
            count != sizeof(special_expected) / sizeof(special_expected[0])) {
            failed = 1;
        }
        for (index = 0; !failed && index < count; ++index)
            if (actual[index] != special_expected[index]) failed = 1;
    }
    if (!failed) {
        static const char rendered[] =
            "<bos><|turn>system\nhello<turn|>\n<|turn>user\nhello"
            "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        uint32_t expected_system[128];
        size_t expected_count = 0;
        if (coli_gemma4_tokenize_ex(
                &tokenizer, rendered, sizeof(rendered) - 1, 0, 1,
                expected_system, 128, &expected_count) != 0 ||
            coli_gemma4_tokenize_chat_turn(
                &tokenizer, "  hello\r\n", strlen("  hello\r\n"),
                "\thello ", strlen("\thello "), 1,
                actual, 128, &count) != 0 || count != expected_count) {
            failed = 1;
        }
        for (index = 0; !failed && index < count; ++index)
            if (actual[index] != expected_system[index]) failed = 1;
        if (!failed && coli_gemma4_tokenize_chat_turn(
                &tokenizer, "hello", 5, "hello", 5, 0,
                actual, 32, &count) == 0) failed = 1;
    }
    if (!failed) {
        char piece[16];
        size_t piece_bytes = 0;
        if (coli_gemma4_decode_token(&tokenizer, 8, piece, sizeof(piece),
                                     &piece_bytes) != 0 ||
            piece_bytes != 1 || strcmp(piece, " ") != 0 ||
            coli_gemma4_decode_token(&tokenizer, 14, piece, sizeof(piece),
                                     &piece_bytes) != 0 ||
            piece_bytes != 1 || (unsigned char)piece[0] != 0xf0U ||
            !coli_gemma4_token_is_control(&tokenizer, 19) ||
            !coli_gemma4_token_is_eog(&tokenizer, 19) ||
            coli_gemma4_token_is_eog(&tokenizer, 18)) failed = 1;
    }
    if (!failed) {
        uint32_t prefix[64], suffix[64], combined[128], expected_image[128];
        size_t prefix_count = 0, suffix_count = 0, expected_image_count = 0;
        static const char image_rendered[] =
            "<bos><|turn>user\n<|image><image|>hello<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>";
        if (coli_gemma4_tokenize_image_chat_turn_with_tools(
                &tokenizer, NULL, 0, NULL, 0, "hello", 5, 1,
                prefix, 64, &prefix_count, suffix, 64, &suffix_count) != 0 ||
            !prefix_count || !suffix_count || prefix[prefix_count-1] != 31 ||
            suffix[0] != 32 ||
            coli_gemma4_tokenize_ex(
                &tokenizer, image_rendered, sizeof(image_rendered)-1, 0, 1,
                expected_image, 128, &expected_image_count) != 0 ||
            prefix_count + suffix_count != expected_image_count) {
            failed = 1;
        }
        memcpy(combined,prefix,prefix_count*sizeof(*combined));
        memcpy(combined+prefix_count,suffix,suffix_count*sizeof(*combined));
        for(index=0;!failed&&index<expected_image_count;++index)
            if(combined[index]!=expected_image[index])failed=1;
    }
    if (failed) {
        fprintf(stderr, "Gemma 4 tokenizer output mismatch (count=%zu)\n", count);
        for (index = 0; index < count; ++index) fprintf(stderr, " %u", actual[index]);
        fputc('\n', stderr);
    }
    coli_gemma4_tokenizer_close(&tokenizer);
    if (!failed) puts("Gemma 4 tokenizer tests passed");
    return failed;
}
