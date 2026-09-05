/* Tokenizer merge-table format gate.  tokenizer.json spells its merges two
 * ways: legacy files write "a b" strings, tokenizers >= 0.20 (transformers
 * 4.45+, the Qwen3.6 checkpoints included) writes ["a","b"] pairs.  The
 * string-only reader silently indexed ZERO merges from the pair form and
 * encode_text degraded to one token per byte-symbol -- observed as a 24-token
 * encoding of a 24-char prompt on the first real-model run, with the model
 * fed a token stream it never saw in training.  bpe_piece treats an empty
 * merge table as "nothing to merge", so nothing ever refused.
 *
 * Both spellings of the SAME tiny model must produce the SAME merged ids:
 * "in in" -> [in, Ġin], exercising both a plain merge (i+n) and a chained
 * one (Ġ + in, rank-ordered after i+n produces the intermediate). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <sys/stat.h>

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                           \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

static const char *tok_pairs =
    "{\"model\":{\"vocab\":{\"i\":0,\"n\":1,\"\\u0120\":2,\"in\":3,"
    "\"\\u0120in\":4},\"merges\":[[\"i\",\"n\"],[\"\\u0120\",\"in\"]]}}";
static const char *tok_strings =
    "{\"model\":{\"vocab\":{\"i\":0,\"n\":1,\"\\u0120\":2,\"in\":3,"
    "\"\\u0120in\":4},\"merges\":[\"i n\",\"\\u0120 in\"]}}";

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(text, 1, strlen(text), f) == strlen(text));
    CHECK(fclose(f) == 0);
}

static void check_encoding(const char *tok_path, const char *label) {
    load_tokenizer(tok_path);
    CHECK(g_tok != NULL && g_tok_n == 5);
    int *ids = NULL, n = 0;
    encode_text("in in", &ids, &n);
    if (n != 2 || ids[0] != 3 || ids[1] != 4) {
        fprintf(stderr, "%s: expected [3, 4], got %d ids:", label, n);
        for (int i = 0; i < n; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "\n");
        exit(1);
    }
    free(ids);
}

int main(void) {
    const char *dir = "tests/tmp_tok_merges";
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0700);
#endif
    char pairs_path[256], strings_path[256];
    snprintf(pairs_path, sizeof(pairs_path), "%s/tokenizer_pairs.json", dir);
    snprintf(strings_path, sizeof(strings_path), "%s/tokenizer_strings.json", dir);
    write_file(pairs_path, tok_pairs);
    write_file(strings_path, tok_strings);
    /* load_tokenizer is re-entrant enough for a test: smap_init re-allocates
     * fresh tables and neither fixture carries added_tokens. */
    check_encoding(strings_path, "string merges");
    check_encoding(pairs_path, "pair merges");
    printf("test_qwen36_tok_merges: OK\n");
    return 0;
}
