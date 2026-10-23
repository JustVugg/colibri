/* #1653: an added token that directly follows punctuation (`X.<|im_end|>`) must
 * still be one token. HF's tokenizers split added tokens out first and run the
 * regex pre-tokenizer on the ordinary text between them; encode_text() used to
 * look for the special only at the start of each piece, and the punctuation
 * rule had already swallowed the `<|`. Pins the split on a tiny tokenizer.json
 * so it needs no model: every added token boundary, the whitespace lookahead
 * at that boundary, and back-to-back specials. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int g_fails = 0;
static void expect(const char *text, const int *want, int nwant){
    int *ids = NULL, n = 0;
    encode_text(text, &ids, &n);
    int ok = (n == nwant);
    for (int i = 0; ok && i < n; i++) ok = (ids[i] == want[i]);
    if (!ok) {
        g_fails++;
        printf("FAIL: %-24s got [", text);
        for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", ids[i]);
        printf("] want [");
        for (int i = 0; i < nwant; i++) printf("%s%d", i ? " " : "", want[i]);
        printf("]\n");
    }
    free(ids);
}
#define EXPECT(text, ...) do { int w[] = {__VA_ARGS__}; expect(text, w, (int)(sizeof w / sizeof w[0])); } while (0)

int main(void){
    /* Byte-level BPE vocabulary: printable ASCII maps to itself, U+0120 (\xC4\xA0)
     * is the space, U+010A (\xC4\x8A) the newline. One merge, the repeated space,
     * as in the real Qwen table. Id 7 is the only added token. */
    const char *path = "test_qwen36_tokenizer.json";
    const char *json =
        "{\"model\":{\"vocab\":{\"X\":0,\".\":1,\"<\":2,\"|\":3,\"i\":4,\"m\":5,\"Y\":6,"
        "\"}\":8,\"\\\"\":9,\"\xC4\x8A\":10,\"\xC4\xA0\":11,\"\xC4\xA0\xC4\xA0\":12,"
        "\"_\":13,\"e\":14,\"n\":15,\"d\":16,\">\":17,\"!\":18,\"\xC4\xA0Y\":19},"
        "\"merges\":[\"\xC4\xA0 \xC4\xA0\",\"\xC4\xA0 Y\"]},"
        "\"added_tokens\":[{\"id\":7,\"content\":\"<|im_end|>\",\"special\":true}]}";
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(json, 1, strlen(json), f) != strlen(json) || fclose(f)) { printf("FAIL: cannot write %s\n", path); return 1; }
    load_tokenizer(path);
    remove(path);

    EXPECT("X",            0);
    EXPECT("X<|im_end|>",  0, 7);
    /* the reporter's cases: punctuation right before the special, +4 tokens before the fix */
    EXPECT("X.",           0, 1);
    EXPECT("X.<|im_end|>", 0, 1, 7);
    EXPECT("X}<|im_end|>", 0, 8, 7);
    EXPECT("X\"<|im_end|>", 0, 9, 7);
    EXPECT("X!<|im_end|>", 0, 18, 7);
    EXPECT("X.\n<|im_end|>", 0, 1, 10, 7);
    /* the special ends the ordinary text: `\s+(?!\S)` must see end-of-input there,
     * so two spaces stay one piece (and one merged token), as in HF */
    EXPECT("X <|im_end|>",  0, 11, 7);
    EXPECT("X  <|im_end|>", 0, 12, 7);
    /* the whitespace rules themselves, with and without a special after them:
     * `\s+(?!\S)` leaves the last space to the next piece, `\s*[\r\n]+` stops at
     * the last newline, a lone space before text is that text's prefix */
    EXPECT("  Y",           11, 19);
    EXPECT("  Y<|im_end|>", 11, 19, 7);
    EXPECT("X  ",           0, 12);
    EXPECT("\n  Y",         10, 11, 19);
    EXPECT("X\n\n<|im_end|>", 0, 10, 10, 7);
    /* back to back, and text on both sides */
    EXPECT("<|im_end|><|im_end|>", 7, 7);
    EXPECT("X.<|im_end|>Y", 0, 1, 7, 6);
    EXPECT("<|im_end|>.X",  7, 1, 0);
    /* the special spelled out as ordinary text is NOT the special: it must stay text */
    EXPECT("<|im_en", 2, 3, 4, 5, 13, 14, 15);

    if (g_fails) { printf("test_qwen36_tokenizer: %d failure(s)\n", g_fails); return 1; }
    printf("OK test_qwen36_tokenizer: added tokens split before the pre-tokenizer (#1653)\n");
    return 0;
}
