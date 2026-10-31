/* o200k's pre-tokenizer in qwen36.c (o200k_encode_span, with tiktoken's whole-piece rule):
 * Muse Spark's tokenizer (l4_200k_base, a GGUF's tokenizer.ggml.pre "gpt-4o"). A byte-level
 * BPE of 512 pieces trained with o200k's Split (tools/gguf_import.py O200K_SPLITS, BPE
 * ignore_merges) and Muse Spark's message specials, and 102 texts whose ids HF's tokenizers
 * gave (tests/fixtures/o200k_tok_tiny.json, tests/fixtures/o200k_tok_cases.txt: hex of the
 * UTF-8, a tab, the ids): Flint's fixture texts, the chat format, paths and slashes,
 * contractions, numbers, CJK, accents, whitespace, code, emoji. The same texts through the
 * release's vocabulary, rebuilt from unsloth's GGUF header by gguf_import.py, give HF's ids
 * too (that file is 11 MB): test_qwen36_tokenizer_o200k TOKENIZER.json CASES.txt. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

int main(int argc, char **argv) {
    const char *tj = argc > 2 ? argv[1] : "tests/fixtures/o200k_tok_tiny.json";
    const char *cs = argc > 2 ? argv[2] : "tests/fixtures/o200k_tok_cases.txt";
    load_tokenizer(tj);
    if (!g_pretok_o200k || !g_ignore_merges) { printf("FAIL: o200k's pre-tokenizer or its whole-piece rule not recognized\n"); return 1; }
    FILE *f = fopen(cs, "rb");
    if (!f) { printf("FAIL: %s\n", cs); return 1; }
    char line[16384]; int cases = 0, fails = 0;
    while (fgets(line, sizeof line, f)) {
        char *tab = strchr(line, '\t'); if (!tab) continue;
        *tab = 0;
        size_t hl = strlen(line); char *text = malloc(hl / 2 + 1);
        for (size_t i = 0; i + 1 < hl; i += 2) { unsigned v; sscanf(line + i, "%2x", &v); text[i / 2] = (char)v; }
        text[hl / 2] = 0;
        int want[2048], nw = 0; char *p = tab + 1;
        while (*p && *p != '\n' && *p != '\r' && nw < 2048) {   /* a CRLF checkout (Windows) ends the ids at \r */
            char *e; long v = strtol(p, &e, 10);
            if (e == p) break;
            want[nw++] = (int)v; p = e; while (*p == ' ') p++;
        }
        int *ids = NULL, n = 0;
        encode_text(text, &ids, &n);
        int ok = n == nw;
        for (int i = 0; ok && i < n; i++) ok = ids[i] == want[i];
        if (!ok && ++fails <= 10) {
            printf("FAIL: case %d (%s): got", cases, line);
            for (int i = 0; i < n; i++) printf(" %d", ids[i]);
            printf(" want"); for (int i = 0; i < nw; i++) printf(" %d", want[i]); printf("\n");
        }
        free(ids); free(text); cases++;
    }
    fclose(f);
    if (fails || cases < 100) { printf("test_qwen36_tokenizer_o200k: %d of %d failed\n", fails, cases); return 1; }
    printf("OK test_qwen36_tokenizer_o200k: %d texts as HF's tokenizers splits them with o200k's pre-tokenizer\n", cases);
    return 0;
}
