/* Flint's pre-tokenizer (qwen36.c flint_encode_span): a byte-level BPE of 512 pieces trained
 * with Flint's own Split sequence (tools/gguf_import.py FLINT_SPLITS, the release's
 * tokenizer.json), and 87 texts whose ids HF's tokenizers gave
 * (tests/fixtures/flint_tok_tiny.json, tests/fixtures/flint_tok_cases.txt: hex of the
 * UTF-8, a tab, the ids). Digits by three, CJK runs, letters by case, punctuation and
 * symbols, whitespace and newlines, marks, controls and added tokens. The same check on the
 * release's tokenizer (447 texts) is identical too; that file is 11 MB. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

int main(void) {
    load_tokenizer("tests/fixtures/flint_tok_tiny.json");
    if (!g_pretok_flint) { printf("FAIL: Flint's pre-tokenizer not recognized\n"); return 1; }
    FILE *f = fopen("tests/fixtures/flint_tok_cases.txt", "rb");
    if (!f) { printf("FAIL: tests/fixtures/flint_tok_cases.txt\n"); return 1; }
    char line[8192]; int cases = 0, fails = 0;
    while (fgets(line, sizeof line, f)) {
        char *tab = strchr(line, '\t'); if (!tab) continue;
        *tab = 0;
        size_t hl = strlen(line); char *text = malloc(hl / 2 + 1);
        for (size_t i = 0; i + 1 < hl; i += 2) { unsigned v; sscanf(line + i, "%2x", &v); text[i / 2] = (char)v; }
        text[hl / 2] = 0;
        int want[1024], nw = 0; char *p = tab + 1;
        while (*p && *p != '\n' && *p != '\r' && nw < 1024) {   /* a CRLF checkout (Windows) ends the ids at \r */
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
    if (fails || cases < 80) { printf("test_qwen36_tokenizer_flint: %d of %d failed\n", fails, cases); return 1; }
    printf("OK test_qwen36_tokenizer_flint: %d texts as HF's tokenizers splits them with Flint's pre-tokenizer\n", cases);
    return 0;
}
