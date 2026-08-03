/* o200k pre-tokenizer validation against HF-tokenizers-generated expectations.
 * Self-contained for the test-c harness: loads tests/tok_o200k_tiny.json (a
 * synthetic byte-level BPE whose Split regex is the o200k pattern — a few KB,
 * no model download) and scores tests/tok_o200k_cases.txt, whose expected ids
 * were produced by HF `tokenizers` on the same file. Guards the case-aware
 * letter matcher, contractions, digit groups, the [\r\n/]* punctuation tail,
 * whitespace branches, and added-token atomicity; round-trips every case.
 * The cl100k path is untouched by construction (dispatch requires \p{Lu} in
 * the tokenizer's own Split pattern) and stays covered by the GLM oracle. */
#define _GNU_SOURCE
#include "../tok.h"

int main(void) {
    Tok T;
    tok_load(&T, "tests/tok_o200k_tiny.json");
    if (!T.o200k) {
        fprintf(stderr, "test_tok_o200k: o200k pattern not detected\n");
        return 1;
    }
    FILE *f = fopen("tests/tok_o200k_cases.txt", "rb");
    if (!f) {
        perror("tests/tok_o200k_cases.txt");
        return 1;
    }
    /* fgets, not getline: MinGW's UCRT lacks getline and this must run on
     * the windows job. Case lines are short; 8 KB is generous. */
    char line[8192];
    int pass = 0, tot = 0, dpass = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t nr = strlen(line);
        while (nr > 0 && (line[nr - 1] == '\n' || line[nr - 1] == '\r')) line[--nr] = 0;
        if (nr == 0) continue;
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        const char *text = line, *idstr = tab + 1;
        char tbuf[4096];
        int tn = 0;
        for (const char *q = text; *q && tn < 4095; q++) {
            if (q[0] == '\\' && q[1] == 'n') {
                tbuf[tn++] = '\n';
                q++;
            } else if (q[0] == '\\' && q[1] == 't') {
                tbuf[tn++] = '\t';
                q++;
            } else if (q[0] == '\\' && q[1] == 'r') {
                tbuf[tn++] = '\r';
                q++;
            } else if (q[0] == '\\' && q[1] == '\\') {
                tbuf[tn++] = '\\';
                q++;
            } else tbuf[tn++] = *q;
        }
        tbuf[tn] = 0;
        int exp[512], ne = 0;
        for (const char *q = idstr; *q;) {
            while (*q == ',' || *q == ' ') q++;
            if (!*q) break;
            exp[ne++] = atoi(q);
            while (*q && *q != ',') q++;
        }
        int got[512];
        int ng = tok_encode(&T, tbuf, tn, got, 512);
        int ok = (ng == ne);
        for (int i = 0; i < ng && ok; i++) ok = (got[i] == exp[i]);
        tot++;
        if (ok) pass++;
        char dec[8192];
        int dn = tok_decode(&T, got, ng, dec, 8191);
        int drt = (dn == tn) && !memcmp(dec, tbuf, tn);
        if (drt) dpass++;
        if (!ok || !drt) {
            fprintf(stderr, "MISMATCH text=%s\n  exp(%d):", text, ne);
            for (int i = 0; i < ne; i++) fprintf(stderr, " %d", exp[i]);
            fprintf(stderr, "\n  got(%d):", ng);
            for (int i = 0; i < ng; i++) fprintf(stderr, " %d", got[i]);
            fprintf(stderr, "\n  decode_ok=%d\n", drt);
        }
    }
    fclose(f);
    printf("test_tok_o200k: ENCODE %d/%d  DECODE %d/%d\n", pass, tot, dpass, tot);
    return (pass == tot && dpass == tot) ? 0 : 2;
}
