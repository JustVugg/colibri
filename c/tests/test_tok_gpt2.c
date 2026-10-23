/* GPT-2 pre-tokenizer family (bare ByteLevel with use_regex, no Split: OLMoE,
 * GPT-NeoX) against HF-tokenizers-generated expectations. tests/tok_gpt2_tiny.json
 * is a byte-level BPE trained with `tokenizers` on a few sentences (14 KB, no
 * model download) with exactly OLMoE's pre_tokenizer block, and
 * tests/tok_gpt2_cases.txt holds the ids HF produced for each case. Guards the
 * rules where GPT-2 and cl100k differ, which tok.h used to get wrong for OLMoE:
 * " 4" and " 20" as one piece, unbounded digit runs, a non-space prefix NOT glued
 * to the word ("(abc"), case-sensitive contractions, whitespace runs before text,
 * and added-token boundaries. Round-trips every case. Measured on the real OLMoE
 * tokenizer: 1560/1708 cases identical before this family, 1708/1708 after. */
#define _GNU_SOURCE
#include "../tok.h"

int main(void) {
    Tok T;
    tok_load(&T, "tests/tok_gpt2_tiny.json");
    if (!T.gpt2) { fprintf(stderr, "test_tok_gpt2: GPT-2 family (bare ByteLevel) not detected\n"); return 1; }
    FILE *f = fopen("tests/tok_gpt2_cases.txt", "rb");
    if (!f) { perror("tests/tok_gpt2_cases.txt"); return 1; }
    /* fgets, not getline: MinGW's UCRT lacks getline and this must run on
     * the windows job. Case lines are short; 8 KB is generous. */
    char line[8192];
    int pass = 0, tot = 0, dpass = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t nr = strlen(line);
        while (nr > 0 && (line[nr-1] == '\n' || line[nr-1] == '\r')) line[--nr] = 0;
        if (nr == 0) continue;
        char *tab = strchr(line, '\t'); if (!tab) continue;
        *tab = 0;
        const char *text = line, *idstr = tab + 1;
        char tbuf[4096]; int tn = 0;
        for (const char *q = text; *q && tn < 4095; q++) {
            if      (q[0]=='\\' && q[1]=='n')  { tbuf[tn++]='\n'; q++; }
            else if (q[0]=='\\' && q[1]=='t')  { tbuf[tn++]='\t'; q++; }
            else if (q[0]=='\\' && q[1]=='r')  { tbuf[tn++]='\r'; q++; }
            else if (q[0]=='\\' && q[1]=='\\') { tbuf[tn++]='\\'; q++; }
            else tbuf[tn++] = *q;
        }
        tbuf[tn] = 0;
        int exp[512], ne = 0;
        for (const char *q = idstr; *q; ) {
            while (*q == ',' || *q == ' ') q++;
            if (!*q) break;
            exp[ne++] = atoi(q);
            while (*q && *q != ',') q++;
        }
        int got[512]; int ng = tok_encode(&T, tbuf, tn, got, 512);
        int ok = (ng == ne);
        for (int i = 0; i < ng && ok; i++) ok = (got[i] == exp[i]);
        tot++; if (ok) pass++;
        char dec[8192]; int dn = tok_decode(&T, got, ng, dec, 8191);
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
    printf("test_tok_gpt2: ENCODE %d/%d  DECODE %d/%d\n", pass, tot, dpass, tot);
    return (pass == tot && dpass == tot) ? 0 : 2;
}
