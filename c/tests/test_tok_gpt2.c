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
#define TOK_NFC   /* as olmoe.c: apply tokenizer.json's NFC normalizer */
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

    /* tok_gpt2_tiny.json carries OLMoE's normalizer, {"type":"NFC"}: HF composes
     * the text before it pre-tokenizes, so a decomposed spelling encodes exactly
     * like the composed one, and decodes to it. Ids from HF tokenizers 0.23.3 on
     * this file. Added tokens are matched on the raw text first, as HF does. */
    static const struct { const char *text, *nfc; int ids[16]; int n; } nfc_cases[] = {
        { "cafe\xcc\x81", "caf\xc3\xa9", {388}, 1 },   /* e + U+0301 */
        { " e\xcc\x81te\xcc\x81 42", " \xc3\xa9t\xc3\xa9 42", {220,342,83,342,366}, 5 },   /* two combining accents */
        { "Cafe\xcc\x81 au lait", "Caf\xc3\xa9 au lait", {34,305,342,264,84,352,72,83}, 8 },
        { "\xe1\x84\x92\xe1\x85\xa1\xe1\x86\xab", "\xed\x95\x9c", {169,243,250}, 3 },   /* Hangul jamo -> one syllable */
        { "\xe2\x84\xab x", "\xc3\x85 x", {127,227,278}, 3 },   /* ANGSTROM SIGN -> A with ring (a singleton) */
        { "o\xcc\x88\xcc\x81", "\xc3\xb6\xcc\x81", {127,114,136,223}, 4 },   /* only the first mark composes */
        { "A\xcc\x8a<|endoftext|>e\xcc\x81", "\xc3\x85<|endoftext|>\xc3\xa9", {127,227,405,342}, 4 },   /* both sides of an added token */
    };
    int nfc_pass = 0, nfc_tot = (int)(sizeof(nfc_cases) / sizeof(nfc_cases[0]));
    for (int c = 0; c < nfc_tot; c++) {
        int ok = 1;
        const char *forms[2] = { nfc_cases[c].text, nfc_cases[c].nfc };
        for (int form = 0; form < 2; form++) {
            int got[64]; int ng = tok_encode(&T, forms[form], (int)strlen(forms[form]), got, 64);
            int same = (ng == nfc_cases[c].n);
            for (int i = 0; i < ng && same; i++) same = (got[i] == nfc_cases[c].ids[i]);
            char dec[256]; int dn = tok_decode(&T, got, ng, dec, 255);
            same = same && dn == (int)strlen(nfc_cases[c].nfc) && !memcmp(dec, nfc_cases[c].nfc, dn);
            if (!same) {
                fprintf(stderr, "NFC MISMATCH case %d (%s form):", c, form ? "composed" : "decomposed");
                for (int i = 0; i < ng; i++) fprintf(stderr, " %d", got[i]);
                fprintf(stderr, "\n");
                ok = 0;
            }
        }
        nfc_pass += ok;
    }
    printf("test_tok_gpt2: NFC %d/%d\n", nfc_pass, nfc_tot);
    return (pass == tot && dpass == tot && nfc_pass == nfc_tot) ? 0 : 2;
}
