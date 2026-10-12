/* The Kolibri-1 tokenizer through tok.h plus the engine-local guarded
 * digit override (k1_tok_load), compared with Hugging Face `tokenizers`.
 *
 *   tests/test_kolibri1_tokenizer                       tiny fixture (the test-c suite)
 *   tests/test_kolibri1_tokenizer <tokenizer.json> <set.hex>...    any tokenizer.json, with
 *       reference sets from tests/kolibri1_tok_ref.py ("hex(text) TAB id,id,..." lines)
 *
 * The tiny fixture (tests/tok_kolibri1_tiny.json) keeps the real pre-tokenizer regex
 * (including the one-digit rule spelled \p{N}{1}), the 98 added tokens with their real
 * ids (holes at 127923/127924 included), ignore_merges false and byte_fallback true, with a
 * few hundred merges. Expected ids in tests/tok_kolibri1_cases.txt come from HF
 * `tokenizers` on the same file.
 *
 * What the tiny run checks: every case encodes to HF's ids and decodes back to its
 * text; digits come out one per piece (and tok_load alone would NOT do that, the reason
 * for the override); the guard accepts exactly the Kolibri regex and refuses a changed one;
 * all 256 byte-level characters are in the vocab (byte_fallback inert); every vocab entry is
 * rebuilt by the merges alone, so the ignore_merges=false vs the whole-piece shortcut
 * question (U1) cannot change an encoding.
 */
#define _GNU_SOURCE
#define KOLIBRI1_NO_MAIN
#include "../kolibri1.c"

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; if (fails < 40) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

static int hexval(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

static int parse_ids(const char *s, int *out, int max) {
    int n = 0;
    while (*s && *s != '\n' && *s != '\r') {
        if (n >= max) return -1;
        out[n++] = atoi(s);
        while (*s && *s != ',' && *s != '\n' && *s != '\r') s++;
        if (*s == ',') s++;
    }
    return n;
}

static void show(const char *what, const int *ids, int n) {
    fprintf(stderr, "  %s(%d):", what, n);
    for (int i = 0; i < n && i < 24; i++) fprintf(stderr, " %d", ids[i]);
    fprintf(stderr, n > 24 ? " ...\n" : "\n");
}

/* a reference set: lines of "hex(text) TAB ids" */
static void run_hex_set(Tok *T, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    size_t cap = 1 << 22; char *line = malloc(cap);
    long total = 0, bad = 0, drt_bad = 0, ntok = 0;
    while (fgets(line, (int)cap, f)) {
        size_t len = strlen(line);
        if (len == cap - 1 && line[len - 1] != '\n') { cap *= 2; line = realloc(line, cap); fseek(f, -(long)len, SEEK_CUR); continue; }
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        int nb = (int)((tab - line) / 2);
        unsigned char *text = malloc((size_t)nb + 1);
        for (int i = 0; i < nb; i++) text[i] = (unsigned char)(hexval(line[2 * i]) * 16 + hexval(line[2 * i + 1]));
        int *want = malloc(((size_t)nb + 8) * sizeof(int)), *got = malloc(((size_t)nb + 8) * sizeof(int));
        int nw = parse_ids(tab + 1, want, nb + 8);
        int ng = tok_encode(T, (const char *)text, nb, got, nb + 8);
        total++; ntok += nw;
        int ok = ng == nw && !memcmp(got, want, (size_t)nw * sizeof(int));
        if (!ok) {
            bad++;
            if (bad <= 12) {
                fprintf(stderr, "MISMATCH in %s: text(%d bytes) ", path, nb);
                for (int i = 0; i < nb && i < 40; i++) fputc(text[i] >= 32 && text[i] < 127 ? text[i] : '.', stderr);
                fputc('\n', stderr); show("HF", want, nw); show("engine", got, ng);
            }
        } else if (nb < 2000) {          /* round trip: decode(encode(text)) == text for text the tokenizer fully covers */
            char *dec = malloc((size_t)nb * 4 + 64);
            int nd = tok_decode(T, got, ng, dec, nb * 4 + 63);
            if (nd != nb || memcmp(dec, text, (size_t)nb)) {
                drt_bad++;
                if (drt_bad <= 5) fprintf(stderr, "ROUND TRIP differs in %s (%d bytes -> %d)\n", path, nb, nd);
            }
            free(dec);
        }
        free(text); free(want); free(got);
    }
    fclose(f); free(line);
    printf("  %-48s %8ld texts, %9ld reference ids: %ld encodings differ, %ld round trips differ\n", path, total, ntok, bad, drt_bad);
    CHECK(bad == 0, "%s: %ld encodings differ from Hugging Face tokenizers", path, bad);
    CHECK(drt_bad == 0, "%s: %ld decode(encode(x)) != x", path, drt_bad);
}

/* the cases file of the tiny fixture: escaped text TAB ids (the o200k test's format) */
static void run_cases(Tok *T, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    char line[8192]; int pass = 0, tot = 0, dpass = 0;
    while (fgets(line, sizeof line, f)) {
        size_t nr = strlen(line);
        while (nr > 0 && (line[nr - 1] == '\n' || line[nr - 1] == '\r')) line[--nr] = 0;
        if (!nr) continue;
        char *tab = strchr(line, '\t'); if (!tab) continue;
        *tab = 0;
        char tbuf[4096]; int tn = 0;
        for (const char *q = line; *q && tn < 4095; q++) {
            if (q[0] == '\\' && q[1] == 'n') { tbuf[tn++] = '\n'; q++; }
            else if (q[0] == '\\' && q[1] == 't') { tbuf[tn++] = '\t'; q++; }
            else if (q[0] == '\\' && q[1] == 'r') { tbuf[tn++] = '\r'; q++; }
            else if (q[0] == '\\' && q[1] == '\\') { tbuf[tn++] = '\\'; q++; }
            else tbuf[tn++] = *q;
        }
        tbuf[tn] = 0;
        int want[512], got[512];
        int nw = parse_ids(tab + 1, want, 512);
        int ng = tok_encode(T, tbuf, tn, got, 512);
        int ok = ng == nw && !memcmp(got, want, (size_t)nw * sizeof(int));
        tot++; pass += ok;
        char dec[8192]; int nd = tok_decode(T, got, ng, dec, 8191);
        int drt = nd == tn && !memcmp(dec, tbuf, (size_t)tn);
        dpass += drt;
        if (!ok || !drt) { fprintf(stderr, "MISMATCH text=%s\n", line); show("HF", want, nw); show("engine", got, ng); fprintf(stderr, "  decode_ok=%d\n", drt); }
    }
    fclose(f);
    printf("  %s: ENCODE %d/%d  DECODE %d/%d\n", path, pass, tot, dpass, tot);
    CHECK(pass == tot && dpass == tot, "tiny cases: encode %d/%d decode %d/%d", pass, tot, dpass, tot);
}

/* merge-only BPE of one vocab entry with the engine's own tables, no whole-piece shortcut */
static int merge_only_symbols(Tok *T, const char *s, int sl) {
    int *soff = malloc((size_t)(sl + 1) * sizeof(int)), *slen = malloc((size_t)(sl + 1) * sizeof(int)), ns = 0;
    for (int i = 0; i < sl;) { uint32_t cp; int k = u8_next((const unsigned char *)s, sl, i, &cp); soff[ns] = i; slen[ns] = k; ns++; i += k; }
    char *kbuf = malloc((size_t)2 * sl + 2);
    for (;;) {
        int best = INT_MAX, bp = -1;
        for (int i = 0; i + 1 < ns; i++) {
            int ll = slen[i], rl = slen[i + 1];
            memcpy(kbuf, s + soff[i], ll); kbuf[ll] = 0; memcpy(kbuf + ll + 1, s + soff[i + 1], rl);
            int rk = hm_get(&T->merges, kbuf, ll + 1 + rl);
            if (rk >= 0 && rk < best) { best = rk; bp = i; }
        }
        if (bp < 0) break;
        slen[bp] = soff[bp + 1] + slen[bp + 1] - soff[bp];
        for (int j = bp + 1; j < ns - 1; j++) { soff[j] = soff[j + 1]; slen[j] = slen[j + 1]; }
        ns--;
    }
    free(soff); free(slen); free(kbuf);
    return ns;
}

static void test_u1_every_vocab_entry_is_rebuilt_by_merges(Tok *T) {
    long n = 0, bad = 0;
    for (int id = 0; id < T->n_ids; id++) {
        if (!T->id2str[id] || T->id_added[id]) continue;
        const char *s = T->id2str[id];
        n++;
        if (merge_only_symbols(T, s, (int)strlen(s)) != 1) { bad++; if (bad < 5) fprintf(stderr, "vocab entry %d is not rebuilt by the merges alone\n", id); }
    }
    printf("  U1: %ld vocab entries, %ld not rebuilt by merges alone (whole-piece shortcut vs ignore_merges=false cannot differ)\n", n, bad);
    CHECK(bad == 0, "U1: %ld vocab entries are not rebuilt by the merges: the shortcut would change encodings", bad);
}

static void test_digits(Tok *T) {
    static const struct { const char *text; int n; } cases[] = { { "2026", 4 }, { "3.14159", 7 }, { "10000", 5 }, { " 42", 3 }, { "a1b22", 5 }, { "x\n7", 3 } };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        int ids[64]; int n = tok_encode(T, cases[i].text, (int)strlen(cases[i].text), ids, 64);
        CHECK(n == cases[i].n, "'%s' -> %d ids, want %d (one digit per piece)", cases[i].text, n, cases[i].n);
    }
}

int main(int argc, char **argv) {
    const char *tok_path = argc > 2 ? argv[1] : "tests/tok_kolibri1_tiny.json";
    Tok plain;
    tok_load(&plain, tok_path);
    printf("tok_load alone: ndig1=%d (the detector looks for '|\\p{N}|', this release spells '\\p{N}{1}')\n", plain.ndig1);
    {
        int ids[16]; int n = tok_encode(&plain, "2026", 4, ids, 16);
        printf("  without the override \"2026\" -> %d ids\n", n);
        if (argc <= 2)       /* the tiny vocab has artificial digit merges, so the override is observable there */
            CHECK(plain.ndig1 || n < 4, "the unguarded tok.h must not already split digits one by one");
        else
            printf("  (the release vocab has no multi-digit entry, so digit grouping is invisible in ids on this vocab: %s)\n",
                   n == 4 ? "confirmed" : "NOT confirmed");
    }
    const char *why = k1_tok_guard(&plain);
    CHECK(why == NULL, "guard refused the Kolibri tokenizer: %s", why ? why : "");
    tok_free(&plain);

    Tok T;
    k1_tok_load(&T, tok_path);
    CHECK(T.ndig1 == 1, "override not applied");
    printf("k1_tok_load: guard passed, ndig1=1, %d ids, %d added tokens, byte level complete: %d\n", T.n_ids, T.nsp, k1_tok_bytes_all_in_vocab(&T));
    test_digits(&T);
    test_u1_every_vocab_entry_is_rebuilt_by_merges(&T);
    if (argc > 2) {
        for (int i = 2; i < argc; i++) run_hex_set(&T, argv[i]);
    } else {
        run_cases(&T, "tests/tok_kolibri1_cases.txt");
        /* guard refusals: a tokenizer.json with another digit rule / another family must not start */
        static const char *mods[][2] = {
            { "\\\\p{N}{1}", "\\\\p{N}{1,3}" },                         /* the cl100k digit rule */
            { "\"behavior\":\"Isolated\"", "\"behavior\":\"Removed\"" },
            { "\"use_regex\":false", "\"use_regex\":true" },
        };
        FILE *f = fopen(tok_path, "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        char *orig = malloc((size_t)sz + 1); if (fread(orig, 1, (size_t)sz, f) != (size_t)sz) return 2; orig[sz] = 0; fclose(f);
        for (unsigned m = 0; m < sizeof(mods) / sizeof(mods[0]); m++) {
            char *pos = strstr(orig, mods[m][0]);
            if (!pos) { fprintf(stderr, "fixture lacks %s\n", mods[m][0]); fails++; continue; }
            size_t a = strlen(mods[m][0]), b = strlen(mods[m][1]);
            char *mod = malloc((size_t)sz + b + 1);
            memcpy(mod, orig, (size_t)(pos - orig)); memcpy(mod + (pos - orig), mods[m][1], b); strcpy(mod + (pos - orig) + b, pos + a);
            const char *tmp = "tests/tmp_kolibri1_tok_mod.json";
            f = fopen(tmp, "wb"); fputs(mod, f); fclose(f);
            Tok M; tok_load(&M, tmp);
            const char *r = k1_tok_guard(&M);
            printf("  guard on a tokenizer with '%s': %s\n", mods[m][1], r ? r : "ACCEPTED");
            CHECK(r != NULL, "guard accepted a modified tokenizer (%s)", mods[m][1]);
            tok_free(&M); remove(tmp); free(mod);
        }
        free(orig);
    }
    tok_free(&T);
    if (fails) { fprintf(stderr, "test_kolibri1_tokenizer: %d FAILED\n", fails); return 1; }
    printf("test_kolibri1_tokenizer: PASS\n");
    return 0;
}
