/* gliner_decide.c -- GLiNER2.5-Decide (fastino/GLiNER2.5-Decide on Hugging Face,
 * Apache-2.0) as a decision engine, in C.
 *
 * GLiNER2.5-Decide is a GLiNER2 "span" checkpoint: a DeBERTa-v3-large encoder
 * and GLiNER2's heads. A decision uses one of them, the classification head:
 * every question of a request becomes one classification task of a single
 * classify_text() call, all tasks and the text are read in one sequence by the
 * encoder, and an MLP scores each label at its [L] marker. The extraction
 * heads (span_rep, count_pred, count_embed) are never run and never loaded.
 *
 * The specification is the model's own Python: the `gliner2` package (2.0.0)
 * and the DeBERTa-v2 modeling code of transformers it builds on. Each step
 * below names the function it follows:
 *
 *   words        processing/word_splitter.WhitespaceTokenSplitter: Python's
 *                regex (URLs, e-mails, @handles, \w+(?:[-_]\w+)*, \S), each
 *                word lower-cased with str.lower(); processor._collate_batch
 *                first appends "." to a text that does not end in . ! or ?
 *   schema       inference/schema.Schema.classification and
 *                processor._transform_schema: per task
 *                ( [P] task[: prompt][ [DESCRIPTION] label: text ...] ( [L] label ... ) )
 *                tasks joined by [SEP_STRUCT], then [SEP_TEXT] and the words
 *   tokens       processor._format_input_with_mapping: every element above is
 *                tokenized on its own (tokenizer.tokenize), no [CLS] / [SEP];
 *                a marker's embedding is its first sub-token
 *   tokenizer    tokenizer.json (DebertaV2TokenizerFast): added tokens split
 *                out, Replace(\s{2,}|[\n\r\t] -> " "), NFC, right strip,
 *                Metaspace (prepend always, split), Unigram (Viterbi, unknown
 *                characters fused into one [UNK])
 *   encoder      DebertaV2Model: word embeddings and LayerNorm (no absolute
 *                positions), 24 post-norm layers of disentangled attention
 *                (content-to-content + content-to-position + position-to-
 *                content, relative positions in log buckets, shared key and
 *                query projections, the relative embeddings LayerNorm'd), GELU
 *                FFN
 *   head         runtime._extract_classification_result: classifier
 *                (Linear -> ReLU -> Linear) at each [L], temperature 1,
 *                softmax over the task's labels (a single-label task)
 *
 * A System One question becomes a task as docs/gliner_decide.md describes: the
 * task name is the question id, the prompt its instructions, the labels a
 * choice's labels, a score's levels "0".."n-1" or a noul's "yes" then "no",
 * and the option texts become label descriptions.
 *
 * Weights are read as stored (F32 in the release) and kept in f32. The
 * relative embeddings are the same for every request, so their projections
 * through each layer's query and key are computed once at load.
 *
 * Ways to run it:
 *   SERVE=1 SNAP=<dir> gliner_decide              the mux serve protocol, DECIDE only
 *   gliner_decide --model DIR --records F [--ids] a JSON array of DECIDE records;
 *                                                 one DECISION object per line
 *   gliner_decide --model DIR --tokenize F        a JSON array of strings; their ids
 *   gliner_decide --split F                       a JSON array of strings; their words
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "compat.h"
#include "json.h"
#include "st.h"
#include "qwen38_nfc.h"
#include "qi_gemm.h"
#include "matmul_f32.h"
#include "serve_codec.h"
#include "decide_serve.h"
#include "omp_tune.h"
#include "gliner_unicode.h"

/* ------------------------------------------------------------------ utils */

static void die(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[gliner_decide] ");
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
    exit(1);
}

static void *xmalloc(size_t size)
{
    void *p = malloc(size ? size : 1);
    if (!p) die("out of memory (%zu bytes)", size);
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) die("out of memory (%zu x %zu bytes)", count, size);
    return p;
}

static void *xrealloc(void *p, size_t size)
{
    void *q = realloc(p, size ? size : 1);
    if (!q) die("out of memory (%zu bytes)", size);
    return q;
}

static char *read_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || n > (1L << 30)) { fclose(f); return NULL; }
    rewind(f);
    char *b = (char *)xmalloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0;
    fclose(f);
    return b;
}

static jval *read_json(const char *dir, const char *name)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    char *text = read_text(path);
    if (!text) die("cannot read %s", path);
    jval *root = json_parse_checked(text);
    free(text);
    if (!root || root->t != J_OBJ) die("%s is not a JSON object", path);
    return root;
}

static double num_or(jval *o, const char *key, double fallback)
{
    jval *v = json_get(o, key);
    return v && v->t == J_NUM ? v->num : fallback;
}

static int bool_or(jval *o, const char *key, int fallback)
{
    jval *v = json_get(o, key);
    return v && v->t == J_BOOL ? v->boolean : fallback;
}

static const char *str_or(jval *o, const char *key, const char *fallback)
{
    jval *v = json_get(o, key);
    return v && v->t == J_STR ? v->str : fallback;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* A growable byte string and a growable int array. */
typedef struct { char *s; size_t n, cap; } Str;
typedef struct { int *v; int n, cap; } IntVec;

static void str_put(Str *b, const char *bytes, size_t count)
{
    if (b->n + count + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 64;
        while (cap < b->n + count + 1) cap *= 2;
        b->s = (char *)xrealloc(b->s, cap);
        b->cap = cap;
    }
    memcpy(b->s + b->n, bytes, count);
    b->n += count;
    b->s[b->n] = 0;
}

static void str_cp(Str *b, uint32_t cp)
{
    char u[4];
    size_t n;
    if (cp < 0x80) { u[0] = (char)cp; n = 1; }
    else if (cp < 0x800) { u[0] = (char)(0xC0 | (cp >> 6)); u[1] = (char)(0x80 | (cp & 0x3F)); n = 2; }
    else if (cp < 0x10000) {
        u[0] = (char)(0xE0 | (cp >> 12)); u[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        u[2] = (char)(0x80 | (cp & 0x3F)); n = 3;
    } else {
        u[0] = (char)(0xF0 | (cp >> 18)); u[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        u[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); u[3] = (char)(0x80 | (cp & 0x3F)); n = 4;
    }
    str_put(b, u, n);
}

static void ivec_push(IntVec *a, int x)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 64;
        a->v = (int *)xrealloc(a->v, (size_t)a->cap * sizeof(int));
    }
    a->v[a->n++] = x;
}

/* One UTF-8 code point at s[*at] (< n). Bytes that are not valid UTF-8 come
 * back one at a time as themselves; the gateway only ever sends valid UTF-8. */
static uint32_t utf8_next(const unsigned char *s, size_t n, size_t *at)
{
    size_t i = *at;
    unsigned char c = s[i];
    uint32_t cp;
    size_t w;
    if (c < 0x80) { *at = i + 1; return c; }
    if (c >= 0xC2 && c <= 0xDF) { cp = c & 0x1F; w = 2; }
    else if (c >= 0xE0 && c <= 0xEF) { cp = c & 0x0F; w = 3; }
    else if (c >= 0xF0 && c <= 0xF4) { cp = c & 0x07; w = 4; }
    else { *at = i + 1; return c; }
    if (i + w > n) { *at = i + 1; return c; }
    for (size_t k = 1; k < w; k++) {
        if ((s[i + k] & 0xC0) != 0x80) { *at = i + 1; return c; }
        cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    *at = i + w;
    return cp;
}

/* ------------------------------------------------- Python's view of a text */

static int in_ranges(const GlUniRange *r, int n, uint32_t cp)
{
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid - 1;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return 1;
    }
    return 0;
}

#define NRANGES(t) ((int)(sizeof(t) / sizeof((t)[0])))

static int py_word(uint32_t cp)       /* \w of a str pattern */
{
    if (cp < 0x80) return (cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z') ||
                          (cp >= 'A' && cp <= 'Z') || cp == '_';
    return in_ranges(gl_uni_word, NRANGES(gl_uni_word), cp);
}

static int py_space(uint32_t cp)      /* \s of a str pattern: str.isspace() */
{
    return in_ranges(gl_uni_space, NRANGES(gl_uni_space), cp);
}

/* [a-z] under re.IGNORECASE: the ASCII letters plus the four code points whose
 * case folding reaches them (U+0130, U+0131, U+017F, U+212A). */
static int py_az(uint32_t cp)
{
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') ||
           cp == 0x130 || cp == 0x131 || cp == 0x17F || cp == 0x212A;
}

static int py_digit_ascii(uint32_t cp) { return cp >= '0' && cp <= '9'; }

/* A literal ASCII letter under re.IGNORECASE ('s' also matches U+017F). */
static int py_lit(uint32_t cp, char want)
{
    if (cp == (uint32_t)want || (want >= 'a' && want <= 'z' && cp == (uint32_t)(want - 32))) return 1;
    return want == 's' && cp == 0x17F;
}

static uint32_t py_lower1(uint32_t cp)
{
    if (cp < 0x80) return cp >= 'A' && cp <= 'Z' ? cp + 32 : cp;
    int lo = 0, hi = NRANGES(gl_uni_lower) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < gl_uni_lower[mid].cp) hi = mid - 1;
        else if (cp > gl_uni_lower[mid].cp) lo = mid + 1;
        else return gl_uni_lower[mid].lower;
    }
    return cp;
}

static int py_ignorable(uint32_t cp) { return in_ranges(gl_uni_ignorable, NRANGES(gl_uni_ignorable), cp); }
static int py_cased(uint32_t cp) { return in_ranges(gl_uni_cased, NRANGES(gl_uni_cased), cp); }

/* str.lower() of cps[0..n), appended to out as UTF-8. Capital sigma follows
 * Python's handle_capital_sigma (the Final_Sigma context of Unicode 3.13). */
static void py_lower(const uint32_t *cps, int n, Str *out)
{
    for (int i = 0; i < n; i++) {
        uint32_t cp = cps[i];
        if (cp == 0x130) { str_cp(out, 'i'); str_cp(out, 0x307); continue; }
        if (cp == 0x3A3) {
            int j = i - 1;
            while (j >= 0 && py_ignorable(cps[j])) j--;
            int final_sigma = j >= 0 && py_cased(cps[j]);
            if (final_sigma && i + 1 < n) {
                j = i + 1;
                while (j < n && py_ignorable(cps[j])) j++;
                final_sigma = j == n || !py_cased(cps[j]);
            }
            str_cp(out, final_sigma ? 0x3C2 : 0x3C3);
            continue;
        }
        str_cp(out, py_lower1(cp));
    }
}

/* ------------------------------------------- WhitespaceTokenSplitter */

typedef struct { char **w; int n, cap; } Words;

static void words_free(Words *w)
{
    for (int i = 0; i < w->n; i++) free(w->w[i]);
    free(w->w);
    memset(w, 0, sizeof(*w));
}

/* The pattern of processing/word_splitter.py, re.VERBOSE | re.IGNORECASE:
 *     (?:https?://[^\s]+|www\.[^\s]+)
 *   | [a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
 *   | @[a-z0-9_]+
 *   | \w+(?:[-_]\w+)*
 *   | \S
 * tried in that order at each position, as re.finditer does; returns the end
 * of the match at p, or -1. */
/* [^\s]+ from k: the end of the run, or -1 when it is empty */
static int nonspace_run(const uint32_t *c, int n, int k)
{
    if (k >= n || py_space(c[k])) return -1;
    while (k < n && !py_space(c[k])) k++;
    return k;
}

/* "://" then [^\s]+ */
static int url_rest(const uint32_t *c, int n, int k)
{
    if (k + 3 > n || c[k] != ':' || c[k + 1] != '/' || c[k + 2] != '/') return -1;
    return nonspace_run(c, n, k + 3);
}

static int split_match(const uint32_t *c, int n, int p)
{
    /* https?://[^\s]+ (the 's' tried first, as the greedy ? does), www\.[^\s]+ */
    if (p + 4 <= n && py_lit(c[p], 'h') && py_lit(c[p + 1], 't') && py_lit(c[p + 2], 't') &&
        py_lit(c[p + 3], 'p')) {
        int k = p + 4, e = -1;
        if (k < n && py_lit(c[k], 's')) e = url_rest(c, n, k + 1);
        if (e < 0) e = url_rest(c, n, k);
        if (e >= 0) return e;
    }
    if (p + 4 <= n && py_lit(c[p], 'w') && py_lit(c[p + 1], 'w') && py_lit(c[p + 2], 'w') && c[p + 3] == '.') {
        int e = nonspace_run(c, n, p + 4);
        if (e >= 0) return e;
    }
    /* e-mail: the local part cannot contain '@', so only its longest run can be
     * followed by one; the domain backtracks to the last ".xx" that fits */
    {
        int k = p;
        while (k < n && (py_az(c[k]) || py_digit_ascii(c[k]) || c[k] == '.' || c[k] == '_' ||
                         c[k] == '%' || c[k] == '+' || c[k] == '-'))
            k++;
        if (k > p && k < n && c[k] == '@') {
            int d0 = k + 1, d1 = d0;
            while (d1 < n && (py_az(c[d1]) || py_digit_ascii(c[d1]) || c[d1] == '.' || c[d1] == '-')) d1++;
            for (int split = d1 - 1; split >= d0 + 1; split--) {
                if (c[split] != '.') continue;
                int e = split + 1;
                while (e < n && py_az(c[e])) e++;
                if (e - (split + 1) >= 2) return e;
            }
        }
    }
    /* @handle */
    if (c[p] == '@') {
        int k = p + 1;
        while (k < n && (py_az(c[k]) || py_digit_ascii(c[k]) || c[k] == '_')) k++;
        if (k > p + 1) return k;
    }
    /* \w+(?:[-_]\w+)* */
    if (py_word(c[p])) {
        int k = p + 1;
        for (;;) {
            while (k < n && py_word(c[k])) k++;
            if (k + 1 < n && (c[k] == '-' || c[k] == '_') && py_word(c[k + 1])) { k += 1; continue; }
            break;
        }
        return k;
    }
    /* \S */
    if (!py_space(c[p])) return p + 1;
    return -1;
}

/* The words of `text`, lower-cased, as WhitespaceTokenSplitter yields them. */
static void split_words(const char *text, size_t len, Words *out)
{
    memset(out, 0, sizeof(*out));
    int n = 0, cap = 64;
    uint32_t *c = (uint32_t *)xmalloc((size_t)cap * sizeof(uint32_t));
    for (size_t at = 0; at < len;) {
        uint32_t cp = utf8_next((const unsigned char *)text, len, &at);
        if (n == cap) { cap *= 2; c = (uint32_t *)xrealloc(c, (size_t)cap * sizeof(uint32_t)); }
        c[n++] = cp;
    }
    for (int p = 0; p < n;) {
        int e = split_match(c, n, p);
        if (e < 0) { p++; continue; }
        Str w = {0};
        py_lower(c + p, e - p, &w);
        if (!w.s) str_put(&w, "", 0);
        if (out->n == out->cap) {
            out->cap = out->cap ? out->cap * 2 : 64;
            out->w = (char **)xrealloc(out->w, (size_t)out->cap * sizeof(char *));
        }
        out->w[out->n++] = w.s;
        p = e;
    }
    free(c);
}

/* -------------------------------------------------------------- tokenizer */

typedef struct { char *s; int len, id, normalized; } AddedTok;

typedef struct {
    int n;                       /* Unigram pieces */
    char **piece;
    int *plen;
    double *score;
    int *slot;                   /* open addressing over the pieces, -1 empty */
    uint64_t mask;
    int max_piece;               /* longest piece, in bytes */
    double unk_score;            /* min score - 10 (Unigram's K_UNK_PENALTY) */
    int unk_id;
    AddedTok *added;
    int n_added;
} GlTok;

static uint64_t fnv1a(const char *s, int n)
{
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static int piece_find(const GlTok *T, const char *s, int n)
{
    uint64_t h = fnv1a(s, n) & T->mask;
    for (;;) {
        int id = T->slot[h];
        if (id < 0) return -1;
        if (T->plen[id] == n && !memcmp(T->piece[id], s, (size_t)n)) return id;
        h = (h + 1) & T->mask;
    }
}

/* White_Space, which both the normalizer's \s (Oniguruma) and its Strip
 * (Rust's char::is_whitespace) mean. */
static int uni_white(uint32_t cp)
{
    return (cp >= 0x9 && cp <= 0xD) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

static void load_tokenizer(GlTok *T, const char *dir)
{
    memset(T, 0, sizeof(*T));
    jval *tj = read_json(dir, "tokenizer.json");
    jval *model = json_get(tj, "model");
    if (!model || strcmp(str_or(model, "type", ""), "Unigram"))
        die("tokenizer.json: only a Unigram model is supported (this is %s)",
            model ? str_or(model, "type", "unknown") : "missing");
    if (bool_or(model, "byte_fallback", 0)) die("tokenizer.json: byte_fallback is not supported");
    /* the normalizer of the release: Replace(\s{2,}|[\n\r\t] -> " "), NFC, Strip(right) */
    jval *norm = json_get(tj, "normalizer"), *seq = norm ? json_get(norm, "normalizers") : NULL;
    int ok = norm && !strcmp(str_or(norm, "type", ""), "Sequence") && seq && seq->t == J_ARR && seq->len == 3;
    if (ok) {
        jval *rep = seq->kids[0], *pat = json_get(rep, "pattern");
        ok = !strcmp(str_or(rep, "type", ""), "Replace") && pat &&
             !strcmp(str_or(pat, "Regex", ""), "\\s{2,}|[\\n\\r\\t]") &&
             !strcmp(str_or(rep, "content", ""), " ") &&
             !strcmp(str_or(seq->kids[1], "type", ""), "NFC") &&
             !strcmp(str_or(seq->kids[2], "type", ""), "Strip") &&
             !bool_or(seq->kids[2], "strip_left", 1) && bool_or(seq->kids[2], "strip_right", 0);
    }
    if (!ok) die("tokenizer.json: expected the normalizer Sequence[Replace(\\s{2,}|[\\n\\r\\t] -> \" \"), "
                 "NFC, Strip(right)] of the DeBERTa-v3 tokenizer");
    jval *pre = json_get(tj, "pre_tokenizer");
    if (pre && !strcmp(str_or(pre, "type", ""), "Sequence")) {
        jval *ps = json_get(pre, "pretokenizers");
        pre = ps && ps->t == J_ARR && ps->len == 1 ? ps->kids[0] : NULL;
    }
    if (!pre || strcmp(str_or(pre, "type", ""), "Metaspace") || strcmp(str_or(pre, "replacement", ""), "\xE2\x96\x81") ||
        strcmp(str_or(pre, "prepend_scheme", "always"), "always") || !bool_or(pre, "split", 1))
        die("tokenizer.json: expected a Metaspace pre-tokenizer (replacement U+2581, prepend always, split)");
    jval *vocab = json_get(model, "vocab");
    if (!vocab || vocab->t != J_ARR || vocab->len < 1) die("tokenizer.json: Unigram vocab missing");
    T->n = vocab->len;
    T->piece = (char **)xcalloc((size_t)T->n, sizeof(char *));
    T->plen = (int *)xcalloc((size_t)T->n, sizeof(int));
    T->score = (double *)xcalloc((size_t)T->n, sizeof(double));
    uint64_t size = 1;
    while (size < (uint64_t)T->n * 2) size <<= 1;
    T->mask = size - 1;
    T->slot = (int *)xmalloc((size_t)size * sizeof(int));
    for (uint64_t i = 0; i < size; i++) T->slot[i] = -1;
    double min_score = INFINITY;
    for (int i = 0; i < T->n; i++) {
        jval *e = vocab->kids[i];
        if (!e || e->t != J_ARR || e->len != 2 || e->kids[0]->t != J_STR || e->kids[1]->t != J_NUM)
            die("tokenizer.json: vocab entry %d is not [piece, score]", i);
        T->piece[i] = strdup(e->kids[0]->str);
        T->plen[i] = (int)strlen(T->piece[i]);
        T->score[i] = e->kids[1]->num;
        if (T->score[i] < min_score) min_score = T->score[i];
        if (T->plen[i] > T->max_piece) T->max_piece = T->plen[i];
        /* a piece listed twice keeps its last id, as the reference's map does */
        uint64_t h = fnv1a(T->piece[i], T->plen[i]) & T->mask;
        for (;;) {
            int id = T->slot[h];
            if (id < 0 || (T->plen[id] == T->plen[i] && !memcmp(T->piece[id], T->piece[i], (size_t)T->plen[i]))) {
                T->slot[h] = i;
                break;
            }
            h = (h + 1) & T->mask;
        }
    }
    T->unk_score = min_score - 10.0;
    jval *unk = json_get(model, "unk_id");
    T->unk_id = unk && unk->t == J_NUM ? (int)unk->num : -1;
    if (T->unk_id < 0 || T->unk_id >= T->n) die("tokenizer.json: the Unigram model needs an unk_id");
    jval *added = json_get(tj, "added_tokens");
    T->added = (AddedTok *)xcalloc((size_t)(added && added->t == J_ARR ? added->len : 0) + 1, sizeof(AddedTok));
    for (int i = 0; added && added->t == J_ARR && i < added->len; i++) {
        jval *a = added->kids[i];
        const char *content = str_or(a, "content", NULL);
        jval *id = json_get(a, "id");
        if (!content || !content[0] || !id || id->t != J_NUM) continue;
        if (bool_or(a, "lstrip", 0) || bool_or(a, "rstrip", 0) || bool_or(a, "single_word", 0))
            die("tokenizer.json: added token %s strips whitespace, which this engine does not implement", content);
        AddedTok *t = &T->added[T->n_added++];
        t->s = strdup(content);
        t->len = (int)strlen(content);
        t->id = (int)id->num;
        t->normalized = bool_or(a, "normalized", 0);
    }
    json_free(tj);
}

/* The longest added token of the given kind starting at s[p], or -1. */
static int added_at(const GlTok *T, const char *s, size_t n, size_t p, int normalized)
{
    int best = -1;
    for (int k = 0; k < T->n_added; k++) {
        const AddedTok *a = &T->added[k];
        if (a->normalized != normalized || (size_t)a->len > n - p) continue;
        if (!memcmp(s + p, a->s, (size_t)a->len) && (best < 0 || a->len > T->added[best].len)) best = k;
    }
    return best;
}

/* Unigram.encode_optimized on one pre-token: the best-scoring segmentation,
 * ties kept by the first candidate (shorter piece, earlier start); a
 * character no single piece covers costs unk_score, and adjacent unknowns
 * come out as one [UNK]. */
static void unigram(const GlTok *T, const char *s, int n, IntVec *out)
{
    if (n <= 0) return;
    double *best = (double *)xmalloc((size_t)(n + 1) * sizeof(double));
    int *from = (int *)xmalloc((size_t)(n + 1) * sizeof(int));
    int *id = (int *)xmalloc((size_t)(n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) { from[i] = -1; best[i] = 0; id[i] = -1; }
    from[0] = 0;
    for (int start = 0; start < n;) {
        size_t at = (size_t)start;
        utf8_next((const unsigned char *)s, (size_t)n, &at);
        int mblen = (int)at - start;
        double here = best[start];
        int single = 0;
        int limit = n - start < T->max_piece ? n - start : T->max_piece;
        for (int len = 1; len <= limit; len++) {
            int pid = piece_find(T, s + start, len);
            if (pid < 0) continue;
            int key = start + len;
            double cand = here + T->score[pid];
            if (from[key] < 0 || cand > best[key]) { best[key] = cand; from[key] = start; id[key] = pid; }
            if (len == mblen) single = 1;
        }
        if (!single) {
            int key = start + mblen;
            double cand = here + T->unk_score;
            if (from[key] < 0 || cand > best[key]) { best[key] = cand; from[key] = start; id[key] = T->unk_id; }
        }
        start += mblen;
    }
    /* backtrack, fusing runs of [UNK] */
    int count = 0;
    int *rev = (int *)xmalloc((size_t)(n + 1) * sizeof(int));
    for (int end = n; end > 0;) {
        int start = from[end];
        if (start < 0 || start >= end) die("unigram: broken lattice");
        if (id[end] == T->unk_id && count > 0 && rev[count - 1] == T->unk_id) { end = start; continue; }
        rev[count++] = id[end];
        end = start;
    }
    for (int i = count - 1; i >= 0; i--) ivec_push(out, rev[i]);
    free(rev); free(best); free(from); free(id);
}

/* Metaspace (prepend always, split) and Unigram on one normalized text. */
static void metaspace_unigram(const GlTok *T, const char *s, size_t n, IntVec *out)
{
    if (!n) return;
    Str m = {0};
    if (!(n >= 3 && (unsigned char)s[0] == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81) &&
        s[0] != ' ')
        str_put(&m, "\xE2\x96\x81", 3);
    for (size_t i = 0; i < n; i++) {
        if (s[i] == ' ') str_put(&m, "\xE2\x96\x81", 3);
        else str_put(&m, s + i, 1);
    }
    /* split before every U+2581, the delimiter kept with what follows it */
    size_t start = 0;
    for (size_t i = 3; i + 3 <= m.n; i++) {
        if ((unsigned char)m.s[i] == 0xE2 && (unsigned char)m.s[i + 1] == 0x96 && (unsigned char)m.s[i + 2] == 0x81) {
            unigram(T, m.s + start, (int)(i - start), out);
            start = i;
            i += 2;
        }
    }
    unigram(T, m.s + start, (int)(m.n - start), out);
    free(m.s);
}

/* Replace(\s{2,}|[\n\r\t] -> " "), NFC, Strip(right); then the normalized
 * added tokens, then Metaspace + Unigram on what is left. */
static void encode_segment(const GlTok *T, const char *s, size_t n, IntVec *out)
{
    Str r = {0};
    int ascii = 1;
    for (size_t at = 0; at < n;) {
        size_t here = at;
        uint32_t cp = utf8_next((const unsigned char *)s, n, &at);
        if (cp >= 0x80) ascii = 0;
        if (!uni_white(cp)) { str_put(&r, s + here, at - here); continue; }
        size_t run_end = at;
        int run = 1;
        while (run_end < n) {
            size_t peek = run_end;
            uint32_t next = utf8_next((const unsigned char *)s, n, &peek);
            if (!uni_white(next)) break;
            run++;
            run_end = peek;
        }
        if (run >= 2) { str_put(&r, " ", 1); at = run_end; }
        else if (cp == '\n' || cp == '\r' || cp == '\t') str_put(&r, " ", 1);
        else str_put(&r, s + here, at - here);
    }
    char *norm = r.s ? r.s : NULL;
    size_t norm_len = r.n;
    if (!ascii && norm_len) {
        char *nfc = NULL;
        size_t nfc_len = 0;
        if (q38_nfc_normalize(r.s, r.n, &nfc, &nfc_len) != 0) die("NFC normalization failed");
        free(r.s);
        norm = nfc;
        norm_len = nfc_len;
    }
    /* strip trailing White_Space */
    while (norm_len > 0) {
        size_t k = norm_len - 1;
        while (k > 0 && ((unsigned char)norm[k] & 0xC0) == 0x80) k--;
        size_t at = k;
        uint32_t cp = utf8_next((const unsigned char *)norm, norm_len, &at);
        if (at != norm_len || !uni_white(cp)) break;
        norm_len = k;
    }
    size_t seg = 0;
    for (size_t p = 0; p < norm_len;) {
        int a = added_at(T, norm, norm_len, p, 1);
        if (a < 0) { p++; continue; }
        metaspace_unigram(T, norm + seg, p - seg, out);
        ivec_push(out, T->added[a].id);
        p += (size_t)T->added[a].len;
        seg = p;
    }
    metaspace_unigram(T, norm + seg, norm_len - seg, out);
    free(norm);
}

/* tokenizer.tokenize(text) as ids: the added tokens split out of the raw text
 * (leftmost, longest), every other segment normalized and encoded. */
static void gl_encode(const GlTok *T, const char *s, size_t n, IntVec *out)
{
    size_t seg = 0;
    for (size_t p = 0; p < n;) {
        int a = added_at(T, s, n, p, 0);
        if (a < 0) { p++; continue; }
        encode_segment(T, s + seg, p - seg, out);
        ivec_push(out, T->added[a].id);
        p += (size_t)T->added[a].len;
        seg = p;
    }
    encode_segment(T, s + seg, n - seg, out);
}

static int added_id(const GlTok *T, const char *content)
{
    for (int k = 0; k < T->n_added; k++)
        if (!strcmp(T->added[k].s, content)) return T->added[k].id;
    return -1;
}

/* ------------------------------------------------------------------ model */

typedef struct {
    QiMat wqkv; float *bqkv;     /* query_proj, key_proj, value_proj stacked: [3d][d] */
    QiMat wo; float *bo;          /* attention.output.dense */
    float *ln1_w, *ln1_b;         /* attention.output.LayerNorm */
    QiMat wi; float *bi;          /* intermediate.dense */
    QiMat wo2; float *bo2;        /* output.dense */
    float *ln2_w, *ln2_b;         /* output.LayerNorm */
    float *pos_q, *pos_k;         /* LayerNorm(rel_embeddings) through query_proj / key_proj: [2*span][d] */
} GlLayer;

typedef struct {
    int d, layers, heads, hd, inter, vocab;
    int span;                     /* pos_ebd_size: relative positions are clamped to [-span, span) */
    int bucket_size, max_position;  /* make_log_bucket_position's arguments */
    int c2p, p2c;                 /* which disentangled terms pos_att_type enables */
    float eps;
    float *tok_emb, *emb_ln_w, *emb_ln_b;
    GlLayer *L;
    QiMat cls1; float *cls1_b;    /* classifier.0: [2d][d] */
    float *cls2_w; float cls2_b;  /* classifier.2: [1][2d] */
    int cls_hidden;
    int max_len;                  /* tokens per request sequence (COLI_GLINER_MAX_LEN) */
    GlTok tok;
    int id_p, id_l, id_sep_struct, id_sep_text;
    char model_name[256];
} Gl;

static float *load_f32(shards *S, const char *name, int64_t expect)
{
    st_tensor *t = st_find(S, name);
    if (!t) die("model.safetensors has no tensor %s", name);
    if (expect >= 0 && t->numel != expect)
        die("%s has %lld elements, expected %lld", name, (long long)t->numel, (long long)expect);
    float *out = (float *)xmalloc((size_t)t->numel * sizeof(float));
    st_read_f32(S, name, out, 0);
    return out;
}

static void layernorm(float *out, const float *x, const float *w, const float *b, int rows, int d, float eps)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * d;
        float *o = out + (size_t)r * d;
        double mean = 0, var = 0;
        for (int i = 0; i < d; i++) mean += xr[i];
        mean /= d;
        for (int i = 0; i < d; i++) { double c = xr[i] - mean; var += c * c; }
        var /= d;
        float rstd = (float)(1.0 / sqrt(var + eps));
        for (int i = 0; i < d; i++) {
            float v = (float)(xr[i] - mean) * rstd * w[i];
            o[i] = b ? v + b[i] : v;
        }
    }
}

static int pos_att_has(jval *types, const char *name)
{
    if (!types) return 0;
    if (types->t == J_STR) return strstr(types->str, name) != NULL;   /* "p2c|c2p" */
    for (int i = 0; types->t == J_ARR && i < types->len; i++)
        if (types->kids[i]->t == J_STR && !strcmp(types->kids[i]->str, name)) return 1;
    return 0;
}

static void load_model(Gl *M, const char *dir)
{
    memset(M, 0, sizeof(*M));
    /* config.json: the GLiNER2 extractor */
    jval *cfg = read_json(dir, "config.json");
    if (strcmp(str_or(cfg, "model_type", ""), "extractor"))
        die("config.json: model_type must be extractor (a GLiNER2 checkpoint)");
    const char *arch = str_or(cfg, "architecture", "span");
    if (strcmp(arch, "span"))
        die("config.json: architecture %s is not supported (only GLiNER2's span architecture)", arch);
    const char *pooling = str_or(cfg, "token_pooling", "first");
    if (strcmp(pooling, "first"))
        die("config.json: token_pooling %s is not supported (only first)", pooling);
    snprintf(M->model_name, sizeof(M->model_name), "%s", str_or(cfg, "model_name", "gliner2"));
    json_free(cfg);

    /* encoder_config/config.json: DeBERTa-v2/v3 */
    jval *ec = read_json(dir, "encoder_config/config.json");
    if (strcmp(str_or(ec, "model_type", ""), "deberta-v2"))
        die("encoder_config/config.json: model_type must be deberta-v2");
    M->d = (int)num_or(ec, "hidden_size", 0);
    M->layers = (int)num_or(ec, "num_hidden_layers", 0);
    M->heads = (int)num_or(ec, "num_attention_heads", 0);
    M->inter = (int)num_or(ec, "intermediate_size", 0);
    M->vocab = (int)num_or(ec, "vocab_size", 0);
    M->eps = (float)num_or(ec, "layer_norm_eps", 1e-7);
    if (M->d < 1 || M->layers < 1 || M->heads < 1 || M->d % M->heads || M->inter < 1 || M->vocab < 1 ||
        M->d > 16384 || M->layers > 256 || M->inter > 1 << 18)
        die("encoder_config/config.json: invalid geometry");
    M->hd = M->d / M->heads;
    if (json_get(ec, "attention_head_size") && (int)num_or(ec, "attention_head_size", 0) != M->hd)
        die("encoder_config/config.json: attention_head_size other than hidden/heads is not supported");
    if (strcmp(str_or(ec, "hidden_act", "gelu"), "gelu"))
        die("encoder_config/config.json: hidden_act %s is not supported (only gelu)", str_or(ec, "hidden_act", ""));
    if (bool_or(ec, "position_biased_input", 1))
        die("encoder_config/config.json: absolute position embeddings (position_biased_input) are not supported");
    if (num_or(ec, "type_vocab_size", 0) > 0) die("encoder_config/config.json: token type embeddings are not supported");
    if (json_get(ec, "embedding_size") && (int)num_or(ec, "embedding_size", M->d) != M->d)
        die("encoder_config/config.json: an embedding_size other than hidden_size is not supported");
    if (num_or(ec, "conv_kernel_size", 0) > 0) die("encoder_config/config.json: the convolution layer is not supported");
    if (!bool_or(ec, "relative_attention", 0)) die("encoder_config/config.json: relative_attention must be true");
    if (!bool_or(ec, "share_att_key", 0)) die("encoder_config/config.json: share_att_key must be true");
    jval *types = json_get(ec, "pos_att_type");
    M->c2p = pos_att_has(types, "c2p");
    M->p2c = pos_att_has(types, "p2c");
    int max_rel = (int)num_or(ec, "max_relative_positions", -1);
    if (max_rel < 1) max_rel = (int)num_or(ec, "max_position_embeddings", 512);
    int buckets = (int)num_or(ec, "position_buckets", -1);
    M->span = buckets > 0 ? buckets : max_rel;
    M->bucket_size = buckets;
    M->max_position = max_rel;
    if (M->span < 1 || M->span > 1 << 16) die("encoder_config/config.json: invalid relative span");
    const char *norm_rel = str_or(ec, "norm_rel_ebd", "none");
    int rel_ln = strstr(norm_rel, "layer_norm") != NULL;
    json_free(ec);

    /* tokenizer */
    load_tokenizer(&M->tok, dir);
    M->id_p = added_id(&M->tok, "[P]");
    M->id_l = added_id(&M->tok, "[L]");
    M->id_sep_struct = added_id(&M->tok, "[SEP_STRUCT]");
    M->id_sep_text = added_id(&M->tok, "[SEP_TEXT]");
    if (M->id_p < 0 || M->id_l < 0 || M->id_sep_struct < 0 || M->id_sep_text < 0 || added_id(&M->tok, "[DESCRIPTION]") < 0)
        die("tokenizer.json: GLiNER2's special tokens ([P], [L], [SEP_STRUCT], [SEP_TEXT], [DESCRIPTION]) "
            "are not added tokens");

    /* weights: the encoder and the classifier; span_rep, count_pred and
     * count_embed serve extraction and are not read */
    shards S;
    st_init(&S, dir);
    int d = M->d, I = M->inter;
    st_tensor *emb = st_find(&S, "encoder.embeddings.word_embeddings.weight");
    if (!emb || emb->rank != 2 || emb->shape[1] != d) die("encoder.embeddings.word_embeddings.weight: expected [vocab, %d]", d);
    M->vocab = (int)emb->shape[0];
    for (int k = 0; k < M->tok.n_added; k++)
        if (M->tok.added[k].id < 0 || M->tok.added[k].id >= M->vocab) die("tokenizer.json: added token id outside the embeddings");
    if (M->tok.n > M->vocab) die("tokenizer.json: %d pieces for %d embedding rows", M->tok.n, M->vocab);
    M->tok_emb = load_f32(&S, "encoder.embeddings.word_embeddings.weight", (int64_t)M->vocab * d);
    M->emb_ln_w = load_f32(&S, "encoder.embeddings.LayerNorm.weight", d);
    M->emb_ln_b = load_f32(&S, "encoder.embeddings.LayerNorm.bias", d);
    float *rel = load_f32(&S, "encoder.encoder.rel_embeddings.weight", (int64_t)2 * M->span * d);
    if (rel_ln) {
        float *w = load_f32(&S, "encoder.encoder.LayerNorm.weight", d);
        float *b = load_f32(&S, "encoder.encoder.LayerNorm.bias", d);
        layernorm(rel, rel, w, b, 2 * M->span, d, M->eps);
        free(w); free(b);
    }
    M->L = (GlLayer *)xcalloc((size_t)M->layers, sizeof(GlLayer));
    char n[256];
    for (int l = 0; l < M->layers; l++) {
        GlLayer *E = &M->L[l];
#define NAME(fmt) (snprintf(n, sizeof(n), "encoder.encoder.layer.%d." fmt, l), n)
        float *wqkv = (float *)xmalloc((size_t)3 * d * d * sizeof(float));
        E->bqkv = (float *)xmalloc((size_t)3 * d * sizeof(float));
        const char *parts[3] = {"query_proj", "key_proj", "value_proj"};
        for (int part = 0; part < 3; part++) {
            char wname[256], bname[256];
            snprintf(wname, sizeof(wname), "encoder.encoder.layer.%d.attention.self.%s.weight", l, parts[part]);
            snprintf(bname, sizeof(bname), "encoder.encoder.layer.%d.attention.self.%s.bias", l, parts[part]);
            float *w = load_f32(&S, wname, (int64_t)d * d), *b = load_f32(&S, bname, d);
            memcpy(wqkv + (size_t)part * d * d, w, (size_t)d * d * sizeof(float));
            memcpy(E->bqkv + (size_t)part * d, b, (size_t)d * sizeof(float));
            free(w); free(b);
        }
        E->wqkv = (QiMat){QI_F32, 3 * d, d, wqkv, NULL, 0};
        E->wo = (QiMat){QI_F32, d, d, load_f32(&S, NAME("attention.output.dense.weight"), (int64_t)d * d), NULL, 0};
        E->bo = load_f32(&S, NAME("attention.output.dense.bias"), d);
        E->ln1_w = load_f32(&S, NAME("attention.output.LayerNorm.weight"), d);
        E->ln1_b = load_f32(&S, NAME("attention.output.LayerNorm.bias"), d);
        E->wi = (QiMat){QI_F32, I, d, load_f32(&S, NAME("intermediate.dense.weight"), (int64_t)I * d), NULL, 0};
        E->bi = load_f32(&S, NAME("intermediate.dense.bias"), I);
        E->wo2 = (QiMat){QI_F32, d, I, load_f32(&S, NAME("output.dense.weight"), (int64_t)d * I), NULL, 0};
        E->bo2 = load_f32(&S, NAME("output.dense.bias"), d);
        E->ln2_w = load_f32(&S, NAME("output.LayerNorm.weight"), d);
        E->ln2_b = load_f32(&S, NAME("output.LayerNorm.bias"), d);
#undef NAME
        /* DisentangledSelfAttention.disentangled_attention_bias with
         * share_att_key: pos_query_layer = query_proj(rel), pos_key_layer =
         * key_proj(rel). The same for every request: computed here, once. */
        QiMat wq = {QI_F32, d, d, wqkv, NULL, 0}, wk = {QI_F32, d, d, wqkv + (size_t)d * d, NULL, 0};
        E->pos_q = (float *)xmalloc((size_t)2 * M->span * d * sizeof(float));
        E->pos_k = (float *)xmalloc((size_t)2 * M->span * d * sizeof(float));
        qi_gemm(E->pos_q, rel, 2 * M->span, &wq, E->bqkv);
        qi_gemm(E->pos_k, rel, 2 * M->span, &wk, E->bqkv + d);
    }
    free(rel);
    st_tensor *c0 = st_find(&S, "classifier.0.weight");
    if (!c0 || c0->rank != 2 || c0->shape[1] != d) die("classifier.0.weight: expected [hidden, %d]", d);
    M->cls_hidden = (int)c0->shape[0];
    M->cls1 = (QiMat){QI_F32, M->cls_hidden, d, load_f32(&S, "classifier.0.weight", (int64_t)M->cls_hidden * d), NULL, 0};
    M->cls1_b = load_f32(&S, "classifier.0.bias", M->cls_hidden);
    M->cls2_w = load_f32(&S, "classifier.2.weight", M->cls_hidden);
    float *b2 = load_f32(&S, "classifier.2.bias", 1);
    M->cls2_b = b2[0];
    free(b2);
    st_destroy(&S);

    M->max_len = 4096;
    const char *env = getenv("COLI_GLINER_MAX_LEN");
    if (env && atoi(env) > 0) M->max_len = atoi(env);
}

/* ------------------------------------------------------------- the forward */

/* make_log_bucket_position, in float32 like the reference (checked against
 * torch for every |relative position| up to 200000). */
static int log_bucket(int rel, int bucket_size, int max_position)
{
    int mid = bucket_size / 2;
    int a = rel < 0 ? -rel : rel;
    if (a <= mid) return rel;
    float lp = ceilf(logf((float)a / (float)mid) / logf((float)(max_position - 1) / (float)mid) *
                     (float)(mid - 1)) + (float)mid;
    return rel < 0 ? -(int)lp : (int)lp;
}

static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

static void add_rows(float *x, const float *y, size_t n)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) x[i] += y[i];
}

/* DisentangledSelfAttention over one sequence of T rows. qkv rows are
 * [q | k | v], heads side by side. With delta = i - j and
 * r = clamp(bucket(delta) + span, 0, 2 span - 1), the score of key j for
 * query i is
 *     (q_i . k_j + q_i . pos_k[r] + k_j . pos_q[r]) / sqrt(hd * (1 + c2p + p2c))
 * (c2p is the second term, p2c the third: the reference gathers p2c at
 * -bucket(j - i), which is the same r because the bucket is odd). */
typedef struct {
    float *p2c;                  /* [H][T][2 span]: k_j . pos_q[r], every head */
    float *vt;                   /* [H][hd][T]: the values, transposed per head */
    int *ridx;                   /* [2T - 1]: r for every delta = i - j */
} AttnScratch;

/* Query rows per task of the attention: a task is one head and ATT_BLOCK rows. */
#define ATT_BLOCK 16

/* DisentangledSelfAttention over one sequence of T rows. qkv rows are
 * [q | k | v], heads side by side. With delta = i - j and
 * r = clamp(bucket(delta) + span, 0, 2 span - 1), the score of key j for
 * query i is
 *     (q_i . k_j + q_i . pos_k[r] + k_j . pos_q[r]) / sqrt(hd * (1 + c2p + p2c))
 * c2p is the second term and p2c the third: the reference gathers p2c at
 * -bucket(j - i), which is the same r because the bucket is odd.
 *
 * Two parallel loops per layer, whatever T: the first projects every key on
 * pos_q and transposes the values, the second runs (head, block of query
 * rows) tasks, each with its own products (qi_gemm on one thread). A loop of
 * small parallel regions per head would be quicker on an idle machine and
 * collapses on a busy one, where every region waits for its slowest thread. */
static void attention(const Gl *M, const GlLayer *E, const float *qkv, float *out, int T, AttnScratch *A)
{
    int d = M->d, H = M->heads, hd = M->hd, R = 2 * M->span;
    int sf = 1 + M->c2p + M->p2c;
    const float inv = 1.0f / sqrtf((float)(hd * sf));
    int *ridx = A->ridx;
    for (int delta = -(T - 1); delta <= T - 1; delta++) {
        int b = M->bucket_size > 0 && M->max_position > 0 ? log_bucket(delta, M->bucket_size, M->max_position) : delta;
        int r = b + M->span;
        ridx[delta + T - 1] = r < 0 ? 0 : r > R - 1 ? R - 1 : r;
    }
    int blocks = (T + ATT_BLOCK - 1) / ATT_BLOCK;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < H * blocks; task++) {
        int h = task / blocks, j0 = (task % blocks) * ATT_BLOCK;
        int n = T - j0 < ATT_BLOCK ? T - j0 : ATT_BLOCK;
        if (M->p2c) {
            QiMat Pq = {QI_F32, R, hd, E->pos_q + (size_t)h * hd, NULL, d};
            qi_gemm_ld(A->p2c + ((size_t)h * T + j0) * R, R, qkv + (size_t)j0 * 3 * d + d + (size_t)h * hd,
                       3 * d, n, &Pq, NULL);
        }
        float *vt = A->vt + (size_t)h * hd * T;
        for (int j = j0; j < j0 + n; j++) {
            const float *v = qkv + (size_t)j * 3 * d + 2 * d + (size_t)h * hd;
            for (int t = 0; t < hd; t++) vt[(size_t)t * T + j] = v[t];
        }
    }
    #pragma omp parallel
    {
        float *score = (float *)xmalloc((size_t)ATT_BLOCK * T * sizeof(float));
        float *c2p = (float *)xmalloc((size_t)ATT_BLOCK * R * sizeof(float));
        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < H * blocks; task++) {
            int h = task / blocks, i0 = (task % blocks) * ATT_BLOCK;
            int n = T - i0 < ATT_BLOCK ? T - i0 : ATT_BLOCK;
            const float *q = qkv + (size_t)i0 * 3 * d + (size_t)h * hd;
            QiMat Kh = {QI_F32, T, hd, qkv + d + (size_t)h * hd, NULL, 3 * d};
            qi_gemm_ld(score, T, q, 3 * d, n, &Kh, NULL);
            if (M->c2p) {
                QiMat Pk = {QI_F32, R, hd, E->pos_k + (size_t)h * hd, NULL, d};
                qi_gemm_ld(c2p, R, q, 3 * d, n, &Pk, NULL);
            }
            const float *p2c = A->p2c + (size_t)h * T * R;
            for (int ii = 0; ii < n; ii++) {
                int i = i0 + ii;
                float *s = score + (size_t)ii * T;
                const float *c = c2p + (size_t)ii * R;
                const int *ri = ridx + i + T - 1;            /* ri[-j] is r for (i, j) */
                float mx = -INFINITY;
                for (int j = 0; j < T; j++) {
                    int r = ri[-j];
                    float x = s[j];
                    if (M->c2p) x += c[r];
                    if (M->p2c) x += p2c[(size_t)j * R + r];
                    x *= inv;
                    s[j] = x;
                    if (x > mx) mx = x;
                }
                double sum = 0;
                for (int j = 0; j < T; j++) { float e = expf(s[j] - mx); s[j] = e; sum += e; }
                float norm = (float)(1.0 / sum);
                for (int j = 0; j < T; j++) s[j] *= norm;
            }
            QiMat Vt = {QI_F32, hd, T, A->vt + (size_t)h * hd * T, NULL, 0};
            qi_gemm_ld(out + (size_t)i0 * d + (size_t)h * hd, d, score, T, n, &Vt, NULL);
        }
        free(score); free(c2p);
    }
}

/* DebertaV2Model.forward on ids[0..T), then the classifier at the rows in
 * marks[0..nm): logits[nm]. */
static void forward(const Gl *M, const int *ids, int T, const int *marks, int nm, double *logits)
{
    int d = M->d, I = M->inter;
    float *x = (float *)xmalloc((size_t)T * d * sizeof(float));
    float *h = (float *)xmalloc((size_t)T * d * sizeof(float));
    float *att = (float *)xmalloc((size_t)T * d * sizeof(float));
    float *big = (float *)xmalloc((size_t)T * (3 * d > I ? 3 * d : I) * sizeof(float));
    AttnScratch A;
    A.p2c = M->p2c ? (float *)xmalloc((size_t)M->heads * T * 2 * M->span * sizeof(float)) : NULL;
    A.vt = (float *)xmalloc((size_t)d * T * sizeof(float));
    A.ridx = (int *)xmalloc((size_t)(2 * T) * sizeof(int));
    for (int r = 0; r < T; r++) {
        int id = ids[r];
        if (id < 0 || id >= M->vocab) id = 0;
        memcpy(h + (size_t)r * d, M->tok_emb + (size_t)id * d, (size_t)d * sizeof(float));
    }
    layernorm(x, h, M->emb_ln_w, M->emb_ln_b, T, d, M->eps);
    for (int l = 0; l < M->layers; l++) {
        const GlLayer *E = &M->L[l];
        qi_gemm(big, x, T, &E->wqkv, E->bqkv);
        attention(M, E, big, att, T, &A);
        qi_gemm(h, att, T, &E->wo, E->bo);
        add_rows(h, x, (size_t)T * d);
        layernorm(x, h, E->ln1_w, E->ln1_b, T, d, M->eps);       /* x = attention_output */
        qi_gemm(big, x, T, &E->wi, E->bi);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)T * I; i++) big[i] = gelu(big[i]);
        qi_gemm(h, big, T, &E->wo2, E->bo2);
        add_rows(h, x, (size_t)T * d);
        layernorm(x, h, E->ln2_w, E->ln2_b, T, d, M->eps);
    }
    /* classifier: Linear -> ReLU -> Linear(., 1) at each [L] */
    float *m = (float *)xmalloc((size_t)(nm > 0 ? nm : 1) * d * sizeof(float));
    float *m1 = (float *)xmalloc((size_t)(nm > 0 ? nm : 1) * M->cls_hidden * sizeof(float));
    for (int k = 0; k < nm; k++) memcpy(m + (size_t)k * d, x + (size_t)marks[k] * d, (size_t)d * sizeof(float));
    if (nm > 0) qi_gemm(m1, m, nm, &M->cls1, M->cls1_b);
    for (int k = 0; k < nm; k++) {
        float *row = m1 + (size_t)k * M->cls_hidden;
        for (int j = 0; j < M->cls_hidden; j++) if (row[j] < 0.f) row[j] = 0.f;
        logits[k] = (double)(dot_f32_lanes(row, M->cls2_w, M->cls_hidden) + M->cls2_b);
    }
    free(m); free(m1); free(x); free(h); free(att); free(big);
    free(A.p2c); free(A.vt); free(A.ridx);
}

/* ------------------------------------------------------- the rendering */

typedef struct {
    IntVec ids;
    int *label_marks;            /* the [L] rows, every question's labels in turn */
    int n_marks;
    int *first_mark;             /* [Q+1]: question q owns label_marks[first_mark[q] .. first_mark[q+1]) */
    int schema_tokens;           /* up to and including [SEP_TEXT] */
    int state_tokens, state_used;
    int words, words_used;
} Rendered;

static void rendered_free(Rendered *r)
{
    free(r->ids.v); free(r->label_marks); free(r->first_mark);
    memset(r, 0, sizeof(*r));
}

/* A question's labels in the order the model reads them: a choice's labels, a
 * score's levels; a noul is asked as "yes" then "no", the order the model
 * card writes every yes/no question in. label_text[i] is option i's text in
 * that order, NULL when it has none. */
static int question_labels(const DecideQuestion *q, const char **label, const char **text)
{
    if (q->type == DECIDE_NOUL) {
        label[0] = "yes"; text[0] = q->options[1].text;
        label[1] = "no";  text[1] = q->options[0].text;
        return 2;
    }
    for (int i = 0; i < q->n_options; i++) { label[i] = q->options[i].label; text[i] = q->options[i].text; }
    return q->n_options;
}

static void push_encoded(const Gl *M, Rendered *r, const char *s)
{
    gl_encode(&M->tok, s, strlen(s), &r->ids);
}

/* processor._transform_record for one record: the schema of every question,
 * [SEP_TEXT], then as many words of the state as fit in max_len tokens.
 * Returns 1, or 0 with the reason when the questions alone do not fit. */
static int render(const Gl *M, const DecideRecord *rec, Rendered *r, char *err, size_t cap)
{
    memset(r, 0, sizeof(*r));
    int total_labels = 0;
    for (int q = 0; q < rec->n_questions; q++) total_labels += rec->questions[q].n_options;
    r->label_marks = (int *)xmalloc((size_t)(total_labels > 0 ? total_labels : 1) * sizeof(int));
    r->first_mark = (int *)xmalloc((size_t)(rec->n_questions + 1) * sizeof(int));
    const char *label[DECIDE_MAX_OPTIONS], *text[DECIDE_MAX_OPTIONS];
    for (int q = 0; q < rec->n_questions; q++) {
        const DecideQuestion *Q = &rec->questions[q];
        int k = question_labels(Q, label, text);
        if (q) ivec_push(&r->ids, M->id_sep_struct);
        /* ( [P] prompt_str ( [L] label ... ) ) */
        push_encoded(M, r, "(");
        ivec_push(&r->ids, M->id_p);
        Str prompt = {0};
        str_put(&prompt, Q->id, strlen(Q->id));
        if (Q->instructions[0]) { str_put(&prompt, ": ", 2); str_put(&prompt, Q->instructions, strlen(Q->instructions)); }
        for (int i = 0; i < k; i++) {
            if (!text[i] || !text[i][0]) continue;
            str_put(&prompt, " [DESCRIPTION] ", 15);
            str_put(&prompt, label[i], strlen(label[i]));
            str_put(&prompt, ": ", 2);
            str_put(&prompt, text[i], strlen(text[i]));
        }
        gl_encode(&M->tok, prompt.s, prompt.n, &r->ids);
        free(prompt.s);
        push_encoded(M, r, "(");
        r->first_mark[q] = r->n_marks;
        for (int i = 0; i < k; i++) {
            r->label_marks[r->n_marks++] = r->ids.n;
            ivec_push(&r->ids, M->id_l);
            push_encoded(M, r, label[i]);
        }
        push_encoded(M, r, ")");
        push_encoded(M, r, ")");
    }
    r->first_mark[rec->n_questions] = r->n_marks;
    ivec_push(&r->ids, M->id_sep_text);
    r->schema_tokens = r->ids.n;
    if (r->schema_tokens > M->max_len) {
        snprintf(err, cap, "questions: the questions and their options take %d tokens, more than "
                 "max_len=%d (COLI_GLINER_MAX_LEN); ask fewer questions or shorten the options",
                 r->schema_tokens, M->max_len);
        return 0;
    }
    /* the state: "." appended unless it ends in . ! or ? (processor._collate_batch) */
    size_t sl = strlen(rec->state);
    char *state = (char *)xmalloc(sl + 2);
    memcpy(state, rec->state, sl + 1);
    if (!sl) { state[0] = '.'; state[1] = 0; sl = 1; }
    else if (state[sl - 1] != '.' && state[sl - 1] != '!' && state[sl - 1] != '?') { state[sl] = '.'; state[sl + 1] = 0; sl++; }
    Words w;
    split_words(state, sl, &w);
    free(state);
    r->words = w.n;
    int full = 1;
    IntVec piece = {0};
    for (int i = 0; i < w.n; i++) {
        piece.n = 0;
        gl_encode(&M->tok, w.w[i], strlen(w.w[i]), &piece);
        r->state_tokens += piece.n;
        if (full && r->ids.n + piece.n <= M->max_len) {
            for (int t = 0; t < piece.n; t++) ivec_push(&r->ids, piece.v[t]);
            r->state_used += piece.n;
            r->words_used++;
        } else {
            full = 0;          /* max_len counts whole words from the start */
        }
    }
    free(piece.v);
    words_free(&w);
    return 1;
}

/* ------------------------------------------------------- the DECIDE call */

typedef struct {
    Gl *M;
    int dump_ids;
    Rendered last;               /* the last record's sequence, for --records --ids */
} GlCtx;

static int gl_decide(void *opaque, const DecideRecord *rec, DecideAnswer *answers,
                     int *input_tokens, char *err, size_t cap)
{
    GlCtx *ctx = (GlCtx *)opaque;
    Gl *M = ctx->M;
    rendered_free(&ctx->last);
    Rendered r;
    if (!render(M, rec, &r, err, cap)) { rendered_free(&r); return 0; }
    double *logits = (double *)xcalloc((size_t)(r.n_marks > 0 ? r.n_marks : 1), sizeof(double));
    forward(M, r.ids.v, r.ids.n, r.label_marks, r.n_marks, logits);
    for (int q = 0; q < rec->n_questions; q++) {
        const DecideQuestion *Q = &rec->questions[q];
        DecideAnswer *a = &answers[q];
        int k = r.first_mark[q + 1] - r.first_mark[q];
        const double *z = logits + r.first_mark[q];
        a->n = k;
        a->logits = (double *)xcalloc((size_t)k, sizeof(double));
        a->probs = (double *)xcalloc((size_t)k, sizeof(double));
        a->temperature = NAN;            /* no calibration ships with the checkpoint */
        /* softmax over the task's labels, as _extract_classification_result does
         * for a single-label task (temperature 1) */
        double mx = -INFINITY, sum = 0;
        for (int i = 0; i < k; i++) if (z[i] > mx) mx = z[i];
        double *p = (double *)xmalloc((size_t)k * sizeof(double));
        for (int i = 0; i < k; i++) { p[i] = exp(z[i] - mx); sum += p[i]; }
        for (int i = 0; i < k; i++) p[i] /= sum;
        for (int i = 0; i < k; i++) {
            /* back to the record's option order: a noul was read as yes, no */
            int src = Q->type == DECIDE_NOUL ? 1 - i : i;
            a->logits[i] = z[src];
            a->probs[i] = p[src];
        }
        free(p);
        a->n_actions = 0;
        a->tokens = r.ids.n;
        a->state_tokens = r.state_tokens;
        a->state_dropped = r.state_tokens - r.state_used;
    }
    free(logits);
    *input_tokens = r.ids.n;
    if (ctx->dump_ids) ctx->last = r;
    else rendered_free(&r);
    return 1;
}

/* ------------------------------------------------------------------- main */

static double rss_gb(void)
{
#ifdef __linux__
    FILE *f = fopen("/proc/self/statm", "r");
    long pages = 0, resident = 0;
    if (f) { if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0; fclose(f); }
    return resident * (double)sysconf(_SC_PAGESIZE) / 1e9;
#else
    return 0.0;
#endif
}

static void print_json_string(FILE *out, const char *s)
{
    DecideBuf b = {0};
    decide_buf_string(&b, s);
    fputs(b.data ? b.data : "\"\"", out);
    free(b.data);
}

static jval *read_string_array(const char *path, char **text_out)
{
    char *text = read_text(path);
    if (!text) die("cannot read %s", path);
    jval *root = json_parse_checked(text);
    if (!root || root->t != J_ARR) die("%s: expected a JSON array", path);
    *text_out = text;
    return root;
}

static int run_records(Gl *M, const char *path, int dump_ids)
{
    char *text;
    jval *root = read_string_array(path, &text);
    GlCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.M = M;
    ctx.dump_ids = dump_ids;
    /* Each record is the text of a DECIDE payload and goes through
     * decide_record_parse, the path the serve loop takes. */
    for (int i = 0; i < root->len; i++) {
        jval *payload = json_get(root->kids[i], "payload");
        if (!payload || payload->t != J_STR) die("%s: record %d has no string \"payload\"", path, i);
        DecideRecord rec;
        char reason[1024];
        if (!decide_record_parse(payload->str, &rec, reason, sizeof(reason))) {
            printf("{\"error\":"); print_json_string(stdout, reason); printf("}\n");
            continue;
        }
        DecideAnswer *answers = (DecideAnswer *)xcalloc((size_t)rec.n_questions, sizeof(DecideAnswer));
        int tokens = 0;
        double t0 = now_ms();
        int ok = gl_decide(&ctx, &rec, answers, &tokens, reason, sizeof(reason));
        double ms = now_ms() - t0;
        if (!ok) {
            printf("{\"error\":"); print_json_string(stdout, reason); printf("}\n");
        } else {
            size_t len = 0;
            char *json = decide_answers_json(&rec, answers, tokens, ms, &len);
            if (dump_ids && json && len > 1) {
                json[len - 1] = 0;                 /* reopen the object */
                fputs(json, stdout);
                printf(",\"sequence\":{\"ids\":[");
                for (int t = 0; t < ctx.last.ids.n; t++) printf("%s%d", t ? "," : "", ctx.last.ids.v[t]);
                printf("],\"markers\":[");
                for (int t = 0; t < ctx.last.n_marks; t++) printf("%s%d", t ? "," : "", ctx.last.label_marks[t]);
                printf("],\"words\":%d,\"words_used\":%d}}\n", ctx.last.words, ctx.last.words_used);
            } else if (json) {
                fputs(json, stdout); fputc('\n', stdout);
            }
            free(json);
        }
        fflush(stdout);
        decide_answers_free(answers, rec.n_questions);
        free(answers);
        decide_record_free(&rec);
    }
    rendered_free(&ctx.last);
    json_free(root);
    free(text);
    return 0;
}

static int run_tokenize(Gl *M, const char *path)
{
    char *text;
    jval *root = read_string_array(path, &text);
    printf("[");
    for (int i = 0; i < root->len; i++) {
        jval *s = root->kids[i];
        if (s->t != J_STR) die("%s: entry %d is not a string", path, i);
        IntVec ids = {0};
        gl_encode(&M->tok, s->str, strlen(s->str), &ids);
        printf("%s[", i ? "," : "");
        for (int t = 0; t < ids.n; t++) printf("%s%d", t ? "," : "", ids.v[t]);
        printf("]");
        free(ids.v);
    }
    printf("]\n");
    json_free(root);
    free(text);
    return 0;
}

static int run_split(const char *path)
{
    char *text;
    jval *root = read_string_array(path, &text);
    printf("[");
    for (int i = 0; i < root->len; i++) {
        jval *s = root->kids[i];
        if (s->t != J_STR) die("%s: entry %d is not a string", path, i);
        Words w;
        split_words(s->str, strlen(s->str), &w);
        printf("%s[", i ? "," : "");
        for (int t = 0; t < w.n; t++) { if (t) printf(","); print_json_string(stdout, w.w[t]); }
        printf("]");
        words_free(&w);
    }
    printf("]\n");
    json_free(root);
    free(text);
    return 0;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *records = NULL, *tokenize = NULL, *split = NULL;
    int dump_ids = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--records") && i + 1 < argc) records = argv[++i];
        else if (!strcmp(argv[i], "--tokenize") && i + 1 < argc) tokenize = argv[++i];
        else if (!strcmp(argv[i], "--split") && i + 1 < argc) split = argv[++i];
        else if (!strcmp(argv[i], "--ids")) dump_ids = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("usage: SERVE=1 SNAP=<dir> gliner_decide            serve DECIDE on stdin/stdout\n"
                   "       gliner_decide --model DIR --records FILE [--ids]\n"
                   "       gliner_decide --model DIR --tokenize FILE\n"
                   "       gliner_decide --split FILE\n");
            return 0;
        }
        /* a bare number is the gateway's cache-cap argument: nothing to cache here */
    }
    if (split) return run_split(split);
    const char *serve = getenv("SERVE");
    int serving = serve && !strcmp(serve, "1") && !records && !tokenize;
    if (!model) model = getenv("SNAP");
    if (!model) die("no model: pass --model DIR or set SNAP");
    if (serving) coli_serve_stdio_init();
    coli_omp_tune_threads("gliner_decide");   /* physical cores: SMT halves the GEMMs here too */
#ifdef _OPENMP
    omp_set_max_active_levels(1);             /* the attention's qi_gemm calls run inside its tasks */
#endif
    double t0 = now_ms();
    Gl *M = (Gl *)xcalloc(1, sizeof(Gl));
    load_model(M, model);
    fprintf(stderr, "[gliner_decide] %s: DeBERTa-v2 %d layers x %d, %d heads, relative span %d; "
            "max_len %d; loaded in %.1f s\n", M->model_name, M->layers, M->d, M->heads, M->span,
            M->max_len, (now_ms() - t0) / 1e3);
    if (tokenize) return run_tokenize(M, tokenize);
    if (records) return run_records(M, records, dump_ids);
    if (!serving) die("nothing to do: set SERVE=1, or pass --records / --tokenize / --split");
    int slots = getenv("KV_SLOTS") ? atoi(getenv("KV_SLOTS")) : 1;
    if (slots < 1) slots = 1;
    char caps[256];
    snprintf(caps, sizeof(caps), "decide=1 chat=0 max_len=%d", M->max_len);
    GlCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.M = M;
    return decide_serve(stdin, stdout, gl_decide, &ctx, caps, rss_gb(), slots);
}
