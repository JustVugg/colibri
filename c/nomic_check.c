/* nomic_check.c -- nomic-embed-text on colibri primitives, checkpoint tool.
 *
 * Steps 1-4 of the sideline plan's proposed order of work
 * (docs/experiments/SIDELINE_PLAN_EMBEDDINGS_GEMMA4_MULTIMODEL.md section 6),
 * in one standalone artifact:
 *
 *   1. loader: read the llama.cpp GGUF in place (gguf.h; nothing is mapped,
 *      tensors are converted to f32 and held in RAM, the laya.c convention);
 *   2. tokenizer: BERT WordPiece, built from the vocab embedded in the GGUF
 *      (ASCII lowercasing only -- no NFD accent stripping yet, documented);
 *   3. forward: the BERT post-norm encoder loop -- laya.c's primitives
 *      (fused-QKV gemm, full bidirectional attention, rotary, layernorm with
 *      optional bias, gated gelu MLP) composed in nomic-bert's order:
 *      norm after each residual add, no final norm on top;
 *   4. head: mean pooling over tokens + L2 normalisation.
 *
 * The oracle comparison (step 5) lives outside this file: run Ollama's
 * /v1/embeddings on the same text and dot the two vectors.
 *
 * Modes:
 *   nomic-check <gguf> --dump            metadata and tensor inventory
 *   nomic-check <gguf> --tok "text"      token ids ([CLS] ... [SEP])
 *   nomic-check <gguf> --embed "text"    the 768-vector, JSON on stdout
 *
 * Verified against the Ollama blob of nomic-embed-text:latest
 * (nomic-embed-text-v1.5, arch nomic-bert, 12 layers x 768, rotary theta
 * 1000, pooling_type 1 = mean). Weights: F16 file, f32 in RAM (~550 MB).
 */
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf.h"
#include "serve_codec.h"

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "nomic-check: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

typedef struct {
    int d, L, nh, hd, inter, vocab;
    float eps, theta;
    float *tok_emb;                     /* [vocab][d] */
    float *tt_emb;                      /* [2][d] token_types.weight, type 0 used */
    float *emb_ln_w, *emb_ln_b;
    float *wqkv, *wo, *wg, *wu, *wd;    /* [L] packed: L * (N*K) each */
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b;   /* [L][d] attn_output_norm, layer_output_norm */
    char **tok_strs;                     /* GGUF strings, borrowed from the ctx */
    int *vocab_sorted;                   /* indices into vocab, strcmp order */
    int cls_id, sep_id, unk_id;
} Nomic;

/* Sweep knobs (oracle search): the tensor data is fixed, so if the reference
 * disagrees the difference is in composition. Each is an env var, so the
 * binary can sweep variants without a rebuild. */
static int   g_pre = 0;        /* NOMIC_NORM=pre: laya-style pre-norm instead of post-norm */
static int   g_act = 2;        /* NOMIC_ACT: 0 gelu-erf, 1 gelu-tanh, 2 silu -- the oracle-resolved value */
static int   g_swap = 0;       /* NOMIC_SWAP=1: gelu applied to the *up* projection */
static int   g_rope = 1;   /* NOMIC_ROPE: 0 none, 1 half-split (oracle-resolved), 2 interleaved */
static int   g_pool_core = 0;  /* NOMIC_POOL=core: pool without [CLS]/[SEP] */
static int   g_no_tt = 0;      /* NOMIC_TT=0: no token-type embedding add */

static void read_env(void)
{
    const char *v;
    if ((v = getenv("NOMIC_NORM")) && !strcmp(v, "pre")) g_pre = 1;
    if ((v = getenv("NOMIC_ACT"))) {
        if (!strcmp(v, "tanh")) g_act = 1;
        else if (!strcmp(v, "silu")) g_act = 2;
    }
    if ((v = getenv("NOMIC_SWAP")) && !strcmp(v, "1")) g_swap = 1;
    if ((v = getenv("NOMIC_ROPE")) && !strcmp(v, "0")) g_rope = 0;
    if ((v = getenv("NOMIC_ROPE")) && !strcmp(v, "2")) g_rope = 2;
    if ((v = getenv("NOMIC_POOL")) && !strcmp(v, "core")) g_pool_core = 1;
    if ((v = getenv("NOMIC_TT")) && !strcmp(v, "0")) g_no_tt = 1;
}

/* Tensor layout: gguf order, ne[0] is the contiguous length. A weight tensor
 * [K, N] is N rows of K, and y[j] = sum_k W[j*K + k] * x[k]. */
static float *load_tensor(gguf_ctx *G, const char *name, int64_t *K, int64_t *N)
{
    const gguf_tensor *t = gguf_find(G, name);
    if (!t) die("GGUF has no tensor %s", name);
    if (t->n_dims > 2) die("%s: %d dims, expected 1 or 2", name, t->n_dims);
    int64_t k = t->ne[0], n = t->n_dims > 1 ? t->ne[1] : 1;
    float *out = xmalloc((size_t)(k * n) * sizeof(float));
    if (gguf_read_f32(G, t, out)) die("read %s failed", name);
    if (K) *K = k;
    if (N) *N = n;
    return out;
}

static float *load_tensor_fixed(gguf_ctx *G, const char *name, int64_t k, int64_t n)
{
    int64_t rk = 0, rn = 0;
    float *out = load_tensor(G, name, &rk, &rn);
    if (rk != k || rn != n) die("%s: [%lld,%lld], expected [%lld,%lld]",
                                name, (long long)rk, (long long)rn, (long long)k, (long long)n);
    return out;
}

static const char *need_str(gguf_ctx *G, const char *key)
{
    const gguf_kv *kv = gguf_get(G, key);
    if (!kv || kv->type != GGUF_T_STRING) die("GGUF metadata %s missing", key);
    return kv->s;
}

static float need_f32(gguf_ctx *G, const char *key)
{
    const gguf_kv *kv = gguf_get(G, key);
    if (!kv || (kv->type != GGUF_T_F32 && kv->type != GGUF_T_F64)) die("GGUF metadata %s missing", key);
    return (float)kv->v.f;
}

static int need_i64(gguf_ctx *G, const char *key)
{
    return (int)gguf_get_int(G, key, -1);
}

static void load_model(Nomic *M, gguf_ctx *G, const char *path)
{
    const char *arch = need_str(G, "general.architecture");
    if (strcmp(arch, "nomic-bert")) die("architecture %s, expected nomic-bert", arch);
    M->d = need_i64(G, "nomic-bert.embedding_length");
    M->L = need_i64(G, "nomic-bert.block_count");
    M->nh = need_i64(G, "nomic-bert.attention.head_count");
    M->inter = need_i64(G, "nomic-bert.feed_forward_length");
    M->eps = need_f32(G, "nomic-bert.attention.layer_norm_epsilon");
    M->theta = need_f32(G, "nomic-bert.rope.freq_base");
    if (M->d <= 0 || M->L <= 0 || M->nh <= 0 || M->inter <= 0) die("bad metadata");
    M->hd = M->d / M->nh;
    if (M->hd * M->nh != M->d) die("heads do not divide hidden");
    M->vocab = need_i64(G, "tokenizer.ggml.token_type_count");  /* placeholder, fixed below */

    const gguf_kv *toks = gguf_get(G, "tokenizer.ggml.tokens");
    if (!toks || toks->type != GGUF_T_ARRAY || toks->atype != GGUF_T_STRING)
        die("no tokenizer.ggml.tokens array");
    M->vocab = (int)toks->n;
    M->tok_strs = toks->strs;
    M->vocab_sorted = xmalloc((size_t)M->vocab * sizeof(int));
    for (int i = 0; i < M->vocab; i++) M->vocab_sorted[i] = i;
    /* insertion sort by strcmp: 30522 entries, a startup cost of milliseconds */
    for (int i = 1; i < M->vocab; i++) {
        int v = M->vocab_sorted[i], j = i - 1;
        while (j >= 0 && strcmp(M->tok_strs[M->vocab_sorted[j]], M->tok_strs[v]) > 0) {
            M->vocab_sorted[j + 1] = M->vocab_sorted[j];
            j--;
        }
        M->vocab_sorted[j + 1] = v;
    }
    M->cls_id = need_i64(G, "tokenizer.ggml.cls_token_id");
    M->sep_id = need_i64(G, "tokenizer.ggml.seperator_token_id");  /* the key's own typo */
    M->unk_id = need_i64(G, "tokenizer.ggml.unknown_token_id");

    M->tok_emb = load_tensor_fixed(G, "token_embd.weight", M->d, M->vocab);
    M->tt_emb = load_tensor_fixed(G, "token_types.weight", M->d, 2);
    M->emb_ln_w = load_tensor_fixed(G, "token_embd_norm.weight", M->d, 1);
    M->emb_ln_b = load_tensor_fixed(G, "token_embd_norm.bias", M->d, 1);
    M->wqkv = xmalloc((size_t)M->L * (size_t)M->d * (size_t)3 * M->d * sizeof(float));
    M->wo = xmalloc((size_t)M->L * (size_t)M->d * M->d * sizeof(float));
    M->wg = xmalloc((size_t)M->L * (size_t)M->d * (size_t)M->inter * sizeof(float));
    M->wu = xmalloc((size_t)M->L * (size_t)M->d * (size_t)M->inter * sizeof(float));
    M->wd = xmalloc((size_t)M->L * (size_t)M->inter * (size_t)M->d * sizeof(float));
    M->ln1_w = xmalloc((size_t)M->L * (size_t)M->d * sizeof(float));
    M->ln1_b = xmalloc((size_t)M->L * (size_t)M->d * sizeof(float));
    M->ln2_w = xmalloc((size_t)M->L * (size_t)M->d * sizeof(float));
    M->ln2_b = xmalloc((size_t)M->L * (size_t)M->d * sizeof(float));
    for (int l = 0; l < M->L; l++) {
        char name[64];
        float *t;
#define PER(BLK, DST, K, N) \
        snprintf(name, sizeof name, "blk.%d." BLK, l); \
        t = load_tensor_fixed(G, name, (K), (N)); \
        memcpy(M->DST + (size_t)l * (N) * (K), t, (size_t)(K) * (N) * sizeof(float)); \
        free(t)
        PER("attn_qkv.weight", wqkv, M->d, 3 * M->d);
        PER("attn_output.weight", wo, M->d, M->d);
        PER("ffn_gate.weight", wg, M->d, M->inter);
        PER("ffn_up.weight", wu, M->d, M->inter);
        PER("ffn_down.weight", wd, M->inter, M->d);
        PER("attn_output_norm.weight", ln1_w, M->d, 1);
        PER("attn_output_norm.bias", ln1_b, M->d, 1);
        PER("layer_output_norm.weight", ln2_w, M->d, 1);
        PER("layer_output_norm.bias", ln2_b, M->d, 1);
#undef PER
    }
    (void)path;
}

/* ------------------------------------------------------- tokenizer ------ */

/* BERT WordPiece (bert-base-uncased style), ASCII-only lowercasing for now.
 * Accent stripping (NFD + combining-mark removal) is not implemented; the
 * oracle inputs avoid accented characters. Documented limitation. */

static int vocab_find(Nomic *M, const char *s)
{
    int lo = 0, hi = M->vocab - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, id = M->vocab_sorted[mid];
        int c = strcmp(M->tok_strs[id], s);
        if (c == 0) return id;
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static int is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static int is_punct(char c)
{
    return strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", c) != NULL && c != '\0';
}

/* Greedy longest-match WordPiece for one already-lowercased word.
 * Returns the number of ids written (<= cap) or -1 if the word exceeds
 * BERT's 100-character limit.
 *
 * The GGUF vocab is converted from the HF WordPiece form: a word-initial
 * piece "the" is stored "▁the" (U+2581 leading) and a continuation "##the"
 * is stored plain "the". So a start-of-word match tries "▁"+piece (falling
 * back to the plain piece, which carries the punctuation originals), and a
 * continuation matches the plain piece. */
static int wordpiece(Nomic *M, const char *w, int *ids, int cap)
{
    int len = (int)strlen(w);
    if (len > 100) return -1;
    static const char *SPM = "\xe2\x96\x81";    /* ▁ */
    int n = 0, start = 0;
    while (start < len) {
        int end = len, found = -1;
        char buf[128];
        while (end > start) {
            int np = end - start;
            if (np + 4 > (int)sizeof buf) { end--; continue; }
            if (start == 0) {
                /* word-initial pieces are the "▁"-prefixed ones; the plain
                 * piece is the HF "##" continuation and is never valid here */
                memcpy(buf, SPM, 3); memcpy(buf + 3, w + start, (size_t)np); buf[np + 3] = 0;
                found = vocab_find(M, buf);
            } else {
                memcpy(buf, w + start, (size_t)np); buf[np] = 0;
                found = vocab_find(M, buf);
            }
            if (found >= 0) break;
            end--;
        }
        if (found < 0) return -1;       /* whole word becomes [UNK] */
        if (n >= cap) return -1;
        ids[n++] = found;
        start = end;
    }
    return n;
}

/* One input string -> ids with [CLS] ... [SEP]. ASCII input assumed. */
static int tokenize(Nomic *M, const char *text, int *ids, int cap)
{
    int n = 0;
    if (n >= cap) return -1;
    ids[n++] = M->cls_id;
    size_t len = strlen(text);
    char word[128];
    size_t i = 0;
    while (i < len) {
        size_t w = 0;
        if (is_space(text[i])) { i++; continue; }
        if (is_punct(text[i])) {              /* punctuation: one token per char */
            word[w++] = text[i]; i++;
        } else {
            size_t wstart = i;
            while (i < len && !is_space(text[i]) && !is_punct(text[i])) {
                if (w + 1 < sizeof word && (size_t)(i - wstart) < 100)
                    word[w++] = (char)tolower((unsigned char)text[i]);
                i++;
            }
        }
        word[w] = 0;
        int out[128], r = wordpiece(M, word, out, 128);
        if (r < 0) { out[0] = M->unk_id; r = 1; }
        for (int j = 0; j < r && n < cap; j++) ids[n++] = out[j];
    }
    if (n >= cap) return -1;
    ids[n++] = M->sep_id;
    return n;
}

/* ------------------------------------------------------------ forward --- */

/* y[m] = sum_k W[m*K + k] * x[k]: a [K,N] tensor read as N rows of K. */
static void matvec(float *y, const float *x, const float *W, int64_t N, int64_t K)
{
    for (int64_t j = 0; j < N; j++) {
        const float *w = W + (size_t)j * K;
        float s = 0.f;
        for (int64_t k = 0; k < K; k++) s += w[k] * x[k];
        y[j] = s;
    }
}

/* LayerNorm over one row: biased variance, eps inside the root. In place
 * (src == dst) or from src into dst (the pre-norm variants). */
static void layernorm_to(float *dst, const float *src, const float *w, const float *b, int d, float eps)
{
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < d; i++) mean += src[i];
    mean /= d;
    for (int i = 0; i < d; i++) { float c = src[i] - mean; var += c * c; }
    var /= d;
    float rstd = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < d; i++) dst[i] = (src[i] - mean) * rstd * w[i] + b[i];
}

static void layernorm(float *x, const float *w, const float *b, int d, float eps)
{
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= d;
    for (int i = 0; i < d; i++) { float c = x[i] - mean; var += c * c; }
    var /= d;
    float rstd = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < d; i++) x[i] = (x[i] - mean) * rstd * w[i] + b[i];
}

/* q and k rows of qkv rotated in place. Two forms: 1 = half-split
 * (v[i], v[i+hd/2] pairs, the rotate_half convention), 2 = interleaved
 * (v[2i], v[2i+1] pairs). Positions 0..T-1, theta from the file. */
static void apply_rope(float *qkv, int T, int d, int hd, float theta)
{
    int half = hd / 2;
    float *cosv = xmalloc((size_t)T * half * sizeof(float));
    float *sinv = xmalloc((size_t)T * half * sizeof(float));
    for (int p = 0; p < T; p++)
        for (int i = 0; i < half; i++) {
            float a = (float)p / powf(theta, (float)(2 * i) / (float)hd);
            cosv[(size_t)p * half + i] = cosf(a);
            sinv[(size_t)p * half + i] = sinf(a);
        }
    for (int r = 0; r < T; r++) {
        const float *c = cosv + (size_t)r * half, *s = sinv + (size_t)r * half;
        for (int part = 0; part < 2; part++) {
            float *base = qkv + (size_t)r * 3 * d + (size_t)part * d;
            for (int h = 0; h * hd < d; h++) {
                float *v = base + h * hd;
                if (g_rope == 2) {          /* interleaved pairs */
                    for (int i = 0; i < half; i++) {
                        float a = v[2 * i], b2 = v[2 * i + 1];
                        v[2 * i] = a * c[i] - b2 * s[i];
                        v[2 * i + 1] = b2 * c[i] + a * s[i];
                    }
                } else {                    /* half-split pairs */
                    for (int i = 0; i < half; i++) {
                        float a = v[i], b2 = v[i + half];
                        v[i] = a * c[i] - b2 * s[i];
                        v[i + half] = b2 * c[i] + a * s[i];
                    }
                }
            }
        }
    }
    free(cosv); free(sinv);
}

/* Full bidirectional softmax attention, single sequence, per head. */
static void attention(float *qkv, float *out, int T, int d, int heads, int hd)
{
    const float scale = 1.0f / sqrtf((float)hd);
    float *score = xmalloc((size_t)(T > 0 ? T : 1) * sizeof(float));
    for (int r = 0; r < T; r++) {
        const float *q = qkv + (size_t)r * 3 * d;
        for (int h = 0; h < heads; h++) {
            float *dst = out + (size_t)r * d + h * hd;
            const float *qp = q + h * hd;
            float mx = -INFINITY;
            for (int j = 0; j < T; j++) {
                const float *k = qkv + (size_t)j * 3 * d + d + h * hd;
                float s = 0.f;
                for (int t = 0; t < hd; t++) s += qp[t] * k[t];
                s *= scale;
                score[j] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.f;
            for (int j = 0; j < T; j++) { score[j] = expf(score[j] - mx); sum += score[j]; }
            float inv = 1.0f / sum;
            for (int t = 0; t < hd; t++) dst[t] = 0.f;
            for (int j = 0; j < T; j++) {
                const float *v = qkv + (size_t)j * 3 * d + 2 * d + h * hd;
                float p = score[j] * inv;
                for (int t = 0; t < hd; t++) dst[t] += p * v[t];
            }
        }
    }
    free(score);
}

static inline float gelu_erf(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

static inline float gelu_tanh(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

static inline float silu(float x)
{
    return x / (1.0f + expf(-x));
}

/* Sweep knobs (oracle search): the tensor data is fixed, so if the reference
 * disagrees the difference is in composition. Each is an env var, so the
 * binary can sweep variants without a rebuild. */
/* The encoder: embeddings + norm, then L post-norm blocks, no final norm. */
static void forward(Nomic *M, const int *ids, int T, float *out /* [d] */)
{
    int d = M->d, I = M->inter;
    float *x = xmalloc((size_t)T * d * sizeof(float));
    float *h = xmalloc((size_t)T * d * sizeof(float));
    float *qkv = xmalloc((size_t)T * 3 * d * sizeof(float));
    float *att = xmalloc((size_t)T * d * sizeof(float));
    float *gi = xmalloc((size_t)I * sizeof(float));
    float *ui = xmalloc((size_t)I * sizeof(float));

    for (int r = 0; r < T; r++) {
        int id = ids[r];
        if (id < 0 || id >= M->vocab) id = M->unk_id;
        const float *e = M->tok_emb + (size_t)id * d;
        const float *tt = M->tt_emb;      /* type 0 */
        for (int i = 0; i < d; i++) x[(size_t)r * d + i] = e[i] + (g_no_tt ? 0.f : tt[i]);
    }
    for (int r = 0; r < T; r++)
        layernorm(x + (size_t)r * d, M->emb_ln_w, M->emb_ln_b, d, M->eps);

    for (int l = 0; l < M->L; l++) {
        const float *Wq = M->wqkv + (size_t)l * d * 3 * d;
        const float *Wo = M->wo + (size_t)l * d * d;
        const float *Wg = M->wg + (size_t)l * d * I;
        const float *Wu = M->wu + (size_t)l * d * I;
        const float *Wd = M->wd + (size_t)l * I * d;
        const float *l1w = M->ln1_w + (size_t)l * d, *l1b = M->ln1_b + (size_t)l * d;
        const float *l2w = M->ln2_w + (size_t)l * d, *l2b = M->ln2_b + (size_t)l * d;

        if (g_pre) {
            /* pre-norm (laya order): norm BEFORE each gemm, residual untouched */
            for (int r = 0; r < T; r++)
                layernorm_to(h + (size_t)r * d, x + (size_t)r * d, l1w, l1b, d, M->eps);
        } else {
            /* post-norm (BERT order): gemm straight from the residual */
            for (int r = 0; r < T; r++)
                memcpy(h + (size_t)r * d, x + (size_t)r * d, (size_t)d * sizeof(float));
        }
        for (int r = 0; r < T; r++)
            matvec(qkv + (size_t)r * 3 * d, h + (size_t)r * d, Wq, 3 * d, d);
        if (g_rope) apply_rope(qkv, T, d, M->hd, M->theta);
        attention(qkv, att, T, d, M->nh, M->hd);
        for (int r = 0; r < T; r++)
            matvec(h + (size_t)r * d, att + (size_t)r * d, Wo, d, d);
        for (int r = 0; r < T; r++) {
            float *xr = x + (size_t)r * d, *hr = h + (size_t)r * d;
            for (int i = 0; i < d; i++) xr[i] += hr[i];
        }
        if (g_pre) {
            for (int r = 0; r < T; r++)
                layernorm_to(h + (size_t)r * d, x + (size_t)r * d, l2w, l2b, d, M->eps);
        } else {
            for (int r = 0; r < T; r++)
                layernorm(x + (size_t)r * d, l1w, l1b, d, M->eps);
            for (int r = 0; r < T; r++)
                memcpy(h + (size_t)r * d, x + (size_t)r * d, (size_t)d * sizeof(float));
        }
        for (int r = 0; r < T; r++) {
            const float *xr = h + (size_t)r * d;
            matvec(gi, xr, g_swap ? Wu : Wg, I, d);
            matvec(ui, xr, g_swap ? Wg : Wu, I, d);
            for (int j = 0; j < I; j++) {
                float a = g_swap ? ui[j] : gi[j];
                float b2 = g_swap ? gi[j] : ui[j];
                gi[j] = (g_act == 2 ? silu(a) : g_act == 1 ? gelu_tanh(a) : gelu_erf(a)) * b2;
            }
            matvec(att + (size_t)r * d, gi, Wd, d, I);
        }
        for (int r = 0; r < T; r++) {
            float *xr = x + (size_t)r * d, *hr = att + (size_t)r * d;
            for (int i = 0; i < d; i++) xr[i] += hr[i];
        }
        if (g_pre) {
            /* pre-norm leaves the residual normalised for the next block's
             * first gemm? No: pre-norm normalises inside the block; x stays
             * the residual stream and is normalised at each use. */
        } else {
            for (int r = 0; r < T; r++)
                layernorm(x + (size_t)r * d, l2w, l2b, d, M->eps);
        }
    }

    /* mean pool over all T tokens (optionally without [CLS]/[SEP]), then L2 */
    int p0 = g_pool_core ? 1 : 0, pn = T - (g_pool_core ? 2 : 0);
    for (int i = 0; i < d; i++) out[i] = 0.f;
    for (int r = p0; r < p0 + pn; r++)
        for (int i = 0; i < d; i++) out[i] += x[(size_t)r * d + i];
    float inv = 1.0f / (float)pn, norm = 0.f;
    for (int i = 0; i < d; i++) { out[i] *= inv; norm += out[i] * out[i]; }
    norm = 1.0f / sqrtf(norm);
    for (int i = 0; i < d; i++) out[i] *= norm;

    free(x); free(h); free(qkv); free(att); free(gi); free(ui);
}

/* ------------------------------------------------------------ serve ------ */

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

/* payload: JSON array of strings -> array of text. Minimal parser: handles
 * the escape forms an embedding payload realistically carries (\" \\ \n etc.)
 * on UTF-8 bytes otherwise. */
typedef struct { char **v; int n; } Strs;

static void strs_free(Strs *s) { for (int i = 0; i < s->n; i++) free(s->v[i]); free(s->v); s->v = NULL; s->n = 0; }

static int json_parse_string(const char **pp, char *out, size_t cap, size_t *len)
{
    const char *p = *pp;
    if (*p != '"') return -1;
    p++;
    size_t w = 0;
    while (*p && *p != '"') {
        char c = *p;
        if (c == '\\' && p[1]) { c = p[1]; p += 2; }
        else p++;
        if (w + 2 > cap) return -1;
        out[w++] = c;
    }
    out[w] = 0;
    *len = w;
    *pp = p + 1;
    return 0;
}

static int json_parse_array(const char *json, Strs *out)
{
    const char *p = json;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '[') return -1;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p == ']') { out->v = NULL; out->n = 0; return 0; }
    out->v = (char **)calloc(8, sizeof(char *));
    out->n = 0;
    size_t cap = 8;
    while (1) {
        if (*p != '"') { strs_free(out); return -1; }
        char buf[8192];
        size_t len = 0;
        if (json_parse_string(&p, buf, sizeof buf, &len)) { strs_free(out); return -1; }
        if ((size_t)out->n == cap) { cap *= 2; out->v = (char **)realloc(out->v, (size_t)cap * sizeof(char *)); }
        out->v[out->n] = (char *)xmalloc(len + 1);
        memcpy(out->v[out->n], buf, len + 1);
        out->n++;
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p == ',') { p++; while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++; continue; }
        if (*p == ']') { p++; break; }
        strs_free(out);
        return -1;
    }
    return 0;
}

/* Serve one model: the mux protocol loop. EMBED in, EMBED out with the JSON
 * answer. */
static int nomic_serve(FILE *input, FILE *output, Nomic *M, gguf_ctx *G)
{
    const char *serve = getenv("SERVE");
    if (!serve || strcmp(serve, "1")) die("nothing to serve: set SERVE=1");
    coli_serve_stdio_init();
    char caps[256];
    snprintf(caps, sizeof(caps), "embed=1 chat=0 decide=0 decide_record=null vision=0");
    if (!coli_serve_write_ready_caps(output, rss_gb(), caps)) return 1;
    static const ColiServeWireProfile profile = {
        .max_header_bytes = 4096, .max_payload_bytes = (uint64_t)64 << 20,
        .max_tokens = 0, .require_exact_lf = 0, .require_finite_sampling = 0,
        .allow_extension_bytes = 0, .allow_prefix_hint = 0,
    };
    for (;;) {
        ColiServeCommand command;
        ColiServeReadResult result = coli_serve_read_command(input, &profile, &command);
        if (result == COLI_SERVE_READ_EOF) return 0;
        if (result == COLI_SERVE_READ_IGNORED) continue;
        if (result == COLI_SERVE_READ_NOMEM || result == COLI_SERVE_READ_BAD_FRAME) {
            coli_serve_write_error(output, command.id, "BAD_FRAME");
            coli_serve_command_dispose(&command);
            return 1;
        }
        if (result == COLI_SERVE_READ_BAD_REQUEST) {
            coli_serve_write_error(output, command.id, "BAD_REQUEST");
            coli_serve_command_dispose(&command);
            continue;
        }
        int ok = 0;
        switch (command.kind) {
        case COLI_SERVE_COMMAND_STOP:
            coli_serve_command_dispose(&command);
            return 0;
        case COLI_SERVE_COMMAND_EMBED: {
            const unsigned char *payload = coli_serve_command_take_payload(&command);
            Strs texts;
            memset(&texts, 0, sizeof texts);
            int parsed_ok = 1;
            if (!payload) { /* empty array -> zero vectors */ }
            else if (json_parse_array((const char *)payload, &texts)) {
                parsed_ok = 0;
                coli_serve_write_error(output, command.id, "EMBED_FAILED payload_not_array");
                strs_free(&texts);
                break;
            }
            size_t nvec = texts.n;
            float *vecs = (float *)xmalloc((size_t)(nvec > 0 ? nvec : 1) * (size_t)M->d * sizeof(float));
            int input_tokens = 0;
            for (int i = 0; i < texts.n; i++) {
                int ids[2048];
                int T = tokenize(M, texts.v[i], ids, (int)(sizeof ids / sizeof ids[0]));
                if (T < 0) {
                    parsed_ok = 0;
                    strs_free(&texts);
                    free(vecs);
                    coli_serve_write_error(output, command.id, "EMBED_FAILED tokenize");
                    break;
                }
                forward(M, ids, T, vecs + (size_t)i * M->d);
                input_tokens += T;
            }
            if (parsed_ok) {
                char *buf = (char *)xmalloc(64 * 1024 + (size_t)(texts.n > 0 ? texts.n : 1) * (20 * 1024));
                char *q = buf;
                char *end = buf + (64 * 1024 + (size_t)(texts.n > 0 ? texts.n : 1) * (20 * 1024));
                int n = snprintf(q, (size_t)(end - q), "{\"embedding\":[");
                q += n;
                for (int i = 0; i < texts.n; i++) {
                    if (i) { int n1 = snprintf(q, (size_t)(end - q), ","); q += n1; }
                    int nopen = snprintf(q, (size_t)(end - q), "[");
                    q += nopen;
                    for (int j = 0; j < M->d; j++) {
                        int n2 = snprintf(q, (size_t)(end - q), "%s%.17g", j ? "," : "", vecs[(size_t)i * M->d + j]);
                        if (n2 < 0 || (size_t)n2 >= (size_t)(end - q)) { free(buf); return -1; }
                        q += n2;
                    }
                    int nclose = snprintf(q, (size_t)(end - q), "]");
                    q += nclose;
                }
                int n3 = snprintf(q, (size_t)(end - q), "],\"input_tokens\":%d}", input_tokens);
                if (n3 < 0 || (size_t)n3 >= (size_t)(end - q)) { free(buf); return -1; }
                q += n3;
                ok = coli_serve_write_embed(output, command.id, buf, (size_t)(q - buf));
                free(buf);
                for (int i = 0; i < texts.n; i++) free(texts.v[i]);
                free(texts.v);
                free(vecs);
            }
            break;
        }
        case COLI_SERVE_COMMAND_SUBMIT:
            coli_serve_write_error(output, command.id, "NOT_SUPPORTED an embedding engine does not generate text");
            break;
        case COLI_SERVE_COMMAND_CANCEL:
            coli_serve_write_error(output, command.id, "NOT_FOUND");
            break;
        default:
            coli_serve_write_error(output, command.id, "BAD_REQUEST");
            break;
        }
        coli_serve_command_dispose(&command);
        if (ok && command.kind == COLI_SERVE_COMMAND_STOP) return 0;
    }
}

/* --------------------------------------------------------------- main --- */

static void print_vec(const float *v, int d)
{
    printf("{\"embedding\":[");
    for (int i = 0; i < d; i++) printf("%s%.9g", i ? "," : "", v[i]);
    printf("],\"dim\":%d}\n", d);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: nomic-check <gguf> --dump | --tok \"text\" | --embed \"text\" | --serve <gguf>\n");
        return 2;
    }
    const char *model = NULL;
    const char *text = NULL;
    int mode = -1;   /* 0 dump, 1 tok, 2 embed, 3 serve */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "--dump")) mode = 0;
        else if (!strcmp(argv[i], "--tok")) mode = 1;
        else if (!strcmp(argv[i], "--embed")) mode = 2;
        else if (!strcmp(argv[i], "--serve")) mode = 3;
    }
    const char *se = getenv("SERVE");
    int serving = -1;
    if (se && !strcmp(se, "1") && mode == -1) serving = 3;
    if (serving == 3) {
        if (!model) model = getenv("SNAP");
        if (!model) { fprintf(stderr, "usage: nomic-check --serve <gguf> | nomic-check <gguf> --dump | --tok | --embed\n"); return 2; }
    } else if (!model) { fprintf(stderr, "usage: nomic-check <gguf> --dump | --tok | --embed\n"); return 2; }
    gguf_ctx G;
    if (gguf_open(&G, model)) die("open %s: %s", model, G.err);
    if (serving == 3) {
        read_env();
        Nomic M;
        memset(&M, 0, sizeof M);
        load_model(&M, &G, model);
        return nomic_serve(stdin, stdout, &M, &G);
    }

    if (mode == 0) {
        printf("arch=%s name=%s tensors=%d\n", need_str(&G, "general.architecture"),
               need_str(&G, "general.name"), G.n_tensors);
        printf("d=%d L=%d heads=%d inter=%d eps=%g theta=%g vocab=%d\n",
               need_i64(&G, "nomic-bert.embedding_length"), need_i64(&G, "nomic-bert.block_count"),
               need_i64(&G, "nomic-bert.attention.head_count"),
               need_i64(&G, "nomic-bert.feed_forward_length"),
               need_f32(&G, "nomic-bert.attention.layer_norm_epsilon"),
               need_f32(&G, "nomic-bert.rope.freq_base"),
               gguf_get(&G, "tokenizer.ggml.tokens") ? (int)gguf_get(&G, "tokenizer.ggml.tokens")->n : -1);
        gguf_close(&G);
        return 0;
    }
    if (mode == 1 || mode == 2) {
        if (!text) die("mode %d needs --text", mode);
        read_env();
        Nomic M;
        memset(&M, 0, sizeof M);
        load_model(&M, &G, model);
        int ids[2048];
        int T = tokenize(&M, text, ids, (int)(sizeof ids / sizeof ids[0]));
        if (T < 0) die("tokenisation failed (input too long?)");
        if (mode == 1) {
            printf("[");
            for (int i = 0; i < T; i++) printf("%s%d", i ? "," : "", ids[i]);
            printf("]\n");
        } else {
            float *v = xmalloc((size_t)M.d * sizeof(float));
            forward(&M, ids, T, v);
            print_vec(v, M.d);
            free(v);
        }
        return 0;
    }
    if (mode == 3) {
        read_env();
        Nomic M;
        memset(&M, 0, sizeof M);
        load_model(&M, &G, model);
        return nomic_serve(stdin, stdout, &M, &G);
    }
    die("mode must be --dump, --tok, --embed or --serve");
}
