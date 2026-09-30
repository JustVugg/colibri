/* qwenimage.c -- Qwen-Image-2.1 text-to-image in pure C.
 *
 * The checkpoint is a diffusers directory with three networks that are never
 * needed at the same time:
 *
 *   text_encoder  Qwen3-VL's language model (36 layers, 8B). Only a prefill: the
 *                 last layer's hidden states, before the final norm, are the
 *                 prompt's embedding. lm_head and the vision tower are not read.
 *   transformer   the DiT: 32 single-stream blocks over [prompt ; image tokens].
 *                 Prompt tokens are modulated as if t = 0 and attend causally, so
 *                 their keys and values do not depend on the step: they are
 *                 computed ONCE per prompt (the "prefix pass") and every step only
 *                 runs the image tokens against that cache.
 *   vae           decodes the final latent to RGBA (qwenimage_vae.h).
 *
 * Weights are read straight from the checkpoint's safetensors (bf16) and kept in
 * int8 with one scale per row by default (COLI_IMG_BITS=16 keeps bf16, 32 f32).
 * Activations stay f32. Every matrix product runs through qi_gemm.h.
 *
 * Usage:
 *   qwenimage --model DIR --prompt "..." [--width W] [--height H] [--steps N]
 *             (sides: multiples of 32, as the pipeline snaps them; default 768x512)
 *             [--seed S] --out image.png
 *   qwenimage --model DIR --serve        line protocol, see serve_loop()
 *   qwenimage --model DIR --ref REFDIR   compare every stage with a reference
 *                                         dump (tools/make_qwenimage_tiny.py) */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "coli_env.h"
#include "st.h"
#include "json.h"
#include "tok.h"
#include "qwen38_nfc.h"
#include "qi_gemm.h"
#include "serve_poll.h"
#if defined(__has_include)
#if __has_include("qwenimage_vae.h")
#include "qwenimage_vae.h"
#define QI_HAVE_VAE 1
#endif
#endif

static double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void *xmalloc(size_t n){
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "[qwenimage] out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}
static float *fmalloc(size_t n){ return (float *)xmalloc(n * sizeof(float)); }

static char *read_file(const char *path){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f);
    return b;
}
static jval *read_json(const char *dir, const char *file, char **arena){
    char path[2048]; snprintf(path, sizeof path, "%s/%s", dir, file);
    char *text = read_file(path);
    if (!text) { fprintf(stderr, "[qwenimage] cannot read %s\n", path); exit(1); }
    jval *v = json_parse(text, arena);
    free(text);
    if (!v) { fprintf(stderr, "[qwenimage] bad JSON in %s\n", path); exit(1); }
    return v;
}
static double jnum(jval *o, const char *k, double def){
    jval *v = o ? json_get(o, k) : NULL;
    return v && v->t == J_NUM ? v->num : def;
}
static int jbool(jval *o, const char *k, int def){
    jval *v = o ? json_get(o, k) : NULL;
    return v && v->t == J_BOOL ? v->boolean : def;
}

/* ---- weights ------------------------------------------------------------ */

static int g_bits = 8;   /* COLI_IMG_BITS: storage of the big matrices */
static int g_act8 = 1;   /* COLI_IMG_ACT8=0 turns off the int8 activations of the DiT blocks (VNNI) */

/* act8: this matrix may take int8 activations (COLI_IMG_ACT8). Only the DiT's
 * block matrices do. Measured on the real model: the text encoder's hidden
 * states carry a few huge channels, and one int8 scale per token wipes out the
 * rest (last layer 124% off, against 7% with int8 weights alone), while the
 * DiT blocks keep the picture (30.1 dB against 35.6 for int8 weights alone). */
typedef struct { QiMat m; void *own; float *own_sc; int act8; } Lin;

static st_tensor *need_tensor(shards *S, const char *name, int64_t n0, int64_t n1){
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "[qwenimage] missing tensor %s\n", name); exit(1); }
    int64_t numel = 1; for (int i = 0; i < t->rank; i++) numel *= t->shape[i];
    if (numel != n0 * n1) {
        fprintf(stderr, "[qwenimage] %s has %lld values, expected %lld x %lld\n",
                name, (long long)numel, (long long)n0, (long long)n1);
        exit(1);
    }
    return t;
}

/* bits: 8 = int8 rows, 16 = bf16, 32 = f32. Small matrices ask for 32. */
static void lin_load(shards *S, const char *name, int N, int K, int bits, Lin *out){
    need_tensor(S, name, N, K);
    float *tmp = fmalloc((size_t)N * K);
    st_read_f32_cap(S, name, tmp, (int64_t)N * K, 1);
    memset(out, 0, sizeof *out);
    out->m.N = N; out->m.K = K;
    if (bits == 32) {
        out->m.fmt = QI_F32; out->m.w = out->own = tmp;
    } else if (bits == 16) {
        uint16_t *b = xmalloc((size_t)N * K * 2);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)N * K; i++) {
            union { float f; uint32_t u; } v; v.f = tmp[i];
            b[i] = (uint16_t)((v.u + 0x7fff + ((v.u >> 16) & 1)) >> 16);
        }
        free(tmp);
        out->m.fmt = QI_BF16; out->m.w = out->own = b;
    } else {
        int8_t *q = xmalloc((size_t)N * K);
        float *sc = fmalloc(N);
        qi_quantize_i8(tmp, N, K, q, sc);
        free(tmp);
        out->m.fmt = QI_I8; out->m.w = out->own = q; out->m.sc = out->own_sc = sc;
    }
}
static void lin_free(Lin *l){ free(l->own); free(l->own_sc); memset(l, 0, sizeof *l); }
static size_t lin_bytes(const Lin *l){
    size_t e = l->m.fmt == QI_F32 ? 4 : l->m.fmt == QI_BF16 ? 2 : 1;
    return (size_t)l->m.N * l->m.K * e + (l->own_sc ? (size_t)l->m.N * 4 : 0);
}
static float *vec_load(shards *S, const char *name, int n){
    need_tensor(S, name, n, 1);
    float *v = fmalloc(n);
    st_read_f32_cap(S, name, v, n, 1);
    return v;
}
/* QWENIMAGE_PROF=1: where a step's time goes (linears, attention, the rest) */
static int g_prof; static double g_t_lin, g_t_att;
static inline void linear(float *y, const float *x, int M, const Lin *l){
    double t0 = g_prof ? now_s() : 0;
    if (g_act8 && l->act8 && l->m.fmt == QI_I8) qi_gemm_act8(y, l->m.N, x, l->m.K, M, &l->m, NULL);
    else qi_gemm(y, x, M, &l->m, NULL);
    if (g_prof) g_t_lin += now_s() - t0;
}

/* ---- small kernels -------------------------------------------------------- */

static void rmsnorm_rows(float *y, const float *x, const float *w, int T, int D, float eps, int plus_one){
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *r = x + (int64_t)t * D; float *o = y + (int64_t)t * D;
        double ss = 0; for (int i = 0; i < D; i++) ss += (double)r[i] * r[i];
        float s = 1.f / sqrtf((float)(ss / D) + eps);
        for (int i = 0; i < D; i++) o[i] = r[i] * s * (plus_one ? w[i] + 1.f : w[i]);
    }
}
/* LayerNorm without affine, then * (1 + scale). */
static void layernorm_mod(float *y, const float *x, const float *scale, int T, int D, float eps){
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *r = x + (int64_t)t * D; float *o = y + (int64_t)t * D;
        double m = 0; for (int i = 0; i < D; i++) m += r[i];
        m /= D;
        double v = 0; for (int i = 0; i < D; i++) { double d = r[i] - m; v += d * d; }
        v /= D;
        float inv = 1.f / sqrtf((float)v + eps), mf = (float)m;
        for (int i = 0; i < D; i++) o[i] = (r[i] - mf) * inv * (1.f + scale[i]);
    }
}
static inline float silu(float x){ return x / (1.f + expf(-x)); }
static inline float gelu_tanh(float x){
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

/* Attention for every head: out[Tq][heads*hd] = softmax(Q K^T / sqrt(hd)) V.
 * Q rows have stride ldq, K/V rows stride ldk; query head h reads kv head
 * h / (heads/kvheads). causal >= 0: query i sees keys j <= i + causal. */
static void attention(float *out, int ldo, const float *Q, int ldq, int Tq,
                      const float *K, const float *V, int ldk, int Tk,
                      int heads, int kvheads, int hd, int causal){
    double tprof = g_prof ? now_s() : 0;
    float *S = fmalloc((size_t)Tq * Tk);
    float *vt = fmalloc((size_t)hd * Tk);
    const float scale = 1.f / sqrtf((float)hd);
    int group = heads / kvheads, vt_head = -1;
    for (int h = 0; h < heads; h++) {
        int kh = h / group;
        QiMat Km = { QI_F32, Tk, hd, K + (int64_t)kh * hd, NULL, ldk };
        qi_gemm_ld(S, Tk, Q + (int64_t)h * hd, ldq, Tq, &Km, NULL);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Tq; i++) {
            float *r = S + (int64_t)i * Tk;
            int lim = causal >= 0 ? i + causal + 1 : Tk;
            if (lim > Tk) lim = Tk;
            float mx = -INFINITY;
            for (int j = 0; j < lim; j++) { r[j] *= scale; if (r[j] > mx) mx = r[j]; }
            double sum = 0;
            for (int j = 0; j < lim; j++) { r[j] = expf(r[j] - mx); sum += r[j]; }
            float inv = (float)(1.0 / sum);
            for (int j = 0; j < lim; j++) r[j] *= inv;
            for (int j = lim; j < Tk; j++) r[j] = 0.f;
        }
        if (kh != vt_head) {
            #pragma omp parallel for schedule(static)
            for (int d = 0; d < hd; d++)
                for (int j = 0; j < Tk; j++) vt[(int64_t)d * Tk + j] = V[(int64_t)j * ldk + (int64_t)kh * hd + d];
            vt_head = kh;
        }
        QiMat Vm = { QI_F32, hd, Tk, vt, NULL, 0 };
        qi_gemm_ld(out + (int64_t)h * hd, ldo, S, Tk, Tq, &Vm, NULL);
    }
    free(S); free(vt);
    if (g_prof) g_t_att += now_s() - tprof;
}

/* ---- text encoder (Qwen3-VL language model, prefill only) ------------------ */

typedef struct { Lin q, k, v, o, gate, up, down; float *ln1, *ln2, *qn, *kn; } TeLayer;
typedef struct {
    int hidden, layers, heads, kv, hd, inter, vocab;
    float eps, theta;
    shards S; int open;
    st_tensor *embed;
    TeLayer *L; int loaded;
    Tok tok; int tok_loaded;
    int im_start, im_end;
} Te;

static void te_config(Te *te, const char *model){
    char dir[2048]; snprintf(dir, sizeof dir, "%s/text_encoder", model);
    char *arena = NULL;
    jval *root = read_json(dir, "config.json", &arena);
    jval *c = json_get(root, "text_config"); if (!c) c = root;
    te->hidden = (int)jnum(c, "hidden_size", 0);
    te->layers = (int)jnum(c, "num_hidden_layers", 0);
    te->heads = (int)jnum(c, "num_attention_heads", 0);
    te->kv = (int)jnum(c, "num_key_value_heads", te->heads);
    te->hd = (int)jnum(c, "head_dim", te->heads ? te->hidden / te->heads : 0);
    te->inter = (int)jnum(c, "intermediate_size", 0);
    te->vocab = (int)jnum(c, "vocab_size", 0);
    te->eps = (float)jnum(c, "rms_norm_eps", 1e-6);
    te->theta = (float)jnum(c, "rope_theta", 0);
    if (te->theta == 0) {
        jval *rp = json_get(c, "rope_parameters");
        te->theta = (float)jnum(rp, "rope_theta", 5000000.0);
    }
    json_free(root); free(arena);
    /* Model files come from mirrors nobody vouches for: every dimension that
     * sizes a buffer is bounded, and products are formed in 64 bits, so a config
     * cannot wrap an int into a small allocation that a loop then overruns. */
    if (te->hidden <= 0 || te->hidden > 65536 || te->layers <= 0 || te->layers > 1024 ||
        te->heads <= 0 || te->heads > 1024 || te->kv <= 0 || te->hd <= 0 || te->hd > 1024 ||
        te->heads % te->kv || te->inter <= 0 || te->inter > (1 << 18) || te->vocab <= 0 ||
        te->vocab > (1 << 22) || te->hd % 2 || (int64_t)te->heads * te->hd > 65536) {
        fprintf(stderr, "[qwenimage] text_encoder/config.json: unsupported shape\n"); exit(1);
    }
}

static void te_open(Te *te, const char *model){
    if (te->open) return;
    char dir[2048]; snprintf(dir, sizeof dir, "%s/text_encoder", model);
    memset(&te->S, 0, sizeof te->S);
    st_init(&te->S, dir);
    te->open = 1;
    te->embed = need_tensor(&te->S, "model.language_model.embed_tokens.weight", te->vocab, te->hidden);
    if (te->embed->dtype != 0 && te->embed->dtype != 2) {
        fprintf(stderr, "[qwenimage] embed_tokens must be bf16 or f32\n"); exit(1);
    }
}

static void te_load(Te *te, const char *model){
    if (te->loaded) return;
    te_open(te, model);
    double t0 = now_s();
    te->L = calloc(te->layers, sizeof(TeLayer));
    int H = te->hidden, qd = te->heads * te->hd, kd = te->kv * te->hd;
    char n[256];
    size_t bytes = 0;
    for (int l = 0; l < te->layers; l++) {
        TeLayer *L = &te->L[l];
#define TN(s) (snprintf(n, sizeof n, "model.language_model.layers.%d.%s", l, s), n)
        lin_load(&te->S, TN("self_attn.q_proj.weight"), qd, H, g_bits, &L->q);
        lin_load(&te->S, TN("self_attn.k_proj.weight"), kd, H, g_bits, &L->k);
        lin_load(&te->S, TN("self_attn.v_proj.weight"), kd, H, g_bits, &L->v);
        lin_load(&te->S, TN("self_attn.o_proj.weight"), H, qd, g_bits, &L->o);
        lin_load(&te->S, TN("mlp.gate_proj.weight"), te->inter, H, g_bits, &L->gate);
        lin_load(&te->S, TN("mlp.up_proj.weight"), te->inter, H, g_bits, &L->up);
        lin_load(&te->S, TN("mlp.down_proj.weight"), H, te->inter, g_bits, &L->down);
        L->ln1 = vec_load(&te->S, TN("input_layernorm.weight"), H);
        L->ln2 = vec_load(&te->S, TN("post_attention_layernorm.weight"), H);
        L->qn = vec_load(&te->S, TN("self_attn.q_norm.weight"), te->hd);
        L->kn = vec_load(&te->S, TN("self_attn.k_norm.weight"), te->hd);
#undef TN
        bytes += lin_bytes(&L->q) + lin_bytes(&L->k) + lin_bytes(&L->v) + lin_bytes(&L->o) +
                 lin_bytes(&L->gate) + lin_bytes(&L->up) + lin_bytes(&L->down);
    }
    te->loaded = 1;
    fprintf(stderr, "[qwenimage] text encoder: %d layers, %.2f GB resident (%s), %.1f s\n",
            te->layers, bytes / 1e9, g_bits == 8 ? "int8" : g_bits == 16 ? "bf16" : "f32", now_s() - t0);
}

static void te_unload(Te *te){
    if (!te->loaded) return;
    for (int l = 0; l < te->layers; l++) {
        TeLayer *L = &te->L[l];
        lin_free(&L->q); lin_free(&L->k); lin_free(&L->v); lin_free(&L->o);
        lin_free(&L->gate); lin_free(&L->up); lin_free(&L->down);
        free(L->ln1); free(L->ln2); free(L->qn); free(L->kn);
    }
    free(te->L); te->L = NULL; te->loaded = 0;
}

static void te_tokenizer(Te *te, const char *model){
    if (te->tok_loaded) return;
    char path[2048]; snprintf(path, sizeof path, "%s/processor/tokenizer.json", model);
    FILE *f = fopen(path, "rb");
    if (!f) snprintf(path, sizeof path, "%s/tokenizer/tokenizer.json", model);
    else fclose(f);
    memset(&te->tok, 0, sizeof te->tok);
    tok_load(&te->tok, path);
    te->tok_loaded = 1;
    te->im_start = tok_id_of(&te->tok, "<|im_start|>");
    te->im_end = tok_id_of(&te->tok, "<|im_end|>");
    if (te->im_start < 0 || te->im_end < 0) {
        fprintf(stderr, "[qwenimage] tokenizer has no <|im_start|>/<|im_end|>\n"); exit(1);
    }
}

/* The pipeline's template, tokenized as one string (not through
 * apply_chat_template: the checkpoint was trained on this one). */
static const char *QI_SYS = "Comprehend and analyze the provided prompt.";

static int *te_encode_prompt(Te *te, const char *prompt, int *n_out, int *drop_out){
    char *norm = NULL; size_t norm_len = 0;
    const char *p = prompt && *prompt ? prompt : " ";   /* the pipeline's rule: no empty prompt */
    if (q38_nfc_normalize(p, strlen(p), &norm, &norm_len) != 0) { norm = strdup(p); norm_len = strlen(p); }
    size_t cap = norm_len + 512;
    char *text = xmalloc(cap);
    snprintf(text, cap, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%.*s<|im_end|>\n<|im_start|>assistant\n",
             QI_SYS, (int)norm_len, norm);
    int max = (int)strlen(text) + 16;
    int *ids = xmalloc(sizeof(int) * max);
    int n = tok_encode(&te->tok, text, (int)strlen(text), ids, max);
    /* drop_idx: the tokens of the system turn, "<|im_start|>system\n...<|im_end|>\n" */
    char sys[512]; snprintf(sys, sizeof sys, "<|im_start|>system\n%s<|im_end|>\n", QI_SYS);
    int tmp[256];
    int drop = tok_encode(&te->tok, sys, (int)strlen(sys), tmp, 256);
    free(text); free(norm);
    *n_out = n; *drop_out = drop;
    return ids;
}

/* rotate-half RoPE over the full head (text only: Qwen3-VL's M-RoPE with three
 * equal position components is the plain 1D rotary). */
static void te_rope(float *x, int T, int nh, int hd, int ld, float theta){
    int half = hd / 2;
    float *inv = fmalloc(half);
    for (int i = 0; i < half; i++) inv[i] = 1.f / powf(theta, (float)(2 * i) / (float)hd);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int h = 0; h < nh; h++) {
            float *r = x + (int64_t)t * ld + (int64_t)h * hd;
            for (int i = 0; i < half; i++) {
                float a = (float)t * inv[i], c = cosf(a), s = sinf(a);
                float x0 = r[i], x1 = r[i + half];
                r[i] = x0 * c - x1 * s;
                r[i + half] = x1 * c + x0 * s;
            }
        }
    free(inv);
}

/* ids[T] -> hidden[T][hidden]: the last layer's output before the final norm. */
static float *te_forward(Te *te, const int *ids, int T){
    int H = te->hidden, qd = te->heads * te->hd, kd = te->kv * te->hd, I = te->inter;
    float *x = fmalloc((size_t)T * H), *h = fmalloc((size_t)T * H);
    float *q = fmalloc((size_t)T * qd), *k = fmalloc((size_t)T * kd), *v = fmalloc((size_t)T * kd);
    float *a = fmalloc((size_t)T * qd), *g = fmalloc((size_t)T * I), *u = fmalloc((size_t)T * I);
    int esz = te->embed->dtype == 0 ? 2 : 4;
    void *row = xmalloc((size_t)H * esz);
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= te->vocab) { fprintf(stderr, "[qwenimage] token id %d out of range\n", ids[t]); exit(1); }
        st_pread_full(te->embed->fd, row, (int64_t)H * esz, te->embed->off + (int64_t)ids[t] * H * esz, "embed row");
        float *dst = x + (int64_t)t * H;
        if (esz == 2) for (int i = 0; i < H; i++) dst[i] = qi_bf16(((uint16_t *)row)[i]);
        else memcpy(dst, row, (size_t)H * 4);
    }
    free(row);
    for (int l = 0; l < te->layers; l++) {
        TeLayer *L = &te->L[l];
        rmsnorm_rows(h, x, L->ln1, T, H, te->eps, 0);
        linear(q, h, T, &L->q); linear(k, h, T, &L->k); linear(v, h, T, &L->v);
        rmsnorm_rows(q, q, L->qn, T * te->heads, te->hd, te->eps, 0);
        rmsnorm_rows(k, k, L->kn, T * te->kv, te->hd, te->eps, 0);
        te_rope(q, T, te->heads, te->hd, qd, te->theta);
        te_rope(k, T, te->kv, te->hd, kd, te->theta);
        attention(a, qd, q, qd, T, k, v, kd, T, te->heads, te->kv, te->hd, 0);
        linear(h, a, T, &L->o);
        for (int64_t i = 0; i < (int64_t)T * H; i++) x[i] += h[i];
        rmsnorm_rows(h, x, L->ln2, T, H, te->eps, 0);
        linear(g, h, T, &L->gate); linear(u, h, T, &L->up);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * I; i++) g[i] = silu(g[i]) * u[i];
        linear(h, g, T, &L->down);
        for (int64_t i = 0; i < (int64_t)T * H; i++) x[i] += h[i];
    }
    free(h); free(q); free(k); free(v); free(a); free(g); free(u);
    return x;
}

/* ---- DiT ------------------------------------------------------------------- */

typedef struct { Lin q, k, v, o, gate, proj, out; float *nq, *nk; } DitBlock;
typedef struct {
    int layers, heads, hd, dim, mlp, in_ch, ctx, axes[3];
    float eps;
    int causal_condition;
    Lin img_in, t1, t2, mod, norm_out, proj_out, txt1, txt2;
    float *txt_norm;
    DitBlock *B;
    int loaded;
} Dit;

static void dit_config(Dit *d, const char *model){
    char dir[2048]; snprintf(dir, sizeof dir, "%s/transformer", model);
    char *arena = NULL;
    jval *c = read_json(dir, "config.json", &arena);
    d->layers = (int)jnum(c, "num_layers", 0);
    d->heads = (int)jnum(c, "num_attention_heads", 0);
    d->hd = (int)jnum(c, "attention_head_dim", 0);
    int mlp_ratio = (int)jnum(c, "mlp_ratio", 3);
    int64_t dim64 = (int64_t)d->heads * d->hd;
    d->dim = dim64 > 0 && dim64 <= 65536 ? (int)dim64 : 0;
    d->mlp = mlp_ratio >= 1 && mlp_ratio <= 16 ? d->dim * mlp_ratio : 0;
    d->in_ch = (int)jnum(c, "in_channels", 64);
    d->ctx = (int)jnum(c, "context_in_dim", 0);
    d->eps = (float)jnum(c, "eps", 1e-6);
    d->causal_condition = jbool(c, "causal_condition", 1);
    int patch = (int)jnum(c, "patch_size", 1), outc = (int)jnum(c, "out_channels", d->in_ch);
    jval *ax = json_get(c, "axes_dims_rope");
    int ok = ax && ax->t == J_ARR && ax->len == 3;
    for (int i = 0; ok && i < 3; i++) {
        d->axes[i] = ax->kids[i]->t == J_NUM ? (int)ax->kids[i]->num : -1;
        ok = d->axes[i] > 0 && d->axes[i] % 2 == 0;   /* each axis fills whole complex pairs */
    }
    json_free(c); free(arena);
    if (!ok || d->axes[0] + d->axes[1] + d->axes[2] != d->hd || d->layers <= 0 || d->layers > 1024 ||
        d->heads <= 0 || d->heads > 1024 || d->hd <= 0 || d->hd > 1024 || d->hd % 2 || d->dim <= 0 ||
        d->mlp <= 0 || d->in_ch <= 0 || d->in_ch > 1024 || patch != 1 || outc != d->in_ch ||
        d->ctx <= 0 || d->ctx > 65536 || !d->causal_condition) {
        fprintf(stderr, "[qwenimage] transformer/config.json: unsupported shape "
                        "(needs patch_size 1, causal_condition, axes summing to the head dim)\n");
        exit(1);
    }
}

static void dit_load(Dit *d, const char *model){
    if (d->loaded) return;
    char dir[2048]; snprintf(dir, sizeof dir, "%s/transformer", model);
    shards S; memset(&S, 0, sizeof S);
    st_init(&S, dir);
    double t0 = now_s();
    int D = d->dim;
    lin_load(&S, "img_in.weight", D, d->in_ch, 32, &d->img_in);
    lin_load(&S, "time_text_embed.timestep_embedder.linear_1.weight", D, 256, 32, &d->t1);
    lin_load(&S, "time_text_embed.timestep_embedder.linear_2.weight", D, D, 32, &d->t2);
    lin_load(&S, "modulation.1.weight", 4 * D, D, g_bits, &d->mod);
    lin_load(&S, "norm_out.linear.weight", D, D, 32, &d->norm_out);
    lin_load(&S, "proj_out.weight", d->in_ch, D, 32, &d->proj_out);
    lin_load(&S, "txt_in.in_layer.weight", D, d->ctx, g_bits, &d->txt1);
    lin_load(&S, "txt_in.out_layer.weight", D, D, g_bits, &d->txt2);
    d->txt_norm = vec_load(&S, "txt_in.text_norm.weight", d->ctx);
    d->B = calloc(d->layers, sizeof(DitBlock));
    char n[256];
    size_t bytes = 0;
    for (int l = 0; l < d->layers; l++) {
        DitBlock *B = &d->B[l];
#define BN(s) (snprintf(n, sizeof n, "transformer_blocks.%d.%s", l, s), n)
        lin_load(&S, BN("attn.to_q.weight"), D, D, g_bits, &B->q);
        lin_load(&S, BN("attn.to_k.weight"), D, D, g_bits, &B->k);
        lin_load(&S, BN("attn.to_v.weight"), D, D, g_bits, &B->v);
        lin_load(&S, BN("attn.to_out.0.weight"), D, D, g_bits, &B->o);
        lin_load(&S, BN("img_mlp.gate_layer.weight"), d->mlp, D, g_bits, &B->gate);
        lin_load(&S, BN("img_mlp.proj.weight"), d->mlp, D, g_bits, &B->proj);
        lin_load(&S, BN("img_mlp.out.weight"), D, d->mlp, g_bits, &B->out);
        B->q.act8 = B->k.act8 = B->v.act8 = B->o.act8 = B->gate.act8 = B->proj.act8 = B->out.act8 = 1;
        B->nq = vec_load(&S, BN("attn.norm_q.weight"), d->hd);
        B->nk = vec_load(&S, BN("attn.norm_k.weight"), d->hd);
#undef BN
        bytes += lin_bytes(&B->q) + lin_bytes(&B->k) + lin_bytes(&B->v) + lin_bytes(&B->o) +
                 lin_bytes(&B->gate) + lin_bytes(&B->proj) + lin_bytes(&B->out);
    }
    st_destroy(&S);
    d->loaded = 1;
    fprintf(stderr, "[qwenimage] transformer: %d blocks, %.2f GB resident, %.1f s\n",
            d->layers, (bytes + lin_bytes(&d->mod) + lin_bytes(&d->txt1) + lin_bytes(&d->txt2)) / 1e9,
            now_s() - t0);
}

/* The sinusoidal embedding of the timestep, then the two linears. `t` is the
 * scheduler's timestep (sigma * 1000); the pipeline divides it by 1000 and the
 * embedding multiplies it back, so do the same float round trip. */
static void dit_temb(Dit *d, float t, float *temb){
    float tt = (t / 1000.f) * 1000.f;
    float e[256];
    const int half = 128;
    for (int i = 0; i < half; i++) {
        float f = expf((float)(-9.210340371976184) * (float)i / (float)half);  /* -ln(10000) */
        float a = tt * f;
        e[i] = cosf(a); e[half + i] = sinf(a);
    }
    float *h = fmalloc(d->dim);
    qi_gemm(h, e, 1, &d->t1.m, NULL);
    for (int i = 0; i < d->dim; i++) h[i] = silu(h[i]);
    qi_gemm(temb, h, 1, &d->t2.m, NULL);
    free(h);
}
/* mod[4*dim] = [scale1 | gate1 | scale2 | gate2], outs[dim] = the final norm's scale */
static void dit_modulation(Dit *d, float t, float *mod, float *outs){
    float *temb = fmalloc(d->dim), *s = fmalloc(d->dim);
    dit_temb(d, t, temb);
    for (int i = 0; i < d->dim; i++) s[i] = silu(temb[i]);
    qi_gemm(mod, s, 1, &d->mod.m, NULL);
    qi_gemm(outs, s, 1, &d->norm_out.m, NULL);
    free(temb); free(s);
}

/* cos/sin for the three-axis RoPE: tokens x hd/2 complex frequencies. */
static void dit_rope_table(const Dit *d, const int *pos3, int T, float *cs, float *sn){
    int half = d->hd / 2;
    float *freq = fmalloc(half); int *axis = xmalloc(sizeof(int) * half);
    int j = 0;
    for (int a = 0; a < 3; a++)
        for (int i = 0; i < d->axes[a] / 2; i++, j++) {
            freq[j] = 1.f / powf(10000.f, (float)(2 * i) / (float)d->axes[a]);
            axis[j] = a;
        }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int i = 0; i < half; i++) {
            float ang = (float)pos3[t * 3 + axis[i]] * freq[i];
            cs[(int64_t)t * half + i] = cosf(ang);
            sn[(int64_t)t * half + i] = sinf(ang);
        }
    free(freq); free(axis);
}
/* interleaved pairs (2i, 2i+1) rotated as complex numbers */
static void dit_rope_apply(float *x, int T, int heads, int hd, const float *cs, const float *sn){
    int half = hd / 2, D = heads * hd;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int h = 0; h < heads; h++) {
            float *r = x + (int64_t)t * D + (int64_t)h * hd;
            const float *c = cs + (int64_t)t * half, *s = sn + (int64_t)t * half;
            for (int i = 0; i < half; i++) {
                float a = r[2 * i], b = r[2 * i + 1];
                r[2 * i] = a * c[i] - b * s[i];
                r[2 * i + 1] = a * s[i] + b * c[i];
            }
        }
}

/* The prompt side of the DiT, computed once per prompt: per block, the keys and
 * values of the text tokens (after norm and RoPE). */
typedef struct { int L; float **K, **V; } Prefix;

static void prefix_free(Prefix *p, int layers){
    if (p->K) for (int l = 0; l < layers; l++) { free(p->K[l]); free(p->V[l]); }
    free(p->K); free(p->V); memset(p, 0, sizeof *p);
}

/* One block over T tokens. q/k/v are written into kbuf/vbuf at row koff (the
 * caller lays the prefix before the image tokens); attention runs over the
 * first ktotal rows. causal: -1 full, >=0 as in attention(). */
typedef struct { float *h, *q, *a, *g, *u; } DitScratch;

static void dit_block(Dit *d, DitBlock *B, float *x, int T, const float *mod,
                      float *kbuf, float *vbuf, int koff, int ktotal, int causal,
                      const float *cs, const float *sn, DitScratch *w, int need_out){
    int D = d->dim;
    const float *scale1 = mod, *gate1 = mod + D, *scale2 = mod + 2 * D, *gate2 = mod + 3 * D;
    layernorm_mod(w->h, x, scale1, T, D, d->eps);
    float *k = kbuf + (int64_t)koff * D, *v = vbuf + (int64_t)koff * D;
    linear(w->q, w->h, T, &B->q); linear(k, w->h, T, &B->k); linear(v, w->h, T, &B->v);
    rmsnorm_rows(w->q, w->q, B->nq, T * d->heads, d->hd, d->eps, 0);
    rmsnorm_rows(k, k, B->nk, T * d->heads, d->hd, d->eps, 0);
    dit_rope_apply(w->q, T, d->heads, d->hd, cs, sn);
    dit_rope_apply(k, T, d->heads, d->hd, cs, sn);
    if (!need_out) return;
    attention(w->a, D, w->q, D, T, kbuf, vbuf, D, ktotal, d->heads, d->heads, d->hd, causal);
    linear(w->h, w->a, T, &B->o);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int i = 0; i < D; i++) x[(int64_t)t * D + i] += tanhf(gate1[i]) * w->h[(int64_t)t * D + i];
    layernorm_mod(w->h, x, scale2, T, D, d->eps);
    linear(w->g, w->h, T, &B->gate); linear(w->u, w->h, T, &B->proj);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * d->mlp; i++) w->g[i] = silu(w->g[i]) * w->u[i];
    linear(w->h, w->g, T, &B->out);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int i = 0; i < D; i++) x[(int64_t)t * D + i] += tanhf(gate2[i]) * w->h[(int64_t)t * D + i];
}

static void scratch_alloc(DitScratch *w, const Dit *d, int T){
    w->h = fmalloc((size_t)T * d->dim); w->q = fmalloc((size_t)T * d->dim); w->a = fmalloc((size_t)T * d->dim);
    w->g = fmalloc((size_t)T * d->mlp); w->u = fmalloc((size_t)T * d->mlp);
}
static void scratch_free(DitScratch *w){ free(w->h); free(w->q); free(w->a); free(w->g); free(w->u); }

/* emb[L][ctx] (the text encoder's output after drop_idx) -> per-block K/V of the prefix. */
static void dit_prefix(Dit *d, const float *emb, int L, Prefix *p, float *txt_out /* optional [L][dim] */){
    int D = d->dim;
    float *n = fmalloc((size_t)L * d->ctx), *x = fmalloc((size_t)L * D);
    rmsnorm_rows(n, emb, d->txt_norm, L, d->ctx, d->eps, 1);
    linear(x, n, L, &d->txt1);
    for (int64_t i = 0; i < (int64_t)L * D; i++) x[i] = gelu_tanh(x[i]);
    free(n);
    float *y = fmalloc((size_t)L * D);
    linear(y, x, L, &d->txt2);
    free(x); x = y;
    if (txt_out) memcpy(txt_out, x, (size_t)L * D * 4);
    float *mod = fmalloc(4 * (size_t)D), *outs = fmalloc(D);
    dit_modulation(d, 0.f, mod, outs);
    int *pos = xmalloc(sizeof(int) * 3 * L);
    for (int t = 0; t < L; t++) pos[3 * t] = pos[3 * t + 1] = pos[3 * t + 2] = t;
    float *cs = fmalloc((size_t)L * d->hd / 2), *sn = fmalloc((size_t)L * d->hd / 2);
    dit_rope_table(d, pos, L, cs, sn);
    p->L = L;
    p->K = calloc(d->layers, sizeof(float *)); p->V = calloc(d->layers, sizeof(float *));
    DitScratch w; scratch_alloc(&w, d, L);
    for (int l = 0; l < d->layers; l++) {
        p->K[l] = fmalloc((size_t)L * D); p->V[l] = fmalloc((size_t)L * D);
        /* the last block's output is never read: only its keys and values are */
        dit_block(d, &d->B[l], x, L, mod, p->K[l], p->V[l], 0, L, 0, cs, sn, &w, l + 1 < d->layers);
    }
    scratch_free(&w);
    free(x); free(mod); free(outs); free(pos); free(cs); free(sn);
}

/* noise_pred[N][in_ch] for the image tokens at timestep t. */
typedef struct { float *kbuf, *vbuf, *cs, *sn, *x; DitScratch w; int N; } DitStep;

static void dit_step_init(DitStep *s, Dit *d, const Prefix *p, int gh, int gw){
    int N = gh * gw, T = p->L + N;
    s->N = N;
    s->kbuf = fmalloc((size_t)T * d->dim); s->vbuf = fmalloc((size_t)T * d->dim);
    s->cs = fmalloc((size_t)N * d->hd / 2); s->sn = fmalloc((size_t)N * d->hd / 2);
    s->x = fmalloc((size_t)N * d->dim);
    scratch_alloc(&s->w, d, N);
    /* image tokens: frame = the position after the text, height and width centred on zero */
    int *pos = xmalloc(sizeof(int) * 3 * N);
    for (int y = 0; y < gh; y++)
        for (int xx = 0; xx < gw; xx++) {
            int t = y * gw + xx;
            pos[3 * t] = p->L;
            pos[3 * t + 1] = y - (gh - gh / 2);
            pos[3 * t + 2] = xx - (gw - gw / 2);
        }
    dit_rope_table(d, pos, N, s->cs, s->sn);
    free(pos);
}
static void dit_step_free(DitStep *s){
    free(s->kbuf); free(s->vbuf); free(s->cs); free(s->sn); free(s->x); scratch_free(&s->w);
}

static void dit_forward(Dit *d, const Prefix *p, DitStep *s, const float *lat, float t, float *out){
    int D = d->dim, N = s->N, L = p->L;
    float *mod = fmalloc(4 * (size_t)D), *outs = fmalloc(D);
    dit_modulation(d, t, mod, outs);
    linear(s->x, lat, N, &d->img_in);
    for (int l = 0; l < d->layers; l++) {
        memcpy(s->kbuf, p->K[l], (size_t)L * D * 4);
        memcpy(s->vbuf, p->V[l], (size_t)L * D * 4);
        dit_block(d, &d->B[l], s->x, N, mod, s->kbuf, s->vbuf, L, L + N, -1, s->cs, s->sn, &s->w, 1);
    }
    layernorm_mod(s->w.h, s->x, outs, N, D, d->eps);
    linear(out, s->w.h, N, &d->proj_out);
    free(mod); free(outs);
}

/* ---- scheduler ------------------------------------------------------------- */

typedef struct { double base_seq, max_seq, base_shift, max_shift, shift_terminal; int dynamic, exponential; } Sched;

static void sched_config(Sched *s, const char *model){
    char dir[2048]; snprintf(dir, sizeof dir, "%s/scheduler", model);
    char *arena = NULL;
    jval *c = read_json(dir, "scheduler_config.json", &arena);
    s->base_seq = jnum(c, "base_image_seq_len", 256);
    s->max_seq = jnum(c, "max_image_seq_len", 4096);
    s->base_shift = jnum(c, "base_shift", 0.5);
    s->max_shift = jnum(c, "max_shift", 1.15);
    jval *st = json_get(c, "shift_terminal");
    s->shift_terminal = st && st->t == J_NUM ? st->num : 0;
    s->dynamic = jbool(c, "use_dynamic_shifting", 1);
    jval *ty = json_get(c, "time_shift_type");
    s->exponential = !(ty && ty->t == J_STR && !strcmp(ty->str, "linear"));
    int shift_one = jnum(c, "shift", 1.0) == 1.0;
    int odd = jbool(c, "use_karras_sigmas", 0) || jbool(c, "use_exponential_sigmas", 0) ||
              jbool(c, "use_beta_sigmas", 0) || jbool(c, "invert_sigmas", 0) || jbool(c, "stochastic_sampling", 0);
    json_free(c); free(arena);
    if (!s->dynamic || odd || !shift_one) {
        fprintf(stderr, "[qwenimage] scheduler_config.json: only the dynamic-shift Euler schedule is implemented\n");
        exit(1);
    }
}

/* sig[steps+1] (last = 0) and ts[steps] = sig * 1000, float math like numpy's
 * float32 arrays in the pipeline. */
static void sched_sigmas(const Sched *s, int steps, int seq_len, float *sig, float *ts, double *mu_out){
    double m = (s->max_shift - s->base_shift) / (s->max_seq - s->base_seq);
    double mu = seq_len * m + (s->base_shift - m * s->base_seq);
    float e = (float)exp(mu);
    for (int i = 0; i < steps; i++) {
        double lin = steps == 1 ? 1.0 : 1.0 + (1.0 / steps - 1.0) * i / (steps - 1);
        float t = (float)lin;
        sig[i] = s->exponential ? e / (e + (1.f / t - 1.f)) : (float)mu / ((float)mu + (1.f / t - 1.f));
    }
    if (s->shift_terminal > 0) {
        float scale = (1.f - sig[steps - 1]) / (float)(1.0 - s->shift_terminal);
        for (int i = 0; i < steps; i++) sig[i] = 1.f - (1.f - sig[i]) / scale;
    }
    sig[steps] = 0.f;
    for (int i = 0; i < steps; i++) ts[i] = sig[i] * 1000.f;
    if (mu_out) *mu_out = mu;
}

/* ---- RNG: splitmix64 + Box-Muller -------------------------------------------- */

static uint64_t g_rng;
static uint64_t rng_next(void){
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static void randn(float *x, size_t n, uint64_t seed){
    g_rng = seed * 0x2545F4914F6CDD1Dull + 1;
    for (size_t i = 0; i < n; i += 2) {
        double u1 = ((rng_next() >> 11) + 1.0) / 9007199254740993.0, u2 = (rng_next() >> 11) / 9007199254740992.0;
        double r = sqrt(-2.0 * log(u1)), a = 6.283185307179586 * u2;
        x[i] = (float)(r * cos(a));
        if (i + 1 < n) x[i + 1] = (float)(r * sin(a));
    }
}

/* ---- PNG (stored deflate blocks: valid, uncompressed, no zlib) ---------------- */

static uint32_t crc_tab[256];
static void crc_init(void){
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
        crc_tab[n] = c;
    }
}
static uint32_t crc_upd(uint32_t c, const uint8_t *b, size_t n){
    for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ b[i]) & 0xff] ^ (c >> 8);
    return c;
}
static void be32(uint8_t *p, uint32_t v){ p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void png_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n){
    uint8_t h[8]; be32(h, n); memcpy(h + 4, type, 4);
    fwrite(h, 1, 8, f);
    if (n) fwrite(data, 1, n, f);
    uint32_t c = crc_upd(0xffffffffu, (const uint8_t *)type, 4);
    c = crc_upd(c, data, n) ^ 0xffffffffu;
    uint8_t t[4]; be32(t, c); fwrite(t, 1, 4, f);
}
static int write_png(const char *path, const uint8_t *rgba, int w, int h){
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    crc_init();
    static const uint8_t sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    fwrite(sig, 1, 8, f);
    uint8_t ih[13]; be32(ih, w); be32(ih + 4, h); ih[8] = 8; ih[9] = 6; ih[10] = ih[11] = ih[12] = 0;
    png_chunk(f, "IHDR", ih, 13);
    size_t raw_n = (size_t)h * (1 + (size_t)w * 4);
    uint8_t *raw = xmalloc(raw_n);
    for (int y = 0; y < h; y++) { raw[(size_t)y * (1 + w * 4)] = 0; memcpy(raw + (size_t)y * (1 + w * 4) + 1, rgba + (size_t)y * w * 4, (size_t)w * 4); }
    size_t blocks = (raw_n + 65534) / 65535;
    size_t z_n = 2 + raw_n + blocks * 5 + 4;
    uint8_t *z = xmalloc(z_n), *o = z;
    *o++ = 0x78; *o++ = 0x01;
    uint32_t a = 1, b = 0;
    for (size_t off = 0; off < raw_n; off += 65535) {
        size_t n = raw_n - off < 65535 ? raw_n - off : 65535;
        *o++ = off + n == raw_n ? 1 : 0;
        *o++ = n & 0xff; *o++ = n >> 8; *o++ = ~n & 0xff; *o++ = (~n >> 8) & 0xff;
        memcpy(o, raw + off, n); o += n;
    }
    for (size_t i = 0; i < raw_n; i++) { a = (a + raw[i]) % 65521; b = (b + a) % 65521; }
    be32(o, (b << 16) | a); o += 4;
    png_chunk(f, "IDAT", z, (uint32_t)(o - z));
    png_chunk(f, "IEND", NULL, 0);
    free(raw); free(z);
    return fclose(f);
}

/* ---- the pipeline ------------------------------------------------------------ */

typedef struct {
    char model[2048];
    Te te; Dit dit; Sched sch;
#ifdef QI_HAVE_VAE
    QiVae *vae;
#endif
    int te_resident;
    /* the last prompt's prefix: a new seed for the same prompt skips the text
     * encoder and the prefix pass entirely */
    char *last_prompt; Prefix prefix;
} Engine;

typedef struct { const char *id; int (*cancelled)(const char *id); int preview; } Progress;

#include "qwenimage_preview.h"   /* QI_PREVIEW_RGB: the fitted latent -> RGB map */

/* The picture as it stands after a step: the flow-matching estimate of the clean
 * latent (x0 = x_t - sigma * v) through a linear latent -> RGB map, one pixel per
 * latent pixel. 64x3 multiply-adds per token: free next to a DiT step. */
static void json_escape_out(const char *s);
static void send_error(const char *id, const char *m);

static void emit_preview(const Progress *pg, int step, const float *lat, const float *v, float sigma,
                         int gh, int gw, int C){
    if (!pg || !pg->id || !pg->preview || C != 64) return;
    size_t n = (size_t)gh * gw * 3;
    uint8_t *rgb = xmalloc(n);
    for (int t = 0; t < gh * gw; t++) {
        const float *x = lat + (int64_t)t * C, *u = v + (int64_t)t * C;
        for (int c = 0; c < 3; c++) {
            float a = QI_PREVIEW_RGB[64][c];
            for (int k = 0; k < 64; k++) a += (x[k] - sigma * u[k]) * QI_PREVIEW_RGB[k][c];
            a = a * 255.f + 0.5f;
            rgb[t * 3 + c] = a < 0 ? 0 : a > 255 ? 255 : (uint8_t)a;
        }
    }
    printf("PREVIEW {\"id\":\""); json_escape_out(pg->id);
    printf("\",\"step\":%d,\"width\":%d,\"height\":%d,\"channels\":3,\"bytes\":%zu}\n", step, gw, gh, n);
    fwrite(rgb, 1, n, stdout);
    putchar('\n');
    fflush(stdout);
    free(rgb);
}

static void emit_progress(const Progress *pg, const char *stage, int step, int steps, double t0){
    if (!pg || !pg->id) return;
    printf("PROGRESS {\"id\":\""); json_escape_out(pg->id);
    printf("\",\"stage\":\"%s\",\"step\":%d,\"steps\":%d,\"elapsed\":%.2f}\n", stage, step, steps, now_s() - t0);
    fflush(stdout);
}

static void engine_init(Engine *e, const char *model){
    memset(e, 0, sizeof *e);
    snprintf(e->model, sizeof e->model, "%s", model);
    te_config(&e->te, model);
    dit_config(&e->dit, model);
    sched_config(&e->sch, model);
    te_tokenizer(&e->te, model);
    if (e->te.hidden != e->dit.ctx) {
        fprintf(stderr, "[qwenimage] text encoder hidden %d != transformer context %d\n", e->te.hidden, e->dit.ctx);
        exit(1);
    }
    /* The VAE is loaded only when the first image is decoded; its geometry is
     * checked now, so a VAE that does not fit the DiT (latent channels, 16x
     * upsampling, which the image buffers are sized from) fails at start-up
     * and not after a whole denoise. */
    char dir[2100]; snprintf(dir, sizeof dir, "%s/vae", model);
    char *arena = NULL;
    jval *v = read_json(dir, "config.json", &arena);
    jval *dm = json_get(v, "dim_mult");
    int zdim = (int)jnum(v, "z_dim", 0), ok = dm && dm->t == J_ARR && dm->len >= 1 && dm->len <= 8;
    for (int i = 0; ok && i < dm->len; i++) ok = dm->kids[i]->t == J_NUM && dm->kids[i]->num >= 1;
    int scale = ok ? 1 << (dm->len - 1) : 0;
    json_free(v); free(arena);
    if (!ok || scale != 16 || zdim != e->dit.in_ch) {
        fprintf(stderr, "[qwenimage] vae/config.json: z_dim %d and a %dx upsampling do not fit the transformer "
                        "(%d latent channels, 16x)\n", zdim, scale, e->dit.in_ch);
        exit(1);
    }
}

#ifdef QI_HAVE_VAE
/* qiv_load plus the same geometry check, against the weights actually loaded. */
static QiVae *engine_vae(Engine *e, char *msg, size_t msgn){
    char dir[2100]; snprintf(dir, sizeof dir, "%s/vae", e->model);
    QiVae *v = qiv_load(dir);
    if (!v) { snprintf(msg, msgn, "cannot load the VAE"); return NULL; }
    if (v->scale != 16 || v->z_dim != e->dit.in_ch) {
        snprintf(msg, msgn, "the VAE (z_dim %d, %dx) does not fit the transformer (%d channels, 16x)",
                 v->z_dim, v->scale, e->dit.in_ch);
        qiv_free(v);
        return NULL;
    }
    return v;
}
#endif

/* The prompt's prefix, cached for the next request with the same prompt. */
/* Prompt tokens, template included. Attention over the prompt keeps a full
 * score matrix, so an unbounded prompt is an unbounded allocation (40000 digits,
 * one token each, took 6.4 GB on the tiny model); the pipeline's own prompts
 * are a few dozen tokens. */
#define QI_MAX_PROMPT_TOKENS 1024

static int engine_prefix(Engine *e, const char *prompt, float **emb_out, int *L_out, char *msg, size_t msgn){
    if (e->last_prompt && !strcmp(e->last_prompt, prompt) && e->prefix.K && !emb_out) {
        *L_out = e->prefix.L; return 1;
    }
    prefix_free(&e->prefix, e->dit.layers);
    free(e->last_prompt); e->last_prompt = NULL;
    int n, drop;
    int *ids = te_encode_prompt(&e->te, prompt, &n, &drop);
    if (n <= drop) { snprintf(msg, msgn, "empty prompt"); free(ids); return -1; }
    if (n > QI_MAX_PROMPT_TOKENS) {
        snprintf(msg, msgn, "prompt too long: %d tokens, the limit is %d", n, QI_MAX_PROMPT_TOKENS);
        free(ids); return -1;
    }
    te_load(&e->te, e->model);
    float *h = te_forward(&e->te, ids, n);
    free(ids);
    if (!e->te_resident) te_unload(&e->te);
    int L = n - drop;
    float *emb = fmalloc((size_t)L * e->te.hidden);
    memcpy(emb, h + (int64_t)drop * e->te.hidden, (size_t)L * e->te.hidden * 4);
    free(h);
    dit_load(&e->dit, e->model);
    dit_prefix(&e->dit, emb, L, &e->prefix, NULL);
    if (emb_out) *emb_out = emb; else free(emb);
    e->last_prompt = strdup(prompt);
    *L_out = L;
    return 0;
}

/* Generates one image. rgba is [height][width][4]. Returns 0, or -1 with msg. */
static int engine_generate(Engine *e, const char *prompt, int width, int height, int steps, uint64_t seed,
                           uint8_t *rgba, double timings[3], const Progress *pg, char *msg, size_t msgn){
    double t0 = now_s();
    int gh = height / 16, gw = width / 16, N = gh * gw, C = e->dit.in_ch;
    emit_progress(pg, "encode", 0, steps, t0);
    int L;
    if (engine_prefix(e, prompt, NULL, &L, msg, msgn) < 0) return -1;
    double t1 = now_s();
    float *lat = fmalloc((size_t)N * C), *np = fmalloc((size_t)N * C);
    randn(lat, (size_t)N * C, seed);
    float *sig = fmalloc(steps + 1), *ts = fmalloc(steps);
    sched_sigmas(&e->sch, steps, N, sig, ts, NULL);
    DitStep s; dit_step_init(&s, &e->dit, &e->prefix, gh, gw);
    int rc = 0;
    for (int i = 0; i < steps; i++) {
        emit_progress(pg, "denoise", i, steps, t0);
        if (pg && pg->cancelled && pg->cancelled(pg->id)) { snprintf(msg, msgn, "cancelled"); rc = -1; break; }
        double ts0 = now_s();
        g_t_lin = g_t_att = 0;
        dit_forward(&e->dit, &e->prefix, &s, lat, ts[i], np);
        emit_preview(pg, i + 1, lat, np, sig[i], gh, gw, C);
        float dt = sig[i + 1] - sig[i];
        for (int64_t k = 0; k < (int64_t)N * C; k++) lat[k] += dt * np[k];
        double tstep = now_s() - ts0;
        fprintf(stderr, "[qwenimage] step %d/%d  t=%.1f  %.2f s", i + 1, steps, ts[i], tstep);
        if (g_prof) fprintf(stderr, "  (linears %.2f, attention %.2f, rest %.2f)", g_t_lin, g_t_att, tstep - g_t_lin - g_t_att);
        fputc('\n', stderr);
    }
    dit_step_free(&s);
    double t2 = now_s();
    if (rc == 0) {
        emit_progress(pg, "decode", steps, steps, t0);
#ifdef QI_HAVE_VAE
        if (!e->vae && !(e->vae = engine_vae(e, msg, msgn))) rc = -1;
        if (rc == 0 && qiv_decode(e->vae, lat, gh, gw, rgba, NULL) != 0) { snprintf(msg, msgn, "VAE decode failed"); rc = -1; }
#else
        /* no VAE compiled in: show the first three latent channels, stretched */
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                const float *z = lat + ((int64_t)(y / 16) * gw + x / 16) * C;
                uint8_t *o = rgba + ((int64_t)y * width + x) * 4;
                for (int c = 0; c < 3; c++) { float v = 128.f + 40.f * z[c]; o[c] = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }
                o[3] = 255;
            }
#endif
    }
    double t3 = now_s();
    if (timings) { timings[0] = t1 - t0; timings[1] = t2 - t1; timings[2] = t3 - t2; }
    free(lat); free(np); free(sig); free(ts);
    return rc;
}

static int check_size(int w, int h, int steps, char *msg, size_t n){
    if (w < 256 || h < 256 || w > 2048 || h > 2048 || w % 32 || h % 32) {
        snprintf(msg, n, "width and height must be multiples of 32 between 256 and 2048 (got %dx%d)", w, h);
        return -1;
    }
    /* One step is refused, as the reference would give NaN: with shift_terminal
     * the schedule's last sigma is also its first, and the stretch divides by
     * 1 - sigma = 0 (diffusers' stretch_shift_to_terminal does the same). */
    if (steps < 2 || steps > 200) { snprintf(msg, n, "steps must be between 2 and 200 (got %d)", steps); return -1; }
    return 0;
}

/* ---- serve ------------------------------------------------------------------- */

/* Our own line reader on fd 0: stdio's buffer would hide a CANCEL that arrived
 * together with the GEN from coli_serve_stdin_ready(). */
/* The buffer grows up to QI_MAX_LINE (the gateway's own body limit). A longer
 * line is read through to its newline and discarded, and read_line hands back
 * "TOOLONG <id>" with the id found at its start, so the client that sent it
 * gets an ERROR instead of waiting forever for an answer to a request the
 * engine never saw. */
#define QI_MAX_LINE ((size_t)4 << 20)
static char *g_in; static size_t g_in_n, g_in_cap;
static int g_skipping; static char g_skip_id[128];

static void note_line_id(const char *buf, size_t n){
    snprintf(g_skip_id, sizeof g_skip_id, "?");
    const char *k = NULL;                /* memmem is not in the Windows CRT */
    for (size_t i = 0; i + 6 <= (n < 4096 ? n : 4096); i++)
        if (!memcmp(buf + i, "\"id\":\"", 6)) { k = buf + i; break; }
    if (!k) return;
    k += 6;
    size_t i = 0;
    while (i + 1 < sizeof g_skip_id && k + i < buf + n && k[i] != '"' && k[i] != '\\' && (unsigned char)k[i] >= 0x20) {
        g_skip_id[i] = k[i]; i++;
    }
    g_skip_id[i] = 0;
}

static char *read_line(int block){
    for (;;) {
        char *nl = g_in_n ? memchr(g_in, '\n', g_in_n) : NULL;
        if (nl) {
            size_t len = (size_t)(nl - g_in);
            char *line;
            if (g_skipping) {                 /* the tail of an overlong line */
                g_skipping = 0;
                line = xmalloc(sizeof g_skip_id + 16);
                snprintf(line, sizeof g_skip_id + 16, "TOOLONG %s", g_skip_id);
            } else {
                line = xmalloc(len + 1);
                memcpy(line, g_in, len); line[len] = 0;
            }
            memmove(g_in, nl + 1, g_in_n - len - 1); g_in_n -= len + 1;
            return line;
        }
        if (g_in_n == QI_MAX_LINE) {
            if (!g_skipping) { note_line_id(g_in, g_in_n); g_skipping = 1; }
            g_in_n = 0;
        }
        if (!block && !coli_serve_stdin_ready()) return NULL;
        if (g_in_n == g_in_cap) {
            size_t cap = g_in_cap ? g_in_cap * 2 : (size_t)1 << 16;
            if (cap > QI_MAX_LINE) cap = QI_MAX_LINE;
            char *grown = realloc(g_in, cap);
            if (!grown) { fprintf(stderr, "[qwenimage] out of memory reading stdin\n"); exit(1); }
            g_in = grown; g_in_cap = cap;
        }
        ssize_t r = read(0, g_in + g_in_n, g_in_cap - g_in_n);
        if (r <= 0) return NULL;
        g_in_n += (size_t)r;
    }
}

static int g_cancel_pending;
static char g_cancel_id[128];
/* Lines that arrive while an image is being made and are not a CANCEL for it
 * (a GEN from a client that gave up waiting, say) are kept for serve_loop,
 * never dropped: a dropped GEN is a client waiting forever for its answer. */
static char *g_later[64]; static int g_later_n;
static int serve_cancelled(const char *id){
    char *line;
    while ((line = read_line(0))) {
        if (strncmp(line, "CANCEL ", 7)) {
            if (g_later_n < (int)(sizeof g_later / sizeof g_later[0])) { g_later[g_later_n++] = line; continue; }
            if (!strncmp(line, "GEN ", 4)) {        /* the queue is full: answer rather than drop */
                char *arena = NULL; jval *j = json_parse(line + 4, &arena);
                jval *v = j ? json_get(j, "id") : NULL;
                send_error(v && v->t == J_STR ? v->str : "?", "engine busy");
                if (j) json_free(j);
                free(arena);
            }
            free(line);
            continue;
        }
        {
            char *arena = NULL; jval *j = json_parse(line + 7, &arena);
            jval *v = j ? json_get(j, "id") : NULL;
            if (v && v->t == J_STR) { snprintf(g_cancel_id, sizeof g_cancel_id, "%s", v->str); g_cancel_pending = 1; }
            if (j) json_free(j);
            free(arena);
        }
        free(line);
    }
    return g_cancel_pending && !strcmp(g_cancel_id, id);
}

static void json_escape_out(const char *s){
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
}
static void send_error(const char *id, const char *m){
    printf("ERROR {\"id\":\""); json_escape_out(id); printf("\",\"message\":\""); json_escape_out(m); printf("\"}\n");
    fflush(stdout);
}

static int serve_loop(Engine *e, int dw, int dh, int dsteps){
    printf("READY {\"model\":\"qwen-image-2.1\",\"default_width\":%d,\"default_height\":%d,\"default_steps\":%d,"
           "\"min_side\":256,\"max_side\":2048,\"multiple\":32}\n", dw, dh, dsteps);
    fflush(stdout);
    char *line;
    for (;;) {
        if (g_later_n) {                              /* what came in during the last image */
            line = g_later[0];
            memmove(g_later, g_later + 1, sizeof g_later[0] * (size_t)--g_later_n);
        } else if (!(line = read_line(1))) break;
        if (!strncmp(line, "TOOLONG ", 8)) { send_error(line + 8, "request line too long"); free(line); continue; }
        if (strncmp(line, "GEN ", 4)) { free(line); continue; }       /* CANCEL with nothing running, noise */
        char *arena = NULL;
        jval *j = json_parse(line + 4, &arena);
        char id[128] = "?", msg[512] = "";
        jval *v;
        if (j && (v = json_get(j, "id")) && v->t == J_STR) snprintf(id, sizeof id, "%s", v->str);
        const char *prompt = j && (v = json_get(j, "prompt")) && v->t == J_STR ? v->str : NULL;
        int w = (int)jnum(j, "width", dw), h = (int)jnum(j, "height", dh), steps = (int)jnum(j, "steps", dsteps);
        uint64_t seed = (uint64_t)jnum(j, "seed", (double)(time(NULL) & 0x7fffffff));
        if (!j) send_error(id, "bad GEN line (not JSON)");
        else if (!prompt || !*prompt) send_error(id, "empty prompt");
        else if (check_size(w, h, steps, msg, sizeof msg)) send_error(id, msg);
        else {
            g_cancel_pending = 0;
            uint8_t *rgba = xmalloc((size_t)w * h * 4);
            double tm[3];
            Progress pg = { id, serve_cancelled, (int)jnum(j, "preview", 0) };
            if (engine_generate(e, prompt, w, h, steps, seed, rgba, tm, &pg, msg, sizeof msg)) send_error(id, msg);
            else {
                printf("IMAGE {\"id\":\""); json_escape_out(id);
                printf("\",\"width\":%d,\"height\":%d,\"channels\":4,\"bytes\":%lld,\"seed\":%llu,\"steps\":%d,"
                       "\"timings\":{\"encode\":%.2f,\"denoise\":%.2f,\"decode\":%.2f}}\n",
                       w, h, (long long)w * h * 4, (unsigned long long)seed, steps, tm[0], tm[1], tm[2]);
                fwrite(rgba, 1, (size_t)w * h * 4, stdout);
                putchar('\n');
                fflush(stdout);
            }
            free(rgba);
        }
        if (j) json_free(j);
        free(arena); free(line);
    }
    return 0;
}

/* ---- oracle ------------------------------------------------------------------ */

/* Compare a computed tensor with the reference: max abs error and the ratio of
 * norms of the error and the reference. */
static double cmp(const char *what, const float *got, const float *want, size_t n){
    double emax = 0, en = 0, wn = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - want[i];
        if (fabs(d) > emax) emax = fabs(d);
        en += d * d; wn += (double)want[i] * want[i];
    }
    double rel = wn > 0 ? sqrt(en / wn) : sqrt(en);
    fprintf(stderr, "[oracle] %-26s n=%-9zu max|err| %.3e  rel %.3e\n", what, n, emax, rel);
    return rel;
}
static float *ref_tensor(shards *S, const char *name, int64_t *n){
    st_tensor *t = st_find(S, name);
    if (!t) return NULL;
    int64_t numel = 1; for (int i = 0; i < t->rank; i++) numel *= t->shape[i];
    float *v = fmalloc((size_t)numel);
    if (t->dtype == 6) {    /* int64 ids */
        int64_t *raw = xmalloc((size_t)numel * 8);
        st_pread_full(t->fd, raw, numel * 8, t->off, "ref");
        for (int64_t i = 0; i < numel; i++) v[i] = (float)raw[i];
        free(raw);
    } else st_read_f32(S, name, v, 0);
    if (n) *n = numel;
    return v;
}

static float *ref_named(shards *S, const char *fmt, int i){
    char name[64]; snprintf(name, sizeof name, fmt, i);
    return ref_tensor(S, name, NULL);
}

/* Every stage against a reference dump (tools/make_qwenimage_tiny.py for the tiny
 * pipeline, tools/qwenimage_ref.py for the real one). Each stage starts from the
 * REFERENCE input of that stage, so an error is reported where it is made and not
 * carried into the next one; the chained run from our own outputs comes last.
 * With COLI_IMG_BITS=16 the weights are the checkpoint's bf16 values exactly, so
 * what remains is summation order. */
static int run_oracle(Engine *e, const char *refdir){
    char *arena = NULL;
    jval *meta = read_json(refdir, "ref.json", &arena);
    jval *pv = json_get(meta, "prompt");
    const char *prompt = pv && pv->t == J_STR ? pv->str : NULL;
    int width = (int)jnum(meta, "width", 0), height = (int)jnum(meta, "height", 0);
    int steps = (int)jnum(meta, "steps", 0), drop_ref = (int)jnum(meta, "drop_idx", -1);
    jval *idv = json_get(meta, "input_ids");
    if (!prompt || !width || !height || !steps || !idv || idv->t != J_ARR) {
        fprintf(stderr, "[oracle] ref.json needs prompt, width, height, steps, input_ids\n"); return 2;
    }
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    int fails = 0;
    const double TIGHT = 2e-4, LOOSE = 2e-3;
    /* 1. tokens */
    int n, drop;
    int *ids = te_encode_prompt(&e->te, prompt, &n, &drop);
    int nref = idv->len, tok_ok = nref == n && drop == drop_ref;
    for (int i = 0; tok_ok && i < n; i++) tok_ok = (int)idv->kids[i]->num == ids[i];
    fprintf(stderr, "[oracle] tokens: %d (ref %d), drop_idx %d (ref %d): %s\n", n, nref, drop, drop_ref,
            tok_ok ? "IDENTICAL" : "DIFFERENT");
    fails += !tok_ok;
    int *use = xmalloc(sizeof(int) * nref);
    for (int i = 0; i < nref; i++) use[i] = (int)idv->kids[i]->num;
    /* 2. text encoder, on the reference ids */
    te_load(&e->te, e->model);
    float *h = te_forward(&e->te, use, nref);
    te_unload(&e->te);
    float *last = ref_named(&R, "te_hidden_%02d", e->te.layers);
    if (last) fails += cmp("text encoder last layer", h, last, (size_t)nref * e->te.hidden) > TIGHT;
    free(last);
    int L = nref - drop_ref;
    float *emb_ref = ref_tensor(&R, "prompt_embeds", NULL);
    if (emb_ref) fails += cmp("prompt_embeds", h + (int64_t)drop_ref * e->te.hidden, emb_ref, (size_t)L * e->te.hidden) > TIGHT;
    /* 3. DiT: the prefix from the REFERENCE embeddings */
    dit_load(&e->dit, e->model);
    const float *emb = emb_ref ? emb_ref : h + (int64_t)drop_ref * e->te.hidden;
    int D = e->dit.dim;
    float *txt = fmalloc((size_t)L * D);
    Prefix P; memset(&P, 0, sizeof P);
    dit_prefix(&e->dit, emb, L, &P, txt);
    float *txt_ref = ref_tensor(&R, "step0_txt_in", NULL);
    if (txt_ref) fails += cmp("txt_in", txt, txt_ref, (size_t)L * D) > TIGHT;
    int gh = height / 16, gw = width / 16, N = gh * gw, C = e->dit.in_ch;
    float *sig = fmalloc(steps + 1), *ts = fmalloc(steps);
    double mu;
    sched_sigmas(&e->sch, steps, N, sig, ts, &mu);
    float *sig_ref = ref_tensor(&R, "sigmas", NULL);
    if (sig_ref) fails += cmp("sigmas", sig, sig_ref, steps + 1) > 1e-6;
    float *temb_ref = ref_tensor(&R, "step0_temb", NULL), *mod_ref = ref_tensor(&R, "step0_modulation", NULL);
    if (temb_ref && mod_ref) {
        float *temb = fmalloc(D), *mod = fmalloc(4 * (size_t)D), *outs = fmalloc(D);
        dit_temb(&e->dit, ts[0], temb);
        fails += cmp("temb (t of step 0)", temb, temb_ref, D) > TIGHT;
        dit_temb(&e->dit, 0.f, temb);
        fails += cmp("temb (t = 0, prompt)", temb, temb_ref + D, D) > TIGHT;
        dit_modulation(&e->dit, ts[0], mod, outs);
        fails += cmp("modulation (step 0)", mod, mod_ref, 4 * (size_t)D) > TIGHT;
        free(temb); free(mod); free(outs);
    }
    DitStep s; dit_step_init(&s, &e->dit, &P, gh, gw);
    float *np = fmalloc((size_t)N * C), *lat0 = ref_tensor(&R, "latents_init", NULL), *mine = fmalloc((size_t)N * C);
    if (!lat0) { fprintf(stderr, "[oracle] ref has no latents_init\n"); return 2; }
    memcpy(mine, lat0, (size_t)N * C * 4);
    double worst = 0;
    for (int i = 0; i < steps; i++) {
        float *in = i == 0 ? lat0 : ref_named(&R, "latents_%03d", i - 1);
        if (!in) { fprintf(stderr, "[oracle] ref has no latents_%03d\n", i - 1); return 2; }
        dit_forward(&e->dit, &P, &s, in, ts[i], np);
        float *want = ref_named(&R, "noise_pred_%03d", i);
        if (want) {
            char w[64]; snprintf(w, sizeof w, "noise_pred step %d", i);
            double r = cmp(w, np, want, (size_t)N * C); if (r > worst) worst = r;
            fails += r > LOOSE; free(want);
        }
        if (in != lat0) free(in);
        /* the chained trajectory, from our own predictions only */
        dit_forward(&e->dit, &P, &s, mine, ts[i], np);
        float dt = sig[i + 1] - sig[i];
        for (int64_t k = 0; k < (int64_t)N * C; k++) mine[k] += dt * np[k];
    }
    float *fin = ref_tensor(&R, "latents_final", NULL);
    if (fin) fails += cmp("final latents, chained", mine, fin, (size_t)N * C) > 2e-2;
    dit_step_free(&s);
#ifdef QI_HAVE_VAE
    {
        char vmsg[256];
        QiVae *vae = engine_vae(e, vmsg, sizeof vmsg);
        if (!vae) { fprintf(stderr, "[oracle] %s\n", vmsg); fails++; }
        if (vae && fin) {
            uint8_t *rgba = xmalloc((size_t)width * height * 4);
            float *of = fmalloc((size_t)4 * width * height);
            qiv_decode(vae, fin, gh, gw, rgba, of);
            float *img_ref = ref_tensor(&R, "vae_out", NULL);
            if (img_ref) fails += cmp("VAE output (ref latents)", of, img_ref, (size_t)4 * width * height) > LOOSE;
            st_tensor *rt = st_find(&R, "rgba");
            if (rt && rt->nbytes == (int64_t)width * height * 4) {
                uint8_t *want = xmalloc((size_t)rt->nbytes);
                st_pread_full(rt->fd, want, rt->nbytes, rt->off, "rgba");
                int64_t diff = 0, big = 0;
                for (int64_t k = 0; k < rt->nbytes; k++) { int d = abs((int)rgba[k] - (int)want[k]); diff += d > 0; big += d > 1; }
                fprintf(stderr, "[oracle] RGBA from ref latents: %lld of %lld bytes differ, %lld by more than 1\n",
                        (long long)diff, (long long)rt->nbytes, (long long)big);
                fails += big > rt->nbytes / 1000;
                free(want);
            }
            const char *tag = getenv("QWENIMAGE_ORACLE_TAG");
            char out[2200]; snprintf(out, sizeof out, "%s/oracle_c%s%s.png", refdir, tag ? "_" : "", tag ? tag : "");
            qiv_decode(vae, mine, gh, gw, rgba, NULL);
            if (!write_png(out, rgba, width, height)) fprintf(stderr, "[oracle] our own chained image: %s\n", out);
            /* PSNR of the RGB of our chained image against the reference's: the
             * number that says what int8 weights or activations cost in pixels */
            if (rt && rt->nbytes == (int64_t)width * height * 4) {
                uint8_t *want = xmalloc((size_t)rt->nbytes);
                st_pread_full(rt->fd, want, rt->nbytes, rt->off, "rgba");
                double se = 0; int64_t np3 = 0;
                for (int64_t k = 0; k < rt->nbytes; k++) if ((k & 3) != 3) { double d = (double)rgba[k] - want[k]; se += d * d; np3++; }
                double mse = se / np3;
                fprintf(stderr, "[oracle] chained image vs reference: PSNR %.2f dB (MSE %.3f)\n",
                        mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0, mse);
                free(want);
            }
            free(img_ref); free(rgba); free(of);
            qiv_free(vae);
        }
    }
#endif
    fprintf(stderr, "[oracle] worst noise_pred rel %.3e -> %s\n", worst, fails ? "MISMATCH" : "all stages within tolerance");
    return fails ? 1 : 0;
}

/* ---- main -------------------------------------------------------------------- */

static void usage(void){
    fprintf(stderr,
        "usage: qwenimage --model DIR --prompt TEXT [--width 768] [--height 512] [--steps 8] [--seed N] --out FILE.png\n"
        "       qwenimage --model DIR --serve\n"
        "       qwenimage --model DIR --ref REFDIR\n"
        "env:   COLI_IMG_BITS=8|16|32 weight storage (default 8: int8 rows)\n"
        "       COLI_IMG_ACT8=0  f32 activations in the DiT (default where VNNI exists: int8, about 2x per step)\n"
        "       COLI_IMG_TE=resident|stage  keep the text encoder loaded between prompts (serve default: resident)\n");
}

int main(int argc, char **argv){
    coli_env_check(CE_QWENIMAGE, "qwenimage");
    coli_env_dump(CE_QWENIMAGE, "qwenimage");
    /* Windows: the IMAGE and PREVIEW frames carry raw bytes; a text-mode stdout
     * would turn every 0x0A inside them into 0x0D 0x0A (#748) */
    coli_serve_binary_mode();
    const char *model = NULL, *prompt = NULL, *out = NULL, *ref = NULL;
    int width = 768, height = 512, steps = 8, serve = 0;
    uint64_t seed = 42; int seed_set = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--prompt") && v) { prompt = v; i++; }
        else if (!strcmp(a, "--out") && v) { out = v; i++; }
        else if (!strcmp(a, "--ref") && v) { ref = v; i++; }
        else if (!strcmp(a, "--width") && v) { width = atoi(v); i++; }
        else if (!strcmp(a, "--height") && v) { height = atoi(v); i++; }
        else if (!strcmp(a, "--steps") && v) { steps = atoi(v); i++; }
        else if (!strcmp(a, "--seed") && v) { seed = strtoull(v, NULL, 10); seed_set = 1; i++; }
        else if (!strcmp(a, "--threads") && v) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(v));
#endif
            i++;
        }
        else if (!strcmp(a, "--serve")) serve = 1;
        else { usage(); return 2; }
    }
    if (!model || (!serve && !ref && !getenv("QWENIMAGE_PRINT_TOKENS") && (!prompt || !out))) { usage(); return 2; }
    const char *b = getenv("COLI_IMG_BITS");
    if (b && *b) { g_bits = atoi(b); if (g_bits != 8 && g_bits != 16 && g_bits != 32) { usage(); return 2; } }
    /* On by default where VNNI exists: measured 2x on the step (768x512: 34.3 -> 18.3 s
     * per step on 8 Zen 4 cores) with pictures that cannot be told apart by eye
     * (fox and neon-sign prompts; 30.1 dB against the f32 reference, 35.6 without). */
    const char *a8 = getenv("COLI_IMG_ACT8");
    g_act8 = !(a8 && *a8 == '0');
    g_prof = getenv("QWENIMAGE_PROF") != NULL;
#ifndef QI_HAVE_VNNI
    if (g_act8 && a8 && *a8 == '1') fprintf(stderr, "[qwenimage] COLI_IMG_ACT8 needs VNNI; this build has none, staying on f32 activations\n");
    g_act8 = 0;
#endif
    if (!seed_set && !serve && !ref) seed = (uint64_t)time(NULL);
    static Engine e;
    engine_init(&e, model);
    if (ref) return run_oracle(&e, ref);
    if (getenv("QWENIMAGE_PRINT_TOKENS")) {          /* tokenizer check against the processor */
        int n, drop; int *ids = te_encode_prompt(&e.te, prompt ? prompt : "", &n, &drop);
        printf("drop %d ids", drop);
        for (int i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n"); free(ids); return 0;
    }
    if (serve) {
        const char *te = getenv("COLI_IMG_TE");
        e.te_resident = !(te && !strcmp(te, "stage"));
        te_load(&e.te, model);
        dit_load(&e.dit, model);
        return serve_loop(&e, width, height, steps);
    }
    char msg[512];
    if (check_size(width, height, steps, msg, sizeof msg)) { fprintf(stderr, "%s\n", msg); return 2; }
    uint8_t *rgba = xmalloc((size_t)width * height * 4);
    double tm[3];
    if (engine_generate(&e, prompt, width, height, steps, seed, rgba, tm, NULL, msg, sizeof msg)) {
        fprintf(stderr, "[qwenimage] %s\n", msg); return 1;
    }
    if (write_png(out, rgba, width, height)) { fprintf(stderr, "[qwenimage] cannot write %s\n", out); return 1; }
    fprintf(stderr, "[qwenimage] %s  %dx%d  %d steps  seed %llu  encode %.1f s  denoise %.1f s  decode %.1f s\n",
            out, width, height, steps, (unsigned long long)seed, tm[0], tm[1], tm[2]);
    return 0;
}
