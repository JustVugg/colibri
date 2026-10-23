/* mimo.c -- Xiaomi MiMo-V2.6 (Flash 309B/15B, Pro 1.02T/42B) on a machine that
 * cannot hold it.
 *
 * The released checkpoint is read as it is, no conversion step:
 *
 *   experts   MXFP4 g32 (e2m1 nibble pairs, low nibble = even column, one e8m0
 *             byte per 32 columns), the six tensors of an expert back to back in
 *             the shard, so an expert is ONE read. 256 of them per layer on
 *             Flash, 8 routed per token, cached in RAM by a per-layer LRU.
 *   dense     qkv_proj and the layer-0 MLP in FP8 e4m3 with a F32 scale per
 *             128x128 block; o_proj, router, norms, sinks, embed, lm_head BF16.
 *
 * What makes MiMo MiMo, each mirroring modeling_mimo_v2.py:
 *   hybrid attention   hybrid_layer_pattern: 1 = sliding window (128 keys,
 *                      itself included), 0 = full attention; the two kinds have
 *                      their own KV head count and their own RoPE theta
 *   partial RoPE       on the first int(head_dim * 0.334) dims of each head,
 *                      rotate-half style; head_dim 192 for q/k, 128 for v
 *   value scale        v *= attention_value_scale before attention
 *   attention sink     SWA layers carry one logit per head that joins the
 *                      softmax denominator and nothing else
 *   router             sigmoid scores, e_score_correction_bias picks but does
 *                      not weigh, top-k renormalised; no shared expert
 *
 * One storage detail the modeling code does not describe: the fused qkv_proj is
 * PRE-SHARDED in num_key_value_heads chunks [Q_0|K_0|V_0|Q_1|K_1|V_1|...] and
 * its FP8 scale grid is tiled per chunk (a Flash full-attention layer has 108
 * scale rows, not the 106 a contiguous grid would have). The layout is vLLM's
 * (vllm/model_executor/models/mimo_v2.py, _shard_fp8_qkv_proj); the loader
 * undoes it and the tiny oracle stores it the same way so a wrong reading fails.
 *
 * Held to Xiaomi's own modeling code through tools/make_mimo_ref.py (vendor
 * files pinned by SHA-256) on the tiny fixture of tools/make_mimo_tiny.py.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cli_args.h"
#include "compat.h"
#include "json.h"
#include "st.h"
#include "quant.h"
#include "omp_tune.h"
#include "kv_prefix.h"
#include "pin_pool.h"
#include "tok.h"
#include "serve_codec.h"
#include "serve_poll.h"
#include "stop_ids.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "vk_tier.h"       /* the shared Vulkan routed-expert tier (COLI_VULKAN=1; stubs without VK=1) */
#ifdef COLI_VULKAN
/* VK=1 and COLI_VULKAN=1: the routed experts go to the shared expert tier
 * (vk_tier.c) as the release's MXFP4, and the dense matrices of the trunk and of
 * the vision tower to the device where coli_vk_dense() puts them, in whatever
 * form MIMO_DENSE_BITS gave them (FP8, BF16, int8 or f32). The router, and every
 * matrix whose upload fails, stays on the CPU path below. The backend's dense
 * matmuls share one command buffer: main thread only. */
#include "backend_vulkan.h"
#include <pthread.h>
static int g_vk_ready;
/* COLI_VK_DENSE_HOST: the dense matrices on the device only (mimo_dho_start); the
 * per-matrix path then runs on the device whatever coli_vk_dense() says */
static int g_mimo_dho;
#endif

#define MIMO_MAX_LAYERS 128
#define MIMO_MAX_TOPK 16
#define MIMO_MAX_CHUNKS 16

/* ------------------------------------------------------------------ utils ---- */

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double rss_gb(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return usage.ru_maxrss / 1e9;
#else
    return usage.ru_maxrss / 1e6;
#endif
}

static void *xmalloc(size_t bytes, const char *what) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "[mimo] out of memory allocating %s (%zu bytes)\n", what, bytes); exit(1); }
    return p;
}

static void *xcalloc(size_t n, size_t size, const char *what) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) { fprintf(stderr, "[mimo] out of memory allocating %s\n", what); exit(1); }
    return p;
}

static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : fallback;
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---------------------------------------------------------------- config ---- */

/* kind 0 = full attention, 1 = sliding window */
typedef struct {
    int vocab, hidden, n_layers, dense_inter, moe_inter, n_experts, topk;
    int heads[2], kv_heads[2], head_dim[2], v_dim[2], rope_dim[2];
    float theta[2];
    int sink[2];
    int swa[MIMO_MAX_LAYERS], moe[MIMO_MAX_LAYERS];
    int window, chunks, norm_topk;
    float v_scale, eps, route_scale;
    int max_positions;
    int image_token_id, vision_start_id, vision_end_id;
} Cfg;

static double jnum(jval *o, const char *key, double fallback) {
    jval *v = o ? json_get(o, key) : NULL;
    return (v && v->t == J_NUM) ? v->num : fallback;
}

static int jbool(jval *o, const char *key, int fallback) {
    jval *v = o ? json_get(o, key) : NULL;
    if (!v) return fallback;
    if (v->t == J_BOOL) return v->boolean != 0;
    if (v->t == J_NUM) return v->num != 0;
    return fallback;
}

static char *read_text(const char *path, long *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    if (size < 0 || size > (64l << 20)) { fclose(f); return NULL; }
    char *text = xmalloc((size_t)size + 1, path);
    if (fread(text, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(text); return NULL; }
    text[size] = 0; fclose(f);
    if (size_out) *size_out = size;
    return text;
}

static void die_cfg(const char *what) {
    fprintf(stderr, "[mimo] config.json: %s -- refusing\n", what);
    exit(1);
}

static void load_cfg(Cfg *c, const char *dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char *text = read_text(path, NULL);
    if (!text) { fprintf(stderr, "[mimo] cannot read %s\n", path); exit(1); }
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    if (!root) die_cfg("not JSON");
    jval *mt = json_get(root, "model_type");
    if (!mt || mt->t != J_STR || strcmp(mt->str, "mimo_v2"))
        die_cfg("model_type is not mimo_v2");
    memset(c, 0, sizeof(*c));
    c->vocab = (int)jnum(root, "vocab_size", 0);
    c->hidden = (int)jnum(root, "hidden_size", 0);
    c->n_layers = (int)jnum(root, "num_hidden_layers", 0);
    c->dense_inter = (int)jnum(root, "intermediate_size", 0);
    c->moe_inter = (int)jnum(root, "moe_intermediate_size", 0);
    c->n_experts = (int)jnum(root, "n_routed_experts", 0);
    c->topk = (int)jnum(root, "num_experts_per_tok", 0);
    c->heads[0] = (int)jnum(root, "num_attention_heads", 0);
    c->kv_heads[0] = (int)jnum(root, "num_key_value_heads", 0);
    c->head_dim[0] = (int)jnum(root, "head_dim", 0);
    c->v_dim[0] = (int)jnum(root, "v_head_dim", c->head_dim[0]);
    c->heads[1] = (int)jnum(root, "swa_num_attention_heads", c->heads[0]);
    c->kv_heads[1] = (int)jnum(root, "swa_num_key_value_heads", c->kv_heads[0]);
    c->head_dim[1] = (int)jnum(root, "swa_head_dim", c->head_dim[0]);
    c->v_dim[1] = (int)jnum(root, "swa_v_head_dim", c->v_dim[0]);
    double prf = jnum(root, "partial_rotary_factor", 1.0);
    for (int k = 0; k < 2; k++) c->rope_dim[k] = (int)(c->head_dim[k] * prf);
    c->theta[0] = (float)jnum(root, "rope_theta", 10000.0);
    c->theta[1] = (float)jnum(root, "swa_rope_theta", c->theta[0]);
    c->sink[0] = jbool(root, "add_full_attention_sink_bias", 0);
    c->sink[1] = jbool(root, "add_swa_attention_sink_bias", 0);
    c->window = (int)jnum(root, "sliding_window", 0);
    c->v_scale = (float)jnum(root, "attention_value_scale", 1.0);
    c->eps = (float)jnum(root, "layernorm_epsilon", 1e-6);
    c->route_scale = (float)jnum(root, "routed_scaling_factor", 1.0);
    c->norm_topk = jbool(root, "norm_topk_prob", 1);
    c->max_positions = (int)jnum(root, "max_position_embeddings", 32768);
    c->image_token_id = (int)jnum(root, "image_token_id", -1);
    c->vision_start_id = (int)jnum(root, "vision_start_token_id", -1);
    c->vision_end_id = (int)jnum(root, "vision_end_token_id", -1);
    c->chunks = c->kv_heads[0];
    /* MIMO_LAYERS=N: only the first N layers, then the final norm and head. A
     * checking aid (tools/mimo_real_check.py compares it with Xiaomi's code on
     * the real weights), never an answer anyone should read. */
    int keep = env_int("MIMO_LAYERS", 0);

    jval *pat = json_get(root, "hybrid_layer_pattern");
    jval *freq = json_get(root, "moe_layer_freq");
    if (c->n_layers < 1 || c->n_layers > MIMO_MAX_LAYERS) die_cfg("num_hidden_layers out of range");
    if (!pat || pat->t != J_ARR || pat->len != c->n_layers) die_cfg("hybrid_layer_pattern must list every layer");
    if (!freq || freq->t != J_ARR || freq->len != c->n_layers) die_cfg("moe_layer_freq must list every layer");
    for (int i = 0; i < c->n_layers; i++) {
        c->swa[i] = (int)pat->kids[i]->num == 1;
        c->moe[i] = (int)freq->kids[i]->num != 0;
    }
    if (keep > 0 && keep < c->n_layers) {
        fprintf(stderr, "[mimo] MIMO_LAYERS=%d: a truncated model for checking, not for answers\n", keep);
        c->n_layers = keep;
    }
    /* Everything below sizes a buffer or indexes one: refuse a config that
     * would make any of it overflow or divide by zero, before reading a byte. */
    if (c->vocab < 2 || c->vocab > (1 << 22)) die_cfg("vocab_size out of range");
    if (c->hidden < 32 || c->hidden > 65536 || c->hidden % 32) die_cfg("hidden_size must be a positive multiple of 32");
    if (c->moe_inter < 32 || c->moe_inter % 32 || c->moe_inter > 65536) die_cfg("moe_intermediate_size must be a positive multiple of 32");
    if (c->n_experts < 1 || c->n_experts > 4096) die_cfg("n_routed_experts out of range");
    if (c->topk < 1 || c->topk > MIMO_MAX_TOPK || c->topk > c->n_experts) die_cfg("num_experts_per_tok out of range");
    for (int k = 0; k < 2; k++) {
        if (c->heads[k] < 1 || c->kv_heads[k] < 1 || c->heads[k] % c->kv_heads[k]) die_cfg("attention heads must be a multiple of kv heads");
        if (c->heads[k] > 1024 || c->kv_heads[k] > 1024) die_cfg("too many attention heads");
        if (c->head_dim[k] < 2 || c->head_dim[k] > 1024 || c->v_dim[k] < 1 || c->v_dim[k] > 1024) die_cfg("head dims out of range");
        if (c->rope_dim[k] < 0 || c->rope_dim[k] > c->head_dim[k] || c->rope_dim[k] % 2) die_cfg("rotary dims must be even and fit the head");
    }
    if (c->chunks < 1 || c->chunks > MIMO_MAX_CHUNKS) die_cfg("num_key_value_heads (the qkv pre-shard count) out of range");
    for (int k = 0; k < 2; k++)
        if (c->heads[k] % c->chunks || c->kv_heads[k] % c->chunks) die_cfg("head counts do not divide into the qkv pre-shard chunks");
    int any_swa = 0, any_moe = 0;
    for (int i = 0; i < c->n_layers; i++) { any_swa |= c->swa[i]; any_moe |= c->moe[i]; }
    if (any_swa && (c->window < 1 || c->window > (1 << 20))) die_cfg("sliding_window out of range");
    for (int i = 0; i < c->n_layers; i++)
        if (!c->moe[i] && (c->dense_inter < 1 || c->dense_inter > (1 << 20))) die_cfg("intermediate_size out of range");
    if (!any_moe && keep <= 0) die_cfg("no MoE layer");
    if (!(c->eps > 0) || !(c->eps < 1)) die_cfg("layernorm_epsilon out of range");
    if (c->max_positions < 1) c->max_positions = 32768;
    json_free(root); free(arena); free(text);
}

/* ----------------------------------------------------------- dense weights ---- */

/* A dense matrix [O, I] in one of four forms. FP8 keeps the checkpoint bytes and
 * expands the 128x128 block grid to one scale per (row, column block), which is
 * what lets a row move (the qkv de-interleave) without touching its bytes. */
enum { DW_F32 = 0, DW_BF16 = 1, DW_FP8 = 2, DW_I8 = 3 };
typedef struct {
    int fmt, O, I, nblk;
    void *w;           /* f32 / bf16 / e4m3 / int8 */
    float *s;          /* FP8: [O][nblk]; I8: [O] */
    void *vk;          /* COLI_VULKAN: the device copy, uploaded at start-up */
    int vk_off;        /* its upload failed: this matrix stays on the CPU */
    /* COLI_VULKAN: how to read it back from disk (dw_reload), and 1 while the device
     * holds it alone (COLI_VK_DENSE_HOST, the host copy dropped) */
    char *vk_name;
    int vk_kind, vk_bits, vk_li, vk_gone;
} DW;

/* 0 (default): the checkpoint's own FP8 and BF16 bytes, exact.
 * 8: int8 per row, about 30% less RAM for the dense part, not exact.
 * 32: f32, exact and four times the RAM: the oracle's configuration. */
static int g_dense_bits;

static void matmul_bf16(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0; int i = 0;
#ifdef __AVX2__
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= I; i += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), bf16_decode8(w + i), acc);
            a = hsum256(acc);
#endif
            for (; i < I; i++) a += xs[i] * bf16_to_f32(w[i]);
            y[(int64_t)s * O + o] = a;
        }
    }
}

/* FP8 with a scale per (row, 128-column block): float inside a block, double
 * across blocks, the convention of quant.h's matmul_fp8. */
static void matmul_fp8_rows(float *y, const float *x, const uint8_t *q8, const float *rs,
                            int nblk, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q8 + (int64_t)o * I;
        const float *scl = rs + (int64_t)o * nblk;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            double a = 0;
            for (int b = 0; b < nblk; b++) {
                int base = b * 128, end = base + 128 < I ? base + 128 : I, i = base;
                float acc = 0;
#ifdef __AVX2__
                __m256 v = _mm256_setzero_ps();
                for (; i + 8 <= end; i += 8)
                    v = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), e4m3_decode8(w + i), v);
                acc = hsum256(v);
#endif
                for (; i < end; i++) acc += e4m3_decode(w[i]) * xs[i];
                a += (double)acc * scl[b];
            }
            y[(int64_t)s * O + o] = (float)a;
        }
    }
}

#ifdef COLI_VULKAN
/* Each form is one of the shader's formats as it is stored, so nothing is
 * converted on the way up:
 *   DW_F32   fmt 10, f32 [O, I], no scales
 *   DW_BF16  fmt 11, bf16 [O, I] (the low half of a word is the even column)
 *   DW_FP8   fmt 12, e4m3 [O, I] with the per-(row, 128-column block) scales
 *            rs[O][nblk] that dw_load_fp8 expanded: a group size of 128
 *   DW_I8    fmt 1, int8 [O, I] and one f32 scale per row after the sum
 * Every one is f32 weights times f32 activations, as on the CPU; what differs
 * is the order of the sums (the CPU's FP8 kernel adds its blocks in double). */
static int vk_main_thread(void) {
#ifdef _OPENMP
    return !omp_in_parallel();
#else
    return 1;
#endif
}

static int dw_vk_fmt(const DW *d, int *gs, const float **sc) {
    *gs = 0; *sc = NULL;
    switch (d->fmt) {
    case DW_F32:  return 10;
    case DW_BF16: return 11;
    case DW_FP8:  *gs = 128; *sc = d->s; return 12;
    case DW_I8:   *sc = d->s; return 1;
    }
    return -1;
}

static int g_mimo_vk_dev;   /* the device dw_upload sends a new matrix to: 1 for the chain's layers on
                            * COLI_VK_DEV2's (mimo_chain.h) */
static int dw_upload(DW *d) {
    int gs; const float *sc;
    int fmt = dw_vk_fmt(d, &gs, &sc);
    if (d->vk || d->vk_off || !d->w) return d->vk != NULL;
    if (fmt < 0 || !(g_mimo_vk_dev ? coli_vk_tensor_ensure2((ColiVkTensor **)&d->vk, d->w, sc, fmt, d->I, d->O, gs)
                                   : coli_vk_tensor_ensure((ColiVkTensor **)&d->vk, d->w, sc, fmt, d->I, d->O, gs))) d->vk_off = 1;
    return d->vk != NULL;
}

static int dw_matmul_vk(float *y, const float *x, int S, const DW *d) {
    if (!g_vk_ready || !(coli_vk_dense() || g_mimo_dho) || d->vk_off || !vk_main_thread()) return 0;
    DW *dev = (DW *)d;     /* the device copy is a cache inside a read-only matrix */
    if (!dw_upload(dev) || coli_vk_tensor_dev((ColiVkTensor *)dev->vk)) return 0;   /* the second device's: only its chain */
    int gs; const float *sc;
    int fmt = dw_vk_fmt(d, &gs, &sc);
    return coli_vk_matmul((ColiVkTensor **)&dev->vk, y, x, d->w, sc, fmt, S, d->I, d->O, gs);
}
static void dw_reload(DW *d);   /* below, beside the loaders it re-runs */
#endif

static void dw_matmul(float *y, const float *x, int S, const DW *d) {
#ifdef COLI_VULKAN
    if (dw_matmul_vk(y, x, S, d)) return;
    if (d->vk_gone) dw_reload((DW *)d);   /* the CPU needs a matrix the device held alone (a lost device) */
#endif
    switch (d->fmt) {
    case DW_F32:  matmul(y, x, (const float *)d->w, S, d->I, d->O); break;
    case DW_BF16: matmul_bf16(y, x, (const uint16_t *)d->w, S, d->I, d->O); break;
    case DW_FP8:  matmul_fp8_rows(y, x, (const uint8_t *)d->w, d->s, d->nblk, S, d->I, d->O); break;
    case DW_I8:   matmul_q(y, x, (const int8_t *)d->w, d->s, S, d->I, d->O); break;
    }
}

static st_tensor *need(shards *S, const char *name, int dtype, int O, int I) {
    st_tensor *t = st_find(S, name);
    if (!t) st_die_missing(S, name);
    int esz = st_dtype_esz(dtype);
    int64_t numel = I > 0 ? (int64_t)O * I : O;
    int rank_ok = I > 0 ? (t->rank == 2 && t->shape[0] == O && t->shape[1] == I)
                        : (t->rank == 1 && t->shape[0] == O);
    if (t->dtype != dtype || !rank_ok || t->nbytes != numel * esz) {
        fprintf(stderr, "[mimo] %s: %s %lld bytes, expected %s [%d%s%d] -- refusing (untrusted container)\n",
                name, st_dtype_name(t->dtype), (long long)t->nbytes, st_dtype_name(dtype), O,
                I > 0 ? ", " : "", I > 0 ? I : 0);
        exit(1);
    }
    return t;
}

/* f32 copy of a BF16 or F32 vector */
static float *load_vec(shards *S, const char *name, int n) {
    st_tensor *t = st_find(S, name);
    if (!t) st_die_missing(S, name);
    int dtype = t->dtype == 2 ? 2 : 0;
    need(S, name, dtype, n, 0);
    float *out = xmalloc((size_t)n * sizeof(float), name);
    st_read_f32(S, name, out, 1);
    return out;
}

/* Rows of an f32 matrix [O, I] into the configured dense form. Takes ownership. */
static void dw_from_f32(DW *d, float *w, int O, int I) {
    d->O = O; d->I = I;
    if (g_dense_bits == 8) {
        d->fmt = DW_I8;
        d->w = xmalloc((size_t)O * I, "int8 weights");
        d->s = xmalloc((size_t)O * sizeof(float), "int8 scales");
        quantize_rows(w, (int8_t *)d->w, d->s, O, I, 8);
        free(w);
    } else {
        d->fmt = DW_F32; d->w = w; d->s = NULL;
    }
}

#ifdef COLI_VULKAN
enum { DWK_NONE = 0, DWK_BF16, DWK_FP8, DWK_QKV };
/* What dw_reload needs to read the matrix back exactly as it was loaded: its name,
 * its loader, the MIMO_DENSE_BITS it was loaded under (the tower forces its own). */
static void dw_note(DW *d, const char *name, int kind) {
    free(d->vk_name);
    d->vk_name = strdup(name);
    if (!d->vk_name) { fprintf(stderr, "OOM weight name\n"); exit(1); }
    d->vk_kind = kind; d->vk_bits = g_dense_bits; d->vk_li = -1;
}
#define DW_NOTE(d, name, kind) dw_note(d, name, kind)
#else
#define DW_NOTE(d, name, kind) ((void)0)
#endif

static void dw_load_bf16(DW *d, shards *S, const char *name, int O, int I) {
    st_tensor *t = need(S, name, 0, O, I);
    DW_NOTE(d, name, DWK_BF16);
    if (g_dense_bits == 0) {
        d->fmt = DW_BF16; d->O = O; d->I = I; d->s = NULL;
        d->w = xmalloc((size_t)t->nbytes, name);
        st_read_raw_cap(S, name, d->w, t->nbytes, 1);
        return;
    }
    float *w = xmalloc((size_t)O * I * sizeof(float), name);
    st_read_f32(S, name, w, 1);
    dw_from_f32(d, w, O, I);
}

/* An FP8 [O, I] with its block grid; `row_block[r]` names the scale row of
 * stored row r (NULL: r / 128). `order[r]` names which stored row becomes row r
 * of the result (NULL: identity). */
static void dw_load_fp8(DW *d, shards *S, const char *name, int O, int I,
                        const int *row_block, int scale_rows, const int *order) {
    char sname[600];
    snprintf(sname, sizeof(sname), "%s_scale_inv", name);
    int nblk = (I + 127) / 128;
    if (scale_rows <= 0) scale_rows = (O + 127) / 128;
    st_tensor *t = need(S, name, 4, O, I);
    need(S, sname, 2, scale_rows, nblk);
    DW_NOTE(d, name, DWK_FP8);
    uint8_t *raw = xmalloc((size_t)t->nbytes, name);
    st_read_raw_cap(S, name, raw, t->nbytes, 1);
    float *grid = xmalloc((size_t)scale_rows * nblk * sizeof(float), sname);
    st_read_f32(S, sname, grid, 1);
    for (int64_t i = 0; i < (int64_t)scale_rows * nblk; i++)
        if (!isfinite(grid[i])) { fprintf(stderr, "[mimo] %s: non-finite scale -- refusing\n", sname); exit(1); }
    d->O = O; d->I = I; d->nblk = nblk;
    if (g_dense_bits == 0) {
        d->fmt = DW_FP8;
        uint8_t *w = xmalloc((size_t)O * I, name);
        float *rs = xmalloc((size_t)O * nblk * sizeof(float), sname);
        for (int r = 0; r < O; r++) {
            int src = order ? order[r] : r;
            int sb = row_block ? row_block[src] : src / 128;
            memcpy(w + (int64_t)r * I, raw + (int64_t)src * I, (size_t)I);
            memcpy(rs + (int64_t)r * nblk, grid + (int64_t)sb * nblk, (size_t)nblk * sizeof(float));
        }
        d->w = w; d->s = rs;
        free(raw); free(grid);
        return;
    }
    float *w = xmalloc((size_t)O * I * sizeof(float), name);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < O; r++) {
        int src = order ? order[r] : r;
        int sb = row_block ? row_block[src] : src / 128;
        const uint8_t *q = raw + (int64_t)src * I;
        float *dst = w + (int64_t)r * I;
        for (int i = 0; i < I; i++) dst[i] = e4m3_decode(q[i]) * grid[(int64_t)sb * nblk + i / 128];
    }
    free(raw); free(grid);
    dw_from_f32(d, w, O, I);
}

/* The fused qkv of layer `li`, read in its pre-sharded order and handed back as
 * [Q | K | V] rows. Two scale grids are legal (vLLM accepts the same two): one
 * tiled per chunk, ceil(rows_per_chunk / 128) rows each, or one continuous grid;
 * they coincide when a chunk is a whole number of blocks. */
static void load_qkv(DW *d, shards *S, const Cfg *c, int li) {
    int k = c->swa[li];
    int T = c->chunks, hd = c->head_dim[k], vd = c->v_dim[k];
    int q_rows = (c->heads[k] / T) * hd, k_rows = (c->kv_heads[k] / T) * hd,
        v_rows = (c->kv_heads[k] / T) * vd;
    int rpc = q_rows + k_rows + v_rows, O = T * rpc;
    int per_chunk = (rpc + 127) / 128, continuous = (O + 127) / 128;
    char name[512], sname[600];
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.qkv_proj.weight", li);
    snprintf(sname, sizeof(sname), "%s_scale_inv", name);
    st_tensor *st = st_find(S, sname);
    if (!st) st_die_missing(S, sname);
    int scale_rows = st->rank == 2 ? (int)st->shape[0] : -1;
    int *row_block = xmalloc((size_t)O * sizeof(int), "qkv scale rows");
    for (int r = 0; r < O; r++) {
        if (scale_rows == T * per_chunk) row_block[r] = (r / rpc) * per_chunk + (r % rpc) / 128;
        else if (scale_rows == continuous) row_block[r] = r / 128;
        else {
            fprintf(stderr, "[mimo] %s: %d scale rows fit neither the per-chunk grid (%d) nor a "
                            "continuous one (%d) -- refusing\n", sname, scale_rows, T * per_chunk, continuous);
            exit(1);
        }
    }
    int *order = xmalloc((size_t)O * sizeof(int), "qkv order");
    int w = 0;
    for (int ch = 0; ch < T; ch++) for (int r = 0; r < q_rows; r++) order[w++] = ch * rpc + r;
    for (int ch = 0; ch < T; ch++) for (int r = 0; r < k_rows; r++) order[w++] = ch * rpc + q_rows + r;
    for (int ch = 0; ch < T; ch++) for (int r = 0; r < v_rows; r++) order[w++] = ch * rpc + q_rows + k_rows + r;
    dw_load_fp8(d, S, name, O, c->hidden, row_block, scale_rows, order);
    free(row_block); free(order);
#ifdef COLI_VULKAN
    d->vk_kind = DWK_QKV; d->vk_li = li;   /* read back through this function: the de-interleave */
#endif
}

#ifdef COLI_VULKAN
/* A matrix whose host copy was dropped (COLI_VK_DENSE_HOST) and that the CPU needs
 * after all: read it back from the checkpoint with the loader and the MIMO_DENSE_BITS
 * it was first read with, the same bytes, and keep it from there on. */
static size_t dw_bytes(const DW *d);
static pthread_mutex_t g_dw_dho_mx = PTHREAD_MUTEX_INITIALIZER;
static shards *g_dw_dho_S;
static const Cfg *g_dw_dho_c;
static void dw_reload(DW *d) {
    pthread_mutex_lock(&g_dw_dho_mx);
    if (d->vk_gone) {
        DW t = {0};
        int saved = g_dense_bits;
        g_dense_bits = d->vk_bits;
        if (!g_dw_dho_S || !d->vk_name) { fprintf(stderr, "[VK] mimo: a dense matrix the device held alone cannot be read back\n"); exit(1); }
        switch (d->vk_kind) {
        case DWK_BF16: dw_load_bf16(&t, g_dw_dho_S, d->vk_name, d->O, d->I); break;
        case DWK_FP8:  dw_load_fp8(&t, g_dw_dho_S, d->vk_name, d->O, d->I, NULL, 0, NULL); break;
        case DWK_QKV:  load_qkv(&t, g_dw_dho_S, g_dw_dho_c, d->vk_li); break;
        default: fprintf(stderr, "[VK] mimo: %s has no loader to read it back\n", d->vk_name); exit(1);
        }
        g_dense_bits = saved;
        if (t.fmt != d->fmt || t.O != d->O || t.I != d->I) { fprintf(stderr, "[VK] mimo: %s came back in another form\n", d->vk_name); exit(1); }
        d->w = t.w; d->s = t.s; d->nblk = t.nblk;
        free(t.vk_name);
        d->vk_gone = 0;
        coli_vk_dense_host_reloaded(dw_bytes(d));
    }
    pthread_mutex_unlock(&g_dw_dho_mx);
}
#endif

/* ------------------------------------------------------------------ model ---- */

typedef struct { int eid; uint64_t used; uint8_t *base, *buf; } Slot;
typedef struct { Slot *s; int n, cap; int *by_expert; } LCache;
typedef struct { int fd[6]; int64_t off[6]; int contig; } ERef;

typedef struct {
    float *ln1, *ln2, *sink;
    DW qkv, o;
    DW gate, up, down;           /* dense MLP (layer 0) */
    float *router, *bias;        /* [E, H], [E] */
    /* KV: full layers [ctx][kvh][hd]; SWA layers a ring of `window` rows */
    float *K, *V;
    int *ring_pos;
    int rows;
} Layer;

/* One conversation's caches for a multiplexed serve (KV_SLOTS>1, serve_mux below):
 * every layer's K and V (a full layer's rows, a sliding window's ring and its
 * positions), where it stands, and the record of the tokens it holds. The Model's
 * layers hold the conversation a prefill runs on (mimo_seq_swap trades it for a parked
 * one); a multiplexed decode step parks them all and reads each row's from its MimoRow. */
typedef struct {
    float *K[MIMO_MAX_LAYERS], *V[MIMO_MAX_LAYERS];
    int *ring_pos[MIMO_MAX_LAYERS];
    int pos;
    kv_prefix kvp;
} MimoSeq;
typedef struct { MimoSeq *seq; int pos; } MimoRow;
static int g_mimo_mux_slots = 1;   /* KV_SLOTS: the conversations a serve decodes at once */

typedef struct {
    Cfg c;
    shards S;
    Layer L[MIMO_MAX_LAYERS];
    uint16_t *embed;             /* bf16 [V, H], rows widened on use */
    DW head;
    float *norm;
    ERef *eref;
    int64_t part[6], e_bytes;    /* down, down scale, gate, gate scale, up, up scale */
    LCache cache[MIMO_MAX_LAYERS];
    int ctx, pos;
    kv_prefix kvp;
    uint64_t clock, hits, miss, bytes_read, forwards;
    double t_disk, t_expert, t_attn;
    int idot, direct;
    uint8_t **ehit;              /* [layer][expert] routed this turn, for HITS */
#ifdef COLI_VULKAN
    void *vkchain;               /* the dense chain's device state (mimo_chain.h), NULL until it runs */
    void *vkchain2;              /* its layers on COLI_VK_DEV2's device, after the primary's (mimo_chain.h) */
#endif
    /* A multiplexed decode step (forward_rows): row t is the token at mux_rows[t].pos
     * of mux_rows[t].seq; NULL in every other forward. */
    const MimoRow *mux_rows;
} Model;

static void expert_table_init(Model *m) {
    Cfg *c = &m->c;
    int H = c->hidden, MI = c->moe_inter;
    m->part[0] = (int64_t)H * (MI / 2);  m->part[1] = (int64_t)H * (MI / 32);
    m->part[2] = (int64_t)MI * (H / 2);  m->part[3] = (int64_t)MI * (H / 32);
    m->part[4] = m->part[2];             m->part[5] = m->part[3];
    m->e_bytes = 0;
    for (int k = 0; k < 6; k++) m->e_bytes += m->part[k];
    m->eref = xcalloc((size_t)c->n_layers * c->n_experts, sizeof(ERef), "expert table");
    static const char *mat[3] = {"down_proj", "gate_proj", "up_proj"};
    int missing = 0, split = 0;
    for (int li = 0; li < c->n_layers; li++) {
        if (!c->moe[li]) continue;
        for (int e = 0; e < c->n_experts; e++) {
            ERef *er = &m->eref[(int64_t)li * c->n_experts + e];
            for (int k = 0; k < 6; k++) {
                char nm[512];
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.%s.%s", li, e,
                         mat[k / 2], (k & 1) ? "weight_scale" : "weight");
                st_tensor *t = st_find(&m->S, nm);
                if (!t) { missing++; er->fd[k] = -1; continue; }
                if (t->dtype != 3 || t->nbytes != m->part[k]) {
                    fprintf(stderr, "[mimo] %s: %s %lld bytes, expected U8 %lld -- refusing "
                                    "(untrusted container)\n", nm, st_dtype_name(t->dtype),
                            (long long)t->nbytes, (long long)m->part[k]);
                    exit(1);
                }
                er->fd[k] = t->fd; er->off[k] = t->off;
            }
            /* The release stores an expert's six tensors back to back in this
             * order (down, down scale, gate, gate scale, up, up scale): one read. */
            er->contig = er->fd[0] >= 0;
            for (int k = 0; k < 5 && er->contig; k++)
                if (er->fd[k] != er->fd[k + 1] || er->off[k] + m->part[k] != er->off[k + 1]) er->contig = 0;
            split += !er->contig;
        }
    }
    if (missing) fprintf(stderr, "[mimo] WARNING: %d expert tensors missing (incomplete download?) -- touching one aborts\n", missing);
    if (split) fprintf(stderr, "[mimo] %d experts are not contiguous on disk: six reads each\n", split);
}

static void cache_init(Model *m, int cap) {
    Cfg *c = &m->c;
    if (cap < c->topk) cap = c->topk;
    if (cap > c->n_experts) cap = c->n_experts;
    for (int li = 0; li < c->n_layers; li++) {
        if (!c->moe[li]) continue;
        LCache *lc = &m->cache[li];
        lc->cap = cap; lc->n = 0;
        lc->s = xcalloc((size_t)cap, sizeof(Slot), "expert slots");
        lc->by_expert = xmalloc((size_t)c->n_experts * sizeof(int), "expert index");
        for (int e = 0; e < c->n_experts; e++) lc->by_expert[e] = -1;
        for (int k = 0; k < cap; k++) lc->s[k].eid = -1;
    }
}

static void expert_read(Model *m, int li, int eid, Slot *s) {
    if (!s->base && posix_memalign((void **)&s->base, 4096, (size_t)m->e_bytes + 8192)) {
        fprintf(stderr, "[mimo] out of memory for an expert slot\n"); exit(1);
    }
    ERef *er = &m->eref[(int64_t)li * m->c.n_experts + eid];
    if (er->fd[0] < 0) { fprintf(stderr, "[mimo] expert L%d E%d missing on disk\n", li, eid); exit(1); }
    if (er->contig) {
        int dfd = m->direct ? st_direct_fd(&m->S, er->fd[0]) : -1;
        if (dfd >= 0) {
            int64_t a0 = er->off[0] & ~4095LL, pad = er->off[0] - a0, want = pad + m->e_bytes;
            int64_t dlen = (want + 4095) & ~4095LL;
            struct stat sb;
            if (fstat(dfd, &sb) == 0 && a0 + dlen > sb.st_size) dlen = (sb.st_size - a0) & ~4095LL;
            if (dlen > 0) st_pread_full(dfd, s->base, dlen, a0, "pread expert direct");
            if (dlen < want) st_pread_full(er->fd[0], s->base + dlen, want - dlen, a0 + dlen, "pread expert tail");
            s->buf = s->base + pad;
        } else {
            st_pread_full(er->fd[0], s->base, m->e_bytes, er->off[0], "pread expert");
            s->buf = s->base;
        }
    } else {
        uint8_t *dst = s->base;
        for (int k = 0; k < 6; k++) {
            if (er->fd[k] < 0) { fprintf(stderr, "[mimo] expert L%d E%d tensor %d missing\n", li, eid, k); exit(1); }
            st_pread_full(er->fd[k], dst, m->part[k], er->off[k], "pread expert");
            dst += m->part[k];
        }
        s->buf = s->base;
    }
}

/* Make the experts `want[0..n)` of layer li resident, n <= cap, and return their
 * slots in `out`. Hits keep their slot; misses take the least recently used
 * slots not in `want`, and are read in parallel. */
static void experts_ensure(Model *m, int li, const int *want, int n, Slot **out) {
    LCache *lc = &m->cache[li];
    int miss_idx[4096], nmiss = 0;
    uint64_t stamp = ++m->clock;
    for (int j = 0; j < n; j++) {
        int si = lc->by_expert[want[j]];
        if (si >= 0) { out[j] = &lc->s[si]; out[j]->used = stamp; m->hits++; }
        else { out[j] = NULL; miss_idx[nmiss++] = j; }
    }
    for (int k = 0; k < nmiss; k++) {
        int best = -1;
        if (lc->n < lc->cap) best = lc->n++;
        else {   /* the least recently used, and before it one the Vulkan tier holds (vkt_ram_first) */
            uint64_t oldest = UINT64_MAX, dev_oldest = UINT64_MAX;
            int dev = -1;
            for (int s = 0; s < lc->n; s++) {
                if (lc->s[s].used >= stamp) continue;
                if (lc->s[s].eid >= 0 && vkt_ram_first(li, lc->s[s].eid)) {
                    if (lc->s[s].used < dev_oldest) { dev_oldest = lc->s[s].used; dev = s; }
                } else if (lc->s[s].used < oldest) { oldest = lc->s[s].used; best = s; }
            }
            if (dev >= 0) { best = dev; vkt_ram_gave(); }
        }
        if (best < 0) { fprintf(stderr, "[mimo] expert cache smaller than one routing step\n"); exit(1); }
        Slot *s = &lc->s[best];
        if (s->eid >= 0) lc->by_expert[s->eid] = -1;
        s->eid = want[miss_idx[k]];
        s->used = stamp;
        lc->by_expert[s->eid] = best;
        out[miss_idx[k]] = s;
    }
    if (!nmiss) return;
    double t0 = now_s();
    int threads = env_int("MIMO_READ_THREADS", 8);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int k = 0; k < nmiss; k++) {
        Slot *s = out[miss_idx[k]];
        expert_read(m, li, s->eid, s);
    }
    m->miss += (uint64_t)nmiss;
    m->bytes_read += (uint64_t)nmiss * (uint64_t)m->e_bytes;
    m->t_disk += now_s() - t0;
}

static void kv_alloc(Model *m, int ctx) {
    Cfg *c = &m->c;
    m->ctx = ctx;
    for (int li = 0; li < c->n_layers; li++) {
        Layer *l = &m->L[li];
        int k = c->swa[li];
        l->rows = k ? (c->window < ctx ? c->window : ctx) : ctx;
        free(l->K); free(l->V); free(l->ring_pos);
        l->K = xmalloc((size_t)l->rows * c->kv_heads[k] * c->head_dim[k] * sizeof(float), "K cache");
        l->V = xmalloc((size_t)l->rows * c->kv_heads[k] * c->v_dim[k] * sizeof(float), "V cache");
        l->ring_pos = xmalloc((size_t)l->rows * sizeof(int), "ring positions");
        for (int r = 0; r < l->rows; r++) l->ring_pos[r] = -1;
    }
    kv_prefix_alloc(&m->kvp, ctx);
}

static void model_reset(Model *m) {
    Cfg *c = &m->c;
    for (int li = 0; li < c->n_layers; li++)
        for (int r = 0; r < m->L[li].rows; r++) m->L[li].ring_pos[r] = -1;
    m->pos = 0;
    kv_prefix_clear(&m->kvp);
}

static void model_load(Model *m, const char *dir, int cap) {
    Cfg *c = &m->c;
    load_cfg(c, dir);
    st_init_multi(&m->S, dir, getenv("MIMO_DIRS"));
    g_dense_bits = env_int("MIMO_DENSE_BITS", 0);
    if (g_dense_bits != 0 && g_dense_bits != 8 && g_dense_bits != 32) {
        fprintf(stderr, "[mimo] MIMO_DENSE_BITS must be 0 (native FP8/BF16), 8 or 32\n"); exit(1);
    }
    m->idot = env_int("MIMO_IDOT", 0);
    m->direct = env_int("MIMO_DIRECT", 1);
    int H = c->hidden;
    need(&m->S, "model.embed_tokens.weight", 0, c->vocab, H);
    m->embed = xmalloc((size_t)c->vocab * H * sizeof(uint16_t), "embeddings");
    st_read_raw_cap(&m->S, "model.embed_tokens.weight", m->embed, (int64_t)c->vocab * H * 2, 1);
    m->norm = load_vec(&m->S, "model.norm.weight", H);
    dw_load_bf16(&m->head, &m->S, "lm_head.weight", c->vocab, H);
    for (int li = 0; li < c->n_layers; li++) {
        Layer *l = &m->L[li];
        int k = c->swa[li];
        char nm[512];
        snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", li);
        l->ln1 = load_vec(&m->S, nm, H);
        snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", li);
        l->ln2 = load_vec(&m->S, nm, H);
        load_qkv(&l->qkv, &m->S, c, li);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", li);
        dw_load_bf16(&l->o, &m->S, nm, H, c->heads[k] * c->v_dim[k]);
        if (c->sink[k]) {
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.attention_sink_bias", li);
            l->sink = load_vec(&m->S, nm, c->heads[k]);
        }
        if (!c->moe[li]) {
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", li);
            dw_load_fp8(&l->gate, &m->S, nm, c->dense_inter, H, NULL, 0, NULL);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", li);
            dw_load_fp8(&l->up, &m->S, nm, c->dense_inter, H, NULL, 0, NULL);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", li);
            dw_load_fp8(&l->down, &m->S, nm, H, c->dense_inter, NULL, 0, NULL);
        } else {
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.weight", li);
            need(&m->S, nm, 0, c->n_experts, H);
            l->router = xmalloc((size_t)c->n_experts * H * sizeof(float), nm);
            st_read_f32(&m->S, nm, l->router, 1);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.e_score_correction_bias", li);
            l->bias = load_vec(&m->S, nm, c->n_experts);
        }
    }
    expert_table_init(m);
    cache_init(m, cap);
    /* coli --ctx arrives as CTX (the family's context_env); MIMO_CTX wins over it */
    const char *ctx_text = getenv("MIMO_CTX");
    if (!ctx_text || !*ctx_text) ctx_text = getenv("CTX");
    int ctx = (ctx_text && *ctx_text) ? atoi(ctx_text) : 8192;
    if (ctx > c->max_positions) ctx = c->max_positions;
    if (ctx < 16) ctx = 16;
    kv_alloc(m, ctx);
    m->ehit = xmalloc((size_t)c->n_layers * sizeof(uint8_t *), "hits");
    for (int li = 0; li < c->n_layers; li++) m->ehit[li] = xcalloc((size_t)c->n_experts, 1, "hits row");
}

/* ------------------------------------------------------------ the forward ---- */

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * r);
}

/* rotate-half RoPE on the first `rd` dims of a head, as the reference builds it:
 * inv_freq and the angle in float32, cos/sin of the float angle */
static void rope(float *v, int rd, float theta, int pos) {
    int half = rd / 2;
    for (int i = 0; i < half; i++) {
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rd);
        float ang = inv * (float)pos;
        float cs = cosf(ang), sn = sinf(ang);
        float a = v[i], b = v[i + half];
        v[i] = a * cs - b * sn;
        v[i + half] = b * cs + a * sn;
    }
}

/* Attention of `n` rows at positions p0.. of layer li. `xn` is the normed input
 * [n, H], `out` receives o_proj's output [n, H]. */
/* A multiplexed decode step's attention (KV_SLOTS, m->mux_rows): row t is the next
 * token of its own conversation, at mux_rows[t].pos, over that conversation's cache --
 * a full layer's rows, or a sliding window's ring. Each row's numbers are attention()'s
 * for a block of that one token; the projections run once over the n rows. */
static void attention_rows(Model *m, int li, const float *xn, int n, float *out) {
    Cfg *c = &m->c;
    Layer *l = &m->L[li];
    const MimoRow *mr = m->mux_rows;
    int k = c->swa[li];
    int nh = c->heads[k], kvh = c->kv_heads[k], hd = c->head_dim[k], vd = c->v_dim[k];
    int qd = nh * hd, kd = kvh * hd, vdd = kvh * vd, rows = qd + kd + vdd;
    double t0 = now_s();
    float *qkv = xmalloc((size_t)n * rows * sizeof(float), "qkv");
    dw_matmul(qkv, xn, n, &l->qkv);
    for (int t = 0; t < n; t++) {
        float *q = qkv + (size_t)t * rows, *kk = q + qd, *vv = kk + kd;
        for (int h = 0; h < nh; h++) rope(q + h * hd, c->rope_dim[k], c->theta[k], mr[t].pos);
        for (int h = 0; h < kvh; h++) rope(kk + h * hd, c->rope_dim[k], c->theta[k], mr[t].pos);
        for (int i = 0; i < vdd; i++) vv[i] *= c->v_scale;
    }
    int window = k ? c->window : 0, most = 1;
    const float **Kc = xmalloc((size_t)n * sizeof *Kc, "row keys"), **Vc = xmalloc((size_t)n * sizeof *Vc, "row values");
    int *lo = xmalloc((size_t)n * sizeof(int), "row first keys");
    for (int t = 0; t < n; t++) {
        MimoSeq *q = mr[t].seq;
        int p = mr[t].pos;
        const float *src = qkv + (size_t)t * rows + qd;
        if (k) {   /* the window's keys before p from the ring, then this token's */
            int l0 = p - (window - 1) > 0 ? p - (window - 1) : 0;
            float *kc = xmalloc((size_t)(p + 1 - l0) * kd * sizeof(float), "window keys");
            float *vc = xmalloc((size_t)(p + 1 - l0) * vdd * sizeof(float), "window values");
            for (int pp = l0; pp < p; pp++) {
                int slot = pp % l->rows;
                if (q->ring_pos[li][slot] != pp) { fprintf(stderr, "[mimo] window ring lost position %d\n", pp); exit(1); }
                memcpy(kc + (size_t)(pp - l0) * kd, q->K[li] + (size_t)slot * kd, (size_t)kd * sizeof(float));
                memcpy(vc + (size_t)(pp - l0) * vdd, q->V[li] + (size_t)slot * vdd, (size_t)vdd * sizeof(float));
            }
            memcpy(kc + (size_t)(p - l0) * kd, src, (size_t)kd * sizeof(float));
            memcpy(vc + (size_t)(p - l0) * vdd, src + kd, (size_t)vdd * sizeof(float));
            Kc[t] = kc; Vc[t] = vc; lo[t] = l0;
        } else {   /* the full cache, this token written in first */
            memcpy(q->K[li] + (size_t)p * kd, src, (size_t)kd * sizeof(float));
            memcpy(q->V[li] + (size_t)p * vdd, src + kd, (size_t)vdd * sizeof(float));
            Kc[t] = q->K[li]; Vc[t] = q->V[li]; lo[t] = 0;
        }
        if (p + 1 - lo[t] > most) most = p + 1 - lo[t];
    }
    float *ctxv = xmalloc((size_t)n * nh * vd * sizeof(float), "attention output");
    float scale = 1.0f / sqrtf((float)hd);
    int group = nh / kvh;
    #pragma omp parallel
    {
        float *score = xmalloc((size_t)most * sizeof(float), "scores");
        #pragma omp for collapse(2) schedule(static)
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < nh; h++) {
                int p = mr[t].pos, kh = h / group, first = lo[t];
                const float *q = qkv + (size_t)t * rows + (size_t)h * hd;
                float best = -INFINITY;
                for (int j = first; j <= p; j++) {
                    const float *kr = Kc[t] + (size_t)(j - lo[t]) * kd + (size_t)kh * hd;
                    float s = 0;
                    for (int i = 0; i < hd; i++) s += q[i] * kr[i];
                    s *= scale;
                    score[j - first] = s;
                    if (s > best) best = s;
                }
                float sink = l->sink ? l->sink[h] : -INFINITY;
                if (sink > best) best = sink;
                float den = l->sink ? expf(sink - best) : 0.0f;
                for (int j = first; j <= p; j++) { score[j - first] = expf(score[j - first] - best); den += score[j - first]; }
                float *o = ctxv + ((size_t)t * nh + h) * vd;
                for (int i = 0; i < vd; i++) o[i] = 0;
                for (int j = first; j <= p; j++) {
                    const float *vr = Vc[t] + (size_t)(j - lo[t]) * vdd + (size_t)kh * vd;
                    float w = score[j - first] / den;
                    for (int i = 0; i < vd; i++) o[i] += w * vr[i];
                }
            }
        }
        free(score);
    }
    if (k)   /* the ring keeps each conversation's last `window` positions */
        for (int t = 0; t < n; t++) {
            MimoSeq *q = mr[t].seq;
            int p = mr[t].pos, slot = p % l->rows;
            const float *src = qkv + (size_t)t * rows + qd;
            memcpy(q->K[li] + (size_t)slot * kd, src, (size_t)kd * sizeof(float));
            memcpy(q->V[li] + (size_t)slot * vdd, src + kd, (size_t)vdd * sizeof(float));
            q->ring_pos[li][slot] = p;
            free((void *)Kc[t]); free((void *)Vc[t]);
        }
    dw_matmul(out, ctxv, n, &l->o);
    free(ctxv); free(qkv); free(Kc); free(Vc); free(lo);
    m->t_attn += now_s() - t0;
}

static void attention(Model *m, int li, const float *xn, int n, int p0, float *out) {
    if (m->mux_rows) { attention_rows(m, li, xn, n, out); return; }
    Cfg *c = &m->c;
    Layer *l = &m->L[li];
    int k = c->swa[li];
    int nh = c->heads[k], kvh = c->kv_heads[k], hd = c->head_dim[k], vd = c->v_dim[k];
    int qd = nh * hd, kd = kvh * hd, vdd = kvh * vd, rows = qd + kd + vdd;
    double t0 = now_s();
    float *qkv = xmalloc((size_t)n * rows * sizeof(float), "qkv");
    dw_matmul(qkv, xn, n, &l->qkv);
    for (int t = 0; t < n; t++) {
        float *q = qkv + (size_t)t * rows, *kk = q + qd, *vv = kk + kd;
        for (int h = 0; h < nh; h++) rope(q + h * hd, c->rope_dim[k], c->theta[k], p0 + t);
        for (int h = 0; h < kvh; h++) rope(kk + h * hd, c->rope_dim[k], c->theta[k], p0 + t);
        for (int i = 0; i < vdd; i++) vv[i] *= c->v_scale;
    }
    /* Keys this block can see: [lo, p0 + n). For a sliding window the ring still
     * holds the ones before p0; for full attention the cache holds everything,
     * and the block is written in first. */
    int window = k ? c->window : 0;
    int lo = k ? (p0 - (window - 1) > 0 ? p0 - (window - 1) : 0) : 0;
    int nkeys = p0 + n - lo;
    float *Kc = NULL, *Vc = NULL;
    if (k) {
        Kc = xmalloc((size_t)nkeys * kd * sizeof(float), "window keys");
        Vc = xmalloc((size_t)nkeys * vdd * sizeof(float), "window values");
        for (int p = lo; p < p0; p++) {
            int slot = p % l->rows;
            if (l->ring_pos[slot] != p) { fprintf(stderr, "[mimo] window ring lost position %d\n", p); exit(1); }
            memcpy(Kc + (size_t)(p - lo) * kd, l->K + (size_t)slot * kd, (size_t)kd * sizeof(float));
            memcpy(Vc + (size_t)(p - lo) * vdd, l->V + (size_t)slot * vdd, (size_t)vdd * sizeof(float));
        }
        for (int t = 0; t < n; t++) {
            const float *src = qkv + (size_t)t * rows + qd;
            memcpy(Kc + (size_t)(p0 + t - lo) * kd, src, (size_t)kd * sizeof(float));
            memcpy(Vc + (size_t)(p0 + t - lo) * vdd, src + kd, (size_t)vdd * sizeof(float));
        }
    } else {
        for (int t = 0; t < n; t++) {
            const float *src = qkv + (size_t)t * rows + qd;
            memcpy(l->K + (size_t)(p0 + t) * kd, src, (size_t)kd * sizeof(float));
            memcpy(l->V + (size_t)(p0 + t) * vdd, src + kd, (size_t)vdd * sizeof(float));
        }
        Kc = l->K; Vc = l->V;
    }
    float *ctxv = xmalloc((size_t)n * nh * vd * sizeof(float), "attention output");
    float scale = 1.0f / sqrtf((float)hd);
    int group = nh / kvh;
    #pragma omp parallel
    {
        float *score = xmalloc((size_t)nkeys * sizeof(float), "scores");
        #pragma omp for collapse(2) schedule(static)
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < nh; h++) {
                int p = p0 + t, kh = h / group;
                int first = window ? (p - (window - 1) > lo ? p - (window - 1) : lo) : 0;
                const float *q = qkv + (size_t)t * rows + (size_t)h * hd;
                float best = -INFINITY;
                for (int j = first; j <= p; j++) {
                    const float *kr = Kc + (size_t)(j - lo) * kd + (size_t)kh * hd;
                    float s = 0;
                    for (int i = 0; i < hd; i++) s += q[i] * kr[i];
                    s *= scale;
                    score[j - first] = s;
                    if (s > best) best = s;
                }
                float sink = l->sink ? l->sink[h] : -INFINITY;
                if (sink > best) best = sink;
                float den = l->sink ? expf(sink - best) : 0.0f;
                for (int j = first; j <= p; j++) { score[j - first] = expf(score[j - first] - best); den += score[j - first]; }
                float *o = ctxv + ((size_t)t * nh + h) * vd;
                for (int i = 0; i < vd; i++) o[i] = 0;
                for (int j = first; j <= p; j++) {
                    const float *vr = Vc + (size_t)(j - lo) * vdd + (size_t)kh * vd;
                    float w = score[j - first] / den;
                    for (int i = 0; i < vd; i++) o[i] += w * vr[i];
                }
            }
        }
        free(score);
    }
    if (k) {
        /* the ring keeps the last `window` positions of the block */
        int from = n > l->rows ? n - l->rows : 0;
        for (int t = from; t < n; t++) {
            int p = p0 + t, slot = p % l->rows;
            const float *src = qkv + (size_t)t * rows + qd;
            memcpy(l->K + (size_t)slot * kd, src, (size_t)kd * sizeof(float));
            memcpy(l->V + (size_t)slot * vdd, src + kd, (size_t)vdd * sizeof(float));
            l->ring_pos[slot] = p;
        }
        free(Kc); free(Vc);
    }
    dw_matmul(out, ctxv, n, &l->o);
    free(ctxv); free(qkv);
    m->t_attn += now_s() - t0;
}

static void dense_mlp(Model *m, Layer *l, const float *xn, int n, float *out) {
    int F = l->gate.O;
    float *g = xmalloc((size_t)n * F * sizeof(float), "mlp gate");
    float *u = xmalloc((size_t)n * F * sizeof(float), "mlp up");
    dw_matmul(g, xn, n, &l->gate);
    dw_matmul(u, xn, n, &l->up);
    for (size_t i = 0; i < (size_t)n * F; i++) g[i] = silu(g[i]) * u[i];
    dw_matmul(out, g, n, &l->down);
    free(g); free(u);
    (void)m;
}

/* An expert's bytes as the shared Vulkan tier reads them (buf: down, its
 * scales, gate, its scales, up, its scales; one e8m0 byte per 32 columns). */
static VktExpertSrc mimo_vk_src(const Model *m, const uint8_t *buf) {
    const uint8_t *dq = buf, *ds = dq + m->part[0], *gq = ds + m->part[1], *gs = gq + m->part[2],
                  *uq = gs + m->part[3], *us = uq + m->part[4];
    return (VktExpertSrc){gq, uq, dq, gs, us, ds};
}
/* The tier's streaming (a big prompt chunk's cold experts on the device): the experts'
 * bytes through the layer cache as moe_cpu gets them (experts_ensure: a group at once,
 * read in parallel, up to the cache's capacity), valid until the next ensure. */
static int mimo_vk_load_batch(void *ctx, int li, const int *e, int n, VktExpertSrc *srcs, void **h) {
    Model *m = ctx;
    int cap = m->cache[li].cap;
    if (n > cap) n = cap;
    if (n < 1 || n > 64) return 0;
    Slot *sl[64];
    experts_ensure(m, li, e, n, sl);
    for (int i = 0; i < n; i++) { srcs[i] = mimo_vk_src(m, sl[i]->buf); h[i] = sl[i]; }
    return n;
}
static int mimo_vk_load(void *ctx, int li, int e, VktExpertSrc *src, void **h) {
    return mimo_vk_load_batch(ctx, li, &e, 1, src, h) == 1;
}
static void mimo_vk_release(void *ctx, void *h) { (void)ctx; (void)h; }

/* The CPU's experts of one MoE step: every (row, choice) pair i of sel -- or,
 * with a mask, those whose mask[i] is mval -- runs its expert, and its output
 * lands in y at row pair[i], numbered on from `total`; returns the new count.
 *
 * Each group of resident experts is computed in two parallel passes -- gate and
 * up of every expert, then down of every expert -- split in row blocks, instead
 * of three OpenMP regions per expert (a thousand a token on Flash, and every
 * barrier costs a scheduling round on a busy machine). The kernel inside a
 * task is the same matmul_mxfp4 on the same rows, so the numbers do not move.
 *
 * Every (row, choice) pair has its own slot in y, across all the groups of the
 * block, so the weighted sum can wait for the last group and run row by row in
 * that row's own routing order. With `note`, every expert computed here is
 * offered to the Vulkan tier while its bytes are in RAM (vkt_note; nothing
 * without the tier). */
static int moe_cpu(Model *m, int li, const float *xn, int n, const int *sel,
                   const uint8_t *mask, int mval, int *pair, float *y, int total, int note) {
    Cfg *c = &m->c;
    int H = c->hidden, E = c->n_experts, K = c->topk, MI = c->moe_inter;
#define MIMO_WANT(i) (!mask || (mask[i] != 0) == mval)
    /* the union of chosen experts, in first-use order */
    int *uni = xmalloc((size_t)E * sizeof(int), "expert union");
    int *seen = xcalloc((size_t)E, sizeof(int), "expert seen");
    int nu = 0;
    for (int i = 0; i < n * K; i++) if (MIMO_WANT(i) && !seen[sel[i]]) { seen[sel[i]] = 1; uni[nu++] = sel[i]; }
    free(seen);
    int cap = m->cache[li].cap;
    int *row = xmalloc((size_t)n * K * sizeof(int), "expert rows index");
    int *first = xmalloc((size_t)(cap + 1) * sizeof(int), "expert row offsets");
    float *xg = xmalloc((size_t)n * K * H * sizeof(float), "expert inputs");
    float *g = xmalloc((size_t)n * K * MI * sizeof(float), "expert gate");
    float *u = xmalloc((size_t)n * K * MI * sizeof(float), "expert up");
    void (*mm)(float *, const float *, const uint8_t *, const uint8_t *, int, int, int)
        = m->idot ? matmul_mxfp4_i8 : matmul_mxfp4;
    const int BLOCK_ROWS = 256;
    Slot *slots[4096];
    int base = total;                /* y rows before this call are someone else's */
    for (int b = 0; b < nu; b += cap) {
        int nb = nu - b < cap ? nu - b : cap;
        experts_ensure(m, li, uni + b, nb, slots);
        double t0 = now_s();
        int from = total;
        for (int j = 0; j < nb; j++) {
            int e = uni[b + j];
            first[j] = total - base;
            for (int t = 0; t < n; t++)
                for (int q = 0; q < K; q++)
                    if (sel[(size_t)t * K + q] == e && MIMO_WANT((size_t)t * K + q)) {
                        row[total - base] = t;
                        pair[(size_t)t * K + q] = total;
                        total++;
                    }
        }
        first[nb] = total - base;
        for (int i = from; i < total; i++) memcpy(xg + (size_t)(i - base) * H, xn + (size_t)row[i - base] * H, (size_t)H * sizeof(float));
        int up_blocks = (MI + BLOCK_ROWS - 1) / BLOCK_ROWS, down_blocks = (H + BLOCK_ROWS - 1) / BLOCK_ROWS;
        int rb_mi = MI / 2, gb_mi = MI / 32, rb_h = H / 2, gb_h = H / 32;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int task = 0; task < nb * 2 * up_blocks; task++) {
            int j = task / (2 * up_blocks), which = (task / up_blocks) % 2, blk = task % up_blocks;
            int r = first[j + 1] - first[j];
            if (!r) continue;
            int o0 = blk * BLOCK_ROWS, rows = MI - o0 < BLOCK_ROWS ? MI - o0 : BLOCK_ROWS;
            const uint8_t *buf = slots[j]->buf;
            const uint8_t *q = buf + m->part[0] + m->part[1] + (which ? m->part[2] + m->part[3] : 0);
            const uint8_t *sc = q + m->part[2];
            float *dst = (which ? u : g) + (size_t)first[j] * MI;
            /* rows [o0, o0+rows) of this expert's gate or up, for its r inputs; the
             * kernel writes [r, rows] contiguously, so it goes through a scratch */
            float tmp[4 * 256];
            float *scratch = r * rows <= (int)(sizeof(tmp) / sizeof(tmp[0])) ? tmp
                           : xmalloc((size_t)r * rows * sizeof(float), "expert scratch");
            mm(scratch, xg + (size_t)first[j] * H, q + (size_t)o0 * rb_h, sc + (size_t)o0 * gb_h, r, H, rows);
            for (int i = 0; i < r; i++)
                memcpy(dst + (size_t)i * MI + o0, scratch + (size_t)i * rows, (size_t)rows * sizeof(float));
            if (scratch != tmp) free(scratch);
        }
        for (size_t i = (size_t)(from - base) * MI; i < (size_t)(total - base) * MI; i++) g[i] = silu(g[i]) * u[i];
        #pragma omp parallel for schedule(dynamic, 1)
        for (int task = 0; task < nb * down_blocks; task++) {
            int j = task / down_blocks, blk = task % down_blocks;
            int r = first[j + 1] - first[j];
            if (!r) continue;
            int o0 = blk * BLOCK_ROWS, rows = H - o0 < BLOCK_ROWS ? H - o0 : BLOCK_ROWS;
            const uint8_t *dq = slots[j]->buf, *ds = dq + m->part[0];
            float tmp[4 * 256];
            float *scratch = r * rows <= (int)(sizeof(tmp) / sizeof(tmp[0])) ? tmp
                           : xmalloc((size_t)r * rows * sizeof(float), "expert scratch");
            mm(scratch, g + (size_t)first[j] * MI, dq + (size_t)o0 * rb_mi, ds + (size_t)o0 * gb_mi, r, MI, rows);
            float *dst = y + (size_t)(base + first[j]) * H;
            for (int i = 0; i < r; i++)
                memcpy(dst + (size_t)i * H + o0, scratch + (size_t)i * rows, (size_t)rows * sizeof(float));
            if (scratch != tmp) free(scratch);
        }
        /* the group's bytes are still in RAM: the tier may promote them */
        if (note && vkt_ready())
            for (int j = 0; j < nb; j++) { VktExpertSrc vs = mimo_vk_src(m, slots[j]->buf); vkt_note(li, uni[b + j], &vs); }
        m->t_expert += now_s() - t0;
    }
#undef MIMO_WANT
    free(first); free(xg); free(g); free(u); free(row); free(uni);
    return total;
}

/* Route every row, then run each chosen expert once over the rows that chose it.
 * `out` [n, H] receives the weighted sum.
 *
 * With the Vulkan tier on, vkt_issue first hands the pairs whose expert is
 * resident on the device to one asynchronous batch; moe_cpu reads and computes
 * the others meanwhile (an expert the device took entirely is never read from
 * disk), and after the join every pair joins the sum in its row's routing order,
 * the device's and the CPU's alike: the order never depends on which experts were
 * resident. The experts the CPU computed are offered to the tier after the join
 * (those still in the RAM cache), not while the batch runs: a promotion that
 * displaces a resident then frees it at once, and its upload starts on a pool
 * with room instead of waiting on the uploader thread for the join's free. */
#define MIMO_VK_ROWS 64              /* rows whose pairs one device batch carries at most */
static void moe(Model *m, int li, const float *xn, int n, float *out) {
    Cfg *c = &m->c;
    Layer *l = &m->L[li];
    int H = c->hidden, E = c->n_experts, K = c->topk;
    int *sel = xmalloc((size_t)n * K * sizeof(int), "routing");
    float *wt = xmalloc((size_t)n * K * sizeof(float), "routing weights");
    float *logits = xmalloc((size_t)n * E * sizeof(float), "router logits");
    matmul(logits, xn, l->router, n, H, E);
    for (int t = 0; t < n; t++) {
        float *score = logits + (size_t)t * E;
        for (int e = 0; e < E; e++) score[e] = sigmoid(score[e]);
        int *s = sel + (size_t)t * K;
        float *w = wt + (size_t)t * K;
        for (int j = 0; j < K; j++) {
            int best = -1; float bv = -INFINITY;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int q = 0; q < j; q++) if (s[q] == e) { taken = 1; break; }
                if (taken) continue;
                float v = score[e] + l->bias[e];
                if (v > bv) { bv = v; best = e; }
            }
            s[j] = best; w[j] = score[best];
        }
        float sum = 0;
        for (int j = 0; j < K; j++) sum += w[j];
        for (int j = 0; j < K; j++) w[j] = (c->norm_topk && K > 1 ? w[j] / (sum + 1e-20f) : w[j]) * c->route_scale;
        for (int j = 0; j < K; j++) m->ehit[li][s[j]] = 1;
    }
    free(logits);
    memset(out, 0, (size_t)n * H * sizeof(float));
    int *pair = xmalloc((size_t)n * K * sizeof(int), "expert pair of a choice");
    float *y = xmalloc((size_t)n * K * H * sizeof(float), "expert out");
    uint8_t *taken = NULL;
    const float **dev = NULL;
    int ndev = 0;
    if (vkt_ready()) {
        taken = xmalloc((size_t)n * K, "device pairs");
        dev = xmalloc((size_t)n * K * sizeof(*dev), "device rows");
        ndev = vkt_issue(li, xn, n, K, sel, taken);
    }
    int total = moe_cpu(m, li, xn, n, sel, ndev ? taken : NULL, 0, pair, y, 0, !ndev);
    if (ndev && !vkt_join(dev)) {    /* the batch failed (the tier stops): those pairs here */
        moe_cpu(m, li, xn, n, sel, taken, 1, pair, y, total, 0);
        ndev = 0;
    } else if (ndev) {               /* the CPU's experts still in RAM, now that nothing is in flight */
        uint8_t *noted = xcalloc((size_t)E, 1, "noted experts");
        int *by = m->cache[li].by_expert;
        for (int i = 0; i < n * K; i++) {
            int e = sel[i];
            if (taken[i] || noted[e] || by[e] < 0) continue;
            noted[e] = 1;
            VktExpertSrc vs = mimo_vk_src(m, m->cache[li].s[by[e]].buf);
            vkt_note(li, e, &vs);
        }
        free(noted);
    }
    /* The weighted sum, row by row, each row's experts in the order ITS router
     * chose them. A row's sum used to follow the block's first-use order, which
     * depends on the rows that happen to share the block: the same position
     * came out with different low bits in a 64-row prefill block and in a
     * one-row resume, so a prompt resumed from a prefix was not the prompt
     * computed cold. Now a row is a function of its own inputs and nothing else,
     * whatever the block boundaries. One row (a decode step) adds in the same
     * order as before: its first-use order IS its routing order. */
    double t1 = now_s();
    for (int t = 0; t < n; t++) {
        float *o = out + (size_t)t * H;
        for (int q = 0; q < K; q++) {
            size_t i = (size_t)t * K + q;
            const float *yy = ndev && taken[i] ? dev[i] : y + (size_t)pair[i] * H;
            float w = wt[i];
            for (int d = 0; d < H; d++) o[d] += w * yy[d];
        }
    }
    m->t_expert += now_s() - t1;
    free(pair); free(y); free(taken); free(dev); free(sel); free(wt);
}

static void embed_rows(Model *m, const int *ids, int n, float *h) {
    int H = m->c.hidden;
    for (int t = 0; t < n; t++) {
        int id = ids[t];
        if (id < 0 || id >= m->c.vocab) { fprintf(stderr, "[mimo] token id %d out of range\n", id); exit(1); }
        for (int i = 0; i < H; i++) h[(size_t)t * H + i] = bf16_to_f32(m->embed[(size_t)id * H + i]);
    }
}

/* Image rows to splice in: `rows[r]` replaces the embedding of the r-th
 * image-pad position of the WHOLE prompt (the first at `first_pad`). */
typedef struct { const float *rows; int n_rows; } ImageRows;

/* Layers [l0, L) on the CPU for the nc rows hc at position pc (in place; xn and tmp hold
 * nc rows each). trace: MIMO_TRACE's file, the residual after every block. */
static void layers_cpu(Model *m, float *hc, int nc, int pc, int l0, float *xn, float *tmp, FILE *trace) {
    Cfg *c = &m->c;
    int H = c->hidden;
    for (int li = l0; li < c->n_layers && nc > 0; li++) {
        Layer *l = &m->L[li];
        for (int t = 0; t < nc; t++) rmsnorm(xn + (size_t)t * H, hc + (size_t)t * H, l->ln1, H, c->eps);
        attention(m, li, xn, nc, pc, tmp);
        for (size_t i = 0; i < (size_t)nc * H; i++) hc[i] += tmp[i];
        if (trace) fwrite(hc, sizeof(float), (size_t)nc * H, trace);
        for (int t = 0; t < nc; t++) rmsnorm(xn + (size_t)t * H, hc + (size_t)t * H, l->ln2, H, c->eps);
        if (c->moe[li]) moe(m, li, xn, nc, tmp);
        else dense_mlp(m, l, xn, nc, tmp);
        for (size_t i = 0; i < (size_t)nc * H; i++) hc[i] += tmp[i];
        if (trace) fwrite(hc, sizeof(float), (size_t)nc * H, trace);
    }
}
/* The final norm and lm_head on the CPU: every row of hc into lc (all_rows), or its last. */
static void head_cpu(Model *m, const float *hc, int nc, float *lc, int all_rows, float *xn) {
    Cfg *c = &m->c;
    int H = c->hidden, from = all_rows ? 0 : nc - 1, rows = nc - from;
    for (int t = from; t < nc; t++) rmsnorm(xn + (size_t)(t - from) * H, hc + (size_t)t * H, m->norm, H, c->eps);
    dw_matmul(lc, xn, rows, &m->head);
}

#ifdef COLI_VULKAN
#include "mimo_chain.h"   /* COLI_VK_CHAIN: every layer's dense chain on the device */
#endif

/* One block of n tokens at m->pos. logits: NULL, or [n, V] (all rows) when
 * all_rows, else [V] for the last row. */
static void forward(Model *m, const int *ids, int n, float *logits, int all_rows,
                    const ImageRows *img, int *img_used) {
    Cfg *c = &m->c;
    int H = c->hidden;
    if (m->pos + n > m->ctx) { fprintf(stderr, "[mimo] context full (%d + %d > %d)\n", m->pos, n, m->ctx); exit(1); }
    float *h = xmalloc((size_t)n * H * sizeof(float), "residual");
    float *xn = xmalloc((size_t)n * H * sizeof(float), "normed");
    float *tmp = xmalloc((size_t)n * H * sizeof(float), "block out");
    embed_rows(m, ids, n, h);
    if (img && img->rows && c->image_token_id >= 0)
        for (int t = 0; t < n; t++)
            if (ids[t] == c->image_token_id && *img_used < img->n_rows) {
                memcpy(h + (size_t)t * H, img->rows + (size_t)(*img_used) * H, (size_t)H * sizeof(float));
                (*img_used)++;
            }
    /* MIMO_TRACE=<file>: the residual after the embedding and after every
     * attention and every MLP of the first block, f32 [2L+1][n][H] -- what
     * tools/make_mimo_ref.py --trace compares against the vendor's hooks. */
    static FILE *trace;
    static int traced;
    if (!traced && getenv("MIMO_TRACE")) { trace = fopen(getenv("MIMO_TRACE"), "wb"); }
    if (trace) fwrite(h, sizeof(float), (size_t)n * H, trace);
    /* The rows the CPU computes: all of them, or with the dense chain on (COLI_VK_CHAIN,
     * mimo_chain.h) the ones it did not (a lost device), from where the host's caches end. */
    int done = 0;
#ifdef COLI_VULKAN
    if (g_vk_chain && !trace) {
        done = mc_forward(m, h, n, m->pos, logits, all_rows, xn, tmp);
        if (done < n) mc_cpu_step(m, m->pos + done);
    }
#endif
    float *hc = h + (size_t)done * H, *lc = logits && all_rows ? logits + (size_t)done * c->vocab : logits;
    int nc = n - done, pc = m->pos + done;
    layers_cpu(m, hc, nc, pc, 0, xn, tmp, trace);
    if (trace) { fclose(trace); trace = NULL; }
    traced = 1;
    if (lc && nc > 0) head_cpu(m, hc, nc, lc, all_rows, xn);
    kv_prefix_record(&m->kvp, ids, m->pos, n);
    m->pos += n;
    m->forwards++;
    free(h); free(xn); free(tmp);
}

/* ---- several conversations at once (KV_SLOTS>1, serve_mux) -------------------
 * Each conversation owns a MimoSeq. The Model's layers hold the conversation a
 * prefill runs on (mimo_seq_swap trades it for a parked one); a decode step parks
 * them all and runs one forward over a row of each (forward_rows). */
static void mimo_seq_swap(Model *m, MimoSeq *q) {
    for (int li = 0; li < m->c.n_layers; li++) {
        Layer *l = &m->L[li];
        float *K = l->K, *V = l->V; int *rp = l->ring_pos;
        l->K = q->K[li]; l->V = q->V[li]; l->ring_pos = q->ring_pos[li];
        q->K[li] = K; q->V[li] = V; q->ring_pos[li] = rp;
    }
    int pos = m->pos; kv_prefix p = m->kvp;
    m->pos = q->pos; m->kvp = q->kvp;
    q->pos = pos; q->kvp = p;
}
/* A conversation's caches of its own, the shape kv_alloc gives the Model's. */
static void mimo_seq_alloc(Model *m, MimoSeq *q) {
    Cfg *c = &m->c;
    memset(q, 0, sizeof *q);
    for (int li = 0; li < c->n_layers; li++) {
        Layer *l = &m->L[li];
        int k = c->swa[li];
        q->K[li] = xmalloc((size_t)l->rows * c->kv_heads[k] * c->head_dim[k] * sizeof(float), "K cache");
        q->V[li] = xmalloc((size_t)l->rows * c->kv_heads[k] * c->v_dim[k] * sizeof(float), "V cache");
        q->ring_pos[li] = xmalloc((size_t)l->rows * sizeof(int), "ring positions");
        for (int r = 0; r < l->rows; r++) q->ring_pos[li][r] = -1;
    }
    kv_prefix_alloc(&q->kvp, m->ctx);
}

/* One decode step of several conversations: row t is the token ids[t] at rows[t].pos
 * of the conversation rows[t].seq, every conversation parked. The matrices, the routed
 * experts and lm_head run once over the n rows; the attention reads and writes each
 * row's own caches (attention_rows). The CPU kernels give a row the same bits whatever
 * n is, so each conversation gets the logits it would alone: [n, V] into logits. */
static void forward_rows(Model *m, const MimoRow *rows, const int *ids, int n, float *logits) {
    Cfg *c = &m->c;
    int H = c->hidden;
    float *h = xmalloc((size_t)n * H * sizeof(float), "residual");
    float *xn = xmalloc((size_t)n * H * sizeof(float), "normed");
    float *tmp = xmalloc((size_t)n * H * sizeof(float), "block out");
    embed_rows(m, ids, n, h);
    m->mux_rows = rows;
    layers_cpu(m, h, n, 0, 0, xn, tmp, NULL);
    m->mux_rows = NULL;
    head_cpu(m, h, n, logits, 1, xn);
    for (int t = 0; t < n; t++) {
        kv_prefix_record(&rows[t].seq->kvp, ids + t, rows[t].pos, 1);
        rows[t].seq->pos = rows[t].pos + 1;
    }
    m->forwards++;
    free(h); free(xn); free(tmp);
}

/* The read-out of a prefill, for the logprobs channel (SUBMIT logprobs=k): one
 * ECHO frame per prompt position this prefill feeds, in position order, before
 * any DATA. The logits at position p predict the token at p + 1, so the frame
 * of a position carries the log-probability of the token that is actually
 * there, over the whole vocabulary, and not only when it is among the top k:
 * that is what scores an option the model would never have written.
 *
 * The first position fed has no predictor among the rows computed here. On a
 * cold prompt it is position 0, which has nothing to condition on, and its
 * frame carries " nan 0". Resumed from a photo, `first` is the photo's own last
 * logits, which predict exactly that token. There is no third case: a prompt
 * read out never resumes from a live prefix, because the positions before it
 * would have no frame at all (see the serve loop).
 *
 * This is its own argument and not a mode of something else: it asks for every
 * row's logits and changes nothing about how the rows are computed. */
typedef struct {
    const char *id;
    int k;
    Tok *tok;
    const float *first;
} Echo;

static void echo_frame(const Echo *e, int pos, int token, const float *lo, int vocab) {
    char tail[1024], piece[512];
    coli_logprob_tail(tail, sizeof(tail), lo, vocab, token, e->k);
    int n = tok_decode(e->tok, &token, 1, piece, (int)sizeof(piece));
    if (n < 0) n = 0;
    printf("ECHO %s %d %d%s\n", e->id, n, pos, tail);
    if (n > 0) fwrite(piece, 1, (size_t)n, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

/* A prompt in blocks of MIMO_CHUNK rows (default 64): the last block's last
 * row (or every row, for the teacher-forcing dump) comes back in `logits`.
 * With `echo`, every row of every block goes through the head and is read out
 * as it comes; only one block of logits is ever held, not the prompt's. */
static void prefill(Model *m, const int *ids, int n, float *logits, float *all_logits,
                    const ImageRows *img, const Echo *echo) {
    int chunk = env_int("MIMO_CHUNK", 64);
#ifdef COLI_VULKAN
    /* the dense chain with a chunk from the budget (vkc_chunk_auto): blocks of its chunk,
     * so a long prompt's MoE steps are big ones the tier can stream (not with a read-out,
     * which holds a block's logits) */
    if (!getenv("MIMO_CHUNK") && g_vk_chain && vkc_chunk_auto() && !echo) {
        MimoChain *ch = mc_setup(m);
        if (ch && !ch->failed) chunk = mc_chunk_rows(ch, m);
    }
#endif
    if (chunk < 1) chunk = 1;
    int used = 0, V = m->c.vocab;
    float *rows = NULL;
    if (echo && !all_logits) {
        rows = xmalloc((size_t)(n < chunk ? n : chunk) * V * sizeof(float), "read-out logits");
        echo_frame(echo, m->pos, ids[0], echo->first, V);
    }
    for (int b = 0; b < n; b += chunk) {
        int nb = n - b < chunk ? n - b : chunk;
        if (all_logits) forward(m, ids + b, nb, all_logits + (size_t)b * V, 1, img, &used);
        else if (rows) {
            int p0 = m->pos;
            forward(m, ids + b, nb, rows, 1, img, &used);
            for (int t = 0; t < nb; t++) {
                if (b + t + 1 < n) echo_frame(echo, p0 + t + 1, ids[b + t + 1], rows + (size_t)t * V, V);
                else if (logits) memcpy(logits, rows + (size_t)t * V, (size_t)V * sizeof(float));
            }
        }
        else forward(m, ids + b, nb, b + nb == n ? logits : NULL, 0, img, &used);
    }
    free(rows);
    if (all_logits && logits) memcpy(logits, all_logits + (size_t)(n - 1) * V, (size_t)V * sizeof(float));
}

/* ----------------------------------------------------------------- vision ---- */

/* The tower, when the container carries one (visual.*): vision_run turns the
 * gateway's patches into the rows that replace the image-pad embeddings. */
typedef struct Vision Vision;
static Vision *g_vision;
#include "mimo_vision.h"

#ifdef COLI_VULKAN
/* The dense matrices go up first, at start-up, the trunk before the tower, so
 * the experts take only what they leave; one that does not fit stays on the
 * CPU. Returns how many went up. The dense chain alone (the per-matrix path off)
 * takes the trunk and leaves the tower on the CPU (vision 0). */
static void mc_place(Model *m, int *dropped);   /* below: a fit's layers, layer by layer */
static int vk_dense_upload(Model *m, int vision) {
    int n = 0;
    if (g_mc_fit.L && !g_mc_placed) mc_place(m, NULL);   /* the chain's fit: its layers, each whole or none */
    for (int li = 0; li < m->c.n_layers; li++) {
        Layer *l = &m->L[li];
        n += dw_upload(&l->qkv) + dw_upload(&l->o);
        if (!m->c.moe[li]) n += dw_upload(&l->gate) + dw_upload(&l->up) + dw_upload(&l->down);
    }
    n += dw_upload(&m->head);
    if (g_vision && vision) {
        Vision *v = g_vision;
        n += dw_upload(&v->embed) + dw_upload(&v->fc1) + dw_upload(&v->fc2);
        for (int i = 0; i < v->depth; i++)
            n += dw_upload(&v->b[i].qkv) + dw_upload(&v->b[i].proj) + dw_upload(&v->b[i].gate) +
                 dw_upload(&v->b[i].up) + dw_upload(&v->b[i].down);
    }
    return n;
}

/* What vk_dense_upload will place, in bytes (the tier's budget leaves it room). */
static size_t dw_bytes(const DW *d) {
    if (!d->w) return 0;
    size_t e = (size_t)d->O * d->I;
    switch (d->fmt) {
    case DW_F32:  return e * 4;
    case DW_BF16: return e * 2;
    case DW_FP8:  return e + (size_t)d->O * d->nblk * 4;
    case DW_I8:   return e + (size_t)d->O * 4;
    }
    return 0;
}
static size_t vk_dense_bytes(const Model *m, int vision) {
    size_t b = dw_bytes(&m->head);
    for (int li = 0; li < m->c.n_layers; li++) {
        const Layer *l = &m->L[li];
        b += dw_bytes(&l->qkv) + dw_bytes(&l->o);
        if (!m->c.moe[li]) b += dw_bytes(&l->gate) + dw_bytes(&l->up) + dw_bytes(&l->down);
    }
    if (g_vision && vision) {
        const Vision *v = g_vision;
        b += dw_bytes(&v->embed) + dw_bytes(&v->fc1) + dw_bytes(&v->fc2);
        for (int i = 0; i < v->depth; i++)
            b += dw_bytes(&v->b[i].qkv) + dw_bytes(&v->b[i].proj) + dw_bytes(&v->b[i].gate) +
                 dw_bytes(&v->b[i].up) + dw_bytes(&v->b[i].down);
    }
    return b;
}

/* ---- a partial chain (vk_chain.h, vkc_fit): the first n layers on the device --------
 * mc_fit_start decides n at start-up, before any upload and before the tier sizes its
 * budget. A layer's bytes: its matrices in their dense form (MIMO_DENSE_BITS: the fused
 * qkv and o_proj, the dense layer's MLP too), its K/V mirror at the size the chain
 * allocates (a sliding layer's ring; a full layer's context, or COLI_VK_KV_DEVICE_ROWS
 * when that asks for fewer: the split covers the rest) and its norms and sink logits.
 * Fixed: the final norm and the scratch of one block of vkc_fit_rows(MIMO_CHUNK) rows
 * (the residual's handoff reuses the normed rows' read-back). Tail: the head, and the
 * vision tower when the per-matrix path takes it (COLI_VK_DENSE). The CPU's layers, the
 * head and the tower of a partial chain stay on the CPU with their host copies (vk_off:
 * the per-matrix path leaves them there). */
static void mc_fit_start(Model *m) {
    Cfg *c = &m->c; int L = c->n_layers, H = c->hidden;
    for (int k = 0; k < 2; k++) if (c->head_dim[k] > 256 || c->v_dim[k] > 256) return;   /* the chain declines */
    size_t *per = calloc((size_t)L, sizeof *per), *mat = calloc((size_t)L, sizeof *mat);
    int ok = per && mat;
    long forced = vkc_kv_env("COLI_VK_KV_DEVICE_ROWS", 0);
#define MC_FITW(d) do { const DW *d_ = (d); int g_; const float *s_; int f_ = d_->w ? dw_vk_fmt(d_, &g_, &s_) : -1; \
        if (f_ < 0) ok = 0; else { per[i] += vkc_fit_tensor(f_, d_->I, d_->O, g_); mat[i] += coli_vk_tensor_payload(f_, d_->I, d_->O, g_); } } while (0)
    for (int i = 0; i < L && ok; i++) {
        const Layer *l = &m->L[i];
        MC_FITW(&l->qkv); MC_FITW(&l->o);
        if (!c->moe[i]) { MC_FITW(&l->gate); MC_FITW(&l->up); MC_FITW(&l->down); }
        McGeo g = mc_geo(m, i);
        size_t rows = !g.swa && forced > 0 && forced < g.rows ? (size_t)forced : (size_t)g.rows;
        per[i] += vkc_fit_buf(rows * g.kd * sizeof(float)) + vkc_fit_buf(rows * g.vdd * sizeof(float)) +
                  ((size_t)2 * H + (l->sink ? (size_t)c->heads[c->swa[i]] : 0)) * sizeof(float);
    }
#undef MC_FITW
    int hg; const float *hs;
    int hf = m->head.w ? dw_vk_fmt(&m->head, &hg, &hs) : -1;
    if (!ok || hf < 0) { free(per); free(mat); return; }   /* a matrix the device does not take: the chain declines as before */
    size_t tail = vkc_fit_tensor(hf, m->head.I, m->head.O, hg);
    if (g_vision && coli_vk_dense()) {   /* the tower the per-matrix path puts on the device */
        Vision *v = g_vision;
        const DW *t[3] = {&v->embed, &v->fc1, &v->fc2};
        for (int k = 0; k < 3; k++) if (t[k]->w && dw_vk_fmt(t[k], &hg, &hs) >= 0) tail += vkc_fit_tensor(dw_vk_fmt(t[k], &hg, &hs), t[k]->I, t[k]->O, hg);
        for (int b = 0; b < v->depth; b++) {
            const DW *d[5] = {&v->b[b].qkv, &v->b[b].proj, &v->b[b].gate, &v->b[b].up, &v->b[b].down};
            for (int k = 0; k < 5; k++) if (d[k]->w && dw_vk_fmt(d[k], &hg, &hs) >= 0) tail += vkc_fit_tensor(dw_vk_fmt(d[k], &hg, &hs), d[k]->I, d[k]->O, hg);
        }
    }
    MimoChain g; memset(&g, 0, sizeof g);
    g.o_kvd = calloc((size_t)L, sizeof(size_t));
    const char *mc = getenv("MIMO_CHUNK");
    int R = vkc_fit_rows(mc && *mc && atoi(mc) > 0 ? atoi(mc) : 64);
    g_mc_count = 0; g_mc_fitcount = 1;
    mc_scratch(&g, m, R, 1);
    size_t fixed = (size_t)g_mc_count + vkc_fit_buf((size_t)H * sizeof(float));
    g_mc_count = -1; g_mc_fitcount = 0;
    free(g.o_kvd);
    vkc_fit("mimo", L, per, mat, fixed, tail, &g_mc_fit);
    /* the layers the primary leaves, on COLI_VK_DEV2's device: a fit of its own from layer
     * n0 with that device's free memory, its pipelines up now (the head goes with it when it
     * ends at the last layer with room; the tower stays on the CPU) */
    int n0 = g_mc_fit.n;
    if (vkc_fit_partial(&g_mc_fit) && n0 > 0 && n0 < L && mc_dev2_wanted() && getenv("COLI_VK_DEV2") &&
        coli_vk_dev2_open_env()) {
        vkc_device(1);
        size_t tail2 = vkc_fit_tensor(hf, m->head.I, m->head.O, hg);
        int n2 = vkc_fit("mimo dev2", L - n0, per + n0, mat + n0, fixed, tail2, &g_mc_fit2);
        if (n2 > 0 && !vkc_init()) {
            fprintf(stderr, "[VK] mimo chain: the second device's pipelines did not come up; its layers stay on the CPU\n");
            n2 = 0;
        }
        g_mc_fit2_on = n2 > 0;
        if (!g_mc_fit2_on) g_mc_fit2.tail = 0;
        vkc_device(0);
    }
    free(per); free(mat);
}
/* Layers from..L-1, the head and the tower refused to the device: what of them is there
 * is freed, and the per-matrix path leaves them on the CPU. */
static void mc_refuse(DW *d) {
    if (d->vk && !d->vk_gone) { coli_vk_tensor_free((ColiVkTensor *)d->vk); d->vk = NULL; }
    if (!d->vk_gone) d->vk_off = 1;
}
static void mc_refuse_from(Model *m, int from) {
    if (vkc_ready()) vkc_finish();
    for (int li = from; li < m->c.n_layers; li++) {
        Layer *l = &m->L[li];
        mc_refuse(&l->qkv); mc_refuse(&l->o); mc_refuse(&l->gate); mc_refuse(&l->up); mc_refuse(&l->down);
    }
    if (!(m->head.vk && coli_vk_tensor_dev((ColiVkTensor *)m->head.vk)))   /* the second device's chain keeps it */
        mc_refuse(&m->head);
    if (g_vision) {
        Vision *v = g_vision;
        mc_refuse(&v->embed); mc_refuse(&v->fc1); mc_refuse(&v->fc2);
        for (int b = 0; b < v->depth; b++) {
            mc_refuse(&v->b[b].qkv); mc_refuse(&v->b[b].proj); mc_refuse(&v->b[b].gate); mc_refuse(&v->b[b].up); mc_refuse(&v->b[b].down);
        }
    }
}
static void dw_drop(DW *d, int *n);   /* below */
/* The fit's layers up, layer by layer: each whole or not at all (a layer that fails is
 * freed, it and the rest refused, vkc_fit_shrink); dropped: with COLI_VK_DENSE_HOST, each
 * layer's host copies given back once all of it is on the device (counted there). */
static void mc_place(Model *m, int *dropped) {
    int n0 = g_mc_fit.n;
    for (int i = 0; i < g_mc_fit.n; i++) {
        Layer *l = &m->L[i];
        DW *d[5] = {&l->qkv, &l->o, &l->gate, &l->up, &l->down};
        int nd = m->c.moe[i] ? 2 : 5, ok = 1;
        for (int k = 0; k < nd && ok; k++) ok = dw_upload(d[k]);
        if (!ok) {
            mc_refuse_from(m, i);
            vkc_fit_shrink("mimo", &g_mc_fit, i, "a dense matrix did not go up");
            break;
        }
        if (dropped) for (int k = 0; k < nd; k++) dw_drop(d[k], dropped);
        vkc_fit_mark(&g_mc_fit, i);
    }
    int end = g_mc_fit.n;
    if (g_mc_fit2_on && g_mc_fit.n < n0) {
        /* the primary placed fewer layers than its fit: the second device's would not follow
         * them, so they stay on the CPU too */
        fprintf(stderr, "[VK] mimo chain: the primary device stopped before layer %d; layers %d..%d stay on the CPU, "
                        "not on the second device\n", n0, n0, n0 + g_mc_fit2.n - 1);
        g_mc_fit2_on = 0;
        vkc_device(1); vkc_shutdown(); vkc_device(0);
    } else if (g_mc_fit2_on) {   /* the second device's layers (their host copies kept), and the head with the last one */
        vkc_device(1); g_mimo_vk_dev = 1;
        for (int i = n0; i < n0 + g_mc_fit2.n; i++) {
            Layer *l = &m->L[i];
            DW *d[5] = {&l->qkv, &l->o, &l->gate, &l->up, &l->down};
            int nd = m->c.moe[i] ? 2 : 5, ok = 1;
            for (int k = 0; k < nd && ok; k++) ok = dw_upload(d[k]);
            if (!ok) {
                g_mimo_vk_dev = 0;
                mc_refuse_from(m, i);
                vkc_fit_shrink("mimo dev2", &g_mc_fit2, i - n0, "a dense matrix did not go up");
                g_mimo_vk_dev = 1;
                break;
            }
            vkc_fit_mark(&g_mc_fit2, i - n0);
        }
        end = n0 + g_mc_fit2.n;
        if (end == m->c.n_layers && g_mc_fit2.tail && !dw_upload(&m->head)) g_mc_fit2.tail = 0;
        g_mimo_vk_dev = 0;
        vkc_fit_placed("mimo dev2", &g_mc_fit2);
        if (!g_mc_fit2.n) { g_mc_fit2_on = 0; vkc_shutdown(); }
        vkc_device(0);
    }
    if (vkc_fit_partial(&g_mc_fit)) mc_refuse_from(m, end);   /* the CPU's layers, the head (unless the second device's), the tower */
    vkc_fit_placed("mimo", &g_mc_fit);
    g_mc_placed = 1;
}

/* ---- the dense matrices on the device only (COLI_VK_DENSE_HOST) ----------------
 * With the trunk on the device (the chain, or COLI_VK_DENSE), every trunk matrix goes
 * up now, before the expert tier sizes its budget, and its host copy is dropped; the
 * tower too when the per-matrix path is on (COLI_VK_DENSE), else it stays on the CPU
 * with its copy. The per-matrix path then answers on the device whatever
 * coli_vk_dense() says (the CPU holds no copy), and the CPU reads a matrix back from
 * disk only when it needs it after all (dw_reload: a lost device). Kept on the host:
 * the embedding (its rows are gathered on the CPU), the routers (the host computes
 * them), norms and vectors, the patch embedding. */
static void dw_drop(DW *d, int *n) {
    if (!d->w || d->vk_off || d->vk_gone || !d->vk_name || !dw_upload(d)) return;
    size_t b = dw_bytes(d);
    free(d->w); free(d->s);
    d->w = NULL; d->s = NULL; d->vk_gone = 1;
    coli_vk_dense_host_dropped(b);
    (*n)++;
}
static void mimo_dho_start(Model *m) {
    if (!g_vk_ready) return;
    int vision = coli_vk_dense();
    /* a partial chain's fit: the first n layers' bytes only (the head and the tower with
     * the tail); n = 0 leaves the dense part on the CPU */
    size_t bytes = vk_dense_bytes(m, vision);
    int fit = g_mc_fit.L > 0, tail = !fit || (g_mc_fit.n == g_mc_fit.L && g_mc_fit.tail);
    if (fit) {
        bytes = 0;
        for (int li = 0; li < g_mc_fit.n; li++) {
            const Layer *l = &m->L[li];
            bytes += dw_bytes(&l->qkv) + dw_bytes(&l->o);
            if (!m->c.moe[li]) bytes += dw_bytes(&l->gate) + dw_bytes(&l->up) + dw_bytes(&l->down);
        }
        if (tail) bytes += vk_dense_bytes(m, vision) - vk_dense_bytes(m, 0) + dw_bytes(&m->head);
    }
    if (!coli_vk_dense_host_decide("mimo", (coli_vk_dense() || g_vk_chain) && !(fit && !g_mc_fit.n), bytes)) return;
    g_mimo_dho = 1;
    g_dw_dho_S = &m->S; g_dw_dho_c = &m->c;
    int n = 0;
    if (fit) {
        coli_vk_dense_host_layers(g_mc_fit.n, g_mc_fit.L);
        mc_place(m, &n);   /* each layer whole, then its host copies */
        tail = g_mc_fit.n == g_mc_fit.L && g_mc_fit.tail;
    } else
        for (int li = 0; li < m->c.n_layers; li++) {
            Layer *l = &m->L[li];
            dw_drop(&l->qkv, &n); dw_drop(&l->o, &n);
            if (!m->c.moe[li]) { dw_drop(&l->gate, &n); dw_drop(&l->up, &n); dw_drop(&l->down, &n); }
        }
    if (tail) dw_drop(&m->head, &n);
    if (g_vision && tail) {
        Vision *v = g_vision;
        DW *t[3] = {&v->embed, &v->fc1, &v->fc2};
        for (int k = 0; k < 3; k++) { if (vision) dw_drop(t[k], &n); else t[k]->vk_off = 1; }
        for (int i = 0; i < v->depth; i++) {
            DW *b[5] = {&v->b[i].qkv, &v->b[i].proj, &v->b[i].gate, &v->b[i].up, &v->b[i].down};
            for (int k = 0; k < 5; k++) { if (vision) dw_drop(b[k], &n); else b[k]->vk_off = 1; }
        }
    }
    coli_vk_dense_host_placed("mimo", vision || !g_vision
        ? "the embedding (its rows are gathered on the CPU), the routers, norms, the patch embedding"
        : "the embedding (its rows are gathered on the CPU), the routers, norms, the vision tower (on the CPU: COLI_VK_DENSE=1 puts it on the device)");
}

/* Is the expert in this layer's RAM cache now (the tier's balance asks)? */
static int vk_in_ram(void *ctx, int li, int e) {
    const Model *m = ctx;
    return m->cache[li].by_expert && m->cache[li].by_expert[e] >= 0;
}

/* The routed experts on the shared tier (vk_tier.c): the release's MXFP4 with its
 * e8m0 group scales, SwiGLU. This engine keeps no expert history, so the tier
 * starts empty and fills as experts pass by, evicting the coldest once its budget
 * is full. MIMO_VK_EXPERTS, the switch of the per-engine tier before this one, is
 * read as follows: 0 keeps the experts on the CPU (no tier), N > 0 sizes the
 * budget at N experts (COLI_VK_TIER_GB, which wins when set); unset, the tier's own
 * budget applies. */
static void vk_tier_start(Model *m) {
    Cfg *c = &m->c;
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    const char *mx = getenv("MIMO_VK_EXPERTS"), *gb = getenv("COLI_VK_TIER_GB");
    if (mx && *mx && atoi(mx) > 0 && !(gb && *gb)) {
        size_t eb = vkt_expert_bytes(c->hidden, c->moe_inter, f, f);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.12g", ((double)atoi(mx) * eb + eb / 2) / 1073741824.0);
        setenv("COLI_VK_TIER_GB", buf, 1);
        fprintf(stderr, "[VK] mimo: MIMO_VK_EXPERTS=%s read as a tier budget of %s experts (COLI_VK_TIER_GB=%s)\n",
                mx, mx, buf);
    }
    int cap = 0, nmoe = 0;
    for (int li = 0; li < c->n_layers; li++) if (c->moe[li]) { nmoe++; if (m->cache[li].cap > cap) cap = m->cache[li].cap; }
    VktConfig vc = {.engine = "mimo", .layers = c->n_layers, .experts = c->n_experts,
                    .hidden = c->hidden, .inter = c->moe_inter, .topk = c->topk,
                    .gate_up = f, .down = f, .act = VKT_ACT_SWIGLU,
                    .max_rows = MIMO_VK_ROWS * c->topk,
                    .ram_reserve = (size_t)(m->e_bytes + 8192) * (size_t)cap * (size_t)nmoe,
                    .dense_bytes = g_mc_fit.L && vkc_fit_partial(&g_mc_fit) ? mc_kv_bytes_n(m, g_mc_fit.n) :   /* a partial chain: placed already, its caches to come */
                                   (coli_vk_dense_device_only() ? 0 :   /* placed already (mimo_dho_start) */
                                    coli_vk_dense() ? vk_dense_bytes(m, 1) : g_vk_chain ? vk_dense_bytes(m, 0) : 0) +
                                   (g_vk_chain ? mc_kv_bytes(m) : 0),   /* the chain's KV caches */
                    .in_ram = vk_in_ram, .ram_ctx = m,
                    .load = mimo_vk_load, .release = mimo_vk_release, .load_ctx = m, .load_batch = mimo_vk_load_batch};
    atexit(coli_vk_shutdown);   /* before vkt_init, which makes the expert batch's pipelines and can still refuse (no room): the device goes at exit either way, after the tier's teardown */
    if (vkt_init(&vc, NULL)) atexit(vkt_shutdown);
}
#endif

/* --------------------------------------------------------------- sampling ---- */

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

typedef struct { float p; int id; } SampleProb;
static int sample_desc(const void *a, const void *b) {
    float pa = ((const SampleProb *)a)->p, pb = ((const SampleProb *)b)->p;
    return (pb > pa) - (pa > pb);
}

static int sample(const float *logits, int vocab, float temperature, float top_p) {
    if (temperature <= 0.0f) return argmax(logits, vocab);
    SampleProb *rank = xmalloc((size_t)vocab * sizeof(SampleProb), "sampling");
    float top = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    for (int i = 0; i < vocab; i++) {
        float p = expf((logits[i] - top) / temperature);
        total += p; rank[i].p = p; rank[i].id = i;
    }
    qsort(rank, (size_t)vocab, sizeof(SampleProb), sample_desc);
    double cut = (top_p > 0.0f && top_p < 1.0f) ? top_p * total : total, kept = 0.0;
    int n = 0;
    while (n < vocab && kept < cut) kept += rank[n++].p;
    double draw = ((double)rand() / RAND_MAX) * kept, acc = 0.0;
    int pick = rank[0].id;
    for (int i = 0; i < n; i++) { acc += rank[i].p; if (acc >= draw) { pick = rank[i].id; break; } }
    free(rank);
    return pick;
}

/* ---------------------------------------------------------------- photos ---- */

/* SUBMIT pin=1: a photo of the state at the end of this prompt, so that the
 * prompts that begin with it (the options of a closed question) resume from
 * there instead of computing it again. pin_pool.h keeps several, nested, and
 * restores the deepest one that is a strict prefix of the new prompt.
 *
 * WHAT A LAYER CARRIES BETWEEN TOKENS, and so what the photo must hold:
 *
 *   full attention   K/V rows indexed by position. Nothing after the photo
 *                    writes below it, so they stay where they are, and the
 *                    photo holds only the token ids; kv_prefix_holds() is what
 *                    says they are still the photo's rows (an unrelated prompt
 *                    in between rewrites them from position 0).
 *   sliding window   a RING of `rows` slots, position p in slot p % rows, and
 *                    the position each slot holds. Every option overwrites the
 *                    ring slots of the photo's last positions -- its own token at
 *                    P lands where P - rows was -- so after one option the ring
 *                    no longer describes the photo. The ring, keys, values and
 *                    positions, is copied into the photo and copied back.
 *   the position     m->pos, which is the photo's length.
 *
 * Nothing else carries over: the sinks are weights, the router has no state,
 * and the expert cache changes the time, never the numbers. On Flash the rings
 * of the 39 windowed layers are 128 x 8 x (192 + 128) floats each, about 50 MB
 * a photo, so COLI_PIN_SLOTS (default 4) is what bounds the memory. */
typedef struct {
    int n_layers;
    int rows[MIMO_MAX_LAYERS];               /* 0: a full-attention layer */
    size_t kd[MIMO_MAX_LAYERS], vd[MIMO_MAX_LAYERS];
    float *K[MIMO_MAX_LAYERS], *V[MIMO_MAX_LAYERS];
    int *ring_pos[MIMO_MAX_LAYERS];
} PinState;

static ColiPinPool g_pins;

static void pin_state_free(void *v) {
    PinState *st = (PinState *)v;
    if (!st) return;
    for (int li = 0; li < st->n_layers; li++) { free(st->K[li]); free(st->V[li]); free(st->ring_pos[li]); }
    free(st);
}

/* Copies the rings into `st` (allocated on first use). NULL when memory runs
 * out: a photo is an optimisation and never the reason a request fails. */
static PinState *pin_state_save(Model *m, PinState *st) {
    Cfg *c = &m->c;
    if (st && st->n_layers != c->n_layers) { pin_state_free(st); st = NULL; }
    if (!st) {
        st = calloc(1, sizeof(*st));
        if (!st) return NULL;
        st->n_layers = c->n_layers;
        for (int li = 0; li < c->n_layers; li++) {
            if (!c->swa[li]) continue;
            Layer *l = &m->L[li];
            st->rows[li] = l->rows;
            st->kd[li] = (size_t)l->rows * c->kv_heads[1] * c->head_dim[1];
            st->vd[li] = (size_t)l->rows * c->kv_heads[1] * c->v_dim[1];
            st->K[li] = malloc(st->kd[li] * sizeof(float));
            st->V[li] = malloc(st->vd[li] * sizeof(float));
            st->ring_pos[li] = malloc((size_t)l->rows * sizeof(int));
            if (!st->K[li] || !st->V[li] || !st->ring_pos[li]) { pin_state_free(st); return NULL; }
        }
    }
    for (int li = 0; li < c->n_layers; li++) {
        if (!st->rows[li]) continue;
        Layer *l = &m->L[li];
        memcpy(st->K[li], l->K, st->kd[li] * sizeof(float));
        memcpy(st->V[li], l->V, st->vd[li] * sizeof(float));
        memcpy(st->ring_pos[li], l->ring_pos, (size_t)l->rows * sizeof(int));
    }
    return st;
}

static int pin_state_load(Model *m, const PinState *st) {
    Cfg *c = &m->c;
    if (!st || st->n_layers != c->n_layers) return 0;
    for (int li = 0; li < c->n_layers; li++)
        if (st->rows[li] != (c->swa[li] ? m->L[li].rows : 0)) return 0;
    for (int li = 0; li < c->n_layers; li++) {
        if (!st->rows[li]) continue;
        Layer *l = &m->L[li];
        memcpy(l->K, st->K[li], st->kd[li] * sizeof(float));
        memcpy(l->V, st->V[li], st->vd[li] * sizeof(float));
        memcpy(l->ring_pos, st->ring_pos[li], (size_t)l->rows * sizeof(int));
    }
#ifdef COLI_VULKAN
    mc_rings_rewritten(m);   /* the dense chain's copies of the rings go up again */
#endif
    return 1;
}

/* Only a request that asks for it takes a photo, and it goes into the pool,
 * where no other request writes: an ordinary prompt can make a photo stale
 * (it rewrites the rows the photo stands on), never replace it. */
static void pin_save(Model *m, const int *ids, int n, const float *logits) {
    coli_pin_pool_init(&g_pins, m->c.vocab);
    ColiPin *k = coli_pin_store(&g_pins, ids, n, logits);
    if (!k) return;
    PinState *st = pin_state_save(m, (PinState *)k->state);
    if (!st) { k->state = NULL; k->len = 0; return; }   /* without its rings it would lie */
    k->state = st;
    fprintf(stderr, "[PIN] photo of %d tokens\n", n);
    fflush(stderr);
}

/* The deepest photo that is a strict prefix of this prompt and whose rows the
 * state still holds, put back: the rings, the position and the record. Returns
 * its length and its logits (the predictor of the first fresh token), or 0.
 * A photo whose rows are gone cannot help any more: it is released, memory and
 * all, and the next deepest is tried instead of giving up. */
static int pin_restore(Model *m, const int *ids, int n, const float **logit) {
    *logit = NULL;
    int s = coli_pin_best(&g_pins, ids, n);
    while (s >= 0) {
        ColiPin *k = &g_pins.slot[s];
        if (k->logit && kv_prefix_holds(&m->kvp, k->ids, k->len) &&
            pin_state_load(m, (const PinState *)k->state)) {
            m->pos = k->len;
            kv_prefix_clear(&m->kvp);
            kv_prefix_record(&m->kvp, k->ids, 0, k->len);
            *logit = k->logit;
            coli_pin_touch(&g_pins, s);
            return k->len;
        }
        fprintf(stderr, "[PIN] released the photo of %d tokens: the state no longer holds it\n", k->len);
        k->len = 0;
        pin_state_free(k->state);
        k->state = NULL;
        s = coli_pin_best(&g_pins, ids, n);
    }
    return 0;
}

/* ------------------------------------------------------------------ serve ---- */

/* The 7th numeric field (the byte count of a payload extension) is accepted, but
 * only as 0: the gateway puts it ahead of the logprobs= key on chat requests, and
 * this engine takes no grammar, so any other value is still a bad header. */
static const ColiServeWireProfile mimo_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 26,
    .max_extension_bytes = 0,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
    .allow_extension_bytes = 1,
};

static void serve_line(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fflush(stdout);
}

/* the dashboard's grid: rows are the MoE layers, columns the routed experts */
static void serve_emap(Model *m) {
    Cfg *c = &m->c;
    int rows = 0;
    for (int li = 0; li < c->n_layers; li++) rows += c->moe[li];
    int cols = c->n_experts;
    char *hex = xmalloc((size_t)rows * cols * 2 + 1, "emap");
    int w = 0;
    for (int li = 0; li < c->n_layers; li++) {
        if (!c->moe[li]) continue;
        for (int e = 0; e < cols; e++) {
            int byte = (vkt_resident(li, e) ? 2 : m->cache[li].by_expert[e] >= 0) << 6;   /* 2 = on the Vulkan device */
            hex[w++] = "0123456789abcdef"[byte >> 4];
            hex[w++] = "0123456789abcdef"[byte & 15];
        }
    }
    hex[w] = 0;
    serve_line("EMAP %d %d %s\n", rows, cols, hex);
    free(hex);
}

static void serve_hits(Model *m) {
    Cfg *c = &m->c;
    int rows = 0;
    for (int li = 0; li < c->n_layers; li++) rows += c->moe[li];
    int cols = c->n_experts, nb = (rows * cols + 7) / 8, bit = 0;
    uint8_t *bitmap = xcalloc((size_t)nb, 1, "hits bitmap");
    for (int li = 0; li < c->n_layers; li++) {
        if (!c->moe[li]) continue;
        for (int e = 0; e < cols; e++, bit++)
            if (m->ehit[li][e]) { bitmap[bit >> 3] |= (uint8_t)(1 << (bit & 7)); m->ehit[li][e] = 0; }
    }
    char *hex = xmalloc((size_t)nb * 2 + 1, "hits hex");
    for (int b = 0; b < nb; b++) {
        hex[2 * b] = "0123456789abcdef"[bitmap[b] >> 4];
        hex[2 * b + 1] = "0123456789abcdef"[bitmap[b] & 15];
    }
    hex[2 * nb] = 0;
    serve_line("HITS %d %d %s\n", rows, cols, hex);
    free(hex); free(bitmap);
}

/* max_tokens is a ceiling; generation needs at least one free position. A
 * read-only request (max_tokens=0, legal only with logprobs) generates nothing,
 * so its prompt may fill the context. */
static int serve_budget(int prompt, int requested, int context, int logprobs) {
    if (prompt < 1 || prompt > context) return -1;
    int budget = requested > 0 ? requested : (logprobs > 0 ? 0 : 256);
    int room = context - prompt;
    if (budget > 0 && room == 0) return -1;
    return budget < room ? budget : room;
}

/* When a request was accepted, and the counters then: DONE and PROF report its share. */
typedef struct { double started, disk0, expert0, attn0; uint64_t hits0, miss0, fw0; int n_prompt, budget; } MimoReq;

/* A request's prompt into the caches the Model's layers hold: its tokens, the budget,
 * the image, the prefix reuse and the photos, ACCEPT, the prefill and its read-out.
 * 1 with the logits after the prompt; 0 when the request ended here, its ERROR written
 * and its command disposed. serve_loop and serve_mux start every request here. */
static int mimo_serve_start(Model *m, Tok *tokenizer, ColiServeCommand *cmd, int *ids, float *logits,
                            float **pending, int *pending_h, int *pending_w, MimoReq *rq) {
    Cfg *c = &m->c;
    double started = now_s();
    double disk0 = m->t_disk, expert0 = m->t_expert, attn0 = m->t_attn;
    uint64_t hits0 = m->hits, miss0 = m->miss, fw0 = m->forwards;
    int n_prompt = tok_encode(tokenizer, (const char *)cmd->payload, (int)cmd->payload_bytes,
                              ids, m->ctx + 1);
    int budget = serve_budget(n_prompt, cmd->max_tokens, m->ctx, cmd->logprobs);
    if (budget < 0) {
        char message[160];
        snprintf(message, sizeof(message), "CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d",
                 n_prompt, cmd->max_tokens, m->ctx);
        coli_serve_write_error(stdout, cmd->id, n_prompt < 1 ? "EMPTY_PROMPT" : message);
        free((*pending)); (*pending) = NULL;
        coli_serve_command_dispose(cmd);
        return 0;
    }
    /* An image: its rows replace the pad ids, which must be exactly as many
     * as the tower produces. Checked before ACCEPT so a mismatch is a clean
     * 400, not a half-answered turn. */
    float *image = NULL;
    int image_rows = 0;
    if ((*pending)) {
        int pads = 0;
        for (int t = 0; t < n_prompt; t++) pads += ids[t] == c->image_token_id;
        int want = ((*pending_h) / 2) * ((*pending_w) / 2);
        if (pads != want) {
            char message[160];
            snprintf(message, sizeof(message), "BAD_IMAGE prompt has %d image pads, the %dx%d grid needs %d",
                     pads, (*pending_h), (*pending_w), want);
            coli_serve_write_error(stdout, cmd->id, message);
            free((*pending)); (*pending) = NULL;
            coli_serve_command_dispose(cmd);
            return 0;
        }
    }
    /* Where this prompt starts. A chat client resends the whole transcript:
     * if this prompt begins with the ids the state was built from, only the
     * tail is fed (COLI_KV_PREFIX=0 turns that off). A photo (SUBMIT pin=1
     * earlier) resumes a prompt that begins with it.
     *
     * With the read-out on (logprobs=k) only a photo will do. The read-out
     * owes a frame to every position it does not inherit a predictor for,
     * and a live prefix comes with none: the frames before it would simply
     * be missing. So such a prompt resumes from the deepest photo, which
     * carries the logits that predict its first fresh token, or is read out
     * from position 0. It resumes at the photo even when the live state
     * shares more: two options share the text before them, and stopping at
     * that would leave the first option token without its predictor.
     *
     * An image refuses both: the pad ids do not describe the picture. */
    int prefix_on = env_int("COLI_KV_PREFIX", 1) != 0;
    int reuse = 0;
    const float *pin_logit = NULL;
    if (!(*pending)) {
        if (cmd->logprobs > 0) reuse = pin_restore(m, ids, n_prompt, &pin_logit);
        else {
            reuse = prefix_on ? kv_prefix_reuse(&m->kvp, ids, n_prompt) : 0;
            if (!reuse) reuse = pin_restore(m, ids, n_prompt, &pin_logit);
        }
    }
    if (getenv("COLI_PREFIX_LOG"))
        fprintf(stderr, "[PREFIX] %s %d of %d prompt tokens%s\n", reuse ? "reusing" : "no reuse,", reuse,
                n_prompt, pin_logit ? " (photo)" : "");
    if (pin_logit) {
        fprintf(stderr, "[PIN] resumed from the photo of %d tokens, %d to prefill\n", reuse, n_prompt - reuse);
        fflush(stderr);
    }
    if (!reuse) model_reset(m);
    coli_serve_write_accept(stdout, cmd->id, n_prompt);
    if ((*pending)) {
        image = vision_run(m, (*pending), (*pending_h), (*pending_w), &image_rows);
        free((*pending)); (*pending) = NULL;
    }
    ImageRows img = { image, image_rows };
    Echo echo = { cmd->id, cmd->logprobs, tokenizer, pin_logit };
    prefill(m, ids + reuse, n_prompt - reuse, logits, NULL, image ? &img : NULL,
            cmd->logprobs > 0 ? &echo : NULL);
    if (image) kv_prefix_taint(&m->kvp);
    if (cmd->pin) {
        /* a photo is described by its ids, and these do not describe the picture */
        if (image) fprintf(stderr, "[PIN] no photo: the prompt carries a picture\n");
        else pin_save(m, ids, n_prompt, logits);
    }
    free(image);
    rq->started = started; rq->disk0 = disk0; rq->expert0 = expert0; rq->attn0 = attn0;
    rq->hits0 = hits0; rq->miss0 = miss0; rq->fw0 = fw0; rq->n_prompt = n_prompt; rq->budget = budget;
    return 1;
}

/* ---- several conversations at once (KV_SLOTS>1) ---------------------------------
 * The gateway's cache slots, each a conversation with caches of its own (MimoSeq). A
 * SUBMIT on a free slot starts its request at once through mimo_serve_start, on that
 * slot's caches: its prefix reuse and photos work as a lone serve's. Then every step
 * picks the next token of each active request and runs one forward over a row of each
 * (forward_rows): the matrices and the experts are read once for all of them. A
 * request's frames are a lone request's; they interleave by id. As alone, STOP ends a
 * request with DONE and CANCEL with ERROR CANCELLED. */
static MimoSeq *g_mimo_mux_seq;   /* [slots]: the conversations the Model does not hold */
static int g_mimo_mux_cur;        /* the slot the Model holds, -1 when every one is parked */

typedef struct {
    char id[COLI_SERVE_ID_CAP];
    float temperature, top_p;
    int logprobs, active, stop, cancel, limited, emitted;
    float *logits;                /* the logits the next pick reads */
    MimoReq rq;
} MimoMuxReq;

static void mimo_mux_bind(Model *m, int slot) {
    if (g_mimo_mux_cur == slot) return;
    if (g_mimo_mux_cur >= 0) mimo_seq_swap(m, &g_mimo_mux_seq[g_mimo_mux_cur]);
    if (slot >= 0) mimo_seq_swap(m, &g_mimo_mux_seq[slot]);
    g_mimo_mux_cur = slot;
}

/* A request's end, as serve_loop ends one. */
static void mimo_mux_finish(Model *m, MimoMuxReq *r) {
    r->active = 0;
    if (r->cancel) { coli_serve_write_error(stdout, r->id, "CANCELLED"); return; }
    if (r->stop) r->limited = 0;
    double wall = now_s() - r->rq.started;
    uint64_t th = m->hits - r->rq.hits0, tm = m->miss - r->rq.miss0;
    ColiServeDone done = { r->emitted, wall > 0 ? r->emitted / wall : 0.0,
                           (th + tm) ? 100.0 * th / (double)(th + tm) : 0.0, rss_gb(), r->rq.n_prompt, r->limited };
    coli_serve_write_done(stdout, r->id, &done);
    serve_line("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n", wall, r->rq.n_prompt, r->emitted,
               m->t_disk - r->rq.disk0, 0.0, m->t_expert - r->rq.expert0, m->t_attn - r->rq.attn0, 0.0,
               (unsigned long long)(m->forwards - r->rq.fw0));
    serve_hits(m);
}

/* The next token of an active request, as serve_loop picks and sends it: 1 with the
 * token when the request goes on, 0 when it ended. */
static int mimo_mux_pick(Model *m, Tok *tokenizer, MimoMuxReq *r, const int *eos, int n_eos, int *tk_out) {
    Cfg *c = &m->c;
    if (r->cancel || r->stop || r->emitted >= r->rq.budget) { mimo_mux_finish(m, r); return 0; }
    int token = sample(r->logits, c->vocab, r->temperature, r->top_p);
    for (int i = 0; i < n_eos; i++) if (token == eos[i]) { r->limited = 0; mimo_mux_finish(m, r); return 0; }
    char piece[512];
    int written = tok_decode(tokenizer, &token, 1, piece, (int)sizeof(piece));
    if (written > 0) {
        if (r->logprobs > 0) {
            char tail[1024];
            coli_logprob_tail(tail, sizeof(tail), r->logits, c->vocab, token, r->logprobs);
            coli_serve_write_data_lp(stdout, r->id, piece, (size_t)written, tail);
        } else coli_serve_write_data(stdout, r->id, piece, (size_t)written);
    }
    r->emitted++;
    if (r->emitted >= r->rq.budget) { mimo_mux_finish(m, r); return 0; }
    *tk_out = token; return 1;
}

static void serve_mux(Model *m, Tok *tokenizer, const char *dir) {
    Cfg *c = &m->c;
    int n = g_mimo_mux_slots, V = c->vocab, input_eof = 0;
    int eos[8];
    int n_eos = coli_load_stop_ids(dir, eos, 8, NULL);
    coli_serve_write_ready_caps(stdout, rss_gb(), g_vision ? "vision=1" : "vision=0");
    serve_emap(m);
    MimoMuxReq *rq = xcalloc((size_t)n, sizeof *rq, "requests");
    MimoRow *rows = xmalloc((size_t)n * sizeof *rows, "rows");
    int *tok = xmalloc((size_t)n * sizeof(int), "tokens"), *who = xmalloc((size_t)n * sizeof(int), "slots");
    float *lo = xmalloc((size_t)n * V * sizeof(float), "step logits");
    int *ids = xmalloc(((size_t)m->ctx + 1) * sizeof(int), "prompt ids");
    for (int i = 0; i < n; i++) rq[i].logits = xmalloc((size_t)V * sizeof(float), "logits");
    float *pending = NULL;
    int pending_h = 0, pending_w = 0;
    unsigned long long steps = 0, nrows = 0;
    fprintf(stderr, "[mimo] serving %d conversations at once (KV_SLOTS)\n", n);
    for (;;) {
        int active = 0; for (int i = 0; i < n; i++) active += rq[i].active;
        /* idle: wait for a command; decoding: take one only when one is there */
        if (!input_eof && (!active || coli_serve_stdin_ready())) {
            ColiServeCommand command;
            ColiServeReadResult result = coli_serve_read_command(stdin, &mimo_wire, &command);
            if (result == COLI_SERVE_READ_EOF || result == COLI_SERVE_READ_BAD_FRAME) input_eof = 1;
            else if (result == COLI_SERVE_READ_NOMEM) { coli_serve_write_error(stdout, command.id, "out of memory"); input_eof = 1; }
            else if (result == COLI_SERVE_READ_BAD_REQUEST) {
                if (command.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, command.id, "bad submit header");
                coli_serve_command_dispose(&command);
            } else if (command.kind == COLI_SERVE_COMMAND_IMAGE) {
                /* one image waits for the SUBMIT that names it; a second replaces it */
                uint64_t expected = g_vision ? (uint64_t)command.grid_h * command.grid_w * g_vision->patch_in * sizeof(float) : 0;
                if (!g_vision) coli_serve_write_error(stdout, command.id, "this container has no vision tower");
                else if (command.grid_h < 2 || command.grid_w < 2 || command.grid_h % 2 || command.grid_w % 2 ||
                         command.payload_bytes != expected)
                    coli_serve_write_error(stdout, command.id, "BAD_IMAGE payload does not match its grid");
                else {
                    free(pending);
                    pending = (float *)coli_serve_command_take_payload(&command);
                    pending_h = command.grid_h; pending_w = command.grid_w;
                }
                coli_serve_command_dispose(&command);
            } else if (command.kind == COLI_SERVE_COMMAND_CANCEL || command.kind == COLI_SERVE_COMMAND_STOP) {
                int found = 0;
                for (int i = 0; i < n; i++) if (rq[i].active && !strcmp(rq[i].id, command.id)) {
                    found = 1;
                    if (command.kind == COLI_SERVE_COMMAND_STOP) rq[i].stop = 1; else rq[i].cancel = 1;
                }
                if (!found && command.kind == COLI_SERVE_COMMAND_CANCEL) coli_serve_write_error(stdout, command.id, "NOT_FOUND");
                coli_serve_command_dispose(&command);
            } else if (command.kind != COLI_SERVE_COMMAND_SUBMIT) {
                coli_serve_command_dispose(&command);
            } else if (command.slot < 0 || command.slot >= n || rq[command.slot].active) {
                coli_serve_write_error(stdout, command.id, command.slot < 0 || command.slot >= n ? "invalid cache slot" : "SLOT_BUSY");
                coli_serve_command_dispose(&command);
            } else {
                MimoMuxReq *t = &rq[command.slot];
                float *keep = t->logits;
                memset(t, 0, sizeof *t); t->logits = keep;
                snprintf(t->id, sizeof t->id, "%s", command.id);
                t->temperature = command.temperature; t->top_p = command.top_p; t->logprobs = command.logprobs;
                mimo_mux_bind(m, command.slot);
                if (mimo_serve_start(m, tokenizer, &command, ids, t->logits, &pending, &pending_h, &pending_w, &t->rq)) {
                    t->active = 1; t->limited = 1;
                    coli_serve_command_dispose(&command);
                }
            }
        }
        active = 0; for (int i = 0; i < n; i++) active += rq[i].active;
        if (!active) { if (input_eof) break; continue; }
        int S = 0, ended = 0;
        for (int i = 0; i < n; i++) if (rq[i].active) {
            int tk;
            if (!mimo_mux_pick(m, tokenizer, &rq[i], eos, n_eos, &tk)) { ended = 1; continue; }
            rows[S] = (MimoRow){&g_mimo_mux_seq[i], g_mimo_mux_cur == i ? m->pos : g_mimo_mux_seq[i].pos};
            tok[S] = tk; who[S] = i; S++;
        }
        if (S) {
            mimo_mux_bind(m, -1);   /* every conversation parked: the rows read theirs */
            for (int s = 0; s < S; s++) rows[s].pos = g_mimo_mux_seq[who[s]].pos;
            forward_rows(m, rows, tok, S, lo);
            steps++; nrows += (unsigned long long)S;
            for (int s = 0; s < S; s++) memcpy(rq[who[s]].logits, lo + (size_t)s * V, (size_t)V * sizeof(float));
        }
        if (ended) {
            serve_emap(m);
            vkt_report("turn", m->hits, m->miss);
        }
    }
    fprintf(stderr, "[mimo] KV_SLOTS=%d: %llu decode steps, %llu rows (%.2f a step)\n", n, steps, nrows,
            steps ? (double)nrows / (double)steps : 0.0);
    mimo_mux_bind(m, 0);
    for (int i = 0; i < n; i++) free(rq[i].logits);
    free(rq); free(rows); free(tok); free(who); free(lo); free(ids); free(pending);
}

static void serve_loop(Model *m, Tok *tokenizer, const char *dir) {
    if (g_mimo_mux_slots > 1) { serve_mux(m, tokenizer, dir); return; }
    Cfg *c = &m->c;
    coli_serve_stdio_init();
    int eos[8];
    int n_eos = coli_load_stop_ids(dir, eos, 8, NULL);
    /* CAPS: the gateway offers pictures only when the tower actually loaded */
    coli_serve_write_ready_caps(stdout, rss_gb(), g_vision ? "vision=1" : "vision=0");
    serve_emap(m);
    float *logits = xmalloc((size_t)c->vocab * sizeof(float), "logits");
    int *ids = xmalloc(((size_t)m->ctx + 1) * sizeof(int), "prompt ids");
    float *pending = NULL;
    int pending_h = 0, pending_w = 0;
    for (;;) {
        ColiServeCommand command;
        ColiServeReadResult result = coli_serve_read_command(stdin, &mimo_wire, &command);
        if (result == COLI_SERVE_READ_EOF || result == COLI_SERVE_READ_BAD_FRAME) break;
        if (result == COLI_SERVE_READ_NOMEM) { coli_serve_write_error(stdout, command.id, "out of memory"); break; }
        if (result == COLI_SERVE_READ_BAD_REQUEST) {
            if (command.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, command.id, "bad submit header");
            coli_serve_command_dispose(&command);
            continue;
        }
        if (command.kind == COLI_SERVE_COMMAND_IMAGE) {
            /* One image waits for the SUBMIT that names it; a second replaces it. */
            if (!g_vision) {
                coli_serve_write_error(stdout, command.id, "this container has no vision tower");
                coli_serve_command_dispose(&command);
                continue;
            }
            uint64_t expected = (uint64_t)command.grid_h * command.grid_w * g_vision->patch_in * sizeof(float);
            if (command.grid_h < 2 || command.grid_w < 2 || command.grid_h % 2 || command.grid_w % 2 ||
                command.payload_bytes != expected) {
                coli_serve_write_error(stdout, command.id, "BAD_IMAGE payload does not match its grid");
                coli_serve_command_dispose(&command);
                continue;
            }
            free(pending);
            pending = (float *)coli_serve_command_take_payload(&command);
            pending_h = command.grid_h; pending_w = command.grid_w;
            coli_serve_command_dispose(&command);
            continue;
        }
        if (command.kind != COLI_SERVE_COMMAND_SUBMIT) {
            if (command.kind == COLI_SERVE_COMMAND_CANCEL) coli_serve_write_error(stdout, command.id, "NOT_FOUND");
            coli_serve_command_dispose(&command);
            continue;
        }
        MimoReq rq;
        if (!mimo_serve_start(m, tokenizer, &command, ids, logits, &pending, &pending_h, &pending_w, &rq)) continue;
        double started = rq.started;
        double disk0 = rq.disk0, expert0 = rq.expert0, attn0 = rq.attn0;
        uint64_t hits0 = rq.hits0, miss0 = rq.miss0, fw0 = rq.fw0;
        int n_prompt = rq.n_prompt, budget = rq.budget;
        int emitted = 0, limited = 1, cancelled = 0, done_early = 0;
        char piece[512];
        while (emitted < budget && !cancelled && !done_early) {
            int token = sample(logits, c->vocab, command.temperature, command.top_p);
            int stop = 0;
            for (int i = 0; i < n_eos; i++) if (token == eos[i]) stop = 1;
            if (stop) { limited = 0; break; }
            int written = tok_decode(tokenizer, &token, 1, piece, (int)sizeof(piece));
            if (written > 0) {
                if (command.logprobs > 0) {
                    /* the distribution the token was drawn from: sample() only reads it */
                    char tail[1024];
                    coli_logprob_tail(tail, sizeof(tail), logits, c->vocab, token, command.logprobs);
                    coli_serve_write_data_lp(stdout, command.id, piece, (size_t)written, tail);
                } else coli_serve_write_data(stdout, command.id, piece, (size_t)written);
            }
            emitted++;
            while (coli_serve_stdin_ready()) {
                ColiServeCommand control;
                ColiServeReadResult inner = coli_serve_read_command(stdin, &mimo_wire, &control);
                if (inner == COLI_SERVE_READ_EOF) { done_early = 1; break; }
                if (control.kind == COLI_SERVE_COMMAND_CANCEL && !strcmp(control.id, command.id)) cancelled = 1;
                else if (control.kind == COLI_SERVE_COMMAND_STOP && !strcmp(control.id, command.id)) { limited = 0; done_early = 1; }
                else if (control.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, control.id, "SLOT_BUSY");
                coli_serve_command_dispose(&control);
            }
            if (cancelled || done_early || emitted >= budget) break;
            prefill(m, &token, 1, logits, NULL, NULL, NULL);
        }
        if (cancelled) {
            coli_serve_write_error(stdout, command.id, "CANCELLED");
            coli_serve_command_dispose(&command);
            continue;
        }
        double wall = now_s() - started;
        uint64_t th = m->hits - hits0, tm = m->miss - miss0;
        ColiServeDone done = { emitted, wall > 0 ? emitted / wall : 0.0,
                               (th + tm) ? 100.0 * th / (double)(th + tm) : 0.0, rss_gb(), n_prompt, limited };
        coli_serve_write_done(stdout, command.id, &done);
        serve_line("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n", wall, n_prompt, emitted,
                   m->t_disk - disk0, 0.0, m->t_expert - expert0, m->t_attn - attn0, 0.0,
                   (unsigned long long)(m->forwards - fw0));
        serve_hits(m);
        serve_emap(m);
        vkt_report("turn", m->hits, m->miss);   /* the Vulkan tier's line, when it runs */
#ifdef COLI_VULKAN
        mc_report(m, "turn");
#endif
        coli_serve_command_dispose(&command);
    }
    free(ids); free(logits); free(pending);
}

/* ------------------------------------------------------------------- main ---- */

/* Once, at the end: how many products the GPU actually took, so a run that
 * asked for Vulkan and quietly stayed on the CPU says so. */
static void vk_report(const Model *m) {
#ifdef COLI_VULKAN
    if (!g_vk_ready) return;
    fprintf(stderr, "[VK] mimo: %llu matmuls on the GPU\n", coli_vk_matmul_calls());
    vkt_report("run", m->hits, m->miss);
    mc_report((Model *)m, "run");
#else
    (void)m;
#endif
}

static int parse_ids(const char *text, int *out, int cap) {
    int n = 0;
    const char *p = text;
    while (*p && n < cap) {
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[n++] = (int)v;
        p = end;
    }
    return n;
}

static int is_dir(const char *path) {
    struct stat sb;
    return stat(path, &sb) == 0 && S_ISDIR(sb.st_mode);
}

int main(int argc, char **argv) {
    coli_omp_tune_threads("mimo");
    /* Two ways in. The gateway: SNAP=<dir> mimo <cap>, SERVE=1. By hand and in
     * the tests: mimo <dir> [--ids "..." | --prompt "..."] [--ngen N] [--cap N]. */
    const char *dir = getenv("SNAP");
    int cap = env_int("MIMO_CAP", 0);
    if (env_int("SERVE", 0)) {   /* KV_SLOTS: the conversations a serve decodes at once */
        const char *ks = getenv("KV_SLOTS");
        if (ks && *ks) {
            char *end = NULL; long v = strtol(ks, &end, 10);
            if (end == ks || *end || v < 1 || v > 16) { fprintf(stderr, "KV_SLOTS must be between 1 and 16\n"); return 2; }
            g_mimo_mux_slots = (int)v;
        }
    }
    const char *ids_text = NULL, *prompt = NULL, *image_path = NULL;
    int ngen = 32, grid_h = 0, grid_w = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_text = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--ngen") && i + 1 < argc) ngen = coli_arg_int(argv[++i], "--ngen");
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = coli_arg_int(argv[++i], "--cap");
        else if (!strcmp(argv[i], "--image") && i + 1 < argc) image_path = argv[++i];
        else if (!strcmp(argv[i], "--grid") && i + 2 < argc) {
            grid_h = coli_arg_int(argv[++i], "--grid h"); grid_w = coli_arg_int(argv[++i], "--grid w"); }
        else if (is_dir(argv[i])) dir = argv[i];
        else cap = coli_arg_int(argv[i], "cache/layer");
    }
    if (!dir) {
        fprintf(stderr, "usage: %s <model dir> [--ids \"1 2 3\" | --prompt TEXT] [--ngen N] [--cap N]\n"
                        "       SNAP=<model dir> SERVE=1 %s <cap>\n", argv[0], argv[0]);
        return 2;
    }
    Model *m = xcalloc(1, sizeof(Model), "model");
    double t0 = now_s();
    if (cap <= 0) cap = 64;
    model_load(m, dir, cap);
    Cfg *c = &m->c;
    g_vision = vision_load(m, dir);
#ifdef COLI_VULKAN
    /* after the weights, so a missing device costs one line and nothing else; the
     * routed-expert tier first, then the dense matrices where coli_vk_dense() puts
     * them (on a device that shares the CPU's RAM they stay on the CPU while the
     * tier is on; COLI_VK_DENSE decides when set) */
    {
        int any_moe = 0;
        for (int i = 0; i < c->n_layers; i++) any_moe |= c->moe[i];
        const char *mx = getenv("MIMO_VK_EXPERTS");
        int tier = vkt_wanted() && any_moe && !(mx && *mx && atoi(mx) <= 0);   /* MIMO_VK_EXPERTS=0: experts on the CPU */
        g_vk_ready = coli_vk_init_env_tier("mimo", tier);
        /* COLI_VK_CHAIN: every layer's dense chain on the device (mimo_chain.h), decided
         * before the tier so the tier's budget leaves the trunk room. Not measured on a
         * MiMo checkpoint: an integrated GPU keeps it opt-in (docs/vulkan.md). */
        if (g_vk_ready) {
            g_vk_chain = coli_vk_chain_decide("mimo", tier, COLI_VK_CHAIN_UNMEASURED);
            if (g_vk_chain && g_mimo_mux_slots > 1) {
                g_vk_chain = 0;
                fprintf(stderr, "[VK] mimo: KV_SLOTS=%d: the dense chain is off (it keeps one conversation's caches on the device); "
                                "the expert tier runs every conversation's experts\n", g_mimo_mux_slots);
            }
            /* the chain's fit before any upload: its first n layers on the device, the
             * others, the head and the tower on the CPU when they do not all fit */
            if (g_vk_chain) mc_fit_start(m);
            if (g_mc_fit.L && !g_mc_fit.n) {   /* its line said the chain stays off: nothing goes up */
                g_vk_chain = 0;
                mc_refuse_from(m, 0);
                vkc_fit_placed("mimo", &g_mc_fit);
                g_mc_placed = 1;
            }
            if (g_vk_chain && !vkc_init()) g_vk_chain = 0;
        }
        mimo_dho_start(m);   /* COLI_VK_DENSE_HOST: the dense matrices on the device only, before the tier sizes its budget */
        if (g_vk_chain && vkc_fit_partial(&g_mc_fit) && !g_mc_placed) mc_place(m, NULL);   /* before the tier, too */
        if (g_vk_ready && tier) vk_tier_start(m);
        if (g_vk_ready && !vkt_ready() && !coli_vk_dense()) coli_vk_dense_decide("mimo", 0, 1);   /* no tier after all */
        if (g_vk_chain) {   /* the chain's teardown before the device's (the tier registered the device's) */
            if (!tier) atexit(coli_vk_shutdown);
            atexit(vkc_shutdown_all);
        }
    }
    if (g_vk_ready && (coli_vk_dense() || g_vk_chain) && !(g_mc_fit.L && !g_mc_fit.n)) {   /* COLI_VK_DENSE=0: the trunk stays on the CPU */
        int up = vk_dense_upload(m, coli_vk_dense());
        size_t used = 0, count = 0;
        coli_vk_mem_info(&used, &count);
        fprintf(stderr, "[VK] mimo: %d dense matrices on the GPU (%.2f GB)\n", up, used / 1e9);
    }
#endif
    int swa = 0, shown_cap = 0;
    for (int i = 0; i < c->n_layers; i++) { swa += c->swa[i]; if (c->moe[i] && !shown_cap) shown_cap = m->cache[i].cap; }
    fprintf(stderr, "[mimo] %d layers (%d sliding window %d, %d full), %d experts top-%d, cache %d/layer, "
                    "dense %s, ctx %d%s -- loaded in %.2fs\n",
            c->n_layers, swa, c->window, c->n_layers - swa, c->n_experts, c->topk, shown_cap,
            g_dense_bits == 0 ? "native" : g_dense_bits == 8 ? "int8" : "f32", m->ctx,
            g_vision ? ", vision" : "", now_s() - t0);

    char tok_path[4096];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", dir);
    int have_tok = access(tok_path, R_OK) == 0;
    Tok tokenizer;
    if (have_tok) tok_load(&tokenizer, tok_path);
    const char *seed = getenv("SEED");
    srand(seed ? (unsigned)strtoul(seed, NULL, 10) : (unsigned)time(NULL));

    if (env_int("SERVE", 0)) {
        if (!have_tok) { fprintf(stderr, "[mimo] SERVE needs %s\n", tok_path); return 1; }
        if (g_mimo_mux_slots > 1) {   /* slot 0 is the Model's own caches */
            g_mimo_mux_seq = xcalloc((size_t)g_mimo_mux_slots, sizeof *g_mimo_mux_seq, "conversations");
            for (int i = 1; i < g_mimo_mux_slots; i++) mimo_seq_alloc(m, &g_mimo_mux_seq[i]);
            g_mimo_mux_cur = 0;
        }
        serve_loop(m, &tokenizer, dir);
        vk_report(m);
        return 0;
    }
    int *ids = xmalloc(((size_t)m->ctx + 1) * sizeof(int), "ids");
    int n = 0;
    if (ids_text) n = parse_ids(ids_text, ids, m->ctx);
    else if (prompt && have_tok) n = tok_encode(&tokenizer, prompt, (int)strlen(prompt), ids, m->ctx);
    else { fprintf(stderr, "[mimo] give --ids, or --prompt with a tokenizer.json in the model dir\n"); return 2; }
    if (n < 1) { fprintf(stderr, "[mimo] empty prompt\n"); return 2; }
    if (n + ngen > m->ctx) { fprintf(stderr, "[mimo] prompt + ngen exceed the context %d\n", m->ctx); return 2; }
    float *logits = xmalloc((size_t)c->vocab * sizeof(float), "logits");
    const char *dump = getenv("MIMO_LOGITS");
    float *all = dump ? xmalloc((size_t)n * c->vocab * sizeof(float), "all logits") : NULL;
    /* --image patches.f32 --grid H W: the gateway's patches for the image-pad
     * ids already in the prompt */
    float *image = NULL;
    int image_rows = 0;
    if (image_path) {
        if (!g_vision) { fprintf(stderr, "[mimo] --image: this container has no vision tower\n"); return 2; }
        size_t want = (size_t)grid_h * grid_w * g_vision->patch_in;
        float *patches = xmalloc(want * sizeof(float), "patches");
        FILE *f = fopen(image_path, "rb");
        if (!f || fread(patches, sizeof(float), want, f) != want || fgetc(f) != EOF) {
            fprintf(stderr, "[mimo] %s does not hold %dx%d patches\n", image_path, grid_h, grid_w); return 2;
        }
        fclose(f);
        int pads = 0;
        for (int t = 0; t < n; t++) pads += ids[t] == c->image_token_id;
        if (pads != (grid_h / 2) * (grid_w / 2)) {
            fprintf(stderr, "[mimo] the prompt has %d image pads, the grid needs %d\n", pads, (grid_h / 2) * (grid_w / 2));
            return 2;
        }
        image = vision_run(m, patches, grid_h, grid_w, &image_rows);
        free(patches);
    }
    ImageRows img = { image, image_rows };
    double tp = now_s();
    prefill(m, ids, n, logits, all, image ? &img : NULL, NULL);
    double prefill_s = now_s() - tp;
    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f || fwrite(all, sizeof(float), (size_t)n * c->vocab, f) != (size_t)n * c->vocab) {
            fprintf(stderr, "[mimo] cannot write %s\n", dump); return 1;
        }
        fclose(f);
    }
    float temperature = getenv("COLI_TEMP") ? (float)atof(getenv("COLI_TEMP")) : 0.0f;
    int eos[8], n_eos = coli_load_stop_ids(dir, eos, 8, NULL);
    double td = now_s();
    int produced = 0;
    char piece[512];
    for (int g = 0; g < ngen; g++) {
        int token = sample(logits, c->vocab, temperature, 1.0f);
        int stop = 0;
        for (int i = 0; i < n_eos; i++) stop |= token == eos[i];
        if (have_tok && !ids_text) {
            int w = tok_decode(&tokenizer, &token, 1, piece, (int)sizeof(piece));
            fwrite(piece, 1, (size_t)(w > 0 ? w : 0), stdout);
        } else printf("%d ", token);
        fflush(stdout);
        produced++;
        if (stop && have_tok && !ids_text) break;
        if (g + 1 < ngen) prefill(m, &token, 1, logits, NULL, NULL, NULL);
    }
    printf("\n");
    fflush(stdout);
    double decode_s = now_s() - td;
    double total = prefill_s + decode_s;
    fprintf(stderr, "[mimo] prefill %d tokens %.2fs, decode %d tokens %.2fs (%.2f tok/s), experts %llu hits "
                    "%llu misses %.1f MB read, rss %.2f GB\n"
                    "[mimo] time: expert reads %.2fs (%.2f GB/s), expert matmuls %.2fs, attention %.2fs, "
                    "the rest (dense MLP, router, head) %.2fs\n",
            n, prefill_s, produced, decode_s, decode_s > 0 ? produced / decode_s : 0.0,
            (unsigned long long)m->hits, (unsigned long long)m->miss, m->bytes_read / 1e6, rss_gb(),
            m->t_disk, m->t_disk > 0 ? m->bytes_read / 1e9 / m->t_disk : 0.0, m->t_expert, m->t_attn,
            total - m->t_disk - m->t_expert - m->t_attn);
    vk_report(m);
    return 0;
}
