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
#include "tok.h"
#include "serve_codec.h"
#include "serve_poll.h"
#include "stop_ids.h"
#if defined(__AVX2__)
#include <immintrin.h>
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

static void dw_matmul(float *y, const float *x, int S, const DW *d) {
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

static void dw_load_bf16(DW *d, shards *S, const char *name, int O, int I) {
    st_tensor *t = need(S, name, 0, O, I);
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
}

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
        else {
            uint64_t oldest = UINT64_MAX;
            for (int s = 0; s < lc->n; s++)
                if (lc->s[s].used < stamp && lc->s[s].used < oldest) { oldest = lc->s[s].used; best = s; }
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
static void attention(Model *m, int li, const float *xn, int n, int p0, float *out) {
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

/* Route every row, then run each chosen expert once over the rows that chose it.
 * `out` [n, H] receives the weighted sum. */
static void moe(Model *m, int li, const float *xn, int n, float *out) {
    Cfg *c = &m->c;
    Layer *l = &m->L[li];
    int H = c->hidden, E = c->n_experts, K = c->topk, MI = c->moe_inter;
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
    /* the union of chosen experts, in first-use order */
    int *uni = xmalloc((size_t)E * sizeof(int), "expert union");
    int *seen = xcalloc((size_t)E, sizeof(int), "expert seen");
    int nu = 0;
    for (int i = 0; i < n * K; i++) if (!seen[sel[i]]) { seen[sel[i]] = 1; uni[nu++] = sel[i]; }
    free(seen);
    memset(out, 0, (size_t)n * H * sizeof(float));
    /* Each group of resident experts is computed in two parallel passes -- gate and
     * up of every expert, then down of every expert -- split in row blocks, instead
     * of three OpenMP regions per expert (a thousand a token on Flash, and every
     * barrier costs a scheduling round on a busy machine). The kernel inside a
     * task is the same matmul_mxfp4 on the same rows, so the numbers do not move. */
    int cap = m->cache[li].cap;
    int *row = xmalloc((size_t)n * K * sizeof(int), "expert rows index");
    float *rw = xmalloc((size_t)n * K * sizeof(float), "expert row weights");
    int *first = xmalloc((size_t)(cap + 1) * sizeof(int), "expert row offsets");
    float *xg = xmalloc((size_t)n * K * H * sizeof(float), "expert inputs");
    float *g = xmalloc((size_t)n * K * MI * sizeof(float), "expert gate");
    float *u = xmalloc((size_t)n * K * MI * sizeof(float), "expert up");
    float *y = xmalloc((size_t)n * K * H * sizeof(float), "expert out");
    void (*mm)(float *, const float *, const uint8_t *, const uint8_t *, int, int, int)
        = m->idot ? matmul_mxfp4_i8 : matmul_mxfp4;
    const int BLOCK_ROWS = 256;
    Slot *slots[4096];
    for (int b = 0; b < nu; b += cap) {
        int nb = nu - b < cap ? nu - b : cap;
        experts_ensure(m, li, uni + b, nb, slots);
        double t0 = now_s();
        int total = 0;
        for (int j = 0; j < nb; j++) {
            int e = uni[b + j];
            first[j] = total;
            for (int t = 0; t < n; t++)
                for (int q = 0; q < K; q++)
                    if (sel[(size_t)t * K + q] == e) { row[total] = t; rw[total] = wt[(size_t)t * K + q]; total++; }
        }
        first[nb] = total;
        for (int i = 0; i < total; i++) memcpy(xg + (size_t)i * H, xn + (size_t)row[i] * H, (size_t)H * sizeof(float));
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
        for (size_t i = 0; i < (size_t)total * MI; i++) g[i] = silu(g[i]) * u[i];
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
            float *dst = y + (size_t)first[j] * H;
            for (int i = 0; i < r; i++)
                memcpy(dst + (size_t)i * H + o0, scratch + (size_t)i * rows, (size_t)rows * sizeof(float));
            if (scratch != tmp) free(scratch);
        }
        /* the weighted sum, expert by expert in the order they were chosen */
        for (int j = 0; j < nb; j++)
            for (int i = first[j]; i < first[j + 1]; i++) {
                float *o = out + (size_t)row[i] * H;
                const float *yy = y + (size_t)i * H;
                for (int d = 0; d < H; d++) o[d] += rw[i] * yy[d];
            }
        m->t_expert += now_s() - t0;
    }
    free(first);
    free(xg); free(g); free(u); free(y); free(row); free(rw); free(uni); free(sel); free(wt);
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
    for (int li = 0; li < c->n_layers; li++) {
        Layer *l = &m->L[li];
        for (int t = 0; t < n; t++) rmsnorm(xn + (size_t)t * H, h + (size_t)t * H, l->ln1, H, c->eps);
        attention(m, li, xn, n, m->pos, tmp);
        for (size_t i = 0; i < (size_t)n * H; i++) h[i] += tmp[i];
        if (trace) fwrite(h, sizeof(float), (size_t)n * H, trace);
        for (int t = 0; t < n; t++) rmsnorm(xn + (size_t)t * H, h + (size_t)t * H, l->ln2, H, c->eps);
        if (c->moe[li]) moe(m, li, xn, n, tmp);
        else dense_mlp(m, l, xn, n, tmp);
        for (size_t i = 0; i < (size_t)n * H; i++) h[i] += tmp[i];
        if (trace) fwrite(h, sizeof(float), (size_t)n * H, trace);
    }
    if (trace) { fclose(trace); trace = NULL; }
    traced = 1;
    if (logits) {
        int from = all_rows ? 0 : n - 1, rows = n - from;
        for (int t = from; t < n; t++) rmsnorm(xn + (size_t)(t - from) * H, h + (size_t)t * H, m->norm, H, c->eps);
        dw_matmul(logits, xn, rows, &m->head);
    }
    kv_prefix_record(&m->kvp, ids, m->pos, n);
    m->pos += n;
    m->forwards++;
    free(h); free(xn); free(tmp);
}

/* A prompt in blocks of MIMO_CHUNK rows (default 64): the last block's last
 * row (or every row, for the teacher-forcing dump) comes back in `logits`. */
static void prefill(Model *m, const int *ids, int n, float *logits, float *all_logits,
                    const ImageRows *img) {
    int chunk = env_int("MIMO_CHUNK", 64);
    if (chunk < 1) chunk = 1;
    int used = 0;
    for (int b = 0; b < n; b += chunk) {
        int nb = n - b < chunk ? n - b : chunk;
        if (all_logits) forward(m, ids + b, nb, all_logits + (size_t)b * m->c.vocab, 1, img, &used);
        else forward(m, ids + b, nb, b + nb == n ? logits : NULL, 0, img, &used);
    }
    if (all_logits && logits) memcpy(logits, all_logits + (size_t)(n - 1) * m->c.vocab, (size_t)m->c.vocab * sizeof(float));
}

/* ----------------------------------------------------------------- vision ---- */

/* The tower, when the container carries one (visual.*): vision_run turns the
 * gateway's patches into the rows that replace the image-pad embeddings. */
typedef struct Vision Vision;
static Vision *g_vision;
#include "mimo_vision.h"

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

/* ------------------------------------------------------------------ serve ---- */

static const ColiServeWireProfile mimo_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 26,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
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
            int byte = (m->cache[li].by_expert[e] >= 0) << 6;
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

/* max_tokens is a ceiling; generation needs at least one free position */
static int serve_budget(int prompt, int requested, int context) {
    if (prompt < 1 || prompt > context) return -1;
    int budget = requested > 0 ? requested : 256;
    int room = context - prompt;
    if (room == 0) return -1;
    return budget < room ? budget : room;
}

static void serve_loop(Model *m, Tok *tokenizer, const char *dir) {
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
        double started = now_s();
        double disk0 = m->t_disk, expert0 = m->t_expert, attn0 = m->t_attn;
        uint64_t hits0 = m->hits, miss0 = m->miss, fw0 = m->forwards;
        int n_prompt = tok_encode(tokenizer, (const char *)command.payload, (int)command.payload_bytes,
                                  ids, m->ctx + 1);
        int budget = serve_budget(n_prompt, command.max_tokens, m->ctx);
        if (budget < 0) {
            char message[160];
            snprintf(message, sizeof(message), "CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d",
                     n_prompt, command.max_tokens, m->ctx);
            coli_serve_write_error(stdout, command.id, n_prompt < 1 ? "EMPTY_PROMPT" : message);
            free(pending); pending = NULL;
            coli_serve_command_dispose(&command);
            continue;
        }
        /* An image: its rows replace the pad ids, which must be exactly as many
         * as the tower produces. Checked before ACCEPT so a mismatch is a clean
         * 400, not a half-answered turn. */
        float *image = NULL;
        int image_rows = 0;
        if (pending) {
            int pads = 0;
            for (int t = 0; t < n_prompt; t++) pads += ids[t] == c->image_token_id;
            int want = (pending_h / 2) * (pending_w / 2);
            if (pads != want) {
                char message[160];
                snprintf(message, sizeof(message), "BAD_IMAGE prompt has %d image pads, the %dx%d grid needs %d",
                         pads, pending_h, pending_w, want);
                coli_serve_write_error(stdout, command.id, message);
                free(pending); pending = NULL;
                coli_serve_command_dispose(&command);
                continue;
            }
        }
        /* A chat client resends the whole transcript: if this prompt begins with
         * the ids the state was built from, only the tail is fed. An image
         * refuses it -- the pad ids do not describe the picture. */
        int prefix_on = env_int("COLI_KV_PREFIX", 1) != 0;
        int reuse = (!pending && prefix_on) ? kv_prefix_reuse(&m->kvp, ids, n_prompt) : 0;
        if (getenv("COLI_PREFIX_LOG"))
            fprintf(stderr, "[PREFIX] %s %d of %d prompt tokens\n", reuse ? "reusing" : "no reuse,", reuse, n_prompt);
        if (!reuse) model_reset(m);
        coli_serve_write_accept(stdout, command.id, n_prompt);
        if (pending) {
            image = vision_run(m, pending, pending_h, pending_w, &image_rows);
            free(pending); pending = NULL;
        }
        ImageRows img = { image, image_rows };
        prefill(m, ids + reuse, n_prompt - reuse, logits, NULL, image ? &img : NULL);
        if (image) { kv_prefix_taint(&m->kvp); free(image); }
        int emitted = 0, limited = 1, cancelled = 0, done_early = 0;
        char piece[512];
        while (emitted < budget && !cancelled && !done_early) {
            int token = sample(logits, c->vocab, command.temperature, command.top_p);
            int stop = 0;
            for (int i = 0; i < n_eos; i++) if (token == eos[i]) stop = 1;
            if (stop) { limited = 0; break; }
            int written = tok_decode(tokenizer, &token, 1, piece, (int)sizeof(piece));
            if (written > 0) coli_serve_write_data(stdout, command.id, piece, (size_t)written);
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
            prefill(m, &token, 1, logits, NULL, NULL);
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
        coli_serve_command_dispose(&command);
    }
    free(ids); free(logits); free(pending);
}

/* ------------------------------------------------------------------- main ---- */

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
        serve_loop(m, &tokenizer, dir);
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
    prefill(m, ids, n, logits, all, image ? &img : NULL);
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
        if (g + 1 < ngen) prefill(m, &token, 1, logits, NULL, NULL);
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
    return 0;
}
