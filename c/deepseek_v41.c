/* deepseek_v41.c — DeepSeek-V4.1-Flash on a machine that cannot hold it.
 *
 * 552B parameters, 510 GB on disk, and the shape of it is why this engine exists:
 *
 *   engram  203 GB  two n-gram memories (384M rows x 264 B) read by hash: at most
 *                   (max_ngram-1) * n_heads rows per layer per token, so ~13 KB of
 *                   random reads for 40% of the model. Streaming is not a compromise
 *                   here, it is the only sane way to serve it.
 *   experts 289 GB  15,360 fp4 experts of 18.8 MB; 6 of 384 routed per layer, so
 *                   4.5 GB per token, cached in RAM by the same LRU as every other
 *                   family here (GLM-5.2 reads 12.7 GB per token for comparison).
 *   dense    18 GB  attention, gates, norms, hyper-connection mixers: resident.
 *
 * No conversion step: the released checkpoint is already fp8 (dense, 32x32 ue8m0
 * block scales) and fp4 (experts, ue8m0 per 32-column group), and the fp4 layout is
 * byte-identical to the mxfp4 colibri already reads for Kimi K3, so quant.h's
 * matmul_mxfp4 is the expert kernel. What the container does need is a sidecar the
 * engine cannot compute (tools/prepare_dsv41.py): the engram bucket primes (sympy),
 * the hash multipliers (numpy PCG64) and the compressed-token map (the tokenizer's
 * normalizer). Those are tables, not weights.
 *
 * Mechanisms, each mirroring a named piece of the vendor's inference/model.py:
 *   hyper-connections    hc_mult parallel residual streams, mixed through a
 *                        doubly-stochastic matrix (Sinkhorn) — hyper_connections.h,
 *                        shared with DeepSeek V4 and GLM-5.3-Flash
 *   sliding window       every layer attends a ring of `window_size` raw KV, MQA:
 *                        one KV vector per position for all heads
 *   compressed KV        kv_source_layers pool `compress_ratio` tokens into one
 *                        latent with a softmax gate and publish it; the layers
 *                        between them read that same cache
 *   DSA indexer          index_source_layers score compressed positions and keep
 *                        index_topk of them, two-level when a candidate source is
 *                        configured
 *   engram               n-gram hash lookups gated into the residual stream
 *   MoE                  sqrtsoftplus scores, noaux_tc bias picks but does not
 *                        scale, one shared expert every token pays
 *
 * The reference this engine is held to is tools/dsv41_ref.py (torch, CPU): the
 * vendor's own forward runs on tilelang GPU kernels and cannot be an oracle here.
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
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cli_args.h"
#include "compat.h"
#include "json.h"
#include "st.h"
#include "quant.h"
#include "hyper_connections.h"
#include "tok.h"
#include "serve_codec.h"
#include "serve_poll.h"

#define V41_MAX_LAYERS 64
#define V41_MAX_ENGRAM 4
#define V41_MAX_NGRAM 8
#define V41_MAX_EHEADS 16

/* ---------------------------------------------------------------- config ---- */

typedef struct {
    int vocab, dim, n_layers, n_heads, head_dim, rope_dim;
    int q_lora, o_lora, o_groups;
    int moe_inter, n_routed, n_activated, n_shared;
    int norm_topk_prob;
    float route_scale, gate_temp, swiglu_limit, norm_eps;
    int window;
    int compress_ratio[V41_MAX_LAYERS];
    int kv_source[V41_MAX_LAYERS];      /* 1 = this layer owns a compressed KV cache */
    int index_source[V41_MAX_LAYERS];   /* 1 = this layer runs its own indexer       */
    int index_owner[V41_MAX_LAYERS];    /* which kv_source layer's caches it reads   */
    float compress_rope_theta, rope_theta, rope_factor;
    int original_seq_len, beta_fast, beta_slow;
    int index_n_heads, index_head_dim, index_topk;
    int candidate_source, candidate_topk_blocks, candidate_block_size;
    int hc_mult, hc_iters;
    float hc_eps;
    int max_positions;
    /* engram */
    int n_engram, engram_layer[V41_MAX_ENGRAM];
    int64_t engram_rows[V41_MAX_ENGRAM];
    int engram_max_ngram, engram_heads, engram_head_dim, engram_pad_id;
    int engram_cols;                     /* (max_ngram - 1) * heads */
    /* DSpark: the MTP draft head. n_mtp == 0 means the checkpoint carries none. */
    int n_mtp, spec_block, spec_noise, spec_targets[8], n_spec_targets;
    int markov_rank, spec_routed, spec_activated;
    /* vision (VL). vision_layers == 0 means a text-only container. */
    int vision_layers, vision_dim, vision_heads, vision_inter, vision_patch;
    int vision_ratio, vision_max_tokens, image_token_id;
    float vision_rope_theta;
} Cfg;

static double jnum(jval *o, const char *key, double fallback) {
    jval *v = o ? json_get(o, key) : NULL;
    return (v && v->t == J_NUM) ? v->num : fallback;
}

static int jints(jval *o, const char *key, int *out, int cap) {
    jval *v = o ? json_get(o, key) : NULL;
    if (!v || v->t != J_ARR) return 0;
    int n = v->len < cap ? v->len : cap;
    for (int i = 0; i < n; i++) out[i] = (int)v->kids[i]->num;
    return n;
}

/* The released config nests everything the text model needs under `text_config`;
 * the tiny fixture does the same, so one reader serves both. */
static void cfg_load(Cfg *c, const char *snap) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[cfg] %s: %s\n", path, strerror(errno)); exit(1); }
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "[cfg] short read on %s\n", path); exit(1); }
    text[size] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    if (!root) { fprintf(stderr, "[cfg] %s is not valid JSON\n", path); exit(1); }
    jval *t = json_get(root, "text_config");
    if (!t) t = root;   /* a flat config is accepted: the fields are the same */

    memset(c, 0, sizeof(*c));
    c->vocab      = (int)jnum(t, "vocab_size", 0);
    c->dim        = (int)jnum(t, "dim", jnum(t, "hidden_size", 0));
    c->n_layers   = (int)jnum(t, "n_layers", jnum(t, "num_hidden_layers", 0));
    c->n_heads    = (int)jnum(t, "n_heads", jnum(t, "num_attention_heads", 0));
    c->head_dim   = (int)jnum(t, "head_dim", 0);
    c->rope_dim   = (int)jnum(t, "rope_head_dim", jnum(t, "qk_rope_head_dim", 0));
    c->q_lora     = (int)jnum(t, "q_lora_rank", 0);
    c->o_lora     = (int)jnum(t, "o_lora_rank", 0);
    c->o_groups   = (int)jnum(t, "o_groups", 1);
    c->moe_inter  = (int)jnum(t, "moe_inter_dim", jnum(t, "moe_intermediate_size", 0));
    c->n_routed   = (int)jnum(t, "n_routed_experts", 0);
    c->n_activated= (int)jnum(t, "n_activated_experts", jnum(t, "num_experts_per_tok", 0));
    c->n_shared   = (int)jnum(t, "n_shared_experts", 1);
    c->norm_topk_prob = (int)jnum(t, "norm_topk_prob", 1);
    c->route_scale= (float)jnum(t, "route_scale", jnum(t, "routed_scaling_factor", 1.0));
    c->gate_temp  = (float)jnum(t, "gate_temp", 1.0);
    c->swiglu_limit = (float)jnum(t, "swiglu_limit", 0.0);
    c->norm_eps   = (float)jnum(t, "norm_eps", jnum(t, "rms_norm_eps", 1e-6));
    c->window     = (int)jnum(t, "window_size", jnum(t, "sliding_window", 128));
    c->compress_rope_theta = (float)jnum(t, "compress_rope_theta", 40000.0);
    c->rope_theta = (float)jnum(t, "rope_theta", 10000.0);
    c->rope_factor= (float)jnum(t, "rope_factor", 1.0);
    c->original_seq_len = (int)jnum(t, "original_seq_len", 0);
    c->beta_fast  = (int)jnum(t, "beta_fast", 32);
    c->beta_slow  = (int)jnum(t, "beta_slow", 1);
    c->index_n_heads = (int)jnum(t, "index_n_heads", 0);
    c->index_head_dim= (int)jnum(t, "index_head_dim", 0);
    c->index_topk = (int)jnum(t, "index_topk", 0);
    c->candidate_source = (int)jnum(t, "candidate_source_layer",
                                    jnum(t, "candidate_source_layer_id", -1));
    c->candidate_topk_blocks = (int)jnum(t, "candidate_topk_blocks", 0);
    c->candidate_block_size  = (int)jnum(t, "candidate_block_size", 0);
    c->hc_mult    = (int)jnum(t, "hc_mult", 1);
    c->hc_iters   = (int)jnum(t, "hc_sinkhorn_iters", 20);
    c->hc_eps     = (float)jnum(t, "hc_eps", 1e-6);
    c->max_positions = (int)jnum(t, "max_seq_len", jnum(t, "max_position_embeddings", 4096));
    /* YaRN, as the vendor spells it: rope_scaling.factor when present */
    jval *scaling = json_get(t, "rope_scaling");
    if (scaling && scaling->t == J_OBJ) {
        c->rope_factor = (float)jnum(scaling, "factor", c->rope_factor);
        c->beta_fast = (int)jnum(scaling, "beta_fast", c->beta_fast);
        c->beta_slow = (int)jnum(scaling, "beta_slow", c->beta_slow);
        c->original_seq_len = (int)jnum(scaling, "original_max_position_embeddings",
                                        c->original_seq_len);
    }

    if (c->n_layers < 1 || c->n_layers > V41_MAX_LAYERS) {
        fprintf(stderr, "[cfg] n_layers %d out of range\n", c->n_layers); exit(1); }
    int ratios[V41_MAX_LAYERS] = {0};
    int nr = jints(t, "compress_ratios", ratios, V41_MAX_LAYERS);
    if (nr < c->n_layers) {
        fprintf(stderr, "[cfg] compress_ratios has %d entries for %d layers\n",
                nr, c->n_layers); exit(1); }
    for (int i = 0; i < c->n_layers; i++) c->compress_ratio[i] = ratios[i];
    int src[V41_MAX_LAYERS];
    int n = jints(t, "kv_source_layers", src, V41_MAX_LAYERS);
    if (!n) n = jints(t, "kv_source_layer_ids", src, V41_MAX_LAYERS);
    for (int i = 0; i < n; i++) if (src[i] >= 0 && src[i] < c->n_layers) c->kv_source[src[i]] = 1;
    n = jints(t, "index_source_layers", src, V41_MAX_LAYERS);
    if (!n) n = jints(t, "index_source_layer_ids", src, V41_MAX_LAYERS);
    for (int i = 0; i < n; i++) if (src[i] >= 0 && src[i] < c->n_layers) c->index_source[src[i]] = 1;
    /* Who owns the caches a layer reads: the nearest kv_source at or before it.
     * The vendor keeps this implicit in a module-level object that the source
     * writes and everyone after reads; making it explicit here means a layer can
     * be served without replaying the ones before it. */
    int owner = -1;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->kv_source[i]) owner = i;
        c->index_owner[i] = owner;
    }
    /* DSpark. The released config spells the activated count two ways depending on
     * which file it comes from (config.json says dspark_num_experts_per_tok,
     * inference/config.json says dspark_n_activated_experts); read either. */
    c->n_mtp = (int)jnum(t, "n_mtp_layers", 0);
    c->spec_block = (int)jnum(t, "dspark_block_size", 0);
    c->spec_noise = (int)jnum(t, "dspark_noise_token_id", 0);
    c->n_spec_targets = jints(t, "dspark_target_layer_ids", c->spec_targets, 8);
    c->markov_rank = (int)jnum(t, "dspark_markov_rank", 0);
    c->spec_routed = (int)jnum(t, "dspark_n_routed_experts", c->n_routed);
    c->spec_activated = (int)jnum(t, "dspark_n_activated_experts",
                                 jnum(t, "dspark_num_experts_per_tok", c->n_activated));
    if (c->spec_block <= 0 || c->n_spec_targets <= 0) c->n_mtp = 0;

    jval *vision = json_get(root, "vision_config");
    if (vision && vision->t == J_OBJ) {
        c->vision_layers = (int)jnum(vision, "num_hidden_layers", 0);
        c->vision_dim    = (int)jnum(vision, "hidden_size", 0);
        c->vision_heads  = (int)jnum(vision, "num_attention_heads", 0);
        c->vision_inter  = (int)jnum(vision, "intermediate_size", 0);
        c->vision_patch  = (int)jnum(vision, "patch_size", 0);
        c->vision_ratio  = (int)jnum(vision, "downsample_ratio", 1);
        c->vision_max_tokens = (int)jnum(vision, "max_image_tokens", 0);
        c->vision_rope_theta = (float)jnum(vision, "rope_theta", 10000.0);
    }
    c->image_token_id = (int)jnum(root, "image_token_id", -1);
    json_free(root); free(arena); free(text);
}

/* ------------------------------------------------------- weights in RAM ---- */

/* The dense trunk stays in the checkpoint's own formats: dequantizing fp8 to f32 at
 * load would turn 18 GB into 72 GB, which is the whole point of not doing it. Each
 * kernel below decodes as it multiplies. */
typedef struct { uint8_t *q; uint8_t *s; int O, I; } W8;   /* e4m3 + 32x32 ue8m0 */
typedef struct { uint16_t *w; int O, I; } WB;              /* bf16                */
typedef struct { float *w; int64_t n; } WF;                /* f32                 */

#define FP8_TILE 32

/* V41_TRACE=1 prints a checksum of the tensors the reference prints too, so a
 * divergence is located by diffing two columns rather than by reading two forwards. */
static int g_trace = 0;
static void trace(const char *name, int layer, const float *v, int n) {
    if (!g_trace) return;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v[i];
    fprintf(stderr, "T %-10s L%-2d n=%-5d sum=%+.6f [%+.6f %+.6f %+.6f]\n",
            name, layer, n, sum, v[0], n > 1 ? v[1] : 0.0f, n > 2 ? v[2] : 0.0f);
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

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *xmalloc(size_t bytes, const char *what) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "OOM allocating %s (%zu bytes)\n", what, bytes); exit(1); }
    return p;
}

static void w8_load(shards *S, W8 *w, const char *name, int O, int I) {
    char scale_name[512];
    snprintf(scale_name, sizeof(scale_name), "%.*s.scale",
             (int)(strlen(name) - strlen(".weight")), name);
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "%s: missing\n", name); exit(1); }
    if (t->nbytes != (int64_t)O * I) {
        fprintf(stderr, "%s: %lld bytes, expected %lld for [%d, %d] fp8\n",
                name, (long long)t->nbytes, (long long)O * I, O, I); exit(1); }
    int tiles = ((O + FP8_TILE - 1) / FP8_TILE) * ((I + FP8_TILE - 1) / FP8_TILE);
    w->O = O; w->I = I;
    w->q = xmalloc((size_t)O * I, name);
    w->s = xmalloc((size_t)tiles, scale_name);
    st_read_raw_cap(S, name, w->q, (int64_t)O * I, 0);
    st_read_raw_cap(S, scale_name, w->s, tiles, 0);
}

static void wb_load(shards *S, WB *w, const char *name, int O, int I) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "%s: missing\n", name); exit(1); }
    if (t->nbytes != (int64_t)O * I * 2) {
        fprintf(stderr, "%s: %lld bytes, expected %lld for [%d, %d] bf16\n",
                name, (long long)t->nbytes, (long long)O * I * 2, O, I); exit(1); }
    w->O = O; w->I = I;
    w->w = xmalloc((size_t)O * I * 2, name);
    st_read_raw_cap(S, name, w->w, (int64_t)O * I * 2, 0);
}

static void wf_load(shards *S, WF *w, const char *name, int64_t n) {
    w->n = n;
    w->w = xmalloc((size_t)n * sizeof(float), name);
    /* st_read_f32 widens bf16/f16 as well, so a checkpoint that stores one of these
     * small tensors in bf16 rather than f32 still loads. */
    if (st_read_f32(S, name, w->w, 0) != n) {
        fprintf(stderr, "%s: expected %lld floats\n", name, (long long)n); exit(1); }
}

/* 2^(byte - 127) without a call: ue8m0 is a bare exponent, so the byte lands straight
 * in an IEEE exponent field. byte 0 and 255 (denormal / inf) never occur in a real
 * checkpoint and decode to +0 and +inf here, matching quant.h's mxfp4 path. */
static inline float ue8m0(uint8_t byte) {
    union { uint32_t u; float f; } value;
    value.u = (uint32_t)byte << 23;
    return value.f;
}

/* y[O] = W [O, I] x[I], W in e4m3 with one ue8m0 scale per 32x32 tile. */
static void mv8(float *y, const W8 *w, const float *x) {
    int I = w->I, tiles_i = (I + FP8_TILE - 1) / FP8_TILE;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < w->O; o++) {
        const uint8_t *row = w->q + (size_t)o * I;
        const uint8_t *scale = w->s + (size_t)(o / FP8_TILE) * tiles_i;
        float sum = 0.0f;
        for (int base = 0; base < I; base += FP8_TILE) {
            int width = I - base < FP8_TILE ? I - base : FP8_TILE;
            float tile = ue8m0(scale[base / FP8_TILE]);
            float part = 0.0f;
            for (int i = 0; i < width; i++) part += e4m3_decode(row[base + i]) * x[base + i];
            sum += part * tile;
        }
        y[o] = sum;
    }
}

static void mvb(float *y, const WB *w, const float *x) {
    int I = w->I;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < w->O; o++) {
        const uint16_t *row = w->w + (size_t)o * I;
        float sum = 0.0f;
        for (int i = 0; i < I; i++) sum += bf16_to_f32(row[i]) * x[i];
        y[o] = sum;
    }
}

/* ------------------------------------------------------------- engram ------ */

/* One n-gram memory. The table itself never enters RAM: `lookup` reads the row it
 * needs (head_dim bytes of e4m3 plus head_dim/32 scale bytes) and keeps it in a small
 * LRU, because n-gram traffic is Zipfian -- the common 2-grams repeat constantly. */
typedef struct {
    int fd_w, fd_s;
    int64_t off_w, off_s, rows;
    /* row cache: open addressing on the row id, LRU by a clock */
    int cap, mask;
    int64_t *key;      /* row id, -1 for empty */
    uint64_t *used;
    float *value;      /* cap x head_dim */
    uint64_t clock, hits, misses;
} EngramTable;

typedef struct {
    int active;
    int max_ngram, heads, head_dim, cols, pad_id;
    int64_t primes[V41_MAX_ENGRAM][V41_MAX_NGRAM][V41_MAX_EHEADS];
    int64_t offsets[V41_MAX_ENGRAM][V41_MAX_NGRAM * V41_MAX_EHEADS];
    int64_t multipliers[V41_MAX_ENGRAM][V41_MAX_NGRAM];
    int layer_of[V41_MAX_ENGRAM];
    int n_layers;
    int32_t *token_map;
    int token_map_len;
    EngramTable table[V41_MAX_ENGRAM];
    /* compressed ids of every position seen so far: the n-gram of a position looks
     * back over these, and an image position is DEAD (-1) so no n-gram spans it */
    int32_t *history;
    int history_len, history_cap;
} Engram;

#define ENGRAM_DEAD (-1)

static void engram_load_sidecar(Engram *e, const char *snap) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/dsv41_engram.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) { e->active = 0; return; }   /* a text-only container without engram is legal */
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = xmalloc((size_t)size + 1, "engram sidecar");
    if (fread(text, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "[engram] short read on %s\n", path); exit(1); }
    text[size] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    if (!root) { fprintf(stderr, "[engram] %s is not valid JSON\n", path); exit(1); }

    e->max_ngram = (int)jnum(root, "max_ngram_size", 0);
    e->heads     = (int)jnum(root, "n_heads", 0);
    e->head_dim  = (int)jnum(root, "head_dim", 0);
    e->pad_id    = (int)jnum(root, "pad_id", 0);
    e->cols      = (e->max_ngram - 1) * e->heads;
    if (e->max_ngram > V41_MAX_NGRAM || e->heads > V41_MAX_EHEADS) {
        fprintf(stderr, "[engram] layout %dx%d exceeds the compiled maximum\n",
                e->max_ngram, e->heads); exit(1); }
    jval *ids = json_get(root, "layer_ids");
    e->n_layers = ids && ids->t == J_ARR ? ids->len : 0;
    if (e->n_layers > V41_MAX_ENGRAM) {
        fprintf(stderr, "[engram] %d tables exceed the compiled maximum\n", e->n_layers); exit(1); }
    for (int i = 0; i < e->n_layers; i++) e->layer_of[i] = (int)ids->kids[i]->num;
    jval *primes = json_get(root, "primes");
    jval *offsets = json_get(root, "offsets");
    jval *multipliers = json_get(root, "multipliers");
    if (!primes || !offsets || !multipliers) {
        fprintf(stderr, "[engram] sidecar lacks primes/offsets/multipliers\n"); exit(1); }
    for (int layer = 0; layer < e->n_layers; layer++) {
        jval *rows = primes->kids[layer];
        for (int n = 0; n < rows->len && n < V41_MAX_NGRAM; n++)
            for (int h = 0; h < rows->kids[n]->len && h < V41_MAX_EHEADS; h++)
                e->primes[layer][n][h] = (int64_t)rows->kids[n]->kids[h]->num;
        jval *off = offsets->kids[layer];
        for (int i = 0; i < off->len && i < V41_MAX_NGRAM * V41_MAX_EHEADS; i++)
            e->offsets[layer][i] = (int64_t)off->kids[i]->num;
        jval *mult = multipliers->kids[layer];
        for (int i = 0; i < mult->len && i < V41_MAX_NGRAM; i++) {
            /* Decimal strings on purpose. The multipliers run to int64 / compressed
             * vocabulary, which is past the 53 bits a double keeps exactly; read as a
             * number, a rounded multiplier would hash every n-gram into a different --
             * and perfectly valid-looking -- row. Refuse rather than round. */
            jval *entry = mult->kids[i];
            if (entry->t != J_STR) {
                fprintf(stderr, "[engram] multipliers must be decimal strings; this "
                                "sidecar stores them as JSON numbers, which cannot hold "
                                "them exactly. Regenerate it with tools/prepare_dsv41.py\n");
                exit(1);
            }
            errno = 0;
            char *end = NULL;
            long long value = strtoll(entry->str, &end, 10);
            if (errno || !end || *end) {
                fprintf(stderr, "[engram] multiplier %s is not an integer\n", entry->str);
                exit(1);
            }
            e->multipliers[layer][i] = (int64_t)value;
        }
    }
    jval *map = json_get(root, "token_map");
    if (!map || map->t != J_ARR) { fprintf(stderr, "[engram] sidecar lacks token_map\n"); exit(1); }
    e->token_map_len = map->len;
    e->token_map = xmalloc((size_t)map->len * sizeof(int32_t), "engram token map");
    for (int i = 0; i < map->len; i++) e->token_map[i] = (int32_t)map->kids[i]->num;
    e->active = 1;
    json_free(root); free(arena); free(text);
}

static void engram_table_open(EngramTable *t, shards *S, int layer, int head_dim, int cache_rows) {
    char name[256];
    snprintf(name, sizeof(name), "layers.%d.engram.embed.weight", layer);
    st_tensor *w = st_find(S, name);
    snprintf(name, sizeof(name), "layers.%d.engram.embed.scale", layer);
    st_tensor *s = st_find(S, name);
    if (!w || !s) { fprintf(stderr, "[engram] layer %d has no table\n", layer); exit(1); }
    t->fd_w = w->fd; t->off_w = w->off;
    t->fd_s = s->fd; t->off_s = s->off;
    t->rows = w->nbytes / head_dim;
    int cap = 1;
    while (cap < cache_rows) cap <<= 1;
    t->cap = cap; t->mask = cap - 1;
    t->key = xmalloc((size_t)cap * sizeof(int64_t), "engram row keys");
    t->used = xmalloc((size_t)cap * sizeof(uint64_t), "engram row clocks");
    t->value = xmalloc((size_t)cap * head_dim * sizeof(float), "engram row cache");
    for (int i = 0; i < cap; i++) { t->key[i] = -1; t->used[i] = 0; }
    t->clock = t->hits = t->misses = 0;
}

/* The row for `id`, dequantized. Reads through a direct-mapped-with-probe cache: on a
 * miss the row and its scales are one pread each, 264 bytes for the released table. */
static const float *engram_row(EngramTable *t, int64_t id, int head_dim) {
    int groups = head_dim / 32;
    int slot = (int)((uint64_t)(id * 0x9E3779B97F4A7C15ull) >> 40) & t->mask;
    int victim = slot;
    for (int probe = 0; probe < 4; probe++) {
        int at = (slot + probe) & t->mask;
        if (t->key[at] == id) { t->used[at] = ++t->clock; t->hits++;
                                return t->value + (size_t)at * head_dim; }
        if (t->key[at] == -1) { victim = at; goto fill; }
        if (t->used[at] < t->used[victim]) victim = at;
    }
fill:
    t->misses++;
    {
        uint8_t bytes[512], scales[64];
        if (head_dim > (int)sizeof(bytes) || groups > (int)sizeof(scales)) {
            fprintf(stderr, "[engram] head_dim %d exceeds the row buffer\n", head_dim); exit(1); }
        st_pread_full(t->fd_w, bytes, head_dim, t->off_w + id * head_dim, "engram row");
        st_pread_full(t->fd_s, scales, groups, t->off_s + id * groups, "engram scale");
        float *out = t->value + (size_t)victim * head_dim;
        for (int g = 0; g < groups; g++) {
            float scale = ue8m0(scales[g]);
            for (int i = 0; i < 32; i++) out[g * 32 + i] = e4m3_decode(bytes[g * 32 + i]) * scale;
        }
        t->key[victim] = id; t->used[victim] = ++t->clock;
        return out;
    }
}

/* engram.py NgramHashState: the ids of the n-grams ending at `pos`, for one table.
 * `out` receives `cols` row ids. */
static void engram_hash(const Engram *e, int table, int pos, int64_t *out) {
    int64_t tokens[V41_MAX_NGRAM];
    int blocked = 0;
    for (int shift = 0; shift < e->max_ngram; shift++) {
        int at = pos - shift;
        int32_t source = at >= 0 ? e->history[at] : ENGRAM_DEAD;
        if (at < 0 || source == ENGRAM_DEAD) blocked = 1;
        tokens[shift] = blocked ? e->pad_id : source;
    }
    /* XOR the multiplied ids one lookback at a time: after step i the running value is
     * the hash of the (i+1)-gram, and each lands in its own prime-sized bucket range.
     * int64 throughout -- the multipliers are chosen so id * multiplier cannot wrap. */
    int64_t rolling = tokens[0] * e->multipliers[table][0];
    int col = 0;
    for (int i = 1; i < e->max_ngram; i++) {
        rolling ^= tokens[i] * e->multipliers[table][i];
        for (int h = 0; h < e->heads; h++, col++) {
            int64_t prime = e->primes[table][i - 1][h];
            int64_t bucket = prime > 0 ? rolling % prime : 0;
            if (bucket < 0) bucket += prime;          /* C99 truncates toward zero */
            out[col] = bucket + e->offsets[table][col];
        }
    }
}

/* `image` may be NULL (all text) or one byte per token: an image position is DEAD, so
 * no n-gram spans it -- model.py passes ~image_mask as the engram's token_mask for the
 * same reason. */
static void engram_push(Engram *e, const int *ids, int n, const uint8_t *image) {
    if (e->history_len + n > e->history_cap) {
        int cap = e->history_cap ? e->history_cap : 1024;
        while (cap < e->history_len + n) cap *= 2;
        e->history = realloc(e->history, (size_t)cap * sizeof(int32_t));
        if (!e->history) { fprintf(stderr, "OOM growing the engram history\n"); exit(1); }
        e->history_cap = cap;
    }
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        int32_t compressed = ENGRAM_DEAD;
        if (!(image && image[i]) && id >= 0 && id < e->token_map_len)
            compressed = e->token_map[id];
        e->history[e->history_len++] = compressed;
    }
}

/* -------------------------------------------------------------- model ------ */

/* One routed expert, resident in a cache slot. The three matrices keep the
 * checkpoint's fp4 packing and their ue8m0 group scales: quant.h's matmul_mxfp4
 * consumes exactly this, so nothing is unpacked on the way in. */
typedef struct {
    int eid;
    uint8_t *w1, *w3, *w2;      /* packed nibbles */
    uint8_t *s1, *s3, *s2;      /* ue8m0, one byte per 32-column group */
    uint64_t used;
} Slot;

typedef struct { Slot *slot; int n, cap; } LCache;

typedef struct {
    W8 wq_a, wq_b, wkv, wo_a, wo_b;
    WF q_norm, kv_norm, attn_sink, attn_norm, ffn_norm;
    WF hc_attn_fn, hc_ffn_fn, hc_attn_base, hc_ffn_base, hc_attn_scale, hc_ffn_scale;
    WB comp_wkv, comp_wgate;
    WF comp_norm;
    W8 idx_wq_b;
    WB idx_wk, idx_wproj;
    WF idx_knorm;
    WB gate_w;
    WF gate_bias;
    W8 sh_w1, sh_w3, sh_w2;
    /* engram (only on engram layers) */
    W8 eng_wkv;
    WF eng_q, eng_k;
    int engram_index;           /* which table, -1 when this layer has none */
    /* per-layer state */
    float *window;              /* [window_size, head_dim] ring of raw KV */
    /* Which position each ring slot holds, -1 when it holds nothing yet. A single
     * decode step needs only "slot i is position i until the ring wraps", but a
     * speculative step writes several positions at once and then keeps only some of
     * them: the slots of the rejected drafts still hold their keys, and the only thing
     * that makes them unreadable is knowing which position they came from. */
    int *window_pos;
    float *ckv;                 /* [max_pos/ratio, head_dim] compressed KV (owners) */
    float *ikey;                /* [max_pos/ratio, index_head_dim] index keys (owners) */
    float *cstate_kv, *cstate_score;   /* compressor group still filling */
    /* What a speculative step displaced, one entry per row of the batch: the ring slot
     * it overwrote and the compressor slot it overwrote. Both are addressed modulo
     * something (the window, the compression ratio), so a draft several positions ahead
     * lands on a slot a committed position still needs -- the key one window back, or
     * the earlier half of a group that has not closed yet. Rejected rows put these
     * back. */
    float *ring_save, *cstate_save_kv, *cstate_save_score;
    int *ring_save_pos, *cstate_save_slot;
} Layer;

/* DSpark: the MTP draft head, under `mtp.*` in the checkpoint.
 *
 * Three stages of the same block the backbone uses, with three differences: the
 * attention never compresses (it is window-only, and the window holds the MAIN
 * stream's keys), the expert set is its own and smaller, and the last stage carries
 * the heads that turn one hidden state into a block of guesses -- a Markov head that
 * biases each position by the token before it, and a confidence head nobody has to
 * trust, because the main model verifies every draft before a single one is emitted.
 *
 * The vendor ships the module and no loop: `forward_spec` exists, nothing calls it.
 * The accept/reject loop here is therefore ours, and it is written to be exact rather
 * than clever -- a token that comes out of a draft is the token the sequential decode
 * would have produced, or it is not emitted. */
typedef struct {
    int active;                 /* the stages loaded                              */
    Layer *stage;
    LCache *cache;              /* one expert cache per stage, its own expert set  */
    W8 main_proj;
    WF main_norm, norm;
    WB markov_embed, markov_out, conf_proj;
    uint64_t proposed, accepted, rounds;
    /* Drafting is not free: a round reads the stages' experts whether or not anything
     * is accepted. When acceptance collapses -- a different domain, a long verbatim
     * quote, anything the head has no signal on -- the drafts are pure disk tax, so
     * they pause for a while and the guard re-measures. */
    int pause, min_accept, max_verify;
    uint64_t window_prop, window_acc;
} Spec;

typedef struct {
    Cfg c;
    shards S;
    Engram engram;
    Layer *L;
    WB embed, head;
    WF norm;
    LCache *cache;
    int ecap;                   /* expert slots per layer */
    uint64_t clock, hits, miss;
    double t_disk, t_expert, t_attn, t_engram, t_spec;
    uint64_t forwards, expert_bytes;
    /* rope tables: [max_pos, rope_dim/2] cos/sin pairs, one per theta in use */
    float *rope_window, *rope_compress;
    int pos;                    /* positions consumed so far */
    /* candidate mask published by the candidate source layer, one row per query */
    uint8_t *candidates;
    int candidate_width, candidate_rows;
    /* What an index source publishes for the layers after it (model.py's
     * shared_attn.topk_idxs): the compressed positions each query keeps, already
     * shifted by the window offset. A layer that compresses but does not index reads
     * this instead of scoring again -- that is the whole point of index_source_layers
     * being a subset. Persisting across forwards is deliberate, as in the vendor. */
    int *shared_topk;
    int shared_topk_rows, shared_topk_width;
    /* The index keys an owner last published, and the layer that owns them.
     *
     * model.py keeps this in one global slot and republishes it only when a layer
     * COMPLETES a compression group (`if self.owns_k and latent is not None`). On a
     * decode step where a ratio-2 layer's group is still filling, that layer therefore
     * scores its queries against whichever cache was published last -- in the released
     * config, layer 20's, whose rows are ratio-1 latents and mean something else.
     *
     * That is the released behaviour and every published number for this model was
     * produced with it, so it is what this engine reproduces by default and what the
     * oracle in tools/dsv41_ref.py encodes. V41_INDEX_OWNER=1 selects the reading the
     * architecture implies instead -- each layer against its own owner's keys -- which
     * is a different model, not a bug fix, and is why it is not the default. */
    const float *published_index_k;
    int published_index_layer;
    /* A speculative step runs several positions in one pass, and the slot above is
     * per-STEP state in the vendor: to stay equal to the same positions decoded one at
     * a time, each row has to read what was published as of its own position. These
     * three hold the schedule -- which layer publishes at each sub-step, computed
     * before the layers run because a layer AFTER this one still publishes BEFORE this
     * one's later rows -- and the pointer the step began with. */
    int *pub_layer;
    int pub_rows;
    const float *pub_before;
    int pub_before_layer;
    uint8_t **ehit;             /* experts routed this turn (dashboard HITS) */
    struct Vision *vision;      /* the VL tower, NULL on a text-only container */
    Spec spec;
    /* the DSpark input the last forward produced: [rows, n_targets * dim] */
    float *main_hidden;
    int main_hidden_rows;
    /* where the last forward started and how many of its rows survived it: a
     * speculative step keeps only the prefix that verified, and the next draft has to
     * seed the stages with exactly the positions that were committed */
    int last_start, last_rows;
} Model;

/* ------------------------------------------------------------ rope --------- */

static float rope_correction_dim(float rotations, int dim, float base, int max_seq) {
    return dim * logf(max_seq / (rotations * 2.0f * (float)M_PI)) / (2.0f * logf(base));
}

/* model.py precompute_freqs_cis, YaRN included, materialized as cos/sin pairs. */
static float *rope_table(int dim, int seqlen, int original_seq_len, float base,
                         float factor, int beta_fast, int beta_slow) {
    int half = dim / 2;
    float *freqs = xmalloc((size_t)half * sizeof(float), "rope freqs");
    for (int i = 0; i < half; i++)
        freqs[i] = 1.0f / powf(base, (float)(2 * i) / (float)dim);
    if (original_seq_len > 0 && seqlen > original_seq_len) {
        float low = floorf(rope_correction_dim((float)beta_fast, dim, base, original_seq_len));
        float high = ceilf(rope_correction_dim((float)beta_slow, dim, base, original_seq_len));
        if (low < 0) low = 0;
        if (high > dim - 1) high = dim - 1;
        if (low == high) high += 0.001f;
        for (int i = 0; i < half; i++) {
            float ramp = ((float)i - low) / (high - low);
            if (ramp < 0) ramp = 0;
            if (ramp > 1) ramp = 1;
            float smooth = 1.0f - ramp;
            freqs[i] = freqs[i] / factor * (1.0f - smooth) + freqs[i] * smooth;
        }
    }
    float *table = xmalloc((size_t)seqlen * half * 2 * sizeof(float), "rope table");
    for (int pos = 0; pos < seqlen; pos++)
        for (int i = 0; i < half; i++) {
            float angle = (float)pos * freqs[i];
            table[((size_t)pos * half + i) * 2 + 0] = cosf(angle);
            table[((size_t)pos * half + i) * 2 + 1] = sinf(angle);
        }
    free(freqs);
    return table;
}

/* Rotate the trailing `rope_dim` entries of `v` in place, as pairs. `inverse` undoes
 * the rotation, which the output projection needs. */
static void rope_apply(float *v, const float *table, int pos, int rope_dim, int inverse) {
    int half = rope_dim / 2;
    const float *row = table + (size_t)pos * half * 2;
    for (int i = 0; i < half; i++) {
        float cosine = row[i * 2 + 0], sine = row[i * 2 + 1];
        if (inverse) sine = -sine;
        float a = v[2 * i], b = v[2 * i + 1];
        v[2 * i]     = a * cosine - b * sine;
        v[2 * i + 1] = a * sine   + b * cosine;
    }
}

static void rms_into(float *out, const float *x, const float *weight, int n, float eps) {
    double square = 0.0;
    for (int i = 0; i < n; i++) square += (double)x[i] * x[i];
    float inverse = 1.0f / sqrtf((float)(square / n) + eps);
    for (int i = 0; i < n; i++) out[i] = weight[i] * x[i] * inverse;
}

/* --------------------------------------------------------- expert cache ---- */

static void expert_read(Model *m, const char *kind, int layer, int eid, Slot *s) {
    Cfg *c = &m->c;
    char name[256];
    int64_t packed_gate = (int64_t)c->moe_inter * c->dim / 2;
    int64_t packed_down = (int64_t)c->dim * c->moe_inter / 2;
    int64_t scales_gate = (int64_t)c->moe_inter * (c->dim / 32);
    int64_t scales_down = (int64_t)c->dim * (c->moe_inter / 32);
    double started = now_s();
    struct { const char *suffix; uint8_t *bytes; int64_t size; uint8_t *scale; int64_t scale_size; } part[3] = {
        { "w1", s->w1, packed_gate, s->s1, scales_gate },
        { "w3", s->w3, packed_gate, s->s3, scales_gate },
        { "w2", s->w2, packed_down, s->s2, scales_down },
    };
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof(name), "%s.%d.ffn.experts.%d.%s.weight", kind, layer, eid, part[i].suffix);
        st_read_raw_cap(&m->S, name, part[i].bytes, part[i].size, 1);
        snprintf(name, sizeof(name), "%s.%d.ffn.experts.%d.%s.scale", kind, layer, eid, part[i].suffix);
        st_read_raw_cap(&m->S, name, part[i].scale, part[i].scale_size, 1);
        m->expert_bytes += part[i].size + part[i].scale_size;
    }
    s->eid = eid;
    m->t_disk += now_s() - started;
}

/* One byte per expert: routed in this turn or not, for the dashboard's HITS line. */
static void ehit_mark(Model *m, int layer, int eid) {
    Cfg *c = &m->c;
    if (!m->ehit) {
        m->ehit = xmalloc((size_t)c->n_layers * sizeof(uint8_t *), "expert hit map");
        for (int i = 0; i < c->n_layers; i++) {
            m->ehit[i] = xmalloc((size_t)c->n_routed, "expert hit row");
            memset(m->ehit[i], 0, (size_t)c->n_routed);
        }
    }
    if (layer >= 0 && layer < c->n_layers && eid >= 0 && eid < c->n_routed) m->ehit[layer][eid] = 1;
}

/* `kind` is the checkpoint namespace, "layers" for the backbone and "mtp" for a
 * DSpark stage: same slot shapes, same LRU, a different set of experts. Only the
 * backbone's routing reaches the dashboard's grid -- the draft head has its own,
 * smaller expert set, and a row of it would not line up with anything. */
static Slot *expert_slot_at(Model *m, LCache *cache, const char *kind, int layer, int eid) {
    if (!strcmp(kind, "layers")) ehit_mark(m, layer, eid);
    for (int i = 0; i < cache->n; i++)
        if (cache->slot[i].eid == eid) {
            cache->slot[i].used = ++m->clock;
            m->hits++;
            return &cache->slot[i];
        }
    m->miss++;
    Slot *victim;
    if (cache->n < cache->cap) {
        victim = &cache->slot[cache->n++];
    } else {
        int oldest = 0;
        for (int i = 1; i < cache->n; i++)
            if (cache->slot[i].used < cache->slot[oldest].used) oldest = i;
        victim = &cache->slot[oldest];
    }
    expert_read(m, kind, layer, eid, victim);
    victim->used = ++m->clock;
    return victim;
}

static Slot *expert_slot(Model *m, int layer, int eid) {
    return expert_slot_at(m, &m->cache[layer], "layers", layer, eid);
}

/* ------------------------------------------------------------- loading ----- */

typedef struct {
    WF norm1, norm2, qkv_b, o_b;
    WB qkv_w, o_w, w1, w2;
} VisionBlock;

typedef struct Vision {
    int active;
    WB proj_w; WF proj_b;
    VisionBlock *block;
    WF norm;
    WB align_w1, align_w2;
    WF align_b1, align_b2;
    WF image_start, image_end, image_newline;
} Vision;

static void vision_load(Model *m, Vision *v);
static void vision_forward(Model *m, Vision *v, const float *patches, int n_h, int n_w, float *out);
static int vision_tokens(const Cfg *c, int n_h, int n_w);
static void spec_load(Model *m, int ecap);

/* One layer's expert cache. The DSpark stages route over their own, smaller expert set
 * but the experts themselves are the same shape, so the slots are too. */
static void cache_init(Model *m, LCache *cache, int ecap) {
    Cfg *c = &m->c;
    int64_t packed_gate = (int64_t)c->moe_inter * c->dim / 2;
    int64_t packed_down = (int64_t)c->dim * c->moe_inter / 2;
    int64_t scales_gate = (int64_t)c->moe_inter * (c->dim / 32);
    int64_t scales_down = (int64_t)c->dim * (c->moe_inter / 32);
    cache->cap = ecap; cache->n = 0;
    cache->slot = xmalloc((size_t)ecap * sizeof(Slot), "expert slots");
    for (int k = 0; k < ecap; k++) {
        Slot *s = &cache->slot[k];
        s->eid = -1; s->used = 0;
        s->w1 = xmalloc((size_t)packed_gate, "expert w1");
        s->w3 = xmalloc((size_t)packed_gate, "expert w3");
        s->w2 = xmalloc((size_t)packed_down, "expert w2");
        s->s1 = xmalloc((size_t)scales_gate, "expert w1 scale");
        s->s3 = xmalloc((size_t)scales_gate, "expert w3 scale");
        s->s2 = xmalloc((size_t)scales_down, "expert w2 scale");
    }
}

static void model_load(Model *m, const char *snap, int ecap, int engram_cache_rows) {
    Cfg *c = &m->c;
    cfg_load(c, snap);
    st_init(&m->S, snap);
    engram_load_sidecar(&m->engram, snap);

    int dim = c->dim, hd = c->head_dim, nh = c->n_heads, hc = c->hc_mult;
    m->L = xmalloc((size_t)c->n_layers * sizeof(Layer), "layers");
    memset(m->L, 0, (size_t)c->n_layers * sizeof(Layer));
    wb_load(&m->S, &m->embed, "embed.weight", c->vocab, dim);
    wb_load(&m->S, &m->head, "head.weight", c->vocab, dim);
    wf_load(&m->S, &m->norm, "norm.weight", dim);

    char name[256];
    #define NAME(fmt, ...) (snprintf(name, sizeof(name), fmt, __VA_ARGS__), name)
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        l->engram_index = -1;
        w8_load(&m->S, &l->wq_a, NAME("layers.%d.attn.wq_a.weight", i), c->q_lora, dim);
        w8_load(&m->S, &l->wq_b, NAME("layers.%d.attn.wq_b.weight", i), nh * hd, c->q_lora);
        w8_load(&m->S, &l->wkv,  NAME("layers.%d.attn.wkv.weight", i), hd, dim);
        w8_load(&m->S, &l->wo_a, NAME("layers.%d.attn.wo_a.weight", i),
                c->o_groups * c->o_lora, nh * hd / c->o_groups);
        w8_load(&m->S, &l->wo_b, NAME("layers.%d.attn.wo_b.weight", i), dim, c->o_groups * c->o_lora);
        wf_load(&m->S, &l->q_norm,   NAME("layers.%d.attn.q_norm.weight", i), c->q_lora);
        wf_load(&m->S, &l->kv_norm,  NAME("layers.%d.attn.kv_norm.weight", i), hd);
        wf_load(&m->S, &l->attn_sink,NAME("layers.%d.attn.attn_sink", i), nh);
        wf_load(&m->S, &l->attn_norm,NAME("layers.%d.attn_norm.weight", i), dim);
        wf_load(&m->S, &l->ffn_norm, NAME("layers.%d.ffn_norm.weight", i), dim);
        int mix = (2 + hc) * hc;
        wf_load(&m->S, &l->hc_attn_fn,   NAME("layers.%d.hc_attn_fn", i), (int64_t)mix * hc * dim);
        wf_load(&m->S, &l->hc_ffn_fn,    NAME("layers.%d.hc_ffn_fn", i), (int64_t)mix * hc * dim);
        wf_load(&m->S, &l->hc_attn_base, NAME("layers.%d.hc_attn_base", i), mix);
        wf_load(&m->S, &l->hc_ffn_base,  NAME("layers.%d.hc_ffn_base", i), mix);
        wf_load(&m->S, &l->hc_attn_scale,NAME("layers.%d.hc_attn_scale", i), 3);
        wf_load(&m->S, &l->hc_ffn_scale, NAME("layers.%d.hc_ffn_scale", i), 3);
        wb_load(&m->S, &l->gate_w, NAME("layers.%d.ffn.gate.weight", i), c->n_routed, dim);
        wf_load(&m->S, &l->gate_bias, NAME("layers.%d.ffn.gate.bias", i), c->n_routed);
        w8_load(&m->S, &l->sh_w1, NAME("layers.%d.ffn.shared_experts.w1.weight", i), c->moe_inter, dim);
        w8_load(&m->S, &l->sh_w3, NAME("layers.%d.ffn.shared_experts.w3.weight", i), c->moe_inter, dim);
        w8_load(&m->S, &l->sh_w2, NAME("layers.%d.ffn.shared_experts.w2.weight", i), dim, c->moe_inter);

        l->window = xmalloc((size_t)c->window * hd * sizeof(float), "window ring");
        memset(l->window, 0, (size_t)c->window * hd * sizeof(float));
        l->window_pos = xmalloc((size_t)c->window * sizeof(int), "window positions");
        for (int k = 0; k < c->window; k++) l->window_pos[k] = -1;
        int rows = c->n_mtp > 0 ? c->spec_block + 1 : 1;
        l->ring_save = xmalloc((size_t)rows * hd * sizeof(float), "displaced ring keys");
        l->ring_save_pos = xmalloc((size_t)rows * sizeof(int), "displaced ring positions");
        int ratio = c->compress_ratio[i];
        if (c->kv_source[i]) {
            wb_load(&m->S, &l->comp_wkv, NAME("layers.%d.attn.compressor.wkv.weight", i), hd, dim);
            wf_load(&m->S, &l->comp_norm, NAME("layers.%d.attn.compressor.norm.weight", i), hd);
            if (ratio > 1) {
                wb_load(&m->S, &l->comp_wgate, NAME("layers.%d.attn.compressor.wgate.weight", i), hd, dim);
                l->cstate_kv = xmalloc((size_t)ratio * hd * sizeof(float), "compressor group");
                l->cstate_score = xmalloc((size_t)ratio * hd * sizeof(float), "compressor scores");
                l->cstate_save_kv = xmalloc((size_t)rows * hd * sizeof(float), "displaced group");
                l->cstate_save_score = xmalloc((size_t)rows * hd * sizeof(float), "displaced scores");
                l->cstate_save_slot = xmalloc((size_t)rows * sizeof(int), "displaced group slots");
                memset(l->cstate_kv, 0, (size_t)ratio * hd * sizeof(float));
                for (int k = 0; k < ratio * hd; k++) l->cstate_score[k] = -INFINITY;
            }
            int slots = c->max_positions / (ratio > 0 ? ratio : 1);
            l->ckv = xmalloc((size_t)slots * hd * sizeof(float), "compressed kv");
            memset(l->ckv, 0, (size_t)slots * hd * sizeof(float));
            l->ikey = xmalloc((size_t)slots * c->index_head_dim * sizeof(float), "index keys");
            memset(l->ikey, 0, (size_t)slots * c->index_head_dim * sizeof(float));
            wb_load(&m->S, &l->idx_wk, NAME("layers.%d.attn.indexer.wk.weight", i),
                    c->index_head_dim, hd);
            wf_load(&m->S, &l->idx_knorm, NAME("layers.%d.attn.indexer.k_norm.weight", i),
                    c->index_head_dim);
        }
        if (c->index_source[i]) {
            w8_load(&m->S, &l->idx_wq_b, NAME("layers.%d.attn.indexer.wq_b.weight", i),
                    c->index_n_heads * c->index_head_dim, c->q_lora);
            wb_load(&m->S, &l->idx_wproj, NAME("layers.%d.attn.indexer.weights_proj.weight", i),
                    c->index_n_heads, dim);
        }
    }
    if (m->engram.active) {
        Engram *e = &m->engram;
        for (int t = 0; t < e->n_layers; t++) {
            int layer = e->layer_of[t];
            if (layer < 0 || layer >= c->n_layers) {
                fprintf(stderr, "[engram] table %d names layer %d\n", t, layer); exit(1); }
            m->L[layer].engram_index = t;
            engram_table_open(&e->table[t], &m->S, layer, e->head_dim, engram_cache_rows);
            w8_load(&m->S, &m->L[layer].eng_wkv, NAME("layers.%d.engram.wkv.weight", layer),
                    dim * (hc + 1), e->cols * e->head_dim);
            wf_load(&m->S, &m->L[layer].eng_q, NAME("layers.%d.engram.q_weight", layer),
                    (int64_t)hc * dim);
            wf_load(&m->S, &m->L[layer].eng_k, NAME("layers.%d.engram.k_weight", layer),
                    (int64_t)hc * dim);
        }
    }
    #undef NAME

    /* expert cache: `ecap` slots per layer, each holding one expert's three matrices */
    m->ecap = ecap;
    m->cache = xmalloc((size_t)c->n_layers * sizeof(LCache), "expert caches");
    for (int i = 0; i < c->n_layers; i++) cache_init(m, &m->cache[i], ecap);
    spec_load(m, ecap);
    if (c->vision_layers > 0) {
        m->vision = xmalloc(sizeof(Vision), "vision tower");
        memset(m->vision, 0, sizeof(Vision));
        vision_load(m, m->vision);
        if (!m->vision->active) { free(m->vision); m->vision = NULL; }
    }
    m->rope_window = rope_table(c->rope_dim, c->max_positions, 0, c->rope_theta,
                                c->rope_factor, c->beta_fast, c->beta_slow);
    m->rope_compress = rope_table(c->rope_dim, c->max_positions, c->original_seq_len,
                                  c->compress_rope_theta, c->rope_factor,
                                  c->beta_fast, c->beta_slow);
    m->pos = 0;
}

static const float *rope_for(const Model *m, int layer) {
    return m->c.compress_ratio[layer] ? m->rope_compress : m->rope_window;
}

/* ----------------------------------------------------------- attention ----- */

/* model.py get_window_topk_idxs, one batch: which ring slots query `t` may read.
 * Prefill indexes the chunk itself (slot i is position i); decode indexes the ring.
 *
 * On decode the readable set is stated in POSITIONS, not in slot numbers: a slot is
 * readable when the position it holds is at or before this query's own and still
 * inside the window. For one token a step that is the same thing as the vendor's
 * `i > start_pos` test, since slot i then holds position i until the ring wraps. For a
 * speculative step it is not: the ring also holds the drafts that were rejected, whose
 * positions are ahead of every query that follows, and the position map is what keeps
 * them out. The traversal stays oldest-slot-first, per query, because the attention
 * kernel sums in list order and a different order is a different rounding. */
static int window_slot_ok(const Layer *l, int slot, int pos, int win) {
    int held = l->window_pos[slot];
    return held >= 0 && held <= pos && held > pos - win;
}

static void window_idxs(const Cfg *c, const Layer *l, int n, int start_pos, int t,
                        int *out, int *count) {
    int win = c->window;
    if (start_pos == 0) {
        int width = n < win ? n : win;
        int first = t - win + 1; if (first < 0) first = 0;
        for (int i = 0; i < width; i++) {
            int idx = first + i;
            out[i] = idx > t ? -1 : idx;
        }
        *count = width;
    } else {
        int pos = start_pos + t;
        int oldest = pos % win + 1;
        int at = 0;
        for (int i = oldest; i < win; i++) out[at++] = window_slot_ok(l, i, pos, win) ? i : -1;
        for (int i = 0; i < oldest; i++) out[at++] = window_slot_ok(l, i, pos, win) ? i : -1;
        *count = win;
    }
}

/* kernel.py sparse_attn for one query: `idx` lists the rows of `kv` this query reads,
 * -1 meaning nothing. The sink joins the denominator only, so a query with no readable
 * row comes out zero instead of NaN. */
static void sparse_attend(float *out, const float *q, const float *kv, const int *idx,
                          int count, int nh, int hd, float sink_h, float scale) {
    float best = -1e30f;
    float *score = xmalloc((size_t)count * sizeof(float), "attention scores");
    for (int j = 0; j < count; j++) {
        if (idx[j] < 0) { score[j] = -INFINITY; continue; }
        const float *k = kv + (size_t)idx[j] * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; i++) dot += q[i] * k[i];
        score[j] = dot * scale;
        if (score[j] > best) best = score[j];
    }
    float denom = expf(sink_h - best);
    for (int i = 0; i < hd; i++) out[i] = 0.0f;
    for (int j = 0; j < count; j++) {
        if (!(score[j] > -INFINITY)) continue;
        float weight = expf(score[j] - best);
        denom += weight;
        const float *k = kv + (size_t)idx[j] * hd;
        for (int i = 0; i < hd; i++) out[i] += weight * k[i];
    }
    for (int i = 0; i < hd; i++) out[i] /= denom;
    free(score);
    (void)nh;
}

/* model.py Compressor.forward. Writes the latents this chunk completes into `latent`
 * and returns how many there are (pre-RoPE, as the indexer needs them unrotated). */
static int compressor_run(Model *m, int layer, const float *x, int n, int start_pos,
                          float *latent, int *rows) {
    Cfg *c = &m->c;
    Layer *l = &m->L[layer];
    int hd = c->head_dim, ratio = c->compress_ratio[layer];
    if (ratio == 1) {
        for (int t = 0; t < n; t++) {
            float kv[512];
            if (hd > (int)(sizeof(kv) / sizeof(kv[0]))) {
                fprintf(stderr, "[compressor] head_dim %d exceeds the scratch\n", hd); exit(1); }
            mvb(kv, &l->comp_wkv, x + (size_t)t * c->dim);
            rms_into(latent + (size_t)t * hd, kv, l->comp_norm.w, hd, c->norm_eps);
            rows[t] = start_pos + t;
        }
        return n;
    }
    float *kv = xmalloc((size_t)n * hd * sizeof(float), "compressor kv");
    float *score = xmalloc((size_t)n * hd * sizeof(float), "compressor scores");
    for (int t = 0; t < n; t++) {
        mvb(kv + (size_t)t * hd, &l->comp_wkv, x + (size_t)t * c->dim);
        mvb(score + (size_t)t * hd, &l->comp_wgate, x + (size_t)t * c->dim);
    }
    int produced = 0;
    if (start_pos == 0) {
        int remainder = n % ratio, cutoff = n - remainder;
        for (int g = 0; g * ratio < cutoff; g++) {
            for (int i = 0; i < hd; i++) {
                float best = -INFINITY;
                for (int k = 0; k < ratio; k++) {
                    float value = score[(size_t)(g * ratio + k) * hd + i];
                    if (value > best) best = value;
                }
                float total = 0.0f, mixed = 0.0f;
                for (int k = 0; k < ratio; k++) {
                    float weight = expf(score[(size_t)(g * ratio + k) * hd + i] - best);
                    total += weight;
                    mixed += weight * kv[(size_t)(g * ratio + k) * hd + i];
                }
                latent[(size_t)g * hd + i] = mixed / total;
            }
            rows[produced++] = g;
        }
        for (int k = 0; k < remainder; k++) {
            memcpy(l->cstate_kv + (size_t)k * hd, kv + (size_t)(cutoff + k) * hd, (size_t)hd * sizeof(float));
            memcpy(l->cstate_score + (size_t)k * hd, score + (size_t)(cutoff + k) * hd, (size_t)hd * sizeof(float));
        }
    } else {
        /* One position at a time, which is what a decode step is -- and a speculative
         * step is several of them in a row. The group state is indexed by position
         * modulo the ratio, so a draft that is rejected leaves its slot to be rewritten
         * by the token that really lands there, before the group can complete. */
        for (int t = 0; t < n; t++) {
            int pos = start_pos + t;
            int slot = pos % ratio;
            if (n > 1) {
                memcpy(l->cstate_save_kv + (size_t)t * hd, l->cstate_kv + (size_t)slot * hd,
                       (size_t)hd * sizeof(float));
                memcpy(l->cstate_save_score + (size_t)t * hd, l->cstate_score + (size_t)slot * hd,
                       (size_t)hd * sizeof(float));
                l->cstate_save_slot[t] = slot;
            }
            memcpy(l->cstate_kv + (size_t)slot * hd, kv + (size_t)t * hd, (size_t)hd * sizeof(float));
            memcpy(l->cstate_score + (size_t)slot * hd, score + (size_t)t * hd, (size_t)hd * sizeof(float));
            if ((pos + 1) % ratio) continue;
            float *out = latent + (size_t)produced * hd;
            for (int i = 0; i < hd; i++) {
                float best = -INFINITY;
                for (int k = 0; k < ratio; k++) {
                    float value = l->cstate_score[(size_t)k * hd + i];
                    if (value > best) best = value;
                }
                float total = 0.0f, mixed = 0.0f;
                for (int k = 0; k < ratio; k++) {
                    float weight = expf(l->cstate_score[(size_t)k * hd + i] - best);
                    total += weight;
                    mixed += weight * l->cstate_kv[(size_t)k * hd + i];
                }
                out[i] = mixed / total;
            }
            rows[produced++] = pos / ratio;
        }
    }
    if (produced) {
        float *pooled = xmalloc((size_t)produced * hd * sizeof(float), "pooled latents");
        memcpy(pooled, latent, (size_t)produced * hd * sizeof(float));
        for (int g = 0; g < produced; g++)
            rms_into(latent + (size_t)g * hd, pooled + (size_t)g * hd, l->comp_norm.w, hd, c->norm_eps);
        free(pooled);
    }
    free(kv); free(score);
    return produced;
}

/* model.py select_candidate_blocks, one query row. `keep` is `width` bytes. */
static void candidate_blocks(const Cfg *c, const float *score, int width, int lens, uint8_t *keep) {
    int block = c->candidate_block_size;
    int blocks = (width + block - 1) / block;
    float *best = xmalloc((size_t)blocks * sizeof(float), "block scores");
    for (int b = 0; b < blocks; b++) {
        float top = -INFINITY;
        for (int i = b * block; i < (b + 1) * block && i < width; i++)
            if (score[i] > top) top = score[i];
        best[b] = top;
    }
    int last = (lens - 1) / block;
    if (last >= 0 && last < blocks) best[last] = INFINITY;   /* the newest block is pinned in */
    int wanted = c->candidate_topk_blocks < blocks ? c->candidate_topk_blocks : blocks;
    memset(keep, 0, (size_t)width);
    for (int pick = 0; pick < wanted; pick++) {
        int chosen = -1;
        for (int b = 0; b < blocks; b++)
            if (best[b] > -INFINITY && (chosen < 0 || best[b] > best[chosen])) chosen = b;
        if (chosen < 0) break;
        for (int i = chosen * block; i < (chosen + 1) * block && i < width; i++) keep[i] = 1;
        best[chosen] = -INFINITY;
    }
    free(best);
}

/* Does this layer complete a compression group at `pos`, and so republish its index
 * keys? model.py: `if self.owns_k and latent is not None`. */
static int layer_publishes(const Cfg *c, int layer, int pos) {
    int ratio = c->compress_ratio[layer];
    if (!c->kv_source[layer] || ratio <= 0) return 0;
    return ratio == 1 || (pos + 1) % ratio == 0;
}

/* Which layer's index keys row `t` reads, or -1 for "whatever the step began with".
 * Sequential order is (position, layer): everything at an earlier position happened,
 * whatever its layer, and at this position only the layers up to this one -- including
 * this one, which publishes before its own indexer runs. */
static int published_owner(const Model *m, int layer, int start_pos, int t) {
    if (m->pub_rows == 0) return -1;              /* one position: the step's own slot */
    for (int L = layer; L >= 0; L--)
        if (layer_publishes(&m->c, L, start_pos + t)) return L;
    for (int j = t - 1; j >= 0; j--)
        if (m->pub_layer[j] >= 0) return m->pub_layer[j];
    return -1;
}

/* model.py Indexer.forward: score every compressed position this query can reach and
 * keep the best `index_topk`, in position order, shifted by `offset`. */
static void indexer_run(Model *m, int layer, const float *x, const float *qr, int n,
                        int start_pos, int offset, int compress_len,
                        int *out, int stride, int topk) {
    Cfg *c = &m->c;
    Layer *l = &m->L[layer];
    int owner = c->index_owner[layer];
    int nh = c->index_n_heads, ihd = c->index_head_dim, ratio = c->compress_ratio[layer];
    int tidy = getenv("V41_INDEX_OWNER") != NULL;
    const float *rope = rope_for(m, layer);
    float *q = xmalloc((size_t)nh * ihd * sizeof(float), "indexer queries");
    float *weights = xmalloc((size_t)nh * sizeof(float), "indexer weights");
    float *score = xmalloc((size_t)compress_len * sizeof(float), "indexer scores");
    float scale = (1.0f / sqrtf((float)ihd)) * (1.0f / sqrtf((float)nh));

    for (int t = 0; t < n; t++) {
        const float *ikey = m->L[owner].ikey;
        if (!tidy) {
            int published = published_owner(m, layer, start_pos, t);
            if (published >= 0) ikey = m->L[published].ikey;
            else if (m->pub_rows > 0) { if (m->pub_before) ikey = m->pub_before; }
            else if (m->published_index_k) ikey = m->published_index_k;
        }
        mv8(q, &l->idx_wq_b, qr + (size_t)t * c->q_lora);
        for (int h = 0; h < nh; h++)
            rope_apply(q + (size_t)h * ihd + ihd - c->rope_dim, rope, start_pos + t, c->rope_dim, 0);
        mvb(weights, &l->idx_wproj, x + (size_t)t * c->dim);
        for (int h = 0; h < nh; h++) weights[h] *= scale;
        /* how many compressed positions this query can reach: the groups closed at or
         * before its own position, which is one expression for prefill and decode
         * alike -- and, for a speculative step, per row rather than per step */
        int lens = (start_pos + t + 1) / ratio;
        for (int j = 0; j < compress_len; j++) {
            if (j >= lens) { score[j] = -INFINITY; continue; }
            float total = 0.0f;
            for (int h = 0; h < nh; h++) {
                const float *qh = q + (size_t)h * ihd;
                const float *k = ikey + (size_t)j * ihd;
                float dot = 0.0f;
                for (int i = 0; i < ihd; i++) dot += qh[i] * k[i];
                if (dot > 0.0f) total += dot * weights[h];    /* relu, then the head weight */
            }
            score[j] = total;
        }
        if (layer == c->candidate_source) {
            if (m->candidate_width != compress_len || m->candidate_rows != n) {
                free(m->candidates);
                m->candidates = xmalloc((size_t)n * compress_len, "candidate mask");
                m->candidate_width = compress_len;
                m->candidate_rows = n;
            }
            candidate_blocks(c, score, compress_len, lens, m->candidates + (size_t)t * compress_len);
        } else if (c->candidate_source >= 0 && c->candidate_source < layer && m->candidates) {
            const uint8_t *keep = m->candidates + (size_t)t * m->candidate_width;
            for (int j = 0; j < compress_len && j < m->candidate_width; j++)
                if (!keep[j]) score[j] = -INFINITY;
        }
        /* top-k, then back into position order: the attention kernel treats every slot
         * independently, but keeping them sorted makes the concatenated index list
         * readable and matches the reference's `.sort()`. */
        int *row = out + (size_t)t * stride;   /* the compressed half of this query's list */
        for (int k = 0; k < topk; k++) row[k] = -1;
        int taken = 0;
        for (int k = 0; k < topk; k++) {
            int best = -1;
            for (int j = 0; j < compress_len; j++) {
                if (score[j] == -INFINITY) continue;
                int already = 0;
                for (int u = 0; u < taken; u++) if (row[u] == j) { already = 1; break; }
                if (already) continue;
                if (best < 0 || score[j] > score[best]) best = j;
            }
            if (best < 0) break;
            row[taken++] = best;
        }
        for (int a = 0; a < taken; a++)          /* insertion sort: taken <= index_topk */
            for (int b = a + 1; b < taken; b++)
                if (row[b] < row[a]) { int swap = row[a]; row[a] = row[b]; row[b] = swap; }
        for (int k = 0; k < topk; k++)
            row[k] = (k < taken && row[k] < lens) ? row[k] + offset : -1;
    }
    free(q); free(weights); free(score);
}

/* wo_a is block diagonal over o_groups: each group projects only its own heads, and
 * only then does wo_b mix the groups. Shared with the DSpark stages, whose attention
 * differs in what it reads but not in how it comes out. */
static void attn_project_out(Model *m, Layer *l, const float *heads, float *out) {
    Cfg *c = &m->c;
    int per_group = c->n_heads * c->head_dim / c->o_groups;
    float *grouped = xmalloc((size_t)c->o_groups * c->o_lora * sizeof(float), "grouped");
    for (int g = 0; g < c->o_groups; g++) {
        const uint8_t *rows = l->wo_a.q + (size_t)g * c->o_lora * per_group;
        const uint8_t *scales = l->wo_a.s;
        int tiles_i = (per_group + FP8_TILE - 1) / FP8_TILE;
        for (int r = 0; r < c->o_lora; r++) {
            int o = g * c->o_lora + r;
            const uint8_t *w = rows + (size_t)r * per_group;
            const uint8_t *scale_row = scales + (size_t)(o / FP8_TILE) * tiles_i;
            const float *xg = heads + (size_t)g * per_group;
            float sum = 0.0f;
            for (int base = 0; base < per_group; base += FP8_TILE) {
                int width = per_group - base < FP8_TILE ? per_group - base : FP8_TILE;
                float tile = ue8m0(scale_row[base / FP8_TILE]);
                float part = 0.0f;
                for (int i = 0; i < width; i++) part += e4m3_decode(w[base + i]) * xg[base + i];
                sum += part * tile;
            }
            grouped[o] = sum;
        }
    }
    mv8(out, &l->wo_b, grouped);
    free(grouped);
}

/* model.py Attention.forward. `x` is the normalized block input [n, dim]. */
static void attention_run(Model *m, int layer, const float *x, int n, int start_pos, float *out) {
    Cfg *c = &m->c;
    Layer *l = &m->L[layer];
    int dim = c->dim, hd = c->head_dim, nh = c->n_heads, rd = c->rope_dim;
    int ratio = c->compress_ratio[layer], owner = c->index_owner[layer];
    const float *rope = rope_for(m, layer);
    double started = now_s();

    float *qr = xmalloc((size_t)n * c->q_lora * sizeof(float), "q latent");
    float *q = xmalloc((size_t)n * nh * hd * sizeof(float), "queries");
    float *kv = xmalloc((size_t)n * hd * sizeof(float), "kv");
    float *scratch = xmalloc((size_t)(nh * hd > dim ? nh * hd : dim) * sizeof(float), "scratch");

    for (int t = 0; t < n; t++) {
        mv8(scratch, &l->wq_a, x + (size_t)t * dim);
        rms_into(qr + (size_t)t * c->q_lora, scratch, l->q_norm.w, c->q_lora, c->norm_eps);
        mv8(q + (size_t)t * nh * hd, &l->wq_b, qr + (size_t)t * c->q_lora);
        for (int h = 0; h < nh; h++)
            rope_apply(q + (size_t)t * nh * hd + (size_t)h * hd + hd - rd, rope, start_pos + t, rd, 0);
        mv8(scratch, &l->wkv, x + (size_t)t * dim);
        rms_into(kv + (size_t)t * hd, scratch, l->kv_norm.w, hd, c->norm_eps);
        rope_apply(kv + (size_t)t * hd + hd - rd, rope, start_pos + t, rd, 0);
    }

    /* the window half: prefill reads the chunk, decode reads the ring */
    const float *window_kv;
    int window_rows;
    if (start_pos == 0) {
        int win = c->window;
        if (n <= win) {
            memcpy(l->window, kv, (size_t)n * hd * sizeof(float));
        } else {
            int cut = n % win;
            const float *tail = kv + (size_t)(n - win) * hd;
            memcpy(l->window + (size_t)cut * hd, tail, (size_t)(win - cut) * hd * sizeof(float));
            memcpy(l->window, tail + (size_t)(win - cut) * hd, (size_t)cut * hd * sizeof(float));
        }
        /* both branches above leave position p in slot p % win */
        for (int p = n > win ? n - win : 0; p < n; p++) l->window_pos[p % win] = p;
        window_kv = kv; window_rows = n;
    } else {
        /* The ring is written one position at a time, down in the attention loop, and
         * not here: a step that carries several positions would otherwise overwrite the
         * oldest slot before the first row has read it. The window a query sees has to
         * be the window it would have seen alone, which means writing its key, reading,
         * and only then writing the next one. */
        window_kv = l->window; window_rows = c->window;
    }

    int window_width = start_pos == 0 ? (n < c->window ? n : c->window) : c->window;
    int compress_len = ratio ? (start_pos + n) / ratio : 0;
    int index_topk = 0;
    if (ratio) {
        index_topk = c->index_topk < compress_len ? c->index_topk : compress_len;
        if (index_topk < 0) index_topk = 0;
    }
    int total_idx = window_width + index_topk;
    int *idx = xmalloc((size_t)n * total_idx * sizeof(int), "attention indices");
    for (int t = 0; t < n; t++) {
        int count = 0;
        window_idxs(c, l, n, start_pos, t, idx + (size_t)t * total_idx, &count);
        for (int k = count; k < window_width; k++) idx[(size_t)t * total_idx + k] = -1;
    }

    float *kv_all = NULL;
    if (ratio) {
        /* the compressor runs before the indexer: the indexer scores the latent before
         * RoPE, and the cache write below rotates it in place */
        float *latent = NULL;
        int *latent_row = NULL;
        int produced = 0;
        if (c->kv_source[layer]) {
            int room = n / (ratio ? ratio : 1) + 1;
            latent = xmalloc((size_t)room * hd * sizeof(float), "latents");
            latent_row = xmalloc((size_t)room * sizeof(int), "latent rows");
            produced = compressor_run(m, layer, x, n, start_pos, latent, latent_row);
            if (produced) {
                /* index keys first, from the un-rotated latent. A latent rotates at the
                 * FIRST position of the group it pools, which is its row times the
                 * ratio -- the same value the vendor spells two different ways on the
                 * prefill and decode paths. */
                for (int g = 0; g < produced; g++) {
                    float key[512];
                    if (c->index_head_dim > (int)(sizeof(key) / sizeof(key[0]))) {
                        fprintf(stderr, "[indexer] index_head_dim too large\n"); exit(1); }
                    mvb(key, &l->idx_wk, latent + (size_t)g * hd);
                    float *dest = l->ikey + (size_t)latent_row[g] * c->index_head_dim;
                    rms_into(dest, key, l->idx_knorm.w, c->index_head_dim, c->norm_eps);
                    rope_apply(dest + c->index_head_dim - rd, rope, latent_row[g] * ratio, rd, 0);
                }
                /* publish, as model.py does, only when a group completed */
                m->published_index_k = l->ikey;
                m->published_index_layer = layer;
            }
        }
        if (c->index_source[layer]) {
            if (compress_len > 0 && index_topk > 0)
                /* The offset is how many ROWS the window half occupies in kv_all, not
                 * how many index columns it takes: during prefill the window half is
                 * the whole chunk (n rows) while each query lists at most `window`
                 * columns of it. model.py passes kv.size(1) here for the same reason. */
                indexer_run(m, layer, x, qr, n, start_pos, window_rows, compress_len,
                            idx + window_width, total_idx, index_topk);
            else
                for (int t = 0; t < n; t++)
                    for (int k = window_width; k < total_idx; k++) idx[(size_t)t * total_idx + k] = -1;
            /* publish for the layers between here and the next index source */
            if (m->shared_topk_rows != n || m->shared_topk_width != index_topk) {
                free(m->shared_topk);
                m->shared_topk = xmalloc((size_t)n * (index_topk > 0 ? index_topk : 1) * sizeof(int),
                                         "published index list");
                m->shared_topk_rows = n; m->shared_topk_width = index_topk;
            }
            for (int t = 0; t < n; t++)
                for (int k = 0; k < index_topk; k++)
                    m->shared_topk[(size_t)t * index_topk + k] = idx[(size_t)t * total_idx + window_width + k];
        } else {
            /* Not an index source: reuse what the last one published, exactly as
             * model.py does through shared_attn.topk_idxs. Recomputing here would be
             * wrong as well as wasteful -- these layers have no indexer weights. */
            for (int t = 0; t < n; t++)
                for (int k = 0; k < index_topk; k++) {
                    int value = -1;
                    if (m->shared_topk && t < m->shared_topk_rows && k < m->shared_topk_width)
                        value = m->shared_topk[(size_t)t * m->shared_topk_width + k];
                    idx[(size_t)t * total_idx + window_width + k] = value;
                }
        }
        if (produced) {
            for (int g = 0; g < produced; g++) {
                float *dest = l->ckv + (size_t)latent_row[g] * hd;
                memcpy(dest, latent + (size_t)g * hd, (size_t)hd * sizeof(float));
                rope_apply(dest + hd - rd, rope, latent_row[g] * ratio, rd, 0);
            }
        }
        free(latent); free(latent_row);
        kv_all = xmalloc((size_t)(window_rows + compress_len) * hd * sizeof(float), "kv all");
        memcpy(kv_all, window_kv, (size_t)window_rows * hd * sizeof(float));
        memcpy(kv_all + (size_t)window_rows * hd, m->L[owner].ckv,
               (size_t)compress_len * hd * sizeof(float));
    }

    const float *kv_read = ratio ? kv_all : window_kv;
    float *heads = xmalloc((size_t)nh * hd * sizeof(float), "attention output");
    float attn_scale = 1.0f / sqrtf((float)hd);
    for (int t = 0; t < n; t++) {
        if (start_pos > 0) {
            int pos = start_pos + t, slot = pos % c->window;
            if (n > 1) {
                memcpy(l->ring_save + (size_t)t * hd, l->window + (size_t)slot * hd,
                       (size_t)hd * sizeof(float));
                l->ring_save_pos[t] = l->window_pos[slot];
            }
            memcpy(l->window + (size_t)slot * hd, kv + (size_t)t * hd, (size_t)hd * sizeof(float));
            l->window_pos[slot] = pos;
            /* kv_all holds a copy of the ring, so it takes the same write */
            if (kv_all) memcpy(kv_all + (size_t)slot * hd, kv + (size_t)t * hd,
                               (size_t)hd * sizeof(float));
            int count = 0;
            window_idxs(c, l, n, start_pos, t, idx + (size_t)t * total_idx, &count);
            for (int k = count; k < window_width; k++) idx[(size_t)t * total_idx + k] = -1;
        }
        const int *row = idx + (size_t)t * total_idx;
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < nh; h++)
            sparse_attend(heads + (size_t)h * hd, q + (size_t)t * nh * hd + (size_t)h * hd,
                          kv_read, row, total_idx, nh, hd, l->attn_sink.w[h], attn_scale);
        for (int h = 0; h < nh; h++)
            rope_apply(heads + (size_t)h * hd + hd - rd, rope, start_pos + t, rd, 1);
        attn_project_out(m, l, heads, out + (size_t)t * dim);
    }
    m->t_attn += now_s() - started;
    free(heads); free(kv_all); free(idx); free(scratch); free(kv); free(q); free(qr);
}

/* ------------------------------------------------------------------ MoE ---- */

/* One expert's SwiGLU. The clamps are the training kernel's, and they matter: they
 * are what keeps fp4 activations in range. */
static void expert_ffn(Model *m, const uint8_t *w1, const uint8_t *s1,
                       const uint8_t *w3, const uint8_t *s3,
                       const uint8_t *w2, const uint8_t *s2,
                       const float *x, float *out, float weight) {
    Cfg *c = &m->c;
    int inter = c->moe_inter, dim = c->dim;
    float *gate = xmalloc((size_t)inter * sizeof(float), "expert gate");
    float *up = xmalloc((size_t)inter * sizeof(float), "expert up");
    float *mid = xmalloc((size_t)inter * sizeof(float), "expert mid");
    double started = now_s();
    matmul_mxfp4(gate, x, w1, s1, 1, dim, inter);
    matmul_mxfp4(up,   x, w3, s3, 1, dim, inter);
    for (int i = 0; i < inter; i++) {
        float g = gate[i], u = up[i];
        if (c->swiglu_limit > 0.0f) {
            if (u >  c->swiglu_limit) u =  c->swiglu_limit;
            if (u < -c->swiglu_limit) u = -c->swiglu_limit;
            if (g >  c->swiglu_limit) g =  c->swiglu_limit;
        }
        mid[i] = (g / (1.0f + expf(-g))) * u;
    }
    float *down = xmalloc((size_t)dim * sizeof(float), "expert down");
    matmul_mxfp4(down, mid, w2, s2, 1, inter, dim);
    for (int i = 0; i < dim; i++) out[i] += weight * down[i];
    m->t_expert += now_s() - started;
    free(down); free(mid); free(up); free(gate);
}

/* The shared expert every token pays for; fp8, so it stays resident and never streams. */
static void shared_ffn(Model *m, Layer *l, const float *x, float *out) {
    Cfg *c = &m->c;
    int inter = c->moe_inter, dim = c->dim;
    float *gate = xmalloc((size_t)inter * sizeof(float), "shared gate");
    float *up = xmalloc((size_t)inter * sizeof(float), "shared up");
    mv8(gate, &l->sh_w1, x);
    mv8(up,   &l->sh_w3, x);
    for (int i = 0; i < inter; i++) {
        float g = gate[i], u = up[i];
        if (c->swiglu_limit > 0.0f) {
            if (u >  c->swiglu_limit) u =  c->swiglu_limit;
            if (u < -c->swiglu_limit) u = -c->swiglu_limit;
            if (g >  c->swiglu_limit) g =  c->swiglu_limit;
        }
        gate[i] = (g / (1.0f + expf(-g))) * u;
    }
    float *down = xmalloc((size_t)dim * sizeof(float), "shared down");
    mv8(down, &l->sh_w2, gate);
    for (int i = 0; i < dim; i++) out[i] += down[i];
    free(down); free(up); free(gate);
}

static void moe_run_at(Model *m, Layer *l, LCache *cache, const char *kind, int layer,
                       int E, int topk, const float *x, float *out);

/* model.py Gate + MoE. The bias steers the choice of experts and nothing else: the
 * weights come from the unbiased scores, which is the whole point of noaux_tc. */
static void moe_run(Model *m, int layer, const float *x, float *out) {
    moe_run_at(m, &m->L[layer], &m->cache[layer], "layers", layer,
               m->c.n_routed, m->c.n_activated, x, out);
}

static void moe_run_at(Model *m, Layer *l, LCache *cache, const char *kind, int layer,
                       int E, int topk, const float *x, float *out) {
    Cfg *c = &m->c;
    float *scores = xmalloc((size_t)E * sizeof(float), "gate scores");
    mvb(scores, &l->gate_w, x);
    for (int e = 0; e < E; e++) {
        float value = scores[e] / c->gate_temp;
        /* sqrtsoftplus: softplus then square root, in fp32 as the vendor does */
        float softplus = value > 20.0f ? value : logf(1.0f + expf(value));
        scores[e] = sqrtf(softplus);
    }
    int chosen[64];
    float weights[64];
    if (topk > (int)(sizeof(chosen) / sizeof(chosen[0]))) {
        fprintf(stderr, "[moe] n_activated %d exceeds the scratch\n", topk); exit(1); }
    for (int k = 0; k < topk; k++) {
        int best = -1;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int u = 0; u < k; u++) if (chosen[u] == e) { taken = 1; break; }
            if (taken) continue;
            if (best < 0 || scores[e] + l->gate_bias.w[e] > scores[best] + l->gate_bias.w[best]) best = e;
        }
        chosen[k] = best;
        weights[k] = scores[best];
    }
    if (c->norm_topk_prob && topk > 1) {
        float total = 0.0f;
        for (int k = 0; k < topk; k++) total += weights[k];
        total += 1e-20f;                       /* the vendor's constant, not norm_eps */
        for (int k = 0; k < topk; k++) weights[k] /= total;
    }
    for (int k = 0; k < topk; k++) weights[k] *= c->route_scale;
    memset(out, 0, (size_t)c->dim * sizeof(float));
    for (int k = 0; k < topk; k++) {
        Slot *s = expert_slot_at(m, cache, kind, layer, chosen[k]);
        expert_ffn(m, s->w1, s->s1, s->w3, s->s3, s->w2, s->s2, x, out, weights[k]);
    }
    shared_ffn(m, l, x, out);
    free(scores);
}

/* --------------------------------------------------------------- engram ---- */

/* model.py Engram.forward: look the n-grams up, project them into one key per residual
 * copy plus a shared value, and gate the value in by how well key and stream agree. */
static void engram_run(Model *m, int layer, float *h, int n, int start_pos) {
    Cfg *c = &m->c;
    Engram *e = &m->engram;
    Layer *l = &m->L[layer];
    int table = l->engram_index, hc = c->hc_mult, dim = c->dim;
    double started = now_s();
    float *rows = xmalloc((size_t)e->cols * e->head_dim * sizeof(float), "engram rows");
    float *kv = xmalloc((size_t)dim * (hc + 1) * sizeof(float), "engram kv");
    int64_t ids[V41_MAX_NGRAM * V41_MAX_EHEADS];
    for (int t = 0; t < n; t++) {
        engram_hash(e, table, start_pos + t, ids);
        for (int col = 0; col < e->cols; col++) {
            const float *row = engram_row(&e->table[table], ids[col], e->head_dim);
            memcpy(rows + (size_t)col * e->head_dim, row, (size_t)e->head_dim * sizeof(float));
        }
        mv8(kv, &l->eng_wkv, rows);
        const float *value = kv + (size_t)hc * dim;
        float *stream = h + (size_t)t * hc * dim;
        for (int copy = 0; copy < hc; copy++) {
            const float *key = kv + (size_t)copy * dim;
            const float *qw = l->eng_q.w + (size_t)copy * dim;
            const float *kw = l->eng_k.w + (size_t)copy * dim;
            float *stream_copy = stream + (size_t)copy * dim;
            double stream_square = 0.0, key_square = 0.0;
            double dot = 0.0;
            for (int i = 0; i < dim; i++) {
                stream_square += (double)stream_copy[i] * stream_copy[i];
                key_square += (double)key[i] * key[i];
                dot += (double)stream_copy[i] * qw[i] * kw[i] * key[i];
            }
            float rstd = (1.0f / sqrtf((float)(stream_square / dim) + c->norm_eps)) *
                         (1.0f / sqrtf((float)(key_square / dim) + c->norm_eps));
            float scaled = (float)dot * rstd / sqrtf((float)dim);
            /* signed square root before the sigmoid, matching the training kernel */
            float magnitude = fabsf(scaled);
            if (magnitude < 1e-6f) magnitude = 1e-6f;
            float signed_root = sqrtf(magnitude);
            if (scaled < 0.0f) signed_root = -signed_root;
            float gate = coli_hc_sigmoid(signed_root);
            for (int i = 0; i < dim; i++) stream_copy[i] += gate * value[i];
        }
    }
    trace("engram", layer, h + (size_t)(n - 1) * hc * dim, hc * dim);
    free(kv); free(rows);
    m->t_engram += now_s() - started;
}

/* --------------------------------------------------------------- vision ---- */

/* The ViT is small next to the language model (about 600 MB in the released
 * checkpoint, bf16) and every image needs all of it, so it stays resident: there is
 * nothing to stream here. What matters for correctness is that its two conventions
 * differ from the language model's and are easy to get silently wrong:
 *
 *   - RoPE is SPLIT-HALF (x[:d/2] against x[d/2:]), not the interleaved pairs the
 *     text side uses, and it is 2D: each patch carries its row and column angle.
 *   - the aligner's 3x3 block is ordered channel-major, because torch's F.unfold
 *     lays a block out as (channel, kh, kw) -- read it row-major and the projection
 *     sees a permuted image.
 */
static void vision_load(Model *m, Vision *v) {
    Cfg *c = &m->c;
    if (c->vision_layers < 1) { v->active = 0; return; }
    if (!st_has(&m->S, "vision.patch_embed.proj.weight")) { v->active = 0; return; }
    char name[256];
    #define VNAME(fmt, ...) (snprintf(name, sizeof(name), fmt, __VA_ARGS__), name)
    int vd = c->vision_dim, patch_in = 3 * c->vision_patch * c->vision_patch;
    wb_load(&m->S, &v->proj_w, "vision.patch_embed.proj.weight", vd, patch_in);
    wf_load(&m->S, &v->proj_b, "vision.patch_embed.proj.bias", vd);
    v->block = xmalloc((size_t)c->vision_layers * sizeof(VisionBlock), "vision blocks");
    for (int i = 0; i < c->vision_layers; i++) {
        VisionBlock *b = &v->block[i];
        wf_load(&m->S, &b->norm1, VNAME("vision.blocks.%d.norm1.weight", i), vd);
        wf_load(&m->S, &b->norm2, VNAME("vision.blocks.%d.norm2.weight", i), vd);
        wb_load(&m->S, &b->qkv_w, VNAME("vision.blocks.%d.attn.wqkv.weight", i), 3 * vd, vd);
        wf_load(&m->S, &b->qkv_b, VNAME("vision.blocks.%d.attn.wqkv.bias", i), 3 * vd);
        wb_load(&m->S, &b->o_w,   VNAME("vision.blocks.%d.attn.wo.weight", i), vd, vd);
        wf_load(&m->S, &b->o_b,   VNAME("vision.blocks.%d.attn.wo.bias", i), vd);
        wb_load(&m->S, &b->w1,    VNAME("vision.blocks.%d.mlp.w1.weight", i), 2 * c->vision_inter, vd);
        wb_load(&m->S, &b->w2,    VNAME("vision.blocks.%d.mlp.w2.weight", i), vd, c->vision_inter);
    }
    wf_load(&m->S, &v->norm, "vision.norm.weight", vd);
    int ratio = c->vision_ratio;
    wb_load(&m->S, &v->align_w1, "aligner.w1.weight", c->dim, vd * ratio * ratio);
    wf_load(&m->S, &v->align_b1, "aligner.w1.bias", c->dim);
    wb_load(&m->S, &v->align_w2, "aligner.w2.weight", c->dim, c->dim);
    wf_load(&m->S, &v->align_b2, "aligner.w2.bias", c->dim);
    wf_load(&m->S, &v->image_start, "image_start", c->dim);
    wf_load(&m->S, &v->image_end, "image_end", c->dim);
    wf_load(&m->S, &v->image_newline, "image_newline", c->dim);
    #undef VNAME
    v->active = 1;
}

/* How many rows the aligner returns for a grid: the grid is padded up to the ratio. */
static int vision_tokens(const Cfg *c, int n_h, int n_w) {
    int ratio = c->vision_ratio;
    return ((n_h + ratio - 1) / ratio) * ((n_w + ratio - 1) / ratio);
}

/* patches: [n_h * n_w, 3 * patch * patch] row-major. out: [vision_tokens, dim]. */
static void vision_forward(Model *m, Vision *v, const float *patches, int n_h, int n_w, float *out) {
    Cfg *c = &m->c;
    int vd = c->vision_dim, heads = c->vision_heads, hd = vd / heads, half = hd / 2;
    int n = n_h * n_w, patch_in = 3 * c->vision_patch * c->vision_patch;
    float *x = xmalloc((size_t)n * vd * sizeof(float), "vision stream");
    for (int i = 0; i < n; i++) {
        mvb(x + (size_t)i * vd, &v->proj_w, patches + (size_t)i * patch_in);
        for (int j = 0; j < vd; j++) x[(size_t)i * vd + j] += v->proj_b.w[j];
    }
    /* 2D RoPE tables: one angle per (patch, frequency), row and column interleaved
     * exactly as vision.py stacks them */
    float *cosine = xmalloc((size_t)n * half * sizeof(float), "vision cos");
    float *sine = xmalloc((size_t)n * half * sizeof(float), "vision sin");
    for (int p = 0; p < n; p++) {
        int row = p / n_w, col = p % n_w;
        for (int f = 0; f < half; f++) {
            int axis = f < half / 2 ? 0 : 1;            /* first half rows, second columns */
            int index = axis ? f - half / 2 : f;
            float inv = 1.0f / powf(c->vision_rope_theta, (float)(2 * index) / (float)half);
            float angle = (axis ? col : row) * inv;
            cosine[(size_t)p * half + f] = cosf(angle);
            sine[(size_t)p * half + f] = sinf(angle);
        }
    }
    float *qkv = xmalloc((size_t)3 * vd * sizeof(float), "vision qkv");
    float *q = xmalloc((size_t)n * vd * sizeof(float), "vision q");
    float *k = xmalloc((size_t)n * vd * sizeof(float), "vision k");
    float *val = xmalloc((size_t)n * vd * sizeof(float), "vision v");
    float *normed = xmalloc((size_t)n * vd * sizeof(float), "vision normed");
    float *attended = xmalloc((size_t)n * vd * sizeof(float), "vision attended");
    float *hidden = xmalloc((size_t)2 * c->vision_inter * sizeof(float), "vision mlp");
    float *scores = xmalloc((size_t)n * sizeof(float), "vision scores");
    for (int layer = 0; layer < c->vision_layers; layer++) {
        VisionBlock *b = &v->block[layer];
        for (int i = 0; i < n; i++)
            rms_into(normed + (size_t)i * vd, x + (size_t)i * vd, b->norm1.w, vd, 1e-6f);
        for (int i = 0; i < n; i++) {
            mvb(qkv, &b->qkv_w, normed + (size_t)i * vd);
            for (int j = 0; j < 3 * vd; j++) qkv[j] += b->qkv_b.w[j];
            for (int h = 0; h < heads; h++) {
                const float *co = cosine + (size_t)i * half, *si = sine + (size_t)i * half;
                for (int part = 0; part < 2; part++) {
                    const float *src = qkv + part * vd + (size_t)h * hd;
                    float *dst = (part ? k : q) + (size_t)i * vd + (size_t)h * hd;
                    for (int f = 0; f < half; f++) {
                        float a = src[f], bb = src[f + half];
                        dst[f] = a * co[f] - bb * si[f];
                        dst[f + half] = bb * co[f] + a * si[f];
                    }
                }
                memcpy(val + (size_t)i * vd + (size_t)h * hd, qkv + 2 * vd + (size_t)h * hd,
                       (size_t)hd * sizeof(float));
            }
        }
        float scale = 1.0f / sqrtf((float)hd);
        for (int i = 0; i < n; i++)
            for (int h = 0; h < heads; h++) {
                const float *qi = q + (size_t)i * vd + (size_t)h * hd;
                float best = -INFINITY;
                for (int j = 0; j < n; j++) {
                    const float *kj = k + (size_t)j * vd + (size_t)h * hd;
                    float dot = 0.0f;
                    for (int d = 0; d < hd; d++) dot += qi[d] * kj[d];
                    scores[j] = dot * scale;
                    if (scores[j] > best) best = scores[j];
                }
                float total = 0.0f;
                for (int j = 0; j < n; j++) { scores[j] = expf(scores[j] - best); total += scores[j]; }
                float *dst = attended + (size_t)i * vd + (size_t)h * hd;
                for (int d = 0; d < hd; d++) dst[d] = 0.0f;
                for (int j = 0; j < n; j++) {
                    const float *vj = val + (size_t)j * vd + (size_t)h * hd;
                    float weight = scores[j] / total;
                    for (int d = 0; d < hd; d++) dst[d] += weight * vj[d];
                }
            }
        for (int i = 0; i < n; i++) {
            float projected[4096];
            if (vd > (int)(sizeof(projected) / sizeof(projected[0]))) {
                fprintf(stderr, "[vision] vision_dim %d exceeds the scratch\n", vd); exit(1); }
            mvb(projected, &b->o_w, attended + (size_t)i * vd);
            for (int j = 0; j < vd; j++) x[(size_t)i * vd + j] += projected[j] + b->o_b.w[j];
        }
        for (int i = 0; i < n; i++)
            rms_into(normed + (size_t)i * vd, x + (size_t)i * vd, b->norm2.w, vd, 1e-6f);
        for (int i = 0; i < n; i++) {
            float projected[4096];
            mvb(hidden, &b->w1, normed + (size_t)i * vd);
            for (int j = 0; j < c->vision_inter; j++) {
                float gate = hidden[j], up = hidden[c->vision_inter + j];
                hidden[j] = (gate / (1.0f + expf(-gate))) * up;
            }
            mvb(projected, &b->w2, hidden);
            for (int j = 0; j < vd; j++) x[(size_t)i * vd + j] += projected[j];
        }
    }
    for (int i = 0; i < n; i++)
        rms_into(normed + (size_t)i * vd, x + (size_t)i * vd, v->norm.w, vd, 1e-6f);

    /* aligner: pad the grid, take ratio x ratio blocks channel-major, project */
    int ratio = c->vision_ratio;
    int blocks_h = (n_h + ratio - 1) / ratio, blocks_w = (n_w + ratio - 1) / ratio;
    int block_dim = vd * ratio * ratio;
    float *block = xmalloc((size_t)block_dim * sizeof(float), "aligner block");
    float *projected = xmalloc((size_t)c->dim * sizeof(float), "aligner hidden");
    for (int bh = 0; bh < blocks_h; bh++)
        for (int bw = 0; bw < blocks_w; bw++) {
            for (int ch = 0; ch < vd; ch++)
                for (int kh = 0; kh < ratio; kh++)
                    for (int kw = 0; kw < ratio; kw++) {
                        int row = bh * ratio + kh, col = bw * ratio + kw;
                        float value = 0.0f;      /* F.pad fills the ragged edge with zeros */
                        if (row < n_h && col < n_w)
                            value = normed[(size_t)(row * n_w + col) * vd + ch];
                        block[((size_t)ch * ratio + kh) * ratio + kw] = value;
                    }
            float *dest = out + (size_t)(bh * blocks_w + bw) * c->dim;
            mvb(projected, &v->align_w1, block);
            for (int j = 0; j < c->dim; j++) {
                float value = projected[j] + v->align_b1.w[j];
                /* exact GELU, matching torch's default (erf, not the tanh approximation) */
                projected[j] = 0.5f * value * (1.0f + erff(value / sqrtf(2.0f)));
            }
            mvb(dest, &v->align_w2, projected);
            for (int j = 0; j < c->dim; j++) dest[j] += v->align_b2.w[j];
        }
    free(projected); free(block); free(scores); free(hidden); free(attended);
    free(normed); free(val); free(k); free(q); free(qkv); free(sine); free(cosine); free(x);
}

/* -------------------------------------------------------------- forward ---- */

/* The mix coefficients a sublayer computes are used by the NEXT one (model.py
 * Block.forward): attention collapses with what the previous block's FFN produced,
 * the FFN with what this attention produced. Keeping `pre` out of coli_hc_pre is
 * exactly why this engine calls coli_hc_split_sinkhorn directly. */
static void hc_mixes(Model *m, const float *hc_fn, const float *scale, const float *base,
                     const float *stream, float *pre, float *post, float *comb) {
    Cfg *c = &m->c;
    int hc = c->hc_mult, dim = c->dim, flat = hc * dim, mix = (2 + hc) * hc;
    double square = 0.0;
    for (int i = 0; i < flat; i++) square += (double)stream[i] * stream[i];
    float inverse = 1.0f / sqrtf((float)(square / flat) + c->norm_eps);
    float *mixes = xmalloc((size_t)mix * sizeof(float), "hc mixes");
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < mix; row++) {
        const float *w = hc_fn + (size_t)row * flat;
        float sum = 0.0f;
        for (int i = 0; i < flat; i++) sum += w[i] * stream[i];
        mixes[row] = sum * inverse;
    }
    if (coli_hc_split_sinkhorn(pre, post, comb, mixes, scale, base, hc, c->hc_iters, c->hc_eps) != 0) {
        fprintf(stderr, "[hc] sinkhorn refused its arguments\n"); exit(1); }
    free(mixes);
}

/* Run one chunk: the whole prompt on the first call, one token per call after.
 * Returns the logits of the last position (the caller owns the buffer). */
/* `image_rows` (or NULL) is the aligner's output for a pending image and `image_at`
 * the position where its span starts; the span's embeddings are replaced rather than
 * looked up. `image_mask` marks those positions for the engram. */
/* ---------------------------------------------------------------- DSpark ---- */

static int argmax(const float *values, int n);

static void spec_load(Model *m, int ecap) {
    Cfg *c = &m->c;
    Spec *sp = &m->spec;
    sp->min_accept = getenv("V41_DSPARK_MINACC") ? atoi(getenv("V41_DSPARK_MINACC")) : 30;
    /* how many of the drafted tokens to put in front of the main model. The head always
     * writes its whole block; verifying fewer of them trades acceptance for a smaller
     * bill when a round is rejected early. */
    sp->max_verify = getenv("V41_DSPARK_MAX") ? atoi(getenv("V41_DSPARK_MAX")) : 0;
    if (sp->min_accept < 0) sp->min_accept = 0;
    if (sp->min_accept > 100) sp->min_accept = 100;
    if (c->n_mtp <= 0) return;
    const char *flag = getenv("V41_DSPARK");
    if (flag && !atoi(flag)) {
        fprintf(stderr, "[v41] DSpark drafts off (V41_DSPARK=0)\n");
        return;
    }
    /* A checkpoint can declare the head in its config and still not ship it -- the
     * vendor's own converter drops the tied embed and head, and a partial download has
     * the same shape. Refusing to start over a missing draft head would be absurd, so
     * say so once and decode without it. */
    if (!st_find(&m->S, "mtp.0.attn.wq_a.weight")) {
        fprintf(stderr, "[v41] no DSpark head in this checkpoint: drafts off\n");
        return;
    }
    int dim = c->dim, hd = c->head_dim, nh = c->n_heads, hc = c->hc_mult;
    sp->stage = xmalloc((size_t)c->n_mtp * sizeof(Layer), "DSpark stages");
    memset(sp->stage, 0, (size_t)c->n_mtp * sizeof(Layer));
    sp->cache = xmalloc((size_t)c->n_mtp * sizeof(LCache), "DSpark expert caches");
    char name[256];
    #define MNAME(fmt, ...) (snprintf(name, sizeof(name), fmt, __VA_ARGS__), name)
    for (int i = 0; i < c->n_mtp; i++) {
        Layer *l = &sp->stage[i];
        l->engram_index = -1;
        w8_load(&m->S, &l->wq_a, MNAME("mtp.%d.attn.wq_a.weight", i), c->q_lora, dim);
        w8_load(&m->S, &l->wq_b, MNAME("mtp.%d.attn.wq_b.weight", i), nh * hd, c->q_lora);
        w8_load(&m->S, &l->wkv,  MNAME("mtp.%d.attn.wkv.weight", i), hd, dim);
        w8_load(&m->S, &l->wo_a, MNAME("mtp.%d.attn.wo_a.weight", i),
                c->o_groups * c->o_lora, nh * hd / c->o_groups);
        w8_load(&m->S, &l->wo_b, MNAME("mtp.%d.attn.wo_b.weight", i), dim, c->o_groups * c->o_lora);
        wf_load(&m->S, &l->q_norm,    MNAME("mtp.%d.attn.q_norm.weight", i), c->q_lora);
        wf_load(&m->S, &l->kv_norm,   MNAME("mtp.%d.attn.kv_norm.weight", i), hd);
        wf_load(&m->S, &l->attn_sink, MNAME("mtp.%d.attn.attn_sink", i), nh);
        wf_load(&m->S, &l->attn_norm, MNAME("mtp.%d.attn_norm.weight", i), dim);
        wf_load(&m->S, &l->ffn_norm,  MNAME("mtp.%d.ffn_norm.weight", i), dim);
        int mix = (2 + hc) * hc;
        wf_load(&m->S, &l->hc_attn_fn,    MNAME("mtp.%d.hc_attn_fn", i), (int64_t)mix * hc * dim);
        wf_load(&m->S, &l->hc_ffn_fn,     MNAME("mtp.%d.hc_ffn_fn", i), (int64_t)mix * hc * dim);
        wf_load(&m->S, &l->hc_attn_base,  MNAME("mtp.%d.hc_attn_base", i), mix);
        wf_load(&m->S, &l->hc_ffn_base,   MNAME("mtp.%d.hc_ffn_base", i), mix);
        wf_load(&m->S, &l->hc_attn_scale, MNAME("mtp.%d.hc_attn_scale", i), 3);
        wf_load(&m->S, &l->hc_ffn_scale,  MNAME("mtp.%d.hc_ffn_scale", i), 3);
        wb_load(&m->S, &l->gate_w, MNAME("mtp.%d.ffn.gate.weight", i), c->spec_routed, dim);
        wf_load(&m->S, &l->gate_bias, MNAME("mtp.%d.ffn.gate.bias", i), c->spec_routed);
        w8_load(&m->S, &l->sh_w1, MNAME("mtp.%d.ffn.shared_experts.w1.weight", i), c->moe_inter, dim);
        w8_load(&m->S, &l->sh_w3, MNAME("mtp.%d.ffn.shared_experts.w3.weight", i), c->moe_inter, dim);
        w8_load(&m->S, &l->sh_w2, MNAME("mtp.%d.ffn.shared_experts.w2.weight", i), dim, c->moe_inter);
        l->window = xmalloc((size_t)c->window * hd * sizeof(float), "DSpark window ring");
        memset(l->window, 0, (size_t)c->window * hd * sizeof(float));
        l->window_pos = xmalloc((size_t)c->window * sizeof(int), "DSpark window positions");
        for (int k = 0; k < c->window; k++) l->window_pos[k] = -1;
        cache_init(m, &sp->cache[i], ecap);
    }
    w8_load(&m->S, &sp->main_proj, "mtp.0.main_proj.weight", dim, dim * c->n_spec_targets);
    wf_load(&m->S, &sp->main_norm, "mtp.0.main_norm.weight", dim);
    int last = c->n_mtp - 1;
    wf_load(&m->S, &sp->norm, MNAME("mtp.%d.norm.weight", last), dim);
    wb_load(&m->S, &sp->markov_embed, MNAME("mtp.%d.markov_head.embed.weight", last),
            c->vocab, c->markov_rank);
    wb_load(&m->S, &sp->markov_out, MNAME("mtp.%d.markov_head.head.weight", last),
            c->vocab, c->markov_rank);
    wb_load(&m->S, &sp->conf_proj, MNAME("mtp.%d.confidence_head.proj.weight", last),
            1, dim + c->markov_rank);
    #undef MNAME
    sp->active = 1;
    fprintf(stderr, "[v41] DSpark on: %d stages, %d experts (%d routed), %d tokens per "
                    "draft — V41_DSPARK=0 turns it off\n",
            c->n_mtp, c->spec_routed, c->spec_activated, c->spec_block);
}

/* model.py DSparkAttention.forward.
 *
 * `main_x` carries the positions the main model has just committed: their keys go into
 * this stage's ring, which is what makes the draft head see the real sequence rather
 * than its own guesses. With `out` NULL that is all this does, which is the prefill
 * call and the catch-up for positions that were accepted from a previous draft.
 *
 * The drafts then query that ring plus each other. Every draft row reads every other
 * one, future included: the vendor hands the same index list to all of them, and it is
 * right that it does -- the block is one guess, not a sequence anything committed to. */
static void spec_attention(Model *m, int stage, const float *x, int block, int start_pos,
                           const float *main_x, int main_rows, float *out) {
    Cfg *c = &m->c;
    Layer *l = &m->spec.stage[stage];
    int dim = c->dim, hd = c->head_dim, nh = c->n_heads, rd = c->rope_dim, win = c->window;
    const float *rope = m->rope_window;          /* a stage never compresses */
    double started = now_s();
    float *scratch = xmalloc((size_t)(nh * hd > dim ? nh * hd : dim) * sizeof(float), "spec scratch");

    for (int t = 0; t < main_rows; t++) {
        int pos = start_pos + t;
        float *slot = l->window + (size_t)(pos % win) * hd;
        mv8(scratch, &l->wkv, main_x + (size_t)t * dim);
        rms_into(slot, scratch, l->kv_norm.w, hd, c->norm_eps);
        rope_apply(slot + hd - rd, rope, pos, rd, 0);
        l->window_pos[pos % win] = pos;
    }
    if (!out) { free(scratch); m->t_attn += now_s() - started; return; }

    float *qr = xmalloc((size_t)c->q_lora * sizeof(float), "spec q latent");
    float *q = xmalloc((size_t)block * nh * hd * sizeof(float), "spec queries");
    float *kv = xmalloc((size_t)block * hd * sizeof(float), "spec kv");
    for (int i = 0; i < block; i++) {
        int pos = start_pos + main_rows + i;
        mv8(scratch, &l->wq_a, x + (size_t)i * dim);
        rms_into(qr, scratch, l->q_norm.w, c->q_lora, c->norm_eps);
        mv8(q + (size_t)i * nh * hd, &l->wq_b, qr);
        for (int h = 0; h < nh; h++)
            rope_apply(q + (size_t)i * nh * hd + (size_t)h * hd + hd - rd, rope, pos, rd, 0);
        mv8(scratch, &l->wkv, x + (size_t)i * dim);
        rms_into(kv + (size_t)i * hd, scratch, l->kv_norm.w, hd, c->norm_eps);
        rope_apply(kv + (size_t)i * hd + hd - rd, rope, pos, rd, 0);
    }

    int filled = start_pos + main_rows;
    int reach = win < filled ? win : filled;
    int total = reach + block;
    int *idx = xmalloc((size_t)total * sizeof(int), "spec indices");
    for (int j = 0; j < reach; j++) idx[j] = j;
    for (int j = 0; j < block; j++) idx[reach + j] = win + j;
    float *kv_all = xmalloc((size_t)(win + block) * hd * sizeof(float), "spec kv all");
    memcpy(kv_all, l->window, (size_t)win * hd * sizeof(float));
    memcpy(kv_all + (size_t)win * hd, kv, (size_t)block * hd * sizeof(float));

    float *heads = xmalloc((size_t)nh * hd * sizeof(float), "spec heads");
    float scale = 1.0f / sqrtf((float)hd);
    for (int i = 0; i < block; i++) {
        int pos = start_pos + main_rows + i;
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < nh; h++)
            sparse_attend(heads + (size_t)h * hd, q + (size_t)i * nh * hd + (size_t)h * hd,
                          kv_all, idx, total, nh, hd, l->attn_sink.w[h], scale);
        for (int h = 0; h < nh; h++)
            rope_apply(heads + (size_t)h * hd + hd - rd, rope, pos, rd, 1);
        attn_project_out(m, l, heads, out + (size_t)i * dim);
    }
    m->t_attn += now_s() - started;
    free(heads); free(kv_all); free(idx); free(kv); free(q); free(qr); free(scratch);
}

/* One DSpark round. `main_rows` positions have just been committed by the main model,
 * starting at `start_pos`; `token` is the one it produced from the last of them.
 *
 * Returns how many drafts were written, 0 when there is nothing to draft from -- the
 * prefill call, which exists only to fill the stages' rings from the prompt. */
static int spec_step(Model *m, int token, int start_pos, int main_rows,
                     int *draft, float *confidence) {
    Cfg *c = &m->c;
    Spec *sp = &m->spec;
    if (!sp->active || !m->main_hidden || m->main_hidden_rows < main_rows) return 0;
    int dim = c->dim, hc = c->hc_mult, block = c->spec_block;
    int targets = c->n_spec_targets, rank = c->markov_rank, vocab = c->vocab;
    double started = now_s();

    float *scratch = xmalloc((size_t)dim * sizeof(float), "spec projection");
    float *main_x = xmalloc((size_t)main_rows * dim * sizeof(float), "spec main input");
    for (int t = 0; t < main_rows; t++) {
        mv8(scratch, &sp->main_proj, m->main_hidden + (size_t)t * targets * dim);
        rms_into(main_x + (size_t)t * dim, scratch, sp->main_norm.w, dim, c->norm_eps);
    }
    if (start_pos == 0) {
        for (int stage = 0; stage < c->n_mtp; stage++)
            spec_attention(m, stage, NULL, 0, 0, main_x, main_rows, NULL);
        free(main_x); free(scratch);
        return 0;
    }

    float *h = xmalloc((size_t)block * hc * dim * sizeof(float), "spec streams");
    for (int i = 0; i < block; i++) {
        int id = i == 0 ? token : c->spec_noise;
        const uint16_t *row = m->embed.w + (size_t)id * dim;
        for (int copy = 0; copy < hc; copy++)
            for (int j = 0; j < dim; j++)
                h[((size_t)i * hc + copy) * dim + j] = bf16_to_f32(row[j]);
    }
    float *pre_mix = xmalloc((size_t)block * hc * sizeof(float), "spec pre mix");
    for (int i = 0; i < block; i++)
        for (int copy = 0; copy < hc; copy++) pre_mix[(size_t)i * hc + copy] = copy == 0 ? 1.0f : 0.0f;

    float *branch_in = xmalloc((size_t)block * dim * sizeof(float), "spec sublayer in");
    float *branch_out = xmalloc((size_t)block * dim * sizeof(float), "spec sublayer out");
    float *residual = xmalloc((size_t)block * hc * dim * sizeof(float), "spec residual");
    float *pre = xmalloc((size_t)block * hc * sizeof(float), "spec hc pre");
    float *post = xmalloc((size_t)block * hc * sizeof(float), "spec hc post");
    float *comb = xmalloc((size_t)block * hc * hc * sizeof(float), "spec hc comb");
    float *collapsed = xmalloc((size_t)dim * sizeof(float), "spec collapsed");

    for (int stage = 0; stage < c->n_mtp; stage++) {
        Layer *l = &sp->stage[stage];
        memcpy(residual, h, (size_t)block * hc * dim * sizeof(float));
        for (int i = 0; i < block; i++) {
            hc_mixes(m, l->hc_attn_fn.w, l->hc_attn_scale.w, l->hc_attn_base.w,
                     h + (size_t)i * hc * dim, pre + (size_t)i * hc,
                     post + (size_t)i * hc, comb + (size_t)i * hc * hc);
            const float *mix = pre_mix + (size_t)i * hc;
            for (int j = 0; j < dim; j++) {
                float sum = 0.0f;
                for (int copy = 0; copy < hc; copy++)
                    sum += mix[copy] * h[((size_t)i * hc + copy) * dim + j];
                collapsed[j] = sum;
            }
            rms_into(branch_in + (size_t)i * dim, collapsed, l->attn_norm.w, dim, c->norm_eps);
        }
        spec_attention(m, stage, branch_in, block, start_pos, main_x, main_rows, branch_out);
        for (int i = 0; i < block; i++)
            coli_hc_post(h + (size_t)i * hc * dim, branch_out + (size_t)i * dim,
                         residual + (size_t)i * hc * dim, post + (size_t)i * hc,
                         comb + (size_t)i * hc * hc, hc, dim);
        memcpy(pre_mix, pre, (size_t)block * hc * sizeof(float));

        memcpy(residual, h, (size_t)block * hc * dim * sizeof(float));
        for (int i = 0; i < block; i++) {
            hc_mixes(m, l->hc_ffn_fn.w, l->hc_ffn_scale.w, l->hc_ffn_base.w,
                     h + (size_t)i * hc * dim, pre + (size_t)i * hc,
                     post + (size_t)i * hc, comb + (size_t)i * hc * hc);
            const float *mix = pre_mix + (size_t)i * hc;
            for (int j = 0; j < dim; j++) {
                float sum = 0.0f;
                for (int copy = 0; copy < hc; copy++)
                    sum += mix[copy] * h[((size_t)i * hc + copy) * dim + j];
                collapsed[j] = sum;
            }
            rms_into(branch_in + (size_t)i * dim, collapsed, l->ffn_norm.w, dim, c->norm_eps);
            moe_run_at(m, l, &sp->cache[stage], "mtp", stage, c->spec_routed, c->spec_activated,
                       branch_in + (size_t)i * dim, branch_out + (size_t)i * dim);
        }
        for (int i = 0; i < block; i++)
            coli_hc_post(h + (size_t)i * hc * dim, branch_out + (size_t)i * dim,
                         residual + (size_t)i * hc * dim, post + (size_t)i * hc,
                         comb + (size_t)i * hc * hc, hc, dim);
        memcpy(pre_mix, pre, (size_t)block * hc * sizeof(float));
    }

    /* the head, then the Markov chain over the block: each position is biased by the
     * token drafted for the one before it, which is the only thing tying the block
     * together -- the stages saw the noise id at every position but the first */
    float *final_x = xmalloc((size_t)block * dim * sizeof(float), "spec final");
    float *logits = xmalloc((size_t)block * vocab * sizeof(float), "spec logits");
    float *bias = xmalloc((size_t)vocab * sizeof(float), "spec markov bias");
    float *embed = xmalloc((size_t)rank * sizeof(float), "spec markov embed");
    float *joined = xmalloc((size_t)(dim + rank) * sizeof(float), "spec confidence input");
    for (int i = 0; i < block; i++) {
        const float *mix = pre_mix + (size_t)i * hc;
        float *x = final_x + (size_t)i * dim;
        for (int j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (int copy = 0; copy < hc; copy++)
                sum += mix[copy] * h[((size_t)i * hc + copy) * dim + j];
            x[j] = sum;
        }
        rms_into(collapsed, x, sp->norm.w, dim, c->norm_eps);
        mvb(logits + (size_t)i * vocab, &m->head, collapsed);
    }
    draft[0] = token;
    for (int i = 0; i < block; i++) {
        const uint16_t *row = sp->markov_embed.w + (size_t)draft[i] * rank;
        for (int j = 0; j < rank; j++) embed[j] = bf16_to_f32(row[j]);
        mvb(bias, &sp->markov_out, embed);
        float *row_logits = logits + (size_t)i * vocab;
        for (int v = 0; v < vocab; v++) row_logits[v] += bias[v];
        draft[i + 1] = argmax(row_logits, vocab);     /* greedy: the drafts are verified */
        memcpy(joined, final_x + (size_t)i * dim, (size_t)dim * sizeof(float));
        memcpy(joined + dim, embed, (size_t)rank * sizeof(float));
        float score = 0.0f;
        mvb(&score, &sp->conf_proj, joined);
        confidence[i] = score;
    }
    sp->rounds++;
    m->t_spec += now_s() - started;
    free(joined); free(embed); free(bias); free(logits); free(final_x);
    free(collapsed); free(comb); free(post); free(pre); free(residual);
    free(branch_out); free(branch_in); free(pre_mix); free(h);
    free(main_x); free(scratch);
    return block;
}

static void forward_full(Model *m, const int *ids, int n, float *logits, int all_logits,
                         const float *image_rows, int image_at, int image_h, int image_w,
                         const uint8_t *image_mask);

static void forward(Model *m, const int *ids, int n, float *logits) {
    forward_full(m, ids, n, logits, 0, NULL, -1, 0, 0, NULL);
}

static void forward_with_image(Model *m, const int *ids, int n, float *logits,
                               const float *image_rows, int image_at, int image_h, int image_w,
                               const uint8_t *image_mask) {
    forward_full(m, ids, n, logits, 0, image_rows, image_at, image_h, image_w, image_mask);
}

/* Several positions past the prompt, with one row of logits each: what a speculative
 * step verifies its drafts with. Every piece of per-position state below -- the window
 * ring's position map, the compressor's group slots, the indexer's reach, the index
 * keys each row reads -- is written so that this is the same computation as the same
 * tokens decoded one at a time, because a draft is only worth anything if accepting it
 * is indistinguishable from having generated it. */
static void forward_batch(Model *m, const int *ids, int n, float *logits) {
    forward_full(m, ids, n, logits, 1, NULL, -1, 0, 0, NULL);
}

static void forward_full(Model *m, const int *ids, int n, float *logits, int all_logits,
                         const float *image_rows, int image_at, int image_h, int image_w,
                         const uint8_t *image_mask) {
    Cfg *c = &m->c;
    int dim = c->dim, hc = c->hc_mult;
    int start_pos = m->pos;
    /* The index-key schedule for a batched step: which layer publishes at each of its
     * sub-steps. It has to be known before the layers run, because a layer that comes
     * later in the stack still publishes before this layer's later rows read. */
    m->pub_before = m->published_index_k;
    m->pub_before_layer = m->published_index_layer;
    m->pub_rows = 0;
    if (n > 1 && start_pos > 0) {
        m->pub_layer = realloc(m->pub_layer, (size_t)n * sizeof(int));
        if (!m->pub_layer) { fprintf(stderr, "OOM sizing the publish schedule\n"); exit(1); }
        m->pub_rows = n;
        for (int t = 0; t < n; t++) {
            m->pub_layer[t] = -1;
            for (int layer = 0; layer < c->n_layers; layer++)
                if (layer_publishes(c, layer, start_pos + t)) m->pub_layer[t] = layer;
        }
    }
    if (start_pos + n > c->max_positions) {
        fprintf(stderr, "CONTEXT_EXCEEDED %d %d\n", start_pos + n, c->max_positions);
        exit(1);
    }
    if (m->engram.active) engram_push(&m->engram, ids, n, image_mask);

    float *h = xmalloc((size_t)n * hc * dim * sizeof(float), "residual streams");
    float *embedded = xmalloc((size_t)dim * sizeof(float), "embedding row");
    for (int t = 0; t < n; t++) {
        const uint16_t *row = m->embed.w + (size_t)ids[t] * dim;
        for (int i = 0; i < dim; i++) embedded[i] = bf16_to_f32(row[i]);
        /* An image span carries the placeholder id at every position; what each one
         * MEANS follows from the grid, exactly as image_processor.py lays it out:
         *   [start] then h rows of (w image tokens + newline) then [end].
         * The delimiters take learned embeddings, the image slots take the aligner's
         * rows in order. Reconstructing it from the grid keeps the wire protocol at
         * one IMAGE frame plus placeholders, with no parallel token-type channel to
         * fall out of step with the ids. */
        if (image_rows && m->vision && t >= image_at) {
            int offset = t - image_at;
            int span = 1 + (image_w + 1) * image_h + 1;
            if (offset < span) {
                const float *source = NULL;
                if (offset == 0) source = m->vision->image_start.w;
                else if (offset == span - 1) source = m->vision->image_end.w;
                else {
                    int inner = offset - 1;
                    int row_index = inner / (image_w + 1), column = inner % (image_w + 1);
                    source = column == image_w ? m->vision->image_newline.w
                                               : image_rows + (size_t)(row_index * image_w + column) * dim;
                }
                memcpy(embedded, source, (size_t)dim * sizeof(float));
            }
        }
        for (int copy = 0; copy < hc; copy++)
            memcpy(h + ((size_t)t * hc + copy) * dim, embedded, (size_t)dim * sizeof(float));
    }
    free(embedded);
    float *pre_mix = xmalloc((size_t)n * hc * sizeof(float), "pre mix");
    for (int t = 0; t < n; t++)
        for (int copy = 0; copy < hc; copy++) pre_mix[(size_t)t * hc + copy] = copy == 0 ? 1.0f : 0.0f;

    /* V41_TRACE=2 follows the FIRST row instead of the last: on a speculative step
     * that is the position the sequential decode would have run on its own, which is
     * what a divergence has to be compared against. */
    int trace_row = (g_trace >= 2 && n > 1) ? 0 : n - 1;
    float *branch_in = xmalloc((size_t)n * dim * sizeof(float), "sublayer input");
    float *branch_out = xmalloc((size_t)n * dim * sizeof(float), "sublayer output");
    float *residual = xmalloc((size_t)n * hc * dim * sizeof(float), "residual copy");
    float *pre = xmalloc((size_t)n * hc * sizeof(float), "hc pre");
    float *post = xmalloc((size_t)n * hc * sizeof(float), "hc post");
    float *comb = xmalloc((size_t)n * hc * hc * sizeof(float), "hc comb");
    float *collapsed = xmalloc((size_t)dim * sizeof(float), "collapsed stream");

    /* What the DSpark head reads: the attention input of its target layers, the hc
     * copies averaged. Kept even when no draft head is loaded is pointless, so it is
     * gated on the model actually having one. */
    int targets = m->spec.active ? c->n_spec_targets : 0;
    if (targets) {
        m->main_hidden = realloc(m->main_hidden,
                                 (size_t)n * targets * dim * sizeof(float));
        if (!m->main_hidden) { fprintf(stderr, "OOM sizing the DSpark input\n"); exit(1); }
        m->main_hidden_rows = n;
    }

    for (int layer = 0; layer < c->n_layers; layer++) {
        Layer *l = &m->L[layer];
        if (l->engram_index >= 0) engram_run(m, layer, h, n, start_pos);
        for (int k = 0; k < targets; k++) {
            if (c->spec_targets[k] != layer) continue;
            for (int t = 0; t < n; t++) {
                float *dest = m->main_hidden + ((size_t)t * targets + k) * dim;
                for (int i = 0; i < dim; i++) {
                    float sum = 0.0f;
                    for (int copy = 0; copy < hc; copy++)
                        sum += h[((size_t)t * hc + copy) * dim + i];
                    dest[i] = sum / (float)hc;
                }
            }
        }

        trace("stream", layer, h + (size_t)trace_row * hc * dim, hc * dim);
        memcpy(residual, h, (size_t)n * hc * dim * sizeof(float));
        for (int t = 0; t < n; t++) {
            hc_mixes(m, l->hc_attn_fn.w, l->hc_attn_scale.w, l->hc_attn_base.w,
                     h + (size_t)t * hc * dim, pre + (size_t)t * hc,
                     post + (size_t)t * hc, comb + (size_t)t * hc * hc);
            const float *mix = pre_mix + (size_t)t * hc;
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int copy = 0; copy < hc; copy++)
                    sum += mix[copy] * h[((size_t)t * hc + copy) * dim + i];
                collapsed[i] = sum;
            }
            rms_into(branch_in + (size_t)t * dim, collapsed, l->attn_norm.w, dim, c->norm_eps);
        }
        trace("attn_in", layer, branch_in + (size_t)trace_row * dim, dim);
        trace("attn_pre", layer, pre + (size_t)trace_row * hc, hc);
        trace("attn_post", layer, post + (size_t)trace_row * hc, hc);
        trace("attn_comb", layer, comb + (size_t)trace_row * hc * hc, hc * hc);
        attention_run(m, layer, branch_in, n, start_pos, branch_out);
        trace("attn_out", layer, branch_out + (size_t)trace_row * dim, dim);
        for (int t = 0; t < n; t++)
            coli_hc_post(h + (size_t)t * hc * dim, branch_out + (size_t)t * dim,
                         residual + (size_t)t * hc * dim, post + (size_t)t * hc,
                         comb + (size_t)t * hc * hc, hc, dim);
        /* the attention's own `pre` collapses the FFN input */
        memcpy(pre_mix, pre, (size_t)n * hc * sizeof(float));

        memcpy(residual, h, (size_t)n * hc * dim * sizeof(float));
        for (int t = 0; t < n; t++) {
            hc_mixes(m, l->hc_ffn_fn.w, l->hc_ffn_scale.w, l->hc_ffn_base.w,
                     h + (size_t)t * hc * dim, pre + (size_t)t * hc,
                     post + (size_t)t * hc, comb + (size_t)t * hc * hc);
            const float *mix = pre_mix + (size_t)t * hc;
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int copy = 0; copy < hc; copy++)
                    sum += mix[copy] * h[((size_t)t * hc + copy) * dim + i];
                collapsed[i] = sum;
            }
            rms_into(branch_in + (size_t)t * dim, collapsed, l->ffn_norm.w, dim, c->norm_eps);
            moe_run(m, layer, branch_in + (size_t)t * dim, branch_out + (size_t)t * dim);
            if (t == trace_row) {
                trace("ffn_in", layer, branch_in + (size_t)t * dim, dim);
                trace("ffn_out", layer, branch_out + (size_t)t * dim, dim);
            }
        }
        for (int t = 0; t < n; t++)
            coli_hc_post(h + (size_t)t * hc * dim, branch_out + (size_t)t * dim,
                         residual + (size_t)t * hc * dim, post + (size_t)t * hc,
                         comb + (size_t)t * hc * hc, hc, dim);
        memcpy(pre_mix, pre, (size_t)n * hc * sizeof(float));   /* the FFN's, for the next block */
    }

    /* the last block's FFN mix collapses the stream one final time */
    for (int t = all_logits ? 0 : n - 1; t < n; t++) {
        const float *mix = pre_mix + (size_t)t * hc;
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int copy = 0; copy < hc; copy++)
                sum += mix[copy] * h[((size_t)t * hc + copy) * dim + i];
            collapsed[i] = sum;
        }
        if (t == n - 1) trace("final", -1, collapsed, dim);
        rms_into(branch_in, collapsed, m->norm.w, dim, c->norm_eps);
        mvb(logits + (size_t)(all_logits ? t : 0) * c->vocab, &m->head, branch_in);
    }
    trace("logits", -1, logits + (size_t)(all_logits ? n - 1 : 0) * c->vocab, c->vocab);

    m->pos += n;
    m->last_start = start_pos;
    m->last_rows = n;
    m->forwards++;
    free(collapsed); free(comb); free(post); free(pre);
    free(residual); free(branch_out); free(branch_in); free(pre_mix); free(h);
}

static int argmax(const float *values, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (values[i] > values[best]) best = i;
    return best;
}

/* ---------------------------------------------------------------- serve ---- */

/* Every turn starts from a clean state: the window rings, the compressed caches, the
 * compressor's partial group and the engram history all describe one sequence. Prefix
 * reuse across turns is a later step (the KV here is four caches and a ring, not one
 * tensor), and getting it wrong would be silent, so it is refused rather than faked. */
static void model_reset(Model *m) {
    Cfg *c = &m->c;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        memset(l->window, 0, (size_t)c->window * c->head_dim * sizeof(float));
        for (int k = 0; k < c->window; k++) l->window_pos[k] = -1;
        int ratio = c->compress_ratio[i];
        if (c->kv_source[i]) {
            int slots = c->max_positions / (ratio > 0 ? ratio : 1);
            memset(l->ckv, 0, (size_t)slots * c->head_dim * sizeof(float));
            memset(l->ikey, 0, (size_t)slots * c->index_head_dim * sizeof(float));
            if (ratio > 1) {
                memset(l->cstate_kv, 0, (size_t)ratio * c->head_dim * sizeof(float));
                for (int k = 0; k < ratio * c->head_dim; k++) l->cstate_score[k] = -INFINITY;
            }
        }
    }
    m->engram.history_len = 0;
    m->published_index_k = NULL;
    m->shared_topk_rows = m->shared_topk_width = 0;
    m->pos = 0;
}

static const ColiServeWireProfile v41_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 24,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
};

typedef struct { float p; int id; } SampleProb;
static int sample_desc(const void *a, const void *b) {
    float pa = ((const SampleProb *)a)->p, pb = ((const SampleProb *)b)->p;
    return (pb > pa) - (pa > pb);
}

static int serve_sample(const float *logits, int vocab, float temperature, float top_p) {
    if (temperature <= 0.0f) return argmax(logits, vocab);
    SampleProb *rank = xmalloc((size_t)vocab * sizeof(SampleProb), "sampling");
    float top = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    for (int i = 0; i < vocab; i++) {
        float p = expf((logits[i] - top) / temperature);
        total += p;
        rank[i].p = p; rank[i].id = i;
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

/* The probability `serve_sample` would give this token: the same temperature, the same
 * top-p truncation, zero if the token falls outside the kept set. A greedy drafter
 * proposes with probability 1, so accepting with probability p(draft) is exactly the
 * speculative-sampling rule, and on a rejection the residual distribution is this one
 * with the rejected token removed -- which is what banning it in the row does. */
static double spec_accept_prob(const float *logits, int vocab, float temperature,
                               float top_p, int token) {
    SampleProb *rank = xmalloc((size_t)vocab * sizeof(SampleProb), "spec sampling");
    float top = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    for (int i = 0; i < vocab; i++) {
        float p = expf((logits[i] - top) / temperature);
        total += p;
        rank[i].p = p; rank[i].id = i;
    }
    qsort(rank, (size_t)vocab, sizeof(SampleProb), sample_desc);
    double cut = (top_p > 0.0f && top_p < 1.0f) ? top_p * total : total, kept = 0.0;
    int n = 0;
    while (n < vocab && kept < cut) kept += rank[n++].p;
    double probability = 0.0;
    for (int i = 0; i < n; i++) if (rank[i].id == token) { probability = rank[i].p / kept; break; }
    free(rank);
    return probability;
}

/* Undo the part of a speculative step that was not committed.
 *
 * Most of the state needs nothing: the window ring keeps the rejected keys but they
 * carry positions ahead of every query that follows, the compressed rows sit past what
 * any query can reach, and the compressor's group slots are addressed by position, so
 * the token that really lands there overwrites the draft before the group closes.
 * Three things are not self-healing: how far the sequence has got, the engram history,
 * and the two slots a layer publishes to -- all three are set here to what they would
 * be after decoding exactly the committed prefix. */
static void spec_rollback(Model *m, int start_pos, int committed, int rows) {
    Cfg *c = &m->c;
    int hd = c->head_dim;
    /* Newest first, so a slot two rows touched comes back to what it held before the
     * first of them -- which is the value the committed prefix left there. */
    for (int layer = 0; layer < c->n_layers; layer++) {
        Layer *l = &m->L[layer];
        for (int t = rows - 1; t >= committed; t--) {
            int slot = (start_pos + t) % c->window;
            memcpy(l->window + (size_t)slot * hd, l->ring_save + (size_t)t * hd,
                   (size_t)hd * sizeof(float));
            l->window_pos[slot] = l->ring_save_pos[t];
            if (!l->cstate_save_kv) continue;
            int group = l->cstate_save_slot[t];
            memcpy(l->cstate_kv + (size_t)group * hd, l->cstate_save_kv + (size_t)t * hd,
                   (size_t)hd * sizeof(float));
            memcpy(l->cstate_score + (size_t)group * hd, l->cstate_save_score + (size_t)t * hd,
                   (size_t)hd * sizeof(float));
        }
    }
    m->pos = start_pos + committed;
    m->last_rows = committed;
    if (m->engram.active && m->engram.history_len > m->pos) m->engram.history_len = m->pos;
    if (m->pub_rows > 0) {
        const float *published = m->pub_before;
        int layer = m->pub_before_layer;
        for (int t = committed - 1; t >= 0; t--)
            if (m->pub_layer[t] >= 0) {
                layer = m->pub_layer[t];
                published = m->L[layer].ikey;
                break;
            }
        m->published_index_k = published;
        m->published_index_layer = layer;
    }
    /* the index list the layers before the next index source read is the one for the
     * last committed position, not for the first row of the batch */
    if (m->shared_topk && m->shared_topk_width > 0 && m->shared_topk_rows > committed - 1) {
        int keep = committed - 1;
        if (keep > 0)
            memmove(m->shared_topk, m->shared_topk + (size_t)keep * m->shared_topk_width,
                    (size_t)m->shared_topk_width * sizeof(int));
        m->shared_topk_rows = 1;
    }
}

/* Verify a block of drafts in one forward. `ids[0]` is the token the main model
 * produced and is always committed; `ids[1..n-1]` are the drafts, accepted while they
 * agree with what this forward says. `logits` comes back holding the continuation of
 * the last committed position, and the caches are rolled back to it, so the caller can
 * carry on as if the accepted tokens had been decoded one at a time -- which, position
 * by position, is what they were. */
static int spec_verify(Model *m, const int *ids, int n, float *logits,
                       float temperature, float top_p) {
    int vocab = m->c.vocab, start_pos = m->pos;
    float *rows = xmalloc((size_t)n * vocab * sizeof(float), "verification logits");
    forward_batch(m, ids, n, rows);
    int accepted = 0, rejected = -1;
    while (accepted < n - 1) {
        const float *row = rows + (size_t)accepted * vocab;
        int proposed = ids[accepted + 1];
        int ok;
        if (temperature <= 0.0f) ok = argmax(row, vocab) == proposed;
        else ok = (double)rand() / RAND_MAX
                  < spec_accept_prob(row, vocab, temperature, top_p, proposed);
        if (!ok) { rejected = proposed; break; }
        accepted++;
    }
    memcpy(logits, rows + (size_t)accepted * vocab, (size_t)vocab * sizeof(float));
    if (rejected >= 0 && temperature > 0.0f) logits[rejected] = -INFINITY;
    spec_rollback(m, start_pos, accepted + 1, n);
    free(rows);
    return accepted;
}

/* Is it worth drafting right now? Drafting reads the stages' experts whether or not
 * anything is accepted, so a run of refusals is a disk bill with no token to show for
 * it. The window is the vendor-independent part of every speculative decoder: measure,
 * pause, measure again. */
static int spec_ready(Spec *sp) {
    if (!sp->active) return 0;
    if (sp->pause > 0) {
        if (--sp->pause == 0) sp->window_prop = sp->window_acc = 0;
        return 0;
    }
    if (sp->window_prop >= 24 &&
        sp->window_acc * 100 < sp->window_prop * (uint64_t)sp->min_accept) {
        fprintf(stderr, "[v41] DSpark: %.0f%% of the last %llu drafts accepted (under "
                        "%d%%), pausing for 64 tokens\n",
                100.0 * (double)sp->window_acc / (double)sp->window_prop,
                (unsigned long long)sp->window_prop, sp->min_accept);
        sp->pause = 64;
        sp->window_prop = sp->window_acc = 0;
        return 0;
    }
    return 1;
}

static void serve_line(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fflush(stdout);
}

/* --- the dashboard's four lines -------------------------------------------
 * Same bytes colibri.c emits, so the gateway parses this engine like any other:
 *   EMAP rows cols hex   the grid, after READY and STAT and after every turn
 *   HITS rows cols hex   the experts a turn routed, after DONE next to PROF
 *   PROF ...             wall, tokens, disk, wait, matmul, attention, head, forwards
 * Rows are the MoE layers (every layer here), columns the routed experts. */
static void serve_emap(Model *m) {
    Cfg *c = &m->c;
    int rows = c->n_layers, cols = c->n_routed;
    char *hex = xmalloc((size_t)rows * cols * 2 + 1, "emap");
    int w = 0;
    for (int i = 0; i < rows; i++) {
        LCache *cache = &m->cache[i];
        for (int e = 0; e < cols; e++) {
            int resident = 0;
            for (int k = 0; k < cache->n; k++) if (cache->slot[k].eid == e) { resident = 1; break; }
            int byte = resident << 6;    /* tier in bits 6-7; no usage counter here, heat 0 */
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
    int rows = c->n_layers, cols = c->n_routed, nb = (rows * cols + 7) / 8;
    uint8_t *bitmap = xmalloc((size_t)nb, "hits bitmap");
    memset(bitmap, 0, (size_t)nb);
    int bit = 0;
    for (int i = 0; i < rows; i++)
        for (int e = 0; e < cols; e++, bit++)
            if (m->ehit && m->ehit[i][e]) { bitmap[bit >> 3] |= (uint8_t)(1 << (bit & 7)); m->ehit[i][e] = 0; }
    char *hex = xmalloc((size_t)nb * 2 + 1, "hits hex");
    int w = 0;
    for (int b = 0; b < nb; b++) {
        hex[w++] = "0123456789abcdef"[bitmap[b] >> 4];
        hex[w++] = "0123456789abcdef"[bitmap[b] & 15];
    }
    hex[w] = 0;
    serve_line("HITS %d %d %s\n", rows, cols, hex);
    free(hex); free(bitmap);
}

static int serve_eos(Model *m, const char *snap, int *ids, int cap) {
    (void)m;
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb");
    int n = 0;
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = xmalloc((size_t)size + 1, "config");
    if (fread(text, 1, (size_t)size, f) == (size_t)size) {
        text[size] = 0;
        char *arena = NULL;
        jval *root = json_parse(text, &arena);
        if (root) {
            jval *eos = json_get(root, "eos_token_id");
            if (eos && eos->t == J_NUM && n < cap) ids[n++] = (int)eos->num;
            else if (eos && eos->t == J_ARR)
                for (int i = 0; i < eos->len && n < cap; i++) ids[n++] = (int)eos->kids[i]->num;
            json_free(root); free(arena);
        }
    }
    fclose(f); free(text);
    return n;
}

static void serve_loop(Model *m, Tok *tokenizer, const char *snap) {
    Cfg *c = &m->c;
    coli_serve_stdio_init();
    int eos_ids[8];
    int n_eos = serve_eos(m, snap, eos_ids, 8);
    coli_serve_write_ready(stdout, rss_gb());
    serve_emap(m);
    float *logits = xmalloc((size_t)c->vocab * sizeof(float), "logits");
    int *ids = xmalloc((size_t)c->max_positions * sizeof(int), "prompt ids");
    float *pending_image = NULL;
    int pending_h = 0, pending_w = 0;

    for (;;) {
        ColiServeCommand command;
        ColiServeReadResult result = coli_serve_read_command(stdin, &v41_wire, &command);
        if (result == COLI_SERVE_READ_EOF || result == COLI_SERVE_READ_BAD_FRAME) break;
        if (result == COLI_SERVE_READ_NOMEM) {
            coli_serve_write_error(stdout, command.id, "out of memory"); break; }
        if (result == COLI_SERVE_READ_BAD_REQUEST) {
            if (command.kind == COLI_SERVE_COMMAND_SUBMIT)
                coli_serve_write_error(stdout, command.id, "bad submit header");
            coli_serve_command_dispose(&command);
            continue;
        }
        if (command.kind == COLI_SERVE_COMMAND_IMAGE) {
            /* One image waits for the SUBMIT that names it. A second one before that
             * SUBMIT replaces the first and says so: answering about the wrong photo
             * is worse than refusing. */
            if (!m->vision) {
                coli_serve_write_error(stdout, command.id, "this container has no vision tower");
                coli_serve_command_dispose(&command);
                continue;
            }
            int patch_in = 3 * c->vision_patch * c->vision_patch;
            uint64_t expected = (uint64_t)command.grid_h * command.grid_w * patch_in * sizeof(float);
            if (command.grid_h < 1 || command.grid_w < 1 || command.payload_bytes != expected) {
                coli_serve_write_error(stdout, command.id, "IMAGE payload does not match its grid");
                coli_serve_command_dispose(&command);
                continue;
            }
            if (pending_image) fprintf(stderr, "[v41] a second IMAGE replaced the first\n");
            free(pending_image);
            pending_image = (float *)coli_serve_command_take_payload(&command);
            pending_h = command.grid_h; pending_w = command.grid_w;
            coli_serve_command_dispose(&command);
            continue;
        }
        if (command.kind != COLI_SERVE_COMMAND_SUBMIT) {
            /* CANCEL and STOP for a turn that is not running: the protocol's NOT_FOUND */
            if (command.kind == COLI_SERVE_COMMAND_CANCEL)
                coli_serve_write_error(stdout, command.id, "NOT_FOUND");
            coli_serve_command_dispose(&command);
            continue;
        }
        double turn_started = now_s();
        double disk0 = m->t_disk, expert0 = m->t_expert, attn0 = m->t_attn, engram0 = m->t_engram;
        uint64_t forwards0 = m->forwards, hits0 = m->hits, miss0 = m->miss;
        model_reset(m);
        int n_prompt = tok_encode(tokenizer, (const char *)command.payload,
                                  (int)command.payload_bytes, ids, c->max_positions);
        if (n_prompt < 1) {
            coli_serve_write_error(stdout, command.id, "EMPTY_PROMPT");
            coli_serve_command_dispose(&command); continue;
        }
        int budget = command.max_tokens > 0 ? command.max_tokens : 256;
        if (n_prompt + budget > c->max_positions) {
            char message[128];
            snprintf(message, sizeof(message), "CONTEXT_EXCEEDED %d %d",
                     n_prompt + budget, c->max_positions);
            coli_serve_write_error(stdout, command.id, message);
            coli_serve_command_dispose(&command); continue;
        }
        coli_serve_write_accept(stdout, command.id, n_prompt);
        float *aligned = NULL;
        uint8_t *image_mask = NULL;
        int image_at = -1, image_h = 0, image_w = 0;
        if (pending_image && m->vision && c->image_token_id >= 0) {
            /* the span is the run of placeholder ids the gateway inserted */
            for (int t = 0; t < n_prompt; t++)
                if (ids[t] == c->image_token_id) { image_at = t; break; }
            int ratio = c->vision_ratio;
            image_h = (pending_h + ratio - 1) / ratio;
            image_w = (pending_w + ratio - 1) / ratio;
            int span = 1 + (image_w + 1) * image_h + 1;
            int run = 0;
            for (int t = image_at; t >= 0 && t < n_prompt && ids[t] == c->image_token_id; t++) run++;
            if (image_at < 0 || run != span) {
                fprintf(stderr, "[v41] image span is %d tokens, the %dx%d grid needs %d: "
                                "answering without the image\n", run, image_h, image_w, span);
                image_at = -1;
            } else {
                int rows = vision_tokens(c, pending_h, pending_w);
                aligned = xmalloc((size_t)rows * c->dim * sizeof(float), "image rows");
                double vision_started = now_s();
                vision_forward(m, m->vision, pending_image, pending_h, pending_w, aligned);
                fprintf(stderr, "[v41] image %dx%d patches -> %d rows in %.2fs\n",
                        pending_h, pending_w, rows, now_s() - vision_started);
                image_mask = xmalloc((size_t)n_prompt, "image mask");
                memset(image_mask, 0, (size_t)n_prompt);
                for (int t = image_at; t < image_at + span; t++) image_mask[t] = 1;
            }
        }
        free(pending_image); pending_image = NULL;
        forward_with_image(m, ids, n_prompt, logits, aligned, image_at, image_h, image_w, image_mask);
        free(aligned); free(image_mask);

        int emitted = 0, limited = 1, cancelled = 0, done_early = 0;
        char piece[512];
        int block = m->spec.active ? c->spec_block : 0;
        int *batch = block ? xmalloc((size_t)(block + 1) * sizeof(int), "draft batch") : NULL;
        float *confidence = block ? xmalloc((size_t)block * sizeof(float), "draft confidence") : NULL;
        uint64_t proposed0 = m->spec.proposed, accepted0 = m->spec.accepted;
        /* the prompt seeds the draft head's windows; nothing is drafted from it */
        if (m->spec.active) spec_step(m, 0, m->last_start, m->last_rows, NULL, NULL);
        while (emitted < budget && !cancelled && !done_early) {
            int token = serve_sample(logits, c->vocab, command.temperature, command.top_p);
            int stop = 0;
            for (int i = 0; i < n_eos; i++) if (token == eos_ids[i]) stop = 1;
            if (stop) { limited = 0; break; }
            int written = tok_decode(tokenizer, &token, 1, piece, (int)sizeof(piece));
            if (written > 0) coli_serve_write_data(stdout, command.id, piece, (size_t)written);
            emitted++;
            /* #1332: look at stdin once per token so a client that walked away stops
             * paying for a turn nobody wants. A CANCEL aborts, a STOP ends it through
             * the normal DONE path -- the two are not the same frame and must not
             * share an outcome. */
            while (coli_serve_stdin_ready()) {
                ColiServeCommand control;
                ColiServeReadResult inner = coli_serve_read_command(stdin, &v41_wire, &control);
                if (inner == COLI_SERVE_READ_EOF) { done_early = 1; break; }
                if (control.kind == COLI_SERVE_COMMAND_CANCEL &&
                    !strcmp(control.id, command.id)) cancelled = 1;
                else if (control.kind == COLI_SERVE_COMMAND_STOP &&
                         !strcmp(control.id, command.id)) { limited = 0; done_early = 1; }
                else if (control.kind == COLI_SERVE_COMMAND_SUBMIT)
                    coli_serve_write_error(stdout, control.id, "SLOT_BUSY");
                coli_serve_command_dispose(&control);
            }
            if (cancelled || done_early || emitted >= budget) break;

            /* DSpark: draft the next few tokens, then verify them in ONE forward. What
             * survives verification is emitted here; what does not costs the round its
             * remaining drafts and nothing else, because the caches are rolled back to
             * the last token that agreed. */
            int drafted = spec_ready(&m->spec)
                        ? spec_step(m, token, m->last_start, m->last_rows, batch, confidence) : 0;
            if (m->spec.max_verify > 0 && drafted > m->spec.max_verify) drafted = m->spec.max_verify;
            if (drafted > budget - emitted) drafted = budget - emitted;
            if (drafted <= 0) { forward(m, &token, 1, logits); continue; }
            batch[0] = token;
            int accepted = spec_verify(m, batch, drafted + 1, logits,
                                       command.temperature, command.top_p);
            m->spec.proposed += (uint64_t)drafted;
            m->spec.accepted += (uint64_t)accepted;
            m->spec.window_prop += (uint64_t)drafted;
            m->spec.window_acc += (uint64_t)accepted;
            for (int i = 0; i < accepted; i++) {
                int drafted_token = batch[1 + i];
                int drafted_stop = 0;
                for (int k = 0; k < n_eos; k++) if (drafted_token == eos_ids[k]) drafted_stop = 1;
                if (drafted_stop) { limited = 0; done_early = 1; break; }
                written = tok_decode(tokenizer, &drafted_token, 1, piece, (int)sizeof(piece));
                if (written > 0) coli_serve_write_data(stdout, command.id, piece, (size_t)written);
                emitted++;
            }
        }
        free(batch); free(confidence);
        if (m->spec.active && m->spec.proposed > proposed0)
            fprintf(stderr, "[v41] DSpark: %llu of %llu drafts accepted this turn\n",
                    (unsigned long long)(m->spec.accepted - accepted0),
                    (unsigned long long)(m->spec.proposed - proposed0));
        if (cancelled) {
            coli_serve_write_error(stdout, command.id, "CANCELLED");
            coli_serve_command_dispose(&command);
            continue;
        }
        double wall = now_s() - turn_started;
        uint64_t turn_hits = m->hits - hits0, turn_miss = m->miss - miss0;
        ColiServeDone done = {
            emitted, wall > 0 ? emitted / wall : 0.0,
            (turn_hits + turn_miss) ? 100.0 * turn_hits / (double)(turn_hits + turn_miss) : 0.0,
            rss_gb(), n_prompt, limited,
        };
        coli_serve_write_done(stdout, command.id, &done);
        serve_line("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n", wall, n_prompt, emitted,
                   m->t_disk - disk0, 0.0, m->t_expert - expert0, m->t_attn - attn0,
                   m->t_engram - engram0, (unsigned long long)(m->forwards - forwards0));
        serve_hits(m);
        serve_emap(m);
        coli_serve_command_dispose(&command);
    }
    free(ids); free(logits);
}

/* ----------------------------------------------------------------- main ---- */

static int *load_ids(jval *root, const char *key, int *count) {
    jval *array = json_get(root, key);
    if (!array || array->t != J_ARR) { *count = 0; return NULL; }
    int *out = xmalloc((size_t)array->len * sizeof(int), "reference ids");
    for (int i = 0; i < array->len; i++) out[i] = (int)array->kids[i]->num;
    *count = array->len;
    return out;
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "SNAP=<container dir> is required\n"); return 2; }
    int cap = argc > 1 ? coli_arg_int(argv[1], "cache/layer") : 8;
    const char *ref_path = argc > 2 ? argv[2] : NULL;
    int engram_rows = getenv("V41_ENGRAM_ROWS") ? atoi(getenv("V41_ENGRAM_ROWS")) : 65536;
    g_trace = getenv("V41_TRACE") ? atoi(getenv("V41_TRACE")) : 0;

    Model m;
    memset(&m, 0, sizeof(m));
    double started = now_s();
    model_load(&m, snap, cap, engram_rows);
    Cfg *c = &m.c;
    fprintf(stderr, "[v41] %d layers, %d experts/layer (%d routed), dim %d, hc %d, "
                    "window %d, engram %s — loaded in %.2fs\n",
            c->n_layers, c->n_routed, c->n_activated, c->dim, c->hc_mult, c->window,
            m.engram.active ? "on" : "off", now_s() - started);

    if (getenv("SERVE") && atoi(getenv("SERVE"))) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/tokenizer.json", snap);
        Tok tokenizer;
        tok_load(&tokenizer, path);
        const char *seed = getenv("SEED");
        srand(seed ? (unsigned)strtoul(seed, NULL, 10) : (unsigned)time(NULL));
        serve_loop(&m, &tokenizer, snap);
        return 0;
    }
    if (!ref_path) {
        fprintf(stderr, "usage: SNAP=<dir> %s <cap> <ref.json>   (or SERVE=1 for the "
                        "serve protocol)\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(ref_path, "rb");
    if (!f) { fprintf(stderr, "%s: %s\n", ref_path, strerror(errno)); return 1; }
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = xmalloc((size_t)size + 1, "reference");
    if (fread(text, 1, (size_t)size, f) != (size_t)size) { fprintf(stderr, "short read\n"); return 1; }
    text[size] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    if (!root) { fprintf(stderr, "%s is not valid JSON\n", ref_path); return 1; }
    int n_prompt = 0, n_expected = 0;
    int *prompt = load_ids(root, "prompt_ids", &n_prompt);
    int *expected = load_ids(root, "output_ids", &n_expected);
    if (!prompt || !expected) { fprintf(stderr, "%s lacks prompt_ids/output_ids\n", ref_path); return 1; }

    /* the vision reference, when the fixture carries one: same patches in, the
     * aligner's rows out, compared as floats rather than as tokens because nothing
     * downstream of it is an argmax. */
    jval *vision_ref = json_get(root, "vision");
    int vision_failed = 0;
    if (vision_ref && vision_ref->t == J_OBJ && m.vision) {
        int n_h = (int)jnum(vision_ref, "grid_h", 0), n_w = (int)jnum(vision_ref, "grid_w", 0);
        jval *patch_values = json_get(vision_ref, "patches");
        jval *expected_values = json_get(vision_ref, "aligned");
        int patch_in = 3 * c->vision_patch * c->vision_patch;
        if (patch_values && expected_values && n_h > 0 && n_w > 0 &&
            patch_values->len == n_h * n_w * patch_in) {
            float *patches = xmalloc((size_t)patch_values->len * sizeof(float), "patches");
            for (int i = 0; i < patch_values->len; i++) patches[i] = (float)patch_values->kids[i]->num;
            int rows = vision_tokens(c, n_h, n_w);
            float *aligned = xmalloc((size_t)rows * c->dim * sizeof(float), "aligned");
            vision_forward(&m, m.vision, patches, n_h, n_w, aligned);
            double worst = 0.0;
            for (int i = 0; i < rows * c->dim && i < expected_values->len; i++) {
                double delta = fabs(aligned[i] - expected_values->kids[i]->num);
                if (delta > worst) worst = delta;
            }
            /* 2e-3: the reference rounds to five decimals in the JSON and both sides
             * accumulate a 32-layer tower in fp32; a real divergence is orders larger. */
            vision_failed = !(rows * c->dim == expected_values->len && worst < 2e-3);
            fprintf(stderr, "[v41] vision: %d x %d patches -> %d rows, max |delta| %.2e%s\n",
                    n_h, n_w, rows, worst, vision_failed ? "  MISMATCH" : "");
            free(aligned); free(patches);
        }
    }

    float *logits = xmalloc((size_t)c->vocab * sizeof(float), "logits");
    double prefill_started = now_s();
    forward(&m, prompt, n_prompt, logits);
    if (m.spec.active) spec_step(&m, 0, m.last_start, m.last_rows, NULL, NULL);
    double prefill = now_s() - prefill_started;

    /* DSpark, when the fixture describes it. Two things are checked, and they are
     * different things: that the draft head produces the vendor's block (the reference
     * records one per step), and that a block VERIFIED in a single forward emits the
     * same tokens as decoding them one at a time. The second is what makes accepting a
     * draft safe, and a tiny fixture's random draft head never proposes anything worth
     * accepting, so V41_SPEC_FORCE drafts the reference's own next tokens instead:
     * 1 accepts the whole block, 2 corrupts the last draft so the round is rejected
     * part way and the rollback runs. */
    jval *spec_ref = json_get(root, "spec");
    jval *spec_drafts = spec_ref && spec_ref->t == J_OBJ ? json_get(spec_ref, "drafts") : NULL;
    jval *spec_conf = spec_ref && spec_ref->t == J_OBJ ? json_get(spec_ref, "confidence") : NULL;
    int force = getenv("V41_SPEC_FORCE") ? atoi(getenv("V41_SPEC_FORCE")) : 0;
    int block = m.spec.active ? c->spec_block : 0;
    int *draft = block ? xmalloc((size_t)(block + 1) * sizeof(int), "drafts") : NULL;
    float *confidence = block ? xmalloc((size_t)block * sizeof(float), "draft confidence") : NULL;
    int spec_failed = 0, spec_checked = 0, round = 0;
    uint64_t forced_prop = 0, forced_acc = 0;

    int matched = 0;
    double decode_started = now_s();
    for (int step = 0; step < n_expected; ) {
        int token = argmax(logits, c->vocab);
        printf("%s%d", step ? " " : "", token);
        if (token == expected[step]) matched++;
        else fprintf(stderr, "\n[mismatch] step %d: got %d, reference %d\n", step, token, expected[step]);
        step++;
        if (step >= n_expected) break;

        int one_at_a_time = m.last_rows == 1;
        int drafted = m.spec.active
                    ? spec_step(&m, token, m.last_start, m.last_rows, draft, confidence) : 0;
        /* The reference drafts after every single-token forward. Once a round has been
         * accepted the engine is a step ahead of that loop -- it committed several
         * positions in one forward -- and its next block is a different, equally
         * correct thing. Compare only where the two loops line up. */
        if (drafted && one_at_a_time && spec_drafts && round < spec_drafts->len) {
            jval *want = spec_drafts->kids[round];
            jval *want_conf = spec_conf && round < spec_conf->len ? spec_conf->kids[round] : NULL;
            for (int i = 0; i <= drafted && i < want->len; i++) {
                int reference_id = (int)want->kids[i]->num;
                if (draft[i] == reference_id) continue;
                fprintf(stderr, "[dspark] round %d, draft %d: got %d, reference %d\n",
                        round, i, draft[i], reference_id);
                spec_failed = 1;
            }
            for (int i = 0; want_conf && i < drafted && i < want_conf->len; i++) {
                double delta = fabs(confidence[i] - want_conf->kids[i]->num);
                if (delta < 2e-3) continue;
                fprintf(stderr, "[dspark] round %d, confidence %d: got %.5f, reference %.5f\n",
                        round, i, confidence[i], want_conf->kids[i]->num);
                spec_failed = 1;
            }
            spec_checked++;
        }
        if (drafted > 0 && one_at_a_time) round++;   /* the prefill call drafts nothing */

        if (force && drafted > 0) {
            if (m.spec.max_verify > 0 && drafted > m.spec.max_verify) drafted = m.spec.max_verify;
            int room = n_expected - step;
            if (drafted > room) drafted = room;
            /* 3 keeps the head's own drafts, which is what serving does; 1 and 2 put
             * the reference's tokens in their place so the verification path runs at
             * full width even on a fixture whose draft head is random noise */
            if (force < 3) {
                for (int i = 0; i < drafted; i++) draft[1 + i] = expected[step + i];
                if (force >= 2 && drafted > 0)
                    draft[drafted] = (draft[drafted] + 1) % c->vocab;   /* one bad draft */
            }
        } else {
            drafted = 0;                       /* the plain oracle decodes one at a time */
        }
        if (drafted > 0) {
            draft[0] = token;
            int accepted = spec_verify(&m, draft, drafted + 1, logits, 0.0f, 1.0f);
            forced_prop += (uint64_t)drafted;
            forced_acc += (uint64_t)accepted;
            for (int i = 0; i < accepted && step < n_expected; i++) {
                int drafted_token = draft[1 + i];
                printf(" %d", drafted_token);
                if (drafted_token == expected[step]) matched++;
                else fprintf(stderr, "\n[mismatch] step %d: accepted draft %d, reference %d\n",
                             step, drafted_token, expected[step]);
                step++;
            }
        } else {
            forward(&m, &token, 1, logits);
        }
    }
    double decode = now_s() - decode_started;
    printf("\n");
    fprintf(stderr, "Matching tokens: %d/%d\n", matched, n_expected);
    fprintf(stderr, "[v41] prefill %d tokens in %.2fs | decode %d in %.2fs (%.2f tok/s)\n",
            n_prompt, prefill, n_expected, decode, n_expected / (decode > 0 ? decode : 1));
    fprintf(stderr, "[v41] experts: %llu hits, %llu misses, %.1f MB read | disk %.2fs, "
                    "expert matmul %.2fs, attention %.2fs, engram %.2fs\n",
            (unsigned long long)m.hits, (unsigned long long)m.miss,
            m.expert_bytes / 1e6, m.t_disk, m.t_expert, m.t_attn, m.t_engram);
    if (m.engram.active)
        for (int t = 0; t < m.engram.n_layers; t++)
            fprintf(stderr, "[v41] engram table %d (layer %d): %lld rows, %llu cache hits, "
                            "%llu reads\n", t, m.engram.layer_of[t],
                    (long long)m.engram.table[t].rows,
                    (unsigned long long)m.engram.table[t].hits,
                    (unsigned long long)m.engram.table[t].misses);
    if (spec_checked)
        fprintf(stderr, "[v41] DSpark: checked %d draft round%s against the reference%s\n",
                spec_checked, spec_checked == 1 ? "" : "s", spec_failed ? "  MISMATCH" : "");
    if (forced_prop)
        fprintf(stderr, "[v41] DSpark: %llu of %llu forced drafts accepted, %d forwards "
                        "for %d tokens\n", (unsigned long long)forced_acc,
                (unsigned long long)forced_prop, (int)m.forwards, n_expected);
    if (force && !forced_prop) {
        fprintf(stderr, "[v41] V41_SPEC_FORCE is set but nothing was drafted: the "
                        "verification path was NOT exercised\n");
        spec_failed = 1;
    }
    free(confidence); free(draft);
    free(logits); free(prompt); free(expected); json_free(root); free(arena); free(text);
    return (matched == n_expected && !vision_failed && !spec_failed) ? 0 : 1;
}
