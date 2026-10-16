/* kolibri1.c -- Aleph Alpha Kolibri-1 (50 layers, 384 experts top-6, FP8 block-128
 * experts, ~80B) on a machine that cannot hold it.
 *
 * A CPU engine: the checkpoint loader and its validation, the engine's own numeric
 * kernels, the RoPE tables, the router, the tokenizer hookup and the forward pass
 * (ring-buffer sliding-window attention, NoPE full layers, MoE with an expert LRU, fp32
 * head), with the test hooks of kolibri1_capture.h (K1_CAPTURE, K1_LOGITS_FIFO, K1_FEED,
 * K1_ATTN_KEYS), K1_PROBE / K1_LOGITS, K1_DECODE_TAIL, K1_LAYERS, and several requests
 * per process with cache reset and prefix snapshots. Every logit the head produces is
 * checked finite. SERVE=1 is the gateway's stdin/stdout protocol with a per-request
 * cache reset, prefix reuse by snapshot, sampling and the finite check as an error
 * reply; it calls the same k1_forward as every other mode.
 *
 * The released checkpoint is read as it is:
 *
 *   attention  separate q/k/v/o, F8_E4M3 with a F32 weight_scale_inv grid
 *              [ceil(O/128), ceil(I/128)]; q is heads*head_dim wide, not hidden
 *   experts    per-expert gate/up/down, F8_E4M3 + F32 grids; 40% of them are NOT
 *              contiguous (a split across two shards), and an expert's scales may
 *              sit in two shards too, so the placement table (K1ERef) is built
 *              from the headers, never from an assumed layout
 *   shared     one ungated shared expert, same FP8 form
 *   router     bf16 gate, bf16 expert_bias: sigmoid_logit_add (select on z+bias,
 *              weigh by sigmoid(z), no renormalisation)
 *   embed/head bf16, resident
 *
 * Numeric stance: fp32 activations, an FP64 residual stream (see k1_layer), fp32 router,
 * fp32 head operands. FP8 weights stay e4m3 bytes with the checkpoint's grid, never
 * expanded and never rounded; a block is summed in fp32 and the blocks are added in
 * double after the block scale (same contract as quant.h).
 *
 * Nothing in the shared headers is edited: the AVX2 e4m3 kernel is duplicated from
 * qwen38_core.h (q38_matmul_fp8_vec) so this file stands alone, and the tokenizer
 * needs a guarded engine-local override (k1_tok_load) because tok.h detects the
 * one-digit rule only as `|\p{N}|` while this release spells it `\p{N}{1}`.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#ifdef __APPLE__
#include <mach-o/dyld.h>      /* _NSGetExecutablePath: the running executable's own path (no /proc on macOS) */
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
#include "tok.h"
#include "stop_ids.h"
#include "serve_codec.h"
#include "serve_poll.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

#define K1_MAX_LAYERS 256
#define K1_MAX_TOPK 32            /* the expert tier's per-row ceiling (qwen36_tier.c) */
#define K1_MAX_HEAD 1024
#define K1_SCALE_GAP (64 * 1024)  /* merge scale reads whose gap is below this */

/* ------------------------------------------------------------------ utils ---- */

static double k1_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double k1_rss_gb(void) {
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return usage.ru_maxrss / 1e9;
#else
    return usage.ru_maxrss / 1e6;
#endif
#else
    return 0;
#endif
}

static void *k1_malloc(size_t bytes, const char *what) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "[kolibri1] out of memory allocating %s (%zu bytes)\n", what, bytes); exit(1); }
    return p;
}

static void *k1_calloc(size_t n, size_t size, const char *what) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) { fprintf(stderr, "[kolibri1] out of memory allocating %s\n", what); exit(1); }
    return p;
}

#if defined(__GNUC__)
static void k1_die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
#endif
static void k1_die(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "[kolibri1] ");
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, " -- refusing\n");
    exit(1);
}

#include "kolibri1_capture.h"

/* ---------------------------------------------------------------- config ---- */

typedef struct {
    int vocab, hidden, n_layers, heads, kv_heads, head_dim;
    int n_experts, topk, moe_inter, shared_inter;
    int window, max_positions, eos;
    float eps;
    double theta;
    unsigned char full[K1_MAX_LAYERS];   /* 1 = full attention (NoPE), 0 = sliding window (RoPE) */
} K1Cfg;

static double k1_jnum(jval *o, const char *key, double fallback) {
    jval *v = o ? json_get(o, key) : NULL;
    return (v && v->t == J_NUM) ? v->num : fallback;
}

static char *k1_read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    if (size < 0 || size > (256l << 20)) { fclose(f); return NULL; }
    char *text = k1_malloc((size_t)size + 1, path);
    if (fread(text, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(text); return NULL; }
    text[size] = 0; fclose(f);
    return text;
}

/* An integer config value: present, numeric, integral and inside [lo, hi]. */
static int k1_cfg_int(jval *root, const char *key, double lo, double hi) {
    jval *v = json_get(root, key);
    if (!v || v->t != J_NUM) k1_die("config.json: '%s' is missing or not a number", key);
    if (v->num != floor(v->num) || v->num < lo || v->num > hi)
        k1_die("config.json: '%s' = %g is outside [%g, %g]", key, v->num, lo, hi);
    return (int)v->num;
}

/* A boolean that must have exactly `want`. A missing key is a refusal too: the
 * release states every one of these, and a default would hide a wrong checkpoint. */
static void k1_cfg_bool_is(jval *root, const char *key, int want) {
    jval *v = json_get(root, key);
    if (!v || v->t != J_BOOL) k1_die("config.json: '%s' is missing or not a boolean", key);
    if ((v->boolean != 0) != want) k1_die("config.json: '%s' must be %s", key, want ? "true" : "false");
}

/* The release's own layer-type rule: a layer is full attention iff l % 5 == 4.
 * A fact about the checkpoint, asserted only under --release. */
static int k1_release_full(int layer) { return layer % 5 == 4; }

static void k1_load_cfg(K1Cfg *c, const char *dir, int release) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char *text = k1_read_text(path);
    if (!text) { fprintf(stderr, "[kolibri1] cannot read %s\n", path); exit(1); }
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    if (!root || root->t != J_OBJ) k1_die("config.json: not a JSON object");
    jval *mt = json_get(root, "model_type");
    if (!mt || mt->t != J_STR || strcmp(mt->str, "kolibri1"))
        k1_die("config.json: model_type is not kolibri1");
    memset(c, 0, sizeof(*c));
    c->vocab = k1_cfg_int(root, "vocab_size", 2, 1 << 22);
    c->hidden = k1_cfg_int(root, "hidden_size", 1, 1 << 20);
    c->n_layers = k1_cfg_int(root, "num_hidden_layers", 1, K1_MAX_LAYERS);
    c->heads = k1_cfg_int(root, "num_attention_heads", 1, 4096);
    c->kv_heads = k1_cfg_int(root, "num_key_value_heads", 1, 4096);
    c->head_dim = k1_cfg_int(root, "head_dim", 2, K1_MAX_HEAD);
    c->n_experts = k1_cfg_int(root, "num_experts", 1, 4096);
    c->topk = k1_cfg_int(root, "num_experts_per_tok", 1, K1_MAX_TOPK);
    c->moe_inter = k1_cfg_int(root, "moe_intermediate_size", 1, 1 << 20);
    c->shared_inter = k1_cfg_int(root, "shared_expert_intermediate_size", 1, 1 << 20);
    c->max_positions = k1_cfg_int(root, "max_position_embeddings", 1, 1 << 30);
    if (c->heads % c->kv_heads) k1_die("config.json: num_attention_heads must be a multiple of num_key_value_heads");
    if (c->head_dim % 2) k1_die("config.json: head_dim must be even (rotate-half RoPE)");
    if (c->topk > c->n_experts) k1_die("config.json: num_experts_per_tok exceeds num_experts");
    c->eps = (float)k1_jnum(root, "rms_norm_eps", 0);
    if (!(c->eps > 0) || !(c->eps < 1)) k1_die("config.json: rms_norm_eps out of range");
    c->theta = k1_jnum(root, "rope_theta", 0);
    if (!(c->theta > 1) || !(c->theta < 1e12)) k1_die("config.json: rope_theta out of range");
    c->eos = (int)k1_jnum(root, "eos_token_id", -1);

    k1_cfg_bool_is(root, "use_sliding_window", 1);
    c->window = k1_cfg_int(root, "sliding_window", 1, 1 << 24);
    k1_cfg_bool_is(root, "norm_topk_prob", 0);          /* the weights are sigmoid(z), never renormalised */
    k1_cfg_bool_is(root, "tie_word_embeddings", 0);     /* lm_head is its own tensor */
    k1_cfg_bool_is(root, "attention_bias", 0);
    jval *act = json_get(root, "hidden_act");
    if (!act || act->t != J_STR || strcmp(act->str, "silu")) k1_die("config.json: hidden_act is not silu");
    /* anything that would change the maths and that this engine does not implement */
    jval *rs = json_get(root, "rope_scaling");
    if (rs && rs->t != J_NULL) k1_die("config.json: rope_scaling is not supported (default RoPE only)");
    jval *rp = json_get(root, "rope_parameters");
    if (rp && rp->t != J_NULL) {
        jval *rt = json_get(rp, "rope_type");
        if (rp->t != J_OBJ || (rt && (rt->t != J_STR || strcmp(rt->str, "default"))))
            k1_die("config.json: rope_parameters is not the default rope");
    }
    jval *mlo = json_get(root, "mlp_only_layers");
    if (mlo && mlo->t == J_ARR && mlo->len > 0) k1_die("config.json: mlp_only_layers is not supported");
    jval *dss = json_get(root, "decoder_sparse_step");
    if (dss && dss->t == J_NUM && dss->num != 1) k1_die("config.json: decoder_sparse_step must be 1");
    jval *qc = json_get(root, "quantization_config");
    if (!qc || qc->t != J_OBJ) k1_die("config.json: quantization_config is missing");
    jval *qm = json_get(qc, "quant_method");
    if (!qm || qm->t != J_STR || strcmp(qm->str, "fp8")) k1_die("config.json: quantization_config.quant_method is not fp8");
    jval *wb = json_get(qc, "weight_block_size");
    if (!wb || wb->t != J_ARR || wb->len != 2 || wb->kids[0]->t != J_NUM || wb->kids[1]->t != J_NUM ||
        wb->kids[0]->num != FP8_BLOCK || wb->kids[1]->num != FP8_BLOCK)
        k1_die("config.json: quantization_config.weight_block_size is not [128, 128]");

    jval *lt = json_get(root, "layer_types");
    if (!lt || lt->t != J_ARR || lt->len != c->n_layers)
        k1_die("config.json: layer_types must list every one of the %d layers", c->n_layers);
    for (int i = 0; i < c->n_layers; i++) {
        jval *e = lt->kids[i];
        if (!e || e->t != J_STR) k1_die("config.json: layer_types[%d] is not a string", i);
        if (!strcmp(e->str, "full_attention")) c->full[i] = 1;
        else if (!strcmp(e->str, "sliding_attention")) c->full[i] = 0;
        else k1_die("config.json: layer_types[%d] = '%s' is neither sliding_attention nor full_attention", i, e->str);
        if (release && c->full[i] != k1_release_full(i))
            k1_die("config.json: layer_types[%d] = '%s' breaks the release pattern (full attention iff layer %% 5 == 4)",
                   i, e->str);
    }
    json_free(root); free(arena); free(text);
}

/* ------------------------------------------------- the expected tensor set ---- */

/* The checkpoint's tensors are generated from the config, and the file's tensor
 * count must equal the generated count: a missing tensor, an extra tensor (MTP,
 * vision, an overlay) and a wrong dtype or shape are all caught by name. */
enum { K1_BF16 = 0, K1_F32 = 2, K1_F8 = 4 };   /* st.h dtype codes */
enum { K1X_PLAIN = 0, K1X_FP8W = 1, K1X_SCALE = 2 };

typedef struct {
    char name[96];
    uint8_t dtype, rank, kind;
    int64_t d0, d1;
} K1Exp;

typedef struct { K1Exp *e; int n, cap; } K1Manifest;

static void k1_man_add(K1Manifest *m, const char *fmt, int dtype, int rank, int kind,
                       int64_t d0, int64_t d1, ...) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 4096;
        m->e = realloc(m->e, (size_t)m->cap * sizeof(K1Exp));
        if (!m->e) k1_die("out of memory growing the tensor manifest");
    }
    K1Exp *x = &m->e[m->n++];
    va_list ap; va_start(ap, d1);
    int n = vsnprintf(x->name, sizeof(x->name), fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof(x->name)) k1_die("tensor name too long for the manifest: %s", fmt);
    x->dtype = (uint8_t)dtype; x->rank = (uint8_t)rank; x->kind = (uint8_t)kind; x->d0 = d0; x->d1 = d1;
}

/* an FP8 projection: the e4m3 weight [O,I] and its F32 grid [ceil(O/128), ceil(I/128)] */
static void k1_man_fp8(K1Manifest *m, int64_t O, int64_t I, const char *fmt, ...) {
    char base[96];
    va_list ap; va_start(ap, fmt); vsnprintf(base, sizeof(base), fmt, ap); va_end(ap);
    k1_man_add(m, "%s.weight", K1_F8, 2, K1X_FP8W, O, I, base);
    k1_man_add(m, "%s.weight_scale_inv", K1_F32, 2, K1X_SCALE,
               fp8_nblk((int)O), fp8_nblk((int)I), base);
}

static void k1_manifest(const K1Cfg *c, K1Manifest *m) {
    memset(m, 0, sizeof(*m));
    int64_t H = c->hidden, qw = (int64_t)c->heads * c->head_dim, kw = (int64_t)c->kv_heads * c->head_dim;
    k1_man_add(m, "model.embed_tokens.weight", K1_BF16, 2, K1X_PLAIN, c->vocab, H);
    k1_man_add(m, "lm_head.weight", K1_BF16, 2, K1X_PLAIN, c->vocab, H);
    k1_man_add(m, "model.norm.weight", K1_BF16, 1, K1X_PLAIN, H, 0);
    for (int l = 0; l < c->n_layers; l++) {
        k1_man_add(m, "model.layers.%d.input_layernorm.weight", K1_BF16, 1, K1X_PLAIN, H, 0, l);
        k1_man_add(m, "model.layers.%d.post_attn_norm.weight", K1_BF16, 1, K1X_PLAIN, H, 0, l);
        k1_man_add(m, "model.layers.%d.post_attention_layernorm.weight", K1_BF16, 1, K1X_PLAIN, H, 0, l);
        k1_man_add(m, "model.layers.%d.post_ffn_norm.weight", K1_BF16, 1, K1X_PLAIN, H, 0, l);
        k1_man_add(m, "model.layers.%d.self_attn.q_norm.weight", K1_BF16, 1, K1X_PLAIN, c->head_dim, 0, l);
        k1_man_add(m, "model.layers.%d.self_attn.k_norm.weight", K1_BF16, 1, K1X_PLAIN, c->head_dim, 0, l);
        k1_man_fp8(m, qw, H, "model.layers.%d.self_attn.q_proj", l);
        k1_man_fp8(m, kw, H, "model.layers.%d.self_attn.k_proj", l);
        k1_man_fp8(m, kw, H, "model.layers.%d.self_attn.v_proj", l);
        k1_man_fp8(m, H, qw, "model.layers.%d.self_attn.o_proj", l);
        k1_man_add(m, "model.layers.%d.mlp.gate.weight", K1_BF16, 2, K1X_PLAIN, c->n_experts, H, l);
        k1_man_add(m, "model.layers.%d.moe.router.expert_bias", K1_BF16, 1, K1X_PLAIN, c->n_experts, 0, l);
        k1_man_fp8(m, c->shared_inter, H, "model.layers.%d.mlp.shared_experts.gate_proj", l);
        k1_man_fp8(m, c->shared_inter, H, "model.layers.%d.mlp.shared_experts.up_proj", l);
        k1_man_fp8(m, H, c->shared_inter, "model.layers.%d.mlp.shared_experts.down_proj", l);
        for (int e = 0; e < c->n_experts; e++) {
            k1_man_fp8(m, H, c->moe_inter, "model.layers.%d.mlp.experts.%d.down_proj", l, e);
            k1_man_fp8(m, c->moe_inter, H, "model.layers.%d.mlp.experts.%d.gate_proj", l, e);
            k1_man_fp8(m, c->moe_inter, H, "model.layers.%d.mlp.experts.%d.up_proj", l, e);
        }
    }
}

/* ---------------------------------------- header scan (independent of st.h) ---- */

/* st.h resolves a duplicated tensor name through model.safetensors.index.json
 * (an "overlay" shard) and does not tell the caller. This scan reads the shard
 * headers itself, so the duplicate / overlay / index policy is the engine's:
 *   generic  a duplicate is refused unless the index names one of its two holders
 *            (st.h's own rule, reported as an overlay); an index, when present,
 *            must be consistent with the shards
 *   release  no duplicate at all, and the index is required and consistent */
typedef struct { char *name; int shard, shard2; } K1ScanEnt;   /* shard2: the other holder of an overlaid name, else -1 */
typedef struct {
    int nshard;
    char **shard;                       /* shard file basenames, sorted */
    K1ScanEnt *e; int n, cap;
    int *h; int hcap;
    int n_dup, n_overlay;
    int has_index;
} K1Scan;

static int k1_cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int k1_scan_find(K1Scan *s, const char *name) {
    uint64_t h = st_hash(name) & (s->hcap - 1);
    while (s->h[h] >= 0) {
        if (!strcmp(s->e[s->h[h]].name, name)) return s->h[h];
        h = (h + 1) & (s->hcap - 1);
    }
    return -1;
}

static void k1_scan_put(K1Scan *s, char *name, int shard) {
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 4096;
        s->e = realloc(s->e, (size_t)s->cap * sizeof(K1ScanEnt));
        if (!s->e) k1_die("out of memory scanning shard headers");
    }
    if (s->n * 2 >= s->hcap) {                        /* grow + rehash */
        int nc = s->hcap ? s->hcap * 2 : 8192;
        int *nh = k1_malloc((size_t)nc * sizeof(int), "scan hash");
        for (int i = 0; i < nc; i++) nh[i] = -1;
        for (int i = 0; i < s->n; i++) {
            uint64_t h = st_hash(s->e[i].name) & (nc - 1);
            while (nh[h] >= 0) h = (h + 1) & (nc - 1);
            nh[h] = i;
        }
        free(s->h); s->h = nh; s->hcap = nc;
    }
    s->e[s->n].name = name; s->e[s->n].shard = shard; s->e[s->n].shard2 = -1;
    uint64_t h = st_hash(name) & (s->hcap - 1);
    while (s->h[h] >= 0) h = (h + 1) & (s->hcap - 1);
    s->h[h] = s->n++;
}

static void k1_scan_free(K1Scan *s) {
    for (int i = 0; i < s->n; i++) free(s->e[i].name);
    for (int i = 0; i < s->nshard; i++) free(s->shard[i]);
    free(s->e); free(s->h); free(s->shard);
    memset(s, 0, sizeof(*s));
}

/* Read the index's weight_map into `map` (borrowed from the returned root); NULL if absent. */
static jval *k1_load_index(const char *dir, char **arena, jval **map_out) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    char *text = k1_read_text(path);
    if (!text) return NULL;
    jval *root = json_parse(text, arena);
    free(text);
    jval *map = root ? json_get(root, "weight_map") : NULL;
    if (!map || map->t != J_OBJ) k1_die("model.safetensors.index.json has no weight_map object");
    *map_out = map;
    return root;
}

static const char *k1_index_shard(jval *map, const char *name) {
    for (int i = 0; i < map->len; i++)
        if (!strcmp(map->keys[i], name))
            return map->kids[i]->t == J_STR ? map->kids[i]->str : NULL;
    return NULL;
}

/* index lookup by hash for the big maps: built once from the weight_map keys */
typedef struct { jval *map; int *h; int hcap; } K1IdxHash;
static void k1_idx_build(K1IdxHash *x, jval *map) {
    x->map = map;
    x->hcap = 1; while (x->hcap < map->len * 2 + 2) x->hcap <<= 1;
    x->h = k1_malloc((size_t)x->hcap * sizeof(int), "index hash");
    for (int i = 0; i < x->hcap; i++) x->h[i] = -1;
    for (int i = 0; i < map->len; i++) {
        uint64_t h = st_hash(map->keys[i]) & (x->hcap - 1);
        while (x->h[h] >= 0) h = (h + 1) & (x->hcap - 1);
        x->h[h] = i;
    }
}
static int k1_idx_find(const K1IdxHash *x, const char *name) {
    uint64_t h = st_hash(name) & (x->hcap - 1);
    while (x->h[h] >= 0) {
        if (!strcmp(x->map->keys[x->h[h]], name)) return x->h[h];
        h = (h + 1) & (x->hcap - 1);
    }
    return -1;
}

static void k1_scan_headers(K1Scan *s, const char *dir, int release) {
    memset(s, 0, sizeof(*s));
    DIR *d = opendir(dir);
    if (!d) { perror(dir); exit(1); }
    struct dirent *de;
    int cap = 64;
    s->shard = k1_malloc((size_t)cap * sizeof(char *), "shard list");
    while ((de = readdir(d))) {
        const char *dot = strrchr(de->d_name, '.');
        if (!dot || strcmp(dot, ".safetensors")) continue;
        if (s->nshard == cap) { cap *= 2; s->shard = realloc(s->shard, (size_t)cap * sizeof(char *)); }
        s->shard[s->nshard++] = strdup(de->d_name);
    }
    closedir(d);
    if (s->nshard == 0) k1_die("%s holds no *.safetensors shard", dir);
    qsort(s->shard, (size_t)s->nshard, sizeof(char *), k1_cmp_str);

    char *iarena = NULL; jval *imap = NULL;
    jval *iroot = k1_load_index(dir, &iarena, &imap);
    s->has_index = iroot != NULL;
    if (release && !s->has_index) k1_die("--release requires model.safetensors.index.json (none found)");

    for (int si = 0; si < s->nshard; si++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, s->shard[si]);
        FILE *f = fopen(path, "rb");
        if (!f) { perror(path); exit(1); }
        fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
        uint64_t hlen = 0;
        if (fsz < 8 || fread(&hlen, 8, 1, f) != 1 || hlen > (uint64_t)(fsz - 8) || hlen > (uint64_t)ST_MAX_HEADER)
            k1_die("%s: bad safetensors header length", s->shard[si]);
        char *hdr = k1_malloc((size_t)hlen + 1, "shard header");
        if (fread(hdr, 1, (size_t)hlen, f) != (size_t)hlen) k1_die("%s: short header read", s->shard[si]);
        fclose(f);
        hdr[hlen] = 0;
        char *arena = NULL;
        jval *root = json_parse(hdr, &arena);
        if (!root || root->t != J_OBJ) k1_die("%s: header is not a JSON object", s->shard[si]);
        for (int i = 0; i < root->len; i++) {
            const char *name = root->keys[i];
            if (!strcmp(name, "__metadata__")) continue;
            int prev = s->hcap ? k1_scan_find(s, name) : -1;
            if (prev >= 0) {
                s->n_dup++;
                const char *auth = imap ? k1_index_shard(imap, name) : NULL;
                int overlay_ok = !release && auth &&
                    (!strcmp(auth, s->shard[si]) || !strcmp(auth, s->shard[s->e[prev].shard]));
                if (!overlay_ok)
                    k1_die("duplicate tensor name '%s' in shards %s and %s%s", name,
                           s->shard[s->e[prev].shard], s->shard[si],
                           release ? (auth ? " (an overlay chosen by the index is not allowed under --release)"
                                           : " (--release allows no duplicate at all)")
                                   : (imap ? " (and model.safetensors.index.json does not name either copy)"
                                           : " (no model.safetensors.index.json to say which copy wins)"));
                s->n_overlay++;
                s->e[prev].shard2 = si;
                continue;       /* the first copy stays; the index decides in st.h */
            }
            k1_scan_put(s, strdup(name), si);
        }
        json_free(root); free(arena); free(hdr);
    }
    /* the index, when present, must say where every tensor is (and nothing else) */
    if (iroot) {
        K1IdxHash ix; k1_idx_build(&ix, imap);
        for (int i = 0; i < imap->len; i++) {
            const char *name = imap->keys[i];
            const char *sh = imap->kids[i]->t == J_STR ? imap->kids[i]->str : NULL;
            if (!sh) k1_die("model.safetensors.index.json: '%s' does not name a shard", name);
            int idx = k1_scan_find(s, name);
            if (idx < 0) k1_die("model.safetensors.index.json lists '%s', which no shard holds", name);
            int named = -1;
            for (int j = 0; j < s->nshard; j++) if (!strcmp(s->shard[j], sh)) { named = j; break; }
            if (named < 0) k1_die("model.safetensors.index.json maps '%s' to %s, which is not in this directory", name, sh);
            if (s->e[idx].shard != named && s->e[idx].shard2 != named)
                k1_die("model.safetensors.index.json maps '%s' to %s but its header is in %s",
                       name, sh, s->shard[s->e[idx].shard]);
        }
        for (int i = 0; i < s->n; i++)
            if (k1_idx_find(&ix, s->e[i].name) < 0)
                k1_die("tensor '%s' (in %s) has no entry in model.safetensors.index.json", s->e[i].name,
                       s->shard[s->e[i].shard]);
        free(ix.h);
        json_free(iroot); free(iarena);
    }
}

/* ------------------------------------------------------- the loaded model ---- */

/* Where one routed expert's three weight tensors are. Index order is the file
 * order of the release (down, gate, up). Experts are NOT necessarily contiguous:
 * 40% of the release has one projection in the neighbouring shard. */
enum { K1_DOWN = 0, K1_GATE = 1, K1_UP = 2 };
typedef struct {
    int fd[3];
    int shard[3];            /* index into the sorted shard list */
    int64_t off[3], len[3];
} K1ERef;

/* The scale bank: float [layer][expert][3][nsc], per-matrix order gate, up, down
 * (the order qwen36_tier's qt_note takes). nsc = ceil(I/128) * ceil(H/128), the
 * same for the gate/up grid [ceil(I/128), ceil(H/128)] and the down grid. */
enum { K1_B_GATE = 0, K1_B_UP = 1, K1_B_DOWN = 2 };

typedef struct {
    K1Cfg c;
    shards S;
    K1ERef *er;              /* [layer][expert] */
    float *bank;             /* [layer][expert][3][nsc] */
    int nsc;
    /* counts for --validate */
    int n_tensors, n_shards;
    long runs1, runs2, runs3, split, gate_prev, up_next, split_other;
    long n_scale_tensors;
    double scale_min, scale_max;      /* of |scale|, nonzero */
    long scale_zero, scale_negative;
} K1Model;

static inline float *k1_bank(const K1Model *m, int layer, int expert, int which) {
    return m->bank + (((size_t)layer * m->c.n_experts + expert) * 3 + which) * (size_t)m->nsc;
}

static const char *k1_dtype_name(int d) { return st_dtype_name(d); }

static const char *k1_shape_str(const st_tensor *t) {
    static char buf[160];
    int n = snprintf(buf, sizeof(buf), "[");
    for (int i = 0; i < t->rank && n < (int)sizeof(buf) - 24; i++)
        n += snprintf(buf + n, sizeof(buf) - n, i ? ",%lld" : "%lld", (long long)t->shape[i]);
    snprintf(buf + n, sizeof(buf) - n, "]");
    return buf;
}

static void k1_check_tensor(shards *S, const K1Exp *x, int *seen, const char *grid_hint) {
    st_tensor *t = st_find(S, x->name);
    if (!t) st_die_missing(S, x->name);
    seen[t - S->t]++;
    if (t->dtype != x->dtype) {
        if (x->kind == K1X_SCALE)
            k1_die("%s: scale dtype %s, expected F32 (a BF16/F16 scale is not this release)", x->name, k1_dtype_name(t->dtype));
        k1_die("%s: dtype %s, expected %s", x->name, k1_dtype_name(t->dtype), k1_dtype_name(x->dtype));
    }
    int ok = t->rank == x->rank && t->shape[0] == x->d0 && (x->rank < 2 || t->shape[1] == x->d1);
    if (!ok) {
        if (x->kind == K1X_SCALE && t->rank == 2 && t->shape[0] == x->d1 && t->shape[1] == x->d0)
            k1_die("%s: scale grid [%lld,%lld] is TRANSPOSED, expected [%lld,%lld] = [ceil(O/128), ceil(I/128)]",
                   x->name, (long long)t->shape[0], (long long)t->shape[1], (long long)x->d0, (long long)x->d1);
        if (x->rank == 2)
            k1_die("%s: shape %s, expected [%lld,%lld]%s", x->name, k1_shape_str(t), (long long)x->d0, (long long)x->d1, grid_hint);
        k1_die("%s: shape %s, expected [%lld]", x->name, k1_shape_str(t), (long long)x->d0);
    }
    int64_t numel = x->rank == 2 ? x->d0 * x->d1 : x->d0;
    if (t->nbytes != numel * st_dtype_esz(x->dtype))
        k1_die("%s: %lld bytes, expected %lld", x->name, (long long)t->nbytes, (long long)(numel * st_dtype_esz(x->dtype)));
}

typedef struct { int fd; int64_t off, len; float *dst; int is_expert; } K1ScaleRef;
static int k1_cmp_scale(const void *a, const void *b) {
    const K1ScaleRef *x = a, *y = b;
    if (x->fd != y->fd) return x->fd < y->fd ? -1 : 1;
    return x->off < y->off ? -1 : x->off > y->off;
}

/* `validate`: scan + manifest + placement + scales. Dies by name on every refusal. */
static void k1_open(K1Model *m, const char *dir, int release) {
    memset(m, 0, sizeof(*m));
    k1_load_cfg(&m->c, dir, release);
    const K1Cfg *c = &m->c;

    K1Scan scan;
    k1_scan_headers(&scan, dir, release);
    m->n_shards = scan.nshard;
    if (release && scan.n_dup) k1_die("internal: --release kept a duplicate");   /* unreachable: refused in the scan */
    st_init(&m->S, dir);
    shards *S = &m->S;
    if (S->n != scan.n) k1_die("tensor count disagrees: st.h indexed %d, the header scan found %d", S->n, scan.n);
    m->n_tensors = S->n;
    if (scan.n_overlay)
        fprintf(stderr, "[kolibri1] %d tensor name(s) present in two shards, resolved by model.safetensors.index.json "
                        "(allowed without --release)\n", scan.n_overlay);

    /* exact manifest */
    K1Manifest man;
    k1_manifest(c, &man);
    int *seen = k1_calloc((size_t)S->n, sizeof(int), "seen");
    for (int i = 0; i < man.n; i++) k1_check_tensor(S, &man.e[i], seen, "");
    for (int i = 0; i < S->n; i++)
        if (!seen[i]) k1_die("unexpected tensor '%s' -- not in the %d-tensor manifest of this config "
                             "(no MTP, vision or leftover tensors are accepted)", S->t[i].name, man.n);
    if (S->n != man.n) k1_die("%d tensors in the checkpoint, the config implies %d", S->n, man.n);

    /* expert placement table + classes */
    size_t ne = (size_t)c->n_layers * c->n_experts;
    m->er = k1_calloc(ne, sizeof(K1ERef), "expert table");
    static const char *proj[3] = { "down_proj", "gate_proj", "up_proj" };
    for (int l = 0; l < c->n_layers; l++)
        for (int e = 0; e < c->n_experts; e++) {
            K1ERef *r = &m->er[(size_t)l * c->n_experts + e];
            for (int p = 0; p < 3; p++) {
                char nm[128];
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.%s.weight", l, e, proj[p]);
                st_tensor *t = st_find(S, nm);       /* present: the manifest check passed */
                r->fd[p] = t->fd; r->off[p] = t->off; r->len[p] = t->nbytes;
                r->shard[p] = st_fidx(S, t->fd);
            }
            /* runs = adjacent (same shard, end == next start) pieces after sorting by (shard, offset) */
            int ord[3] = { 0, 1, 2 };
            for (int a = 0; a < 3; a++) for (int b = a + 1; b < 3; b++) {
                int x = ord[a], y = ord[b];
                if (r->shard[y] < r->shard[x] || (r->shard[y] == r->shard[x] && r->off[y] < r->off[x])) { ord[a] = y; ord[b] = x; }
            }
            int runs = 1;
            for (int a = 1; a < 3; a++) {
                int p0 = ord[a - 1], p1 = ord[a];
                if (!(r->shard[p0] == r->shard[p1] && r->off[p0] + r->len[p0] == r->off[p1])) runs++;
            }
            if (runs == 1) m->runs1++; else if (runs == 2) m->runs2++; else m->runs3++;
            int dist = 1 + (r->shard[1] != r->shard[0]) + (r->shard[2] != r->shard[0] && r->shard[2] != r->shard[1]);
            if (dist >= 2) {
                m->split++;
                if (dist == 2 && r->shard[K1_DOWN] == r->shard[K1_UP] && r->shard[K1_GATE] < r->shard[K1_DOWN]) m->gate_prev++;
                else if (dist == 2 && r->shard[K1_DOWN] == r->shard[K1_GATE] && r->shard[K1_UP] > r->shard[K1_DOWN]) m->up_next++;
                else m->split_other++;
            }
        }

    /* scales: every grid read, every value finite; the experts' into the bank */
    m->nsc = (int)(fp8_nblk(c->moe_inter) * fp8_nblk(c->hidden));
    m->bank = k1_calloc(ne * 3 * (size_t)m->nsc, sizeof(float), "scale bank");
    K1ScaleRef *refs = k1_malloc((size_t)man.n / 2 * sizeof(K1ScaleRef) + sizeof(K1ScaleRef), "scale refs");
    int nref = 0;
    for (int i = 0; i < man.n; i++) {
        if (man.e[i].kind != K1X_SCALE) continue;
        st_tensor *t = st_find(S, man.e[i].name);
        K1ScaleRef *r = &refs[nref++];
        r->fd = t->fd; r->off = t->off; r->len = t->nbytes; r->dst = NULL; r->is_expert = 0;
        const char *p = strstr(man.e[i].name, ".mlp.experts.");
        if (p) {
            int l, e; char which[32];
            if (sscanf(man.e[i].name, "model.layers.%d.mlp.experts.%d.%31[a-z]_proj", &l, &e, which) == 3) {
                int w = !strcmp(which, "gate") ? K1_B_GATE : !strcmp(which, "up") ? K1_B_UP : K1_B_DOWN;
                r->dst = k1_bank(m, l, e, w); r->is_expert = 1;
            }
        }
    }
    m->n_scale_tensors = nref;
    qsort(refs, (size_t)nref, sizeof(K1ScaleRef), k1_cmp_scale);
    m->scale_min = INFINITY; m->scale_max = 0;
    size_t bufcap = 1 << 20; uint8_t *buf = k1_malloc(bufcap, "scale read buffer");
    for (int i = 0; i < nref; ) {
        int j = i + 1;
        int64_t end = refs[i].off + refs[i].len;
        while (j < nref && refs[j].fd == refs[i].fd && refs[j].off - end < K1_SCALE_GAP && refs[j].off + refs[j].len - refs[i].off < (64 << 20)) {
            end = refs[j].off + refs[j].len; j++;
        }
        int64_t span = end - refs[i].off;
        if ((size_t)span > bufcap) { bufcap = (size_t)span; buf = realloc(buf, bufcap); if (!buf) k1_die("out of memory for a scale range"); }
        st_read_range_raw_cap(S, refs[i].fd, refs[i].off, span, buf, (int64_t)bufcap, 0, "kolibri1 scales");
        for (int k = i; k < j; k++) {
            const float *v = (const float *)(buf + (refs[k].off - refs[i].off));
            int64_t n = refs[k].len / 4;
            float tmp[1];
            for (int64_t q = 0; q < n; q++) {
                memcpy(tmp, (const uint8_t *)v + q * 4, 4);
                float f = tmp[0];
                if (!isfinite(f)) {
                    /* name the tensor: the sorted refs lost it, so find it again by (fd, off) */
                    for (int a = 0; a < man.n; a++) {
                        if (man.e[a].kind != K1X_SCALE) continue;
                        st_tensor *t = st_find(S, man.e[a].name);
                        if (t->fd == refs[k].fd && t->off == refs[k].off)
                            k1_die("%s: non-finite scale at element %lld", man.e[a].name, (long long)q);
                    }
                    k1_die("non-finite scale value");
                }
                float af = fabsf(f);
                if (f == 0) m->scale_zero++;
                else {
                    if (f < 0) m->scale_negative++;
                    if (af < m->scale_min) m->scale_min = af;
                    if (af > m->scale_max) m->scale_max = af;
                }
            }
            if (refs[k].dst) memcpy(refs[k].dst, v, (size_t)refs[k].len);
        }
        i = j;
    }
    free(buf); free(refs); free(seen); free(man.e);
    k1_scan_free(&scan);
}

static void k1_close(K1Model *m) {
    free(m->er); free(m->bank);
    st_destroy(&m->S);
}

/* ------------------------------------------- the dequantisation contract ---- */

/* w[o,i] = e4m3(byte) * scale[o/128, i/128]: one fp32 multiply (the grid MULTIPLIES;
 * "weight_scale_inv" is a misnomer: it is the multiplier). Equal to torch's
 * `fp8.to(float32) * scale`. A NaN code (0x7F, 0xFF: e4m3fn has no infinity) stays
 * NaN for every finite scale. The product must be computed with FTZ/DAZ off so that
 * subnormal results are exact (k1_fp_env_check). */
static inline float k1_dequant(uint8_t code, float scale) { return e4m3_decode(code) * scale; }
static inline int k1_e4m3_is_nan(uint8_t code) { return (code & 0x7F) == 0x7F; }

/* Dequantise a [O,I] e4m3 matrix with its grid (row-major [ceil(O/128)][ceil(I/128)]). */
static void k1_dequant_matrix(float *W, const uint8_t *q, const float *grid, int O, int I) {
    int64_t nbI = fp8_nblk(I);
    for (int o = 0; o < O; o++) {
        const float *g = grid + (int64_t)(o / FP8_BLOCK) * nbI;
        for (int i = 0; i < I; i++) W[(int64_t)o * I + i] = k1_dequant(q[(int64_t)o * I + i], g[i / FP8_BLOCK]);
    }
}

/* 1 iff a scale tensor of shape [rows, cols] is the grid of an [O,I] matrix */
static int k1_grid_ok(int64_t O, int64_t I, int64_t rows, int64_t cols) {
    return rows == fp8_nblk((int)O) && cols == fp8_nblk((int)I);
}

/* 0 when the floating-point environment keeps subnormals (no flush-to-zero, no
 * denormals-are-zero), else a bit mask of what is wrong. */
static int k1_fp_env_check(void) {
    int bad = 0;
#if defined(__SSE__) || defined(__x86_64__)
    unsigned csr = _mm_getcsr();
    if (csr & 0x8000u) bad |= 1;      /* FTZ */
    if (csr & 0x0040u) bad |= 2;      /* DAZ */
#endif
    volatile float tiny = 0x1p-126f, half = 0.5f;
    volatile float p = tiny * half;           /* 2^-127: subnormal, nonzero when FTZ is off */
    if (p == 0.0f) bad |= 4;
    volatile float sub = 0x1p-140f, one = 1.0f;
    volatile float q = sub * one;             /* a subnormal operand survives when DAZ is off */
    if (q == 0.0f) bad |= 8;
    return bad;
}

/* ------------------------------------------------------------------ kernels ---- */

/* y[S,O] = x[S,I] @ W^T with W e4m3 [O,I] and a F32 grid [ceil(O/128), ceil(I/128)].
 * Per output row: fp32 sums inside each 128-block, then (double)block * scale added
 * in double (quant.h / the CUDA tier contract). A row's result does not depend on S,
 * on its position in the block or on the block size: each (o, s) runs the same
 * instruction sequence. Two implementations, each bit-stable across S on its own and
 * NOT bit-equal to each other (different reduction order inside a block). */
static void k1_matmul_fp8_scalar(float *y, const float *x, const uint8_t *q8, const float *bscale,
                                 int S, int I, int O) {
    int64_t nbI = fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q8 + (int64_t)o * I;
        const float *scl = bscale + ((int64_t)o / FP8_BLOCK) * nbI;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            double a = 0;
            for (int64_t bi = 0; bi * FP8_BLOCK < I; bi++) {
                int base = (int)(bi * FP8_BLOCK), blen = I - base < FP8_BLOCK ? I - base : FP8_BLOCK;
                float acc = 0;
                for (int i = 0; i < blen; i++) acc += e4m3_decode(w[base + i]) * xs[base + i];
                a += (double)acc * scl[bi];
            }
            y[(int64_t)s * O + o] = (float)a;
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
#define K1_HAVE_VEC 1
#define K1_FP8_ROWS 8
/* Eight FMA lanes per block, hsum256, then a scalar fused tail. Decode (S == 1)
 * decodes and multiplies in one pass; S > 1 decodes a block once into wf[] and runs
 * every row through it, with the same lane arithmetic, so a row's bits do not depend
 * on S. Duplicated from qwen38_core.h's q38_matmul_fp8_vec (no shared-header edit);
 * the tails use an explicit fmaf so that no compiler contraction choice can make the
 * S == 1 and S > 1 paths differ. */
static void k1_matmul_fp8_vec(float *y, const float *x, const uint8_t *q8, const float *bscale,
                              int S, int I, int O) {
    int64_t nbI = fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q8 + (int64_t)o * I;
        const float *scl = bscale + ((int64_t)o / FP8_BLOCK) * nbI;
        if (S == 1) {
            double a = 0;
            for (int64_t bi = 0; bi * FP8_BLOCK < I; bi++) {
                int base = (int)(bi * FP8_BLOCK), blen = I - base < FP8_BLOCK ? I - base : FP8_BLOCK, i = 0;
                __m256 acc = _mm256_setzero_ps();
                for (; i + 8 <= blen; i += 8)
                    acc = _mm256_fmadd_ps(e4m3_decode8(w + base + i), _mm256_loadu_ps(x + base + i), acc);
                float part = hsum256(acc);
                for (; i < blen; i++) part = fmaf(e4m3_decode(w[base + i]), x[base + i], part);
                a += (double)part * scl[bi];
            }
            y[o] = (float)a;
            continue;
        }
        for (int s0 = 0; s0 < S; s0 += K1_FP8_ROWS) {
            int ns = S - s0 < K1_FP8_ROWS ? S - s0 : K1_FP8_ROWS;
            double a[K1_FP8_ROWS] = { 0 };
            for (int64_t bi = 0; bi * FP8_BLOCK < I; bi++) {
                int base = (int)(bi * FP8_BLOCK), blen = I - base < FP8_BLOCK ? I - base : FP8_BLOCK, i = 0;
                float wf[FP8_BLOCK];
                for (; i + 8 <= blen; i += 8) _mm256_storeu_ps(wf + i, e4m3_decode8(w + base + i));
                for (; i < blen; i++) wf[i] = e4m3_decode(w[base + i]);
                float sc = scl[bi];
                for (int r = 0; r < ns; r++) {
                    const float *xs = x + (int64_t)(s0 + r) * I + base;
                    int k = 0;
                    __m256 acc = _mm256_setzero_ps();
                    for (; k + 8 <= blen; k += 8)
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(wf + k), _mm256_loadu_ps(xs + k), acc);
                    float part = hsum256(acc);
                    for (; k < blen; k++) part = fmaf(wf[k], xs[k], part);
                    a[r] += (double)part * sc;
                }
            }
            for (int r = 0; r < ns; r++) y[(int64_t)(s0 + r) * O + o] = (float)a[r];
        }
    }
}
#endif

static int k1_fp8_vector_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("K1_FP8_KERNEL"); v = !(e && !strcmp(e, "scalar")); }
    return v;
}

static void k1_matmul_fp8(float *y, const float *x, const uint8_t *q8, const float *bscale,
                          int S, int I, int O) {
#ifdef K1_HAVE_VEC
    if (k1_fp8_vector_on()) { k1_matmul_fp8_vec(y, x, q8, bscale, S, I, O); return; }
#endif
    k1_matmul_fp8_scalar(y, x, q8, bscale, S, I, O);
}

/* y[S,O] = x[S,I] @ W^T, W bf16 [O,I] widened exactly, fp32 activations (the
 * embedding table, the LM head). Same accumulation contract as the FP8 kernels: fp32
 * FMA lanes inside a 128-element chunk, the chunks added in double. (A plain fp32
 * accumulator over the whole row failed the kernel test on cancelling inputs: nRMS
 * 1.23e-6 against the 1e-6 limit; the double chunk sum costs nothing on a kernel
 * that is bound by the weight stream.) Row-independent like the FP8 kernels: a
 * row's bits do not depend on S. */
static void k1_matmul_bf16(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            double a = 0;
            for (int base = 0; base < I; base += FP8_BLOCK) {
                int blen = I - base < FP8_BLOCK ? I - base : FP8_BLOCK, i = 0;
                float part = 0;
#if defined(__AVX2__) && defined(__FMA__)
                __m256 acc = _mm256_setzero_ps();
                for (; i + 8 <= blen; i += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + base + i), bf16_decode8(w + base + i), acc);
                part = hsum256(acc);
                for (; i < blen; i++) part = fmaf(xs[base + i], bf16_to_f32(w[base + i]), part);
#else
                for (; i < blen; i++) part += xs[base + i] * bf16_to_f32(w[base + i]);
#endif
                a += (double)part;
            }
            y[(int64_t)s * O + o] = (float)a;
        }
    }
}

/* Router logits z[S,E] = x[S,H] @ gate^T with a DOUBLE accumulator over exact
 * (float x float) products: 1M MACs per token-layer, and the router weights are held
 * to atol 1e-6. */
static void k1_router_logits(float *z, const float *x, const uint16_t *W, int S, int H, int E) {
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < E; e++) {
        const uint16_t *w = W + (int64_t)e * H;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * H;
            double a = 0;
            for (int i = 0; i < H; i++) a += (double)xs[i] * (double)bf16_to_f32(w[i]);
            z[(int64_t)s * E + e] = (float)a;
        }
    }
}

/* RMSNorm: w * (x * rsqrt(mean(x^2) + eps)), the sum of squares and the reciprocal
 * root in double (n = 2560, free). out may alias x. */
static void k1_rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    double r = 1.0 / sqrt(ss / n + (double)eps);                 /* the reciprocal stays double */
    for (int i = 0; i < n; i++) out[i] = (float)((double)w[i] * ((double)x[i] * r));   /* one rounding into fp32 */
}

/* The same RMSNorm reading the NATIVE FP64 residual stream (never narrowed to fp32 first): double sum of squares, double
 * reciprocal root, one rounding into the fp32 activation that feeds the router / FP8 GEMMs / head. Identical arithmetic to k1_rmsnorm
 * for an input that happens to be fp32-representable. */
static void k1_rmsnorm_d(float *out, const double *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / n + (double)eps);
    for (int i = 0; i < n; i++) out[i] = (float)((double)w[i] * (x[i] * r));
}

/* RoPE tables. cos/sin per POSITION (head_dim/2 values), computed in double and
 * rounded once to float: inv_freq[i] = 1/theta^(2i/head_dim), angle = pos * inv_freq,
 * cos/sin of the double angle. mimo.c computes the angle in float32; at position 16k
 * that is off by about 1e-3 rad for the low pairs. */
typedef struct { int half; double inv[K1_MAX_HEAD / 2]; } K1Rope;

static void k1_rope_init(K1Rope *r, int head_dim, double theta) {
    r->half = head_dim / 2;
    for (int i = 0; i < r->half; i++) r->inv[i] = 1.0 / pow(theta, (double)(2 * i) / (double)head_dim);
}

static void k1_rope_tables(const K1Rope *r, int64_t pos, float *c, float *s) {
    for (int i = 0; i < r->half; i++) {
        double a = (double)pos * r->inv[i];
        c[i] = (float)cos(a);
        s[i] = (float)sin(a);
    }
}

/* rotate-half (neox) on one whole head: o1 = x1 c - x2 s, o2 = x2 c + x1 s */
static void k1_rope_apply(float *v, int head_dim, const float *c, const float *s) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float a = v[i], b = v[i + half];
        v[i] = a * c[i] - b * s[i];
        v[i + half] = b * c[i] + a * s[i];
    }
}

/* sigmoid_logit_add routing: ids = top-K of (z + bias) (selection on the BIASED raw
 * logits), w = sigmoid(z[id]) (UNBIASED), no renormalisation, no scaling. Selection
 * is a repeated max scan, so the first (lowest) id wins an exact tie, and the K
 * picks come out in descending biased order. A non-finite z or bias is a hard
 * error (return -1), never a silent fallback. */
typedef struct {
    const float *z;               /* the unbiased logits the caller passed (valid until it reuses them) */
    int ids[K1_MAX_TOPK];
    float w[K1_MAX_TOPK];
} K1Route;

static inline float k1_sigmoid(float x) {
    double v = x;
    return (float)(v >= 0 ? 1.0 / (1.0 + exp(-v)) : exp(v) / (1.0 + exp(v)));
}

static int k1_route(K1Route *r, const float *z, const float *bias, int E, int K) {
    if (K < 1 || K > K1_MAX_TOPK || K > E) return -1;
    float b[4096];
    if (E > 4096) return -1;
    for (int e = 0; e < E; e++) {
        if (!isfinite(z[e]) || !isfinite(bias[e])) return -1;
        b[e] = z[e] + bias[e];
        if (!isfinite(b[e])) return -1;
    }
    r->z = z;
    for (int k = 0; k < K; k++) {
        int best = -1;
        for (int e = 0; e < E; e++) if (b[e] > -INFINITY && (best < 0 || b[e] > b[best])) best = e;
        r->ids[k] = best;
        r->w[k] = k1_sigmoid(z[best]);
        b[best] = -INFINITY;
    }
    return 0;
}

/* K1_REPLAY_IDS (test only): layer l, rows p0..p0+n-1.  The natural selection (ids/w just computed by k1_route) is RECORDED, then the
 * schedule's IDs are executed instead, with w = sigmoid(this engine's own z[forced id]): no bias, no renormalisation. */
static void k1_replay_apply(int l, int p0, int n, int K, int E, const float *z, int *ids, float *w) {
    K1Replay *R = &k1rp;
    if (l >= R->L) k1_die("replay: layer %d is outside the schedule's %d layers", l, R->L);
    if (p0 < 0 || p0 + n > R->T) k1_die("replay: block [%d,+%d) is outside the schedule's %d positions", p0, n, R->T);
    for (int r = 0; r < n; r++) {
        size_t cell = (size_t)l * R->T + (size_t)(p0 + r);
        if (R->seen[cell]) k1_die("replay: layer %d position %d routed twice", l, p0 + r);
        R->seen[cell] = 1;
        int16_t *nat = R->nat + cell * (size_t)K;
        const int16_t *frc = R->ids + cell * (size_t)K;
        int same = 1;
        for (int k = 0; k < K; k++) nat[k] = (int16_t)ids[(size_t)r * K + k];
        for (int k = 0; k < K; k++) {
            int in = 0;
            for (int j = 0; j < K; j++) if (frc[j] == nat[k]) in = 1;
            if (!in) same = 0;
        }
        if (!same) R->differ++;
        R->rows++;
        if (l == R->mut_l && p0 + r == R->mut_p) {      /* control: the recorded natural 6th pick becomes an expert outside the natural set */
            int cand = nat[K - 1];
            for (int t = 0; t < E; t++) {
                cand = (cand + 1) % E;
                int used = 0;
                for (int k = 0; k < K; k++) if (nat[k] == cand) used = 1;
                if (!used) break;
            }
            nat[K - 1] = (int16_t)cand;
        }
        for (int k = 0; k < K; k++) {
            int id = frc[k];
            ids[(size_t)r * K + k] = id;
            w[(size_t)r * K + k] = k1_sigmoid(z[(size_t)r * E + id]);
        }
    }
}

/* ---------------------------------------------------------------- tokenizer ---- */

/* The release's pre-tokenizer regex, byte for byte. tok.h implements exactly this
 * shape (pretok_chunk: cl100k with the digit rule narrowed), but its detector for the
 * one-digit rule looks for `|\p{N}|` and this release writes `\p{N}{1}`, so tok_load
 * leaves ndig1 = 0 and digits would come out grouped by three. */
static const char K1_PRETOK_REGEX[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/* tok_load, then the engine-local, GUARDED digit override. Applied only when the
 * loaded pre-tokenizer is exactly [Split(Isolated, K1_PRETOK_REGEX), ByteLevel
 * (no regex, no prefix space)] with no normaliser and tok.h took the plain cl100k
 * path (if a future tok.h already sets ndig1 for this regex the override is a
 * no-op); anything else refuses to start. Returns 0 on success, otherwise the reason
 * (a static string) so a test can check it without a child process. */
static const char *k1_tok_guard(Tok *T) {
    jval *root = T->json_root;
    if (!root) return "tokenizer has no parsed JSON";
    jval *norm = json_get(root, "normalizer");
    if (norm && norm->t != J_NULL) return "tokenizer.json has a normalizer (this engine applies none)";
    jval *pt = json_get(root, "pre_tokenizer");
    if (!pt || pt->t != J_OBJ) return "no pre_tokenizer";
    jval *ty = json_get(pt, "type");
    jval *ps = json_get(pt, "pretokenizers");
    if (!ty || ty->t != J_STR || strcmp(ty->str, "Sequence") || !ps || ps->t != J_ARR || ps->len != 2)
        return "pre_tokenizer is not Sequence[Split, ByteLevel]";
    jval *sp = ps->kids[0], *bl = ps->kids[1];
    jval *t0 = json_get(sp, "type"), *t1 = json_get(bl, "type");
    if (!t0 || t0->t != J_STR || strcmp(t0->str, "Split") || !t1 || t1->t != J_STR || strcmp(t1->str, "ByteLevel"))
        return "pre_tokenizer is not Sequence[Split, ByteLevel]";
    jval *pat = json_get(sp, "pattern");
    jval *rx = pat ? json_get(pat, "Regex") : NULL;
    if (!rx || rx->t != J_STR || strcmp(rx->str, K1_PRETOK_REGEX))
        return "Split regex is not the Kolibri-1 pattern (the one-digit rule \\p{N}{1} and the rest must match exactly)";
    jval *bh = json_get(sp, "behavior");
    if (!bh || bh->t != J_STR || strcmp(bh->str, "Isolated")) return "Split behavior is not Isolated";
    jval *inv = json_get(sp, "invert");
    if (inv && inv->t == J_BOOL && inv->boolean) return "Split is inverted";
    jval *ur = json_get(bl, "use_regex");
    if (!ur || ur->t != J_BOOL || ur->boolean) return "ByteLevel uses its own regex";
    jval *aps = json_get(bl, "add_prefix_space");
    if (aps && aps->t == J_BOOL && aps->boolean) return "ByteLevel adds a prefix space";
    if (T->o200k || T->kimi || T->gpt2 || T->deepseek) return "tok.h chose a non-cl100k pre-tokenizer family";
    jval *model = json_get(root, "model");
    jval *mty = model ? json_get(model, "type") : NULL;
    if (!mty || mty->t != J_STR || strcmp(mty->str, "BPE")) return "model is not BPE";
    return NULL;
}

/* 1 when every one of the 256 byte-level characters is a vocab entry, which makes the
 * release's byte_fallback: true inert (no <0xNN> token is ever needed). */
static int k1_tok_bytes_all_in_vocab(Tok *T) {
    for (int b = 0; b < 256; b++)
        if (hm_get(&T->vocab, T->byte2str[b], T->byte2cp_len[b]) < 0) return 0;
    return 1;
}

static void k1_tok_load(Tok *T, const char *path) {
    tok_load(T, path);
    const char *why = k1_tok_guard(T);
    if (why) { fprintf(stderr, "[kolibri1] %s: %s -- refusing\n", path, why); exit(1); }
    if (!k1_tok_bytes_all_in_vocab(T)) {
        fprintf(stderr, "[kolibri1] %s: a byte-level character is missing from the vocab, so byte_fallback "
                        "would matter and this engine does not implement it -- refusing\n", path);
        exit(1);
    }
    T->ndig1 = 1;    /* \p{N}{1}: one digit per piece */
}

/* ============================================================ the forward pass ==== */

/* Everything below follows the checkpoint's documented forward pass with fp32 activations,
 * an FP64 residual stream, an fp32 router and fp32 head operands. The vendor stack's bf16
 * residual and Marlin rounding sequences do not exist here.
 *
 * Determinism contract: a row's result does not depend on the
 * block it was computed in. Every kernel computes a row with one fixed instruction
 * sequence, attention runs each query over its visible keys in ascending key
 * order, the routed sum is replayed per token in router rank order, and the expert cache
 * only decides where a weight lives, never what it is. OpenMP only splits independent
 * rows/heads, there is no cross-thread reduction, so two runs agree bit for bit. */

typedef struct { uint8_t *q; float *g; int O, I; } K1W;      /* FP8 matrix + its F32 grid */

typedef struct {
    float *ln_in, *ln_post_attn, *ln_pre_ffn, *ln_post_ffn, *qn, *kn;
    K1W q, k, v, o, sg, su, sd;
    uint16_t *gate;          /* bf16 [E,H] */
    float *bias;             /* [E] */
} K1Layer;

/* Routed-expert weights: a per-layer LRU of `cap` slots; a miss preads the expert's three
 * tensors from wherever the placement table says they are (they may be in two shards). */
typedef struct {
    int cap, used;
    int *slot_of;            /* [E] -> slot, -1 if absent */
    int *exp_of;             /* [cap] -> expert */
    uint64_t *tick, clock;
    uint8_t **w[3];          /* [K1_DOWN|K1_GATE|K1_UP][slot] */
    long hits, loads, evicts;
} K1Cache;

typedef struct {
    double *x, *xo, *ffn;  /* the residual stream (layer input, layer output, post-attention state) is stored and accumulated in FP64 */
    float *xin, *q, *k, *v, *ao, *o, *a, *x2, *z, *w, *routed, *shared, *msum, *f;
    double *routed64;      /* the routed sum before its fp32 rounding, consumed by op 13 */
    int *ids;
    float *xs, *eg, *eu, *ea, *ed, *yb, *sg_, *su_, *sa_;
    float *hn, *logit;
    int *tl, *rl;            /* (token, rank) lists of one expert */
} K1Ws;

typedef struct {
    K1Model m;
    K1Cfg c;
    K1Layer *L;
    uint16_t *embed, *head;
    float *norm;
    K1Cache *ec;
    int ctx, nmax, nlogit_rows;
    float **kc, **vc;        /* per layer: ctx rows (full) or ring rows (sliding) of kv_heads*head_dim floats */
    int *rows;
    int64_t **rpos;          /* sliding layers: the position held by each ring slot, -1 = empty */
    float *stk, *stv;        /* sliding-window staging */
    K1Rope rope;
    K1Ws ws;
} K1Eng;

static int k1_env_int(const char *name, int def, int lo, int hi) {
    const char *e = getenv(name);
    if (!e || !*e) return def;
    char *end; long v = strtol(e, &end, 10);
    if (*end || v < lo || v > hi) k1_die("%s=%s is not an integer in [%d, %d]", name, e, lo, hi);
    return (int)v;
}

static void k1_load_w(shards *S, K1W *w, int O, int I, const char *fmt, ...) {
    char base[128], nm[160];
    va_list ap; va_start(ap, fmt); vsnprintf(base, sizeof(base), fmt, ap); va_end(ap);
    w->O = O; w->I = I;
    w->q = k1_malloc((size_t)O * I, base);
    snprintf(nm, sizeof(nm), "%s.weight", base);
    st_read_raw_cap(S, nm, w->q, (int64_t)O * I, 0);
    int64_t ng = fp8_nblk(O) * fp8_nblk(I);
    w->g = k1_malloc((size_t)ng * sizeof(float), base);
    snprintf(nm, sizeof(nm), "%s.weight_scale_inv", base);
    st_read_f32_cap(S, nm, w->g, ng, 0);
}

static float *k1_load_vec(shards *S, const char *name, int64_t n) {
    float *v = k1_malloc((size_t)n * sizeof(float), name);
    st_read_f32_cap(S, name, v, n, 0);
    return v;
}

static uint16_t *k1_load_bf16(shards *S, const char *name, int64_t n) {
    uint16_t *v = k1_malloc((size_t)n * 2, name);
    st_read_raw_cap(S, name, v, n * 2, 0);
    return v;
}

#define K1_WS(field, count, type) E->ws.field = k1_malloc((size_t)(count) * sizeof(type), #field)

static void k1_ws_alloc(K1Eng *E) {
    const K1Cfg *c = &E->c;
    size_t n = (size_t)E->nmax, H = c->hidden, qw = (size_t)c->heads * c->head_dim, kw = (size_t)c->kv_heads * c->head_dim;
    size_t I = c->moe_inter, Is = c->shared_inter, EX = c->n_experts, K = c->topk;
    K1_WS(x, n * H, double); K1_WS(xo, n * H, double); K1_WS(xin, n * H, float);
    K1_WS(q, n * qw, float); K1_WS(k, n * kw, float); K1_WS(v, n * kw, float); K1_WS(ao, n * qw, float);
    K1_WS(o, n * H, float); K1_WS(a, n * H, float); K1_WS(ffn, n * H, double); K1_WS(x2, n * H, float);
    K1_WS(z, n * EX, float); K1_WS(w, n * K, float); K1_WS(ids, n * K, int);
    K1_WS(routed, n * H, float); K1_WS(routed64, n * H, double); K1_WS(shared, n * H, float); K1_WS(msum, n * H, float); K1_WS(f, n * H, float);
    K1_WS(xs, n * H, float); K1_WS(eg, n * I, float); K1_WS(eu, n * I, float); K1_WS(ea, n * I, float);
    K1_WS(ed, n * H, float); K1_WS(yb, n * K * H, float);
    K1_WS(sg_, n * Is, float); K1_WS(su_, n * Is, float); K1_WS(sa_, n * Is, float);
    K1_WS(hn, n * H, float); K1_WS(logit, n * (size_t)c->vocab, float);
    K1_WS(tl, n * K, int); K1_WS(rl, n * K, int);
}

/* Build the engine: validate (k1_open), then make every non-expert tensor resident.
 * `want_head`: embedding table + LM head (the probe modes that do not touch them skip the
 * 2 x vocab x hidden bf16 read). */
static void k1_engine_open(K1Eng *E, const char *dir, int release, int ctx, int nmax, int cap, int want_head) {
    memset(E, 0, sizeof(*E));
    k1_open(&E->m, dir, release);
    E->c = E->m.c;
    {   /* K1_LAYERS=n: a truncated model, for checking only. The whole checkpoint was validated above. */
        int nl = k1_env_int("K1_LAYERS", 0, 0, K1_MAX_LAYERS);
        if (nl) {
            if (nl > E->c.n_layers) k1_die("K1_LAYERS=%d exceeds the checkpoint's %d layers", nl, E->c.n_layers);
            fprintf(stderr, "[kolibri1] ******** K1_LAYERS=%d: TRUNCATED MODEL (%d of %d layers), for checking only ********\n",
                    nl, nl, E->c.n_layers);
            E->c.n_layers = nl;
        }
    }
    const K1Cfg *c = &E->c;
    shards *S = &E->m.S;
    int64_t H = c->hidden, qw = (int64_t)c->heads * c->head_dim, kw = (int64_t)c->kv_heads * c->head_dim;
    if (ctx < 1 || ctx > c->max_positions) k1_die("context %d is outside [1, max_position_embeddings %d]", ctx, c->max_positions);
    E->ctx = ctx; E->nmax = nmax;
    E->norm = k1_load_vec(S, "model.norm.weight", H);
    if (want_head) {
        E->embed = k1_load_bf16(S, "model.embed_tokens.weight", (int64_t)c->vocab * H);
        E->head = k1_load_bf16(S, "lm_head.weight", (int64_t)c->vocab * H);
    }
    E->L = k1_calloc((size_t)c->n_layers, sizeof(K1Layer), "layers");
    char nm[160];
    for (int l = 0; l < c->n_layers; l++) {
        K1Layer *L = &E->L[l];
#define K1_V(field, fmt, n) do { snprintf(nm, sizeof(nm), fmt, l); L->field = k1_load_vec(S, nm, n); } while (0)
        K1_V(ln_in, "model.layers.%d.input_layernorm.weight", H);
        K1_V(ln_post_attn, "model.layers.%d.post_attn_norm.weight", H);
        K1_V(ln_pre_ffn, "model.layers.%d.post_attention_layernorm.weight", H);
        K1_V(ln_post_ffn, "model.layers.%d.post_ffn_norm.weight", H);
        K1_V(qn, "model.layers.%d.self_attn.q_norm.weight", c->head_dim);
        K1_V(kn, "model.layers.%d.self_attn.k_norm.weight", c->head_dim);
        K1_V(bias, "model.layers.%d.moe.router.expert_bias", c->n_experts);
#undef K1_V
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.weight", l);
        L->gate = k1_load_bf16(S, nm, (int64_t)c->n_experts * H);
        k1_load_w(S, &L->q, (int)qw, (int)H, "model.layers.%d.self_attn.q_proj", l);
        k1_load_w(S, &L->k, (int)kw, (int)H, "model.layers.%d.self_attn.k_proj", l);
        k1_load_w(S, &L->v, (int)kw, (int)H, "model.layers.%d.self_attn.v_proj", l);
        k1_load_w(S, &L->o, (int)H, (int)qw, "model.layers.%d.self_attn.o_proj", l);
        k1_load_w(S, &L->sg, c->shared_inter, (int)H, "model.layers.%d.mlp.shared_experts.gate_proj", l);
        k1_load_w(S, &L->su, c->shared_inter, (int)H, "model.layers.%d.mlp.shared_experts.up_proj", l);
        k1_load_w(S, &L->sd, (int)H, c->shared_inter, "model.layers.%d.mlp.shared_experts.down_proj", l);
    }
    /* expert caches */
    if (cap < 1 || cap > c->n_experts) cap = c->n_experts;
    E->ec = k1_calloc((size_t)c->n_layers, sizeof(K1Cache), "expert caches");
    for (int l = 0; l < c->n_layers; l++) {
        K1Cache *C = &E->ec[l];
        C->cap = cap;
        C->slot_of = k1_malloc((size_t)c->n_experts * sizeof(int), "slot_of");
        for (int e = 0; e < c->n_experts; e++) C->slot_of[e] = -1;
        C->exp_of = k1_malloc((size_t)cap * sizeof(int), "exp_of");
        C->tick = k1_calloc((size_t)cap, sizeof(uint64_t), "tick");
        for (int p = 0; p < 3; p++) C->w[p] = k1_calloc((size_t)cap, sizeof(uint8_t *), "slots");
    }
    /* KV: fp32; full layers hold ctx rows, sliding layers a ring of min(window, ctx) rows */
    E->kc = k1_calloc((size_t)c->n_layers, sizeof(float *), "kc");
    E->vc = k1_calloc((size_t)c->n_layers, sizeof(float *), "vc");
    E->rows = k1_calloc((size_t)c->n_layers, sizeof(int), "rows");
    E->rpos = k1_calloc((size_t)c->n_layers, sizeof(int64_t *), "rpos");
    for (int l = 0; l < c->n_layers; l++) {
        int rows = c->full[l] ? ctx : (c->window < ctx ? c->window : ctx);
        E->rows[l] = rows;
        E->kc[l] = k1_calloc((size_t)rows * kw, sizeof(float), "K cache");
        E->vc[l] = k1_calloc((size_t)rows * kw, sizeof(float), "V cache");
        if (!c->full[l]) {
            E->rpos[l] = k1_malloc((size_t)rows * sizeof(int64_t), "ring positions");
            for (int i = 0; i < rows; i++) E->rpos[l][i] = -1;
        }
    }
    size_t stage = (size_t)(c->window - 1 + nmax);
    if (stage > (size_t)ctx) stage = (size_t)ctx;
    E->stk = k1_malloc(stage * kw * sizeof(float), "key staging");
    E->stv = k1_malloc(stage * kw * sizeof(float), "value staging");
    k1_rope_init(&E->rope, c->head_dim, c->theta);
    k1_ws_alloc(E);
}

static void k1_engine_close(K1Eng *E) {
    const K1Cfg *c = &E->c;
    long h = 0, ld = 0, ev = 0;
    for (int l = 0; l < c->n_layers; l++) {
        K1Cache *C = &E->ec[l];
        h += C->hits; ld += C->loads; ev += C->evicts;
    }
    fprintf(stderr, "[kolibri1] expert cache: %ld hits, %ld loads, %ld evictions (cap %d of %d per layer)\n",
            h, ld, ev, E->ec[0].cap, c->n_experts);
    k1_close(&E->m);
}

/* ------------------------------------------------------------------ the cache ---- */

/* the slot holding expert e of layer l, reading it from the shards on a miss (LRU victim) */
static int k1_expert_slot(K1Eng *E, int l, int e) {
    K1Cache *C = &E->ec[l];
    const K1Cfg *c = &E->c;
    int s = C->slot_of[e];
    if (s >= 0) { C->hits++; C->tick[s] = ++C->clock; return s; }
    size_t bytes = (size_t)c->hidden * c->moe_inter;
    if (C->used < C->cap) {
        s = C->used++;
        for (int p = 0; p < 3; p++) C->w[p][s] = k1_malloc(bytes, "expert slot");
    } else {
        s = 0;
        for (int i = 1; i < C->cap; i++) if (C->tick[i] < C->tick[s]) s = i;
        if (C->exp_of[s] >= 0) C->slot_of[C->exp_of[s]] = -1;     /* -1: a slot that holds no expert */
        C->evicts++;
    }
    const K1ERef *r = &E->m.er[(size_t)l * c->n_experts + e];
    for (int p = 0; p < 3; p++) {
        if ((size_t)r->len[p] != bytes) k1_die("expert %d of layer %d: tensor %d holds %lld bytes, expected %zu", e, l, p, (long long)r->len[p], bytes);
        st_read_range_raw_cap(&E->m.S, r->fd[p], r->off[p], r->len[p], C->w[p][s], (int64_t)bytes, 0, "kolibri1 expert");
    }
    C->slot_of[e] = s; C->exp_of[s] = e; C->tick[s] = ++C->clock; C->loads++;
    return s;
}

/* ------------------------------------------------------------------ operators ---- */

static void k1_rows_rms(float *out, const float *x, const float *w, int n, int dim, float eps) {
    for (int r = 0; r < n; r++) k1_rmsnorm(out + (size_t)r * dim, x + (size_t)r * dim, w, dim, eps);
}

static void k1_rows_rms_d(float *out, const double *x, const float *w, int n, int dim, float eps) {   /* FP64 residual input */
    for (int r = 0; r < n; r++) k1_rmsnorm_d(out + (size_t)r * dim, x + (size_t)r * dim, w, dim, eps);
}

static inline float k1_silu_mul(float g, float u) {
    return (float)(((double)g / (1.0 + exp(-(double)g))) * (double)u);   /* silu(g)*u in double, one cast */
}

/* op 2: q, k, v from the same normed input (separate tensors in this checkpoint) */
static void k1_op_qkv(K1Eng *E, int l, const float *xin, int n, float *q, float *k, float *v) {
    K1Layer *L = &E->L[l];
    int H = E->c.hidden;
    k1_matmul_fp8(q, xin, L->q.q, L->q.g, n, H, L->q.O);
    k1_matmul_fp8(k, xin, L->k.q, L->k.g, n, H, L->k.O);
    k1_matmul_fp8(v, xin, L->v.q, L->v.g, n, H, L->v.O);
}

/* op 3: per-head RMSNorm of q and k over head_dim, shared [head_dim] weights; v is not normed */
static void k1_op_qknorm(K1Eng *E, int l, float *q, float *k, int n) {
    const K1Cfg *c = &E->c; K1Layer *L = &E->L[l];
    int hd = c->head_dim;
    for (int r = 0; r < n; r++) {
        for (int h = 0; h < c->heads; h++) { float *p = q + ((size_t)r * c->heads + h) * hd; k1_rmsnorm(p, p, L->qn, hd, c->eps); }
        for (int h = 0; h < c->kv_heads; h++) { float *p = k + ((size_t)r * c->kv_heads + h) * hd; k1_rmsnorm(p, p, L->kn, hd, c->eps); }
    }
}

/* op 4: rotate-half RoPE on the whole head, absolute positions p0 + row (sliding layers only) */
static void k1_op_rope(K1Eng *E, float *q, float *k, int n, int p0) {
    const K1Cfg *c = &E->c;
    int hd = c->head_dim;
    float cs[K1_MAX_HEAD / 2], sn[K1_MAX_HEAD / 2];
    for (int r = 0; r < n; r++) {
        k1_rope_tables(&E->rope, (int64_t)p0 + r, cs, sn);
        for (int h = 0; h < c->heads; h++) k1_rope_apply(q + ((size_t)r * c->heads + h) * hd, hd, cs, sn);
        for (int h = 0; h < c->kv_heads; h++) k1_rope_apply(k + ((size_t)r * c->kv_heads + h) * hd, hd, cs, sn);
    }
}

/* op 5: scaled causal GQA attention over the fp32 cache. Block of n queries at positions
 * p0..p0+n-1; the block's K/V rows are given (post norm, post rope). Sliding layers see the
 * current token plus window-1 previous ones; the ring is written AFTER the attention (token t
 * of a block would otherwise overwrite the slot of position p0+t-window, which the queries
 * before t of the same block still need). Full layers write first and read the cache. */
/* K1_ATTN_KEYS=<file>: for every query of every attention call, the keys the score loop really
 * visited (first position, count, last position), taken from the loop itself, not from a formula. */
static FILE *k1_ak;

static void k1_attention(K1Eng *E, int l, const float *q, const float *k, const float *v, int n, int p0, float *ao) {
    const K1Cfg *c = &E->c;
    int hd = c->head_dim, kvw = c->kv_heads * hd, G = c->heads / c->kv_heads, full = c->full[l], win = c->window;
    int rows = E->rows[l];
    if (p0 < 0 || p0 + n > E->ctx) k1_die("attention block [%d,+%d) exceeds the context %d", p0, n, E->ctx);
    const float *Kb, *Vb; int kpos0;
    if (full) {
        memcpy(E->kc[l] + (size_t)p0 * kvw, k, (size_t)n * kvw * sizeof(float));
        memcpy(E->vc[l] + (size_t)p0 * kvw, v, (size_t)n * kvw * sizeof(float));
        Kb = E->kc[l]; Vb = E->vc[l]; kpos0 = 0;
    } else {
        int lo = p0 - (win - 1); if (lo < 0) lo = 0;
        for (int j = lo; j < p0; j++) {
            int slot = j % rows;
            if (E->rpos[l][slot] != j) k1_die("layer %d: ring slot %d holds position %lld, expected %d", l, slot, (long long)E->rpos[l][slot], j);
            memcpy(E->stk + (size_t)(j - lo) * kvw, E->kc[l] + (size_t)slot * kvw, (size_t)kvw * sizeof(float));
            memcpy(E->stv + (size_t)(j - lo) * kvw, E->vc[l] + (size_t)slot * kvw, (size_t)kvw * sizeof(float));
        }
        memcpy(E->stk + (size_t)(p0 - lo) * kvw, k, (size_t)n * kvw * sizeof(float));
        memcpy(E->stv + (size_t)(p0 - lo) * kvw, v, (size_t)n * kvw * sizeof(float));
        Kb = E->stk; Vb = E->stv; kpos0 = lo;
    }
    const double scale = 1.0 / sqrt((double)hd);
    int *akv = k1_ak ? k1_malloc((size_t)n * c->heads * 3 * sizeof(int), "attention key record") : NULL;
    int nk_max = full ? p0 + n : (p0 + n - kpos0);
    #pragma omp parallel
    {
        double *sc = k1_malloc((size_t)nk_max * sizeof(double), "scores");
        double *acc = k1_malloc((size_t)hd * sizeof(double), "attention accumulator");
        #pragma omp for schedule(static) collapse(2)
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < c->heads; h++) {
                int pq = p0 + t;
                int jlo = full ? 0 : (pq - (win - 1) < 0 ? 0 : pq - (win - 1));
                int kvh = h / G;
                const float *qh = q + ((size_t)t * c->heads + h) * hd;
                double m = -INFINITY;
                int vis_first = -1, vis_n = 0, vis_last = -1;
                for (int j = jlo; j <= pq; j++) {
                    if (vis_first < 0) vis_first = j;
                    vis_n++; vis_last = j;
                    const float *kr = Kb + (size_t)(j - kpos0) * kvw + (size_t)kvh * hd;
                    double s = 0;
                    for (int d = 0; d < hd; d++) s += (double)qh[d] * (double)kr[d];
                    s *= scale;
                    sc[j - jlo] = s;
                    if (s > m) m = s;
                }
                double sum = 0;
                for (int j = jlo; j <= pq; j++) { double e = exp(sc[j - jlo] - m); sc[j - jlo] = e; sum += e; }
                for (int d = 0; d < hd; d++) acc[d] = 0;
                for (int j = jlo; j <= pq; j++) {
                    const float *vr = Vb + (size_t)(j - kpos0) * kvw + (size_t)kvh * hd;
                    double p = sc[j - jlo];
                    for (int d = 0; d < hd; d++) acc[d] += p * (double)vr[d];
                }
                float *out = ao + ((size_t)t * c->heads + h) * hd;
                for (int d = 0; d < hd; d++) out[d] = (float)(acc[d] / sum);
                if (akv) { int *r = akv + ((size_t)t * c->heads + h) * 3; r[0] = vis_first; r[1] = vis_n; r[2] = vis_last; }
            }
        }
        free(sc); free(acc);
    }
    if (akv) {
        for (int t = 0; t < n; t++) {
            const int *r0 = akv + (size_t)t * c->heads * 3;
            for (int h = 1; h < c->heads; h++)
                if (memcmp(r0, r0 + (size_t)h * 3, 3 * sizeof(int))) k1_die("layer %d position %d: heads saw different key ranges", l, p0 + t);
            fprintf(k1_ak, "K %d %d %d %d %d\n", l, p0 + t, r0[0], r0[1], r0[2]);
        }
        free(akv);
    }
    if (!full) {
        int from = n > rows ? n - rows : 0;
        for (int t = from; t < n; t++) {
            int slot = (p0 + t) % rows;
            memcpy(E->kc[l] + (size_t)slot * kvw, k + (size_t)t * kvw, (size_t)kvw * sizeof(float));
            memcpy(E->vc[l] + (size_t)slot * kvw, v + (size_t)t * kvw, (size_t)kvw * sizeof(float));
            E->rpos[l][slot] = p0 + t;
        }
    }
}

/* op 9-10: router logits z = x2 gate^T (fp32 result, double accumulation), then
 * sigmoid_logit_add routing. A non-finite value is a hard error. */
static void k1_op_route(K1Eng *E, int l, const float *z, int n, int *ids, float *w) {
    const K1Cfg *c = &E->c;
    for (int r = 0; r < n; r++) {
        K1Route R;
        if (k1_route(&R, z + (size_t)r * c->n_experts, E->L[l].bias, c->n_experts, c->topk) < 0)
            k1_die("layer %d row %d: the router saw a non-finite logit or bias", l, r);
        for (int k = 0; k < c->topk; k++) { ids[(size_t)r * c->topk + k] = R.ids[k]; w[(size_t)r * c->topk + k] = R.w[k]; }
    }
}

/* op 11: routed experts. Tokens are grouped by expert (so each expert's weights are read
 * once per block), each y = w_e * down(silu(gate x) * up x) lands in yb[token][rank], and the
 * sum per token runs in rank order: the same association for any chunking or cache size. */
static void k1_op_routed(K1Eng *E, int l, const float *x2, const int *ids, const float *w, int n, float *routed, double *routed64) {
    const K1Cfg *c = &E->c; K1Ws *W = &E->ws;
    int H = c->hidden, I = c->moe_inter, K = c->topk;
    for (int e = 0; e < c->n_experts; e++) {
        int cnt = 0;
        for (int r = 0; r < n; r++)
            for (int k = 0; k < K; k++)
                if (ids[(size_t)r * K + k] == e) { W->tl[cnt] = r; W->rl[cnt] = k; cnt++; }
        if (!cnt) continue;
        int s = k1_expert_slot(E, l, e);
        K1Cache *C = &E->ec[l];
        for (int i = 0; i < cnt; i++) memcpy(W->xs + (size_t)i * H, x2 + (size_t)W->tl[i] * H, (size_t)H * sizeof(float));
        k1_matmul_fp8(W->eg, W->xs, C->w[K1_GATE][s], k1_bank(&E->m, l, e, K1_B_GATE), cnt, H, I);
        k1_matmul_fp8(W->eu, W->xs, C->w[K1_UP][s], k1_bank(&E->m, l, e, K1_B_UP), cnt, H, I);
        for (size_t i = 0; i < (size_t)cnt * I; i++) W->ea[i] = k1_silu_mul(W->eg[i], W->eu[i]);
        k1_matmul_fp8(W->ed, W->ea, C->w[K1_DOWN][s], k1_bank(&E->m, l, e, K1_B_DOWN), cnt, I, H);
        /* yb keeps the UNWEIGHTED expert output (fp32, as produced by the down projection); the weight multiplication
         * and the rank-order sum both happen in double below, with one fp32 rounding at the end */
        for (int i = 0; i < cnt; i++)
            memcpy(W->yb + ((size_t)W->tl[i] * K + W->rl[i]) * H, W->ed + (size_t)i * H, (size_t)H * sizeof(float));
    }
    for (int r = 0; r < n; r++) {
        float *o = routed + (size_t)r * H;
        double *o64 = routed64 ? routed64 + (size_t)r * H : NULL;
        for (int j = 0; j < H; j++) {
            double acc = 0;
            for (int k = 0; k < K; k++) acc += (double)w[(size_t)r * K + k] * (double)W->yb[((size_t)r * K + k) * H + j];
            o[j] = (float)acc;
            if (o64) o64[j] = acc;
        }
    }
}

/* op 12: the shared expert, ungated, on the same x2 */
static void k1_op_shared(K1Eng *E, int l, const float *x2, int n, float *out) {
    K1Layer *L = &E->L[l]; K1Ws *W = &E->ws;
    int H = E->c.hidden, Is = E->c.shared_inter;
    k1_matmul_fp8(W->sg_, x2, L->sg.q, L->sg.g, n, H, Is);
    k1_matmul_fp8(W->su_, x2, L->su.q, L->su.g, n, H, Is);
    for (size_t i = 0; i < (size_t)n * Is; i++) W->sa_[i] = k1_silu_mul(W->sg_[i], W->su_[i]);
    k1_matmul_fp8(out, W->sa_, L->sd.q, L->sd.g, n, Is, H);
}

/* One decoder layer on n rows at positions p0..: x_prev -> x_out (ops 1-15).
 * The intermediates stay in the workspace (the probe reads them). */
static void k1_layer(K1Eng *E, int l, const double *xp, int n, int p0) {
    const K1Cfg *c = &E->c; K1Layer *L = &E->L[l]; K1Ws *W = &E->ws;
    int H = c->hidden;
    size_t nH = (size_t)n * H;
    k1_rows_rms_d(W->xin, xp, L->ln_in, n, H, c->eps);                     /* 1: reads the FP64 stream */
    k1_op_qkv(E, l, W->xin, n, W->q, W->k, W->v);                          /* 2 */
    k1_op_qknorm(E, l, W->q, W->k, n);                                     /* 3 */
    if (!c->full[l]) k1_op_rope(E, W->q, W->k, n, p0);                     /* 4: sliding layers only, full layers are NoPE */
    k1_attention(E, l, W->q, W->k, W->v, n, p0, W->ao);                    /* 5 */
    k1_matmul_fp8(W->o, W->ao, L->o.q, L->o.g, n, c->heads * c->head_dim, H);   /* 6 */
    k1_rows_rms(W->a, W->o, L->ln_post_attn, n, H, c->eps);                /* 7 */
    for (size_t i = 0; i < nH; i++) W->ffn[i] = xp[i] + (double)W->a[i];   /* 8: FP64 residual add, stored FP64 */
    k1_rows_rms_d(W->x2, W->ffn, L->ln_pre_ffn, n, H, c->eps);
    k1_router_logits(W->z, W->x2, L->gate, n, H, c->n_experts);            /* 9 */
    k1_op_route(E, l, W->z, n, W->ids, W->w);                              /* 10 */
    if (k1rp.on) k1_replay_apply(l, p0, n, c->topk, c->n_experts, W->z, W->ids, W->w);   /* test-only K1_REPLAY_IDS: execute the other side's IDs */
    k1_op_routed(E, l, W->x2, W->ids, W->w, n, W->routed, W->routed64);   /* 11 */
    k1_op_shared(E, l, W->x2, n, W->shared);                               /* 12 */
    for (size_t i = 0; i < nH; i++) W->msum[i] = (float)((double)W->shared[i] + W->routed64[i]);   /* 13: before the norm; fp64 sum, one fp32 store */
    k1_rows_rms(W->f, W->msum, L->ln_post_ffn, n, H, c->eps);              /* 14 */
    for (size_t i = 0; i < nH; i++) W->xo[i] = W->ffn[i] + (double)W->f[i];   /* 15: the completed stream, FP64 */
    if (k1cap.on) k1_cap_layer(l, p0, n, W->xo, W->z, W->ids, W->w);      /* test-only capture (K1_CAPTURE) */
}

/* op 16-17: final norm and the fp32 head on the given rows */
static void k1_head(K1Eng *E, const double *x, int n, float *hn, float *logits) {
    k1_rows_rms_d(hn, x, E->norm, n, E->c.hidden, E->c.eps);
    if (logits) k1_matmul_bf16(logits, hn, E->head, n, E->c.hidden, E->c.vocab);
}

/* Every logit the head produces is checked before it is written, streamed or sampled: a NaN or Inf
 * (corrupt head row, overflow upstream) is a hard error with a diagnostic, never a silent argmax. */
static int k1_logits_soft;          /* SERVE: report a non-finite logit through k1_logits_bad (an error reply) instead of exit(3) */
static char k1_logits_bad[200];

static void k1_check_logits(const float *lg, int rows, int V, int pos0) {
    for (int r = 0; r < rows; r++)
        for (int v = 0; v < V; v++)
            if (!isfinite(lg[(size_t)r * V + v])) {
                if (k1_logits_soft) {
                    snprintf(k1_logits_bad, sizeof(k1_logits_bad), "NONFINITE_LOGITS %g at position %d, vocabulary id %d",
                             (double)lg[(size_t)r * V + v], pos0 + r, v);
                    fprintf(stderr, "[kolibri1] non-finite logit %g at position %d, vocabulary id %d -- the request is refused\n",
                            (double)lg[(size_t)r * V + v], pos0 + r, v);
                    return;
                }
                fprintf(stderr, "[kolibri1] non-finite logit %g at position %d, vocabulary id %d (head output of the model) -- refusing\n",
                        (double)lg[(size_t)r * V + v], pos0 + r, v);
                exit(3);
            }
}

/* A block of n token ids at positions p0..: embedding, all layers, then the head on the last
 * row (all rows when `all_rows`). Logits land in ws.logit[rows][vocab]; returns the row count. */
static int k1_forward(K1Eng *E, const int *ids, int n, int p0, int all_rows) {
    const K1Cfg *c = &E->c; K1Ws *W = &E->ws;
    int H = c->hidden;
    if (n > E->nmax) k1_die("block of %d rows exceeds the workspace (%d)", n, E->nmax);
    for (int r = 0; r < n; r++) {
        if (ids[r] < 0 || ids[r] >= c->vocab) k1_die("token id %d is outside the vocabulary [0, %d)", ids[r], c->vocab);
        const uint16_t *e = E->embed + (size_t)ids[r] * H;
        for (int j = 0; j < H; j++) W->x[(size_t)r * H + j] = (double)bf16_to_f32(e[j]);   /* op 0: no scaling; FP64 (BF16 -> FP64 is exact) */
    }
    double *cur = W->x;
    for (int l = 0; l < c->n_layers; l++) {
        k1_layer(E, l, cur, n, p0);
        double *t = W->xo; W->xo = W->x; W->x = t; cur = W->x;     /* ping-pong: cur holds the layer's output */
    }
    if (k1cap.fifo_fd >= 0) {   /* K1_LOGITS_FIFO: the engine's own final norm + head in slices that never cross a multiple of 128 */
        k1_rows_rms_d(W->hn, cur, E->norm, n, H, c->eps);
        float *fb = k1_malloc((size_t)K1_CAP_CHUNK * c->vocab * sizeof(float), "logit frame");
        for (int s = 0; s < n; ) {
            int pos = p0 + s, m = K1_CAP_CHUNK - pos % K1_CAP_CHUNK;
            if (m > n - s) m = n - s;
            k1_matmul_bf16(fb, W->hn + (size_t)s * H, E->head, m, H, c->vocab);
            k1_check_logits(fb, m, c->vocab, pos);
            k1_cap_frame(pos, m, c->vocab, fb);
            s += m;
        }
        free(fb);
    }
    int first = all_rows ? 0 : n - 1, cnt = n - first;
    k1_head(E, cur + (size_t)first * H, cnt, W->hn, W->logit);
    k1_check_logits(W->logit, cnt, c->vocab, p0 + first);
    return cnt;
}

/* ------------------------------------------------------------- run / generate ---- */

static int *k1_parse_ids(const char *s, int *n_out) {
    int cap = 64, n = 0;
    int *a = k1_malloc((size_t)cap * sizeof(int), "ids");
    const char *p = s;
    while (*p) {
        char *end; long v = strtol(p, &end, 10);
        if (end == p) k1_die("--ids: '%s' is not a comma/space separated list of integers", s);
        if (n == cap) { cap *= 2; a = realloc(a, (size_t)cap * sizeof(int)); if (!a) k1_die("out of memory"); }
        a[n++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    *n_out = n;
    return a;
}

static int k1_argmax(const float *x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

/* ---- requests, cache reset, snapshots (prefix reuse) ---- */

typedef struct { int *ids; int n; } K1Req;

/* "a,b,c;d,e" -> two requests */
static K1Req *k1_parse_reqs(const char *s, int *nreq) {
    int n = 1;
    for (const char *p = s; *p; p++) if (*p == ';') n++;
    K1Req *r = k1_calloc((size_t)n, sizeof(K1Req), "requests");
    char *copy = strdup(s); if (!copy) k1_die("out of memory");
    char *tok = copy;
    int i = 0;
    for (char *p = copy; ; p++) {
        if (*p == ';' || !*p) {
            int end = !*p; *p = 0;
            r[i].ids = k1_parse_ids(tok, &r[i].n); i++;
            if (end) break;
            tok = p + 1;
        }
    }
    free(copy);
    *nreq = n;
    return r;
}

/* the cache of the previous request must not be readable by the next one: every K/V row is poisoned with NaN
 * (a stale read would turn the output into NaN) and every ring slot is marked empty */
static void k1_reset(K1Eng *E) {
    const K1Cfg *c = &E->c;
    size_t kw = (size_t)c->kv_heads * c->head_dim;
    for (int l = 0; l < c->n_layers; l++) {
        size_t n = (size_t)E->rows[l] * kw;
        for (size_t i = 0; i < n; i++) { E->kc[l][i] = NAN; E->vc[l][i] = NAN; }
        if (E->rpos[l]) for (int i = 0; i < E->rows[l]; i++) E->rpos[l][i] = -1;
    }
}

typedef struct { int P; float **kc, **vc; int64_t **rpos; } K1Snap;

static K1Snap *k1_snap_take(const K1Eng *E, int P) {
    const K1Cfg *c = &E->c;
    size_t kw = (size_t)c->kv_heads * c->head_dim;
    K1Snap *S = k1_calloc(1, sizeof(*S), "snapshot");
    S->P = P;
    S->kc = k1_calloc((size_t)c->n_layers, sizeof(float *), "snap kc");
    S->vc = k1_calloc((size_t)c->n_layers, sizeof(float *), "snap vc");
    S->rpos = k1_calloc((size_t)c->n_layers, sizeof(int64_t *), "snap rpos");
    for (int l = 0; l < c->n_layers; l++) {
        size_t rows = c->full[l] ? (size_t)P : (size_t)E->rows[l];     /* full: the history [0,P); sliding: the whole ring */
        S->kc[l] = k1_malloc(rows * kw * sizeof(float), "snap K");
        S->vc[l] = k1_malloc(rows * kw * sizeof(float), "snap V");
        memcpy(S->kc[l], E->kc[l], rows * kw * sizeof(float));
        memcpy(S->vc[l], E->vc[l], rows * kw * sizeof(float));
        if (!c->full[l]) { S->rpos[l] = k1_malloc((size_t)E->rows[l] * sizeof(int64_t), "snap rpos"); memcpy(S->rpos[l], E->rpos[l], (size_t)E->rows[l] * sizeof(int64_t)); }
    }
    return S;
}

static void k1_snap_restore(K1Eng *E, const K1Snap *S) {
    const K1Cfg *c = &E->c;
    size_t kw = (size_t)c->kv_heads * c->head_dim;
    for (int l = 0; l < c->n_layers; l++) {
        size_t rows = c->full[l] ? (size_t)S->P : (size_t)E->rows[l];
        memcpy(E->kc[l], S->kc[l], rows * kw * sizeof(float));
        memcpy(E->vc[l], S->vc[l], rows * kw * sizeof(float));
        if (!c->full[l]) memcpy(E->rpos[l], S->rpos[l], (size_t)E->rows[l] * sizeof(int64_t));
    }
}

static void k1_ak_dump_cache(const K1Eng *E, int req) {
    if (!k1_ak) return;
    for (int l = 0; l < E->c.n_layers; l++) {
        if (E->c.full[l]) { fprintf(k1_ak, "F %d %d %d\n", req, l, E->rows[l]); continue; }
        for (int i = 0; i < E->rows[l]; i++) fprintf(k1_ak, "R %d %d %d %lld\n", req, l, i, (long long)E->rpos[l][i]);
    }
}

typedef struct {
    int chunk, tail, snap_at;
    FILE *lf;
    K1Snap *snap;
} K1Feed;

/* feed ids[from..n) as positions base+from.. : prefill chunks (cut at the snapshot position), then `tail`
 * single-token steps (S = 1, the path the CUDA tier serves). Returns the greedy token after the last one. */
static int k1_feed(K1Eng *E, const int *ids, int n, int from, int base, K1Feed *F, int *nextp) {
    int V = E->c.vocab, last = -1;
    int tail = F->tail < n - from ? F->tail : n - from;
    int pre_end = n - tail;
    for (int i = from; i < pre_end; ) {
        int m = pre_end - i < F->chunk ? pre_end - i : F->chunk;
        if (F->snap_at > i && F->snap_at < i + m) m = F->snap_at - i;
        int rows = k1_forward(E, ids + i, m, base + i, F->lf != NULL);
        if (F->lf && fwrite(E->ws.logit, sizeof(float) * (size_t)V, (size_t)rows, F->lf) != (size_t)rows) k1_die("short write to K1_LOGITS");
        last = k1_argmax(E->ws.logit + (size_t)(rows - 1) * V, V);
        i += m;
        if (F->snap_at && i == F->snap_at && !F->snap) F->snap = k1_snap_take(E, F->snap_at);
    }
    for (int i = pre_end; i < n; i++) {
        int rows = k1_forward(E, ids + i, 1, base + i, F->lf != NULL);
        if (F->lf && fwrite(E->ws.logit, sizeof(float) * (size_t)V, (size_t)rows, F->lf) != (size_t)rows) k1_die("short write to K1_LOGITS");
        last = k1_argmax(E->ws.logit + (size_t)(rows - 1) * V, V);
        if (F->snap_at && i + 1 == F->snap_at && !F->snap) F->snap = k1_snap_take(E, F->snap_at);
    }
    *nextp = base + n;
    return last;
}

static int k1_cmp_name(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

/* sha256 of the RUNNING executable: /proc/self/exe on Linux; on macOS the path the loader reports (so the Mac tests record a real hash). "" if neither can be read. */
static int k1_self_exe_sha(char hex[65]) {
    if (k1_sha_file("/proc/self/exe", hex) == 0) return 0;
#ifdef __APPLE__
    char path[4096]; uint32_t n = sizeof(path);
    if (_NSGetExecutablePath(path, &n) == 0 && k1_sha_file(path, hex) == 0) return 0;
#endif
    hex[0] = 0;
    return -1;
}

/* what ENGINE.json says about the inputs and the run (the comparator adds the checkpoint manifest and the verdicts) */
static void k1_engine_meta(char *out, size_t cap, const char *dir, const char *feed_path, int T, const K1Eng *E,
                           int chunk, int tail, int ctx, int cap_slots, double wall, double load_s, int release) {
    char h_cfg[65] = "", h_tok[65] = "", h_feed[65] = "", h_exe[65] = "", p[4200];
    snprintf(p, sizeof(p), "%s/config.json", dir); k1_sha_file(p, h_cfg);
    snprintf(p, sizeof(p), "%s/tokenizer.json", dir); k1_sha_file(p, h_tok);
    if (feed_path) k1_sha_file(feed_path, h_feed);
    k1_self_exe_sha(h_exe);
    long h = 0, ld = 0, ev = 0;
    for (int l = 0; l < E->c.n_layers; l++) { h += E->ec[l].hits; ld += E->ec[l].loads; ev += E->ec[l].evicts; }
    static char names[1024][96];
    int nn = 0;
    DIR *d = opendir(dir);
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) && nn < 1024) {
            size_t L = strlen(de->d_name);
            if (L > 12 && !strcmp(de->d_name + L - 12, ".safetensors") && L < 96) snprintf(names[nn++], 96, "%s", de->d_name);
        }
        closedir(d);
    }
    qsort(names, (size_t)nn, 96, k1_cmp_name);
    size_t n = 0;
    n += snprintf(out + n, cap - n, "\"config_sha256\":\"%s\",\"tokenizer_sha256\":\"%s\",\"feed_sha256\":\"%s\",\"exe_sha256\":\"%s\","
                  "\"n_layers_run\":%d,\"n_layers_checkpoint\":%d,\"truncated\":%s,\"k1_chunk\":%d,\"k1_decode_tail\":%d,\"k1_ctx\":%d,\"k1_cap\":%d,"
                  "\"fp8_kernel\":\"%s\",\"expert_cache\":{\"hits\":%ld,\"loads\":%ld,\"evictions\":%ld},\"peak_rss_gb\":%.3f,\"wall_s\":%.2f,\"load_s\":%.2f,"
                  "\"input_positions\":%d,\"release\":%s,\"residual_policy\":\"kolibri1-residual-fp64-c1\",\"shards\":[",
                  h_cfg, h_tok, h_feed, h_exe, E->c.n_layers, E->m.c.n_layers, E->c.n_layers != E->m.c.n_layers ? "true" : "false",
                  chunk, tail, ctx, cap_slots, k1_fp8_vector_on() ? "vector" : "scalar", h, ld, ev, k1_rss_gb(), wall, load_s, T,
                  release ? "true" : "false");
    for (int i = 0; i < nn && n + 200 < cap; i++) {
        struct stat st; snprintf(p, sizeof(p), "%.3000s/%.90s", dir, names[i]);
        long long sz = stat(p, &st) == 0 ? (long long)st.st_size : -1;
        n += snprintf(out + n, cap - n, "%s{\"name\":\"%s\",\"bytes\":%lld}", i ? "," : "", names[i], sz);
    }
    snprintf(out + n, cap - n, "]");
}

/* requests (';'-separated id lists) -> greedy continuation of `ngen` tokens each.
 *   K1_CHUNK=n           prefill rows per block (default 64)
 *   K1_DECODE_TAIL=m     prefill all but the last m tokens in chunks, then feed those m one at a time
 *   K1_CAP, K1_CTX       expert slots per layer; context rows (default: the longest request + ngen)
 *   K1_LOGITS=<file>     logits of EVERY fed position as float32 [rows][vocab] (<file>.r<k> with several requests)
 *   K1_RESET=0           do not reset between requests: request k continues after request k-1 (control)
 *   K1_REUSE_PREFIX=P    request 0 snapshots its cache after P tokens; a later request that starts with the same
 *                        P tokens restores it and feeds only the rest
 *   K1_FEED=<int32 file> (with no --ids) the input ids; K1_FEED_SHA256 must match when set
 *   K1_CAPTURE, K1_LOGITS_FIFO, K1_ATTN_KEYS: the test hooks (kolibri1_capture.h) */
static int k1_cmd_run(const char *dir, int release, K1Req *reqs, int nreq, int ngen, const char *feed_path) {
    int chunk = k1_env_int("K1_CHUNK", 64, 1, 1 << 16);
    int tail = k1_env_int("K1_DECODE_TAIL", 0, 0, 1 << 30);
    int cap = k1_env_int("K1_CAP", 0, 0, 1 << 20);
    int reset = k1_env_int("K1_RESET", 1, 0, 1);
    int reuse = k1_env_int("K1_REUSE_PREFIX", 0, 0, 1 << 30);
    int total = 0, longest = 0;
    for (int k = 0; k < nreq; k++) {
        if (reqs[k].n < 1) k1_die("--ids: request %d has no tokens", k);
        total += reqs[k].n + (ngen > 0 ? ngen - 1 : 0);
        if (reqs[k].n > longest) longest = reqs[k].n;
    }
    int need = reset ? longest + (ngen > 0 ? ngen - 1 : 0) : total;
    /* K1_CTX wins; otherwise CTX, the variable `coli --ctx` and the registry's context_env set (unset: no change) */
    const char *ctx_env = getenv("K1_CTX");
    int ctx = k1_env_int("K1_CTX", need, 1, 1 << 30);
    if (!(ctx_env && *ctx_env) && getenv("CTX") && *getenv("CTX")) ctx = k1_env_int("CTX", need, 1, 1 << 30);
    if (ctx < need) k1_die("K1_CTX/CTX %d is smaller than the %d positions this run feeds", ctx, need);
    const char *lp = getenv("K1_LOGITS"), *cp = getenv("K1_CAPTURE"), *fp = getenv("K1_LOGITS_FIFO"), *akp = getenv("K1_ATTN_KEYS"), *rpp = getenv("K1_REPLAY_IDS");
    if (cp && *cp && (nreq != 1 || ngen != 0)) k1_die("K1_CAPTURE needs exactly one request and --ngen 0 (teacher-forced input only)");
    if (fp && *fp && !(cp && *cp)) k1_die("K1_LOGITS_FIFO needs K1_CAPTURE (the comparator pairs them)");
    if (feed_path) {
        const char *want = getenv("K1_FEED_SHA256"); char have[65];
        if (k1_sha_file(feed_path, have)) k1_die("K1_FEED=%s: cannot read", feed_path);
        if (want && *want && strcmp(want, have)) k1_die("K1_FEED sha256 %s differs from K1_FEED_SHA256 %s", have, want);
    }
    double t0 = k1_now();
    K1Eng E;
    k1_engine_open(&E, dir, release, ctx, chunk, cap, 1);       /* a malformed checkpoint dies here, before any forward */
    double load_s = k1_now() - t0;
    if (akp && *akp) {
        k1_ak = fopen(akp, "w");
        if (!k1_ak) { perror(akp); return 1; }
        fprintf(k1_ak, "# K1_ATTN_KEYS v1 window %d ctx %d layers %d full", E.c.window, ctx, E.c.n_layers);
        for (int l = 0; l < E.c.n_layers; l++) fprintf(k1_ak, " %d", E.c.full[l]);
        fprintf(k1_ak, "\n");
    }
    k1cap.fifo_fd = -1;
    if (cp && *cp) {
        k1_cap_open(cp, reqs[0].n, E.c.n_layers, E.c.hidden, E.c.n_experts, E.c.topk);
        if (fp && *fp) k1_cap_fifo_open(fp);
    }
    if (rpp && *rpp) {
        if (!(cp && *cp) || !feed_path) k1_die("K1_REPLAY_IDS needs K1_CAPTURE and K1_FEED (the schedule is bound to the verified input)");
        char fh[65];
        if (k1_sha_file(feed_path, fh)) k1_die("K1_FEED=%s: cannot read", feed_path);
        k1_replay_open(rpp, reqs[0].n, E.c.n_layers, E.c.topk, E.c.n_experts, fh);
    } else if (getenv("K1_TEST_MUT_NATSEL") && *getenv("K1_TEST_MUT_NATSEL")) k1_die("K1_TEST_MUT_NATSEL needs K1_REPLAY_IDS");
    int V = E.c.vocab;
    int *gen = k1_malloc((size_t)(ngen + 1) * sizeof(int), "generated");
    K1Feed F = { chunk, tail, 0, NULL, NULL };
    int base = 0;
    for (int k = 0; k < nreq; k++) {
        const K1Req *R = &reqs[k];
        FILE *lf = NULL;
        if (lp && *lp) {
            char path[4300];
            if (nreq > 1) snprintf(path, sizeof(path), "%s.r%d", lp, k); else snprintf(path, sizeof(path), "%s", lp);
            lf = fopen(path, "wb");
            if (!lf) { perror(path); return 1; }
        }
        F.lf = lf;
        int from = 0, reused = 0;
        if (k > 0 && reset) { k1_reset(&E); base = 0; }
        if (k > 0 && reset && reuse && F.snap && R->n > reuse && !memcmp(R->ids, reqs[0].ids, (size_t)reuse * sizeof(int))) {
            k1_snap_restore(&E, F.snap); from = reuse; reused = reuse;
        }
        if (k1_ak) fprintf(k1_ak, "Q %d %d %d\n", k, base + from, R->n);
        F.snap_at = (k == 0 && reuse) ? reuse : 0;
        int nextp = 0;
        int last = k1_feed(&E, R->ids, R->n, from, base, &F, &nextp);
        int ng = 0;
        for (int g = 0; g < ngen; g++) {
            gen[ng++] = last;
            if (g == ngen - 1) break;
            int rows = k1_forward(&E, &last, 1, nextp, lf != NULL);
            if (lf && fwrite(E.ws.logit, sizeof(float) * (size_t)V, (size_t)rows, lf) != (size_t)rows) k1_die("short write to K1_LOGITS");
            last = k1_argmax(E.ws.logit + (size_t)(rows - 1) * V, V);
            nextp++;
        }
        k1_ak_dump_cache(&E, k);
        if (lf) fclose(lf);
        if (nreq > 1) printf("generated[%d]:", k);
        else printf("generated:");
        for (int i = 0; i < ng; i++) printf(" %d", gen[i]);
        printf("\n");
        if (nreq > 1) printf("[kolibri1] request %d: %d tokens fed (positions %d..%d)%s\n", k, R->n - from, base + from, base + R->n - 1,
                             reused ? ", prefix restored from the snapshot" : "");
        if (!reset) base = nextp;
    }
    printf("[kolibri1] %d request(s), chunk %d, decode tail %d, %.2f s, peak RSS %.3f GB\n", nreq, chunk, tail, k1_now() - t0, k1_rss_gb());
    if (k1_ak) fclose(k1_ak);
    if (k1cap.on) {
        static char meta[16000];
        k1_engine_meta(meta, sizeof(meta), dir, feed_path, reqs[0].n, &E, chunk, tail, ctx, cap, k1_now() - t0, load_s, release);
        if (k1rp.on) {
            char rm[2000];
            k1_replay_finish(k1cap.dir, rm, sizeof(rm));
            size_t ml = strlen(meta);
            if (ml + strlen(rm) + 3 >= sizeof(meta)) k1_die("ENGINE.json metadata buffer is full");
            snprintf(meta + ml, sizeof(meta) - ml, ",%s", rm);
        }
        k1_cap_finish(meta);
    }
    k1_engine_close(&E);
    free(gen);
    return 0;
}

/* ------------------------------------------------------------------- probes ---- */

/* K1_PROBE=layer|ops, K1_PROBE_IN=<dir>, K1_PROBE_OUT=<dir>[, K1_PROBE_LAYER=l][, K1_PROBE_T=rows].
 * Files are raw little-endian float32 (int32 for sel_ids) named L<l>.<tensor>.f32 / .i32, rows
 * of T positions 0..T-1 (position = row index). An external comparator writes the
 * inputs from the reference implementation and reads the outputs; nothing here knows about it.
 *   layer  L<l>.x_prev                -> the whole decoder layer: attn_pre_o attn_o post_attn_norm
 *          ffn_inp x2 router_logits sel_ids sel_w routed shared moe_sum post_ffn x_out
 *   ops    every operator on ITS OWN input from the oracle (not the engine's previous stage):
 *          x_prev->xin; xin->qkv; qkv->q_n,k_n; q_n,k_n->q_a,k_a; q_a,k_a,v->attn_pre_o;
 *          attn_pre_o->attn_o; attn_o->post_attn_norm; post_attn_norm,x_prev->ffn_inp,x2;
 *          x2->router_logits; router_logits->sel_ids,sel_w; x2,sel_ids,sel_w->routed; x2->shared;
 *          shared,routed->moe_sum; moe_sum->post_ffn; ffn_inp,post_ffn->x_out
 *   K1_PROBE_TAILS=1 (ops mode, no layer runs): ids.i32->embed; final_in->final_norm;
 *   final_norm -> logits_tail (the last min(T,128) rows). K1_PROBE_LAYER=l runs one layer. */
/* is_int: 0 = fp32 (.f32), 1 = int32 (.i32), 2 = fp64 (.f64; the residual-stream tensors x_prev, ffn_inp, x_out, final_in are native FP64 files and a
 * file of the other width is simply not found: an .f32 residual is never read as FP64 or converted) */
static const char *k1_probe_sfx(int is_int) { return is_int == 2 ? "f64" : is_int ? "i32" : "f32"; }
static size_t k1_probe_w(int is_int) { return is_int == 2 ? 8 : 4; }

static void *k1_probe_read(const char *dir, int l, const char *name, size_t count, int is_int) {
    char p[4096];
    if (l >= 0) snprintf(p, sizeof(p), "%s/L%d.%s.%s", dir, l, name, k1_probe_sfx(is_int));
    else snprintf(p, sizeof(p), "%s/%s.%s", dir, name, k1_probe_sfx(is_int));
    FILE *f = fopen(p, "rb");
    if (!f) k1_die("probe input %s is missing", p);
    size_t w = k1_probe_w(is_int);
    void *a = k1_malloc(count * w, p);
    if (fread(a, w, count, f) != count) k1_die("probe input %s is shorter than %zu values", p, count);
    fclose(f);
    return a;
}

static void k1_probe_write(const char *dir, int l, const char *name, const void *data, size_t count, int is_int) {
    char p[4096];
    if (l >= 0) snprintf(p, sizeof(p), "%s/L%d.%s.%s", dir, l, name, k1_probe_sfx(is_int));
    else snprintf(p, sizeof(p), "%s/%s.%s", dir, name, k1_probe_sfx(is_int));
    FILE *f = fopen(p, "wb");
    if (!f) k1_die("cannot write probe output %s", p);
    if (fwrite(data, k1_probe_w(is_int), count, f) != count) k1_die("short write to %s", p);
    fclose(f);
}

typedef struct { const char *name; size_t width; int is_int; unsigned char *buf; } K1ProbeOut;

static void k1_probe_layer_mode(K1Eng *E, int l, const char *in, const char *out, int T, int chunk) {
    const K1Cfg *c = &E->c; K1Ws *W = &E->ws;
    size_t H = c->hidden, qw = (size_t)c->heads * c->head_dim, K = c->topk, EX = c->n_experts;
    double *xp = k1_probe_read(in, l, "x_prev", (size_t)T * H, 2);
    K1ProbeOut o[] = {
        { "attn_pre_o", qw, 0, NULL }, { "attn_o", H, 0, NULL }, { "post_attn_norm", H, 0, NULL }, { "ffn_inp", H, 2, NULL }, { "x2", H, 0, NULL },
        { "router_logits", EX, 0, NULL }, { "sel_ids", K, 1, NULL }, { "sel_w", K, 0, NULL }, { "routed", H, 0, NULL }, { "shared", H, 0, NULL },
        { "moe_sum", H, 0, NULL }, { "post_ffn", H, 0, NULL }, { "x_out", H, 2, NULL },
    };
    int no = (int)(sizeof(o) / sizeof(o[0]));
    for (int i = 0; i < no; i++) o[i].buf = k1_malloc((size_t)T * o[i].width * k1_probe_w(o[i].is_int), o[i].name);
    for (int p0 = 0; p0 < T; p0 += chunk) {
        int n = T - p0 < chunk ? T - p0 : chunk;
        k1_layer(E, l, xp + (size_t)p0 * H, n, p0);
        const void *src[] = { W->ao, W->o, W->a, W->ffn, W->x2, W->z, W->ids, W->w, W->routed, W->shared, W->msum, W->f, W->xo };
        for (int i = 0; i < no; i++) memcpy(o[i].buf + (size_t)p0 * o[i].width * k1_probe_w(o[i].is_int), src[i], (size_t)n * o[i].width * k1_probe_w(o[i].is_int));
    }
    for (int i = 0; i < no; i++) { k1_probe_write(out, l, o[i].name, o[i].buf, (size_t)T * o[i].width, o[i].is_int); free(o[i].buf); }
    free(xp);
}

static void k1_probe_ops_mode(K1Eng *E, int l, const char *in, const char *out, int T, int chunk) {
    const K1Cfg *c = &E->c; K1Layer *L = &E->L[l]; K1Ws *W = &E->ws;
    size_t H = c->hidden, qw = (size_t)c->heads * c->head_dim, kw = (size_t)c->kv_heads * c->head_dim, K = c->topk, EX = c->n_experts;
    size_t qkvw = qw + 2 * kw, Tz = (size_t)T;
#define RD(var, name, w, isint) float *var = k1_probe_read(in, l, name, Tz * (w), isint)
#define NEWF(var, w) float *var = k1_malloc(Tz * (w) * 4, #var)
#define RD64(var, name, w) double *var = k1_probe_read(in, l, name, Tz * (w), 2)       /* FP64 residual-stream tensors */
#define NEWD(var, w) double *var = k1_malloc(Tz * (w) * 8, #var)
    RD64(x_prev, "x_prev", H); RD(xin_i, "xin", H, 0); RD(qkv_i, "qkv", qkvw, 0);
    RD(qn_i, "q_n", qw, 0); RD(kn_i, "k_n", kw, 0); RD(qa_i, "q_a", qw, 0); RD(ka_i, "k_a", kw, 0); RD(v_i, "v", kw, 0);
    RD(apo_i, "attn_pre_o", qw, 0); RD(ao_i, "attn_o", H, 0); RD(pan_i, "post_attn_norm", H, 0);
    RD(x2_i, "x2", H, 0); RD(z_i, "router_logits", EX, 0); RD(sid_i, "sel_ids", K, 1); RD(sw_i, "sel_w", K, 0);
    RD(rt_i, "routed", H, 0); RD(sh_i, "shared", H, 0); RD(ms_i, "moe_sum", H, 0); RD(pf_i, "post_ffn", H, 0); RD64(ffn_i, "ffn_inp", H);
    NEWF(xin_o, H); NEWF(qkv_o, qkvw); NEWF(qn_o, qw); NEWF(kn_o, kw); NEWF(qa_o, qw); NEWF(ka_o, kw);
    NEWF(apo_o, qw); NEWF(ao_o, H); NEWF(pan_o, H); NEWD(ffn_o, H); NEWF(x2_o, H); NEWF(z_o, EX);
    NEWF(sid_o, K); NEWF(sw_o, K); NEWF(rt_o, H); NEWF(sh_o, H); NEWF(ms_o, H); NEWF(pf_o, H); NEWD(xo_o, H);
    for (int p0 = 0; p0 < T; p0 += chunk) {
        int n = T - p0 < chunk ? T - p0 : chunk;
        size_t r0 = (size_t)p0, nn = (size_t)n;
        /* 1 */ k1_rows_rms_d(xin_o + r0 * H, x_prev + r0 * H, L->ln_in, n, (int)H, c->eps);
        /* 2 */ k1_op_qkv(E, l, xin_i + r0 * H, n, W->q, W->k, W->v);
        for (size_t r = 0; r < nn; r++) {
            memcpy(qkv_o + (r0 + r) * qkvw, W->q + r * qw, qw * 4);
            memcpy(qkv_o + (r0 + r) * qkvw + qw, W->k + r * kw, kw * 4);
            memcpy(qkv_o + (r0 + r) * qkvw + qw + kw, W->v + r * kw, kw * 4);
        }
        /* 3 */ for (size_t r = 0; r < nn; r++) {
            memcpy(W->q + r * qw, qkv_i + (r0 + r) * qkvw, qw * 4);
            memcpy(W->k + r * kw, qkv_i + (r0 + r) * qkvw + qw, kw * 4);
        }
        k1_op_qknorm(E, l, W->q, W->k, n);
        memcpy(qn_o + r0 * qw, W->q, nn * qw * 4); memcpy(kn_o + r0 * kw, W->k, nn * kw * 4);
        /* 4 */ memcpy(W->q, qn_i + r0 * qw, nn * qw * 4); memcpy(W->k, kn_i + r0 * kw, nn * kw * 4);
        if (!c->full[l]) k1_op_rope(E, W->q, W->k, n, p0);
        memcpy(qa_o + r0 * qw, W->q, nn * qw * 4); memcpy(ka_o + r0 * kw, W->k, nn * kw * 4);
        /* 5 */ k1_attention(E, l, qa_i + r0 * qw, ka_i + r0 * kw, v_i + r0 * kw, n, p0, apo_o + r0 * qw);
        /* 6 */ k1_matmul_fp8(ao_o + r0 * H, apo_i + r0 * qw, L->o.q, L->o.g, n, (int)qw, (int)H);
        /* 7 */ k1_rows_rms(pan_o + r0 * H, ao_i + r0 * H, L->ln_post_attn, n, (int)H, c->eps);
        /* 8 */ for (size_t i = 0; i < nn * H; i++) ffn_o[r0 * H + i] = x_prev[r0 * H + i] + (double)pan_i[r0 * H + i];
        k1_rows_rms_d(x2_o + r0 * H, ffn_o + r0 * H, L->ln_pre_ffn, n, (int)H, c->eps);
        /* 9 */ k1_router_logits(z_o + r0 * EX, x2_i + r0 * H, L->gate, n, (int)H, (int)EX);
        /* 10 */ k1_op_route(E, l, z_i + r0 * EX, n, (int *)(sid_o + r0 * K), sw_o + r0 * K);
        /* 11 */ k1_op_routed(E, l, x2_i + r0 * H, (const int *)(sid_i + r0 * K), sw_i + r0 * K, n, rt_o + r0 * H, NULL);
        /* 12 */ k1_op_shared(E, l, x2_i + r0 * H, n, sh_o + r0 * H);
        /* 13 */ /* This standalone probe's op 13 consumes the declared FP32 routed input, NOT production's FP64 routed sum
         * (ws.routed64 in k1_layer); the sum of two fp32 values is the same single rounding either way, but the information lost when routed was
         * rounded cannot be recovered. The combined routed+shared operation is therefore qualified through the production path (K1_PROBE=layer and
         * K1_CAPTURE both run k1_layer), not through K1_PROBE=ops op 13. */
        for (size_t i = 0; i < nn * H; i++) ms_o[r0 * H + i] = sh_i[r0 * H + i] + rt_i[r0 * H + i];
        /* 14 */ k1_rows_rms(pf_o + r0 * H, ms_i + r0 * H, L->ln_post_ffn, n, (int)H, c->eps);
        /* 15 */ for (size_t i = 0; i < nn * H; i++) xo_o[r0 * H + i] = ffn_i[r0 * H + i] + (double)pf_i[r0 * H + i];
    }
#define WR(var, name, w, isint) k1_probe_write(out, l, name, var, Tz * (w), isint)
    WR(xin_o, "xin", H, 0); WR(qkv_o, "qkv", qkvw, 0); WR(qn_o, "q_n", qw, 0); WR(kn_o, "k_n", kw, 0);
    WR(qa_o, "q_a", qw, 0); WR(ka_o, "k_a", kw, 0); WR(apo_o, "attn_pre_o", qw, 0); WR(ao_o, "attn_o", H, 0);
    WR(pan_o, "post_attn_norm", H, 0); WR(ffn_o, "ffn_inp", H, 2); WR(x2_o, "x2", H, 0); WR(z_o, "router_logits", EX, 0);
    WR(sid_o, "sel_ids", K, 1); WR(sw_o, "sel_w", K, 0); WR(rt_o, "routed", H, 0); WR(sh_o, "shared", H, 0);
    WR(ms_o, "moe_sum", H, 0); WR(pf_o, "post_ffn", H, 0); WR(xo_o, "x_out", H, 2);
    void *all[] = { x_prev, xin_i, qkv_i, qn_i, kn_i, qa_i, ka_i, v_i, apo_i, ao_i, pan_i, x2_i, z_i, sid_i, sw_i, rt_i, sh_i, ms_i, pf_i, ffn_i,
                     xin_o, qkv_o, qn_o, kn_o, qa_o, ka_o, apo_o, ao_o, pan_o, ffn_o, x2_o, z_o, sid_o, sw_o, rt_o, sh_o, ms_o, pf_o, xo_o };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) free(all[i]);
#undef RD64
#undef NEWD
#undef RD
#undef NEWF
#undef WR
}

static int k1_cmd_probe(const char *dir, int release, const char *mode) {
    const char *in = getenv("K1_PROBE_IN"), *out = getenv("K1_PROBE_OUT");
    if (!in || !out) k1_die("K1_PROBE needs K1_PROBE_IN and K1_PROBE_OUT");
    int isops = !strcmp(mode, "ops");
    if (!isops && strcmp(mode, "layer")) k1_die("K1_PROBE=%s: expected layer or ops", mode);
    int T = k1_env_int("K1_PROBE_T", 0, 1, 1 << 24);
    if (!T) k1_die("K1_PROBE needs K1_PROBE_T (rows)");
    int chunk = k1_env_int("K1_CHUNK", 64, 1, 1 << 16);
    int cap = k1_env_int("K1_CAP", 0, 0, 1 << 20);
    int only = k1_env_int("K1_PROBE_LAYER", -1, -1, K1_MAX_LAYERS);
    int tails = k1_env_int("K1_PROBE_TAILS", 0, 0, 1);   /* 1: only embedding / final norm / head (ops mode), no layer */
    K1Eng E;
    k1_engine_open(&E, dir, release, T, chunk, cap, tails);
    const K1Cfg *c = &E.c;
    for (int l = 0; l < c->n_layers && !tails; l++) {
        if (only >= 0 && l != only) continue;
        if (isops) k1_probe_ops_mode(&E, l, in, out, T, chunk);
        else k1_probe_layer_mode(&E, l, in, out, T, chunk);
    }
    if (tails && !isops) k1_die("K1_PROBE_TAILS needs K1_PROBE=ops");
    if (tails) {
        int H = c->hidden, V = c->vocab;
        int *ids = (int *)k1_probe_read(in, -1, "ids", (size_t)T, 1);
        float *emb = k1_malloc((size_t)T * H * 4, "embed");
        for (int r = 0; r < T; r++)
            for (int j = 0; j < H; j++) emb[(size_t)r * H + j] = bf16_to_f32(E.embed[(size_t)ids[r] * H + j]);
        k1_probe_write(out, -1, "embed", emb, (size_t)T * H, 0);
        double *fin = k1_probe_read(in, -1, "final_in", (size_t)T * H, 2);
        float *hn = k1_malloc((size_t)T * H * 4, "final norm");
        k1_head(&E, fin, T, hn, NULL);
        k1_probe_write(out, -1, "final_norm", hn, (size_t)T * H, 0);
        float *hn_i = k1_probe_read(in, -1, "final_norm", (size_t)T * H, 0);
        int tail = T < 128 ? T : 128;
        float *lg = k1_malloc((size_t)tail * V * 4, "logits tail");
        k1_matmul_bf16(lg, hn_i + (size_t)(T - tail) * H, E.head, tail, H, V);
        k1_probe_write(out, -1, "logits_tail", lg, (size_t)tail * V, 0);
        free(ids); free(emb); free(fin); free(hn); free(hn_i); free(lg);
    }
    k1_engine_close(&E);
    return 0;
}

/* ------------------------------------------------------------------- serve ---- */

/* SERVE=1 (the gateway: `SNAP=<dir> SERVE=1 kolibri1 <cap>`, openai_server.py's Engine): the stdin/stdout
 * protocol of docs/serve_protocol.md, the same frames and the same conventions as mimo.c.
 *
 *   engine -> gateway  READY, CAPS vision=0, STAT; per request ACCEPT <id> <prompt tokens>, DATA <id> <n>
 *                      (the decoded bytes of one token, a frame per token), DONE <id> STAT ..., or
 *                      ERROR <id> <message> (CONTEXT_EXCEEDED, EMPTY_PROMPT, CANCELLED, NOT_FOUND,
 *                      SLOT_BUSY, NONFINITE_LOGITS, BAD_TOKEN, BAD_SLOT, LOGPROBS_UNSUPPORTED).
 *   gateway -> engine  SUBMIT <id> <slot> <bytes> <max_tokens> <temperature> <top_p> [<0>] [pin=1]\n<prompt>\n
 *                      (the prompt is the rendered chat text; this engine tokenises it), STOP <id>, CANCEL <id>.
 *
 * The wire carries temperature and top_p only (0 < top_p <= 1; a negative top_p is rejected with an ERROR frame,
 * there is no private sign encoding). Request top_k and request seed are NOT on the wire: they are ignored (the
 * gateway logs one warning per request that sends them). top_k and the seed are launch-level engine settings:
 *   K1_TOP_K     top-k cut before top-p (default 128, the vendor default; 0 = off). Vendor-stack semantics: every logit equal
 *                to the K-th largest also survives, so a tie at the cut keeps more than K candidates. The threshold
 *                and top-p are computed on the temperature-scaled FP32 logits (logit / temperature), like the vendor stack
 *   K1_SEED/SEED a fixed seed makes every request draw from the same stream (reproducible); unset: time+pid
 * The gateway fills temperature/top_p when the client sends none (0.7 / 0.9, COLI_TEMP overrides the first).
 *
 * KV and prefix reuse. One slot. Before a request the cache is either (a) kept, when the prompt begins with
 * every id the state was built from (the previous prompt and its reply), (b) restored from the snapshot taken
 * at the end of the previous PROMPT when the prompt begins with that prompt (a chat client resends the
 * transcript, but the reply is rendered back differently from the raw ids), or (c) reset. A sliding ring only
 * holds the last `window` positions, so a partial-prefix match is NOT reusable (the ring slots of the dropped
 * positions were overwritten); that is why reuse is all-or-nothing on the recorded ids. K1_SERVE_PREFIX=0
 * (or COLI_KV_PREFIX=0) turns reuse off. A reset poisons every K/V row that was written (NaN), so a stale
 * read is an error reply, never another request's number.
 *
 * Stop: the checkpoint's eos ids (config.json eos_token_id = <|im_end|> 127906) plus K1_STOP_IDS (comma list).
 * The forward is the one the other modes run (k1_forward, same chunking K1_CHUNK, default 64): a greedy
 * request is token for token the `--ids` greedy continuation.
 * Benchmark channel (K1_BENCH_TOKENS=1, off by default; K1_BENCH_FILE=<path> appends to a file instead of stderr; K1_IGNORE_EOS=1
 * needs K1_BENCH_TOKENS=1): per request one `K1BENCH start` line, one `K1BENCH prefill_end` line (prompt prefilled, first sample about to
 * be drawn), then one `K1BENCH tok` line per SAMPLED token, the stop token (eos) included, carrying its index, its id, whether it is a stop id and
 * a CLOCK_MONOTONIC nanosecond timestamp taken the instant the sampler returned, and a `K1BENCH end` line. DATA frames are decoded text
 * (empty pieces are skipped), so they are not a per-token measurement; this channel is. With K1_IGNORE_EOS=1 a stop id is NOT a stop condition
 * (it is emitted and fed like any token; K1_STOP_IDS too), so a request of max_tokens = D yields exactly D tokens, room in the context
 * permitting; the protocol's STOP / CANCEL still work. The channel only reads the clock and writes lines after the sampler returned: the
 * forward, the sampler and its RNG stream are untouched (a greedy request is token for token the default one up to the first stop id).
 * K1_SERVE_LOG=1 writes one stderr line per request. K1_TEST_SERVE_SCRIPT=<file> (test only, loud): the
 * sampled ids are replaced by the tokenised lines of the file, request k taking line k, then eos. */

static const ColiServeWireProfile k1_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 26,
    .max_extension_bytes = 0,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
    .allow_extension_bytes = 1,
};

static uint64_t k1_splitmix(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double k1_unit(uint64_t *s) { return (double)(k1_splitmix(s) >> 11) * (1.0 / 9007199254740992.0); }

typedef struct { float l; int id; } K1Cand;

/* descending by logit; ties by DESCENDING id. This is the reverse of an ascending stable sort (the vendor
 * stack sorts ascending, so among equal logits the lower id is removed first by top-p and
 * the highest id survives, "at least one" included) */
static int k1_cand_cmp(const void *a, const void *b) {
    const K1Cand *x = a, *y = b;
    if (x->l != y->l) return x->l > y->l ? -1 : 1;
    return (x->id < y->id) - (x->id > y->id);
}

/* The next token from one row of logits, or -1 when any logit is not finite (the caller answers with an error;
 * nothing is sampled from such a row). temperature <= 0: argmax, first maximum (the greedy rule).
 * Otherwise the order of the vendor stack's sampler (vLLM v0.29.0 documents it):
 *   0. temperature: s = logit / temperature in FP32 (`logits.div_(temp)` on a float32 tensor); everything below
 *      works on s, so division can create ties (1.5000001192092896 and 1.500000238418579 at T=1.5 are equal);
 *   1. top-k: the threshold is the K-th largest logit and EVERY logit >= it survives (`logits_sort < thr` is
 *      masked), so ties at the K-th value keep more than K candidates;
 *   2. top-p over the survivors, smallest first: a token is removed while the cumulative mass up to and
 *      including it is <= 1 - p; the largest token is never removed (equal logits: the highest id survives);
 *   3. one draw from the rest (our own RNG; the vendor stack's exponential race draws differently, same distribution).
 * `cand` has room for V entries. */
static int k1_sample(const float *lg, int V, float temp, float top_p, int top_k, uint64_t *rng, K1Cand *cand) {
    for (int i = 0; i < V; i++) if (!isfinite(lg[i])) return -1;
    if (!(temp >= 1e-5f)) return k1_argmax(lg, V);     /* below this temperature a request is greedy, as in the vendor stack */
    int k = (top_k > 0 && top_k < V) ? top_k : V, n = 0;
    float thr = -INFINITY;
    for (int i = 0; i < V; i++) if (!isfinite(lg[i] / temp)) return -1;   /* FP32 overflow of the division: an error, not a token */
    if (k < V) {
        for (int i = 0; i < V; i++) {                 /* the k largest SCALED values, descending; cand[k-1] is the threshold */
            float x = lg[i] / temp;                   /* FP32 division, as `logits.div_(temp)` */
            if (n == k && !(x > cand[n - 1].l)) continue;
            int j = n < k ? n++ : n - 1;
            while (j > 0 && cand[j - 1].l < x) { cand[j] = cand[j - 1]; j--; }
            cand[j].l = x; cand[j].id = i;
        }
        thr = cand[k - 1].l;
    }
    n = 0;
    for (int i = 0; i < V; i++) { float x = lg[i] / temp; if (x >= thr) { cand[n].l = x; cand[n].id = i; n++; } }
    qsort(cand, (size_t)n, sizeof(K1Cand), k1_cand_cmp);
    double top = cand[0].l, total = 0.0;
#define K1_W(i) exp((double)cand[i].l - top)          /* cand[].l is already logit / temperature */
    for (int i = 0; i < n; i++) total += K1_W(i);
    double cut = (top_p > 0.0f && top_p < 1.0f) ? (double)top_p * total : total, kept = 0.0;
    int m = 0;
    while (m < n && kept < cut) kept += K1_W(m++);
    double draw = k1_unit(rng) * kept, acc = 0.0;
    int pick = cand[m - 1].id;
    for (int i = 0; i < m; i++) { acc += K1_W(i); if (acc >= draw) { pick = cand[i].id; break; } }
#undef K1_W
    return pick;
}

/* a reset that touches only what was written: the K/V rows [0, hi) of the full layers and every ring slot */
static void k1_reset_used(K1Eng *E, int hi) {
    const K1Cfg *c = &E->c;
    size_t kw = (size_t)c->kv_heads * c->head_dim;
    for (int l = 0; l < c->n_layers; l++) {
        int rows = c->full[l] ? (hi < E->rows[l] ? hi : E->rows[l]) : E->rows[l];
        size_t n = (size_t)rows * kw;
        for (size_t i = 0; i < n; i++) { E->kc[l][i] = NAN; E->vc[l][i] = NAN; }
        if (E->rpos[l]) for (int i = 0; i < E->rows[l]; i++) E->rpos[l][i] = -1;
    }
}

static void k1_snap_free(K1Snap *S, int n_layers) {
    if (!S) return;
    for (int l = 0; l < n_layers; l++) { free(S->kc[l]); free(S->vc[l]); free(S->rpos[l]); }
    free(S->kc); free(S->vc); free(S->rpos); free(S);
}

/* ids of the test script, one array per line */
typedef struct { int **ids; int *n; int lines; } K1Script;

static void k1_script_load(K1Script *S, const char *path, Tok *T, int vocab) {
    FILE *f = fopen(path, "rb");
    if (!f) k1_die("K1_TEST_SERVE_SCRIPT=%s: %s", path, strerror(errno));
    char *line = NULL; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, f)) >= 0) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        if (!len) continue;
        char *text = k1_malloc((size_t)len + 1, "script text");
        size_t w = 0;
        for (ssize_t i = 0; i < len; i++) {            /* \n and \\ are the only escapes */
            if (line[i] == '\\' && i + 1 < len && line[i + 1] == 'n') { text[w++] = '\n'; i++; }
            else if (line[i] == '\\' && i + 1 < len && line[i + 1] == '\\') { text[w++] = '\\'; i++; }
            else text[w++] = line[i];
        }
        text[w] = 0;
        int *ids = k1_malloc((w + 1) * sizeof(int), "script ids");
        int n = tok_encode(T, text, (int)w, ids, (int)w + 1);
        for (int i = 0; i < n; i++) if (ids[i] < 0 || ids[i] >= vocab) k1_die("K1_TEST_SERVE_SCRIPT: id %d outside the vocabulary", ids[i]);
        S->ids = realloc(S->ids, (size_t)(S->lines + 1) * sizeof(int *));
        S->n = realloc(S->n, (size_t)(S->lines + 1) * sizeof(int));
        if (!S->ids || !S->n) k1_die("out of memory");
        S->ids[S->lines] = ids; S->n[S->lines] = n; S->lines++;
        free(text);
    }
    free(line);
    fclose(f);
    if (!S->lines) k1_die("K1_TEST_SERVE_SCRIPT=%s has no lines", path);
}

/* Control frames that arrive while a request runs. Returns 1 when stdin ended. */
static int k1_serve_controls(const char *id, int *cancelled, int *stopped) {
    int eof = 0;
    while (coli_serve_stdin_ready()) {
        ColiServeCommand control;
        memset(&control, 0, sizeof(control));
        ColiServeReadResult r = coli_serve_read_command(stdin, &k1_wire, &control);
        if (r == COLI_SERVE_READ_EOF || r == COLI_SERVE_READ_BAD_FRAME) { eof = 1; break; }
        if (r == COLI_SERVE_READ_NOMEM) { coli_serve_write_error(stdout, control.id, "out of memory"); eof = 1; break; }
        if (r == COLI_SERVE_READ_BAD_REQUEST) {
            if (control.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, control.id, "bad submit header");
        } else if (control.kind == COLI_SERVE_COMMAND_CANCEL && !strcmp(control.id, id)) *cancelled = 1;
        else if (control.kind == COLI_SERVE_COMMAND_STOP && !strcmp(control.id, id)) *stopped = 1;
        else if (control.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, control.id, "SLOT_BUSY");
        coli_serve_command_dispose(&control);
    }
    return eof;
}

static int k1_serve_budget(int prompt, int requested, int context) {
    if (prompt < 1 || prompt > context) return -1;
    int budget = requested > 0 ? requested : 256;
    int room = context - prompt;
    if (room == 0) return -1;
    return budget < room ? budget : room;
}

static uint64_t k1_ids_hash(const int *ids, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++) { h ^= (uint32_t)ids[i]; h *= 1099511628211ull; }
    return h;
}

/* Benchmark channel (see the SERVE header comment): never active unless K1_BENCH_TOKENS=1 */
typedef struct { int on; FILE *f; int owned; } K1Bench;

static uint64_t k1_mono_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void k1_bench_open(K1Bench *B, int *ignore_eos) {
    memset(B, 0, sizeof(*B));
    B->on = k1_env_int("K1_BENCH_TOKENS", 0, 0, 1);
    *ignore_eos = k1_env_int("K1_IGNORE_EOS", 0, 0, 1);
    const char *path = getenv("K1_BENCH_FILE");
    if (!B->on) {
        if (*ignore_eos) k1_die("K1_IGNORE_EOS=1 needs K1_BENCH_TOKENS=1 (a request that ignores the stop ids is a benchmark request, and its tokens must be recorded)");
        if (path && *path) k1_die("K1_BENCH_FILE needs K1_BENCH_TOKENS=1");
        return;
    }
    B->f = stderr;
    if (path && *path) {
        B->f = fopen(path, "a");
        if (!B->f) k1_die("K1_BENCH_FILE=%s: %s", path, strerror(errno));
        B->owned = 1;
    }
    fprintf(stderr, "[kolibri1] ******** K1_BENCH_TOKENS: per-token ids and monotonic timestamps to %s; K1_IGNORE_EOS=%d (benchmark use only) ********\n",
            B->owned ? path : "stderr", *ignore_eos);
}

static void k1_bench_line(const K1Bench *B, const char *fmt, ...) {
    if (!B->on) return;
    va_list ap; va_start(ap, fmt);
    vfprintf(B->f, fmt, ap);
    va_end(ap);
    fflush(B->f);                       /* one line per write, same discipline as the protocol's own frames */
}

/* cumulative counters at the start and at the end of a request (the harness subtracts): the engine's expert LRU (a `load` is one pread of an expert from the
 * checkpoint file: an ENGINE read, which the page cache may or may not have satisfied from RAM) and nothing else: there is no byte counter in the engine, so
 * transferred bytes are not measured here. */
static void k1_bench_counters(const K1Bench *B, const K1Eng *E, const char *req, const char *at) {
    if (!B->on) return;
    char buf[1536]; size_t n = 0;
#define K1B_AP(...) do { int w_ = snprintf(buf + n, sizeof(buf) - n, __VA_ARGS__); if (w_ > 0 && (size_t)w_ < sizeof(buf) - n) n += (size_t)w_; } while (0)
    long h = 0, ld = 0, ev = 0;
    for (int l = 0; l < E->c.n_layers; l++) { h += E->ec[l].hits; ld += E->ec[l].loads; ev += E->ec[l].evicts; }
    K1B_AP("K1BENCH counters req=%s at=%s ec_hits=%ld ec_loads=%ld ec_evicts=%ld", req, at, h, ld, ev);
#undef K1B_AP
    k1_bench_line(B, "%s\n", buf);
}

static int k1_cmd_serve(const char *dir, int release, int cap_arg) {
    int chunk = k1_env_int("K1_CHUNK", 64, 1, 1 << 16);
    int cap = cap_arg > 0 ? cap_arg : k1_env_int("K1_CAP", 0, 0, 1 << 20);
    int top_k = k1_env_int("K1_TOP_K", 128, 0, 1 << 20);
    int prefix_on = k1_env_int("K1_SERVE_PREFIX", 1, 0, 1) && k1_env_int("COLI_KV_PREFIX", 1, 0, 1);
    int log_on = k1_env_int("K1_SERVE_LOG", 0, 0, 2);
    int test_ids = k1_env_int("K1_TEST_SERVE_IDS", 0, 0, 1);
    int test_nan_req = k1_env_int("K1_TEST_SERVE_NAN_REQ", 0, 0, 1 << 30);   /* test only: request k gets a NaN logit before its 3rd token is drawn */
    static const char *const hooks[] = { "K1_CAPTURE", "K1_LOGITS_FIFO", "K1_ATTN_KEYS", "K1_REPLAY_IDS", "K1_LOGITS", "K1_FEED", "K1_PROBE" };
    for (size_t i = 0; i < sizeof(hooks) / sizeof(hooks[0]); i++)
        if (getenv(hooks[i]) && *getenv(hooks[i])) k1_die("SERVE does not run with %s (a test hook of the other modes)", hooks[i]);
    if (getenv("K1_TEST_SERVE_IDS") && *getenv("K1_TEST_SERVE_IDS") && strcmp(getenv("K1_TEST_SERVE_IDS"), "0"))
        fprintf(stderr, "[kolibri1] ******** K1_TEST_SERVE_IDS: a prompt that starts with \"ids:\" is read as token ids, for checking only ********\n");
    if (k1_env_int("K1_TEST_SERVE_NAN_REQ", 0, 0, 1 << 30))
        fprintf(stderr, "[kolibri1] ******** K1_TEST_SERVE_NAN_REQ: one request gets a NaN logit, for checking only ********\n");
    K1Bench bench; int ignore_eos = 0;
    k1_bench_open(&bench, &ignore_eos);
    int ctx = 4096;
    {   /* K1_CTX, then CTX (what `coli --ctx` sets), else 4096; never above the model's positions */
        const char *k = getenv("K1_CTX");
        if (k && *k) ctx = k1_env_int("K1_CTX", ctx, 1, 1 << 30);
        else if (getenv("CTX") && *getenv("CTX")) ctx = k1_env_int("CTX", ctx, 1, 1 << 30);
    }
    char tok_path[4200];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", dir);
    if (access(tok_path, R_OK) != 0) k1_die("SERVE needs %s", tok_path);
    K1Eng E;
    double t0 = k1_now();
    k1_engine_open(&E, dir, release, ctx, chunk, cap, 1);
    const K1Cfg *c = &E.c;
    int V = c->vocab;
    k1cap.fifo_fd = -1;
    k1_logits_soft = 1;
    Tok T;
    k1_tok_load(&T, tok_path);
    int eos[16], n_eos = coli_load_stop_ids(dir, eos, 8, NULL);
    {
        const char *extra = getenv("K1_STOP_IDS");
        while (extra && *extra && n_eos < 16) {
            char *end; long v = strtol(extra, &end, 10);
            if (end == extra || v < 0 || v >= V) k1_die("K1_STOP_IDS=%s is not a comma list of ids in [0, %d)", getenv("K1_STOP_IDS"), V);
            eos[n_eos++] = (int)v;
            extra = *end == ',' ? end + 1 : end;
            if (*end && *end != ',') k1_die("K1_STOP_IDS=%s is not a comma list of ids", getenv("K1_STOP_IDS"));
        }
    }
    if (!n_eos) fprintf(stderr, "[kolibri1] serve: the checkpoint declares no eos id; a request ends at max_tokens only\n");
    K1Script script = {0};
    if (getenv("K1_TEST_SERVE_SCRIPT") && *getenv("K1_TEST_SERVE_SCRIPT")) {
        k1_script_load(&script, getenv("K1_TEST_SERVE_SCRIPT"), &T, V);
        fprintf(stderr, "[kolibri1] ******** K1_TEST_SERVE_SCRIPT: sampled ids are REPLACED by a script (%d lines), for checking only ********\n", script.lines);
    }
    const char *seed_env = getenv("K1_SEED");
    if (!seed_env || !*seed_env) seed_env = getenv("SEED");
    int seeded = seed_env && *seed_env;
    uint64_t seed = seeded ? strtoull(seed_env, NULL, 10) : ((uint64_t)time(NULL) << 20) ^ (uint64_t)getpid();
    uint64_t rng = seed;
    fprintf(stderr, "[kolibri1] serve: ctx %d, chunk %d, top_k %d, prefix reuse %s, stop ids %d, %s seed -- loaded in %.2fs\n",
            ctx, chunk, top_k, prefix_on ? "on" : "off", n_eos, seeded ? "fixed" : "random", k1_now() - t0);
    coli_serve_stdio_init();
    coli_serve_write_ready_caps(stdout, k1_rss_gb(), "vision=0");

    K1Cand *cand = k1_malloc((size_t)V * sizeof(K1Cand), "sampler");
    int *ids = k1_malloc(((size_t)ctx + 1) * sizeof(int), "prompt ids");
    int *fed = k1_malloc(((size_t)ctx + 1) * sizeof(int), "fed ids");     /* the ids the cache holds, by position */
    int fed_len = 0, hi = 0;
    int *genlog = log_on >= 2 ? k1_malloc(((size_t)ctx + 1) * sizeof(int), "generated log") : NULL;
    K1Snap *snap = NULL; int *snap_ids = NULL;
    long served = 0;
    for (;;) {
        ColiServeCommand command;
        memset(&command, 0, sizeof(command));
        ColiServeReadResult result = coli_serve_read_command(stdin, &k1_wire, &command);
        if (result == COLI_SERVE_READ_EOF || result == COLI_SERVE_READ_BAD_FRAME) break;
        if (result == COLI_SERVE_READ_NOMEM) { coli_serve_write_error(stdout, command.id, "out of memory"); break; }
        if (result == COLI_SERVE_READ_BAD_REQUEST) {
            if (command.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, command.id, "bad submit header");
            coli_serve_command_dispose(&command);
            continue;
        }
        if (command.kind == COLI_SERVE_COMMAND_IMAGE) {
            coli_serve_write_error(stdout, command.id, "this container has no vision tower");
            coli_serve_command_dispose(&command);
            continue;
        }
        if (command.kind != COLI_SERVE_COMMAND_SUBMIT) {
            if (command.kind == COLI_SERVE_COMMAND_CANCEL) coli_serve_write_error(stdout, command.id, "NOT_FOUND");
            coli_serve_command_dispose(&command);
            continue;
        }
        double started = k1_now();
        long h0 = 0, l0 = 0;
        for (int l = 0; l < c->n_layers; l++) { h0 += E.ec[l].hits; l0 += E.ec[l].loads; }
        if (command.slot != 0) { coli_serve_write_error(stdout, command.id, "BAD_SLOT this engine has one cache slot (0)"); coli_serve_command_dispose(&command); continue; }
        if (!(command.top_p >= 0.0f)) { coli_serve_write_error(stdout, command.id, "BAD_SAMPLING top_p must be positive (a negative top_p is not a protocol extension of this engine)"); coli_serve_command_dispose(&command); continue; }
        if (command.logprobs > 0) { coli_serve_write_error(stdout, command.id, "LOGPROBS_UNSUPPORTED"); coli_serve_command_dispose(&command); continue; }
        int n_prompt;
        if (test_ids && command.payload_bytes >= 4 && !memcmp(command.payload, "ids:", 4)) {   /* test only: the prompt as token ids */
            n_prompt = 0;
            const char *q = (const char *)command.payload + 4;
            while (*q && n_prompt <= ctx) {
                while (*q == ',' || *q == ' ') q++;
                char *end; long v = strtol(q, &end, 10);
                if (end == q) break;
                ids[n_prompt++] = (int)v; q = end;
            }
        } else n_prompt = tok_encode(&T, (const char *)command.payload, (int)command.payload_bytes, ids, ctx + 1);
        int budget = k1_serve_budget(n_prompt, command.max_tokens, ctx);
        if (budget < 0) {
            char message[160];
            snprintf(message, sizeof(message), "CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d", n_prompt, command.max_tokens, ctx);
            coli_serve_write_error(stdout, command.id, n_prompt < 1 ? "EMPTY_PROMPT" : message);
            coli_serve_command_dispose(&command);
            continue;
        }
        int bad_id = -1;
        for (int i = 0; i < n_prompt && bad_id < 0; i++) if (ids[i] < 0 || ids[i] >= V) bad_id = ids[i];
        if (bad_id >= 0) {
            char message[160];
            snprintf(message, sizeof(message), "BAD_TOKEN id %d is outside the model's vocabulary [0, %d)", bad_id, V);
            coli_serve_write_error(stdout, command.id, message);
            coli_serve_command_dispose(&command);
            continue;
        }
        /* where this prompt starts: see the header comment */
        int reuse = 0; const char *how = "none";
        int live_ok = prefix_on && fed_len > 0 && fed_len < n_prompt && !memcmp(fed, ids, (size_t)fed_len * sizeof(int));
        int snap_ok = prefix_on && snap && snap->P < n_prompt && !memcmp(snap_ids, ids, (size_t)snap->P * sizeof(int));
        if (live_ok && (!snap_ok || fed_len >= snap->P)) { reuse = fed_len; how = "live"; }
        else if (snap_ok) {
            k1_reset_used(&E, hi);
            k1_snap_restore(&E, snap);
            reuse = snap->P; how = "snapshot";
            memcpy(fed, snap_ids, (size_t)reuse * sizeof(int));
            fed_len = reuse; hi = reuse;
        } else {
            k1_reset_used(&E, hi);
            fed_len = 0; hi = 0;
        }
        coli_serve_write_accept(stdout, command.id, n_prompt);
        k1_bench_line(&bench, "K1BENCH start req=%s prompt_tokens=%d reused=%d budget=%d t_ns=%llu\n", command.id, n_prompt, reuse, budget, (unsigned long long)k1_mono_ns());
        k1_bench_counters(&bench, &E, command.id, "start");
        if (seeded) rng = seed;
        k1_logits_bad[0] = 0;
        int cancelled = 0, stopped = 0, eof = 0, failed = 0;
        /* prefill the rest of the prompt, chunk by chunk, at positions reuse.. */
        for (int i = reuse; i < n_prompt && !failed; ) {
            int m = n_prompt - i < chunk ? n_prompt - i : chunk;
            k1_forward(&E, ids + i, m, i, 0);
            memcpy(fed + i, ids + i, (size_t)m * sizeof(int));
            i += m; fed_len = i; if (i > hi) hi = i;
            if (k1_logits_bad[0]) { failed = 1; break; }
            if (i < n_prompt) {
                eof |= k1_serve_controls(command.id, &cancelled, &stopped);
                if (cancelled || stopped || eof) break;
            }
        }
        if (!failed && !cancelled && !stopped && !eof && prefix_on) {     /* the photo of the state at the end of the prompt */
            k1_snap_free(snap, c->n_layers);
            free(snap_ids);
            snap = k1_snap_take(&E, n_prompt);
            snap_ids = k1_malloc((size_t)n_prompt * sizeof(int), "snapshot ids");
            memcpy(snap_ids, ids, (size_t)n_prompt * sizeof(int));
        }
        k1_bench_line(&bench, "K1BENCH prefill_end req=%s failed=%d cancelled=%d t_ns=%llu\n", command.id, failed, cancelled || stopped || eof, (unsigned long long)k1_mono_ns());
        int emitted = 0, limited = 1;
        const int *scr = script.lines ? script.ids[served % script.lines] : NULL;
        int scr_n = script.lines ? script.n[served % script.lines] : 0;
        char piece[512];
        while (!failed && !cancelled && !stopped && !eof && emitted < budget) {
            int token;
            if (scr) token = emitted < scr_n ? scr[emitted] : (n_eos ? eos[0] : 0);
            else {
                if (test_nan_req && served == test_nan_req - 1 && emitted == 2) E.ws.logit[V / 2] = NAN;
                token = k1_sample(E.ws.logit, V, command.temperature, command.top_p, top_k, &rng, cand);
            }
            const uint64_t t_sampled = bench.on ? k1_mono_ns() : 0;      /* the instant the sampler returned */
            if (token < 0) {
                snprintf(k1_logits_bad, sizeof(k1_logits_bad), "NONFINITE_LOGITS a non-finite logit at the sampler (position %d)", fed_len - 1);
                failed = 1; break;
            }
            int stop = 0;
            for (int i = 0; i < n_eos; i++) if (token == eos[i]) stop = 1;
            k1_bench_line(&bench, "K1BENCH tok req=%s i=%d id=%d stop_id=%d t_ns=%llu\n", command.id, emitted, token, stop, (unsigned long long)t_sampled);
            if (ignore_eos) stop = 0;                   /* benchmark mode: a stop id is an ordinary token (recorded above) */
            if (stop) { limited = 0; break; }
            int written = tok_decode(&T, &token, 1, piece, (int)sizeof(piece));
            if (written > 0) coli_serve_write_data(stdout, command.id, piece, (size_t)written);
            if (genlog) genlog[emitted] = token;
            emitted++;
            eof |= k1_serve_controls(command.id, &cancelled, &stopped);
            if (stopped) limited = 0;
            if (cancelled || stopped || eof || emitted >= budget) break;
            k1_forward(&E, &token, 1, fed_len, 0);
            fed[fed_len++] = token; if (fed_len > hi) hi = fed_len;
            if (k1_logits_bad[0]) { failed = 1; break; }
        }
        k1_bench_line(&bench, "K1BENCH end req=%s emitted=%d limited=%d status=%s t_ns=%llu\n", command.id, emitted, limited,
                      failed ? "error" : cancelled ? "cancelled" : "ok", (unsigned long long)k1_mono_ns());
        k1_bench_counters(&bench, &E, command.id, "end");
        if (failed) {
            coli_serve_write_error(stdout, command.id, k1_logits_bad[0] ? k1_logits_bad : "engine failure");
            /* the state is not trusted any more: poison it and forget both records */
            k1_reset_used(&E, hi); fed_len = 0; hi = 0;
            k1_snap_free(snap, c->n_layers); free(snap_ids); snap = NULL; snap_ids = NULL;
        } else if (cancelled) coli_serve_write_error(stdout, command.id, "CANCELLED");
        else {
            double wall = k1_now() - started;
            long h1 = 0, l1 = 0;
            for (int l = 0; l < c->n_layers; l++) { h1 += E.ec[l].hits; l1 += E.ec[l].loads; }
            long th = h1 - h0, tl = l1 - l0;
            ColiServeDone done = { emitted, wall > 0 ? emitted / wall : 0.0,
                                   (th + tl) ? 100.0 * th / (double)(th + tl) : 0.0, k1_rss_gb(), n_prompt, limited };
            coli_serve_write_done(stdout, command.id, &done);
        }
        if (log_on)
            fprintf(stderr, "[kolibri1] serve req %s: prompt_tokens=%d reused=%d(%s) idhash=%016llx emitted=%d limited=%d status=%s temperature=%.8g top_p=%.8g top_k=%d\n",
                    command.id, n_prompt, reuse, how, (unsigned long long)k1_ids_hash(ids, n_prompt), emitted, limited,
                    failed ? "error" : cancelled ? "cancelled" : "ok", (double)command.temperature, (double)command.top_p, top_k);
        if (genlog) {
            fprintf(stderr, "[kolibri1] serve req %s generated:", command.id);
            for (int i = 0; i < emitted; i++) fprintf(stderr, " %d", genlog[i]);
            fprintf(stderr, "\n");
        }
        served++;
        coli_serve_command_dispose(&command);
        if (eof) break;
    }
    k1_snap_free(snap, c->n_layers); free(snap_ids);
    free(cand); free(ids); free(fed); free(genlog);
    if (bench.owned) fclose(bench.f);
    k1_engine_close(&E);
    return 0;
}

/* ------------------------------------------------------------------- main ---- */

#ifndef KOLIBRI1_NO_MAIN
static int k1_is_dir(const char *p) { struct stat st; return stat(p, &st) == 0 && S_ISDIR(st.st_mode); }

static void k1_usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s <model dir> --validate [--release] [--expect-tensors N] [--expect-split N]\n"
        "       %s --tokenize <model dir | tokenizer.json>   (text on stdin -> ids)\n"
        "       %s <model dir> --ids <t0,t1,...> [--ngen N] [--release]   (greedy; K1_CHUNK, K1_CAP, K1_CTX, K1_LOGITS)\n"
        "       SNAP=<model dir> SERVE=1 %s <cap>   (the gateway's serve loop; K1_CTX/CTX, K1_CHUNK, K1_TOP_K, K1_SEED)\n"
        "       K1_PROBE=layer|ops K1_PROBE_IN=<dir> K1_PROBE_OUT=<dir> K1_PROBE_T=<rows> %s <model dir>   (operator/layer probes)\n",
        argv0, argv0, argv0, argv0, argv0);
}

static int k1_cmd_validate(const char *dir, int release, long expect_tensors, long expect_split) {
    double t0 = k1_now();
    int fp = k1_fp_env_check();
    if (fp) k1_die("floating-point environment flushes subnormals (code %d)", fp);
    K1Model m;
    k1_open(&m, dir, release);
    const K1Cfg *c = &m.c;
    int nfull = 0;
    for (int i = 0; i < c->n_layers; i++) nfull += c->full[i];
    printf("[kolibri1] validate OK%s: %d layers (%d sliding window %d, %d full/NoPE), hidden %d, heads %d/%d x %d, "
           "%d experts top-%d, moe %d shared %d, vocab %d\n",
           release ? " (release policy)" : "", c->n_layers, c->n_layers - nfull, c->window, nfull, c->hidden,
           c->heads, c->kv_heads, c->head_dim, c->n_experts, c->topk, c->moe_inter, c->shared_inter, c->vocab);
    printf("[kolibri1] tensors %d in %d shards; %ld scale grids read, all finite (|scale| %.4g..%.4g, %ld zero, %ld negative)\n",
           m.n_tensors, m.n_shards, m.n_scale_tensors, m.scale_min, m.scale_max, m.scale_zero, m.scale_negative);
    printf("[kolibri1] experts %d: contiguous (1 read) %ld, 2 runs %ld, 3 runs %ld; split across shards %ld "
           "(gate in previous shard %ld, up in next shard %ld, other %ld)\n",
           c->n_layers * c->n_experts, m.runs1, m.runs2, m.runs3, m.split, m.gate_prev, m.up_next, m.split_other);
    printf("[kolibri1] scale bank %.1f MB (%d floats per matrix); peak RSS %.3f GB; %.2f s\n",
           (double)c->n_layers * c->n_experts * 3 * m.nsc * 4 / 1e6, m.nsc, k1_rss_gb(), k1_now() - t0);
    int rc = 0;
    if (expect_tensors >= 0 && m.n_tensors != expect_tensors) {
        fprintf(stderr, "[kolibri1] --expect-tensors %ld, found %d -- FAIL\n", expect_tensors, m.n_tensors); rc = 1; }
    if (expect_split >= 0 && m.split != expect_split) {
        fprintf(stderr, "[kolibri1] --expect-split %ld, found %ld -- FAIL\n", expect_split, m.split); rc = 1; }
    k1_close(&m);
    return rc;
}

static int k1_cmd_tokenize(const char *path) {
    char tp[4096];
    if (k1_is_dir(path)) snprintf(tp, sizeof(tp), "%s/tokenizer.json", path); else snprintf(tp, sizeof(tp), "%s", path);
    Tok T;
    k1_tok_load(&T, tp);
    size_t cap = 1 << 16, n = 0;
    char *text = k1_malloc(cap, "stdin");
    size_t r;
    while ((r = fread(text + n, 1, cap - n, stdin)) > 0) {
        n += r;
        if (n == cap) { cap *= 2; text = realloc(text, cap); if (!text) k1_die("out of memory reading stdin"); }
    }
    int *ids = k1_malloc((n + 1) * sizeof(int), "ids");
    int cnt = tok_encode(&T, text, (int)n, ids, (int)n + 1);
    for (int i = 0; i < cnt; i++) printf(i ? " %d" : "%d", ids[i]);
    printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    coli_omp_tune_threads("kolibri1");
    k1_cap_invalidate_early();      /* before any fallible step: a stale completion marker must not outlive a failed run */
    const char *dir = getenv("SNAP"), *tok_path = NULL;
    int validate = 0, release = 0, ngen = 0, nids = 0, serve_cap = 0;
    int *ids = NULL;
    K1Req *reqs = NULL; int nreq = 0;
    long expect_tensors = -1, expect_split = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--validate")) validate = 1;
        else if (!strcmp(argv[i], "--release")) release = 1;
        else if (!strcmp(argv[i], "--expect-tensors") && i + 1 < argc) expect_tensors = coli_arg_int(argv[++i], "--expect-tensors");
        else if (!strcmp(argv[i], "--expect-split") && i + 1 < argc) expect_split = coli_arg_int(argv[++i], "--expect-split");
        else if (!strcmp(argv[i], "--tokenize") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) reqs = k1_parse_reqs(argv[++i], &nreq);
        else if (!strcmp(argv[i], "--ngen") && i + 1 < argc) ngen = (int)coli_arg_int(argv[++i], "--ngen");
        else if (k1_is_dir(argv[i])) dir = argv[i];
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') serve_cap = (int)coli_arg_int(argv[i], "cache/layer");   /* the gateway's argv: SNAP=<dir> SERVE=1 kolibri1 <cap> */
        else { k1_usage(argv[0]); return 2; }
    }
    if (tok_path) return k1_cmd_tokenize(tok_path);
    if (validate && dir) return k1_cmd_validate(dir, release, expect_tensors, expect_split);
    if (k1_env_int("SERVE", 0, 0, 1) && dir) return k1_cmd_serve(dir, release || k1_env_int("K1_RELEASE", 0, 0, 1), serve_cap);
    const char *probe = getenv("K1_PROBE");
    if (probe && *probe && dir) return k1_cmd_probe(dir, release, probe);
    const char *feed = getenv("K1_FEED");
    if (!reqs && feed && *feed && dir) {      /* the teacher-forced token array (int32) */
        FILE *ff = fopen(feed, "rb");
        if (!ff) k1_die("K1_FEED=%s: %s", feed, strerror(errno));
        fseek(ff, 0, SEEK_END); long sz = ftell(ff); fseek(ff, 0, SEEK_SET);
        if (sz < 4 || sz % 4) k1_die("K1_FEED=%s: not an int32 array", feed);
        reqs = k1_calloc(1, sizeof(K1Req), "feed request");
        reqs[0].n = (int)(sz / 4); reqs[0].ids = k1_malloc((size_t)sz, "feed ids");
        if (fread(reqs[0].ids, 4, (size_t)reqs[0].n, ff) != (size_t)reqs[0].n) k1_die("K1_FEED=%s: short read", feed);
        fclose(ff); nreq = 1;
    } else feed = NULL;
    (void)ids; (void)nids;
    if (reqs && dir) return k1_cmd_run(dir, release, reqs, nreq, ngen, feed);
    k1_usage(argv[0]);
    return 2;
}
#endif /* KOLIBRI1_NO_MAIN */
