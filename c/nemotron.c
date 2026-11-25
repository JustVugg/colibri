/* nemotron.c -- NVIDIA Nemotron-H / Nemotron-3.5-Super-VL-120B-A12B native engine
 * for Colibri.
 *
 * Reads the official Hugging Face safetensors checkpoints directly with zero
 * conversion step:
 *   - NemotronHForCausalLM (model_type="nemotron_h", e.g. Nemotron-3-Super-120B)
 *   - NemotronH_Omni_Reasoning_V3 (model_type="nemotron_h_omni", e.g. Nemotron-3.5-Super-VL-120B)
 *
 * Weight formats supported natively:
 *   - NVFP4 (U8 E2M1 nibbles + F8_E4M3 per-16 weight_scale + F32 weight_scale_2):
 *     routed experts (512/layer, streamed through a per-layer RAM LRU cache) and
 *     shared experts
 *   - FP8 (F8_E4M3 + scalar F32 weight_scale): Mamba-2 in_proj/out_proj,
 *     Attention q/k/v/o_proj, lm_head
 *   - BF16 / F16 / F32: embeddings, norms, conv1d, latent projections, router,
 *     RADIO vision tower, mlp1 projector, and unquantized checkpoints
 *
 * Architecture (exact mirror of modeling_nemotron_h.py + modeling_nemotron_h_omni.py):
 *   - Hybrid 88-layer backbone: Mamba-2 SSM ("mamba"/"linear_attention"),
 *     NoPE GQA Attention ("attention"/"full_attention"), LatentMoE ("moe",
 *     non-gated 2-matrix ReLU^2 experts with fc1/fc2_latent_proj + shared_experts),
 *     and plain ReLU^2 MLP ("mlp")
 *   - RADIO ViT-Huge vision tower + pixel-shuffle v2 + ReLU^2 mlp1 projector
 *     (c/nemotron_vision.h)
 */
#define _GNU_SOURCE
#define TOK_NFC
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "cpu_check.h"
#include "cli_args.h"
#include "compat.h"
#include "coli_env.h"
#include "json.h"
#include "st.h"
#include "tok.h"
#include "serve_codec.h"
#include "serve_poll.h"
#include "stop_ids.h"

static double jnum(jval *o, const char *k, double fb) {
    jval *v = o ? json_get(o, k) : NULL;
    return (v && v->t == J_NUM) ? v->num : fb;
}
static const char *jstr(jval *o, const char *k, const char *fb) {
    jval *v = o ? json_get(o, k) : NULL;
    return (v && v->t == J_STR && v->str) ? v->str : fb;
}
static int jbool(jval *o, const char *k, int fb) {
    jval *v = o ? json_get(o, k) : NULL;
    if (!v) return fb;
    if (v->t == J_BOOL) return v->boolean != 0;
    if (v->t == J_NUM) return v->num != 0.0;
    return fb;
}

#define NEMO_MAX_LAYERS 160
#define NEMO_MAX_TOPK   64

/* ------------------------------------------------------------------ utils ---- */

static double now_s(void) { return compat_now_s(); }
static double rss_gb(void) { return compat_rss_gb(); }

static void *xmalloc(size_t n, const char *what) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "[nemotron] OOM allocating %s (%zu bytes)\n", what, n); exit(1); }
    return p;
}

static void *xcalloc(size_t n, size_t sz, const char *what) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) { fprintf(stderr, "[nemotron] OOM allocating %s (%zu x %zu)\n", what, n, sz); exit(1); }
    return p;
}

/* -------------------------------------------------------- E4M3 / FP4 LUTs ---- */

static float g_e4m3_lut[256];
static const float g_fp4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
static int g_luts_init = 0;

static void nemo_init_luts(void) {
    if (g_luts_init) return;
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1;
        int e = (i >> 3) & 0xf;
        int m = i & 0x7;
        float v;
        if (e == 0xf && m == 0x7) v = 0.0f; /* NaN -> 0 */
        else if (e == 0) v = ldexpf((float)m, -9);
        else v = ldexpf(1.0f + (float)m / 8.0f, e - 7);
        g_e4m3_lut[i] = s ? -v : v;
    }
    g_luts_init = 1;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
static inline float softplus_f(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

static void nemo_rmsnorm(float *out, const float *in, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)in[i] * (double)in[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; i++) out[i] = in[i] * inv * w[i];
}

/* ----------------------------------------------------------- NemoMat --------- */

enum {
    NEMO_FMT_F32   = 0,
    NEMO_FMT_BF16  = 1,
    NEMO_FMT_FP8   = 2,   /* F8_E4M3 [O, I] + scalar F32 weight_scale */
    NEMO_FMT_NVFP4 = 3    /* U8 [O, I/2] + F8_E4M3 [O, I/16] + F32 weight_scale_2 */
};

typedef struct {
    int fmt, O, I;
    void *data;         /* f32*, uint16_t*, or uint8_t* */
    uint8_t *scales;    /* NVFP4: F8_E4M3 [O, I/16] */
    float scale_scalar; /* FP8 weight_scale or NVFP4 weight_scale_2 */
} NemoMat;

static float *nemo_load_f32(shards *S, const char *name, size_t n) {
    float *p = (float *)xmalloc(n * sizeof(float), name);
    st_read_f32(S, name, p, 0);
    return p;
}

static float nemo_load_scalar_f32(shards *S, const char *name, float fallback) {
    if (!st_has(S, name)) return fallback;
    float v = fallback;
    st_read_f32(S, name, &v, 0);
    return v;
}

static void nemo_mat_free(NemoMat *m) {
    if (!m) return;
    free(m->data);
    free(m->scales);
    memset(m, 0, sizeof(*m));
}

/* Load a linear weight matrix: auto-detects NVFP4, FP8, BF16, or F32 from safetensors. */
static void nemo_mat_load(NemoMat *m, shards *S, const char *wname, int O, int I) {
    memset(m, 0, sizeof(*m));
    m->O = O; m->I = I;
    st_tensor *t = st_find(S, wname);
    if (!t) { fprintf(stderr, "[nemotron] missing tensor: %s\n", wname); exit(1); }

    /* Derive prefix before ".weight" for scale lookups */
    char base[320], s1[360], s2[360];
    size_t wlen = strlen(wname);
    if (wlen > 7 && strcmp(wname + wlen - 7, ".weight") == 0) {
        size_t bl = wlen - 7;
        if (bl >= sizeof(base)) bl = sizeof(base) - 1;
        memcpy(base, wname, bl); base[bl] = '\0';
        snprintf(s1, sizeof(s1), "%s.weight_scale", base);
        snprintf(s2, sizeof(s2), "%s.weight_scale_2", base);
    } else {
        s1[0] = s2[0] = '\0';
    }

    if (t->dtype == 3 && s1[0] && st_has(S, s1)) {
        /* NVFP4: U8 [O, I/2] + F8_E4M3 [O, I/16] + F32 weight_scale_2 */
        m->fmt = NEMO_FMT_NVFP4;
        size_t wb = (size_t)O * (I / 2);
        size_t sb = (size_t)O * (I / 16);
        m->data = xmalloc(wb, wname);
        m->scales = (uint8_t *)xmalloc(sb, s1);
        st_read_raw(S, wname, m->data, 0);
        st_read_raw(S, s1, m->scales, 0);
        m->scale_scalar = nemo_load_scalar_f32(S, s2, 1.0f);
    } else if (t->dtype == 4) {
        /* FP8: F8_E4M3 [O, I] + scalar F32 weight_scale */
        m->fmt = NEMO_FMT_FP8;
        m->data = xmalloc((size_t)O * I, wname);
        st_read_raw(S, wname, m->data, 0);
        m->scale_scalar = s1[0] ? nemo_load_scalar_f32(S, s1, 1.0f) : 1.0f;
    } else if (t->dtype == 0) {
        /* BF16 */
        m->fmt = NEMO_FMT_BF16;
        m->data = xmalloc((size_t)O * I * 2, wname);
        st_read_raw(S, wname, m->data, 0);
        m->scale_scalar = 1.0f;
    } else {
        /* F32 or F16 -> load as F32 */
        m->fmt = NEMO_FMT_F32;
        m->data = nemo_load_f32(S, wname, (size_t)O * I);
        m->scale_scalar = 1.0f;
    }
}

static inline float nemo_dot_f32(const float *a, const float *b, int n) {
    float s = 0.0f;
    int i = 0;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    s = _mm_cvtss_f32(sum);
#endif
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float nemo_dot_bf16(const uint16_t *w, const float *x, int n) {
    float s = 0.0f;
    int i = 0;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m128i b16 = _mm_loadu_si128((const __m128i *)(w + i));
        __m256i i32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b16), 16);
        __m256 wf = _mm256_castsi256_ps(i32);
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), acc);
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    s = _mm_cvtss_f32(sum);
#endif
    for (; i < n; i++) s += bf16_to_f32(w[i]) * x[i];
    return s;
}

static inline float nemo_dot_fp8(const uint8_t *w, const float *x, int n, float scale) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += g_e4m3_lut[w[i]] * x[i];
    return s * scale;
}

/* NVFP4 row dot: 16 weights = 8 packed U8 bytes (low nibble = even, high = odd),
 * 1 F8_E4M3 scale byte per 16 weights, times scalar weight_scale_2. */
static inline float nemo_dot_nvfp4(const uint8_t *q, const uint8_t *sc, const float *x,
                                   int n, float scale2) {
    float total = 0.0f;
    int ng = n >> 4;
    for (int g = 0; g < ng; g++) {
        float gs = g_e4m3_lut[sc[g]];
        const uint8_t *qg = q + (size_t)g * 8;
        const float *xg = x + (size_t)g * 16;
        float acc = 0.0f;
        for (int j = 0; j < 8; j++) {
            uint8_t b = qg[j];
            acc += g_fp4_lut[b & 0xf] * xg[2 * j] + g_fp4_lut[b >> 4] * xg[2 * j + 1];
        }
        total += acc * gs;
    }
    return total * scale2;
}

static void nemo_mat_matvec(float *y, const NemoMat *m, const float *x) {
    int O = m->O, I = m->I;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (O >= 128)
    #endif
    for (int r = 0; r < O; r++) {
        switch (m->fmt) {
            case NEMO_FMT_F32:
                y[r] = nemo_dot_f32((const float *)m->data + (size_t)r * I, x, I);
                break;
            case NEMO_FMT_BF16:
                y[r] = nemo_dot_bf16((const uint16_t *)m->data + (size_t)r * I, x, I);
                break;
            case NEMO_FMT_FP8:
                y[r] = nemo_dot_fp8((const uint8_t *)m->data + (size_t)r * I, x, I, m->scale_scalar);
                break;
            case NEMO_FMT_NVFP4:
                y[r] = nemo_dot_nvfp4((const uint8_t *)m->data + (size_t)r * (I / 2),
                                      m->scales + (size_t)r * (I / 16), x, I, m->scale_scalar);
                break;
        }
    }
}

static void nemo_mat_matmul(float *Y, const NemoMat *m, const float *X, int S) {
    for (int s = 0; s < S; s++)
        nemo_mat_matvec(Y + (size_t)s * m->O, m, X + (size_t)s * m->I);
}

#include "nemotron_vision.h"

/* ----------------------------------------------------------- Config & Model -- */

enum {
    NEMO_BLK_MAMBA = 0,
    NEMO_BLK_ATTN  = 1,
    NEMO_BLK_MOE   = 2,
    NEMO_BLK_MLP   = 3
};

typedef struct {
    int n_layers;
    int block_type[NEMO_MAX_LAYERS];
    int attn_idx[NEMO_MAX_LAYERS];  /* 0..n_attn-1 for attn layers, else -1 */
    int mamba_idx[NEMO_MAX_LAYERS]; /* 0..n_mamba-1 for mamba layers, else -1 */
    int moe_idx[NEMO_MAX_LAYERS];   /* 0..n_moe-1 for moe layers, else -1 */
    int n_mamba, n_attn, n_moe, n_mlp;

    int vocab, hidden, inter;
    int n_heads, n_kv_heads, head_dim;
    /* Mamba-2 */
    int mamba_heads, mamba_head_dim, ssm_state, conv_kernel, n_groups;
    int mamba_d_inner, mamba_conv_dim;
    float dt_min, dt_max;
    /* LatentMoE */
    int n_experts, top_k, n_group, topk_group;
    int moe_inter, moe_latent, shared_inter, n_shared;
    int norm_topk_prob;
    float routed_scale;
    float rms_eps;
    int bos_id, eos_id, img_token_id;
    int is_omni;
} NemoCfg;

/* Cached routed expert: up_proj [moe_inter, exp_in] + down_proj [exp_in, moe_inter] */
typedef struct {
    int expert_id;
    uint64_t stamp;
    NemoMat up, down;
} NemoExpertSlot;

typedef struct {
    int cap;
    uint64_t clock;
    NemoExpertSlot *slots;
    int *lut; /* [n_experts] -> slot idx or -1 */
} NemoExpertCache;

typedef struct {
    int type;
    float *norm_w;
    /* Mamba-2 */
    NemoMat in_proj, out_proj;
    float *conv_w;      /* [conv_dim, conv_kernel] */
    float *conv_b;      /* [conv_dim] */
    float *dt_bias;     /* [mamba_heads] */
    float *A_neg_exp;   /* -exp(A_log) [mamba_heads] */
    float *D;           /* [mamba_heads] */
    float *mamba_norm;  /* [mamba_d_inner] */
    /* Attention */
    NemoMat q_proj, k_proj, v_proj, o_proj;
    /* MoE */
    float *gate_w;      /* F32 [n_experts, hidden] */
    float *gate_bias;   /* F32 [n_experts] (e_score_correction_bias) */
    NemoMat fc1_lat, fc2_lat;
    int has_latent;
    NemoMat sh_up, sh_down;
    int has_shared;
    NemoExpertCache ecache;
    /* Plain MLP */
    NemoMat mlp_up, mlp_down;
} NemoLayer;

/* Per-slot conversation state (KV cache for attention + conv/ssm state for Mamba-2) */
typedef struct {
    int pos, max_ctx;
    /* Attention KV: [n_attn][max_ctx, n_kv_heads * head_dim] stored as f32 */
    float **k_cache;
    float **v_cache;
    /* Mamba-2: conv_state [n_mamba][conv_dim * conv_kernel], ssm_state [n_mamba][mamba_heads * mamba_head_dim * ssm_state] */
    float **conv_state;
    float **ssm_state;
} NemoState;

typedef struct {
    NemoCfg c;
    shards S;
    char prefix[64];    /* "language_model.backbone." or "backbone." */
    char head_name[64]; /* "language_model.lm_head.weight" or "lm_head.weight" */
    NemoMat emb;        /* [vocab, hidden] */
    float *norm_f;      /* [hidden] */
    NemoMat lm_head;    /* [vocab, hidden] */
    NemoLayer *layers;
    NemoVision *vision;
    /* Scratch buffers (single-token forward) */
    float *x, *xn, *res, *logits;
    float *mb_proj, *mb_xbc, *mb_y;
    float *q_buf, *k_buf, *v_buf, *attn_out;
    float *lat_in, *lat_out, *exp_h, *exp_d, *sh_h, *sh_out;
    float *router_logits;
    uint64_t expert_hits, expert_misses;
} NemoModel;

/* ----------------------------------------------------- State Management ------ */

static void nemo_state_init(NemoState *st, const NemoCfg *c, int max_ctx) {
    memset(st, 0, sizeof(*st));
    st->max_ctx = max_ctx;
    int kv_dim = c->n_kv_heads * c->head_dim;
    if (c->n_attn > 0) {
        st->k_cache = (float **)xcalloc((size_t)c->n_attn, sizeof(float *), "k_cache_ptrs");
        st->v_cache = (float **)xcalloc((size_t)c->n_attn, sizeof(float *), "v_cache_ptrs");
        for (int a = 0; a < c->n_attn; a++) {
            st->k_cache[a] = (float *)xcalloc((size_t)max_ctx * kv_dim, sizeof(float), "k_cache");
            st->v_cache[a] = (float *)xcalloc((size_t)max_ctx * kv_dim, sizeof(float), "v_cache");
        }
    }
    if (c->n_mamba > 0) {
        st->conv_state = (float **)xcalloc((size_t)c->n_mamba, sizeof(float *), "conv_ptrs");
        st->ssm_state  = (float **)xcalloc((size_t)c->n_mamba, sizeof(float *), "ssm_ptrs");
        size_t cs = (size_t)c->mamba_conv_dim * c->conv_kernel;
        size_t ss = (size_t)c->mamba_heads * c->mamba_head_dim * c->ssm_state;
        for (int m = 0; m < c->n_mamba; m++) {
            st->conv_state[m] = (float *)xcalloc(cs, sizeof(float), "conv_state");
            st->ssm_state[m]  = (float *)xcalloc(ss, sizeof(float), "ssm_state");
        }
    }
}

static void nemo_state_clear(NemoState *st, const NemoCfg *c) {
    st->pos = 0;
    size_t cs = (size_t)c->mamba_conv_dim * c->conv_kernel * sizeof(float);
    size_t ss = (size_t)c->mamba_heads * c->mamba_head_dim * c->ssm_state * sizeof(float);
    for (int m = 0; m < c->n_mamba; m++) {
        memset(st->conv_state[m], 0, cs);
        memset(st->ssm_state[m], 0, ss);
    }
}

static void nemo_state_free(NemoState *st, const NemoCfg *c) {
    for (int a = 0; a < c->n_attn; a++) { free(st->k_cache[a]); free(st->v_cache[a]); }
    free(st->k_cache); free(st->v_cache);
    for (int m = 0; m < c->n_mamba; m++) { free(st->conv_state[m]); free(st->ssm_state[m]); }
    free(st->conv_state); free(st->ssm_state);
    memset(st, 0, sizeof(*st));
}

/* ----------------------------------------------------- Expert LRU Cache ------ */

static void nemo_ecache_init(NemoExpertCache *ec, int n_experts, int cap) {
    if (cap < 1) cap = 1;
    if (cap > n_experts) cap = n_experts;
    ec->cap = cap;
    ec->clock = 0;
    ec->slots = (NemoExpertSlot *)xcalloc((size_t)cap, sizeof(NemoExpertSlot), "ecache_slots");
    for (int i = 0; i < cap; i++) ec->slots[i].expert_id = -1;
    ec->lut = (int *)xmalloc((size_t)n_experts * sizeof(int), "ecache_lut");
    for (int i = 0; i < n_experts; i++) ec->lut[i] = -1;
}

static void nemo_ecache_free(NemoExpertCache *ec) {
    if (!ec->slots) return;
    for (int i = 0; i < ec->cap; i++) {
        nemo_mat_free(&ec->slots[i].up);
        nemo_mat_free(&ec->slots[i].down);
    }
    free(ec->slots);
    free(ec->lut);
    memset(ec, 0, sizeof(*ec));
}

static NemoExpertSlot *nemo_get_expert(NemoModel *m, int layer_idx, int exp_id) {
    NemoLayer *L = &m->layers[layer_idx];
    NemoExpertCache *ec = &L->ecache;
    int s = ec->lut[exp_id];
    if (s >= 0) {
        ec->slots[s].stamp = ++ec->clock;
        m->expert_hits++;
        return &ec->slots[s];
    }
    m->expert_misses++;
    /* Pick victim slot (empty or oldest stamp) */
    int victim = 0;
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < ec->cap; i++) {
        if (ec->slots[i].expert_id < 0) { victim = i; break; }
        if (ec->slots[i].stamp < best) { best = ec->slots[i].stamp; victim = i; }
    }
    NemoExpertSlot *sl = &ec->slots[victim];
    if (sl->expert_id >= 0) {
        ec->lut[sl->expert_id] = -1;
        nemo_mat_free(&sl->up);
        nemo_mat_free(&sl->down);
    }
    int exp_in = m->c.moe_latent > 0 ? m->c.moe_latent : m->c.hidden;
    char un[256], dn[256];
    snprintf(un, sizeof(un), "%slayers.%d.mixer.experts.%d.up_proj.weight", m->prefix, layer_idx, exp_id);
    snprintf(dn, sizeof(dn), "%slayers.%d.mixer.experts.%d.down_proj.weight", m->prefix, layer_idx, exp_id);
    nemo_mat_load(&sl->up, &m->S, un, m->c.moe_inter, exp_in);
    nemo_mat_load(&sl->down, &m->S, dn, exp_in, m->c.moe_inter);
    sl->expert_id = exp_id;
    sl->stamp = ++ec->clock;
    ec->lut[exp_id] = victim;
    return sl;
}

/* ------------------------------------------------------- Model Loading ------- */

static char *nemo_read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)xmalloc((size_t)sz + 1, "read_text");
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static void nemo_load(NemoModel *m, const char *dir, int ram_cap_slots) {
    memset(m, 0, sizeof(*m));
    nemo_init_luts();

    char cfg_path[4096];
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", dir);
    char *text = nemo_read_text(cfg_path);
    if (!text) { fprintf(stderr, "[nemotron] cannot read %s\n", cfg_path); exit(1); }
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    if (!root || root->t != J_OBJ) { fprintf(stderr, "[nemotron] invalid config.json\n"); exit(1); }

    const char *top_mt = jstr(root, "model_type", "");
    jval *lc = root;
    if (top_mt && strcmp(top_mt, "nemotron_h_omni") == 0) {
        m->c.is_omni = 1;
        jval *sub = json_get(root, "llm_config");
        if (sub && sub->t == J_OBJ) lc = sub;
        m->c.img_token_id = (int)jnum(root, "img_context_token_id", 18);
    }

    NemoCfg *c = &m->c;
    c->vocab = (int)jnum(lc, "vocab_size", 131072);
    c->hidden = (int)jnum(lc, "hidden_size", 4096);
    c->inter = (int)jnum(lc, "intermediate_size", 2688);
    c->n_heads = (int)jnum(lc, "num_attention_heads", 32);
    c->n_kv_heads = (int)jnum(lc, "num_key_value_heads", 2);
    c->head_dim = (int)jnum(lc, "head_dim", c->hidden / c->n_heads);
    c->mamba_heads = (int)jnum(lc, "mamba_num_heads", 128);
    c->mamba_head_dim = (int)jnum(lc, "mamba_head_dim", 64);
    c->ssm_state = (int)jnum(lc, "ssm_state_size", 128);
    c->conv_kernel = (int)jnum(lc, "conv_kernel", 4);
    c->n_groups = (int)jnum(lc, "n_groups", 8);
    c->mamba_d_inner = c->mamba_heads * c->mamba_head_dim;
    c->mamba_conv_dim = c->mamba_d_inner + 2 * c->n_groups * c->ssm_state;
    c->dt_min = 0.0f;
    c->dt_max = 1e30f;
    jval *dtl = json_get(lc, "time_step_limit");
    if (dtl && dtl->t == J_ARR && dtl->len == 2) {
        if (dtl->kids[0]->t == J_NUM) c->dt_min = (float)dtl->kids[0]->num;
        if (dtl->kids[1]->t == J_NUM) c->dt_max = (float)dtl->kids[1]->num;
    }

    c->n_experts = (int)jnum(lc, "n_routed_experts", 0);
    c->top_k = (int)jnum(lc, "num_experts_per_tok", 1);
    c->n_group = (int)jnum(lc, "n_group", 1);
    c->topk_group = (int)jnum(lc, "topk_group", 1);
    c->moe_inter = (int)jnum(lc, "moe_intermediate_size", c->inter);
    c->moe_latent = (int)jnum(lc, "moe_latent_size", 0);
    c->shared_inter = (int)jnum(lc, "moe_shared_expert_intermediate_size", 0);
    c->n_shared = (int)jnum(lc, "n_shared_experts", 0);
    c->norm_topk_prob = jbool(lc, "norm_topk_prob", 1);
    c->routed_scale = (float)jnum(lc, "routed_scaling_factor", 1.0);
    c->rms_eps = (float)jnum(lc, "layer_norm_epsilon", 1e-5);
    c->bos_id = (int)jnum(lc, "bos_token_id", 1);
    c->eos_id = (int)jnum(lc, "eos_token_id", 11);

    /* Parse layers_block_type (or hybrid_override_pattern) */
    jval *lbt = json_get(lc, "layers_block_type");
    if (lbt && lbt->t == J_ARR) {
        c->n_layers = (int)lbt->len;
        if (c->n_layers > NEMO_MAX_LAYERS) { fprintf(stderr, "[nemotron] too many layers\n"); exit(1); }
        for (int i = 0; i < c->n_layers; i++) {
            const char *s = lbt->kids[i]->str ? lbt->kids[i]->str : "";
            c->attn_idx[i] = c->mamba_idx[i] = c->moe_idx[i] = -1;
            if (!strcmp(s, "mamba") || !strcmp(s, "linear_attention")) {
                c->block_type[i] = NEMO_BLK_MAMBA; c->mamba_idx[i] = c->n_mamba++;
            } else if (!strcmp(s, "attention") || !strcmp(s, "full_attention")) {
                c->block_type[i] = NEMO_BLK_ATTN; c->attn_idx[i] = c->n_attn++;
            } else if (!strcmp(s, "moe")) {
                c->block_type[i] = NEMO_BLK_MOE; c->moe_idx[i] = c->n_moe++;
            } else {
                c->block_type[i] = NEMO_BLK_MLP; c->n_mlp++;
            }
        }
    } else {
        const char *pat = jstr(lc, "hybrid_override_pattern", "");
        c->n_layers = (int)strlen(pat);
        for (int i = 0; i < c->n_layers; i++) {
            c->attn_idx[i] = c->mamba_idx[i] = c->moe_idx[i] = -1;
            if (pat[i] == 'M') { c->block_type[i] = NEMO_BLK_MAMBA; c->mamba_idx[i] = c->n_mamba++; }
            else if (pat[i] == '*') { c->block_type[i] = NEMO_BLK_ATTN; c->attn_idx[i] = c->n_attn++; }
            else if (pat[i] == 'E') { c->block_type[i] = NEMO_BLK_MOE; c->moe_idx[i] = c->n_moe++; }
            else { c->block_type[i] = NEMO_BLK_MLP; c->n_mlp++; }
        }
    }

    /* Open safetensors shards */
    st_init(&m->S, dir);

    if (st_has(&m->S, "language_model.backbone.embeddings.weight")) {
        snprintf(m->prefix, sizeof(m->prefix), "language_model.backbone.");
        snprintf(m->head_name, sizeof(m->head_name), "language_model.lm_head.weight");
    } else {
        snprintf(m->prefix, sizeof(m->prefix), "backbone.");
        snprintf(m->head_name, sizeof(m->head_name), "lm_head.weight");
    }

    char tn[256];
    snprintf(tn, sizeof(tn), "%sembeddings.weight", m->prefix);
    nemo_mat_load(&m->emb, &m->S, tn, c->vocab, c->hidden);
    snprintf(tn, sizeof(tn), "%snorm_f.weight", m->prefix);
    m->norm_f = nemo_load_f32(&m->S, tn, c->hidden);
    if (st_has(&m->S, m->head_name)) {
        nemo_mat_load(&m->lm_head, &m->S, m->head_name, c->vocab, c->hidden);
    } else {
        /* Tied embeddings */
        m->lm_head = m->emb;
    }

    m->layers = (NemoLayer *)xcalloc((size_t)c->n_layers, sizeof(NemoLayer), "nemo_layers");
    int ecap = ram_cap_slots > 0 ? ram_cap_slots : (c->n_experts > 64 ? 64 : c->n_experts);

    for (int i = 0; i < c->n_layers; i++) {
        NemoLayer *L = &m->layers[i];
        L->type = c->block_type[i];
        snprintf(tn, sizeof(tn), "%slayers.%d.norm.weight", m->prefix, i);
        L->norm_w = nemo_load_f32(&m->S, tn, c->hidden);

        if (L->type == NEMO_BLK_MAMBA) {
            int proj_out = c->mamba_d_inner + c->mamba_conv_dim + c->mamba_heads;
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.in_proj.weight", m->prefix, i);
            nemo_mat_load(&L->in_proj, &m->S, tn, proj_out, c->hidden);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.out_proj.weight", m->prefix, i);
            nemo_mat_load(&L->out_proj, &m->S, tn, c->hidden, c->mamba_d_inner);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.conv1d.weight", m->prefix, i);
            L->conv_w = nemo_load_f32(&m->S, tn, (size_t)c->mamba_conv_dim * c->conv_kernel);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.conv1d.bias", m->prefix, i);
            if (st_has(&m->S, tn)) L->conv_b = nemo_load_f32(&m->S, tn, c->mamba_conv_dim);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.dt_bias", m->prefix, i);
            L->dt_bias = nemo_load_f32(&m->S, tn, c->mamba_heads);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.A_log", m->prefix, i);
            L->A_neg_exp = nemo_load_f32(&m->S, tn, c->mamba_heads);
            for (int h = 0; h < c->mamba_heads; h++) L->A_neg_exp[h] = -expf(L->A_neg_exp[h]);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.D", m->prefix, i);
            L->D = nemo_load_f32(&m->S, tn, c->mamba_heads);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.norm.weight", m->prefix, i);
            L->mamba_norm = nemo_load_f32(&m->S, tn, c->mamba_d_inner);
        } else if (L->type == NEMO_BLK_ATTN) {
            int qd = c->n_heads * c->head_dim, kvd = c->n_kv_heads * c->head_dim;
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.q_proj.weight", m->prefix, i);
            nemo_mat_load(&L->q_proj, &m->S, tn, qd, c->hidden);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.k_proj.weight", m->prefix, i);
            nemo_mat_load(&L->k_proj, &m->S, tn, kvd, c->hidden);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.v_proj.weight", m->prefix, i);
            nemo_mat_load(&L->v_proj, &m->S, tn, kvd, c->hidden);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.o_proj.weight", m->prefix, i);
            nemo_mat_load(&L->o_proj, &m->S, tn, c->hidden, qd);
        } else if (L->type == NEMO_BLK_MOE) {
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.gate.weight", m->prefix, i);
            L->gate_w = nemo_load_f32(&m->S, tn, (size_t)c->n_experts * c->hidden);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.gate.e_score_correction_bias", m->prefix, i);
            if (st_has(&m->S, tn)) L->gate_bias = nemo_load_f32(&m->S, tn, c->n_experts);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.fc1_latent_proj.weight", m->prefix, i);
            if (c->moe_latent > 0 && st_has(&m->S, tn)) {
                L->has_latent = 1;
                nemo_mat_load(&L->fc1_lat, &m->S, tn, c->moe_latent, c->hidden);
                snprintf(tn, sizeof(tn), "%slayers.%d.mixer.fc2_latent_proj.weight", m->prefix, i);
                nemo_mat_load(&L->fc2_lat, &m->S, tn, c->hidden, c->moe_latent);
            }
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.shared_experts.up_proj.weight", m->prefix, i);
            if (c->n_shared > 0 && c->shared_inter > 0 && st_has(&m->S, tn)) {
                L->has_shared = 1;
                nemo_mat_load(&L->sh_up, &m->S, tn, c->shared_inter, c->hidden);
                snprintf(tn, sizeof(tn), "%slayers.%d.mixer.shared_experts.down_proj.weight", m->prefix, i);
                nemo_mat_load(&L->sh_down, &m->S, tn, c->hidden, c->shared_inter);
            }
            nemo_ecache_init(&L->ecache, c->n_experts, ecap);
        } else {
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.up_proj.weight", m->prefix, i);
            nemo_mat_load(&L->mlp_up, &m->S, tn, c->inter, c->hidden);
            snprintf(tn, sizeof(tn), "%slayers.%d.mixer.down_proj.weight", m->prefix, i);
            nemo_mat_load(&L->mlp_down, &m->S, tn, c->hidden, c->inter);
        }
    }

    if (c->is_omni) m->vision = nemo_vision_load(&m->S, root, c->hidden, c->rms_eps);
    json_free(root); free(arena); free(text);

    /* Allocate scratch buffers */
    int exp_in = c->moe_latent > 0 ? c->moe_latent : c->hidden;
    int proj_out = c->mamba_d_inner + c->mamba_conv_dim + c->mamba_heads;
    m->x = (float *)xmalloc(c->hidden * sizeof(float), "x");
    m->xn = (float *)xmalloc(c->hidden * sizeof(float), "xn");
    m->res = (float *)xmalloc(c->hidden * sizeof(float), "res");
    m->logits = (float *)xmalloc(c->vocab * sizeof(float), "logits");
    if (c->n_mamba > 0) {
        m->mb_proj = (float *)xmalloc(proj_out * sizeof(float), "mb_proj");
        m->mb_xbc  = (float *)xmalloc(c->mamba_conv_dim * sizeof(float), "mb_xbc");
        m->mb_y    = (float *)xmalloc(c->mamba_d_inner * sizeof(float), "mb_y");
    }
    if (c->n_attn > 0) {
        m->q_buf    = (float *)xmalloc((size_t)c->n_heads * c->head_dim * sizeof(float), "q");
        m->k_buf    = (float *)xmalloc((size_t)c->n_kv_heads * c->head_dim * sizeof(float), "k");
        m->v_buf    = (float *)xmalloc((size_t)c->n_kv_heads * c->head_dim * sizeof(float), "v");
        m->attn_out = (float *)xmalloc((size_t)c->n_heads * c->head_dim * sizeof(float), "ao");
    }
    if (c->n_moe > 0) {
        m->lat_in  = (float *)xmalloc(exp_in * sizeof(float), "lat_in");
        m->lat_out = (float *)xmalloc(exp_in * sizeof(float), "lat_out");
        m->exp_h   = (float *)xmalloc(c->moe_inter * sizeof(float), "exp_h");
        m->exp_d   = (float *)xmalloc(exp_in * sizeof(float), "exp_d");
        m->router_logits = (float *)xmalloc(c->n_experts * sizeof(float), "rlogits");
        if (c->shared_inter > 0) {
            m->sh_h   = (float *)xmalloc(c->shared_inter * sizeof(float), "sh_h");
            m->sh_out = (float *)xmalloc(c->hidden * sizeof(float), "sh_out");
        }
    } else if (c->n_mlp > 0) {
        m->sh_h = (float *)xmalloc(c->inter * sizeof(float), "mlp_h");
    }

    fprintf(stderr, "[nemotron] loaded %d layers (Mamba=%d Attn=%d MoE=%d MLP=%d) hidden=%d vocab=%d experts=%d/%d latent=%d rss=%.2f GB\n",
            c->n_layers, c->n_mamba, c->n_attn, c->n_moe, c->n_mlp,
            c->hidden, c->vocab, c->top_k, c->n_experts, c->moe_latent, rss_gb());
}

/* ----------------------------------------------------- Forward Pass ---------- */

static void nemo_embed_token(const NemoModel *m, int tok, float *out) {
    int D = m->c.hidden;
    if (tok < 0 || tok >= m->c.vocab) tok = 0;
    if (m->emb.fmt == NEMO_FMT_BF16) {
        const uint16_t *row = (const uint16_t *)m->emb.data + (size_t)tok * D;
        for (int i = 0; i < D; i++) out[i] = bf16_to_f32(row[i]);
    } else {
        memcpy(out, (const float *)m->emb.data + (size_t)tok * D, (size_t)D * sizeof(float));
    }
}

static void nemo_mamba_step(NemoModel *m, int li, NemoState *st, float *out) {
    const NemoCfg *c = &m->c;
    const NemoLayer *L = &m->layers[li];
    int mi = c->mamba_idx[li];
    int D_in = c->mamba_d_inner, C_dim = c->mamba_conv_dim;
    int H = c->mamba_heads, P = c->mamba_head_dim, N = c->ssm_state;
    int G = c->n_groups, K = c->conv_kernel;
    int hpg = H / G;

    /* 1. in_proj: [z (D_in), xBC (C_dim), dt (H)] */
    nemo_mat_matvec(m->mb_proj, &L->in_proj, m->xn);
    const float *z = m->mb_proj;
    const float *xbc_in = m->mb_proj + D_in;
    const float *dt_raw = m->mb_proj + D_in + C_dim;

    /* 2. Causal depthwise conv1d (kernel=K) + SiLU */
    float *cs = st->conv_state[mi];
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (C_dim >= 1024)
    #endif
    for (int ch = 0; ch < C_dim; ch++) {
        float *s = cs + (size_t)ch * K;
        for (int k = 0; k < K - 1; k++) s[k] = s[k + 1];
        s[K - 1] = xbc_in[ch];
        const float *w = L->conv_w + (size_t)ch * K;
        float acc = L->conv_b ? L->conv_b[ch] : 0.0f;
        for (int k = 0; k < K; k++) acc += s[k] * w[k];
        m->mb_xbc[ch] = silu_f(acc);
    }

    const float *x_ssm = m->mb_xbc;
    const float *B = m->mb_xbc + D_in;
    const float *C = m->mb_xbc + D_in + G * N;

    /* 3. Selective SSM state update (exact Mamba-2 recurrence) */
    float *ss = st->ssm_state[mi];
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (H >= 16)
    #endif
    for (int h = 0; h < H; h++) {
        float dt = softplus_f(dt_raw[h] + L->dt_bias[h]);
        if (dt < c->dt_min) dt = c->dt_min;
        if (dt > c->dt_max) dt = c->dt_max;
        float dA = expf(L->A_neg_exp[h] * dt);
        float Dh = L->D[h];
        int g = h / hpg;
        const float *Bg = B + g * N;
        const float *Cg = C + g * N;
        const float *xh = x_ssm + h * P;
        float *yh = m->mb_y + h * P;
        float *sh = ss + (size_t)h * P * N;
        for (int p = 0; p < P; p++) {
            float xp = xh[p];
            float *shp = sh + (size_t)p * N;
            float yv = Dh * xp;
            float dt_xp = dt * xp;
            for (int n = 0; n < N; n++) {
                float nv = shp[n] * dA + dt_xp * Bg[n];
                shp[n] = nv;
                yv += nv * Cg[n];
            }
            yh[p] = yv;
        }
    }

    /* 4. NemotronHRMSNormGated: y = y * silu(z), then per-group RMSNorm * mamba_norm */
    int gsz = D_in / G;
    for (int g = 0; g < G; g++) {
        int off = g * gsz;
        double ssq = 0.0;
        for (int j = 0; j < gsz; j++) {
            float v = m->mb_y[off + j] * silu_f(z[off + j]);
            m->mb_y[off + j] = v;
            ssq += (double)v * (double)v;
        }
        float inv = 1.0f / sqrtf((float)(ssq / (double)gsz) + c->rms_eps);
        for (int j = 0; j < gsz; j++)
            m->mb_y[off + j] = m->mb_y[off + j] * inv * L->mamba_norm[off + j];
    }

    /* 5. out_proj */
    nemo_mat_matvec(out, &L->out_proj, m->mb_y);
}

static void nemo_attn_step(NemoModel *m, int li, NemoState *st, float *out) {
    const NemoCfg *c = &m->c;
    const NemoLayer *L = &m->layers[li];
    int ai = c->attn_idx[li];
    int H = c->n_heads, KVH = c->n_kv_heads, hd = c->head_dim;
    int kv_dim = KVH * hd, gqa = H / KVH;
    int pos = st->pos;

    nemo_mat_matvec(m->q_buf, &L->q_proj, m->xn);
    nemo_mat_matvec(m->k_buf, &L->k_proj, m->xn);
    nemo_mat_matvec(m->v_buf, &L->v_proj, m->xn);

    /* Store K, V at st->pos (NoPE: no RoPE in NemotronH) */
    memcpy(st->k_cache[ai] + (size_t)pos * kv_dim, m->k_buf, (size_t)kv_dim * sizeof(float));
    memcpy(st->v_cache[ai] + (size_t)pos * kv_dim, m->v_buf, (size_t)kv_dim * sizeof(float));

    int seq = pos + 1;
    float scale = 1.0f / sqrtf((float)hd);
    const float *kc = st->k_cache[ai];
    const float *vc = st->v_cache[ai];

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (H >= 8)
    #endif
    for (int h = 0; h < H; h++) {
        int kh = h / gqa;
        const float *qh = m->q_buf + h * hd;
        float *oh = m->attn_out + h * hd;
        float *sc = (float *)malloc((size_t)seq * sizeof(float));
        float mx = -1e30f;
        for (int t = 0; t < seq; t++) {
            const float *kt = kc + (size_t)t * kv_dim + kh * hd;
            float s = nemo_dot_f32(qh, kt, hd) * scale;
            sc[t] = s;
            if (s > mx) mx = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < seq; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        float inv = 1.0f / sum;
        memset(oh, 0, (size_t)hd * sizeof(float));
        for (int t = 0; t < seq; t++) {
            float w = sc[t] * inv;
            const float *vt = vc + (size_t)t * kv_dim + kh * hd;
            for (int d = 0; d < hd; d++) oh[d] += w * vt[d];
        }
        free(sc);
    }

    nemo_mat_matvec(out, &L->o_proj, m->attn_out);
}

static void nemo_moe_step(NemoModel *m, int li, float *out) {
    const NemoCfg *c = &m->c;
    const NemoLayer *L = &m->layers[li];
    int E = c->n_experts, K = c->top_k, D = c->hidden;
    int exp_in = L->has_latent ? c->moe_latent : D;

    /* 1. Router: sigmoid scores + e_score_correction_bias + group top-k */
    float scores[1024], choice[1024];
    for (int e = 0; e < E; e++) {
        float lg = nemo_dot_f32(L->gate_w + (size_t)e * D, m->xn, D);
        float sc = 1.0f / (1.0f + expf(-lg));
        scores[e] = sc;
        choice[e] = sc + (L->gate_bias ? L->gate_bias[e] : 0.0f);
    }
    if (c->n_group > 1 && c->topk_group < c->n_group) {
        int epg = E / c->n_group;
        float gs[64];
        for (int g = 0; g < c->n_group; g++) {
            float m1 = -1e30f, m2 = -1e30f;
            for (int j = 0; j < epg; j++) {
                float v = choice[g * epg + j];
                if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) m2 = v;
            }
            gs[g] = m1 + m2;
        }
        int keep[64] = {0};
        for (int k = 0; k < c->topk_group; k++) {
            int bi = -1; float bv = -1e30f;
            for (int g = 0; g < c->n_group; g++)
                if (!keep[g] && gs[g] > bv) { bv = gs[g]; bi = g; }
            if (bi >= 0) keep[bi] = 1;
        }
        for (int g = 0; g < c->n_group; g++)
            if (!keep[g]) for (int j = 0; j < epg; j++) choice[g * epg + j] = -1e30f;
    }

    int top_idx[NEMO_MAX_TOPK];
    float top_w[NEMO_MAX_TOPK];
    int used[1024] = {0};
    float wsum = 0.0f;
    for (int k = 0; k < K; k++) {
        int bi = 0; float bv = -1e30f;
        for (int e = 0; e < E; e++)
            if (!used[e] && choice[e] > bv) { bv = choice[e]; bi = e; }
        used[bi] = 1;
        top_idx[k] = bi;
        top_w[k] = scores[bi];
        wsum += scores[bi];
    }
    if (c->norm_topk_prob) {
        float inv = 1.0f / (wsum + 1e-20f);
        for (int k = 0; k < K; k++) top_w[k] *= inv;
    }
    for (int k = 0; k < K; k++) top_w[k] *= c->routed_scale;

    /* 2. Latent down-projection (fc1_latent_proj) */
    if (L->has_latent) nemo_mat_matvec(m->lat_in, &L->fc1_lat, m->xn);
    else memcpy(m->lat_in, m->xn, (size_t)D * sizeof(float));

    /* 3. Routed experts (non-gated ReLU^2: down(relu2(up(lat_in)))) */
    memset(m->lat_out, 0, (size_t)exp_in * sizeof(float));
    for (int k = 0; k < K; k++) {
        NemoExpertSlot *sl = nemo_get_expert(m, li, top_idx[k]);
        nemo_mat_matvec(m->exp_h, &sl->up, m->lat_in);
        for (int j = 0; j < c->moe_inter; j++) {
            float v = m->exp_h[j];
            m->exp_h[j] = v > 0.0f ? v * v : 0.0f;
        }
        nemo_mat_matvec(m->exp_d, &sl->down, m->exp_h);
        float wk = top_w[k];
        for (int j = 0; j < exp_in; j++) m->lat_out[j] += wk * m->exp_d[j];
    }

    /* 4. Latent up-projection (fc2_latent_proj) */
    if (L->has_latent) nemo_mat_matvec(out, &L->fc2_lat, m->lat_out);
    else memcpy(out, m->lat_out, (size_t)D * sizeof(float));

    /* 5. Shared expert (non-gated ReLU^2) */
    if (L->has_shared) {
        nemo_mat_matvec(m->sh_h, &L->sh_up, m->xn);
        for (int j = 0; j < c->shared_inter; j++) {
            float v = m->sh_h[j];
            m->sh_h[j] = v > 0.0f ? v * v : 0.0f;
        }
        nemo_mat_matvec(m->sh_out, &L->sh_down, m->sh_h);
        for (int j = 0; j < D; j++) out[j] += m->sh_out[j];
    }
}

/* Run one token (from embedding vector `emb_in`, or token id if `emb_in == NULL`). */
static void nemo_forward_one(NemoModel *m, NemoState *st, int tok, const float *emb_in, int want_logits) {
    const NemoCfg *c = &m->c;
    int D = c->hidden;
    if (st->pos >= st->max_ctx) {
        fprintf(stderr, "[nemotron] context overflow (%d >= %d)\n", st->pos, st->max_ctx);
        exit(1);
    }
    if (emb_in) memcpy(m->x, emb_in, (size_t)D * sizeof(float));
    else nemo_embed_token(m, tok, m->x);

    for (int li = 0; li < c->n_layers; li++) {
        const NemoLayer *L = &m->layers[li];
        nemo_rmsnorm(m->xn, m->x, L->norm_w, D, c->rms_eps);
        if (L->type == NEMO_BLK_MAMBA) {
            nemo_mamba_step(m, li, st, m->res);
        } else if (L->type == NEMO_BLK_ATTN) {
            nemo_attn_step(m, li, st, m->res);
        } else if (L->type == NEMO_BLK_MOE) {
            nemo_moe_step(m, li, m->res);
        } else {
            nemo_mat_matvec(m->sh_h, &L->mlp_up, m->xn);
            for (int j = 0; j < c->inter; j++) {
                float v = m->sh_h[j];
                m->sh_h[j] = v > 0.0f ? v * v : 0.0f;
            }
            nemo_mat_matvec(m->res, &L->mlp_down, m->sh_h);
        }
        for (int j = 0; j < D; j++) m->x[j] += m->res[j];
    }
    st->pos++;
    if (want_logits) {
        nemo_rmsnorm(m->xn, m->x, m->norm_f, D, c->rms_eps);
        nemo_mat_matvec(m->logits, &m->lm_head, m->xn);
    }
}

static int nemo_argmax(const float *v, int n) {
    int bi = 0; float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
    return bi;
}

#include "nemotron_cuda_chain.h"

/* ----------------------------------------------------- CLI & --serve --------- */

int main(int argc, char **argv) {
    coli_cpu_check();
    const char *dir = NULL;
    const char *prompt = NULL;
    const char *dump_logits = NULL;
    const char *img_bin = NULL;
    int img_h = 0, img_w = 0;
    int max_tokens = 64, ctx = 4096, cap = 64, serve = 0;
    const char *env_ctx = getenv("CTX");
    if (env_ctx && atoi(env_ctx) > 0) ctx = atoi(env_ctx);
    int force_cpu = 0, gpu_dev = 0, gpu_slots = 0;
    float temp = 0.0f, top_p = 0.95f;
    int top_k = 40;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) force_cpu = 1;
        else if (!strcmp(argv[i], "--gpu") && i + 1 < argc) gpu_dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gpu-slots") && i + 1 < argc) gpu_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump_logits = argv[++i];
        else if (!strcmp(argv[i], "--image-f32") && i + 3 < argc) {
            img_bin = argv[++i]; img_h = atoi(argv[++i]); img_w = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--serve")) serve = 1;
        else if (argv[i][0] != '-' && !dir) dir = argv[i];
    }
    if (getenv("COLI_NEMOTRON_CPU") && atoi(getenv("COLI_NEMOTRON_CPU")) != 0) force_cpu = 1;
    if (!dir) {
        fprintf(stderr, "Usage: nemotron <model_dir> [--prompt TEXT] [--max-tokens N] [--ctx N] [--cap N] [--cpu] [--gpu N] [--serve]\n");
        return 1;
    }

    NemoModel m;
    nemo_load(&m, dir, cap);
    NemoState st;
    nemo_state_init(&st, &m.c, ctx);
    NemoCuda cu;
    memset(&cu, 0, sizeof(cu));
    if (!force_cpu) {
        if (nemo_cuda_init(&cu, &m, ctx, gpu_dev, gpu_slots) != 0) {
            cu.active = 0;
        }
    }

    Tok tok;
    char tok_path[4096];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", dir);
    int has_tok = 0;
    {
        FILE *tf = fopen(tok_path, "rb");
        if (tf) { fclose(tf); tok_load(&tok, tok_path); has_tok = 1; }
    }

    if (serve) {
        coli_serve_write_ready_caps(stdout, rss_gb(), m.vision ? "vision=1" : "vision=0");
        char *line = NULL;
        size_t lcap = 0;
        while (compat_getline(&line, &lcap, stdin) > 0) {
            if (!strncmp(line, "QUIT", 4)) break;
            if (!strncmp(line, "KV_CLEAR", 8)) {
                nemo_state_clear(&st, &m.c);
                if (cu.active) nemo_cuda_state_clear(&cu, &m.c);
                fprintf(stdout, "OK\n"); fflush(stdout);
                continue;
            }
            if (!strncmp(line, "PROMPT_TOKENS ", 14)) {
                int ids[8192], n = 0;
                char *p = line + 14;
                while (*p && n < 8192) {
                    while (*p == ' ' || *p == '\t') p++;
                    if (!*p || *p == '\n' || *p == '\r') break;
                    ids[n++] = (int)strtol(p, &p, 10);
                }
                nemo_state_clear(&st, &m.c);
                if (cu.active) nemo_cuda_state_clear(&cu, &m.c);
                for (int i = 0; i < n; i++) {
                    if (cu.active) nemo_cuda_forward_one(&cu, &m, &st, ids[i], NULL, i == n - 1);
                    else nemo_forward_one(&m, &st, ids[i], NULL, i == n - 1);
                }
                for (int g = 0; g < max_tokens; g++) {
                    int next = nemo_argmax(m.logits, m.c.vocab);
                    fprintf(stdout, "TOK %d\n", next); fflush(stdout);
                    if (next == m.c.eos_id) break;
                    if (cu.active) nemo_cuda_forward_one(&cu, &m, &st, next, NULL, 1);
                    else nemo_forward_one(&m, &st, next, NULL, 1);
                }
                fprintf(stdout, "DONE\n"); fflush(stdout);
            }
        }
        free(line);
    } else {
        int ids[4096], n = 0;
        if (prompt && has_tok) {
            n = tok_encode(&tok, prompt, (int)strlen(prompt), ids, 4096);
        } else if (prompt) {
            /* Space-separated token ids fallback when no tokenizer.json */
            const char *p = prompt;
            while (*p && n < 4096) {
                while (*p == ' ' || *p == ',') p++;
                if (!*p) break;
                ids[n++] = (int)strtol(p, (char **)&p, 10);
            }
        } else {
            ids[n++] = m.c.bos_id;
        }

        /* Optional vision image embedding scatter onto img_token_id */
        float *img_embeds = NULL;
        int n_img_toks = 0, img_cursor = 0;
        if (img_bin && m.vision && img_h > 0 && img_w > 0) {
            size_t npix = (size_t)3 * img_h * img_w;
            float *chw = (float *)xmalloc(npix * sizeof(float), "img_chw");
            FILE *f = fopen(img_bin, "rb");
            if (!f || fread(chw, sizeof(float), npix, f) != npix) {
                fprintf(stderr, "[nemotron] failed to read %s\n", img_bin); exit(1);
            }
            fclose(f);
            int max_out = (img_h / (m.vision->patch * 2)) * (img_w / (m.vision->patch * 2));
            img_embeds = (float *)xmalloc((size_t)max_out * m.c.hidden * sizeof(float), "img_embeds");
            n_img_toks = nemo_vision_encode(m.vision, chw, img_h, img_w, img_embeds);
            free(chw);
        }

        double t0 = now_s();
        for (int i = 0; i < n; i++) {
            const float *emb = NULL;
            if (img_embeds && ids[i] == m.c.img_token_id && img_cursor < n_img_toks)
                emb = img_embeds + (size_t)(img_cursor++) * m.c.hidden;
            if (cu.active) nemo_cuda_forward_one(&cu, &m, &st, ids[i], emb, i == n - 1);
            else nemo_forward_one(&m, &st, ids[i], emb, i == n - 1);
        }
        double t_prefill = now_s() - t0;

        if (dump_logits) {
            FILE *lf = fopen(dump_logits, "wb");
            if (lf) { fwrite(m.logits, sizeof(float), (size_t)m.c.vocab, lf); fclose(lf); }
        }

        double t_dec0 = now_s();
        int gen = 0;
        (void)temp; (void)top_p; (void)top_k; (void)seed;
        for (int g = 0; g < max_tokens; g++) {
            int next = nemo_argmax(m.logits, m.c.vocab);
            gen++;
            if (has_tok) {
                char buf[256];
                int len = tok_decode(&tok, &next, 1, buf, (int)sizeof(buf));
                if (len > 0) fwrite(buf, 1, (size_t)len, stdout);
                fflush(stdout);
            } else {
                printf("%d%s", next, g + 1 < max_tokens ? " " : "\n");
            }
            if (next == m.c.eos_id) break;
            if (g + 1 < max_tokens) {
                if (cu.active) nemo_cuda_forward_one(&cu, &m, &st, next, NULL, 1);
                else nemo_forward_one(&m, &st, next, NULL, 1);
            }
        }
        double t_dec = now_s() - t_dec0;
        fprintf(stderr, "\n[nemotron] %s | prefill %d tok in %.2fs | decode %d tok in %.2fs (%.2f tok/s) | ram_exp=%llu/%llu vram_exp=%llu/%llu\n",
                cu.active ? "CUDA" : "CPU",
                n, t_prefill, gen, t_dec, gen / (t_dec > 1e-6 ? t_dec : 1e-6),
                (unsigned long long)m.expert_hits, (unsigned long long)m.expert_misses,
                (unsigned long long)cu.vram_hits, (unsigned long long)cu.vram_misses);
        free(img_embeds);
    }

    if (has_tok) tok_free(&tok);
    nemo_state_free(&st, &m.c);
    nemo_vision_free(m.vision);
    return 0;
}
