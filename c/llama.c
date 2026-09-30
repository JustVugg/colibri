/* Llama-family dense decoder (model_type "llama", LlamaForCausalLM).
 *
 * The simplest trunk in the repo: RMSNorm (no bias), RoPE with optional
 * llama3/linear frequency scaling, grouped-query attention, SwiGLU MLP
 * (gate_proj/up_proj/down_proj), optional tied embeddings. No MoE, no gates,
 * no DeltaNet.
 *
 * Weights are read straight from the checkpoint's safetensors (BF16/F16/F32 ->
 * f32, fully resident). Every tensor goes through load_t, which checks the
 * header's element count against the count the config implies BEFORE it
 * allocates or reads, and refuses a mismatch by name -- same contract as
 * olmoe.c/inkling.c load_t and qwen36.c load_t_n. A short tensor is never
 * padded, a long one never truncated.
 *
 * Harness mode (the only mode): `llama [ref.json]` with SNAP=<checkpoint dir>.
 * ref.json carries prompt_ids and full_ids (tools/make_llama_oracle.py);
 * greedy decode, then `Matching tokens: N/M`.
 *
 * ENV: SNAP=dir (required); COLI_CUDA=1 moves every projection and the LM head
 * to the GPU (f32, fmt 0) when built with CUDA=1; COLI_GPU=n picks the device.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include "st.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

typedef struct {
    int hidden, n_layers, n_heads, n_kv, head_dim, inter, vocab, tie;
    float theta, eps;
    /* rope_scaling: 0 none, 1 linear, 2 llama3 */
    int rope_kind; float rs_factor, rs_low, rs_high, rs_orig;
} Cfg;

typedef struct {
    float *w; int I, O;
#ifdef COLI_CUDA
    ColiCudaTensor *cu;
#endif
} Mat;

typedef struct {
    float *attn_norm, *mlp_norm;
    Mat q, k, v, o, gate, up, down;
} Layer;

typedef struct {
    Cfg c; shards S;
    float *embed, *final_norm, *inv_freq;
    Mat lm_head;
    Layer *L;
    float **K, **V; int max_t;
} Model;

static int g_cuda = 0;
static long long g_mm_gpu, g_mm_cpu;   /* matmuls per backend: a silent CPU fallback shows here */
#ifdef COLI_CUDA
static int g_cuda_dev = 0;
#endif

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
static float *falloc(int64_t n) {
    float *p = malloc((size_t)n * sizeof(float));
    if (!p) { fprintf(stderr, "OOM %lld floats\n", (long long)n); exit(1); }
    return p;
}

/* ---------- config ---------- */
static double req_num(jval *r, const char *k) {
    jval *v = json_get(r, k);
    if (!v || v->t != J_NUM) { fprintf(stderr, "config.json: missing or non-numeric \"%s\"\n", k); exit(1); }
    return v->num;
}
static double opt_num(jval *r, const char *k, double dflt) {
    jval *v = json_get(r, k); return v && v->t == J_NUM ? v->num : dflt;
}
static int opt_true(jval *r, const char *k) {
    jval *v = json_get(r, k); return v && v->t == J_BOOL && v->boolean;
}
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof path, "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0 || n > (16L << 20)) { fprintf(stderr, "%s: missing or larger than 16 MB\n", path); exit(1); }
    char *buf = malloc((size_t)n + 1); if (!buf) { fprintf(stderr, "OOM reading %s\n", path); exit(1); }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    buf[n] = 0; fclose(f);
    char *arena = NULL; jval *r = json_parse(buf, &arena);
    jval *mt = json_get(r, "model_type");
    if (!mt || mt->t != J_STR || strcmp(mt->str, "llama")) {
        fprintf(stderr, "config.json: model_type is not \"llama\" -- refusing\n"); exit(1); }
    /* Closed feature set: anything this engine does not implement is refused,
     * never approximated. */
    if (opt_true(r, "attention_bias") || opt_true(r, "mlp_bias")) {
        fprintf(stderr, "config.json: attention_bias/mlp_bias not supported -- refusing\n"); exit(1); }
    jval *act = json_get(r, "hidden_act");
    if (act && (act->t != J_STR || strcmp(act->str, "silu"))) {
        fprintf(stderr, "config.json: hidden_act must be silu -- refusing\n"); exit(1); }
    if (opt_num(r, "pretraining_tp", 1) != 1) {
        fprintf(stderr, "config.json: pretraining_tp != 1 not supported -- refusing\n"); exit(1); }
    c->hidden   = (int)req_num(r, "hidden_size");
    c->n_layers = (int)req_num(r, "num_hidden_layers");
    c->n_heads  = (int)req_num(r, "num_attention_heads");
    c->n_kv     = (int)opt_num(r, "num_key_value_heads", c->n_heads);
    c->inter    = (int)req_num(r, "intermediate_size");
    c->vocab    = (int)req_num(r, "vocab_size");
    if (c->hidden < 1 || c->hidden > (1 << 20) || c->n_heads < 1 || c->n_heads > (1 << 12) ||
        c->n_kv < 1 || c->n_kv > c->n_heads || c->n_heads % c->n_kv ||
        c->inter < 1 || c->inter > (1 << 24) || c->vocab < 1 || c->vocab > (1 << 24) ||
        c->n_layers < 1 || c->n_layers > 4096) {
        fprintf(stderr, "config.json: dimension out of range\n"); exit(1); }
    c->head_dim = (int)opt_num(r, "head_dim", c->hidden / c->n_heads);
    if (c->head_dim < 2 || c->head_dim > 1024 || c->head_dim % 2) {
        fprintf(stderr, "config.json: head_dim out of range\n"); exit(1); }
    c->eps   = (float)opt_num(r, "rms_norm_eps", 1e-6);
    c->tie   = opt_true(r, "tie_word_embeddings");
    c->rope_kind = 0;
    /* transformers < 5 writes top-level rope_theta + rope_scaling; 5.x writes
     * one rope_parameters object carrying rope_theta and the scaling keys.
     * Reading only the old form silently ran a 5.x llama3 export unscaled. */
    jval *rs = json_get(r, "rope_parameters");
    if (rs && rs->t != J_OBJ) { fprintf(stderr, "config.json: rope_parameters is not an object\n"); exit(1); }
    if (rs && json_get(r, "rope_scaling") && json_get(r, "rope_scaling")->t != J_NULL) {
        fprintf(stderr, "config.json: both rope_parameters and rope_scaling -- ambiguous, refusing\n"); exit(1); }
    c->theta = (float)opt_num(rs ? rs : r, "rope_theta", opt_num(r, "rope_theta", 10000.0));
    if (!rs) rs = json_get(r, "rope_scaling");
    if (rs && rs->t == J_OBJ) {
        jval *ty = json_get(rs, "rope_type"); if (!ty) ty = json_get(rs, "type");
        const char *k = ty && ty->t == J_STR ? ty->str : "";
        c->rs_factor = (float)opt_num(rs, "factor", 1.0);
        if (!strcmp(k, "default")) c->rope_kind = 0;
        else if (!strcmp(k, "linear")) c->rope_kind = 1;
        else if (!strcmp(k, "llama3")) {
            c->rope_kind = 2;
            c->rs_low  = (float)req_num(rs, "low_freq_factor");
            c->rs_high = (float)req_num(rs, "high_freq_factor");
            c->rs_orig = (float)req_num(rs, "original_max_position_embeddings");
            if (!(c->rs_high > c->rs_low)) { fprintf(stderr, "config.json: llama3 rope_scaling needs high_freq_factor > low_freq_factor\n"); exit(1); }
        } else { fprintf(stderr, "config.json: rope_scaling type \"%s\" not supported -- refusing\n", k); exit(1); }
        if (c->rope_kind && !(c->rs_factor > 0)) { fprintf(stderr, "config.json: rope_scaling factor must be > 0\n"); exit(1); }
    }
    free(buf); free(arena);
}

/* `want` is the element count the forward pass indexes with (config dims).
 * Checked against the header BEFORE allocating or reading: a short tensor
 * would be read past its end at inference, a long one silently truncated.
 * Both are refused by name. */
static float *load_t(Model *m, const char *name, int64_t want) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    if (n != want) {
        fprintf(stderr, "%s: %lld elements, config implies %lld -- refusing\n",
                name, (long long)n, (long long)want); exit(1);
    }
    float *p = falloc(want);
    st_read_f32_cap(&m->S, name, p, want, 0);
    return p;
}
static Mat load_mat(Model *m, const char *name, int O, int I) {
    Mat w; memset(&w, 0, sizeof w);
    w.w = load_t(m, name, (int64_t)O * I); w.O = O; w.I = I;
    return w;
}

/* HF LlamaRotaryEmbedding inv_freq, float32 like the reference. */
static void rope_init(Model *m) {
    Cfg *c = &m->c; int h = c->head_dim / 2;
    m->inv_freq = falloc(h);
    for (int j = 0; j < h; j++) {
        float inv = 1.0f / powf(c->theta, (float)(2 * j) / (float)c->head_dim);
        if (c->rope_kind == 1) inv /= c->rs_factor;
        else if (c->rope_kind == 2) {
            float wl = 2.0f * (float)M_PI / inv;
            float low_wl = c->rs_orig / c->rs_low, high_wl = c->rs_orig / c->rs_high;
            if (wl > low_wl) inv /= c->rs_factor;
            else if (wl >= high_wl) {
                float sm = (c->rs_orig / wl - c->rs_low) / (c->rs_high - c->rs_low);
                inv = (1 - sm) * inv / c->rs_factor + sm * inv;
            }
        }
        m->inv_freq[j] = inv;
    }
}

static void model_init(Model *m, const char *snap) {
    memset(m, 0, sizeof *m);
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    int D = c->hidden, hd = c->head_dim, QD = c->n_heads * hd, KD = c->n_kv * hd;
    m->embed = load_t(m, "model.embed_tokens.weight", (int64_t)c->vocab * D);
    m->final_norm = load_t(m, "model.norm.weight", D);
    if (c->tie) { m->lm_head.w = m->embed; m->lm_head.O = c->vocab; m->lm_head.I = D; }
    else m->lm_head = load_mat(m, "lm_head.weight", c->vocab, D);
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define P(s) (snprintf(nm, sizeof nm, "model.layers.%d." s, i), nm)
        l->attn_norm = load_t(m, P("input_layernorm.weight"), D);
        l->mlp_norm  = load_t(m, P("post_attention_layernorm.weight"), D);
        l->q    = load_mat(m, P("self_attn.q_proj.weight"), QD, D);
        l->k    = load_mat(m, P("self_attn.k_proj.weight"), KD, D);
        l->v    = load_mat(m, P("self_attn.v_proj.weight"), KD, D);
        l->o    = load_mat(m, P("self_attn.o_proj.weight"), D, QD);
        l->gate = load_mat(m, P("mlp.gate_proj.weight"), c->inter, D);
        l->up   = load_mat(m, P("mlp.up_proj.weight"), c->inter, D);
        l->down = load_mat(m, P("mlp.down_proj.weight"), D, c->inter);
        #undef P
    }
    rope_init(m);
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c; m->max_t = max_t;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_kv * max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_kv * max_t * c->head_dim);
    }
}

/* ---------- math ---------- */
/* y[S,O] = x[S,I] @ W^T, W [O,I] row-major */
static void matmul(float *y, const float *x, Mat *W, int S) {
    int I = W->I, O = W->O;
#ifdef COLI_CUDA
    if (g_cuda && coli_cuda_matmul(&W->cu, y, x, W->w, NULL, 0, S, I, O, g_cuda_dev, 0)) { g_mm_gpu++; return; }
#endif
    g_mm_cpu++;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W->w + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            #pragma omp simd reduction(+:acc)
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ss = 0; for (int i = 0; i < D; i++) ss += (double)x[i] * x[i];
    float r = (float)(1.0 / sqrt(ss / D + eps));
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}
/* rotate_half convention: pairs (j, j+h) */
static void rope_head(float *x, int pos, const Model *m) {
    int h = m->c.head_dim / 2;
    for (int j = 0; j < h; j++) {
        float ang = (float)pos * m->inv_freq[j];
        float cs = cosf(ang), sn = sinf(ang), a = x[j], b = x[j + h];
        x[j] = a * cs - b * sn; x[j + h] = b * cs + a * sn;
    }
}

static void attention(Model *m, Layer *l, int layer, const float *xn, int S, int pos0, float *out) {
    Cfg *c = &m->c; int H = c->n_heads, KV = c->n_kv, hd = c->head_dim, grp = H / KV;
    int QD = H * hd, KD = KV * hd;
    float *q = falloc((int64_t)S * QD), *k = falloc((int64_t)S * KD), *v = falloc((int64_t)S * KD);
    matmul(q, xn, &l->q, S); matmul(k, xn, &l->k, S); matmul(v, xn, &l->v, S);
    for (int s = 0; s < S; s++) {
        int t = pos0 + s;
        for (int h = 0; h < H; h++) rope_head(q + (int64_t)s * QD + h * hd, t, m);
        for (int h = 0; h < KV; h++) {
            rope_head(k + (int64_t)s * KD + h * hd, t, m);
            memcpy(m->K[layer] + ((int64_t)h * m->max_t + t) * hd, k + (int64_t)s * KD + h * hd, hd * sizeof(float));
            memcpy(m->V[layer] + ((int64_t)h * m->max_t + t) * hd, v + (int64_t)s * KD + h * hd, hd * sizeof(float));
        }
    }
    float *ctx = falloc((int64_t)S * QD);
    float scale = 1.0f / sqrtf((float)hd);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        int Tk = pos0 + s + 1;   /* causal: keys 0..pos0+s */
        const float *qh = q + (int64_t)s * QD + h * hd;
        const float *Kh = m->K[layer] + (int64_t)(h / grp) * m->max_t * hd;
        const float *Vh = m->V[layer] + (int64_t)(h / grp) * m->max_t * hd;
        float *sc = falloc(Tk), mx = -INFINITY;
        for (int t = 0; t < Tk; t++) {
            float d = 0; for (int i = 0; i < hd; i++) d += qh[i] * Kh[(int64_t)t * hd + i];
            sc[t] = d * scale; if (sc[t] > mx) mx = sc[t];
        }
        double sum = 0; for (int t = 0; t < Tk; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        float *o = ctx + (int64_t)s * QD + h * hd;
        for (int i = 0; i < hd; i++) o[i] = 0;
        for (int t = 0; t < Tk; t++) {
            float p = (float)(sc[t] / sum); const float *vt = Vh + (int64_t)t * hd;
            for (int i = 0; i < hd; i++) o[i] += p * vt[i];
        }
        free(sc);
    }
    matmul(out, ctx, &l->o, S);
    free(q); free(k); free(v); free(ctx);
}

static void mlp(Model *m, Layer *l, const float *xn, int S, float *out) {
    int F = m->c.inter;
    float *g = falloc((int64_t)S * F), *u = falloc((int64_t)S * F);
    matmul(g, xn, &l->gate, S); matmul(u, xn, &l->up, S);
    for (int64_t i = 0; i < (int64_t)S * F; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
    matmul(out, g, &l->down, S);
    free(g); free(u);
}

/* Runs S new tokens at positions pos0..pos0+S-1; returns logits of the last one. */
static float *step(Model *m, const int *ids, int S, int pos0) {
    Cfg *c = &m->c; int D = c->hidden;
    if (pos0 + S > m->max_t) { fprintf(stderr, "context %d exceeds KV capacity %d\n", pos0 + S, m->max_t); exit(1); }
    float *x = falloc((int64_t)S * D), *xn = falloc((int64_t)S * D), *t = falloc((int64_t)S * D);
    for (int s = 0; s < S; s++) {
        if (ids[s] < 0 || ids[s] >= c->vocab) { fprintf(stderr, "token id %d out of vocab\n", ids[s]); exit(1); }
        memcpy(x + (int64_t)s * D, m->embed + (int64_t)ids[s] * D, D * sizeof(float));
    }
    for (int li = 0; li < c->n_layers; li++) {
        Layer *l = &m->L[li];
        for (int s = 0; s < S; s++) rmsnorm_row(xn + (int64_t)s * D, x + (int64_t)s * D, l->attn_norm, D, c->eps);
        attention(m, l, li, xn, S, pos0, t);
        for (int64_t i = 0; i < (int64_t)S * D; i++) x[i] += t[i];
        for (int s = 0; s < S; s++) rmsnorm_row(xn + (int64_t)s * D, x + (int64_t)s * D, l->mlp_norm, D, c->eps);
        mlp(m, l, xn, S, t);
        for (int64_t i = 0; i < (int64_t)S * D; i++) x[i] += t[i];
    }
    rmsnorm_row(xn, x + (int64_t)(S - 1) * D, m->final_norm, D, c->eps);
    float *logits = falloc(c->vocab);
    matmul(logits, xn, &m->lm_head, 1);
    free(x); free(xn); free(t);
    return logits;
}

static int argmax(const float *x, int n) {
    int b = 0; for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i; return b;
}

/* ---------- ref.json harness ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR || a->len < 1) { fprintf(stderr, "ref: missing array \"%s\"\n", key); exit(1); }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "usage: SNAP=<llama checkpoint dir> %s [ref.json]\n", argv[0]); return 1; }
    const char *refpath = argc > 1 ? argv[1] : "ref.json";
    FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1); if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "%s: short read\n", refpath); return 1; }
    buf[n] = 0; fclose(f);
    char *arena = NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull; int *prompt = read_int_array(ref, "prompt_ids", &np), *full = read_int_array(ref, "full_ids", &nfull);
    int n_new = nfull - np;
    if (n_new < 1) { fprintf(stderr, "ref: full_ids must extend prompt_ids\n"); return 1; }
#ifdef COLI_CUDA
    if (getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))) {
        g_cuda_dev = getenv("COLI_GPU") ? atoi(getenv("COLI_GPU")) : 0;
        g_cuda = coli_cuda_init(&g_cuda_dev, 1);
        if (!g_cuda) fprintf(stderr, "[CUDA] init failed -- running on CPU\n");
    }
#endif
    static Model m;
    double t0 = now_s();
    model_init(&m, snap);
    kv_alloc(&m, nfull);
    printf("llama: %d layers, hidden %d, heads %d/%d kv, head_dim %d, ffn %d, vocab %d, tie %d, rope %d\n",
           m.c.n_layers, m.c.hidden, m.c.n_heads, m.c.n_kv, m.c.head_dim, m.c.inter, m.c.vocab, m.c.tie, m.c.rope_kind);
    printf("weights loaded in %.1fs | RSS %.2f GB | backend %s\n", now_s() - t0, rss_gb(), g_cuda ? "cuda" : "cpu");

    int *out = malloc(nfull * sizeof(int));
    memcpy(out, prompt, np * sizeof(int));
    double t1 = now_s();
    float *lg = step(&m, prompt, np, 0);
    double t_prefill = now_s() - t1;
    for (int i = 0; i < n_new; i++) {
        out[np + i] = argmax(lg, m.c.vocab); free(lg);
        if (i + 1 < n_new) lg = step(&m, &out[np + i], 1, np + i);
    }
    double dt = now_s() - t1;
    int match = 0;
    printf("\nReference: "); for (int i = np; i < nfull; i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i = np; i < nfull; i++) { printf("%d ", out[i]); if (out[i] == full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    printf("Prefill: %d tokens in %.3fs\n", np, t_prefill);
    printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", n_new / dt, dt, n_new, rss_gb());
    printf("TUNE decode: %d tokens in %.3fs\n", n_new, dt);
    printf("matmuls: gpu %lld, cpu %lld\n", g_mm_gpu, g_mm_cpu);
#ifdef COLI_CUDA
    if (g_cuda) { size_t tn = 0, tb = 0; coli_cuda_stats(-1, &tn, &tb);
        printf("[CUDA] resident: %zu tensors, %.2f GB\n", tn, tb / 1e9); }
#endif
    free(buf); free(arena); free(out); free(prompt); free(full);
    return match == n_new ? 0 : 2;
}
