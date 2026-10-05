/* qwen36_mtp.h -- the multi-token prediction (MTP) head of Qwen3.6, included by
 * qwen36.c. Q36_MTP=<gguf> loads the checkpoint's one trained MTP block from a GGUF
 * that carries it (llama.cpp's nextn layer, e.g. mtp-Qwen3.6-35B-A3B-Q4_0.gguf from
 * ggml-org/Qwen3.6-35B-A3B-GGUF) and the speculative step (q36_spec_step) drafts one
 * token from it, beside prompt lookup; the gate (spec_draft.h) picks the source.
 *
 * One head row at position p (llama.cpp's qwen35moe graph_mtp):
 *   x  = eh_proj([enorm(embed(tok_p)), hnorm(h_{p-1})])     h: the model's row after the final norm
 *   x += attention(attn_norm(x))                             gated attention, the head's own KV rows
 *   x += moe(post_attention_norm(x))                         top-k routed experts + gated shared expert
 *   logits_{p+1} = lm_head(shared_head_norm(x))              the model's lm_head
 * The head is fed by every forward (step_ex: q36_mtp_feed) with the rows it computed:
 * row p of a forward completes the head's row p+1 once token p+1 is known, so the last
 * row waits (pending) for the token the caller picks next; q36_mtp_draft runs that row
 * with the picked token and names its argmax. A rewind (a verify's rejected rows) only
 * moves where the head continues: the head rows past it are a stale tail the next rows
 * overwrite, and the hidden row the head needs there is the forward's own.
 * The head runs on the CPU, but for lm_head (the chain's, on the device when it holds it);
 * its matrices are int4 in groups of 64 (Q36_MTP_BITS=8: int8 rows), the GGUF's Q4_0
 * decoded first. A fed row only stores its K/V row: nothing else of it is read later. */

#include <stdio.h>

/* ---------------- a minimal GGUF v3 reader (tensor infos; f32, f16 and Q4_0 data) ---------------- */
typedef struct { char name[96]; int nd; uint64_t ne[4]; uint32_t type; uint64_t off; } Q36GgTensor;
typedef struct { FILE *f; uint64_t base; int n; Q36GgTensor *t; } Q36Gguf;

static int q36gg_skip(FILE *f, uint32_t t) {
    static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t == 8) { uint64_t n; if (fread(&n, 8, 1, f) != 1) return 0; return fseek(f, (long)n, SEEK_CUR) == 0; }
    if (t == 9) {
        uint32_t et; uint64_t n;
        if (fread(&et, 4, 1, f) != 1 || fread(&n, 8, 1, f) != 1) return 0;
        for (uint64_t i = 0; i < n; i++) if (!q36gg_skip(f, et)) return 0;
        return 1;
    }
    if (t > 12) return 0;
    return fseek(f, sz[t], SEEK_CUR) == 0;
}
static int q36gg_open(Q36Gguf *g, const char *path) {
    memset(g, 0, sizeof *g);
    g->f = fopen(path, "rb");
    if (!g->f) return 0;
    uint32_t magic, ver, align = 32; uint64_t nt, nkv;
    if (fread(&magic, 4, 1, g->f) != 1 || magic != 0x46554747u || fread(&ver, 4, 1, g->f) != 1 ||
        fread(&nt, 8, 1, g->f) != 1 || fread(&nkv, 8, 1, g->f) != 1 || nt > (1u << 20)) return 0;
    for (uint64_t i = 0; i < nkv; i++) {
        uint64_t kl; char key[256]; uint32_t t;
        if (fread(&kl, 8, 1, g->f) != 1 || kl >= sizeof key || fread(key, 1, kl, g->f) != kl) return 0;
        key[kl] = 0;
        if (fread(&t, 4, 1, g->f) != 1) return 0;
        if (!strcmp(key, "general.alignment") && t == 4) { if (fread(&align, 4, 1, g->f) != 1 || !align) return 0; }
        else if (!q36gg_skip(g->f, t)) return 0;
    }
    g->t = calloc(nt, sizeof(Q36GgTensor)); g->n = (int)nt;
    if (!g->t) return 0;
    for (int i = 0; i < g->n; i++) {
        uint64_t nl; Q36GgTensor *x = &g->t[i];
        if (fread(&nl, 8, 1, g->f) != 1 || nl >= sizeof x->name || fread(x->name, 1, nl, g->f) != nl) return 0;
        x->name[nl] = 0;
        if (fread(&x->nd, 4, 1, g->f) != 1 || x->nd < 1 || x->nd > 4) return 0;
        for (int k = 0; k < 4; k++) x->ne[k] = 1;
        for (int k = 0; k < x->nd; k++) if (fread(&x->ne[k], 8, 1, g->f) != 1) return 0;
        if (fread(&x->type, 4, 1, g->f) != 1 || fread(&x->off, 8, 1, g->f) != 1) return 0;
    }
    long pos = ftell(g->f);
    g->base = ((uint64_t)pos + align - 1) / align * align;
    return 1;
}
static void q36gg_close(Q36Gguf *g) { if (g->f) fclose(g->f); free(g->t); memset(g, 0, sizeof *g); }
static Q36GgTensor *q36gg_find(Q36Gguf *g, const char *name) {
    for (int i = 0; i < g->n; i++) if (!strcmp(g->t[i].name, name)) return &g->t[i];
    return NULL;
}
static float q36gg_h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, mm = h & 1023, u;
    if (e == 0) {
        if (!mm) u = s;
        else { e = 1; while (!(mm & 1024)) { mm <<= 1; e--; } mm &= 1023; u = s | ((e + 112) << 23) | (mm << 13); }
    } else if (e == 31) u = s | 0x7f800000u | (mm << 13);
    else u = s | ((e + 112) << 23) | (mm << 13);
    float f; memcpy(&f, &u, 4); return f;
}
/* elements [e0, e0 + n) of t (ne0 fastest) as f32; Q4_0 needs e0 and n multiples of 32 */
static float *q36gg_f32(Q36Gguf *g, Q36GgTensor *t, uint64_t e0, uint64_t n) {
    uint64_t total = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    if (e0 + n > total) return NULL;
    float *out = malloc(n * sizeof(float));
    if (!out) return NULL;
    if (t->type == 0) {           /* F32 */
        if (fseek(g->f, (long)(g->base + t->off + e0 * 4), SEEK_SET) || fread(out, 4, n, g->f) != n) { free(out); return NULL; }
    } else if (t->type == 1) {    /* F16 */
        uint16_t *raw = malloc(n * 2);
        if (!raw || fseek(g->f, (long)(g->base + t->off + e0 * 2), SEEK_SET) || fread(raw, 2, n, g->f) != n) { free(raw); free(out); return NULL; }
        for (uint64_t i = 0; i < n; i++) out[i] = q36gg_h2f(raw[i]);
        free(raw);
    } else if (t->type == 2) {    /* Q4_0: per 32 values an f16 scale and 16 bytes of nibbles (v - 8) */
        if (e0 % 32 || n % 32) { free(out); return NULL; }
        uint64_t nb = n / 32;
        uint8_t *raw = malloc(nb * 18);
        if (!raw || fseek(g->f, (long)(g->base + t->off + e0 / 32 * 18), SEEK_SET) || fread(raw, 18, nb, g->f) != nb) { free(raw); free(out); return NULL; }
        for (uint64_t b = 0; b < nb; b++) {
            uint16_t hs; memcpy(&hs, raw + b * 18, 2);
            float d = q36gg_h2f(hs); const uint8_t *q = raw + b * 18 + 2;
            for (int j = 0; j < 16; j++) {
                out[b * 32 + j] = ((q[j] & 15) - 8) * d;
                out[b * 32 + 16 + j] = ((q[j] >> 4) - 8) * d;
            }
        }
        free(raw);
    } else { free(out); return NULL; }
    return out;
}

/* ---------------- the head ---------------- */
typedef struct {
    int on, E, K, I, ish;
    Layer L;                         /* q, k, v, o, qn, kn, in_ln (attn_norm), post_ln: the engine's (1 + w) norms */
    QW eh;                           /* [D x 2D] */
    float *enorm, *hnorm, *head_norm;
    float *router;                   /* [E][D] */
    float *sh_gate;                  /* [D] */
    QW sh_g, sh_u, sh_d;
    QW *eg, *eu, *ed;                /* [E] */
    float *Kc, *Vc; int stride;      /* the head's KV rows [KV][stride][hd] */
    float **Kp, **Vp;                /* the model's per-layer pointers and the head's after them */
    /* where the head is: rows up to `len` read; `pend` the model's row at len (after the
     * final norm), waiting for the token at len + 1; `done`: the row at done_pos already
     * run by a draft with token done_tok */
    int len, has_pend, done_pos, done_tok;
    float *pend;
    float *last; int last_base, last_n, last_cap;   /* the newest forward's rows (a rewind lands in them) */
    unsigned long long rows, drafts, kv_rows;
    double ms, kv_ms;
} Q36Mtp;
static Q36Mtp g_q36_mtp;

/* GGUF norms hold HF's 1 + w; the engine's rmsnorm_row adds the 1 */
static float *q36_mtp_norm(Q36Gguf *g, const char *name, int n) {
    Q36GgTensor *t = q36gg_find(g, name);
    if (!t || (int)(t->ne[0] * t->ne[1]) != n) return NULL;
    float *v = q36gg_f32(g, t, 0, (uint64_t)n);
    if (v) for (int i = 0; i < n; i++) v[i] -= 1.f;
    return v;
}
static int q36_mtp_qw(Q36Gguf *g, const char *name, int I, int O, uint64_t e0, QW *out) {
    Q36GgTensor *t = q36gg_find(g, name);
    if (!t || (int)t->ne[0] != I) return 0;
    float *w = q36gg_f32(g, t, e0, (uint64_t)I * O);
    if (!w) return 0;
    memset(out, 0, sizeof *out);
    qw_quantize(w, I, O, "mtp", out);
    /* Q36_MTP_BITS (4 by default: the GGUF holds Q4_0 anyway) -- the head's own width,
     * whatever the trunk's: int4 in groups of 64 when the rows allow it */
    static int bits = -1;
    if (bits < 0) { const char *e = getenv("Q36_MTP_BITS"); bits = e && *e ? atoi(e) : 4; }
    if (bits == 4 && !out->q4 && I % 64 == 0) {
        uint8_t *q4 = malloc((size_t)O * (I / 2)); float *sg = malloc((size_t)O * (I / 64) * sizeof(float));
        if (q4 && sg) {
            pack_int4_g64_planar(w, q4, sg, O, I);
            out->q4 = q4; out->sg = sg; out->ng = I / 64;
            q36_wfree(out->q); free(out->sc); out->q = NULL; out->sc = NULL;
        } else { free(q4); free(sg); }
    }
    free(w);
    return 1;
}
static int q36_mtp_load(Model *m, const char *path) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, I = c->inter, Ish = c->shared_inter;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    if (E <= 0 || Ish <= 0) { fprintf(stderr, "[mtp] Q36_MTP: the head needs a MoE model with a shared expert\n"); return 0; }
    Q36Gguf g;
    if (!q36gg_open(&g, path)) { fprintf(stderr, "[mtp] %s: not a readable GGUF\n", path); q36gg_close(&g); return 0; }
    char nm[128]; int L = -1;
    for (int i = 0; i < g.n; i++)
        if (!strncmp(g.t[i].name, "blk.", 4) && strstr(g.t[i].name, ".nextn.eh_proj")) L = atoi(g.t[i].name + 4);
    if (L < 0) { fprintf(stderr, "[mtp] %s: no nextn block\n", path); q36gg_close(&g); return 0; }
    double t0 = now_s();
    Q36Mtp *p = &g_q36_mtp; memset(p, 0, sizeof *p);
    p->E = E; p->K = c->topk; p->I = I; p->ish = Ish;
    int ok = 1;
#define N(s) (snprintf(nm, sizeof nm, "blk.%d.%s", L, s), nm)
    ok &= !!(p->enorm = q36_mtp_norm(&g, N("nextn.enorm.weight"), D));
    ok &= !!(p->hnorm = q36_mtp_norm(&g, N("nextn.hnorm.weight"), D));
    ok &= !!(p->head_norm = q36_mtp_norm(&g, N("nextn.shared_head_norm.weight"), D));
    ok &= !!(p->L.in_ln = q36_mtp_norm(&g, N("attn_norm.weight"), D));
    ok &= !!(p->L.post_ln = q36_mtp_norm(&g, N("post_attention_norm.weight"), D));
    ok &= !!(p->L.qn = q36_mtp_norm(&g, N("attn_q_norm.weight"), hd));
    ok &= !!(p->L.kn = q36_mtp_norm(&g, N("attn_k_norm.weight"), kvd));
    ok &= q36_mtp_qw(&g, N("nextn.eh_proj.weight"), 2 * D, D, 0, &p->eh);
    ok &= q36_mtp_qw(&g, N("attn_q.weight"), D, H * qdim, 0, &p->L.q);
    ok &= q36_mtp_qw(&g, N("attn_k.weight"), D, KV * kvd, 0, &p->L.k);
    ok &= q36_mtp_qw(&g, N("attn_v.weight"), D, KV * kvd, 0, &p->L.v);
    ok &= q36_mtp_qw(&g, N("attn_output.weight"), H * hd, D, 0, &p->L.o);
    ok &= q36_mtp_qw(&g, N("ffn_gate_shexp.weight"), D, Ish, 0, &p->sh_g);
    ok &= q36_mtp_qw(&g, N("ffn_up_shexp.weight"), D, Ish, 0, &p->sh_u);
    ok &= q36_mtp_qw(&g, N("ffn_down_shexp.weight"), Ish, D, 0, &p->sh_d);
    Q36GgTensor *tr = q36gg_find(&g, N("ffn_gate_inp.weight")), *ts = q36gg_find(&g, N("ffn_gate_inp_shexp.weight"));
    ok &= tr && ts && (p->router = q36gg_f32(&g, tr, 0, (uint64_t)E * D)) && (p->sh_gate = q36gg_f32(&g, ts, 0, (uint64_t)D));
    p->eg = calloc(E, sizeof(QW)); p->eu = calloc(E, sizeof(QW)); p->ed = calloc(E, sizeof(QW));
    ok &= p->eg && p->eu && p->ed;
    for (int e = 0; ok && e < E; e++) {
        ok &= q36_mtp_qw(&g, N("ffn_gate_exps.weight"), D, I, (uint64_t)e * I * D, &p->eg[e]);
        ok &= q36_mtp_qw(&g, N("ffn_up_exps.weight"), D, I, (uint64_t)e * I * D, &p->eu[e]);
        ok &= q36_mtp_qw(&g, N("ffn_down_exps.weight"), I, D, (uint64_t)e * D * I, &p->ed[e]);
    }
#undef N
    q36gg_close(&g);
    if (!ok) { fprintf(stderr, "[mtp] %s: block %d incomplete, or its shapes are not this model's\n", path, L); return 0; }
    p->pend = falloc(D);
    p->on = 1;
    fprintf(stderr, "[mtp] Qwen3.6 MTP head (block %d of %s): %d experts, loaded in %.1f s\n", L, path, E, now_s() - t0);
    return 1;
}

/* the head's KV rows at the model's row stride (attention() indexes K[layer] by max_t),
 * and the pointer tables that put them after the model's layers */
static int q36_mtp_kv(Model *m) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp;
    if (p->stride != m->max_t || !p->Kc) {
        free(p->Kc); free(p->Vc);
        size_t n = (size_t)c->kv_heads * m->max_t * c->k_head_dim;
        p->Kc = calloc(n, sizeof(float)); p->Vc = calloc(n, sizeof(float)); p->stride = m->max_t;
        if (!p->Kc || !p->Vc) { p->on = 0; return 0; }
        p->len = 0; p->has_pend = 0; p->done_pos = -1;
    }
    if (!p->Kp) {
        p->Kp = malloc(sizeof(float *) * (size_t)(c->n_layers + 1));
        p->Vp = malloc(sizeof(float *) * (size_t)(c->n_layers + 1));
        if (!p->Kp || !p->Vp) { p->on = 0; return 0; }
    }
    for (int i = 0; i < c->n_layers; i++) { p->Kp[i] = m->K[i]; p->Vp[i] = m->V[i]; }
    p->Kp[c->n_layers] = p->Kc; p->Vp[c->n_layers] = p->Vc;
    return 1;
}

/* routed experts (softmax top-K, renormalised) + the sigmoid-gated shared expert;
 * a prompt's rows grouped per expert, so its catch-up is a few batched matmuls */
static void q36_mtp_moe(const float *x, int R, int D, float *out) {
    Q36Mtp *p = &g_q36_mtp; int E = p->E, K = p->K, I = p->I, Ish = p->ish;
    int *idx = malloc(sizeof(int) * (size_t)R * K); float *val = falloc((int64_t)R * K), *lg = falloc((int64_t)R * E);
    /* the router's logits: every (row, expert) dot apart */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < R * E; j++) {
        const float *xr = x + (int64_t)(j / E) * D, *w = p->router + (int64_t)(j % E) * D;
        float a = 0; for (int i = 0; i < D; i++) a += xr[i] * w[i];
        lg[j] = a;
    }
    #pragma omp parallel for schedule(static) if (R > 1)
    for (int r = 0; r < R; r++) {
        float *l = lg + (int64_t)r * E;
        softmax_row(l, E);
        for (int k = 0; k < K; k++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < k; j++) taken |= idx[r * K + j] == e;
                if (!taken && l[e] > bv) { bv = l[e]; best = e; }
            }
            idx[r * K + k] = best < 0 ? k : best; val[r * K + k] = best < 0 ? 0.f : bv;
        }
        float sm = 0; for (int k = 0; k < K; k++) sm += val[r * K + k];
        if (sm > 0) for (int k = 0; k < K; k++) val[r * K + k] /= sm;
    }
    memset(out, 0, sizeof(float) * (size_t)R * D);
    {   /* the shared expert, every row */
        float *g = falloc((int64_t)R * Ish), *u = falloc((int64_t)R * Ish), *y = falloc((int64_t)R * D);
        matmul_d(g, x, &p->sh_g, R, D, Ish); matmul_d(u, x, &p->sh_u, R, D, Ish);
        for (int64_t i = 0; i < (int64_t)R * Ish; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
        matmul_d(y, g, &p->sh_d, R, Ish, D);
        for (int r = 0; r < R; r++) {
            const float *xr = x + (int64_t)r * D; float a = 0; for (int i = 0; i < D; i++) a += xr[i] * p->sh_gate[i];
            float sg = 1.f / (1.f + expf(-a));
            for (int i = 0; i < D; i++) out[(int64_t)r * D + i] += sg * y[(int64_t)r * D + i];
        }
        free(g); free(u); free(y);
    }
    if (R * K <= 64) {   /* a decode row's experts: one per thread (the matmuls inside run serially) */
        float *ye = falloc((int64_t)R * K * D);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < R * K; j++) {
            int e = idx[j];
            float *g = falloc(I), *u = falloc(I);
            const float *xr = x + (int64_t)(j / K) * D;
            matmul_d(g, xr, &p->eg[e], 1, D, I); matmul_d(u, xr, &p->eu[e], 1, D, I);
            for (int i = 0; i < I; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
            matmul_d(ye + (int64_t)j * D, g, &p->ed[e], 1, I, D);
            free(g); free(u);
        }
        for (int j = 0; j < R * K; j++) {   /* rank order */
            float w = val[j], *o = out + (int64_t)(j / K) * D; const float *yr = ye + (int64_t)j * D;
            for (int i = 0; i < D; i++) o[i] += w * yr[i];
        }
        free(ye); free(idx); free(val); free(lg);
        return;
    }
    int *rows = malloc(sizeof(int) * (size_t)R * K);
    float *xe = falloc((int64_t)R * K * D), *ge = falloc((int64_t)R * K * I), *ue = falloc((int64_t)R * K * I), *ye = falloc((int64_t)R * K * D);
    for (int e = 0; e < E; e++) {   /* the routed experts, each over its rows */
        int n = 0;
        for (int i = 0; i < R * K; i++) if (idx[i] == e) rows[n++] = i;
        if (!n) continue;
        for (int j = 0; j < n; j++) memcpy(xe + (int64_t)j * D, x + (int64_t)(rows[j] / K) * D, sizeof(float) * D);
        matmul_d(ge, xe, &p->eg[e], n, D, I); matmul_d(ue, xe, &p->eu[e], n, D, I);
        for (int64_t i = 0; i < (int64_t)n * I; i++) ge[i] = ge[i] / (1.f + expf(-ge[i])) * ue[i];
        matmul_d(ye, ge, &p->ed[e], n, I, D);
        for (int j = 0; j < n; j++) {
            float w = val[rows[j]]; float *o = out + (int64_t)(rows[j] / K) * D; const float *yr = ye + (int64_t)j * D;
            for (int i = 0; i < D; i++) o[i] += w * yr[i];
        }
    }
    free(idx); free(val); free(lg); free(rows); free(xe); free(ge); free(ue); free(ye);
}

/* The model's lm_head on one normed row: on the device when the chain holds it */
static void q36_mtp_lm_head(Model *m, const float *row, float *logits);

/* A fed row's only lasting effect is its K/V row (the head reads nothing else of it
 * later): eh_proj, the input norm, k and v, the k norm and RoPE, into the head's cache,
 * as attention() stores them. */
static void q36_mtp_kv_rows(Model *m, const int *tok, const float *hid, int R, int pos0) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp; int D = c->hidden, KV = c->kv_heads, kvd = c->k_head_dim, kvo = KV * kvd;
    double t0 = now_s();
    float *cat = falloc((int64_t)R * 2 * D), *x = falloc((int64_t)R * D), *nrm = falloc((int64_t)R * D);
    float *k = falloc((int64_t)R * kvo), *v = falloc((int64_t)R * kvo);
    for (int r = 0; r < R; r++) {
        rmsnorm_row(cat + (int64_t)r * 2 * D, m->embed + (int64_t)tok[r] * D, p->enorm, D, c->eps);
        rmsnorm_row(cat + (int64_t)r * 2 * D + D, hid + (int64_t)r * D, p->hnorm, D, c->eps);
    }
    matmul_d(x, cat, &p->eh, R, 2 * D, D);
    for (int r = 0; r < R; r++) rmsnorm_row(nrm + (int64_t)r * D, x + (int64_t)r * D, p->L.in_ln, D, c->eps);
    matmul_d(k, nrm, &p->L.k, R, D, kvo); matmul_d(v, nrm, &p->L.v, R, D, kvo);
    for (int r = 0; r < R; r++) for (int h = 0; h < KV; h++) {
        float *kh = k + (int64_t)r * kvo + h * kvd;
        if (p->L.kn) rmsnorm_row(kh, kh, p->L.kn, kvd, c->eps);
        if (m->mpos || m->rope_delta) { int p3[3]; mrope_at(m, pos0 + r, p3); rope_head_mrope(kh, p3, c->mrope_section, c->rotary_dim, c->theta); }
        else rope_head_partial(kh, pos0 + r, c->rotary_dim, kvd, c->theta);
        size_t dst = ((size_t)h * m->max_t + pos0 + r) * kvd;
        memcpy(p->Kc + dst, kh, (size_t)kvd * sizeof(float));
        memcpy(p->Vc + dst, v + (int64_t)r * kvo + h * kvd, (size_t)kvd * sizeof(float));
    }
    free(cat); free(x); free(nrm); free(k); free(v);
    p->kv_rows += (unsigned long long)R; p->kv_ms += (now_s() - t0) * 1e3;
}

/* R head rows at positions pos0 .. pos0+R-1: token tok[r] with the model's row hid[r]
 * (its row at pos0+r-1, after the final norm); the last row's logits when logits.
 * Without logits only the rows' K/V matter (q36_mtp_kv_rows). */
static void q36_mtp_rows(Model *m, const int *tok, const float *hid, int R, int pos0, float *logits, float *hid_out) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp; int D = c->hidden;
    if (!q36_mtp_kv(m)) return;
    if (!logits) { q36_mtp_kv_rows(m, tok, hid, R, pos0); return; }
    double t0 = now_s();
    float *cat = falloc((int64_t)R * 2 * D), *x = falloc((int64_t)R * D), *nrm = falloc((int64_t)R * D), *tmp = falloc((int64_t)R * D);
    for (int r = 0; r < R; r++) {
        rmsnorm_row(cat + (int64_t)r * 2 * D, m->embed + (int64_t)tok[r] * D, p->enorm, D, c->eps);
        rmsnorm_row(cat + (int64_t)r * 2 * D + D, hid + (int64_t)r * D, p->hnorm, D, c->eps);
    }
    matmul_d(x, cat, &p->eh, R, 2 * D, D);
    for (int r = 0; r < R; r++) rmsnorm_row(nrm + (int64_t)r * D, x + (int64_t)r * D, p->L.in_ln, D, c->eps);
    float **K0 = m->K, **V0 = m->V;
    m->K = p->Kp; m->V = p->Vp;
    attention(m, &p->L, c->n_layers, nrm, R, pos0, tmp);
    m->K = K0; m->V = V0;
    for (int64_t i = 0; i < (int64_t)R * D; i++) x[i] += tmp[i];
    for (int r = 0; r < R; r++) rmsnorm_row(nrm + (int64_t)r * D, x + (int64_t)r * D, p->L.post_ln, D, c->eps);
    q36_mtp_moe(nrm, R, D, tmp);
    for (int64_t i = 0; i < (int64_t)R * D; i++) x[i] += tmp[i];
    if (logits) {
        float *last = falloc(D);
        rmsnorm_row(last, x + (int64_t)(R - 1) * D, p->head_norm, D, c->eps);
        q36_mtp_lm_head(m, last, logits);
        /* the row's own state for a deeper draft, standing in for the model's row (only
         * the verify computes that): after the head's norm, or Q36_MTP_CHAIN=raw before it */
        if (hid_out) {
            static int raw = -1;
            if (raw < 0) { const char *e = getenv("Q36_MTP_CHAIN"); raw = e && !strcmp(e, "raw"); }
            memcpy(hid_out, raw ? x + (int64_t)(R - 1) * D : last, sizeof(float) * D);
        }
        free(last);
    }
    free(cat); free(x); free(nrm); free(tmp);
    p->rows += (unsigned long long)R; p->ms += (now_s() - t0) * 1e3;
}

/* After a forward fed ids[0..S) at pos_base with its rows hid[S] (after the final norm):
 * the head reads the rows it completes and keeps the last one pending. A forward that
 * starts behind the head (a verify's rejected rows undone) continues from the forward
 * before's row there; one that leaves a gap stops the head until the next prompt. */
static void q36_mtp_feed(Model *m, const int *ids, int S, int pos_base, const float *hid) {
    Q36Mtp *p = &g_q36_mtp; int D = m->c.hidden;
    if (!p->on || S < 1) return;
    if (pos_base == 0) { p->len = 0; p->has_pend = 0; p->done_pos = -1; }
    else {
        /* the model's row at pos_base - 1: pending, or in the newest forward's rows */
        int prev = pos_base - 1;
        if (!(p->has_pend && p->len == prev)) {
            if (p->last && prev >= p->last_base && prev < p->last_base + p->last_n) {
                memcpy(p->pend, p->last + (int64_t)(prev - p->last_base) * D, sizeof(float) * D);
                p->len = prev; p->has_pend = 1;
            } else { p->has_pend = 0; p->len = -1; }
        }
        if (p->has_pend) {
            if (!(p->done_pos == pos_base && p->done_tok == ids[0]))
                q36_mtp_rows(m, ids, p->pend, 1, pos_base, NULL, NULL);
        }
    }
    p->done_pos = -1;
    if (pos_base == 0 || p->has_pend) {
        if (S > 1) q36_mtp_rows(m, ids + 1, hid, S - 1, pos_base + 1, NULL, NULL);
        memcpy(p->pend, hid + (int64_t)(S - 1) * D, sizeof(float) * D);
        p->len = pos_base + S - 1; p->has_pend = 1;
    }
    if (S > p->last_cap) { free(p->last); p->last = falloc((int64_t)S * D); p->last_cap = S; }
    memcpy(p->last, hid, sizeof(float) * (size_t)S * D);
    p->last_base = pos_base; p->last_n = S;
}

/* Whether the head can draft for a token about to be fed at pos */
static int q36_mtp_ready(int pos) { const Q36Mtp *p = &g_q36_mtp; return p->on && p->has_pend && p->len == pos - 1; }
/* The head's draft for the token after `tok` (about to be fed at pos): its row at pos;
 * `hidden` (D floats, may be NULL) gets the row's state for a deeper draft */
static int q36_mtp_draft(Model *m, int tok, int pos, float *hidden) {
    Q36Mtp *p = &g_q36_mtp; int V = m->c.vocab;
    float *lg = falloc(V);
    q36_mtp_rows(m, &tok, p->pend, 1, pos, lg, hidden);
    p->done_pos = pos; p->done_tok = tok; p->drafts++;
    int best = 0; for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
    free(lg);
    return best;
}

/* A deeper draft: the head's row at `pos` on its own state of the row before (`hidden`,
 * becomes this row's) with the draft just proposed. Its K/V row there is a tail: the
 * fed row the verify's standing rows bring overwrites it. */
static int q36_mtp_draft_more(Model *m, float *hidden, int tok, int pos) {
    Q36Mtp *p = &g_q36_mtp; int V = m->c.vocab;
    float *lg = falloc(V), *h = falloc(m->c.hidden);
    memcpy(h, hidden, sizeof(float) * (size_t)m->c.hidden);
    q36_mtp_rows(m, &tok, h, 1, pos, lg, hidden);
    p->drafts++;
    int best = 0; for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
    free(lg); free(h);
    return best;
}

/* Q36_MTP=<gguf>: the head, after the model and the device are up */
static void q36_mtp_attach(Model *m) {
    const char *path = getenv("Q36_MTP");
    if (!path || !*path || g_q36_mtp.on) return;
    if (!q36_mtp_load(m, path)) { fprintf(stderr, "[mtp] Q36_MTP=%s: the head stays off\n", path); return; }
}
static void q36_mtp_report(void) {
    const Q36Mtp *p = &g_q36_mtp;
    if (!p->on) return;
    fprintf(stderr, "[mtp] head rows %llu (%.2f ms each), fed rows %llu (K/V only, %.2f ms each), drafts %llu\n", p->rows,
            p->rows ? p->ms / (double)p->rows : 0.0, p->kv_rows, p->kv_rows ? p->kv_ms / (double)p->kv_rows : 0.0, p->drafts);
}
