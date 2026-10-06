/* Qwen3.6 checkpoint MTP head, included by qwen36.c.
 * Pair the backbone's output h_p (after its final norm, as vLLM's Qwen3-Next MTP takes
 * the target model's hidden states) with embed(token_{p+1}):
 * fc(cat(pre_fc_norm_embedding(embed), pre_fc_norm_hidden(h))), one gated
 * full-attention/MoE decoder layer, mtp.norm and the shared lm_head.
 * tools/qwen36_mtp_ref.py checks this against Qwen's transformers decoder.
 * The head owns its KV cache; verification alone commits backbone tokens. */

typedef struct {
    int on, E, K, I, ish;
    Layer L;                         /* q, k, v, o, qn, kn, in_ln (attn_norm), post_ln: the engine's (1 + w) norms */
    QW eh;                           /* [D x 2D] */
    float *enorm, *hnorm, *head_norm;
    float *router;                   /* [E][D] */
    float *sh_gate;                  /* [D] */
    QW sh_g, sh_u, sh_d;
    QW *eg, *eu, *ed;                /* [E] */
    QW dvoc; int nvoc;               /* the shared lm_head rows selected by vid */
    float last_p;                    /* the newest draft's probability under the head */
    int pick_id, pick_ok;            /* q36_mg_row's compact pick (a listed vocabulary) */
    int *vid;                        /* Q36_MTP_VOCAB_IDS: row j of dvoc scores id vid[j] (NULL: id j) */
    float *Kc, *Vc; int stride;      /* the head's KV rows [KV][stride][hd] */
    float **Kp, **Vp;                /* the model's per-layer pointers and the head's after them */
    /* where the head is: rows up to `len` read; `pend` the model's row at len (before the
     * final norm), waiting for the token at len + 1; `done`: the row at done_pos already
     * run by a draft with token done_tok */
    int len, has_pend, done_pos, done_tok;
    float *pend;
    float *last; int last_base, last_n, last_cap;   /* the newest forward's rows (a rewind lands in them) */
    unsigned long long rows, drafts, kv_rows;
    double ms, kv_ms;
} Q36Mtp;
static Q36Mtp g_q36_mtp;

/* Keep the container's expert quantization, without a second quantization.
 * QW's packed format is gs64 planar; other grouped layouts widen exactly. */
static void q36_mtp_expert_qw(QW *w, const int8_t *q, const float *sc,
                              int I, int O, int gs, int packed) {
    w->I = I; w->O = O;
    int ng = gs ? (I + gs - 1) / gs : 1;
    if (packed && gs == 64 && I % 64 == 0 && dense_idot_on()) {
        w->q4 = malloc((size_t)I * O / 2);
        w->sg = falloc((int64_t)O * ng); w->ng = ng;
        if (!w->q4) { fprintf(stderr, "OOM MTP expert\n"); exit(1); }
        for (int o = 0; o < O; o++) for (int j = 0; j < I; j += 64)
            for (int k = 0; k < 32; k++)
                w->q4[(size_t)o * I / 2 + j / 2 + k] =
                    (q[(size_t)o * I + j + k] + 8) | ((q[(size_t)o * I + j + k + 32] + 8) << 4);
        memcpy(w->sg, sc, (size_t)O * ng * sizeof(float));
    } else if (!gs) {
        w->q = q36_walloc((size_t)I * O); w->sc = falloc(O);
        if (!w->q) { fprintf(stderr, "OOM MTP expert\n"); exit(1); }
        memcpy(w->q, q, (size_t)I * O); memcpy(w->sc, sc, (size_t)O * sizeof(float));
    } else {
        float *f = falloc((int64_t)I * O);
        for (int o = 0; o < O; o++) for (int j = 0; j < I; j++)
            f[(size_t)o * I + j] = q[(size_t)o * I + j] * sc[o * ng + j / gs];
        w->w = f;
    }
}
static int q36_mtp_load(Model *m) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp;
    int D = c->hidden, E = c->n_experts, I = c->inter, Ish = c->shared_inter;
    if (!st_find(&m->S, "mtp.fc.weight")) return 0;
    if (E <= 0 || E > 1024 || c->topk > 16 || Ish <= 0) {
        fprintf(stderr, "[mtp] head requires a MoE model with shared expert\n"); return 0;
    }
    p->E = E; p->K = c->topk; p->I = I; p->ish = Ish;
#define MN(field, name, n) p->field = load_norm_n(m, "mtp." name ".weight", n)
    MN(enorm, "pre_fc_norm_embedding", D); MN(hnorm, "pre_fc_norm_hidden", D);
    MN(head_norm, "norm", D); MN(L.in_ln, "layers.0.input_layernorm", D);
    MN(L.post_ln, "layers.0.post_attention_layernorm", D);
    MN(L.qn, "layers.0.self_attn.q_norm", c->head_dim);
    MN(L.kn, "layers.0.self_attn.k_norm", c->k_head_dim);
#undef MN
#define MW(field, name, ni, no) load_tq(m, "mtp." name ".weight", ni, no, 1, "mtp", &p->field)
    MW(eh, "fc", 2 * D, D);
    MW(L.q, "layers.0.self_attn.q_proj", D, c->q_heads * c->q_head_dim);
    MW(L.k, "layers.0.self_attn.k_proj", D, c->kv_heads * c->k_head_dim);
    MW(L.v, "layers.0.self_attn.v_proj", D, c->kv_heads * c->v_head_dim);
    MW(L.o, "layers.0.self_attn.o_proj", c->q_heads * c->head_dim, D);
    MW(sh_g, "layers.0.mlp.shared_expert.gate_proj", D, Ish);
    MW(sh_u, "layers.0.mlp.shared_expert.up_proj", D, Ish);
    MW(sh_d, "layers.0.mlp.shared_expert.down_proj", Ish, D);
#undef MW
    p->router = load_t_n(m, "mtp.layers.0.mlp.gate.weight", (int64_t)E * D);
    p->sh_gate = load_t_n(m, "mtp.layers.0.mlp.shared_expert_gate.weight", D);
    p->eg = calloc(E, sizeof(QW)); p->eu = calloc(E, sizeof(QW)); p->ed = calloc(E, sizeof(QW));
    if (!p->eg || !p->eu || !p->ed) { fprintf(stderr, "OOM MTP experts\n"); exit(1); }
    for (int e = 0; e < E; e++) {
        char name[128], sn[128];
        snprintf(name, sizeof name, "mtp.layers.0.mlp.experts.%d.merged_weight", e);
        snprintf(sn, sizeof sn, "mtp.layers.0.mlp.experts.%d.qs", e);
        st_tensor *t = st_find(&m->S, name);
        int64_t n = (int64_t)I * D;
        if (!t || (t->nbytes != 3*n && t->nbytes != 3*n/2 && t->nbytes != 2*n)) {
            fprintf(stderr, "invalid MTP expert %s\n", name); exit(1);
        }
        int packed = t->nbytes == 3*n/2, mixed = t->nbytes == 2*n;
        int8_t *q = malloc((size_t)3*n); uint8_t *raw = malloc((size_t)t->nbytes);
        if (!q || !raw) { fprintf(stderr, "OOM MTP expert\n"); exit(1); }
        st_read_raw(&m->S, name, raw, 1);
        if (packed) unpack_int4_to_int8(q, raw, 3*n);
        else if (mixed) { unpack_int4_to_int8(q, raw, 2*n); memcpy(q + 2*n, raw + n, (size_t)n); }
        else memcpy(q, raw, (size_t)3*n);
        int64_t ns = scale_count_gu(c), ds = scale_count_d(c);
        float *sc = load_t_n(m, sn, 2*ns + ds);
        q36_mtp_expert_qw(&p->eg[e], q, sc, D, I, g_expert_gs, packed || mixed);
        q36_mtp_expert_qw(&p->eu[e], q+n, sc+ns, D, I, g_expert_gs, packed || mixed);
        q36_mtp_expert_qw(&p->ed[e], q+2*n, sc+2*ns, I, D, mixed ? g_expert_down_gs : g_expert_gs, packed);
        free(q); free(raw); free(sc);
    }
    const char *file = getenv("Q36_MTP_VOCAB_IDS");
    if (file && *file) {
        FILE *f = fopen(file, "r");
        if (!f) { fprintf(stderr, "cannot read Q36_MTP_VOCAB_IDS %s\n", file); exit(1); }
        int *ids = malloc((size_t)c->vocab * sizeof(int));
        unsigned char *seen = calloc(c->vocab, 1); int id, n = 0;
        if (!ids || !seen) { fprintf(stderr, "OOM MTP vocabulary\n"); exit(1); }
        while (fscanf(f, "%d", &id) == 1) {
            if (id < 0 || id >= c->vocab) { fprintf(stderr, "invalid MTP vocabulary id %d\n", id); exit(1); }
            if (!seen[id]) { ids[n++] = id; seen[id] = 1; }
        }
        if (!feof(f) || !n) { fprintf(stderr, "invalid/empty MTP vocabulary\n"); exit(1); }
        fclose(f); free(seen);
        /* Read the shared checkpoint matrix, not a separately trained output head. */
        char resolved[QW_DENSE_NAME_MAX];
        const char *lm = dense_resolve(m, "lm_head.weight", resolved, sizeof resolved);
        st_tensor *weight = st_find(&m->S, lm);
        if (!weight || weight->numel != (int64_t)c->vocab * D || weight->dtype > 2) {
            fprintf(stderr, "MTP vocabulary needs a float shared lm_head\n"); exit(1);
        }
        float *rows = falloc((int64_t)n * D);
        for (int j = 0; j < n; j++)
            st_read_slice_f32(&m->S, lm, (int64_t)ids[j]*D, D, rows + (int64_t)j*D, 0);
        p->dvoc.I = D; p->dvoc.O = n; p->dvoc.w = rows;
        if (dense_i8_on()) { qw_quantize(rows, D, n, "lm_head", &p->dvoc); free(rows); p->dvoc.w = NULL; }
        p->nvoc = n; p->vid = ids;
    }
    p->pend = falloc(D); p->on = 1;
    fprintf(stderr, "[mtp] Qwen3.6 checkpoint head: %d experts\n", E);
    return 1;
}

/* the head's KV rows at the model's row stride (attention() indexes K[layer] by max_t),
 * and the pointer tables that put them after the model's layers */
static int q36_mtp_kv(Model *m) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp;
    if (p->stride != m->max_t || !p->Kc) {
        size_t n = (size_t)c->kv_heads * m->max_t * c->k_head_dim;
        float *kc = calloc(n, sizeof(float)), *vc = calloc(n, sizeof(float));
        if (!kc || !vc) { free(kc); free(vc); p->on = 0; return 0; }
        if (p->Kc && p->Vc) {
            int keep = p->stride < m->max_t ? p->stride : m->max_t;
            for (int h = 0; h < c->kv_heads; h++) {
                memcpy(kc + (size_t)h*m->max_t*c->k_head_dim, p->Kc + (size_t)h*p->stride*c->k_head_dim,
                       (size_t)keep*c->k_head_dim*sizeof(float));
                memcpy(vc + (size_t)h*m->max_t*c->k_head_dim, p->Vc + (size_t)h*p->stride*c->k_head_dim,
                       (size_t)keep*c->k_head_dim*sizeof(float));
            }
        }
        free(p->Kc); free(p->Vc); p->Kc = kc; p->Vc = vc; p->stride = m->max_t;
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

/* one row's routing from its router logits l[E] (softmax in place): top-K by the CPU's
 * rule, renormalised */
static void q36_mtp_route(float *l, int E, int K, int *idx, float *val) {
    softmax_row(l, E);
    for (int k = 0; k < K; k++) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0; for (int j = 0; j < k; j++) taken |= idx[j] == e;
            if (!taken && l[e] > bv) { bv = l[e]; best = e; }
        }
        idx[k] = best < 0 ? k : best; val[k] = best < 0 ? 0.f : bv;
    }
    float sm = 0; for (int k = 0; k < K; k++) sm += val[k];
    if (sm > 0) for (int k = 0; k < K; k++) val[k] /= sm;
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
    for (int r = 0; r < R; r++) q36_mtp_route(lg + (int64_t)r * E, E, K, idx + r * K, val + r * K);
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
 * later): fc, the input norm, k and v, the k norm and RoPE, into the head's cache,
 * as attention() stores them. */
static void q36_mtp_kv_rows(Model *m, const int *tok, const float *hid, int R, int pos0) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp; int D = c->hidden, KV = c->kv_heads, kvd = c->k_head_dim, kvo = KV * kvd;
    double t0 = now_s();
    float *cat = falloc((int64_t)R * 2 * D), *x = falloc((int64_t)R * D), *nrm = falloc((int64_t)R * D);
    float *k = falloc((int64_t)R * kvo), *v = falloc((int64_t)R * kvo);
    for (int r = 0; r < R; r++) {
        float *em = cat + (int64_t)r * 2 * D;
        q36_embed_row(m, tok[r], pos0 + r, em);
        rmsnorm_row(em, em, p->enorm, D, c->eps);
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

#ifdef COLI_VULKAN
/* ---------------- the head on the device (the chain's ops; Q36_MTP_GPU=0: the CPU's) ----------------
 * A fed row is one frame not waited for: the embedding and the model's row up, the two
 * input norms, fc, the input norm, k and v, the k norm and RoPE, the K/V row into
 * the head's device cache. A draft row is two frames: A up to the router's logits and
 * the shared expert (waited for: the host picks the top-k as the CPU head does), B the
 * routed experts, their weighted sum, the residual, the head's norm and lm_head (the
 * listed vocabulary, or the full shared matrix). The residual before
 * the head's norm stays on the device for a deeper draft. */
typedef struct {
    int on, failed, rows, cap;
    VkcBuf *prm, *emb, *hin, *cat, *x, *nrm, *q, *k, *v, *ctx, *tmp, *cs, *kc, *vc;
    VkcBuf *lg, *lgd, *gs, *us, *hs, *sd, *sg, *ge, *ue, *he, *ys, *wt, *rt, *fin, *hid, *outd;
    int dm; VkcBuf *dgu, *ddn, *degu, *dedn, *dix, *ditg, *ditd, *damap, *duse;   /* the head's experts through the device MoE */
    QW rq, sq;   /* the router and the shared expert's gate, f32 rows */
    size_t o_en, o_hn, o_in, o_post, o_qn, o_kn, o_head;
} Q36MtpGpu;
static Q36MtpGpu g_q36_mg;

static int q36_mg_res(VkcBuf **b, size_t floats, int kind) { return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }
static int q36_mg_init(Model *m) {
    Q36MtpGpu *g = &g_q36_mg; Q36Mtp *p = &g_q36_mtp; Cfg *c = &m->c;
    if (g->on) return 1;
    if (g->failed) return 0;
    const char *e = getenv("Q36_MTP_GPU");
    if ((e && *e == '0') || !g_vk_chain || !vkc_ready() || m->mpos || m->rope_delta) { g->failed = 1; return 0; }
    int D = c->hidden, E = p->E, K = p->K, I = p->I, Ish = p->ish, hd = c->head_dim, kvd = c->k_head_dim;
    /* the norms' weights, one arena */
    size_t n = 0;
    g->o_en = n; n += D; g->o_hn = n; n += D; g->o_in = n; n += D; g->o_post = n; n += D;
    g->o_qn = n; n += hd; g->o_kn = n; n += kvd; g->o_head = n; n += D;
    if (!q36_mg_res(&g->prm, n, VKC_UP)) { g->failed = 1; return 0; }
    float *w = vkc_ptr(g->prm);
    memcpy(w + g->o_en, p->enorm, D * 4); memcpy(w + g->o_hn, p->hnorm, D * 4);
    memcpy(w + g->o_in, p->L.in_ln, D * 4); memcpy(w + g->o_post, p->L.post_ln, D * 4);
    memcpy(w + g->o_qn, p->L.qn, hd * 4); memcpy(w + g->o_kn, p->L.kn, kvd * 4); memcpy(w + g->o_head, p->head_norm, D * 4);
    memset(&g->rq, 0, sizeof g->rq); g->rq.w = p->router; g->rq.I = D; g->rq.O = E;
    memset(&g->sq, 0, sizeof g->sq); g->sq.w = p->sh_gate; g->sq.I = D; g->sq.O = 1;
    /* every matrix on the device now, not at the first draft */
    int ok = vk_qw_tensor(&p->eh) && vk_qw_tensor(&p->L.q) && vk_qw_tensor(&p->L.k) && vk_qw_tensor(&p->L.v) &&
             vk_qw_tensor(&p->L.o) && vk_qw_tensor(&p->sh_g) && vk_qw_tensor(&p->sh_u) && vk_qw_tensor(&p->sh_d) &&
             vk_qw_tensor(&g->rq) && vk_qw_tensor(&g->sq) && (p->nvoc ? vk_qw_tensor(&p->dvoc) : vk_qw_tensor(&m->lm_head));
    for (int x = 0; ok && x < E; x++) ok = vk_qw_tensor(&p->eg[x]) && vk_qw_tensor(&p->eu[x]) && vk_qw_tensor(&p->ed[x]);
    ok = ok && q36_mg_res(&g->lg, E, VKC_DEV) && q36_mg_res(&g->lgd, E, VKC_DOWN) && q36_mg_res(&g->gs, Ish, VKC_DEV) &&
         q36_mg_res(&g->us, Ish, VKC_DEV) && q36_mg_res(&g->hs, Ish, VKC_DEV) && q36_mg_res(&g->sd, D, VKC_DEV) &&
         q36_mg_res(&g->sg, 1, VKC_DEV) && q36_mg_res(&g->ge, (size_t)K * I, VKC_DEV) && q36_mg_res(&g->ue, (size_t)K * I, VKC_DEV) &&
         q36_mg_res(&g->he, (size_t)K * I, VKC_DEV) && q36_mg_res(&g->ys, (size_t)K * D, VKC_DEV) && q36_mg_res(&g->wt, K, VKC_UP) &&
         q36_mg_res(&g->rt, D, VKC_DEV) && q36_mg_res(&g->fin, D, VKC_DEV) && q36_mg_res(&g->hid, D, VKC_DEV) &&
         q36_mg_res(&g->outd, p->nvoc ? p->nvoc : c->vocab, VKC_DOWN);
    if (!ok) { fprintf(stderr, "[mtp] the head stays on the CPU (device memory or a matrix refused)\n"); g->failed = 1; return 0; }
    g->on = 1;
    fprintf(stderr, "[mtp] the head runs on the device (the chain's ops)%s\n", p->vid ? ", drafts score a listed vocabulary" : p->nvoc ? ", drafts score the vocabulary's first ids" : "");
    return 1;
}
/* scratch for R rows, the head's device KV cache at the model's row stride */
static int q36_mg_bufs(Model *m, int R) {
    Q36MtpGpu *g = &g_q36_mg; Cfg *c = &m->c;
    int D = c->hidden, H = c->q_heads, KV = c->kv_heads, kvd = c->k_head_dim, qo = H * c->q_head_dim;
    size_t r = (size_t)R;
    if (g->cap != m->max_t) {
        size_t kn = (size_t)KV * m->max_t * kvd;
        VkcBuf *kc = NULL, *vc = NULL;
        int ok = q36_mg_res(&kc, kn, VKC_DEV) && q36_mg_res(&vc, kn, VKC_DEV);
        if (ok && g->cap) {
            int keep = g->cap < m->max_t ? g->cap : m->max_t;
            ok = vkc_begin();
            for (int h = 0; ok && h < KV; h++) {
                size_t dst = (size_t)h*m->max_t*kvd, src = (size_t)h*g->cap*kvd;
                ok = vkc_copy(kc, dst, g->kc, src, (size_t)keep*kvd) &&
                     vkc_copy(vc, dst, g->vc, src, (size_t)keep*kvd);
            }
            ok = ok && vkc_submit(1);
        }
        if (!ok) { vkc_free(kc); vkc_free(vc); return 0; }
        vkc_free(g->kc); vkc_free(g->vc); g->kc = kc; g->vc = vc;
        g->cap = m->max_t;
    }
    return q36_mg_res(&g->emb, r * D, VKC_UP) && q36_mg_res(&g->hin, r * D, VKC_UP) && q36_mg_res(&g->cat, r * 2 * D, VKC_DEV) &&
           q36_mg_res(&g->x, r * D, VKC_DEV) && q36_mg_res(&g->nrm, r * D, VKC_DEV) && q36_mg_res(&g->q, r * qo, VKC_DEV) &&
           q36_mg_res(&g->k, r * KV * kvd, VKC_DEV) && q36_mg_res(&g->v, r * KV * kvd, VKC_DEV) &&
           q36_mg_res(&g->ctx, r * H * c->head_dim, VKC_DEV) && q36_mg_res(&g->tmp, r * D, VKC_DEV) &&
           q36_mg_res(&g->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2), VKC_UP);
}
static int q36_mg_norm(VkcBuf *x, size_t xo, int xrow, size_t wo, VkcBuf *y, size_t yo, int yrow, int rows, int D, float eps) {
    VkcNorm np = {rows, D, 1, (int)xo, xrow, D, (int)yo, yrow, D, (int)wo, 0, VKC_NORM_ADD1, eps, 1.f};
    return vkc_norm(x, g_q36_mg.prm, y, &np);
}
/* rows R at pos0: tokens tok[r], the model's rows hid[r] (host) or, hid NULL, the device
 * state of the row before (R = 1); the head's input up to its K/V rows, stored */
static int q36_mg_input(Model *m, const int *tok, const float *hid, int R, int pos0) {
    Q36MtpGpu *g = &g_q36_mg; Cfg *c = &m->c;
    int D = c->hidden, KV = c->kv_heads, kvd = c->k_head_dim, kvo = KV * kvd, half = c->rotary_dim / 2;
    float *em = vkc_ptr(g->emb), *cs = vkc_ptr(g->cs);
    for (int r = 0; r < R; r++) q36_embed_row(m, tok[r], pos0 + r, em + (size_t)r * D);
    if (hid) memcpy(vkc_ptr(g->hin), hid, (size_t)R * D * 4);
    for (int s = 0; s < R; s++)
        for (int j = 0; j < half; j++) {
            float ang = (pos0 + s) * powf(c->theta, -2.0f * j / c->rotary_dim);
            cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
        }
    VkcRegion *rg = malloc(sizeof *rg * (size_t)R * KV);
    if (!rg) return 0;
    for (int s = 0; s < R; s++) for (int h = 0; h < KV; h++)
        rg[s * KV + h] = (VkcRegion){((size_t)h * g->cap + pos0 + s) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
    VkcNorm kn = {R * KV, kvd, KV, 0, kvo, kvd, 0, kvo, kvd, (int)g->o_kn, 0, VKC_NORM_ADD1, c->eps, 1.f};
    VkcRope kr = {R * KV, KV, 0, kvo, kvd, half, 0, 2 * half};
    Q36Mtp *p = &g_q36_mtp;
    int ok = q36_mg_norm(g->emb, 0, D, g->o_en, g->cat, 0, 2 * D, R, D, c->eps) &&
             q36_mg_norm(hid ? g->hin : g->hid, 0, D, g->o_hn, g->cat, D, 2 * D, R, D, c->eps) &&
             vkc_matmul(vk_qw_tensor(&p->eh), g->cat, 0, g->x, 0, R) &&
             q36_mg_norm(g->x, 0, D, g->o_in, g->nrm, 0, D, R, D, c->eps) &&
             vkc_matmul(vk_qw_tensor(&p->L.k), g->nrm, 0, g->k, 0, R) &&
             vkc_matmul(vk_qw_tensor(&p->L.v), g->nrm, 0, g->v, 0, R) &&
             vkc_norm(g->k, g->prm, g->k, &kn) && (!half || vkc_rope(g->k, g->cs, &kr)) &&
             vkc_copy_regions(g->kc, g->k, rg, R * KV) && vkc_copy_regions(g->vc, g->v, rg, R * KV);
    free(rg);
    return ok;
}
/* fed rows: their K/V rows only, one frame not waited for */
static int q36_mg_kv_rows(Model *m, const int *tok, const float *hid, int R, int pos0) {
    double t0 = now_s();
    int ok = q36_mg_bufs(m, R) && vkc_begin() && q36_mg_input(m, tok, hid, R, pos0) && vkc_submit(0);
    g_q36_mtp.kv_rows += (unsigned long long)R; g_q36_mtp.kv_ms += (now_s() - t0) * 1e3;
    return ok;
}
/* a draft row at pos: its logits into `logits` (the vocabulary or the drafts' first ids,
 * the rest -inf); its output after mtp.norm stays on the device for a deeper draft */
/* The head's routed experts as the chain's device MoE does the model's (Q36_MTP_DMOE=0:
 * the host's routing and a GEMV an expert): its experts' grouped-GEMV tables, once. */
static int q36_mg_dmoe(Model *m) {
    Q36MtpGpu *g = &g_q36_mg; Q36Mtp *p = &g_q36_mtp; Cfg *c = &m->c;
    if (g->dm) return g->dm > 0;
    g->dm = -1;
    const char *e = getenv("Q36_MTP_DMOE");
    int E = p->E, K = p->K, I = p->I, D = c->hidden, ng = (I + 63) / 64, nd = (D + 63) / 64;
    if ((e && *e == '0') || !vkc_moe_ready() || E % 64 || E > 1024 || K > 16) return 0;
    if (!vkc_reserve(&g->dgu, (size_t)E * 64, VKC_UP) || !vkc_reserve(&g->ddn, (size_t)E * 64, VKC_UP) ||
        !vkc_reserve(&g->degu, (size_t)K * 64, VKC_DEV) || !vkc_reserve(&g->dedn, (size_t)K * 64, VKC_DEV) ||
        !vkc_reserve(&g->dix, (size_t)K * 4, VKC_DOWN) || !vkc_reserve(&g->ditg, (size_t)K * ng * 16, VKC_UP) ||
        !vkc_reserve(&g->ditd, (size_t)K * nd * 16, VKC_UP) || !vkc_reserve(&g->damap, (size_t)K * 4, VKC_UP) ||
        !vkc_reserve(&g->duse, (size_t)K * 4, VKC_UP)) return 0;
    uint32_t *gu = vkc_ptr(g->dgu), *dn = vkc_ptr(g->ddn);
    for (int x = 0; x < E; x++)
        if (!coli_vk_tensor_entries(vk_qw_tensor(&p->eg[x]), vk_qw_tensor(&p->eu[x]), vk_qw_tensor(&p->ed[x]),
                                    gu + (size_t)x * 16, dn + (size_t)x * 16)) {
            fprintf(stderr, "[mtp] head expert %d off the grouped GEMV: the host routes the head\n", x);
            return 0;
        }
    uint32_t *ig = vkc_ptr(g->ditg), *id = vkc_ptr(g->ditd), *am = vkc_ptr(g->damap), *us = vkc_ptr(g->duse);
    for (int j = 0; j < K; j++) {
        am[j] = (uint32_t)j; us[j] = 1;
        for (int b = 0; b < ng; b++) { uint32_t *q = ig + ((size_t)j * ng + b) * 4; q[0] = j; q[1] = 0; q[2] = b * 64; q[3] = 1; }
        for (int b = 0; b < nd; b++) { uint32_t *q = id + ((size_t)j * nd + b) * 4; q[0] = j; q[1] = 0; q[2] = b * 64; q[3] = 1; }
    }
    g->dm = 1;
    fprintf(stderr, "[mtp] the head routes and runs its experts on the device (one frame a draft)\n");
    return 1;
}
static int q36_mg_row(Model *m, int tok, const float *hid, int pos, float *logits) {
    Q36MtpGpu *g = &g_q36_mg; Q36Mtp *p = &g_q36_mtp; Cfg *c = &m->c;
    int D = c->hidden, H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim;
    int qo = H * qdim, half = c->rotary_dim / 2, gate_dim = qdim > hd ? qdim - hd : 0;
    int E = p->E, K = p->K, I = p->I, Ish = p->ish, V = p->nvoc ? p->nvoc : c->vocab;
    double t0 = now_s();
    if (!q36_mg_bufs(m, 1) || !vkc_begin()) return 0;
    VkcNorm qn = {H, hd, H, 0, qo, qdim, 0, qo, qdim, (int)g->o_qn, 0, VKC_NORM_ADD1, c->eps, 1.f};
    VkcRope qr = {H, H, 0, qo, qdim, half, 0, 2 * half};
    VkcAttn a = {1, H, KV, hd, pos, g->cap, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0, 1.f / sqrtf((float)hd), 0, 0};
    VkcEw add = {VKC_EW_ADD, D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    VkcEw sw = {VKC_EW_SWIGLU, Ish, Ish, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    int ok = q36_mg_input(m, &tok, hid, 1, pos) &&
             vkc_matmul(vk_qw_tensor(&p->L.q), g->nrm, 0, g->q, 0, 1) &&
             vkc_norm(g->q, g->prm, g->q, &qn) && (!half || vkc_rope(g->q, g->cs, &qr)) &&
             vkc_attn(g->q, g->kc, g->vc, g->ctx, g->q, NULL, &a) &&
             vkc_matmul(vk_qw_tensor(&p->L.o), g->ctx, 0, g->tmp, 0, 1) &&
             vkc_ew(g->x, g->x, g->tmp, NULL, NULL, &add) &&
             q36_mg_norm(g->x, 0, D, g->o_post, g->nrm, 0, D, 1, D, c->eps) &&
             vkc_matmul(vk_qw_tensor(&g->rq), g->nrm, 0, g->lg, 0, 1) && vkc_copy(g->lgd, 0, g->lg, 0, E) &&
             vkc_matmul(vk_qw_tensor(&p->sh_g), g->nrm, 0, g->gs, 0, 1) &&
             vkc_matmul(vk_qw_tensor(&p->sh_u), g->nrm, 0, g->us, 0, 1) &&
             vkc_ew(g->hs, g->gs, g->us, NULL, NULL, &sw) &&
             vkc_matmul(vk_qw_tensor(&p->sh_d), g->hs, 0, g->sd, 0, 1) &&
             vkc_matmul(vk_qw_tensor(&g->sq), g->nrm, 0, g->sg, 0, 1);
    if (ok && q36_mg_dmoe(m)) {   /* one frame: routing, grouped experts, the head's end */
        int ng = (I + 63) / 64, nd = (D + 63) / 64;
        VkcEw cb1 = {VKC_EW_COMBINE, D, D, 1, 1 | 2 | 4, 1, 0, 0, 0, 0, 0, 1.f};
        ok = vkc_moe_route(g->lg, g->dgu, g->ddn, g->degu, g->dedn, g->wt, g->dix, 1, E, K, 0) &&
             vkc_moe_gemv(1, g->nrm, g->ditg, g->degu, g->he, g->damap, K * ng, D, I, 0.f, K, 64, 0) &&
             vkc_moe_gemv(0, g->he, g->ditd, g->dedn, g->ys, g->damap, K * nd, I, D, 0.f, 0, 64, 0) &&
             vkc_moe_sum(g->ys, g->wt, g->duse, g->rt, 1, K, D) &&
             vkc_ew(g->x, g->x, g->rt, g->sd, g->sg, &cb1) &&
             q36_mg_norm(g->x, 0, D, g->o_head, g->fin, 0, D, 1, D, c->eps) && vkc_copy(g->hid, 0, g->fin, 0, D) &&
             vkc_matmul(vk_qw_tensor(p->nvoc ? &p->dvoc : &m->lm_head), g->fin, 0, g->outd, 0, 1) && vkc_submit(1);
        if (!ok) return 0;
        goto picked;
    }
    ok = ok && vkc_submit(1);
    if (!ok) return 0;
    int idx[16]; float val[16], lg[1024];
    memcpy(lg, vkc_ptr(g->lgd), (size_t)E * 4);
    q36_mtp_route(lg, E, K, idx, val);
    memcpy(vkc_ptr(g->wt), val, (size_t)K * 4);
    VkcEw swk = {VKC_EW_SWIGLU, K * I, K * I, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    if (!vkc_reserve(&g->duse, (size_t)K * sizeof(uint32_t), VKC_UP)) return 0;
    uint32_t *used = vkc_ptr(g->duse);
    for (int j = 0; j < K; j++) used[j] = 1;
    VkcEw cb = {VKC_EW_COMBINE, D, D, 1, 1 | 2 | 4, 1, 0, 0, 0, 0, 0, 1.f};
    ok = vkc_begin();
    for (int k = 0; ok && k < K; k++)
        ok = vkc_matmul(vk_qw_tensor(&p->eg[idx[k]]), g->nrm, 0, g->ge, (size_t)k * I, 1) &&
             vkc_matmul(vk_qw_tensor(&p->eu[idx[k]]), g->nrm, 0, g->ue, (size_t)k * I, 1);
    ok = ok && vkc_ew(g->he, g->ge, g->ue, NULL, NULL, &swk);
    for (int k = 0; ok && k < K; k++) ok = vkc_matmul(vk_qw_tensor(&p->ed[idx[k]]), g->he, (size_t)k * I, g->ys, (size_t)k * D, 1);
    ok = ok && vkc_moe_sum(g->ys, g->wt, g->duse, g->rt, 1, K, D) &&
         vkc_ew(g->x, g->x, g->rt, g->sd, g->sg, &cb) &&
         q36_mg_norm(g->x, 0, D, g->o_head, g->fin, 0, D, 1, D, c->eps) && vkc_copy(g->hid, 0, g->fin, 0, D) &&
         vkc_matmul(vk_qw_tensor(p->nvoc ? &p->dvoc : &m->lm_head), g->fin, 0, g->outd, 0, 1) && vkc_submit(1);
    if (!ok) return 0;
picked:
    if (p->vid) {   /* the listed ids' scores, every other id out */
        /* the pick and its probability on the compact scores (the list's padding repeats
         * its last id: counted once); the full logits only for the CPU's own path */
        const float *o = vkc_ptr(g->outd);
        if (getenv("Q36_MTP_DUMP")) {
            for (int i = 0; i < c->vocab; i++) logits[i] = -INFINITY;
            for (int j = 0; j < V; j++) logits[p->vid[j]] = o[j];
        }
        int b = 0;
        for (int j = 1; j < V; j++) if (o[j] > o[b] || (o[j] == o[b] && p->vid[j] < p->vid[b])) b = j;
        /* scores more than 20 below the best add under 2e-9 each: skipped, the rest in float */
        float z = 0.f, ob = o[b];
        for (int j = 0; j < V; j++) {
            float d = o[j] - ob;
            if (d > -20.f && !(j > 0 && p->vid[j] == p->vid[j - 1])) z += expf(d);
        }
        p->pick_id = p->vid[b]; p->last_p = z > 0 ? (float)(1.0 / z) : 0.f; p->pick_ok = 1;
    } else {
        memcpy(logits, vkc_ptr(g->outd), (size_t)V * 4);
        for (int i = V; i < c->vocab; i++) logits[i] = -INFINITY;
    }
    p->rows++; p->ms += (now_s() - t0) * 1e3;
    return 1;
}
#endif

/* R head rows at positions pos0 .. pos0+R-1: token tok[r] with the model's row hid[r]
 * (its output at pos0+r-1, after the final norm); the last row's logits when logits,
 * and with hid_out that row's output after mtp.norm (a deeper draft's input).
 * Without logits only the rows' K/V matter (q36_mtp_kv_rows). */
static void q36_mtp_rows(Model *m, const int *tok, const float *hid, int R, int pos0, float *logits, float *hid_out) {
    Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp; int D = c->hidden;
    if (!q36_mtp_kv(m)) return;
    pos0--; /* pair h_p/token_{p+1} occupies head cache row p; no dummy row */
#ifdef COLI_VULKAN
    if (q36_mg_init(m)) {   /* the head on the device; hid NULL: its own state of the row before */
        if (!logits ? q36_mg_kv_rows(m, tok, hid, R, pos0) : R == 1 && q36_mg_row(m, tok[0], hid, pos0, logits)) return;
        fprintf(stderr, "[mtp] the head's device frames failed: it stays off\n");
        p->on = 0; return;
    }
#endif
    if (!logits) { q36_mtp_kv_rows(m, tok, hid, R, pos0); return; }
    double t0 = now_s();
    float *cat = falloc((int64_t)R * 2 * D), *x = falloc((int64_t)R * D), *nrm = falloc((int64_t)R * D), *tmp = falloc((int64_t)R * D);
    for (int r = 0; r < R; r++) {
        float *em = cat + (int64_t)r * 2 * D;
        q36_embed_row(m, tok[r], pos0 + r, em);
        rmsnorm_row(em, em, p->enorm, D, c->eps);
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
        if (p->nvoc) {
            float *scores = falloc(p->nvoc);
            matmul_d(scores, last, &p->dvoc, 1, D, p->nvoc);
            for (int i = 0; i < c->vocab; i++) logits[i] = -INFINITY;
            for (int j = 0; j < p->nvoc; j++) logits[p->vid[j]] = scores[j];
            free(scores);
        } else q36_mtp_lm_head(m, last, logits);
        if (hid_out) memcpy(hid_out, last, sizeof(float) * D);   /* a deeper draft's: after mtp.norm */
        free(last);
    }
    free(cat); free(x); free(nrm); free(tmp);
    p->rows += (unsigned long long)R; p->ms += (now_s() - t0) * 1e3;
}

/* After a forward fed ids[0..S) at pos_base with its residual rows res[S] (normalized here):
 * the head reads the rows it completes and keeps the last one pending. A forward that
 * starts behind the head (a verify's rejected rows undone) continues from the forward
 * before's row there; one that leaves a gap stops the head until the next prompt. */
static void q36_mtp_feed_rows(Model *m, const int *ids, int S, int pos_base, const float *hid);
static void q36_mtp_feed(Model *m, const int *ids, int S, int pos_base, const float *res) {
    Q36Mtp *p = &g_q36_mtp; int D = m->c.hidden;
    if (!p->on || S < 1 || !q36_mtp_kv(m)) return;
    float *hid = falloc((int64_t)S * D);   /* h_p: the model's output, after its final norm */
    #pragma omp parallel for schedule(static) if(S > 8)
    for (int s = 0; s < S; s++) rmsnorm_row(hid + (int64_t)s * D, res + (int64_t)s * D, m->final_norm, D, m->c.eps);
    q36_mtp_feed_rows(m, ids, S, pos_base, hid);
    free(hid);
}
static void q36_mtp_feed_rows(Model *m, const int *ids, int S, int pos_base, const float *hid) {
    Q36Mtp *p = &g_q36_mtp; int D = m->c.hidden;
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
static int q36_mtp_ready(int pos) {
    const Q36Mtp *p = &g_q36_mtp;
    if (!p->on) return 0;
    if (p->has_pend && p->len == pos - 1) return 1;
    /* after a verify's rejected rows: the model's row at pos - 1 is one the newest
     * forward computed (the head reads it from there, as q36_mtp_feed does) */
    return p->last && pos - 1 >= p->last_base && pos - 1 < p->last_base + p->last_n;
}
/* The pending row made the model's row at pos - 1 (q36_mtp_ready said it is there). */
static void q36_mtp_pend_at(Model *m, int pos) {
    Q36Mtp *p = &g_q36_mtp; int D = m->c.hidden;
    if (p->has_pend && p->len == pos - 1) return;
    memcpy(p->pend, p->last + (int64_t)(pos - 1 - p->last_base) * D, sizeof(float) * D);
    p->len = pos - 1; p->has_pend = 1;
}
/* The head's draft for the token after `tok` (about to be fed at pos): its row at pos;
 * `hidden` (D floats, may be NULL) gets the row's state for a deeper draft */
static int q36_mtp_draft(Model *m, int tok, int pos, float *hidden) {
    Q36Mtp *p = &g_q36_mtp; int V = m->c.vocab;
    float *lg = falloc(V);
    q36_mtp_pend_at(m, pos);
    q36_mtp_rows(m, &tok, p->pend, 1, pos, lg, hidden);
    const char *dump = getenv("Q36_MTP_DUMP");
    if (dump && *dump) {
        FILE *f = fopen(dump, "ab"); int row = pos - 1;
        if (!f || fwrite(&row, 4, 1, f) != 1 || fwrite(&tok, 4, 1, f) != 1 ||
            fwrite(lg, 4, V, f) != (size_t)V) { fprintf(stderr, "Q36_MTP_DUMP write failed\n"); exit(1); }
        fclose(f);
    }
    p->done_pos = pos; p->done_tok = tok; p->drafts++;
    int best = 0;
    if (p->pick_ok) { best = p->pick_id; p->pick_ok = 0; }   /* last_p set there */
    else {
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        /* the draft's softmax probability: how deep a verify should go (Q36_MTP_PMIN) */
        float z = 0, mx = lg[best];
        for (int i = 0; i < V; i++) if (lg[i] - mx > -20.f) z += expf(lg[i] - mx);
        p->last_p = z > 0 ? (float)(1.0 / z) : 0.f;
    }
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
#ifdef COLI_VULKAN
    if (g_q36_mg.on) q36_mtp_rows(m, &tok, NULL, 1, pos, lg, NULL);   /* the state before stays on the device */
    else
#endif
    q36_mtp_rows(m, &tok, h, 1, pos, lg, hidden);
    p->drafts++;
    int best = 0;
    if (p->pick_ok) { best = p->pick_id; p->pick_ok = 0; }   /* last_p set there */
    else {
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        /* the draft's softmax probability: how deep a verify should go (Q36_MTP_PMIN) */
        float z = 0, mx = lg[best];
        for (int i = 0; i < V; i++) if (lg[i] - mx > -20.f) z += expf(lg[i] - mx);
        p->last_p = z > 0 ? (float)(1.0 / z) : 0.f;
    }
    free(lg); free(h);
    return best;
}

/* Q36_MTP=1: the head, after the model and the device are up */
static void q36_mtp_attach(Model *m) {
    const char *path = getenv("Q36_MTP");
    if (!path || strcmp(path, "1") || g_q36_mtp.on) return;
    if (!q36_mtp_load(m)) { fprintf(stderr, "[mtp] Q36_MTP=%s: the head stays off\n", path); return; }
}
static void q36_mtp_report(void) {
    const Q36Mtp *p = &g_q36_mtp;
    if (!p->on) return;
    fprintf(stderr, "[mtp] head rows %llu (%.2f ms each), fed rows %llu (K/V only, %.2f ms each), drafts %llu\n", p->rows,
            p->rows ? p->ms / (double)p->rows : 0.0, p->kv_rows, p->kv_rows ? p->kv_ms / (double)p->kv_rows : 0.0, p->drafts);
}
