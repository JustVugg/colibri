/* inkling2_mtp.h — Inkling-2's drafter (INK2_MTP=1): mtp/mtp.safetensors and config.json's
 * mtp_config, read as the DFlare paper (block diffusion drafting over the target's fused layer
 * states) and the DSpark paper (a sequential "rnn" head over the block) describe them, with the
 * checkpoint's tensor shapes. Not transformers': it has no drafter for Inkling-2.
 *
 *  - The target's streams target_layer_ids name, counted as transformers' hidden states are
 *    (0 the embedding, k the stream after layer k - 1, so n_layers the last layer's before the
 *    output's mix), are fused per drafter layer:
 *    f = RMSNorm(sum_j softmax(fuse_weight[i])_j h_j) * dflare_norm. Each drafter layer
 *    projects f to its context keys (k_proj_ctx, a per-head RMSNorm k_norm_ctx, rope at the
 *    position) and values (v_proj_ctx), kept per position as the target goes.
 *  - A block of block_size rows at the decode position t: the token the target just chose
 *    (embedded and normed as the target embeds it) and mask_embed in the others. Each layer
 *    is a Qwen3-like one (q/k norms per head, rope, GQA, SwiGLU) whose queries see the context
 *    before t and every row of the block. The final norm's rows, row k for the token at t + k.
 *  - The head runs over the rows in order: z = [s; markov_w1[x_prev]; h_k; embed(x_prev)],
 *    rnn_proj(z) split into gate, candidate and output, s = sigmoid(g) s + (1 - sigmoid(g))
 *    tanh(c), and row k's logits are lm_head(h_k + hidden_out(tanh(o))); x_prev is the token
 *    just drafted (the block's first token for the first). Which streams the ids name and the
 *    fourth input's reading were chosen by measuring the drafts against the target's own
 *    tokens (docs/inkling.md).
 *
 * The verify is one forward over the token and the drafts; the target's greedy token after
 * each row decides how many it keeps. A draft it rejects leaves the convolution banks and the
 * sliding layers' rings past what it keeps: they come back from a snapshot, and the kept rows
 * run again. Greedy decoding only, and the CPU's layers (the dense chains keep state of their
 * own: INK2_MTP turns them off). The CPU's kernels compute a row the same way whatever rows run
 * beside it, so there the text is the same bits as without the drafter; a device's kernels for
 * several rows may round a near tie the other way. */
#ifndef INKLING2_MTP_H
#define INKLING2_MTP_H

#define INK2M_MAXL 8
#define INK2M_MAXT 16
#define INK2M_MAXB 16

typedef struct {
    Wt q, k, v, kc, vc, o, g, u, d;
    float *in_ln, *post_ln, *dn, *qn, *kn, *kcn;
    int local, rows;          /* rows: the context cache's (a ring of the window and a block's slack for local layers) */
    float *Kc, *Vc;           /* [rows][KV][hd], the keys after their norm and rope */
} Ink2MtpLayer;

typedef struct Ink2Mtp {
    int L, D, H, KV, hd, I, B, win, R, K;
    int nt, tl[INK2M_MAXT];   /* the target streams fused (k = target layer id) */
    float theta, eps;
    float fuse[INK2M_MAXL][INK2M_MAXT];
    Ink2MtpLayer l[INK2M_MAXL];
    float *mask, *norm, *rnn_b, *hout_b;
    Wt w1, rnn, hout;
    float *fused;             /* a forward's fused streams [L][S][D] */
    int64_t fused_cap;        /* rows it holds */
    int fS;                   /* the rows of the forward being captured, 0 = none */
    int max_t;
    unsigned long long steps, drafted, kept;
    double t_draft, t_verify, t_redo;
} Ink2Mtp;

static int ink2m_wanted(void) { const char *e = getenv("INK2_MTP"); return e && *e == '1'; }

static float *ink2m_vec(Model *m, const char *name, int n) {
    if (st_numel(&m->S, name) != n) { fprintf(stderr, "inkling2 mtp: %s is not %d values\n", name, n); exit(1); }
    return load_t(m, name);
}

/* config.json's mtp_config and mtp/mtp.safetensors; NULL (said why) = no drafter */
static Ink2Mtp *ink2m_load(Model *m, const char *snap) {
    const Cfg *c = &m->c;
    char path[2100];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (!buf || fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    buf[n] = 0; fclose(f);
    char *arena = NULL; jval *root = json_parse(buf, &arena);
    jval *mc = root ? json_get(root, "mtp_config") : NULL;
    Ink2Mtp *mt = NULL;
    const char *why = NULL;
    snprintf(path, sizeof(path), "%s/mtp/mtp.safetensors", snap);
    struct stat sb;
    if (!mc) why = "config.json has no mtp_config";
    else if (stat(path, &sb) != 0) why = "no mtp/mtp.safetensors beside it";
    else {
        jval *ht = json_get(mc, "markov_head_type"), *tl = json_get(mc, "target_layer_ids"), *ll = json_get(mc, "local_layer_ids");
        mt = calloc(1, sizeof *mt);
        mt->D = (int)jnum(mc, "hidden_size", 0); mt->L = (int)jnum(mc, "num_hidden_layers", 0);
        mt->H = (int)jnum(mc, "num_attention_heads", 0); mt->KV = (int)jnum(mc, "num_key_value_heads", 0);
        mt->hd = (int)jnum(mc, "head_dim", 0); mt->I = (int)jnum(mc, "intermediate_size", 0);
        mt->B = (int)jnum(mc, "block_size", 0); mt->win = (int)jnum(mc, "sliding_window_size", 0);
        mt->R = (int)jnum(mc, "markov_rank", 0);
        mt->theta = (float)jnum(mc, "rope_theta", 1e6); mt->eps = (float)jnum(mc, "rms_norm_eps", 1e-6);
        if (!ht || ht->t != J_STR || strcmp(ht->str, "rnn_hidden")) why = "a markov head other than rnn_hidden";
        else if (!tl || tl->t != J_ARR || tl->len < 1 || tl->len > INK2M_MAXT) why = "target_layer_ids";
        else if (mt->D != c->hidden || mt->L < 1 || mt->L > INK2M_MAXL || mt->B < 2 || mt->B > INK2M_MAXB || mt->hd < 2 ||
                 mt->hd % 2 || mt->KV < 1 || mt->H % mt->KV || mt->R < 1 || mt->I < 1)
            why = "a geometry it does not take";
        else {
            mt->nt = tl->len;
            for (int j = 0; j < mt->nt; j++) {
                mt->tl[j] = (int)tl->kids[j]->num;
                if (mt->tl[j] < 0 || mt->tl[j] > c->n_layers) why = "a target layer past the model's";
            }
            for (int i = 0; i < mt->L; i++) mt->l[i].local = 0;
            for (int j = 0; ll && ll->t == J_ARR && j < ll->len; j++) {
                int i = (int)ll->kids[j]->num;
                if (i >= 0 && i < mt->L) mt->l[i].local = 1;
            }
        }
    }
    free(buf); free(arena);
    if (why) { fprintf(stderr, "[inkling2] INK2_MTP: %s; decoding without the drafter\n", why); free(mt); return NULL; }
    /* the drafter's tensors through the model's loaders: its shards stand in for the model's a moment */
    shards keep = m->S, sm;
    snprintf(path, sizeof(path), "%s/mtp", snap);
    st_init(&sm, path);
    m->S = sm;
    char nm[160];
    const int D = mt->D, qo = mt->H * mt->hd, kvo = mt->KV * mt->hd;
    float *fw = ink2m_vec(m, "fuse_weight", (mt->L + 1) * mt->nt);
    for (int i = 0; i < mt->L; i++) {   /* the softmax of each layer's row; the head's row (the last) is not read */
        float mx = -1e30f, s = 0;
        for (int j = 0; j < mt->nt; j++) if (fw[i * mt->nt + j] > mx) mx = fw[i * mt->nt + j];
        for (int j = 0; j < mt->nt; j++) s += mt->fuse[i][j] = expf(fw[i * mt->nt + j] - mx);
        for (int j = 0; j < mt->nt; j++) mt->fuse[i][j] /= s;
    }
    free(fw);
    for (int i = 0; i < mt->L; i++) {
        Ink2MtpLayer *l = &mt->l[i];
#define INK2M_W(f, s) (snprintf(nm, sizeof nm, "layers.%d.%s.weight", i, s), l->f = load_w(m, nm, 1))
#define INK2M_V(f, s, n) (snprintf(nm, sizeof nm, "layers.%d.%s.weight", i, s), l->f = ink2m_vec(m, nm, n))
        INK2M_W(q, "self_attn.q_proj"); INK2M_W(k, "self_attn.k_proj"); INK2M_W(v, "self_attn.v_proj");
        INK2M_W(kc, "self_attn.k_proj_ctx"); INK2M_W(vc, "self_attn.v_proj_ctx"); INK2M_W(o, "self_attn.o_proj");
        INK2M_W(g, "mlp.gate_proj"); INK2M_W(u, "mlp.up_proj"); INK2M_W(d, "mlp.down_proj");
        INK2M_V(in_ln, "input_layernorm", D); INK2M_V(post_ln, "post_attention_layernorm", D);
        INK2M_V(dn, "dflare_norm", D); INK2M_V(qn, "self_attn.q_norm", mt->hd);
        INK2M_V(kn, "self_attn.k_norm", mt->hd); INK2M_V(kcn, "self_attn.k_norm_ctx", mt->hd);
#undef INK2M_W
#undef INK2M_V
        if (st_numel(&m->S, (snprintf(nm, sizeof nm, "layers.%d.self_attn.q_proj.weight", i), nm)) != (int64_t)qo * D ||
            st_numel(&m->S, (snprintf(nm, sizeof nm, "layers.%d.self_attn.k_proj_ctx.weight", i), nm)) != (int64_t)kvo * D ||
            st_numel(&m->S, (snprintf(nm, sizeof nm, "layers.%d.mlp.gate_proj.weight", i), nm)) != (int64_t)mt->I * D) {
            fprintf(stderr, "inkling2 mtp: layer %d's matrices do not match mtp_config\n", i); exit(1);
        }
    }
    mt->mask = ink2m_vec(m, "mask_embed", D);
    mt->norm = ink2m_vec(m, "norm.weight", D);
    mt->w1 = load_w(m, "markov_head.markov_w1.weight", 0);
    mt->rnn = load_w(m, "markov_head.rnn_proj.weight", 0);
    mt->rnn_b = ink2m_vec(m, "markov_head.rnn_proj.bias", 3 * mt->R);
    mt->hout = load_w(m, "markov_head.hidden_out.weight", 0);
    mt->hout_b = ink2m_vec(m, "markov_head.hidden_out.bias", D);
    if (st_numel(&m->S, "markov_head.rnn_proj.weight") != (int64_t)3 * mt->R * (2 * mt->R + 2 * D) ||
        st_numel(&m->S, "markov_head.markov_w1.weight") != (int64_t)c->vocab * mt->R) {
        fprintf(stderr, "inkling2 mtp: the markov head's matrices do not match mtp_config\n"); exit(1);
    }
    m->S = keep;
    const char *ke = getenv("INK2_MTP_K");
    mt->K = ke && atoi(ke) > 0 ? atoi(ke) : 3;
    if (mt->K > mt->B - 1) mt->K = mt->B - 1;
    fprintf(stderr, "[inkling2] INK2_MTP: the drafter, %d layers, blocks of %d, %d drafts a step (INK2_MTP_K)\n",
            mt->L, mt->B, mt->K);
    return mt;
}

/* the context caches at the served length (kv_alloc's max_t); the positions they hold kept */
static void ink2m_alloc(Ink2Mtp *mt, int max_t, int len) {
    if (max_t <= mt->max_t) return;
    const int kvo = mt->KV * mt->hd;
    for (int i = 0; i < mt->L; i++) {
        Ink2MtpLayer *l = &mt->l[i];
        const int rows = l->local && mt->win > 0 && mt->win + 2 * mt->B < max_t ? mt->win + 2 * mt->B : max_t;
        float *K = falloc((int64_t)rows * kvo), *V = falloc((int64_t)rows * kvo);
        for (int p = len - rows > 0 ? len - rows : 0; l->Kc && p < len; p++) {
            memcpy(K + (int64_t)(p % rows) * kvo, l->Kc + (int64_t)(p % l->rows) * kvo, (size_t)kvo * sizeof(float));
            memcpy(V + (int64_t)(p % rows) * kvo, l->Vc + (int64_t)(p % l->rows) * kvo, (size_t)kvo * sizeof(float));
        }
        free(l->Kc); free(l->Vc);
        l->Kc = K; l->Vc = V; l->rows = rows;
    }
    mt->max_t = max_t;
}

static void ink2m_rope(float *x, int hd, int pos, float theta) {
    const int half = hd / 2;
    for (int i = 0; i < half; i++) {
        const double inv = pow((double)theta, -2.0 * i / hd);
        const float ang = (float)((double)pos * inv), cs = cosf(ang), sn = sinf(ang);
        const float a = x[i], b = x[i + half];
        x[i] = a * cs - b * sn; x[i + half] = b * cs + a * sn;
    }
}
static void ink2m_head_norm(float *x, int nh, int hd, const float *w, float eps) {
    for (int h = 0; h < nh; h++) rmsnorm_row(x + (int64_t)h * hd, x + (int64_t)h * hd, w, hd, eps);
}

/* ink2_layers_forward: the start of a forward over S rows (fused cleared) */
static void ink2m_begin(Model *m, int S) {
    Ink2Mtp *mt = m->mtp;
    if (!mt || m->mux_rows) { if (mt) mt->fS = 0; return; }
    const int64_t need = (int64_t)mt->L * S * mt->D;
    if (need > mt->fused_cap) { free(mt->fused); mt->fused = falloc(need); mt->fused_cap = need; }
    memset(mt->fused, 0, (size_t)need * sizeof(float));
    mt->fS = S;
}
/* ...the stream k (0 the embedding, i + 1 after layer i, n_layers + 1 the output's mix): what the
 * drafter's layers fuse of it */
static void ink2m_capture(Model *m, int k, const float *x, int S) {
    Ink2Mtp *mt = m->mtp;
    if (!mt || mt->fS != S) return;
    for (int j = 0; j < mt->nt; j++) {
        if (mt->tl[j] != k) continue;
        for (int i = 0; i < mt->L; i++) {
            const float a = mt->fuse[i][j];
            float *f = mt->fused + (int64_t)i * S * mt->D;
            for (int64_t e = 0; e < (int64_t)S * mt->D; e++) f[e] += a * x[e];
        }
    }
}
/* ...and its end: the rows' context keys and values at positions pos0.. */
static void ink2m_end(Model *m, int pos0, int S) {
    Ink2Mtp *mt = m->mtp;
    if (!mt || mt->fS != S) return;
    mt->fS = 0;
    ink2m_alloc(mt, m->max_t, pos0);
    const int D = mt->D, kvo = mt->KV * mt->hd;
    float *fn = falloc((int64_t)S * D), *kr = falloc((int64_t)S * kvo), *vr = falloc((int64_t)S * kvo);
    for (int i = 0; i < mt->L; i++) {
        Ink2MtpLayer *l = &mt->l[i];
        const float *f = mt->fused + (int64_t)i * S * D;
        for (int s = 0; s < S; s++) rmsnorm_row(fn + (int64_t)s * D, f + (int64_t)s * D, l->dn, D, mt->eps);
        matmul_w(kr, fn, l->kc, S, D, kvo);
        matmul_w(vr, fn, l->vc, S, D, kvo);
        for (int s = 0; s < S; s++) {
            float *kk = kr + (int64_t)s * kvo;
            ink2m_head_norm(kk, mt->KV, mt->hd, l->kcn, mt->eps);
            for (int h = 0; h < mt->KV; h++) ink2m_rope(kk + (int64_t)h * mt->hd, mt->hd, pos0 + s, mt->theta);
            const int slot = (pos0 + s) % l->rows;
            memcpy(l->Kc + (int64_t)slot * kvo, kk, (size_t)kvo * sizeof(float));
            memcpy(l->Vc + (int64_t)slot * kvo, vr + (int64_t)s * kvo, (size_t)kvo * sizeof(float));
        }
    }
    free(fn); free(kr); free(vr);
}

/* the target's embedding of a token, as step_mm embeds it */
static void ink2m_embed(Model *m, int tok, float *x) {
    const int D = m->c.hidden;
    wt_row_f32(m->embed, (int64_t)tok * D, x, D);
    if (m->embed_norm) rmsnorm_row(x, x, m->embed_norm, D, m->c.eps);
}

/* Tests (INK2_MTP_ORACLE=1 in the oracle harness): the drafts are the oracle's own tokens, one of
 * them changed on most steps, so that the verify keeps all, some or none of a block */
static const int *g_ink2m_known;
static int g_ink2m_nknown;

/* K drafts after `tok` at position t, the context holding positions before t */
static void ink2m_draft(Model *m, int tok, int t, int K, int *out) {
    Ink2Mtp *mt = m->mtp;
    if (g_ink2m_known) {
        const int bad = (t / 2) % (K + 2);   /* K or K + 1: every draft right */
        for (int k = 1; k <= K; k++) {
            const int p = t + k;
            out[k - 1] = p < g_ink2m_nknown ? g_ink2m_known[p] : 0;
            if (k - 1 == bad) out[k - 1] = (out[k - 1] + 1) % m->c.unpad_vocab;
        }
        return;
    }
    const Cfg *c = &m->c;
    const int B = mt->B, D = mt->D, hd = mt->hd, H = mt->H, KV = mt->KV, qo = H * hd, kvo = KV * hd, grp = H / KV;
    double t0 = now_s();
    float *h = falloc((int64_t)B * D), *x = falloc((int64_t)B * D), *q = falloc((int64_t)B * qo);
    float *kb = falloc((int64_t)B * kvo), *vb = falloc((int64_t)B * kvo), *ao = falloc((int64_t)B * qo);
    float *g = falloc((int64_t)B * mt->I), *u = falloc((int64_t)B * mt->I);
    ink2m_embed(m, tok, h);
    for (int r = 1; r < B; r++) memcpy(h + (int64_t)r * D, mt->mask, (size_t)D * sizeof(float));
    const float scale = 1.f / sqrtf((float)hd);
    for (int i = 0; i < mt->L; i++) {
        Ink2MtpLayer *l = &mt->l[i];
        for (int r = 0; r < B; r++) rmsnorm_row(x + (int64_t)r * D, h + (int64_t)r * D, l->in_ln, D, mt->eps);
        matmul_w(q, x, l->q, B, D, qo);
        matmul_w(kb, x, l->k, B, D, kvo);
        matmul_w(vb, x, l->v, B, D, kvo);
        for (int r = 0; r < B; r++) {
            ink2m_head_norm(q + (int64_t)r * qo, H, hd, l->qn, mt->eps);
            ink2m_head_norm(kb + (int64_t)r * kvo, KV, hd, l->kn, mt->eps);
            for (int hh = 0; hh < H; hh++) ink2m_rope(q + (int64_t)r * qo + (int64_t)hh * hd, hd, t + r, mt->theta);
            for (int hh = 0; hh < KV; hh++) ink2m_rope(kb + (int64_t)r * kvo + (int64_t)hh * hd, hd, t + r, mt->theta);
        }
        const int lo = l->local && mt->win > 0 && t - mt->win > 0 ? t - mt->win : 0, n = t - lo + B;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int hh = 0; hh < H; hh++) {
            for (int r = 0; r < B; r++) {
                float *sc = malloc((size_t)n * sizeof(float));
                const float *qv = q + (int64_t)r * qo + (int64_t)hh * hd;
                const int kh = hh / grp;
                float mx = -1e30f;
                for (int j = 0; j < n; j++) {
                    const float *kv = j < t - lo ? l->Kc + (int64_t)((lo + j) % l->rows) * kvo + (int64_t)kh * hd
                                                 : kb + (int64_t)(j - (t - lo)) * kvo + (int64_t)kh * hd;
                    float a = 0; for (int e = 0; e < hd; e++) a += qv[e] * kv[e];
                    sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
                }
                float den = 0; for (int j = 0; j < n; j++) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
                float *ov = ao + (int64_t)r * qo + (int64_t)hh * hd;
                for (int e = 0; e < hd; e++) ov[e] = 0;
                for (int j = 0; j < n; j++) {
                    const float *vv = j < t - lo ? l->Vc + (int64_t)((lo + j) % l->rows) * kvo + (int64_t)kh * hd
                                                 : vb + (int64_t)(j - (t - lo)) * kvo + (int64_t)kh * hd;
                    const float a = sc[j] / den;
                    for (int e = 0; e < hd; e++) ov[e] += a * vv[e];
                }
                free(sc);
            }
        }
        matmul_w(x, ao, l->o, B, qo, D);
        for (int64_t e = 0; e < (int64_t)B * D; e++) h[e] += x[e];
        for (int r = 0; r < B; r++) rmsnorm_row(x + (int64_t)r * D, h + (int64_t)r * D, l->post_ln, D, mt->eps);
        matmul_w(g, x, l->g, B, D, mt->I);
        matmul_w(u, x, l->u, B, D, mt->I);
        for (int64_t e = 0; e < (int64_t)B * mt->I; e++) g[e] = g[e] / (1.f + expf(-g[e])) * u[e];
        matmul_w(x, g, l->d, B, mt->I, D);
        for (int64_t e = 0; e < (int64_t)B * D; e++) h[e] += x[e];
    }
    for (int r = 0; r < B; r++) rmsnorm_row(h + (int64_t)r * D, h + (int64_t)r * D, mt->norm, D, mt->eps);
    /* the head, row by row */
    const int R = mt->R, Z = 2 * R + 2 * D;
    float *z = falloc(Z), *gco = falloc(3 * R), *s = calloc((size_t)R, sizeof(float)), *ov = falloc(R);
    float *y = falloc(D), *add = falloc(D), *lo = falloc(c->unpad_vocab);
    int prev = tok;
    for (int k = 1; k <= K; k++) {
        memcpy(z, s, (size_t)R * sizeof(float));
        wt_row_f32(mt->w1, (int64_t)prev * R, z + R, R);
        memcpy(z + 2 * R, h + (int64_t)k * D, (size_t)D * sizeof(float));
        ink2m_embed(m, prev, z + 2 * R + D);
        matmul_w(gco, z, mt->rnn, 1, Z, 3 * R);
        for (int e = 0; e < R; e++) {
            const float gt = 1.f / (1.f + expf(-(gco[e] + mt->rnn_b[e])));
            s[e] = gt * s[e] + (1.f - gt) * tanhf(gco[R + e] + mt->rnn_b[R + e]);
            ov[e] = tanhf(gco[2 * R + e] + mt->rnn_b[2 * R + e]);
        }
        matmul_w(add, ov, mt->hout, 1, R, D);
        for (int e = 0; e < D; e++) y[e] = h[(int64_t)k * D + e] + add[e] + mt->hout_b[e];
        matmul_w(lo, y, m->lm_head, 1, D, c->unpad_vocab);
        int best = 0;
        for (int v = 1; v < c->unpad_vocab; v++) if (lo[v] > lo[best]) best = v;
        out[k - 1] = prev = best;
    }
    free(h); free(x); free(q); free(kb); free(vb); free(ao); free(g); free(u);
    free(z); free(gco); free(s); free(ov); free(y); free(add); free(lo);
    mt->t_draft += now_s() - t0;
}

/* One step at the decode position t, `tok` (the target's choice there) not fed yet: its drafts,
 * the verify over [tok, drafts], and the state as if the target had decoded what it keeps.
 * out: the kept drafts and the target's token after them; returns how many (1..K+1). The
 * forward over the kept rows runs again when a draft is rejected (the convolution banks). */
static float *step_mm(Model *m, const int *ids, int S, int pos0, int *tf_out, const uint8_t *dmel, int naud);
static InkPinState *ink_pin_state_save(Model *m, InkPinState *reuse);
static void ink_pin_state_restore(Model *m, const InkPinState *st);
static void ink_pin_state_free(void *v);
static int ink2m_step(Model *m, int tok, int t, int *out) {
    Ink2Mtp *mt = m->mtp;
    const Cfg *c = &m->c;
    int K = mt->K;
    if (t + K + 1 > m->max_t) K = m->max_t - t - 1;
    if (K < 1) {   /* no room for a draft: one plain step */
        int *tf = malloc((size_t)(t + 1) * sizeof(int));
        float *lg = step_mm(m, &tok, 1, t, tf, NULL, 0);
        free(lg); out[0] = tf[t]; free(tf);
        return 1;
    }
    int ids[INK2M_MAXB];
    ids[0] = tok;
    ink2m_draft(m, tok, t, K, ids + 1);
    /* the snapshot a rejection needs: the banks, and the sliding layers' ring rows the verify overwrites */
    static InkPinState *snap;
    snap = ink_pin_state_save(m, snap);
    size_t ringn = 0;
    for (int i = 0; i < c->n_layers; i++) {
        const int win = kv_ring_rows(c, i, m->max_t);
        if (win < m->max_t) ringn += (size_t)2 * L_KV(c, i) * (K + 1) * L_HD(c, i);
    }
    float *ring = ringn ? falloc((int64_t)ringn) : NULL, *rp = ring;
    for (int i = 0; i < c->n_layers; i++) {
        const int win = kv_ring_rows(c, i, m->max_t), hd = L_HD(c, i);
        if (win >= m->max_t) continue;
        for (int j = 0; j <= K; j++)
            for (int h = 0; h < L_KV(c, i); h++, rp += 2 * hd) {
                const int64_t off = ((int64_t)h * win + (t + j) % win) * hd;
                memcpy(rp, m->K[i] + off, (size_t)hd * sizeof(float));
                memcpy(rp + hd, m->V[i] + off, (size_t)hd * sizeof(float));
            }
    }
    double t0 = now_s();
    int *tf = malloc((size_t)(t + K + 1) * sizeof(int));
    float *lg = step_mm(m, ids, K + 1, t, tf, NULL, 0);
    free(lg);
    mt->t_verify += now_s() - t0;
    int a = 0;
    while (a < K && ids[1 + a] == tf[t + a]) a++;
    for (int j = 0; j < a; j++) out[j] = ids[1 + j];
    out[a] = tf[t + a];
    free(tf);
    mt->steps++; mt->drafted += (unsigned long long)K; mt->kept += (unsigned long long)a;
    if (a < K) {   /* the state back to before the verify, then the kept rows again */
        double t1 = now_s();
        ink_pin_state_restore(m, snap);
        rp = ring;
        for (int i = 0; i < c->n_layers; i++) {
            const int win = kv_ring_rows(c, i, m->max_t), hd = L_HD(c, i);
            if (win >= m->max_t) continue;
            /* every row: the forward again reads the older positions there before it writes the
             * kept ones (a slot of a kept row held a position the window still sees) */
            for (int j = 0; j <= K; j++)
                for (int h = 0; h < L_KV(c, i); h++, rp += 2 * hd) {
                    const int64_t off = ((int64_t)h * win + (t + j) % win) * hd;
                    memcpy(m->K[i] + off, rp, (size_t)hd * sizeof(float));
                    memcpy(m->V[i] + off, rp + hd, (size_t)hd * sizeof(float));
                }
        }
        if (m->kvp.len > t) m->kvp.len = t;
        lg = step_mm(m, ids, a + 1, t, NULL, NULL, 0);
        free(lg);
        mt->t_redo += now_s() - t1;
    }
    free(ring);
    return a + 1;
}

static void ink2m_report(const Model *m) {
    const Ink2Mtp *mt = m->mtp;
    if (!mt || !mt->steps) return;
    fprintf(stderr, "[inkling2] INK2_MTP: %llu steps, %llu of %llu drafts kept (%.2f a step, %.2f tokens a step), "
                    "drafting %.2f s, verifying %.2f s, the kept rows again %.2f s\n",
            mt->steps, mt->kept, mt->drafted, (double)mt->kept / mt->steps, 1.0 + (double)mt->kept / mt->steps,
            mt->t_draft, mt->t_verify, mt->t_redo);
}

#endif /* INKLING2_MTP_H */
