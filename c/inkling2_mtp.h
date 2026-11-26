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
    void *mc;                 /* the drafter on the Vulkan chain (Ink2Mc, g_ink2mc), NULL = none yet */
    uint8_t *lmq; float *lms; /* the drafts' int4 lm_head until the chain takes it (ink2m_lm_q4) */
    /* how many drafts a step (ink2m_pick_k; INK2_MTP_ADAPT=0: always K): acc[j] the share of steps
     * that kept draft j of those that kept the ones before it, tk[k] a step's seconds with k drafts
     * (0: a plain step), both moving averages; plain: plain steps since the last draft */
    double acc[INK2M_MAXB], tk[INK2M_MAXB];
    unsigned long long plain, noplain, explore, ksteps[INK2M_MAXB];
    int probe_iv, probing;
} Ink2Mtp;

static int ink2m_wanted(void) { const char *e = getenv("INK2_MTP"); return e && *e == '1'; }

static float *ink2m_vec(Model *m, const char *name, int n) {
    if (st_numel(&m->S, name) != n) { fprintf(stderr, "inkling2 mtp: %s is not %d values\n", name, n); exit(1); }
    return load_t(m, name);
}

/* The drafts' own lm_head: the target's rows in int4 groups of 64 (the CPU's matmul_i4g form, fmt 4),
 * a quarter of the bf16 rows' bytes. On an RTX 5090 the bf16 head's GEMV took 1.9-2.0 ms a row,
 * three of them a step at K = 3: the drafts' largest cost. A draft only proposes; the verify's
 * head is the target's own. INK2_MTP_HEAD=bf16: the target's head for the drafts too. Made at load
 * (about a second), so the first prompt does not wait for it. */
static void ink2m_lm_q4(Model *m, Ink2Mtp *mt) {
    const char *e = getenv("INK2_MTP_HEAD"), *nm = "model.llm.unembed.weight";
    if ((e && !strcmp(e, "bf16")) || !st_has(&m->S, nm)) return;
    const int I = m->c.hidden, O = m->c.unpad_vocab, gs = 64, ng = I / gs;
    if (I % gs) return;
    uint8_t *q = malloc((size_t)O * I / 2); float *sc = malloc((size_t)O * ng * sizeof(float));
    const int blk = 4096;
    float *w = malloc((size_t)blk * I * sizeof(float));
    if (q && sc && w) {
        for (int r0 = 0; r0 < O; r0 += blk) {
            const int nr = O - r0 < blk ? O - r0 : blk;
            read_f32_slice(&m->S, nm, w, (int64_t)r0 * I, (int64_t)nr * I);
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < nr; r++)
                for (int g = 0; g < ng; g++) {
                    const float *x = w + (size_t)r * I + (size_t)g * gs;
                    float a = 0.f;
                    for (int i = 0; i < gs; i++) a = fmaxf(a, fabsf(x[i]));
                    const float s = a / 7.f, inv = s > 0.f ? 1.f / s : 0.f;
                    sc[(size_t)(r0 + r) * ng + g] = s;
                    uint8_t *qb = q + (size_t)(r0 + r) * I / 2 + (size_t)g * gs / 2;
                    for (int i = 0; i < gs; i += 2) {
                        int lo = (int)lrintf(x[i] * inv), hi = (int)lrintf(x[i + 1] * inv);
                        lo = lo < -8 ? -8 : lo > 7 ? 7 : lo; hi = hi < -8 ? -8 : hi > 7 ? 7 : hi;
                        qb[i / 2] = (uint8_t)((lo + 8) | (hi + 8) << 4);
                    }
                }
        }
        mt->lmq = q; mt->lms = sc; q = NULL; sc = NULL;   /* up at the chain's first forward (ink2mc_prepare) */
    }
    free(q); free(sc); free(w);
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
                /* counted over the layers as they run: a loop's passes each (Inkling-2-Mini: 5, 23, 41, 48 of 48) */
                if (mt->tl[j] < 0 || mt->tl[j] > c->n_state) why = "a target layer past the model's";
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
    for (int j = 0; j < INK2M_MAXB; j++) mt->acc[j] = 0.6;   /* a start that drafts; the steps correct it */
#ifdef COLI_VULKAN
    if (ink2mc_wanted()) ink2m_lm_q4(m, mt);
#endif
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

#ifdef COLI_VULKAN
static float *step_mm(Model *m, const int *ids, int S, int pos0, int *tf_out, const uint8_t *dmel, int naud);
/* ---- INK2_MTP on the Vulkan chain (g_ink2mc; INK2_MTP_CHAIN=0: the CPU's drafter above) ----
 * The chain keeps the target's state on the device, so the drafter runs there too:
 *  - every chain forward copies the streams target_layer_ids name (ink2mc_capture) and, in the
 *    same frame, makes each drafter layer's context keys and values of them (ink2mc_ctx: the
 *    fuse, dflare_norm, k/v_proj_ctx, k_norm_ctx, rope) into caches on the device, position-major
 *    (a ring of the window and two blocks' slack on a local layer);
 *  - a draft is one frame of the drafter's layers over the block, its queries over the context
 *    and the whole block (one attention call: the block's rows as heads of one row at the block's
 *    last position), then the head on the host, row by row, with lm_head on the device;
 *  - the verify is one chain forward of 1 + K rows: each row's logits as a decode step's (the
 *    decode step's GEMVs, vkc_gemm_rows(0)), the convolutions' states after each row kept beside
 *    the live ones (ink2c_sconv's snap). A rejection copies back the states after the kept rows
 *    and the sliding layers' ring rows the rejected rows overwrote (saved by ink2mc_save before
 *    the verify): no forward again, so a step costs the draft and one forward.
 * The drafts only decide how many tokens a forward yields: the tokens are the target's own. */
typedef struct {
    ColiVkTensor *q, *k, *v, *kc, *vc, *o, *g, *u, *d;
    VkcBuf *K, *V; int rows, ring;    /* [rows][KV][hd]; ring: position p in row p % rows, else in row p */
    size_t o_in, o_post, o_dn, o_qn, o_kn, o_kcn;
} Ink2McLayer;
typedef struct {
    Ink2McLayer l[INK2M_MAXL];
    VkcBuf *prm; size_t o_norm;
    VkcBuf *cap; int cap_rows;                        /* the captured streams [nt][cap_rows][D] */
    int pend0, pendn;                                 /* their rows not in the context yet: positions pend0.., rows 0.. */
    VkcBuf *f, *tmp, *fk, *fv, *cs;                   /* a forward's context rows */
    VkcBuf *h, *x, *q, *qp, *kb, *vb, *ao, *aop, *g, *u, *hdn, *dcs, *y, *lo;   /* a draft's */
    VkcBuf *ksv, *vsv; float *hsv; size_t sv_n;       /* a verify's ring rows (ink2mc_save), device and host */
    float *rnn, *hout;                                /* the head's matrices, f32 on the host */
    ColiVkTensor *lm;                                 /* the drafts' lm_head (ink2mc_lm_q4), NULL: the target's */
    unsigned long long rollbacks;
    double t_layers, t_cpu, t_lm;                     /* a draft's: its layers' frame, the head on the host, lm_head's */
} Ink2Mc;

static int ink2mc_layer_rows(const Ink2Mtp *mt, int i, int max_t, int *ring) {
    *ring = mt->l[i].local && mt->win > 0;
    if (*ring) return mt->win + 2 * mt->B;
    return (max_t + mt->B + 1023) / 1024 * 1024;   /* the block's rows past the context too */
}
/* the rope table of n positions from p0: (cos, sin) pairs, ink2m_rope's angles */
static float *ink2mc_rope_table(const Ink2Mtp *mt, int p0, int n) {
    const int hd = mt->hd, half = hd / 2;
    float *cs = falloc((int64_t)n * hd);
    for (int r = 0; r < n; r++)
        for (int j = 0; j < half; j++) {
            const double inv = pow((double)mt->theta, -2.0 * j / hd);
            const float ang = (float)((double)(p0 + r) * inv);
            cs[(int64_t)r * hd + 2 * j] = cosf(ang); cs[(int64_t)r * hd + 2 * j + 1] = sinf(ang);
        }
    return cs;
}
/* the rows of positions p0..p0+n-1 of src (kvo floats a row, from src row s0) into a layer's cache */
static int ink2mc_put_rows(VkcBuf *dst, VkcBuf *src, const Ink2McLayer *d, int p0, int n, int s0, int kvo) {
    VkcRegion rg[3]; int nr = 0;
    for (int r = 0; r < n; ) {
        const int p = p0 + r, slot = d->ring ? p % d->rows : p;
        const int run = d->ring && n - r > d->rows - slot ? d->rows - slot : n - r;
        rg[nr++] = (VkcRegion){(size_t)slot * kvo, (size_t)(s0 + r) * kvo, (size_t)run * kvo};
        r += run;
    }
    return vkc_copy_regions(dst, src, rg, nr);
}

/* Before a chain forward of `rows` rows opens a frame: the drafter's matrices and buffers (once),
 * the capture's room and the context caches at the served length. 0 = no drafter on the chain. */
static int ink2mc_prepare(Model *m, int rows) {
    Ink2Mtp *mt = m->mtp; Ink2Chain *ch = ink2c_of(m);
    if (!mt || !ch || !ch->ok) return 0;
    Ink2Mc *mc = mt->mc;
    const Cfg *c = &m->c;
    const int D = mt->D, hd = mt->hd, kvo = mt->KV * hd, qo = mt->H * hd, B = mt->B, I = mt->I, R = mt->R;
    if (!mc) {
        if (!(mc = calloc(1, sizeof *mc))) return 0;
        size_t n = 0;
        for (int i = 0; i < mt->L; i++) {
            Ink2MtpLayer *l = &mt->l[i]; Ink2McLayer *d = &mc->l[i];
            if (!(d->q = inkc_tensor(0, l->q, D, qo)) || !(d->k = inkc_tensor(0, l->k, D, kvo)) ||
                !(d->v = inkc_tensor(0, l->v, D, kvo)) || !(d->kc = inkc_tensor(0, l->kc, D, kvo)) ||
                !(d->vc = inkc_tensor(0, l->vc, D, kvo)) || !(d->o = inkc_tensor(0, l->o, qo, D)) ||
                !(d->g = inkc_tensor(0, l->g, D, I)) || !(d->u = inkc_tensor(0, l->u, D, I)) ||
                !(d->d = inkc_tensor(0, l->d, I, D))) {
                fprintf(stderr, "[VK] INK2_MTP: a drafter matrix did not reach the device; no drafter\n");
                free(mc); return 0;
            }
            d->o_in = n; n += D; d->o_post = n; n += D; d->o_dn = n; n += D;
            d->o_qn = n; n += hd; d->o_kn = n; n += hd; d->o_kcn = n; n += hd;
        }
        mc->o_norm = n; n += D;
        float *a = falloc((int64_t)n);
        for (int i = 0; i < mt->L; i++) {
            Ink2MtpLayer *l = &mt->l[i]; Ink2McLayer *d = &mc->l[i];
            memcpy(a + d->o_in, l->in_ln, D * sizeof(float)); memcpy(a + d->o_post, l->post_ln, D * sizeof(float));
            memcpy(a + d->o_dn, l->dn, D * sizeof(float)); memcpy(a + d->o_qn, l->qn, hd * sizeof(float));
            memcpy(a + d->o_kn, l->kn, hd * sizeof(float)); memcpy(a + d->o_kcn, l->kcn, hd * sizeof(float));
        }
        memcpy(a + mc->o_norm, mt->norm, D * sizeof(float));
        int ok = (mc->prm = vkc_buf(n * sizeof(float), VKC_DEV)) && vkc_begin() && vkc_write(mc->prm, 0, a, n * sizeof(float)) &&
                 vkc_submit(1);
        free(a);
        const int64_t Z = 2 * (int64_t)R + 2 * D;
        mc->rnn = falloc(3 * R * Z); mc->hout = falloc((int64_t)D * R);
        wt_row_f32(mt->rnn, 0, mc->rnn, (int)(3 * R * Z));
        wt_row_f32(mt->hout, 0, mc->hout, D * R);
        size_t sv = 0;
        for (int v = 0; v < c->n_state; v++) sv += (size_t)L_KV(c, v) * L_HD(c, v) * VKC_DMOE_ROWS;
        mc->sv_n = sv;
        ok = ok && (mc->h = vkc_buf((size_t)B * D * 4, VKC_DEV)) && (mc->x = vkc_buf((size_t)B * D * 4, VKC_DEV)) &&
             (mc->q = vkc_buf((size_t)B * qo * 4, VKC_DEV)) && (mc->qp = vkc_buf((size_t)B * qo * 4, VKC_DEV)) &&
             (mc->ao = vkc_buf((size_t)B * qo * 4, VKC_DEV)) && (mc->aop = vkc_buf((size_t)B * qo * 4, VKC_DEV)) &&
             (mc->kb = vkc_buf((size_t)B * kvo * 4, VKC_DEV)) && (mc->vb = vkc_buf((size_t)B * kvo * 4, VKC_DEV)) &&
             (mc->g = vkc_buf((size_t)B * I * 4, VKC_DEV)) && (mc->u = vkc_buf((size_t)B * I * 4, VKC_DEV)) &&
             (mc->hdn = vkc_buf((size_t)B * D * 4, VKC_DOWN)) && (mc->dcs = vkc_buf((size_t)B * hd * 4, VKC_DEV)) &&
             (mc->y = vkc_buf((size_t)D * 4, VKC_DEV)) && (mc->lo = vkc_buf((size_t)c->unpad_vocab * 4, VKC_DOWN)) &&
             (mc->ksv = vkc_buf(sv * 4, VKC_DEV)) && (mc->vsv = vkc_buf(sv * 4, VKC_DEV)) && (mc->hsv = malloc(2 * sv * 4));
        if (!ok) { fprintf(stderr, "[VK] INK2_MTP: no device memory for the drafter; no drafter\n"); return 0; }
        if (mt->lmq && !coli_vk_tensor_ensure(&mc->lm, mt->lmq, mt->lms, 4, c->hidden, c->unpad_vocab, 64)) mc->lm = NULL;
        free(mt->lmq); free(mt->lms); mt->lmq = NULL; mt->lms = NULL;
        mt->mc = mc;
        fprintf(stderr, "[VK] INK2_MTP: the drafter on the chain (%d layers, blocks of %d, %d drafts a step, its head %s)\n",
                mt->L, B, mt->K, mc->lm ? "int4 g64" : "the target's");
    }
    if (rows < 256) rows = 256;   /* the plain steps' rows between two drafts */
    if (rows > mc->cap_rows) {
        mc->pendn = 0;   /* a bigger capture drops what was pending: those positions' context stays as it was */
        const size_t r = (size_t)rows;
        if (!vkc_reserve(&mc->cap, (size_t)mt->nt * r * D * 4, VKC_DEV) || !vkc_reserve(&mc->f, r * D * 4, VKC_DEV) ||
            !vkc_reserve(&mc->tmp, r * D * 4, VKC_DEV) || !vkc_reserve(&mc->fk, r * kvo * 4, VKC_DEV) ||
            !vkc_reserve(&mc->fv, r * kvo * 4, VKC_DEV) || !vkc_reserve(&mc->cs, r * hd * 4, VKC_DEV)) return 0;
        mc->cap_rows = rows;
    }
    for (int i = 0; i < mt->L; i++) {   /* the context caches at the served length, the positions held kept */
        Ink2McLayer *d = &mc->l[i];
        int ring, want = ink2mc_layer_rows(mt, i, m->max_t, &ring);
        if (want <= d->rows) continue;
        VkcBuf *K = vkc_buf((size_t)want * kvo * 4, VKC_DEV), *V = vkc_buf((size_t)want * kvo * 4, VKC_DEV);
        if (!K || !V || (d->K && (!vkc_begin() || !vkc_copy(K, 0, d->K, 0, (size_t)d->rows * kvo) ||
                                  !vkc_copy(V, 0, d->V, 0, (size_t)d->rows * kvo) || !vkc_submit(1)))) {
            vkc_free(K); vkc_free(V); return 0;
        }
        vkc_free(d->K); vkc_free(d->V);
        d->K = K; d->V = V; d->rows = want; d->ring = ring;
    }
    return 1;
}

/* in a chain forward's frame: stream k of its n rows (0 the embedding, k after k layer steps,
 * n_state + 1 the output's mix) where the drafter fuses it */
static int ink2mc_capture(Model *m, int k, VkcBuf *x, int n) {
    Ink2Mtp *mt = m->mtp; Ink2Mc *mc = mt ? mt->mc : NULL;
    if (!mc || mc->pendn + n > mc->cap_rows) return 1;
    for (int j = 0; j < mt->nt; j++)
        if (mt->tl[j] == k && !vkc_copy(mc->cap, ((size_t)j * mc->cap_rows + mc->pendn) * mt->D, x, 0, (size_t)n * mt->D)) return 0;
    return 1;
}
/* The captured rows' context keys and values (ink2m_end's), in the open frame: kept until a draft
 * needs them (ink2mc_flush), so a plain step pays only its four copies and the drafts' frame makes
 * the context of all the rows since the last one at once. */
static int ink2mc_flush(Model *m) {
    Ink2Mtp *mt = m->mtp; Ink2Mc *mc = mt ? mt->mc : NULL;
    if (!mc || !mc->pendn) return 1;
    const int n = mc->pendn, pb = mc->pend0;
    mc->pend0 += n; mc->pendn = 0;
    const int D = mt->D, hd = mt->hd, KV = mt->KV, kvo = KV * hd;
    float *cs = ink2mc_rope_table(mt, pb, n);
    int ok = vkc_write(mc->cs, 0, cs, (size_t)n * hd * 4);
    free(cs);
    for (int i = 0; i < mt->L && ok; i++) {
        Ink2McLayer *d = &mc->l[i];
        for (int j = 0; j < mt->nt && ok; j++) {
            VkcEw sc = {VKC_EW_SCALE, n * D, D, 1, 0, 1, 0, (int)((size_t)j * mc->cap_rows * D), 0, 0, 0, mt->fuse[i][j]};
            VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = j == 0 ? vkc_ew(mc->f, mc->cap, NULL, NULL, NULL, &sc)
                        : vkc_ew(mc->tmp, mc->cap, NULL, NULL, NULL, &sc) && vkc_ew(mc->f, mc->f, mc->tmp, NULL, NULL, &add);
        }
        VkcNorm fn = {n, D, 1, 0, D, D, 0, D, D, (int)d->o_dn, 0, 0, mt->eps, 1.f};
        VkcNorm kn = {n * KV, hd, KV, 0, kvo, hd, 0, kvo, hd, (int)d->o_kcn, 0, 0, mt->eps, 1.f};
        VkcRope rp = {n * KV, KV, 0, kvo, hd, hd / 2, 0, hd};
        const int s0 = d->ring && n > d->rows ? n - d->rows : 0;   /* a ring keeps the last of them */
        ok = ok && vkc_norm(mc->f, mc->prm, mc->f, &fn) && vkc_matmul(d->kc, mc->f, 0, mc->fk, 0, n) &&
             vkc_matmul(d->vc, mc->f, 0, mc->fv, 0, n) && vkc_norm(mc->fk, mc->prm, mc->fk, &kn) && vkc_rope(mc->fk, mc->cs, &rp) &&
             ink2mc_put_rows(d->K, mc->fk, d, pb + s0, n - s0, s0, kvo) && ink2mc_put_rows(d->V, mc->fv, d, pb + s0, n - s0, s0, kvo);
    }
    return ok;
}

/* at a chain forward's end: its n captured rows pending */
static int ink2mc_commit(Model *m, int n) {
    Ink2Mtp *mt = m->mtp; Ink2Mc *mc = mt ? mt->mc : NULL;
    if (mc && mc->pendn + n <= mc->cap_rows) mc->pendn += n;
    return 1;
}
/* at the start of a chain forward's frame of n rows at pb: the pending rows that are still the
 * context before pb, and room for n more (the context made now when there is none) */
static int ink2mc_begin(Model *m, int pb, int n) {
    Ink2Mtp *mt = m->mtp; Ink2Mc *mc = mt ? mt->mc : NULL;
    if (!mc) return 1;
    if (pb != mc->pend0 + mc->pendn) {   /* another position: a rollback's rows, a new prompt */
        if (pb > mc->pend0 && pb < mc->pend0 + mc->pendn) mc->pendn = pb - mc->pend0;
        else if (!ink2mc_flush(m)) return 0;
        if (!mc->pendn) mc->pend0 = pb;
    }
    if (mc->pendn + n > mc->cap_rows) { if (!ink2mc_flush(m)) return 0; mc->pend0 = pb; }
    return 1;
}

/* in a verify's frame, before its layers: the rows of the target's sliding layers (rings) its n
 * rows at pb will overwrite, on the device and on the host */
static int ink2mc_save(Model *m, int pb, int n) {
    Ink2Mtp *mt = m->mtp; Ink2Mc *mc = mt ? mt->mc : NULL; Ink2Chain *ch = ink2c_of(m);
    const Cfg *c = &m->c;
    if (!mc || n > VKC_DMOE_ROWS) return 0;
    size_t o = 0; int ok = 1;
    VkcRegion rg[VKC_DMOE_ROWS * 64];
    for (int v = 0; v < ch->NS && ok; v++) {
        const int cap = ch->cap[v], KV = L_KV(c, v), hd = L_HD(c, v);
        if (cap >= m->max_t) continue;
        if (n * KV > VKC_DMOE_ROWS * 64) return 0;
        int nr = 0;
        for (int r = 0; r < n; r++) for (int h = 0; h < KV; h++, nr++) {
            const size_t off = ((size_t)h * cap + (size_t)(pb + r) % cap) * hd, s = o + (size_t)nr * hd;
            rg[nr] = (VkcRegion){s, off, (size_t)hd};
            memcpy(mc->hsv + s, m->K[v] + off, hd * sizeof(float));
            memcpy(mc->hsv + mc->sv_n + s, m->V[v] + off, hd * sizeof(float));
        }
        ok = vkc_copy_regions(mc->ksv, ch->kc[v], rg, nr) && vkc_copy_regions(mc->vsv, ch->vc[v], rg, nr);
        o += (size_t)nr * hd;
    }
    return ok;
}
/* after a verify of n rows at t that kept a + 1 of them: the states after row a, and the ring rows
 * of rows a + 1.. back (device and host); the host's convolution states the device's */
static int ink2mc_rollback(Model *m, int t, int n, int a) {
    Ink2Mtp *mt = m->mtp; Ink2Mc *mc = mt->mc; Ink2Chain *ch = ink2c_of(m);
    Cfg *c = &m->c;
    size_t o = 0; int ok = vkc_begin();
    VkcRegion rg[VKC_DMOE_ROWS * 64];
    for (int v = 0; v < ch->NS && ok; v++) {
        const int cap = ch->cap[v], KV = L_KV(c, v), hd = L_HD(c, v);
        if (cap >= m->max_t) continue;
        int nr = 0, k = 0;
        for (int r = 0; r < n; r++) for (int h = 0; h < KV; h++, k++) {
            if (r <= a) continue;
            const size_t off = ((size_t)h * cap + (size_t)(t + r) % cap) * hd, s = o + (size_t)k * hd;
            rg[nr++] = (VkcRegion){off, s, (size_t)hd};
            memcpy(m->K[v] + off, mc->hsv + s, hd * sizeof(float));
            memcpy(m->V[v] + off, mc->hsv + mc->sv_n + s, hd * sizeof(float));
        }
        ok = vkc_copy_regions(ch->kc[v], mc->ksv, rg, nr) && vkc_copy_regions(ch->vc[v], mc->vsv, rg, nr);
        o += (size_t)k * hd;
    }
    ok = ok && (!ch->csd_n || (vkc_copy(ch->rings, 0, ch->rings, (size_t)(a + 1) * ch->csd_n, ch->csd_n) &&
                               vkc_copy(ch->csd, 0, ch->rings, 0, ch->csd_n))) && vkc_submit(1);
    if (!ok) return 0;
    const float *csd = (const float *)vkc_ptr(ch->csd);
    for (int i = 0; i < ch->NS; i++) for (int b = 0; b < 4; b++)
        memcpy(m->cs[b][i], csd + ch->o_csd[(size_t)b * ch->NS + i], (size_t)ink_cs_cells(c, b, i) * sizeof(float));
    mc->pendn -= n - (a + 1) < mc->pendn ? n - (a + 1) : mc->pendn;   /* the rejected rows' captures */
    mc->rollbacks++;
    return 1;
}

/* K drafts after `tok` at position t, the context holding positions before t (ink2m_draft's, on
 * the device); 0 = the device would not */
static int ink2mc_draft(Model *m, int tok, int t, int K, int *out) {
    Ink2Mtp *mt = m->mtp; Ink2Mc *mc = mt->mc; Ink2Chain *ch = ink2c_of(m);
    const Cfg *c = &m->c;
    const int B = mt->B, D = mt->D, H = mt->H, KV = mt->KV, hd = mt->hd, qo = H * hd, kvo = KV * hd, grp = H / KV, I = mt->I;
    const double t0 = now_s();
    float *h0 = falloc((int64_t)B * D), *cs = ink2mc_rope_table(mt, t, B);
    ink2m_embed(m, tok, h0);
    for (int r = 1; r < B; r++) memcpy(h0 + (int64_t)r * D, mt->mask, (size_t)D * sizeof(float));
    int ok = vkc_begin() && ink2mc_flush(m) && vkc_write(mc->h, 0, h0, (size_t)B * D * 4) &&
             vkc_write(mc->dcs, 0, cs, (size_t)B * hd * 4);
    free(h0); free(cs);
    VkcRegion qr[INK2M_MAXB * 64], orr[INK2M_MAXB * 64];
    int nq = 0;
    if (B * KV > INK2M_MAXB * 64) return 0;
    for (int r = 0; r < B; r++) for (int g = 0; g < KV; g++, nq++) {   /* q [row][head] <-> qp [kv head][row][its group] */
        qr[nq] = (VkcRegion){((size_t)g * B + r) * grp * hd, (size_t)r * qo + (size_t)g * grp * hd, (size_t)grp * hd};
        orr[nq] = (VkcRegion){qr[nq].src, qr[nq].dst, qr[nq].n};
    }
    VkcEw add = {VKC_EW_ADD, B * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    VkcEw glu = {VKC_EW_SWIGLU, B * I, I, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    for (int i = 0; i < mt->L && ok; i++) {
        Ink2McLayer *d = &mc->l[i];
        VkcNorm ni = {B, D, 1, 0, D, D, 0, D, D, (int)d->o_in, 0, 0, mt->eps, 1.f};
        VkcNorm np = {B, D, 1, 0, D, D, 0, D, D, (int)d->o_post, 0, 0, mt->eps, 1.f};
        VkcNorm qn = {B * H, hd, H, 0, qo, hd, 0, qo, hd, (int)d->o_qn, 0, 0, mt->eps, 1.f};
        VkcNorm kn = {B * KV, hd, KV, 0, kvo, hd, 0, kvo, hd, (int)d->o_kn, 0, 0, mt->eps, 1.f};
        VkcRope rq = {B * H, H, 0, qo, hd, hd / 2, 0, hd}, rk = {B * KV, KV, 0, kvo, hd, hd / 2, 0, hd};
        VkcAttnW w; memset(&w, 0, sizeof w);
        w.a = (VkcAttn){1, B * H, KV, hd, t + B - 1, d->rows, 0, B * qo, hd, 0, 0, 0, 0, 0, B * qo, 0, 0,
                        (float)(1.0 / sqrt((double)hd)), 0, 0};
        w.win = mt->l[i].local && mt->win > 0 ? mt->win + B : 0;
        w.ring = d->ring ? d->rows : 0; w.kv_pm = 1;
        ok = vkc_norm(mc->h, mc->prm, mc->x, &ni) && vkc_matmul(d->q, mc->x, 0, mc->q, 0, B) &&
             vkc_matmul(d->k, mc->x, 0, mc->kb, 0, B) && vkc_matmul(d->v, mc->x, 0, mc->vb, 0, B) &&
             vkc_norm(mc->q, mc->prm, mc->q, &qn) && vkc_norm(mc->kb, mc->prm, mc->kb, &kn) &&
             vkc_rope(mc->q, mc->dcs, &rq) && vkc_rope(mc->kb, mc->dcs, &rk) &&
             ink2mc_put_rows(d->K, mc->kb, d, t, B, 0, kvo) && ink2mc_put_rows(d->V, mc->vb, d, t, B, 0, kvo) &&
             vkc_copy_regions(mc->qp, mc->q, qr, nq) && vkc_attn_w(mc->qp, d->K, d->V, mc->aop, NULL, NULL, NULL, &w) &&
             vkc_copy_regions(mc->ao, mc->aop, orr, nq) && vkc_matmul(d->o, mc->ao, 0, mc->x, 0, B) &&
             vkc_ew(mc->h, mc->h, mc->x, NULL, NULL, &add) && vkc_norm(mc->h, mc->prm, mc->x, &np) &&
             vkc_matmul(d->g, mc->x, 0, mc->g, 0, B) && vkc_matmul(d->u, mc->x, 0, mc->u, 0, B) &&
             vkc_ew(mc->g, mc->g, mc->u, NULL, NULL, &glu) && vkc_matmul(d->d, mc->g, 0, mc->x, 0, B) &&
             vkc_ew(mc->h, mc->h, mc->x, NULL, NULL, &add);
    }
    VkcNorm nf = {B, D, 1, 0, D, D, 0, D, D, (int)mc->o_norm, 0, 0, mt->eps, 1.f};
    ok = ok && vkc_norm(mc->h, mc->prm, mc->x, &nf) && vkc_copy(mc->hdn, 0, mc->x, 0, (size_t)B * D) && vkc_submit(1);
    if (!ok) return 0;
    double tc = now_s();
    mc->t_layers += tc - t0;
    /* the head, row by row (ink2m_draft's), lm_head on the device */
    const int R = mt->R, Z = 2 * R + 2 * D, V = c->unpad_vocab;
    float *h = falloc((int64_t)B * D), *z = falloc(Z), *gco = falloc(3 * R), *s = calloc((size_t)R, sizeof(float)), *ov = falloc(R);
    float *y = falloc(D);
    memcpy(h, vkc_ptr(mc->hdn), (size_t)B * D * sizeof(float));
    int prev = tok;
    for (int k = 1; k <= K && ok; k++) {
        memcpy(z, s, (size_t)R * sizeof(float));
        wt_row_f32(mt->w1, (int64_t)prev * R, z + R, R);
        memcpy(z + 2 * R, h + (int64_t)k * D, (size_t)D * sizeof(float));
        ink2m_embed(m, prev, z + 2 * R + D);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < 3 * R; o++) {
            const float *w = mc->rnn + (int64_t)o * Z; float a = 0.f;
            for (int e = 0; e < Z; e++) a += w[e] * z[e];
            gco[o] = a;
        }
        for (int e = 0; e < R; e++) {
            const float gt = 1.f / (1.f + expf(-(gco[e] + mt->rnn_b[e])));
            s[e] = gt * s[e] + (1.f - gt) * tanhf(gco[R + e] + mt->rnn_b[R + e]);
            ov[e] = tanhf(gco[2 * R + e] + mt->rnn_b[2 * R + e]);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < D; o++) {
            const float *w = mc->hout + (int64_t)o * R; float a = 0.f;
            for (int e = 0; e < R; e++) a += w[e] * ov[e];
            y[o] = h[(int64_t)k * D + o] + a + mt->hout_b[o];
        }
        const double tl = now_s();
        mc->t_cpu += tl - tc;
        ok = vkc_begin() && vkc_write(mc->y, 0, y, (size_t)D * 4) && vkc_matmul(mc->lm ? mc->lm : ch->t_lm, mc->y, 0, mc->lo, 0, 1) && vkc_submit(1);
        tc = now_s();
        mc->t_lm += tc - tl;
        if (!ok) break;
        const float *lo = (const float *)vkc_ptr(mc->lo);
        int best = 0;
        for (int v = 1; v < V; v++) if (lo[v] > lo[best]) best = v;
        out[k - 1] = prev = best;
    }
    free(h); free(z); free(gco); free(s); free(ov); free(y);
    return ok;
}

/* The drafts for the next step: the k that maximizes the tokens a step's second yields, from the
 * kept shares and the measured step times (a k not timed yet: the plain step's time and a quarter
 * of it a draft, as an RTX 5090 measured at k = 3). 0 (plain steps) while drafting does not pay,
 * with a one-draft probe now and then so the shares follow the text; while it pays, every 8th
 * step one draft more than the best k, so the next k gets measured too; and every 65th step a
 * plain one, so its time follows the context's length. */
static int ink2m_pick_k(Ink2Mtp *mt, int kmax) {
    static int adapt = -1;
    if (adapt < 0) { const char *e = getenv("INK2_MTP_ADAPT"); adapt = !(e && *e == '0'); }
    if (!adapt) return kmax;
    if (mt->tk[0] <= 0.0 || ++mt->noplain > 64) { mt->noplain = 0; return 0; }   /* a plain step's time, now and then */
    int best = 0; double bv = 1.0 / mt->tk[0], e = 1.0, pr = 1.0;
    for (int k = 1; k <= kmax; k++) {
        pr *= mt->acc[k]; e += pr;
        const double t = mt->tk[k] > 0.0 ? mt->tk[k] : mt->tk[0] * (1.0 + 0.25 * k);
        if (e / t > bv) { bv = e / t; best = k; }
    }
    if (best == 0) {   /* plain steps, and a one-draft probe every probe_iv of them (4, doubling to 32 while they miss) */
        if (mt->probe_iv < 4) mt->probe_iv = 4;
        if (++mt->plain < (unsigned long long)mt->probe_iv) return 0;
        mt->plain = 0; mt->probing = 1;
        return 1;
    }
    mt->plain = 0;
    if (best < kmax && ++mt->explore >= 8) { mt->explore = 0; best++; }   /* one draft more now and then: its share and time */
    return best;
}
static void ink2m_ema(double *v, double x) { *v = *v > 0.0 ? 0.9 * *v + 0.1 * x : x; }

/* one plain step: the target's token after `tok` at t */
static int ink2m_plain(Model *m, int tok, int t, int *out) {
    float *lg = step_mm(m, &tok, 1, t, NULL, NULL, 0);
    if (!lg) { out[0] = m->c.eos; return 1; }
    int best = 0;
    for (int v = 1; v < m->c.unpad_vocab; v++) if (lg[v] > lg[best]) best = v;
    free(lg);
    out[0] = best;
    return 1;
}
/* ink2m_step on the chain: the drafts, the verify of [tok, drafts] as one chain forward, the
 * states back to after the rows it keeps */
static int ink2mc_step(Model *m, int tok, int t, int *out) {
    Ink2Mtp *mt = m->mtp; Ink2Chain *ch = ink2c_of(m);
    const Cfg *c = &m->c;
    const int V = c->unpad_vocab, D = c->hidden;
    int K = mt->K;
    if (K > VKC_DMOE_ROWS - 1) K = VKC_DMOE_ROWS - 1;
    if (t + K + 1 > m->max_t) K = m->max_t - t - 1;
    if (K < 1 || !ch || !ch->ok || ch->failed || !mt->mc) return ink2m_plain(m, tok, t, out);
    double t0 = now_s();
    if (!g_ink2m_known) K = ink2m_pick_k(mt, K);
    mt->ksteps[K]++;
    if (K == 0) {   /* drafting does not pay here: a plain step, timed */
        const int r = ink2m_plain(m, tok, t, out);
        ink2m_ema(&mt->tk[0], now_s() - t0);
        return r;
    }
    int ids[INK2M_MAXB];
    ids[0] = tok;
    if (g_ink2m_known) ink2m_draft(m, tok, t, K, ids + 1);   /* tests: the oracle's drafts */
    else if (!ink2mc_draft(m, tok, t, K, ids + 1)) return ink2m_plain(m, tok, t, out);
    mt->t_draft += now_s() - t0;
    const int n = K + 1;
    float *x = falloc((int64_t)n * D), *lg = falloc(V), *vl = falloc((int64_t)n * V);
    for (int s = 0; s < n; s++) ink2m_embed(m, ids[s], x + (int64_t)s * D);
    double t1 = now_s();
    ch->vrows = n; ch->snap = (int)ch->csd_n; ch->vlog = vl;
    const int took = ink2c_forward(m, x, n, t, 0, lg);
    ch->vrows = 0; ch->snap = 0; ch->vlog = NULL;
    free(x); free(lg);
    if (took != 1) { free(vl); return ink2m_plain(m, tok, t, out); }   /* nothing fed: declined, or the CPU took over at t */
    mt->t_verify += now_s() - t1;
    m->kv_len = t + n;
    kv_prefix_record(&m->kvp, ids, t, n);
    int tf[INK2M_MAXB];
    for (int r = 0; r < n; r++) {
        const float *lr = vl + (int64_t)r * V;
        int best = 0;
        for (int v = 1; v < V; v++) if (lr[v] > lr[best]) best = v;
        tf[r] = best;
    }
    free(vl);
    int a = 0;
    while (a < K && ids[1 + a] == tf[a]) a++;
    for (int j = 0; j < a; j++) out[j] = ids[1 + j];
    out[a] = tf[a];
    mt->steps++; mt->drafted += (unsigned long long)K; mt->kept += (unsigned long long)a;
    for (int j = 1; j <= K && j <= a + 1; j++) mt->acc[j] = 0.8 * mt->acc[j] + 0.2 * (j <= a);
    if (mt->probing) { mt->probing = 0; mt->probe_iv = a ? 4 : mt->probe_iv < 32 ? 2 * mt->probe_iv : 32; }
    if (a < K) {
        double t2 = now_s();
        m->kv_len = t + a + 1;
        if (m->kvp.len > t + a + 1) m->kvp.len = t + a + 1;
        if (!ink2mc_rollback(m, t, n, a)) {   /* the device was lost: the CPU from the kept rows on */
            if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(0); }
            ink2c_recover(m, t + a + 1);
        }
        mt->t_redo += now_s() - t2;
    }
    ink2m_ema(&mt->tk[K], now_s() - t0);
    return a + 1;
}
#endif /* COLI_VULKAN */

/* One step at the decode position t, `tok` (the target's choice there) not fed yet: its drafts,
 * the verify over [tok, drafts], and the state as if the target had decoded what it keeps.
 * out: the kept drafts and the target's token after them; returns how many (1..K+1). The
 * forward over the kept rows runs again when a draft is rejected (the convolution banks). */
static float *step_mm(Model *m, const int *ids, int S, int pos0, int *tf_out, const uint8_t *dmel, int naud);
static InkPinState *ink_pin_state_save(Model *m, InkPinState *reuse);
static void ink_pin_state_restore(Model *m, const InkPinState *st);
static void ink_pin_state_free(void *v);
static int ink2m_step(Model *m, int tok, int t, int *out) {
#ifdef COLI_VULKAN
    if (g_vk_chain) return g_ink2mc ? ink2mc_step(m, tok, t, out) : ink2m_plain(m, tok, t, out);   /* the state on the device */
#endif
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
    for (int i = 0; i < c->n_state; i++) {   /* a loop pass's layers have rings of their own */
        const int win = kv_ring_rows(c, i, m->max_t);
        if (win < m->max_t) ringn += (size_t)2 * L_KV(c, i) * (K + 1) * L_HD(c, i);
    }
    float *ring = ringn ? falloc((int64_t)ringn) : NULL, *rp = ring;
    for (int i = 0; i < c->n_state; i++) {
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
        for (int i = 0; i < c->n_state; i++) {
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
#ifdef COLI_VULKAN
    const Ink2Mc *mc = mt->mc;
    if (mc) fprintf(stderr, "[VK] INK2_MTP: the drafts: layers %.3f s, the head on the host %.3f s, lm_head %.3f s; %llu rollbacks; "
                            "steps with 0/1/2/3/4+ drafts %llu/%llu/%llu/%llu/%llu\n",
                    mc->t_layers, mc->t_cpu, mc->t_lm, mc->rollbacks, mt->ksteps[0], mt->ksteps[1], mt->ksteps[2], mt->ksteps[3],
                    mt->ksteps[4] + mt->ksteps[5] + mt->ksteps[6] + mt->ksteps[7]);
#endif
}

#endif /* INKLING2_MTP_H */
