/* inkling2.h: Inkling-2 (Thinking Machines, model_type inkling2) inside inkling.c.
 *
 * Inkling-2-Small keeps Inkling's attention core (grouped heads, sliding and global layers,
 * the relative-position bias bank, short convolutions on K and V, the log-length tau of the
 * global layers) and its routed experts, and changes what is around them:
 *
 *  - block attention residuals: a layer does not read the residual stream as it is. Every
 *    n_layers_per_attn_res_block layers the stream is kept as a block state; a layer's
 *    attention and its MLP read a mix of the block states and the current partial stream,
 *    weighted by a softmax of a learned query against the states, each RMS-normalised
 *    without a weight (the mix is of the normalised states). A global layer has three such
 *    queries, one for each of q, k and v, and q/k/v input norms; the model's output is one
 *    more mix (output_query) before the final norm.
 *  - short convolutions after the attention and after the MLP too, every one of them
 *    (conv + input) / sqrt(K + 1); a global layer's K and V convolutions take per-token taps
 *    projected from their input on top of the static ones, (conv + input) / sqrt(2K + 1).
 *  - attention: no per-head q/k RMSNorm but L2 clips (q and the relative states after a
 *    learned per-head scale), scaling 1/sqrt(head_dim), V normalised without a weight, one
 *    sink logit per head in the softmax's denominator, the output gated by sigmoid(gate_attn).
 *  - experts: no shared ones, every layer sparse, the SwiGLU softcapped
 *    (softcap(g, 4) * sigmoid(g) * softcap(u, 25)), and the logits softcapped at 30.
 *
 * Names are the checkpoint's own (model.llm.*), which transformers' save_pretrained writes
 * too: the tiny oracle's fixture, the int4 container and the real checkpoint read the same.
 * The routed experts' w13 interleaves gate and up rows (even = gate); slot_fill puts them
 * back in gate-then-up order. The reference is transformers' models/inkling2. */
#ifndef INKLING2_H
#define INKLING2_H

typedef struct Ink2Layer {
    float *in_ln;                    /* sliding: attn_norm */
    float *qin_ln, *kin_ln, *vin_ln; /* global: q/k/v_input_norm */
    float *post_ln;                  /* mlp_norm (attn.gate_attn is the Layer's ag) */
    float *sink;                   /* attn.attn_sink_proj [H, D] */
    float *q_clip, *r_clip;          /* attn.q_l2_clip.scale [H, hd], attn.rel_l2_clip.scale [H, d_rel] */
    float *k_st, *v_st;              /* global: attn.k/v_sconv.static_w [K, kvdim] */
    float *k_wp, *v_wp;              /* global: attn.k/v_sconv.weight_proj [KV*K, D] */
    float *aq; int naq;              /* attention_input queries [naq, D]; layer 0 has none */
    float *mq;                       /* mlp_input query [D] */
    /* Inkling-2-Mini's loop layers: the message gates of the query (the checkpoint's controllers.attn),
     * key, value and MLP inputs, down [rank, 2D], forget and gain [D, rank] (NULL off the loop) */
    float *lg_down[4], *lg_fgt[4], *lg_gain[4];
} Ink2Layer;

/* ink_glu (the experts' softcapped SwiGLU) and ink2_head_post (the logits' softcap) are in
 * inkling.c, next to siluf: the expert kernels above this header use them. */

/* x * rsqrt(mean(x^2) + eps), no weight */
static inline void ink2_unorm(float *out, const float *x, int n, float eps) {
    double ms = 0; for (int i = 0; i < n; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r;
}
/* scale v down to norm thr when longer (s: a per-element scale first, or NULL) */
static inline void ink2_l2clip(float *v, const float *s, int n, float thr) {
    if (s) for (int i = 0; i < n; i++) v[i] *= s[i];
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)v[i] * v[i];
    float f = thr / ((float)sqrt(ss) + 1e-6f);
    if (f < 1.f) for (int i = 0; i < n; i++) v[i] *= f;
}

/* A GGUF's short convolutions (st_gguf.h) carry their residual and their scale: a static one
 * g[ch][j] = (w[ch][j] + (j == K-1)) / sqrt(K + 1). Back to the checkpoint's w, in place. */
static void ink2_unfold_sconv(float *g, int64_t C, int K) {
    const float sq = (float)sqrt((double)(K + 1));
    for (int64_t ch = 0; ch < C; ch++) for (int j = 0; j < K; j++)
        g[ch * K + j] = g[ch * K + j] * sq - (j == K - 1 ? 1.f : 0.f);
}
/* a global layer's static taps: g[ch][j] = (static_w[K-1-j][ch] + (j == K-1)) / sqrt(2K + 1), in
 * convolution order; returns static_w [K][C] (malloc'd) */
static float *ink2_unfold_static(const float *g, int64_t C, int K) {
    const float sq = (float)sqrt((double)(2 * K + 1));
    float *st = falloc((int64_t)K * C);
    for (int64_t ch = 0; ch < C; ch++) for (int j = 0; j < K; j++)
        st[(int64_t)(K - 1 - j) * C + ch] = g[ch * K + j] * sq - (j == K - 1 ? 1.f : 0.f);
    return st;
}

/* ---- loading: one layer's v2 tensors (official names) ---- */
static void ink2_load_layer(Model *m, int i, char *nm, size_t nmz) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    Ink2Layer *x = calloc(1, sizeof *x);
    if (!x) { fprintf(stderr, "OOM\n"); exit(1); }
    l->x2 = x;
    #define P2(suffix) (snprintf(nm, nmz, "model.llm.layers.%d." suffix, i), nm)
    if (c->local[i]) x->in_ln = load_t(m, P2("attn_norm.weight"));
    else {
        x->qin_ln = load_t(m, P2("q_input_norm.weight"));
        x->kin_ln = load_t(m, P2("k_input_norm.weight"));
        x->vin_ln = load_t(m, P2("v_input_norm.weight"));
    }
    x->post_ln = load_t(m, P2("mlp_norm.weight"));
    l->q = load_w(m, P2("attn.wq_du.weight"), 1);
    l->k = load_w(m, P2("attn.wk_dv.weight"), 1);
    l->v = load_w(m, P2("attn.wv_dv.weight"), 1);
    l->r = load_w(m, P2("attn.wr_du.weight"), 1);
    l->o = load_w(m, P2("attn.wo_ud.weight"), 1);
    l->ag = load_w(m, P2("attn.gate_attn.weight"), 1);
    x->sink = load_t(m, P2("attn.attn_sink_proj.weight"));
    x->q_clip = load_t(m, P2("attn.q_l2_clip.scale"));
    x->r_clip = load_t(m, P2("attn.rel_l2_clip.scale"));
    const int K = c->conv_k, gg = st_has(&m->S, P2("attn_sconv.gguf"));   /* a GGUF's folded convolutions */
    /* llama.cpp's later names (Inkling-2-Mini's GGUFs): the convolutions as the checkpoint has them but
     * a global layer's static taps in convolution order, the relative bias transposed */
    const int ggn = st_has(&m->S, P2("attn.k_sconv.ggufn")), ggq = st_has(&m->S, P2("attn_res.mlp_query"));
    if (st_has(&m->S, P2("attn.rel_logits_proj.gguf_t"))) {
        const int ext = L_EXT(c, i), dr = c->d_rel;
        float *t = load_t(m, P2("attn.rel_logits_proj.gguf_t"));   /* [ext, d_rel] */
        if (st_numel(&m->S, nm) != (int64_t)ext * dr) { fprintf(stderr, "layer %d: %s is not %d x %d\n", i, nm, ext, dr); exit(1); }
        l->relp = falloc((int64_t)ext * dr);
        for (int a = 0; a < ext; a++) for (int b = 0; b < dr; b++) l->relp[(int64_t)b * ext + a] = t[(int64_t)a * dr + b];
        free(t);
    } else l->relp = load_t(m, P2("attn.rel_logits_proj.proj"));
    const int64_t kvdim = (int64_t)L_KV(c, i) * L_HD(c, i);
    if (gg) {
        const float sq = (float)sqrt((double)(2 * K + 1));
        if (c->local[i]) {
            ink2_unfold_sconv(l->k_cw = load_t(m, P2("attn.k_sconv.gguf")), kvdim, K);
            ink2_unfold_sconv(l->v_cw = load_t(m, P2("attn.v_sconv.gguf")), kvdim, K);
        } else {
            float *g = load_t(m, P2("attn.k_sconv.gguf")); x->k_st = ink2_unfold_static(g, kvdim, K); free(g);
            g = load_t(m, P2("attn.v_sconv.gguf")); x->v_st = ink2_unfold_static(g, kvdim, K); free(g);
            x->k_wp = load_t(m, P2("attn.k_sconv.gguf_dyn"));
            x->v_wp = load_t(m, P2("attn.v_sconv.gguf_dyn"));
            for (int64_t j = 0; j < (int64_t)L_KV(c, i) * K * c->hidden; j++) { x->k_wp[j] *= sq; x->v_wp[j] *= sq; }
        }
        ink2_unfold_sconv(l->a_cw = load_t(m, P2("attn_sconv.gguf")), c->hidden, K);
        ink2_unfold_sconv(l->m_cw = load_t(m, P2("mlp_sconv.gguf")), c->hidden, K);
    } else if (ggn && c->local[i]) {   /* [C, K] as the checkpoint's [C, 1, K] */
        l->k_cw = load_t(m, P2("attn.k_sconv.ggufn"));
        l->v_cw = load_t(m, P2("attn.v_sconv.ggufn"));
    } else if (ggn) {   /* the static taps [C, K] in convolution order: the checkpoint's static_w [K, C] reversed */
        float **dst[2] = { &x->k_st, &x->v_st }; const char *src[2] = { "attn.k_sconv.ggufn", "attn.v_sconv.ggufn" };
        for (int j = 0; j < 2; j++) {
            snprintf(nm, nmz, "model.llm.layers.%d.%s", i, src[j]);
            if (st_numel(&m->S, nm) != kvdim * K) { fprintf(stderr, "layer %d: %s is not %lld x %d\n", i, nm, (long long)kvdim, K); exit(1); }
            float *g = load_t(m, nm), *o = falloc(kvdim * K);
            for (int64_t ch = 0; ch < kvdim; ch++) for (int k = 0; k < K; k++) o[(int64_t)k * kvdim + ch] = g[ch * K + (K - 1 - k)];
            free(g); *dst[j] = o;
        }
        x->k_wp = load_t(m, P2("attn.k_sconv.weight_proj"));
        x->v_wp = load_t(m, P2("attn.v_sconv.weight_proj"));
    } else if (c->local[i]) {
        l->k_cw = load_t(m, P2("attn.k_sconv.weight"));
        l->v_cw = load_t(m, P2("attn.v_sconv.weight"));
    } else {
        x->k_st = load_t(m, P2("attn.k_sconv.static_w"));
        x->v_st = load_t(m, P2("attn.v_sconv.static_w"));
        x->k_wp = load_t(m, P2("attn.k_sconv.weight_proj"));
        x->v_wp = load_t(m, P2("attn.v_sconv.weight_proj"));
    }
    if (!gg) {
        l->a_cw = load_t(m, P2("attn_sconv.weight"));
        l->m_cw = load_t(m, P2("mlp_sconv.weight"));
    }
    /* a GGUF's NVFP4 layers: the experts come without their global scale (one F32 an expert) */
    if (st_has(&m->S, P2("mlp.experts.gate_scale"))) {
        l->xgsc = load_t(m, P2("mlp.experts.gate_scale"));
        l->xusc = load_t(m, P2("mlp.experts.up_scale"));
        l->xdsc = load_t(m, P2("mlp.experts.down_scale"));
    }
    l->router = load_t(m, P2("mlp.gate.weight"));
    l->rbias  = load_t(m, P2("mlp.gate.bias"));
    l->rgs = 1.f;
    #undef P2
    /* the attention residuals' queries: layer i reads attn_queries.(i-1) ([D], or [3, D] on a
     * global layer: q, k and v) and mlp_queries.i */
    if (i > 0) {
        if (gg || ggq) snprintf(nm, nmz, "model.llm.layers.%d.attn_res.attn_query", i);   /* the GGUF's, by layer */
        else snprintf(nm, nmz, "model.llm.attn_res.attn_queries.%d", i - 1);
        int64_t n = st_numel(&m->S, nm);
        x->naq = n == (int64_t)c->hidden ? 1 : n == 3LL * c->hidden ? 3 : 0;
        if (!x->naq || (x->naq == 3) != !c->local[i]) {
            fprintf(stderr, "layer %d: %s has %lld values, not %s x %d\n", i, nm, (long long)n,
                    c->local[i] ? "1" : "3", c->hidden);
            exit(1);
        }
        x->aq = load_t(m, nm);
    }
    if (gg || ggq) snprintf(nm, nmz, "model.llm.layers.%d.attn_res.mlp_query", i);
    else snprintf(nm, nmz, "model.llm.attn_res.mlp_queries.%d", i);
    x->mq = load_t(m, nm);
    if (c->loop_p > 1 && i >= c->loop_s && i < c->loop_e) {   /* Inkling-2-Mini: a loop layer's message gates */
        static const char *const g[4] = { "attn", "k", "v", "mlp" };
        const int64_t D = c->hidden, r = c->loop_r;
        for (int j = 0; j < 4; j++) {
            static const char *const part[3] = { "down_weight", "forget_weight", "gain_weight" };
            float **dst[3] = { &x->lg_down[j], &x->lg_fgt[j], &x->lg_gain[j] };
            const int64_t want[3] = { r * 2 * D, D * r, D * r };
            for (int k = 0; k < 3; k++) {
                snprintf(nm, nmz, "model.llm.layers.%d.controllers.%s.%s", i, g[j], part[k]);
                if (st_numel(&m->S, nm) != want[k]) {
                    fprintf(stderr, "layer %d: %s has %lld values, not %lld\n", i, nm, (long long)st_numel(&m->S, nm), (long long)want[k]);
                    exit(1);
                }
                *dst[k] = load_t(m, nm);
            }
        }
    }
}

/* ---- the short convolutions ---- */
/* static: (conv + input) / sqrt(K + 1), over a sequence through bank `bank` or one row of each
 * multiplexed conversation */
static void ink2_sconv(Model *m, float *seq, int S, int C, const float *w, int bank, int li) {
    ink_sconv(m, seq, S, C, w, bank, li);
    const float sc = (float)sqrt((double)(m->c.conv_k + 1));
    for (int64_t j = 0; j < (int64_t)S * C; j++) seq[j] /= sc;
}
/* dynamic, a global layer's K or V: tap `lag` (0 = this token) of channel ch in KV head h is
 * static_w[lag][ch] + (weight_proj . xin)[h*K + lag]; (conv + input) / sqrt(2K + 1). state
 * keeps the last K-1 raw inputs per channel, oldest first, as sconv_apply's does. */
static void ink2_dsconv_rows(float *seq, int S, int C, int hd, const float *st, const float *dyn,
                             float *state, int K) {
    const int P = K - 1;
    const float sc = (float)sqrt((double)(2 * K + 1));
    float *col = malloc((size_t)(P + S) * sizeof(float));
    if (!col) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int ch = 0; ch < C; ch++) {
        const int h = ch / hd;
        for (int j = 0; j < P; j++) col[j] = state[(int64_t)ch * P + j];
        for (int t = 0; t < S; t++) col[P + t] = seq[(int64_t)t * C + ch];
        for (int t = 0; t < S; t++) {
            const float *dw = dyn + (int64_t)t * (C / hd) * K + (int64_t)h * K;
            float acc = 0.f;
            for (int lag = 0; lag < K; lag++) acc += col[P + t - lag] * (st[(int64_t)lag * C + ch] + dw[lag]);
            seq[(int64_t)t * C + ch] = (acc + col[P + t]) / sc;
        }
        for (int j = 0; j < P; j++) state[(int64_t)ch * P + j] = col[S + j];
    }
    free(col);
}
static void ink2_dsconv(Model *m, float *seq, const float *xin, int S, int C, int hd, const float *st,
                        const float *wp, int bank, int li) {
    const int K = m->c.conv_k, KV = C / hd, D = m->c.hidden;
    float *dyn = falloc((int64_t)S * KV * K);
    matmul(dyn, xin, wp, S, D, KV * K);
    if (!m->mux_rows) ink2_dsconv_rows(seq, S, C, hd, st, dyn, m->cs[bank][li], K);
    else for (int s = 0; s < S; s++)
        ink2_dsconv_rows(seq + (int64_t)s * C, 1, C, hd, st, dyn + (int64_t)s * KV * K,
                         m->mux_rows[s].seq->cs[bank][li], K);
    free(dyn);
}

/* ---- attention ----
 * xq feeds q, the relative states, the gate and the sinks; xk and xv the K and V
 * projections (and their dynamic taps). A sliding layer passes the same rows three times. */
static void ink2_attention(Model *m, Layer *l, int li, const float *xq, const float *xk, const float *xv,
                           int S, int pos0, float *out) {
    Cfg *c = &m->c;
    Ink2Layer *x2 = l->x2;
    const int D = c->hidden, H = L_HEADS(c, li), KV = L_KV(c, li), hd = L_HD(c, li), ext = L_EXT(c, li);
    const int local = c->local[li], dr = c->d_rel;
    const InkRow *mr = m->mux_rows;
    if (mr) { for (int s = 0; s < S; s++) if (mr[s].pos >= m->max_t) { fprintf(stderr, "attention: pos %d exceeds kv alloc %d\n", mr[s].pos, m->max_t); exit(1); } }
    else if (pos0 + S > m->max_t) { fprintf(stderr, "attention: pos %d+%d exceeds kv alloc %d\n", pos0, S, m->max_t); exit(1); }
    const int qdim = H * hd, kvdim = KV * hd, group = H / KV;
    float *q = falloc((int64_t)S * qdim), *k = falloc((int64_t)S * kvdim), *vv = falloc((int64_t)S * kvdim);
    float *rr = falloc((int64_t)S * H * dr), *gt = falloc((int64_t)S * qdim), *sk = falloc((int64_t)S * H);
    matmul_w(q, xq, l->q, S, D, qdim);
    matmul_w(k, xk, l->k, S, D, kvdim);
    matmul_w(vv, xv, l->v, S, D, kvdim);
    matmul_w(rr, xq, l->r, S, D, H * dr);
    matmul_w(gt, xq, l->ag, S, D, qdim);
    matmul(sk, xq, x2->sink, S, D, H);
    if (local) {
        ink2_sconv(m, k, S, kvdim, l->k_cw, 0, li);
        ink2_sconv(m, vv, S, kvdim, l->v_cw, 1, li);
    } else {
        ink2_dsconv(m, k, xk, S, kvdim, hd, x2->k_st, x2->k_wp, 0, li);
        ink2_dsconv(m, vv, xv, S, kvdim, hd, x2->v_st, x2->v_wp, 1, li);
    }
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H; h++) {
            ink2_l2clip(q + (int64_t)s * qdim + h * hd, x2->q_clip + (int64_t)h * hd, hd, c->qk_clip);
            ink2_l2clip(rr + (int64_t)s * H * dr + h * dr, x2->r_clip + (int64_t)h * dr, dr, c->rel_clip);
        }
        for (int h = 0; h < KV; h++) {
            ink2_l2clip(k + (int64_t)s * kvdim + h * hd, NULL, hd, c->qk_clip);
            ink2_unorm(vv + (int64_t)s * kvdim + h * hd, vv + (int64_t)s * kvdim + h * hd, hd, c->eps);
        }
    }
    const int win = kv_ring_rows(c, li, m->max_t);
    const float scale = (float)(1.0 / sqrt((double)hd));
    float *ctx = falloc((int64_t)S * qdim);
    #pragma omp parallel
    {
        float *rl = malloc((size_t)ext * sizeof(float));
        float *sc = malloc((size_t)m->max_t * sizeof(float));
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < S; s++) {
                const int qpos = mr ? mr[s].pos : pos0 + s;
                const int sp0 = mr ? qpos : pos0;
                const int t0 = local && qpos - c->window + 1 > 0 ? qpos - c->window + 1 : 0;
                const int tb = sp0 > t0 ? sp0 : t0;
                float **Kc = mr ? mr[s].seq->K : m->K, **Vc = mr ? mr[s].seq->V : m->V;
                const int64_t srow = mr ? (int64_t)s * kvdim : 0;
                const float *rv = rr + (int64_t)s * H * dr + h * dr;
                for (int e = 0; e < ext; e++) {
                    float acc = 0.f;
                    for (int d = 0; d < dr; d++) acc += rv[d] * l->relp[(int64_t)d * ext + e];
                    rl[e] = acc;
                }
                float tau = 1.f;   /* global layers: log-length scaling of q and the relative states */
                if (!local && c->log_floor > 0) {
                    double en = (double)(qpos + 1) / c->log_floor;
                    if (en > 1.0) tau = 1.f + c->log_alpha * (float)log(en);
                }
                const float *qv = q + (int64_t)s * qdim + h * hd;
                const float *Kh = Kc[li] + ((int64_t)(h / group) * win) * hd;
                const float *Kb = k + srow + (int64_t)(h / group) * hd;
                float mx = sk[(int64_t)s * H + h];
                for (int t = t0; t <= qpos; t++) {
                    const float *kv = t < tb ? Kh + (int64_t)(t % win) * hd : Kb + (int64_t)(t - sp0) * kvdim;
                    const int dist = qpos - t;
                    float lg = tau * (dot_f32_lanes(qv, kv, hd) * scale + (dist < ext ? rl[dist] : 0.f));
                    sc[t - t0] = lg;
                    if (lg > mx) mx = lg;
                }
                const int n = qpos - t0 + 1;
                float den = expf(sk[(int64_t)s * H + h] - mx);   /* the sink: in the denominator only */
                for (int j = 0; j < n; j++) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
                float *cx = ctx + (int64_t)s * qdim + h * hd;
                for (int d = 0; d < hd; d++) cx[d] = 0.f;
                const float *Vh = Vc[li] + ((int64_t)(h / group) * win) * hd;
                const float *Vb = vv + srow + (int64_t)(h / group) * hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *vrow = t < tb ? Vh + (int64_t)(t % win) * hd : Vb + (int64_t)(t - sp0) * kvdim;
                    const float a = sc[t - t0] / den;
                    for (int d = 0; d < hd; d++) cx[d] += a * vrow[d];
                }
                const float *gr = gt + (int64_t)s * qdim + h * hd;
                for (int d = 0; d < hd; d++) cx[d] *= 1.f / (1.f + expf(-gr[d]));
            }
        }
        free(rl); free(sc);
    }
    const int s0 = !mr && S - win > 0 ? S - win : 0;
    for (int s = s0; s < S; s++) for (int h = 0; h < KV; h++) {
        const int t = mr ? mr[s].pos : pos0 + s;
        float *Kd = mr ? mr[s].seq->K[li] : m->K[li], *Vd = mr ? mr[s].seq->V[li] : m->V[li];
        memcpy(Kd + ((int64_t)h * win + t % win) * hd, k + (int64_t)s * kvdim + h * hd, hd * sizeof(float));
        memcpy(Vd + ((int64_t)h * win + t % win) * hd, vv + (int64_t)s * kvdim + h * hd, hd * sizeof(float));
    }
    matmul_w(out, ctx, l->o, S, qdim, D);
    free(q); free(k); free(vv); free(rr); free(gt); free(sk); free(ctx);
}

/* ---- block attention residuals ----
 * The mix the queries qs[nq][D] take of the B block states and the partial stream at each of S
 * rows: the states RMS-normalised without a weight, softmax(q . n / sqrt(D)) over them, the
 * normalised states weighted (outs[j] for query j, [S, D] each). */
static void ink2_attn_res(const Cfg *c, const float *qs, int nq, float *const *blocks, int B,
                          const float *partial, int S, float **outs) {
    const int D = c->hidden, n = B + 1;
    const float scale = (float)(1.0 / sqrt((double)D));
    #pragma omp parallel
    {
        float *nrm = malloc((size_t)n * D * sizeof(float)), *w = malloc((size_t)n * sizeof(float));
        #pragma omp for schedule(static)
        for (int s = 0; s < S; s++) {
            for (int b = 0; b < n; b++) {
                const float *src = b < B ? blocks[b] + (int64_t)s * D : partial + (int64_t)s * D;
                ink2_unorm(nrm + (int64_t)b * D, src, D, 1e-6f);
            }
            for (int j = 0; j < nq; j++) {
                const float *q = qs + (int64_t)j * D;
                float mx = -1e30f;
                for (int b = 0; b < n; b++) { w[b] = dot_f32_lanes(q, nrm + (int64_t)b * D, D) * scale; if (w[b] > mx) mx = w[b]; }
                float sum = 0.f;
                for (int b = 0; b < n; b++) { w[b] = expf(w[b] - mx); sum += w[b]; }
                float *o = outs[j] + (int64_t)s * D;
                for (int d = 0; d < D; d++) o[d] = 0.f;
                for (int b = 0; b < n; b++) {
                    const float a = w[b] / sum, *nb = nrm + (int64_t)b * D;
                    for (int d = 0; d < D; d++) o[d] += a * nb[d];
                }
            }
        }
        free(nrm); free(w);
    }
}

static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out);

/* ---- Inkling-2-Mini's loop ----
 * A message gate: out = (1 - tanh(F z)) * h + tanh(G z) * m, z = W [unorm(h); unorm(m)] (W [rank, 2D],
 * F and G [D, rank]), row by row in f32. out may be h. */
static void ink2_loop_gate(const Cfg *c, const float *W, const float *F, const float *G, float *out,
                           const float *h, const float *msg, int S) {
    const int D = c->hidden, r = c->loop_r;
    #pragma omp parallel
    {
        float *hn = malloc((size_t)2 * D * sizeof(float)), *z = malloc((size_t)r * sizeof(float));
        if (!hn || !z) { fprintf(stderr, "OOM\n"); exit(1); }
        #pragma omp for schedule(static)
        for (int s = 0; s < S; s++) {
            const float *hs = h + (int64_t)s * D, *ms = msg + (int64_t)s * D;
            float *os = out + (int64_t)s * D;
            ink2_unorm(hn, hs, D, c->eps);
            ink2_unorm(hn + D, ms, D, c->eps);
            for (int k = 0; k < r; k++) z[k] = dot_f32_lanes(W + (int64_t)k * 2 * D, hn, 2 * D);
            for (int d = 0; d < D; d++) {
                const float *fr = F + (int64_t)d * r, *gr = G + (int64_t)d * r;
                float f = 0.f, g = 0.f;
                for (int k = 0; k < r; k++) { f += fr[k] * z[k]; g += gr[k] * z[k]; }
                os[d] = (1.f - tanhf(f)) * hs[d] + tanhf(g) * ms[d];
            }
        }
        free(hn); free(z);
    }
}

static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out);

/* the scratch one layer needs, [S, D] each */
typedef struct { float *hq, *hk, *hv, *nq, *nk, *nv, *a, *res; } Ink2Scratch;

/* One layer: weight layer i, its state (key/value rows, convolutions) in state layer v (i outside a
 * loop's later passes). msg: the four messages of a loop pass after the first (q, k, v, mlp), NULL
 * otherwise. x [S, D] in and out; a block boundary appends this layer's input to blocks. */
static void ink2_layer(Model *m, int i, int v, float *x, int S, int pos0, float **blocks, int *Bp, int maxb,
                       float *const *msg, Ink2Scratch *w) {
    Cfg *c = &m->c;
    const int D = c->hidden, blk = c->res_block > 0 ? c->res_block : 1;
    const int64_t SD = (int64_t)S * D;
    Layer *l = &m->L[i];
    Ink2Layer *x2 = l->x2;
    const int local = c->local[i];
    float *hq = w->hq, *hk = w->hk, *hv = w->hv, *nq = w->nq, *nk = w->nk, *nv = w->nv, *a = w->a, *res = w->res;
    int B = *Bp;
    /* the attention's input: the mix of the blocks so far and this stream (layer 0: the stream) */
    const float *iq = x, *ik = x, *iv = x;
    if (x2->aq) {
        float *outs[3] = { hq, hk, hv };
        ink2_attn_res(c, x2->aq, x2->naq, blocks, B, x, S, outs);
        iq = hq; ik = local ? hq : hk; iv = local ? hq : hv;
    }
    if (msg) {   /* a later loop pass: each input mixed with its message (a sliding layer's one mix three ways) */
        ink2_loop_gate(c, x2->lg_down[1], x2->lg_fgt[1], x2->lg_gain[1], hk, ik, msg[1], S);
        ink2_loop_gate(c, x2->lg_down[2], x2->lg_fgt[2], x2->lg_gain[2], hv, iv, msg[2], S);
        ink2_loop_gate(c, x2->lg_down[0], x2->lg_fgt[0], x2->lg_gain[0], hq, iq, msg[0], S);
        iq = hq; ik = hk; iv = hv;
    }
    double ta = now_s();
    if (local && !msg) {
        for (int s = 0; s < S; s++) rmsnorm_row(nq + (int64_t)s * D, iq + (int64_t)s * D, x2->in_ln, D, c->eps);
        ink2_attention(m, l, v, nq, nq, nq, S, pos0, a);
    } else {
        const float *gq = local ? x2->in_ln : x2->qin_ln, *gk = local ? x2->in_ln : x2->kin_ln, *gv = local ? x2->in_ln : x2->vin_ln;
        for (int s = 0; s < S; s++) {
            rmsnorm_row(nq + (int64_t)s * D, iq + (int64_t)s * D, gq, D, c->eps);
            rmsnorm_row(nk + (int64_t)s * D, ik + (int64_t)s * D, gk, D, c->eps);
            rmsnorm_row(nv + (int64_t)s * D, iv + (int64_t)s * D, gv, D, c->eps);
        }
        ink2_attention(m, l, v, nq, nk, nv, S, pos0, a);
    }
    m->t_attn += now_s() - ta;
    ink2_sconv(m, a, S, D, l->a_cw, 2, v);
    /* a block boundary keeps this layer's input stream as a block, and the stream starts over
     * from the attention's output; elsewhere the output adds to it */
    if (i % blk == 0) {
        if (B == maxb) { fprintf(stderr, "inkling2: more than %d blocks\n", maxb); exit(1); }
        blocks[B] = falloc(SD);
        memcpy(blocks[B++], x, (size_t)SD * sizeof(float));
        memcpy(res, a, (size_t)SD * sizeof(float));
    } else for (int64_t j = 0; j < SD; j++) res[j] = x[j] + a[j];
    float *mo = hq;   /* the MLP's input mix, then its normed input, then its output */
    float *outs[1] = { mo };
    ink2_attn_res(c, x2->mq, 1, blocks, B, res, S, outs);
    if (msg) ink2_loop_gate(c, x2->lg_down[3], x2->lg_fgt[3], x2->lg_gain[3], mo, mo, msg[3], S);
    for (int s = 0; s < S; s++) rmsnorm_row(nq + (int64_t)s * D, mo + (int64_t)s * D, x2->post_ln, D, c->eps);
    moe(m, l, i, nq, S, mo);
    ink2_sconv(m, mo, S, D, l->m_cw, 3, v);
    for (int64_t j = 0; j < SD; j++) x[j] = res[j] + mo[j];
    *Bp = B;
}

/* ---- the layers: x [S, D] in, the stream the final norm reads out (the output mix applied) ----
 * Inkling-2-Mini: layers [loop_s, loop_e) run loop_p passes. Every pass starts from the stream and
 * the blocks the layers before the loop left, with key/value and convolution states of its own
 * (state layers ink_sl), and from the second pass on each loop layer takes four messages: the mix
 * its queries take of the previous pass's blocks (but the first, its own input) and that pass's
 * output stream. The layers after the loop read the last pass's. The drafter's capture points
 * count the layers as they run (n_state of them). */
static void ink2_layers_forward(Model *m, float *x, int S, int pos0) {
    Cfg *c = &m->c;
    const int D = c->hidden, L = c->n_layers, blk = c->res_block > 0 ? c->res_block : 1;
    const int64_t SD = (int64_t)S * D;
    const int maxb = (L + blk - 1) / blk;
    const int loop = c->loop_p > 1, ls = loop ? c->loop_s : L, le = loop ? c->loop_e : L;
    float **blocks = calloc((size_t)maxb, sizeof(float *)), **prevb = calloc((size_t)maxb, sizeof(float *));
    Ink2Scratch w = { falloc(SD), falloc(SD), falloc(SD), falloc(SD), falloc(SD), falloc(SD), falloc(SD), falloc(SD) };
    if (!blocks || !prevb) { fprintf(stderr, "OOM\n"); exit(1); }
    int B = 0, app = 0;
    ink2m_begin(m, S);   /* INK2_MTP: the drafter's fused streams of this forward */
    ink2m_capture(m, 0, x, S);
    for (int i = 0; i < ls && !coli_prompt_break(); i++) {
        ink2_layer(m, i, i, x, S, pos0, blocks, &B, maxb, NULL, &w);
        ink2m_capture(m, ++app, x, S);
    }
    if (loop && !coli_prompt_broken()) {
        float *pre = falloc(SD), *xprev = falloc(SD), *mbuf[4] = { falloc(SD), falloc(SD), falloc(SD), falloc(SD) };
        memcpy(pre, x, (size_t)SD * sizeof(float));
        const int preB = B; int nprev = 0;
        for (int p = 0; p < c->loop_p && !coli_prompt_broken(); p++) {
            if (p > 0) {   /* the previous pass's blocks but its first, and its output, for the messages */
                for (int b = 0; b < nprev; b++) free(prevb[b]);
                nprev = 0;
                for (int b = preB; b < B; b++) { if (b == preB) free(blocks[b]); else prevb[nprev++] = blocks[b]; blocks[b] = NULL; }
                B = preB;
                memcpy(xprev, x, (size_t)SD * sizeof(float));
                memcpy(x, pre, (size_t)SD * sizeof(float));
            }
            for (int i = ls; i < le && !coli_prompt_break(); i++) {
                if (p > 0) ink2_attn_res(c, m->x2_loopq + (int64_t)(i - ls) * 4 * D, 4, prevb, nprev, xprev, S, mbuf);
                ink2_layer(m, i, ink_sl(c, p, i), x, S, pos0, blocks, &B, maxb, p > 0 ? mbuf : NULL, &w);
                ink2m_capture(m, ++app, x, S);
            }
        }
        for (int b = 0; b < nprev; b++) free(prevb[b]);
        free(pre); free(xprev); for (int j = 0; j < 4; j++) free(mbuf[j]);
    }
    for (int i = le; i < L && !coli_prompt_break(); i++) {
        ink2_layer(m, i, i, x, S, pos0, blocks, &B, maxb, NULL, &w);
        ink2m_capture(m, ++app, x, S);
    }
    if (!coli_prompt_broken()) {   /* the output's mix of the blocks and the last stream */
        float *outs[1] = { w.hq };
        ink2_attn_res(c, m->x2_out_query, 1, blocks, B, x, S, outs);
        memcpy(x, w.hq, (size_t)SD * sizeof(float));
        ink2m_capture(m, c->n_state + 1, x, S);
        ink2m_end(m, pos0, S);
    }
    for (int b = 0; b < B; b++) free(blocks[b]);
    free(blocks); free(prevb);
    free(w.hq); free(w.hk); free(w.hv); free(w.nq); free(w.nk); free(w.nv); free(w.a); free(w.res);
}

#endif /* INKLING2_H */
