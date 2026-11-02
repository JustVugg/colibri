/* Qwen3.6's MTP head (qwen36_mtp.h) on the CUDA chain's device, included by qwen36.c after
 * qwen36_cuda_chain.h. The Vulkan head (q36_mg_*) with the CUDA chain's ops, which are field
 * for field the Vulkan chain's (cuda_chain.h): a fed row is one frame not waited for (the
 * embedding and the model's row in, the input norms, fc, the input norm, k and v, the k norm
 * and RoPE, the K/V row into the head's own device cache); a draft row is one frame waited
 * for (q, attention over that cache, o, the residual, the shared expert, the routing and the
 * routed experts through cc_route and cc_moe, their rank-order sum, mtp.norm, lm_head and,
 * with no listed vocabulary and no Q36_MTP_DUMP, cc_pick: the draft's id and probability).
 * The output after mtp.norm stays on the device for a deeper draft.
 *
 * The head's routed experts are uploaded once as the CUDA tier's formats (int8 rows, or int4
 * with a scale per 64 inputs, the engine's planar blocks turned into the backend's pairs) and
 * listed in an expert table of their device addresses, as the chain's own MoE reads the
 * tier's residents. A head the device cannot hold (another expert format, the f16 dense
 * form, device memory) stays off under the CUDA chain: beside the card the CPU's head costs
 * about half a decode step a draft. */
typedef struct {
    int on, failed, cap, fmt, gs, pick_dev, vids_n;
    uint64_t vids_gen;
    CcBuf *prm, *emb, *hin, *cat, *x, *nrm, *q, *k, *v, *ctx, *tmp, *cs, *kc, *vc;
    CcBuf *lg, *gsb, *us, *hs, *sd, *sg, *ids, *wt, *rt, *fin, *hid, *outd, *pick, *vids, *tab, *mh;
    ColiCudaTensor *eh, *lq, *lk, *lv, *lo, *shg, *shu, *shd, *rq, *sq, *lm, **ex;
    CcBuf *pk, *pv; int pn;   /* a parked conversation's first pn rows of kc and vc (q36_mc_park) */
    size_t o_en, o_hn, o_in, o_post, o_qn, o_kn, o_head;
} Q36MtpCuda;
static Q36MtpCuda g_q36_mc;

static int q36_mc_res(CcBuf **b, size_t floats, int kind) { return cc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }

/* An expert's matrix as the tier uploads it: int8 rows (fmt 1), or the engine's planar int4
 * blocks of 64 as the backend's offset-binary pairs (fmt 4, gs 64; qwen36_tier.c's
 * planar_to_pairs: byte j = element 2j low, 2j + 1 high). */
static ColiCudaTensor *q36_mc_expert(int dev, const QW *w, int *fmt) {
    ColiCudaTensor *t = NULL;
    if (w->q4 && w->sg && w->ng * 64 == w->I) {
        size_t nb = (size_t)w->I * w->O / 2;
        uint8_t *pairs = malloc(nb);
        if (!pairs) return NULL;
        for (size_t b = 0; b < nb; b += 32) {
            const uint8_t *p = w->q4 + b; uint8_t *q = pairs + b;
            for (int j = 0; j < 16; j++) {
                q[j] = (uint8_t)((p[2 * j] & 0x0F) | (p[2 * j + 1] << 4));
                q[16 + j] = (uint8_t)((p[2 * j] >> 4) | (p[2 * j + 1] & 0xF0));
            }
        }
        int ok = coli_cuda_tensor_upload_g(&t, pairs, w->sg, 4, w->I, w->O, dev, 64);
        free(pairs);
        *fmt = 4;
        return ok ? t : NULL;
    }
    if (w->q && w->sc && !w->q8r16) { *fmt = 1; return coli_cuda_tensor_upload(&t, w->q, w->sc, 1, w->I, w->O, dev) ? t : NULL; }
    return NULL;
}

static int q36_mc_init(Model *m) {
    Q36MtpCuda *g = &g_q36_mc; Q36Mtp *p = &g_q36_mtp; Cfg *c = &m->c;
    if (g->on) return 1;
    if (g->failed) return 0;
    Q36CChain *ch = q36cc_of(m);
    const char *e = getenv("Q36_MTP_GPU");
    if ((e && *e == '0') || !g_cuda_chain || !ch || !ch->ok || ch->failed || !cc_moe_ready() || m->mpos || m->rope_delta ||
        p->E > CC_ROUTE_EMAX || p->K > CC_ROUTE_KMAX || p->I > CC_MOE_DIM_MAX || c->hidden > CC_MOE_DIM_MAX) { g->failed = 1; return 0; }
    int dev = ch->dev, D = c->hidden, E = p->E, K = p->K, I = p->I, Ish = p->ish, hd = c->head_dim, kvd = c->k_head_dim;
    g->failed = 1;   /* until every piece is up */
    size_t n = 0;
    g->o_en = n; n += D; g->o_hn = n; n += D; g->o_in = n; n += D; g->o_post = n; n += D;
    g->o_qn = n; n += hd; g->o_kn = n; n += kvd; g->o_head = n; n += D;
    float *w = falloc((int64_t)n);
    memcpy(w + g->o_en, p->enorm, D * 4); memcpy(w + g->o_hn, p->hnorm, D * 4);
    memcpy(w + g->o_in, p->L.in_ln, D * 4); memcpy(w + g->o_post, p->L.post_ln, D * 4);
    memcpy(w + g->o_qn, p->L.qn, hd * 4); memcpy(w + g->o_kn, p->L.kn, kvd * 4); memcpy(w + g->o_head, p->head_norm, D * 4);
    int ok = q36_mc_res(&g->prm, n, CC_DEV) && cc_begin() && cc_write(g->prm, 0, w, n * 4) && cc_submit(1);
    free(w);
    /* the dense matrices as the chain places the trunk's; the router and the shared expert's gate f32 */
    g->eh = ok ? q36cc_qw_tensor(dev, &p->eh) : NULL;
    g->lq = g->eh ? q36cc_qw_tensor(dev, &p->L.q) : NULL; g->lk = g->lq ? q36cc_qw_tensor(dev, &p->L.k) : NULL;
    g->lv = g->lk ? q36cc_qw_tensor(dev, &p->L.v) : NULL; g->lo = g->lv ? q36cc_qw_tensor(dev, &p->L.o) : NULL;
    g->shg = g->lo ? q36cc_qw_tensor(dev, &p->sh_g) : NULL; g->shu = g->shg ? q36cc_qw_tensor(dev, &p->sh_u) : NULL;
    g->shd = g->shu ? q36cc_qw_tensor(dev, &p->sh_d) : NULL;
    g->rq = g->shd ? q36cc_f32_tensor(dev, p->router, D, E) : NULL;
    g->sq = g->rq ? q36cc_f32_tensor(dev, p->sh_gate, D, 1) : NULL;
    g->lm = !g->sq ? NULL : p->nvoc ? q36cc_qw_tensor(dev, &p->dvoc) : ch->t_lm;
    ok = g->lm != NULL;
    /* the routed experts, one format, and their table of device addresses */
    g->ex = ok ? calloc((size_t)3 * E, sizeof *g->ex) : NULL;
    CcExpert *tab = ok ? calloc((size_t)E, sizeof *tab) : NULL;
    ok = ok && g->ex && tab;
    g->fmt = 0;
    for (int x = 0; ok && x < E; x++) {
        const QW *qw[3] = {&p->eg[x], &p->eu[x], &p->ed[x]};
        for (int j = 0; ok && j < 3; j++) {
            int f = 0;
            ColiCudaTensor *t = q36_mc_expert(dev, qw[j], &f);
            ok = t && (!g->fmt || f == g->fmt);
            g->fmt = f; g->ex[(size_t)3 * x + j] = t;
        }
        ok = ok && cc_tensor_dev(g->ex[3 * x], &tab[x].g, &tab[x].gs, NULL) && cc_tensor_dev(g->ex[3 * x + 1], &tab[x].u, &tab[x].us, NULL) &&
             cc_tensor_dev(g->ex[3 * x + 2], &tab[x].d, &tab[x].ds, NULL);
    }
    g->gs = g->fmt == 4 ? 64 : 0;
    ok = ok && cc_reserve(&g->tab, (size_t)E * sizeof(CcExpert), CC_DEV) && cc_begin() &&
         cc_write(g->tab, 0, tab, (size_t)E * sizeof(CcExpert)) && cc_submit(1);
    free(tab);
    const char *dmp = getenv("Q36_MTP_DUMP");
    if (p->voc && !(cc_matmul_rows_ready() && cc_pick_ready())) {   /* an older backend: the whole vocabulary */
        fprintf(stderr, "[mtp] the CUDA backend cannot score a row list (no matmul_rows): the head scores the whole "
                        "vocabulary\n");
        draft_vocab_free(p->voc); p->voc = NULL;
    }
    g->pick_dev = !p->vid && !(dmp && *dmp) && cc_pick_ready();   /* the adaptive list too (never with a dump) */
    ok = ok && (!p->voc || q36_mc_res(&g->vids, c->vocab, CC_DEV)) &&
         q36_mc_res(&g->lg, E, CC_DEV) && q36_mc_res(&g->gsb, Ish, CC_DEV) && q36_mc_res(&g->us, Ish, CC_DEV) &&
         q36_mc_res(&g->hs, Ish, CC_DEV) && q36_mc_res(&g->sd, D, CC_DEV) && q36_mc_res(&g->sg, 1, CC_DEV) &&
         q36_mc_res(&g->ids, K, CC_DEV) && q36_mc_res(&g->wt, K, CC_DEV) && q36_mc_res(&g->rt, D, CC_DEV) &&
         q36_mc_res(&g->mh, cc_moe_hid_floats(1, K, D, I, E), CC_DEV) && q36_mc_res(&g->fin, D, CC_DEV) &&
         q36_mc_res(&g->hid, D, CC_DEV) &&
         q36_mc_res(&g->outd, p->nvoc ? p->nvoc : c->vocab, g->pick_dev ? CC_DEV : CC_DOWN) &&
         (!g->pick_dev || q36_mc_res(&g->pick, 2, CC_DOWN));
    if (!ok) {
        fprintf(stderr, "[mtp] the head does not fit the CUDA chain's device (an expert format, the f16 dense form or "
                        "device memory): it stays off\n");
        return 0;
    }
    g->vids_n = 0; g->vids_gen = 0;
    g->failed = 0; g->on = 1;
    fprintf(stderr, "[mtp] the head runs on the CUDA chain's device, its experts in the frame (format %d)%s\n", g->fmt,
            p->voc ? ", drafts score an adaptive vocabulary" : p->vid ? ", drafts score a listed vocabulary" : "");
    return 1;
}
/* scratch for R rows, the head's device KV cache at the model's row stride */
static int q36_mc_bufs(Model *m, int R) {
    Q36MtpCuda *g = &g_q36_mc; Cfg *c = &m->c;
    int D = c->hidden, H = c->q_heads, KV = c->kv_heads, kvd = c->k_head_dim, qo = H * c->q_head_dim;
    size_t r = (size_t)R;
    if (g->cap != m->max_t) {
        size_t kn = (size_t)KV * m->max_t * kvd;
        CcBuf *kc = NULL, *vc = NULL;
        int ok = q36_mc_res(&kc, kn, CC_DEV) && q36_mc_res(&vc, kn, CC_DEV);
        if (ok && g->cap) {
            int keep = g->cap < m->max_t ? g->cap : m->max_t;
            ok = cc_begin();
            for (int h = 0; ok && h < KV; h++) {
                size_t dst = (size_t)h * m->max_t * kvd, src = (size_t)h * g->cap * kvd;
                ok = cc_copy(kc, dst, g->kc, src, (size_t)keep * kvd) && cc_copy(vc, dst, g->vc, src, (size_t)keep * kvd);
            }
            ok = ok && cc_submit(1);
        }
        if (!ok) { cc_free(kc); cc_free(vc); return 0; }
        cc_free(g->kc); cc_free(g->vc); g->kc = kc; g->vc = vc;
        g->cap = m->max_t;
    }
    return q36_mc_res(&g->emb, r * D, CC_DEV) && q36_mc_res(&g->hin, r * D, CC_DEV) && q36_mc_res(&g->cat, r * 2 * D, CC_DEV) &&
           q36_mc_res(&g->x, r * D, CC_DEV) && q36_mc_res(&g->nrm, r * D, CC_DEV) && q36_mc_res(&g->q, r * qo, CC_DEV) &&
           q36_mc_res(&g->k, r * KV * kvd, CC_DEV) && q36_mc_res(&g->v, r * KV * kvd, CC_DEV) &&
           q36_mc_res(&g->ctx, r * H * c->head_dim, CC_DEV) && q36_mc_res(&g->tmp, r * D, CC_DEV) &&
           q36_mc_res(&g->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2), CC_DEV);
}
/* The first n rows of the head's device cache into a copy of their own (out), or back from
 * it: qwen36_mtp.h's q36_mg_park on the CUDA chain's device. */
static int q36_mc_park(Model *m, int n, int out) {
    Q36MtpCuda *g = &g_q36_mc; int KV = m->c.kv_heads, kvd = m->c.k_head_dim;
    if (out < 0) { cc_free(g->pk); cc_free(g->pv); g->pk = g->pv = NULL; g->pn = 0; return 1; }   /* the copy dropped */
    if (out) {
        g->pn = 0;
        if (!g->on || !g->kc || !g->vc) return 1;
        if (n > g->cap) n = g->cap;
        if (!q36_mc_res(&g->pk, (size_t)KV * n * kvd, CC_DEV) || !q36_mc_res(&g->pv, (size_t)KV * n * kvd, CC_DEV)) return 0;
    } else {
        if (!g->pn) return 1;
        n = g->pn;
    }
    int ok = g->kc && g->vc && n <= g->cap && cc_begin();
    for (int h = 0; ok && h < KV; h++) {
        size_t at = (size_t)h * g->cap * kvd, pa = (size_t)h * n * kvd, len = (size_t)n * kvd;
        ok = out ? cc_copy(g->pk, pa, g->kc, at, len) && cc_copy(g->pv, pa, g->vc, at, len)
                 : cc_copy(g->kc, at, g->pk, pa, len) && cc_copy(g->vc, at, g->pv, pa, len);
    }
    ok = ok && cc_submit(1);
    g->pn = out && ok ? n : 0;
    if (!g->pn) { cc_free(g->pk); cc_free(g->pv); g->pk = g->pv = NULL; }
    return ok;
}
static int q36_mc_norm(CcBuf *x, size_t xo, int xrow, size_t wo, CcBuf *y, size_t yo, int yrow, int rows, int D, float eps) {
    CcNorm np = {rows, D, 1, (int)xo, xrow, D, (int)yo, yrow, D, (int)wo, 0, CC_NORM_ADD1, eps, 1.f};
    return cc_norm(x, g_q36_mc.prm, y, &np);
}
/* rows R at pos0: tokens tok[r], the model's rows hid[r] (host) or, hid NULL, the device
 * state of the row before (R = 1); the head's input up to its K/V rows, stored. The host's
 * rows go in through the open frame (cc_write), ordered after the frames before. */
static int q36_mc_input(Model *m, const int *tok, const float *hid, int R, int pos0) {
    Q36MtpCuda *g = &g_q36_mc; Cfg *c = &m->c; Q36Mtp *p = &g_q36_mtp;
    int D = c->hidden, KV = c->kv_heads, kvd = c->k_head_dim, kvo = KV * kvd, half = c->rotary_dim / 2;
    float *em = falloc((int64_t)R * D), *cs = falloc((int64_t)R * (half > 0 ? 2 * half : 2));
    CcRegion *rg = malloc(sizeof *rg * (size_t)R * KV);
    if (!rg) { free(em); free(cs); return 0; }
    for (int r = 0; r < R; r++) q36_embed_row(m, tok[r], pos0 + r, em + (size_t)r * D);
    for (int s = 0; s < R; s++)
        for (int j = 0; j < half; j++) {
            float ang = (pos0 + s) * powf(c->theta, -2.0f * j / c->rotary_dim);
            cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
        }
    for (int s = 0; s < R; s++) for (int h = 0; h < KV; h++)
        rg[s * KV + h] = (CcRegion){((size_t)h * g->cap + pos0 + s) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
    CcNorm kn = {R * KV, kvd, KV, 0, kvo, kvd, 0, kvo, kvd, (int)g->o_kn, 0, CC_NORM_ADD1, c->eps, 1.f};
    CcRope kr = {R * KV, KV, 0, kvo, kvd, half, 0, 2 * half};
    CcEw r16 = {CC_EW_F16, R * kvo, kvo, 0, 0, 0, 0, 0, 0, 0, 0, 1.f};   /* the rows as the host's cache keeps them */
    int ok = cc_write(g->emb, 0, em, (size_t)R * D * 4) && (!hid || cc_write(g->hin, 0, hid, (size_t)R * D * 4)) &&
             (!half || cc_write(g->cs, 0, cs, (size_t)R * 2 * half * 4)) &&
             q36_mc_norm(g->emb, 0, D, g->o_en, g->cat, 0, 2 * D, R, D, c->eps) &&
             q36_mc_norm(hid ? g->hin : g->hid, 0, D, g->o_hn, g->cat, D, 2 * D, R, D, c->eps) &&
             cc_matmul(g->eh, g->cat, 0, g->x, 0, R) &&
             q36_mc_norm(g->x, 0, D, g->o_in, g->nrm, 0, D, R, D, c->eps) &&
             cc_matmul(g->lk, g->nrm, 0, g->k, 0, R) && cc_matmul(g->lv, g->nrm, 0, g->v, 0, R) &&
             cc_norm(g->k, g->prm, g->k, &kn) && (!half || cc_rope(g->k, g->cs, &kr)) &&
             (!q36_kv16() || !cc_kv16_ready() || (cc_ew(g->k, g->k, NULL, NULL, NULL, &r16) && cc_ew(g->v, g->v, NULL, NULL, NULL, &r16))) &&
             cc_copy_regions(g->kc, g->k, rg, R * KV) && cc_copy_regions(g->vc, g->v, rg, R * KV);
    (void)p;
    free(em); free(cs); free(rg);
    return ok;
}
/* fed rows: their K/V rows only, one frame not waited for */
static int q36_mc_kv_rows(Model *m, const int *tok, const float *hid, int R, int pos0) {
    double t0 = now_s();
    int ok = q36_mc_bufs(m, R) && cc_begin() && q36_mc_input(m, tok, hid, R, pos0) && cc_submit(0);
    g_q36_mtp.kv_rows += (unsigned long long)R; g_q36_mtp.kv_ms += (now_s() - t0) * 1e3;
    return ok;
}
/* a draft row at pos. Its draft id and probability picked on the device (pick_dev: the whole
 * vocabulary or the adaptive list, no Q36_MTP_VOCAB_IDS, no Q36_MTP_DUMP), or its logits brought
 * down: the listed ids' (picked here) or the vocabulary's into `logits`. Its output after mtp.norm
 * stays on the device for a deeper draft */
static int q36_mc_row(Model *m, int tok, const float *hid, int pos, float *logits) {
    Q36MtpCuda *g = &g_q36_mc; Q36Mtp *p = &g_q36_mtp; Cfg *c = &m->c;
    int D = c->hidden, H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim;
    int qo = H * qdim, half = c->rotary_dim / 2, gate_dim = qdim > hd ? qdim - hd : 0;
    int E = p->E, K = p->K, I = p->I, Ish = p->ish, V = p->voc ? p->voc->list_n : p->nvoc ? p->nvoc : c->vocab;
    double t0 = now_s();
    if (!q36_mc_bufs(m, 1) || !cc_begin()) return 0;
    int vok = 1;
    if (p->voc) {   /* the device's copy of the list: all of it after a rebuild, else the ids appended since */
        const int *l = p->voc->list;
        if (g->vids_gen != p->voc_gen) vok = cc_write(g->vids, 0, l, (size_t)V * sizeof(int));
        else if (V > g->vids_n)
            vok = cc_write(g->vids, (size_t)g->vids_n, l + g->vids_n, (size_t)(V - g->vids_n) * sizeof(int));
        g->vids_gen = p->voc_gen; g->vids_n = V;
    }
    CcNorm qn = {H, hd, H, 0, qo, qdim, 0, qo, qdim, (int)g->o_qn, 0, CC_NORM_ADD1, c->eps, 1.f};
    CcRope qr = {H, H, 0, qo, qdim, half, 0, 2 * half};
    CcAttn a = {1, H, KV, hd, pos, g->cap, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0, 1.f / sqrtf((float)hd), 0, 0};
    CcEw add = {CC_EW_ADD, D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    CcEw sw = {CC_EW_SWIGLU, Ish, Ish, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    CcRoute rp = {.S = 1, .E = E, .K = K, .kind = 0, .l_row = E, .i_row = K, .w_row = K,
                  .n_group = 1, .topk_group = 1, .norm = 1, .temp = 1.f, .scale = 1.f};
    CcMoe mp = {.S = 1, .K = K, .D = D, .I = I, .fmt = g->fmt, .gs = g->gs, .act = 0, .parts = 3,
                .x_row = D, .i_row = K, .w_row = K, .y_row = D, .h_off = -1, .E = E};
    CcEw cb1 = {CC_EW_COMBINE, D, D, 1, 1 | 2 | 4, 1, 0, 0, 0, 0, 0, 1.f};
    int ok = vok && q36_mc_input(m, &tok, hid, 1, pos) &&
             cc_matmul(g->lq, g->nrm, 0, g->q, 0, 1) &&
             cc_norm(g->q, g->prm, g->q, &qn) && (!half || cc_rope(g->q, g->cs, &qr)) &&
             cc_attn(g->q, g->kc, g->vc, g->ctx, g->q, NULL, &a) &&
             cc_matmul(g->lo, g->ctx, 0, g->tmp, 0, 1) &&
             cc_ew(g->x, g->x, g->tmp, NULL, NULL, &add) &&
             q36_mc_norm(g->x, 0, D, g->o_post, g->nrm, 0, D, 1, D, c->eps) &&
             cc_matmul(g->rq, g->nrm, 0, g->lg, 0, 1) &&
             cc_matmul(g->shg, g->nrm, 0, g->gsb, 0, 1) && cc_matmul(g->shu, g->nrm, 0, g->us, 0, 1) &&
             cc_ew(g->hs, g->gsb, g->us, NULL, NULL, &sw) &&
             cc_matmul(g->shd, g->hs, 0, g->sd, 0, 1) && cc_matmul(g->sq, g->nrm, 0, g->sg, 0, 1) &&
             cc_route(g->lg, NULL, g->ids, g->wt, &rp) &&
             cc_moe(g->tab, g->nrm, g->ids, g->wt, g->mh, g->rt, NULL, &mp) &&
             cc_ew(g->x, g->x, g->rt, g->sd, g->sg, &cb1) &&
             q36_mc_norm(g->x, 0, D, g->o_head, g->fin, 0, D, 1, D, c->eps) && cc_copy(g->hid, 0, g->fin, 0, D) &&
             (p->voc ? cc_matmul_rows(g->lm, g->vids, g->fin, g->outd, V) : cc_matmul(g->lm, g->fin, 0, g->outd, 0, 1)) &&
             (!g->pick_dev || cc_pick(g->outd, g->pick, V)) && cc_submit(1);
    if (!ok) return 0;
    if (g->pick_dev) {   /* the vocabulary stays in VRAM; only the draft id and its probability cross */
        const void *pk = cc_ptr(g->pick);
        int b = ((const int *)pk)[0];
        p->pick_id = p->voc ? p->voc->list[b] : b; p->last_p = ((const float *)pk)[1]; p->pick_ok = 1;
    } else if (p->vid) {   /* the listed ids' scores, every other id out (as the Vulkan head) */
        const float *o = cc_ptr(g->outd);
        if (getenv("Q36_MTP_DUMP")) {
            for (int i = 0; i < c->vocab; i++) logits[i] = -INFINITY;
            for (int j = 0; j < V; j++) logits[p->vid[j]] = o[j];
        }
        int b = 0;
        for (int j = 1; j < V; j++) if (o[j] > o[b] || (o[j] == o[b] && p->vid[j] < p->vid[b])) b = j;
        float z = 0.f, ob = o[b];
        for (int j = 0; j < V; j++) {
            float d = o[j] - ob;
            if (d > -20.f && !(j > 0 && p->vid[j] == p->vid[j - 1])) z += expf(d);
        }
        p->pick_id = p->vid[b]; p->last_p = z > 0 ? (float)(1.0 / z) : 0.f; p->pick_ok = 1;
    } else {
        const float *o = cc_ptr(g->outd);
        memcpy(logits, o, (size_t)V * 4);
        for (int i = V; i < c->vocab; i++) logits[i] = -INFINITY;
    }
    p->rows++; p->ms += (now_s() - t0) * 1e3;
    return 1;
}
static int q36_mc_on(void) { return g_q36_mc.on; }
