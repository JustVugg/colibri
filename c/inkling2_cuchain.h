/* inkling2_cuchain.h -- Inkling-2's layers as a dense chain on a CUDA device (cuda_chain.h), the
 * twin of inkling2_chain.h (Vulkan): the same frames, state and fallbacks, read there; the ops are
 * cuda_chain.h's, Inkling's and Inkling-2's in cuda_inkc.cuh. Included by inkling.c in a COLI_CUDA
 * build. COLI_CUDA_CHAIN decides: 1 on, 2 prompts only, off by default. Measured on an RTX PRO 4000
 * with Inkling-2-Small int4 g64 (62 GB of RAM, the experts from RAM and disk), 64 tokens on three
 * prompts: 1.24-1.27 tok/s on against 1.26-1.29 off, the same text; the dense part went from 4.7 to
 * 3.4 s a run but the routed experts and the reads took that back (the chain and the expert tier
 * share the card), and the first prompt pays about 4.5 s to move the matrices (below).
 *
 * What differs from the Vulkan chain:
 *   - the dense bf16 matrices: the CUDA build keeps them on its own small backend
 *     (backend_cuda_ink.cu, a GEMV that rounds the activations to bf16). The chain needs the
 *     backend's tensors: at its first forward each matrix is read back once, freed there and
 *     uploaded as BF16 rows (a GGUF's type 30, the same bytes), so the card holds one copy.
 *     The per-matrix path then runs through those tensors (matmul_w's W.cut) where the chain
 *     declines a step.
 *   - prompt chunks of COLI_CUDA_CHAIN_ROWS rows (1024), as OLMoE's CUDA chain.
 *   - a GGUF's matrices stay with the per-matrix path (the chain declines). */

#define INK2CC_HOST 0   /* the host's convolution states are newer than the device's */
#define INK2CC_BOTH 1

typedef struct {
    int ok, failed, tried;
    int rows;                                  /* scratch capacity in rows */
    int max_t; float **hostK;                  /* the host cache the mirrors copy */
    int maxb, hmax, kvo_max, qo_max, ro_max, dyn_max;
    CcBuf *prm;
    size_t *o_n1, *o_n2, *o_n3, *o_post, *o_qc, *o_rc, *o_relp, *o_cw[4], *o_aq, *o_mq, o_outq, o_final;
    ColiCudaTensor *t_lm, **t_q, **t_k, **t_v, **t_r, **t_o, **t_ag, **t_sink, **t_router, **t_wp[2];
    CcBuf **kc, **vc, *rings;                 /* rings: every layer's four convolution states, at o_csd */
    int *cap, *dlo, *dhi;
    size_t vs_off, *o_csd, csd_n;
    int cs_where, host_zero;
    CcBuf *x, *res, *blk, *hq, *hk, *hv, *nq, *nk, *nv, *q, *kvs, *r, *gt, *ctx, *a, *mo, *dyn, *ts, *lg, *fin;
    CcBuf *h2d, *lgd, *kvd, *csd, *outd, *xd, *routed;
    float *host_routed;
    unsigned long long forwards;
    double host_ms;
} Ink2CuChain;

static void ink2cc_fatal(const char *what) {
    fprintf(stderr, "[CUDA] inkling2 chain: %s -- stopping (COLI_VK_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
static int ink2cc_maxb(const Cfg *c) {
    int blk = c->res_block > 0 ? c->res_block : 1;
    return (c->n_layers + blk - 1) / blk;
}
static int ink2cc_geometry_ok(const Model *m, int say) {
    const Cfg *c = &m->c; int dr = c->d_rel, CK = c->conv_k;
    for (int i = 0; i < c->n_layers; i++) {
        const Layer *l = &m->L[i];
        int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i);
        if (!l->x2 || hd > 256 || hd < 1 || KV < 1 || H % KV || dr < 1 || dr > 64 || L_EXT(c, i) < 1 || CK < 1 || CK > 9 ||
            (c->local[i] && c->window < 1) || !c->sparse[i] || c->n_shared > 0 || ink2cc_maxb(c) > 15) {
            if (say)
                fprintf(stderr, "[CUDA] inkling2 chain: a geometry its shaders do not take (layer %d: head dim %d, %d/%d heads, "
                                "bias bank %d x %d, %d taps, %d block states); per-matrix path\n",
                        i, hd, H, KV, dr, L_EXT(c, i), CK, ink2cc_maxb(c));
            return 0;
        }
    }
    return 1;
}

/* The device bytes the chain will take beside the dense matrices (placed already by the
 * per-matrix path): the f32 router, sink and dynamic-tap projections, the parameters, the
 * convolution states, the K/V mirrors at the served context (ink_ctx_max) and the scratch of a
 * prompt chunk of 256 rows. The tier's budget leaves them (ink_vk_dense_bytes). */
static size_t ink2cc_plan_bytes(const Model *m) {
    const Cfg *c = &m->c;
    size_t D = (size_t)c->hidden, E = (size_t)c->n_experts, CK = (size_t)c->conv_k, b = 0, row = 0;
    int t = ink_ctx_max() + 8;
    for (int i = 0; i < c->n_layers; i++) {
        size_t H = (size_t)L_HEADS(c, i), KV = (size_t)L_KV(c, i), hd = (size_t)L_HD(c, i), kvo = KV * hd;
        b += (E + H) * D * 4;                                          /* router, sinks */
        if (!c->local[i]) b += 2 * KV * CK * D * 4;                   /* the dynamic taps' projections */
        b += (6 * D + H * hd + H * c->d_rel + (size_t)c->d_rel * L_EXT(c, i) + 2 * kvo * CK + 2 * D * CK) * 4;
        b += (2 * kvo + 2 * D) * (CK - 1) * 4 * 2;                     /* the states and their read-back */
        b += 2 * kvo * (size_t)kv_ring_rows(c, i, t) * 4;             /* the K/V mirror */
        size_t r1 = (14 * D + 3 * H * hd + 4 * kvo + H * c->d_rel + 2 * E + KV * CK + 2 * H) * 4;
        if (r1 > row) row = r1;
    }
    return b + row * 256 + (size_t)c->unpad_vocab * 4 + ((size_t)ink2cc_maxb(c) * D * 4) * 256;
}

static Ink2CuChain *ink2cc_of(Model *m) { return (Ink2CuChain *)m->ink2cuchain; }
static Ink2CuChain *ink2cc_new(Model *m) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (ch) return ch;
    if (!(ch = calloc(1, sizeof *ch))) return NULL;
    m->ink2cuchain = ch;
    int L = m->c.n_layers;
#define INK2CC_ARR(f, T) (ch->f = calloc((size_t)L, sizeof(T)))
    if (!INK2CC_ARR(o_n1, size_t) || !INK2CC_ARR(o_n2, size_t) || !INK2CC_ARR(o_n3, size_t) || !INK2CC_ARR(o_post, size_t) ||
        !INK2CC_ARR(o_qc, size_t) || !INK2CC_ARR(o_rc, size_t) || !INK2CC_ARR(o_relp, size_t) || !INK2CC_ARR(o_cw[0], size_t) ||
        !INK2CC_ARR(o_cw[1], size_t) || !INK2CC_ARR(o_cw[2], size_t) || !INK2CC_ARR(o_cw[3], size_t) || !INK2CC_ARR(o_aq, size_t) ||
        !INK2CC_ARR(o_mq, size_t) || !INK2CC_ARR(t_q, void *) || !INK2CC_ARR(t_k, void *) || !INK2CC_ARR(t_v, void *) ||
        !INK2CC_ARR(t_r, void *) || !INK2CC_ARR(t_o, void *) || !INK2CC_ARR(t_ag, void *) || !INK2CC_ARR(t_sink, void *) ||
        !INK2CC_ARR(t_router, void *) || !INK2CC_ARR(t_wp[0], void *) || !INK2CC_ARR(t_wp[1], void *) ||
        !INK2CC_ARR(kc, void *) || !INK2CC_ARR(vc, void *) || !INK2CC_ARR(cap, int) || !INK2CC_ARR(dlo, int) ||
        !INK2CC_ARR(dhi, int)) return NULL;
#undef INK2CC_ARR
    if (!(ch->o_csd = calloc((size_t)4 * L, sizeof(size_t)))) return NULL;
    return ch;
}

/* The chain's own device memory freed (the per-matrix path's copies of the dense matrices stay). */
static void ink2cc_free_own(Model *m, Ink2CuChain *ch) {
    if (cc_ready()) cc_finish();
    for (int i = 0; i < m->c.n_layers; i++) {
        ColiCudaTensor **own[] = {&ch->t_sink[i], &ch->t_router[i], &ch->t_wp[0][i], &ch->t_wp[1][i]};
        for (size_t k = 0; k < sizeof own / sizeof own[0]; k++) { if (*own[k]) coli_cuda_tensor_free(*own[k]); *own[k] = NULL; }
        ch->t_q[i] = ch->t_k[i] = ch->t_v[i] = ch->t_r[i] = ch->t_o[i] = ch->t_ag[i] = NULL;   /* the per-matrix path's */
        cc_free(ch->kc[i]); cc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
    }
    ch->t_lm = NULL;
    cc_free(ch->rings); ch->rings = NULL;
    cc_free(ch->prm); ch->prm = NULL;
    ch->max_t = 0; ch->hostK = NULL;
}

/* A dense matrix [O x I] as the backend's tensor, shared with the per-matrix path (W.cut): f32
 * rows as they are; bf16 rows as a GGUF's BF16 (type 30: the same bytes), read back once from the
 * CUDA build's own backend where it holds them, freed there before the upload; a GGUF's blocks as
 * the file has them (ink2_cuda_gguf_dense puts them up at the start). NULL: a form the backend
 * does not take. */
static ColiCudaTensor *ink2cc_mat(Wt *w, int I, int O) {
    if (w->cut) return (ColiCudaTensor *)w->cut;
    ColiCudaTensor *t = NULL;
    int fmt;
    if (w->f) {
        fmt = 0;
        if (!coli_cuda_tensor_upload(&t, w->f, NULL, 0, I, O, g_ink_cuda_dev)) return NULL;
    } else if (w->h || w->dev) {
        fmt = 130;
        uint16_t *tmp = NULL;
        const void *src = w->h;
        const size_t n = (size_t)I * O * 2;
        if (!src) {
            if (!(tmp = malloc(n)) || ink_cuda_download(tmp, w->dev, n)) { free(tmp); return NULL; }
            ink_cuda_free(w->dev);
            w->dev = NULL;
            src = tmp;
        }
        int ok = coli_cuda_tensor_upload(&t, src, NULL, 130, I, O, g_ink_cuda_dev);
        if (!ok && tmp) w->dev = ink_cuda_upload(tmp, n);   /* back where it was */
        free(tmp);
        if (!ok) return NULL;
    } else if (w->gq) {
        /* a GGUF's blocks (cuda_bd.cuh), the rows the loader interleaved by 16 back to the file's
         * first; the head's unpadded vocabulary stops inside the last group, whose rows qn holds.
         * The host copy stays: the CPU's step reads it where the device declines one. */
        const int b = gq_r16_base(w->gqt), ty = b >= 0 ? b : w->gqt, be = coli_cuda_block_elems(100 + ty);
        if (be <= 0 || I % be || I % 32) return NULL;
        fmt = 100 + ty;
        const int Og = (O + 15) / 16 * 16;
        uint8_t *rows = NULL;
        if (b >= 0) {
            if (w->qn < (int64_t)Og * gq_row_bytes(w->gqt, I) || !(rows = malloc((size_t)gq_row_bytes(ty, I) * (size_t)Og)))
                return NULL;
            gq_r16_unpack(w->gqt, rows, w->gq, Og, I);
        }
        int ok = coli_cuda_tensor_upload(&t, rows ? (const void *)rows : (const void *)w->gq, NULL, fmt, I, O, g_ink_cuda_dev);
        free(rows);
        if (!ok) return NULL;
    } else return NULL;
    w->cut = t; w->cutf = fmt;
    return t;
}
/* an f32 matrix [O][I] of the chain's own (the router, the sinks, the dynamic taps' projections) */
static int ink2cc_f32(ColiCudaTensor **t, const float *w, int I, int O) {
    return w && coli_cuda_tensor_upload(t, w, NULL, 0, I, O, g_ink_cuda_dev);
}

/* The model's parameters on the device, its tensors resolved; NULL = the chain cannot run. */
static Ink2CuChain *ink2cc_setup(Model *m) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (ch && ch->tried) return ch->ok ? ch : NULL;
    if (!ch && !(ch = ink2cc_new(m))) return NULL;
    ch->tried = 1;
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, CK = c->conv_k, dr = c->d_rel, E = c->n_experts;
    if (!ink2cc_geometry_ok(m, 1)) return NULL;
    if (!cc_ink_ready() || !cc_muse_ready()) {
        fprintf(stderr, "[CUDA] inkling2 chain: the CUDA backend has no Inkling ops (an older build); per-matrix path\n");
        return NULL;
    }
    const char *what = NULL;
    for (int i = 0; i < L && !what; i++) {
        Layer *l = &m->L[i]; Ink2Layer *x2 = l->x2;
        int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd;
        if (!(ch->t_q[i] = ink2cc_mat(&l->q, D, H * hd)) || !(ch->t_k[i] = ink2cc_mat(&l->k, D, kvo)) ||
            !(ch->t_v[i] = ink2cc_mat(&l->v, D, kvo)) || !(ch->t_r[i] = ink2cc_mat(&l->r, D, H * dr)) ||
            !(ch->t_o[i] = ink2cc_mat(&l->o, H * hd, D)) || !(ch->t_ag[i] = ink2cc_mat(&l->ag, D, H * hd)))
            what = "an attention matrix";
        else if (!ink2cc_f32(&ch->t_sink[i], x2->sink, D, H)) what = "a sink projection";
        else if (!ink2cc_f32(&ch->t_router[i], l->router, D, E)) what = "the router";
        else if (!c->local[i] && (!ink2cc_f32(&ch->t_wp[0][i], x2->k_wp, D, KV * CK) ||
                                  !ink2cc_f32(&ch->t_wp[1][i], x2->v_wp, D, KV * CK)))
            what = "a dynamic convolution's projection";
    }
    if (!what && !(ch->t_lm = ink2cc_mat(&m->lm_head, D, c->unpad_vocab))) what = "lm_head";
    if (what) {
        fprintf(stderr, "[CUDA] inkling2 chain: %s did not reach the device; per-matrix path\n", what);
        ink2cc_free_own(m, ch);
        return NULL;
    }
    /* the parameter arena: offsets, then one upload */
    size_t n = 0;
    ch->kvo_max = ch->qo_max = ch->ro_max = ch->dyn_max = ch->hmax = 0;
    for (int i = 0; i < L; i++) {
        Ink2Layer *x2 = m->L[i].x2;
        int H = L_HEADS(c, i), hd = L_HD(c, i), kvo = L_KV(c, i) * hd;
        if (kvo > ch->kvo_max) ch->kvo_max = kvo;
        if (H * hd > ch->qo_max) ch->qo_max = H * hd;
        if (H * dr > ch->ro_max) ch->ro_max = H * dr;
        if (H > ch->hmax) ch->hmax = H;
        if (!c->local[i] && L_KV(c, i) * CK > ch->dyn_max) ch->dyn_max = L_KV(c, i) * CK;
        ch->o_n1[i] = n; n += D;
        if (!c->local[i]) { ch->o_n2[i] = n; n += D; ch->o_n3[i] = n; n += D; }
        ch->o_post[i] = n; n += D;
        ch->o_qc[i] = n; n += (size_t)H * hd;
        ch->o_rc[i] = n; n += (size_t)H * dr;
        ch->o_relp[i] = n; n += (size_t)dr * L_EXT(c, i);
        ch->o_cw[0][i] = n; n += (size_t)kvo * CK; ch->o_cw[1][i] = n; n += (size_t)kvo * CK;
        ch->o_cw[2][i] = n; n += (size_t)D * CK; ch->o_cw[3][i] = n; n += (size_t)D * CK;
        ch->o_aq[i] = n; n += (size_t)x2->naq * D;
        ch->o_mq[i] = n; n += D;
    }
    ch->o_outq = n; n += D;
    ch->o_final = n; n += D;
    float *arena = calloc(n, sizeof(float));
    if (!arena) { ink2cc_free_own(m, ch); return NULL; }
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i]; Ink2Layer *x2 = l->x2;
        int H = L_HEADS(c, i), hd = L_HD(c, i), kvo = L_KV(c, i) * hd;
        if (c->local[i]) memcpy(arena + ch->o_n1[i], x2->in_ln, D * sizeof(float));
        else {
            memcpy(arena + ch->o_n1[i], x2->qin_ln, D * sizeof(float));
            memcpy(arena + ch->o_n2[i], x2->kin_ln, D * sizeof(float));
            memcpy(arena + ch->o_n3[i], x2->vin_ln, D * sizeof(float));
        }
        memcpy(arena + ch->o_post[i], x2->post_ln, D * sizeof(float));
        memcpy(arena + ch->o_qc[i], x2->q_clip, (size_t)H * hd * sizeof(float));
        memcpy(arena + ch->o_rc[i], x2->r_clip, (size_t)H * dr * sizeof(float));
        memcpy(arena + ch->o_relp[i], l->relp, (size_t)dr * L_EXT(c, i) * sizeof(float));
        /* a sliding layer's K/V taps [C][K] as sconv reads them; a global one's static taps [K][C] */
        memcpy(arena + ch->o_cw[0][i], c->local[i] ? l->k_cw : x2->k_st, (size_t)kvo * CK * sizeof(float));
        memcpy(arena + ch->o_cw[1][i], c->local[i] ? l->v_cw : x2->v_st, (size_t)kvo * CK * sizeof(float));
        memcpy(arena + ch->o_cw[2][i], l->a_cw, (size_t)D * CK * sizeof(float));
        memcpy(arena + ch->o_cw[3][i], l->m_cw, (size_t)D * CK * sizeof(float));
        if (x2->naq) memcpy(arena + ch->o_aq[i], x2->aq, (size_t)x2->naq * D * sizeof(float));
        memcpy(arena + ch->o_mq[i], x2->mq, D * sizeof(float));
    }
    memcpy(arena + ch->o_outq, m->x2_out_query, D * sizeof(float));
    memcpy(arena + ch->o_final, m->final_norm, D * sizeof(float));
    ch->prm = cc_buf(n * sizeof(float), CC_DEV);
    int ok = ch->prm && cc_begin() && cc_write(ch->prm, 0, arena, n * sizeof(float)) && cc_submit(1);
    free(arena);
    if (!ok) { ink2cc_free_own(m, ch); return NULL; }
    ch->maxb = ink2cc_maxb(c);
    ch->csd_n = 0;   /* the states one after another, the same layout on the device and in the read-back */
    for (int i = 0; i < L; i++)
        for (int b = 0; b < 4; b++) { ch->o_csd[(size_t)b * L + i] = ch->csd_n; ch->csd_n += (size_t)ink_cs_cells(c, b, i); }
    if (!(ch->rings = cc_buf((ch->csd_n ? ch->csd_n : 1) * sizeof(float), CC_DEV))) { ink2cc_free_own(m, ch); return NULL; }
    ch->cs_where = INK2CC_HOST; ch->host_zero = 0;
    ch->ok = 1;
    fprintf(stderr, "[CUDA] inkling2 chain: %d layers on the device, %.1f MiB of parameters\n", L, n * 4 / 1048576.0);
    return ch;
}

/* the scratch buffers for `rows` rows */
static int ink2cc_bufs(Ink2CuChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; size_t D = (size_t)c->hidden, E = (size_t)c->n_experts, r = (size_t)rows;
    ch->vs_off = (r * ch->kvo_max + 63) & ~(size_t)63;
    size_t dyn = ch->dyn_max > 0 ? (size_t)ch->dyn_max : 1;
    return cc_reserve(&ch->x, r * D * 4, CC_DEV) && cc_reserve(&ch->res, r * D * 4, CC_DEV) &&
           cc_reserve(&ch->blk, r * ch->maxb * D * 4, CC_DEV) && cc_reserve(&ch->hq, r * D * 4, CC_DEV) &&
           cc_reserve(&ch->hk, r * D * 4, CC_DEV) && cc_reserve(&ch->hv, r * D * 4, CC_DEV) &&
           cc_reserve(&ch->nq, r * D * 4, CC_DEV) && cc_reserve(&ch->nk, r * D * 4, CC_DEV) &&
           cc_reserve(&ch->nv, r * D * 4, CC_DEV) && cc_reserve(&ch->q, r * ch->qo_max * 4, CC_DEV) &&
           cc_reserve(&ch->kvs, (ch->vs_off + r * ch->kvo_max) * 4, CC_DEV) && cc_reserve(&ch->r, r * ch->ro_max * 4, CC_DEV) &&
           cc_reserve(&ch->gt, r * ch->qo_max * 4, CC_DEV) && cc_reserve(&ch->ctx, r * ch->qo_max * 4, CC_DEV) &&
           cc_reserve(&ch->a, r * D * 4, CC_DEV) && cc_reserve(&ch->mo, r * D * 4, CC_DEV) &&
           cc_reserve(&ch->dyn, r * dyn * 4, CC_DEV) && cc_reserve(&ch->ts, (2 * r + r * ch->hmax) * 4, CC_DEV) &&
           cc_reserve(&ch->lg, r * E * 4, CC_DEV) && cc_reserve(&ch->fin, D * 4, CC_DEV) &&
           cc_reserve(&ch->h2d, r * D * 4, CC_DOWN) && cc_reserve(&ch->lgd, r * E * 4, CC_DOWN) &&
           cc_reserve(&ch->kvd, 2 * r * ch->kvo_max * 4, CC_DOWN) && cc_reserve(&ch->csd, (ch->csd_n ? ch->csd_n : 1) * 4, CC_DOWN) &&
           cc_reserve(&ch->outd, (size_t)c->unpad_vocab * 4, CC_DOWN) && cc_reserve(&ch->routed, r * D * 4, CC_UP);
}
/* Prompt rows per chunk: COLI_CUDA_CHAIN_ROWS (1 to 8192), else 1024, as OLMoE's CUDA chain. */
static int ink2cc_chunk_rows(Ink2CuChain *ch, Model *m) {
    (void)ch; (void)m;
    const char *e = getenv("COLI_CUDA_CHAIN_ROWS");
    int r = e && *e ? atoi(e) : 1024;
    return r < 1 ? 1 : r > 8192 ? 8192 : r;
}
/* the K/V mirrors at the host's layout, made again when the host's cache was */
static int ink2cc_mirror(Ink2CuChain *ch, Model *m) {
    Cfg *c = &m->c;
    if (ch->max_t == m->max_t && ch->hostK == m->K) return 1;
    for (int i = 0; i < c->n_layers; i++) {
        cc_free(ch->kc[i]); cc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
        int cap = kv_ring_rows(c, i, m->max_t), hd = L_HD(c, i), KV = L_KV(c, i);
        ch->cap[i] = cap;
        ch->dlo[i] = 0; ch->dhi[i] = cap < m->max_t ? m->max_t : m->kv_len;   /* a ring: every row; else the positions held */
        ch->kc[i] = cc_buf((size_t)KV * cap * hd * sizeof(float), CC_DEV);
        ch->vc[i] = cc_buf((size_t)KV * cap * hd * sizeof(float), CC_DEV);
        if (!ch->kc[i] || !ch->vc[i]) { ch->max_t = 0; ch->hostK = NULL; return 0; }
    }
    ch->max_t = m->max_t; ch->hostK = m->K;
    return 1;
}
static int ink2cc_scratch(Ink2CuChain *ch, Model *m, int rows) {
    if (rows < ch->rows) rows = ch->rows;   /* the layout of the widest step so far: ch->rows is its stride */
    if (!ink2cc_bufs(ch, m, rows)) return 0;
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, (size_t)rows * m->c.hidden * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    return 1;
}

/* ---- state between the host and the device (inkling_chain.h's, for Inkling-2) ---- */
static void ink2cc_host_wrote(Model *m, int zero) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (!ch || !ch->ok) return;
    ch->cs_where = INK2CC_HOST; ch->host_zero = zero;
}
static void ink2cc_cpu_step(Model *m, int pos_base, int S) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (!ch || !ch->ok) return;
    ch->cs_where = INK2CC_HOST; ch->host_zero = 0;
    for (int i = 0; i < m->c.n_layers; i++) {
        if (ch->dlo[i] >= ch->dhi[i]) { ch->dlo[i] = pos_base; ch->dhi[i] = pos_base + S; continue; }
        if (pos_base < ch->dlo[i]) ch->dlo[i] = pos_base;
        if (pos_base + S > ch->dhi[i]) ch->dhi[i] = pos_base + S;
    }
}
static int ink2cc_push_state(Ink2CuChain *ch, Model *m) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->cs_where == INK2CC_HOST) {
        if (ch->host_zero) ok = !ch->csd_n || cc_zero(ch->rings, 0, ch->csd_n);
        else for (int i = 0; i < c->n_layers && ok; i++) for (int b = 0; b < 4 && ok; b++) {
            size_t nc = (size_t)ink_cs_cells(c, b, i);
            if (nc) ok = cc_write(ch->rings, ch->o_csd[(size_t)b * c->n_layers + i], m->cs[b][i], nc * sizeof(float));
        }
        ch->cs_where = INK2CC_BOTH;
    }
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (ch->dlo[i] >= ch->dhi[i]) continue;
        int cap = ch->cap[i], hd = L_HD(c, i), KV = L_KV(c, i);
        int t0 = ch->dhi[i] - cap > ch->dlo[i] ? ch->dhi[i] - cap : ch->dlo[i];
        for (int t = t0; t < ch->dhi[i] && ok; ) {        /* a run of rows up to the ring's end */
            int row = t % cap, run = ch->dhi[i] - t < cap - row ? ch->dhi[i] - t : cap - row;
            for (int h = 0; h < KV && ok; h++) {
                size_t off = ((size_t)h * cap + row) * hd;
                ok = cc_write(ch->kc[i], off, m->K[i] + off, (size_t)run * hd * sizeof(float)) &&
                     cc_write(ch->vc[i], off, m->V[i] + off, (size_t)run * hd * sizeof(float));
            }
            t += run;
        }
        ch->dlo[i] = ch->dhi[i] = 0;
    }
    return ok;
}
/* The device was lost with the newest state on it: rebuild the K/V cache and the convolution
 * states on the CPU from the `upto` positions the prefix record names, and leave the chain off. */
static void ink2cc_recover(Model *m, int upto) {
    Ink2CuChain *ch = ink2cc_of(m);
    Cfg *c = &m->c; int D = c->hidden;
    g_cu_chain = 0;
    if (ch) ch->failed = 1;
    for (int i = 0; i < c->n_layers; i++)
        for (int b = 0; b < 4; b++) memset(m->cs[b][i], 0, (size_t)ink_cs_cells(c, b, i) * sizeof(float));
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        ink2cc_fatal("the device was lost with a state its token ids do not describe (audio or an image)");
    fprintf(stderr, "[CUDA] inkling2 chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", upto);
    float *x = falloc((int64_t)upto * D);
    for (int s = 0; s < upto; s++) {
        wt_row_f32(m->embed, (int64_t)m->kvp.fed[s] * D, x + (int64_t)s * D, D);
        if (m->embed_norm) rmsnorm_row(x + (int64_t)s * D, x + (int64_t)s * D, m->embed_norm, D, c->eps);
    }
    ink2_layers_forward(m, x, upto, 0);
    free(x);
}

/* ---- one layer's pieces ---------------------------------------------------------- */
static int ink2cc_norm(CcBuf *x, CcBuf *w, size_t wo, CcBuf *y, int rows, int D, float eps) {
    CcNorm p = {rows, D, 1, 0, D, D, 0, D, D, (int)wo, 0, 0, eps, 1.f};
    return cc_norm(x, w, y, &p);
}
/* a static short convolution through bank `bank` and its scale: (conv + input) / sqrt(K + 1) */
static int ink2cc_sconv(Ink2CuChain *ch, CcBuf *x, size_t xo, int n, int C, int bank, int i, const Cfg *c) {
    CcSconv p = {0, n, C, c->conv_k, (int)xo, C, (int)ch->o_cw[bank][i], (int)ch->o_csd[(size_t)bank * c->n_layers + i], 0, 1.f};
    CcSconv dv = {2, 0, 0, 0, (int)xo, 0, 0, 0, n * C, (float)sqrt((double)(c->conv_k + 1))};
    return cc_sconv(x, ch->prm, ch->rings, &p) && cc_sconv(x, NULL, NULL, &dv);
}
/* a global layer's dynamic convolution of K (bank 0, from nk) or V (bank 1, from nv) in kvs */
static int ink2cc_dconv(Ink2CuChain *ch, Model *m, int i, int bank, CcBuf *xin, int n) {
    Cfg *c = &m->c; int KV = L_KV(c, i), hd = L_HD(c, i), K = c->conv_k;
    CcInk2Dconv p = {n, KV * hd, K, hd, bank ? (int)ch->vs_off : 0, KV * hd, 0, KV * K, (int)ch->o_cw[bank][i],
                      (int)ch->o_csd[(size_t)bank * c->n_layers + i], (float)sqrt((double)(2 * K + 1))};
    return cc_matmul(ch->t_wp[bank][i], xin, 0, ch->dyn, 0, n) && cc_ink2_dconv(ch->kvs, ch->dyn, ch->prm, ch->rings, &p);
}
/* the attention from nq (and nk, nv on a global layer): o_proj's rows into a, the step's K/V
 * rows into the device's cache and into kvd */
static int ink2cc_attention(Ink2CuChain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, dr = c->d_rel;
    int cap = ch->cap[i], local = c->local[i], R = ch->rows;
    size_t vs = ch->vs_off;
    CcBuf *xk = local ? ch->nq : ch->nk, *xv = local ? ch->nq : ch->nv;
    int ok = cc_matmul(ch->t_q[i], ch->nq, 0, ch->q, 0, n) && cc_matmul(ch->t_k[i], xk, 0, ch->kvs, 0, n) &&
             cc_matmul(ch->t_v[i], xv, 0, ch->kvs, vs, n) && cc_matmul(ch->t_r[i], ch->nq, 0, ch->r, 0, n) &&
             cc_matmul(ch->t_ag[i], ch->nq, 0, ch->gt, 0, n) && cc_matmul(ch->t_sink[i], ch->nq, 0, ch->ts, (size_t)2 * R, n);
    if (local) ok = ok && ink2cc_sconv(ch, ch->kvs, 0, n, kvo, 0, i, c) && ink2cc_sconv(ch, ch->kvs, vs, n, kvo, 1, i, c);
    else ok = ok && ink2cc_dconv(ch, m, i, 0, ch->nk, n) && ink2cc_dconv(ch, m, i, 1, ch->nv, n);
    CcInk2Clip qc = {n * H, hd, H, 0, H * hd, 1, (int)ch->o_qc[i], c->qk_clip};
    CcInk2Clip rc = {n * H, dr, H, 0, H * dr, 1, (int)ch->o_rc[i], c->rel_clip};
    CcInk2Clip kc = {n * KV, hd, KV, 0, kvo, 0, 0, c->qk_clip};
    CcNorm vn = {n * KV, hd, KV, (int)vs, kvo, hd, (int)vs, kvo, hd, 0, 0, 2, c->eps, 1.f};
    ok = ok && cc_ink2_clip(ch->q, ch->prm, &qc) && cc_ink2_clip(ch->r, ch->prm, &rc) &&
         cc_ink2_clip(ch->kvs, NULL, &kc) && cc_norm(ch->kvs, ch->prm, ch->kvs, &vn);
    CcRelAttn a = {n, H, KV, hd, pb, cap, local ? c->window : 0, L_EXT(c, i), dr,
                    0, H * hd, 0, H * hd, 0, 0, 0, (int)vs, kvo, 0, H * dr, (int)ch->o_relp[i], local ? R : 0,
                    (float)(1.0 / sqrt((double)hd)), 1, 2 * R};
    CcEw gate = {CC_EW_SIGMUL, n * H * hd, H * hd, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    ok = ok && cc_relattn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->kvs, ch->r, ch->prm, ch->ts, &a) &&
         cc_ew(ch->ctx, ch->ctx, ch->gt, NULL, NULL, &gate);
    if (!ok) return 0;
    /* the step's rows into the cache after the attention read them: the last `cap` of them */
    int s0 = n > cap ? n - cap : 0, nr = (n - s0) * KV;
    CcRegion *rk = malloc(sizeof *rk * (size_t)nr * 2), *rv = rk ? rk + nr : NULL;
    if (!rk) return 0;
    for (int s = s0; s < n; s++) for (int h = 0; h < KV; h++) {
        size_t dst = ((size_t)h * cap + (size_t)(pb + s) % cap) * hd, src = (size_t)s * kvo + (size_t)h * hd;
        rk[(s - s0) * KV + h] = (CcRegion){dst, src, (size_t)hd};
        rv[(s - s0) * KV + h] = (CcRegion){dst, vs + src, (size_t)hd};
    }
    ok = cc_copy_regions(ch->kc[i], ch->kvs, rk, nr) && cc_copy_regions(ch->vc[i], ch->kvs, rv, nr) &&
         cc_copy(ch->kvd, 0, ch->kvs, 0, (size_t)n * kvo) && cc_copy(ch->kvd, (size_t)R * ch->kvo_max, ch->kvs, vs, (size_t)n * kvo);
    free(rk);
    return ok && cc_matmul(ch->t_o[i], ch->ctx, 0, ch->a, 0, n);
}
/* layer i's new K/V rows (in kvd) into the host's cache, as ink2_attention writes them */
static void ink2cc_kv_down(Ink2CuChain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c;
    int KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, cap = ch->cap[i], s0 = n > cap ? n - cap : 0;
    const float *kv = (const float *)cc_ptr(ch->kvd), *vv = kv + (size_t)ch->rows * ch->kvo_max;
    for (int s = s0; s < n; s++) for (int h = 0; h < KV; h++) {
        size_t dst = ((size_t)h * cap + (size_t)(pb + s) % cap) * hd, src = (size_t)s * kvo + (size_t)h * hd;
        memcpy(m->K[i] + dst, kv + src, hd * sizeof(float));
        memcpy(m->V[i] + dst, vv + src, hd * sizeof(float));
    }
}
/* the MoE output of layer i (the host's routed sum) joining the stream: x = res + sconv(mo) */
static int ink2cc_join(Ink2CuChain *ch, Model *m, int i, int n) {
    Cfg *c = &m->c; int D = c->hidden;
    CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_copy(ch->mo, 0, ch->routed, 0, (size_t)n * D) && ink2cc_sconv(ch, ch->mo, 0, n, D, 3, i, c) &&
           cc_ew(ch->x, ch->res, ch->mo, NULL, NULL, &add);
}
/* the residual mix of the nb block states and `part` into y with query q (at qo in prm) */
static int ink2cc_mix(Ink2CuChain *ch, CcBuf *part, int nb, size_t qo, CcBuf *y, int n, int D) {
    CcInk2Mix p = {n, D, nb, 0, D, 0, ch->maxb * D, (int)qo, 0, D, 1e-6f, (float)(1.0 / sqrt((double)D))};
    return cc_ink2_mix(part, ch->blk, ch->prm, y, &p);
}

/* Every layer for S rows from host rows xh, the last row's logits (softcapped) into `logit`;
 * xh gets the rows the final norm reads (the output mix applied) when want_x. 0 = not taken:
 * xh is as it was and the caller runs the step on the CPU (after ink2cc_recover when the device
 * was lost). 1 = taken. */
static int ink2cc_forward(Model *m, float *xh, int S, int pos_base, int want_x, float *logit) {
    if (!g_cu_chain) return 0;
    if (g_cu_chain == 2 && S <= 2) return 0;   /* prompts only: decode on the CPU */
    if (m->mux_rows) return 0;
    Ink2CuChain *ch = ink2cc_setup(m);
    if (!ch || ch->failed) return 0;
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, E = c->n_experts, blkn = c->res_block > 0 ? c->res_block : 1;
    int mirror_ok = ink2cc_mirror(ch, m);
    int CH = mirror_ok ? ink2cc_chunk_rows(ch, m) : 1, rows = S < CH ? S : CH;
    if (!mirror_ok || pos_base + S > m->max_t || !ink2cc_scratch(ch, m, rows) ||
        (want_x && !cc_reserve(&ch->xd, (size_t)rows * D * sizeof(float), CC_DOWN))) {
        fprintf(stderr, "[CUDA] inkling2 chain: device memory for %d rows refused; per-matrix path\n", rows);
        ink2cc_free_own(m, ch);
        ch->failed = 1;
        return 0;
    }
    float *xfin = want_x ? malloc((size_t)S * D * sizeof(float)) : NULL;
    if (want_x && !xfin) return 0;
    for (int c0 = 0; c0 < S; c0 += rows) {
        if (coli_prompt_break()) { free(xfin); return 0; }
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0, R = ch->rows;
        if (!cc_begin() || !cc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !ink2cc_push_state(ch, m)) goto lost;
        {   /* attention's tau: the global layers' log-length scale, then 1 for the sliding ones */
            float tau[2 * 512];
            float *tp = n <= 512 ? tau : malloc((size_t)2 * R * sizeof(float));
            if (!tp) goto lost;
            for (int s = 0; s < n; s++) {
                float t = 1.f;
                if (c->log_floor > 0) { double en = (double)(pb + s + 1) / c->log_floor; if (en > 1.0) t = 1.f + c->log_alpha * (float)log(en); }
                tp[s] = t;
            }
            for (int s = 0; s < n; s++) tp[n + s] = 1.f;
            int wok = cc_write(ch->ts, 0, tp, (size_t)n * sizeof(float)) && cc_write(ch->ts, (size_t)R, tp + n, (size_t)n * sizeof(float));
            if (tp != tau) free(tp);
            if (!wok) goto lost;
        }
        int ok = 1, pending = 0, B = 0;
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i]; Ink2Layer *x2 = l->x2;
            int local = c->local[i];
            if (pending) ok = ink2cc_join(ch, m, i - 1, n);
            /* the attention's input: the mix of the block states so far and the stream (layer 0: the stream) */
            CcBuf *iq = ch->x, *ik = ch->x, *iv = ch->x;
            if (ok && x2->aq) {
                ok = ink2cc_mix(ch, ch->x, B, ch->o_aq[i], ch->hq, n, D);
                if (ok && !local) ok = ink2cc_mix(ch, ch->x, B, ch->o_aq[i] + D, ch->hk, n, D) &&
                                       ink2cc_mix(ch, ch->x, B, ch->o_aq[i] + 2 * (size_t)D, ch->hv, n, D);
                iq = ch->hq; ik = local ? ch->hq : ch->hk; iv = local ? ch->hq : ch->hv;
            }
            ok = ok && ink2cc_norm(iq, ch->prm, ch->o_n1[i], ch->nq, n, D, c->eps);
            if (!local) ok = ok && ink2cc_norm(ik, ch->prm, ch->o_n2[i], ch->nk, n, D, c->eps) &&
                                   ink2cc_norm(iv, ch->prm, ch->o_n3[i], ch->nv, n, D, c->eps);
            ok = ok && ink2cc_attention(ch, m, i, n, pb) && ink2cc_sconv(ch, ch->a, 0, n, D, 2, i, c);
            /* a block boundary keeps this layer's input stream as a block state and starts the
             * stream over from the attention's output; elsewhere the output adds to it */
            if (ok && i % blkn == 0) {
                CcRegion *rg = malloc(sizeof *rg * (size_t)n);
                if (!rg) { ok = 0; break; }
                for (int s = 0; s < n; s++) rg[s] = (CcRegion){((size_t)s * ch->maxb + B) * D, (size_t)s * D, (size_t)D};
                ok = cc_copy_regions(ch->blk, ch->x, rg, n) && cc_copy(ch->res, 0, ch->a, 0, (size_t)n * D);
                free(rg);
                B++;
            } else if (ok) {
                CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = cc_ew(ch->res, ch->x, ch->a, NULL, NULL, &add);
            }
            /* the MLP's input mix, its norm, the router */
            ok = ok && ink2cc_mix(ch, ch->res, B, ch->o_mq[i], ch->hq, n, D) &&
                 ink2cc_norm(ch->hq, ch->prm, ch->o_post[i], ch->nq, n, D, c->eps) &&
                 cc_matmul(ch->t_router[i], ch->nq, 0, ch->lg, 0, n) &&
                 cc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && cc_copy(ch->h2d, 0, ch->nq, 0, (size_t)n * D);
            double t0 = now_s();
            ok = ok && cc_submit(0);
            if (ok) cut_stream_prefetch(i, n);
            ok = ok && cc_finish();
            m->t_attn += now_s() - t0;
            if (!ok) break;
            ink2cc_kv_down(ch, m, i, n, pb);
            double t1 = now_s();
            moe_ex(m, l, i, (float *)cc_ptr(ch->h2d), n, ch->host_routed, (const float *)cc_ptr(ch->lgd), NULL);
            memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = cc_begin();
            pending = 1;
        }
        if (ok && pending) ok = ink2cc_join(ch, m, L - 1, n);
        /* the output's mix of the block states and the last stream: the rows the final norm reads */
        ok = ok && ink2cc_mix(ch, ch->x, B, ch->o_outq, ch->hq, n, D);
        if (ok && want_x) ok = cc_copy(ch->xd, 0, ch->hq, 0, (size_t)n * D);
        if (ok && ch->csd_n) ok = cc_copy(ch->csd, 0, ch->rings, 0, ch->csd_n);   /* the convolution states, for the host */
        int last = c0 + n == S;
        if (ok && last) {
            CcSconv dv = {2, 0, 0, 0, 0, 0, 0, 0, D, c->mup};
            CcNorm fp = {1, D, 1, (n - 1) * D, D, D, 0, D, D, (int)ch->o_final, 0, 0, c->eps, 1.f};
            ok = cc_norm(ch->hq, ch->prm, ch->fin, &fp) && cc_sconv(ch->fin, NULL, NULL, &dv) &&
                 cc_matmul(ch->t_lm, ch->fin, 0, ch->outd, 0, 1);
        }
        double t0 = now_s();
        ok = ok && cc_submit(1);
        m->t_attn += now_s() - t0;
        if (!ok) goto lost;
        const float *csd = (const float *)cc_ptr(ch->csd);
        for (int i = 0; i < L; i++) for (int b = 0; b < 4; b++)
            memcpy(m->cs[b][i], csd + ch->o_csd[(size_t)b * L + i], (size_t)ink_cs_cells(c, b, i) * sizeof(float));
        ch->cs_where = INK2CC_BOTH; ch->host_zero = 0;
        if (want_x) memcpy(xfin + (size_t)c0 * D, cc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        if (last) {
            memcpy(logit, cc_ptr(ch->outd), (size_t)c->unpad_vocab * sizeof(float));
            ink2_head_post(c, logit, c->unpad_vocab);
        }
    }
    if (want_x) { memcpy(xh, xfin, (size_t)S * D * sizeof(float)); free(xfin); }
    ch->forwards++;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!cc_lost()) cc_finish();
    free(xfin);
    ink2cc_recover(m, pos_base);
    return 0;
}

static void ink2cc_report(Model *m) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (!ch || !ch->ok || !ch->forwards) return;
    CcStats st; cc_stats(&st);
    fprintf(stderr, "[CUDA] inkling2 chain: %llu forwards, %llu frames (%llu ops, %llu matmuls), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
}

/* COLI_CUDA_CHAIN for Inkling-2, after the CUDA tier (ink_cu_tier_start): 1 on, 2 prompts only
 * (g_ink2cc_want: asked for, and the tier left its memory). */
static void ink2cc_start(Model *m) {
    g_cu_chain = 0;
    if (!m->c.v2 || !g_cuda || !g_ink2cc_want) return;
    const char *e = getenv("COLI_CUDA_CHAIN");
    if (g_ink_mux_slots > 1) {
        fprintf(stderr, "[CUDA] inkling2: KV_SLOTS=%d: the dense chain is off (it keeps one conversation's state on the device)\n",
                g_ink_mux_slots);
        return;
    }
    if (!ink2cc_geometry_ok(m, 1)) return;
    if (!cc_available() || !cc_ink_ready() || !cc_muse_ready()) {
        fprintf(stderr, "[CUDA] inkling2: the CUDA backend has no chain with Inkling's ops (an older build); the dense matrices "
                        "a call at a time\n");
        return;
    }
    int dev = g_ink_cuda_dev;
    if (!coli_cuda_init(&dev, 1) || !cc_init(dev) || !cc_ready()) {
        fprintf(stderr, "[CUDA] inkling2: the chain's context did not come up; the dense matrices a call at a time\n");
        return;
    }
    g_cu_chain = e && *e == '2' ? 2 : 1;
    atexit(cc_shutdown);
    fprintf(stderr, "[CUDA] inkling2: dense chain on%s (COLI_CUDA_CHAIN)\n", g_cu_chain == 2 ? " for prompts" : "");
}
