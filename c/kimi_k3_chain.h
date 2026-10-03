/* kimi_k3_chain.h -- Kimi K3's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by kimi_k3.c in a COLI_VULKAN build, before step_chunk_ex, which calls
 * k3c_forward in place of the CPU's layers; COLI_VK_CHAIN decides (coli_vk_chain_decide,
 * this engine's integrated-GPU default COLI_VK_CHAIN_UNMEASURED: off, not measured on a
 * Kimi K3 checkpoint).
 *
 * The residual is AttnRes: per row a running prefix and the block snapshots (one every
 * attn_res_block_size layers), mixed by a softmax before the attention and before the
 * MLP. What runs where, per layer, for one block of rows (decode: one row):
 *   device, frame A1: the previous layer's MoE output joins the prefix (the routed sum
 *     from the host: its RMSNorm and the latent up-projection, plus the shared experts,
 *     then the add: the CPU's order); the attention site's residual mix, a block
 *     boundary's snapshot, the input RMSNorm; then the KDA layer (q, k, v, the output
 *     gate, the decay's two f32 matrices and beta's; the short convolution with its
 *     window, the delta rule with its state, the output norm and gate, o_proj) or the
 *     gated MLA layer (vkc_mla_qkv: q_a, its norm, q_b, kv_a, the latent norm, the new
 *     rows into the cache; the gate; vkc_mla_attn: the absorbed core over the cache,
 *     the values times the gate, o_proj); the prefix update; the MLP site's residual
 *     mix and post-attention RMSNorm; then a dense layer's MLP (SiTU-GLU) and its add,
 *     with no host step, or the router's logits (the f32 router) and the latent
 *     down-projection. Then the frame is waited for.
 *   host: the router's sigmoid and top-k with the correction bias, K3_TOPP; the routed
 *     experts in the latent (the expert tier's batch and the CPU's share, joined in the
 *     union's order: k3_moe_experts); the new MLA rows into the host's cache.
 *   device, frame A2 (not waited for): the shared experts (SiTU-GLU at full width).
 *   after the last layer: the output residual mix and the final RMSNorm of every row,
 *     lm_head on the last one. The normalized rows come back when the host wants the
 *     logits of every row (K3_LOGITS, K3_VAL_LOGITS, a logprobs request), and the host
 *     runs lm_head on those as before.
 * Crossing per sparse layer: the router logits (E floats a row) and the latent (latent
 * floats a row) down, the new MLA rows down, the routed sum (latent floats a row) up.
 *
 * Kimi K3's MLA is NoPE: the query's and the shared key's qk_rope parts are used as
 * they come out of the projections. vkc_mla_qkv rotates them with a table of cos 1 and
 * sin 0, which is the identity in float (a*1 - b*0 = a).
 *
 * State and who owns it (one Model, the one k3c_start set up):
 *   - the MLA caches (Lc latent, Rc rope part): the host's stay canonical (every step
 *     copies its new rows back); the device mirrors them behind a watermark per layer
 *     that a CPU forward, a reset and a grown cache (kv_alloc) lower;
 *   - the KDA state and the three convolution windows of every KDA layer: on the
 *     device while the chain runs (too large to copy per token: about 6.9 MB a layer on
 *     the full model). `where` says which side holds the newest copy; the host's is
 *     brought back before anything reads it there (a checkpoint photo, a CPU forward)
 *     and pushed up after anything writes it there (a reset: a fill with zeros on the
 *     device; a restored photo: an upload).
 * Prompt-cache and prefix reuse need nothing more: a reused prefix is rows below the
 * watermark and a KDA state that already sits where the next step expects it.
 *
 * A device lost while it holds the newest KDA state: the state is rebuilt on the CPU
 * from the host's copy (current at host_pos) and the prefix record's ids up to where
 * the device was (a prefill's worth of CPU work), the forward runs again on the CPU
 * from its input rows (the chain never writes them), and the CPU runs from there.
 * COLI_VK_CHAIN_FAULT=n fakes the loss at the n-th frame.
 *
 * The chain declines (the CPU path runs, the state synced first): under the CUDA expert
 * tier, with KIMI_DSA_INDEXER=1 (its index cache is filled on the CPU), for the
 * validation dumps that read every layer on the host (K3_TRACE, K3_VALIDATE_LAYER,
 * K3_DEBUG_OUT), without a head (K3_LAYERS), and for a geometry past the ops' limits
 * (a KDA head above 128 floats, kv_lora above 1024, qk_rope above 128 or odd). */
#include "vk_chain.h"

#define K3C_HOST 0
#define K3C_DEV  1
#define K3C_BOTH 2

typedef struct { ColiVkTensor *q, *k, *v, *g, *o, *fa, *fb, *bp; } K3cKda;
typedef struct { ColiVkTensor *router, *down, *up, *sg, *su, *sd, *dg, *du, *dd; } K3cFfn;

typedef struct {
    const Model *m;
    int ok, failed, rows, cap, nbmax;
    VkcBuf *prm;                              /* every norm, AttnRes weight, conv tap and KDA parameter */
    size_t *o_in, *o_post, *o_asw, *o_msw, *o_qn, *o_kn, *o_conv, *o_kda, *o_latn, o_osw, o_final;
    K3cKda *kda; K3cFfn *ffn; VkcMla *mla; ColiVkTensor **mg, *head;
    VkcMlaCache *kv; VkcBuf **win, **st;
    int *kv_valid, *mla_ord, n_mla;
    int where, host_zero, dev_pos, host_pos;  /* the KDA state: who holds the newest, where each side is */
    VkcMlaScratch sc;
    VkcBuf *x, *bres, *hm, *nrm, *att, *qkv3, *cm, *t1, *kf, *kb, *kg, *ky, *gate, *lg, *z;
    VkcBuf *gs, *us, *hs, *ds, *un, *lu, *fin, *cs;
    VkcBuf *lgd, *zd, *kvd, *find, *outd, *routed;
    size_t kvd_layer;
    float *host_u, *host_sco;
    unsigned long long forwards;
    double host_ms;
} K3Chain;

static K3Chain *g_k3c;
static int g_k3c_on;       /* COLI_VK_CHAIN as decided (on, or prompts only), the chain set up */
static int g_k3c_inited;   /* vkc_init ran: vkc_shutdown goes at exit */

static int k3c_chunk_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}
static int k3c_res(VkcBuf **b, size_t floats, int kind) { return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }

/* A W's device copy into *t: the per-matrix path's own where it made one (the shared
 * experts under COLI_VK_DENSE), else one of the chain's, which w_matmul never sees (so
 * a forward the chain declines stays on the CPU, as before). */
static ColiVkTensor *k3c_w(W *w, ColiVkTensor **t) {
    if (!*t && w->vk) *t = (ColiVkTensor *)w->vk;
    if (*t) return *t;
    int fmt = k3_vk_fmt(w);
    if (fmt < 0 || w->mapped) return NULL;
    const void *src = fmt == 1 ? (const void *)w->q8 : fmt == 4 ? (const void *)w->q4 : (const void *)w->f;
    return coli_vk_tensor_ensure(t, src, fmt == 10 ? NULL : w->s, fmt, w->I, w->O, fmt == 4 ? w->gs : 0) ? *t : NULL;
}
/* the shared experts go where the per-matrix path puts them when it puts them on the
 * device (COLI_VK_DENSE): one copy for both */
static ColiVkTensor *k3c_w_shared(W *w, ColiVkTensor **t) {
    if (!*t && coli_vk_dense() && !w->vk) w_vk_upload(w);
    return k3c_w(w, t);
}
static ColiVkTensor *k3c_f32(const float *w, int I, int O, ColiVkTensor **own) {
    return coli_vk_tensor_ensure(own, w, NULL, 10, I, O, 0) ? *own : NULL;
}

/* ---- the KDA state between the host and the device ------------------------------ */
static void k3c_recover(Model *m, int dev, int upto);
/* The host's KDA state brought up to date (before anything reads it there). */
static void k3c_sync_host(Model *m) {
    K3Chain *ch = g_k3c;
    if (!ch || !ch->ok || ch->m != m || ch->where != K3C_DEV) return;
    const Cfg *c = &m->c;
    size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)c->kda_proj * c->conv_k;
    int reads = 0;   /* the copies that landed: after one, the host's copy is no longer the old one */
    for (int i = 0; i < c->n_layers; i++) {
        if (!m->L[i].kda) continue;
        if (!(vkc_read(ch->st[i], 0, m->kstate[i], ns * sizeof(float)) && ++reads) ||
            !(vkc_read(ch->win[i], 0, m->cwq[i], nw * sizeof(float)) && ++reads) ||
            !(vkc_read(ch->win[i], nw, m->cwk[i], nw * sizeof(float)) && ++reads) ||
            !(vkc_read(ch->win[i], 2 * nw, m->cwv[i], nw * sizeof(float)) && ++reads)) {
            /* lost halfway: the host's copy is part old, part new; it is rebuilt from zeros */
            if (reads) { ch->host_pos = 0; ch->host_zero = 1; }
            k3c_recover(m, 1, ch->dev_pos);
            return;
        }
    }
    ch->where = K3C_BOTH; ch->host_pos = ch->dev_pos; ch->host_zero = 0;
}
/* The host wrote its KDA state (zero: model_state_reset's zeros; else a restored
 * photo). A reset also drops the MLA rows: the mirror's watermark goes to 0. */
static void k3c_host_wrote(Model *m, int zero) {
    K3Chain *ch = g_k3c;
    if (!ch || !ch->ok || ch->m != m) return;
    ch->where = K3C_HOST; ch->host_zero = zero;
    if (zero) for (int i = 0; i < m->c.n_layers; i++) ch->kv_valid[i] = 0;
}
/* A CPU forward from pos0: the host's state current before it, the device's stale after
 * (and its MLA rows from pos0 on). */
static void k3c_cpu_step(Model *m, int pos0) {
    K3Chain *ch = g_k3c;
    if (!ch || !ch->ok || ch->m != m) return;
    k3c_sync_host(m);
    ch->where = K3C_HOST; ch->host_zero = 0;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid[i] > pos0) ch->kv_valid[i] = pos0;
}

/* The device was lost. If it held the newest KDA state (dev), the state is rebuilt on
 * the CPU: from the host's copy, current at host_pos, through the prefix record's ids
 * up to `upto`. The CPU runs from here on. */
static void k3c_recover(Model *m, int dev, int upto) {
    K3Chain *ch = g_k3c;
    g_k3c_on = 0;
    if (!ch) return;
    ch->failed = 1;
    int from = ch->host_pos, zero = ch->host_zero;
    int had = dev && ch->where == K3C_DEV;
    ch->where = K3C_HOST;
    if (!had || upto <= from) {
        fprintf(stderr, "[VK] kimi_k3 chain: the device was lost; the host's state is current, the CPU runs from here on\n");
        return;
    }
    if (m->kvp.tainted || !m->kvp.fed || m->kvp.len < upto) {
        fprintf(stderr, "[VK] kimi_k3 chain: the device was lost with a KDA state its token ids do not describe -- stopping "
                        "(COLI_VK_CHAIN=0 keeps the state on the CPU)\n");
        exit(1);
    }
    fprintf(stderr, "[VK] kimi_k3 chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", upto - from);
    const Cfg *c = &m->c;
    if (zero) for (int i = 0; i < c->n_layers; i++) {   /* the host was told "zeros" and kept its old arrays */
        if (!m->L[i].kda) continue;
        memset(m->kstate[i], 0, (size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float));
        memset(m->cwq[i], 0, (size_t)c->kda_proj * c->conv_k * sizeof(float));
        memset(m->cwk[i], 0, (size_t)c->kda_proj * c->conv_k * sizeof(float));
        memset(m->cwv[i], 0, (size_t)c->kda_proj * c->conv_k * sizeof(float));
    }
    int D = c->hidden, nbmax = ch->nbmax, chunk = 32;
    float *h = falloc((int64_t)chunk * D), *bres = falloc((int64_t)chunk * nbmax * D);
    for (int p = from; p < upto; p += chunk) {
        int n = upto - p < chunk ? upto - p : chunk, nb = 0;
        k3_embed(m, m->kvp.fed + p, p, n, h);
        k3_layers_forward_range(m, h, bres, &nb, p, n, 0, c->n_layers, NULL, NULL, NULL);
    }
    free(h); free(bres);
}

/* ---- setup ----------------------------------------------------------------------- */
static int k3c_setup(Model *m) {
    const Cfg *c = &m->c;
    int L = c->n_layers, D = c->hidden, P = c->kda_proj, LT = c->latent;
    int nbmax = (L + c->res_bs - 1) / c->res_bs, any_kda = 0, any_mla = 0;
    for (int i = 0; i < L; i++) { any_kda |= m->L[i].kda; any_mla |= !m->L[i].kda; }
    const char *why = NULL;
    if (!m->has_head) why = "no head (a layer range)";
    else if (any_kda && (c->kda_hd > 128 || c->conv_k > 8)) why = "a KDA head its shaders do not take";
    else if (any_mla && (c->kv_lora > 1024 || c->qk_rope > 128 || (c->qk_rope & 1) || c->qk_nope > 1024 || c->q_lora < 1))
        why = "an attention geometry its shaders do not take";
    else if (nbmax > 15) why = "more AttnRes blocks than its shader takes";
    if (why) { fprintf(stderr, "[VK] kimi_k3 chain: %s; the CPU runs the layers\n", why); return 0; }
    K3Chain *ch = calloc(1, sizeof *ch);
    if (!ch) return 0;
    size_t **offs[] = {&ch->o_in, &ch->o_post, &ch->o_asw, &ch->o_msw, &ch->o_qn, &ch->o_kn, &ch->o_conv, &ch->o_kda, &ch->o_latn};
    for (size_t k = 0; k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = calloc(L, sizeof(size_t)))) return 0;
    ch->kda = calloc(L, sizeof(K3cKda)); ch->ffn = calloc(L, sizeof(K3cFfn)); ch->mla = calloc(L, sizeof(VkcMla));
    ch->mg = calloc(L, sizeof(void *)); ch->kv = calloc(L, sizeof(VkcMlaCache));
    ch->win = calloc(L, sizeof(void *)); ch->st = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->mla_ord = calloc(L, sizeof(int));
    if (!ch->kda || !ch->ffn || !ch->mla || !ch->mg || !ch->kv || !ch->win || !ch->st || !ch->kv_valid || !ch->mla_ord) return 0;
    ch->m = m; ch->nbmax = nbmax;
    g_k3c = ch;
    /* the parameter arena: offsets, then one upload */
    size_t n = 0;
    for (int i = 0; i < L; i++) {
        const Layer *l = &m->L[i];
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D; ch->o_asw[i] = n; n += D; ch->o_msw[i] = n; n += D;
        if (l->kda) {
            ch->o_conv[i] = n; n += (size_t)3 * P * c->conv_k;
            ch->o_kda[i] = n; n += (size_t)c->kda_heads + P + c->kda_hd;
        } else {
            ch->mla_ord[i] = ch->n_mla++;
            ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += c->kv_lora;
        }
        if (l->sparse) { ch->o_latn[i] = n; n += LT; }
    }
    ch->o_osw = n; n += D; ch->o_final = n; n += D;
    float *a = calloc(n, sizeof(float));
    if (!a) return 0;
    for (int i = 0; i < L; i++) {
        const Layer *l = &m->L[i];
        memcpy(a + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(a + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(a + ch->o_asw[i], l->attn_sw, D * sizeof(float));
        memcpy(a + ch->o_msw[i], l->mlp_sw, D * sizeof(float));
        if (l->kda) {
            size_t ck = (size_t)P * c->conv_k;
            memcpy(a + ch->o_conv[i], l->a.conv_q, ck * sizeof(float));
            memcpy(a + ch->o_conv[i] + ck, l->a.conv_k, ck * sizeof(float));
            memcpy(a + ch->o_conv[i] + 2 * ck, l->a.conv_v, ck * sizeof(float));
            memcpy(a + ch->o_kda[i], l->a.A, c->kda_heads * sizeof(float));
            memcpy(a + ch->o_kda[i] + c->kda_heads, l->a.dt, P * sizeof(float));
            memcpy(a + ch->o_kda[i] + c->kda_heads + P, l->a.onw, c->kda_hd * sizeof(float));
        } else {
            memcpy(a + ch->o_qn[i], l->m.qa_ln, c->q_lora * sizeof(float));
            memcpy(a + ch->o_kn[i], l->m.kva_ln, c->kv_lora * sizeof(float));
        }
        if (l->sparse) memcpy(a + ch->o_latn[i], l->moe.lat_norm, LT * sizeof(float));
    }
    memcpy(a + ch->o_osw, m->out_sw, D * sizeof(float));
    memcpy(a + ch->o_final, m->final_norm, D * sizeof(float));
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, a, n * sizeof(float)) && vkc_submit(1);
    free(a);
    /* the tensors */
    for (int i = 0; i < L && ok; i++) {
        Layer *l = &m->L[i];
        if (l->kda) {
            Kda *k = &l->a; K3cKda *t = &ch->kda[i];
            ok = k3c_w(&k->q, &t->q) && k3c_w(&k->k, &t->k) && k3c_w(&k->v, &t->v) && k3c_w(&k->g, &t->g) &&
                 k3c_w(&k->o, &t->o) && k3c_f32(k->fa, D, c->kda_hd, &t->fa) && k3c_f32(k->fb, c->kda_hd, P, &t->fb) &&
                 k3c_f32(k->bp, D, c->kda_heads, &t->bp);
            if (ok) {
                ch->win[i] = vkc_buf((size_t)3 * P * c->conv_k * sizeof(float), VKC_DEV);
                ch->st[i] = vkc_buf((size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float), VKC_DEV);
                ok = ch->win[i] && ch->st[i];
            }
        } else {
            Mla *q = &l->m; VkcMla *t = &ch->mla[i];
            /* kv_b as it is: per head Q key rows then V value rows, the absorption reads it transposed */
            *t = (VkcMla){c->n_heads, c->qk_nope, c->qk_rope, c->v_head, c->kv_lora, D, c->q_lora, c->eps,
                          c->attn_scale, VKC_ROPE_HALF, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                          ch->prm, ch->o_qn[i], ch->o_kn[i]};
            ok = k3c_w(&q->qa, &t->q_a) && k3c_w(&q->qb, &t->q_b) && k3c_w(&q->kva, &t->kv_a) && k3c_w(&q->kvb, &t->kv_b) &&
                 k3c_w(&q->o, &t->o) && k3c_w(&q->g, &ch->mg[i]);
        }
        K3cFfn *f = &ch->ffn[i];
        if (ok && l->sparse) {
            Moe *o = &l->moe;
            ok = k3c_f32(o->router, D, c->n_experts, &f->router) && k3c_w(&o->lat_down, &f->down) &&
                 k3c_w(&o->lat_up, &f->up) && k3c_w_shared(&o->sh_gate, &f->sg) && k3c_w_shared(&o->sh_up, &f->su) &&
                 k3c_w_shared(&o->sh_down, &f->sd);
        } else if (ok)
            ok = k3c_w(&l->d_gate, &f->dg) && k3c_w(&l->d_up, &f->du) && k3c_w(&l->d_down, &f->dd);
    }
    ok = ok && k3c_w(&m->lm_head, &ch->head);
    if (!ok) { fprintf(stderr, "[VK] kimi_k3 chain: a matrix did not reach the device (or has no device form); the CPU runs the layers\n"); return 0; }
    ch->where = K3C_HOST; ch->host_zero = 1;   /* the host's KDA state is the zeros it was allocated with */
    ch->ok = 1;
    int nkda = 0, nsparse = 0; for (int i = 0; i < L; i++) { nkda += m->L[i].kda; nsparse += m->L[i].sparse; }
    fprintf(stderr, "[VK] kimi_k3 chain: %d layers on the device (%d KDA, %d MLA, %d dense MLP), %d AttnRes blocks, "
                    "%.1f MiB of parameters\n", L, nkda, ch->n_mla, L - nsparse, nbmax, n * 4 / 1048576.0);
    return 1;
}

/* scratch for `rows` rows (grows, never shrinks) */
static int k3c_scratch(K3Chain *ch, const Model *m, int rows) {
    const Cfg *c = &m->c;
    int D = c->hidden, P = c->kda_proj, H = c->kda_heads, LT = c->latent, E = c->n_experts, R = c->qk_rope;
    int SI = c->moe_inter * c->n_shared, MI = SI > c->dense_inter ? SI : c->dense_inter;
    int mfull = -1; for (int i = 0; i < c->n_layers; i++) if (!m->L[i].kda) { mfull = i; break; }
    size_t r = (size_t)rows;
    ch->kvd_layer = r * (c->kv_lora + R);
    int ok = (mfull < 0 || vkc_mla_scratch(&ch->sc, &ch->mla[mfull], rows)) &&
             k3c_res(&ch->x, r * D, VKC_DEV) && k3c_res(&ch->bres, r * ch->nbmax * D, VKC_DEV) &&
             k3c_res(&ch->hm, r * D, VKC_DEV) && k3c_res(&ch->nrm, r * D, VKC_DEV) && k3c_res(&ch->att, r * D, VKC_DEV) &&
             k3c_res(&ch->qkv3, 3 * r * P, VKC_DEV) && k3c_res(&ch->cm, r * 3 * P, VKC_DEV) &&
             k3c_res(&ch->t1, r * c->kda_hd, VKC_DEV) && k3c_res(&ch->kf, r * P, VKC_DEV) && k3c_res(&ch->kb, r * H, VKC_DEV) &&
             k3c_res(&ch->kg, r * P, VKC_DEV) && k3c_res(&ch->ky, r * P, VKC_DEV) &&
             k3c_res(&ch->gate, r * c->n_heads * c->v_head, VKC_DEV) && k3c_res(&ch->lg, r * E, VKC_DEV) &&
             k3c_res(&ch->z, r * LT, VKC_DEV) && k3c_res(&ch->gs, r * MI, VKC_DEV) && k3c_res(&ch->us, r * MI, VKC_DEV) &&
             k3c_res(&ch->hs, r * MI, VKC_DEV) && k3c_res(&ch->ds, r * D, VKC_DEV) && k3c_res(&ch->un, r * LT, VKC_DEV) &&
             k3c_res(&ch->lu, r * D, VKC_DEV) && k3c_res(&ch->fin, r * D, VKC_DEV) &&
             k3c_res(&ch->lgd, r * E, VKC_DOWN) && k3c_res(&ch->zd, r * LT, VKC_DOWN) &&
             k3c_res(&ch->kvd, (size_t)(ch->n_mla ? ch->n_mla : 1) * ch->kvd_layer, VKC_DOWN) &&
             k3c_res(&ch->find, r * D, VKC_DOWN) && k3c_res(&ch->outd, c->vocab, VKC_DOWN) &&
             k3c_res(&ch->routed, r * LT, VKC_UP);
    if (ok && R > 0) {   /* NoPE: the rope parts pass through a rotation by angle 0 */
        size_t had = vkc_bytes(ch->cs) / sizeof(float);
        ok = k3c_res(&ch->cs, r * R, VKC_UP);
        if (ok && vkc_bytes(ch->cs) / sizeof(float) != had) {
            float *cs = (float *)vkc_ptr(ch->cs);
            size_t nf = vkc_bytes(ch->cs) / sizeof(float);
            for (size_t i = 0; i + 1 < nf; i += 2) { cs[i] = 1.f; cs[i + 1] = 0.f; }
        }
    }
    if (!ok) return 0;
    if (ch->rows < rows) {
        float *hu = realloc(ch->host_u, r * LT * sizeof(float)), *hs = hu ? realloc(ch->host_sco, r * E * sizeof(float)) : NULL;
        if (hu) ch->host_u = hu;
        if (hs) ch->host_sco = hs;
        if (!hu || !hs) return 0;
        ch->rows = rows;
    }
    return 1;
}

/* The MLA mirror at the host's capacity (m->max_t), filled again when that grows. */
static int k3c_mirror(K3Chain *ch, const Model *m) {
    const Cfg *c = &m->c;
    if (ch->cap == m->max_t) return 1;
    for (int i = 0; i < c->n_layers; i++) {
        if (m->L[i].kda) continue;
        vkc_free(ch->kv[i].lat); vkc_free(ch->kv[i].rope);
        ch->kv[i] = (VkcMlaCache){NULL, NULL, 0};
        ch->kv_valid[i] = 0;
    }
    ch->cap = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (m->L[i].kda) continue;
        ch->kv[i].lat = vkc_buf((size_t)m->max_t * c->kv_lora * sizeof(float), VKC_DEV);
        if (c->qk_rope > 0) ch->kv[i].rope = vkc_buf((size_t)m->max_t * c->qk_rope * sizeof(float), VKC_DEV);
        ch->kv[i].cap = m->max_t;
        if (!ch->kv[i].lat || (c->qk_rope > 0 && !ch->kv[i].rope)) return 0;
    }
    ch->cap = m->max_t;
    return 1;
}

/* Record the uploads that make the device the host's below pb. */
static int k3c_push(K3Chain *ch, const Model *m, int pb) {
    const Cfg *c = &m->c; int ok = 1, K = c->kv_lora, R = c->qk_rope;
    if (ch->where == K3C_HOST) {
        size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)c->kda_proj * c->conv_k;
        for (int i = 0; i < c->n_layers && ok; i++) {
            if (!m->L[i].kda) continue;
            if (ch->host_zero) ok = vkc_zero(ch->st[i], 0, ns) && vkc_zero(ch->win[i], 0, 3 * nw);
            else ok = vkc_write(ch->st[i], 0, m->kstate[i], ns * sizeof(float)) &&
                      vkc_write(ch->win[i], 0, m->cwq[i], nw * sizeof(float)) &&
                      vkc_write(ch->win[i], nw, m->cwk[i], nw * sizeof(float)) &&
                      vkc_write(ch->win[i], 2 * nw, m->cwv[i], nw * sizeof(float));
        }
        ch->where = K3C_BOTH; ch->host_pos = pb;
    }
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (m->L[i].kda || ch->kv_valid[i] >= pb) continue;
        int t0 = ch->kv_valid[i], n = pb - t0;
        ok = vkc_write(ch->kv[i].lat, (size_t)t0 * K, m->Lc[i] + (size_t)t0 * K, (size_t)n * K * sizeof(float)) &&
             (R == 0 || vkc_write(ch->kv[i].rope, (size_t)t0 * R, m->Rc[i] + (size_t)t0 * R, (size_t)n * R * sizeof(float)));
        ch->kv_valid[i] = pb;
    }
    return ok;
}
/* the new MLA rows of layers [from, to) into the host's cache */
static void k3c_pull(K3Chain *ch, Model *m, int from, int to, int pb, int n) {
    const Cfg *c = &m->c; int K = c->kv_lora, R = c->qk_rope;
    for (int i = from; i < to; i++) {
        if (m->L[i].kda) continue;
        const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->mla_ord[i] * ch->kvd_layer;
        memcpy(m->Lc[i] + (size_t)pb * K, kv, (size_t)n * K * sizeof(float));
        if (R) memcpy(m->Rc[i] + (size_t)pb * R, kv + (size_t)n * K, (size_t)n * R * sizeof(float));
    }
}

/* ---- one layer's pieces ------------------------------------------------------------ */
static int k3c_norm(VkcBuf *x, VkcBuf *w, size_t wo, VkcBuf *y, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, 0, D, D, 0, D, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
static int k3c_mix(K3Chain *ch, const Model *m, size_t wo, int n, int nb) {   /* res_mix(hm, x, bres) */
    VkcAres a = {n, m->c.hidden, nb, 0, m->c.hidden, 0, ch->nbmax * m->c.hidden, (int)wo, 0, m->c.hidden, m->c.eps};
    return vkc_ares_mix(ch->x, ch->bres, ch->prm, ch->hm, &a);
}
static int k3c_kda(K3Chain *ch, const Model *m, int i, int n) {
    const Cfg *c = &m->c; int P = c->kda_proj, H = c->kda_heads, hd = c->kda_hd, r = ch->rows;
    const K3cKda *t = &ch->kda[i];
    VkcKdaConv cp = {n, 3 * P, c->conv_k, P, 0, P, r * P, 0, 3 * P, (int)ch->o_conv[i], 0};
    VkcKdaRec rp = {n, H, hd, P, 0, 3 * P, 0, P, 0, H, 0, P, 0, P, 0, (int)ch->o_kda[i], c->gate_lb, 1e-6f, c->eps};
    return vkc_matmul(t->q, ch->nrm, 0, ch->qkv3, 0, n) && vkc_matmul(t->k, ch->nrm, 0, ch->qkv3, (size_t)r * P, n) &&
           vkc_matmul(t->v, ch->nrm, 0, ch->qkv3, (size_t)2 * r * P, n) && vkc_matmul(t->g, ch->nrm, 0, ch->kg, 0, n) &&
           vkc_matmul(t->fa, ch->nrm, 0, ch->t1, 0, n) && vkc_matmul(t->fb, ch->t1, 0, ch->kf, 0, n) &&
           vkc_matmul(t->bp, ch->nrm, 0, ch->kb, 0, n) &&
           vkc_kda_conv(ch->qkv3, ch->prm, ch->win[i], ch->cm, &cp) &&
           vkc_kda_rec_flags(hd, ch->cm, ch->kf, ch->kb, ch->kg, ch->prm, ch->st[i], ch->ky, &rp, VKC_KDA_EXP_A | VKC_KDA_K3) &&
           vkc_matmul(t->o, ch->ky, 0, ch->att, 0, n);
}
static int k3c_mla(K3Chain *ch, const Model *m, int i, int n, int pb) {
    size_t ko = (size_t)ch->mla_ord[i] * ch->kvd_layer;
    return vkc_mla_qkv(&ch->mla[i], &ch->sc, ch->nrm, 0, n, pb, m->c.qk_rope > 0 ? ch->cs : NULL, &ch->kv[i], ch->kvd, ko) &&
           vkc_matmul(ch->mg[i], ch->nrm, 0, ch->gate, 0, n) &&
           vkc_mla_attn(&ch->mla[i], &ch->sc, n, pb, 0, &ch->kv[i], NULL, 0, 0, ch->gate, 0, ch->att, 0);
}
/* SiTU-GLU MLP of nrm into out: gate, up, the activation, down */
static int k3c_glu(K3Chain *ch, const Model *m, ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *d, int inter, VkcBuf *out, int n) {
    VkcSitu sp = {n * inter, 0, 0, 0, m->c.situ_b1, m->c.situ_b2};
    return vkc_matmul(g, ch->nrm, 0, ch->gs, 0, n) && vkc_matmul(u, ch->nrm, 0, ch->us, 0, n) &&
           vkc_situ(ch->gs, ch->us, ch->hs, &sp) && vkc_matmul(d, ch->hs, 0, out, 0, n);
}
/* the MoE output joining the prefix: x += lat_up(rmsnorm(routed)) + shared, moe_forward's order */
static int k3c_join(K3Chain *ch, const Model *m, int i, int n) {
    const Cfg *c = &m->c; int D = c->hidden, LT = c->latent;
    VkcNorm un = {n, LT, 1, 0, LT, LT, 0, LT, LT, (int)ch->o_latn[i], 0, 0, c->eps, 1.f};
    VkcEw cb = {VKC_EW_COMBINE, n * D, D, 1, 1 | 2, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_norm(ch->routed, ch->prm, ch->un, &un) && vkc_matmul(ch->ffn[i].up, ch->un, 0, ch->lu, 0, n) &&
           vkc_ew(ch->x, ch->x, ch->lu, ch->ds, NULL, &cb);
}
/* the host's MoE step of layer li: the router's choice from the device's logits, the
 * routed experts on the device's latent rows, their sum into the upload buffer */
static void k3c_moe_host(K3Chain *ch, Model *m, int li, int n) {
    const Cfg *c = &m->c; int E = c->n_experts, K = c->topk, LT = c->latent;
    Moe *o = &m->L[li].moe;
    memcpy(ch->host_sco, vkc_ptr(ch->lgd), (size_t)n * E * sizeof(float));
    int *idxs = malloc((size_t)n * K * sizeof(int)), *keff = malloc((size_t)n * sizeof(int));
    float *wsels = falloc((int64_t)n * K);
    if (!idxs || !keff) { fprintf(stderr, "OOM moe sel\n"); exit(1); }
    k3_moe_route(m, o, li, ch->host_sco, n, idxs, wsels, keff);
    memset(ch->host_u, 0, (size_t)n * LT * sizeof(float));
    float *sd = k3_moe_experts(m, o, li, NULL, (const float *)vkc_ptr(ch->zd), n, keff, idxs, wsels, ch->host_u);
    free(sd); free(idxs); free(keff); free(wsels);
    memcpy(vkc_ptr(ch->routed), ch->host_u, (size_t)n * LT * sizeof(float));
}

/* Every layer for C rows of `hidden` (positions pos0..), then the output mix and the
 * final norm: the rows into *fin (when need_rows), the last row's logits into *last.
 * 0: not taken (the CPU runs the layers; nothing the caller holds changed). On a
 * cancel (poll), 1 with *cancelled set: the state is the caller's to reset. */
static int k3c_forward(Model *m, const float *hidden, int pos0, int C, int need_rows, float **fin_out,
                       float **last_out, K3CancelPoll poll, void *pctx, int *cancelled) {
    K3Chain *ch = g_k3c;
    if (!g_k3c_on || !ch || !ch->ok || ch->failed || ch->m != m) return 0;
    if (g_k3c_on == COLI_VK_CHAIN_PREFILL && C <= 2) return 0;   /* prompts only: decode on the CPU */
    if (m->trace || g_k3_val_layer >= 0 || g_k3_dfp) {
        static int said = 0;
        if (!said++) fprintf(stderr, "[VK] kimi_k3 chain: a validation dump reads every layer on the host; the CPU runs the layers\n");
        return 0;
    }
    const Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, E = c->n_experts, LT = c->latent;
    int CH = k3c_chunk_rows(), rows = C < CH ? C : CH;
    int dev0 = ch->where == K3C_DEV, dev_start = ch->dev_pos;
    if (vkc_lost()) { k3c_recover(m, dev0, dev_start); return 0; }
    if (!k3c_mirror(ch, m) || !k3c_scratch(ch, m, rows)) {
        if (vkc_lost()) { k3c_recover(m, dev0, dev_start); return 0; }   /* lost while its buffers were made */
        fprintf(stderr, "[VK] kimi_k3 chain: device memory for %d rows at %d positions refused; the CPU runs the layers\n",
                rows, m->max_t);
        k3c_cpu_step(m, pos0);
        ch->failed = 1; g_k3c_on = 0;
        return 0;
    }
    float *fin = need_rows ? falloc((int64_t)C * D) : NULL, *last = falloc(c->vocab);
    vkc_gemm_rows(-1);
    for (int c0 = 0; c0 < C; c0 += rows) {
        int n = C - c0 < rows ? C - c0 : rows, pb = pos0 + c0, end = c0 + n == C;
        if (!vkc_begin() || !k3c_push(ch, m, pb) || !vkc_write(ch->x, 0, hidden + (size_t)c0 * D, (size_t)n * D * sizeof(float)))
            goto lost;
        int ok = 1, pending = 0, pulled = 0, nb = 0;
        for (int i = 0; i < L && ok; i++) {
            const Layer *l = &m->L[i];
            if (pending) { ok = k3c_join(ch, m, i - 1, n); pending = 0; }
            int snap = i % c->res_bs == 0;
            double ta = now_s();
            /* the attention site: the mix over the snapshots so far, this block's snapshot */
            VkcBuf *src = ch->x;
            if (ok && nb > 0) { ok = k3c_mix(ch, m, ch->o_asw[i], n, nb); src = ch->hm; }
            if (ok && snap) {
                VkcRegion rg[64];
                for (int s0 = 0; s0 < n && ok; s0 += 64) {
                    int k = n - s0 < 64 ? n - s0 : 64;
                    for (int s = 0; s < k; s++)
                        rg[s] = (VkcRegion){(size_t)(s0 + s) * ch->nbmax * D + (size_t)nb * D, (size_t)(s0 + s) * D, (size_t)D};
                    ok = vkc_copy_regions(ch->bres, ch->x, rg, k);
                }
            }
            ok = ok && k3c_norm(src, ch->prm, ch->o_in[i], ch->nrm, n, D, c->eps);
            int have_prefix = !snap;
            if (snap) nb++;
            ok = ok && (l->kda ? k3c_kda(ch, m, i, n) : k3c_mla(ch, m, i, n, pb));
            if (ok) {
                VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = have_prefix ? vkc_ew(ch->x, ch->x, ch->att, NULL, NULL, &add) : vkc_copy(ch->x, 0, ch->att, 0, (size_t)n * D);
            }
            /* the MLP site */
            ok = ok && k3c_mix(ch, m, ch->o_msw[i], n, nb) && k3c_norm(ch->hm, ch->prm, ch->o_post[i], ch->nrm, n, D, c->eps);
            if (!ok) break;
            if (!l->sparse) {          /* a dense layer: its MLP and the add, no host step */
                VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                const K3cFfn *f = &ch->ffn[i];
                ok = k3c_glu(ch, m, f->dg, f->du, f->dd, c->dense_inter, ch->att, n) &&
                     vkc_ew(ch->x, ch->x, ch->att, NULL, NULL, &add);
                m->t_attn += now_s() - ta;
                continue;
            }
            ok = vkc_matmul(ch->ffn[i].router, ch->nrm, 0, ch->lg, 0, n) && vkc_matmul(ch->ffn[i].down, ch->nrm, 0, ch->z, 0, n) &&
                 vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && vkc_copy(ch->zd, 0, ch->z, 0, (size_t)n * LT) &&
                 vkc_submit(1);                                    /* A1 */
            m->t_attn += now_s() - ta;
            if (!ok) break;
            k3c_pull(ch, m, pulled, i + 1, pb, n); pulled = i + 1;
            /* A2: the shared experts, while the host computes the routed ones */
            const K3cFfn *f = &ch->ffn[i];
            ok = vkc_begin() && k3c_glu(ch, m, f->sg, f->su, f->sd, c->moe_inter * c->n_shared, ch->ds, n) && vkc_submit(0);
            double t1 = now_s();
            k3c_moe_host(ch, m, i, n);
            double dt = now_s() - t1; ch->host_ms += dt * 1e3; m->t_moe += dt;
            if (ok && poll && poll(pctx)) {   /* a cancel: what the device advanced is the caller's to reset */
                if (cancelled) *cancelled = 1;
                vkc_finish();
                ch->where = K3C_DEV; ch->dev_pos = pb + n;
                free(fin); free(last);
                return 1;
            }
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok && pending) ok = k3c_join(ch, m, L - 1, n);
        /* the output mix and the final norm of every row, lm_head on the last */
        ok = ok && k3c_mix(ch, m, ch->o_osw, n, nb) && k3c_norm(ch->hm, ch->prm, ch->o_final, ch->fin, n, D, c->eps);
        if (ok && need_rows) ok = vkc_copy(ch->find, 0, ch->fin, 0, (size_t)n * D);
        if (ok && end) ok = vkc_matmul(ch->head, ch->fin, (size_t)(n - 1) * D, ch->outd, 0, 1);
        ok = ok && vkc_submit(1);
        if (!ok) goto lost;
        k3c_pull(ch, m, pulled, L, pb, n);
        if (need_rows) memcpy(fin + (size_t)c0 * D, vkc_ptr(ch->find), (size_t)n * D * sizeof(float));
        for (int i = 0; i < L; i++) if (!m->L[i].kda) ch->kv_valid[i] = pb + n;
        ch->where = K3C_DEV;
        if (end) memcpy(last, vkc_ptr(ch->outd), (size_t)c->vocab * sizeof(float));
    }
    ch->dev_pos = pos0 + C;
    ch->forwards++;
    *fin_out = fin; *last_out = last;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    free(fin); free(last);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    ch->where = dev0 ? K3C_DEV : K3C_HOST;   /* the state as it was when this forward began */
    k3c_recover(m, dev0, dev_start);
    return 0;
}

static void k3c_report(void) {
    K3Chain *ch = g_k3c;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] kimi_k3 chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
    vkc_prof_print();
}

/* COLI_VK_CHAIN at startup, before the tier sizes itself: the decision, the pipelines,
 * the tensors. tier_on: whether the routed-expert tier is about to start. */
static void k3c_start(Model *m, int tier_on) {
    if (!g_k3_vk) return;
    int on = coli_vk_chain_decide("kimi_k3", tier_on, COLI_VK_CHAIN_UNMEASURED);
    const char *no = NULL;
    if (on) {
#ifdef COLI_CUDA
        if (g_k3_cuda) no = "the CUDA expert tier is on and keeps its priority";
#endif
        if (!no && k3_dsa_indexer_on()) no = "KIMI_DSA_INDEXER=1 fills its index cache on the CPU";
        if (!no && !(g_k3c_inited = vkc_init())) no = "the chain's pipelines did not come up";
        if (!no && !(vkc_mla_ready() && vkc_kda_ready() && vkc_ares_ready())) no = "the MLA, KDA or AttnRes shaders are missing";
    }
    if (no) fprintf(stderr, "[VK] kimi_k3: %s: the dense chain stays off\n", no);
    if (!on || no) return;
    if (!k3c_setup(m)) return;
    g_k3c_on = on;
}
/* After the tier's: at exit the chain goes before the device. */
static void k3c_atexit(void) {
    if (g_k3c_inited) atexit(vkc_shutdown);
}
/* The per-matrix path is about to put the shared experts on the device (no tier after
 * all): the chain's copies become its own. */
static void k3c_adopt_shared(Model *m) {
    K3Chain *ch = g_k3c;
    if (!ch || !ch->ok || ch->m != m) return;
    for (int i = 0; i < m->c.n_layers; i++) {
        if (!m->L[i].sparse) continue;
        Moe *o = &m->L[i].moe; K3cFfn *f = &ch->ffn[i];
        if (!o->sh_gate.vk) o->sh_gate.vk = f->sg;
        if (!o->sh_up.vk) o->sh_up.vk = f->su;
        if (!o->sh_down.vk) o->sh_down.vk = f->sd;
    }
}
