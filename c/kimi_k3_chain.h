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
 * Past the device's budget (vk_kvsplit.h) each MLA layer keeps only a window of the
 * newest blocks of its latent and rope rows on the device; the attention core runs
 * over those on the device and over the older positions on the CPU (from the host's
 * canonical cache) at once, merged through the softmax statistics.
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
 * (a KDA head above 128 floats, kv_lora above 1024, qk_rope above 128 or odd).
 *
 * Layers on two devices (docs/vulkan.md, "Layers on two devices"): with COLI_VK_DEV2 a
 * second chain takes the layers after the primary's on that device (its own fit from
 * layer N, COLI_VK_CHAIN_LAYERS2; its own copies of their matrices, which the per-matrix
 * path never sees). The primary hands each row's AttnRes state (the prefix, the block
 * snapshots and their count) to it as it would to the CPU; the head stays on the host.
 * Both chains' KDA state moves together (the host's copy current for both, or for
 * neither; the primary's host_pos, host_zero and dev_pos speak for both): the second
 * device's is read back first, so a read that fails leaves the primary's as it was.
 * Losing either device turns both off and rebuilds the state of the layers whose newest
 * state was on a device. */
#include "vk_chain.h"
#include "vk_kvsplit.h"

#define K3C_HOST 0
#define K3C_DEV  1
#define K3C_BOTH 2

typedef struct { ColiVkTensor *q, *k, *v, *g, *o, *fa, *fb, *bp; } K3cKda;
typedef struct { ColiVkTensor *router, *down, *up, *sg, *su, *sd, *dg, *du, *dd; } K3cFfn;

typedef struct {
    const Model *m;
    int ok, failed, rows, cap, nbmax;
    int lo, d;                                /* its layers start at lo, on device d (vkc_device) */
    VkcBuf *prm;                              /* every norm, AttnRes weight, conv tap and KDA parameter */
    size_t *o_in, *o_post, *o_asw, *o_msw, *o_qn, *o_kn, *o_conv, *o_kda, *o_latn, o_osw, o_final;
    K3cKda *kda; K3cFfn *ffn; VkcMla *mla; ColiVkTensor **mg, *head;
    VkcMlaCache *kv; VkcBuf **win, **st;
    int *kv_valid, *mla_ord, n_mla;
    int dev_rows;                             /* device latent/rope rows a layer: cap, or the split's */
    VkcKvSplit ks;                            /* the split past the device's budget (ks.on) */
    VkcMlaCache kvtmp;                        /* the split: a step's new rows before their slots */
    int where, host_zero, dev_pos, host_pos;  /* the KDA state: who holds the newest, where each side is */
    VkcMlaScratch sc;
    VkcBuf *x, *bres, *hm, *nrm, *att, *qkv3, *cm, *t1, *kf, *kb, *kg, *ky, *gate, *lg, *z;
    VkcBuf *gs, *us, *hs, *ds, *un, *lu, *fin, *cs;
    VkcBuf *lgd, *zd, *kvd, *find, *outd, *routed;
    size_t kvd_layer;
    float *host_u, *host_sco;
    unsigned long long forwards;
    double host_ms;
    int n;                                    /* the layers on the device: lo..lo+n-1 (a partial chain) */
    VkcBuf *xo, *bo;                          /* the handoff after layer n - 1: the prefix and the block snapshots, down */
} K3Chain;

static K3Chain *g_k3c;
static int g_k3c_on;       /* COLI_VK_CHAIN as decided (on, or prompts only), the chain set up */
static int g_k3c_inited;   /* vkc_init ran: vkc_shutdown goes at exit */
/* the second device's chain (layers g_k3_fit.n..), its fit; g_k3c_dev: the device the
 * chain's uploads go to */
static K3Chain *g_k3c2;
static VkcFit g_k3c_fit2;
static int g_k3c_fit2_on, g_k3c_dev;
static K3Chain *k3c_of(int d) { return d ? g_k3c2 : g_k3c; }
static const char *k3c_name(const K3Chain *ch) { return ch && ch->d ? "kimi_k3 dev2" : "kimi_k3"; }
static int k3c_dev2_wanted(void) { const char *e = getenv("COLI_VK_CHAIN_DEV2"); return !(e && *e == '0'); }
/* The chain's own f32 tensors (the router, the KDA decay pair, beta): with the dense
 * weights on the device only (COLI_VK_DENSE_HOST) the per-matrix path multiplies by them
 * too, and the CPU drops its copies. which: 0 the router, 1 f_a, 2 f_b, 3 b_proj. */
static ColiVkTensor *k3c_f32_tensor(int li, int which) {
    K3Chain *ch = g_k3c;
    if (!ch || !ch->ok || li < 0 || li >= ch->m->c.n_layers) return NULL;
    switch (which) {
    case 0: return ch->ffn[li].router;
    case 1: return ch->kda[li].fa;
    case 2: return ch->kda[li].fb;
    case 3: return ch->kda[li].bp;
    }
    return NULL;
}

/* k3c_res counts instead of reserving while g_k3c_count >= 0 (the chunk's sizing) */
static long long g_k3c_count = -1;
static int k3c_res(VkcBuf **b, size_t floats, int kind) {
    if (g_k3c_count >= 0) { g_k3c_count += (long long)(floats ? floats : 1) * (long long)sizeof(float); return 1; }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}
static int k3c_scratch(K3Chain *ch, const Model *m, int rows);
/* Prompt rows per chunk (vkc_chunk_rows): the chain's scratch a row, counted from the
 * reservations (the MLA scratch as vkc_mla_scratch sizes it), and the routed experts'
 * outputs (the tier's rows, the CPU's, the host's sum, in the latent space) for it. */
static int k3c_chunk_rows(K3Chain *ch, const Model *m) {
    if (!vkc_chunk_auto()) return vkc_chunk_rows(k3c_name(ch), 0);
    const Cfg *c = &m->c;
    size_t mla = 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) if (!m->L[i].kda) {
        const VkcMla *a = &ch->mla[i];
        mla = (size_t)(a->q_lora > 0 ? a->q_lora : 1) + (size_t)a->H * (a->Q + a->R) + (size_t)(a->K + a->R) +
              2 * (size_t)a->H * a->K + (size_t)a->H * a->V;
        break;
    }
    size_t ksz = ch->kvd_layer;
    g_k3c_count = 0; k3c_scratch(ch, m, 1); long long b1 = g_k3c_count;
    g_k3c_count = 0; k3c_scratch(ch, m, 2); long long b2 = g_k3c_count;
    g_k3c_count = -1; ch->kvd_layer = ksz;
    size_t row = (size_t)(b2 - b1) + mla * sizeof(float) + (size_t)(2 * c->topk + 1) * c->latent * sizeof(float);
    return vkc_chunk_rows(k3c_name(ch), row);
}

/* A W's device copy into *t: the per-matrix path's own where it made one (the shared
 * experts under COLI_VK_DENSE), else one of the chain's, which w_matmul never sees (so
 * a forward the chain declines stays on the CPU, as before). The second device's chain
 * makes its own there, always. */
static ColiVkTensor *k3c_w(W *w, ColiVkTensor **t) {
    if (g_k3c_dev) {
        if (*t) return *t;
        int fmt = k3_vk_fmt(w);
        if (fmt < 0 || w->mapped) return NULL;
        const void *src = fmt == 1 ? (const void *)w->q8 : fmt == 4 ? (const void *)w->q4 : (const void *)w->f;
        return coli_vk_tensor_ensure2(t, src, fmt == 10 ? NULL : w->s, fmt, w->I, w->O, fmt == 4 ? w->gs : 0) ? *t : NULL;
    }
    if (!*t && w->vk) *t = (ColiVkTensor *)w->vk;
    if (*t) return *t;
    /* the dense weights on the device only, the chain's fit: the per-matrix path's copy,
     * made here layer by layer (k3_vk_tier_start leaves the chain's layers to it) */
    if (g_k3_dho && g_k3_fit.L > 0 && w_vk_upload(w)) return *t = (ColiVkTensor *)w->vk;
    int fmt = k3_vk_fmt(w);
    if (fmt < 0 || w->mapped) return NULL;
    const void *src = fmt == 1 ? (const void *)w->q8 : fmt == 4 ? (const void *)w->q4 : (const void *)w->f;
    return coli_vk_tensor_ensure(t, src, fmt == 10 ? NULL : w->s, fmt, w->I, w->O, fmt == 4 ? w->gs : 0) ? *t : NULL;
}
/* the shared experts go where the per-matrix path puts them when it puts them on the
 * device (COLI_VK_DENSE): one copy for both */
static ColiVkTensor *k3c_w_shared(W *w, ColiVkTensor **t) {
    if (!*t && !g_k3c_dev && coli_vk_dense() && !w->vk) w_vk_upload(w);
    return k3c_w(w, t);
}
static ColiVkTensor *k3c_f32(const float *w, int I, int O, ColiVkTensor **own) {
    return (g_k3c_dev ? coli_vk_tensor_ensure2(own, w, NULL, 10, I, O, 0) : coli_vk_tensor_ensure(own, w, NULL, 10, I, O, 0)) ? *own : NULL;
}

/* ---- the KDA state between the host and the device ------------------------------ */
/* (each covers both chains: the second device's first; the primary's host_pos, host_zero
 * and dev_pos speak for both, see the top) */
static void k3c_recover(Model *m, int dev0, int dev2, int upto, const K3Chain *who);
/* One chain's KDA state read back into the host's copy; 0: lost on the way (*reads: how
 * many copies landed before). */
static int k3c_sync_one(Model *m, K3Chain *ch, int *reads) {
    const Cfg *c = &m->c;
    size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)c->kda_proj * c->conv_k;
    int was = vkc_device(ch->d), ok = 1;
    for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {   /* the chain's layers: the CPU's hold their own */
        if (!m->L[i].kda) continue;
        ok = (vkc_read(ch->st[i], 0, m->kstate[i], ns * sizeof(float)) && ++*reads) &&
             (vkc_read(ch->win[i], 0, m->cwq[i], nw * sizeof(float)) && ++*reads) &&
             (vkc_read(ch->win[i], nw, m->cwk[i], nw * sizeof(float)) && ++*reads) &&
             (vkc_read(ch->win[i], 2 * nw, m->cwv[i], nw * sizeof(float)) && ++*reads);
    }
    vkc_device(was);
    if (ok) ch->where = K3C_BOTH;
    return ok;
}
/* The host's KDA state brought up to date (before anything reads it there). */
static void k3c_sync_host(Model *m) {
    K3Chain *ch = g_k3c, *ch2 = g_k3c2;
    if (!ch || !ch->ok || ch->m != m) return;
    int synced = 0;
    for (int d = 1; d >= 0; d--) {
        K3Chain *x = k3c_of(d);
        if (!x || !x->ok || x->where != K3C_DEV) continue;
        int reads = 0;   /* the copies that landed: after one, the host's copy is no longer the old one */
        if (!k3c_sync_one(m, x, &reads)) {
            /* lost halfway: the host's copy is part old, part new; it is rebuilt from zeros */
            if (reads) { ch->host_pos = 0; ch->host_zero = 1; }
            k3c_recover(m, ch->where == K3C_DEV, ch2 && ch2->ok && ch2->where == K3C_DEV, ch->dev_pos, x);
            return;
        }
        synced = 1;
    }
    if (!synced) return;
    ch->host_pos = ch->dev_pos;
    for (int d = 0; d < 2; d++) { K3Chain *x = k3c_of(d); if (x) x->host_zero = 0; }
}
/* The host wrote its KDA state (zero: model_state_reset's zeros; else a restored
 * photo). A reset also drops the MLA rows: the mirror's watermark goes to 0. */
static void k3c_host_wrote(Model *m, int zero) {
    for (int d = 0; d < 2; d++) {
        K3Chain *ch = k3c_of(d);
        if (!ch || !ch->ok || ch->m != m) continue;
        ch->where = K3C_HOST; ch->host_zero = zero;
        if (zero) { for (int i = 0; i < m->c.n_layers; i++) ch->kv_valid[i] = 0; vkc_kv_reset(&ch->ks); }
    }
}
/* The CPU runs one chain's layers from pos0: the device's state stale after (and its MLA
 * rows from pos0 on). */
static void k3c_cpu_one(Model *m, K3Chain *ch, int pos0) {
    ch->where = K3C_HOST; ch->host_zero = 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        if (ch->kv_valid[i] > pos0) ch->kv_valid[i] = pos0;
        if (!m->L[i].kda) vkc_kv_lower(&ch->ks, ch->mla_ord[i], pos0);
    }
}
/* A CPU forward from pos0: the host's state current before it. */
static void k3c_cpu_step(Model *m, int pos0) {
    K3Chain *ch = g_k3c;
    if (!ch || !ch->ok || ch->m != m) return;
    k3c_sync_host(m);
    for (int d = 0; d < 2; d++) { K3Chain *x = k3c_of(d); if (x && x->ok && x->m == m) k3c_cpu_one(m, x, pos0); }
}

/* A device was lost (who: the chain whose device it was); both chains go off. If the
 * devices held the newest KDA state (dev0, dev2: the primary's and the second device's),
 * the state of their layers is rebuilt on the CPU: from the host's copy, current at
 * host_pos, through the prefix record's ids up to `upto`. The CPU runs from here on. */
static void k3c_recover(Model *m, int dev0, int dev2, int upto, const K3Chain *who) {
    K3Chain *ch = g_k3c, *ch2 = g_k3c2;
    g_k3c_on = 0;
    if (!ch) return;
    int from = ch->host_pos, zero = ch->host_zero;
    int had = dev0 || dev2, hi = ch->lo + ch->n + (dev2 && ch2 ? ch2->n : 0);   /* the layers whose state the devices held */
    for (int d = 0; d < 2; d++) { K3Chain *x = k3c_of(d); if (x) { x->failed = 1; x->where = K3C_HOST; } }
    const char *nm = k3c_name(who);
    if (!had || upto <= from) {
        fprintf(stderr, "[VK] %s chain: the device was lost; the host's state is current, the CPU runs from here on\n", nm);
        return;
    }
    if (m->kvp.tainted || !m->kvp.fed || m->kvp.len < upto || !dev0) {
        fprintf(stderr, "[VK] %s chain: the device was lost with a KDA state its token ids do not describe -- stopping "
                        "(COLI_VK_CHAIN=0 keeps the state on the CPU)\n", nm);
        exit(1);
    }
    fprintf(stderr, "[VK] %s chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", nm, upto - from);
    const Cfg *c = &m->c;
    if (zero) for (int i = 0; i < hi; i++) {   /* the host was told "zeros" and kept its old arrays (the devices' layers) */
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
        k3_layers_forward_range(m, h, bres, &nb, p, n, 0, hi, NULL, NULL, NULL);   /* the devices' layers: the CPU's are current */
    }
    free(h); free(bres);
}

/* ---- setup ----------------------------------------------------------------------- */
/* ---- the partial chain (vk_chain.h, vkc_fit): the first N layers on the device ----------
 * N is decided once, before anything goes up and before the expert cache is sized
 * (k3c_fit_now, from k3_dho_decide once the device is open): the layers that fit the
 * device's free memory less the tier's reserve, the chain's scratch for one prompt chunk
 * and the pools' granularity; the head is the tail. The CPU runs layers N.. and the head:
 * each row's AttnRes state after layer N - 1 (the prefix and the block snapshots) crosses
 * to the host once per forward (k3c_forward). The chain's layers keep their KDA state and
 * MLA mirrors on the device as before; the CPU's layers keep theirs on the host. */
static const char *k3c_unsupported(const Model *m) {
    const Cfg *c = &m->c;
    int L = c->n_layers, nbmax = (L + c->res_bs - 1) / c->res_bs, any_kda = 0, any_mla = 0;
    for (int i = 0; i < L; i++) { any_kda |= m->L[i].kda; any_mla |= !m->L[i].kda; }
    if (!m->has_head) return "no head (a layer range)";
    if (any_kda && (c->kda_hd > 128 || c->conv_k > 8)) return "a KDA head its shaders do not take";
    if (any_mla && (c->kv_lora > 1024 || c->qk_rope > 128 || (c->qk_rope & 1) || c->qk_nope > 1024 || c->q_lora < 1))
        return "an attention geometry its shaders do not take";
    if (nbmax > 15) return "more AttnRes blocks than its shader takes";
    return NULL;
}
/* a W as k3c_w puts it on the device: its bytes there and its payload */
static void k3c_fit_w(const W *w, size_t *b, size_t *mm) {
    int fmt = k3_vk_fmt(w);
    if (fmt < 0 || w->mapped || !(w->f || w->q8 || w->q4) || w->O < 1 || w->I < 1) return;
    int gs = fmt == 4 ? w->gs : 0;
    *b += vkc_fit_tensor(fmt, w->I, w->O, gs); *mm += coli_vk_tensor_payload(fmt, w->I, w->O, gs);
}
static void k3c_fit_f32(int I, int O, size_t *b, size_t *mm) {
    *b += vkc_fit_tensor(10, I, O, 0); *mm += coli_vk_tensor_payload(10, I, O, 0);
}
/* What layer i puts on the device, as vkc_fit counts it: its matrices (the router and the
 * KDA decay and beta projections as f32), its state at its first size (the KDA state and
 * convolution windows, an MLA layer's mirror at `rows` positions and its rows of the down
 * buffer), its part of the parameters; *mat its matrices' payload. */
static size_t k3c_layer_bytes(const Model *m, int i, int rows, size_t *mat) {
    const Cfg *c = &m->c; const Layer *l = &m->L[i];
    int D = c->hidden, P = c->kda_proj;
    size_t b = 0, mm = 0, f = (size_t)4 * D;
    if (l->kda) {
        const Kda *k = &l->a;
        k3c_fit_w(&k->q, &b, &mm); k3c_fit_w(&k->k, &b, &mm); k3c_fit_w(&k->v, &b, &mm);
        k3c_fit_w(&k->g, &b, &mm); k3c_fit_w(&k->o, &b, &mm);
        k3c_fit_f32(D, c->kda_hd, &b, &mm); k3c_fit_f32(c->kda_hd, P, &b, &mm); k3c_fit_f32(D, c->kda_heads, &b, &mm);
        b += vkc_fit_buf((size_t)3 * P * c->conv_k * sizeof(float)) +
             vkc_fit_buf((size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float));
        f += (size_t)3 * P * c->conv_k + c->kda_heads + P + c->kda_hd;
    } else {
        const Mla *q = &l->m;
        k3c_fit_w(&q->qa, &b, &mm); k3c_fit_w(&q->qb, &b, &mm); k3c_fit_w(&q->kva, &b, &mm);
        k3c_fit_w(&q->kvb, &b, &mm); k3c_fit_w(&q->o, &b, &mm); k3c_fit_w(&q->g, &b, &mm);
        b += vkc_fit_buf((size_t)rows * c->kv_lora * sizeof(float)) +
             (c->qk_rope > 0 ? vkc_fit_buf((size_t)rows * c->qk_rope * sizeof(float)) : 0);
        f += (size_t)c->q_lora + c->kv_lora + (size_t)rows * (c->kv_lora + c->qk_rope);
    }
    if (l->sparse) {
        const Moe *o = &l->moe;
        k3c_fit_f32(D, c->n_experts, &b, &mm);
        k3c_fit_w(&o->lat_down, &b, &mm); k3c_fit_w(&o->lat_up, &b, &mm);
        k3c_fit_w(&o->sh_gate, &b, &mm); k3c_fit_w(&o->sh_up, &b, &mm); k3c_fit_w(&o->sh_down, &b, &mm);
        f += c->latent;
    } else { k3c_fit_w(&l->d_gate, &b, &mm); k3c_fit_w(&l->d_up, &b, &mm); k3c_fit_w(&l->d_down, &b, &mm); }
    *mat = mm;
    return b + f * sizeof(float);
}
/* What the chain allocates whatever N: the scratch of one prompt chunk of `rows` rows (the
 * counting pass of k3c_scratch, the MLA scratch as k3c_chunk_rows counts it), the
 * parameters after the last layer (the output mix and the final norm). */
static size_t k3c_fixed_bytes(Model *m, int rows) {
    const Cfg *c = &m->c;
    K3Chain t;
    memset(&t, 0, sizeof t);
    t.m = m; t.n = c->n_layers; t.nbmax = (c->n_layers + c->res_bs - 1) / c->res_bs;
    g_k3c_count = 0;
    k3c_scratch(&t, m, rows);
    size_t b = (size_t)g_k3c_count;
    g_k3c_count = -1;
    for (int i = 0; i < c->n_layers; i++) if (!m->L[i].kda) {
        b += (size_t)rows * ((size_t)c->q_lora + (size_t)c->n_heads * (c->qk_nope + c->qk_rope) + (size_t)(c->kv_lora + c->qk_rope) +
                             2 * (size_t)c->n_heads * c->kv_lora + (size_t)c->n_heads * c->v_head) * sizeof(float);
        break;
    }
    return b + (size_t)2 * c->hidden * sizeof(float);
}
static int k3c_fit_now(Model *m, int tier_on, int cuda_on) {
    if (g_k3_fit_done) return g_k3_fit.L > 0;
    g_k3_fit_done = 1;
    if (!g_k3_vk || cuda_on || k3_dsa_indexer_on() || k3c_unsupported(m)) return 0;
    if (coli_vk_chain_decide(NULL, tier_on, COLI_VK_CHAIN_UNMEASURED) == COLI_VK_CHAIN_OFF) return 0;
    const Cfg *c = &m->c;
    int L = c->n_layers, rows = vkc_fit_rows(256);
    size_t *per = calloc((size_t)L, sizeof(size_t)), *mat = calloc((size_t)L, sizeof(size_t)), tail = 0, tm = 0;
    if (!per || !mat) { free(per); free(mat); return 0; }
    for (int i = 0; i < L; i++) per[i] = k3c_layer_bytes(m, i, rows, &mat[i]);
    k3c_fit_w(&m->lm_head, &tail, &tm);
    size_t fixed = k3c_fixed_bytes(m, rows);
    vkc_fit("kimi_k3", L, per, mat, fixed, tail, &g_k3_fit);
    g_k3_partial = vkc_fit_partial(&g_k3_fit);
    /* the layers the primary leaves, on COLI_VK_DEV2's device: a fit of its own from layer
     * n0 (the head stays on the host), with that device's free memory, its pipelines up now */
    int n0 = g_k3_fit.n;
    if (g_k3_partial && n0 > 0 && n0 < L && k3c_dev2_wanted() && getenv("COLI_VK_DEV2") && coli_vk_dev2_open_env()) {
        vkc_device(1);
        int n2 = vkc_fit("kimi_k3 dev2", L - n0, per + n0, mat + n0, fixed, 0, &g_k3c_fit2);
        if (n2 > 0 && !(vkc_init() && vkc_mla_ready() && vkc_kda_ready() && vkc_ares_ready())) {
            fprintf(stderr, "[VK] kimi_k3 chain: the second device's pipelines did not come up; its layers stay on the CPU\n");
            n2 = 0;
        }
        g_k3c_fit2_on = n2 > 0;
        vkc_device(0);
    }
    free(per); free(mat);
    return 1;
}

/* Layer i's tensors and state on the device. 0 with *why set. */
static int k3c_layer_up(K3Chain *ch, Model *m, int i, const char **why) {
    const Cfg *c = &m->c; Layer *l = &m->L[i];
    int D = c->hidden, P = c->kda_proj;
    int ok;
    *why = "a matrix the device refused (or one with no device form)";
    if (l->kda) {
        Kda *k = &l->a; K3cKda *t = &ch->kda[i];
        ok = k3c_w(&k->q, &t->q) && k3c_w(&k->k, &t->k) && k3c_w(&k->v, &t->v) && k3c_w(&k->g, &t->g) &&
             k3c_w(&k->o, &t->o) && k3c_f32(k->fa, D, c->kda_hd, &t->fa) && k3c_f32(k->fb, c->kda_hd, P, &t->fb) &&
             k3c_f32(k->bp, D, c->kda_heads, &t->bp);
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
    if (!ok) return 0;
    if (l->kda) {
        *why = "device memory for its state refused";
        ch->win[i] = vkc_buf((size_t)3 * P * c->conv_k * sizeof(float), VKC_DEV);
        ch->st[i] = vkc_buf((size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float), VKC_DEV);
        if (!ch->win[i] || !ch->st[i]) return 0;
    }
    return 1;
}
/* A device copy the chain holds for W w (*t): freed, and the W's own pointer to it with it
 * (the per-matrix path's copy, which the chain adopts). */
static void k3c_drop_w(W *w, ColiVkTensor **t) {
    if (*t && w && w->vk == (void *)*t) w->vk = NULL;
    coli_vk_tensor_free(*t); *t = NULL;
}
/* Everything of layer i off the device: the chain's tensors (and the W copies it adopted),
 * its state. The CPU multiplies its matrices from their host copies, which stay. */
static void k3c_layer_free(K3Chain *ch, Model *m, int i) {
    Layer *l = &m->L[i];
    if (vkc_ready()) vkc_finish();
    K3cKda *k = &ch->kda[i]; K3cFfn *f = &ch->ffn[i]; VkcMla *t = &ch->mla[i];
    if (l->kda) {
        k3c_drop_w(&l->a.q, &k->q); k3c_drop_w(&l->a.k, &k->k); k3c_drop_w(&l->a.v, &k->v);
        k3c_drop_w(&l->a.g, &k->g); k3c_drop_w(&l->a.o, &k->o);
        k3c_drop_w(NULL, &k->fa); k3c_drop_w(NULL, &k->fb); k3c_drop_w(NULL, &k->bp);
    } else {
        k3c_drop_w(&l->m.qa, &t->q_a); k3c_drop_w(&l->m.qb, &t->q_b); k3c_drop_w(&l->m.kva, &t->kv_a);
        k3c_drop_w(&l->m.kvb, &t->kv_b); k3c_drop_w(&l->m.o, &t->o); k3c_drop_w(&l->m.g, &ch->mg[i]);
    }
    if (l->sparse) {
        k3c_drop_w(NULL, &f->router); k3c_drop_w(&l->moe.lat_down, &f->down); k3c_drop_w(&l->moe.lat_up, &f->up);
        k3c_drop_w(&l->moe.sh_gate, &f->sg); k3c_drop_w(&l->moe.sh_up, &f->su); k3c_drop_w(&l->moe.sh_down, &f->sd);
    } else { k3c_drop_w(&l->d_gate, &f->dg); k3c_drop_w(&l->d_up, &f->du); k3c_drop_w(&l->d_down, &f->dd); }
    vkc_free(ch->win[i]); ch->win[i] = NULL;
    vkc_free(ch->st[i]); ch->st[i] = NULL;
}
/* Layer i did not reach the device: it (and any layer after it) off the device, the chain
 * cut before it; with the dense weights on the device only, the layers from i keep their
 * host copies (nothing of theirs was dropped yet: that comes after the setup). */
static void k3c_cut(K3Chain *ch, Model *m, int i, const char *why) {
    for (int k = i; k < ch->lo + ch->n; k++) k3c_layer_free(ch, m, k);
    if (ch->lo + ch->n > i) ch->n = i - ch->lo;
    if (ch->d) { vkc_fit_shrink("kimi_k3 dev2", &g_k3c_fit2, i - ch->lo, why); return; }
    if (g_k3_dho_layers > i) { g_k3_dho_layers = i; g_k3_dho_head = 0; coli_vk_dense_host_layers(i, m->c.n_layers); }
    vkc_fit_shrink("kimi_k3", &g_k3_fit, i, why);
    g_k3_partial = 1;
}

/* d = 1: the second device's chain, its layers from g_k3_fit.n (device 1 current). */
static int k3c_setup_dev(Model *m, int d) {
    const Cfg *c = &m->c;
    int L = c->n_layers, D = c->hidden, P = c->kda_proj, LT = c->latent;
    int nbmax = (L + c->res_bs - 1) / c->res_bs;
    const char *why = k3c_unsupported(m);
    if (why) { fprintf(stderr, "[VK] kimi_k3 chain: %s; the CPU runs the layers\n", why); return 0; }
    VkcFit *fit = d ? &g_k3c_fit2 : &g_k3_fit;
    const char *nmc = d ? "kimi_k3 dev2" : "kimi_k3";
    int N = d ? g_k3c_fit2.n : g_k3_fit.L > 0 ? g_k3_fit.n : L, lo = d ? g_k3_fit.n : 0;
    if (N < 1) return 0;
    K3Chain *ch = calloc(1, sizeof *ch);
    if (!ch) return 0;
    ch->d = d; ch->lo = lo;
    size_t **offs[] = {&ch->o_in, &ch->o_post, &ch->o_asw, &ch->o_msw, &ch->o_qn, &ch->o_kn, &ch->o_conv, &ch->o_kda, &ch->o_latn};
    for (size_t k = 0; k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = calloc(L, sizeof(size_t)))) return 0;
    ch->kda = calloc(L, sizeof(K3cKda)); ch->ffn = calloc(L, sizeof(K3cFfn)); ch->mla = calloc(L, sizeof(VkcMla));
    ch->mg = calloc(L, sizeof(void *)); ch->kv = calloc(L, sizeof(VkcMlaCache));
    ch->win = calloc(L, sizeof(void *)); ch->st = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->mla_ord = calloc(L, sizeof(int));
    if (!ch->kda || !ch->ffn || !ch->mla || !ch->mg || !ch->kv || !ch->win || !ch->st || !ch->kv_valid || !ch->mla_ord) return 0;
    ch->m = m; ch->nbmax = nbmax; ch->n = N;
    if (d) g_k3c2 = ch; else g_k3c = ch;
    /* the parameter arena: offsets for the chain's layers and the output mix, then one upload */
    size_t n = 0;
    for (int i = lo; i < lo + N; i++) {
        const Layer *l = &m->L[i];
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D; ch->o_asw[i] = n; n += D; ch->o_msw[i] = n; n += D;
        if (l->kda) {
            ch->o_conv[i] = n; n += (size_t)3 * P * c->conv_k;
            ch->o_kda[i] = n; n += (size_t)c->kda_heads + P + c->kda_hd;
        } else {
            ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += c->kv_lora;
        }
        if (l->sparse) { ch->o_latn[i] = n; n += LT; }
    }
    ch->o_osw = n; n += D; ch->o_final = n; n += D;
    if (!(ch->prm = vkc_buf(n * sizeof(float), VKC_DEV))) {
        k3c_cut(ch, m, lo, vkc_lost() ? "the device was lost" : "device memory for the parameters refused");
        vkc_fit_placed(nmc, fit);
        return 0;
    }
    /* the layers, each whole or the chain stops before it */
    g_k3c_dev = d;
    for (int i = lo; i < lo + N; i++) {
        if (!k3c_layer_up(ch, m, i, &why)) { k3c_cut(ch, m, i, vkc_lost() ? "the device was lost" : why); break; }
        vkc_fit_mark(fit, i - lo);
    }
    g_k3c_dev = 0;
    N = ch->n;
    for (int i = lo; i < lo + N; i++) if (!m->L[i].kda) ch->mla_ord[i] = ch->n_mla++;
    float *a = N > 0 ? calloc(n, sizeof(float)) : NULL;
    int ok = a != NULL;
    for (int i = lo; ok && i < lo + N; i++) {
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
    if (ok) {
        memcpy(a + ch->o_osw, m->out_sw, D * sizeof(float));
        memcpy(a + ch->o_final, m->final_norm, D * sizeof(float));
    }
    ok = ok && vkc_begin() && vkc_write(ch->prm, 0, a, n * sizeof(float)) && vkc_submit(1);
    free(a);
    if (!ok) {
        if (N > 0) {
            if (!d) fprintf(stderr, "[VK] kimi_k3 chain: the parameters did not reach the device; the CPU runs the layers\n");
            k3c_cut(ch, m, lo, vkc_lost() ? "the device was lost" : "the parameters did not reach it");
        }
        vkc_free(ch->prm); ch->prm = NULL;
        vkc_fit_placed(nmc, fit);
        return 0;
    }
    vkc_fit_placed(nmc, fit);   /* the layers' matrices only, before the head */
    /* the head goes up with every layer and the tail (the fit's; without a fit, as before);
     * if the device refuses it the CPU multiplies it from the chain's final rows (the
     * second device's: the host's always) */
    if (!d && N == L && (g_k3_fit.L < 1 || g_k3_fit.tail)) {
        if (g_k3_dho && g_k3_fit.L > 0) w_vk_upload(&m->lm_head);
        if (!k3c_w(&m->lm_head, &ch->head)) {
            if (g_k3_fit.L < 1) {
                fprintf(stderr, "[VK] kimi_k3 chain: a matrix did not reach the device (or has no device form); the CPU runs the layers\n");
                return 0;
            }
            fprintf(stderr, "[VK] kimi_k3 chain: the head did not reach the device; the CPU multiplies it\n");
            g_k3_fit.tail = 0; g_k3_partial = 1; g_k3_dho_head = 0;
        }
    }
    ch->where = K3C_HOST; ch->host_zero = 1;   /* the host's KDA state is the zeros it was allocated with */
    ch->ok = 1;
    int nkda = 0, nsparse = 0; for (int i = lo; i < lo + N; i++) { nkda += m->L[i].kda; nsparse += m->L[i].sparse; }
    size_t bytes = 0, tensors = 0;
    coli_vk_mem_info_dev(d, &bytes, &tensors);
    if (d) fprintf(stderr, "[VK] kimi_k3 chain: layers %d..%d on the second device (%d KDA, %d MLA, %d dense MLP), %d AttnRes blocks, "
                           "%.1f MiB of parameters, %zu matrices (%.1f MiB) there\n", lo, lo + N - 1, nkda, ch->n_mla, N - nsparse,
                   nbmax, n * 4 / 1048576.0, tensors, bytes / 1048576.0);
    else fprintf(stderr, "[VK] kimi_k3 chain: %d layers on the device (%d KDA, %d MLA, %d dense MLP), %d AttnRes blocks, "
                         "%.1f MiB of parameters, %zu matrices (%.1f MiB) on the device\n", N, nkda, ch->n_mla, N - nsparse, nbmax,
                 n * 4 / 1048576.0, tensors, bytes / 1048576.0);
    return 1;
}
static int k3c_setup(Model *m) { return k3c_setup_dev(m, 0); }

/* scratch for `rows` rows (grows, never shrinks) */
static int k3c_scratch(K3Chain *ch, const Model *m, int rows) {
    const Cfg *c = &m->c;
    int D = c->hidden, P = c->kda_proj, H = c->kda_heads, LT = c->latent, E = c->n_experts, R = c->qk_rope;
    int SI = c->moe_inter * c->n_shared, MI = SI > c->dense_inter ? SI : c->dense_inter;
    int mfull = -1; for (int i = ch->lo; i < ch->lo + ch->n; i++) if (!m->L[i].kda) { mfull = i; break; }   /* the chain's layers */
    size_t r = (size_t)rows;
    ch->kvd_layer = r * (c->kv_lora + R);
    int ok = (mfull < 0 || g_k3c_count >= 0 || vkc_mla_scratch(&ch->sc, &ch->mla[mfull], rows)) &&
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
    if (g_k3c_count >= 0) {   /* count the split without allocating or changing capacities */
        if (ch->ks.on && mfull >= 0)
            g_k3c_count += (long long)r * (3LL * c->n_heads * (c->kv_lora + 2) +
                              (long long)c->n_heads * (c->kv_lora + R) + c->kv_lora + R) * sizeof(float);
        return ok && (R <= 0 || k3c_res(&ch->cs, r * R, VKC_UP));
    }
    if (ok && R > 0) {   /* NoPE: the rope parts pass through a rotation by angle 0 */
        size_t had = vkc_bytes(ch->cs) / sizeof(float);
        ok = k3c_res(&ch->cs, r * R, VKC_UP);
        if (ok && vkc_bytes(ch->cs) / sizeof(float) != had) {
            float *cs = (float *)vkc_ptr(ch->cs);
            size_t nf = vkc_bytes(ch->cs) / sizeof(float);
            for (size_t i = 0; i + 1 < nf; i += 2) { cs[i] = 1.f; cs[i + 1] = 0.f; }
        }
    }
    if (ok && ch->ks.on && mfull >= 0)   /* the split: the partial results, a step's new rows before their slots */
        ok = vkc_kv_parts(&ch->ks, r * c->n_heads * (c->kv_lora + 2)) &&
             (ch->kvtmp.cap >= rows || (k3c_res(&ch->kvtmp.lat, r * c->kv_lora, VKC_DEV) &&
                                        (R == 0 || k3c_res(&ch->kvtmp.rope, r * R, VKC_DEV)) && (ch->kvtmp.cap = rows)));
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

/* The MLA mirror at the host's capacity (m->max_t), filled again when that grows, and
 * planned again then (vkc_kv_plan): whole when it fits the device's budget, else the
 * split, for steps of `rows` rows. */
static int k3c_mirror(K3Chain *ch, const Model *m, int rows) {
    const Cfg *c = &m->c;
    if (ch->cap == m->max_t) return 1;
    size_t row = (size_t)(c->kv_lora + c->qk_rope) * sizeof(float);
    if (ch->n_mla && !vkc_kv_plan(&ch->ks, k3c_name(ch), ch->n_mla, row, m->max_t, rows, 0, (size_t)ch->n_mla * ch->dev_rows * row))
        return 0;
    int dr = ch->ks.on ? ch->ks.rows : m->max_t;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        if (m->L[i].kda) continue;
        vkc_free(ch->kv[i].lat); vkc_free(ch->kv[i].rope);
        ch->kv[i] = (VkcMlaCache){NULL, NULL, 0};
        ch->kv_valid[i] = 0;
    }
    ch->cap = 0; ch->dev_rows = 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        if (m->L[i].kda) continue;
        ch->kv[i].lat = vkc_buf((size_t)dr * c->kv_lora * sizeof(float), VKC_DEV);
        if (c->qk_rope > 0) ch->kv[i].rope = vkc_buf((size_t)dr * c->qk_rope * sizeof(float), VKC_DEV);
        ch->kv[i].cap = dr;
        if (!ch->kv[i].lat || (c->qk_rope > 0 && !ch->kv[i].rope)) return 0;
    }
    ch->cap = m->max_t; ch->dev_rows = dr;
    return 1;
}

/* Record the uploads that make the device the host's below pb, for a step of n_rows
 * rows (the split: its window placed first). */
static int k3c_push(K3Chain *ch, const Model *m, int pb, int n_rows) {
    const Cfg *c = &m->c; int ok = 1, K = c->kv_lora, R = c->qk_rope;
    if (ch->where == K3C_HOST) {
        size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)c->kda_proj * c->conv_k;
        for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
            if (!m->L[i].kda) continue;
            if (ch->host_zero) ok = vkc_zero(ch->st[i], 0, ns) && vkc_zero(ch->win[i], 0, 3 * nw);
            else ok = vkc_write(ch->st[i], 0, m->kstate[i], ns * sizeof(float)) &&
                      vkc_write(ch->win[i], 0, m->cwq[i], nw * sizeof(float)) &&
                      vkc_write(ch->win[i], nw, m->cwk[i], nw * sizeof(float)) &&
                      vkc_write(ch->win[i], 2 * nw, m->cwv[i], nw * sizeof(float));
        }
        ch->where = K3C_BOTH; ch->host_pos = pb;
    }
    for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
        if (!m->L[i].kda && ch->ks.on) {   /* the split: the window placed, its rows below pb uploaded */
            VkcKvPart pt[2] = {{1, K, m->Lc[i], 0, ch->kv[i].lat, 0}, {1, R, m->Rc[i], 0, ch->kv[i].rope, 0}};
            vkc_kv_place(&ch->ks, ch->mla_ord[i], pb, n_rows);
            ok = vkc_kv_push(&ch->ks, ch->mla_ord[i], pt, R ? 2 : 1, pb);
            ch->kv_valid[i] = pb;
            continue;
        }
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
    if (ch->ks.on)   /* the split: the new rows into their slots, the core in two parts */
        return vkc_kv_mla_qkv(&ch->ks, ch->mla_ord[i], &ch->mla[i], &ch->sc, ch->nrm, 0, n, pb, m->c.qk_rope > 0 ? ch->cs : NULL,
                              &ch->kv[i], &ch->kvtmp, ch->kvd, ko) &&
               vkc_matmul(ch->mg[i], ch->nrm, 0, ch->gate, 0, n) &&
               vkc_kv_mla_attn(&ch->ks, ch->mla_ord[i], &ch->mla[i], &ch->sc, n, pb, 0, &ch->kv[i], NULL, 0, 0, ch->gate, 0,
                               ch->att, 0, m->Lc[i], m->Rc[i]);
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

/* The prompt block the engine hands the chain: its chunk, when the chain runs with a
 * chunk from the budget (vkc_chunk_auto) and K3_CHUNK is not set, the smaller of the two
 * chains'; 0 = K3_CHUNK's (32). */
static int k3c_prefill_rows(const Model *m) {
    K3Chain *ch = g_k3c, *ch2 = g_k3c2;
    if (!g_k3c_on || !ch || !ch->ok || ch->failed || ch->m != m || getenv("K3_CHUNK") || !vkc_chunk_auto()) return 0;
    if (m->trace || g_k3_val_layer >= 0 || g_k3_dfp) return 0;
    if (!k3c_mirror(ch, m, 1)) return 0;
    int rows = k3c_chunk_rows(ch, m);
    if (ch->ks.on && rows > ch->ks.chunk) rows = ch->ks.chunk;
    if (ch2 && ch2->ok && !ch2->failed) {
        vkc_device(1);
        if (k3c_mirror(ch2, m, 1)) {
            int r2 = k3c_chunk_rows(ch2, m);
            if (ch2->ks.on && r2 > ch2->ks.chunk) r2 = ch2->ks.chunk;
            if (r2 < rows) rows = r2;
        }
        vkc_device(0);
    }
    return rows;
}
/* One chain's layers (lo..lo+n-1, on its device, current) for C rows (positions
 * pos0..), from each row's AttnRes state as it comes in: the prefix in `hidden`, *nb
 * block snapshots in `bres` (none at layer 0). Its last layer the model's last: the
 * output mix and the final norm, the rows into *fin (when need_rows, or when the head
 * stays on the host), the last row's logits into *last (the head on the device). Else
 * each row's AttnRes state after its last layer back into hidden, bres and *nb. Returns
 * ch->n; 0: not taken (*lost = 1: a frame failed, the device marked lost; else memory
 * refused), nothing the caller holds changed. On a cancel (poll), ch->n with *cancelled
 * set: the state is the caller's to reset. */
static int k3c_forward_seg(Model *m, K3Chain *ch, float *hidden, float *bres, int *nb_io, int pos0, int C, int need_rows,
                           float **fin_out, float **last_out, K3CancelPoll poll, void *pctx, int *cancelled, int *lost) {
    *lost = 0;
    const Cfg *c = &m->c; int L = c->n_layers, N = ch->n, lo = ch->lo, hi = lo + N, D = c->hidden, E = c->n_experts, LT = c->latent;
    int part = hi < L, nbx = ch->nbmax, nb_in = *nb_io;
    if (!part && !ch->head) need_rows = 1;   /* the head on the CPU: it reads the final rows */
    int mirror_ok = k3c_mirror(ch, m, 1);
    int CH = mirror_ok ? k3c_chunk_rows(ch, m) : 1, rows = C < CH ? C : CH;
    if (ch->ks.on && rows > ch->ks.chunk) rows = ch->ks.chunk;   /* a step's rows fit the split's window */
    if (!mirror_ok || !k3c_scratch(ch, m, rows) ||
        (part && !(k3c_res(&ch->xo, (size_t)rows * D, VKC_DOWN) && k3c_res(&ch->bo, (size_t)rows * nbx * D, VKC_DOWN)))) {
        if (vkc_lost()) { *lost = 1; return 0; }   /* lost while its buffers were made */
        fprintf(stderr, "[VK] %s chain: device memory for %d rows at %d positions refused; the CPU runs the layers\n",
                k3c_name(ch), rows, m->max_t);
        return 0;
    }
    float *fin = need_rows && !part ? falloc((int64_t)C * D) : NULL, *last = !part && ch->head ? falloc(c->vocab) : NULL;
    /* a partial chain's handoff, written to the caller's buffers once every chunk is through
     * (a lost device reruns the forward on the CPU from its untouched input) */
    float *xh = part ? falloc((int64_t)C * D) : NULL, *bh = part ? falloc((int64_t)C * nbx * D) : NULL;
    int nb_end = 0;
    vkc_gemm_rows(-1);
    for (int c0 = 0; c0 < C; c0 += rows) {
        int n = C - c0 < rows ? C - c0 : rows, pb = pos0 + c0, end = c0 + n == C;
        if (!vkc_begin() || !k3c_push(ch, m, pb, n) || !vkc_write(ch->x, 0, hidden + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            (nb_in > 0 && !vkc_write(ch->bres, 0, bres + (size_t)c0 * nbx * D, (size_t)n * nbx * D * sizeof(float))))
            goto lost;
        int ok = 1, pending = 0, pulled = lo, nb = nb_in;
        for (int i = lo; i < hi && ok; i++) {
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
                 vkc_submit(0);                                    /* A1 */
            /* while it runs, the tier loads the experts this layer will likely stream (a
             * big prompt chunk only) */
            if (ok) vkt_stream_prefetch(i, n);
            ok = ok && vkc_finish();
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
                free(fin); free(last); free(xh); free(bh);
                return N;
            }
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok && pending) ok = k3c_join(ch, m, hi - 1, n);
        nb_end = nb;
        if (part)   /* the handoff: each row's prefix and its block snapshots after the chain's last layer */
            ok = ok && vkc_copy(ch->xo, 0, ch->x, 0, (size_t)n * D) && vkc_copy(ch->bo, 0, ch->bres, 0, (size_t)n * nbx * D);
        else {
            /* the output mix and the final norm of every row, lm_head on the last */
            ok = ok && k3c_mix(ch, m, ch->o_osw, n, nb) && k3c_norm(ch->hm, ch->prm, ch->o_final, ch->fin, n, D, c->eps);
            if (ok && need_rows) ok = vkc_copy(ch->find, 0, ch->fin, 0, (size_t)n * D);
            if (ok && end && ch->head) ok = vkc_matmul(ch->head, ch->fin, (size_t)(n - 1) * D, ch->outd, 0, 1);
        }
        ok = ok && vkc_submit(1);
        if (!ok) goto lost;
        k3c_pull(ch, m, pulled, hi, pb, n);
        if (part) {
            memcpy(xh + (size_t)c0 * D, vkc_ptr(ch->xo), (size_t)n * D * sizeof(float));
            memcpy(bh + (size_t)c0 * nbx * D, vkc_ptr(ch->bo), (size_t)n * nbx * D * sizeof(float));
        }
        if (need_rows && fin) memcpy(fin + (size_t)c0 * D, vkc_ptr(ch->find), (size_t)n * D * sizeof(float));
        for (int i = lo; i < hi; i++) if (!m->L[i].kda) { ch->kv_valid[i] = pb + n; vkc_kv_done(&ch->ks, ch->mla_ord[i], pb + n); }
        ch->where = K3C_DEV;
        if (end && last) memcpy(last, vkc_ptr(ch->outd), (size_t)c->vocab * sizeof(float));
    }
    ch->dev_pos = pos0 + C;
    ch->forwards++;
    if (part) {
        memcpy(hidden, xh, (size_t)C * D * sizeof(float));
        memcpy(bres, bh, (size_t)C * nbx * D * sizeof(float));
        *nb_io = nb_end;
        free(xh); free(bh);
    }
    *fin_out = fin; *last_out = last;
    return N;
lost:   /* a frame failed: the device is gone (or would not take a command) */
    free(fin); free(last); free(xh); free(bh);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(ch->d); }
    *lost = 1;
    return 0;
}
/* The devices' layers (every layer, or a partial chain's first ch->n, and the second
 * device's after them) for C rows of `hidden` (positions pos0..), as k3c_forward_seg
 * describes, from layer 0. Returns the layers that ran; 0: not taken (the CPU runs every
 * layer; nothing the caller holds changed). */
static int k3c_forward(Model *m, float *hidden, float *bres, int *nb_out, int pos0, int C, int need_rows, float **fin_out,
                       float **last_out, K3CancelPoll poll, void *pctx, int *cancelled) {
    K3Chain *ch = g_k3c, *ch2 = g_k3c2;
    if (!g_k3c_on || !ch || !ch->ok || ch->failed || ch->m != m) return 0;
    if (g_k3c_on == COLI_VK_CHAIN_PREFILL && C <= 2) return 0;   /* prompts only: decode on the CPU */
    if (m->trace || g_k3_val_layer >= 0 || g_k3_dfp) {
        static int said = 0;
        if (!said++) fprintf(stderr, "[VK] kimi_k3 chain: a validation dump reads every layer on the host; the CPU runs the layers\n");
        return 0;
    }
    if (ch2 && (!ch2->ok || ch2->failed || ch2->m != m || ch2->lo != ch->lo + ch->n)) ch2 = NULL;
    int D = m->c.hidden;
    int dev0 = ch->where == K3C_DEV, dev2 = ch2 && ch2->where == K3C_DEV, dev_start = ch->dev_pos, lost2 = 0;
    if (ch2) { vkc_device(1); lost2 = vkc_lost(); vkc_device(0); }
    if (vkc_lost() || lost2) { k3c_recover(m, dev0, dev2, dev_start, lost2 ? ch2 : ch); return 0; }
    /* the rows as they came in: the primary's handoff replaces them, and if the second
     * device is lost the CPU runs the forward again from these */
    float *keep = ch2 ? falloc((int64_t)C * D) : NULL;
    if (keep) memcpy(keep, hidden, (size_t)C * D * sizeof(float));
    int lost = 0, nb = 0;
    float *fin = NULL, *last = NULL;
    int r = k3c_forward_seg(m, ch, hidden, bres, &nb, pos0, C, need_rows, &fin, &last, poll, pctx, cancelled, &lost);
    if (!r) {
        free(keep);
        if (lost) {
            ch->where = dev0 ? K3C_DEV : K3C_HOST;   /* the state as it was when this forward began */
            k3c_recover(m, dev0, dev2, dev_start, ch);
            return 0;
        }
        k3c_cpu_step(m, pos0);   /* memory refused: the CPU from here on, the state current first */
        ch->failed = 1; g_k3c_on = 0;
        return 0;
    }
    if (!ch2 || (cancelled && *cancelled)) { free(keep); *nb_out = nb; *fin_out = fin; *last_out = last; return r; }
    vkc_device(1);
    int r2 = k3c_forward_seg(m, ch2, hidden, bres, &nb, pos0, C, need_rows, &fin, &last, poll, pctx, cancelled, &lost);
    vkc_device(0);
    if (lost) {
        memcpy(hidden, keep, (size_t)C * D * sizeof(float));
        free(keep);
        ch->where = dev0 ? K3C_DEV : K3C_HOST; ch2->where = dev2 ? K3C_DEV : K3C_HOST;
        k3c_recover(m, dev0, dev2, dev_start, ch2);
        return 0;
    }
    if (!r2) {   /* memory refused there: its layers on the CPU from this step on, their state current first */
        int reads = 0;
        if (ch2->where == K3C_DEV && !k3c_sync_one(m, ch2, &reads)) {   /* lost on the way: the forward again, from its input */
            memcpy(hidden, keep, (size_t)C * D * sizeof(float));
            free(keep);
            if (reads) { ch->host_pos = 0; ch->host_zero = 1; }
            ch->where = dev0 ? K3C_DEV : K3C_HOST;
            k3c_recover(m, dev0, 1, dev_start, ch2);
            return 0;
        }
        free(keep);
        k3c_cpu_one(m, ch2, pos0);
        ch2->failed = 1;
        *nb_out = nb;
        return r;
    }
    free(keep);
    *nb_out = nb; *fin_out = fin; *last_out = last;
    return r + r2;
}

static void k3c_report(void) {
    for (int d = 0; d < 2; d++) {
        K3Chain *ch = k3c_of(d);
        if (!ch || !ch->ok || !ch->forwards) continue;
        int was = vkc_device(d);
        VkcStats st; vkc_stats(&st);
        fprintf(stderr, "[VK] %s chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                        "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
                k3c_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms,
                st.dev_bytes / 1048576.0);
        vkc_kv_report(&ch->ks);
        vkc_prof_print();
        vkc_device(was);
    }
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
        if (!no && g_k3_mux_slots > 1) no = "several conversations at once (KV_SLOTS): it keeps one conversation's state on the device";
        if (!no && g_k3_fit.L > 0 && g_k3_fit.n < 1) {   /* the fit's line said so: nothing of the chain on the device */
            vkc_fit_placed("kimi_k3", &g_k3_fit);
            return;
        }
        if (!no && !(g_k3c_inited = vkc_init())) no = "the chain's pipelines did not come up";
        if (!no && !(vkc_mla_ready() && vkc_kda_ready() && vkc_ares_ready())) no = "the MLA, KDA or AttnRes shaders are missing";
    }
    if (no) fprintf(stderr, "[VK] kimi_k3: %s: the dense chain stays off\n", no);
    if (!on || no) return;
    int n0 = g_k3_fit.n;
    if (!k3c_setup(m)) return;
    g_k3c_on = on;
    if (g_k3c_fit2_on) {
        vkc_device(1);
        if (g_k3c->n < n0) {
            /* the primary placed fewer layers than its fit: the second device's would not
             * follow them, so they stay on the CPU too */
            fprintf(stderr, "[VK] kimi_k3 chain: the primary device stopped before layer %d; layers %d..%d stay on the CPU, "
                            "not on the second device\n", n0, n0, n0 + g_k3c_fit2.n - 1);
            g_k3c_fit2_on = 0;
            vkc_shutdown();
        } else if (!k3c_setup_dev(m, 1)) { g_k3c_fit2_on = 0; vkc_shutdown(); }
        vkc_device(0);
    }
}
/* After the tier's: at exit the chain goes before the device. */
static void k3c_atexit(void) {
    if (g_k3c_inited) atexit(vkc_shutdown_all);
}
/* The per-matrix path is about to put the shared experts on the device (no tier after
 * all): the chain's copies become its own. */
static void k3c_adopt_shared(Model *m) {
    K3Chain *ch = g_k3c;
    if (!ch || !ch->ok || ch->m != m) return;
    for (int i = 0; i < ch->n; i++) {
        if (!m->L[i].sparse) continue;
        Moe *o = &m->L[i].moe; K3cFfn *f = &ch->ffn[i];
        if (!o->sh_gate.vk) o->sh_gate.vk = f->sg;
        if (!o->sh_up.vk) o->sh_up.vk = f->su;
        if (!o->sh_down.vk) o->sh_down.vk = f->sd;
    }
}
