/* qwen36_chain.h -- Qwen3.6's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by qwen36.c in a COLI_VULKAN build, after the CPU forward it stands in
 * for; COLI_VK_CHAIN decides (backend_vulkan.c, coli_vk_chain_decide).
 *
 * What runs where, per layer, for one block of rows (decode: one row):
 *   device, frame A1: the routed MoE output of the layer before joins the residual
 *     (x += routed + gate * shared, chain_ew COMBINE, the CPU's order), the input
 *     RMSNorm, then either the gated attention (q/k/v, per-head q/k norm, RoPE from a
 *     host table, the K/V rows into the device cache, attention with the output gate,
 *     o_proj) or the Gated DeltaNet (qkv/z/b/a, the convolution with its ring, the
 *     recurrence with its state, the gated norm, out_proj), the residual add, the
 *     post-attention RMSNorm and the router logits. Then the frame is waited for.
 *   host: the router's softmax and top-k, the routed experts (the Vulkan expert tier's
 *     device batch and the CPU's share, joined in rank order: moe_ex routed_only), the
 *     K/V rows copied into the host cache.
 *   device, frame A2 (not waited for): the shared expert and its gate, while the host
 *     computes the routed experts.
 * Crossing per layer: the normalized rows (D floats per row) and the router logits
 * (E per row) down, the K/V rows of an attention layer down, the routed sum (D per
 * row) up. A dense model (no routed experts) records the whole forward in one frame.
 * The last frame normalizes the last row and runs lm_head; its logits come back.
 *
 * State and who owns it:
 *   - the residual stream: on the device for the whole forward;
 *   - the attention KV cache: the host's stays canonical (every chain step copies its
 *     new rows back), the device holds a mirror with a watermark per layer, kv_valid:
 *     rows [0, kv_valid) equal the host's. A step that starts at pos_base uploads the
 *     rows [kv_valid, pos_base) first; a CPU step lowers the watermark to its
 *     pos_base; a cache that grows (ensure_kv) is mirrored again from the host;
 *   - the DeltaNet recurrent state and conv ring: on the device while the chain runs
 *     (60 MB on the 35B: too much to copy per token). dn_where says which side holds
 *     the newest copy; the host one is brought back before anything reads it there
 *     (a CPU step, a pinned snapshot) and pushed up after anything writes it there
 *     (reset_recurrent: a fill with zeros on the device, pin_restore: an upload).
 * Prompt-cache and prefix reuse need nothing more: a reused prefix is rows below the
 * watermark and a DeltaNet state that already sits where the next step expects it.
 * A speculative verify (prompt lookup, q36_spec_step) runs here too: its matrices take
 * the per-row GEMV (a decode step's bits), lm_head reads each of its rows, and the
 * DeltaNet state and conv rings are copied on the device after each of its rows but the
 * last, one slot per row (the recurrent ops split into one dispatch per copy, each
 * ending on its row: the same bits), so a draft rejected after row r rolls back by
 * swapping slot r's buffers in (q36c_rollback).
 *
 * Past the device's budget (vk_kvsplit.h) each attention layer keeps only a window of
 * the newest positions on the device; the step's attention runs over that window on
 * the device and over the older positions on the CPU (from the host's canonical cache)
 * at once, the two merged through their softmax statistics. Below the budget the
 * mirrors are whole, as above.
 *
 * The chain declines (the per-matrix path runs, state synced first) under the CUDA
 * expert tier, a qpack container, PILOT prefetch, or geometry outside its shaders
 * (head dim > 256, DeltaNet value head > 128, key head > 256, conv kernel > 9).
 * A device lost while the chain holds the recurrent state: the engine rebuilds that
 * state on the CPU from the prefix record (the ids it was built from; the KV rows are
 * the host's already), stays on the CPU, and the step runs there. Only a state that
 * the ids do not describe (an image) cannot be rebuilt: that stops the engine.
 * COLI_VK_CHAIN_FAULT=n (vk_chain.c) fakes the loss at the n-th frame, for tests.
 *
 * A partial chain (vk_chain.h, vkc_fit; docs/vulkan.md, "A partial chain"): when the
 * layers do not all fit the device, the chain takes layers 0..N-1 and the CPU the rest
 * and the head. q36c_start decides N at start, before any upload and before the tier
 * sizes its budget; q36c_setup then places the N layers there and then, one layer at a
 * time (a layer that fails is freed whole and N stops before it), and with the dense
 * weights on the device only drops each layer's host copies once the layer is there.
 * The matrices of the CPU's layers (and of the head, unless the fit puts it up) are
 * marked vk_off: the per-matrix path never uploads them. A forward runs the N layers
 * on the device chunk by chunk as above, brings the residual rows after layer N-1 back
 * to the host once per chunk, and returns; the caller runs layers N..L-1 over all the
 * rows on the CPU (layer by layer, as without the chain) and the head. The device's
 * layers keep their state on the device (KV mirrors, the DeltaNet state and its verify
 * copies), the CPU's layers on the host (their KV rows, DeltaNet state and the CPU's
 * verify snapshots); every loop over layers here runs over the N. A device lost
 * mid-forward rebuilds the N layers' recurrent state from the prefix record: the CPU's
 * layers have not run that forward yet, so theirs is already the host's. */
#include "vk_chain.h"
#include "vk_kvsplit.h"

#define Q36C_HOST 0
#define Q36C_DEV  1
#define Q36C_BOTH 2

typedef struct {
    int ok, failed;
    int lo, d;                                 /* its layers start at lo, on device d (vkc_device) */
    int n, head;                               /* layers lo..lo+n-1 on the device; lm_head there too */
    int rows;                                  /* scratch capacity in rows */
    int cap;                                   /* the host's kv_cap the mirrors follow */
    int dev_rows;                              /* device KV rows a layer: cap, or the split's */
    VkcKvSplit ks;                             /* the split past the device's budget (ks.on) */
    VkcBuf *prm;                               /* every norm / conv / DeltaNet parameter */
    size_t *o_in, *o_post, *o_qn, *o_kn, *o_conv, *o_dn, o_final;
    ColiVkTensor **t_ab, **t_sg;               /* DeltaNet b|a rows, shared gate row (f32) */
    VkcBuf **rec, **ring, **kc, **vc;          /* per layer state */
    VkcBuf **rec_snap[Q36_SPEC_SNAPS], **ring_snap[Q36_SPEC_SNAPS];   /* a verify's copies, slot r after row r */
    int snap_slots, snap_valid;                /* allocated; written by the last verify */
    int *kv_valid, *attn_ord, n_attn;
    int dn_where, host_zero;
    VkcBuf *x, *nrm, *tmp, *q, *k, *v, *ctx, *qkv, *z, *ab, *cv, *dny, *h2, *lg, *gs, *us, *hs, *ds, *sgd, *fin;
    VkcBuf *h2d, *lgd, *kvd, *outd, *xd, *lfd, *routed, *cs;
    float *host_routed;
    /* the routed experts of a decode step on the device (q36c_moe_*): the layers' tables of
     * resident experts' entries, the step's chosen entries, weights and ids, the items of
     * the grouped GEMV, its hidden and output rows, the raw logits for the bookkeeping */
    VkcBuf **mgu, **mdn, *egu, *edn, *wt, *ix, *itg, *itd, *amap, *use, *hid, *ys, *lga, *rdev;
    int moe_tab;   /* 1 built, -1 an expert the grouped GEMV cannot take */
    unsigned long long forwards, frames, host_ms_n, moe_steps;
    double wait_ms, host_ms;
} Q36Chain;

static int g_vk_chain = 0;     /* COLI_VK_CHAIN decided on, and the chain's pipelines are up */
static VkcFit g_q36c_fit;      /* the partial chain's fit (q36c_start) */
static int g_q36c_fit_on;      /* the fit ran: the chain sets itself up at start (q36c_place) */
/* The layers after the primary's on COLI_VK_DEV2's device (docs/vulkan.md, "Layers on two
 * devices"): a second chain from layer g_q36c_fit.n, its own fit on that device's memory.
 * A forward runs the primary's layers, brings the residual rows back and runs these;
 * the CPU runs what is left. */
static VkcFit g_q36c_fit2;
static int g_q36c_fit2_on;
static Q36Chain *q36c_of(Model *m, int d) { return (Q36Chain *)(d ? m->vkchain2 : m->vkchain); }
static const char *q36c_name(const Q36Chain *ch) { return ch && ch->d ? "qwen36 dev2" : "qwen36"; }
/* COLI_VK_CHAIN_DEV2=0: the second device keeps routed experts only */
static int q36c_dev2_wanted(void) { const char *e = getenv("COLI_VK_CHAIN_DEV2"); return !(e && *e == '0'); }

static void q36c_fatal(const char *what) {
    fprintf(stderr, "[VK] qwen36 chain: %s -- stopping (COLI_VK_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
/* The device was lost with the newest recurrent state on it. The host's KV rows are
 * canonical, the DeltaNet state is not: rebuild it on the CPU by running the `upto`
 * positions the prefix record names (a prefill on the CPU), and leave the chain off.
 * A partial chain rebuilds its N layers only: the CPU's layers hold their state on the
 * host already, at `upto` (a forward the device was lost in has not reached them). */
static void q36c_recover(Model *m, int upto) {
    Q36Chain *ch = (Q36Chain *)m->vkchain, *ch2 = (Q36Chain *)m->vkchain2;
    Cfg *c = &m->c; int D = c->hidden;
    int nd = ch && ch->ok ? ch->n : c->n_layers;
    if (ch && ch->ok && ch2 && ch2->ok) nd += ch2->n;   /* the second device's layers ran in that forward too */
    g_vk_chain = 0;
    for (int d = 0; d < 2; d++) {
        Q36Chain *x = q36c_of(m, d);
        if (x) { x->failed = 1; x->dn_where = Q36C_HOST; x->host_zero = 0; x->snap_valid = 0; }
    }
    for (int i = 0; i < nd; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        q36c_fatal("the device was lost with a recurrent state its token ids do not describe (an image)");
    fprintf(stderr, "[VK] qwen36 chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", upto);
    int *ids = malloc((size_t)upto * sizeof(int));
    float *x = falloc((int64_t)upto * D);
    if (!ids) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    memcpy(ids, m->kvp.fed, (size_t)upto * sizeof(int));
    for (int s = 0; s < upto; s++) memcpy(x + (int64_t)s * D, m->embed + (int64_t)ids[s] * D, D * sizeof(float));
    /* a rebuild, not the verify being run: no copies, the batch's own matmuls */
    int snap = m->snap_rows, rowwise = g_q36_rowwise;
    m->snap_rows = 0; g_q36_rowwise = 0;
    layers_forward_range(m, x, upto, 0, 0, nd, 0, NULL);
    m->snap_rows = snap; g_q36_rowwise = rowwise;
    free(ids); free(x);
}

/* f32 rows [O x I] as a resident fmt 10 tensor */
static ColiVkTensor *q36c_f32_tensor(int d, const float *w, int I, int O) {
    ColiVkTensor *t = NULL;
    return (d ? coli_vk_tensor_ensure2(&t, w, NULL, 10, I, O, 0) : coli_vk_tensor_ensure(&t, w, NULL, 10, I, O, 0)) ? t : NULL;
}

static int q36c_geometry_ok(const Cfg *c) {
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) {
            if (c->head_dim > 256 || c->head_dim != c->k_head_dim || c->q_heads % c->kv_heads ||
                c->q_head_dim < c->head_dim || (c->rotary_dim & 1) || c->rotary_dim > c->head_dim) return 0;
        } else {
            if (c->dn_vdim > 128 || c->dn_kdim > 256 || c->dn_convk < 2 || c->dn_convk > 9 ||
                c->dn_vheads % c->dn_kheads) return 0;
        }
    }
    return 1;
}

/* The matrices of layer i: every one load_tq may have loaded (the per-matrix path's),
 * and those the chain multiplies by, in its setup order (*chain of them first). */
static int q36c_layer_all(Model *m, int i, QW **ws) {
    Layer *l = &m->L[i];
    QW *all[] = {&l->q, &l->k, &l->v, &l->o, &l->gate, &l->sh_g, &l->sh_u, &l->sh_d, &l->dn_qkv, &l->dn_z, &l->dn_out};
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) ws[k] = all[k];
    return (int)(sizeof all / sizeof *all);
}
static int q36c_layer_qws(Model *m, int i, QW **ws) {
    Cfg *c = &m->c; Layer *l = &m->L[i]; int n = 0;
    if (c->is_attn[i]) { ws[n++] = &l->q; ws[n++] = &l->k; ws[n++] = &l->v; ws[n++] = &l->o; }
    else { ws[n++] = &l->dn_qkv; ws[n++] = &l->dn_z; ws[n++] = &l->dn_out; }
    if (c->n_experts > 0) ws[n++] = &l->gate;
    if (c->shared_inter > 0) { ws[n++] = &l->sh_g; ws[n++] = &l->sh_u; ws[n++] = &l->sh_d; }
    return n;
}

/* Layer i stays on the CPU for good: what the chain placed of it is freed (after every
 * frame that may read it) and its matrices are marked so the per-matrix path never
 * uploads them. Called for a layer that did not fully reach the device, before any of
 * its host copies was dropped. */
static void q36c_layer_cpu(Q36Chain *ch, Model *m, int i) {
    if (vkc_ready()) vkc_finish();
    QW *ws[16]; int nw = q36c_layer_all(m, i, ws);
    for (int k = 0; k < nw; k++) {
        QW *w = ws[k];
        if (w->vk) {
            const void *wq; const float *sc;
            int fmt = vk_qw_fmt(w, &wq, &sc);
            unsigned *pc = &g_vk_placed[w->vk_imported ? (fmt == 1 ? 0 : 3) : fmt == 1 ? 0 : fmt == 4 ? 1 : fmt == 14 ? 3 : 2];
            if (*pc) (*pc)--;
            coli_vk_tensor_free((ColiVkTensor *)w->vk);
            w->vk = NULL; w->vk_imported = 0;
        }
        w->vk_off = 1;
    }
    if (ch && ch->t_ab) {
        ColiVkTensor **t[] = {&ch->t_ab[i], &ch->t_sg[i]};
        VkcBuf **b[] = {&ch->rec[i], &ch->ring[i]};
        vkc_layer_free(t, 2, b, 2);
    }
}

/* A verify's DeltaNet copies written by the one convolution and recurrence dispatch,
 * after each row, into slots behind the state in its own buffer; a rejection copies its
 * slot back on the device (Q36_DN_SNAP1=0: a dispatch and a buffer a copy, as before) */
static int q36c_snap1(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("Q36_DN_SNAP1"); v = !(e && *e == '0'); }
    return v;
}
/* Layer i's matrices, its DeltaNet b|a rows and shared-expert gate row, and its state on
 * the device; 0 = something did not get there (the caller frees the layer). */
static int q36c_place_layer(Q36Chain *ch, Model *m, int i) {
    Cfg *c = &m->c; Layer *l = &m->L[i]; int D = c->hidden, vh = c->dn_vheads;
    QW *ws[16]; int nw = q36c_layer_qws(m, i, ws), ok = 1;
    g_q36_vk_dev = ch->d;
    for (int k = 0; k < nw && ok; k++) ok = vk_qw_tensor(ws[k]) != NULL;
    g_q36_vk_dev = 0;
    if (ok && !c->is_attn[i]) {
        float *ab = malloc((size_t)2 * vh * D * sizeof(float));
        if (ab) {
            memcpy(ab, l->dn_b, (size_t)vh * D * sizeof(float));
            memcpy(ab + (size_t)vh * D, l->dn_a, (size_t)vh * D * sizeof(float));
            ch->t_ab[i] = q36c_f32_tensor(ch->d, ab, D, 2 * vh);
            free(ab);
        }
        /* the state, then (q36c_snap1) a verify's copies after its rows 0.. in the same buffer */
        size_t ns1 = q36c_snap1() ? 1 + Q36_SPEC_SNAPS : 1;
        ch->rec[i] = ch->t_ab[i] ? vkc_buf(ns1 * vh * c->dn_kdim * c->dn_vdim * sizeof(float), VKC_DEV) : NULL;
        ch->ring[i] = ch->rec[i] ? vkc_buf(ns1 * c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float), VKC_DEV) : NULL;
        ok = ch->t_ab[i] && ch->rec[i] && ch->ring[i];
    }
    if (ok && l->sh_gate) ok = (ch->t_sg[i] = q36c_f32_tensor(ch->d, l->sh_gate, D, 1)) != NULL;
    return ok;
}

/* The chain on the device: the parameter arena, then layer after layer (each whole, its
 * host copies dropped once it is there when the dense weights live on the device only),
 * then the head when the fit puts it up. With the fit (q36c_start) it runs at start,
 * before the tier sizes its budget; a layer that does not reach the device is freed
 * whole and the chain keeps the layers before it (vkc_fit_shrink). NULL = the chain
 * cannot run. */
/* d = 1: the second device's chain, its layers from g_q36c_fit.n on (the caller has made
 * device d current, vkc_device). */
static Q36Chain *q36c_setup_dev(Model *m, int d) {
    void **slot = d ? &m->vkchain2 : &m->vkchain;
    Q36Chain *ch = (Q36Chain *)*slot;
    if (ch) return ch->ok ? ch : NULL;
    if (d && !g_q36c_fit2_on) return NULL;
    ch = (Q36Chain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    *slot = ch;
    ch->d = d;
    const char *nm = q36c_name(ch);
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden;
    if (!q36c_geometry_ok(c)) {
        fprintf(stderr, "[VK] qwen36 chain: a geometry its shaders do not take (head dim %d, DeltaNet %dx%d); per-matrix path\n",
                c->head_dim, c->dn_kdim, c->dn_vdim);
        return NULL;
    }
    VkcFit *fit = d ? &g_q36c_fit2 : g_q36c_fit_on ? &g_q36c_fit : NULL;
    int lo = d ? g_q36c_fit.n : 0, n = fit ? fit->n : L;
    ch->lo = lo;
    ch->o_in = calloc(L, sizeof(size_t)); ch->o_post = calloc(L, sizeof(size_t)); ch->o_qn = calloc(L, sizeof(size_t));
    ch->o_kn = calloc(L, sizeof(size_t)); ch->o_conv = calloc(L, sizeof(size_t)); ch->o_dn = calloc(L, sizeof(size_t));
    ch->t_ab = calloc(L, sizeof(void *)); ch->t_sg = calloc(L, sizeof(void *));
    ch->rec = calloc(L, sizeof(void *)); ch->ring = calloc(L, sizeof(void *));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->attn_ord = calloc(L, sizeof(int));
    if (!ch->o_in || !ch->o_post || !ch->o_qn || !ch->o_kn || !ch->o_conv || !ch->o_dn || !ch->t_ab || !ch->t_sg ||
        !ch->rec || !ch->ring || !ch->kc || !ch->vc || !ch->kv_valid || !ch->attn_ord) return NULL;
    /* the parameter arena of the n layers: offsets, then one upload */
    size_t np = 0;
    int vh = c->dn_vheads, conv_dim = c->dn_conv_dim, convk = c->dn_convk;
    for (int i = lo; i < lo + n; i++) {
        ch->o_in[i] = np; np += D; ch->o_post[i] = np; np += D;
        if (c->is_attn[i]) {
            if (m->L[i].qn) { ch->o_qn[i] = np; np += c->head_dim; }
            if (m->L[i].kn) { ch->o_kn[i] = np; np += c->k_head_dim; }
        } else {
            ch->o_conv[i] = np; np += (size_t)conv_dim * convk;
            ch->o_dn[i] = np; np += 2 * (size_t)vh + c->dn_vdim;
        }
    }
    ch->o_final = np; np += D;
    float *arena = calloc(np, sizeof(float));
    if (!arena) return NULL;
    for (int i = lo; i < lo + n; i++) {
        Layer *l = &m->L[i];
        memcpy(arena + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(arena + ch->o_post[i], l->post_ln, D * sizeof(float));
        if (c->is_attn[i]) {
            if (l->qn) memcpy(arena + ch->o_qn[i], l->qn, c->head_dim * sizeof(float));
            if (l->kn) memcpy(arena + ch->o_kn[i], l->kn, c->k_head_dim * sizeof(float));
        } else {
            memcpy(arena + ch->o_conv[i], l->dn_conv, (size_t)conv_dim * convk * sizeof(float));
            memcpy(arena + ch->o_dn[i], l->dn_alog, vh * sizeof(float));
            memcpy(arena + ch->o_dn[i] + vh, l->dn_dtbias, vh * sizeof(float));
            memcpy(arena + ch->o_dn[i] + 2 * vh, l->dn_norm, c->dn_vdim * sizeof(float));
        }
    }
    memcpy(arena + ch->o_final, m->final_norm, D * sizeof(float));
    ch->prm = vkc_buf(np * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, np * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) {
        if (!fit) return NULL;
        vkc_free(ch->prm); ch->prm = NULL;
        for (int i = lo; i < lo + n; i++) q36c_layer_cpu(ch, m, i);
        vkc_fit_shrink(nm, fit, 0, vkc_lost() ? "the device was lost" : "its parameter buffer was refused");
        if (!d || fit->tail) m->lm_head.vk_off = 1;
        if (!d) g_vk_chain = 0;   /* the second device's failing: the primary's chain stands, the CPU runs the rest */
        vkc_shutdown();   /* the chain's own blocks and frames go too */
        vkc_fit_placed(nm, fit);
        return NULL;
    }
    /* the layers: the same device copies the per-matrix path uses; a layer that does
     * not get there whole is freed and the chain stops before it */
    int placed = 0;
    for (int i = lo; i < lo + n; i++) {
        if (!q36c_place_layer(ch, m, i)) {
            if (!fit) { fprintf(stderr, "[VK] qwen36 chain: a matrix did not reach the device; per-matrix path\n"); return NULL; }
            for (int j = i; j < lo + n; j++) q36c_layer_cpu(ch, m, j);
            vkc_fit_shrink(nm, fit, i - lo, vkc_lost() ? "the device was lost" : "an upload was refused");
            break;
        }
        /* the layer is there whole: its host copies go (the second device's only where the
         * chain runs every step: with prompts only, decode reads them on the CPU) */
        if (coli_vk_dense_device_only() && (!d || g_vk_chain == COLI_VK_CHAIN_ON)) {
            QW *ws[16]; int nw = q36c_layer_qws(m, i, ws), nd = 0; size_t b = 0;
            for (int k = 0; k < nw; k++) q36_dho_drop(ws[k], &b, &nd);
        }
        if (fit) vkc_fit_mark(fit, i - lo);
        placed = i + 1 - lo;
    }
    ch->n = placed;
    for (int i = lo; i < lo + placed; i++) if (c->is_attn[i]) ch->attn_ord[i] = ch->n_attn++;
    if (!placed) {   /* nothing of the chain stays: its arena, blocks and frames go */
        vkc_free(ch->prm); ch->prm = NULL;
        if (!d || (fit && fit->tail)) m->lm_head.vk_off = 1;
        if (!d) g_vk_chain = 0;
        vkc_shutdown();
    }
    if (fit) vkc_fit_placed(nm, fit);
    if (!placed) return NULL;
    /* the head: with every layer, and room for it (the fit's tail), on the chain that
     * ends at the last layer */
    g_q36_vk_dev = d;
    ch->head = lo + placed == L && (!fit || fit->tail) && vk_qw_tensor(&m->lm_head);
    g_q36_vk_dev = 0;
    if (!ch->head) {
        if (fit) fit->tail = 0;
        /* the second device's chain may still take it (it is set up after this one) */
        if (!m->lm_head.vk && (d || !g_q36c_fit2_on || !g_q36c_fit2.tail)) m->lm_head.vk_off = 1;
    } else if (coli_vk_dense_device_only() && !d) { int nd = 0; size_t b = 0; q36_dho_drop(&m->lm_head, &b, &nd); }
    ch->dn_where = Q36C_HOST; ch->host_zero = 0;
    ch->ok = 1;
    if (d) fprintf(stderr, "[VK] qwen36 chain: layers %d..%d on the second device (%d attention)%s, %.1f MiB of parameters\n",
                   lo, lo + placed - 1, ch->n_attn, ch->head ? ", the head too" : "", np * 4 / 1048576.0);
    else fprintf(stderr, "[VK] qwen36 chain: %d layers on the device (%d attention)%s, %.1f MiB of parameters\n",
                 placed, ch->n_attn, ch->head ? "" : ", the head on the CPU", np * 4 / 1048576.0);
    return ch;
}
static Q36Chain *q36c_setup(Model *m) { return q36c_setup_dev(m, 0); }

/* q36c_res counts instead of reserving while g_q36c_count >= 0 (the chunk's sizing), and
 * adds each buffer's device bytes to *g_q36c_fitsum while it is set (the fit) */
static long long g_q36c_count = -1;
static size_t *g_q36c_fitsum;
static int q36c_res(VkcBuf **b, size_t floats, int kind) {
    if (g_q36c_fitsum) { *g_q36c_fitsum += vkc_fit_buf((floats ? floats : 1) * sizeof(float)); return 1; }
    if (g_q36c_count >= 0) { g_q36c_count += (long long)(floats ? floats : 1) * (long long)sizeof(float); return 1; }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}
static int q36c_bufs(Q36Chain *ch, Model *m, int rows);

/* ---- the fit: how many layers the chain places (vk_chain.h, vkc_fit) -------------- */
/* Does vk_qw_tensor read w's rows in place (COLI_VK_IMPORT)? Then its scales alone take
 * device memory. The dense-host decision comes after the fit, so its one input that
 * changes the answer, COLI_VK_DENSE_HOST=0, is read here too. */
static int q36c_imports(const QW *w, int fmt, const void *wq) {
    const char *e = getenv("COLI_VK_DENSE_HOST");
    int only = coli_vk_dense_device_only() || (e && *e && atoi(e) == 0);
    return g_vk_import && (coli_vk_device_integrated() || !only) && fmt != 4 && w->w != wq;
}
/* w as vk_qw_tensor places it: its device bytes (dev) and what coli_vk_mem_info counts
 * for it (mat); nothing for a matrix that is not loaded or stays on the CPU */
static void q36c_fit_qw(const QW *w, size_t *dev, size_t *mat) {
    const void *wq; const float *sc;
    if (w->vk_off) return;
    int fmt = vk_qw_fmt(w, &wq, &sc), gs = fmt == 4 ? 64 : 0;
    if (!wq) return;
    if (q36c_imports(w, fmt, wq)) {
        size_t s = coli_vk_tensor_scale_count(fmt, w->I, w->O, gs) * sizeof(float), a = coli_vk_buffer_alignment();
        *dev += ((s ? s : 4) + a - 1) / a * a; *mat += s;
        return;
    }
    *dev += vkc_fit_tensor(fmt, w->I, w->O, gs);
    *mat += coli_vk_tensor_payload(fmt, w->I, w->O, gs);
}
/* Layer i on the device: its matrices, the DeltaNet b|a rows and shared-expert gate row
 * (f32 tensors), the DeltaNet state and conv ring, the K/V mirror at its starting size
 * (`kv_rows` positions: the split's floor of three blocks; past it the split decides
 * from what is free then), and its share of the parameter arena. */
static void q36c_fit_layer(Model *m, int i, int kv_rows, size_t *dev, size_t *mat) {
    Cfg *c = &m->c; Layer *l = &m->L[i]; int D = c->hidden, vh = c->dn_vheads;
    QW *ws[16]; int nw = q36c_layer_qws(m, i, ws);
    for (int k = 0; k < nw; k++) q36c_fit_qw(ws[k], dev, mat);
    size_t prm = 2 * (size_t)D;
    if (c->is_attn[i]) {
        if (l->qn) prm += c->head_dim;
        if (l->kn) prm += c->k_head_dim;
        *dev += 2 * vkc_fit_buf((size_t)c->kv_heads * kv_rows * c->k_head_dim * sizeof(float));
    } else {
        prm += (size_t)c->dn_conv_dim * c->dn_convk + 2 * (size_t)vh + c->dn_vdim;
        *dev += vkc_fit_tensor(10, D, 2 * vh, 0); *mat += coli_vk_tensor_payload(10, D, 2 * vh, 0);
        *dev += vkc_fit_buf((size_t)vh * c->dn_kdim * c->dn_vdim * sizeof(float)) +
                vkc_fit_buf((size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
    if (l->sh_gate) { *dev += vkc_fit_tensor(10, D, 1, 0); *mat += coli_vk_tensor_payload(10, D, 1, 0); }
    *dev += prm * sizeof(float);
}
/* The fit, once at start (main, before the dense-host decision and the tier): N layers
 * from layer 0 on the device, the CPU's layers and (unless the fit puts it up) the head
 * marked vk_off so that nothing uploads them. N = 0 turns the chain off. Not run where
 * the chain cannot run at all (the CUDA tier, a qpack container, PILOT, a geometry the
 * shaders do not take): those keep today's handling. */
static void q36c_start(Model *m) {
    if (!g_vk_chain || qt_ready() || qq_active() || g_pilot || !q36c_geometry_ok(&m->c)) return;
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden;
    size_t *per = calloc((size_t)L, sizeof *per), *mat = calloc((size_t)L, sizeof *mat);
    if (!per || !mat) { free(per); free(mat); return; }
    const char *e = getenv("COLI_VK_KV_BLOCK");
    int B = e && *e ? atoi(e) : 64;
    if (B < 1) B = 1;
    for (int i = 0; i < L; i++) q36c_fit_layer(m, i, 3 * B, &per[i], &mat[i]);
    /* fixed: the arena's final norm (and one alignment of its rounding), the scratch of
     * one prompt chunk (q36c_bufs, counted with every attention layer's K/V rows) and its
     * rows' read-back */
    Q36Chain cnt; memset(&cnt, 0, sizeof cnt);
    for (int i = 0; i < L; i++) cnt.n_attn += c->is_attn[i] != 0;
    int rows = vkc_fit_rows(256);
    size_t fixed = vkc_fit_buf((size_t)D * sizeof(float)) + vkc_fit_buf(4);
    g_q36c_fitsum = &fixed;
    q36c_bufs(&cnt, m, rows);
    q36c_res(&cnt.xd, (size_t)rows * D, VKC_DOWN);
    g_q36c_fitsum = NULL;
    size_t tail = 0, tail_m = 0;
    q36c_fit_qw(&m->lm_head, &tail, &tail_m);
    int n = vkc_fit("qwen36", L, per, mat, fixed, tail, &g_q36c_fit);
    /* the chain's pipelines now, after the fit read the free memory: nothing of it was on
     * the device before (vkc_init's own buffer counts in the pools' share) */
    if (n > 0 && !vkc_init()) { g_vk_chain = 0; free(per); free(mat); return; }   /* no chain after all: no fit, as before */
    g_q36c_fit_on = 1;
    /* the layers the primary leaves, on COLI_VK_DEV2's device: a fit of its own from layer
     * n, with that device's free memory (the tier there sizes itself after the placement) */
    int n2 = 0;
    if (n > 0 && n < L && q36c_dev2_wanted() && getenv("COLI_VK_DEV2") && coli_vk_dev2_open_env()) {
        vkc_device(1);
        size_t fixed2 = vkc_fit_buf((size_t)D * sizeof(float)) + vkc_fit_buf(4);
        g_q36c_fitsum = &fixed2;
        q36c_bufs(&cnt, m, rows);
        q36c_res(&cnt.xd, (size_t)rows * D, VKC_DOWN);
        g_q36c_fitsum = NULL;
        n2 = vkc_fit("qwen36 dev2", L - n, per + n, mat + n, fixed2, tail, &g_q36c_fit2);
        if (n2 > 0 && !vkc_init()) {
            fprintf(stderr, "[VK] qwen36 chain: the second device's pipelines did not come up; its layers stay on the CPU\n");
            n2 = 0;
        }
        g_q36c_fit2_on = n2 > 0;
        if (!g_q36c_fit2_on) g_q36c_fit2.tail = 0;
        vkc_device(0);
    }
    free(per); free(mat);
    for (int i = n + n2; i < L; i++) q36c_layer_cpu(NULL, m, i);
    if (!g_q36c_fit.tail && !g_q36c_fit2.tail) m->lm_head.vk_off = 1;
    if (n == 0) {   /* the chain off: nothing of it on the device, every byte is the tier's */
        g_vk_chain = 0;
        vkc_fit_placed("qwen36", &g_q36c_fit);
    }
}
/* After the dense-host decision, before the tier: the chain's N layers (and the head)
 * on the device now, so the tier sizes its budget from what is left. */
static void q36c_place(Model *m) {
    if (!g_q36c_fit_on || !g_q36c_fit.n || !g_vk_chain) return;
    Q36Chain *ch = q36c_setup(m);
    if (!g_q36c_fit2_on) return;
    int lo = g_q36c_fit.n, n2 = g_q36c_fit2.n;
    vkc_device(1);
    if (!ch || ch->n < lo) {
        /* the primary placed fewer layers than its fit: the second device's would not follow
         * them, so they go to the CPU too */
        fprintf(stderr, "[VK] qwen36 chain: the primary device stopped before layer %d; layers %d..%d stay on the CPU, "
                        "not on the second device\n", lo, lo, lo + n2 - 1);
        for (int i = lo; i < lo + n2; i++) q36c_layer_cpu(NULL, m, i);
        if (g_q36c_fit2.tail) m->lm_head.vk_off = 1;
        g_q36c_fit2_on = 0;
        vkc_shutdown();
    } else q36c_setup_dev(m, 1);
    vkc_device(0);
}
/* Prompt rows per chunk (vkc_chunk_rows): the chain's scratch a row, counted from the
 * reservations themselves, and the routed experts' outputs (the tier's rows, the
 * CPU's contributions, and the host's sum) for it. */
static int q36c_chunk_rows(Q36Chain *ch, Model *m) {
    g_q36c_count = 0; q36c_bufs(ch, m, 1); long long b1 = g_q36c_count;
    g_q36c_count = 0; q36c_bufs(ch, m, 2); long long b2 = g_q36c_count;
    g_q36c_count = -1;
    size_t row = (size_t)(b2 - b1) + (size_t)(m->c.n_experts > 0 ? 2 * m->c.topk + 1 : 0) * m->c.hidden * sizeof(float);
    if (ch->ks.on) row += (long long)m->c.q_heads * (3 * (m->c.head_dim + 2) + m->c.q_head_dim) * sizeof(float);
    return vkc_chunk_rows("qwen36", row);
}

/* the scratch buffers for `rows` rows */
static int q36c_bufs(Q36Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    int qo = H * c->q_head_dim, kvo = KV * c->k_head_dim, vd = c->dn_vheads * c->dn_vdim;
    int E = c->n_experts > 0 ? c->n_experts : 1, SI = c->shared_inter > 0 ? c->shared_inter : 1;
    size_t r = (size_t)rows;
    int ok = q36c_res(&ch->x, r * D, VKC_DEV) && q36c_res(&ch->nrm, r * D, VKC_DEV) && q36c_res(&ch->tmp, r * D, VKC_DEV) &&
             q36c_res(&ch->q, r * qo, VKC_DEV) && q36c_res(&ch->k, r * kvo, VKC_DEV) && q36c_res(&ch->v, r * kvo, VKC_DEV) &&
             q36c_res(&ch->ctx, r * H * hd, VKC_DEV) && q36c_res(&ch->qkv, r * c->dn_conv_dim, VKC_DEV) &&
             q36c_res(&ch->z, r * vd, VKC_DEV) && q36c_res(&ch->ab, r * 2 * c->dn_vheads, VKC_DEV) &&
             q36c_res(&ch->cv, r * c->dn_conv_dim, VKC_DEV) && q36c_res(&ch->dny, r * vd, VKC_DEV) &&
             q36c_res(&ch->h2, r * D, VKC_DEV) && q36c_res(&ch->lg, r * E, VKC_DEV) &&
             q36c_res(&ch->gs, r * SI, VKC_DEV) && q36c_res(&ch->us, r * SI, VKC_DEV) && q36c_res(&ch->hs, r * SI, VKC_DEV) &&
             q36c_res(&ch->ds, r * D, VKC_DEV) && q36c_res(&ch->sgd, r, VKC_DEV) && q36c_res(&ch->fin, D, VKC_DEV) &&
             q36c_res(&ch->h2d, r * D, VKC_DOWN) && q36c_res(&ch->lgd, r * E, VKC_DOWN) &&
             q36c_res(&ch->kvd, (size_t)(ch->n_attn ? ch->n_attn : 1) * 2 * r * kvo, VKC_DOWN) &&
             q36c_res(&ch->outd, c->vocab, VKC_DOWN) && q36c_res(&ch->routed, r * D, VKC_UP) &&
             q36c_res(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2), VKC_UP);
    return ok;
}
/* Plan and allocate the KV mirror before sizing a prompt chunk. */
static int q36c_mirror(Q36Chain *ch, Model *m) {
    Cfg *c = &m->c; int KV = c->kv_heads;
    if (ch->cap != m->kv_cap) {        /* the host cache grew: mirror it again (whole, or split past the budget) */
        size_t row = (size_t)KV * c->k_head_dim * 2 * sizeof(float);
        int plan = ch->n_attn ? vkc_kv_plan(&ch->ks, q36c_name(ch), ch->n_attn, row, m->kv_cap, 1, 0,
                                            (size_t)ch->n_attn * ch->dev_rows * row) : 1;
        if (!plan) { ch->cap = 0; return 0; }
        ch->dev_rows = ch->ks.on ? ch->ks.rows : m->kv_cap;
        for (int i = ch->lo; i < ch->lo + ch->n; i++) {
            if (!c->is_attn[i]) continue;
            vkc_free(ch->kc[i]); vkc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
            ch->kv_valid[i] = 0;
            ch->kc[i] = vkc_buf((size_t)KV * ch->dev_rows * c->k_head_dim * sizeof(float), VKC_DEV);
            ch->vc[i] = vkc_buf((size_t)KV * ch->dev_rows * c->k_head_dim * sizeof(float), VKC_DEV);
            if (!ch->kc[i] || !ch->vc[i]) { ch->cap = 0; return 0; }
        }
        ch->cap = m->kv_cap;
    }
    return 1;
}

/* scratch for `rows` rows, and the KV mirrors at the host's capacity */
static int q36c_scratch(Q36Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden;
    size_t r = (size_t)rows;
    if (!q36c_bufs(ch, m, rows)) return 0;
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, r * D * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    if (ch->ks.on && !vkc_kv_parts(&ch->ks, r * c->q_heads * (c->head_dim + 2))) return 0;
    return 1;
}

/* ---- state between the host and the device ----------------------------------- */
/* The host's DeltaNet state brought up to date (before anything reads it there). */
/* (every function here covers both chains: the primary's layers and the second device's) */
static int q36c_sync_one(Model *m, Q36Chain *ch) {
    if (!ch || !ch->ok || ch->dn_where != Q36C_DEV) return 1;
    Cfg *c = &m->c;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    int was = vkc_device(ch->d), ok = 1;
    for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
        if (c->is_attn[i]) continue;
        ok = vkc_read(ch->rec[i], 0, m->DN_rec[i], nr * sizeof(float)) &&
             vkc_read(ch->ring[i], 0, m->DN_conv[i], nc * sizeof(float));
    }
    vkc_device(was);
    if (ok) ch->dn_where = Q36C_BOTH;
    return ok;
}
static void q36c_sync_host(Model *m) {
    for (int d = 0; d < 2; d++)
        if (!q36c_sync_one(m, q36c_of(m, d))) { q36c_recover(m, m->kv_len); return; }
}
/* The host wrote its DeltaNet state (zero: reset_recurrent's zeros). */
static void q36c_host_wrote(Model *m, int zero) {
    for (int d = 0; d < 2; d++) {
        Q36Chain *ch = q36c_of(m, d);
        if (!ch || !ch->ok) continue;
        ch->dn_where = Q36C_HOST; ch->host_zero = zero; ch->snap_valid = 0;
    }
}
/* A CPU step from pos_base over one chain's layers: its host state current before it,
 * the device's stale after. */
static void q36c_cpu_one(Model *m, Q36Chain *ch, int pos_base) {
    if (!ch || !ch->ok) return;
    if (!q36c_sync_one(m, ch)) { q36c_recover(m, m->kv_len); return; }
    ch->dn_where = Q36C_HOST; ch->host_zero = 0; ch->snap_valid = 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
        if (m->c.is_attn[i]) vkc_kv_lower(&ch->ks, ch->attn_ord[i], pos_base);
    }
}
static void q36c_cpu_step(Model *m, int pos_base) {
    for (int d = 0; d < 2; d++) q36c_cpu_one(m, q36c_of(m, d), pos_base);
}
/* A rejected draft: the state after the verify's row `slot` (its first slot+1 rows) is
 * the device's copy in that slot; `len` positions stand, and the KV watermark comes
 * down to them (the rows past them are the rejected drafts'). */
static void q36c_rollback(Model *m, int slot, int len) {
    for (int d = 0; d < 2; d++) {
        Q36Chain *ch = q36c_of(m, d);
        if (!ch || !ch->ok) continue;
        for (int i = ch->lo; i < ch->lo + ch->n; i++) {
            if (ch->kv_valid[i] > len) ch->kv_valid[i] = len;
            if (m->c.is_attn[i]) vkc_kv_lower(&ch->ks, ch->attn_ord[i], len);
        }
        if (slot < 0 || slot >= ch->snap_valid || ch->dn_where != Q36C_DEV) { ch->snap_valid = 0; continue; }
        if (q36c_snap1()) {   /* the slot back over the state, on the device, ahead of the next step */
            Cfg *c = &m->c;
            size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
            int was = vkc_device(ch->d), ok = vkc_begin();
            for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
                if (c->is_attn[i]) continue;
                ok = vkc_copy(ch->rec[i], 0, ch->rec[i], (1 + (size_t)slot) * nr, nr) &&
                     vkc_copy(ch->ring[i], 0, ch->ring[i], (1 + (size_t)slot) * nc, nc);
            }
            ok = ok && vkc_submit(0);
            vkc_device(was);
            if (!ok) q36c_recover(m, len);
            ch->snap_valid = 0;
            continue;
        }
        for (int i = ch->lo; i < ch->lo + ch->n; i++) {
            if (m->c.is_attn[i]) continue;
            VkcBuf *t = ch->rec[i]; ch->rec[i] = ch->rec_snap[slot][i]; ch->rec_snap[slot][i] = t;
            t = ch->ring[i]; ch->ring[i] = ch->ring_snap[slot][i]; ch->ring_snap[slot][i] = t;
        }
        ch->snap_valid = 0;
    }
}
/* The device's copy slots for a verify that copies `rows` rows, allocated the first
 * time a verify that deep runs. */
static int q36c_spec_slots(Q36Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int L = c->n_layers;
    if (q36c_snap1()) return rows <= Q36_SPEC_SNAPS;   /* the slots came with the state */
    if (rows > Q36_SPEC_SNAPS) rows = Q36_SPEC_SNAPS;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int sl = ch->snap_slots; sl < rows; sl++) {
        if (!ch->rec_snap[sl] && !(ch->rec_snap[sl] = (VkcBuf **)calloc((size_t)L, sizeof(VkcBuf *)))) return 0;
        if (!ch->ring_snap[sl] && !(ch->ring_snap[sl] = (VkcBuf **)calloc((size_t)L, sizeof(VkcBuf *)))) return 0;
        for (int i = ch->lo; i < ch->lo + ch->n; i++) {
            if (c->is_attn[i]) continue;
            if (!ch->rec_snap[sl][i] && !(ch->rec_snap[sl][i] = vkc_buf(nr * sizeof(float), VKC_DEV))) return 0;
            if (!ch->ring_snap[sl][i] && !(ch->ring_snap[sl][i] = vkc_buf(nc * sizeof(float), VKC_DEV))) return 0;
        }
        ch->snap_slots = sl + 1;
    }
    return 1;
}
/* Record the uploads that make the device state the host's, as the step at pos_base needs it. */
static int q36c_push_state(Q36Chain *ch, Model *m, int pos_base, int n_rows) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->dn_where == Q36C_HOST) {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
            if (c->is_attn[i]) continue;
            if (ch->host_zero) ok = vkc_zero(ch->rec[i], 0, nr) && vkc_zero(ch->ring[i], 0, nc);
            else ok = vkc_write(ch->rec[i], 0, m->DN_rec[i], nr * sizeof(float)) &&
                      vkc_write(ch->ring[i], 0, m->DN_conv[i], nc * sizeof(float));
        }
        ch->dn_where = Q36C_BOTH;
    }
    int kvd = c->k_head_dim, KV = c->kv_heads;
    for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
        if (c->is_attn[i] && ch->ks.on) {     /* the split: the window placed, its rows below pos_base uploaded */
            VkcKvPart pt[2] = {{KV, kvd, m->K[i], (size_t)m->max_t * kvd, ch->kc[i], 0},
                               {KV, kvd, m->V[i], (size_t)m->max_t * kvd, ch->vc[i], 0}};
            vkc_kv_place(&ch->ks, ch->attn_ord[i], pos_base, n_rows);
            ok = vkc_kv_push(&ch->ks, ch->attn_ord[i], pt, 2, pos_base);
            ch->kv_valid[i] = pos_base;
            continue;
        }
        if (!c->is_attn[i] || ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < KV && ok; h++) {
            size_t src = ((size_t)h * m->max_t + t0) * kvd, dst = ((size_t)h * ch->dev_rows + t0) * kvd;
            ok = vkc_write(ch->kc[i], dst, m->K[i] + src, (size_t)n * kvd * sizeof(float)) &&
                 vkc_write(ch->vc[i], dst, m->V[i] + src, (size_t)n * kvd * sizeof(float));
        }
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}

/* ---- one layer's pieces ---------------------------------------------------------- */
static int q36c_norm(VkcBuf *x, size_t xo, VkcBuf *w, size_t wo, VkcBuf *y, size_t yo, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, VKC_NORM_ADD1, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
static int q36c_attention(Q36Chain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, half = c->rotary_dim / 2, gate_dim = qdim > hd ? qdim - hd : 0;
    int ok = vkc_matmul(vk_qw_tensor(&l->q), ch->nrm, 0, ch->q, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->k), ch->nrm, 0, ch->k, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->v), ch->nrm, 0, ch->v, 0, n);
    if (ok && l->qn) { VkcNorm p = {n * H, hd, H, 0, qo, qdim, 0, qo, qdim, (int)ch->o_qn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
                       ok = vkc_norm(ch->q, ch->prm, ch->q, &p); }
    if (ok && half) { VkcRope p = {n * H, H, 0, qo, qdim, half, 0, 2 * half}; ok = vkc_rope(ch->q, ch->cs, &p); }
    if (ok && l->kn) { VkcNorm p = {n * KV, kvd, KV, 0, kvo, kvd, 0, kvo, kvd, (int)ch->o_kn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
                       ok = vkc_norm(ch->k, ch->prm, ch->k, &p); }
    if (ok && half) { VkcRope p = {n * KV, KV, 0, kvo, kvd, half, 0, 2 * half}; ok = vkc_rope(ch->k, ch->cs, &p); }
    if (!ok) return 0;
    size_t ko = (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    if (ch->ks.on) {   /* the split: the rows into their window slots, the attention in two parts */
        VkcKvPart pk = {KV, kvd, m->K[i], (size_t)m->max_t * kvd, ch->kc[i], 0}, pv = {KV, kvd, m->V[i], (size_t)m->max_t * kvd, ch->vc[i], 0};
        VkcKvGqa a = {&ch->ks, ch->attn_ord[i], ch->q, ch->kc[i], ch->vc[i], ch->q, NULL, NULL, ch->ctx, n, H, KV, hd, kvd, pb, 0, 0, 0, 0,
                      qo, qdim, gate_dim > 0, hd, qo, qdim, H * hd, 0, 0, 1.f / sqrtf((float)hd),
                      m->K[i], m->V[i], (size_t)m->max_t * kvd, (size_t)kvd, (size_t)m->max_t * kvd, (size_t)kvd};
        return vkc_kv_store(&ch->ks, ch->attn_ord[i], &pk, ch->k, 0, (size_t)kvo, (size_t)kvd, pb, n) &&
               vkc_kv_store(&ch->ks, ch->attn_ord[i], &pv, ch->v, 0, (size_t)kvo, (size_t)kvd, pb, n) &&
               vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo) &&
               vkc_kv_gqa(&a) && vkc_matmul(vk_qw_tensor(&l->o), ch->ctx, 0, ch->tmp, 0, n);
    }
    /* the new rows into the device cache, and to the host's */
    VkcRegion *rg = malloc(sizeof *rg * (size_t)n * KV);
    if (!rg) return 0;
    for (int s = 0; s < n; s++) for (int h = 0; h < KV; h++)
        rg[s * KV + h] = (VkcRegion){((size_t)h * ch->dev_rows + pb + s) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
    ok = vkc_copy_regions(ch->kc[i], ch->k, rg, n * KV) && vkc_copy_regions(ch->vc[i], ch->v, rg, n * KV) &&
         vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    free(rg);
    VkcAttn a = {n, H, KV, hd, pb, ch->dev_rows, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0,
                 1.f / sqrtf((float)hd), 0, 0};
    return ok && vkc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, NULL, &a) &&
           vkc_matmul(vk_qw_tensor(&l->o), ch->ctx, 0, ch->tmp, 0, n);
}
/* ns: the chunk's rows a verify copies the state after (rows 0..ns-1, slots c0..). One
 * copy rides on the one dispatch as the shaders pass its row; more split the
 * convolution and the recurrence into one dispatch per copy, each ending on its row
 * (the last takes the chunk's remaining rows): the state crosses between them through
 * memory in f32, so every row gets the single dispatch's bits. */
static int q36c_deltanet(Q36Chain *ch, Model *m, Layer *l, int i, int n, int c0, int ns) {
    Cfg *c = &m->c;
    int vh = c->dn_vheads, CD = c->dn_conv_dim, vd = vh * c->dn_vdim;
    int ok = vkc_matmul(vk_qw_tensor(&l->dn_qkv), ch->nrm, 0, ch->qkv, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->dn_z), ch->nrm, 0, ch->z, 0, n) &&
             vkc_matmul(ch->t_ab[i], ch->nrm, 0, ch->ab, 0, n);
    if (ns > 0 && q36c_snap1()) {   /* one dispatch each: copies after rows 0..ns-1 into slots 1 + c0 .. */
        size_t nrs = (size_t)vh * c->dn_kdim * c->dn_vdim, ncs = (size_t)CD * (c->dn_convk - 1);
        VkcDnConv cp = {n, CD, c->dn_convk, 0, CD, 0, CD, -1, 0, (int)ch->o_conv[i], 0, (int)((1 + c0) * ncs), ns, (int)ncs};
        VkcDnRec rp = {n, vh, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, 0, CD, 0, 2 * vh,
                       vh, 2 * vh, 0, vd, 0, vd, -1, 0, c->eps,
                       1.f / sqrtf((float)c->dn_kdim), 0, (int)((1 + c0) * nrs), (int)ch->o_dn[i], ns, (int)nrs};
        ok = ok && vkc_dnconv(ch->qkv, ch->prm, ch->ring[i], ch->cv, ch->ring[i], &cp) &&
             vkc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, ch->rec[i], ch->prm, ch->dny, ch->rec[i], &rp);
        return ok && vkc_matmul(vk_qw_tensor(&l->dn_out), ch->dny, 0, ch->tmp, 0, n);
    }
    int segs = ns > 1 ? ns : 1;
    for (int r = 0; r < segs && ok; r++) {
        int s0 = ns > 1 ? r : 0, len = ns > 1 && r < ns - 1 ? 1 : n - s0, snap_row = ns ? 0 : -1;
        VkcBuf *rs = ns ? ch->rec_snap[c0 + r][i] : NULL, *cs = ns ? ch->ring_snap[c0 + r][i] : NULL;
        VkcDnConv cp = {len, CD, c->dn_convk, s0 * CD, CD, s0 * CD, CD, snap_row, 0, (int)ch->o_conv[i], 0, 0, 0, 0};
        VkcDnRec rp = {len, vh, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, s0 * CD, CD, s0 * 2 * vh, 2 * vh,
                       vh + s0 * 2 * vh, 2 * vh, s0 * vd, vd, s0 * vd, vd, snap_row, 0, c->eps,
                       1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i], 0, 0};
        ok = vkc_dnconv(ch->qkv, ch->prm, ch->ring[i], ch->cv, cs, &cp) &&
             vkc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, ch->rec[i], ch->prm, ch->dny, rs, &rp);
    }
    return ok && vkc_matmul(vk_qw_tensor(&l->dn_out), ch->dny, 0, ch->tmp, 0, n);
}
static int q36c_shared(Q36Chain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c; int ok = 1, SI = c->shared_inter;
    if (SI > 0) {
        VkcEw p = {VKC_EW_SWIGLU, n * SI, SI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = vkc_matmul(vk_qw_tensor(&l->sh_g), ch->h2, 0, ch->gs, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->sh_u), ch->h2, 0, ch->us, 0, n) &&
             vkc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &p) &&
             vkc_matmul(vk_qw_tensor(&l->sh_d), ch->hs, 0, ch->ds, 0, n);
    }
    if (ok && ch->t_sg[i]) ok = vkc_matmul(ch->t_sg[i], ch->h2, 0, ch->sgd, 0, n);
    return ok;
}
/* The routed experts on the device for a decode step: every expert resident (the tier's
 * entries stay put for the step), plain routing, a few rows. */
static int q36c_moe_ok(Q36Chain *ch, Model *m, int n) {
    Cfg *c = &m->c;
    if (c->n_experts <= 0 || c->n_experts % 64 || n > 8 || ch->moe_tab < 0 || !vkc_moe_ready() || !vkt_ready() ||
        !vkt_all_resident() || !q36_route_plain(m)) return 0;
    if (!ch->moe_tab) {   /* the layers' tables, once (every expert resident: nothing moves) */
        int L = ch->n, E = c->n_experts;
        ch->mgu = calloc((size_t)L, sizeof *ch->mgu); ch->mdn = calloc((size_t)L, sizeof *ch->mdn);
        if (!ch->mgu || !ch->mdn) { ch->moe_tab = -1; return 0; }
        for (int i = 0; i < L; i++) {
            ch->mgu[i] = vkc_buf((size_t)E * 64, VKC_UP); ch->mdn[i] = vkc_buf((size_t)E * 64, VKC_UP);
            if (!ch->mgu[i] || !ch->mdn[i]) { ch->moe_tab = -1; return 0; }
            uint32_t *gu = vkc_ptr(ch->mgu[i]), *dn = vkc_ptr(ch->mdn[i]);
            for (int e = 0; e < E; e++)
                if (!coli_vk_xb_expert_entries((const ColiVkExpert *)vkt_expert(i, e), gu + (size_t)e * 16, dn + (size_t)e * 16)) {
                    fprintf(stderr, "[VK] qwen36 chain: expert %d of layer %d off the grouped GEMV; experts stay on the host\n", e, i);
                    ch->moe_tab = -1; return 0;
                }
        }
        ch->moe_tab = 1;
        fprintf(stderr, "[VK] qwen36 chain: decode steps route and run their experts on the device (%d layers x %d experts)\n", L, E);
    }
    int L = ch->n, E = c->n_experts, K = c->topk, D = c->hidden, I = c->inter, nk = n * K;
    int ng = (I + 63) / 64, nd = (D + 63) / 64;
    if (!q36c_res(&ch->egu, (size_t)nk * 16, VKC_DEV) || !q36c_res(&ch->edn, (size_t)nk * 16, VKC_DEV) ||
        !q36c_res(&ch->wt, nk, VKC_DEV) || !q36c_res(&ch->ix, (size_t)L * nk, VKC_DOWN) ||
        !q36c_res(&ch->itg, (size_t)nk * ng * 4, VKC_UP) || !q36c_res(&ch->itd, (size_t)nk * nd * 4, VKC_UP) ||
        !q36c_res(&ch->amap, nk, VKC_UP) || !q36c_res(&ch->use, nk, VKC_UP) ||
        !q36c_res(&ch->hid, (size_t)nk * I, VKC_DEV) || !q36c_res(&ch->ys, (size_t)nk * D, VKC_DEV) ||
        !q36c_res(&ch->lga, (size_t)L * n * E, VKC_DOWN) || !q36c_res(&ch->rdev, (size_t)n * D, VKC_DEV)) return 0;
    /* items (packed row j, row 0, first output, 1), the identity map, the use flags */
    uint32_t *ig = vkc_ptr(ch->itg), *id = vkc_ptr(ch->itd), *am = vkc_ptr(ch->amap), *us = vkc_ptr(ch->use);
    for (int j = 0; j < nk; j++) {
        am[j] = (uint32_t)j; us[j] = 1;
        for (int b = 0; b < ng; b++) { uint32_t *q = ig + ((size_t)j * ng + b) * 4; q[0] = j; q[1] = 0; q[2] = b * 64; q[3] = 1; }
        for (int b = 0; b < nd; b++) { uint32_t *q = id + ((size_t)j * nd + b) * 4; q[0] = j; q[1] = 0; q[2] = b * 64; q[3] = 1; }
    }
    return 1;
}
/* layer i's routed experts for n rows: the router logits in ch->lg -> ch->routed */
static int q36c_moe(Q36Chain *ch, Model *m, int i, int n) {
    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, D = c->hidden, I = c->inter, nk = n * K;
    int ng = (I + 63) / 64, nd = (D + 63) / 64;
    return vkc_copy(ch->lga, (size_t)i * n * E, ch->lg, 0, (size_t)n * E) &&
           vkc_moe_route(ch->lg, ch->mgu[i], ch->mdn[i], ch->egu, ch->edn, ch->wt, ch->ix, n, E, K, i * nk) &&
           vkc_moe_gemv(1, ch->h2, ch->itg, ch->egu, ch->hid, ch->amap, nk * ng, D, I, 0.f, K, 64) &&
           vkc_moe_gemv(0, ch->hid, ch->itd, ch->edn, ch->ys, ch->amap, nk * nd, I, D, 0.f, 0, 64) &&
           vkc_moe_sum(ch->ys, ch->wt, ch->use, ch->rdev, n, K, D);   /* device memory: the next combine reads it */
}
/* x += routed + gate * shared, as moe() leaves `out` and layers_forward_range adds it */
static int q36c_combine(Q36Chain *ch, Model *m, int i, int n, int dm) {
    Cfg *c = &m->c;
    int flags = (c->n_experts > 0 ? 1 : 0) | (c->shared_inter > 0 ? 2 : 0) | (m->L[i].sh_gate && c->shared_inter > 0 ? 4 : 0);
    VkcEw p = {VKC_EW_COMBINE, n * c->hidden, c->hidden, 1, flags, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->x, ch->x, dm ? ch->rdev : ch->routed, ch->ds, ch->sgd, &p);
}

/* Every layer for S rows from host rows xh, the last row's logits into `logit`; xh gets
 * the final rows back when want_x. 0 = not taken (nothing on the device changed).
 * Otherwise the layers it ran: with a partial chain (N layers, or the head on the CPU)
 * *rows_only is 1, xh holds every row's residual after those layers, and the caller
 * runs the rest of the layers and the head on the CPU. */
/* One chain's layers (lo..lo+n-1 on its device) over the S rows, from the residual rows in
 * xh. Returns its layer count, or 0: *lost = 1 when a frame failed (the state was rebuilt
 * on the CPU, every chain is off), else nothing on its device changed. tail: whether the
 * layers after it run elsewhere (the second device's chain, the CPU). */
static int q36c_forward_seg(Model *m, Q36Chain *ch, float *xh, int S, int pos_base, FILE *lf, int want_x, int nlogits,
                            float *logit, int *rows_only, int *lost) {
    *rows_only = 0; *lost = 0;
    Cfg *c = &m->c; int D = c->hidden, L = ch->n, lo = ch->lo, hi = lo + L, E = c->n_experts;   /* L: the layers on the device */
    int part = hi < c->n_layers || !ch->head;
    if (part) { want_x = 1; nlogits = 0; }   /* the residual comes back; the CPU runs the rest and the head */
    int mirror_ok = q36c_mirror(ch, m);
    int CH = mirror_ok ? q36c_chunk_rows(ch, m) : 1, rows = S < CH ? S : CH;
    if (ch->ks.on && rows > ch->ks.chunk) rows = ch->ks.chunk;   /* a step's rows fit the split's window */
    int kvo = c->kv_heads * c->k_head_dim, half = c->rotary_dim / 2;
    if (m->snap_rows > 0 && !q36c_spec_slots(ch, m, m->snap_rows)) {
        fprintf(stderr, "[VK] qwen36 chain: device memory for a verify's %d copies refused; this verify on the CPU\n",
                m->snap_rows);
        return 0;
    }
    if (!mirror_ok || !q36c_scratch(ch, m, rows) || (lf && !q36c_res(&ch->lfd, (size_t)L * 3 * D, VKC_DOWN)) ||
        (want_x && !q36c_res(&ch->xd, (size_t)rows * D, VKC_DOWN)) ||
        (nlogits > 1 && (!q36c_res(&ch->fin, (size_t)nlogits * D, VKC_DEV) ||
                         !q36c_res(&ch->outd, (size_t)nlogits * c->vocab, VKC_DOWN)))) {
        fprintf(stderr, "[VK] qwen36 chain: device memory for %d rows refused; per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    vkc_gemm_rows(g_q36_rowwise ? 0 : -1);   /* a verify's rows get a decode step's bits */
    int snapped = 0;
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        int ns = m->snap_rows - c0 < n ? m->snap_rows - c0 : n;   /* the rows of this chunk a verify copies */
        if (ns < 0) ns = 0;
        if (ns) snapped = c0 + ns;
        if (!vkc_begin() || !vkc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !q36c_push_state(ch, m, pb, n)) goto lost;
        if (half) {   /* the CPU's own angles, cosines and sines (rope_head_partial / _mrope) */
            float *cs = (float *)vkc_ptr(ch->cs);
            for (int s = 0; s < n; s++) {
                int p3[3]; int mr = m->mpos || m->rope_delta;
                if (mr) mrope_at(m, pb + s, p3);
                for (int j = 0; j < half; j++) {
                    float inv = powf(c->theta, -2.0f * j / c->rotary_dim);
                    int axis = !mr ? 0 : (j % 3 == 1 && j < 3 * c->mrope_section[1]) ? 1 : (j % 3 == 2 && j < 3 * c->mrope_section[2]) ? 2 : 0;
                    float ang = (mr ? p3[axis] : pb + s) * inv;
                    cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
                }
            }
        }
        int ok = 1, pending = 0;
        int dm = lo == 0 && hi == L && q36c_moe_ok(ch, m, n);   /* the whole step in one submission */
        for (int i = lo; i < hi && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) {
                ok = q36c_combine(ch, m, i - 1, n, dm);
                if (ok && lf) ok = vkc_copy(ch->lfd, ((size_t)(i - 1 - lo) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
            }
            ok = ok && q36c_norm(ch->x, 0, ch->prm, ch->o_in[i], ch->nrm, 0, n, D, c->eps);
            ok = ok && (c->is_attn[i] ? q36c_attention(ch, m, l, i, n, pb) : q36c_deltanet(ch, m, l, i, n, c0, ns));
            if (ok && lf) ok = vkc_copy(ch->lfd, (size_t)(i - lo) * 3 * D, ch->tmp, (size_t)(n - 1) * D, D);
            VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = ok && vkc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
            if (ok && lf) ok = vkc_copy(ch->lfd, ((size_t)(i - lo) * 3 + 1) * D, ch->x, (size_t)(n - 1) * D, D);
            ok = ok && q36c_norm(ch->x, 0, ch->prm, ch->o_post[i], ch->h2, 0, n, D, c->eps);
            if (ok && E > 0 && dm) {
                ok = vkc_matmul(vk_qw_tensor(&l->gate), ch->h2, 0, ch->lg, 0, n) && q36c_shared(ch, m, l, i, n) &&
                     q36c_moe(ch, m, i, n);
            } else if (ok && E > 0) {
                ok = vkc_matmul(vk_qw_tensor(&l->gate), ch->h2, 0, ch->lg, 0, n) &&
                     vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && vkc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D);
                double t0 = tm_now();
                /* A1; while it runs, the tier loads the experts this layer will likely
                 * stream (a big prompt chunk only) */
                ok = ok && vkc_submit(0);
                if (ok) vkt_stream_prefetch(i, n);
                ok = ok && vkc_finish();
                ch->frames++; ch->wait_ms += tm_now() - t0;
                if (!ok) break;
                if (c->is_attn[i]) {                           /* the rows into the host's cache */
                    const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
                    for (int s = 0; s < n; s++) for (int h = 0; h < c->kv_heads; h++) {
                        size_t dst = ((size_t)h * m->max_t + pb + s) * c->k_head_dim, src = (size_t)s * kvo + (size_t)h * c->k_head_dim;
                        memcpy(m->K[i] + dst, kv + src, c->k_head_dim * sizeof(float));
                        memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->k_head_dim * sizeof(float));
                    }
                }
                /* A2: the shared expert, while the host computes the routed experts */
                ok = vkc_begin() && q36c_shared(ch, m, l, i, n) && vkc_submit(0);
                ch->frames++;
                double t1 = tm_now();
                moe_ex(m, l, i, (float *)vkc_ptr(ch->h2d), n, ch->host_routed, (const float *)vkc_ptr(ch->lgd), 1);
                memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
                double t2 = tm_now();
                tm_add(n, 2, t2 - t1); ch->host_ms += t2 - t1;
                ok = ok && vkc_begin();
            } else if (ok) {
                ok = q36c_shared(ch, m, l, i, n);
                /* A dense backbone has no expert readback to end the frame.
                 * Keep each layer in its own submission: a full 27B prompt in
                 * one submission can exceed the driver's watchdog. The next
                 * frame's barrier orders its residual add after this MLP. */
                if (ok && i + 1 < hi) ok = vkc_submit(0) && vkc_begin();
            }
            pending = 1;
        }
        if (ok) ok = q36c_combine(ch, m, hi - 1, n, dm);
        if (ok && lf) ok = vkc_copy(ch->lfd, ((size_t)(L - 1) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
        if (ok && want_x) ok = vkc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D);
        int last = c0 + n == S;
        /* the final norm and lm_head on this chunk's share of the last nlogits rows */
        int lo0 = S - nlogits > c0 ? S - nlogits - c0 : 0, lo_n = nlogits ? n - lo0 : 0;
        if (ok && lo_n > 0) {
            int dst = c0 + lo0 - (S - nlogits);
            ok = q36c_norm(ch->x, (size_t)lo0 * D, ch->prm, ch->o_final, ch->fin, (size_t)dst * D, lo_n, D, c->eps) &&
                 vkc_matmul(vk_qw_tensor(&m->lm_head), ch->fin, (size_t)dst * D, ch->outd, (size_t)dst * c->vocab, lo_n);
        }
        double t0 = tm_now();
        ok = ok && vkc_submit(1);
        ch->frames++; ch->wait_ms += tm_now() - t0;
        if (!ok) goto lost;
        if (dm) {   /* the device's routing, layer by layer, into the host's books */
            const int *ixh = vkc_ptr(ch->ix); const float *lgh = vkc_ptr(ch->lga);
            for (int i = 0; i < L; i++)
                q36_route_note(m, i, ixh + (size_t)i * n * c->topk, lgh + (size_t)i * n * E, n);
            ch->moe_steps++;
        }
        if (E == 0 || dm) for (int i = lo; i < hi; i++) {     /* a dense model (or a device MoE step) reads its K/V rows back here */
            if (!c->is_attn[i]) continue;
            const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
            for (int s = 0; s < n; s++) for (int h = 0; h < c->kv_heads; h++) {
                size_t dst = ((size_t)h * m->max_t + pb + s) * c->k_head_dim, src = (size_t)s * kvo + (size_t)h * c->k_head_dim;
                memcpy(m->K[i] + dst, kv + src, c->k_head_dim * sizeof(float));
                memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->k_head_dim * sizeof(float));
            }
        }
        if (want_x) memcpy(xh + (size_t)c0 * D, vkc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        for (int i = lo; i < hi; i++) if (c->is_attn[i]) { ch->kv_valid[i] = pb + n; vkc_kv_done(&ch->ks, ch->attn_ord[i], pb + n); }
        ch->dn_where = Q36C_DEV; ch->host_zero = 0;
        if (last && !part) memcpy(logit, vkc_ptr(ch->outd), (size_t)nlogits * c->vocab * sizeof(float));
    }
    if (lf) fwrite(vkc_ptr(ch->lfd), sizeof(float), (size_t)L * 3 * D, lf);
    ch->snap_valid = snapped;
    vkc_gemm_rows(-1);
    ch->forwards++;
    *rows_only = part;
    return L;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    vkc_gemm_rows(-1);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(ch->d); }
    vkc_device(0);
    q36c_recover(m, pos_base);
    *lost = 1;
    return 0;
}

/* Every layer the chains hold for S rows from host rows xh, the last row's logits into
 * `logit`; xh gets the final rows back when want_x. 0 = not taken (nothing on the devices
 * changed, or a device was lost and the state rebuilt: the CPU runs the step). Otherwise
 * the layers they ran, from layer 0: the primary's, then the second device's (from the
 * residual rows the primary's brought back). With layers left (or the head on the CPU)
 * *rows_only is 1, xh holds every row's residual after them, and the caller runs the
 * rest of the layers and the head on the CPU. */
static int q36c_forward(Model *m, float *xh, int S, int pos_base, FILE *lf, int want_x, int nlogits, float *logit,
                        int *rows_only) {
    *rows_only = 0;
    if (!g_vk_chain || qt_ready() || qq_active() || g_pilot) return 0;
    /* prompts only: decode and verifies on the CPU */
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && (S <= 2 || g_q36_rowwise)) return 0;
    Q36Chain *ch = q36c_setup(m), *ch2 = q36c_of(m, 1);
    if (!ch || ch->failed) return 0;
    if (ch2 && (!ch2->ok || ch2->failed || ch2->lo != ch->lo + ch->n)) ch2 = NULL;
    int lost = 0, ro = 0;
    g_q36_vk_dev = 0;
    int n = q36c_forward_seg(m, ch, xh, S, pos_base, lf, ch2 ? 1 : want_x, ch2 ? 0 : nlogits, logit, &ro, &lost);
    if (!n || !ch2) { *rows_only = ro; return n; }
    vkc_device(1); g_q36_vk_dev = 1;
    int n2 = q36c_forward_seg(m, ch2, xh, S, pos_base, lf, want_x, nlogits, logit, &ro, &lost);
    vkc_device(0); g_q36_vk_dev = 0;
    if (lost) return 0;
    if (!n2) {   /* declined before touching its device: the CPU runs its layers this step */
        q36c_cpu_one(m, ch2, pos_base);
        *rows_only = 1;
        return n;
    }
    *rows_only = ro;
    return n + n2;
}

static void q36c_report(Model *m) {
    for (int d = 0; d < 2; d++) {
        Q36Chain *ch = q36c_of(m, d);
        if (!ch || !ch->ok || !ch->forwards) continue;
        int was = vkc_device(d);
        VkcStats st; vkc_stats(&st);
        fprintf(stderr, "[VK] %s chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                        "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
                q36c_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms,
                st.dev_bytes / 1048576.0);
        vkc_kv_report(&ch->ks);
        vkc_prof_print();
        vkc_device(was);
    }
}
