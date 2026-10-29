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
#include "vk_dmoe.h"

#define Q36C_GGUF_ROWS 16   /* a GGUF's steps of more rows than this decline (q36c_forward) */
#define Q36C_HOST 0
#define Q36C_DEV  1
#define Q36C_BOTH 2

/* KV_SLOTS > 1 (serve_mux): one conversation's state on the device, as the chain keeps
 * the held one's (q36c_conv_*): its DeltaNet state and conv ring, its KV mirrors and
 * their watermarks, and which side holds the newest DeltaNet state */
typedef struct {
    int slot;                                  /* the conversation's KV slot; -1: free */
    VkcBuf **rec, **ring, **kc, **vc;          /* per layer, as Q36Chain's */
    int *kv_valid, *kv_rows;                   /* kv_rows: its mirrors' rows (a ring's, else cap) */
    int dn_where, host_zero, cap;              /* cap: the rows its full layers' mirrors hold (0: none now) */
    unsigned long long used;                   /* the step that last used it (least recently used goes first) */
} Q36cConv;

typedef struct {
    int ok, failed;
    int lo, d;                                 /* its layers start at lo, on device d (vkc_device) */
    int n, head;                               /* layers lo..lo+n-1 on the device; lm_head there too */
    int rows;                                  /* scratch capacity in rows */
    int cap;                                   /* positions the mirrors hold (q36c_mirror: grown with the context, at most kv_cap) */
    int host_cap;                              /* the host's kv_cap when they were planned (its sliding layers' form) */
    size_t keep;                               /* the device memory a mirror of Q36C_KV_START positions leaves free (q36c_mirror) */
    int dev_rows;                              /* device KV rows a layer: cap, or the split's */
    int kvh, kvh_set;                          /* the mirrors in float16 (q36c_kvh_want), decided at the first */
    VkcKvSplit ks;                             /* the split past the device's budget (ks.on) */
    VkcBuf *prm;                               /* every norm / conv / DeltaNet parameter */
    size_t *o_in, *o_post, *o_qn, *o_kn, *o_conv, *o_dn, o_final;
    size_t *o_snk;                             /* Flint: a layer's sink logits in prm */
    size_t *o_ra, *o_rf, *o_pre, *o_pf;        /* Muse Spark: residual gates (alpha | beta), pre_en, post_fn */
    ColiVkTensor **t_ab, **t_sg;               /* DeltaNet b|a rows, shared gate row (f32) */
    VkcBuf **rec, **ring, **kc, **vc;          /* per layer state */
    VkcBuf **rec_snap[Q36_SPEC_SNAPS], **ring_snap[Q36_SPEC_SNAPS];   /* a verify's copies, slot r after row r */
    int snap_slots, snap_valid;                /* allocated; written by the last verify */
    int *kv_valid, *attn_ord, n_attn;
    int *kv_rows;                              /* per attention layer: its mirror's rows (a sliding layer's ring, else dev_rows) */
    int dn_where, host_zero;
    VkcBuf *x, *nrm, *tmp, *q, *k, *v, *ctx, *qkv, *z, *ab, *cv, *dny, *h2, *lg, *gs, *us, *hs, *ds, *sgd, *fin;
    VkcBuf *h2d, *lgd, *kvd, *outd, *xd, *lfd, *routed, *cs;
    VkcBuf *kg, *vg;                           /* a long step's window over a ring: its rows before the step and the step's (q36c_attention) */
    VkcBuf *agb, *lat, *latd;                  /* Muse Spark: the attention gate's rows, the experts' input rows */
    float *host_routed;
    unsigned long long forwards, frames, host_ms_n;
    double wait_ms, host_ms;
    VkcDmoe dm;                                /* the routed experts in the frame (vk_dmoe.h) */
    /* KV_SLOTS > 1: the conversations' states (conv[conv_cur] is the held one's, whose
     * arrays the fields above point at), and a multiplexed decode step's rows while it
     * runs (q36c_mux_step: mx_rows set, each row's conversation in mx_conv) */
    Q36cConv *conv; int nconv, conv_cur, conv_full;
    unsigned long long conv_clock, conv_handed, mux_steps, mux_rows, mux_host_steps;
    const Q36Row *mx_rows; Q36cConv **mx_conv;
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
    QW *all[] = {&l->q, &l->k, &l->v, &l->o, &l->gate, &l->sh_g, &l->sh_u, &l->sh_d, &l->dn_qkv, &l->dn_z, &l->dn_out,
                 &l->ag, &l->lat_d, &l->lat_u};
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) ws[k] = all[k];
    return (int)(sizeof all / sizeof *all);
}
static int q36c_layer_qws(Model *m, int i, QW **ws) {
    Cfg *c = &m->c; Layer *l = &m->L[i]; int n = 0;
    if (c->is_attn[i]) { ws[n++] = &l->q; ws[n++] = &l->k; ws[n++] = &l->v; ws[n++] = &l->o; }
    else { ws[n++] = &l->dn_qkv; ws[n++] = &l->dn_z; ws[n++] = &l->dn_out; }
    if (c->n_experts > 0 && !q36_dense_l(c, i)) ws[n++] = &l->gate;   /* a Flint dense layer has no router */
    if (l->sh_I > 0) { ws[n++] = &l->sh_g; ws[n++] = &l->sh_u; ws[n++] = &l->sh_d; }
    if (c->muse) { ws[n++] = &l->ag; ws[n++] = &l->lat_d; ws[n++] = &l->lat_u; }
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
        ch->rec[i] = ch->t_ab[i] ? vkc_buf((size_t)vh * c->dn_kdim * c->dn_vdim * sizeof(float), VKC_DEV) : NULL;
        ch->ring[i] = ch->rec[i] ? vkc_buf((size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float), VKC_DEV) : NULL;
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
    ch->o_snk = calloc(L, sizeof(size_t));
    ch->o_ra = calloc(L, sizeof(size_t)); ch->o_rf = calloc(L, sizeof(size_t));
    ch->o_pre = calloc(L, sizeof(size_t)); ch->o_pf = calloc(L, sizeof(size_t));
    ch->t_ab = calloc(L, sizeof(void *)); ch->t_sg = calloc(L, sizeof(void *));
    ch->rec = calloc(L, sizeof(void *)); ch->ring = calloc(L, sizeof(void *));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->attn_ord = calloc(L, sizeof(int)); ch->kv_rows = calloc(L, sizeof(int));
    if (!ch->o_in || !ch->o_post || !ch->o_qn || !ch->o_kn || !ch->o_conv || !ch->o_dn || !ch->o_snk || !ch->o_ra || !ch->o_rf ||
        !ch->o_pre || !ch->o_pf || !ch->t_ab || !ch->t_sg ||
        !ch->rec || !ch->ring || !ch->kc || !ch->vc || !ch->kv_valid || !ch->attn_ord || !ch->kv_rows) return NULL;
    /* the parameter arena of the n layers: offsets, then one upload */
    size_t np = 0;
    int vh = c->dn_vheads, conv_dim = c->dn_conv_dim, convk = c->dn_convk;
    for (int i = lo; i < lo + n; i++) {
        ch->o_in[i] = np; np += D; ch->o_post[i] = np; np += D;
        if (c->is_attn[i]) {
            if (m->L[i].qn) { ch->o_qn[i] = np; np += c->head_dim; }
            if (m->L[i].kn) { ch->o_kn[i] = np; np += c->k_head_dim; }
            if (m->L[i].sinks) { ch->o_snk[i] = np; np += c->q_heads; }
            if (c->muse) {
                ch->o_ra[i] = np; np += 2 * (size_t)D; ch->o_rf[i] = np; np += 2 * (size_t)D;
                ch->o_pre[i] = np; np += q36_eh(c); ch->o_pf[i] = np; np += D;
            }
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
            if (l->sinks) memcpy(arena + ch->o_snk[i], l->sinks, c->q_heads * sizeof(float));
            if (c->muse) {
                memcpy(arena + ch->o_ra[i], l->ra_a, D * sizeof(float)); memcpy(arena + ch->o_ra[i] + D, l->ra_b, D * sizeof(float));
                memcpy(arena + ch->o_rf[i], l->rf_a, D * sizeof(float)); memcpy(arena + ch->o_rf[i] + D, l->rf_b, D * sizeof(float));
                memcpy(arena + ch->o_pre[i], l->pre_en, q36_eh(c) * sizeof(float));
                memcpy(arena + ch->o_pf[i], l->post_fn, D * sizeof(float));
            }
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
    return g_vk_import && (coli_vk_device_integrated() || !only) && fmt != 4 && fmt < 100 && w->w != wq;   /* block formats: copied */
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
        if (l->sinks) prm += c->q_heads;
        if (c->muse) prm += 5 * (size_t)D + q36_eh(c);
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
/* The pipelines the chain makes at first use, made once its layers are placed and before
 * the tier fills the device (vk_chain.h, vkc_prewarm): the block formats' for the types of
 * its matrices, the experts' in the frame, DeltaNet's recurrence. */
static void q36c_prewarm(Model *m, const Q36Chain *ch) {
    Cfg *c = &m->c;
    int dn = 0;
    for (int i = 0; i < c->n_layers; i++) dn |= !c->is_attn[i];
    VkcPrewarm pw = {.moe = c->n_experts > 0 ? (g_gguf ? 2 : 1) : 0, .dn_kd = dn ? c->dn_kdim : 0,
                     .attn_hd = c->head_dim, .attn_gq = c->kv_heads > 0 ? c->q_heads / c->kv_heads : 0};
    const void *wq; const float *sc;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        QW *ws[16]; int nw = q36c_layer_qws(m, i, ws);
        for (int k = 0; k < nw; k++) {
            int f = vk_qw_fmt(ws[k], &wq, &sc);
            if (f >= 100 && f < 164) { pw.types |= 1ull << (f - 100); vkc_prewarm_width(&pw, ws[k]->I); }
        }
    }
    if (ch->head) {
        int f = vk_qw_fmt(&m->lm_head, &wq, &sc);
        if (f >= 100 && f < 164) { pw.types |= 1ull << (f - 100); vkc_prewarm_width(&pw, m->lm_head.I); }
    }
    pw.moe_d = c->hidden; pw.moe_i = c->inter;
    pw.ex_ours = c->n_experts > 0 && !g_gguf;   /* our formats' experts: their grouped GEMMs */
    /* a GGUF's experts, as the file has them: a prompt chunk's grouped GEMMs (vk_dmoe.h) */
    for (int i = 0; g_gguf && m->gqx && i < c->n_layers; i++) {
        int bg = gq_r16_base(m->gqx[i].tg), bd = gq_r16_base(m->gqx[i].td);
        int tg = bg >= 0 ? bg : m->gqx[i].tg, td = bd >= 0 ? bd : m->gqx[i].td;
        if (tg >= 0 && tg < 64) pw.ex_gu |= 1ull << tg;
        if (td >= 0 && td < 64) pw.ex_dn |= 1ull << td;
    }
    vkc_prewarm(&pw);
}
static void q36c_place(Model *m) {
    if (!g_q36c_fit_on || !g_q36c_fit.n || !g_vk_chain) return;
    Q36Chain *ch = q36c_setup(m);
    if (ch) q36c_prewarm(m, ch);
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
static size_t q36c_chunk_row_bytes(Q36Chain *ch, Model *m) {
    g_q36c_count = 0; q36c_bufs(ch, m, 1); long long b1 = g_q36c_count;
    g_q36c_count = 0; q36c_bufs(ch, m, 2); long long b2 = g_q36c_count;
    g_q36c_count = -1;
    size_t row = (size_t)(b2 - b1) + (size_t)(m->c.n_experts > 0 ? 2 * m->c.topk + 1 : 0) * m->c.hidden * sizeof(float);
    if (ch->ks.on) row += (long long)m->c.q_heads * (3 * (m->c.head_dim + 2) + m->c.q_head_dim) * sizeof(float);
    return row;
}
static int q36c_chunk_rows(Q36Chain *ch, Model *m) { return vkc_chunk_rows("qwen36", q36c_chunk_row_bytes(ch, m)); }

/* the scratch buffers for `rows` rows */
static int q36c_bufs(Q36Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    int qo = H * c->q_head_dim, kvo = KV * c->k_head_dim, vd = c->dn_vheads * c->dn_vdim;
    int E = c->n_experts > 0 ? c->n_experts : 1, SI = c->shared_inter > 0 ? c->shared_inter : 1;
    if (c->dense_layer && c->dense_inter > SI) SI = c->dense_inter;   /* Flint's dense layers: a wider MLP */
    size_t r = (size_t)rows;
    /* the routed sum's rows: the experts' width (Muse Spark's latent), at least the hidden */
    const size_t RW = (size_t)(q36_eh(c) > D ? q36_eh(c) : D), LT = (size_t)q36_eh(c), AG = (size_t)H * hd;
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
             q36c_res(&ch->outd, c->vocab, VKC_DOWN) && q36c_res(&ch->routed, r * RW, VKC_UP) &&
             q36c_res(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2), VKC_UP);
    /* Muse Spark's alone (the attention gate's rows, the experts' input rows): another
     * model's scratch, and so coli plan's count of it, stays as it was */
    if (ok && c->muse)
        ok = q36c_res(&ch->agb, r * AG, VKC_DEV) && q36c_res(&ch->lat, r * LT, VKC_DEV) && q36c_res(&ch->latd, r * LT, VKC_DOWN);
    if (ok && m->kv_ring)   /* the sliding layers' rings: a long step's window, linear (the window's rows before it and its own) */
        ok = q36c_res(&ch->kg, (r + c->window) * kvo, VKC_DEV) && q36c_res(&ch->vg, (r + c->window) * kvo, VKC_DEV);   /* floats: a float16 mirror's fit */
    return ok;
}
/* The mirrors in float16 where the host's cache is (q36_kv16) and the device reads one
 * (vkc_kvh_ready): the host's numbers in half the memory and half a decode step's reads after
 * a long prompt. One conversation's chain only (the places of KV_SLOTS share buffers with it),
 * and while its mirror does not split (chain_kvs reads float32: q36c_mirror goes back to
 * float32 when a plan splits). COLI_VK_KV16=0 keeps float32. */
static int q36c_kvh_want(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_KV16"); on = !(e && *e == '0'); }
    return on && q36_kv16() && g_q36_mux_slots <= 1 && !getenv("COLI_VK_CHAIN_MUX") && vkc_kvh_ready();
}
static size_t q36c_kves(const Q36Chain *ch) { return ch && ch->kvh ? 2 : sizeof(float); }
/* a step's float32 K or V rows into a mirror */
static int q36c_kv_put(const Q36Chain *ch, VkcBuf *dst, VkcBuf *src, const VkcRegion *r, int n) {
    return ch->kvh ? vkc_copy_regions_h(dst, src, r, n) : vkc_copy_regions(dst, src, r, n);
}
/* a mirror's rows into another of its type (r in its elements; a float16 mirror's even, copied as
 * floats two halves each: r is changed) */
static int q36c_kv_copy(const Q36Chain *ch, VkcBuf *dst, VkcBuf *src, VkcRegion *r, int n) {
    if (ch->kvh) for (int k = 0; k < n; k++) { r[k].dst /= 2; r[k].src /= 2; r[k].n /= 2; }
    return vkc_copy_regions(dst, src, r, n);
}
static int q36c_attn_c(const Q36Chain *ch, VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, const VkcAttn *a) {
    return ch->kvh ? vkc_attn_h(q, kc, vc, o, gate, NULL, a) : vkc_attn(q, kc, vc, o, gate, NULL, a);
}
static int q36c_attn_wc(const Q36Chain *ch, VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *snk, const VkcAttnW *w) {
    return ch->kvh ? vkc_attn_wh(q, kc, vc, o, gate, NULL, snk, w) : vkc_attn_w(q, kc, vc, o, gate, NULL, snk, w);
}
/* A sliding layer's mirror is a ring of the host's rows (q36_kv_rows), position t in row
 * t % ring on both sides; 0 for a full layer. */
static int q36c_ring(Model *m, int i) { return q36_kv_rows(m, i) != m->max_t ? m->kv_ring : 0; }
/* n of the host's K or V elements from idx (q36_kv_es() bytes each, q36_kv16) into a mirror at
 * element offset at: a float16 mirror (h16) takes the host's halves as they are (at and n even),
 * a float32 one the floats they hold */
static int q36c_kv_up(int h16, VkcBuf *dst, size_t at, const void *base, int64_t idx, size_t n) {
    if (h16) return vkc_write(dst, at / 2, (const uint16_t *)base + idx, n * 2);
    if (!q36_kv16()) return vkc_write(dst, at, (const float *)base + idx, n * sizeof(float));
    float *f = malloc(n * sizeof(float));
    if (!f) return 0;
    kv16_unpack(f, (const uint16_t *)base + idx, (int64_t)n);
    int ok = vkc_write(dst, at, f, n * sizeof(float));   /* copied into the frame's staging here */
    free(f);
    return ok;
}
/* What the chain's KV mirror takes at the context's capacity, after the tier has filled
 * the device, for a model whose mirror cannot split (a sliding window, sinks: Flint, Muse
 * Spark, whose chain stops when the mirror does not fit): the full layers' rows at ctx, a
 * sliding layer's ring (q36_ring_for), at most half of what the device has free. The tier
 * leaves it free (VktConfig.chain_bytes). Before, Flint int4 at 32K of context mirrored
 * 6.6 GB beside a tier that had left 1 GiB, and its chain ran only with
 * COLI_VK_TIER_RESERVE_GB=12. */
#define Q36C_CONV_ROWS 2048
#define Q36C_KV_START 8192   /* the positions the tier leaves the mirror at the start (q36c_mirror grows past them) */
static size_t q36c_late_bytes(Model *m, int ctx) {
    Cfg *c = &m->c;
    if (ctx < 1 || !(c->win_layer || c->attn_sinks)) return 0;
    int R = q36_ring_for(c, ctx), hi = g_q36c_fit_on ? g_q36c_fit.n : c->n_layers;
    size_t row = (size_t)c->kv_heads * c->k_head_dim * 2 * sizeof(float), kv = 0, half = vkc_dev_avail() / 2;   /* as float32 (q36cc_mirror_at) */
    for (int i = 0; i < hi && i < c->n_layers; i++)
        if (c->is_attn[i]) kv += row * (size_t)(R && c->win_layer && c->win_layer[i] ? R : ctx);
    return kv > half ? half : kv;
}
/* Plan and allocate the KV mirror before sizing a prompt chunk: positions [0, need) of the one
 * conversation the chain holds. The mirror grows with the context to what the step needs and
 * an eighth more (1024 at least), rounded to 1024, at most the context, its rows again from
 * the host. Doubling gave the tier's experts back for positions no step reached: to 32768
 * positions for a 17.6K-token prompt (1.2 GiB of Qwen3.6's), and to 32768 at the first token
 * after an 18K prompt (256 MiB of them; decode 55.5 tok/s against 75.5 after 1K tokens on an
 * RTX PRO 4000). A growth uploads the rows again: 0.8 GB at 20K positions, once every few
 * thousand tokens. A device without the room takes it from the expert tier (vkt_give_back),
 * which reserved only the start (q36c_late_bytes at Q36C_KV_START). Before, the mirror took
 * the host's whole capacity at the first forward: 2.79 GiB of an 8 GB card for Flint at 64K
 * of context, whose tier then held 1.41 GiB of experts against 3.67 at 8K (decode 43.1
 * against 49.4 tok/s on an RTX PRO 4000 at 7.4 GB). */
#define Q36C_KV_MARGIN ((size_t)256 << 20)
static int q36c_mirror(Q36Chain *ch, Model *m, int need) {
    Cfg *c = &m->c; int KV = c->kv_heads;
    if (need < 1) need = 1;
    /* the positions a step can reach: the mirror's rows follow them, not the host's capacity,
     * which grows request by request (its rows keep their positions when it does) */
    const int top = qwen36_max_ctx() > m->kv_cap ? qwen36_max_ctx() : m->kv_cap;
    int cap = ch->cap;
    if (cap >= need) {
        if (ch->host_cap == m->kv_cap) return 1;
        /* the host's capacity moved: a sliding layer may have turned from its first rows into a
         * ring (q36c_ring), the rest keep theirs */
    } else {
        cap = need + (need / 8 > 1024 ? need / 8 : 1024);
        if (cap < Q36C_CONV_ROWS) cap = Q36C_CONV_ROWS;
        cap = (cap + 1023) / 1024 * 1024;
        if (cap > top) cap = top;
        if (cap < need) { ch->cap = 0; return 0; }
    }
    if (!ch->kvh_set) { ch->kvh = !ch->d && q36c_kvh_want(); ch->kvh_set = 1; }
    size_t row = (size_t)KV * c->k_head_dim * 2 * q36c_kves(ch);
    /* the rings counted at their rows, as that many layers of the mirror's length */
    int nf = 0, nr = 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) if (c->is_attn[i]) { if (q36c_ring(m, i)) nr++; else nf++; }
    int leq = nf + (nr ? (int)(((size_t)nr * m->kv_ring + cap - 1) / cap) : 0);
    size_t held = 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) if (c->is_attn[i] && ch->kc[i]) held += vkc_bytes(ch->kc[i]) + vkc_bytes(ch->vc[i]);
    /* the full layers' rows past Q36C_KV_START come out of the tier where the device lacks
     * them, so a long conversation's prompt chunks are the size a short one's are: once the
     * prompt scratch exists, what it lacks for a chunk of the most rows stays free (and a
     * margin); before that, the memory free beside a mirror of that start (keep, measured at
     * the first). Keeping all of it took 512 MiB of Qwen3.6's experts for 240 MiB of mirror at
     * 4K on an RTX PRO 4000 whose budget left 2.36 GiB unused (1805 against 773 tok/s) */
    size_t want = (size_t)nf * cap * row, more = want > held ? want - held : 0;
    if (!ch->keep) {
        const size_t start = (size_t)nf * (top < Q36C_KV_START ? top : Q36C_KV_START) * row, av = vkc_dev_avail() + held;
        ch->keep = av > start + Q36C_KV_MARGIN ? av - start : Q36C_KV_MARGIN;
    }
    size_t keep = ch->keep;
    if (ch->rows > 0) {
        const int rmax = q36c_chunk_rows(ch, m);
        keep = (rmax > ch->rows ? (size_t)(rmax - ch->rows) * q36c_chunk_row_bytes(ch, m) : 0) + Q36C_KV_MARGIN;
    }
    /* a mirror that cannot split (a sliding window, sinks: Flint) must fit the plan's room
     * whole (vkc_kv_plan_need: four fifths of the free memory and what the mirror holds):
     * past it the chain stops, and the prompt's rest takes the per-matrix path (Flint with
     * 7 GiB free, a 17K-token prompt: 16256 of 17408 positions fit, 32.8 tok/s) */
    size_t target = more + keep;
    if (c->attn_sinks || c->win_layer) {
        const size_t whole = (size_t)leq * cap * row / 8 * 10 + ((size_t)1 << 20);
        if (whole > held && whole - held > target) target = whole - held;
    }
    size_t took = 0;
    const size_t av0 = vkc_dev_avail();
    if (more && av0 < target) took = vkt_give_back(target - av0);
    if (ch->cap && cap != ch->cap)
        fprintf(stderr, "[VK] qwen36 chain: the KV mirror grows from %d to %d positions (%.1f MiB more, %.1f MiB of them from the "
                        "expert tier)\n", ch->cap, cap, more / 1048576.0, took / 1048576.0);
    int plan = ch->n_attn ? vkc_kv_plan(&ch->ks, q36c_name(ch), leq, row, cap, 1, 0, held) : 1;
    if (plan && ch->ks.on && ch->kvh) {   /* the split reads float32: the mirrors go back to it, planned again */
        fprintf(stderr, "[VK] qwen36 chain: the KV mirror splits past the device's budget: float32 from here on\n");
        vkc_kv_free(&ch->ks); ch->kvh = 0; row *= 2;
        for (int i = ch->lo; i < ch->lo + ch->n; i++) ch->kv_rows[i] = -1;   /* every mirror again, as floats */
        plan = vkc_kv_plan(&ch->ks, q36c_name(ch), leq, row, cap, 1, 0, held);
    }
    if (!plan) { ch->cap = 0; return 0; }
    if (ch->ks.on && (c->attn_sinks || c->win_layer)) {   /* the split's attention knows no window or sinks */
        fprintf(stderr, "[VK] qwen36 chain: the KV cache does not fit the device, and its split has no "
                        "sliding window or sinks (Flint); the chain stops\n");
        vkc_kv_free(&ch->ks); ch->cap = 0; return 0;
    }
    ch->dev_rows = ch->ks.on ? ch->ks.rows : cap;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        if (!c->is_attn[i]) continue;
        const int rows = q36c_ring(m, i) ? m->kv_ring : ch->dev_rows;
        /* a mirror whose rows stay keeps them (a ring's, a full layer's at the same length);
         * a split's go again, as its window moves */
        if (ch->kc[i] && ch->kv_rows[i] == rows && !ch->ks.on) continue;
        vkc_free(ch->kc[i]); vkc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
        ch->kv_valid[i] = 0;
        ch->kv_rows[i] = rows;
        ch->kc[i] = vkc_buf((size_t)KV * ch->kv_rows[i] * c->k_head_dim * q36c_kves(ch), VKC_DEV);
        ch->vc[i] = vkc_buf((size_t)KV * ch->kv_rows[i] * c->k_head_dim * q36c_kves(ch), VKC_DEV);
        if (!ch->kc[i] || !ch->vc[i]) { ch->cap = 0; return 0; }
    }
    ch->cap = cap; ch->host_cap = m->kv_cap;
    return 1;
}

/* scratch for `rows` rows, and the KV mirrors at the host's capacity */
static int q36c_scratch(Q36Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = q36_eh(c) > c->hidden ? q36_eh(c) : c->hidden;   /* the routed sum's width */
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
/* ---- several conversations (KV_SLOTS > 1, serve_mux) ------------------------------
 * The held conversation's state is the chain's own (the fields above, which point at
 * its place); the others stay on the device beside it, in as many places as the device's
 * free memory takes once the tier has its budget (COLI_VK_CHAIN_MUX caps the places beside the
 * chain's own, as colibri's). A conversation the Model holds again finds its state where it left it, and
 * a decode step runs every row whose conversation has a place in one forward
 * (q36c_mux_step): the matrices once for all of them, each row's attention and DeltaNet on
 * its own place. A conversation without a place takes the least recently used one, whose
 * DeltaNet state goes to its host arrays first (its KV rows are the host's already).
 * Without places (a second device, a partial chain, the KV split) the held conversation's
 * DeltaNet state comes to the host when it leaves the Model, and the next one's state goes
 * up from the host, every KV row again, at its next prompt. */
static int q36c_pool_ok(Q36Chain *ch, Model *m) {
    return ch && ch->ok && !ch->failed && g_q36_mux_slots > 1 && ch->d == 0 && !q36c_of(m, 1) &&
           ch->lo == 0 && ch->n == m->c.n_layers && ch->head && !ch->ks.on;
}
static void q36c_conv_save(Q36Chain *ch) {
    if (!ch->conv || ch->conv_cur < 0) return;
    Q36cConv *v = &ch->conv[ch->conv_cur];
    v->rec = ch->rec; v->ring = ch->ring; v->kc = ch->kc; v->vc = ch->vc; v->kv_valid = ch->kv_valid;
    v->kv_rows = ch->kv_rows; v->dn_where = ch->dn_where; v->host_zero = ch->host_zero; v->cap = ch->cap;
}
static void q36c_conv_load(Q36Chain *ch, int k) {
    Q36cConv *v = &ch->conv[k];
    ch->rec = v->rec; ch->ring = v->ring; ch->kc = v->kc; ch->vc = v->vc; ch->kv_valid = v->kv_valid;
    ch->kv_rows = v->kv_rows; ch->dn_where = v->dn_where; ch->host_zero = v->host_zero; ch->cap = v->cap;
    ch->dev_rows = v->cap; ch->snap_valid = 0; ch->conv_cur = k;
}
/* A place's DeltaNet state into its conversation's host arrays (the Model's while it holds it) */
static int q36c_conv_down(Q36Chain *ch, Model *m, Q36cConv *v) {
    if (v->slot < 0 || v->dn_where != Q36C_DEV) return 1;
    Cfg *c = &m->c;
    float **rec = v->slot == g_q36_mux_cur ? m->DN_rec : g_q36_mux_seq[v->slot].DN_rec;
    float **cv = v->slot == g_q36_mux_cur ? m->DN_conv : g_q36_mux_seq[v->slot].DN_conv;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    int ok = rec && cv;
    for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
        if (c->is_attn[i]) continue;
        ok = vkc_read(v->rec[i], 0, rec[i], nr * sizeof(float)) && vkc_read(v->ring[i], 0, cv[i], nc * sizeof(float));
    }
    if (ok) v->dn_where = Q36C_BOTH;
    return ok;
}
static void q36c_conv_free(Model *m, Q36cConv *v) {
    for (int i = 0; i < m->c.n_layers; i++) {
        if (v->rec) vkc_free(v->rec[i]); if (v->ring) vkc_free(v->ring[i]);
        if (v->kc) vkc_free(v->kc[i]); if (v->vc) vkc_free(v->vc[i]);
    }
    free(v->rec); free(v->ring); free(v->kc); free(v->vc); free(v->kv_valid); free(v->kv_rows);
    memset(v, 0, sizeof *v); v->slot = -1;
}
/* A place's mirrors start at Q36C_CONV_ROWS rows and double as its conversation grows
 * (q36c_conv_grow): a conversation takes the device memory its length needs, not the
 * context's. A sliding layer's ring keeps its rows. */
static size_t q36c_conv_kv_bytes(Model *m, Q36Chain *ch, int cap) {
    Cfg *c = &m->c; size_t b = 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++)
        if (c->is_attn[i]) b += 2 * (size_t)c->kv_heads * (q36c_ring(m, i) ? m->kv_ring : cap) * c->k_head_dim * sizeof(float);
    return b;
}
/* Room on the device for `bytes` more (and 512 MiB besides): the KV mirrors of the places
 * used least recently go first, not place `keep`'s nor this step's. Their DeltaNet state
 * stays; their KV rows are the host's, uploaded again when they run. */
static int q36c_conv_room(Q36Chain *ch, Model *m, int keep, size_t bytes) {
    const size_t margin = (size_t)512 << 20;
    while (vkc_dev_avail() < bytes + margin) {
        int k = -1;
        for (int j = 0; j < ch->nconv; j++) {
            Q36cConv *v = &ch->conv[j];
            if (j == keep || v->used == ch->conv_clock || !v->cap || j == ch->conv_cur) continue;
            if (k < 0 || v->used < ch->conv[k].used) k = j;
        }
        if (k < 0) return 0;
        Q36cConv *v = &ch->conv[k];
        for (int i = ch->lo; i < ch->lo + ch->n; i++) {
            if (!m->c.is_attn[i] || q36c_ring(m, i)) continue;
            vkc_free(v->kc[i]); vkc_free(v->vc[i]); v->kc[i] = v->vc[i] = NULL; v->kv_valid[i] = 0; v->kv_rows[i] = 0;
        }
        v->cap = 0;
    }
    return 1;
}
/* place k's full layers' mirrors holding `need` positions: twice what they held (or need,
 * rounded up to 1024), at most the host's capacity; their rows again from the host */
static int q36c_conv_grow(Q36Chain *ch, Model *m, int k, int need) {
    Q36cConv *v = &ch->conv[k];
    if (v->cap >= need) return 1;
    Cfg *c = &m->c; int KV = c->kv_heads, kvd = c->k_head_dim;
    int cap = v->cap ? 2 * v->cap : Q36C_CONV_ROWS;
    if (cap < need) cap = (need + 1023) / 1024 * 1024;
    if (cap > m->kv_cap) cap = m->kv_cap;
    if (cap < need) return 0;
    size_t more = q36c_conv_kv_bytes(m, ch, cap) - q36c_conv_kv_bytes(m, ch, v->cap);
    if (!q36c_conv_room(ch, m, k, more)) return 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        if (!c->is_attn[i]) continue;
        int R = q36c_ring(m, i);
        if (R && v->kc[i]) continue;   /* a ring keeps its rows */
        int rows = R ? m->kv_ring : cap;
        vkc_free(v->kc[i]); vkc_free(v->vc[i]);
        v->kc[i] = vkc_buf((size_t)KV * rows * kvd * sizeof(float), VKC_DEV);
        v->vc[i] = vkc_buf((size_t)KV * rows * kvd * sizeof(float), VKC_DEV);
        v->kv_valid[i] = 0; v->kv_rows[i] = rows;
        if (!v->kc[i] || !v->vc[i]) { v->cap = 0; return 0; }
    }
    v->cap = cap;
    return 1;
}
/* One more place, when the device has the room for its DeltaNet state and its first mirrors */
static int q36c_conv_new(Q36Chain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->n_layers;
    const char *e = getenv("COLI_VK_CHAIN_MUX");   /* places beside the chain's own, as colibri's */
    int most = e && *e ? 1 + atoi(e) : g_q36_mux_slots;
    if (most > g_q36_mux_slots) most = g_q36_mux_slots;
    if (ch->conv_full || ch->nconv >= most) return -1;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float);
    size_t nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float), dn = 0;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) if (!c->is_attn[i]) dn += nr + nc;
    int cap0 = m->kv_cap < Q36C_CONV_ROWS ? m->kv_cap : Q36C_CONV_ROWS;
    size_t need = dn + q36c_conv_kv_bytes(m, ch, cap0), avail = vkc_dev_avail();
    Q36cConv v = {.slot = -1, .dn_where = Q36C_HOST};
    int ok = need + ((size_t)512 << 20) <= avail;
    if (ok) {
        v.rec = calloc(L, sizeof *v.rec); v.ring = calloc(L, sizeof *v.ring);
        v.kc = calloc(L, sizeof *v.kc); v.vc = calloc(L, sizeof *v.vc);
        v.kv_valid = calloc(L, sizeof(int)); v.kv_rows = calloc(L, sizeof(int));
        ok = v.rec && v.ring && v.kc && v.vc && v.kv_valid && v.kv_rows;
    }
    for (int i = ch->lo; i < ch->lo + ch->n && ok; i++)
        if (!c->is_attn[i]) ok = (v.rec[i] = vkc_buf(nr, VKC_DEV)) && (v.ring[i] = vkc_buf(nc, VKC_DEV));
    if (ok) {
        ch->conv[ch->nconv] = v;
        if (q36c_conv_grow(ch, m, ch->nconv, cap0)) return ch->nconv++;
        v = ch->conv[ch->nconv];
        ok = 0;
    }
    if (v.kv_valid) q36c_conv_free(m, &v);
    ch->conv_full = 1;
    fprintf(stderr, "[VK] qwen36 chain: the state of %d conversation%s stays on the device (one more takes %.2f GiB, "
                    "%.2f GiB are free); the others' go through the host when they come back\n",
            ch->nconv, ch->nconv == 1 ? "" : "s", need / 1073741824.0, avail / 1073741824.0);
    return -1;
}
/* slot's place: its own, a free one, a new one, or (evict) the least recently used one
 * that this step does not use, its DeltaNet state to its host arrays first; -1 = none */
static int q36c_conv_get(Q36Chain *ch, Model *m, int slot, int evict) {
    for (int k = 0; k < ch->nconv; k++) if (ch->conv[k].slot == slot) return k;
    int k = -1;
    for (int j = 0; j < ch->nconv && k < 0; j++) if (ch->conv[j].slot < 0) k = j;
    if (k < 0) k = q36c_conv_new(ch, m);
    if (k < 0 && evict) {
        for (int j = 0; j < ch->nconv; j++)
            if (ch->conv[j].used != ch->conv_clock && (k < 0 || ch->conv[j].used < ch->conv[k].used)) k = j;
        if (k < 0) return -1;
        if (k == ch->conv_cur) q36c_conv_save(ch);
        if (!q36c_conv_down(ch, m, &ch->conv[k])) return -1;
        ch->conv_handed++;
    }
    if (k < 0) return -1;
    Q36cConv *v = &ch->conv[k];   /* the conversation's state is the host's: it goes up at its next step */
    v->slot = slot; v->dn_where = Q36C_HOST; v->host_zero = 0;
    for (int i = 0; i < m->c.n_layers; i++) v->kv_valid[i] = 0;
    return k;
}
/* serve_mux's start: the chain's own arrays become the held conversation's place */
static void q36c_mux_init(Model *m) {
    Q36Chain *ch = q36c_of(m, 0);
    if (ch && ch->ok && !q36c_pool_ok(ch, m)) {
        fprintf(stderr, "[VK] qwen36 chain: KV_SLOTS=%d with %s: the chain runs each conversation's prompts, its state "
                        "handed over through the host when the conversation changes; the decode steps take the host's path\n",
                g_q36_mux_slots, q36c_of(m, 1) ? "a second device" : ch->ks.on ? "the KV split" : "a partial chain");
        return;
    }
    if (!q36c_pool_ok(ch, m) || ch->conv || !(ch->conv = calloc((size_t)g_q36_mux_slots, sizeof *ch->conv))) return;
    for (int k = 0; k < g_q36_mux_slots; k++) ch->conv[k].slot = -1;
    ch->nconv = 1; ch->conv_cur = 0;
    q36c_conv_save(ch);
    ch->conv[0].slot = g_q36_mux_cur;
}
/* The device lost with places' DeltaNet state on it: each such conversation's state
 * rebuilt on the CPU from its prefix record (q36c_recover, the conversation swapped in
 * when the Model does not hold it) */
static void q36c_mux_lost(Model *m) {
    Q36Chain *ch = q36c_of(m, 0);
    if (!ch || !ch->conv) return;
    q36c_conv_save(ch);
    for (int k = 0; k < ch->nconv; k++) {
        Q36cConv *v = &ch->conv[k];
        if (v->slot < 0 || v->dn_where != Q36C_DEV) continue;
        int held = v->slot == g_q36_mux_cur;
        if (!held) q36_seq_swap(m, &g_q36_mux_seq[v->slot]);
        q36c_recover(m, m->kv_len);
        if (!held) q36_seq_swap(m, &g_q36_mux_seq[v->slot]);
        v->dn_where = Q36C_HOST;
    }
}
/* The held conversation leaves the Model: with places its state stays on the device;
 * without, its DeltaNet state comes to the host (whose arrays are still its own) */
/* The chain turned itself off (no device memory for a step's rows): every place's DeltaNet
 * state that is on the device only comes to its host arrays, which the host's path reads */
static void q36c_mux_drop(Model *m) {
    Q36Chain *ch = q36c_of(m, 0);
    if (!ch || !ch->conv) return;
    q36c_conv_save(ch);
    for (int k = 0; k < ch->nconv; k++)
        if (!q36c_conv_down(ch, m, &ch->conv[k])) { q36c_mux_lost(m); return; }
}
/* The held conversation's place cannot hold its prompt even with the others' mirrors gone:
 * the places close (every state to the host, the others' buffers freed) and the chain keeps
 * one conversation's state again, its mirror at the host's capacity or split past the
 * device (q36c_mirror); the conversations' turns then go through the host as without places */
static void q36c_pool_off(Q36Chain *ch, Model *m) {
    q36c_mux_drop(m);
    for (int k = 0; k < ch->nconv; k++) if (k != ch->conv_cur) q36c_conv_free(m, &ch->conv[k]);
    free(ch->conv); ch->conv = NULL; ch->nconv = 0; ch->conv_cur = -1;
    fprintf(stderr, "[VK] qwen36 chain: a prompt's KV mirror does not fit beside the other conversations' state: "
                    "each conversation's state goes through the host from here on\n");
}
static void q36c_mux_leave(Model *m) {
    Q36Chain *ch = q36c_of(m, 0);
    if (ch && ch->conv && q36c_pool_ok(ch, m)) { q36c_conv_save(ch); return; }
    for (int d = 0; d < 2; d++) q36c_cpu_one(m, q36c_of(m, d), m->kv_len);
}
/* slot is held now (-1: none, a decode step's turn): its place, or nothing on the device
 * counts as its own */
static void q36c_mux_enter(Model *m, int slot) {
    Q36Chain *ch = q36c_of(m, 0);
    if (ch && ch->conv && q36c_pool_ok(ch, m)) {
        if (slot < 0) return;
        ch->conv_clock++;   /* a new turn of the clock: no place is a step's now, any may be handed over */
        int k = q36c_conv_get(ch, m, slot, 1);
        if (k >= 0) { q36c_conv_load(ch, k); ch->conv[k].used = ch->conv_clock; return; }
        q36c_recover(m, m->kv_len);   /* a place's state did not come back: the device is gone */
        q36c_mux_lost(m);
        return;
    }
    for (int d = 0; d < 2; d++) {
        Q36Chain *x = q36c_of(m, d);
        if (!x || !x->ok) continue;
        x->dn_where = Q36C_HOST; x->host_zero = 0; x->snap_valid = 0;
        for (int i = x->lo; i < x->lo + x->n; i++) {
            x->kv_valid[i] = 0;
            if (m->c.is_attn[i]) vkc_kv_lower(&x->ks, x->attn_ord[i], 0);
        }
    }
}
/* A decode step on the host's path (q36_step_rows): every row's DeltaNet state on the host
 * before (after = 0), the places' copies stale after it (after = 1) */
static void q36c_mux_host(Model *m, const Q36Row *rows, int S, int after) {
    Q36Chain *ch = q36c_of(m, 0);
    if (!ch || !ch->conv) return;   /* also after the chain turned off: its places' states come home */
    q36c_conv_save(ch);
    for (int s = 0; s < S; s++) {
        int slot = (int)(rows[s].seq - g_q36_mux_seq);
        for (int k = 0; k < ch->nconv; k++) {
            Q36cConv *v = &ch->conv[k];
            if (v->slot != slot) continue;
            if (after) v->dn_where = Q36C_HOST;
            else if (!q36c_conv_down(ch, m, v)) { q36c_mux_lost(m); return; }
        }
    }
    if (ch->conv_cur >= 0) q36c_conv_load(ch, ch->conv_cur);
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
            VkcKvPart pt[2] = {{KV, kvd, m->K[i], (size_t)m->max_t * kvd, ch->kc[i], 0, q36_kv16()},
                               {KV, kvd, m->V[i], (size_t)m->max_t * kvd, ch->vc[i], 0, q36_kv16()}};
            vkc_kv_place(&ch->ks, ch->attn_ord[i], pos_base, n_rows);
            ok = vkc_kv_push(&ch->ks, ch->attn_ord[i], pt, 2, pos_base);
            ch->kv_valid[i] = pos_base;
            continue;
        }
        /* a ring rewound (a pin, a reused prefix shorter than what the device holds): its window
         * before the step again from the host. A step uploads only the window's rows, so rows
         * below a later window may never have reached the device, and a rewind reads them */
        if (c->is_attn[i] && q36c_ring(m, i) && ch->kv_valid[i] > pos_base)
            ch->kv_valid[i] = pos_base - c->window + 1 > 0 ? pos_base - c->window + 1 : 0;
        if (!c->is_attn[i] || ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0, R = q36c_ring(m, i);
        if (R) {   /* a ring: the window's rows before pos_base, the host's ring rows into the same device rows */
            if (t0 < pos_base - c->window + 1) t0 = pos_base - c->window + 1;
            if (t0 < 0) t0 = 0;
            for (int t = t0; t < pos_base && ok; ) {
                int r = t % R, run = pos_base - t < R - r ? pos_base - t : R - r;
                for (int h = 0; h < KV && ok; h++) {
                    size_t at = ((size_t)h * R + r) * kvd;
                    ok = q36c_kv_up(ch->kvh, ch->kc[i], at, m->K[i], (int64_t)at, (size_t)run * kvd) &&
                         q36c_kv_up(ch->kvh, ch->vc[i], at, m->V[i], (int64_t)at, (size_t)run * kvd);
                }
                t += run;
            }
            ch->kv_valid[i] = pos_base;
            continue;
        }
        for (int h = 0; h < KV && ok; h++) {
            size_t src = ((size_t)h * m->max_t + t0) * kvd, dst = ((size_t)h * ch->dev_rows + t0) * kvd;
            ok = q36c_kv_up(ch->kvh, ch->kc[i], dst, m->K[i], (int64_t)src, (size_t)n * kvd) &&
                 q36c_kv_up(ch->kvh, ch->vc[i], dst, m->V[i], (int64_t)src, (size_t)n * kvd);
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
/* The heads' output (ctx) through o_proj into tmp; Muse Spark normalises every head's
 * output first, then gates it by sigmoid(attn_gate(x)) (agb, recorded with q/k/v). */
static int q36c_attn_out(Q36Chain *ch, Model *m, Layer *l, int n) {
    Cfg *c = &m->c; int H = c->q_heads, hd = c->head_dim;
    if (c->muse) {
        VkcNorm p = {n * H, hd, H, 0, H * hd, hd, 0, H * hd, hd, 0, 0, VKC_NORM_NOW, c->eps, 1.f};
        VkcEw g = {VKC_EW_SIGMUL, n * H * hd, H * hd, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        if (!vkc_norm(ch->ctx, ch->prm, ch->ctx, &p) || !vkc_ew(ch->ctx, ch->ctx, ch->agb, NULL, NULL, &g)) return 0;
    }
    return vkc_matmul(vk_qw_tensor(&l->o), ch->ctx, 0, ch->tmp, 0, n);
}
/* The attention's rows before its cache: q, k and v, their norms and RoPE, K and V as the
 * host's cache holds them */
static int q36c_attn_qkv(Q36Chain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, half = c->rotary_dim / 2;
    if (c->nope_layer && c->nope_layer[i]) half = 0;   /* Flint: a layer without RoPE */
    int ok = vkc_matmul(vk_qw_tensor(&l->q), ch->nrm, 0, ch->q, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->k), ch->nrm, 0, ch->k, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->v), ch->nrm, 0, ch->v, 0, n);
    if (ok && c->muse) ok = vkc_matmul(vk_qw_tensor(&l->ag), ch->nrm, 0, ch->agb, 0, n);
    if (ok && l->qn) { VkcNorm p = {n * H, hd, H, 0, qo, qdim, 0, qo, qdim, (int)ch->o_qn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
                       ok = vkc_norm(ch->q, ch->prm, ch->q, &p); }
    if (ok && half) { VkcRope p = {n * H, H, 0, qo, qdim, half, 0, 2 * half}; ok = vkc_rope(ch->q, ch->cs, &p); }
    if (ok && l->kn) { VkcNorm p = {n * KV, kvd, KV, 0, kvo, kvd, 0, kvo, kvd, (int)ch->o_kn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
                       ok = vkc_norm(ch->k, ch->prm, ch->k, &p); }
    if (ok && half) { VkcRope p = {n * KV, KV, 0, kvo, kvd, half, 0, 2 * half}; ok = vkc_rope(ch->k, ch->cs, &p); }
    if (ok && q36_kv16()) {   /* K and V as the host's cache keeps them (float16): the CPU's numbers, before anything reads them */
        VkcEw r = {VKC_EW_F16, n * kvo, kvo, 0, 0, 0, 0, 0, 0, 0, 0, 1.f};
        ok = vkc_ew(ch->k, ch->k, NULL, NULL, NULL, &r) && vkc_ew(ch->v, ch->v, NULL, NULL, NULL, &r);
    }
    return ok;
}
static int q36c_attention(Q36Chain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, gate_dim = qdim > hd ? qdim - hd : 0, ok = 1;
    const int win = c->win_layer && c->win_layer[i] ? c->window : 0;
    if (!q36c_attn_qkv(ch, m, l, i, n)) return 0;
    size_t ko = (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    if (ch->ks.on) {   /* the split: the rows into their window slots, the attention in two parts */
        VkcKvPart pk = {KV, kvd, m->K[i], (size_t)m->max_t * kvd, ch->kc[i], 0, q36_kv16()},
                  pv = {KV, kvd, m->V[i], (size_t)m->max_t * kvd, ch->vc[i], 0, q36_kv16()};
        VkcKvGqa a = {&ch->ks, ch->attn_ord[i], ch->q, ch->kc[i], ch->vc[i], ch->q, NULL, NULL, ch->ctx, n, H, KV, hd, kvd, pb, 0, 0, 0, 0,
                      qo, qdim, gate_dim > 0, hd, qo, qdim, H * hd, 0, 0, 1.f / sqrtf((float)hd),
                      m->K[i], m->V[i], (size_t)m->max_t * kvd, (size_t)kvd, (size_t)m->max_t * kvd, (size_t)kvd, q36_kv16()};
        return vkc_kv_store(&ch->ks, ch->attn_ord[i], &pk, ch->k, 0, (size_t)kvo, (size_t)kvd, pb, n) &&
               vkc_kv_store(&ch->ks, ch->attn_ord[i], &pv, ch->v, 0, (size_t)kvo, (size_t)kvd, pb, n) &&
               vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo) &&
               vkc_kv_gqa(&a) && q36c_attn_out(ch, m, l, n);
    }
    /* the new rows into the device cache, and to the host's */
    const int R = q36c_ring(m, i), rows = ch->kv_rows[i];   /* a ring: position t in row t % R */
    if (R && n > R - win + 1) {
        /* A step longer than the ring's spare rows would overwrite rows its first rows still see:
         * its window is gathered instead, as MiMo's chain does, the ring's rows of the positions
         * before it and then its own, head-major in kg/vg, where the attention reads them as a
         * cache from position b0 (the same values in the same order: the same bits); then its
         * last R rows go into the ring. */
        int b0 = pb - win + 1 > 0 ? pb - win + 1 : 0, nh = pb - b0, tot = nh + n;
        VkcRegion *g = malloc(sizeof *g * (size_t)KV * (n > 2 ? n : 2));
        if (!g) return 0;
        int ng = 0;
        for (int h = 0; h < KV; h++)
            for (int t = b0; t < pb; ) {
                int r = t % R, run = pb - t < R - r ? pb - t : R - r;
                g[ng++] = (VkcRegion){((size_t)h * tot + (t - b0)) * kvd, ((size_t)h * R + r) * kvd, (size_t)run * kvd};
                t += run;
            }
        if (ng) {   /* q36c_kv_copy halves a float16 mirror's offsets: V's copy from the same rows */
            VkcRegion *gv = malloc(sizeof *gv * (size_t)ng);
            ok = gv != NULL;
            if (ok) memcpy(gv, g, sizeof *gv * (size_t)ng);
            ok = ok && q36c_kv_copy(ch, ch->kg, ch->kc[i], g, ng) && q36c_kv_copy(ch, ch->vg, ch->vc[i], gv, ng);
            free(gv);
        }
        ng = 0;
        for (int s = 0; s < n; s++) for (int h = 0; h < KV; h++)
            g[ng++] = (VkcRegion){((size_t)h * tot + nh + s) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
        ok = ok && q36c_kv_put(ch, ch->kg, ch->k, g, ng) && q36c_kv_put(ch, ch->vg, ch->v, g, ng);
        VkcAttn a = {n, H, KV, hd, nh, tot, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0,
                     1.f / sqrtf((float)hd), 0, 0};
        VkcAttnW w = {a, win, 0, 0, 0, l->sinks != NULL, (int)ch->o_snk[i]};
        ok = ok && q36c_attn_wc(ch, ch->q, ch->kg, ch->vg, ch->ctx, ch->q, l->sinks ? ch->prm : NULL, &w);
        ng = 0;   /* after the attention read the ring's rows: the step's last R into it */
        for (int s = n > R ? n - R : 0; s < n; s++) for (int h = 0; h < KV; h++)
            g[ng++] = (VkcRegion){((size_t)h * R + (pb + s) % R) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
        ok = ok && q36c_kv_put(ch, ch->kc[i], ch->k, g, ng) && q36c_kv_put(ch, ch->vc[i], ch->v, g, ng) &&
             vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
        free(g);
        return ok && q36c_attn_out(ch, m, l, n);
    }
    VkcRegion *rg = malloc(sizeof *rg * (size_t)n * KV);
    if (!rg) return 0;
    for (int s = 0; s < n; s++) for (int h = 0; h < KV; h++)
        rg[s * KV + h] = (VkcRegion){((size_t)h * rows + (R ? (pb + s) % R : pb + s)) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
    ok = q36c_kv_put(ch, ch->kc[i], ch->k, rg, n * KV) && q36c_kv_put(ch, ch->vc[i], ch->v, rg, n * KV) &&
         vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    free(rg);
    VkcAttn a = {n, H, KV, hd, pb, rows, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0,
                 1.f / sqrtf((float)hd), 0, 0};
    if (win || l->sinks) {   /* Flint: a sliding window and a sink per head (vkc_attn_w, as MiMo's) */
        VkcAttnW w = {a, win, R, 0, 0, l->sinks != NULL, (int)ch->o_snk[i]};
        return ok && q36c_attn_wc(ch, ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, l->sinks ? ch->prm : NULL, &w) &&
               q36c_attn_out(ch, m, l, n);
    }
    return ok && q36c_attn_c(ch, ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, &a) && q36c_attn_out(ch, m, l, n);
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
    int segs = ns > 1 ? ns : 1;
    for (int r = 0; r < segs && ok; r++) {
        int s0 = ns > 1 ? r : 0, len = ns > 1 && r < ns - 1 ? 1 : n - s0, snap_row = ns ? 0 : -1;
        VkcBuf *rs = ns ? ch->rec_snap[c0 + r][i] : NULL, *cs = ns ? ch->ring_snap[c0 + r][i] : NULL;
        VkcDnConv cp = {len, CD, c->dn_convk, s0 * CD, CD, s0 * CD, CD, snap_row, 0, (int)ch->o_conv[i], 0, 0};
        VkcDnRec rp = {len, vh, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, s0 * CD, CD, s0 * 2 * vh, 2 * vh,
                       vh + s0 * 2 * vh, 2 * vh, s0 * vd, vd, s0 * vd, vd, snap_row,
                       (g_ssm_a_direct ? 2 : 0) | (g_dn_tiled ? 4 : 0), c->eps,
                       1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
        ok = vkc_dnconv(ch->qkv, ch->prm, ch->ring[i], ch->cv, cs, &cp) &&
             vkc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, ch->rec[i], ch->prm, ch->dny, rs, &rp);
    }
    return ok && vkc_matmul(vk_qw_tensor(&l->dn_out), ch->dny, 0, ch->tmp, 0, n);
}
/* A multiplexed step's rows (ch->mx_rows from c0): the projections over all of them, each
 * row's cache, attention and DeltaNet on its own conversation's place (one row each) */
static int q36c_attention_mx(Q36Chain *ch, Model *m, Layer *l, int i, int n, int c0) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, gate_dim = qdim > hd ? qdim - hd : 0;
    const int win = c->win_layer && c->win_layer[i] ? c->window : 0, R = q36c_ring(m, i);
    if (!q36c_attn_qkv(ch, m, l, i, n)) return 0;
    size_t ko = (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    VkcRegion *rg = malloc(sizeof *rg * (size_t)KV);
    int ok = rg && vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) &&
             vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);   /* the rows for the host */
    for (int s = 0; s < n && ok; s++) {
        Q36cConv *v = ch->mx_conv[c0 + s];
        int pos = ch->mx_rows[c0 + s].pos, rows = v->kv_rows[i];
        for (int h = 0; h < KV; h++)
            rg[h] = (VkcRegion){((size_t)h * rows + (R ? pos % R : pos)) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
        ok = vkc_copy_regions(v->kc[i], ch->k, rg, KV) && vkc_copy_regions(v->vc[i], ch->v, rg, KV);
        VkcAttn a = {1, H, KV, hd, pos, rows, s * qo, qo, qdim, s * qo + hd, qo, qdim, gate_dim > 0, s * H * hd, H * hd, 0, 0,
                     1.f / sqrtf((float)hd), 0, 0};
        if (win || l->sinks) {
            VkcAttnW w = {a, win, R, 0, 0, l->sinks != NULL, (int)ch->o_snk[i]};
            ok = ok && vkc_attn_w(ch->q, v->kc[i], v->vc[i], ch->ctx, ch->q, NULL, l->sinks ? ch->prm : NULL, &w);
        } else ok = ok && vkc_attn(ch->q, v->kc[i], v->vc[i], ch->ctx, ch->q, NULL, &a);
    }
    free(rg);
    return ok && q36c_attn_out(ch, m, l, n);
}
static int q36c_deltanet_mx(Q36Chain *ch, Model *m, Layer *l, int i, int n, int c0) {
    Cfg *c = &m->c;
    int vh = c->dn_vheads, CD = c->dn_conv_dim, vd = vh * c->dn_vdim;
    int ok = vkc_matmul(vk_qw_tensor(&l->dn_qkv), ch->nrm, 0, ch->qkv, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->dn_z), ch->nrm, 0, ch->z, 0, n) &&
             vkc_matmul(ch->t_ab[i], ch->nrm, 0, ch->ab, 0, n);
    for (int s = 0; s < n && ok; s++) {
        Q36cConv *v = ch->mx_conv[c0 + s];
        VkcDnConv cp = {1, CD, c->dn_convk, s * CD, CD, s * CD, CD, -1, 0, (int)ch->o_conv[i], 0, 0};
        VkcDnRec rp = {1, vh, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, s * CD, CD, s * 2 * vh, 2 * vh,
                       vh + s * 2 * vh, 2 * vh, s * vd, vd, s * vd, vd, -1,
                       (g_ssm_a_direct ? 2 : 0) | (g_dn_tiled ? 4 : 0), c->eps,
                       1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
        ok = vkc_dnconv(ch->qkv, ch->prm, v->ring[i], ch->cv, NULL, &cp) &&
             vkc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, v->rec[i], ch->prm, ch->dny, NULL, &rp);
    }
    return ok && vkc_matmul(vk_qw_tensor(&l->dn_out), ch->dny, 0, ch->tmp, 0, n);
}
/* A multiplexed step's places, before its frame: a DeltaNet state the host holds goes up,
 * and every KV row a place lacks below its row's position (the rows the host's path or a
 * prompt wrote) */
static int q36c_mux_push(Q36Chain *ch, Model *m, int c0, int n) {
    Cfg *c = &m->c; int kvd = c->k_head_dim, KV = c->kv_heads, ok = 1;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int s = 0; s < n && ok; s++) {
        Q36cConv *v = ch->mx_conv[c0 + s];
        const Q36Seq *q = ch->mx_rows[c0 + s].seq;
        int pos = ch->mx_rows[c0 + s].pos;
        if (v->dn_where == Q36C_HOST) {
            for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
                if (c->is_attn[i]) continue;
                if (v->host_zero) ok = vkc_zero(v->rec[i], 0, nr) && vkc_zero(v->ring[i], 0, nc);
                else ok = vkc_write(v->rec[i], 0, q->DN_rec[i], nr * sizeof(float)) &&
                          vkc_write(v->ring[i], 0, q->DN_conv[i], nc * sizeof(float));
            }
            v->dn_where = Q36C_BOTH;
        }
        for (int i = ch->lo; i < ch->lo + ch->n && ok; i++) {
            if (c->is_attn[i] && q36c_ring(m, i) && v->kv_valid[i] > pos)   /* a ring rewound (q36c_push_state) */
                v->kv_valid[i] = pos - c->window + 1 > 0 ? pos - c->window + 1 : 0;
            if (!c->is_attn[i] || v->kv_valid[i] >= pos) continue;
            int t0 = v->kv_valid[i], R = q36c_ring(m, i);
            if (R) {   /* a ring: the window's rows before pos */
                if (t0 < pos - c->window + 1) t0 = pos - c->window + 1;
                if (t0 < 0) t0 = 0;
                for (int t = t0; t < pos && ok; ) {
                    int r = t % R, run = pos - t < R - r ? pos - t : R - r;
                    for (int h = 0; h < KV && ok; h++) {
                        size_t at = ((size_t)h * R + r) * kvd;
                        ok = q36c_kv_up(0, v->kc[i], at, q->K[i], (int64_t)at, (size_t)run * kvd) &&
                             q36c_kv_up(0, v->vc[i], at, q->V[i], (int64_t)at, (size_t)run * kvd);
                    }
                    t += run;
                }
            } else
                for (int h = 0; h < KV && ok; h++) {
                    size_t src = ((size_t)h * m->max_t + t0) * kvd, dst = ((size_t)h * v->kv_rows[i] + t0) * kvd;
                    ok = q36c_kv_up(0, v->kc[i], dst, q->K[i], (int64_t)src, (size_t)(pos - t0) * kvd) &&
                         q36c_kv_up(0, v->vc[i], dst, q->V[i], (int64_t)src, (size_t)(pos - t0) * kvd);
                }
            v->kv_valid[i] = pos;
        }
    }
    return ok;
}
static int q36c_shared(Q36Chain *ch, Model *m, Layer *l, int i, int n) {
    int ok = 1, SI = l->sh_I;   /* shared_inter, or a Flint dense layer's MLP width */
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
/* x += routed + gate * shared, as moe() leaves `out` and layers_forward_range adds it;
 * routed: the host's sum (ch->routed) or the device's (ch->dm.y) */
static int q36c_combine(Q36Chain *ch, Model *m, int i, int n, VkcBuf *routed) {
    Cfg *c = &m->c;
    if (c->muse) {   /* Muse Spark: the routed sum back to the hidden width, normalised, through the residual gate */
        const int D = c->hidden;
        VkcNorm p = {n, D, 1, 0, D, D, 0, D, D, (int)ch->o_pf[i], 0, VKC_NORM_PRESCALE, c->post_eps, 1.f};
        VkcEw g = {VKC_EW_RESGATE, n * D, D, 1, 0, 1, 0, 0, 0, (int)ch->o_rf[i] + D, (int)ch->o_rf[i], 1.f};
        return vkc_matmul(vk_qw_tensor(&m->L[i].lat_u), routed, 0, ch->tmp, 0, n) &&
               vkc_norm(ch->tmp, ch->prm, ch->tmp, &p) && vkc_ew(ch->x, ch->x, ch->tmp, ch->prm, ch->prm, &g);
    }
    const int routed_l = c->n_experts > 0 && !q36_dense_l(c, i), SI = m->L[i].sh_I;   /* a Flint dense layer routes nothing */
    int flags = (routed_l ? 1 : 0) | (SI > 0 ? 2 : 0) | (m->L[i].sh_gate && SI > 0 ? 4 : 0);
    VkcEw p = {VKC_EW_COMBINE, n * c->hidden, c->hidden, 1, flags, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->x, ch->x, routed, ch->ds, ch->sgd, &p);
}
/* Layer i's new K/V rows (an attention layer's) into the host's cache once their frame is done. */
static void q36c_kv_down(Q36Chain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c; int kvo = c->kv_heads * c->k_head_dim;
    if (!c->is_attn[i]) return;
    const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    const int R = q36c_ring(m, i);
    if (ch->mx_rows) {   /* a multiplexed step: each row into its own conversation's cache */
        for (int s = 0; s < n; s++) {
            Q36Seq *q = ch->mx_rows[s].seq;
            int pos = ch->mx_rows[s].pos;
            for (int h = 0; h < c->kv_heads; h++) {
                size_t dst = q36_kv_at(m, i, h, pos), src = (size_t)s * kvo + (size_t)h * c->k_head_dim;
                q36_kv_put(q->K[i], (int64_t)dst, kv + src, c->k_head_dim);
                q36_kv_put(q->V[i], (int64_t)dst, kv + (size_t)ch->rows * kvo + src, c->k_head_dim);
            }
            if (R && q->ring_pos) q->ring_pos[pos % R] = pos;
        }
        return;
    }
    for (int s = 0; s < n; s++) {
        for (int h = 0; h < c->kv_heads; h++) {
            size_t dst = q36_kv_at(m, i, h, pb + s), src = (size_t)s * kvo + (size_t)h * c->k_head_dim;
            q36_kv_put(m->K[i], (int64_t)dst, kv + src, c->k_head_dim);
            q36_kv_put(m->V[i], (int64_t)dst, kv + (size_t)ch->rows * kvo + src, c->k_head_dim);
        }
        if (R && m->ring_pos) m->ring_pos[(pb + s) % R] = pb + s;
    }
}

/* ---- the routed experts in the frame (vk_dmoe.h) ---------------------------------------
 * A decode-sized step (or a verify's rows) runs a layer's shared expert, routing and
 * routed experts in the frame when the tier holds all of that layer's experts here:
 * moe_ex's softmax, group-limited top-k (n_group <= E), renormalization when the sum is
 * above 0 (unconditional, as HF), SwiGLU experts and rank-order sum, then the next
 * layer's combine reads the device's sum. Each row is computed alone (a verify's rows have
 * a decode step's bits). The books follow at the frame's end from the ids: the resident
 * collection, the hot-pin frequencies, the history (rt_count) and HITS, the tier's heat.
 * Off with cache routing or the agreement meter (their state crosses tokens on the host),
 * a gate bias, a qpack container or the CUDA tier (the experts are not the Vulkan tier's),
 * more groups than the shader keeps, and on the second device. */
static int q36c_dmoe(Q36Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int E = c->n_experts;
    int ng = c->n_group > 1 && c->n_group <= E ? c->n_group : 1;
    if (ch->d || E <= 0 || g_cache_route || g_route_agree || c->has_bias || c->muse || qq_active() || qt_ready() ||
        (ng > 1 && (ng > 64 || E % ng || c->topk_group < 1 || c->topk_group > ng))) return 0;
    ch->dm.big = 1;   /* a prompt chunk too: a GGUF's experts and ours through the grouped GEMMs */
    return vkc_dmoe_prepare(&ch->dm, ch->lo, ch->n, E, c->topk, c->hidden, c->inter, rows);
}
/* After the frame that ran layers [from, to) for n rows from pb: the K/V rows and the
 * books of those that ran their experts in it. */
static void q36c_dmoe_done(Q36Chain *ch, Model *m, int from, int to, int n, int pb) {
    Cfg *c = &m->c; int K = c->topk, E = c->n_experts;
    for (int i = from; i < to; i++) {
        if (!vkc_dmoe_on(&ch->dm, i)) continue;
        q36c_kv_down(ch, m, i, n, pb);
        const int *ids = vkc_dmoe_ids(&ch->dm, i);
        for (int s = 0; s < n; s++) {
            const int *idx = ids + s * K;
            if (m->resident_collecting) for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) m->seen[(int64_t)i * E + idx[kk]] = 1;
            if (!m->hot_pinned && m->freq) for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) m->freq[(int64_t)i * E + idx[kk]]++;
            if (m->vk_hist) rt_count(i, idx, K);
        }
        for (int j = 0; j < n * K; j++) if (ids[j] >= 0) ehit_mark(m, i, ids[j]);
        vkt_dev_count(i, n, K, ids);
    }
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
    double t_fwd = tm_now();   /* the hybrid steps' gate times the forward (vkc_dmoe_forward_done) */
    *rows_only = 0; *lost = 0;
    Cfg *c = &m->c; int D = c->hidden, L = ch->n, lo = ch->lo, hi = lo + L, E = c->n_experts;   /* L: the layers on the device */
    int part = hi < c->n_layers || !ch->head;
    if (part) { want_x = 1; nlogits = 0; }   /* the residual comes back; the CPU runs the rest and the head */
    /* a multiplexed step's places are grown (q36c_mux_step); the held conversation's place to
     * this step's last position; one conversation alone: the mirror at the host's capacity */
    int mirror_ok = 1;
    if (ch->mx_rows) mirror_ok = 1;
    else if (ch->conv && ch->conv_cur >= 0) {
        q36c_conv_save(ch);
        mirror_ok = q36c_conv_grow(ch, m, ch->conv_cur, pos_base + S);
        q36c_conv_load(ch, ch->conv_cur);
        if (!mirror_ok) { q36c_pool_off(ch, m); mirror_ok = q36c_mirror(ch, m, pos_base + S); }
    } else mirror_ok = q36c_mirror(ch, m, pos_base + S);
    int CH = mirror_ok ? q36c_chunk_rows(ch, m) : 1, rows = S < CH ? S : CH;
    if (ch->ks.on && rows > ch->ks.chunk) rows = ch->ks.chunk;   /* a step's rows fit the split's window */
    int half = c->rotary_dim / 2;
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
        if (ch->conv) q36c_mux_drop(m);   /* the conversations' states the device holds, to the host */
        return 0;
    }
    /* a verify's rows get a decode step's bits, and so do a multiplexed step's: the GEMV that
     * shares each weight's load between up to four rows, not a prompt's tiles */
    vkc_gemm_rows(g_q36_rowwise || ch->mx_rows ? 0 : -1);
    int snapped = 0, dmoe = q36c_dmoe(ch, m, rows);
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        if (c0 > 0 && q36_chunk_control()) {   /* a STOP or CANCEL between two prompt chunks */
            for (int i = lo; i < hi; i++) {   /* the rows past pos_base were this prompt's */
                if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
                if (c->is_attn[i]) vkc_kv_lower(&ch->ks, ch->attn_ord[i], pos_base);
            }
            ch->snap_valid = 0;
            vkc_gemm_rows(-1);
            return 0;
        }
        int ns = m->snap_rows - c0 < n ? m->snap_rows - c0 : n;   /* the rows of this chunk a verify copies */
        if (ns < 0) ns = 0;
        if (ns) snapped = c0 + ns;
        if (!vkc_begin() || !vkc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !(ch->mx_rows ? q36c_mux_push(ch, m, c0, n) : q36c_push_state(ch, m, pb, n))) goto lost;
        if (dmoe && !vkc_dmoe_upload(&ch->dm)) goto lost;   /* the tier's residents changed: the table again */
        if (half) {   /* the CPU's own angles, cosines and sines (rope_head_partial / _mrope) */
            float *cs = (float *)vkc_ptr(ch->cs);
            for (int s = 0; s < n; s++) {
                int p3[3]; int mr = !ch->mx_rows && (m->mpos || m->rope_delta);
                int at = ch->mx_rows ? ch->mx_rows[c0 + s].pos : pb + s;   /* a multiplexed row: its own position */
                if (mr) mrope_at(m, pb + s, p3);
                for (int j = 0; j < half; j++) {
                    float inv = powf(c->theta, -2.0f * j / c->rotary_dim);
                    int axis = !mr ? 0 : (j % 3 == 1 && j < 3 * c->mrope_section[1]) ? 1 : (j % 3 == 2 && j < 3 * c->mrope_section[2]) ? 2 : 0;
                    float ang = (mr ? p3[axis] : at) * inv;
                    cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
                }
            }
        }
        int ok = 1, pending = 0, prev_dev = 0, from = lo, sent = 0;   /* prev_dev: the layer before summed its experts here */
        for (int i = lo; i < hi && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) {
                ok = q36c_combine(ch, m, i - 1, n, prev_dev ? ch->dm.y : ch->routed);
                if (ok && lf) ok = vkc_copy(ch->lfd, ((size_t)(i - 1 - lo) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
            }
            ok = ok && q36c_norm(ch->x, 0, ch->prm, ch->o_in[i], ch->nrm, 0, n, D, c->eps);
            if (ch->mx_rows) ok = ok && (c->is_attn[i] ? q36c_attention_mx(ch, m, l, i, n, c0) : q36c_deltanet_mx(ch, m, l, i, n, c0));
            else ok = ok && (c->is_attn[i] ? q36c_attention(ch, m, l, i, n, pb) : q36c_deltanet(ch, m, l, i, n, c0, ns));
            if (ok && lf) ok = vkc_copy(ch->lfd, (size_t)(i - lo) * 3 * D, ch->tmp, (size_t)(n - 1) * D, D);
            VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            if (c->muse) {   /* Muse Spark: the branch normalised, then the residual gate */
                VkcNorm pn = {n, D, 1, 0, D, D, 0, D, D, 0, 0, VKC_NORM_NOW, c->post_eps, 1.f};
                VkcEw g = {VKC_EW_RESGATE, n * D, D, 1, 0, 1, 0, 0, 0, (int)ch->o_ra[i] + D, (int)ch->o_ra[i], 1.f};
                ok = ok && vkc_norm(ch->tmp, ch->prm, ch->tmp, &pn) && vkc_ew(ch->x, ch->x, ch->tmp, ch->prm, ch->prm, &g);
            } else
            ok = ok && vkc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
            if (ok && lf) ok = vkc_copy(ch->lfd, ((size_t)(i - lo) * 3 + 1) * D, ch->x, (size_t)(n - 1) * D, D);
            ok = ok && q36c_norm(ch->x, 0, ch->prm, ch->o_post[i], ch->h2, 0, n, D, c->eps);
            const int routed_l = E > 0 && !q36_dense_l(c, i);   /* a Flint dense layer: its MLP only */
            if (ok && routed_l && dmoe && vkc_dmoe_on(&ch->dm, i)) {   /* everything here, and on to the next layer */
                int ng = c->n_group > 1 && c->n_group <= E ? c->n_group : 1;
                VkcRoute rp = {.kind = c->router_sigmoid ? 1 : 0, .n_group = ng, .topk_group = ng > 1 ? c->topk_group : 1,
                               .norm = 1, .temp = 1.f, .scale = 1.f};
                VkcMoe mp = {.D = D, .I = c->inter, .act = 0, .blocks = g_gguf};
                ok = vkc_matmul(vk_qw_tensor(&l->gate), ch->h2, 0, ch->lg, 0, n) && q36c_shared(ch, m, l, i, n) &&
                     vkc_dmoe_layer(&ch->dm, i, n, ch->lg, NULL, ch->h2, &rp, &mp);
                prev_dev = 1; pending = 1;
                if (ok && i + 1 < hi) ok = vkc_layer_recorded(&sent);   /* the device starts on these layers */
                continue;
            }
            if (ok && routed_l) {
                ok = vkc_matmul(vk_qw_tensor(&l->gate), ch->h2, 0, ch->lg, 0, n) && vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E);
                if (c->muse) {   /* Muse Spark: the experts read the rows at their width, normalised */
                    VkcNorm pn = {n, q36_eh(c), 1, 0, q36_eh(c), q36_eh(c), 0, q36_eh(c), q36_eh(c), (int)ch->o_pre[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
                    ok = ok && vkc_matmul(vk_qw_tensor(&l->lat_d), ch->h2, 0, ch->lat, 0, n) && vkc_norm(ch->lat, ch->prm, ch->lat, &pn) &&
                         vkc_copy(ch->latd, 0, ch->lat, 0, (size_t)n * q36_eh(c));
                } else
                ok = ok && vkc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D);
                double t0 = tm_now();
                /* A1; while it runs, the tier loads the experts this layer will likely
                 * stream (a big prompt chunk only) */
                ok = ok && vkc_submit(0);
                if (ok) vkt_stream_prefetch(i, n);
                ok = ok && vkc_finish();
                ch->frames++; ch->wait_ms += tm_now() - t0;
                if (!ok) break;
                if (dmoe) q36c_dmoe_done(ch, m, from, i, n, pb);   /* the layers before it in this frame */
                from = i + 1; prev_dev = 0;
                q36c_kv_down(ch, m, i, n, pb);                   /* the rows into the host's cache */
                /* A2: the shared expert, while the host computes the routed experts */
                ok = vkc_begin() && q36c_shared(ch, m, l, i, n) && vkc_submit(0);
                ch->frames++;
                double t1 = tm_now();
                moe_ex(m, l, i, (float *)vkc_ptr(c->muse ? ch->latd : ch->h2d), n, ch->host_routed, (const float *)vkc_ptr(ch->lgd), 1);
                memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * q36_eh(c) * sizeof(float));
                double t2 = tm_now();
                tm_add(n, 2, t2 - t1); ch->host_ms += t2 - t1;
                ok = ok && vkc_begin();
                /* the host's step ran the tier: the table again before a layer reads it */
                if (ok && dmoe) ok = vkc_dmoe_upload(&ch->dm);
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
        if (ok) ok = q36c_combine(ch, m, hi - 1, n, prev_dev ? ch->dm.y : ch->routed);
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
        /* A dense model reads its K/V rows back here, and so does a Flint dense layer (its MLP
         * only, layers 0 and 1): no expert step ends its frame. Without it the host's rows of
         * those layers stayed unwritten, and the first turn to grow the cache mirrored them
         * back to the device: a continued agent turn answered "!!!!" (NaN). */
        for (int i = lo; i < hi; i++) if (E == 0 || q36_dense_l(c, i)) q36c_kv_down(ch, m, i, n, pb);
        if (dmoe) q36c_dmoe_done(ch, m, from, hi, n, pb);
        if (want_x) memcpy(xh + (size_t)c0 * D, vkc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        if (ch->mx_rows)   /* each row's place: its row on the device, its DeltaNet state there only */
            for (int s = 0; s < n; s++) {
                Q36cConv *v = ch->mx_conv[c0 + s];
                for (int i = lo; i < hi; i++) if (c->is_attn[i]) v->kv_valid[i] = ch->mx_rows[c0 + s].pos + 1;
                v->dn_where = Q36C_DEV; v->host_zero = 0;
            }
        else {
            for (int i = lo; i < hi; i++) if (c->is_attn[i]) { ch->kv_valid[i] = pb + n; vkc_kv_done(&ch->ks, ch->attn_ord[i], pb + n); }
            ch->dn_where = Q36C_DEV; ch->host_zero = 0;
        }
        if (last && !part) memcpy(logit, vkc_ptr(ch->outd), (size_t)nlogits * c->vocab * sizeof(float));
    }
    if (lf) fwrite(vkc_ptr(ch->lfd), sizeof(float), (size_t)L * 3 * D, lf);
    ch->snap_valid = snapped;
    vkc_gemm_rows(-1);
    ch->forwards++;
    if (dmoe) vkc_dmoe_forward_done(&ch->dm, S, tm_now() - t_fwd);
    *rows_only = part;
    return L;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    vkc_gemm_rows(-1);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(ch->d); }
    vkc_device(0);
    if (!ch->mx_rows) q36c_recover(m, pos_base);   /* the held conversation (none in a multiplexed step) */
    g_vk_chain = 0; ch->failed = 1;
    if (ch->conv) q36c_mux_lost(m);                /* the others with their state on the device */
    *lost = 1;
    return 0;
}

/* One decode step of several conversations in the chain (serve_mux, KV_SLOTS > 1): row s is
 * the token whose embedding is xh's row s, at rows[s].pos of rows[s].seq, every conversation
 * parked. The matrices and the routed experts run once over the S rows, each row's
 * attention and DeltaNet on its conversation's place; the K/V rows go to each conversation's
 * host cache, its DeltaNet state stays on the device. 1 = done, logit holds S rows; 0 = the
 * host's path runs the step (a conversation without a place, an image turn's rope positions,
 * the device lost: every place's state is the host's again then). */
static int q36c_mux_step(Model *m, const Q36Row *rows, int S, float *xh, float *logit) {
    Q36Chain *ch = q36c_of(m, 0);
    if (!g_vk_chain || !ch || !ch->conv || !q36c_pool_ok(ch, m) || S < 1 || S > q36c_chunk_rows(ch, m)) return 0;
    for (int s = 0; s < S; s++) if (rows[s].seq->mpos || rows[s].seq->rope_delta) return 0;
    Q36cConv **e = malloc((size_t)S * sizeof *e);
    if (!e) return 0;
    q36c_conv_save(ch);
    ch->conv_clock++;
    int ok = 1;
    for (int s = 0; s < S && ok; s++) {
        int k = q36c_conv_get(ch, m, (int)(rows[s].seq - g_q36_mux_seq), 1);
        if ((ok = k >= 0)) { ch->conv[k].used = ch->conv_clock; e[s] = &ch->conv[k]; }
    }
    for (int s = 0; s < S && ok; s++) ok = q36c_conv_grow(ch, m, (int)(e[s] - ch->conv), rows[s].pos + 1);
    int n = 0, ro = 0, lost = 0;
    if (ok) {
        ch->mx_rows = rows; ch->mx_conv = e;
        n = q36c_forward_seg(m, ch, xh, S, 0, NULL, 0, S, logit, &ro, &lost);
        ch->mx_rows = NULL; ch->mx_conv = NULL;
    }
    free(e);
    if (lost) return 0;
    if (ch->conv_cur >= 0) q36c_conv_load(ch, ch->conv_cur);
    if (n) { ch->mux_steps++; ch->mux_rows += (unsigned long long)S; }
    else ch->mux_host_steps++;
    return n > 0;
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
    /* A GGUF's dense matrices: decode and verifies on the block formats' GEMV, a prompt's
     * chunks on their tiled GEMM (vkc_gemm_bd_ready); without it those chunks go through the
     * CPU and the expert tier (measured on an RTX PRO 4000, Qwen3.6-35B-A3B: a 456-token
     * prompt 37-42 tok/s on the GEMV, 45 the other way) */
    if (g_gguf && S > Q36C_GGUF_ROWS && !vkc_gemm_bd_ready()) return 0;
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
    if (!n2 && g_q36_ctl && g_q36_ctl->result) return 0;   /* stopped between two chunks: the caller drops the prompt */
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
                        "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %llu layer steps with "
                        "the routed experts in the frame (%llu of them a prompt chunk's, %llu in the integer form), %.1f MiB on the device\n",
                q36c_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms,
                ch->dm.steps, ch->dm.big_steps, st.moe_i8, st.dev_bytes / 1048576.0);
        if (ch->conv)
            fprintf(stderr, "[VK] %s chain: KV_SLOTS=%d: %llu decode steps of several conversations in the chain (%llu rows, "
                            "%.2f a step), %llu on the host's path; %d conversation%s state on the device, %llu handed over\n",
                    q36c_name(ch), g_q36_mux_slots, ch->mux_steps, ch->mux_rows,
                    ch->mux_steps ? (double)ch->mux_rows / (double)ch->mux_steps : 0.0, ch->mux_host_steps,
                    ch->nconv, ch->nconv == 1 ? "'s" : "s'", ch->conv_handed);
        vkc_kv_report(&ch->ks);
        vkc_dmoe_hyb_report(q36c_name(ch), &ch->dm);
        vkc_prof_print();
        vkc_device(was);
    }
}
