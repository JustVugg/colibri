/* inkling_chain.h -- Inkling's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by inkling.c in a COLI_VULKAN build, after the CPU forward it stands in
 * for; COLI_VK_CHAIN decides (backend_vulkan.c, coli_vk_chain_decide).
 *
 * What runs where, per layer, for one block of rows (decode: one row):
 *   device, frame A1: the MoE output of the layer before joins the residual (the routed
 *     sum from the host, plus each shared expert times its combine weight, in moe()'s
 *     order; then that layer's MLP short convolution and the residual add), the input
 *     RMSNorm, q/k/v and the relative-bias projection r, the short convolutions on K and
 *     V with their rings, the per-head q/k RMSNorms, the attention (chain_relattn.comp:
 *     the bias bank mixed per row and head, tau, the sliding window over a ring cache),
 *     the step's K/V rows into the device ring, o_proj, the attention's short
 *     convolution, the residual add, the post-attention RMSNorm and the router logits
 *     (E routed and the shared experts' logits). Then the frame is waited for.
 *     A dense-MLP layer has no host step: its MLP (gate, up, SwiGLU, down, the global
 *     scale), its short convolution and the residual add follow in the same frame.
 *   host: the sigmoid router with its bias, top-k and the joint combine weights (TOPP
 *     included), the routed experts (the expert tier's device batch and the CPU's share,
 *     joined in rank order: moe_ex, the code moe() runs), the K/V rows copied into the
 *     host's cache.
 *   device, frame A2 (not waited for): the shared experts, while the host computes the
 *     routed experts; their combine weights come up with the routed sum.
 * Crossing per MoE layer: the normalized rows (D floats a row), the router logits
 * (E + n_shared a row) and the K/V rows down; the routed sum (D a row) and the shared
 * experts' weights up. The last frame normalizes the last row, divides it by the logits'
 * width multiplier and runs lm_head; its logits come back. The per-position heads (the
 * logprobs channel, teacher forcing) read the final rows on the host, as before.
 *
 * State and who owns it:
 *   - the residual stream: on the device for the whole forward;
 *   - the K/V cache: the host's stays canonical (every chain step copies its new rows
 *     back). The device holds a mirror per layer with the host's layout, a ring of
 *     `cap` rows on a sliding layer (position t at row t % cap) and the whole context
 *     on a global one. What the host wrote alone (a CPU step) is a range of positions
 *     per layer, [dlo, dhi): the next chain step uploads the rows of its last `cap`
 *     positions first, whatever positions they now hold, so that every row of the
 *     device ring is the host's even after a rewind (a pinned snapshot) over a wrapped
 *     ring. A cache allocated anew (kv_alloc) is mirrored again;
 *   - the four short-convolution states per layer: on the device while the chain runs,
 *     and copied back to the host at the end of every chain step (a few KB a layer), so
 *     the host's are always current; anything that writes the host's (a reset, a pinned
 *     snapshot restored, a CPU step) has the next chain step upload them (or fill zeros).
 *
 * The chain declines (the per-matrix path runs, the state marked as the host's) under
 * CUDA or Metal (they keep their priority), where a matrix is in a form the shaders do
 * not read as this CPU does (bf16 on a CPU whose bf16 dot rounds the activations, as
 * the per-matrix path keeps it on the CPU there), and for a geometry outside its shaders
 * (head dim above 256, a bias bank wider than 64, a convolution above 9 taps).
 * A device lost mid-step: the engine rebuilds the K/V cache and the convolution states
 * on the CPU from the prefix record (the ids the state was built from), a prefill's
 * worth of CPU work, and runs on the CPU from there. Only a state the ids do not
 * describe (audio) cannot be rebuilt: that stops the engine with a message.
 * COLI_VK_CHAIN_FAULT=n (vk_chain.c) fakes the loss at the n-th frame, for tests.
 *
 * Past the device's budget (vk_kvsplit.h) each global layer keeps only a window of
 * blocks of its K/V on the device (the sliding layers' rings are small and stay whole):
 * the step's new rows go into their slots before the attention, which runs over the
 * device's rows (chain_kvs.comp's form of chain_relattn) and on the CPU over the older
 * positions from the host's canonical cache at once, the two merged through their
 * softmax statistics. A split layer's mirror follows a watermark (vkc_kv_lower on a CPU
 * step) instead of the [dlo, dhi) ranges. Below the budget the mirrors are whole.
 *
 * A partial chain (vk_chain.h, vkc_fit; inkc_fit_start at start-up): when the dense
 * layers do not all fit the device, the first n run here and the CPU runs the others and
 * lm_head. Every chunk of a forward crosses the n layers; the residual rows then come down
 * once (the frame that ends the chunk), and the CPU runs layers n.. over all the rows
 * layer by layer, as its own forward does. The device's layers keep their K/V mirrors,
 * rings and convolution states as above; the CPU's layers keep theirs on the host only,
 * so a CPU step lowers and a lost device rebuilds the device's layers alone. A partial
 * chain's matrices (and every layer's, with the host copies dropped) go up at start-up,
 * before the expert cache and the tier size themselves; with every layer fitting and the
 * host copies kept, at the first forward as before. Either way layer by layer: a layer
 * that fails is freed whole and the chain keeps the layers before it. The per-matrix
 * path never uploads a CPU layer's matrices or lm_head. */
#include "vk_chain.h"
#include "vk_kvsplit.h"

/* COLI_VK_CHAIN unset on an integrated GPU with the expert tier (coli_vk_chain_decide):
 * no Inkling checkpoint has been timed on one */
#ifndef INKLING_CHAIN_IGPU
#define INKLING_CHAIN_IGPU COLI_VK_CHAIN_UNMEASURED
#endif
#define INKC_HOST 0   /* the host's convolution states are newer than the device's */
#define INKC_BOTH 1

typedef struct {
    int ok, failed, tried;                     /* tried: inkc_setup ran (ok: the chain runs) */
    int lo, d;                                 /* its layers start at lo, on device d (vkc_device) */
    int nl, full;                              /* layers on the device (lo..lo+nl-1); lm_head there too */
    int rows;                                  /* scratch capacity in rows */
    int max_t; float **hostK;                  /* the host cache the mirrors copy */
    VkcBuf *prm;                               /* norms, bias banks, convolution taps */
    size_t *o_in, *o_post, *o_qn, *o_kn, *o_relp, *o_cw[4], o_final;
    ColiVkTensor *t_lm, **t_q, **t_k, **t_v, **t_r, **t_o, **t_router, **t_dg, **t_du, **t_dd, **t_sg, **t_su, **t_sd;
    VkcBuf **kc, **vc, **ring[4];
    int *cap, *dlo, *dhi;                      /* per layer: ring rows; positions the host wrote alone */
    VkcKvSplit ks;                             /* the global layers' split past the device's budget (ks.on) */
    int *kli, nglob;                           /* per layer: its split table (-1: a sliding layer); global layers */
    int *drows;                                /* per layer: the device mirror's rows (cap, or ks.rows when split) */
    int *slot, nslot;                          /* per layer: its K/V region in kvd, within its frame */
    int kvo_max, qo_max, ro_max, mi_max;
    size_t vs_off;                             /* the V rows' offset in kvs */
    size_t *o_csd, csd_n;                      /* per (bank, layer): the state's offset in csd */
    int cs_where, host_zero;
    VkcBuf *x, *nrm, *tmp, *q, *kvs, *r, *ctx, *h2, *lg, *gs, *us, *sh, *mlp, *fin;
    VkcBuf *h2d, *lgd, *kvd, *csd, *outd, *xd, *routed, *shw, *tau;
    float *host_routed, *host_w;
    unsigned long long forwards;
    double host_ms;
} InkChain;

static void inkc_fatal(const char *what) {
    fprintf(stderr, "[VK] inkling chain: %s -- stopping (COLI_VK_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
static int64_t inkc_cs_cells(const Cfg *c, int bank, int i) {
    return (int64_t)(bank < 2 ? L_KV(c, i) * L_HD(c, i) : c->hidden) * (c->conv_k - 1);
}
/* The layers the chain runs on the device: the first n of a partial chain's fit, else all. */
static int inkc_layers(const Model *m) { return g_inkc_fit.L ? g_inkc_fit.n : m->c.n_layers; }
/* The layers after the primary's on COLI_VK_DEV2's device (docs/vulkan.md, "Layers on two
 * devices"): a second chain from layer g_inkc_fit.n, its own fit (inkc_fit_start), its own
 * copies of their matrices there (the per-matrix path leaves those layers to the CPU). A
 * forward runs the primary's layers, brings the residual rows back and runs these; the CPU
 * runs what is left. */
static VkcFit g_inkc_fit2;
static int g_inkc_fit2_on;
static int g_inkc_ran;   /* the layers the last forward ran on the devices, from layer 0 */
static int inkc_ran(void) { return g_inkc_ran; }
static InkChain *inkc_of(Model *m, int d) { return (InkChain *)(d ? m->vkchain2 : m->vkchain); }
static const char *inkc_name(const InkChain *ch) { return ch && ch->d ? "inkling dev2" : "inkling"; }
static int inkc_dev2_wanted(void) { const char *e = getenv("COLI_VK_CHAIN_DEV2"); return !(e && *e == '0'); }

/* The device copy of a resident matrix or of a view of one (wt_off_i: a shared expert
 * of the fused [ns][R][I] tensors): the per-matrix path's copy when it keeps a table
 * for the tensor (ink_vk_view_tensor, keyed by the view's first element in the tensor),
 * else the chain's own. NULL: not a form the shaders read as this CPU does (ink_vk_fmt). */
static ColiVkTensor *inkc_tensor(int d, Wt view, int I, int O) {
    if (view.vk && !d) return ink_vk_view_tensor(&view, I, O);   /* the per-matrix path's table, keyed by the view */
    int fmt = ink_vk_fmt(&view);
    const void *data = view.q4 ? (const void *)view.q4 : view.f ? (const void *)view.f : (const void *)view.h;
    const float *sc = view.q4 ? view.qs : NULL;
    int gs = view.q4 ? view.gs : 0;
    if (!fmt || !data) return NULL;
    ColiVkTensor *t = NULL;   /* the second device's: its own copy there */
    return (d ? coli_vk_tensor_ensure2(&t, data, sc, fmt, I, O, gs) : coli_vk_tensor_ensure(&t, data, sc, fmt, I, O, gs)) ? t : NULL;
}

/* A geometry the shaders take (head dim up to 256, a bias bank up to 64 wide, up to 9
 * taps); say: the line when one does not. */
static int inkc_geometry_ok(const Model *m, int say) {
    const Cfg *c = &m->c; int dr = c->d_rel, CK = c->conv_k;
    for (int i = 0; i < c->n_layers; i++) {
        int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i);
        if (hd > 256 || hd < 1 || KV < 1 || H % KV || dr < 1 || dr > 64 || L_EXT(c, i) < 1 || CK < 1 || CK > 9 ||
            (c->local[i] && c->window < 1)) {
            if (say)
                fprintf(stderr, "[VK] inkling chain: a geometry its shaders do not take (layer %d: head dim %d, %d/%d heads, "
                                "bias bank %d x %d, %d taps); per-matrix path\n", i, hd, H, KV, dr, L_EXT(c, i), CK);
            return 0;
        }
    }
    return 1;
}
/* The scratch's geometry over every layer (the widest a frame may be) and the frames: a
 * MoE layer ends one (its host step); ch->slot, when allocated, gets each layer's place. */
static void inkc_geom(InkChain *ch, const Model *m) {
    const Cfg *c = &m->c;
    ch->kvo_max = ch->qo_max = ch->ro_max = ch->mi_max = ch->nslot = 0;
    for (int i = 0, fr = 0; i < c->n_layers; i++) {
        int H = L_HEADS(c, i), hd = L_HD(c, i), kvo = L_KV(c, i) * hd;
        if (kvo > ch->kvo_max) ch->kvo_max = kvo;
        if (H * hd > ch->qo_max) ch->qo_max = H * hd;
        if (H * c->d_rel > ch->ro_max) ch->ro_max = H * c->d_rel;
        int mi = c->sparse[i] ? c->moe_inter : c->dense_inter;
        if (mi > ch->mi_max) ch->mi_max = mi;
        if (ch->slot) ch->slot[i] = fr;
        fr++;
        if (fr > ch->nslot) ch->nslot = fr;
        if (c->sparse[i]) fr = 0;
    }
}
/* The chain's state, its tables empty (inkc_setup and inkc_place_init fill them). */
static InkChain *inkc_new_dev(Model *m, int d) {
    InkChain *ch = inkc_of(m, d);
    if (ch) return ch;
    ch = (InkChain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    if (d) m->vkchain2 = ch; else m->vkchain = ch;
    ch->d = d; ch->lo = d ? g_inkc_fit.n : 0;
    int L = m->c.n_layers, nsl = m->c.n_shared > 0 ? m->c.n_shared : 1;
#define INKC_ARR(f, T) (ch->f = calloc((size_t)L, sizeof(T)))
    if (!INKC_ARR(o_in, size_t) || !INKC_ARR(o_post, size_t) || !INKC_ARR(o_qn, size_t) || !INKC_ARR(o_kn, size_t) ||
        !INKC_ARR(o_relp, size_t) || !INKC_ARR(o_cw[0], size_t) || !INKC_ARR(o_cw[1], size_t) || !INKC_ARR(o_cw[2], size_t) ||
        !INKC_ARR(o_cw[3], size_t) || !INKC_ARR(t_q, void *) || !INKC_ARR(t_k, void *) || !INKC_ARR(t_v, void *) ||
        !INKC_ARR(t_r, void *) || !INKC_ARR(t_o, void *) || !INKC_ARR(t_router, void *) || !INKC_ARR(t_dg, void *) ||
        !INKC_ARR(t_du, void *) || !INKC_ARR(t_dd, void *) || !INKC_ARR(kc, void *) || !INKC_ARR(vc, void *) ||
        !INKC_ARR(ring[0], void *) || !INKC_ARR(ring[1], void *) || !INKC_ARR(ring[2], void *) || !INKC_ARR(ring[3], void *) ||
        !INKC_ARR(cap, int) || !INKC_ARR(dlo, int) || !INKC_ARR(dhi, int) || !INKC_ARR(slot, int) ||
        !INKC_ARR(kli, int) || !INKC_ARR(drows, int)) return NULL;
#undef INKC_ARR
    ch->t_sg = calloc((size_t)L * nsl, sizeof(void *)); ch->t_su = calloc((size_t)L * nsl, sizeof(void *));
    ch->t_sd = calloc((size_t)L * nsl, sizeof(void *)); ch->o_csd = calloc((size_t)4 * L, sizeof(size_t));
    if (!ch->t_sg || !ch->t_su || !ch->t_sd || !ch->o_csd) return NULL;
    return ch;
}
static InkChain *inkc_new(Model *m) { return inkc_new_dev(m, 0); }

/* ---- a partial chain (vk_chain.h, vkc_fit): the first n layers on the device ----------
 * Layer i's matrices, all of them or none: q, k, v, the relative-bias projection r and
 * o_proj; a dense layer's MLP, or a MoE layer's router (the chain's own f32 copy) and
 * its shared experts. what: the first that did not go up. */
static void inkc_unplace_layer(Model *m, InkChain *ch, int i);
static int inkc_place_layer(Model *m, InkChain *ch, int i, const char **what) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    int D = c->hidden, E = c->n_experts, ns = c->n_shared, nsl = ns > 0 ? ns : 1, dr = c->d_rel;
    int H = L_HEADS(c, i), hd = L_HD(c, i), kvo = L_KV(c, i) * hd, I = c->moe_inter;
    *what = NULL;
    int d = ch->d;
    if (!(ch->t_q[i] = inkc_tensor(d, l->q, D, H * hd)) || !(ch->t_k[i] = inkc_tensor(d, l->k, D, kvo)) ||
        !(ch->t_v[i] = inkc_tensor(d, l->v, D, kvo)) || !(ch->t_r[i] = inkc_tensor(d, l->r, D, H * dr)) ||
        !(ch->t_o[i] = inkc_tensor(d, l->o, H * hd, D))) *what = "an attention matrix";
    else if (!c->sparse[i]) {
        if (!(ch->t_dg[i] = inkc_tensor(d, l->dg, D, c->dense_inter)) || !(ch->t_du[i] = inkc_tensor(d, l->du, D, c->dense_inter)) ||
            !(ch->t_dd[i] = inkc_tensor(d, l->dd, c->dense_inter, D))) *what = "a dense MLP matrix";
    } else {
        if (!(d ? coli_vk_tensor_ensure2(&ch->t_router[i], l->router, NULL, 10, D, E + ns, 0)
                : coli_vk_tensor_ensure(&ch->t_router[i], l->router, NULL, 10, D, E + ns, 0))) *what = "the router";
        for (int j = 0; j < ns && !*what; j++)
            if (!(ch->t_sg[(size_t)i * nsl + j] = inkc_tensor(d, wt_off_i(l->sh_g, (int64_t)j * I * D, D), D, I)) ||
                !(ch->t_su[(size_t)i * nsl + j] = inkc_tensor(d, wt_off_i(l->sh_u, (int64_t)j * I * D, D), D, I)) ||
                !(ch->t_sd[(size_t)i * nsl + j] = inkc_tensor(d, wt_off_i(l->sh_d, (int64_t)j * D * I, I), I, D)))
                *what = "a shared expert";
    }
    if (*what && g_inkc_fit.L) inkc_unplace_layer(m, ch, i);   /* without a fit the chain declines as before */
    return *what == NULL;
}
/* Everything layer i placed, freed: the chain's own copies, and the per-matrix path's
 * tables of its tensors (their copies with them: the CPU runs the layer). */
static void inkc_unplace_layer(Model *m, InkChain *ch, int i) {
    Cfg *c = &m->c; Layer *l = &m->L[i]; int nsl = c->n_shared > 0 ? c->n_shared : 1;
    if (vkc_ready()) vkc_finish();
    struct { ColiVkTensor **t; Wt *w; } own[] = {{&ch->t_q[i], &l->q}, {&ch->t_k[i], &l->k}, {&ch->t_v[i], &l->v},
        {&ch->t_r[i], &l->r}, {&ch->t_o[i], &l->o}, {&ch->t_dg[i], &l->dg}, {&ch->t_du[i], &l->du}, {&ch->t_dd[i], &l->dd}};
    for (size_t k = 0; k < sizeof own / sizeof own[0]; k++) {   /* a table's copy goes with its table; the second device's are own */
        if (*own[k].t && (ch->d || !own[k].w->vk)) coli_vk_tensor_free(*own[k].t);
        *own[k].t = NULL;
    }
    for (int j = 0; j < nsl; j++) {
        ColiVkTensor **t3[3] = {&ch->t_sg[(size_t)i * nsl + j], &ch->t_su[(size_t)i * nsl + j], &ch->t_sd[(size_t)i * nsl + j]};
        Wt *w3[3] = {&l->sh_g, &l->sh_u, &l->sh_d};
        for (int k = 0; k < 3; k++) { if (*t3[k] && (ch->d || !w3[k]->vk)) coli_vk_tensor_free(*t3[k]); *t3[k] = NULL; }
    }
    if (ch->t_router[i]) coli_vk_tensor_free(ch->t_router[i]);
    ch->t_router[i] = NULL;
    Wt *tab[] = {&l->q, &l->k, &l->v, &l->r, &l->o, &l->dg, &l->du, &l->dd, &l->sh_g, &l->sh_u, &l->sh_d};
    for (size_t k = 0; k < sizeof tab / sizeof tab[0]; k++) if (tab[k]->vk && !((InkVk *)tab[k]->vk)->gone) ink_vk_release(tab[k]);
}
/* Layers from..L-1 and lm_head refused to the device: their tables go (the per-matrix
 * path computes them on the CPU, from their host copies). */
static void inkc_refuse_from(Model *m, int from) {
    for (int i = from; i < m->c.n_layers; i++) {
        Layer *l = &m->L[i];
        Wt *tab[] = {&l->q, &l->k, &l->v, &l->r, &l->o, &l->dg, &l->du, &l->dd, &l->sh_g, &l->sh_u, &l->sh_d};
        for (size_t k = 0; k < sizeof tab / sizeof tab[0]; k++)
            if (tab[k]->vk && !((InkVk *)tab[k]->vk)->gone) ink_vk_release(tab[k]);
    }
    if (m->lm_head.vk && !((InkVk *)m->lm_head.vk)->gone) ink_vk_release(&m->lm_head);
}
/* Layer i failed during setup: free it, refuse it and the layers after it, and go on with
 * the layers before it (vkc_fit_shrink says so). */
static void inkc_shrink(Model *m, InkChain *ch, int i, const char *what) {
    char why[96];
    snprintf(why, sizeof why, "%s did not go up", what ? what : "a matrix");
    inkc_unplace_layer(m, ch, i);
    inkc_refuse_from(m, i);
    vkc_fit_shrink(inkc_name(ch), ch->d ? &g_inkc_fit2 : &g_inkc_fit, i - ch->lo, why);
}

/* The fit (vkc_fit), at start-up before any upload and before the expert cache is sized.
 * A layer's bytes: its matrices in the form each goes up in (the shared experts' views one
 * tensor each), its share of the parameter arena (norms, bias bank, convolution taps), its
 * four convolution rings and their place in the state's read-back, and its K/V mirror at
 * the window's rows (a global layer's grows with the context: the KV split covers that).
 * Fixed: the final norm and the scratch of one prompt chunk of vkc_fit_rows(256) rows, the
 * residual's read-back included. Tail: lm_head. No fit when the chain would decline anyway
 * (a geometry outside its shaders, a matrix in no form the shaders read as this CPU does,
 * bf16 under a rounding bf16 dot included): it says so at its first forward, as before. */
static long long g_inkc_count = -1;
static int g_inkc_fitcount;   /* the counting pass rounds each buffer as the chain's pools do */
static int g_inkc_placed;     /* vkc_fit_placed said it (at start-up, or at the chain's first forward) */
static int inkc_bufs(InkChain *ch, Model *m, int rows);
static void inkc_fit_start(Model *m) {
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, E = c->n_experts, ns = c->n_shared, CK = c->conv_k, dr = c->d_rel;
    if (L < 1 || !inkc_geometry_ok(m, 0)) return;
    size_t *per = calloc((size_t)L, sizeof *per), *mat = calloc((size_t)L, sizeof *mat);
    int ok = per && mat;
#define INKC_FITW(w, In, Out) do { const Wt *w_ = (w); int f_ = ink_vk_fmt(w_), g_ = w_->q4 ? w_->gs : 0; \
        if (!f_) ok = 0; else { per[i] += vkc_fit_tensor(f_, (In), (Out), g_); mat[i] += coli_vk_tensor_payload(f_, (In), (Out), g_); } } while (0)
    for (int i = 0; i < L && ok; i++) {
        const Layer *l = &m->L[i];
        int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, I = c->moe_inter;
        INKC_FITW(&l->q, D, H * hd); INKC_FITW(&l->k, D, kvo); INKC_FITW(&l->v, D, kvo);
        INKC_FITW(&l->r, D, H * dr); INKC_FITW(&l->o, H * hd, D);
        if (!c->sparse[i]) { INKC_FITW(&l->dg, D, c->dense_inter); INKC_FITW(&l->du, D, c->dense_inter); INKC_FITW(&l->dd, c->dense_inter, D); }
        else {
            per[i] += vkc_fit_tensor(10, D, E + ns, 0); mat[i] += coli_vk_tensor_payload(10, D, E + ns, 0);
            for (int j = 0; j < ns; j++) { INKC_FITW(&l->sh_g, D, I); INKC_FITW(&l->sh_u, D, I); INKC_FITW(&l->sh_d, I, D); }
        }
        per[i] += ((size_t)2 * D + 2 * (size_t)hd + (size_t)dr * L_EXT(c, i) + (size_t)2 * kvo * CK + (size_t)2 * D * CK) * sizeof(float);
        for (int b = 0; b < 4; b++) {
            size_t cells = (size_t)inkc_cs_cells(c, b, i);
            per[i] += vkc_fit_buf((cells ? cells : 1) * sizeof(float)) + cells * sizeof(float);
        }
        int rows0 = c->window > 0 ? c->window : 1;
        per[i] += 2 * vkc_fit_buf((size_t)KV * rows0 * hd * sizeof(float));
    }
#undef INKC_FITW
    if (!ink_vk_fmt(&m->lm_head)) ok = 0;
    if (!ok) {   /* a matrix the shaders do not read as this CPU does: the chain declines as before */
        free(per); free(mat);
        return;
    }
    const Wt *h = &m->lm_head;
    int hf = ink_vk_fmt(h);
    size_t tail = vkc_fit_tensor(hf, D, c->unpad_vocab, h->q4 ? h->gs : 0);
    InkChain g; memset(&g, 0, sizeof g);
    inkc_geom(&g, m);
    int R = vkc_fit_rows(256);
    g_inkc_count = 0; g_inkc_fitcount = 1;
    inkc_bufs(&g, m, R);
    size_t fixed = (size_t)g_inkc_count + vkc_fit_buf((size_t)R * D * sizeof(float)) + vkc_fit_buf((size_t)D * sizeof(float));
    g_inkc_count = -1; g_inkc_fitcount = 0;
    vkc_fit("inkling", L, per, mat, fixed, tail, &g_inkc_fit);
    /* the layers the primary leaves, on COLI_VK_DEV2's device: a fit of its own from layer
     * n0 with that device's free memory, its pipelines up now (inkc_place_init places its
     * layers; the head goes with the last one when there is room) */
    int n0 = g_inkc_fit.n;
    if (vkc_fit_partial(&g_inkc_fit) && n0 > 0 && n0 < L && inkc_dev2_wanted() && getenv("COLI_VK_DEV2") &&
        coli_vk_dev2_open_env()) {
        vkc_device(1);
        int n2 = vkc_fit("inkling dev2", L - n0, per + n0, mat + n0, fixed, tail, &g_inkc_fit2);
        if (n2 > 0 && !vkc_init()) {
            fprintf(stderr, "[VK] inkling chain: the second device's pipelines did not come up; its layers stay on the CPU\n");
            n2 = 0;
        }
        g_inkc_fit2_on = n2 > 0;
        if (!g_inkc_fit2_on) g_inkc_fit2.tail = 0;
        vkc_device(0);
    }
    free(per); free(mat);
}
/* What a partial chain still allocates on the device after its layers went up (the
 * tier's dense_bytes): their state and parameters, the layer bytes less the matrices. */
static size_t inkc_fit_later(const Model *m) {
    const Cfg *c = &m->c; size_t b = 0;
    for (int i = 0; i < g_inkc_fit.n; i++) {
        const Layer *l = &m->L[i];
        int H = L_HEADS(c, i), hd = L_HD(c, i), kvo = L_KV(c, i) * hd, D = c->hidden, I = c->moe_inter;
        size_t w = 0;
#define INKC_LW(x, In, Out) do { const Wt *x_ = (x); InkVk *v_ = x_->vk; int f_ = v_ && v_->gone ? v_->fmt : ink_vk_fmt(x_); \
        w += vkc_fit_tensor(f_, (In), (Out), v_ && v_->gone ? v_->gs : x_->q4 ? x_->gs : 0); } while (0)
        INKC_LW(&l->q, D, H * hd); INKC_LW(&l->k, D, kvo); INKC_LW(&l->v, D, kvo);
        INKC_LW(&l->r, D, H * c->d_rel); INKC_LW(&l->o, H * hd, D);
        if (!c->sparse[i]) { INKC_LW(&l->dg, D, c->dense_inter); INKC_LW(&l->du, D, c->dense_inter); INKC_LW(&l->dd, c->dense_inter, D); }
        else {
            w += vkc_fit_tensor(10, D, c->n_experts + c->n_shared, 0);
            for (int j = 0; j < c->n_shared; j++) { INKC_LW(&l->sh_g, D, I); INKC_LW(&l->sh_u, D, I); INKC_LW(&l->sh_d, I, D); }
        }
#undef INKC_LW
        b += g_inkc_fit.per[i] > w ? g_inkc_fit.per[i] - w : 0;
    }
    return b;
}
/* A fit's layers up at start-up (a partial chain, or the dense weights on the device only):
 * layer by layer, each whole or not at all; dropped: with COLI_VK_DENSE_HOST, each layer's
 * host copies given back once all of it is on the device (counted there). */
static void inkc_place_init(Model *m, int *dropped) {
    InkChain *ch = g_inkc_fit.n ? inkc_new(m) : NULL;
    if (g_inkc_fit.n && !ch) {
        inkc_refuse_from(m, 0);
        vkc_fit_shrink("inkling", &g_inkc_fit, 0, "no host memory for the chain's tables");
    }
    int n0 = g_inkc_fit.n;
    for (int i = 0; ch && i < g_inkc_fit.n; i++) {
        const char *what = NULL;
        if (!inkc_place_layer(m, ch, i, &what)) { inkc_shrink(m, ch, i, what); break; }
        if (dropped) ink_dho_drop_layer(m, i, dropped);
        vkc_fit_mark(&g_inkc_fit, i);
    }
    int end = g_inkc_fit.n;
    if (vkc_fit_partial(&g_inkc_fit)) inkc_refuse_from(m, end);   /* the CPU's layers and lm_head stay there */
    vkc_fit_placed("inkling", &g_inkc_fit);
    if (g_inkc_fit2_on && g_inkc_fit.n < n0) {
        /* the primary placed fewer layers than its fit: the second device's would not follow
         * them, so they stay on the CPU too */
        fprintf(stderr, "[VK] inkling chain: the primary device stopped before layer %d; layers %d..%d stay on the CPU, "
                        "not on the second device\n", n0, n0, n0 + g_inkc_fit2.n - 1);
        g_inkc_fit2_on = 0;
        vkc_device(1); vkc_shutdown(); vkc_device(0);
    } else if (g_inkc_fit2_on) {   /* the second device's layers: their own copies there, host copies kept */
        vkc_device(1);
        InkChain *c2 = inkc_new_dev(m, 1);
        for (int i = n0; c2 && i < n0 + g_inkc_fit2.n; i++) {
            const char *what = NULL;
            if (!inkc_place_layer(m, c2, i, &what)) { inkc_shrink(m, c2, i, what); break; }
            vkc_fit_mark(&g_inkc_fit2, i - n0);
        }
        if (!c2) g_inkc_fit2.n = 0;
        vkc_fit_placed("inkling dev2", &g_inkc_fit2);
        if (!g_inkc_fit2.n) { g_inkc_fit2_on = 0; vkc_shutdown(); }
        vkc_device(0);
    }
    g_inkc_placed = 1;
}

/* The model's parameters on the device, its tensors resolved; NULL = the chain cannot run.
 * With a fit the first n layers only (placed at start-up when the chain is partial or the
 * host copies go; here at the first forward otherwise), and lm_head when the tail is on. */
static void inkc_free_own(Model *m, InkChain *ch);
/* d = 1: the second device's chain (its layers placed by inkc_place_init; device 1 current). */
static InkChain *inkc_setup_dev(Model *m, int d) {
    InkChain *ch = inkc_of(m, d);
    if (ch && ch->tried) return ch->ok ? ch : NULL;
    if (d && !g_inkc_fit2_on) return NULL;
    if (!ch && !(ch = inkc_new_dev(m, d))) return NULL;
    ch->tried = 1;
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, CK = c->conv_k, dr = c->d_rel;
    if (!inkc_geometry_ok(m, 1)) return NULL;
    if (!vkc_sconv_ready() || !vkc_relattn_ready()) {
        fprintf(stderr, "[VK] inkling chain: chain_sconv.spv or chain_relattn.spv is missing; per-matrix path\n");
        inkc_free_own(m, ch);
        return NULL;
    }
    /* the tensors: the per-matrix path's device copies where it keeps them */
    const char *what = NULL;
    VkcFit *fitp = d ? &g_inkc_fit2 : &g_inkc_fit;
    int fit = fitp->L > 0, lo = ch->lo;
    for (int i = lo; i < lo + (d ? g_inkc_fit2.n : inkc_layers(m)) && !what; i++) {
        if (ch->t_q[i]) continue;   /* placed at start-up */
        if (!inkc_place_layer(m, ch, i, &what)) {
            if (fit) { inkc_shrink(m, ch, i, what); what = NULL; break; }
            continue;
        }
        if (fit) vkc_fit_mark(fitp, i - lo);
    }
    int nl = d ? g_inkc_fit2.n : inkc_layers(m);
    if (fit && !d && !g_inkc_placed) { vkc_fit_placed("inkling", &g_inkc_fit); g_inkc_placed = 1; }   /* placed here, lm_head next */
    if (fit && !d && vkc_fit_partial(&g_inkc_fit)) inkc_refuse_from(m, lo + nl);
    if (!what && lo + nl == L && (!fit || fitp->tail) && !(ch->t_lm = inkc_tensor(d, m->lm_head, D, c->unpad_vocab))) {
        if (!fit) what = "lm_head";
        else {   /* the fit's tail did not go up: every layer here, the head on the CPU */
            fitp->tail = 0;
            inkc_refuse_from(m, L);
            fprintf(stderr, "[VK] inkling chain: lm_head did not reach the device; it runs on the CPU, the layers here\n");
        }
    }
    if (what || !nl) {
        if (what) {
            int bf16 = m->L[0].q.h && !m->L[0].q.q4 && !ink_vk_fmt(&m->L[0].q);
            fprintf(stderr, "[VK] inkling chain: %s did not reach the device%s; per-matrix path\n", what,
                    bf16 ? " (bf16: this CPU's bf16 dot rounds the activations, the device would not, so bf16 stays on the CPU)" : "");
        }
        inkc_free_own(m, ch);
        return NULL;
    }
    ch->nl = nl; ch->full = ch->t_lm != NULL;
    /* the parameter arena of the device's layers: offsets, then one upload */
    size_t n = 0;
    for (int i = lo; i < lo + nl; i++) {
        int hd = L_HD(c, i), kvo = L_KV(c, i) * hd;
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D;
        ch->o_qn[i] = n; n += hd; ch->o_kn[i] = n; n += hd;
        ch->o_relp[i] = n; n += (size_t)dr * L_EXT(c, i);
        ch->o_cw[0][i] = n; n += (size_t)kvo * CK; ch->o_cw[1][i] = n; n += (size_t)kvo * CK;
        ch->o_cw[2][i] = n; n += (size_t)D * CK; ch->o_cw[3][i] = n; n += (size_t)D * CK;
    }
    ch->o_final = n; n += D;
    float *arena = calloc(n, sizeof(float));
    if (!arena) { inkc_free_own(m, ch); return NULL; }
    for (int i = lo; i < lo + nl; i++) {
        Layer *l = &m->L[i];
        int hd = L_HD(c, i), kvo = L_KV(c, i) * hd;
        memcpy(arena + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(arena + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(arena + ch->o_qn[i], l->qn, hd * sizeof(float));
        memcpy(arena + ch->o_kn[i], l->kn, hd * sizeof(float));
        memcpy(arena + ch->o_relp[i], l->relp, (size_t)dr * L_EXT(c, i) * sizeof(float));
        memcpy(arena + ch->o_cw[0][i], l->k_cw, (size_t)kvo * CK * sizeof(float));
        memcpy(arena + ch->o_cw[1][i], l->v_cw, (size_t)kvo * CK * sizeof(float));
        memcpy(arena + ch->o_cw[2][i], l->a_cw, (size_t)D * CK * sizeof(float));
        memcpy(arena + ch->o_cw[3][i], l->m_cw, (size_t)D * CK * sizeof(float));
    }
    memcpy(arena + ch->o_final, m->final_norm, D * sizeof(float));
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, n * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) {
        inkc_free_own(m, ch);
        if (d && fit) {   /* the second device's parameter buffer: its layers stay on the CPU */
            vkc_fit_shrink(inkc_name(ch), fitp, 0, vkc_lost() ? "the device was lost" : "its parameter buffer was refused");
            g_inkc_fit2_on = 0;
            vkc_shutdown();
        }
        return NULL;
    }
    /* geometry of the scratch; the frames: a MoE layer ends one (its host step) */
    inkc_geom(ch, m);
    for (int i = lo; i < lo + nl; i++)
        for (int b = 0; b < 4; b++) {
            ch->o_csd[(size_t)b * L + i] = ch->csd_n; ch->csd_n += (size_t)inkc_cs_cells(c, b, i);
            if (!(ch->ring[b][i] = vkc_buf((size_t)(inkc_cs_cells(c, b, i) > 0 ? inkc_cs_cells(c, b, i) : 1) * sizeof(float), VKC_DEV))) {
                inkc_free_own(m, ch);
                return NULL;
            }
        }
    for (int i = 0; i < L; i++) ch->kli[i] = i >= lo && i < lo + nl && !c->local[i] ? ch->nglob++ : -1;
    ch->cs_where = INKC_HOST; ch->host_zero = 0;
    ch->ok = 1;
    int nsp = 0; for (int i = lo; i < lo + nl; i++) nsp += c->sparse[i];
    if (d) fprintf(stderr, "[VK] inkling chain: layers %d..%d on the second device (%d MoE)%s, %.1f MiB of parameters\n",
                   lo, lo + nl - 1, nsp, ch->full ? ", lm_head too" : "", n * 4 / 1048576.0);
    else fprintf(stderr, "[VK] inkling chain: %d layers on the device (%d MoE), %.1f MiB of parameters\n",
                 nl, nsp, n * 4 / 1048576.0);
    return ch;
}
static InkChain *inkc_setup(Model *m) { return inkc_setup_dev(m, 0); }
/* The chain declined after its layers went up: the copies only it holds (its own, not the
 * per-matrix path's tables; the second device's are all its own) are freed, with whatever device memory it took. */
static void inkc_free_own(Model *m, InkChain *ch) {
    if (vkc_ready()) vkc_finish();
    int L = m->c.n_layers, nsl = m->c.n_shared > 0 ? m->c.n_shared : 1;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        ColiVkTensor **t[] = {&ch->t_q[i], &ch->t_k[i], &ch->t_v[i], &ch->t_r[i], &ch->t_o[i], &ch->t_dg[i], &ch->t_du[i], &ch->t_dd[i]};
        Wt *w[] = {&l->q, &l->k, &l->v, &l->r, &l->o, &l->dg, &l->du, &l->dd};
        for (size_t k = 0; k < sizeof t / sizeof t[0]; k++) { if (*t[k] && (ch->d || !w[k]->vk)) coli_vk_tensor_free(*t[k]); *t[k] = NULL; }
        for (int j = 0; j < nsl; j++) {
            ColiVkTensor **s[3] = {&ch->t_sg[(size_t)i * nsl + j], &ch->t_su[(size_t)i * nsl + j], &ch->t_sd[(size_t)i * nsl + j]};
            Wt *sw[3] = {&l->sh_g, &l->sh_u, &l->sh_d};
            for (int k = 0; k < 3; k++) { if (*s[k] && (ch->d || !sw[k]->vk)) coli_vk_tensor_free(*s[k]); *s[k] = NULL; }
        }
        if (ch->t_router[i]) coli_vk_tensor_free(ch->t_router[i]);
        ch->t_router[i] = NULL;
        for (int b = 0; b < 4; b++) { vkc_free(ch->ring[b][i]); ch->ring[b][i] = NULL; }
    }
    if (ch->t_lm && (ch->d || !m->lm_head.vk)) coli_vk_tensor_free(ch->t_lm);
    ch->t_lm = NULL;
    vkc_free(ch->prm); ch->prm = NULL;
}

/* inkc_res counts instead of reserving while g_inkc_count >= 0 (the chunk's sizing; the
 * fit's, g_inkc_fitcount, each buffer as the pools round it) */
static int inkc_res(VkcBuf **b, size_t floats, int kind) {
    if (g_inkc_count >= 0) {
        size_t by = (floats ? floats : 1) * sizeof(float);
        g_inkc_count += (long long)(g_inkc_fitcount ? vkc_fit_buf(by) : by);
        return 1;
    }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}
/* Prompt rows per chunk (vkc_chunk_rows): the chain's scratch a row, counted from the
 * reservations, and the routed experts' outputs (the tier's rows, the CPU's
 * contributions, and the host's sum). */
static int inkc_chunk_rows(InkChain *ch, Model *m) {
    size_t vs = ch->vs_off;
    g_inkc_count = 0; inkc_bufs(ch, m, 1); long long b1 = g_inkc_count;
    g_inkc_count = 0; inkc_bufs(ch, m, 2); long long b2 = g_inkc_count;
    g_inkc_count = -1; ch->vs_off = vs;
    size_t row = (size_t)(b2 - b1) + (size_t)(2 * m->c.topk + 2) * m->c.hidden * sizeof(float);
    if (ch->ks.on) {
        int heads = 1, dim = 1;
        for (int i = 0; i < m->c.n_layers; i++) if (ch->kli[i] >= 0) {
            if (L_HEADS(&m->c, i) > heads) heads = L_HEADS(&m->c, i);
            if (L_HD(&m->c, i) > dim) dim = L_HD(&m->c, i);
        }
        row += (3 * (size_t)heads * (dim + 2) + ch->qo_max + ch->ro_max) * sizeof(float);
    }
    return vkc_chunk_rows(inkc_name(ch), row);
}
/* the scratch buffers for `rows` rows */
static int inkc_bufs(InkChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, ET = c->n_experts + c->n_shared, ns = c->n_shared > 0 ? c->n_shared : 1;
    size_t r = (size_t)rows;
    ch->vs_off = (r * ch->kvo_max + 63) & ~(size_t)63;
    return inkc_res(&ch->x, r * D, VKC_DEV) && inkc_res(&ch->nrm, r * D, VKC_DEV) && inkc_res(&ch->tmp, r * D, VKC_DEV) &&
             inkc_res(&ch->q, r * ch->qo_max, VKC_DEV) && inkc_res(&ch->kvs, ch->vs_off + r * ch->kvo_max, VKC_DEV) &&
             inkc_res(&ch->r, r * ch->ro_max, VKC_DEV) && inkc_res(&ch->ctx, r * ch->qo_max, VKC_DEV) &&
             inkc_res(&ch->h2, r * D, VKC_DEV) && inkc_res(&ch->lg, r * ET, VKC_DEV) &&
             inkc_res(&ch->gs, r * ch->mi_max, VKC_DEV) && inkc_res(&ch->us, r * ch->mi_max, VKC_DEV) &&
             inkc_res(&ch->sh, (size_t)ns * r * D, VKC_DEV) && inkc_res(&ch->mlp, r * D, VKC_DEV) && inkc_res(&ch->fin, D, VKC_DEV) &&
             inkc_res(&ch->h2d, r * D, VKC_DOWN) && inkc_res(&ch->lgd, r * ET, VKC_DOWN) &&
             inkc_res(&ch->kvd, (size_t)ch->nslot * 2 * r * ch->kvo_max, VKC_DOWN) && inkc_res(&ch->csd, ch->csd_n, VKC_DOWN) &&
             inkc_res(&ch->outd, c->unpad_vocab, VKC_DOWN) && inkc_res(&ch->routed, r * D, VKC_UP) &&
             inkc_res(&ch->shw, (size_t)ns * r, VKC_UP) && inkc_res(&ch->tau, 2 * r, VKC_UP);
}
/* Plan and allocate the KV mirror before sizing a prompt chunk. */
static int inkc_mirror(InkChain *ch, Model *m) {
    Cfg *c = &m->c;
    if (ch->max_t != m->max_t || ch->hostK != m->K) {   /* a cache allocated anew: mirror it again */
        /* the global layers' mirrors: whole, or split past the device's budget */
        size_t row = 0, held = 0;
        for (int i = 0; i < c->n_layers; i++) {
            if (ch->kli[i] < 0) continue;
            size_t b = (size_t)2 * L_KV(c, i) * L_HD(c, i) * sizeof(float);
            if (b > row) row = b;
            held += b * (size_t)ch->drows[i];
        }
        int plan = ch->nglob ? vkc_kv_plan(&ch->ks, inkc_name(ch), ch->nglob, row, m->max_t, 1, 0, held) : 1;
        if (!plan) { ch->max_t = 0; ch->hostK = NULL; return 0; }
        for (int i = ch->lo; i < ch->lo + ch->nl; i++) {
            vkc_free(ch->kc[i]); vkc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
            int cap = kv_ring_rows(c, i, m->max_t), hd = L_HD(c, i), KV = L_KV(c, i);
            int split = ch->ks.on && ch->kli[i] >= 0;
            ch->cap[i] = cap;
            ch->drows[i] = split ? ch->ks.rows : cap;
            /* a ring: every row, whatever positions it holds; a whole cache: the positions held;
             * a split layer: its watermark (vkc_kv_plan reset it) */
            ch->dlo[i] = 0; ch->dhi[i] = split ? 0 : cap < m->max_t ? m->max_t : m->kv_len;
            ch->kc[i] = vkc_buf((size_t)KV * ch->drows[i] * hd * sizeof(float), VKC_DEV);
            ch->vc[i] = vkc_buf((size_t)KV * ch->drows[i] * hd * sizeof(float), VKC_DEV);
            if (!ch->kc[i] || !ch->vc[i]) { ch->max_t = 0; ch->hostK = NULL; return 0; }
        }
        ch->max_t = m->max_t; ch->hostK = m->K;
    }
    return 1;
}

/* scratch for `rows` rows, and the K/V mirrors at the host's layout */
static int inkc_scratch(InkChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, ns = c->n_shared > 0 ? c->n_shared : 1;
    size_t r = (size_t)rows;
    if (!inkc_bufs(ch, m, rows)) return 0;
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, r * D * sizeof(float)), *hw = hr ? realloc(ch->host_w, (size_t)ns * r * sizeof(float)) : NULL;
        if (hr) ch->host_routed = hr;
        if (!hr || !hw) return 0;
        ch->host_w = hw; ch->rows = rows;
    }
    if (ch->ks.on) {
        int hmax = 1, hdm = 1;
        for (int i = 0; i < c->n_layers; i++) if (ch->kli[i] >= 0) {
            if (L_HEADS(c, i) > hmax) hmax = L_HEADS(c, i);
            if (L_HD(c, i) > hdm) hdm = L_HD(c, i);
        }
        if (!vkc_kv_parts(&ch->ks, r * hmax * (hdm + 2))) return 0;
    }
    return 1;
}

/* ---- state between the host and the device ----------------------------------- */
/* The host wrote its convolution states (zero: state_reset's zeros). */
static void inkc_host_wrote(Model *m, int zero) {
    for (int d = 0; d < 2; d++) {
        InkChain *ch = inkc_of(m, d);
        if (!ch || !ch->ok) continue;
        ch->cs_where = INKC_HOST; ch->host_zero = zero;
    }
}
/* A CPU step of S rows from pos_base over one chain's layers: their K/V rows and states are
 * the host's alone. */
static void inkc_cpu_one(InkChain *ch, int pos_base, int S) {
    if (!ch || !ch->ok) return;
    ch->cs_where = INKC_HOST; ch->host_zero = 0;
    for (int i = ch->lo; i < ch->lo + ch->nl; i++) {   /* the device's layers: the CPU's keep their state on the host */
        if (ch->ks.on && ch->kli[i] >= 0) { vkc_kv_lower(&ch->ks, ch->kli[i], pos_base); continue; }
        if (ch->dlo[i] >= ch->dhi[i]) { ch->dlo[i] = pos_base; ch->dhi[i] = pos_base + S; continue; }
        if (pos_base < ch->dlo[i]) ch->dlo[i] = pos_base;
        if (pos_base + S > ch->dhi[i]) ch->dhi[i] = pos_base + S;
    }
}
static void inkc_cpu_step(Model *m, int pos_base, int S) {
    for (int d = 0; d < 2; d++) inkc_cpu_one(inkc_of(m, d), pos_base, S);
}
/* Record the uploads that make the device state the host's: the convolution states if
 * the host's are newer, and the rows of the last `cap` positions the host wrote alone;
 * a split layer: its window placed for the n rows from pos_base, its resident rows below
 * pos_base uploaded. */
static int inkc_push_state(InkChain *ch, Model *m, int pos_base, int n) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->cs_where == INKC_HOST) {
        for (int i = ch->lo; i < ch->lo + ch->nl && ok; i++) for (int b = 0; b < 4 && ok; b++) {
            size_t nc = (size_t)inkc_cs_cells(c, b, i);
            ok = ch->host_zero ? vkc_zero(ch->ring[b][i], 0, nc) : vkc_write(ch->ring[b][i], 0, m->cs[b][i], nc * sizeof(float));
        }
        ch->cs_where = INKC_BOTH;
    }
    for (int i = ch->lo; i < ch->lo + ch->nl && ok; i++) {
        if (ch->ks.on && ch->kli[i] >= 0) {
            int KV = L_KV(c, i), hd = L_HD(c, i);
            VkcKvPart pt[2] = {{KV, hd, m->K[i], (size_t)ch->cap[i] * hd, ch->kc[i], 0},
                               {KV, hd, m->V[i], (size_t)ch->cap[i] * hd, ch->vc[i], 0}};
            vkc_kv_place(&ch->ks, ch->kli[i], pos_base, n);
            ok = vkc_kv_push(&ch->ks, ch->kli[i], pt, 2, pos_base);
            continue;
        }
        if (ch->dlo[i] >= ch->dhi[i]) continue;
        int cap = ch->cap[i], hd = L_HD(c, i), KV = L_KV(c, i);
        int t0 = ch->dhi[i] - cap > ch->dlo[i] ? ch->dhi[i] - cap : ch->dlo[i];
        for (int t = t0; t < ch->dhi[i] && ok; ) {       /* a run of rows up to the ring's end */
            int row = t % cap, run = ch->dhi[i] - t < cap - row ? ch->dhi[i] - t : cap - row;
            for (int h = 0; h < KV && ok; h++) {
                size_t off = ((size_t)h * cap + row) * hd;
                ok = vkc_write(ch->kc[i], off, m->K[i] + off, (size_t)run * hd * sizeof(float)) &&
                     vkc_write(ch->vc[i], off, m->V[i] + off, (size_t)run * hd * sizeof(float));
            }
            t += run;
        }
        ch->dlo[i] = ch->dhi[i] = 0;
    }
    return ok;
}

/* The device was lost with the newest state on it. The host's K/V rows below pos_base
 * may already hold some of the lost step's (a ring wrapped by it), and its convolution
 * states are the step's start: rebuild both on the CPU from the `upto` positions the
 * prefix record names, and leave the chain off. */
static void inkc_recover(Model *m, int upto) {
    InkChain *ch = (InkChain *)m->vkchain, *ch2 = (InkChain *)m->vkchain2;
    Cfg *c = &m->c; int D = c->hidden, nl = ch ? ch->nl : c->n_layers;
    if (ch && ch->ok && ch2 && ch2->ok) nl += ch2->nl;   /* the second device's layers ran in that forward too */
    g_vk_chain = 0;
    if (ch) ch->failed = 1;
    if (ch2) ch2->failed = 1;
    /* the device's layers only: a partial chain's CPU layers hold their state on the host,
     * whole up to the lost step (they run after the device's, once per forward) */
    for (int i = 0; i < nl; i++)
        for (int b = 0; b < 4; b++) memset(m->cs[b][i], 0, (size_t)inkc_cs_cells(c, b, i) * sizeof(float));
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        inkc_fatal("the device was lost with a state its token ids do not describe (audio)");
    fprintf(stderr, "[VK] inkling chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on%s\n", upto, nl < c->n_layers ? " (the device's layers; the CPU's have theirs)" : "");
    int *ids = malloc((size_t)upto * sizeof(int));
    float *x = falloc((int64_t)upto * D);
    if (!ids) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    memcpy(ids, m->kvp.fed, (size_t)upto * sizeof(int));
    for (int s = 0; s < upto; s++) {
        wt_row_f32(m->embed, (int64_t)ids[s] * D, x + (int64_t)s * D, D);
        if (m->embed_norm) rmsnorm_row(x + (int64_t)s * D, x + (int64_t)s * D, m->embed_norm, D, c->eps);
    }
    inkling_layers_forward_range(m, x, upto, 0, 0, nl);
    free(ids); free(x);
}

/* ---- one layer's pieces ---------------------------------------------------------- */
static int inkc_norm(VkcBuf *x, size_t xo, VkcBuf *w, size_t wo, VkcBuf *y, size_t yo, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
static int inkc_conv(InkChain *ch, VkcBuf *x, size_t xo, int n, int C, int bank, int i, const Cfg *c) {
    VkcSconv p = {0, n, C, c->conv_k, (int)xo, C, (int)ch->o_cw[bank][i], 0, 0, 1.f};
    return vkc_sconv(x, ch->prm, ch->ring[bank][i], &p);
}
/* nrm in; the attention block's o_proj rows in tmp; the new K/V rows into the device
 * ring and into the frame's region of kvd */
static int inkc_attention(InkChain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, dr = c->d_rel;
    int cap = ch->cap[i], local = c->local[i];
    size_t vs = ch->vs_off;
    int ok = vkc_matmul(ch->t_q[i], ch->nrm, 0, ch->q, 0, n) && vkc_matmul(ch->t_k[i], ch->nrm, 0, ch->kvs, 0, n) &&
             vkc_matmul(ch->t_v[i], ch->nrm, 0, ch->kvs, vs, n) && vkc_matmul(ch->t_r[i], ch->nrm, 0, ch->r, 0, n) &&
             inkc_conv(ch, ch->kvs, 0, n, kvo, 0, i, c) && inkc_conv(ch, ch->kvs, vs, n, kvo, 1, i, c);
    VkcNorm qn = {n * H, hd, H, 0, H * hd, hd, 0, H * hd, hd, (int)ch->o_qn[i], 0, 0, c->eps, 1.f};
    VkcNorm kn = {n * KV, hd, KV, 0, kvo, hd, 0, kvo, hd, (int)ch->o_kn[i], 0, 0, c->eps, 1.f};
    ok = ok && vkc_norm(ch->q, ch->prm, ch->q, &qn) && vkc_norm(ch->kvs, ch->prm, ch->kvs, &kn);
    if (ok && ch->ks.on && ch->kli[i] >= 0) {
        /* the split: the step's rows into their window slots first (the window holds every
         * row of a step), then the attention over the device's rows and the host's */
        int li = ch->kli[i];
        VkcKvPart pk = {KV, hd, m->K[i], (size_t)cap * hd, ch->kc[i], 0}, pv = {KV, hd, m->V[i], (size_t)cap * hd, ch->vc[i], 0};
        size_t ko = (size_t)ch->slot[i] * 2 * ch->rows * ch->kvo_max;
        VkcKvRel a = {&ch->ks, li, ch->q, ch->kc[i], ch->vc[i], ch->r, ch->tau, ch->prm, ch->ctx, n, H, KV, hd, pb,
                      L_EXT(c, i), dr, H * hd, H * dr, (int)ch->o_relp[i], 0, H * hd, 1.f / (float)hd,
                      m->K[i], m->V[i], (size_t)cap * hd, (size_t)hd, (size_t)cap * hd, (size_t)hd,
                      (const float *)vkc_ptr(ch->tau), m->L[i].relp, 0};
        return vkc_kv_store(&ch->ks, li, &pk, ch->kvs, 0, (size_t)kvo, (size_t)hd, pb, n) &&
               vkc_kv_store(&ch->ks, li, &pv, ch->kvs, vs, (size_t)kvo, (size_t)hd, pb, n) &&
               vkc_copy(ch->kvd, ko, ch->kvs, 0, (size_t)n * kvo) &&
               vkc_copy(ch->kvd, ko + (size_t)ch->rows * ch->kvo_max, ch->kvs, vs, (size_t)n * kvo) &&
               vkc_kv_rel(&a) && vkc_matmul(ch->t_o[i], ch->ctx, 0, ch->tmp, 0, n);
    }
    VkcRelAttn a = {n, H, KV, hd, pb, cap, local ? c->window : 0, L_EXT(c, i), dr,
                    0, H * hd, 0, H * hd, 0, 0, 0, (int)vs, kvo, 0, H * dr, (int)ch->o_relp[i], local ? ch->rows : 0,
                    1.f / (float)hd};
    ok = ok && vkc_relattn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->kvs, ch->r, ch->prm, ch->tau, &a);
    if (!ok) return 0;
    /* the step's rows into the ring after the attention read them: the last `cap` of them */
    int s0 = n > cap ? n - cap : 0, nr = (n - s0) * KV;
    VkcRegion *rk = malloc(sizeof *rk * (size_t)nr * 2), *rv = rk ? rk + nr : NULL;
    if (!rk) return 0;
    for (int s = s0; s < n; s++) for (int h = 0; h < KV; h++) {
        size_t dst = ((size_t)h * cap + (size_t)(pb + s) % cap) * hd, src = (size_t)s * kvo + (size_t)h * hd;
        rk[(s - s0) * KV + h] = (VkcRegion){dst, src, (size_t)hd};
        rv[(s - s0) * KV + h] = (VkcRegion){dst, vs + src, (size_t)hd};
    }
    size_t ko = (size_t)ch->slot[i] * 2 * ch->rows * ch->kvo_max;
    ok = vkc_copy_regions(ch->kc[i], ch->kvs, rk, nr) && vkc_copy_regions(ch->vc[i], ch->kvs, rv, nr) &&
         vkc_copy(ch->kvd, ko, ch->kvs, 0, (size_t)n * kvo) &&
         vkc_copy(ch->kvd, ko + (size_t)ch->rows * ch->kvo_max, ch->kvs, vs, (size_t)n * kvo);
    free(rk);
    return ok && vkc_matmul(ch->t_o[i], ch->ctx, 0, ch->tmp, 0, n);
}
/* The rows a frame's layers [i0, i1] wrote, into the host's cache (attention()'s rows). */
static void inkc_kv_down(InkChain *ch, Model *m, int i0, int i1, int n, int pb) {
    Cfg *c = &m->c;
    for (int i = i0; i <= i1; i++) {
        int KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, cap = ch->cap[i], s0 = n > cap ? n - cap : 0;
        const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->slot[i] * 2 * ch->rows * ch->kvo_max;
        const float *vv = kv + (size_t)ch->rows * ch->kvo_max;
        for (int s = s0; s < n; s++) for (int h = 0; h < KV; h++) {
            size_t dst = ((size_t)h * cap + (size_t)(pb + s) % cap) * hd, src = (size_t)s * kvo + (size_t)h * hd;
            memcpy(m->K[i] + dst, kv + src, hd * sizeof(float));
            memcpy(m->V[i] + dst, vv + src, hd * sizeof(float));
        }
    }
}
/* the MLP output in mlp: its short convolution, then the residual add */
static int inkc_mlp_out(InkChain *ch, int i, int n, const Cfg *c) {
    VkcEw add = {VKC_EW_ADD, n * c->hidden, c->hidden, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return inkc_conv(ch, ch->mlp, 0, n, c->hidden, 3, i, c) && vkc_ew(ch->x, ch->x, ch->mlp, NULL, NULL, &add);
}
/* a dense-MLP layer's MLP from h2, as dense_mlp() */
static int inkc_dense(InkChain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c; int DI = c->dense_inter;
    VkcEw sw = {VKC_EW_SWIGLU, n * DI, DI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    VkcSconv sc = {1, 0, 0, 0, 0, 0, 0, 0, n * c->hidden, l->dgs};
    return vkc_matmul(ch->t_dg[i], ch->h2, 0, ch->gs, 0, n) && vkc_matmul(ch->t_du[i], ch->h2, 0, ch->us, 0, n) &&
           vkc_ew(ch->gs, ch->gs, ch->us, NULL, NULL, &sw) && vkc_matmul(ch->t_dd[i], ch->gs, 0, ch->mlp, 0, n) &&
           vkc_sconv(ch->mlp, NULL, NULL, &sc) && inkc_mlp_out(ch, i, n, c);
}
/* frame A2: each shared expert's output from h2, unweighted, into sh[j] */
static int inkc_shared(InkChain *ch, Model *m, int i, int n) {
    Cfg *c = &m->c; int I = c->moe_inter, ns = c->n_shared, nsl = ns > 0 ? ns : 1, ok = 1;
    VkcEw sw = {VKC_EW_SWIGLU, n * I, I, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    for (int j = 0; j < ns && ok; j++)
        ok = vkc_matmul(ch->t_sg[(size_t)i * nsl + j], ch->h2, 0, ch->gs, 0, n) &&
             vkc_matmul(ch->t_su[(size_t)i * nsl + j], ch->h2, 0, ch->us, 0, n) &&
             vkc_ew(ch->gs, ch->gs, ch->us, NULL, NULL, &sw) &&
             vkc_matmul(ch->t_sd[(size_t)i * nsl + j], ch->gs, 0, ch->sh, (size_t)j * ch->rows * c->hidden, n);
    return ok;
}
/* a MoE layer's output joining the residual, in moe()'s order: the routed sum, then each
 * shared expert times its weight; then the MLP's short convolution and the add */
static int inkc_join(InkChain *ch, Model *m, int i, int n) {
    Cfg *c = &m->c; int D = c->hidden, ok = vkc_copy(ch->mlp, 0, ch->routed, 0, (size_t)n * D);
    for (int j = 0; j < c->n_shared && ok; j++) {
        VkcEw p = {VKC_EW_HC_APPLY, n * D, D, 1, 0, 1, 0, j * ch->rows, j * ch->rows * D, 0, 0, 1.f};
        ok = vkc_ew(ch->mlp, ch->shw, ch->sh, NULL, NULL, &p);
    }
    return ok && inkc_mlp_out(ch, i, n, c);
}

/* Every layer for S rows from host rows xh, the last row's logits into `logit`; xh gets
 * the final rows back when want_x. 0 = not taken: xh is as it was, and the caller runs
 * the step on the CPU (after inkc_recover when the device was lost). 1 = taken. 2 = a
 * partial chain's share taken: its first ch->nl layers ran here for every row, xh holds
 * the residual after the last of them, and the caller runs the CPU's layers (from
 * inkc_layers) and lm_head on it. */
/* One chain's layers (lo..lo+nl-1, on its device) from the residual rows in xh: 1 with the
 * head, 2 with the residual after its last layer in xh, 0 not taken (*lost = 1: a frame
 * failed, the state rebuilt on the CPU and the chains off; else nothing changed). */
static int inkc_forward_seg(Model *m, InkChain *ch, float *xh, int S, int pos_base, int want_x, float *logit, int *lost) {
    *lost = 0;
    Cfg *c = &m->c; int D = c->hidden, L = ch->nl, lo = ch->lo, hi = lo + L, ET = c->n_experts + c->n_shared, ns = c->n_shared;
    int full = ch->full, LT = c->n_layers;   /* full: the last layer and lm_head here */
    if (!full) want_x = 1;                    /* the CPU's layers read the residual */
    int mirror_ok = inkc_mirror(ch, m);
    int CH = mirror_ok ? inkc_chunk_rows(ch, m) : 1, rows = S < CH ? S : CH;
    if (ch->ks.on && rows > ch->ks.chunk) rows = ch->ks.chunk;   /* a step's rows fit the split's window */
    if (!mirror_ok || pos_base + S > m->max_t || !inkc_scratch(ch, m, rows) || (want_x && !inkc_res(&ch->xd, (size_t)rows * D, VKC_DOWN))) {
        fprintf(stderr, "[VK] inkling chain: device memory for %d rows refused; per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    float *xfin = want_x ? malloc((size_t)S * D * sizeof(float)) : NULL;
    if (want_x && !xfin) return 0;
    vkc_gemm_rows(-1);
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        if (!vkc_begin() || !vkc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !inkc_push_state(ch, m, pb, n)) goto lost;
        float *tau = (float *)vkc_ptr(ch->tau);   /* attention()'s tau: global rows, then 1 for sliding ones */
        for (int s = 0; s < n; s++) {
            float t = 1.f;
            if (c->log_floor > 0) { double en = (double)(pb + s + 1) / c->log_floor; if (en > 1.0) t = 1.f + c->log_alpha * (float)log(en); }
            tau[s] = t; tau[ch->rows + s] = 1.f;
        }
        int ok = 1, pending = 0, f0 = lo;
        for (int i = lo; i < hi && ok; i++) {
            Layer *l = &m->L[i];
            VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            if (pending) ok = inkc_join(ch, m, i - 1, n);
            ok = ok && inkc_norm(ch->x, 0, ch->prm, ch->o_in[i], ch->nrm, 0, n, D, c->eps) &&
                 inkc_attention(ch, m, i, n, pb) && inkc_conv(ch, ch->tmp, 0, n, D, 2, i, c) &&
                 vkc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add) &&
                 inkc_norm(ch->x, 0, ch->prm, ch->o_post[i], ch->h2, 0, n, D, c->eps);
            pending = 0;
            if (ok && !c->sparse[i]) { ok = inkc_dense(ch, m, l, i, n); continue; }
            ok = ok && vkc_matmul(ch->t_router[i], ch->h2, 0, ch->lg, 0, n) &&
                 vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * ET) && vkc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D);
            double t0 = now_s();
            /* A1; while it runs, the tier loads the experts this layer will likely
             * stream (a big prompt chunk only) */
            ok = ok && vkc_submit(0);
            if (ok) vkt_stream_prefetch(i, n);
            ok = ok && vkc_finish();
            m->t_attn += now_s() - t0;
            if (!ok) break;
            inkc_kv_down(ch, m, f0, i, n, pb);
            f0 = i + 1;
            /* A2: the shared experts, while the host computes the routed experts */
            ok = vkc_begin() && inkc_shared(ch, m, i, n) && vkc_submit(0);
            double t1 = now_s();
            moe_ex(m, l, i, (float *)vkc_ptr(ch->h2d), n, ch->host_routed, (const float *)vkc_ptr(ch->lgd), ch->host_w);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
            for (int j = 0; j < ns; j++)
                memcpy((float *)vkc_ptr(ch->shw) + (size_t)j * ch->rows, ch->host_w + (size_t)j * n, (size_t)n * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok && pending) ok = inkc_join(ch, m, hi - 1, n);
        if (ok && want_x) ok = vkc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D);   /* a partial chain: the handoff */
        for (int i = lo; i < hi && ok; i++) for (int b = 0; b < 4 && ok; b++)   /* the convolution states, for the host */
            ok = vkc_copy(ch->csd, ch->o_csd[(size_t)b * LT + i], ch->ring[b][i], 0, (size_t)inkc_cs_cells(c, b, i));
        int last = c0 + n == S;
        if (ok && last && full) {
            VkcSconv dv = {2, 0, 0, 0, 0, 0, 0, 0, D, c->mup};
            ok = inkc_norm(ch->x, (size_t)(n - 1) * D, ch->prm, ch->o_final, ch->fin, 0, 1, D, c->eps) &&
                 vkc_sconv(ch->fin, NULL, NULL, &dv) && vkc_matmul(ch->t_lm, ch->fin, 0, ch->outd, 0, 1);
        }
        double t0 = now_s();
        ok = ok && vkc_submit(1);
        m->t_attn += now_s() - t0;
        if (!ok) goto lost;
        if (f0 < hi) inkc_kv_down(ch, m, f0, hi - 1, n, pb);
        const float *csd = (const float *)vkc_ptr(ch->csd);
        for (int i = lo; i < hi; i++) for (int b = 0; b < 4; b++)
            memcpy(m->cs[b][i], csd + ch->o_csd[(size_t)b * LT + i], (size_t)inkc_cs_cells(c, b, i) * sizeof(float));
        ch->cs_where = INKC_BOTH; ch->host_zero = 0;
        for (int i = lo; i < hi; i++) if (ch->kli[i] >= 0) vkc_kv_done(&ch->ks, ch->kli[i], pb + n);
        if (want_x) memcpy(xfin + (size_t)c0 * D, vkc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        if (last && full) memcpy(logit, vkc_ptr(ch->outd), (size_t)c->unpad_vocab * sizeof(float));
    }
    if (want_x) { memcpy(xh, xfin, (size_t)S * D * sizeof(float)); free(xfin); }
    ch->forwards++;
    return full ? 1 : 2;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(ch->d); }
    free(xfin);
    vkc_device(0);
    inkc_recover(m, pos_base);
    *lost = 1;
    return 0;
}

/* Every layer the chains hold for S rows from host rows xh, the last row's logits into
 * `logit`; xh gets the final rows back when want_x. 0 = not taken: xh is as it was, and
 * the caller runs the step on the CPU (after inkc_recover when a device was lost). 1 =
 * taken. 2 = a partial share taken: the primary's layers, then the second device's, ran
 * here for every row, xh holds the residual after the last of them, and the caller runs
 * the CPU's layers (from inkc_ran) and lm_head on it. */
static int inkc_forward(Model *m, float *xh, int S, int pos_base, int want_x, float *logit) {
    g_inkc_ran = 0;
    if (!g_vk_chain) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && S <= 2) return 0;   /* prompts only: decode on the CPU */
    InkChain *ch = inkc_setup(m), *ch2 = NULL;
    if (!ch || ch->failed) return 0;
    if (g_inkc_fit2_on && !ch->full && ch->nl == g_inkc_fit.n) {
        vkc_device(1);
        ch2 = inkc_setup_dev(m, 1);
        vkc_device(0);
        if (ch2 && (ch2->failed || ch2->lo != ch->lo + ch->nl)) ch2 = NULL;
    }
    int lost = 0, D = m->c.hidden;
    /* the rows as they came in: the primary's residual replaces them, and if the second
     * device is lost the caller runs the step from these */
    float *keep = ch2 ? malloc((size_t)S * D * sizeof(float)) : NULL;
    if (ch2 && !keep) ch2 = NULL;
    if (keep) memcpy(keep, xh, (size_t)S * D * sizeof(float));
    int r = inkc_forward_seg(m, ch, xh, S, pos_base, ch2 ? 1 : want_x, logit, &lost);
    if (!r || !ch2) { free(keep); if (r) g_inkc_ran = ch->nl; return r; }
    g_inkc_ran = ch->nl;
    vkc_device(1);
    int r2 = inkc_forward_seg(m, ch2, xh, S, pos_base, want_x, logit, &lost);
    vkc_device(0);
    if (lost) { memcpy(xh, keep, (size_t)S * D * sizeof(float)); free(keep); g_inkc_ran = 0; return 0; }
    free(keep);
    if (!r2) { inkc_cpu_one(ch2, pos_base, S); return 2; }   /* declined: the CPU runs its layers this step */
    g_inkc_ran += ch2->nl;
    return r2;
}

static void inkc_report(Model *m) {
    for (int d = 0; d < 2; d++) {
        InkChain *ch = inkc_of(m, d);
        if (!ch || !ch->ok || !ch->forwards) continue;
        int was = vkc_device(d);
        VkcStats st; vkc_stats(&st);
        fprintf(stderr, "[VK] %s chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                        "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
                inkc_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms,
                st.dev_bytes / 1048576.0);
        vkc_kv_report(&ch->ks);
        vkc_prof_print();
        vkc_device(was);
    }
}

/* COLI_VK_CHAIN, decided after the tier (ink_vk_tier_start): CUDA and Metal keep their
 * priority; the chain's pipelines, and its teardown registered after the tier's so
 * that it runs first at exit. */
static void inkc_start(Model *m) {
    (void)m;
    if (!g_vk_ready) return;
    const char *e = getenv("COLI_VK_CHAIN");
#ifdef COLI_CUDA
    if (g_cuda) { if (e && *e && *e != '0') fprintf(stderr, "[VK] inkling: dense chain off, the CUDA backend is on\n"); return; }
#endif
#ifdef COLI_METAL
    if (g_metal) { if (e && *e && *e != '0') fprintf(stderr, "[VK] inkling: dense chain off, the Metal expert path is on\n"); return; }
#endif
    (void)e;
    g_vk_chain = coli_vk_chain_decide("inkling", vkt_ready(), INKLING_CHAIN_IGPU);
    if (g_inkc_fit.L && !g_inkc_fit.n) g_vk_chain = 0;   /* the fit left no layer to the device (its line said so) */
    if (g_vk_chain && !vkc_init()) g_vk_chain = 0;
    if (g_vk_chain || g_inkc_fit2_on) atexit(vkc_shutdown_all);
}
