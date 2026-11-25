/* inkling2_chain.h -- Inkling-2's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included by inkling.c in a COLI_VULKAN build after inkling_chain.h (Inkling's chain), whose
 * entry points hand an Inkling-2 model here; COLI_VK_CHAIN decides, as for Inkling.
 *
 * Why: the per-matrix path sent every dense matrix to the device a call at a time, 289 calls a
 * decode token, each a submit and a wait (0.19 ms a call on an RTX PRO 4000: 55 ms of a 510 ms
 * step, of which the matrices' bytes are about 25), and ran the attention, the residual mixes
 * and the convolutions on the CPU between them.
 *
 * What runs where, per layer, for a block of rows (decode: one row), in one frame:
 *   device: the MoE output of the layer before joins the stream (the routed sum from the host,
 *     the MLP's short convolution, the residual add); the attention's input mix of the block
 *     states and the stream (chain_ink2 mode 0: one query on a sliding layer, three on a
 *     global one), its norms, q/k/v, the relative states r, the gate and the sinks; the short
 *     convolutions of K and V (static on a sliding layer; on a global one the dynamic taps
 *     projected from the normed input, chain_ink2 mode 2); the L2 clips of q, r and k
 *     (mode 1) and V normalised without a weight; the attention (chain_relattn with its sink
 *     logits), the gate, the step's K/V rows into the device's cache, o_proj, its short
 *     convolution, the block state when a block starts, the residual, the MLP's input mix, its
 *     norm and the router's logits. Then the frame is waited for.
 *   host: the router's pick, the routed experts (the expert tier's device batch and the CPU's
 *     share, moe_ex as on the per-matrix path), the K/V rows copied into the host's cache.
 * Crossing a layer: the MLP's normed rows (D floats a row), the router logits (E a row) and
 * the K/V rows down; the routed sum (D a row) up. The last frame mixes the output (the
 * output query), normalises the last row and runs lm_head; its logits come back and the host
 * softcaps them (ink2_head_post).
 *
 * State and who owns it, as Inkling's chain keeps it: the host's K/V cache stays canonical (every
 * step's rows come back) and the device holds a mirror per layer at the host's layout (a ring of
 * the window's rows on a sliding layer, the context on a global one); the four short-convolution
 * states per layer live on the device while the chain runs and come back at the end of every
 * step. A CPU step (the chain declined) marks the positions the host wrote alone and the states
 * as the host's; the next chain step uploads them. A device lost mid-step: the state is rebuilt
 * on the CPU from the prefix record, as for Inkling, and the CPU runs from there.
 *
 * Inkling-2-Mini's loop: the layers [loop_s, loop_e) run loop_p passes, in the order the CPU's
 * ink2_layers_forward runs them, each pass from the stream that entered the loop and with its
 * own K/V mirror and convolution states (state layer v = ink_sl, the weights of layer i). From
 * the second pass every loop layer takes four messages, the mixes of the previous pass's block
 * states (its first, the loop's input, left out) and its output with the layer's four queries;
 * a low-rank gate (ink2_loop_gate: z = W [unorm(h); unorm(m)], (1 - tanh(F z)) h + tanh(G z) m)
 * mixes each into the attention's inputs (a sliding layer's one mix three ways: its q, k and v
 * then normed apart) and the MLP's.
 *
 * The routed experts in the frame (vk_dmoe.h): a step of up to VKC_DMOE_ROWS rows (decode) routes
 * on the device and computes a layer's experts there when the tier holds all of that layer's on
 * this device: Inkling-2's router (the sigmoids, the bias for the pick only, the K picks' sigmoids
 * divided by their sum times route_scale) and its SiTU-GLU experts, then on to the next layer with
 * no wait. With every layer so, a decode step is one frame. The K/V rows of such layers wait in
 * kvl (a region a state layer) for the frame's end; the books (the history, the hits, the tier's
 * heat) follow from the ids. A GGUF's NVFP4 layers keep each expert's global scale outside its
 * blocks: their entries carry it (VkcDmoe.escale; gate's and up's one number, as the tier
 * requires). TOPP and ROUTE_TRACE take the host's step.
 *
 * The chain declines (the per-matrix path runs) with several conversations at once (KV_SLOTS),
 * on a geometry outside its shaders (head dim above 256, a bias bank wider than 64, more than 9
 * taps or 15 block states), when a matrix is not on the device, and when the device refuses its
 * memory. It needs the dense matrices on the device (the per-matrix path's copies); the router,
 * the sink projection and the dynamic taps' projections go up as f32. */

#include "vk_dmoe.h"   /* the routed experts in the frame */

#define INK2C_HOST 0   /* the host's convolution states are newer than the device's */
#define INK2C_BOTH 1

typedef struct {
    int ok, failed, tried;
    int rows;                                  /* scratch capacity in rows */
    int max_t; float **hostK;                  /* the host cache the mirrors copy */
    int maxb, hmax, kvo_max, qo_max, ro_max, dyn_max;
    VkcBuf *prm;
    size_t *o_n1, *o_n2, *o_n3, *o_post, *o_qc, *o_rc, *o_relp, *o_cw[4], *o_aq, *o_mq, o_outq, o_final;
    ColiVkTensor *t_lm, **t_q, **t_k, **t_v, **t_r, **t_o, **t_ag, **t_sink, **t_router, **t_wp[2];
    VkcBuf **kc, **vc, *rings;                 /* rings: every layer's four convolution states, at o_csd */
    int *cap, *dlo, *dhi;
    size_t vs_off, *o_csd, csd_n;
    int cs_where, host_zero;
    VkcBuf *x, *res, *blk, *hq, *hk, *hv, *nq, *nk, *nv, *q, *kvs, *r, *gt, *ctx, *a, *mo, *dyn, *ts, *lg, *fin;
    VkcBuf *h2d, *lgd, *kvd, *csd, *outd, *xd, *routed;
    int NS;                                    /* state layers (n_state): kc, vc, cap, dlo, dhi, o_csd by them */
    VkcDmoe dm;                                /* the routed experts in the frame (vk_dmoe.h) */
    VkcBuf *kvl; int kvl_rows;                 /* their layers' K/V rows until the frame ends, a region a state layer */
    size_t *o_rb;                              /* the router's bias in prm, by layer */
    float *escale;                             /* [L][E][2] the experts' global scales (a GGUF's NVFP4 layers), NULL none */
    VkcBuf *relpart;                           /* a decode step's split attention's partials (vkc_relattn_part), for max_t */
    unsigned long long dm_layers;              /* layer steps whose experts ran in the frame */
    ColiVkTensor **t_lg;                       /* the loop's gates, [layer][q, k, v, mlp][down, forget, gain] */
    size_t o_lq;                               /* the loop's message queries, [loop layer][q, k, v, mlp][D] */
    VkcBuf *xpre, *xprev, *pblk, *msg[4], *hn2, *lz, *lf, *lgv;
    float *host_routed;
    unsigned long long forwards;
    double host_ms;
} Ink2Chain;

static void ink2c_fatal(const char *what) {
    fprintf(stderr, "[VK] inkling2 chain: %s -- stopping (COLI_VK_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
static int ink2c_maxb(const Cfg *c) {
    int blk = c->res_block > 0 ? c->res_block : 1;
    return (c->n_layers + blk - 1) / blk;
}
static int ink2c_geometry_ok(const Model *m, int say) {
    const Cfg *c = &m->c; int dr = c->d_rel, CK = c->conv_k;
    for (int i = 0; i < c->n_layers; i++) {
        const Layer *l = &m->L[i];
        int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i);
        if (!l->x2 || hd > 256 || hd < 1 || KV < 1 || H % KV || dr < 1 || dr > 64 || L_EXT(c, i) < 1 || CK < 1 || CK > 9 ||
            (c->local[i] && c->window < 1) || !c->sparse[i] || c->n_shared > 0 || ink2c_maxb(c) > 15) {
            if (say)
                fprintf(stderr, "[VK] inkling2 chain: a geometry its shaders do not take (layer %d: head dim %d, %d/%d heads, "
                                "bias bank %d x %d, %d taps, %d block states); per-matrix path\n",
                        i, hd, H, KV, dr, L_EXT(c, i), CK, ink2c_maxb(c));
            return 0;
        }
    }
    return 1;
}

/* The device bytes the chain will take beside the dense matrices (placed already by the
 * per-matrix path): the f32 router, sink and dynamic-tap projections, the parameters, the
 * convolution states, the K/V mirrors at the served context (ink_ctx_max) and the scratch of a
 * prompt chunk of 256 rows. The tier's budget leaves them (ink_vk_dense_bytes). */
static size_t ink2c_plan_bytes(const Model *m) {
    const Cfg *c = &m->c;
    size_t D = (size_t)c->hidden, E = (size_t)c->n_experts, CK = (size_t)c->conv_k, b = 0, row = 0;
    /* Mirrors grow lazily with m->max_t; reserving the full 1M+ served context upfront starves
     * the Vulkan expert tier on 32 GiB cards. Reserve an initial working horizon and let
     * ink2c_mirror claim more via vkt_give_back. */
    int t = ink_ctx_max() + 8;
    if (t > 16392) t = 16392;
    for (int v = c->n_layers; v < c->n_state; v++) {                   /* the loop's later passes' state */
        size_t kvo = (size_t)L_KV(c, v) * L_HD(c, v);
        b += (2 * kvo + 2 * D) * (CK - 1) * 4 * 2 + 2 * kvo * (size_t)kv_ring_rows(c, v, t) * 4;
    }
    if (c->loop_p > 1) {                                               /* the gates and the queries */
        size_t nl = (size_t)(c->loop_e - c->loop_s), r = (size_t)c->loop_r;
        b += nl * 4 * (r * 2 * D + 2 * D * r) * 4 + nl * 4 * D * 4;
    }
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
    if (c->loop_p > 1) row += ((size_t)(7 + ink2c_maxb(c)) * D + (size_t)c->loop_r) * 4;   /* a row's loop scratch */
    {   /* the split attention's partials at the planned context (ink2c_mirror) */
        size_t hm = 0, hdm = 0;
        for (int i = 0; i < c->n_layers; i++) { if ((size_t)L_HEADS(c, i) > hm) hm = L_HEADS(c, i); if ((size_t)L_HD(c, i) > hdm) hdm = L_HD(c, i); }
        b += (size_t)VKC_DMOE_ROWS * hm * (((size_t)t + 127) / 128) * (hdm + 2) * 4;
    }
    return b + row * 256 + (size_t)c->unpad_vocab * 4 + ((size_t)ink2c_maxb(c) * D * 4) * 256;
}

static Ink2Chain *ink2c_of(Model *m) { return (Ink2Chain *)m->ink2chain; }
static Ink2Chain *ink2c_new(Model *m) {
    Ink2Chain *ch = ink2c_of(m);
    if (ch) return ch;
    if (!(ch = calloc(1, sizeof *ch))) return NULL;
    m->ink2chain = ch;
    int L = m->c.n_layers, NS = m->c.n_state;
    ch->NS = NS;
#define INK2C_ARR(f, T) (ch->f = calloc((size_t)L, sizeof(T)))
#define INK2C_ARS(f, T) (ch->f = calloc((size_t)NS, sizeof(T)))
    if (!INK2C_ARR(o_n1, size_t) || !INK2C_ARR(o_n2, size_t) || !INK2C_ARR(o_n3, size_t) || !INK2C_ARR(o_post, size_t) ||
        !INK2C_ARR(o_qc, size_t) || !INK2C_ARR(o_rc, size_t) || !INK2C_ARR(o_relp, size_t) || !INK2C_ARR(o_cw[0], size_t) ||
        !INK2C_ARR(o_cw[1], size_t) || !INK2C_ARR(o_cw[2], size_t) || !INK2C_ARR(o_cw[3], size_t) || !INK2C_ARR(o_aq, size_t) ||
        !INK2C_ARR(o_mq, size_t) || !INK2C_ARR(o_rb, size_t) || !INK2C_ARR(t_q, void *) || !INK2C_ARR(t_k, void *) || !INK2C_ARR(t_v, void *) ||
        !INK2C_ARR(t_r, void *) || !INK2C_ARR(t_o, void *) || !INK2C_ARR(t_ag, void *) || !INK2C_ARR(t_sink, void *) ||
        !INK2C_ARR(t_router, void *) || !INK2C_ARR(t_wp[0], void *) || !INK2C_ARR(t_wp[1], void *) ||
        !INK2C_ARS(kc, void *) || !INK2C_ARS(vc, void *) || !INK2C_ARS(cap, int) || !INK2C_ARS(dlo, int) ||
        !INK2C_ARS(dhi, int)) return NULL;
#undef INK2C_ARR
#undef INK2C_ARS
    if (!(ch->o_csd = calloc((size_t)4 * NS, sizeof(size_t))) || !(ch->t_lg = calloc((size_t)L * 12, sizeof(void *)))) return NULL;
    return ch;
}

/* The chain's own device memory freed (the per-matrix path's copies of the dense matrices stay). */
static void ink2c_free_own(Model *m, Ink2Chain *ch) {
    if (vkc_ready()) vkc_finish();
    for (int i = 0; i < m->c.n_layers; i++) {
        ColiVkTensor **own[] = {&ch->t_sink[i], &ch->t_router[i], &ch->t_wp[0][i], &ch->t_wp[1][i]};
        for (size_t k = 0; k < sizeof own / sizeof own[0]; k++) { if (*own[k]) coli_vk_tensor_free(*own[k]); *own[k] = NULL; }
        ch->t_q[i] = ch->t_k[i] = ch->t_v[i] = ch->t_r[i] = ch->t_o[i] = ch->t_ag[i] = NULL;   /* the per-matrix path's */
        for (int k = 0; k < 12; k++) { ColiVkTensor **t = &ch->t_lg[(size_t)i * 12 + k]; if (*t) coli_vk_tensor_free(*t); *t = NULL; }
    }
    for (int v = 0; v < ch->NS; v++) { vkc_free(ch->kc[v]); vkc_free(ch->vc[v]); ch->kc[v] = ch->vc[v] = NULL; }
    ch->t_lm = NULL;
    vkc_dmoe_free(&ch->dm);
    vkc_relattn_part(NULL); vkc_free(ch->relpart); ch->relpart = NULL;
    free(ch->escale); ch->escale = NULL;
    vkc_free(ch->kvl); ch->kvl = NULL; ch->kvl_rows = 0;
    vkc_free(ch->rings); ch->rings = NULL;
    vkc_free(ch->prm); ch->prm = NULL;
    ch->max_t = 0; ch->hostK = NULL;
}

/* The model's parameters on the device, its tensors resolved; NULL = the chain cannot run. */
static Ink2Chain *ink2c_setup(Model *m) {
    Ink2Chain *ch = ink2c_of(m);
    if (ch && ch->tried) return ch->ok ? ch : NULL;
    if (!ch && !(ch = ink2c_new(m))) return NULL;
    ch->tried = 1;
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, CK = c->conv_k, dr = c->d_rel, E = c->n_experts;
    if (!ink2c_geometry_ok(m, 1)) return NULL;
    if (!vkc_sconv_ready() || !vkc_relattn_ready() || !vkc_ink2_ready()) {
        fprintf(stderr, "[VK] inkling2 chain: chain_sconv.spv, chain_relattn.spv or chain_ink2.spv is missing; per-matrix path\n");
        return NULL;
    }
    const char *what = NULL;
    for (int i = 0; i < L && !what; i++) {
        Layer *l = &m->L[i]; Ink2Layer *x2 = l->x2;
        int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd;
        if (!(ch->t_q[i] = inkc_tensor(0, l->q, D, H * hd)) || !(ch->t_k[i] = inkc_tensor(0, l->k, D, kvo)) ||
            !(ch->t_v[i] = inkc_tensor(0, l->v, D, kvo)) || !(ch->t_r[i] = inkc_tensor(0, l->r, D, H * dr)) ||
            !(ch->t_o[i] = inkc_tensor(0, l->o, H * hd, D)) || !(ch->t_ag[i] = inkc_tensor(0, l->ag, D, H * hd)))
            what = "an attention matrix";
        else if (!coli_vk_tensor_ensure_f32(&ch->t_sink[i], x2->sink, D, H, 0)) what = "a sink projection";
        else if (!coli_vk_tensor_ensure_f32(&ch->t_router[i], l->router, D, E, 0)) what = "the router";
        else if (!c->local[i] && (!coli_vk_tensor_ensure_f32(&ch->t_wp[0][i], x2->k_wp, D, KV * CK, 0) ||
                                  !coli_vk_tensor_ensure_f32(&ch->t_wp[1][i], x2->v_wp, D, KV * CK, 0)))
            what = "a dynamic convolution's projection";
        else if (c->loop_p > 1 && i >= c->loop_s && i < c->loop_e)
            for (int g = 0; g < 4 && !what; g++) {
                ColiVkTensor **t = &ch->t_lg[((size_t)i * 4 + g) * 3];
                if (!coli_vk_tensor_ensure_f32(&t[0], x2->lg_down[g], 2 * D, c->loop_r, 0) ||
                    !coli_vk_tensor_ensure_f32(&t[1], x2->lg_fgt[g], c->loop_r, D, 0) ||
                    !coli_vk_tensor_ensure_f32(&t[2], x2->lg_gain[g], c->loop_r, D, 0))
                    what = "a loop gate";
            }
    }
    if (!what && !(ch->t_lm = inkc_tensor(0, m->lm_head, D, c->unpad_vocab))) what = "lm_head";
    if (what) {
        fprintf(stderr, "[VK] inkling2 chain: %s did not reach the device; per-matrix path\n", what);
        ink2c_free_own(m, ch);
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
        ch->o_rb[i] = n; n += (size_t)c->n_experts;
    }
    ch->o_outq = n; n += D;
    ch->o_final = n; n += D;
    ch->o_lq = n; if (c->loop_p > 1) n += (size_t)(c->loop_e - c->loop_s) * 4 * D;
    float *arena = calloc(n, sizeof(float));
    if (!arena) { ink2c_free_own(m, ch); return NULL; }
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
        memcpy(arena + ch->o_rb[i], l->rbias, (size_t)c->n_experts * sizeof(float));
    }
    memcpy(arena + ch->o_outq, m->x2_out_query, D * sizeof(float));
    memcpy(arena + ch->o_final, m->final_norm, D * sizeof(float));
    if (c->loop_p > 1) memcpy(arena + ch->o_lq, m->x2_loopq, (size_t)(c->loop_e - c->loop_s) * 4 * D * sizeof(float));
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, n * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) { ink2c_free_own(m, ch); return NULL; }
    ch->maxb = ink2c_maxb(c);
    ch->csd_n = 0;   /* the states one after another, the same layout on the device and in the read-back */
    for (int v = 0; v < ch->NS; v++)
        for (int b = 0; b < 4; b++) { ch->o_csd[(size_t)b * ch->NS + v] = ch->csd_n; ch->csd_n += (size_t)ink_cs_cells(c, b, v); }
    if (!(ch->rings = vkc_buf((ch->csd_n ? ch->csd_n : 1) * sizeof(float), VKC_DEV))) { ink2c_free_own(m, ch); return NULL; }
    ch->cs_where = INK2C_HOST; ch->host_zero = 0;
    ch->ok = 1;
    if (c->loop_p > 1)
        fprintf(stderr, "[VK] inkling2 chain: %d layers on the device, layers %d-%d %d passes, %.1f MiB of parameters\n",
                L, c->loop_s, c->loop_e - 1, c->loop_p, n * 4 / 1048576.0);
    else fprintf(stderr, "[VK] inkling2 chain: %d layers on the device, %.1f MiB of parameters\n", L, n * 4 / 1048576.0);
    return ch;
}

/* the scratch buffers for `rows` rows */
static int ink2c_bufs(Ink2Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; size_t D = (size_t)c->hidden, E = (size_t)c->n_experts, r = (size_t)rows;
    ch->vs_off = (r * ch->kvo_max + 63) & ~(size_t)63;
    size_t dyn = ch->dyn_max > 0 ? (size_t)ch->dyn_max : 1;
    return vkc_reserve(&ch->x, r * D * 4, VKC_DEV) && vkc_reserve(&ch->res, r * D * 4, VKC_DEV) &&
           vkc_reserve(&ch->blk, r * ch->maxb * D * 4, VKC_DEV) && vkc_reserve(&ch->hq, r * D * 4, VKC_DEV) &&
           vkc_reserve(&ch->hk, r * D * 4, VKC_DEV) && vkc_reserve(&ch->hv, r * D * 4, VKC_DEV) &&
           vkc_reserve(&ch->nq, r * D * 4, VKC_DEV) && vkc_reserve(&ch->nk, r * D * 4, VKC_DEV) &&
           vkc_reserve(&ch->nv, r * D * 4, VKC_DEV) && vkc_reserve(&ch->q, r * ch->qo_max * 4, VKC_DEV) &&
           vkc_reserve(&ch->kvs, (ch->vs_off + r * ch->kvo_max) * 4, VKC_DEV) && vkc_reserve(&ch->r, r * ch->ro_max * 4, VKC_DEV) &&
           vkc_reserve(&ch->gt, r * ch->qo_max * 4, VKC_DEV) && vkc_reserve(&ch->ctx, r * ch->qo_max * 4, VKC_DEV) &&
           vkc_reserve(&ch->a, r * D * 4, VKC_DEV) && vkc_reserve(&ch->mo, r * D * 4, VKC_DEV) &&
           vkc_reserve(&ch->dyn, r * dyn * 4, VKC_DEV) && vkc_reserve(&ch->ts, (2 * r + r * ch->hmax) * 4, VKC_DEV) &&
           vkc_reserve(&ch->lg, r * E * 4, VKC_DEV) && vkc_reserve(&ch->fin, D * 4, VKC_DEV) &&
           vkc_reserve(&ch->h2d, r * D * 4, VKC_DOWN) && vkc_reserve(&ch->lgd, r * E * 4, VKC_DOWN) &&
           vkc_reserve(&ch->kvd, 2 * r * ch->kvo_max * 4, VKC_DOWN) && vkc_reserve(&ch->csd, (ch->csd_n ? ch->csd_n : 1) * 4, VKC_DOWN) &&
           vkc_reserve(&ch->outd, (size_t)c->unpad_vocab * 4, VKC_DOWN) && vkc_reserve(&ch->routed, r * D * 4, VKC_UP) &&
           (c->loop_p <= 1 ||
            (vkc_reserve(&ch->xpre, r * D * 4, VKC_DEV) && vkc_reserve(&ch->xprev, r * D * 4, VKC_DEV) &&
             vkc_reserve(&ch->pblk, r * ch->maxb * D * 4, VKC_DEV) && vkc_reserve(&ch->msg[0], r * D * 4, VKC_DEV) &&
             vkc_reserve(&ch->msg[1], r * D * 4, VKC_DEV) && vkc_reserve(&ch->msg[2], r * D * 4, VKC_DEV) &&
             vkc_reserve(&ch->msg[3], r * D * 4, VKC_DEV) && vkc_reserve(&ch->hn2, r * 2 * D * 4, VKC_DEV) &&
             vkc_reserve(&ch->lz, r * (size_t)c->loop_r * 4, VKC_DEV) && vkc_reserve(&ch->lf, r * D * 4, VKC_DEV) &&
             vkc_reserve(&ch->lgv, r * D * 4, VKC_DEV)));
}
/* Prompt rows per chunk (vkc_chunk_rows): the scratch a row, and the routed experts' outputs. */
static int ink2c_chunk_rows(Ink2Chain *ch, Model *m) {
    const Cfg *c = &m->c; size_t D = (size_t)c->hidden;
    size_t row = (14 * D + (size_t)ch->maxb * D + 4 * (size_t)ch->qo_max + 4 * (size_t)ch->kvo_max + ch->ro_max +
                  2 * (size_t)c->n_experts + (size_t)ch->dyn_max + 2 + ch->hmax) * 4 + (size_t)(2 * c->topk + 2) * D * 4 +
                 (c->loop_p > 1 ? ((size_t)(9 + ch->maxb) * D + (size_t)c->loop_r) * 4 : 0);
    return vkc_chunk_rows("inkling2", row);
}
static VkcBuf *ink2c_dev_buf(size_t bytes) {
    VkcBuf *b = vkc_buf(bytes, VKC_DEV);
    if (!b) { vkt_give_back(bytes + (64u << 20)); b = vkc_buf(bytes, VKC_DEV); }
    return b;
}
/* the K/V mirrors at the host's layout, made again when the host's cache was */
static int ink2c_mirror(Ink2Chain *ch, Model *m) {
    Cfg *c = &m->c;
    if (ch->max_t == m->max_t && ch->hostK == m->K) return 1;
    for (int i = 0; i < ch->NS; i++) {
        vkc_free(ch->kc[i]); vkc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
        int cap = kv_ring_rows(c, i, m->max_t), hd = L_HD(c, i), KV = L_KV(c, i);
        ch->cap[i] = cap;
        ch->dlo[i] = 0; ch->dhi[i] = cap < m->max_t ? m->max_t : m->kv_len;   /* a ring: every row; else the positions held */
        size_t nb = (size_t)KV * cap * hd * sizeof(float);
        ch->kc[i] = ink2c_dev_buf(nb);
        ch->vc[i] = ink2c_dev_buf(nb);
        if (!ch->kc[i] || !ch->vc[i]) { ch->max_t = 0; ch->hostK = NULL; return 0; }
    }
    /* the split attention's partials: a decode step's rows (VKC_DMOE_ROWS) at the whole context;
     * without them every step's attention runs whole, as before */
    vkc_relattn_part(NULL); vkc_free(ch->relpart);
    int hdmax = 0;
    for (int i = 0; i < c->n_layers; i++) if (L_HD(c, i) > hdmax) hdmax = L_HD(c, i);
    ch->relpart = vkc_buf((size_t)VKC_DMOE_ROWS * ch->hmax * ((m->max_t + 8 + 127) / 128) * (hdmax + 2) * sizeof(float), VKC_DEV);
    ch->max_t = m->max_t; ch->hostK = m->K;
    return 1;
}
static int ink2c_scratch(Ink2Chain *ch, Model *m, int rows) {
    if (rows < ch->rows) rows = ch->rows;   /* the layout of the widest step so far: ch->rows is its stride */
    if (!ink2c_bufs(ch, m, rows)) return 0;
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, (size_t)rows * m->c.hidden * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    return 1;
}

/* ---- state between the host and the device (inkling_chain.h's, for Inkling-2) ---- */
static void ink2c_host_wrote(Model *m, int zero) {
    Ink2Chain *ch = ink2c_of(m);
    if (!ch || !ch->ok) return;
    ch->cs_where = INK2C_HOST; ch->host_zero = zero;
}
static void ink2c_cpu_step(Model *m, int pos_base, int S) {
    Ink2Chain *ch = ink2c_of(m);
    if (!ch || !ch->ok) return;
    ch->cs_where = INK2C_HOST; ch->host_zero = 0;
    for (int i = 0; i < ch->NS; i++) {
        if (ch->dlo[i] >= ch->dhi[i]) { ch->dlo[i] = pos_base; ch->dhi[i] = pos_base + S; continue; }
        if (pos_base < ch->dlo[i]) ch->dlo[i] = pos_base;
        if (pos_base + S > ch->dhi[i]) ch->dhi[i] = pos_base + S;
    }
}
static int ink2c_push_state(Ink2Chain *ch, Model *m) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->cs_where == INK2C_HOST) {
        if (ch->host_zero) ok = !ch->csd_n || vkc_zero(ch->rings, 0, ch->csd_n);
        else for (int i = 0; i < ch->NS && ok; i++) for (int b = 0; b < 4 && ok; b++) {
            size_t nc = (size_t)ink_cs_cells(c, b, i);
            if (nc) ok = vkc_write(ch->rings, ch->o_csd[(size_t)b * ch->NS + i], m->cs[b][i], nc * sizeof(float));
        }
        ch->cs_where = INK2C_BOTH;
    }
    for (int i = 0; i < ch->NS && ok; i++) {
        if (ch->dlo[i] >= ch->dhi[i]) continue;
        int cap = ch->cap[i], hd = L_HD(c, i), KV = L_KV(c, i);
        int t0 = ch->dhi[i] - cap > ch->dlo[i] ? ch->dhi[i] - cap : ch->dlo[i];
        for (int t = t0; t < ch->dhi[i] && ok; ) {        /* a run of rows up to the ring's end */
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
/* The device was lost with the newest state on it: rebuild the K/V cache and the convolution
 * states on the CPU from the `upto` positions the prefix record names, and leave the chain off. */
static void ink2c_recover(Model *m, int upto) {
    Ink2Chain *ch = ink2c_of(m);
    Cfg *c = &m->c; int D = c->hidden;
    g_vk_chain = 0;
    if (ch) ch->failed = 1;
    for (int i = 0; i < c->n_state; i++)
        for (int b = 0; b < 4; b++) memset(m->cs[b][i], 0, (size_t)ink_cs_cells(c, b, i) * sizeof(float));
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        ink2c_fatal("the device was lost with a state its token ids do not describe (audio or an image)");
    fprintf(stderr, "[VK] inkling2 chain: the device was lost; rebuilding the state of %d positions on the CPU, "
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
static int ink2c_norm(VkcBuf *x, VkcBuf *w, size_t wo, VkcBuf *y, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, 0, D, D, 0, D, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
/* a static short convolution through bank `bank` and its scale: (conv + input) / sqrt(K + 1);
 * the taps of weight layer i, the state of state layer v */
static int ink2c_sconv(Ink2Chain *ch, VkcBuf *x, size_t xo, int n, int C, int bank, int i, int v, const Cfg *c) {
    VkcSconv p = {0, n, C, c->conv_k, (int)xo, C, (int)ch->o_cw[bank][i], (int)ch->o_csd[(size_t)bank * ch->NS + v], 0, 1.f};
    VkcSconv dv = {2, 0, 0, 0, (int)xo, 0, 0, 0, n * C, (float)sqrt((double)(c->conv_k + 1))};
    return vkc_sconv(x, ch->prm, ch->rings, &p) && vkc_sconv(x, NULL, NULL, &dv);
}
/* a global layer's dynamic convolution of K (bank 0, from nk) or V (bank 1, from nv) in kvs */
static int ink2c_dconv(Ink2Chain *ch, Model *m, int i, int v, int bank, VkcBuf *xin, int n) {
    Cfg *c = &m->c; int KV = L_KV(c, i), hd = L_HD(c, i), K = c->conv_k;
    VkcInk2Dconv p = {n, KV * hd, K, hd, bank ? (int)ch->vs_off : 0, KV * hd, 0, KV * K, (int)ch->o_cw[bank][i],
                      (int)ch->o_csd[(size_t)bank * ch->NS + v], (float)sqrt((double)(2 * K + 1))};
    return vkc_matmul(ch->t_wp[bank][i], xin, 0, ch->dyn, 0, n) && vkc_ink2_dconv(ch->kvs, ch->dyn, ch->prm, ch->rings, &p);
}
/* the attention from nq (and nk, nv on a global layer, or with a loop pass's messages: sep): o_proj's
 * rows into a, the step's K/V rows into state layer v's cache on the device and into kvd */
static int ink2c_attention(Ink2Chain *ch, Model *m, int i, int v, int sep, int n, int pb, VkcBuf *kvdst, size_t ko, size_t vstride) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, dr = c->d_rel;
    int cap = ch->cap[v], local = c->local[i], R = ch->rows;
    size_t vs = ch->vs_off;
    VkcBuf *xk = sep ? ch->nk : ch->nq, *xv = sep ? ch->nv : ch->nq;
    int ok = vkc_matmul(ch->t_q[i], ch->nq, 0, ch->q, 0, n) && vkc_matmul(ch->t_k[i], xk, 0, ch->kvs, 0, n) &&
             vkc_matmul(ch->t_v[i], xv, 0, ch->kvs, vs, n) && vkc_matmul(ch->t_r[i], ch->nq, 0, ch->r, 0, n) &&
             vkc_matmul(ch->t_ag[i], ch->nq, 0, ch->gt, 0, n) && vkc_matmul(ch->t_sink[i], ch->nq, 0, ch->ts, (size_t)2 * R, n);
    if (local) ok = ok && ink2c_sconv(ch, ch->kvs, 0, n, kvo, 0, i, v, c) && ink2c_sconv(ch, ch->kvs, vs, n, kvo, 1, i, v, c);
    else ok = ok && ink2c_dconv(ch, m, i, v, 0, ch->nk, n) && ink2c_dconv(ch, m, i, v, 1, ch->nv, n);
    VkcInk2Clip qc = {n * H, hd, H, 0, H * hd, 1, (int)ch->o_qc[i], c->qk_clip};
    VkcInk2Clip rc = {n * H, dr, H, 0, H * dr, 1, (int)ch->o_rc[i], c->rel_clip};
    VkcInk2Clip kc = {n * KV, hd, KV, 0, kvo, 0, 0, c->qk_clip};
    VkcNorm vn = {n * KV, hd, KV, (int)vs, kvo, hd, (int)vs, kvo, hd, 0, 0, 2, c->eps, 1.f};
    ok = ok && vkc_ink2_clip(ch->q, ch->prm, &qc) && vkc_ink2_clip(ch->r, ch->prm, &rc) &&
         vkc_ink2_clip(ch->kvs, NULL, &kc) && vkc_norm(ch->kvs, ch->prm, ch->kvs, &vn);
    VkcRelAttn a = {n, H, KV, hd, pb, cap, local ? c->window : 0, L_EXT(c, i), dr,
                    0, H * hd, 0, H * hd, 0, 0, 0, (int)vs, kvo, 0, H * dr, (int)ch->o_relp[i], local ? R : 0,
                    (float)(1.0 / sqrt((double)hd)), 1, 2 * R};
    VkcEw gate = {VKC_EW_SIGMUL, n * H * hd, H * hd, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    ok = ok && vkc_relattn(ch->q, ch->kc[v], ch->vc[v], ch->ctx, ch->kvs, ch->r, ch->prm, ch->ts, &a) &&
         vkc_ew(ch->ctx, ch->ctx, ch->gt, NULL, NULL, &gate);
    if (!ok) return 0;
    /* the step's rows into the cache after the attention read them: the last `cap` of them */
    int s0 = n > cap ? n - cap : 0, nr = (n - s0) * KV;
    VkcRegion *rk = malloc(sizeof *rk * (size_t)nr * 2), *rv = rk ? rk + nr : NULL;
    if (!rk) return 0;
    for (int s = s0; s < n; s++) for (int h = 0; h < KV; h++) {
        size_t dst = ((size_t)h * cap + (size_t)(pb + s) % cap) * hd, src = (size_t)s * kvo + (size_t)h * hd;
        rk[(s - s0) * KV + h] = (VkcRegion){dst, src, (size_t)hd};
        rv[(s - s0) * KV + h] = (VkcRegion){dst, vs + src, (size_t)hd};
    }
    ok = vkc_copy_regions(ch->kc[v], ch->kvs, rk, nr) && vkc_copy_regions(ch->vc[v], ch->kvs, rv, nr) &&
         vkc_copy(kvdst, ko, ch->kvs, 0, (size_t)n * kvo) && vkc_copy(kvdst, ko + vstride, ch->kvs, vs, (size_t)n * kvo);
    free(rk);
    return ok && vkc_matmul(ch->t_o[i], ch->ctx, 0, ch->a, 0, n);
}
/* state layer v's new K/V rows (kv: K, then V vstride floats on) into the host's cache, as
 * ink2_attention writes them */
static void ink2c_kv_down(Ink2Chain *ch, Model *m, int v, int n, int pb, const float *kv, size_t vstride) {
    Cfg *c = &m->c;
    int KV = L_KV(c, v), hd = L_HD(c, v), kvo = KV * hd, cap = ch->cap[v], s0 = n > cap ? n - cap : 0;
    const float *vv = kv + vstride;
    for (int s = s0; s < n; s++) for (int h = 0; h < KV; h++) {
        size_t dst = ((size_t)h * cap + (size_t)(pb + s) % cap) * hd, src = (size_t)s * kvo + (size_t)h * hd;
        memcpy(m->K[v] + dst, kv + src, hd * sizeof(float));
        memcpy(m->V[v] + dst, vv + src, hd * sizeof(float));
    }
}
/* the MoE output of layer i, state layer v (the host's routed sum) joining the stream: x = res + sconv(mo) */
static int ink2c_join(Ink2Chain *ch, Model *m, int i, int v, int n, VkcBuf *routed) {
    Cfg *c = &m->c; int D = c->hidden;
    VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_copy(ch->mo, 0, routed, 0, (size_t)n * D) && ink2c_sconv(ch, ch->mo, 0, n, D, 3, i, v, c) &&
           vkc_ew(ch->x, ch->res, ch->mo, NULL, NULL, &add);
}
/* the residual mix of the nb block states (of blk: ch->blk, or a loop pass's pblk) and `part` into y
 * with query q (at qo in prm) */
static int ink2c_mix_of(Ink2Chain *ch, VkcBuf *blk, VkcBuf *part, int nb, size_t qo, VkcBuf *y, int n, int D) {
    VkcInk2Mix p = {n, D, nb, 0, D, 0, ch->maxb * D, (int)qo, 0, D, 1e-6f, (float)(1.0 / sqrt((double)D))};
    return vkc_ink2_mix(part, blk, ch->prm, y, &p);
}
static int ink2c_mix(Ink2Chain *ch, VkcBuf *part, int nb, size_t qo, VkcBuf *y, int n, int D) {
    return ink2c_mix_of(ch, ch->blk, part, nb, qo, y, n, D);
}
/* a loop pass's gate g (0 q, 1 k, 2 v, 3 mlp) of layer i: y = (1 - tanh(F z)) h + tanh(G z) msg,
 * z = W [unorm(h); unorm(msg)] (ink2_loop_gate); y may be h */
static int ink2c_loop_gate(Ink2Chain *ch, Model *m, int i, int g, VkcBuf *h, VkcBuf *y, int n) {
    Cfg *c = &m->c; int D = c->hidden;
    ColiVkTensor **t = &ch->t_lg[((size_t)i * 4 + g) * 3];
    VkcNorm nh = {n, D, 1, 0, D, D, 0, 2 * D, D, 0, 0, VKC_NORM_NOW, c->eps, 1.f};
    VkcNorm nm = {n, D, 1, 0, D, D, D, 2 * D, D, 0, 0, VKC_NORM_NOW, c->eps, 1.f};
    VkcEw mix = {VKC_EW_LOOPGATE, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_norm(h, ch->prm, ch->hn2, &nh) && vkc_norm(ch->msg[g], ch->prm, ch->hn2, &nm) &&
           vkc_matmul(t[0], ch->hn2, 0, ch->lz, 0, n) && vkc_matmul(t[1], ch->lz, 0, ch->lf, 0, n) &&
           vkc_matmul(t[2], ch->lz, 0, ch->lgv, 0, n) && vkc_ew(y, ch->lf, ch->lgv, h, ch->msg[g], &mix);
}

/* ---- the routed experts in the frame (vk_dmoe.h) ---- */
/* the step's table and buffers, 1 when a layer of it may run its experts here: not with TOPP (its
 * kept ranks are the host's), ROUTE_TRACE, or shared experts (Inkling's first, not Inkling-2) */
static int ink2c_dmoe(Ink2Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; const int E = c->n_experts;
    if (!ch->escale) {   /* the scales an expert outside its blocks, once */
        int any = 0;
        for (int i = 0; i < c->n_layers; i++) any |= m->L[i].xgsc != NULL;
        if (any && (ch->escale = calloc((size_t)c->n_layers * E * 2, sizeof(float))))
            for (int i = 0; i < c->n_layers; i++)
                for (int e = 0; m->L[i].xgsc && e < E; e++) {
                    ch->escale[((size_t)i * E + e) * 2] = m->L[i].xgsc[e];
                    ch->escale[((size_t)i * E + e) * 2 + 1] = m->L[i].xdsc[e];
                }
        if (any && !ch->escale) return 0;
        ch->dm.escale = ch->escale;
    }
    if ((g_topp > 0.f && g_topp < 1.f) || rt_fp || c->n_shared > 0 ||
        !vkc_dmoe_prepare(&ch->dm, 0, c->n_layers, c->n_experts, c->topk, c->hidden, c->moe_inter, rows)) return 0;
    if (!ch->kvl && !(ch->kvl = vkc_buf((size_t)ch->NS * 2 * VKC_DMOE_ROWS * ch->kvo_max * sizeof(float), VKC_DOWN))) return 0;
    ch->kvl_rows = VKC_DMOE_ROWS;
    return 1;
}
static int ink2c_dmoe_on(Ink2Chain *ch, Model *m, int i) {
    /* a GGUF's NVFP4 layer: gate's and up's scale one number an expert (the tier's start checks) */
    return vkc_dmoe_on(&ch->dm, i) && (!m->L[i].xgsc || ch->escale);
}
/* after the frame: a step whose experts ran there (layer i, state layer v), its K/V rows into the
 * host's cache and its books, as moe_ex and ink_vk_moe keep them (a loop's later passes read the
 * layer's ids as the frame left them: the last pass's) */
static void ink2c_dmoe_done(Ink2Chain *ch, Model *m, int i, int v, int n, int pb) {
    const int K = m->c.topk;
    const size_t reg = (size_t)2 * ch->kvl_rows * ch->kvo_max;
    ink2c_kv_down(ch, m, v, n, pb, (const float *)vkc_ptr(ch->kvl) + (size_t)v * reg, reg / 2);
    const int *ids = vkc_dmoe_ids(&ch->dm, i);
    m->ereq += (uint64_t)n * K; m->euse += (uint64_t)n * K;
    for (int j = 0; j < n * K; j++) {
        if (ids[j] < 0) continue;
        if (m->eusage && m->eusage[i]) m->eusage[i][ids[j]]++;
        ehit_mark(m, i, ids[j]);
    }
    vkt_dev_count(i, n, K, ids);
    ch->dm_layers++;
}

/* Every layer for S rows from host rows xh, the last row's logits (softcapped) into `logit`;
 * xh gets the rows the final norm reads (the output mix applied) when want_x. 0 = not taken:
 * xh is as it was and the caller runs the step on the CPU (after ink2c_recover when the device
 * was lost). 1 = taken. */
static int ink2c_forward(Model *m, float *xh, int S, int pos_base, int want_x, float *logit) {
    if (!g_vk_chain) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && S <= 2) return 0;   /* prompts only: decode on the CPU */
    if (m->mux_rows) return 0;
    Ink2Chain *ch = ink2c_setup(m);
    if (!ch || ch->failed) return 0;
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, E = c->n_experts, blkn = c->res_block > 0 ? c->res_block : 1;
    int mirror_ok = ink2c_mirror(ch, m);
    int CH = mirror_ok ? ink2c_chunk_rows(ch, m) : 1, rows = S < CH ? S : CH;
    if (!mirror_ok || pos_base + S > m->max_t || !ink2c_scratch(ch, m, rows) ||
        (want_x && !vkc_reserve(&ch->xd, (size_t)rows * D * sizeof(float), VKC_DOWN))) {
        fprintf(stderr, "[VK] inkling2 chain: device memory for %d rows refused; per-matrix path\n", rows);
        ink2c_free_own(m, ch);
        ch->failed = 1;
        return 0;
    }
    float *xfin = want_x ? malloc((size_t)S * D * sizeof(float)) : NULL;
    if (want_x && !xfin) return 0;
    const int dmoe = ink2c_dmoe(ch, m, rows);
    vkc_relattn_part(ch->relpart);   /* a decode step's attention split over its positions */
    int *dq = dmoe ? malloc((size_t)2 * c->n_state * sizeof(int)) : NULL, ndq = 0;   /* this frame's in-frame steps (i, v) */
    if (dmoe && !dq) { free(xfin); return 0; }
    vkc_gemm_rows(-1);
    for (int c0 = 0; c0 < S; c0 += rows) {
        if (coli_prompt_break()) { free(xfin); free(dq); return 0; }
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0, R = ch->rows;
        if (!vkc_begin() || !vkc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !ink2c_push_state(ch, m)) goto lost;
        if (dmoe && !vkc_dmoe_upload(&ch->dm)) goto lost;   /* the tier's residents changed: the table again */
        ndq = 0;
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
            int wok = vkc_write(ch->ts, 0, tp, (size_t)n * sizeof(float)) && vkc_write(ch->ts, (size_t)R, tp + n, (size_t)n * sizeof(float));
            if (tp != tau) free(tp);
            if (!wok) goto lost;
        }
        int ok = 1, pending = 0, B = 0, pi = 0, pv = 0, preB = 0, nprev = 0, prev_dev = 0, sent = 0;
        /* the steps in the CPU's order: the layers before the loop, its passes, the layers after */
        const int lp = c->loop_p > 1, ls = lp ? c->loop_s : L, le = lp ? c->loop_e : L;
        const int steps = L + (lp ? (c->loop_p - 1) * (le - ls) : 0);
        for (int st = 0; st < steps && ok; st++) {
            int i, v, pass = 0;
            if (st < le) i = v = st;
            else if (st < le + (c->loop_p - 1) * (le - ls)) {
                pass = 1 + (st - le) / (le - ls); i = ls + (st - le) % (le - ls); v = ink_sl(c, pass, i);
            } else i = v = st - (lp ? (c->loop_p - 1) * (le - ls) : 0);
            Layer *l = &m->L[i]; Ink2Layer *x2 = l->x2;
            int local = c->local[i];
            if (pending) ok = ink2c_join(ch, m, pi, pv, n, prev_dev ? ch->dm.y : ch->routed);
            const int dl = dmoe && n <= ch->kvl_rows && ink2c_dmoe_on(ch, m, i);   /* its experts here, no wait */
            const int msg = lp && pass > 0;   /* a later pass: this layer's four messages */
            if (ok && lp && i == ls) {
                if (pass == 0) {   /* the loop's input stream, which every pass starts from */
                    ok = vkc_copy(ch->xpre, 0, ch->x, 0, (size_t)n * D);
                    preB = B;
                } else {   /* the last pass's output and its block states but its first, for the messages */
                    nprev = B - preB - 1;
                    ok = vkc_copy(ch->xprev, 0, ch->x, 0, (size_t)n * D) && vkc_copy(ch->x, 0, ch->xpre, 0, (size_t)n * D);
                    if (ok && nprev > 0) {
                        VkcRegion *rg = malloc(sizeof *rg * (size_t)n);
                        if (!rg) { ok = 0; break; }
                        for (int s = 0; s < n; s++)
                            rg[s] = (VkcRegion){(size_t)s * ch->maxb * D, ((size_t)s * ch->maxb + preB + 1) * D, (size_t)nprev * D};
                        ok = vkc_copy_regions(ch->pblk, ch->blk, rg, n);
                        free(rg);
                    }
                    B = preB;
                }
            }
            if (ok && msg)   /* the messages: the mixes of the last pass's blocks and output, q k v mlp */
                for (int g = 0; g < 4 && ok; g++)
                    ok = ink2c_mix_of(ch, ch->pblk, ch->xprev, nprev, ch->o_lq + ((size_t)(i - ls) * 4 + g) * D, ch->msg[g], n, D);
            /* the attention's input: the mix of the block states so far and the stream (layer 0: the stream) */
            VkcBuf *iq = ch->x, *ik = ch->x, *iv = ch->x;
            if (ok && x2->aq) {
                ok = ink2c_mix(ch, ch->x, B, ch->o_aq[i], ch->hq, n, D);
                if (ok && !local) ok = ink2c_mix(ch, ch->x, B, ch->o_aq[i] + D, ch->hk, n, D) &&
                                       ink2c_mix(ch, ch->x, B, ch->o_aq[i] + 2 * (size_t)D, ch->hv, n, D);
                iq = ch->hq; ik = local ? ch->hq : ch->hk; iv = local ? ch->hq : ch->hv;
            }
            if (ok && msg) {   /* each input gated with its message (a sliding layer's one mix three ways), q last: in place */
                ok = ink2c_loop_gate(ch, m, i, 1, ik, ch->hk, n) && ink2c_loop_gate(ch, m, i, 2, iv, ch->hv, n) &&
                     ink2c_loop_gate(ch, m, i, 0, iq, ch->hq, n);
                iq = ch->hq; ik = ch->hk; iv = ch->hv;
            }
            const int sep = !local || msg;
            ok = ok && ink2c_norm(iq, ch->prm, ch->o_n1[i], ch->nq, n, D, c->eps);
            if (sep) ok = ok && ink2c_norm(ik, ch->prm, local ? ch->o_n1[i] : ch->o_n2[i], ch->nk, n, D, c->eps) &&
                              ink2c_norm(iv, ch->prm, local ? ch->o_n1[i] : ch->o_n3[i], ch->nv, n, D, c->eps);
            {   /* the step's K/V rows: kvd for the host's step after this frame, kvl's region when the frame goes on */
                const size_t reg = (size_t)2 * ch->kvl_rows * ch->kvo_max;
                ok = ok && (dl ? ink2c_attention(ch, m, i, v, sep, n, pb, ch->kvl, (size_t)v * reg, reg / 2)
                               : ink2c_attention(ch, m, i, v, sep, n, pb, ch->kvd, 0, (size_t)R * ch->kvo_max)) &&
                     ink2c_sconv(ch, ch->a, 0, n, D, 2, i, v, c);
            }
            /* a block boundary keeps this layer's input stream as a block state and starts the
             * stream over from the attention's output; elsewhere the output adds to it */
            if (ok && i % blkn == 0) {
                VkcRegion *rg = malloc(sizeof *rg * (size_t)n);
                if (!rg) { ok = 0; break; }
                for (int s = 0; s < n; s++) rg[s] = (VkcRegion){((size_t)s * ch->maxb + B) * D, (size_t)s * D, (size_t)D};
                ok = vkc_copy_regions(ch->blk, ch->x, rg, n) && vkc_copy(ch->res, 0, ch->a, 0, (size_t)n * D);
                free(rg);
                B++;
            } else if (ok) {
                VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = vkc_ew(ch->res, ch->x, ch->a, NULL, NULL, &add);
            }
            /* the MLP's input mix (a later pass: gated with its message), its norm, the router */
            ok = ok && ink2c_mix(ch, ch->res, B, ch->o_mq[i], ch->hq, n, D) &&
                 (!msg || ink2c_loop_gate(ch, m, i, 3, ch->hq, ch->hq, n)) &&
                 ink2c_norm(ch->hq, ch->prm, ch->o_post[i], ch->nq, n, D, c->eps) &&
                 vkc_matmul(ch->t_router[i], ch->nq, 0, ch->lg, 0, n);
            if (dl) {   /* Inkling-2's routing and SiTU-GLU experts here (moe_ex's), and on */
                VkcRoute rp = {.kind = 1, .b_off = (int)ch->o_rb[i], .has_bias = 1, .n_group = 1, .topk_group = 1, .norm = 1,
                               .temp = 1.f, .scale = c->route_scale * l->rgs, .norm_eps = 0.f};
                VkcMoe mp = {.D = D, .I = c->moe_inter, .act = 1, .a = c->gate_cap, .b = c->up_cap, .blocks = m->gx != 0};
                ok = ok && vkc_dmoe_layer(&ch->dm, i, n, ch->lg, ch->prm, ch->nq, &rp, &mp);
                if (ok) { dq[2 * ndq] = i; dq[2 * ndq + 1] = v; ndq++; }
                pending = 1; pi = i; pv = v; prev_dev = 1;
                if (ok && st + 1 < steps) ok = vkc_layer_recorded(&sent);   /* the device starts on these layers */
                continue;
            }
            ok = ok && vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && vkc_copy(ch->h2d, 0, ch->nq, 0, (size_t)n * D);
            double t0 = now_s();
            ok = ok && vkc_submit(0);
            if (ok) vkt_stream_prefetch(i, n);
            ok = ok && vkc_finish();
            m->t_attn += now_s() - t0;
            if (!ok) break;
            for (int k = 0; k < ndq; k++) ink2c_dmoe_done(ch, m, dq[2 * k], dq[2 * k + 1], n, pb);
            ndq = 0;
            ink2c_kv_down(ch, m, v, n, pb, (const float *)vkc_ptr(ch->kvd), (size_t)ch->rows * ch->kvo_max);
            double t1 = now_s();
            moe_ex(m, l, i, (float *)vkc_ptr(ch->h2d), n, ch->host_routed, (const float *)vkc_ptr(ch->lgd), NULL);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = vkc_begin();
            if (ok && dmoe) ok = vkc_dmoe_upload(&ch->dm);   /* the host's step ran the tier: the table again */
            pending = 1; pi = i; pv = v; prev_dev = 0;
        }
        if (ok && pending) ok = ink2c_join(ch, m, pi, pv, n, prev_dev ? ch->dm.y : ch->routed);
        /* the output's mix of the block states and the last stream: the rows the final norm reads */
        ok = ok && ink2c_mix(ch, ch->x, B, ch->o_outq, ch->hq, n, D);
        if (ok && want_x) ok = vkc_copy(ch->xd, 0, ch->hq, 0, (size_t)n * D);
        if (ok && ch->csd_n) ok = vkc_copy(ch->csd, 0, ch->rings, 0, ch->csd_n);   /* the convolution states, for the host */
        int last = c0 + n == S;
        if (ok && last) {
            VkcSconv dv = {2, 0, 0, 0, 0, 0, 0, 0, D, c->mup};
            VkcNorm fp = {1, D, 1, (n - 1) * D, D, D, 0, D, D, (int)ch->o_final, 0, 0, c->eps, 1.f};
            ok = vkc_norm(ch->hq, ch->prm, ch->fin, &fp) && vkc_sconv(ch->fin, NULL, NULL, &dv) &&
                 vkc_matmul(ch->t_lm, ch->fin, 0, ch->outd, 0, 1);
        }
        double t0 = now_s();
        ok = ok && vkc_submit(1);
        m->t_attn += now_s() - t0;
        if (!ok) goto lost;
        for (int k = 0; k < ndq; k++) ink2c_dmoe_done(ch, m, dq[2 * k], dq[2 * k + 1], n, pb);
        ndq = 0;
        const float *csd = (const float *)vkc_ptr(ch->csd);
        for (int i = 0; i < ch->NS; i++) for (int b = 0; b < 4; b++)
            memcpy(m->cs[b][i], csd + ch->o_csd[(size_t)b * ch->NS + i], (size_t)ink_cs_cells(c, b, i) * sizeof(float));
        ch->cs_where = INK2C_BOTH; ch->host_zero = 0;
        if (want_x) memcpy(xfin + (size_t)c0 * D, vkc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        if (last) {
            memcpy(logit, vkc_ptr(ch->outd), (size_t)c->unpad_vocab * sizeof(float));
            ink2_head_post(c, logit, c->unpad_vocab);
        }
    }
    if (want_x) { memcpy(xh, xfin, (size_t)S * D * sizeof(float)); free(xfin); }
    free(dq);
    vkc_relattn_part(NULL);
    ch->forwards++;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(0); }
    free(xfin); free(dq);
    vkc_relattn_part(NULL);
    ink2c_recover(m, pos_base);
    return 0;
}

static void ink2c_report(Model *m) {
    Ink2Chain *ch = ink2c_of(m);
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] inkling2 chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %llu layer steps' experts in the frame, "
                    "%.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms, ch->dm_layers, st.dev_bytes / 1048576.0);
    vkc_prof_print();
}

/* COLI_VK_CHAIN for Inkling-2, decided after the tier as for Inkling (inkc_start). */
static void ink2c_start(Model *m) {
    g_vk_chain = g_ink2c_want ? coli_vk_chain_decide("inkling2", vkt_ready(), INKLING_CHAIN_IGPU) : 0;
    if (g_vk_chain && g_ink_mux_slots > 1) {
        g_vk_chain = 0;
        fprintf(stderr, "[VK] inkling2: KV_SLOTS=%d: the dense chain is off (it keeps one conversation's state on the device); "
                        "the expert tier runs every conversation's experts\n", g_ink_mux_slots);
    }
    if (g_vk_chain && !ink2c_geometry_ok(m, 1)) g_vk_chain = 0;
    if (g_vk_chain && !vkc_init()) g_vk_chain = 0;
    if (g_vk_chain) atexit(vkc_shutdown_all);
}
