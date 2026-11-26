/* inkling2_cuchain.h -- Inkling-2's layers as a dense chain on a CUDA device (cuda_chain.h), the
 * twin of inkling2_chain.h (Vulkan): the same frames, state, loop and fallbacks, read there; the
 * ops are cuda_chain.h's, Inkling's and Inkling-2's in cuda_inkc.cuh. Included by inkling.c in a
 * COLI_CUDA build. COLI_CUDA_CHAIN decides: 1 on, 2 prompts only, 0 off; unset, on for
 * Inkling-2-Mini (the loop) where the backend has its ops (cc_ink2_caps), off for the others.
 * Measured on an RTX PRO 4000 with Inkling-2-Small int4 g64 (62 GB of RAM, the experts from RAM
 * and disk), 64 tokens on three prompts: 1.24-1.27 tok/s on against 1.26-1.29 off, the same text;
 * the dense part went from 4.7 to 3.4 s a run but the routed experts and the reads took that back
 * (the chain and the expert tier share the card), and the first prompt pays about 4.5 s to move
 * the matrices (below).
 *
 * The routed experts in the frame (Inkling-2-Mini's: every expert on the card): a step of up to
 * CC_MOE_FRAME_ROWS rows (decode) routes on the device (cc_route: the sigmoids, the bias for the
 * pick only, the K picks' sigmoids divided by their sum times route_scale) and computes a layer's
 * experts there (cc_moe: SiTU-GLU, act 1) when the CUDA tier holds all of that layer's on the
 * device (cut_dev_table: CcExpert entries by address, cu_xb.cu), then goes on to the next layer
 * with no wait: with every layer so a decode step is one frame. A GGUF's NVFP4 layers keep each
 * expert's global scale outside its blocks: the table carries them (CcMoe.esc_off; gate's and up's
 * one number, as the tier requires). The K/V rows of such steps wait in kvl (a region a state
 * layer) for the frame's end, their ids in dm_ids (one region a state layer) for the books (the
 * history, the hits, the tier's heat). The table is read again when the tier's residents moved
 * (cut_epoch: a host step ran the tier), at the start of a frame, never while one runs. TOPP and
 * ROUTE_TRACE take the host's step, as does a backend without CC_INK2_MOE.
 *
 * What differs from the Vulkan chain:
 *   - the dense bf16 matrices: the CUDA build keeps them on its own small backend
 *     (backend_cuda_ink.cu, a GEMV that rounds the activations to bf16). The chain needs the
 *     backend's tensors: at its first forward each matrix is read back once, freed there and
 *     uploaded as BF16 rows (a GGUF's type 30, the same bytes), so the card holds one copy.
 *     The per-matrix path then runs through those tensors (matmul_w's W.cut) where the chain
 *     declines a step.
 *   - prompt chunks of COLI_CUDA_CHAIN_ROWS rows (1024), as OLMoE's CUDA chain. */

#define INK2CC_HOST 0   /* the host's convolution states are newer than the device's */
#define INK2CC_ENT 32   /* the words of a cut_dev_table entry: backend_vulkan.h's COLI_VK_XB_ENTRY, which cu_xb.cu fills */
#ifdef COLI_VK_XB_ENTRY
_Static_assert(COLI_VK_XB_ENTRY == INK2CC_ENT, "the tier's entry words");
#endif
#define INK2CC_BOTH 1
#define INK2CC_DEV  2   /* the device's convolution states are newer than the host's (lazy sync) */
#define INK2G_SEGS 2
#define INK2G_WAYS 4
#define INK2G_FULL_WAYS 16
typedef struct { unsigned long long key, fp, tick; void *exec; } Ink2CcGraph;

typedef struct {
    int ok, failed, tried;
    int rows;                                  /* scratch capacity in rows */
    int nslot, q8_kv, fslot;                   /* KV_SLOTS (>1: per-slot packed Q8 KV & batched mux), current frame slot (-1 mux) */
    int max_t; float **hostK;                  /* single-slot FP32: the host cache the mirrors copy */
    CcBuf **slot_kv;                           /* [nslot] packed Q8 KV buffer per slot (q8_kv == 1) */
    int *smax, *scap, *scs_where, *shost_zero; /* [nslot] / [nslot*NS] per-slot Q8 KV & conv ring tracking */
    size_t *sq_kq, *sq_ks, *sq_vq, *sq_vs;     /* [nslot*NS] 128B-aligned byte offsets inside slot_kv[k] */
    CcBuf *mux_rb, *kv_tab;                    /* device ring base offsets [nslot] & CcInk2KvRow table [NS*nslot] */
    CcBuf *pack_segs, *pack_tiles, *pack_rseg; /* packed multi-prompt prefill device tables */
    int pack_nseg, pack_ntiles;
    CcInk2KvRow *kv_tabh; int *mux_rbh;        /* reusable host staging for kv_tab and mux_rb */
    int out_dev_rows;
    int maxb, hmax, kvo_max, qo_max, ro_max, dyn_max;
    CcBuf *prm;
    size_t *o_n1, *o_n2, *o_n3, *o_post, *o_qc, *o_rc, *o_relp, *o_cw[4], *o_aq, *o_mq, *o_rb, o_outq, o_final, o_lq;
    ColiCudaTensor *t_lm, **t_q, **t_k, **t_v, **t_r, **t_o, **t_ag, **t_sink, **t_router, **t_wp[2];
    ColiCudaTensor **t_lg;                     /* the loop's gates, [layer][q, k, v, mlp][down, forget, gain] */
    int NS;                                    /* state layers (n_state): kc, vc, cap, dlo, dhi, o_csd by them */
    CcBuf **kc, **vc, *rings;                 /* rings: every state layer's four convolution states, at o_csd */
    int *cap, *dlo, *dhi;
    size_t vs_off, *o_csd, csd_n;
    int cs_where, host_zero;
    CcBuf *x, *res, *blk, *hq, *hk, *hv, *nq, *nk, *nv, *q, *kvs, *r, *gt, *ctx, *a, *mo, *dyn, *ts, *lg, *fin;
    CcBuf *h2d, *lgd, *kvd, *csd, *outd, *xd, *routed;
    CcBuf *xpre, *xprev, *pblk, *msg[4], *hn2, *lz, *lf, *lgv;   /* the loop's passes */
    /* the routed experts in the frame: the table ([L][E] CcExpert, then the scales' pairs at
     * dm_esc_off bytes), per layer whether it runs there and cc_moe's formats; the ids [NS][8][K]
     * (down), the weights, the hidden rows and the routed sum */
    int dm_state;                              /* 0 untried, 1 on, -1 off */
    int dm_read; unsigned dm_epoch;
    uint8_t *dm_on; int *dm_fmt, *dm_gs, *dm_fmtd;
    uint32_t *dm_ent; CcExpert *dm_tabh; float *dm_esch;
    size_t dm_esc_off;
    CcBuf *dm_tab, *dm_ids, *dm_w, *dm_hid, *dm_y, *kvl;
    int pf_rows;
    CcBuf *pf_ids, *pf_w, *pf_hid, *pf_y;      /* prefill in-frame grouped MoE buffers (keeps decode graph addresses intact) */
    unsigned long long dm_layers, dm_reads;
    float *host_routed;
    unsigned long long forwards, mux_forwards, mux_rows;
    double host_ms;
    /* CUDA graphs (INK2_CC_GRAPH, default on): whole-forward single graph when all layers are in-frame,
     * falling back to per-step segments 0 (pre-attn) and 1 (post-attn+moe) */
    int g_on, g_steps;
    Ink2CcGraph *gr, *gr_full;
    CcBuf *pb_dev, *rel_part;
    CcBuf *out_dev, *pick_out;                 /* GPU greedy argmax (cc_pick): 8 B D2H instead of 800 KB logits */
    int kv_dev_lo, kv_dev_hi;                  /* lazy KV sync range on device [lo, hi) not yet copied to host m->K/V */
    unsigned long long g_tick, g_capture, g_launch, g_eager, g_full_capture, g_full_launch;
} Ink2CuChain;

static void ink2cc_fatal(const char *what) {
    fprintf(stderr, "[CUDA] inkling2 chain: %s -- stopping (COLI_CUDA_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
static int ink2cc_maxb(const Cfg *c) {
    int blk = c->res_block > 0 ? c->res_block : 1;
    return (c->n_layers + blk - 1) / blk;
}
static int ink2cc_geometry_ok(const Model *m, int say) {
    const Cfg *c = &m->c; int dr = c->d_rel, CK = c->conv_k;
    if (c->loop_p > 1 && !(cc_ink2_caps() & CC_INK2_LOOPGATE)) {
        if (say) fprintf(stderr, "[CUDA] inkling2 chain: the backend has no loop gate (an older build); the loop's passes "
                                 "(Inkling-2-Mini) run on the per-matrix path\n");
        return 0;
    }
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
 * convolution states, the K/V mirrors at the served context (ink_ctx_max), the loop's gates and
 * state, the routed experts' table and buffers and the scratch of a prompt chunk of 256 rows. The
 * tier's budget leaves them (ink_vk_dense_bytes). */
static size_t ink2cc_plan_bytes(const Model *m) {
    const Cfg *c = &m->c;
    size_t D = (size_t)c->hidden, E = (size_t)c->n_experts, CK = (size_t)c->conv_k, K = (size_t)c->topk, b = 0, row = 0;
    /* KV mirrors grow dynamically with actual prompt length (kv_alloc), so reserving for
     * a 1M-token CTX_MAX upfront would starve the VRAM expert tier. Cap the upfront KV
     * planning horizon to 65536 tokens while keeping dynamic growth beyond 64K possible. */
    int t = ink_ctx_max();
    if (g_ink_mux_slots > 1) {
        if (t > 16384) t = 16384;
    } else if (t > 65536) t = 65536;
    t += 8;
    const int q8_plan = g_ink_mux_slots > 1;
    size_t kvm = 0;
    for (int v = 0; v < c->n_state; v++) { size_t kvo = (size_t)L_KV(c, v) * L_HD(c, v); if (kvo > kvm) kvm = kvo; }
    for (int v = c->n_layers; v < c->n_state; v++) {                   /* the loop's later passes' state */
        size_t kvo = (size_t)L_KV(c, v) * L_HD(c, v), kvb = q8_plan ? (size_t)L_KV(c, v) * (L_HD(c, v) + 4) : kvo * 4;
        b += (2 * kvo + 2 * D) * (CK - 1) * 4 * (q8_plan ? (size_t)g_ink_mux_slots : 2) + 2 * kvb * (size_t)kv_ring_rows(c, v, t);
    }
    if (c->loop_p > 1) {                                               /* the gates and the queries */
        size_t nl = (size_t)(c->loop_e - c->loop_s), r = (size_t)c->loop_r;
        b += nl * 4 * (r * 2 * D + 2 * D * r) * 4 + nl * 4 * D * 4;
    }
    for (int i = 0; i < c->n_layers; i++) {
        size_t H = (size_t)L_HEADS(c, i), KV = (size_t)L_KV(c, i), hd = (size_t)L_HD(c, i), kvo = KV * hd;
        size_t kvb = q8_plan ? KV * (hd + 4) : kvo * 4;
        b += (E + H) * D * 4;                                          /* router, sinks */
        if (!c->local[i]) b += 2 * KV * CK * D * 4;                   /* the dynamic taps' projections */
        b += (6 * D + H * hd + H * c->d_rel + (size_t)c->d_rel * L_EXT(c, i) + 2 * kvo * CK + 2 * D * CK + E) * 4;
        b += (2 * kvo + 2 * D) * (CK - 1) * 4 * (q8_plan ? (size_t)g_ink_mux_slots : 2);
        b += 2 * kvb * (size_t)kv_ring_rows(c, i, t);                 /* the K/V mirror (slot 0) */
        size_t r1 = (14 * D + 3 * H * hd + 4 * kvo + H * c->d_rel + 2 * E + KV * CK + 2 * H) * 4;
        if (r1 > row) row = r1;
    }
    if (q8_plan) {
        /* Reserve initial Q8 KV headroom for up to 8 active subagent slots (at 2048 tokens) so the
         * CUDA expert tier still keeps 100% of experts resident while leaving ample VRAM for 64 slots. */
        int extra = g_ink_mux_slots - 1;
        if (extra > 8) extra = 8;
        int tk = ink_ctx_max(); if (tk > 2048) tk = 2048;
        for (int v = 0; v < c->n_state; v++)
            b += (size_t)extra * 2 * (size_t)L_KV(c, v) * (L_HD(c, v) + 4) * (size_t)kv_ring_rows(c, v, tk);
    }
    if (c->loop_p > 1) row += ((size_t)(7 + ink2cc_maxb(c)) * D + (size_t)c->loop_r) * 4;   /* a row's loop scratch */
    /* the routed experts in the frame: the table and its scales, the ids, the weights, the hidden rows, the sum, kvl */
    const size_t R8 = CC_MOE_FRAME_ROWS;
    b += (size_t)c->n_layers * E * (sizeof(CcExpert) + 8) + (size_t)c->n_state * R8 * K * 4 + R8 * K * 4 +
         cc_moe_hid_floats((int)R8, (int)K, (int)D, c->moe_inter, (int)E) * 4 + R8 * D * 4 + (size_t)c->n_state * 2 * R8 * kvm * 4;
    return b + row * 256 + (size_t)c->unpad_vocab * 4 + ((size_t)ink2cc_maxb(c) * D * 4) * 256;
}

static Ink2CuChain *ink2cc_of(Model *m) { return (Ink2CuChain *)m->ink2cuchain; }
static Ink2CuChain *ink2cc_new(Model *m) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (ch) return ch;
    if (!(ch = calloc(1, sizeof *ch))) return NULL;
    m->ink2cuchain = ch;
    int L = m->c.n_layers, NS = m->c.n_state, NK = g_ink_mux_slots > 1 ? g_ink_mux_slots : 1;
    ch->NS = NS; ch->nslot = NK;
#define INK2CC_ARR(f, T) (ch->f = calloc((size_t)L, sizeof(T)))
#define INK2CC_ARS(f, T) (ch->f = calloc((size_t)NS, sizeof(T)))
#define INK2CC_ARK(f, T) (ch->f = calloc((size_t)NS * NK, sizeof(T)))
    if (!INK2CC_ARR(o_n1, size_t) || !INK2CC_ARR(o_n2, size_t) || !INK2CC_ARR(o_n3, size_t) || !INK2CC_ARR(o_post, size_t) ||
        !INK2CC_ARR(o_qc, size_t) || !INK2CC_ARR(o_rc, size_t) || !INK2CC_ARR(o_relp, size_t) || !INK2CC_ARR(o_cw[0], size_t) ||
        !INK2CC_ARR(o_cw[1], size_t) || !INK2CC_ARR(o_cw[2], size_t) || !INK2CC_ARR(o_cw[3], size_t) || !INK2CC_ARR(o_aq, size_t) ||
        !INK2CC_ARR(o_mq, size_t) || !INK2CC_ARR(o_rb, size_t) || !INK2CC_ARR(t_q, void *) || !INK2CC_ARR(t_k, void *) ||
        !INK2CC_ARR(t_v, void *) || !INK2CC_ARR(t_r, void *) || !INK2CC_ARR(t_o, void *) || !INK2CC_ARR(t_ag, void *) ||
        !INK2CC_ARR(t_sink, void *) || !INK2CC_ARR(t_router, void *) || !INK2CC_ARR(t_wp[0], void *) || !INK2CC_ARR(t_wp[1], void *) ||
        !INK2CC_ARS(kc, void *) || !INK2CC_ARS(vc, void *) || !INK2CC_ARS(cap, int) || !INK2CC_ARS(dlo, int) ||
        !INK2CC_ARS(dhi, int) || !INK2CC_ARK(scap, int) || !INK2CC_ARK(sq_kq, size_t) || !INK2CC_ARK(sq_ks, size_t) ||
        !INK2CC_ARK(sq_vq, size_t) || !INK2CC_ARK(sq_vs, size_t)) return NULL;
#undef INK2CC_ARR
#undef INK2CC_ARS
#undef INK2CC_ARK
    if (!(ch->slot_kv = calloc((size_t)NK, sizeof(CcBuf *))) || !(ch->smax = calloc((size_t)NK, sizeof(int))) ||
        !(ch->scs_where = calloc((size_t)NK, sizeof(int))) || !(ch->shost_zero = calloc((size_t)NK, sizeof(int))) ||
        !(ch->kv_tabh = calloc((size_t)NS * NK, sizeof(CcInk2KvRow))) || !(ch->mux_rbh = calloc((size_t)NK, sizeof(int)))) return NULL;
    if (!(ch->o_csd = calloc((size_t)4 * NS, sizeof(size_t))) || !(ch->t_lg = calloc((size_t)L * 12, sizeof(void *)))) return NULL;
    return ch;
}

static void ink2cc_free_graphs(Ink2CuChain *ch) {
    if (!ch) return;
    if (ch->gr) {
        for (size_t k = 0; k < (size_t)ch->g_steps * INK2G_SEGS * INK2G_WAYS; k++) {
            cc_graph_free(ch->gr[k].exec);
            ch->gr[k].exec = NULL;
        }
    }
    if (ch->gr_full) {
        for (size_t k = 0; k < INK2G_FULL_WAYS; k++) {
            cc_graph_free(ch->gr_full[k].exec);
            ch->gr_full[k].exec = NULL;
        }
    }
}
/* The chain's own device memory freed (the per-matrix path's copies of the dense matrices stay). */
static void ink2cc_free_own(Model *m, Ink2CuChain *ch) {
    if (cc_ready()) cc_finish();
    ink2cc_free_graphs(ch);
    free(ch->gr); ch->gr = NULL;
    free(ch->gr_full); ch->gr_full = NULL;
    ch->g_on = 0;
    cc_free(ch->pb_dev); ch->pb_dev = NULL;
    cc_free(ch->rel_part); ch->rel_part = NULL;
    cc_free(ch->out_dev); ch->out_dev = NULL;
    cc_free(ch->pick_out); ch->pick_out = NULL;
    ch->kv_dev_lo = ch->kv_dev_hi = 0;
    for (int i = 0; i < m->c.n_layers; i++) {
        ColiCudaTensor **own[] = {&ch->t_sink[i], &ch->t_router[i], &ch->t_wp[0][i], &ch->t_wp[1][i]};
        for (size_t k = 0; k < sizeof own / sizeof own[0]; k++) { if (*own[k]) coli_cuda_tensor_free(*own[k]); *own[k] = NULL; }
        ch->t_q[i] = ch->t_k[i] = ch->t_v[i] = ch->t_r[i] = ch->t_o[i] = ch->t_ag[i] = NULL;   /* the per-matrix path's */
        for (int k = 0; k < 12; k++) { ColiCudaTensor **t = &ch->t_lg[(size_t)i * 12 + k]; if (*t) coli_cuda_tensor_free(*t); *t = NULL; }
    }
    for (int v = 0; v < ch->NS; v++) { cc_free(ch->kc[v]); cc_free(ch->vc[v]); ch->kc[v] = ch->vc[v] = NULL; }
    for (int k = 0; k < ch->nslot; k++) { cc_free(ch->slot_kv[k]); ch->slot_kv[k] = NULL; ch->smax[k] = 0; }
    cc_free(ch->mux_rb); ch->mux_rb = NULL;
    cc_free(ch->kv_tab); ch->kv_tab = NULL;
    cc_free(ch->pack_segs); ch->pack_segs = NULL;
    cc_free(ch->pack_tiles); ch->pack_tiles = NULL;
    cc_free(ch->pack_rseg); ch->pack_rseg = NULL;
    ch->out_dev_rows = 0;
    ch->t_lm = NULL;
    CcBuf **dm[] = {&ch->dm_tab, &ch->dm_ids, &ch->dm_w, &ch->dm_hid, &ch->dm_y, &ch->kvl,
                    &ch->pf_ids, &ch->pf_w, &ch->pf_hid, &ch->pf_y};
    for (size_t k = 0; k < sizeof dm / sizeof dm[0]; k++) { cc_free(*dm[k]); *dm[k] = NULL; }
    ch->dm_read = 0; ch->dm_state = -1; ch->pf_rows = 0;
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
/* an f32 matrix [O][I] of the chain's own (the router, the sinks, the dynamic taps' projections, the loop's gates) */
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
        else if (c->loop_p > 1 && i >= c->loop_s && i < c->loop_e)
            for (int g = 0; g < 4 && !what; g++) {
                ColiCudaTensor **t = &ch->t_lg[((size_t)i * 4 + g) * 3];
                if (!ink2cc_f32(&t[0], x2->lg_down[g], 2 * D, c->loop_r) || !ink2cc_f32(&t[1], x2->lg_fgt[g], c->loop_r, D) ||
                    !ink2cc_f32(&t[2], x2->lg_gain[g], c->loop_r, D))
                    what = "a loop gate";
            }
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
        ch->o_rb[i] = n; n += (size_t)E;
    }
    ch->o_outq = n; n += D;
    ch->o_final = n; n += D;
    ch->o_lq = n; if (c->loop_p > 1) n += (size_t)(c->loop_e - c->loop_s) * 4 * D;
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
        if (l->rbias) memcpy(arena + ch->o_rb[i], l->rbias, (size_t)E * sizeof(float));
    }
    memcpy(arena + ch->o_outq, m->x2_out_query, D * sizeof(float));
    memcpy(arena + ch->o_final, m->final_norm, D * sizeof(float));
    if (c->loop_p > 1) memcpy(arena + ch->o_lq, m->x2_loopq, (size_t)(c->loop_e - c->loop_s) * 4 * D * sizeof(float));
    ch->prm = cc_buf(n * sizeof(float), CC_DEV);
    int ok = ch->prm && cc_begin() && cc_write(ch->prm, 0, arena, n * sizeof(float)) && cc_submit(1);
    free(arena);
    if (!ok) { ink2cc_free_own(m, ch); return NULL; }
    ch->maxb = ink2cc_maxb(c);
    ch->csd_n = 0;   /* the states one after another, the same layout on the device and in the read-back */
    for (int v = 0; v < ch->NS; v++)
        for (int b = 0; b < 4; b++) { ch->o_csd[(size_t)b * ch->NS + v] = ch->csd_n; ch->csd_n += (size_t)ink_cs_cells(c, b, v); }
    ch->q8_kv = (ch->nslot > 1 && cc_ink2_mux_ready() && cc_dev_addr_ready()) ? 1 : 0;
    if (!(ch->rings = cc_buf((ch->csd_n ? ch->csd_n : 1) * (size_t)ch->nslot * sizeof(float), CC_DEV))) { ink2cc_free_own(m, ch); return NULL; }
    if (ch->q8_kv) {
        if (!(ch->mux_rb = cc_buf((size_t)ch->nslot * sizeof(int), CC_DEV)) ||
            !(ch->kv_tab = cc_buf((size_t)ch->NS * (size_t)ch->nslot * sizeof(CcInk2KvRow), CC_DEV))) {
            ink2cc_free_own(m, ch); return NULL;
        }
    }
    ch->cs_where = INK2CC_HOST; ch->host_zero = 0;
    for (int k = 0; k < ch->nslot; k++) { ch->scs_where[k] = INK2CC_HOST; ch->shost_zero[k] = 0; }
    {
        const char *ge = getenv("INK2_CC_GRAPH");
        const int g_want = !(ge && *ge == '0');
        const int lp = c->loop_p > 1, ls = lp ? c->loop_s : L, le = lp ? c->loop_e : L;
        ch->g_steps = L + (lp ? (c->loop_p - 1) * (le - ls) : 0);
        if (g_want && cc_graph_ready() && cc_dev_addr_ready()) {
            if (!ch->q8_kv) ch->gr = calloc((size_t)ch->g_steps * INK2G_SEGS * INK2G_WAYS, sizeof *ch->gr);
            ch->gr_full = calloc(INK2G_FULL_WAYS, sizeof *ch->gr_full);
            if (ch->gr_full && (ch->q8_kv || ch->gr)) ch->g_on = 1;
        }
    }
    ch->ok = 1;
    if (c->loop_p > 1)
        fprintf(stderr, "[CUDA] inkling2 chain: %d layers on the device, layers %d-%d %d passes, %.1f MiB of parameters%s\n",
                L, c->loop_s, c->loop_e - 1, c->loop_p, n * 4 / 1048576.0,
                ch->q8_kv ? " (multi-slot Q8 KV)" : "");
    else fprintf(stderr, "[CUDA] inkling2 chain: %d layers on the device, %.1f MiB of parameters%s\n", L, n * 4 / 1048576.0,
                 ch->q8_kv ? " (multi-slot Q8 KV)" : "");
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
           cc_reserve(&ch->outd, (size_t)c->unpad_vocab * 4, CC_DOWN) && cc_reserve(&ch->routed, r * D * 4, CC_UP) &&
           cc_reserve(&ch->pb_dev, sizeof(int), CC_DEV) &&
           (c->loop_p <= 1 ||
            (cc_reserve(&ch->xpre, r * D * 4, CC_DEV) && cc_reserve(&ch->xprev, r * D * 4, CC_DEV) &&
             cc_reserve(&ch->pblk, r * ch->maxb * D * 4, CC_DEV) && cc_reserve(&ch->msg[0], r * D * 4, CC_DEV) &&
             cc_reserve(&ch->msg[1], r * D * 4, CC_DEV) && cc_reserve(&ch->msg[2], r * D * 4, CC_DEV) &&
             cc_reserve(&ch->msg[3], r * D * 4, CC_DEV) && cc_reserve(&ch->hn2, r * 2 * D * 4, CC_DEV) &&
             cc_reserve(&ch->lz, r * (size_t)c->loop_r * 4, CC_DEV) && cc_reserve(&ch->lf, r * D * 4, CC_DEV) &&
             cc_reserve(&ch->lgv, r * D * 4, CC_DEV)));
}
/* Prompt rows per chunk: COLI_CUDA_CHAIN_ROWS (1 to 8192), default 4096 for Inkling-2-Mini
 * so each grouped MoE expert receives ~256 rows/chunk (enabling BM=256 tiles & 4x fewer weight reads vs 1024). */
static int ink2cc_chunk_rows(Ink2CuChain *ch, Model *m) {
    (void)ch; (void)m;
    const char *e = getenv("COLI_CUDA_CHAIN_ROWS");
    int r = e && *e ? atoi(e) : 4096;
    return r < 1 ? 1 : r > 8192 ? 8192 : r;
}
/* ---- multi-slot Q8 KV helpers (q8_kv == 1) ---- */
#define INK2CC_ALIGN128(x) (((size_t)(x) + 127u) & ~(size_t)127u)
typedef struct { float **K, **V, ***cs; int max_t, kv_len; } Ink2ccHost;
static int ink2cc_cur(void) { return g_ink_mux_slots > 1 && g_ink_mux_cur >= 0 ? g_ink_mux_cur : 0; }
static Ink2ccHost ink2cc_host(Model *m, int k) {
    if (g_ink_mux_slots <= 1 || k == g_ink_mux_cur) return (Ink2ccHost){m->K, m->V, m->cs, m->max_t, m->kv_len};
    InkSeq *q = &g_ink_mux_seq[k];
    return (Ink2ccHost){q->K, q->V, q->cs, q->max_t, q->kv_len};
}
/* Lazily allocate or grow slot k's single packed Q8 KV buffer in VRAM (115.5 MiB at 2048 tokens).
 * Rounded to 2048-token steps so subagents never reallocate; preserves live prefix rows [0..kv_len) on growth. */
static int ink2cc_ensure_slot(Ink2CuChain *ch, Model *m, int k) {
    Cfg *c = &m->c;
    Ink2ccHost hs = ink2cc_host(m, k);
    int need = hs.max_t;
    if (need < 1) need = 1;
    if (ch->slot_kv[k] && need <= ch->smax[k]) return 1;
    int step = 2048;
    int cap_alloc = ((need + step - 1) / step) * step;
    int lim = ink_ctx_max() + 64;
    if (cap_alloc > lim && need <= lim) cap_alloc = lim;
    if (cap_alloc < need) cap_alloc = need;
    const int NS = ch->NS;
    size_t *nkq = malloc((size_t)NS * sizeof(size_t)), *nks = malloc((size_t)NS * sizeof(size_t));
    size_t *nvq = malloc((size_t)NS * sizeof(size_t)), *nvs = malloc((size_t)NS * sizeof(size_t));
    int *ncap = malloc((size_t)NS * sizeof(int));
    if (!nkq || !nks || !nvq || !nvs || !ncap) { free(nkq); free(nks); free(nvq); free(nvs); free(ncap); return 0; }
    size_t off = 0;
    for (int v = 0; v < NS; v++) {
        int cap = kv_ring_rows(c, v, cap_alloc), KV = L_KV(c, v), hd = L_HD(c, v);
        size_t nrow = (size_t)KV * cap;
        ncap[v] = cap;
        nkq[v] = off; off += INK2CC_ALIGN128(nrow * (size_t)hd);
        nks[v] = off; off += INK2CC_ALIGN128(nrow * sizeof(float));
        nvq[v] = off; off += INK2CC_ALIGN128(nrow * (size_t)hd);
        nvs[v] = off; off += INK2CC_ALIGN128(nrow * sizeof(float));
    }
    CcBuf *nb = cc_buf(off, CC_DEV);
    if (!nb) { free(nkq); free(nks); free(nvq); free(nvs); free(ncap); return 0; }
    if (ch->slot_kv[k]) {
        int keep = hs.kv_len > 0 && hs.kv_len <= cap_alloc ? hs.kv_len : 0;
        if (keep > 0) {
            int ok = cc_begin();
            for (int v = 0; v < NS && ok; v++) {
                int x = k * NS + v, ocap = ch->scap[x], ncp = ncap[v], KV = L_KV(c, v), hd = L_HD(c, v);
                if (ocap == ncp) {
                    size_t qf = ((size_t)KV * ocap * hd) / 4, sf = (size_t)KV * ocap;
                    ok = cc_copy(nb, nkq[v] / 4, ch->slot_kv[k], ch->sq_kq[x] / 4, qf) &&
                         cc_copy(nb, nks[v] / 4, ch->slot_kv[k], ch->sq_ks[x] / 4, sf) &&
                         cc_copy(nb, nvq[v] / 4, ch->slot_kv[k], ch->sq_vq[x] / 4, qf) &&
                         cc_copy(nb, nvs[v] / 4, ch->slot_kv[k], ch->sq_vs[x] / 4, sf);
                } else if ((hd & 3) == 0) {
                    for (int h = 0; h < KV && ok; h++) {
                        size_t qf = ((size_t)keep * hd) / 4;
                        ok = cc_copy(nb, (nkq[v] + (size_t)h * ncp * hd) / 4, ch->slot_kv[k], (ch->sq_kq[x] + (size_t)h * ocap * hd) / 4, qf) &&
                             cc_copy(nb, nks[v] / 4 + (size_t)h * ncp, ch->slot_kv[k], ch->sq_ks[x] / 4 + (size_t)h * ocap, (size_t)keep) &&
                             cc_copy(nb, (nvq[v] + (size_t)h * ncp * hd) / 4, ch->slot_kv[k], (ch->sq_vq[x] + (size_t)h * ocap * hd) / 4, qf) &&
                             cc_copy(nb, nvs[v] / 4 + (size_t)h * ncp, ch->slot_kv[k], ch->sq_vs[x] / 4 + (size_t)h * ocap, (size_t)keep);
                    }
                }
            }
            ok = ok && cc_submit(1);
            if (!ok) { cc_free(nb); free(nkq); free(nks); free(nvq); free(nvs); free(ncap); return 0; }
        }
        cc_free(ch->slot_kv[k]);
    }
    ch->slot_kv[k] = nb;
    ch->smax[k] = cap_alloc;
    for (int v = 0; v < NS; v++) {
        int x = k * NS + v;
        ch->scap[x] = ncap[v];
        ch->sq_kq[x] = nkq[v]; ch->sq_ks[x] = nks[v];
        ch->sq_vq[x] = nvq[v]; ch->sq_vs[x] = nvs[v];
    }
    free(nkq); free(nks); free(nvq); free(nvs); free(ncap);
    return 1;
}
/* the K/V mirrors at the host's layout, made again when the host's cache was */
static int ink2cc_mirror(Ink2CuChain *ch, Model *m) {
    if (ch->q8_kv) return ink2cc_ensure_slot(ch, m, ink2cc_cur());
    Cfg *c = &m->c;
    if (ch->max_t == m->max_t && ch->hostK == m->K) return 1;
    for (int i = 0; i < ch->NS; i++) {
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
    if (ch->q8_kv) {
        int k = ink2cc_cur();
        ch->scs_where[k] = INK2CC_HOST; ch->shost_zero[k] = zero;
        return;
    }
    ch->cs_where = INK2CC_HOST; ch->host_zero = zero;
    ch->kv_dev_lo = ch->kv_dev_hi = 0;
}
/* Pull device KV rows [kv_dev_lo, kv_dev_hi) back to host m->K/V only when needed (lazy sync). */
static void ink2cc_sync_kv(Model *m) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (!ch || !ch->ok || ch->q8_kv || ch->kv_dev_lo >= ch->kv_dev_hi || cc_lost()) return;
    Cfg *c = &m->c;
    for (int v = 0; v < ch->NS; v++) {
        if (!ch->kc[v] || !ch->vc[v] || !m->K || !m->K[v] || !m->V || !m->V[v]) continue;
        int cap = ch->cap[v], hd = L_HD(c, v), KV = L_KV(c, v);
        int lo = ch->kv_dev_hi - cap > ch->kv_dev_lo ? ch->kv_dev_hi - cap : ch->kv_dev_lo;
        for (int t = lo; t < ch->kv_dev_hi; ) {
            int row = t % cap, run = ch->kv_dev_hi - t < cap - row ? ch->kv_dev_hi - t : cap - row;
            for (int h = 0; h < KV; h++) {
                size_t off = ((size_t)h * cap + row) * hd;
                cc_read(ch->kc[v], off, m->K[v] + off, (size_t)run * hd * sizeof(float));
                cc_read(ch->vc[v], off, m->V[v] + off, (size_t)run * hd * sizeof(float));
            }
            t += run;
        }
    }
    ch->kv_dev_lo = ch->kv_dev_hi = 0;
}
/* Pull device convolution rings (and any lazy KV rows) back to host only when the device is newer. */
static void ink2cc_sync_cs(Model *m) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (!ch || !ch->ok || cc_lost()) return;
    if (ch->q8_kv) {
        int k = ink2cc_cur();
        if (ch->scs_where[k] != INK2CC_DEV || !ch->csd_n || !ch->csd) return;
        if (!cc_begin() || !cc_copy(ch->csd, 0, ch->rings, (size_t)k * ch->csd_n, ch->csd_n) || !cc_submit(1)) return;
        Cfg *c = &m->c;
        const float *csd = (const float *)cc_ptr(ch->csd);
        for (int i = 0; i < ch->NS; i++) for (int b = 0; b < 4; b++)
            memcpy(m->cs[b][i], csd + ch->o_csd[(size_t)b * ch->NS + i], (size_t)ink_cs_cells(c, b, i) * sizeof(float));
        ch->scs_where[k] = INK2CC_BOTH; ch->shost_zero[k] = 0;
        return;
    }
    ink2cc_sync_kv(m);
    if (ch->cs_where != INK2CC_DEV || !ch->csd_n) return;
    if (!cc_begin() || !cc_copy(ch->csd, 0, ch->rings, 0, ch->csd_n) || !cc_submit(1)) return;
    Cfg *c = &m->c;
    const float *csd = (const float *)cc_ptr(ch->csd);
    for (int i = 0; i < ch->NS; i++) for (int b = 0; b < 4; b++)
        memcpy(m->cs[b][i], csd + ch->o_csd[(size_t)b * ch->NS + i], (size_t)ink_cs_cells(c, b, i) * sizeof(float));
    ch->cs_where = INK2CC_BOTH; ch->host_zero = 0;
}
static int ink2cc_push_state_slots(Ink2CuChain *ch, Model *m, const int *rk, int n) {
    Cfg *c = &m->c; int ok = 1;
    for (int s = 0; s < n && ok; s++) {
        int k = rk[s];
        if (ch->scs_where[k] != INK2CC_HOST) continue;
        size_t rb = (size_t)k * ch->csd_n;
        if (ch->shost_zero[k]) ok = !ch->csd_n || cc_zero(ch->rings, rb, ch->csd_n);
        else {
            Ink2ccHost hs = ink2cc_host(m, k);
            for (int i = 0; i < ch->NS && ok; i++) for (int b = 0; b < 4 && ok; b++) {
                size_t nc = (size_t)ink_cs_cells(c, b, i);
                if (nc) ok = cc_write(ch->rings, rb + ch->o_csd[(size_t)b * ch->NS + i], hs.cs[b][i], nc * sizeof(float));
            }
        }
        ch->scs_where[k] = INK2CC_BOTH; ch->shost_zero[k] = 0;
    }
    return ok;
}
static void ink2cc_cpu_step(Model *m, int pos_base, int S) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (!ch || !ch->ok) return;
    ch->cs_where = INK2CC_HOST; ch->host_zero = 0;
    for (int i = 0; i < ch->NS; i++) {
        if (ch->dlo[i] >= ch->dhi[i]) { ch->dlo[i] = pos_base; ch->dhi[i] = pos_base + S; continue; }
        if (pos_base < ch->dlo[i]) ch->dlo[i] = pos_base;
        if (pos_base + S > ch->dhi[i]) ch->dhi[i] = pos_base + S;
    }
}
static int ink2cc_push_state(Ink2CuChain *ch, Model *m) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->cs_where == INK2CC_HOST) {
        if (ch->host_zero) ok = !ch->csd_n || cc_zero(ch->rings, 0, ch->csd_n);
        else for (int i = 0; i < ch->NS && ok; i++) for (int b = 0; b < 4 && ok; b++) {
            size_t nc = (size_t)ink_cs_cells(c, b, i);
            if (nc) ok = cc_write(ch->rings, ch->o_csd[(size_t)b * ch->NS + i], m->cs[b][i], nc * sizeof(float));
        }
        ch->cs_where = INK2CC_BOTH;
    }
    for (int i = 0; i < ch->NS && ok; i++) {
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
    if (g_ink_mux_slots > 1) ink2cc_fatal("the device was lost with several conversations on it");
    g_cu_chain = 0;
    if (ch) ch->failed = 1;
    for (int i = 0; i < c->n_state; i++)
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
/* a static short convolution through bank `bank` and its scale: (conv + input) / sqrt(K + 1);
 * the taps of weight layer i, the state of state layer v (fused / sqrt(K + 1) in mode 0) */
static int ink2cc_sconv(Ink2CuChain *ch, CcBuf *x, size_t xo, int n, int C, int bank, int i, int v, const Cfg *c) {
    const size_t o_csd = ch->o_csd[(size_t)bank * ch->NS + v];
    if (ch->fslot == -2) {
        CcSconv p = {0, n, C, c->conv_k, (int)xo, C, (int)ch->o_cw[bank][i], (int)o_csd, 0,
                     (float)sqrt((double)(c->conv_k + 1))};
        return cc_sconv_packed(x, ch->prm, ch->rings, ch->pack_segs, ch->pack_nseg, &p);
    }
    if (ch->fslot < 0) {
        CcSconv p = {0, n, C, c->conv_k, (int)xo, C, (int)ch->o_cw[bank][i], (int)o_csd, 0,
                     (float)sqrt((double)(c->conv_k + 1))};
        return cc_sconv_mux(x, ch->prm, ch->rings, ch->mux_rb, &p);
    }
    const size_t rb = ch->fslot > 0 ? (size_t)ch->fslot * ch->csd_n : 0;
    CcSconv p = {0, n, C, c->conv_k, (int)xo, C, (int)ch->o_cw[bank][i], (int)(rb + o_csd), 0,
                 (float)sqrt((double)(c->conv_k + 1))};
    return cc_sconv(x, ch->prm, ch->rings, &p);
}
/* a global layer's dynamic convolution of K (bank 0, from nk) or V (bank 1, from nv) in kvs */
static int ink2cc_dconv(Ink2CuChain *ch, Model *m, int i, int v, int bank, CcBuf *xin, int n) {
    Cfg *c = &m->c; int KV = L_KV(c, i), hd = L_HD(c, i), K = c->conv_k;
    const size_t o_csd = ch->o_csd[(size_t)bank * ch->NS + v];
    if (ch->fslot == -2) {
        CcInk2Dconv p = {n, KV * hd, K, hd, bank ? (int)ch->vs_off : 0, KV * hd, 0, KV * K, (int)ch->o_cw[bank][i],
                         (int)o_csd, (float)sqrt((double)(2 * K + 1))};
        return cc_matmul(ch->t_wp[bank][i], xin, 0, ch->dyn, 0, n) &&
               cc_ink2_dconv_packed(ch->kvs, ch->dyn, ch->prm, ch->rings, ch->pack_segs, ch->pack_nseg, &p);
    }
    if (ch->fslot < 0) {
        CcInk2Dconv p = {n, KV * hd, K, hd, bank ? (int)ch->vs_off : 0, KV * hd, 0, KV * K, (int)ch->o_cw[bank][i],
                         (int)o_csd, (float)sqrt((double)(2 * K + 1))};
        return cc_matmul(ch->t_wp[bank][i], xin, 0, ch->dyn, 0, n) && cc_ink2_dconv_mux(ch->kvs, ch->dyn, ch->prm, ch->rings, ch->mux_rb, &p);
    }
    const size_t rb = ch->fslot > 0 ? (size_t)ch->fslot * ch->csd_n : 0;
    CcInk2Dconv p = {n, KV * hd, K, hd, bank ? (int)ch->vs_off : 0, KV * hd, 0, KV * K, (int)ch->o_cw[bank][i],
                      (int)(rb + o_csd), (float)sqrt((double)(2 * K + 1))};
    return cc_matmul(ch->t_wp[bank][i], xin, 0, ch->dyn, 0, n) && cc_ink2_dconv(ch->kvs, ch->dyn, ch->prm, ch->rings, &p);
}
/* attention pre-relattn (independent of pb): projections, conv, clip, v-norm */
static int ink2cc_attn_pre(Ink2CuChain *ch, Model *m, int i, int v, int sep, int n) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, dr = c->d_rel;
    int local = c->local[i], R = ch->rows;
    size_t vs = ch->vs_off;
    CcBuf *xk = sep ? ch->nk : ch->nq, *xv = sep ? ch->nv : ch->nq;
    int ok = cc_matmul(ch->t_q[i], ch->nq, 0, ch->q, 0, n) && cc_matmul(ch->t_k[i], xk, 0, ch->kvs, 0, n) &&
             cc_matmul(ch->t_v[i], xv, 0, ch->kvs, vs, n) && cc_matmul(ch->t_r[i], ch->nq, 0, ch->r, 0, n) &&
             cc_matmul(ch->t_ag[i], ch->nq, 0, ch->gt, 0, n) && cc_matmul(ch->t_sink[i], ch->nq, 0, ch->ts, (size_t)2 * R, n);
    if (local) ok = ok && ink2cc_sconv(ch, ch->kvs, 0, n, kvo, 0, i, v, c) && ink2cc_sconv(ch, ch->kvs, vs, n, kvo, 1, i, v, c);
    else ok = ok && ink2cc_dconv(ch, m, i, v, 0, ch->nk, n) && ink2cc_dconv(ch, m, i, v, 1, ch->nv, n);
    CcInk2Clip qc = {n * H, hd, H, 0, H * hd, 1, (int)ch->o_qc[i], c->qk_clip};
    CcInk2Clip rc = {n * H, dr, H, 0, H * dr, 1, (int)ch->o_rc[i], c->rel_clip};
    CcInk2Clip kc = {n * KV, hd, KV, 0, kvo, 0, 0, c->qk_clip};
    CcNorm vn = {n * KV, hd, KV, (int)vs, kvo, hd, (int)vs, kvo, hd, 0, 0, 2, c->eps, 1.f};
    return ok && cc_ink2_clip(ch->q, ch->prm, &qc) && cc_ink2_clip(ch->r, ch->prm, &rc) &&
           cc_ink2_clip(ch->kvs, NULL, &kc) && cc_norm(ch->kvs, ch->prm, ch->kvs, &vn);
}
/* attention dynamic part (depends on pb): relattn + KV cache scatter */
static int ink2cc_attn_dyn(Ink2CuChain *ch, Model *m, int i, int v, int n, int pb) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, dr = c->d_rel;
    int cap = ch->cap[v], local = c->local[i], R = ch->rows;
    size_t vs = ch->vs_off;
    CcRelAttn a = {n, H, KV, hd, pb, cap, local ? c->window : 0, L_EXT(c, i), dr,
                    0, H * hd, 0, H * hd, 0, 0, 0, (int)vs, kvo, 0, H * dr, (int)ch->o_relp[i], local ? R : 0,
                    (float)(1.0 / sqrt((double)hd)), 1, 2 * R};
    int ok = cc_relattn(ch->q, ch->kc[v], ch->vc[v], ch->ctx, ch->kvs, ch->r, ch->prm, ch->ts, &a);
    if (!ok) return 0;
    if (cc_ink2_pb_ready() && ch->pb_dev)
        return cc_ink2_kv_put(ch->kc[v], ch->vc[v], ch->kvs, ch->pb_dev, n, KV, hd, cap, (int)vs);
    int s0 = n > cap ? n - cap : 0, nr = (n - s0) * KV;
    CcRegion *rk = malloc(sizeof *rk * (size_t)nr * 2), *rv = rk ? rk + nr : NULL;
    if (!rk) return 0;
    for (int s = s0; s < n; s++) for (int h = 0; h < KV; h++) {
        size_t dst = ((size_t)h * cap + (size_t)(pb + s) % cap) * hd, src = (size_t)s * kvo + (size_t)h * hd;
        rk[(s - s0) * KV + h] = (CcRegion){dst, src, (size_t)hd};
        rv[(s - s0) * KV + h] = (CcRegion){dst, vs + src, (size_t)hd};
    }
    ok = cc_copy_regions(ch->kc[v], ch->kvs, rk, nr) && cc_copy_regions(ch->vc[v], ch->kvs, rv, nr);
    free(rk);
    return ok;
}
/* CUDA-Graph-safe attention dynamic part: reads pos_base from ch->pb_dev on device (no host pb, no malloc),
 * with preallocated split-K scratch buffer ch->rel_part (nc_glob for global layers, window-based nc for local). */
static int ink2cc_attn_dyn_pb(Ink2CuChain *ch, Model *m, int i, int v, int n, int nc_glob) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, dr = c->d_rel;
    int cap = ch->cap[v], local = c->local[i], R = ch->rows;
    size_t vs = ch->vs_off;
    int nc_loc = (c->window + 127) / 128;
    int nc = local ? (nc_loc < nc_glob ? nc_loc : nc_glob) : nc_glob;
    if (nc > 64) nc = 64;
    CcRelAttn a = {n, H, KV, hd, 0, cap, local ? c->window : 0, L_EXT(c, i), dr,
                    0, H * hd, 0, H * hd, 0, 0, 0, (int)vs, kvo, 0, H * dr, (int)ch->o_relp[i], local ? R : 0,
                    (float)(1.0 / sqrt((double)hd)), 1, 2 * R};
    return cc_relattn_pb(ch->q, ch->kc[v], ch->vc[v], ch->ctx, ch->kvs, ch->r, ch->prm, ch->ts,
                         ch->pb_dev, nc >= 2 ? ch->rel_part : NULL, nc, &a) &&
           cc_ink2_kv_put(ch->kc[v], ch->vc[v], ch->kvs, ch->pb_dev, n, KV, hd, cap, (int)vs);
}
/* Q8 KV attention + Q8 KV scatter for both single-slot prefill/decode (mux == 0) and batched mux decode (mux == 1). */
static int ink2cc_attn_dyn_q8(Ink2CuChain *ch, Model *m, int i, int v, int n, int pb, int max_pos, int mux, int tab_stride, int nc_cap) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, dr = c->d_rel;
    int local = c->local[i], R = ch->rows;
    size_t vs = ch->vs_off;
    int nc = 1;
    if (nc_cap >= 2 && (mux || n < 16)) {
        if (mux) {
            int nc_loc = (c->window + 127) / 128;
            nc = local ? (nc_loc < nc_cap ? nc_loc : nc_cap) : nc_cap;
        } else {
            int nmax = local ? (c->window < max_pos + 1 ? c->window : max_pos + 1) : (max_pos + 1);
            nc = (nmax + 127) / 128;
            if (nc > nc_cap) nc = nc_cap;
        }
    }
    CcRelAttn a = {n, H, KV, hd, pb, 1, local ? c->window : 0, L_EXT(c, i), dr,
                   0, H * hd, 0, H * hd, 0, 0, 0, (int)vs, kvo, 0, H * dr, (int)ch->o_relp[i], local ? R : 0,
                   (float)(1.0 / sqrt((double)hd)), 1, 2 * R};
    size_t toff = (size_t)v * (size_t)tab_stride;
    return cc_relattn_q8(ch->q, ch->kv_tab, toff, ch->ctx, ch->kvs, ch->r, ch->prm, ch->ts,
                         nc >= 2 ? ch->rel_part : NULL, nc, &a, mux) &&
           cc_ink2_kv_put_q8(ch->kv_tab, toff, ch->kvs, n, KV, hd, (int)vs, mux);
}
/* Packed multi-prompt prefill attention + Q8 KV scatter (fslot == -2). */
static int ink2cc_attn_dyn_q8_packed(Ink2CuChain *ch, Model *m, int i, int v, int N, int M) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd, dr = c->d_rel;
    int local = c->local[i], R = ch->rows;
    size_t vs = ch->vs_off;
    CcRelAttn a = {N, H, KV, hd, 0, 1, local ? c->window : 0, L_EXT(c, i), dr,
                   0, H * hd, 0, H * hd, 0, 0, 0, (int)vs, kvo, 0, H * dr, (int)ch->o_relp[i], local ? R : 0,
                   (float)(1.0 / sqrt((double)hd)), 1, 2 * R};
    size_t toff = (size_t)v * (size_t)M;
    return cc_relattn_q8_packed(ch->q, ch->kv_tab, toff, ch->ctx, ch->kvs, ch->r, ch->prm, ch->ts,
                                ch->pack_segs, ch->pack_tiles, ch->pack_ntiles, &a) &&
           cc_ink2_kv_put_q8_packed(ch->kv_tab, toff, ch->kvs, ch->pack_segs, ch->pack_rseg, N, KV, hd, (int)vs);
}
/* attention post-relattn (independent of pb): gate, optional K/V host-staging copy, o_proj */
static int ink2cc_attn_post(Ink2CuChain *ch, Model *m, int i, int n, CcBuf *kvdst, size_t ko, size_t vstride) {
    Cfg *c = &m->c;
    int H = L_HEADS(c, i), KV = L_KV(c, i), hd = L_HD(c, i), kvo = KV * hd;
    size_t vs = ch->vs_off;
    CcEw gate = {CC_EW_SIGMUL, n * H * hd, H * hd, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    if (!cc_ew(ch->ctx, ch->ctx, ch->gt, NULL, NULL, &gate)) return 0;
    if (kvdst && (!cc_copy(kvdst, ko, ch->kvs, 0, (size_t)n * kvo) ||
                  !cc_copy(kvdst, ko + vstride, ch->kvs, vs, (size_t)n * kvo)))
        return 0;
    return cc_matmul(ch->t_o[i], ch->ctx, 0, ch->a, 0, n);
}
/* state layer v's new K/V rows (kv: K, then V vstride floats on) into the host's cache, as
 * ink2_attention writes them */
static void ink2cc_kv_down(Ink2CuChain *ch, Model *m, int v, int n, int pb, const float *kv, size_t vstride) {
    Cfg *c = &m->c;
    int KV = L_KV(c, v), hd = L_HD(c, v), kvo = KV * hd, cap = ch->cap[v], s0 = n > cap ? n - cap : 0;
    const float *vv = kv + vstride;
    for (int s = s0; s < n; s++) for (int h = 0; h < KV; h++) {
        size_t dst = ((size_t)h * cap + (size_t)(pb + s) % cap) * hd, src = (size_t)s * kvo + (size_t)h * hd;
        memcpy(m->K[v] + dst, kv + src, hd * sizeof(float));
        memcpy(m->V[v] + dst, vv + src, hd * sizeof(float));
    }
}
/* the MoE output of layer i, state layer v (routed: the host's sum, or the frame's) joining the stream:
 * x = res + sconv(mo) */
static int ink2cc_join(Ink2CuChain *ch, Model *m, int i, int v, int n, CcBuf *routed) {
    Cfg *c = &m->c; int D = c->hidden;
    CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_copy(ch->mo, 0, routed, 0, (size_t)n * D) && ink2cc_sconv(ch, ch->mo, 0, n, D, 3, i, v, c) &&
           cc_ew(ch->x, ch->res, ch->mo, NULL, NULL, &add);
}
/* the residual mix of the nb block states (of blk: ch->blk, or a loop pass's pblk) and `part` into y
 * with query q (at qo in prm) */
static int ink2cc_mix_of(Ink2CuChain *ch, CcBuf *blk, CcBuf *part, int nb, size_t qo, CcBuf *y, int n, int D) {
    CcInk2Mix p = {n, D, nb, 0, D, 0, ch->maxb * D, (int)qo, 0, D, 1e-6f, (float)(1.0 / sqrt((double)D))};
    return cc_ink2_mix(part, blk, ch->prm, y, &p);
}
static int ink2cc_mix(Ink2CuChain *ch, CcBuf *part, int nb, size_t qo, CcBuf *y, int n, int D) {
    return ink2cc_mix_of(ch, ch->blk, part, nb, qo, y, n, D);
}
/* a loop pass's gate g (0 q, 1 k, 2 v, 3 mlp) of layer i: y = (1 - tanh(F z)) h + tanh(G z) msg,
 * z = W [unorm(h); unorm(msg)] (ink2_loop_gate); y may be h */
static int ink2cc_loop_gate(Ink2CuChain *ch, Model *m, int i, int g, CcBuf *h, CcBuf *y, int n) {
    Cfg *c = &m->c; int D = c->hidden;
    ColiCudaTensor **t = &ch->t_lg[((size_t)i * 4 + g) * 3];
    CcNorm nh = {n, D, 1, 0, D, D, 0, 2 * D, D, 0, 0, CC_NORM_NOW, c->eps, 1.f};
    CcNorm nm = {n, D, 1, 0, D, D, D, 2 * D, D, 0, 0, CC_NORM_NOW, c->eps, 1.f};
    CcEw mix = {CC_EW_LOOPGATE, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_norm(h, ch->prm, ch->hn2, &nh) && cc_norm(ch->msg[g], ch->prm, ch->hn2, &nm) &&
           cc_matmul(t[0], ch->hn2, 0, ch->lz, 0, n) && cc_matmul(t[1], ch->lz, 0, ch->lf, 0, n) &&
           cc_matmul(t[2], ch->lz, 0, ch->lgv, 0, n) && cc_ew(y, ch->lf, ch->lgv, h, ch->msg[g], &mix);
}

/* ---- the routed experts in the frame (cc_route, cc_moe; the CUDA tier's table) ---- */
static int ink2cc_dm_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN_MOE"); on = !(e && *e == '0'); }
    return on;
}
/* layer i's formats as cc_moe takes them, from expert 0's entry words (cu_xb_expert_entry); every
 * expert of the layer the same, on the device (not the host pool); 0 = the host's step */
static int ink2cc_dm_layer(Ink2CuChain *ch, Model *m, int i, const uint32_t *ent) {
    Cfg *c = &m->c; const int E = c->n_experts;
    const uint32_t fg = ent[14], gs = ent[15], fd = ent[16], gd = ent[17];
    for (int e = 0; e < E; e++) {
        const uint32_t *w = ent + (size_t)e * INK2CC_ENT;
        CcExpert x; memcpy(&x, w, sizeof x);
        if (w[18] != 1u || w[14] != fg || w[15] != gs || w[16] != fd || w[17] != gd || !x.g || x.host) return 0;
    }
    int ok;
    if (fg >= 100) ok = fd >= 100 && (cc_moe_block_type((int)fg - 100) || fg == 140) && (cc_moe_block_type((int)fd - 100) || fd == 140);
    else ok = (fg == 1 || fg == 2 || (fg == 4 && gs >= 16 && gs % 16 == 0)) && fd == fg && gd == gs;
    if (!ok || c->hidden % 32 || c->moe_inter % 32 || c->hidden > CC_MOE_DIM_MAX || c->moe_inter > CC_MOE_DIM_MAX) return 0;
    if (m->L[i].xgsc)   /* a GGUF's NVFP4 layer: gate's and up's scale one number an expert (the tier's start checks) */
        for (int e = 0; e < E; e++) if (m->L[i].xgsc[e] != m->L[i].xusc[e]) return 0;
    ch->dm_fmt[i] = (int)fg; ch->dm_gs[i] = fg == 4 ? (int)gs : 0; ch->dm_fmtd[i] = fg >= 100 ? (int)fd : 0;
    return 1;
}
/* the tier's table read again when its residents moved since (and uploaded: between frames, the
 * previous frame waited for), 0 = no experts in the frame this step */
static int ink2cc_dm_table(Ink2CuChain *ch, Model *m) {
    Cfg *c = &m->c; const int L = c->n_layers, E = c->n_experts;
    if (ch->dm_read && ch->dm_epoch == cut_epoch()) return 1;
    int any = 0;
    for (int i = 0; i < L; i++) {
        int nres = cut_dev_table(i, ch->dm_ent);
        if (nres < 0) { ch->dm_state = -1; return 0; }
        ch->dm_on[i] = nres == E && ink2cc_dm_layer(ch, m, i, ch->dm_ent);
        any |= ch->dm_on[i];
        for (int e = 0; e < E; e++) {
            memcpy(&ch->dm_tabh[(size_t)i * E + e], ch->dm_ent + (size_t)e * INK2CC_ENT, sizeof(CcExpert));
            ch->dm_esch[((size_t)i * E + e) * 2] = m->L[i].xgsc ? m->L[i].xgsc[e] : 1.f;
            ch->dm_esch[((size_t)i * E + e) * 2 + 1] = m->L[i].xdsc ? m->L[i].xdsc[e] : 1.f;
        }
    }
    const size_t nt = (size_t)L * E;
    if (!cc_begin() || !cc_write(ch->dm_tab, 0, ch->dm_tabh, nt * sizeof(CcExpert)) ||
        !cc_write(ch->dm_tab, ch->dm_esc_off / 4, ch->dm_esch, nt * 2 * sizeof(float)) || !cc_submit(1)) {
        ch->dm_state = -1; return 0;
    }
    ch->dm_epoch = cut_epoch(); ch->dm_read = 1; ch->dm_reads++;
    if (ch->dm_reads == 1) {
        int nl = 0; for (int i = 0; i < L; i++) nl += ch->dm_on[i];
        fprintf(stderr, "[CUDA] inkling2 chain: the routed experts of %d of %d layers in the frame (the tier holds them all on the device)\n",
                nl, L);
    }
    return any;
}
/* Reserve prefill in-frame MoE buffers for `rows` (> CC_MOE_FRAME_ROWS) without touching decode buffers. */
static int ink2cc_dmoe_pf_reserve(Ink2CuChain *ch, Model *m, int rows) {
    if (rows <= ch->pf_rows && ch->pf_ids && ch->pf_w && ch->pf_hid && ch->pf_y) return 1;
    Cfg *c = &m->c; const int D = c->hidden, I = c->moe_inter, E = c->n_experts, K = c->topk;
    const size_t hid_bytes = cc_moe_hid_floats(rows, K, D, I, E) * sizeof(float);
    if (!cc_reserve(&ch->pf_ids, (size_t)ch->NS * (size_t)rows * K * sizeof(int), CC_DOWN) ||
        !cc_reserve(&ch->pf_w, (size_t)rows * K * sizeof(float), CC_DEV) ||
        !cc_reserve(&ch->pf_hid, hid_bytes, CC_DEV) ||
        !cc_reserve(&ch->pf_y, (size_t)rows * D * sizeof(float), CC_DEV))
        return 0;
    ch->pf_rows = rows;
    return 1;
}
/* The step's buffers and table, 1 when a layer of it may run its experts here: both decode rows
 * and full-GPU prefill chunks (when all layers are VRAM-resident fmt=4), not with TOPP, ROUTE_TRACE,
 * shared experts or a backend without CC_INK2_MOE */
static int ink2cc_dmoe(Ink2CuChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; const int L = c->n_layers, E = c->n_experts, K = c->topk;
    if (ch->dm_state < 0 || (g_topp > 0.f && g_topp < 1.f) || rt_fp || c->n_shared > 0) return 0;
    if (ch->dm_state == 0) {
        ch->dm_state = -1;
        if (!ink2cc_dm_wanted() || !(cc_ink2_caps() & CC_INK2_MOE) || !cc_moe_ready() || !cut_ready() || cut_devices() > 1 ||
            E < 1 || E > CC_ROUTE_EMAX || K < 1 || K > CC_ROUTE_KMAX) return 0;
        const size_t nt = (size_t)L * E, R8 = CC_MOE_FRAME_ROWS;
        ch->dm_esc_off = (nt * sizeof(CcExpert) + 255) & ~(size_t)255;
        ch->dm_on = calloc((size_t)L, 1); ch->dm_fmt = calloc((size_t)L, sizeof(int));
        ch->dm_gs = calloc((size_t)L, sizeof(int)); ch->dm_fmtd = calloc((size_t)L, sizeof(int));
        ch->dm_ent = malloc((size_t)E * INK2CC_ENT * sizeof(uint32_t));
        ch->dm_tabh = calloc(nt, sizeof(CcExpert)); ch->dm_esch = malloc(nt * 2 * sizeof(float));
        if (!ch->dm_on || !ch->dm_fmt || !ch->dm_gs || !ch->dm_fmtd || !ch->dm_ent || !ch->dm_tabh || !ch->dm_esch) return 0;
        if (!(ch->dm_tab = cc_buf(ch->dm_esc_off + nt * 2 * sizeof(float), CC_DEV)) ||
            !(ch->dm_ids = cc_buf((size_t)ch->NS * R8 * K * sizeof(int), CC_DOWN)) ||
            !(ch->dm_w = cc_buf((size_t)ch->NS * R8 * K * sizeof(float), CC_DEV)) ||
            !(ch->dm_hid = cc_buf(cc_moe_hid_floats((int)R8, K, c->hidden, c->moe_inter, E) * sizeof(float), CC_DEV)) ||
            !(ch->dm_y = cc_buf(R8 * (size_t)c->hidden * sizeof(float), CC_DEV)) ||
            !(ch->kvl = cc_buf((size_t)ch->NS * 2 * R8 * ch->kvo_max * sizeof(float), CC_DOWN))) {
            fprintf(stderr, "[CUDA] inkling2 chain: no device memory for the routed experts' buffers; their host's step\n");
            return 0;
        }
        ch->dm_state = 1;
    }
    if (!ink2cc_dm_table(ch, m)) return 0;
    if (rows > CC_MOE_FRAME_ROWS) {
        for (int i = 0; i < L; i++) if (!ch->dm_on[i] || ch->dm_fmt[i] != 4) return 0;
        return ink2cc_dmoe_pf_reserve(ch, m, rows);
    }
    return 1;
}
/* after the frame: a step whose experts ran there (layer i, state layer v), its K/V rows into the
 * host's cache (unless lazy-synced) and its books, as moe_ex and ink_vk_moe keep them */
static void ink2cc_dmoe_done(Ink2CuChain *ch, Model *m, int i, int v, int n, int pb, int skip_kv) {
    const int K = m->c.topk;
    if (!skip_kv) {
        const size_t reg = (size_t)2 * CC_MOE_FRAME_ROWS * ch->kvo_max;
        ink2cc_kv_down(ch, m, v, n, pb, (const float *)cc_ptr(ch->kvl) + (size_t)v * reg, reg / 2);
    }
    const int nk = n * K;
    int id_loc[CC_MOE_FRAME_ROWS * CC_ROUTE_KMAX];
    const int *ids_src = (n > CC_MOE_FRAME_ROWS)
                             ? ((const int *)cc_ptr(ch->pf_ids) + (size_t)v * ch->pf_rows * K)
                             : ((const int *)cc_ptr(ch->dm_ids) + (size_t)v * CC_MOE_FRAME_ROWS * K);
    const int *ids = (nk <= CC_MOE_FRAME_ROWS * CC_ROUTE_KMAX)
                         ? (memcpy(id_loc, ids_src, (size_t)nk * sizeof(int)), id_loc)
                         : ids_src;
    m->ereq += (uint64_t)nk; m->euse += (uint64_t)nk;
    uint32_t *eu = m->eusage ? m->eusage[i] : NULL;
    for (int j = 0; j < nk; j++) {
        int eid = ids[j];
        if (eid < 0) continue;
        if (eu) eu[eid]++;
        ehit_mark(m, i, eid);
    }
    cut_dev_count(i, n, K, ids);
    ch->dm_layers++;
}

/* ---- CUDA graphs (INK2_CC_GRAPH, default on): per-step segments 0 (pre-attn) and 1 (post-attn+moe) ---- */
static unsigned long long ink2g_mix(unsigned long long h, const void *p) {
    uintptr_t v = (uintptr_t)p;
    for (size_t k = 0; k < sizeof v; k++) { h ^= (unsigned char)(v >> (8 * k)); h *= 0x100000001b3ULL; }
    return h;
}
static unsigned long long ink2g_buf(unsigned long long h, const CcBuf *b) { return ink2g_mix(h, b ? cc_dev_addr(b) : NULL); }
static unsigned long long ink2g_fp(Ink2CuChain *ch) {
    CcBuf *all[] = {ch->prm, ch->rings, ch->x, ch->res, ch->blk, ch->hq, ch->hk, ch->hv, ch->nq, ch->nk, ch->nv,
                    ch->q, ch->kvs, ch->r, ch->gt, ch->ctx, ch->a, ch->mo, ch->dyn, ch->ts, ch->lg, ch->fin, ch->outd, ch->out_dev, ch->pick_out, ch->pb_dev, ch->rel_part,
                    ch->h2d, ch->lgd, ch->kvd, ch->routed, ch->xpre, ch->xprev, ch->pblk,
                    ch->msg[0], ch->msg[1], ch->msg[2], ch->msg[3], ch->hn2, ch->lz, ch->lf, ch->lgv,
                    ch->dm_tab, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y, ch->kvl};
    unsigned long long h = 0xcbf29ce484222325ULL;
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->rows);
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->vs_off);
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->dm_epoch);
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->dm_state);
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->max_t);
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) h = ink2g_buf(h, all[k]);
    for (int v = 0; v < ch->NS; v++) { h = ink2g_buf(h, ch->kc[v]); h = ink2g_buf(h, ch->vc[v]); }
    return h;
}
static int ink2cc_seg0_ops(Ink2CuChain *ch, Model *m, int i, int v, int pass, int n,
                           int pending, int pi, int pv, int prev_dev, int B, int preB, int nprev, int lp, int ls, int st) {
    (void)st;
    Cfg *c = &m->c; int D = c->hidden, local = c->local[i];
    Ink2Layer *x2 = m->L[i].x2;
    const int msg = lp && pass > 0;
    int ok = 1;
    if (pending) ok = ink2cc_join(ch, m, pi, pv, n, prev_dev ? (n > CC_MOE_FRAME_ROWS ? ch->pf_y : ch->dm_y) : ch->routed);
    if (ok && lp && i == ls) {
        if (pass == 0) {
            ok = cc_copy(ch->xpre, 0, ch->x, 0, (size_t)n * D);
        } else {
            ok = cc_copy(ch->xprev, 0, ch->x, 0, (size_t)n * D) && cc_copy(ch->x, 0, ch->xpre, 0, (size_t)n * D);
            if (ok && nprev > 0) {
                CcRegion *rg = malloc(sizeof *rg * (size_t)n);
                if (!rg) return 0;
                for (int s = 0; s < n; s++)
                    rg[s] = (CcRegion){(size_t)s * ch->maxb * D, ((size_t)s * ch->maxb + preB + 1) * D, (size_t)nprev * D};
                ok = cc_copy_regions(ch->pblk, ch->blk, rg, n);
                free(rg);
            }
        }
    }
    if (ok && msg)
        for (int g = 0; g < 4 && ok; g++)
            ok = ink2cc_mix_of(ch, ch->pblk, ch->xprev, nprev, ch->o_lq + ((size_t)(i - ls) * 4 + g) * D, ch->msg[g], n, D);
    CcBuf *iq = ch->x, *ik = ch->x, *iv = ch->x;
    if (ok && x2->aq) {
        ok = ink2cc_mix(ch, ch->x, B, ch->o_aq[i], ch->hq, n, D);
        if (ok && !local) ok = ink2cc_mix(ch, ch->x, B, ch->o_aq[i] + D, ch->hk, n, D) &&
                               ink2cc_mix(ch, ch->x, B, ch->o_aq[i] + 2 * (size_t)D, ch->hv, n, D);
        iq = ch->hq; ik = local ? ch->hq : ch->hk; iv = local ? ch->hq : ch->hv;
    }
    if (ok && msg) {
        ok = ink2cc_loop_gate(ch, m, i, 1, ik, ch->hk, n) && ink2cc_loop_gate(ch, m, i, 2, iv, ch->hv, n) &&
             ink2cc_loop_gate(ch, m, i, 0, iq, ch->hq, n);
        iq = ch->hq; ik = ch->hk; iv = ch->hv;
    }
    const int sep = !local || msg;
    ok = ok && ink2cc_norm(iq, ch->prm, ch->o_n1[i], ch->nq, n, D, c->eps);
    if (sep) ok = ok && ink2cc_norm(ik, ch->prm, local ? ch->o_n1[i] : ch->o_n2[i], ch->nk, n, D, c->eps) &&
                      ink2cc_norm(iv, ch->prm, local ? ch->o_n1[i] : ch->o_n3[i], ch->nv, n, D, c->eps);
    return ok && ink2cc_attn_pre(ch, m, i, v, sep, n);
}
static int ink2cc_seg1_ops(Ink2CuChain *ch, Model *m, int i, int v, int pass, int n, int dl, int B, int lp, int lazy_kv) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, blkn = c->res_block > 0 ? c->res_block : 1;
    Layer *l = &m->L[i];
    const int msg = lp && pass > 0;
    const size_t reg = (size_t)2 * CC_MOE_FRAME_ROWS * ch->kvo_max;
    int ok = (dl ? ink2cc_attn_post(ch, m, i, n, lazy_kv ? NULL : ch->kvl, (size_t)v * reg, reg / 2)
                 : ink2cc_attn_post(ch, m, i, n, ch->kvd, 0, (size_t)ch->rows * ch->kvo_max)) &&
             ink2cc_sconv(ch, ch->a, 0, n, D, 2, i, v, c);
    int b_after = B;
    if (ok && i % blkn == 0) {
        CcRegion *rg = malloc(sizeof *rg * (size_t)n);
        if (!rg) return 0;
        for (int s = 0; s < n; s++) rg[s] = (CcRegion){((size_t)s * ch->maxb + B) * D, (size_t)s * D, (size_t)D};
        ok = cc_copy_regions(ch->blk, ch->x, rg, n) && cc_copy(ch->res, 0, ch->a, 0, (size_t)n * D);
        free(rg);
        b_after = B + 1;
    } else if (ok) {
        CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = cc_ew(ch->res, ch->x, ch->a, NULL, NULL, &add);
    }
    ok = ok && ink2cc_mix(ch, ch->res, b_after, ch->o_mq[i], ch->hq, n, D) &&
         (!msg || ink2cc_loop_gate(ch, m, i, 3, ch->hq, ch->hq, n)) &&
         ink2cc_norm(ch->hq, ch->prm, ch->o_post[i], ch->nq, n, D, c->eps) &&
         cc_matmul(ch->t_router[i], ch->nq, 0, ch->lg, 0, n);
    if (dl) {
        const int pf = n > CC_MOE_FRAME_ROWS;
        const int ko = pf ? v * ch->pf_rows * K : v * CC_MOE_FRAME_ROWS * K;
        const int wo = pf ? 0 : ko;
        CcBuf *b_ids = pf ? ch->pf_ids : ch->dm_ids;
        CcBuf *b_w   = pf ? ch->pf_w   : ch->dm_w;
        CcBuf *b_hid = pf ? ch->pf_hid : ch->dm_hid;
        CcBuf *b_y   = pf ? ch->pf_y   : ch->dm_y;
        CcRoute rp = {n, E, K, 1, 0, E, (int)ch->o_rb[i], 1, ko, K, wo, K, 1, 1, 1, 1.f, c->route_scale * l->rgs, 0.f};
        CcMoe mp = {0};
        mp.S = n; mp.K = K; mp.D = D; mp.I = c->moe_inter; mp.fmt = ch->dm_fmt[i]; mp.gs = ch->dm_gs[i]; mp.act = 1; mp.parts = 3;
        mp.t_off = i * E; mp.x_row = D; mp.i_off = ko; mp.i_row = K; mp.w_off = wo; mp.w_row = K; mp.y_row = D; mp.h_off = -1;
        mp.E = E; mp.fmt_d = ch->dm_fmtd[i];
        mp.esc_off = m->L[i].xgsc ? (int)ch->dm_esc_off : 0; mp.act_a = c->gate_cap; mp.act_b = c->up_cap;
        return ok && cc_route(ch->lg, ch->prm, b_ids, b_w, &rp) &&
               cc_moe(ch->dm_tab, ch->nq, b_ids, b_w, b_hid, b_y, NULL, &mp);
    }
    return ok && cc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && cc_copy(ch->h2d, 0, ch->nq, 0, (size_t)n * D);
}
static int ink2cc_seg(Ink2CuChain *ch, Model *m, int st, int seg, int gr_on, unsigned long long fp,
                      int i, int v, int pass, int n, int pending, int pi, int pv, int prev_dev,
                      int dl, int B, int preB, int nprev, int lp, int ls) {
    if (!gr_on || !ch->g_on || st >= ch->g_steps)
        return seg == 0 ? ink2cc_seg0_ops(ch, m, i, v, pass, n, pending, pi, pv, prev_dev, B, preB, nprev, lp, ls, st)
                        : ink2cc_seg1_ops(ch, m, i, v, pass, n, dl, B, lp, 0);
    unsigned long long key = 1ULL + (unsigned long long)n + ((unsigned long long)pending << 8) +
                             ((unsigned long long)prev_dev << 9) + ((unsigned long long)dl << 10) +
                             ((unsigned long long)B << 12);
    fp = ink2g_mix(fp, cc_graph_scratch());
    Ink2CcGraph *w = ch->gr + ((size_t)st * INK2G_SEGS + seg) * INK2G_WAYS, *e = NULL, *old = w;
    for (int k = 0; k < INK2G_WAYS && !e; k++) {
        if (w[k].key == key && w[k].fp == fp) e = &w[k];
        else if (w[k].tick < old->tick) old = &w[k];
    }
    ch->g_tick++;
    if (!e) {
        cc_graph_free(old->exec);
        *old = (Ink2CcGraph){key, fp, ch->g_tick, NULL};
        ch->g_eager++;
        return seg == 0 ? ink2cc_seg0_ops(ch, m, i, v, pass, n, pending, pi, pv, prev_dev, B, preB, nprev, lp, ls, st)
                        : ink2cc_seg1_ops(ch, m, i, v, pass, n, dl, B, lp, 0);
    }
    e->tick = ch->g_tick;
    if (e->exec) { ch->g_launch++; return cc_graph_launch(e->exec); }
    void *x = NULL; int ok = 0, broke = 1;
    if (cc_graph_begin()) {
        ok = seg == 0 ? ink2cc_seg0_ops(ch, m, i, v, pass, n, pending, pi, pv, prev_dev, B, preB, nprev, lp, ls, st)
                      : ink2cc_seg1_ops(ch, m, i, v, pass, n, dl, B, lp, 0);
        x = cc_graph_end(&broke);
        if (x && ok) { e->exec = x; ch->g_capture++; return cc_graph_launch(x); }
        cc_graph_free(x);
        if (!x && !broke) return 0;
    } else if (cc_lost()) return 0;
    fprintf(stderr, "[CUDA] inkling2 chain: INK2_CC_GRAPH: step %d seg %d did not capture; running ops one by one\n", st, seg);
    ch->g_on = 0;
    ink2cc_free_graphs(ch);
    return seg == 0 ? ink2cc_seg0_ops(ch, m, i, v, pass, n, pending, pi, pv, prev_dev, B, preB, nprev, lp, ls, st)
                    : ink2cc_seg1_ops(ch, m, i, v, pass, n, dl, B, lp, 0);
}

/* Whole-forward ops (all steps + final join + output mix + final norm + lm_head [+ GPU cc_pick]) for single-graph capture. */
static int ink2cc_full_ops(Ink2CuChain *ch, Model *m, int n, int last, int nc_glob, int greedy) {
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, blkn = c->res_block > 0 ? c->res_block : 1;
    const int lp = c->loop_p > 1, ls = lp ? c->loop_s : L, le = lp ? c->loop_e : L;
    const int steps = L + (lp ? (c->loop_p - 1) * (le - ls) : 0);
    int ok = 1, pending = 0, B = 0, pi = 0, pv = 0, preB = 0, nprev = 0;
    for (int st = 0; st < steps && ok; st++) {
        int i, v, pass = 0;
        if (st < le) i = v = st;
        else if (st < le + (c->loop_p - 1) * (le - ls)) {
            pass = 1 + (st - le) / (le - ls); i = ls + (st - le) % (le - ls); v = ink_sl(c, pass, i);
        } else i = v = st - (lp ? (c->loop_p - 1) * (le - ls) : 0);
        if (lp && i == ls) {
            if (pass == 0) preB = B;
            else { nprev = B - preB - 1; B = preB; }
        }
        ok = ink2cc_seg0_ops(ch, m, i, v, pass, n, pending, pi, pv, 1, B, preB, nprev, lp, ls, st) &&
             ink2cc_attn_dyn_pb(ch, m, i, v, n, nc_glob) &&
             ink2cc_seg1_ops(ch, m, i, v, pass, n, 1, B, lp, 1);
        if (ok && i % blkn == 0) B++;
        pending = 1; pi = i; pv = v;
    }
    if (ok && pending) ok = ink2cc_join(ch, m, pi, pv, n, ch->dm_y);
    ok = ok && ink2cc_mix(ch, ch->x, B, ch->o_outq, ch->hq, n, D);
    if (ok && last) {
        CcSconv dv = {2, 0, 0, 0, 0, 0, 0, 0, D, c->mup};
        CcNorm fp = {1, D, 1, (n - 1) * D, D, D, 0, D, D, (int)ch->o_final, 0, 0, c->eps, 1.f};
        ok = cc_norm(ch->hq, ch->prm, ch->fin, &fp) && cc_sconv(ch->fin, NULL, NULL, &dv);
        if (ok) {
            if (greedy)
                ok = cc_matmul(ch->t_lm, ch->fin, 0, ch->out_dev, 0, 1) &&
                     cc_pick(ch->out_dev, ch->pick_out, c->unpad_vocab);
            else
                ok = cc_matmul(ch->t_lm, ch->fin, 0, ch->outd, 0, 1);
        }
    }
    return ok;
}
static int ink2cc_full_run(Ink2CuChain *ch, Model *m, int n, int last, int nc_glob, int greedy, unsigned long long fp) {
    const unsigned long long key = 1ULL + (unsigned long long)n + ((unsigned long long)(last ? 1 : 0) << 8) +
                                   ((unsigned long long)nc_glob << 10) + ((unsigned long long)(greedy ? 1 : 0) << 17);
    fp = ink2g_mix(fp, cc_graph_scratch());
    Ink2CcGraph *w = ch->gr_full, *e = NULL, *old = w;
    for (int k = 0; k < INK2G_FULL_WAYS && !e; k++) {
        if (w[k].key == key && w[k].fp == fp) e = &w[k];
        else if (w[k].tick < old->tick) old = &w[k];
    }
    ch->g_tick++;
    if (!e) {
        cc_graph_free(old->exec);
        *old = (Ink2CcGraph){key, fp, ch->g_tick, NULL};
        ch->g_eager++;
        return ink2cc_full_ops(ch, m, n, last, nc_glob, greedy);
    }
    e->tick = ch->g_tick;
    if (e->exec) { ch->g_full_launch++; return cc_graph_launch(e->exec); }
    void *x = NULL; int ok = 0, broke = 1;
    if (cc_graph_begin()) {
        ok = ink2cc_full_ops(ch, m, n, last, nc_glob, greedy);
        x = cc_graph_end(&broke);
        if (x && ok) { e->exec = x; ch->g_full_capture++; return cc_graph_launch(x); }
        cc_graph_free(x);
        if (!x && !broke) return 0;
    } else if (cc_lost()) return 0;
    fprintf(stderr, "[CUDA] inkling2 chain: INK2_CC_GRAPH: whole-forward graph did not capture; falling back to per-step\n");
    ch->g_on = 0;
    ink2cc_free_graphs(ch);
    return ink2cc_full_ops(ch, m, n, last, nc_glob, greedy);
}

/* Every layer for S rows from host rows xh, the last row's logits (softcapped) into `logit`
 * (or greedy argmax directly into `*greedy_tok` on GPU via cc_pick when greedy_tok != NULL). */
static int ink2cc_forward_ex(Model *m, float *xh, int S, int pos_base, int want_x, float *logit, int *greedy_tok) {
    if (!g_cu_chain) return 0;
    if (g_cu_chain == 2 && S <= 2) { ink2cc_sync_cs(m); return 0; }   /* prompts only: decode on the CPU */
    if (m->mux_rows) { ink2cc_sync_cs(m); return 0; }
    Ink2CuChain *ch = ink2cc_setup(m);
    if (!ch || ch->failed) { ink2cc_sync_cs(m); return 0; }
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, blkn = c->res_block > 0 ? c->res_block : 1;
    int mirror_ok = ink2cc_mirror(ch, m);
    int CH = mirror_ok ? ink2cc_chunk_rows(ch, m) : 1;
    int nchunks = CH > 0 ? (S + CH - 1) / CH : 1;
    int rows = nchunks > 1 ? (S + nchunks - 1) / nchunks : S;
    int scr_ok = 0;
    if (mirror_ok && pos_base + S <= m->max_t) {
        for (;;) {
            if (ink2cc_scratch(ch, m, rows) && (!want_x || cc_reserve(&ch->xd, (size_t)rows * D * sizeof(float), CC_DOWN))) {
                scr_ok = 1;
                break;
            }
            if (rows <= 256 || rows <= ch->rows) break;
            rows = (rows + 1) / 2;
            if (rows < 256) rows = 256;
        }
    }
    if (!scr_ok) {
        ink2cc_sync_cs(m);
        fprintf(stderr, "[CUDA] inkling2 chain: device memory for %d rows refused; per-matrix path\n", rows);
        ink2cc_free_own(m, ch);
        ch->failed = 1;
        return 0;
    }
    const int want_greedy = greedy_tok != NULL && !want_x && cc_pick_ready() && !getenv("COLI_LOGIT_DUMP");
    if (want_greedy) {
        if (!ch->out_dev) ch->out_dev = cc_buf((size_t)c->unpad_vocab * sizeof(float), CC_DEV);
        if (!ch->pick_out) ch->pick_out = cc_buf(2 * sizeof(int), CC_DOWN);
    }
    const int can_greedy = want_greedy && ch->out_dev != NULL && ch->pick_out != NULL;
    if (greedy_tok && !can_greedy && !logit) { ink2cc_sync_cs(m); return 0; }
    float *xfin = want_x ? malloc((size_t)S * D * sizeof(float)) : NULL;
    if (want_x && !xfin) { ink2cc_sync_cs(m); return 0; }
    int *dq = malloc((size_t)2 * c->n_state * sizeof(int)), ndq = 0;   /* this frame's in-frame steps (i, v) */
    if (!dq) { free(xfin); ink2cc_sync_cs(m); return 0; }
    const int q8 = ch->q8_kv;
    const int cur_slot = q8 ? ink2cc_cur() : 0;
    int nc_q8_cap = 1;
    if (q8 && rows <= 8) {
        int hd_max = 0;
        for (int li = 0; li < L; li++) if (L_HD(c, li) > hd_max) hd_max = L_HD(c, li);
        int nmax = pos_base + S;
        int ncr = (nmax + 127) / 128;
        int nc_lim = rows <= 1 ? 128 : rows <= 2 ? 64 : rows <= 4 ? 32 : 16;
        if (ncr > nc_lim) ncr = nc_lim;
        if (ncr >= 2) {
            size_t pbytes = (size_t)rows * (size_t)ch->hmax * (size_t)ncr * (size_t)(hd_max + 2) * sizeof(float);
            if (cc_reserve(&ch->rel_part, pbytes, CC_DEV)) nc_q8_cap = ncr;
        }
    }
    for (int c0 = 0; c0 < S; c0 += rows) {
        if (coli_prompt_break()) { free(xfin); free(dq); return 0; }
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0, R = ch->rows;
        ch->fslot = cur_slot;
        int dmoe = ink2cc_dmoe(ch, m, n);   /* the table read (and uploaded) before the frame begins */
        if (!cc_begin() || !cc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            (q8 ? !ink2cc_push_state_slots(ch, m, &cur_slot, 1) : !ink2cc_push_state(ch, m))) goto lost;
        if (q8) {
            char *base = (char *)cc_dev_addr(ch->slot_kv[cur_slot]);
            for (int v = 0; v < ch->NS; v++) {
                int x = cur_slot * ch->NS + v;
                ch->kv_tabh[v] = (CcInk2KvRow){
                    base + ch->sq_kq[x], (float *)(base + ch->sq_ks[x]),
                    base + ch->sq_vq[x], (float *)(base + ch->sq_vs[x]),
                    ch->scap[x], pb
                };
            }
            if (!cc_write(ch->kv_tab, 0, ch->kv_tabh, (size_t)ch->NS * sizeof(CcInk2KvRow))) goto lost;
        }
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
            int wok = cc_write(ch->ts, 0, tp, (size_t)n * sizeof(float)) && cc_write(ch->ts, (size_t)R, tp + n, (size_t)n * sizeof(float));
            if (tp != tau) free(tp);
            if (!wok) goto lost;
        }
        int ok = 1, pending = 0, B = 0, pi = 0, pv = 0, preB = 0, nprev = 0, prev_dev = 0;
        /* the steps in the CPU's order: the layers before the loop, its passes, the layers after */
        const int lp = c->loop_p > 1, ls = lp ? c->loop_s : L, le = lp ? c->loop_e : L;
        const int steps = L + (lp ? (c->loop_p - 1) * (le - ls) : 0);
        const int gr_on = !q8 && ch->g_on && n <= CC_MOE_FRAME_ROWS;
        const unsigned long long g_fp = gr_on ? ink2g_fp(ch) : 0;
        int last = c0 + n == S;
        int all_dl_pf = dmoe && (n > CC_MOE_FRAME_ROWS || q8);
        int all_dl = !q8 && dmoe && !want_x && gr_on && cc_ink2_pb_ready() && ch->gr_full != NULL;
        if (all_dl) {
            for (int li = 0; li < L && all_dl; li++) if (!ch->dm_on[li]) all_dl = 0;
            for (int vi = 0; vi < ch->NS && all_dl; vi++) if (n > ch->cap[vi]) all_dl = 0;
        }
        if (!all_dl && !q8 && cc_ink2_pb_ready() && ch->pb_dev) {
            ok = cc_write(ch->pb_dev, 0, &pb, sizeof(int));
        }
        if (all_dl) {
            /* Compute bucketed split-K tile count nc_glob so the graph stays valid across long runs without re-capturing. */
            int nc_need = (pb + n + 127) / 128, nc_glob = 1;
            if (nc_need > 1) { nc_glob = 4; while (nc_glob < nc_need && nc_glob < 64) nc_glob <<= 1; }
            int nc_loc = c->window > 0 ? (c->window + 127) / 128 : 1;
            if (nc_loc > 64) nc_loc = 64;
            int nc_max = nc_glob > nc_loc ? nc_glob : nc_loc;
            if (nc_max >= 2) {
                int hd_max = 0;
                for (int li = 0; li < L; li++) if (L_HD(c, li) > hd_max) hd_max = L_HD(c, li);
                size_t pbytes = (size_t)CC_MOE_FRAME_ROWS * (size_t)ch->hmax * (size_t)nc_max * (size_t)(hd_max + 2) * sizeof(float);
                if (!cc_reserve(&ch->rel_part, pbytes, CC_DEV)) nc_glob = 1;
            }
            const unsigned long long g_fp_full = ink2g_fp(ch);
            const int use_greedy = can_greedy && last && n == 1;
            ok = cc_write(ch->pb_dev, 0, &pb, sizeof(int)) && ink2cc_full_run(ch, m, n, last, nc_glob, use_greedy, g_fp_full);
            if (ok) {
                for (int st = 0; st < steps; st++) {
                    int i, v;
                    if (st < le) i = v = st;
                    else if (st < le + (c->loop_p - 1) * (le - ls)) {
                        int pass = 1 + (st - le) / (le - ls); i = ls + (st - le) % (le - ls); v = ink_sl(c, pass, i);
                    } else i = v = st - (lp ? (c->loop_p - 1) * (le - ls) : 0);
                    dq[2 * ndq] = i; dq[2 * ndq + 1] = v; ndq++;
                }
            }
        } else {
            for (int st = 0; st < steps && ok; st++) {
                int i, v, pass = 0;
                if (st < le) i = v = st;
                else if (st < le + (c->loop_p - 1) * (le - ls)) {
                    pass = 1 + (st - le) / (le - ls); i = ls + (st - le) % (le - ls); v = ink_sl(c, pass, i);
                } else i = v = st - (lp ? (c->loop_p - 1) * (le - ls) : 0);
                Layer *l = &m->L[i];
                const int dl = dmoe && ch->dm_on[i];   /* its experts here, no wait */
                if (lp && i == ls) {
                    if (pass == 0) preB = B;
                    else { nprev = B - preB - 1; B = preB; }
                }
                ok = ink2cc_seg(ch, m, st, 0, gr_on, g_fp, i, v, pass, n, pending, pi, pv, prev_dev, dl, B, preB, nprev, lp, ls) &&
                     (q8 ? ink2cc_attn_dyn_q8(ch, m, i, v, n, pb, pb + n - 1, 0, 1, nc_q8_cap)
                         : ink2cc_attn_dyn(ch, m, i, v, n, pb)) &&
                     (all_dl_pf ? ink2cc_seg1_ops(ch, m, i, v, pass, n, dl, B, lp, 1)
                                : ink2cc_seg(ch, m, st, 1, gr_on, g_fp, i, v, pass, n, pending, pi, pv, prev_dev, dl, B, preB, nprev, lp, ls));
                if (ok && i % blkn == 0) B++;
                if (dl) {
                    if (ok) { dq[2 * ndq] = i; dq[2 * ndq + 1] = v; ndq++; }
                    pending = 1; pi = i; pv = v; prev_dev = 1;
                    continue;
                }
                double t0 = now_s();
                ok = ok && cc_submit(0);
                if (ok) cut_stream_prefetch(i, n);
                ok = ok && cc_finish();
                m->t_attn += now_s() - t0;
                if (!ok) break;
                for (int k = 0; k < ndq; k++) ink2cc_dmoe_done(ch, m, dq[2 * k], dq[2 * k + 1], n, pb, q8);
                ndq = 0;
                if (!q8) ink2cc_kv_down(ch, m, v, n, pb, (const float *)cc_ptr(ch->kvd), (size_t)ch->rows * ch->kvo_max);
                double t1 = now_s();
                moe_ex(m, l, i, (float *)cc_ptr(ch->h2d), n, ch->host_routed, (const float *)cc_ptr(ch->lgd), NULL);
                memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
                ch->host_ms += (now_s() - t1) * 1e3;
                /* the host's step ran the tier: where its residents moved, the layers after this one take
                 * the host's step too for the rest of this frame (the table is read again before the next) */
                if (dmoe && ch->dm_epoch != cut_epoch()) dmoe = 0;
                ok = cc_begin();
                pending = 1; pi = i; pv = v; prev_dev = 0;
            }
            if (ok && pending) ok = ink2cc_join(ch, m, pi, pv, n, prev_dev ? (n > CC_MOE_FRAME_ROWS ? ch->pf_y : ch->dm_y) : ch->routed);
            /* the output's mix of the block states and the last stream: the rows the final norm reads */
            ok = ok && ink2cc_mix(ch, ch->x, B, ch->o_outq, ch->hq, n, D);
            if (ok && want_x) ok = cc_copy(ch->xd, 0, ch->hq, 0, (size_t)n * D);
            if (ok && last) {
                CcSconv dv = {2, 0, 0, 0, 0, 0, 0, 0, D, c->mup};
                CcNorm fp = {1, D, 1, (n - 1) * D, D, D, 0, D, D, (int)ch->o_final, 0, 0, c->eps, 1.f};
                ok = cc_norm(ch->hq, ch->prm, ch->fin, &fp) && cc_sconv(ch->fin, NULL, NULL, &dv);
                if (ok) {
                    if (can_greedy)
                        ok = cc_matmul(ch->t_lm, ch->fin, 0, ch->out_dev, 0, 1) &&
                             cc_pick(ch->out_dev, ch->pick_out, c->unpad_vocab);
                    else
                        ok = cc_matmul(ch->t_lm, ch->fin, 0, ch->outd, 0, 1);
                }
            }
        }
        double t0 = now_s();
        ok = ok && cc_submit(1);
        m->t_attn += now_s() - t0;
        if (!ok) goto lost;
        const int lazy_kv_used = all_dl || all_dl_pf;
        for (int k = 0; k < ndq; k++) ink2cc_dmoe_done(ch, m, dq[2 * k], dq[2 * k + 1], n, pb, lazy_kv_used);
        ndq = 0;
        if (lazy_kv_used && !q8) {
            if (ch->kv_dev_lo >= ch->kv_dev_hi) { ch->kv_dev_lo = pb; ch->kv_dev_hi = pb + n; }
            else { if (pb < ch->kv_dev_lo) ch->kv_dev_lo = pb; if (pb + n > ch->kv_dev_hi) ch->kv_dev_hi = pb + n; }
        }
        /* Lazy sync: keep convolution rings on device until host actually reads m->cs (ink2cc_sync_cs). */
        if (q8) { ch->scs_where[cur_slot] = INK2CC_DEV; ch->shost_zero[cur_slot] = 0; }
        else { ch->cs_where = INK2CC_DEV; ch->host_zero = 0; }
        if (want_x) memcpy(xfin + (size_t)c0 * D, cc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        if (last) {
            if (can_greedy) {
                *greedy_tok = ((const int *)cc_ptr(ch->pick_out))[0];
            } else {
                memcpy(logit, cc_ptr(ch->outd), (size_t)c->unpad_vocab * sizeof(float));
                ink2_head_post(c, logit, c->unpad_vocab);
                if (greedy_tok) {
                    int bi = 0;
                    for (int k = 1; k < c->unpad_vocab; k++) if (logit[k] > logit[bi]) bi = k;
                    *greedy_tok = bi;
                }
            }
        }
    }
    if (want_x) { memcpy(xh, xfin, (size_t)S * D * sizeof(float)); free(xfin); }
    free(dq);
    ch->forwards++;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!cc_lost()) cc_finish();
    free(xfin); free(dq);
    ch->kv_dev_lo = ch->kv_dev_hi = 0;
    ink2cc_recover(m, pos_base);
    return 0;
}

static int ink2cc_forward(Model *m, float *xh, int S, int pos_base, int want_x, float *logit) {
    return ink2cc_forward_ex(m, xh, S, pos_base, want_x, logit, NULL);
}

/* Whole-forward mux graph ops + fingerprint + capture/replay (eliminates ~17 ms/step WDDM launch overhead for S=1..64). */
static unsigned long long ink2cc_mux_fp(Ink2CuChain *ch) {
    CcBuf *all[] = {ch->prm, ch->rings, ch->x, ch->res, ch->blk, ch->hq, ch->hk, ch->hv, ch->nq, ch->nk, ch->nv,
                    ch->q, ch->kvs, ch->r, ch->gt, ch->ctx, ch->a, ch->mo, ch->dyn, ch->ts, ch->lg, ch->fin,
                    ch->outd, ch->out_dev, ch->pick_out, ch->mux_rb, ch->kv_tab, ch->rel_part,
                    ch->xpre, ch->xprev, ch->pblk, ch->msg[0], ch->msg[1], ch->msg[2], ch->msg[3],
                    ch->hn2, ch->lz, ch->lf, ch->lgv,
                    ch->dm_tab, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y,
                    ch->pf_ids, ch->pf_w, ch->pf_hid, ch->pf_y};
    unsigned long long h = 0xcbf29ce484222325ULL;
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->rows);
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->vs_off);
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->pf_rows);
    h = ink2g_mix(h, (const void *)(uintptr_t)ch->dm_epoch);
    h = ink2g_mix(h, cc_graph_scratch());
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) h = ink2g_buf(h, all[k]);
    return h;
}
static int ink2cc_mux_ops(Ink2CuChain *ch, Model *m, int S, int nc_cap, int want_greedy) {
    Cfg *c = &m->c;
    const int D = c->hidden, L = c->n_layers, V = c->unpad_vocab, blkn = c->res_block > 0 ? c->res_block : 1;
    const int lp = c->loop_p > 1, ls = lp ? c->loop_s : L, le = lp ? c->loop_e : L;
    const int steps = L + (lp ? (c->loop_p - 1) * (le - ls) : 0);
    int ok = 1, pending = 0, B = 0, pi = 0, pv = 0, preB = 0, nprev = 0;
    for (int st = 0; st < steps && ok; st++) {
        int i, v, pass = 0;
        if (st < le) i = v = st;
        else if (st < le + (c->loop_p - 1) * (le - ls)) {
            pass = 1 + (st - le) / (le - ls); i = ls + (st - le) % (le - ls); v = ink_sl(c, pass, i);
        } else i = v = st - (lp ? (c->loop_p - 1) * (le - ls) : 0);
        if (lp && i == ls) {
            if (pass == 0) preB = B;
            else { nprev = B - preB - 1; B = preB; }
        }
        ok = ink2cc_seg0_ops(ch, m, i, v, pass, S, pending, pi, pv, 1, B, preB, nprev, lp, ls, st) &&
             ink2cc_attn_dyn_q8(ch, m, i, v, S, 0, 0, 1, S, nc_cap) &&
             ink2cc_seg1_ops(ch, m, i, v, pass, S, 1, B, lp, 1);
        if (ok && i % blkn == 0) B++;
        pending = 1; pi = i; pv = v;
    }
    if (ok && pending) ok = ink2cc_join(ch, m, pi, pv, S, S > CC_MOE_FRAME_ROWS ? ch->pf_y : ch->dm_y);
    ok = ok && ink2cc_mix(ch, ch->x, B, ch->o_outq, ch->hq, S, D);
    if (ok) {
        CcSconv dv = {2, 0, 0, 0, 0, 0, 0, 0, S * D, c->mup};
        CcNorm fp = {S, D, 1, 0, D, D, 0, D, D, (int)ch->o_final, 0, 0, c->eps, 1.f};
        ok = cc_norm(ch->hq, ch->prm, ch->fin, &fp) && cc_sconv(ch->fin, NULL, NULL, &dv) &&
             cc_matmul(ch->t_lm, ch->fin, 0, ch->out_dev, 0, S);
        if (ok) {
            if (want_greedy) ok = cc_pick_rows(ch->out_dev, 0, ch->pick_out, V, S);
            else ok = cc_ink2_softcap(ch->out_dev, S * V, c->logit_cap) &&
                      cc_copy(ch->outd, 0, ch->out_dev, 0, (size_t)S * V);
        }
    }
    return ok;
}
static int ink2cc_mux_full_run(Ink2CuChain *ch, Model *m, int S, int nc_cap, int want_greedy, unsigned long long fp) {
    if (!ch->g_on || !ch->gr_full) return ink2cc_mux_ops(ch, m, S, nc_cap, want_greedy);
    const unsigned long long key = 1ULL + (unsigned long long)S + ((unsigned long long)nc_cap << 8) +
                                   ((unsigned long long)(want_greedy ? 1 : 0) << 16);
    Ink2CcGraph *w = ch->gr_full, *e = NULL, *old = w;
    for (int k = 0; k < INK2G_FULL_WAYS && !e; k++) {
        if (w[k].key == key && w[k].fp == fp) e = &w[k];
        else if (w[k].tick < old->tick) old = &w[k];
    }
    ch->g_tick++;
    if (!e) {
        cc_graph_free(old->exec);
        *old = (Ink2CcGraph){key, fp, ch->g_tick, NULL};
        ch->g_eager++;
        return ink2cc_mux_ops(ch, m, S, nc_cap, want_greedy);
    }
    e->tick = ch->g_tick;
    if (e->exec) { ch->g_full_launch++; return cc_graph_launch(e->exec); }
    void *x = NULL; int ok = 0, broke = 1;
    if (cc_graph_begin()) {
        ok = ink2cc_mux_ops(ch, m, S, nc_cap, want_greedy);
        x = cc_graph_end(&broke);
        if (x && ok) { e->exec = x; ch->g_full_capture++; return cc_graph_launch(x); }
        cc_graph_free(x);
        if (!x && !broke) return 0;
    } else if (cc_lost()) return 0;
    fprintf(stderr, "[CUDA] inkling2 chain: INK2_CC_GRAPH: mux graph (S=%d) did not capture; falling back to eager\n", S);
    ch->g_on = 0;
    ink2cc_free_graphs(ch);
    return ink2cc_mux_ops(ch, m, S, nc_cap, want_greedy);
}

/* Multiplexed batched decode step of S independent conversations (1..KV_SLOTS) in ONE GPU frame
 * (captured into a whole-forward CUDA Graph per (S, nc_cap, greedy) so WDDM launch overhead -> ~50 us). */
static int ink2cc_forward_mux(Model *m, float *xh, int S, int *greedy_toks, const float **logits_out) {
    if (!g_cu_chain || g_cu_chain == 2 || !m->mux_rows || S < 1) return 0;
    Ink2CuChain *ch = ink2cc_setup(m);
    if (!ch || ch->failed || !ch->q8_kv || S > ch->nslot) return 0;
    Cfg *c = &m->c;
    const int D = c->hidden, L = c->n_layers, V = c->unpad_vocab;
    const InkRow *mr = m->mux_rows;
    int rk[64], rp[64];
    int max_pos = 0;
    for (int s = 0; s < S; s++) {
        rk[s] = (int)(mr[s].seq - g_ink_mux_seq);
        rp[s] = mr[s].pos;
        if (rk[s] < 0 || rk[s] >= ch->nslot || rp[s] >= mr[s].seq->max_t) return 0;
        if (rp[s] > max_pos) max_pos = rp[s];
    }
    for (int s = 0; s < S; s++)
        if (!ink2cc_ensure_slot(ch, m, rk[s])) return 0;
    if (!ink2cc_scratch(ch, m, S)) return 0;
    const int dmoe = ink2cc_dmoe(ch, m, S);
    if (!dmoe) return 0;
    for (int li = 0; li < L; li++) if (!ch->dm_on[li]) return 0;
    const int want_greedy = greedy_toks != NULL && cc_pick_rows_ready() && !getenv("COLI_LOGIT_DUMP");
    if (!cc_reserve(&ch->fin, (size_t)S * D * sizeof(float), CC_DEV) ||
        !cc_reserve(&ch->out_dev, (size_t)S * V * sizeof(float), CC_DEV)) return 0;
    if (want_greedy) {
        if (!cc_reserve(&ch->pick_out, (size_t)S * 2 * sizeof(int), CC_DOWN)) return 0;
    } else {
        if (!cc_reserve(&ch->outd, (size_t)S * V * sizeof(float), CC_DOWN)) return 0;
    }
    /* Bucket split-K tile count nc_cap to powers of 2 for S<=8 so the CUDA Graph stays valid as pos grows;
     * scale cap with S (128 for S=1, 64 for S=2, 32 for S<=4, 16 for S<=8) to saturate wide GPUs (e.g. 170 SMs);
     * for S>=9, H*S >= 144 blocks already saturate all SMs so nc_cap = 1 always. */
    int nc_cap = 1;
    if (S <= 8) {
        int hd_max = 0;
        for (int li = 0; li < L; li++) if (L_HD(c, li) > hd_max) hd_max = L_HD(c, li);
        int nc_need = (max_pos + 1 + 127) / 128;
        if (nc_need > 1) {
            int nc_lim = S <= 1 ? 128 : S <= 2 ? 64 : S <= 4 ? 32 : 16;
            int ncb = 4;
            while (ncb < nc_need && ncb < nc_lim) ncb <<= 1;
            size_t pbytes = (size_t)S * (size_t)ch->hmax * (size_t)ncb * (size_t)(hd_max + 2) * sizeof(float);
            if (cc_reserve(&ch->rel_part, pbytes, CC_DEV)) nc_cap = ncb;
        }
    }
    for (int s = 0; s < S; s++) ch->mux_rbh[s] = (int)((size_t)rk[s] * ch->csd_n);
    for (int v = 0; v < ch->NS; v++) {
        for (int s = 0; s < S; s++) {
            int k = rk[s], x = k * ch->NS + v;
            char *base = (char *)cc_dev_addr(ch->slot_kv[k]);
            ch->kv_tabh[(size_t)v * S + s] = (CcInk2KvRow){
                base + ch->sq_kq[x], (float *)(base + ch->sq_ks[x]),
                base + ch->sq_vq[x], (float *)(base + ch->sq_vs[x]),
                ch->scap[x], rp[s]
            };
        }
    }
    float tau[2 * 64];
    for (int s = 0; s < S; s++) {
        float t = 1.f;
        if (c->log_floor > 0) { double en = (double)(rp[s] + 1) / c->log_floor; if (en > 1.0) t = 1.f + c->log_alpha * (float)log(en); }
        tau[s] = t;
        tau[S + s] = 1.f;
    }
    int R = ch->rows;
    ch->fslot = -1;
    const unsigned long long fp = ink2cc_mux_fp(ch);
    int ok = cc_begin() &&
             cc_write(ch->x, 0, xh, (size_t)S * D * sizeof(float)) &&
             cc_write(ch->mux_rb, 0, ch->mux_rbh, (size_t)S * sizeof(int)) &&
             cc_write(ch->kv_tab, 0, ch->kv_tabh, (size_t)ch->NS * (size_t)S * sizeof(CcInk2KvRow)) &&
             cc_write(ch->ts, 0, tau, (size_t)S * sizeof(float)) &&
             cc_write(ch->ts, (size_t)R, tau + S, (size_t)S * sizeof(float)) &&
             ink2cc_push_state_slots(ch, m, rk, S) &&
             ink2cc_mux_full_run(ch, m, S, nc_cap, want_greedy, fp);
    double t0 = now_s();
    ok = ok && cc_submit(1);
    m->t_attn += now_s() - t0;
    if (!ok) {
        if (!cc_lost()) cc_finish();
        ink2cc_fatal("the device failed during a multiplexed step");
        return 0;
    }
    const int lp = c->loop_p > 1, ls = lp ? c->loop_s : L, le = lp ? c->loop_e : L;
    const int steps = L + (lp ? (c->loop_p - 1) * (le - ls) : 0);
    for (int st = 0; st < steps; st++) {
        int i, v;
        if (st < le) i = v = st;
        else if (st < le + (c->loop_p - 1) * (le - ls)) {
            int pass = 1 + (st - le) / (le - ls); i = ls + (st - le) % (le - ls); v = ink_sl(c, pass, i);
        } else i = v = st - (lp ? (c->loop_p - 1) * (le - ls) : 0);
        ink2cc_dmoe_done(ch, m, i, v, S, 0, 1);
    }
    for (int s = 0; s < S; s++) { ch->scs_where[rk[s]] = INK2CC_DEV; ch->shost_zero[rk[s]] = 0; }
    if (want_greedy) {
        const int *pk = (const int *)cc_ptr(ch->pick_out);
        for (int s = 0; s < S; s++) greedy_toks[s] = pk[2 * s];
    } else if (logits_out) {
        *logits_out = (const float *)cc_ptr(ch->outd);
    }
    ch->forwards++; ch->mux_forwards++; ch->mux_rows += (unsigned long long)S;
    return 1;
}

/* Packed multi-prompt prefill: runs M independent prompt chunks (N total tokens) in ONE GPU frame
 * at peak Blackwell Tensor-Core utilization (~9K tok/s instead of 63 separate small forwards).
 * Sets *logits_out to cc_ptr(ch->outd) holding M_out rows (for segments with want_logit[j] != 0),
 * already softcapped on GPU. Returns 1 on success, 0 to fall back to single-prompt prefill. */
static int ink2cc_forward_prefill_packed(Model *m, const int *slots, const int *pos_bases, const int *lens,
                                         const int *want_logit, int M, float *xh, int N, const float **logits_out) {
    if (!g_cu_chain || !cc_ink2_pack_ready() || M < 2 || M > 64 || N <= CC_MOE_FRAME_ROWS) return 0;
    Ink2CuChain *ch = ink2cc_setup(m);
    if (!ch || ch->failed || !ch->q8_kv) return 0;
    Cfg *c = &m->c;
    const int D = c->hidden, L = c->n_layers, V = c->unpad_vocab, blkn = c->res_block > 0 ? c->res_block : 1;
    if (c->d_rel > 16 || (c->d_rel & 1) != 0) return 0;
    for (int li = 0; li < L; li++) if (L_HD(c, li) != 128) return 0;
    for (int j = 0; j < M; j++) {
        if (slots[j] < 0 || slots[j] >= ch->nslot || lens[j] < 1) return 0;
        Ink2ccHost hs = ink2cc_host(m, slots[j]);
        if (pos_bases[j] + lens[j] > hs.max_t) return 0;
        if (!ink2cc_ensure_slot(ch, m, slots[j])) return 0;
    }
    if (!ink2cc_scratch(ch, m, N)) return 0;
    const int dmoe = ink2cc_dmoe(ch, m, N);
    if (!dmoe) return 0;
    for (int li = 0; li < L; li++) if (!ch->dm_on[li]) return 0;
    CcInk2Seg segs[64];
    int ntiles = 0, m_out = 0, off = 0;
    for (int j = 0; j < M; j++) {
        segs[j] = (CcInk2Seg){off, lens[j], (int)((size_t)slots[j] * ch->csd_n), pos_bases[j]};
        ntiles += (lens[j] + 63) / 64;
        if (want_logit[j]) m_out++;
        off += lens[j];
    }
    if (off != N || ntiles < 1) return 0;
    if (!cc_reserve(&ch->pack_segs, (size_t)M * sizeof(CcInk2Seg), CC_DEV) ||
        !cc_reserve(&ch->pack_tiles, (size_t)ntiles * 2 * sizeof(int), CC_DEV) ||
        !cc_reserve(&ch->pack_rseg, (size_t)N * 2 * sizeof(int), CC_DEV)) return 0;
    if (m_out > 0 && (!cc_reserve(&ch->fin, (size_t)m_out * D * sizeof(float), CC_DEV) ||
                      !cc_reserve(&ch->out_dev, (size_t)m_out * V * sizeof(float), CC_DEV) ||
                      !cc_reserve(&ch->outd, (size_t)m_out * V * sizeof(float), CC_DOWN))) return 0;
    int *tile_h = malloc((size_t)ntiles * 2 * sizeof(int));
    int *rseg_h = malloc((size_t)N * 2 * sizeof(int));
    float *tau_h = malloc((size_t)2 * N * sizeof(float));
    if (!tile_h || !rseg_h || !tau_h) { free(tile_h); free(rseg_h); free(tau_h); return 0; }
    int ti = 0;
    for (int j = 0; j < M; j++) {
        for (int qs = 0; qs < segs[j].len; qs += 64) { tile_h[2 * ti] = j; tile_h[2 * ti + 1] = qs; ti++; }
        for (int ls = 0; ls < segs[j].len; ls++) {
            int gs = segs[j].start + ls;
            rseg_h[2 * gs] = j; rseg_h[2 * gs + 1] = ls;
            float t = 1.f;
            if (c->log_floor > 0) {
                double en = (double)(segs[j].pos_base + ls + 1) / c->log_floor;
                if (en > 1.0) t = 1.f + c->log_alpha * (float)log(en);
            }
            tau_h[gs] = t;
            tau_h[N + gs] = 1.f;
        }
    }
    for (int v = 0; v < ch->NS; v++) {
        for (int j = 0; j < M; j++) {
            int k = slots[j], x = k * ch->NS + v;
            char *base = (char *)cc_dev_addr(ch->slot_kv[k]);
            ch->kv_tabh[(size_t)v * M + j] = (CcInk2KvRow){
                base + ch->sq_kq[x], (float *)(base + ch->sq_ks[x]),
                base + ch->sq_vq[x], (float *)(base + ch->sq_vs[x]),
                ch->scap[x], segs[j].pos_base
            };
        }
    }
    int R = ch->rows;
    ch->fslot = -2; ch->pack_nseg = M; ch->pack_ntiles = ntiles;
    int ok = cc_begin() &&
             cc_write(ch->x, 0, xh, (size_t)N * D * sizeof(float)) &&
             cc_write(ch->pack_segs, 0, segs, (size_t)M * sizeof(CcInk2Seg)) &&
             cc_write(ch->pack_tiles, 0, tile_h, (size_t)ntiles * 2 * sizeof(int)) &&
             cc_write(ch->pack_rseg, 0, rseg_h, (size_t)N * 2 * sizeof(int)) &&
             cc_write(ch->kv_tab, 0, ch->kv_tabh, (size_t)ch->NS * (size_t)M * sizeof(CcInk2KvRow)) &&
             cc_write(ch->ts, 0, tau_h, (size_t)N * sizeof(float)) &&
             cc_write(ch->ts, (size_t)R, tau_h + N, (size_t)N * sizeof(float)) &&
             ink2cc_push_state_slots(ch, m, slots, M);
    free(tile_h); free(rseg_h); free(tau_h);
    const int lp = c->loop_p > 1, ls = lp ? c->loop_s : L, le = lp ? c->loop_e : L;
    const int steps = L + (lp ? (c->loop_p - 1) * (le - ls) : 0);
    int pending = 0, B = 0, pi = 0, pv = 0, preB = 0, nprev = 0;
    for (int st = 0; st < steps && ok; st++) {
        int i, v, pass = 0;
        if (st < le) i = v = st;
        else if (st < le + (c->loop_p - 1) * (le - ls)) {
            pass = 1 + (st - le) / (le - ls); i = ls + (st - le) % (le - ls); v = ink_sl(c, pass, i);
        } else i = v = st - (lp ? (c->loop_p - 1) * (le - ls) : 0);
        if (lp && i == ls) {
            if (pass == 0) preB = B;
            else { nprev = B - preB - 1; B = preB; }
        }
        ok = ink2cc_seg0_ops(ch, m, i, v, pass, N, pending, pi, pv, 1, B, preB, nprev, lp, ls, st) &&
             ink2cc_attn_dyn_q8_packed(ch, m, i, v, N, M) &&
             ink2cc_seg1_ops(ch, m, i, v, pass, N, 1, B, lp, 1);
        if (ok && i % blkn == 0) B++;
        pending = 1; pi = i; pv = v;
    }
    if (ok && pending) ok = ink2cc_join(ch, m, pi, pv, N, ch->pf_y);
    ok = ok && ink2cc_mix(ch, ch->x, B, ch->o_outq, ch->hq, N, D);
    if (ok && m_out > 0) {
        CcRegion rg[64]; int oi = 0;
        for (int j = 0; j < M; j++) if (want_logit[j]) {
            rg[oi] = (CcRegion){(size_t)oi * D, (size_t)(segs[j].start + segs[j].len - 1) * D, (size_t)D};
            oi++;
        }
        CcSconv dv = {2, 0, 0, 0, 0, 0, 0, 0, m_out * D, c->mup};
        CcNorm fp = {m_out, D, 1, 0, D, D, 0, D, D, (int)ch->o_final, 0, 0, c->eps, 1.f};
        ok = cc_copy_regions(ch->res, ch->hq, rg, m_out) &&
             cc_norm(ch->res, ch->prm, ch->fin, &fp) && cc_sconv(ch->fin, NULL, NULL, &dv) &&
             cc_matmul(ch->t_lm, ch->fin, 0, ch->out_dev, 0, m_out) &&
             cc_ink2_softcap(ch->out_dev, m_out * V, c->logit_cap) &&
             cc_copy(ch->outd, 0, ch->out_dev, 0, (size_t)m_out * V);
    }
    double t0 = now_s();
    ok = ok && cc_submit(1);
    m->t_attn += now_s() - t0;
    if (!ok) {
        if (!cc_lost()) cc_finish();
        ink2cc_fatal("the device failed during a packed prefill step");
        return 0;
    }
    for (int st = 0; st < steps; st++) {
        int i, v;
        if (st < le) i = v = st;
        else if (st < le + (c->loop_p - 1) * (le - ls)) {
            int pass = 1 + (st - le) / (le - ls); i = ls + (st - le) % (le - ls); v = ink_sl(c, pass, i);
        } else i = v = st - (lp ? (c->loop_p - 1) * (le - ls) : 0);
        ink2cc_dmoe_done(ch, m, i, v, N, 0, 1);
    }
    for (int j = 0; j < M; j++) { ch->scs_where[slots[j]] = INK2CC_DEV; ch->shost_zero[slots[j]] = 0; }
    if (logits_out) *logits_out = m_out > 0 ? (const float *)cc_ptr(ch->outd) : NULL;
    ch->forwards++;
    return 1;
}

static void ink2cc_report(Model *m) {
    Ink2CuChain *ch = ink2cc_of(m);
    if (!ch || !ch->ok || !ch->forwards) return;
    CcStats st; cc_stats(&st);
    fprintf(stderr, "[CUDA] inkling2 chain: %llu forwards (%llu mux, %llu rows), %llu frames (%llu ops, %llu matmuls), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %llu layer steps' experts in the frame, "
                    "%.1f MiB on the device\n",
            ch->forwards, ch->mux_forwards, ch->mux_rows, st.frames, st.ops, st.matmuls, st.wait_ms, ch->host_ms, ch->dm_layers, st.dev_bytes / 1048576.0);
    if (ch->g_full_capture || ch->g_full_launch || ch->g_capture || ch->g_launch)
        fprintf(stderr, "[CUDA] inkling2 chain: INK2_CC_GRAPH: full %llu captured, %llu replayed | seg %llu captured, %llu replayed, %llu eager\n",
                ch->g_full_capture, ch->g_full_launch, ch->g_capture, ch->g_launch, ch->g_eager);
}

/* the chain asked for at the tier's start (g_ink2cc_want): COLI_CUDA_CHAIN 1 or 2; unset, Inkling-2-Mini's
 * loop where the backend has its ops (its experts in the frame, cc_ink2_caps); 0 never */
static int ink2cc_wanted(const Cfg *c) {
    const char *e = getenv("COLI_CUDA_CHAIN");
    if (e && *e) return *e == '1' || *e == '2';
    return c->loop_p > 1 && cc_available() && (cc_ink2_caps() & (CC_INK2_MOE | CC_INK2_LOOPGATE)) == (CC_INK2_MOE | CC_INK2_LOOPGATE);
}
/* COLI_CUDA_CHAIN for Inkling-2, after the CUDA tier (ink_cu_tier_start): 1 on, 2 prompts only
 * (g_ink2cc_want: asked for, and the tier left its memory). */
static void ink2cc_start(Model *m) {
    g_cu_chain = 0;
    if (!m->c.v2 || !g_cuda || !g_ink2cc_want) return;
    const char *e = getenv("COLI_CUDA_CHAIN");
    if (g_ink_mux_slots > 1 && !cc_ink2_mux_ready()) {
        fprintf(stderr, "[CUDA] inkling2: KV_SLOTS=%d: the CUDA backend has no multi-slot mux ops (an older DLL); dense chain off\n",
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
    fprintf(stderr, "[CUDA] inkling2: dense chain on%s%s (COLI_CUDA_CHAIN%s)\n", g_cu_chain == 2 ? " for prompts" : "",
            g_ink_mux_slots > 1 ? ", KV_SLOTS batched Q8 KV" : "", e && *e ? "" : " unset: Inkling-2-Mini's default");
}
