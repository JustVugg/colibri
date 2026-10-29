/* qwen36_cuda_chain.h -- Qwen3.6's layers as a dense chain on a CUDA device (cuda_chain.h).
 * Included once by qwen36.c in a COLI_CUDA build, after the CPU forward it stands in
 * for; on unless COLI_CUDA_CHAIN=0 (q36cc_setup decides, at startup, after the tier
 * placed the trunk). The CUDA twin of qwen36_chain.h: the same frames, the same
 * crossings, the same state contract, so the engine wires both chains the same way.
 *
 * What runs where, per layer, for one block of rows (decode: one row):
 *   device, frame A1: the routed MoE output of the layer before joins the residual
 *     (x += routed + gate * shared, CC_EW_COMBINE, the CPU's order), the input
 *     RMSNorm, then either the gated attention (q/k/v, per-head q/k norm, RoPE from a
 *     host table, the K/V rows into the device cache, attention with the output gate,
 *     o_proj) or the Gated DeltaNet (the tier's fused qkv ++ z projection, b|a, the
 *     convolution with its ring, the recurrence with its state, the gated norm,
 *     out_proj), the residual add, the post-attention RMSNorm and the router logits.
 *     Then the frame is waited for.
 *   host: the router's softmax and top-k, the routed experts (the CUDA expert tier's
 *     groups and the CPU's share, joined in rank order: moe_ex routed_only), the K/V
 *     rows copied into the host cache.
 *   device, frame A2 (not waited for): the shared expert and its gate, on the chain's
 *     stream, while the host computes the routed experts on the tier's.
 * Crossing per layer: the normalized rows (D floats per row) and the router logits
 * (E per row) down, the K/V rows of an attention layer down, the routed sum (D per
 * row) up. The last frame normalizes the last row(s) and runs lm_head; its logits
 * come back.
 *
 * The matrices are the tier's resident copies (qt_dense_tensor and friends): the
 * chain needs every projection of every layer, the shared expert's three matrices
 * and lm_head on ONE device, so the chain (on unless COLI_CUDA_CHAIN=0) also offers the shared expert to
 * the placer (Q36_OFFER_SHEXP's effect) and declines, naming the first matrix on the
 * CPU, when the placement left anything behind (COLI_PLACE=auto on an 8 GB card
 * takes the whole 35B trunk). It uploads itself the router's matrix, the DeltaNet b|a
 * rows and the shared expert's gate row (small, f32 or int8), and the parameter arena.
 *
 * State and who owns it: as the Vulkan chain's. The residual stream lives on the
 * device for a forward. The host's KV cache stays canonical (every step copies its new
 * rows back); the device holds a mirror per layer with a watermark kv_valid. The
 * DeltaNet state and conv rings stay on the device while the chain runs (dn_where);
 * the host copy is brought back before anything reads it there (a pinned snapshot, a
 * CPU step) and pushed up after anything writes it there (reset_recurrent: zeros;
 * pin_restore: an upload). A speculative verify copies the state after each of its
 * rows but the last into a device slot (q36cc_rollback swaps a slot in). Q36_DN_GPU's
 * per-layer step (qt_dn_gpu_*) is not set up while the chain is on: the chain owns the
 * state. A multiplexed decode (KV_SLOTS > 1) and PILOT prefetch keep the CPU path.
 *
 * A CUDA error inside a frame loses the device to the chain: the engine rebuilds the
 * recurrent state on the CPU from the prefix record and runs there from then on
 * (q36cc_recover, as q36c_recover). COLI_CUDA_CHAIN_FAULT=n fakes it at the n-th
 * frame, for the tests. */
#include "cuda_chain.h"

#define Q36CC_HOST 0
#define Q36CC_DEV  1
#define Q36CC_BOTH 2

/* KV_SLOTS > 1: one conversation's state on the device (q36cc_conv_*), as qwen36_chain.h's Q36cConv */
typedef struct {
    int slot;                                  /* -1: free */
    CcBuf **rec, **ring, **kc, **vc;
    int *kv_valid, *kv_rows;
    int dn_where, host_zero, cap;              /* cap: the rows its full layers' mirrors hold (0: none now) */
    unsigned long long used;
} Q36ccConv;

#define Q36CC_PROMO_SLOTS 32   /* promotions in flight at most (their staging: an expert each) */
#define Q36CC_PROMO_WIN 64     /* decode steps the brake looks back on (q36cc_promote) */
typedef struct {
    int ok, failed;
    int dev;                                   /* the CUDA ordinal every matrix sits on */
    int n, head;                               /* every layer, and lm_head, on the device */
    int rows;                                  /* scratch capacity in rows */
    int cap;                                   /* the host's kv_cap the mirrors follow */
    CcBuf *prm;                                /* every norm / conv / DeltaNet parameter */
    size_t *o_in, *o_post, *o_qn, *o_kn, *o_conv, *o_dn, o_final;
    size_t *o_snk;                             /* Flint: a layer's sink logits in prm */
    size_t *o_ra, *o_rf, *o_pre, *o_pf;        /* Muse Spark: residual gates (alpha | beta), pre_en, post_fn */
    ColiCudaTensor **t_ag, **t_ld, **t_lu;     /* Muse Spark: the attention gate and the latent projections (the tier's) */
    ColiCudaTensor **t_ab, **t_sg, **t_gate;   /* the chain's own: DeltaNet b|a rows, shared gate row, router */
    uint8_t *rgf;                              /* t_gate holds the shared gate's row after the router's (lg: E + 1 a row) */
/* the rows up to which a step reads the shared gate from the router's GEMV (rgf): the backend's
 * GEMV (GEMV_I8_MAX_S), where each output row has the separate row's bits; a prompt chunk's GEMM
 * sums in another order, so a chunk keeps the separate row (t_sg) and the prompt its bits */
#define Q36CC_RG_ROWS 8
    ColiCudaTensor **t_q, **t_k, **t_v, **t_o, **t_dnp, **t_dno, **t_shg, **t_shu, **t_shd, *t_lm;   /* the tier's */
    CcBuf **rec, **ring, **kc, **vc;           /* per layer state */
    CcBuf **rec_snap[Q36_SPEC_SNAPS], **ring_snap[Q36_SPEC_SNAPS];   /* a verify's copies, slot r after row r */
    int snap_slots, snap_valid;
    int *kv_valid, *attn_ord, n_attn;
    int *kv_rows;                              /* per attention layer: its mirror's rows (a sliding layer's ring, else cap) */
    int dn_where, host_zero;
    CcBuf *x, *nrm, *tmp, *q, *k, *v, *ctx, *qkvz, *ab, *cv, *dny, *h2, *lg, *gs, *us, *hs, *ds, *sgd, *fin, *out;
    CcBuf *h2d, *lgd, *kvd, *outd, *xd, *lfd, *routed, *cs;
    CcBuf *csd;   /* cs on the device: every RoPE of a step reads it there, not over the bus */
    CcBuf *kg, *vg;   /* a long step's window over a ring: its rows before the step and the step's (q36cc_attention) */
    CcBuf *agb, *lat, *latd;   /* Muse Spark: the attention gate's rows, the experts' input rows */
    float *host_routed;
    /* the routed experts in the frame (q36cc_dm_*): the tier's residents on this device by
     * address ([L][E] entries, the host's copy of who is in it, the layers with some expert
     * elsewhere), the routes ([L][rows][K], on the device and down), the experts' rows,
     * their sum, and a hybrid layer's host rows and words */
    int dm_on, dm_fmt, dm_gs, dm_read, dm_held;
    unsigned dm_epoch, dm_seq;
    int *dm_nres;   /* a layer's experts the table lacks (dm_part = any) */
    int dm_stage_dirty;   /* the staged table is behind the residents */
    /* promotions (q36cc_promote): the rate a step, a ring of page-locked staging slots the
     * newcomers' bytes go up from, the pairs whose copies are in flight */
    int promo_rate, pr_n, pr_busy[Q36CC_PROMO_SLOTS]; size_t pr_stride; CcBuf *pr_buf;
    struct { int ci, vi, slot; unsigned tk; } pr_pend[Q36CC_PROMO_SLOTS]; int pr_npend;
    unsigned long long dm_promoted, dm_landed, dm_rebuilds, dm_patches; double promo_ms, patch_ms;
    /* the brake (q36cc_promote): the decode steps of the last Q36CC_PROMO_WIN, their routes and
     * misses; above promo_hits percent hits nothing moves (the baselines: the counters at the
     * last step that was looked at) */
    int promo_hits, pr_wi, pr_wn; unsigned pr_wroute[Q36CC_PROMO_WIN], pr_wmiss[Q36CC_PROMO_WIN];
    unsigned long long pr_pairs0, pr_steps0, pr_braked;
    CcExpert *dm_host; uint8_t *dm_res; int *dm_part; uint8_t *dm_ran;   /* dm_ran: the step's layers that ran so */
    /* a prompt chunk's table (more than Q36CC_STAGE_ROWS rows): the residents, and the tier's host
     * pool staged a layer at a time into dm_stage (planar; dm_fr the copies, dm_fro and dm_frn each
     * layer's), the layers with some expert in neither; dm_stage_off: no staging from here on */
    CcExpert *dm_hostb; int *dm_partb, *dm_fro, *dm_frn; CcFetch *dm_fr; CcBuf *dm_tabb, *dm_stage;
    int dm_stage_off; unsigned long long dm_staged, dm_stage_bytes;
    unsigned long long hp_follows, hp_joined, hp_left; double hp_ms;   /* the pool following the residents */
    int rows_max;   /* a prompt chunk's rows (q36cc_rows_max) */
    CcBuf *dm_tab, *dm_ids, *dm_w, *dm_idsd, *dm_wd, *dm_hid, *dm_y, *dm_hy, *dm_dfl, *dm_hfl;
    float *dm_g, *dm_u, *dm_hh;
    unsigned long long dm_steps, dm_hyb, dm_pairs;   /* layer steps in the frame, the hybrid ones, slots the host computed */
    double dm_host_ms, dm_spin_ms;
    unsigned long long forwards, frames, fault_at, dec_n;   /* dec_n: one-row forwards (decode tokens) */
    double wait_ms, host_ms, dec_wait_ms, dec_host_ms;      /* and their shares of the waits and the host's experts */
    /* KV_SLOTS > 1: the conversations' states, and a multiplexed step's rows while it runs */
    Q36ccConv *conv; int nconv, conv_cur, conv_full;
    unsigned long long conv_clock, conv_handed, mux_steps, mux_rows, mux_host_steps;
    const Q36Row *mx_rows; Q36ccConv **mx_conv;
} Q36CChain;

static int g_cuda_chain = 0;     /* COLI_CUDA_CHAIN asked for it and the chain is up */
static Q36CChain *q36cc_of(Model *m) { return (Q36CChain *)m->cchain; }
static int q36cc_dm_able(Q36CChain *ch, Model *m);

/* on unless COLI_CUDA_CHAIN=0: the chain took 34 -> 59 tok/s of decode on Qwen3.6-35B int4-g64, 85 with its
 * routed experts in the frame (an RTX PRO 4000); q36cc_setup declines where it cannot run */
static int q36cc_env_on(void) { const char *e = getenv("COLI_CUDA_CHAIN"); return !(e && *e == '0'); }
static int q36cc_rows_env(void) {
    const char *e = getenv("COLI_CUDA_CHAIN_ROWS");
    int r = e && *e ? atoi(e) : 0;
    return r < 0 ? 0 : r > 8192 ? 8192 : r;
}
static void q36cc_fatal(const char *what) {
    fprintf(stderr, "[chain] qwen36: %s -- stopping (COLI_CUDA_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
/* The device was lost with the newest recurrent state on it. The host's KV rows are
 * canonical, the DeltaNet state is not: rebuild it on the CPU by running the `upto`
 * positions the prefix record names (a prefill on the CPU), and leave the chain off. */
static void q36cc_recover(Model *m, int upto) {
    Q36CChain *ch = q36cc_of(m);
    Cfg *c = &m->c; int D = c->hidden;
    g_cuda_chain = 0;
    if (ch) { ch->failed = 1; ch->dn_where = Q36CC_HOST; ch->host_zero = 0; ch->snap_valid = 0; }
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        q36cc_fatal("the device was lost with a recurrent state its token ids do not describe (an image)");
    fprintf(stderr, "[chain] qwen36: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", upto);
    int *ids = malloc((size_t)upto * sizeof(int));
    float *x = falloc((int64_t)upto * D);
    if (!ids) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    memcpy(ids, m->kvp.fed, (size_t)upto * sizeof(int));
    for (int s = 0; s < upto; s++) q36_embed_row(m, ids[s], s, x + (int64_t)s * D);
    int snap = m->snap_rows, rowwise = g_q36_rowwise;
    m->snap_rows = 0; g_q36_rowwise = 0;
    layers_forward_range(m, x, upto, 0, 0, c->n_layers, 0, NULL);
    m->snap_rows = snap; g_q36_rowwise = rowwise;
    free(ids); free(x);
}

static int q36cc_geometry_ok(const Cfg *c) {
    int kd = c->dn_kdim, kd_ok = kd == 8 || kd == 16 || kd == 32 || kd == 64 || kd == 128 || kd == 256;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) {
            if (c->head_dim > 256 || c->head_dim != c->k_head_dim || c->q_heads % c->kv_heads ||
                c->q_head_dim < c->head_dim || (c->rotary_dim & 1) || c->rotary_dim > c->head_dim) return 0;
        } else {
            if (c->dn_vdim > 128 || !kd_ok || c->dn_convk < 2 || c->dn_convk > 9 || c->dn_vheads % c->dn_kheads) return 0;
        }
    }
    return 1;
}

/* f32 rows [O x I] as a resident fmt 0 tensor (unit scales: the kernel reads none) */
static ColiCudaTensor *q36cc_f32_tensor(int dev, const float *w, int I, int O) {
    ColiCudaTensor *t = NULL;
    float *ones = falloc(O);
    for (int o = 0; o < O; o++) ones[o] = 1.f;
    int ok = coli_cuda_tensor_upload(&t, w, ones, 0, I, O, dev);
    free(ones);
    return ok ? t : NULL;
}
/* a dense QW as the tier would place it: its int8 rows, else its f32 copy */
static ColiCudaTensor *q36cc_qw_tensor(int dev, const QW *w) {
    ColiCudaTensor *t = NULL;
    if (w->q && w->sc) return coli_cuda_tensor_upload(&t, w->q, w->sc, 1, w->I, w->O, dev) ? t : NULL;
    if (w->w) return q36cc_f32_tensor(dev, w->w, w->I, w->O);
    return NULL;
}

/* The tier's tensor behind a handle, on the chain's device: 0 when it is elsewhere */
static int q36cc_take(Q36CChain *ch, ColiCudaTensor **slot, ColiCudaTensor *t, int dev, const char *what, int layer) {
    if (!t) { fprintf(stderr, "[chain] qwen36: %s of layer %d is not on a device; the chain stays off\n", what, layer); return 0; }
    if (ch->dev < 0) ch->dev = dev;
    if (dev != ch->dev) {
        fprintf(stderr, "[chain] qwen36: %s of layer %d sits on device %d, the chain's matrices on %d; the chain stays off\n",
                what, layer, dev, ch->dev);
        return 0;
    }
    *slot = t;
    return 1;
}

/* The device bytes q36cc_setup's own tensors take (the router, b|a and gate rows), as it
 * uploads them: qwen36's main keeps them out of the tier's allowance (qt_reserve). */
static size_t q36cc_own_bytes(Model *m) {
    Cfg *c = &m->c; size_t b = 0;
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        if (!c->is_attn[i]) b += (size_t)2 * c->dn_vheads * c->hidden * 4;
        if (c->n_experts > 0 && !q36_dense_l(c, i)) b += (size_t)l->gate.I * l->gate.O * (l->gate.q ? 1 : 4);
        if (l->sh_gate && l->sh_I > 0) b += (size_t)c->hidden * 4;
    }
    return b;
}
/* The chain's KV mirrors at the context's capacity, for a model with a sliding window or
 * sinks (Flint, Muse Spark): the tier leaves them free (qt_reserve), as the Vulkan tier does
 * (q36c_late_bytes). A sliding layer's ring where the backend's attention reads one, the full
 * layers' rows at ctx; at most half of the device's memory. Before, Flint int4 at 32K filled
 * the card with experts and the chain's prompt scratch found no room ("CPU from here on"). */
static int q36cc_gguf_dims(Model *m, int i, size_t *mg, size_t *md);
/* The mirrors in float16 where the host's cache is (q36_kv16) and the backend reads one
 * (cc_kvh_ready): the host's numbers, in half the memory and half of a decode step's reads.
 * COLI_CUDA_KV16=0 keeps float32. */
static int q36cc_kvh(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_KV16"); on = !(e && *e == '0'); }
    return on && q36_kv16() && cc_kvh_ready();
}
static size_t q36cc_kves(void) { return q36cc_kvh() ? 2 : sizeof(float); }   /* a mirror's element */
/* a step's float32 K or V rows into a mirror */
static int q36cc_kv_put(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) {
    return q36cc_kvh() ? cc_copy_regions_h(dst, src, r, n) : cc_copy_regions(dst, src, r, n);
}
/* a mirror's rows into another of its type (r's offsets and lengths in its elements; a float16
 * mirror's are even, copied as floats two halves each) */
static int q36cc_kv_copy(CcBuf *dst, CcBuf *src, CcRegion *r, int n) {
    if (q36cc_kvh()) for (int k = 0; k < n; k++) { r[k].dst /= 2; r[k].src /= 2; r[k].n /= 2; }
    return cc_copy_regions(dst, src, r, n);
}
/* the attention over mirrors of either type: plain, or a window and sinks (ring: its rows) */
static int q36cc_attn_c(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, const CcAttn *a) {
    return q36cc_kvh() ? cc_attn_h(q, kc, vc, o, gate, NULL, a) : cc_attn(q, kc, vc, o, gate, NULL, a);
}
static int q36cc_attn_wc(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *snk, const CcAttnW *w, int ring) {
    if (q36cc_kvh()) return cc_attn_wh(q, kc, vc, o, gate, snk, w, ring);
    return ring ? cc_attn_wr(q, kc, vc, o, gate, snk, w, ring) : cc_attn_w(q, kc, vc, o, gate, snk, w);
}
/* the mirrors at ctx positions, any model's (a sliding layer's ring where the backend reads one),
 * as float32: the tier's room for them. A float16 mirror takes half, and the half it leaves goes
 * to the prompt's chunk rows: with the tier's room halved, Flint UD-Q3_K_XL with 7 GiB free on
 * an RTX PRO 4000 took 2048 rows a chunk instead of 4096, its 4K-token prompt 1768 -> 1190 tok/s */
static size_t q36cc_mirror_at(Model *m, int ctx) {
    Cfg *c = &m->c;
    if (ctx < 1) return 0;
    int R = c->win_layer && cc_attn_wr_ready() ? q36_ring_for(c, ctx) : 0;
    size_t row = (size_t)c->kv_heads * c->k_head_dim * 2 * sizeof(float), kv = 0;
    for (int i = 0; i < c->n_layers; i++)
        if (c->is_attn[i]) kv += row * (size_t)(R && c->win_layer[i] ? R : ctx);
    return kv;
}
static size_t q36cc_mirror_bytes(Model *m, int ctx) {
    Cfg *c = &m->c;
    if (ctx < 1 || !(c->win_layer || c->attn_sinks)) return 0;
    size_t kv = q36cc_mirror_at(m, ctx), fr = 0, tot = 0;
    if (coli_cuda_mem_info(0, &fr, &tot) && tot && kv > tot / 2) kv = tot / 2;
    return kv;
}
/* A GGUF's prompt staging (q36cc_stage_table_gguf): its most experts of a layer, every one where
 * the card holds fewer than all of them, at the largest layer's slot. The tier keeps it out of
 * its expert budget when not every expert fits (qwen36's main, qt_reserve_short): taken from the
 * mirrors' room, a 16K prompt on Flint (RTX PRO 4000) found no memory for the routed experts'
 * buffers and took the host's step from then on (309.7 tok/s against 1032). */
static size_t q36cc_gguf_stage_bytes(Model *m) {
    Cfg *c = &m->c; size_t st = 0;
    if (c->n_experts <= 0) return 0;
    for (int i = 0; i < c->n_layers; i++) {
        size_t mg, md;
        if (!q36_dense_l(c, i) && q36cc_gguf_dims(m, i, &mg, &md)) {
            size_t b = (size_t)c->n_experts * (2 * ((mg + 255) / 256 * 256) + (md + 255) / 256 * 256);
            if (b > st) st = b;
        }
    }
    return st;
}
static void q36cc_promote_settle(Q36CChain *ch);
static void q36cc_free_own(Q36CChain *ch, Model *m) {
    q36cc_promote_settle(ch);   /* the copies in flight: landed (the finish before) or failed (a lost device) */
    int L = m->c.n_layers;
    if (ch->t_ab) for (int i = 0; i < L; i++) { if (ch->t_ab[i]) coli_cuda_tensor_free(ch->t_ab[i]); ch->t_ab[i] = NULL; }
    if (ch->t_sg) for (int i = 0; i < L; i++) { if (ch->t_sg[i]) coli_cuda_tensor_free(ch->t_sg[i]); ch->t_sg[i] = NULL; }
    if (ch->t_gate) for (int i = 0; i < L; i++) { if (ch->t_gate[i]) coli_cuda_tensor_free(ch->t_gate[i]); ch->t_gate[i] = NULL; }
    if (ch->rec) for (int i = 0; i < L; i++) { cc_free(ch->rec[i]); ch->rec[i] = NULL; }
    if (ch->ring) for (int i = 0; i < L; i++) { cc_free(ch->ring[i]); ch->ring[i] = NULL; }
    if (ch->kc) for (int i = 0; i < L; i++) { cc_free(ch->kc[i]); ch->kc[i] = NULL; }
    if (ch->vc) for (int i = 0; i < L; i++) { cc_free(ch->vc[i]); ch->vc[i] = NULL; }
    cc_free(ch->prm); ch->prm = NULL;
}

/* The chain on the device, at startup after the tier placed the trunk: every layer's
 * matrices found on one device, the parameter arena, the chain's own small tensors and
 * the DeltaNet state. NULL = the chain cannot run (the message says why). */
static Q36CChain *q36cc_setup(Model *m, int mux_slots) {
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden;
    if (m->cchain) return ((Q36CChain *)m->cchain)->ok ? (Q36CChain *)m->cchain : NULL;
    if (!cc_available()) { fprintf(stderr, "[chain] qwen36: the CUDA backend has no chain (an older DLL); the per-matrix path\n"); return NULL; }
    if (!qt_ready()) { fprintf(stderr, "[chain] qwen36: no CUDA expert tier; the chain stays off\n"); return NULL; }
    if (qq_active() || g_pilot) {
        fprintf(stderr, "[chain] qwen36: %s; the chain stays off\n",
                qq_active() ? "a qpack container" : "PILOT prefetch reads the residual on the host");
        return NULL;
    }
    if (mux_slots > 1)
        fprintf(stderr, "[chain] qwen36: KV_SLOTS=%d: the chain keeps each conversation's state on the card while there is "
                        "room, and runs its prompts and every conversation's decode rows (COLI_VK_CHAIN_MUX caps the places)\n", mux_slots);
    if (c->muse && !cc_muse_ready()) {
        fprintf(stderr, "[chain] qwen36: the CUDA backend has no Muse Spark norms and gates (an older DLL); the per-matrix path\n");
        return NULL;
    }
    if ((c->attn_sinks || c->win_layer) && !cc_attn_w_ready()) {
        fprintf(stderr, "[chain] qwen36: the CUDA backend has no attention with a window and sinks (an older DLL); "
                        "the per-matrix path\n");
        return NULL;
    }
    if (q36_kv16() && !cc_kv16_ready()) {   /* its K and V would not be the host's float16 numbers */
        if (!m->K) {
            g_q36_kv16 = 0;
            fprintf(stderr, "[chain] qwen36: the CUDA backend does not round K and V to float16 (an older DLL): the KV cache stays float32\n");
        } else {
            fprintf(stderr, "[chain] qwen36: the CUDA backend does not round K and V to float16 (an older DLL); the per-matrix path\n");
            return NULL;
        }
    }
    if (!q36cc_geometry_ok(c)) {
        fprintf(stderr, "[chain] qwen36: a geometry its kernels do not take (head dim %d, DeltaNet %dx%d); the per-matrix path\n",
                c->head_dim, c->dn_kdim, c->dn_vdim);
        return NULL;
    }
    Q36CChain *ch = (Q36CChain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    m->cchain = ch;
    ch->dev = -1;
    ch->o_in = calloc(L, sizeof(size_t)); ch->o_post = calloc(L, sizeof(size_t)); ch->o_qn = calloc(L, sizeof(size_t));
    ch->o_kn = calloc(L, sizeof(size_t)); ch->o_conv = calloc(L, sizeof(size_t)); ch->o_dn = calloc(L, sizeof(size_t));
    ch->o_snk = calloc(L, sizeof(size_t));
    ch->o_ra = calloc(L, sizeof(size_t)); ch->o_rf = calloc(L, sizeof(size_t));
    ch->o_pre = calloc(L, sizeof(size_t)); ch->o_pf = calloc(L, sizeof(size_t));
    ch->t_ag = calloc(L, sizeof(void *)); ch->t_ld = calloc(L, sizeof(void *)); ch->t_lu = calloc(L, sizeof(void *));
    ch->t_ab = calloc(L, sizeof(void *)); ch->t_sg = calloc(L, sizeof(void *)); ch->t_gate = calloc(L, sizeof(void *));
    ch->rgf = calloc(L, 1);
    ch->t_q = calloc(L, sizeof(void *)); ch->t_k = calloc(L, sizeof(void *)); ch->t_v = calloc(L, sizeof(void *));
    ch->t_o = calloc(L, sizeof(void *)); ch->t_dnp = calloc(L, sizeof(void *)); ch->t_dno = calloc(L, sizeof(void *));
    ch->t_shg = calloc(L, sizeof(void *)); ch->t_shu = calloc(L, sizeof(void *)); ch->t_shd = calloc(L, sizeof(void *));
    ch->rec = calloc(L, sizeof(void *)); ch->ring = calloc(L, sizeof(void *));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->attn_ord = calloc(L, sizeof(int)); ch->kv_rows = calloc(L, sizeof(int));
    if (!ch->o_in || !ch->o_post || !ch->o_qn || !ch->o_kn || !ch->o_conv || !ch->o_dn || !ch->o_snk || !ch->t_ab || !ch->t_sg || !ch->t_gate || !ch->rgf ||
        !ch->o_ra || !ch->o_rf || !ch->o_pre || !ch->o_pf || !ch->t_ag || !ch->t_ld || !ch->t_lu ||
        !ch->t_q || !ch->t_k || !ch->t_v || !ch->t_o || !ch->t_dnp || !ch->t_dno || !ch->t_shg || !ch->t_shu || !ch->t_shd ||
        !ch->rec || !ch->ring || !ch->kc || !ch->vc || !ch->kv_valid || !ch->attn_ord || !ch->kv_rows) return NULL;
    /* the tier's matrices, every one on one device */
    /* (the accessor fills `d` before q36cc_take reads it: two statements, not one call) */
#define Q36CC_TAKE(slot, expr, what) do { int d_ = -1; ColiCudaTensor *t_ = (expr); if (!q36cc_take(ch, (slot), t_, d_, (what), i)) return NULL; } while (0)
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        if (c->is_attn[i]) {
            Q36CC_TAKE(&ch->t_q[i], qt_dense_tensor(l->qth_q - 1, &d_), "q_proj");
            Q36CC_TAKE(&ch->t_k[i], qt_dense_tensor(l->qth_k - 1, &d_), "k_proj");
            Q36CC_TAKE(&ch->t_v[i], qt_dense_tensor(l->qth_v - 1, &d_), "v_proj");
            Q36CC_TAKE(&ch->t_o[i], qt_dense_tensor(l->qth_o - 1, &d_), "o_proj");
            if (c->muse) Q36CC_TAKE(&ch->t_ag[i], qt_dense_tensor(l->qth_ag - 1, &d_), "the attention's gate");
        } else {
            Q36CC_TAKE(&ch->t_dnp[i], qt_dnproj_tensor(i, &d_), "the DeltaNet in_proj");
            Q36CC_TAKE(&ch->t_dno[i], qt_dense_tensor(l->qth_dnout - 1, &d_), "the DeltaNet out_proj");
        }
        if (l->sh_I > 0) {   /* shared_inter, or a Flint dense layer's MLP */
            Q36CC_TAKE(&ch->t_shg[i], qt_dense_tensor(l->qth_shg - 1, &d_), "the shared expert's gate");
            Q36CC_TAKE(&ch->t_shu[i], qt_dense_tensor(l->qth_shu - 1, &d_), "the shared expert's up");
            Q36CC_TAKE(&ch->t_shd[i], qt_dense_tensor(l->qth_shd - 1, &d_), "the shared expert's down");
        }
        if (c->muse) {
            Q36CC_TAKE(&ch->t_ld[i], qt_dense_tensor(l->qth_ld - 1, &d_), "the projection to the experts' width");
            Q36CC_TAKE(&ch->t_lu[i], qt_dense_tensor(l->qth_lu - 1, &d_), "the projection from the experts' width");
        }
    }
    { int i = 0; Q36CC_TAKE(&ch->t_lm, qt_lmhead_tensor(&d_), "lm_head"); }
#undef Q36CC_TAKE
    ch->head = 1;
    if (!cc_init(ch->dev)) { fprintf(stderr, "[chain] qwen36: the chain's context on device %d did not come up\n", ch->dev); return NULL; }
    /* the parameter arena: offsets, then one upload */
    size_t np = 0;
    int vh = c->dn_vheads, conv_dim = c->dn_conv_dim, convk = c->dn_convk;
    for (int i = 0; i < L; i++) {
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
    for (int i = 0; i < L; i++) {
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
    ch->prm = cc_buf(np * sizeof(float), CC_DEV);
    int ok = ch->prm && cc_begin() && cc_write(ch->prm, 0, arena, np * sizeof(float)) && cc_submit(1);
    free(arena);
    /* the chain's own tensors and the DeltaNet state */
    size_t own = 0;
    for (int i = 0; i < L && ok; i++) {
        Layer *l = &m->L[i];
        if (!c->is_attn[i]) {
            float *ab = malloc((size_t)2 * vh * D * sizeof(float));
            if (!ab) { ok = 0; break; }
            memcpy(ab, l->dn_b, (size_t)vh * D * sizeof(float));
            memcpy(ab + (size_t)vh * D, l->dn_a, (size_t)vh * D * sizeof(float));
            ch->t_ab[i] = q36cc_f32_tensor(ch->dev, ab, D, 2 * vh);
            free(ab);
            own += (size_t)2 * vh * D * 4;
            ch->rec[i] = ch->t_ab[i] ? cc_buf((size_t)vh * c->dn_kdim * c->dn_vdim * sizeof(float), CC_DEV) : NULL;
            ch->ring[i] = ch->rec[i] ? cc_buf((size_t)conv_dim * (convk - 1) * sizeof(float), CC_DEV) : NULL;
            ok = ch->t_ab[i] && ch->rec[i] && ch->ring[i];
        } else ch->attn_ord[i] = ch->n_attn++;
        if (ok && c->n_experts > 0 && !q36_dense_l(c, i)) {
            /* the router and the shared expert's gate row read the same rows: where both are f32, one
             * matrix of E + 1 rows and one GEMV a layer (Qwen3.6-35B UD-Q4_K_M: 40 launches of about
             * 3 us a decode token fewer); COLI_CUDA_CHAIN_ROUTER_GATE=0 keeps two */
            const char *rge = getenv("COLI_CUDA_CHAIN_ROUTER_GATE");
            if (!(rge && *rge == '0') && !c->muse && l->sh_gate && l->sh_I > 0 && !l->gate.q && l->gate.w &&
                l->gate.I == D && l->gate.O == c->n_experts) {
                float *rg = malloc((size_t)(l->gate.O + 1) * D * sizeof(float));
                if (rg) {
                    memcpy(rg, l->gate.w, (size_t)l->gate.O * D * sizeof(float));
                    memcpy(rg + (size_t)l->gate.O * D, l->sh_gate, (size_t)D * sizeof(float));
                    ch->t_gate[i] = q36cc_f32_tensor(ch->dev, rg, D, l->gate.O + 1);
                    free(rg);
                    ch->rgf[i] = ch->t_gate[i] != NULL;
                }
            }
            if (!ch->t_gate[i]) ch->t_gate[i] = q36cc_qw_tensor(ch->dev, &l->gate);
            ok = ch->t_gate[i] != NULL; own += (size_t)l->gate.I * (l->gate.O + ch->rgf[i]) * (l->gate.q ? 1 : 4);
        }
        if (ok && l->sh_gate && l->sh_I > 0) { ch->t_sg[i] = q36cc_f32_tensor(ch->dev, l->sh_gate, D, 1); ok = ch->t_sg[i] != NULL; own += (size_t)D * 4; }
    }
    if (!ok) {
        fprintf(stderr, "[chain] qwen36: device memory for the chain's parameters refused; the per-matrix path\n");
        q36cc_free_own(ch, m);
        return NULL;
    }
    ch->n = L;
    ch->dn_where = Q36CC_HOST; ch->host_zero = 0;
    { const char *e = getenv("COLI_CUDA_CHAIN_FAULT"); ch->fault_at = e && *e ? strtoull(e, NULL, 10) : 0; }
    ch->ok = 1;
    ch->dm_on = q36cc_dm_able(ch, m);
    if (ch->dm_on) fprintf(stderr, "[chain] qwen36: a decode step's routed experts in the frame (the tier's residents by address, "
                                   "format %d%s, group %d; COLI_CUDA_CHAIN_MOE=0: the host's step)\n", ch->dm_fmt,
                                   ch->dm_fmt >= 100 ? " (a GGUF's blocks, by layer)" : "", ch->dm_gs);
    fprintf(stderr, "[chain] qwen36: %d layers on CUDA device %d (%d attention), lm_head too; %.1f MiB of parameters, "
                    "%.1f MiB of router, b|a and gate rows, %.0f MB of DeltaNet state\n",
            L, ch->dev, ch->n_attn, np * 4 / 1048576.0, own / 1048576.0,
            (double)(L - ch->n_attn) * ((double)vh * c->dn_kdim * c->dn_vdim + (double)conv_dim * (convk - 1)) * 4 / 1e6);
    return ch;
}

/* the scratch buffers for `rows` rows */
static int q36cc_bufs(Q36CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    int qo = H * c->q_head_dim, kvo = KV * c->k_head_dim, vd = c->dn_vheads * c->dn_vdim;
    int E = c->n_experts > 0 ? c->n_experts : 1, SI = c->shared_inter > 0 ? c->shared_inter : 1;
    if (c->dense_layer && c->dense_inter > SI) SI = c->dense_inter;   /* Flint's dense layers: a wider MLP */
    size_t r = (size_t)rows, F = sizeof(float);
    /* the routed sum's rows: the experts' width (Muse Spark's latent), at least the hidden */
    const size_t RW = (size_t)(q36_eh(c) > D ? q36_eh(c) : D), LT = (size_t)q36_eh(c), AG = (size_t)H * hd;
    int ok = cc_reserve(&ch->x, r * D * F, CC_DEV) && cc_reserve(&ch->nrm, r * D * F, CC_DEV) && cc_reserve(&ch->tmp, r * D * F, CC_DEV) &&
             cc_reserve(&ch->q, r * qo * F, CC_DEV) && cc_reserve(&ch->k, r * kvo * F, CC_DEV) && cc_reserve(&ch->v, r * kvo * F, CC_DEV) &&
             cc_reserve(&ch->ctx, r * H * hd * F, CC_DEV) && cc_reserve(&ch->qkvz, r * (c->dn_conv_dim + vd) * F, CC_DEV) &&
             cc_reserve(&ch->ab, r * 2 * c->dn_vheads * F, CC_DEV) && cc_reserve(&ch->cv, r * c->dn_conv_dim * F, CC_DEV) &&
             cc_reserve(&ch->dny, r * vd * F, CC_DEV) && cc_reserve(&ch->h2, r * D * F, CC_DEV) && cc_reserve(&ch->lg, r * (E + 1) * F, CC_DEV) &&
             cc_reserve(&ch->gs, r * SI * F, CC_DEV) && cc_reserve(&ch->us, r * SI * F, CC_DEV) && cc_reserve(&ch->hs, r * SI * F, CC_DEV) &&
             cc_reserve(&ch->ds, r * D * F, CC_DEV) && cc_reserve(&ch->sgd, r * F, CC_DEV) && cc_reserve(&ch->fin, D * F, CC_DEV) &&
             cc_reserve(&ch->h2d, r * D * F, CC_DOWN) && cc_reserve(&ch->lgd, r * E * F, CC_DOWN) &&
             cc_reserve(&ch->kvd, (size_t)(ch->n_attn ? ch->n_attn : 1) * 2 * r * kvo * F, CC_DOWN) &&
             cc_reserve(&ch->out, (size_t)c->vocab * F, CC_DEV) && cc_reserve(&ch->outd, (size_t)c->vocab * F, CC_DOWN) &&
             cc_reserve(&ch->routed, r * RW * F, CC_UP) &&
             cc_reserve(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_UP) &&
             cc_reserve(&ch->csd, r * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_DEV);
    if (ok && c->muse)   /* Muse Spark's alone: the attention gate's rows, the experts' input rows */
        ok = cc_reserve(&ch->agb, r * AG * F, CC_DEV) && cc_reserve(&ch->lat, r * LT * F, CC_DEV) && cc_reserve(&ch->latd, r * LT * F, CC_DOWN);
    if (ok && m->kv_ring && cc_attn_wr_ready())   /* the device's rings: a long step's window, linear */
        ok = cc_reserve(&ch->kg, (r + c->window) * kvo * q36cc_kves(), CC_DEV) && cc_reserve(&ch->vg, (r + c->window) * kvo * q36cc_kves(), CC_DEV);
    return ok;
}
/* n of the host's K or V elements from idx (q36_kv_es() bytes each, q36_kv16) into a mirror at
 * element offset at: a float16 mirror takes the host's halves as they are (at and n even), a
 * float32 one the floats they hold */
static int q36cc_kv_up(CcBuf *dst, size_t at, const void *base, int64_t idx, size_t n) {
    if (q36cc_kvh()) return cc_write(dst, at / 2, (const uint16_t *)base + idx, n * 2);
    if (!q36_kv16()) return cc_write(dst, at, (const float *)base + idx, n * sizeof(float));
    float *f = malloc(n * sizeof(float));
    if (!f) return 0;
    kv16_unpack(f, (const uint16_t *)base + idx, (int64_t)n);
    int ok = cc_write(dst, at, f, n * sizeof(float));   /* a pageable source is staged before it returns */
    free(f);
    return ok;
}
/* A sliding layer's ring (q36_kv_rows) on the device too, where the backend's attention
 * reads one (cc_attn_wr; an older backend: a mirror of the cache's length, filled from the
 * host's ring): position t in row t % ring on both sides. 0 for a full layer. */
static int q36cc_ring(Model *m, int i) { return cc_attn_wr_ready() && q36_kv_rows(m, i) != m->max_t ? m->kv_ring : 0; }
/* the KV mirrors at the host's capacity (a cache that grew is mirrored again) */
static int q36cc_mirror(Q36CChain *ch, Model *m) {
    Cfg *c = &m->c; int KV = c->kv_heads;
    if (ch->cap != m->kv_cap) {
        for (int i = 0; i < c->n_layers; i++) {
            if (!c->is_attn[i]) continue;
            cc_free(ch->kc[i]); cc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
            ch->kv_valid[i] = 0;
            int R = q36cc_ring(m, i);
            ch->kv_rows[i] = R ? R : m->kv_cap;
            size_t bytes = (size_t)KV * ch->kv_rows[i] * c->k_head_dim * q36cc_kves();
            ch->kc[i] = cc_buf(bytes, CC_DEV);
            ch->vc[i] = cc_buf(bytes, CC_DEV);
            if (!ch->kc[i] || !ch->vc[i]) { ch->cap = 0; return 0; }
            /* the tensor cores' attention reads whole blocks of keys and masks the window's
             * outside: in a ring those rows are other positions', or never written */
            if (R && (!cc_zero(ch->kc[i], 0, bytes / sizeof(float)) || !cc_zero(ch->vc[i], 0, bytes / sizeof(float)))) { ch->cap = 0; return 0; }
        }
        ch->cap = m->kv_cap;
    }
    return 1;
}
static int q36cc_scratch(Q36CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = q36_eh(c) > c->hidden ? q36_eh(c) : c->hidden;   /* the routed sum's width */
    if (!q36cc_bufs(ch, m, rows)) return 0;
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, (size_t)rows * D * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    return 1;
}

/* ---- state between the host and the device ----------------------------------- */
static int q36cc_sync_one(Model *m, Q36CChain *ch) {
    if (!ch || !ch->ok || ch->dn_where != Q36CC_DEV) return 1;
    Cfg *c = &m->c;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    int ok = 1;
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (c->is_attn[i]) continue;
        ok = cc_read(ch->rec[i], 0, m->DN_rec[i], nr * sizeof(float)) &&
             cc_read(ch->ring[i], 0, m->DN_conv[i], nc * sizeof(float));
    }
    if (ok) ch->dn_where = Q36CC_BOTH;
    return ok;
}
/* The host's DeltaNet state brought up to date (before anything reads it there). */
static void q36cc_sync_host(Model *m) {
    if (!q36cc_sync_one(m, q36cc_of(m))) q36cc_recover(m, m->kv_len);
}
/* The host wrote its DeltaNet state (zero: reset_recurrent's zeros). */
static void q36cc_host_wrote(Model *m, int zero) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->ok) return;
    ch->dn_where = Q36CC_HOST; ch->host_zero = zero; ch->snap_valid = 0;
}
/* A CPU step from pos_base: the host state current before it, the device's stale after. */
static void q36cc_cpu_step(Model *m, int pos_base) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->ok) return;
    if (!q36cc_sync_one(m, ch)) { q36cc_recover(m, m->kv_len); return; }
    ch->dn_where = Q36CC_HOST; ch->host_zero = 0; ch->snap_valid = 0;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
}
/* ---- several conversations (KV_SLOTS > 1, serve_mux): qwen36_chain.h's scheme --------
 * Each conversation's state in a place on the card while there is room (COLI_VK_CHAIN_MUX
 * caps the places beside the chain's own), its mirrors sized to its length; the held one's
 * place is the chain's own fields; a decode step runs every row whose conversation has a
 * place in one forward (q36cc_mux_step). */
static int q36cc_pool_ok(Q36CChain *ch) { return ch && ch->ok && !ch->failed && g_q36_mux_slots > 1; }
static void q36cc_conv_save(Q36CChain *ch) {
    if (!ch->conv || ch->conv_cur < 0) return;
    Q36ccConv *v = &ch->conv[ch->conv_cur];
    v->rec = ch->rec; v->ring = ch->ring; v->kc = ch->kc; v->vc = ch->vc; v->kv_valid = ch->kv_valid;
    v->kv_rows = ch->kv_rows; v->dn_where = ch->dn_where; v->host_zero = ch->host_zero; v->cap = ch->cap;
}
static void q36cc_conv_load(Q36CChain *ch, int k) {
    Q36ccConv *v = &ch->conv[k];
    ch->rec = v->rec; ch->ring = v->ring; ch->kc = v->kc; ch->vc = v->vc; ch->kv_valid = v->kv_valid;
    ch->kv_rows = v->kv_rows; ch->dn_where = v->dn_where; ch->host_zero = v->host_zero; ch->cap = v->cap;
    ch->snap_valid = 0; ch->conv_cur = k;
}
static int q36cc_conv_down(Q36CChain *ch, Model *m, Q36ccConv *v) {
    if (v->slot < 0 || v->dn_where != Q36CC_DEV) return 1;
    Cfg *c = &m->c;
    float **rec = v->slot == g_q36_mux_cur ? m->DN_rec : g_q36_mux_seq[v->slot].DN_rec;
    float **cv = v->slot == g_q36_mux_cur ? m->DN_conv : g_q36_mux_seq[v->slot].DN_conv;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    int ok = rec && cv;
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (c->is_attn[i]) continue;
        ok = cc_read(v->rec[i], 0, rec[i], nr * sizeof(float)) && cc_read(v->ring[i], 0, cv[i], nc * sizeof(float));
    }
    if (ok) v->dn_where = Q36CC_BOTH;
    return ok;
}
static void q36cc_conv_free(Model *m, Q36ccConv *v) {
    for (int i = 0; i < m->c.n_layers; i++) {
        if (v->rec) cc_free(v->rec[i]); if (v->ring) cc_free(v->ring[i]);
        if (v->kc) cc_free(v->kc[i]); if (v->vc) cc_free(v->vc[i]);
    }
    free(v->rec); free(v->ring); free(v->kc); free(v->vc); free(v->kv_valid); free(v->kv_rows);
    memset(v, 0, sizeof *v); v->slot = -1;
}
#define Q36CC_CONV_ROWS 2048
static size_t q36cc_conv_kv_bytes(Model *m, int cap) {
    Cfg *c = &m->c; size_t b = 0;
    for (int i = 0; i < c->n_layers; i++)
        if (c->is_attn[i]) b += 2 * (size_t)c->kv_heads * (q36cc_ring(m, i) ? m->kv_ring : cap) * c->k_head_dim * q36cc_kves();
    return b;
}
static size_t q36cc_free_bytes(Q36CChain *ch) { size_t fr = 0, tot = 0; coli_cuda_mem_info(ch->dev, &fr, &tot); return fr; }
/* room for `bytes` more (and 512 MiB): the least recently used places' KV mirrors go first */
static int q36cc_conv_room(Q36CChain *ch, Model *m, int keep, size_t bytes) {
    const size_t margin = (size_t)512 << 20;
    while (q36cc_free_bytes(ch) < bytes + margin) {
        int k = -1;
        for (int j = 0; j < ch->nconv; j++) {
            Q36ccConv *v = &ch->conv[j];
            if (j == keep || v->used == ch->conv_clock || !v->cap || j == ch->conv_cur) continue;
            if (k < 0 || v->used < ch->conv[k].used) k = j;
        }
        if (k < 0) return 0;
        Q36ccConv *v = &ch->conv[k];
        for (int i = 0; i < m->c.n_layers; i++) {
            if (!m->c.is_attn[i] || q36cc_ring(m, i)) continue;
            cc_free(v->kc[i]); cc_free(v->vc[i]); v->kc[i] = v->vc[i] = NULL; v->kv_valid[i] = 0; v->kv_rows[i] = 0;
        }
        v->cap = 0;
    }
    return 1;
}
static int q36cc_conv_grow(Q36CChain *ch, Model *m, int k, int need) {
    Q36ccConv *v = &ch->conv[k];
    if (v->cap >= need) return 1;
    Cfg *c = &m->c; int KV = c->kv_heads, kvd = c->k_head_dim;
    int cap = v->cap ? 2 * v->cap : Q36CC_CONV_ROWS;
    if (cap < need) cap = (need + 1023) / 1024 * 1024;
    if (cap > m->kv_cap) cap = m->kv_cap;
    if (cap < need) return 0;
    if (!q36cc_conv_room(ch, m, k, q36cc_conv_kv_bytes(m, cap) - q36cc_conv_kv_bytes(m, v->cap))) return 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (!c->is_attn[i]) continue;
        int R = q36cc_ring(m, i);
        if (R && v->kc[i]) continue;   /* a ring keeps its rows */
        int rows = R ? R : cap;
        size_t bytes = (size_t)KV * rows * kvd * q36cc_kves();
        cc_free(v->kc[i]); cc_free(v->vc[i]);
        v->kc[i] = cc_buf(bytes, CC_DEV); v->vc[i] = cc_buf(bytes, CC_DEV);
        v->kv_valid[i] = 0; v->kv_rows[i] = rows;
        if (!v->kc[i] || !v->vc[i]) { v->cap = 0; return 0; }
        /* the tensor cores' attention reads whole blocks of a ring (q36cc_mirror) */
        if (R && (!cc_zero(v->kc[i], 0, bytes / sizeof(float)) || !cc_zero(v->vc[i], 0, bytes / sizeof(float)))) { v->cap = 0; return 0; }
    }
    v->cap = cap;
    return 1;
}
static int q36cc_conv_new(Q36CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->n_layers;
    const char *e = getenv("COLI_VK_CHAIN_MUX");
    int most = e && *e ? 1 + atoi(e) : g_q36_mux_slots;
    if (most > g_q36_mux_slots) most = g_q36_mux_slots;
    if (ch->conv_full || ch->nconv >= most) return -1;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float);
    size_t nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float), dn = 0;
    for (int i = 0; i < L; i++) if (!c->is_attn[i]) dn += nr + nc;
    int cap0 = m->kv_cap < Q36CC_CONV_ROWS ? m->kv_cap : Q36CC_CONV_ROWS;
    size_t need = dn + q36cc_conv_kv_bytes(m, cap0), avail = q36cc_free_bytes(ch);
    Q36ccConv v = {.slot = -1, .dn_where = Q36CC_HOST};
    int ok = need + ((size_t)512 << 20) <= avail;
    if (ok) {
        v.rec = calloc(L, sizeof *v.rec); v.ring = calloc(L, sizeof *v.ring);
        v.kc = calloc(L, sizeof *v.kc); v.vc = calloc(L, sizeof *v.vc);
        v.kv_valid = calloc(L, sizeof(int)); v.kv_rows = calloc(L, sizeof(int));
        ok = v.rec && v.ring && v.kc && v.vc && v.kv_valid && v.kv_rows;
    }
    for (int i = 0; i < L && ok; i++)
        if (!c->is_attn[i]) ok = (v.rec[i] = cc_buf(nr, CC_DEV)) && (v.ring[i] = cc_buf(nc, CC_DEV));
    if (ok) {
        ch->conv[ch->nconv] = v;
        if (q36cc_conv_grow(ch, m, ch->nconv, cap0)) return ch->nconv++;
        v = ch->conv[ch->nconv];
    }
    if (v.kv_valid) q36cc_conv_free(m, &v);
    ch->conv_full = 1;
    fprintf(stderr, "[chain] qwen36: the state of %d conversation%s stays on the device (one more takes %.2f GiB, %.2f GiB are "
                    "free); the others' go through the host when they come back\n",
            ch->nconv, ch->nconv == 1 ? "" : "s", need / 1073741824.0, avail / 1073741824.0);
    return -1;
}
static int q36cc_conv_get(Q36CChain *ch, Model *m, int slot, int evict) {
    for (int k = 0; k < ch->nconv; k++) if (ch->conv[k].slot == slot) return k;
    int k = -1;
    for (int j = 0; j < ch->nconv && k < 0; j++) if (ch->conv[j].slot < 0) k = j;
    if (k < 0) k = q36cc_conv_new(ch, m);
    if (k < 0 && evict) {
        for (int j = 0; j < ch->nconv; j++)
            if (ch->conv[j].used != ch->conv_clock && (k < 0 || ch->conv[j].used < ch->conv[k].used)) k = j;
        if (k < 0) return -1;
        if (k == ch->conv_cur) q36cc_conv_save(ch);
        if (!q36cc_conv_down(ch, m, &ch->conv[k])) return -1;
        ch->conv_handed++;
    }
    if (k < 0) return -1;
    Q36ccConv *v = &ch->conv[k];
    v->slot = slot; v->dn_where = Q36CC_HOST; v->host_zero = 0;
    for (int i = 0; i < m->c.n_layers; i++) v->kv_valid[i] = 0;
    return k;
}
static void q36cc_mux_init(Model *m) {
    Q36CChain *ch = q36cc_of(m);
    if (!q36cc_pool_ok(ch) || ch->conv || !(ch->conv = calloc((size_t)g_q36_mux_slots, sizeof *ch->conv))) return;
    for (int k = 0; k < g_q36_mux_slots; k++) ch->conv[k].slot = -1;
    ch->nconv = 1; ch->conv_cur = 0;
    q36cc_conv_save(ch);
    ch->conv[0].slot = g_q36_mux_cur;
}
static void q36cc_mux_lost(Model *m) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->conv) return;
    q36cc_conv_save(ch);
    for (int k = 0; k < ch->nconv; k++) {
        Q36ccConv *v = &ch->conv[k];
        if (v->slot < 0 || v->dn_where != Q36CC_DEV) continue;
        int held = v->slot == g_q36_mux_cur;
        if (!held) q36_seq_swap(m, &g_q36_mux_seq[v->slot]);
        q36cc_recover(m, m->kv_len);
        if (!held) q36_seq_swap(m, &g_q36_mux_seq[v->slot]);
        v->dn_where = Q36CC_HOST;
    }
}
/* The chain turned itself off (no device memory for a step's rows): every place's DeltaNet
 * state that is on the device only comes to its host arrays, which the host's path reads */
static void q36cc_mux_drop(Model *m) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->conv) return;
    q36cc_conv_save(ch);
    for (int k = 0; k < ch->nconv; k++)
        if (!q36cc_conv_down(ch, m, &ch->conv[k])) { q36cc_mux_lost(m); return; }
}
/* q36c_pool_off's twin */
static void q36cc_pool_off(Q36CChain *ch, Model *m) {
    q36cc_mux_drop(m);
    for (int k = 0; k < ch->nconv; k++) if (k != ch->conv_cur) q36cc_conv_free(m, &ch->conv[k]);
    free(ch->conv); ch->conv = NULL; ch->nconv = 0; ch->conv_cur = -1;
    fprintf(stderr, "[chain] qwen36: a prompt's KV mirror does not fit beside the other conversations' state: "
                    "each conversation's state goes through the host from here on\n");
}
static void q36cc_mux_leave(Model *m) {
    Q36CChain *ch = q36cc_of(m);
    if (ch && ch->conv && q36cc_pool_ok(ch)) { q36cc_conv_save(ch); return; }
    q36cc_cpu_step(m, m->kv_len);
}
static void q36cc_mux_enter(Model *m, int slot) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->ok) return;
    if (ch->conv && q36cc_pool_ok(ch)) {
        if (slot < 0) return;
        ch->conv_clock++;
        int k = q36cc_conv_get(ch, m, slot, 1);
        if (k >= 0) { q36cc_conv_load(ch, k); ch->conv[k].used = ch->conv_clock; return; }
        q36cc_recover(m, m->kv_len);
        q36cc_mux_lost(m);
        return;
    }
    ch->dn_where = Q36CC_HOST; ch->host_zero = 0; ch->snap_valid = 0;
    for (int i = 0; i < m->c.n_layers; i++) ch->kv_valid[i] = 0;
}
static void q36cc_mux_host(Model *m, const Q36Row *rows, int S, int after) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->conv) return;   /* also after the chain turned off: its places' states come home */
    q36cc_conv_save(ch);
    for (int s = 0; s < S; s++) {
        int slot = (int)(rows[s].seq - g_q36_mux_seq);
        for (int k = 0; k < ch->nconv; k++) {
            Q36ccConv *v = &ch->conv[k];
            if (v->slot != slot) continue;
            if (after) v->dn_where = Q36CC_HOST;
            else if (!q36cc_conv_down(ch, m, v)) { q36cc_mux_lost(m); return; }
        }
    }
    if (ch->conv_cur >= 0) q36cc_conv_load(ch, ch->conv_cur);
}
/* A rejected draft: the state after the verify's row `slot` is the device's copy in
 * that slot; `len` positions stand, and the KV watermark comes down to them. */
static void q36cc_rollback(Model *m, int slot, int len) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->ok) return;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid[i] > len) ch->kv_valid[i] = len;
    if (slot < 0 || slot >= ch->snap_valid || ch->dn_where != Q36CC_DEV) { ch->snap_valid = 0; return; }
    for (int i = 0; i < m->c.n_layers; i++) {
        if (m->c.is_attn[i]) continue;
        CcBuf *t = ch->rec[i]; ch->rec[i] = ch->rec_snap[slot][i]; ch->rec_snap[slot][i] = t;
        t = ch->ring[i]; ch->ring[i] = ch->ring_snap[slot][i]; ch->ring_snap[slot][i] = t;
    }
    ch->snap_valid = 0;
}
static int q36cc_spec_slots(Q36CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int L = c->n_layers;
    if (rows > Q36_SPEC_SNAPS) rows = Q36_SPEC_SNAPS;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int sl = ch->snap_slots; sl < rows; sl++) {
        if (!ch->rec_snap[sl] && !(ch->rec_snap[sl] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) return 0;
        if (!ch->ring_snap[sl] && !(ch->ring_snap[sl] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) return 0;
        for (int i = 0; i < L; i++) {
            if (c->is_attn[i]) continue;
            if (!ch->rec_snap[sl][i] && !(ch->rec_snap[sl][i] = cc_buf(nr * sizeof(float), CC_DEV))) return 0;
            if (!ch->ring_snap[sl][i] && !(ch->ring_snap[sl][i] = cc_buf(nc * sizeof(float), CC_DEV))) return 0;
        }
        ch->snap_slots = sl + 1;
    }
    return 1;
}
/* The uploads that make the device state the host's, as the step at pos_base needs it. */
static int q36cc_push_state(Q36CChain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->dn_where == Q36CC_HOST) {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        for (int i = 0; i < c->n_layers && ok; i++) {
            if (c->is_attn[i]) continue;
            if (ch->host_zero) ok = cc_zero(ch->rec[i], 0, nr) && cc_zero(ch->ring[i], 0, nc);
            else ok = cc_write(ch->rec[i], 0, m->DN_rec[i], nr * sizeof(float)) &&
                      cc_write(ch->ring[i], 0, m->DN_conv[i], nc * sizeof(float));
        }
        ch->dn_where = Q36CC_BOTH;
    }
    int kvd = c->k_head_dim, KV = c->kv_heads;
    for (int i = 0; i < c->n_layers && ok; i++) {
        /* a ring rewound (a pin, a reused prefix shorter than what the device holds): its window
         * before the step again from the host. A step uploads only the window's rows, so rows
         * below a later window may never have reached the device, and a rewind reads them */
        if (c->is_attn[i] && q36_kv_rows(m, i) != m->max_t && ch->kv_valid[i] > pos_base)
            ch->kv_valid[i] = pos_base - c->window + 1 > 0 ? pos_base - c->window + 1 : 0;
        if (!c->is_attn[i] || ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0, R = q36_kv_rows(m, i) != m->max_t ? m->kv_ring : 0;
        if (R) {   /* the host's ring (q36_kv_rows): the window's rows before pos_base, each run of its rows to the
                    * same rows of the device's ring, or to their positions in a mirror of the cache's length */
            const int dring = q36cc_ring(m, i) != 0;
            if (t0 < pos_base - c->window + 1) t0 = pos_base - c->window + 1;
            if (t0 < 0) t0 = 0;
            for (int t = t0; t < pos_base && ok; ) {
                int r = t % R, run = pos_base - t < R - r ? pos_base - t : R - r;
                for (int h = 0; h < KV && ok; h++) {
                    size_t src = ((size_t)h * R + r) * kvd, dst = dring ? src : ((size_t)h * ch->cap + t) * kvd;
                    ok = q36cc_kv_up(ch->kc[i], dst, m->K[i], (int64_t)src, (size_t)run * kvd) &&
                         q36cc_kv_up(ch->vc[i], dst, m->V[i], (int64_t)src, (size_t)run * kvd);
                }
                t += run;
            }
            ch->kv_valid[i] = pos_base;
            continue;
        }
        for (int h = 0; h < KV && ok; h++) {
            size_t src = ((size_t)h * m->max_t + t0) * kvd, dst = ((size_t)h * ch->cap + t0) * kvd;
            ok = q36cc_kv_up(ch->kc[i], dst, m->K[i], (int64_t)src, (size_t)n * kvd) &&
                 q36cc_kv_up(ch->vc[i], dst, m->V[i], (int64_t)src, (size_t)n * kvd);
        }
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}

/* ---- one layer's pieces ---------------------------------------------------------- */
static int q36cc_norm(CcBuf *x, size_t xo, CcBuf *w, size_t wo, CcBuf *y, size_t yo, int rows, int D, float eps) {
    CcNorm p = {rows, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, CC_NORM_ADD1, eps, 1.f};
    return cc_norm(x, w, y, &p);
}
/* The heads' output (ctx) through o_proj into tmp; Muse Spark normalises every head's
 * output first, then gates it by sigmoid(attn_gate(x)) (agb, recorded with q/k/v). */
static int q36cc_attn_out(Q36CChain *ch, Model *m, int i, int n) {
    Cfg *c = &m->c; int H = c->q_heads, hd = c->head_dim;
    if (c->muse) {
        CcNorm p = {n * H, hd, H, 0, H * hd, hd, 0, H * hd, hd, 0, 0, CC_NORM_NOW, c->eps, 1.f};
        CcEw g = {CC_EW_SIGMUL, n * H * hd, H * hd, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        if (!cc_norm(ch->ctx, ch->prm, ch->ctx, &p) || !cc_ew(ch->ctx, ch->ctx, ch->agb, NULL, NULL, &g)) return 0;
    }
    return cc_matmul(ch->t_o[i], ch->ctx, 0, ch->tmp, 0, n);
}
/* The attention's rows before its cache (q36c_attn_qkv's twin) */
static int q36cc_attn_qkv(Q36CChain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, half = c->rotary_dim / 2;
    if (c->nope_layer && c->nope_layer[i]) half = 0;   /* Flint: a layer without RoPE */
    int ok = cc_matmul(ch->t_q[i], ch->nrm, 0, ch->q, 0, n) &&
             cc_matmul(ch->t_k[i], ch->nrm, 0, ch->k, 0, n) &&
             cc_matmul(ch->t_v[i], ch->nrm, 0, ch->v, 0, n);
    if (ok && c->muse) ok = cc_matmul(ch->t_ag[i], ch->nrm, 0, ch->agb, 0, n);
    if (ok && l->qn) { CcNorm p = {n * H, hd, H, 0, qo, qdim, 0, qo, qdim, (int)ch->o_qn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
                       ok = cc_norm(ch->q, ch->prm, ch->q, &p); }
    if (ok && half) { CcRope p = {n * H, H, 0, qo, qdim, half, 0, 2 * half}; ok = cc_rope(ch->q, ch->csd, &p); }
    if (ok && l->kn) { CcNorm p = {n * KV, kvd, KV, 0, kvo, kvd, 0, kvo, kvd, (int)ch->o_kn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
                       ok = cc_norm(ch->k, ch->prm, ch->k, &p); }
    /* K and V as the host's cache keeps them (float16): the CPU's numbers, before anything
     * reads them. With a rotation, in the rope's launch (two launches a layer fewer: 0.8% of a
     * Qwen3.6 GGUF's decode on an RTX PRO 4000) */
    const int f16 = q36_kv16(), fused = f16 && half && cc_rope_f16_ready();
    if (ok && half) { CcRope p = {n * KV, KV, 0, kvo, kvd, half, 0, 2 * half};
                      ok = fused ? cc_rope_f16(ch->k, ch->v, ch->csd, &p, kvd) : cc_rope(ch->k, ch->csd, &p); }
    if (ok && f16 && !fused) {
        CcEw r = {CC_EW_F16, n * kvo, kvo, 0, 0, 0, 0, 0, 0, 0, 0, 1.f};
        ok = cc_ew(ch->k, ch->k, NULL, NULL, NULL, &r) && cc_ew(ch->v, ch->v, NULL, NULL, NULL, &r);
    }
    return ok;
}
static int q36cc_attention(Q36CChain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, gate_dim = qdim > hd ? qdim - hd : 0, ok = 1;
    const int win = c->win_layer && c->win_layer[i] ? c->window : 0;
    if (!q36cc_attn_qkv(ch, m, l, i, n)) return 0;
    /* the new rows into the device cache, and to the host's */
    size_t ko = (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    CcRegion *rg = malloc(sizeof *rg * (size_t)n * KV);
    if (!rg) return 0;
    const int R = q36cc_ring(m, i), rows = ch->kv_rows[i];   /* a ring: position t in row t % R */
    if (R && n > R - win + 1) {   /* a step past the ring's spare rows: its window gathered (q36c_attention says how) */
        free(rg);
        int b0 = pb - win + 1 > 0 ? pb - win + 1 : 0, nh = pb - b0, tot = nh + n;
        CcRegion *g = malloc(sizeof *g * (size_t)KV * (n > 2 ? n : 2));
        if (!g) return 0;
        int ng = 0;
        for (int h = 0; h < KV; h++)
            for (int t = b0; t < pb; ) {
                int r = t % R, run = pb - t < R - r ? pb - t : R - r;
                g[ng++] = (CcRegion){((size_t)h * tot + (t - b0)) * kvd, ((size_t)h * R + r) * kvd, (size_t)run * kvd};
                t += run;
            }
        if (ng) {   /* q36cc_kv_copy halves a float16 mirror's offsets: V's copy from the same rows */
            CcRegion *gv = malloc(sizeof *gv * (size_t)ng);
            ok = gv != NULL;
            if (ok) memcpy(gv, g, sizeof *gv * (size_t)ng);
            ok = ok && q36cc_kv_copy(ch->kg, ch->kc[i], g, ng) && q36cc_kv_copy(ch->vg, ch->vc[i], gv, ng);
            free(gv);
        }
        ng = 0;
        for (int s = 0; s < n; s++) for (int h = 0; h < KV; h++)
            g[ng++] = (CcRegion){((size_t)h * tot + nh + s) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
        ok = ok && q36cc_kv_put(ch->kg, ch->k, g, ng) && q36cc_kv_put(ch->vg, ch->v, g, ng);
        CcAttn a = {n, H, KV, hd, nh, tot, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0,
                    1.f / sqrtf((float)hd), 0, 0};
        CcAttnW w = {a, win, l->sinks != NULL, (int)ch->o_snk[i]};
        ok = ok && q36cc_attn_wc(ch->q, ch->kg, ch->vg, ch->ctx, ch->q, l->sinks ? ch->prm : NULL, &w, 0);
        ng = 0;   /* after the attention read the ring's rows: the step's last R into it */
        for (int s = n > R ? n - R : 0; s < n; s++) for (int h = 0; h < KV; h++)
            g[ng++] = (CcRegion){((size_t)h * R + (pb + s) % R) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
        ok = ok && q36cc_kv_put(ch->kc[i], ch->k, g, ng) && q36cc_kv_put(ch->vc[i], ch->v, g, ng) &&
             cc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && cc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
        free(g);
        return ok && q36cc_attn_out(ch, m, i, n);
    }
    for (int s = 0; s < n; s++) for (int h = 0; h < KV; h++)
        rg[s * KV + h] = (CcRegion){((size_t)h * rows + (R ? (pb + s) % R : pb + s)) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
    ok = q36cc_kv_put(ch->kc[i], ch->k, rg, n * KV) && q36cc_kv_put(ch->vc[i], ch->v, rg, n * KV) &&
         cc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && cc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    free(rg);
    CcAttn a = {n, H, KV, hd, pb, rows, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0,
                1.f / sqrtf((float)hd), 0, 0};
    if (win || l->sinks) {   /* Flint: a sliding window and a sink per head */
        CcAttnW w = {a, win, l->sinks != NULL, (int)ch->o_snk[i]};
        return ok && q36cc_attn_wc(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, l->sinks ? ch->prm : NULL, &w, R) &&
               q36cc_attn_out(ch, m, i, n);
    }
    return ok && q36cc_attn_c(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, &a) && q36cc_attn_out(ch, m, i, n);
}
/* ns: the chunk's rows a verify copies the state after (rows 0..ns-1, slots c0..): the
 * convolution and the recurrence split into one dispatch per copy, each ending on its
 * row, as the Vulkan chain does (the same bits). The in_proj is the tier's fused
 * qkv ++ z tensor: z sits at conv_dim inside each proj_dim-wide row of qkvz. */
static int q36cc_deltanet(Q36CChain *ch, Model *m, Layer *l, int i, int n, int c0, int ns) {
    Cfg *c = &m->c; (void)l;
    int vh = c->dn_vheads, CD = c->dn_conv_dim, vd = vh * c->dn_vdim, PD = CD + vd;
    int ok = cc_matmul(ch->t_dnp[i], ch->nrm, 0, ch->qkvz, 0, n) &&
             cc_matmul(ch->t_ab[i], ch->nrm, 0, ch->ab, 0, n);
    int segs = ns > 1 ? ns : 1;
    for (int r = 0; r < segs && ok; r++) {
        int s0 = ns > 1 ? r : 0, len = ns > 1 && r < ns - 1 ? 1 : n - s0, snap_row = ns ? 0 : -1;
        CcBuf *rs = ns ? ch->rec_snap[c0 + r][i] : NULL, *cs = ns ? ch->ring_snap[c0 + r][i] : NULL;
        CcDnConv cp = {len, CD, c->dn_convk, s0 * PD, PD, s0 * CD, CD, snap_row, 0, (int)ch->o_conv[i], 0, 0};
        CcDnRec rp = {len, vh, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, s0 * CD, CD, s0 * 2 * vh, 2 * vh,
                      vh + s0 * 2 * vh, 2 * vh, CD + s0 * PD, PD, s0 * vd, vd, snap_row,
                      (g_ssm_a_direct ? 2 : 0) | (g_dn_tiled ? 4 : 0), c->eps,   /* a GGUF's layout (qwen36.c) */
                      1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
        ok = cc_dnconv(ch->qkvz, ch->prm, ch->ring[i], ch->cv, cs, &cp) &&
             cc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->qkvz, ch->rec[i], ch->prm, ch->dny, rs, &rp);
    }
    return ok && cc_matmul(ch->t_dno[i], ch->dny, 0, ch->tmp, 0, n);
}
/* A multiplexed step's rows (ch->mx_rows from c0): q36c_attention_mx's twin */
static int q36cc_attention_mx(Q36CChain *ch, Model *m, Layer *l, int i, int n, int c0) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, gate_dim = qdim > hd ? qdim - hd : 0;
    const int win = c->win_layer && c->win_layer[i] ? c->window : 0, R = q36cc_ring(m, i);
    if (!q36cc_attn_qkv(ch, m, l, i, n)) return 0;
    size_t ko = (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    CcRegion *rg = malloc(sizeof *rg * (size_t)KV);
    int ok = rg && cc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) &&
             cc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    for (int s = 0; s < n && ok; s++) {
        Q36ccConv *v = ch->mx_conv[c0 + s];
        int pos = ch->mx_rows[c0 + s].pos, rows = v->kv_rows[i];
        for (int h = 0; h < KV; h++)
            rg[h] = (CcRegion){((size_t)h * rows + (R ? pos % R : pos)) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
        ok = q36cc_kv_put(v->kc[i], ch->k, rg, KV) && q36cc_kv_put(v->vc[i], ch->v, rg, KV);
        CcAttn a = {1, H, KV, hd, pos, rows, s * qo, qo, qdim, s * qo + hd, qo, qdim, gate_dim > 0, s * H * hd, H * hd, 0, 0,
                    1.f / sqrtf((float)hd), 0, 0};
        if (win || l->sinks) {
            CcAttnW w = {a, win, l->sinks != NULL, (int)ch->o_snk[i]};
            ok = ok && q36cc_attn_wc(ch->q, v->kc[i], v->vc[i], ch->ctx, ch->q, l->sinks ? ch->prm : NULL, &w, R);
        } else ok = ok && q36cc_attn_c(ch->q, v->kc[i], v->vc[i], ch->ctx, ch->q, &a);
    }
    free(rg);
    return ok && q36cc_attn_out(ch, m, i, n);
}
static int q36cc_deltanet_mx(Q36CChain *ch, Model *m, Layer *l, int i, int n, int c0) {
    Cfg *c = &m->c; (void)l;
    int vh = c->dn_vheads, CD = c->dn_conv_dim, vd = vh * c->dn_vdim, PD = CD + vd;
    int ok = cc_matmul(ch->t_dnp[i], ch->nrm, 0, ch->qkvz, 0, n) &&
             cc_matmul(ch->t_ab[i], ch->nrm, 0, ch->ab, 0, n);
    for (int s = 0; s < n && ok; s++) {
        Q36ccConv *v = ch->mx_conv[c0 + s];
        CcDnConv cp = {1, CD, c->dn_convk, s * PD, PD, s * CD, CD, -1, 0, (int)ch->o_conv[i], 0, 0};
        CcDnRec rp = {1, vh, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, s * CD, CD, s * 2 * vh, 2 * vh,
                      vh + s * 2 * vh, 2 * vh, CD + s * PD, PD, s * vd, vd, -1,
                      (g_ssm_a_direct ? 2 : 0) | (g_dn_tiled ? 4 : 0), c->eps,
                      1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
        ok = cc_dnconv(ch->qkvz, ch->prm, v->ring[i], ch->cv, NULL, &cp) &&
             cc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->qkvz, v->rec[i], ch->prm, ch->dny, NULL, &rp);
    }
    return ok && cc_matmul(ch->t_dno[i], ch->dny, 0, ch->tmp, 0, n);
}
/* A multiplexed step's places before its frame (q36c_mux_push's twin, the CUDA ring rules) */
static int q36cc_mux_push(Q36CChain *ch, Model *m, int c0, int n) {
    Cfg *c = &m->c; int kvd = c->k_head_dim, KV = c->kv_heads, ok = 1;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int s = 0; s < n && ok; s++) {
        Q36ccConv *v = ch->mx_conv[c0 + s];
        const Q36Seq *q = ch->mx_rows[c0 + s].seq;
        int pos = ch->mx_rows[c0 + s].pos;
        if (v->dn_where == Q36CC_HOST) {
            for (int i = 0; i < c->n_layers && ok; i++) {
                if (c->is_attn[i]) continue;
                if (v->host_zero) ok = cc_zero(v->rec[i], 0, nr) && cc_zero(v->ring[i], 0, nc);
                else ok = cc_write(v->rec[i], 0, q->DN_rec[i], nr * sizeof(float)) &&
                          cc_write(v->ring[i], 0, q->DN_conv[i], nc * sizeof(float));
            }
            v->dn_where = Q36CC_BOTH;
        }
        for (int i = 0; i < c->n_layers && ok; i++) {
            if (c->is_attn[i] && q36_kv_rows(m, i) != m->max_t && v->kv_valid[i] > pos)   /* a ring rewound */
                v->kv_valid[i] = pos - c->window + 1 > 0 ? pos - c->window + 1 : 0;
            if (!c->is_attn[i] || v->kv_valid[i] >= pos) continue;
            int t0 = v->kv_valid[i], R = q36_kv_rows(m, i) != m->max_t ? m->kv_ring : 0;
            if (R) {
                const int dring = q36cc_ring(m, i) != 0;
                if (t0 < pos - c->window + 1) t0 = pos - c->window + 1;
                if (t0 < 0) t0 = 0;
                for (int t = t0; t < pos && ok; ) {
                    int r = t % R, run = pos - t < R - r ? pos - t : R - r;
                    for (int h = 0; h < KV && ok; h++) {
                        size_t src = ((size_t)h * R + r) * kvd, dst = dring ? src : ((size_t)h * v->cap + t) * kvd;
                        ok = q36cc_kv_up(v->kc[i], dst, q->K[i], (int64_t)src, (size_t)run * kvd) &&
                             q36cc_kv_up(v->vc[i], dst, q->V[i], (int64_t)src, (size_t)run * kvd);
                    }
                    t += run;
                }
            } else
                for (int h = 0; h < KV && ok; h++) {
                    size_t src = ((size_t)h * m->max_t + t0) * kvd, dst = ((size_t)h * v->cap + t0) * kvd;
                    ok = q36cc_kv_up(v->kc[i], dst, q->K[i], (int64_t)src, (size_t)(pos - t0) * kvd) &&
                         q36cc_kv_up(v->vc[i], dst, q->V[i], (int64_t)src, (size_t)(pos - t0) * kvd);
                }
            v->kv_valid[i] = pos;
        }
    }
    return ok;
}
static int q36cc_shared(Q36CChain *ch, Model *m, Layer *l, int i, int n) {
    int ok = 1, SI = l->sh_I; (void)m;   /* shared_inter, or a Flint dense layer's MLP width */
    if (SI > 0) {
        CcEw p = {CC_EW_SWIGLU, n * SI, SI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = cc_matmul(ch->t_shg[i], ch->h2, 0, ch->gs, 0, n) &&
             cc_matmul(ch->t_shu[i], ch->h2, 0, ch->us, 0, n) &&
             cc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &p) &&
             cc_matmul(ch->t_shd[i], ch->hs, 0, ch->ds, 0, n);
    }
    if (ok && ch->t_sg[i] && !(ch->rgf[i] && n <= Q36CC_RG_ROWS)) ok = cc_matmul(ch->t_sg[i], ch->h2, 0, ch->sgd, 0, n);
    return ok;
}
/* x += routed + gate * shared, as moe() leaves `out` and layers_forward_range adds it */
static int q36cc_combine(Q36CChain *ch, Model *m, int i, int n, CcBuf *routed) {
    Cfg *c = &m->c;
    if (c->muse) {   /* Muse Spark: the routed sum back to the hidden width, normalised, through the residual gate */
        const int D = c->hidden;
        CcNorm p = {n, D, 1, 0, D, D, 0, D, D, (int)ch->o_pf[i], 0, CC_NORM_PRESCALE, c->post_eps, 1.f};
        CcEw g = {CC_EW_RESGATE, n * D, D, 1, 0, 1, 0, 0, 0, (int)ch->o_rf[i] + D, (int)ch->o_rf[i], 1.f};
        return cc_matmul(ch->t_lu[i], routed, 0, ch->tmp, 0, n) &&
               cc_norm(ch->tmp, ch->prm, ch->tmp, &p) && cc_ew(ch->x, ch->x, ch->tmp, ch->prm, ch->prm, &g);
    }
    const int routed_l = c->n_experts > 0 && !q36_dense_l(c, i), SI = m->L[i].sh_I;   /* a Flint dense layer routes nothing */
    int flags = (routed_l ? 1 : 0) | (SI > 0 ? 2 : 0) | (m->L[i].sh_gate && SI > 0 ? 4 : 0);
    const int rg = ch->rgf[i] && n <= Q36CC_RG_ROWS, E1 = c->n_experts + 1;   /* the gate after the router's logits in lg */
    CcEw p = {CC_EW_COMBINE, n * c->hidden, c->hidden, 1, flags, rg ? E1 : 1, 0, 0, 0, 0, rg ? c->n_experts : 0, 1.f};
    return cc_ew(ch->x, ch->x, routed, ch->ds, rg ? ch->lg : ch->sgd, &p);
}
/* COLI_CUDA_CHAIN_FAULT=n: the n-th frame fails (for the tests) */
static int q36cc_frame_begin(Q36CChain *ch) {
    ch->frames++;
    if (ch->fault_at && ch->frames == ch->fault_at) { fprintf(stderr, "[chain] qwen36: fault injected at frame %llu\n", ch->frames); return 0; }
    return cc_begin();
}

/* ---- the routed experts in the frame (cuda_chain.h: cc_route, cc_moe) -----------------
 * A step of up to Q36CC_DM_ROWS rows (decode, a verify) routes on the device and runs the
 * tier's residents on the chain's device there, by address from a table of the tier's
 * tensors: built again when the tier's epoch moved, the tier held for the forward (no
 * swap writes into a resident while it runs). A layer whose experts are all in the table
 * has no host step at all. A layer with some elsewhere (hybrid) signals the host once its
 * routes and rows are down; the host computes the slots the table lacks as the tier's
 * misses always were (qt_cpu_misses: the RAM cache, the disk) while the device computes
 * the rest, and answers in an UP buffer the device waits for before the sum (the host's
 * misses, then the device's slots in rank order: qt_take's order). The host serves a
 * hybrid layer after recording the next one, so the device has that layer queued when the
 * answer lands. The books the host's step kept (the tier's heat, hits and LFRU tick, the
 * resident collection, the hot-pin counts) follow from the ids after the frame, with the
 * K/V rows. Off with cache routing or the agreement meter, a gate bias, groups the kernel
 * does not keep, a tier on more than one device, a format the kernels do not take, or
 * COLI_CUDA_CHAIN_MOE=0. */
#define Q36CC_DM_ROWS 8
static int q36cc_dm_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN_MOE"); on = !(e && *e == '0'); }
    return on;
}
static int q36cc_dm_able(Q36CChain *ch, Model *m) {
    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, ng = c->n_group > 1 && c->n_group <= E ? c->n_group : 1;
    if (!q36cc_dm_wanted() || !cc_moe_ready() || E <= 0 || E > CC_ROUTE_EMAX || K < 1 || K > 32 || g_cache_route || g_route_agree ||
        c->has_bias || !qt_experts_on() || qt_chain_devices() != 1) return 0;
    if (ng > 1 && (ng > 64 || E % ng || c->topk_group < 1 || c->topk_group > ng)) return 0;
    if (!qt_chain_format(&ch->dm_fmt, &ch->dm_gs)) return 0;
    if (c->hidden % 32 || c->inter % 32 || c->hidden > CC_MOE_DIM_MAX || c->inter > CC_MOE_DIM_MAX ||
        (ch->dm_fmt == 4 && (ch->dm_gs < 16 || ch->dm_gs % 16))) return 0;
    for (int i = 0; ch->dm_fmt >= 100 && i < c->n_layers; i++) {   /* a GGUF's experts: every layer's types, cc_moe's */
        int fg, fd;
        if (!qt_chain_layer_format(i, &fg, &fd) || !cc_moe_block_type(fg - 100) || !cc_moe_block_type(fd - 100)) return 0;
    }
    return 1;
}
static void q36cc_kv_down(Q36CChain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c; int kvo = c->kv_heads * c->k_head_dim;
    if (!c->is_attn[i]) return;
    const float *kv = (const float *)cc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    const int R = q36_kv_rows(m, i) != m->max_t ? m->kv_ring : 0;   /* a sliding layer's ring on the host */
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
/* A prompt chunk routes nearly every expert of a layer, and the host's step costs per row
 * where a copy over the bus costs per expert: more than Q36CC_STAGE_ROWS rows copy a layer's
 * experts in the tier's host pool (its RAM cache, page-locked) into dm_stage and run them
 * with the residents, the copy of the next such layer on the copy stream while the device
 * works through the layers before it. dm_stage holds the most experts any layer has there
 * (one layer's: the copy is the longer of the two, a second buffer would buy little); the
 * table points each such expert at its place there (planar, as the RAM holds it). 0: no
 * VRAM for it. */
#define Q36CC_STAGE_ROWS 32
/* The host pool follows the residents (qt_chain_host_follow) before a prompt chunk of
 * Q36CC_FOLLOW_ROWS rows or more: the experts that left the card join it, those that came in
 * leave it, or a layer holding one that left goes to the host's step whole. A smaller chunk
 * leaves it behind for the next large one. On the Colab T4 high-RAM VM of the measures
 * (docs/experiments/qwen36-cuda-host-pool-follow-2026-10-26) a follow costs about 0.3 ms an
 * expert moved. On another VM of the same shape an earlier version, which followed before
 * every chunk of more than 32 rows, moved 317 + 317 experts in 193 ms before a 36-row chunk
 * and saved about 50 ms of it, while a 348-row chunk after 50 + 50 moves saved 1.1 s: through
 * those two points the follow pays from about 80 rows, and at 128 the copies save about
 * twice the 193 ms (threshold.txt there). */
#define Q36CC_FOLLOW_ROWS 128
/* Promotions (qt_chain_promote): after a decode-sized step, up to this many of its misses move
 * into the buffers of the coldest residents by this conversation's routes. The chain copies
 * each newcomer's RAM bytes into a page-locked staging slot and from there into the victim's
 * device buffers on its copy stream, while the next steps run; once the copies landed the tier
 * hands the buffers to the newcomer (qt_chain_promote_done) and the table takes the change
 * entry by entry (qt_chain_changes). The victim computes on the CPU from the next step on. The
 * warm start's set comes from the history, and a conversation's experts drift from it.
 * COLI_CUDA_CHAIN_PROMOTE=0 keeps the warm start's set, =n sets the rate (qwen38's chain reads
 * the same knob). The brake: where the last Q36CC_PROMO_WIN decode steps found at least
 * COLI_CUDA_CHAIN_PROMOTE_HITS percent (75) of their routes on the card, the step promotes
 * nothing -- a conversation the history fits (80-82 % on an RTX 3070) pays for every swap (a
 * promoted expert displaces a resident the history still routes to) and gains little; at 80 the
 * brake was marginal for it. It holds only where the warm start's set came from a history
 * (HEAT_FILE). Without one the set is the experts in index order, no conversation's, and the
 * brake starts off: on a T4 (64 % of the experts resident) it held a story at 78-79 % hits,
 * where the promotions reach 86-94 % and a token costs 7-17 % less; on an RTX 3070 about 2 %
 * either way. The variable, when set, wins; 0 brakes never. */
#define Q36CC_PROMOTE 4
#define Q36CC_PROMO_HITS 75
#define QT_CHG_LIST 512   /* residency changes read at once (the tier's ring holds as many) */
static int q36cc_promote_rate(void) { const char *e = getenv("COLI_CUDA_CHAIN_PROMOTE"); return e && *e ? atoi(e) : Q36CC_PROMOTE; }
static int q36cc_promote_hits(void) {
    const char *e = getenv("COLI_CUDA_CHAIN_PROMOTE_HITS");
    int h = e && *e ? atoi(e) : qt_chain_history() ? Q36CC_PROMO_HITS : 0;
    return h < 0 ? 0 : h > 100 ? 100 : h;
}
/* entry (i, e) of the host's table from the tier: 1 resident on this device */
static int q36cc_dm_entry(Q36CChain *ch, int E, int i, int e) {
    CcExpert *x = &ch->dm_host[(size_t)i * E + e];
    ColiCudaTensor *g, *u, *d;
    memset(x, 0, sizeof *x);
    int on = qt_chain_resident(i, e, ch->dev, &g, &u, &d) && cc_tensor_dev(g, &x->g, &x->gs, NULL) &&
             cc_tensor_dev(u, &x->u, &x->us, NULL) && cc_tensor_dev(d, &x->d, &x->ds, NULL);
    if (!on) memset(x, 0, sizeof *x);
    return on;
}
/* the pairs whose copies landed: the tier hands the buffers over; then this step's promotions,
 * unless the brake holds: the last Q36CC_PROMO_WIN decode steps (this one among them, rows
 * rows each) found promo_hits percent or more of their routes on the card */
static void q36cc_promote(Q36CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, I = c->inter, K = c->topk;
    if (ch->promo_rate <= 0) return;
    double t0 = tm_now();
    unsigned route = (unsigned)((ch->dm_steps - ch->pr_steps0) * (unsigned long long)rows * (unsigned long long)K), miss = (unsigned)(ch->dm_pairs - ch->pr_pairs0);
    ch->pr_steps0 = ch->dm_steps; ch->pr_pairs0 = ch->dm_pairs;
    ch->pr_wroute[ch->pr_wi] = route; ch->pr_wmiss[ch->pr_wi] = miss;
    ch->pr_wi = (ch->pr_wi + 1) % Q36CC_PROMO_WIN; if (ch->pr_wn < Q36CC_PROMO_WIN) ch->pr_wn++;
    int w = 0;
    for (int p = 0; p < ch->pr_npend; p++) {
        if (!cc_host_up_done(ch->pr_pend[p].tk)) { ch->pr_pend[w++] = ch->pr_pend[p]; continue; }
        qt_chain_promote_done(ch->pr_pend[p].ci, ch->pr_pend[p].vi, 1);
        ch->pr_busy[ch->pr_pend[p].slot] = 0; ch->dm_landed++;
    }
    ch->pr_npend = w;
    int room = Q36CC_PROMO_SLOTS - ch->pr_npend, max = room < ch->promo_rate ? room : ch->promo_rate;
    if (max > 0 && ch->promo_hits > 0) {
        unsigned long long wr = 0, wm = 0;
        for (int j = 0; j < ch->pr_wn; j++) { wr += ch->pr_wroute[j]; wm += ch->pr_wmiss[j]; }
        if (wr > 0 && (wr - wm) * 100 >= (unsigned long long)ch->promo_hits * wr) { max = 0; ch->pr_braked++; }
    }
    if (max <= 0) { qt_chain_promote(0, NULL); ch->promo_ms += tm_now() - t0; return; }   /* the step's misses dropped: the tier's list is a step's */
    if (!ch->pr_buf) {   /* the staging: three matrices and their scales an expert, 256-byte aligned */
        size_t mb = ch->dm_fmt == 2 || ch->dm_fmt == 4 ? (size_t)D * I / 2 : (size_t)D * I, sb = (size_t)(2 * I + D) * sizeof(float);
        if (ch->dm_fmt == 4) sb = (size_t)(2 * I * ((D + ch->dm_gs - 1) / ch->dm_gs) + D * ((I + ch->dm_gs - 1) / ch->dm_gs)) * sizeof(float);
        ch->pr_stride = (3 * mb + sb + 255) / 256 * 256;
        if (!cc_host_up_ready() || !cc_reserve(&ch->pr_buf, ch->pr_stride * Q36CC_PROMO_SLOTS, CC_UP)) {
            fprintf(stderr, "[chain] qwen36: no page-locked staging for promotions; the warm start's set stays\n");
            ch->promo_rate = 0; return;
        }
    }
    QtPromo pr[Q36CC_PROMO_SLOTS];
    int n = qt_chain_promote(max, pr);
    for (int k = 0; k < n; k++) {
        QtPromo *q = &pr[k];
        int slot = -1;
        for (int j = 0; j < Q36CC_PROMO_SLOTS; j++) if (!ch->pr_busy[j]) { slot = j; break; }
        const void *wg, *wu, *wd; const float *sg, *su, *sd;
        if (slot < 0 || 3 * q->mb + 2 * q->sb_gu + q->sb_d > ch->pr_stride ||
            !cc_tensor_dev(q->tg, &wg, &sg, NULL) || !cc_tensor_dev(q->tu, &wu, &su, NULL) || !cc_tensor_dev(q->td, &wd, &sd, NULL)) {
            qt_chain_promote_done(q->ci, q->vi, 0); continue;
        }
        char *st = (char *)cc_ptr(ch->pr_buf) + (size_t)slot * ch->pr_stride, *sc = st + 3 * q->mb;
        qt_chain_promote_stage(st, q);
        unsigned tk = 0;
        if (!cc_host_up((void *)wg, st, q->mb, &tk) || !cc_host_up((void *)wu, st + q->mb, q->mb, &tk) || !cc_host_up((void *)wd, st + 2 * q->mb, q->mb, &tk) ||
            !cc_host_up((void *)sg, sc, q->sb_gu, &tk) || !cc_host_up((void *)su, sc + q->sb_gu, q->sb_gu, &tk) || !cc_host_up((void *)sd, sc + 2 * q->sb_gu, q->sb_d, &tk)) {
            qt_chain_promote_done(q->ci, q->vi, 0);   /* the victim's buffers hold neither now */
            continue;
        }
        ch->pr_busy[slot] = 1;
        ch->pr_pend[ch->pr_npend].ci = q->ci; ch->pr_pend[ch->pr_npend].vi = q->vi; ch->pr_pend[ch->pr_npend].slot = slot;
        ch->pr_pend[ch->pr_npend].tk = tk; ch->pr_npend++;
        ch->dm_promoted++;
    }
    ch->promo_ms += tm_now() - t0;
}
/* at the chain's end: the copies in flight finished first (cc_finish), the pairs settled */
static void q36cc_promote_settle(Q36CChain *ch) {
    for (int p = 0; p < ch->pr_npend; p++) { qt_chain_promote_done(ch->pr_pend[p].ci, ch->pr_pend[p].vi, cc_host_up_done(ch->pr_pend[p].tk)); ch->pr_busy[ch->pr_pend[p].slot] = 0; }
    ch->pr_npend = 0;
}
/* The staging buffer for a layer's host-pool experts. With the residents spread over the layers hmax (the most
 * experts one layer has off the card) starts below the layer's 256 and every promotion can raise it; cc_reserve
 * makes the new buffer before it frees the old one, so each growth wanted both at once. On an 8 GB card the
 * second allocation failed in the middle of a conversation and the staging stayed off for the process (the
 * RTX 3070 agent session in docs/experiments). A bigger buffer takes the old one's place: free it first. */
static int q36cc_stage_reserve(CcBuf **stage, size_t bytes) {
    if (*stage && cc_bytes(*stage) < bytes) { cc_free(*stage); *stage = NULL; }
    return cc_reserve(stage, bytes, CC_DEV);
}
/* A GGUF layer's expert in the staging: its gate, up and down rows as the file has them, which the
 * device reads in place where a row is whole words (bd_rows_copy pads the others); mg a gate's (or
 * an up's) bytes, md a down's. 0: rows of other widths, the layer's experts the host's. */
static int q36cc_gguf_dims(Model *m, int i, size_t *mg, size_t *md) {
    if (!m->gqx || i < 0 || i >= m->c.n_layers) return 0;
    const int D = q36_eh(&m->c), I = m->c.inter;   /* the file's types, the tier's (qt_init_gguf) */
    int64_t rg = gq_row_bytes(m->gqx[i].tg, D), rd = gq_row_bytes(m->gqx[i].td, I);
    if (rg <= 0 || rd <= 0 || rg % 4 || rd % 4) return 0;
    *mg = (size_t)rg * I; *md = (size_t)rd * D;
    return 1;
}
/* the staging's VRAM back (the mirrors or the step's buffers need it): a prompt chunk's experts in
 * the host pool go to the host's step from here on; 1 when there was one */
static int q36cc_stage_drop(Q36CChain *ch) {
    if (!ch->dm_stage) return 0;
    cc_free(ch->dm_stage); ch->dm_stage = NULL; ch->dm_stage_off = 1;
    fprintf(stderr, "[chain] qwen36: the staging of the host pool's experts gives its device memory back; a prompt chunk's go "
                    "to the host's step from here on\n");
    return 1;
}
/* the mirrors' bytes on the device now */
static size_t q36cc_mirror_held(Q36CChain *ch, Model *m) {
    size_t b = 0;
    for (int i = 0; i < m->c.n_layers; i++) if (m->c.is_attn[i]) b += cc_bytes(ch->kc[i]) + cc_bytes(ch->vc[i]);
    return b;
}
/* The staged table of a GGUF's experts (q36cc_stage_table): each layer's experts in the host pool
 * at slots of the largest layer's three matrices, 256-byte aligned each; one copy an expert where
 * its gate and up end on 256 bytes (its rows back to back as the RAM holds them), else three. */
static int q36cc_stage_table_gguf(Q36CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->n_layers, E = c->n_experts;
    size_t ne = (size_t)L * E, stride = 0;
    int hmax = 0, tot = 0;
    for (int i = 0; i < L; i++) {
        size_t mg, md;
        if (q36_dense_l(c, i) || !q36cc_gguf_dims(m, i, &mg, &md)) continue;
        size_t st = 2 * ((mg + 255) / 256 * 256) + (md + 255) / 256 * 256;
        if (st > stride) stride = st;
        int h = 0;
        for (int e = 0; e < E; e++) {
            const void *w; const float *sc; size_t a, b;
            if (!ch->dm_res[(size_t)i * E + e] && qt_chain_host(i, e, &w, &a, &sc, &b) && a == 2 * mg + md) h++;
        }
        if (h > hmax) hmax = h;
        tot += h;
    }
    if (!ch->dm_hostb) {
        ch->dm_hostb = malloc(ne * sizeof(CcExpert));
        ch->dm_partb = calloc(L, sizeof(int)); ch->dm_fro = calloc(L, sizeof(int)); ch->dm_frn = calloc(L, sizeof(int));
        if (!ch->dm_hostb || !ch->dm_partb || !ch->dm_fro || !ch->dm_frn) return 0;
    }
    CcFetch *fr = realloc(ch->dm_fr, (size_t)(3 * tot + 1) * sizeof(CcFetch));
    if (!fr) return 0;
    ch->dm_fr = fr;
    /* the room the tier left the mirrors at the context's length stays free (a window's or sinks'
     * model's, q36cc_mirror_bytes): another model's mirrors take the staging's memory back when
     * they grow (q36cc_stage_drop). Kept for every model, Qwen3.6 UD-Q4_K_M with 7 GiB free kept
     * 1.3 GB that no one had set aside and found none to stage in: 97 tok/s for any prompt. */
    size_t fb = 0, tb = 0, full = q36cc_mirror_bytes(m, qwen36_max_ctx()), held = q36cc_mirror_held(ch, m);
    size_t left = full > held ? full - held : 0, need = (size_t)(hmax ? hmax : 1) * stride;
    if (!ch->dm_stage && coli_cuda_mem_info(ch->dev, &fb, &tb) && fb < need + left + ((size_t)256 << 20)) return 0;
    if (!stride || !q36cc_stage_reserve(&ch->dm_stage, (size_t)(hmax ? hmax : 1) * stride) ||
        !cc_reserve(&ch->dm_tabb, ne * sizeof(CcExpert), CC_DEV)) return 0;
    const char *base = (const char *)cc_dev_addr(ch->dm_stage);
    if (!base) return 0;
    int k = 0;
    for (int i = 0; i < L; i++) {
        size_t mg = 0, md = 0;
        const int dims = !q36_dense_l(c, i) && q36cc_gguf_dims(m, i, &mg, &md);
        const size_t ag = (mg + 255) / 256 * 256;
        const int one = mg % 256 == 0;   /* gate and up end on 256 bytes: the rows as they are, one copy */
        int h = 0, miss = 0;
        ch->dm_fro[i] = k;
        for (int e = 0; e < E; e++) {
            size_t j = (size_t)i * E + e;
            CcExpert *x = &ch->dm_hostb[j];
            const void *w; const float *sc; size_t a, b;
            *x = ch->dm_host[j];
            if (ch->dm_res[j]) continue;
            memset(x, 0, sizeof *x);
            if (!dims || h >= hmax || k + 3 > 3 * tot || !qt_chain_host(i, e, &w, &a, &sc, &b) || a != 2 * mg + md) { miss = 1; continue; }
            const size_t at = (size_t)h * stride;
            const char *g = base + at;
            x->g = g; x->u = g + ag; x->d = g + 2 * ag;
            if (one) fr[k++] = (CcFetch){w, at, 2 * mg + md};
            else {
                fr[k++] = (CcFetch){w, at, mg};
                fr[k++] = (CcFetch){(const char *)w + mg, at + ag, mg};
                fr[k++] = (CcFetch){(const char *)w + 2 * mg, at + 2 * ag, md};
            }
            h++;
        }
        ch->dm_frn[i] = k - ch->dm_fro[i]; ch->dm_partb[i] = miss;
    }
    return cc_begin() && cc_write(ch->dm_tabb, 0, ch->dm_hostb, ne * sizeof(CcExpert)) && cc_submit(1);
}
static int q36cc_stage_table(Q36CChain *ch, Model *m) {
    if (ch->dm_fmt >= 100) return q36cc_stage_table_gguf(ch, m);
    Cfg *c = &m->c; int L = c->n_layers, E = c->n_experts, D = c->hidden, I = c->inter;
    size_t ne = (size_t)L * E, mb = (size_t)D * I / 2, nsc = (size_t)I * (D / ch->dm_gs), wb = 3 * mb, sb = 3 * nsc * sizeof(float);
    size_t wa = (wb + 255) / 256 * 256, stride = wa + (sb + 255) / 256 * 256;
    int hmax = 0, tot = 0;
    for (int i = 0; i < L; i++) {
        int h = 0;
        for (int e = 0; e < E; e++) {
            const void *w; const float *sc; size_t a, b;
            if (ch->dm_res[(size_t)i * E + e] || !qt_chain_host(i, e, &w, &a, &sc, &b)) continue;
            if (a != wb || b != sb) return 0;   /* not the layout the table describes */
            h++;
        }
        if (h > hmax) hmax = h;
        tot += h;
    }
    if (!ch->dm_hostb) {
        ch->dm_hostb = malloc(ne * sizeof(CcExpert));
        ch->dm_partb = calloc(L, sizeof(int)); ch->dm_fro = calloc(L, sizeof(int)); ch->dm_frn = calloc(L, sizeof(int));
        if (!ch->dm_hostb || !ch->dm_partb || !ch->dm_fro || !ch->dm_frn) return 0;
    }
    CcFetch *fr = realloc(ch->dm_fr, (size_t)(2 * tot + 1) * sizeof(CcFetch));
    if (!fr) return 0;
    ch->dm_fr = fr;
    if (!q36cc_stage_reserve(&ch->dm_stage, (size_t)(hmax ? hmax : 1) * stride) || !cc_reserve(&ch->dm_tabb, ne * sizeof(CcExpert), CC_DEV))
        return 0;
    const char *base = (const char *)cc_dev_addr(ch->dm_stage);
    if (!base) return 0;
    int k = 0;
    for (int i = 0; i < L; i++) {
        int h = 0, miss = 0;
        ch->dm_fro[i] = k;
        for (int e = 0; e < E; e++) {
            size_t j = (size_t)i * E + e;
            CcExpert *x = &ch->dm_hostb[j];
            const void *w; const float *sc; size_t a, b;
            *x = ch->dm_host[j];
            if (ch->dm_res[j]) continue;
            memset(x, 0, sizeof *x);
            if (h >= hmax || k + 2 > 2 * tot || !qt_chain_host(i, e, &w, &a, &sc, &b) || a != wb || b != sb) { miss = 1; continue; }
            const char *g = base + (size_t)h * stride;
            x->g = g; x->u = g + mb; x->d = g + 2 * mb;
            x->gs = (const float *)(g + wa); x->us = x->gs + nsc; x->ds = x->gs + 2 * nsc;
            x->planar = 1;
            fr[k++] = (CcFetch){w, (size_t)h * stride, wb};
            fr[k++] = (CcFetch){sc, (size_t)h * stride + wa, sb};
            h++;
        }
        ch->dm_frn[i] = k - ch->dm_fro[i]; ch->dm_partb[i] = miss;
    }
    return cc_begin() && cc_write(ch->dm_tabb, 0, ch->dm_hostb, ne * sizeof(CcExpert)) && cc_submit(1);
}
/* before a prompt chunk of n rows that stages: on one of Q36CC_FOLLOW_ROWS rows or more the
 * pool follows the residents, and a change stales the staged table */
static void q36cc_follow(Q36CChain *ch, int n) {
    if (n < Q36CC_FOLLOW_ROWS) return;
    double th = tm_now(); int hj, hl;
    if (qt_chain_host_follow(&hj, &hl)) { ch->hp_follows++; ch->hp_joined += (unsigned)hj; ch->hp_left += (unsigned)hl; ch->dm_stage_dirty = 1; }
    ch->hp_ms += tm_now() - th;
}
/* the step's buffers, and the table when the tier's residents moved (the tier held) */
static int q36cc_dm_begin(Q36CChain *ch, Model *m, int n) {
    Cfg *c = &m->c; int L = c->n_layers, E = c->n_experts, K = c->topk, D = c->hidden, I = c->inter;
    size_t r = (size_t)ch->rows, ne = (size_t)L * E, F = sizeof(float);
    if (!ch->dm_host) {
        ch->dm_host = malloc(ne * sizeof(CcExpert)); ch->dm_res = calloc(ne, 1); ch->dm_part = calloc(L, sizeof(int));
        ch->dm_ran = calloc(L, 1); ch->dm_nres = calloc(L, sizeof(int));
        ch->dm_g = falloc(I); ch->dm_u = falloc(I); ch->dm_hh = falloc(D);
        if (!ch->dm_host || !ch->dm_res || !ch->dm_part || !ch->dm_ran || !ch->dm_nres) return 0;
        ch->promo_rate = q36cc_promote_rate(); ch->promo_hits = q36cc_promote_hits();
        if (ch->promo_rate > 0 && qt_chain_devices() == 1 && ch->dm_fmt < 100 && cc_host_up_ready()) {
            char brake[96];
            if (ch->promo_hits > 0) snprintf(brake, sizeof brake, "none while the last %d steps hit %d %% or more", Q36CC_PROMO_WIN, ch->promo_hits);
            else snprintf(brake, sizeof brake, "no brake%s", qt_chain_history() ? "" : " (no heat file: the warm start's set is no conversation's)");
            fprintf(stderr, "[chain] qwen36: up to %d experts a step promoted into the card by this conversation's routes, "
                            "the coldest residents giving way, %s (COLI_CUDA_CHAIN_PROMOTE=0 keeps the warm start's set)\n", ch->promo_rate, brake);
        } else ch->promo_rate = 0;
    }
    if (!cc_reserve(&ch->dm_ids, (size_t)L * r * K * 4, CC_DEV) || !cc_reserve(&ch->dm_w, (size_t)L * r * K * F, CC_DEV) ||
        !cc_reserve(&ch->dm_idsd, (size_t)L * r * K * 4, CC_DOWN) || !cc_reserve(&ch->dm_wd, (size_t)L * r * K * F, CC_DOWN) ||
        !cc_reserve(&ch->dm_hid, cc_moe_hid_floats((int)r, K, D, I, E) * F, CC_DEV) || !cc_reserve(&ch->dm_y, r * D * F, CC_DEV) ||
        !cc_reserve(&ch->dm_hy, r * D * F, CC_UP) || !cc_reserve(&ch->dm_dfl, 64, CC_DOWN) || !cc_reserve(&ch->dm_hfl, 64, CC_UP) ||
        !cc_reserve(&ch->dm_tab, ne * sizeof(CcExpert), CC_DEV)) return 0;
    if (!ch->dm_held) { qt_chain_hold(1); ch->dm_held = 1; }
    unsigned ep = qt_chain_epoch();
    if (!ch->dm_read || ep != ch->dm_epoch) {
        /* the residents that moved since the last read: those entries alone, or every entry
         * where the tier lost count (the first read, a host-pool registration, too many) */
        double tp = tm_now();
        int chg[QT_CHG_LIST], nchg, over;
        nchg = qt_chain_changes(chg, QT_CHG_LIST, &over);
        if (over || !ch->dm_read) {
            for (int i = 0; i < L; i++) {
                int miss = 0;
                for (int e = 0; e < E; e++) { int on = q36cc_dm_entry(ch, E, i, e); ch->dm_res[(size_t)i * E + e] = (uint8_t)on; miss += !on; }
                ch->dm_nres[i] = miss; ch->dm_part[i] = miss > 0;
            }
            if (!cc_begin() || !cc_write(ch->dm_tab, 0, ch->dm_host, ne * sizeof(CcExpert)) || !cc_submit(1)) return 0;
            ch->dm_rebuilds++;
        } else if (nchg > 0) {
            int ok = cc_begin();
            for (int k = 0; k < nchg && ok; k++) {
                int j = chg[k], i = j / E, e = j % E;
                if (j < 0 || (size_t)j >= ne) continue;
                int was = ch->dm_res[j], on = q36cc_dm_entry(ch, E, i, e);
                ch->dm_res[j] = (uint8_t)on;
                if (was != on) { ch->dm_nres[i] += was ? 1 : -1; ch->dm_part[i] = ch->dm_nres[i] > 0; }
                ok = cc_write(ch->dm_tab, (size_t)j * (sizeof(CcExpert) / 4), &ch->dm_host[j], sizeof(CcExpert));   /* offsets in floats */
                ch->dm_patches++;
            }
            if (!ok || !cc_submit(1)) return 0;
        }
        ch->dm_epoch = ep; ch->dm_read = 1; ch->dm_stage_dirty = 1;
        ch->patch_ms += tm_now() - tp;
    }
    /* the staged table (a prompt chunk's host-pool copies) follows the residents when a chunk needs it,
     * and on a large chunk the pool first (q36cc_follow) */
    int stage = n > Q36CC_STAGE_ROWS && qt_chain_host_count() > 0 && !ch->dm_stage_off && cc_fetch_ready();
    if (stage) q36cc_follow(ch, n);
    if (stage && ch->dm_stage_dirty) {
        if (!q36cc_stage_table(ch, m)) {
            if (cc_lost()) return 0;
            fprintf(stderr, "[chain] qwen36: no VRAM to stage the host pool's experts in; a prompt chunk's go to the host's step\n");
            ch->dm_stage_off = 1;
            cc_free(ch->dm_stage); ch->dm_stage = NULL;
        }
        ch->dm_stage_dirty = 0;
    }
    return 1;
}
/* layer i's table and whether some of its experts are elsewhere (the host's), for a step of n rows */
static CcBuf *q36cc_dm_tab(Q36CChain *ch, int i, int n, int *part) {
    if (n > Q36CC_STAGE_ROWS && !ch->dm_stage_off && ch->dm_stage && ch->dm_tabb) { *part = ch->dm_partb[i]; return ch->dm_tabb; }
    *part = ch->dm_part[i];
    return ch->dm_tab;
}
/* whether layer i of a step on table tab stages its experts, and the copy of the first such layer
 * from layer i on into the staging (none: nothing to copy) */
static int q36cc_stages(Q36CChain *ch, CcBuf *tab, int i) { return tab == ch->dm_tabb && ch->dm_frn[i] > 0 && !ch->dm_partb[i]; }
static int q36cc_stage_from(Q36CChain *ch, Model *m, int i) {
    for (; i < m->c.n_layers; i++) {
        if (!q36cc_stages(ch, ch->dm_tabb, i)) continue;
        ch->dm_staged++;
        for (int k = 0; k < ch->dm_frn[i]; k++) ch->dm_stage_bytes += ch->dm_fr[ch->dm_fro[i] + k].bytes;
        return cc_fetch(ch->dm_stage, ch->dm_fr + ch->dm_fro[i], ch->dm_frn[i]);
    }
    return 1;
}
static void q36cc_dm_end(Q36CChain *ch) { if (ch->dm_held) { qt_chain_hold(0); ch->dm_held = 0; } }
/* layer i's router, routing, shared expert and the device's experts; *v: the host's word
 * a hybrid layer waits for (0: none) */
static int q36cc_dm_layer(Q36CChain *ch, Model *m, Layer *l, int i, int n, unsigned *v) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    int ng = c->n_group > 1 && c->n_group <= E ? c->n_group : 1, io = i * n * K;
    CcRoute rp = {n, E, K, c->router_sigmoid ? 1 : 0, 0, E + ch->rgf[i], 0, 0, io, K, io, K, ng, ng > 1 ? c->topk_group : 1, 1, 1.f, 1.f, 0.f};
    int part;
    CcBuf *tab = q36cc_dm_tab(ch, i, n, &part);
    CcMoe mp = {n, K, D, I, ch->dm_fmt, ch->dm_gs, 0, 1, i * E, 0, D, io, K, io, K, 0, D, -1, E, tab == ch->dm_tabb && ch->dm_fmt < 100, 0};
    if (ch->dm_fmt >= 100) qt_chain_layer_format(i, &mp.fmt, &mp.fmt_d);   /* a GGUF's types change from layer to layer */
    int stg = q36cc_stages(ch, tab, i);
    int ok = cc_matmul(ch->t_gate[i], ch->h2, 0, ch->lg, 0, n) && cc_route(ch->lg, NULL, ch->dm_ids, ch->dm_w, &rp);
    *v = 0;
    /* a layer whose experts are all on the device needs no weights on the host, and its route ids
     * only after the frame (one copy for the chunk before cc_submit); a hybrid layer copies both
     * and the input rows before signalling the host */
    if (ok && part) {
        *v = ++ch->dm_seq;
        ok = cc_copy(ch->dm_idsd, io, ch->dm_ids, io, (size_t)n * K) &&
             cc_copy(ch->dm_wd, io, ch->dm_w, io, (size_t)n * K) &&
             cc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D) && cc_signal(ch->dm_dfl, 0, *v);
    }
    ok = ok && q36cc_shared(ch, m, l, i, n) && (!stg || cc_fetch_wait()) &&
         cc_moe(tab, ch->h2, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y, NULL, &mp);
    if (ok && stg) ok = cc_fetch_done() && q36cc_stage_from(ch, m, i + 1);   /* the sum below reads no weights */
    if (ok && *v) ok = cc_wait_host(ch->dm_hfl, 0, *v);
    CcMoe ms = mp; ms.parts = 2; ms.h_off = *v ? 0 : -1;
    ok = ok && cc_moe(tab, NULL, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y, ch->dm_hy, &ms);
    ch->dm_steps++;
    return ok;
}
/* the host's share of hybrid layer i: once the device's word says its routes and rows are
 * down, the slots the table lacks (the tier's misses), its rows up, its word */
static int q36cc_dm_serve(Q36CChain *ch, Model *m, int i, int n, unsigned v) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk;
    volatile unsigned *dw = (volatile unsigned *)cc_ptr(ch->dm_dfl);
    double t0 = tm_now();
    for (unsigned long spin = 0; *dw < v; spin++)
        if ((spin & 0xFFFFu) == 0 && (cc_lost() || tm_now() - t0 > 60e3)) {
            fprintf(stderr, "[chain] qwen36: the device did not reach layer %d's routes in 60 s\n", i);
            return 0;
        }
    double t1 = tm_now();
    const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * n * K;
    const float *w = (const float *)cc_ptr(ch->dm_wd) + (size_t)i * n * K, *x = (const float *)cc_ptr(ch->h2d);
    for (int s = 0; s < n; s++) {
        uint32_t mask = 0; int any = 0;
        for (int k = 0; k < K; k++) { int e = ids[s * K + k]; if (e < 0 || ch->dm_res[(size_t)i * E + e]) mask |= 1u << k; else { any = 1; ch->dm_pairs++; } }
        float *o = ch->host_routed + (size_t)s * D;
        memset(o, 0, (size_t)D * sizeof(float));
        if (any) qt_cpu_misses(m, i, x + (size_t)s * D, ids + s * K, w + s * K, mask, o, ch->dm_g, ch->dm_u, ch->dm_hh);
    }
    memcpy(cc_ptr(ch->dm_hy), ch->host_routed, (size_t)n * D * sizeof(float));
    __sync_synchronize();   /* the rows (write-combined memory) before the word */
    ((volatile unsigned *)cc_ptr(ch->dm_hfl))[0] = v;
    __sync_synchronize();
    double t2 = tm_now();
    ch->dm_hyb++; ch->dm_spin_ms += t1 - t0; ch->dm_host_ms += t2 - t1; ch->host_ms += t2 - t1;
    return 1;
}
/* after the frame: every layer's books from its ids, and the K/V rows */
static void q36cc_dm_books(Q36CChain *ch, Model *m, int n, int pb) {
    Cfg *c = &m->c; int L = c->n_layers, E = c->n_experts, K = c->topk;
    for (int i = 0; i < L; i++) {
        if (!ch->dm_ran[i]) continue;   /* a layer that took the host's step kept its own books */
        q36cc_kv_down(ch, m, i, n, pb);
        const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * n * K;
        for (int s = 0; s < n; s++) {
            const int *idx = ids + s * K;
            if (m->resident_collecting) for (int k = 0; k < K; k++) if (idx[k] >= 0) m->seen[(int64_t)i * E + idx[k]] = 1;
            if (!m->hot_pinned && m->freq) for (int k = 0; k < K; k++) if (idx[k] >= 0) m->freq[(int64_t)i * E + idx[k]]++;
            qt_chain_count(i, idx, K);
        }
    }
}

/* Rows per prompt chunk: COLI_CUDA_CHAIN_ROWS, else the most (a power of two from 256 to 4096)
 * whose scratch takes at most half the device's free memory once the tier filled it, less the
 * host pool's staging (the other half for the KV mirrors as the context grows). A chunk reads
 * every routed expert's weights once, the host pool's over the bus: the more rows, the fewer
 * reads (q36c on an RTX PRO 4000, 1727 tokens: 256 rows 1704 tok/s, 1024 2270; the experts
 * past 8 GB in the host pool: 215 against 1036 at 2048). */
static int q36cc_rows_max(Q36CChain *ch, Model *m) {
    if (ch->rows_max) return ch->rows_max;
    int env = q36cc_rows_env();
    if (env) return ch->rows_max = env;
    Cfg *c = &m->c; int D = c->hidden, I = c->inter, L = c->n_layers, K = c->topk > 0 ? c->topk : 1;
    int E = c->n_experts > 0 ? c->n_experts : 1, SI = c->shared_inter > 0 ? c->shared_inter : 1;
    if (c->dense_layer && c->dense_inter > SI) SI = c->dense_inter;   /* Flint's dense layers: a wider MLP */
    int qo = c->q_heads * c->q_head_dim, kvo = c->kv_heads * c->k_head_dim, vd = c->dn_vheads * c->dn_vdim;
    /* q36cc_bufs' device rows and the routed experts' (q36cc_dm_begin), per row */
    size_t row = ((size_t)6 * D + qo + 2 * kvo + (size_t)c->q_heads * c->head_dim + 2 * (size_t)c->dn_conv_dim + 2 * vd +
                  2 * c->dn_vheads + E + 3 * SI + 1 + (size_t)K * (I + D) + 2 * (size_t)L * K + 4 * K) * sizeof(float);
    size_t fr = 0, tot = 0, stage = 0;
    if (c->n_experts > 0 && qt_chain_host_count() > 0 && ch->dm_fmt >= 100) stage = q36cc_gguf_stage_bytes(m);
    else if (c->n_experts > 0 && qt_chain_host_count() > 0 && ch->dm_gs > 0)
        stage = (size_t)c->n_experts * (3 * ((size_t)D * I / 2) + 3 * ((size_t)D * I / ch->dm_gs) * sizeof(float) + 512);
    coli_cuda_mem_info(ch->dev, &fr, &tot);
    size_t room = fr > stage ? (fr - stage) / 2 : 0;
    int r = 4096;
    while (r > 256 && (size_t)r * row > room) r /= 2;
    fprintf(stderr, "[chain] qwen36: %d rows per prompt chunk (%.0f KiB a row, %.1f GB free on the device; COLI_CUDA_CHAIN_ROWS sets it)\n",
            r, row / 1024.0, fr / 1073741824.0);
    return ch->rows_max = r;
}

/* Every layer for S rows from host rows xh, the last nlogits rows' logits into `logit`;
 * xh gets the final rows back when want_x. 0 = not taken (nothing on the device
 * changed, or the device was lost and the state rebuilt: the CPU runs the step).
 * Otherwise the layers it ran (every one: *rows_only stays 0). */
static int q36cc_forward(Model *m, float *xh, int S, int pos_base, FILE *lf, int want_x, int nlogits, float *logit,
                         int *rows_only) {
    *rows_only = 0;
    if (!g_cuda_chain || qq_active() || g_pilot || m->mux_rows) return 0;
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->ok || ch->failed) return 0;
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, E = c->n_experts;
    int CH = q36cc_rows_max(ch, m), rows = S < CH ? S : CH;
    int half = c->rotary_dim / 2;
    if (m->snap_rows > 0 && !q36cc_spec_slots(ch, m, m->snap_rows)) {
        fprintf(stderr, "[chain] qwen36: device memory for a verify's %d copies refused; this verify on the CPU\n", m->snap_rows);
        return 0;
    }
    int mirror_ok = 1;   /* a multiplexed step's places are grown (q36cc_mux_step); the held one's to this step's end */
    if (!ch->mx_rows && ch->conv && ch->conv_cur >= 0) {
        q36cc_conv_save(ch);
        mirror_ok = q36cc_conv_grow(ch, m, ch->conv_cur, pos_base + S);
        q36cc_conv_load(ch, ch->conv_cur);
        if (!mirror_ok && q36cc_stage_drop(ch)) {
            q36cc_conv_save(ch);
            mirror_ok = q36cc_conv_grow(ch, m, ch->conv_cur, pos_base + S);
            q36cc_conv_load(ch, ch->conv_cur);
        }
        if (!mirror_ok) { q36cc_pool_off(ch, m); mirror_ok = q36cc_mirror(ch, m); }
    } else if (!ch->mx_rows) {
        mirror_ok = q36cc_mirror(ch, m);
        if (!mirror_ok && q36cc_stage_drop(ch)) mirror_ok = q36cc_mirror(ch, m);   /* the staging's memory to the mirrors */
    }
    if (!mirror_ok || !q36cc_scratch(ch, m, rows) || (lf && !cc_reserve(&ch->lfd, (size_t)L * 3 * D * sizeof(float), CC_DOWN)) ||
        (want_x && !cc_reserve(&ch->xd, (size_t)rows * D * sizeof(float), CC_DOWN)) ||
        (nlogits > 1 && (!cc_reserve(&ch->fin, (size_t)nlogits * D * sizeof(float), CC_DEV) ||
                         !cc_reserve(&ch->out, (size_t)nlogits * c->vocab * sizeof(float), CC_DEV) ||
                         !cc_reserve(&ch->outd, (size_t)nlogits * c->vocab * sizeof(float), CC_DOWN)))) {
        fprintf(stderr, "[chain] qwen36: device memory for %d rows refused; the per-matrix path\n", rows);
        ch->failed = 1;
        if (ch->conv) q36cc_mux_drop(m);   /* the conversations' states the device holds, to the host */
        return 0;
    }
    int snapped = 0;
    double w0 = ch->wait_ms, h0 = ch->host_ms;
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        if (c0 > 0 && q36_chunk_control()) {   /* a STOP or CANCEL between two prompt chunks */
            q36cc_dm_end(ch);
            for (int i = 0; i < L; i++) if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
            ch->snap_valid = 0;
            return 0;
        }
        int ns = m->snap_rows - c0 < n ? m->snap_rows - c0 : n;   /* the rows of this chunk a verify copies */
        if (ns < 0) ns = 0;
        if (ns) snapped = c0 + ns;
        if (half) {   /* the CPU's own angles, cosines and sines (rope_head_partial / _mrope) */
            float *cs = (float *)cc_ptr(ch->cs);
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
        /* the routed experts in the frame (the table before the frame): a decode-sized step's
         * every layer; a prompt chunk's layers whose experts are all in the table (a layer
         * with some elsewhere, or the prompt's re-plan of the residents, keeps the host's step) */
        int dm = E > 0 && ch->dm_on && (n <= Q36CC_DM_ROWS || !prefill_replan_on());
        if (dm && !q36cc_dm_begin(ch, m, n) && !(!cc_lost() && q36cc_stage_drop(ch) && q36cc_dm_begin(ch, m, n))) {
            if (cc_lost()) goto lost;
            fprintf(stderr, "[chain] qwen36: the routed experts' buffers or table refused; the host's step from here on\n");
            q36cc_dm_end(ch); ch->dm_on = dm = 0;
        }
        if (dm) memset(ch->dm_ran, 0, (size_t)L);
        if (!q36cc_frame_begin(ch) || !cc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !(ch->mx_rows ? q36cc_mux_push(ch, m, c0, n) : q36cc_push_state(ch, m, pb))) goto lost;
        /* the angles to the device once a step: a RoPE that read them where the host wrote them
         * paid the bus on every element, 11-16 us a call (Flint: 70 calls a token) */
        if (half && !cc_copy(ch->csd, 0, ch->cs, 0, (size_t)n * 2 * half)) goto lost;
        if (dm && q36cc_dm_tab(ch, 0, n, &(int){0}) == ch->dm_tabb && !q36cc_stage_from(ch, m, 0)) goto lost;   /* the first staged layer's copy */
        int ok = 1, pending = 0, pend_i = -1, dm_ids_defer = 0;   /* pend_i: the hybrid layer the host serves after recording the next */
        unsigned pend_v = 0;
        CcBuf *routed = ch->routed;              /* where the layer before left its routed sum */
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) {
                ok = q36cc_combine(ch, m, i - 1, n, routed);
                if (ok && lf) ok = cc_copy(ch->lfd, ((size_t)(i - 1) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
            }
            ok = ok && q36cc_norm(ch->x, 0, ch->prm, ch->o_in[i], ch->nrm, 0, n, D, c->eps);
            if (ch->mx_rows) ok = ok && (c->is_attn[i] ? q36cc_attention_mx(ch, m, l, i, n, c0) : q36cc_deltanet_mx(ch, m, l, i, n, c0));
            else ok = ok && (c->is_attn[i] ? q36cc_attention(ch, m, l, i, n, pb) : q36cc_deltanet(ch, m, l, i, n, c0, ns));
            if (ok && lf) ok = cc_copy(ch->lfd, (size_t)i * 3 * D, ch->tmp, (size_t)(n - 1) * D, D);
            CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            if (c->muse) {   /* Muse Spark: the branch normalised, then the residual gate */
                CcNorm pn = {n, D, 1, 0, D, D, 0, D, D, 0, 0, CC_NORM_NOW, c->post_eps, 1.f};
                CcEw g = {CC_EW_RESGATE, n * D, D, 1, 0, 1, 0, 0, 0, (int)ch->o_ra[i] + D, (int)ch->o_ra[i], 1.f};
                ok = ok && cc_norm(ch->tmp, ch->prm, ch->tmp, &pn) && cc_ew(ch->x, ch->x, ch->tmp, ch->prm, ch->prm, &g);
            } else
            ok = ok && cc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
            if (ok && lf) ok = cc_copy(ch->lfd, ((size_t)i * 3 + 1) * D, ch->x, (size_t)(n - 1) * D, D);
            ok = ok && q36cc_norm(ch->x, 0, ch->prm, ch->o_post[i], ch->h2, 0, n, D, c->eps);
            const int routed_l = E > 0 && !q36_dense_l(c, i);   /* a Flint dense layer: its MLP only */
            int dpart = 0;
            if (dm && routed_l) (void)q36cc_dm_tab(ch, i, n, &dpart);
            if (ok && dm && routed_l && (n <= Q36CC_DM_ROWS || !dpart)) {   /* routed in the frame, the residents (and a prompt chunk's
                                                              * host pool) here; a hybrid layer's host share once the next is recorded */
                unsigned v = 0;
                ch->dm_ran[i] = 1;
                ok = q36cc_dm_layer(ch, m, l, i, n, &v);
                if (!v) dm_ids_defer = 1;
                if (ok && pend_v) ok = q36cc_dm_serve(ch, m, pend_i, n, pend_v);
                pend_i = i; pend_v = v; routed = ch->dm_y; pending = 1;
                continue;
            }
            if (ok && routed_l) {
                ok = cc_matmul(ch->t_gate[i], ch->h2, 0, ch->lg, 0, n);
                if (ok && ch->rgf[i]) {   /* the router's rows down without the gate after each */
                    CcRegion *rgn = malloc((size_t)n * sizeof *rgn);
                    ok = rgn != NULL;
                    for (int s = 0; ok && s < n; s++) rgn[s] = (CcRegion){(size_t)s * E, (size_t)s * (E + 1), (size_t)E};
                    ok = ok && cc_copy_regions(ch->lgd, ch->lg, rgn, n);
                    free(rgn);
                } else ok = ok && cc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E);
                if (c->muse) {   /* Muse Spark: the experts read the rows at their width, normalised */
                    CcNorm pn = {n, q36_eh(c), 1, 0, q36_eh(c), q36_eh(c), 0, q36_eh(c), q36_eh(c), (int)ch->o_pre[i], 0, CC_NORM_ADD1, c->eps, 1.f};
                    ok = ok && cc_matmul(ch->t_ld[i], ch->h2, 0, ch->lat, 0, n) && cc_norm(ch->lat, ch->prm, ch->lat, &pn) &&
                         cc_copy(ch->latd, 0, ch->lat, 0, (size_t)n * q36_eh(c));
                } else
                ok = ok && cc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D);
                if (ok && pend_v) { ok = q36cc_dm_serve(ch, m, pend_i, n, pend_v); pend_v = 0; }   /* before a wait for the frame */
                double t0 = tm_now();
                ok = ok && cc_submit(1);   /* A1 */
                ch->wait_ms += tm_now() - t0;
                if (!ok) break;
                if (c->is_attn[i]) q36cc_kv_down(ch, m, i, n, pb);   /* the rows into the host's cache */
                /* A2: the shared expert, while the host computes the routed experts */
                ok = q36cc_frame_begin(ch) && q36cc_shared(ch, m, l, i, n) && cc_submit(0);
                double t1 = tm_now();
                moe_ex(m, l, i, (float *)cc_ptr(c->muse ? ch->latd : ch->h2d), n, ch->host_routed, (const float *)cc_ptr(ch->lgd), 1);
                memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)n * q36_eh(c) * sizeof(float));
                double t2 = tm_now();
                tm_add(n, 2, t2 - t1); ch->host_ms += t2 - t1;
                ok = ok && q36cc_frame_begin(ch);
                routed = ch->routed;
            } else if (ok) {
                ok = q36cc_shared(ch, m, l, i, n);
                if (ok && i + 1 < L) ok = cc_submit(0) && q36cc_frame_begin(ch);
            }
            pending = 1;
        }
        if (ok) ok = q36cc_combine(ch, m, L - 1, n, routed);
        if (ok && dm_ids_defer) ok = cc_copy(ch->dm_idsd, 0, ch->dm_ids, 0, (size_t)L * n * c->topk);
        if (ok && lf) ok = cc_copy(ch->lfd, ((size_t)(L - 1) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
        if (ok && want_x) ok = cc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D);
        int last = c0 + n == S;
        /* the final norm and lm_head on this chunk's share of the last nlogits rows. The head writes its
         * logits on the card and one copy takes them to outd: written into outd itself, one float per
         * warp over the bus, the head took 7.4 ms of a 13.5 ms step on an RTX PRO 4000, 0.9 ms this way */
        int lo0 = S - nlogits > c0 ? S - nlogits - c0 : 0, lo_n = nlogits ? n - lo0 : 0;
        if (ok && lo_n > 0) {
            int dst = c0 + lo0 - (S - nlogits);
            ok = q36cc_norm(ch->x, (size_t)lo0 * D, ch->prm, ch->o_final, ch->fin, (size_t)dst * D, lo_n, D, c->eps) &&
                 cc_matmul(ch->t_lm, ch->fin, (size_t)dst * D, ch->out, (size_t)dst * c->vocab, lo_n) &&
                 cc_copy(ch->outd, (size_t)dst * c->vocab, ch->out, (size_t)dst * c->vocab, (size_t)lo_n * c->vocab);
        }
        double t0 = tm_now();
        ok = ok && cc_submit(0);
        if (ok && pend_v) ok = q36cc_dm_serve(ch, m, pend_i, n, pend_v);   /* the last hybrid layer's, the tail recorded */
        ok = ok && cc_finish();
        ch->wait_ms += tm_now() - t0;
        if (!ok) goto lost;
        if (dm) q36cc_dm_books(ch, m, n, pb);
        if (dm && n <= Q36CC_DM_ROWS) q36cc_promote(ch, m, n);
        else if (dm) { ch->pr_steps0 = ch->dm_steps; ch->pr_pairs0 = ch->dm_pairs; }   /* a prompt chunk's misses are not the brake's */
        /* a dense model reads its K/V rows back here, and so does a Flint dense layer (no
         * expert step ends its frame; q36c_forward_seg says what went wrong without it) */
        for (int i = 0; i < L; i++) {
            if (!c->is_attn[i] || (E > 0 && !q36_dense_l(c, i))) continue;
            q36cc_kv_down(ch, m, i, n, pb);
        }
        if (want_x) memcpy(xh + (size_t)c0 * D, cc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        if (ch->mx_rows)   /* each row's place: its row on the device, its DeltaNet state there only */
            for (int s = 0; s < n; s++) {
                Q36ccConv *v = ch->mx_conv[c0 + s];
                for (int i = 0; i < L; i++) if (c->is_attn[i]) v->kv_valid[i] = ch->mx_rows[c0 + s].pos + 1;
                v->dn_where = Q36CC_DEV; v->host_zero = 0;
            }
        else {
            for (int i = 0; i < L; i++) if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
            ch->dn_where = Q36CC_DEV; ch->host_zero = 0;
        }
        if (last) memcpy(logit, cc_ptr(ch->outd), (size_t)nlogits * c->vocab * sizeof(float));
    }
    q36cc_dm_end(ch);
    if (lf) fwrite(cc_ptr(ch->lfd), sizeof(float), (size_t)L * 3 * D, lf);
    ch->snap_valid = snapped;
    ch->forwards++;
    if (S == 1) { ch->dec_n++; ch->dec_wait_ms += ch->wait_ms - w0; ch->dec_host_ms += ch->host_ms - h0; }
    return L;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!cc_lost()) cc_finish();
    q36cc_dm_end(ch);
    if (!ch->mx_rows) q36cc_recover(m, pos_base);   /* the held conversation (none in a multiplexed step) */
    g_cuda_chain = 0; ch->failed = 1;
    if (ch->conv) q36cc_mux_lost(m);
    return 0;
}

/* One decode step of several conversations on the CUDA chain (q36c_mux_step's twin) */
static int q36cc_mux_step(Model *m, const Q36Row *rows, int S, float *xh, float *logit) {
    Q36CChain *ch = q36cc_of(m);
    if (!g_cuda_chain || !ch || !ch->conv || !q36cc_pool_ok(ch) || S < 1 || S > q36cc_rows_max(ch, m)) return 0;
    for (int s = 0; s < S; s++) if (rows[s].seq->mpos || rows[s].seq->rope_delta) return 0;
    Q36ccConv **e = malloc((size_t)S * sizeof *e);
    if (!e) return 0;
    q36cc_conv_save(ch);
    ch->conv_clock++;
    int ok = 1;
    for (int s = 0; s < S && ok; s++) {
        int k = q36cc_conv_get(ch, m, (int)(rows[s].seq - g_q36_mux_seq), 1);
        if ((ok = k >= 0)) { ch->conv[k].used = ch->conv_clock; e[s] = &ch->conv[k]; }
    }
    for (int s = 0; s < S && ok; s++) ok = q36cc_conv_grow(ch, m, (int)(e[s] - ch->conv), rows[s].pos + 1);
    int n = 0, ro = 0, failed_before = ch->failed;
    if (ok) {
        ch->mx_rows = rows; ch->mx_conv = e;
        n = q36cc_forward(m, xh, S, 0, NULL, 0, S, logit, &ro);
        ch->mx_rows = NULL; ch->mx_conv = NULL;
    }
    free(e);
    if (ch->failed && !failed_before) return 0;   /* lost: every place's state rebuilt on the host */
    if (ch->conv_cur >= 0) q36cc_conv_load(ch, ch->conv_cur);
    if (n) { ch->mux_steps++; ch->mux_rows += (unsigned long long)S; }
    else ch->mux_host_steps++;
    return n > 0;
}

static void q36cc_report(Model *m) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->ok || !ch->forwards) return;
    CcStats st; cc_stats(&st);
    fprintf(stderr, "[chain] qwen36: %llu forwards, %llu frames (%llu ops, %llu matmuls), %.1f ms waiting for the device, "
                    "%.1f ms of routed experts on the host, %.1f MiB on the device; per decode token %.2f ms waiting, %.2f ms of experts (%llu tokens)\n",
            ch->forwards, st.frames, st.ops, st.matmuls, ch->wait_ms, ch->host_ms, st.dev_bytes / 1048576.0,
            ch->dec_n ? ch->dec_wait_ms / ch->dec_n : 0.0, ch->dec_n ? ch->dec_host_ms / ch->dec_n : 0.0, ch->dec_n);
    if (ch->conv)
        fprintf(stderr, "[chain] qwen36: KV_SLOTS=%d: %llu decode steps of several conversations in the chain (%llu rows, "
                        "%.2f a step), %llu on the host's path; %d conversation%s state on the device, %llu handed over\n",
                g_q36_mux_slots, ch->mux_steps, ch->mux_rows, ch->mux_steps ? (double)ch->mux_rows / (double)ch->mux_steps : 0.0,
                ch->mux_host_steps, ch->nconv, ch->nconv == 1 ? "'s" : "s'", ch->conv_handed);
    if (ch->dm_steps)
        fprintf(stderr, "[chain] qwen36: %llu layer steps with the routed experts in the frame, %llu of them hybrid "
                        "(%llu slots on the host in %.1f ms, %.1f ms waiting for the device's routes)\n",
                ch->dm_steps, ch->dm_hyb, ch->dm_pairs, ch->dm_host_ms, ch->dm_spin_ms);
    if (ch->dm_promoted || ch->pr_braked || ch->dm_rebuilds > 1)
        fprintf(stderr, "[chain] qwen36: %llu experts promoted into the card by this conversation's routes (%llu landed), %llu steps braked; "
                        "the table read whole %llu times, %llu entries patched; %.1f ms choosing and staging, %.1f ms patching\n",
                ch->dm_promoted, ch->dm_landed, ch->pr_braked, ch->dm_rebuilds, ch->dm_patches, ch->promo_ms, ch->patch_ms);
    if (ch->dm_staged)
        fprintf(stderr, "[chain] qwen36: %llu prompt layer steps staged the host pool's experts (%.2f GB copied over the bus)\n",
                ch->dm_staged, ch->dm_stage_bytes / 1073741824.0);
    if (ch->hp_follows)
        fprintf(stderr, "[chain] qwen36: the host pool followed the residents before %llu prompt chunks: %llu experts joined it, "
                        "%llu left it, %.1f ms\n", ch->hp_follows, ch->hp_joined, ch->hp_left, ch->hp_ms);
}
