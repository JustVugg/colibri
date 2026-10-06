/* qwen36_cuda_chain.h -- Qwen3.6's layers as a dense chain on a CUDA device (cuda_chain.h).
 * Included once by qwen36.c in a COLI_CUDA build, after the CPU forward it stands in
 * for; COLI_CUDA_CHAIN=1 asks for it (q36cc_setup decides, at startup, after the tier
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
 * and lm_head on ONE device, so COLI_CUDA_CHAIN=1 also offers the shared expert to
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

typedef struct {
    int ok, failed;
    int dev;                                   /* the CUDA ordinal every matrix sits on */
    int n, head;                               /* every layer, and lm_head, on the device */
    int rows;                                  /* scratch capacity in rows */
    int cap;                                   /* the host's kv_cap the mirrors follow */
    CcBuf *prm;                                /* every norm / conv / DeltaNet parameter */
    size_t *o_in, *o_post, *o_qn, *o_kn, *o_conv, *o_dn, o_final;
    ColiCudaTensor **t_ab, **t_sg, **t_gate;   /* the chain's own: DeltaNet b|a rows, shared gate row, router */
    ColiCudaTensor **t_q, **t_k, **t_v, **t_o, **t_dnp, **t_dno, **t_shg, **t_shu, **t_shd, *t_lm;   /* the tier's */
    CcBuf **rec, **ring, **kc, **vc;           /* per layer state */
    CcBuf **rec_snap[Q36_SPEC_SNAPS], **ring_snap[Q36_SPEC_SNAPS];   /* a verify's copies, slot r after row r */
    int snap_slots, snap_valid;
    int *kv_valid, *attn_ord, n_attn;
    int dn_where, host_zero;
    CcBuf *x, *nrm, *tmp, *q, *k, *v, *ctx, *qkvz, *ab, *cv, *dny, *h2, *lg, *gs, *us, *hs, *ds, *sgd, *fin;
    CcBuf *h2d, *lgd, *kvd, *outd, *xd, *lfd, *routed, *cs;
    float *host_routed;
    unsigned long long forwards, frames, fault_at;
    double wait_ms, host_ms;
} Q36CChain;

static int g_cuda_chain = 0;     /* COLI_CUDA_CHAIN asked for it and the chain is up */
static Q36CChain *q36cc_of(Model *m) { return (Q36CChain *)m->cchain; }

static int q36cc_env_on(void) { const char *e = getenv("COLI_CUDA_CHAIN"); return e && *e == '1'; }
static int q36cc_rows_env(void) {
    const char *e = getenv("COLI_CUDA_CHAIN_ROWS");
    int r = e && *e ? atoi(e) : 256;
    return r < 1 ? 1 : r > 8192 ? 8192 : r;
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

static void q36cc_free_own(Q36CChain *ch, Model *m) {
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
    if (qq_active() || g_pilot || mux_slots > 1) {
        fprintf(stderr, "[chain] qwen36: %s; the chain stays off\n",
                qq_active() ? "a qpack container" : g_pilot ? "PILOT prefetch reads the residual on the host" : "KV_SLOTS > 1 keeps one state per conversation on the host");
        return NULL;
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
    ch->t_ab = calloc(L, sizeof(void *)); ch->t_sg = calloc(L, sizeof(void *)); ch->t_gate = calloc(L, sizeof(void *));
    ch->t_q = calloc(L, sizeof(void *)); ch->t_k = calloc(L, sizeof(void *)); ch->t_v = calloc(L, sizeof(void *));
    ch->t_o = calloc(L, sizeof(void *)); ch->t_dnp = calloc(L, sizeof(void *)); ch->t_dno = calloc(L, sizeof(void *));
    ch->t_shg = calloc(L, sizeof(void *)); ch->t_shu = calloc(L, sizeof(void *)); ch->t_shd = calloc(L, sizeof(void *));
    ch->rec = calloc(L, sizeof(void *)); ch->ring = calloc(L, sizeof(void *));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->attn_ord = calloc(L, sizeof(int));
    if (!ch->o_in || !ch->o_post || !ch->o_qn || !ch->o_kn || !ch->o_conv || !ch->o_dn || !ch->t_ab || !ch->t_sg || !ch->t_gate ||
        !ch->t_q || !ch->t_k || !ch->t_v || !ch->t_o || !ch->t_dnp || !ch->t_dno || !ch->t_shg || !ch->t_shu || !ch->t_shd ||
        !ch->rec || !ch->ring || !ch->kc || !ch->vc || !ch->kv_valid || !ch->attn_ord) return NULL;
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
        } else {
            Q36CC_TAKE(&ch->t_dnp[i], qt_dnproj_tensor(i, &d_), "the DeltaNet in_proj");
            Q36CC_TAKE(&ch->t_dno[i], qt_dense_tensor(l->qth_dnout - 1, &d_), "the DeltaNet out_proj");
        }
        if (c->shared_inter > 0) {
            Q36CC_TAKE(&ch->t_shg[i], qt_dense_tensor(l->qth_shg - 1, &d_), "the shared expert's gate");
            Q36CC_TAKE(&ch->t_shu[i], qt_dense_tensor(l->qth_shu - 1, &d_), "the shared expert's up");
            Q36CC_TAKE(&ch->t_shd[i], qt_dense_tensor(l->qth_shd - 1, &d_), "the shared expert's down");
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
        if (ok && c->n_experts > 0) { ch->t_gate[i] = q36cc_qw_tensor(ch->dev, &l->gate); ok = ch->t_gate[i] != NULL; own += (size_t)l->gate.I * l->gate.O * (l->gate.q ? 1 : 4); }
        if (ok && l->sh_gate && c->shared_inter > 0) { ch->t_sg[i] = q36cc_f32_tensor(ch->dev, l->sh_gate, D, 1); ok = ch->t_sg[i] != NULL; own += (size_t)D * 4; }
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
    size_t r = (size_t)rows, F = sizeof(float);
    int ok = cc_reserve(&ch->x, r * D * F, CC_DEV) && cc_reserve(&ch->nrm, r * D * F, CC_DEV) && cc_reserve(&ch->tmp, r * D * F, CC_DEV) &&
             cc_reserve(&ch->q, r * qo * F, CC_DEV) && cc_reserve(&ch->k, r * kvo * F, CC_DEV) && cc_reserve(&ch->v, r * kvo * F, CC_DEV) &&
             cc_reserve(&ch->ctx, r * H * hd * F, CC_DEV) && cc_reserve(&ch->qkvz, r * (c->dn_conv_dim + vd) * F, CC_DEV) &&
             cc_reserve(&ch->ab, r * 2 * c->dn_vheads * F, CC_DEV) && cc_reserve(&ch->cv, r * c->dn_conv_dim * F, CC_DEV) &&
             cc_reserve(&ch->dny, r * vd * F, CC_DEV) && cc_reserve(&ch->h2, r * D * F, CC_DEV) && cc_reserve(&ch->lg, r * E * F, CC_DEV) &&
             cc_reserve(&ch->gs, r * SI * F, CC_DEV) && cc_reserve(&ch->us, r * SI * F, CC_DEV) && cc_reserve(&ch->hs, r * SI * F, CC_DEV) &&
             cc_reserve(&ch->ds, r * D * F, CC_DEV) && cc_reserve(&ch->sgd, r * F, CC_DEV) && cc_reserve(&ch->fin, D * F, CC_DEV) &&
             cc_reserve(&ch->h2d, r * D * F, CC_DOWN) && cc_reserve(&ch->lgd, r * E * F, CC_DOWN) &&
             cc_reserve(&ch->kvd, (size_t)(ch->n_attn ? ch->n_attn : 1) * 2 * r * kvo * F, CC_DOWN) &&
             cc_reserve(&ch->outd, (size_t)c->vocab * F, CC_DOWN) && cc_reserve(&ch->routed, r * D * F, CC_UP) &&
             cc_reserve(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_UP);
    return ok;
}
/* the KV mirrors at the host's capacity (a cache that grew is mirrored again) */
static int q36cc_mirror(Q36CChain *ch, Model *m) {
    Cfg *c = &m->c; int KV = c->kv_heads;
    if (ch->cap != m->kv_cap) {
        for (int i = 0; i < c->n_layers; i++) {
            if (!c->is_attn[i]) continue;
            cc_free(ch->kc[i]); cc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
            ch->kv_valid[i] = 0;
            ch->kc[i] = cc_buf((size_t)KV * m->kv_cap * c->k_head_dim * sizeof(float), CC_DEV);
            ch->vc[i] = cc_buf((size_t)KV * m->kv_cap * c->k_head_dim * sizeof(float), CC_DEV);
            if (!ch->kc[i] || !ch->vc[i]) { ch->cap = 0; return 0; }
        }
        ch->cap = m->kv_cap;
    }
    return 1;
}
static int q36cc_scratch(Q36CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden;
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
        if (!c->is_attn[i] || ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < KV && ok; h++) {
            size_t src = ((size_t)h * m->max_t + t0) * kvd, dst = ((size_t)h * ch->cap + t0) * kvd;
            ok = cc_write(ch->kc[i], dst, m->K[i] + src, (size_t)n * kvd * sizeof(float)) &&
                 cc_write(ch->vc[i], dst, m->V[i] + src, (size_t)n * kvd * sizeof(float));
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
static int q36cc_attention(Q36CChain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, half = c->rotary_dim / 2, gate_dim = qdim > hd ? qdim - hd : 0;
    int ok = cc_matmul(ch->t_q[i], ch->nrm, 0, ch->q, 0, n) &&
             cc_matmul(ch->t_k[i], ch->nrm, 0, ch->k, 0, n) &&
             cc_matmul(ch->t_v[i], ch->nrm, 0, ch->v, 0, n);
    if (ok && l->qn) { CcNorm p = {n * H, hd, H, 0, qo, qdim, 0, qo, qdim, (int)ch->o_qn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
                       ok = cc_norm(ch->q, ch->prm, ch->q, &p); }
    if (ok && half) { CcRope p = {n * H, H, 0, qo, qdim, half, 0, 2 * half}; ok = cc_rope(ch->q, ch->cs, &p); }
    if (ok && l->kn) { CcNorm p = {n * KV, kvd, KV, 0, kvo, kvd, 0, kvo, kvd, (int)ch->o_kn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
                       ok = cc_norm(ch->k, ch->prm, ch->k, &p); }
    if (ok && half) { CcRope p = {n * KV, KV, 0, kvo, kvd, half, 0, 2 * half}; ok = cc_rope(ch->k, ch->cs, &p); }
    if (!ok) return 0;
    /* the new rows into the device cache, and to the host's */
    size_t ko = (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    CcRegion *rg = malloc(sizeof *rg * (size_t)n * KV);
    if (!rg) return 0;
    for (int s = 0; s < n; s++) for (int h = 0; h < KV; h++)
        rg[s * KV + h] = (CcRegion){((size_t)h * ch->cap + pb + s) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
    ok = cc_copy_regions(ch->kc[i], ch->k, rg, n * KV) && cc_copy_regions(ch->vc[i], ch->v, rg, n * KV) &&
         cc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && cc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    free(rg);
    CcAttn a = {n, H, KV, hd, pb, ch->cap, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0,
                1.f / sqrtf((float)hd), 0, 0};
    return ok && cc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, NULL, &a) &&
           cc_matmul(ch->t_o[i], ch->ctx, 0, ch->tmp, 0, n);
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
                      vh + s0 * 2 * vh, 2 * vh, CD + s0 * PD, PD, s0 * vd, vd, snap_row, 0, c->eps,
                      1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
        ok = cc_dnconv(ch->qkvz, ch->prm, ch->ring[i], ch->cv, cs, &cp) &&
             cc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->qkvz, ch->rec[i], ch->prm, ch->dny, rs, &rp);
    }
    return ok && cc_matmul(ch->t_dno[i], ch->dny, 0, ch->tmp, 0, n);
}
static int q36cc_shared(Q36CChain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c; int ok = 1, SI = c->shared_inter; (void)l;
    if (SI > 0) {
        CcEw p = {CC_EW_SWIGLU, n * SI, SI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = cc_matmul(ch->t_shg[i], ch->h2, 0, ch->gs, 0, n) &&
             cc_matmul(ch->t_shu[i], ch->h2, 0, ch->us, 0, n) &&
             cc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &p) &&
             cc_matmul(ch->t_shd[i], ch->hs, 0, ch->ds, 0, n);
    }
    if (ok && ch->t_sg[i]) ok = cc_matmul(ch->t_sg[i], ch->h2, 0, ch->sgd, 0, n);
    return ok;
}
/* x += routed + gate * shared, as moe() leaves `out` and layers_forward_range adds it */
static int q36cc_combine(Q36CChain *ch, Model *m, int i, int n) {
    Cfg *c = &m->c;
    int flags = (c->n_experts > 0 ? 1 : 0) | (c->shared_inter > 0 ? 2 : 0) | (m->L[i].sh_gate && c->shared_inter > 0 ? 4 : 0);
    CcEw p = {CC_EW_COMBINE, n * c->hidden, c->hidden, 1, flags, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_ew(ch->x, ch->x, ch->routed, ch->ds, ch->sgd, &p);
}
/* COLI_CUDA_CHAIN_FAULT=n: the n-th frame fails (for the tests) */
static int q36cc_frame_begin(Q36CChain *ch) {
    ch->frames++;
    if (ch->fault_at && ch->frames == ch->fault_at) { fprintf(stderr, "[chain] qwen36: fault injected at frame %llu\n", ch->frames); return 0; }
    return cc_begin();
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
    int CH = q36cc_rows_env(), rows = S < CH ? S : CH;
    int kvo = c->kv_heads * c->k_head_dim, half = c->rotary_dim / 2;
    if (m->snap_rows > 0 && !q36cc_spec_slots(ch, m, m->snap_rows)) {
        fprintf(stderr, "[chain] qwen36: device memory for a verify's %d copies refused; this verify on the CPU\n", m->snap_rows);
        return 0;
    }
    if (!q36cc_mirror(ch, m) || !q36cc_scratch(ch, m, rows) || (lf && !cc_reserve(&ch->lfd, (size_t)L * 3 * D * sizeof(float), CC_DOWN)) ||
        (want_x && !cc_reserve(&ch->xd, (size_t)rows * D * sizeof(float), CC_DOWN)) ||
        (nlogits > 1 && (!cc_reserve(&ch->fin, (size_t)nlogits * D * sizeof(float), CC_DEV) ||
                         !cc_reserve(&ch->outd, (size_t)nlogits * c->vocab * sizeof(float), CC_DOWN)))) {
        fprintf(stderr, "[chain] qwen36: device memory for %d rows refused; the per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    int snapped = 0;
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        int ns = m->snap_rows - c0 < n ? m->snap_rows - c0 : n;   /* the rows of this chunk a verify copies */
        if (ns < 0) ns = 0;
        if (ns) snapped = c0 + ns;
        if (half) {   /* the CPU's own angles, cosines and sines (rope_head_partial / _mrope) */
            float *cs = (float *)cc_ptr(ch->cs);
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
        if (!q36cc_frame_begin(ch) || !cc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !q36cc_push_state(ch, m, pb)) goto lost;
        int ok = 1, pending = 0;
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) {
                ok = q36cc_combine(ch, m, i - 1, n);
                if (ok && lf) ok = cc_copy(ch->lfd, ((size_t)(i - 1) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
            }
            ok = ok && q36cc_norm(ch->x, 0, ch->prm, ch->o_in[i], ch->nrm, 0, n, D, c->eps);
            ok = ok && (c->is_attn[i] ? q36cc_attention(ch, m, l, i, n, pb) : q36cc_deltanet(ch, m, l, i, n, c0, ns));
            if (ok && lf) ok = cc_copy(ch->lfd, (size_t)i * 3 * D, ch->tmp, (size_t)(n - 1) * D, D);
            CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = ok && cc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
            if (ok && lf) ok = cc_copy(ch->lfd, ((size_t)i * 3 + 1) * D, ch->x, (size_t)(n - 1) * D, D);
            ok = ok && q36cc_norm(ch->x, 0, ch->prm, ch->o_post[i], ch->h2, 0, n, D, c->eps);
            if (ok && E > 0) {
                ok = cc_matmul(ch->t_gate[i], ch->h2, 0, ch->lg, 0, n) &&
                     cc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && cc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D);
                double t0 = tm_now();
                ok = ok && cc_submit(1);   /* A1 */
                ch->wait_ms += tm_now() - t0;
                if (!ok) break;
                if (c->is_attn[i]) {                           /* the rows into the host's cache */
                    const float *kv = (const float *)cc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
                    for (int s = 0; s < n; s++) for (int h = 0; h < c->kv_heads; h++) {
                        size_t dst = ((size_t)h * m->max_t + pb + s) * c->k_head_dim, src = (size_t)s * kvo + (size_t)h * c->k_head_dim;
                        memcpy(m->K[i] + dst, kv + src, c->k_head_dim * sizeof(float));
                        memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->k_head_dim * sizeof(float));
                    }
                }
                /* A2: the shared expert, while the host computes the routed experts */
                ok = q36cc_frame_begin(ch) && q36cc_shared(ch, m, l, i, n) && cc_submit(0);
                double t1 = tm_now();
                moe_ex(m, l, i, (float *)cc_ptr(ch->h2d), n, ch->host_routed, (const float *)cc_ptr(ch->lgd), 1);
                memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
                double t2 = tm_now();
                tm_add(n, 2, t2 - t1); ch->host_ms += t2 - t1;
                ok = ok && q36cc_frame_begin(ch);
            } else if (ok) {
                ok = q36cc_shared(ch, m, l, i, n);
                if (ok && i + 1 < L) ok = cc_submit(0) && q36cc_frame_begin(ch);
            }
            pending = 1;
        }
        if (ok) ok = q36cc_combine(ch, m, L - 1, n);
        if (ok && lf) ok = cc_copy(ch->lfd, ((size_t)(L - 1) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
        if (ok && want_x) ok = cc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D);
        int last = c0 + n == S;
        /* the final norm and lm_head on this chunk's share of the last nlogits rows */
        int lo0 = S - nlogits > c0 ? S - nlogits - c0 : 0, lo_n = nlogits ? n - lo0 : 0;
        if (ok && lo_n > 0) {
            int dst = c0 + lo0 - (S - nlogits);
            ok = q36cc_norm(ch->x, (size_t)lo0 * D, ch->prm, ch->o_final, ch->fin, (size_t)dst * D, lo_n, D, c->eps) &&
                 cc_matmul(ch->t_lm, ch->fin, (size_t)dst * D, ch->outd, (size_t)dst * c->vocab, lo_n);
        }
        double t0 = tm_now();
        ok = ok && cc_submit(1);
        ch->wait_ms += tm_now() - t0;
        if (!ok) goto lost;
        if (E == 0) for (int i = 0; i < L; i++) {           /* a dense model reads its K/V rows back here */
            if (!c->is_attn[i]) continue;
            const float *kv = (const float *)cc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
            for (int s = 0; s < n; s++) for (int h = 0; h < c->kv_heads; h++) {
                size_t dst = ((size_t)h * m->max_t + pb + s) * c->k_head_dim, src = (size_t)s * kvo + (size_t)h * c->k_head_dim;
                memcpy(m->K[i] + dst, kv + src, c->k_head_dim * sizeof(float));
                memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->k_head_dim * sizeof(float));
            }
        }
        if (want_x) memcpy(xh + (size_t)c0 * D, cc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        for (int i = 0; i < L; i++) if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
        ch->dn_where = Q36CC_DEV; ch->host_zero = 0;
        if (last) memcpy(logit, cc_ptr(ch->outd), (size_t)nlogits * c->vocab * sizeof(float));
    }
    if (lf) fwrite(cc_ptr(ch->lfd), sizeof(float), (size_t)L * 3 * D, lf);
    ch->snap_valid = snapped;
    ch->forwards++;
    return L;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!cc_lost()) cc_finish();
    q36cc_recover(m, pos_base);
    return 0;
}

static void q36cc_report(Model *m) {
    Q36CChain *ch = q36cc_of(m);
    if (!ch || !ch->ok || !ch->forwards) return;
    CcStats st; cc_stats(&st);
    fprintf(stderr, "[chain] qwen36: %llu forwards, %llu frames (%llu ops, %llu matmuls), %.1f ms waiting for the device, "
                    "%.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, ch->wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
}
