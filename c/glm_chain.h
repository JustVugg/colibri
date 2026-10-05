/* glm_chain.h -- GLM-5.2's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by colibri.c in a COLI_VULKAN build, after the CPU layer forward it
 * stands in for; COLI_VK_CHAIN decides (coli_vk_chain_decide, this engine's integrated
 * GPU default COLI_VK_CHAIN_UNMEASURED: off, not measured on a GLM checkpoint).
 *
 * What runs where, per layer, for one block of rows (decode: one row):
 *   device, frame A1: the routed MoE output of the layer before joins the residual
 *     (x += routed + shared, chain_ew COMBINE: the CPU's order), the input RMSNorm, the
 *     MLA attention (vkc_mla_qkv: q_a, its norm, q_b, kv_a, the latent norm, RoPE on
 *     the rotated dims, the new latent and rope rows into the device cache; the DSA
 *     indexer on a full layer: the index key (wk, LayerNorm, RoPE) into its cache and,
 *     once the context passes index_topk, every row's top-k (vkc_dsa_select), reused by
 *     the shared layers after it; vkc_mla_attn: the absorbed core over the cache or the
 *     selection, the value rows, o_proj), the residual add, the post-attention RMSNorm.
 *     A dense layer (the first first_k_dense_replace) runs its MLP here too and has no
 *     host step. Then the frame is waited for.
 *   host: the router (moe() reads the normalized rows: its f32 router, sigmoid, bias,
 *     top-k, every routing option it has) and the routed experts (the Vulkan expert
 *     tier's device batch and the CPU's share), without the shared expert; the new KV
 *     rows copied into the host's cache.
 *   device, frame A2 (not waited for): the shared expert, while the host computes the
 *     routed experts.
 * Crossing per sparse layer: the normalized rows (D floats per row) and the new KV rows
 * (kv_lora + qk_rope, and index_head_dim on a full DSA layer) down, the routed sum (D per
 * row) up. The final rows come back to the host, which runs the final norm and lm_head as
 * before (step, step_all and the teacher-forced pass all read them from there).
 *
 * State and who owns it:
 *   - the residual stream: on the device for the whole forward;
 *   - the KV cache (latent, rope key, DSA index keys): the host's stays canonical (every
 *     chain step copies its new rows back), the device holds a mirror per layer behind a
 *     watermark, kv_valid: rows [0, kv_valid) equal the host's for the KV state `owner`.
 *     A step from pos_base uploads [kv_valid, pos_base) first. Every host write lowers it:
 *     the CPU's attention (glmc_cpu_rows, at attention_rows' head), kv_alloc
 *     (glmc_kv_reset), a slot adopting another's rows (glmc_host_rows), a different KV
 *     state bound (the owner check). The mirror grows with the context (powers of two
 *     up to the host's max_t), and a grown mirror is filled again from the host.
 * Past the device's budget (vk_kvsplit.h) each layer keeps only some blocks of its
 * latent and rope rows on the device: the window of the newest, and the blocks the DSA
 * selection reads most. The attention core then runs over those on the device and over
 * the rest on the CPU (from the host's canonical cache) at once, merged through the
 * softmax statistics; the index keys, which every row scores every step, stay whole.
 * MLA has no recurrent state: a rejected draft is rows the next step rewrites, prefix
 * reuse is rows below the watermark, and a device lost mid-forward leaves nothing to
 * rebuild: the forward runs again on the CPU from its input rows (still on the host)
 * and the CPU runs from there (COLI_VK_CHAIN_FAULT=n fakes the loss at the n-th frame).
 * The layers already done in the lost forward count their experts' routing twice in
 * the usage statistics.
 *
 * The chain declines (the CPU path runs, the watermark follows) for a ragged
 * multi-slot decode batch, a layer range (segments), a quantized KV cache (KV8, KV_TQ),
 * PILOT or LOOKA (they read the residual on the host between layers), the exact verify
 * of COLI_EXACT_VERIFY, the CUDA backend, and a model whose matrices or geometry its
 * shaders do not take (int2, E8/IQ3, fp8 dense matrices; kv_lora above 1024, qk_rope
 * above 128, qk_nope above 1024; an indexer above 64 heads or 4096 query floats). The
 * MTP head runs on the CPU, as before: its own KV row is not mirrored.
 *
 * A partial chain (vk_chain.h, vkc_fit; docs/vulkan.md, "A partial chain"): when the
 * device's free memory does not hold every layer, the chain takes layers 0..N-1 and the
 * CPU runs N..L-1, the final norm and lm_head (and the MTP head, eh_proj and lm_head stay
 * on the host unless the whole chain and that tail fit). N is decided at startup, before
 * any upload and before the pins and the RAM cap (glmc_start, from colibri.c's main right
 * after the load); with COLI_VK_DENSE_HOST only the N layers give their host copies back,
 * each once all of it reached the device. A forward runs every chunk of its rows through
 * the N device layers, then hands the rows to the CPU's layer loop from layer N. The
 * crossing: the residual rows (D floats a row) and, when layer N reuses a DSA selection
 * (a shared indexer layer after a full one on the device), that selection (1 + index_topk
 * ints a row), once per chunk. The device's KV mirror, its watermarks and the KV split
 * cover the N layers; the CPU layers' rows are the host's alone. */
#include "vk_chain.h"
#include "vk_kvsplit.h"

typedef struct {
    int ok, failed;
    int n;                                     /* layers on the device: 0..n-1 (the rest run on the CPU) */
    int rows;                                  /* scratch capacity in rows */
    int cap;                                   /* host positions the mirror covers */
    int dev_rows;                              /* device latent/rope rows a layer: cap, or the split's */
    VkcKvSplit ks;                             /* the split past the device's budget (ks.on) */
    VkcMlaCache kvtmp;                         /* the split: a step's new rows before their slots */
    KVState *owner;                            /* the KV state the mirror holds */
    VkcBuf *prm;                               /* norm weights */
    size_t *o_in, *o_post, *o_ixw, *o_ixb;
    VkcMla *mla;                               /* per layer */
    VkcMlaCache *kv; VkcBuf **ik; int *kv_valid;
    VkcMlaScratch sc;
    VkcBuf *x, *nrm, *tmp, *h2, *gs, *us, *hs, *ds, *ikd, *iq, *ihw, *isc, *sel;
    VkcBuf *h2d, *kvd, *xd, *routed, *cs;
    VkcBuf *seld;                              /* a partial chain: the DSA selection layer n reuses, read back */
    size_t kvd_layer;                          /* floats per layer in kvd */
    float *host_routed;
    unsigned long long forwards;
    double host_ms;
} GlmChain;

static int g_vk_chain = 0;     /* COLI_VK_CHAIN decided on, and the chain's pipelines are up */
static int g_glmc_inited = 0; /* vkc_init ran: vkc_shutdown at exit, whatever came after */
static VkcFit g_glmc_fit;      /* how many layers the device holds (vkc_fit), when g_glmc_fitted */
static int g_glmc_fitted;
static size_t g_glmc_lazy;     /* what the chain allocates at its first forward: one chunk's
                                * scratch and the N layers' KV mirrors at their first size */
static size_t *g_glmc_mir;     /* each layer's KV mirror at its first size */

/* The device's copy of a resident QT, uploaded once (the per-matrix path's own copy
 * when COLI_VK_DENSE put it there); f32 goes up as the backend's fmt 10. NULL: no
 * device form (int2, E8/IQ3, fp8, planar int4). */
static ColiVkTensor *glmc_tensor(QT *t) {
    if (t->vk) return t->vk;
    int fmt; const void *w;
    switch (t->fmt) {
    case 0: fmt = 10; w = t->qf; break;
    case 1: fmt = 1; w = t->q8; break;
    case 2: if (t->planar) return NULL; fmt = 2; w = t->q4; break;
    case 4: if (t->planar || t->gs < 8 || t->gs % 8) return NULL; fmt = 4; w = t->q4; break;
    case 5: fmt = 5; w = t->q4; break;
    default: return NULL;
    }
    if (!w || t->O <= 0 || t->I <= 0) return NULL;
    return coli_vk_tensor_ensure(&t->vk, w, t->fmt == 0 ? NULL : t->s, fmt, t->I, t->O, t->fmt == 4 ? t->gs : 0) ? t->vk : NULL;
}
/* glmc_tensor's backend format for t without uploading it (-1: no device form), and its
 * group size; the fit counts the bytes from it. */
static int glmc_form(const QT *t, int *gs) {
    *gs = 0;
    if (t->O <= 0 || t->I <= 0) return -1;
    switch (t->fmt) {
    case 0: return t->qf ? 10 : -1;
    case 1: return t->q8 ? 1 : -1;
    case 2: return t->planar || !t->q4 ? -1 : 2;
    case 4: if (t->planar || t->gs < 8 || t->gs % 8 || !t->q4) return -1; *gs = t->gs; return 4;
    case 5: return t->q4 ? 5 : -1;
    default: return -1;
    }
}
static int glmc_full(const Model *m, int i);
/* Layer i's matrices on the device (the chain's set; up to 11) */
static int glmc_layer_qts(Model *m, int i, QT **q) {
    Layer *l = &m->L[i]; int n = 0;
    q[n++] = &l->q_a; q[n++] = &l->q_b; q[n++] = &l->kv_a; q[n++] = &l->kv_b; q[n++] = &l->o;
    if (l->sparse) { q[n++] = &l->sh_gate; q[n++] = &l->sh_up; q[n++] = &l->sh_down; }
    else { q[n++] = &l->gate_proj; q[n++] = &l->up_proj; q[n++] = &l->down_proj; }
    if (glmc_full(m, i)) { q[n++] = &m->ix_wq[i]; q[n++] = &m->ix_wk[i]; q[n++] = &m->ix_wp[i]; }
    return n;
}

/* ---- the host's writes to its KV cache: the mirror's watermark follows ------------ */
/* The CPU's attention writes rows [pos..] of each row's KV state for this layer. */
static void glmc_cpu_rows(Model *m, int layer, KVState *const *kvs, const int *positions, int pos_base, int S) {
    GlmChain *ch = (GlmChain *)m->vkchain;
    if (!ch || !ch->kv_valid || layer < 0 || layer >= ch->n) return;   /* a CPU layer of a partial chain: no mirror */
    for (int s = 0; s < S; s++) {
        KVState *ks = kvs ? kvs[s] : m->kv;
        int pos = positions ? positions[s] : pos_base + s;
        if (ks == ch->owner && ch->kv_valid[layer] > pos) ch->kv_valid[layer] = pos;
        if (ks == ch->owner) vkc_kv_lower(&ch->ks, layer, pos);
    }
}
/* Rows from `from` of every layer of k were written on the host. */
static void glmc_host_rows(Model *m, KVState *k, int from) {
    GlmChain *ch = (GlmChain *)m->vkchain;
    if (!ch || !ch->kv_valid || k != ch->owner) return;
    for (int i = 0; i < ch->n; i++) {
        if (ch->kv_valid[i] > from) ch->kv_valid[i] = from < 0 ? 0 : from;
        vkc_kv_lower(&ch->ks, i, from < 0 ? 0 : from);
    }
}
/* kv_alloc gave the bound state new arrays: nothing on the device is theirs. */
static void glmc_kv_reset(Model *m) {
    GlmChain *ch = (GlmChain *)m->vkchain;
    if (!ch || !ch->kv_valid) return;
    for (int i = 0; i < ch->n; i++) ch->kv_valid[i] = 0;
    vkc_kv_reset(&ch->ks);
    ch->owner = NULL;
}

/* ---- setup ---------------------------------------------------------------------------- */
static int glmc_full(const Model *m, int i) { return m->has_dsa && m->c.idx_type[i]; }

/* What the chain's shaders do not take (NULL: nothing), from the shapes alone: no upload. */
static const char *glmc_check(Model *m) {
    Cfg *c = &m->c;
    if (c->kv_lora > 1024 || c->qk_rope > 128 || (c->qk_rope & 1) || c->qk_nope > 1024 || c->q_lora < 1)
        return "an attention geometry its shaders do not take";
    if (m->has_dsa && (c->index_nh > 64 || c->index_nh * c->index_hd > 4096 || c->index_hd < c->qk_rope || !c->idx_type[0]))
        return "an indexer its shaders do not take";
    QT *q[11]; int gs;
    for (int i = 0; i < c->n_layers; i++) {
        int n = glmc_layer_qts(m, i, q);
        for (int k = 0; k < n; k++)
            if (glmc_form(q[k], &gs) < 0) return "a matrix with no device form (int2, E8/IQ3, fp8, planar int4)";
    }
    return NULL;
}

/* glmc_res counts instead of reserving while g_glmc_count >= 0 (the chunk's sizing); with
 * g_glmc_count_fit, each buffer at the bytes the device gives it (the fit's scratch) */
static long long g_glmc_count = -1;
static int g_glmc_count_fit;
static int glmc_res(VkcBuf **b, size_t floats, int kind) {
    size_t bytes = (floats ? floats : 1) * sizeof(float);
    if (g_glmc_count >= 0) { g_glmc_count += (long long)(g_glmc_count_fit ? vkc_fit_buf(bytes) : bytes); return 1; }
    return vkc_reserve(b, bytes, kind);
}
static int glmc_scratch(GlmChain *ch, Model *m, int rows, int ctx);

/* The fit (vkc_fit), before any upload: each layer's device bytes (its matrices as
 * glmc_tensor uploads them, its share of the parameter arena, its KV mirror at the first
 * size glmc_mirror gives it), the chain's fixed bytes (the scratch of one prompt chunk of
 * vkc_fit_rows(256) rows at the context CTX, the arena's alignment) and the tail: what
 * goes up only with every layer there (the MTP layer, eh_proj and lm_head, which
 * COLI_VK_DENSE_HOST and the per-matrix path place). */
static void glmc_fit_plan(Model *m) {
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, ID = c->index_hd;
    size_t *lb = calloc(L ? L : 1, sizeof(size_t)), *mb = calloc(L ? L : 1, sizeof(size_t));
    size_t *mir = calloc(L ? L : 1, sizeof(size_t));
    if (!lb || !mb || !mir) { free(lb); free(mb); free(mir); return; }
    const int cap0 = 256;   /* glmc_mirror's first size */
    for (int i = 0; i < L; i++) {
        QT *q[11]; int gs, n = glmc_layer_qts(m, i, q);
        for (int k = 0; k < n; k++) {
            int f = glmc_form(q[k], &gs);
            lb[i] += vkc_fit_tensor(f, q[k]->I, q[k]->O, gs);
            mb[i] += coli_vk_tensor_payload(f, q[k]->I, q[k]->O, gs);
        }
        lb[i] += (size_t)(2 * D + c->q_lora + c->kv_lora + (glmc_full(m, i) ? 2 * ID : 0)) * sizeof(float);
        mir[i] = vkc_fit_buf((size_t)cap0 * c->kv_lora * sizeof(float)) +
                 (c->qk_rope > 0 ? vkc_fit_buf((size_t)cap0 * c->qk_rope * sizeof(float)) : 0) +
                 (glmc_full(m, i) ? vkc_fit_buf((size_t)cap0 * ID * sizeof(float)) : 0);
        lb[i] += mir[i];
    }
    /* one chunk's scratch: glmc_scratch's buffers (every layer's KV rows down) and the MLA
     * scratch, at the context the run plans for (the DSA scores: rows x context, at most
     * the 32M floats the forward allows a selecting chunk) */
    int R = vkc_fit_rows(256), ctx = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    if (ctx < 1) ctx = 4096;
    if ((long long)R * ctx > (32LL << 20)) ctx = (int)((32LL << 20) / R);
    if (ctx < 1) ctx = 1;
    GlmChain tmp; memset(&tmp, 0, sizeof tmp); tmp.n = L;
    g_glmc_count = 0; g_glmc_count_fit = 1; glmc_scratch(&tmp, m, R, ctx);
    size_t scratch = (size_t)g_glmc_count;
    g_glmc_count = -1; g_glmc_count_fit = 0;
    size_t r = (size_t)R, H = (size_t)c->n_heads;
    scratch += vkc_fit_buf(r * c->q_lora * 4) + vkc_fit_buf(r * H * (c->qk_nope + c->qk_rope) * 4) +
               vkc_fit_buf(r * (c->kv_lora + c->qk_rope) * 4) + 2 * vkc_fit_buf(r * H * c->kv_lora * 4) +
               vkc_fit_buf(r * H * c->v_head * 4);
    size_t fixed = scratch + vkc_fit_buf(4);   /* + the arena's alignment */
    size_t tail = 0;
    {
        QT *t[10]; int nt = 0, gs;
        if (m->has_mtp) {
            Layer *l = &m->mtpL;
            t[nt++] = &l->q_a; t[nt++] = &l->q_b; t[nt++] = &l->kv_a; t[nt++] = &l->kv_b; t[nt++] = &l->o;
            t[nt++] = &l->sh_gate; t[nt++] = &l->sh_up; t[nt++] = &l->sh_down; t[nt++] = &m->eh_proj;
        }
        t[nt++] = &m->lm_head;
        for (int k = 0; k < nt; k++) {
            int f = glmc_form(t[k], &gs);
            if (f >= 0) tail += vkc_fit_tensor(f, t[k]->I, t[k]->O, gs);
        }
    }
    vkc_fit("colibri", L, lb, mb, fixed, tail, &g_glmc_fit);
    g_glmc_fitted = 1;
    g_glmc_lazy = scratch;
    g_glmc_mir = mir;
    free(lb); free(mb);
}

/* The parameters and layers 0..N-1 on the device (N = the fit's), layer by layer: each
 * layer's matrices, then (COLI_VK_DENSE_HOST) its host copies given back once all of it is
 * there. A layer that does not reach the device (a driver refusing what the budget
 * promised, a failed upload) is freed whole and the chain keeps the layers before it
 * (vkc_fit_shrink); nothing of it stays on the device and its host copies were never
 * dropped. 0 = no layer on the device: the chain stays off. */
static int glmc_setup(Model *m) {
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, N = g_glmc_fit.n;
    GlmChain *ch = (GlmChain *)calloc(1, sizeof *ch);
    if (!ch) return 0;
    ch->o_in = calloc(L, sizeof(size_t)); ch->o_post = calloc(L, sizeof(size_t));
    ch->o_ixw = calloc(L, sizeof(size_t)); ch->o_ixb = calloc(L, sizeof(size_t));
    ch->mla = calloc(L, sizeof(VkcMla)); ch->kv = calloc(L, sizeof(VkcMlaCache));
    ch->ik = calloc(L, sizeof(VkcBuf *)); ch->kv_valid = calloc(L, sizeof(int));
    if (!ch->o_in || !ch->o_post || !ch->o_ixw || !ch->o_ixb || !ch->mla || !ch->kv || !ch->ik || !ch->kv_valid) return 0;
    m->vkchain = ch;
    /* the parameter arena of the N layers: in_ln, post_ln, the q and kv latent norms, the
     * index key norm */
    size_t n = 0, *o_qn = calloc(L, sizeof(size_t)), *o_kn = calloc(L, sizeof(size_t));
    if (!o_qn || !o_kn) { free(o_qn); free(o_kn); return 0; }
    for (int i = 0; i < N; i++) {
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D;
        o_qn[i] = n; n += c->q_lora; o_kn[i] = n; n += c->kv_lora;
        if (glmc_full(m, i)) { ch->o_ixw[i] = n; n += c->index_hd; ch->o_ixb[i] = n; n += c->index_hd; }
    }
    float *arena = calloc(n ? n : 1, sizeof(float));
    if (!arena) { free(o_qn); free(o_kn); return 0; }
    for (int i = 0; i < N; i++) {
        Layer *l = &m->L[i];
        memcpy(arena + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(arena + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(arena + o_qn[i], l->q_a_ln, c->q_lora * sizeof(float));
        memcpy(arena + o_kn[i], l->kv_a_ln, c->kv_lora * sizeof(float));
        if (glmc_full(m, i)) {
            memcpy(arena + ch->o_ixw[i], m->ix_knw[i], c->index_hd * sizeof(float));
            memcpy(arena + ch->o_ixb[i], m->ix_knb[i], c->index_hd * sizeof(float));
        }
    }
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, n * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) {
        vkc_free(ch->prm); ch->prm = NULL;
        free(o_qn); free(o_kn);
        vkc_fit_shrink("colibri", &g_glmc_fit, 0, "the parameters' buffer was refused");
        return 0;
    }
    /* the tensors: the same device copies the per-matrix path uses */
    for (int i = 0; i < N; i++) {
        Layer *l = &m->L[i];
        QT *q[11]; int nq = glmc_layer_qts(m, i, q), up = 1;
        for (int k = 0; k < nq && up; k++) up = glmc_tensor(q[k]) != NULL;
        if (!up) {   /* free what it placed: the layer runs on the CPU, from its host copies */
            ColiVkTensor **t[11];
            for (int k = 0; k < nq; k++) t[k] = &q[k]->vk;
            vkc_layer_free(t, nq, NULL, 0);
            vkc_fit_shrink("colibri", &g_glmc_fit, i, vkc_lost() ? "the device was lost" : "a matrix was refused");
            break;
        }
        ch->mla[i] = (VkcMla){c->n_heads, c->qk_nope, c->qk_rope, c->v_head, c->kv_lora, D, c->q_lora, c->eps, c->attn_scale,
                              VKC_ROPE_INTERLEAVED, l->q_a.vk, l->q_b.vk, l->kv_a.vk, l->kv_b.vk, NULL, NULL, l->o.vk,
                              ch->prm, o_qn[i], o_kn[i]};
        glm_dho_drop_layer(m, i);   /* COLI_VK_DENSE_HOST: all of the layer is there, its host copies go */
        vkc_fit_mark(&g_glmc_fit, i);
    }
    free(o_qn); free(o_kn);
    ch->n = g_glmc_fit.n;
    if (!ch->n) { vkc_free(ch->prm); ch->prm = NULL; return 0; }
    ch->ok = 1;
    int nfull = 0, ndense = 0;
    for (int i = 0; i < ch->n; i++) { nfull += glmc_full(m, i); ndense += !m->L[i].sparse; }
    fprintf(stderr, "[VK] colibri chain: %d layers on the device (%d dense, %d with the DSA indexer), %.1f MiB of parameters\n",
            ch->n, ndense, nfull, n * 4 / 1048576.0);
    return 1;
}

/* Prompt rows per chunk (vkc_chunk_rows): the chain's scratch a row, counted from the
 * reservations (the MLA scratch as vkc_mla_scratch sizes it; the DSA scores at the
 * model's context), and the routed experts' outputs (the tier's rows, the CPU's
 * contributions, and the host's sum). */
static int glmc_chunk_rows(GlmChain *ch, Model *m) {
    if (!vkc_chunk_auto()) return vkc_chunk_rows("colibri", 0);
    const VkcMla *a = &ch->mla[0];
    size_t mla = (size_t)(a->q_lora > 0 ? a->q_lora : 1) + (size_t)a->H * (a->Q + a->R) + (size_t)(a->K + a->R) +
                 (size_t)a->H * a->K + (size_t)a->H * a->K + (size_t)a->H * a->V;
    size_t ksz = (size_t)ch->kvd_layer;
    g_glmc_count = 0; glmc_scratch(ch, m, 1, m->max_t); long long b1 = g_glmc_count;
    g_glmc_count = 0; glmc_scratch(ch, m, 2, m->max_t); long long b2 = g_glmc_count;
    g_glmc_count = -1; ch->kvd_layer = ksz;
    size_t row = (size_t)(b2 - b1) + mla * sizeof(float) + (size_t)(2 * m->c.topk + 2) * m->c.hidden * sizeof(float);
    /* moe_vk's CPU expert gather/gate/up/output workspace, plus the optional
     * second device's packed inputs and outputs. */
    row += 2 * ((size_t)m->c.hidden + m->c.moe_inter) * sizeof(float);
    if (g_vk_reg_n2 > 0) row += 2 * (size_t)m->c.topk * m->c.hidden * sizeof(float);
    return vkc_chunk_rows("colibri", row);
}

/* scratch for `rows` rows at contexts up to `ctx` positions */
static int glmc_scratch(GlmChain *ch, Model *m, int rows, int ctx) {
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, IH = c->index_nh, ID = c->index_hd;
    int SI = c->moe_inter * c->n_shared, DI = c->dense_inter, MI = SI > DI ? SI : DI;
    size_t r = (size_t)rows;
    ch->kvd_layer = r * (c->kv_lora + c->qk_rope + (m->has_dsa ? ID : 0));
    int ok = (g_glmc_count >= 0 || vkc_mla_scratch(&ch->sc, &ch->mla[0], rows)) &&
             glmc_res(&ch->x, r * D, VKC_DEV) && glmc_res(&ch->nrm, r * D, VKC_DEV) && glmc_res(&ch->tmp, r * D, VKC_DEV) &&
             glmc_res(&ch->h2, r * D, VKC_DEV) && glmc_res(&ch->gs, r * MI, VKC_DEV) && glmc_res(&ch->us, r * MI, VKC_DEV) &&
             glmc_res(&ch->hs, r * MI, VKC_DEV) && glmc_res(&ch->ds, r * D, VKC_DEV) &&
             glmc_res(&ch->h2d, r * D, VKC_DOWN) && glmc_res(&ch->kvd, (size_t)ch->n * ch->kvd_layer, VKC_DOWN) &&
             glmc_res(&ch->xd, r * D, VKC_DOWN) && glmc_res(&ch->routed, r * D, VKC_UP) &&
             glmc_res(&ch->cs, r * (c->qk_rope > 0 ? c->qk_rope : 2), VKC_UP);
    if (ok && ch->ks.on && g_glmc_count >= 0)
        g_glmc_count += (long long)r * (3LL * c->n_heads * (c->kv_lora + 2) +
                           (long long)c->n_heads * (c->kv_lora + c->qk_rope) + c->kv_lora + c->qk_rope +
                           (m->has_dsa ? 1 + c->index_topk : 0)) * sizeof(float);
    else if (ok && ch->ks.on)   /* the split: the partial results, a step's new rows before their slots */
        ok = vkc_kv_parts(&ch->ks, r * c->n_heads * (c->kv_lora + 2)) &&
             (ch->kvtmp.cap >= rows || (glmc_res(&ch->kvtmp.lat, r * c->kv_lora, VKC_DEV) &&
                                        (c->qk_rope == 0 || glmc_res(&ch->kvtmp.rope, r * c->qk_rope, VKC_DEV)) &&
                                        (ch->kvtmp.cap = rows)));
    if (ok && m->has_dsa)
        ok = glmc_res(&ch->ikd, r * ID, VKC_DEV) && glmc_res(&ch->iq, r * IH * ID, VKC_DEV) &&
             glmc_res(&ch->ihw, r * IH, VKC_DEV) && glmc_res(&ch->isc, r * (size_t)ctx, VKC_DEV) &&
             glmc_res(&ch->sel, r * (1 + (size_t)c->index_topk), VKC_DEV);
    /* a partial chain whose CPU layer n reuses the selection of a device layer: it comes
     * back with the rows (the fit counts it whenever there is an indexer) */
    if (ok && m->has_dsa && (g_glmc_count_fit || (ch->n < L && !glmc_full(m, ch->n))))
        ok = glmc_res(&ch->seld, r * (1 + (size_t)c->index_topk), VKC_DOWN);
    if (!ok || g_glmc_count >= 0) return ok;   /* counting: the buffers only */
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, r * D * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    return 1;
}

/* The device's KV mirror: room for `need` positions of the bound state, and the
 * watermarks reset when the state is not the one mirrored. A mirror that grows is
 * planned again (vkc_kv_plan): whole when it fits the device's budget, else the split
 * (latent and rope rows; the index keys stay whole), for steps of `rows` rows. */
static int glmc_mirror(GlmChain *ch, Model *m, int need, int rows) {
    Cfg *c = &m->c; int L = ch->n;   /* the device's layers (a partial chain: the first n) */
    if (ch->owner != m->kv) { for (int i = 0; i < L; i++) ch->kv_valid[i] = 0; ch->owner = m->kv; vkc_kv_reset(&ch->ks); }
    if (ch->cap >= need) return 1;
    int cap = 256; while (cap < need) cap *= 2;
    if (cap > m->max_t) cap = m->max_t;
    if (cap < need) return 0;
    size_t row = (size_t)(c->kv_lora + c->qk_rope) * sizeof(float);
    if (!vkc_kv_plan(&ch->ks, "colibri", L, row, cap, rows, m->has_dsa, (size_t)L * ch->dev_rows * row)) return 0;
    int dr = ch->ks.on ? ch->ks.rows : cap;
    for (int i = 0; i < L; i++) {
        vkc_free(ch->kv[i].lat); vkc_free(ch->kv[i].rope); vkc_free(ch->ik[i]);
        ch->kv[i] = (VkcMlaCache){NULL, NULL, 0}; ch->ik[i] = NULL; ch->kv_valid[i] = 0;
    }
    ch->cap = 0; ch->dev_rows = 0;
    for (int i = 0; i < L; i++) {
        ch->kv[i].lat = vkc_buf((size_t)dr * c->kv_lora * sizeof(float), VKC_DEV);
        ch->kv[i].rope = c->qk_rope > 0 ? vkc_buf((size_t)dr * c->qk_rope * sizeof(float), VKC_DEV) : NULL;
        ch->kv[i].cap = dr;
        if (glmc_full(m, i)) ch->ik[i] = vkc_buf((size_t)cap * c->index_hd * sizeof(float), VKC_DEV);
        if (!ch->kv[i].lat || (c->qk_rope > 0 && !ch->kv[i].rope) || (glmc_full(m, i) && !ch->ik[i])) return 0;
    }
    ch->cap = cap; ch->dev_rows = dr;
    return 1;
}

/* Record the uploads that make the mirror the host's below pos_base, for a step of
 * n_rows rows (the split: its window placed first). */
static int glmc_push_kv(GlmChain *ch, Model *m, int pos_base, int n_rows) {
    Cfg *c = &m->c; int ok = 1, K = c->kv_lora, R = c->qk_rope, ID = c->index_hd;
    for (int i = 0; i < ch->n && ok; i++) {
        if (ch->ks.on) {      /* the split: the window placed, its rows below pos_base uploaded */
            VkcKvPart pt[2] = {{1, K, m->Lc[i], 0, ch->kv[i].lat, 0}, {1, R, m->Rc[i], 0, ch->kv[i].rope, 0}};
            vkc_kv_place(&ch->ks, i, pos_base, n_rows);
            ok = vkc_kv_push(&ch->ks, i, pt, R ? 2 : 1, pos_base);
            int t0 = ch->kv_valid[i], n = pos_base - t0;   /* the index keys stay whole */
            if (ok && n > 0 && glmc_full(m, i))
                ok = vkc_write(ch->ik[i], (size_t)t0 * ID, coli_kv_row(m->Ic[i], t0, ID), (size_t)n * ID * sizeof(float));
            ch->kv_valid[i] = pos_base;
            continue;
        }
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        if (n <= 0) continue;
        ok = vkc_write(ch->kv[i].lat, (size_t)t0 * K, coli_kv_row(m->Lc[i], t0, K), (size_t)n * K * sizeof(float)) &&
             (R == 0 || vkc_write(ch->kv[i].rope, (size_t)t0 * R, coli_kv_row(m->Rc[i], t0, R), (size_t)n * R * sizeof(float))) &&
             (!glmc_full(m, i) || vkc_write(ch->ik[i], (size_t)t0 * ID, coli_kv_row(m->Ic[i], t0, ID), (size_t)n * ID * sizeof(float)));
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}
/* The new rows of layers [from, to) into the host's cache (after their frame's wait). */
static void glmc_pull_kv(GlmChain *ch, Model *m, int from, int to, int pb, int n) {
    Cfg *c = &m->c; int K = c->kv_lora, R = c->qk_rope, ID = c->index_hd;
    for (int i = from; i < to; i++) {
        const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)i * ch->kvd_layer;
        for (int s = 0; s < n; s++) {
            memcpy(coli_kv_row(m->Lc[i], pb + s, K), kv + (size_t)s * K, K * sizeof(float));
            if (R) memcpy(coli_kv_row(m->Rc[i], pb + s, R), kv + (size_t)n * K + (size_t)s * R, R * sizeof(float));
            if (glmc_full(m, i))
                memcpy(coli_kv_row(m->Ic[i], pb + s, ID), kv + (size_t)n * (K + R) + (size_t)s * ID, ID * sizeof(float));
        }
    }
}

/* ---- one layer's pieces ---------------------------------------------------------- */
static int glmc_norm(VkcBuf *x, VkcBuf *w, size_t wo, VkcBuf *y, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, 0, D, D, 0, D, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
/* The DSA indexer of a full layer: the rows' index keys into the cache (and the host's
 * copy), then, when a row's context passes index_topk (or DSA_FORCE), every row's
 * selection. *active: the selection list holds this layer's choice. */
static int glmc_dsa(GlmChain *ch, Model *m, int i, int n, int pb, int *active) {
    Cfg *c = &m->c; int IH = c->index_nh, ID = c->index_hd, R = c->qk_rope;
    size_t kof = (size_t)i * ch->kvd_layer + (size_t)n * (c->kv_lora + R);
    VkcMlaRow ln = {n, 1, 0, ID, 0, 0, ID, 0, pb * ID, ID, 0, 0, 0, (int)ch->o_ixw[i], (int)ch->o_ixb[i], 1, 1e-6f};
    VkcMlaRow rk = {n, 1, R, ID, VKC_ROPE_INTERLEAVED, pb * ID, ID, 0, pb * ID, ID, 0, 0, R, 0, 0, 0, 0.f};
    int ok = vkc_matmul(glmc_tensor(&m->ix_wk[i]), ch->nrm, 0, ch->ikd, 0, n) &&
             vkc_mla_lnorm(ch->ikd, ch->prm, ch->ik[i], &ln) &&
             (R == 0 || vkc_mla_rope(ch->ik[i], ch->cs, ch->ik[i], &rk)) &&
             vkc_copy(ch->kvd, kof, ch->ik[i], (size_t)pb * ID, (size_t)n * ID);
    *active = pb + n > c->index_topk || g_dsa_force;
    if (!ok || !*active) return ok;
    VkcMlaRow rq = {n * IH, IH, R, ID, VKC_ROPE_INTERLEAVED, 0, IH * ID, ID, 0, IH * ID, ID, 0, R, 0, 0, 0, 0.f};
    VkcDsa ds = {n, pb, IH, ID, c->index_topk, g_dsa_force, 0, IH * ID, 0, IH, 0, ID, pb + n, 1 + c->index_topk,
                 1.f / sqrtf((float)ID), 1.f / sqrtf((float)IH)};
    return vkc_matmul(glmc_tensor(&m->ix_wq[i]), ch->sc.qa, 0, ch->iq, 0, n) &&
           (R == 0 || vkc_mla_rope(ch->iq, ch->cs, ch->iq, &rq)) &&
           vkc_matmul(glmc_tensor(&m->ix_wp[i]), ch->nrm, 0, ch->ihw, 0, n) &&
           vkc_dsa_select(ch->iq, ch->ihw, ch->ik[i], ch->isc, ch->sel, &ds);
}
/* silu(gate(h)) * up(h), then down: the shared expert (A2) or a dense layer's MLP */
static int glmc_mlp(GlmChain *ch, QT *g, QT *u, QT *d, VkcBuf *in, VkcBuf *out, int n) {
    int I = g->O;
    VkcEw p = {VKC_EW_SWIGLU, n * I, I, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_matmul(glmc_tensor(g), in, 0, ch->gs, 0, n) && vkc_matmul(glmc_tensor(u), in, 0, ch->us, 0, n) &&
           vkc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &p) && vkc_matmul(glmc_tensor(d), ch->hs, 0, out, 0, n);
}
/* x += routed + shared, as moe() leaves `out` (routed, then the shared expert) and
 * layer_forward_rows adds it */
static int glmc_combine(GlmChain *ch, Model *m, int n) {
    VkcEw p = {VKC_EW_COMBINE, n * m->c.hidden, m->c.hidden, 1, 1 | 2, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->x, ch->x, ch->routed, ch->ds, NULL, &p);
}

/* The device's layers (every layer, or the first n of a partial chain) for S rows of xh
 * (positions pos_base..), the rows after the last of them back into xh. Returns how many
 * layers ran: L, or n (the caller's CPU loop runs layers n..L-1 from xh); 0 = not taken:
 * the CPU runs the forward (from xh, untouched). */
static int glmc_forward(Model *m, float *xh, int S, int pos_base) {
    if (!g_vk_chain) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && S <= 2) return 0;   /* prompts only: decode on the CPU */
    if (g_pilot || g_pilot_real || g_looka || g_kv8 || g_tq || (exact_verify_on() && g_spec_live)) return 0;
    GlmChain *ch = (GlmChain *)m->vkchain;
    if (!ch || !ch->ok || ch->failed || vkc_lost()) return 0;
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, R = c->qk_rope, N = ch->n;
    if (!m->Lc || !m->Rc || (m->has_dsa && !m->Ic)) return 0;
    for (int i = 0; i < L; i++) if (m->kv_start[i] != 0 || !m->Lc[i]) return 0;
    /* layer N on the CPU reuses the DSA selection of the device's last full layer (a
     * shared indexer layer): the selection comes back with the rows, into the CPU's */
    int hand_sel = N < L && m->has_dsa && !glmc_full(m, N), TK = c->index_topk;
    if (hand_sel && (int64_t)S * TK > m->dsa_scap) {
        int *sl = malloc((size_t)S * TK * sizeof(int)), *ns = malloc((size_t)S * sizeof(int));
        if (!sl || !ns) { free(sl); free(ns); return 0; }
        free(m->dsa_sel); free(m->dsa_nsel);
        m->dsa_sel = sl; m->dsa_nsel = ns; m->dsa_scap = (int)((int64_t)S * TK);
    }
    int mirror_ok = glmc_mirror(ch, m, pos_base + S, 1);
    int CH = mirror_ok ? glmc_chunk_rows(ch, m) : 1;
    if (m->has_dsa && (pos_base + S > c->index_topk || g_dsa_force)) {   /* the scores' scratch: rows x context */
        int64_t lim = ((int64_t)32 << 20) / (pos_base + S);
        if (lim < CH) CH = lim < 1 ? 1 : (int)lim;
    }
    int rows = S < CH ? S : CH;
    if (ch->ks.on && rows > ch->ks.chunk) rows = ch->ks.chunk;   /* a step's rows fit the split's window */
    if (!mirror_ok || !glmc_scratch(ch, m, rows, pos_base + S)) {
        fprintf(stderr, "[VK] colibri chain: device memory for %d rows at %d positions refused; per-matrix path\n",
                rows, pos_base + S);
        ch->failed = 1;
        return 0;
    }
    vkc_gemm_rows(-1);
    /* the final rows wait here until every chunk is through: a lost device leaves xh the
     * forward's input, for the CPU to run again */
    float *outs = malloc((size_t)S * D * sizeof(float));
    if (!outs) return 0;
    float *inv = R ? malloc((size_t)(R / 2) * sizeof(float)) : NULL;
    for (int j = 0; j < R / 2; j++) inv[j] = powf(c->theta, -2.0f * j / R);   /* rope_interleave's frequencies */
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        for (int s = 0; s < n && R; s++)    /* the CPU's own angles, cosines and sines */
            for (int j = 0; j < R / 2; j++) {
                float ang = (pb + s) * inv[j];
                float *cs = (float *)vkc_ptr(ch->cs) + (size_t)s * R + 2 * j;
                cs[0] = cosf(ang); cs[1] = sinf(ang);
            }
        if (!vkc_begin() || !vkc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !glmc_push_kv(ch, m, pb, n)) goto lost;
        int ok = 1, pending = 0, pulled = 0, sel_on = 0;
        for (int i = 0; i < N && ok; i++) {
            Layer *l = &m->L[i];
            if (g_spec && g_prefetch && l->sparse && m->enr[i] > 0)      /* the CPU path's I/O hint */
                for (int z = 0; z < m->enr[i]; z++) if (!(n <= 4 && vk_reg_served(i, m->eroute[i][z]))) expert_prefetch(m, i, m->eroute[i][z]);
            if (pending) { ok = glmc_combine(ch, m, n); pending = 0; }
            ok = ok && glmc_norm(ch->x, ch->prm, ch->o_in[i], ch->nrm, n, D, c->eps) &&
                 (ch->ks.on ? vkc_kv_mla_qkv(&ch->ks, i, &ch->mla[i], &ch->sc, ch->nrm, 0, n, pb, ch->cs, &ch->kv[i], &ch->kvtmp,
                                             ch->kvd, (size_t)i * ch->kvd_layer)
                            : vkc_mla_qkv(&ch->mla[i], &ch->sc, ch->nrm, 0, n, pb, ch->cs, &ch->kv[i], ch->kvd,
                                          (size_t)i * ch->kvd_layer));
            if (ok && glmc_full(m, i)) ok = glmc_dsa(ch, m, i, n, pb, &sel_on);
            if (ok && ch->ks.on)   /* the split: the core over the device's blocks and the host's rows */
                ok = vkc_kv_mla_attn(&ch->ks, i, &ch->mla[i], &ch->sc, n, pb, 0, &ch->kv[i], sel_on ? ch->sel : NULL, 0,
                                     1 + c->index_topk, NULL, 0, ch->tmp, 0, m->Lc[i], m->Rc[i]);
            else ok = ok && vkc_mla_attn(&ch->mla[i], &ch->sc, n, pb, 0, &ch->kv[i], sel_on ? ch->sel : NULL, 0,
                                         1 + c->index_topk, NULL, 0, ch->tmp, 0);
            VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = ok && vkc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add) &&
                 glmc_norm(ch->x, ch->prm, ch->o_post[i], ch->h2, n, D, c->eps);
            if (!ok) break;
            if (!l->sparse) {                                  /* a dense layer: its MLP here, no host step */
                ok = glmc_mlp(ch, &l->gate_proj, &l->up_proj, &l->down_proj, ch->h2, ch->tmp, n) &&
                     vkc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
                continue;
            }
            /* A1; while it runs, the tier loads the experts this layer will likely
             * stream (a big prompt chunk only) */
            ok = vkc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D) && vkc_submit(0);
            if (ok) vkt_stream_prefetch(i, n);
            ok = ok && vkc_finish();
            if (!ok) break;
            glmc_pull_kv(ch, m, pulled, i + 1, pb, n); pulled = i + 1;
            /* A2: the shared expert, while the host computes the routed experts */
            ok = vkc_begin() && glmc_mlp(ch, &l->sh_gate, &l->sh_up, &l->sh_down, ch->h2, ch->ds, n) && vkc_submit(0);
            double t1 = now_s();
            moe(m, l, i, (float *)vkc_ptr(ch->h2d), n, ch->host_routed, 0);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok && pending) ok = glmc_combine(ch, m, n);
        ok = ok && vkc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D) &&
             (!hand_sel || !sel_on || vkc_copy(ch->seld, 0, ch->sel, 0, (size_t)n * (1 + TK))) && vkc_submit(1);
        if (!ok) goto lost;
        glmc_pull_kv(ch, m, pulled, N, pb, n);
        memcpy(outs + (size_t)c0 * D, vkc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        for (int s = 0; hand_sel && s < n; s++) {   /* in the CPU's form: a count (0 = every position), the positions */
            const int *sr = (const int *)vkc_ptr(ch->seld) + (size_t)s * (1 + TK);
            int k = sel_on && sr[0] > 0 ? (sr[0] < TK ? sr[0] : TK) : 0;
            if (k) memcpy(m->dsa_sel + (size_t)(c0 + s) * TK, sr + 1, (size_t)k * sizeof(int));
            m->dsa_nsel[c0 + s] = k;
        }
        for (int i = 0; i < N; i++) { ch->kv_valid[i] = pb + n; vkc_kv_done(&ch->ks, i, pb + n); }
    }
    memcpy(xh, outs, (size_t)S * D * sizeof(float));
    free(inv); free(outs);
    ch->forwards++;
    return N;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    free(inv); free(outs);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    g_vk_chain = 0; ch->failed = 1;
    fprintf(stderr, "[VK] colibri chain: the device was lost; the CPU runs this forward again and from here on "
                    "(the KV cache is the host's: there is no state to rebuild)\n");
    return 0;
}

static void glmc_report(Model *m) {
    GlmChain *ch = m ? (GlmChain *)m->vkchain : NULL;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] colibri chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
    vkc_kv_report(&ch->ks);
    vkc_prof_print();
}
static Model *g_glmc_model;
static void glmc_report_atexit(void) { glmc_report(g_glmc_model); }

/* COLI_VK_CHAIN at startup, right after the load: before the pins, the RAM cap and the
 * tier (the trunk's device copies count as used): the decision, the pipelines, the fit
 * (how many layers the device holds), COLI_VK_DENSE_HOST's decision for those layers
 * (glm_dho_start), the tensors layer by layer, then the tail's host copies and the RAM
 * given back (glm_dho_finish). Without a fit (the chain off) glm_dho_start does as before. */
static void glmc_start(Model *m) {
    if (!g_vulkan) return;
    int tier_on = vkt_wanted() && g_vk_experts != 0 && m->c.n_experts > 0;
    int on = coli_vk_chain_decide("colibri", tier_on, COLI_VK_CHAIN_UNMEASURED);
    const char *no = NULL;
#ifdef COLI_CUDA
    if (on && g_cuda_enabled) no = "the CUDA backend is on and keeps the trunk";
#endif
    if (on && !no && (g_kv8 || g_tq)) no = "a quantized KV cache (KV8, KV_TQ) stays on the CPU";
    if (on && !no && g_pilot) no = "PILOT prefetch reads the residual on the host";
    const char *why = on && !no ? glmc_check(m) : NULL;
    if (why) fprintf(stderr, "[VK] colibri chain: %s; per-matrix path\n", why);
    /* the fit before anything of the chain is on the device (its pools' first blocks are
     * the fit's own granularity, as coli plan counts them) */
    if (on && !no && !why) glmc_fit_plan(m);
    if (on && !no && !(g_glmc_inited = vkc_init())) no = "the chain's pipelines did not come up";
    if (on && !no && !vkc_mla_ready()) no = "the MLA shaders are missing (chain_mla, chain_hgemv, chain_dsa)";
    if (no) fprintf(stderr, "[VK] colibri: %s: the dense chain stays off\n", no);
    if (no) g_glmc_fitted = 0;   /* no chain after all: everything as before */
    glm_dho_start(m);
    if (!g_glmc_fitted) return;
    if (g_glmc_fit.n > 0 && glmc_setup(m)) g_vk_chain = on;
    g_glmc_partial = vkc_fit_partial(&g_glmc_fit);
    for (int i = 0; i < g_glmc_fit.n; i++) g_glmc_lazy += g_glmc_mir[i];   /* for the tier (vk_tier_start) */
    vkc_fit_placed("colibri", &g_glmc_fit);
    glm_dho_finish(m);
}
/* A partial chain at exit: the matrices the device holds, which must be the ones the
 * setup placed (the per-matrix path put nothing of a CPU layer or the tail there). */
static void glmc_held_atexit(void) {
    size_t w = 0, nt = 0;
    coli_vk_mem_info(&w, &nt);
    fprintf(stderr, "[VK] colibri chain: %d of %d layers held at exit: %zu B of matrices on the device\n",
            g_glmc_fit.n, g_glmc_fit.L, w);
}
/* After the tier's: at exit the report runs first, then the chain goes, then the device. */
static void glmc_atexit(Model *m) {
    if (!g_glmc_inited) return;
    atexit(vkc_shutdown);
    if (g_vk_chain) { g_glmc_model = m; atexit(glmc_report_atexit); }
    if (g_glmc_fitted && g_glmc_partial) atexit(glmc_held_atexit);
}
