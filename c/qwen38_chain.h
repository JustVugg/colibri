/* qwen38_chain.h -- Qwen3.8 Flash Next's layers as a dense chain on the Vulkan device
 * (vk_chain.h). Included once by qwen38_core.h in a COLI_VULKAN build, after the CPU
 * forward it stands in for; COLI_VK_CHAIN decides (coli_vk_chain_decide).
 *
 * The same shape as qwen36_chain.h, with what Qwen3.8 adds:
 *   - four hyper-connection streams (hyper, S x 4H) stay on the device; every block
 *     reads them through its gated residual (per-stream norm, the low-rank down/up
 *     pair, the stream mix, the inject weights) and writes back hyper += inject x block;
 *   - the PLE layer: the n-gram table rows stay on the host (disk reads, in token
 *     order, with the n-gram history); only their rows go up, then key and value
 *     projections, the gate and the dilated convolution with its ring on the device;
 *   - QSA attention: the K/V/index-key rows on the device (the host's copies kept
 *     canonical as in qwen36), each block's pooled key computed once on the device
 *     when a step completes the block, and the indexer's top-k selection per query
 *     row on the device (chain_qsa.comp), feeding the attention's selection list;
 *   - the MTP head stays on the CPU (its experts are FP8 beside the int4 sidecar,
 *     and it reads a few rows per draft): the chain hands it every row's four streams
 *     when the model has one; a verify forward (S = 2 to Q38_SPEC_ROWS: the token and
 *     its MTP or prompt-lookup drafts) runs here row-wise (GEMV for every matrix, so its
 *     rows get a decode step's bits) and snapshots the DeltaNet state, the conv rings
 *     and the PLE ring after each of its rows but the last on the device, one slot per
 *     row, so a draft rejected after row r rolls back by swapping slot r's device
 *     buffers in (q38c_rollback). One snapshot is written by the shaders as they pass
 *     its row; more split the recurrent ops into one dispatch per snapshot, each ending
 *     on its row (q38c_deltanet, q38c_ple), which gives the same bits.
 * Per layer the host gets the MoE input rows and the router logits, and the K/V and
 * index-key rows of an attention layer; it sends the routed sum back. The shared
 * expert runs on the device (frame A2) while the host computes the routed experts.
 *
 * State ownership as in qwen36: the host's KV/index caches canonical, device mirrors
 * behind a watermark (kv_valid), pooled block keys behind pk_valid; the DeltaNet
 * state, conv rings and PLE ring on the device while the chain runs (dn_where),
 * brought back before the host reads them (prefix cache, pins) and pushed up after
 * the host writes them (reset, restore). The CUDA tier keeps its priority: with it
 * on, the chain stays off. A device lost while it holds that state: the engine rebuilds
 * it on the CPU from the prefix record and continues there (q38c_recover).
 *
 * Past the device's budget (vk_kvsplit.h) each attention layer's K/V keep only some
 * blocks on the device: a window of the newest, and the blocks QSA's selections read
 * most (pinned by their counts). The selection list goes to both parts: the device
 * attends over its listed positions, the CPU over the rest from the host's cache, and
 * the two merge through their softmax statistics. The index keys and the pooled block
 * keys stay whole on the device: every row scores every pooled key every step, and an
 * index-key row is read once, to pool its block (ID floats a position beside K/V's
 * 2*KVH*D).
 *
 * When the layers do not all fit the device, the chain takes the first N (q38c_start,
 * vkc_fit; docs/vulkan.md, "A partial chain"): their matrices, their state (DeltaNet,
 * conv rings, K/V/index-key mirrors, the PLE ring when the PLE layer is one of them) and
 * every chunk's streams through them; after layer N-1 the four streams of every row come
 * back to the host once per chunk, and the CPU runs layers N.. (their state on the host,
 * as without the chain), the final mixer and lm_head. The MTP head reads the final streams
 * from whichever side ran the last layer; it always runs on the CPU. */
#include "vk_chain.h"
#include "vk_kvsplit.h"

#define Q38C_HOST 0
#define Q38C_DEV  1
#define Q38C_BOTH 2

typedef struct {
    int ok, failed, rows, cap;
    int lo, d;                        /* its layers start at lo, on device d (vkc_device) */
    int nl, head;                     /* the layers on the device (lo..lo+nl-1 of c.layers) and whether the final
                                       * mixer and lm_head are there too; a partial chain hands the rest to the CPU */
    int built, placed_said;           /* q38c_setup ran; the fit's placed line was printed */
    size_t prm_floats;
    VkcBuf *prm;
    size_t *o_an, *o_mn, *o_qn, *o_kn, *o_iqn, *o_ikn, *o_conv, *o_dn;
    size_t o_fn, o_ple, o_pconv;
    ColiVkTensor **t_sg;
    VkcBuf **rec, **ring, **kc, **vc, **ik, **pk;
    VkcBuf **rec_snap[Q38_SPEC_SNAPS], **ring_snap[Q38_SPEC_SNAPS];   /* a verify's copies, slot r after row r */
    int snap_slots;                   /* slots allocated (1 at setup, more when a deeper verify first runs) */
    VkcBuf *ple_ring;                 /* [ple_nslots][W][SL]: current at ple_slot[0], the copy after row r at ple_slot[1+r] */
    int ple_slot[Q38_SPEC_ROWS], ple_nslots;
    int *kv_valid, *pk_valid, *attn_ord, n_attn;
    int dev_rows;                     /* device K/V rows a layer: cap, or the split's */
    VkcKvSplit ks;                    /* the K/V split past the device's budget (ks.on) */
    int dn_where, host_zero, snap_valid;
    VkcBuf *hyper, *hn, *low, *mix, *mixed, *inj_a, *inj_m, *blk, *q, *k, *v, *ip, *ctx, *qsc, *sel;
    VkcBuf *qkv, *z, *ab, *cv, *dny, *lg, *gs, *us, *hs, *ds, *sgd, *fin;
    VkcBuf *emb, *keys, *val, *gated, *normv;
    VkcBuf *mixd, *lgd, *kvd, *outd, *hypd, *find, *routed, *cs, *pcs;
    VkcBuf *gseg, *nseg;              /* a deep verify's PLE rows past its first copy (q38c_ple) */
    float *host_routed, *host_emb;
    unsigned long long forwards;
    double host_ms;
} Q38Chain;
/* the primary device's chain (d 0) or the second device's (d 1, after it) */
static Q38Chain *q38c_of(Model *m, int d) { return (Q38Chain *)(d ? m->vkchain2 : m->vkchain); }
static const char *q38c_name(const Q38Chain *ch) { return ch && ch->d ? "qwen38 dev2" : "qwen38"; }
static int q38c_dev2_wanted(void) { const char *e = getenv("COLI_VK_CHAIN_DEV2"); return !(e && *e == '0'); }

static int g_vk_chain = 0;
static void q38c_fatal(const char *what) {
    fprintf(stderr, "[VK] qwen38 chain: %s -- stopping (COLI_VK_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
static int q38c_geometry_ok(const Cfg *c) {
    int attn = 0, dn = 0;
    for (int i = 0; i < c->layers; i++) { if (c->is_attn[i]) attn = 1; else dn = 1; }
    if (attn && (c->head_dim > 256 || c->q_heads % c->kv_heads || (c->rotary_dim & 1) || c->rotary_dim > c->head_dim ||
                 c->idx_kheads != 1 || c->idx_dim > 256 || c->rotary_dim > c->idx_dim || c->idx_ratio < 1)) return 0;
    if (dn && (c->dn_vdim > 128 || c->dn_kdim > 256 || c->dn_convk < 2 || c->dn_convk > 9 || c->dn_vheads % c->dn_kheads)) return 0;
    if (c->ple_layer >= 0 && c->ple_layer < c->layers && (c->ple_convk - 1) * c->ngram_size > 32) return 0;
    return c->hc_count > 0 && c->hc_width == c->hc_count * c->hidden;
}
static ColiVkTensor *q38c_t(const Q38Weight *w) { return q38_vk_tensor(w); }
static size_t q38c_ple_cells(const Cfg *c) { return (size_t)c->hc_width * (c->ple_convk - 1) * c->ngram_size; }

/* q38c_res counts instead of reserving while g_q38c_count >= 0 (the chunk's sizing) */
static long long g_q38c_count = -1;
static int q38c_res(VkcBuf **b, size_t floats, int kind) {
    if (g_q38c_count >= 0) { g_q38c_count += (long long)(floats ? floats : 1) * (long long)sizeof(float); return 1; }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}
static int q38c_bufs(Q38Chain *ch, Model *m, int rows);
static int q38c_selrow(const Cfg *c) { return 1 + c->idx_budget + c->idx_ratio - 1; }
static size_t q38c_kv_stride(Q38Chain *ch, const Cfg *c) {   /* per attention layer in kvd: K, V, IK rows */
    return (size_t)ch->rows * (2 * (size_t)c->kv_heads * c->head_dim + c->idx_dim);
}
/* the scratch buffers for `rows` rows (kvd's stride is ch->rows') */
static int q38c_bufs(Q38Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    int QH = c->q_heads, D = c->head_dim, kvo = c->kv_heads * D, ipw = (c->idx_qheads + 1) * c->idx_dim;
    int V = c->dn_vheads * c->dn_vdim, E = c->experts, SI = c->shared_inter;
    int Ep = c->ngram_heads * c->ngram_head_dim; if (Ep < 1) Ep = 1;
    size_t r = (size_t)rows;
    int nbmax = m->kv_cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    int ok = q38c_res(&ch->hyper, r * W, VKC_DEV) && q38c_res(&ch->hn, r * W, VKC_DEV) && q38c_res(&ch->low, r * R, VKC_DEV) &&
             q38c_res(&ch->mix, r * W, VKC_DEV) && q38c_res(&ch->mixed, r * H, VKC_DEV) && q38c_res(&ch->inj_a, r * C, VKC_DEV) &&
             q38c_res(&ch->inj_m, r * C, VKC_DEV) && q38c_res(&ch->blk, r * H, VKC_DEV) && q38c_res(&ch->q, r * QH * 2 * D, VKC_DEV) &&
             q38c_res(&ch->k, r * kvo, VKC_DEV) && q38c_res(&ch->v, r * kvo, VKC_DEV) && q38c_res(&ch->ip, r * ipw, VKC_DEV) &&
             q38c_res(&ch->ctx, r * QH * D, VKC_DEV) && q38c_res(&ch->qsc, r * 2 * (size_t)nbmax, VKC_DEV) &&
             q38c_res(&ch->sel, r * q38c_selrow(c), VKC_DEV) && q38c_res(&ch->qkv, r * c->dn_conv_dim, VKC_DEV) &&
             q38c_res(&ch->z, r * V, VKC_DEV) && q38c_res(&ch->ab, r * 2 * c->dn_vheads, VKC_DEV) &&
             q38c_res(&ch->cv, r * c->dn_conv_dim, VKC_DEV) && q38c_res(&ch->dny, r * V, VKC_DEV) &&
             q38c_res(&ch->lg, r * E, VKC_DEV) && q38c_res(&ch->gs, r * SI, VKC_DEV) && q38c_res(&ch->us, r * SI, VKC_DEV) &&
             q38c_res(&ch->hs, r * SI, VKC_DEV) && q38c_res(&ch->ds, r * H, VKC_DEV) && q38c_res(&ch->sgd, r, VKC_DEV) &&
             q38c_res(&ch->fin, r * H, VKC_DEV) && q38c_res(&ch->keys, r * W, VKC_DEV) && q38c_res(&ch->val, r * H, VKC_DEV) &&
             q38c_res(&ch->gated, r * W, VKC_DEV) && q38c_res(&ch->normv, r * W, VKC_DEV) &&
             q38c_res(&ch->mixd, r * H, VKC_DOWN) && q38c_res(&ch->lgd, r * E, VKC_DOWN) &&
             q38c_res(&ch->kvd, (size_t)(ch->n_attn ? ch->n_attn : 1) * q38c_kv_stride(ch, c), VKC_DOWN) &&
             (!ch->head || q38c_res(&ch->outd, 2 * (size_t)c->vocab, VKC_DOWN)) && q38c_res(&ch->routed, r * H, VKC_UP) &&
             q38c_res(&ch->emb, r * Ep, VKC_UP) && q38c_res(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2), VKC_UP) &&
             q38c_res(&ch->pcs, (size_t)nbmax * (c->rotary_dim > 0 ? c->rotary_dim : 2), VKC_UP);
    return ok;
}
/* Prompt rows per chunk (vkc_chunk_rows): the chain's scratch a row, counted from the
 * reservations themselves (at the cache's capacity: QSA's scores grow with it), and the
 * routed experts' outputs (the tier's rows and the host's sum) for it. */
static int q38c_chunk_rows(Q38Chain *ch, Model *m) {
    int keep = ch->rows;
    ch->rows = 1; g_q38c_count = 0; q38c_bufs(ch, m, 1); long long b1 = g_q38c_count;
    ch->rows = 2; g_q38c_count = 0; q38c_bufs(ch, m, 2); long long b2 = g_q38c_count;
    g_q38c_count = -1; ch->rows = keep;
    const Cfg *c = &m->c;
    size_t row = (size_t)(b2 - b1) + (size_t)(c->topk + 1) * c->hidden * sizeof(float);
    if (ch->ks.on) row += ((long long)m->c.q_heads * (5 * m->c.head_dim + 6) + q38c_selrow(&m->c)) * sizeof(float);
    /* q38_moe_prefill keeps CPU input, gate, up and routed output buffers even
     * when the tier takes every expert: all must fit beside the device scratch. */
    size_t assignment = 2 * ((size_t)c->hidden + c->inter) * sizeof(float)
                      + sizeof(Q38RouteAssignment) + 4 * sizeof(int) + sizeof(uint8_t) + sizeof(float *);
    row += (size_t)c->topk * assignment
         + ((size_t)c->experts + 3 * (size_t)c->shared_inter + c->hidden + 1) * sizeof(float);
    return vkc_chunk_rows(q38c_name(ch), row);
}
/* Plan and allocate the KV mirror before sizing a prompt chunk. */
static int q38c_mirror(Q38Chain *ch, Model *m) {
    Cfg *c = &m->c; int kvo = c->kv_heads * c->head_dim;
    int nbmax = m->kv_cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    if (ch->cap != m->kv_cap) {   /* the host cache is new: mirror it again (K/V whole, or split past the budget) */
        size_t row = (size_t)kvo * 2 * 4;
        int plan = ch->n_attn ? vkc_kv_plan(&ch->ks, q38c_name(ch), ch->n_attn, row, m->kv_cap, 1, 1,
                                            (size_t)ch->n_attn * ch->dev_rows * row) : 1;
        if (!plan) { ch->cap = 0; return 0; }
        ch->dev_rows = ch->ks.on ? ch->ks.rows : m->kv_cap;
        for (int i = ch->lo; i < ch->lo + ch->nl; i++) {
            if (!c->is_attn[i]) continue;
            vkc_free(ch->kc[i]); vkc_free(ch->vc[i]); vkc_free(ch->ik[i]); vkc_free(ch->pk[i]);
            ch->kv_valid[i] = ch->pk_valid[i] = 0;
            ch->kc[i] = vkc_buf((size_t)kvo * ch->dev_rows * 4, VKC_DEV);
            ch->vc[i] = vkc_buf((size_t)kvo * ch->dev_rows * 4, VKC_DEV);
            ch->ik[i] = vkc_buf((size_t)c->idx_dim * m->kv_cap * 4, VKC_DEV);
            ch->pk[i] = vkc_buf((size_t)c->idx_dim * nbmax * 4, VKC_DEV);
            if (!ch->kc[i] || !ch->vc[i] || !ch->ik[i] || !ch->pk[i]) { ch->cap = 0; return 0; }
        }
        ch->cap = m->kv_cap;
    }
    return 1;
}

static int q38c_scratch(Q38Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int H = c->hidden;
    int Ep = c->ngram_heads * c->ngram_head_dim; if (Ep < 1) Ep = 1;
    size_t r = (size_t)rows;
    int need_rows = ch->rows < rows;
    if (need_rows) ch->rows = rows;   /* kvd's stride follows */
    if (!q38c_bufs(ch, m, rows)) return 0;
    if (need_rows) {
        float *hr = realloc(ch->host_routed, r * H * sizeof(float)), *he;
        if (!hr) return 0;
        ch->host_routed = hr;
        he = realloc(ch->host_emb, r * Ep * sizeof(float));
        if (!he) return 0;
        ch->host_emb = he;
    }
    if (ch->ks.on && !vkc_kv_parts(&ch->ks, r * c->q_heads * (c->head_dim + 2))) return 0;
    return 1;
}

/* ---- the layers on the device: a prefix of N (docs/vulkan.md, "A partial chain") -------
 * q38c_start decides N at startup (vkc_fit), before any upload, the dense-host pass and the
 * expert tier, from what each layer takes on the device: its matrices in the format they
 * go up in (the trunk's int8 rows, bf16 or f32) and its shared expert's gate; its state at
 * its starting size (a DeltaNet layer's recurrent state and conv ring with a verify's
 * first copy of each; an attention layer's K/V and index-key mirror at the KV split's
 * floor of three blocks, the split covering the growth, and its rows in the chunk's K/V
 * read-back; the PLE ring with the PLE layer); its share of the parameter buffer. Fixed:
 * the scratch of one prompt chunk of vkc_fit_rows(256) rows, the streams' read-back and
 * the final norm. The tail: the final mixer and lm_head, and the MTP head's matrices (they
 * go up with the full chain, through the per-matrix path or the dense-host pass).
 *
 * A partial chain (fewer layers, or the head kept on the CPU) is placed at startup, layer
 * by layer, so the tier sizes itself after it; the full chain keeps its setup at the first
 * forward, as before. A layer that does not fully reach the device goes, with every layer
 * after it and the tail (host copies read back where the dense-host pass had dropped
 * them), and the chain keeps the layers before it (vkc_fit_shrink). From then on nothing
 * new goes up through the per-matrix path (g_q38_vk_noup): the CPU's layers and the head
 * keep their host copies and run on the CPU. */
static VkcFit g_q38c_fit;             /* q38c_start's decision */
static int g_q38c_fitted;
/* The layers after the primary's on COLI_VK_DEV2's device (docs/vulkan.md, "Layers on two
 * devices"): a second chain from layer g_q38c_fit.n, its own fit on that device's memory,
 * placed at startup after the primary's. A forward runs the primary's layers, brings the
 * streams back and runs these; the CPU runs what is left and, unless this chain ends at
 * the last layer with room for it, the head. */
static VkcFit g_q38c_fit2;
static int g_q38c_fit2_on;
static int g_q38c_ran;                /* the layers the last forward ran on the devices (q38c_layers) */
#define Q38C_MATS 18
#define Q38C_TAIL 32
/* Layer i's matrices as the chain reads them (the PLE projections with their layer); the
 * shared expert's gate (t_sg) goes up after the first ten, as it always did. */
static int q38c_mats(Model *m, int i, Q38Weight **w) {
    Cfg *c = &m->c; Layer *l = &m->L[i]; int n = 0;
    Q38Weight *all[] = {&l->attn_gr.down, &l->attn_gr.up, &l->attn_gr.inject, &l->mlp_gr.down, &l->mlp_gr.up,
                        &l->mlp_gr.inject, &l->router, &l->sh_g, &l->sh_u, &l->sh_d};
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) w[n++] = all[k];
    if (c->is_attn[i]) { w[n++] = &l->q; w[n++] = &l->k; w[n++] = &l->v; w[n++] = &l->o; w[n++] = &l->idx_qk; }
    else { w[n++] = &l->dn_qkv; w[n++] = &l->dn_z; w[n++] = &l->dn_b; w[n++] = &l->dn_a; w[n++] = &l->dn_out; }
    if (i == c->ple_layer) { w[n++] = &l->ple_key; w[n++] = &l->ple_value; }
    return n;
}
/* What goes up beside the layers with the full chain: the final mixer and lm_head (the
 * chain's own, the first three), then the MTP head's matrices. */
static int q38c_tail_mats(Model *m, Q38Weight **w, int head_only) {
    int n = 0;
    w[n++] = &m->final_gr.down; w[n++] = &m->final_gr.up; w[n++] = &m->lm_head;
    if (head_only || !m->mtp) return n;
    Layer *l = &m->L[m->c.layers];
    Q38Weight *all[] = {&l->attn_gr.down, &l->attn_gr.up, &l->attn_gr.inject, &l->mlp_gr.down, &l->mlp_gr.up,
                        &l->mlp_gr.inject, &l->router, &l->sh_g, &l->sh_u, &l->sh_d, &l->q, &l->k, &l->v, &l->o,
                        &l->idx_qk, &l->dn_qkv, &l->dn_z, &l->dn_b, &l->dn_a, &l->dn_out, &l->ple_key, &l->ple_value,
                        &m->mtp_fc_emb, &m->mtp_fc_hid, &m->mtp_mixer.down, &m->mtp_mixer.up};
    for (size_t k = 0; k < sizeof all / sizeof *all && n < Q38C_TAIL; k++) if (q38_vk_eligible(all[k])) w[n++] = all[k];
    return n;
}
/* a matrix's device bytes as the weight pool places it (payload: as coli_vk_mem_info counts it) */
static size_t q38c_fit_w(const Q38Weight *w, int payload) {
    if (!q38_vk_eligible(w)) return 0;
    int fmt = q38_vk_fmt(w);
    return payload ? coli_vk_tensor_payload(fmt, w->cols, w->rows, 0) : vkc_fit_tensor(fmt, w->cols, w->rows, 0);
}
static int q38c_kv_rows0(void) { long b = vkc_kv_env("COLI_VK_KV_BLOCK", 64); return 3 * (int)(b < 1 ? 1 : b); }
/* layer i's device bytes (as q38c_setup places it, its state at its starting size) and its matrices' payload */
static void q38c_fit_layer(Model *m, int i, int rows, size_t *bytes, size_t *payload) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width;
    Q38Weight *w[Q38C_MATS]; int n = q38c_mats(m, i, w);
    size_t b = vkc_fit_tensor(10, H, 1, 0), p = coli_vk_tensor_payload(10, H, 1, 0);   /* the shared expert's gate */
    for (int k = 0; k < n; k++) { b += q38c_fit_w(w[k], 0); p += q38c_fit_w(w[k], 1); }
    size_t prm = 2 * (size_t)W;   /* the two gated residuals' norms */
    if (c->is_attn[i]) {
        size_t kvo = (size_t)c->kv_heads * c->head_dim, r0 = (size_t)q38c_kv_rows0(), ID = (size_t)c->idx_dim;
        size_t R = (size_t)(c->idx_ratio > 0 ? c->idx_ratio : 1);
        prm += 2 * (size_t)c->head_dim + 2 * ID;
        b += 2 * vkc_fit_buf(kvo * r0 * 4) + vkc_fit_buf(ID * r0 * 4) + vkc_fit_buf(ID * (r0 / R + 1) * 4);
        b += (size_t)rows * (2 * kvo + ID) * 4;   /* its rows in the chunk's K/V read-back (kvd) */
    } else {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        prm += (size_t)c->dn_conv_dim * c->dn_convk + 2 * (size_t)c->dn_vheads + c->dn_vdim;
        b += 2 * vkc_fit_buf(nr * 4) + 2 * vkc_fit_buf(nc * 4);
    }
    if (i == c->ple_layer) {
        prm += 3 * (size_t)W + (size_t)W * c->ple_convk;
        b += vkc_fit_buf(2 * q38c_ple_cells(c) * 4);
    }
    *bytes = b + prm * 4; *payload = p;
}
/* the chain's scratch for one chunk of `rows` rows, counted as q38c_chunk_rows counts it
 * (no attention layer: their read-back rows are theirs), the streams' read-back and the
 * final norm */
static size_t q38c_fit_fixed(Model *m, int rows) {
    Q38Chain t; memset(&t, 0, sizeof t);
    t.rows = rows; t.head = 1;
    g_q38c_count = 0; q38c_bufs(&t, m, rows);
    size_t b = (size_t)g_q38c_count;
    g_q38c_count = -1;
    return b + ((size_t)rows + 1) * m->c.hc_width * 4;
}
static size_t q38c_fit_tail(Model *m) {
    Q38Weight *w[Q38C_TAIL]; int n = q38c_tail_mats(m, w, 0); size_t b = 0;
    for (int k = 0; k < n; k++) b += q38c_fit_w(w[k], 0);
    return b;
}

/* Layer i's matrices and its shared expert's gate on the device. 0: one did not get there. */
static int q38c_place_mats(Q38Chain *ch, Model *m, int i) {
    Q38Weight *w[Q38C_MATS]; int n = q38c_mats(m, i, w);
    for (int k = 0; k < n; k++) {
        if (!q38c_t(w[k])) return 0;
        if (k == 9 && !ch->t_sg[i]) {
            ColiVkTensor *t = NULL;
            if (!(ch->d ? coli_vk_tensor_ensure2(&t, m->L[i].sh_gate, NULL, 10, m->c.hidden, 1, 0)
                        : coli_vk_tensor_ensure(&t, m->L[i].sh_gate, NULL, 10, m->c.hidden, 1, 0))) return 0;
            ch->t_sg[i] = t;
        }
    }
    return 1;
}
/* Layer i's state on the device: a DeltaNet layer's recurrent state and conv ring with a
 * verify's first copy of each, the PLE ring (current and one copy) with the PLE layer. */
static int q38c_place_bufs(Q38Chain *ch, Model *m, int i) {
    Cfg *c = &m->c;
    if (!c->is_attn[i]) {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        if (!(ch->rec[i] = vkc_buf(nr * 4, VKC_DEV)) || !(ch->ring[i] = vkc_buf(nc * 4, VKC_DEV)) ||
            !(ch->rec_snap[0][i] = vkc_buf(nr * 4, VKC_DEV)) || !(ch->ring_snap[0][i] = vkc_buf(nc * 4, VKC_DEV))) return 0;
    }
    if (i == c->ple_layer) {
        if (!(ch->ple_ring = vkc_buf(2 * q38c_ple_cells(c) * 4, VKC_DEV))) return 0;
        ch->ple_nslots = 2; ch->ple_slot[0] = 0; ch->ple_slot[1] = 1;
    }
    return 1;
}
/* A matrix's device copy goes: its host copy comes back first where the dense-host pass
 * dropped it, and vk_off keeps the per-matrix path from uploading it again. */
static void q38c_drop_dev(Q38Weight *w) {
    if (!w->vk && !w->vk_gone) return;
    if (w->vk_gone) q38_dho_reload(w);
    if (w->vk) { coli_vk_tensor_free((ColiVkTensor *)w->vk); w->vk = NULL; }
    w->vk_off = 1;
}
/* Everything of layer i on the device goes: its matrices, its gate, its state. */
static void q38c_unplace(Q38Chain *ch, Model *m, int i) {
    if (vkc_ready()) vkc_finish();   /* no frame may still read them */
    Q38Weight *w[Q38C_MATS]; int n = q38c_mats(m, i, w);
    for (int k = 0; k < n; k++) q38c_drop_dev(w[k]);
    if (ch->t_sg[i]) { coli_vk_tensor_free(ch->t_sg[i]); ch->t_sg[i] = NULL; }
    VkcBuf **b[] = {&ch->rec[i], &ch->ring[i], &ch->kc[i], &ch->vc[i], &ch->ik[i], &ch->pk[i]};
    for (size_t k = 0; k < sizeof b / sizeof *b; k++) { vkc_free(*b[k]); *b[k] = NULL; }
    for (int s = 0; s < Q38_SPEC_SNAPS; s++) {
        if (ch->rec_snap[s]) { vkc_free(ch->rec_snap[s][i]); ch->rec_snap[s][i] = NULL; }
        if (ch->ring_snap[s]) { vkc_free(ch->ring_snap[s][i]); ch->ring_snap[s][i] = NULL; }
    }
    if (i == m->c.ple_layer && ch->ple_ring) { vkc_free(ch->ple_ring); ch->ple_ring = NULL; ch->ple_nslots = 0; }
}
static void q38c_unplace_tail(Model *m) {
    if (vkc_ready()) vkc_finish();
    Q38Weight *w[Q38C_TAIL]; int n = q38c_tail_mats(m, w, 0);
    for (int k = 0; k < n; k++) q38c_drop_dev(w[k]);
}
/* Layer k did not fully reach the device: it goes, with every layer after it (one the
 * dense-host pass or the per-matrix path placed) and the tail; the chain keeps layers
 * 0..k-1, the CPU runs the others and the head. */
static void q38c_shrink(Q38Chain *ch, Model *m, int k, const char *why) {
    for (int j = k; j < m->c.layers; j++) q38c_unplace(ch, m, j);
    q38c_unplace_tail(m);
    if (k - ch->lo < ch->nl) ch->nl = k - ch->lo;
    ch->head = 0;
    g_q38_vk_noup = 1;
    if (g_q38c_fitted) vkc_fit_shrink(q38c_name(ch), ch->d ? &g_q38c_fit2 : &g_q38c_fit, k - ch->lo, why);
}
/* The head kept on the CPU (an upload of it refused): what of the tail went up goes. */
static void q38c_head_off(Q38Chain *ch, Model *m) {
    q38c_unplace_tail(m);
    ch->head = 0; (ch->d ? &g_q38c_fit2 : &g_q38c_fit)->tail = 0; g_q38_vk_noup = 1;
    fprintf(stderr, "[VK] qwen38 chain: the head did not reach the device: the final mixer and lm_head run on the CPU "
                    "(what it had placed was freed)\n");
}
/* The parameters the shaders read (norm weights, the DeltaNet constants, the PLE's) of
 * the layers on the device, in one buffer; the attention layers' order (n_attn). */
static int q38c_arena(Q38Chain *ch, Model *m) {
    Cfg *c = &m->c; int lo = ch->lo, hi = ch->lo + ch->nl, W = c->hc_width;
    int VH = c->dn_vheads, CD = c->dn_conv_dim, CK = c->dn_convk;
    int ple = c->ple_layer >= lo && c->ple_layer < hi;
    size_t n = 0;
    ch->n_attn = 0;
    for (int i = lo; i < hi; i++) {
        ch->o_an[i] = n; n += W; ch->o_mn[i] = n; n += W;
        if (c->is_attn[i]) {
            ch->attn_ord[i] = ch->n_attn++;
            ch->o_qn[i] = n; n += c->head_dim; ch->o_kn[i] = n; n += c->head_dim;
            ch->o_iqn[i] = n; n += c->idx_dim; ch->o_ikn[i] = n; n += c->idx_dim;
        } else {
            ch->o_conv[i] = n; n += (size_t)CD * CK;
            ch->o_dn[i] = n; n += 2 * (size_t)VH + c->dn_vdim;
        }
    }
    ch->o_fn = n; n += W;
    if (ple) { ch->o_ple = n; n += 3 * (size_t)W; ch->o_pconv = n; n += (size_t)W * c->ple_convk; }
    float *a = calloc(n, sizeof(float));
    if (!a) return 0;
    for (int i = lo; i < hi; i++) {
        Layer *l = &m->L[i];
        memcpy(a + ch->o_an[i], l->attn_gr.norm, W * sizeof(float));
        memcpy(a + ch->o_mn[i], l->mlp_gr.norm, W * sizeof(float));
        if (c->is_attn[i]) {
            memcpy(a + ch->o_qn[i], l->qn, c->head_dim * sizeof(float)); memcpy(a + ch->o_kn[i], l->kn, c->head_dim * sizeof(float));
            memcpy(a + ch->o_iqn[i], l->idx_qn, c->idx_dim * sizeof(float)); memcpy(a + ch->o_ikn[i], l->idx_kn, c->idx_dim * sizeof(float));
        } else {
            memcpy(a + ch->o_conv[i], l->dn_conv, (size_t)CD * CK * sizeof(float));
            memcpy(a + ch->o_dn[i], l->dn_alog, VH * sizeof(float));
            memcpy(a + ch->o_dn[i] + VH, l->dn_dtbias, VH * sizeof(float));
            memcpy(a + ch->o_dn[i] + 2 * VH, l->dn_norm, c->dn_vdim * sizeof(float));
        }
    }
    memcpy(a + ch->o_fn, m->final_gr.norm, W * sizeof(float));
    if (ple) {
        Layer *pl = &m->L[c->ple_layer];
        memcpy(a + ch->o_ple, pl->ple_norm_key, W * sizeof(float));
        memcpy(a + ch->o_ple + W, pl->ple_norm_query, W * sizeof(float));
        memcpy(a + ch->o_ple + 2 * (size_t)W, pl->ple_norm_conv, W * sizeof(float));
        memcpy(a + ch->o_pconv, pl->ple_conv, (size_t)W * c->ple_convk * sizeof(float));
    }
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, a, n * sizeof(float)) && vkc_submit(1);
    free(a);
    if (!ok) { vkc_free(ch->prm); ch->prm = NULL; return 0; }
    ch->prm_floats = n;
    return 1;
}
/* The chain's host-side state, no device memory yet. NULL: a model or geometry its
 * shaders do not take (the per-matrix path), or no memory. */
static Q38Chain *q38c_new_dev(Model *m, int d) {
    Q38Chain *ch = (Q38Chain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    if (d) m->vkchain2 = ch; else m->vkchain = ch;
    ch->d = d;
    Cfg *c = &m->c; int L = c->layers;
    if (m->range_begin != 0 || m->range_end != L || !m->lm_head.rows || !q38c_geometry_ok(c)) {
        fprintf(stderr, "[VK] qwen38 chain: a model or geometry its shaders do not take; per-matrix path\n");
        ch->built = 1;
        return NULL;
    }
#define Q38C_ARR(f, t) if (!(ch->f = (t *)calloc((size_t)L, sizeof(t)))) { ch->built = 1; return NULL; }
    Q38C_ARR(o_an, size_t); Q38C_ARR(o_mn, size_t); Q38C_ARR(o_qn, size_t); Q38C_ARR(o_kn, size_t);
    Q38C_ARR(o_iqn, size_t); Q38C_ARR(o_ikn, size_t); Q38C_ARR(o_conv, size_t); Q38C_ARR(o_dn, size_t);
    Q38C_ARR(t_sg, ColiVkTensor *); Q38C_ARR(rec, VkcBuf *); Q38C_ARR(ring, VkcBuf *); Q38C_ARR(rec_snap[0], VkcBuf *);
    Q38C_ARR(ring_snap[0], VkcBuf *); Q38C_ARR(kc, VkcBuf *); Q38C_ARR(vc, VkcBuf *); Q38C_ARR(ik, VkcBuf *); Q38C_ARR(pk, VkcBuf *);
    Q38C_ARR(kv_valid, int); Q38C_ARR(pk_valid, int); Q38C_ARR(attn_ord, int);
#undef Q38C_ARR
    ch->snap_slots = 1;
    if (d) {   /* from the primary's last layer; the head with the last layer and room for it */
        ch->lo = g_q38c_fit.n; ch->nl = g_q38c_fit2.n;
        ch->head = ch->lo + ch->nl == L && !vkc_fit_partial(&g_q38c_fit2);
    } else {
        ch->nl = g_q38c_fitted ? g_q38c_fit.n : L;
        ch->head = !g_q38c_fitted || !vkc_fit_partial(&g_q38c_fit);
    }
    return ch;
}
static Q38Chain *q38c_new(Model *m) { return q38c_new_dev(m, 0); }
/* Every layer the fit gave the device, one at a time (its matrices, its state), then the
 * parameters, the placed line and the head. A partial chain runs it at startup
 * (q38c_start), the full chain at its first forward, as before. */
/* d = 1: the second device's chain (the caller has made that device current). */
static Q38Chain *q38c_setup_dev(Model *m, int d) {
    Q38Chain *ch = q38c_of(m, d);
    if (ch && ch->built) return ch->ok ? ch : NULL;
    if (d && !g_q38c_fit2_on) return NULL;
    if (!ch && !(ch = q38c_new_dev(m, d))) return NULL;
    ch->built = 1;
    const char *nm = q38c_name(ch);
    VkcFit *fit = d ? &g_q38c_fit2 : g_q38c_fitted ? &g_q38c_fit : NULL;
    g_q38_vk_dev = d;
    for (int i = ch->lo; i < ch->lo + ch->nl; i++) {
        if (q38c_place_mats(ch, m, i) && q38c_place_bufs(ch, m, i)) { if (fit) vkc_fit_mark(fit, i - ch->lo); continue; }
        q38c_shrink(ch, m, i, vkc_lost() ? "the device was lost" : "the device refused an upload or an allocation");
        break;
    }
    if (ch->nl && !q38c_arena(ch, m))
        q38c_shrink(ch, m, ch->lo, vkc_lost() ? "the device was lost" : "the device refused the chain's parameter buffer");
    if (fit && !ch->placed_said) { vkc_fit_placed(nm, fit); ch->placed_said = 1; }
    if (ch->nl && ch->head) {
        Q38Weight *w[Q38C_TAIL]; int n = q38c_tail_mats(m, w, 1), ok = 1;
        for (int k = 0; k < n && ok; k++) ok = q38c_t(w[k]) != NULL;
        if (!ok) q38c_head_off(ch, m);
    }
    g_q38_vk_dev = 0;
    if (!ch->nl) {   /* nothing of the chain on the device: its own pools go too */
        if (!fit) fprintf(stderr, "[VK] qwen38 chain: a matrix did not reach the device; per-matrix path\n");
        vkc_shutdown();
        if (!d) g_vk_chain = 0;   /* the second device's failing: the primary's chain stands, the CPU runs the rest */
        return NULL;
    }
    ch->dn_where = Q38C_HOST;
    ch->ok = 1;
    size_t wb = 0, wn = 0;
    coli_vk_mem_info_dev(d, &wb, &wn);
    if (d) fprintf(stderr, "[VK] qwen38 chain: layers %d..%d on the second device (%d QSA), %zu matrices resident there, "
                           "%.1f MiB of parameters%s\n", ch->lo, ch->lo + ch->nl - 1, ch->n_attn, wn,
                   ch->prm_floats * 4 / 1048576.0, ch->head ? ", the final mixer and lm_head too" : "");
    else fprintf(stderr, "[VK] qwen38 chain: %d layers on the device (%d QSA), %zu matrices resident, %.1f MiB of parameters%s\n",
                 ch->nl, ch->n_attn, wn, ch->prm_floats * 4 / 1048576.0,
                 ch->head ? "" : "; the CPU runs the other layers, the final mixer and lm_head");
    return ch;
}
static Q38Chain *q38c_setup(Model *m) { return q38c_setup_dev(m, 0); }
/* the layers the last forward ran on the devices, from layer 0 */
static int q38c_layers(Model *m) { (void)m; return g_q38c_ran; }

/* At startup, with the chain on (before anything of it is on the device, the dense-host
 * pass and the expert tier): N, the chain's pipelines (vkc_init), and a partial chain
 * placed now. N = 0: the chain off, nothing of it on the device, nothing more going up
 * (every byte left to the tier). A model the chain declines keeps today's path (the
 * caller's vkc_init, the decline at its first forward). */
static void q38c_start(Model *m) {
    if (!g_vk_chain || qt_ready()) return;
    Cfg *c = &m->c; int L = c->layers;
    if (m->range_begin != 0 || m->range_end != L || !m->lm_head.rows || !q38c_geometry_ok(c)) return;
    int rows = vkc_fit_rows(256);
    size_t *lb = (size_t *)calloc((size_t)L, sizeof *lb), *mb = (size_t *)calloc((size_t)L, sizeof *mb);
    if (!lb || !mb) { free(lb); free(mb); return; }
    for (int i = 0; i < L; i++) q38c_fit_layer(m, i, rows, &lb[i], &mb[i]);
    size_t fixed = q38c_fit_fixed(m, rows), tail = q38c_fit_tail(m);
    vkc_fit("qwen38", L, lb, mb, fixed, tail, &g_q38c_fit);
    g_q38c_fitted = 1;
    int partial = vkc_fit_partial(&g_q38c_fit), n0 = g_q38c_fit.n;
    if (!n0) { g_vk_chain = 0; g_q38_vk_noup = 1; free(lb); free(mb); return; }   /* the chain off: its pipelines never come up */
    if (!vkc_init()) { g_vk_chain = 0; g_q38c_fitted = 0; free(lb); free(mb); return; }   /* the per-matrix path, as before */
    /* the layers the primary leaves, on COLI_VK_DEV2's device: a fit of its own from layer
     * n0, with that device's free memory (the tier there sizes itself after the placement) */
    if (partial && n0 < L && q38c_dev2_wanted() && getenv("COLI_VK_DEV2") && coli_vk_dev2_open_env()) {
        vkc_device(1);
        int n2 = vkc_fit("qwen38 dev2", L - n0, lb + n0, mb + n0, fixed, tail, &g_q38c_fit2);
        if (n2 > 0 && !vkc_init()) {
            fprintf(stderr, "[VK] qwen38 chain: the second device's pipelines did not come up; its layers stay on the CPU\n");
            n2 = 0;
        }
        g_q38c_fit2_on = n2 > 0;
        vkc_device(0);
    }
    free(lb); free(mb);
    if (!partial) return;   /* every layer and the head: set up at the first forward, as before */
    Q38Chain *ch = q38c_setup(m);
    if (!ch) { vkc_shutdown(); g_vk_chain = 0; }
    if (g_q38c_fit2_on) {
        vkc_device(1);
        if (!ch || ch->nl < n0) {
            /* the primary placed fewer layers than its fit: the second device's would not
             * follow them, so they stay on the CPU too */
            fprintf(stderr, "[VK] qwen38 chain: the primary device stopped before layer %d; layers %d..%d stay on the CPU, "
                            "not on the second device\n", n0, n0, n0 + g_q38c_fit2.n - 1);
            g_q38c_fit2_on = 0;
            vkc_shutdown();
        } else if (!q38c_setup_dev(m, 1)) g_q38c_fit2_on = 0;
        vkc_device(0);
    }
    g_q38_vk_noup = 1;   /* after the setup's own uploads */
}
/* The dense-host pass's bound (q38_dho_start): the N layers' matrices of a partial chain,
 * the head's and the MTP head's only with the tail on the device; with N = 0 the dense
 * part is on the CPU. */
static void q38c_dho_bound(int *layers, int *head, int *dev) {
    if (!g_q38c_fitted) return;
    if (vkc_fit_partial(&g_q38c_fit)) {
        *layers = g_q38c_fit.n; *head = 0;
        /* the second device's layers too, where the chain runs every step (with prompts
         * only, decode reads them on the CPU); the head keeps its host copy */
        if (g_q38c_fit2_on && g_vk_chain == COLI_VK_CHAIN_ON) *layers += g_q38c_fit2.n;
    }
    if (!*layers) *dev = 0;
}
/* The dense-host pass with the full chain fitted (the matrices go up at startup in this
 * mode): each layer's matrices go up and their host copies go only once the whole layer
 * is there; a layer that does not get there leaves the chain the layers before it, placed
 * now as a partial chain. Then the placed line, the head, and the MTP head's matrices.
 * 0: not this case, the caller's pass runs. */
static int q38c_dho_place(Model *m, size_t *bytes, int *n) {
    if (!g_q38c_fitted || vkc_fit_partial(&g_q38c_fit) || !g_vk_chain) return 0;
    Q38Chain *ch = (Q38Chain *)m->vkchain;
    if (!ch && !(ch = q38c_new(m))) return 0;
    if (ch->built) return 0;
    int L = m->c.layers;
    for (int i = 0; i < L; i++) {
        if (!q38c_place_mats(ch, m, i)) {
            q38c_shrink(ch, m, i, vkc_lost() ? "the device was lost" : "the device refused an upload");
            break;
        }
        vkc_fit_mark(&g_q38c_fit, i);
        Q38Weight *w[Q38C_MATS]; int nw = q38c_mats(m, i, w);
        for (int k = 0; k < nw; k++) q38_dho_drop(w[k], bytes, n);
    }
    vkc_fit_placed("qwen38", &g_q38c_fit); ch->placed_said = 1;
    if (ch->nl == L && ch->head) {
        Q38Weight *w[Q38C_TAIL]; int nh = q38c_tail_mats(m, w, 1), ok = 1;
        for (int k = 0; k < nh && ok; k++) ok = q38c_t(w[k]) != NULL;
        if (!ok) q38c_head_off(ch, m);
        else { nh = q38c_tail_mats(m, w, 0); for (int k = 0; k < nh; k++) q38_dho_drop(w[k], bytes, n); }
    }
    if (vkc_fit_partial(&g_q38c_fit)) {   /* what reached the device is a partial chain now: its state goes up now */
        g_q38_vk_noup = 1;
        if (!ch->nl || !q38c_setup(m)) { vkc_shutdown(); g_vk_chain = 0; }
    }
    return 1;
}
/* The dense-host lines say which layers dropped their host copies. */
static void q38c_dho_layers(void) {
    if (g_q38c_fitted)
        coli_vk_dense_host_layers(g_q38c_fit.n + (g_q38c_fit2_on && g_vk_chain == COLI_VK_CHAIN_ON ? g_q38c_fit2.n : 0),
                                  g_q38c_fit.L);
}

/* ---- state between the host and the device ------------------------------------ */
static void q38_layer_forward(Model *m,int i,float *hyper,const int *ids,int S,int pos_base,
                              float *mixed,float *inject,float *block);
static void q38_embed_row(Model *m,int id,int abs_pos,float *out);
/* The device was lost with the newest DeltaNet state, conv rings and PLE ring on it.
 * The host's K/V/index rows are canonical, those are not: rebuild them on the CPU by
 * running the `upto` positions the prefix record names (the PLE n-gram history replayed
 * with them), then leave the chain off. The MTP head's own rows live on the host. With a
 * partial chain only the device's layers are rebuilt: the CPU's layers (and the PLE
 * layer's state and history when it is one of them) hold their own state already. */
static void q38c_recover(Model *m, int upto) {
    Q38Chain *ch = (Q38Chain *)m->vkchain, *ch2 = (Q38Chain *)m->vkchain2;
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count;
    int nl = ch ? ch->nl : c->layers;
    if (ch && ch->ok && ch2 && ch2->ok) nl += ch2->nl;   /* the second device's layers ran in that forward too */
    int ple = c->ple_layer >= 0 && c->ple_layer < nl;
    g_vk_chain = 0;
    for (int d = 0; d < 2; d++) {
        Q38Chain *x = q38c_of(m, d);
        if (x) { x->failed = 1; x->dn_where = Q38C_HOST; x->host_zero = 0; x->snap_valid = 0; }
    }
    for (int i = 0; i < nl; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
    if (ple && m->PLE_conv_state) memset(m->PLE_conv_state, 0, q38c_ple_cells(c) * sizeof(float));
    if (ple && m->ple_history) m->ple_history_len = 0;
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        q38c_fatal("the device was lost with a recurrent state its token ids do not describe (an image)");
    fprintf(stderr, "[VK] qwen38 chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", upto);
    int *ids = (int *)malloc((size_t)upto * sizeof(int));
    float *hyper = falloc((int64_t)upto * W), *mixed = falloc((int64_t)upto * H);
    float *inject = falloc((int64_t)upto * C), *block = falloc((int64_t)upto * H);
    if (!ids) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    memcpy(ids, m->kvp.fed, (size_t)upto * sizeof(int));
    for (int s = 0; s < upto; s++) {
        float *e = hyper + (int64_t)s * W;
        q38_embed_row(m, ids[s], s, e);
        for (int b = 1; b < C; b++) memcpy(e + (int64_t)b * H, e, (size_t)H * sizeof(float));
    }
    /* the step being run keeps its prefetched n-gram rows and its snapshot request */
    float *pref = m->ple_pref; int pref_rows = m->ple_pref_rows, snap = m->snap_rows, rowwise = g_q38_rowwise;
    m->ple_pref = NULL; m->ple_pref_rows = 0; m->snap_rows = 0; g_q38_rowwise = 0;
    for (int i = 0; i < nl; i++) q38_layer_forward(m, i, hyper, ids, upto, 0, mixed, inject, block);
    free(m->ple_pref);
    m->ple_pref = pref; m->ple_pref_rows = pref_rows; m->snap_rows = snap; g_q38_rowwise = rowwise;
    free(ids); free(hyper); free(mixed); free(inject); free(block);
}
/* (every function here covers both chains: the primary's layers and the second device's) */
static int q38c_sync_one(Model *m, Q38Chain *ch) {
    if (!ch || !ch->ok || ch->dn_where != Q38C_DEV) return 1;
    Cfg *c = &m->c;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    int was = vkc_device(ch->d), ok = 1;
    for (int i = ch->lo; i < ch->lo + ch->nl && ok; i++) {
        if (c->is_attn[i]) continue;
        ok = vkc_read(ch->rec[i], 0, m->DN_rec[i], nr * 4) && vkc_read(ch->ring[i], 0, m->DN_conv[i], nc * 4);
    }
    if (ok && ch->ple_ring && m->PLE_conv_state)
        ok = vkc_read(ch->ple_ring, (size_t)ch->ple_slot[0] * q38c_ple_cells(c), m->PLE_conv_state, q38c_ple_cells(c) * 4);
    vkc_device(was);
    if (ok) ch->dn_where = Q38C_BOTH;
    return ok;
}
static void q38c_sync_host(Model *m) {
    for (int d = 0; d < 2; d++)
        if (!q38c_sync_one(m, q38c_of(m, d))) { q38c_recover(m, m->kv_len); return; }
}
static void q38c_host_wrote(Model *m, int zero) {
    for (int d = 0; d < 2; d++) {
        Q38Chain *ch = q38c_of(m, d);
        if (!ch || !ch->ok) continue;
        ch->dn_where = Q38C_HOST; ch->host_zero = zero; ch->snap_valid = 0;
    }
}
/* A CPU step from pos_base over one chain's layers: its host state current first. */
static void q38c_cpu_one(Model *m, Q38Chain *ch, int pos_base) {
    if (!ch || !ch->ok) return;
    if (!q38c_sync_one(m, ch)) { q38c_recover(m, m->kv_len); return; }
    ch->dn_where = Q38C_HOST; ch->host_zero = 0; ch->snap_valid = 0;
    for (int i = ch->lo; i < ch->lo + ch->nl; i++) {
        if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
        if (m->c.is_attn[i]) vkc_kv_lower(&ch->ks, ch->attn_ord[i], pos_base);
        int R = m->c.idx_ratio > 0 ? m->c.idx_ratio : 1;
        if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;
    }
}
static void q38c_cpu_step(Model *m, int pos_base) {
    for (int d = 0; d < 2; d++) q38c_cpu_one(m, q38c_of(m, d), pos_base);
}
/* A rejected draft: the state after the verify's row `slot` (its first slot+1 rows) is
 * the device's copy in that slot; `len` positions stand. The KV and pooled-key
 * watermarks come down to them: the rows past them are the rejected drafts'. */
static void q38c_rollback(Model *m, int slot, int len) {
    int R = m->c.idx_ratio > 0 ? m->c.idx_ratio : 1;
    for (int d = 0; d < 2; d++) {
        Q38Chain *ch = q38c_of(m, d);
        if (!ch || !ch->ok) continue;
        for (int i = ch->lo; i < ch->lo + ch->nl; i++) {
            if (ch->kv_valid[i] > len) ch->kv_valid[i] = len;
            if (m->c.is_attn[i]) vkc_kv_lower(&ch->ks, ch->attn_ord[i], len);
            if (ch->pk_valid[i] > len / R) ch->pk_valid[i] = len / R;
        }
        if (slot < 0 || slot >= ch->snap_valid || ch->dn_where != Q38C_DEV) { ch->snap_valid = 0; continue; }
        for (int i = ch->lo; i < ch->lo + ch->nl; i++) {
            if (m->c.is_attn[i]) continue;
            VkcBuf *t = ch->rec[i]; ch->rec[i] = ch->rec_snap[slot][i]; ch->rec_snap[slot][i] = t;
            t = ch->ring[i]; ch->ring[i] = ch->ring_snap[slot][i]; ch->ring_snap[slot][i] = t;
        }
        if (ch->ple_ring) { int t = ch->ple_slot[0]; ch->ple_slot[0] = ch->ple_slot[1 + slot]; ch->ple_slot[1 + slot] = t; }
        ch->snap_valid = 0;
    }
}
/* Copy slots on the device for a verify that snapshots `rows` rows: the DeltaNet states
 * and conv rings (slot 0 since setup), and the PLE ring's slots, all in one buffer since
 * the shader writes the copy beside the ring (it grows once, keeping the current ring). */
static int q38c_spec_slots(Q38Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int L = c->layers;
    if (rows > Q38_SPEC_SNAPS) rows = Q38_SPEC_SNAPS;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int sl = ch->snap_slots; sl < rows; sl++) {
        if (!ch->rec_snap[sl] && !(ch->rec_snap[sl] = (VkcBuf **)calloc((size_t)L, sizeof(VkcBuf *)))) return 0;
        if (!ch->ring_snap[sl] && !(ch->ring_snap[sl] = (VkcBuf **)calloc((size_t)L, sizeof(VkcBuf *)))) return 0;
        for (int i = ch->lo; i < ch->lo + ch->nl; i++) {
            if (c->is_attn[i]) continue;
            if (!ch->rec_snap[sl][i] && !(ch->rec_snap[sl][i] = vkc_buf(nr * 4, VKC_DEV))) return 0;
            if (!ch->ring_snap[sl][i] && !(ch->ring_snap[sl][i] = vkc_buf(nc * 4, VKC_DEV))) return 0;
        }
        ch->snap_slots = sl + 1;
    }
    if (ch->ple_ring && ch->ple_nslots < rows + 1) {
        size_t pc = q38c_ple_cells(c);
        VkcBuf *nb = vkc_buf((size_t)Q38_SPEC_ROWS * pc * 4, VKC_DEV);
        if (!nb) return 0;
        if (!vkc_begin() || !vkc_copy(nb, 0, ch->ple_ring, (size_t)ch->ple_slot[0] * pc, pc) || !vkc_submit(1)) {
            vkc_free(nb); return 0;
        }
        vkc_free(ch->ple_ring);
        ch->ple_ring = nb; ch->ple_nslots = Q38_SPEC_ROWS;
        for (int r = 0; r < Q38_SPEC_ROWS; r++) ch->ple_slot[r] = r;
        ch->snap_valid = 0;
    }
    return 1;
}
static int q38c_push_state(Q38Chain *ch, Model *m, int pos_base, int n_rows) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->dn_where == Q38C_HOST) {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        for (int i = ch->lo; i < ch->lo + ch->nl && ok; i++) {
            if (c->is_attn[i]) continue;
            if (ch->host_zero) ok = vkc_zero(ch->rec[i], 0, nr) && vkc_zero(ch->ring[i], 0, nc);
            else ok = vkc_write(ch->rec[i], 0, m->DN_rec[i], nr * 4) && vkc_write(ch->ring[i], 0, m->DN_conv[i], nc * 4);
        }
        if (ok && ch->ple_ring && m->PLE_conv_state) {
            size_t pc = q38c_ple_cells(c), off = (size_t)ch->ple_slot[0] * pc;
            ok = ch->host_zero ? vkc_zero(ch->ple_ring, off, pc) : vkc_write(ch->ple_ring, off, m->PLE_conv_state, pc * 4);
        }
        ch->dn_where = Q38C_BOTH;
    }
    int D = c->head_dim, KVH = c->kv_heads, ID = c->idx_dim, R = c->idx_ratio;
    for (int i = ch->lo; i < ch->lo + ch->nl && ok; i++) {
        if (!c->is_attn[i]) continue;
        if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;   /* blocks this step rewrites */
        if (ch->ks.on) {   /* the split: the window placed, the K/V rows below pos_base of its blocks uploaded */
            VkcKvPart pt[2] = {{KVH, D, m->K[i], (size_t)m->kv_cap * D, ch->kc[i], 0},
                               {KVH, D, m->V[i], (size_t)m->kv_cap * D, ch->vc[i], 0}};
            vkc_kv_place(&ch->ks, ch->attn_ord[i], pos_base, n_rows);
            ok = vkc_kv_push(&ch->ks, ch->attn_ord[i], pt, 2, pos_base);
        }
        if (ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < KVH && ok && !ch->ks.on; h++) {
            size_t src = ((size_t)h * m->kv_cap + t0) * D, dst = ((size_t)h * ch->dev_rows + t0) * D;
            ok = vkc_write(ch->kc[i], dst, m->K[i] + src, (size_t)n * D * 4) && vkc_write(ch->vc[i], dst, m->V[i] + src, (size_t)n * D * 4);
        }
        ok = ok && vkc_write(ch->ik[i], (size_t)t0 * ID, m->IK[i] + (size_t)t0 * ID, (size_t)n * ID * 4);
        ch->kv_valid[i] = pos_base;
        if (ch->pk_valid[i] > t0 / R) ch->pk_valid[i] = t0 / R;
    }
    return ok;
}

/* ---- pieces ---------------------------------------------------------------------- */
/* the gated residual's read (q38_gr_read): mixed [n][H], inject [n][C] when inj */
static int q38c_gr_read(Q38Chain *ch, Model *m, const GatedResidual *g, size_t norm_off, int n, VkcBuf *inj) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    VkcNorm np = {n * C, H, C, 0, W, H, 0, W, H, (int)norm_off, C, VKC_NORM_ADD1, c->eps, 1.f};
    VkcEw lo = {VKC_EW_HC_LOW, n * R, R, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    VkcEw mx = {VKC_EW_HC_MIX, n * H, H, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    int ok = vkc_norm(ch->hyper, ch->prm, ch->hn, &np) &&
             vkc_matmul(q38c_t(&g->down), ch->hn, 0, ch->low, 0, n) && vkc_ew(ch->low, ch->low, NULL, NULL, NULL, &lo) &&
             vkc_matmul(q38c_t(&g->up), ch->low, 0, ch->mix, 0, n) && vkc_ew(ch->mixed, ch->mix, ch->hn, NULL, NULL, &mx);
    if (ok && inj) {
        VkcEw ij = {VKC_EW_HC_INJ, n * C, C, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
        ok = vkc_matmul(q38c_t(&g->inject), ch->hn, 0, inj, 0, n) && vkc_ew(inj, inj, NULL, NULL, NULL, &ij);
    }
    return ok;
}
static int q38c_gr_apply(Q38Chain *ch, Model *m, VkcBuf *inj, int n) {
    Cfg *c = &m->c;
    VkcEw ap = {VKC_EW_HC_APPLY, n * c->hc_width, c->hidden, c->hc_count, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->hyper, inj, ch->blk, NULL, NULL, &ap);
}
static int q38c_attention(Q38Chain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int QH = c->q_heads, KVH = c->kv_heads, D = c->head_dim, kvo = KVH * D, qo = QH * 2 * D, half = c->rotary_dim / 2;
    int IQ = c->idx_qheads, ID = c->idx_dim, R = c->idx_ratio, ipw = (IQ + 1) * ID;
    int ok = vkc_matmul(q38c_t(&l->q), ch->mixed, 0, ch->q, 0, n) && vkc_matmul(q38c_t(&l->k), ch->mixed, 0, ch->k, 0, n) &&
             vkc_matmul(q38c_t(&l->v), ch->mixed, 0, ch->v, 0, n) && vkc_matmul(q38c_t(&l->idx_qk), ch->mixed, 0, ch->ip, 0, n);
    VkcNorm kn = {n * KVH, D, KVH, 0, kvo, D, 0, kvo, D, (int)ch->o_kn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
    VkcNorm qn = {n * QH, D, QH, 0, qo, 2 * D, 0, qo, 2 * D, (int)ch->o_qn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
    VkcNorm in = {n * IQ, ID, IQ, 0, ipw, ID, 0, ipw, ID, (int)ch->o_iqn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
    ok = ok && vkc_norm(ch->k, ch->prm, ch->k, &kn) && vkc_norm(ch->q, ch->prm, ch->q, &qn) && vkc_norm(ch->ip, ch->prm, ch->ip, &in);
    if (ok && half) {
        VkcRope rk = {n * KVH, KVH, 0, kvo, D, half, 0, 2 * half}, rq = {n * QH, QH, 0, qo, 2 * D, half, 0, 2 * half};
        VkcRope ri = {n * IQ, IQ, 0, ipw, ID, half, 0, 2 * half};
        ok = vkc_rope(ch->k, ch->cs, &rk) && vkc_rope(ch->q, ch->cs, &rq) && vkc_rope(ch->ip, ch->cs, &ri);
    }
    if (!ok) return 0;
    /* the rows into the device caches and the host's */
    VkcRegion *rg = malloc(sizeof *rg * (size_t)n * (KVH > 1 ? KVH : 1));
    VkcRegion *ri = malloc(sizeof *ri * (size_t)n);
    if (!rg || !ri) { free(rg); free(ri); return 0; }
    for (int s = 0; s < n; s++) {
        for (int h = 0; h < KVH; h++)
            rg[s * KVH + h] = (VkcRegion){((size_t)h * ch->dev_rows + pb + s) * D, (size_t)s * kvo + (size_t)h * D, (size_t)D};
        ri[s] = (VkcRegion){(size_t)(pb + s) * ID, (size_t)s * ipw + (size_t)IQ * ID, (size_t)ID};
    }
    size_t ko = (size_t)ch->attn_ord[i] * q38c_kv_stride(ch, c);
    VkcKvPart pk_ = {KVH, D, m->K[i], (size_t)m->kv_cap * D, ch->kc[i], 0}, pv_ = {KVH, D, m->V[i], (size_t)m->kv_cap * D, ch->vc[i], 0};
    ok = (ch->ks.on ? vkc_kv_store(&ch->ks, ch->attn_ord[i], &pk_, ch->k, 0, (size_t)kvo, (size_t)D, pb, n) &&
                      vkc_kv_store(&ch->ks, ch->attn_ord[i], &pv_, ch->v, 0, (size_t)kvo, (size_t)D, pb, n)
                    : vkc_copy_regions(ch->kc[i], ch->k, rg, n * KVH) && vkc_copy_regions(ch->vc[i], ch->v, rg, n * KVH)) &&
         vkc_copy_regions(ch->ik[i], ch->ip, ri, n) &&
         vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) &&
         vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    for (int s = 0; s < n && ok; s++)   /* index keys back, compact */
        ok = vkc_copy(ch->kvd, ko + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, ch->ip, (size_t)s * ipw + (size_t)IQ * ID, ID);
    free(rg); free(ri);
    /* the pooled keys of the blocks this step completes (and any behind the watermark) */
    int nb = (pb + n) / R, b0 = ch->pk_valid[i];
    if (ok && nb > b0) {
        float *pc = (float *)vkc_ptr(ch->pcs);
        if (half) for (int b = b0; b < nb; b++) for (int j = 0; j < half; j++) {
            float ang = (float)(b * R) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
            pc[((b - b0) * half + j) * 2] = cosf(ang); pc[((b - b0) * half + j) * 2 + 1] = sinf(ang);
        }
        VkcQsa pp = {0, ID, R, b0, half, 0, 0, 0, 0, 0, 0, 0, 0, c->eps, 0, (int)ch->o_ikn[i], 0, nb - b0};
        ok = vkc_qsa(ch->ik[i], ch->prm, ch->pk[i], ch->pcs, NULL, NULL, &pp);
        ch->pk_valid[i] = nb;
    }
    int nbmax = (int)(vkc_bytes(ch->qsc) / 4 / (2 * (size_t)ch->rows));
    VkcQsa ps = {1, ID, R, 0, 0, n, pb, c->idx_budget, IQ, 0, ipw, nbmax, q38c_selrow(c), c->eps, 0, 0, 0, 0};
    if (ch->ks.on) {   /* the split: the selection's listed positions in two parts */
        VkcKvGqa a = {&ch->ks, ch->attn_ord[i], ch->q, ch->kc[i], ch->vc[i], ch->q, ch->sel, NULL, ch->ctx, n, QH, KVH, D, D, pb,
                      0, 0, 0, 0, qo, 2 * D, 1, D, qo, 2 * D, QH * D, 0, q38c_selrow(c), 1.f / sqrtf((float)D),
                      m->K[i], m->V[i], (size_t)m->kv_cap * D, (size_t)D, (size_t)m->kv_cap * D, (size_t)D};
        return ok && vkc_qsa(ch->ip, NULL, ch->pk[i], NULL, ch->qsc, ch->sel, &ps) && vkc_kv_gqa(&a) &&
               vkc_matmul(q38c_t(&l->o), ch->ctx, 0, ch->blk, 0, n);
    }
    VkcAttn at = {n, QH, KVH, D, pb, ch->dev_rows, 0, qo, 2 * D, D, qo, 2 * D, 1, 0, QH * D, 0, q38c_selrow(c),
                  1.f / sqrtf((float)D), 0, 0};
    return ok && vkc_qsa(ch->ip, NULL, ch->pk[i], NULL, ch->qsc, ch->sel, &ps) &&
           vkc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, ch->sel, &at) &&
           vkc_matmul(q38c_t(&l->o), ch->ctx, 0, ch->blk, 0, n);
}
/* ns: the chunk's rows whose state a verify copies (rows 0..ns-1, into slots c0..c0+ns-1).
 * One copy rides on the one dispatch, as the shaders pass its row; more split the
 * convolution and the recurrence into one dispatch per copy, each ending on its row
 * (the last one takes the chunk's remaining rows): the state crosses between them
 * through memory in f32, so every row gets the bits of the single dispatch. */
static int q38c_deltanet(Q38Chain *ch, Model *m, Layer *l, int i, int n, int c0, int ns) {
    Cfg *c = &m->c;
    int VH = c->dn_vheads, CD = c->dn_conv_dim, V = VH * c->dn_vdim;
    int ok = vkc_matmul(q38c_t(&l->dn_qkv), ch->mixed, 0, ch->qkv, 0, n) && vkc_matmul(q38c_t(&l->dn_z), ch->mixed, 0, ch->z, 0, n) &&
             vkc_matmul(q38c_t(&l->dn_b), ch->mixed, 0, ch->ab, 0, n) &&
             vkc_matmul(q38c_t(&l->dn_a), ch->mixed, 0, ch->ab, (size_t)ch->rows * VH, n);
    int segs = ns > 1 ? ns : 1;
    for (int r = 0; r < segs && ok; r++) {
        int s0 = ns > 1 ? r : 0, len = ns > 1 && r < ns - 1 ? 1 : n - s0, snap_row = ns ? 0 : -1, slot = ns ? c0 + r : 0;
        VkcDnConv cp = {len, CD, c->dn_convk, s0 * CD, CD, s0 * CD, CD, snap_row, 1, (int)ch->o_conv[i], 0, 0, 0, 0};
        VkcDnRec rp = {len, VH, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, s0 * CD, CD, s0 * VH, VH,
                       ch->rows * VH + s0 * VH, VH, s0 * V, V, s0 * V, V, snap_row, 1, c->eps,
                       1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i], 0, 0};
        ok = vkc_dnconv(ch->qkv, ch->prm, ch->ring[i], ch->cv, ch->ring_snap[slot][i], &cp) &&
             vkc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, ch->rec[i], ch->prm, ch->dny, ch->rec_snap[slot][i], &rp);
    }
    return ok && vkc_matmul(q38c_t(&l->dn_out), ch->dny, 0, ch->blk, 0, n);
}
/* the n-gram rows of n tokens, in order, as q38_ple fills them (history and the verify's snapshot) */
static void q38c_ple_rows(Model *m, const int *ids, int c0, int n, float *emb) {
    Cfg *c = &m->c; int E = c->ngram_heads * c->ngram_head_dim;
    for (int r = 0; r < n; r++) {
        int s = c0 + r; float *e = emb + (size_t)r * E;
        int64_t p1 = m->ple_history_len >= 1 ? m->ple_history[m->ple_history_len - 1] : c->eos_id;
        int64_t p2 = m->ple_history_len >= 2 ? m->ple_history[m->ple_history_len - 2] : c->eos_id;
        if (m->ple_pref && s < m->ple_pref_rows)
            memcpy(e, m->ple_pref + (int64_t)s * c->ngram_heads * c->ngram_head_dim, (size_t)E * sizeof(float));
        else for (int h = 0; h < c->ngram_heads; h++) {
            int ng = h < c->heads_per_ngram ? 2 : 3;
            q38_ple_row(m, q38_hash_row(m, h, ng, ids[s], p1, p2), e + (int64_t)h * c->ngram_head_dim);
        }
        if (ids[s] == c->eos_id) m->ple_history_len = 0;
        else if (m->ple_history_len == 0) { m->ple_history[0] = ids[s]; m->ple_history_len = 1; }
        else if (m->ple_history_len == 1) { m->ple_history[1] = ids[s]; m->ple_history_len = 2; }
        else { m->ple_history[0] = m->ple_history[1]; m->ple_history[1] = ids[s]; }
        if (s < m->snap_rows) {
            memcpy(m->snap_ple_history[s], m->ple_history, sizeof(m->snap_ple_history[s]));
            m->snap_ple_history_len[s] = m->ple_history_len;
        }
    }
}
/* ns as q38c_deltanet's: the copies of the PLE ring after rows 0..ns-1 go to the ring
 * buffer's slots ple_slot[1+c0+r]. The convolution reads a row's gate and norm outputs
 * from the start of their buffers, so with more than one copy each dispatch past the
 * first reads its rows from a copy of them at the start of the scratch (gseg, nseg). */
static int q38c_ple(Q38Chain *ch, Model *m, int n, int c0, int ns) {
    Cfg *c = &m->c; Layer *l = &m->L[c->ple_layer];
    int W = c->hc_width, H = c->hidden, E = c->ngram_heads * c->ngram_head_dim;
    size_t pc = q38c_ple_cells(c);
    int ok = vkc_write(ch->emb, 0, ch->host_emb, (size_t)n * E * 4) &&
             vkc_matmul(q38c_t(&l->ple_key), ch->emb, 0, ch->keys, 0, n) &&
             vkc_matmul(q38c_t(&l->ple_value), ch->emb, 0, ch->val, 0, n);
    VkcPle g = {0, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, 0, 0, -1, 0, c->eps, (int)ch->o_ple, 0, 0};
    ok = ok && vkc_ple(ch->keys, ch->hyper, ch->val, ch->prm, ch->gated, ch->normv, NULL, NULL, &g);
    if (ns <= 1) {
        int snap = ns ? ch->ple_slot[1 + c0] : ch->ple_slot[1];
        VkcPle cv = {1, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, 0, 0, ns ? 0 : -1,
                     (int)((size_t)snap * pc), c->eps, 0, (int)ch->o_pconv, (int)((size_t)ch->ple_slot[0] * pc)};
        return ok && vkc_ple(NULL, ch->hyper, NULL, NULL, ch->gated, ch->normv, ch->prm, ch->ple_ring, &cv);
    }
    ok = ok && q38c_res(&ch->gseg, (size_t)n * W, VKC_DEV) && q38c_res(&ch->nseg, (size_t)n * W, VKC_DEV);
    for (int r = 0; r < ns && ok; r++) {
        int len = r < ns - 1 ? 1 : n - r;
        VkcBuf *gb = ch->gated, *nb = ch->normv;
        if (r) {
            ok = vkc_copy(ch->gseg, 0, ch->gated, (size_t)r * W, (size_t)len * W) &&
                 vkc_copy(ch->nseg, 0, ch->normv, (size_t)r * W, (size_t)len * W);
            gb = ch->gseg; nb = ch->nseg;
        }
        VkcPle cv = {1, len, c->hc_count, H, c->ple_convk, c->ngram_size, 0, r * W, 0, 0,
                     (int)((size_t)ch->ple_slot[1 + c0 + r] * pc), c->eps, 0, (int)ch->o_pconv,
                     (int)((size_t)ch->ple_slot[0] * pc)};
        ok = ok && vkc_ple(NULL, ch->hyper, NULL, NULL, gb, nb, ch->prm, ch->ple_ring, &cv);
    }
    return ok;
}
static int q38c_shared(Q38Chain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c; int SI = c->shared_inter;
    VkcEw sw = {VKC_EW_SWIGLU, n * SI, SI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_matmul(q38c_t(&l->sh_g), ch->mixed, 0, ch->gs, 0, n) && vkc_matmul(q38c_t(&l->sh_u), ch->mixed, 0, ch->us, 0, n) &&
           vkc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &sw) && vkc_matmul(q38c_t(&l->sh_d), ch->hs, 0, ch->ds, 0, n) &&
           vkc_matmul(ch->t_sg[i], ch->mixed, 0, ch->sgd, 0, n);
}
/* block = routed + sigmoid(gate) * shared (q38_moe's out), then hyper += inject x block */
static int q38c_moe_apply(Q38Chain *ch, Model *m, int n) {
    Cfg *c = &m->c;
    VkcEw cb = {VKC_EW_COMBINE, n * c->hidden, c->hidden, 1, 1 | 2 | 4 | 8, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->blk, NULL, ch->routed, ch->ds, ch->sgd, &cb) && q38c_gr_apply(ch, m, ch->inj_m, n);
}

/* Every layer for S rows. hyper_h: the rows' streams in, the final ones out when
 * want_streams; mixed_h (or NULL): every row's final mixed row, for the prefill
 * read-out; logit: the last nlogits rows' logits. 0 = not taken (nothing changed), 1 =
 * done. A partial chain runs its N layers, every chunk across all of them, and returns
 * 2: hyper_h holds every row's streams after layer N-1 (the one copy to the host per
 * chunk), and the caller runs layers N.. and the head on the CPU from them. */
/* One chain's layers (lo..lo+nl-1 on its device), from the streams in hyper_h. 0: not
 * taken, *lost = 1 when a frame failed (the state rebuilt on the CPU, the chains off), else
 * nothing on its device changed; 1: done with the head; 2: hyper_h holds the streams after
 * its last layer. */
static int q38c_forward_seg(Model *m, Q38Chain *ch, const int *ids, int S, int pos_base, int nlogits, float *hyper_h,
                            int want_streams, float *mixed_h, float *logit, int *lost) {
    *lost = 0;
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, L = ch->nl, lo = ch->lo, hi = lo + L, E = c->experts, V = c->vocab;
    if (!ch->head) { want_streams = 1; mixed_h = NULL; }   /* a partial chain: the streams after its last layer */
    int mirror_ok = q38c_mirror(ch, m);
    int CH = mirror_ok ? q38c_chunk_rows(ch, m) : 1, rows = S < CH ? S : CH;
    if (ch->ks.on && rows > ch->ks.chunk) rows = ch->ks.chunk;   /* a step's rows fit the split's window */
    if (m->snap_rows > 1 && !q38c_spec_slots(ch, m, m->snap_rows)) {
        fprintf(stderr, "[VK] qwen38 chain: device memory for a verify's %d copies refused; this verify on the CPU\n",
                m->snap_rows);
        return 0;
    }
    if (!mirror_ok || !q38c_scratch(ch, m, rows) || (want_streams && !q38c_res(&ch->hypd, (size_t)rows * W, VKC_DOWN)) ||
        (ch->head && nlogits > 2 && !q38c_res(&ch->outd, (size_t)nlogits * V, VKC_DOWN)) ||
        (mixed_h && !q38c_res(&ch->find, (size_t)rows * H, VKC_DOWN))) {
        fprintf(stderr, "[VK] qwen38 chain: device memory for %d rows refused; per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    int kvo = c->kv_heads * c->head_dim, ID = c->idx_dim, half = c->rotary_dim / 2;
    vkc_gemm_rows(g_q38_rowwise ? 0 : -1);   /* a verify's rows get a decode step's bits */
    int snapped = 0;
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        int ns = m->snap_rows - c0 < n ? m->snap_rows - c0 : n;   /* the rows of this chunk a verify copies */
        if (ns < 0) ns = 0;
        if (ns) snapped = c0 + ns;
        if (!vkc_begin() || !vkc_write(ch->hyper, 0, hyper_h + (size_t)c0 * W, (size_t)n * W * 4) ||
            !q38c_push_state(ch, m, pb, n)) goto lost;
        if (half) {
            float *cs = (float *)vkc_ptr(ch->cs);
            for (int s = 0; s < n; s++) for (int j = 0; j < half; j++) {
                float ang = (float)(pb + s) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
                cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
            }
        }
        if (c->ple_layer >= lo && c->ple_layer < hi) q38c_ple_rows(m, ids, c0, n, ch->host_emb);
        int ok = 1, pending = 0;
        for (int i = lo; i < hi && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) ok = q38c_moe_apply(ch, m, n);
            if (ok && i == c->ple_layer) ok = q38c_ple(ch, m, n, c0, ns);
            ok = ok && q38c_gr_read(ch, m, &l->attn_gr, ch->o_an[i], n, ch->inj_a);
            ok = ok && (c->is_attn[i] ? q38c_attention(ch, m, l, i, n, pb) : q38c_deltanet(ch, m, l, i, n, c0, ns));
            ok = ok && q38c_gr_apply(ch, m, ch->inj_a, n);
            ok = ok && q38c_gr_read(ch, m, &l->mlp_gr, ch->o_mn[i], n, ch->inj_m);
            ok = ok && vkc_matmul(q38c_t(&l->router), ch->mixed, 0, ch->lg, 0, n) &&
                 vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && vkc_copy(ch->mixd, 0, ch->mixed, 0, (size_t)n * H);
            double t0 = now_s();
            /* A1; while it runs, the tier loads the experts this layer will likely
             * stream (a big prompt chunk only) */
            ok = ok && vkc_submit(0);
            if (ok) vkt_stream_prefetch(i, n);
            ok = ok && vkc_finish();
            m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0;
            if (!ok) break;
            if (c->is_attn[i]) {
                const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * q38c_kv_stride(ch, c);
                for (int s = 0; s < n; s++) {
                    for (int h = 0; h < c->kv_heads; h++) {
                        size_t dst = ((size_t)h * m->kv_cap + pb + s) * c->head_dim, src = (size_t)s * kvo + (size_t)h * c->head_dim;
                        memcpy(m->K[i] + dst, kv + src, c->head_dim * sizeof(float));
                        memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->head_dim * sizeof(float));
                    }
                    memcpy(m->IK[i] + (size_t)(pb + s) * ID, kv + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, ID * sizeof(float));
                }
            }
            ok = vkc_begin() && q38c_shared(ch, m, l, i, n) && vkc_submit(0);   /* A2 */
            double t1 = now_s();
            q38_moe_ex(m, l, i, (const float *)vkc_ptr(ch->mixd), n, ch->host_routed, (const float *)vkc_ptr(ch->lgd), 1);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * H * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok) ok = q38c_moe_apply(ch, m, n);
        if (ok && want_streams) ok = vkc_copy(ch->hypd, 0, ch->hyper, 0, (size_t)n * W);
        /* the final mixer for the rows that need it: the read-out wants every row, the
         * logits the last nlogits */
        int lo0 = S - nlogits > c0 ? S - nlogits - c0 : 0, lo_n = n - lo0;   /* this chunk's logit rows */
        int need_final = ch->head && (mixed_h || (lo0 < n && c0 + n > S - nlogits));
        if (ok && need_final) ok = q38c_gr_read(ch, m, &m->final_gr, ch->o_fn, n, NULL);
        if (ok && mixed_h) ok = vkc_copy(ch->find, 0, ch->mixed, 0, (size_t)n * H);
        if (ok && need_final && lo_n > 0 && c0 + n > S - nlogits) {
            int dst = c0 + lo0 - (S - nlogits);
            ok = vkc_matmul(q38c_t(&m->lm_head), ch->mixed, (size_t)lo0 * H, ch->outd, (size_t)dst * V, lo_n);
        }
        double t0 = now_s();
        ok = ok && vkc_submit(1);
        m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0;
        if (!ok) goto lost;
        if (want_streams) memcpy(hyper_h + (size_t)c0 * W, vkc_ptr(ch->hypd), (size_t)n * W * 4);
        if (mixed_h) memcpy(mixed_h + (size_t)c0 * H, vkc_ptr(ch->find), (size_t)n * H * 4);
        for (int i = lo; i < hi; i++) if (c->is_attn[i]) { ch->kv_valid[i] = pb + n; vkc_kv_done(&ch->ks, ch->attn_ord[i], pb + n); }
        ch->dn_where = Q38C_DEV; ch->host_zero = 0;
    }
    if (ch->head) memcpy(logit, vkc_ptr(ch->outd), (size_t)nlogits * V * sizeof(float));
    ch->snap_valid = snapped;
    vkc_gemm_rows(-1);
    ch->forwards++;
    return ch->head ? 1 : 2;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    vkc_gemm_rows(-1);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(ch->d); }
    vkc_device(0);
    q38c_recover(m, pos_base);
    *lost = 1;
    return 0;
}

/* Every layer the chains hold for S rows. hyper_h: the rows' streams in, the final ones
 * out when want_streams; mixed_h (or NULL): every row's final mixed row, for the prefill
 * read-out; logit: the last nlogits rows' logits. 0 = not taken (nothing changed, or a
 * device lost and the state rebuilt: the CPU runs the step), 1 = done. 2: hyper_h holds
 * every row's streams after the layers the chains ran (q38c_layers says how many: the
 * primary's, then the second device's), and the caller runs the rest and the head on the
 * CPU from them. */
static int q38c_forward(Model *m, const int *ids, int S, int pos_base, int nlogits, float *hyper_h,
                        int want_streams, float *mixed_h, float *logit) {
    g_q38c_ran = 0;
    if (!g_vk_chain || qt_ready()) return 0;
    /* prompts only: decode and verifies on the CPU */
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && (S <= 2 || g_q38_rowwise)) return 0;
    Q38Chain *ch = q38c_setup(m), *ch2 = q38c_of(m, 1);
    if (!ch || ch->failed) return 0;
    if (ch2 && (!ch2->ok || ch2->failed || ch2->lo != ch->lo + ch->nl || ch->head)) ch2 = NULL;
    int lost = 0;
    int r = q38c_forward_seg(m, ch, ids, S, pos_base, nlogits, hyper_h, ch2 ? 1 : want_streams, ch2 ? NULL : mixed_h,
                             logit, &lost);
    if (!r) return 0;
    g_q38c_ran = ch->nl;
    if (!ch2 || r == 1) return r;
    vkc_device(1);
    int r2 = q38c_forward_seg(m, ch2, ids, S, pos_base, nlogits, hyper_h, want_streams, mixed_h, logit, &lost);
    vkc_device(0);
    if (lost) { g_q38c_ran = 0; return 0; }
    if (!r2) { q38c_cpu_one(m, ch2, pos_base); return 2; }   /* declined: the CPU runs its layers this step */
    g_q38c_ran += ch2->nl;
    return r2;
}

static void q38c_report(Model *m) {
    for (int d = 0; d < 2 && m; d++) {
        Q38Chain *ch = q38c_of(m, d);
        if (!ch || !ch->ok || !ch->forwards) continue;
        int was = vkc_device(d);
        VkcStats st; vkc_stats(&st);
        fprintf(stderr, "[VK] %s chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                        "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
                q38c_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms,
                st.dev_bytes / 1048576.0);
        vkc_kv_report(&ch->ks);
        vkc_prof_print();
        vkc_device(was);
    }
}
