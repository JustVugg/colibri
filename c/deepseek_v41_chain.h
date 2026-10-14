/* deepseek_v41_chain.h -- DeepSeek V4.1 Flash's layers as a dense chain on the Vulkan
 * device (vk_chain.h). Included once by deepseek_v41.c in a COLI_VULKAN build, before
 * forward_full, whose layer loop it stands in for; COLI_VK_CHAIN decides
 * (coli_vk_chain_decide, this engine's integrated GPU default COLI_VK_CHAIN_UNMEASURED:
 * off, not measured on a V4.1 checkpoint).
 *
 * The residual is hc_mult streams per position (mHC), and a site collapses them with the
 * mix the site BEFORE it computed (forward_full: pre_mix). What runs where, per layer, for
 * one block of rows (decode: one row; a prompt in chunks of COLI_VK_CHAIN_ROWS):
 *   device, frame A1: the previous layer's FFN branch (the routed sum from the host plus
 *     the shared expert) written back into the streams (mHC post); on an engram layer the
 *     engram (eng_wkv over the n-gram rows the host looked up, the gate into each stream);
 *     on a DSpark target layer the streams' mean for the draft head; the attention site:
 *     hc_attn_fn, the split with Sinkhorn, the collapse with the previous site's mix, the
 *     RMSNorm; the attention: wq_a, its norm, wq_b, wkv, its norm, RoPE on interleaved
 *     pairs, the new rows into the window ring; on a kv_source layer the compressor (its
 *     rolling group on the device, the pooled latent's norm), the index keys (idx_wk,
 *     norm, RoPE) and the compressed rows' RoPE; on an index source the indexer (idx_wq_b
 *     and its RoPE, weights_proj, the scores against the keys the CPU would read, the
 *     candidate blocks, the top-k); the sparse attention with the sink over the window
 *     and the selection, the inverse RoPE, the grouped wo_a and wo_b; mHC post; the FFN
 *     site's split, collapse and norm. Then the frame is waited for.
 *   host: the router and the routed experts (moe_run_at without the shared expert: the
 *     expert tier's batch and the CPU's share).
 *   device, frame A2 (not waited for): the shared expert (clamped SwiGLU).
 * The final streams and the last FFN site's mix come back to the host, which collapses
 * them and runs the final norm and the head as before.
 *
 * State, and who owns it: the host's stays canonical. Everything a forward changes there
 * (the window ring and its position map, the compressed rows and index keys, the
 * compressor's group, the published index keys) is written back once the forward's last
 * frame is through, as the CPU would have left it (a speculative verify's undo rows
 * included), so a lost device, a rollback, a reset or a CPU forward never has to rebuild
 * anything. The device mirrors it: the window ring holds window + chunk rows (so no row
 * of a chunk overwrites one an earlier row still reads) behind a watermark win_valid;
 * the compressed rows and index keys of each kv_source layer behind kv_valid[layer]; the
 * compressor's group goes up at every forward's start (two groups of head_dim floats).
 * Every host write lowers the watermarks: a CPU forward (v41c_cpu_step), a rejected
 * draft (v41c_rollback), a reset (v41c_reset). A lost device (COLI_VK_CHAIN_FAULT=n
 * fakes it): the forward runs again on the CPU from its input, which the chain leaves
 * untouched until every chunk is through, and the CPU runs from there.
 *
 * Past the device's budget (vk_kvsplit.h) each kv_source layer's compressed rows keep
 * only some blocks on the device: a window over the newest rows and the blocks the
 * indexer picks most; the rest stay in the host's ckv, which holds every row. A layer's
 * sparse attention stages only selected missing rows in bounded device scratch and
 * reads them alongside resident rows in the original list order (vkc_kv_ds), preserving
 * the complete device result regardless of residency. New compressed rows go through a
 * scratch into their slots, and come back into the host's ckv as soon as their layer's
 * frame is through, so the host part of a later chunk finds every row it reads there.
 * The window ring and the index keys stay whole on the device (every row reads them
 * every step). Below the budget the mirrors are whole, as above.
 *
 * The chain declines (the CPU runs the forward, the watermarks follow) under V41_TRACE,
 * for prompts only when COLI_VK_CHAIN=2 and the forward has two rows or fewer, and for a
 * model its shaders or this file do not take: head_dim above 1024, a window plus top-k
 * above 3072 entries, an indexer above 64 heads or 4096 query floats, more than 8 hyper-
 * connection streams, a compressed layer that is not an index source and reads the list
 * of an earlier index source with another ratio (or of none), a candidate-mask consumer
 * with another ratio than the candidate source.
 *
 * Layers on two devices (docs/vulkan.md, "Layers on two devices"): with COLI_VK_DEV2 a
 * second chain takes the layers after the primary's on that device (its own fit from
 * layer N, COLI_VK_CHAIN_LAYERS2; its own copies of their matrices, which the per-matrix
 * path never sees); the head stays on the host. It starts only at a layer that reads
 * nothing the layers before it make in a forward (v41c_dev2_from: its first compressed
 * layer owns its caches and runs its indexer, and no candidate mask crosses): the
 * primary's fit comes down to that layer when it is the fit's choice. The index keys a
 * layer there reads of an earlier layer (the published-key rule) come from the host,
 * which the primary's forward has written back by then. A forward runs the primary's
 * layers, then the second device's from the streams it hands over; a lost second device
 * leaves the CPU to run its layers from there (the host's state is current: nothing to
 * rebuild), and both chains go off. */
#include "vk_chain.h"
#include "vk_kvsplit.h"

typedef struct {
    int ok, failed;
    int rows;                                  /* scratch rows */
    int Wd, K, W, rmax;                        /* device ring rows, top-k slots, window, largest ratio */
    VkcBuf *prm;
    size_t *o_an, *o_fn, *o_qn, *o_kn, *o_sink, *o_hca, *o_hcf, *o_cn, *o_ikn, *o_eq, *o_ek;
    ColiVkTensor **fna, **fnf;                 /* the mHC mix matrices (f32) */
    VkcBuf **win, **ckv, **ikey, **ring;
    int *kv_valid, *ccap;
    int win_valid;
    VkcBuf *xs, *xn, *mix, *hpa, *hpf, *col, *nrm, *qa, *qr, *q, *kv, *heads, *grp, *br, *craw, *cscr;
    VkcBuf *iq, *ihw, *isc, *mask, *list, *cs, *erows, *ekv, *gs, *us, *hs, *ds;
    VkcBuf *h2d, *mh, *xd, *hpd, *pull, *routed;
    size_t wcap;                               /* score columns per row */
    float *host_routed;
    unsigned long long forwards;
    double host_ms;
    VkcKvSplit ks;                             /* the compressed rows split past the device's budget (ks.on) */
    int *sli, nsl, rmin, planned;              /* per layer its table (-1 none), tables, the least ratio */
    VkcBuf *cnew, *cdn;                        /* a chunk's new compressed rows, and their copy for the host */
    int lo, d;                                 /* its layers start at lo, on device d (vkc_device) */
    int n;                                     /* the layers on the device: lo..lo+n-1 (a partial chain) */
    VkcBuf *cmd, *ld;                          /* the handoff: the candidate mask and the index list, down */
} V41Chain;

static V41Chain *g_v41c;
static int g_vk_chain = 0;
static int g_v41c_inited = 0;
/* the second device's chain (layers g_v41_fit.n..) and its fit */
static V41Chain *g_v41c2;
static VkcFit g_v41c_fit2;
static int g_v41c_fit2_on;
static V41Chain *v41c_of(int d) { return d ? g_v41c2 : g_v41c; }
static const char *v41c_name(const V41Chain *ch) { return ch && ch->d ? "deepseek_v41 dev2" : "deepseek_v41"; }
static int v41c_dev2_wanted(void) { const char *e = getenv("COLI_VK_CHAIN_DEV2"); return !(e && *e == '0'); }
static int v41c_own(const V41Chain *ch, int i) { return i >= ch->lo && i < ch->lo + ch->n; }

/* The window ring's extra rows at setup: COLI_VK_CHAIN_ROWS when set, else 512; a forward
 * whose chunk is bigger grows the rings first (v41c_grow_win). */
static int v41c_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e && strcmp(e, "auto") ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}
/* v41c_res counts instead of reserving while g_v41c_count >= 0 (the chunk's sizing) */
static long long g_v41c_count = -1;
static int v41c_res(VkcBuf **b, size_t floats, int kind) {
    if (g_v41c_count >= 0) { g_v41c_count += (long long)(floats ? floats : 1) * (long long)sizeof(float); return 1; }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}

/* The second device's copies, by host pointer (the per-matrix path's map is the
 * primary's): an open table, grown at half full. */
typedef struct { const void *k; ColiVkTensor *t; } V41cD2;
static V41cD2 *g_v41c_d2;
static size_t g_v41c_d2cap, g_v41c_d2n;
static ColiVkTensor **v41c_d2_slot(const void *k) {
    if ((g_v41c_d2n + 1) * 2 > g_v41c_d2cap) {
        size_t nc = g_v41c_d2cap ? g_v41c_d2cap * 2 : 256;
        V41cD2 *nt = calloc(nc, sizeof *nt);
        if (!nt) return NULL;
        for (size_t i = 0; i < g_v41c_d2cap; i++) {
            if (!g_v41c_d2[i].k) continue;
            size_t h = ((uintptr_t)g_v41c_d2[i].k >> 4) & (nc - 1);
            while (nt[h].k) h = (h + 1) & (nc - 1);
            nt[h] = g_v41c_d2[i];
        }
        free(g_v41c_d2); g_v41c_d2 = nt; g_v41c_d2cap = nc;
    }
    size_t h = ((uintptr_t)k >> 4) & (g_v41c_d2cap - 1);
    while (g_v41c_d2[h].k && g_v41c_d2[h].k != k) h = (h + 1) & (g_v41c_d2cap - 1);
    if (!g_v41c_d2[h].k) { g_v41c_d2[h].k = k; g_v41c_d2n++; }
    return &g_v41c_d2[h].t;
}
static void v41c_d2_forget(const void *k) {   /* its copy freed; the key stays, empty */
    if (!k || !g_v41c_d2cap) return;
    size_t h = ((uintptr_t)k >> 4) & (g_v41c_d2cap - 1);
    while (g_v41c_d2[h].k && g_v41c_d2[h].k != k) h = (h + 1) & (g_v41c_d2cap - 1);
    if (g_v41c_d2[h].k) { coli_vk_tensor_free(g_v41c_d2[h].t); g_v41c_d2[h].t = NULL; }
}
/* fp8 tiles' scales written out for their rows (fmt 12's form) */
static float *v41c_fp8_scales(const W8 *w) {
    int groups = (w->I + FP8_TILE - 1) / FP8_TILE;
    float *scales = malloc((size_t)w->O * groups * sizeof(float));
    if (!scales) return NULL;
    for (int o = 0; o < w->O; o++)
        for (int g = 0; g < groups; g++) scales[(size_t)o * groups + g] = ue8m0(w->s[(size_t)(o / FP8_TILE) * groups + g]);
    return scales;
}
/* The device copies the per-matrix path keeps (vk_entry), uploaded here if it has not:
 * fp8 as fmt 12 with each 32x32 tile's scale written out for its rows, bf16 as fmt 11.
 * With the second device current, its own copies there. */
static ColiVkTensor *v41c_w8(const W8 *w) {
    if (!w->q || w->O < 1 || w->I < 1) return NULL;
    if (vkc_device_now()) {
        ColiVkTensor **t = v41c_d2_slot(w->q);
        if (!t || *t) return t ? *t : NULL;
        float *scales = v41c_fp8_scales(w);
        int ok = scales && coli_vk_tensor_ensure2(t, w->q, scales, 12, w->I, w->O, FP8_TILE);
        free(scales);
        return ok ? *t : NULL;
    }
    VkEntry *e = vk_entry(w->q, 12, w->O, w->I);
    if (!e || e->refused) return NULL;
    if (e->t) return e->t;
    v41_dho_host(w->q);   /* not placed this way (device only): its rows back first */
    float *scales = v41c_fp8_scales(w);
    if (!scales) return NULL;
    int ok = coli_vk_tensor_ensure(&e->t, w->q, scales, 12, w->I, w->O, FP8_TILE);
    free(scales);
    return ok ? e->t : NULL;
}
static ColiVkTensor *v41c_wb(const WB *w) {
    if (!w->w || w->O < 1 || w->I < 1) return NULL;
    if (vkc_device_now()) {
        ColiVkTensor **t = v41c_d2_slot(w->w);
        if (!t || *t) return t ? *t : NULL;
        return coli_vk_tensor_ensure2(t, w->w, NULL, 11, w->I, w->O, 0) ? *t : NULL;
    }
    VkEntry *e = vk_entry(w->w, 11, w->O, w->I);
    if (!e || e->refused) return NULL;
    if (e->t) return e->t;
    v41_dho_host(w->w);
    return coli_vk_tensor_ensure(&e->t, w->w, NULL, 11, w->I, w->O, 0) ? e->t : NULL;
}

/* ---- the host's writes: the device's copies follow ------------------------------------- */
static void v41c_lower_one(Model *m, V41Chain *ch, int pos) {
    if (!ch || !ch->ok) return;
    if (pos < 0) pos = 0;
    if (ch->win_valid > pos) ch->win_valid = pos;
    /* every kv_source layer: the chain's own, and (a partial chain) another layer whose
     * index keys a chain layer reads; the split's tables are the chain's layers' only */
    for (int i = 0; i < m->c.n_layers; i++) {
        int r = m->c.compress_ratio[i];
        if (m->c.kv_source[i] && r > 0 && ch->kv_valid[i] > pos / r) ch->kv_valid[i] = pos / r;
        if (m->c.kv_source[i] && r > 0 && ch->sli[i] >= 0) vkc_kv_lower(&ch->ks, ch->sli[i], pos / r);
    }
}
static void v41c_lower(Model *m, int pos) { for (int d = 0; d < 2; d++) v41c_lower_one(m, v41c_of(d), pos); }   /* both chains */
static void v41c_cpu_step(Model *m, int start_pos) { v41c_lower(m, start_pos); }      /* a CPU forward from start_pos */
static void v41c_rollback(Model *m, int committed_end) { v41c_lower(m, committed_end); }  /* rows past it rejected */
static void v41c_reset(Model *m) { v41c_lower(m, 0); }

/* ---- setup ----------------------------------------------------------------------------- */
static const char *v41c_unsupported(const Model *m) {
    const Cfg *c = &m->c;
    int L = c->n_layers, K = c->index_topk > 0 ? c->index_topk : 0;
    if (c->head_dim > 1024 || c->head_dim < 1 || c->rope_dim < 2 || (c->rope_dim & 1) || c->rope_dim > c->head_dim)
        return "an attention geometry its shaders do not take";
    if (c->window + K > 3072 || c->window < 1) return "a window and top-k its attention does not take";
    if (c->hc_mult < 1 || c->hc_mult > 8 || c->hc_iters < 1) return "hyper-connections its shaders do not take";
    if ((c->n_heads * c->head_dim) % c->o_groups) return "an output grouping it does not take";
    int any_index = 0;
    for (int i = 0; i < L; i++) any_index |= c->index_source[i] && c->compress_ratio[i] > 0;
    if (any_index && (c->index_n_heads < 1 || c->index_n_heads > 64 || c->index_n_heads * c->index_head_dim > 4096 ||
                      c->index_head_dim < c->rope_dim || K < 1 || K > 4096)) return "an indexer its shaders do not take";
    int pub = -1;   /* the ratio of the last index source's list */
    for (int i = 0; i < L; i++) {
        int r = c->compress_ratio[i];
        if (r <= 0) continue;
        if (c->index_owner[i] < 0) return "a compressed layer with no kv_source before it";
        if (c->kv_source[i] && r > c->max_positions) return "a compression ratio past the context";
        if (c->index_source[i]) pub = r;
        else if (pub != r) return "a compressed layer that reads another ratio's index list";
    }
    int cs = c->candidate_source;
    if (cs >= 0 && cs < L && c->index_source[cs] && c->compress_ratio[cs] > 0) {
        if (c->candidate_block_size < 1) return "candidate blocks it does not take";
        for (int i = cs + 1; i < L; i++)
            if (c->index_source[i] && c->compress_ratio[i] > 0 && c->compress_ratio[i] != c->compress_ratio[cs])
                return "a candidate mask read by a layer of another ratio";
    }
    return NULL;
}
static int v41c_cand_on(const Cfg *c) {
    int cs = c->candidate_source;
    return cs >= 0 && cs < c->n_layers && c->index_source[cs] && c->compress_ratio[cs] > 0;
}
/* Whether the second device's chain may start at layer b: its layers read nothing the
 * layers before b make in a forward but index keys (the host's, written back by then).
 * The first compressed layer at or after b owns its caches and runs its indexer (no
 * compressed rows or index list from before b), and no candidate mask made before b is
 * read at or after it. */
static int v41c_clean_at(const Model *m, int b) {
    const Cfg *c = &m->c; int L = c->n_layers;
    for (int i = b; i < L; i++) {
        if (c->compress_ratio[i] <= 0) continue;
        if (!c->kv_source[i] || !c->index_source[i]) return 0;
        break;
    }
    if (v41c_cand_on(c) && c->candidate_source < b)
        for (int i = b; i < L; i++) if (c->index_source[i] && c->compress_ratio[i] > 0) return 0;
    return 1;
}

/* ---- the partial chain (vk_chain.h, vkc_fit): the first N layers on the device ------------
 * N is decided once, before anything goes up (v41c_fit_now: at v41_dho_open when the trunk
 * may live on the device only, else when the device opens after the weights): the layers
 * that fit the device's free memory less the tier's reserve, the chain's scratch for one
 * prompt chunk and the pools' granularity. The CPU runs layers N.. and the head; the
 * streams of every row and the last site's mix cross to the host once per forward
 * (v41c_forward), with the host state the CPU's layers read from a chain layer: the
 * candidate mask (m->candidates) and the index list (m->shared_topk). A chain layer that
 * reads the index keys of a CPU layer (the published-key rule) reads the host's rows, as
 * the CPU's layer order has them, from a mirror of that layer's keys. */
static int v41c_scratch(V41Chain *ch, Model *m, int rows, int ctx);
static int v41c_engram_layer(const Model *m, int i) {
    for (int t = 0; m->engram.active && t < m->engram.n_layers; t++) if (m->engram.layer_of[t] == i) return 1;
    return 0;
}
/* What layer i puts on the device, as vkc_fit counts it: its matrices in the forms they go
 * up in (dho: the device-only placement's set, views: wo_a per output group as well), its
 * state at its first size (the window ring, the compressor's group, 64 compressed rows and
 * keys), its part of the parameters and of a forward's pull buffer at `rows` rows; *mat
 * its matrices' payload (coli_vk_mem_info's count). From the config alone: it runs before
 * the layers are read. */
static size_t v41c_layer_bytes(const Model *m, int i, int rows, int dho, int views, size_t *mat) {
    const Cfg *c = &m->c;
    int D = c->dim, H = c->hc_mult, nm = (2 + H) * H, hd = c->head_dim, nh = c->n_heads, r = c->compress_ratio[i];
    int IH = c->index_n_heads, ID = c->index_head_dim, og = c->o_groups, ol = c->o_lora, W = c->window, QL = c->q_lora;
    int eng = v41c_engram_layer(m, i);
    size_t b = 0, mm = 0;
    #define T(fmt, I, O, gs) do { b += vkc_fit_tensor(fmt, I, O, gs); mm += coli_vk_tensor_payload(fmt, I, O, gs); } while (0)
    T(10, H * D, nm, 0); T(10, H * D, nm, 0);                          /* the mHC mix matrices */
    T(12, D, QL, FP8_TILE); T(12, QL, nh * hd, FP8_TILE); T(12, D, hd, FP8_TILE);
    T(12, nh * hd / og, og * ol, FP8_TILE);                            /* wo_a whole, the chain's */
    for (int g = 0; views && g < og; g++) T(12, nh * hd / og, ol, FP8_TILE);   /* and per group, the per-matrix path's */
    T(12, og * ol, D, FP8_TILE);
    T(12, D, c->moe_inter, FP8_TILE); T(12, D, c->moe_inter, FP8_TILE); T(12, c->moe_inter, D, FP8_TILE);
    if (c->kv_source[i]) { T(11, D, hd, 0); if (r > 1) T(11, D, hd, 0); T(11, hd, ID, 0); }
    if (c->index_source[i] && (r > 0 || dho)) { T(12, QL, IH * ID, FP8_TILE); T(11, D, IH, 0); }
    if (eng) T(12, m->engram.cols * m->engram.head_dim, D * (H + 1), FP8_TILE);
    #undef T
    b += vkc_fit_buf((size_t)(W + v41c_rows()) * hd * sizeof(float));
    size_t f = (size_t)2 * D + QL + hd + nh + (size_t)2 * (3 + nm);
    if (c->kv_source[i] && r > 0) {
        int host = c->max_positions / r, first = host < 64 ? (host > 0 ? host : 1) : 64;
        if (r > 1) b += vkc_fit_buf((size_t)2 * r * hd * sizeof(float));
        b += vkc_fit_buf((size_t)first * hd * sizeof(float)) + vkc_fit_buf((size_t)first * ID * sizeof(float));
        f += hd + ID + (size_t)(rows / r) * (hd + ID) + (r > 1 ? (size_t)2 * r * hd : 0);
    }
    if (eng) f += (size_t)2 * H * D;
    f += (size_t)(rows < W ? rows : W) * hd;
    *mat = mm;
    return b + f * sizeof(float);
}
/* What the chain allocates whatever N: the scratch of one prompt chunk of `rows` rows at
 * window + rows positions (v41c_scratch's counting pass). */
static size_t v41c_fixed_bytes(Model *m, int rows) {
    const Cfg *c = &m->c;
    V41Chain t;
    memset(&t, 0, sizeof t);
    t.W = c->window; t.K = c->index_topk > 0 ? c->index_topk : 0; t.rmax = 1;
    for (int i = 0; i < c->n_layers; i++) if (c->kv_source[i] && c->compress_ratio[i] > t.rmax) t.rmax = c->compress_ratio[i];
    g_v41c_count = 0;
    v41c_scratch(&t, m, rows, t.W + rows);
    size_t b = (size_t)g_v41c_count;
    g_v41c_count = -1;
    return b;
}
/* With every layer on the device, the per-matrix path (COLI_VK_DENSE) also puts the bf16
 * head and each layer's router there as they are first used (and, for prompts only, wo_a
 * per output group): the tail. */
static size_t v41c_tail_bytes(const Model *m, int views) {
    if (!coli_vk_dense()) return 0;
    const Cfg *c = &m->c;
    size_t b = vkc_fit_tensor(11, c->dim, c->vocab, 0) + (size_t)c->n_layers * vkc_fit_tensor(11, c->dim, c->n_routed, 0);
    if (views) b += (size_t)c->n_layers * c->o_groups * vkc_fit_tensor(12, c->n_heads * c->head_dim / c->o_groups, c->o_lora, FP8_TILE);
    return b;
}
static int v41c_fit_now(Model *m) {
    if (g_v41_fit_done) return g_v41_fit.L > 0;
    g_v41_fit_done = 1;
    if (!g_vk_ready) return 0;
    int chain = v41c_decide(m);
    if (!chain || v41c_unsupported(m)) return 0;
    const Cfg *c = &m->c;
    int L = c->n_layers, rows = vkc_fit_rows(512), views = chain == COLI_VK_CHAIN_PREFILL;
    size_t *per = calloc((size_t)L, sizeof(size_t)), *mat = calloc((size_t)L, sizeof(size_t));
    if (!per || !mat) { free(per); free(mat); return 0; }
    for (int i = 0; i < L; i++) per[i] = v41c_layer_bytes(m, i, rows, g_v41_dho_try, views && g_v41_dho_try, &mat[i]);
    size_t fixed = v41c_fixed_bytes(m, rows);
    vkc_fit("deepseek_v41", L, per, mat, fixed, v41c_tail_bytes(m, views && !g_v41_dho_try), &g_v41_fit);
    g_v41_partial = vkc_fit_partial(&g_v41_fit);
    /* the layers the primary leaves, on COLI_VK_DEV2's device: from a layer that reads
     * nothing the layers before it make (v41c_clean_at; the fit's N comes down to the last
     * such layer, a forced one stays and the second device stays off), its own fit (its
     * layers' own set: no device-only placement there, the head on the host), its
     * pipelines up now */
    int n0 = g_v41_fit.n;
    if (g_v41_partial && n0 > 0 && n0 < L && v41c_dev2_wanted() && getenv("COLI_VK_DEV2") && coli_vk_dev2_open_env()) {
        int b = n0;
        while (b > 0 && !v41c_clean_at(m, b)) b--;
        if (b != n0 && g_v41_fit.forced) {
            fprintf(stderr, "[VK] deepseek_v41 chain: layer %d reads what the layers before it make in a forward; the second device's "
                            "chain starts only at a layer that does not (COLI_VK_CHAIN_LAYERS=%d would do): its layers stay on the CPU\n",
                    n0, b);
            b = 0;
        } else if (b != n0 && b > 0) {
            g_v41_fit.n = b; g_v41_fit.tail = 0; g_v41_fit.used_b = 0;
            for (int i = 0; i < b; i++) g_v41_fit.used_b += per[i];
            fprintf(stderr, "[VK] deepseek_v41 chain: %d of %d layers on the device: the second device's layers start at layer %d, "
                            "the last at or before %d that reads nothing the layers before it make in a forward\n", b, L, b, n0);
        }
        if (b > 0) {
            for (int i = b; i < L; i++) per[i] = v41c_layer_bytes(m, i, rows, 0, 0, &mat[i]);
            vkc_device(1);
            int n2 = vkc_fit("deepseek_v41 dev2", L - b, per + b, mat + b, fixed, 0, &g_v41c_fit2);
            if (n2 > 0 && !(vkc_init() && vkc_mla_ready() && vkc_mhc_ready() && vkc_dsv4_ready())) {
                fprintf(stderr, "[VK] deepseek_v41 chain: the second device's pipelines did not come up; its layers stay on the CPU\n");
                n2 = 0;
            }
            g_v41c_fit2_on = n2 > 0;
            vkc_device(0);
        }
    }
    free(per); free(mat);
    return 1;
}
/* The device-only placement was not chosen after all (the auto rule): the layers' matrices
 * are the chain's set alone, as its setup will place them (the placed line's count). */
static void v41c_fit_recount(Model *m) {
    if (g_v41_fit.L < 1 || !g_v41_fit.per || !g_v41_fit.mat) return;
    int rows = vkc_fit_rows(512);
    for (int i = 0; i < g_v41_fit.L; i++) g_v41_fit.per[i] = v41c_layer_bytes(m, i, rows, 0, 0, &g_v41_fit.mat[i]);
}

/* Layer i's state on the device, nothing of it read from disk: its mHC mix matrices (f32,
 * the host keeps them), its window ring and its compressor's group. Kept when there. */
static int v41c_layer_state(V41Chain *ch, Model *m, int i, const char **why) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    int H = c->hc_mult, D = c->dim, nm = (2 + H) * H, r = c->compress_ratio[i];
    *why = "a matrix the device refused";
    if (ch->d ? !coli_vk_tensor_ensure2(&ch->fna[i], l->hc_attn_fn.w, NULL, 10, H * D, nm, 0) ||
                !coli_vk_tensor_ensure2(&ch->fnf[i], l->hc_ffn_fn.w, NULL, 10, H * D, nm, 0)
              : !coli_vk_tensor_ensure(&ch->fna[i], l->hc_attn_fn.w, NULL, 10, H * D, nm, 0) ||
                !coli_vk_tensor_ensure(&ch->fnf[i], l->hc_ffn_fn.w, NULL, 10, H * D, nm, 0)) return 0;
    *why = "device memory for its state refused";
    if (!ch->win[i] && !(ch->win[i] = vkc_buf((size_t)ch->Wd * c->head_dim * sizeof(float), VKC_DEV))) return 0;
    if (c->kv_source[i] && r > 1 && !ch->ring[i] && !(ch->ring[i] = vkc_buf((size_t)2 * r * c->head_dim * sizeof(float), VKC_DEV)))
        return 0;
    return 1;
}
/* Layer i's tensors and state on the device (its matrices through the per-matrix path's
 * map: already there when the trunk lives on the device only). 0 with *why set. */
static int v41c_layer_up(V41Chain *ch, Model *m, int i, const char **why) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    int r = c->compress_ratio[i];
    *why = "a matrix the device refused";
    int ok = v41c_w8(&l->wq_a) && v41c_w8(&l->wq_b) && v41c_w8(&l->wkv) && v41c_w8(&l->wo_a) && v41c_w8(&l->wo_b) &&
             v41c_w8(&l->sh_w1) && v41c_w8(&l->sh_w3) && v41c_w8(&l->sh_w2);
    if (ok && c->kv_source[i]) ok = v41c_wb(&l->comp_wkv) && (r <= 1 || v41c_wb(&l->comp_wgate)) && v41c_wb(&l->idx_wk);
    if (ok && c->index_source[i] && r > 0) ok = v41c_w8(&l->idx_wq_b) && v41c_wb(&l->idx_wproj);
    if (ok && l->eng_wkv.q) ok = v41c_w8(&l->eng_wkv) != NULL;
    return ok && v41c_layer_state(ch, m, i, why);
}
/* Everything of layer i off the device: the chain's tensors and buffers, and the device
 * copies of its matrices (their entries refused: the CPU multiplies them from now on). */
static void v41c_layer_free(V41Chain *ch, Model *m, int i) {
    Layer *l = &m->L[i];
    if (vkc_ready()) vkc_finish();
    coli_vk_tensor_free(ch->fna[i]); ch->fna[i] = NULL;
    coli_vk_tensor_free(ch->fnf[i]); ch->fnf[i] = NULL;
    VkcBuf **bs[] = {&ch->win[i], &ch->ring[i], &ch->ckv[i], &ch->ikey[i]};
    for (size_t k = 0; k < sizeof bs / sizeof *bs; k++) { vkc_free(*bs[k]); *bs[k] = NULL; }
    ch->ccap[i] = ch->kv_valid[i] = 0;
    const W8 *w8[] = {&l->wq_a, &l->wq_b, &l->wkv, &l->wo_a, &l->wo_b, &l->sh_w1, &l->sh_w3, &l->sh_w2, &l->idx_wq_b, &l->eng_wkv};
    const WB *wb[] = {&l->comp_wkv, &l->comp_wgate, &l->idx_wk, &l->idx_wproj};
    if (ch->d) {   /* the second device's own copies */
        for (size_t k = 0; k < sizeof w8 / sizeof *w8; k++) v41c_d2_forget(w8[k]->q);
        for (size_t k = 0; k < sizeof wb / sizeof *wb; k++) v41c_d2_forget(wb[k]->w);
        return;
    }
    for (size_t k = 0; k < sizeof w8 / sizeof *w8; k++) if (w8[k]->q) v41_vk_forget_range(w8[k]->q, (size_t)w8[k]->O * w8[k]->I);
    for (size_t k = 0; k < sizeof wb / sizeof *wb; k++) if (wb[k]->w) v41_vk_forget_range(wb[k]->w, (size_t)wb[k]->O * wb[k]->I * 2);
}
/* Layer i did not reach the device: it and the layers after it off the device (with the
 * trunk on the device only, their host copies read back), the chain cut before it. */
static void v41c_cut(V41Chain *ch, Model *m, int i, const char *why) {
    for (int k = i; k < ch->lo + ch->n; k++) v41c_layer_free(ch, m, k);
    if (ch->lo + ch->n > i) ch->n = i - ch->lo;
    if (ch->d) { vkc_fit_shrink("deepseek_v41 dev2", &g_v41c_fit2, i - ch->lo, why); return; }
    if (g_v41_dho) v41_dho_unplace(m, i);
    vkc_fit_shrink("deepseek_v41", &g_v41_fit, i, why);
}

/* The chain's struct, made once: by the device-only placement, which builds each of the
 * chain's layers whole as model_load reads it (v41c_build_layer), else by the setup. */
static V41Chain *v41c_alloc_dev(Model *m, int d) {
    if (v41c_of(d)) return v41c_of(d);
    Cfg *c = &m->c;
    int L = c->n_layers;
    V41Chain *ch = calloc(1, sizeof *ch);
    if (!ch) return NULL;
    ch->d = d; ch->lo = d ? g_v41_fit.n : 0;
    size_t **offs[] = {&ch->o_an, &ch->o_fn, &ch->o_qn, &ch->o_kn, &ch->o_sink, &ch->o_hca, &ch->o_hcf, &ch->o_cn, &ch->o_ikn,
                       &ch->o_eq, &ch->o_ek};
    for (size_t k = 0; k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = calloc(L, sizeof(size_t)))) return NULL;
    ch->fna = calloc(L, sizeof(void *)); ch->fnf = calloc(L, sizeof(void *));
    ch->win = calloc(L, sizeof(void *)); ch->ckv = calloc(L, sizeof(void *));
    ch->ikey = calloc(L, sizeof(void *)); ch->ring = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->ccap = calloc(L, sizeof(int)); ch->sli = malloc(L * sizeof(int));
    if (!ch->fna || !ch->fnf || !ch->win || !ch->ckv || !ch->ikey || !ch->ring || !ch->kv_valid || !ch->ccap || !ch->sli) return NULL;
    ch->n = d ? g_v41c_fit2.n : g_v41_fit.L > 0 ? g_v41_fit.n : L;
    ch->W = c->window; ch->K = c->index_topk > 0 ? c->index_topk : 0; ch->rmax = 1;
    for (int i = 0; i < L; i++) if (c->kv_source[i] && c->compress_ratio[i] > ch->rmax) ch->rmax = c->compress_ratio[i];
    ch->Wd = ch->W + v41c_rows();
    if (d) g_v41c2 = ch; else g_v41c = ch;
    return ch;
}
static V41Chain *v41c_alloc(Model *m) { return v41c_alloc_dev(m, 0); }
/* The device-only placement of layer i put its matrices up (model_load, v41_dho_place):
 * its state goes up too before its host pages are given back, so the layer reaches the
 * device whole or not at all and a failure reads nothing back. 0 with *why set. */
static int g_v41c_build;   /* the chain's layers are built in the placement (v41_dho_open) */
static int v41c_build_layer(Model *m, int i, const char **why) {
    V41Chain *ch = g_v41c;
    if (!g_v41c_build || !ch || i >= ch->n) return 1;
    return v41c_layer_state(ch, m, i, why);
}

/* Layer i did not reach the device (v41_dho_place, as model_load reads it): the chain cut
 * before it, whatever of it and of the layers after it is up freed. */
static void v41c_cut_layer(Model *m, int i, const char *why) {
    if (g_v41c) { v41c_cut(g_v41c, m, i, why); return; }
    if (g_v41_dho) v41_dho_unplace(m, i);
    vkc_fit_shrink("deepseek_v41", &g_v41_fit, i, why);
}
/* Before the layers are read, when the trunk lives on the device only: the chain's
 * pipelines and its struct, so the placement builds each of its layers whole. 0: it will
 * not (the setup says why later). */
static int v41c_prepare(Model *m) {
    if (!g_v41_dho || g_v41_fit.L < 1 || g_v41_fit.n < 1 || !v41c_decide(m) || v41c_unsupported(m)) return 0;
    if (!(g_v41c_inited = vkc_init()) || !(vkc_mla_ready() && vkc_mhc_ready() && vkc_dsv4_ready())) return 0;
    return g_v41c_build = v41c_alloc(m) != NULL;
}

/* d = 1: the second device's chain, its layers from g_v41_fit.n (device 1 current). */
static int v41c_setup_dev(Model *m, int d) {
    Cfg *c = &m->c;
    int L = c->n_layers, D = c->dim, H = c->hc_mult, nm = (2 + H) * H;
    const char *why = v41c_unsupported(m);
    if (why) { fprintf(stderr, "[VK] deepseek_v41 chain: %s; the CPU runs the layers\n", why); return 0; }
    if ((d ? g_v41c_fit2.n : g_v41_fit.L > 0 ? g_v41_fit.n : L) < 1) return 0;
    V41Chain *ch = v41c_alloc_dev(m, d);
    if (!ch) return 0;
    VkcFit *fit = d ? &g_v41c_fit2 : &g_v41_fit;
    const char *nmc = v41c_name(ch);
    int N = ch->n, lo = ch->lo;
    if (N < 1) return 0;
    /* the parameter arena, the chain's layers' */
    size_t n = 0;
    for (int i = lo; i < lo + N; i++) {
        ch->o_an[i] = n; n += D; ch->o_fn[i] = n; n += D;
        ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += c->head_dim;
        ch->o_sink[i] = n; n += c->n_heads;
        ch->o_hca[i] = n; n += 3 + nm; ch->o_hcf[i] = n; n += 3 + nm;
        if (c->kv_source[i]) { ch->o_cn[i] = n; n += c->head_dim; ch->o_ikn[i] = n; n += c->index_head_dim; }
        if (m->L[i].engram_index >= 0) { ch->o_eq[i] = n; n += (size_t)H * D; ch->o_ek[i] = n; n += (size_t)H * D; }
    }
    if (!(ch->prm = vkc_buf((n ? n : 1) * sizeof(float), VKC_DEV))) {
        v41c_cut(ch, m, lo, vkc_lost() ? "the device was lost" : "device memory for the parameters refused");
        vkc_fit_placed(nmc, fit);
        return 0;
    }
    /* the layers, each whole or the chain stops before it */
    for (int i = lo; i < lo + N; i++) {
        if (!v41c_layer_up(ch, m, i, &why)) { v41c_cut(ch, m, i, vkc_lost() ? "the device was lost" : why); break; }
        if (d || !g_v41_dho) vkc_fit_mark(fit, i - lo);   /* device only: noted as each layer was placed */
    }
    N = ch->n;
    for (int i = 0; i < L; i++) {               /* the split's tables: one per chain layer that makes compressed rows */
        ch->sli[i] = v41c_own(ch, i) && c->kv_source[i] && c->compress_ratio[i] > 0 ? ch->nsl++ : -1;
        if (ch->sli[i] >= 0 && (ch->rmin == 0 || c->compress_ratio[i] < ch->rmin)) ch->rmin = c->compress_ratio[i];
    }
    int ok = N > 0;
    float *a = ok ? calloc(n ? n : 1, sizeof(float)) : NULL;
    ok = ok && a;
    for (int i = lo; ok && i < lo + N; i++) {
        Layer *l = &m->L[i];
        memcpy(a + ch->o_an[i], l->attn_norm.w, D * sizeof(float));
        memcpy(a + ch->o_fn[i], l->ffn_norm.w, D * sizeof(float));
        memcpy(a + ch->o_qn[i], l->q_norm.w, c->q_lora * sizeof(float));
        memcpy(a + ch->o_kn[i], l->kv_norm.w, c->head_dim * sizeof(float));
        memcpy(a + ch->o_sink[i], l->attn_sink.w, c->n_heads * sizeof(float));
        memcpy(a + ch->o_hca[i], l->hc_attn_scale.w, 3 * sizeof(float));
        memcpy(a + ch->o_hca[i] + 3, l->hc_attn_base.w, nm * sizeof(float));
        memcpy(a + ch->o_hcf[i], l->hc_ffn_scale.w, 3 * sizeof(float));
        memcpy(a + ch->o_hcf[i] + 3, l->hc_ffn_base.w, nm * sizeof(float));
        if (c->kv_source[i]) {
            memcpy(a + ch->o_cn[i], l->comp_norm.w, c->head_dim * sizeof(float));
            memcpy(a + ch->o_ikn[i], l->idx_knorm.w, c->index_head_dim * sizeof(float));
        }
        if (l->engram_index >= 0) {
            memcpy(a + ch->o_eq[i], l->eng_q.w, (size_t)H * D * sizeof(float));
            memcpy(a + ch->o_ek[i], l->eng_k.w, (size_t)H * D * sizeof(float));
        }
    }
    ok = ok && vkc_begin() && vkc_write(ch->prm, 0, a, n * sizeof(float)) && vkc_submit(1);
    free(a);
    if (!ok) {
        if (N > 0) {
            if (!d) fprintf(stderr, "[VK] deepseek_v41 chain: the parameters did not reach the device; the CPU runs the layers\n");
            v41c_cut(ch, m, lo, vkc_lost() ? "the device was lost" : "the parameters did not reach it");
        }
        vkc_free(ch->prm); ch->prm = NULL;
        vkc_fit_placed(nmc, fit);
        return 0;
    }
    ch->ok = 1;
    int nsrc = 0, nidx = 0, neng = 0;
    for (int i = lo; i < lo + N; i++) { nsrc += c->kv_source[i]; nidx += c->index_source[i] && c->compress_ratio[i] > 0; neng += m->L[i].engram_index >= 0; }
    size_t bytes = 0, tensors = 0;
    coli_vk_mem_info_dev(d, &bytes, &tensors);
    if (d) fprintf(stderr, "[VK] deepseek_v41 chain: layers %d..%d on the second device (%d compressing, %d indexing, %d engram), "
                           "%d streams, %.1f MiB of parameters, %zu matrices (%.1f MiB) there\n", lo, lo + N - 1, nsrc, nidx, neng, H,
                   n * 4 / 1048576.0, tensors, bytes / 1048576.0);
    else fprintf(stderr, "[VK] deepseek_v41 chain: %d layers on the device (%d compressing, %d indexing, %d engram), %d streams, "
                         "%.1f MiB of parameters, %zu matrices (%.1f MiB) on the device\n", N, nsrc, nidx, neng, H, n * 4 / 1048576.0,
                 tensors, bytes / 1048576.0);
    vkc_fit_placed(nmc, fit);
    return 1;
}
static int v41c_setup(Model *m) { return v41c_setup_dev(m, 0); }

/* The compressed rows of the kv_source layers, whole or split: planned when the whole
 * mirrors would grow (and at the first forward); once split, the split stays. ctx: the
 * positions the forward reaches. 0 = the device refused the split's rows. */
static int v41c_plan(V41Chain *ch, Model *m, int ctx) {
    Cfg *c = &m->c;
    if (!ch->nsl || ch->ks.on) return 1;
    int L = c->n_layers, hd = c->head_dim, tcap = 1, grow = !ch->planned;
    size_t need = 0, held = 0;
    for (int i = 0; i < L; i++) {
        if (ch->sli[i] < 0) continue;
        int r = c->compress_ratio[i], host = c->max_positions / r, want = ctx / r > 0 ? ctx / r : 1, cap = ch->ccap[i];
        if (host > tcap) tcap = host;
        if (cap < want) { cap = 64; while (cap < want) cap *= 2; if (cap > host) cap = host; grow = 1; }
        need += (size_t)cap * hd * sizeof(float); held += (size_t)ch->ccap[i] * hd * sizeof(float);
    }
    if (!grow) return 1;
    ch->planned = 1;
    /* a window of half the slots, the rest pinned by the indexer's picks; the forward's
     * chunks shrink to what the window takes (v41c_forward) */
    if (vkc_kv_plan_need(&ch->ks, v41c_name(ch), ch->nsl, (size_t)hd * sizeof(float), tcap, 1, 1, held, need) != 2) return 1;
    for (int i = 0; i < L; i++) {
        if (ch->sli[i] < 0) continue;
        vkc_free(ch->ckv[i]);
        if (!(ch->ckv[i] = vkc_buf((size_t)ch->ks.rows * hd * sizeof(float), VKC_DEV))) return 0;
    }
    return 1;
}

/* the compressed rows and index keys of a kv_source layer: room for `rows` rows (a CPU
 * layer of a partial chain: its index keys only, which a chain layer reads) */
static int v41c_cache(V41Chain *ch, Model *m, int i, int rows) {
    Cfg *c = &m->c;
    if (ch->ccap[i] >= rows) return 1;
    int r = c->compress_ratio[i], host = c->max_positions / (r > 0 ? r : 1), own = v41c_own(ch, i);
    int cap = 64; while (cap < rows) cap *= 2;
    if (cap > host) cap = host;
    if (cap < rows) return 0;
    if (own && !ch->ks.on) {   /* under the split the compressed rows keep their ks.rows */
        vkc_free(ch->ckv[i]);
        ch->ckv[i] = vkc_buf((size_t)cap * c->head_dim * sizeof(float), VKC_DEV);
    }
    vkc_free(ch->ikey[i]);
    ch->ikey[i] = vkc_buf((size_t)cap * c->index_head_dim * sizeof(float), VKC_DEV);
    ch->kv_valid[i] = 0; ch->ccap[i] = 0;
    if ((own && !ch->ckv[i]) || !ch->ikey[i]) return 0;
    ch->ccap[i] = cap;
    return 1;
}

/* Prompt rows per chunk (vkc_chunk_rows, decided at the first forward, after the tier
 * filled): the chain's scratch a row at the model's context (the indexer's scores and
 * mask) at the forward's context, the window ring's row, and the routed experts' outputs. */
static int v41c_chunk_rows(V41Chain *ch, Model *m, int ctx) {
    if (!vkc_chunk_auto()) return v41c_rows();
    size_t wcap = ch->wcap;
    g_v41c_count = 0; v41c_scratch(ch, m, 1, ctx); long long b1 = g_v41c_count;
    g_v41c_count = 0; v41c_scratch(ch, m, 2, ctx); long long b2 = g_v41c_count;
    g_v41c_count = -1; ch->wcap = wcap;
    const Cfg *c = &m->c;
    size_t row = (size_t)(b2 - b1) + (size_t)ch->n * c->head_dim * sizeof(float) +
                 (size_t)(2 * c->n_activated + 1) * c->dim * sizeof(float);
    /* The CPU expert path gathers one expert's rows and keeps its gate/up and
     * output beside the rank-order contribution buffer. */
    row += 2 * ((size_t)c->dim + c->moe_inter) * sizeof(float);
    if (ch->ks.on) row += (size_t)(ch->W + ch->K) * sizeof(int);
    return vkc_chunk_rows(v41c_name(ch), row);
}
/* The window rings for chunks of `rows`: W + rows rows each (a chunk never overwrites a
 * row one of its earlier queries reads); a ring that grows is mirrored again. */
static int v41c_grow_win(V41Chain *ch, Model *m, int rows) {
    if (ch->W + rows <= ch->Wd) return 1;
    const Cfg *c = &m->c;
    int Wd = ch->W + rows;
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        vkc_free(ch->win[i]);
        if (!(ch->win[i] = vkc_buf((size_t)Wd * c->head_dim * sizeof(float), VKC_DEV))) return 0;
    }
    ch->Wd = Wd; ch->win_valid = 0;
    return 1;
}
static int v41c_scratch(V41Chain *ch, Model *m, int rows, int ctx) {
    Cfg *c = &m->c;
    int D = c->dim, H = c->hc_mult, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H, nh = c->n_heads, hd = c->head_dim;
    int IH = c->index_n_heads > 0 ? c->index_n_heads : 1, ID = c->index_head_dim > 0 ? c->index_head_dim : 1;
    int ew = m->engram.active ? m->engram.cols * m->engram.head_dim : 1, T = c->n_spec_targets > 0 ? c->n_spec_targets : 1;
    size_t r = (size_t)rows;
    ch->wcap = (size_t)ctx;
    int ok = v41c_res(&ch->xs, r * HD, VKC_DEV) && v41c_res(&ch->xn, r * HD, VKC_DEV) && v41c_res(&ch->mix, r * nm, VKC_DEV) &&
             v41c_res(&ch->hpa, r * hr, VKC_DEV) && v41c_res(&ch->hpf, r * hr, VKC_DEV) && v41c_res(&ch->col, r * D, VKC_DEV) &&
             v41c_res(&ch->nrm, r * D, VKC_DEV) && v41c_res(&ch->qa, r * c->q_lora, VKC_DEV) && v41c_res(&ch->qr, r * c->q_lora, VKC_DEV) &&
             v41c_res(&ch->q, r * nh * hd, VKC_DEV) && v41c_res(&ch->kv, r * hd, VKC_DEV) && v41c_res(&ch->heads, r * nh * hd, VKC_DEV) &&
             v41c_res(&ch->grp, r * c->o_groups * c->o_lora, VKC_DEV) && v41c_res(&ch->br, r * D, VKC_DEV) &&
             v41c_res(&ch->craw, r * hd, VKC_DEV) && v41c_res(&ch->cscr, r * hd, VKC_DEV) &&
             v41c_res(&ch->iq, r * IH * ID, VKC_DEV) && v41c_res(&ch->ihw, r * IH, VKC_DEV) &&
             v41c_res(&ch->isc, r * ch->wcap, VKC_DEV) && v41c_res(&ch->mask, r * ch->wcap, VKC_DEV) &&
             v41c_res(&ch->list, r * (ch->W + ch->K), VKC_DEV) &&
             v41c_res(&ch->cs, (r + ch->rmax) * 2 * c->rope_dim, VKC_DEV) && v41c_res(&ch->erows, r * ew, VKC_DEV) &&
             v41c_res(&ch->ekv, r * (size_t)D * (H + 1), VKC_DEV) && v41c_res(&ch->gs, r * c->moe_inter, VKC_DEV) &&
             v41c_res(&ch->us, r * c->moe_inter, VKC_DEV) && v41c_res(&ch->hs, r * c->moe_inter, VKC_DEV) &&
             v41c_res(&ch->ds, r * D, VKC_DEV) && v41c_res(&ch->h2d, r * D, VKC_DOWN) && v41c_res(&ch->mh, r * T * D, VKC_DOWN) &&
             v41c_res(&ch->xd, r * HD, VKC_DOWN) && v41c_res(&ch->hpd, r * hr, VKC_DOWN) && v41c_res(&ch->routed, r * D, VKC_UP);
    if (ok && ch->ks.on) {     /* the split: a chunk's new compressed rows, the two parts of an attention */
        size_t nc = r / (size_t)ch->rmin + 2;
        if (g_v41c_count >= 0)
            g_v41c_count += (long long)((2 * r * hd * sizeof(float) + ch->rmin - 1) / ch->rmin);
        else ok = v41c_res(&ch->cnew, nc * hd, VKC_DEV) && v41c_res(&ch->cdn, nc * hd, VKC_DOWN);
    }
    if (!ok || g_v41c_count >= 0) return ok;   /* counting: the buffers only */
    if (ch->rows < rows) {
        float *hrt = realloc(ch->host_routed, r * D * sizeof(float));
        if (!hrt) return 0;
        ch->host_routed = hrt; ch->rows = rows;
    }
    return 1;
}

/* ---- one layer's pieces ----------------------------------------------------------------- */
static int v41c_norm(const V41Chain *ch, VkcBuf *x, size_t xo, int xrow, size_t wo, VkcBuf *y, size_t yo, int yrow, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, (int)xo, xrow, D, (int)yo, yrow, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, ch->prm, y, &p);
}
/* mHC's entry for a site: the mix, the split into hp, the collapse with `prev`'s pre, the norm into nrm */
static int v41c_pre(V41Chain *ch, Model *m, ColiVkTensor *fn, size_t hco, VkcBuf *hp, VkcBuf *prev, size_t lno, int n) {
    Cfg *c = &m->c; int H = c->hc_mult, D = c->dim, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H;
    VkcMhc sp = {n, H, D, c->hc_iters, 0, HD, 0, nm, 0, hr, 0, D, (int)hco, 0, c->norm_eps, c->hc_eps, 0.f};
    VkcMhc co = sp;
    return vkc_matmul(fn, ch->xs, 0, ch->mix, 0, n) && vkc_mhc(VKC_MHC_SPLIT, ch->xs, ch->mix, hp, ch->prm, NULL, &sp) &&
           vkc_mhc(VKC_MHC_COLLAPSE, ch->xs, NULL, prev, NULL, ch->col, &co) &&
           v41c_norm(ch, ch->col, 0, D, lno, ch->nrm, 0, D, n, D, c->norm_eps);
}
/* mHC's exit: xn = comb(xs) + post * br, then the two swap */
static int v41c_post(V41Chain *ch, Model *m, VkcBuf *hp, int n) {
    Cfg *c = &m->c; int H = c->hc_mult, D = c->dim, HD = H * D, hr = 2 * H + H * H;
    VkcMhc po = {n, H, D, c->hc_iters, 0, HD, 0, D, 0, hr, 0, HD, 0, 0, c->norm_eps, c->hc_eps, 0.f};
    if (!vkc_mhc(VKC_MHC_POST, ch->xs, ch->br, hp, NULL, ch->xn, &po)) return 0;
    VkcBuf *t = ch->xs; ch->xs = ch->xn; ch->xn = t;
    return 1;
}
/* RoPE on interleaved pairs from this chunk's tables (ch->cs): nseg segments, per_row a row */
static int v41c_rope(V41Chain *ch, Model *m, VkcBuf *x, int nseg, int per_row, int x_off, int x_row, int x_seg,
                     int cs_off, int cs_row, int inverse) {
    VkcDsRope p = {nseg, per_row, m->c.rope_dim, x_off, x_row, x_seg, cs_off, cs_row, inverse, 0};
    return vkc_dsv4_rope(x, ch->cs, &p);
}

/* The forward being recorded: positions, its rows, and where its pieces go. */
typedef struct {
    int start, n, pb, nr, spec;
    int csB;              /* the compress table's offset in ch->cs: its row 0 is position pb - rmax + 1 */
    int *keyl;            /* per index source layer and row: whose index keys it scores (spec: per row) */
    size_t *praw;         /* per layer: where a speculative step's raw compressor rows go in ch->pull */
} V41Fwd;

/* attention_run for nr rows on the device: the projections, the window ring, the
 * compressor, the indexer, the sparse attention and the output projection, into ch->br */
static int v41c_attention(V41Chain *ch, Model *m, int i, const V41Fwd *f) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    int hd = c->head_dim, nh = c->n_heads, rd = c->rope_dim, QL = c->q_lora, r = c->compress_ratio[i];
    int nr = f->nr, pb = f->pb, qrow = nh * hd, LR = ch->W + ch->K;
    int tcs = r > 0 ? f->csB + (ch->rmax - 1) * rd : 0;            /* rope_for(layer): the row of position pb */
    int ok = vkc_matmul(v41c_w8(&l->wq_a), ch->nrm, 0, ch->qa, 0, nr) &&
             v41c_norm(ch, ch->qa, 0, QL, ch->o_qn[i], ch->qr, 0, QL, nr, QL, c->norm_eps) &&
             vkc_matmul(v41c_w8(&l->wq_b), ch->qr, 0, ch->q, 0, nr) &&
             vkc_matmul(v41c_w8(&l->wkv), ch->nrm, 0, ch->kv, 0, nr) &&
             v41c_norm(ch, ch->kv, 0, hd, ch->o_kn[i], ch->kv, 0, hd, nr, hd, c->norm_eps) &&
             v41c_rope(ch, m, ch->q, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 0) &&
             v41c_rope(ch, m, ch->kv, nr, 1, hd - rd, hd, 0, tcs, rd, 0);
    if (!ok) return 0;
    VkcRegion *rg = malloc((size_t)nr * sizeof *rg);   /* the new rows into the window ring */
    if (!rg) return 0;
    for (int s = 0; s < nr; s++) rg[s] = (VkcRegion){(size_t)((pb + s) % ch->Wd) * hd, (size_t)s * hd, (size_t)hd};
    ok = vkc_copy_regions(ch->win[i], ch->kv, rg, nr);
    free(rg);
    if (ok && r > 0 && c->kv_source[i]) {                           /* the compressor, its latents, the index keys */
        int g0 = pb / r, np = (pb + nr) / r - g0, ihd = c->index_head_dim, sp = ch->ks.on;
        VkcBuf *cb = sp ? ch->cnew : ch->ckv[i];                     /* the split: the new rows through a scratch */
        size_t co = sp ? 0 : (size_t)g0 * hd;
        ok = vkc_matmul(v41c_wb(&l->comp_wkv), ch->nrm, 0, ch->craw, 0, nr);
        if (ok && r == 1) ok = v41c_norm(ch, ch->craw, 0, hd, ch->o_cn[i], cb, co, hd, nr, hd, c->norm_eps);
        else if (ok) {
            VkcDsComp cp = {nr, pb, r, hd, hd, 0, 0, hd, 0, hd, -1, sp ? -g0 * hd : 0, hd, 0};
            ok = vkc_matmul(v41c_wb(&l->comp_wgate), ch->nrm, 0, ch->cscr, 0, nr) &&
                 vkc_dsv4_compress(ch->craw, ch->cscr, ch->ring[i], NULL, cb, &cp) &&
                 (np == 0 || v41c_norm(ch, cb, co, hd, ch->o_cn[i], cb, co, hd, np, hd, c->norm_eps));
            if (ok && f->spec && f->praw)                           /* a verify's raw rows, for its undo rows */
                ok = vkc_copy(ch->pull, f->praw[i], ch->craw, 0, (size_t)nr * hd) &&
                     vkc_copy(ch->pull, f->praw[i] + (size_t)nr * hd, ch->cscr, 0, (size_t)nr * hd);
        }
        if (ok && np > 0) {
            int bc = f->csB + (g0 * r - (pb - ch->rmax + 1)) * rd;   /* a latent rotates at its group's first position */
            ok = vkc_matmul(v41c_wb(&l->idx_wk), cb, co, ch->ikey[i], (size_t)g0 * ihd, np) &&
                 v41c_norm(ch, ch->ikey[i], (size_t)g0 * ihd, ihd, ch->o_ikn[i], ch->ikey[i], (size_t)g0 * ihd, ihd, np, ihd, c->norm_eps) &&
                 v41c_rope(ch, m, ch->ikey[i], np, 1, g0 * ihd + ihd - rd, ihd, 0, bc, r * rd, 0) &&
                 v41c_rope(ch, m, cb, np, 1, (int)co + hd - rd, hd, 0, bc, r * rd, 0);
            if (ok && sp) {   /* into their slots, and down for the host's ckv once the frame is through */
                VkcKvPart pt = {1, hd, l->ckv, 0, ch->ckv[i], 0};
                ok = vkc_kv_store(&ch->ks, ch->sli[i], &pt, cb, 0, (size_t)hd, 0, g0, np) &&
                     vkc_copy(ch->cdn, 0, cb, 0, (size_t)np * hd);
                vkc_kv_done(&ch->ks, ch->sli[i], g0 + np);
            }
        }
    }
    if (ok && r > 0 && c->index_source[i]) {                        /* the indexer: this layer's selection */
        int IH = c->index_n_heads, ID = c->index_head_dim, width = (pb + nr) / r, cs = c->candidate_source;
        int cand = v41c_cand_on(c), mrow = cand && cs < i ? (int)ch->wcap : 0;
        float wscale = (1.0f / sqrtf((float)ID)) * (1.0f / sqrtf((float)IH));
        ok = vkc_matmul(v41c_w8(&l->idx_wq_b), ch->qr, 0, ch->iq, 0, nr) &&
             v41c_rope(ch, m, ch->iq, nr * IH, IH, ID - rd, IH * ID, ID, tcs, rd, 0) &&
             vkc_matmul(v41c_wb(&l->idx_wproj), ch->nrm, 0, ch->ihw, 0, nr);
        const int *kl = f->keyl + (size_t)i * f->n + (pb - f->start);
        int same = 1;
        for (int s = 1; s < nr; s++) same &= kl[s] == kl[0];
        for (int s = 0; ok && s < nr; s += same ? nr : 1) {
            VkcDsScore sp = {same ? nr : 1, pb + s, r, IH, ID, width, s * IH * ID, IH * ID, s * IH, IH, 0, ID, mrow,
                             (int)ch->wcap, wscale, s * (int)ch->wcap, s * mrow};
            ok = vkc_dsv4_score(ch->iq, ch->ihw, ch->ikey[kl[s]], ch->mask, ch->isc, &sp);
        }
        if (ok && cand && cs == i) {
            VkcDsCand cp = {nr, pb, r, width, c->candidate_block_size, c->candidate_topk_blocks, (int)ch->wcap, (int)ch->wcap};
            ok = vkc_dsv4_cand(ch->isc, ch->mask, &cp);
        }
        VkcDsTopk tp = {nr, width, ch->K, (int)ch->wcap, ch->W, LR, ch->Wd, 0};
        ok = ok && vkc_dsv4_topk(ch->isc, ch->list, &tp);
    }
    if (!ok) return 0;
    VkcDsAttn a = {nr, nh, hd, ch->W + (r > 0 ? ch->K : 0), 0, LR, ch->Wd, 0, 0, 0, qrow, 0, qrow, (int)ch->o_sink[i], 0,
                   1.0f / sqrtf((float)hd)};
    int og = c->o_groups, ol = c->o_lora, pg = qrow / og;
    VkcHgemv g = {0, nr, og, ol, ol, 0, 0, qrow, pg, 0, og * ol, ol, 0, 0, 0};
    if (ch->ks.on && r > 0) {   /* stage selected cold rows, preserving the original sparse attention order */
        int own = c->index_owner[i], li = ch->sli[own];
        VkcKvPart pt = {1, hd, m->L[own].ckv, 0, ch->ckv[own], 0};
        VkcKvDs d = {&ch->ks, li, ch->q, ch->win[i], ch->ckv[own], ch->prm, ch->list, ch->heads, nr, nh, hd, a.cnt, 0, LR,
                     ch->Wd, 0, 0, qrow, (int)ch->o_sink[i], 0, 0, qrow, (pb + nr) / r, a.scale, m->L[own].ckv};
        /* the blocks pinned since the chunk began, and the table, before this read */
        ok = vkc_kv_push(&ch->ks, li, &pt, 1, ch->ks.t[li].valid) && vkc_kv_ds(&d);
    } else ok = vkc_dsv4_attn(ch->q, ch->win[i], r > 0 ? ch->ckv[c->index_owner[i]] : NULL, ch->list, ch->prm, ch->heads, &a);
    return ok &&
           v41c_rope(ch, m, ch->heads, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 1) &&
           vkc_mla_hgemv(v41c_w8(&l->wo_a), ch->heads, ch->grp, NULL, &g) &&
           vkc_matmul(v41c_w8(&l->wo_b), ch->grp, 0, ch->br, 0, nr);
}

/* the shared expert: silu(min(w1 x, lim)) * clamp(w3 x, +-lim), then w2 (swiglu_into) */
static int v41c_shared(V41Chain *ch, Model *m, Layer *l, int nr) {
    Cfg *c = &m->c; int I = c->moe_inter;
    VkcMhc sw = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nr * I, 0.f, 0.f, c->swiglu_limit > 0.0f ? c->swiglu_limit : INFINITY};
    return vkc_matmul(v41c_w8(&l->sh_w1), ch->nrm, 0, ch->gs, 0, nr) && vkc_matmul(v41c_w8(&l->sh_w3), ch->nrm, 0, ch->us, 0, nr) &&
           vkc_mhc(VKC_SWIGLU_CLAMP, ch->gs, ch->us, NULL, NULL, ch->hs, &sw) &&
           vkc_matmul(v41c_w8(&l->sh_w2), ch->hs, 0, ch->ds, 0, nr);
}
/* engram_run on the device: the host's n-gram rows, eng_wkv, the gate into each stream */
static int v41c_engram(V41Chain *ch, Model *m, int i, int nr, int pb, float *rows) {
    Cfg *c = &m->c; Engram *e = &m->engram; Layer *l = &m->L[i];
    int table = l->engram_index, width = e->cols * e->head_dim, H = c->hc_mult, D = c->dim;
    int64_t ids[V41_MAX_NGRAM * V41_MAX_EHEADS];
    double t0 = now_s();
    for (int r = 0; r < nr; r++) {
        engram_hash(e, table, pb + r, ids);
        for (int col = 0; col < e->cols; col++)
            memcpy(rows + (size_t)r * width + (size_t)col * e->head_dim, engram_row(&e->table[table], ids[col], e->head_dim),
                   (size_t)e->head_dim * sizeof(float));
    }
    m->t_engram += now_s() - t0;
    VkcDsEngram p = {nr, H, D, 0, D * (H + 1), 0, H * D, (int)ch->o_eq[i], (int)ch->o_ek[i], c->norm_eps};
    return vkc_write(ch->erows, 0, rows, (size_t)nr * width * sizeof(float)) &&
           vkc_matmul(v41c_w8(&l->eng_wkv), ch->erows, 0, ch->ekv, 0, nr) && vkc_dsv4_engram(ch->ekv, ch->prm, ch->xs, &p);
}

/* One chain's layers (lo..lo+n-1, on its device, current) for n rows of streams h
 * (positions start..), from h and pre_mix (the mix of the FFN site before its first
 * layer: [1, 0, ...] at layer 0): the streams after its last layer and that layer's FFN
 * mix back into h and pre_mix (every row when all, or when layers run after it, which
 * read them all; else the last). Returns ch->n; 0 = not taken: h and pre_mix untouched
 * (*lost = 1: a frame failed, the device marked lost; else declined, or memory refused:
 * the chain off). */
static void v41c_off(V41Chain *ch) { ch->failed = 1; if (!ch->d) g_vk_chain = 0; }
static int v41c_forward_seg(Model *m, V41Chain *ch, float *h, float *pre_mix, int n, int start, int spec, int all, int *lost) {
    *lost = 0;
    Cfg *c = &m->c;
    int L = c->n_layers, lo = ch->lo, hi = lo + ch->n, D = c->dim, H = c->hc_mult, HD = H * D, hr = 2 * H + H * H, hd = c->head_dim, rd = c->rope_dim;
    int W = ch->W, LR = W + ch->K, T = m->spec.active ? c->n_spec_targets : 0, E = start + n;
    const char *nmc = v41c_name(ch);
    if (hi < L) all = 1;
    if (spec && n > W) return 0;                           /* the undo rows: one window at most */
    if (!v41c_plan(ch, m, E)) {
        fprintf(stderr, "[VK] %s chain: device memory for the split's compressed rows refused; the CPU runs the layers\n", nmc);
        v41c_off(ch);
        return 0;
    }
    int CH = v41c_chunk_rows(ch, m, E), rows;
    { int64_t lim = ((int64_t)32 << 20) / (E > 1 ? E : 1); if (lim < CH) CH = lim < 1 ? 1 : (int)lim; }
    if (ch->ks.on) {   /* a chunk's new compressed rows fit the split's window */
        int mx = ch->rmin == 1 ? ch->ks.chunk : (ch->ks.chunk - 1) * ch->rmin;
        if (CH > mx) CH = mx < 1 ? 1 : mx;
    }
    if (spec && n > CH) return 0;                          /* a verify is one chunk */
    rows = n < CH ? n : CH;
    if (!v41c_grow_win(ch, m, rows)) {
        fprintf(stderr, "[VK] %s chain: device memory for window rings of %d rows refused; the CPU runs the layers\n", nmc, rows);
        v41c_off(ch);
        return 0;
    }
    /* A partial chain's handoff beyond the streams: the CPU's layers read the candidate
     * mask a chain layer made (the candidate source is a chain layer, an index source
     * after N reads it) and the index list the last chain index source published (a
     * compressed layer after N reads it before the next index source runs). */
    int cs = c->candidate_source, cand_out = 0, list_out = -1, list_w = 0;
    if (hi < L) {
        if (v41c_cand_on(c) && cs >= lo && cs < hi)
            for (int i = hi; i < L; i++) cand_out |= c->index_source[i] && c->compress_ratio[i] > 0;
        int last = -1, reader = 0;
        for (int i = lo; i < hi; i++) if (c->index_source[i] && c->compress_ratio[i] > 0) last = i;
        for (int i = hi; i < L && last >= 0; i++) {
            if (c->compress_ratio[i] <= 0) continue;
            if (c->index_source[i]) break;
            reader = 1;
        }
        if (reader) {
            int cl = E / c->compress_ratio[last];
            list_out = last; list_w = c->index_topk < cl ? c->index_topk : cl;
            if (list_w < 0) list_w = 0;
        }
    }
    V41Fwd f = {start, n, 0, 0, spec, rows * rd, NULL, NULL};
    /* whose index keys each index source reads, per row (indexer_run, published_owner) */
    f.keyl = malloc((size_t)L * n * sizeof(int));
    f.praw = calloc((size_t)L, sizeof(size_t));
    int *need = calloc((size_t)L, sizeof(int));
    float *outs = all ? malloc((size_t)n * HD * sizeof(float)) : malloc((size_t)HD * sizeof(float));
    float *pres = all ? malloc((size_t)n * H * sizeof(float)) : malloc((size_t)H * sizeof(float));
    int *wl = malloc((size_t)rows * LR * sizeof(int));
    float *cs_ = malloc((size_t)(rows + ch->rmax) * 2 * rd * sizeof(float));
    float *er = m->engram.active ? malloc((size_t)rows * m->engram.cols * m->engram.head_dim * sizeof(float)) : NULL;
    if (!f.keyl || !f.praw || !need || !outs || !pres || !wl || !cs_ || (m->engram.active && !er)) {
        free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs_); free(er);
        return 0;
    }
    static int tidy = -1;
    if (tidy < 0) tidy = getenv("V41_INDEX_OWNER") != NULL;
    /* the rows of each kv_source layer the forward reads that it does not produce first:
     * the complete groups before it, and whatever another layer's selection reaches (the
     * host's rows there, as the CPU reads them; a partial chain: also the index keys of a
     * CPU layer a chain layer reads, the host's as the CPU's layer order has them then) */
    for (int i = lo; i < hi; i++) if (c->kv_source[i] && c->compress_ratio[i] > 0) need[i] = start / c->compress_ratio[i];
    int pub = m->published_index_k ? m->published_index_layer : -1, last_pub = -1;
    for (int i = lo; i < hi; i++) {
        int r = c->compress_ratio[i];
        if (r <= 0) continue;
        int owner = c->index_owner[i];
        if (c->kv_source[i] && E / r - start / r > 0) { pub = i; last_pub = i; }
        if (E / r > need[owner]) need[owner] = E / r;
        for (int t = 0; t < n; t++) {
            int k = owner;
            if (c->index_source[i] && !tidy) {
                if (m->pub_rows > 0) {
                    int p = published_owner(m, i, start, t);
                    k = p >= 0 ? p : m->pub_before ? m->pub_before_layer : owner;
                } else if (n == 1 && pub >= 0) k = pub;
            }
            f.keyl[(size_t)i * n + t] = k;
            if (c->index_source[i] && (start + t + 1) / r > need[k]) need[k] = (start + t + 1) / r;
        }
    }
    int ok = 1;
    for (int i = 0; i < L && ok; i++) {
        if (!c->kv_source[i]) continue;
        int r = c->compress_ratio[i], host = c->max_positions / r;
        if (need[i] > host) need[i] = host;
        if (!v41c_own(ch, i)) { if (need[i] > 0) ok = v41c_cache(ch, m, i, need[i]); continue; }   /* another layer's keys a chain layer reads */
        int want = need[i] > E / r ? need[i] : E / r;
        ok = v41c_cache(ch, m, i, want > 0 ? want : 1);
    }
    /* what comes back at the end: the window rows, the compressed rows and keys, the groups,
     * a verify's raw compressor rows */
    size_t pw = (size_t)(n < W ? n : W) * hd, pn = 0, *pc = calloc((size_t)L, sizeof(size_t));
    for (int i = lo; ok && pc && i < hi; i++) {
        pc[i] = pn; pn += pw;
        int r = c->compress_ratio[i];
        if (!c->kv_source[i]) continue;
        pn += (size_t)(E / r - start / r) * ((ch->ks.on ? 0 : hd) + c->index_head_dim);   /* split: the rows came back per chunk */
        if (r > 1) { pn += (size_t)2 * r * hd; if (spec) { f.praw[i] = pn; pn += (size_t)2 * n * hd; } }
    }
    ok = ok && pc && v41c_scratch(ch, m, rows, E) && v41c_res(&ch->pull, pn, VKC_DOWN) && (!T || m->main_hidden) &&
         (!cand_out || v41c_res(&ch->cmd, (size_t)rows * ch->wcap, VKC_DOWN)) &&
         (list_out < 0 || v41c_res(&ch->ld, (size_t)rows * LR, VKC_DOWN));
    if (!ok) {
        fprintf(stderr, "[VK] %s chain: device memory for %d rows at %d positions refused; the CPU runs the layers\n", nmc, rows, E);
        free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs_); free(er); free(pc);
        v41c_off(ch);
        return 0;
    }
    if (cand_out) {   /* the CPU's mask, as indexer_run sizes it */
        int width = E / c->compress_ratio[cs];
        if (m->candidate_width != width || m->candidate_rows != n) {
            free(m->candidates);
            m->candidates = xmalloc((size_t)n * (width > 0 ? width : 1), "candidate mask");
            m->candidate_width = width; m->candidate_rows = n;
        }
    }
    if (list_out >= 0 && (m->shared_topk_rows != n || m->shared_topk_width != list_w)) {   /* the CPU's list, as attention_run sizes it */
        free(m->shared_topk);
        m->shared_topk = xmalloc((size_t)n * (list_w > 0 ? list_w : 1) * sizeof(int), "published index list");
        m->shared_topk_rows = n; m->shared_topk_width = list_w;
    }
    vkc_gemm_rows(spec ? 0 : -1);                          /* a verify's rows get a decode step's bits */
    for (int c0 = 0; c0 < n; c0 += rows) {
        int nr = n - c0 < rows ? n - c0 : rows, pb = start + c0, last = c0 + nr == n;
        f.pb = pb; f.nr = nr;
        /* this chunk's RoPE rows (the CPU's own tables): window ones at 0, compress ones at
         * csB from position pb - rmax + 1 */
        for (int s = 0; s < nr; s++) memcpy(cs_ + (size_t)s * rd, m->rope_window + (size_t)(pb + s) * rd, rd * sizeof(float));
        for (int s = 0; s < nr + ch->rmax - 1; s++) {
            int p = pb - ch->rmax + 1 + s;
            float *dst = cs_ + f.csB + (size_t)s * rd;
            if (p < 0) memset(dst, 0, rd * sizeof(float));
            else memcpy(dst, m->rope_compress + (size_t)p * rd, rd * sizeof(float));
        }
        /* the window half of each row's list (window_idxs): the positions of the last
         * `window` the row reaches, oldest first, as device ring rows; -1 where the CPU's
         * ring does not hold the position (window_pos: the chain's first layer's, which
         * holds what every layer held when the forward began; the layers before it have
         * written theirs by now). The selection half starts empty. */
        const Layer *l0 = &m->L[lo];
        for (int s = 0; s < nr; s++) {
            int p = pb + s;
            for (int k = 0; k < W; k++) {
                int q = p - W + 1 + k, e = -1;
                if (q >= 0 && (q >= start || l0->window_pos[q % W] == q)) e = q % ch->Wd;
                wl[(size_t)s * LR + k] = e;
            }
            for (int k = W; k < LR; k++) wl[(size_t)s * LR + k] = -1;
        }
        if (!vkc_begin() || !vkc_write(ch->xs, 0, h + (size_t)c0 * HD, (size_t)nr * HD * sizeof(float)) ||
            !vkc_write(ch->list, 0, wl, (size_t)nr * LR * sizeof(int)) ||
            !vkc_write(ch->cs, 0, cs_, (size_t)(f.csB + (nr + ch->rmax - 1) * rd) * sizeof(float))) goto lost;
        {   /* the first site collapses with [1, 0, ...] (forward_full's pre_mix), a later
             * chain's with the mix the layers before it handed over */
            float *hp0 = calloc((size_t)nr * hr, sizeof(float));
            if (!hp0) goto lost;
            for (int s = 0; s < nr; s++) {
                if (lo == 0) hp0[(size_t)s * hr] = 1.0f;
                else memcpy(hp0 + (size_t)s * hr, pre_mix + (size_t)(c0 + s) * H, (size_t)H * sizeof(float));
            }
            ok = vkc_write(ch->hpf, 0, hp0, (size_t)nr * hr * sizeof(float));
            free(hp0);
            if (!ok) goto lost;
        }
        if (c0 == 0) {   /* the device's copies made the host's, as the forward needs them */
            for (int q = ch->win_valid > start - W + 1 ? ch->win_valid : start - W + 1; q < start && ok; q++) {
                if (q < 0) continue;
                for (int i = lo; i < hi && ok; i++) {
                    const Layer *l = &m->L[i];
                    if (l->window_pos[q % W] != q) continue;
                    ok = vkc_write(ch->win[i], (size_t)(q % ch->Wd) * hd, l->window + (size_t)(q % W) * hd, (size_t)hd * sizeof(float));
                }
            }
            for (int i = 0; i < L && ok; i++) {
                int own = v41c_own(ch, i);
                if (!c->kv_source[i] || (!own && !ch->ikey[i])) continue;
                const Layer *l = &m->L[i];
                int r = c->compress_ratio[i], ihd = c->index_head_dim, t0 = ch->kv_valid[i], t1 = need[i];
                if (t1 > t0) ok = (!own || ch->ks.on || vkc_write(ch->ckv[i], (size_t)t0 * hd, l->ckv + (size_t)t0 * hd, (size_t)(t1 - t0) * hd * sizeof(float))) &&
                                  vkc_write(ch->ikey[i], (size_t)t0 * ihd, l->ikey + (size_t)t0 * ihd, (size_t)(t1 - t0) * ihd * sizeof(float));
                if (!own) { if (ok && t1 > t0) ch->kv_valid[i] = t1; continue; }
                if (ok && r > 1) ok = vkc_write(ch->ring[i], 0, l->cstate_kv, (size_t)r * hd * sizeof(float)) &&
                                      vkc_write(ch->ring[i], (size_t)r * hd, l->cstate_score, (size_t)r * hd * sizeof(float));
            }
            if (!ok) goto lost;
        }
        for (int i = 0; ch->ks.on && i < L && ok; i++) {   /* the split: each table's window over this chunk's new rows */
            if (ch->sli[i] < 0) continue;
            int r = c->compress_ratio[i], g0 = pb / r;
            VkcKvPart pt = {1, hd, m->L[i].ckv, 0, ch->ckv[i], 0};
            vkc_kv_place(&ch->ks, ch->sli[i], g0, (pb + nr) / r - g0);
            ok = vkc_kv_push(&ch->ks, ch->sli[i], &pt, 1, g0);
        }
        if (!ok) goto lost;
        int pending = 0;
        for (int i = lo; i < hi && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) {                                 /* the FFN branch of the layer before */
                VkcEw add = {VKC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = vkc_ew(ch->br, ch->routed, ch->ds, NULL, NULL, &add) && v41c_post(ch, m, ch->hpf, nr);
                pending = 0;
            }
            if (ok && l->engram_index >= 0) ok = v41c_engram(ch, m, i, nr, pb, er);
            for (int k = 0; ok && k < T; k++) {
                if (c->spec_targets[k] != i) continue;
                VkcMhc mean = {nr, H, D, 0, 0, HD, 0, 0, 0, 0, k * D, T * D, 0, 0, 0.f, 0.f, 0.f};
                ok = vkc_mhc(VKC_MHC_MEAN, ch->xs, NULL, NULL, NULL, ch->mh, &mean);
            }
            double ta = now_s();
            ok = ok && v41c_pre(ch, m, ch->fna[i], ch->o_hca[i], ch->hpa, ch->hpf, ch->o_an[i], nr) &&
                 v41c_attention(ch, m, i, &f) && v41c_post(ch, m, ch->hpa, nr) &&
                 v41c_pre(ch, m, ch->fnf[i], ch->o_hcf[i], ch->hpf, ch->hpa, ch->o_fn[i], nr) &&
                 vkc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D) &&
                 (!cand_out || i != cs || vkc_copy(ch->cmd, 0, ch->mask, 0, (size_t)nr * ch->wcap)) && vkc_submit(0);   /* A1 */
            /* while it runs, the tier loads the experts this layer will likely stream (a
             * big prompt chunk only) */
            if (ok) vkt_stream_prefetch(i, nr);
            ok = ok && vkc_finish();
            m->t_attn += now_s() - ta;
            if (!ok) break;
            if (ch->ks.on && ch->sli[i] >= 0) {            /* this chunk's new compressed rows into the host's ckv */
                int r = c->compress_ratio[i], g0 = pb / r, np = (pb + nr) / r - g0;
                if (np > 0) memcpy(l->ckv + (size_t)g0 * hd, vkc_ptr(ch->cdn), (size_t)np * hd * sizeof(float));
            }
            if (cand_out && i == cs) {                     /* the candidate source's mask, for the CPU's readers */
                const int32_t *mk = (const int32_t *)vkc_ptr(ch->cmd);
                int wc = (pb + nr) / c->compress_ratio[cs], wh = m->candidate_width;
                for (int s = 0; s < nr; s++) {
                    uint8_t *keep = m->candidates + (size_t)(c0 + s) * wh;
                    for (int j = 0; j < wh; j++) keep[j] = j < wc && mk[(size_t)s * ch->wcap + j] != 0;
                }
            }
            /* A2: the shared expert, while the host computes the routed experts */
            ok = vkc_begin() && v41c_shared(ch, m, l, nr) && vkc_submit(0);
            double t1 = now_s();
            moe_run_at(m, l, &m->cache[i], "layers", i, c->n_routed, c->n_activated, (const float *)vkc_ptr(ch->h2d), nr,
                       ch->host_routed, 0);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)nr * D * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok && pending) {
            VkcEw add = {VKC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = vkc_ew(ch->br, ch->routed, ch->ds, NULL, NULL, &add) && v41c_post(ch, m, ch->hpf, nr);
        }
        ok = ok && vkc_copy(ch->xd, 0, ch->xs, 0, (size_t)nr * HD) && vkc_copy(ch->hpd, 0, ch->hpf, 0, (size_t)nr * hr) &&
             (list_out < 0 || vkc_copy(ch->ld, 0, ch->list, 0, (size_t)nr * LR));
        if (ok && last) {                                  /* what the host's state gets back */
            for (int i = lo; i < hi && ok; i++) {
                int q0 = E - W > start ? E - W : start, nq = E - q0;
                VkcRegion *rg = malloc((size_t)nq * sizeof *rg);
                if (!rg) { ok = 0; break; }
                for (int q = q0; q < E; q++) rg[q - q0] = (VkcRegion){pc[i] + (size_t)(q - q0) * hd, (size_t)(q % ch->Wd) * hd, (size_t)hd};
                ok = vkc_copy_regions(ch->pull, ch->win[i], rg, nq);
                free(rg);
                if (!ok || !c->kv_source[i]) continue;
                int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim, cw = ch->ks.on ? 0 : hd;
                size_t at = pc[i] + pw;
                ok = (cw == 0 || vkc_copy(ch->pull, at, ch->ckv[i], (size_t)g0 * hd, (size_t)np * hd)) &&
                     vkc_copy(ch->pull, at + (size_t)np * cw, ch->ikey[i], (size_t)g0 * ihd, (size_t)np * ihd) &&
                     (r <= 1 || vkc_copy(ch->pull, at + (size_t)np * (cw + ihd), ch->ring[i], 0, (size_t)2 * r * hd));
            }
        }
        ok = ok && vkc_submit(1);
        if (!ok) goto lost;
        const float *xd = (const float *)vkc_ptr(ch->xd), *hpd = (const float *)vkc_ptr(ch->hpd);
        for (int s = 0; s < nr; s++) {
            int t = c0 + s;
            if (!all && t != n - 1) continue;
            memcpy(outs + (size_t)(all ? t : 0) * HD, xd + (size_t)s * HD, (size_t)HD * sizeof(float));
            memcpy(pres + (size_t)(all ? t : 0) * H, hpd + (size_t)s * hr, (size_t)H * sizeof(float));
        }
        if (list_out >= 0) {   /* the index list the CPU's layers after N read (attention_run's values) */
            const int32_t *ld = (const int32_t *)vkc_ptr(ch->ld);
            int wrows = start == 0 ? n : W;
            for (int s = 0; s < nr; s++)
                for (int k = 0; k < list_w; k++) {
                    int v = ld[(size_t)s * LR + W + k];
                    m->shared_topk[(size_t)(c0 + s) * list_w + k] = v >= 0 ? v - ch->Wd + wrows : -1;
                }
        }
        for (int k = 0; k < T; k++) {   /* the chain layers' DSpark inputs (the other layers' fill the others) */
            if (!v41c_own(ch, c->spec_targets[k])) continue;
            const float *mh = (const float *)vkc_ptr(ch->mh);
            for (int s = 0; s < nr; s++)
                memcpy(m->main_hidden + ((size_t)(c0 + s) * T + k) * D, mh + ((size_t)s * T + k) * D, (size_t)D * sizeof(float));
        }
    }
    /* every chunk is through: the host's state as the CPU would have left it */
    {
        const float *pl = (const float *)vkc_ptr(ch->pull);
        int save = spec && m->rollback_save && n > 1;
        for (int i = lo; i < hi; i++) {
            Layer *l = &m->L[i];
            int q0 = E - W > start ? E - W : start;
            for (int t = 0; t < n; t++) {
                int p = start + t, slot = p % W;
                if (save) {
                    memcpy(l->ring_save + (size_t)t * hd, l->window + (size_t)slot * hd, (size_t)hd * sizeof(float));
                    l->ring_save_pos[t] = l->window_pos[slot];
                }
                if (p < q0) continue;                      /* a later row takes its slot */
                memcpy(l->window + (size_t)slot * hd, pl + pc[i] + (size_t)(p - q0) * hd, (size_t)hd * sizeof(float));
                l->window_pos[slot] = p;
            }
            if (!c->kv_source[i]) continue;
            int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim, cw = ch->ks.on ? 0 : hd;
            const float *src = pl + pc[i] + pw;
            if (cw) memcpy(l->ckv + (size_t)g0 * hd, src, (size_t)np * hd * sizeof(float));   /* split: back per chunk */
            memcpy(l->ikey + (size_t)g0 * ihd, src + (size_t)np * cw, (size_t)np * ihd * sizeof(float));
            if (r <= 1) continue;
            if (save) {   /* the groups' undo rows, as compressor_run's decode path writes them */
                const float *raw = pl + f.praw[i];
                for (int t = 0; t < n; t++) {
                    int slot = (start + t) % r;
                    memcpy(l->cstate_save_kv + (size_t)t * hd, l->cstate_kv + (size_t)slot * hd, (size_t)hd * sizeof(float));
                    memcpy(l->cstate_save_score + (size_t)t * hd, l->cstate_score + (size_t)slot * hd, (size_t)hd * sizeof(float));
                    l->cstate_save_slot[t] = slot;
                    memcpy(l->cstate_kv + (size_t)slot * hd, raw + (size_t)t * hd, (size_t)hd * sizeof(float));
                    memcpy(l->cstate_score + (size_t)slot * hd, raw + (size_t)(n + t) * hd, (size_t)hd * sizeof(float));
                }
            }
            const float *ring = src + (size_t)np * (cw + ihd);
            memcpy(l->cstate_kv, ring, (size_t)r * hd * sizeof(float));
            memcpy(l->cstate_score, ring + (size_t)r * hd, (size_t)r * hd * sizeof(float));
        }
        if (last_pub >= 0) { m->published_index_k = m->L[last_pub].ikey; m->published_index_layer = last_pub; }
        ch->win_valid = E;
        for (int i = 0; i < L; i++) {
            if (!c->kv_source[i]) continue;
            int r = c->compress_ratio[i];
            if (v41c_own(ch, i)) ch->kv_valid[i] = E / r;
            else if (ch->kv_valid[i] > start / r) ch->kv_valid[i] = start / r;   /* another layer writes its keys from here */
        }
    }
    if (all) { memcpy(h, outs, (size_t)n * HD * sizeof(float)); memcpy(pre_mix, pres, (size_t)n * H * sizeof(float)); }
    else { memcpy(h + (size_t)(n - 1) * HD, outs, (size_t)HD * sizeof(float)); memcpy(pre_mix + (size_t)(n - 1) * H, pres, (size_t)H * sizeof(float)); }
    free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs_); free(er); free(pc);
    ch->forwards++;
    return ch->n;
lost:   /* a frame failed: the device is gone (or would not take a command) */
    free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs_); free(er); free(pc);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(ch->d); }
    *lost = 1;
    return 0;
}
/* The devices' layers (every layer, or the primary's first n of a partial chain and the
 * second device's after them) for n rows of streams h (positions start..), as
 * v41c_forward_seg describes, from layer 0. Returns the layers that ran: the caller's CPU
 * loop runs the others from there. 0 = not taken: the CPU runs every layer, h and pre_mix
 * untouched. */
static int v41c_forward(Model *m, float *h, float *pre_mix, int n, int start, int spec, int all) {
    V41Chain *ch = g_v41c, *ch2 = g_v41c2;
    if (!g_vk_chain || !ch || !ch->ok || ch->failed || g_trace) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && n <= 2) return 0;
    if (ch2 && (!ch2->ok || ch2->failed || ch2->lo != ch->lo + ch->n)) ch2 = NULL;
    int lost2 = 0;
    if (ch2) { vkc_device(1); lost2 = vkc_lost(); vkc_device(0); }
    if (vkc_lost() || lost2) {
        for (int d = 0; d < 2; d++) if (v41c_of(d)) v41c_of(d)->failed = 1;
        g_vk_chain = 0;
        return 0;
    }
    int lost = 0;
    int r = v41c_forward_seg(m, ch, h, pre_mix, n, start, spec, all, &lost);
    if (!r) {
        if (lost) {
            for (int d = 0; d < 2; d++) if (v41c_of(d)) v41c_of(d)->failed = 1;
            g_vk_chain = 0;
            fprintf(stderr, "[VK] deepseek_v41 chain: the device was lost; the CPU runs this forward again and from here on "
                            "(the host's state is current: there is nothing to rebuild)\n");
        }
        return 0;
    }
    if (!ch2) return r;
    vkc_device(1);
    int r2 = v41c_forward_seg(m, ch2, h, pre_mix, n, start, spec, all, &lost);
    vkc_device(0);
    if (r2) return r + r2;
    /* not taken there: the CPU runs its layers from the streams the primary handed over
     * (the second device's watermarks follow) */
    v41c_lower_one(m, ch2, start);
    if (lost) {
        for (int d = 0; d < 2; d++) if (v41c_of(d)) v41c_of(d)->failed = 1;
        g_vk_chain = 0;
        fprintf(stderr, "[VK] deepseek_v41 dev2 chain: the device was lost; the CPU runs its layers, and every layer from the next "
                        "forward on (the host's state is current: there is nothing to rebuild)\n");
    }
    return r;
}

static void v41c_report(void) {
    for (int d = 0; d < 2; d++) {
        V41Chain *ch = v41c_of(d);
        if (!ch || !ch->ok || !ch->forwards) continue;
        int was = vkc_device(d);
        VkcStats st; vkc_stats(&st);
        fprintf(stderr, "[VK] %s chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                        "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
                v41c_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms,
                st.dev_bytes / 1048576.0);
        vkc_kv_report(&ch->ks);
        vkc_prof_print();
        vkc_device(was);
    }
}

/* COLI_VK_CHAIN at startup, before the tier sizes itself (the trunk's device copies count
 * as used): the decision, the pipelines, the tensors. */
static int g_v41c_decision = -1;
static int v41c_decide(Model *m) {   /* once: early when the trunk may live on the device only */
    if (g_v41c_decision >= 0) return g_v41c_decision;
    if (!g_vk_ready) return 0;
    int tier_on = vkt_wanted() && m->c.n_routed > 0;
    g_v41c_decision = coli_vk_chain_decide("deepseek_v41", tier_on, COLI_VK_CHAIN_UNMEASURED);
    if (g_v41c_decision && g_v41_mux_slots > 1) {
        g_v41c_decision = 0;
        fprintf(stderr, "[VK] deepseek_v41: KV_SLOTS=%d: the dense chain is off (it keeps one conversation's state on the device); "
                        "the expert tier runs every conversation's experts\n", g_v41_mux_slots);
    }
    return g_v41c_decision;
}
static void v41c_start(Model *m) {
    if (!g_vk_ready) return;
    int on = v41c_decide(m);
    if (on && v41c_fit_now(m) && g_v41_fit.n < 1) {   /* the fit's line said so: nothing of the chain on the device */
        vkc_fit_placed("deepseek_v41", &g_v41_fit);
        return;
    }
    const char *no = NULL;
    if (on && !(g_v41c_inited = vkc_init())) no = "the chain's pipelines did not come up";
    if (on && !no && !(vkc_mla_ready() && vkc_mhc_ready() && vkc_dsv4_ready()))
        no = "the MLA, mHC or DeepSeek shaders are missing (chain_hgemv, chain_mhc, chain_dsv4)";
    if (no) fprintf(stderr, "[VK] deepseek_v41: %s: the dense chain stays off\n", no);
    if (!on || no) return;
    int n0 = g_v41_fit.n;
    if (!v41c_setup(m)) return;
    g_vk_chain = on;
    if (g_v41c_fit2_on) {
        vkc_device(1);
        if (g_v41c->n < n0) {
            /* the primary placed fewer layers than its fit: the second device's would not
             * follow them, so they stay on the CPU too */
            fprintf(stderr, "[VK] deepseek_v41 chain: the primary device stopped before layer %d; layers %d..%d stay on the CPU, "
                            "not on the second device\n", n0, n0, n0 + g_v41c_fit2.n - 1);
            g_v41c_fit2_on = 0;
            vkc_shutdown();
        } else if (!v41c_setup_dev(m, 1)) { g_v41c_fit2_on = 0; vkc_shutdown(); }
        vkc_device(0);
    }
}
/* After the tier's: at exit the report, then the chain goes, then the device. */
static void v41c_atexit(void) {
    if (!g_v41c_inited) return;
    atexit(vkc_shutdown_all);
    atexit(v41c_report);
}
