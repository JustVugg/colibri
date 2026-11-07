/* deepseek_v41_cuchain.h -- DeepSeek V4.1's layers on the CUDA chain for a prompt, included by
 * deepseek_v41.c in a COLI_CUDA build (after cu_dense.h, beside deepseek_v41_chain.h, the
 * Vulkan chain this follows op for op).
 *
 * On the CUDA tier a prompt ran its layers on the CPU (forward_full): the sparse attention, the
 * hyper-connections and the router were three fifths of a 4104-token prompt of V4.1-Flash
 * UD-IQ1_S on an RTX PRO 4000 (526 s). Here a GGUF's prompt of more than one chunk runs on the
 * card layer by layer, as the Vulkan chain's v41c_prompt_lw: each layer's chunks in order
 * through the mHC split and collapse, the attention (the window ring, the compressor and its
 * latents, the indexer's selection, the sparse attention, wo_a's groups, wo_b), the write
 * back, the FFN's split and the shared expert, every row's streams, attention list (the
 * selection its last index source made) and candidate mask kept on the device; then the
 * layer's routed experts for all its rows in one step of the CUDA tier (moe_run_at, its
 * streaming), then each chunk's FFN sum. A longer prompt than COLI_CUDA_V41_SEG rows goes in
 * segments, each through every layer before the next. The dense matrices are cu_dense.h's
 * device copies (cud_tensor), wo_a as its output groups' matrices; the ops are cuda_chain.cu's
 * and cuda_dsv4c.cuh's.
 *
 * The host state (each layer's window ring and its position map, the compressed rows and index
 * keys, the compressor's group) goes up as far as the device does not hold it and comes back
 * as the CPU would have left it, so the decode, which stays on the CPU and the tier, and the
 * next prompt read the same state. The CPU's steps lower what the device holds (v41u_lower).
 *
 * Not here: a decode step, a speculative verify, DSpark's targets, a model past the device's
 * memory (a matrix the dense budget does not take, scratch refused): forward_full runs the
 * layers; KV_SLOTS above 1. COLI_CUDA_CHAIN=0: never. COLI_CUDA_V41_ROWS: a chunk's rows (256). */
#ifndef DEEPSEEK_V41_CUCHAIN_H
#define DEEPSEEK_V41_CUCHAIN_H
#include "cuda_chain.h"

typedef struct {
    int start, n, pb, nr;
    int csB;        /* the compress table's offset in cs: its row 0 is position pb - rmax + 1 */
    int *keyl;      /* per index source layer and row: whose index keys it scores */
    CcBuf *list;    /* every row's list, the chunk's at lo: the window half, then the selection the last
                     * index source wrote for these rows, which the compressed layers after it read */
    int lo;
    int s0;         /* the segment's first row: lw_xs, lw_hp, lw_ds, lw_mask and lw_in hold its rows from it */
} V41UFwd;

typedef struct {
    int ok, W, Wd, K, rmax, rows;
    size_t wcap;
    CcBuf *prm;
    size_t *o_an, *o_fn, *o_qn, *o_kn, *o_sink, *o_hca, *o_hcf, *o_cn, *o_ikn, *o_eq, *o_ek;
    ColiCudaTensor **fna, **fnf;
    CcBuf **win, **ckv, **ikey, **ring;
    int *kv_valid, *ccap, win_valid;
    CcBuf *xs, *xn, *mix, *hpa, *hpf, *col, *nrm, *qa, *qr, *q, *kv, *heads, *grp, *br, *craw, *cscr;
    CcBuf *iq, *ihw, *isc, *cs, *erows, *ekv, *gs, *us, *hs, *ds, *h2d, *xd, *hpd, *pull, *routed;
    CcBuf *lw_xs, *lw_hp, *lw_ds, *lw_list, *lw_mask;
    float *lw_in, *lw_out;
    size_t lw_cap;
    unsigned long long forwards;
    double host_ms;
} V41U;
static V41U *g_v41u;
static int g_v41u_off;   /* declined for good (set up refused, or COLI_CUDA_CHAIN=0) */

static int v41u_res(CcBuf **b, size_t floats, int kind) { return cc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }
static ColiCudaTensor *v41u_t8(const W8 *w) { return w->gx && w->q ? cud_tensor(w->q, 100 + w->gt, w->O, w->I) : NULL; }
/* a WB: a GGUF's blocks, or the bf16 rows the loader keeps of a GGUF's BF16 tensor (unsloth's
 * compressor and indexer matrices), those as floats */
static ColiCudaTensor *v41u_tb(const WB *w) {
    if (w->gx) return w->g ? cud_tensor(w->g, 100 + w->gt, w->O, w->I) : NULL;
    return w->w ? cud_tensor(w->w, 100 + GQ_BF16, w->O, w->I) : NULL;
}
static ColiCudaTensor *v41u_woa(const W8 *w, int g, int ol) {
    const int64_t rb = gq_row_bytes(w->gt, w->I);   /* an interleaved type's: 16 of them step a group */
    return w->gx && w->q && rb > 0 ? cud_tensor(w->q + (size_t)g * ol * (size_t)rb, 100 + w->gt, ol, w->I) : NULL;
}

/* the window ring and position map, the compressed rows and keys the device holds from pos on
 * are stale: a CPU step from pos, a rollback, a new sequence */
static void v41u_lower(Model *m, int pos) {
    V41U *ch = g_v41u;
    if (!ch || !ch->ok) return;
    if (pos < 0) pos = 0;
    if (ch->win_valid > pos) ch->win_valid = pos;
    for (int i = 0; i < m->c.n_layers; i++) {
        int r = m->c.compress_ratio[i];
        if (m->c.kv_source[i] && r > 0 && ch->kv_valid[i] > pos / r) ch->kv_valid[i] = pos / r;
    }
}

static int v41u_cand_on(const Cfg *c) {   /* deepseek_v41_chain.h's v41c_cand_on */
    int cs = c->candidate_source;
    return cs >= 0 && cs < c->n_layers && c->index_source[cs] && c->compress_ratio[cs] > 0;
}
/* A prompt's rows layer by layer, at most COLI_CUDA_V41_SEG of them (4096) at once: a longer one
 * goes in segments of whole chunks, each through every layer before the next, so what a prompt's
 * rows hold on the device (their streams, mixes, shared expert and candidate mask) stays what
 * the tier left the chain (v41u_plan_bytes). */
static int v41u_seg_env(void) {
    static int seg = -1;
    if (seg < 0) { const char *e = getenv("COLI_CUDA_V41_SEG"); seg = e && atoi(e) > 0 ? atoi(e) : 4096; }
    return seg;
}
static int v41u_rows_env(void) {
    static int rows = -1;
    if (rows < 0) { const char *e = getenv("COLI_CUDA_V41_ROWS"); rows = e && atoi(e) > 0 ? atoi(e) : 256; }
    return rows;
}
static int v41u_seg_rows(const V41U *ch) {
    const int s = v41u_seg_env();
    return s < 2 * ch->rows ? 2 * ch->rows : s / ch->rows * ch->rows;
}
/* The device memory the chain takes at most for a prompt in a context of ctx positions: a
 * segment's rows, a chunk's scratch, the attention's, every layer's state and the mixers'
 * matrices; the routed experts' tier leaves it (VktConfig.chain_bytes). */
static size_t v41u_plan_bytes(const Model *m, int ctx) {
    const Cfg *c = &m->c;
    const size_t L = (size_t)c->n_layers, D = (size_t)c->dim, H = (size_t)c->hc_mult, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H;
    const size_t R = (size_t)v41u_rows_env(), hd = (size_t)c->head_dim, nh = (size_t)c->n_heads, E = ctx > 0 ? (size_t)ctx : 1;
    const size_t W = (size_t)c->window, K = c->index_topk > 0 ? (size_t)c->index_topk : 0, LR = W + K;
    const size_t IH = c->index_n_heads > 0 ? (size_t)c->index_n_heads : 1, ID = c->index_head_dim > 0 ? (size_t)c->index_head_dim : 1;
    const size_t ew = m->engram.active ? (size_t)m->engram.cols * m->engram.head_dim : 1;
    size_t P = (size_t)v41u_seg_env();
    if (P < 2 * R) P = 2 * R;
    if (P > E) P = E;
    int rmax = 1;
    for (int i = 0; i < c->n_layers; i++) if (c->kv_source[i] && c->compress_ratio[i] > rmax) rmax = c->compress_ratio[i];
    size_t f = P * (HD + hr + D) + E * LR + (v41u_cand_on(c) ? (P * E + 3) / 4 : 0);   /* the segment's rows, every row's list */
    f += R * (2 * HD + nm + 2 * hr + 3 * D + 2 * (size_t)c->q_lora + 2 * nh * hd + 3 * hd + (size_t)c->o_groups * c->o_lora +
              IH * ID + IH + E + ew + D * (H + 1) + 3 * (size_t)c->moe_inter) + (R + (size_t)rmax) * 2 * (size_t)c->rope_dim;
    f += ((size_t)64 << 20) / 4 + R * nh;                        /* the attention's scores (cc_ds_attn's 64 MiB) */
    f += L * (W + R) * hd + L * 2 * nm * HD;                     /* the window rings, the mixers' matrices */
    for (int i = 0; i < c->n_layers; i++) {
        const int r = c->compress_ratio[i];
        if (!c->kv_source[i] || r < 1) continue;
        size_t cap = 64;
        while (cap < E / (size_t)r) cap *= 2;
        f += cap * (hd + ID) + (r > 1 ? 2 * (size_t)r * hd : 0);
    }
    return f * sizeof(float) + ((size_t)64 << 20);               /* and the GEMMs' passing scratch */
}
/* the chain keeps one conversation's state on the device: KV_SLOTS > 1 (v41_seq_swap trades the
 * host's) turns it off, as it does the Vulkan chain */
static int v41u_wanted(Model *m) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN"); on = !(e && *e == '0'); }
    return on && !g_v41u_off && g_v41_mux_slots <= 1 && xt_cuda() && g_cud.on && m->gguf && cc_available() && cc_dsv4_ready();
}
/* the struct, its device's context, the parameters, every layer's matrices and state: 0 =
 * declined (the CPU runs the layers from then on) */
static int v41u_setup(Model *m) {
    if (g_v41u) return g_v41u->ok;
    Cfg *c = &m->c;
    const int L = c->n_layers, D = c->dim, H = c->hc_mult, nm = (2 + H) * H, hd = c->head_dim;
    V41U *ch = (V41U *)calloc(1, sizeof *ch);
    if (!ch) return 0;
    g_v41u = ch;
    const char *why = NULL;
    if (!cc_init(g_cud.dev) || !cc_ready()) why = "the chain's context did not come up";
    size_t **offs[] = {&ch->o_an, &ch->o_fn, &ch->o_qn, &ch->o_kn, &ch->o_sink, &ch->o_hca, &ch->o_hcf, &ch->o_cn, &ch->o_ikn, &ch->o_eq, &ch->o_ek};
    for (size_t k = 0; !why && k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = (size_t *)calloc((size_t)L, sizeof(size_t)))) why = "memory";
    if (!why && (!(ch->fna = (ColiCudaTensor **)calloc((size_t)L, sizeof *ch->fna)) || !(ch->fnf = (ColiCudaTensor **)calloc((size_t)L, sizeof *ch->fnf)) ||
                 !(ch->win = (CcBuf **)calloc((size_t)L, sizeof *ch->win)) || !(ch->ckv = (CcBuf **)calloc((size_t)L, sizeof *ch->ckv)) ||
                 !(ch->ikey = (CcBuf **)calloc((size_t)L, sizeof *ch->ikey)) || !(ch->ring = (CcBuf **)calloc((size_t)L, sizeof *ch->ring)) ||
                 !(ch->kv_valid = (int *)calloc((size_t)L, sizeof(int))) || !(ch->ccap = (int *)calloc((size_t)L, sizeof(int)))))
        why = "memory";
    ch->W = c->window; ch->K = c->index_topk > 0 ? c->index_topk : 0; ch->rmax = 1;
    for (int i = 0; i < L; i++) if (c->kv_source[i] && c->compress_ratio[i] > ch->rmax) ch->rmax = c->compress_ratio[i];
    ch->rows = v41u_rows_env();   /* COLI_CUDA_V41_ROWS: a chunk's rows */
    ch->Wd = ch->W + ch->rows;
    /* the parameter arena */
    size_t n = 0;
    for (int i = 0; i < L; i++) {
        ch->o_an[i] = n; n += D; ch->o_fn[i] = n; n += D;
        ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += hd;
        ch->o_sink[i] = n; n += c->n_heads;
        ch->o_hca[i] = n; n += 3 + nm; ch->o_hcf[i] = n; n += 3 + nm;
        if (c->kv_source[i]) { ch->o_cn[i] = n; n += hd; ch->o_ikn[i] = n; n += c->index_head_dim; }
        if (m->L[i].engram_index >= 0) { ch->o_eq[i] = n; n += (size_t)H * D; ch->o_ek[i] = n; n += (size_t)H * D; }
    }
    float *a = why ? NULL : (float *)calloc(n ? n : 1, sizeof(float));
    if (!why && !a) why = "memory";
    for (int i = 0; !why && i < L; i++) {
        Layer *l = &m->L[i];
        memcpy(a + ch->o_an[i], l->attn_norm.w, D * sizeof(float));
        memcpy(a + ch->o_fn[i], l->ffn_norm.w, D * sizeof(float));
        memcpy(a + ch->o_qn[i], l->q_norm.w, c->q_lora * sizeof(float));
        memcpy(a + ch->o_kn[i], l->kv_norm.w, hd * sizeof(float));
        memcpy(a + ch->o_sink[i], l->attn_sink.w, c->n_heads * sizeof(float));
        memcpy(a + ch->o_hca[i], l->hc_attn_scale.w, 3 * sizeof(float));
        memcpy(a + ch->o_hca[i] + 3, l->hc_attn_base.w, nm * sizeof(float));
        memcpy(a + ch->o_hcf[i], l->hc_ffn_scale.w, 3 * sizeof(float));
        memcpy(a + ch->o_hcf[i] + 3, l->hc_ffn_base.w, nm * sizeof(float));
        if (c->kv_source[i]) {
            memcpy(a + ch->o_cn[i], l->comp_norm.w, hd * sizeof(float));
            memcpy(a + ch->o_ikn[i], l->idx_knorm.w, c->index_head_dim * sizeof(float));
        }
        if (l->engram_index >= 0) {
            memcpy(a + ch->o_eq[i], l->eng_q.w, (size_t)H * D * sizeof(float));
            memcpy(a + ch->o_ek[i], l->eng_k.w, (size_t)H * D * sizeof(float));
        }
    }
    if (!why && !(v41u_res(&ch->prm, n, CC_DEV) && cc_begin() && cc_write(ch->prm, 0, a, n * sizeof(float)) && cc_submit(1)))
        why = "the parameters did not reach the device";
    free(a);
    /* every layer's matrices (cu_dense.h's copies) and state */
    for (int i = 0; !why && i < L; i++) {
        Layer *l = &m->L[i];
        const int r = c->compress_ratio[i], og = c->o_groups, ol = c->o_lora;
        const char *bad = NULL; int bt = -1, bo = 0, bi = 0;
        #define V41U_NEED(t_, w_, nm_) do { if (!bad && !(t_)) { bad = nm_; bt = (w_)->gx ? (w_)->gt : -1; bo = (w_)->O; bi = (w_)->I; } } while (0)
        if (!coli_cuda_tensor_upload(&ch->fna[i], l->hc_attn_fn.w, NULL, 0, H * D, nm, g_cud.dev) ||
            !coli_cuda_tensor_upload(&ch->fnf[i], l->hc_ffn_fn.w, NULL, 0, H * D, nm, g_cud.dev)) bad = "hc_fn";
        V41U_NEED(v41u_t8(&l->wq_a), &l->wq_a, "wq_a"); V41U_NEED(v41u_t8(&l->wq_b), &l->wq_b, "wq_b");
        V41U_NEED(v41u_t8(&l->wkv), &l->wkv, "wkv"); V41U_NEED(v41u_t8(&l->wo_b), &l->wo_b, "wo_b");
        V41U_NEED(v41u_t8(&l->sh_w1), &l->sh_w1, "sh_w1"); V41U_NEED(v41u_t8(&l->sh_w3), &l->sh_w3, "sh_w3");
        V41U_NEED(v41u_t8(&l->sh_w2), &l->sh_w2, "sh_w2");
        if (!bad && (l->wo_a.O != og * ol || ol % 16)) bad = "wo_a's groups";
        for (int g = 0; !bad && g < og; g++) V41U_NEED(v41u_woa(&l->wo_a, g, ol), &l->wo_a, "wo_a");
        if (c->kv_source[i]) {
            V41U_NEED(v41u_tb(&l->comp_wkv), &l->comp_wkv, "comp_wkv");
            if (r > 1) V41U_NEED(v41u_tb(&l->comp_wgate), &l->comp_wgate, "comp_wgate");
            V41U_NEED(v41u_tb(&l->idx_wk), &l->idx_wk, "idx_wk");
        }
        if (c->index_source[i] && r > 0) { V41U_NEED(v41u_t8(&l->idx_wq_b), &l->idx_wq_b, "idx_wq_b"); V41U_NEED(v41u_tb(&l->idx_wproj), &l->idx_wproj, "idx_wproj"); }
        if (l->eng_wkv.q) V41U_NEED(v41u_t8(&l->eng_wkv), &l->eng_wkv, "eng_wkv");
        #undef V41U_NEED
        if (bad) {
            fprintf(stderr, "[CUDA] deepseek_v41 chain: layer %d's %s (ggml type %d, %d x %d) not on the device (the dense budget "
                            "%.2f of %.2f GiB used)\n", i, bad, bt, bo, bi, g_cud.used / 1073741824.0, g_cud.budget / 1073741824.0);
            why = "a matrix the device's dense budget or format does not take"; break;
        }
        if (!(ch->win[i] = cc_buf((size_t)ch->Wd * hd * sizeof(float), CC_DEV)) ||
            (c->kv_source[i] && r > 1 && !(ch->ring[i] = cc_buf((size_t)2 * r * hd * sizeof(float), CC_DEV))))
            why = "device memory for the layers' state";
    }
    if (why) {
        fprintf(stderr, "[CUDA] deepseek_v41 chain: %s; the CPU runs a prompt's layers\n", why);
        g_v41u_off = 1;
        return 0;
    }
    ch->ok = 1;
    fprintf(stderr, "[CUDA] deepseek_v41 chain: a prompt's %d layers on the card, layer by layer (%.1f MiB of parameters; "
                    "COLI_CUDA_CHAIN=0: the CPU)\n", L, n * 4 / 1048576.0);
    return 1;
}
static int v41u_cache(V41U *ch, Model *m, int i, int rows) {
    Cfg *c = &m->c;
    if (ch->ccap[i] >= rows) return 1;
    int r = c->compress_ratio[i], host = c->max_positions / (r > 0 ? r : 1), cap = 64;
    while (cap < rows) cap *= 2;
    if (cap > host) cap = host;
    if (cap < rows) return 0;
    cc_free(ch->ckv[i]); cc_free(ch->ikey[i]);
    ch->ckv[i] = cc_buf((size_t)cap * c->head_dim * sizeof(float), CC_DEV);
    ch->ikey[i] = cc_buf((size_t)cap * c->index_head_dim * sizeof(float), CC_DEV);
    ch->kv_valid[i] = 0; ch->ccap[i] = 0;
    if (!ch->ckv[i] || !ch->ikey[i]) return 0;
    ch->ccap[i] = cap;
    return 1;
}
static int v41u_scratch(V41U *ch, Model *m, int rows, int ctx) {
    Cfg *c = &m->c;
    const int D = c->dim, H = c->hc_mult, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H, nh = c->n_heads, hd = c->head_dim;
    const int IH = c->index_n_heads > 0 ? c->index_n_heads : 1, ID = c->index_head_dim > 0 ? c->index_head_dim : 1;
    const int ew = m->engram.active ? m->engram.cols * m->engram.head_dim : 1;
    const size_t r = (size_t)rows;
    ch->wcap = (size_t)ctx;
    return v41u_res(&ch->xs, r * HD, CC_DEV) && v41u_res(&ch->xn, r * HD, CC_DEV) && v41u_res(&ch->mix, r * nm, CC_DEV) &&
           v41u_res(&ch->hpa, r * hr, CC_DEV) && v41u_res(&ch->hpf, r * hr, CC_DEV) && v41u_res(&ch->col, r * D, CC_DEV) &&
           v41u_res(&ch->nrm, r * D, CC_DEV) && v41u_res(&ch->qa, r * c->q_lora, CC_DEV) && v41u_res(&ch->qr, r * c->q_lora, CC_DEV) &&
           v41u_res(&ch->q, r * nh * hd, CC_DEV) && v41u_res(&ch->kv, r * hd, CC_DEV) && v41u_res(&ch->heads, r * nh * hd, CC_DEV) &&
           v41u_res(&ch->grp, r * c->o_groups * c->o_lora, CC_DEV) && v41u_res(&ch->br, r * D, CC_DEV) &&
           v41u_res(&ch->craw, r * hd, CC_DEV) && v41u_res(&ch->cscr, r * hd, CC_DEV) &&
           v41u_res(&ch->iq, r * IH * ID, CC_DEV) && v41u_res(&ch->ihw, r * IH, CC_DEV) &&
           v41u_res(&ch->isc, r * ch->wcap, CC_DEV) &&
           v41u_res(&ch->cs, (r + ch->rmax) * 2 * c->rope_dim, CC_DEV) && v41u_res(&ch->erows, r * ew, CC_DEV) &&
           v41u_res(&ch->ekv, r * (size_t)D * (H + 1), CC_DEV) && v41u_res(&ch->gs, r * c->moe_inter, CC_DEV) &&
           v41u_res(&ch->us, r * c->moe_inter, CC_DEV) && v41u_res(&ch->hs, r * c->moe_inter, CC_DEV) &&
           v41u_res(&ch->ds, r * D, CC_DEV) && v41u_res(&ch->h2d, r * D, CC_DOWN) &&
           v41u_res(&ch->xd, r * HD, CC_DOWN) && v41u_res(&ch->hpd, r * hr, CC_DOWN) && v41u_res(&ch->routed, r * D, CC_UP);
}

/* ---- one layer's pieces (deepseek_v41_chain.h's, op for op) --------------------------------- */
static int v41u_norm(const V41U *ch, CcBuf *x, size_t xo, int xrow, size_t wo, CcBuf *y, size_t yo, int yrow, int rows, int D, float eps) {
    CcNorm p = {rows, D, 1, (int)xo, xrow, D, (int)yo, yrow, D, (int)wo, 0, 0, eps, 1.f};
    return cc_ops()->norm(x, ch->prm, y, &p);
}
static int v41u_pre(V41U *ch, Model *m, ColiCudaTensor *fn, size_t hco, CcBuf *hp, CcBuf *prev, size_t lno, int n) {
    Cfg *c = &m->c; const int H = c->hc_mult, D = c->dim, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H;
    CcMhc sp = {n, H, D, c->hc_iters, 0, HD, 0, nm, 0, hr, 0, D, (int)hco, 0, c->norm_eps, c->hc_eps, 0.f};
    CcMhc co = sp;
    return cc_ops()->matmul(fn, ch->xs, 0, ch->mix, 0, n) && cc_mhc(CC_MHC_SPLIT, ch->xs, ch->mix, hp, ch->prm, NULL, &sp) &&
           cc_mhc(CC_MHC_COLLAPSE, ch->xs, NULL, prev, NULL, ch->col, &co) &&
           v41u_norm(ch, ch->col, 0, D, lno, ch->nrm, 0, D, n, D, c->norm_eps);
}
static int v41u_post(V41U *ch, Model *m, CcBuf *hp, int n) {   /* xn = comb(xs) + post * br, then the two swap */
    Cfg *c = &m->c; const int H = c->hc_mult, D = c->dim, HD = H * D, hr = 2 * H + H * H;
    CcMhc po = {n, H, D, c->hc_iters, 0, HD, 0, D, 0, hr, 0, HD, 0, 0, c->norm_eps, c->hc_eps, 0.f};
    if (!cc_mhc(CC_MHC_POST, ch->xs, ch->br, hp, NULL, ch->xn, &po)) return 0;
    CcBuf *t = ch->xs; ch->xs = ch->xn; ch->xn = t;
    return 1;
}
static int v41u_rope(V41U *ch, Model *m, CcBuf *x, int nseg, int per_row, int x_off, int x_row, int x_seg, int cs_off, int cs_row, int inverse) {
    CcDsRope p = {nseg, per_row, m->c.rope_dim, x_off, x_row, x_seg, cs_off, cs_row, inverse, 0};
    return cc_ds_rope(x, ch->cs, &p);
}
/* wo_a: its output groups' matrices; a chunk's rows to group-major order in q, a GEMM a group
 * into heads, the rows back in grp's order (deepseek_v41_chain.h's v41c_wo_a) */
static int v41u_wo_a(V41U *ch, Model *m, Layer *l, int nr) {
    const Cfg *c = &m->c; const W8 *w = &l->wo_a;
    const int og = c->o_groups, ol = c->o_lora, pg = w->I;
    int ok = 1;
    if (nr == 1) {
        for (int k = 0; ok && k < og; k++) ok = cc_ops()->matmul(v41u_woa(w, k, ol), ch->heads, (size_t)k * pg, ch->grp, (size_t)k * ol, 1);
        return ok;
    }
    static CcRegion *rg; static int crg;
    const int n = nr * og;
    if (n > crg) {
        CcRegion *nrg = (CcRegion *)realloc(rg, (size_t)n * sizeof *rg);
        if (!nrg) return 0;
        rg = nrg; crg = n;
    }
    for (int s = 0; s < nr; s++)
        for (int k = 0; k < og; k++) rg[s * og + k] = (CcRegion){(size_t)k * nr * pg + (size_t)s * pg, (size_t)s * og * pg + (size_t)k * pg, (size_t)pg};
    ok = cc_ops()->copy_regions(ch->q, ch->heads, rg, n);
    for (int k = 0; ok && k < og; k++) ok = cc_ops()->matmul(v41u_woa(w, k, ol), ch->q, (size_t)k * nr * pg, ch->heads, (size_t)k * nr * ol, nr);
    for (int s = 0; s < nr; s++)
        for (int k = 0; k < og; k++) rg[s * og + k] = (CcRegion){(size_t)s * og * ol + (size_t)k * ol, (size_t)k * nr * ol + (size_t)s * ol, (size_t)ol};
    return ok && cc_ops()->copy_regions(ch->grp, ch->heads, rg, n);
}
static int v41u_attention(V41U *ch, Model *m, int i, const V41UFwd *f) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    const int hd = c->head_dim, nh = c->n_heads, rd = c->rope_dim, QL = c->q_lora, r = c->compress_ratio[i];
    const int nr = f->nr, pb = f->pb, qrow = nh * hd, LR = ch->W + ch->K;
    const int tcs = r > 0 ? f->csB + (ch->rmax - 1) * rd : 0;   /* rope_for(layer): the row of position pb */
    const ColiCudaChainOps *o = cc_ops();
    int ok = o->matmul(v41u_t8(&l->wq_a), ch->nrm, 0, ch->qa, 0, nr) &&
             v41u_norm(ch, ch->qa, 0, QL, ch->o_qn[i], ch->qr, 0, QL, nr, QL, c->norm_eps) &&
             o->matmul(v41u_t8(&l->wq_b), ch->qr, 0, ch->q, 0, nr) &&
             o->matmul(v41u_t8(&l->wkv), ch->nrm, 0, ch->kv, 0, nr) &&
             v41u_norm(ch, ch->kv, 0, hd, ch->o_kn[i], ch->kv, 0, hd, nr, hd, c->norm_eps) &&
             v41u_rope(ch, m, ch->q, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 0) &&
             v41u_rope(ch, m, ch->kv, nr, 1, hd - rd, hd, 0, tcs, rd, 0);
    if (!ok) return 0;
    CcRegion *rg = (CcRegion *)malloc((size_t)nr * sizeof *rg);   /* the new rows into the window ring */
    if (!rg) return 0;
    for (int s = 0; s < nr; s++) rg[s] = (CcRegion){(size_t)((pb + s) % ch->Wd) * hd, (size_t)s * hd, (size_t)hd};
    ok = o->copy_regions(ch->win[i], ch->kv, rg, nr);
    free(rg);
    if (ok && r > 0 && c->kv_source[i]) {   /* the compressor, its latents, the index keys */
        const int g0 = pb / r, np = (pb + nr) / r - g0, ihd = c->index_head_dim;
        CcBuf *cb = ch->ckv[i];
        const size_t co = (size_t)g0 * hd;
        ok = o->matmul(v41u_tb(&l->comp_wkv), ch->nrm, 0, ch->craw, 0, nr);
        if (ok && r == 1) ok = v41u_norm(ch, ch->craw, 0, hd, ch->o_cn[i], cb, co, hd, nr, hd, c->norm_eps);
        else if (ok) {
            CcDsComp cp = {nr, pb, r, hd, hd, 0, 0, hd, 0, hd, -1, 0, hd, 0};
            ok = o->matmul(v41u_tb(&l->comp_wgate), ch->nrm, 0, ch->cscr, 0, nr) &&
                 cc_ds_compress(ch->craw, ch->cscr, ch->ring[i], NULL, cb, &cp) &&
                 (np == 0 || v41u_norm(ch, cb, co, hd, ch->o_cn[i], cb, co, hd, np, hd, c->norm_eps));
        }
        if (ok && np > 0) {
            const int bc = f->csB + (g0 * r - (pb - ch->rmax + 1)) * rd;   /* a latent rotates at its group's first position */
            ok = o->matmul(v41u_tb(&l->idx_wk), cb, co, ch->ikey[i], (size_t)g0 * ihd, np) &&
                 v41u_norm(ch, ch->ikey[i], (size_t)g0 * ihd, ihd, ch->o_ikn[i], ch->ikey[i], (size_t)g0 * ihd, ihd, np, ihd, c->norm_eps) &&
                 v41u_rope(ch, m, ch->ikey[i], np, 1, g0 * ihd + ihd - rd, ihd, 0, bc, r * rd, 0) &&
                 v41u_rope(ch, m, cb, np, 1, (int)co + hd - rd, hd, 0, bc, r * rd, 0);
        }
    }
    if (ok && r > 0 && c->index_source[i]) {   /* the indexer: this layer's selection */
        const int IH = c->index_n_heads, ID = c->index_head_dim, width = (pb + nr) / r, cs = c->candidate_source;
        const int cand = v41u_cand_on(c), mrow = cand && cs < i ? (int)ch->wcap : 0;
        const float wscale = (1.0f / sqrtf((float)ID)) * (1.0f / sqrtf((float)IH));
        ok = o->matmul(v41u_t8(&l->idx_wq_b), ch->qr, 0, ch->iq, 0, nr) &&
             v41u_rope(ch, m, ch->iq, nr * IH, IH, ID - rd, IH * ID, ID, tcs, rd, 0) &&
             o->matmul(v41u_tb(&l->idx_wproj), ch->nrm, 0, ch->ihw, 0, nr);
        const int *kl = f->keyl + (size_t)i * f->n + (pb - f->start);
        int same = 1;
        for (int s = 1; s < nr; s++) same &= kl[s] == kl[0];
        /* the candidate source's mask of these rows, a row each in lw_mask (a byte an entry): every
         * chunk of a layer runs before the next layer, so the later index sources read each row's own */
        const int lr = pb - f->start - f->s0;
        for (int s = 0; ok && s < nr; s += same ? nr : 1) {
            CcDsScore sp = {same ? nr : 1, pb + s, r, IH, ID, width, s * IH * ID, IH * ID, s * IH, IH, 0, ID, mrow,
                            (int)ch->wcap, wscale, s * (int)ch->wcap, (lr + s) * mrow};
            ok = cc_ds_score(ch->iq, ch->ihw, ch->ikey[kl[s]], mrow ? ch->lw_mask : NULL, ch->isc, &sp);
        }
        if (ok && cand && cs == i) {
            CcDsCand cp = {nr, pb, r, width, c->candidate_block_size, c->candidate_topk_blocks, (int)ch->wcap, (int)ch->wcap,
                           lr * (int)ch->wcap};
            ok = cc_ds_cand(ch->isc, ch->lw_mask, &cp);
        }
        CcDsTopk tp = {nr, width, ch->K, (int)ch->wcap, f->lo + ch->W, LR, ch->Wd, 0};
        ok = ok && cc_ds_topk(ch->isc, f->list, &tp);
    }
    if (!ok) return 0;
    CcDsAttn a = {nr, nh, hd, ch->W + (r > 0 ? ch->K : 0), f->lo, LR, ch->Wd, 0, 0, 0, qrow, 0, qrow, (int)ch->o_sink[i], 0,
                  1.0f / sqrtf((float)hd)};
    return cc_ds_attn(ch->q, ch->win[i], r > 0 ? ch->ckv[c->index_owner[i]] : NULL, f->list, ch->prm, ch->heads, &a) &&
           v41u_rope(ch, m, ch->heads, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 1) &&
           v41u_wo_a(ch, m, l, nr) && o->matmul(v41u_t8(&l->wo_b), ch->grp, 0, ch->br, 0, nr);
}
static int v41u_shared(V41U *ch, Model *m, Layer *l, int nr) {   /* silu(min(w1 x, lim)) * clamp(w3 x, +-lim), then w2 */
    Cfg *c = &m->c; const int I = c->moe_inter;
    CcMhc sw = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nr * I, 0.f, 0.f, c->swiglu_limit > 0.0f ? c->swiglu_limit : INFINITY};
    const ColiCudaChainOps *o = cc_ops();
    return o->matmul(v41u_t8(&l->sh_w1), ch->nrm, 0, ch->gs, 0, nr) && o->matmul(v41u_t8(&l->sh_w3), ch->nrm, 0, ch->us, 0, nr) &&
           cc_mhc(CC_SWIGLU_CLAMP, ch->gs, ch->us, NULL, NULL, ch->hs, &sw) &&
           o->matmul(v41u_t8(&l->sh_w2), ch->hs, 0, ch->ds, 0, nr);
}
static int v41u_engram(V41U *ch, Model *m, int i, int nr, int pb, float *rows) {
    Cfg *c = &m->c; Engram *e = &m->engram; Layer *l = &m->L[i];
    const int table = l->engram_index, width = e->cols * e->head_dim, H = c->hc_mult, D = c->dim;
    int64_t ids[V41_MAX_NGRAM * V41_MAX_EHEADS];
    double t0 = now_s();
    engram_advise_rows(e, table, pb, nr);
    for (int r = 0; r < nr; r++) {
        engram_hash(e, table, pb + r, ids);
        for (int col = 0; col < e->cols; col++)
            memcpy(rows + (size_t)r * width + (size_t)col * e->head_dim, engram_row(&e->table[table], ids[col], e->head_dim),
                   (size_t)e->head_dim * sizeof(float));
    }
    m->t_engram += now_s() - t0;
    CcDsEngram p = {nr, H, D, 0, D * (H + 1), 0, H * D, (int)ch->o_eq[i], (int)ch->o_ek[i], c->norm_eps};
    return cc_write(ch->erows, 0, rows, (size_t)nr * width * sizeof(float)) &&
           cc_ops()->matmul(v41u_t8(&l->eng_wkv), ch->erows, 0, ch->ekv, 0, nr) && cc_ds_engram(ch->ekv, ch->prm, ch->xs, &p);
}
/* a chunk's RoPE rows and the window half of each row's list (deepseek_v41_chain.h's v41c_chunk_tables) */
static void v41u_chunk_tables(const V41U *ch, const Model *m, int start, int pb, int nr, int csB, float *cs_, int *wl) {
    const Cfg *c = &m->c;
    const int rd = c->rope_dim, W = ch->W, LR = W + ch->K;
    for (int s = 0; s < nr; s++) memcpy(cs_ + (size_t)s * rd, m->rope_window + (size_t)(pb + s) * rd, rd * sizeof(float));
    for (int s = 0; s < nr + ch->rmax - 1; s++) {
        const int p = pb - ch->rmax + 1 + s;
        float *dst = cs_ + csB + (size_t)s * rd;
        if (p < 0) memset(dst, 0, rd * sizeof(float));
        else memcpy(dst, m->rope_compress + (size_t)p * rd, rd * sizeof(float));
    }
    const Layer *l0 = &m->L[0];
    for (int s = 0; s < nr; s++) {
        const int p = pb + s;
        for (int k = 0; k < W; k++) {
            const int q = p - W + 1 + k;
            int e = -1;
            if (q >= 0 && (q >= start || l0->window_pos[q % W] == q)) e = q % ch->Wd;
            wl[(size_t)s * LR + k] = e;
        }
        for (int k = W; k < LR; k++) wl[(size_t)s * LR + k] = -1;
    }
}
static int v41u_state_up(V41U *ch, Model *m, int start, const int *need) {
    const Cfg *c = &m->c;
    const int L = c->n_layers, W = ch->W, hd = c->head_dim;
    int ok = 1;
    for (int q = ch->win_valid > start - W + 1 ? ch->win_valid : start - W + 1; q < start && ok; q++) {
        if (q < 0) continue;
        for (int i = 0; i < L && ok; i++) {
            const Layer *l = &m->L[i];
            if (l->window_pos[q % W] != q) continue;
            ok = cc_write(ch->win[i], (size_t)(q % ch->Wd) * hd, l->window + (size_t)(q % W) * hd, (size_t)hd * sizeof(float));
        }
    }
    for (int i = 0; i < L && ok; i++) {
        if (!c->kv_source[i]) continue;
        const Layer *l = &m->L[i];
        const int r = c->compress_ratio[i], ihd = c->index_head_dim, t0 = ch->kv_valid[i], t1 = need[i];
        if (t1 > t0) ok = cc_write(ch->ckv[i], (size_t)t0 * hd, l->ckv + (size_t)t0 * hd, (size_t)(t1 - t0) * hd * sizeof(float)) &&
                          cc_write(ch->ikey[i], (size_t)t0 * ihd, l->ikey + (size_t)t0 * ihd, (size_t)(t1 - t0) * ihd * sizeof(float));
        if (ok && r > 1) ok = cc_write(ch->ring[i], 0, l->cstate_kv, (size_t)r * hd * sizeof(float)) &&
                              cc_write(ch->ring[i], (size_t)r * hd, l->cstate_score, (size_t)r * hd * sizeof(float));
    }
    return ok;
}
static int v41u_state_pull(V41U *ch, Model *m, int start, int E, const size_t *pc, size_t pw) {
    const Cfg *c = &m->c;
    const int L = c->n_layers, W = ch->W, hd = c->head_dim;
    int ok = 1;
    for (int i = 0; i < L && ok; i++) {
        const int q0 = E - W > start ? E - W : start, nq = E - q0;
        CcRegion *rg = (CcRegion *)malloc((size_t)nq * sizeof *rg);
        if (!rg) return 0;
        for (int q = q0; q < E; q++) rg[q - q0] = (CcRegion){pc[i] + (size_t)(q - q0) * hd, (size_t)(q % ch->Wd) * hd, (size_t)hd};
        ok = cc_ops()->copy_regions(ch->pull, ch->win[i], rg, nq);
        free(rg);
        if (!ok || !c->kv_source[i]) continue;
        const int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim;
        const size_t at = pc[i] + pw;
        ok = cc_copy(ch->pull, at, ch->ckv[i], (size_t)g0 * hd, (size_t)np * hd) &&
             cc_copy(ch->pull, at + (size_t)np * hd, ch->ikey[i], (size_t)g0 * ihd, (size_t)np * ihd) &&
             (r <= 1 || cc_copy(ch->pull, at + (size_t)np * (hd + ihd), ch->ring[i], 0, (size_t)2 * r * hd));
    }
    return ok;
}

/* Every layer for a prompt's n rows of streams h (positions start..): the streams after the
 * last layer and its FFN mix back into h and pre_mix (every row when all, else the last), the
 * host's state as the CPU would have left it. Returns the layers run (all of them), 0: not
 * taken (h and pre_mix untouched; *lost: a frame failed). */
static int v41u_forward(Model *m, float *h, float *pre_mix, int n, int start, int spec, int all, int *lost) {
    *lost = 0;
    Cfg *c = &m->c;
    const int L = c->n_layers, D = c->dim, H = c->hc_mult, HD = H * D, hr = 2 * H + H * H, hd = c->head_dim, rd = c->rope_dim;
    const int T = m->spec.active ? c->n_spec_targets : 0, E = start + n;
    if (spec || T || !v41u_wanted(m)) return 0;
    if (!v41u_setup(m)) return 0;
    V41U *ch = g_v41u;
    v41u_lower(m, start);   /* this forward writes from start: nothing the device holds past it is the host's */
    const int W = ch->W, LR = W + ch->K, rows = n < ch->rows ? n : ch->rows;
    if (n <= rows) return 0;   /* a prompt of more than one chunk */
    /* in segments of whole chunks, as even as they come, up to v41u_seg_rows each */
    const int chunks = (n + rows - 1) / rows, nseg = (n + v41u_seg_rows(ch) - 1) / v41u_seg_rows(ch);
    const int cand = v41u_cand_on(c);
    int per = (chunks + nseg - 1) / nseg * rows;
    if (cand && (size_t)per * (size_t)E >= (size_t)0x7fffffff) return 0;   /* the mask's offsets */
    V41UFwd f = {start, n, 0, 0, rows * rd, NULL, NULL, 0, 0};
    f.keyl = (int *)malloc((size_t)L * n * sizeof(int));
    int *need = (int *)calloc((size_t)L, sizeof(int));
    size_t *pc = (size_t *)calloc((size_t)L, sizeof(size_t));
    int *wl = (int *)malloc((size_t)rows * LR * sizeof(int));
    float *cs_ = (float *)malloc((size_t)(rows + ch->rmax) * 2 * rd * sizeof(float));
    float *er = m->engram.active ? (float *)malloc((size_t)rows * m->engram.cols * m->engram.head_dim * sizeof(float)) : NULL;
    float *outs = (float *)malloc((size_t)(all ? n : 1) * HD * sizeof(float)), *pres = (float *)malloc((size_t)(all ? n : 1) * H * sizeof(float));
    int ok = f.keyl && need && pc && wl && cs_ && (!m->engram.active || er) && outs && pres;
    /* whose index keys each index source reads, per row (indexer_run, published_owner), and the
     * rows of each kv_source layer the forward reads that it does not produce first */
    static int tidy = -1;
    if (tidy < 0) tidy = getenv("V41_INDEX_OWNER") != NULL;
    for (int i = 0; ok && i < L; i++) if (c->kv_source[i] && c->compress_ratio[i] > 0) need[i] = start / c->compress_ratio[i];
    int pub = m->published_index_k ? m->published_index_layer : -1, last_pub = -1;
    for (int i = 0; ok && i < L; i++) {
        const int r = c->compress_ratio[i];
        if (r <= 0) continue;
        const int owner = c->index_owner[i];
        if (c->kv_source[i] && E / r - start / r > 0) { pub = i; last_pub = i; }
        if (E / r > need[owner]) need[owner] = E / r;
        for (int t = 0; t < n; t++) {
            int k = owner;
            if (c->index_source[i] && !tidy) {
                if (m->pub_rows > 0) {
                    const int p = published_owner(m, i, start, t);
                    k = p >= 0 ? p : m->pub_before ? m->pub_before_layer : owner;
                } else if (n == 1 && pub >= 0) k = pub;
            }
            f.keyl[(size_t)i * n + t] = k;
            if (c->index_source[i] && (start + t + 1) / r > need[k]) need[k] = (start + t + 1) / r;
        }
    }
    for (int i = 0; ok && i < L; i++) {
        if (!c->kv_source[i]) continue;
        const int r = c->compress_ratio[i], host = c->max_positions / r;
        if (need[i] > host) need[i] = host;
        const int want = need[i] > E / r ? need[i] : E / r;
        ok = v41u_cache(ch, m, i, want > 0 ? want : 1);
    }
    /* what comes back at the end: each layer's window rows, its compressed rows, keys and group */
    const size_t pw = (size_t)(n < W ? n : W) * hd;
    size_t pn = 0;
    for (int i = 0; ok && i < L; i++) {
        pc[i] = pn; pn += pw;
        const int r = c->compress_ratio[i];
        if (!c->kv_source[i]) continue;
        pn += (size_t)(E / r - start / r) * (hd + c->index_head_dim);
        if (r > 1) pn += (size_t)2 * r * hd;
    }
    ok = ok && v41u_scratch(ch, m, rows, E) && v41u_res(&ch->pull, pn, CC_DOWN) && v41u_res(&ch->lw_list, (size_t)n * LR, CC_DEV);
    for (int seg_ok = 0; ok && !seg_ok;) {   /* a segment's rows on the device; refused, segments half as long */
        seg_ok = v41u_res(&ch->lw_xs, (size_t)per * HD, CC_DEV) && v41u_res(&ch->lw_hp, (size_t)per * hr, CC_DEV) &&
                 v41u_res(&ch->lw_ds, (size_t)per * D, CC_DEV) && (!cand || v41u_res(&ch->lw_mask, ((size_t)per * E + 3) / 4, CC_DEV));
        if (seg_ok) break;
        CcBuf **lb[] = {&ch->lw_xs, &ch->lw_hp, &ch->lw_ds, &ch->lw_mask};
        for (size_t k = 0; k < sizeof lb / sizeof *lb; k++) { cc_free(*lb[k]); *lb[k] = NULL; }
        if (per <= 2 * rows) ok = 0;
        else per = (per / 2 + rows - 1) / rows * rows;
    }
    if (ok && per < n) {
        static int said;
        if (!said++) fprintf(stderr, "[CUDA] deepseek_v41 chain: a prompt of %d rows in segments of %d, each through every layer "
                                     "(COLI_CUDA_V41_SEG sets their rows)\n", n, per);
    }
    f.list = ch->lw_list;
    if (ok && ch->lw_cap < (size_t)per) {
        float *a = (float *)realloc(ch->lw_in, (size_t)per * D * sizeof(float));
        if (a) ch->lw_in = a;
        float *b = a ? (float *)realloc(ch->lw_out, (size_t)per * D * sizeof(float)) : NULL;
        if (b) ch->lw_out = b;
        ok = a && b;
        if (ok) ch->lw_cap = (size_t)per;
    }
    float *hp0 = ok ? (float *)calloc((size_t)rows * hr, sizeof(float)) : NULL;
    if (!hp0) {
        static int said;
        if (ok && !said++) fprintf(stderr, "[CUDA] deepseek_v41 chain: no memory for a prompt of %d rows; the CPU runs its layers\n", n);
        else if (!said++) fprintf(stderr, "[CUDA] deepseek_v41 chain: device memory for a prompt of %d rows refused; the CPU runs its layers\n", n);
        free(f.keyl); free(need); free(pc); free(wl); free(cs_); free(er); free(outs); free(pres);
        return 0;
    }
    for (int s = 0; s < rows; s++) hp0[(size_t)s * hr] = 1.0f;   /* the first site collapses with [1, 0, ...] */
    ok = cc_begin() && v41u_state_up(ch, m, start, need) && cc_submit(1);
    for (int c0 = 0; ok && c0 < n; c0 += rows) {   /* the window halves of every row's list */
        const int nr = n - c0 < rows ? n - c0 : rows;
        v41u_chunk_tables(ch, m, start, start + c0, nr, f.csB, cs_, wl);
        ok = cc_begin() && cc_write(ch->lw_list, (size_t)c0 * LR, wl, (size_t)nr * LR * sizeof(int)) && cc_submit(1);
    }
    /* past the cross-layer source only the chunks of the prompt's tail (v41_prefill_tail) from tf on */
    const int tail = m->ptail_rows > 0 && m->ptail_after < L - 1;
    int tf = 0;
    if (tail) while (tf < n && tf + (n - tf < rows ? n - tf : rows) <= n - m->ptail_rows) tf += rows;
    int stopped = 0;
    for (int s0 = 0; ok && !stopped && s0 < n; s0 += per) {   /* a segment's rows through every layer */
        const int se = s0 + per < n ? s0 + per : n;
        f.s0 = s0;
        for (int c0 = s0; ok && c0 < se; c0 += rows) {   /* its streams, a chunk at a time */
            const int nr = se - c0 < rows ? se - c0 : rows;
            ok = cc_begin() && cc_write(ch->lw_xs, (size_t)(c0 - s0) * HD, h + (size_t)c0 * HD, (size_t)nr * HD * sizeof(float)) &&
                 cc_write(ch->lw_hp, (size_t)(c0 - s0) * hr, hp0, (size_t)nr * hr * sizeof(float)) && cc_submit(1);
        }
        for (int i = 0; ok && i < L; i++) {
            Layer *l = &m->L[i];
            const int first = tail && i > m->ptail_after && tf > s0 ? tf : s0;
            if (first >= se) continue;
            double ta = now_s();
            XT(stream_prefetch)(i, se - first);
            for (int c0 = first; ok && c0 < se; c0 += rows) {
                if (coli_prompt_break()) { stopped = 1; break; }
                const int nr = se - c0 < rows ? se - c0 : rows, pb = start + c0, lc = c0 - s0;
                f.pb = pb; f.nr = nr; f.lo = c0 * LR;
                v41u_chunk_tables(ch, m, start, pb, nr, f.csB, cs_, wl);   /* the RoPE rows (the list's window half is up) */
                ok = cc_begin() && cc_write(ch->cs, 0, cs_, (size_t)(f.csB + (nr + ch->rmax - 1) * rd) * sizeof(float)) &&
                     cc_copy(ch->xs, 0, ch->lw_xs, (size_t)lc * HD, (size_t)nr * HD) && cc_copy(ch->hpf, 0, ch->lw_hp, (size_t)lc * hr, (size_t)nr * hr);
                if (ok && l->engram_index >= 0) ok = v41u_engram(ch, m, i, nr, pb, er);
                ok = ok && v41u_pre(ch, m, ch->fna[i], ch->o_hca[i], ch->hpa, ch->hpf, ch->o_an[i], nr) &&
                     v41u_attention(ch, m, i, &f) && v41u_post(ch, m, ch->hpa, nr) &&
                     v41u_pre(ch, m, ch->fnf[i], ch->o_hcf[i], ch->hpf, ch->hpa, ch->o_fn[i], nr) &&
                     cc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D) && v41u_shared(ch, m, l, nr) &&
                     cc_copy(ch->lw_xs, (size_t)lc * HD, ch->xs, 0, (size_t)nr * HD) && cc_copy(ch->lw_hp, (size_t)lc * hr, ch->hpf, 0, (size_t)nr * hr) &&
                     cc_copy(ch->lw_ds, (size_t)lc * D, ch->ds, 0, (size_t)nr * D) && cc_submit(1);
                if (ok) memcpy(ch->lw_in + (size_t)lc * D, cc_ptr(ch->h2d), (size_t)nr * D * sizeof(float));
            }
            m->t_attn += now_s() - ta;
            if (!ok || stopped) break;
            const double t1 = now_s();
            moe_run_at(m, l, &m->cache[i], "layers", i, c->n_routed, c->n_activated, ch->lw_in + (size_t)(first - s0) * D, se - first,
                       ch->lw_out + (size_t)(first - s0) * D, 0);
            ch->host_ms += (now_s() - t1) * 1e3;
            for (int c0 = first; ok && c0 < se; c0 += rows) {   /* each chunk's FFN branch into its streams */
                const int nr = se - c0 < rows ? se - c0 : rows, lc = c0 - s0;
                memcpy(cc_ptr(ch->routed), ch->lw_out + (size_t)lc * D, (size_t)nr * D * sizeof(float));
                CcEw add = {CC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = cc_begin() && cc_copy(ch->xs, 0, ch->lw_xs, (size_t)lc * HD, (size_t)nr * HD) &&
                     cc_copy(ch->hpf, 0, ch->lw_hp, (size_t)lc * hr, (size_t)nr * hr) && cc_copy(ch->ds, 0, ch->lw_ds, (size_t)lc * D, (size_t)nr * D) &&
                     cc_ops()->ew(ch->br, ch->routed, ch->ds, NULL, NULL, &add) && v41u_post(ch, m, ch->hpf, nr) &&
                     cc_copy(ch->lw_xs, (size_t)lc * HD, ch->xs, 0, (size_t)nr * HD) && cc_submit(1);
            }
        }
        /* the segment's streams and mixes (every row's, or the prompt's last) and, after the last
         * one, what the host's state gets back */
        for (int c0 = s0; ok && !stopped && c0 < se; c0 += rows) {
            const int nr = se - c0 < rows ? se - c0 : rows, last = c0 + nr == n;
            if (!all && !last) continue;
            ok = cc_begin() && cc_copy(ch->xd, 0, ch->lw_xs, (size_t)(c0 - s0) * HD, (size_t)nr * HD) &&
                 cc_copy(ch->hpd, 0, ch->lw_hp, (size_t)(c0 - s0) * hr, (size_t)nr * hr) &&
                 (!last || v41u_state_pull(ch, m, start, E, pc, pw)) && cc_submit(1);
            if (!ok) break;
            const float *xd = (const float *)cc_ptr(ch->xd), *hpd = (const float *)cc_ptr(ch->hpd);
            for (int s = 0; s < nr; s++) {
                const int t = c0 + s;
                if (!all && t != n - 1) continue;
                memcpy(outs + (size_t)(all ? t : 0) * HD, xd + (size_t)s * HD, (size_t)HD * sizeof(float));
                memcpy(pres + (size_t)(all ? t : 0) * H, hpd + (size_t)s * hr, (size_t)H * sizeof(float));
            }
        }
    }
    free(hp0);
    if (!ok || stopped) {   /* a frame failed, or the prompt was stopped: the host state as it was, the device's stale */
        free(f.keyl); free(need); free(pc); free(wl); free(cs_); free(er); free(outs); free(pres);
        v41u_lower(m, start);
        if (!ok) { *lost = 1; g_v41u_off = 1; ch->ok = 0; fprintf(stderr, "[CUDA] deepseek_v41 chain: a frame failed; the CPU runs the layers from now on\n"); }
        return 0;
    }
    {   /* every chunk is through: the host's state as the CPU would have left it */
        const float *pl = (const float *)cc_ptr(ch->pull);
        for (int i = 0; i < L; i++) {
            Layer *l = &m->L[i];
            const int q0 = E - W > start ? E - W : start;
            for (int t = 0; t < n; t++) {
                const int p = start + t, slot = p % W;
                if (p < q0) continue;
                memcpy(l->window + (size_t)slot * hd, pl + pc[i] + (size_t)(p - q0) * hd, (size_t)hd * sizeof(float));
                l->window_pos[slot] = p;
            }
            if (!c->kv_source[i]) continue;
            const int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim;
            const float *src = pl + pc[i] + pw;
            memcpy(l->ckv + (size_t)g0 * hd, src, (size_t)np * hd * sizeof(float));
            memcpy(l->ikey + (size_t)g0 * ihd, src + (size_t)np * hd, (size_t)np * ihd * sizeof(float));
            if (r <= 1) continue;
            const float *ring = src + (size_t)np * (hd + ihd);
            memcpy(l->cstate_kv, ring, (size_t)r * hd * sizeof(float));
            memcpy(l->cstate_score, ring + (size_t)r * hd, (size_t)r * hd * sizeof(float));
        }
        if (last_pub >= 0) { m->published_index_k = m->L[last_pub].ikey; m->published_index_layer = last_pub; }
        ch->win_valid = E;
        for (int i = 0; i < L; i++) if (c->kv_source[i]) ch->kv_valid[i] = E / c->compress_ratio[i];
    }
    if (all) { memcpy(h, outs, (size_t)n * HD * sizeof(float)); memcpy(pre_mix, pres, (size_t)n * H * sizeof(float)); }
    else { memcpy(h + (size_t)(n - 1) * HD, outs, (size_t)HD * sizeof(float)); memcpy(pre_mix + (size_t)(n - 1) * H, pres, (size_t)H * sizeof(float)); }
    free(f.keyl); free(need); free(pc); free(wl); free(cs_); free(er); free(outs); free(pres);
    ch->forwards++;
    return L;
}
#endif
