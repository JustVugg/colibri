/* qwen38_cuda_chain.h -- Qwen3.8 Flash Next's layers as a dense chain on the CUDA device
 * (cuda_chain.h). Included once by qwen38_core.h in a COLI_CUDA build, after the CPU
 * forward it stands in for; COLI_CUDA_CHAIN=1 asks for it (q38cc_setup decides at
 * startup, after the tier placed the trunk). The CUDA twin of qwen38_chain.h, with the
 * same frames and the same state contract, on qwen36_cuda_chain.h's table:
 *   - four hyper-connection streams (hyper, S x 4H) stay on the device; every block
 *     reads them through its gated residual (per-stream norm, the low-rank down/up
 *     pair, the stream mix, the inject weights) and writes back hyper += inject x block;
 *   - the PLE layer: the n-gram table rows stay on the host (disk reads, in token
 *     order, with the n-gram history); only their rows go up, then key and value
 *     projections, the gate and the dilated convolution with its ring on the device;
 *   - QSA attention: the K/V/index-key rows on the device (the host's copies kept
 *     canonical), each block's pooled key computed once on the device when a step
 *     completes the block, and the indexer's top-k selection per query row on the
 *     device, feeding the attention's selection list;
 *   - the MTP head stays on the CPU: the chain hands it every row's four streams when
 *     the model has one; a verify forward (S = 2 to Q38_SPEC_ROWS) runs row-wise (the
 *     warp GEMV computes each row on its own: a decode step's bits) and snapshots the
 *     DeltaNet state, the conv rings and the PLE ring after each of its rows but the
 *     last on the device, one slot per row, so a draft rejected after row r rolls back
 *     by swapping slot r's device buffers in (q38cc_rollback).
 * Per layer the host gets the MoE input rows and the router logits, and the K/V and
 * index-key rows of an attention layer; it sends the routed sum back. The shared
 * expert runs on the device (frame A2) while the host computes the routed experts
 * through the fp8 tier.
 *
 * The matrices: the tier's resident int8 copies where the placer put them (w->gpu);
 * every other matrix of a layer, the final mixer and lm_head the chain uploads itself
 * (w->cc): the trunk's int8 rows when the CPU holds them (the same bits), else the
 * BF16/F32 rows as f32. So the chain runs whatever the placement did; what it costs in
 * VRAM beyond the tier's copies are the small matrices the placer never sees (the
 * gated residuals' low-rank pairs, the DeltaNet b/a rows, the PLE projections).
 *
 * State ownership as in qwen36's: the host's KV/index caches canonical, device mirrors
 * behind a watermark (kv_valid), pooled block keys behind pk_valid; the DeltaNet
 * state, conv rings and PLE ring on the device while the chain runs (dn_where),
 * brought back before the host reads them (prefix cache, pins) and pushed up after
 * the host writes them (reset, restore). Q38_DN_GPU's per-layer step is not set up
 * beside the chain. A device lost while it holds that state: the engine rebuilds it
 * on the CPU from the prefix record and continues there (q38cc_recover).
 * COLI_CUDA_CHAIN_FAULT=n fakes the loss at the n-th frame. */
#include "cuda_chain.h"

#define Q38CC_HOST 0
#define Q38CC_DEV  1
#define Q38CC_BOTH 2
#define Q38CC_PROMO_WIN 64   /* frames the brake looks back on (q38cc_braked) */

typedef struct {
    int ok, failed, rows, cap, dev;
    int nl, head;                     /* every layer, and the final mixer and lm_head */
    size_t prm_floats;
    CcBuf *prm;
    size_t *o_an, *o_mn, *o_qn, *o_kn, *o_iqn, *o_ikn, *o_conv, *o_dn;
    size_t o_fn, o_ple, o_pconv;
    ColiCudaTensor **t_sg;            /* the shared expert's gate row, f32 */
    CcBuf **rec, **ring, **kc, **vc, **ik, **pk;
    CcBuf **rec_snap[Q38_SPEC_SNAPS], **ring_snap[Q38_SPEC_SNAPS];   /* a verify's copies, slot r after row r */
    int snap_slots;
    CcBuf *ple_ring;                  /* [ple_nslots][W][SL]: current at ple_slot[0], the copy after row r at ple_slot[1+r] */
    int ple_slot[Q38_SPEC_ROWS], ple_nslots;
    int *kv_valid, *pk_valid, *attn_ord, n_attn;
    int dn_where, host_zero, snap_valid;
    CcBuf *hyper, *hn, *low, *mix, *mixed, *inj_a, *inj_m, *blk, *q, *k, *v, *ip, *ctx, *qsc, *sel;
    CcBuf *hyper_all;                          /* every row's streams through a prompt run layer by layer (q38cc_forward_lm) */
    CcBuf *qkv, *z, *ab, *cv, *dny, *lg, *gs, *us, *hs, *ds, *sgd;
    CcBuf *emb, *keys, *val, *gated, *normv;
    CcBuf *mixd, *lgd, *kvd, *outd, *hypd, *find, *routed, *cs, *pcs;
    CcBuf *gseg, *nseg;               /* a deep verify's PLE rows past its first copy (q38cc_ple) */
    float *host_routed, *host_emb;
    unsigned long long forwards, frames, fault_at, dec_n;
    double host_ms, wait_ms, dec_wait_ms, dec_host_ms;
    /* the routed experts in the frame (q38cc_dm_*) */
    int dm_on, dm_fmt, dm_gs, dm_read, dm_held;
    unsigned dm_epoch, dm_seq;
    CcExpert *dm_host; uint8_t *dm_res, *dm_ran, *dm_dev; int *dm_part;   /* dm_ran: the step's layers that ran so */
    CcBuf *dm_ids, *dm_w, *dm_idsd, *dm_wd, *dm_hid, *dm_y, *dm_hy, *dm_dfl, *dm_hfl, *dm_tab;
    /* pool_cpu: a decode step's pooled experts on the CPU (q38cc_dm_serve), from the decode table
     * dm_tabd that lacks them (dm_partd: a layer with any such slot); pc_*: the CPU's scratch */
    CcBuf *dm_tabd; int *dm_partd, pool_cpu, pc_cap; float *pc_gu, *pc_hh, *pc_dn;
    unsigned long long pc_pairs; double pc_ms;
    unsigned long long dm_steps, dm_hyb, dm_pairs;   /* layer steps in the frame, the hybrid ones, slots the host computed */
    double dm_host_ms, dm_spin_ms;
    /* the host pool (q38cc_pool_init): pool_n experts in pool_nb buffers of pool_per, pool_of[l*E+e] its slot or -1 */
    CcBuf **pool_buf; int pool_nb, pool_per, pool_n; size_t pool_stride; int32_t *pool_of;
    CcBuf *pool_stage;               /* a decode step's pooled experts copied here first (cc_moe_stage) */
    /* a prompt chunk's: a layer's pooled experts copied to dm_stage (cc_fetch) while the layers before it run */
    CcBuf *dm_stage, *dm_tabb; CcExpert *dm_hostb; CcFetch *dm_fr; int *dm_fro, *dm_frn, dm_stage_cap;
    unsigned long long dm_staged; double dm_stage_bytes;
    unsigned long long lm_forwards;            /* prompts run layer by layer */
    /* adaptive promotion (q38cc_promote_*): the tier's arena is the chain's (own) */
    int own, promo_rate, stage_dirty;
    char *ar_base; size_t ar_slot, ar_off_u, ar_off_d; int ar_n;
    int32_t *vslot, *owner;              /* [L*E] an expert's slot (-1), [ar_n] a slot's expert (-1) */
    uint8_t *busy;                       /* [ar_n] a newcomer's bytes on their way */
    uint32_t *heat; uint32_t heat_tick;  /* [L*E] this conversation's routes, halved every Q38CC_DECAY rows */
    struct Q38CCPend { int k, slot; unsigned ticket; } *pend; int npend, cpend;
    int32_t *chg; int nchg;              /* table entries rewritten since the last write ([L*E] at most) */
    unsigned rng;
    unsigned long long promos, promo_victims;
    unsigned long long pool_reads;   /* slots a step's table read from the pool */
    /* the brake (q38cc_braked): the routes and pool reads of the last Q38CC_PROMO_WIN frames;
     * at promo_hits percent on the card or more nothing moves (the baselines: the counters at
     * the last frame that was looked at) */
    int promo_hits, pr_wi, pr_wn; unsigned pr_wroute[Q38CC_PROMO_WIN], pr_wmiss[Q38CC_PROMO_WIN];
    unsigned long long routes, pr_routes0, pr_reads0, pr_braked;
} Q38CChain;
static Q38CChain *q38cc_of(Model *m) { return (Q38CChain *)m->cchain; }
static int q38cc_dm_able(Q38CChain *ch, Model *m);
static int g_cuda_chain = 0;     /* COLI_CUDA_CHAIN asked for it and the chain is up */

/* COLI_CUDA_CHAIN=1 asks for the chain, 0 turns it off; unset, on for a GGUF (without it the
 * experts the card leaves run on the CPU, a round trip a matrix: Qwen3.8-Flash-Next IQ2_XS on
 * an RTX PRO 4000, prompt 19-21 and decode 11.6 tok/s, against 185-187 and 47-48 with it) */
static int q38cc_env_on(void) { const char *e = getenv("COLI_CUDA_CHAIN"); return e && *e ? *e == '1' : g_q38_gguf; }
static int q38cc_rows_env(void) {
    const char *e = getenv("COLI_CUDA_CHAIN_ROWS");
    /* a chunk reads every pooled expert over the bus once: at 512 rows a 1628-token prompt read them four times
     * (90.8 GB on an RTX 4090), at 2048 once (22.8 GB), 148.7 -> 352.4 tok/s. Each row keeps ~0.6 MiB of the
     * card for the chain (q38cc_reserve_bytes): 2048 rows hold 0.9 GiB more than 512, fewer experts there */
    int r = e && *e ? atoi(e) : 2048;
    return r < 1 ? 1 : r > 8192 ? 8192 : r;
}
/* COLI_CUDA_CHAIN_LAYERWISE (q38cc_forward_lm): 0 chunk by chunk */
static int q38cc_layer_major(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN_LAYERWISE"); on = !(e && *e == '0'); }
    return on;
}
/* a prompt run layer by layer goes in segments of this many rows (whole chunks, at least one):
 * a segment's streams on the device (hyper_all) stay bounded whatever the prompt's length.
 * COLI_CUDA_CHAIN_LAYERWISE_ROWS sets it (tests: two segments of a short prompt) */
static int q38cc_lm_rows(void) {
    static int r = -1;
    if (r < 0) { const char *e = getenv("COLI_CUDA_CHAIN_LAYERWISE_ROWS"); r = e && atoi(e) > 0 ? atoi(e) : 8192; }
    return r;
}
static void q38cc_fatal(const char *what) {
    fprintf(stderr, "[chain] qwen38: %s -- stopping (COLI_CUDA_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
static int q38cc_geometry_ok(const Cfg *c) {
    int attn = 0, dn = 0, kd = c->dn_kdim;
    int kd_ok = kd == 4 || kd == 8 || kd == 16 || kd == 32 || kd == 64 || kd == 128 || kd == 256;
    for (int i = 0; i < c->layers; i++) { if (c->is_attn[i]) attn = 1; else dn = 1; }
    if (attn && (c->head_dim > 256 || c->q_heads % c->kv_heads || (c->rotary_dim & 1) || c->rotary_dim > c->head_dim ||
                 c->idx_kheads != 1 || c->idx_dim > 256 || c->rotary_dim > c->idx_dim || c->idx_ratio < 1)) return 0;
    if (dn && (c->dn_vdim > 128 || !kd_ok || c->dn_convk < 2 || c->dn_convk > 9 || c->dn_vheads % c->dn_kheads)) return 0;
    if (c->ple_layer >= 0 && c->ple_layer < c->layers && (c->ple_convk - 1) * c->ngram_size > 32) return 0;
    return c->hc_count > 0 && c->hc_width == c->hc_count * c->hidden;
}
static size_t q38cc_ple_cells(const Cfg *c) { return (size_t)c->hc_width * (c->ple_convk - 1) * c->ngram_size; }
static int q38cc_selrow(const Cfg *c) { return 1 + c->idx_budget + c->idx_ratio - 1; }
static size_t q38cc_kv_stride(Q38CChain *ch, const Cfg *c) {   /* per attention layer in kvd: K, V, IK rows */
    return (size_t)ch->rows * (2 * (size_t)c->kv_heads * c->head_dim + c->idx_dim);
}

/* ---- the matrices ------------------------------------------------------------------ */
/* A weight's device copy: the tier's (w->gpu) or the chain's own (w->cc, made here from
 * the trunk's int8 rows, a GGUF's blocks, else the BF16/F32 rows as f32). NULL: nothing to
 * upload, or the device refused. */
static ColiCudaTensor *q38cc_t(Q38CChain *ch, Q38Weight *w) {
    if (!w) return NULL;
    if (w->gpu) {
        int d = -1;
        ColiCudaTensor *t = qt_dense_tensor(w->gpu - 1, &d);
        if (t && d == ch->dev) return t;
        if (t) return NULL;   /* on another card: the chain does not span cards */
    }
    if (w->cc) return (ColiCudaTensor *)w->cc;
    ColiCudaTensor *t = NULL;
    const int gf = q38_cuda_gq_fmt(w);
    if (w->q8 && w->q8sc) {
        if (!coli_cuda_tensor_upload(&t, w->q8, w->q8sc, 1, w->cols, w->rows, ch->dev)) return NULL;
    } else if (gf) {   /* a GGUF's blocks the placer did not take (below the trunk's floor): the file's rows */
        uint8_t *rows = NULL;
        if (gq_r16_base(w->gq_type) >= 0) {
            if (!(rows = (uint8_t *)malloc((size_t)w->rows * gq_row_bytes(gf - 100, w->cols)))) return NULL;
            gq_r16_unpack(w->gq_type, rows, (const uint8_t *)w->data, w->rows, w->cols);
        }
        int ok = coli_cuda_tensor_upload(&t, rows ? (const void *)rows : w->data, NULL, gf, w->cols, w->rows, ch->dev);
        free(rows);
        if (!ok) return NULL;
    } else if (w->data && (w->kind == Q38_WEIGHT_F32 || w->kind == Q38_WEIGHT_BF16)) {
        size_t n = (size_t)w->rows * w->cols;
        float *f = NULL, *ones = falloc(w->rows);
        for (int o = 0; o < w->rows; o++) ones[o] = 1.f;
        if (w->kind == Q38_WEIGHT_BF16) {
            f = falloc((int64_t)n);
            const uint16_t *src = (const uint16_t *)w->data;
            for (size_t i = 0; i < n; i++) f[i] = bf16_to_f32(src[i]);
        }
        int ok = coli_cuda_tensor_upload(&t, f ? f : (const float *)w->data, ones, 0, w->cols, w->rows, ch->dev);
        free(f); free(ones);
        if (!ok) return NULL;
    } else return NULL;
    w->cc = t;
    return t;
}
/* f32 rows [O x I] as a resident fmt 0 tensor (unit scales: the kernel reads none) */
static ColiCudaTensor *q38cc_f32_rows(int dev, const float *w, int I, int O) {
    ColiCudaTensor *t = NULL;
    float *ones = falloc(O);
    for (int o = 0; o < O; o++) ones[o] = 1.f;
    int ok = coli_cuda_tensor_upload(&t, w, ones, 0, I, O, dev);
    free(ones);
    return ok ? t : NULL;
}
#define Q38CC_MATS 24
static int q38cc_mats(Model *m, int i, Q38Weight **w, const char **nm) {
    Cfg *c = &m->c; Layer *l = &m->L[i]; int n = 0;
    Q38Weight *all[] = {&l->attn_gr.down, &l->attn_gr.up, &l->attn_gr.inject, &l->mlp_gr.down, &l->mlp_gr.up,
                        &l->mlp_gr.inject, &l->router, &l->sh_g, &l->sh_u, &l->sh_d};
    static const char *names[] = {"hcad", "hcau", "hcai", "hcmd", "hcmu", "hcmi", "router", "shg", "shu", "shd"};
    for (size_t k = 0; k < sizeof all / sizeof *all; k++) { nm[n] = names[k]; w[n++] = all[k]; }
    if (c->is_attn[i]) { nm[n] = "attnq"; w[n++] = &l->q; nm[n] = "attnk"; w[n++] = &l->k; nm[n] = "attnv"; w[n++] = &l->v;
                         nm[n] = "attno"; w[n++] = &l->o; nm[n] = "qsaidx"; w[n++] = &l->idx_qk; }
    else { nm[n] = "dnqkv"; w[n++] = &l->dn_qkv; nm[n] = "dnz"; w[n++] = &l->dn_z; nm[n] = "dnb"; w[n++] = &l->dn_b;
           nm[n] = "dna"; w[n++] = &l->dn_a; nm[n] = "dnout"; w[n++] = &l->dn_out; }
    if (i == c->ple_layer) { nm[n] = "plekey"; w[n++] = &l->ple_key; nm[n] = "plevalue"; w[n++] = &l->ple_value; }
    return n;
}
/* The device memory the chain takes after the tier sized its experts, which q38_tier_start
 * offers the placer as a component of its own ("chain") so a GGUF's arena leaves it: the
 * K/V, index-key and pooled-key mirrors at the context's capacity, the DeltaNet state and a
 * verify's copy of it, a chunk's scratch (q38cc_bufs) and the read-backs, and the matrices
 * the chain uploads itself (those the trunk's table does not offer: below its floor), a
 * quarter more. Measured on Qwen3.8 IQ2_XS: the arena took the whole budget and the chain
 * found no room for its 256 rows. */
/* an expert's place in the host pool and the stagings (gate, up and down each from a 256-byte
 * boundary), from the GGUF's tensors (before the tier knows them); 0: not a GGUF */
#define Q38CC_PAD(b) (((size_t)(b) + 255) / 256 * 256)
#define Q38CC_PROMPT_STAGE (512u << 20)   /* a prompt chunk's staging of a layer's pooled experts, bytes at most */
static size_t q38cc_gguf_stride(Model *m) {
    Cfg *c = &m->c; size_t st = 0;
    if (!g_q38_gguf || c->experts <= 0) return 0;
    for (int i = 0; i < c->layers; i++) {
        const gguf_tensor *t[3]; q38_gg_expert_tensors(m, i, t);
        size_t gb = (size_t)gq_row_bytes(t[0]->type, c->hidden) * c->inter, db = (size_t)gq_row_bytes(t[2]->type, c->inter) * c->hidden;
        size_t b = 2 * Q38CC_PAD(gb) + Q38CC_PAD(db);
        if (b > st) st = b;
    }
    return st;
}
static size_t q38cc_prompt_stage_bytes(Model *m) {
    size_t st = q38cc_gguf_stride(m);
    if (!st) return 0;
    size_t n = Q38CC_PROMPT_STAGE / st;
    if (n > (size_t)m->c.experts) n = (size_t)m->c.experts;
    return n * st;
}
static size_t q38cc_reserve_bytes(Model *m) {
    Cfg *c = &m->c; int L = c->layers, n_attn = 0;
    for (int i = 0; i < L; i++) n_attn += c->is_attn[i] != 0;
    int ctx = m->kv_cap > 0 ? m->kv_cap : m->max_t > 0 ? m->max_t : qwen38_max_ctx();   /* the Vulkan chain plans the same context */
    if (c->max_positions > 0 && ctx > c->max_positions) ctx = c->max_positions;
    size_t F = 4, cap = (size_t)(ctx > 0 ? ctx : 1), kvo = (size_t)c->kv_heads * c->head_dim, idx = (size_t)c->idx_dim;
    size_t nb = cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1, b = 0;
    b += (size_t)n_attn * ((2 * kvo + idx) * cap + idx * nb) * F;
    b += (size_t)(L - n_attn) * 2 * ((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim + (size_t)c->dn_conv_dim * (c->dn_convk - 1)) * F;
    size_t W = c->hc_width, H = c->hidden, QH = c->q_heads, D = c->head_dim, V = (size_t)c->dn_vheads * c->dn_vdim, E = c->experts;
    size_t SI = c->shared_inter, ipw = (size_t)(c->idx_qheads + 1) * c->idx_dim, Ep = (size_t)c->ngram_heads * c->ngram_head_dim;
    size_t row = 7 * W + c->hc_rank + 6 * H + 2 * (size_t)c->hc_count + 3 * QH * D + 2 * kvo + ipw + 2 * nb + (size_t)q38cc_selrow(c) +
                 2 * (size_t)c->dn_conv_dim + 2 * V + 2 * (size_t)c->dn_vheads + 2 * E + 3 * SI + Ep + (size_t)(c->rotary_dim > 0 ? c->rotary_dim : 2);
    b += (size_t)q38cc_rows_env() * row * F + 2 * (size_t)c->vocab * F;
    /* a verify's copies of the DeltaNet state and rings (q38cc_spec_slots), the routed experts'
     * scratch for a prompt chunk (q38cc_dm_begin: their hidden rows, the sum, the routes), and a
     * segment's streams of a prompt run layer by layer: past the estimate, 2048-row chunks left
     * the chain short of the copies, and every verify went to the CPU (Qwen3.8-Flash-Next IQ2_XS,
     * RTX PRO 4000: 3.21 GiB kept, 3.34-3.51 used, decode 43.6 -> 20.4 tok/s) */
    {
        size_t R = (size_t)q38cc_rows_env(), K = (size_t)c->topk;
        b += (size_t)Q38_SPEC_SNAPS * (L - n_attn) *
             ((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim + (size_t)c->dn_conv_dim * (c->dn_convk - 1)) * F;
        if (E > 0) b += (cc_moe_hid_floats((int)R, (int)K, (int)H, c->inter, (int)E) + R * H + 2 * (size_t)L * R * K) * F;
        if (q38cc_layer_major()) b += (size_t)(q38cc_lm_rows() > (int)R ? q38cc_lm_rows() : (int)R) * W * F;
    }
    for (int i = 0; i < L; i++) {   /* the matrices the trunk's table does not offer: the chain's own copies */
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n; k++) {
            int offered = 0;
            for (int j = 0; j < g_trunk_n && !offered; j++) offered = g_trunk[j].w == w[k] && !g_trunk[j].cpu_only;
            if (offered) continue;
            int gf = q38_cuda_gq_fmt(w[k]);
            b += gf ? (size_t)w[k]->rows * (((size_t)gq_row_bytes(gf - 100, w[k]->cols) + 3) / 4 * 4)
                    : (size_t)w[k]->rows * w[k]->cols * 4;   /* a small BF16/F32 matrix goes up as f32 */
        }
        b += H * F;   /* the shared expert's gate row */
    }
    /* a sixteenth more: the chain's buffers came to 3.92 GiB against 4.31 estimated (Qwen3.8 IQ2_XS, 65536 positions) */
    return b + b / 16 + q38cc_prompt_stage_bytes(m);
}
static void q38cc_free_own(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers;
    for (int i = 0; i < L; i++) {
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n; k++) if (w[k]->cc) { coli_cuda_tensor_free((ColiCudaTensor *)w[k]->cc); w[k]->cc = NULL; }
        if (ch->t_sg && ch->t_sg[i]) { coli_cuda_tensor_free(ch->t_sg[i]); ch->t_sg[i] = NULL; }
    }
    Q38Weight *tail[] = {&m->final_gr.down, &m->final_gr.up, &m->lm_head};
    for (size_t k = 0; k < 3; k++) if (tail[k]->cc) { coli_cuda_tensor_free((ColiCudaTensor *)tail[k]->cc); tail[k]->cc = NULL; }
    CcBuf ***arrs[] = {&ch->rec, &ch->ring, &ch->kc, &ch->vc, &ch->ik, &ch->pk};
    for (size_t a = 0; a < sizeof arrs / sizeof *arrs; a++) if (*arrs[a]) for (int i = 0; i < L; i++) { cc_free((*arrs[a])[i]); (*arrs[a])[i] = NULL; }
    for (int sl = 0; sl < Q38_SPEC_SNAPS; sl++) {
        if (ch->rec_snap[sl]) for (int i = 0; i < L; i++) { cc_free(ch->rec_snap[sl][i]); ch->rec_snap[sl][i] = NULL; }
        if (ch->ring_snap[sl]) for (int i = 0; i < L; i++) { cc_free(ch->ring_snap[sl][i]); ch->ring_snap[sl][i] = NULL; }
    }
    cc_free(ch->ple_ring); ch->ple_ring = NULL;
    cc_free(ch->prm); ch->prm = NULL;
}

/* The parameters the kernels read (norm weights, the DeltaNet constants, the PLE's), in
 * one buffer; the attention layers' order (n_attn). */
static int q38cc_arena(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers, W = c->hc_width;
    int VH = c->dn_vheads, CD = c->dn_conv_dim, CK = c->dn_convk;
    int ple = c->ple_layer >= 0 && c->ple_layer < L;
    size_t n = 0;
    ch->n_attn = 0;
    for (int i = 0; i < L; i++) {
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
    for (int i = 0; i < L; i++) {
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
    ch->prm = cc_buf(n * sizeof(float), CC_DEV);
    int ok = ch->prm && cc_begin() && cc_write(ch->prm, 0, a, n * sizeof(float)) && cc_submit(1);
    free(a);
    if (!ok) { cc_free(ch->prm); ch->prm = NULL; return 0; }
    ch->prm_floats = n;
    return 1;
}

/* The chain on the device, at startup after the tier placed the trunk and the CPU
 * quantized its int8 rows: every layer's matrices on one device (the tier's or the
 * chain's own), the gate rows, the state, the parameters. NULL = the chain cannot run
 * (the message says why). */
static Q38CChain *q38cc_setup(Model *m, int mux_slots) {
    Cfg *c = &m->c; int L = c->layers;
    if (m->cchain) return ((Q38CChain *)m->cchain)->ok ? (Q38CChain *)m->cchain : NULL;
    if (!cc_available()) { fprintf(stderr, "[chain] qwen38: the CUDA backend has no chain (an older DLL); the per-matrix path\n"); return NULL; }
    if (!qt_ready()) { fprintf(stderr, "[chain] qwen38: no CUDA expert tier; the chain stays off\n"); return NULL; }
    if (mux_slots > 1) { fprintf(stderr, "[chain] qwen38: KV_SLOTS > 1 keeps one state per conversation on the host; the chain stays off\n"); return NULL; }
    if (m->range_begin != 0 || m->range_end != L || !m->lm_head.rows || !q38cc_geometry_ok(c)) {
        fprintf(stderr, "[chain] qwen38: a model or geometry its kernels do not take; the per-matrix path\n");
        return NULL;
    }
    Q38CChain *ch = (Q38CChain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    m->cchain = ch;
    ch->dev = -1;
#define Q38CC_ARR(f, t) if (!(ch->f = (t *)calloc((size_t)L, sizeof(t)))) return NULL;
    Q38CC_ARR(o_an, size_t); Q38CC_ARR(o_mn, size_t); Q38CC_ARR(o_qn, size_t); Q38CC_ARR(o_kn, size_t);
    Q38CC_ARR(o_iqn, size_t); Q38CC_ARR(o_ikn, size_t); Q38CC_ARR(o_conv, size_t); Q38CC_ARR(o_dn, size_t);
    Q38CC_ARR(t_sg, ColiCudaTensor *); Q38CC_ARR(rec, CcBuf *); Q38CC_ARR(ring, CcBuf *); Q38CC_ARR(rec_snap[0], CcBuf *);
    Q38CC_ARR(ring_snap[0], CcBuf *); Q38CC_ARR(kc, CcBuf *); Q38CC_ARR(vc, CcBuf *); Q38CC_ARR(ik, CcBuf *); Q38CC_ARR(pk, CcBuf *);
    Q38CC_ARR(kv_valid, int); Q38CC_ARR(pk_valid, int); Q38CC_ARR(attn_ord, int);
#undef Q38CC_ARR
    ch->snap_slots = 1;
    /* the device: the card the tier placed the trunk on (the first placed matrix names it) */
    for (int i = 0; i < L && ch->dev < 0; i++) {
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n && ch->dev < 0; k++) if (w[k]->gpu) { int d = -1; if (qt_dense_tensor(w[k]->gpu - 1, &d)) ch->dev = d; }
    }
    if (ch->dev < 0 && m->lm_head.gpu) { int d = -1; if (qt_dense_tensor(m->lm_head.gpu - 1, &d)) ch->dev = d; }
    if (ch->dev < 0) { fprintf(stderr, "[chain] qwen38: the tier placed no trunk matrix (Q38_TRUNK_GPU); the chain stays off\n"); return NULL; }
    if (!cc_init(ch->dev)) { fprintf(stderr, "[chain] qwen38: the chain's context on device %d did not come up\n", ch->dev); return NULL; }
    size_t own = 0, tier = 0;
    for (int i = 0; i < L; i++) {
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n; k++) {
            if (!q38cc_t(ch, w[k])) {
                fprintf(stderr, "[chain] qwen38: %s of layer %d did not reach device %d; the chain stays off\n", nm[k], i, ch->dev);
                q38cc_free_own(ch, m);
                return NULL;
            }
            if (!w[k]->cc) tier += (size_t)w[k]->rows * w[k]->cols;
            else own += (size_t)w[k]->rows * w[k]->cols * (w[k]->q8 ? 1 : 4);
        }
        if (!(ch->t_sg[i] = q38cc_f32_rows(ch->dev, m->L[i].sh_gate, c->hidden, 1))) {
            fprintf(stderr, "[chain] qwen38: the shared expert's gate row of layer %d did not reach the device; the chain stays off\n", i);
            q38cc_free_own(ch, m);
            return NULL;
        }
        own += (size_t)c->hidden * 4;
        if (!c->is_attn[i]) {
            size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
            if (!(ch->rec[i] = cc_buf(nr * 4, CC_DEV)) || !(ch->ring[i] = cc_buf(nc * 4, CC_DEV)) ||
                !(ch->rec_snap[0][i] = cc_buf(nr * 4, CC_DEV)) || !(ch->ring_snap[0][i] = cc_buf(nc * 4, CC_DEV))) {
                fprintf(stderr, "[chain] qwen38: device memory for layer %d's DeltaNet state refused; the chain stays off\n", i);
                q38cc_free_own(ch, m);
                return NULL;
            }
        }
        if (i == c->ple_layer) {
            if (!(ch->ple_ring = cc_buf(2 * q38cc_ple_cells(c) * 4, CC_DEV))) { q38cc_free_own(ch, m); return NULL; }
            ch->ple_nslots = 2; ch->ple_slot[0] = 0; ch->ple_slot[1] = 1;
        }
    }
    Q38Weight *tail[] = {&m->final_gr.down, &m->final_gr.up, &m->lm_head};
    static const char *tnm[] = {"the final mixer's down", "the final mixer's up", "lm_head"};
    for (int k = 0; k < 3; k++) if (!q38cc_t(ch, tail[k])) {
        fprintf(stderr, "[chain] qwen38: %s did not reach device %d; the chain stays off\n", tnm[k], ch->dev);
        q38cc_free_own(ch, m);
        return NULL;
    }
    if (!q38cc_arena(ch, m)) {
        fprintf(stderr, "[chain] qwen38: device memory for the chain's parameters refused; the chain stays off\n");
        q38cc_free_own(ch, m);
        return NULL;
    }
    ch->nl = L; ch->head = 1;
    ch->dn_where = Q38CC_HOST;
    { const char *e = getenv("COLI_CUDA_CHAIN_FAULT"); ch->fault_at = e && *e ? strtoull(e, NULL, 10) : 0; }
    ch->ok = 1;
    ch->dm_on = q38cc_dm_able(ch, m);
    if (ch->dm_on) fprintf(stderr, "[chain] qwen38: a decode step's routed experts in the frame (the tier's residents by address, "
                                   "format %d%s, group %d; COLI_CUDA_CHAIN_MOE=0: the host's step)\n", ch->dm_fmt,
                                   ch->dm_fmt >= 100 ? " (a GGUF's blocks, by layer)" : "", ch->dm_gs);
    fprintf(stderr, "[chain] qwen38: %d layers on CUDA device %d (%d QSA), the final mixer and lm_head too; %.2f GiB of the tier's "
                    "trunk, %.1f MiB uploaded by the chain, %.1f MiB of parameters\n",
            L, ch->dev, ch->n_attn, tier / 1073741824.0, own / 1048576.0, ch->prm_floats * 4 / 1048576.0);
    return ch;
}

/* the scratch buffers for `rows` rows (kvd's stride is ch->rows') */
static int q38cc_bufs(Q38CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    int QH = c->q_heads, D = c->head_dim, kvo = c->kv_heads * D, ipw = (c->idx_qheads + 1) * c->idx_dim;
    int V = c->dn_vheads * c->dn_vdim, E = c->experts, SI = c->shared_inter;
    int Ep = c->ngram_heads * c->ngram_head_dim; if (Ep < 1) Ep = 1;
    size_t r = (size_t)rows, F = 4;
    int nbmax = m->kv_cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    int ok = cc_reserve(&ch->hyper, r * W * F, CC_DEV) && cc_reserve(&ch->hn, r * W * F, CC_DEV) && cc_reserve(&ch->low, r * R * F, CC_DEV) &&
             cc_reserve(&ch->mix, r * W * F, CC_DEV) && cc_reserve(&ch->mixed, r * H * F, CC_DEV) && cc_reserve(&ch->inj_a, r * C * F, CC_DEV) &&
             cc_reserve(&ch->inj_m, r * C * F, CC_DEV) && cc_reserve(&ch->blk, r * H * F, CC_DEV) && cc_reserve(&ch->q, r * QH * 2 * D * F, CC_DEV) &&
             cc_reserve(&ch->k, r * kvo * F, CC_DEV) && cc_reserve(&ch->v, r * kvo * F, CC_DEV) && cc_reserve(&ch->ip, r * ipw * F, CC_DEV) &&
             cc_reserve(&ch->ctx, r * QH * D * F, CC_DEV) && cc_reserve(&ch->qsc, r * 2 * (size_t)nbmax * F, CC_DEV) &&
             cc_reserve(&ch->sel, r * q38cc_selrow(c) * F, CC_DEV) && cc_reserve(&ch->qkv, r * c->dn_conv_dim * F, CC_DEV) &&
             cc_reserve(&ch->z, r * V * F, CC_DEV) && cc_reserve(&ch->ab, r * 2 * c->dn_vheads * F, CC_DEV) &&
             cc_reserve(&ch->cv, r * c->dn_conv_dim * F, CC_DEV) && cc_reserve(&ch->dny, r * V * F, CC_DEV) &&
             cc_reserve(&ch->lg, r * E * F, CC_DEV) && cc_reserve(&ch->gs, r * SI * F, CC_DEV) && cc_reserve(&ch->us, r * SI * F, CC_DEV) &&
             cc_reserve(&ch->hs, r * SI * F, CC_DEV) && cc_reserve(&ch->ds, r * H * F, CC_DEV) && cc_reserve(&ch->sgd, r * F, CC_DEV) &&
             cc_reserve(&ch->keys, r * W * F, CC_DEV) && cc_reserve(&ch->val, r * H * F, CC_DEV) &&
             cc_reserve(&ch->gated, r * W * F, CC_DEV) && cc_reserve(&ch->normv, r * W * F, CC_DEV) &&
             cc_reserve(&ch->mixd, r * H * F, CC_DOWN) && cc_reserve(&ch->lgd, r * E * F, CC_DOWN) &&
             cc_reserve(&ch->kvd, (size_t)(ch->n_attn ? ch->n_attn : 1) * q38cc_kv_stride(ch, c) * F, CC_DOWN) &&
             cc_reserve(&ch->outd, 2 * (size_t)c->vocab * F, CC_DOWN) && cc_reserve(&ch->routed, r * H * F, CC_UP) &&
             cc_reserve(&ch->emb, r * Ep * F, CC_UP) && cc_reserve(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_UP) &&
             cc_reserve(&ch->pcs, (size_t)nbmax * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_UP);
    /* a verify's PLE segments (q38cc_ple) here, not in the frame: growing a buffer there waits
     * for the stream, which a hybrid layer before it holds until the host serves it */
    if (ok && c->ple_layer >= 0)
        ok = cc_reserve(&ch->gseg, r * W * F, CC_DEV) && cc_reserve(&ch->nseg, r * W * F, CC_DEV);
    return ok;
}
/* the K/V, index-key and pooled-key mirrors at the host's capacity */
static int q38cc_mirror(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int kvo = c->kv_heads * c->head_dim;
    int nbmax = m->kv_cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    if (ch->cap != m->kv_cap) {
        for (int i = 0; i < c->layers; i++) {
            if (!c->is_attn[i]) continue;
            cc_free(ch->kc[i]); cc_free(ch->vc[i]); cc_free(ch->ik[i]); cc_free(ch->pk[i]);
            ch->kv_valid[i] = ch->pk_valid[i] = 0;
            ch->kc[i] = cc_buf((size_t)kvo * m->kv_cap * 4, CC_DEV);
            ch->vc[i] = cc_buf((size_t)kvo * m->kv_cap * 4, CC_DEV);
            ch->ik[i] = cc_buf((size_t)c->idx_dim * m->kv_cap * 4, CC_DEV);
            ch->pk[i] = cc_buf((size_t)c->idx_dim * nbmax * 4, CC_DEV);
            if (!ch->kc[i] || !ch->vc[i] || !ch->ik[i] || !ch->pk[i]) { ch->cap = 0; return 0; }
        }
        ch->cap = m->kv_cap;
    }
    return 1;
}
static int q38cc_scratch(Q38CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int H = c->hidden;
    int Ep = c->ngram_heads * c->ngram_head_dim; if (Ep < 1) Ep = 1;
    size_t r = (size_t)rows;
    int need_rows = ch->rows < rows;
    if (need_rows) ch->rows = rows;   /* kvd's stride follows */
    if (!q38cc_bufs(ch, m, rows)) return 0;
    if (need_rows) {
        float *hr = realloc(ch->host_routed, r * H * sizeof(float)), *he;
        if (!hr) return 0;
        ch->host_routed = hr;
        he = realloc(ch->host_emb, r * Ep * sizeof(float));
        if (!he) return 0;
        ch->host_emb = he;
    }
    return 1;
}

/* ---- state between the host and the device ------------------------------------ */
static void q38_layer_forward(Model *m,int i,float *hyper,const int *ids,int S,int pos_base,
                              float *mixed,float *inject,float *block);
static void q38_embed_row(Model *m,int id,int abs_pos,float *out);
/* The device was lost with the newest DeltaNet state, conv rings and PLE ring on it.
 * The host's K/V/index rows are canonical, those are not: rebuild them on the CPU by
 * running the `upto` positions the prefix record names (the PLE n-gram history replayed
 * with them), then leave the chain off. */
static void q38cc_recover(Model *m, int upto) {
    Q38CChain *ch = q38cc_of(m);
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, nl = c->layers;
    int ple = c->ple_layer >= 0 && c->ple_layer < nl;
    g_cuda_chain = 0;
    if (ch) { ch->failed = 1; ch->dn_where = Q38CC_HOST; ch->host_zero = 0; ch->snap_valid = 0; }
    for (int i = 0; i < nl; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
    if (ple && m->PLE_conv_state) memset(m->PLE_conv_state, 0, q38cc_ple_cells(c) * sizeof(float));
    if (ple && m->ple_history) m->ple_history_len = 0;
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        q38cc_fatal("the device was lost with a recurrent state its token ids do not describe (an image)");
    fprintf(stderr, "[chain] qwen38: the device was lost; rebuilding the state of %d positions on the CPU, "
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
    float *pref = m->ple_pref; int pref_rows = m->ple_pref_rows, snap = m->snap_rows, rowwise = g_q38_rowwise;
    m->ple_pref = NULL; m->ple_pref_rows = 0; m->snap_rows = 0; g_q38_rowwise = 0;
    for (int i = 0; i < nl; i++) q38_layer_forward(m, i, hyper, ids, upto, 0, mixed, inject, block);
    free(m->ple_pref);
    m->ple_pref = pref; m->ple_pref_rows = pref_rows; m->snap_rows = snap; g_q38_rowwise = rowwise;
    free(ids); free(hyper); free(mixed); free(inject); free(block);
}
static int q38cc_sync_one(Model *m, Q38CChain *ch) {
    if (!ch || !ch->ok || ch->dn_where != Q38CC_DEV) return 1;
    Cfg *c = &m->c;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    int ok = 1;
    for (int i = 0; i < c->layers && ok; i++) {
        if (c->is_attn[i]) continue;
        ok = cc_read(ch->rec[i], 0, m->DN_rec[i], nr * 4) && cc_read(ch->ring[i], 0, m->DN_conv[i], nc * 4);
    }
    if (ok && ch->ple_ring && m->PLE_conv_state)
        ok = cc_read(ch->ple_ring, (size_t)ch->ple_slot[0] * q38cc_ple_cells(c), m->PLE_conv_state, q38cc_ple_cells(c) * 4);
    if (ok) ch->dn_where = Q38CC_BOTH;
    return ok;
}
static void q38cc_sync_host(Model *m) {
    if (!q38cc_sync_one(m, q38cc_of(m))) q38cc_recover(m, m->kv_len);
}
static void q38cc_host_wrote(Model *m, int zero) {
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok) return;
    ch->dn_where = Q38CC_HOST; ch->host_zero = zero; ch->snap_valid = 0;
}
static void q38cc_cpu_step(Model *m, int pos_base) {
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok) return;
    if (!q38cc_sync_one(m, ch)) { q38cc_recover(m, m->kv_len); return; }
    ch->dn_where = Q38CC_HOST; ch->host_zero = 0; ch->snap_valid = 0;
    int R = m->c.idx_ratio > 0 ? m->c.idx_ratio : 1;
    for (int i = 0; i < m->c.layers; i++) {
        if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
        if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;
    }
}
static void q38cc_rollback(Model *m, int slot, int len) {
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok) return;
    int R = m->c.idx_ratio > 0 ? m->c.idx_ratio : 1;
    for (int i = 0; i < m->c.layers; i++) {
        if (ch->kv_valid[i] > len) ch->kv_valid[i] = len;
        if (ch->pk_valid[i] > len / R) ch->pk_valid[i] = len / R;
    }
    if (slot < 0 || slot >= ch->snap_valid || ch->dn_where != Q38CC_DEV) { ch->snap_valid = 0; return; }
    for (int i = 0; i < m->c.layers; i++) {
        if (m->c.is_attn[i]) continue;
        CcBuf *t = ch->rec[i]; ch->rec[i] = ch->rec_snap[slot][i]; ch->rec_snap[slot][i] = t;
        t = ch->ring[i]; ch->ring[i] = ch->ring_snap[slot][i]; ch->ring_snap[slot][i] = t;
    }
    if (ch->ple_ring) { int t = ch->ple_slot[0]; ch->ple_slot[0] = ch->ple_slot[1 + slot]; ch->ple_slot[1 + slot] = t; }
    ch->snap_valid = 0;
}
static int q38cc_spec_slots(Q38CChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int L = c->layers;
    if (rows > Q38_SPEC_SNAPS) rows = Q38_SPEC_SNAPS;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int sl = ch->snap_slots; sl < rows; sl++) {
        if (!ch->rec_snap[sl] && !(ch->rec_snap[sl] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) return 0;
        if (!ch->ring_snap[sl] && !(ch->ring_snap[sl] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) return 0;
        for (int i = 0; i < L; i++) {
            if (c->is_attn[i]) continue;
            if (!ch->rec_snap[sl][i] && !(ch->rec_snap[sl][i] = cc_buf(nr * 4, CC_DEV))) return 0;
            if (!ch->ring_snap[sl][i] && !(ch->ring_snap[sl][i] = cc_buf(nc * 4, CC_DEV))) return 0;
        }
        ch->snap_slots = sl + 1;
    }
    if (ch->ple_ring && ch->ple_nslots < rows + 1) {
        size_t pc = q38cc_ple_cells(c);
        CcBuf *nb = cc_buf((size_t)Q38_SPEC_ROWS * pc * 4, CC_DEV);
        if (!nb) return 0;
        if (!cc_begin() || !cc_copy(nb, 0, ch->ple_ring, (size_t)ch->ple_slot[0] * pc, pc) || !cc_submit(1)) { cc_free(nb); return 0; }
        cc_free(ch->ple_ring);
        ch->ple_ring = nb; ch->ple_nslots = Q38_SPEC_ROWS;
        for (int r = 0; r < Q38_SPEC_ROWS; r++) ch->ple_slot[r] = r;
        ch->snap_valid = 0;
    }
    return 1;
}
static int q38cc_push_state(Q38CChain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->dn_where == Q38CC_HOST) {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        for (int i = 0; i < c->layers && ok; i++) {
            if (c->is_attn[i]) continue;
            if (ch->host_zero) ok = cc_zero(ch->rec[i], 0, nr) && cc_zero(ch->ring[i], 0, nc);
            else ok = cc_write(ch->rec[i], 0, m->DN_rec[i], nr * 4) && cc_write(ch->ring[i], 0, m->DN_conv[i], nc * 4);
        }
        if (ok && ch->ple_ring && m->PLE_conv_state) {
            size_t pc = q38cc_ple_cells(c), off = (size_t)ch->ple_slot[0] * pc;
            ok = ch->host_zero ? cc_zero(ch->ple_ring, off, pc) : cc_write(ch->ple_ring, off, m->PLE_conv_state, pc * 4);
        }
        ch->dn_where = Q38CC_BOTH;
    }
    int D = c->head_dim, KVH = c->kv_heads, ID = c->idx_dim, R = c->idx_ratio;
    for (int i = 0; i < c->layers && ok; i++) {
        if (!c->is_attn[i]) continue;
        if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;   /* blocks this step rewrites */
        if (ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < KVH && ok; h++) {
            size_t src = ((size_t)h * m->kv_cap + t0) * D, dst = ((size_t)h * ch->cap + t0) * D;
            ok = cc_write(ch->kc[i], dst, m->K[i] + src, (size_t)n * D * 4) && cc_write(ch->vc[i], dst, m->V[i] + src, (size_t)n * D * 4);
        }
        ok = ok && cc_write(ch->ik[i], (size_t)t0 * ID, m->IK[i] + (size_t)t0 * ID, (size_t)n * ID * 4);
        ch->kv_valid[i] = pos_base;
        if (ch->pk_valid[i] > t0 / R) ch->pk_valid[i] = t0 / R;
    }
    return ok;
}

/* ---- pieces ---------------------------------------------------------------------- */
static int q38cc_gr_read(Q38CChain *ch, Model *m, GatedResidual *g, size_t norm_off, int n, CcBuf *inj) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    CcNorm np = {n * C, H, C, 0, W, H, 0, W, H, (int)norm_off, C, CC_NORM_ADD1, c->eps, 1.f};
    CcEw lo = {CC_EW_HC_LOW, n * R, R, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    CcEw mx = {CC_EW_HC_MIX, n * H, H, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    int ok = cc_norm(ch->hyper, ch->prm, ch->hn, &np) &&
             cc_matmul(q38cc_t(ch, &g->down), ch->hn, 0, ch->low, 0, n) && cc_ew(ch->low, ch->low, NULL, NULL, NULL, &lo) &&
             cc_matmul(q38cc_t(ch, &g->up), ch->low, 0, ch->mix, 0, n) && cc_ew(ch->mixed, ch->mix, ch->hn, NULL, NULL, &mx);
    if (ok && inj) {
        CcEw ij = {CC_EW_HC_INJ, n * C, C, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
        ok = cc_matmul(q38cc_t(ch, &g->inject), ch->hn, 0, inj, 0, n) && cc_ew(inj, inj, NULL, NULL, NULL, &ij);
    }
    return ok;
}
static int q38cc_gr_apply(Q38CChain *ch, Model *m, CcBuf *inj, int n) {
    Cfg *c = &m->c;
    CcEw ap = {CC_EW_HC_APPLY, n * c->hc_width, c->hidden, c->hc_count, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_ew(ch->hyper, inj, ch->blk, NULL, NULL, &ap);
}
static int q38cc_attention(Q38CChain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int QH = c->q_heads, KVH = c->kv_heads, D = c->head_dim, kvo = KVH * D, qo = QH * 2 * D, half = c->rotary_dim / 2;
    int IQ = c->idx_qheads, ID = c->idx_dim, R = c->idx_ratio, ipw = (IQ + 1) * ID;
    int ok = cc_matmul(q38cc_t(ch, &l->q), ch->mixed, 0, ch->q, 0, n) && cc_matmul(q38cc_t(ch, &l->k), ch->mixed, 0, ch->k, 0, n) &&
             cc_matmul(q38cc_t(ch, &l->v), ch->mixed, 0, ch->v, 0, n) && cc_matmul(q38cc_t(ch, &l->idx_qk), ch->mixed, 0, ch->ip, 0, n);
    CcNorm kn = {n * KVH, D, KVH, 0, kvo, D, 0, kvo, D, (int)ch->o_kn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
    CcNorm qn = {n * QH, D, QH, 0, qo, 2 * D, 0, qo, 2 * D, (int)ch->o_qn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
    CcNorm in = {n * IQ, ID, IQ, 0, ipw, ID, 0, ipw, ID, (int)ch->o_iqn[i], 0, CC_NORM_ADD1, c->eps, 1.f};
    ok = ok && cc_norm(ch->k, ch->prm, ch->k, &kn) && cc_norm(ch->q, ch->prm, ch->q, &qn) && cc_norm(ch->ip, ch->prm, ch->ip, &in);
    if (ok && half) {
        CcRope rk = {n * KVH, KVH, 0, kvo, D, half, 0, 2 * half}, rq = {n * QH, QH, 0, qo, 2 * D, half, 0, 2 * half};
        CcRope ri = {n * IQ, IQ, 0, ipw, ID, half, 0, 2 * half};
        ok = cc_rope(ch->k, ch->cs, &rk) && cc_rope(ch->q, ch->cs, &rq) && cc_rope(ch->ip, ch->cs, &ri);
    }
    if (!ok) return 0;
    /* the rows into the device caches and the host's */
    CcRegion *rg = malloc(sizeof *rg * (size_t)n * (KVH > 1 ? KVH : 1));
    CcRegion *ri = malloc(sizeof *ri * (size_t)n);
    if (!rg || !ri) { free(rg); free(ri); return 0; }
    for (int s = 0; s < n; s++) {
        for (int h = 0; h < KVH; h++)
            rg[s * KVH + h] = (CcRegion){((size_t)h * ch->cap + pb + s) * D, (size_t)s * kvo + (size_t)h * D, (size_t)D};
        ri[s] = (CcRegion){(size_t)(pb + s) * ID, (size_t)s * ipw + (size_t)IQ * ID, (size_t)ID};
    }
    size_t ko = (size_t)ch->attn_ord[i] * q38cc_kv_stride(ch, c);
    ok = cc_copy_regions(ch->kc[i], ch->k, rg, n * KVH) && cc_copy_regions(ch->vc[i], ch->v, rg, n * KVH) &&
         cc_copy_regions(ch->ik[i], ch->ip, ri, n) &&
         cc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) &&
         cc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    for (int s = 0; s < n && ok; s++)   /* index keys back, compact */
        ok = cc_copy(ch->kvd, ko + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, ch->ip, (size_t)s * ipw + (size_t)IQ * ID, ID);
    free(rg); free(ri);
    /* the pooled keys of the blocks this step completes (and any behind the watermark) */
    int nb = (pb + n) / R, b0 = ch->pk_valid[i];
    if (ok && nb > b0) {
        float *pc = (float *)cc_ptr(ch->pcs);
        if (half) for (int b = b0; b < nb; b++) for (int j = 0; j < half; j++) {
            float ang = (float)(b * R) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
            pc[((b - b0) * half + j) * 2] = cosf(ang); pc[((b - b0) * half + j) * 2 + 1] = sinf(ang);
        }
        CcQsa pp = {0, ID, R, b0, half, 0, 0, 0, 0, 0, 0, 0, 0, c->eps, 0, (int)ch->o_ikn[i], 0, nb - b0};
        ok = cc_qsa(ch->ik[i], ch->prm, ch->pk[i], ch->pcs, NULL, NULL, &pp);
        ch->pk_valid[i] = nb;
    }
    int nbmax = (int)(cc_bytes(ch->qsc) / 4 / (2 * (size_t)ch->rows));
    CcQsa ps = {1, ID, R, 0, 0, n, pb, c->idx_budget, IQ, 0, ipw, nbmax, q38cc_selrow(c), c->eps, 0, 0, 0, 0};
    CcAttn at = {n, QH, KVH, D, pb, ch->cap, 0, qo, 2 * D, D, qo, 2 * D, 1, 0, QH * D, 0, q38cc_selrow(c),
                 1.f / sqrtf((float)D), 0, 0};
    return ok && cc_qsa(ch->ip, NULL, ch->pk[i], NULL, ch->qsc, ch->sel, &ps) &&
           cc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, ch->sel, &at) &&
           cc_matmul(q38cc_t(ch, &l->o), ch->ctx, 0, ch->blk, 0, n);
}
/* ns: the chunk's rows whose state a verify copies (rows 0..ns-1, into slots c0..): the
 * convolution and the recurrence split into one dispatch per copy, each ending on its
 * row (the same bits). */
static int q38cc_deltanet(Q38CChain *ch, Model *m, Layer *l, int i, int n, int c0, int ns) {
    Cfg *c = &m->c;
    int VH = c->dn_vheads, CD = c->dn_conv_dim, V = VH * c->dn_vdim;
    int ok = cc_matmul(q38cc_t(ch, &l->dn_qkv), ch->mixed, 0, ch->qkv, 0, n) && cc_matmul(q38cc_t(ch, &l->dn_z), ch->mixed, 0, ch->z, 0, n) &&
             cc_matmul(q38cc_t(ch, &l->dn_b), ch->mixed, 0, ch->ab, 0, n) &&
             cc_matmul(q38cc_t(ch, &l->dn_a), ch->mixed, 0, ch->ab, (size_t)ch->rows * VH, n);
    int segs = ns > 1 ? ns : 1;
    for (int r = 0; r < segs && ok; r++) {
        int s0 = ns > 1 ? r : 0, len = ns > 1 && r < ns - 1 ? 1 : n - s0, snap_row = ns ? 0 : -1, slot = ns ? c0 + r : 0;
        CcDnConv cp = {len, CD, c->dn_convk, s0 * CD, CD, s0 * CD, CD, snap_row, 1, (int)ch->o_conv[i], 0, 0};
        CcDnRec rp = {len, VH, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, s0 * CD, CD, s0 * VH, VH,
                      ch->rows * VH + s0 * VH, VH, s0 * V, V, s0 * V, V, snap_row,
                      1 | (g_q38_a_direct ? 2 : 0) | (g_q38_dn_tiled ? 4 : 0), c->eps,   /* a GGUF's layout (qwen38_core.h) */
                      1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
        ok = cc_dnconv(ch->qkv, ch->prm, ch->ring[i], ch->cv, ch->ring_snap[slot][i], &cp) &&
             cc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, ch->rec[i], ch->prm, ch->dny, ch->rec_snap[slot][i], &rp);
    }
    return ok && cc_matmul(q38cc_t(ch, &l->dn_out), ch->dny, 0, ch->blk, 0, n);
}
/* the n-gram rows of n tokens, in order, as q38_ple fills them (history and the verify's snapshot) */
static void q38cc_ple_rows(Model *m, const int *ids, int c0, int n, float *emb) {
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
static int q38cc_ple(Q38CChain *ch, Model *m, int n, int c0, int ns) {
    Cfg *c = &m->c; Layer *l = &m->L[c->ple_layer];
    int W = c->hc_width, H = c->hidden, E = c->ngram_heads * c->ngram_head_dim;
    size_t pc = q38cc_ple_cells(c);
    int ok = cc_write(ch->emb, 0, ch->host_emb, (size_t)n * E * 4) &&
             cc_matmul(q38cc_t(ch, &l->ple_key), ch->emb, 0, ch->keys, 0, n) &&
             cc_matmul(q38cc_t(ch, &l->ple_value), ch->emb, 0, ch->val, 0, n);
    CcPle g = {0, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, 0, 0, -1, 0, c->eps, (int)ch->o_ple, 0, 0};
    ok = ok && cc_ple(ch->keys, ch->hyper, ch->val, ch->prm, ch->gated, ch->normv, NULL, NULL, &g);
    if (ns <= 1) {
        int snap = ns ? ch->ple_slot[1 + c0] : ch->ple_slot[1];
        CcPle cv = {1, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, 0, 0, ns ? 0 : -1,
                    (int)((size_t)snap * pc), c->eps, 0, (int)ch->o_pconv, (int)((size_t)ch->ple_slot[0] * pc)};
        return ok && cc_ple(NULL, ch->hyper, NULL, NULL, ch->gated, ch->normv, ch->prm, ch->ple_ring, &cv);
    }
    ok = ok && cc_reserve(&ch->gseg, (size_t)n * W * 4, CC_DEV) && cc_reserve(&ch->nseg, (size_t)n * W * 4, CC_DEV);
    for (int r = 0; r < ns && ok; r++) {
        int len = r < ns - 1 ? 1 : n - r;
        CcBuf *gb = ch->gated, *nb = ch->normv;
        if (r) {
            ok = cc_copy(ch->gseg, 0, ch->gated, (size_t)r * W, (size_t)len * W) &&
                 cc_copy(ch->nseg, 0, ch->normv, (size_t)r * W, (size_t)len * W);
            gb = ch->gseg; nb = ch->nseg;
        }
        CcPle cv = {1, len, c->hc_count, H, c->ple_convk, c->ngram_size, 0, r * W, 0, 0,
                    (int)((size_t)ch->ple_slot[1 + c0 + r] * pc), c->eps, 0, (int)ch->o_pconv,
                    (int)((size_t)ch->ple_slot[0] * pc)};
        ok = ok && cc_ple(NULL, ch->hyper, NULL, NULL, gb, nb, ch->prm, ch->ple_ring, &cv);
    }
    return ok;
}
static int q38cc_shared(Q38CChain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c; int SI = c->shared_inter;
    CcEw sw = {CC_EW_SWIGLU, n * SI, SI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_matmul(q38cc_t(ch, &l->sh_g), ch->mixed, 0, ch->gs, 0, n) && cc_matmul(q38cc_t(ch, &l->sh_u), ch->mixed, 0, ch->us, 0, n) &&
           cc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &sw) && cc_matmul(q38cc_t(ch, &l->sh_d), ch->hs, 0, ch->ds, 0, n) &&
           cc_matmul(ch->t_sg[i], ch->mixed, 0, ch->sgd, 0, n);
}
/* block = routed + sigmoid(gate) * shared (q38_moe's out), then hyper += inject x block */
static int q38cc_moe_apply(Q38CChain *ch, Model *m, int n, CcBuf *routed) {
    Cfg *c = &m->c;
    CcEw cb = {CC_EW_COMBINE, n * c->hidden, c->hidden, 1, 1 | 2 | 4 | 8, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_ew(ch->blk, NULL, routed, ch->ds, ch->sgd, &cb) && q38cc_gr_apply(ch, m, ch->inj_m, n);
}
/* Layer i's new K/V and index-key rows (an attention layer's) into the host's caches once
 * their frame is done */
static void q38cc_kv_down(Q38CChain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c; int kvo = c->kv_heads * c->head_dim, ID = c->idx_dim;
    if (!c->is_attn[i]) return;
    const float *kv = (const float *)cc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * q38cc_kv_stride(ch, c);
    for (int s = 0; s < n; s++) {
        for (int h = 0; h < c->kv_heads; h++) {
            size_t dst = ((size_t)h * m->kv_cap + pb + s) * c->head_dim, src = (size_t)s * kvo + (size_t)h * c->head_dim;
            memcpy(m->K[i] + dst, kv + src, c->head_dim * sizeof(float));
            memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->head_dim * sizeof(float));
        }
        memcpy(m->IK[i] + (size_t)(pb + s) * ID, kv + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, ID * sizeof(float));
    }
}

/* ---- the routed experts in the frame (cuda_chain.h: cc_route, cc_moe) -----------------
 * qwen36_cuda_chain.h's, for qwen38: a step of up to Q38CC_DM_ROWS rows (decode, a verify)
 * routes on the device (q38_moe_decode_grouped's softmax, top-k with the lower expert on a
 * tie, gates over the top-k's sum with norm_topk or over every expert's) and runs the
 * tier's residents there by address, from a table built again when the tier's epoch moved
 * (the tier held for the forward: no swap writes into a resident while it runs). A layer
 * whose experts the table holds has no host step. A hybrid layer signals the host once its
 * routes and rows are down; the host computes the slots the table lacks through
 * q38_moe_decode_grouped (the RAM cache, the disk, the tier's notes) while the device
 * computes the rest, and answers in an UP buffer the device waits for before the sum (the
 * host's slots, then the device's in rank order: the CPU tier's order). The host serves a
 * hybrid layer after recording the next one. The books (the history, HITS, the tier's heat
 * and LFRU tick) and the K/V rows follow from the ids after the frame. Off with
 * ROUTE_TRACE, the expert statistics dumps, a tier on more than one device, a format the
 * kernels do not take, or COLI_CUDA_CHAIN_MOE=0. */
#define Q38CC_DM_ROWS 8
/* ---- the host pool -----------------------------------------------------------------------
 * The Vulkan tier's host pool, for this chain: the experts the card does not hold, in
 * page-locked RAM mapped into the device (cc_buf CC_DOWN: cached, the CPU reads it at its
 * memory's speed), as the file has them (a GGUF's rows, whole words). A decode step's
 * pooled experts are the CPU's (pool_cpu): their rows eight at a time where they are
 * (q38cc_pool_pairs, as vk_tier.c's hybrid steps), the device computing its residents meanwhile,
 * from a decode table without them (dm_tabd). A prompt chunk's table points at them, so a
 * layer whose experts are all on the card or here has no host step: the GEMV reads a pooled
 * expert's matrices over PCIe once. COLI_CUDA_CHAIN_POOL_READ=1 has the device read a decode
 * step's too, as before. Filled once after the tier's warm start, the history's
 * most used first. Its size: the RAM free less 4 GiB and less what the RAM cache may still
 * grow by (the experts the card leaves, the CPU's step computing nothing else), at most the
 * experts the card leaves. COLI_CUDA_CHAIN_POOL=0 turns it off,
 * COLI_CUDA_CHAIN_POOL_GB sets its size. */
#define Q38CC_STAGE 16   /* the staging's places: a step's distinct pooled experts copied to the device first */
static int q38cc_pool_types(Model *m, int i, int64_t *gb, int64_t *db) {
    Cfg *c = &m->c; int fg, fd;
    if (!qt_chain_layer_format(i, &fg, &fd) || fg < 100 || fd < 100) return 0;
    int64_t rg = gq_row_bytes(fg - 100, c->hidden), rd = gq_row_bytes(fd - 100, c->inter);
    if (rg <= 0 || rd <= 0 || rg % 4 || rd % 4) return 0;   /* rows not whole words: not the device's layout */
    *gb = rg * c->inter; *db = rd * c->hidden;
    return 1;
}
/* layer i's expert in pool slot k, as a table entry */
static int q38cc_pool_entry(Q38CChain *ch, Model *m, int i, int k, CcExpert *x) {
    int64_t gb, db;
    if (k < 0 || k >= ch->pool_n || !q38cc_pool_types(m, i, &gb, &db)) return 0;
    const char *base = (const char *)cc_dev_addr(ch->pool_buf[k / ch->pool_per]);
    if (!base) return 0;
    const char *g = base + (size_t)(k % ch->pool_per) * ch->pool_stride;
    memset(x, 0, sizeof *x);
    x->g = g; x->u = g + Q38CC_PAD(gb); x->d = g + 2 * Q38CC_PAD(gb); x->host = 1;
    return 1;
}
/* ---- adaptive promotion: the chain moves experts between the card and the host pool ------
 * With the host pool holding every expert, the card's too, the tier's arena is the chain's
 * (qt_chain_own). After a frame, the routed experts the pool served promote into a free slot
 * or the coldest resident's (the least of 64 drawn) by this conversation's routes (halved
 * every Q38CC_DECAY rows), then the history's, up to the rate a forward. A victim reads from
 * the pool from the next frame on (its table entry rewritten before it); the newcomer's bytes
 * go over the bus on the copy stream (cc_host_up) while that frame runs, into a slot no entry
 * points to, and its entry turns to the slot once they landed. The warm start's set comes
 * from the history: a conversation on another topic found 69.5% of its experts there
 * (Qwen3.8-Flash-Next IQ2_XS on a 24 GB card, a story; simulated over its routes, 32 a
 * forward: 95%). COLI_CUDA_CHAIN_PROMOTE=0 keeps the warm start's set, =n sets the rate.
 * The brake, like qwen36's (the same knob) but over every frame, a prompt chunk weighing its
 * rows (qwen36's window leaves those out): where the last Q38CC_PROMO_WIN frames found at least
 * COLI_CUDA_CHAIN_PROMOTE_HITS percent (90) of their routes on the card, the frame promotes
 * nothing -- a conversation the history fits pays for every swap (the newcomer's bytes on the
 * bus, a resident the history still routes to back to the pool) and gains little. 0 brakes
 * never. UD-Q2_K_XL on an RTX PRO 4000, 128 tokens, 32 a forward without the brake against 128
 * with it: code after a prose history 30.5 -> 42.2 tok/s (58% of its routes on the card),
 * with the MTP head 30.7 -> 39.9; the prose that wrote the history 52.1 -> 52.7 (98%, the
 * brake held in every frame; 128 without it: 51.9 -> 48.8). */
#define Q38CC_PROMOTE 128
#define Q38CC_PROMO_HITS 90
#define Q38CC_DECAY 1024
static int q38cc_promote_rate(void) { const char *e = getenv("COLI_CUDA_CHAIN_PROMOTE"); return e && *e ? atoi(e) : Q38CC_PROMOTE; }
static int q38cc_promote_hits(void) {
    const char *e = getenv("COLI_CUDA_CHAIN_PROMOTE_HITS"); int h = e && *e ? atoi(e) : Q38CC_PROMO_HITS;
    return h < 0 ? 0 : h > 100 ? 100 : h;
}
/* the frame's routes and pool reads into the window; 1 when the brake holds: the last
 * Q38CC_PROMO_WIN frames (this one among them) found promo_hits percent or more of their
 * routes on the card */
static int q38cc_braked(Q38CChain *ch) {
    unsigned route = (unsigned)(ch->routes - ch->pr_routes0), miss = (unsigned)(ch->pool_reads - ch->pr_reads0);
    ch->pr_routes0 = ch->routes; ch->pr_reads0 = ch->pool_reads;
    ch->pr_wroute[ch->pr_wi] = route; ch->pr_wmiss[ch->pr_wi] = miss;
    ch->pr_wi = (ch->pr_wi + 1) % Q38CC_PROMO_WIN; if (ch->pr_wn < Q38CC_PROMO_WIN) ch->pr_wn++;
    if (ch->promo_hits <= 0) return 0;
    unsigned long long wr = 0, wm = 0;
    for (int j = 0; j < ch->pr_wn; j++) { wr += ch->pr_wroute[j]; wm += ch->pr_wmiss[j]; }
    if (!wr || (wr - wm) * 100 < (unsigned long long)ch->promo_hits * wr) return 0;
    ch->pr_braked++;
    return 1;
}
static void q38cc_vram_entry(Q38CChain *ch, int slot, CcExpert *x) {
    const char *g = ch->ar_base + (size_t)slot * ch->ar_slot;
    memset(x, 0, sizeof *x);
    x->g = g; x->u = g + ch->ar_off_u; x->d = g + ch->ar_off_d;
}
static int q38cc_pool_entry(Q38CChain *ch, Model *m, int i, int k, CcExpert *x);
static int q38cc_promote_init(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts;
    size_t ne = (size_t)L * E, slot, ou, od;
    void *base; int n;
    if (!qt_chain_arena(&base, &slot, &ou, &od, &n) || n < 1) return 0;
    for (int i = 0; i < L; i++) {
        int64_t gb, db;
        if (!q38cc_pool_types(m, i, &gb, &db) || (size_t)gb > ou || (size_t)gb > od - ou || (size_t)db > slot - od) return 0;
    }
    ch->vslot = (int32_t *)malloc(ne * sizeof(int32_t)); ch->owner = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    ch->busy = (uint8_t *)calloc((size_t)n, 1); ch->heat = (uint32_t *)calloc(ne, sizeof(uint32_t)); ch->chg = (int32_t *)malloc(ne * sizeof(int32_t));
    if (!ch->vslot || !ch->owner || !ch->busy || !ch->heat || !ch->chg) {
        free(ch->vslot); free(ch->owner); free(ch->busy); free(ch->heat); free(ch->chg);
        ch->vslot = ch->owner = ch->chg = NULL; ch->busy = NULL; ch->heat = NULL;
        return 0;
    }
    for (size_t j = 0; j < ne; j++) ch->vslot[j] = -1;
    for (int k = 0; k < n; k++) ch->owner[k] = -1;
    int held = 0;
    for (int i = 0; i < L; i++) for (int e = 0; e < E; e++) {
        int k = qt_chain_slot(i, e);
        if (k >= 0 && k < n && ch->owner[k] < 0) { ch->vslot[(size_t)i * E + e] = k; ch->owner[k] = i * E + e; held++; }
    }
    qt_chain_own();
    ch->own = 1; ch->ar_base = (char *)base; ch->ar_slot = slot; ch->ar_off_u = ou; ch->ar_off_d = od; ch->ar_n = n;
    ch->promo_rate = q38cc_promote_rate(); ch->rng = 0x9e3779b9u; ch->dm_read = 0; ch->stage_dirty = 1;
    ch->promo_hits = q38cc_promote_hits(); ch->pr_routes0 = ch->routes; ch->pr_reads0 = ch->pool_reads;
    fprintf(stderr, "[chain] qwen38: the card's %d slots are the chain's to fill (%d held): up to %d experts a forward promoted from the "
                    "pool by this conversation's routes, none while the last %d frames found %d %% or more of their routes on the card "
                    "(COLI_CUDA_CHAIN_PROMOTE=0 keeps the warm start's set)\n", n, held, ch->promo_rate, Q38CC_PROMO_WIN, ch->promo_hits);
    return 1;
}
static const char *q38cc_pool_host(Q38CChain *ch, int k) {
    int ps = ch->pool_of[k];
    return ps < 0 ? NULL : (const char *)cc_ptr(ch->pool_buf[ps / ch->pool_per]) + (size_t)(ps % ch->pool_per) * ch->pool_stride;
}
/* before a frame: the newcomers whose bytes landed take their slots; the entries rewritten
 * since the last frame to the device's table */
static int q38cc_promote_settle(Q38CChain *ch, Model *m) {
    int E = m->c.experts, w = 0;
    for (int p = 0; p < ch->npend; p++) {
        struct Q38CCPend *q = &ch->pend[p];
        if (!cc_host_up_done(q->ticket)) { ch->pend[w++] = *q; continue; }
        ch->vslot[q->k] = q->slot; ch->busy[q->slot] = 0; ch->dm_res[q->k] = 1;
        q38cc_vram_entry(ch, q->slot, &ch->dm_host[q->k]);
        ch->chg[ch->nchg++] = q->k;
    }
    ch->npend = w;
    if (!ch->nchg) return 1;
    int ok = cc_begin();
    static const CcExpert none;
    for (int j = 0; j < ch->nchg && ok; j++) {
        const int k = ch->chg[j];
        ok = cc_write(ch->dm_tab, (size_t)k * (sizeof(CcExpert) / 4), &ch->dm_host[k], sizeof(CcExpert));   /* offsets in floats */
        /* the decode table: a slot back in the pool is the CPU's (pool_cpu) */
        if (ok && ch->pool_cpu && ch->dm_tabd)
            ok = cc_write(ch->dm_tabd, (size_t)k * (sizeof(CcExpert) / 4), ch->dm_res[k] >= 2 ? &none : &ch->dm_host[k], sizeof(CcExpert));
    }
    ok = ok && cc_submit(1);
    if (ch->pool_cpu && ch->dm_tabd)   /* the changed layers' CPU slots */
        for (int j = 0; j < ch->nchg; j++) {
            const int i = ch->chg[j] / E; int any = 0;
            for (int e = 0; e < E && !any; e++) any = ch->dm_res[(size_t)i * E + e] != 1;
            ch->dm_partd[i] = any;
        }
    ch->nchg = 0; ch->stage_dirty = 1;
    return ok;
}
/* after a frame: this conversation's routes, and the promotions they earn */
static void q38cc_promote_books(Q38CChain *ch, Model *m, int n) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts, K = c->topk;
    uint32_t **hist = rt_counts_all();
    #define Q38CC_HIST(j) (hist && hist[(j) / E] ? hist[(j) / E][(j) % E] : 0u)
    for (int i = 0; i < L; i++) {
        if (!ch->dm_ran[i]) continue;
        const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
        for (int a = 0; a < n * K; a++) if (ids[a] >= 0 && ids[a] < E && ch->heat[(size_t)i * E + ids[a]] < 0xFFFFFFFFu) ch->heat[(size_t)i * E + ids[a]]++;
    }
    if ((ch->heat_tick += (uint32_t)n) >= Q38CC_DECAY) {
        ch->heat_tick = 0;
        for (size_t j = 0; j < (size_t)L * E; j++) ch->heat[j] >>= 1;
    }
    if (q38cc_braked(ch)) return;
    int done = 0;
    for (int i = 0; i < L && done < ch->promo_rate; i++) {
        if (!ch->dm_ran[i]) continue;
        const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
        for (int a = 0; a < n * K && done < ch->promo_rate; a++) {
            int e = ids[a];
            if (e < 0 || e >= E) continue;
            int k = i * E + e;
            if (ch->dm_res[k] != 2) continue;   /* on the card, or on its way */
            /* a free slot, else the coldest of 64 drawn (none on its way) */
            int s = -1, v = -1;
            for (int t = 0; t < 64; t++) {
                ch->rng = ch->rng * 1664525u + 1013904223u;
                int r = (int)((ch->rng >> 8) % (unsigned)ch->ar_n);
                if (ch->busy[r]) continue;
                if (ch->owner[r] < 0) { s = r; v = -1; break; }
                int o = ch->owner[r];
                if (v < 0 || ch->heat[o] < ch->heat[v] || (ch->heat[o] == ch->heat[v] && Q38CC_HIST(o) < Q38CC_HIST(v))) { v = o; s = r; }
            }
            if (s < 0) continue;
            if (v >= 0 && !(ch->heat[k] > ch->heat[v] || (ch->heat[k] == ch->heat[v] && Q38CC_HIST(k) > Q38CC_HIST(v)))) continue;
            int64_t gb, db;
            const char *src = q38cc_pool_host(ch, k);
            if (!src || !q38cc_pool_types(m, i, &gb, &db)) continue;
            if (ch->npend == ch->cpend) {
                int cap = ch->cpend ? 2 * ch->cpend : 64;
                struct Q38CCPend *np = (struct Q38CCPend *)realloc(ch->pend, (size_t)cap * sizeof *np);
                if (!np) return;
                ch->pend = np; ch->cpend = cap;
            }
            if (v >= 0) {   /* the victim to the pool, its entry rewritten before the next frame */
                ch->vslot[v] = -1; ch->dm_res[v] = 2;
                if (!q38cc_pool_entry(ch, m, v / E, ch->pool_of[v], &ch->dm_host[v])) memset(&ch->dm_host[v], 0, sizeof(CcExpert));
                ch->chg[ch->nchg++] = v;
                ch->promo_victims++;
            }
            char *dst = ch->ar_base + (size_t)s * ch->ar_slot;
            unsigned tk = 0;
            if (!cc_host_up(dst, src, (size_t)gb, &tk) || !cc_host_up(dst + ch->ar_off_u, src + Q38CC_PAD(gb), (size_t)gb, &tk) ||
                !cc_host_up(dst + ch->ar_off_d, src + 2 * Q38CC_PAD(gb), (size_t)db, &tk)) {
                ch->owner[s] = -1;   /* the slot's bytes undefined now: nobody's */
                return;
            }
            ch->owner[s] = k; ch->busy[s] = 1; ch->dm_res[k] = 3;
            ch->pend[ch->npend++] = (struct Q38CCPend){k, s, tk};
            ch->promos++; done++;
        }
    }
    #undef Q38CC_HIST
}
typedef struct { int i, e; uint32_t n; } Q38CCHot;
static int q38cc_hot_cmp(const void *a, const void *b) {
    const Q38CCHot *x = (const Q38CCHot *)a, *y = (const Q38CCHot *)b;
    if (x->n != y->n) return x->n > y->n ? -1 : 1;
    return x->i != y->i ? x->i - y->i : x->e - y->e;
}
static int q38cc_layer_cmp(const void *a, const void *b) {
    const Q38CCHot *x = (const Q38CCHot *)a, *y = (const Q38CCHot *)b;
    return x->i != y->i ? x->i - y->i : x->e - y->e;
}
static size_t q38_state_later(Model *m);   /* qwen38.c: the KV and state a serve allocates once it runs */
static void q38cc_pool_init(Model *m) {
    Q38CChain *ch = q38cc_of(m);
    const char *on = getenv("COLI_CUDA_CHAIN_POOL"), *gbs = getenv("COLI_CUDA_CHAIN_POOL_GB");
    if (!ch || !ch->ok || !ch->dm_on || ch->dm_fmt < 100 || !g_q38_gguf || (on && *on == '0') || !cc_fetch_ready()) return;
    Cfg *c = &m->c; int L = c->layers, E = c->experts;
    size_t stride = 0;
    for (int i = 0; i < L; i++) {
        int64_t gb, db;
        if (!q38cc_pool_types(m, i, &gb, &db)) { fprintf(stderr, "[chain] qwen38: layer %d's expert rows are not whole words; no host pool\n", i); return; }
        size_t b = 2 * Q38CC_PAD(gb) + Q38CC_PAD(db);
        if (b > stride) stride = b;
    }
    /* the experts the card leaves, the history's most used first, then the card's own (the pool
     * takes those too where RAM holds every expert: the chain then moves them, below) */
    Q38CCHot *hot = (Q38CCHot *)malloc((size_t)L * E * sizeof *hot);
    if (!hot) return;
    uint32_t **hist = rt_counts_all();
    int left = 0, all = L * E, on_card = 0;
    for (int i = 0; i < L; i++) for (int e = 0; e < E; e++) {
        ColiCudaTensor *g, *u, *d;
        if (qt_chain_resident(i, e, ch->dev, &g, &u, &d)) continue;
        hot[left++] = (Q38CCHot){i, e, hist && hist[i] ? hist[i][e] : 0};
    }
    for (int i = 0; i < L; i++) for (int e = 0; e < E; e++) {
        ColiCudaTensor *g, *u, *d;
        if (qt_chain_resident(i, e, ch->dev, &g, &u, &d)) hot[left + on_card++] = (Q38CCHot){i, e, hist && hist[i] ? hist[i][e] : 0};
    }
    if (!left) { free(hot); return; }
    qsort(hot, (size_t)left, sizeof *hot, q38cc_hot_cmp);   /* the hottest first; those the pool takes then by layer (below) */
    const double GiB = 1073741824.0;
    double grow = 0;   /* what the RAM cache may still take: the experts the card leaves, at most */
    for (int i = 0; i < L; i++) {
        int room = m->cache[i].cap - m->cache[i].n, out = 0;
        for (int k = 0; k < left; k++) out += hot[k].i == i;
        grow += (double)(room < out ? room : out) * (double)stride;
    }
    /* and the KV and state the serve allocates once it runs, which the free RAM does not show
     * yet (#2314: a host pool took the room of a 131072-token KV cache, whose allocation failed) */
    const double later = (double)q38_state_later(m);
    double want = gbs && *gbs ? atof(gbs) * GiB : compat_mem_available_gb() * 1e9 - 4.0 * GiB - grow - later;
    size_t blk = (size_t)256 << 20;
    int per = (int)(blk / stride);
    long long fit = want > 0 ? (long long)(want / (double)stride) : 0;
    if (fit > left) fit = left;
    /* every expert, where RAM holds them (the CPU then computes none: no RAM cache to come) and
     * the chain may take the arena: an eviction then copies nothing back */
    int full = 0;
    {   void *ab; size_t as, au, ad; int an;
        if (!(gbs && *gbs) && q38cc_promote_rate() > 0 && left + on_card == all &&
            compat_mem_available_gb() * 1e9 - 4.0 * GiB - later >= (double)all * stride && qt_chain_arena(&ab, &as, &au, &ad, &an)) { full = 1; fit = all; }
    }
    if (per < 1 || fit < 1) {
        fprintf(stderr, "[chain] qwen38: no RAM for a host pool (%.1f GiB free less 4 GiB, the RAM cache's %.1f GiB to come "
                        "and the serve's KV and state, %.1f GiB; COLI_CUDA_CHAIN_POOL_GB sets it)\n",
                compat_mem_available_gb() * 1e9 / GiB, grow / GiB, later / GiB);
        free(hot);
        return;
    }
    qsort(hot, (size_t)fit, sizeof *hot, q38cc_layer_cmp);   /* a layer's pooled experts side by side: a prompt chunk's copies in few runs */
    int nb = (int)((fit + per - 1) / per);
    double t0 = now_s();
    ch->pool_buf = (CcBuf **)calloc((size_t)nb, sizeof(CcBuf *));
    ch->pool_of = (int32_t *)malloc((size_t)L * E * sizeof(int32_t));
    if (!ch->pool_buf || !ch->pool_of) { free(ch->pool_buf); free(ch->pool_of); ch->pool_buf = NULL; ch->pool_of = NULL; free(hot); return; }
    for (size_t j = 0; j < (size_t)L * E; j++) ch->pool_of[j] = -1;
    int got = 0;
    for (; got < nb; got++) {
        size_t bytes = (size_t)(fit - (long long)got * per < per ? fit - (long long)got * per : per) * stride;
        if (!(ch->pool_buf[got] = cc_buf(bytes, CC_DOWN)) || !cc_dev_addr(ch->pool_buf[got])) { cc_free(ch->pool_buf[got]); ch->pool_buf[got] = NULL; break; }
    }
    if ((long long)got * per < fit) fit = (long long)got * per;
    ch->pool_nb = got; ch->pool_per = per; ch->pool_stride = stride; ch->pool_n = (int)fit;
    double ta = now_s();
    #pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k < (int)fit; k++) {
        const gguf_tensor *t[3]; q38_gg_expert_tensors(m, hot[k].i, t);
        int64_t gb, db; q38cc_pool_types(m, hot[k].i, &gb, &db);
        char *dst = (char *)cc_ptr(ch->pool_buf[k / per]) + (size_t)(k % per) * stride;
        const int64_t eb[3] = {gb, gb, db}; const size_t off[3] = {0, Q38CC_PAD(gb), 2 * Q38CC_PAD(gb)};
        for (int q = 0; q < 3; q++) q38_gg_read(t[q], dst + off[q], (int64_t)hot[k].e * eb[q], eb[q]);
    }
    for (int k = 0; k < (int)fit; k++) ch->pool_of[(size_t)hot[k].i * E + hot[k].e] = k;
    const char *pr = getenv("COLI_CUDA_CHAIN_POOL_READ");
    ch->pool_cpu = !(pr && *pr == '1');
    ch->dm_read = 0;   /* the table again, with the pool */
    /* a prompt chunk's staging: the most pooled experts any layer has, within Q38CC_PROMPT_STAGE
     * (the chain's reserve counted it); a layer's past it are read where they are */
    int hmax = 0;
    for (int i = 0, k = 0; i < L; i++) { int h = 0; while (k < (int)fit && hot[k].i == i) { h++; k++; } if (h > hmax) hmax = h; }
    int pcap = (int)(Q38CC_PROMPT_STAGE / stride);
    if (pcap > hmax) pcap = hmax;
    if (pcap > 0 && (ch->dm_stage = cc_buf((size_t)pcap * stride, CC_DEV))) ch->dm_stage_cap = pcap;
    /* the staging: the GEMV kernels' word loads over the bus came at a third of its rate, a
     * copy in 16-byte loads first at the whole of it (Q38CC_STAGE places; past them, a step's
     * pooled experts are read where they are) */
    const char *pst = getenv("COLI_CUDA_CHAIN_POOL_STAGE");   /* 0: the kernels read the pool where it is (A/B) */
    int nst = pst && *pst == '0' ? 0 : Q38CC_STAGE;
    if (nst && cc_moe_stage_ready() && (ch->pool_stage = cc_buf(CC_MOE_STAGE_MAP + (size_t)nst * stride, CC_DEV)) &&
        !cc_moe_stage(ch->pool_stage, stride, nst)) { cc_free(ch->pool_stage); ch->pool_stage = NULL; }
    if (ch->pool_n == all && full)
        fprintf(stderr, "[chain] qwen38: host pool, every expert (the %d the card leaves and its own %d) in %.2f GiB of page-locked RAM "
                        "the device reads over the bus (%.1f s to allocate, %.1f s to read; COLI_CUDA_CHAIN_POOL=0 off)\n",
                left, on_card, (double)fit * stride / GiB, ta - t0, now_s() - ta);
    else
        fprintf(stderr, "[chain] qwen38: host pool, %d of the %d experts the card leaves in %.2f GiB of page-locked RAM the device "
                        "reads over the bus, the history's most used first (%.1f s to allocate, %.1f s to read; COLI_CUDA_CHAIN_POOL=0 off, "
                        "COLI_CUDA_CHAIN_POOL_GB sets it)\n",
                ch->pool_n, left, (double)fit * stride / GiB, ta - t0, now_s() - ta);
    if (ch->pool_stage) fprintf(stderr, "[chain] qwen38: a step's pooled experts copied to the device first (%d places, %.1f MiB)\n",
                                nst, (double)nst * stride / 1048576.0);
    if (ch->dm_stage) fprintf(stderr, "[chain] qwen38: a prompt chunk's layer's pooled experts copied to the device while the layers "
                                      "before it run (%d places, %.1f MiB; the most a layer has: %d)\n",
                              ch->dm_stage_cap, (double)ch->dm_stage_cap * stride / 1048576.0, hmax);
    free(hot);
    if (ch->pool_n == all && full) q38cc_promote_init(ch, m);
}
static int q38cc_dm_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN_MOE"); on = !(e && *e == '0'); }
    return on;
}
static int q38cc_dm_able(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int E = c->experts, K = c->topk, L = c->layers;
    const char *g = getenv("Q38_MOE_GROUPED");
    if (!q38cc_dm_wanted() || !cc_moe_ready() || E <= 0 || E > CC_ROUTE_EMAX || K < 1 || K > 32 || rt_fp ||
        g_q38_act || g_q38_dump || g_q38_cov || (g && g[0] == '0') || !qt_experts_on() || qt_chain_devices() != 1) return 0;
    if (c->hidden % 32 || c->inter % 32 || c->hidden > CC_MOE_DIM_MAX || c->inter > CC_MOE_DIM_MAX) return 0;
    if (!qt_chain_format(&ch->dm_fmt, &ch->dm_gs) || (ch->dm_fmt == 4 && (ch->dm_gs < 16 || ch->dm_gs % 16))) return 0;
    int need = Q38CC_DM_ROWS * K < E ? Q38CC_DM_ROWS * K : E;   /* the host's step: a step's distinct experts in the RAM cache at once */
    for (int i = 0; i < L; i++) {
        if (!(q38_layer_int4(m, i) || g_q38_gguf) || m->cache[i].cap < need) return 0;
        int fg, fd;
        if (ch->dm_fmt >= 100 && (!qt_chain_layer_format(i, &fg, &fd) || !cc_moe_block_type(fg - 100) || !cc_moe_block_type(fd - 100)))
            return 0;
    }
    return 1;
}
static int q38cc_stage_table(Q38CChain *ch, Model *m);
/* the step's buffers, and the table when the tier's residents moved (the tier held) */
static int q38cc_dm_begin(Q38CChain *ch, Model *m, int n) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts, K = c->topk, D = c->hidden, I = c->inter;
    size_t r = (size_t)ch->rows, ne = (size_t)L * E, F = sizeof(float);
    if (!ch->dm_host) {
        ch->dm_host = malloc(ne * sizeof(CcExpert)); ch->dm_res = calloc(ne, 1); ch->dm_part = calloc(L, sizeof(int));
        ch->dm_ran = calloc(L, 1); ch->dm_dev = malloc((size_t)Q38CC_DM_ROWS * K); ch->dm_partd = calloc(L, sizeof(int));
        if (!ch->dm_host || !ch->dm_res || !ch->dm_part || !ch->dm_ran || !ch->dm_dev || !ch->dm_partd) return 0;
    }
    if (!cc_reserve(&ch->dm_ids, (size_t)L * r * K * 4, CC_DEV) || !cc_reserve(&ch->dm_w, (size_t)L * r * K * F, CC_DEV) ||
        !cc_reserve(&ch->dm_idsd, (size_t)L * r * K * 4, CC_DOWN) || !cc_reserve(&ch->dm_wd, (size_t)L * r * K * F, CC_DOWN) ||
        !cc_reserve(&ch->dm_hid, cc_moe_hid_floats((int)r, K, D, I, E) * F, CC_DEV) || !cc_reserve(&ch->dm_y, r * D * F, CC_DEV) ||
        !cc_reserve(&ch->dm_hy, r * D * F, CC_UP) || !cc_reserve(&ch->dm_dfl, 64, CC_DOWN) || !cc_reserve(&ch->dm_hfl, 64, CC_UP) ||
        !cc_reserve(&ch->dm_tab, ne * sizeof(CcExpert), CC_DEV)) return 0;
    if (ch->pool_cpu && ch->pool_of && !cc_reserve(&ch->dm_tabd, ne * sizeof(CcExpert), CC_DEV)) ch->pool_cpu = 0;
    if (!ch->dm_held) { qt_chain_hold(1); ch->dm_held = 1; }
    if (ch->own && ch->dm_read) {   /* the chain's slots: the newcomers landed, the entries rewritten, then the frame */
        if (!q38cc_promote_settle(ch, m)) return 0;
        if (n > Q38CC_DM_ROWS && ch->stage_dirty && ch->dm_stage && cc_fetch_ready()) {
            if (!q38cc_stage_table(ch, m)) return 0;
            ch->stage_dirty = 0;
        }
        return 1;
    }
    unsigned ep = qt_chain_epoch();
    if (ch->dm_read && ep == ch->dm_epoch) return 1;
    for (int i = 0; i < L; i++) {
        int miss = 0;
        for (int e = 0; e < E; e++) {
            CcExpert *x = &ch->dm_host[(size_t)i * E + e];
            ColiCudaTensor *g, *u, *d;
            memset(x, 0, sizeof *x);
            int on = 0;
            if (ch->own) { if (ch->vslot[(size_t)i * E + e] >= 0) { q38cc_vram_entry(ch, ch->vslot[(size_t)i * E + e], x); on = 1; } }
            else on = qt_chain_resident(i, e, ch->dev, &g, &u, &d) && cc_tensor_dev(g, &x->g, &x->gs, NULL) &&
                      cc_tensor_dev(u, &x->u, &x->us, NULL) && cc_tensor_dev(d, &x->d, &x->ds, NULL);
            if (!on) memset(x, 0, sizeof *x);
            if (!on && ch->pool_of && ch->pool_of[(size_t)i * E + e] >= 0 && q38cc_pool_entry(ch, m, i, ch->pool_of[(size_t)i * E + e], x))
                on = 2;   /* the host pool's: the device reads it over the bus */
            ch->dm_res[(size_t)i * E + e] = (uint8_t)on; miss |= !on;
        }
        ch->dm_part[i] = miss;
    }
    CcExpert *td = NULL;   /* the decode table: the pool's entries left to the CPU */
    if (ch->pool_cpu && ch->dm_tabd) {
        if (!(td = malloc(ne * sizeof(CcExpert)))) ch->pool_cpu = 0;
        else {
            memcpy(td, ch->dm_host, ne * sizeof(CcExpert));
            for (int i = 0; i < L; i++) {
                int any = 0;
                for (int e = 0; e < E; e++) { size_t j = (size_t)i * E + e; if (ch->dm_res[j] != 1) { any = 1; if (ch->dm_res[j] >= 2) memset(&td[j], 0, sizeof *td); } }
                ch->dm_partd[i] = any;
            }
        }
    }
    int wok = cc_begin() && cc_write(ch->dm_tab, 0, ch->dm_host, ne * sizeof(CcExpert)) && (!td || cc_write(ch->dm_tabd, 0, td, ne * sizeof(CcExpert))) && cc_submit(1);
    free(td);
    if (!wok) return 0;
    if (ch->dm_stage && cc_fetch_ready() && !q38cc_stage_table(ch, m)) {
        if (cc_lost()) return 0;
        fprintf(stderr, "[chain] qwen38: no room for a prompt chunk's staging table; its pooled experts are read where they are\n");
        cc_free(ch->dm_stage); ch->dm_stage = NULL;
    }
    ch->dm_epoch = ep; ch->dm_read = 1; ch->stage_dirty = 0; ch->nchg = 0;
    return 1;
}
/* A prompt chunk's table: each layer's pooled experts (up to dm_stage_cap) at their places in
 * dm_stage, and the copies that put them there (dm_fr, dm_fro[i] and dm_frn[i] a layer's, the
 * runs of the pool's places a layer has side by side merged) */
static int q38cc_stage_table(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts;
    size_t ne = (size_t)L * E, st = ch->pool_stride;
    if (!ch->dm_hostb) {
        ch->dm_hostb = malloc(ne * sizeof(CcExpert)); ch->dm_fro = calloc(L, sizeof(int)); ch->dm_frn = calloc(L, sizeof(int));
        ch->dm_fr = malloc(ne * sizeof(CcFetch));
        if (!ch->dm_hostb || !ch->dm_fro || !ch->dm_frn || !ch->dm_fr) return 0;
    }
    if (!cc_reserve(&ch->dm_tabb, ne * sizeof(CcExpert), CC_DEV)) return 0;
    const char *base = (const char *)cc_dev_addr(ch->dm_stage);
    if (!base) return 0;
    int k = 0;
    for (int i = 0; i < L; i++) {
        int h = 0; int64_t gb = 0, db = 0;
        ch->dm_fro[i] = k;
        int typed = q38cc_pool_types(m, i, &gb, &db);
        for (int e = 0; e < E; e++) {
            size_t j = (size_t)i * E + e;
            CcExpert *x = &ch->dm_hostb[j];
            *x = ch->dm_host[j];
            if (ch->dm_res[j] < 2 || h >= ch->dm_stage_cap || !typed) continue;
            int ps = ch->pool_of[j];
            const char *src = (const char *)cc_ptr(ch->pool_buf[ps / ch->pool_per]) + (size_t)(ps % ch->pool_per) * st;
            size_t dst = (size_t)h * st, span = 2 * Q38CC_PAD(gb) + (size_t)db;
            const char *g = base + dst;
            memset(x, 0, sizeof *x);
            x->g = g; x->u = g + Q38CC_PAD(gb); x->d = g + 2 * Q38CC_PAD(gb);
            CcFetch *pr = k > ch->dm_fro[i] ? &ch->dm_fr[k - 1] : NULL;
            if (pr && src - (const char *)pr->src == (ptrdiff_t)(dst - pr->dst)) pr->bytes = dst + span - pr->dst;   /* the run goes on */
            else ch->dm_fr[k++] = (CcFetch){src, dst, span};
            h++;
        }
        ch->dm_frn[i] = k - ch->dm_fro[i];
    }
    return cc_begin() && cc_write(ch->dm_tabb, 0, ch->dm_hostb, ne * sizeof(CcExpert)) && cc_submit(1);
}
/* the copy of the first layer from i on that stages its experts (none: nothing to copy) */
static int q38cc_stage_from(Q38CChain *ch, Model *m, int i) {
    for (; i < m->c.layers; i++) {
        if (ch->dm_frn[i] <= 0) continue;
        ch->dm_staged++;
        for (int k = 0; k < ch->dm_frn[i]; k++) ch->dm_stage_bytes += (double)ch->dm_fr[ch->dm_fro[i] + k].bytes;
        return cc_fetch(ch->dm_stage, ch->dm_fr + ch->dm_fro[i], ch->dm_frn[i]);
    }
    return 1;
}
static void q38cc_dm_end(Q38CChain *ch) { if (ch->dm_held) { qt_chain_hold(0); ch->dm_held = 0; } }
/* layer i's router, routing, shared expert and the device's experts; *v: the host's word
 * a hybrid layer waits for (0: none) */
/* a step's table: a prompt chunk's stages its layers' pooled experts */
static CcBuf *q38cc_dm_tab(Q38CChain *ch, int n) {
    if (n <= Q38CC_DM_ROWS && ch->pool_cpu && ch->dm_tabd) return ch->dm_tabd;   /* the pool's experts on the CPU */
    return n > Q38CC_DM_ROWS && ch->dm_stage && ch->dm_tabb ? ch->dm_tabb : ch->dm_tab;
}
/* layer i has slots the host computes in a step of n rows */
static int q38cc_dm_hosted(Q38CChain *ch, int i, int n) { return n <= Q38CC_DM_ROWS && ch->pool_cpu && ch->dm_tabd ? ch->dm_partd[i] : ch->dm_part[i]; }
static int q38cc_dm_layer(Q38CChain *ch, Model *m, Layer *l, int i, int n, unsigned *v) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, K = c->topk, io = i * ch->rows * K;
    CcBuf *tab = q38cc_dm_tab(ch, n);
    int stg = tab == ch->dm_tabb && ch->dm_frn[i] > 0;   /* this layer's copy to wait for */
    CcRoute rp = {n, E, K, 0, 0, E, 0, 0, io, K, io, K, 1, 1, c->norm_topk ? 1 : 0, 1.f, 1.f, 0.f};
    CcMoe mp = {n, K, H, c->inter, ch->dm_fmt, ch->dm_gs, 0, 1, i * E, 0, H, io, K, io, K, 0, H, -1, E, 0, 0};
    if (ch->dm_fmt >= 100) qt_chain_layer_format(i, &mp.fmt, &mp.fmt_d);   /* a GGUF's types change from layer to layer */
    int ok = cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) && cc_route(ch->lg, NULL, ch->dm_ids, ch->dm_w, &rp) &&
             cc_copy(ch->dm_idsd, io, ch->dm_ids, io, (size_t)n * K) && cc_copy(ch->dm_wd, io, ch->dm_w, io, (size_t)n * K);
    *v = 0;
    if (ok && q38cc_dm_hosted(ch, i, n)) { *v = ++ch->dm_seq; ok = cc_copy(ch->mixd, 0, ch->mixed, 0, (size_t)n * H) && cc_signal(ch->dm_dfl, 0, *v); }
    ok = ok && q38cc_shared(ch, m, l, i, n) && (!stg || cc_fetch_wait()) &&
         cc_moe(tab, ch->mixed, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y, NULL, &mp);
    if (ok && stg) ok = cc_fetch_done() && q38cc_stage_from(ch, m, i + 1);   /* the sum below reads no weights */
    if (ok && *v) ok = cc_wait_host(ch->dm_hfl, 0, *v);
    CcMoe ms = mp; ms.parts = 2; ms.h_off = *v ? 0 : -1;
    ok = ok && cc_moe(tab, NULL, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y, ch->dm_hy, &ms);
    ch->dm_steps++;
    return ok;
}
/* np pooled slots on the CPU, their rows where the pool holds them (g[p]: gate, up and down at
 * 0, Q38CC_PAD(gate bytes), twice that), x[p] each slot's input row: gate and up, SwiGLU, down
 * into dn[p * H]. A matrix's rows by eight (ggml_quants.h gq_rows_f8, whose bits do not
 * depend on the other rows of a call) where its type has the kernel, the row kernels where
 * not: vk_tier.c's host_pairs computes the same, in one parallel region. */
static int q38cc_pool_pairs(const uint8_t *const *g, const float *const *x, int np, int H, int I, int tg, int td,
                            float *gu, float *hh, float *dn) {
    const int64_t rg = gq_row_bytes(tg, H), rd = gq_row_bytes(td, I);
    const size_t pg = Q38CC_PAD(rg * I);
    gq_act *ax = calloc((size_t)np, sizeof *ax), *ah = calloc((size_t)np, sizeof *ah);
    int err = !ax || !ah;
    for (int p = 0; p < np && !err; p++) err = gq_act_set(&ax[p], x[p], 1, H) != 0;
    const int cgu = (2 * I + 7) / 8, cdn = (H + 7) / 8, f8g = gq_f8_ok(tg, H), f8d = gq_f8_ok(td, I);
    if (!err) {
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 4)
            for (int c = 0; c < np * cgu; c++) {
                const int p = c / cgu, o0 = (c % cgu) * 8, o1 = o0 + 8 < 2 * I ? o0 + 8 : 2 * I;
                if (o1 - o0 == 8 && (o1 <= I || o0 >= I) && f8g) {
                    const uint8_t *m = o0 < I ? g[p] : g[p] + pg; const int oo = o0 < I ? o0 : o0 - I;
                    gq_rows_f8(gu + (size_t)p * 2 * I + o0, 1, &ax[p], m + (int64_t)oo * rg, rg, tg, H, 8);
                    continue;
                }
                for (int o = o0; o < o1; o++) {
                    const uint8_t *m = o < I ? g[p] : g[p] + pg; const int oo = o < I ? o : o - I;
                    gq_row_all(gu + (size_t)p * 2 * I + o, 1, &ax[p], m + (int64_t)oo * rg, tg, H);
                }
            }
            #pragma omp for schedule(static)
            for (int p = 0; p < np; p++) {
                const float *gg = gu + (size_t)p * 2 * I, *uu = gg + I; float *h = hh + (size_t)p * I;
                for (int o = 0; o < I; o++) h[o] = gg[o] / (1.f + expf(-gg[o])) * uu[o];
                if (gq_act_set(&ah[p], h, 1, I)) {
                    #pragma omp atomic write
                    err = 1;
                }
            }
            #pragma omp for schedule(dynamic, 8)
            for (int c = 0; c < np * cdn; c++) {
                const int p = c / cdn, o0 = (c % cdn) * 8, o1 = o0 + 8 < H ? o0 + 8 : H;
                if (!ah[p].q) continue;
                const uint8_t *m = g[p] + 2 * pg;
                if (o1 - o0 == 8 && f8d) { gq_rows_f8(dn + (size_t)p * H + o0, 1, &ah[p], m + (int64_t)o0 * rd, rd, td, I, 8); continue; }
                for (int o = o0; o < o1; o++) gq_row_all(dn + (size_t)p * H + o, 1, &ah[p], m + (int64_t)o * rd, td, I);
            }
        }
    }
    for (int p = 0; ax && ah && p < np; p++) { gq_act_free(&ax[p]); gq_act_free(&ah[p]); }
    free(ax); free(ah);
    return !err;
}
/* the host's share of hybrid layer i: once the device's word says its routes and rows are
 * down, the slots the table lacks, its rows up, its word */
static int q38cc_dm_serve(Q38CChain *ch, Model *m, int i, int n, unsigned v) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, K = c->topk;
    volatile unsigned *dw = (volatile unsigned *)cc_ptr(ch->dm_dfl);
    double t0 = now_s();
    for (unsigned long spin = 0; *dw < v; spin++)
        if ((spin & 0xFFFFu) == 0 && (cc_lost() || now_s() - t0 > 60.0)) {
            fprintf(stderr, "[chain] qwen38: the device did not reach layer %d's routes in 60 s\n", i);
            return 0;
        }
    double t1 = now_s();
    const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
    const float *w = (const float *)cc_ptr(ch->dm_wd) + (size_t)i * ch->rows * K;
    const float *x = (const float *)cc_ptr(ch->mixd);
    /* the slots the table lacks: the pool's (pool_cpu: their rows in the pool's cached RAM,
     * computed in place as vk_tier.c's hybrid steps do) and the true misses (the RAM cache) */
    const int pc = ch->pool_cpu && ch->dm_tabd && n <= Q38CC_DM_ROWS;
    int npc = 0, nmiss = 0, pa[Q38CC_DM_ROWS * 32];
    for (int a = 0; a < n * K; a++) {
        int e = ids[a]; const uint8_t r = e < 0 || e >= E ? 1 : ch->dm_res[(size_t)i * E + e];
        const int pool = pc && r >= 2 && npc < Q38CC_DM_ROWS * 32;
        ch->dm_dev[a] = (uint8_t)(r && !(pc && r >= 2 && !pool) ? 1 : 0);   /* the grouped path's: a miss, or a pooled slot past pa */
        if (pool) pa[npc++] = a;
        if (!ch->dm_dev[a]) nmiss++;
        if (!ch->dm_dev[a] || pool) ch->dm_pairs++;
    }
    if (nmiss && !q38_moe_decode_grouped(m, &m->L[i], i, x, n, ch->host_routed, NULL, 1, 0, ids, w, ch->dm_dev)) {
        fprintf(stderr, "[chain] qwen38: the host's step refused layer %d's experts\n", i);
        return 0;
    }
    if (!nmiss) memset(ch->host_routed, 0, (size_t)n * H * sizeof(float));
    if (npc) {
        double tp = now_s();
        int fg, fd;
        if (!qt_chain_layer_format(i, &fg, &fd) || fg < 100 || fd < 100) {
            fprintf(stderr, "[chain] qwen38: layer %d's pooled experts have no block types for the CPU\n", i);
            return 0;
        }
        const int I = c->inter, tg = fg - 100, td = fd - 100;
        if (ch->pc_cap < npc) {
            free(ch->pc_gu); free(ch->pc_hh); free(ch->pc_dn);
            ch->pc_gu = malloc((size_t)npc * 2 * I * sizeof(float)); ch->pc_hh = malloc((size_t)npc * I * sizeof(float));
            ch->pc_dn = malloc((size_t)npc * H * sizeof(float));
            ch->pc_cap = ch->pc_gu && ch->pc_hh && ch->pc_dn ? npc : 0;
            if (!ch->pc_cap) { fprintf(stderr, "[chain] qwen38: no memory for the CPU's pool slots\n"); return 0; }
        }
        const uint8_t *gp[Q38CC_DM_ROWS * 32]; const float *xp[Q38CC_DM_ROWS * 32];
        for (int p = 0; p < npc; p++) {
            const int a = pa[p]; const char *g = q38cc_pool_host(ch, i * E + ids[a]);
            if (!g) { fprintf(stderr, "[chain] qwen38: layer %d's expert %d has no place in the host pool\n", i, ids[a]); return 0; }
            gp[p] = (const uint8_t *)g; xp[p] = x + (size_t)(a / K) * H;
        }
        if (!q38cc_pool_pairs(gp, xp, npc, H, I, tg, td, ch->pc_gu, ch->pc_hh, ch->pc_dn)) {
            fprintf(stderr, "[chain] qwen38: the CPU's pool slots of layer %d failed\n", i);
            return 0;
        }
        for (int p = 0; p < npc; p++) {   /* after the misses, in rank order within a row */
            const int a = pa[p]; float *ys = ch->host_routed + (size_t)(a / K) * H; const float *o = ch->pc_dn + (size_t)p * H, wv = w[a];
            for (int d = 0; d < H; d++) ys[d] += wv * o[d];
        }
        ch->pc_pairs += (unsigned long long)npc; ch->pc_ms += (now_s() - tp) * 1e3;
    }
    memcpy(cc_ptr(ch->dm_hy), ch->host_routed, (size_t)n * H * sizeof(float));
    __sync_synchronize();   /* the rows (write-combined memory) before the word */
    ((volatile unsigned *)cc_ptr(ch->dm_hfl))[0] = v;
    __sync_synchronize();
    double t2 = now_s();
    ch->dm_hyb++; ch->dm_spin_ms += (t1 - t0) * 1e3; ch->dm_host_ms += (t2 - t1) * 1e3; ch->host_ms += (t2 - t1) * 1e3;
    return 1;
}
/* after the frame: every layer's books from its ids, and the K/V rows */
static void q38cc_dm_books(Q38CChain *ch, Model *m, int n, int pb) {
    Cfg *c = &m->c; int L = c->layers, K = c->topk;
    for (int i = 0; i < L; i++) {
        if (!ch->dm_ran[i]) continue;
        q38cc_kv_down(ch, m, i, n, pb);
        const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
        for (int s = 0; s < n; s++) { rt_count(i, ids + s * K, K); if (!ch->own) qt_chain_count(i, ids + s * K, K); }
        for (int j = 0; j < n * K; j++) if (ids[j] >= 0) {
            q38_ehit_mark(m, i, ids[j]);
            ch->routes++;
            if (ch->dm_res[(size_t)i * c->experts + ids[j]] >= 2) ch->pool_reads++;
        }
    }
    if (ch->own && ch->promo_rate > 0) q38cc_promote_books(ch, m, n);
}
static int q38cc_frame_begin(Q38CChain *ch) {
    ch->frames++;
    if (ch->fault_at && ch->frames == ch->fault_at) { fprintf(stderr, "[chain] qwen38: fault injected at frame %llu\n", ch->frames); return 0; }
    return cc_begin();
}

/* ---- a long prompt layer by layer (qwen38_chain.h's q38c_forward_lm, on CUDA) ----------
 * A prompt of more chunks than one staged every layer's host pool experts once a chunk.
 * Where every layer runs its experts in the frame (no hybrid layer) and a chunk stages them,
 * the prompt runs layer by layer: each layer's chunks in order, its experts' copy waited for
 * at its first chunk and the next staged layer's copy started after its last, every row's
 * streams on the device between layers (hyper_all), each layer's chunk a frame waited for
 * (the host's tables it writes and its K/V rows down are a chunk's, each chunk the bits it
 * gets chunk by chunk). The promotions' books wait for the decode. COLI_CUDA_CHAIN_LAYERWISE:
 * 0 chunk by chunk. */
/* layer i's routed experts for one chunk of a prompt run layer by layer: wait for the layer's
 * copy at its first chunk (first), start the next staged layer's after its last (last) */
static int q38cc_dm_layer_lm(Q38CChain *ch, Model *m, Layer *l, int i, int n, int first, int last) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, K = c->topk, io = i * ch->rows * K;
    CcBuf *tab = ch->dm_tabb;
    int stg = ch->dm_frn[i] > 0;
    CcRoute rp = {n, E, K, 0, 0, E, 0, 0, io, K, io, K, 1, 1, c->norm_topk ? 1 : 0, 1.f, 1.f, 0.f};
    CcMoe mp = {n, K, H, c->inter, ch->dm_fmt, ch->dm_gs, 0, 1, i * E, 0, H, io, K, io, K, 0, H, -1, E, 0, 0};
    if (ch->dm_fmt >= 100) qt_chain_layer_format(i, &mp.fmt, &mp.fmt_d);
    int ok = cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) && cc_route(ch->lg, NULL, ch->dm_ids, ch->dm_w, &rp) &&
             cc_copy(ch->dm_idsd, io, ch->dm_ids, io, (size_t)n * K) && cc_copy(ch->dm_wd, io, ch->dm_w, io, (size_t)n * K);
    ok = ok && q38cc_shared(ch, m, l, i, n) && (!stg || !first || cc_fetch_wait()) &&
         cc_moe(tab, ch->mixed, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y, NULL, &mp);
    if (ok && stg && last) ok = cc_fetch_done() && q38cc_stage_from(ch, m, i + 1);
    CcMoe ms = mp; ms.parts = 2; ms.h_off = -1;
    ok = ok && cc_moe(tab, NULL, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y, ch->dm_hy, &ms);
    ch->dm_steps++;
    return ok;
}
/* 1 done, 0 declined before any layer ran, -1 lost */
static int q38cc_forward_lm(Model *m, Q38CChain *ch, const int *ids, int S, int pos_base, int nlogits,
                            const float *hyper_h, float *logit, int rows) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, L = c->layers, E = c->experts, V = c->vocab, K = c->topk;
    int half = c->rotary_dim / 2;
    if (!q38cc_dm_begin(ch, m, rows)) return cc_lost() ? -1 : 0;
    if (q38cc_dm_tab(ch, rows) != ch->dm_tabb) return 0;   /* nothing stages: the chunks read it either way */
    for (int i = 0; i < L; i++) if (ch->dm_part[i]) return 0;   /* a hybrid layer: the chunks' host steps */
    const int SEG = (q38cc_lm_rows() / rows > 0 ? q38cc_lm_rows() / rows : 1) * rows, seg = S < SEG ? S : SEG;
    if (!cc_reserve(&ch->hyper_all, (size_t)seg * W * 4, CC_DEV)) return 0;
    float *cs_all = half ? malloc((size_t)S * half * 2 * sizeof(float)) : NULL;   /* every row's angles, once */
    if (half && !cs_all) return 0;
    for (int r = 0; half && r < S; r++) for (int j = 0; j < half; j++) {
        float ang = (float)(pos_base + r) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
        cs_all[((size_t)r * half + j) * 2] = cosf(ang); cs_all[((size_t)r * half + j) * 2 + 1] = sinf(ang);
    }
    for (int g0 = 0; g0 < S; g0 += SEG) {   /* a segment: its rows' streams up, the first staged layer's copy */
        int gn = S - g0 < SEG ? S - g0 : SEG;
        if (!q38cc_frame_begin(ch) || !cc_write(ch->hyper_all, 0, hyper_h + (size_t)g0 * W, (size_t)gn * W * 4) ||
            !q38cc_push_state(ch, m, pos_base + g0) || !q38cc_stage_from(ch, m, 0)) { free(cs_all); return -1; }
        for (int i = 0; i < L; i++) {
            Layer *l = &m->L[i];
            for (int c0 = g0; c0 < g0 + gn; c0 += rows) {
                int n = g0 + gn - c0 < rows ? g0 + gn - c0 : rows, pb = pos_base + c0;
                if (!q38cc_frame_begin(ch)) { free(cs_all); return -1; }
                if (half) memcpy(cc_ptr(ch->cs), cs_all + (size_t)c0 * half * 2, (size_t)n * half * 2 * sizeof(float));
                int ok = cc_copy(ch->hyper, 0, ch->hyper_all, (size_t)(c0 - g0) * W, (size_t)n * W);
                if (ok && i == c->ple_layer) { q38cc_ple_rows(m, ids, c0, n, ch->host_emb); ok = q38cc_ple(ch, m, n, c0, 0); }
                ok = ok && q38cc_gr_read(ch, m, &l->attn_gr, ch->o_an[i], n, ch->inj_a);
                ok = ok && (c->is_attn[i] ? q38cc_attention(ch, m, l, i, n, pb) : q38cc_deltanet(ch, m, l, i, n, c0, 0));
                ok = ok && q38cc_gr_apply(ch, m, ch->inj_a, n);
                ok = ok && q38cc_gr_read(ch, m, &l->mlp_gr, ch->o_mn[i], n, ch->inj_m);
                ok = ok && q38cc_dm_layer_lm(ch, m, l, i, n, c0 == g0, c0 + n >= g0 + gn) && q38cc_moe_apply(ch, m, n, ch->dm_y);
                ok = ok && cc_copy(ch->hyper_all, (size_t)(c0 - g0) * W, ch->hyper, 0, (size_t)n * W);
                int lo0 = S - nlogits > c0 ? S - nlogits - c0 : 0, lo_n = n - lo0;
                if (ok && i == L - 1 && lo0 < n && c0 + n > S - nlogits) {
                    int dst = c0 + lo0 - (S - nlogits);
                    ok = q38cc_gr_read(ch, m, &m->final_gr, ch->o_fn, n, NULL) &&
                         cc_matmul(q38cc_t(ch, &m->lm_head), ch->mixed, (size_t)lo0 * H, ch->outd, (size_t)dst * V, lo_n);
                }
                double t0 = now_s();
                ok = ok && cc_submit(1);   /* the frame's stream: the next staged layer's copy goes on beside the next frames */
                m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0; ch->wait_ms += (now_s() - t0) * 1e3;
                if (!ok) { free(cs_all); return -1; }
                /* layer i's books for these rows (q38cc_dm_books' for one layer) */
                q38cc_kv_down(ch, m, i, n, pb);
                const int *idv = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
                for (int s = 0; s < n; s++) { rt_count(i, idv + s * K, K); if (!ch->own) qt_chain_count(i, idv + s * K, K); }
                for (int j = 0; j < n * K; j++) if (idv[j] >= 0) {
                    q38_ehit_mark(m, i, idv[j]);
                    ch->routes++;
                    if (ch->dm_res[(size_t)i * E + idv[j]] >= 2) ch->pool_reads++;
                }
                if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
                ch->dn_where = Q38CC_DEV; ch->host_zero = 0;
            }
        }
    }
    free(cs_all);
    if (!cc_finish()) return -1;
    memcpy(logit, cc_ptr(ch->outd), (size_t)nlogits * V * sizeof(float));
    ch->lm_forwards++;
    return 1;
}

/* Every layer for S rows. hyper_h: the rows' streams in, the final ones out when
 * want_streams; mixed_h (or NULL): every row's final mixed row, for the prefill
 * read-out; logit: the last nlogits rows' logits. 0 = not taken (nothing changed, or
 * the device was lost and the state rebuilt: the CPU runs the step), 1 = done. */
static int q38cc_forward(Model *m, const int *ids, int S, int pos_base, int nlogits, float *hyper_h,
                         int want_streams, float *mixed_h, float *logit) {
    if (!g_cuda_chain || m->mux_rows) return 0;
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok || ch->failed) return 0;
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, L = c->layers, E = c->experts, V = c->vocab;
    int CH = q38cc_rows_env(), rows = S < CH ? S : CH;
    if (m->snap_rows > 1 && !q38cc_spec_slots(ch, m, m->snap_rows)) {
        fprintf(stderr, "[chain] qwen38: device memory for a verify's %d copies refused; this verify on the CPU\n", m->snap_rows);
        return 0;
    }
    if (!q38cc_mirror(ch, m) || !q38cc_scratch(ch, m, rows) || (want_streams && !cc_reserve(&ch->hypd, (size_t)rows * W * 4, CC_DOWN)) ||
        (nlogits > 2 && !cc_reserve(&ch->outd, (size_t)nlogits * V * 4, CC_DOWN)) ||
        (mixed_h && !cc_reserve(&ch->find, (size_t)rows * H * 4, CC_DOWN))) {
        fprintf(stderr, "[chain] qwen38: device memory for %d rows refused; the per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    int half = c->rotary_dim / 2;
    int snapped = 0;
    double w0 = ch->wait_ms, h0 = ch->host_ms;
    if (S > rows && rows > Q38CC_DM_ROWS && E > 0 && ch->dm_on && q38cc_layer_major() && !want_streams && !mixed_h &&
        m->snap_rows <= 0) {
        int r = q38cc_forward_lm(m, ch, ids, S, pos_base, nlogits, hyper_h, logit, rows);
        if (r < 0) goto lost;
        if (r > 0) { q38cc_dm_end(ch); ch->snap_valid = 0; ch->forwards++; return 1; }
    }
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        int ns = m->snap_rows - c0 < n ? m->snap_rows - c0 : n;   /* the rows of this chunk a verify copies */
        if (ns < 0) ns = 0;
        if (ns) snapped = c0 + ns;
        if (half) {
            float *cs = (float *)cc_ptr(ch->cs);
            for (int s = 0; s < n; s++) for (int j = 0; j < half; j++) {
                float ang = (float)(pb + s) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
                cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
            }
        }
        if (c->ple_layer >= 0 && c->ple_layer < L) q38cc_ple_rows(m, ids, c0, n, ch->host_emb);
        /* the routed experts in the frame (the table before the frame): a decode-sized step's every
         * layer, a prompt chunk's layers whose experts are all on the device or in the host pool (a
         * layer with some on the host keeps the host's step there) */
        int dm = E > 0 && ch->dm_on;
        if (dm && !q38cc_dm_begin(ch, m, n)) {
            if (cc_lost()) goto lost;
            fprintf(stderr, "[chain] qwen38: the routed experts' buffers or table refused; the host's step from here on\n");
            q38cc_dm_end(ch); ch->dm_on = dm = 0;
        }
        if (dm) memset(ch->dm_ran, 0, (size_t)L);
        if (!q38cc_frame_begin(ch) || !cc_write(ch->hyper, 0, hyper_h + (size_t)c0 * W, (size_t)n * W * 4) ||
            !q38cc_push_state(ch, m, pb)) goto lost;
        if (dm && q38cc_dm_tab(ch, n) == ch->dm_tabb && !q38cc_stage_from(ch, m, 0)) goto lost;   /* the first staged layer's copy */
        int ok = 1, pending = 0, pend_i = -1;   /* pend_i: the hybrid layer the host serves after recording the next */
        unsigned pend_v = 0;
        CcBuf *routed = ch->routed;              /* where the layer before left its routed sum */
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) ok = q38cc_moe_apply(ch, m, n, routed);
            if (ok && i == c->ple_layer) ok = q38cc_ple(ch, m, n, c0, ns);
            ok = ok && q38cc_gr_read(ch, m, &l->attn_gr, ch->o_an[i], n, ch->inj_a);
            ok = ok && (c->is_attn[i] ? q38cc_attention(ch, m, l, i, n, pb) : q38cc_deltanet(ch, m, l, i, n, c0, ns));
            ok = ok && q38cc_gr_apply(ch, m, ch->inj_a, n);
            ok = ok && q38cc_gr_read(ch, m, &l->mlp_gr, ch->o_mn[i], n, ch->inj_m);
            if (ok && dm && (n <= Q38CC_DM_ROWS || !ch->dm_part[i])) {   /* routed in the frame; a hybrid layer's host share once the next is recorded */
                unsigned v = 0;
                ch->dm_ran[i] = 1;
                ok = q38cc_dm_layer(ch, m, l, i, n, &v);
                if (ok && pend_v) ok = q38cc_dm_serve(ch, m, pend_i, n, pend_v);
                pend_i = i; pend_v = v; routed = ch->dm_y; pending = 1;
                continue;
            }
            ok = ok && cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) &&
                 cc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && cc_copy(ch->mixd, 0, ch->mixed, 0, (size_t)n * H);
            if (ok && pend_v) { ok = q38cc_dm_serve(ch, m, pend_i, n, pend_v); pend_v = 0; }   /* before a wait for the frame */
            double t0 = now_s();
            ok = ok && cc_submit(1);   /* A1 */
            m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0; ch->wait_ms += (now_s() - t0) * 1e3;
            if (!ok) break;
            q38cc_kv_down(ch, m, i, n, pb);
            ok = q38cc_frame_begin(ch) && q38cc_shared(ch, m, l, i, n) && cc_submit(0);   /* A2 */
            double t1 = now_s();
            q38_moe_ex(m, l, i, (const float *)cc_ptr(ch->mixd), n, ch->host_routed, (const float *)cc_ptr(ch->lgd), 1);
            memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)n * H * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && q38cc_frame_begin(ch);
            routed = ch->routed; pending = 1;
        }
        if (ok) ok = q38cc_moe_apply(ch, m, n, routed);
        if (ok && want_streams) ok = cc_copy(ch->hypd, 0, ch->hyper, 0, (size_t)n * W);
        int lo0 = S - nlogits > c0 ? S - nlogits - c0 : 0, lo_n = n - lo0;   /* this chunk's logit rows */
        int need_final = mixed_h || (lo0 < n && c0 + n > S - nlogits);
        if (ok && need_final) ok = q38cc_gr_read(ch, m, &m->final_gr, ch->o_fn, n, NULL);
        if (ok && mixed_h) ok = cc_copy(ch->find, 0, ch->mixed, 0, (size_t)n * H);
        if (ok && need_final && lo_n > 0 && c0 + n > S - nlogits) {
            int dst = c0 + lo0 - (S - nlogits);
            ok = cc_matmul(q38cc_t(ch, &m->lm_head), ch->mixed, (size_t)lo0 * H, ch->outd, (size_t)dst * V, lo_n);
        }
        double t0 = now_s();
        ok = ok && cc_submit(0);
        if (ok && pend_v) ok = q38cc_dm_serve(ch, m, pend_i, n, pend_v);   /* the last hybrid layer's, the tail recorded */
        ok = ok && cc_finish();
        m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0; ch->wait_ms += (now_s() - t0) * 1e3;
        if (!ok) goto lost;
        if (dm) q38cc_dm_books(ch, m, n, pb);
        if (want_streams) memcpy(hyper_h + (size_t)c0 * W, cc_ptr(ch->hypd), (size_t)n * W * 4);
        if (mixed_h) memcpy(mixed_h + (size_t)c0 * H, cc_ptr(ch->find), (size_t)n * H * 4);
        for (int i = 0; i < L; i++) if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
        ch->dn_where = Q38CC_DEV; ch->host_zero = 0;
    }
    q38cc_dm_end(ch);
    memcpy(logit, cc_ptr(ch->outd), (size_t)nlogits * V * sizeof(float));
    ch->snap_valid = snapped;
    ch->forwards++;
    if (S <= Q38_SPEC_ROWS) { ch->dec_n++; ch->dec_wait_ms += ch->wait_ms - w0; ch->dec_host_ms += ch->host_ms - h0; }   /* a decode step, or a verify's */
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!cc_lost()) cc_finish();
    q38cc_dm_end(ch);
    q38cc_recover(m, pos_base);
    return 0;
}

static void q38cc_report(Model *m) {
    Q38CChain *ch = m ? q38cc_of(m) : NULL;
    if (!ch || !ch->ok || !ch->forwards) return;
    CcStats st; cc_stats(&st);
    fprintf(stderr, "[chain] qwen38: %llu forwards, %llu frames (%llu ops, %llu matmuls), %.1f ms waiting for the device, "
                    "%.1f ms of routed experts on the host, %.1f MiB on the device; per decode step %.2f ms waiting, %.2f ms of experts (%llu steps)\n",
            ch->forwards, st.frames, st.ops, st.matmuls, ch->wait_ms, ch->host_ms, st.dev_bytes / 1048576.0,
            ch->dec_n ? ch->dec_wait_ms / ch->dec_n : 0.0, ch->dec_n ? ch->dec_host_ms / ch->dec_n : 0.0, ch->dec_n);
    if (ch->dm_steps)
        fprintf(stderr, "[chain] qwen38: %llu layer steps with the routed experts in the frame, %llu of them hybrid "
                        "(%llu slots on the host in %.1f ms, %.1f ms waiting for the device's routes), %llu slots from the host pool\n",
                ch->dm_steps, ch->dm_hyb, ch->dm_pairs, ch->dm_host_ms, ch->dm_spin_ms, ch->pool_reads);
    if (ch->pc_pairs)
        fprintf(stderr, "[chain] qwen38: %llu of a decode step's slots from the host pool on the CPU in %.1f ms (%.1f us a slot; "
                        "COLI_CUDA_CHAIN_POOL_READ=1: the device reads them over the bus)\n",
                ch->pc_pairs, ch->pc_ms, ch->pc_ms * 1e3 / (double)ch->pc_pairs);
    if (ch->dm_staged)
        fprintf(stderr, "[chain] qwen38: %llu prompt layer steps staged the host pool's experts (%.2f GB copied over the bus)\n",
                ch->dm_staged, ch->dm_stage_bytes / 1073741824.0);
    if (ch->lm_forwards)
        fprintf(stderr, "[chain] qwen38: %llu prompts layer by layer, each layer's host pool experts staged once "
                        "(COLI_CUDA_CHAIN_LAYERWISE=0: chunk by chunk)\n", ch->lm_forwards);
    if (ch->own)
        fprintf(stderr, "[chain] qwen38: %llu experts promoted to the card (%llu residents back to the pool), %d on their way; "
                        "%llu frames braked, %.1f %% of the routes on the card\n",
                ch->promos, ch->promo_victims, ch->npend, ch->pr_braked,
                ch->routes ? 100.0 * (double)(ch->routes - ch->pool_reads) / (double)ch->routes : 0.0);
}
