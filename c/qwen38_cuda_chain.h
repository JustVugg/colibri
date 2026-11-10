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
 *   - the MTP head's layer (index c.layers) runs here too where Q38_MTP=1 asks for the head
 *     (q38cc_mtp_rows, as qwen38_chain.h's q38c_mtp_rows): its input, its attention over its
 *     own K/V mirror (the last attention ordinal, its own RoPE base), the shared expert, the
 *     routed experts the host's step (q38_moe_ex: the head's own GGUF), the mixer and lm_head
 *     for the draft; Q38_MTP_CHAIN=0 keeps it on the CPU; a verify forward (S = 2 to Q38_SPEC_ROWS) runs row-wise (the
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

/* A prompt chunk in two lanes (q38cc_chunk_lanes): every buffer that holds a lane's rows between
 * the ops of one step and the next, the other lane's kept here while one lane's ops are recorded
 * (q38cc_lane_use swaps them in); the host's arrays of rows too. */
#define Q38CC_LANE_BUFS(X) X(hyper) X(hn) X(low) X(mix) X(mixed) X(inj_a) X(inj_m) X(blk) X(q) X(k) X(v) X(ip) X(ctx) \
    X(qsc) X(sel) X(qkv) X(z) X(ab) X(cv) X(dny) X(lg) X(gs) X(us) X(hs) X(ds) X(sgd) X(keys) X(val) X(gated) X(normv) \
    X(mixd) X(lgd) X(kvd) X(routed) X(emb) X(embd) X(cs) X(pcs) X(gseg) X(nseg) X(dm_ids) X(dm_w) X(dm_idsd) X(dm_wd) \
    X(dm_hid) X(dm_y) X(dm_hy) X(dm_dfl) X(dm_hfl) X(hyper_up)
typedef struct {
#define Q38CC_LB(f) CcBuf *f;
    Q38CC_LANE_BUFS(Q38CC_LB)
#undef Q38CC_LB
    float *host_routed, *host_emb;
    int rows;   /* the lane's rows: its buffers' strides (kvd's, the routes' by layer) */
} Q38CCLane;
typedef struct {
    int ok, failed, rows, cap, dev;
    int kvh;                          /* the K/V mirrors in float16 (q38cc_kvh) */
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
    /* the streams the hyper-connections read and write (q38cc_gr_read, q38cc_gr_apply, q38cc_ple):
     * hyper, or (hyp) a prompt chunk's rows of hyper_all from row hoff, in place */
    CcBuf *hyp; int hoff;
    /* a group of a segment's chunks whose routed experts run in one call (q38cc_moe_group): their mixed
     * rows, injections, routes and weights, the experts' sums; the groups, their chunks, those that
     * went chunk by chunk after all */
    CcBuf *mg_x, *mg_y, *mg_inj, *mg_ids, *mg_w; unsigned long long mg_groups, mg_chunks, mg_declined;
    CcBuf *hyper_up, *ones_at; int ones_n;     /* a step's streams on their way in (q38cc_hyper_in); the rows of ones written in ones_at */
    /* a lane chunk's layer's pooled experts lane A routes most, copied over for the card's call
     * (q38cc_lane_stage): ls_i the layer (-1 none), ls_e its ls_n experts (ls_mark by expert:
     * left out of the host's serves), ls_ent their entries in mapped memory (ls_cap of them and a
     * zero one), ls_rg the copies of the entries into the table and back */
    int ls_i, ls_n, ls_cap; int *ls_cnt, *ls_e, *ls_cand; uint8_t *ls_mark; CcFetch *ls_fr; CcRegion *ls_rg; CcBuf *ls_ent;
    unsigned long long ls_experts, ls_layers;
    CcBuf *ps; int ps_valid;   /* a prompt-end snapshot's DeltaNet states and PLE ring, copied down by the frame (q38cc_snap_down) */
    Q38CCLane lane2; int lane_cur, lane_rows;  /* a prompt chunk's other lane (q38cc_chunk_lanes), the lane whose buffers are in, the rows its host arrays hold */
    unsigned long long lane_chunks;
    CcBuf *qkv, *z, *ab, *cv, *dny, *lg, *gs, *us, *hs, *ds, *sgd;
    CcBuf *emb, *embd, *keys, *val, *gated, *normv;   /* emb: the PLE rows the host writes, embd: on the device */
    CcBuf *mixd, *lgd, *kvd, *outd, *hypd, *find, *routed, *cs, *pcs;
    CcBuf *outdev, *gpick;   /* a greedy caller's logits on the card and their picks down (q38cc_greedy_ok) */
    unsigned long long greedy_rows;
    CcBuf *gseg, *nseg;               /* a deep verify's PLE rows past its first copy (q38cc_ple) */
    /* the MTP head's layer (index c.layers) on the device (q38cc_mtp_rows): its norms in prm, the
     * embedding rows in (fin), a one per stream (ones), the rows and drafts it ran */
    int mtp_on; size_t o_mtp_emb, o_mtp_hid, o_mtp_mix; CcBuf *fin, *ones;
    /* the head's routed experts in RAM for the pool's server (q38cc_head_pool): gate, up and down of
     * expert e at hd_base + e hd_stride, as the pool lays them; hd_k0 = layers x experts, the head's
     * keys from there (q38cc_srv_where); hd_res: every one in the pool */
    uint8_t *hd_base, *hd_res; size_t hd_stride; int hd_tg, hd_td, hd_k0;
    CcBuf *hd_ids, *hd_w, *hd_idsd, *hd_wd; unsigned long long hd_rows;
    /* or on the card in Q2_0 (q38cc_head_card): hd_dev their bytes, hd_tab their entries */
    CcBuf *hd_dev, *hd_tab; unsigned long long hd_card_rows;
    CcBuf *dv_ids, *dv_y, *dv_x, *dv_pick; unsigned long long dv_drafts;   /* the draft list scored on the card */
    unsigned long long mtp_rows, drafts;
    float *host_routed, *host_emb;
    unsigned long long forwards, frames, fault_at, dec_n;
    double host_ms, wait_ms, dec_wait_ms, dec_host_ms;
    /* the routed experts in the frame (q38cc_dm_*) */
    int dm_on, dm_fmt, dm_gs, dm_read, dm_held;
    unsigned dm_epoch, dm_seq;
    CcExpert *dm_host; uint8_t *dm_res, *dm_ran, *dm_dev; int *dm_part;   /* dm_ran: the step's layers that ran so */
    CcBuf *dm_ids, *dm_w, *dm_idsd, *dm_wd, *dm_hid, *dm_y, *dm_hy, *dm_dfl, *dm_hfl, *dm_tab;
    CcBuf *dm_tabu, *dm_tabdu;   /* dm_tab and dm_tabd in mapped memory: a settle's changed entries copied from there (q38cc_promote_settle) */
    /* pool_cpu: a decode step's pooled experts on the CPU (q38cc_dm_serve), from the decode table
     * dm_tabd that lacks them (dm_partd: a layer with any such slot); pc_*: the CPU's scratch */
    CcBuf *dm_tabd; int *dm_partd, pool_cpu, pc_cap; float *pc_gu, *pc_hh, *pc_dn;
    unsigned long long pc_pairs; double pc_ms;
    struct Q38CCSrv *srv; unsigned long long sv_jobs, sv_slots, sv_ns;   /* the pool's server (q38cc_srv_*) and its books merged so far */
    /* pc_prompt: a short prompt's chunks take the decode table too, their pooled slots on the CPU
     * by expert (q38cc_pool_groups) instead of every pooled expert staged over the bus; pg_*: the
     * scratch (slots in expert order, their gathered rows, an activation per expert) */
    int pc_prompt, dm_dev_cap, pg_cap, pg_ecap, pg_qcap, pg_scap; int *pg_a, *pg_pos, *pg_start, *pg_o; const uint8_t **pg_g;
    float *pg_x, *pg_gu, *pg_hh, *pg_dn; gq_act *pg_ax, *pg_ah;
    unsigned long long pg_pairs, pg_groups; double pg_ms;
    /* the choice (q38cc_pc_take): the bus's rate and a pooled slot's CPU time as measured at the
     * pool's start (q38cc_bus_probe, q38cc_cpu_probe), the routes the pool takes by the history, the bytes a chunk
     * stages; the prompts that took the CPU and those that staged */
    double pc_bus_gbs, pc_slot_us, pc_frac0, pc_stage_bytes;
    int pc_team;                         /* the grouped kernel's team the probe kept */
    unsigned long long pc_took, pc_staged;
    unsigned long long dm_steps, dm_hyb, dm_pairs;   /* layer steps in the frame, the hybrid ones, slots the host computed */
    double dm_host_ms, dm_spin_ms;
    /* the host pool (q38cc_pool_init): pool_n experts in pool_nb buffers of pool_per, pool_of[l*E+e] its slot or -1 */
    CcBuf **pool_buf; int pool_nb, pool_per, pool_n; size_t pool_stride; int32_t *pool_of;
    /* the pool packed at each layer's own expert size (NULL: places of the largest layer's): a
     * place's block and offset in it, each layer's expert bytes there and in a staging */
    int32_t *pool_bi; uint32_t *pool_bo; size_t *pool_lst;
    CcBuf *pool_stage;               /* a decode step's pooled experts copied here first (cc_moe_stage) */
    /* a prompt chunk's: a layer's pooled experts copied to dm_stage (cc_fetch) while the layers before it run */
    CcBuf *dm_stage, *dm_tabb; CcExpert *dm_hostb; CcFetch *dm_fr; int *dm_fro, *dm_frn, dm_stage_cap;
    size_t *dm_iv;                       /* layer i's places in dm_stage: bytes [iv[4i], iv[4i+1]) and [iv[4i+2], iv[4i+3]) (q38cc_stage_table) */
    unsigned long long dm_early, dm_early_bytes;   /* the copies started beside the layer before's experts (q38cc_stage_part) */
    size_t dm_stage_size;                /* dm_stage's bytes */
    unsigned long long dm_staged; double dm_stage_bytes;
    unsigned long long lm_forwards;            /* prompts run layer by layer */
    /* adaptive promotion (q38cc_promote_*): the tier's arena is the chain's (own) */
    int own, promo_rate, stage_dirty;
    char *ar_base; size_t ar_slot, ar_off_u, ar_off_d; int ar_n;
    /* the exact arena (qt_chain_arena_exact; ar_soff NULL: slots of one size): a slot's byte offset
     * (ar_cap at most), each layer's slot bytes, gate/up/down offsets and size class, a class's slots,
     * the bytes carved (ar_bump) of the arena's */
    size_t *ar_soff, ar_bump, ar_bytes; const size_t *ar_lsl, *ar_lou, *ar_lod; const int *ar_lcls;
    int ar_cap, ar_ncls, **ar_cls, *ar_cn, *ar_cc;
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
/* layer i is an attention layer: the MTP head's (i == c.layers) is */
static int q38cc_attn_at(const Cfg *c, int i) { return i < c->layers ? c->is_attn[i] != 0 : 1; }
/* the MTP head's layer beside the chain: the model has the head, Q38_MTP=0 does not turn it off and
 * Q38_MTP_CHAIN=0 does not keep it on the CPU (unset, the engine keeps its drafts where the head's
 * routed experts reached the card too: q38cc_head_on_card) */
static int q38cc_mtp_planned(Model *m) {
    const char *e = getenv("Q38_MTP"), *k = getenv("Q38_MTP_CHAIN");
    return m->mtp && !(e && *e == '0') && !(k && *k == '0');
}
static int q38cc_dm_able(Q38CChain *ch, Model *m);
static int q38cc_head_card(Q38CChain *ch, Model *m);
/* the MTP head's layer and its routed experts on the card (q38cc_head_card) */
static int q38cc_head_on_card(Model *m) { Q38CChain *ch = q38cc_of(m); return ch && ch->ok && ch->mtp_on && ch->hd_tab; }
static int g_cuda_chain = 0;     /* COLI_CUDA_CHAIN asked for it and the chain is up */
static unsigned long long g_q38cc_mtp_rows;   /* the MTP head's rows the chain ran, every model (tests) */

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
/* a prompt of up to this many rows may compute its pooled experts on the CPU (q38cc_pool_groups)
 * while the card computes its own (q38cc_pc_take decides): COLI_CUDA_CHAIN_POOL_CPU_ROWS sets
 * it, 0 stages them all over the bus as a longer prompt's chunks do */
#define Q38CC_PC_ROWS 2048
/* the CPU where its estimate is at most this many times the bus's copy: its share overlaps the
 * card's own experts, the copy the layers before. Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000
 * (PCIe 4.0 x8, 11.8 GB/s, 14 cores): at 1912 rows the CPU's estimate was 1.41 times the copy's
 * and the prompt still 3.5% faster (418 -> 433 tok/s), at 452 rows 214 -> 406 */
#define Q38CC_PC_BUS 1.5
static int q38cc_pc_rows(void) {
    static int r = -1;
    if (r < 0) { const char *e = getenv("COLI_CUDA_CHAIN_POOL_CPU_ROWS"); r = e && *e ? atoi(e) : Q38CC_PC_ROWS; if (r < 0) r = 0; }
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
    if (q38cc_attn_at(c, i)) { nm[n] = "attnq"; w[n++] = &l->q; nm[n] = "attnk"; w[n++] = &l->k; nm[n] = "attnv"; w[n++] = &l->v;
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
/* a prompt chunk's staging of a layer's pooled experts, bytes at most: 512 MiB of the chain's reserve
 * and what the expert budget gives it past them (qt_reserve_stage); COLI_CUDA_CHAIN_PROMPT_STAGE_MB
 * sets it, none from the budget then */
static size_t q38cc_stage_mb_env(void) {
    static long v = -1;
    if (v < 0) { const char *e = getenv("COLI_CUDA_CHAIN_PROMPT_STAGE_MB"); v = e && atol(e) > 0 ? atol(e) : 0; }
    return (size_t)v << 20;
}
#define Q38CC_PROMPT_STAGE (q38cc_stage_mb_env() ? q38cc_stage_mb_env() : (size_t)512 << 20)
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
/* The MTP head's routed experts on the card in Q2_0 (q38cc_head_card), COLI_CUDA_CHAIN_MTP_CARD=0
 * off: the head's file has them in Q8_0 (2.49 GiB for Qwen3.8's 512), and a draft's ten from RAM
 * kept the card waiting 1.9 ms for the CPU (an i5-13500, 16 threads) against 0.6 ms of the
 * head's own work. A draft is only a guess the verify judges: its experts' precision moves how
 * often a draft passes, never the text. Gate, up and down at 0, Q38CC_PAD(gate), twice that. */
static int q38cc_head_card_on(Model *m) {
    const char *e = getenv("COLI_CUDA_CHAIN_MTP_CARD");
    if ((e && *e == '0') || !q38cc_mtp_planned(m) || !g_q38_gguf || m->c.experts <= 0 || m->c.hidden % 64 || m->c.inter % 64 ||
        !cc_moe_block_type(GQ_Q2_0))
        return 0;
    const gguf_tensor *g[3]; q38_gg_expert_tensors(m, m->c.layers, g);
    for (int q = 0; q < 3; q++) if (!g[q] || !gq_has_int(g[q]->type)) return 0;
    return 1;
}
static size_t q38cc_head_card_stride(const Cfg *c) {
    return 2 * Q38CC_PAD(gq_row_bytes(GQ_Q2_0, c->hidden) * c->inter) + Q38CC_PAD(gq_row_bytes(GQ_Q2_0, c->inter) * c->hidden);
}
/* the K/V mirrors in float16: the rows rounded to it (q38_kv16) and a backend that reads such a
 * cache (cc_attn_h); a mirror holds their values exactly, in half the device memory */
static int q38cc_kvh(void) { return q38_kv16() && cc_kvh_ready(); }
/* a row's floats in the chain's buffers (q38cc_bufs), positions up to cap */
static size_t q38cc_row_floats(const Cfg *c, size_t cap) {
    size_t W = c->hc_width, H = c->hidden, QH = c->q_heads, D = c->head_dim, V = (size_t)c->dn_vheads * c->dn_vdim, E = c->experts;
    size_t SI = c->shared_inter, ipw = (size_t)(c->idx_qheads + 1) * c->idx_dim, Ep = (size_t)c->ngram_heads * c->ngram_head_dim;
    size_t kvo = (size_t)c->kv_heads * c->head_dim, nb = cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    return 7 * W + c->hc_rank + 6 * H + 2 * (size_t)c->hc_count + 3 * QH * D + 2 * kvo + ipw + 2 * nb + (size_t)q38cc_selrow(c) +
           2 * (size_t)c->dn_conv_dim + 2 * V + 2 * (size_t)c->dn_vheads + 2 * E + 3 * SI + 2 * Ep /* emb, embd */ +
           (size_t)(c->rotary_dim > 0 ? c->rotary_dim : 2);
}
static size_t q38cc_reserve_bytes(Model *m) {
    Cfg *c = &m->c; int L = c->layers, n_attn = 0, head = q38cc_mtp_planned(m);
    for (int i = 0; i < L; i++) n_attn += c->is_attn[i] != 0;
    n_attn += head;   /* the head's K/V mirror */
    int ctx = m->kv_cap > 0 ? m->kv_cap : m->max_t > 0 ? m->max_t : qwen38_max_ctx();   /* the Vulkan chain plans the same context */
    if (c->max_positions > 0 && ctx > c->max_positions) ctx = c->max_positions;
    size_t F = 4, cap = (size_t)(ctx > 0 ? ctx : 1), kvo = (size_t)c->kv_heads * c->head_dim, idx = (size_t)c->idx_dim;
    size_t nb = cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1, b = 0;
    b += (size_t)n_attn * (2 * kvo * cap * (q38cc_kvh() ? 2 : F) + (idx * cap + idx * nb) * F);
    b += (size_t)(L + head - n_attn) * 2 * ((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim + (size_t)c->dn_conv_dim * (c->dn_convk - 1)) * F;
    size_t W = c->hc_width, H = c->hidden, E = c->experts;
    b += (size_t)q38cc_rows_env() * q38cc_row_floats(c, cap) * F + 2 * (size_t)c->vocab * F;
    /* a verify's copies of the DeltaNet state and rings (q38cc_spec_slots), the routed experts'
     * scratch for a prompt chunk (q38cc_dm_begin: their hidden rows, the sum, the routes), and a
     * segment's streams of a prompt run layer by layer: past the estimate, 2048-row chunks left
     * the chain short of the copies, and every verify went to the CPU (Qwen3.8-Flash-Next IQ2_XS,
     * RTX PRO 4000: 3.21 GiB kept, 3.34-3.51 used, decode 43.6 -> 20.4 tok/s) */
    {
        size_t R = (size_t)q38cc_rows_env(), K = (size_t)c->topk;
        b += (size_t)Q38_SPEC_SNAPS * (L + head - n_attn) *
             ((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim + (size_t)c->dn_conv_dim * (c->dn_convk - 1)) * F;
        if (E > 0) b += (cc_moe_hid_floats((int)R, (int)K, (int)H, c->inter, (int)E) + R * H + 2 * (size_t)L * R * K) * F;
        if (q38cc_layer_major()) b += (size_t)(q38cc_lm_rows() > (int)R ? q38cc_lm_rows() : (int)R) * W * F;
        if (q38cc_layer_major() && E > 0) {   /* a group of chunks' rows (q38cc_moe_group_rows): as many as dm_hid holds with 512 of down's outputs */
            size_t hc = c->hidden < 512 ? (size_t)c->hidden : 512, G = R * ((size_t)c->inter + H) / ((size_t)c->inter + hc);
            if (G > (size_t)q38cc_lm_rows()) G = (size_t)q38cc_lm_rows();
            b += G * (2 * H + (size_t)c->hc_count + 2 * K) * F;
        }
    }
    for (int i = 0; i < L + head; i++) {   /* the matrices the trunk's table does not offer: the chain's own copies */
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n; k++) {
            int offered = 0;
            for (int j = 0; j < g_trunk_n && !offered; j++) offered = g_trunk[j].w == w[k] && !g_trunk[j].cpu_only;
            if (offered) continue;
            int gf = q38_cuda_gq_fmt(w[k]);
            const int ple16 = (!strcmp(nm[k], "plekey") || !strcmp(nm[k], "plevalue")) && w[k]->kind == Q38_WEIGHT_BF16;   /* q38cc_tb */
            b += gf ? (size_t)w[k]->rows * (((size_t)gq_row_bytes(gf - 100, w[k]->cols) + 3) / 4 * 4)
                    : (size_t)w[k]->rows * w[k]->cols * (ple16 ? 2 : 4);   /* a small BF16/F32 matrix goes up as f32 */
        }
        b += H * F;   /* the shared expert's gate row */
    }
    if (head) {   /* the head's own: fc_embedding, fc_hidden, the mixer's pair */
        Q38Weight *hw[] = {&m->mtp_fc_emb, &m->mtp_fc_hid, &m->mtp_mixer.down, &m->mtp_mixer.up};
        for (size_t k = 0; k < sizeof hw / sizeof *hw; k++) {
            int offered = 0;
            for (int j = 0; j < g_trunk_n && !offered; j++) offered = g_trunk[j].w == hw[k] && !g_trunk[j].cpu_only;
            if (!offered) b += (size_t)hw[k]->rows * hw[k]->cols * 4;
        }
    }
    if (head && q38cc_head_card_on(m)) b += (size_t)c->experts * q38cc_head_card_stride(c);   /* the head's experts in Q2_0 */
    /* a sixteenth more: the chain's buffers came to 3.92 GiB against 4.31 estimated (Qwen3.8 IQ2_XS, 65536 positions) */
    return b + b / 16 + q38cc_prompt_stage_bytes(m);
}
static void q38cc_free_own(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers;
    for (int i = 0; i < L + (ch->mtp_on ? 1 : 0); i++) {
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n; k++) if (w[k]->cc) { coli_cuda_tensor_free((ColiCudaTensor *)w[k]->cc); w[k]->cc = NULL; }
        if (ch->t_sg && ch->t_sg[i]) { coli_cuda_tensor_free(ch->t_sg[i]); ch->t_sg[i] = NULL; }
    }
    Q38Weight *tail[] = {&m->final_gr.down, &m->final_gr.up, &m->lm_head, &m->mtp_fc_emb, &m->mtp_fc_hid, &m->mtp_mixer.down, &m->mtp_mixer.up};
    for (size_t k = 0; k < sizeof tail / sizeof *tail; k++) if (tail[k]->cc) { coli_cuda_tensor_free((ColiCudaTensor *)tail[k]->cc); tail[k]->cc = NULL; }
    CcBuf ***arrs[] = {&ch->rec, &ch->ring, &ch->kc, &ch->vc, &ch->ik, &ch->pk};
    for (size_t a = 0; a < sizeof arrs / sizeof *arrs; a++) if (*arrs[a]) for (int i = 0; i <= L; i++) { cc_free((*arrs[a])[i]); (*arrs[a])[i] = NULL; }
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
    if (ch->mtp_on) {   /* the head's layer, the last attention ordinal, and the head's norms */
        int i = L;
        ch->o_an[i] = n; n += W; ch->o_mn[i] = n; n += W;
        ch->attn_ord[i] = ch->n_attn++;
        ch->o_qn[i] = n; n += c->head_dim; ch->o_kn[i] = n; n += c->head_dim;
        ch->o_iqn[i] = n; n += c->idx_dim; ch->o_ikn[i] = n; n += c->idx_dim;
        ch->o_mtp_emb = n; n += c->hidden; ch->o_mtp_hid = n; n += W; ch->o_mtp_mix = n; n += W;
    }
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
    if (ch->mtp_on) {
        Layer *l = &m->L[L];
        memcpy(a + ch->o_an[L], l->attn_gr.norm, W * sizeof(float)); memcpy(a + ch->o_mn[L], l->mlp_gr.norm, W * sizeof(float));
        memcpy(a + ch->o_qn[L], l->qn, c->head_dim * sizeof(float)); memcpy(a + ch->o_kn[L], l->kn, c->head_dim * sizeof(float));
        memcpy(a + ch->o_iqn[L], l->idx_qn, c->idx_dim * sizeof(float)); memcpy(a + ch->o_ikn[L], l->idx_kn, c->idx_dim * sizeof(float));
        memcpy(a + ch->o_mtp_emb, m->mtp_norm_emb, (size_t)c->hidden * sizeof(float));
        memcpy(a + ch->o_mtp_hid, m->mtp_norm_hid, W * sizeof(float));
        memcpy(a + ch->o_mtp_mix, m->mtp_mixer.norm, W * sizeof(float));
    }
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

static ColiCudaTensor *q38cc_tb(Q38CChain *ch, Q38Weight *w);
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
#define Q38CC_ARR(f, t) if (!(ch->f = (t *)calloc((size_t)L + 1, sizeof(t)))) return NULL;   /* the head's layer at L */
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
    ch->kvh = q38cc_kvh();
    if (q38_kv16() && !ch->kvh && !cc_kv16_ready()) {
        fprintf(stderr, "[chain] qwen38: the CUDA backend neither holds a float16 cache nor rounds to one (Q38_KV=f16); the chain stays off\n");
        return NULL;
    }
    size_t own = 0, tier = 0;
    for (int i = 0; i < L; i++) {
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, i, w, nm);
        for (int k = 0; k < n; k++) {
            const int ple = !strcmp(nm[k], "plekey") || !strcmp(nm[k], "plevalue");   /* as BF16 rows (q38cc_tb) */
            if (!(ple ? q38cc_tb(ch, w[k]) : q38cc_t(ch, w[k]))) {
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
    if (q38cc_mtp_planned(m)) {   /* the head's layer: its matrices, its shared gate row, the head's own */
        Q38Weight *w[Q38CC_MATS]; const char *nm[Q38CC_MATS]; int n = q38cc_mats(m, L, w, nm), ok = 1;
        for (int k = 0; k < n && ok; k++) ok = q38cc_t(ch, w[k]) != NULL;
        Q38Weight *hw[] = {&m->mtp_fc_emb, &m->mtp_fc_hid, &m->mtp_mixer.down, &m->mtp_mixer.up};
        for (size_t k = 0; k < sizeof hw / sizeof *hw && ok; k++) ok = q38cc_t(ch, hw[k]) != NULL;
        ok = ok && (ch->t_sg[L] = q38cc_f32_rows(ch->dev, m->L[L].sh_gate, c->hidden, 1)) != NULL;
        ch->mtp_on = ok;
        if (!ok) fprintf(stderr, "[chain] qwen38: the MTP head's matrices did not reach device %d; the head runs on the CPU\n", ch->dev);
        else fprintf(stderr, "[chain] qwen38: the MTP head's layer on device %d too (its routed experts the host's step; "
                             "Q38_MTP_CHAIN=0: the head on the CPU)\n", ch->dev);
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
    if (ch->dm_on) q38cc_head_card(ch, m);   /* the MTP head's experts in Q2_0 on the card */
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
             cc_reserve(&ch->emb, r * Ep * F, CC_UP) && cc_reserve(&ch->embd, r * Ep * F, CC_DEV) &&
             cc_reserve(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_UP) &&
             cc_reserve(&ch->pcs, (size_t)nbmax * (c->rotary_dim > 0 ? c->rotary_dim : 2) * F, CC_UP);
    /* a verify's PLE segments (q38cc_ple) here, not in the frame: growing a buffer there waits
     * for the stream, which a hybrid layer before it holds until the host serves it */
    if (ok && c->ple_layer >= 0)
        ok = cc_reserve(&ch->gseg, r * W * F, CC_DEV) && cc_reserve(&ch->nseg, r * W * F, CC_DEV);
    return ok;
}
/* n of the host's K or V values from src into a mirror at element at (a float16 mirror takes
 * them as halves: the values the rounding left, exact) */
static int q38cc_kv_up(const Q38CChain *ch, CcBuf *dst, size_t at, const float *src, size_t n) {
    if (!ch->kvh) return cc_write(dst, at, src, n * sizeof(float));
    uint16_t *h = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!h) return 0;
    kv16_pack(h, src, (int64_t)n);
    int ok = cc_write(dst, at / 2, h, n * sizeof(uint16_t));   /* at even: a head's rows start on whole floats */
    free(h);
    return ok;
}
/* every attention layer's mirrors off the card (the host's K/V rows stay canonical) */
static void q38cc_mirror_free(Q38CChain *ch, const Cfg *c) {
    for (int i = 0; i < c->layers + (ch->mtp_on ? 1 : 0); i++) {
        if (!q38cc_attn_at(c, i)) continue;
        cc_free(ch->kc[i]); cc_free(ch->vc[i]); cc_free(ch->ik[i]); cc_free(ch->pk[i]);
        ch->kc[i] = ch->vc[i] = ch->ik[i] = ch->pk[i] = NULL;
        ch->kv_valid[i] = ch->pk_valid[i] = 0;
    }
    ch->cap = 0;
}
/* the K/V, index-key and pooled-key mirrors at the host's capacity, every layer's or none: a
 * refused one frees the layers already placed, which the chain, off from then on, never uses */
static int q38cc_mirror(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int kvo = c->kv_heads * c->head_dim;
    int nbmax = m->kv_cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    if (ch->cap != m->kv_cap) {
        for (int i = 0; i < c->layers + (ch->mtp_on ? 1 : 0); i++) {
            if (!q38cc_attn_at(c, i)) continue;
            cc_free(ch->kc[i]); cc_free(ch->vc[i]); cc_free(ch->ik[i]); cc_free(ch->pk[i]);
            ch->kv_valid[i] = ch->pk_valid[i] = 0;
            ch->kc[i] = cc_buf((size_t)kvo * m->kv_cap * (ch->kvh ? 2 : 4), CC_DEV);
            ch->vc[i] = cc_buf((size_t)kvo * m->kv_cap * (ch->kvh ? 2 : 4), CC_DEV);
            ch->ik[i] = cc_buf((size_t)c->idx_dim * m->kv_cap * 4, CC_DEV);
            ch->pk[i] = cc_buf((size_t)c->idx_dim * nbmax * 4, CC_DEV);
            if (!ch->kc[i] || !ch->vc[i] || !ch->ik[i] || !ch->pk[i]) { q38cc_mirror_free(ch, c); return 0; }
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
/* A prompt-end snapshot of the state the card holds: its DeltaNet states and PLE ring copied
 * into mapped memory by the stream itself, nothing waited for; q38cc_snap_take hands them over
 * once they are needed (a later prompt's restore, a parked conversation). The host's copies
 * after the prompt's last frame came to 25 ms before its first token (a 470-token prompt of
 * Qwen3.8-Flash-Next on an RTX PRO 4000: 36 layers' states, 118 MB, read one by one, then
 * copied again into the cache). 0: not taken (the host's copy then). */
/* a snapshot's floats: every DeltaNet layer's state and conv ring, then the PLE ring */
static size_t q38cc_snap_floats(const Q38CChain *ch, const Model *m) {
    const Cfg *c = &m->c;
    const size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    size_t total = ch->ple_ring && m->PLE_conv_state ? q38cc_ple_cells(c) : 0;
    for (int i = 0; i < c->layers; i++) if (!c->is_attn[i]) total += nr + nc;
    return total;
}
static int q38cc_snap_down(Model *m) {
    Q38CChain *ch = g_cuda_chain ? q38cc_of(m) : NULL;
    if (!ch || !ch->ok || ch->failed || ch->dn_where != Q38CC_DEV) return 0;
    Cfg *c = &m->c;
    const size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    const size_t np = ch->ple_ring && m->PLE_conv_state ? q38cc_ple_cells(c) : 0;
    size_t o = 0;
    ch->ps_valid = 0;
    if (!cc_reserve(&ch->ps, q38cc_snap_floats(ch, m) * 4, CC_DOWN)) return 0;
    int ok = cc_begin();
    for (int i = 0; i < c->layers && ok; i++) {
        if (c->is_attn[i]) continue;
        ok = cc_copy(ch->ps, o, ch->rec[i], 0, nr) && cc_copy(ch->ps, o + nr, ch->ring[i], 0, nc);
        o += nr + nc;
    }
    if (ok && np) ok = cc_copy(ch->ps, o, ch->ple_ring, (size_t)ch->ple_slot[0] * np, np);
    ok = ok && cc_submit(0);
    if (!ok) { if (cc_lost()) q38cc_recover(m, m->kv_len); return 0; }
    ch->ps_valid = 1;
    return 1;
}
/* the snapshot q38cc_snap_down took, into the cache's arrays (a DeltaNet layer's recurrent state
 * and conv ring, the PLE ring): 0 where there is none or the device is gone */
static int q38cc_snap_take(Model *m, float **rec, float **conv, float *ple) {
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ps_valid || !cc_finish()) return 0;
    Cfg *c = &m->c;
    const size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    const float *p = (const float *)cc_ptr(ch->ps);
    size_t o = 0;
    for (int i = 0; i < c->layers; i++) {
        if (c->is_attn[i]) continue;
        memcpy(rec[i], p + o, nr * 4); memcpy(conv[i], p + o + nr, nc * 4);
        o += nr + nc;
    }
    if (ple && ch->ple_ring && m->PLE_conv_state) memcpy(ple, p + o, q38cc_ple_cells(c) * 4);
    ch->ps_valid = 0;
    return 1;
}
/* the chain takes a prompt of n rows in one chunk: a served prompt's stop lands before it or not
 * at all (q38cc_forward checks it between chunks), so its state needs no copy to come back to */
static int q38cc_one_chunk(Model *m, int n) {
    const Q38CChain *ch = g_cuda_chain ? q38cc_of(m) : NULL;
    return ch && ch->ok && !ch->failed && n <= q38cc_rows_env();
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
/* attention layer i's K/V and index-key rows the host has and the mirror lacks, up to pos_base */
static int q38cc_push_kv(Q38CChain *ch, Model *m, int i, int pos_base) {
    Cfg *c = &m->c; int D = c->head_dim, KVH = c->kv_heads, ID = c->idx_dim, R = c->idx_ratio, ok = 1;
    if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;   /* blocks this step rewrites */
    if (ch->kv_valid[i] >= pos_base) return 1;
    int t0 = ch->kv_valid[i], n = pos_base - t0;
    for (int h = 0; h < KVH && ok; h++) {
        size_t src = ((size_t)h * m->kv_cap + t0) * D, dst = ((size_t)h * ch->cap + t0) * D;
        ok = q38cc_kv_up(ch, ch->kc[i], dst, m->K[i] + src, (size_t)n * D) && q38cc_kv_up(ch, ch->vc[i], dst, m->V[i] + src, (size_t)n * D);
    }
    ok = ok && cc_write(ch->ik[i], (size_t)t0 * ID, m->IK[i] + (size_t)t0 * ID, (size_t)n * ID * 4);
    ch->kv_valid[i] = pos_base;
    if (ch->pk_valid[i] > t0 / R) ch->pk_valid[i] = t0 / R;
    return ok;
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
    for (int i = 0; i < c->layers && ok; i++) if (c->is_attn[i]) ok = q38cc_push_kv(ch, m, i, pos_base);
    return ok;
}

/* ---- pieces ---------------------------------------------------------------------- */
static CcBuf *q38cc_hb(const Q38CChain *ch) { return ch->hyp ? ch->hyp : ch->hyper; }
static int q38cc_ho(const Q38CChain *ch, int W) { return ch->hyp ? ch->hoff * W : 0; }
static int q38cc_gr_read(Q38CChain *ch, Model *m, GatedResidual *g, size_t norm_off, int n, CcBuf *inj) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    CcNorm np = {n * C, H, C, q38cc_ho(ch, W), W, H, 0, W, H, (int)norm_off, C, CC_NORM_ADD1, c->eps, 1.f};
    CcEw lo = {CC_EW_HC_LOW, n * R, R, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    CcEw mx = {CC_EW_HC_MIX, n * H, H, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    int ok = cc_norm(q38cc_hb(ch), ch->prm, ch->hn, &np) &&
             cc_matmul(q38cc_t(ch, &g->down), ch->hn, 0, ch->low, 0, n) && cc_ew(ch->low, ch->low, NULL, NULL, NULL, &lo);
    /* the up projection and the mix in one pass where the backend takes them (cc_matmul_hcmix: the same
     * bits, the product's n rows of W never written), else the two ops */
    const int fused = ok ? cc_matmul_hcmix(q38cc_t(ch, &g->up), ch->low, 0, ch->hn, ch->mixed, n, H, C, (float)C) : 0;
    if (fused == 0) ok = 0;
    else if (fused < 0) ok = cc_matmul(q38cc_t(ch, &g->up), ch->low, 0, ch->mix, 0, n) && cc_ew(ch->mixed, ch->mix, ch->hn, NULL, NULL, &mx);
    if (ok && inj) {
        CcEw ij = {CC_EW_HC_INJ, n * C, C, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
        ok = cc_matmul(q38cc_t(ch, &g->inject), ch->hn, 0, inj, 0, n) && cc_ew(inj, inj, NULL, NULL, NULL, &ij);
    }
    return ok;
}
static int q38cc_gr_apply(Q38CChain *ch, Model *m, CcBuf *inj, int n) {
    Cfg *c = &m->c;
    CcEw ap = {CC_EW_HC_APPLY, n * c->hc_width, c->hidden, c->hc_count, 0, 1, q38cc_ho(ch, c->hc_width), 0, 0, 0, 0, 1.f};
    return cc_ew(q38cc_hb(ch), inj, ch->blk, NULL, NULL, &ap);
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
    if (ok && q38_kv16()) {   /* K and V as every backend keeps them (float16), before anything reads them */
        CcEw r = {CC_EW_F16, n * kvo, kvo, 0, 0, 0, 0, 0, 0, 0, 0, 1.f};
        ok = cc_ew(ch->k, ch->k, NULL, NULL, NULL, &r) && cc_ew(ch->v, ch->v, NULL, NULL, NULL, &r);
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
    ok = (ch->kvh ? cc_copy_regions_h(ch->kc[i], ch->k, rg, n * KVH) && cc_copy_regions_h(ch->vc[i], ch->v, rg, n * KVH)
                  : cc_copy_regions(ch->kc[i], ch->k, rg, n * KVH) && cc_copy_regions(ch->vc[i], ch->v, rg, n * KVH)) &&
         cc_copy_regions(ch->ik[i], ch->ip, ri, n) &&
         cc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) &&
         cc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    if (ok) {   /* index keys back, compact: one copy over the rows (a copy a row filled the driver's queue while
                 * a hybrid layer before held the stream, and the host waited on it before serving that layer) */
        for (int s = 0; s < n; s++) ri[s] = (CcRegion){ko + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, (size_t)s * ipw + (size_t)IQ * ID, (size_t)ID};
        ok = cc_copy_regions(ch->kvd, ch->ip, ri, n);
    }
    free(rg); free(ri);
    /* the pooled keys of the blocks this step completes (and any behind the watermark) */
    int nb = (pb + n) / R, b0 = ch->pk_valid[i];
    if (ok && nb > b0) {
        float *pc = (float *)cc_ptr(ch->pcs);
        if (half) for (int b = b0; b < nb; b++) for (int j = 0; j < half; j++) {
            float ang = (float)(b * R) / powf(i < c->layers ? c->theta : c->mtp_theta, (float)(2 * j) / c->rotary_dim);   /* the head's own base */
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
           (ch->kvh ? cc_attn_h(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, ch->sel, &at)
                    : cc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, ch->sel, &at)) &&
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
/* The PLE's key and value projections (BF16 in a GGUF) as BF16 rows, a GGUF's type 30, where the
 * backend has its kernels: half the bytes of the f32 copy the chain uploaded before. A decode step
 * reads them through the BF16 GEMV (each weight exact as a float); a prompt chunk's rows go through
 * the f16 tiles (a BF16 weight of magnitude 2^-17 or more is exact in f16, 0.02 % of Qwen3.8's are
 * smaller; the activations in f16 as the trunk's GGUF GEMMs take them), where the three-part TF32
 * GEMM over the f32 copy took 140 ms of a 470-token prompt's 1,000 on an RTX PRO 4000.
 * COLI_CUDA_CHAIN_PLE_BF16=0 keeps the f32 copy. */
static ColiCudaTensor *q38cc_tb(Q38CChain *ch, Q38Weight *w) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN_PLE_BF16"); on = !(e && *e == '0'); }
    if (!on || !w || w->gpu || w->cc || w->q8 || !w->data || w->kind != Q38_WEIGHT_BF16 || w->cols % 32 ||
        coli_cuda_block_elems(130) <= 0) return q38cc_t(ch, w);
    ColiCudaTensor *t = NULL;
    if (!coli_cuda_tensor_upload(&t, w->data, NULL, 130, w->cols, w->rows, ch->dev)) return q38cc_t(ch, w);
    w->cc = t;
    return t;
}
static int q38cc_ple(Q38CChain *ch, Model *m, int n, int c0, int ns) {
    Cfg *c = &m->c; Layer *l = &m->L[c->ple_layer];
    int W = c->hc_width, H = c->hidden, E = c->ngram_heads * c->ngram_head_dim;
    size_t pc = q38cc_ple_cells(c);
    /* the rows to the device once: the projections read their input row again for every output
     * row, which from the host's memory was over the bus every time (a decode step's key and value
     * 424 and 408 us on an RTX PRO 4000, 124 and 32 GB/s) */
    int ok = cc_write(ch->emb, 0, ch->host_emb, (size_t)n * E * 4) && cc_copy(ch->embd, 0, ch->emb, 0, (size_t)n * E) &&
             cc_matmul(q38cc_tb(ch, &l->ple_key), ch->embd, 0, ch->keys, 0, n) &&
             cc_matmul(q38cc_tb(ch, &l->ple_value), ch->embd, 0, ch->val, 0, n);
    CcPle g = {0, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, q38cc_ho(ch, W), 0, -1, 0, c->eps, (int)ch->o_ple, 0, 0};
    ok = ok && cc_ple(ch->keys, q38cc_hb(ch), ch->val, ch->prm, ch->gated, ch->normv, NULL, NULL, &g);
    if (ns <= 1) {
        int snap = ns ? ch->ple_slot[1 + c0] : ch->ple_slot[1];
        CcPle cv = {1, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, q38cc_ho(ch, W), 0, ns ? 0 : -1,
                    (int)((size_t)snap * pc), c->eps, 0, (int)ch->o_pconv, (int)((size_t)ch->ple_slot[0] * pc)};
        return ok && cc_ple(NULL, q38cc_hb(ch), NULL, NULL, ch->gated, ch->normv, ch->prm, ch->ple_ring, &cv);
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
    const int glu = cc_matmul_glu(q38cc_t(ch, &l->sh_g), q38cc_t(ch, &l->sh_u), ch->mixed, ch->hs, n);   /* the gate's and up's sums in one pass */
    return (glu > 0 || (glu < 0 && cc_matmul(q38cc_t(ch, &l->sh_g), ch->mixed, 0, ch->gs, 0, n) &&
                        cc_matmul(q38cc_t(ch, &l->sh_u), ch->mixed, 0, ch->us, 0, n) && cc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &sw))) &&
           cc_matmul(q38cc_t(ch, &l->sh_d), ch->hs, 0, ch->ds, 0, n) && cc_matmul(ch->t_sg[i], ch->mixed, 0, ch->sgd, 0, n);
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
    if (!q38cc_attn_at(c, i)) return;
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
 * expert's matrices over PCIe once; a prompt of one chunk takes the decode table too where the
 * CPU is measured faster than the copy (q38cc_pc_take: its slots by expert, q38cc_pool_groups).
 * COLI_CUDA_CHAIN_POOL_READ=1 has the device read a decode step's too, as before. Filled once after the tier's warm start, the history's
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
/* pool place k: its block and byte offset there; its bytes on the host and as the device reads them */
static int q38cc_pool_blk(const Q38CChain *ch, int k) { return ch->pool_bi ? ch->pool_bi[k] : k / ch->pool_per; }
static size_t q38cc_pool_off(const Q38CChain *ch, int k) { return ch->pool_bi ? (size_t)ch->pool_bo[k] : (size_t)(k % ch->pool_per) * ch->pool_stride; }
static const char *q38cc_pool_hptr(const Q38CChain *ch, int k) { return (const char *)cc_ptr(ch->pool_buf[q38cc_pool_blk(ch, k)]) + q38cc_pool_off(ch, k); }
/* layer i's expert's bytes in the pool and in a prompt's staging: its own, or the largest layer's */
static size_t q38cc_lstride(const Q38CChain *ch, int i) { return ch->pool_lst ? ch->pool_lst[i] : ch->pool_stride; }
/* the places a prompt's staging has for layer i's experts */
static int q38cc_stage_cap(const Q38CChain *ch, int i) {
    if (!ch->pool_lst) return ch->dm_stage_cap;
    size_t n = ch->dm_stage_size / ch->pool_lst[i];
    return n < (size_t)ch->dm_stage_cap ? (int)n : ch->dm_stage_cap;
}
/* layer i's expert in pool slot k, as a table entry */
static int q38cc_pool_entry(Q38CChain *ch, Model *m, int i, int k, CcExpert *x) {
    int64_t gb, db;
    if (k < 0 || k >= ch->pool_n || !q38cc_pool_types(m, i, &gb, &db)) return 0;
    const char *base = (const char *)cc_dev_addr(ch->pool_buf[q38cc_pool_blk(ch, k)]);
    if (!base) return 0;
    const char *g = base + q38cc_pool_off(ch, k);
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
 * COLI_CUDA_CHAIN_PROMOTE_HITS percent (97) of their routes on the card, the frame promotes
 * nothing -- a conversation the history fits pays for every swap (the newcomer's bytes on the
 * bus, a resident the history still routes to back to the pool) and gains little. 0 brakes
 * never. UD-Q2_K_XL on an RTX PRO 4000, 128 tokens, 32 a forward without the brake against 128
 * with it: code after a prose history 30.5 -> 42.2 tok/s (58% of its routes on the card),
 * with the MTP head 30.7 -> 39.9; the prose that wrote the history 52.1 -> 52.7 (98%, the
 * brake held in every frame; 128 without it: 51.9 -> 48.8). It was 90: a verify's layer whose
 * experts miss waits for the CPU's slots, and with the MTP head on the card those waits were
 * what the brake kept (Qwen3.8-Flash-Next IQ2_XS, RTX PRO 4000, served 512-token answers, six
 * runs: 92-94% of the routes on the card at 90, 96.1-96.7% at 97; the card's wait for the CPU
 * 5.1 -> 1.9 ms a verify; decode 79.3 -> 82.4 tok/s, the first answer 75.1 -> 79.3; without the
 * brake 81.5, the prompt 3% slower). */
#define Q38CC_PROMOTE 128
#define Q38CC_PROMO_HITS 97
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
/* slot r's byte offset in the arena, and layer i's up and down offsets in a slot */
static size_t q38cc_slot_off(const Q38CChain *ch, int r) { return ch->ar_soff ? ch->ar_soff[r] : (size_t)r * ch->ar_slot; }
static size_t q38cc_slot_u(const Q38CChain *ch, int i) { return ch->ar_soff ? ch->ar_lou[i] : ch->ar_off_u; }
static size_t q38cc_slot_d(const Q38CChain *ch, int i) { return ch->ar_soff ? ch->ar_lod[i] : ch->ar_off_d; }
static void q38cc_vram_entry(Q38CChain *ch, int slot, int i, CcExpert *x) {
    const char *g = ch->ar_base + q38cc_slot_off(ch, slot);
    memset(x, 0, sizeof *x);
    x->g = g; x->u = g + q38cc_slot_u(ch, i); x->d = g + q38cc_slot_d(ch, i);
}
/* the exact arena: a slot at byte offset off of size class cls, nobody's; -1 no room in the books */
static int q38cc_slot_add(Q38CChain *ch, size_t off, int cls) {
    if (ch->ar_n >= ch->ar_cap || cls < 0 || cls >= ch->ar_ncls) return -1;
    if (ch->ar_cn[cls] == ch->ar_cc[cls]) {
        int nc = ch->ar_cc[cls] ? 2 * ch->ar_cc[cls] : 64;
        int *q = (int *)realloc(ch->ar_cls[cls], (size_t)nc * sizeof(int));
        if (!q) return -1;
        ch->ar_cls[cls] = q; ch->ar_cc[cls] = nc;
    }
    int r = ch->ar_n++;
    ch->ar_soff[r] = off; ch->owner[r] = -1; ch->busy[r] = 0;
    ch->ar_cls[cls][ch->ar_cn[cls]++] = r;
    return r;
}
static int q38cc_pool_entry(Q38CChain *ch, Model *m, int i, int k, CcExpert *x);
static int q38cc_promote_init(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts;
    size_t ne = (size_t)L * E, slot = 0, ou = 0, od = 0;
    void *base = NULL; int n = 0;
    QtChainArena xa; int exact = 0;
    if (!qt_chain_arena(&base, &slot, &ou, &od, &n) || n < 1) {
        /* the exact arena: the warm start's slots (its residents' and the freed ones) and as many
         * more as the bytes it left hold of the smallest layer's */
        if (!qt_chain_arena_exact(&xa) || xa.nl != L || xa.ncls < 1 || xa.bump > xa.bytes) return 0;
        size_t mn = 0;
        for (int i = 0; i < L; i++) if (!mn || xa.lsl[i] < mn) mn = xa.lsl[i];
        if (!mn) return 0;
        exact = 1; base = xa.base; n = 1;
        for (int i = 0; i < L; i++) for (int e = 0; e < E; e++) n += qt_chain_slot(i, e) >= 0;
        for (int k = 0; k < xa.ncls; k++) n += xa.cnfree[k];
        n += (int)((xa.bytes - xa.bump) / mn);
    }
    for (int i = 0; i < L; i++) {
        int64_t gb, db;
        size_t u = exact ? xa.lou[i] : ou, d = exact ? xa.lod[i] : od, sl = exact ? xa.lsl[i] : slot;
        if (!q38cc_pool_types(m, i, &gb, &db) || (size_t)gb > u || (size_t)gb > d - u || (size_t)db > sl - d) return 0;
    }
    ch->vslot = (int32_t *)malloc(ne * sizeof(int32_t)); ch->owner = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    ch->busy = (uint8_t *)calloc((size_t)n, 1); ch->heat = (uint32_t *)calloc(ne, sizeof(uint32_t)); ch->chg = (int32_t *)malloc(ne * sizeof(int32_t));
    if (exact) {
        ch->ar_soff = (size_t *)malloc((size_t)n * sizeof(size_t));
        ch->ar_cls = (int **)calloc((size_t)xa.ncls, sizeof(int *));
        ch->ar_cn = (int *)calloc((size_t)xa.ncls, sizeof(int)); ch->ar_cc = (int *)calloc((size_t)xa.ncls, sizeof(int));
    }
    if (!ch->vslot || !ch->owner || !ch->busy || !ch->heat || !ch->chg || (exact && (!ch->ar_soff || !ch->ar_cls || !ch->ar_cn || !ch->ar_cc))) {
        free(ch->vslot); free(ch->owner); free(ch->busy); free(ch->heat); free(ch->chg);
        free(ch->ar_soff); free(ch->ar_cls); free(ch->ar_cn); free(ch->ar_cc);
        ch->vslot = ch->owner = ch->chg = NULL; ch->busy = NULL; ch->heat = NULL;
        ch->ar_soff = NULL; ch->ar_cls = NULL; ch->ar_cn = ch->ar_cc = NULL;
        return 0;
    }
    for (size_t j = 0; j < ne; j++) ch->vslot[j] = -1;
    for (int k = 0; k < n; k++) ch->owner[k] = -1;
    int held = 0;
    if (exact) {
        ch->ar_cap = n; ch->ar_n = 0; ch->ar_ncls = xa.ncls; ch->ar_bump = xa.bump; ch->ar_bytes = xa.bytes;
        ch->ar_lsl = xa.lsl; ch->ar_lou = xa.lou; ch->ar_lod = xa.lod; ch->ar_lcls = xa.lcls;
        for (int i = 0; i < L; i++) for (int e = 0; e < E; e++) {
            int k = qt_chain_slot(i, e);   /* its offset in 256-byte units */
            if (k < 0) continue;
            int r = q38cc_slot_add(ch, (size_t)k * 256u, xa.lcls[i]);
            if (r >= 0) { ch->vslot[(size_t)i * E + e] = r; ch->owner[r] = i * E + e; held++; }
        }
        for (int k = 0; k < xa.ncls; k++) for (int j = 0; j < xa.cnfree[k]; j++) q38cc_slot_add(ch, xa.cfree[k][j], k);
        n = ch->ar_n;
    } else {
        for (int i = 0; i < L; i++) for (int e = 0; e < E; e++) {
            int k = qt_chain_slot(i, e);
            if (k >= 0 && k < n && ch->owner[k] < 0) { ch->vslot[(size_t)i * E + e] = k; ch->owner[k] = i * E + e; held++; }
        }
    }
    qt_chain_own();
    ch->own = 1; ch->ar_base = (char *)base; ch->ar_slot = slot; ch->ar_off_u = ou; ch->ar_off_d = od; if (!exact) ch->ar_n = n;
    ch->promo_rate = q38cc_promote_rate(); ch->rng = 0x9e3779b9u; ch->dm_read = 0; ch->stage_dirty = 1;
    ch->promo_hits = q38cc_promote_hits(); ch->pr_routes0 = ch->routes; ch->pr_reads0 = ch->pool_reads;
    fprintf(stderr, "[chain] qwen38: the card's %d slots are the chain's to fill (%d held): up to %d experts a forward promoted from the "
                    "pool by this conversation's routes, none while the last %d frames found %d %% or more of their routes on the card "
                    "(COLI_CUDA_CHAIN_PROMOTE=0 keeps the warm start's set)\n", n, held, ch->promo_rate, Q38CC_PROMO_WIN, ch->promo_hits);
    return 1;
}
static const char *q38cc_pool_host(Q38CChain *ch, int k) {
    int ps = ch->pool_of[k];
    return ps < 0 ? NULL : q38cc_pool_hptr(ch, ps);
}
/* before a frame: the newcomers whose bytes landed take their slots; the entries rewritten
 * since the last frame to the device's table */
static int q38cc_promote_settle(Q38CChain *ch, Model *m) {
    int E = m->c.experts, w = 0;
    for (int p = 0; p < ch->npend; p++) {
        struct Q38CCPend *q = &ch->pend[p];
        if (!cc_host_up_done(q->ticket)) { ch->pend[w++] = *q; continue; }
        ch->vslot[q->k] = q->slot; ch->busy[q->slot] = 0; ch->dm_res[q->k] = 1;
        q38cc_vram_entry(ch, q->slot, q->k / E, &ch->dm_host[q->k]);
        ch->chg[ch->nchg++] = q->k;
    }
    ch->npend = w;
    if (!ch->nchg) return 1;
    int ok = cc_begin();
    static const CcExpert none;
    const size_t F = sizeof(CcExpert) / 4;   /* offsets in floats */
    const int tabd = ch->pool_cpu && ch->dm_tabd;   /* the decode table: a slot back in the pool is the CPU's (pool_cpu) */
    CcRegion *rg = ch->dm_tabu && (!tabd || ch->dm_tabdu) ? (CcRegion *)malloc((size_t)ch->nchg * sizeof *rg) : NULL;
    if (rg) {   /* the entries into the mapped copies, then a kernel copies them over: no copy engine (cc_copy_regions) */
        CcExpert *tu = (CcExpert *)cc_ptr(ch->dm_tabu), *tdu = tabd ? (CcExpert *)cc_ptr(ch->dm_tabdu) : NULL;
        for (int j = 0; j < ch->nchg; j++) {
            const int k = ch->chg[j];
            tu[k] = ch->dm_host[k];
            if (tdu) tdu[k] = ch->dm_res[k] >= 2 ? none : ch->dm_host[k];
            rg[j] = (CcRegion){(size_t)k * F, (size_t)k * F, F};
        }
        ok = ok && cc_copy_regions(ch->dm_tab, ch->dm_tabu, rg, ch->nchg) && (!tdu || cc_copy_regions(ch->dm_tabd, ch->dm_tabdu, rg, ch->nchg));
        free(rg);
    } else
        for (int j = 0; j < ch->nchg && ok; j++) {
            const int k = ch->chg[j];
            ok = cc_write(ch->dm_tab, (size_t)k * F, &ch->dm_host[k], sizeof(CcExpert));
            if (ok && tabd) ok = cc_write(ch->dm_tabd, (size_t)k * F, ch->dm_res[k] >= 2 ? &none : &ch->dm_host[k], sizeof(CcExpert));
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
/* the promotions' copies gathered so far in one call (cc_host_up_many, Q38CC_UP_BATCH experts at
 * most): the pending entries from b0 on take its ticket; refused, their slots are nobody's again
 * (their bytes undefined) and their experts the pool's. 1 sent */
#define Q38CC_UP_BATCH 64
static int q38cc_up_flush(Q38CChain *ch, void **ud, const void **us, const size_t *ub, int *nu, int b0) {
    if (!*nu) return 1;
    unsigned tk = 0;
    const int ok = cc_host_up_many(ud, us, ub, *nu, &tk);
    *nu = 0;
    if (ok) { for (int p = b0; p < ch->npend; p++) ch->pend[p].ticket = tk; return 1; }
    for (int p = b0; p < ch->npend; p++) {
        ch->owner[ch->pend[p].slot] = -1; ch->busy[ch->pend[p].slot] = 0; ch->dm_res[ch->pend[p].k] = 2; ch->promos--;
    }
    ch->npend = b0;
    return 0;
}
/* after a frame: this conversation's routes, and the promotions they earn */
/* the promotions' heat: layer i's routes of n rows (its ids in dm_idsd), then n rows' tick */
static void q38cc_heat_layer(Q38CChain *ch, Model *m, int i, int n) {
    const int E = m->c.experts, K = m->c.topk;
    const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
    for (int a = 0; a < n * K; a++) if (ids[a] >= 0 && ids[a] < E && ch->heat[(size_t)i * E + ids[a]] < 0xFFFFFFFFu) ch->heat[(size_t)i * E + ids[a]]++;
}
static void q38cc_heat_tick(Q38CChain *ch, Model *m, int n) {
    if ((ch->heat_tick += (uint32_t)n) >= Q38CC_DECAY) {
        ch->heat_tick = 0;
        for (size_t j = 0; j < (size_t)m->c.layers * m->c.experts; j++) ch->heat[j] >>= 1;
    }
}
static void q38cc_promote_pick(Q38CChain *ch, Model *m, int n);
static void q38cc_promote_books(Q38CChain *ch, Model *m, int n) {
    for (int i = 0; i < m->c.layers; i++) if (ch->dm_ran[i]) q38cc_heat_layer(ch, m, i, n);
    q38cc_heat_tick(ch, m, n);
    q38cc_promote_pick(ch, m, n);
}
/* the promotions from the routes of n rows each layer that ran has in dm_idsd: up to promo_rate
 * of the pool's experts they route to, hotter than the coldest of 64 slots drawn */
static void q38cc_promote_pick(Q38CChain *ch, Model *m, int n) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts, K = c->topk;
    uint32_t **hist = rt_counts_all();
    #define Q38CC_HIST(j) (hist && hist[(j) / E] ? hist[(j) / E][(j) % E] : 0u)
    if (q38cc_braked(ch)) return;
    int done = 0, nu = 0, b0 = ch->npend;   /* nu: the copies gathered for one call, from pending entry b0 */
    void *ud[3 * Q38CC_UP_BATCH]; const void *us[3 * Q38CC_UP_BATCH]; size_t ub[3 * Q38CC_UP_BATCH];
    for (int i = 0; i < L && done < ch->promo_rate; i++) {
        if (!ch->dm_ran[i]) continue;
        const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
        for (int a = 0; a < n * K && done < ch->promo_rate; a++) {
            int e = ids[a];
            if (e < 0 || e >= E) continue;
            int k = i * E + e;
            if (ch->dm_res[k] != 2) continue;   /* on the card, or on its way */
            /* a free slot, else the coldest of 64 drawn (none on its way); the exact arena's of the
             * layer's size class, and a slot carved from its room before any resident goes */
            int s = -1, v = -1;
            const int cls = ch->ar_soff ? ch->ar_lcls[i] : 0, nr = ch->ar_soff ? ch->ar_cn[cls] : ch->ar_n;
            for (int t = 0; t < 64 && nr > 0; t++) {
                ch->rng = ch->rng * 1664525u + 1013904223u;
                int r = (int)((ch->rng >> 8) % (unsigned)nr);
                if (ch->ar_soff) r = ch->ar_cls[cls][r];
                if (ch->busy[r]) continue;
                if (ch->owner[r] < 0) { s = r; v = -1; break; }
                int o = ch->owner[r];
                if (v < 0 || ch->heat[o] < ch->heat[v] || (ch->heat[o] == ch->heat[v] && Q38CC_HIST(o) < Q38CC_HIST(v))) { v = o; s = r; }
            }
            if ((s < 0 || v >= 0) && ch->ar_soff && ch->ar_bump + ch->ar_lsl[i] <= ch->ar_bytes) {
                int r = q38cc_slot_add(ch, ch->ar_bump, cls);
                if (r >= 0) { ch->ar_bump += ch->ar_lsl[i]; s = r; v = -1; }
            }
            if (s < 0) continue;
            if (v >= 0 && !(ch->heat[k] > ch->heat[v] || (ch->heat[k] == ch->heat[v] && Q38CC_HIST(k) > Q38CC_HIST(v)))) continue;
            int64_t gb, db;
            const char *src = q38cc_pool_host(ch, k);
            if (!src || !q38cc_pool_types(m, i, &gb, &db)) continue;
            if (ch->npend == ch->cpend) {
                int cap = ch->cpend ? 2 * ch->cpend : 64;
                struct Q38CCPend *np = (struct Q38CCPend *)realloc(ch->pend, (size_t)cap * sizeof *np);
                if (!np) { q38cc_up_flush(ch, ud, us, ub, &nu, b0); return; }
                ch->pend = np; ch->cpend = cap;
            }
            if (v >= 0) {   /* the victim to the pool, its entry rewritten before the next frame */
                ch->vslot[v] = -1; ch->dm_res[v] = 2;
                if (!q38cc_pool_entry(ch, m, v / E, ch->pool_of[v], &ch->dm_host[v])) memset(&ch->dm_host[v], 0, sizeof(CcExpert));
                ch->chg[ch->nchg++] = v;
                ch->promo_victims++;
            }
            char *dst = ch->ar_base + q38cc_slot_off(ch, s);
            ud[nu] = dst; us[nu] = src; ub[nu++] = (size_t)gb;   /* gate, up and down, sent together below */
            ud[nu] = dst + q38cc_slot_u(ch, i); us[nu] = src + Q38CC_PAD(gb); ub[nu++] = (size_t)gb;
            ud[nu] = dst + q38cc_slot_d(ch, i); us[nu] = src + 2 * Q38CC_PAD(gb); ub[nu++] = (size_t)db;
            ch->owner[s] = k; ch->busy[s] = 1; ch->dm_res[k] = 3;
            ch->pend[ch->npend++] = (struct Q38CCPend){k, s, 0};
            ch->promos++; done++;
            if (nu == 3 * Q38CC_UP_BATCH) {
                if (!q38cc_up_flush(ch, ud, us, ub, &nu, b0)) return;
                b0 = ch->npend;
            }
        }
    }
    q38cc_up_flush(ch, ud, us, ub, &nu, b0);
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
/* the grouped kernel's team: COLI_CUDA_CHAIN_POOL_THREADS, else the one the CPU probe found faster
 * (g_q38cc_pg_team, q38cc_pool_init: OpenMP's team or every logical CPU), else OpenMP's. The
 * launcher's team is the performance cores (omp_tune.h); the kernel unpacks blocks more than it
 * reads memory, and on an i5-13500 the 12 threads of its 6 P-cores and all 20 took a 452-token
 * prompt to 482-486 tok/s against 455-458 with 6 */
static int g_q38cc_pg_team;
static int q38cc_pg_threads(void) {
    static int n = -1;
    if (n < 0) { const char *e = getenv("COLI_CUDA_CHAIN_POOL_THREADS"); n = e && atoi(e) > 0 ? atoi(e) : 0; }
#ifdef _OPENMP
    return n > 0 ? n : g_q38cc_pg_team > 0 ? g_q38cc_pg_team : omp_get_max_threads();
#else
    return 1;
#endif
}
static int q38cc_pg_reserve(Q38CChain *ch, Model *m, int npc);
static int q38cc_pool_groups(const uint8_t *const *g, const int *o, int ng, const float *xg, int H, int I, int tg, int td,
                             float *gu, float *hh, float *dn, gq_act *ax, gq_act *ah);
/* the choice's measures (q38cc_pc_take), once the pool is filled. The bus: a copy of up to 256
 * MiB of the pool into the staging, timed twice, the second kept. */
static void q38cc_bus_probe(Q38CChain *ch) {
    if (!ch->dm_stage || ch->pool_nb < 1) return;
    size_t b = cc_bytes(ch->dm_stage), pb = cc_bytes(ch->pool_buf[0]);
    if (pb < b) b = pb;
    if (b > ((size_t)256 << 20)) b = (size_t)256 << 20;
    CcFetch f = {cc_ptr(ch->pool_buf[0]), 0, b};
    for (int k = 0; k < 2; k++) {
        double t0 = now_s();
        if (!cc_begin() || !cc_fetch(ch->dm_stage, &f, 1) || !cc_fetch_wait() || !cc_fetch_done() || !cc_submit(1)) { ch->pc_bus_gbs = 0; return; }
        double dt = now_s() - t0;
        ch->pc_bus_gbs = dt > 0 ? (double)b / dt / 1e9 : 0;
    }
}
/* The CPU: q38cc_pool_groups over nex of layer i's pooled experts ex (24 at most) at 7 rows each
 * (a 452-token prompt's experts averaged 5.5 to 6.9), twice; the second's time a slot in us, 0
 * when it did not run. A layer's types are its own: the caller spreads its layers. */
static double q38cc_cpu_probe(Q38CChain *ch, Model *m, int i, const int *ex, int nex) {
    Cfg *c = &m->c; int H = c->hidden, I = c->inter, E = c->experts, fg, fd;
    enum { R = 7, NX = 24 };
    if (nex > NX) nex = NX;
    if (nex < 1 || !qt_chain_layer_format(i, &fg, &fd) || fg < 100 || fd < 100 || !q38cc_pg_reserve(ch, m, nex * R)) return 0;
    const uint8_t *g[NX]; int o[NX + 1];
    for (int q = 0; q < nex; q++) { if (!(g[q] = (const uint8_t *)q38cc_pool_host(ch, i * E + ex[q]))) return 0; o[q] = q * R; }
    o[nex] = nex * R;
    for (size_t k = 0; k < (size_t)nex * R * H; k++) ch->pg_x[k] = (float)((k * 2654435761u >> 16) & 1023) / 512.f - 1.f;
    double us = 0;
    for (int k = 0; k < 2; k++) {
        double t0 = now_s();
        if (!q38cc_pool_groups(g, o, nex, ch->pg_x, H, I, fg - 100, fd - 100, ch->pg_gu, ch->pg_hh, ch->pg_dn, ch->pg_ax, ch->pg_ah)) return 0;
        us = (now_s() - t0) * 1e6 / (nex * R);
    }
    return us;
}
static void q38cc_srv_up(Q38CChain *ch, Model *m);
static void q38cc_head_pool(Q38CChain *ch, Model *m);
static void q38cc_pool_init(Model *m) {
    Q38CChain *ch = q38cc_of(m);
    const char *on = getenv("COLI_CUDA_CHAIN_POOL"), *gbs = getenv("COLI_CUDA_CHAIN_POOL_GB");
    if (!ch || !ch->ok || !ch->dm_on || ch->dm_fmt < 100 || !g_q38_gguf || (on && *on == '0') || !cc_fetch_ready()) return;
    Cfg *c = &m->c; int L = c->layers, E = c->experts;
    size_t stride = 0;
    size_t *lst = (size_t *)malloc((size_t)L * sizeof(size_t));   /* each layer's expert bytes */
    if (!lst) return;
    for (int i = 0; i < L; i++) {
        int64_t gb, db;
        if (!q38cc_pool_types(m, i, &gb, &db)) { fprintf(stderr, "[chain] qwen38: layer %d's expert rows are not whole words; no host pool\n", i); free(lst); return; }
        size_t b = 2 * Q38CC_PAD(gb) + Q38CC_PAD(db);
        lst[i] = b;
        if (b > stride) stride = b;
    }
    /* the pool and a prompt's staging at each layer's own expert size: unsloth's UD-IQ2_XS has layers
     * of 1.45 to 2.09 MB experts, and places of the largest took 49.8 GiB of RAM for its 40.3 GiB of
     * experts. COLI_CUDA_CHAIN_POOL_EXACT=0: the largest layer's for every place */
    const char *pxe = getenv("COLI_CUDA_CHAIN_POOL_EXACT");
    const int packed = !(pxe && *pxe == '0');
    if (!packed) { free(lst); lst = NULL; }
    #define Q38CC_LST(i) (lst ? lst[(i)] : stride)
    /* the experts the card leaves, the history's most used first, then the card's own (the pool
     * takes those too where RAM holds every expert: the chain then moves them, below) */
    Q38CCHot *hot = (Q38CCHot *)malloc((size_t)L * E * sizeof *hot);
    if (!hot) { free(lst); return; }
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
    if (!left) { free(hot); free(lst); return; }
    qsort(hot, (size_t)left, sizeof *hot, q38cc_hot_cmp);   /* the hottest first; those the pool takes then by layer (below) */
    const double GiB = 1073741824.0;
    double grow = 0;   /* what the RAM cache may still take: the experts the card leaves, at most */
    for (int i = 0; i < L; i++) {
        int room = m->cache[i].cap - m->cache[i].n, out = 0;
        for (int k = 0; k < left; k++) out += hot[k].i == i;
        grow += (double)(room < out ? room : out) * (double)Q38CC_LST(i);
    }
    /* and the KV and state the serve allocates once it runs, which the free RAM does not show
     * yet (#2314: a host pool took the room of a 131072-token KV cache, whose allocation failed) */
    const double later = (double)q38_state_later(m);
    double want = gbs && *gbs ? atof(gbs) * GiB : compat_mem_available_gb() * 1e9 - 4.0 * GiB - grow - later;
    size_t blk = (size_t)256 << 20;
    int per = (int)(blk / stride);
    long long fit = 0;
    if (lst) { double b = 0; while (fit < left && b + (double)lst[hot[fit].i] <= want) b += (double)lst[hot[fit++].i]; }
    else fit = want > 0 ? (long long)(want / (double)stride) : 0;
    if (fit > left) fit = left;
    double all_b = 0;   /* every expert's bytes in the pool */
    for (int i = 0; i < L; i++) all_b += (double)E * (double)Q38CC_LST(i);
    /* every expert, where RAM holds them (the CPU then computes none: no RAM cache to come) and
     * the chain may take the arena: an eviction then copies nothing back */
    int full = 0;
    {   void *ab; size_t as, au, ad; int an; QtChainArena xa;
        if (!(gbs && *gbs) && q38cc_promote_rate() > 0 && left + on_card == all &&
            compat_mem_available_gb() * 1e9 - 4.0 * GiB - later >= all_b && (qt_chain_arena(&ab, &as, &au, &ad, &an) || qt_chain_arena_exact(&xa))) { full = 1; fit = all; }
    }
    if (per < 1 || fit < 1) {
        fprintf(stderr, "[chain] qwen38: no RAM for a host pool (%.1f GiB free less 4 GiB, the RAM cache's %.1f GiB to come "
                        "and the serve's KV and state, %.1f GiB; COLI_CUDA_CHAIN_POOL_GB sets it)\n",
                compat_mem_available_gb() * 1e9 / GiB, grow / GiB, later / GiB);
        free(hot); free(lst);
        return;
    }
    /* the routes the pool takes by the history (to the experts the card leaves), and each layer's of
     * them the pool holds (the hottest first: the first fit of them) */
    int *pl = (int *)calloc((size_t)L, sizeof(int));
    {   double hl = 0, ha = 0;
        for (int k = 0; k < left + on_card; k++) { ha += hot[k].n; if (k < left) hl += hot[k].n; }
        ch->pc_frac0 = ha > 0 ? hl / ha : (double)left / all;
        for (int k = 0; pl && k < (fit < left ? fit : left); k++) pl[hot[k].i]++;
    }
    qsort(hot, (size_t)fit, sizeof *hot, q38cc_layer_cmp);   /* a layer's pooled experts side by side: a prompt chunk's copies in few runs */
    /* packed: each place at its layer's size, a block's places side by side up to blk */
    int32_t *pbi = NULL; uint32_t *pbo = NULL; size_t *bsz = NULL;
    int nb = (int)((fit + per - 1) / per);
    if (lst) {
        pbi = (int32_t *)malloc((size_t)(fit > 0 ? fit : 1) * sizeof(int32_t)); pbo = (uint32_t *)malloc((size_t)(fit > 0 ? fit : 1) * sizeof(uint32_t));
        bsz = (size_t *)calloc((size_t)(fit > 0 ? fit : 1), sizeof(size_t));
        if (!pbi || !pbo || !bsz) { free(pbi); free(pbo); free(bsz); free(hot); free(lst); return; }
        size_t o = 0; int b = 0;
        for (long long k = 0; k < fit; k++) {
            const size_t e = lst[hot[k].i];
            if (o + e > blk) { bsz[b++] = o; o = 0; }
            pbi[k] = b; pbo[k] = (uint32_t)o; o += e;
        }
        bsz[b] = o; nb = fit > 0 ? b + 1 : 0;
    }
    double t0 = now_s();
    ch->pool_buf = (CcBuf **)calloc((size_t)(nb > 0 ? nb : 1), sizeof(CcBuf *));
    ch->pool_of = (int32_t *)malloc((size_t)L * E * sizeof(int32_t));
    if (!ch->pool_buf || !ch->pool_of) { free(ch->pool_buf); free(ch->pool_of); ch->pool_buf = NULL; ch->pool_of = NULL; free(hot); free(lst); free(pbi); free(pbo); free(bsz); return; }
    for (size_t j = 0; j < (size_t)L * E; j++) ch->pool_of[j] = -1;
    int got = 0;
    for (; got < nb; got++) {
        size_t bytes = lst ? bsz[got] : (size_t)(fit - (long long)got * per < per ? fit - (long long)got * per : per) * stride;
        if (!(ch->pool_buf[got] = cc_buf(bytes, CC_DOWN)) || !cc_dev_addr(ch->pool_buf[got])) { cc_free(ch->pool_buf[got]); ch->pool_buf[got] = NULL; break; }
    }
    if (lst) { long long k = 0; while (k < fit && pbi[k] < got) k++; fit = k; }   /* the places in the blocks it got */
    else if ((long long)got * per < fit) fit = (long long)got * per;
    free(bsz);
    ch->pool_nb = got; ch->pool_per = per; ch->pool_stride = stride; ch->pool_n = (int)fit;
    ch->pool_bi = pbi; ch->pool_bo = pbo; ch->pool_lst = lst;
    double pool_b = 0;   /* the bytes its places take */
    for (long long k = 0; k < fit; k++) pool_b += (double)Q38CC_LST(hot[k].i);
    double ta = now_s();
    #pragma omp parallel for schedule(dynamic, 4)
    for (int k = 0; k < (int)fit; k++) {
        const gguf_tensor *t[3]; q38_gg_expert_tensors(m, hot[k].i, t);
        int64_t gb, db; q38cc_pool_types(m, hot[k].i, &gb, &db);
        char *dst = (char *)q38cc_pool_hptr(ch, k);
        const int64_t eb[3] = {gb, gb, db}; const size_t off[3] = {0, Q38CC_PAD(gb), 2 * Q38CC_PAD(gb)};
        for (int q = 0; q < 3; q++) q38_gg_read(t[q], dst + off[q], (int64_t)hot[k].e * eb[q], eb[q]);
    }
    for (int k = 0; k < (int)fit; k++) ch->pool_of[(size_t)hot[k].i * E + hot[k].e] = k;
    const char *pr = getenv("COLI_CUDA_CHAIN_POOL_READ");
    ch->pool_cpu = !(pr && *pr == '1');
    ch->dm_read = 0;   /* the table again, with the pool */
    /* a prompt chunk's staging: the most pooled experts any layer has, within Q38CC_PROMPT_STAGE
     * (the chain's reserve counted it) and what the expert budget gave it past that
     * (qt_stage_extra); a layer's past it are read where they are */
    int hmax = 0;
    for (int i = 0, k = 0; i < L; i++) { int h = 0; while (k < (int)fit && hot[k].i == i) { h++; k++; } if (h > hmax) hmax = h; }
    const size_t stage_max = Q38CC_PROMPT_STAGE + qt_stage_extra();
    int pcap = (int)(stage_max / stride);
    if (pcap > hmax) pcap = hmax;
    size_t sbytes = (size_t)pcap * stride;
    if (lst) {   /* places at each layer's size: as many bytes as the layer that wants most has, within the same */
        sbytes = 0;
        for (int i = 0, k = 0; i < L; i++) {
            int h = 0; while (k < (int)fit && hot[k].i == i) { h++; k++; }
            size_t b = (size_t)h * lst[i];
            if (b > sbytes) sbytes = b;
        }
        if (sbytes > stage_max) sbytes = stage_max;
        pcap = hmax;
    }
    if (pcap > 0 && sbytes > 0 && (ch->dm_stage = cc_buf(sbytes, CC_DEV))) { ch->dm_stage_cap = pcap; ch->dm_stage_size = sbytes; }
    /* the staging: the GEMV kernels' word loads over the bus came at a third of its rate, a
     * copy in 16-byte loads first at the whole of it (Q38CC_STAGE places; past them, a step's
     * pooled experts are read where they are). None where the CPU takes the pool's slots: a
     * decode step's table then holds none of them and a prompt chunk has more slots than the
     * staging maps, while its kernel scanned every step's routes for nothing (3.8 us a layer of
     * a Qwen3.8 decode step on an RTX PRO 4000) */
    const char *pst = getenv("COLI_CUDA_CHAIN_POOL_STAGE");   /* 0: the kernels read the pool where it is (A/B) */
    int nst = (pst && *pst == '0') || ch->pool_cpu ? 0 : Q38CC_STAGE;
    if (nst && cc_moe_stage_ready() && (ch->pool_stage = cc_buf(CC_MOE_STAGE_MAP + (size_t)nst * stride, CC_DEV)) &&
        !cc_moe_stage(ch->pool_stage, stride, nst)) { cc_free(ch->pool_stage); ch->pool_stage = NULL; }
    /* what a chunk stages, and the measures */
    ch->pc_stage_bytes = 0;
    for (int i = 0; pl && i < L; i++) ch->pc_stage_bytes += (double)(pl[i] < q38cc_stage_cap(ch, i) ? pl[i] : q38cc_stage_cap(ch, i)) * (double)q38cc_lstride(ch, i);
    free(pl);
    if (ch->pool_cpu && fit > 0) {   /* the CPU's time a slot over four layers spread (hot: by layer now), for each team */
        q38cc_bus_probe(ch);
        int teams[2] = {0, 0}, nt = 1;
#ifdef _OPENMP
        teams[0] = omp_get_max_threads();
        if (omp_get_num_procs() > teams[0]) teams[nt++] = omp_get_num_procs();
#endif
        for (int ti = 0; ti < nt; ti++) {
            g_q38cc_pg_team = teams[ti];
            double us = 0; int nl = 0;
            for (int t = 0, k = 0; t < 4; t++) {
                int ex[24], nex = 0, li = t * L / 4;
                while (k < (int)fit && hot[k].i < li) k++;
                for (; k < (int)fit && hot[k].i == li; k++) if (nex < 24) ex[nex++] = hot[k].e;
                double u = nex ? q38cc_cpu_probe(ch, m, li, ex, nex) : 0;
                if (u > 0) { us += u; nl++; }
            }
            us = nl ? us / nl : 0;
            if (us > 0 && (ch->pc_slot_us <= 0 || us < ch->pc_slot_us)) { ch->pc_slot_us = us; ch->pc_team = teams[ti]; }
        }
        g_q38cc_pg_team = ch->pc_team;
    }
    if (ch->pool_cpu && fit > 0 && !ch->srv) q38cc_srv_up(ch, m);
    if (ch->srv && !ch->hd_tab) q38cc_head_pool(ch, m);
    if (ch->pool_n == all && full)
        fprintf(stderr, "[chain] qwen38: host pool, every expert (the %d the card leaves and its own %d) in %.2f GiB of page-locked RAM "
                        "the device reads over the bus (%.1f s to allocate, %.1f s to read; COLI_CUDA_CHAIN_POOL=0 off)\n",
                left, on_card, pool_b / GiB, ta - t0, now_s() - ta);
    else
        fprintf(stderr, "[chain] qwen38: host pool, %d of the %d experts the card leaves in %.2f GiB of page-locked RAM the device "
                        "reads over the bus, the history's most used first (%.1f s to allocate, %.1f s to read; COLI_CUDA_CHAIN_POOL=0 off, "
                        "COLI_CUDA_CHAIN_POOL_GB sets it)\n",
                ch->pool_n, left, pool_b / GiB, ta - t0, now_s() - ta);
    if (ch->pool_stage) fprintf(stderr, "[chain] qwen38: a step's pooled experts copied to the device first (%d places, %.1f MiB)\n",
                                nst, (double)nst * stride / 1048576.0);
    if (ch->dm_stage) {
        int pmin = q38cc_stage_cap(ch, 0), pmax = pmin;
        for (int i = 1; i < L; i++) { int q = q38cc_stage_cap(ch, i); if (q < pmin) pmin = q; if (q > pmax) pmax = q; }
        fprintf(stderr, "[chain] qwen38: a prompt chunk's layer's pooled experts copied to the device while the layers "
                        "before it run (%d to %d places a layer, %.1f MiB; the most a layer has: %d%s)\n",
                pmin, pmax, (double)ch->dm_stage_size / 1048576.0, hmax, ch->pool_lst ? "; each at its layer's size" : "");
    }
    if (ch->pool_cpu && ch->dm_stage && ch->pc_bus_gbs > 0 && ch->pc_slot_us > 0 && ch->pc_frac0 > 0) {
        const double slot_ms = (double)m->c.topk * L * ch->pc_frac0 * ch->pc_slot_us / 1e3;   /* a prompt row's on the CPU */
        const double rows = Q38CC_PC_BUS * ch->pc_stage_bytes / ch->pc_bus_gbs / 1e6 / slot_ms;
        fprintf(stderr, "[chain] qwen38: a prompt's pooled experts on the CPU where that takes at most %.1f times their copy (%.2f GB "
                        "a chunk at %.1f GB/s; %.1f us a slot on the CPU's %d threads, %.1f %% of the routes to the pool by the history): "
                        "prompts of up to about %d rows (COLI_CUDA_CHAIN_POOL_CPU_ROWS=%d at most, 0 off)\n",
                Q38CC_PC_BUS, ch->pc_stage_bytes / 1e9, ch->pc_bus_gbs, ch->pc_slot_us, q38cc_pg_threads(), 100.0 * ch->pc_frac0,
                rows < q38cc_pc_rows() ? (int)rows : q38cc_pc_rows(), q38cc_pc_rows());
    }
    free(hot);
    if (ch->pool_n == all && full) q38cc_promote_init(ch, m);
    #undef Q38CC_LST
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
/* the routed experts' scratch for ch->rows rows: their routes up and down, their hidden rows and
 * sum, the experts' table */
static int q38cc_dm_bufs(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts, K = c->topk, D = c->hidden, I = c->inter;
    size_t r = (size_t)ch->rows, ne = (size_t)L * E, F = sizeof(float);
    return cc_reserve(&ch->dm_ids, (size_t)L * r * K * 4, CC_DEV) && cc_reserve(&ch->dm_w, (size_t)L * r * K * F, CC_DEV) &&
           cc_reserve(&ch->dm_idsd, (size_t)L * r * K * 4, CC_DOWN) && cc_reserve(&ch->dm_wd, (size_t)L * r * K * F, CC_DOWN) &&
           cc_reserve(&ch->dm_hid, cc_moe_hid_floats((int)r, K, D, I, E) * F, CC_DEV) && cc_reserve(&ch->dm_y, r * D * F, CC_DEV) &&
           cc_reserve(&ch->dm_hy, r * D * F, CC_UP) && cc_reserve(&ch->dm_dfl, 64, CC_DOWN) && cc_reserve(&ch->dm_hfl, 64, CC_UP) &&
           cc_reserve(&ch->dm_tab, ne * sizeof(CcExpert), CC_DEV);
}
static int q38cc_dm_begin(Q38CChain *ch, Model *m, int n) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts, K = c->topk;
    size_t ne = (size_t)L * E;
    if (!ch->dm_host) {
        ch->dm_host = malloc(ne * sizeof(CcExpert)); ch->dm_res = calloc(ne, 1); ch->dm_part = calloc(L, sizeof(int));
        ch->dm_ran = calloc(L, 1); ch->dm_dev = malloc((size_t)Q38CC_DM_ROWS * K); ch->dm_partd = calloc(L, sizeof(int));
        if (!ch->dm_host || !ch->dm_res || !ch->dm_part || !ch->dm_ran || !ch->dm_dev || !ch->dm_partd) return 0;
    }
    if (!q38cc_dm_bufs(ch, m)) return 0;
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
            if (ch->own) { if (ch->vslot[(size_t)i * E + e] >= 0) { q38cc_vram_entry(ch, ch->vslot[(size_t)i * E + e], i, x); on = 1; } }
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
    /* the mapped copies a settle's changes go through (q38cc_promote_settle); without them it writes each entry */
    if (wok && ch->own && cc_reserve(&ch->dm_tabu, ne * sizeof(CcExpert), CC_UP)) {
        memcpy(cc_ptr(ch->dm_tabu), ch->dm_host, ne * sizeof(CcExpert));
        if (td && cc_reserve(&ch->dm_tabdu, ne * sizeof(CcExpert), CC_UP)) memcpy(cc_ptr(ch->dm_tabdu), td, ne * sizeof(CcExpert));
        else if (td) { cc_free(ch->dm_tabu); ch->dm_tabu = NULL; }
    }
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
/* COLI_CUDA_CHAIN_STAGE_RING=0: every staged layer's places from the staging's start, its copy after the
 * layer before read its own */
static int q38cc_ring_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN_STAGE_RING"); on = !(e && *e == '0'); }
    return on;
}
static int q38cc_stage_table(Q38CChain *ch, Model *m) {
    Cfg *c = &m->c; int L = c->layers, E = c->experts;
    size_t ne = (size_t)L * E;
    if (!ch->dm_hostb) {
        ch->dm_hostb = malloc(ne * sizeof(CcExpert)); ch->dm_fro = calloc(L, sizeof(int)); ch->dm_frn = calloc(L, sizeof(int));
        ch->dm_fr = malloc(ne * sizeof(CcFetch)); ch->dm_iv = calloc((size_t)4 * L, sizeof(size_t));
        if (!ch->dm_hostb || !ch->dm_fro || !ch->dm_frn || !ch->dm_fr || !ch->dm_iv) return 0;
    }
    if (!cc_reserve(&ch->dm_tabb, ne * sizeof(CcExpert), CC_DEV)) return 0;
    const char *base = (const char *)cc_dev_addr(ch->dm_stage);
    if (!base) return 0;
    int k = 0;
    /* a ring: each layer's places after the last staged layer's, on from the start where the next
     * would pass the end, never over its own first place; so the next layer's copy has the part of
     * the staging this layer leaves to start in before this layer's experts are read
     * (q38cc_stage_part). COLI_CUDA_CHAIN_STAGE_RING=0: every layer from the staging's start */
    const size_t S = ch->dm_stage_size;
    size_t pos = 0;
    for (int i = 0; i < L; i++) {
        int h = 0; int64_t gb = 0, db = 0;
        ch->dm_fro[i] = k;
        int typed = q38cc_pool_types(m, i, &gb, &db);
        const size_t st = q38cc_lstride(ch, i); const int cap = q38cc_stage_cap(ch, i);   /* places at the layer's size */
        if (!q38cc_ring_on()) pos = 0;
        size_t a0 = pos, end1 = 0, lim = S; int wrapped = 0;
        for (int e = 0; e < E; e++) {
            size_t j = (size_t)i * E + e;
            CcExpert *x = &ch->dm_hostb[j];
            *x = ch->dm_host[j];
            if (ch->dm_res[j] < 2 || h >= cap || !typed) continue;
            if (pos + st > lim) {   /* past the end */
                if (!h) a0 = pos = 0;                                                 /* the layer from the staging's start */
                else if (!wrapped) { wrapped = 1; end1 = pos; pos = 0; lim = a0; }    /* on from the start, up to its first place */
                if (pos + st > lim) continue;                                         /* no room: the rest are read where they are */
            }
            int ps = ch->pool_of[j];
            const char *src = q38cc_pool_hptr(ch, ps);
            size_t dst = pos, span = 2 * Q38CC_PAD(gb) + (size_t)db;
            pos += st;
            const char *g = base + dst;
            memset(x, 0, sizeof *x);
            x->g = g; x->u = g + Q38CC_PAD(gb); x->d = g + 2 * Q38CC_PAD(gb);
            CcFetch *pr = k > ch->dm_fro[i] ? &ch->dm_fr[k - 1] : NULL;
            if (pr && src - (const char *)pr->src == (ptrdiff_t)(dst - pr->dst)) pr->bytes = dst + span - pr->dst;   /* the run goes on */
            else ch->dm_fr[k++] = (CcFetch){src, dst, span};
            h++;
        }
        ch->dm_frn[i] = k - ch->dm_fro[i];
        size_t *iv = ch->dm_iv + 4 * (size_t)i;
        if (!h) { iv[0] = iv[1] = iv[2] = iv[3] = 0; continue; }
        iv[0] = a0; iv[1] = wrapped ? end1 : pos; iv[2] = 0; iv[3] = wrapped ? pos : 0;
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
/* The first layer from i on that stages its experts, a part of its copy: part 1 the bytes clear of
 * layer prev's places, which may start while prev's experts are read (the copy waits for the reads
 * of the layer before prev, which are done), part 2 the rest, once cc_fetch_done marked prev's read;
 * part 0 all of it after that (q38cc_stage_from). Layer prev is the last staged layer before it. */
static int q38cc_stage_part(Q38CChain *ch, Model *m, int i, int prev, int part) {
    for (; i < m->c.layers && ch->dm_frn[i] <= 0; i++) {}
    if (i >= m->c.layers) return 1;
    if (!part || prev < 0 || prev >= i || ch->dm_frn[prev] <= 0) return part == 1 ? 1 : q38cc_stage_from(ch, m, i);
    const size_t *iv = ch->dm_iv + 4 * (size_t)prev;
    CcFetch *out = malloc((size_t)5 * ch->dm_frn[i] * sizeof *out);
    if (!out) return 0;
    int n = 0;
    for (int k = 0; k < ch->dm_frn[i]; k++) {
        const CcFetch f = ch->dm_fr[ch->dm_fro[i] + k];
        const size_t a = f.dst, b = f.dst + f.bytes;
        size_t cut[6]; int nc = 0;
        cut[nc++] = a;
        for (int q = 0; q < 4; q++) if (iv[q] > a && iv[q] < b) cut[nc++] = iv[q];
        cut[nc++] = b;
        for (int x = 1; x < nc; x++) for (int y = x; y > 0 && cut[y - 1] > cut[y]; y--) { size_t t = cut[y]; cut[y] = cut[y - 1]; cut[y - 1] = t; }
        for (int q = 0; q + 1 < nc; q++) {
            const size_t x = cut[q], y = cut[q + 1];
            if (y <= x) continue;
            const int in = (x >= iv[0] && x < iv[1]) || (x >= iv[2] && x < iv[3]);
            if ((part == 2) != in) continue;
            out[n++] = (CcFetch){(const char *)f.src + (x - a), x, y - x};
        }
    }
    if (part == 1) {
        ch->dm_staged++;
        for (int k = 0; k < ch->dm_frn[i]; k++) ch->dm_stage_bytes += (double)ch->dm_fr[ch->dm_fro[i] + k].bytes;
        for (int k = 0; k < n; k++) ch->dm_early_bytes += out[k].bytes;
        if (n) ch->dm_early++;
    }
    const int ok = cc_fetch(ch->dm_stage, out, n);
    free(out);
    return ok;
}
static void q38cc_dm_end(Q38CChain *ch) { if (ch->dm_held) { qt_chain_hold(0); ch->dm_held = 0; } }
/* layer i's router, routing, shared expert and the device's experts; *v: the host's word
 * a hybrid layer waits for (0: none) */
/* a step's table: a prompt chunk's stages its layers' pooled experts */
/* a prompt of S rows (one chunk, the pool on the CPU's side): its pooled slots on the CPU where the
 * CPU's estimate (the rows' slots by the pool's share of the routes, the books' once there are
 * enough, at the measured time a slot, the prompts' own once there are some) is at most
 * Q38CC_PC_BUS times the copy of every layer's pooled experts at the measured rate */
static int q38cc_lanes_on(void);
static int q38cc_lanes_min(void);
static int q38cc_lane_stage_cap(void);
static int q38cc_lane_fits(Q38CChain *ch, Model *m, int n);
static int q38cc_pc_take(Q38CChain *ch, Model *m, int S) {
    if (S <= Q38CC_DM_ROWS || S > q38cc_pc_rows() || !ch->pool_cpu || !ch->pool_of) return 0;
    if (!ch->dm_stage || ch->pc_bus_gbs <= 0 || ch->pc_slot_us <= 0) return 1;   /* nothing stages: the GEMMs would read them in place */
    /* one chunk in two lanes (q38cc_chunk_lanes): the experts a lane routes twice or more go over
     * the bus and the CPU's slots run beside the card's work, where the chunk's copy of every
     * pooled expert costs its whole time (a 452-token prompt the history's estimate staged:
     * 218 tok/s on an RTX PRO 4000) */
    if (q38cc_lanes_on() && q38cc_lane_stage_cap() > 0 && S >= q38cc_lanes_min() && S <= q38cc_rows_env() && q38cc_lane_fits(ch, m, S)) return 1;
    const Cfg *c = &m->c;
    const double frac = ch->routes >= 100000 ? (double)ch->pool_reads / (double)ch->routes : ch->pc_frac0;
    const double us = ch->pg_pairs >= 2000 ? ch->pg_ms * 1e3 / (double)ch->pg_pairs : ch->pc_slot_us;
    const double cpu_ms = (double)S * c->topk * c->layers * frac * us / 1e3, bus_ms = ch->pc_stage_bytes / ch->pc_bus_gbs / 1e6;
    return cpu_ms <= Q38CC_PC_BUS * bus_ms;
}
/* a step of n rows has the CPU compute its pooled slots: a decode step's, a short prompt's chunk's */
static int q38cc_pc_step(Q38CChain *ch, int n) { return (n <= Q38CC_DM_ROWS || ch->pc_prompt) && ch->pool_cpu && ch->dm_tabd; }
static CcBuf *q38cc_dm_tab(Q38CChain *ch, int n) {
    if (q38cc_pc_step(ch, n)) return ch->dm_tabd;   /* the pool's experts on the CPU */
    return n > Q38CC_DM_ROWS && ch->dm_stage && ch->dm_tabb ? ch->dm_tabb : ch->dm_tab;
}
/* layer i has slots the host computes in a step of n rows */
static int q38cc_dm_hosted(Q38CChain *ch, int i, int n) { return q38cc_pc_step(ch, n) ? ch->dm_partd[i] : ch->dm_part[i]; }
/* a step's layer i up to its routes: the router, the routes (and their copy down), the host's
 * word where the host has slots (*v), the shared expert */
static int q38cc_dm_route(Q38CChain *ch, Model *m, Layer *l, int i, int n, unsigned *v) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, K = c->topk, io = i * ch->rows * K;
    CcRoute rp = {n, E, K, 0, 0, E, 0, 0, io, K, io, K, 1, 1, c->norm_topk ? 1 : 0, 1.f, 1.f, 0.f};
    int ok = cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) && cc_route(ch->lg, NULL, ch->dm_ids, ch->dm_w, &rp) &&
             cc_copy(ch->dm_idsd, io, ch->dm_ids, io, (size_t)n * K) && cc_copy(ch->dm_wd, io, ch->dm_w, io, (size_t)n * K);
    *v = 0;
    if (ok && q38cc_dm_hosted(ch, i, n)) { *v = ++ch->dm_seq; ok = cc_copy(ch->mixd, 0, ch->mixed, 0, (size_t)n * H) && cc_signal(ch->dm_dfl, 0, *v); }
    return ok && q38cc_shared(ch, m, l, i, n);
}
/* the device's slots of layer i's n rows (their routes in) into dm_hid */
static int q38cc_dm_card(Q38CChain *ch, Model *m, int i, int n) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, K = c->topk, io = i * ch->rows * K;
    CcBuf *tab = q38cc_dm_tab(ch, n);
    int stg = tab == ch->dm_tabb && ch->dm_frn[i] > 0;   /* this layer's copy to wait for */
    CcMoe mp = {n, K, H, c->inter, ch->dm_fmt, ch->dm_gs, 0, 1, i * E, 0, H, io, K, io, K, 0, H, -1, E, 0, 0, 0, 0};
    if (ch->dm_fmt >= 100) qt_chain_layer_format(i, &mp.fmt, &mp.fmt_d);   /* a GGUF's types change from layer to layer */
    int ok = (!stg || cc_fetch_wait()) && cc_moe(tab, ch->mixed, ch->dm_ids, ch->dm_w, ch->dm_hid, ch->dm_y, NULL, &mp);
    if (ok && stg) ok = cc_fetch_done() && q38cc_stage_from(ch, m, i + 1);   /* the sum below reads no weights */
    return ok;
}
/* layer i up to the device's own experts: q38cc_dm_route, then q38cc_dm_card */
static int q38cc_dm_layer_a(Q38CChain *ch, Model *m, Layer *l, int i, int n, unsigned *v) {
    return q38cc_dm_route(ch, m, l, i, n, v) && q38cc_dm_card(ch, m, i, n);
}
/* the rest: the host's word where it has slots, the sum of the slots into dm_y; hid as a
 * q38cc_dm_card of lay_S rows left it, these n rows its rows lay_s0.. (0, 0: this step's) */
static int q38cc_dm_sum(Q38CChain *ch, Model *m, int i, int n, unsigned v, CcBuf *hid, int lay_S, int lay_s0) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, K = c->topk, io = i * ch->rows * K;
    CcBuf *tab = q38cc_dm_tab(ch, n);
    CcMoe ms = {n, K, H, c->inter, ch->dm_fmt, ch->dm_gs, 0, 2, i * E, 0, H, io, K, io, K, 0, H, v ? 0 : -1, E, 0, 0, lay_S, lay_s0};
    if (ch->dm_fmt >= 100) qt_chain_layer_format(i, &ms.fmt, &ms.fmt_d);
    int ok = !v || cc_wait_host(ch->dm_hfl, 0, v);
    ok = ok && cc_moe(tab, NULL, ch->dm_ids, ch->dm_w, hid, ch->dm_y, ch->dm_hy, &ms);
    ch->dm_steps++;
    return ok;
}
static int q38cc_dm_layer_b(Q38CChain *ch, Model *m, int i, int n, unsigned v) { return q38cc_dm_sum(ch, m, i, n, v, ch->dm_hid, 0, 0); }
static int q38cc_dm_layer(Q38CChain *ch, Model *m, Layer *l, int i, int n, unsigned *v) {
    return q38cc_dm_layer_a(ch, m, l, i, n, v) && q38cc_dm_layer_b(ch, m, i, n, *v);
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
/* q38cc_pool_pairs for a prompt chunk's pooled slots, by expert: ng experts, expert q's
 * matrices at g[q] and its slots' input rows at xg rows o[q] to o[q + 1], each slot's down
 * row into dn at the same row. An expert's rows go through its matrices together (gq_rows_f8
 * unpacks a block once for four rows), ax[q] and ah[q] its activations (kept from call to call). */
static int q38cc_pool_groups(const uint8_t *const *g, const int *o, int ng, const float *xg, int H, int I, int tg, int td,
                             float *gu, float *hh, float *dn, gq_act *ax, gq_act *ah) {
    const int64_t rg = gq_row_bytes(tg, H), rd = gq_row_bytes(td, I);
    const size_t pg = Q38CC_PAD(rg * I);
    const int cgu = (2 * I + 7) / 8, cdn = (H + 7) / 8, f8g = gq_f8_ok(tg, H), f8d = gq_f8_ok(td, I);
    int err = 0;
    #pragma omp parallel num_threads(q38cc_pg_threads())
    {
        #pragma omp for schedule(dynamic, 4)
        for (int q = 0; q < ng; q++)
            if (gq_act_set(&ax[q], xg + (size_t)o[q] * H, o[q + 1] - o[q], H)) {
                #pragma omp atomic write
                err = 1;
            }
        #pragma omp for schedule(dynamic, 4)
        for (int c = 0; c < ng * cgu; c++) {
            const int q = c / cgu, o0 = (c % cgu) * 8, o1 = o0 + 8 < 2 * I ? o0 + 8 : 2 * I;
            if (err) continue;
            float *y = gu + (size_t)o[q] * 2 * I;
            if (o1 - o0 == 8 && (o1 <= I || o0 >= I) && f8g) {
                const uint8_t *m = o0 < I ? g[q] : g[q] + pg; const int oo = o0 < I ? o0 : o0 - I;
                gq_rows_f8(y + o0, 2 * I, &ax[q], m + (int64_t)oo * rg, rg, tg, H, 8);
                continue;
            }
            for (int r = o0; r < o1; r++) {
                const uint8_t *m = r < I ? g[q] : g[q] + pg; const int oo = r < I ? r : r - I;
                gq_row_all(y + r, 2 * I, &ax[q], m + (int64_t)oo * rg, tg, H);
            }
        }
        #pragma omp for schedule(dynamic, 4)
        for (int q = 0; q < ng; q++) {
            if (err) continue;
            for (int s = o[q]; s < o[q + 1]; s++) {
                const float *gg = gu + (size_t)s * 2 * I, *uu = gg + I; float *h = hh + (size_t)s * I;
                for (int r = 0; r < I; r++) h[r] = gg[r] / (1.f + expf(-gg[r])) * uu[r];
            }
            if (gq_act_set(&ah[q], hh + (size_t)o[q] * I, o[q + 1] - o[q], I)) {
                #pragma omp atomic write
                err = 1;
            }
        }
        #pragma omp for schedule(dynamic, 8)
        for (int c = 0; c < ng * cdn; c++) {
            const int q = c / cdn, o0 = (c % cdn) * 8, o1 = o0 + 8 < H ? o0 + 8 : H;
            if (err) continue;
            float *y = dn + (size_t)o[q] * H; const uint8_t *m = g[q] + 2 * pg;
            if (o1 - o0 == 8 && f8d) { gq_rows_f8(y + o0, H, &ah[q], m + (int64_t)o0 * rd, rd, td, I, 8); continue; }
            for (int r = o0; r < o1; r++) gq_row_all(y + r, H, &ah[q], m + (int64_t)r * rd, td, I);
        }
    }
    return !err;
}
/* a prompt chunk's npc pooled slots of layer i (pa: the slots, in order) on the CPU, added to
 * host_routed: the slots in expert order (a counting sort, a slot's place in pg_pos), their
 * rows gathered, q38cc_pool_groups, then each slot's row in rank order within its row, as
 * q38cc_dm_serve's decode step adds them */
/* the grouped path's scratch for npc slots of a layer's E experts */
static int q38cc_pg_reserve(Q38CChain *ch, Model *m, int npc) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, I = c->inter;
    if (ch->pg_ecap < E + 1) {
        int *st = realloc(ch->pg_start, (size_t)(E + 1) * sizeof(int)), *go = st ? realloc(ch->pg_o, (size_t)(E + 1) * sizeof(int)) : NULL;
        const uint8_t **gg = go ? realloc(ch->pg_g, (size_t)E * sizeof *gg) : NULL;
        if (st) ch->pg_start = st;
        if (go) ch->pg_o = go;
        if (gg) ch->pg_g = gg;
        if (!st || !go || !gg) { fprintf(stderr, "[chain] qwen38: no memory for the CPU's pool slots\n"); return 0; }
        ch->pg_ecap = E + 1;
    }
    if (ch->pg_qcap < E) {   /* an activation per expert, kept */
        gq_act *a2 = realloc(ch->pg_ax, (size_t)E * sizeof *a2), *h2 = a2 ? realloc(ch->pg_ah, (size_t)E * sizeof *h2) : NULL;
        if (a2) { memset(a2 + ch->pg_qcap, 0, (size_t)(E - ch->pg_qcap) * sizeof *a2); ch->pg_ax = a2; }
        if (h2) { memset(h2 + ch->pg_qcap, 0, (size_t)(E - ch->pg_qcap) * sizeof *h2); ch->pg_ah = h2; }
        if (!a2 || !h2) { fprintf(stderr, "[chain] qwen38: no memory for the CPU's pool slots\n"); return 0; }
        ch->pg_qcap = E;
    }
    if (ch->pg_scap < npc) {
        free(ch->pg_x); free(ch->pg_gu); free(ch->pg_hh); free(ch->pg_dn);
        ch->pg_x = malloc((size_t)npc * H * sizeof(float)); ch->pg_gu = malloc((size_t)npc * 2 * I * sizeof(float));
        ch->pg_hh = malloc((size_t)npc * I * sizeof(float)); ch->pg_dn = malloc((size_t)npc * H * sizeof(float));
        ch->pg_scap = ch->pg_x && ch->pg_gu && ch->pg_hh && ch->pg_dn ? npc : 0;
        if (!ch->pg_scap) { fprintf(stderr, "[chain] qwen38: no memory for the CPU's pool slots\n"); return 0; }
    }
    return 1;
}
static int q38cc_serve_groups(Q38CChain *ch, Model *m, int i, int n, const int *ids, const float *w, const float *x,
                              const int *pa, int npc) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, K = c->topk, I = c->inter, fg, fd;
    double tp = now_s();
    (void)n;
    if (!qt_chain_layer_format(i, &fg, &fd) || fg < 100 || fd < 100) {
        fprintf(stderr, "[chain] qwen38: layer %d's pooled experts have no block types for the CPU\n", i);
        return 0;
    }
    if (!q38cc_pg_reserve(ch, m, npc)) return 0;
    int *st = ch->pg_start, *o = ch->pg_o, ng = 0;
    memset(st, 0, (size_t)(E + 1) * sizeof(int));
    for (int p = 0; p < npc; p++) st[ids[pa[p]] + 1]++;
    for (int e = 0; e < E; e++) {
        if (st[e + 1]) {
            const char *g = q38cc_pool_host(ch, i * E + e);
            if (!g) { fprintf(stderr, "[chain] qwen38: layer %d's expert %d has no place in the host pool\n", i, e); return 0; }
            o[ng] = st[e]; ch->pg_g[ng++] = (const uint8_t *)g;
        }
        st[e + 1] += st[e];
    }
    o[ng] = npc;
    for (int p = 0; p < npc; p++) {   /* each slot's place, and its row there */
        const int a = pa[p], at = st[ids[a]]++;
        ch->pg_pos[p] = at;
        memcpy(ch->pg_x + (size_t)at * H, x + (size_t)(a / K) * H, (size_t)H * sizeof(float));
    }
    if (!q38cc_pool_groups(ch->pg_g, o, ng, ch->pg_x, H, I, fg - 100, fd - 100, ch->pg_gu, ch->pg_hh, ch->pg_dn, ch->pg_ax, ch->pg_ah)) {
        fprintf(stderr, "[chain] qwen38: the CPU's pool slots of layer %d failed\n", i);
        return 0;
    }
    for (int p = 0; p < npc; p++) {   /* after the misses, in rank order within a row */
        const int a = pa[p]; float *ys = ch->host_routed + (size_t)(a / K) * H;
        const float *dv = ch->pg_dn + (size_t)ch->pg_pos[p] * H, wv = w[a];
        for (int d = 0; d < H; d++) ys[d] += wv * dv[d];
    }
    ch->pg_pairs += (unsigned long long)npc; ch->pg_groups += (unsigned long long)ng; ch->pg_ms += (now_s() - tp) * 1e3;
    return 1;
}
/* the host's share of hybrid layer i: once the device's word says its routes and rows are
 * down, the slots the table lacks, its rows up, its word */
/* the host waits for the device's word v: a hybrid layer's routes and rows down */
static int q38cc_dm_wait(Q38CChain *ch, int i, unsigned v) {
    volatile unsigned *dw = (volatile unsigned *)cc_ptr(ch->dm_dfl);
    double t0 = now_s();
    for (unsigned long spin = 0; *dw < v; spin++)
        if ((spin & 0xFFFFu) == 0 && (cc_lost() || now_s() - t0 > 60.0)) {
            fprintf(stderr, "[chain] qwen38: the device did not reach layer %d's routes in 60 s\n", i);
            return 0;
        }
    ch->dm_spin_ms += (now_s() - t0) * 1e3;
    return 1;
}
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
    const int pc = q38cc_pc_step(ch, n), big = n > Q38CC_DM_ROWS;   /* big: a prompt chunk's, by expert */
    if (n * K > ch->dm_dev_cap) {
        uint8_t *dd = realloc(ch->dm_dev, (size_t)n * K);
        if (!dd) { fprintf(stderr, "[chain] qwen38: no memory for layer %d's slots\n", i); return 0; }
        ch->dm_dev = dd; ch->dm_dev_cap = n * K;
    }
    if (big && ch->pg_cap < n * K) {
        int *pa2 = realloc(ch->pg_a, (size_t)n * K * sizeof(int)), *ps2 = pa2 ? realloc(ch->pg_pos, (size_t)n * K * sizeof(int)) : NULL;
        if (pa2) ch->pg_a = pa2;
        if (ps2) ch->pg_pos = ps2;
        if (!pa2 || !ps2) { fprintf(stderr, "[chain] qwen38: no memory for layer %d's slots\n", i); return 0; }
        ch->pg_cap = n * K;
    }
    int npc = 0, nmiss = 0, pa_d[Q38CC_DM_ROWS * 32], *pa = big ? ch->pg_a : pa_d;
    const int pmax = big ? n * K : Q38CC_DM_ROWS * 32;
    const uint8_t *lsm = ch->ls_i == i ? ch->ls_mark : NULL;   /* a lane chunk's experts staged for the card (q38cc_lane_stage) */
    for (int a = 0; a < n * K; a++) {
        int e = ids[a]; const uint8_t r = e < 0 || e >= E ? 1 : lsm && lsm[e] ? 1 : ch->dm_res[(size_t)i * E + e];
        const int pool = pc && r >= 2 && npc < pmax;
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
    if (npc && big) {   /* a prompt chunk's, by expert */
        if (!q38cc_serve_groups(ch, m, i, n, ids, w, x, pa, npc)) return 0;
    } else if (npc) {
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
    if (!cc_host_word_skip(v, "qwen38")) ((volatile unsigned *)cc_ptr(ch->dm_hfl))[0] = v;
    __sync_synchronize();
    double t2 = now_s();
    ch->dm_hyb++; ch->dm_spin_ms += (t1 - t0) * 1e3; ch->dm_host_ms += (t2 - t1) * 1e3; ch->host_ms += (t2 - t1) * 1e3;
    return 1;
}
/* ---- the pool's server: a decode step's pooled slots on threads of their own ----------
 * A decode step's hybrid layer had the recording thread, once it had recorded the layer
 * after, spin for the layer's routes and then wake OpenMP's team for its few pooled slots (a
 * sleeping team's wake, three barriers): 83-89 us a slot of 1.44 MiB on an i5-13500's six
 * performance cores, the device waiting on it. The server's threads take a step's hybrid
 * layers as jobs the recording thread posts (q38cc_srv_post): each waits for the device's
 * word itself, takes the gate and up rows eight at a time, then (once every gate and up row
 * is in) the down rows eight at a time for every slot, and writes those rows' sums where the
 * device reads them; the thread that ends the last writes the host's word. They spin only
 * while a forward is in flight (q38cc_srv_open to q38cc_srv_close) and sleep between
 * forwards. Each slot's bits are q38cc_pool_pairs's: the same activations, the same eight-row
 * kernels, a row's slots added in rank order from zero (q38cc_dm_serve's sum).
 * A forward takes it with at most Q38CC_DM_ROWS rows when no layer can miss (every expert on
 * the card or in the pool: the RAM cache's path stays the recording thread's).
 * COLI_CUDA_CHAIN_SERVE_THREADS: its threads (q38cc_srv_threads; 0: the recording thread
 * serves as before). With 16 an i5-13500 took 64-68 us a slot, with 20 105. */
#include <pthread.h>
#define Q38CC_SRV_MAX 64                     /* threads at most */
#define Q38CC_SRV_RING 64                    /* a forward's jobs at most (its hybrid layers) */
#define Q38CC_SRV_SLOTS (Q38CC_DM_ROWS * 32)
#define Q38CC_SRV_GU 2                       /* gate/up chunks of eight rows a claim */
#define Q38CC_SRV_DN 4                       /* down chunks a claim */
typedef struct {
    int layer, n, tg, td; unsigned v;
    const int *ids; const float *w;          /* the layer's routes */
    const uint8_t *res;                      /* the layer's experts: 2 or more in the pool */
    int cgu, gud, cdn, dnd;                  /* claims and the chunks done */
    int xs[Q38CC_DM_ROWS];                   /* a row's activation: 0 not made, 1 being made, 2 made (3 failed) */
    int gs[Q38CC_SRV_SLOTS], hs[Q38CC_SRV_SLOTS];   /* a group's input rows and its SwiGLU rows, the same states */
    long long t0;                            /* ns: the device's word first seen */
} Q38CCJob;
typedef struct Q38CCSrv Q38CCSrv;
typedef const uint8_t *(*Q38CCWhere)(void *ctx, int k);
struct Q38CCSrv {
    int nt, H, I, K, E;
    pthread_t th[Q38CC_SRV_MAX]; int started;
    struct Q38CCSrvArg { Q38CCSrv *sv; int me; } arg[Q38CC_SRV_MAX];
    pthread_mutex_t mx; pthread_cond_t cv;
    int open, quit, abort_, err;
    unsigned posted, seen[Q38CC_SRV_MAX];    /* jobs posted; each thread's next job */
    Q38CCJob job[Q38CC_SRV_RING];
    const volatile unsigned *dfl; volatile unsigned *hfl;   /* the device's word, the host's */
    const float *x; float *out;              /* the input rows, the rows the device reads */
    Q38CCWhere where; void *ctx;             /* an expert's place in the pool */
    float *gu, *hh;                          /* [SLOTS][2I], [SLOTS][I]: a slot's at its place in group order */
    gq_act ax[Q38CC_DM_ROWS];                /* a row's input */
    gq_act gx[Q38CC_SRV_SLOTS], ah[Q38CC_SRV_SLOTS];   /* a group's input rows, its SwiGLU rows */
    unsigned long long jobs, slots, groups, ns;   /* books (the last thread of a job) */
};
static inline void q38cc_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}
static long long q38cc_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}
#define Q38CC_LD(p) __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define Q38CC_ST(p, v) __atomic_store_n(p, v, __ATOMIC_RELEASE)
#define Q38CC_ADD(p, v) __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL)
/* an activation made once a job, by the first thread to need it */
static int q38cc_srv_act(int *st, gq_act *a, const float *x, int I) {
    int z = 0;
    if (__atomic_compare_exchange_n(st, &z, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        const int bad = gq_act_set(a, x, 1, I) != 0;
        Q38CC_ST(st, bad ? 3 : 2);
        return !bad;
    }
    int s;
    while ((s = Q38CC_LD(st)) == 1) q38cc_relax();
    return s == 2;
}
static void q38cc_srv_fail(Q38CCSrv *sv, Q38CCJob *J, const char *why) {
    if (!__atomic_exchange_n(&sv->err, 1, __ATOMIC_ACQ_REL))
        fprintf(stderr, "[chain] qwen38: the pool's server: %s (layer %d)\n", why, J->layer);
    Q38CC_ST(&sv->abort_, 1);
    __sync_synchronize();
    *sv->hfl = J->v;   /* the device goes on; the forward is the CPU's again (q38cc_srv_close) */
    __sync_synchronize();
}
/* a group's input rows: its slots' rows' activations copied (each made once a job, q38cc_srv_act) */
static int q38cc_srv_gact(Q38CCSrv *sv, Q38CCJob *J, int q, const int *mem, int g, int K) {
    int z = 0;
    if (__atomic_compare_exchange_n(&J->gs[q], &z, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        gq_act *d = &sv->gx[q];
        int bad = 0;
        const int H = sv->H;
        d->S = g; d->I = H;
        for (int r = 0; r < g && !bad; r++) {
            const int s = mem[r] / K;
            if (!q38cc_srv_act(&J->xs[s], &sv->ax[s], sv->x + (size_t)s * H, H)) { bad = 1; break; }
            const gq_act *a = &sv->ax[s];
            memcpy(d->q + (int64_t)r * H, a->q, (size_t)H);
            memcpy(d->d + (int64_t)r * (H / 32), a->d, (size_t)(H / 32) * sizeof(float));
            memcpy(d->s + (int64_t)r * (H / 16), a->s, (size_t)(H / 16) * sizeof(int32_t));
            memcpy(d->l4 + (int64_t)r * (H / 4), a->l4, (size_t)(H / 4) * sizeof(int32_t));
        }
        Q38CC_ST(&J->gs[q], bad ? 3 : 2);
        return !bad;
    }
    int st;
    while ((st = Q38CC_LD(&J->gs[q])) == 1) q38cc_relax();
    return st == 2;
}
static void q38cc_srv_job(Q38CCSrv *sv, Q38CCJob *J) {
    const int H = sv->H, I = sv->I, K = sv->K, E = sv->E, n = J->n, tg = J->tg, td = J->td;
    if (Q38CC_LD(&sv->abort_)) return;
    long long tw = 0;
    for (unsigned long spin = 0; *sv->dfl < J->v; spin++) {
        q38cc_relax();
        if ((spin & 0xFFFu) == 0) {
            if (Q38CC_LD(&sv->abort_)) return;
            long long t = q38cc_ns();
            if (!tw) tw = t;
            else if (t - tw > 60000000000LL) { q38cc_srv_fail(sv, J, "the device did not reach the routes in 60 s"); return; }
        }
    }
    long long z = 0, t0 = q38cc_ns();
    __atomic_compare_exchange_n(&J->t0, &z, t0, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    /* the pooled slots in rank order, and their groups (every thread the same): a verify's slots
     * of one expert, up to GQ_FUSED_ROWS of them in rank order, go through its matrices together
     * (gq_rows_f8 on up to that many rows gives each the bits of its row alone); a decode step's
     * slot is a group of its own */
    int pa[Q38CC_SRV_SLOTS], np = 0, ge[Q38CC_SRV_SLOTS], gn[Q38CC_SRV_SLOTS], g0[Q38CC_SRV_SLOTS + 1], at[Q38CC_SRV_SLOTS], ng = 0;
    const uint8_t *gp[Q38CC_SRV_SLOTS];
    int mem[Q38CC_SRV_SLOTS];   /* the groups' slots, group after group */
    for (int a = 0; a < n * K; a++) {
        const int e = J->ids[a];
        if (e < 0 || e >= E || J->res[e] < 2) continue;
        int q = -1;
        for (int t = 0; t < ng && n > 1; t++) if (ge[t] == e && gn[t] < GQ_FUSED_ROWS) { q = t; break; }
        if (q < 0) {
            if (!(gp[ng] = sv->where(sv->ctx, J->layer * E + e))) { q38cc_srv_fail(sv, J, "an expert without a place in the pool"); return; }
            q = ng++; ge[q] = e; gn[q] = 0;
        }
        at[np] = q * 8 + gn[q]++;   /* group and place in it, resolved to a slot's row below */
        pa[np++] = a;
    }
    g0[0] = 0;
    for (int q = 0; q < ng; q++) g0[q + 1] = g0[q] + gn[q];
    for (int p = 0; p < np; p++) { const int q = at[p] / 8, r = at[p] % 8; at[p] = g0[q] + r; mem[at[p]] = pa[p]; }
    const int64_t rg = gq_row_bytes(tg, H), rd = gq_row_bytes(td, I);
    const size_t pg = Q38CC_PAD(rg * I);
    const int cgu = (2 * I + 7) / 8, ngu = ng * cgu, cdn = (H + 7) / 8, f8g = gq_f8_ok(tg, H), f8d = gq_f8_ok(td, I);
    /* gate and up, eight rows a chunk of a group */
    for (;;) {
        int c = Q38CC_ADD(&J->cgu, Q38CC_SRV_GU), c1 = c + Q38CC_SRV_GU < ngu ? c + Q38CC_SRV_GU : ngu, c0 = c;
        if (c >= ngu) break;
        for (; c < c1; c++) {
            const int q = c / cgu, o0 = (c % cgu) * 8, o1 = o0 + 8 < 2 * I ? o0 + 8 : 2 * I;
            if (!q38cc_srv_gact(sv, J, q, mem + g0[q], gn[q], K)) { q38cc_srv_fail(sv, J, "no memory for an activation"); return; }
            float *y = sv->gu + (size_t)g0[q] * 2 * I;
            if (o1 - o0 == 8 && (o1 <= I || o0 >= I) && f8g) {
                const uint8_t *m = o0 < I ? gp[q] : gp[q] + pg; const int oo = o0 < I ? o0 : o0 - I;
                gq_rows_f8(y + o0, 2 * I, &sv->gx[q], m + (int64_t)oo * rg, rg, tg, H, 8);
                continue;
            }
            for (int o = o0; o < o1; o++) {
                const uint8_t *m = o < I ? gp[q] : gp[q] + pg; const int oo = o < I ? o : o - I;
                gq_row_all(y + o, 2 * I, &sv->gx[q], m + (int64_t)oo * rg, tg, H);
            }
        }
        Q38CC_ADD(&J->gud, c1 - c0);
    }
    while (Q38CC_LD(&J->gud) < ngu) { q38cc_relax(); if (Q38CC_LD(&sv->abort_)) return; }
    /* each group's SwiGLU rows and their activation, by whoever comes first */
    for (int q = 0; q < ng; q++) {
        int st = 0;
        if (!__atomic_compare_exchange_n(&J->hs[q], &st, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
        for (int r = g0[q]; r < g0[q + 1]; r++) {
            const float *gg = sv->gu + (size_t)r * 2 * I, *uu = gg + I; float *h = sv->hh + (size_t)r * I;
            for (int o = 0; o < I; o++) h[o] = gg[o] / (1.f + expf(-gg[o])) * uu[o];
        }
        Q38CC_ST(&J->hs[q], gq_act_set(&sv->ah[q], sv->hh + (size_t)g0[q] * I, gn[q], I) ? 3 : 2);
    }
    for (int q = 0; q < ng; q++) {
        int st;
        while ((st = Q38CC_LD(&J->hs[q])) == 1) q38cc_relax();
        if (st != 2) { q38cc_srv_fail(sv, J, "no memory for an activation"); return; }
    }
    /* down, eight rows a chunk for every group, and those rows' sums: a row's slots in rank order from zero */
    float dl[Q38CC_SRV_SLOTS * 8];
    for (;;) {
        int c = Q38CC_ADD(&J->cdn, Q38CC_SRV_DN), c1 = c + Q38CC_SRV_DN < cdn ? c + Q38CC_SRV_DN : cdn, c0 = c;
        if (c >= cdn) break;
        for (; c < c1; c++) {
            const int o0 = c * 8, o1 = o0 + 8 < H ? o0 + 8 : H, w8 = o1 - o0;
            for (int q = 0; q < ng; q++) {
                const uint8_t *m = gp[q] + 2 * pg;
                if (w8 == 8 && f8d) { gq_rows_f8(dl + g0[q] * 8, 8, &sv->ah[q], m + (int64_t)o0 * rd, rd, td, I, 8); continue; }
                for (int o = o0; o < o1; o++) gq_row_all(dl + g0[q] * 8 + (o - o0), 8, &sv->ah[q], m + (int64_t)o * rd, td, I);
            }
            float ys[Q38CC_DM_ROWS][8];
            memset(ys, 0, sizeof ys);
            for (int p = 0; p < np; p++) {
                const float wv = J->w[pa[p]]; float *y = ys[pa[p] / K]; const float *o = dl + at[p] * 8;
                for (int d = 0; d < w8; d++) y[d] += wv * o[d];
            }
            for (int s = 0; s < n; s++) memcpy(sv->out + (size_t)s * H + o0, ys[s], (size_t)w8 * sizeof(float));
        }
        __sync_synchronize();   /* the rows (write-combined memory) out before they count */
        if (Q38CC_ADD(&J->dnd, c1 - c0) + (c1 - c0) == cdn) {   /* the last: the host's word */
            if (!cc_host_word_skip(J->v, "qwen38")) *sv->hfl = J->v;
            __sync_synchronize();
            Q38CC_ADD(&sv->jobs, 1ULL);
            if (np) {
                Q38CC_ADD(&sv->slots, (unsigned long long)np);
                Q38CC_ADD(&sv->groups, (unsigned long long)ng);
                Q38CC_ADD(&sv->ns, (unsigned long long)(q38cc_ns() - Q38CC_LD(&J->t0)));
            }
        }
    }
}
static void *q38cc_srv_main(void *p) {
    struct Q38CCSrvArg *a = (struct Q38CCSrvArg *)p; Q38CCSrv *sv = a->sv;
    unsigned next = 0;   /* from the first job: a thread may start after it was posted */
    for (;;) {
        if (next < Q38CC_LD(&sv->posted)) {
            q38cc_srv_job(sv, &sv->job[next % Q38CC_SRV_RING]);
            Q38CC_ST(&sv->seen[a->me], ++next);
            continue;
        }
        if (Q38CC_LD(&sv->quit)) break;
        if (Q38CC_LD(&sv->open)) { q38cc_relax(); continue; }
        pthread_mutex_lock(&sv->mx);
        while (!sv->open && !sv->quit && next >= Q38CC_LD(&sv->posted)) pthread_cond_wait(&sv->cv, &sv->mx);
        pthread_mutex_unlock(&sv->mx);
    }
    return NULL;
}
/* nt threads for experts of H by I, K slots a row, E a layer; NULL where it cannot */
static void q38cc_srv_stop(Q38CCSrv *sv);
static void q38cc_srv_free(Q38CCSrv *sv) {
    for (int s = 0; s < Q38CC_DM_ROWS; s++) gq_act_free(&sv->ax[s]);
    for (int p = 0; p < Q38CC_SRV_SLOTS; p++) { gq_act_free(&sv->gx[p]); gq_act_free(&sv->ah[p]); }
    free(sv->gu); free(sv->hh); free(sv);
}
static Q38CCSrv *q38cc_srv_start(int nt, int H, int I, int K, int E, Q38CCWhere where, void *ctx) {
    if (nt <= 0 || K > 32) return NULL;
    if (nt > Q38CC_SRV_MAX) nt = Q38CC_SRV_MAX;
    Q38CCSrv *sv = (Q38CCSrv *)calloc(1, sizeof *sv);
    if (!sv) return NULL;
    sv->nt = nt; sv->H = H; sv->I = I; sv->K = K; sv->E = E; sv->where = where; sv->ctx = ctx;
    sv->gu = (float *)malloc((size_t)Q38CC_SRV_SLOTS * 2 * I * sizeof(float));
    sv->hh = (float *)malloc((size_t)Q38CC_SRV_SLOTS * I * sizeof(float));
    int ok = sv->gu && sv->hh;
    /* the activations' room before the forwards (no allocation while the device waits) */
    float *zx = ok ? (float *)calloc((size_t)GQ_FUSED_ROWS * (H > I ? H : I), sizeof(float)) : NULL;
    ok = ok && zx;
    for (int s = 0; s < Q38CC_DM_ROWS && ok; s++) ok = gq_act_set(&sv->ax[s], zx, 1, H) == 0;
    for (int p = 0; p < Q38CC_SRV_SLOTS && ok; p++)
        ok = gq_act_set(&sv->gx[p], zx, GQ_FUSED_ROWS, H) == 0 && gq_act_set(&sv->ah[p], zx, GQ_FUSED_ROWS, I) == 0;
    free(zx);
    if (ok) { pthread_mutex_init(&sv->mx, NULL); pthread_cond_init(&sv->cv, NULL); }
    for (int t = 0; t < nt && ok; t++) {
        sv->arg[t] = (struct Q38CCSrvArg){sv, t};
        if (pthread_create(&sv->th[t], NULL, q38cc_srv_main, &sv->arg[t]) != 0) break;
        sv->started++;
    }
    if (!ok || sv->started < nt) {
        if (ok) q38cc_srv_stop(sv);
        else q38cc_srv_free(sv);
        return NULL;
    }
    return sv;
}
static void q38cc_srv_stop(Q38CCSrv *sv) {
    if (!sv) return;
    pthread_mutex_lock(&sv->mx); Q38CC_ST(&sv->quit, 1); pthread_cond_broadcast(&sv->cv); pthread_mutex_unlock(&sv->mx);
    for (int t = 0; t < sv->started; t++) pthread_join(sv->th[t], NULL);
    pthread_mutex_destroy(&sv->mx); pthread_cond_destroy(&sv->cv);
    q38cc_srv_free(sv);
}
/* a forward's jobs read these (the same buffers every layer) */
static void q38cc_srv_open(Q38CCSrv *sv, const volatile unsigned *dfl, volatile unsigned *hfl, const float *x, float *out) {
    sv->dfl = dfl; sv->hfl = hfl; sv->x = x; sv->out = out;
    Q38CC_ST(&sv->abort_, 0); Q38CC_ST(&sv->err, 0);
    pthread_mutex_lock(&sv->mx); Q38CC_ST(&sv->open, 1); pthread_cond_broadcast(&sv->cv); pthread_mutex_unlock(&sv->mx);
}
/* layer's pooled slots of a step of n rows, once the device's word reaches v (res: the layer's
 * experts, 2 or more where the pool holds them) */
static void q38cc_srv_post(Q38CCSrv *sv, int layer, int n, unsigned v, int tg, int td, const int *ids, const float *w,
                           const uint8_t *res) {
    unsigned k = sv->posted;
    Q38CCJob *J = &sv->job[k % Q38CC_SRV_RING];
    memset(J, 0, sizeof *J);
    J->layer = layer; J->n = n; J->v = v; J->tg = tg; J->td = td; J->ids = ids; J->w = w; J->res = res;
    Q38CC_ST(&sv->posted, k + 1);
}
/* the forward's end: every thread past every job (ok 0: they skip what is left and the host's
 * word goes to vmax, the highest the frame waits for, so a device still running waits for
 * nothing); 0 when a job failed */
static int q38cc_srv_close(Q38CCSrv *sv, int ok, unsigned vmax) {
    const unsigned k = Q38CC_LD(&sv->posted);
    if (!ok) { Q38CC_ST(&sv->abort_, 1); __sync_synchronize(); *sv->hfl = vmax; __sync_synchronize(); }
    long long t0 = q38cc_ns();
    for (int t = 0; t < sv->started; t++)
        for (unsigned long spin = 0; Q38CC_LD(&sv->seen[t]) < k; spin++) {
            q38cc_relax();
            if ((spin & 0xFFFu) == 0 && q38cc_ns() - t0 > 60000000000LL) {
                fprintf(stderr, "[chain] qwen38: the pool's server: a thread did not end its jobs in 60 s\n");
                Q38CC_ST(&sv->err, 1);
                break;
            }
        }
    Q38CC_ST(&sv->open, 0);
    return !Q38CC_LD(&sv->err);
}
/* the chain's side of the server: an expert's place in the pool, the forwards that take it, the books */
static const uint8_t *q38cc_srv_where(void *ctx, int k) {
    Q38CChain *ch = (Q38CChain *)ctx;
    if (ch->hd_base && k >= ch->hd_k0) return ch->hd_base + (size_t)(k - ch->hd_k0) * ch->hd_stride;   /* the MTP head's */
    return (const uint8_t *)q38cc_pool_host(ch, k);
}
/* COLI_CUDA_CHAIN_SERVE_THREADS, else the logical CPUs less four (the recording thread, the
 * driver's, the system's), at most 16, none below two: on an i5-13500 (20 logical CPUs) a decode
 * after a 452-token prompt took 49.7-50.2 tok/s on the recording thread, 52.4-52.7 with 12
 * threads, 52.3-52.9 with 16 and 46.1-46.9 with 20 */
static int q38cc_srv_threads(void) {
    const char *e = getenv("COLI_CUDA_CHAIN_SERVE_THREADS");
    if (e && *e) return atoi(e) > 0 ? atoi(e) : 0;
#ifdef _OPENMP
    const int n = omp_get_num_procs() - 4;
    return n < 2 ? 0 : n > 16 ? 16 : n;
#else
    return 0;
#endif
}
static void q38cc_srv_up(Q38CChain *ch, Model *m) {
    const int nt = q38cc_srv_threads();
    if (nt <= 0) return;
    if (!(ch->srv = q38cc_srv_start(nt, m->c.hidden, m->c.inter, m->c.topk, m->c.experts, q38cc_srv_where, ch))) {
        fprintf(stderr, "[chain] qwen38: the pool's server did not start; the recording thread serves a decode step's pooled experts\n");
        return;
    }
    fprintf(stderr, "[chain] qwen38: the pool's server: %d threads for a decode step's pooled experts (COLI_CUDA_CHAIN_SERVE_THREADS=0: "
                    "the recording thread's)\n", ch->srv->started);
}
/* One pass of a block's Q2_0 search: v = s x, its code q = round(v id) within -1..2, the sums of v q and
 * q q (eight lanes, integer rounding and clamps: the loop vectorizes) */
static void q38cc_q2_0_pass(const float *x, float s, float id, float *sxq, float *sqq) {
    float a[8] = {0}, c8[8] = {0};
    for (int j = 0; j < 64; j += 8)
        for (int l = 0; l < 8; l++) {
            const float v = s * x[j + l];
            int q = (int)(v * id + 8.5f) - 8; q = q < -1 ? -1 : q; q = q > 2 ? 2 : q;
            a[l] += v * (float)q; c8[l] += (float)(q * q);
        }
    *sxq = *sqq = 0.f;
    for (int l = 0; l < 8; l++) { *sxq += a[l]; *sqq += c8[l]; }
}
/* 64 weights as a Q2_0 block (ggml_quants.h's gq_blk_q2_0: an f16 scale d, then a 2-bit code c a
 * weight, weight 4k + r in bits 2r of byte k, its value (c - 1) d). The grid {-1, 0, 1, 2} is not
 * symmetric, so both signs of d are tried, each from two starts (the wider side's end on its code,
 * 0.6 of it), every start refitted by least squares twice; the least squared error wins, and the
 * codes are taken again for the f16 scale. */
static void q38cc_q2_0_block(const float *x, uint8_t *b) {
    float mx = x[0], mn = x[0];
    for (int j = 1; j < 64; j++) { mx = x[j] > mx ? x[j] : mx; mn = x[j] < mn ? x[j] : mn; }
    float best = INFINITY, bd = 0.f;
    for (int s = -1; s <= 1; s += 2) {
        const float hi = s > 0 ? mx : -mn, lo = s > 0 ? -mn : mx, big = hi / 2 > lo ? hi / 2 : lo, st[2] = {big, 0.6f * big};
        for (int k = 0; k < 2; k++) {
            float d = st[k];
            for (int it = 0; it < 3 && d > 0.f; it++) {
                float sxq, sqq;
                q38cc_q2_0_pass(x, (float)s, 1.f / d, &sxq, &sqq);
                const float e = d * d * sqq - 2.f * d * sxq;   /* the squared error less the block's sum of squares */
                if (e < best) { best = e; bd = s * d; }
                if (sqq <= 0.f || sxq <= 0.f) break;
                d = sxq / sqq;
            }
        }
    }
    const uint16_t h = kv16_from(bd);
    const float d = kv16_to(h), id = d != 0.f ? 1.f / d : 0.f;
    uint8_t c[64];
    for (int j = 0; j < 64; j++) { int q = (int)(x[j] * id + 8.5f) - 8; q = q < -1 ? -1 : q; q = q > 2 ? 2 : q; c[j] = (uint8_t)(q + 1); }
    b[0] = (uint8_t)(h & 0xff); b[1] = (uint8_t)(h >> 8);
    for (int k = 0; k < 16; k++) b[2 + k] = (uint8_t)(c[4 * k] | c[4 * k + 1] << 2 | c[4 * k + 2] << 4 | c[4 * k + 3] << 6);
}
/* The MTP head's routed experts on the card in Q2_0 (q38cc_head_card_on): read from the head's
 * file a few at a time, each row's blocks quantized, copied up; the table points at them. A draft
 * routes, computes them and sums them on the card (q38cc_mtp_rows). 0: they stay where they were
 * (the pool's server, q38cc_head_pool), the card had no room or no copy. */
static int q38cc_head_card(Q38CChain *ch, Model *m) {
    if (!q38cc_head_card_on(m) || !ch->mtp_on || !ch->dm_on || !cc_fetch_ready() || ch->hd_tab) return 0;
    Cfg *c = &m->c; const int L = c->layers, E = c->experts, K = c->topk, H = c->hidden, I = c->inter;
    const gguf_tensor *t[3]; q38_gg_expert_tensors(m, L, t);
    const int64_t eb[3] = {gq_row_bytes(t[0]->type, H) * I, gq_row_bytes(t[1]->type, H) * I, gq_row_bytes(t[2]->type, I) * H};
    const int64_t qg = gq_row_bytes(GQ_Q2_0, H), qd = gq_row_bytes(GQ_Q2_0, I);
    const size_t pg = Q38CC_PAD(qg * I), stride = q38cc_head_card_stride(c), r = (size_t)Q38CC_DM_ROWS;
    const double t0 = now_s();
    if (!cc_reserve(&ch->hd_dev, (size_t)E * stride, CC_DEV) || !cc_reserve(&ch->hd_tab, (size_t)E * sizeof(CcExpert), CC_DEV) ||
        !cc_reserve(&ch->hd_ids, r * K * 4, CC_DEV) || !cc_reserve(&ch->hd_w, r * K * 4, CC_DEV)) {
        cc_free(ch->hd_dev); cc_free(ch->hd_tab); ch->hd_dev = ch->hd_tab = NULL;
        fprintf(stderr, "[chain] qwen38: no room on the card for the MTP head's experts in Q2_0 (%.2f GiB); they stay in RAM\n",
                (double)E * stride / 1073741824.0);
        return 0;
    }
    const char *base = (const char *)cc_dev_addr(ch->hd_dev);
    enum { B = 32 };   /* experts a copy */
    uint8_t *hb = base ? (uint8_t *)malloc((size_t)B * stride) : NULL;
    CcExpert *tab = hb ? (CcExpert *)calloc((size_t)E, sizeof *tab) : NULL;
    int ok = tab != NULL;
    for (int e0 = 0; e0 < E && ok; e0 += B) {
        const int ne = E - e0 < B ? E - e0 : B;
        int bad = 0;
        #pragma omp parallel for schedule(dynamic, 1) reduction(| : bad) num_threads(omp_get_num_procs())
        for (int k = 0; k < ne * 3; k++) {   /* an expert's matrix */
            const int x = k / 3, q = k % 3, rows = q < 2 ? I : H, cols = q < 2 ? H : I;
            const int64_t rb = gq_row_bytes(t[q]->type, cols), qb = q < 2 ? qg : qd;
            uint8_t *raw = (uint8_t *)malloc((size_t)eb[q]), *dst = hb + (size_t)x * stride + (size_t)q * pg;
            float *f = (float *)malloc((size_t)cols * sizeof(float));
            if (!raw || !f) bad = 1;
            else {
                q38_gg_read(t[q], raw, (int64_t)(e0 + x) * eb[q], eb[q]);
                for (int o = 0; o < rows && !bad; o++) {
                    if (gq_dequant_row(t[q]->type, raw + (int64_t)o * rb, f, cols)) { bad = 1; break; }
                    for (int j = 0; j < cols; j += 64) q38cc_q2_0_block(f + j, dst + (int64_t)o * qb + (j / 64) * 18);
                }
            }
            free(raw); free(f);
        }
        ok = !bad && cc_begin() && cc_write(ch->hd_dev, (size_t)e0 * stride / 4, hb, (size_t)ne * stride) && cc_submit(1);
    }
    for (int e = 0; e < E && ok; e++) {
        const char *g = base + (size_t)e * stride;
        tab[e].g = g; tab[e].u = g + pg; tab[e].d = g + 2 * pg;
    }
    ok = ok && cc_begin() && cc_write(ch->hd_tab, 0, tab, (size_t)E * sizeof *tab) && cc_submit(1);
    free(hb); free(tab);
    if (!ok) {
        cc_free(ch->hd_dev); cc_free(ch->hd_tab); ch->hd_dev = ch->hd_tab = NULL;
        if (cc_lost()) return 0;
        fprintf(stderr, "[chain] qwen38: the MTP head's experts did not reach the card in Q2_0; they stay in RAM\n");
        return 0;
    }
    fprintf(stderr, "[chain] qwen38: the MTP head's %d routed experts on the card in Q2_0 (%.2f GiB, from %s in %.1f s; "
                    "COLI_CUDA_CHAIN_MTP_CARD=0: in RAM)\n", E, (double)E * stride / 1073741824.0, gq_type_name(t[0]->type), now_s() - t0);
    return 1;
}
/* The MTP head's routed experts for the pool's server: every one in RAM as the pool lays them out
 * (gate, up and down at 0, Q38CC_PAD(gate), twice that), read once from the head's GGUF. A draft
 * then routes on the device and the server computes its experts while the card runs the shared
 * expert, in the frame (q38cc_mtp_rows), where the host's step computed them between two frames
 * (3.8 ms a draft row on an i5-13500). COLI_CUDA_CHAIN_MTP_POOL=0: the host's step. */
static void q38cc_head_pool(Q38CChain *ch, Model *m) {
    const char *e = getenv("COLI_CUDA_CHAIN_MTP_POOL");
    if ((e && *e == '0') || !ch->srv || !ch->mtp_on || !g_q38_gguf || ch->hd_base) return;
    Cfg *c = &m->c; const int L = c->layers, E = c->experts, K = c->topk, H = c->hidden, I = c->inter;
    const gguf_tensor *t[3]; q38_gg_expert_tensors(m, L, t);
    const int tg = t[0]->type, td = t[2]->type;
    if (t[1]->type != tg || !gq_f8_ok(tg, H) || !gq_f8_ok(td, I)) return;
    const int64_t gb = gq_row_bytes(tg, H) * I, db = gq_row_bytes(td, I) * H;
    const size_t stride = 2 * Q38CC_PAD(gb) + Q38CC_PAD(db);
    double t0 = now_s();
    uint8_t *base = (uint8_t *)malloc((size_t)E * stride);
    if (!base) {
        fprintf(stderr, "[chain] qwen38: no RAM for the MTP head's experts (%.2f GiB); its host step computes them\n", (double)E * stride / 1073741824.0);
        return;
    }
    #pragma omp parallel for schedule(dynamic, 4)
    for (int x = 0; x < E; x++) {
        uint8_t *dst = base + (size_t)x * stride;
        const int64_t eb[3] = {gb, gb, db}; const size_t off[3] = {0, Q38CC_PAD(gb), 2 * Q38CC_PAD(gb)};
        for (int q = 0; q < 3; q++) q38_gg_read(t[q], dst + off[q], (int64_t)x * eb[q], eb[q]);
    }
    size_t r = (size_t)Q38CC_DM_ROWS;
    if (!(ch->hd_res = malloc((size_t)E)) || !cc_reserve(&ch->hd_ids, r * K * 4, CC_DEV) || !cc_reserve(&ch->hd_w, r * K * 4, CC_DEV) ||
        !cc_reserve(&ch->hd_idsd, r * K * 4, CC_DOWN) || !cc_reserve(&ch->hd_wd, r * K * 4, CC_DOWN)) {
        free(base); free(ch->hd_res); ch->hd_res = NULL;
        fprintf(stderr, "[chain] qwen38: the MTP head's routes found no room; its host step computes its experts\n");
        return;
    }
    memset(ch->hd_res, 2, (size_t)E);
    ch->hd_stride = stride; ch->hd_tg = tg; ch->hd_td = td; ch->hd_k0 = L * E;
    __sync_synchronize();
    ch->hd_base = base;
    fprintf(stderr, "[chain] qwen38: the MTP head's %d routed experts (%s) in %.2f GiB of RAM for the pool's server, read in %.1f s "
                    "(COLI_CUDA_CHAIN_MTP_POOL=0: the host's step)\n", E, gq_type_name(tg), (double)E * stride / 1073741824.0, now_s() - t0);
}
static Q38CCSrv *q38cc_srv_of(Q38CChain *ch, Model *m, int n) {
    if (!ch->srv || n > Q38CC_DM_ROWS || !q38cc_pc_step(ch, n) || m->c.layers > Q38CC_SRV_RING) return NULL;
    for (int i = 0; i < m->c.layers; i++) if (ch->dm_part[i]) return NULL;   /* a layer that can miss */
    return ch->srv;
}
static void q38cc_srv_books(Q38CChain *ch) {
    Q38CCSrv *sv = ch->srv;
    const unsigned long long j = Q38CC_LD(&sv->jobs), sl = Q38CC_LD(&sv->slots), ns = Q38CC_LD(&sv->ns);
    const double ms = (double)(ns - ch->sv_ns) / 1e6;
    ch->dm_hyb += j - ch->sv_jobs; ch->dm_pairs += sl - ch->sv_slots; ch->pc_pairs += sl - ch->sv_slots;
    ch->pc_ms += ms; ch->dm_host_ms += ms; ch->host_ms += ms;
    ch->sv_jobs = j; ch->sv_slots = sl; ch->sv_ns = ns;
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
/* n rows of streams into hyper: a decode step's, a verify's or a draft's through mapped memory and a
 * copy kernel (cc_copy), not the copy engine, where they waited behind the megabytes of the experts
 * the chain promotes (cc_host_up: a draft's frame up to 12.5 ms late on an RTX PRO 4000); a
 * prompt chunk's through the engine. The frames before have finished: the mapped rows are free. */
static int q38cc_hyper_in(Q38CChain *ch, const float *h, int n, int W) {
    const size_t b = (size_t)n * W * 4;
    if (n <= Q38CC_DM_ROWS && cc_reserve(&ch->hyper_up, (size_t)Q38CC_DM_ROWS * W * 4, CC_UP)) {
        memcpy(cc_ptr(ch->hyper_up), h, b);
        return cc_copy(ch->hyper, 0, ch->hyper_up, 0, (size_t)n * W);
    }
    return cc_write(ch->hyper, 0, h, b);
}
static int q38cc_frame_begin(Q38CChain *ch) {
    ch->frames++;
    if (ch->fault_at && ch->frames == ch->fault_at) { fprintf(stderr, "[chain] qwen38: fault injected at frame %llu\n", ch->frames); return 0; }
    return cc_begin();
}

/* ---- a long prompt layer by layer (qwen38_chain.h's q38c_forward_lm, on CUDA) ----------
 * A prompt of more chunks than one staged every layer's host pool experts once a chunk.
 * Where every layer runs its experts in the frame (no hybrid layer) and a chunk stages them,
 * the prompt runs layer by layer: each layer's chunks in order, twice (the attention or
 * DeltaNet of every chunk, then the experts of every chunk), its experts' copy waited for at
 * the second pass's first chunk and the next staged layer's copy started after its last, so
 * that copy runs beside the next layer's whole first pass; every row's
 * streams on the device between layers (hyper_all), each layer's chunk a frame waited for
 * (the host's tables it writes and its K/V rows down are a chunk's, each chunk the bits it
 * gets chunk by chunk). The promotions' heat counts each layer's chunks as they run, and their
 * picks run at the end, one a chunk from each layer's last chunk's routes, as chunk by chunk
 * they ran after each chunk: without them the decode after a 4220-token prompt (Qwen3.8-Flash-
 * Next IQ2_XS, an RTX PRO 4000) ran at 63.3 tok/s against 69.2 chunk by chunk, with them at
 * 67.6 and 68.2 against 66.2 and 67.5. A caller that wants the rows' final streams (an MTP
 * head's prefill: want_streams) gets them as the chunk path gives them, each chunk's after the
 * last layer; such a prompt ran chunk by chunk, every chunk staging every layer again: 56 GB
 * over the bus for that prompt with its MTP head, 19 GB layer by layer.
 * COLI_CUDA_CHAIN_LAYERWISE: 0 chunk by chunk. */
/* layer i's routed experts for one chunk of a prompt run layer by layer: wait for the layer's
 * copy at its first chunk (first), start the next staged layer's after its last (last) */
static int q38cc_dm_layer_lm(Q38CChain *ch, Model *m, Layer *l, int i, int n, int first, int last) {
    Cfg *c = &m->c; int H = c->hidden, E = c->experts, K = c->topk, io = i * ch->rows * K;
    CcBuf *tab = ch->dm_tabb;
    int stg = ch->dm_frn[i] > 0;
    CcRoute rp = {n, E, K, 0, 0, E, 0, 0, io, K, io, K, 1, 1, c->norm_topk ? 1 : 0, 1.f, 1.f, 0.f};
    CcMoe mp = {n, K, H, c->inter, ch->dm_fmt, ch->dm_gs, 0, 1, i * E, 0, H, io, K, io, K, 0, H, -1, E, 0, 0, 0, 0};
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
/* COLI_CUDA_CHAIN_MOE_GROUP=0: a long prompt's routed experts a chunk at a time */
static int q38cc_moe_group_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN_MOE_GROUP"); on = !(e && *e == '0'); }
    return on;
}
/* the outputs of down a slice G rows' routed experts take in a chunk's scratch (dm_hid): the gate's
 * and up's outputs of every slot, then down's for the slice (cc_moe_cols), a multiple of 128; 0 none */
static int q38cc_group_cols(const Q38CChain *ch, const Model *m, int G) {
    const Cfg *c = &m->c;
    const size_t A = (size_t)G * c->topk, cap = cc_bytes(ch->dm_hid) / 4, ints = cc_moe_hid_floats(G, c->topk, 0, 0, c->experts);
    if (!A || ints >= cap || (cap - ints) / A <= (size_t)c->inter) return 0;
    size_t cols = ((cap - ints) / A - (size_t)c->inter) / 128 * 128;
    const size_t all = ((size_t)c->hidden + 127) / 128 * 128;
    return (int)(cols < all ? cols : all);
}
/* The rows from c0 (a chunk's start) to end whose routed experts run in one call: whole chunks, as many
 * as leave a slice of 512 of down's outputs (all of them where hidden is smaller) in dm_hid, their
 * buffers reserved; 0: this chunk alone. A chunk's tiles decode each expert's weights for its rows: a
 * 4180-token prompt's chunks of 2048, 2048 and 84 rows decoded a layer's experts three times
 * (Qwen3.8's 512 experts of 2560 x 640 in Q2_0, test_cuda_bd's BENCH_MOE on an RTX PRO 4000: 9.4,
 * 9.4 and 4.4 ms, against 11.7 ms for the 4180 rows in one call) */
static int q38cc_moe_group_rows(Q38CChain *ch, Model *m, int c0, int end, int rows) {
    Cfg *c = &m->c; int H = c->hidden, C = c->hc_count, K = c->topk;
    if (!q38cc_moe_group_on() || !cc_moe_cols_ready() || end - c0 <= rows || ch->dm_fmt < 100) return 0;
    const int want = H < 512 ? (H + 127) / 128 * 128 : 512;
    for (int k = (end - c0 + rows - 1) / rows; k >= 2; k--) {
        const int G = end - c0 < k * rows ? end - c0 : k * rows;
        if (q38cc_group_cols(ch, m, G) < want) continue;
        if (!cc_reserve(&ch->mg_x, (size_t)G * H * 4, CC_DEV) || !cc_reserve(&ch->mg_y, (size_t)G * H * 4, CC_DEV) ||
            !cc_reserve(&ch->mg_inj, (size_t)G * C * 4, CC_DEV) || !cc_reserve(&ch->mg_ids, (size_t)G * K * 4, CC_DEV) ||
            !cc_reserve(&ch->mg_w, (size_t)G * K * 4, CC_DEV)) return 0;
        return G;
    }
    return 0;
}
/* Layer i's routed experts for a group's G rows from segment row c0 - g0 in one call: each chunk's
 * mixer, router and routes first (the rows into mg_x, mg_inj, mg_ids and mg_w), then the experts of
 * all of them into mg_y (cc_moe_cols), the layer's staged copy waited for before (first) and the
 * next staged layer's started after (last), as q38cc_dm_layer_lm does around its chunk's. Where
 * moe_cols does not take the layer each chunk's call on the same rows, as before. 1 done, 0 failed */
static int q38cc_moe_group(Q38CChain *ch, Model *m, Layer *l, int i, int c0, int g0, int G, int rows, int first, int last) {
    Cfg *c = &m->c; int H = c->hidden, C = c->hc_count, E = c->experts, K = c->topk;
    for (int r = 0; r < G; r += rows) {
        const int n = G - r < rows ? G - r : rows;
        CcRoute rp = {n, E, K, 0, 0, E, 0, 0, r * K, K, r * K, K, 1, 1, c->norm_topk ? 1 : 0, 1.f, 1.f, 0.f};
        if (!q38cc_frame_begin(ch)) return 0;
        ch->hyp = ch->hyper_all; ch->hoff = c0 + r - g0;
        int ok = q38cc_gr_read(ch, m, &l->mlp_gr, ch->o_mn[i], n, ch->inj_m) &&
                 cc_copy(ch->mg_x, (size_t)r * H, ch->mixed, 0, (size_t)n * H) && cc_copy(ch->mg_inj, (size_t)r * C, ch->inj_m, 0, (size_t)n * C) &&
                 cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) && cc_route(ch->lg, NULL, ch->mg_ids, ch->mg_w, &rp);
        if (!ok || !cc_submit(0)) return 0;
    }
    CcMoe mp = {G, K, H, c->inter, ch->dm_fmt, ch->dm_gs, 0, 3, i * E, 0, H, 0, K, 0, K, 0, H, -1, E, 0, 0, 0, 0};
    if (ch->dm_fmt >= 100) qt_chain_layer_format(i, &mp.fmt, &mp.fmt_d);
    const int stg = ch->dm_frn[i] > 0;
    if (!q38cc_frame_begin(ch) || (stg && first && !cc_fetch_wait())) return 0;
    /* the next staged layer's copy where it does not cover this layer's places, beside these experts */
    const int early = stg && last && q38cc_ring_on();
    if (early && !q38cc_stage_part(ch, m, i + 1, i, 1)) return 0;
    int r = cc_moe_cols(ch->dm_tabb, ch->mg_x, ch->mg_ids, ch->mg_w, ch->dm_hid, ch->mg_y, &mp, q38cc_group_cols(ch, m, G));
    if (r < 0) {   /* not taken: each chunk's rows as their own call, the same ops as q38cc_dm_layer_lm's */
        ch->mg_declined++;
        r = 1;
        for (int o = 0; o < G && r; o += rows) {
            CcMoe q = mp; q.S = G - o < rows ? G - o : rows; q.x_off = q.y_off = o * H; q.i_off = q.w_off = o * K;
            r = cc_moe(ch->dm_tabb, ch->mg_x, ch->mg_ids, ch->mg_w, ch->dm_hid, ch->mg_y, NULL, &q);
        }
    }
    if (!r || (stg && last && !(cc_fetch_done() && q38cc_stage_part(ch, m, i + 1, i, early ? 2 : 0))) || !cc_submit(0)) return 0;
    ch->mg_groups++;
    return 1;
}
/* a group's chunk at row o of the group, n rows: its mixed rows and injection back, the shared expert,
 * its routes where the books read them (dm_idsd, dm_wd at layer i) and its experts' sums into dm_y */
static int q38cc_moe_group_take(Q38CChain *ch, Model *m, Layer *l, int i, int o, int n) {
    Cfg *c = &m->c; int H = c->hidden, C = c->hc_count, K = c->topk;
    const size_t io = (size_t)i * ch->rows * K;
    ch->dm_steps++; ch->mg_chunks++;
    return cc_copy(ch->mixed, 0, ch->mg_x, (size_t)o * H, (size_t)n * H) && cc_copy(ch->inj_m, 0, ch->mg_inj, (size_t)o * C, (size_t)n * C) &&
           q38cc_shared(ch, m, l, i, n) && cc_copy(ch->dm_y, 0, ch->mg_y, (size_t)o * H, (size_t)n * H) &&
           cc_copy(ch->dm_idsd, io, ch->mg_ids, (size_t)o * K, (size_t)n * K) && cc_copy(ch->dm_wd, io, ch->mg_w, (size_t)o * K, (size_t)n * K);
}
/* 1 done, 0 declined before any layer ran, -1 lost */
static int q38cc_forward_lm(Model *m, Q38CChain *ch, const int *ids, int S, int pos_base, int nlogits,
                            float *hyper_h, int want_streams, float *logit, int rows) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, L = c->layers, E = c->experts, V = c->vocab, K = c->topk;
    int half = c->rotary_dim / 2;
    ch->hyp = NULL; ch->hoff = 0;
    if (!q38cc_dm_begin(ch, m, rows)) return cc_lost() ? -1 : 0;
    if (q38cc_dm_tab(ch, rows) != ch->dm_tabb) return 0;   /* nothing stages: the chunks read it either way */
    for (int i = 0; i < L; i++) if (ch->dm_part[i]) return 0;   /* a hybrid layer: the chunks' host steps */
    const int SEG = (q38cc_lm_rows() / rows > 0 ? q38cc_lm_rows() / rows : 1) * rows, seg = S < SEG ? S : SEG;
    int nlast = 0;   /* the rows of the last chunk: each layer's routes in dm_idsd are its */
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
            /* a served prompt's STOP or CANCEL between two layers (prompt_break.h): 2, the caller
             * declines the step and q38_serve_prefill puts back the state it copied */
            if (coli_prompt_break()) { cc_finish(); free(cs_all); return 2; }
            Layer *l = &m->L[i];
            /* the layer's attention (or DeltaNet) for every chunk first, then its experts for every
             * chunk: a chunk's experts read only its own rows, and the layer's copy over the bus runs
             * beside all of the first pass instead of the first chunk's attention alone (the
             * staging is free once the layer before read its weights). Each chunk takes the ops it
             * took, on the same values: the streams after the attention wait in hyper_all */
            for (int c0 = g0; c0 < g0 + gn; c0 += rows) {
                int n = g0 + gn - c0 < rows ? g0 + gn - c0 : rows, pb = pos_base + c0;
                if (!q38cc_frame_begin(ch)) { free(cs_all); return -1; }
                if (half) memcpy(cc_ptr(ch->cs), cs_all + (size_t)c0 * half * 2, (size_t)n * half * 2 * sizeof(float));
                ch->hyp = ch->hyper_all; ch->hoff = c0 - g0;   /* the chunk's streams where they wait, in place */
                int ok = 1;
                if (ok && i == c->ple_layer) { q38cc_ple_rows(m, ids, c0, n, ch->host_emb); ok = q38cc_ple(ch, m, n, c0, 0); }
                ok = ok && q38cc_gr_read(ch, m, &l->attn_gr, ch->o_an[i], n, ch->inj_a);
                ok = ok && (c->is_attn[i] ? q38cc_attention(ch, m, l, i, n, pb) : q38cc_deltanet(ch, m, l, i, n, c0, 0));
                ok = ok && q38cc_gr_apply(ch, m, ch->inj_a, n);
                double t0 = now_s();
                ok = ok && cc_submit(1);
                m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0; ch->wait_ms += (now_s() - t0) * 1e3;
                if (!ok) { free(cs_all); return -1; }
                q38cc_kv_down(ch, m, i, n, pb);
                if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
            }
            for (int c0 = g0, gs = g0, ge = g0; c0 < g0 + gn; c0 += rows) {
                int n = g0 + gn - c0 < rows ? g0 + gn - c0 : rows;
                if (c0 >= ge) {   /* the next chunks' routed experts in one call (q38cc_moe_group_rows), or this chunk's alone */
                    const int G = q38cc_moe_group_rows(ch, m, c0, g0 + gn, rows);
                    gs = c0; ge = G > 0 ? c0 + G : c0;
                    if (G > 0 && !q38cc_moe_group(ch, m, l, i, c0, g0, G, rows, c0 == g0, c0 + G >= g0 + gn)) { free(cs_all); return -1; }
                }
                if (!q38cc_frame_begin(ch)) { free(cs_all); return -1; }
                ch->hyp = ch->hyper_all; ch->hoff = c0 - g0;
                int ok = 1;
                if (c0 < ge) ok = ok && q38cc_moe_group_take(ch, m, l, i, c0 - gs, n);
                else ok = ok && q38cc_gr_read(ch, m, &l->mlp_gr, ch->o_mn[i], n, ch->inj_m) &&
                          q38cc_dm_layer_lm(ch, m, l, i, n, c0 == g0, c0 + n >= g0 + gn);
                ok = ok && q38cc_moe_apply(ch, m, n, ch->dm_y);
                if (ok && want_streams && i == L - 1) ok = cc_copy(ch->hypd, 0, ch->hyper_all, (size_t)(c0 - g0) * W, (size_t)n * W);   /* q38cc_forward reserved it */
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
                if (want_streams && i == L - 1) memcpy(hyper_h + (size_t)c0 * W, cc_ptr(ch->hypd), (size_t)n * W * 4);   /* the frame is done */
                /* layer i's books for these rows (q38cc_dm_books' for one layer; its K/V rows went down after the first pass) */
                const int *idv = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
                for (int s = 0; s < n; s++) { rt_count(i, idv + s * K, K); if (!ch->own) qt_chain_count(i, idv + s * K, K); }
                for (int j = 0; j < n * K; j++) if (idv[j] >= 0) {
                    q38_ehit_mark(m, i, idv[j]);
                    ch->routes++;
                    if (ch->dm_res[(size_t)i * E + idv[j]] >= 2) ch->pool_reads++;
                }
                if (ch->own && ch->promo_rate > 0) { q38cc_heat_layer(ch, m, i, n); if (i == L - 1) q38cc_heat_tick(ch, m, n); }
                if (i == L - 1) nlast = n;
                ch->dn_where = Q38CC_DEV; ch->host_zero = 0;
            }
        }
    }
    free(cs_all);
    if (!cc_finish()) return -1;
    if (ch->own && ch->promo_rate > 0 && nlast > 0) {   /* the picks chunk by chunk ran after each chunk */
        memset(ch->dm_ran, 1, (size_t)L);
        for (int k = 0; k < (S + rows - 1) / rows; k++) q38cc_promote_pick(ch, m, nlast);
    }
    memcpy(logit, cc_ptr(ch->outd), (size_t)nlogits * V * sizeof(float));
    ch->lm_forwards++;
    return 1;
}

/* ---- a prompt chunk in two lanes --------------------------------------------------------
 * A short prompt's chunk with the pool's slots on the CPU (q38cc_pc_step) left the card waiting
 * at every layer for the host's share of the routed experts: the next layer reads this one's
 * sum. Split in two lanes (the chunk's first rows and the rest), the device records a lane's
 * layer up to its own experts, then the other lane's, then each lane's sum once the host served
 * it, and the next layer's ops of a lane right after that lane's sum: while the host computes
 * lane A's slots of layer i the card runs lane B's layer i, and while it computes B's the card
 * runs A's sum and A's layer i + 1. Each lane's rows take the ops a chunk of its rows would (a
 * DeltaNet's or an attention's state and the PLE's ring see A's rows, then B's), each in its own
 * buffers (Q38CC_LANE_BUFS); COLI_CUDA_CHAIN_LANES=0 keeps one lane. */
#define Q38CC_LANE_ROWS 384   /* COLI_CUDA_CHAIN_LANES unset: a chunk of this many rows or more */
static int q38cc_lanes_min(void) {
    static int n = -1;
    if (n < 0) {
        const char *e = getenv("COLI_CUDA_CHAIN_LANES");
        n = e && *e ? atoi(e) : Q38CC_LANE_ROWS;
        if (n < 0) n = 0;
        if (n > 0 && n < 32) n = 32;   /* each lane a tile of rows at least */
    }
    return n;
}
static int q38cc_lanes_on(void) { return q38cc_lanes_min() > 0; }
/* lane k's buffers in (0: the chunk's first rows, 1: the rest) */
static void q38cc_lane_use(Q38CChain *ch, int k) {
    if (ch->lane_cur == k) return;
#define Q38CC_LS(f) { CcBuf *t = ch->f; ch->f = ch->lane2.f; ch->lane2.f = t; }
    Q38CC_LANE_BUFS(Q38CC_LS)
#undef Q38CC_LS
    float *t = ch->host_routed; ch->host_routed = ch->lane2.host_routed; ch->lane2.host_routed = t;
    t = ch->host_emb; ch->host_emb = ch->lane2.host_emb; ch->lane2.host_emb = t;
    int r = ch->rows; ch->rows = ch->lane2.rows; ch->lane2.rows = r;
    ch->lane_cur = k;
}
/* the first lane's rows of a chunk of n: half, on a multiple of 16 */
static int q38cc_lane_split(int n) { int a = (n / 2 + 15) / 16 * 16; return a < n ? a : n / 2; }
/* whether a chunk of n rows can have its second lane's buffers (the first lane's are the chain's
 * own): they are there, or within the rows the chain's reserve counted (q38cc_reserve_bytes,
 * COLI_CUDA_CHAIN_ROWS of them), else where the card has their bytes and half a GiB besides (the
 * tier's allowance keeps a GiB of the card out of its own). With a GiB besides, after a long prompt
 * had the chain's rows at 2048, a 585-token prompt found 1.1-1.4 GB free (unsloth's UD-IQ2_XS on an
 * RTX PRO 4000) and ran in one lane with every pooled slot on the CPU: 320-353 tok/s against
 * 541 in two (UD-Q2_0: 376-392 against 681) */
static int q38cc_lane_fits(Q38CChain *ch, Model *m, int n) {
    Cfg *c = &m->c; int H = c->hidden;
    const int nb = n - q38cc_lane_split(n), rb = ch->lane2.rows > nb ? ch->lane2.rows : nb;
    if ((long long)ch->rows + rb <= q38cc_rows_env() || rb <= ch->lane2.rows) return 1;
    const size_t K = (size_t)c->topk, cap = (size_t)(m->kv_cap > 0 ? m->kv_cap : 1);
    const size_t per = (q38cc_row_floats(c, cap) + K * ((size_t)c->inter + (size_t)H) + (size_t)H + 2 * (size_t)c->layers * K) * 4;
    size_t fr = 0, tot = 0;
    return coli_cuda_mem_info(ch->dev, &fr, &tot) && fr >= (size_t)(rb - ch->lane2.rows) * per + ((size_t)1 << 29);
}
static int q38cc_lane_ready(Q38CChain *ch, Model *m, int n) {
    Cfg *c = &m->c; int H = c->hidden, Ep = c->ngram_heads * c->ngram_head_dim; if (Ep < 1) Ep = 1;
    const int nb = n - q38cc_lane_split(n), rb = ch->lane2.rows > nb ? ch->lane2.rows : nb;
    if (!q38cc_lane_fits(ch, m, n)) return 0;
    q38cc_lane_use(ch, 1);
    ch->rows = rb;
    int ok = q38cc_bufs(ch, m, rb) && q38cc_dm_begin(ch, m, nb);
    if (ok && ch->lane_rows < rb) {
        float *hr = realloc(ch->host_routed, (size_t)rb * H * sizeof(float)), *he;
        if (hr) ch->host_routed = hr;
        he = hr ? realloc(ch->host_emb, (size_t)rb * Ep * sizeof(float)) : NULL;
        if (he) ch->host_emb = he;
        ok = hr && he;
        if (ok) ch->lane_rows = rb;
    }
    q38cc_lane_use(ch, 0);
    return ok;
}
/* lane k's layer i up to its routes and shared expert (*v: its host word, 0 none); pending: the layer before left its sum in dm_y */
static int q38cc_lane_a(Q38CChain *ch, Model *m, int i, int n, int c0, int pb, int pending, unsigned *v) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    int ok = !pending || q38cc_moe_apply(ch, m, n, ch->dm_y);
    if (ok && i == c->ple_layer) ok = q38cc_ple(ch, m, n, c0, 0);
    ok = ok && q38cc_gr_read(ch, m, &l->attn_gr, ch->o_an[i], n, ch->inj_a);
    ok = ok && (c->is_attn[i] ? q38cc_attention(ch, m, l, i, n, pb) : q38cc_deltanet(ch, m, l, i, n, c0, 0));
    ok = ok && q38cc_gr_apply(ch, m, ch->inj_a, n);
    ok = ok && q38cc_gr_read(ch, m, &l->mlp_gr, ch->o_mn[i], n, ch->inj_m);
    ch->dm_ran[i] = 1;
    return ok && q38cc_dm_route(ch, m, l, i, n, v);
}
/* both lanes' slots of layer i on the device in one call (lane A's buffers in): lane B's rows,
 * routes and weights after lane A's in A's buffers, the experts' weights read once for the chunk */
static int q38cc_lane_card(Q38CChain *ch, Model *m, int i, int na, int nb) {
    Cfg *c = &m->c; int H = c->hidden, K = c->topk;
    const Q38CCLane *b = &ch->lane2;
    const size_t io = (size_t)i * ch->rows * K, ib = (size_t)i * b->rows * K;   /* layer i's routes in each lane's layout */
    const int st = ch->ls_i == i && ch->ls_n > 0;   /* its staged experts' entries in the table for the call, then back */
    int ok = cc_copy(ch->mixed, (size_t)na * H, b->mixed, 0, (size_t)nb * H) && cc_copy(ch->dm_ids, io + (size_t)na * K, b->dm_ids, ib, (size_t)nb * K) &&
             cc_copy(ch->dm_w, io + (size_t)na * K, b->dm_w, ib, (size_t)nb * K);
    ok = ok && (!st || (cc_fetch_wait() && cc_copy_regions(ch->dm_tabd, ch->ls_ent, ch->ls_rg, ch->ls_n)));
    ok = ok && q38cc_dm_card(ch, m, i, na + nb);
    return ok && (!st || (cc_copy_regions(ch->dm_tabd, ch->ls_ent, ch->ls_rg + ch->ls_cap, ch->ls_n) && cc_fetch_done()));
}
/* Lane A's routes of layer i down, the layer's pooled experts it routes at least twice (the
 * most routed first, up to COLI_CUDA_CHAIN_LANE_STAGE, 0 none) go over the bus into the
 * staging (dm_stage) on the copy stream while the card runs lane B's layer, and the chunk's
 * call of the device's slots (q38cc_lane_card) holds them; the host's serves of both lanes
 * leave them out. A prompt's pooled slots gather on few experts (a 470-token prompt's, two
 * thirds on experts with 16 or more of them) and an expert's copy (1.44 MiB at 12.9 GB/s on
 * an RTX PRO 4000) costs a few of its slots on the CPU. */
static int q38cc_lane_stage_cap(void) {
    static int n = -1;
    if (n < 0) { const char *e = getenv("COLI_CUDA_CHAIN_LANE_STAGE"); n = e && *e ? atoi(e) : 24; if (n < 0) n = 0; }
    return n;
}
static int q38cc_lane_stage(Q38CChain *ch, Model *m, int i, int na) {
    Cfg *c = &m->c; int E = c->experts, K = c->topk;
    for (int j = 0; j < ch->ls_n; j++) ch->ls_mark[ch->ls_e[j]] = 0;
    ch->ls_n = 0; ch->ls_i = -1;
    int cap = q38cc_lane_stage_cap(); if (cap > q38cc_stage_cap(ch, i)) cap = q38cc_stage_cap(ch, i);
    int64_t gb, db;
    if (cap <= 0 || !ch->dm_stage || !ch->pool_of || !cc_fetch_ready() || !q38cc_pool_types(m, i, &gb, &db)) return 1;
    if (ch->ls_cap < cap) {
        free(ch->ls_cnt); free(ch->ls_e); free(ch->ls_cand); free(ch->ls_mark); free(ch->ls_fr); free(ch->ls_rg);
        ch->ls_cnt = calloc((size_t)E, sizeof(int)); ch->ls_e = malloc((size_t)cap * sizeof(int)); ch->ls_cand = malloc((size_t)E * sizeof(int));
        ch->ls_mark = calloc((size_t)E, 1); ch->ls_fr = malloc((size_t)cap * sizeof(CcFetch)); ch->ls_rg = malloc((size_t)2 * cap * sizeof(CcRegion));
        ch->ls_cap = 0;
        if (!ch->ls_cnt || !ch->ls_e || !ch->ls_cand || !ch->ls_mark || !ch->ls_fr || !ch->ls_rg ||
            !cc_reserve(&ch->ls_ent, (size_t)(cap + 1) * sizeof(CcExpert), CC_UP)) return 1;   /* no staging: the CPU serves them */
        memset((CcExpert *)cc_ptr(ch->ls_ent) + cap, 0, sizeof(CcExpert));   /* the entry a pooled expert has in the decode table */
        ch->ls_cap = cap;
    }
    const int *ids = (const int *)cc_ptr(ch->dm_idsd) + (size_t)i * ch->rows * K;
    const uint8_t *res = ch->dm_res + (size_t)i * E;
    const int *po = ch->pool_of + (size_t)i * E;
    int *cnt = ch->ls_cnt, *cand = ch->ls_cand, nc = 0;
    for (int a = 0; a < na * K; a++) {
        const int e = ids[a];
        if (e >= 0 && e < E && res[e] >= 2 && po[e] >= 0 && cnt[e]++ == 1) cand[nc++] = e;   /* its second slot */
    }
    int h = 0;
    for (int q = 0; q < nc && h < cap; q++) {   /* the most routed first: a selection over few */
        int best = q;
        for (int r = q + 1; r < nc; r++) if (cnt[cand[r]] > cnt[cand[best]] || (cnt[cand[r]] == cnt[cand[best]] && cand[r] < cand[best])) best = r;
        int t = cand[q]; cand[q] = cand[best]; cand[best] = t;
        ch->ls_e[h++] = cand[q];
    }
    for (int q = 0; q < nc; q++) cnt[cand[q]] = 0;
    for (int a = 0; a < na * K; a++) { const int e = ids[a]; if (e >= 0 && e < E) cnt[e] = 0; }
    if (!h) return 1;
    const size_t st = q38cc_lstride(ch, i), span = 2 * Q38CC_PAD(gb) + (size_t)db, EW = sizeof(CcExpert) / 4;
    const char *base = (const char *)cc_dev_addr(ch->dm_stage);
    CcExpert *ent = (CcExpert *)cc_ptr(ch->ls_ent);
    if (!base) return 1;
    for (int j = 0; j < h; j++) {
        const int e = ch->ls_e[j], ps = po[e];
        const char *g = base + (size_t)j * st;
        ch->ls_fr[j] = (CcFetch){q38cc_pool_hptr(ch, ps), (size_t)j * st, span};
        memset(&ent[j], 0, sizeof ent[j]);
        ent[j].g = g; ent[j].u = g + Q38CC_PAD(gb); ent[j].d = g + 2 * Q38CC_PAD(gb);
        const size_t dst = ((size_t)i * E + (size_t)e) * EW;
        ch->ls_rg[j] = (CcRegion){dst, (size_t)j * EW, EW};
        ch->ls_rg[ch->ls_cap + j] = (CcRegion){dst, (size_t)ch->ls_cap * EW, EW};
        ch->ls_mark[e] = 1;
    }
    ch->ls_n = h; ch->ls_i = i;
    ch->ls_experts += (unsigned long long)h; ch->ls_layers++;
    return cc_fetch(ch->dm_stage, ch->ls_fr, h);
}
/* the chunk [c0, c0 + n) in two lanes (the frame begun, the routed experts' table and buffers the
 * first lane's): 1 done, 0 lost; the logits, streams and mixed rows as q38cc_forward leaves them */
static int q38cc_chunk_lanes(Q38CChain *ch, Model *m, const int *ids, int c0, int n, int pb, int S, int nlogits,
                             float *hyper_h, int want_streams, float *mixed_h) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, L = c->layers, V = c->vocab, half = c->rotary_dim / 2;
    const int na = q38cc_lane_split(n);
    const int cl[2] = {c0, c0 + na}, nl[2] = {na, n - na}, pl[2] = {pb, pb + na};
    int ok = 1;
    unsigned v[2] = {0, 0};
    for (int k = 0; k < 2 && ok; k++) {   /* each lane's rows: their angles, PLE rows and streams */
        q38cc_lane_use(ch, k);
        if (half) {
            float *cs = (float *)cc_ptr(ch->cs);
            for (int s = 0; s < nl[k]; s++) for (int j = 0; j < half; j++) {
                float ang = (float)(pl[k] + s) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
                cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
            }
        }
        if (c->ple_layer >= 0 && c->ple_layer < L) q38cc_ple_rows(m, ids, cl[k], nl[k], ch->host_emb);
        ok = q38cc_hyper_in(ch, hyper_h + (size_t)cl[k] * W, nl[k], W);
    }
    for (int k = 0; k < 2 && ok; k++) { q38cc_lane_use(ch, k); ok = q38cc_lane_a(ch, m, 0, nl[k], cl[k], pl[k], 0, &v[k]); }
    for (int i = 0; i < L && ok; i++)
        for (int k = 0; k < 2 && ok; k++) {
            q38cc_lane_use(ch, k);
            if (!k) {   /* lane A's routes down: its most routed pooled experts over the bus, then the chunk's slots on the card */
                if (v[0] && q38cc_lane_stage_cap() > 0) ok = q38cc_dm_wait(ch, i, v[0]) && q38cc_lane_stage(ch, m, i, nl[0]);
                ok = ok && q38cc_lane_card(ch, m, i, nl[0], nl[1]);
            }
            if (ok && v[k]) ok = q38cc_dm_serve(ch, m, i, nl[k], v[k]);   /* the device runs the other lane meanwhile */
            ok = ok && q38cc_dm_sum(ch, m, i, nl[k], v[k], k ? ch->lane2.dm_hid : ch->dm_hid, n, cl[k] - c0);   /* lane A's hid: the chunk's */
            if (ok && i + 1 < L) ok = q38cc_lane_a(ch, m, i + 1, nl[k], cl[k], pl[k], 1, &v[k]);
        }
    for (int j = 0; j < ch->ls_n; j++) ch->ls_mark[ch->ls_e[j]] = 0;
    ch->ls_n = 0; ch->ls_i = -1;
    for (int k = 0; k < 2 && ok; k++) {   /* each lane's last sum, its streams, mixed rows and logits */
        q38cc_lane_use(ch, k);
        const int r0 = cl[k] - c0, n1 = nl[k];
        ok = q38cc_moe_apply(ch, m, n1, ch->dm_y);
        if (ok && want_streams) ok = cc_copy(ch->hypd, (size_t)r0 * W, ch->hyper, 0, (size_t)n1 * W);
        int lo0 = S - nlogits > cl[k] ? S - nlogits - cl[k] : 0, lo_n = n1 - lo0;
        int need_final = mixed_h || (lo0 < n1 && cl[k] + n1 > S - nlogits);
        if (ok && need_final) ok = q38cc_gr_read(ch, m, &m->final_gr, ch->o_fn, n1, NULL);
        if (ok && mixed_h) ok = cc_copy(ch->find, (size_t)r0 * H, ch->mixed, 0, (size_t)n1 * H);
        if (ok && need_final && lo_n > 0 && cl[k] + n1 > S - nlogits) {
            int dst = cl[k] + lo0 - (S - nlogits);
            ok = cc_matmul(q38cc_t(ch, &m->lm_head), ch->mixed, (size_t)lo0 * H, ch->outd, (size_t)dst * V, lo_n);
        }
    }
    ok = ok && cc_submit(0) && cc_finish();
    for (int k = 0; k < 2 && ok; k++) { q38cc_lane_use(ch, k); q38cc_dm_books(ch, m, nl[k], pl[k]); }
    q38cc_lane_use(ch, 0);
    ch->lane_chunks++;
    return ok;
}

/* A caller that only wants each logit row's argmax (m->greedy_out: a greedy decode's verify or
 * step) has the card pick them (cc_pick_rows) and copy down two words a row instead of the
 * rows' 248,320 logits, which the host copied and scanned again (Qwen3.8-Flash-Next on an RTX
 * PRO 4000: 4 MB a verify of four rows, then a row's copy and scan a token, 0.7-0.9 ms between
 * a verify and the next draft). A decode step's or verify's rows of one chunk.
 * COLI_CUDA_CHAIN_GREEDY_CARD=0: the logits down. */
static int q38cc_greedy_ok(Model *m, int S, int nlogits) {
    const char *e = getenv("COLI_CUDA_CHAIN_GREEDY_CARD");
    const int on = !(e && *e == '0');
    const Q38CChain *ch = g_cuda_chain ? q38cc_of(m) : NULL;
    return on && m->greedy_out && ch && ch->ok && !ch->failed && S <= Q38_SPEC_ROWS && nlogits >= 1 && nlogits <= S &&
           S <= q38cc_rows_env() && cc_pick_rows_ready();
}
/* A server's first long prompt grew the chain's buffers to a chunk's rows on its way: page-locked
 * host memory at 0.2 ms a MB (the K/V rows down, 122 MB, the streams down, 84 MB, a prompt-end
 * snapshot, 118 MB), and a growth frees the buffer it replaces, which waits for the stream (the
 * MTP head's ones: 32 ms of queued work). With the backend's module, which the runtime loads at the
 * first tile, a 4180-token prompt's first token came 0.25-0.3 s later as a server's first request
 * than as its second (unsloth UD-Q2_0 on an RTX PRO 4000). They are reserved before the first
 * request instead, at the sizes that prompt leaves, inside what q38cc_reserve_bytes kept for them. */
static void q38cc_prewarm(Model *m) {
    Q38CChain *ch = g_cuda_chain ? q38cc_of(m) : NULL;
    if (!ch || !ch->ok || ch->failed || cc_lost()) return;
    Cfg *c = &m->c; const int CH = q38cc_rows_env(), W = c->hc_width, H = c->hidden, C = c->hc_count;
    const double t0 = now_s();
    int ok = q38cc_mirror(ch, m) && q38cc_scratch(ch, m, CH);
    if (ok && m->mtp) ok = cc_reserve(&ch->hypd, (size_t)CH * W * 4, CC_DOWN);   /* q38cc_forward's want_streams */
    if (ok && ch->dm_on && c->experts > 0) ok = q38cc_dm_bufs(ch, m);
    if (ok && ch->mtp_on && m->mtp) ok = cc_reserve(&ch->fin, (size_t)CH * H * 4, CC_UP) && cc_reserve(&ch->ones, (size_t)CH * C * 4, CC_DEV);
    if (ok) ok = cc_reserve(&ch->ps, q38cc_snap_floats(ch, m) * 4, CC_DOWN);
    const int loaded = ok && cc_preload();
    if (!ok) { fprintf(stderr, "[chain] qwen38: a chunk's buffers not reserved before the first request; its prompt grows them\n"); return; }
    fprintf(stderr, "[chain] qwen38: a %d-row chunk's buffers reserved%s before the first request (%.0f ms)\n", CH,
            loaded ? " and the backend's kernels loaded" : "", (now_s() - t0) * 1e3);
}
/* Every layer for S rows. hyper_h: the rows' streams in, the final ones out when
 * want_streams; mixed_h (or NULL): every row's final mixed row, for the prefill
 * read-out; logit: the last nlogits rows' logits (a greedy caller's picks instead, in
 * m->greedy_out, q38cc_greedy_ok). 0 = not taken (nothing changed, or the device was lost
 * and the state rebuilt: the CPU runs the step), 1 = done. */
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
        q38cc_mirror_free(ch, c);   /* the chain stays off (failed): its mirrors would hold the card until the process ends */
        ch->failed = 1;
        return 0;
    }
    Q38CCSrv *sv = NULL;   /* a decode step's pooled slots on the server's threads (q38cc_srv_of) */
    const int greedy = !mixed_h && q38cc_greedy_ok(m, S, nlogits);   /* the rows' argmax only (q38_forward sized logit for it) */
    if (greedy && (!cc_reserve(&ch->outdev, (size_t)nlogits * V * 4, CC_DEV) || !cc_reserve(&ch->gpick, (size_t)nlogits * 8, CC_DOWN))) {
        if (cc_lost()) goto lost;
        fprintf(stderr, "[chain] qwen38: device memory for a greedy step's logits refused; this step on the CPU\n");
        return 0;
    }
    int half = c->rotary_dim / 2;
    int snapped = 0;
    double w0 = ch->wait_ms, h0 = ch->host_ms;
    ch->pc_prompt = q38cc_pc_take(ch, m, S);   /* a short prompt's pooled experts on the CPU */
    if (S > Q38CC_DM_ROWS && ch->pool_of) { if (ch->pc_prompt) ch->pc_took++; else ch->pc_staged++; }
    if (S > rows && rows > Q38CC_DM_ROWS && E > 0 && ch->dm_on && q38cc_layer_major() && !mixed_h &&
        m->snap_rows <= 0) {
        int r = q38cc_forward_lm(m, ch, ids, S, pos_base, nlogits, hyper_h, want_streams, logit, rows);
        ch->hyp = NULL; ch->hoff = 0;   /* the streams' ops back on hyper, however it ended */
        if (r < 0) goto lost;
        if (r == 2) { q38cc_dm_end(ch); return 0; }   /* stopped: not taken */
        if (r > 0) { q38cc_dm_end(ch); ch->snap_valid = 0; ch->forwards++; return 1; }
    }
    for (int c0 = 0; c0 < S; c0 += rows) {
        if (coli_prompt_break()) { q38cc_dm_end(ch); return 0; }   /* a served prompt's stop, between two chunks */
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
        /* a short prompt's chunk with the pool's slots on the CPU: two lanes (q38cc_chunk_lanes) */
        int lanes = dm && q38cc_lanes_on() && n >= q38cc_lanes_min() && ns == 0 && q38cc_pc_step(ch, n) && q38cc_dm_tab(ch, n) == ch->dm_tabd;
        for (int i = 0; i < L && lanes; i++) lanes = !ch->dm_part[i];
        if (lanes && !q38cc_lane_ready(ch, m, n)) {
            if (cc_lost()) goto lost;
            lanes = 0;
        }
        if (lanes) {
            if (!q38cc_frame_begin(ch) || !q38cc_push_state(ch, m, pb) ||
                !q38cc_chunk_lanes(ch, m, ids, c0, n, pb, S, nlogits, hyper_h, want_streams, mixed_h)) goto lost;
            if (want_streams) memcpy(hyper_h + (size_t)c0 * W, cc_ptr(ch->hypd), (size_t)n * W * 4);
            if (mixed_h) memcpy(mixed_h + (size_t)c0 * H, cc_ptr(ch->find), (size_t)n * H * 4);
            for (int i = 0; i < L; i++) if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
            ch->dn_where = Q38CC_DEV; ch->host_zero = 0;
            continue;
        }
        if (dm && (sv = q38cc_srv_of(ch, m, n)))
            q38cc_srv_open(sv, (const volatile unsigned *)cc_ptr(ch->dm_dfl), (volatile unsigned *)cc_ptr(ch->dm_hfl),
                           (const float *)cc_ptr(ch->mixd), (float *)cc_ptr(ch->dm_hy));
        if (dm) cc_host_wait_arm(ch->dm_hfl);
        if (!q38cc_frame_begin(ch) || !q38cc_hyper_in(ch, hyper_h + (size_t)c0 * W, n, W) ||
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
                int fg, fd;
                if (ok && v && sv && qt_chain_layer_format(i, &fg, &fd) && fg >= 100 && fd >= 100) {   /* the server's from here */
                    const size_t io = (size_t)i * ch->rows * c->topk;
                    q38cc_srv_post(sv, i, n, v, fg - 100, fd - 100, (const int *)cc_ptr(ch->dm_idsd) + io, (const float *)cc_ptr(ch->dm_wd) + io,
                                   ch->dm_res + (size_t)i * E);
                    v = 0;
                }
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
            ok = cc_matmul(q38cc_t(ch, &m->lm_head), ch->mixed, (size_t)lo0 * H, greedy ? ch->outdev : ch->outd, (size_t)dst * V, lo_n);
            if (ok && greedy) ok = cc_pick_rows(ch->outdev, (size_t)dst * V, ch->gpick, V, lo_n);   /* one chunk: dst 0 */
        }
        double t0 = now_s();
        ok = ok && cc_submit(0);
        if (ok && pend_v) ok = q38cc_dm_serve(ch, m, pend_i, n, pend_v);   /* the last hybrid layer's, the tail recorded */
        /* a decode step's or a verify's frame, waited for alone: cc_finish waits for the copy stream too,
         * the experts the chain promotes, which land on their own (q38cc_promote_settle); a prompt
         * chunk's, everything (its stagings) */
        ok = ok && (n > Q38CC_DM_ROWS ? cc_finish() : cc_submit(1));
        if (sv) { ok = q38cc_srv_close(sv, ok, ch->dm_seq) && ok; q38cc_srv_books(ch); sv = NULL; }
        m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0; ch->wait_ms += (now_s() - t0) * 1e3;
        if (!ok || (dm && cc_host_wait_gave_up(ch->dm_hfl, "qwen38"))) goto lost;
        if (dm) q38cc_dm_books(ch, m, n, pb);
        if (want_streams) memcpy(hyper_h + (size_t)c0 * W, cc_ptr(ch->hypd), (size_t)n * W * 4);
        if (mixed_h) memcpy(mixed_h + (size_t)c0 * H, cc_ptr(ch->find), (size_t)n * H * 4);
        for (int i = 0; i < L; i++) if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
        ch->dn_where = Q38CC_DEV; ch->host_zero = 0;
    }
    q38cc_dm_end(ch);
    if (greedy) {
        const int *pk = (const int *)cc_ptr(ch->gpick);
        for (int r = 0; r < nlogits; r++) m->greedy_out[r] = pk[2 * r];
        m->greedy_done = 1;
        ch->greedy_rows += (unsigned long long)nlogits;
    } else memcpy(logit, cc_ptr(ch->outd), (size_t)nlogits * V * sizeof(float));
    ch->snap_valid = snapped;
    ch->forwards++;
    if (S <= Q38_SPEC_ROWS) { ch->dec_n++; ch->dec_wait_ms += ch->wait_ms - w0; ch->dec_host_ms += ch->host_ms - h0; }   /* a decode step, or a verify's */
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (sv) { q38cc_srv_close(sv, 0, ch->dm_seq); q38cc_srv_books(ch); }
    if (!cc_lost()) cc_finish();
    q38cc_dm_end(ch);
    q38cc_recover(m, pos_base);
    return 0;
}

/* The MTP head's rows from head row pos on are the host's (the CPU ran them): the device's
 * mirror of its layer behind them */
static void q38cc_mtp_host(Model *m, int pos) {
    Q38CChain *ch = q38cc_of(m);
    if (!ch || !ch->ok || !ch->mtp_on) return;
    int L = m->c.layers, R = m->c.idx_ratio > 0 ? m->c.idx_ratio : 1;
    if (ch->kv_valid[L] > pos) ch->kv_valid[L] = pos;
    if (ch->pk_valid[L] > pos / R) ch->pk_valid[L] = pos / R;
}
/* The MTP head over S pairs on the device, as q38_mtp_rows runs them on the CPU (qwen38_chain.h's
 * q38c_mtp_rows, on CUDA): row s pairs the model's four streams (streams, row s) with the token
 * next[s], at head row pos_base+s. Its input: the embedding row's norm and fc_embedding added to
 * each stream's norm and fc_hidden (wiring a: one norm over the four streams); its layer:
 * attention over its own K/V mirror (its own RoPE base), the shared expert on the device and the
 * routed experts the host's step (q38_moe_ex over the head's own experts, the K/V rows down
 * meanwhile); with `logit`, the last row's mixer and lm_head; with `hidden`, the last row's
 * streams as the layer leaves them (what a deeper draft reads). settle: settled pairs (the
 * head's rows end there, mtp_len), else a deeper draft's tail row, which the settled pair at
 * that row overwrites before anything reads past it. With `pick` and the model's draft list
 * (draft_vocab.h), only the listed rows of lm_head are scored (cc_matmul_rows: each the bits the
 * whole lm_head gives it) and the card picks (cc_pick: the best, the lowest place on a tie, and its
 * probability): *pick and *pick_p, and 2 is returned; else 1 with logit. 0: not here (the CPU runs
 * it); a frame that fails gives the device up, the state on the CPU (q38cc_recover), and the CPU runs it. */
static int q38cc_mtp_rows(Model *m, const float *streams, const int *next, int S, int pos_base, float *logit,
                          float *hidden, int settle, int *pick, float *pick_p) {
    Q38CChain *ch = q38cc_of(m);
    if (!g_cuda_chain || !ch || !ch->ok || ch->failed || !ch->mtp_on || S < 1 || cc_lost()) return 0;
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, E = c->experts, V = c->vocab, L = c->layers;
    int half = c->rotary_dim / 2, R = c->idx_ratio > 0 ? c->idx_ratio : 1;
    Layer *l = &m->L[L];
    Q38CCSrv *sv = NULL;   /* the head's experts on the pool's server (q38cc_head_pool) */
    if (!q38cc_mirror(ch, m)) return 0;
    /* the draft list scored on the card: the list (rebuilt by a request, tokens appended after) into
     * host memory the card reads, the listed lm_head rows, the pick */
    DraftVocab *dv = pick ? m->draft_vocab : NULL;
    ColiCudaTensor *lmt = dv ? q38cc_t(ch, &m->lm_head) : NULL;
    int vn = dv ? dv->list_n : 0;   /* matmul_rows takes the lm_head as the card holds it: f32, int8 rows or a GGUF's blocks */
    int voc = logit && lmt && vn > 0 && cc_matmul_rows_ready() && cc_pick_ready() &&
              cc_reserve(&ch->dv_ids, (size_t)V * 4, CC_UP) && cc_reserve(&ch->dv_y, (size_t)V * 4, CC_DEV) &&
              cc_reserve(&ch->dv_x, (size_t)H * 4, CC_DEV) && cc_reserve(&ch->dv_pick, 64, CC_DOWN);
    if (voc) memcpy(cc_ptr(ch->dv_ids), dv->list, (size_t)vn * sizeof(int));
    int CH = q38cc_rows_env(), rows = S < CH ? S : CH;
    float *emb = (float *)malloc((size_t)rows * (H + C) * sizeof(float)), *ones = emb ? emb + (size_t)rows * H : NULL;
    if (!emb || !q38cc_scratch(ch, m, rows) || (hidden && !cc_reserve(&ch->hypd, (size_t)rows * W * 4, CC_DOWN)) ||
        !cc_reserve(&ch->fin, (size_t)rows * H * 4, CC_UP) || !cc_reserve(&ch->ones, (size_t)rows * C * 4, CC_DEV)) { free(emb); return 0; }
    if (ch->ones != ch->ones_at) { ch->ones_at = ch->ones; ch->ones_n = 0; }   /* a new buffer: its ones to write */
    /* the head's experts on the card (q38cc_head_card) for a prompt's rows too: their routes and
     * the routed experts' scratch for `rows` (a prompt's head on the host's step took 100 ms of a
     * 452-token prompt's 700 on an i5-13500) */
    const size_t K = (size_t)c->topk;
    const int card_ok = ch->hd_tab && cc_reserve(&ch->hd_ids, (size_t)rows * K * 4, CC_DEV) && cc_reserve(&ch->hd_w, (size_t)rows * K * 4, CC_DEV) &&
                        cc_reserve(&ch->dm_hid, cc_moe_hid_floats(rows, (int)K, H, c->inter, E) * 4, CC_DEV) &&
                        cc_reserve(&ch->dm_y, (size_t)rows * H * 4, CC_DEV);
    for (int k = 0; k < rows * C; k++) ones[k] = 1.f;
    /* the rows from pos_base on are rewritten: the mirror's watermarks behind them */
    if (ch->kv_valid[L] > pos_base) ch->kv_valid[L] = pos_base;
    if (ch->pk_valid[L] > pos_base / R) ch->pk_valid[L] = pos_base / R;
    for (int base = 0; base < S; base += rows) {
        int n = S - base < rows ? S - base : rows, pb = pos_base + base, last = base + n == S;
        for (int s = 0; s < n; s++) q38_embed_row(m, next[base + s], pb + s + 1, emb + (size_t)s * H);
        memcpy(cc_ptr(ch->fin), emb, (size_t)n * H * sizeof(float));
        if (half) {
            float *cs = (float *)cc_ptr(ch->cs);
            for (int s = 0; s < n; s++) for (int j = 0; j < half; j++) {
                float ang = (float)(pb + s) / powf(c->mtp_theta, (float)(2 * j) / c->rotary_dim);
                cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
            }
        }
        if (!q38cc_frame_begin(ch) || !q38cc_hyper_in(ch, streams + (size_t)base * W, n, W) ||
            (n > ch->ones_n && !cc_write(ch->ones, 0, ones, (size_t)n * C * 4)) || !q38cc_push_kv(ch, m, L, pb)) goto lost;
        if (n > ch->ones_n) ch->ones_n = n;   /* ones are ones: written once for as many rows */
        CcNorm en = {n, H, 1, 0, H, H, 0, H, H, (int)ch->o_mtp_emb, 0, CC_NORM_ADD1, c->eps, 1.f};
        CcNorm hn = m->mtp_wiring == Q38_MTP_WHOLE_NORM
                        ? (CcNorm){n, W, 1, 0, W, W, 0, W, W, (int)ch->o_mtp_hid, 0, CC_NORM_ADD1, c->eps, 1.f}
                        : (CcNorm){n * C, H, C, 0, W, H, 0, W, H, (int)ch->o_mtp_hid, C, CC_NORM_ADD1, c->eps, 1.f};
        CcEw add = {CC_EW_HC_APPLY, n * W, H, C, 0, 1, 0, 0, 0, 0, 0, 1.f};   /* each stream += the embedding's row */
        int ok = cc_norm(ch->fin, ch->prm, ch->mixed, &en) && cc_matmul(q38cc_t(ch, &m->mtp_fc_emb), ch->mixed, 0, ch->blk, 0, n) &&
                 cc_norm(ch->hyper, ch->prm, ch->hn, &hn) && cc_matmul(q38cc_t(ch, &m->mtp_fc_hid), ch->hn, 0, ch->hyper, 0, n * C) &&
                 cc_ew(ch->hyper, ch->ones, ch->blk, NULL, NULL, &add);
        ok = ok && q38cc_gr_read(ch, m, &l->attn_gr, ch->o_an[L], n, ch->inj_a) && q38cc_attention(ch, m, l, L, n, pb) &&
             q38cc_gr_apply(ch, m, ch->inj_a, n) && q38cc_gr_read(ch, m, &l->mlp_gr, ch->o_mn[L], n, ch->inj_m);
        const int card = card_ok;   /* the head's experts on the card */
        const int hyb = !card && ch->hd_base && ch->srv && ch->dm_dfl && n <= Q38CC_DM_ROWS;   /* on the pool's server */
        if (card) {   /* routed, computed and summed on the card, in Q2_0 (q38cc_head_card) */
            CcRoute rp = {n, E, c->topk, 0, 0, E, 0, 0, 0, c->topk, 0, c->topk, 1, 1, c->norm_topk ? 1 : 0, 1.f, 1.f, 0.f};
            CcMoe mp = {n, c->topk, H, c->inter, 100 + GQ_Q2_0, 0, 0, 3, 0, 0, H, 0, c->topk, 0, c->topk, 0, H, -1, E, 0, 100 + GQ_Q2_0, 0, 0};
            ok = ok && cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) && cc_route(ch->lg, NULL, ch->hd_ids, ch->hd_w, &rp) &&
                 q38cc_shared(ch, m, l, L, n) && cc_moe(ch->hd_tab, ch->mixed, ch->hd_ids, ch->hd_w, ch->dm_hid, ch->dm_y, NULL, &mp) &&
                 q38cc_moe_apply(ch, m, n, ch->dm_y);
            ch->hd_card_rows += (unsigned long long)n;
        } else if (hyb) {   /* routed on the device, the server computes them while the card runs the shared expert, one frame */
            CcRoute rp = {n, E, c->topk, 0, 0, E, 0, 0, 0, c->topk, 0, c->topk, 1, 1, c->norm_topk ? 1 : 0, 1.f, 1.f, 0.f};
            const unsigned v = ++ch->dm_seq;
            const size_t nk = (size_t)n * c->topk;
            sv = ch->srv;
            cc_host_wait_arm(ch->dm_hfl);
            q38cc_srv_open(sv, (const volatile unsigned *)cc_ptr(ch->dm_dfl), (volatile unsigned *)cc_ptr(ch->dm_hfl),
                           (const float *)cc_ptr(ch->mixd), (float *)cc_ptr(ch->dm_hy));
            ok = ok && cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) && cc_route(ch->lg, NULL, ch->hd_ids, ch->hd_w, &rp) &&
                 cc_copy(ch->hd_idsd, 0, ch->hd_ids, 0, nk) && cc_copy(ch->hd_wd, 0, ch->hd_w, 0, nk) &&
                 cc_copy(ch->mixd, 0, ch->mixed, 0, (size_t)n * H) && cc_signal(ch->dm_dfl, 0, v);
            if (ok) q38cc_srv_post(sv, L, n, v, ch->hd_tg, ch->hd_td, (const int *)cc_ptr(ch->hd_idsd), (const float *)cc_ptr(ch->hd_wd), ch->hd_res);
            ok = ok && q38cc_shared(ch, m, l, L, n) && cc_wait_host(ch->dm_hfl, 0, v) && q38cc_moe_apply(ch, m, n, ch->dm_hy);
            ch->hd_rows += (unsigned long long)n;
        } else {
            /* the routed experts the host's step: the frame ends at the router (A1), the shared expert
             * runs while the host routes them (A2) */
            ok = ok && cc_matmul(q38cc_t(ch, &l->router), ch->mixed, 0, ch->lg, 0, n) &&
                 cc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && cc_copy(ch->mixd, 0, ch->mixed, 0, (size_t)n * H) && cc_submit(1);
            if (!ok) goto lost;
            q38cc_kv_down(ch, m, L, n, pb);
            ok = q38cc_frame_begin(ch) && q38cc_shared(ch, m, l, L, n) && cc_submit(0);
            double t1 = now_s();
            q38_moe_ex(m, l, L, (const float *)cc_ptr(ch->mixd), n, ch->host_routed, (const float *)cc_ptr(ch->lgd), 1);
            memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)n * H * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && q38cc_frame_begin(ch) && q38cc_moe_apply(ch, m, n, ch->routed);
        }
        if (ok && last && hidden) ok = cc_copy(ch->hypd, 0, ch->hyper, (size_t)(n - 1) * W, (size_t)W);
        if (ok && last && logit)
            ok = q38cc_gr_read(ch, m, &m->mtp_mixer, ch->o_mtp_mix, n, NULL) &&
                 (voc ? cc_copy(ch->dv_x, 0, ch->mixed, (size_t)(n - 1) * H, (size_t)H) && cc_matmul_rows(lmt, ch->dv_ids, ch->dv_x, ch->dv_y, vn) &&
                            cc_pick(ch->dv_y, ch->dv_pick, vn)
                      : cc_matmul(q38cc_t(ch, &m->lm_head), ch->mixed, (size_t)(n - 1) * H, ch->outd, 0, 1));
        ok = ok && cc_submit(1);
        if (sv) { ok = q38cc_srv_close(sv, ok, ch->dm_seq) && ok; q38cc_srv_books(ch); sv = NULL; }
        if (!ok || (hyb && cc_host_wait_gave_up(ch->dm_hfl, "qwen38"))) goto lost;
        if (hyb || card) q38cc_kv_down(ch, m, L, n, pb);
        if (settle) { ch->kv_valid[L] = pb + n; m->mtp_len = pb + n; }
        if (last && hidden) memcpy(hidden, cc_ptr(ch->hypd), (size_t)W * sizeof(float));
        if (last && logit && voc) {
            const void *pk = cc_ptr(ch->dv_pick);
            const int b = ((const int *)pk)[0];
            *pick = dv->list[b >= 0 && b < vn ? b : 0]; if (pick_p) *pick_p = ((const float *)pk)[1];
            ch->drafts++; ch->dv_drafts++;
        } else if (last && logit) { memcpy(logit, cc_ptr(ch->outd), (size_t)V * sizeof(float)); ch->drafts++; }
        ch->mtp_rows += (unsigned long long)n; g_q38cc_mtp_rows += (unsigned long long)n;
    }
    free(emb);
    return voc ? 2 : 1;
lost:   /* a frame failed: the device is gone; the CPU takes over, the head too */
    free(emb);
    if (sv) { q38cc_srv_close(sv, 0, ch->dm_seq); q38cc_srv_books(ch); }
    if (!cc_lost()) cc_finish();
    q38cc_recover(m, m->kv_len);
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
    if (ch->dv_drafts) fprintf(stderr, "[chain] qwen38: %llu drafts from the draft list's lm_head rows on the card\n", ch->dv_drafts);
    if (ch->hd_rows) fprintf(stderr, "[chain] qwen38: %llu MTP head rows with their routed experts on the pool's server\n", ch->hd_rows);
    if (ch->hd_card_rows) fprintf(stderr, "[chain] qwen38: %llu MTP head rows with their routed experts on the card (Q2_0)\n", ch->hd_card_rows);
    if (ch->srv && ch->sv_jobs)
        fprintf(stderr, "[chain] qwen38: the pool's server (%d threads): %llu layer steps, %llu pooled slots (%llu experts' groups) "
                        "in %.1f ms (%.1f us a slot)\n", ch->srv->started, ch->sv_jobs, ch->sv_slots, (unsigned long long)Q38CC_LD(&ch->srv->groups),
                ch->sv_ns / 1e6, ch->sv_slots ? ch->sv_ns / 1e3 / ch->sv_slots : 0.0);
    if (ch->pc_pairs)
        fprintf(stderr, "[chain] qwen38: %llu of a decode step's slots from the host pool on the CPU in %.1f ms (%.1f us a slot; "
                        "COLI_CUDA_CHAIN_POOL_READ=1: the device reads them over the bus)\n",
                ch->pc_pairs, ch->pc_ms, ch->pc_ms * 1e3 / (double)ch->pc_pairs);
    if (ch->pg_pairs)
        fprintf(stderr, "[chain] qwen38: %llu prompts' pooled experts on the CPU, %llu staged; %llu slots from the host pool on the CPU, "
                        "%llu experts, in %.1f ms (%.1f us a slot; COLI_CUDA_CHAIN_POOL_CPU_ROWS=0: the prompt's chunks stage them over the bus)\n",
                ch->pc_took, ch->pc_staged, ch->pg_pairs, ch->pg_groups, ch->pg_ms, ch->pg_ms * 1e3 / (double)ch->pg_pairs);
    if (ch->lane_chunks)
        fprintf(stderr, "[chain] qwen38: %llu prompt chunks in two lanes, one lane's pooled experts on the CPU while the card runs the other "
                        "(COLI_CUDA_CHAIN_LANES=0: one lane); %llu pooled experts copied over for the card in %llu layer steps "
                        "(COLI_CUDA_CHAIN_LANE_STAGE=0: none)\n", ch->lane_chunks, ch->ls_experts, ch->ls_layers);
    if (ch->dm_staged)
        fprintf(stderr, "[chain] qwen38: %llu prompt layer steps staged the host pool's experts (%.2f GB copied over the bus)\n",
                ch->dm_staged, ch->dm_stage_bytes / 1073741824.0);
    if (ch->lm_forwards)
        fprintf(stderr, "[chain] qwen38: %llu prompts layer by layer, each layer's host pool experts staged once "
                        "(COLI_CUDA_CHAIN_LAYERWISE=0: chunk by chunk)\n", ch->lm_forwards);
    if (ch->dm_early)
        fprintf(stderr, "[chain] qwen38: %llu prompt layer steps' copies started beside the layer before's experts (%.2f GB of them; "
                        "COLI_CUDA_CHAIN_STAGE_RING=0: after them)\n", ch->dm_early, ch->dm_early_bytes / 1073741824.0);
    if (ch->mg_groups)
        fprintf(stderr, "[chain] qwen38: %llu prompt layer steps' routed experts in one call for %llu chunks (%llu of them chunk by chunk "
                        "after all; COLI_CUDA_CHAIN_MOE_GROUP=0: each chunk's)\n", ch->mg_groups, ch->mg_chunks, ch->mg_declined);
    if (ch->mtp_rows)
        fprintf(stderr, "[chain] qwen38: %llu MTP head rows and %llu drafts' logits on the card\n", ch->mtp_rows, ch->drafts);
    if (ch->own)
        fprintf(stderr, "[chain] qwen38: %llu experts promoted to the card (%llu residents back to the pool), %d on their way; "
                        "%llu frames braked, %.1f %% of the routes on the card\n",
                ch->promos, ch->promo_victims, ch->npend, ch->pr_braked,
                ch->routes ? 100.0 * (double)(ch->routes - ch->pool_reads) / (double)ch->routes : 0.0);
    if (ch->greedy_rows) fprintf(stderr, "[chain] qwen38: %llu greedy rows picked on the card, no logits down (COLI_CUDA_CHAIN_GREEDY_CARD=0: the logits)\n", ch->greedy_rows);
}
