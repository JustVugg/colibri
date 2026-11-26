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
 * layers. KV_SLOTS above 1: every slot's state in its own region of the state buffers (v41u_wo,
 * v41u_co, v41u_io, v41u_ro), and only a decode step's one row on the chain, a slot at a time
 * (forward_rows, v41u_mux_rows; COLI_CUDA_V41_MIN=1), the prompts on the CPU; V41_CHAIN_ROWS=1: the
 * active conversations' decode rows together, a frame a layer (v41u_forward_rows). COLI_CUDA_CHAIN=0:
 * never. COLI_CUDA_V41_ROWS: a chunk's rows (256). */
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

/* The buffers a device holds that are not a layer's own: the parameters, a step's scratch, the routed experts' rows
 * and tables. One set a device (V41_CHAIN_SPLIT: two devices); V41U's members of these names are the current
 * device's (v41u_use trades the two sets). A layer's own buffers and matrices are on its device. */
#define V41U_BANK \
    CcBuf *prm; \
    CcBuf *xs, *xn, *mix, *hpa, *hpf, *col, *nrm, *qa, *qr, *q, *kv, *heads, *grp, *br, *craw, *cscr; \
    CcBuf *iq, *ihw, *isc, *cs, *erows, *ekv, *gs, *us, *hs, *ds, *h2d, *xd, *hpd, *pull, *routed; \
    CcBuf *lw_xs, *lw_hp, *lw_ds, *lw_list, *lw_mask; \
    CcBuf *cm_bias, *cm_tab;      /* every layer's gate bias (E floats) and table (E CcExpert), at i * E */ \
    CcBuf *cm_lg, *cm_ids, *cm_w, *cm_hid, *cm_y, *cm_hy, *cm_yd; \
    CcBuf *cm_flag;
typedef struct { V41U_BANK } V41UBank;
typedef struct {
    int ok, W, Wd, K, rmax, rows;
    size_t wcap;
    union { V41UBank bank; struct { V41U_BANK }; };   /* the current device's */
    V41UBank bank2;                                   /* the other device's (V41_CHAIN_SPLIT) */
    /* V41_CHAIN_SPLIT: layers split.. run on the second device (devs[1]), the ones before on the primary (devs[0]);
     * nd 2 then, dcur the device whose bank and chain context are current. ikm[k], kvm: a second-device kv_source
     * layer's index keys mirrored on the primary for its index sources that read them in a decode step (the keys
     * published last), and the rows each slot's mirror holds ([SL * n_layers]) */
    int split, nd, dcur, devs[2];
    CcBuf **ikm; int *kvm;
    unsigned long long sp_steps; double sp_ms;        /* the hand-offs between the devices and their time */
    size_t *o_an, *o_fn, *o_qn, *o_kn, *o_sink, *o_hca, *o_hcf, *o_cn, *o_ikn, *o_eq, *o_ek;
    ColiCudaTensor **fna, **fnf;
    CcBuf **win, **ckv, **ikey, **ring;
    /* KV_SLOTS > 1 (serve_mux): every slot's state in its own region of each layer's buffer at a fixed
     * stride (v41u_wo, v41u_co, v41u_io, v41u_ro), SL of them; one slot: slot 0's region, at 0 */
    int SL, hd, ihd;
    int *kv_valid;   /* [SL * n_layers]: slot s, layer i at s * n_layers + i */
    int *ccap, *win_valid;   /* win_valid: [SL] */
    float *lw_in, *lw_out;
    size_t lw_cap;
    unsigned long long forwards, one_row;
    double host_ms, attn_ms, ffn_ms, fwd_ms;   /* the exit report's split (v41u_report) */
    /* V41_CHAIN_MOE=1 (the SYCL backend only, v41u_cmoe_*): a decode step's routed experts in
     * the chain -- the router, the resident experts (cc_route kind 2, cc_moe fmt 7), the host's
     * misses only on the CPU */
    int cm_state;                 /* 0 not tried yet, 1 on, -1 off for good */
    int cm_E, cm_K, cm_act, cm_check;   /* cm_check: V41_CHAIN_MOE_CHECK's (layer, step) pairs still to check */
    float cm_lim;
    unsigned cm_epoch;            /* the tier's epoch the device tables were last built at */
    ColiCudaTensor **cm_gate;     /* every layer's gate [E][dim] (v41u_tb) */
    float *cm_mbuf;               /* the batched misses' gate/up and down rows */
    int cm_batch;                 /* V41_CHAIN_MOE_BATCH (default 1): a layer's misses together; -1 unread */               /* DOWN: the frame's word once its routes are down (cc_signal), the host's misses start then */
    unsigned cm_seq;
    CcExpert *cm_tabh;            /* the tables as the device holds them: g NULL, the host's miss */
    uint32_t *cm_ent;             /* one layer's tier entries (E * COLI_VK_XB_ENTRY words) */
    int *cm_chg;
    unsigned char *cm_dirty;
    float *cm_contrib, *cm_ref, *cm_sc;
    unsigned long long cm_steps, cm_layers, cm_dev_slots, cm_host_slots, cm_reads, cm_checked, cm_route_diff;
    unsigned long long cm_d2_slots;   /* COLI_VK_DEV2: the misses the second card took (both forms) */
    unsigned long long cm_tails;      /* V41_CHAIN_TAIL: layer steps whose last frame was one launch (cc_moe_tail) */
    double cm_dev_ms, cm_host_ms, cm_p2_ms, cm_err_max, cm_err_sum, cm_wdiff;
    double cm_tab_ms, cm_cpu_ms, cm_cnt_ms, cm_wait_ms;   /* cm_wait_ms: the host waiting for the frame's routes */   /* within the above: the tables' refresh, the misses' CPU experts, vkt_dev_count */
    /* V41_CHAIN_ROWS=1 (v41u_forward_rows): a decode step's rows of every active conversation in one
     * frame a layer, the attention a slot at a time inside it; the exit report's books */
    unsigned long long rw_steps, rw_rows, rw_fallbacks, rw_layers, rw_dev_slots, rw_host_slots;
    unsigned long long rw_checked_steps, rw_checked_rows;
    double rw_dev_ms, rw_host_ms, rw_p2_ms, rw_book_ms, rw_head_ms, rw_fwd_ms, rw_err_max, rw_err_sum;
    double rw_wait_ms;   /* within rw_dev_ms: the host waiting for the frame's routes */
    float *rw_mbuf, *rw_contrib;   /* the rows' host misses: [pair][gate I | up I] then [pair][D]; [row * K + k][D] */
} V41U;
static V41U *g_v41u;
static int g_v41u_off;   /* declined for good (set up refused, or COLI_CUDA_CHAIN=0) */

/* ---- V41_CHAIN_SPLIT: the layers on two devices ------------------------------------------------
 * V41_CHAIN_SPLIT=<layer>|auto with a second device open (COLI_VK_DEV2): the chain runs the layers before
 * <layer> on the primary device and the ones from it on on the second. Each device holds its layers' matrices,
 * their state (window rings, compressed rows, index keys) and, in the tier, its layers' routed experts only
 * (VktConfig.home_split: an expert has one home, its layer's device), so two cards hold twice the residents a
 * layer and each card half the trunk. A decode step's streams and mixes cross once, through the host (a few
 * KB a row, no peer copy): down from the primary after its last layer, up to the second before its first.
 * Decode steps only (one row, or V41_CHAIN_ROWS's rows): a prompt's layers stay the CPU's, as with KV_SLOTS > 1.
 *
 * Where a split may be (v41u_split_ok): at a kv_source layer of compress ratio 1 that is an index source. Then
 * every layer from it on reads compressed rows and, in a decode step, index keys of layers from it on only
 * (the layer publishes at every position), its selection is written before its attention reads the list, and
 * the candidate mask (written at or after it) is the second device's. The layers before it may read the
 * second device's index keys -- the ones published last, in a step where their own source does not publish:
 * rows of earlier positions only (their ratios are above 1), which the host holds; those go up to a mirror on
 * the primary (ikm) as far as it does not hold them. The same kernels on either device: the same bits.
 * With one device, or unset: nothing here runs and the chain is as before. */
static int g_v41u_split;   /* the first layer of the second device, decided at start (v41u_split_decide); 0 none */
static int v41u_cand_on(const Cfg *c);
static int v41u_split_ok(const Cfg *c, int s) {
    if (s <= 0 || s >= c->n_layers || !c->kv_source[s] || c->compress_ratio[s] != 1 || !c->index_source[s]) return 0;
    for (int i = 0; i < s; i++) if (c->index_source[i] && c->compress_ratio[i] == 1) return 0;   /* would read this step's keys */
    if (v41u_cand_on(c) && c->candidate_source < s)   /* a mask written on the primary, read on the second */
        for (int i = s; i < c->n_layers; i++) if (c->index_source[i] && c->compress_ratio[i] > 0) return 0;
    return 1;
}
static int v41u_dev_of(const V41U *ch, int i) { return ch->split > 0 && i >= ch->split; }
static V41UBank *v41u_bank(V41U *ch, int d) { return d == ch->dcur ? &ch->bank : &ch->bank2; }
/* device d's chain context and bank current (0 the primary; 1 the second, V41_CHAIN_SPLIT only) */
static void v41u_use(V41U *ch, int d) {
    if (ch->nd < 2 || d == ch->dcur) return;
    const V41UBank t = ch->bank; ch->bank = ch->bank2; ch->bank2 = t;
    ch->dcur = d;
    cc_device(ch->devs[d]);
}
/* a device of the chain's is lost (the current one stays current) */
static int v41u_any_lost(V41U *ch) {
    if (ch->nd < 2) return cc_lost();
    int lost = 0;
    for (int d = 0; d < ch->nd; d++) { cc_device(ch->devs[d]); lost |= cc_lost(); }
    cc_device(ch->devs[ch->dcur]);
    return lost;
}

/* the slot whose state the chain reads and writes: the one the Model holds (v41_mux_bind), -1 none */
static int v41u_slot(const V41U *ch) { return ch->SL > 1 ? g_v41_mux_cur : 0; }
/* slot s's region in a layer's window ring (Wd rows), compressed rows and index keys (ccap[i] each),
 * the compressor's group (2r rows: the kv, then the scores), in floats; slot 0's is at 0 */
static size_t v41u_wo(const V41U *ch, int s) { return (size_t)s * ch->Wd * ch->hd; }
static size_t v41u_co(const V41U *ch, int i, int s) { return (size_t)s * ch->ccap[i] * ch->hd; }
static size_t v41u_io(const V41U *ch, int i, int s) { return (size_t)s * ch->ccap[i] * ch->ihd; }
static size_t v41u_ro(const V41U *ch, const Model *m, int i, int s) { return (size_t)s * 2 * m->c.compress_ratio[i] * ch->hd; }
static int *v41u_kvv(const V41U *ch, const Model *m, int s) { return ch->kv_valid + (size_t)s * m->c.n_layers; }

static int v41u_res(CcBuf **b, size_t floats, int kind) { return cc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }
/* a W8: a GGUF's blocks, or the native checkpoint's e4m3 with its 32x32 ue8m0 tiles (fmt 9, mv8_rows_cpu's bits) */
static ColiCudaTensor *v41u_t8(const W8 *w) {
    if (w->gx) return w->q ? cud_tensor(w->q, 100 + w->gt, w->O, w->I) : NULL;
    return w->q && w->s ? cud_tensor_e4t(w->q, w->s, w->O, w->I) : NULL;
}
/* a WB: a GGUF's blocks, or the bf16 rows the loader keeps of a GGUF's BF16 tensor (unsloth's
 * compressor and indexer matrices), those as floats */
static ColiCudaTensor *v41u_tb(const WB *w) {
    if (w->gx) return w->g ? cud_tensor(w->g, 100 + w->gt, w->O, w->I) : NULL;
    return w->w ? cud_tensor(w->w, 100 + GQ_BF16, w->O, w->I) : NULL;
}
static ColiCudaTensor *v41u_woa(const W8 *w, int g, int ol) {
    if (!w->gx) {   /* the native FP8: group g is rows g*ol.. of the matrix and their rows of tiles (ol a multiple of 32) */
        if (!w->q || !w->s || ol % FP8_TILE || w->I % FP8_TILE) return NULL;
        const size_t tiles_i = (size_t)w->I / FP8_TILE;
        return cud_tensor_e4t(w->q + (size_t)g * ol * w->I, w->s + (size_t)g * (ol / FP8_TILE) * tiles_i, ol, w->I);
    }
    const int64_t rb = gq_row_bytes(w->gt, w->I);   /* an interleaved type's: 16 of them step a group */
    return w->gx && w->q && rb > 0 ? cud_tensor(w->q + (size_t)g * ol * (size_t)rb, 100 + w->gt, ol, w->I) : NULL;
}

/* the window ring and position map, the compressed rows and keys the device holds from pos on
 * are stale: a CPU step from pos, a rollback, a new sequence */
static void v41u_lower(Model *m, int pos) {
    V41U *ch = g_v41u;
    if (!ch || !ch->ok) return;
    if (pos < 0) pos = 0;
    /* the bound slot's; KV_SLOTS > 1 with none bound (or one out of range): every slot's */
    const int cur = v41u_slot(ch), all = cur < 0 || cur >= ch->SL;
    for (int s = all ? 0 : cur; s < (all ? ch->SL : cur + 1); s++) {
        int *kvv = v41u_kvv(ch, m, s);
        if (ch->win_valid[s] > pos) ch->win_valid[s] = pos;
        for (int i = 0; i < m->c.n_layers; i++) {
            int r = m->c.compress_ratio[i];
            if (m->c.kv_source[i] && r > 0 && kvv[i] > pos / r) kvv[i] = pos / r;
            if (ch->kvm && m->c.kv_source[i] && r > 0 && ch->kvm[(size_t)s * m->c.n_layers + i] > pos / r)
                ch->kvm[(size_t)s * m->c.n_layers + i] = pos / r;
        }
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
    /* every slot's state (KV_SLOTS > 1: SL of them, each compressed cache sized once for max_positions
     * as v41u_cache does, so a slot's growth never drops another's) */
    const size_t SL = g_v41_mux_slots > 1 ? (size_t)g_v41_mux_slots : 1;
    f += SL * L * (W + R) * hd + L * 2 * nm * HD;                /* the window rings, the mixers' matrices */
    for (int i = 0; i < c->n_layers; i++) {
        const int r = c->compress_ratio[i];
        if (!c->kv_source[i] || r < 1) continue;
        size_t cap = 64;
        if (SL > 1) {
            const size_t host = (size_t)(c->max_positions / r);
            while (cap < host) cap *= 2;
            if (cap > host) cap = host;
        } else
            while (cap < E / (size_t)r) cap *= 2;
        f += SL * (cap * (hd + ID) + (r > 1 ? 2 * (size_t)r * hd : 0));
    }
    return f * sizeof(float) + ((size_t)64 << 20);               /* and the GEMMs' passing scratch */
}
/* COLI_CUDA_V41_MIN=<rows>: a forward of that many rows on takes the chain (1: every step, a decode's
 * one row too -- PLAN-scale S3); 0 (unset): a prompt of more than one chunk */
static int v41u_min_rows(void) {
    static int min_rows = -1;
    if (min_rows < 0) { const char *e = getenv("COLI_CUDA_V41_MIN"); min_rows = e && atoi(e) > 0 ? atoi(e) : 0; }
    return min_rows;
}
/* KV_SLOTS > 1 (v41_seq_swap trades the host's state): the chain holds every slot's state (M1) and
 * takes their decode steps one row and one slot at a time (forward_rows, v41u_mux_rows), so only
 * when the decode steps go to the chain (COLI_CUDA_V41_MIN=1); a prompt stays the CPU's
 * (v41u_wanted) */
static int v41u_slots_ok(void) { return g_v41_mux_slots <= 1 || v41u_min_rows() == 1; }
/* the native checkpoint the chain takes: its W8 are e4m3 with tiles (not a GGUF's blocks) */
static int v41u_native(const Model *m) { return !m->gguf && m->c.n_layers > 0 && !m->L[0].wq_a.gx && m->L[0].wq_a.s; }
/* The layer frame's idle card (a frame profile showed one gap a layer, the card waiting for the host's misses, and the
 * frame after it a command at a time). Three switches, each the same arithmetic in another order or launch:
 *   V41_CHAIN_SHARED_LATE (default 1)  the router and its word BEFORE the shared expert: the host's misses start a
 *       shared expert earlier (3 products and the SwiGLU, 0.09 ms a layer on one row, 0.25 on four), the card runs it
 *       and the residents meanwhile. 0: the shared expert first, as before.
 *   V41_CHAIN_TAIL (default 1)  the layer's last frame -- the residents' sum, the FFN add, the streams' write back --
 *       in one launch (cc_moe_tail) where the backend has it; each command of that frame waited for the host's submit
 *       with the card idle. 0: cc_moe parts 2, ew ADD, mhc POST.
 *   V41_CHAIN_DIRECT (default 1)  a decode step's one row keeps its streams, mixes and shared branch in the layer's
 *       buffers from frame to frame: the nine copies a layer to and from the segment's stash (lw_xs, lw_hp, lw_ds:
 *       what a prompt's chunks need, one row does not) are not made. 0: as before. */
static int v41u_env01(const char *name, int def) { const char *e = getenv(name); return e && *e ? *e != '0' : def; }
static int v41u_shared_late(void) { static int v = -1; if (v < 0) v = v41u_env01("V41_CHAIN_SHARED_LATE", 1); return v; }
static int v41u_tail_on(void) { static int v = -1; if (v < 0) v = v41u_env01("V41_CHAIN_TAIL", 1); return v; }
static int v41u_direct_on(void) { static int v = -1; if (v < 0) v = v41u_env01("V41_CHAIN_DIRECT", 1); return v; }
/*   V41_CHAIN_MARK (default 0, opt-in)  "the routes are down" as a mark the host polls (cc_mark / cc_reached)
 *       in place of cc_signal's host task, which held the frame's later commands until the runtime had run it. */
static int v41u_mark_on(void) { static int v = -1; if (v < 0) v = v41u_env01("V41_CHAIN_MARK", 0) && cc_mark_ready(); return v; }
/* the open frame's routes are down (their rows and h2d in DOWN memory): 1; 0 not yet; -1 a lost device */
static int v41u_routes_down(const V41U *ch) {
    if (v41u_mark_on()) return cc_reached();
    return *(volatile unsigned *)cc_ptr(ch->cm_flag) >= ch->cm_seq;
}
/* V41_CHAIN_SPLIT at start, before the dense budgets and the tier: the first layer of the second device, 0 for one
 * device (not asked; or said why not). auto: the layer nearest the middle a split may be at (v41u_split_ok). */
static int v41u_split_decide(const Model *m, int two_devices) {
    const char *e = getenv("V41_CHAIN_SPLIT");
    if (!e || !*e || !strcmp(e, "0")) return 0;
    const Cfg *c = &m->c;
    const char *why = NULL;
    int s = 0;
    if (!two_devices) why = "no second device is open (COLI_VK_DEV2 and a second entry in COLI_GPUS)";
    else if (!v41u_native(m)) why = "the model is not the native FP8 checkpoint";
    else if (v41u_min_rows() != 1) why = "the decode steps are not on the chain (COLI_CUDA_V41_MIN=1)";
    else if (!strcmp(e, "auto")) {
        for (int i = 1; i < c->n_layers; i++)
            if (v41u_split_ok(c, i) && (!s || abs(2 * i - c->n_layers) < abs(2 * s - c->n_layers))) s = i;
        if (!s) why = "no layer of this model can start the second device (a kv_source of ratio 1 that is an index source)";
    } else {
        s = atoi(e);
        if (!v41u_split_ok(c, s)) { s = 0; why = "that layer cannot start the second device (a kv_source of ratio 1 that is an index source)"; }
    }
    if (why) fprintf(stderr, "[CUDA] deepseek_v41 chain: V41_CHAIN_SPLIT=%s ignored: %s; one device as before\n", e, why);
    return s;
}
static void v41u_report(void) {   /* where a chain forward's time went (PLAN-scale S3) */
    { const char *fp = getenv("COLI_SYCL_FRAME_PROF"); if (fp && *fp && strcmp(fp, "0")) cc_shutdown(); }   /* its per-command books, printed once */
    const V41U *ch = g_v41u;
    if (!ch) return;
    if (ch->ok && ch->forwards) {
        const double f = (double)ch->forwards;
        fprintf(stderr, "[CUDA] deepseek_v41 chain: %llu forwards (%llu of one row), %.2f ms each: attention+shared frames %.2f, "
                        "routed experts on the host %.2f, FFN frames %.2f, the rest %.2f\n", ch->forwards, ch->one_row, ch->fwd_ms / f,
                ch->attn_ms / f, ch->host_ms / f, ch->ffn_ms / f, (ch->fwd_ms - ch->attn_ms - ch->host_ms - ch->ffn_ms) / f);
    }
    if (ch->cm_layers) {   /* V41_CHAIN_MOE's decode steps */
        const double s = ch->cm_steps ? (double)ch->cm_steps : 1.0, ly = (double)ch->cm_layers;
        fprintf(stderr, "[CUDA] deepseek_v41 chain MoE: %llu decode steps (%llu layer steps), of a layer's %d slots %.2f resident, "
                        "%.2f host misses; per step: attention+shared+router+resident experts frames %.2f ms, host misses %.2f ms, "
                        "parts-2+FFN frames %.2f ms; tables read %llu times\n", ch->cm_steps, ch->cm_layers, ch->cm_K,
                ch->cm_dev_slots / ly, ch->cm_host_slots / ly, ch->cm_dev_ms / s, ch->cm_host_ms / s, ch->cm_p2_ms / s, ch->cm_reads);
        fprintf(stderr, "[CUDA] deepseek_v41 chain MoE: within those, per step: the tables' refresh %.2f ms, the misses' CPU experts "
                        "%.2f ms, vkt_dev_count %.2f ms, the host waiting for the routes %.2f ms\n", ch->cm_tab_ms / s, ch->cm_cpu_ms / s,
                ch->cm_cnt_ms / s, ch->cm_wait_ms / s);
    }
    if (ch->cm_d2_slots)
        fprintf(stderr, "[CUDA] deepseek_v41 chain MoE: %llu missed slots on the second card (COLI_VK_DEV2), the CPU beside it\n",
                ch->cm_d2_slots);
    if (ch->cm_layers || ch->rw_steps)   /* the layer-frame switches as this run had them (deepseek_v41_cuchain.h, v41u_shared_late) */
        fprintf(stderr, "[CUDA] deepseek_v41 chain MoE: the routes %s the shared expert (V41_CHAIN_SHARED_LATE), the last frame in one "
                        "launch for %llu layer steps (V41_CHAIN_TAIL %d), one row's streams %s (V41_CHAIN_DIRECT)\n",
                v41u_shared_late() ? "before" : "after", ch->cm_tails, v41u_tail_on(), v41u_direct_on() ? "in place" : "through the stash");
    if (ch->cm_layers || ch->rw_steps)
        fprintf(stderr, "[CUDA] deepseek_v41 chain MoE: the routes' word %s (V41_CHAIN_MARK); batched steps: the host waited %.2f ms a "
                        "step for the routes\n", v41u_mark_on() ? "a mark the host polls" : "a host task", ch->rw_steps ? ch->rw_wait_ms / (double)ch->rw_steps : 0.0);
    if (ch->split && ch->sp_steps)
        fprintf(stderr, "[CUDA] deepseek_v41 chain (V41_CHAIN_SPLIT): %llu decode steps crossed from device %d to device %d before layer %d, "
                        "%.3f ms each (the primary's state and the rows down, the rows up)\n", ch->sp_steps, ch->devs[0], ch->devs[1],
                ch->split, ch->sp_ms / (double)ch->sp_steps);
    if (ch->cm_checked)
        fprintf(stderr, "[CUDA] deepseek_v41 chain MoE check: %llu pairs, max rel err %.3g, mean %.3g (the CPU's routing differed "
                        "in %llu, max route weight diff %.3g)\n", ch->cm_checked, ch->cm_err_max, ch->cm_err_sum / (double)ch->cm_checked,
                ch->cm_route_diff, ch->cm_wdiff);
    if (ch->rw_steps || ch->rw_fallbacks) {   /* V41_CHAIN_ROWS's batched decode steps */
        const double s = ch->rw_steps ? (double)ch->rw_steps : 1.0, ly = ch->rw_layers ? (double)ch->rw_layers : 1.0;
        fprintf(stderr, "[CUDA] deepseek_v41 chain rows: %llu batched decode steps (%llu rows, %.2f a step; %llu fell back to a row at "
                        "a time), %.2f ms each: device frames %.2f (attention+shared+router+resident experts %.2f, parts-2+FFN %.2f), "
                        "host misses %.2f, host bookkeeping %.2f, heads %.2f; of a layer's slots %.2f resident, %.2f host misses\n",
                ch->rw_steps, ch->rw_rows, ch->rw_rows / s, ch->rw_fallbacks, ch->rw_fwd_ms / s, (ch->rw_dev_ms + ch->rw_p2_ms) / s,
                ch->rw_dev_ms / s, ch->rw_p2_ms / s, ch->rw_host_ms / s, ch->rw_book_ms / s, ch->rw_head_ms / s,
                ch->rw_dev_slots / ly, ch->rw_host_slots / ly);
    }
    if (ch->rw_checked_rows)
        fprintf(stderr, "[CUDA] deepseek_v41 chain rows check: %llu steps (%llu rows) against a row at a time, logits max rel err %.3g, "
                        "mean %.3g\n", ch->rw_checked_steps, ch->rw_checked_rows, ch->rw_err_max, ch->rw_err_sum / (double)ch->rw_checked_rows);
}
static int v41u_wanted(Model *m, int n) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN"); on = !(e && *e == '0'); }
    /* a GGUF, or the native FP8 checkpoint (its W8 as fmt 9: v41u_t8); KV_SLOTS > 1: a decode step's
     * one row only (the caller refuses a speculative batch), a prompt the CPU's */
    const int slots_ok = g_v41_mux_slots <= 1 || (n == 1 && v41u_slots_ok());
    return on && !g_v41u_off && slots_ok && xt_cuda() && g_cud.on && (m->gguf || v41u_native(m)) && cc_available() && cc_dsv4_ready();
}
/* forward_rows (KV_SLOTS > 1): each row a one-row forward_full of its own bound slot through the
 * chain, as a KV_SLOTS=1 decode step (nothing drafts with KV_SLOTS > 1; DSpark stays unloaded) */
static int v41u_mux_rows(Model *m) {
    return g_v41_mux_slots > 1 && v41u_slots_ok() && !m->spec.active && v41u_wanted(m, 1) && (!g_v41u || g_v41u->ok);
}
/* the struct, its device's context, the parameters, every layer's matrices and state: 0 =
 * declined (the CPU runs the layers from then on) */
static int v41u_setup(Model *m) {
    if (g_v41u) return g_v41u->ok;
    Cfg *c = &m->c;
    const int L = c->n_layers, D = c->dim, H = c->hc_mult, nm = (2 + H) * H, hd = c->head_dim;
    V41U *ch = (V41U *)calloc(1, sizeof *ch);
    if (!ch) return 0;
    const int SL = g_v41_mux_slots > 1 ? g_v41_mux_slots : 1;
    ch->SL = SL; ch->hd = hd; ch->ihd = c->index_head_dim;
    ch->cm_batch = -1;
    g_v41u = ch;
    const char *why = NULL;
    /* V41_CHAIN_SPLIT, in force only with the tier's home by layer at the same layer and a second dense budget */
    ch->nd = 1; ch->dcur = 0;
    cud_place(0);
    ch->devs[0] = ch->devs[1] = g_cud.dev;
    if (g_v41u_split > 0 && g_cud.two && XT(home)() == g_v41u_split && v41u_split_ok(c, g_v41u_split)) {
        ch->split = g_v41u_split; ch->nd = 2; ch->devs[1] = g_cud.alt_dev;
        if (!(ch->ikm = (CcBuf **)calloc((size_t)L, sizeof *ch->ikm)) || !(ch->kvm = (int *)calloc((size_t)L * SL, sizeof(int)))) why = "memory";
        else if (!cc_init(ch->devs[1]) || !cc_ready()) why = "the chain's context on the second device did not come up";
    }
    if (!why && (!cc_init(g_cud.dev) || !cc_ready())) why = "the chain's context did not come up";
    size_t **offs[] = {&ch->o_an, &ch->o_fn, &ch->o_qn, &ch->o_kn, &ch->o_sink, &ch->o_hca, &ch->o_hcf, &ch->o_cn, &ch->o_ikn, &ch->o_eq, &ch->o_ek};
    for (size_t k = 0; !why && k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = (size_t *)calloc((size_t)L, sizeof(size_t)))) why = "memory";
    if (!why && (!(ch->fna = (ColiCudaTensor **)calloc((size_t)L, sizeof *ch->fna)) || !(ch->fnf = (ColiCudaTensor **)calloc((size_t)L, sizeof *ch->fnf)) ||
                 !(ch->win = (CcBuf **)calloc((size_t)L, sizeof *ch->win)) || !(ch->ckv = (CcBuf **)calloc((size_t)L, sizeof *ch->ckv)) ||
                 !(ch->ikey = (CcBuf **)calloc((size_t)L, sizeof *ch->ikey)) || !(ch->ring = (CcBuf **)calloc((size_t)L, sizeof *ch->ring)) ||
                 !(ch->kv_valid = (int *)calloc((size_t)L * SL, sizeof(int))) || !(ch->ccap = (int *)calloc((size_t)L, sizeof(int))) ||
                 !(ch->win_valid = (int *)calloc((size_t)SL, sizeof(int)))))
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
    for (int d = 0; !why && d < ch->nd; d++) {   /* the whole arena on each device: a layer's offsets are the same on both */
        v41u_use(ch, d);
        if (!(v41u_res(&ch->prm, n, CC_DEV) && cc_begin() && cc_write(ch->prm, 0, a, n * sizeof(float)) && cc_submit(1)))
            why = "the parameters did not reach the device";
    }
    free(a);
    /* every layer's matrices (cu_dense.h's copies) and state, on its device */
    for (int i = 0; !why && i < L; i++) {
        Layer *l = &m->L[i];
        v41u_use(ch, v41u_dev_of(ch, i));
        cud_place(v41u_dev_of(ch, i));
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
        if (!(ch->win[i] = cc_buf((size_t)SL * ch->Wd * hd * sizeof(float), CC_DEV)) ||
            (c->kv_source[i] && r > 1 && !(ch->ring[i] = cc_buf((size_t)SL * 2 * r * hd * sizeof(float), CC_DEV))))
            why = "device memory for the layers' state";
    }
    v41u_use(ch, 0);
    cud_place(0);
    if (why) {
        fprintf(stderr, "[CUDA] deepseek_v41 chain: %s; the CPU runs a prompt's layers\n", why);
        g_v41u_off = 1;
        return 0;
    }
    ch->ok = 1;
    atexit(v41u_report);
    if (ch->split)
        fprintf(stderr, "[CUDA] deepseek_v41 chain (V41_CHAIN_SPLIT): layers 0..%d on device %d, layers %d..%d on device %d, each with "
                        "its matrices, state and resident experts; a decode step's streams cross once through the host; a prompt's "
                        "layers on the CPU\n", ch->split - 1, ch->devs[0], ch->split, L - 1, ch->devs[1]);
    if (SL > 1) fprintf(stderr, "[CUDA] deepseek_v41 chain: %d conversations' state on the card (KV_SLOTS), their decode steps "
                                "one row at a time; their prompts on the CPU\n", SL);
    fprintf(stderr, "[CUDA] deepseek_v41 chain: a prompt's %d layers on the card, layer by layer (%.1f MiB of parameters; "
                    "COLI_CUDA_CHAIN=0: the CPU)\n", L, n * 4 / 1048576.0);
    return 1;
}
/* layer i's compressed rows and index keys for rows groups. One slot: grown as needed (what it held is
 * dropped). KV_SLOTS > 1: every slot's region sized once for max_positions (the same rule at the
 * host's groups), so no slot's growth drops another's; refused past the ops' int offsets */
static int v41u_cache(V41U *ch, Model *m, int i, int rows) {
    Cfg *c = &m->c;
    if (ch->ccap[i] >= rows) return 1;
    int r = c->compress_ratio[i], host = c->max_positions / (r > 0 ? r : 1), cap = 64;
    if (ch->SL > 1 && ch->ccap[i] > 0) return 0;   /* already max_positions' */
    const int target = ch->SL > 1 ? (host > rows ? host : rows) : rows;
    while (cap < target) cap *= 2;
    if (cap > host) cap = host;
    if (cap < rows) return 0;
    const size_t wide = (size_t)(c->head_dim > c->index_head_dim ? c->head_dim : c->index_head_dim);
    if (ch->SL > 1 && (size_t)ch->SL * (size_t)cap * wide > (size_t)0x7fffffff) {
        static int said;
        if (!said++) fprintf(stderr, "[CUDA] deepseek_v41 chain: %d slots of %d compressed rows pass the ops' offsets; "
                                     "the CPU decodes\n", ch->SL, cap);
        g_v41u_off = 1;   /* for good: forward_rows batches its rows on the CPU again */
        return 0;
    }
    const int was = ch->dcur, home = v41u_dev_of(ch, i);
    v41u_use(ch, home);
    cc_free(ch->ckv[i]); cc_free(ch->ikey[i]);
    ch->ckv[i] = cc_buf((size_t)ch->SL * cap * c->head_dim * sizeof(float), CC_DEV);
    ch->ikey[i] = cc_buf((size_t)ch->SL * cap * c->index_head_dim * sizeof(float), CC_DEV);
    for (int s = 0; s < ch->SL; s++) v41u_kvv(ch, m, s)[i] = 0;
    ch->ccap[i] = 0;
    int ok = ch->ckv[i] && ch->ikey[i];
    if (ok && ch->split && home == 1) {   /* a second-device source's index keys mirrored on the primary, the same regions */
        v41u_use(ch, 0);
        cc_free(ch->ikm[i]);
        ch->ikm[i] = cc_buf((size_t)ch->SL * cap * c->index_head_dim * sizeof(float), CC_DEV);
        for (int s = 0; s < ch->SL; s++) ch->kvm[(size_t)s * c->n_layers + i] = 0;
        ok = ch->ikm[i] != NULL;
    }
    v41u_use(ch, was);
    if (!ok) return 0;
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
    if (og <= 16 && nr <= 4) {   /* the groups in one launch (cc_matmul_grp): heads' row s is [og][pg], grp's [og][ol] */
        ColiCudaTensor *tg[16];
        int all = 1;
        for (int k = 0; k < og; k++) all &= (tg[k] = v41u_woa(w, k, ol)) != NULL;
        const int r = all ? cc_matmul_grp(tg, og, ch->heads, 0, og * pg, pg, ch->grp, 0, og * ol, nr) : -1;
        if (r >= 0) return r;
    }
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
/* the index keys of layer k as layer i's device holds them: k's own buffer on its device, the mirror on the primary
 * for a second-device source (V41_CHAIN_SPLIT); NULL where the other device's would be needed (the op refuses) */
static CcBuf *v41u_keys(const V41U *ch, int i, int k) {
    if (!ch->split || v41u_dev_of(ch, i) == v41u_dev_of(ch, k)) return ch->ikey[k];
    return v41u_dev_of(ch, i) == 0 ? ch->ikm[k] : NULL;
}
static int v41u_attention(V41U *ch, Model *m, int i, const V41UFwd *f) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    const int hd = c->head_dim, nh = c->n_heads, rd = c->rope_dim, QL = c->q_lora, r = c->compress_ratio[i];
    const int nr = f->nr, pb = f->pb, qrow = nh * hd, LR = ch->W + ch->K;
    const int tcs = r > 0 ? f->csB + (ch->rmax - 1) * rd : 0;   /* rope_for(layer): the row of position pb */
    const ColiCudaChainOps *o = cc_ops();
    const int sl = v41u_slot(ch);   /* the bound slot's regions of the state buffers */
    const size_t wo = v41u_wo(ch, sl);
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
    for (int s = 0; s < nr; s++) rg[s] = (CcRegion){wo + (size_t)((pb + s) % ch->Wd) * hd, (size_t)s * hd, (size_t)hd};
    ok = o->copy_regions(ch->win[i], ch->kv, rg, nr);
    free(rg);
    if (ok && r > 0 && c->kv_source[i]) {   /* the compressor, its latents, the index keys */
        const int g0 = pb / r, np = (pb + nr) / r - g0, ihd = c->index_head_dim;
        CcBuf *cb = ch->ckv[i];
        const size_t cso = v41u_co(ch, i, sl), iso = v41u_io(ch, i, sl), co = cso + (size_t)g0 * hd, io = iso + (size_t)g0 * ihd;
        ok = o->matmul(v41u_tb(&l->comp_wkv), ch->nrm, 0, ch->craw, 0, nr);
        if (ok && r == 1) ok = v41u_norm(ch, ch->craw, 0, hd, ch->o_cn[i], cb, co, hd, nr, hd, c->norm_eps);
        else if (ok) {
            CcDsComp cp = {nr, pb, r, hd, hd, 0, 0, hd, 0, hd, -1, (int)cso, hd, (int)v41u_ro(ch, m, i, sl)};
            ok = o->matmul(v41u_tb(&l->comp_wgate), ch->nrm, 0, ch->cscr, 0, nr) &&
                 cc_ds_compress(ch->craw, ch->cscr, ch->ring[i], NULL, cb, &cp) &&
                 (np == 0 || v41u_norm(ch, cb, co, hd, ch->o_cn[i], cb, co, hd, np, hd, c->norm_eps));
        }
        if (ok && np > 0) {
            const int bc = f->csB + (g0 * r - (pb - ch->rmax + 1)) * rd;   /* a latent rotates at its group's first position */
            ok = o->matmul(v41u_tb(&l->idx_wk), cb, co, ch->ikey[i], io, np) &&
                 v41u_norm(ch, ch->ikey[i], io, ihd, ch->o_ikn[i], ch->ikey[i], io, ihd, np, ihd, c->norm_eps) &&
                 v41u_rope(ch, m, ch->ikey[i], np, 1, (int)io + ihd - rd, ihd, 0, bc, r * rd, 0) &&
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
            CcDsScore sp = {same ? nr : 1, pb + s, r, IH, ID, width, s * IH * ID, IH * ID, s * IH, IH, (int)v41u_io(ch, kl[s], sl), ID, mrow,
                            (int)ch->wcap, wscale, s * (int)ch->wcap, (lr + s) * mrow};
            ok = cc_ds_score(ch->iq, ch->ihw, v41u_keys(ch, i, kl[s]), mrow ? ch->lw_mask : NULL, ch->isc, &sp);
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
    CcDsAttn a = {nr, nh, hd, ch->W + (r > 0 ? ch->K : 0), f->lo, LR, ch->Wd, (int)wo, r > 0 ? (int)v41u_co(ch, c->index_owner[i], sl) : 0,
                  0, qrow, 0, qrow, (int)ch->o_sink[i], 0,
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
/* the engram's rows of layer i for positions pb.. (the bound sequence's history), on the host */
static void v41u_engram_host(Model *m, int i, int nr, int pb, float *rows) {
    Engram *e = &m->engram; Layer *l = &m->L[i];
    const int table = l->engram_index, width = e->cols * e->head_dim;
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
}
/* those rows up, eng_wkv, and the gate into the nr rows of streams in xs (in the open frame) */
static int v41u_engram_dev(V41U *ch, Model *m, int i, int nr, const float *rows) {
    Cfg *c = &m->c; Engram *e = &m->engram; Layer *l = &m->L[i];
    const int width = e->cols * e->head_dim, H = c->hc_mult, D = c->dim;
    CcDsEngram p = {nr, H, D, 0, D * (H + 1), 0, H * D, (int)ch->o_eq[i], (int)ch->o_ek[i], c->norm_eps};
    return cc_write(ch->erows, 0, rows, (size_t)nr * width * sizeof(float)) &&
           cc_ops()->matmul(v41u_t8(&l->eng_wkv), ch->erows, 0, ch->ekv, 0, nr) && cc_ds_engram(ch->ekv, ch->prm, ch->xs, &p);
}
static int v41u_engram(V41U *ch, Model *m, int i, int nr, int pb, float *rows) {
    v41u_engram_host(m, i, nr, pb, rows);
    return v41u_engram_dev(ch, m, i, nr, rows);
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
/* the layers the current device runs: [*lo, *hi) (all of them without V41_CHAIN_SPLIT) */
static void v41u_span(const V41U *ch, const Model *m, int *lo, int *hi) {
    *lo = ch->split && ch->dcur == 1 ? ch->split : 0;
    *hi = ch->split && ch->dcur == 0 ? ch->split : m->c.n_layers;
}
/* The bound slot's state up as far as the current device does not hold it, its layers' only (V41_CHAIN_SPLIT);
 * needm (the primary, with a split): the rows of each second-device source's index keys its layers read, to the
 * mirror. */
static int v41u_state_up(V41U *ch, Model *m, int start, const int *need, const int *needm) {
    const Cfg *c = &m->c;
    const int L = c->n_layers, W = ch->W, hd = c->head_dim, sl = v41u_slot(ch);
    const size_t wo = v41u_wo(ch, sl);
    const int *kvv = v41u_kvv(ch, m, sl);
    int ok = 1, lo, hi;
    v41u_span(ch, m, &lo, &hi);
    for (int q = ch->win_valid[sl] > start - W + 1 ? ch->win_valid[sl] : start - W + 1; q < start && ok; q++) {
        if (q < 0) continue;
        for (int i = lo; i < hi && ok; i++) {
            const Layer *l = &m->L[i];
            if (l->window_pos[q % W] != q) continue;
            ok = cc_write(ch->win[i], wo + (size_t)(q % ch->Wd) * hd, l->window + (size_t)(q % W) * hd, (size_t)hd * sizeof(float));
        }
    }
    for (int k = ch->split; ok && ch->split && ch->dcur == 0 && needm && k < L; k++) {   /* the mirrors: rows of earlier positions, the host's */
        if (!c->kv_source[k] || !ch->ikm[k]) continue;
        int *km = ch->kvm + (size_t)sl * L + k;
        const int ihd = c->index_head_dim, t0 = *km, t1 = needm[k];
        if (t1 <= t0) continue;
        ok = cc_write(ch->ikm[k], v41u_io(ch, k, sl) + (size_t)t0 * ihd, m->L[k].ikey + (size_t)t0 * ihd, (size_t)(t1 - t0) * ihd * sizeof(float));
        if (ok) *km = t1;   /* a frame that fails lowers it (v41u_mirror_reset) */
    }
    for (int i = lo; i < hi && ok; i++) {
        if (!c->kv_source[i]) continue;
        const Layer *l = &m->L[i];
        const int r = c->compress_ratio[i], ihd = c->index_head_dim, t0 = kvv[i], t1 = need[i];
        const size_t co = v41u_co(ch, i, sl), io = v41u_io(ch, i, sl);
        if (t1 > t0) ok = cc_write(ch->ckv[i], co + (size_t)t0 * hd, l->ckv + (size_t)t0 * hd, (size_t)(t1 - t0) * hd * sizeof(float)) &&
                          cc_write(ch->ikey[i], io + (size_t)t0 * ihd, l->ikey + (size_t)t0 * ihd, (size_t)(t1 - t0) * ihd * sizeof(float));
        if (ok && r > 1) {
            const size_t ro = v41u_ro(ch, m, i, sl);
            ok = cc_write(ch->ring[i], ro, l->cstate_kv, (size_t)r * hd * sizeof(float)) &&
                 cc_write(ch->ring[i], ro + (size_t)r * hd, l->cstate_score, (size_t)r * hd * sizeof(float));
        }
    }
    return ok;
}
/* a failed frame: what the bound slot's mirrors hold is not known */
static void v41u_mirror_reset(V41U *ch, Model *m) {
    const int sl = v41u_slot(ch);
    if (!ch->kvm || sl < 0 || sl >= ch->SL) return;
    memset(ch->kvm + (size_t)sl * m->c.n_layers, 0, (size_t)m->c.n_layers * sizeof(int));
}
/* the current device's layers' new state into its pull buffer (the plan's offsets are the same on both) */
static int v41u_state_pull(V41U *ch, Model *m, int start, int E, const size_t *pc, size_t pw) {
    const Cfg *c = &m->c;
    const int W = ch->W, hd = c->head_dim, sl = v41u_slot(ch);
    const size_t wo = v41u_wo(ch, sl);
    int ok = 1, lo, hi;
    v41u_span(ch, m, &lo, &hi);
    for (int i = lo; i < hi && ok; i++) {
        const int q0 = E - W > start ? E - W : start, nq = E - q0;
        CcRegion *rg = (CcRegion *)malloc((size_t)nq * sizeof *rg);
        if (!rg) return 0;
        for (int q = q0; q < E; q++) rg[q - q0] = (CcRegion){pc[i] + (size_t)(q - q0) * hd, wo + (size_t)(q % ch->Wd) * hd, (size_t)hd};
        ok = cc_ops()->copy_regions(ch->pull, ch->win[i], rg, nq);
        free(rg);
        if (!ok || !c->kv_source[i]) continue;
        const int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim;
        const size_t at = pc[i] + pw;
        ok = cc_copy(ch->pull, at, ch->ckv[i], v41u_co(ch, i, sl) + (size_t)g0 * hd, (size_t)np * hd) &&
             cc_copy(ch->pull, at + (size_t)np * hd, ch->ikey[i], v41u_io(ch, i, sl) + (size_t)g0 * ihd, (size_t)np * ihd) &&
             (r <= 1 || cc_copy(ch->pull, at + (size_t)np * (hd + ihd), ch->ring[i], v41u_ro(ch, m, i, sl), (size_t)2 * r * hd));
    }
    return ok;
}
/* A forward's n rows from start of the bound sequence: whose index keys each index source reads, per
 * row (keyl[i * n + t]: indexer_run, published_owner), the rows of each kv_source layer the forward
 * reads that it does not produce first (need), the last layer that publishes new keys (*last_pub, -1
 * none), and every kv_source layer's device cache for them (v41u_cache). 0: a cache refused. */
static int v41u_plan_keys(V41U *ch, Model *m, int start, int n, int *keyl, int *need, int *needm, int *last_pub) {
    const Cfg *c = &m->c;
    const int L = c->n_layers, E = start + n;
    static int tidy = -1;
    if (tidy < 0) tidy = getenv("V41_INDEX_OWNER") != NULL;
    for (int i = 0; i < L; i++) need[i] = c->kv_source[i] && c->compress_ratio[i] > 0 ? start / c->compress_ratio[i] : 0;
    if (needm) memset(needm, 0, (size_t)L * sizeof(int));   /* V41_CHAIN_SPLIT: a second-device source's rows the primary's layers read */
    int pub = m->published_index_k ? m->published_index_layer : -1, lp = -1;
    for (int i = 0; i < L; i++) {
        const int r = c->compress_ratio[i];
        if (r <= 0) continue;
        const int owner = c->index_owner[i];
        if (c->kv_source[i] && E / r - start / r > 0) { pub = i; lp = i; }
        if (E / r > need[owner]) need[owner] = E / r;
        for (int t = 0; t < n; t++) {
            int k = owner;
            if (c->index_source[i] && !tidy) {
                if (m->pub_rows > 0) {
                    const int p = published_owner(m, i, start, t);
                    k = p >= 0 ? p : m->pub_before ? m->pub_before_layer : owner;
                } else if (n == 1 && pub >= 0) k = pub;
            }
            keyl[(size_t)i * n + t] = k;
            if (c->index_source[i] && (start + t + 1) / r > need[k]) need[k] = (start + t + 1) / r;
            if (needm && ch->split && i < ch->split && k >= ch->split && c->index_source[i] && (start + t + 1) / r > needm[k])
                needm[k] = (start + t + 1) / r;
        }
    }
    *last_pub = lp;
    int ok = 1;
    for (int i = 0; ok && i < L; i++) {
        if (!c->kv_source[i]) continue;
        const int r = c->compress_ratio[i], host = c->max_positions / r;
        if (need[i] > host) need[i] = host;
        if (needm && needm[i] > host) needm[i] = host;
        const int want = need[i] > E / r ? need[i] : E / r;
        ok = v41u_cache(ch, m, i, want > 0 ? want : 1);
    }
    return ok;
}
/* what comes back at a forward's end (v41u_state_pull), from float base of pull: each layer's window
 * rows, its compressed rows, keys and group at pc[i]; the floats it takes */
static size_t v41u_pull_plan(const V41U *ch, const Model *m, int start, int n, size_t base, size_t *pc) {
    const Cfg *c = &m->c;
    const int E = start + n;
    const size_t pw = (size_t)(n < ch->W ? n : ch->W) * c->head_dim;
    size_t pn = 0;
    for (int i = 0; i < c->n_layers; i++) {
        pc[i] = base + pn; pn += pw;
        const int r = c->compress_ratio[i];
        if (!c->kv_source[i]) continue;
        pn += (size_t)(E / r - start / r) * (c->head_dim + c->index_head_dim);
        if (r > 1) pn += (size_t)2 * r * c->head_dim;
    }
    return pn;
}
/* a forward's pulled state (pull at pc, its frame waited for) into the bound sequence's host arrays,
 * as the CPU would have left them; the device's copy of that slot valid up to start + n */
static void v41u_state_apply(V41U *ch, Model *m, int start, int n, const size_t *pc, size_t pw, int last_pub) {
    const Cfg *c = &m->c;
    const int L = c->n_layers, W = ch->W, hd = c->head_dim, E = start + n, sl = v41u_slot(ch);
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        const float *pl = (const float *)cc_ptr(v41u_bank(ch, v41u_dev_of(ch, i))->pull);   /* its device's */
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
    ch->win_valid[sl] = E;
    int *kvv = v41u_kvv(ch, m, sl);
    for (int i = 0; i < L; i++) if (c->kv_source[i]) kvv[i] = E / c->compress_ratio[i];
}

/* ---- V41_CHAIN_MOE: a decode step's routed experts in the chain (PLAN-scale S3c) ------------
 * Opt-in (V41_CHAIN_MOE=1), the SYCL backend only (cc_moe's fmt 7 and act 1 are its), a decode
 * step's one row through the chain only. Per layer, in the attention+shared frame: the gate's
 * logits from the normalized FFN input (nrm), DeepSeek's router (cc_route kind 2: moe_gate's
 * arithmetic), the experts the tier holds on this device (cc_moe parts 1, the tier's tables).
 * Then the host computes the misses (the table's zero entries) on the CPU, moe_vk_cpu's
 * w * expert(x) per slot, summed in rank order into hy; then a frame: y = hy + each resident
 * slot's w * down in rank order (cc_moe parts 2), and the FFN add with y in place of the host's
 * routed row. moe_run_at(with_shared 0) is what it stands for: the routed experts only, the
 * shared expert on the device (v41u_shared). Anything refused: today's host step for good.
 * V41_CHAIN_MOE_CHECK=n: the first n (layer, step) pairs also through moe_run_at, compared. */
#if defined(COLI_CUDA) && defined(COLI_SYCL)
#define V41U_CMOE 1
#ifndef COLI_VK_XB_ENTRY
#define COLI_VK_XB_ENTRY 32
#endif
static void v41u_cmoe_off(V41U *ch, const char *why) {
    if (ch->cm_state >= 0)
        fprintf(stderr, "[CUDA] deepseek_v41 chain MoE (V41_CHAIN_MOE): %s; the host computes the routed experts\n", why);
    ch->cm_state = -1;
}
/* layer i's table from the tier: each entry's first sizeof(CcExpert) bytes are a CcExpert on
 * this backend (zero: not resident on this device) */
static int v41u_cmoe_read(V41U *ch, int i) {
    const int E = ch->cm_E;
    if (XT(dev_table)(i, ch->cm_ent) < 0) return 0;
    CcExpert *t = ch->cm_tabh + (size_t)i * E;
    for (int e = 0; e < E; e++) memcpy(&t[e], ch->cm_ent + (size_t)e * COLI_VK_XB_ENTRY, sizeof(CcExpert));
    return 1;
}
/* At a frame's start, nothing in flight (every chain frame here is waited for, no tier batch is
 * out): the tier settles, and the layers whose entries changed since the tables were built
 * (vkt_dev_changes; every layer when it cannot say, or whole) are read again and go up. Called
 * before each layer's frame, since the host's misses (vkt_note) and vkt_dev_count are quiescent
 * points that may move residents between two layers. 0 = refused. */
static int v41u_cmoe_tables(V41U *ch, Model *m, int whole) {
    const int L = m->c.n_layers, E = ch->cm_E;
    XT(settle)();
    const unsigned ep = XT(epoch)();
    if (!whole && ep == ch->cm_epoch) return 1;
    memset(ch->cm_dirty, whole ? 1 : 0, (size_t)L);
    if (!whole) {
        const int nc = XT(dev_changes)(ch->cm_epoch, ch->cm_chg, L * E);
        if (nc < 0) memset(ch->cm_dirty, 1, (size_t)L);
        for (int k = 0; k < nc; k++) {
            const int l = ch->cm_chg[k] / E;
            if (l >= 0 && l < L) ch->cm_dirty[l] = 1;   /* a DSpark stage's (an extra layer) is not ours */
        }
    }
    int any = 0;
    for (int i = 0; i < L; i++) any |= ch->cm_dirty[i];
    if (!any) { ch->cm_epoch = ep; return 1; }
    const size_t ef = sizeof(CcExpert) / sizeof(float);   /* cc_write's offsets are in floats */
    const int was = ch->dcur;
    int ok = 1;
    for (int d = 0; ok && d < ch->nd; d++) {   /* a layer's table on its device (V41_CHAIN_SPLIT: a frame a device with dirty layers) */
        int lo, hi, some = 0;
        v41u_use(ch, d);
        v41u_span(ch, m, &lo, &hi);
        for (int i = lo; i < hi; i++) some |= ch->cm_dirty[i];
        if (!some) continue;
        ok = cc_begin();
        const int opened = ok;
        for (int i = lo; ok && i < hi; i++) {
            if (!ch->cm_dirty[i]) continue;
            ok = v41u_cmoe_read(ch, i) &&
                 cc_write(ch->cm_tab, (size_t)i * E * ef, ch->cm_tabh + (size_t)i * E, (size_t)E * sizeof(CcExpert));
        }
        if (opened) ok = cc_submit(1) && ok;
    }
    v41u_use(ch, was);
    if (!ok) return 0;
    ch->cm_reads++;
    ch->cm_epoch = ep;
    return 1;
}
/* once: V41_CHAIN_MOE=1 and everything it needs, else off for good (said once, why) */
static int v41u_cmoe_setup(V41U *ch, Model *m) {
    if (ch->cm_state) return ch->cm_state > 0;
    const char *on = getenv("V41_CHAIN_MOE");
    if (!(on && atoi(on) > 0)) { ch->cm_state = -2; return 0; }   /* not asked: silent */
    Cfg *c = &m->c;
    const int L = c->n_layers, D = c->dim, I = c->moe_inter, E = c->n_routed, K = c->n_activated;
    const char *why = NULL;
    if (sizeof(CcExpert) % sizeof(float) || sizeof(CcExpert) > COLI_VK_XB_ENTRY * sizeof(uint32_t)) why = "CcExpert's size";
    else if (!cc_moe_ready()) why = "the backend has no cc_route / cc_moe";
    else if (!xt_cuda() || !XT(ready)()) why = "the expert tier is not on this device";
    else if (XT(layers)() < L) why = "the tier does not serve every layer";
    else if (m->gx) why = "a GGUF's experts (cc_moe fmt 7 is the checkpoint's MXFP4)";
    else if (E < 1 || E > CC_ROUTE_EMAX || K < 1 || K > CC_ROUTE_KMAX || K > MOE_TOPK_MAX || K > E) why = "the router's shape";
    else if (D % 32 || I % 32 || D > CC_MOE_DIM_MAX || I > CC_MOE_DIM_MAX) why = "the experts' shape";
    ch->cm_E = E; ch->cm_K = K;
    ch->cm_act = c->swiglu_limit > 0.0f;   /* act 1: deepseek_v41.c's swiglu_into clamps; act 0 without a limit */
    ch->cm_lim = c->swiglu_limit > 0.0f ? c->swiglu_limit : 0.0f;
    float *bias = NULL;
    if (!why && (!(ch->cm_gate = (ColiCudaTensor **)calloc((size_t)L, sizeof *ch->cm_gate)) ||
                 !(ch->cm_tabh = (CcExpert *)calloc((size_t)L * E, sizeof(CcExpert))) ||
                 !(ch->cm_ent = (uint32_t *)calloc((size_t)E * COLI_VK_XB_ENTRY, sizeof(uint32_t))) ||
                 !(ch->cm_chg = (int *)malloc((size_t)L * E * sizeof(int))) || !(ch->cm_dirty = (unsigned char *)calloc((size_t)L, 1)) ||
                 !(ch->cm_contrib = (float *)malloc((size_t)K * D * sizeof(float))) ||
                 !(ch->cm_ref = (float *)malloc((size_t)D * sizeof(float))) || !(ch->cm_sc = (float *)malloc((size_t)E * sizeof(float))) ||
                 !(bias = (float *)malloc((size_t)L * E * sizeof(float)))))
        why = "memory";
    for (int i = 0; !why && i < L; i++) {
        Layer *l = &m->L[i];
        if (!l->gate_bias.w) { why = "a layer without a gate bias"; break; }
        memcpy(bias + (size_t)i * E, l->gate_bias.w, (size_t)E * sizeof(float));
        v41u_use(ch, v41u_dev_of(ch, i));
        cud_place(v41u_dev_of(ch, i));
        if (!(ch->cm_gate[i] = v41u_tb(&l->gate_w))) {
            fprintf(stderr, "[CUDA] deepseek_v41 chain MoE: layer %d's gate not on the device (the dense budget %.2f of %.2f GiB used)\n",
                    i, g_cud.used / 1073741824.0, g_cud.budget / 1073741824.0);
            why = "a gate the device's dense budget does not take";
        }
    }
    cud_place(0);
    for (int d = 0; d < ch->nd; d++) {   /* the router's and the experts' rows, every layer's bias and table, on each device */
        v41u_use(ch, d);
        if (!why && !(v41u_res(&ch->cm_bias, (size_t)L * E, CC_DEV) &&
                      cc_reserve(&ch->cm_tab, (size_t)L * E * sizeof(CcExpert), CC_DEV) && v41u_res(&ch->cm_lg, (size_t)E, CC_DEV) &&
                      cc_reserve(&ch->cm_ids, (size_t)K * sizeof(int), CC_DOWN) && v41u_res(&ch->cm_w, (size_t)K, CC_DOWN) &&
                      v41u_res(&ch->cm_hid, cc_moe_hid_floats(1, K, D, I, E), CC_DEV) && v41u_res(&ch->cm_y, (size_t)D, CC_DEV) &&
                      v41u_res(&ch->cm_hy, (size_t)D, CC_UP) && v41u_res(&ch->cm_yd, (size_t)D, CC_DOWN) &&
                      v41u_res(&ch->cm_flag, 16, CC_DOWN)))
            why = "device memory for the router and the experts' rows";
        if (!why && !(cc_begin() && cc_write(ch->cm_bias, 0, bias, (size_t)L * E * sizeof(float)) && cc_submit(1)))
            why = "the gate biases did not reach the device";
    }
    v41u_use(ch, 0);
    free(bias);
    if (!why && !v41u_cmoe_tables(ch, m, 1)) why = "the tier's tables (vkt_dev_table) refused";
    if (why) { v41u_cmoe_off(ch, why); return 0; }
    const char *ck = getenv("V41_CHAIN_MOE_CHECK");
    ch->cm_check = ck && atoi(ck) > 0 ? atoi(ck) : 0;
    int res = 0;
    for (size_t j = 0; j < (size_t)L * E; j++) res += ch->cm_tabh[j].g != NULL;
    ch->cm_state = 1;
    fprintf(stderr, "[CUDA] deepseek_v41 chain MoE (V41_CHAIN_MOE): a decode step's router and resident experts in the chain, "
                    "the misses on the CPU (%d of %d experts resident now%s)\n", res, L * E, ch->cm_check ? "; V41_CHAIN_MOE_CHECK on" : "");
    if (ch->cm_check) fprintf(stderr, "[CUDA] deepseek_v41 chain MoE: checking the first %d (layer, step) pairs against moe_run_at\n", ch->cm_check);
    return 1;
}
static CcMoe v41u_cmoe_args_s(const V41U *ch, const Model *m, int i, int parts, int S) {   /* S rows (V41_CHAIN_ROWS) */
    const Cfg *c = &m->c;
    CcMoe p;
    memset(&p, 0, sizeof p);
    p.S = S; p.K = ch->cm_K; p.D = c->dim; p.I = c->moe_inter; p.fmt = 7; p.gs = 32; p.act = ch->cm_act; p.parts = parts;
    p.t_off = i * ch->cm_E; p.x_off = 0; p.x_row = c->dim; p.i_off = 0; p.i_row = ch->cm_K; p.w_off = 0; p.w_row = ch->cm_K;
    p.y_off = 0; p.y_row = c->dim; p.h_off = 0; p.E = ch->cm_E; p.planar = 0; p.fmt_d = 0; p.lay_S = 0; p.lay_s0 = 0;
    p.act_lim = ch->cm_lim;
    return p;
}
static CcMoe v41u_cmoe_args(const V41U *ch, const Model *m, int i, int parts) { return v41u_cmoe_args_s(ch, m, i, parts, 1); }
/* layer i's router and resident experts, appended to the open attention+shared frame (nrm: the
 * normalized FFN input the host step reads as h2d) */
static int v41u_cmoe_route_s(V41U *ch, Model *m, int i, int S) {   /* nrm's S rows (V41_CHAIN_ROWS: every conversation's) */
    const Cfg *c = &m->c;
    const int E = ch->cm_E, K = ch->cm_K;
    CcRoute r;
    memset(&r, 0, sizeof r);
    r.S = S; r.E = E; r.K = K; r.kind = 2;
    r.l_off = 0; r.l_row = E; r.b_off = i * E; r.has_bias = 1;   /* moe_gate: the bias for the pick only */
    r.i_off = 0; r.i_row = K; r.w_off = 0; r.w_row = K;
    r.n_group = 0; r.topk_group = 0;
    r.norm = c->norm_topk_prob && K > 1 ? 1 : 0;   /* moe_gate: by the sum + 1e-20 only with norm_topk_prob */
    r.temp = c->gate_temp; r.scale = c->route_scale; r.norm_eps = 1e-20f;
    /* the routes down, then the word: the host's misses run while the frame's later ops compute */
    return cc_matmul(ch->cm_gate[i], ch->nrm, 0, ch->cm_lg, 0, S) && cc_route(ch->cm_lg, ch->cm_bias, ch->cm_ids, ch->cm_w, &r) &&
           (v41u_mark_on() ? cc_mark() : cc_signal(ch->cm_flag, 0, ++ch->cm_seq));
}
static int v41u_cmoe_res_s(V41U *ch, Model *m, int i, int S) {   /* the resident experts (cc_moe parts 1) */
    CcMoe p = v41u_cmoe_args_s(ch, m, i, 1, S);
    return cc_moe(ch->cm_tab, ch->nrm, ch->cm_ids, ch->cm_w, ch->cm_hid, ch->cm_y, ch->cm_hy, &p);
}
static int v41u_cmoe_dev_s(V41U *ch, Model *m, int i, int S) { return v41u_cmoe_route_s(ch, m, i, S) && v41u_cmoe_res_s(ch, m, i, S); }
static int v41u_cmoe_dev(V41U *ch, Model *m, int i) { return v41u_cmoe_dev_s(ch, m, i, 1); }
/* the last frame's three ops in one launch: 1 done (the streams swapped, as v41u_post leaves them), else not taken */
static int v41u_tail(V41U *ch, Model *m, const CcMoe *p2, CcBuf *hp, int n) {
    Cfg *c = &m->c; const int H = c->hc_mult, D = c->dim, HD = H * D, hr = 2 * H + H * H;
    CcMhc po = {n, H, D, c->hc_iters, 0, HD, 0, D, 0, hr, 0, HD, 0, 0, c->norm_eps, c->hc_eps, 0.f};
    if (cc_moe_tail(ch->cm_tab, ch->cm_ids, ch->cm_w, ch->cm_hid, ch->cm_hy, ch->cm_y, ch->ds, ch->br, ch->xs, hp, ch->xn, p2, &po) <= 0) return 0;
    CcBuf *t = ch->xs; ch->xs = ch->xn; ch->xn = t;
    ch->cm_tails++;
    return 1;
}
/* The host's part of layer i after the device's frame: the routing it made (ids, w back to the
 * host), the misses -- slots whose expert the device's table does not hold -- on the CPU
 * (moe_vk_cpu: w * expert(x) per slot, through the expert cache, each passed to the tier as
 * moe_vk_block's are), hy = their sum in rank order; the books moe_vk_block keeps. x: the host's
 * copy of the same normalized FFN input. A version 1, synchronous: the frame was waited for
 * before this; later the frame's resident experts can overlap this (cc_signal after cc_route,
 * the host's misses while cc_moe parts 1 runs, cc_wait_host before parts 2). */
/* One row of quant.h's matmul_mxfp4 (its I % 32 == 0 AVX2 branch, S = 1): the same operations in the same order, so a
 * miss computed here is the bits the one-row matmul_mxfp4 gives (moe_vk_cpu's, for a decode step) */
static inline float v41u_mx4_dot(const float *xs, const uint8_t *w, const uint8_t *scl, int I) {
#ifdef __AVX2__
    const int ng = I / 32;
    const __m128i lut2 = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < ng; g++) {
        __m128i by = _mm_loadu_si128((const __m128i *)(w + g * 16));
        __m128i lo = _mm_and_si128(by, m4), hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
        __m128i n0 = _mm_shuffle_epi8(lut2, _mm_unpacklo_epi8(lo, hi)), n1 = _mm_shuffle_epi8(lut2, _mm_unpackhi_epi8(lo, hi));
        const float *xg = xs + g * 32;
        __m256 ga = _mm256_mul_ps(_mm256_loadu_ps(xg), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n0)));
        ga = _mm256_fmadd_ps(_mm256_loadu_ps(xg + 8), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n0, 8))), ga);
        ga = _mm256_fmadd_ps(_mm256_loadu_ps(xg + 16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(n1)), ga);
        ga = _mm256_fmadd_ps(_mm256_loadu_ps(xg + 24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(n1, 8))), ga);
        acc = _mm256_fmadd_ps(ga, _mm256_set1_ps(mx4_scale(scl[g]) * 0.5f), acc);
    }
    return hsum256(acc);
#else
    float y; matmul_mxfp4(&y, xs, w, scl, 1, I, 1); return y;
#endif
}
/* A decode step's misses of layer i, all of them together (V41_CHAIN_MOE): the experts through the cache at once, then
 * one parallel loop over every miss's gate and up rows, the clamped SwiGLU per miss (swiglu_into), one over every miss's
 * down rows; contrib[k] = w[k] * down_k, the tier's notes as moe_vk_cpu's. The bits of moe_vk_cpu's one-row path (each
 * row v41u_mx4_dot = matmul_mxfp4's), in two OpenMP regions a layer instead of three a miss. 0: not this form (a GGUF
 * expert, an interleaved slot): the caller takes moe_vk_cpu. */
static int v41u_cmoe_misses(V41U *ch, Model *m, int i, const float *x, const int *ids, const float *w, const uint8_t *want) {
    const Cfg *c = &m->c;
    const int D = c->dim, I = c->moe_inter, K = ch->cm_K;
    int eid[CC_ROUTE_KMAX], slotk[CC_ROUTE_KMAX], nm = 0;
    for (int k = 0; k < K; k++) if (want[k]) { eid[nm] = ids[k]; slotk[nm] = k; nm++; }
    if (!nm) return 1;
    if (D % 32 || I % 32) return 0;
    Slot *sl[CC_ROUTE_KMAX];
    expert_slots_at(m, &m->cache[i], "layers", i, eid, nm, sl);
    for (int u = 0; u < nm; u++) if (!sl[u] || sl[u]->gt[0] >= 0 || sl[u]->gt[1] >= 0 || sl[u]->gt[2] >= 0 || vk_slot_ilv(sl[u])) return 0;
    if (!ch->cm_mbuf && !(ch->cm_mbuf = (float *)malloc((size_t)CC_ROUTE_KMAX * (2 * (size_t)I + D) * sizeof(float)))) return 0;
    float *gu = ch->cm_mbuf, *dn = ch->cm_mbuf + (size_t)CC_ROUTE_KMAX * 2 * I;   /* [miss][gate I | up I], [miss][D] */
    const double t0 = now_s();
    const int rb = I / 2, rd = D / 2;   /* the packed row bytes: gate/up rows of D inputs (D/2 bytes), down rows of I (I/2) */
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < nm * 2 * I; q++) {
        const int u = q / (2 * I), r = q - u * 2 * I, up = r >= I, o = up ? r - I : r;
        const Slot *s = sl[u];
        const uint8_t *wq = (up ? s->w3 : s->w1) + (size_t)o * rd, *sc = (up ? s->s3 : s->s1) + (size_t)o * (D / 32);
        gu[(size_t)u * 2 * I + r] = v41u_mx4_dot(x, wq, sc, D);
    }
    for (int u = 0; u < nm; u++) swiglu_into(c, gu + (size_t)u * 2 * I, gu + (size_t)u * 2 * I + I, (int64_t)I);
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < nm * D; q++) {
        const int u = q / D, o = q - u * D;
        const Slot *s = sl[u];
        dn[(size_t)u * D + o] = v41u_mx4_dot(gu + (size_t)u * 2 * I, s->w2 + (size_t)o * rb, s->s2 + (size_t)o * (I / 32), I);
    }
    m->t_expert += now_s() - t0;
    const int tl = v41_tier_layer(m, "layers", i);
    for (int u = 0; u < nm; u++) {
        float *dst = ch->cm_contrib + (size_t)slotk[u] * D;
        const float wk = w[slotk[u]];
        for (int d = 0; d < D; d++) dst[d] = wk * dn[(size_t)u * D + d];
        VktExpertSrc src = vk_slot_src(m, sl[u]);
        XT(note)(tl, eid[u], &src);
        vk_drop_rows(sl[u]);
    }
    return 1;
}
/* COLI_VK_DEV2 (a second card in the tier): the chain's tables name the first card's residents only
 * (dev_entry), so the second card's are misses here. Those go to the second card as one batch (XT(issue),
 * every other draw -1) while the CPU computes the rest; taken[d]: not the CPU's. 0 when nothing went. */
static int v41u_cmoe_d2_issue(Model *m, int i, int S, int K, const float *x, const int *ids, uint8_t *want, uint8_t *taken) {
    const int draws = S * K;
    memset(taken, 0, (size_t)draws);
    if (XT(devices)() < 2 || XT(home)() > 0) return 0;
    int idm[4 * CC_ROUTE_KMAX];
    for (int d = 0; d < draws; d++) idm[d] = want[d] ? ids[d] : -1;
    const int n = XT(issue)(v41_tier_layer(m, "layers", i), x, S, K, idm, taken);
    if (n) for (int d = 0; d < draws; d++) if (taken[d]) want[d] = 0;
    return n;
}
/* the second card's rows back (contrib[d * D] = w * row, want[d] set again for the rank-order sum); 0: its batch
 * failed (the tier goes on without it), the caller computes taken on the CPU */
static int v41u_cmoe_d2_join(int draws, int D, const float *w, uint8_t *want, const uint8_t *taken, float *contrib) {
    const float *dv[4 * CC_ROUTE_KMAX];
    if (!XT(join)(dv)) return 0;
    for (int d = 0; d < draws; d++) {
        if (!taken[d]) continue;
        float *dst = contrib + (size_t)d * D;
        const float wk = w[d], *src = dv[d];
        for (int e = 0; e < D; e++) dst[e] = wk * src[e];
        want[d] = 1;
    }
    return 1;
}
static void v41u_cmoe_host(V41U *ch, Model *m, int i, const float *x, int *ids, float *w) {
    const int D = m->c.dim, E = ch->cm_E, K = ch->cm_K;
    memcpy(ids, cc_ptr(ch->cm_ids), (size_t)K * sizeof(int));
    memcpy(w, cc_ptr(ch->cm_w), (size_t)K * sizeof(float));
    const CcExpert *t = ch->cm_tabh + (size_t)i * E;
    uint8_t want[CC_ROUTE_KMAX];
    int nmiss = 0;
    rt_count(i, ids, K);   /* the history (moe_vk_block's, backbone only; -1 is skipped) */
    for (int k = 0; k < K; k++) {
        const int e = ids[k], valid = e >= 0 && e < E, dev = valid && t[e].g != NULL;
        want[k] = valid && !dev;
        nmiss += want[k];
        if (dev) { ehit_mark(m, i, e); ch->cm_dev_slots++; }   /* expert_slots_at marks the misses */
    }
    ch->cm_host_slots += (unsigned long long)nmiss;
    float *hy = (float *)cc_ptr(ch->cm_hy);
    memset(hy, 0, (size_t)D * sizeof(float));
    if (!nmiss) return;
    const double tc0 = now_s();
    if (ch->cm_batch < 0) { const char *e = getenv("V41_CHAIN_MOE_BATCH"); ch->cm_batch = !(e && *e == '0'); }
    uint8_t taken[CC_ROUTE_KMAX];
    const int n2 = v41u_cmoe_d2_issue(m, i, 1, K, x, ids, want, taken);
    int ncpu = 0;
    for (int k = 0; k < K; k++) ncpu += want[k];
    if (ncpu && (!ch->cm_batch || !v41u_cmoe_misses(ch, m, i, x, ids, w, want)))   /* V41_CHAIN_MOE_BATCH=0: one miss at a time */
        moe_vk_cpu(m, &m->cache[i], "layers", i, K, x, 1, ids, w, want, ch->cm_contrib);
    if (n2 && !v41u_cmoe_d2_join(K, D, w, want, taken, ch->cm_contrib)) {
        if (!ch->cm_batch || !v41u_cmoe_misses(ch, m, i, x, ids, w, taken))
            moe_vk_cpu(m, &m->cache[i], "layers", i, K, x, 1, ids, w, taken, ch->cm_contrib);
        for (int k = 0; k < K; k++) want[k] |= taken[k];
    }
    ch->cm_d2_slots += (unsigned long long)n2;
    ch->cm_cpu_ms += (now_s() - tc0) * 1e3;
    for (int k = 0; k < K; k++) {   /* rank order, as moe_vk_block's sum */
        if (!want[k]) continue;
        const float *cv = ch->cm_contrib + (size_t)k * D;
        for (int d = 0; d < D; d++) hy[d] += cv[d];
    }
}
/* After layer i's parts-2+FFN frame (nothing in flight): the device's books for the step
 * (vkt_dev_count: heat, routed and served; it may move residents, which the next layer's
 * v41u_cmoe_tables picks up), and V41_CHAIN_MOE_CHECK's comparison with moe_run_at on the same
 * input. The check's moe_run_at is a second, real step: it goes through the cache and the tier
 * (issue, join, notes), so a checked run's hit counts and the tier's heat are not a plain run's. */
static void v41u_cmoe_after(V41U *ch, Model *m, Layer *l, int i, const float *x, const int *ids, const float *w, int checked) {
    const Cfg *c = &m->c;
    const int D = c->dim, E = ch->cm_E, K = ch->cm_K;
    int valid = 1;
    for (int k = 0; k < K; k++) valid &= ids[k] >= 0 && ids[k] < E;
    if (valid) XT(dev_count)(i, 1, K, ids);
    if (!checked) return;
    const float *y = (const float *)cc_ptr(ch->cm_yd);
    moe_run_at(m, l, &m->cache[i], "layers", i, E, K, x, 1, ch->cm_ref, 0);
    double mx = 0.0, md = 0.0;
    for (int d = 0; d < D; d++) {
        const double r = fabs((double)ch->cm_ref[d]), df = fabs((double)ch->cm_ref[d] - (double)y[d]);
        if (r > mx) mx = r;
        if (df > md || df != df) md = df;
    }
    const double rel = mx > 0.0 ? md / mx : md;
    if (rel > ch->cm_err_max || rel != rel) ch->cm_err_max = rel;
    ch->cm_err_sum += rel;
    ch->cm_checked++;
    /* the CPU's own routing of the same row (moe_run_at's: mvb_rows, moe_gate), against the device's */
    int cid[MOE_TOPK_MAX];
    float cw[MOE_TOPK_MAX];
    mvb_rows(ch->cm_sc, E, &l->gate_w, x, D, 1);
    moe_gate(m, l, E, K, ch->cm_sc, cid, cw);
    int same = 1;
    for (int k = 0; k < K; k++) {
        same &= cid[k] == ids[k];
        const double wd = fabs((double)cw[k] - (double)w[k]);
        if (cid[k] == ids[k] && wd > ch->cm_wdiff) ch->cm_wdiff = wd;
    }
    if (!same) ch->cm_route_diff++;
    if (ch->cm_checked <= 4 || rel > 1e-2)
        fprintf(stderr, "[CUDA] deepseek_v41 chain MoE check: layer %d, rel err %.3g%s\n", i, rel, same ? "" : " (the CPU routed otherwise)");
}
/* V41_CHAIN_ROWS: a batched decode step's misses of layer i for its S rows (x: S rows of D; want: S * K).
 * The union of the missed experts through the cache once (expert_slots_at, a cache's worth at a time),
 * then every (row, slot) pair's gate and up rows in one parallel loop, the clamped SwiGLU a pair, every
 * pair's down rows in another: each output v41u_mx4_dot of that row's own input, so a row's
 * contributions are the bits v41u_cmoe_misses gives it alone. contrib[(row * K + k) * D] = w * down;
 * the tier's notes once an expert. 0: not this form (the caller takes moe_vk_cpu a row at a time). */
static int v41u_cmoe_misses_rows(V41U *ch, Model *m, int i, int S, const float *x, const int *ids, const float *w, const uint8_t *want) {
    const Cfg *c = &m->c;
    const int D = c->dim, I = c->moe_inter, K = ch->cm_K, draws = S * K;
    if (D % 32 || I % 32) return 0;
    int uq[4 * CC_ROUTE_KMAX], du[4 * CC_ROUTE_KMAX], nu = 0;   /* the missed experts; each draw's place among them */
    for (int d = 0; d < draws; d++) {
        du[d] = -1;
        if (!want[d]) continue;
        int at = -1;
        for (int u = 0; u < nu; u++) if (uq[u] == ids[d]) { at = u; break; }
        if (at < 0) { at = nu; uq[nu++] = ids[d]; }
        du[d] = at;
    }
    if (!nu) return 1;
    const size_t pairs_max = (size_t)4 * CC_ROUTE_KMAX;
    if (!ch->rw_mbuf && !(ch->rw_mbuf = (float *)malloc(pairs_max * (2 * (size_t)I + D) * sizeof(float)))) return 0;
    float *gu = ch->rw_mbuf, *dn = ch->rw_mbuf + pairs_max * 2 * I;   /* [pair][gate I | up I], [pair][D] */
    const int cap = m->cache[i].cap, step = cap < MOE_ROW_CHUNK ? cap : MOE_ROW_CHUNK;
    if (step < 1) return 0;
    const int rb = I / 2, rd = D / 2, tl = v41_tier_layer(m, "layers", i);
    for (int u0 = 0; u0 < nu; u0 += step) {
        const int ne = nu - u0 < step ? nu - u0 : step;
        Slot *sl[MOE_ROW_CHUNK];
        expert_slots_at(m, &m->cache[i], "layers", i, uq + u0, ne, sl);
        for (int u = 0; u < ne; u++) if (!sl[u] || sl[u]->gt[0] >= 0 || sl[u]->gt[1] >= 0 || sl[u]->gt[2] >= 0 || vk_slot_ilv(sl[u])) return 0;
        int pd[4 * CC_ROUTE_KMAX], np = 0;   /* this part's pairs: their draws */
        const Slot *ps[4 * CC_ROUTE_KMAX];
        for (int d = 0; d < draws; d++) if (du[d] >= u0 && du[d] < u0 + ne) { pd[np] = d; ps[np] = sl[du[d] - u0]; np++; }
        const double t0 = now_s();
        #pragma omp parallel for schedule(static)
        for (int q = 0; q < np * 2 * I; q++) {
            const int u = q / (2 * I), r = q - u * 2 * I, up = r >= I, o = up ? r - I : r;
            const Slot *s = ps[u];
            const uint8_t *wq = (up ? s->w3 : s->w1) + (size_t)o * rd, *sc = (up ? s->s3 : s->s1) + (size_t)o * (D / 32);
            gu[(size_t)u * 2 * I + r] = v41u_mx4_dot(x + (size_t)(pd[u] / K) * D, wq, sc, D);
        }
        for (int u = 0; u < np; u++) swiglu_into(c, gu + (size_t)u * 2 * I, gu + (size_t)u * 2 * I + I, (int64_t)I);
        #pragma omp parallel for schedule(static)
        for (int q = 0; q < np * D; q++) {
            const int u = q / D, o = q - u * D;
            const Slot *s = ps[u];
            dn[(size_t)u * D + o] = v41u_mx4_dot(gu + (size_t)u * 2 * I, s->w2 + (size_t)o * rb, s->s2 + (size_t)o * (I / 32), I);
        }
        m->t_expert += now_s() - t0;
        for (int u = 0; u < np; u++) {
            float *dst = ch->rw_contrib + (size_t)pd[u] * D;
            const float wk = w[pd[u]];
            for (int d = 0; d < D; d++) dst[d] = wk * dn[(size_t)u * D + d];
        }
        for (int u = 0; u < ne; u++) {
            VktExpertSrc src = vk_slot_src(m, sl[u]);
            XT(note)(tl, uq[u0 + u], &src);
            vk_drop_rows(sl[u]);
        }
    }
    return 1;
}
/* v41u_cmoe_host for a batched step's S rows: the routing back (S * K), each row's books as a one-row
 * step keeps them (rt_count, the device's hits), the misses of every row (v41u_cmoe_misses_rows, or
 * moe_vk_cpu a row at a time), hy's row t = row t's misses summed in rank order */
static void v41u_cmoe_host_rows(V41U *ch, Model *m, int i, int S, const float *x, int *ids, float *w) {
    const int D = m->c.dim, E = ch->cm_E, K = ch->cm_K, draws = S * K;
    memcpy(ids, cc_ptr(ch->cm_ids), (size_t)draws * sizeof(int));
    memcpy(w, cc_ptr(ch->cm_w), (size_t)draws * sizeof(float));
    const CcExpert *t = ch->cm_tabh + (size_t)i * E;
    uint8_t want[4 * CC_ROUTE_KMAX];
    int nmiss = 0;
    for (int r = 0; r < S; r++) {
        rt_count(i, ids + (size_t)r * K, K);
        for (int k = 0; k < K; k++) {
            const int d = r * K + k, e = ids[d], valid = e >= 0 && e < E, dev = valid && t[e].g != NULL;
            want[d] = valid && !dev;
            nmiss += want[d];
            if (dev) { ehit_mark(m, i, e); ch->rw_dev_slots++; }
        }
    }
    ch->rw_host_slots += (unsigned long long)nmiss;
    float *hy = (float *)cc_ptr(ch->cm_hy);
    memset(hy, 0, (size_t)S * D * sizeof(float));
    if (!nmiss) return;
    if (ch->cm_batch < 0) { const char *e = getenv("V41_CHAIN_MOE_BATCH"); ch->cm_batch = !(e && *e == '0'); }
    uint8_t taken[4 * CC_ROUTE_KMAX];
    const int n2 = v41u_cmoe_d2_issue(m, i, S, K, x, ids, want, taken);
    int ncpu = 0;
    for (int d = 0; d < draws; d++) ncpu += want[d];
    if (ncpu && (!ch->cm_batch || !v41u_cmoe_misses_rows(ch, m, i, S, x, ids, w, want)))
        for (int r = 0; r < S; r++)
            moe_vk_cpu(m, &m->cache[i], "layers", i, K, x + (size_t)r * D, 1, ids + (size_t)r * K, w + (size_t)r * K, want + (size_t)r * K,
                       ch->rw_contrib + (size_t)r * K * D);
    if (n2 && !v41u_cmoe_d2_join(draws, D, w, want, taken, ch->rw_contrib)) {
        if (!ch->cm_batch || !v41u_cmoe_misses_rows(ch, m, i, S, x, ids, w, taken))
            for (int r = 0; r < S; r++)
                moe_vk_cpu(m, &m->cache[i], "layers", i, K, x + (size_t)r * D, 1, ids + (size_t)r * K, w + (size_t)r * K,
                           taken + (size_t)r * K, ch->rw_contrib + (size_t)r * K * D);
        for (int d = 0; d < draws; d++) want[d] |= taken[d];
    }
    ch->cm_d2_slots += (unsigned long long)n2;
    for (int r = 0; r < S; r++)
        for (int k = 0; k < K; k++) {   /* rank order, as moe_vk_block's sum */
            const int d = r * K + k;
            if (!want[d]) continue;
            const float *cv = ch->rw_contrib + (size_t)d * D;
            float *hr = hy + (size_t)r * D;
            for (int e = 0; e < D; e++) hr[e] += cv[e];
        }
}
#else
#define V41U_CMOE 0
#endif

/* Every layer for a prompt's n rows of streams h (positions start..): the streams after the
 * last layer and its FFN mix back into h and pre_mix (every row when all, else the last), the
 * host's state as the CPU would have left it. Returns the layers run (all of them), 0: not
 * taken (h and pre_mix untouched; *lost: a frame failed). */
static int v41u_forward(Model *m, float *h, float *pre_mix, int n, int start, int spec, int all, int *lost) {
    *lost = 0;
    Cfg *c = &m->c;
    const int L = c->n_layers, D = c->dim, H = c->hc_mult, HD = H * D, hr = 2 * H + H * H, hd = c->head_dim, rd = c->rope_dim;
    const int T = m->spec.active ? c->n_spec_targets : 0, E = start + n;
    if (spec || T || !v41u_wanted(m, n)) return 0;
    if (!v41u_setup(m)) return 0;
    V41U *ch = g_v41u;
    if (ch->split && n != 1) return 0;      /* V41_CHAIN_SPLIT: a decode step's one row; a prompt's layers the CPU's */
    const int sl = v41u_slot(ch);
    if (sl < 0 || sl >= ch->SL) return 0;   /* no slot bound (serve_mux binds one before every forward) */
    const double tfw = now_s();
    v41u_lower(m, start);   /* this forward writes from start: nothing the device holds past it is the host's */
    const int W = ch->W, LR = W + ch->K, rows = n < ch->rows ? n : ch->rows;
    /* a prompt of more than one chunk; COLI_CUDA_V41_MIN=<rows>: from that many rows on (1: every step, a decode's
     * one row too -- PLAN-scale S3) */
    const int min_rows = v41u_min_rows();
    if (min_rows ? n < min_rows : n <= rows) return 0;
    /* in segments of whole chunks, as even as they come, up to v41u_seg_rows each */
    const int chunks = (n + rows - 1) / rows, nseg = (n + v41u_seg_rows(ch) - 1) / v41u_seg_rows(ch);
    const int cand = v41u_cand_on(c);
    int per = (chunks + nseg - 1) / nseg * rows;
    if (cand && (size_t)per * (size_t)E >= (size_t)0x7fffffff) return 0;   /* the mask's offsets */
    V41UFwd f = {start, n, 0, 0, rows * rd, NULL, NULL, 0, 0};
    f.keyl = (int *)malloc((size_t)L * n * sizeof(int));
    int *need = (int *)calloc((size_t)2 * L, sizeof(int)), *needm = need && ch->split ? need + L : NULL;   /* needm: the mirrors' rows */
    size_t *pc = (size_t *)calloc((size_t)L, sizeof(size_t));
    int *wl = (int *)malloc((size_t)rows * LR * sizeof(int));
    float *cs_ = (float *)malloc((size_t)(rows + ch->rmax) * 2 * rd * sizeof(float));
    float *er = m->engram.active ? (float *)malloc((size_t)rows * m->engram.cols * m->engram.head_dim * sizeof(float)) : NULL;
    float *outs = (float *)malloc((size_t)(all ? n : 1) * HD * sizeof(float)), *pres = (float *)malloc((size_t)(all ? n : 1) * H * sizeof(float));
    int ok = f.keyl && need && pc && wl && cs_ && (!m->engram.active || er) && outs && pres;
    /* whose index keys each index source reads, per row (indexer_run, published_owner), and the
     * rows of each kv_source layer the forward reads that it does not produce first */
    int last_pub = -1;
    ok = ok && v41u_plan_keys(ch, m, start, n, f.keyl, need, needm, &last_pub);
    /* what comes back at the end: each layer's window rows, its compressed rows, keys and group */
    const size_t pw = (size_t)(n < W ? n : W) * hd;
    const size_t pn = ok ? v41u_pull_plan(ch, m, start, n, 0, pc) : 0;
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
    if (ok && ch->split) {   /* the same buffers on the second device (one row: one chunk, one segment) */
        v41u_use(ch, 1);
        ok = v41u_scratch(ch, m, rows, E) && v41u_res(&ch->pull, pn, CC_DOWN) && v41u_res(&ch->lw_list, (size_t)n * LR, CC_DEV) &&
             v41u_res(&ch->lw_xs, (size_t)per * HD, CC_DEV) && v41u_res(&ch->lw_hp, (size_t)per * hr, CC_DEV) &&
             v41u_res(&ch->lw_ds, (size_t)per * D, CC_DEV) && (!cand || v41u_res(&ch->lw_mask, ((size_t)per * E + 3) / 4, CC_DEV));
        v41u_use(ch, 0);
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
    for (int d = 0; ok && d < ch->nd; d++) {   /* each device its layers' state and the lists (V41_CHAIN_SPLIT: two) */
        v41u_use(ch, d);
        ok = cc_begin() && v41u_state_up(ch, m, start, need, needm) && cc_submit(1);
        for (int c0 = 0; ok && c0 < n; c0 += rows) {   /* the window halves of every row's list */
            const int nr = n - c0 < rows ? n - c0 : rows;
            v41u_chunk_tables(ch, m, start, start + c0, nr, f.csB, cs_, wl);
            ok = cc_begin() && cc_write(ch->lw_list, (size_t)c0 * LR, wl, (size_t)nr * LR * sizeof(int)) && cc_submit(1);
        }
    }
    v41u_use(ch, 0);
    /* past the cross-layer source only the chunks of the prompt's tail (v41_prefill_tail) from tf on */
    const int tail = m->ptail_rows > 0 && m->ptail_after < L - 1;
    int tf = 0;
    if (tail) while (tf < n && tf + (n - tf < rows ? n - tf : rows) <= n - m->ptail_rows) tf += rows;
    int stopped = 0;
    /* V41_CHAIN_MOE: a decode step's routed experts in the chain (v41u_cmoe_*); off: moe_run_at */
    int cmoe = 0;
#if V41U_CMOE
    cmoe = ok && n == 1 && v41u_cmoe_setup(ch, m);
    if (cmoe) {
        static long warm = -1;   /* V41_CHAIN_MOE_WARM=<n>: the split's books start over after n decode steps (the warm ones) */
        if (warm < 0) { const char *e = getenv("V41_CHAIN_MOE_WARM"); warm = e && atol(e) > 0 ? atol(e) : 0; }
        if (warm && ch->cm_steps == (unsigned long long)warm) {
            ch->cm_steps = 0; ch->cm_layers = 0; ch->cm_dev_slots = ch->cm_host_slots = 0;
            ch->cm_dev_ms = ch->cm_host_ms = ch->cm_p2_ms = ch->cm_tab_ms = ch->cm_cpu_ms = ch->cm_cnt_ms = 0;
            ch->cm_wait_ms = 0; warm = 0;
        }
        ch->cm_steps++;
    }
#endif
    (void)cmoe;
    /* V41_CHAIN_DIRECT: a decode step's one row (one chunk of one segment) keeps its streams in the layer's buffers */
    const int direct = n == 1 && v41u_direct_on();
    for (int s0 = 0; ok && !stopped && s0 < n; s0 += per) {   /* a segment's rows through every layer */
        const int se = s0 + per < n ? s0 + per : n;
        int loaded = 0;   /* direct: the row's streams and mixes are in xs and hpf (from the first layer's frame on) */
        f.s0 = s0;
        for (int c0 = s0; ok && c0 < se; c0 += rows) {   /* its streams, a chunk at a time */
            const int nr = se - c0 < rows ? se - c0 : rows;
            ok = cc_begin() && cc_write(ch->lw_xs, (size_t)(c0 - s0) * HD, h + (size_t)c0 * HD, (size_t)nr * HD * sizeof(float)) &&
                 cc_write(ch->lw_hp, (size_t)(c0 - s0) * hr, hp0, (size_t)nr * hr * sizeof(float)) && cc_submit(1);
        }
        for (int i = 0; ok && i < L; i++) {
            Layer *l = &m->L[i];
            if (ch->split && i == ch->split) {   /* the primary's layers are through: its state back, the row's streams and mix to the second device */
                const double th0 = now_s();
                /* V41_CHAIN_DIRECT: the row is in the layer's buffers (xs, hpf), not in the stash; it goes to the second
                 * device's stash either way, and that device's first layer loads it from there (loaded = 0) */
                ok = cc_begin() && cc_copy(ch->xd, 0, direct ? ch->xs : ch->lw_xs, 0, (size_t)HD) &&
                     cc_copy(ch->hpd, 0, direct ? ch->hpf : ch->lw_hp, 0, (size_t)hr) &&
                     v41u_state_pull(ch, m, start, E, pc, pw) && cc_submit(1);
                const float *x0 = (const float *)cc_ptr(ch->xd), *p0 = (const float *)cc_ptr(ch->hpd);
                v41u_use(ch, 1);
                /* the frames from here on read and write the second device's list: f.list was the primary's, taken before
                 * the layers (it did once: layer 20's top-k wrote the primary's buffer from a kernel on the second card, a GPU
                 * page fault there; on the CPU test device, one memory for both, it had computed the right selection) */
                f.list = ch->lw_list;
                ok = ok && cc_begin() && cc_write(ch->lw_xs, 0, x0, (size_t)HD * sizeof(float)) &&
                     cc_write(ch->lw_hp, 0, p0, (size_t)hr * sizeof(float)) && cc_submit(1);
                loaded = 0;
                ch->sp_steps++; ch->sp_ms += (now_s() - th0) * 1e3;
                if (!ok) break;
            }
            const int first = tail && i > m->ptail_after && tf > s0 ? tf : s0;
            if (first >= se) continue;
            double ta = now_s();
            const double ta0 = ta;
            XT(stream_prefetch)(i, se - first);
#if V41U_CMOE
            const double tt0 = now_s();
            const int tabs_ok = !cmoe || v41u_cmoe_tables(ch, m, 0);
            if (cmoe) ch->cm_tab_ms += (now_s() - tt0) * 1e3;
            if (!tabs_ok) {   /* the tier's residents moved: this layer's frame reads the new tables */
                if (cc_lost()) { ok = 0; break; }
                v41u_cmoe_off(ch, "the tier's tables refused");
                cmoe = 0;
            }
            const int cm_layer = cmoe;   /* this layer's routed experts in the chain */
#endif
            for (int c0 = first; ok && c0 < se; c0 += rows) {
                if (coli_prompt_break()) { stopped = 1; break; }
                const int nr = se - c0 < rows ? se - c0 : rows, pb = start + c0, lc = c0 - s0;
                f.pb = pb; f.nr = nr; f.lo = c0 * LR;
                v41u_chunk_tables(ch, m, start, pb, nr, f.csB, cs_, wl);   /* the RoPE rows (the list's window half is up) */
                ok = cc_begin() && cc_write(ch->cs, 0, cs_, (size_t)(f.csB + (nr + ch->rmax - 1) * rd) * sizeof(float));
                if (ok && (!direct || !loaded)) {   /* the chunk's streams and mixes from the stash (direct: the row's, once) */
                    ok = cc_copy(ch->xs, 0, ch->lw_xs, (size_t)lc * HD, (size_t)nr * HD) && cc_copy(ch->hpf, 0, ch->lw_hp, (size_t)lc * hr, (size_t)nr * hr);
                    loaded = 1;
                }
                if (ok && l->engram_index >= 0) ok = v41u_engram(ch, m, i, nr, pb, er);
                ok = ok && v41u_pre(ch, m, ch->fna[i], ch->o_hca[i], ch->hpa, ch->hpf, ch->o_an[i], nr) &&
                     v41u_attention(ch, m, i, &f) && v41u_post(ch, m, ch->hpa, nr) &&
                     v41u_pre(ch, m, ch->fnf[i], ch->o_hcf[i], ch->hpf, ch->hpa, ch->o_fn[i], nr) &&
                     cc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D);
                /* the shared expert, then the chunk's streams, mixes and shared branch to the stash (direct: they stay where they are) */
#define V41U_SHARED_STASH() (v41u_shared(ch, m, l, nr) && \
                     (direct || (cc_copy(ch->lw_xs, (size_t)lc * HD, ch->xs, 0, (size_t)nr * HD) && \
                                 cc_copy(ch->lw_hp, (size_t)lc * hr, ch->hpf, 0, (size_t)nr * hr) && \
                                 cc_copy(ch->lw_ds, (size_t)lc * D, ch->ds, 0, (size_t)nr * D))))
                int shared_done = 0;
#if V41U_CMOE
                /* the router and the resident experts in the same frame; refused (not a lost device): the host step from here on */
                if (ok && cmoe) {
                    int r;
                    if (v41u_shared_late()) {   /* the routes and their word first: the shared expert and the residents run beside the host's misses */
                        r = v41u_cmoe_route_s(ch, m, i, 1);
                        if (r) { shared_done = 1; ok = V41U_SHARED_STASH(); r = ok && v41u_cmoe_res_s(ch, m, i, 1); }
                    } else { shared_done = 1; ok = V41U_SHARED_STASH(); r = ok && v41u_cmoe_dev(ch, m, i); }
                    if (ok && !r) {
                        if (cc_lost()) ok = 0;
                        else { v41u_cmoe_off(ch, "the device's router or experts refused (cc_matmul/cc_route/cc_moe parts 1)"); cmoe = 0; }
                    }
                }
#endif
                if (ok && !shared_done) ok = V41U_SHARED_STASH();
#undef V41U_SHARED_STASH
#if V41U_CMOE
                if (cmoe) {   /* the frame left running: its word says the routes (and h2d, copied before them) are down */
                    ok = ok && cc_submit(0);
                    if (ok) {
                        const double tw = now_s();
                        int rd = 0;
                        for (unsigned long spin = 0; (rd = v41u_routes_down(ch)) == 0; spin++)
                            if ((spin & 0xFFFFu) == 0 && (cc_lost() || now_s() - tw > 60.0)) {
                                fprintf(stderr, "[CUDA] deepseek_v41 chain MoE: the device did not reach layer %d's routes in 60 s\n", i);
                                ok = 0; break;
                            }
                        if (rd < 0) ok = 0;
                        ch->cm_wait_ms += (now_s() - tw) * 1e3;
                    }
                } else
#endif
                ok = ok && cc_submit(1);
                if (ok) memcpy(ch->lw_in + (size_t)lc * D, cc_ptr(ch->h2d), (size_t)nr * D * sizeof(float));
            }
            m->t_attn += now_s() - ta;
            ch->attn_ms += (now_s() - ta0) * 1e3;
            if (!ok || stopped) break;
            const double t1 = now_s();
#if V41U_CMOE
            int cm_ids[CC_ROUTE_KMAX];
            float cm_w[CC_ROUTE_KMAX];
            if (cm_layer && cmoe) ch->cm_dev_ms += (t1 - ta0) * 1e3;
            if (cmoe) {   /* n == 1: one chunk, its row at lw_in's 0 */
                v41u_cmoe_host(ch, m, i, ch->lw_in, cm_ids, cm_w);
                ch->cm_host_ms += (now_s() - t1) * 1e3;
            } else
#endif
            moe_run_at(m, l, &m->cache[i], "layers", i, c->n_routed, c->n_activated, ch->lw_in + (size_t)(first - s0) * D, se - first,
                       ch->lw_out + (size_t)(first - s0) * D, 0);
            ch->host_ms += (now_s() - t1) * 1e3;
            const double tf = now_s();
            int cm_checked = 0;
            for (int c0 = first; ok && c0 < se; c0 += rows) {   /* each chunk's FFN branch into its streams */
                const int nr = se - c0 < rows ? se - c0 : rows, lc = c0 - s0;
                CcBuf *rin = ch->routed;   /* the routed row the add reads: the host's, or the device's y */
                CcEw add = {CC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                int tailed = 0, restored = 0;
                ok = 1;
/* the chunk's streams, mixes and shared branch back from the stash (direct: they never left) */
#define V41U_RESTORE() (direct || (cc_copy(ch->xs, 0, ch->lw_xs, (size_t)lc * HD, (size_t)nr * HD) && \
                     cc_copy(ch->hpf, 0, ch->lw_hp, (size_t)lc * hr, (size_t)nr * hr) && cc_copy(ch->ds, 0, ch->lw_ds, (size_t)lc * D, (size_t)nr * D)))
#if V41U_CMOE
                if (cmoe) {   /* y = hy + the resident slots' w * down in rank order (cc_moe parts 2), then the add from y */
                    CcMoe p2 = v41u_cmoe_args(ch, m, i, 2);
                    cm_checked = ch->cm_check > 0;
                    ok = cc_begin();
                    if (ok && !cm_checked && v41u_tail_on()) {   /* that, the add and the write back in one launch (a checked step: the three ops) */
                        ok = V41U_RESTORE(); restored = 1;
                        if (ok && v41u_tail(ch, m, &p2, ch->hpf, nr)) { tailed = 1; rin = ch->cm_y; }
                    }
                    if (tailed) ;
                    else if (ok && cc_moe(ch->cm_tab, ch->nrm, ch->cm_ids, ch->cm_w, ch->cm_hid, ch->cm_y, ch->cm_hy, &p2) &&
                        (!cm_checked || cc_copy(ch->cm_yd, 0, ch->cm_y, 0, (size_t)D))) rin = ch->cm_y;
                    else if (ok) {   /* refused: this layer's routed row from moe_run_at after all, the host step from here on */
                        ok = !cc_lost() && cc_submit(1);
                        if (ok) {
                            v41u_cmoe_off(ch, "the device's experts refused (cc_moe parts 2)");
                            cmoe = 0; cm_checked = 0;
                            moe_run_at(m, l, &m->cache[i], "layers", i, c->n_routed, c->n_activated, ch->lw_in, 1, ch->lw_out, 0);
                        }
                    }
                }
#endif
                if (ok && rin == ch->routed) {
                    memcpy(cc_ptr(ch->routed), ch->lw_out + (size_t)lc * D, (size_t)nr * D * sizeof(float));
                    ok = cc_begin();
                }
                if (ok && !restored) ok = V41U_RESTORE();
                if (!tailed) ok = ok && cc_ops()->ew(ch->br, rin, ch->ds, NULL, NULL, &add) && v41u_post(ch, m, ch->hpf, nr);
                ok = ok && (direct || cc_copy(ch->lw_xs, (size_t)lc * HD, ch->xs, 0, (size_t)nr * HD)) && cc_submit(1);
#undef V41U_RESTORE
            }
            ch->ffn_ms += (now_s() - tf) * 1e3;
#if V41U_CMOE
            if (ok && cmoe) {
                ch->cm_p2_ms += (now_s() - tf) * 1e3;
                ch->cm_layers++;
                if (cm_checked) ch->cm_check--;
                { const double tn0 = now_s();
                  v41u_cmoe_after(ch, m, l, i, ch->lw_in, cm_ids, cm_w, cm_checked);
                  if (!cm_checked) ch->cm_cnt_ms += (now_s() - tn0) * 1e3; }
            }
#else
            (void)cm_checked;
#endif
        }
        /* the segment's streams and mixes (every row's, or the prompt's last) and, after the last
         * one, what the host's state gets back */
        for (int c0 = s0; ok && !stopped && c0 < se; c0 += rows) {
            const int nr = se - c0 < rows ? se - c0 : rows, last = c0 + nr == n;
            if (!all && !last) continue;
            ok = cc_begin() && cc_copy(ch->xd, 0, direct ? ch->xs : ch->lw_xs, direct ? 0 : (size_t)(c0 - s0) * HD, (size_t)nr * HD) &&
                 cc_copy(ch->hpd, 0, direct ? ch->hpf : ch->lw_hp, direct ? 0 : (size_t)(c0 - s0) * hr, (size_t)nr * hr) &&
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
    v41u_use(ch, 0);
    if (!ok || stopped) {   /* a frame failed, or the prompt was stopped: the host state as it was, the device's stale */
        free(f.keyl); free(need); free(pc); free(wl); free(cs_); free(er); free(outs); free(pres);
        v41u_lower(m, start);
        v41u_mirror_reset(ch, m);
        if (!ok) { *lost = 1; g_v41u_off = 1; ch->ok = 0; fprintf(stderr, "[CUDA] deepseek_v41 chain: a frame failed; the CPU runs the layers from now on\n"); }
        return 0;
    }
    v41u_state_apply(ch, m, start, n, pc, pw, last_pub);   /* every chunk is through: the host's state as the CPU would have left it */
    if (all) { memcpy(h, outs, (size_t)n * HD * sizeof(float)); memcpy(pre_mix, pres, (size_t)n * H * sizeof(float)); }
    else { memcpy(h + (size_t)(n - 1) * HD, outs, (size_t)HD * sizeof(float)); memcpy(pre_mix + (size_t)(n - 1) * H, pres, (size_t)H * sizeof(float)); }
    free(f.keyl); free(need); free(pc); free(wl); free(cs_); free(er); free(outs); free(pres);
    ch->forwards++;
    if (n == 1) ch->one_row++;
    ch->fwd_ms += (now_s() - tfw) * 1e3;
    return L;
}

/* ---- V41_CHAIN_ROWS: a decode step's rows of every active conversation together (M3) --------
 * Opt-in (V41_CHAIN_ROWS=1; off, forward_rows takes M2's row at a time as before), KV_SLOTS > 1
 * with the decode steps on the chain (v41u_mux_rows), 2..4 rows of different slots. Per layer one
 * frame for the S rows where the arithmetic is a row's own: the engram gate, the hyper-connections'
 * split, collapse and write back, the trunk's matrices (cc_matmul with S rows), the shared expert,
 * the router and the resident experts (V41_CHAIN_MOE: cc_route and cc_moe parts 1 with S rows, one
 * cc_signal); the host's misses of every row (v41u_cmoe_host_rows; without V41_CHAIN_MOE moe_run_at
 * over the S rows); then a frame of cc_moe parts 2, the FFN add and write back for the S rows. A
 * slot at a time inside the first frame: the window ring's row, the compressor and the index keys,
 * the indexer's scores, candidates and top-k (each row's own width, positions and key source), the
 * sparse attention (cc_ds_attn S = 1 on that slot's regions). RoPE reads a table of a row each: the
 * cs buffer holds [S rows of rope_window at each row's position | S rows of rope_compress at it |
 * a block of rmax rows of rope_compress a row, positions pos - rmax + 1 .. pos (a latent's group)].
 * Each row's host side is what forward_full does for a one-row step: the slot's plan, state_up,
 * list and engram rows with it bound before the frames, its pulled state applied with it bound
 * after (apply 0: neither the pull nor the validity -- V41_CHAIN_ROWS_CHECK). Returns 1 with h
 * (S rows of streams in: the embeddings; out: the last layer's) and pre_mix (S rows of H); 0 not
 * taken or refused (h untouched, the slots' device copies lowered to their positions; *lost: the
 * device is gone, the chain off for good). Leaves some slot bound: the caller binds. */
static void v41_mux_bind(Model *m, int slot);   /* deepseek_v41.c's: the Model holds that slot's sequence */
#define V41U_ROWS_MAX 4
static int v41u_rows_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("V41_CHAIN_ROWS"); on = e && atoi(e) > 0; }
    return on;
}
static int g_v41u_rows_check = -1;   /* V41_CHAIN_ROWS_CHECK: the batched steps still to check */
static int v41u_rows_check_take(void) {
    if (g_v41u_rows_check < 0) { const char *e = getenv("V41_CHAIN_ROWS_CHECK"); g_v41u_rows_check = e && atoi(e) > 0 ? atoi(e) : 0; }
    if (g_v41u_rows_check <= 0) return 0;
    g_v41u_rows_check--;
    return 1;
}
/* forward_rows' step for the batched path: asked for, the decode steps on the chain, 2..4 rows of
 * different slots in range */
static int v41u_rows_wanted(Model *m, const int *slots, int n) {
    if (!v41u_rows_on() || n < 2 || n > V41U_ROWS_MAX || !v41u_mux_rows(m)) return 0;
    for (int t = 0; t < n; t++) {
        if (slots[t] < 0 || slots[t] >= g_v41_mux_slots) return 0;
        for (int u = 0; u < t; u++) if (slots[u] == slots[t]) return 0;
    }
    return 1;
}
static int v41u_close(int ok) { return cc_submit(1) && ok; }   /* the open frame ends (waited for) whatever was refused in it */
/* layer i's attention for a batched step's S rows (slots[t] at pos[t]; keyl[t * L + i] row t's key
 * source): v41u_attention's ops, the projections, norms and RoPE over the S rows, the state's a slot
 * at a time */
static int v41u_attention_rows(V41U *ch, Model *m, int i, int S, const int *slots, const int *pos, const int *keyl) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    const int L = c->n_layers, hd = c->head_dim, nh = c->n_heads, rd = c->rope_dim, QL = c->q_lora, r = c->compress_ratio[i];
    const int qrow = nh * hd, W = ch->W, LR = W + ch->K, rmax = ch->rmax;
    const int tcs = r > 0 ? S * rd : 0;   /* row t's RoPE row at tcs + t rd: its position's in rope_compress (r > 0) or rope_window */
    const ColiCudaChainOps *o = cc_ops();
    int ok = o->matmul(v41u_t8(&l->wq_a), ch->nrm, 0, ch->qa, 0, S) &&
             v41u_norm(ch, ch->qa, 0, QL, ch->o_qn[i], ch->qr, 0, QL, S, QL, c->norm_eps) &&
             o->matmul(v41u_t8(&l->wq_b), ch->qr, 0, ch->q, 0, S) &&
             o->matmul(v41u_t8(&l->wkv), ch->nrm, 0, ch->kv, 0, S) &&
             v41u_norm(ch, ch->kv, 0, hd, ch->o_kn[i], ch->kv, 0, hd, S, hd, c->norm_eps) &&
             v41u_rope(ch, m, ch->q, S * nh, nh, hd - rd, qrow, hd, tcs, rd, 0) &&
             v41u_rope(ch, m, ch->kv, S, 1, hd - rd, hd, 0, tcs, rd, 0);
    if (!ok) return 0;
    CcRegion rg[V41U_ROWS_MAX];   /* each row into its slot's window ring */
    for (int t = 0; t < S; t++) rg[t] = (CcRegion){v41u_wo(ch, slots[t]) + (size_t)(pos[t] % ch->Wd) * hd, (size_t)t * hd, (size_t)hd};
    ok = o->copy_regions(ch->win[i], ch->kv, rg, S);
    if (ok && r > 0 && c->kv_source[i]) {   /* the compressor, its latents, the index keys: a slot at a time */
        const int ihd = c->index_head_dim;
        CcBuf *cb = ch->ckv[i];
        ok = o->matmul(v41u_tb(&l->comp_wkv), ch->nrm, 0, ch->craw, 0, S) &&
             (r == 1 || o->matmul(v41u_tb(&l->comp_wgate), ch->nrm, 0, ch->cscr, 0, S));
        for (int t = 0; ok && t < S; t++) {
            const int sl = slots[t], pb = pos[t], g0 = pb / r, np = (pb + 1) / r - g0;
            const size_t cso = v41u_co(ch, i, sl), iso = v41u_io(ch, i, sl), co = cso + (size_t)g0 * hd, io = iso + (size_t)g0 * ihd;
            if (r == 1) ok = v41u_norm(ch, ch->craw, (size_t)t * hd, hd, ch->o_cn[i], cb, co, hd, 1, hd, c->norm_eps);
            else {
                CcDsComp cp = {1, pb, r, hd, hd, 0, t * hd, hd, t * hd, hd, -1, (int)cso, hd, (int)v41u_ro(ch, m, i, sl)};
                ok = cc_ds_compress(ch->craw, ch->cscr, ch->ring[i], NULL, cb, &cp) &&
                     (np == 0 || v41u_norm(ch, cb, co, hd, ch->o_cn[i], cb, co, hd, np, hd, c->norm_eps));
            }
            if (ok && np > 0) {   /* a latent rotates at its group's first position: row t's block of rmax rows */
                const int bc = 2 * S * rd + t * rmax * rd + (g0 * r - (pb - rmax + 1)) * rd;
                ok = o->matmul(v41u_tb(&l->idx_wk), cb, co, ch->ikey[i], io, np) &&
                     v41u_norm(ch, ch->ikey[i], io, ihd, ch->o_ikn[i], ch->ikey[i], io, ihd, np, ihd, c->norm_eps) &&
                     v41u_rope(ch, m, ch->ikey[i], np, 1, (int)io + ihd - rd, ihd, 0, bc, r * rd, 0) &&
                     v41u_rope(ch, m, cb, np, 1, (int)co + hd - rd, hd, 0, bc, r * rd, 0);
            }
        }
    }
    if (ok && r > 0 && c->index_source[i]) {   /* the indexer: the queries over the S rows, each row's selection on its own */
        const int IH = c->index_n_heads, ID = c->index_head_dim, cs = c->candidate_source;
        const int cand = v41u_cand_on(c), mrow = cand && cs < i ? (int)ch->wcap : 0;
        const float wscale = (1.0f / sqrtf((float)ID)) * (1.0f / sqrtf((float)IH));
        ok = o->matmul(v41u_t8(&l->idx_wq_b), ch->qr, 0, ch->iq, 0, S) &&
             v41u_rope(ch, m, ch->iq, S * IH, IH, ID - rd, IH * ID, ID, tcs, rd, 0) &&
             o->matmul(v41u_tb(&l->idx_wproj), ch->nrm, 0, ch->ihw, 0, S);
        /* the scores in isc's row 0, a row at a time (the queue runs them in order; cand and topk read row 0) */
        for (int t = 0; ok && t < S; t++) {
            const int sl = slots[t], pb = pos[t], width = (pb + 1) / r, kl = keyl[(size_t)t * L + i];
            CcDsScore sp = {1, pb, r, IH, ID, width, t * IH * ID, IH * ID, t * IH, IH, (int)v41u_io(ch, kl, sl), ID, mrow,
                            (int)ch->wcap, wscale, 0, t * mrow};
            ok = cc_ds_score(ch->iq, ch->ihw, v41u_keys(ch, i, kl), mrow ? ch->lw_mask : NULL, ch->isc, &sp);
            if (ok && cand && cs == i) {
                CcDsCand cp = {1, pb, r, width, c->candidate_block_size, c->candidate_topk_blocks, (int)ch->wcap, (int)ch->wcap,
                               t * (int)ch->wcap};
                ok = cc_ds_cand(ch->isc, ch->lw_mask, &cp);
            }
            CcDsTopk tp = {1, width, ch->K, (int)ch->wcap, t * LR + W, LR, ch->Wd, 0};
            ok = ok && cc_ds_topk(ch->isc, ch->lw_list, &tp);
        }
    }
    int done = 0;
    if (ok && S > 1) {   /* the rows' attention in one call where the backend has it (cc_ds_attn_rows), each on its slot's regions */
        int wo[V41U_ROWS_MAX], co[V41U_ROWS_MAX];
        for (int t = 0; t < S; t++) {
            wo[t] = (int)v41u_wo(ch, slots[t]);
            co[t] = r > 0 ? (int)v41u_co(ch, c->index_owner[i], slots[t]) : 0;
        }
        CcDsAttn a = {1, nh, hd, W + (r > 0 ? ch->K : 0), 0, LR, ch->Wd, wo[0], co[0], 0, qrow, 0, qrow, (int)ch->o_sink[i], 0,
                      1.0f / sqrtf((float)hd)};
        const int ra = cc_ds_attn_rows(ch->q, ch->win[i], r > 0 ? ch->ckv[c->index_owner[i]] : NULL, ch->lw_list, ch->prm, ch->heads, &a, wo, co, S);
        if (ra == 0) ok = 0;
        done = ra == 1;
    }
    for (int t = 0; ok && !done && t < S; t++) {   /* the sparse attention: a slot's row on that slot's regions */
        const int sl = slots[t];
        CcDsAttn a = {1, nh, hd, W + (r > 0 ? ch->K : 0), t * LR, LR, ch->Wd, (int)v41u_wo(ch, sl),
                      r > 0 ? (int)v41u_co(ch, c->index_owner[i], sl) : 0, t * qrow, qrow, t * qrow, qrow, (int)ch->o_sink[i], 0,
                      1.0f / sqrtf((float)hd)};
        ok = cc_ds_attn(ch->q, ch->win[i], r > 0 ? ch->ckv[c->index_owner[i]] : NULL, ch->lw_list, ch->prm, ch->heads, &a);
    }
    return ok && v41u_rope(ch, m, ch->heads, S * nh, nh, hd - rd, qrow, hd, tcs, rd, 1) &&
           v41u_wo_a(ch, m, l, S) && o->matmul(v41u_t8(&l->wo_b), ch->grp, 0, ch->br, 0, S);
}
static int v41u_forward_rows(Model *m, const int *slots, const int *pos, int S, float *h, float *pre_mix, int apply, int *lost) {
    *lost = 0;
    Cfg *c = &m->c;
    const int L = c->n_layers, D = c->dim, H = c->hc_mult, HD = H * D, hr = 2 * H + H * H, hd = c->head_dim, rd = c->rope_dim;
    if (S < 1 || S > V41U_ROWS_MAX || m->spec.active || !v41u_wanted(m, 1) || !v41u_setup(m)) return 0;
    V41U *ch = g_v41u;
    for (int t = 0; t < S; t++) {
        if (slots[t] < 0 || slots[t] >= ch->SL || pos[t] < 0) return 0;
        for (int u = 0; u < t; u++) if (slots[u] == slots[t]) return 0;
    }
    const double t0 = now_s();
    const int W = ch->W, LR = W + ch->K, rmax = ch->rmax, cand = v41u_cand_on(c);
    const size_t ncs = (size_t)S * (2 + rmax) * rd;
    int maxE = 1;
    for (int t = 0; t < S; t++) if (pos[t] + 1 > maxE) maxE = pos[t] + 1;
    Engram *eg = &m->engram;
    const int ew = eg->active ? eg->cols * eg->head_dim : 0;
    int neng = 0;
    for (int i = 0; i < L; i++) neng += m->L[i].engram_index >= 0;
    int *keyl = (int *)malloc((size_t)S * L * sizeof(int)), *need = (int *)malloc((size_t)2 * S * L * sizeof(int));
    int *needm = need && ch->split ? need + (size_t)S * L : NULL;   /* V41_CHAIN_SPLIT: the mirrors' rows, a row's at t * L */
    int *wl = (int *)malloc((size_t)S * LR * sizeof(int)), *eng = (int *)malloc((size_t)L * sizeof(int));
    size_t *pc = (size_t *)malloc((size_t)S * L * sizeof(size_t));
    float *cs1 = (float *)malloc((size_t)(1 + rmax) * rd * sizeof(float)), *csa = (float *)malloc(ncs * sizeof(float));
    float *er = neng && ew ? (float *)malloc((size_t)neng * S * ew * sizeof(float)) : NULL;
    float *hp0 = (float *)calloc((size_t)S * hr, sizeof(float));
    int last_pub[V41U_ROWS_MAX];
    int ok = keyl && need && wl && eng && pc && cs1 && csa && hp0 && (!neng || er);
    for (int i = 0, j = 0; ok && i < L; i++) eng[i] = m->L[i].engram_index >= 0 ? j++ : -1;
    for (int t = 0; hp0 && t < S; t++) hp0[(size_t)t * hr] = 1.0f;   /* the first site collapses with [1, 0, ...] */
    /* each conversation's plan, RoPE rows, list and engram rows, with it bound (its host state) */
    size_t pn = 0;
    for (int t = 0; ok && t < S; t++) {
        v41_mux_bind(m, slots[t]);
        v41u_lower(m, pos[t]);   /* this step writes the slot from pos[t] */
        ok = v41u_plan_keys(ch, m, pos[t], 1, keyl + (size_t)t * L, need + (size_t)t * L, needm ? needm + (size_t)t * L : NULL, &last_pub[t]);
        if (!ok) break;
        v41u_chunk_tables(ch, m, pos[t], pos[t], 1, rd, cs1, wl + (size_t)t * LR);
        memcpy(csa + (size_t)t * rd, cs1, (size_t)rd * sizeof(float));
        memcpy(csa + (size_t)(S + t) * rd, cs1 + (size_t)rmax * rd, (size_t)rd * sizeof(float));   /* rope_compress[pos]: cs1's last row */
        memcpy(csa + (size_t)2 * S * rd + (size_t)t * rmax * rd, cs1 + rd, (size_t)rmax * rd * sizeof(float));
        for (int i = 0; i < L; i++) if (eng[i] >= 0) v41u_engram_host(m, i, 1, pos[t], er + ((size_t)eng[i] * S + t) * ew);
        pn += v41u_pull_plan(ch, m, pos[t], 1, pn, pc + (size_t)t * L);
    }
    /* the step's buffers, between frames (a reserve that grows drops what the buffer held) */
    for (int d = 0; ok && d < ch->nd; d++) {   /* on each device (V41_CHAIN_SPLIT: two) */
        v41u_use(ch, d);
        ok = v41u_scratch(ch, m, S, maxE) && v41u_res(&ch->cs, ncs, CC_DEV) && v41u_res(&ch->pull, pn, CC_DOWN) &&
             v41u_res(&ch->lw_list, (size_t)S * LR, CC_DEV) && (!cand || v41u_res(&ch->lw_mask, ((size_t)S * ch->wcap + 3) / 4, CC_DEV));
    }
    v41u_use(ch, 0);
    if (ok && ch->lw_cap < (size_t)S) {
        float *a = (float *)realloc(ch->lw_in, (size_t)S * D * sizeof(float));
        if (a) ch->lw_in = a;
        float *b = a ? (float *)realloc(ch->lw_out, (size_t)S * D * sizeof(float)) : NULL;
        if (b) ch->lw_out = b;
        ok = a && b;
        if (ok) ch->lw_cap = (size_t)S;
    }
    int cmoe = 0;
#if V41U_CMOE
    int cm_ids[V41U_ROWS_MAX * CC_ROUTE_KMAX];
    float cm_w[V41U_ROWS_MAX * CC_ROUTE_KMAX];
    cmoe = ok && v41u_cmoe_setup(ch, m);
    if (cmoe) {
        const int E = ch->cm_E, K = ch->cm_K;
        for (int d = 0; ok && d < ch->nd; d++) {
            v41u_use(ch, d);
            ok = v41u_res(&ch->cm_lg, (size_t)S * E, CC_DEV) && cc_reserve(&ch->cm_ids, (size_t)S * K * sizeof(int), CC_DOWN) &&
                 v41u_res(&ch->cm_w, (size_t)S * K, CC_DOWN) && v41u_res(&ch->cm_hid, cc_moe_hid_floats(S, K, D, c->moe_inter, E), CC_DEV) &&
                 v41u_res(&ch->cm_y, (size_t)S * D, CC_DEV) && v41u_res(&ch->cm_hy, (size_t)S * D, CC_UP) &&
                 v41u_res(&ch->cm_yd, (size_t)S * D, CC_DOWN);
        }
        v41u_use(ch, 0);
        if (ok && !ch->rw_contrib) ok = (ch->rw_contrib = (float *)malloc((size_t)V41U_ROWS_MAX * CC_ROUTE_KMAX * D * sizeof(float))) != NULL;
    }
#endif
    /* the frame that brings each slot's state up as far as the device does not hold it, the lists, the
     * RoPE rows, the streams and the first collapse's mixes */
    for (int d = 0; ok && d < ch->nd; d++) {   /* each device its layers' state, the lists and RoPE rows; the streams to the first */
        v41u_use(ch, d);
        ok = cc_begin();
        for (int t = 0; ok && t < S; t++) {
            v41_mux_bind(m, slots[t]);
            ok = v41u_state_up(ch, m, pos[t], need + (size_t)t * L, needm ? needm + (size_t)t * L : NULL);
        }
        ok = ok && cc_write(ch->lw_list, 0, wl, (size_t)S * LR * sizeof(int)) && cc_write(ch->cs, 0, csa, ncs * sizeof(float)) &&
             (d || (cc_write(ch->xs, 0, h, (size_t)S * HD * sizeof(float)) && cc_write(ch->hpf, 0, hp0, (size_t)S * hr * sizeof(float))));
        ok = v41u_close(ok);
    }
    v41u_use(ch, 0);
    double dev_ms = 0.0, host_ms = 0.0, p2_ms = 0.0;
    for (int i = 0; ok && i < L; i++) {
        Layer *l = &m->L[i];
        const double ta = now_s();
        if (ch->split && i == ch->split) {   /* the primary's layers are through: their state back (apply), the rows' streams and mixes to the second device */
            ok = cc_begin();
            ok = ok && cc_copy(ch->xd, 0, ch->xs, 0, (size_t)S * HD) && cc_copy(ch->hpd, 0, ch->hpf, 0, (size_t)S * hr);
            for (int t = 0; ok && apply && t < S; t++) {
                v41_mux_bind(m, slots[t]);
                ok = v41u_state_pull(ch, m, pos[t], pos[t] + 1, pc + (size_t)t * L, (size_t)hd);
            }
            ok = v41u_close(ok);
            const float *x0 = (const float *)cc_ptr(ch->xd), *p0 = (const float *)cc_ptr(ch->hpd);
            v41u_use(ch, 1);
            ok = ok && cc_begin();
            ok = ok && cc_write(ch->xs, 0, x0, (size_t)S * HD * sizeof(float)) && cc_write(ch->hpf, 0, p0, (size_t)S * hr * sizeof(float));
            ok = v41u_close(ok);
            ch->sp_steps++; ch->sp_ms += (now_s() - ta) * 1e3;
            if (!ok) break;
        }
        XT(stream_prefetch)(i, S);
#if V41U_CMOE
        if (cmoe && !v41u_cmoe_tables(ch, m, 0)) {   /* the tier's residents moved: this layer's frame reads the new tables */
            if (cc_lost()) { ok = 0; break; }
            v41u_cmoe_off(ch, "the tier's tables refused");
            cmoe = 0;
        }
#endif
        const int opened = cc_begin();
        ok = opened;
        if (ok && eng[i] >= 0) ok = v41u_engram_dev(ch, m, i, S, er + (size_t)eng[i] * S * ew);
        ok = ok && v41u_pre(ch, m, ch->fna[i], ch->o_hca[i], ch->hpa, ch->hpf, ch->o_an[i], S) &&
             v41u_attention_rows(ch, m, i, S, slots, pos, keyl) && v41u_post(ch, m, ch->hpa, S) &&
             v41u_pre(ch, m, ch->fnf[i], ch->o_hcf[i], ch->hpf, ch->hpa, ch->o_fn[i], S) &&
             cc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)S * D);
        int shared_done = 0;
#if V41U_CMOE
        if (ok && cmoe) {
            int r;
            if (v41u_shared_late()) {   /* the routes and their word first: the shared expert and the residents run beside the host's misses */
                r = v41u_cmoe_route_s(ch, m, i, S);
                if (r) { shared_done = 1; ok = v41u_shared(ch, m, l, S); r = ok && v41u_cmoe_res_s(ch, m, i, S); }
            } else { shared_done = 1; ok = v41u_shared(ch, m, l, S); r = ok && v41u_cmoe_dev_s(ch, m, i, S); }
            if (ok && !r) {   /* refused (not a lost device): the host's routed experts from here on */
                if (cc_lost()) ok = 0;
                else { v41u_cmoe_off(ch, "the device's router or experts refused (cc_matmul/cc_route/cc_moe parts 1, S rows)"); cmoe = 0; }
            }
        }
#endif
        if (ok && !shared_done) ok = v41u_shared(ch, m, l, S);
#if V41U_CMOE
        if (ok && cmoe) {   /* the frame left running: its word says the routes (and h2d, copied before them) are down */
            ok = cc_submit(0);
            if (ok) {
                const double tw = now_s();
                int rd = 0;
                for (unsigned long spin = 0; (rd = v41u_routes_down(ch)) == 0; spin++)
                    if ((spin & 0xFFFFu) == 0 && (cc_lost() || now_s() - tw > 60.0)) {
                        fprintf(stderr, "[CUDA] deepseek_v41 chain rows: the device did not reach layer %d's routes in 60 s\n", i);
                        ok = 0; break;
                    }
                if (rd < 0) ok = 0;
                ch->rw_wait_ms += (now_s() - tw) * 1e3;
            }
        } else
#endif
        if (opened) ok = v41u_close(ok);
        if (ok) memcpy(ch->lw_in, cc_ptr(ch->h2d), (size_t)S * D * sizeof(float));
        m->t_attn += now_s() - ta;
        dev_ms += (now_s() - ta) * 1e3;
        if (!ok) break;
        const double th = now_s();
#if V41U_CMOE
        if (cmoe) v41u_cmoe_host_rows(ch, m, i, S, ch->lw_in, cm_ids, cm_w);
        else
#endif
        moe_run_at(m, l, &m->cache[i], "layers", i, c->n_routed, c->n_activated, ch->lw_in, S, ch->lw_out, 0);
        host_ms += (now_s() - th) * 1e3;
        const double tp = now_s();
        CcBuf *rin = ch->routed;   /* the routed rows the add reads: the host's, or the device's y */
        CcEw add = {CC_EW_ADD, S * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        int open = 0, tailed = 0;
#if V41U_CMOE
        if (cmoe) {   /* y = hy + the resident slots' w * down in rank order (cc_moe parts 2, S rows), then the add from y */
            CcMoe p2 = v41u_cmoe_args_s(ch, m, i, 2, S);
            ok = open = cc_begin();
            if (ok && v41u_tail_on() && v41u_tail(ch, m, &p2, ch->hpf, S)) { tailed = 1; rin = ch->cm_y; }   /* that, the add and the write back in one launch */
            else if (ok && cc_moe(ch->cm_tab, ch->nrm, ch->cm_ids, ch->cm_w, ch->cm_hid, ch->cm_y, ch->cm_hy, &p2)) rin = ch->cm_y;
            else if (ok) {   /* refused: this layer's routed rows from moe_run_at after all, the host step from here on */
                ok = !cc_lost() && cc_submit(1);
                open = 0;
                if (ok) {
                    v41u_cmoe_off(ch, "the device's experts refused (cc_moe parts 2, S rows)");
                    cmoe = 0;
                    moe_run_at(m, l, &m->cache[i], "layers", i, c->n_routed, c->n_activated, ch->lw_in, S, ch->lw_out, 0);
                }
            }
        }
#endif
        if (ok && rin == ch->routed) {
            memcpy(cc_ptr(ch->routed), ch->lw_out, (size_t)S * D * sizeof(float));
            ok = open = cc_begin();
        }
        if (!tailed) ok = ok && cc_ops()->ew(ch->br, rin, ch->ds, NULL, NULL, &add) && v41u_post(ch, m, ch->hpf, S);
        if (open) ok = v41u_close(ok);
        p2_ms += (now_s() - tp) * 1e3;
#if V41U_CMOE
        if (ok && rin == ch->cm_y) {   /* the device's books for each row (vkt_dev_count), as a one-row step's */
            const int E = ch->cm_E, K = ch->cm_K;
            for (int t = 0; t < S; t++) {
                int valid = 1;
                for (int k = 0; k < K; k++) valid &= cm_ids[t * K + k] >= 0 && cm_ids[t * K + k] < E;
                if (valid) XT(dev_count)(i, 1, K, cm_ids + t * K);
            }
        }
#endif
        if (ok) ch->rw_layers++;
    }
    (void)cmoe;
    /* the streams and mixes back, and (apply) each slot's state */
    if (ok) {
        const double tz = now_s();
        ok = cc_begin();
        ok = ok && cc_copy(ch->xd, 0, ch->xs, 0, (size_t)S * HD) && cc_copy(ch->hpd, 0, ch->hpf, 0, (size_t)S * hr);
        for (int t = 0; ok && apply && t < S; t++) {
            v41_mux_bind(m, slots[t]);
            ok = v41u_state_pull(ch, m, pos[t], pos[t] + 1, pc + (size_t)t * L, (size_t)hd);
        }
        ok = v41u_close(ok);
        dev_ms += (now_s() - tz) * 1e3;
    }
    if (ok) {
        const float *xd = (const float *)cc_ptr(ch->xd), *hpd = (const float *)cc_ptr(ch->hpd);
        for (int t = 0; t < S; t++) {
            memcpy(h + (size_t)t * HD, xd + (size_t)t * HD, (size_t)HD * sizeof(float));
            memcpy(pre_mix + (size_t)t * H, hpd + (size_t)t * hr, (size_t)H * sizeof(float));
        }
        for (int t = 0; apply && t < S; t++) {   /* the window row (pw: a row's), compressed rows, keys and group of each */
            v41_mux_bind(m, slots[t]);
            v41u_state_apply(ch, m, pos[t], 1, pc + (size_t)t * L, (size_t)hd, last_pub[t]);
        }
    }
    free(keyl); free(need); free(wl); free(eng); free(pc); free(cs1); free(csa); free(er); free(hp0);
    v41u_use(ch, 0);
    if (!ok) {   /* the slots' device copies from their positions are stale; the host's state is as it was */
        for (int t = 0; t < S; t++) { v41_mux_bind(m, slots[t]); v41u_lower(m, pos[t]); v41u_mirror_reset(ch, m); }
        if (v41u_any_lost(ch)) {
            *lost = 1; g_v41u_off = 1; ch->ok = 0;
            fprintf(stderr, "[CUDA] deepseek_v41 chain rows: a frame failed; the CPU runs the layers from now on\n");
        }
        return 0;
    }
    const double tot = (now_s() - t0) * 1e3;
    ch->rw_steps++; ch->rw_rows += (unsigned long long)S;
    ch->rw_dev_ms += dev_ms; ch->rw_host_ms += host_ms; ch->rw_p2_ms += p2_ms;
    ch->rw_book_ms += tot - dev_ms - host_ms - p2_ms; ch->rw_fwd_ms += tot;
    if (ch->rw_steps == 1)
        fprintf(stderr, "[CUDA] deepseek_v41 chain rows (V41_CHAIN_ROWS): a decode step's %d conversations' rows in one frame a layer, "
                        "their attention a slot at a time%s\n", S, cmoe ? "; the routed experts in the chain" : "");
    return 1;
}
/* forward_rows' books: the heads' time of a batched step; a step that fell back (said once); a checked
 * step's comparison of each row's logits (lg: the batched step's, ref: a row at a time's) */
static void v41u_rows_head(double ms) {
    if (g_v41u) { g_v41u->rw_head_ms += ms; g_v41u->rw_fwd_ms += ms; }
}
static void v41u_rows_fallback(int lost) {
    static int said;
    if (g_v41u) g_v41u->rw_fallbacks++;
    if (!said++) fprintf(stderr, "[CUDA] deepseek_v41 chain rows (V41_CHAIN_ROWS): a batched step %s; that step a row at a time%s\n",
                         lost ? "lost the device" : "was not taken or refused", lost ? " (the CPU from now on)" : "");
}
static void v41u_rows_compare(const float *lg, const float *ref, int n, int V) {
    V41U *ch = g_v41u;
    if (!ch) return;
    ch->rw_checked_steps++;
    for (int t = 0; t < n; t++) {
        double mx = 0.0, md = 0.0;
        int a0 = 0, a1 = 0;
        for (int v = 0; v < V; v++) {
            const double r = fabs((double)ref[(size_t)t * V + v]), df = fabs((double)ref[(size_t)t * V + v] - (double)lg[(size_t)t * V + v]);
            if (r > mx) mx = r;
            if (df > md || df != df) md = df;
            if (lg[(size_t)t * V + v] > lg[(size_t)t * V + a0]) a0 = v;
            if (ref[(size_t)t * V + v] > ref[(size_t)t * V + a1]) a1 = v;
        }
        const double rel = mx > 0.0 ? md / mx : md;
        if (rel > ch->rw_err_max || rel != rel) ch->rw_err_max = rel;
        ch->rw_err_sum += rel;
        ch->rw_checked_rows++;
        if (ch->rw_checked_steps <= 4 || rel > 1e-3 || a0 != a1)
            fprintf(stderr, "[CUDA] deepseek_v41 chain rows check: step %llu row %d of %d, logits max rel err %.3g%s\n", ch->rw_checked_steps,
                    t, n, rel, a0 != a1 ? " (the argmax differs)" : "");
    }
}
#endif
