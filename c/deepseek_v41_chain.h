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
#include "dsv41_kvq.h"

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
    VkcBuf *qck, *qik;                         /* the packed cache's chunks: new rows staged fp32 before their pack */
    int kvq, kvq_checked;                      /* the mirrors quantized (COLI_VK_KV_FP32 unset), the widths checked once */
    int lo, d;                                 /* its layers start at lo, on device d (vkc_device) */
    int n;                                     /* the layers on the device: lo..lo+n-1 (a partial chain) */
    VkcBuf *cmd, *ld;                          /* the handoff: the candidate mask and the index list, down */
    VkcBuf *lw_xs, *lw_hp, *lw_ds;             /* a prompt layer by layer: every row's streams, FFN mix, shared expert, */
    VkcBuf *lw_list, *lw_mask;                 /* list (the selection an index source left for the layers after it) and candidate mask */
    float *lw_in, *lw_out;                     /* and the routed experts' rows, in and out (host) */
    size_t lw_cap;
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

/* COLI_VK_KV_FP32: the packed cache's ONE flag, parsed once here at the setup's start.
 * Unset (or 0) is the default: the device KV mirrors in the checkpoint's quantized formats
 * (c/dsv41_kvq.h, DeepInfer's cache) instead of fp32 -- approximate (a row quantized at its
 * store, dequantized where read), and they give way to the KV split: when the split arms
 * the mirrors are fp32 again -- AND the HOST's own arrays in the same formats (v41_host_kvq,
 * deepseek_v41.c: one condition now, the codes on the host, dequantized on read there). The
 * host layout never gives way to the split, only these mirrors do. 1 (or true) is the exact
 * fp32 opt-out, byte for byte the pre-default path, and it takes the host with it. There is
 * no in-between setting: the flag's old =1 held the host fp32 while the mirrors packed,
 * which measured byte-identical to the packed host on the 12-prompt battery (and at 16K
 * only the packed host was ever needed) while paying the dequant->re-pack round trip the
 * ckv subnormal-scale fix (5c4eb903) had to repair. The geometry guard applies to the
 * default as to an explicit 0: at any other head_dim/index_head_dim the packed cache
 * declines (one line) and BOTH sides stay fp32, so non-512/128 models are untouched. The
 * three row widths below are the ONLY widths the mirrors are ever allocated at, so a
 * missed branch shows up as bytes that do not add up (v41c_kvq_check, once a forward). */
static int g_v41c_kvq = -1;
static int g_v41c_kvq_unset = 1;
static int v41c_kvq_on(const Cfg *c) {
    if (g_v41c_kvq < 0) {
        const char *e = getenv("COLI_VK_KV_FP32");
        int fp32 = e && *e && *e != '0' && strcmp(e, "false") && strcmp(e, "off") && strcmp(e, "no");
        int ok = !fp32 && c->head_dim == 512 && c->index_head_dim == 128;
        g_v41c_kvq_unset = !(e && *e);
        g_v41c_kvq = ok;                 /* the setup print and the host read both ask */
        if (!fp32 && !ok)
            fprintf(stderr, "[VK] deepseek_v41 chain: COLI_VK_KV_FP32%s wants head_dim 512 and index_head_dim 128; "
                            "the mirrors stay fp32\n", g_v41c_kvq_unset ? " (the packed default)" : "=0");
        /* the retired COLI_VK_KV_QUANT's one warning is v41_host_kvq's (deepseek_v41.c): the
         * host parses at every model_load, so it has said so before this runs */
    }
    return g_v41c_kvq;
}
static size_t v41c_win_row(const V41Chain *ch, const Cfg *c) { return ch->kvq ? KVQ_WIN_BYTES : (size_t)c->head_dim * sizeof(float); }
static size_t v41c_ckv_row(const V41Chain *ch, const Cfg *c) { return ch->kvq ? KVQ_CKV_BYTES : (size_t)c->head_dim * sizeof(float); }
static size_t v41c_ikey_row(const V41Chain *ch, const Cfg *c) { return ch->kvq ? KVQ_IKEY_BYTES : (size_t)c->index_head_dim * sizeof(float); }

/* The window ring's extra rows at setup: COLI_VK_CHAIN_ROWS when set, else 512; a forward
 * whose chunk is bigger grows the rings first (v41c_grow_win). */
static int v41c_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e && strcmp(e, "auto") ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}
/* v41c_res counts instead of reserving while g_v41c_count >= 0 (the chunk's sizing) */
static long long g_v41c_count = -1;
/* the quadratic cap on the chunk (v41c_forward_seg) said its line once */
static int g_v41c_cap_said;
static int v41c_res(VkcBuf **b, size_t floats, int kind) {
    if (g_v41c_count >= 0) { g_v41c_count += (long long)(floats ? floats : 1) * (long long)sizeof(float); return 1; }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}
/* The candidate mask's words a row: one bit a column (vk_chain.h, vkc_dsv4_cand), read
 * back by the host and by the index sources after the candidate one; the scores stay
 * fp32. `wcap` columns a row is the forward's end position, so the pair is quadratic in
 * the prompt and the mask was most of it (4 bytes a column). */
static size_t v41c_mask_words(size_t wcap) { return (wcap + 31) / 32; }

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
/* A GGUF's matrix (a W8 or WB the loader read as ggml rows of type gt, interleaved by 16
 * where the type has that form) on the device, as deepseek_v4's chain takes them: its
 * blocks as the file has them (100 + the type), a float type as floats (f16 as fmt 14).
 * For the per-head reader (vkc_mla_hgemv, which has no block form: wo_a) Q8_0 goes up as
 * int8 with a scale per 32 (fmt 13, the same numbers) and any other type as floats.
 * -1: no device form. */
static int v41c_gq_fmt(int gt, int I, int hgemv) {
    const int b = gq_r16_base(gt), base = b >= 0 ? b : gt;
    if (base == GQ_F32) return 10;
    if (base == GQ_BF16) return 11;
    if (base == GQ_F16) return hgemv ? 10 : 14;
    if (hgemv) return base == GQ_Q8_0 && I % 32 == 0 ? 13 : 10;
    const int be = coli_vk_block_elems(100 + base);
    return be > 0 && I % be == 0 ? 100 + base : -1;
}
static ColiVkTensor *v41c_gq(const uint8_t *data, int gt, int O, int I, int hgemv) {
    const int fmt = v41c_gq_fmt(gt, I, hgemv);
    if (!data || fmt < 0 || O < 1 || I < 1) return NULL;
    const int b = gq_r16_base(gt), base = b >= 0 ? b : gt;
    const int64_t rb = gq_row_bytes(base, I);   /* a file row */
    if (rb <= 0 || (b >= 0 && O % 16)) return NULL;
    ColiVkTensor **slot = NULL;
    VkEntry *e = NULL;
    if (vkc_device_now()) {
        if (!(slot = v41c_d2_slot(data))) return NULL;
        if (*slot) return *slot;
    } else {
        if (!(e = vk_entry(data, fmt, O, I)) || e->refused) return NULL;
        if (e->t) return e->t;
    }
    const uint8_t *src = data;
    uint8_t *file = NULL;   /* the file's rows of an interleaved matrix */
    void *conv = NULL;      /* fmt 13's codes or fmt 10's floats */
    float *sc = NULL;
    int gs = 0, ok = 1;
    if (b >= 0) {
        if (!(file = malloc((size_t)rb * O))) ok = 0;
        else { gq_r16_unpack(gt, file, data, O, I); src = file; }
    }
    const void *weights = src;
    if (ok && fmt == 13) {   /* Q8_0 (a half scale, 32 int8 codes a block): codes and scales apart */
        const int nb = I / 32;
        int8_t *q = malloc((size_t)O * I);
        sc = malloc((size_t)O * nb * sizeof(float));
        if (!q || !sc) { free(q); ok = 0; }
        else {
            for (int r = 0; r < O; r++)
                for (int k = 0; k < nb; k++) {
                    const uint8_t *blk = src + (size_t)r * rb + (size_t)k * 34;
                    uint16_t h; memcpy(&h, blk, 2);
                    sc[(size_t)r * nb + k] = gq_f16(h);
                    memcpy(q + (size_t)r * I + (size_t)k * 32, blk + 2, 32);
                }
            weights = conv = q; gs = 32;
        }
    } else if (ok && fmt == 10 && base != GQ_F32) {   /* the per-head reader's floats */
        float *f = malloc((size_t)O * I * sizeof(float));
        if (!f) ok = 0;
        else {
            for (int r = 0; r < O; r++) {
                const uint8_t *row = src + (size_t)r * rb;
                float *o = f + (size_t)r * I;
                if (base == GQ_F16) for (int c = 0; c < I; c++) { uint16_t h; memcpy(&h, row + 2 * c, 2); o[c] = gq_f16(h); }
                else if (base == GQ_BF16) for (int c = 0; c < I; c++) { uint16_t h; memcpy(&h, row + 2 * c, 2); uint32_t u = (uint32_t)h << 16; memcpy(&o[c], &u, 4); }
                else gq_dequant_row(base, row, o, I);
            }
            weights = conv = f;
        }
    }
    if (ok) ok = slot ? coli_vk_tensor_ensure2(slot, weights, sc, fmt, I, O, gs)
                      : coli_vk_tensor_ensure(&e->t, weights, sc, fmt, I, O, gs);
    free(file); free(conv); free(sc);
    if (slot) return ok ? *slot : NULL;
    if (!ok && !e->t) e->refused = 1;
    return ok ? e->t : NULL;
}
/* The device copies the per-matrix path keeps (vk_entry), uploaded here if it has not:
 * fp8 as fmt 12 with each 32x32 tile's scale written out for its rows, bf16 as fmt 11.
 * With the second device current, its own copies there. A GGUF's: v41c_gq. */
static ColiVkTensor *v41c_w8(const W8 *w) {
    if (w->gx) return v41c_gq(w->q, w->gt, w->O, w->I, 0);
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
    if (w->gx) return v41c_gq(w->g, w->gt, w->O, w->I, 0);
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

/* wo_a for the per-head reader (vkc_mla_hgemv): a GGUF's in the reader's form */
static ColiVkTensor *v41c_w8h(const W8 *w) { return w->gx ? v41c_gq(w->q, w->gt, w->O, w->I, 1) : v41c_w8(w); }
/* A GGUF's wo_a as its output groups' GEMMs, each group's rows in the file's blocks (as wo_b
 * goes up), not the per-head reader's floats: the reader takes every row of a prompt chunk
 * through the whole matrix again (8192 x 4096 floats a layer of V4.1-Flash, 128 MiB, 40
 * layers 5 GiB of the device), a GEMM reads each tile once for all the chunk's rows.
 * COLI_VK_V41_WOA=0: the per-head reader. */
#define V41C_OG_MAX 64
static int v41c_woa_groups(const W8 *w, int og, int ol) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_V41_WOA"); on = !(e && *e == '0'); }
    return on && w->gx && w->q && og >= 1 && og <= V41C_OG_MAX && ol % 16 == 0 && w->O == og * ol &&
           v41c_gq_fmt(w->gt, w->I, 0) >= 0;
}
static const uint8_t *v41c_woa_rows(const W8 *w, int g, int ol) {   /* group g's first row (16-row interleave kept) */
    const int64_t rb = gq_row_bytes(w->gt, w->I);   /* an interleaved type's: 16 of them step a group */
    return rb > 0 ? w->q + (size_t)g * ol * (size_t)rb : NULL;
}
static ColiVkTensor *v41c_woa_group(const W8 *w, int g, int ol) {
    const uint8_t *q = v41c_woa_rows(w, g, ol);
    return q ? v41c_gq(q, w->gt, ol, w->I, 0) : NULL;
}
static int v41c_woa_up(const W8 *w, int og, int ol) {
    if (!v41c_woa_groups(w, og, ol)) return v41c_w8h(w) != NULL;
    for (int g = 0; g < og; g++) if (!v41c_woa_group(w, g, ol)) return 0;
    return 1;
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
 * the layers are read (a GGUF's forms from its loaded layers: a GGUF has no device-only
 * placement, so the fit runs after the load). */
static size_t v41c_layer_bytes(const Model *m, int i, int rows, int dho, int views, size_t *mat) {
    const Cfg *c = &m->c;
    int D = c->dim, H = c->hc_mult, nm = (2 + H) * H, hd = c->head_dim, nh = c->n_heads, r = c->compress_ratio[i];
    int IH = c->index_n_heads, ID = c->index_head_dim, og = c->o_groups, ol = c->o_lora, W = c->window, QL = c->q_lora;
    int eng = v41c_engram_layer(m, i);
    size_t b = 0, mm = 0;
    #define T(fmt, I, O, gs) do { b += vkc_fit_tensor(fmt, I, O, gs); mm += coli_vk_tensor_payload(fmt, I, O, gs); } while (0)
    /* fp8 (or a GGUF's W8: its form, from the loaded layer, as v41c_gq puts it up) */
    /* l: the loaded layer, a GGUF's only (the checkpoint's fit with the dense weights on
     * the device runs before the layers are read, when m->L is not there yet) */
    #define T8(w, I, O, hg) do { const W8 *w_ = l ? (w) : NULL; \
        if (w_ && w_->gx) { int f_ = v41c_gq_fmt(w_->gt, I, hg); if (f_ >= 0) T(f_, I, O, f_ == 13 ? 32 : 0); } \
        else T(12, I, O, FP8_TILE); } while (0)
    #define TB(w, I, O) do { const WB *w_ = l ? (w) : NULL; \
        if (w_ && w_->gx) { int f_ = v41c_gq_fmt(w_->gt, I, 0); if (f_ >= 0) T(f_, I, O, 0); } \
        else T(11, I, O, 0); } while (0)
    const Layer *l = m->gguf && m->L ? &m->L[i] : NULL;
    T(10, H * D, nm, 0); T(10, H * D, nm, 0);                          /* the mHC mix matrices */
    T8(&l->wq_a, D, QL, 0); T8(&l->wq_b, QL, nh * hd, 0); T8(&l->wkv, D, hd, 0);
    if (l && v41c_woa_groups(&l->wo_a, og, ol)) {                     /* wo_a's groups in the file's blocks */
        int f_ = v41c_gq_fmt(l->wo_a.gt, nh * hd / og, 0);
        for (int g = 0; g < og; g++) T(f_, nh * hd / og, ol, 0);
    } else T8(&l->wo_a, nh * hd / og, og * ol, 1);                     /* wo_a whole, the chain's */
    for (int g = 0; views && g < og; g++) T(12, nh * hd / og, ol, FP8_TILE);   /* and per group, the per-matrix path's */
    T8(&l->wo_b, og * ol, D, 0);
    T8(&l->sh_w1, D, c->moe_inter, 0); T8(&l->sh_w3, D, c->moe_inter, 0); T8(&l->sh_w2, c->moe_inter, D, 0);
    if (c->kv_source[i]) { TB(&l->comp_wkv, D, hd); if (r > 1) TB(&l->comp_wgate, D, hd); TB(&l->idx_wk, hd, ID); }
    if (c->index_source[i] && (r > 0 || dho)) { T8(&l->idx_wq_b, QL, IH * ID, 0); TB(&l->idx_wproj, D, IH); }
    if (eng) T8(&l->eng_wkv, m->engram.cols * m->engram.head_dim, D * (H + 1), 0);
    #undef TB
    #undef T8
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
    if (!ch->win[i] && !(ch->win[i] = vkc_buf((size_t)ch->Wd * v41c_win_row(ch, c), VKC_DEV))) return 0;
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
    int ok = v41c_w8(&l->wq_a) && v41c_w8(&l->wq_b) && v41c_w8(&l->wkv) && v41c_woa_up(&l->wo_a, c->o_groups, c->o_lora) && v41c_w8(&l->wo_b) &&
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
        for (int g = 1; g < m->c.o_groups && v41c_woa_groups(&l->wo_a, m->c.o_groups, m->c.o_lora); g++)
            v41c_d2_forget(v41c_woa_rows(&l->wo_a, g, m->c.o_lora));
        for (size_t k = 0; k < sizeof wb / sizeof *wb; k++) v41c_d2_forget(wb[k]->gx ? (const void *)wb[k]->g : (const void *)wb[k]->w);
        return;
    }
    for (size_t k = 0; k < sizeof w8 / sizeof *w8; k++) if (w8[k]->q) v41_vk_forget_range(w8[k]->q, (size_t)w8[k]->O * w8[k]->I);
    for (size_t k = 0; k < sizeof wb / sizeof *wb; k++) {
        if (wb[k]->w) v41_vk_forget_range(wb[k]->w, (size_t)wb[k]->O * wb[k]->I * 2);
        if (wb[k]->gx && wb[k]->g) v41_vk_forget_range(wb[k]->g, 1);   /* a GGUF's rows: the entry is keyed on their start */
    }
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
    ch->kvq = v41c_kvq_on(c);
    /* THE invariant the single flag rests on: the device mirrors pack only beside a packed
     * host (v41_host_kvq, deepseek_v41.c). The pull's write-back, the re-upload and the
     * dequant-on-read paths all assume the host's rows ARE the codes; a setup where the
     * two disagree would read one layout as the other. */
    if (ch->kvq && !v41_host_kvq(c)) {
        fprintf(stderr, "[VK] %s chain: COLI_VK_KV_FP32 assertion failed: the KV mirrors are packed and the "
                        "host's rows are not -- the one flag cannot pick sides\n", v41c_name(ch));
        abort();
    }
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
    if (ch->kvq) fprintf(stderr, "[VK] %s chain: COLI_VK_KV_FP32%s: the KV mirrors quantized as the checkpoint's cache "
                                 "(a window row 528 B, a compressed row 288 B, an index key 68 B) and the host's own KV "
                                 "rows the cache's codes too (dequantized on read) -- approximate, and the mirrors give "
                                 "way once the KV split arms (the host's rows do not)\n", nmc,
                         g_v41c_kvq_unset ? " unset" : "=0");
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
/* the split armed under the packed cache: the split stages whole fp32 rows, so the
 * quantized mirrors give way -- freed and re-mirrored fp32 from the host at the next
 * forward (the watermarks drop). The host's rows by then hold the codes (the flag's host
 * side never gives way), so they dequantize up into the fp32 mirrors: one
 * quantize->dequant hop, bounded by the pack's idempotence the kv-quant test pins. 0 =
 * memory refused. */
static int v41c_kvq_give_way(V41Chain *ch, Model *m) {
    Cfg *c = &m->c;
    fprintf(stderr, "[VK] %s chain: the KV split armed: the packed cache gives way to it -- the whole mirrors fp32 "
                    "again, dequantized from the host's code rows\n", v41c_name(ch));
    ch->kvq = 0;
    for (int i = 0; i < c->n_layers; i++) {
        vkc_free(ch->ckv[i]); ch->ckv[i] = NULL;
        vkc_free(ch->ikey[i]); ch->ikey[i] = NULL;
        ch->kv_valid[i] = ch->ccap[i] = 0;
    }
    for (int i = ch->lo; i < ch->lo + ch->n; i++) {
        vkc_free(ch->win[i]); ch->win[i] = NULL;
        if (!(ch->win[i] = vkc_buf((size_t)ch->Wd * v41c_win_row(ch, c), VKC_DEV))) return 0;
    }
    ch->win_valid = 0;
    return 1;
}
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
    if (ch->kvq && !v41c_kvq_give_way(ch, m)) return 0;
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
        ch->ckv[i] = vkc_buf((size_t)cap * v41c_ckv_row(ch, c), VKC_DEV);
    }
    vkc_free(ch->ikey[i]);
    ch->ikey[i] = vkc_buf(ch->kvq ? kvq_ikey_mirror_bytes((size_t)cap) : (size_t)cap * v41c_ikey_row(ch, c), VKC_DEV);
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
     * output beside the rank-order contribution buffer. The index list is already in the
     * per-row count above (v41c_scratch reserves r * (W + K) floats a chunk for it), so
     * charging (W + K) * sizeof(int) again here would count it twice. */
    row += 2 * ((size_t)c->dim + c->moe_inter) * sizeof(float);
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
        if (!(ch->win[i] = vkc_buf((size_t)Wd * v41c_win_row(ch, c), VKC_DEV))) return 0;
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
             v41c_res(&ch->isc, r * ch->wcap, VKC_DEV) && v41c_res(&ch->mask, r * v41c_mask_words(ch->wcap), VKC_DEV) &&
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
    if (ok && ch->kvq) {       /* the packed cache: a chunk's new compressed and index rows, fp32 until their pack */
        size_t nq = r + 2;   /* the rows span at most rows/ratio + 1 groups (a ratio-1 layer: the chunk's rows) */
        if (g_v41c_count >= 0)
            g_v41c_count += (long long)(nq * (size_t)hd * sizeof(float)) + (long long)(nq * (size_t)ID * sizeof(float));
        else ok = v41c_res(&ch->qck, nq * hd, VKC_DEV) && v41c_res(&ch->qik, nq * ID, VKC_DEV);   /* an index row is ID floats */
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
/* wo_a: heads [nr][og x pg] -> grp [nr][og x ol]. Its groups' GEMMs (v41c_woa_groups): one
 * row reads its groups' slices in place; more rows go to group-major order first (q, free
 * after the attention), each group's GEMM writes [nr][ol] into heads, and the rows come back
 * in grp's order. Else the per-head reader. */
static int v41c_wo_a(V41Chain *ch, Model *m, Layer *l, int nr, const VkcHgemv *g) {
    const Cfg *c = &m->c; const W8 *w = &l->wo_a;
    const int og = c->o_groups, ol = c->o_lora, pg = w->I;
    if (!v41c_woa_groups(w, og, ol) || og * pg != c->n_heads * c->head_dim)
        return vkc_mla_hgemv(v41c_w8h(w), ch->heads, ch->grp, NULL, g);
    ColiVkTensor *t[V41C_OG_MAX];
    for (int k = 0; k < og; k++) if (!(t[k] = v41c_woa_group(w, k, ol))) return 0;
    int ok = 1;
    if (nr == 1) {
        for (int k = 0; ok && k < og; k++) ok = vkc_matmul(t[k], ch->heads, (size_t)k * pg, ch->grp, (size_t)k * ol, 1);
        return ok;
    }
    static VkcRegion *rg; static int crg;
    const int n = nr * og;
    if (n > crg) {
        VkcRegion *nrg = realloc(rg, (size_t)n * sizeof *rg);
        if (!nrg) return 0;
        rg = nrg; crg = n;
    }
    for (int s = 0; s < nr; s++)
        for (int k = 0; k < og; k++) rg[s * og + k] = (VkcRegion){(size_t)k * nr * pg + (size_t)s * pg, (size_t)s * og * pg + (size_t)k * pg, (size_t)pg};
    ok = vkc_copy_regions(ch->q, ch->heads, rg, n);
    for (int k = 0; ok && k < og; k++) ok = vkc_matmul(t[k], ch->q, (size_t)k * nr * pg, ch->heads, (size_t)k * nr * ol, nr);
    for (int s = 0; s < nr; s++)
        for (int k = 0; k < og; k++) rg[s * og + k] = (VkcRegion){(size_t)s * og * ol + (size_t)k * ol, (size_t)k * nr * ol + (size_t)s * ol, (size_t)ol};
    return ok && vkc_copy_regions(ch->grp, ch->heads, rg, n);
}
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
    if (ch->kvq) {   /* the packed cache: the new rows packed into the ring (dsv41_kvq.h's win format) */
        ok = vkc_dsv4_quant(ch->win[i], ch->kv, &(VkcDsQuant){0, nr, hd, 0, hd, pb, ch->Wd});
    } else {
        VkcRegion *rg = malloc((size_t)nr * sizeof *rg);   /* the new rows into the window ring */
        if (!rg) return 0;
        for (int s = 0; s < nr; s++) rg[s] = (VkcRegion){(size_t)((pb + s) % ch->Wd) * hd, (size_t)s * hd, (size_t)hd};
        ok = vkc_copy_regions(ch->win[i], ch->kv, rg, nr);
        free(rg);
    }
    if (ok && r > 0 && c->kv_source[i]) {                           /* the compressor, its latents, the index keys */
        int g0 = pb / r, np = (pb + nr) / r - g0, ihd = c->index_head_dim, sp = ch->ks.on, q = ch->kvq;
        VkcBuf *cb = sp ? ch->cnew : q ? ch->qck : ch->ckv[i];       /* the split, and the pack: the new rows through a scratch */
        size_t co = (sp || q) ? 0 : (size_t)g0 * hd;
        VkcBuf *kb = q ? ch->qik : ch->ikey[i];                     /* a quantized mirror cannot hold the fp32 rows */
        size_t ko = q ? 0 : (size_t)g0 * ihd;
        ok = vkc_matmul(v41c_wb(&l->comp_wkv), ch->nrm, 0, ch->craw, 0, nr);
        if (ok && r == 1) ok = v41c_norm(ch, ch->craw, 0, hd, ch->o_cn[i], cb, co, hd, nr, hd, c->norm_eps);
        else if (ok) {
            VkcDsComp cp = {nr, pb, r, hd, hd, 0, 0, hd, 0, hd, -1, (sp || q) ? -g0 * hd : 0, hd, 0};
            ok = vkc_matmul(v41c_wb(&l->comp_wgate), ch->nrm, 0, ch->cscr, 0, nr) &&
                 vkc_dsv4_compress(ch->craw, ch->cscr, ch->ring[i], NULL, cb, &cp) &&
                 (np == 0 || v41c_norm(ch, cb, co, hd, ch->o_cn[i], cb, co, hd, np, hd, c->norm_eps));
            if (ok && f->spec && f->praw)                           /* a verify's raw rows, for its undo rows */
                ok = vkc_copy(ch->pull, f->praw[i], ch->craw, 0, (size_t)nr * hd) &&
                     vkc_copy(ch->pull, f->praw[i] + (size_t)nr * hd, ch->cscr, 0, (size_t)nr * hd);
        }
        if (ok && np > 0) {
            int bc = f->csB + (g0 * r - (pb - ch->rmax + 1)) * rd;   /* a latent rotates at its group's first position */
            ok = vkc_matmul(v41c_wb(&l->idx_wk), cb, co, kb, ko, np) &&
                 v41c_norm(ch, kb, ko, ihd, ch->o_ikn[i], kb, ko, ihd, np, ihd, c->norm_eps) &&
                 v41c_rope(ch, m, kb, np, 1, (int)ko + ihd - rd, ihd, 0, bc, r * rd, 0) &&
                 v41c_rope(ch, m, cb, np, 1, (int)co + hd - rd, hd, 0, bc, r * rd, 0);
            if (ok && q)   /* the packed rows into their mirrors, after their last fp32 reader */
                ok = vkc_dsv4_quant(ch->ckv[i], cb, &(VkcDsQuant){1, np, hd, 0, hd, g0, 0}) &&
                     vkc_dsv4_quant(ch->ikey[i], kb, &(VkcDsQuant){2, np, ihd, 0, ihd, g0, 0});
            if (ok && sp) {   /* into their slots, and down for the host's ckv once the frame is through */
                VkcKvPart pt = {1, hd, l->ckv, 0, ch->ckv[i], 0, 0, v41_host_kvq(c)};
                ok = vkc_kv_store(&ch->ks, ch->sli[i], &pt, cb, 0, (size_t)hd, 0, g0, np) &&
                     vkc_copy(ch->cdn, 0, cb, 0, (size_t)np * hd);
                vkc_kv_done(&ch->ks, ch->sli[i], g0 + np);
            }
        }
    }
    if (ok && r > 0 && c->index_source[i]) {                        /* the indexer: this layer's selection */
        int IH = c->index_n_heads, ID = c->index_head_dim, width = (pb + nr) / r, cs = c->candidate_source;
        int cand = v41c_cand_on(c), mrow = cand && cs < i ? (int)v41c_mask_words(ch->wcap) : 0;
        float wscale = (1.0f / sqrtf((float)ID)) * (1.0f / sqrtf((float)IH));
        ok = vkc_matmul(v41c_w8(&l->idx_wq_b), ch->qr, 0, ch->iq, 0, nr) &&
             v41c_rope(ch, m, ch->iq, nr * IH, IH, ID - rd, IH * ID, ID, tcs, rd, 0) &&
             vkc_matmul(v41c_wb(&l->idx_wproj), ch->nrm, 0, ch->ihw, 0, nr);
        const int *kl = f->keyl + (size_t)i * f->n + (pb - f->start);
        int same = 1;
        for (int s = 1; s < nr; s++) same &= kl[s] == kl[0];
        for (int s = 0; ok && s < nr; s += same ? nr : 1) {
            VkcDsScore sp = {same ? nr : 1, pb + s, r, IH, ID, width, s * IH * ID, IH * ID, s * IH, IH, 0, ID, mrow,
                             (int)ch->wcap, wscale, s * (int)ch->wcap, s * mrow, ch->kvq};   /* mrow: mask words a row */
            ok = vkc_dsv4_score(ch->iq, ch->ihw, ch->ikey[kl[s]], ch->mask, ch->isc, &sp);
        }
        if (ok && cand && cs == i) {
            VkcDsCand cp = {nr, pb, r, width, c->candidate_block_size, c->candidate_topk_blocks, (int)ch->wcap,
                            (int)v41c_mask_words(ch->wcap)};
            ok = vkc_dsv4_cand(ch->isc, ch->mask, &cp);
        }
        VkcDsTopk tp = {nr, width, ch->K, (int)ch->wcap, ch->W, LR, ch->Wd, 0};
        ok = ok && vkc_dsv4_topk(ch->isc, ch->list, &tp);
    }
    if (!ok) return 0;
    VkcDsAttn a = {nr, nh, hd, ch->W + (r > 0 ? ch->K : 0), 0, LR, ch->Wd, 0, 0, 0, qrow, 0, qrow, (int)ch->o_sink[i],
                   ch->kvq ? 2 : 0, 1.0f / sqrtf((float)hd)};   /* flags 2: the quantized mirrors (COLI_VK_KV_FP32 unset) */
    int og = c->o_groups, ol = c->o_lora, pg = qrow / og;
    VkcHgemv g = {0, nr, og, ol, ol, 0, 0, qrow, pg, 0, og * ol, ol, 0, 0, 0};
    if (ch->ks.on && r > 0) {   /* stage selected cold rows, preserving the original sparse attention order */
        int own = c->index_owner[i], li = ch->sli[own];
        VkcKvPart pt = {1, hd, m->L[own].ckv, 0, ch->ckv[own], 0, 0, v41_host_kvq(c)};
        VkcKvDs d = {&ch->ks, li, ch->q, ch->win[i], ch->ckv[own], ch->prm, ch->list, ch->heads, nr, nh, hd, a.cnt, 0, LR,
                     ch->Wd, 0, 0, qrow, (int)ch->o_sink[i], 0, 0, qrow, (pb + nr) / r, a.scale, m->L[own].ckv,
                     v41_host_kvq(c)};
        /* the blocks pinned since the chunk began, and the table, before this read */
        ok = vkc_kv_push(&ch->ks, li, &pt, 1, ch->ks.t[li].valid) && vkc_kv_ds(&d);
    } else ok = vkc_dsv4_attn(ch->q, ch->win[i], r > 0 ? ch->ckv[c->index_owner[i]] : NULL, ch->list, ch->prm, ch->heads, &a);
    return ok &&
           v41c_rope(ch, m, ch->heads, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 1) &&
           v41c_wo_a(ch, m, l, nr, &g) &&
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
    engram_advise_rows(e, table, pb, nr);
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
/* the packed cache's own guard, once a forward: the mirrors' bytes summed against the
 * flag's widths, so a width a branch missed fails loudly instead of corrupting quietly.
 * The split cannot be on beside the quantized mirrors (v41c_kvq_give_way); each chain
 * (each device's) checks its own. */
static void v41c_kvq_check(const V41Chain *ch, const Model *m) {
    const Cfg *c = &m->c;
    size_t act = 0, exp = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (ch->win[i]) { act += vkc_bytes(ch->win[i]); exp += (size_t)ch->Wd * v41c_win_row(ch, c); }
        if (ch->ckv[i]) { act += vkc_bytes(ch->ckv[i]); exp += (size_t)ch->ccap[i] * v41c_ckv_row(ch, c); }
        if (ch->ikey[i]) { act += vkc_bytes(ch->ikey[i]); exp += ch->kvq ? kvq_ikey_mirror_bytes((size_t)ch->ccap[i])
                                                                        : (size_t)ch->ccap[i] * v41c_ikey_row(ch, c); }
    }
    if (act != exp || ch->ks.on)
        fprintf(stderr, "[VK] %s chain: COLI_VK_KV_FP32 assertion failed: the KV mirrors hold %zu B where the flag's "
                        "widths say %zu B (the split %d) -- a width a branch missed\n", v41c_name(ch), act, exp, ch->ks.on),
        abort();
}
/* the host's rows up into the quantized mirrors (a re-upload, dsv41_kvq.h): a window
 * row (528 B, its ring slot), a run of compressed rows (288 B each, contiguous) and a run
 * of index keys (68 B each in block-wide sections: each block's touched data bytes and
 * scale bytes written apart). With the packed cache the host's own rows ARE the codes
 * (v41_host_kvq), so every one of these is a raw copy -- the host's row IS the mirror's
 * row, no pack, and the index keys go per touched block (their sections move as sections,
 * never as a linear row span). The give-way's fp32 mirrors are the one side that packs,
 * and they pack on the pull's write-back, not here. */
static int v41c_win_up(V41Chain *ch, const Layer *l, int i, int q, int W) {
    return vkc_write(ch->win[i], (size_t)(q % ch->Wd) * KVQ_WIN_BYTES / 4,
                     l->window + (size_t)(q % W) * KVQ_WIN_BYTES, KVQ_WIN_BYTES);
}
static int v41c_ckv_up(V41Chain *ch, const Layer *l, int i, int t0, int t1) {
    return vkc_write(ch->ckv[i], (size_t)t0 * KVQ_CKV_BYTES / 4, l->ckv + (size_t)t0 * KVQ_CKV_BYTES,
                     (size_t)(t1 - t0) * KVQ_CKV_BYTES);
}
static int v41c_ikey_up(V41Chain *ch, const Layer *l, int i, int t0, int t1) {
    for (int b = t0 / KVQ_IKEY_BLOCK, be = (t1 - 1) / KVQ_IKEY_BLOCK; b <= be; b++) {
        int lo = t0 > b * KVQ_IKEY_BLOCK ? t0 : b * KVQ_IKEY_BLOCK;
        int hi = t1 < (b + 1) * KVQ_IKEY_BLOCK ? t1 : (b + 1) * KVQ_IKEY_BLOCK;
        size_t w = (size_t)b * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES);
        const uint8_t *hb = l->ikey + w;   /* the host's block is the mirror's block: its sections, as sections */
        if (!vkc_write(ch->ikey[i], (w + (size_t)(lo - b * KVQ_IKEY_BLOCK) * 64) / 4,
                       hb + (size_t)(lo - b * KVQ_IKEY_BLOCK) * 64, (size_t)(hi - lo) * 64)) return 0;
        if (!vkc_write(ch->ikey[i], (w + 2048 + (size_t)(lo - b * KVQ_IKEY_BLOCK) * 4) / 4,
                       hb + 2048 + (size_t)(lo - b * KVQ_IKEY_BLOCK) * 4, (size_t)(hi - lo) * 4)) return 0;
    }
    return 1;
}
/* A chunk's tables (rows pb.. of a forward from start): its RoPE rows (the CPU's own
 * tables), window ones at 0 and compress ones at csB from position pb - rmax + 1; and the
 * window half of each row's list (window_idxs): the positions of the last `window` the row
 * reaches, oldest first, as device ring rows; -1 where the CPU's ring does not hold the
 * position (window_pos: the chain's first layer's, which holds what every layer held when
 * the forward began; the layers before it have written theirs by now). The selection half
 * starts empty. */
static void v41c_chunk_tables(const V41Chain *ch, const Model *m, int start, int pb, int nr, int csB, float *cs_, int *wl) {
    const Cfg *c = &m->c;
    int rd = c->rope_dim, W = ch->W, LR = W + ch->K;
    for (int s = 0; s < nr; s++) memcpy(cs_ + (size_t)s * rd, m->rope_window + (size_t)(pb + s) * rd, rd * sizeof(float));
    for (int s = 0; s < nr + ch->rmax - 1; s++) {
        int p = pb - ch->rmax + 1 + s;
        float *dst = cs_ + csB + (size_t)s * rd;
        if (p < 0) memset(dst, 0, rd * sizeof(float));
        else memcpy(dst, m->rope_compress + (size_t)p * rd, rd * sizeof(float));
    }
    const Layer *l0 = &m->L[ch->lo];
    for (int s = 0; s < nr; s++) {
        int p = pb + s;
        for (int k = 0; k < W; k++) {
            int q = p - W + 1 + k, e = -1;
            if (q >= 0 && (q >= start || l0->window_pos[q % W] == q)) e = q % ch->Wd;
            wl[(size_t)s * LR + k] = e;
        }
        for (int k = W; k < LR; k++) wl[(size_t)s * LR + k] = -1;
    }
}
/* The device's copies made the host's, as a forward from start needs them: the window rows
 * since the device last had them, each kv_source layer's compressed rows and keys up to
 * need[], its compressor's group. In the open frame. Under the packed cache the rows go up
 * as the flag's widths (the host's rows are the codes, a raw copy), and the split's
 * give-way leaves the mirrors fp32, so the host's codes dequantize up into them. */
static int v41c_state_up(V41Chain *ch, Model *m, int start, const int *need) {
    const Cfg *c = &m->c;
    int L = c->n_layers, lo = ch->lo, hi = lo + ch->n, W = ch->W, hd = c->head_dim, ok = 1;
    int hostq = v41_host_kvq(c);   /* the host's rows are codes whatever the mirrors are */
    for (int q = ch->win_valid > start - W + 1 ? ch->win_valid : start - W + 1; q < start && ok; q++) {
        if (q < 0) continue;
        for (int i = lo; i < hi && ok; i++) {
            const Layer *l = &m->L[i];
            if (l->window_pos[q % W] != q) continue;
            if (ch->kvq) ok = v41c_win_up(ch, l, i, q, W);
            else if (hostq) {   /* the split's give-way left the mirrors fp32: the codes dequantize up */
                float row[512];
                kvq_unpack_win(row, l->window + (size_t)(q % W) * KVQ_WIN_BYTES);
                ok = vkc_write(ch->win[i], (size_t)(q % ch->Wd) * hd, row, (size_t)hd * sizeof(float));
            } else ok = vkc_write(ch->win[i], (size_t)(q % ch->Wd) * hd,
                                 (const float *)(const void *)l->window + (size_t)(q % W) * hd,
                                 (size_t)hd * sizeof(float));
        }
    }
    for (int i = 0; i < L && ok; i++) {
        int own = v41c_own(ch, i);
        if (!c->kv_source[i] || (!own && !ch->ikey[i])) continue;
        const Layer *l = &m->L[i];
        int r = c->compress_ratio[i], ihd = c->index_head_dim, t0 = ch->kv_valid[i], t1 = need[i];
        if (t1 > t0) {
            if (ch->kvq)
                ok = (!own || v41c_ckv_up(ch, l, i, t0, t1)) && v41c_ikey_up(ch, l, i, t0, t1);
            else if (hostq) {   /* the give-way's fp32 mirrors, from the host's codes */
                float *rows = malloc((size_t)(t1 - t0) * (hd > ihd ? hd : ihd) * sizeof(float));
                if (!rows) ok = 0;
                else {
                    for (int t = t0; t < t1; t++)
                        kvq_unpack_ckv(rows + (size_t)(t - t0) * hd, l->ckv + (size_t)t * KVQ_CKV_BYTES);
                    ok = !own || ch->ks.on || vkc_write(ch->ckv[i], (size_t)t0 * hd, rows,
                                                         (size_t)(t1 - t0) * hd * sizeof(float));
                    for (int t = t0; ok && t < t1; t++) {
                        kvq_unpack_ikey(rows + (size_t)(t - t0) * ihd, l->ikey, t);
                        ok = vkc_write(ch->ikey[i], (size_t)t * ihd, rows + (size_t)(t - t0) * ihd,
                                       (size_t)ihd * sizeof(float));
                    }
                    free(rows);
                }
            } else ok = (!own || ch->ks.on || vkc_write(ch->ckv[i], (size_t)t0 * hd,
                                 (const float *)(const void *)l->ckv + (size_t)t0 * hd,
                                 (size_t)(t1 - t0) * hd * sizeof(float))) &&
                         vkc_write(ch->ikey[i], (size_t)t0 * ihd,
                                   (const float *)(const void *)l->ikey + (size_t)t0 * ihd,
                                   (size_t)(t1 - t0) * ihd * sizeof(float));
        }
        if (!own) { if (ok && t1 > t0) ch->kv_valid[i] = t1; continue; }
        if (ok && r > 1) ok = vkc_write(ch->ring[i], 0, l->cstate_kv, (size_t)r * hd * sizeof(float)) &&
                              vkc_write(ch->ring[i], (size_t)r * hd, l->cstate_score, (size_t)r * hd * sizeof(float));
    }
    return ok;
}
/* What the host's state gets back after a forward from start to E, into pull at pc[] (each
 * layer's window rows, pw floats, then its compressed rows, keys and group). In the open
 * frame. Under the packed cache the code rows come back RAW (the compressed rows
 * contiguous, the index keys as whole block-sections blocks, the group after them); the
 * give-way's fp32 mirrors pull floats and pack at the write-back. */
static int v41c_state_pull(V41Chain *ch, Model *m, int start, int E, const size_t *pc, size_t pw) {
    const Cfg *c = &m->c;
    int lo = ch->lo, hi = lo + ch->n, W = ch->W, hd = c->head_dim, ok = 1;
    for (int i = lo; i < hi && ok; i++) {
        int q0 = E - W > start ? E - W : start, nq = E - q0;
        size_t wb = ch->kvq ? KVQ_WIN_BYTES / 4 : (size_t)hd;   /* floats a window row travels: the codes, or fp32 */
        VkcRegion *rg = malloc((size_t)nq * sizeof *rg);
        if (!rg) return 0;
        for (int q = q0; q < E; q++) rg[q - q0] = (VkcRegion){pc[i] + (size_t)(q - q0) * wb, (size_t)(q % ch->Wd) * wb, wb};
        ok = vkc_copy_regions(ch->pull, ch->win[i], rg, nq);
        free(rg);
        if (!ok || !c->kv_source[i]) continue;
        int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim, cw = ch->ks.on ? 0 : hd;
        size_t at = pc[i] + pw;
        if (ch->kvq) {   /* raw: the compressed rows contiguous, the index keys whole blocks */
            size_t ik_at = at + (cw ? (size_t)np * (KVQ_CKV_BYTES / 4) : 0);
            if (cw) ok = vkc_copy(ch->pull, at, ch->ckv[i], (size_t)g0 * (KVQ_CKV_BYTES / 4),
                                 (size_t)np * (KVQ_CKV_BYTES / 4));
            for (int b = g0 / KVQ_IKEY_BLOCK, be = (g0 + np - 1) / KVQ_IKEY_BLOCK; ok && b <= be; b++)
                ok = vkc_copy(ch->pull, ik_at + (size_t)(b - g0 / KVQ_IKEY_BLOCK) * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES / 4),
                              ch->ikey[i], (size_t)b * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES / 4),
                              KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES / 4);   /* vkc_copy counts floats, as ever */
            if (ok && r > 1)
                ok = vkc_copy(ch->pull, ik_at + (size_t)(((g0 + np + KVQ_IKEY_BLOCK - 1) / KVQ_IKEY_BLOCK) - g0 / KVQ_IKEY_BLOCK)
                                           * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES / 4),
                              ch->ring[i], 0, (size_t)2 * r * hd);
        } else
        ok = (cw == 0 || vkc_copy(ch->pull, at, ch->ckv[i], (size_t)g0 * hd, (size_t)np * hd)) &&
             vkc_copy(ch->pull, at + (size_t)np * cw, ch->ikey[i], (size_t)g0 * ihd, (size_t)np * ihd) &&
             (r <= 1 || vkc_copy(ch->pull, at + (size_t)np * (cw + ihd), ch->ring[i], 0, (size_t)2 * r * hd));
    }
    return ok;
}
/* A prompt of more than one chunk layer by layer: each layer's chunks in order through the
 * attention, the FFN's mix and the shared expert, every row's streams and mixes kept on the
 * device between layers, then the layer's routed experts for all its rows in one step, so
 * the tier copies each streamed expert once a layer, not once a chunk (a 4104-token prompt
 * of V4.1-Flash UD-IQ1_S at 512-row chunks copied 150 GiB of the host pool's experts), and
 * then each chunk's FFN sum into its streams. The same ops on the same rows as chunk by
 * chunk, in another order. Only a whole chain's plain prompt (no verify, no DSpark targets,
 * no split compressed rows). COLI_VK_V41_LAYERWISE=0: chunk by chunk. 1 done, 0 not taken
 * (nothing recorded), -1 a frame failed, -2 the prompt was stopped. */
static int v41c_prompt_lw(Model *m, V41Chain *ch, const float *h, int n, int start, int all, int rows, V41Fwd *f,
                          const int *need, const size_t *pc, size_t pw, float *cs_, int *wl, float *er, float *outs, float *pres) {
    const Cfg *c = &m->c;
    const int L = c->n_layers, D = c->dim, H = c->hc_mult, HD = H * D, hr = 2 * H + H * H, rd = c->rope_dim;
    const int LR = ch->W + ch->K, E = start + n, T = m->spec.active ? c->n_spec_targets : 0;
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_V41_LAYERWISE"); on = !(e && *e == '0'); }
    if (!on || f->spec || T || ch->lo != 0 || ch->n != L || ch->ks.on || n <= rows) return 0;
    if (ch->lw_cap < (size_t)n) {
        float *a = realloc(ch->lw_in, (size_t)n * D * sizeof(float)), *b = a ? realloc(ch->lw_out, (size_t)n * D * sizeof(float)) : NULL;
        if (a) ch->lw_in = a;
        if (b) ch->lw_out = b;
        if (!a || !b) return 0;
        ch->lw_cap = (size_t)n;
    }
    const int cand = v41c_cand_on(c), cs = c->candidate_source;
    if (!v41c_res(&ch->lw_xs, (size_t)n * HD, VKC_DEV) || !v41c_res(&ch->lw_hp, (size_t)n * hr, VKC_DEV) ||
        !v41c_res(&ch->lw_ds, (size_t)n * D, VKC_DEV) || !v41c_res(&ch->lw_list, (size_t)n * LR, VKC_DEV) ||
        (cand && !v41c_res(&ch->lw_mask, (size_t)n * v41c_mask_words(ch->wcap), VKC_DEV))) {
        static int said;
        if (!said++) fprintf(stderr, "[VK] %s chain: no device memory for a prompt of %d rows layer by layer; chunk by chunk\n", v41c_name(ch), n);
        return 0;
    }
    float *hp0 = calloc((size_t)rows * hr, sizeof(float));
    if (!hp0) return 0;
    for (int s = 0; s < rows; s++) hp0[(size_t)s * hr] = 1.0f;   /* the first site collapses with [1, 0, ...] */
    int ok = vkc_begin() && v41c_state_up(ch, m, start, need) && vkc_submit(1);
    for (int c0 = 0; ok && c0 < n; c0 += rows) {   /* the rows' streams and their lists' window halves a chunk at a time */
        int nr = n - c0 < rows ? n - c0 : rows;
        v41c_chunk_tables(ch, m, start, start + c0, nr, f->csB, cs_, wl);
        ok = vkc_begin() && vkc_write(ch->lw_xs, (size_t)c0 * HD, h + (size_t)c0 * HD, (size_t)nr * HD * sizeof(float)) &&
             vkc_write(ch->lw_hp, (size_t)c0 * hr, hp0, (size_t)nr * hr * sizeof(float)) &&
             vkc_write(ch->lw_list, (size_t)c0 * LR, wl, (size_t)nr * LR * sizeof(int)) && vkc_submit(1);
    }
    free(hp0);
    if (!ok) return -1;
    static int said_lw;
    if (!said_lw++) fprintf(stderr, "[VK] %s chain: a prompt of more than one chunk layer by layer, each layer's routed experts in one step "
                                    "(COLI_VK_V41_LAYERWISE=0: chunk by chunk)\n", v41c_name(ch));
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        /* past the cross-layer source only the chunks of the prompt's tail (v41_prefill_tail) */
        int first = 0;
        if (m->ptail_rows > 0 && m->ptail_after < L - 1 && i > m->ptail_after)
            while (first < n && first + (n - first < rows ? n - first : rows) <= n - m->ptail_rows) first += rows;
        if (first >= n) continue;
        double ta = now_s();
        vkt_stream_prefetch(i, n - first);
        for (int c0 = first; c0 < n; c0 += rows) {
            if (coli_prompt_break()) return -2;
            int nr = n - c0 < rows ? n - c0 : rows, pb = start + c0;
            f->pb = pb; f->nr = nr;
            v41c_chunk_tables(ch, m, start, pb, nr, f->csB, cs_, wl);
            /* a chunk's rows keep, from one layer to the next, the selection the last index source made
             * for them (the compressed layers after it attend to it) and the candidate source's mask:
             * the chunk-by-chunk chain holds both for its chunk across the layers, here every chunk of a
             * layer runs before the next layer */
            const int sel = c->compress_ratio[i] > 0 && c->index_source[i];
            ok = vkc_begin() && vkc_copy(ch->list, 0, ch->lw_list, (size_t)c0 * LR, (size_t)nr * LR) &&
                 (!cand || !sel || cs >= i || vkc_copy(ch->mask, 0, ch->lw_mask, (size_t)c0 * v41c_mask_words(ch->wcap),
                                                       (size_t)nr * v41c_mask_words(ch->wcap))) &&
                 vkc_write(ch->cs, 0, cs_, (size_t)(f->csB + (nr + ch->rmax - 1) * rd) * sizeof(float)) &&
                 vkc_copy(ch->xs, 0, ch->lw_xs, (size_t)c0 * HD, (size_t)nr * HD) &&
                 vkc_copy(ch->hpf, 0, ch->lw_hp, (size_t)c0 * hr, (size_t)nr * hr);
            if (ok && l->engram_index >= 0) ok = v41c_engram(ch, m, i, nr, pb, er);
            ok = ok && v41c_pre(ch, m, ch->fna[i], ch->o_hca[i], ch->hpa, ch->hpf, ch->o_an[i], nr) &&
                 v41c_attention(ch, m, i, f) &&
                 (!sel || vkc_copy(ch->lw_list, (size_t)c0 * LR, ch->list, 0, (size_t)nr * LR)) &&
                 (!cand || i != cs || vkc_copy(ch->lw_mask, (size_t)c0 * v41c_mask_words(ch->wcap), ch->mask, 0,
                                               (size_t)nr * v41c_mask_words(ch->wcap))) &&
                 v41c_post(ch, m, ch->hpa, nr) &&
                 v41c_pre(ch, m, ch->fnf[i], ch->o_hcf[i], ch->hpf, ch->hpa, ch->o_fn[i], nr) &&
                 vkc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D) && v41c_shared(ch, m, l, nr) &&
                 vkc_copy(ch->lw_xs, (size_t)c0 * HD, ch->xs, 0, (size_t)nr * HD) &&
                 vkc_copy(ch->lw_hp, (size_t)c0 * hr, ch->hpf, 0, (size_t)nr * hr) &&
                 vkc_copy(ch->lw_ds, (size_t)c0 * D, ch->ds, 0, (size_t)nr * D) && vkc_submit(1);
            if (!ok) return -1;
            memcpy(ch->lw_in + (size_t)c0 * D, vkc_ptr(ch->h2d), (size_t)nr * D * sizeof(float));
        }
        m->t_attn += now_s() - ta;
        double t1 = now_s();
        moe_run_at(m, l, &m->cache[i], "layers", i, c->n_routed, c->n_activated, ch->lw_in + (size_t)first * D, n - first,
                   ch->lw_out + (size_t)first * D, 0);
        ch->host_ms += (now_s() - t1) * 1e3;
        for (int c0 = first; c0 < n; c0 += rows) {   /* each chunk's FFN branch into its streams */
            int nr = n - c0 < rows ? n - c0 : rows;
            memcpy(vkc_ptr(ch->routed), ch->lw_out + (size_t)c0 * D, (size_t)nr * D * sizeof(float));
            VkcEw add = {VKC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = vkc_begin() && vkc_copy(ch->xs, 0, ch->lw_xs, (size_t)c0 * HD, (size_t)nr * HD) &&
                 vkc_copy(ch->hpf, 0, ch->lw_hp, (size_t)c0 * hr, (size_t)nr * hr) &&
                 vkc_copy(ch->ds, 0, ch->lw_ds, (size_t)c0 * D, (size_t)nr * D) &&
                 vkc_ew(ch->br, ch->routed, ch->ds, NULL, NULL, &add) && v41c_post(ch, m, ch->hpf, nr) &&
                 vkc_copy(ch->lw_xs, (size_t)c0 * HD, ch->xs, 0, (size_t)nr * HD) && vkc_submit(1);
            if (!ok) return -1;
        }
    }
    /* the rows' streams and mixes (every row's, or the last's) and what the host's state gets back */
    for (int c0 = 0; c0 < n; c0 += rows) {
        int nr = n - c0 < rows ? n - c0 : rows, last = c0 + nr == n;
        if (!all && !last) continue;
        ok = vkc_begin() && vkc_copy(ch->xd, 0, ch->lw_xs, (size_t)c0 * HD, (size_t)nr * HD) &&
             vkc_copy(ch->hpd, 0, ch->lw_hp, (size_t)c0 * hr, (size_t)nr * hr) &&
             (!last || v41c_state_pull(ch, m, start, E, pc, pw)) && vkc_submit(1);
        if (!ok) return -1;
        const float *xd = (const float *)vkc_ptr(ch->xd), *hpd = (const float *)vkc_ptr(ch->hpd);
        for (int s = 0; s < nr; s++) {
            int t = c0 + s;
            if (!all && t != n - 1) continue;
            memcpy(outs + (size_t)(all ? t : 0) * HD, xd + (size_t)s * HD, (size_t)HD * sizeof(float));
            memcpy(pres + (size_t)(all ? t : 0) * H, hpd + (size_t)s * hr, (size_t)H * sizeof(float));
        }
    }
    return 1;
}
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
    /* The indexer's scores and mask are quadratic in the prompt: a row costs 4E bytes of
     * scores and E/8 of the packed mask, so the two together keep to the 256 MiB they took
     * as 32M/E rows of fp32 scores and mask (v41c_scratch), and the mask being a bit a
     * column is what lets twice the rows into it. */
    {
        int64_t per_row = 4 * (int64_t)E + (int64_t)v41c_mask_words((size_t)E) * 4;
        int64_t lim = ((int64_t)256 << 20) / (per_row > 0 ? per_row : 1);
        if (lim < CH) {
            CH = lim < 1 ? 1 : (int)lim;
            if (!g_v41c_cap_said) {
                g_v41c_cap_said = 1;
                fprintf(stderr, "[VK] %s chain: a prompt of %d positions caps a chunk at %d rows (%.1f KiB a row of scores and "
                                "packed candidate mask, %.0f MiB of 256; COLI_VK_CHAIN_ROWS sets it)\n",
                        nmc, E, CH, per_row / 1024.0, (double)lim * per_row / 1048576.0);
            }
        }
    }
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
    if (ok && ch->kvq && !ch->kvq_checked) { v41c_kvq_check(ch, m); ch->kvq_checked = 1; }   /* the mirrors' widths, once */
    /* what comes back at the end: the window rows, the compressed rows and keys, the groups,
     * a verify's raw compressor rows. Offsets in floats, as every pull command's: under the
     * packed cache the code rows themselves -- the window's 528 B and the compressed rows'
     * 288 B a row, the index keys as whole blocks of 2176 B (their data and scale sections
     * never travel as one linear row span) -- and fp32 rows when the mirrors declined the
     * flag or gave way to the split. The give-way (the split armed, ch->kvq 0) pulls
     * FLOATS while the host keeps the codes and packs at the write-back, so the staging's
     * widths follow what the pull carries. The groups and a verify's raw rows stay fp32
     * either way: the pooling reads raw values. */
    int hostq = v41_host_kvq(c), pullq = ch->kvq;
    size_t pw = (size_t)(n < W ? n : W) * (pullq ? KVQ_WIN_BYTES / 4 : (size_t)hd), pn = 0, *pc = calloc((size_t)L, sizeof(size_t));
    for (int i = lo; ok && pc && i < hi; i++) {
        pc[i] = pn; pn += pw;
        int r = c->compress_ratio[i];
        if (!c->kv_source[i]) continue;
        int nr2 = E / r - start / r;
        if (pullq) {
            if (!ch->ks.on) pn += (size_t)nr2 * (KVQ_CKV_BYTES / 4);
            pn += (size_t)(((start / r + nr2 + KVQ_IKEY_BLOCK - 1) / KVQ_IKEY_BLOCK) - (start / r) / KVQ_IKEY_BLOCK)
                  * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES / 4);
        } else
            pn += (size_t)nr2 * ((ch->ks.on ? 0 : hd) + c->index_head_dim);   /* split: the rows came back per chunk */
        if (r > 1) { pn += (size_t)2 * r * hd; if (spec) { f.praw[i] = pn; pn += (size_t)2 * n * hd; } }
    }
    ok = ok && pc && v41c_scratch(ch, m, rows, E) && v41c_res(&ch->pull, pn, VKC_DOWN) && (!T || m->main_hidden) &&
         (!cand_out || v41c_res(&ch->cmd, (size_t)rows * v41c_mask_words(ch->wcap), VKC_DOWN)) &&
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
    const int lwr = v41c_prompt_lw(m, ch, h, n, start, all, rows, &f, need, pc, pw, cs_, wl, er, outs, pres);
    if (lwr == -1) goto lost;
    if (lwr == -2) {   /* stopped between two chunks (prompt_break.h), as below */
        free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs_); free(er); free(pc);
        return 0;
    }
    for (int c0 = 0; lwr == 0 && c0 < n; c0 += rows) {
        /* a served prompt's STOP or CANCEL between two chunks (prompt_break.h): the caller
         * starts the state over, on the host and on the device (model_reset, v41c_reset) */
        if (coli_prompt_break()) {
            free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs_); free(er); free(pc);
            return 0;
        }
        int nr = n - c0 < rows ? n - c0 : rows, pb = start + c0, last = c0 + nr == n;
        f.pb = pb; f.nr = nr;
        v41c_chunk_tables(ch, m, start, pb, nr, f.csB, cs_, wl);
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
        if (c0 == 0 && !v41c_state_up(ch, m, start, need)) goto lost;   /* the device's copies made the host's */
        for (int i = 0; ch->ks.on && i < L && ok; i++) {   /* the split: each table's window over this chunk's new rows */
            if (ch->sli[i] < 0) continue;
            int r = c->compress_ratio[i], g0 = pb / r;
            VkcKvPart pt = {1, hd, m->L[i].ckv, 0, ch->ckv[i], 0, 0, v41_host_kvq(c)};
            vkc_kv_place(&ch->ks, ch->sli[i], g0, (pb + nr) / r - g0);
            ok = vkc_kv_push(&ch->ks, ch->sli[i], &pt, 1, g0);
        }
        if (!ok) goto lost;
        int pending = 0;
        /* a chunk all before the prompt's tail stops after the cross-layer source
         * (v41_prefill_tail): the layers after it run on the tail only */
        int top = hi;
        if (m->ptail_rows > 0 && m->ptail_after < hi - 1 && c0 + nr <= n - m->ptail_rows)
            top = m->ptail_after + 1 > lo ? m->ptail_after + 1 : lo;   /* a second device's segment: none of it */
        for (int i = lo; i < top && ok; i++) {
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
                 (!cand_out || i != cs || vkc_copy(ch->cmd, 0, ch->mask, 0, (size_t)nr * v41c_mask_words(ch->wcap))) &&
                 vkc_submit(0);   /* A1 */
            /* while it runs, the tier loads the experts this layer will likely stream (a
             * big prompt chunk only) */
            if (ok) vkt_stream_prefetch(i, nr);
            ok = ok && vkc_finish();
            m->t_attn += now_s() - ta;
            if (!ok) break;
            if (ch->ks.on && ch->sli[i] >= 0) {            /* this chunk's new compressed rows into the host's ckv */
                int r = c->compress_ratio[i], g0 = pb / r, np = (pb + nr) / r - g0;
                if (np > 0) {   /* the rows pack into the host's code rows (the host never gives way) */
                    if (hostq) for (int t = 0; t < np; t++)
                        kvq_pack_ckv(l->ckv + (size_t)(g0 + t) * KVQ_CKV_BYTES, (const float *)vkc_ptr(ch->cdn) + (size_t)t * hd);
                    else memcpy((float *)(void *)l->ckv + (size_t)g0 * hd, vkc_ptr(ch->cdn), (size_t)np * hd * sizeof(float));
                }
            }
            if (cand_out && i == cs) {                     /* the candidate source's mask, for the CPU's readers */
                const int32_t *mk = (const int32_t *)vkc_ptr(ch->cmd);
                int wc = (pb + nr) / c->compress_ratio[cs], wh = m->candidate_width, mw = (int)v41c_mask_words(ch->wcap);
                for (int s = 0; s < nr; s++) {
                    uint8_t *keep = m->candidates + (size_t)(c0 + s) * wh;
                    const int32_t *row = mk + (size_t)s * mw;
                    for (int j = 0; j < wh; j++)
                        keep[j] = j < wc && ((uint32_t)row[j >> 5] >> (uint32_t)(j & 31)) & 1u;
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
        if (ok && last) ok = v41c_state_pull(ch, m, start, E, pc, pw);   /* what the host's state gets back */
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
    /* every chunk is through: the host's state as the CPU would have left it. Under the
     * flag the staging may hold codes (raw pulls) or floats (the give-way's fp32 mirrors,
     * packed into the codes here) -- `raw` says which, and the groups' parts are floats
     * either way */
    {
        const uint8_t *pl = (const uint8_t *)vkc_ptr(ch->pull);
        const float *plf = (const float *)vkc_ptr(ch->pull);
        int raw = ch->kvq;
        size_t roww = v41_row_win(c);
        int save = spec && m->rollback_save && n > 1;
        for (int i = lo; i < hi; i++) {
            Layer *l = &m->L[i];
            int q0 = E - W > start ? E - W : start;
            for (int t = 0; t < n; t++) {
                int p = start + t, slot = p % W;
                if (save) {
                    memcpy(l->ring_save + (size_t)t * roww, l->window + (size_t)slot * roww, roww);
                    l->ring_save_pos[t] = l->window_pos[slot];
                }
                if (p < q0) continue;                      /* a later row takes its slot */
                if (raw) memcpy(l->window + (size_t)slot * roww, pl + pc[i] * 4 + (size_t)(p - q0) * roww, roww);
                else if (hostq) kvq_pack_win(l->window + (size_t)slot * KVQ_WIN_BYTES, plf + pc[i] + (size_t)(p - q0) * hd);
                else memcpy((float *)(void *)l->window + (size_t)slot * hd, plf + pc[i] + (size_t)(p - q0) * hd,
                           (size_t)hd * sizeof(float));
                l->window_pos[slot] = p;
            }
            if (!c->kv_source[i]) continue;
            int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim, cw = ch->ks.on ? 0 : hd;
            size_t at = pc[i] + pw;   /* floats, as the pull's */
            const float *src = plf + at;
            const uint8_t *srcb = pl + at * 4;
            size_t ik_at = at + (cw ? (size_t)np * (KVQ_CKV_BYTES / 4) : 0);   /* the ikey blocks' start, raw */
            if (raw) {
                if (cw) memcpy(l->ckv + (size_t)g0 * KVQ_CKV_BYTES, srcb, (size_t)np * KVQ_CKV_BYTES);   /* split: back per chunk */
                for (int b = g0 / KVQ_IKEY_BLOCK, be = (g0 + np - 1) / KVQ_IKEY_BLOCK; b <= be; b++)
                    memcpy(l->ikey + (size_t)b * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES),
                           srcb + (cw ? (size_t)np * KVQ_CKV_BYTES : 0) + (size_t)(b - g0 / KVQ_IKEY_BLOCK) * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES),
                           KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES);
            } else if (hostq) {   /* the give-way's fp32 rows, packed into the codes */
                if (cw) for (int t = 0; t < np; t++)
                    kvq_pack_ckv(l->ckv + (size_t)(g0 + t) * KVQ_CKV_BYTES, src + (size_t)t * hd);
                for (int t = 0; t < np; t++)
                    kvq_pack_ikey(l->ikey, g0 + t, src + (size_t)np * cw + (size_t)t * ihd);
            } else {
                if (cw) memcpy((float *)(void *)l->ckv + (size_t)g0 * hd, src, (size_t)np * hd * sizeof(float));   /* split: back per chunk */
                memcpy((float *)(void *)l->ikey + (size_t)g0 * ihd, src + (size_t)np * cw, (size_t)np * ihd * sizeof(float));
            }
            if (r <= 1) continue;
            if (save) {   /* the groups' undo rows, as compressor_run's decode path writes them */
                const float *rawp = plf + f.praw[i];
                for (int t = 0; t < n; t++) {
                    int slot = (start + t) % r;
                    memcpy(l->cstate_save_kv + (size_t)t * hd, l->cstate_kv + (size_t)slot * hd, (size_t)hd * sizeof(float));
                    memcpy(l->cstate_save_score + (size_t)t * hd, l->cstate_score + (size_t)slot * hd, (size_t)hd * sizeof(float));
                    l->cstate_save_slot[t] = slot;
                    memcpy(l->cstate_kv + (size_t)slot * hd, rawp + (size_t)t * hd, (size_t)hd * sizeof(float));
                    memcpy(l->cstate_score + (size_t)slot * hd, rawp + (size_t)(n + t) * hd, (size_t)hd * sizeof(float));
                }
            }
            const float *ring = plf + (raw ? ik_at + (size_t)(((g0 + np + KVQ_IKEY_BLOCK - 1) / KVQ_IKEY_BLOCK) - g0 / KVQ_IKEY_BLOCK)
                                               * (KVQ_IKEY_BLOCK * KVQ_IKEY_BYTES / 4)
                                          : at + (size_t)np * (cw + ihd));
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
        fprintf(stderr, "[VK] %s chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM, "
                        "%llu indexer tiles), %.1f ms waiting for the device, %.1f ms of routed experts on "
                        "the host, %.1f MiB on the device\n",
                v41c_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.dsa_tiled, st.wait_ms,
                ch->host_ms, st.dev_bytes / 1048576.0);
        vkc_kv_report(&ch->ks);
        vkc_prof_print();
        vkc_device(was);
    }
}

/* The turn-end VRAM ledger: one line, everything the engine holds on the device, so that
 * a VRAM question is a lookup and not a sysfs reconstruction (the audit, §6.3: nothing
 * printed the pool blocks, the streaming slots or the extra layers' pool, so 2.14 GiB of a
 * short-prompt run could not be attributed).
 *
 *   weights   the trunk's matrices: the weight pool's blocks (their bytes together; the
 *             rows handed out of them are a little less)
 *   chain     vkc_stats' dev_bytes (the chain's live buffers) and the three pools' blocks
 *             (dev, down, up: what the device has actually allocated for it)
 *   tier      its experts' pool: the resident experts and the streaming slots, with the
 *             slots' own bytes, then the extra layers' (DSpark's) pool
 *   kv        the chain's KV mirrors: the window rings, the compressed rows, the index keys
 *             and the compressor rings (all of them VKC_DEV buffers, so a part of `chain`)
 *   host kv   the HOST's own KV arrays (the same state on the CPU side): the packed code
 *             rows under COLI_VK_KV_FP32 unset, fp32 rows under =1 -- the line that makes
 *             the flag's host savings visible at all (printed only when they are packed;
 *             a second device's mirrors, when there are two, are not in this line -- the
 *             pre-existing gap)
 *   process   the engine's own device allocations (backend_vulkan.c's books)
 *   device    the card's live mem_info_vram_used, the whole card
 *   gtt       the card's mem_info_gtt_used: system RAM behind the aperture, where the
 *             host-visible pools land when device-local is exhausted (the spill is host
 *             RAM pressure that process RSS does not show)
 * `unaccounted` = process - (the pools' blocks): the chain's frame staging, the batch's
 * scratch, the imported host buffers -- what no pool holds.
 * sys_used/sys_total/card: the device's live usage from sysfs (v41_ledger in the engine);
 * sys_gtt: the same card's mem_info_gtt_used. */
static void v41c_ledger(unsigned long long sys_used, unsigned long long sys_total, unsigned long long sys_gtt, const char *card,
                        size_t host_kv) {
    const double M = 1048576.0, G = 1073741824.0;
    V41Chain *ch = v41c_of(0);
    if (!ch || !ch->ok) return;
    VkcStats st; vkc_stats(&st);
    VkcPoolBytes dev = {0, 0, 0}, dn = dev, up = dev;
    vkc_pool_bytes(VKC_DEV, &dev); vkc_pool_bytes(VKC_DOWN, &dn); vkc_pool_bytes(VKC_UP, &up);
    size_t kv = 0;   /* the mirrors are VKC_DEV buffers: a part of the chain's dev_bytes */
    for (int i = ch->lo; i < ch->lo + ch->n; i++)
        kv += vkc_bytes(ch->win[i]) + vkc_bytes(ch->ring[i]) + vkc_bytes(ch->ckv[i]) + vkc_bytes(ch->ikey[i]);
    /* COLI_VK_KV_FP32: the line walks this chain (device 0) only; a second device's
     * mirrors are as quantized but not counted here (the pre-existing gap, noted) */
    const char *qa = ch->kvq ? " (quantized; a second device's mirrors, when there are two, are not in this line)" : "";
    char hk[96] = "";   /* the host's own arrays, only when they are packed (the savings) */
    if (host_kv) snprintf(hk, sizeof hk, " | host kv %.1f MiB (packed)", host_kv / M);
    ColiVkPoolStats w; coli_vk_pool_stats(0, &w);
    VktVram tv; vkt_vram(&tv);
    size_t pools = w.total + dev.total + dn.total + up.total + tv.res_total + tv.x_total;
    size_t proc = coli_vk_device_used();
    fprintf(stderr, "[VK] deepseek_v41 vram: weights %.1f MiB in %d blocks (%.1f MiB of rows in %zu tensors) | chain %.1f MiB (dev %.1f in %d "
                    "blocks, down %.1f in %d, up %.1f in %d) | tier resident %.1f MiB + %d slots %.1f MiB = %.1f of %.1f in %d "
                    "blocks | extra layers %.1f in %d blocks | kv mirror %.1f MiB%s%s | process %.2f GiB of %.2f GiB in pools "
                    "(%.1f MiB outside) | device %.2f GiB used of %.2f, gtt %.2f GiB%s%s%s\n",
            w.total / M, w.blocks, w.used / M, w.tensors, st.dev_bytes / M, dev.total / M, dev.blocks, dn.total / M, dn.blocks,
            up.total / M, up.blocks, (double)(tv.res_used - tv.slot_bytes) / M, tv.slots, (double)tv.slot_bytes / M,
            (double)tv.res_used / M, (double)tv.res_total / M, tv.res_blocks, tv.x_total / M, tv.x_blocks, kv / M, qa, hk,
            (double)proc / G, (double)pools / G, proc > pools ? (double)(proc - pools) / M : 0.0,
            (double)sys_used / G, (double)sys_total / G, (double)sys_gtt / G, *card ? " (" : "", *card ? card : "", *card ? ")" : "");
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
