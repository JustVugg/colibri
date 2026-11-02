/* deepseek_v4_chain.h -- DeepSeek V4's layers as a dense chain on the Vulkan device
 * (vk_chain.h). Included once by deepseek_v4.c's GENERATE_STATS unit in a COLI_VULKAN
 * build, after the per-matrix path (v4_vk_*): the chain's matrices are that path's
 * device copies, looked up in the same map, one copy per matrix. target_batch and
 * target_token hand it their forwards (v4c_forward); COLI_VK_CHAIN decides
 * (coli_vk_chain_decide, this engine's integrated GPU default COLI_VK_CHAIN_UNMEASURED:
 * off, not measured on a V4 checkpoint).
 *
 * The residual is hc_mult streams per position (mHC); each site collapses them with the
 * mix it computes itself (hc_pre). DeepSeek V4 rounds to bf16 after nearly every step
 * and rounds the input of every fp8 matrix to E4M3 per 128 (the CPU kernel's activation):
 * the chain makes each of those roundings where the CPU makes it (vkc_dsv4_round), so
 * the device and the CPU differ only by the order of the sums and the device's exp and
 * sqrt, which the roundings mostly absorb (a value that lands on a bf16 boundary can go
 * the other way, and an E4M3 block then rounds on another step: docs/vulkan.md has what
 * that does to the logits). Every matrix takes the per-row GEMV, so a row's bits do not
 * depend on how a forward is cut into chunks, nor on prompt against decode against a
 * verify, as on the CPU. What runs where, per layer, for one block of rows (decode: one
 * row; a prompt in chunks of the CPU's prefill chunk):
 *   device, frame A1: the previous layer's FFN branch (the routed sum from the host plus
 *     the shared expert, to bf16) written back into the streams (mHC post, bf16); the
 *     attention site: hc_attn_fn, the split with Sinkhorn, the collapse, bf16, the
 *     RMSNorm, bf16; the attention: the input to E4M3, wq_a, bf16, q_norm, bf16; on a
 *     compressed layer the compressor (wkv and wgate in bf16 on the f32 input, the ring
 *     with its position bias, the pooled rows to bf16, their norm, bf16, RoPE at the
 *     group's first position, the no-position part to E4M3 per 64); on a ratio-4 layer
 *     the indexer (its own compressor into the keys: norm, RoPE, the Hadamard transform,
 *     E2M1 per 32; the queries: indexer wq_b on the q latent, bf16, RoPE, Hadamard,
 *     E2M1; weights_proj; the relu-weighted scores and the top-k in score order); wq_b,
 *     bf16, the per-head RMS without weight, bf16; wkv, bf16, kv_norm, bf16; RoPE (YaRN's
 *     table on compressed layers) with bf16 on the rotated part; the key's no-position
 *     part to E4M3 per 64, bf16; the new rows into the window ring; the sparse attention
 *     with the sink over the window and the compressed rows (V4's bf16 weights and
 *     output), the inverse RoPE, bf16; wo_a per group (each group's input to E4M3),
 *     bf16, wo_b, bf16; mHC post, bf16; the FFN site's split, collapse, norm. Then the
 *     frame is waited for.
 *   host: the routed experts (coli_v4_moe_routed: the bf16 or the hash router, the
 *     expert tier's batch and the CPU's share, summed in the CPU block's order).
 *   device, frame A2 (not waited for): the shared expert (the input to E4M3, w1 and w3,
 *     V4's SwiGLU between bf16 roundings, the input to E4M3, w2, bf16).
 * The decode attention cell (COLI_VK_CHAIN_ATTN_CELL, default off) is the fork's cell
 * discipline on that standing structure, a decode forward of one row only: the layer's
 * A2 record+submit moves to UNDER the cell's run -- right after A1's submit, while the
 * device still computes the attention -- and the settle narrows from vkc_finish (every
 * frame in flight) to the cell's own fence (vkc_wait_serial): one fence settle a
 * layer-token, the shared leg's record and submit leaving the critical path between the
 * attention and the routed step it overlaps. Nothing else moves: the mHC head and post
 * tails ride the same frame either way (with COLI_VK_CHAIN_MHC_S1 they are one dispatch
 * each), A2 still lands unwaited so the device keeps running it through the host's
 * routed step (measured at the W-CS operating point: A2's device time is hidden under
 * the host-MoE span, which is why folding it into A1's waited frame would move it onto
 * the critical path -- the lever does not do that), and every frame's barrier discipline
 * is the ring's own. Multi-row forwards (a prompt chunk, a verify), the layer-outer
 * order and COLI_VK_DMOE's in-frame layers keep today's A1/A2 exactly; the KV split's
 * staged readback still settles its own submit inside the attention where it must (the
 * cell's fence is then the frame the h2d copy rode).
 * The final streams come back to the host, which runs final_hidden and the head as
 * before; DSpark's taps of the last three layers' streams too (v4_mainh_tap).
 *
 * State, and who owns it: the host's stays canonical. What a forward changes there (the
 * window ring, the compressed rows and their count, each compressor's ring, the
 * indexer's keys and count and its compressor's ring) is written back once the
 * forward's last frame is through, as the CPU would have left it, so a lost device, a
 * rejected draft (spec_attention_restore), a prefix checkpoint, a reset or a CPU forward
 * never has to rebuild anything. The device mirrors it: the window ring holds window +
 * chunk rows (no row of a chunk overwrites one an earlier row still reads) behind a
 * watermark win_valid, a row below it valid unless a later position took its slot (a
 * rejected draft longer than a chunk does: slot_pos says); the compressed rows and the
 * indexer's keys of each layer behind kv_valid / ik_valid; each ring behind a flag.
 * Every host write lowers them: a CPU
 * forward (v4c_cpu_step), a restored snapshot (v4c_restored), a reset or a checkpoint
 * restore (v4c_reset), a forward that does not continue where the chain left off, an
 * attention state the mirrors do not describe. A lost device (COLI_VK_CHAIN_FAULT=n
 * fakes it): the forward runs again on the CPU from its input, which the chain leaves
 * untouched until every chunk is through, and the CPU runs from there.
 *
 * Past the device's budget (vk_kvsplit.h) each compressed layer keeps only some blocks of
 * its compressed rows on the device: a window over the newest rows and the blocks its
 * list reads most; the rest stay in the host's compressed rows, which hold every row. The
 * sparse attention stages only selected missing rows in bounded device scratch and
 * reads them alongside resident rows in the original list order (vkc_kv_ds). This
 * preserves one global score maximum and V4's bf16 weights and output exactly. A
 * chunk's new compressed rows go through a scratch into their slots and come back into
 * the host's rows (reserved for the forward at its start) as soon as their layer's frame
 * is through, so the host part of a later chunk finds every row it reads; the count is
 * set at the end as before. The window ring and the indexer's keys stay whole on the
 * device. Below the budget the mirrors are whole, as above.
 *
 * The chain declines (the CPU runs, the watermarks follow) when the dense layers are not
 * resident (a low-memory plan reloads them per forward; the --oracle path's own copies
 * too), under the CUDA tier, for prompts only when COLI_VK_CHAIN=2 and the forward has
 * two rows or fewer, for a forward whose compressed rows (window plus every row of a
 * layer without an indexer) pass the attention's 3072 entries, and for a model its
 * shaders do not take: head_dim above 1024, an indexer head that is not a power of two
 * (the Hadamard transform) or above 4096 floats, a top-k above 4096, more than 8
 * streams.
 *
 * Layers on two devices (docs/vulkan.md, "Layers on two devices"): with COLI_VK_DEV2 a
 * second chain takes the layers after the primary's on that device (its own fit from
 * layer N, COLI_VK_CHAIN_LAYERS2; its own copies of their matrices, in a table of their
 * own: the per-matrix path's map stays the primary's); the head stays on the host. A
 * layer reads only its own state, so only the streams cross: a forward runs the
 * primary's layers, then the second device's from the streams it hands over. A second
 * device that is lost, or declines a forward, leaves the CPU to run its layers from
 * there (the host's state is current: nothing to rebuild); lost, both chains go off. */
#include "vk_chain.h"
#include "vk_kvsplit.h"
#include "vk_dmoe.h"

typedef struct {
    int ratio, idx, kind;                       /* compress ratio (0 none), an indexer, its list */
    int crows, cproj;                           /* the compressor's ring: rows and floats a row */
    size_t o_an, o_fn, o_qn, o_kn, o_sink, o_hca, o_hcf, o_cn, o_ape, o_icn, o_iape;
    ColiVkTensor *fna, *fnf, *wq_a, *wq_b, *wkv, *wo_b, *sh1, *sh2, *sh3, **wo_a;
    ColiVkTensor *cwkv, *cwg, *iwkv, *iwg, *iwq, *iwp;
    VkcBuf *win, *ckv, *ring, *ikey, *iring;
    int ccap, kv_valid, ring_ok, icap, ik_valid, iring_ok;
    int built;                                  /* its tensors and state reached the device */
    size_t pw, pc, pr, pk, pir;                 /* this forward's places in the pull buffer */
    int ns, q1;                                 /* the ring rows it pulls: all (the overlap), or the slots of q1.. */
} V4cLayer;

typedef struct {
    int ok, failed, told;
    ColiV4Engine *engine;
    int Lm;                                     /* the model's layers; lo..L-1 below: the chain's (the primary: its first L) */
    int lo, d;                                  /* its first layer, its device (vkc_device) */
    int L, D, H, HD, nm, hr, nh, hd, rd, QL, og, ol, gw, W, Wd, IH, ID, I, K, rmax, rows, nkind;
    int kratio[COLI_V4_MAX_LAYERS + 1];          /* the ratio of each list kind (0: the indexer's) */
    V4cLayer *ly;
    VkcBuf *prm;
    ColiDeepSeekV4WindowAttentionState **attn;  /* the states the mirrors describe */
    int E, win_valid;                           /* the position after the last chained forward */
    int *slot_pos;                              /* the position each device ring row holds (-1 none) */
    size_t wcap;                                /* score columns a row */
    VkcBuf *xs, *xn, *mix, *hpa, *hpf, *col, *nrm, *nq, *qa, *qr, *qrq, *q, *kv, *heads, *grp, *br;
    VkcBuf *craw, *cscr, *icraw, *icscr, *iq, *ihw, *isc, *list, *cs, *sq, *gs, *us, *hs, *ds;
    VkcBuf *h2d, *routed, *xd, *taps, *pull;
    VkcBuf *lg, *gbias;                     /* COLI_VK_DMOE: the router's logits, its gate bias */
    VkcDmoe dm;                             /* that arm: one layer's routed experts in the frame */
    ColiVkTensor *gate;                     /* the layer's ffn.gate.weight, bf16 on the device */
    const float *gate_bias;                 /* and its ffn.gate.bias, written in every armed frame */
    uint8_t dmar[COLI_V4_MAX_LAYERS];       /* COLI_VK_DMOE's armed layers (all's lo..L mask; a layer without is the host's step) */
    VkcBuf *xall;                           /* COLI_VK_CHAIN_ORDER=layer: the prompt's streams, resident across a layer's pass */
    unsigned long long layer_fwd, layer_units;   /* the layer-ordered prompt forwards, their layer-chunk units (the report's line) */
    float *host_routed;
    unsigned long long forwards;
    unsigned long long cell_steps;               /* COLI_VK_CHAIN_ATTN_CELL: decode layer-tokens through the cell (the report's line) */
    double host_ms;
    VkcKvSplit ks;                              /* the compressed rows split past the device's budget (ks.on) */
    int *sli, nsl, rmin, planned;               /* per layer its table (-1 none), tables, the least ratio */
    VkcBuf *cnew, *cdn;                         /* a chunk's new compressed rows, and their copy for the host */
} V4Chain;

static V4Chain *g_v4c;
static int g_v4c_mode = 0;                      /* COLI_VK_CHAIN as decided */
static int g_v4c_inited = 0;
/* the second device's chain (layers g_v4c_fit.n..) and its fit */
static V4Chain *g_v4c2;
static VkcFit g_v4c_fit2;
static int g_v4c_fit2_on;
static V4Chain *v4c_of(int d) { return d ? g_v4c2 : g_v4c; }
static const char *v4c_name(const V4Chain *ch) { return ch && ch->d ? "deepseek_v4 dev2" : "deepseek_v4"; }
static int v4c_dev2_wanted(void) { const char *e = getenv("COLI_VK_CHAIN_DEV2"); return !(e && *e == '0'); }
/* The second device's copies, by their host data (the per-matrix path's map is the
 * primary's): an open table, grown at half full. */
typedef struct { const void *k; ColiVkTensor *t; } V4cD2;
static V4cD2 *g_v4c_d2;
static size_t g_v4c_d2cap, g_v4c_d2n;
static ColiVkTensor **v4c_d2_slot(const void *k) {
    if ((g_v4c_d2n + 1) * 2 > g_v4c_d2cap) {
        size_t nc = g_v4c_d2cap ? g_v4c_d2cap * 2 : 256;
        V4cD2 *nt = calloc(nc, sizeof *nt);
        if (!nt) return NULL;
        for (size_t i = 0; i < g_v4c_d2cap; i++) {
            if (!g_v4c_d2[i].k) continue;
            size_t h = ((uintptr_t)g_v4c_d2[i].k >> 4) & (nc - 1);
            while (nt[h].k) h = (h + 1) & (nc - 1);
            nt[h] = g_v4c_d2[i];
        }
        free(g_v4c_d2); g_v4c_d2 = nt; g_v4c_d2cap = nc;
    }
    size_t h = ((uintptr_t)k >> 4) & (g_v4c_d2cap - 1);
    while (g_v4c_d2[h].k && g_v4c_d2[h].k != k) h = (h + 1) & (g_v4c_d2cap - 1);
    if (!g_v4c_d2[h].k) { g_v4c_d2[h].k = k; g_v4c_d2n++; }
    return &g_v4c_d2[h].t;
}
static void v4c_d2_forget(ColiVkTensor *t) {   /* its copy freed; the key stays, empty */
    if (!t) return;
    for (size_t i = 0; i < g_v4c_d2cap; i++) if (g_v4c_d2[i].t == t) g_v4c_d2[i].t = NULL;
    coli_vk_tensor_free(t);
}

/* Rows per chunk with COLI_VK_CHAIN_ROWS set (or V4_PREFILL_CHUNK): the CPU's prefill
 * chunk (V4_PREFILL_CHUNK, 128 at most), so the host's MoE sees the batches the CPU block
 * would; COLI_VK_CHAIN_ROWS lowers it. Unset (the chunk from the budget, vkc_chunk_auto):
 * up to 8192 rows (v4c_chunk_rows), the MoE step taking the whole chunk; this is then the
 * window rings' first size, grown by the forward that needs more (v4c_grow_win). */
static int v4c_rows(void) {
    static int rows;
    if (!rows) {
        const char *c = getenv("V4_PREFILL_CHUNK"), *e = getenv("COLI_VK_CHAIN_ROWS");
        int cw = c ? atoi(c) : 128;
        if (cw < 1 || cw > 128) cw = 128;
        int v = e && *e && strcmp(e, "auto") ? atoi(e) : cw;
        rows = v < 1 ? 1 : v > cw ? cw : v;
    }
    return rows;
}
/* COLI_VK_CHAIN_ORDER: the prompt loop's order. `chunk` (the default, unset or any other
 * value) is today's -- a chunk of rows through every layer, then the next chunk; the
 * default-off path is byte-identical by construction (this parse only picks a loop).
 * `layer` is the fork's prefill order: every chunk of the prompt crosses one layer
 * before the next layer begins, so the tier's stream window (or the bank's rotation)
 * serves a layer's whole tenure and the carve the chunk leg measured cannot happen --
 * the reorder is the regime where S1's bank stops coexisting with streaming and becomes
 * it. The per-row GEMV invariant (COLI_VK_CHAIN_GEMM_ROWS' comment above the loop) makes
 * the reorder bit-neutral at GEMM_ROWS=0: the same (layer, chunk) batches run, only their
 * order across the forward changes. At GEMM_ROWS=-1 the tiled path's bits depend on the
 * chunk cut, and the cut sequence is pinned identical per row (the same v4c_chunk_rows
 * for every layer), so the reorder still changes only the serving path. */
static int v4c_order_layer(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_CHAIN_ORDER"); on = e && !strcmp(e, "layer"); }
    return on;
}
/* COLI_VK_CHAIN_ATTN_CELL: the decode attention cell (this file's header comment, the
 * A1 site in the chunk loop). A one-row forward's layer settles ONE fence a
 * layer-token -- the cell frame's own -- with A2's record+submit riding the cell's
 * device run; unset keeps the standing A1/A2 structure byte-identical (the branch is
 * not even taken). */
static int v4c_cell_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_CHAIN_ATTN_CELL"); on = e && *e && *e != '0'; }
    return on;
}
/* v4c_res counts instead of reserving while g_v4c_count >= 0 (the chunk's sizing) */
static long long g_v4c_count = -1;
static int v4c_res(VkcBuf **b, size_t floats, int kind) {
    if (g_v4c_count >= 0) { g_v4c_count += (long long)(floats ? floats : 1) * (long long)sizeof(float); return 1; }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}
static const void *v4c_val(const ColiDeepSeekV4LayerWeights *w, const char *suffix) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof name, "layers.%d.%s", w->plan.layer, suffix);
    return coli_v4_layer_data(w, name, NULL);
}

/* A resident matrix's device copy, the per-matrix path's own (v4_vk_matmul_impl's map
 * and layout): fp8 as fmt 12 (rows unpacked from the rows8 tiles, each 128x128 block
 * scale written out for its rows), bf16 as fmt 11, f32 as fmt 10 (the mHC mixes, which
 * only the chain multiplies). */
static ColiVkTensor *v4c_tensor(int fmt, const void *data, const float *scales, int rows8, int rows, int columns) {
    if (!g_v4_vk_ready || !data || rows < 1 || columns < 1) return NULL;
    if (fmt == 12 && (!scales || columns % 128 || (rows8 && rows % 8))) return NULL;
    int d2 = vkc_device_now();   /* the second device's chain: its own copy there */
    ColiVkTensor **slot = NULL;
    V4VkEntry *e = NULL;
    if (d2) {
        if (!(slot = v4c_d2_slot(data))) return NULL;
        if (*slot) return *slot;
    } else {
        e = v4_vk_find(data, fmt, rows, columns);
        if (e && e->tensor) return e->tensor;
        if (e && e->refused) return NULL;
    }
    size_t bytes = (size_t)rows * columns * (fmt == 10 ? 4 : fmt == 11 ? 2 : 1);
    if (!v4_vk_resident(data, bytes)) return NULL;
    coli_v4_vk_host_ensure(data);   /* a view the placement did not make (device only): its rows back first */
    if (!d2 && !e && !(e = v4_vk_insert(data, fmt, rows, columns))) return NULL;
    const unsigned char *weights = data;
    unsigned char *unpacked = NULL;
    float *expanded = NULL;
    if (fmt == 12) {
        int groups = columns / 128;
        expanded = malloc((size_t)rows * groups * sizeof(float));
        if (rows8) unpacked = malloc((size_t)rows * columns);
        if (!expanded || (rows8 && !unpacked)) { free(expanded); free(unpacked); return NULL; }
        for (int row = 0; row < rows; row++)
            memcpy(expanded + (size_t)row * groups, scales + (size_t)(row / 128) * groups, (size_t)groups * sizeof(float));
        if (rows8) {
            const unsigned char *packed = data;
            for (int tile = 0; tile < rows / 8; tile++)
                for (int column = 0; column < columns; column++)
                    for (int lane = 0; lane < 8; lane++)
                        unpacked[((size_t)tile * 8 + lane) * columns + column] = packed[((size_t)tile * columns + column) * 8 + lane];
            weights = unpacked;
        }
    }
    if (d2) {
        int ok = coli_vk_tensor_ensure2(slot, weights, expanded, fmt, columns, rows, fmt == 12 ? 128 : 0);
        free(unpacked); free(expanded);
        return ok ? *slot : NULL;
    }
    int ok = coli_vk_tensor_ensure(&e->tensor, weights, expanded, fmt, columns, rows, fmt == 12 ? 128 : 0);
    free(unpacked); free(expanded);
    if (!ok && !e->tensor) e->refused = 1;
    return ok ? e->tensor : NULL;
}
/* A GGUF's matrix (its dtype a ggml view) on the device: its blocks as the file has them
 * (100 + the type; rows the loader interleaved by 16 go back to the file's), a float view as
 * the loader stored it (f32 10, bf16 11, f16 14). For the grouped output's per-head reader
 * (vkc_mla_hgemv, which has no block form) Q8_0 goes up as int8 with a scale per 32 (fmt 13,
 * the same numbers) and any other type as floats. -1: no device form. */
static int v4c_gq_fmt(int dtype, int I, int hgemv) {
    const int t = st_ggml_type(dtype), b = gq_r16_base(t), base = b >= 0 ? b : t;
    if (base == GQ_F32) return 10;
    if (base == GQ_BF16) return 11;
    if (base == GQ_F16) return hgemv ? 10 : 14;
    if (hgemv) return base == GQ_Q8_0 && I % 32 == 0 ? 13 : 10;
    const int be = coli_vk_block_elems(100 + base);
    return be > 0 && I % be == 0 ? 100 + base : -1;
}
/* v4c_tensor for a GGUF's matrix in v4c_gq_fmt's form `fmt`: the same per-matrix map (or the
 * second device's slot), its rows converted as that form wants them. */
static ColiVkTensor *v4c_gq_tensor(int fmt, int dtype, const unsigned char *data, int rows, int columns) {
    if (!g_v4_vk_ready || !data || fmt < 0 || rows < 1 || columns < 1) return NULL;
    const int t = st_ggml_type(dtype), b = gq_r16_base(t), base = b >= 0 ? b : t;
    const int fl = base == GQ_F32 || base == GQ_F16 || base == GQ_BF16;   /* a float view, stored as that float */
    const int64_t held = fl ? (int64_t)columns * (base == GQ_F32 ? 4 : 2) : gq_row_bytes(t, columns),
                  rb = fl ? held : gq_row_bytes(base, columns);
    if (held <= 0 || rb <= 0 || (b >= 0 && rows % 16)) return NULL;
    int d2 = vkc_device_now();
    ColiVkTensor **slot = NULL;
    V4VkEntry *e = NULL;
    if (d2) {
        if (!(slot = v4c_d2_slot(data))) return NULL;
        if (*slot) return *slot;
    } else {
        e = v4_vk_find(data, fmt, rows, columns);
        if (e && e->tensor) return e->tensor;
        if (e && e->refused) return NULL;
    }
    if (!v4_vk_resident(data, (size_t)held * rows)) return NULL;
    coli_v4_vk_host_ensure(data);
    if (!d2 && !e && !(e = v4_vk_insert(data, fmt, rows, columns))) return NULL;
    const unsigned char *src = data;
    unsigned char *file = NULL;   /* the file's rows of an interleaved matrix */
    void *conv = NULL;            /* fmt 13's codes or fmt 10's floats */
    float *sc = NULL;
    int gs = 0, ok = 1;
    if (b >= 0) {
        if (!(file = malloc((size_t)rb * rows))) ok = 0;
        else { gq_r16_unpack(t, file, data, rows, columns); src = file; }
    }
    const void *weights = src;
    if (ok && fmt == 13) {   /* Q8_0 (a half scale, 32 int8 codes a block): codes and scales apart */
        const int nb = columns / 32;
        int8_t *q = malloc((size_t)rows * columns);
        sc = malloc((size_t)rows * nb * sizeof(float));
        if (!q || !sc) { free(q); ok = 0; }
        else {
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < nb; k++) {
                    const unsigned char *blk = src + (size_t)r * rb + (size_t)k * 34;
                    uint16_t h; memcpy(&h, blk, 2);
                    sc[(size_t)r * nb + k] = gq_f16(h);
                    memcpy(q + (size_t)r * columns + (size_t)k * 32, blk + 2, 32);
                }
            weights = conv = q; gs = 32;
        }
    } else if (ok && fmt == 10 && base != GQ_F32) {   /* the per-head reader's floats */
        float *f = malloc((size_t)rows * columns * sizeof(float));
        if (!f) ok = 0;
        else {
            for (int r = 0; r < rows; r++) {
                const unsigned char *row = src + (size_t)r * rb;
                float *o = f + (size_t)r * columns;
                if (base == GQ_F16) for (int c = 0; c < columns; c++) { uint16_t h; memcpy(&h, row + 2 * c, 2); o[c] = gq_f16(h); }
                else if (base == GQ_BF16) for (int c = 0; c < columns; c++) { uint16_t h; memcpy(&h, row + 2 * c, 2); uint32_t u = (uint32_t)h << 16; memcpy(&o[c], &u, 4); }
                else gq_dequant_row(base, row, o, columns);
            }
            weights = conv = f;
        }
    }
    if (ok) ok = d2 ? coli_vk_tensor_ensure2(slot, weights, sc, fmt, columns, rows, gs)
                    : coli_vk_tensor_ensure(&e->tensor, weights, sc, fmt, columns, rows, gs);
    free(file); free(conv); free(sc);
    if (d2) return ok ? *slot : NULL;
    if (!ok && !e->tensor) e->refused = 1;
    return ok ? e->tensor : NULL;
}
/* an fp8 matrix "<prefix>.weight" with its ".scale", a group of rows of it when rows > 0; a
 * GGUF's in its own types (v4c_gq_fmt: the group's rows for the per-head reader) */
static char g_v4c_refused[COLI_V4_MAX_TENSOR_NAME + 64];   /* the last matrix the device did not take, and why */
static ColiVkTensor *v4c_refused(const char *name, int dtype, int fmt, ColiVkTensor *t) {
    if (!t) snprintf(g_v4c_refused, sizeof g_v4c_refused, "%s (%s, %s)", name,
                     st_is_ggml(dtype) ? gq_type_name(st_ggml_type(dtype)) : "the checkpoint's",
                     fmt < 0 ? "no device form" : "the upload failed");
    return t;
}
static ColiVkTensor *v4c_fp8(const ColiDeepSeekV4LayerWeights *w, const char *prefix, int group, int rows) {
    char name[COLI_V4_MAX_TENSOR_NAME], sname[COLI_V4_MAX_TENSOR_NAME];
    const ColiDeepSeekV4TensorSpec *ws = NULL;
    snprintf(name, sizeof name, "layers.%d.%s.weight", w->plan.layer, prefix);
    snprintf(sname, sizeof sname, "layers.%d.%s.scale", w->plan.layer, prefix);
    const unsigned char *data = coli_v4_layer_data(w, name, &ws);
    if (data && ws && ws->rank == 2 && st_is_ggml(ws->dtype)) {   /* a GGUF's: no .scale beside it */
        const int O = (int)ws->shape[0], I = (int)ws->shape[1], f = v4c_gq_fmt(ws->dtype, I, rows > 0);
        if (f < 0) return v4c_refused(name, ws->dtype, f, NULL);
        if (rows <= 0) return v4c_refused(name, ws->dtype, f, v4c_gq_tensor(f, ws->dtype, data, O, I));
        const int t = st_ggml_type(ws->dtype), fl = t == GQ_F32 || t == GQ_F16 || t == GQ_BF16;
        const int64_t rb = fl ? (int64_t)I * (t == GQ_F32 ? 4 : 2) : gq_row_bytes(t, I);
        if (rb <= 0 || (int64_t)(group + 1) * rows > O) return v4c_refused(name, ws->dtype, -1, NULL);
        return v4c_refused(name, ws->dtype, f, v4c_gq_tensor(f, ws->dtype, data + (size_t)group * rows * rb, rows, I));
    }
    const float *scales = coli_v4_layer_data(w, sname, NULL);
    if (!data || !scales || !ws || ws->rank != 2 || ws->dtype != COLI_ST_F8_E4M3) return NULL;
    int O = (int)ws->shape[0], I = (int)ws->shape[1];
    if (rows <= 0) return v4c_tensor(12, data, scales, ws->packed_rows8, O, I);
    /* the CPU's group view (attention_token_impl): rows group*rows.., its own scale rows */
    int scale_columns = (I + 127) / 128, scale_rows = (rows + 127) / 128;
    return v4c_tensor(12, data + (size_t)group * rows * I, scales + (size_t)group * scale_rows * scale_columns,
                      ws->packed_rows8, rows, I);
}
static ColiVkTensor *v4c_plain(const ColiDeepSeekV4LayerWeights *w, const char *suffix, int fmt) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    const ColiDeepSeekV4TensorSpec *ws = NULL;
    snprintf(name, sizeof name, "layers.%d.%s", w->plan.layer, suffix);
    const void *data = coli_v4_layer_data(w, name, &ws);
    if (!data || !ws || ws->rank != 2) return NULL;
    if (st_is_ggml(ws->dtype)) {   /* a GGUF's, in its own type */
        const int f = v4c_gq_fmt(ws->dtype, (int)ws->shape[1], 0);
        return v4c_refused(name, ws->dtype, f, v4c_gq_tensor(f, ws->dtype, data, (int)ws->shape[0], (int)ws->shape[1]));
    }
    return v4c_tensor(fmt, data, NULL, 0, (int)ws->shape[0], (int)ws->shape[1]);
}

/* ---- the host's writes: the device's copies follow ------------------------------------- */
static void v4c_lower_one(V4Chain *ch, int pos) {
    if (!ch || !ch->ok) return;
    if (pos < 0) pos = 0;
    if (ch->win_valid > pos) ch->win_valid = pos;
    for (int i = ch->lo; i < ch->L; i++) {
        V4cLayer *ly = &ch->ly[i];
        if (ly->ratio > 0 && ly->kv_valid > pos / ly->ratio) ly->kv_valid = pos / ly->ratio;
        if (ly->ratio > 0 && ch->sli[i] >= 0) vkc_kv_lower(&ch->ks, ch->sli[i], pos / ly->ratio);
        if (ly->idx && ly->ik_valid > pos / 4) ly->ik_valid = pos / 4;
        ly->ring_ok = ly->iring_ok = 0;
    }
    ch->E = -1;
}
static void v4c_lower(int pos) { for (int d = 0; d < 2; d++) v4c_lower_one(v4c_of(d), pos); }   /* both chains */
static void v4c_cpu_step(int start) { v4c_lower(start); }     /* a CPU forward from start */
static void v4c_restored(int pos) { v4c_lower(pos); }         /* the state restored to pos, the rows before it kept */
static void v4c_reset(void) { v4c_lower(0); }                 /* a reset, or a state from elsewhere */

/* ---- setup ----------------------------------------------------------------------------- */
static const char *v4c_unsupported(const ColiDeepSeekV4Config *c) {
    int any_idx = 0;
    for (int i = 0; i < c->num_hidden_layers; i++) any_idx |= c->compress_ratios[i] == 4;
    if (c->head_dim > 1024 || c->head_dim < 2 || c->qk_rope_head_dim < 2 || (c->qk_rope_head_dim & 1) ||
        c->qk_rope_head_dim > c->head_dim) return "an attention geometry its shaders do not take";
    if (c->hc_mult < 1 || c->hc_mult > 8 || c->hc_sinkhorn_iters < 1) return "hyper-connections its shaders do not take";
    if (c->o_groups < 1 || c->num_attention_heads % c->o_groups) return "an output grouping it does not take";
    if (c->sliding_window < 1 || c->sliding_window > 3072) return "a window its attention does not take";
    if (any_idx && (c->index_n_heads < 1 || c->index_n_heads > 4096 || c->index_head_dim > 4096 ||
                    c->index_head_dim < c->qk_rope_head_dim || (c->index_head_dim & (c->index_head_dim - 1)) ||
                    c->index_topk < 1 || c->index_topk > 4096 || c->sliding_window + c->index_topk > 3072))
        return "an indexer its shaders do not take";
    for (int i = 0; i < c->num_hidden_layers; i++)
        if (c->compress_ratios[i] < 0) return "a compression ratio it does not take";
    return NULL;
}

/* ---- the partial chain (vk_chain.h, vkc_fit): the first N layers on the device ---------- */
static VkcFit g_v4c_fit;
static int g_v4c_fit_done, g_v4c_building;

/* The chain's geometry for its layers lo..L-1 (nothing allocated): the per-layer kinds,
 * the split's tables, the list kinds. Pointers are left alone, so it runs again when a
 * layer that did not reach the device lowers L. */
static void v4c_geom(V4Chain *ch, ColiV4Engine *engine, int lo, int L) {
    const ColiDeepSeekV4Config *c = &engine->config;
    ch->engine = engine; ch->Lm = c->num_hidden_layers; ch->lo = lo; ch->L = L;
    ch->D = c->hidden_size; ch->H = c->hc_mult; ch->HD = ch->H * ch->D;
    ch->nm = (2 + ch->H) * ch->H; ch->hr = 2 * ch->H + ch->H * ch->H;
    ch->nh = c->num_attention_heads; ch->hd = c->head_dim; ch->rd = c->qk_rope_head_dim; ch->QL = c->q_lora_rank;
    ch->og = c->o_groups; ch->ol = c->o_lora_rank; ch->gw = (ch->nh / ch->og) * ch->hd;
    ch->W = c->sliding_window;
    ch->IH = c->index_n_heads > 0 ? c->index_n_heads : 1; ch->ID = c->index_head_dim > 0 ? c->index_head_dim : 2;
    ch->I = c->moe_intermediate_size; ch->rmax = 1; ch->nkind = 1; ch->nsl = 0; ch->rmin = 0;
    for (int i = 0; i < ch->Lm; i++) ch->sli[i] = -1;
    for (int i = lo; i < L; i++) {
        V4cLayer *ly = &ch->ly[i];
        ly->ratio = c->compress_ratios[i]; ly->idx = ly->ratio == 4; ly->kind = 0;
        if (ly->ratio > ch->rmax) ch->rmax = ly->ratio;
        ly->crows = ly->ratio == 4 ? 8 : ly->ratio; ly->cproj = (ly->ratio == 4 ? 2 : 1) * ch->hd;
        ch->sli[i] = ly->ratio > 0 ? ch->nsl++ : -1;    /* the split's tables: one per compressed layer */
        if (ly->ratio > 0 && (ch->rmin == 0 || ly->ratio < ch->rmin)) ch->rmin = ly->ratio;
        if (ly->ratio > 0 && !ly->idx) {
            int k = 1;
            while (k < ch->nkind && ch->kratio[k] != ly->ratio) k++;
            if (k == ch->nkind) ch->kratio[ch->nkind++] = ly->ratio;
            ly->kind = k;
        }
    }
}
/* The parameter arena's offsets for the chain's layers: norms decoded from bf16, sinks,
 * hc scales and bases, the rings' biases. Returns its floats. */
static size_t v4c_offsets(V4Chain *ch) {
    size_t n = 0;
    for (int i = ch->lo; i < ch->L; i++) {
        V4cLayer *ly = &ch->ly[i];
        ly->o_an = n; n += ch->D; ly->o_fn = n; n += ch->D; ly->o_qn = n; n += ch->QL; ly->o_kn = n; n += ch->hd;
        ly->o_sink = n; n += ch->nh; ly->o_hca = n; n += 3 + ch->nm; ly->o_hcf = n; n += 3 + ch->nm;
        if (ly->ratio > 0) { ly->o_cn = n; n += ch->hd; ly->o_ape = n; n += (size_t)ly->ratio * ly->cproj; }
        if (ly->idx) { ly->o_icn = n; n += ch->ID; ly->o_iape = n; n += (size_t)4 * 2 * ch->ID; }
    }
    return n;
}
/* What layer i puts on the device, as vkc_fit counts it: its matrices (the chain's set,
 * which holds the device-only placement's), its state at its first size (the window
 * ring, the compressors' rings, 64 compressed rows and keys), its part of the parameters
 * and of a forward's pull buffer at `rows` rows; *mat its matrices' payload. */
static size_t v4c_layer_bytes(const V4Chain *ch, int i, int rows, size_t *mat) {
    const ColiDeepSeekV4Config *c = &ch->engine->config;
    ColiDeepSeekV4LayerPlan plan;
    size_t b = 0, m = 0;
    *mat = 0;
    if (coli_v4_layer_plan(&plan, c, i, NULL, 0)) return 0;
    const ColiSafetensorsIndex *ix = coli_v4_engine_target_index(ch->engine);
    if (ix && coli_v4_index_gguf(ix)) coli_v4_layer_plan_index(&plan, ix);   /* a GGUF's types */
    for (size_t t = 0; t < plan.tensor_count; t++) {
        const ColiDeepSeekV4TensorSpec *sp = &plan.tensors[t];
        if (sp->rank != 2) continue;
        const char *suf = sp->name;
        if (!strncmp(suf, "layers.", 7)) { const char *d = strchr(suf + 7, '.'); suf = d ? d + 1 : suf; }
        int O = (int)sp->shape[0], I = (int)sp->shape[1], fmt = 0, parts = 1, gs = 0;
        if (st_is_ggml(sp->dtype)) {   /* a GGUF's: the chain's matrices in v4c_gq_fmt's forms */
            static const char *const mats[] = {
                "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight", "attn.wo_a.weight", "attn.wo_b.weight",
                "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight",
                "attn.compressor.wkv.weight", "attn.compressor.wgate.weight", "attn.indexer.compressor.wkv.weight",
                "attn.indexer.compressor.wgate.weight", "attn.indexer.weights_proj.weight", "attn.indexer.wq_b.weight",
                "hc_attn_fn", "hc_ffn_fn"};
            int chained = 0;
            for (size_t k = 0; k < sizeof mats / sizeof *mats && !chained; k++) chained = !strcmp(suf, mats[k]);
            if (!chained) continue;
            const int hg = !strcmp(suf, "attn.wo_a.weight");
            if (hg) { parts = ch->og; O = ch->ol; }
            if ((fmt = v4c_gq_fmt(sp->dtype, I, hg)) < 0) continue;
            gs = fmt == 13 ? 32 : 0;
            for (int p = 0; p < parts; p++) {
                b += vkc_fit_tensor(fmt, I, O, gs);
                m += coli_vk_tensor_payload(fmt, I, O, gs);
            }
            continue;
        }
        if (sp->dtype == COLI_ST_F8_E4M3) {
            fmt = 12;
            if (!strcmp(suf, "attn.wo_a.weight")) { parts = ch->og; O = ch->ol; }   /* one tensor per output group */
        } else if (sp->dtype == COLI_ST_BF16 &&
                   (coli_v4_dense_device_only_tensor(sp) || !strcmp(suf, "attn.indexer.weights_proj.weight"))) fmt = 11;
        else if (sp->dtype == COLI_ST_F32 && (!strcmp(suf, "hc_attn_fn") || !strcmp(suf, "hc_ffn_fn"))) fmt = 10;
        if (!fmt) continue;
        for (int p = 0; p < parts; p++) {
            b += vkc_fit_tensor(fmt, I, O, fmt == 12 ? 128 : 0);
            m += coli_vk_tensor_payload(fmt, I, O, fmt == 12 ? 128 : 0);
        }
    }
    const V4cLayer *ly = &ch->ly[i];
    int W = ch->W, hd = ch->hd, ID = ch->ID, E = rows;
    b += vkc_fit_buf((size_t)(W + v4c_rows()) * hd * sizeof(float));
    if (ly->ratio > 0) b += vkc_fit_buf((size_t)2 * ly->crows * ly->cproj * sizeof(float)) + vkc_fit_buf((size_t)64 * hd * sizeof(float));
    if (ly->idx) b += vkc_fit_buf((size_t)2 * 8 * 2 * ID * sizeof(float)) + vkc_fit_buf((size_t)64 * ID * sizeof(float));
    size_t f = (size_t)2 * ch->D + ch->QL + hd + ch->nh + (size_t)2 * (3 + ch->nm);
    if (ly->ratio > 0) f += hd + (size_t)ly->ratio * ly->cproj;
    if (ly->idx) f += ID + (size_t)8 * ID;
    f += (size_t)(E < W ? E : W) * hd;
    if (ly->ratio > 0) f += (size_t)(E / ly->ratio) * hd + (size_t)2 * (ly->ratio == 4 ? ly->crows : E < ly->ratio ? E : ly->ratio) * ly->cproj;
    if (ly->idx) f += (size_t)(E / 4) * ID + (size_t)2 * 8 * 2 * ID;
    *mat = m;
    return b + f * sizeof(float);
}
static int v4c_scratch(V4Chain *ch, int rows, int E, int LR);
/* What the chain allocates whatever N: the scratch of one forward of `rows` rows (the
 * counting pass of v4c_scratch, every layer's list kinds). */
static size_t v4c_fixed_bytes(V4Chain *ch, int rows) {
    const ColiDeepSeekV4Config *c = &ch->engine->config;
    int E = ch->W + rows, K = 0;
    for (int i = ch->lo; i < ch->L; i++) {
        V4cLayer *ly = &ch->ly[i];
        int k = ly->idx ? c->index_topk : ly->ratio > 0 ? E / ly->ratio : 0;
        if (k > K) K = k;
    }
    g_v4c_count = 0;
    v4c_scratch(ch, rows, E, ch->W + (K > 0 ? K : 1));
    size_t b = (size_t)g_v4c_count;
    g_v4c_count = -1;
    return b;
}
/* With every layer on the device, the per-matrix path (COLI_VK_DENSE) also puts the bf16
 * head and each layer's router there as they are first used: the tail. */
static size_t v4c_tail_bytes(const ColiV4Engine *engine) {
    if (!coli_vk_dense()) return 0;
    const ColiDeepSeekV4Config *c = &engine->config;
    return vkc_fit_tensor(11, c->hidden_size, c->vocab_size, 0) +
           (size_t)c->num_hidden_layers * vkc_fit_tensor(11, c->hidden_size, c->n_routed_experts, 0);
}
static const char *v4c_unsupported(const ColiDeepSeekV4Config *c);
static int v4c_decide(const ColiV4Engine *engine);
/* The decision (vkc_fit), once and before any upload: by the RAM plan when it asks about
 * the device-only weights, else at v4c_start. 0: no fit (the chain will not run). */
static int v4c_fit_now(ColiV4Engine *engine) {
    if (g_v4c_fit_done) return g_v4c_fit.L > 0;
    g_v4c_fit_done = 1;
    if (!g_v4_vk_ready || !engine || !v4c_decide(engine) || v4c_unsupported(&engine->config)) return 0;
    int L = engine->config.num_hidden_layers;
    V4Chain t;
    memset(&t, 0, sizeof t);
    t.ly = calloc((size_t)L, sizeof *t.ly); t.sli = malloc((size_t)L * sizeof(int));
    size_t *per = calloc((size_t)L, sizeof(size_t)), *mat = calloc((size_t)L, sizeof(size_t));
    int ok = t.ly && t.sli && per && mat;
    if (ok) {
        v4c_geom(&t, engine, 0, L);
        int rows = v4c_rows();
        for (int i = 0; i < L; i++) per[i] = v4c_layer_bytes(&t, i, rows, &mat[i]);
        size_t fixed = v4c_fixed_bytes(&t, rows);
        vkc_fit("deepseek_v4", L, per, mat, fixed, v4c_tail_bytes(engine), &g_v4c_fit);
        /* the layers the primary leaves, on COLI_VK_DEV2's device: a fit of its own from
         * layer n0 (the head stays on the host), with that device's free memory, its
         * pipelines up now */
        int n0 = g_v4c_fit.n;
        if (vkc_fit_partial(&g_v4c_fit) && n0 > 0 && n0 < L && v4c_dev2_wanted() && getenv("COLI_VK_DEV2") &&
            coli_vk_dev2_open_env()) {
            vkc_device(1);
            int n2 = vkc_fit("deepseek_v4 dev2", L - n0, per + n0, mat + n0, fixed, 0, &g_v4c_fit2);
            if (n2 > 0 && !(vkc_init() && vkc_mla_ready() && vkc_mhc_ready() && vkc_dsv4_ready())) {
                fprintf(stderr, "[VK] deepseek_v4 chain: the second device's pipelines did not come up; its layers stay on the CPU\n");
                n2 = 0;
            }
            g_v4c_fit2_on = n2 > 0;
            vkc_device(0);
        }
    }
    free(t.ly); free(t.sli); free(per); free(mat);
    return ok;
}
static int v4c_fit_n(void) { return g_v4c_fit.n; }
/* 1: something the full chain would place stays on the CPU (the CPU layers, the head):
 * the per-matrix path uploads nothing new. */
static int v4c_partial(void) { return g_v4c_fit_done && vkc_fit_partial(&g_v4c_fit); }
/* 1: the chain holds every layer and the head: nothing of the dense part goes up after it */
static int v4c_whole(void) { return g_v4c_fit_done && g_v4c_fit.n > 0 && !vkc_fit_partial(&g_v4c_fit); }

/* Layer i's matrices (the per-matrix path's device copies, one per matrix) and its state
 * on the device. 0 with *why set: something did not reach it (v4c_layer_free undoes it). */
static int v4c_layer_up(V4Chain *ch, int i, const char **why) {
    V4cLayer *ly = &ch->ly[i];
    const ColiDeepSeekV4LayerWeights *w = &ch->engine->dense_resident.layers[i];
    *why = "a matrix the device refused";
    ly->wo_a = calloc((size_t)ch->og, sizeof(void *));
    int ok = ly->wo_a && (ly->fna = v4c_plain(w, "hc_attn_fn", 10)) && (ly->fnf = v4c_plain(w, "hc_ffn_fn", 10)) &&
             (ly->wq_a = v4c_fp8(w, "attn.wq_a", 0, 0)) && (ly->wq_b = v4c_fp8(w, "attn.wq_b", 0, 0)) &&
             (ly->wkv = v4c_fp8(w, "attn.wkv", 0, 0)) && (ly->wo_b = v4c_fp8(w, "attn.wo_b", 0, 0)) &&
             (ly->sh1 = v4c_fp8(w, "ffn.shared_experts.w1", 0, 0)) && (ly->sh2 = v4c_fp8(w, "ffn.shared_experts.w2", 0, 0)) &&
             (ly->sh3 = v4c_fp8(w, "ffn.shared_experts.w3", 0, 0));
    for (int g = 0; ok && g < ch->og; g++) ok = (ly->wo_a[g] = v4c_fp8(w, "attn.wo_a", g, ch->ol)) != NULL;
    if (ok && ly->ratio > 0)
        ok = (ly->cwkv = v4c_plain(w, "attn.compressor.wkv.weight", 11)) && (ly->cwg = v4c_plain(w, "attn.compressor.wgate.weight", 11));
    if (ok && ly->idx)
        ok = (ly->iwkv = v4c_plain(w, "attn.indexer.compressor.wkv.weight", 11)) &&
             (ly->iwg = v4c_plain(w, "attn.indexer.compressor.wgate.weight", 11)) &&
             (ly->iwp = v4c_plain(w, "attn.indexer.weights_proj.weight", 11)) && (ly->iwq = v4c_fp8(w, "attn.indexer.wq_b", 0, 0));
    if (!ok) {
        static char msg[sizeof g_v4c_refused + 32];
        snprintf(msg, sizeof msg, "a matrix the device refused: %s", g_v4c_refused[0] ? g_v4c_refused : "?");
        *why = msg;
        return 0;
    }
    *why = "device memory for its state refused";
    if (ly->ratio > 0 && !(ly->ring = vkc_buf((size_t)2 * ly->crows * ly->cproj * sizeof(float), VKC_DEV))) return 0;
    if (ly->idx && !(ly->iring = vkc_buf((size_t)2 * 8 * 2 * ch->ID * sizeof(float), VKC_DEV))) return 0;
    return (ly->win = vkc_buf((size_t)ch->Wd * ch->hd * sizeof(float), VKC_DEV)) != NULL;
}
static void v4_vk_forget(ColiVkTensor *t);   /* the per-matrix map's entry: its tensor freed, the matrix the CPU's */
/* Everything of layer i off the device: its tensors (and their per-matrix entries, which
 * stay refused: the matrix is the CPU's) and its buffers, after the frames that read them. */
static void v4c_layer_free(V4Chain *ch, int i) {
    V4cLayer *ly = &ch->ly[i];
    if (vkc_ready()) vkc_finish();
    ColiVkTensor **ts[] = {&ly->fna, &ly->fnf, &ly->wq_a, &ly->wq_b, &ly->wkv, &ly->wo_b, &ly->sh1, &ly->sh2, &ly->sh3,
                           &ly->cwkv, &ly->cwg, &ly->iwkv, &ly->iwg, &ly->iwq, &ly->iwp};
    void (*forget)(ColiVkTensor *) = ch->d ? v4c_d2_forget : v4_vk_forget;   /* the second device's: its own */
    for (size_t k = 0; k < sizeof ts / sizeof *ts; k++) { forget(*ts[k]); *ts[k] = NULL; }
    for (int g = 0; ly->wo_a && g < ch->og; g++) forget(ly->wo_a[g]);
    free(ly->wo_a); ly->wo_a = NULL;
    VkcBuf **bs[] = {&ly->win, &ly->ckv, &ly->ring, &ly->ikey, &ly->iring};
    for (size_t k = 0; k < sizeof bs / sizeof *bs; k++) { vkc_free(*bs[k]); *bs[k] = NULL; }
    ly->ccap = ly->icap = ly->kv_valid = ly->ik_valid = ly->ring_ok = ly->iring_ok = 0;
    ly->built = 0;
}
/* COLI_VK_DMOE's arm (below) off: its buffers, after the frames that read them. */
static void v4c_dmoe_drop(V4Chain *ch) {
    if (vkc_ready()) vkc_finish();
    vkc_dmoe_free(&ch->dm);
    vkc_free(ch->lg); vkc_free(ch->gbias); ch->lg = ch->gbias = NULL;
    void (*forget)(ColiVkTensor *) = ch->d ? v4c_d2_forget : v4_vk_forget;
    forget(ch->gate); ch->gate = NULL; ch->gate_bias = NULL;
    memset(ch->dmar, 0, sizeof ch->dmar);
}
/* ---- COLI_VK_DMOE: one layer's routed experts in the frame (vk_dmoe.h) -------------------
 * Scaffolding for the L2 measurement (docs/research/Q-l2-dmoe-map-2026-10-22.md, S1):
 * with COLI_VK_DMOE=n and the tier holding all of layer n's experts on the primary
 * device (a warm start that pins them, COLI_VK_TIER_PIN_LAYER, over a COLI_VK_TIER_GB
 * that holds them), a step of up to VKC_DMOE_ROWS rows (decode, a short verify) routes
 * and computes that one layer's experts in the frame (vkc_route + vkc_moe) instead of
 * the host's coli_v4_moe_routed step -- the shared expert and the gate logits recorded
 * with them, no wait at the layer, the tier's books after the frame from the ids
 * (vkt_dev_count). The router's parameters are route_bf16_impl's: sqrt(softplus) of the
 * logit, the gate bias for the pick only, the weights score/sum * routed_scaling_factor.
 * The experts' sum is chain_moe.comp's rank order, not the CPU block's ascending expert
 * id, and the logits come from the chain's GEMV: a token's bits can differ where a bf16
 * round does not absorb it -- the greedy/oracle check is the gate, not bit equality. A
 * hash layer (vkc_route cannot express its token table), a layer outside this chain and
 * the second device keep the host's step. 0, the default, changes nothing.
 *
 * COLI_VK_DMOE=all is the multi-layer form the layer-outer loop (COLI_VK_CHAIN_ORDER=
 * layer, below) drives: one prepare over lo..L (vk_dmoe.h's own range), the per-layer
 * gate looked up at each visit (one gate tensor a layer -- the single-layer arm caches
 * its one, the loop cannot), hash layers and layers whose gate did not reach the device
 * masked to the host's step (ch->dmar). The arm needs the layer order: the chunk-outer
 * loop caches one gate for its one armed layer, so `all` without the reorder declines
 * with a line instead of computing with the wrong gate. With the bank's rotation
 * (COLI_VK_TIER_BANK_LAYER) each layer's tenure arms its own visits: the first chunk of
 * a layer runs the host's step (the fill seam refills the bank post-router), the rest
 * compute in the frame from the banked residency. */
/* COLI_VK_DMOE_ROWS: the opt-in that lets the armed layer take a prompt chunk's rows too
 * (more than VKC_DMOE_ROWS of them) through vkc_moe_rows' grouped GEMMs -- the fmt-7
 * (MXFP4) pipelines of L2 Stage B. Off (the default) the arm keeps the per-pair decode
 * path only, and a step of more than VKC_DMOE_ROWS rows takes the host's step as before.
 * A prompt row's routed experts are then chain_moe_rows.comp's rank-order sums over the
 * chunk's packed rows, so its bits can depend on how the prompt is chunked -- the same
 * invariant break COLI_VK_CHAIN_GEMM_ROWS documents above (its row too), and the reason
 * this stays opt-in. */
static int v4c_dmoe_rows_wanted(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("COLI_VK_DMOE_ROWS"); on = e && *e && atoi(e) != 0; }
    return on;
}
static int v4c_dmoe_arm(V4Chain *ch, int rows, int layer_order) {
    const ColiDeepSeekV4Config *c = &ch->engine->config;
    int n = vkc_dmoe_layer_env(), all = vkc_dmoe_all_env();
    static int told = 0;
    int big = v4c_dmoe_rows_wanted();   /* a prompt chunk's rows through the grouped GEMMs */
    if ((!n && !all) || ch->d || (!big && rows > VKC_DMOE_ROWS)) return 0;
    if (all) {   /* COLI_VK_DMOE=all: the chain's layers in one arm (the layer-outer loop's) */
        if (!layer_order) {
            if (!told) fprintf(stderr, "[VK] %s chain: COLI_VK_DMOE=all follows COLI_VK_CHAIN_ORDER=layer's prompt path "
                                       "(this forward is not one it takes: decode, a verify, a single chunk, or the reservoir "
                                       "did not fit); the host's step keeps every layer\n", v4c_name(ch));
            told = 1;
            return 0;
        }
        memset(ch->dmar, 0, sizeof ch->dmar);
        int armed = 0;
        for (int i = ch->lo; i < ch->L && i < COLI_V4_MAX_LAYERS; i++) {
            const ColiDeepSeekV4LayerWeights *w = &ch->engine->dense_resident.layers[i];
            if (w->plan.uses_hash_router) continue;   /* vkc_route cannot give its token table */
            if (!v4c_plain(w, "ffn.gate.weight", 11)) continue;   /* no gate on the device: the host's step */
            ch->dmar[i] = 1; armed++;
        }
        if (!armed) {
            if (!told) fprintf(stderr, "[VK] %s chain: COLI_VK_DMOE=all: no layer of %d..%d can route in the frame (hash "
                                       "layers, or gates off the device); the host's step keeps them\n", v4c_name(ch), ch->lo, ch->L - 1);
            told = 1;
            return 0;
        }
        if (!vkc_reserve(&ch->gbias, (size_t)c->n_routed_experts * sizeof(float), VKC_DEV) ||
            !vkc_reserve(&ch->lg, (size_t)rows * c->n_routed_experts * sizeof(float), VKC_DEV)) return 0;
        ch->dm.big = big;
        if (!vkc_dmoe_prepare(&ch->dm, ch->lo, ch->L - ch->lo, c->n_routed_experts, c->num_experts_per_tok,
                              ch->D, ch->I, rows)) {
            if (!told && !ch->dm.off && vkt_ready())
                fprintf(stderr, "[VK] %s chain: COLI_VK_DMOE=all: the tier does not hold any of layers %d..%d's experts "
                                "whole yet; the host's step keeps them (the bank's rotation arms them layer by layer)\n",
                        v4c_name(ch), ch->lo, ch->L - 1);
            told = 1;
            return 0;
        }
        return 1;
    }
    if (n < ch->lo || n >= ch->L) return 0;
    memset(ch->dmar, 0, sizeof ch->dmar);
    if (ch->engine->dense_resident.layers[n].plan.uses_hash_router) {
        if (!told) fprintf(stderr, "[VK] %s chain: COLI_VK_DMOE=%d is a hash layer (vkc_route cannot give its token "
                                   "table); the host's step keeps it\n", v4c_name(ch), n);
        told = 1;
        return 0;
    }
    const ColiDeepSeekV4LayerWeights *w = &ch->engine->dense_resident.layers[n];
    if (!ch->gate && !(ch->gate = v4c_plain(w, "ffn.gate.weight", 11))) {
        if (!told) fprintf(stderr, "[VK] %s chain: layer %d's ffn.gate.weight did not reach the device; COLI_VK_DMOE "
                                   "stays off\n", v4c_name(ch), n);
        told = 1;
        return 0;
    }
    if (!ch->gate_bias) ch->gate_bias = v4c_val(w, "ffn.gate.bias");
    ch->dmar[n] = 1;
    if ((ch->gate_bias && !vkc_reserve(&ch->gbias, (size_t)c->n_routed_experts * sizeof(float), VKC_DEV)) ||
        !vkc_reserve(&ch->lg, (size_t)rows * c->n_routed_experts * sizeof(float), VKC_DEV)) return 0;
    ch->dm.big = big;   /* vk_dmoe.h's gate: a step of more than VKC_DMOE_ROWS rows */
    if (!vkc_dmoe_prepare(&ch->dm, n, 1, c->n_routed_experts, c->num_experts_per_tok, ch->D, ch->I, rows)) {
        if (!told && !ch->dm.off && vkt_ready())
            fprintf(stderr, "[VK] %s chain: COLI_VK_DMOE=%d: the tier does not hold all %d of layer %d's experts yet; "
                            "the host's step keeps it (pin them warm: COLI_VK_TIER_PIN_LAYER=%d, over a COLI_VK_TIER_GB "
                            "that holds them)\n", v4c_name(ch), n, c->n_routed_experts, n, n);
        told = 1;
        return 0;
    }
    return 1;
}
/* Layer i did not reach the device: the chain keeps the layers before it, and with the
 * dense weights on the device only, the layers from i on keep their host copies. */
static void v4c_cut(V4Chain *ch, int i, const char *why) {
    ColiV4Engine *e = ch->engine;
    if (i < ch->L) ch->L = i;
    if (vkc_dmoe_layer_env() >= i || vkc_dmoe_all_env()) v4c_dmoe_drop(ch);   /* the armed layer(s) left the chain */
    if (ch->d) { vkc_fit_shrink("deepseek_v4 dev2", &g_v4c_fit2, i - ch->lo, why); return; }
    if (e->dense_resident.device_only && e->dense_resident.device_layers > i) {
        fprintf(stderr, "[VK] deepseek_v4: layers %d to %d keep their host copies after all (the RAM plan had counted "
                        "them on the device)\n", i, e->dense_resident.device_layers - 1);
        e->dense_resident.device_layers = i;
        coli_vk_dense_host_layers(i, ch->Lm);
    }
    vkc_fit_shrink("deepseek_v4", &g_v4c_fit, i, why);
}
/* The placement of the dense weights on the device only (v4_dho_place) builds the chain's
 * layer first while the chain is being set up: 1 built, 0 not the chain's, -1 it did not
 * reach the device (the chain was cut there; the layer keeps its host copies). */
static int v4c_build_layer(int layer) {
    V4Chain *ch = g_v4c;
    if (!g_v4c_building || !ch || layer >= ch->L) return 0;
    if (ch->ly[layer].built) return 1;
    const char *why = NULL;
    if (v4c_layer_up(ch, layer, &why)) { ch->ly[layer].built = 1; return 1; }
    v4c_layer_free(ch, layer);
    v4c_cut(ch, layer, why);
    return -1;
}
/* The device-only placement of a chained layer failed after the chain built it: all of
 * the layer off the device, the chain cut before it. */
static void v4c_drop_layer(int layer, const char *why) {
    if (!g_v4c) return;
    v4c_layer_free(g_v4c, layer);
    v4c_cut(g_v4c, layer, why);
}
/* Every chained layer off the device and the chain off (a layer that would not load, the
 * parameters refused): what the device held alone comes back from disk when the CPU asks. */
static void v4c_teardown(V4Chain *ch, const char *why) {
    for (int i = ch->lo; i < ch->L; i++) if (ch->ly[i].built) v4c_layer_free(ch, i);
    vkc_free(ch->prm); ch->prm = NULL;
    v4c_cut(ch, ch->lo, why);
}

/* d = 1: the second device's chain, its layers from g_v4c_fit.n (device 1 current; the
 * primary's setup loaded every resident layer). */
static int v4c_setup_dev(ColiV4Engine *engine, int d) {
    const ColiDeepSeekV4Config *c = &engine->config;
    const char *why = v4c_unsupported(c);
    if (!why && !engine->runtime.dense_resident) why = "the dense layers are not resident (a low-memory plan reloads them per forward)";
    if (!why && (engine->gpu.enabled || (engine->experts && engine->experts->gpu))) why = "the CUDA tier is on and keeps its place";
    if (why) { fprintf(stderr, "[VK] deepseek_v4 chain: %s; the CPU runs the layers\n", why); return 0; }
    int L = c->num_hidden_layers, N = d ? g_v4c_fit2.n : g_v4c_fit.n, lo = d ? g_v4c_fit.n : 0;
    VkcFit *fit = d ? &g_v4c_fit2 : &g_v4c_fit;
    const char *nmc = d ? "deepseek_v4 dev2" : "deepseek_v4";
    if (N < 1) return 0;                            /* the fit's line said so: the chain stays off */
    V4Chain *ch = calloc(1, sizeof *ch);
    if (!ch || !(ch->ly = calloc((size_t)L, sizeof *ch->ly)) || !(ch->sli = malloc((size_t)L * sizeof(int)))) {
        if (ch) free(ch->ly);
        free(ch); return 0;
    }
    ch->d = d;
    if (d) g_v4c2 = ch; else g_v4c = ch;
    v4c_geom(ch, engine, lo, lo + N);
    ch->Wd = ch->W + v4c_rows(); ch->E = -1;
    if (!(ch->slot_pos = malloc((size_t)ch->Wd * sizeof(int)))) return 0;
    for (int k = 0; k < ch->Wd; k++) ch->slot_pos[k] = -1;
    size_t n = v4c_offsets(ch);
    if (!(ch->prm = vkc_buf((n ? n : 1) * sizeof(float), VKC_DEV))) {
        v4c_cut(ch, lo, vkc_lost() ? "the device was lost" : "device memory for the parameters refused");
        if (d) vkc_fit_placed(nmc, fit);
        return 0;
    }
    if (d) {   /* its layers' matrices and state, each whole or the chain stops before it */
        for (int i = lo; i < ch->L; i++) {
            if (v4c_layer_up(ch, i, &why)) { ch->ly[i].built = 1; vkc_fit_mark(fit, i - lo); continue; }
            v4c_layer_free(ch, i);
            v4c_cut(ch, i, vkc_lost() ? "the device was lost" : why);
        }
        if (ch->L <= lo) { vkc_free(ch->prm); ch->prm = NULL; vkc_fit_placed(nmc, fit); return 0; }
        goto params;
    }
    /* the resident layers, loaded now; each chained one goes up whole (with the dense
     * weights on the device only, in its placement), or the chain stops before it */
    char err[256] = {0};
    const ColiSafetensorsIndex *index = coli_v4_engine_target_index(engine);
    g_v4c_building = 1;
    for (int i = 0; i < L; i++) {
        ColiDeepSeekV4LayerWeights w;
        if (coli_v4_layer_load(engine, &w, c, index, i, err, sizeof err) || !engine->dense_resident.ready[i]) {
            g_v4c_building = 0;
            fprintf(stderr, "[VK] deepseek_v4 chain: layer %d did not load (%s); the CPU runs the layers\n", i, err);
            v4c_teardown(ch, "a layer did not load");
            return 0;
        }
        if (i >= ch->L) continue;
        if (v4c_build_layer(i) > 0) vkc_fit_mark(&g_v4c_fit, i);
    }
    g_v4c_building = 0;
    if (ch->L < 1) {   /* layer 0 did not reach the device: nothing of the chain stays there */
        vkc_free(ch->prm); ch->prm = NULL;
        vkc_fit_placed("deepseek_v4", &g_v4c_fit);
        return 0;
    }
params:
    v4c_geom(ch, engine, lo, ch->L);                /* the split's tables and list kinds for the layers it kept */
    float *a = calloc(n ? n : 1, sizeof(float));
    int ok = a != NULL;
    for (int i = lo; ok && i < ch->L; i++) {
        V4cLayer *ly = &ch->ly[i];
        const ColiDeepSeekV4LayerWeights *w = &engine->dense_resident.layers[i];
        int D = ch->D, nm = ch->nm;
        const uint16_t *an = v4c_val(w, "attn_norm.weight"), *fn = v4c_val(w, "ffn_norm.weight"),
                       *qn = v4c_val(w, "attn.q_norm.weight"), *kn = v4c_val(w, "attn.kv_norm.weight");
        const float *sink = v4c_val(w, "attn.attn_sink"), *hsa = v4c_val(w, "hc_attn_scale"), *hba = v4c_val(w, "hc_attn_base"),
                    *hsf = v4c_val(w, "hc_ffn_scale"), *hbf = v4c_val(w, "hc_ffn_base");
        ok = an && fn && qn && kn && sink && hsa && hba && hsf && hbf;
        for (int k = 0; ok && k < D; k++) { a[ly->o_an + k] = coli_bf16_decode(an[k]); a[ly->o_fn + k] = coli_bf16_decode(fn[k]); }
        for (int k = 0; ok && k < ch->QL; k++) a[ly->o_qn + k] = coli_bf16_decode(qn[k]);
        for (int k = 0; ok && k < ch->hd; k++) a[ly->o_kn + k] = coli_bf16_decode(kn[k]);
        if (ok) {
            memcpy(a + ly->o_sink, sink, (size_t)ch->nh * sizeof(float));
            memcpy(a + ly->o_hca, hsa, 3 * sizeof(float)); memcpy(a + ly->o_hca + 3, hba, (size_t)nm * sizeof(float));
            memcpy(a + ly->o_hcf, hsf, 3 * sizeof(float)); memcpy(a + ly->o_hcf + 3, hbf, (size_t)nm * sizeof(float));
        }
        if (ok && ly->ratio > 0) {
            const uint16_t *cn = v4c_val(w, "attn.compressor.norm.weight");
            const float *ape = v4c_val(w, "attn.compressor.ape");
            ok = cn && ape;
            for (int k = 0; ok && k < ch->hd; k++) a[ly->o_cn + k] = coli_bf16_decode(cn[k]);
            if (ok) memcpy(a + ly->o_ape, ape, (size_t)ly->ratio * ly->cproj * sizeof(float));
        }
        if (ok && ly->idx) {
            const uint16_t *icn = v4c_val(w, "attn.indexer.compressor.norm.weight");
            const float *iape = v4c_val(w, "attn.indexer.compressor.ape");
            ok = icn && iape;
            for (int k = 0; ok && k < ch->ID; k++) a[ly->o_icn + k] = coli_bf16_decode(icn[k]);
            if (ok) memcpy(a + ly->o_iape, iape, (size_t)8 * ch->ID * sizeof(float));
        }
    }
    ok = ok && vkc_begin() && vkc_write(ch->prm, 0, a, n * sizeof(float)) && vkc_submit(1);
    free(a);
    if (!ok) {
        if (!d) fprintf(stderr, "[VK] deepseek_v4 chain: the parameters did not reach the device; the CPU runs the layers\n");
        v4c_teardown(ch, vkc_lost() ? "the device was lost" : "the parameters did not reach it");
        vkc_fit_placed(nmc, fit);
        return 0;
    }
    ch->ok = 1;
    int ncomp = 0, nidx = 0;
    for (int i = lo; i < ch->L; i++) { ncomp += ch->ly[i].ratio > 0; nidx += ch->ly[i].idx; }
    size_t bytes = 0, tensors = 0;
    coli_vk_mem_info_dev(d, &bytes, &tensors);
    if (d) fprintf(stderr, "[VK] deepseek_v4 chain: layers %d..%d on the second device (%d compressing, %d indexing), %d streams, "
                           "%.1f MiB of parameters, %zu matrices (%.1f MiB) there\n",
                   lo, ch->L - 1, ncomp, nidx, ch->H, n * 4 / 1048576.0, tensors, bytes / 1048576.0);
    else fprintf(stderr, "[VK] deepseek_v4 chain: %d layers on the device (%d compressing, %d indexing), %d streams, "
                         "%.1f MiB of parameters, %zu matrices (%.1f MiB) on the device\n",
                 ch->L, ncomp, nidx, ch->H, n * 4 / 1048576.0, tensors, bytes / 1048576.0);
    if (!d && v4c_cell_wanted())
        fprintf(stderr, "[VK] deepseek_v4 chain: the decode attention cell is on -- a one-row forward's layer settles one fence "
                        "(its attention cell's) with the shared expert's record+submit riding the cell's device run "
                        "(COLI_VK_CHAIN_ATTN_CELL)\n");
    vkc_fit_placed(nmc, fit);
    return 1;
}
static int v4c_setup(ColiV4Engine *engine) { return v4c_setup_dev(engine, 0); }

/* room on the device for `rows` compressed rows (and keys) of layer i; growing drops them */
static int v4c_cache(V4Chain *ch, V4cLayer *ly, int rows) {
    if (ly->ratio > 0 && ly->ccap < rows && !ch->ks.on) {   /* under the split the rows keep their ks.rows */
        int cap = 64; while (cap < rows) cap *= 2;
        vkc_free(ly->ckv);
        ly->ckv = vkc_buf((size_t)cap * ch->hd * sizeof(float), VKC_DEV);
        ly->kv_valid = 0; ly->ccap = ly->ckv ? cap : 0;
        if (!ly->ckv) return 0;
    }
    if (ly->idx && ly->icap < rows) {
        int cap = 64; while (cap < rows) cap *= 2;
        vkc_free(ly->ikey);
        ly->ikey = vkc_buf((size_t)cap * ch->ID * sizeof(float), VKC_DEV);
        ly->ik_valid = 0; ly->icap = ly->ikey ? cap : 0;
        if (!ly->ikey) return 0;
    }
    return 1;
}

static int v4c_scratch(V4Chain *ch, int rows, int E, int LR);
/* Prompt rows per chunk (vkc_chunk_rows, decided at the first forward, after the tier
 * filled; V4_PREFILL_CHUNK or COLI_VK_CHAIN_ROWS set: v4c_rows): the chain's scratch a
 * row at the forward's context, the window rings' row and the routed experts' outputs. */
static int v4c_chunk_rows(V4Chain *ch, int E, int LR) {
    if (!vkc_chunk_auto() || getenv("V4_PREFILL_CHUNK")) return v4c_rows();
    size_t wcap = ch->wcap;
    g_v4c_count = 0; v4c_scratch(ch, 1, E, LR); long long b1 = g_v4c_count;
    g_v4c_count = 0; v4c_scratch(ch, 2, E, LR); long long b2 = g_v4c_count;
    g_v4c_count = -1; ch->wcap = wcap;
    const ColiDeepSeekV4Config *c = &ch->engine->config;
    size_t row = (size_t)(b2 - b1) + (size_t)(ch->L - ch->lo) * ch->hd * sizeof(float) +
                 (size_t)(2 * c->num_experts_per_tok + 1) * ch->D * sizeof(float);
    if (ch->ks.on) row += (size_t)LR * sizeof(int);
    return vkc_chunk_rows(v4c_name(ch), row);
}
/* The window rings for chunks of `rows`: W + rows rows each; rings that grow are
 * mirrored again (every slot stale). */
static int v4c_grow_win(V4Chain *ch, int rows) {
    if (ch->W + rows <= ch->Wd) return 1;
    int Wd = ch->W + rows;
    int *sp = realloc(ch->slot_pos, (size_t)Wd * sizeof(int));
    if (!sp) return 0;
    ch->slot_pos = sp;
    for (int k = 0; k < Wd; k++) sp[k] = -1;
    for (int i = ch->lo; i < ch->L; i++) {
        vkc_free(ch->ly[i].win);
        if (!(ch->ly[i].win = vkc_buf((size_t)Wd * ch->hd * sizeof(float), VKC_DEV))) return 0;
    }
    ch->Wd = Wd; ch->win_valid = 0;
    return 1;
}
/* The compressed rows, whole or split: planned when the whole mirrors would grow (and at
 * the first forward); once split, the split stays. E: the positions the forward reaches.
 * 0 = the device refused the split's rows. */
static int v4c_plan(V4Chain *ch, int E) {
    if (!ch->nsl || ch->ks.on) return 1;
    int grow = !ch->planned, tcap = 1, hd = ch->hd;
    size_t need = 0, held = 0;
    for (int i = ch->lo; i < ch->L; i++) {
        V4cLayer *ly = &ch->ly[i];
        if (ly->ratio <= 0) continue;
        int want = E / ly->ratio + 1, cap = ly->ccap, host = ch->engine->config.max_position_embeddings / ly->ratio + 1;
        if (host < want) host = want;
        if (host > tcap) tcap = host;
        if (cap < want) { cap = 64; while (cap < want) cap *= 2; grow = 1; }
        need += (size_t)cap * hd * sizeof(float); held += (size_t)ly->ccap * hd * sizeof(float);
    }
    if (!grow) return 1;
    ch->planned = 1;
    /* a window over recent slots, with optional read-based pins; the forward's
     * chunks shrink to what the window takes (v4c_forward) */
    if (vkc_kv_plan_need(&ch->ks, v4c_name(ch), ch->nsl, (size_t)hd * sizeof(float), tcap, 1, 1, held, need) != 2) return 1;
    for (int i = ch->lo; i < ch->L; i++) {
        V4cLayer *ly = &ch->ly[i];
        if (ly->ratio <= 0) continue;
        vkc_free(ly->ckv);
        ly->ccap = 0; ly->kv_valid = 0;
        if (!(ly->ckv = vkc_buf((size_t)ch->ks.rows * hd * sizeof(float), VKC_DEV))) return 0;
    }
    return 1;
}

static int v4c_scratch(V4Chain *ch, int rows, int E, int LR) {
    size_t r = (size_t)rows;
    int cp = 2 * ch->hd, taps = coli_v4_full_dspark_wanted ? 3 : 1;
    ch->wcap = (size_t)(E / 4 > 1 ? E / 4 : 1);
    int ok = v4c_res(&ch->xs, r * ch->HD, VKC_DEV) && v4c_res(&ch->xn, r * ch->HD, VKC_DEV) &&
             v4c_res(&ch->mix, r * ch->nm, VKC_DEV) && v4c_res(&ch->hpa, r * ch->hr, VKC_DEV) &&
             v4c_res(&ch->hpf, r * ch->hr, VKC_DEV) && v4c_res(&ch->col, r * ch->D, VKC_DEV) &&
             v4c_res(&ch->nrm, r * ch->D, VKC_DEV) && v4c_res(&ch->nq, r * ch->D, VKC_DEV) &&
             v4c_res(&ch->qa, r * ch->QL, VKC_DEV) && v4c_res(&ch->qr, r * ch->QL, VKC_DEV) &&
             v4c_res(&ch->qrq, r * ch->QL, VKC_DEV) && v4c_res(&ch->q, r * ch->nh * ch->hd, VKC_DEV) &&
             v4c_res(&ch->kv, r * ch->hd, VKC_DEV) && v4c_res(&ch->heads, r * ch->nh * ch->hd, VKC_DEV) &&
             v4c_res(&ch->grp, r * ch->og * ch->ol, VKC_DEV) && v4c_res(&ch->br, r * ch->D, VKC_DEV) &&
             v4c_res(&ch->craw, r * cp, VKC_DEV) && v4c_res(&ch->cscr, r * cp, VKC_DEV) &&
             v4c_res(&ch->icraw, r * 2 * ch->ID, VKC_DEV) && v4c_res(&ch->icscr, r * 2 * ch->ID, VKC_DEV) &&
             v4c_res(&ch->iq, r * ch->IH * ch->ID, VKC_DEV) && v4c_res(&ch->ihw, r * ch->IH, VKC_DEV) &&
             v4c_res(&ch->isc, r * ch->wcap, VKC_DEV) && v4c_res(&ch->list, (size_t)ch->nkind * r * LR, VKC_DEV) &&
             v4c_res(&ch->cs, (2 * r + ch->rmax) * ch->rd, VKC_DEV) && v4c_res(&ch->sq, r * ch->D, VKC_DEV) &&
             v4c_res(&ch->gs, r * ch->I, VKC_DEV) && v4c_res(&ch->us, r * ch->I, VKC_DEV) &&
             v4c_res(&ch->hs, r * ch->I, VKC_DEV) && v4c_res(&ch->ds, r * ch->D, VKC_DEV) &&
             v4c_res(&ch->h2d, r * ch->D, VKC_DOWN) && v4c_res(&ch->routed, r * ch->D, VKC_UP) &&
             v4c_res(&ch->xd, r * ch->HD, VKC_DOWN) && v4c_res(&ch->taps, (size_t)taps * r * ch->HD, VKC_DOWN);
    if (ok && ch->ks.on) {     /* the split: a chunk's new compressed rows and their canonical host copy */
        size_t nc = r / (size_t)ch->rmin + 2;
        if (g_v4c_count >= 0)
            g_v4c_count += (long long)((2 * r * ch->hd * sizeof(float) + ch->rmin - 1) / ch->rmin);
        else ok = v4c_res(&ch->cnew, nc * ch->hd, VKC_DEV) && v4c_res(&ch->cdn, nc * ch->hd, VKC_DOWN);
    }
    if (!ok || g_v4c_count >= 0) return ok;   /* counting: the buffers only */
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, r * ch->D * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    return 1;
}

/* ---- one layer's pieces ----------------------------------------------------------------- */
static int v4c_round(VkcBuf *x, VkcBuf *y, int kind, int nseg, int per_row, int len, int block, int flags,
                     int x_off, int x_row, int x_seg, int y_off, int y_row, int y_seg) {
    VkcDsRound p = {kind, nseg, per_row, len, block, flags, x_off, x_row, x_seg, y_off, y_row, y_seg,
                    kind == VKC_DS_HADAMARD ? 1.0f / sqrtf((float)len) : 0.0f};
    return vkc_dsv4_round(x, y, &p);
}
/* n rows of len floats at off, row apart, to bf16 in place */
static int v4c_bf16(VkcBuf *x, int n, int len, int off, int row) {
    return v4c_round(x, x, VKC_DS_BF16, n, 1, len, 0, 0, off, row, 0, off, row, 0);
}
/* n contiguous rows of len floats to E4M3 per 128 (an fp8 matrix's input) */
static int v4c_e4m3(VkcBuf *x, VkcBuf *y, int n, int len) {
    return v4c_round(x, y, VKC_DS_E4M3, n, 1, len, 128, 0, 0, len, 0, 0, len, 0);
}
static int v4c_norm(V4Chain *ch, VkcBuf *x, int xo, int xrow, size_t wo, VkcBuf *y, int yo, int yrow, int rows, int D, int flags) {
    VkcNorm p = {rows, D, 1, xo, xrow, D, yo, yrow, D, (int)wo, 0, flags, ch->engine->config.rms_norm_eps, 1.f};
    return vkc_norm(x, ch->prm, y, &p);
}
/* RoPE on interleaved pairs, the last rd floats of each segment, then bf16 */
static int v4c_rope(V4Chain *ch, VkcBuf *x, int nseg, int per_row, int x_off, int x_row, int x_seg,
                    int cs_off, int cs_row, int inverse) {
    VkcDsRope p = {nseg, per_row, ch->rd, x_off, x_row, x_seg, cs_off, cs_row, inverse, 1};
    return vkc_dsv4_rope(x, ch->cs, &p);
}
/* mHC's entry for a site (hc_pre with its own mix): the mix, the split into hp, the
 * collapse, bf16, the norm, bf16, into nrm; at one row COLI_VK_CHAIN_MHC_S1 takes the
 * fork's single-dispatch tail instead (the whole chain after the mix in one workgroup) */
static int v4c_pre(V4Chain *ch, ColiVkTensor *fn, size_t hco, VkcBuf *hp, size_t lno, int n) {
    const ColiDeepSeekV4Config *c = &ch->engine->config;
    int H = ch->H, D = ch->D, HD = ch->HD;
    VkcMhc sp = {n, H, D, c->hc_sinkhorn_iters, 0, HD, 0, ch->nm, 0, ch->hr, 0, D, (int)hco, 0, c->rms_norm_eps, c->hc_eps, 0.f};
    VkcMhcDec dp = {H, D, c->hc_sinkhorn_iters, 0, 0, 0, 0, (int)hco, (int)lno, c->rms_norm_eps, c->hc_eps, c->rms_norm_eps};
    if (!vkc_matmul(fn, ch->xs, 0, ch->mix, 0, n)) return 0;
    if (n == 1 && vkc_mhc_dec(0, ch->xs, ch->mix, hp, ch->prm, ch->nrm, &dp)) return 1;
    return vkc_mhc(VKC_MHC_SPLIT, ch->xs, ch->mix, hp, ch->prm, NULL, &sp) &&
           vkc_mhc(VKC_MHC_COLLAPSE, ch->xs, NULL, hp, NULL, ch->col, &sp) && v4c_bf16(ch->col, n, D, 0, D) &&
           v4c_norm(ch, ch->col, 0, D, lno, ch->nrm, 0, D, n, D, 0) && v4c_bf16(ch->nrm, n, D, 0, D);
}
/* mHC's exit: xn = comb(xs) + post * br, bf16, then the two swap */
static int v4c_post(V4Chain *ch, VkcBuf *hp, int n) {
    const ColiDeepSeekV4Config *c = &ch->engine->config;
    VkcMhc po = {n, ch->H, ch->D, c->hc_sinkhorn_iters, 0, ch->HD, 0, ch->D, 0, ch->hr, 0, ch->HD, 0, 0, c->rms_norm_eps, c->hc_eps, 0.f};
    VkcBuf *t;
    VkcMhcDec dp = {ch->H, ch->D, 0, 0, 0, 0, 0, 0, 0, 0.f, 0.f, 0.f};
    if (n == 1 && vkc_mhc_dec(1, ch->xs, ch->br, hp, NULL, ch->xn, &dp)) {
        t = ch->xs; ch->xs = ch->xn; ch->xn = t;
        return 1;
    }
    if (!vkc_mhc(VKC_MHC_POST, ch->xs, ch->br, hp, NULL, ch->xn, &po) || !v4c_bf16(ch->xn, n, ch->HD, 0, ch->HD)) return 0;
    t = ch->xs; ch->xs = ch->xn; ch->xn = t;
    return 1;
}

/* The forward being recorded: positions, rows, the RoPE tables' places in ch->cs (the
 * window table's row 0 is position pb; the compress table's row 0 is pb - rmax + 1),
 * and the list rows' place in ch->list (0: the buffer's first rows -- the chunk order
 * rewrites it a chunk at a time; the layer order's chunks live side by side, lbase the
 * chunk's own rows, so a later layer's unit reads the chunk it is visiting). */
typedef struct { int start, n, pb, nr, csW, csC, LR, s_ident; int lbase; } V4cFwd;

/* the compressor's ring for nr rows into out (rows of D floats), then the produced rows'
 * pooled-latent pipeline: bf16, norm, bf16, RoPE at the group's first position, and the
 * attention's E4M3 per 64 on the no-position part (or the indexer's Hadamard and E2M1) */
/* out's row 0 is compressed row obase (0: out holds every row; the split's scratch: the
 * chunk's first new row) */
static int v4c_compress(V4Chain *ch, const V4cFwd *f, ColiVkTensor *wkv, ColiVkTensor *wg, VkcBuf *kvb, VkcBuf *scb,
                        VkcBuf *ring, size_t ape, VkcBuf *out, int obase, int r, int D, size_t norm, int fp4) {
    int nr = f->nr, pb = f->pb, P = r == 4 ? 2 * D : D, g0 = pb / r, np = (pb + nr) / r - g0, rd = ch->rd;
    VkcDsComp cp = {nr, pb, r, P, D, r == 4, 0, P, 0, P, (int)ape, -obase * D, D, 0};
    if (!vkc_matmul(wkv, ch->nrm, 0, kvb, 0, nr) || !vkc_matmul(wg, ch->nrm, 0, scb, 0, nr) ||
        !vkc_dsv4_compress(kvb, scb, ring, ch->prm, out, &cp)) return 0;
    if (np <= 0) return 1;
    int o = (g0 - obase) * D, cs = f->csC + (g0 * r - (pb - ch->rmax + 1)) * rd;
    if (!v4c_bf16(out, np, D, o, D) || !v4c_norm(ch, out, o, D, norm, out, o, D, np, D, 0) || !v4c_bf16(out, np, D, o, D) ||
        !v4c_rope(ch, out, np, 1, o + D - rd, D, 0, cs, r * rd, 0)) return 0;
    if (!fp4) return v4c_round(out, out, VKC_DS_E4M3, np, 1, D - rd, 64, 1, o, D, 0, o, D, 0);
    return v4c_round(out, out, VKC_DS_HADAMARD, np, 1, D, 0, 0, o, D, 0, o, D, 0) &&
           v4c_round(out, out, VKC_DS_E2M1, np, 1, D, 32, 1, o, D, 0, o, D, 0);
}

/* attention_token_impl for nr rows on the device, into ch->br */
static int v4c_attention(V4Chain *ch, int i, const V4cFwd *f) {
    const ColiDeepSeekV4Config *c = &ch->engine->config;
    V4cLayer *ly = &ch->ly[i];
    int hd = ch->hd, nh = ch->nh, rd = ch->rd, QL = ch->QL, D = ch->D, nr = f->nr, pb = f->pb, qrow = nh * hd;
    int tcs = ly->ratio > 0 ? f->csC + (ch->rmax - 1) * rd : f->csW;      /* the table row of position pb */
    int ok = v4c_e4m3(ch->nrm, ch->nq, nr, D) &&
             vkc_matmul(ly->wq_a, ch->nq, 0, ch->qa, 0, nr) && v4c_bf16(ch->qa, nr, QL, 0, QL) &&
             v4c_norm(ch, ch->qa, 0, QL, ly->o_qn, ch->qr, 0, QL, nr, QL, 0) && v4c_bf16(ch->qr, nr, QL, 0, QL) &&
             v4c_e4m3(ch->qr, ch->qrq, nr, QL);
    int sp = ch->ks.on && ly->ratio > 0, li = ch->sli[i];
    if (ok && ly->ratio > 0) {
        int r = ly->ratio, g0 = pb / r, np = (pb + nr) / r - g0;
        ok = v4c_compress(ch, f, ly->cwkv, ly->cwg, ch->craw, ch->cscr, ly->ring, ly->o_ape, sp ? ch->cnew : ly->ckv,
                          sp ? g0 : 0, r, hd, ly->o_cn, 0);
        if (ok && sp && np > 0) {   /* the split: into their slots, and down for the host's rows once the frame is through */
            VkcKvPart pt = {1, hd, NULL, 0, ly->ckv, 0, 0};
            ok = vkc_kv_store(&ch->ks, li, &pt, ch->cnew, 0, (size_t)hd, 0, g0, np) && vkc_copy(ch->cdn, 0, ch->cnew, 0, (size_t)np * hd);
            vkc_kv_done(&ch->ks, li, g0 + np);
        }
    }
    if (ok && ly->idx) {                                                /* the indexer: keys, queries, scores, top-k */
        int IH = ch->IH, ID = ch->ID, width = (pb + nr) / 4, K = c->index_topk;
        ok = v4c_compress(ch, f, ly->iwkv, ly->iwg, ch->icraw, ch->icscr, ly->iring, ly->o_iape, ly->ikey, 0, 4, ID, ly->o_icn, 1) &&
             vkc_matmul(ly->iwq, ch->qrq, 0, ch->iq, 0, nr) && v4c_bf16(ch->iq, nr, IH * ID, 0, IH * ID) &&
             v4c_rope(ch, ch->iq, nr * IH, IH, ID - rd, IH * ID, ID, tcs, rd, 0) &&
             v4c_round(ch->iq, ch->iq, VKC_DS_HADAMARD, nr * IH, IH, ID, 0, 0, 0, IH * ID, ID, 0, IH * ID, ID) &&
             v4c_round(ch->iq, ch->iq, VKC_DS_E2M1, nr * IH, IH, ID, 32, 1, 0, IH * ID, ID, 0, IH * ID, ID) &&
             vkc_matmul(ly->iwp, ch->nrm, 0, ch->ihw, 0, nr);
        if (ok && width > 0) {
            VkcDsScore sp = {nr, pb, 4, IH, ID, width, 0, IH * ID, 0, IH, 0, ID, 0, (int)ch->wcap,
                             1.0f / sqrtf((float)(ID * IH)), 0, 0};
            VkcDsTopk tp = {nr, width, K, (int)ch->wcap, f->lbase + ch->W, f->LR, ch->Wd, 1};
            ok = vkc_dsv4_score(ch->iq, ch->ihw, ly->ikey, NULL, ch->isc, &sp) && vkc_dsv4_topk(ch->isc, ch->list, &tp);
            if (ok && f->s_ident > 0) {                                 /* V4_IDX_IDENTITY: index order while all fit */
                VkcDsTopk ti = tp; ti.S = f->s_ident; ti.order = 0;
                ok = vkc_dsv4_topk(ch->isc, ch->list, &ti);
            }
        }
    }
    ok = ok && vkc_matmul(ly->wq_b, ch->qrq, 0, ch->q, 0, nr) && v4c_bf16(ch->q, nr, qrow, 0, qrow);
    if (ok) {                                                           /* the per-head RMS without weight, bf16 */
        VkcNorm p = {nr * nh, hd, nh, 0, qrow, hd, 0, qrow, hd, 0, 0, VKC_NORM_NOW, c->rms_norm_eps, 1.f};
        ok = vkc_norm(ch->q, ch->prm, ch->q, &p) && v4c_bf16(ch->q, nr, qrow, 0, qrow);
    }
    ok = ok && vkc_matmul(ly->wkv, ch->nq, 0, ch->kv, 0, nr) && v4c_bf16(ch->kv, nr, hd, 0, hd) &&
         v4c_norm(ch, ch->kv, 0, hd, ly->o_kn, ch->kv, 0, hd, nr, hd, 0) && v4c_bf16(ch->kv, nr, hd, 0, hd) &&
         v4c_rope(ch, ch->q, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 0) &&
         v4c_rope(ch, ch->kv, nr, 1, hd - rd, hd, 0, tcs, rd, 0) &&
         v4c_round(ch->kv, ch->kv, VKC_DS_E4M3, nr, 1, hd - rd, 64, 1, 0, hd, 0, 0, hd, 0);
    if (!ok) return 0;
    VkcRegion *rg = malloc((size_t)nr * sizeof *rg);                    /* the new rows into the window ring */
    if (!rg) return 0;
    for (int s = 0; s < nr; s++) rg[s] = (VkcRegion){(size_t)((pb + s) % ch->Wd) * hd, (size_t)s * hd, (size_t)hd};
    ok = vkc_copy_regions(ly->win, ch->kv, rg, nr);
    free(rg);
    int cnt = ch->W;
    if (ly->idx) cnt += c->index_topk;
    else if (ly->ratio > 0) cnt += (pb + nr) / ly->ratio;
    int kind = ly->ratio > 0 && !ly->idx ? ly->kind : 0;
    VkcDsAttn a = {nr, nh, hd, cnt, f->lbase + kind * nr * f->LR, f->LR, ch->Wd, 0, 0, 0, qrow, 0, qrow, (int)ly->o_sink, 1,
                   1.0f / sqrtf((float)hd)};
    if (ok && sp) {   /* stage selected cold rows, preserving V4's complete softmax and bf16 rounding */
        ColiV4AttentionView v;
        ok = !coli_v4_attention_view(ch->attn[i], &v);
        VkcKvDs d = {&ch->ks, li, ch->q, ly->win, ly->ckv, ch->prm, ch->list, ch->heads, nr, nh, hd, cnt, a.l_off, a.l_row,
                     ch->Wd, 0, 0, qrow, (int)ly->o_sink, 1, 0, qrow, (pb + nr) / ly->ratio, a.scale, ok ? v.compressed : NULL};
        ok = ok && vkc_kv_ds(&d);
    } else ok = ok && vkc_dsv4_attn(ch->q, ly->win, ly->ratio > 0 ? ly->ckv : NULL, ch->list, ch->prm, ch->heads, &a);
    ok = ok && v4c_rope(ch, ch->heads, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 1) &&
         v4c_round(ch->heads, ch->heads, VKC_DS_E4M3, nr, 1, qrow, 128, 0, 0, qrow, 0, 0, qrow, 0);
    int og = ch->og, ol = ch->ol, gw = ch->gw;
    if (ok && og == 1) ok = vkc_matmul(ly->wo_a[0], ch->heads, 0, ch->grp, 0, nr);
    for (int g = 0; ok && og > 1 && g < og; g++) {
        VkcHgemv h = {0, nr, 1, ol, ol, 0, g * gw, qrow, 0, g * ol, og * ol, 0, 0, 0, 0};
        ok = vkc_mla_hgemv(ly->wo_a[g], ch->heads, ch->grp, NULL, &h);
    }
    return ok && v4c_bf16(ch->grp, nr, og * ol, 0, og * ol) && v4c_e4m3(ch->grp, ch->grp, nr, og * ol) &&
           vkc_matmul(ly->wo_b, ch->grp, 0, ch->br, 0, nr) && v4c_bf16(ch->br, nr, D, 0, D);
}

/* the shared expert: E4M3 input, w1 and w3, V4's SwiGLU, E4M3, w2, bf16 into ds */
static int v4c_shared(V4Chain *ch, V4cLayer *ly, int nr) {
    float lim = ch->engine->config.swiglu_limit;
    VkcDsSwiglu sw = {nr * ch->I, 0, 0, 0, lim > 0.0f ? lim : 0.0f};
    return v4c_e4m3(ch->nrm, ch->sq, nr, ch->D) && vkc_matmul(ly->sh1, ch->sq, 0, ch->gs, 0, nr) &&
           vkc_matmul(ly->sh3, ch->sq, 0, ch->us, 0, nr) && vkc_dsv4_swiglu(ch->gs, ch->us, ch->hs, &sw) &&
           v4c_e4m3(ch->hs, ch->hs, nr, ch->I) && vkc_matmul(ly->sh2, ch->hs, 0, ch->ds, 0, nr) &&
           v4c_bf16(ch->ds, nr, ch->D, 0, ch->D);
}
/* the FFN branch joins: br = bf16(routed + shared), then mHC's exit; routed the device
 * wrote (dm.y, a layer that ran its experts in the frame) or the host's copy */
static int v4c_join(V4Chain *ch, int nr, int dev) {
    VkcEw add = {VKC_EW_ADD, nr * ch->D, ch->D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->br, dev ? ch->dm.y : ch->routed, ch->ds, NULL, NULL, &add) && v4c_bf16(ch->br, nr, ch->D, 0, ch->D) &&
           v4c_post(ch, ch->hpf, nr);
}

static int v4c_read_count(ColiDeepSeekV4WindowAttentionState *s, int *count, int *keys) {
    ColiV4AttentionView v;
    if (coli_v4_attention_view(s, &v)) return 0;
    *count = v.compressed_count;
    *keys = 0;
    if (v.indexer) { ColiV4IndexerView iv; if (coli_v4_indexer_view(v.indexer, &iv)) return 0; *keys = iv.count; }
    return 1;
}

/* One chain's layers (lo..L-1, on its device, current) for the n rows of streams in
 * *state_ptr (positions start..), the streams after its last layer into *state_ptr
 * (swapped with *next_ptr, as the CPU's layer loop leaves them): the handoff of a partial
 * chain, where the caller's CPU loop (or the second device's chain) runs the remaining
 * layers from there. Returns the layer after its last (ch->L); 0: not taken (nothing
 * changed; *lost = 1: a frame failed, the device marked lost); -1: the MoE failed (error
 * set, as a CPU forward failing at that layer leaves the host state). */
static void v4c_off(V4Chain *ch) { v4c_dmoe_drop(ch); ch->failed = 1; if (!ch->d) g_v4c_mode = 0; }
/* ---- COLI_VK_CHAIN_ORDER=layer: the prompt loop layer-outer ---------------------------------
 * The fork's prefill order (its own comment: "A true prefill is layer-major: every prompt
 * position crosses this layer before the next one begins"): for each layer, every chunk
 * of the prompt; only then the next layer. What makes it legal is that a layer's pass is
 * self-contained -- a layer reads its own attention state (its window ring, its compressed
 * rows, its ring and indexer keys, which the chunk order lowers at its first chunk and
 * this order lowers at the layer's head, against a snapshot of the rings taken before the
 * first layer's pass: a later layer's mirrors still hold the state the snapshot describes
 * while the live slot_pos has followed the earlier layers' chunks) and the streams the
 * layer before left, resident for the whole prompt (ch->xall: each unit's rows go in and
 * out of the chunk's scratch as it runs). The window rings never see a position of a
 * later layer before an earlier one, so the chunk order's ring discipline holds per layer
 * exactly, and the per-row GEMV keeps a row's bits the same (the same (layer, chunk)
 * batches run -- only their order across the forward changes; the serving path is the
 * only thing an arm moves).
 *
 * What the reorder buys is tenure: the tier's stream window (or the bank, whose explicit
 * rotation this loop drives at the layer head -- nothing reads the old layer's experts
 * there, the layer tail's submit waited its frames) serves every chunk of one layer
 * before it moves, so a layer's experts are read once instead of once a chunk: the bank
 * stops coexisting with the streaming and becomes it. The unit (one layer, one chunk) is
 * the chunk path's layer body: the same frames, the join of the unit before deferred to
 * the next unit's head as the chunk path defers it across the layer boundary. Two of the
 * chunk path's per-chunk closes become per-layer: a unit's dmoe frames need not wait
 * before the next unit records (vkc_layer_recorded bounds them), but the tier's books
 * (vkt_dev_count) and DSpark's taps read host-mapped memory, so they drain at the loop's
 * wait points -- a host-step unit's A1, the split's copy-back, each layer tail -- where
 * the frames that wrote them are through; a tap is read from xall (the joined rows'
 * home), never off xs, which the next unit's load takes.
 *
 * Returns 1 (the prompt is through: `next` holds the streams, `pull` the state, the
 * caller's write-back does what it does for a chunk pass), 0 (a frame failed: the
 * caller's lost path) or -1 (the MoE failed: the caller's failure path). cs/wl/cos_/
 * sin_ stay the caller's to free either way. */
static int v4c_layer_outer(V4Chain *ch, ColiV4Engine *engine,
                           ColiDeepSeekV4WindowAttentionState **attention, const ColiDeepSeekV4Config *config,
                           ColiExpertStore *experts, const int *tokens, const float *state, float *next,
                           int start, int n, int pool, int rows, int LR, int dmoe, int ident,
                           float *cs, int *wl, float *cos_, float *sin_, char *error, size_t error_size) {
    const ColiDeepSeekV4Config *c = config;
    int lo = ch->lo, L = ch->L, D = ch->D, HD = ch->HD, hd = ch->hd, rd = ch->rd, W = ch->W, E = start + n;
    int ok = 1, T0 = ch->Lm - 3, taps = coli_v4_full_dspark_wanted;
    int q0 = E - W > start ? E - W : start, nq = E - q0, q_lo = start - W + 1 < 0 ? 0 : start - W + 1;
    int pending = 0, prev_dev = 0, prev_i = 0, prev_c0 = 0, prev_nr = 0, sent = 0;
    int pend_tap = 0, tap_i = 0, tap_c0 = 0, tap_nr = 0;
    int nunit = (n + rows - 1) / rows * (L - lo);
    struct { int i, c0, nr; } *dq = malloc((size_t)(nunit > 0 ? nunit : 1) * sizeof *dq);   /* dmoe units awaiting a wait point */
    int dqh = 0, dqt = 0;
    int *sp0 = malloc((size_t)ch->Wd * sizeof(int));   /* the rings as the forward found them: every layer's mirror of it */
    if (!sp0 || !dq || !vkc_begin() || !vkc_write(ch->xall, 0, state, (size_t)n * HD * sizeof(float))) {
        free(sp0); free(dq);
        return 0;
    }
    memcpy(sp0, ch->slot_pos, (size_t)ch->Wd * sizeof(int));
    ch->layer_fwd++;
#define V4C_LO_TAP() do {   /* the unit before's tap, from the reservoir its join went to */ \
        if (ok && pend_tap) ok = vkc_copy(ch->taps, (size_t)(tap_i - T0) * tap_nr * HD, ch->xall, (size_t)tap_c0 * HD, (size_t)tap_nr * HD); \
    } while (0)
#define V4C_LO_DRAIN() do {   /* after a wait: the ids and the tap the waited frames wrote */ \
        if (ok) { \
            if (pend_tap && taps) { \
                const float *tp = (const float *)vkc_ptr(ch->taps) + (size_t)(tap_i - T0) * tap_nr * HD; \
                for (int s = 0; s < tap_nr; s++) v4_mainh_tap(c, tap_i, tp + (size_t)s * HD, (int64_t)(start + tap_c0) + s); \
            } \
            pend_tap = 0; \
            while (dqh < dqt) { vkt_dev_count(dq[dqh].i, dq[dqh].nr, ch->dm.K, vkc_dmoe_ids(&ch->dm, dq[dqh].i)); dqh++; } \
        } \
    } while (0)
    for (int i = lo; ok && i < L; i++) {
        V4cLayer *ly = &ch->ly[i];
        if (!vkc_begin()) { ok = 0; break; }
        {   /* the layer's head: the host state lowered into this layer's mirrors, exactly the
             * chunk order's first-chunk lowering for this layer alone, the stale test on the
             * snapshot (the live slot_pos has followed the layers before) */
            ColiV4AttentionView v;
            if (coli_v4_attention_view(attention[i], &v)) { ok = 0; break; }
            for (int q = q_lo; ok && q < start; q++)
                if (q >= ch->win_valid || sp0[q % ch->Wd] != q)   /* stale, or its row taken since */
                    ok = vkc_write(ly->win, (size_t)(q % ch->Wd) * hd, v.kv + (size_t)(q % W) * hd, (size_t)hd * sizeof(float));
            if (ok && ly->ratio > 0) {
                int t0 = ly->kv_valid, t1 = start / ly->ratio;
                if (ok && t1 > t0 && !ch->ks.on)
                    ok = vkc_write(ly->ckv, (size_t)t0 * hd, v.compressed + (size_t)t0 * hd, (size_t)(t1 - t0) * hd * sizeof(float));
                ColiV4CompressorView cv;
                size_t rf = (size_t)ly->crows * ly->cproj;
                if (ok && !ly->ring_ok)
                    ok = !coli_v4_compressor_view(v.compressor, &cv) && cv.rows == ly->crows && cv.projection == ly->cproj &&
                         vkc_write(ly->ring, 0, cv.kv, rf * sizeof(float)) && vkc_write(ly->ring, rf, cv.score, rf * sizeof(float));
                if (ok && ly->idx) {
                    ColiV4IndexerView iv;
                    ok = ok && !coli_v4_indexer_view(v.indexer, &iv);
                    int k0 = ly->ik_valid, k1 = start / 4;
                    if (ok && k1 > k0)
                        ok = vkc_write(ly->ikey, (size_t)k0 * ch->ID, iv.keys + (size_t)k0 * ch->ID,
                                       (size_t)(k1 - k0) * ch->ID * sizeof(float));
                    size_t ir = (size_t)8 * 2 * ch->ID;
                    if (ok && !ly->iring_ok)
                        ok = !coli_v4_compressor_view(iv.compressor, &cv) && cv.rows == 8 && cv.projection == 2 * ch->ID &&
                             vkc_write(ly->iring, 0, cv.kv, ir * sizeof(float)) && vkc_write(ly->iring, ir, cv.score, ir * sizeof(float));
                }
            }
            for (int q = q_lo; ok && q < start; q++) ch->slot_pos[q % ch->Wd] = q;   /* this layer's mirrors hold them now */
        }
        if (ok && !ch->d) vkt_bank_switch(i);   /* the rotation, at the head: the layer tail's submit waited the old layer's frames */
        for (int c0 = 0; ok && c0 < n; c0 += rows) {
            vkc_dwin(2);   /* DPROF: a layer-chunk unit's window, closed at its end (the layer tail for the last) */
            int nr = n - c0 < rows ? n - c0 : rows, pb = start + c0, ci = c0 / rows;
            int lbase = ci * ch->nkind * rows * LR, csbase = ci * (2 * rows + ch->rmax) * rd;
            V4cFwd f = {start, n, pb, nr, csbase, csbase + nr * rd, LR, 0, lbase};
            if (ident) for (int s = 0; s < nr && (pb + s + 1) / 4 <= c->index_topk; s++) f.s_ident = s + 1;
            vkc_role(VKC_R_A1);   /* DPROF: the dense frames this unit submits (the join below is its head) */
            if (pending) {   /* the unit before: its FFN branch joins, its rows back to the reservoir */
                ok = v4c_join(ch, prev_nr, prev_dev);
                if (ok) ok = vkc_copy(ch->xall, (size_t)prev_c0 * HD, ch->xs, 0, (size_t)prev_nr * HD);
                if (taps && prev_i >= T0) { pend_tap = 1; tap_i = prev_i; tap_c0 = prev_c0; tap_nr = prev_nr; }
                pending = 0;
            }
            if (i == lo) {   /* this chunk's RoPE rows and lists, built and written once, into the chunk's own place:
             * every layer's unit of this chunk reads them there -- a later chunk's build
             * at layer lo must not take them away before the later layers visit the chunk.
             * The build is chunk-local (the host buffers hold one chunk); the write lands
             * at the chunk's region, where f.csW/f.csC/lbase point the kernels. */
                for (int s = 0; s < nr + ch->rmax - 1; s++) {
                    for (int t = 0; t < 2; t++) {
                        if (t == 0 && s >= nr) continue;
                        int p = t == 0 ? pb + s : pb - ch->rmax + 1 + s;
                        float *dst = cs + (t == 0 ? 0 : nr * rd) + (size_t)s * rd;
                        if (p < 0 || coli_v4_rope_position(cos_, sin_, rd, p, t ? c->original_max_position_embeddings : 0,
                                                           t ? c->compress_rope_theta : c->rope_theta, c->rope_factor,
                                                           c->rope_beta_fast, c->rope_beta_slow)) {
                            memset(dst, 0, (size_t)rd * sizeof(float));
                            continue;
                        }
                        for (int k = 0; k < rd / 2; k++) { dst[2 * k] = cos_[k]; dst[2 * k + 1] = sin_[k]; }
                    }
                }
                for (int k = 0; k < ch->nkind; k++)
                    for (int s = 0; s < nr; s++) {
                        int p = pb + s, *row = wl + ((size_t)k * nr + s) * LR;
                        for (int j = 0; j < W; j++) { int q = p - W + 1 + j; row[j] = q >= 0 ? q % ch->Wd : -1; }
                        for (int j = W; j < LR; j++) {
                            int g = j - W;
                            row[j] = k > 0 && g < (p + 1) / ch->kratio[k] ? ch->Wd + g : -1;
                        }
                    }
                ok = ok && vkc_write(ch->list, lbase, wl, (size_t)ch->nkind * nr * LR * sizeof(int)) &&
                     vkc_write(ch->cs, csbase, cs, (size_t)(2 * nr + ch->rmax - 1) * rd * sizeof(float));
            }
            if (ok && ch->ks.on && ly->ratio > 0) {   /* the split: this layer's table over the chunk's new rows */
                int r = ly->ratio, g0 = pb / r;
                ColiV4AttentionView v;
                if (coli_v4_attention_view(attention[i], &v)) { ok = 0; break; }
                VkcKvPart pt = {1, hd, v.compressed, 0, ly->ckv, 0, 0};
                vkc_kv_place(&ch->ks, ch->sli[i], g0, (pb + nr) / r - g0);
                ok = vkc_kv_push(&ch->ks, ch->sli[i], &pt, 1, g0);
            }
            ok = ok && vkc_copy(ch->xs, 0, ch->xall, (size_t)c0 * HD, (size_t)nr * HD);   /* the chunk's rows in */
            if (ok) {   /* the armed layers' table for this frame, this layer's router's bias with it */
                const float *bias = dmoe && ch->dmar[i] ? v4c_val(&engine->dense_resident.layers[i], "ffn.gate.bias") : NULL;
                ok = !dmoe || (vkc_dmoe_upload(&ch->dm) &&
                               (!bias || vkc_write(ch->gbias, 0, bias, (size_t)c->n_routed_experts * sizeof(float))));
            }
            if (!ok) break;
            for (int s = 0; s < nr; s++) ch->slot_pos[(pb + s) % ch->Wd] = pb + s;   /* this chunk's rows go in below */
            ch->layer_units++;
            ok = ok && v4c_pre(ch, ly->fna, ly->o_hca, ch->hpa, ly->o_an, nr) && v4c_attention(ch, i, &f) &&
                 v4c_post(ch, ch->hpa, nr) && v4c_pre(ch, ly->fnf, ly->o_hcf, ch->hpf, ly->o_fn, nr);
            ColiVkTensor *gate = ok && dmoe && ch->dmar[i] ?
                v4c_plain(&engine->dense_resident.layers[i], "ffn.gate.weight", 11) : NULL;
            if (ok && gate && vkc_dmoe_on(&ch->dm, i)) {     /* COLI_VK_DMOE: this layer's experts in the frame */
                vkc_role(VKC_R_DMOE);   /* DPROF: the frame the dmoe path submits, not an A1 */
                float lim = c->swiglu_limit;
                VkcRoute rp = {.kind = 2, .n_group = 1, .topk_group = 1, .has_bias = ch->gbias ? 1 : 0,
                               .norm = 1, .temp = 1.f, .scale = c->routed_scaling_factor};
                VkcMoe mp = {.D = D, .I = ch->I, .act = 0, .limit = lim > 0.0f ? lim : 0.0f};
                /* the shared expert and the router's logits in the same frame: no wait, no host step */
                ok = v4c_shared(ch, ly, nr) && vkc_matmul(gate, ch->nrm, 0, ch->lg, 0, nr) &&
                     vkc_dmoe_layer(&ch->dm, i, nr, ch->lg, ch->gbias, ch->nrm, &rp, &mp);
                if (ok && ch->ks.on) {   /* the split's host mirror takes this chunk's rows: the frame ends here */
                    V4C_LO_TAP();
                    ok = vkc_submit(0) && vkc_finish();
                    if (ok && ly->ratio > 0) {
                        int r = ly->ratio, g0 = pb / r, np = (pb + nr) / r - g0;
                        ColiV4AttentionView v;
                        if (np > 0 && !coli_v4_attention_view(attention[i], &v))
                            memcpy(v.compressed + (size_t)g0 * hd, vkc_ptr(ch->cdn), (size_t)np * hd * sizeof(float));
                    }
                    ok = ok && vkc_begin();
                    V4C_LO_DRAIN();
                } else if (ok && (c0 + nr < n || i + 1 < L))
                    ok = vkc_layer_recorded(&sent);   /* a long frame: the device starts on these units */
                if (ok && dqt < nunit) { dq[dqt] = (typeof(dq[0])){i, c0, nr}; dqt++; }   /* the books at the next wait */
                prev_dev = 1;
            } else {
                V4C_LO_TAP();   /* the tap rides the A1 frame; the drain reads it under the wait */
                ok = ok && vkc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D) && vkc_submit(0);    /* A1 */
                /* while it runs, the tier loads the experts this layer will likely stream (a
                 * big prompt chunk only): at this layer's next unit, the same layer's set */
                if (ok) { vkc_role(VKC_R_STAGE); vkt_stream_prefetch(i, nr); vkc_role(VKC_R_A1); }   /* DPROF: the staging beside A1's wait */
                ok = ok && vkc_finish();
                V4C_LO_DRAIN();
                if (!ok) break;
                if (ch->ks.on && ly->ratio > 0) {               /* this chunk's new compressed rows into the host's */
                    int r = ly->ratio, g0 = pb / r, np = (pb + nr) / r - g0;
                    ColiV4AttentionView v;
                    if (np > 0 && !coli_v4_attention_view(attention[i], &v))
                        memcpy(v.compressed + (size_t)g0 * hd, vkc_ptr(ch->cdn), (size_t)np * hd * sizeof(float));
                }
                /* A2: the shared expert, while the host computes the routed experts */
                vkc_role(VKC_R_A2);
                ok = vkc_begin() && v4c_shared(ch, ly, nr) && vkc_submit(0);
                if (!ok) break;
                double t1 = spec_now();
                vkc_role(VKC_R_MOEH);   /* DPROF: the host's routed-expert step (ch->host_ms above; the tier's own line splits it) */
                if (pool) coli_v4_expert_store_prefill_pool(experts, i);
                if (coli_v4_moe_routed(ch->host_routed, &engine->dense_resident.layers[i], config, experts,
                                       (const float *)vkc_ptr(ch->h2d), tokens + c0, nr, error, error_size)) {
                    free(sp0); free(dq);
                    return -1;
                }
                vkc_role(VKC_R_STAGE);   /* DPROF: the routed rows go up, and the tier's table again after its step */
                memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)nr * D * sizeof(float));
                ch->host_ms += (spec_now() - t1) * 1e3;
                ok = vkc_begin();
                if (ok && dmoe) ok = vkc_dmoe_upload(&ch->dm);   /* the host's step ran the tier: the table again */
                prev_dev = 0;
            }
            pending = 1; prev_i = i; prev_c0 = c0; prev_nr = nr;
            if (c0 + nr < n) vkc_dwin(0);   /* DPROF: the unit's window closes; a layer's last stays open over its tail */
        }
        if (!ok) break;
        {   /* the layer's tail: the last unit's join, the state final, the frames waited -- the next
             * layer's rotation (and its mirrors' lowering) start from a quiet device */
            vkc_role(VKC_R_TAIL);
            if (pending) {
                ok = v4c_join(ch, prev_nr, prev_dev);
                if (ok) ok = vkc_copy(ch->xall, (size_t)prev_c0 * HD, ch->xs, 0, (size_t)prev_nr * HD);
                if (ok && taps && prev_i >= T0) { pend_tap = 1; tap_i = prev_i; tap_c0 = prev_c0; tap_nr = prev_nr; }
                pending = 0;
            }
            V4C_LO_TAP();
            ok = ok && vkc_submit(1);
            V4C_LO_DRAIN();
            if (ok) vkc_dwin(0);   /* DPROF: the layer's last window closes here, its frames waited */
        }
    }
    if (ok) {   /* the forward's tail: the streams back whole, the state's copy-outs as the chunk path's last chunk */
        vkc_dwin(2);   /* DPROF: the closing frame's own window (the copy-outs and the download it waits) */
        vkc_role(VKC_R_TAIL);
        ok = vkc_begin();   /* the last layer tail submitted: a fresh frame for what closes the forward */
        if (ok && pending) {
            ok = v4c_join(ch, prev_nr, prev_dev);
            if (ok) ok = vkc_copy(ch->xall, (size_t)prev_c0 * HD, ch->xs, 0, (size_t)prev_nr * HD);
            if (ok && taps && prev_i >= T0) { pend_tap = 1; tap_i = prev_i; tap_c0 = prev_c0; tap_nr = prev_nr; }
            pending = 0;
        }
        V4C_LO_TAP();
        ok = ok && vkc_copy(ch->xd, 0, ch->xall, 0, (size_t)n * HD);
        for (int i = lo; ok && i < L; i++) {
            V4cLayer *ly = &ch->ly[i];
            VkcRegion *rg = malloc((size_t)(nq > 0 ? nq : 1) * sizeof *rg);
            if (!rg) { ok = 0; break; }
            for (int q = q0; q < E; q++) rg[q - q0] = (VkcRegion){ly->pw + (size_t)(q - q0) * hd, (size_t)(q % ch->Wd) * hd, (size_t)hd};
            ok = vkc_copy_regions(ch->pull, ly->win, rg, nq);
            free(rg);
            if (!ok || ly->ratio <= 0) continue;
            int r = ly->ratio, g0 = start / r, np = E / r - g0;
            size_t rf = (size_t)ly->crows * ly->cproj, cp = (size_t)ly->cproj;
            ok = ch->ks.on || vkc_copy(ch->pull, ly->pc, ly->ckv, (size_t)g0 * hd, (size_t)np * hd);
            if (ok && r == 4) ok = vkc_copy(ch->pull, ly->pr, ly->ring, 0, 2 * rf);
            else if (ok) {
                VkcRegion *rr = malloc((size_t)2 * ly->ns * sizeof *rr);
                if (!rr) { ok = 0; break; }
                for (int j = 0; j < ly->ns; j++) {
                    size_t slot = (size_t)((ly->q1 + j) % r);
                    rr[2 * j] = (VkcRegion){ly->pr + j * cp, slot * cp, cp};
                    rr[2 * j + 1] = (VkcRegion){ly->pr + ((size_t)ly->ns + j) * cp, rf + slot * cp, cp};
                }
                ok = vkc_copy_regions(ch->pull, ly->ring, rr, 2 * ly->ns);
                free(rr);
            }
            if (!ok || !ly->idx) continue;
            int k0 = start / 4, nk = E / 4 - k0;
            ok = vkc_copy(ch->pull, ly->pk, ly->ikey, (size_t)k0 * ch->ID, (size_t)nk * ch->ID) &&
                 vkc_copy(ch->pull, ly->pir, ly->iring, 0, (size_t)2 * 8 * 2 * ch->ID);
        }
        ok = ok && vkc_submit(1);
        V4C_LO_DRAIN();
        if (ok) memcpy(next, vkc_ptr(ch->xd), (size_t)n * HD * sizeof(float));
        if (ok) vkc_dwin(0);
    }
#undef V4C_LO_TAP
#undef V4C_LO_DRAIN
    if (!ok) fprintf(stderr, "[VK] %s chain: a frame failed in the layer-ordered prompt\n", v4c_name(ch));
    free(sp0); free(dq);
    return ok ? 1 : 0;
}

static int v4c_forward_seg(V4Chain *ch, ColiV4Engine *engine, float **state_ptr, float **next_ptr,
                           ColiDeepSeekV4WindowAttentionState **attention, const ColiDeepSeekV4Config *config,
                           ColiExpertStore *experts, const int *tokens, int start, int n, int pool,
                           int prompt, char *error, size_t error_size, int *lost) {
    *lost = 0;
    const ColiDeepSeekV4Config *c = config;
    int lo = ch->lo, L = ch->L, D = ch->D, HD = ch->HD, hd = ch->hd, rd = ch->rd, W = ch->W, E = start + n;
    const char *nmc = v4c_name(ch);
    int K = 0;                                          /* the compressed slots of a list row */
    for (int i = lo; i < L; i++) {
        V4cLayer *ly = &ch->ly[i];
        int k = ly->idx ? c->index_topk : ly->ratio > 0 ? E / ly->ratio : 0;
        if (k > K) K = k;
    }
    if (W + K > 3072) {
        if (!ch->told) fprintf(stderr, "[VK] %s chain: %d compressed rows at position %d pass the attention's "
                                       "3072 entries; the CPU runs the layers from here\n", nmc, K, E);
        ch->told = 1;
        return 0;
    }
    char err[256] = {0};
    for (int i = lo; i < L; i++)                        /* the compressors and the indexer exist on the host */
        if (ch->ly[i].ratio > 0 &&
            coli_v4_window_attention_prepare(attention[i], &engine->dense_resident.layers[i], config, err, sizeof err))
            return 0;
    if (attention != ch->attn) { v4c_lower_one(ch, 0); ch->attn = attention; }
    if (start != ch->E) v4c_lower_one(ch, start);
    for (int i = lo; i < L; i++) {                      /* the host's state is where the forward starts */
        V4cLayer *ly = &ch->ly[i];
        int cnt, keys;
        if (ly->ratio <= 0) continue;
        if (!v4c_read_count(attention[i], &cnt, &keys) || cnt != start / ly->ratio || (ly->idx && keys != start / 4)) {
            if (!ch->told) fprintf(stderr, "[VK] %s chain: layer %d holds %d compressed rows at position %d; "
                                           "the CPU runs this forward\n", nmc, i, cnt, start);
            ch->told = 1;
            return 0;
        }
    }
    if (!v4c_plan(ch, E)) {
        fprintf(stderr, "[VK] %s chain: device memory for the split's compressed rows refused; the CPU runs the layers\n", nmc);
        v4c_off(ch);
        return 0;
    }
    for (int i = lo; ch->ks.on && i < L; i++)           /* the split: the host's rows take this forward's new ones as it goes */
        if (ch->ly[i].ratio > 0 && coli_v4_attention_reserve(attention[i], E / ch->ly[i].ratio)) return 0;
    int LR = W + (K > 0 ? K : 1), CH = v4c_chunk_rows(ch, E, LR);
    if (ch->ks.on) {   /* a chunk's new compressed rows fit the split's window */
        int mx = ch->rmin == 1 ? ch->ks.chunk : (ch->ks.chunk - 1) * ch->rmin;
        if (CH > mx) CH = mx < 1 ? 1 : mx;
    }
    int rows = n < CH ? n : CH;
    if (!v4c_grow_win(ch, rows)) {
        fprintf(stderr, "[VK] %s chain: device memory for window rings of %d rows refused; the CPU runs the layers\n", nmc, rows);
        v4c_off(ch);
        return 0;
    }
    int ok = v4c_scratch(ch, rows, E, LR);
    for (int i = lo; ok && i < L; i++) ok = v4c_cache(ch, &ch->ly[i], E / (ch->ly[i].ratio > 0 ? ch->ly[i].ratio : 1) + 1);
    /* COLI_VK_CHAIN_ORDER=layer: the prompt loop layer-outer (the fork's prefill order --
     * every prompt position crosses a layer before the next one begins), so one bank or
     * slot window serves every chunk of a layer. Prompts of more than one chunk only:
     * one chunk is the same order either way, and a verify keeps its chunk (the GEMV's
     * bits, as GEMM_ROWS documents). The activations of the whole prompt stay resident
     * across a layer's pass (xall, and xd grown to the prompt); a refusal runs this
     * prompt in today's chunk order -- the reorder is an order, never a requirement. */
    int order_layer = v4c_order_layer() && prompt && rows < n;
    if (order_layer) {   /* the layer order's own buffers: the chunks' lists and RoPE tables side by side, the streams resident */
        int nch = (n + rows - 1) / rows;
        if (!v4c_res(&ch->xall, (size_t)n * ch->HD, VKC_DEV) || !v4c_res(&ch->xd, (size_t)n * ch->HD, VKC_DOWN) ||
            !v4c_res(&ch->list, (size_t)nch * ch->nkind * rows * LR, VKC_DEV) ||
            !v4c_res(&ch->cs, (size_t)nch * (2 * rows + ch->rmax) * rd, VKC_DEV)) {
            fprintf(stderr, "[VK] %s chain: COLI_VK_CHAIN_ORDER=layer: the %d-row prompt's streams, lists and tables did not "
                            "fit beside the chunk's; this prompt keeps the chunk order\n", nmc, n);
            order_layer = 0;
            if (!v4c_res(&ch->xd, (size_t)rows * ch->HD, VKC_DOWN) ||
                !v4c_res(&ch->list, (size_t)ch->nkind * rows * LR, VKC_DEV) ||
                !v4c_res(&ch->cs, (size_t)(2 * rows + ch->rmax) * rd, VKC_DEV)) {   /* the reserves freed them */
                fprintf(stderr, "[VK] %s chain: device memory for %d rows at %d positions refused; the CPU runs the layers\n", nmc, rows, E);
                v4c_off(ch);
                return 0;
            }
        }
    }
    int dmoe = ok && v4c_dmoe_arm(ch, rows, order_layer);   /* COLI_VK_DMOE: the armed layer(s)' experts in the frame, or none */
    /* what comes back at the end: the window rows, each layer's new compressed rows and
     * keys, the rings (a ring of ratio r without the overlap: the slots this forward wrote) */
    size_t pn = 0;
    int q0 = E - W > start ? E - W : start, nq = E - q0;
    for (int i = lo; ok && i < L; i++) {
        V4cLayer *ly = &ch->ly[i];
        ly->pw = pn; pn += (size_t)nq * hd;
        if (ly->ratio <= 0) continue;
        int r = ly->ratio;
        ly->pc = pn; pn += ch->ks.on ? 0 : (size_t)(E / r - start / r) * hd;   /* split: back per chunk */
        ly->q1 = E - r > start ? E - r : start;     /* without the overlap only the slots written change */
        ly->ns = r == 4 ? ly->crows : E - ly->q1;
        ly->pr = pn; pn += (size_t)2 * ly->ns * ly->cproj;
        if (!ly->idx) continue;
        ly->pk = pn; pn += (size_t)(E / 4 - start / 4) * ch->ID;
        ly->pir = pn; pn += (size_t)2 * 8 * 2 * ch->ID;
    }
    ok = ok && v4c_res(&ch->pull, pn, VKC_DOWN);
    float *cs = malloc((size_t)(2 * rows + ch->rmax) * rd * sizeof(float));
    int *wl = malloc((size_t)ch->nkind * rows * LR * sizeof(int));
    float *cos_ = malloc((size_t)rd / 2 * sizeof(float)), *sin_ = malloc((size_t)rd / 2 * sizeof(float));
    if (!ok || !cs || !wl || !cos_ || !sin_) {
        free(cs); free(wl); free(cos_); free(sin_);
        fprintf(stderr, "[VK] %s chain: device memory for %d rows at %d positions refused; the CPU runs the layers\n", nmc, rows, E);
        v4c_off(ch);
        return 0;
    }
    /* every matrix on the per-row GEMV: a row's bits do not depend on the chunk it sits
     * in (a prompt, a verify, a decode step, a prefix reused or computed cold), as on the
     * CPU, whose batched and per-token kernels give the same bits. The default since
     * the self-tuning defaults: an unset env is `-1`, the backend's generic fill rule
     * (its `gemm_min_s`) -- prompts only, for a verify's rows and every decode step the
     * pin below stays 0 and keeps the GEMV's bits, as deepseek_v41's chain keeps a
     * verify's rows on the decode step's. Measured on the RX 6900 XT, a
     * 2,515-token prompt's first token 189.6 s -> 152.1 s; the GEMM is the resident-matrix
     * path's row-canonical class, so a prompt row's bits can then depend on how the
     * prompt is chunked. `0` keeps the position-independent bits (the per-lever hatch
     * and COLI_VK_LEGACY=1's base), `N` uses the GEMM from N rows, and a negative `N`
     * reads as -1. */
    static int gemm_rows = -2;
    if (gemm_rows == -2)
        gemm_rows = coli_vk_env_def("COLI_VK_CHAIN_GEMM_ROWS", -1, 0);
    vkc_gemm_rows(prompt ? gemm_rows : 0);   /* a verify's rows get a decode step's bits */
    static int ident = -1;
    if (ident < 0) { const char *e = getenv("V4_IDX_IDENTITY"); ident = e && *e == '1'; }
    float *state = *state_ptr, *next = *next_ptr;
    int T0 = ch->Lm - 3, taps = coli_v4_full_dspark_wanted;   /* DSpark's targets: the model's last three layers
                                                                  * (those the chain runs; the CPU taps the rest) */
    int done = 0;
    if (order_layer) {   /* COLI_VK_CHAIN_ORDER=layer: the layer-outer path runs the prompt; 1 done, 0 a lost frame, -1 the MoE's failure */
        int r = v4c_layer_outer(ch, engine, attention, config, experts, tokens, state, next, start, n,
                                pool, rows, LR, dmoe, ident, cs, wl, cos_, sin_, error, error_size);
        if (r < 0) {   /* as the chunk path leaves a failed MoE: the host state is where the forward began */
            if (pool) coli_v4_expert_store_prefill_pool(experts, -1);
            vkc_finish();
            for (int k = 0; k < ch->Wd; k++) ch->slot_pos[k] = -1;   /* some layers wrote this forward's rows */
            v4c_lower_one(ch, start);
            free(cs); free(wl); free(cos_); free(sin_);
            return -1;
        }
        if (r == 0) goto lost;
        done = 1;
    }
    for (int c0 = 0; !done && c0 < n; c0 += rows) {
        vkc_dwin(n == 1 ? 1 : prompt ? 2 : 0);   /* DPROF: a forward of one row opens the decode window; a prompt chunk its own (closed at the chunk's end) */
        int nr = n - c0 < rows ? n - c0 : rows, pb = start + c0, last = c0 + nr == n;
        V4cFwd f = {start, n, pb, nr, 0, nr * rd, LR, 0, 0};
        if (ident) for (int s = 0; s < nr && (pb + s + 1) / 4 <= c->index_topk; s++) f.s_ident = s + 1;
        /* this chunk's RoPE rows, the CPU's own tables: the window table (the plain base)
         * from position pb, the compress table (YaRN, compress_rope_theta) from pb - rmax + 1 */
        for (int s = 0; s < nr + ch->rmax - 1; s++) {
            for (int t = 0; t < 2; t++) {
                if (t == 0 && s >= nr) continue;
                int p = t == 0 ? pb + s : pb - ch->rmax + 1 + s;
                float *dst = cs + (t == 0 ? f.csW : f.csC) + (size_t)s * rd;
                if (p < 0 || coli_v4_rope_position(cos_, sin_, rd, p, t ? c->original_max_position_embeddings : 0,
                                                   t ? c->compress_rope_theta : c->rope_theta, c->rope_factor,
                                                   c->rope_beta_fast, c->rope_beta_slow)) {
                    memset(dst, 0, (size_t)rd * sizeof(float));
                    continue;
                }
                for (int k = 0; k < rd / 2; k++) { dst[2 * k] = cos_[k]; dst[2 * k + 1] = sin_[k]; }
            }
        }
        /* each list kind's rows: the window (the last W positions a row reaches, oldest
         * first, as device ring rows), then the compressed rows: the indexer's top-k slots
         * (written on the device), or every row a layer of that ratio has made, in order */
        for (int k = 0; k < ch->nkind; k++)
            for (int s = 0; s < nr; s++) {
                int p = pb + s, *row = wl + ((size_t)k * nr + s) * LR;
                for (int j = 0; j < W; j++) { int q = p - W + 1 + j; row[j] = q >= 0 ? q % ch->Wd : -1; }
                for (int j = W; j < LR; j++) {
                    int g = j - W;
                    row[j] = k > 0 && g < (p + 1) / ch->kratio[k] ? ch->Wd + g : -1;
                }
            }
        if (!vkc_begin() || !vkc_write(ch->xs, 0, state + (size_t)c0 * HD, (size_t)nr * HD * sizeof(float)) ||
            !vkc_write(ch->list, 0, wl, (size_t)ch->nkind * nr * LR * sizeof(int)) ||
            !vkc_write(ch->cs, 0, cs, (size_t)(f.csC + (nr + ch->rmax - 1) * rd) * sizeof(float))) goto lost;
        if (c0 == 0) {   /* the device's copies made the host's, as the forward needs them */
            int q_lo = start - W + 1 < 0 ? 0 : start - W + 1;
            for (int i = lo; ok && i < L; i++) {
                V4cLayer *ly = &ch->ly[i];
                ColiV4AttentionView v;
                if (coli_v4_attention_view(attention[i], &v)) goto lost;
                for (int q = q_lo; ok && q < start; q++)
                    if (q >= ch->win_valid || ch->slot_pos[q % ch->Wd] != q)   /* stale, or its row taken since */
                        ok = vkc_write(ly->win, (size_t)(q % ch->Wd) * hd, v.kv + (size_t)(q % W) * hd, (size_t)hd * sizeof(float));
                if (ly->ratio <= 0) continue;
                int t0 = ly->kv_valid, t1 = start / ly->ratio;
                if (ok && t1 > t0 && !ch->ks.on) ok = vkc_write(ly->ckv, (size_t)t0 * hd, v.compressed + (size_t)t0 * hd, (size_t)(t1 - t0) * hd * sizeof(float));
                ColiV4CompressorView cv;
                size_t rf = (size_t)ly->crows * ly->cproj;
                if (ok && !ly->ring_ok)
                    ok = !coli_v4_compressor_view(v.compressor, &cv) && cv.rows == ly->crows && cv.projection == ly->cproj &&
                         vkc_write(ly->ring, 0, cv.kv, rf * sizeof(float)) && vkc_write(ly->ring, rf, cv.score, rf * sizeof(float));
                if (!ly->idx) continue;
                ColiV4IndexerView iv;
                ok = ok && !coli_v4_indexer_view(v.indexer, &iv);
                int k0 = ly->ik_valid, k1 = start / 4;
                if (ok && k1 > k0) ok = vkc_write(ly->ikey, (size_t)k0 * ch->ID, iv.keys + (size_t)k0 * ch->ID, (size_t)(k1 - k0) * ch->ID * sizeof(float));
                size_t ir = (size_t)8 * 2 * ch->ID;
                if (ok && !ly->iring_ok)
                    ok = !coli_v4_compressor_view(iv.compressor, &cv) && cv.rows == 8 && cv.projection == 2 * ch->ID &&
                         vkc_write(ly->iring, 0, cv.kv, ir * sizeof(float)) && vkc_write(ly->iring, ir, cv.score, ir * sizeof(float));
            }
            if (!ok) goto lost;
            for (int q = q_lo; q < start; q++) ch->slot_pos[q % ch->Wd] = q;
        }
        for (int i = lo; ch->ks.on && ok && i < L; i++) {  /* the split: each table's window over this chunk's new rows */
            V4cLayer *ly = &ch->ly[i];
            if (ly->ratio <= 0) continue;
            ColiV4AttentionView v;
            if (coli_v4_attention_view(attention[i], &v)) goto lost;
            int r = ly->ratio, g0 = pb / r;
            VkcKvPart pt = {1, hd, v.compressed, 0, ly->ckv, 0, 0};
            vkc_kv_place(&ch->ks, ch->sli[i], g0, (pb + nr) / r - g0);
            ok = vkc_kv_push(&ch->ks, ch->sli[i], &pt, 1, g0);
        }
        if (!ok) goto lost;
        for (int s = 0; s < nr; s++) ch->slot_pos[(pb + s) % ch->Wd] = pb + s;   /* this chunk's rows go in below */
        int pending = 0, prev_dev = 0, dm_ran = 0, sent = 0;   /* prev_dev: the layer before ran its experts in the frame */
        if (dmoe) {   /* the table again for this frame (the host's steps ran the tier), the router's bias with it */
            if (!vkc_dmoe_upload(&ch->dm) ||
                (ch->gbias && !vkc_write(ch->gbias, 0, ch->gate_bias, (size_t)c->n_routed_experts * sizeof(float)))) goto lost;
        }
        for (int i = lo; i < L && ok; i++) {
            V4cLayer *ly = &ch->ly[i];
            vkc_role(VKC_R_A1);   /* DPROF: the dense frame this layer submits (the join below is its head) */
            if (pending) {                                  /* the FFN branch of the layer before */
                ok = v4c_join(ch, nr, prev_dev);
                if (ok && taps && i - 1 >= T0) ok = vkc_copy(ch->taps, (size_t)(i - 1 - T0) * nr * HD, ch->xs, 0, (size_t)nr * HD);
                pending = 0;
            }
            ok = ok && v4c_pre(ch, ly->fna, ly->o_hca, ch->hpa, ly->o_an, nr) && v4c_attention(ch, i, &f) &&
                 v4c_post(ch, ch->hpa, nr) && v4c_pre(ch, ly->fnf, ly->o_hcf, ch->hpf, ly->o_fn, nr);
            if (ok && dmoe && vkc_dmoe_on(&ch->dm, i)) {     /* COLI_VK_DMOE: this layer's experts in the frame */
                vkc_role(VKC_R_DMOE);   /* DPROF: the frame the dmoe path submits, not an A1 */
                float lim = c->swiglu_limit;
                VkcRoute rp = {.kind = 2, .n_group = 1, .topk_group = 1, .has_bias = ch->gbias ? 1 : 0,
                               .norm = 1, .temp = 1.f, .scale = c->routed_scaling_factor};
                VkcMoe mp = {.D = D, .I = ch->I, .act = 0, .limit = lim > 0.0f ? lim : 0.0f};
                /* the shared expert and the router's logits in the same frame: no wait, no host step */
                ok = v4c_shared(ch, ly, nr) && vkc_matmul(ch->gate, ch->nrm, 0, ch->lg, 0, nr) &&
                     vkc_dmoe_layer(&ch->dm, i, nr, ch->lg, ch->gbias, ch->nrm, &rp, &mp);
                if (ok && ch->ks.on) {   /* the split's host mirror takes this chunk's rows: the frame ends here */
                    ok = vkc_submit(0) && vkc_finish();
                    if (ok && ly->ratio > 0) {
                        int r = ly->ratio, g0 = pb / r, np = (pb + nr) / r - g0;
                        ColiV4AttentionView v;
                        if (np > 0 && !coli_v4_attention_view(attention[i], &v))
                            memcpy(v.compressed + (size_t)g0 * hd, vkc_ptr(ch->cdn), (size_t)np * hd * sizeof(float));
                    }
                    ok = ok && vkc_begin();
                } else if (ok && i + 1 < L)
                    ok = vkc_layer_recorded(&sent);   /* a long frame: the device starts on these layers */
                prev_dev = 1; dm_ran = 1; pending = 1;
                continue;
            }
            ok = ok && vkc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D) && vkc_submit(0);    /* A1 */
            /* while it runs, the tier loads the experts this layer will likely stream (a
             * big prompt chunk only) */
            if (ok) {
                vkc_role(VKC_R_STAGE);
                vkt_stream_prefetch(i, nr);
                /* COLI_VK_TIER_PILOT: a rows-1 unit's next hash layer's selection is
                 * tid2eid-known now, a full layer before its MoE entry -- fire its
                 * store reads so they land under this layer's attention and host step
                 * (non-hash or past-the-end layers keep today's behaviour; the reads
                 * only warm, the fork's demand diamond stays the entry's) */
                if (nr == 1 && v4_pilot_wanted()) {
                    int pl = i + v4_pilot_depth();
                    if (pl < L && engine->dense_resident.layers[pl].plan.uses_hash_router &&
                        !(dmoe && vkc_dmoe_on(&ch->dm, pl))) {   /* a dmoe layer's experts never reach the store */
                        const int64_t *tid2eid = v4c_val(&engine->dense_resident.layers[pl],
                                                         "ffn.gate.tid2eid");
                        if (tid2eid)
                            v4_pilot_fire(experts, pl, tokens[c0],
                                          c->num_experts_per_tok, tid2eid,
                                          c->n_routed_experts, c->vocab_size);
                    }
                }
                vkc_role(VKC_R_A1);
            }   /* DPROF: the staging beside A1's wait */
            int cell = ok && v4c_cell_wanted() && nr == 1 && n == 1 && !prompt;
            if (cell) {   /* COLI_VK_CHAIN_ATTN_CELL: the decode attention cell -- A2's record+submit
                          * rides the cell's device run, and the settle is the cell's own fence */
                unsigned long long cell_ser = vkc_serial();
                vkc_role(VKC_R_A2);   /* DPROF: A2's span and frame stamp stay A2's, wherever it records */
                ok = vkc_begin() && v4c_shared(ch, ly, nr) && vkc_submit(0);
                vkc_role(VKC_R_A1);
                ok = ok && vkc_wait_serial(cell_ser);   /* one fence a layer-token: the cell's */
                if (ok) ch->cell_steps++;
                if (!ok) break;
            } else {
                ok = ok && vkc_finish();
                if (!ok) break;
            }
            if (ch->ks.on && ly->ratio > 0) {               /* this chunk's new compressed rows into the host's */
                int r = ly->ratio, g0 = pb / r, np = (pb + nr) / r - g0;
                ColiV4AttentionView v;
                if (np > 0 && !coli_v4_attention_view(attention[i], &v))
                    memcpy(v.compressed + (size_t)g0 * hd, vkc_ptr(ch->cdn), (size_t)np * hd * sizeof(float));
            }
            if (!cell) {   /* A2: the shared expert, while the host computes the routed experts
                            * (the cell armed above already has A2 in flight under the attention's run) */
                vkc_role(VKC_R_A2);
                ok = vkc_begin() && v4c_shared(ch, ly, nr) && vkc_submit(0);
                if (!ok) break;
            }
            double t1 = spec_now();
            vkc_role(VKC_R_MOEH);   /* DPROF: the host's routed-expert step (ch->host_ms above; the tier's own line splits it) */
            if (pool) coli_v4_expert_store_prefill_pool(experts, i);
            if (coli_v4_moe_routed(ch->host_routed, &engine->dense_resident.layers[i], config, experts,
                                   (const float *)vkc_ptr(ch->h2d), tokens + c0, nr, error, error_size)) {
                if (pool) coli_v4_expert_store_prefill_pool(experts, -1);
                vkc_finish();
                for (int k = 0; k < ch->Wd; k++) ch->slot_pos[k] = -1;   /* some layers wrote this chunk's rows */
                v4c_lower_one(ch, start);
                free(cs); free(wl); free(cos_); free(sin_);
                return -1;
            }
            vkc_role(VKC_R_STAGE);   /* DPROF: the routed rows go up, and the tier's table again after its step */
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)nr * D * sizeof(float));
            ch->host_ms += (spec_now() - t1) * 1e3;
            ok = vkc_begin();
            if (ok && dmoe) ok = vkc_dmoe_upload(&ch->dm);   /* the host's step ran the tier: the table again */
            prev_dev = 0;
            pending = 1;
        }
        vkc_role(VKC_R_TAIL);   /* DPROF: the closing frame: the last join, xd, the last chunk's pull copies */
        if (ok && pending) {
            ok = v4c_join(ch, nr, prev_dev);
            if (ok && taps && L - 1 >= T0) ok = vkc_copy(ch->taps, (size_t)(L - 1 - T0) * nr * HD, ch->xs, 0, (size_t)nr * HD);
        }
        ok = ok && vkc_copy(ch->xd, 0, ch->xs, 0, (size_t)nr * HD);
        if (ok && last) {                                   /* what the host's state gets back */
            for (int i = lo; i < L && ok; i++) {
                V4cLayer *ly = &ch->ly[i];
                VkcRegion *rg = malloc((size_t)(nq > 0 ? nq : 1) * sizeof *rg);
                if (!rg) { ok = 0; break; }
                for (int q = q0; q < E; q++) rg[q - q0] = (VkcRegion){ly->pw + (size_t)(q - q0) * hd, (size_t)(q % ch->Wd) * hd, (size_t)hd};
                ok = vkc_copy_regions(ch->pull, ly->win, rg, nq);
                free(rg);
                if (!ok || ly->ratio <= 0) continue;
                int r = ly->ratio, g0 = start / r, np = E / r - g0;
                size_t rf = (size_t)ly->crows * ly->cproj, cp = (size_t)ly->cproj;
                ok = ch->ks.on || vkc_copy(ch->pull, ly->pc, ly->ckv, (size_t)g0 * hd, (size_t)np * hd);
                if (ok && r == 4) ok = vkc_copy(ch->pull, ly->pr, ly->ring, 0, 2 * rf);
                else if (ok) {
                    VkcRegion *rr = malloc((size_t)2 * ly->ns * sizeof *rr);
                    if (!rr) { ok = 0; break; }
                    for (int j = 0; j < ly->ns; j++) {
                        size_t slot = (size_t)((ly->q1 + j) % r);
                        rr[2 * j] = (VkcRegion){ly->pr + j * cp, slot * cp, cp};
                        rr[2 * j + 1] = (VkcRegion){ly->pr + ((size_t)ly->ns + j) * cp, rf + slot * cp, cp};
                    }
                    ok = vkc_copy_regions(ch->pull, ly->ring, rr, 2 * ly->ns);
                    free(rr);
                }
                if (!ok || !ly->idx) continue;
                int k0 = start / 4, nk = E / 4 - k0;
                ok = vkc_copy(ch->pull, ly->pk, ly->ikey, (size_t)k0 * ch->ID, (size_t)nk * ch->ID) &&
                     vkc_copy(ch->pull, ly->pir, ly->iring, 0, (size_t)2 * 8 * 2 * ch->ID);
            }
        }
        ok = ok && vkc_submit(1);
        if (!ok) goto lost;
        if (dmoe && dm_ran)   /* the frame's books: the tier's heat from the ids it routed */
            vkt_dev_count(ch->dm.lo, nr, ch->dm.K, vkc_dmoe_ids(&ch->dm, ch->dm.lo));
        memcpy(next + (size_t)c0 * HD, vkc_ptr(ch->xd), (size_t)nr * HD * sizeof(float));
        if (taps) {
            const float *tp = (const float *)vkc_ptr(ch->taps);
            for (int t = T0 > lo ? T0 : lo; t < L; t++)   /* the chain's layers' taps only */
                for (int s = 0; s < nr; s++)
                    v4_mainh_tap(c, t, tp + ((size_t)(t - T0) * nr + s) * HD, (int64_t)pb + s);
        }
        if (prompt && n > 1) vkc_dwin(0);   /* DPROF: the chunk's window closes at its end; a decode step's stays open (the landed discipline) */
    }
    if (pool) coli_v4_expert_store_prefill_pool(experts, -1);
    /* every chunk is through: the host's state as the CPU would have left it */
    {
        const float *pl = (const float *)vkc_ptr(ch->pull);
        for (int i = lo; i < L; i++) {
            V4cLayer *ly = &ch->ly[i];
            ColiV4AttentionView v;
            coli_v4_attention_view(attention[i], &v);
            for (int q = q0; q < E; q++) memcpy(v.kv + (size_t)(q % W) * hd, pl + ly->pw + (size_t)(q - q0) * hd, (size_t)hd * sizeof(float));
            if (ly->ratio <= 0) continue;
            int r = ly->ratio, g0 = start / r, np = E / r - g0;
            if (coli_v4_attention_reserve(attention[i], E / r)) goto host_oom;
            coli_v4_attention_view(attention[i], &v);
            if (!ch->ks.on) memcpy(v.compressed + (size_t)g0 * hd, pl + ly->pc, (size_t)np * hd * sizeof(float));
            coli_v4_attention_set_count(attention[i], E / r);
            ColiV4CompressorView cv;
            coli_v4_compressor_view(v.compressor, &cv);
            size_t rf = (size_t)ly->crows * ly->cproj, cp = (size_t)ly->cproj;
            if (r == 4) {
                memcpy(cv.kv, pl + ly->pr, rf * sizeof(float));
                memcpy(cv.score, pl + ly->pr + rf, rf * sizeof(float));
            } else for (int j = 0; j < ly->ns; j++) {
                size_t slot = (size_t)((ly->q1 + j) % r);
                memcpy(cv.kv + slot * cp, pl + ly->pr + j * cp, cp * sizeof(float));
                memcpy(cv.score + slot * cp, pl + ly->pr + ((size_t)ly->ns + j) * cp, cp * sizeof(float));
            }
            if (!ly->idx) continue;
            ColiV4IndexerView iv;
            int k0 = start / 4, nk = E / 4 - k0;
            if (coli_v4_indexer_reserve(v.indexer, E / 4)) goto host_oom;
            coli_v4_indexer_view(v.indexer, &iv);
            memcpy(iv.keys + (size_t)k0 * ch->ID, pl + ly->pk, (size_t)nk * ch->ID * sizeof(float));
            coli_v4_indexer_set_count(v.indexer, E / 4);
            coli_v4_compressor_view(iv.compressor, &cv);
            size_t ir = (size_t)8 * 2 * ch->ID;
            memcpy(cv.kv, pl + ly->pir, ir * sizeof(float));
            memcpy(cv.score, pl + ly->pir + ir, ir * sizeof(float));
        }
        ch->win_valid = E; ch->E = E;
        for (int i = lo; i < L; i++) {
            V4cLayer *ly = &ch->ly[i];
            if (ly->ratio > 0) { ly->kv_valid = E / ly->ratio; ly->ring_ok = 1; }
            if (ly->idx) { ly->ik_valid = E / 4; ly->iring_ok = 1; }
        }
    }
    *state_ptr = next; *next_ptr = state;
    free(cs); free(wl); free(cos_); free(sin_);
    ch->forwards++;
    return L;   /* a partial chain: the CPU runs layers L.. from these streams */
host_oom:   /* the host state half written: as a failed CPU forward leaves it, an error */
    free(cs); free(wl); free(cos_); free(sin_);
    v4c_lower(0);
    if (error && error_size) snprintf(error, error_size, "out of memory writing the chained forward's state back");
    return -1;
lost:   /* a frame failed: the device is gone (or would not take a command) */
    free(cs); free(wl); free(cos_); free(sin_);
    if (pool) coli_v4_expert_store_prefill_pool(experts, -1);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(ch->d); }
    *lost = 1;
    return 0;
}
/* The devices' layers (the primary's first layers, the second device's after them) for
 * the n rows of streams in *state_ptr, as v4c_forward_seg describes, from layer 0.
 * Returns the layer after the last that ran (the caller's CPU loop runs the rest from
 * there); 0: not taken (the CPU runs every layer, nothing changed); -1: the MoE failed. */
static int v4c_forward(ColiV4Engine *engine, float **state_ptr, float **next_ptr,
                       ColiDeepSeekV4WindowAttentionState **attention, const ColiDeepSeekV4Config *config,
                       ColiExpertStore *experts, const int *tokens, int start, int n, int pool,
                       int prompt, char *error, size_t error_size) {
    V4Chain *ch = g_v4c, *ch2 = g_v4c2;
    if (!g_v4c_mode || !ch || !ch->ok || ch->failed || !engine || engine != ch->engine || !attention || n < 1) return 0;
    if (g_v4c_mode == COLI_VK_CHAIN_PREFILL && n <= 2) return 0;
    if (ch2 && (!ch2->ok || ch2->failed || ch2->engine != engine || ch2->lo != ch->L)) ch2 = NULL;
    int lost = 0;
    if (ch2) { vkc_device(1); lost = vkc_lost(); vkc_device(0); }
    if (vkc_lost() || lost) {
        for (int d = 0; d < 2; d++) if (v4c_of(d)) v4c_of(d)->failed = 1;
        g_v4c_mode = 0;
        return 0;
    }
    int r = v4c_forward_seg(ch, engine, state_ptr, next_ptr, attention, config, experts, tokens, start, n, pool,
                            prompt, error, error_size, &lost);
    if (r <= 0) {
        if (lost) {
            for (int d = 0; d < 2; d++) if (v4c_of(d)) v4c_of(d)->failed = 1;
            g_v4c_mode = 0;
            v4c_lower(0);
            fprintf(stderr, "[VK] deepseek_v4 chain: the device was lost; the CPU runs this forward again and from here on "
                            "(the host's state is current: there is nothing to rebuild)\n");
        }
        return r;
    }
    if (!ch2) return r;
    vkc_device(1);
    int r2 = v4c_forward_seg(ch2, engine, state_ptr, next_ptr, attention, config, experts, tokens, start, n, pool,
                             prompt, error, error_size, &lost);
    vkc_device(0);
    if (r2 != 0) return r2;   /* the layer after its last, or the MoE's failure */
    /* not taken there: the CPU runs its layers from the streams the primary handed over
     * (the second device's watermarks follow) */
    v4c_lower_one(ch2, start);
    if (lost) {
        for (int d = 0; d < 2; d++) if (v4c_of(d)) v4c_of(d)->failed = 1;
        g_v4c_mode = 0;
        v4c_lower(0);
        fprintf(stderr, "[VK] deepseek_v4 dev2 chain: the device was lost; the CPU runs its layers, and every layer from the next "
                        "forward on (the host's state is current: there is nothing to rebuild)\n");
    }
    return r;
}

static void v4c_report(void) {
    for (int d = 0; d < 2; d++) {
        V4Chain *ch = v4c_of(d);
        if (!ch || !ch->ok || !ch->forwards) continue;
        int was = vkc_device(d);
        VkcStats st; vkc_stats(&st);
        char wn[48];   /* DPROF: the fence-wait count the report never printed (VkcStats.waits) */
        if (vkc_dprof() && st.waits) snprintf(wn, sizeof wn, " over %llu fence waits", st.waits);
        else wn[0] = 0;
        fprintf(stderr, "[VK] %s chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                        "%.1f ms waiting for the device%s, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
                v4c_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, wn, ch->host_ms,
                st.dev_bytes / 1048576.0);
        if (st.d_fwd) {   /* DPROF: the decode window's split -- the wait/device gap G1 measures */
            double wsum = 0, dsum = 0;
            for (int r = 0; r < VKC_R_N; r++) { wsum += st.d_wait_ms[r]; dsum += st.d_dev_ms[r]; }
            fprintf(stderr, "[VK] %s chain decode: %llu fwd, %llu submits, %llu waits, %.1f ms wait "
                            "(A1 %.1f / A2 %.1f / dmoe %.1f / tail %.1f / other %.1f), %.1f ms device "
                            "(A1 %.1f / A2 %.1f / dmoe %.1f / tail %.1f / other %.1f), %.1f ms in vkQueueSubmit\n",
                    v4c_name(ch), st.d_fwd, st.d_frames, st.d_wait_n, wsum,
                    st.d_wait_ms[VKC_R_A1], st.d_wait_ms[VKC_R_A2], st.d_wait_ms[VKC_R_DMOE], st.d_wait_ms[VKC_R_TAIL],
                    st.d_wait_ms[VKC_R_OTHER], dsum,
                    st.d_dev_ms[VKC_R_A1], st.d_dev_ms[VKC_R_A2], st.d_dev_ms[VKC_R_DMOE], st.d_dev_ms[VKC_R_TAIL],
                    st.d_dev_ms[VKC_R_OTHER], st.d_submit_ms);
        }
        if (st.d_wall_ms > 0) {   /* DPROF: the decode window's host timeline, the prompt line's shape (CS-0) */
            double h[VKC_R_N], hsum = 0;
            for (int r = 0; r < VKC_R_N; r++) { h[r] = st.d_host_ms[r] / 1e3; hsum += h[r]; }
            fprintf(stderr, "[VK] %s chain decode host: %.1f s wall: host spans A1 %.1f s, A2 %.1f s, dmoe %.1f s, "
                            "host-MoE %.1f s, staging %.1f s, tail %.1f s, other %.1f s (%.1f s of the wall unnamed)\n",
                    v4c_name(ch), st.d_wall_ms / 1e3,
                    h[VKC_R_A1], h[VKC_R_A2], h[VKC_R_DMOE], h[VKC_R_MOEH], h[VKC_R_STAGE], h[VKC_R_TAIL], h[VKC_R_OTHER],
                    st.d_wall_ms / 1e3 - hsum);
        }
        if (st.p_fwd) {   /* DPROF: the prompt chunks' split -- the TTFT window S0 measures; the host spans tile the wall, the host-MoE step is the lever */
            double h[VKC_R_N], wsum = 0, dsum = 0;
            for (int r = 0; r < VKC_R_N; r++) { h[r] = st.p_host_ms[r] / 1e3; wsum += st.p_wait_ms[r]; dsum += st.p_dev_ms[r]; }
            fprintf(stderr, "[VK] %s chain prompt: %llu chunk windows, %.1f s wall: host spans A1 %.1f s, A2 %.1f s, dmoe %.1f s, "
                            "host-MoE %.1f s, staging-upload %.1f s, tail %.1f s, other %.1f s; waits %.1f s, device %.1f s, %.1f ms in vkQueueSubmit\n",
                    v4c_name(ch), st.p_fwd, st.p_wall_ms / 1e3,
                    h[VKC_R_A1], h[VKC_R_A2], h[VKC_R_DMOE], h[VKC_R_MOEH], h[VKC_R_STAGE], h[VKC_R_TAIL], h[VKC_R_OTHER],
                    wsum / 1e3, dsum / 1e3, st.p_submit_ms);
        }
        vkc_kv_report(&ch->ks);
        {   /* COLI_VK_DMOE's own line: one line whenever the env named a layer, steps or none --
             * an armed run whose gate kept the host's step (the tier never held the layer's experts
             * whole) must still say so at exit, not only in the one-time line at its first forward */
            int dn = !ch->d ? vkc_dmoe_layer_env() : 0;   /* the arm is the primary device's */
            int da = !ch->d ? vkc_dmoe_all_env() : 0;
            if (dn)
                fprintf(stderr, "[VK] %s chain: COLI_VK_DMOE=%d: layer %d, %llu layer steps through the dmoe path "
                                "(%llu in-frame, %llu hybrid)%s\n",
                        v4c_name(ch), dn, dn, ch->dm.steps, ch->dm.steps - ch->dm.hyb_steps, ch->dm.hyb_steps,
                        ch->dm.steps ? "" : "; none: the host's step kept every one");
            else if (da)
                fprintf(stderr, "[VK] %s chain: COLI_VK_DMOE=all: layers %d..%d, %llu layer steps through the dmoe path "
                                "(%llu in-frame, %llu hybrid)%s\n",
                        v4c_name(ch), ch->lo, ch->L - 1, ch->dm.steps, ch->dm.steps - ch->dm.hyb_steps, ch->dm.hyb_steps,
                        ch->dm.steps ? "" : "; none: the host's step kept every one");
            if (ch->layer_fwd)
                fprintf(stderr, "[VK] %s chain: COLI_VK_CHAIN_ORDER=layer: %llu layer-ordered prompt forwards, %llu "
                                "layer-chunk units (%d layers, chunks of %d rows)\n",
                        v4c_name(ch), ch->layer_fwd, ch->layer_units, ch->L - ch->lo, ch->rows);
            if (ch->cell_steps)   /* COLI_VK_CHAIN_ATTN_CELL: the cell's own line, whatever armed it */
                fprintf(stderr, "[VK] %s chain: COLI_VK_CHAIN_ATTN_CELL: %llu decode layer-tokens through the attention cell "
                                "(one settled fence each, A2 recorded under the cell's run)\n",
                        v4c_name(ch), ch->cell_steps);
        }
        vkc_dmoe_hyb_report(v4c_name(ch), &ch->dm);
        vkc_prof_print();
        vkc_device(was);
    }
}

/* COLI_VK_CHAIN at startup, before the tier sizes itself (the trunk's device copies count
 * as used): the decision, the pipelines, the resident layers and their tensors. */
/* The decision, made once: by the RAM plan when it asks whether the dense layers may live
 * on the device only (the expert store is not open then), else at v4c_start. */
static int g_v4c_decision = -1;
static int v4c_decide(const ColiV4Engine *engine) {
    if (g_v4c_decision >= 0) return g_v4c_decision;
    if (!g_v4_vk_ready || !engine) return 0;
    int tier_on = vkt_wanted() && !(engine->experts && engine->experts->gpu) && engine->config.n_routed_experts > 0;
    g_v4c_decision = coli_vk_chain_decide("deepseek_v4", tier_on, COLI_VK_CHAIN_UNMEASURED);
    if (g_v4c_decision && g_v4_mux_slots > 1) {
        g_v4c_decision = 0;
        fprintf(stderr, "[VK] deepseek_v4: KV_SLOTS=%d: the dense chain is off (it keeps one conversation's state on the device); "
                        "the expert tier runs every conversation's experts\n", g_v4_mux_slots);
    }
    return g_v4c_decision;
}
static void v4c_start(const ColiV4Engine *cengine) {
    ColiV4Engine *engine = (ColiV4Engine *)cengine;
    if (!g_v4_vk_ready || !engine) return;
    int on = v4c_decide(engine);
    const char *no = NULL;
    if (on && !(g_v4c_inited = vkc_init())) no = "the chain's pipelines did not come up";
    if (on && !no && !(vkc_mla_ready() && vkc_mhc_ready() && vkc_dsv4_ready()))
        no = "the MLA, mHC or DeepSeek shaders are missing (chain_hgemv, chain_mhc, chain_dsv4)";
    if (no) fprintf(stderr, "[VK] deepseek_v4: %s: the dense chain stays off\n", no);
    if (!on || no) return;
    if (!v4c_fit_now(engine) || g_v4c_fit.n < 1) return;   /* how many layers fit (vkc_fit), once */
    int n0 = g_v4c_fit.n;
    if (!v4c_setup(engine)) return;
    g_v4c_mode = on;
    if (g_v4c_fit2_on) {
        vkc_device(1);
        if (g_v4c->L < n0) {
            /* the primary placed fewer layers than its fit: the second device's would not
             * follow them, so they stay on the CPU too */
            fprintf(stderr, "[VK] deepseek_v4 chain: the primary device stopped before layer %d; layers %d..%d stay on the CPU, "
                            "not on the second device\n", n0, n0, n0 + g_v4c_fit2.n - 1);
            g_v4c_fit2_on = 0;
            vkc_shutdown();
        } else if (!v4c_setup_dev(engine, 1)) { g_v4c_fit2_on = 0; vkc_shutdown(); }
        vkc_device(0);
    }
}
/* After the tier's: at exit the chain goes before the device. */
static void v4c_atexit(void) {
    if (g_v4c_inited) atexit(vkc_shutdown_all);
}
/* v4_vk_close, before the matrices go: the line, then every frame through (both devices). */
static void v4c_close(void) {
    v4c_report();
    if (g_v4c_inited && !vkc_lost()) vkc_finish();
    if (g_v4c2) { vkc_device(1); if (!vkc_lost()) vkc_finish(); vkc_device(0); }
}
