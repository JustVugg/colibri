/* mimo_chain.h -- MiMo-V2.6's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by mimo.c in a COLI_VULKAN build, after the CPU pieces it reuses (moe()
 * for the routed experts) and before forward(), which calls it; COLI_VK_CHAIN decides
 * (coli_vk_chain_decide, in main).
 *
 * What runs where, per layer, for one block of rows (decode: one row):
 *   device: the routed MoE output of the layer before joins the residual (x += out,
 *     the CPU's add), the input RMSNorm, the fused qkv projection, partial RoPE on q and
 *     k (rotate-half on the first rope_dim dims, the CPU's own cosf/sinf in a host
 *     table, one table per layer kind: the two kinds have their own theta), the value
 *     scale (chain_ew SCALE), the new K/V rows into the device's cache, attention with
 *     the layer's sliding window and sink logit (chain_attn), o_proj, the residual add,
 *     the post-attention RMSNorm; on the dense layer (layer 0 on the release) the MLP
 *     too (gate, up, SwiGLU, down, add), with no host step. A MoE layer's frame ends
 *     with the normalized rows going down and is waited for.
 *   host: moe() as the CPU runs it, on the device's normalized rows: the router
 *     (sigmoid scores, the correction bias picks, top-k renormalized), the routed
 *     experts (the Vulkan expert tier's batch and the CPU's share, joined in each row's
 *     routing order). MiMo has no shared expert, so nothing runs beside it.
 *   last frame: the final norm and lm_head on the last row (every row for a read-out).
 * Crossing per MoE layer: D floats a row down, D floats a row up; per step the new K/V
 * rows of every layer down (a few KB a row) and the logits.
 *
 * The KV caches, and who owns them. The host's stay canonical, in the layout the CPU
 * keeps (mimo.c's kv_alloc): a full-attention layer [ctx][kvh][hd] (V [ctx][kvh][vd]),
 * a sliding-window layer a RING of rows = min(window, ctx) rows, position p in row
 * p % rows. The device mirrors each one in exactly that layout (chain_attn reads
 * position-major rows, and a ring), behind a watermark kv_valid per layer: positions
 * [0, kv_valid) -- for a ring, the last `rows` of them -- equal the host's. A step at
 * pos_base lowers the watermark to pos_base (a reset, a rollback), then uploads the
 * host's rows between it and pos_base (for a ring only those the window still sees); a
 * CPU step lowers it to its pos_base; a photo restored into the host's rings
 * (pin_state_load) drops the rings' watermarks to 0, so the next step uploads them.
 *
 * A windowed layer's attention: one decode row reads the ring in place (its window is
 * exactly the ring's rows); a block of rows cannot (its later rows overwrite slots its
 * earlier rows still see), so it gathers the window's earlier rows from the ring and its
 * own new rows into one position-major scratch, attends there, and then writes its last
 * `rows` rows into the ring, as the CPU's attention() does with its window copy. The two
 * visit the positions in the same order, so a row gets the same bits either way.
 *
 * The new rows reach the host's caches only when the step's last frame has completed,
 * so the host's state always describes whole steps: a device lost in the middle of a
 * step loses nothing the host holds, and the CPU runs the step's remaining rows from
 * the position the host's caches end at (no rebuild, a turn with a picture included).
 * COLI_VK_CHAIN_FAULT=n (vk_chain.c) fakes the loss at the n-th frame, for tests.
 *
 * Past the device's budget (vk_kvsplit.h) each full-attention layer keeps a window of
 * the newest blocks on the device (the windowed layers keep their rings, which are
 * small): the step's attention runs there, sink included, and on the CPU over the older
 * positions of the host's cache at once, merged through the softmax statistics. The
 * host's caches hold every position before the step (each chunk commits its rows), so
 * the host's part always reads them there.
 *
 * The chain declines (the per-matrix path or the CPU runs) when a dense matrix did not
 * reach the device, for geometry outside chain_attn (a head dim above 256) and under
 * MIMO_TRACE (the CPU's per-layer dump).
 *
 * A partial chain (vk_chain.h, vkc_fit; mc_fit_start in mimo.c, at start-up): when the
 * dense layers do not all fit the device, the first nl run here and the CPU runs the
 * others and the head. Each chunk crosses the nl layers; its residual rows come down with
 * the frame that ends it, and the CPU runs layers nl.. and the head on them (layers_cpu,
 * head_cpu) before the next chunk. The device's layers keep their mirrors behind the
 * watermarks; the CPU's keep their caches on the host only. */
#include "vk_chain.h"
#include "vk_kvsplit.h"

typedef struct {
    int ok, failed;
    int lo, d;                                 /* its layers start at lo, on device d (vkc_device) */
    int nl, full;                              /* layers on the device (lo..lo+nl-1); the head there too */
    int rows;                                  /* scratch capacity in rows */
    int rd_off[2];                             /* the RoPE table of each layer kind in cs */
    VkcBuf *prm;                               /* the norms and the sink logits */
    size_t *o_ln1, *o_ln2, *o_sink, o_final;
    VkcBuf **kc, **vc;                         /* the device's caches, per layer */
    int *kv_valid;
    int *kvi;                                  /* a full layer's index in the split (ks), -1 windowed */
    VkcKvSplit ks;                             /* the full layers' split past the device's budget (ks.on) */
    size_t *o_kvd;                             /* a layer's new K rows in kvd (V after cap rows) */
    VkcBuf *x, *nrm, *tmp, *qkv, *kn, *vn, *wk, *wv, *ctx, *g, *u, *fin;
    VkcBuf *h2d, *kvd, *outd, *routed, *cs;
    float *host_out;
    float *host_in;                            /* a chunk's rows before the primary's layers (the second device's chain) */
    unsigned long long forwards, frames;
    double wait_ms, host_ms;
} MimoChain;

static int g_vk_chain = 0;     /* COLI_VK_CHAIN decided on, and the chain's pipelines are up */
/* The partial chain's fit (mc_fit_start, mimo.c): the first n of L layers on the device;
 * L = 0 while there is none (the chain off, or one that declines as before). */
static VkcFit g_mc_fit;
static int g_mc_placed;   /* vkc_fit_placed said it (mc_place, mimo.c) */
static int mc_layers(const Model *m) { return g_mc_fit.L ? g_mc_fit.n : m->c.n_layers; }
/* The layers after the primary's on COLI_VK_DEV2's device (docs/vulkan.md, "Layers on two
 * devices"): a second chain from layer g_mc_fit.n, its own fit (mc_fit_start), its
 * matrices placed by mc_place. Each chunk crosses the primary's layers, comes down, crosses
 * these, comes down, and the CPU runs what is left. */
static VkcFit g_mc_fit2;
static int g_mc_fit2_on;
static MimoChain *mc_of(Model *m, int d) { return (MimoChain *)(d ? m->vkchain2 : m->vkchain); }
static const char *mc_name(const MimoChain *ch) { return ch && ch->d ? "mimo dev2" : "mimo"; }
static int mc_dev2_wanted(void) { const char *e = getenv("COLI_VK_CHAIN_DEV2"); return !(e && *e == '0'); }


/* the geometry of layer li's attention */
typedef struct { int k, nh, kvh, hd, vd, qd, kd, vdd, rw, half, rows, swa; } McGeo;
static McGeo mc_geo(const Model *m, int li) {
    const Cfg *c = &m->c;
    McGeo g;
    g.k = c->swa[li]; g.swa = g.k;
    g.nh = c->heads[g.k]; g.kvh = c->kv_heads[g.k]; g.hd = c->head_dim[g.k]; g.vd = c->v_dim[g.k];
    g.qd = g.nh * g.hd; g.kd = g.kvh * g.hd; g.vdd = g.kvh * g.vd; g.rw = g.qd + g.kd + g.vdd;
    g.half = c->rope_dim[g.k] / 2;
    g.rows = m->L[li].rows;
    return g;
}

static int mc_ctensor(DW *d) { return dw_upload(d) && d->vk; }

/* the device's KV caches, in bytes (the tier's budget leaves them room); a full layer
 * holds COLI_VK_KV_DEVICE_ROWS rows when that asks for fewer than the context */
static size_t mc_kv_bytes_n(const Model *m, int nl) {
    size_t b = 0;
    long forced = vkc_kv_env("COLI_VK_KV_DEVICE_ROWS", 0);
    for (int i = 0; i < nl; i++) {
        McGeo g = mc_geo(m, i);
        size_t rows = !g.swa && forced > 0 && forced < g.rows ? (size_t)forced : (size_t)g.rows;
        b += rows * (g.kd + g.vdd) * sizeof(float);
    }
    return b;
}
static size_t mc_kv_bytes(const Model *m) { return mc_kv_bytes_n(m, m->c.n_layers); }

/* The model's parameters on the device, its tensors resolved, its caches allocated;
 * NULL = the chain cannot run. With a fit, the first n layers only (their matrices placed
 * at start-up, mc_place), and the head when the tail went up too. */
/* d = 1: the second device's chain (its matrices placed by mc_place; device 1 current). */
static MimoChain *mc_setup_dev(Model *m, int d) {
    MimoChain *ch = mc_of(m, d);
    if (ch) return ch->ok ? ch : NULL;
    if (d && !g_mc_fit2_on) return NULL;
    ch = (MimoChain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    if (d) m->vkchain2 = ch; else m->vkchain = ch;
    ch->d = d;
    Cfg *c = &m->c; int L = c->n_layers, H = c->hidden, lo = d ? g_mc_fit.n : 0, nl = d ? g_mc_fit2.n : mc_layers(m);
    ch->lo = lo;
    for (int k = 0; k < 2; k++)
        if (c->head_dim[k] > 256 || c->v_dim[k] > 256) {
            fprintf(stderr, "[VK] mimo chain: head dims %d/%d are past its attention shader's 256; per-matrix path\n",
                    c->head_dim[k], c->v_dim[k]);
            return NULL;
        }
    if (nl < 1) return NULL;
    ch->o_ln1 = calloc(L, sizeof(size_t)); ch->o_ln2 = calloc(L, sizeof(size_t)); ch->o_sink = calloc(L, sizeof(size_t));
    ch->o_kvd = calloc(L, sizeof(size_t)); ch->kv_valid = calloc(L, sizeof(int)); ch->kvi = calloc(L, sizeof(int));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *));
    if (!ch->o_ln1 || !ch->o_ln2 || !ch->o_sink || !ch->o_kvd || !ch->kv_valid || !ch->kvi || !ch->kc || !ch->vc) return NULL;
    /* the matrices: the device copies the per-matrix path uploads (vk_dense_upload) */
    int ok = 1;
    g_mimo_vk_dev = d;
    for (int i = lo; i < lo + nl && ok; i++) {
        Layer *l = &m->L[i];
        ok = mc_ctensor(&l->qkv) && mc_ctensor(&l->o) &&
             (c->moe[i] || (mc_ctensor(&l->gate) && mc_ctensor(&l->up) && mc_ctensor(&l->down)));
    }
    VkcFit *fit = d ? &g_mc_fit2 : &g_mc_fit;
    int full = lo + nl == L && (!fit->L || fit->tail);
    if (ok && full && fit->L && !mc_ctensor(&m->head)) {   /* the fit's tail did not go up: the head on the CPU */
        full = 0; fit->tail = 0;
        fprintf(stderr, "[VK] mimo chain: the head did not reach the device; it runs on the CPU, the layers here\n");
    }
    ok = ok && (!full || mc_ctensor(&m->head));
    g_mimo_vk_dev = 0;
    if (!ok) { fprintf(stderr, "[VK] mimo chain: a dense matrix did not reach the device; per-matrix path\n"); return NULL; }
    size_t n = 0;
    for (int i = lo; i < lo + nl; i++) {
        ch->o_ln1[i] = n; n += H; ch->o_ln2[i] = n; n += H;
        if (m->L[i].sink) { ch->o_sink[i] = n; n += c->heads[c->swa[i]]; }
    }
    ch->o_final = n; n += H;
    float *arena = calloc(n, sizeof(float));
    if (!arena) return NULL;
    for (int i = lo; i < lo + nl; i++) {
        Layer *l = &m->L[i];
        memcpy(arena + ch->o_ln1[i], l->ln1, H * sizeof(float));
        memcpy(arena + ch->o_ln2[i], l->ln2, H * sizeof(float));
        if (l->sink) memcpy(arena + ch->o_sink[i], l->sink, c->heads[c->swa[i]] * sizeof(float));
    }
    memcpy(arena + ch->o_final, m->norm, H * sizeof(float));
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, n * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return NULL;
    /* the caches, in the host's layout and at its size; the full layers' split past the
     * device's budget (a window of blocks a layer, the rest in the host's cache) */
    int nfull = 0, frows = 0;
    size_t frow = 0;
    for (int i = 0; i < L; i++) {
        McGeo g = mc_geo(m, i);
        int here = i >= lo && i < lo + nl;
        ch->kvi[i] = g.swa || !here ? -1 : nfull++;
        if (!g.swa && here) { frows = g.rows; frow = (size_t)(g.kd + g.vdd) * sizeof(float); }
    }
    /* KV placement is row-anchored; its legacy chunk argument is not used. The
     * actual prompt chunk is sized after setup and clamped to ks.chunk. */
    if (nfull && !vkc_kv_plan(&ch->ks, mc_name(ch), nfull, frow, frows, 1, 0, 0)) {
        fprintf(stderr, "[VK] mimo chain: the KV split's tables refused; per-matrix path\n");
        return NULL;
    }
    size_t kv = 0;
    for (int i = lo; i < lo + nl; i++) {
        McGeo g = mc_geo(m, i);
        if (!g.swa && ch->ks.on) g.rows = ch->ks.rows;
        ch->kc[i] = vkc_buf((size_t)g.rows * g.kd * sizeof(float), VKC_DEV);
        ch->vc[i] = vkc_buf((size_t)g.rows * g.vdd * sizeof(float), VKC_DEV);
        if (!ch->kc[i] || !ch->vc[i]) {
            fprintf(stderr, "[VK] mimo chain: device memory for the KV caches refused; per-matrix path\n");
            return NULL;
        }
        kv += (size_t)g.rows * (g.kd + g.vdd);
    }
    ch->nl = nl; ch->full = full;
    ch->ok = 1;
    int swa = 0;
    for (int i = lo; i < lo + nl; i++) swa += c->swa[i];
    if (d) fprintf(stderr, "[VK] mimo chain: layers %d..%d on the second device (%d sliding window)%s, %.1f MiB of KV mirrors\n",
                   lo, lo + nl - 1, swa, full ? ", the head too" : "", kv * 4 / 1048576.0);
    else fprintf(stderr, "[VK] mimo chain: %d layers on the device (%d sliding window), %.1f MiB of KV mirrors\n",
                 nl, swa, kv * 4 / 1048576.0);
    return ch;
}
static MimoChain *mc_setup(Model *m) { return mc_setup_dev(m, 0); }

/* mc_res counts instead of reserving while g_mc_count >= 0 (the chunk's sizing; the fit's,
 * g_mc_fitcount, each buffer as the pools round it) */
static long long g_mc_count = -1;
static int g_mc_fitcount;
static int mc_res(VkcBuf **b, size_t floats, int kind) {
    if (g_mc_count >= 0) {
        size_t by = (floats ? floats : 1) * sizeof(float);
        g_mc_count += (long long)(g_mc_fitcount ? vkc_fit_buf(by) : by);
        return 1;
    }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}
static int mc_scratch(MimoChain *ch, Model *m, int rows, int out);
/* Prompt rows per chunk (vkc_chunk_rows): the chain's scratch a row, counted from the
 * reservations (mc_scratch stops at its first host allocation in this mode), and the
 * routed experts' outputs (the tier's rows, the CPU's, the host's sum) for it. */
static int mc_chunk_rows(MimoChain *ch, Model *m) {
    g_mc_count = 0; mc_scratch(ch, m, 1, 1); long long b1 = g_mc_count;
    g_mc_count = 0; mc_scratch(ch, m, 2, 1); long long b2 = g_mc_count;
    g_mc_count = -1;
    size_t row = (size_t)(b2 - b1) + (size_t)(2 * m->c.topk + 1) * m->c.hidden * sizeof(float);
    /* moe_cpu retains a whole chunk's gathered inputs and gate/up outputs. */
    row += (size_t)m->c.topk * ((size_t)m->c.hidden + 2 * (size_t)m->c.moe_inter) * sizeof(float);
    return vkc_chunk_rows(mc_name(ch), row);
}

/* scratch for `rows` rows, `out` rows of logits */
static int mc_scratch(MimoChain *ch, Model *m, int rows, int out) {
    Cfg *c = &m->c; int H = c->hidden, L = c->n_layers;
    size_t r = (size_t)rows, rw = 0, kd = 0, vdd = 0, ctxw = 0, wkd = 0, wvd = 0, kvd = 0;
    int dense = 0, win = 0;
    for (int i = 0; i < L; i++) {
        McGeo g = mc_geo(m, i);
        if ((size_t)g.rw > rw) rw = g.rw;
        if ((size_t)g.kd > kd) kd = g.kd;
        if ((size_t)g.vdd > vdd) vdd = g.vdd;
        if ((size_t)g.nh * g.vd > ctxw) ctxw = (size_t)g.nh * g.vd;
        if (g.swa) {
            if ((size_t)g.kd > wkd) wkd = g.kd;
            if ((size_t)g.vdd > wvd) wvd = g.vdd;
            if (g.rows - 1 > win) win = g.rows - 1;
        }
        int cap = g.swa && g.rows < rows ? g.rows : rows;
        ch->o_kvd[i] = kvd; kvd += (size_t)cap * (g.kd + g.vdd);
        dense |= !c->moe[i];
    }
    ch->rd_off[0] = 0; ch->rd_off[1] = rows * 2 * (c->rope_dim[0] / 2);
    if (ch->ks.on) {   /* the split's partial results: a full layer's heads */
        size_t part = 0;
        for (int i = 0; i < L; i++) { McGeo g = mc_geo(m, i); if (!g.swa && (size_t)g.nh * (g.vd + 2) > part) part = (size_t)g.nh * (g.vd + 2); }
        if (g_mc_count >= 0) g_mc_count += (long long)r * (3 * part + rw) * sizeof(float);
        else if (!vkc_kv_parts(&ch->ks, r * part)) return 0;
    }
    size_t cs = r * 2 * (size_t)(c->rope_dim[0] / 2 + c->rope_dim[1] / 2);
    int ok = mc_res(&ch->x, r * H, VKC_DEV) && mc_res(&ch->nrm, r * H, VKC_DEV) && mc_res(&ch->tmp, r * H, VKC_DEV) &&
             mc_res(&ch->qkv, r * rw, VKC_DEV) && mc_res(&ch->kn, r * kd, VKC_DEV) && mc_res(&ch->vn, r * vdd, VKC_DEV) &&
             mc_res(&ch->ctx, r * ctxw, VKC_DEV) && mc_res(&ch->fin, (size_t)out * H, VKC_DEV) &&
             mc_res(&ch->h2d, r * H, VKC_DOWN) && mc_res(&ch->kvd, kvd, VKC_DOWN) &&
             mc_res(&ch->outd, (size_t)out * c->vocab, VKC_DOWN) && mc_res(&ch->routed, r * H, VKC_UP) &&
             mc_res(&ch->cs, cs, VKC_UP);
    if (ok && wkd) ok = mc_res(&ch->wk, ((size_t)win + r) * wkd, VKC_DEV) && mc_res(&ch->wv, ((size_t)win + r) * wvd, VKC_DEV);
    if (ok && dense) ok = mc_res(&ch->g, r * c->dense_inter, VKC_DEV) && mc_res(&ch->u, r * c->dense_inter, VKC_DEV);
    if (!ok || g_mc_count >= 0) return ok;   /* counting: the buffers only */
    if (ch->rows < rows) {
        float *ho = realloc(ch->host_out, r * H * sizeof(float));
        if (!ho) return 0;
        ch->host_out = ho;
    }
    ch->rows = rows;   /* o_kvd and rd_off are laid out for exactly this many */
    return 1;
}

/* ---- the KV caches between the host and the device ------------------------------- */
/* A CPU step from pos_base writes the host's rows from there on. */
static void mc_cpu_step(Model *m, int pos_base) {
    for (int d = 0; d < 2; d++) {
        MimoChain *ch = mc_of(m, d);
        if (!ch || !ch->ok) continue;
        for (int i = ch->lo; i < ch->lo + ch->nl; i++) {   /* the devices' layers: the CPU's keep their caches on the host */
            if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
            if (ch->kvi[i] >= 0) vkc_kv_lower(&ch->ks, ch->kvi[i], pos_base);
        }
    }
}
/* The host's rings were replaced (a photo restored): the devices' no longer match. */
static void mc_rings_rewritten(Model *m) {
    for (int d = 0; d < 2; d++) {
        MimoChain *ch = mc_of(m, d);
        if (!ch || !ch->ok) continue;
        for (int i = ch->lo; i < ch->lo + ch->nl; i++) if (m->c.swa[i]) ch->kv_valid[i] = 0;
    }
}
/* host rows [t0, t1) of a ring into the device's ring (at most two runs of slots) */
static int mc_ring_up(VkcBuf *dst, const float *src, int R, int t0, int t1, int w) {
    int ok = 1;
    while (ok && t0 < t1) {
        int slot = t0 % R, run = R - slot < t1 - t0 ? R - slot : t1 - t0;
        ok = vkc_write(dst, (size_t)slot * w, src + (size_t)slot * w, (size_t)run * w * sizeof(float));
        t0 += run;
    }
    return ok;
}
/* device rows of positions [t0, t1) between a ring (rows R) and a linear buffer whose
 * row 0 is position `base`; to_ring 1: linear -> ring, 0: ring -> linear */
static int mc_ring_copy(VkcBuf *ring, VkcBuf *lin, int R, int base, int t0, int t1, int w, int to_ring) {
    int ok = 1;
    while (ok && t0 < t1) {
        int slot = t0 % R, run = R - slot < t1 - t0 ? R - slot : t1 - t0;
        size_t ro = (size_t)slot * w, lo = (size_t)(t0 - base) * w, nf = (size_t)run * w;
        ok = to_ring ? vkc_copy(ring, ro, lin, lo, nf) : vkc_copy(lin, lo, ring, ro, nf);
        t0 += run;
    }
    return ok;
}
/* Record the uploads that make the device's caches the host's, as the step of n rows at
 * pos_base reads them. */
static int mc_push_state(MimoChain *ch, Model *m, int pos_base, int n_rows) {
    Cfg *c = &m->c; int ok = 1;
    for (int i = ch->lo; i < ch->lo + ch->nl && ok; i++) {
        McGeo g = mc_geo(m, i);
        Layer *l = &m->L[i];
        if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
        if (!g.swa && ch->ks.on) {     /* the split: its window placed, the rows below pos_base uploaded */
            VkcKvPart pt[2] = {{1, g.kd, l->K, 0, ch->kc[i], 0}, {1, g.vdd, l->V, 0, ch->vc[i], 0}};
            vkc_kv_lower(&ch->ks, ch->kvi[i], pos_base);
            vkc_kv_place(&ch->ks, ch->kvi[i], pos_base, n_rows);
            ok = vkc_kv_push(&ch->ks, ch->kvi[i], pt, 2, pos_base);
        } else if (g.swa) {
            /* the positions the window still sees must be in the host's ring, as the
             * CPU's attention() demands */
            int lo = pos_base - (c->window - 1) > 0 ? pos_base - (c->window - 1) : 0;
            for (int p = lo; p < pos_base; p++)
                if (l->ring_pos[p % g.rows] != p) { fprintf(stderr, "[mimo] window ring lost position %d\n", p); exit(1); }
            int from = ch->kv_valid[i] > pos_base - g.rows ? ch->kv_valid[i] : pos_base - g.rows;
            if (from < 0) from = 0;
            ok = mc_ring_up(ch->kc[i], l->K, g.rows, from, pos_base, g.kd) &&
                 mc_ring_up(ch->vc[i], l->V, g.rows, from, pos_base, g.vdd);
        } else if (ch->kv_valid[i] < pos_base) {
            int t0 = ch->kv_valid[i], n = pos_base - t0;
            ok = vkc_write(ch->kc[i], (size_t)t0 * g.kd, l->K + (size_t)t0 * g.kd, (size_t)n * g.kd * sizeof(float)) &&
                 vkc_write(ch->vc[i], (size_t)t0 * g.vdd, l->V + (size_t)t0 * g.vdd, (size_t)n * g.vdd * sizeof(float));
        }
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}

/* ---- one layer's pieces ------------------------------------------------------------- */
static int mc_norm(VkcBuf *x, size_t xo, VkcBuf *w, size_t wo, VkcBuf *y, size_t yo, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
static int mc_add(VkcBuf *x, VkcBuf *b, int n) {
    VkcEw p = {VKC_EW_ADD, n, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(x, x, b, NULL, NULL, &p);
}
/* Attention of layer i for n rows at pb, o_proj into tmp; the new rows into the
 * device's cache and into kvd for the host. */
static int mc_attention(MimoChain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    McGeo g = mc_geo(m, i);
    int ok = vkc_matmul((ColiVkTensor *)l->qkv.vk, ch->nrm, 0, ch->qkv, 0, n);
    if (ok && g.half) {   /* q's heads and k's heads are contiguous in a qkv row: one rotation */
        VkcRope p = {n * (g.nh + g.kvh), g.nh + g.kvh, 0, g.rw, g.hd, g.half, ch->rd_off[g.k], 2 * g.half};
        ok = vkc_rope(ch->qkv, ch->cs, &p);
    }
    VkcRegion *rg = ok ? malloc(sizeof *rg * (size_t)n) : NULL;
    if (!rg) return 0;
    for (int s = 0; s < n; s++) rg[s] = (VkcRegion){(size_t)s * g.kd, (size_t)s * g.rw + g.qd, (size_t)g.kd};
    ok = vkc_copy_regions(ch->kn, ch->qkv, rg, n);
    for (int s = 0; s < n; s++) rg[s] = (VkcRegion){(size_t)s * g.vdd, (size_t)s * g.rw + g.qd + g.kd, (size_t)g.vdd};
    ok = ok && vkc_copy_regions(ch->vn, ch->qkv, rg, n);
    free(rg);
    if (ok && c->v_scale != 1.0f) {   /* v *= attention_value_scale, as the CPU does before caching */
        VkcEw p = {VKC_EW_SCALE, n * g.vdd, 1, 1, 0, 1, 0, 0, 0, 0, 0, c->v_scale};
        ok = vkc_ew(ch->vn, ch->vn, NULL, NULL, NULL, &p);
    }
    if (!ok) return 0;
    VkcAttnW a;
    memset(&a, 0, sizeof a);
    a.a = (VkcAttn){n, g.nh, g.kvh, g.hd, pb, g.rows, 0, g.rw, g.hd, 0, 0, 0, 0, 0, g.nh * g.vd, 0, 0,
                    1.0f / sqrtf((float)g.hd), 0, 0};
    a.vd = g.vd; a.kv_pm = 1;
    a.sink = l->sink != NULL; a.sink_off = (int)ch->o_sink[i];
    VkcBuf *kc = ch->kc[i], *vc = ch->vc[i];
    if (!g.swa && ch->ks.on) {         /* the split: the rows into their window slots, the attention in two parts */
        VkcKvPart pk = {1, g.kd, l->K, 0, kc, 0}, pv = {1, g.vdd, l->V, 0, vc, 0};
        VkcKvGqa ga = {&ch->ks, ch->kvi[i], ch->qkv, kc, vc, NULL, NULL, l->sink ? ch->prm : NULL, ch->ctx,
                       n, g.nh, g.kvh, g.hd, g.vd, pb, 0, 1, l->sink != NULL, (int)ch->o_sink[i],
                       g.rw, g.hd, 0, 0, 0, 0, g.nh * g.vd, 0, 0, 1.0f / sqrtf((float)g.hd),
                       l->K, l->V, (size_t)g.hd, (size_t)g.kd, (size_t)g.vd, (size_t)g.vdd};
        ok = vkc_kv_store(&ch->ks, ch->kvi[i], &pk, ch->kn, 0, (size_t)g.kd, 0, pb, n) &&
             vkc_kv_store(&ch->ks, ch->kvi[i], &pv, ch->vn, 0, (size_t)g.vdd, 0, pb, n) &&
             vkc_copy(ch->kvd, ch->o_kvd[i], ch->kn, 0, (size_t)n * g.kd) &&
             vkc_copy(ch->kvd, ch->o_kvd[i] + (size_t)ch->rows * g.kd, ch->vn, 0, (size_t)n * g.vdd);
        return ok && vkc_kv_gqa(&ga) && vkc_matmul((ColiVkTensor *)l->o.vk, ch->ctx, 0, ch->tmp, 0, n);
    }
    if (!g.swa) {                      /* full attention: the rows in at pb, then every position */
        ok = vkc_copy(kc, (size_t)pb * g.kd, ch->kn, 0, (size_t)n * g.kd) &&
             vkc_copy(vc, (size_t)pb * g.vdd, ch->vn, 0, (size_t)n * g.vdd);
    } else if (n == 1) {               /* one row: its window is the ring, read in place */
        int slot = pb % g.rows;
        ok = vkc_copy(kc, (size_t)slot * g.kd, ch->kn, 0, g.kd) && vkc_copy(vc, (size_t)slot * g.vdd, ch->vn, 0, g.vdd);
        a.win = c->window; a.ring = g.rows;
    } else {                           /* a block: the window's earlier rows and the block's own, linear */
        int lo = pb - (c->window - 1) > 0 ? pb - (c->window - 1) : 0, nb = pb - lo;
        ok = mc_ring_copy(kc, ch->wk, g.rows, lo, lo, pb, g.kd, 0) && mc_ring_copy(vc, ch->wv, g.rows, lo, lo, pb, g.vdd, 0) &&
             vkc_copy(ch->wk, (size_t)nb * g.kd, ch->kn, 0, (size_t)n * g.kd) &&
             vkc_copy(ch->wv, (size_t)nb * g.vdd, ch->vn, 0, (size_t)n * g.vdd);
        a.a.pos_base = nb; a.win = c->window;
        kc = ch->wk; vc = ch->wv;
    }
    ok = ok && vkc_attn_w(ch->qkv, kc, vc, ch->ctx, NULL, NULL, ch->prm, &a);
    if (ok && g.swa && n > 1) {        /* the ring keeps the block's last `rows` positions */
        int from = n > g.rows ? n - g.rows : 0;
        ok = mc_ring_copy(ch->kc[i], ch->kn, g.rows, pb, pb + from, pb + n, g.kd, 1) &&
             mc_ring_copy(ch->vc[i], ch->vn, g.rows, pb, pb + from, pb + n, g.vdd, 1);
    }
    /* for the host: every row of a full layer, the ring's share of a windowed one */
    int from = g.swa && n > g.rows ? n - g.rows : 0, cap = g.swa && g.rows < ch->rows ? g.rows : ch->rows;
    ok = ok && vkc_copy(ch->kvd, ch->o_kvd[i], ch->kn, (size_t)from * g.kd, (size_t)(n - from) * g.kd) &&
         vkc_copy(ch->kvd, ch->o_kvd[i] + (size_t)cap * g.kd, ch->vn, (size_t)from * g.vdd, (size_t)(n - from) * g.vdd);
    return ok && vkc_matmul((ColiVkTensor *)l->o.vk, ch->ctx, 0, ch->tmp, 0, n);
}
/* layer 0's dense MLP: tmp = down(silu(gate(nrm)) * up(nrm)) */
static int mc_dense_mlp(MimoChain *ch, Model *m, int i, int n) {
    Layer *l = &m->L[i];
    int F = l->gate.O;
    VkcEw p = {VKC_EW_SWIGLU, n * F, F, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_matmul((ColiVkTensor *)l->gate.vk, ch->nrm, 0, ch->g, 0, n) &&
           vkc_matmul((ColiVkTensor *)l->up.vk, ch->nrm, 0, ch->u, 0, n) &&
           vkc_ew(ch->g, ch->g, ch->u, NULL, NULL, &p) &&
           vkc_matmul((ColiVkTensor *)l->down.vk, ch->g, 0, ch->tmp, 0, n);
}

/* The rows the step at pb wrote, into the host's caches (the CPU's own placement). */
static void mc_commit(MimoChain *ch, Model *m, int n, int pb) {
    const float *kv = (const float *)vkc_ptr(ch->kvd);
    for (int i = ch->lo; i < ch->lo + ch->nl; i++) {
        McGeo g = mc_geo(m, i);
        Layer *l = &m->L[i];
        int from = g.swa && n > g.rows ? n - g.rows : 0, cap = g.swa && g.rows < ch->rows ? g.rows : ch->rows;
        const float *K = kv + ch->o_kvd[i], *V = K + (size_t)cap * g.kd;
        for (int t = from; t < n; t++) {
            int p = pb + t, slot = g.swa ? p % g.rows : p;
            memcpy(l->K + (size_t)slot * g.kd, K + (size_t)(t - from) * g.kd, (size_t)g.kd * sizeof(float));
            memcpy(l->V + (size_t)slot * g.vdd, V + (size_t)(t - from) * g.vdd, (size_t)g.vdd * sizeof(float));
            if (g.swa) l->ring_pos[slot] = p;
        }
        ch->kv_valid[i] = pb + n;
        if (ch->kvi[i] >= 0) vkc_kv_done(&ch->ks, ch->kvi[i], pb + n);
    }
}

/* One chain's layers (lo..lo+nl-1, on its device) for one chunk of n rows at pb, from
 * the residual rows in hr: their K/V rows into the host's caches, and either the rows'
 * logits (ro of them into lc: a chain with the head) or the residual after its last layer
 * back in hr. 0: a frame failed (the device lost). */
static int mc_chunk(MimoChain *ch, Model *m, float *hr, int n, int pb, int ro, float *lc) {
    Cfg *c = &m->c; int H = c->hidden, lo = ch->lo, hi = lo + ch->nl, V = c->vocab;
    if (!vkc_begin() || !vkc_write(ch->x, 0, hr, (size_t)n * H * sizeof(float)) || !mc_push_state(ch, m, pb, n)) return 0;
    float *cs = (float *)vkc_ptr(ch->cs);   /* the CPU's angles, cosines and sines (rope()) */
    for (int k = 0; k < 2; k++) {
        int rd = c->rope_dim[k], half = rd / 2;
        for (int s = 0; s < n; s++) for (int j = 0; j < half; j++) {
            float inv = 1.0f / powf(c->theta[k], (float)(2 * j) / (float)rd);
            float ang = inv * (float)(pb + s);
            cs[ch->rd_off[k] + (s * half + j) * 2] = cosf(ang);
            cs[ch->rd_off[k] + (s * half + j) * 2 + 1] = sinf(ang);
        }
    }
    int ok = 1, pending = 0;
    for (int i = lo; i < hi && ok; i++) {
        if (pending) ok = mc_add(ch->x, ch->routed, n * H);   /* the routed experts join, as h += out */
        pending = 0;
        ok = ok && mc_norm(ch->x, 0, ch->prm, ch->o_ln1[i], ch->nrm, 0, n, H, c->eps) &&
             mc_attention(ch, m, i, n, pb) && mc_add(ch->x, ch->tmp, n * H) &&
             mc_norm(ch->x, 0, ch->prm, ch->o_ln2[i], ch->nrm, 0, n, H, c->eps);
        if (ok && !c->moe[i]) ok = mc_dense_mlp(ch, m, i, n) && mc_add(ch->x, ch->tmp, n * H);
        else if (ok) {
            ok = vkc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)n * H);
            double t0 = now_s();
            /* while the frame runs, the tier loads the experts this layer will likely
             * stream (a big prompt chunk only) */
            ok = ok && vkc_submit(0);
            if (ok) vkt_stream_prefetch(i, n);
            ok = ok && vkc_finish();
            ch->frames++; ch->wait_ms += (now_s() - t0) * 1e3;
            if (!ok) break;
            double t1 = now_s();
            moe(m, i, (const float *)vkc_ptr(ch->h2d), n, ch->host_out);
            memcpy(vkc_ptr(ch->routed), ch->host_out, (size_t)n * H * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = vkc_begin();
            pending = 1;
        }
    }
    if (ok && pending) ok = mc_add(ch->x, ch->routed, n * H);
    if (ok && ro) ok = mc_norm(ch->x, (size_t)(n - ro) * H, ch->prm, ch->o_final, ch->fin, 0, ro, H, c->eps) &&
                       vkc_matmul((ColiVkTensor *)m->head.vk, ch->fin, 0, ch->outd, 0, ro);
    if (ok && !ch->full) ok = vkc_copy(ch->h2d, 0, ch->x, 0, (size_t)n * H);   /* the handoff to what follows */
    double t0 = now_s();
    ok = ok && vkc_submit(1);
    ch->frames++; ch->wait_ms += (now_s() - t0) * 1e3;
    if (!ok) return 0;
    mc_commit(ch, m, n, pb);
    if (ro) memcpy(lc, vkc_ptr(ch->outd), (size_t)ro * V * sizeof(float));
    if (!ch->full) memcpy(hr, vkc_ptr(ch->h2d), (size_t)n * H * sizeof(float));
    return 1;
}

/* Every layer for the S rows h (the embedded rows, pictures spliced) at pos_base, the
 * logits as forward() wants them (NULL: none; all_rows: [S][V], else the last row's).
 * Returns how many rows it computed: S, or fewer when a device was lost or the chain
 * declined -- the host's caches then hold every position before pos_base + that many,
 * and the CPU runs the rest. Each chunk crosses the primary's layers, then (with one)
 * the second device's from the rows the primary's brought down; a partial chain's last
 * chain brings the chunk's residual rows down with the frame that ends it, and the CPU
 * runs the layers after it (layers_cpu) and the head (head_cpu) on them before the next
 * chunk, in xn and tmp (forward()'s rows): a chunk counts as computed once every side ran
 * it, so a device lost in a later chunk loses nothing any side holds. */
static int mc_forward(Model *m, float *h, int S, int pos_base, float *logits, int all_rows, float *xn, float *tmp) {
    if (!g_vk_chain) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && S <= 2) return 0;   /* prompts only: decode on the CPU */
    MimoChain *ch = mc_setup(m), *ch2 = NULL;
    if (!ch || ch->failed) return 0;
    if (g_mc_fit2_on && !ch->full && ch->nl == g_mc_fit.n) {
        vkc_device(1);
        ch2 = mc_setup_dev(m, 1);
        vkc_device(0);
        if (ch2 && (ch2->failed || ch2->lo != ch->lo + ch->nl)) ch2 = NULL;
    }
    MimoChain *last_ch = ch2 ? ch2 : ch;
    Cfg *c = &m->c; int H = c->hidden, V = c->vocab, full = last_ch->full, done_layers = last_ch->lo + last_ch->nl;
    int CH = mc_chunk_rows(ch, m), rows = S < CH ? S : CH;
    if (ch->ks.on && rows > ch->ks.chunk) rows = ch->ks.chunk;   /* a step's rows fit the split's window */
    if (ch2) {
        vkc_device(1);
        int CH2 = mc_chunk_rows(ch2, m);
        vkc_device(0);
        if (rows > CH2) rows = CH2;
        if (ch2->ks.on && rows > ch2->ks.chunk) rows = ch2->ks.chunk;
    }
    int out = logits && all_rows ? rows : 1;
    int sc = mc_scratch(ch, m, rows, ch->full ? out : 1);
    if (sc && ch2) { vkc_device(1); sc = mc_scratch(ch2, m, rows, ch2->full ? out : 1); vkc_device(0); }
    if (!sc) {
        fprintf(stderr, "[VK] mimo chain: device memory for %d rows refused; per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    vkc_gemm_rows(-1);
    int done = 0;
    MimoChain *at = ch;
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0, last = c0 + n == S;
        int ro = !full ? 0 : logits && all_rows ? n : logits && last ? 1 : 0;
        float *hr = h + (size_t)c0 * H, *lc = all_rows ? logits + (size_t)c0 * V : logits;
        at = ch;
        /* the chunk's rows as they came in: the primary's residual replaces them, and if the
         * second device is lost the CPU runs the chunk from these */
        if (ch2) {
            float *keep = (float *)realloc(ch2->host_in, (size_t)rows * H * sizeof(float));
            if (!keep) goto lost;
            ch2->host_in = keep;
            memcpy(keep, hr, (size_t)n * H * sizeof(float));
        }
        if (!mc_chunk(ch, m, hr, n, pb, ch2 ? 0 : ro, lc)) goto lost;
        if (ch2) {
            at = ch2;
            vkc_device(1);
            int ok = mc_chunk(ch2, m, hr, n, pb, ro, lc);
            vkc_device(0);
            if (!ok) { memcpy(hr, ch2->host_in, (size_t)n * H * sizeof(float)); goto lost; }
        }
        if (!full) {   /* the CPU's layers and the head on the chunk's rows */
            double t1 = now_s();
            layers_cpu(m, hr, n, pb, done_layers, xn, tmp, NULL);
            float *lcc = logits && all_rows ? logits + (size_t)c0 * V : logits && last ? logits : NULL;
            if (lcc) head_cpu(m, hr, n, lcc, all_rows, xn);
            last_ch->host_ms += (now_s() - t1) * 1e3;
        }
        done = c0 + n;
    }
    ch->forwards++;
    if (ch2) ch2->forwards++;
    return S;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(at->d); }
    vkc_device(0);
    g_vk_chain = 0;
    ch->failed = 1;
    if (ch2) ch2->failed = 1;
    fprintf(stderr, "[VK] mimo chain: the device was lost at position %d; the host's caches hold every "
                    "position before it, and the CPU runs from there on\n", pos_base + done);
    return done;
}

static void mc_report(Model *m, const char *what) {
    for (int d = 0; d < 2; d++) {
        MimoChain *ch = mc_of(m, d);
        if (!ch || !ch->ok || !ch->forwards) continue;
        int was = vkc_device(d);
        VkcStats st; vkc_stats(&st);
        fprintf(stderr, "[VK] %s chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                        "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device (%s)\n",
                mc_name(ch), ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, ch->wait_ms, ch->host_ms,
                st.dev_bytes / 1048576.0, what);
        vkc_kv_report(&ch->ks);
        vkc_prof_print();
        vkc_device(was);
    }
}
