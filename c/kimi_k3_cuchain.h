/* kimi_k3_cuchain.h -- Kimi K3's layers on the CUDA chain (cuda_chain.h), included once by kimi_k3.c
 * in a COLI_CUDA build before step_chunk_ex, which calls k3u_forward in place of the CPU's first
 * layers. kimi_k3_chain.h's layout, op for op on the CUDA chain's ops: GLM-5.3's KDA (the K3 form)
 * and MLA pieces, GLM-5.2's indexer selection, and Kimi K3's residual mix and SiTU-GLU
 * (cuda_k3c.cuh). Per layer, for one block of rows (decode: one row):
 *   device, frame A1: the previous layer's MoE output joins the prefix (the host's routed sum: its
 *     RMSNorm and the latent up-projection, plus the shared experts, then the add: the CPU's order);
 *     the attention site's residual mix, a block boundary's snapshot, the input RMSNorm; the KDA
 *     layer (q, k, v, the output gate, the decay's two f32 matrices and beta's; the short
 *     convolution with its window, the delta rule with its state, o_proj) or the gated MLA layer
 *     (q_a, its norm, q_b; kv_a, the latent norm into the latent cache, the rope part as it is
 *     (Kimi K3's MLA is NoPE); Kimi K3.1's indexer on a DSA layer: the index keys with their
 *     LayerNorm into the key cache and, past index_topk positions, every row's selection; the
 *     absorbed query from kv_b's key rows, the core over the cache or the selection, the values
 *     times the gate, o_proj); the prefix update; the MLP site's mix and post-attention RMSNorm;
 *     then a dense layer's MLP (SiTU-GLU) and its add, or the router's logits (the f32 router)
 *     and the latent down-projection. Then the frame is waited for.
 *   host: the router's sigmoid and top-k with the correction bias, K3_TOPP, the routed experts
 *     (the CUDA tier's batch and the CPU's share: k3_moe_experts), the new MLA rows (and index
 *     keys) into the host's cache.
 *   device, frame A2 (not waited for): the shared experts (SiTU-GLU at full width).
 * After the chain's last layer each row's AttnRes state (the prefix and the block snapshots)
 * comes back, and the CPU runs the layers after it, then the output mix, the final norm and the
 * head (lm_head on the device through cu_dense.h).
 *
 * The layers on the card are the first N whose matrices the dense budget holds (cu_dense.h:
 * every dense matrix where they fit with the chain's own bytes, else its first layers), or
 * COLI_CUDA_CHAIN_LAYERS of them; COLI_CUDA_CHAIN=0: the CPU's layers.
 *
 * State and who owns it (one Model): the host's MLA caches (Lc, Rc and the indexer's Ic) stay
 * canonical (every step copies its new rows back); the device mirrors them behind a watermark
 * per layer that a CPU forward, a reset and a grown cache lower. The KDA state and the three
 * convolution windows of the chain's KDA layers live on the device while the chain runs: `where`
 * says which side holds the newest copy, brought back before anything reads it on the host (a
 * photo, a parked conversation, a CPU forward) and pushed up after anything writes it there.
 * A device lost while it holds the newest KDA state: the state is rebuilt on the CPU from the
 * host's copy (current at host_pos) and the prefix record's ids, and the CPU runs from there.
 *
 * The chain declines (the CPU's layers, the state synced first): for the validation dumps that
 * read every layer on the host (K3_TRACE, K3_VALIDATE_LAYER, K3_DEBUG_OUT), several conversations
 * at once (KV_SLOTS), a GGUF's blocks (on the CUDA chain later), a geometry past its ops' limits
 * and memory the device refuses. */
#ifndef KIMI_K3_CUCHAIN_H
#define KIMI_K3_CUCHAIN_H
#include "cuda_chain.h"

#define K3U_HOST 0
#define K3U_DEV  1
#define K3U_BOTH 2
#define K3U_ROWS 64   /* a prompt chunk's rows through the chain (COLI_CUDA_K3_ROWS) */

typedef struct { ColiCudaTensor *q, *k, *v, *g, *o, *fa, *fb, *bp; } K3uKda;
typedef struct { ColiCudaTensor *qa, *qb, *kva, *o, *g, *iwq, *iwk, *iwp; CcBuf *kvb; } K3uMla;   /* kvb: kv_b's rows as floats */
typedef struct { ColiCudaTensor *router, *down, *up, *sg, *su, *sd, *dg, *du, *dd; } K3uFfn;

typedef struct {
    int ok, failed, rows, cap, n, nbmax;
    CcBuf *prm;
    size_t *o_in, *o_post, *o_asw, *o_msw, *o_qn, *o_kn, *o_conv, *o_kda, *o_latn, *o_ikw, *o_ikb;
    K3uKda *kda; K3uMla *mla; K3uFfn *ffn;
    CcBuf **lat, **rope, **ik, **win, **st;
    int *kv_valid, *mla_ord, n_mla;
    int where, host_zero, dev_pos, host_pos;   /* the KDA state: who holds the newest, where each side is */
    CcBuf *x, *bres, *hm, *nrm, *att, *qkv3, *cm, *t1, *kf, *kb, *kg, *ky, *gate, *lg, *z;
    CcBuf *gs, *us, *hs, *ds, *un, *lu, *qa, *q, *kv, *qabs, *clat, *ctx;
    CcBuf *ikd, *iq, *ihw, *isc, *sel;
    CcBuf *lgd, *zd, *kvd, *routed, *xo, *bo;
    size_t kvd_layer;
    float *host_u, *host_sco;
    unsigned long long forwards;
    double host_ms;
} K3U;

static K3U *g_k3u;
static int g_k3u_on;    /* the chain set up, its layers on the card */

static int k3u_rows_env(void) {
    static int r = -1;
    if (r < 0) { const char *e = getenv("COLI_CUDA_K3_ROWS"); r = e && atoi(e) > 0 ? atoi(e) : K3U_ROWS; }
    return r;
}
static int k3u_res(CcBuf **b, size_t floats, int kind) { return cc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }
static int k3u_dsa(const Model *m, int i) { return m->c.index_hd > 0 && m->c.idx_type[i] && !m->L[i].kda; }

/* ---- the matrices ------------------------------------------------------------------------ */
/* w's CUDA format (cud_mm's: 0 f32, 1 int8 rows, 4 int4 groups); -1 none (a GGUF's blocks) */
static int k3u_form(const W *w) {
    if (w->O < 1 || w->I < 1 || w->mapped) return -1;
    switch (w->fmt) {
    case 0: return w->f ? 0 : -1;
    case 1: return w->q8 && w->s ? 1 : -1;
    case 4: return w->q4 && w->s && w->gs > 0 && !(w->gs & 1) ? 4 : -1;
    default: return -1;
    }
}
/* w's device copy (cud_mm's, w->cu: the CPU path's calls reuse it), uploaded now within the
 * dense budget; NULL: not on the card */
static ColiCudaTensor *k3u_tensor(W *w) {
    if (w->cu) return (ColiCudaTensor *)w->cu;
    if (w->cu_no || !g_cud.on) return NULL;
    const int f = k3u_form(w);
    if (f < 0) return NULL;
    const size_t b = cud_bytes(f, w->O, w->I, f == 4 ? w->gs : 0);
    if (!b || g_cud.used + b > g_cud.budget) return NULL;
    ColiCudaTensor *t = NULL;
    int up = 0;
    if (f == 0) up = coli_cuda_tensor_upload(&t, w->f, NULL, 0, w->I, w->O, g_cud.dev);
    else if (f == 1) up = coli_cuda_tensor_upload(&t, w->q8, w->s, 1, w->I, w->O, g_cud.dev);
    else up = coli_cuda_tensor_upload_g(&t, w->q4, w->s, 4, w->I, w->O, g_cud.dev, w->gs);
    if (!up || !t) { if (t) coli_cuda_tensor_free(t); return NULL; }
    w->cu = t;
    g_cud.used += b;
    g_cud.mats++;
    return t;
}
/* an f32 matrix [O][I] of the chain's own (the router, the KDA decay pair, beta) */
static ColiCudaTensor *k3u_f32(const float *w, int I, int O, ColiCudaTensor **own) {
    if (*own) return *own;
    return w && coli_cuda_tensor_upload(own, w, NULL, 0, I, O, g_cud.dev) ? *own : NULL;
}
/* w's rows as floats (kv_b, which the chain's per-head matmuls read so); NULL: none */
static float *k3u_floats(const W *w) {
    const size_t n = (size_t)w->O * w->I;
    const int f = k3u_form(w), I = w->I;
    if (f < 0) return NULL;
    float *fl = (float *)malloc(n * sizeof(float));
    if (!fl) return NULL;
    if (f == 0) memcpy(fl, w->f, n * sizeof(float));
    else if (f == 1) {
        for (int r = 0; r < w->O; r++) for (int i = 0; i < I; i++) fl[(size_t)r * I + i] = (float)w->q8[(size_t)r * I + i] * w->s[r];
    } else {
        const int rb = (I + 1) / 2, ng = (I + w->gs - 1) / w->gs;
        for (int r = 0; r < w->O; r++)
            for (int i = 0; i < I; i++) {
                const uint8_t by = w->q4[(size_t)r * rb + (i >> 1)];
                fl[(size_t)r * I + i] = (float)((int)((i & 1) ? by >> 4 : by & 15) - 8) * w->s[(size_t)r * ng + i / w->gs];
            }
    }
    return fl;
}

/* what the chain does not take (NULL: nothing), from the shapes alone */
static const char *k3u_unsupported(const Model *m) {
    const Cfg *c = &m->c;
    int L = c->n_layers, nbmax = (L + c->res_bs - 1) / c->res_bs, any_kda = 0, any_mla = 0;
    for (int i = 0; i < L; i++) { any_kda |= m->L[i].kda; any_mla |= !m->L[i].kda; }
    if (m->gx) return "a GGUF's blocks (the Vulkan chain runs them)";
    if (g_k3_mux_slots > 1) return "several conversations at once (KV_SLOTS): it keeps one conversation's state on the device";
    if (any_kda && ((c->kda_hd != 16 && c->kda_hd != 32 && c->kda_hd != 64 && c->kda_hd != 128) || c->conv_k > 8))
        return "a KDA head its ops do not take";
    if (any_mla && (c->kv_lora > 1024 || c->qk_rope > 128 || (c->qk_rope & 1) || c->qk_nope > 1024 || c->q_lora < 1))
        return "an attention geometry its ops do not take";
    if (nbmax > 15) return "more AttnRes blocks than its ops take";
    if (c->index_hd > 0 && (c->index_nh > 64 || c->index_nh * c->index_hd > 4096)) return "an indexer its ops do not take";
    return NULL;
}
static int k3u_wanted(const Model *m) {
    const char *e = getenv("COLI_CUDA_CHAIN");
    if (e && *e == '0') return 0;
    if (!xt_cuda() || !cc_available() || !cc_glm53_ready() || !cc_k3_ready()) return 0;   /* the dense budget comes after */
    const char *why = k3u_unsupported(m);
    if (why) { fprintf(stderr, "[CUDA] kimi_k3 chain: %s; the CPU runs the layers\n", why); return 0; }
    return 1;
}
/* the chain's device bytes past the dense matrices' (which cu_dense.h counts): kv_b as floats, the
 * parameters, the KDA states and windows, the MLA caches and index keys at ctx positions, a chunk's
 * scratch; for the tier's room (VktConfig.chain_bytes) and the whole budget (cud_open_whole) */
static size_t k3u_plan_bytes(const Model *m, int ctx) {
    const Cfg *c = &m->c; const size_t L = (size_t)c->n_layers, D = (size_t)c->hidden, P = (size_t)c->kda_proj;
    const size_t K = (size_t)c->kv_lora, R = (size_t)c->qk_rope, H = (size_t)c->n_heads, ID = (size_t)c->index_hd, IH = (size_t)c->index_nh;
    const size_t r = (size_t)k3u_rows_env(), nbmax = (L + c->res_bs - 1) / c->res_bs;
    const size_t MI = (size_t)(c->moe_inter * c->n_shared > c->dense_inter ? c->moe_inter * c->n_shared : c->dense_inter);
    size_t b = 0, fl = 0;
    for (size_t i = 0; i < L; i++) {
        fl += 4 * D;
        if (m->L[i].kda) {
            b += ((size_t)c->kda_hd * D + P * c->kda_hd + (size_t)c->kda_heads * D) * 4;   /* the f32 decay and beta */
            b += (3 * P * c->conv_k + (size_t)c->kda_heads * c->kda_hd * c->kda_hd) * 4;
            fl += 3 * P * c->conv_k + c->kda_heads + P + c->kda_hd;
        } else {
            b += H * (c->qk_nope + c->v_head) * K * 4 + (size_t)ctx * (K + R) * 4;
            fl += (size_t)c->q_lora + K;
            if (k3u_dsa(m, (int)i)) { b += (size_t)ctx * ID * 4; fl += 2 * ID; }
        }
        if (m->L[i].sparse) { b += (size_t)c->n_experts * D * 4; fl += c->latent; }
    }
    size_t row = (5 + nbmax) * D + 3 * P + 3 * P + c->kda_hd + 3 * P + c->kda_heads + H * c->v_head + 2 * (size_t)c->n_experts +
                 2 * (size_t)c->latent + 3 * MI + 2 * D + (size_t)c->latent + (size_t)c->q_lora + H * (c->qk_nope + R) + (K + R) +
                 2 * H * K + H * c->v_head + (1 + nbmax) * D;
    if (c->index_hd > 0) row += ID + IH * ID + IH + (size_t)ctx + 1 + (size_t)c->index_topk;
    b += r * row * 4 + L * r * (K + R + ID) * 4 + fl * 4;
    return b + ((size_t)64 << 20);
}

/* ---- the KDA state between the host and the device --------------------------------------- */
static void k3u_recover(Model *m, int dev, int upto);
/* The host's KDA state brought up to date (before anything reads it there). */
static void k3u_sync_host(Model *m) {
    K3U *ch = g_k3u;
    if (!ch || !ch->ok || ch->where != K3U_DEV) return;
    const Cfg *c = &m->c;
    const size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)c->kda_proj * c->conv_k;
    int ok = cc_finish();
    for (int i = 0; i < ch->n && ok; i++) {
        if (!m->L[i].kda) continue;
        ok = cc_read(ch->st[i], 0, m->kstate[i], ns * sizeof(float)) && cc_read(ch->win[i], 0, m->cwq[i], nw * sizeof(float)) &&
             cc_read(ch->win[i], nw, m->cwk[i], nw * sizeof(float)) && cc_read(ch->win[i], 2 * nw, m->cwv[i], nw * sizeof(float));
    }
    if (!ok) { ch->host_pos = 0; ch->host_zero = 1; k3u_recover(m, 1, ch->dev_pos); return; }
    ch->where = K3U_BOTH; ch->host_pos = ch->dev_pos; ch->host_zero = 0;
}
/* The host wrote its KDA state (zero: model_state_reset's zeros; else a restored photo). A reset
 * also drops the MLA rows: the mirror's watermark goes to 0. */
static void k3u_host_wrote(Model *m, int zero) {
    K3U *ch = g_k3u;
    if (!ch || !ch->ok) return;
    ch->where = K3U_HOST; ch->host_zero = zero;
    if (zero) for (int i = 0; i < m->c.n_layers; i++) ch->kv_valid[i] = 0;
}
/* The host's state rewritten from position 0 (a parked conversation back): everything goes up again. */
static void k3u_host_rewrote(Model *m) {
    K3U *ch = g_k3u;
    if (!ch || !ch->ok) return;
    ch->where = K3U_HOST; ch->host_zero = 0;
    for (int i = 0; i < m->c.n_layers; i++) ch->kv_valid[i] = 0;
}
/* A CPU forward from pos0: the host's state current before it, the device's stale after (its MLA
 * rows from pos0 on). */
static void k3u_cpu_step(Model *m, int pos0) {
    K3U *ch = g_k3u;
    if (!ch || !ch->ok) return;
    k3u_sync_host(m);
    if (!ch->ok) return;
    ch->where = K3U_HOST; ch->host_zero = 0;
    for (int i = 0; i < ch->n; i++) if (ch->kv_valid[i] > pos0) ch->kv_valid[i] = pos0;
}
/* The device was lost: the chain goes off; when it held the newest KDA state (dev) that state is
 * rebuilt on the CPU from the host's copy (current at host_pos) through the prefix record's ids up
 * to `upto`. The CPU runs from here on. */
static void k3u_recover(Model *m, int dev, int upto) {
    K3U *ch = g_k3u;
    g_k3u_on = 0;
    if (!ch) return;
    const int from = ch->host_pos, zero = ch->host_zero, hi = ch->n;
    ch->ok = 0; ch->failed = 1; ch->where = K3U_HOST;
    if (!dev || upto <= from) {
        fprintf(stderr, "[CUDA] kimi_k3 chain: the device was lost; the host's state is current, the CPU runs from here on\n");
        return;
    }
    if (m->kvp.tainted || !m->kvp.fed || m->kvp.len < upto) {
        fprintf(stderr, "[CUDA] kimi_k3 chain: the device was lost with a KDA state its token ids do not describe -- stopping "
                        "(COLI_CUDA_CHAIN=0 keeps the state on the CPU)\n");
        exit(1);
    }
    fprintf(stderr, "[CUDA] kimi_k3 chain: the device was lost; rebuilding the state of %d positions on the CPU, which runs "
                    "from here on\n", upto - from);
    const Cfg *c = &m->c;
    if (zero) for (int i = 0; i < hi; i++) {   /* the host was told "zeros" and kept its old arrays */
        if (!m->L[i].kda) continue;
        memset(m->kstate[i], 0, (size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float));
        memset(m->cwq[i], 0, (size_t)c->kda_proj * c->conv_k * sizeof(float));
        memset(m->cwk[i], 0, (size_t)c->kda_proj * c->conv_k * sizeof(float));
        memset(m->cwv[i], 0, (size_t)c->kda_proj * c->conv_k * sizeof(float));
    }
    const int D = c->hidden, nbmax = ch->nbmax, chunk = 32;
    float *h = falloc((int64_t)chunk * D), *bres = falloc((int64_t)chunk * nbmax * D);
    for (int p = from; p < upto; p += chunk) {
        int n = upto - p < chunk ? upto - p : chunk, nb = 0;
        k3_embed(m, m->kvp.fed + p, p, n, h);
        k3_layers_forward_range(m, h, bres, &nb, p, n, 0, hi, NULL, NULL, NULL);   /* the chain's layers: the CPU's are current */
    }
    free(h); free(bres);
}

/* ---- setup ------------------------------------------------------------------------------- */
/* Layer i's tensors and state on the card; 0 with *why set (the chain stops before it). */
static int k3u_layer_up(K3U *ch, Model *m, int i, const char **why) {
    const Cfg *c = &m->c; Layer *l = &m->L[i];
    const int D = c->hidden, P = c->kda_proj;
    *why = "a matrix past the dense budget (or one with no CUDA form)";
    int ok;
    if (l->kda) {
        Kda *k = &l->a; K3uKda *t = &ch->kda[i];
        ok = (t->q = k3u_tensor(&k->q)) && (t->k = k3u_tensor(&k->k)) && (t->v = k3u_tensor(&k->v)) && (t->g = k3u_tensor(&k->g)) &&
             (t->o = k3u_tensor(&k->o)) && k3u_f32(k->fa, D, c->kda_hd, &t->fa) && k3u_f32(k->fb, c->kda_hd, P, &t->fb) &&
             k3u_f32(k->bp, D, c->kda_heads, &t->bp);
    } else {
        Mla *q = &l->m; K3uMla *t = &ch->mla[i];
        ok = (t->qa = k3u_tensor(&q->qa)) && (t->qb = k3u_tensor(&q->qb)) && (t->kva = k3u_tensor(&q->kva)) &&
             (t->o = k3u_tensor(&q->o)) && (t->g = k3u_tensor(&q->g)) &&
             (!k3u_dsa(m, i) || ((t->iwq = k3u_tensor(&q->wq)) && (t->iwk = k3u_tensor(&q->wk)) && (t->iwp = k3u_tensor(&q->wp))));
        if (ok) {   /* kv_b's rows as floats: the absorbed query reads its key rows, the values its value rows */
            float *fl = k3u_floats(&q->kvb);
            const size_t bytes = (size_t)q->kvb.O * q->kvb.I * sizeof(float);
            ok = fl && (t->kvb = cc_buf(bytes, CC_DEV)) && cc_begin() && cc_write(t->kvb, 0, fl, bytes) && cc_submit(1);
            free(fl);
            if (!ok) *why = "device memory for kv_b's floats refused";
        }
    }
    K3uFfn *f = &ch->ffn[i];
    if (ok && l->sparse) {
        Moe *o = &l->moe;
        ok = k3u_f32(o->router, D, c->n_experts, &f->router) && (f->down = k3u_tensor(&o->lat_down)) &&
             (f->up = k3u_tensor(&o->lat_up)) && (f->sg = k3u_tensor(&o->sh_gate)) && (f->su = k3u_tensor(&o->sh_up)) &&
             (f->sd = k3u_tensor(&o->sh_down));
    } else if (ok)
        ok = (f->dg = k3u_tensor(&l->d_gate)) && (f->du = k3u_tensor(&l->d_up)) && (f->dd = k3u_tensor(&l->d_down));
    if (ok && l->kda) {
        ch->win[i] = cc_buf((size_t)3 * P * c->conv_k * sizeof(float), CC_DEV);
        ch->st[i] = cc_buf((size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float), CC_DEV);
        if (!ch->win[i] || !ch->st[i]) { *why = "device memory for a KDA state refused"; ok = 0; }
    }
    return ok;
}
/* the chain's own buffers of layer i off the card (the dense matrices stay cu_dense.h's) */
static void k3u_layer_free(K3U *ch, int i) {
    K3uKda *k = &ch->kda[i]; K3uFfn *f = &ch->ffn[i];
    coli_cuda_tensor_free(k->fa); coli_cuda_tensor_free(k->fb); coli_cuda_tensor_free(k->bp); k->fa = k->fb = k->bp = NULL;
    coli_cuda_tensor_free(f->router); f->router = NULL;
    cc_free(ch->mla[i].kvb); ch->mla[i].kvb = NULL;
    cc_free(ch->win[i]); cc_free(ch->st[i]); ch->win[i] = ch->st[i] = NULL;
}
/* The chain: the first layers whose matrices fit (COLI_CUDA_CHAIN_LAYERS of them at most), their
 * parameters in one arena. 1 when at least one layer is on the card. */
static int k3u_setup(Model *m) {
    const Cfg *c = &m->c;
    const int L = c->n_layers, D = c->hidden, P = c->kda_proj, LT = c->latent;
    if (!cc_init(g_cud.dev) || !cc_ready()) { fprintf(stderr, "[CUDA] kimi_k3 chain: the chain's context did not come up\n"); return 0; }
    K3U *ch = (K3U *)calloc(1, sizeof *ch);
    if (!ch) return 0;
    size_t **offs[] = {&ch->o_in, &ch->o_post, &ch->o_asw, &ch->o_msw, &ch->o_qn, &ch->o_kn, &ch->o_conv, &ch->o_kda, &ch->o_latn,
                       &ch->o_ikw, &ch->o_ikb};
    for (size_t k = 0; k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = (size_t *)calloc((size_t)L, sizeof(size_t)))) return 0;
    CcBuf ***bufs[] = {&ch->lat, &ch->rope, &ch->ik, &ch->win, &ch->st};
    for (size_t k = 0; k < sizeof bufs / sizeof *bufs; k++) if (!(*bufs[k] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) return 0;
    ch->kda = (K3uKda *)calloc((size_t)L, sizeof(K3uKda)); ch->mla = (K3uMla *)calloc((size_t)L, sizeof(K3uMla));
    ch->ffn = (K3uFfn *)calloc((size_t)L, sizeof(K3uFfn));
    ch->kv_valid = (int *)calloc((size_t)L, sizeof(int)); ch->mla_ord = (int *)calloc((size_t)L, sizeof(int));
    if (!ch->kda || !ch->mla || !ch->ffn || !ch->kv_valid || !ch->mla_ord) return 0;
    ch->nbmax = (L + c->res_bs - 1) / c->res_bs;
    g_k3u = ch;
    int N = L;
    { const char *e = getenv("COLI_CUDA_CHAIN_LAYERS"); if (e && *e && atoi(e) >= 0 && atoi(e) < N) N = atoi(e); }
    const char *why = N < L ? "COLI_CUDA_CHAIN_LAYERS" : NULL;
    for (int i = 0; i < N; i++)
        if (!k3u_layer_up(ch, m, i, &why)) { k3u_layer_free(ch, i); N = i; break; }
    ch->n = N;
    if (N < 1) { fprintf(stderr, "[CUDA] kimi_k3 chain: no layer on the card (%s); the CPU runs the layers\n", why ? why : "none"); return 0; }
    for (int i = 0; i < N; i++) if (!m->L[i].kda) ch->mla_ord[i] = ch->n_mla++;
    /* the parameter arena */
    size_t n = 0;
    for (int i = 0; i < N; i++) {
        const Layer *l = &m->L[i];
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D; ch->o_asw[i] = n; n += D; ch->o_msw[i] = n; n += D;
        if (l->kda) { ch->o_conv[i] = n; n += (size_t)3 * P * c->conv_k; ch->o_kda[i] = n; n += (size_t)c->kda_heads + P + c->kda_hd; }
        else {
            ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += c->kv_lora;
            if (k3u_dsa(m, i)) { ch->o_ikw[i] = n; n += c->index_hd; ch->o_ikb[i] = n; n += c->index_hd; }
        }
        if (l->sparse) { ch->o_latn[i] = n; n += LT; }
    }
    float *a = (float *)calloc(n, sizeof(float));
    int ok = a != NULL;
    for (int i = 0; ok && i < N; i++) {
        const Layer *l = &m->L[i];
        memcpy(a + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(a + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(a + ch->o_asw[i], l->attn_sw, D * sizeof(float));
        memcpy(a + ch->o_msw[i], l->mlp_sw, D * sizeof(float));
        if (l->kda) {
            const size_t ck = (size_t)P * c->conv_k;
            memcpy(a + ch->o_conv[i], l->a.conv_q, ck * sizeof(float));
            memcpy(a + ch->o_conv[i] + ck, l->a.conv_k, ck * sizeof(float));
            memcpy(a + ch->o_conv[i] + 2 * ck, l->a.conv_v, ck * sizeof(float));
            memcpy(a + ch->o_kda[i], l->a.A, c->kda_heads * sizeof(float));
            memcpy(a + ch->o_kda[i] + c->kda_heads, l->a.dt, P * sizeof(float));
            memcpy(a + ch->o_kda[i] + c->kda_heads + P, l->a.onw, c->kda_hd * sizeof(float));
        } else {
            memcpy(a + ch->o_qn[i], l->m.qa_ln, c->q_lora * sizeof(float));
            memcpy(a + ch->o_kn[i], l->m.kva_ln, c->kv_lora * sizeof(float));
            if (k3u_dsa(m, i)) {
                memcpy(a + ch->o_ikw[i], l->m.knw, c->index_hd * sizeof(float));
                memcpy(a + ch->o_ikb[i], l->m.knb, c->index_hd * sizeof(float));
            }
        }
        if (l->sparse) memcpy(a + ch->o_latn[i], l->moe.lat_norm, LT * sizeof(float));
    }
    ok = ok && (ch->prm = cc_buf(n * sizeof(float), CC_DEV)) && cc_begin() && cc_write(ch->prm, 0, a, n * sizeof(float)) && cc_submit(1);
    free(a);
    if (!ok) {
        fprintf(stderr, "[CUDA] kimi_k3 chain: the parameters did not reach the card; the CPU runs the layers\n");
        for (int i = 0; i < N; i++) k3u_layer_free(ch, i);
        ch->n = 0;
        return 0;
    }
    ch->where = K3U_HOST; ch->host_zero = 1;   /* the host's KDA state is the zeros it was allocated with */
    ch->ok = 1;
    int nkda = 0, nsparse = 0, ndsa = 0;
    for (int i = 0; i < N; i++) { nkda += m->L[i].kda; nsparse += m->L[i].sparse; ndsa += k3u_dsa(m, i); }
    fprintf(stderr, "[CUDA] kimi_k3 chain: %d of %d layers on the card (%d KDA, %d MLA%s, %d dense MLP), %d AttnRes blocks, "
                    "%.1f MiB of parameters%s (COLI_CUDA_CHAIN=0: the CPU)\n", N, L, nkda, ch->n_mla,
            ndsa ? " with the DSA indexer" : "", N - nsparse, ch->nbmax, n * 4 / 1048576.0,
            N < L ? (why ? ", the rest on the CPU" : "") : "");
    if (N < L && why) fprintf(stderr, "[CUDA] kimi_k3 chain: layer %d stays on the CPU: %s\n", N, why);
    return 1;
}

/* a chunk's scratch at `rows` rows and the selection's scores at `ctx` positions */
static int k3u_scratch(K3U *ch, const Model *m, int rows, int ctx) {
    const Cfg *c = &m->c;
    const int D = c->hidden, P = c->kda_proj, H = c->n_heads, LT = c->latent, E = c->n_experts, K = c->kv_lora, R = c->qk_rope;
    const int SI = c->moe_inter * c->n_shared, MI = SI > c->dense_inter ? SI : c->dense_inter, IH = c->index_nh, ID = c->index_hd;
    int dsa = 0; for (int i = 0; i < ch->n; i++) dsa |= k3u_dsa(m, i);
    const size_t r = (size_t)rows;
    ch->kvd_layer = r * (K + R + (dsa ? ID : 0));
    int ok = k3u_res(&ch->x, r * D, CC_DEV) && k3u_res(&ch->bres, r * ch->nbmax * D, CC_DEV) && k3u_res(&ch->hm, r * D, CC_DEV) &&
             k3u_res(&ch->nrm, r * D, CC_DEV) && k3u_res(&ch->att, r * D, CC_DEV) && k3u_res(&ch->qkv3, 3 * r * P, CC_DEV) &&
             k3u_res(&ch->cm, r * 3 * P, CC_DEV) && k3u_res(&ch->t1, r * c->kda_hd, CC_DEV) && k3u_res(&ch->kf, r * P, CC_DEV) &&
             k3u_res(&ch->kb, r * c->kda_heads, CC_DEV) && k3u_res(&ch->kg, r * P, CC_DEV) && k3u_res(&ch->ky, r * P, CC_DEV) &&
             k3u_res(&ch->gate, r * H * c->v_head, CC_DEV) && k3u_res(&ch->lg, r * E, CC_DEV) && k3u_res(&ch->z, r * LT, CC_DEV) &&
             k3u_res(&ch->gs, r * MI, CC_DEV) && k3u_res(&ch->us, r * MI, CC_DEV) && k3u_res(&ch->hs, r * MI, CC_DEV) &&
             k3u_res(&ch->ds, r * D, CC_DEV) && k3u_res(&ch->un, r * LT, CC_DEV) && k3u_res(&ch->lu, r * D, CC_DEV) &&
             k3u_res(&ch->qa, r * c->q_lora, CC_DEV) && k3u_res(&ch->q, r * H * (c->qk_nope + R), CC_DEV) &&
             k3u_res(&ch->kv, r * (K + R), CC_DEV) && k3u_res(&ch->qabs, r * H * K, CC_DEV) &&
             k3u_res(&ch->clat, r * H * K, CC_DEV) && k3u_res(&ch->ctx, r * H * c->v_head, CC_DEV) &&
             k3u_res(&ch->lgd, r * E, CC_DOWN) && k3u_res(&ch->zd, r * LT, CC_DOWN) &&
             k3u_res(&ch->kvd, (size_t)(ch->n_mla ? ch->n_mla : 1) * ch->kvd_layer, CC_DOWN) &&
             k3u_res(&ch->routed, r * LT, CC_UP) && k3u_res(&ch->xo, r * D, CC_DOWN) && k3u_res(&ch->bo, r * ch->nbmax * D, CC_DOWN) &&
             (!dsa || (k3u_res(&ch->ikd, r * ID, CC_DEV) && k3u_res(&ch->iq, r * IH * ID, CC_DEV) && k3u_res(&ch->ihw, r * IH, CC_DEV) &&
                       k3u_res(&ch->isc, r * (size_t)ctx, CC_DEV) && k3u_res(&ch->sel, r * (1 + (size_t)c->index_topk), CC_DEV)));
    if (!ok) return 0;
    if (ch->rows < rows) {
        float *hu = (float *)realloc(ch->host_u, r * LT * sizeof(float));
        if (hu) ch->host_u = hu;
        float *hs = hu ? (float *)realloc(ch->host_sco, r * E * sizeof(float)) : NULL;
        if (hs) ch->host_sco = hs;
        if (!hu || !hs) return 0;
        ch->rows = rows;
    }
    return 1;
}
/* the MLA mirror at the host's capacity (m->max_t), filled again when that grows */
static int k3u_mirror(K3U *ch, const Model *m) {
    const Cfg *c = &m->c;
    if (ch->cap == m->max_t) return 1;
    for (int i = 0; i < ch->n; i++) {
        if (m->L[i].kda) continue;
        cc_free(ch->lat[i]); cc_free(ch->rope[i]); cc_free(ch->ik[i]);
        ch->lat[i] = ch->rope[i] = ch->ik[i] = NULL;
        ch->kv_valid[i] = 0;
    }
    ch->cap = 0;
    for (int i = 0; i < ch->n; i++) {
        if (m->L[i].kda) continue;
        ch->lat[i] = cc_buf((size_t)m->max_t * c->kv_lora * sizeof(float), CC_DEV);
        if (c->qk_rope > 0) ch->rope[i] = cc_buf((size_t)m->max_t * c->qk_rope * sizeof(float), CC_DEV);
        if (k3u_dsa(m, i)) ch->ik[i] = cc_buf((size_t)m->max_t * c->index_hd * sizeof(float), CC_DEV);
        if (!ch->lat[i] || (c->qk_rope > 0 && !ch->rope[i]) || (k3u_dsa(m, i) && !ch->ik[i])) return 0;
    }
    ch->cap = m->max_t;
    return 1;
}
/* the uploads that make the device the host's below pb (into the open frame) */
static int k3u_push(K3U *ch, const Model *m, int pb) {
    const Cfg *c = &m->c; const int K = c->kv_lora, R = c->qk_rope, ID = c->index_hd;
    int ok = 1;
    if (ch->where == K3U_HOST) {
        const size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)c->kda_proj * c->conv_k;
        for (int i = 0; i < ch->n && ok; i++) {
            if (!m->L[i].kda) continue;
            if (ch->host_zero) ok = cc_zero(ch->st[i], 0, ns) && cc_zero(ch->win[i], 0, 3 * nw);
            else ok = cc_write(ch->st[i], 0, m->kstate[i], ns * sizeof(float)) && cc_write(ch->win[i], 0, m->cwq[i], nw * sizeof(float)) &&
                      cc_write(ch->win[i], nw, m->cwk[i], nw * sizeof(float)) && cc_write(ch->win[i], 2 * nw, m->cwv[i], nw * sizeof(float));
        }
        ch->where = K3U_BOTH; ch->host_pos = pb;
    }
    for (int i = 0; i < ch->n && ok; i++) {
        if (m->L[i].kda || ch->kv_valid[i] >= pb) continue;
        const int t0 = ch->kv_valid[i], n = pb - t0;
        ok = cc_write(ch->lat[i], (size_t)t0 * K, m->Lc[i] + (size_t)t0 * K, (size_t)n * K * sizeof(float)) &&   /* offsets in floats */
             (R == 0 || cc_write(ch->rope[i], (size_t)t0 * R, m->Rc[i] + (size_t)t0 * R, (size_t)n * R * sizeof(float))) &&
             (!k3u_dsa(m, i) || cc_write(ch->ik[i], (size_t)t0 * ID, m->L[i].m.Ic + (size_t)t0 * ID, (size_t)n * ID * sizeof(float)));
        ch->kv_valid[i] = pb;
    }
    return ok;
}
/* the new MLA rows (and index keys) of layers [from, to) into the host's cache */
static void k3u_pull(K3U *ch, Model *m, int from, int to, int pb, int n) {
    const Cfg *c = &m->c; const int K = c->kv_lora, R = c->qk_rope, ID = c->index_hd;
    for (int i = from; i < to; i++) {
        if (m->L[i].kda) continue;
        const float *kv = (const float *)cc_ptr(ch->kvd) + (size_t)ch->mla_ord[i] * ch->kvd_layer;
        memcpy(m->Lc[i] + (size_t)pb * K, kv, (size_t)n * K * sizeof(float));
        if (R) memcpy(m->Rc[i] + (size_t)pb * R, kv + (size_t)n * K, (size_t)n * R * sizeof(float));
        if (k3u_dsa(m, i)) memcpy(m->L[i].m.Ic + (size_t)pb * ID, kv + (size_t)n * (K + R), (size_t)n * ID * sizeof(float));
    }
}

/* ---- one layer's pieces ---------------------------------------------------------------- */
static int k3u_norm(CcBuf *x, CcBuf *w, size_t wo, CcBuf *y, int rows, int D, float eps) {
    CcNorm p = {rows, D, 1, 0, D, D, 0, D, D, (int)wo, 0, 0, eps, 1.f};
    return cc_norm(x, w, y, &p);
}
static int k3u_mix(K3U *ch, const Model *m, size_t wo, int n, int nb) {   /* res_mix(hm, x, bres) */
    CcAres a = {n, m->c.hidden, nb, 0, m->c.hidden, 0, ch->nbmax * m->c.hidden, (int)wo, 0, m->c.hidden, m->c.eps};
    return cc_ares_mix(ch->x, ch->bres, ch->prm, ch->hm, &a);
}
static int k3u_kda(K3U *ch, const Model *m, int i, int n) {
    const Cfg *c = &m->c; const int P = c->kda_proj, H = c->kda_heads, hd = c->kda_hd, r = ch->rows;
    const K3uKda *t = &ch->kda[i];
    CcKdaConv cp = {n, 3 * P, c->conv_k, P, 0, P, r * P, 0, 3 * P, (int)ch->o_conv[i], 0};
    CcKdaRec rp = {n, H, hd, P, 0, 3 * P, 0, P, 0, H, 0, P, 0, P, 0, (int)ch->o_kda[i], c->gate_lb, 1e-6f, c->eps};
    return cc_matmul(t->q, ch->nrm, 0, ch->qkv3, 0, n) && cc_matmul(t->k, ch->nrm, 0, ch->qkv3, (size_t)r * P, n) &&
           cc_matmul(t->v, ch->nrm, 0, ch->qkv3, (size_t)2 * r * P, n) && cc_matmul(t->g, ch->nrm, 0, ch->kg, 0, n) &&
           cc_matmul(t->fa, ch->nrm, 0, ch->t1, 0, n) && cc_matmul(t->fb, ch->t1, 0, ch->kf, 0, n) &&
           cc_matmul(t->bp, ch->nrm, 0, ch->kb, 0, n) &&
           cc_kda_conv(ch->qkv3, ch->prm, ch->win[i], ch->cm, &cp) &&
           cc_kda_rec(hd, ch->cm, ch->kf, ch->kb, ch->kg, ch->prm, ch->st[i], ch->ky, &rp, 1 | 2) &&   /* VKC_KDA_EXP_A | VKC_KDA_K3 */
           cc_matmul(t->o, ch->ky, 0, ch->att, 0, n);
}
/* the gated MLA layer (vk_chain.c's vkc_mla_qkv and vkc_mla_attn, NoPE: the rope part as it is),
 * with Kimi K3.1's indexer on a DSA layer (kimi_k3_chain.h's k3c_dsa_select) */
static int k3u_mla(K3U *ch, const Model *m, int i, int n, int pb) {
    const Cfg *c = &m->c; const K3uMla *t = &ch->mla[i];
    const int H = c->n_heads, Q = c->qk_nope, R = c->qk_rope, V = c->v_head, K = c->kv_lora, QL = c->q_lora;
    const int QR = Q + R, KR = K + R, HK = H * K, HV = H * V;
    const size_t ko = (size_t)ch->mla_ord[i] * ch->kvd_layer;
    CcNorm qn = {n, QL, 1, 0, QL, QL, 0, QL, QL, (int)ch->o_qn[i], 0, 0, c->eps, 1.f};
    CcNorm ln = {n, K, 1, 0, KR, KR, pb * K, K, K, (int)ch->o_kn[i], 0, 0, c->eps, 1.f};
    int ok = cc_matmul(t->qa, ch->nrm, 0, ch->qa, 0, n) && cc_norm(ch->qa, ch->prm, ch->qa, &qn) &&
             cc_matmul(t->qb, ch->qa, 0, ch->q, 0, n) && cc_matmul(t->kva, ch->nrm, 0, ch->kv, 0, n) &&
             cc_norm(ch->kv, ch->prm, ch->lat[i], &ln);
    for (int s0 = 0; ok && R > 0 && s0 < n; s0 += 64) {   /* the rope rows as kv_a gives them */
        CcRegion rg[64];
        const int k = n - s0 < 64 ? n - s0 : 64;
        for (int s = 0; s < k; s++) rg[s] = (CcRegion){(size_t)(pb + s0 + s) * R, (size_t)(s0 + s) * KR + K, (size_t)R};
        ok = cc_copy_regions(ch->rope[i], ch->kv, rg, k);
    }
    ok = ok && cc_copy(ch->kvd, ko, ch->lat[i], (size_t)pb * K, (size_t)n * K) &&
         (R == 0 || cc_copy(ch->kvd, ko + (size_t)n * K, ch->rope[i], (size_t)pb * R, (size_t)n * R)) &&
         cc_matmul(t->g, ch->nrm, 0, ch->gate, 0, n);
    int active = 0;
    if (ok && k3u_dsa(m, i)) {   /* the index keys, then past index_topk every row's selection */
        const int IH = c->index_nh, ID = c->index_hd;
        CcMlaRow kl = {n, 1, 0, ID, 0, 0, ID, 0, pb * ID, ID, 0, 0, 0, (int)ch->o_ikw[i], (int)ch->o_ikb[i], 1, 1e-6f};
        ok = cc_matmul(t->iwk, ch->nrm, 0, ch->ikd, 0, n) && cc_mla_lnorm(ch->ikd, ch->prm, ch->ik[i], &kl) &&
             cc_copy(ch->kvd, ko + (size_t)n * KR, ch->ik[i], (size_t)pb * ID, (size_t)n * ID);
        active = pb + n > c->index_topk;
        if (ok && active) {
            CcDsaSel ds = {n, pb, IH, ID, c->index_topk, 0, 0, IH * ID, 0, IH, 0, pb + n, 1 + c->index_topk,
                           1.f / sqrtf((float)ID), 1.f / sqrtf((float)IH)};
            ok = cc_matmul(t->iwq, ch->qa, 0, ch->iq, 0, n) && cc_matmul(t->iwp, ch->nrm, 0, ch->ihw, 0, n) &&
                 cc_dsa_select(ch->iq, ch->ihw, ch->ik[i], ch->isc, ch->sel, &ds);
        }
    }
    CcHgemv ab = {1, n, H, Q, Q + V, 0, 0, H * QR, QR, 0, HK, K, 0, 0, 0, K, 0};
    CcMlaCore cp = {n, H, K, R, pb, 0, 0, HK, K, Q, H * QR, QR, 0, K, 0, R, 0, active ? 1 + c->index_topk : 0, 0, HK, K,
                    c->attn_scale};
    CcHgemv va = {0, n, H, V, Q + V, Q, 0, HK, K, 0, HV, V, 0, HV, 1, K, 0};
    return ok && cc_mla_hgemv(t->kvb, ch->q, ch->qabs, NULL, &ab) &&
           cc_mla_core(ch->qabs, ch->q, ch->lat[i], R > 0 ? ch->rope[i] : NULL, active ? ch->sel : NULL, ch->clat, &cp) &&
           cc_mla_hgemv(t->kvb, ch->clat, ch->ctx, ch->gate, &va) && cc_matmul(t->o, ch->ctx, 0, ch->att, 0, n);
}
/* SiTU-GLU MLP of nrm into out: gate, up, the activation, down */
static int k3u_glu(K3U *ch, const Model *m, ColiCudaTensor *g, ColiCudaTensor *u, ColiCudaTensor *d, int inter, CcBuf *out, int n) {
    CcSitu sp = {n * inter, 0, 0, 0, m->c.situ_b1, m->c.situ_b2};
    return cc_matmul(g, ch->nrm, 0, ch->gs, 0, n) && cc_matmul(u, ch->nrm, 0, ch->us, 0, n) && cc_situ(ch->gs, ch->us, ch->hs, &sp) &&
           cc_matmul(d, ch->hs, 0, out, 0, n);
}
/* the MoE output joining the prefix: x += lat_up(rmsnorm(routed)) + shared, moe_forward's order */
static int k3u_join(K3U *ch, const Model *m, int i, int n) {
    const Cfg *c = &m->c; const int D = c->hidden, LT = c->latent;
    CcNorm un = {n, LT, 1, 0, LT, LT, 0, LT, LT, (int)ch->o_latn[i], 0, 0, c->eps, 1.f};
    CcEw cb = {CC_EW_COMBINE, n * D, D, 1, 1 | 2, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_norm(ch->routed, ch->prm, ch->un, &un) && cc_matmul(ch->ffn[i].up, ch->un, 0, ch->lu, 0, n) &&
           cc_ew(ch->x, ch->x, ch->lu, ch->ds, NULL, &cb);
}
/* the host's MoE step of layer li: the router's choice from the device's logits, the routed
 * experts on the device's latent rows, their sum into the upload buffer */
static void k3u_moe_host(K3U *ch, Model *m, int li, int n) {
    const Cfg *c = &m->c; const int E = c->n_experts, K = c->topk, LT = c->latent;
    Moe *o = &m->L[li].moe;
    memcpy(ch->host_sco, cc_ptr(ch->lgd), (size_t)n * E * sizeof(float));
    int *idxs = (int *)malloc((size_t)n * K * sizeof(int)), *keff = (int *)malloc((size_t)n * sizeof(int));
    float *wsels = falloc((int64_t)n * K);
    if (!idxs || !keff) { fprintf(stderr, "OOM moe sel\n"); exit(1); }
    k3_moe_route(m, o, li, ch->host_sco, n, idxs, wsels, keff);
    memset(ch->host_u, 0, (size_t)n * LT * sizeof(float));
    float *sd = k3_moe_experts(m, o, li, NULL, (const float *)cc_ptr(ch->zd), n, keff, idxs, wsels, ch->host_u);
    free(sd); free(idxs); free(keff); free(wsels);
    memcpy(cc_ptr(ch->routed), ch->host_u, (size_t)n * LT * sizeof(float));
}

/* ---- the forward ------------------------------------------------------------------------- */
/* The chain's layers 0..N-1 for C rows (positions pos0..) of `hidden` (the embedded rows): each
 * row's AttnRes state after layer N - 1 back into hidden, bres and *nb_out. Returns N; 0: not
 * taken (the CPU runs every layer, nothing the caller holds changed). On a cancel (poll), N with
 * *cancelled set: the state is the caller's to reset. */
static int k3u_forward(Model *m, float *hidden, float *bres, int *nb_out, int pos0, int C, K3CancelPoll poll, void *pctx,
                       int *cancelled) {
    K3U *ch = g_k3u;
    if (!g_k3u_on || !ch || !ch->ok || ch->failed) return 0;
    if (m->trace || g_k3_val_layer >= 0 || g_k3_dfp || m->mux_rows) {
        static int said = 0;
        if (!said++ && !m->mux_rows)
            fprintf(stderr, "[CUDA] kimi_k3 chain: a validation dump reads every layer on the host; the CPU runs the layers\n");
        return 0;
    }
    const Cfg *c = &m->c; const int N = ch->n, D = c->hidden, E = c->n_experts, LT = c->latent, nbx = ch->nbmax;
    const int dev = ch->where == K3U_DEV, dev_start = ch->dev_pos;
    if (!cc_ready()) { k3u_recover(m, dev, dev_start); return 0; }
    const int CH = k3u_rows_env(), rows = C < CH ? C : CH;
    if (!k3u_mirror(ch, m) || !k3u_scratch(ch, m, rows, pos0 + C)) {
        fprintf(stderr, "[CUDA] kimi_k3 chain: device memory for %d rows at %d positions refused; the CPU runs the layers\n",
                rows, m->max_t);
        k3u_cpu_step(m, pos0);
        ch->failed = 1; g_k3u_on = 0;
        return 0;
    }
    float *xh = falloc((int64_t)C * D), *bh = falloc((int64_t)C * nbx * D);
    int nb_end = 0;
    for (int c0 = 0; c0 < C; c0 += rows) {
        const int n = C - c0 < rows ? C - c0 : rows, pb = pos0 + c0;
        if (!cc_begin() || !k3u_push(ch, m, pb) || !cc_write(ch->x, 0, hidden + (size_t)c0 * D, (size_t)n * D * sizeof(float))) goto lost;
        int ok = 1, pending = 0, pulled = 0, nb = 0;
        for (int i = 0; i < N && ok; i++) {
            const Layer *l = &m->L[i];
            if (pending) { ok = k3u_join(ch, m, i - 1, n); pending = 0; }
            const int snap = i % c->res_bs == 0;
            const double ta = now_s();
            CcBuf *src = ch->x;   /* the attention site: the mix over the snapshots so far, this block's snapshot */
            if (ok && nb > 0) { ok = k3u_mix(ch, m, ch->o_asw[i], n, nb); src = ch->hm; }
            for (int s0 = 0; ok && snap && s0 < n; s0 += 64) {
                CcRegion rg[64];
                const int k = n - s0 < 64 ? n - s0 : 64;
                for (int s = 0; s < k; s++) rg[s] = (CcRegion){(size_t)(s0 + s) * nbx * D + (size_t)nb * D, (size_t)(s0 + s) * D, (size_t)D};
                ok = cc_copy_regions(ch->bres, ch->x, rg, k);
            }
            ok = ok && k3u_norm(src, ch->prm, ch->o_in[i], ch->nrm, n, D, c->eps);
            const int have_prefix = !snap;
            if (snap) nb++;
            ok = ok && (l->kda ? k3u_kda(ch, m, i, n) : k3u_mla(ch, m, i, n, pb));
            if (ok) {
                CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = have_prefix ? cc_ew(ch->x, ch->x, ch->att, NULL, NULL, &add) : cc_copy(ch->x, 0, ch->att, 0, (size_t)n * D);
            }
            ok = ok && k3u_mix(ch, m, ch->o_msw[i], n, nb) && k3u_norm(ch->hm, ch->prm, ch->o_post[i], ch->nrm, n, D, c->eps);
            if (!ok) break;
            if (!l->sparse) {   /* a dense layer: its MLP and the add, no host step */
                CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                const K3uFfn *f = &ch->ffn[i];
                ok = k3u_glu(ch, m, f->dg, f->du, f->dd, c->dense_inter, ch->att, n) && cc_ew(ch->x, ch->x, ch->att, NULL, NULL, &add);
                m->t_attn += now_s() - ta;
                continue;
            }
            ok = cc_matmul(ch->ffn[i].router, ch->nrm, 0, ch->lg, 0, n) && cc_matmul(ch->ffn[i].down, ch->nrm, 0, ch->z, 0, n) &&
                 cc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && cc_copy(ch->zd, 0, ch->z, 0, (size_t)n * LT) && cc_submit(0);   /* A1 */
            if (ok) XT(stream_prefetch)(i, n);   /* while it runs, the experts this layer will likely stream (a big prompt chunk) */
            ok = ok && cc_finish();
            m->t_attn += now_s() - ta;
            if (!ok) break;
            k3u_pull(ch, m, pulled, i + 1, pb, n); pulled = i + 1;
            const K3uFfn *f = &ch->ffn[i];   /* A2: the shared experts, while the host computes the routed ones */
            ok = cc_begin() && k3u_glu(ch, m, f->sg, f->su, f->sd, c->moe_inter * c->n_shared, ch->ds, n) && cc_submit(0);
            const double t1 = now_s();
            k3u_moe_host(ch, m, i, n);
            const double dt = now_s() - t1; ch->host_ms += dt * 1e3; m->t_moe += dt;
            if (ok && poll && poll(pctx)) {   /* a cancel: what the device advanced is the caller's to reset */
                if (cancelled) *cancelled = 1;
                cc_finish();
                ch->where = K3U_DEV; ch->dev_pos = pb + n;
                free(xh); free(bh);
                return N;
            }
            ok = ok && cc_begin();
            pending = 1;
        }
        if (ok && pending) ok = k3u_join(ch, m, N - 1, n);
        nb_end = nb;
        ok = ok && cc_copy(ch->xo, 0, ch->x, 0, (size_t)n * D) && cc_copy(ch->bo, 0, ch->bres, 0, (size_t)n * nbx * D) && cc_submit(1);
        if (!ok) goto lost;
        k3u_pull(ch, m, pulled, N, pb, n);
        memcpy(xh + (size_t)c0 * D, cc_ptr(ch->xo), (size_t)n * D * sizeof(float));
        memcpy(bh + (size_t)c0 * nbx * D, cc_ptr(ch->bo), (size_t)n * nbx * D * sizeof(float));
        for (int i = 0; i < N; i++) if (!m->L[i].kda) ch->kv_valid[i] = pb + n;
        ch->where = K3U_DEV;
    }
    ch->dev_pos = pos0 + C;
    ch->forwards++;
    memcpy(hidden, xh, (size_t)C * D * sizeof(float));
    memcpy(bres, bh, (size_t)C * nbx * D * sizeof(float));
    *nb_out = nb_end;
    free(xh); free(bh);
    return N;
lost:   /* a frame failed: the device is gone (or would not take a command) */
    free(xh); free(bh);
    cc_finish();
    ch->where = dev ? K3U_DEV : K3U_HOST;   /* the state as it was when this forward began */
    k3u_recover(m, dev, dev_start);
    return 0;
}
/* a prompt's chunk with the chain on: its rows (0: K3_CHUNK's) */
static int k3u_prefill_rows(const Model *m) {
    (void)m;
    return g_k3u_on && g_k3u && g_k3u->ok && !g_k3u->failed && !getenv("K3_CHUNK") ? k3u_rows_env() : 0;
}
static void k3u_report(void) {
    K3U *ch = g_k3u;
    if (!ch || !ch->forwards) return;
    fprintf(stderr, "[CUDA] kimi_k3 chain: %llu forwards, %d layers on the card, %.1f ms of routed experts on the host\n",
            ch->forwards, ch->n, ch->host_ms);
}
/* At the CUDA run's start, after cu_dense.h's budget and before the tier sizes itself: the layers up. */
static void k3u_start(Model *m) {
    if (!g_cud.on || !k3u_wanted(m)) return;
    g_k3u_on = k3u_setup(m);
}
#endif
