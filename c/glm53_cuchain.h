/* glm53_cuchain.h -- GLM-5.3 Flash's layers on the CUDA chain (cuda_chain.h), included once by
 * glm53.c in a COLI_CUDA build after run_layers, the CPU forward it stands in for, with the
 * routed experts on the CUDA tier (cu_tier.h). The Vulkan chain's layout (glm53_chain.h), op for op
 * on the CUDA chain's ops and GLM-5.3's (cuda_glm53c.cuh), for a whole forward and a decode step:
 *   device, frame A1: the previous layer's FFN branch joins the streams (the host's routed sum
 *     plus the shared expert, then mHC's write back); the attention site: mHC's mix, split and
 *     collapse, the input RMSNorm, then the KDA layer (its projections, the short convolution
 *     with its window, the delta rule with its state, the output norm and gate, o) or the MLA
 *     layer (q's latent and its norm, the kv latent and its norm into the latent cache, the
 *     k-pooled indexer: the index keys with their LayerNorm, the pool gates, each completed
 *     pool's key, every row's pools; the absorbed query, the core over the selection, the
 *     values, o), mHC's write back; the FFN site's mix, collapse and post-attention RMSNorm, and
 *     on a dense layer its MLP (clamped SwiGLU) and write back. Then the frame is waited for.
 *   host: the router and the routed experts (ffn_layer_ex: the tier's batch and the CPU's
 *     share), the new MLA rows into the host's cache.
 *   device, frame A2 (not waited for): the shared expert.
 * The final streams come back to the host, which collapses them and runs the final norm and the
 * head as before.
 *
 * State and who owns it (one session at a time, `owner`): the host's MLA caches stay canonical
 * (every step copies its new rows back) and the device mirrors them behind a watermark that a
 * host write lowers; the KDA state and windows live on the device while the chain runs, `where`
 * saying which side holds the newest, brought back before anything reads the host's (a pin, a
 * state capture, a CPU forward, another session) and pushed up after anything writes it there.
 * The input rows since the host's copy was current are kept: a lost device rebuilds the state on
 * the CPU from them, which runs from there.
 *
 * Every layer or none: the chain declines (the CPU's layers, the state synced first) for a layer
 * range, a geometry its ops do not take (as the Vulkan chain's, glm53_chain.h's g53c_check) or a
 * matrix the CUDA device has no form for, and memory the device refuses. COLI_CUDA_CHAIN=0: the
 * CPU's layers. */
#ifndef GLM53_CUCHAIN_H
#define GLM53_CUCHAIN_H
#include "cuda_chain.h"

#define G53U_HOST 0
#define G53U_DEV  1
#define G53U_BOTH 2
#define G53U_ROWS 256   /* a prompt chunk's rows through the chain (COLI_CUDA_G53_ROWS) */

typedef struct {
    int ok, failed, rows, cap;
    const GSession *owner;
    int where;                                /* G53U_HOST / DEV / BOTH for owner's KDA state */
    CcBuf *prm;
    size_t *o_in, *o_post, *o_hca, *o_hcf, *o_qn, *o_kn, *o_ikw, *o_ikb, *o_ape, *o_conv, *o_kda;
    ColiCudaTensor **fna, **fnf;              /* the mHC mix matrices, f32 */
    CcBuf **kbt, **kbv;                       /* an MLA layer's k_b ([H*K x Q]) and v_b ([H*V x K]) as floats */
    CcBuf **lat, **ik, **ig, **pk, **win, **st;
    int *kv_valid, *pool_valid;
    CcBuf *xs, *xn, *col, *nrm, *br, *mix, *hp, *gs, *us, *hs, *ds, *qkv3, *cm, *kf, *kb, *kg, *klow, *ky;
    CcBuf *ikd, *iq, *ihw, *igd, *isc, *sel, *qa, *q, *kvs, *qabs, *clat, *ctx;
    CcBuf *h2d, *kvd, *xd, *routed;
    size_t kvd_layer;
    float *host_routed;
    float *rec; int rec_base, rec_len, rec_cap, rec_ok;
    unsigned long long forwards;
    double host_ms;
} G53U;

static G53U *g_g53u;
static int g_g53u_off;   /* declined for good */

static int g53u_rows_env(void) {
    static int r = -1;
    if (r < 0) { const char *e = getenv("COLI_CUDA_G53_ROWS"); r = e && atoi(e) > 0 ? atoi(e) : G53U_ROWS; }
    return r;
}
static int g53u_res(CcBuf **b, size_t floats, int kind) { return cc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }

/* ---- the matrices ------------------------------------------------------------------------ */
/* w's CUDA format (cud_mm's: 0 f32, 1 int8 rows, 4 int4 groups, 100 + a ggml type, the rows a
 * matrix the CPU holds 16 interleaved going up as the file's); -1 none */
static int g53u_form(const Mat *w) {
    if (w->rows < 1 || w->columns < 1) return -1;
    switch (w->fmt) {
    case 0: return w->f ? 0 : -1;
    case 1: return w->q8 && w->s ? 1 : -1;
    case 4: return w->q4 && w->s && w->gs > 0 && !(w->gs & 1) ? 4 : -1;
    case G53_GGML: {
        if (!w->q4) return -1;
        const int b = gq_r16_base(w->gs), t = b >= 0 ? b : w->gs, be = coli_cuda_block_elems(100 + t);
        if (b >= 0 && w->rows % 16) return -1;
        return be > 0 && w->columns % be == 0 ? 100 + t : -1;
    }
    default: return -1;
    }
}
/* w's device copy (cud_mm's, w->cu: the CPU path's calls reuse it), uploaded now if it is not there */
static ColiCudaTensor *g53u_tensor(const Mat *wc) {
    Mat *w = (Mat *)wc;
    if (w->cu) return (ColiCudaTensor *)w->cu;
    if (w->cu_no) return NULL;
    const int f = g53u_form(w);
    if (f < 0) return NULL;
    ColiCudaTensor *t = NULL;
    int up = 0;
    if (f == 0) up = coli_cuda_tensor_upload(&t, w->f, NULL, 0, w->columns, w->rows, g_cud.dev);
    else if (f == 1) up = coli_cuda_tensor_upload(&t, w->q8, w->s, 1, w->columns, w->rows, g_cud.dev);
    else if (f == 4) up = coli_cuda_tensor_upload_g(&t, w->q4, w->s, 4, w->columns, w->rows, g_cud.dev, w->gs);
    else {
        const int b = gq_r16_base(w->gs);
        uint8_t *rows = NULL;
        if (b >= 0) {
            if (!(rows = (uint8_t *)malloc((size_t)gq_row_bytes(b, w->columns) * (size_t)w->rows))) return NULL;
            gq_r16_unpack(w->gs, rows, w->q4, w->rows, w->columns);
        }
        up = coli_cuda_tensor_upload(&t, rows ? (const void *)rows : (const void *)w->q4, NULL, f, w->columns, w->rows, g_cud.dev);
        free(rows);
    }
    if (!up || !t) { if (t) coli_cuda_tensor_free(t); return NULL; }
    w->cu = t;
    g_cud.used += cud_bytes(f, w->rows, w->columns, f == 4 ? w->gs : 0);
    g_cud.mats++;
    return t;
}
/* w's rows as floats (the MLA's per-head matrices, which the chain's hgemv reads so); NULL: none */
static float *g53u_floats(const Mat *w) {
    const size_t n = (size_t)w->rows * w->columns;
    float *fl = (float *)malloc(n * sizeof(float));
    if (!fl) return NULL;
    const int f = g53u_form(w), I = w->columns;
    if (f == 0) memcpy(fl, w->f, n * sizeof(float));
    else if (f == 1) {
        for (int r = 0; r < w->rows; r++) for (int i = 0; i < I; i++) fl[(size_t)r * I + i] = (float)w->q8[(size_t)r * I + i] * w->s[r];
    } else if (f == 4) {
        const int rb = (I + 1) / 2, ng = (I + w->gs - 1) / w->gs;
        for (int r = 0; r < w->rows; r++)
            for (int i = 0; i < I; i++) {
                const uint8_t by = w->q4[(size_t)r * rb + (i >> 1)];
                fl[(size_t)r * I + i] = (float)((int)((i & 1) ? by >> 4 : by & 15) - 8) * w->s[(size_t)r * ng + i / w->gs];
            }
    } else if (f >= 100) {
        const int b = gq_r16_base(w->gs), t = b >= 0 ? b : w->gs;
        const int64_t rb = gq_row_bytes(t, I);
        uint8_t *rows = NULL;
        if (b >= 0) {
            if (!(rows = (uint8_t *)malloc((size_t)rb * w->rows))) { free(fl); return NULL; }
            gq_r16_unpack(w->gs, rows, w->q4, w->rows, I);
        }
        const uint8_t *src = rows ? rows : w->q4;
        for (int r = 0; r < w->rows; r++) gq_dequant_row(t, src + (int64_t)r * rb, fl + (size_t)r * I, I);
        free(rows);
    } else { free(fl); return NULL; }
    return fl;
}
static int g53u_layer_mats(const GModel *m, int i, Mat **w) {   /* glm53_chain.h's g53c_layer_mats */
    const Cfg *c = &m->c; GLayer *l = (GLayer *)&m->layer[i]; int n = 0;
    if (c->is_full[i]) {
        Mat *a[] = {&l->qa, &l->qb, &l->kva, &l->kvb_kt, &l->kvb_v, &l->o, &l->iwq, &l->iwk, &l->iwp, &l->ikpg};
        for (size_t k = 0; k < sizeof a / sizeof *a; k++) w[n++] = a[k];
    } else {
        Mat *a[] = {&l->kq, &l->kk, &l->kv, &l->ko, &l->kga, &l->kgb, &l->kfa, &l->kfb, &l->kb};
        for (size_t k = 0; k < sizeof a / sizeof *a; k++) w[n++] = a[k];
    }
    if (i < c->first_dense) { w[n++] = &l->dg; w[n++] = &l->du; w[n++] = &l->dd; }
    else { w[n++] = &l->rg; w[n++] = &l->ru; w[n++] = &l->rd; }
    return n;
}
/* what the chain does not take (NULL: nothing), from the shapes alone */
static const char *g53u_check(const GModel *m) {
    const Cfg *c = &m->c; int L = c->n_layers, H = c->hc_mult, P = c->kda_proj;
    if (m->layer_begin != 0 || m->layer_end != L || !m->has_io) return "a layer range";
    if (c->kv_lora > 1024 || c->qk_rope != 0 || c->qk_nope > 1024 || c->q_lora < 1) return "an attention geometry its ops do not take";
    if (c->index_nh > 64 || c->index_nh * c->index_hd > 4096 || c->index_kpool < 2 || c->index_topk % c->index_kpool ||
        c->index_topk / c->index_kpool > 1024) return "an indexer its ops do not take";
    if (P > 0 && ((c->kda_hd != 16 && c->kda_hd != 32 && c->kda_hd != 64 && c->kda_hd != 128) || c->conv_k > 8)) return "a KDA head its ops do not take";
    if (H < 1 || H > 8 || c->hc_iters < 1) return "hyper-connections its ops do not take";
    Mat *w[16];
    for (int i = 0; i < L; i++) {
        const int n = g53u_layer_mats(m, i, w);
        for (int k = 0; k < n; k++) if (g53u_form(w[k]) < 0) return "a matrix with no CUDA form";
    }
    return NULL;
}
/* the chain's device bytes past the dense matrices' (which cu_dense.h counts already): the mHC
 * mixes, the MLA's per-head matrices as floats, the parameters, the KDA states and windows, the
 * MLA caches at ctx positions, and a chunk's scratch; for the tier's room (VktConfig.chain_bytes) */
static size_t g53u_plan_bytes(const GModel *m, int ctx) {
    const Cfg *c = &m->c; const size_t L = (size_t)c->n_layers, D = (size_t)c->hidden, H = (size_t)c->hc_mult, nm = (2 + H) * H;
    const size_t P = (size_t)c->kda_proj, K = (size_t)c->kv_lora, ID = (size_t)c->index_hd, IH = (size_t)c->index_nh, NH = (size_t)c->n_heads;
    const size_t r = (size_t)g53u_rows_env(), MI = (size_t)(c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter);
    const size_t width = (size_t)coli_sparse_index_width(c->index_topk, c->index_kpool, c->index_kpool_tail);
    size_t b = 0;
    for (size_t i = 0; i < L; i++) {
        b += 2 * nm * H * D * 4 + (2 * D + 2 * (3 + nm)) * 4;
        if (c->is_full[i]) {
            b += (NH * K * c->qk_nope + NH * c->v_head * K) * 4 + ((size_t)c->q_lora + K + 2 * ID + (size_t)c->index_kpool * ID) * 4;
            b += ((size_t)ctx * (K + 2 * ID) + ((size_t)ctx / c->index_kpool + 1) * ID) * 4;
        } else b += (3 * P * c->conv_k * 2 + (size_t)c->kda_heads * c->kda_hd * c->kda_hd + c->kda_heads + P + c->kda_hd) * 4;
    }
    size_t row = 2 * H * D + 4 * D + nm + 2 * H + H * H + 3 * MI + 2 * D + H * D;   /* streams, mixes, the MLP, h2d, xd, routed */
    row += 3 * P + 3 * P + 2 * P + c->kda_heads + c->kda_hd + P;                    /* the KDA layer's */
    row += ID + IH * ID + IH + ID + (size_t)ctx / c->index_kpool + 1 + 1 + width;    /* the indexer's */
    row += (size_t)c->q_lora + NH * c->qk_nope + K + 2 * NH * K + NH * c->v_head;   /* the MLA's */
    b += r * row * 4 + L * r * (K + 2 * ID) * 4;
    return b + ((size_t)64 << 20);
}
static int g53u_wanted(const GModel *m) {
    static int on = -1, said;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN"); on = !(e && *e == '0'); }
    if (!on || g_g53u_off || !xt_cuda() || !g_cud.on || !cc_available() || !cc_glm53_ready() || !cc_dsv4_ready()) return 0;
    if (!said) {   /* the shapes, once */
        said = 1;
        const char *why = g53u_check(m);
        if (why) { fprintf(stderr, "[CUDA] glm53 chain: %s; the CPU runs the layers\n", why); g_g53u_off = 1; return 0; }
    }
    return 1;
}

/* ---- setup ----------------------------------------------------------------------------- */
static void g53u_decline(const char *why) {
    if (!g_g53u_off) fprintf(stderr, "[CUDA] glm53 chain: %s; the CPU runs the layers\n", why);
    g_g53u_off = 1;
    if (g_g53u) g_g53u->ok = 0;
}
static int g53u_setup(GModel *m) {
    if (g_g53u) return g_g53u->ok;
    const Cfg *c = &m->c;
    const int L = c->n_layers, D = c->hidden, H = c->hc_mult, HD = H * D, nm = (2 + H) * H, P = c->kda_proj;
    G53U *ch = (G53U *)calloc(1, sizeof *ch);
    if (!ch) return 0;
    g_g53u = ch;
    const char *why = NULL;
    if (!cc_init(g_cud.dev) || !cc_ready()) why = "the chain's context did not come up";
    size_t **offs[] = {&ch->o_in, &ch->o_post, &ch->o_hca, &ch->o_hcf, &ch->o_qn, &ch->o_kn, &ch->o_ikw, &ch->o_ikb,
                       &ch->o_ape, &ch->o_conv, &ch->o_kda};
    for (size_t k = 0; !why && k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = (size_t *)calloc((size_t)L, sizeof(size_t)))) why = "memory";
    CcBuf ***bufs[] = {&ch->kbt, &ch->kbv, &ch->lat, &ch->ik, &ch->ig, &ch->pk, &ch->win, &ch->st};
    for (size_t k = 0; !why && k < sizeof bufs / sizeof *bufs; k++) if (!(*bufs[k] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) why = "memory";
    if (!why && (!(ch->fna = (ColiCudaTensor **)calloc((size_t)L, sizeof *ch->fna)) || !(ch->fnf = (ColiCudaTensor **)calloc((size_t)L, sizeof *ch->fnf)) ||
                 !(ch->kv_valid = (int *)calloc((size_t)L, sizeof(int))) || !(ch->pool_valid = (int *)calloc((size_t)L, sizeof(int)))))
        why = "memory";
    /* the parameter arena */
    size_t n = 0;
    for (int i = 0; !why && i < L; i++) {
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D;
        ch->o_hca[i] = n; n += 3 + nm; ch->o_hcf[i] = n; n += 3 + nm;
        if (c->is_full[i]) {
            ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += c->kv_lora;
            ch->o_ikw[i] = n; n += c->index_hd; ch->o_ikb[i] = n; n += c->index_hd;
            ch->o_ape[i] = n; n += (size_t)c->index_kpool * c->index_hd;
        } else {
            ch->o_conv[i] = n; n += (size_t)3 * P * c->conv_k;
            ch->o_kda[i] = n; n += (size_t)c->kda_heads + P + c->kda_hd;
        }
    }
    float *a = why ? NULL : (float *)calloc(n ? n : 1, sizeof(float));
    if (!why && !a) why = "memory";
    for (int i = 0; !why && i < L; i++) {
        const GLayer *l = &m->layer[i];
        memcpy(a + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(a + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(a + ch->o_hca[i], l->hc_attn_scale, 3 * sizeof(float));
        memcpy(a + ch->o_hca[i] + 3, l->hc_attn_base, nm * sizeof(float));
        memcpy(a + ch->o_hcf[i], l->hc_ffn_scale, 3 * sizeof(float));
        memcpy(a + ch->o_hcf[i] + 3, l->hc_ffn_base, nm * sizeof(float));
        if (c->is_full[i]) {
            memcpy(a + ch->o_qn[i], l->qa_ln, c->q_lora * sizeof(float));
            memcpy(a + ch->o_kn[i], l->kva_ln, c->kv_lora * sizeof(float));
            memcpy(a + ch->o_ikw[i], l->ik_nw, c->index_hd * sizeof(float));
            if (l->ik_nb) memcpy(a + ch->o_ikb[i], l->ik_nb, c->index_hd * sizeof(float));
            memcpy(a + ch->o_ape[i], l->ikpa, (size_t)c->index_kpool * c->index_hd * sizeof(float));
        } else {
            memcpy(a + ch->o_conv[i], l->conv, (size_t)3 * P * c->conv_k * sizeof(float));
            memcpy(a + ch->o_kda[i], l->alog, c->kda_heads * sizeof(float));
            memcpy(a + ch->o_kda[i] + c->kda_heads, l->dt, P * sizeof(float));
            memcpy(a + ch->o_kda[i] + c->kda_heads + P, l->onorm, c->kda_hd * sizeof(float));
        }
    }
    if (!why && !(ch->prm = cc_buf(n * sizeof(float), CC_DEV))) why = "device memory for the parameters";
    if (!why && !(cc_begin() && cc_write(ch->prm, 0, a, n * sizeof(float)) && cc_submit(1))) why = "the parameters' upload";
    free(a);
    /* every layer's mixes, matrices, per-head floats and KDA state */
    for (int i = 0; !why && i < L; i++) {
        const GLayer *l = &m->layer[i];
        if (!coli_cuda_tensor_upload(&ch->fna[i], l->hc_attn_fn, NULL, 0, HD, nm, g_cud.dev) ||
            !coli_cuda_tensor_upload(&ch->fnf[i], l->hc_ffn_fn, NULL, 0, HD, nm, g_cud.dev)) { why = "a mix matrix the device refused"; break; }
        Mat *w[16]; const int nw = g53u_layer_mats(m, i, w);
        for (int k = 0; k < nw && !why; k++) {
            if (w[k]->vk_hgemv) continue;   /* below, as floats */
            if (!g53u_tensor(w[k])) why = "a matrix the device refused";
        }
        if (!why && c->is_full[i]) {
            const Mat *hm[2] = {&l->kvb_kt, &l->kvb_v};
            CcBuf **hb[2] = {&ch->kbt[i], &ch->kbv[i]};
            for (int k = 0; k < 2 && !why; k++) {
                float *fl = g53u_floats(hm[k]);
                const size_t bytes = (size_t)hm[k]->rows * hm[k]->columns * sizeof(float);
                if (!fl) { why = "memory"; break; }
                if (!(*hb[k] = cc_buf(bytes, CC_DEV)) || !(cc_begin() && cc_write(*hb[k], 0, fl, bytes) && cc_submit(1)))
                    why = "an MLA matrix's floats the device refused";
                free(fl);
            }
        }
        if (!why && !c->is_full[i] && !((ch->win[i] = cc_buf((size_t)3 * P * c->conv_k * sizeof(float), CC_DEV)) &&
                                        (ch->st[i] = cc_buf((size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float), CC_DEV))))
            why = "device memory for a KDA state";
    }
    if (why) { g53u_decline(why); return 0; }
    ch->ok = 1;
    int full = 0, dense = 0;
    for (int i = 0; i < L; i++) { full += c->is_full[i]; dense += i < c->first_dense; }
    fprintf(stderr, "[CUDA] glm53 chain: %d layers on the card (%d KDA, %d MLA with the k-pooled indexer, %d dense), %d streams, "
                    "%.1f MiB of parameters (COLI_CUDA_CHAIN=0: the CPU)\n", L, L - full, full, dense, H, n * 4 / 1048576.0);
    return 1;
}

/* a chunk's scratch at `rows` rows and the indexer's scores at `ctx` positions */
static int g53u_scratch(G53U *ch, const GModel *m, int rows, int ctx) {
    const Cfg *c = &m->c;
    const int D = c->hidden, H = c->hc_mult, HD = H * D, nm = (2 + H) * H, P = c->kda_proj, NH = c->n_heads, K = c->kv_lora;
    const int IH = c->index_nh, ID = c->index_hd, MI = c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter;
    const int width = coli_sparse_index_width(c->index_topk, c->index_kpool, c->index_kpool_tail);
    const size_t r = (size_t)rows;
    ch->kvd_layer = r * (K + 2 * ID);
    int ok = g53u_res(&ch->xs, r * HD, CC_DEV) && g53u_res(&ch->xn, r * HD, CC_DEV) && g53u_res(&ch->col, r * D, CC_DEV) &&
             g53u_res(&ch->nrm, r * D, CC_DEV) && g53u_res(&ch->br, r * D, CC_DEV) && g53u_res(&ch->mix, r * nm, CC_DEV) &&
             g53u_res(&ch->hp, r * (2 * H + H * H), CC_DEV) && g53u_res(&ch->gs, r * MI, CC_DEV) &&
             g53u_res(&ch->us, r * MI, CC_DEV) && g53u_res(&ch->hs, r * MI, CC_DEV) && g53u_res(&ch->ds, r * D, CC_DEV) &&
             g53u_res(&ch->h2d, r * D, CC_DOWN) && g53u_res(&ch->kvd, (size_t)c->n_layers * ch->kvd_layer, CC_DOWN) &&
             g53u_res(&ch->xd, r * HD, CC_DOWN) && g53u_res(&ch->routed, r * D, CC_UP);
    if (ok && P > 0)
        ok = g53u_res(&ch->qkv3, 3 * r * P, CC_DEV) && g53u_res(&ch->cm, r * 3 * P, CC_DEV) && g53u_res(&ch->kf, r * P, CC_DEV) &&
             g53u_res(&ch->kb, r * c->kda_heads, CC_DEV) && g53u_res(&ch->kg, r * P, CC_DEV) &&
             g53u_res(&ch->klow, r * c->kda_hd, CC_DEV) && g53u_res(&ch->ky, r * P, CC_DEV);
    if (ok)
        ok = g53u_res(&ch->ikd, r * ID, CC_DEV) && g53u_res(&ch->iq, r * IH * ID, CC_DEV) && g53u_res(&ch->ihw, r * IH, CC_DEV) &&
             g53u_res(&ch->igd, r * ID, CC_DEV) && g53u_res(&ch->isc, r * (size_t)(ctx / c->index_kpool + 1), CC_DEV) &&
             g53u_res(&ch->sel, r * (size_t)(1 + width), CC_DEV) &&
             g53u_res(&ch->qa, r * c->q_lora, CC_DEV) && g53u_res(&ch->q, r * NH * c->qk_nope, CC_DEV) && g53u_res(&ch->kvs, r * K, CC_DEV) &&
             g53u_res(&ch->qabs, r * NH * K, CC_DEV) && g53u_res(&ch->clat, r * NH * K, CC_DEV) && g53u_res(&ch->ctx, r * NH * c->v_head, CC_DEV);
    if (!ok) return 0;
    if (ch->rows < rows) {
        float *hr = (float *)realloc(ch->host_routed, r * D * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    return 1;
}
/* the MLA mirror: room for `need` positions, the rows filled again when it grows */
static int g53u_mirror(G53U *ch, const GModel *m, int need, int limit) {
    const Cfg *c = &m->c; const int L = c->n_layers, ID = c->index_hd, pool = c->index_kpool;
    if (ch->cap >= need) return 1;
    int cap = 256; while (cap < need) cap *= 2;
    if (cap > limit) cap = limit;
    if (cap < need) return 0;
    for (int i = 0; i < L; i++) {
        if (!c->is_full[i]) continue;
        cc_free(ch->lat[i]); cc_free(ch->ik[i]); cc_free(ch->ig[i]); cc_free(ch->pk[i]);
        ch->lat[i] = ch->ik[i] = ch->ig[i] = ch->pk[i] = NULL;
        ch->kv_valid[i] = ch->pool_valid[i] = 0;
    }
    ch->cap = 0;
    for (int i = 0; i < L; i++) {
        if (!c->is_full[i]) continue;
        if (!(ch->lat[i] = cc_buf((size_t)cap * c->kv_lora * sizeof(float), CC_DEV)) ||
            !(ch->ik[i] = cc_buf((size_t)cap * ID * sizeof(float), CC_DEV)) || !(ch->ig[i] = cc_buf((size_t)cap * ID * sizeof(float), CC_DEV)) ||
            !(ch->pk[i] = cc_buf((size_t)(cap / pool + 1) * ID * sizeof(float), CC_DEV))) return 0;
    }
    ch->cap = cap;
    return 1;
}
/* the uploads that make the device the owner's below `start` (into the open frame) */
static int g53u_push(G53U *ch, const GModel *m, const GSession *s, int start) {
    const Cfg *c = &m->c; const int L = c->n_layers, K = c->kv_lora, ID = c->index_hd;
    int ok = 1;
    if (ch->where == G53U_HOST) {
        const size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)3 * c->kda_proj * c->conv_k;
        for (int i = 0; i < L && ok; i++)
            if (!c->is_full[i])
                ok = cc_write(ch->st[i], 0, s->layer[i].kda_state, ns * sizeof(float)) &&
                     cc_write(ch->win[i], 0, s->layer[i].kda_window, nw * sizeof(float));
        ch->where = G53U_BOTH;
    }
    for (int i = 0; i < L && ok; i++) {
        if (!c->is_full[i]) continue;
        const GLayerState *ls = &s->layer[i];
        const int t0 = ch->kv_valid[i], n = start - t0;
        if (ch->pool_valid[i] > start / c->index_kpool) ch->pool_valid[i] = start / c->index_kpool;   /* rows from start change */
        if (n <= 0) continue;
        ok = cc_write(ch->lat[i], (size_t)t0 * K, ls->latent + (size_t)t0 * K, (size_t)n * K * sizeof(float)) &&
             cc_write(ch->ik[i], (size_t)t0 * ID, ls->ikeys + (size_t)t0 * ID, (size_t)n * ID * sizeof(float)) &&
             cc_write(ch->ig[i], (size_t)t0 * ID, ls->igates + (size_t)t0 * ID, (size_t)n * ID * sizeof(float));
        ch->kv_valid[i] = start;
    }
    return ok;
}
static void g53u_pull(G53U *ch, const GModel *m, GSession *s, int from, int to, int pb, int n) {
    const Cfg *c = &m->c; const int K = c->kv_lora, ID = c->index_hd;
    for (int i = from; i < to; i++) {
        if (!c->is_full[i]) continue;
        const float *kv = (const float *)cc_ptr(ch->kvd) + (size_t)i * ch->kvd_layer;
        GLayerState *ls = &s->layer[i];
        memcpy(ls->latent + (size_t)pb * K, kv, (size_t)n * K * sizeof(float));
        memcpy(ls->ikeys + (size_t)pb * ID, kv + (size_t)n * K, (size_t)n * ID * sizeof(float));
        memcpy(ls->igates + (size_t)pb * ID, kv + (size_t)n * (K + ID), (size_t)n * ID * sizeof(float));
    }
}

/* ---- the KDA state between the host and the device --------------------------------------- */
static void g53u_rec_reset(G53U *ch, int base) { ch->rec_base = base; ch->rec_len = 0; ch->rec_ok = 1; }
/* the host's copy of s's KDA state made current (before anything reads it there) */
static void g53u_sync_host(const GModel *m, const GSession *s) {
    G53U *ch = g_g53u;
    if (!ch || !ch->ok || ch->owner != s || ch->where != G53U_DEV || !s) return;
    const Cfg *c = &m->c;
    const size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)3 * c->kda_proj * c->conv_k;
    int ok = cc_finish();
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (c->is_full[i]) continue;
        ok = cc_read(ch->st[i], 0, s->layer[i].kda_state, ns * sizeof(float)) && cc_read(ch->win[i], 0, s->layer[i].kda_window, nw * sizeof(float));
    }
    if (!ok) { fprintf(stderr, "[CUDA] glm53 chain: the device was lost holding the KDA state; it is rebuilt at the next step\n"); return; }
    ch->where = G53U_BOTH;
    g53u_rec_reset(ch, s->filled);
}
static void g53u_host_wrote(const GSession *s) { if (g_g53u && g_g53u->owner == s) g_g53u->where = G53U_HOST; }
static void g53u_session_gone(const GSession *s) { if (g_g53u && g_g53u->owner == s) { g_g53u->owner = NULL; g_g53u->where = G53U_HOST; } }
/* the CPU runs s's layers from `start`: the device's state stale after, the MLA rows from start on */
static void g53u_cpu_one(const GModel *m, G53U *ch, int start) {
    ch->where = G53U_HOST;
    for (int i = 0; i < m->c.n_layers; i++) {
        if (ch->kv_valid[i] > start) ch->kv_valid[i] = start;
        if (ch->pool_valid[i] > start / m->c.index_kpool) ch->pool_valid[i] = start / m->c.index_kpool;
    }
}
static void g53u_cpu_step(const GModel *m, const GSession *s, int start) {
    G53U *ch = g_g53u;
    if (!ch || !ch->ok || ch->owner != s) return;
    g53u_sync_host(m, s);
    g53u_cpu_one(m, ch, start);
}
/* the device lost: the host's state rebuilt on the CPU from the recorded input rows when the device
 * held the newest, the chain off from here on */
static void g53u_recover(GModel *m, GSession *s, int start, int dev) {
    G53U *ch = g_g53u;
    const int had = ch->owner == s && dev;
    ch->ok = 0; ch->failed = 1; ch->owner = NULL; ch->where = G53U_HOST;
    g_g53u_off = 1;
    if (!had) { fprintf(stderr, "[CUDA] glm53 chain: the device was lost; the host's state is current, the CPU runs from here on\n"); return; }
    const Cfg *c = &m->c; const int H = c->hc_mult, D = c->hidden;
    if (!ch->rec_ok || ch->rec_base + ch->rec_len < start) {
        fprintf(stderr, "[CUDA] glm53 chain: the device was lost with a KDA state its recorded rows do not describe -- stopping "
                        "(COLI_CUDA_CHAIN=0 keeps the state on the CPU)\n");
        exit(1);
    }
    const int n = start - ch->rec_base;
    fprintf(stderr, "[CUDA] glm53 chain: the device was lost; rebuilding the state of %d positions on the CPU, which runs from here on\n", n);
    if (n <= 0) return;
    float *xs = (float *)malloc((size_t)n * H * D * sizeof(float)), *xn = (float *)malloc((size_t)n * H * D * sizeof(float));
    if (!xs || !xn) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    for (int t = 0; t < n; t++) for (int h = 0; h < H; h++) memcpy(xs + ((size_t)t * H + h) * D, ch->rec + (size_t)t * D, (size_t)D * sizeof(float));
    run_layers(m, s, xs, xn, n, ch->rec_base, 0, c->n_layers);
    free(xs); free(xn);
}

/* ---- one layer's pieces ---------------------------------------------------------------- */
static int g53u_pre(G53U *ch, const GModel *m, ColiCudaTensor *fn, size_t hco, size_t lno, int n) {
    const Cfg *c = &m->c; const int H = c->hc_mult, D = c->hidden, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H;
    CcMhc sp = {n, H, D, c->hc_iters, 0, HD, 0, nm, 0, hr, 0, D, (int)hco, 0, c->eps, c->hc_eps, 0.f};
    CcNorm nr = {n, D, 1, 0, D, D, 0, D, D, (int)lno, 0, 0, c->eps, 1.f};
    return cc_matmul(fn, ch->xs, 0, ch->mix, 0, n) && cc_mhc(CC_MHC_SPLIT, ch->xs, ch->mix, ch->hp, ch->prm, NULL, &sp) &&
           cc_mhc(CC_MHC_COLLAPSE, ch->xs, NULL, ch->hp, NULL, ch->col, &sp) && cc_norm(ch->col, ch->prm, ch->nrm, &nr);
}
static int g53u_post(G53U *ch, const GModel *m, int n) {   /* xn = mix(xs) + post * br, then the two swap */
    const Cfg *c = &m->c; const int H = c->hc_mult, D = c->hidden, HD = H * D, hr = 2 * H + H * H;
    CcMhc po = {n, H, D, c->hc_iters, 0, HD, 0, D, 0, hr, 0, HD, 0, 0, c->eps, c->hc_eps, 0.f};
    if (!cc_mhc(CC_MHC_POST, ch->xs, ch->br, ch->hp, NULL, ch->xn, &po)) return 0;
    CcBuf *t = ch->xs; ch->xs = ch->xn; ch->xn = t;
    return 1;
}
static int g53u_mlp(G53U *ch, const GModel *m, const Mat *g, const Mat *u, const Mat *d, CcBuf *out, int n) {
    const int I = g->rows;
    CcMhc sw = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n * I, 0.f, 0.f, m->c.swiglu_limit};
    return cc_matmul(g53u_tensor(g), ch->nrm, 0, ch->gs, 0, n) && cc_matmul(g53u_tensor(u), ch->nrm, 0, ch->us, 0, n) &&
           cc_mhc(CC_SWIGLU_CLAMP, ch->gs, ch->us, NULL, NULL, ch->hs, &sw) && cc_matmul(g53u_tensor(d), ch->hs, 0, out, 0, n);
}
static int g53u_kda(G53U *ch, const GModel *m, const GLayer *l, int i, int n) {
    const Cfg *c = &m->c; const int P = c->kda_proj, HK = c->kda_heads, KD = c->kda_hd, r = ch->rows;
    CcKdaConv cp = {n, 3 * P, c->conv_k, P, 0, P, r * P, 0, 3 * P, (int)ch->o_conv[i], 0};
    CcKdaRec rp = {n, HK, KD, P, 0, 3 * P, 0, P, 0, HK, 0, P, 0, P, 0, (int)ch->o_kda[i], c->gate_lb, 1e-6f, c->eps};
    return cc_matmul(g53u_tensor(&l->kq), ch->nrm, 0, ch->qkv3, 0, n) &&
           cc_matmul(g53u_tensor(&l->kk), ch->nrm, 0, ch->qkv3, (size_t)r * P, n) &&
           cc_matmul(g53u_tensor(&l->kv), ch->nrm, 0, ch->qkv3, (size_t)2 * r * P, n) &&
           cc_matmul(g53u_tensor(&l->kfa), ch->nrm, 0, ch->klow, 0, n) &&
           cc_matmul(g53u_tensor(&l->kfb), ch->klow, 0, ch->kf, 0, n) &&
           cc_matmul(g53u_tensor(&l->kb), ch->nrm, 0, ch->kb, 0, n) &&
           cc_matmul(g53u_tensor(&l->kga), ch->nrm, 0, ch->klow, 0, n) &&
           cc_matmul(g53u_tensor(&l->kgb), ch->klow, 0, ch->kg, 0, n) &&
           cc_kda_conv(ch->qkv3, ch->prm, ch->win[i], ch->cm, &cp) &&
           cc_kda_rec(KD, ch->cm, ch->kf, ch->kb, ch->kg, ch->prm, ch->st[i], ch->ky, &rp, 0) &&
           cc_matmul(g53u_tensor(&l->ko), ch->ky, 0, ch->br, 0, n);
}
/* the MLA layer (vk_chain.c's vkc_mla_qkv and vkc_mla_attn with no rotated part, k_b and v_b split)
 * with the k-pooled indexer, glm53_chain.h's g53c_mla */
static int g53u_mla(G53U *ch, const GModel *m, const GLayer *l, int i, int n, int pb) {
    const Cfg *c = &m->c;
    const int NH = c->n_heads, Q = c->qk_nope, V = c->v_head, K = c->kv_lora, QL = c->q_lora, HK = NH * K, HV = NH * V;
    const int IH = c->index_nh, ID = c->index_hd, pool = c->index_kpool;
    const int width = coli_sparse_index_width(c->index_topk, pool, c->index_kpool_tail);
    const size_t ko = (size_t)i * ch->kvd_layer;
    CcNorm qn = {n, QL, 1, 0, QL, QL, 0, QL, QL, (int)ch->o_qn[i], 0, 0, c->eps, 1.f};
    CcNorm kn = {n, K, 1, 0, K, K, pb * K, K, K, (int)ch->o_kn[i], 0, 0, c->eps, 1.f};
    CcMlaRow ln = {n, 1, 0, ID, 0, 0, ID, 0, pb * ID, ID, 0, 0, 0, (int)ch->o_ikw[i], (int)ch->o_ikb[i], l->ik_nb != NULL, 1e-5f};
    const int done = (pb + n) / pool, p0 = ch->pool_valid[i];
    CcDsaPool kp = {done > p0 ? done - p0 : 0, pool, p0, ID, 0, ID, (int)ch->o_ape[i], 0, 0, ID};
    CcDsaPick sp = {n, pb, IH, ID, c->index_topk, pool, 0, IH * ID, 0, IH, 0, c->index_kpool_tail, (pb + n) / pool + 1,
                    1 + width, sqrtf((float)IH), 1.f / sqrtf((float)ID)};
    CcHgemv qa = {0, n, NH, K, K, 0, 0, NH * Q, Q, 0, HK, K, 0, 0, 0, Q, 0};
    CcMlaCore cp = {n, NH, K, 0, pb, 0, 0, HK, K, 0, 0, 0, 0, K, 0, 0, 0, 1 + width, 0, HK, K, 1.0f / sqrtf((float)Q)};
    CcHgemv va = {0, n, NH, V, V, 0, 0, HK, K, 0, HV, V, 0, HV, 0, K, 0};
    const int ok =
        cc_matmul(g53u_tensor(&l->qa), ch->nrm, 0, ch->qa, 0, n) && cc_norm(ch->qa, ch->prm, ch->qa, &qn) &&
        cc_matmul(g53u_tensor(&l->qb), ch->qa, 0, ch->q, 0, n) &&
        cc_matmul(g53u_tensor(&l->kva), ch->nrm, 0, ch->kvs, 0, n) && cc_norm(ch->kvs, ch->prm, ch->lat[i], &kn) &&
        cc_copy(ch->kvd, ko, ch->lat[i], (size_t)pb * K, (size_t)n * K) &&
        cc_matmul(g53u_tensor(&l->iwq), ch->qa, 0, ch->iq, 0, n) &&
        cc_matmul(g53u_tensor(&l->iwk), ch->nrm, 0, ch->ikd, 0, n) &&
        cc_mla_lnorm(ch->ikd, ch->prm, ch->ik[i], &ln) &&
        cc_matmul(g53u_tensor(&l->ikpg), ch->nrm, 0, ch->igd, 0, n) &&
        cc_copy(ch->ig[i], (size_t)pb * ID, ch->igd, 0, (size_t)n * ID) &&
        cc_matmul(g53u_tensor(&l->iwp), ch->nrm, 0, ch->ihw, 0, n) &&
        cc_copy(ch->kvd, ko + (size_t)n * K, ch->ik[i], (size_t)pb * ID, (size_t)n * ID) &&
        cc_copy(ch->kvd, ko + (size_t)n * (K + ID), ch->igd, 0, (size_t)n * ID) &&
        cc_dsa_pool_keys(ch->ik[i], ch->ig[i], ch->prm, ch->pk[i], &kp) &&
        cc_dsa_pool_select(ch->iq, ch->ihw, ch->pk[i], ch->isc, ch->sel, &sp) &&
        cc_mla_hgemv(ch->kbt[i], ch->q, ch->qabs, NULL, &qa) &&
        cc_mla_core(ch->qabs, NULL, ch->lat[i], NULL, ch->sel, ch->clat, &cp) &&
        cc_mla_hgemv(ch->kbv[i], ch->clat, ch->ctx, NULL, &va) &&
        cc_matmul(g53u_tensor(&l->o), ch->ctx, 0, ch->br, 0, n);
    if (ok && done > ch->pool_valid[i]) ch->pool_valid[i] = done;
    return ok;
}

/* ---- the forward ------------------------------------------------------------------------- */
/* every layer for n rows of `streams` (the session's positions start..), the streams after the
 * last layer back into it: L; 0 = not taken (memory refused, the streams untouched), or *lost = 1
 * (a frame failed); -1 = g_layer_halt stopped it between two layers */
static int g53u_forward_seg(GModel *m, GSession *s, G53U *ch, float *streams, int n, int start, int *lost) {
    *lost = 0;
    const Cfg *c = &m->c; const int D = c->hidden, H = c->hc_mult, HD = H * D, L = c->n_layers;
    const int CH = g53u_rows_env(), rows = n < CH ? n : CH;
    if (!g53u_mirror(ch, m, start + n, s->cap) || !g53u_scratch(ch, m, rows, start + n)) {
        fprintf(stderr, "[CUDA] glm53 chain: device memory for %d rows at %d positions refused\n", rows, start + n);
        return 0;
    }
    float *outs = (float *)malloc((size_t)n * HD * sizeof(float));
    if (!outs) return 0;
    for (int c0 = 0; c0 < n; c0 += rows) {
        const int nr = n - c0 < rows ? n - c0 : rows, pb = start + c0;
        if (!cc_begin() || !g53u_push(ch, m, s, pb) || !cc_write(ch->xs, 0, streams + (size_t)c0 * HD, (size_t)nr * HD * sizeof(float)))
            goto lost;
        int ok = 1, pending = 0, pulled = 0;
        for (int i = 0; i < L && ok; i++) {
            if (g_layer_halt && (i > 0 || c0 > 0) && g_layer_halt(g_layer_halt_arg)) {   /* between two layers, as run_layers */
                if (!cc_finish()) goto lost;
                g_halted_at_layer = i;
                g_halted_on_device = 1;
                free(outs);
                return -1;
            }
            const GLayer *l = &m->layer[i];
            if (pending) {   /* the FFN branch of the layer before: shared + routed, written back */
                CcEw add = {CC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = cc_ew(ch->br, ch->ds, ch->routed, NULL, NULL, &add) && g53u_post(ch, m, nr);
                pending = 0;
            }
            const double ta = now_s();
            ok = ok && g53u_pre(ch, m, ch->fna[i], ch->o_hca[i], ch->o_in[i], nr) &&
                 (c->is_full[i] ? g53u_mla(ch, m, l, i, nr, pb) : g53u_kda(ch, m, l, i, nr)) &&
                 g53u_post(ch, m, nr) && g53u_pre(ch, m, ch->fnf[i], ch->o_hcf[i], ch->o_post[i], nr);
            if (!ok) break;
            if (i < c->first_dense) {
                ok = g53u_mlp(ch, m, &l->dg, &l->du, &l->dd, ch->br, nr) && g53u_post(ch, m, nr);
                m->t_attn += now_s() - ta;
                continue;
            }
            /* A1; while it runs, the tier loads the experts this layer will likely stream (a big
             * prompt chunk only) */
            ok = cc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D) && cc_submit(0);
            if (ok) XT(stream_prefetch)(i, nr);
            ok = ok && cc_finish();
            m->t_attn += now_s() - ta;
            if (!ok) break;
            g53u_pull(ch, m, s, pulled, i + 1, pb, nr); pulled = i + 1;
            /* A2: the shared expert, while the host computes the routed experts */
            ok = cc_begin() && g53u_mlp(ch, m, &l->rg, &l->ru, &l->rd, ch->ds, nr) && cc_submit(0);
            const double t1 = now_s();
            ffn_layer_ex(m, l, i, (const float *)cc_ptr(ch->h2d), nr, ch->host_routed, 0);
            memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)nr * D * sizeof(float));
            const double dt = now_s() - t1; ch->host_ms += dt * 1e3; m->t_ffn += dt;
            ok = ok && cc_begin();
            pending = 1;
        }
        if (ok && pending) {
            CcEw add = {CC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = cc_ew(ch->br, ch->ds, ch->routed, NULL, NULL, &add) && g53u_post(ch, m, nr);
        }
        ok = ok && cc_copy(ch->xd, 0, ch->xs, 0, (size_t)nr * HD) && cc_submit(1);
        if (!ok) goto lost;
        g53u_pull(ch, m, s, pulled, L, pb, nr);
        memcpy(outs + (size_t)c0 * HD, cc_ptr(ch->xd), (size_t)nr * HD * sizeof(float));
        for (int i = 0; i < L; i++) if (c->is_full[i]) ch->kv_valid[i] = pb + nr;
        ch->where = G53U_DEV;
    }
    memcpy(streams, outs, (size_t)n * HD * sizeof(float));
    free(outs);
    ch->forwards++;
    return L;
lost:
    free(outs);
    cc_finish();
    *lost = 1;
    return 0;
}
/* the chain's forward for n rows of `streams` (positions start..): L, 0 = not taken (the CPU runs
 * the layers, the streams as they came in), -1 = stopped between layers */
static int g53u_forward(GModel *m, GSession *s, float *streams, int n, int start) {
    if (!g53u_wanted(m) || !g53u_setup(m)) return 0;
    G53U *ch = g_g53u;
    if (!ch->ok || ch->failed) return 0;
    const Cfg *c = &m->c; const int L = c->n_layers, D = c->hidden, H = c->hc_mult, HD = H * D;
    if (ch->owner != s) {   /* another session takes the device */
        if (ch->owner) g53u_sync_host(m, ch->owner);
        ch->owner = s; ch->where = G53U_HOST;
        for (int i = 0; i < L; i++) ch->kv_valid[i] = ch->pool_valid[i] = 0;
    }
    const int dev = ch->where == G53U_DEV;
    if (!cc_ready()) { g53u_recover(m, s, start, dev); return 0; }
    /* the input rows, for a rebuild should the device go: from where the host's state is current */
    if (ch->where == G53U_HOST) g53u_rec_reset(ch, start);
    else if (ch->rec_ok && start != ch->rec_base + ch->rec_len) {
        if (start < ch->rec_base + ch->rec_len && start >= ch->rec_base) ch->rec_len = start - ch->rec_base;
        else ch->rec_ok = 0;
    }
    if (ch->rec_ok) {
        if (ch->rec_len + n > ch->rec_cap) {
            int nc = ch->rec_cap ? ch->rec_cap : 256; while (nc < ch->rec_len + n) nc *= 2;
            float *r = (float *)realloc(ch->rec, (size_t)nc * D * sizeof(float));
            if (!r) ch->rec_ok = 0; else { ch->rec = r; ch->rec_cap = nc; }
        }
        if (ch->rec_ok) {
            for (int t = 0; t < n; t++) memcpy(ch->rec + (size_t)(ch->rec_len + t) * D, streams + (size_t)t * HD, (size_t)D * sizeof(float));
            ch->rec_len += n;
        }
    }
    int lost = 0;
    const int done = g53u_forward_seg(m, s, ch, streams, n, start, &lost);
    if (done < 0) { g53u_cpu_one(m, ch, start); return -1; }   /* the chunk is the caller's to throw away */
    if (!done) {
        if (lost) { g53u_recover(m, s, start, dev); return 0; }
        g53u_cpu_step(m, s, start);   /* memory refused: the CPU from here on, the state current first */
        g53u_decline("device memory refused");
        return 0;
    }
    return done;
}
/* a served prompt's chunk with the chain on: its rows */
static int g53u_prefill_rows(GModel *m) { return g53u_wanted(m) && !g_g53u_off ? g53u_rows_env() : 0; }
static void g53u_report(void) {
    G53U *ch = g_g53u;
    if (!ch || !ch->forwards) return;
    fprintf(stderr, "[CUDA] glm53 chain: %llu forwards, %.1f ms of routed experts on the host\n", ch->forwards, ch->host_ms);
}
#endif
