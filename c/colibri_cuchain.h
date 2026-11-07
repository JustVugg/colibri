/* colibri_cuchain.h -- GLM-5.2 and GLM-5.3's layers on the CUDA chain (cuda_chain.h), included once
 * by colibri.c in a COLI_CUDA build after the CPU layer forward it stands in for, when a GGUF runs
 * on the CUDA device (colibri_cu_tier_start: the routed experts on the CUDA tier, the dense
 * matrices on cu_dense.h's copies). The Vulkan chain's layout (glm_chain.h), op for op on the CUDA
 * chain's ops and GLM-5.3 Flash's MLA ops (cuda_glm53c.cuh), for a whole forward and a decode step:
 *   device, frame A1: the routed MoE output of the layer before joins the residual (x += routed +
 *     shared, the CPU's order), the input RMSNorm, the MLA attention (q_a, its norm, q_b, RoPE on
 *     each head's rotated dims; kv_a, the latent norm into the latent cache, RoPE on the rotated
 *     key into the rope cache; on a full DSA layer the index key (wk, LayerNorm, RoPE) into its
 *     cache and, once a row's context passes index_topk, every row's top-k, reused by the shared
 *     layers after it; the absorbed query from k_b, the core over the cache or the selection, the
 *     values from v_b, o_proj), the residual add, the post-attention RMSNorm. A dense layer (the
 *     first first_k_dense_replace) runs its MLP here too and has no host step. Then the frame is
 *     waited for.
 *   host: the router and the routed experts (moe(): the tier's batch and the CPU's share), without
 *     the shared expert; the new KV rows copied into the host's cache.
 *   device, frame A2 (not waited for): the shared expert, while the host computes the routed experts.
 * The final rows come back to the host, which runs the final norm and lm_head as before.
 *
 * State: the host's KV cache (latent, rope key, index keys) stays canonical (every step copies its
 * new rows back); the device mirrors the bound state's per layer behind a watermark, kv_valid, that
 * every host write lowers (the CPU's attention, kv_alloc, a slot adopting rows, a parked prefix
 * restored, another state bound). MLA has no recurrent state: a device lost mid-forward leaves
 * nothing to rebuild, the forward runs again on the CPU from its input rows and the CPU runs from
 * there.
 *
 * Every layer or none: the chain declines (the CPU's layers, the watermark follows) for a layer
 * range, a KV_SLOTS batch of several states, a coded KV cache (KV8, KV_TQ, KV16), PILOT and LOOKA,
 * Mistral-Large-4, a geometry its ops do not take, a matrix the device has no form for or that the
 * dense budget does not hold, and memory the device refuses. COLI_CUDA_CHAIN=0: the CPU's layers. */
#ifndef COLIBRI_CUCHAIN_H
#define COLIBRI_CUCHAIN_H
#include "cuda_chain.h"

#define GLMU_ROWS 256   /* a prompt chunk's rows through the chain (COLI_CUDA_GLM_ROWS) */

typedef struct {
    int ok, failed, rows, cap;
    KVState *owner;                           /* the state the mirror holds */
    CcBuf *prm;
    size_t *o_in, *o_post, *o_qn, *o_kn, *o_ixw, *o_ixb;
    CcBuf **kbt, **kbv;                       /* each layer's k_b ([H*K x Q]) and v_b ([H*V x K]) as floats */
    int *trans;                               /* kv_b as one f32 matrix [H*(Q+V) x K]: k read transposed */
    CcBuf **lat, **rope, **ik;
    int *kv_valid;
    CcBuf *x, *nrm, *tmp, *h2, *gs, *us, *hs, *ds, *qa, *q, *kvs, *qabs, *clat, *ctx;
    CcBuf *ikd, *iq, *ihw, *isc, *sel, *cs, *h2d, *kvd, *xd, *routed;
    size_t kvd_layer;
    float *host_routed;
    unsigned long long forwards;
    double host_ms;
} GlmU;

static GlmU *g_glmu;
static int g_glmu_off;   /* declined for good */

static int glmu_rows_env(void) {
    static int r = -1;
    if (r < 0) { const char *e = getenv("COLI_CUDA_GLM_ROWS"); r = e && atoi(e) > 0 ? atoi(e) : GLMU_ROWS; }
    return r;
}
static int glmu_res(CcBuf **b, size_t floats, int kind) { return cc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }
static int glmu_full(const Model *m, int i) { return m->has_dsa && m->c.idx_type[i]; }

/* ---- the matrices ------------------------------------------------------------------------ */
/* t's CUDA format (cu_dense.h's: 100 + the file's ggml type, the rows a matrix the CPU holds 16
 * interleaved going up as the file's; a float type goes up as f32 rows); -1 none */
static int glmu_form(const QT *t) {
    if (t->O < 1 || t->I < 1) return -1;
    /* a GGUF's F32 tensor (the indexer's head weights) as the load leaves it: f32, or quantized to the
     * dense bits (int8 rows, int4 groups) */
    if (t->fmt == 0) return t->qf ? 0 : -1;
    if (t->fmt == 1) return t->q8 && t->s ? 1 : -1;
    if (t->fmt == 4) return t->q4 && t->s && !t->planar && t->gs > 0 && !(t->gs & 1) ? 4 : -1;
    if (t->fmt != QT_GGML || !t->q4) return -1;
    const int b = gq_r16_base(t->gt), f = 100 + (b >= 0 ? b : t->gt);
    if (b >= 0 && t->O % 16) return -1;
    if (cud_float(f)) return f;
    return cud_bytes(f, t->O, t->I, 0) ? f : -1;
}
static size_t glmu_bytes(const QT *t) {
    const int f = glmu_form(t);
    return f < 0 ? 0 : f == 0 || cud_float(f) ? (size_t)t->O * t->I * 4 : cud_bytes(f, t->O, t->I, f == 4 ? t->gs : 0);
}
/* t's rows as floats, the file's values exactly; NULL: no memory, or no rows to read */
static float *glmu_floats(const QT *t) {
    if (t->fmt == 0) {
        if (!t->qf) return NULL;
        float *f = (float *)malloc((size_t)t->O * t->I * sizeof(float));
        if (f) memcpy(f, t->qf, (size_t)t->O * t->I * sizeof(float));
        return f;
    }
    if (t->fmt != QT_GGML || !t->q4) return NULL;
    const int b = gq_r16_base(t->gt), ty = b >= 0 ? b : t->gt;
    const int64_t rb = gq_row_bytes(ty, t->I);
    if (rb <= 0 || (b >= 0 && t->O % 16)) return NULL;
    uint8_t *rows = NULL;
    if (b >= 0) {
        if (!(rows = (uint8_t *)malloc((size_t)rb * t->O))) return NULL;
        gq_r16_unpack(t->gt, rows, t->q4, t->O, t->I);
    }
    const uint8_t *src = rows ? rows : t->q4;
    float *f = (float *)malloc((size_t)t->O * t->I * sizeof(float));
    int ok = f != NULL;
    for (int r = 0; ok && r < t->O; r++) ok = !gq_dequant_row(ty, src + (int64_t)r * rb, f + (size_t)r * t->I, t->I);
    free(rows);
    if (!ok) { free(f); return NULL; }
    return f;
}
/* t's device copy (cud_mm's, t->cuda: the CPU path's calls reuse it), uploaded now within the dense
 * budget if it is not there; a matrix of colibri's own formats (f32, int8 rows, int4 groups: outside
 * the budget, which counts a GGUF's blocks; glmu_plan_bytes counts it) goes up as it is */
static ColiCudaTensor *glmu_tensor(QT *t) {
    if (t->cuda) return t->cuda;
    if (t->cud_no || !g_cud.on) return NULL;
    const int f = glmu_form(t);
    const size_t bytes = glmu_bytes(t);
    if (f < 0 || !bytes || (f >= 100 && g_cud.used + bytes > g_cud.budget)) return NULL;
    ColiCudaTensor *ct = NULL;
    int up = 0;
    if (f < 100) {
        if (f == 0) up = coli_cuda_tensor_upload(&ct, t->qf, NULL, 0, t->I, t->O, g_cud.dev);
        else if (f == 1) up = coli_cuda_tensor_upload(&ct, t->q8, t->s, 1, t->I, t->O, g_cud.dev);
        else up = coli_cuda_tensor_upload_g(&ct, t->q4, t->s, 4, t->I, t->O, g_cud.dev, t->gs);
        if (!up || !ct) { if (ct) coli_cuda_tensor_free(ct); return NULL; }
        t->cuda = ct;
        return ct;
    }
    if (cud_float(f)) {
        float *fl = glmu_floats(t);
        up = fl && coli_cuda_tensor_upload(&ct, fl, NULL, 0, t->I, t->O, g_cud.dev);
        free(fl);
    } else {
        const int b = gq_r16_base(t->gt);
        uint8_t *rows = NULL;
        if (b >= 0) {
            if (!(rows = (uint8_t *)malloc((size_t)gq_row_bytes(b, t->I) * t->O))) return NULL;
            gq_r16_unpack(t->gt, rows, t->q4, t->O, t->I);
        }
        up = coli_cuda_tensor_upload(&ct, rows ? (const void *)rows : (const void *)t->q4, NULL, f, t->I, t->O, g_cud.dev);
        free(rows);
    }
    if (!up || !ct) { if (ct) coli_cuda_tensor_free(ct); return NULL; }
    t->cuda = ct;
    g_cud.used += bytes;
    g_cud.mats++;
    return ct;
}
/* layer i's matrices the chain uploads (kv_b's halves go as floats, below) */
static int glmu_layer_qts(Model *m, int i, QT **q) {
    Layer *l = &m->L[i]; int n = 0;
    q[n++] = &l->q_a; q[n++] = &l->q_b; q[n++] = &l->kv_a; q[n++] = &l->o;
    if (l->sparse) { q[n++] = &l->sh_gate; q[n++] = &l->sh_up; q[n++] = &l->sh_down; }
    else { q[n++] = &l->gate_proj; q[n++] = &l->up_proj; q[n++] = &l->down_proj; }
    if (glmu_full(m, i)) { q[n++] = &m->ix_wq[i]; q[n++] = &m->ix_wk[i]; q[n++] = &m->ix_wp[i]; }
    return n;
}
/* kv_b as the chain reads it: a GGUF's k_b and v_b (QT_KVB), or one f32 matrix */
static int glmu_kvb_ok(const Model *m, const Layer *l) {
    const Cfg *c = &m->c;
    if (l->kv_b.fmt == QT_KVB) {
        const QtKvb *k = l->kv_b.kvb;
        return k && k->H == c->n_heads && k->nope == c->qk_nope && k->V == c->v_head && k->L == c->kv_lora &&
               (k->kb.fmt == QT_GGML || k->kb.fmt == 0) && (k->vb.fmt == QT_GGML || k->vb.fmt == 0);
    }
    return l->kv_b.fmt == 0 && l->kv_b.qf && l->kv_b.O == c->n_heads * (c->qk_nope + c->v_head) && l->kv_b.I == c->kv_lora;
}
/* what the chain does not take (NULL: nothing), from the shapes alone */
static const char *glmu_check(Model *m) {
    const Cfg *c = &m->c;
    if (c->ml4) return "Mistral-Large-4's layers";
    if (c->kv_lora < 1 || c->kv_lora > 1024 || c->qk_rope < 2 || c->qk_rope > 128 || (c->qk_rope & 1) || c->qk_nope > 1024 ||
        c->v_head > 1024 || c->q_lora < 1 || c->qk_head != c->qk_nope + c->qk_rope) return "an attention geometry its ops do not take";
    if (m->has_dsa && (c->index_nh > 64 || c->index_nh * c->index_hd > 4096 || c->index_hd < c->qk_rope || !c->idx_type[0] ||
                       c->index_topk < 1))
        return "an indexer its ops do not take";
    QT *q[16];
    static char why[160];
    for (int i = 0; i < c->n_layers; i++) {
        const int n = glmu_layer_qts(m, i, q);
        for (int k = 0; k < n; k++)
            if (glmu_form(q[k]) < 0) {
                snprintf(why, sizeof why, "layer %d's matrix %d (fmt %d, ggml type %d, %d x %d) has no CUDA form", i, k, q[k]->fmt,
                         q[k]->fmt == QT_GGML ? q[k]->gt : -1, q[k]->O, q[k]->I);
                return why;
            }
        if (!glmu_kvb_ok(m, &m->L[i])) return "a kv_b its ops do not take";
    }
    return NULL;
}
/* the chain's device bytes past the dense matrices' (cu_dense.h's budget holds those): kv_b's
 * halves as floats, the parameters, the KV mirror at ctx positions and a chunk's scratch; for the
 * tier's room (VktConfig.dense_bytes) */
static size_t glmu_plan_bytes(const Model *m, int ctx) {
    const Cfg *c = &m->c; const size_t L = (size_t)c->n_layers, D = (size_t)c->hidden, NH = (size_t)c->n_heads;
    const size_t K = (size_t)c->kv_lora, R = (size_t)c->qk_rope, Q = (size_t)c->qk_nope, V = (size_t)c->v_head;
    const size_t ID = (size_t)c->index_hd, IH = (size_t)c->index_nh, r = (size_t)glmu_rows_env();
    const size_t SI = (size_t)c->moe_inter * (c->n_shared > 0 ? c->n_shared : 1), MI = SI > (size_t)c->dense_inter ? SI : (size_t)c->dense_inter;
    size_t b = 0;
    for (size_t i = 0; i < L; i++) {
        b += NH * (Q + V) * K * 4 + (2 * D + c->q_lora + K) * 4 + (size_t)ctx * (K + R) * 4;
        if (glmu_full(m, (int)i)) b += 2 * ID * 4 + (size_t)ctx * ID * 4;
        QT *q[16]; const int n = glmu_layer_qts((Model *)m, (int)i, q);   /* the f32 matrices, outside the dense budget */
        for (int k = 0; k < n; k++) if (q[k]->fmt != QT_GGML) b += glmu_bytes(q[k]);
    }
    size_t row = 4 * D + 3 * MI + D + 3 * D;                                         /* x, nrm, tmp, h2; the MLP; ds; h2d, xd, routed */
    row += (size_t)c->q_lora + NH * (Q + R) + K + R + 2 * NH * K + NH * V + R;       /* the MLA's */
    if (m->has_dsa) row += ID + IH * ID + IH + (size_t)ctx + 1 + (size_t)c->index_topk;   /* the indexer's */
    b += r * row * 4 + L * r * (K + R + ID) * 4;
    return b + ((size_t)64 << 20);
}
/* the chain will run if the dense part reaches the card: COLI_CUDA_CHAIN, the backend's ops, the shapes
 * (said once); the tier's start asks it before the dense budget (colibri_cu_tier_start) */
static int glmu_planned(Model *m) {
    static int on = -1, said;
    if (on < 0) { const char *e = getenv("COLI_CUDA_CHAIN"); on = !(e && *e == '0'); }
    if (!on || g_glmu_off || !cc_available() || !cc_glmdsa_ready()) return 0;
    if (!said) {   /* the shapes, once */
        said = 1;
        const char *why = glmu_check(m);
        if (why) { fprintf(stderr, "[CUDA] colibri chain: %s; the CPU runs the attention\n", why); g_glmu_off = 1; return 0; }
    }
    return 1;
}
static int glmu_wanted(Model *m) { return xt_cuda() && g_cud.on && glmu_planned(m); }

/* ---- the host's writes to its KV cache: the mirror's watermark follows ------------------- */
static void glmu_cpu_rows(Model *m, int layer, KVState *const *kvs, const int *positions, int pos_base, int S) {
    GlmU *ch = g_glmu;
    if (!ch || !ch->kv_valid || layer < 0 || layer >= m->c.n_layers) return;
    for (int s = 0; s < S; s++) {
        KVState *ks = kvs ? kvs[s] : m->kv;
        const int pos = positions ? positions[s] : pos_base + s;
        if (ks == ch->owner && ch->kv_valid[layer] > pos) ch->kv_valid[layer] = pos < 0 ? 0 : pos;
    }
}
static void glmu_host_rows(Model *m, KVState *k, int from) {
    GlmU *ch = g_glmu;
    if (!ch || !ch->kv_valid || k != ch->owner) return;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid[i] > from) ch->kv_valid[i] = from < 0 ? 0 : from;
}
static void glmu_kv_reset(Model *m) {
    GlmU *ch = g_glmu;
    if (!ch || !ch->kv_valid) return;
    for (int i = 0; i < m->c.n_layers; i++) ch->kv_valid[i] = 0;
    ch->owner = NULL;
}

/* ---- setup ----------------------------------------------------------------------------- */
static void glmu_decline(const char *why) {
    if (!g_glmu_off) fprintf(stderr, "[CUDA] colibri chain: %s; the CPU runs the attention\n", why);
    g_glmu_off = 1;
    if (g_glmu) g_glmu->ok = 0;
}
static int glmu_setup(Model *m) {
    if (g_glmu) return g_glmu->ok;
    const Cfg *c = &m->c;
    const int L = c->n_layers, D = c->hidden;
    GlmU *ch = (GlmU *)calloc(1, sizeof *ch);
    if (!ch) return 0;
    g_glmu = ch;
    const char *why = NULL;
    if (!cc_init(g_cud.dev) || !cc_ready()) why = "the chain's context did not come up";
    size_t **offs[] = {&ch->o_in, &ch->o_post, &ch->o_qn, &ch->o_kn, &ch->o_ixw, &ch->o_ixb};
    for (size_t k = 0; !why && k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = (size_t *)calloc((size_t)L, sizeof(size_t)))) why = "memory";
    CcBuf ***bufs[] = {&ch->kbt, &ch->kbv, &ch->lat, &ch->rope, &ch->ik};
    for (size_t k = 0; !why && k < sizeof bufs / sizeof *bufs; k++) if (!(*bufs[k] = (CcBuf **)calloc((size_t)L, sizeof(CcBuf *)))) why = "memory";
    if (!why && (!(ch->kv_valid = (int *)calloc((size_t)L, sizeof(int))) || !(ch->trans = (int *)calloc((size_t)L, sizeof(int)))))
        why = "memory";
    /* the parameter arena: in_ln, post_ln, the q and kv latent norms, the index key's LayerNorm */
    size_t n = 0;
    for (int i = 0; !why && i < L; i++) {
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D;
        ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += c->kv_lora;
        if (glmu_full(m, i)) { ch->o_ixw[i] = n; n += c->index_hd; ch->o_ixb[i] = n; n += c->index_hd; }
    }
    float *a = why ? NULL : (float *)calloc(n ? n : 1, sizeof(float));
    if (!why && !a) why = "memory";
    for (int i = 0; !why && i < L; i++) {
        const Layer *l = &m->L[i];
        memcpy(a + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(a + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(a + ch->o_qn[i], l->q_a_ln, c->q_lora * sizeof(float));
        memcpy(a + ch->o_kn[i], l->kv_a_ln, c->kv_lora * sizeof(float));
        if (glmu_full(m, i)) {
            memcpy(a + ch->o_ixw[i], m->ix_knw[i], c->index_hd * sizeof(float));
            memcpy(a + ch->o_ixb[i], m->ix_knb[i], c->index_hd * sizeof(float));
        }
    }
    if (!why && !(ch->prm = cc_buf(n * sizeof(float), CC_DEV))) why = "device memory for the parameters";
    if (!why && !(cc_begin() && cc_write(ch->prm, 0, a, n * sizeof(float)) && cc_submit(1))) why = "the parameters' upload";
    free(a);
    /* every layer's matrices and kv_b's halves as floats */
    for (int i = 0; !why && i < L; i++) {
        Layer *l = &m->L[i];
        QT *q[16]; const int nq = glmu_layer_qts(m, i, q);
        for (int k = 0; k < nq && !why; k++) if (!glmu_tensor(q[k])) why = "a matrix the dense budget or the device refused";
        const QT *hm[2] = {NULL, NULL};
        if (l->kv_b.fmt == QT_KVB) { hm[0] = &l->kv_b.kvb->kb; hm[1] = &l->kv_b.kvb->vb; }
        else { hm[0] = &l->kv_b; ch->trans[i] = 1; }
        CcBuf **hb[2] = {&ch->kbt[i], &ch->kbv[i]};
        for (int k = 0; k < 2 && !why && hm[k]; k++) {
            float *fl = glmu_floats(hm[k]);
            const size_t bytes = (size_t)hm[k]->O * hm[k]->I * sizeof(float);
            if (!fl) { why = "memory"; break; }
            if (!(*hb[k] = cc_buf(bytes, CC_DEV)) || !(cc_begin() && cc_write(*hb[k], 0, fl, bytes) && cc_submit(1)))
                why = "an MLA matrix's floats the device refused";
            free(fl);
        }
    }
    if (why) { glmu_decline(why); return 0; }
    ch->ok = 1;
    int full = 0, dense = 0;
    for (int i = 0; i < L; i++) { full += glmu_full(m, i); dense += !m->L[i].sparse; }
    fprintf(stderr, "[CUDA] colibri chain: %d layers on the card (%d dense, %d with the DSA indexer), %.1f MiB of parameters "
                    "(COLI_CUDA_CHAIN=0: the CPU's attention)\n", L, dense, full, n * 4 / 1048576.0);
    return 1;
}

/* a chunk's scratch at `rows` rows and the indexer's scores at `ctx` positions */
static int glmu_scratch(GlmU *ch, const Model *m, int rows, int ctx) {
    const Cfg *c = &m->c;
    const int D = c->hidden, NH = c->n_heads, K = c->kv_lora, R = c->qk_rope, Q = c->qk_nope, V = c->v_head;
    const int IH = c->index_nh, ID = c->index_hd;
    const int SI = c->moe_inter * (c->n_shared > 0 ? c->n_shared : 1), MI = SI > c->dense_inter ? SI : c->dense_inter;
    const size_t r = (size_t)rows;
    ch->kvd_layer = r * (K + R + (m->has_dsa ? ID : 0));
    int ok = glmu_res(&ch->x, r * D, CC_DEV) && glmu_res(&ch->nrm, r * D, CC_DEV) && glmu_res(&ch->tmp, r * D, CC_DEV) &&
             glmu_res(&ch->h2, r * D, CC_DEV) && glmu_res(&ch->gs, r * MI, CC_DEV) && glmu_res(&ch->us, r * MI, CC_DEV) &&
             glmu_res(&ch->hs, r * MI, CC_DEV) && glmu_res(&ch->ds, r * D, CC_DEV) &&
             glmu_res(&ch->qa, r * c->q_lora, CC_DEV) && glmu_res(&ch->q, r * NH * (Q + R), CC_DEV) &&
             glmu_res(&ch->kvs, r * (K + R), CC_DEV) && glmu_res(&ch->qabs, r * NH * K, CC_DEV) &&
             glmu_res(&ch->clat, r * NH * K, CC_DEV) && glmu_res(&ch->ctx, r * NH * V, CC_DEV) &&
             glmu_res(&ch->cs, r * R, CC_UP) &&
             glmu_res(&ch->h2d, r * D, CC_DOWN) && glmu_res(&ch->kvd, (size_t)c->n_layers * ch->kvd_layer, CC_DOWN) &&
             glmu_res(&ch->xd, r * D, CC_DOWN) && glmu_res(&ch->routed, r * D, CC_UP);
    if (ok && m->has_dsa)
        ok = glmu_res(&ch->ikd, r * ID, CC_DEV) && glmu_res(&ch->iq, r * IH * ID, CC_DEV) && glmu_res(&ch->ihw, r * IH, CC_DEV) &&
             glmu_res(&ch->isc, r * (size_t)ctx, CC_DEV) && glmu_res(&ch->sel, r * (1 + (size_t)c->index_topk), CC_DEV);
    if (!ok) return 0;
    if (ch->rows < rows) {
        float *hr = (float *)realloc(ch->host_routed, r * D * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    return 1;
}
/* the KV mirror: room for `need` positions of the bound state, the rows filled again when it grows
 * or another state is bound */
static int glmu_mirror(GlmU *ch, Model *m, int need) {
    const Cfg *c = &m->c; const int L = c->n_layers, K = c->kv_lora, R = c->qk_rope, ID = c->index_hd;
    if (ch->owner != m->kv) { for (int i = 0; i < L; i++) ch->kv_valid[i] = 0; ch->owner = m->kv; }
    if (ch->cap >= need) return 1;
    int cap = 256; while (cap < need) cap *= 2;
    if (cap > m->max_t) cap = m->max_t;
    if (cap < need) return 0;
    for (int i = 0; i < L; i++) {
        cc_free(ch->lat[i]); cc_free(ch->rope[i]); cc_free(ch->ik[i]);
        ch->lat[i] = ch->rope[i] = ch->ik[i] = NULL;
        ch->kv_valid[i] = 0;
    }
    ch->cap = 0;
    for (int i = 0; i < L; i++) {
        if (!(ch->lat[i] = cc_buf((size_t)cap * K * sizeof(float), CC_DEV)) || !(ch->rope[i] = cc_buf((size_t)cap * R * sizeof(float), CC_DEV)) ||
            (glmu_full(m, i) && !(ch->ik[i] = cc_buf((size_t)cap * ID * sizeof(float), CC_DEV)))) return 0;
    }
    ch->cap = cap;
    return 1;
}
/* the uploads that make the mirror the host's below pos_base (into the open frame) */
static int glmu_push(GlmU *ch, Model *m, int pos_base) {
    const Cfg *c = &m->c; const int L = c->n_layers, K = c->kv_lora, R = c->qk_rope, ID = c->index_hd;
    int ok = 1;
    for (int i = 0; i < L && ok; i++) {
        const int t0 = ch->kv_valid[i], n = pos_base - t0;
        if (n <= 0) continue;
        ok = cc_write(ch->lat[i], (size_t)t0 * K, m->kv->Lc[i] + (size_t)t0 * K, (size_t)n * K * sizeof(float)) &&
             cc_write(ch->rope[i], (size_t)t0 * R, m->kv->Rc[i] + (size_t)t0 * R, (size_t)n * R * sizeof(float)) &&
             (!glmu_full(m, i) || cc_write(ch->ik[i], (size_t)t0 * ID, coli_kv_row(m->kv->Ic[i], t0, ID), (size_t)n * ID * sizeof(float)));
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}
/* the new rows of layers [from, to) into the host's cache (after their frame's wait) */
static void glmu_pull(GlmU *ch, Model *m, int from, int to, int pb, int n) {
    const Cfg *c = &m->c; const int K = c->kv_lora, R = c->qk_rope, ID = c->index_hd;
    for (int i = from; i < to; i++) {
        const float *kv = (const float *)cc_ptr(ch->kvd) + (size_t)i * ch->kvd_layer;
        memcpy(m->kv->Lc[i] + (size_t)pb * K, kv, (size_t)n * K * sizeof(float));
        memcpy(m->kv->Rc[i] + (size_t)pb * R, kv + (size_t)n * K, (size_t)n * R * sizeof(float));
        if (glmu_full(m, i)) memcpy(coli_kv_row(m->kv->Ic[i], pb, ID), kv + (size_t)n * (K + R), (size_t)n * ID * sizeof(float));
    }
}

/* ---- one layer's pieces ---------------------------------------------------------------- */
static int glmu_norm(CcBuf *x, CcBuf *prm, size_t wo, CcBuf *y, int rows, int D, float eps) {
    CcNorm p = {rows, D, 1, 0, D, D, 0, D, D, (int)wo, 0, 0, eps, 1.f};
    return cc_norm(x, prm, y, &p);
}
static int glmu_mlp(GlmU *ch, QT *g, QT *u, QT *d, CcBuf *in, CcBuf *out, int n) {
    const int I = g->O;
    CcEw p = {CC_EW_SWIGLU, n * I, I, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return cc_matmul(glmu_tensor(g), in, 0, ch->gs, 0, n) && cc_matmul(glmu_tensor(u), in, 0, ch->us, 0, n) &&
           cc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &p) && cc_matmul(glmu_tensor(d), ch->hs, 0, out, 0, n);
}
/* the MLA attention of layer i for n rows at positions pb.. (attention_rows' with the absorbed
 * core), its output into tmp; *sel_on: the selection list holds a full layer's choice */
static int glmu_attn(GlmU *ch, Model *m, int i, int n, int pb, int *sel_on) {
    const Cfg *c = &m->c; Layer *l = &m->L[i];
    const int NH = c->n_heads, Q = c->qk_nope, R = c->qk_rope, QR = Q + R, V = c->v_head, K = c->kv_lora, QL = c->q_lora;
    const int HK = NH * K, HV = NH * V, TK = c->index_topk;
    const size_t ko = (size_t)i * ch->kvd_layer;
    CcNorm qn = {n, QL, 1, 0, QL, QL, 0, QL, QL, (int)ch->o_qn[i], 0, 0, c->eps, 1.f};
    CcNorm kn = {n, K, 1, 0, K + R, K + R, pb * K, K, K, (int)ch->o_kn[i], 0, 0, c->eps, 1.f};
    CcMlaRow rq = {n * NH, NH, R, QR, 1, Q, NH * QR, QR, Q, NH * QR, QR, 0, R, 0, 0, 0, 0.f};
    CcMlaRow rk = {n, 1, R, R, 1, K, K + R, K + R, pb * R, R, R, 0, R, 0, 0, 0, 0.f};
    int ok = cc_matmul(glmu_tensor(&l->q_a), ch->nrm, 0, ch->qa, 0, n) && cc_norm(ch->qa, ch->prm, ch->qa, &qn) &&
             cc_matmul(glmu_tensor(&l->q_b), ch->qa, 0, ch->q, 0, n) && cc_mla_rope(ch->q, ch->cs, ch->q, &rq) &&
             cc_matmul(glmu_tensor(&l->kv_a), ch->nrm, 0, ch->kvs, 0, n) && cc_norm(ch->kvs, ch->prm, ch->lat[i], &kn) &&
             cc_mla_rope(ch->kvs, ch->cs, ch->rope[i], &rk) &&
             cc_copy(ch->kvd, ko, ch->lat[i], (size_t)pb * K, (size_t)n * K) &&
             cc_copy(ch->kvd, ko + (size_t)n * K, ch->rope[i], (size_t)pb * R, (size_t)n * R);
    if (ok && glmu_full(m, i)) {   /* the indexer: the keys into the cache, then (past index_topk) every row's selection */
        const int IH = c->index_nh, ID = c->index_hd;
        CcMlaRow ln = {n, 1, 0, ID, 0, 0, ID, 0, pb * ID, ID, 0, 0, 0, (int)ch->o_ixw[i], (int)ch->o_ixb[i], 1, 1e-6f};
        CcMlaRow ri = {n, 1, R, ID, 1, pb * ID, ID, 0, pb * ID, ID, 0, 0, R, 0, 0, 0, 0.f};
        ok = cc_matmul(glmu_tensor(&m->ix_wk[i]), ch->nrm, 0, ch->ikd, 0, n) && cc_mla_lnorm(ch->ikd, ch->prm, ch->ik[i], &ln) &&
             cc_mla_rope(ch->ik[i], ch->cs, ch->ik[i], &ri) &&
             cc_copy(ch->kvd, ko + (size_t)n * (K + R), ch->ik[i], (size_t)pb * ID, (size_t)n * ID);
        *sel_on = pb + n > TK || g_dsa_force;
        if (ok && *sel_on) {
            CcMlaRow rq2 = {n * IH, IH, R, ID, 1, 0, IH * ID, ID, 0, IH * ID, ID, 0, R, 0, 0, 0, 0.f};
            CcDsaSel ds = {n, pb, IH, ID, TK, g_dsa_force, 0, IH * ID, 0, IH, 0, pb + n, 1 + TK,
                           1.f / sqrtf((float)ID), 1.f / sqrtf((float)IH)};
            ok = cc_matmul(glmu_tensor(&m->ix_wq[i]), ch->qa, 0, ch->iq, 0, n) && cc_mla_rope(ch->iq, ch->cs, ch->iq, &rq2) &&
                 cc_matmul(glmu_tensor(&m->ix_wp[i]), ch->nrm, 0, ch->ihw, 0, n) &&
                 cc_dsa_select(ch->iq, ch->ihw, ch->ik[i], ch->isc, ch->sel, &ds);
        }
    }
    /* the absorbed query from k_b, the core, the values from v_b */
    CcHgemv qa = ch->trans[i] ? (CcHgemv){1, n, NH, Q, Q + V, 0, 0, NH * QR, QR, 0, HK, K, 0, 0, 0, K, 0}
                              : (CcHgemv){0, n, NH, K, K, 0, 0, NH * QR, QR, 0, HK, K, 0, 0, 0, Q, 0};
    CcHgemv va = ch->trans[i] ? (CcHgemv){0, n, NH, V, Q + V, Q, 0, HK, K, 0, HV, V, 0, HV, 0, K, 0}
                              : (CcHgemv){0, n, NH, V, V, 0, 0, HK, K, 0, HV, V, 0, HV, 0, K, 0};
    CcMlaCore cp = {n, NH, K, R, pb, 0, 0, HK, K, Q, NH * QR, QR, 0, K, 0, R, 0, *sel_on ? 1 + TK : 0, 0, HK, K, c->attn_scale};
    return ok && cc_mla_hgemv(ch->kbt[i], ch->q, ch->qabs, NULL, &qa) &&
           cc_mla_core(ch->qabs, ch->q, ch->lat[i], ch->rope[i], *sel_on ? ch->sel : NULL, ch->clat, &cp) &&
           cc_mla_hgemv(ch->trans[i] ? ch->kbt[i] : ch->kbv[i], ch->clat, ch->ctx, NULL, &va) &&
           cc_matmul(glmu_tensor(&l->o), ch->ctx, 0, ch->tmp, 0, n);
}

/* ---- the forward ------------------------------------------------------------------------- */
/* every layer for S rows of xh (positions pos_base..), the rows after the last layer back into xh:
 * L; 0 = not taken (xh as it came in), *lost = 1 when a frame failed */
static int glmu_forward_seg(Model *m, GlmU *ch, float *xh, int S, int pos_base, int *lost) {
    *lost = 0;
    const Cfg *c = &m->c; const int D = c->hidden, L = c->n_layers, R = c->qk_rope;
    int rows = glmu_rows_env();
    if (m->has_dsa && (pos_base + S > c->index_topk || g_dsa_force)) {   /* the scores' scratch: rows x context */
        const int64_t lim = ((int64_t)32 << 20) / (pos_base + S);
        if (lim < rows) rows = lim < 1 ? 1 : (int)lim;
    }
    if (rows > S) rows = S;
    if (!glmu_mirror(ch, m, pos_base + S) || !glmu_scratch(ch, m, rows, pos_base + S)) {
        fprintf(stderr, "[CUDA] colibri chain: device memory for %d rows at %d positions refused\n", rows, pos_base + S);
        return 0;
    }
    float *outs = (float *)malloc((size_t)S * D * sizeof(float));
    float *inv = (float *)malloc((size_t)(R / 2) * sizeof(float));
    if (!outs || !inv) { free(outs); free(inv); return 0; }
    for (int j = 0; j < R / 2; j++) inv[j] = c->rope_inv_on ? c->rope_inv[j] : powf(c->theta, -2.0f * j / R);   /* rope_interleave's */
    for (int c0 = 0; c0 < S; c0 += rows) {
        /* a served prompt's STOP or CANCEL between two chunks (prompt_break.h): not taken, and the
         * CPU's loop sees the stop before its first layer */
        if (coli_prompt_break()) { free(inv); free(outs); return 0; }
        const int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        float *cs = (float *)cc_ptr(ch->cs);
        for (int s = 0; s < n; s++)    /* the CPU's own angles, cosines and sines */
            for (int j = 0; j < R / 2; j++) {
                const float ang = (pb + s) * inv[j];
                cs[(size_t)s * R + 2 * j] = cosf(ang); cs[(size_t)s * R + 2 * j + 1] = sinf(ang);
            }
        if (!cc_begin() || !cc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) || !glmu_push(ch, m, pb)) goto lost;
        int ok = 1, pending = 0, pulled = 0, sel_on = 0;
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            const double ta = now_s();
            if (pending) {   /* x += routed + shared, the layer before's */
                CcEw cb = {CC_EW_COMBINE, n * D, D, 1, 1 | 2, 1, 0, 0, 0, 0, 0, 1.f};
                ok = cc_ew(ch->x, ch->x, ch->routed, ch->ds, NULL, &cb);
                pending = 0;
            }
            CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = ok && glmu_norm(ch->x, ch->prm, ch->o_in[i], ch->nrm, n, D, c->eps) && glmu_attn(ch, m, i, n, pb, &sel_on) &&
                 cc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add) && glmu_norm(ch->x, ch->prm, ch->o_post[i], ch->h2, n, D, c->eps);
            if (!ok) break;
            if (!l->sparse) {   /* a dense layer: its MLP here, no host step */
                ok = glmu_mlp(ch, &l->gate_proj, &l->up_proj, &l->down_proj, ch->h2, ch->tmp, n) && cc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
                m->t_attn += now_s() - ta;
                continue;
            }
            /* A1; while it runs, the tier loads the experts this layer will likely stream (a big
             * prompt chunk only) */
            ok = cc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D) && cc_submit(0);
            if (ok) XT(stream_prefetch)(i, n);
            ok = ok && cc_finish();
            m->t_attn += now_s() - ta;
            if (!ok) break;
            glmu_pull(ch, m, pulled, i + 1, pb, n); pulled = i + 1;
            /* A2: the shared expert, while the host computes the routed experts */
            ok = cc_begin() && glmu_mlp(ch, &l->sh_gate, &l->sh_up, &l->sh_down, ch->h2, ch->ds, n) && cc_submit(0);
            const double t1 = now_s();
            moe(m, l, i, (float *)cc_ptr(ch->h2d), n, ch->host_routed, 0);
            memcpy(cc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && cc_begin();
            pending = 1;
        }
        if (ok && pending) {
            CcEw cb = {CC_EW_COMBINE, n * D, D, 1, 1 | 2, 1, 0, 0, 0, 0, 0, 1.f};
            ok = cc_ew(ch->x, ch->x, ch->routed, ch->ds, NULL, &cb);
        }
        ok = ok && cc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D) && cc_submit(1);
        if (!ok) goto lost;
        glmu_pull(ch, m, pulled, L, pb, n);
        memcpy(outs + (size_t)c0 * D, cc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        for (int i = 0; i < L; i++) ch->kv_valid[i] = pb + n;
    }
    memcpy(xh, outs, (size_t)S * D * sizeof(float));
    free(inv); free(outs);
    ch->forwards++;
    return L;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    free(inv); free(outs);
    cc_finish();
    *lost = 1;
    return 0;
}
/* the chain's forward for S rows of xh (positions pos_base.., the bound state's): L, or 0 = not
 * taken (the CPU runs the forward from xh as it came in) */
static int glmu_forward(Model *m, float *xh, int S, int pos_base) {
    if (!glmu_wanted(m)) return 0;
    if (g_pilot || g_pilot_real || g_looka || g_kv8 || g_tq || g_kv16 || (exact_verify_on() && g_spec_live)) return 0;
    const Cfg *c = &m->c; const int L = c->n_layers;
    if (!m->kv || !m->kv->Lc || !m->kv->Rc || (m->has_dsa && !m->kv->Ic)) return 0;
    for (int i = 0; i < L; i++) if (m->kv_start[i] != 0 || !m->kv->Lc[i] || !m->kv->Rc[i]) return 0;
    if (!glmu_setup(m)) return 0;
    GlmU *ch = g_glmu;
    if (!ch->ok || ch->failed) return 0;
    if (!cc_ready()) { glmu_decline("the device was lost"); return 0; }
    int lost = 0;
    const int done = glmu_forward_seg(m, ch, xh, S, pos_base, &lost);
    if (done) return done;
    for (int i = 0; i < L; i++) if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;   /* the CPU writes from here */
    if (lost) {
        ch->failed = 1;
        glmu_decline("the device was lost; the CPU runs this forward again and from here on (the KV cache is the host's)");
    }
    return 0;
}
static void glmu_report(void) {
    GlmU *ch = g_glmu;
    if (!ch || !ch->forwards) return;
    fprintf(stderr, "[CUDA] colibri chain: %llu forwards, %.1f ms of routed experts on the host\n", ch->forwards, ch->host_ms);
}
#endif
