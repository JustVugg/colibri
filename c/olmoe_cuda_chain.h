/* olmoe_cuda_chain.h -- OLMoE's layers as one chain on a CUDA device (cuda_chain.h).
 * Included once by olmoe.c in a COLI_CUDA build, after the CPU forward it stands in for.
 *
 * olmoe_chain.h's chain on CUDA, with every routed expert on the card: OLMoE-1B-7B is small
 * enough to hold whole (Q4_K_M: 3.6 GiB of experts, 0.6 GiB of the rest), so a step of any
 * rows is one frame over every layer, the experts in it, and nothing waits for the host
 * between layers. Per layer: the input RMSNorm, q/k/v, OLMoE's q and k RMSNorms over the
 * whole projection, RoPE per head from a host table (the CPU's own powf/cosf/sinf), the new
 * K/V rows into the device's cache (and into a host-visible buffer for the host's cache),
 * the attention, o_proj, the residual add, the post-attention RMSNorm, the router's logits,
 * its softmax and top-k (cc_route, moe_routed's rules), the experts (cc_moe), the second
 * residual add. The last frame normalizes the last row and runs lm_head.
 *
 * Taken with COLI_CUDA=1 (COLI_CUDA_CHAIN=0: the CPU; COLI_VULKAN=1: the Vulkan device), when
 * every matrix, every expert, a prompt chunk's scratch and the KV mirror fit the card's free
 * memory: the dense matrices in their checkpoint's form (a GGUF's ggml blocks, else f32), the
 * experts as a GGUF's blocks or the merged checkpoint's int8 rows. Not with KV_SLOTS > 1 (the
 * chain keeps one conversation's KV) or a geometry its attention does not take.
 *
 * State: while the chain runs, the device's KV cache is the one that counts. Each layer has a
 * mirror of the host's cache behind a watermark (kv_valid: rows [0, kv_valid) are the step's
 * own), and the host's cache holds the device's rows only up to host_kv: the steps' K/V rows stay
 * on the device (bringing a 1K-token prompt's 131 MB down and into the host's cache took longer
 * than the prompt's kernels). The host's cache catches up when the CPU needs it: a step the
 * chain declines first reads the rows [host_kv, pos_base) down (occ_sync); a device lost
 * mid-step rebuilds them on the CPU from the token ids the chain recorded (occ_recover), and the
 * CPU runs from then on. A step from pos_base uploads the host's rows [kv_valid, pos_base)
 * first (a CPU step lowers the watermark), and a host cache allocated anew is mirrored again.
 * The mirror holds float16 where the host's cache holds float16's values (KV16, olmoe.c's
 * default) and the backend reads such a cache. COLI_CUDA_CHAIN_FAULT=n fakes the loss at the
 * n-th frame, for tests. */
#include "cuda_chain.h"

typedef struct {
    int ok, failed, dev;
    int rows;                                  /* scratch rows: a prompt chunk */
    int cap;                                   /* the host's max_t the mirrors follow */
    float **hostK;
    int kvh;                                   /* the mirrors in float16 */
    CcBuf *prm;                                /* every norm weight */
    size_t *o_in, *o_post, *o_qn, *o_kn, o_final;
    ColiCudaTensor **wq, **wk, **wv, **wo, **wg, *head;
    int *fq;                                   /* a layer's projections in one matrix: 3 q, k and v, 2 q and k, 0 none */
    CcBuf **kc, **vc;                          /* kc: a layer's mirror, K then V (vc unused: V at kv_v() in kc) */
    int *kv_valid;
    int host_kv;                               /* the host's cache holds the device's rows below this */
    int *fed;                                  /* [cap]: the token id each position was built from (occ_recover) */
    CcBuf *ex, *tab;                           /* every routed expert, and their table */
    size_t ex_stride;                          /* an expert's bytes in ex */
    int *fmt, *fmt_d;                          /* a layer's experts' formats */
    CcBuf *x, *nrm, *tmp, *q, *k, *v, *ctx, *h2, *lg, *ids, *wt, *hid, *y, *fin, *cs, *qkv;
    CcBuf *csd;                                /* cs on the device: every RoPE reads it there, not over the bus */
    CcBuf *idsd, *outd, *xd;                   /* down: every layer's routes, the logits, the rows */
    unsigned long long forwards, frames, fault_at, steps;
    double wait_ms;
} OccChain;
static OccChain *g_occ;
#define OCC_PAD(b) (((size_t)(b) + 255) / 256 * 256)

static int occ_geometry_ok(const Cfg *c) {
    return !(c->head_dim > 256 || (c->head_dim & 1) || c->head_dim * c->n_heads != c->hidden ||
             c->n_kv_heads != c->n_heads || c->n_experts < 1 || c->n_experts > CC_ROUTE_EMAX ||
             c->topk < 1 || c->topk > CC_ROUTE_KMAX || c->hidden % 32 || c->inter % 32);
}
/* prompt rows a frame: COLI_CUDA_CHAIN_ROWS (1..8192), else 1024 */
static int occ_rows_env(void) {
    const char *e = getenv("COLI_CUDA_CHAIN_ROWS");
    int r = e && *e ? atoi(e) : 1024;
    return r < 1 ? 1 : r > 8192 ? 8192 : r;
}

/* A dense matrix [O x I] on the device: a GGUF's blocks as the file has them (rows held
 * interleaved by 16 go back to the file's first), else the f32 rows. NULL: it cannot be. */
static ColiCudaTensor *occ_mat(int dev, const float *w, const OlmGx *g, int I, int O) {
    ColiCudaTensor *t = NULL;
    if (g && g->b) {
        const int b = gq_r16_base(g->t), ty = b >= 0 ? b : g->t;
        uint8_t *rows = NULL;
        if (b >= 0) {
            if (O % 16 || !(rows = malloc((size_t)gq_row_bytes(ty, I) * (size_t)O))) return NULL;
            gq_r16_unpack(g->t, rows, g->b, O, I);
        }
        int ok = coli_cuda_tensor_upload(&t, rows ? (const void *)rows : (const void *)g->b, NULL, 100 + ty, I, O, dev);
        free(rows);
        return ok ? t : NULL;
    }
    if (!w) return NULL;   /* a half-resident twin (COLI_DENSE_F16): not taken */
    return coli_cuda_tensor_upload(&t, w, NULL, 0, I, O, dev) ? t : NULL;
}

/* The projections of n matrices [O x I] of one form in one tensor [n*O x I] (rows one after
 * another: one GEMV for them, and the rows they write side by side), the first form a GGUF's
 * blocks or f32 as occ_mat takes them; NULL when their forms differ. */
static ColiCudaTensor *occ_mat_cat(int dev, const float *const *w, const OlmGx *const *g, int n, int I, int O) {
    const int gx = g[0] && g[0]->b, ty0 = gx ? (gq_r16_base(g[0]->t) >= 0 ? gq_r16_base(g[0]->t) : g[0]->t) : -1;
    for (int k = 0; k < n; k++) {
        if (gx != (g[k] && g[k]->b) || (!gx && !w[k])) return NULL;
        if (gx && (gq_r16_base(g[k]->t) >= 0 ? gq_r16_base(g[k]->t) : g[k]->t) != ty0) return NULL;
        if (gx && gq_r16_base(g[k]->t) >= 0 && O % 16) return NULL;
    }
    const size_t mb = gx ? (size_t)gq_row_bytes(ty0, I) * O : (size_t)I * O * sizeof(float);
    uint8_t *all = malloc(mb * n);
    if (!all) return NULL;
    for (int k = 0; k < n; k++) {
        if (!gx) memcpy(all + k * mb, w[k], mb);
        else if (gq_r16_base(g[k]->t) >= 0) gq_r16_unpack(g[k]->t, all + k * mb, g[k]->b, O, I);
        else memcpy(all + k * mb, g[k]->b, mb);
    }
    ColiCudaTensor *t = NULL;
    int ok = coli_cuda_tensor_upload(&t, all, NULL, gx ? 100 + ty0 : 0, I, n * O, dev);
    free(all);
    return ok ? t : NULL;
}

/* One expert's matrices into dst (stride bytes): gate at 0, up at OCC_PAD(gate's bytes), down
 * after it, each row padded to a word as cc_moe reads a GGUF's; the int8 rows' scales after
 * down's. Its gate/up bytes and down bytes as the device holds them in *gb, *db. */
static int occ_expert_bytes(Model *m, int layer, int eid, uint8_t *dst, size_t stride, uint8_t *tmp) {
    Cfg *c = &m->c; int D = c->hidden, I = c->inter;
    if (m->gx) {
        static const char *const k3[3] = {"gate_proj", "up_proj", "down_proj"};
        size_t off = 0;
        for (int k = 0; k < 3; k++) {
            const int t = m->gx_t[3 * layer + k], cols = k == 2 ? I : D, rows = k == 2 ? D : I;
            const size_t rb = (size_t)gq_row_bytes(t, cols), rw = (rb + 3) / 4 * 4;
            char nm[160];
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%d.%s.weight", layer, eid, k3[k]);
            st_tensor *st = st_find(&m->S, nm);
            if (!st || st_ggml_type_any(st->dtype) != t || st->nbytes != (int64_t)(rb * rows) ||
                off + rw * rows > stride) return 0;
            if (rw == rb) st_read_raw(&m->S, nm, dst + off, g_expert_drop);
            else {
                st_read_raw(&m->S, nm, tmp, g_expert_drop);
                for (int r = 0; r < rows; r++) { memcpy(dst + off + r * rw, tmp + r * rb, rb); memset(dst + off + r * rw + rb, 0, rw - rb); }
            }
            off += OCC_PAD(rw * rows);
        }
        return 1;
    }
    Slot s; memset(&s, 0, sizeof s);
    slot_ensure_allocated(m, &s);
    load_expert_merged(m, layer, eid, &s);
    const size_t ng = (size_t)I * D, nd = (size_t)D * I, og = 0, ou = OCC_PAD(ng), od = 2 * OCC_PAD(ng), osc = od + OCC_PAD(nd);
    int ok = osc + OCC_PAD((size_t)(2 * I + D) * sizeof(float)) <= stride;
    if (ok) {
        memcpy(dst + og, s.g, ng); memcpy(dst + ou, s.u, ng); memcpy(dst + od, s.d, nd);
        memcpy(dst + osc, s.gs, (size_t)(2 * I + D) * sizeof(float));   /* gs, us, ds side by side */
    }
    free(s.g); free(s.gs);
    return ok;
}
/* the bytes an expert takes in ex */
static size_t occ_stride(const Model *m) {
    const Cfg *c = &m->c; int D = c->hidden, I = c->inter;
    if (!m->gx) {
        size_t ng = (size_t)I * D;
        return 2 * OCC_PAD(ng) + OCC_PAD((size_t)D * I) + OCC_PAD((size_t)(2 * I + D) * sizeof(float));
    }
    size_t st = 0;
    for (int i = 0; i < c->n_layers; i++) {
        size_t b = 0;
        for (int k = 0; k < 3; k++) {
            const int cols = k == 2 ? I : D, rows = k == 2 ? D : I;
            b += OCC_PAD(((size_t)gq_row_bytes(m->gx_t[3 * i + k], cols) + 3) / 4 * 4 * rows);
        }
        if (b > st) st = b;
    }
    return st;
}
/* layer i's table entries: each expert's matrices (and the int8 rows' scales) in ex */
static void occ_entry(const OccChain *ch, const Model *m, int i, int e, const char *base, CcExpert *x) {
    const Cfg *c = &m->c; int D = c->hidden, I = c->inter;
    const char *g = base + ((size_t)i * c->n_experts + e) * ch->ex_stride;
    memset(x, 0, sizeof *x);
    if (m->gx) {
        const size_t gb = OCC_PAD(((size_t)gq_row_bytes(m->gx_t[3 * i], D) + 3) / 4 * 4 * I);
        x->g = g; x->u = g + gb; x->d = g + 2 * gb;
        return;
    }
    const size_t ng = (size_t)I * D, od = 2 * OCC_PAD(ng), osc = od + OCC_PAD((size_t)D * I);
    const float *sc = (const float *)(g + osc);
    x->g = g; x->u = g + OCC_PAD(ng); x->d = g + od;
    x->gs = sc; x->us = sc + I; x->ds = sc + 2 * I;
}

static int occ_res(CcBuf **b, size_t floats, int kind) { return cc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }
/* the scratch for `rows` rows (every layer's routes and K/V rows down at a frame's end) */
static int occ_bufs(OccChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, L = c->n_layers, hd = c->head_dim;
    (void)D;
    size_t r = (size_t)rows;
    return occ_res(&ch->x, r * D, CC_DEV) && occ_res(&ch->nrm, r * D, CC_DEV) && occ_res(&ch->tmp, r * D, CC_DEV) &&
           occ_res(&ch->q, r * D, CC_DEV) && occ_res(&ch->k, r * D, CC_DEV) && occ_res(&ch->v, r * D, CC_DEV) &&
           occ_res(&ch->ctx, r * D, CC_DEV) && occ_res(&ch->h2, r * D, CC_DEV) && occ_res(&ch->lg, r * E, CC_DEV) &&
           occ_res(&ch->ids, (size_t)L * r * K, CC_DEV) && occ_res(&ch->wt, (size_t)L * r * K, CC_DEV) &&
           occ_res(&ch->hid, cc_moe_hid_floats(rows, K, D, c->inter, E), CC_DEV) && occ_res(&ch->y, r * D, CC_DEV) &&
           occ_res(&ch->fin, D, CC_DEV) && occ_res(&ch->cs, r * hd, CC_UP) && occ_res(&ch->csd, r * hd, CC_DEV) &&
           occ_res(&ch->qkv, r * 4 * D, CC_DEV) &&
           occ_res(&ch->idsd, (size_t)L * r * K, CC_DOWN) &&
           occ_res(&ch->outd, c->vocab, CC_DOWN);
}
/* the device bytes of the scratch for `rows` rows, the KV mirror at t positions */
static size_t occ_need(const Model *m, int rows, int t, int kvh) {
    const Cfg *c = &m->c; size_t D = c->hidden, E = c->n_experts, K = c->topk, L = c->n_layers, r = (size_t)rows;
    size_t f = r * D * 9 + r * E + 2 * L * r * K + cc_moe_hid_floats(rows, (int)K, (int)D, c->inter, (int)E) + D;
    return f * sizeof(float) + L * 2 * D * (size_t)t * (kvh ? 2 : 4);
}

/* a setup that stopped: its matrices and experts freed, the chain left off (ok 0) */
static void occ_drop(OccChain *ch, int L) {
    ColiCudaTensor **a[5] = {ch->wq, ch->wk, ch->wv, ch->wo, ch->wg};
    for (int k = 0; k < 5; k++) for (int i = 0; a[k] && i < L; i++) if (a[k][i]) { coli_cuda_tensor_free(a[k][i]); a[k][i] = NULL; }
    if (ch->head) { coli_cuda_tensor_free(ch->head); ch->head = NULL; }
    cc_free(ch->ex); ch->ex = NULL;
    cc_free(ch->tab); ch->tab = NULL;
}
static void occ_shutdown(void) {
    if (!g_occ) return;
    if (g_occ->ok && !cc_lost()) cc_finish();
    cc_shutdown();
    coli_cuda_shutdown();
}

/* COLI_CUDA=1, after the weights: the device, then every matrix, every expert, the norms'
 * weights and the experts' table on it; a line saying why the CPU runs it otherwise. */
static void occ_start(Model *m) {
    const char *e = getenv("COLI_CUDA");
    if (!e || strcmp(e, "1")) return;
    const char *vk = getenv("COLI_VULKAN");
    if (vk && *vk && strcmp(vk, "0")) { fprintf(stderr, "[CUDA] olmoe: COLI_VULKAN=1 takes the device; the CUDA chain stays off\n"); return; }
    const char *cf = getenv("COLI_CUDA_CHAIN");
    if (cf && *cf == '0') { fprintf(stderr, "[CUDA] olmoe: COLI_CUDA_CHAIN=0: the CPU\n"); return; }
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, E = c->n_experts, I = c->inter;
    if (g_olm_mux_slots > 1) { fprintf(stderr, "[CUDA] olmoe: KV_SLOTS=%d: the chain keeps one conversation's KV; the CPU\n", g_olm_mux_slots); return; }
    if (!occ_geometry_ok(c)) {
        fprintf(stderr, "[CUDA] olmoe: a geometry the chain does not take (head dim %d, %d heads, %d KV heads, %d experts); the CPU\n",
                c->head_dim, c->n_heads, c->n_kv_heads, E);
        return;
    }
    if (!cc_available() || !cc_moe_ready() || !cc_fetch_ready()) {
        fprintf(stderr, "[CUDA] olmoe: the CUDA backend has no chain with experts (an older build); the CPU\n");
        return;
    }
    if (m->gx) for (int i = 0; i < L; i++) {
        CcMoe pr = {.S = 1, .K = c->topk, .D = D, .I = I, .fmt = 100 + m->gx_t[3 * i], .fmt_d = 100 + m->gx_t[3 * i + 2]};
        if (m->gx_t[3 * i] != m->gx_t[3 * i + 1] || !cc_moe_block_type(pr.fmt - 100) || !cc_moe_block_type(pr.fmt_d - 100)) {
            fprintf(stderr, "[CUDA] olmoe: layer %d's experts (%s, %s, %s) are not in the chain's block types; the CPU\n", i,
                    gq_type_name(m->gx_t[3 * i]), gq_type_name(m->gx_t[3 * i + 1]), gq_type_name(m->gx_t[3 * i + 2]));
            return;
        }
    }
    const char *gl = getenv("COLI_GPUS");
    if (!gl || !*gl) gl = getenv("COLI_GPU");
    int dev = gl && *gl ? atoi(gl) : 0;
    if (!coli_cuda_init(&dev, 1) || !cc_init(dev)) { fprintf(stderr, "[CUDA] olmoe: device %d did not come up; the CPU\n", dev); return; }
    OccChain *ch = calloc(1, sizeof *ch);
    if (!ch) return;
    g_occ = ch; ch->dev = dev;
    atexit(occ_shutdown);
    ch->kvh = olm_kv16() && cc_kvh_ready();
    if (olm_kv16() && !ch->kvh && !cc_kv16_ready()) {
        fprintf(stderr, "[CUDA] olmoe: the backend neither holds a float16 cache nor rounds to one (KV16); the CPU\n");
        return;
    }
    const char *fa = getenv("COLI_CUDA_CHAIN_FAULT");
    ch->fault_at = fa && *fa ? strtoull(fa, NULL, 10) : 0;
    /* the room: every expert, a prompt chunk's scratch, the mirror at the host's positions,
     * the dense matrices counted as they go up */
    ch->ex_stride = occ_stride(m);
    const char *cx = getenv("CTX");   /* the host's cache is sized after model_init, from CTX as here */
    int t = m->max_t > 0 ? m->max_t : cx && atoi(cx) > 0 ? atoi(cx) : 4096;
    size_t fr = 0, tot = 0, ex = (size_t)L * E * ch->ex_stride, need = ex + occ_need(m, occ_rows_env(), t, ch->kvh);
    coli_cuda_mem_info(dev, &fr, &tot);
    if (need + ((size_t)768 << 20) > fr) {
        fprintf(stderr, "[CUDA] olmoe: the experts (%.2f GiB), a prompt chunk's scratch and the KV mirror need %.2f GiB "
                        "of the card's free %.2f GiB; the CPU\n", ex / 1073741824.0, need / 1073741824.0, fr / 1073741824.0);
        return;
    }
    double t0 = now_s();
    ch->o_in = calloc(L, sizeof(size_t)); ch->o_post = calloc(L, sizeof(size_t));
    ch->o_qn = calloc(L, sizeof(size_t)); ch->o_kn = calloc(L, sizeof(size_t));
    ch->wq = calloc(L, sizeof(void *)); ch->wk = calloc(L, sizeof(void *)); ch->wv = calloc(L, sizeof(void *));
    ch->wo = calloc(L, sizeof(void *)); ch->wg = calloc(L, sizeof(void *));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *)); ch->kv_valid = calloc(L, sizeof(int));
    ch->fmt = calloc(L, sizeof(int)); ch->fmt_d = calloc(L, sizeof(int)); ch->fq = calloc(L, sizeof(int));
    if (!ch->o_in || !ch->o_post || !ch->o_qn || !ch->o_kn || !ch->wq || !ch->wk || !ch->wv || !ch->wo || !ch->wg ||
        !ch->kc || !ch->vc || !ch->kv_valid || !ch->fmt || !ch->fmt_d || !ch->fq) return;
    /* the norms' weights: offsets, then one upload */
    size_t np = 0;
    for (int i = 0; i < L; i++) { ch->o_in[i] = np; np += D; ch->o_post[i] = np; np += D; ch->o_qn[i] = np; np += D; ch->o_kn[i] = np; np += D; }
    ch->o_final = np; np += D;
    float *arena = calloc(np, sizeof(float));
    if (!arena) return;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        memcpy(arena + ch->o_in[i], l->in_ln, D * sizeof(float)); memcpy(arena + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(arena + ch->o_qn[i], l->qn, D * sizeof(float)); memcpy(arena + ch->o_kn[i], l->kn, D * sizeof(float));
    }
    memcpy(arena + ch->o_final, m->final_norm, D * sizeof(float));
    ch->prm = cc_buf(np * sizeof(float), CC_DEV);
    int ok = ch->prm && cc_begin() && cc_write(ch->prm, 0, arena, np * sizeof(float)) && cc_submit(1);
    free(arena);
    /* the dense matrices: q, k and v in one where their forms agree (q and k where only they
     * do), one GEMV for them and one norm and one RoPE for q and k; not with a float32 mirror
     * of KV16's values, whose K and V rows are rounded where they lie */
    const int fuse = !(olm_kv16() && !ch->kvh) && !(getenv("COLI_CUDA_CHAIN_QKV") && *getenv("COLI_CUDA_CHAIN_QKV") == '0');
    int nf3 = 0, nf2 = 0;
    for (int i = 0; i < L && ok; i++) {
        Layer *l = &m->L[i];
        const float *w3[3] = {l->q, l->k, l->v}; const OlmGx *g3[3] = {&l->gq, &l->gk, &l->gv};
        if (fuse && (ch->wq[i] = occ_mat_cat(dev, w3, g3, 3, D, D))) { ch->fq[i] = 3; nf3++; }
        else if (fuse && (ch->wq[i] = occ_mat_cat(dev, w3, g3, 2, D, D))) {
            ch->fq[i] = 2; nf2++;
            ok = (ch->wv[i] = occ_mat(dev, l->v, &l->gv, D, D)) != NULL;
        } else ok = (ch->wq[i] = occ_mat(dev, l->q, &l->gq, D, D)) && (ch->wk[i] = occ_mat(dev, l->k, &l->gk, D, D)) &&
                    (ch->wv[i] = occ_mat(dev, l->v, &l->gv, D, D));
        ok = ok && (ch->wo[i] = occ_mat(dev, l->o, &l->go, D, D)) && (ch->wg[i] = occ_mat(dev, l->gate, &l->ggate, D, E));
    }
    ok = ok && (ch->head = occ_mat(dev, m->lm_head, &m->ghead, D, c->vocab));
    if (!ok) {
        fprintf(stderr, "[CUDA] olmoe: a dense matrix did not reach the device (its form: %s); the CPU\n",
                m->L[0].gq.b ? gq_type_name(m->L[0].gq.t) : m->L[0].q ? "f32" : "half-resident, COLI_DENSE_F16");
        occ_drop(ch, L);
        return;
    }
    /* the experts, a layer a frame, and their table */
    ch->ex = cc_buf(ex, CC_DEV);
    uint8_t *buf = ch->ex ? malloc((size_t)E * ch->ex_stride) : NULL, *tmp = buf ? malloc(ch->ex_stride) : NULL;
    ok = buf && tmp;
    for (int i = 0; i < L && ok; i++) {
        for (int x = 0; x < E && ok; x++) ok = occ_expert_bytes(m, i, x, buf + (size_t)x * ch->ex_stride, ch->ex_stride, tmp);
        ok = ok && cc_begin() && cc_write(ch->ex, (size_t)i * E * ch->ex_stride / 4, buf, (size_t)E * ch->ex_stride) && cc_submit(1);
        if (m->gx) { ch->fmt[i] = 100 + m->gx_t[3 * i]; ch->fmt_d[i] = 100 + m->gx_t[3 * i + 2]; }
        else { ch->fmt[i] = 1; ch->fmt_d[i] = 0; }
    }
    free(buf); free(tmp);
    const char *base = ok ? (const char *)cc_dev_addr(ch->ex) : NULL;
    CcExpert *tb = base ? malloc((size_t)L * E * sizeof(CcExpert)) : NULL;
    ok = tb != NULL;
    for (int i = 0; ok && i < L; i++) for (int x = 0; x < E; x++) occ_entry(ch, m, i, x, base, &tb[(size_t)i * E + x]);
    ch->tab = ok ? cc_buf((size_t)L * E * sizeof(CcExpert), CC_DEV) : NULL;
    ok = ok && ch->tab && cc_begin() && cc_write(ch->tab, 0, tb, (size_t)L * E * sizeof(CcExpert)) && cc_submit(1);
    free(tb);
    if (!ok) {
        fprintf(stderr, "[CUDA] olmoe: the experts did not reach the device%s; the CPU\n", cc_lost() ? " (the device was lost)" : "");
        occ_drop(ch, L);
        return;
    }
    ch->ok = 1;
    fprintf(stderr, "[CUDA] olmoe chain: every layer on device %d with its %d experts (%s; %.2f GiB), the head too, "
                    "the KV mirror in %s, q/k/v in one matrix in %d layers and q/k in %d; %.1f s to place "
                    "(COLI_CUDA_CHAIN=0: the CPU)\n", dev, E, m->gx ? gq_type_name(m->gx_t[0]) : "int8 rows", ex / 1073741824.0,
            ch->kvh ? "float16" : "float32", nf3, nf2, now_s() - t0);
}

/* where V starts in a layer's mirror, in its elements */
static size_t occ_kv_v(const OccChain *ch, const Model *m) { return (size_t)m->c.n_heads * ch->cap * m->c.head_dim; }
/* the mirror at the host's capacity (a host cache allocated anew: again, from nothing) */
static int occ_mirror(OccChain *ch, Model *m) {
    Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim;
    if (ch->cap == m->max_t && ch->hostK == m->K) return 1;
    size_t es = ch->kvh ? sizeof(uint16_t) : sizeof(float);
    for (int i = 0; i < c->n_layers; i++) {
        cc_free(ch->kc[i]); ch->kc[i] = NULL; ch->kv_valid[i] = 0;
        ch->kc[i] = cc_buf((size_t)2 * H * m->max_t * hd * es, CC_DEV);   /* K, then V: one copy takes a step's rows into both */
        if (!ch->kc[i]) { ch->cap = 0; ch->hostK = NULL; return 0; }
    }
    int *fd = realloc(ch->fed, (size_t)m->max_t * sizeof(int));
    if (!fd) { ch->cap = 0; ch->hostK = NULL; return 0; }
    ch->fed = fd;
    ch->cap = m->max_t; ch->hostK = m->K; ch->host_kv = 0;
    return 1;
}
/* n of the host's K or V values from src into a mirror at element at (a float16 mirror takes
 * them as halves: float16's values, exact) */
static int occ_kv_up(const OccChain *ch, CcBuf *dst, size_t at, const float *src, size_t n) {
    if (!ch->kvh) return cc_write(dst, at, src, n * sizeof(float));
    uint16_t *h = malloc(n * sizeof(uint16_t));
    if (!h) return 0;
    kv16_pack(h, src, (int64_t)n);
    int ok = cc_write(dst, at / 2, h, n * sizeof(uint16_t));   /* a DEV buffer: the copy has the bytes once it returns */
    free(h);
    return ok;
}
/* the uploads that make the device's cache the host's below pos_base */
static int occ_push_kv(OccChain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int hd = c->head_dim, ok = 1;
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (ch->kv_valid[i] >= pos_base) continue;   /* below kv_valid the device's rows are the step's own */
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < c->n_heads && ok; h++) {
            size_t off = ((size_t)h * m->max_t + t0) * hd;
            ok = occ_kv_up(ch, ch->kc[i], off, m->K[i] + off, (size_t)n * hd) &&
                 occ_kv_up(ch, ch->kc[i], occ_kv_v(ch, m) + off, m->V[i] + off, (size_t)n * hd);
        }
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}
/* A CPU step from pos_base: the rows it writes into the host cache are not the device's. */
static void occ_cpu_step(const Model *m, int pos_base) {
    OccChain *ch = g_occ;
    if (!ch || !ch->ok) return;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
    if (ch->host_kv > pos_base) ch->host_kv = pos_base;
}
/* The host's cache up to pos_base from the device's rows (a step the chain declines, the device
 * alive): every layer's and head's rows [host_kv, pos_base) read down. 0: they could not be. */
static int occ_sync(OccChain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int L = c->n_layers, H = c->n_heads, hd = c->head_dim;
    int t1 = pos_base;
    for (int i = 0; i < L; i++) if (ch->kv_valid[i] < t1) t1 = ch->kv_valid[i];
    if (ch->host_kv >= t1) return 1;
    int t0 = ch->host_kv; size_t n = (size_t)(t1 - t0) * hd;
    uint16_t *h = ch->kvh ? malloc(n * sizeof(uint16_t)) : NULL;
    int ok = !ch->kvh || h;
    for (int i = 0; i < L && ok; i++)
        for (int hh = 0; hh < H && ok; hh++) {
            size_t off = ((size_t)hh * m->max_t + t0) * hd;
            size_t at[2] = {off, occ_kv_v(ch, m) + off}; float *dst[2] = {m->K[i] + off, m->V[i] + off};
            for (int k = 0; k < 2 && ok; k++) {
                if (ch->kvh) { ok = cc_read(ch->kc[i], at[k] / 2, h, n * sizeof(uint16_t)); if (ok) kv16_unpack(dst[k], h, (int64_t)n); }
                else ok = cc_read(ch->kc[i], at[k], dst[k], n * sizeof(float));
            }
        }
    free(h);
    if (ok) ch->host_kv = t1;
    return ok;
}
/* The host's cache below pos_base rebuilt on the CPU (the device lost with its rows): the token
 * ids the chain's steps recorded run through every layer again, from host_kv (the rows below it
 * are the host's already). */
static void occ_recover(Model *m, int pos_base) {
    OccChain *ch = g_occ;
    if (!ch || ch->host_kv >= pos_base) return;
    Cfg *c = &m->c; int D = c->hidden, t0 = ch->host_kv, n = pos_base - t0;
    fprintf(stderr, "[CUDA] olmoe chain: rebuilding the KV of positions %d..%d on the CPU\n", t0, pos_base - 1);
    float *x = falloc((int64_t)n * D);
    for (int s = 0; s < n; s++) olm_embed_row(m, ch->fed[t0 + s], x + (int64_t)s * D);
    layers_forward_range(m, x, n, t0, 0, c->n_layers, 0);
    free(x);
    ch->host_kv = pos_base;
}
static int occ_norm(CcBuf *x, size_t xo, CcBuf *w, size_t wo, CcBuf *y, size_t yo, int rows, int D, float eps) {
    CcNorm p = {rows, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, 0, eps, 1.f};
    return cc_norm(x, w, y, &p);
}
static int occ_begin(OccChain *ch) {
    ch->frames++;
    if (ch->fault_at && ch->frames == ch->fault_at) { fprintf(stderr, "[CUDA] olmoe chain: frame %llu failed (COLI_CUDA_CHAIN_FAULT)\n", ch->frames); return 0; }
    return cc_begin();
}

/* One layer for n rows from pb */
static int occ_layer(OccChain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c;
    int D = c->hidden, H = c->n_heads, hd = c->head_dim, half = hd / 2, E = c->n_experts, K = c->topk, io = i * ch->rows * K;
    CcEw add = {CC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    /* q, k and v: one buffer [n][fq * D] where the layer's projections are one matrix */
    /* with q and k in one matrix and v in its own (fq 2), v's rows go after the chunk's [rows][3D] in
     * qkv: K and V then come from one buffer, into the mirror in one copy */
    const int fq = ch->fq[i], qs = fq ? fq * D : D, vs = fq == 3 ? 3 * D : D;
    CcBuf *qb = fq ? ch->qkv : ch->q, *kb = fq ? ch->qkv : ch->k, *vb = fq ? ch->qkv : ch->v;
    const size_t ko = fq ? (size_t)D : 0, vo = fq == 3 ? 2 * (size_t)D : fq == 2 ? (size_t)ch->rows * 3 * D : 0;
    int ok = occ_norm(ch->x, 0, ch->prm, ch->o_in[i], ch->nrm, 0, n, D, c->eps);
    if (fq) {   /* one GEMV; the q and k norms in one launch (their weights side by side), RoPE on both heads' sets */
        CcNorm qk = {2 * n, D, 2, 0, qs, D, 0, qs, D, (int)ch->o_qn[i], 2, 0, c->eps, 1.f};
        CcRope rp = {n * 2 * H, 2 * H, 0, qs, hd, half, 0, 2 * half};
        ok = ok && cc_matmul(ch->wq[i], ch->nrm, 0, ch->qkv, 0, n) && (fq == 3 || cc_matmul(ch->wv[i], ch->nrm, 0, ch->qkv, vo, n)) &&
             cc_norm(ch->qkv, ch->prm, ch->qkv, &qk) && cc_rope(ch->qkv, ch->csd, &rp);
    } else {
        CcNorm qn = {n, D, 1, 0, D, D, 0, D, D, (int)ch->o_qn[i], 0, 0, c->eps, 1.f};
        CcNorm kn = {n, D, 1, 0, D, D, 0, D, D, (int)ch->o_kn[i], 0, 0, c->eps, 1.f};
        CcRope rp = {n * H, H, 0, D, hd, half, 0, 2 * half};
        ok = ok && cc_matmul(ch->wq[i], ch->nrm, 0, ch->q, 0, n) && cc_matmul(ch->wk[i], ch->nrm, 0, ch->k, 0, n) &&
             cc_matmul(ch->wv[i], ch->nrm, 0, ch->v, 0, n) &&
             cc_norm(ch->q, ch->prm, ch->q, &qn) && cc_norm(ch->k, ch->prm, ch->k, &kn) &&
             cc_rope(ch->q, ch->csd, &rp) && cc_rope(ch->k, ch->csd, &rp);
    }
    if (ok && olm_kv16() && !ch->kvh) {   /* a float32 mirror: K and V as the host's cache keeps them, before anything reads them */
        CcEw f = {CC_EW_F16, n * D, D, 0, 0, 0, 0, 0, 0, 0, 0, 1.f};
        ok = cc_ew(ch->k, ch->k, NULL, NULL, NULL, &f) && cc_ew(ch->v, ch->v, NULL, NULL, NULL, &f);
    }
    if (!ok) return 0;
    /* the new rows into the device cache (a float16 mirror's rounded on the way) */
    const size_t vk = occ_kv_v(ch, m);
    CcRegion *rg = malloc(sizeof *rg * (size_t)n * H * 2), *rv = rg ? rg + (size_t)n * H : NULL;
    if (!rg) return 0;
    for (int s = 0; s < n; s++) for (int h = 0; h < H; h++) {
        size_t dst = ((size_t)h * ch->cap + pb + s) * hd;
        rg[s * H + h] = (CcRegion){dst, (size_t)s * qs + ko + (size_t)h * hd, (size_t)hd};
        rv[s * H + h] = (CcRegion){vk + dst, (size_t)s * vs + vo + (size_t)h * hd, (size_t)hd};
    }
    if (fq) ok = ok && (ch->kvh ? cc_copy_regions_h(ch->kc[i], ch->qkv, rg, 2 * n * H) : cc_copy_regions(ch->kc[i], ch->qkv, rg, 2 * n * H));
    else ok = ok && (ch->kvh ? cc_copy_regions_h(ch->kc[i], kb, rg, n * H) && cc_copy_regions_h(ch->kc[i], vb, rv, n * H)
                             : cc_copy_regions(ch->kc[i], kb, rg, n * H) && cc_copy_regions(ch->kc[i], vb, rv, n * H));
    free(rg);
    CcAttn a = {n, H, H, hd, pb, ch->cap, 0, qs, hd, 0, 0, 0, 0, 0, D, 0, 0, 1.f / sqrtf((float)hd), 0, (int)vk};
    ok = ok && (ch->kvh ? cc_attn_h(qb, ch->kc[i], ch->kc[i], ch->ctx, NULL, NULL, &a)
                        : cc_attn(qb, ch->kc[i], ch->kc[i], ch->ctx, NULL, NULL, &a)) &&
         cc_matmul(ch->wo[i], ch->ctx, 0, ch->tmp, 0, n) && cc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
    /* the router, its softmax and top-k (moe_routed's rules), the experts, the residual */
    CcRoute ro = {n, E, K, 0, 0, E, 0, 0, io, K, io, K, 1, 1, c->norm_topk ? 1 : 0, 1.f, 1.f, 0.f};
    CcMoe mp = {n, K, D, c->inter, ch->fmt[i], 0, 0, 3, i * E, 0, D, io, K, io, K, 0, D, -1, E, 0, ch->fmt_d[i]};
    ok = ok && occ_norm(ch->x, 0, ch->prm, ch->o_post[i], ch->h2, 0, n, D, c->eps) &&
         cc_matmul(ch->wg[i], ch->h2, 0, ch->lg, 0, n) && cc_route(ch->lg, NULL, ch->ids, ch->wt, &ro) &&
         cc_moe(ch->tab, ch->h2, ch->ids, ch->wt, ch->hid, ch->y, NULL, &mp) && cc_ew(ch->x, ch->x, ch->y, NULL, NULL, &add);
    ch->steps++;
    return ok;
}

/* After a frame of n rows from pb: the routes' books, the watermarks */
static void occ_done(OccChain *ch, Model *m, int n, int pb) {
    Cfg *c = &m->c; int K = c->topk, L = c->n_layers, r = ch->rows;
    const int *ids = (const int *)cc_ptr(ch->idsd);
    for (int i = 0; i < L; i++) {
        const int *li = ids + (size_t)i * r * K;
        if (!m->hot_pinned) for (int s = 0; s < n; s++) rt_count(i, li + s * K, K);
        pthread_mutex_lock(&g_pilot_mx);
        for (int j = 0; j < n * K; j++) ehit_mark(m, i, li[j]);
        pthread_mutex_unlock(&g_pilot_mx);
        ch->kv_valid[i] = pb + n;
    }
}

/* Every layer for S rows from host rows xh, the last row's logits into `logit`; xh gets the
 * final rows back when want_x. 0: not taken (the host cache below pos_base and xh as they
 * were: the caller runs the step on the CPU); else the layers run. */
static int occ_forward(Model *m, const int *ids, float *xh, int S, int pos_base, int want_x, float *logit) {
    OccChain *ch = g_occ;
    if (!ch || !ch->ok || ch->failed || m->mux_rows || S < 1) return 0;
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, hd = c->head_dim, half = hd / 2;
    int CH = occ_rows_env(), rows = S < CH ? S : CH;
    if (rows > ch->rows) ch->rows = rows;
    if (pos_base + S > m->max_t || !occ_mirror(ch, m) || !occ_bufs(ch, m, ch->rows) ||
        (want_x && !occ_res(&ch->xd, (size_t)ch->rows * D, CC_DOWN))) {
        fprintf(stderr, "[CUDA] olmoe chain: device memory for %d rows refused; the CPU from here on\n", rows);
        ch->failed = 1;
        if (cc_lost() || !occ_sync(ch, m, pos_base)) occ_recover(m, pos_base);   /* the host's cache, for the CPU */
        return 0;
    }
    memcpy(ch->fed + pos_base, ids, (size_t)S * sizeof(int));
    if (ch->host_kv > pos_base) ch->host_kv = pos_base;   /* the step rewrites from pos_base on the device only */
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        float *cs = (float *)cc_ptr(ch->cs);   /* rope_head's angles, cosines and sines */
        for (int s = 0; s < n; s++)
            for (int j = 0; j < half; j++) {
                float inv = powf(c->theta, -2.0f * j / c->head_dim), ang = (pb + s) * inv;
                cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
            }
        /* the angles to the device first: a RoPE reading them over the bus took 22.5 us a call, 0.7 ms
         * of a 3.9 ms decode token (OLMoE-1B-7B Q4_K_M, RTX PRO 4000) */
        int ok = occ_begin(ch) && cc_copy(ch->csd, 0, ch->cs, 0, (size_t)n * hd) &&
                 cc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) && occ_push_kv(ch, m, pb);
        for (int i = 0; i < L && ok; i++) ok = occ_layer(ch, m, i, n, pb);
        if (ok) {   /* every layer's routes down for the books, in one launch */
            CcRegion rg[256];
            size_t lk = (size_t)ch->rows * c->topk;
            for (int i = 0; i < L && i < 256; i++) rg[i] = (CcRegion){(size_t)i * lk, (size_t)i * lk, (size_t)n * c->topk};
            ok = L <= 256 && cc_copy_regions(ch->idsd, ch->ids, rg, L);
        }
        if (ok && want_x) ok = cc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D);
        int last = c0 + n == S;
        if (ok && last) ok = occ_norm(ch->x, (size_t)(n - 1) * D, ch->prm, ch->o_final, ch->fin, 0, 1, D, c->eps) &&
                             cc_matmul(ch->head, ch->fin, 0, ch->outd, 0, 1);
        double t0 = now_s();
        ok = ok && cc_submit(1);
        ch->wait_ms += (now_s() - t0) * 1e3; g_prof_attn_s += now_s() - t0;
        if (!ok) {
            if (!cc_lost()) cc_finish();
            ch->failed = 1;
            for (int i = 0; i < L; i++) if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
            fprintf(stderr, "[CUDA] olmoe chain: the device was lost at position %d; the CPU runs from here on\n", pos_base);
            occ_recover(m, pos_base);   /* then the CPU redoes the step from the embedding rows */
            return 0;
        }
        occ_done(ch, m, n, pb);
        if (want_x) memcpy(xh + (size_t)c0 * D, cc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        if (last) memcpy(logit, cc_ptr(ch->outd), (size_t)c->vocab * sizeof(float));
    }
    ch->forwards++;
    return L;
}

static void occ_report(void) {
    OccChain *ch = g_occ;
    if (!ch || !ch->ok || !ch->forwards) return;
    CcStats st; memset(&st, 0, sizeof st);
    if (!cc_lost()) cc_stats(&st);
    fprintf(stderr, "[CUDA] olmoe chain: %llu forwards, %llu frames (%llu ops, %llu matmuls), %.1f ms waiting for the device, "
                    "%llu layer steps, %.1f MiB of chain buffers\n",
            ch->forwards, st.frames, st.ops, st.matmuls, ch->wait_ms, ch->steps, st.dev_bytes / 1048576.0);
}
