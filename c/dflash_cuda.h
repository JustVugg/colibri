/* dflash_cuda.h -- DFlash2's drafter on a CUDA chain's device (cuda_chain.h's ops), for an engine
 * whose chain hands it the rows of its target layers: qwen36.c (Qwen3.8-27B) and qwen38.c
 * (Qwen3.8-Flash-Next). Written from the DFlash paper and the DFlash2 description (block diffusion
 * drafting, the two-tap convolutions, the pairwise path selector) and the checkpoint's own names
 * and shapes; the arrangement below is the one whose drafts the target accepts (measured offline,
 * docs/speculative.md).
 *
 * The drafter (a directory with config.json and model.safetensors: z-lab/Qwen3.8-27B-DFlash2,
 * incoai/Qwen3.8-Flash-Next-DFlash2) is a few decoder layers of the target's width that draft a
 * whole block in one pass. Its context is the target itself: for every position the target runs,
 * its rows after the target layers (dflash_config.target_layer_ids, a row each), side by side,
 * through fc and hidden_norm, give each drafter layer that position's key and value (k_proj,
 * k_norm, RoPE; v_proj), kept in the drafter's own cache. A block is the last verified token and
 * block_size - 1 mask tokens at the positions that follow, each the target's embedding of its
 * token. In each layer the block's rows attend to the cache before the block and to each other;
 * before and after each sublayer a two-tap convolution (cuda_dflash.cuh's dfconv) mixes each row
 * with the row before it in the block, its coefficients a learned base plus a correction the
 * sublayer's normed input gives each group of 16 channels. The mask rows' outputs through the
 * drafter's norm and the target's lm_head are the drafts' logits.
 *
 * The selector: the top 16 of each position's logits (dftopk), each scored with its predecessor,
 * S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)> (U the logit, A and B the predecessor and successor
 * codebooks, H the hidden projection of the position's normed output), the path walked greedily
 * from the verified token: draft t is the best successor of draft t - 1.
 *
 * Matrices as Q8_0 blocks (a GGUF's, the chain's block GEMVs), the codebooks on the host.
 * The engine owns when: df_tap during its layers, df_ctx after them (both inside its open frame),
 * df_note after the forward, df_draft before a verify. */
#ifndef DFLASH_CUDA_H
#define DFLASH_CUDA_H
#include "kv16.h"   /* float16 bits (a Q8_0 block's scale) */

#define DF_MAX_L 16
#define DF_MAX_T 8
#define DF_MAX_B 16
#define DF_MAX_K 32
#define DF_MAX_RANK 512
#define DF_TOPK_SPLIT 64   /* a row's slices the top-k's first pass takes apart (cc_dftopk's scratch) */

typedef struct {
    ColiCudaTensor *q, *k, *v, *o, *gate, *up, *down, *acp, *mcp;
    size_t o_ln1, o_ln2, o_qn, o_kn, o_acb, o_mcb;
} DfLayer;

typedef struct {
    int on, failed, dev;
    int D, L, H, KV, hd, I, bs, mask, gs, G, rank, topk, nt, tl[DF_MAX_T], V, causal, select;
    float eps, theta;
    ColiCudaTensor *fc, *hp, *lm;
    DfLayer ly[DF_MAX_L];
    CcBuf *prm; size_t o_hn, o_norm;
    CcBuf *kc, *vc; int cap;                 /* layer l's cache at l KV cap hd: K[(h cap + t) hd + d] */
    CcBuf *feat, *cx, *cn, *ck, *cv, *ccs; int rows;   /* a chunk's tapped rows and their K/V rows */
    CcBuf *h, *n, *a, *t, *kp, *qb, *kb, *vb, *att, *g, *u, *sw, *cs, *fin, *lg, *top, *topd, *hg, *hgd, *pick, *tsc;
    CcBuf *up, *cup; int cup_rows;           /* the block's rows and angles, a chunk's angles: mapped, copied by a kernel */
    float *mask_row;                         /* the target's embedding of the mask token */
    uint16_t *A, *B;                         /* the codebooks, bf16 [V][rank] */
    int *ids; int ids_cap, len;              /* the tokens of the positions the cache holds, [0, len) */
    int fed;                                 /* rows df_ctx took since the last df_note */
    unsigned long long blocks, ctx_rows; double ms, ctx_ms;
} DfDrafter;

static int df_res(CcBuf **b, size_t floats, int kind) { return cc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }

static int df_cfg_int(jval *o, const char *k, int *out) {
    jval *v = o ? json_get(o, k) : NULL;
    if (!v || v->t != J_NUM) return 0;
    *out = (int)v->num; return 1;
}

/* a matrix [O][I] of the checkpoint on the device as a GGUF's Q8_0 rows (blocks of 32: a float16 scale,
 * 32 int8; format 100 + 8): the chain's block GEMVs take a block's 8 rows on the tensor cores at any
 * width (the int8 rows' GEMV stops at 12288 inputs, under the MLP's down and fc) */
static ColiCudaTensor *df_upload_q8(shards *S, const char *name, int I, int O, int dev) {
    st_tensor *t = st_find(S, name);
    if (!t || t->numel != (int64_t)I * O || I % 32) { fprintf(stderr, "[dflash] %s: missing or not [%d, %d]\n", name, O, I); return NULL; }
    float *w = falloc((int64_t)I * O);
    const size_t rb = (size_t)I / 32 * 34;
    uint8_t *q = (uint8_t *)malloc(rb * O);
    ColiCudaTensor *out = NULL;
    if (q && st_read_f32(S, name, w, 1) == (int64_t)I * O) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++)
            for (int b = 0; b < I / 32; b++) {
                const float *x = w + (size_t)o * I + (size_t)b * 32;
                uint8_t *blk = q + (size_t)o * rb + (size_t)b * 34;
                float am = 0.f;
                for (int i = 0; i < 32; i++) { float a = fabsf(x[i]); if (a > am) am = a; }
                uint16_t dh = kv16_from(am / 127.f);
                float dd = kv16_to(dh), id = dd > 0.f ? 1.f / dd : 0.f;
                memcpy(blk, &dh, 2);
                for (int i = 0; i < 32; i++) {
                    float v = rintf(x[i] * id);
                    blk[2 + i] = (uint8_t)(int8_t)(v > 127.f ? 127.f : v < -127.f ? -127.f : v);
                }
            }
        if (!coli_cuda_tensor_upload(&out, q, NULL, 108, I, O, dev)) out = NULL;
    }
    free(w); free(q);
    return out;
}
static int df_read_vec(shards *S, const char *name, float *out, int64_t n) {
    st_tensor *t = st_find(S, name);
    if (!t || t->numel != n) { fprintf(stderr, "[dflash] %s: missing or not %lld values\n", name, (long long)n); return 0; }
    return st_read_f32(S, name, out, 1) == n;
}

static void df_free(DfDrafter *d) {
    CcBuf **bufs[] = {&d->prm, &d->kc, &d->vc, &d->feat, &d->cx, &d->cn, &d->ck, &d->cv, &d->ccs, &d->h, &d->n, &d->a, &d->t,
                      &d->kp, &d->qb, &d->kb, &d->vb, &d->att, &d->g, &d->u, &d->sw, &d->cs, &d->fin, &d->lg, &d->top, &d->topd,
                      &d->hg, &d->hgd, &d->pick, &d->tsc, &d->up, &d->cup};
    for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++) { cc_free(*bufs[i]); *bufs[i] = NULL; }
    ColiCudaTensor *ts[] = {d->fc, d->hp};
    for (size_t i = 0; i < 2; i++) if (ts[i]) coli_cuda_tensor_free(ts[i]);
    for (int l = 0; l < d->L && l < DF_MAX_L; l++) {
        DfLayer *y = &d->ly[l];
        ColiCudaTensor *lt[] = {y->q, y->k, y->v, y->o, y->gate, y->up, y->down, y->acp, y->mcp};
        for (size_t i = 0; i < 9; i++) if (lt[i]) coli_cuda_tensor_free(lt[i]);
    }
    free(d->mask_row); free(d->A); free(d->B); free(d->ids);
    memset(d, 0, sizeof *d);
}

/* The drafter from dir for a target of width D and vocabulary V whose lm_head is lm on device dev;
 * mask_row(ctx, id, out) gives the target's embedding of a token. 1 up, 0 off (said why). */
static int df_load(DfDrafter *d, const char *dir, int D, int V, int dev, ColiCudaTensor *lm,
                   void (*embed)(void *ctx, int id, float *out), void *ectx) {
    memset(d, 0, sizeof *d);
    d->failed = 1;
    if (!cc_dflash_ready() || !cc_pick_rows_ready()) { fprintf(stderr, "[dflash] the CUDA module has no drafter ops (an older backend): off\n"); return 0; }
    char path[4200]; snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[dflash] %s: cannot open\n", path); return 0; }
    fseek(f, 0, SEEK_END); long fl = ftell(f); fseek(f, 0, SEEK_SET);
    char *txt = (char *)malloc((size_t)fl + 1);
    if (!txt || fread(txt, 1, (size_t)fl, f) != (size_t)fl) { fclose(f); free(txt); return 0; }
    fclose(f); txt[fl] = 0;
    char *arena = NULL; jval *cfg = json_parse(txt, &arena);
    jval *dc = cfg ? json_get(cfg, "dflash_config") : NULL, *rp = cfg ? json_get(cfg, "rope_parameters") : NULL;
    jval *tl = dc ? json_get(dc, "target_layer_ids") : NULL, *eps = cfg ? json_get(cfg, "rms_norm_eps") : NULL;
    jval *th = rp ? json_get(rp, "rope_theta") : cfg ? json_get(cfg, "rope_theta") : NULL;
    int ok = cfg && dc && tl && tl->t == J_ARR && df_cfg_int(cfg, "hidden_size", &d->D) && df_cfg_int(cfg, "num_hidden_layers", &d->L) &&
             df_cfg_int(cfg, "num_attention_heads", &d->H) && df_cfg_int(cfg, "num_key_value_heads", &d->KV) &&
             df_cfg_int(cfg, "head_dim", &d->hd) && df_cfg_int(cfg, "intermediate_size", &d->I) && df_cfg_int(cfg, "vocab_size", &d->V) &&
             df_cfg_int(dc, "block_size", &d->bs) && df_cfg_int(dc, "mask_token_id", &d->mask) &&
             df_cfg_int(dc, "conv_group_size", &d->gs) && df_cfg_int(dc, "selector_rank", &d->rank) && df_cfg_int(dc, "selector_top_k", &d->topk) &&
             eps && eps->t == J_NUM && th && th->t == J_NUM;
    if (ok) {
        d->eps = (float)eps->num; d->theta = (float)th->num; d->nt = tl->len;
        for (int i = 0; ok && i < d->nt && i < DF_MAX_T; i++) ok = tl->kids[i]->t == J_NUM && (d->tl[i] = (int)tl->kids[i]->num) >= 0;
    }
    json_free(cfg); free(arena); free(txt);
    if (!ok) { fprintf(stderr, "[dflash] %s: not a DFlash2 drafter's config (dflash_config and its fields)\n", path); return 0; }
    if (d->D != D || d->V != V || d->L < 1 || d->L > DF_MAX_L || d->nt < 1 || d->nt > DF_MAX_T || d->bs < 2 || d->bs > DF_MAX_B ||
        d->gs < 1 || d->D % d->gs || d->topk < 1 || d->topk > DF_MAX_K || d->rank < 1 || d->rank > DF_MAX_RANK || d->H % d->KV || d->hd % 2 || d->mask < 0 || d->mask >= V) {
        fprintf(stderr, "[dflash] %s: a drafter of width %d, vocabulary %d, block %d, %d layers does not fit this model (width %d, vocabulary %d)\n",
                dir, d->D, d->V, d->bs, d->L, D, V);
        return 0;
    }
    d->G = d->D / d->gs; d->dev = dev; d->lm = lm;
    const char *e = getenv("COLI_DFLASH_SELECT");
    d->select = !(e && *e == '0');
    e = getenv("COLI_DFLASH_CAUSAL");
    d->causal = e && *e == '1';
    shards S; memset(&S, 0, sizeof S);
    st_init(&S, dir);
    const int Dm = d->D, hd = d->hd, Ln = d->L;
    size_t n = 0;
    d->o_hn = n; n += Dm; d->o_norm = n; n += Dm;
    for (int l = 0; l < Ln; l++) {
        DfLayer *y = &d->ly[l];
        y->o_ln1 = n; n += Dm; y->o_ln2 = n; n += Dm; y->o_qn = n; n += hd; y->o_kn = n; n += hd;
        y->o_acb = n; n += 4 * (size_t)Dm; y->o_mcb = n; n += 4 * (size_t)Dm;
    }
    float *w = falloc((int64_t)n);
    ok = df_read_vec(&S, "hidden_norm.weight", w + d->o_hn, Dm) && df_read_vec(&S, "norm.weight", w + d->o_norm, Dm);
    char nm[256];
#define DF_NAME(fmt) (snprintf(nm, sizeof nm, fmt, l), nm)
    for (int l = 0; ok && l < Ln; l++) {
        DfLayer *y = &d->ly[l];
        ok = df_read_vec(&S, DF_NAME("layers.%d.input_layernorm.weight"), w + y->o_ln1, Dm) &&
             df_read_vec(&S, DF_NAME("layers.%d.post_attention_layernorm.weight"), w + y->o_ln2, Dm) &&
             df_read_vec(&S, DF_NAME("layers.%d.self_attn.q_norm.weight"), w + y->o_qn, hd) &&
             df_read_vec(&S, DF_NAME("layers.%d.self_attn.k_norm.weight"), w + y->o_kn, hd) &&
             df_read_vec(&S, DF_NAME("layers.%d.attention_conv.base_kernel"), w + y->o_acb, 4 * (int64_t)Dm) &&
             df_read_vec(&S, DF_NAME("layers.%d.mlp_conv.base_kernel"), w + y->o_mcb, 4 * (int64_t)Dm);
    }
    ok = ok && df_res(&d->prm, n, CC_DEV) && cc_begin() && cc_write(d->prm, 0, w, n * sizeof(float)) && cc_submit(1);
    free(w);
    const int qo = d->H * hd, kvo = d->KV * hd;
    d->fc = ok ? df_upload_q8(&S, "fc.weight", d->nt * Dm, Dm, dev) : NULL;
    ok = d->fc != NULL;
    for (int l = 0; ok && l < Ln; l++) {
        DfLayer *y = &d->ly[l];
        ok = (y->q = df_upload_q8(&S, DF_NAME("layers.%d.self_attn.q_proj.weight"), Dm, qo, dev)) &&
             (y->k = df_upload_q8(&S, DF_NAME("layers.%d.self_attn.k_proj.weight"), Dm, kvo, dev)) &&
             (y->v = df_upload_q8(&S, DF_NAME("layers.%d.self_attn.v_proj.weight"), Dm, kvo, dev)) &&
             (y->o = df_upload_q8(&S, DF_NAME("layers.%d.self_attn.o_proj.weight"), qo, Dm, dev)) &&
             (y->gate = df_upload_q8(&S, DF_NAME("layers.%d.mlp.gate_proj.weight"), Dm, d->I, dev)) &&
             (y->up = df_upload_q8(&S, DF_NAME("layers.%d.mlp.up_proj.weight"), Dm, d->I, dev)) &&
             (y->down = df_upload_q8(&S, DF_NAME("layers.%d.mlp.down_proj.weight"), d->I, Dm, dev)) &&
             (y->acp = df_upload_q8(&S, DF_NAME("layers.%d.attention_conv.kernel_projection.weight"), Dm, 4 * d->G, dev)) &&
             (y->mcp = df_upload_q8(&S, DF_NAME("layers.%d.mlp_conv.kernel_projection.weight"), Dm, 4 * d->G, dev));
    }
#undef DF_NAME
    if (ok && d->select) {   /* the selector: its projection on the device (f32), the codebooks here */
        float *hp = falloc((int64_t)d->rank * Dm), *ones = falloc(d->rank);
        for (int i = 0; i < d->rank; i++) ones[i] = 1.f;
        ok = df_read_vec(&S, "candidate_selector.hidden_projection.weight", hp, (int64_t)d->rank * Dm) &&
             coli_cuda_tensor_upload(&d->hp, hp, ones, 0, Dm, d->rank, dev);
        free(hp); free(ones);
        st_tensor *ta = st_find(&S, "candidate_selector.predecessor_codebook"), *tb = st_find(&S, "candidate_selector.successor_codebook");
        size_t cb = (size_t)V * d->rank;
        ok = ok && ta && tb && ta->dtype == 0 && tb->dtype == 0 && ta->numel == (int64_t)cb && tb->numel == (int64_t)cb &&
             (d->A = (uint16_t *)malloc(cb * 2)) && (d->B = (uint16_t *)malloc(cb * 2));
        if (ok) { st_read_raw(&S, "candidate_selector.predecessor_codebook", d->A, 1); st_read_raw(&S, "candidate_selector.successor_codebook", d->B, 1); }
    }
    st_destroy(&S);
    const size_t B = (size_t)d->bs;
    ok = ok && (d->mask_row = falloc(Dm)) && df_res(&d->h, B * Dm, CC_DEV) && df_res(&d->n, B * Dm, CC_DEV) &&
         df_res(&d->a, B * Dm, CC_DEV) && df_res(&d->t, B * Dm, CC_DEV) && df_res(&d->kp, B * 4 * d->G, CC_DEV) &&
         df_res(&d->qb, B * qo, CC_DEV) && df_res(&d->kb, B * kvo, CC_DEV) && df_res(&d->vb, B * kvo, CC_DEV) &&
         df_res(&d->att, B * qo, CC_DEV) && df_res(&d->g, B * d->I, CC_DEV) && df_res(&d->u, B * d->I, CC_DEV) &&
         df_res(&d->sw, B * d->I, CC_DEV) && df_res(&d->cs, B * hd, CC_DEV) && df_res(&d->fin, B * Dm, CC_DEV) &&
         df_res(&d->lg, (B - 1) * V, CC_DEV) && df_res(&d->top, (B - 1) * 2 * d->topk, CC_DEV) &&
         df_res(&d->topd, (B - 1) * 2 * d->topk, CC_DOWN) && df_res(&d->pick, (B - 1) * 2, CC_DOWN) &&
         df_res(&d->tsc, (B - 1) * DF_TOPK_SPLIT * 2 * d->topk, CC_DEV) && df_res(&d->up, B * (Dm + hd), CC_UP) &&
         (!d->select || (df_res(&d->hg, (B - 1) * d->rank, CC_DEV) && df_res(&d->hgd, (B - 1) * d->rank, CC_DOWN)));
    if (!ok) {
        fprintf(stderr, "[dflash] %s: the drafter does not fit the device (or a tensor is missing): off\n", dir);
        df_free(d); d->failed = 1;
        return 0;
    }
    embed(ectx, d->mask, d->mask_row);
    d->failed = 0; d->on = 1;
    fprintf(stderr, "[dflash] %s: %d layers of %d, target layers", dir, d->L, d->D);
    for (int i = 0; i < d->nt; i++) fprintf(stderr, " %d", d->tl[i]);
    fprintf(stderr, ", blocks of %d, %s attention in the block, %s\n", d->bs, d->causal ? "causal" : "bidirectional",
            d->select ? "the path selector" : "each position's best");
    return 1;
}

/* the index of target layer `layer` among the drafter's, -1 when it is not one */
static int df_tap_index(const DfDrafter *d, int layer) {
    if (!d->on) return -1;
    for (int i = 0; i < d->nt; i++) if (d->tl[i] == layer) return i;
    return -1;
}
/* room for a chunk of rows tapped rows, and the cache for cap positions (kept when it grows) */
static int df_reserve(DfDrafter *d, int rows, int cap) {
    if (!d->on) return 0;
    const size_t D = (size_t)d->D, kvo = (size_t)d->KV * d->hd;
    if (rows > d->rows) {
        if (!df_res(&d->feat, (size_t)rows * d->nt * D, CC_DEV) || !df_res(&d->cx, (size_t)rows * D, CC_DEV) ||
            !df_res(&d->cn, (size_t)rows * D, CC_DEV) || !df_res(&d->ck, (size_t)rows * kvo, CC_DEV) ||
            !df_res(&d->cv, (size_t)rows * kvo, CC_DEV) || !df_res(&d->ccs, (size_t)rows * d->hd, CC_DEV)) return 0;
        d->rows = rows;
    }
    if (cap > d->cap) {
        size_t per = (size_t)d->L * kvo;
        CcBuf *kc = NULL, *vc = NULL;
        if (!df_res(&kc, per * cap, CC_DEV) || !df_res(&vc, per * cap, CC_DEV)) { cc_free(kc); cc_free(vc); return 0; }
        int ok = 1;
        if (d->cap && d->len) {   /* the rows so far, at the new stride */
            ok = cc_begin();
            for (int l = 0; ok && l < d->L; l++)
                for (int h = 0; ok && h < d->KV; h++) {
                    size_t dst = ((size_t)l * d->KV + h) * cap * d->hd, src = ((size_t)l * d->KV + h) * d->cap * d->hd;
                    ok = cc_copy(kc, dst, d->kc, src, (size_t)d->len * d->hd) && cc_copy(vc, dst, d->vc, src, (size_t)d->len * d->hd);
                }
            ok = ok && cc_submit(1);
        }
        if (!ok) { cc_free(kc); cc_free(vc); d->len = 0; return 0; }
        cc_free(d->kc); cc_free(d->vc); d->kc = kc; d->vc = vc; d->cap = cap;
    }
    return 1;
}
/* in the engine's open frame: rows s < n of x (x_off + s x_row, C streams of D: their sum) are the
 * target's rows after target layer j, the chunk's rows row0.. */
static int df_tap(DfDrafter *d, CcBuf *x, size_t x_off, int x_row, int C, int n, int j, int row0) {
    if (row0 + n > d->rows) return 0;
    CcDfTap p = {n, d->D, C, (int)x_off, x_row, (row0 * d->nt + j) * d->D, d->nt * d->D};
    return cc_dftap(x, d->feat, &p);
}
static void df_angles(const DfDrafter *d, int pos0, int n, float *cs) {
    const int half = d->hd / 2;
    for (int s = 0; s < n; s++)
        for (int j = 0; j < half; j++) {
            double inv = pow((double)d->theta, -2.0 * j / d->hd), ang = (double)(pos0 + s) * inv;
            cs[(s * half + j) * 2] = (float)cos(ang); cs[(s * half + j) * 2 + 1] = (float)sin(ang);
        }
}
/* rows (n, at positions pos0..) of a layer's K and V scratch into its cache */
static int df_store(DfDrafter *d, int l, CcBuf *k, CcBuf *v, int n, int pos0) {
    const int KV = d->KV, hd = d->hd;
    CcRegion *rg = (CcRegion *)malloc(sizeof *rg * (size_t)n * KV);
    if (!rg) return 0;
    for (int s = 0; s < n; s++) for (int h = 0; h < KV; h++)
        rg[s * KV + h] = (CcRegion){(((size_t)l * KV + h) * d->cap + pos0 + s) * hd, ((size_t)s * KV + h) * hd, (size_t)hd};
    int ok = cc_copy_regions(d->kc, k, rg, n * KV) && cc_copy_regions(d->vc, v, rg, n * KV);
    free(rg);
    return ok;
}
static int df_norm(DfDrafter *d, CcBuf *x, size_t xo, int rows, int D, int per_row, size_t w_off, CcBuf *y, size_t yo) {
    CcNorm np = {rows * per_row, D, per_row, (int)xo, per_row * D, D, (int)yo, per_row * D, D, (int)w_off, 0, 0, d->eps, 1.f};
    return cc_norm(x, d->prm, y, &np);
}
static int df_rope(CcBuf *x, CcBuf *cs, int rows, int heads, int hd) {
    CcRope r = {rows * heads, heads, 0, heads * hd, hd, hd / 2, 0, hd};
    return cc_rope(x, cs, &r);
}
/* in the engine's open frame, after df_tap of every target layer: the n tapped rows' keys and values
 * in each layer's cache at positions pos0.. */
static int df_ctx(DfDrafter *d, int n, int pos0) {
    if (!d->on || n < 1 || n > d->rows || pos0 + n > d->cap) return 0;
    double t0 = now_s();
    /* the angles through mapped memory and a kernel's copy: a write (cudaMemcpyAsync) waits on the copy
     * engine behind the experts a chain promotes (cuda_chain.cu's cc_copy_k) */
    if (n > d->cup_rows) { if (!df_res(&d->cup, (size_t)n * d->hd, CC_UP)) return 0; d->cup_rows = n; }
    df_angles(d, pos0, n, (float *)cc_ptr(d->cup));
    int ok = cc_copy(d->ccs, 0, d->cup, 0, (size_t)n * d->hd) &&
             cc_matmul(d->fc, d->feat, 0, d->cx, 0, n) && df_norm(d, d->cx, 0, n, d->D, 1, d->o_hn, d->cn, 0);
    for (int l = 0; ok && l < d->L; l++) {
        DfLayer *y = &d->ly[l];
        ok = cc_matmul(y->k, d->cn, 0, d->ck, 0, n) && cc_matmul(y->v, d->cn, 0, d->cv, 0, n) &&
             df_norm(d, d->ck, 0, n, d->hd, d->KV, y->o_kn, d->ck, 0) && df_rope(d->ck, d->ccs, n, d->KV, d->hd) &&
             df_store(d, l, d->ck, d->cv, n, pos0);
    }
    d->ctx_rows += (unsigned long long)n; d->ctx_ms += (now_s() - t0) * 1e3;
    if (ok) d->fed += n;
    return ok;
}
/* after a forward of ids[0..S) at pos_base, fed[0..fed_len) the tokens the target's state holds.
 * The forward's rows are in the cache when the chain gave df_ctx every one of them. The cache's
 * rows count while they are the target's own: a forward the chain did not run, or a state the
 * target took from elsewhere (another conversation's prefix), leaves none (until a prompt from
 * position 0). */
static void df_note(DfDrafter *d, const int *ids, int S, int pos_base, const int *fed, int fed_len) {
    if (!d->on) return;
    const int ran = d->fed == S;
    d->fed = 0;
    int ok = ran && pos_base + S <= d->cap &&
             (pos_base == 0 || (pos_base <= d->len && fed && fed_len >= pos_base && !memcmp(d->ids, fed, (size_t)pos_base * sizeof(int))));
    if (ok && pos_base + S > d->ids_cap) {
        int c = d->ids_cap ? d->ids_cap : 1024;
        while (c < pos_base + S) c *= 2;
        int *t = (int *)realloc(d->ids, (size_t)c * sizeof(int));
        if (!t) ok = 0; else { d->ids = t; d->ids_cap = c; }
    }
    if (!ok) { d->len = 0; return; }
    memcpy(d->ids + pos_base, ids, (size_t)S * sizeof(int));
    d->len = pos_base + S;
}
/* whether a block can follow the token about to be fed at pos */
static int df_ready(const DfDrafter *d, int pos) { return d->on && pos >= 1 && d->len >= pos && pos + d->bs <= d->cap; }

static inline float df_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

/* The block after tok (about to be fed at pos, its embedding emb): up to kmax drafts into out, their
 * probabilities among the position's candidates into prob. One frame, waited for. 0: none. */
static int df_draft(DfDrafter *d, const float *emb, int tok, int pos, int kmax, int *out, float *prob) {
    if (!df_ready(d, pos) || kmax < 1) return 0;
    double t0 = now_s();
    const int B = d->bs, D = d->D, hd = d->hd, H = d->H, KV = d->KV, qo = H * hd, I = d->I, V = d->V, K = d->topk, G = d->G;
    float *rows = (float *)cc_ptr(d->up), *cs = rows + (size_t)B * D;   /* mapped: a kernel copies them (see df_ctx) */
    memcpy(rows, emb, (size_t)D * sizeof(float));
    for (int r = 1; r < B; r++) memcpy(rows + (size_t)r * D, d->mask_row, (size_t)D * sizeof(float));
    df_angles(d, pos, B, cs);
    int ok = cc_begin() && cc_copy(d->h, 0, d->up, 0, (size_t)B * D) && cc_copy(d->cs, 0, d->up, (size_t)B * D, (size_t)B * hd);
    CcEw add = {CC_EW_ADD, B * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    CcEw sw = {CC_EW_SWIGLU, B * I, I, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    for (int l = 0; ok && l < d->L; l++) {
        DfLayer *y = &d->ly[l];
        /* the two convolutions of a sublayer: pre on its normed input, post on its output, the
         * coefficients' corrections both from the normed input's projection */
        CcDfConv pre = {B, D, B, d->gs, 0, D, 0, 4 * G, G, 1, 0, D, (int)y->o_acb, 0};
        CcDfConv post = pre; post.kp_off = 2 * G; post.b_off = (int)y->o_acb + 2 * D;
        CcAttn at = {d->causal ? B : 1, H, KV, hd, d->causal ? pos : pos + B - 1, d->cap, 0, qo, hd, 0, 0, 0, 0, 0, qo, 0, 0,
                     1.f / sqrtf((float)hd), (int)((size_t)l * KV * d->cap * hd), (int)((size_t)l * KV * d->cap * hd)};
        ok = df_norm(d, d->h, 0, B, D, 1, y->o_ln1, d->n, 0) && cc_matmul(y->acp, d->n, 0, d->kp, 0, B) &&
             cc_dfconv(d->n, d->kp, d->prm, d->a, &pre) &&
             cc_matmul(y->q, d->a, 0, d->qb, 0, B) && cc_matmul(y->k, d->a, 0, d->kb, 0, B) && cc_matmul(y->v, d->a, 0, d->vb, 0, B) &&
             df_norm(d, d->qb, 0, B, hd, H, y->o_qn, d->qb, 0) && df_norm(d, d->kb, 0, B, hd, KV, y->o_kn, d->kb, 0) &&
             df_rope(d->qb, d->cs, B, H, hd) && df_rope(d->kb, d->cs, B, KV, hd) && df_store(d, l, d->kb, d->vb, B, pos);
        if (d->causal) ok = ok && cc_attn(d->qb, d->kc, d->vc, d->att, d->qb, NULL, &at);
        else for (int r = 0; ok && r < B; r++) {   /* every row sees the whole block: one query at the block's end */
            CcAttn a1 = at; a1.q_off = r * qo; a1.o_off = r * qo;
            ok = cc_attn(d->qb, d->kc, d->vc, d->att, d->qb, NULL, &a1);
        }
        ok = ok && cc_matmul(y->o, d->att, 0, d->t, 0, B) && cc_dfconv(d->t, d->kp, d->prm, d->a, &post) &&
             cc_ew(d->h, d->h, d->a, NULL, NULL, &add);
        pre.b_off = (int)y->o_mcb; post.b_off = (int)y->o_mcb + 2 * D;
        ok = ok && df_norm(d, d->h, 0, B, D, 1, y->o_ln2, d->n, 0) && cc_matmul(y->mcp, d->n, 0, d->kp, 0, B) &&
             cc_dfconv(d->n, d->kp, d->prm, d->a, &pre) &&
             cc_matmul(y->gate, d->a, 0, d->g, 0, B) && cc_matmul(y->up, d->a, 0, d->u, 0, B) &&
             cc_ew(d->sw, d->g, d->u, NULL, NULL, &sw) && cc_matmul(y->down, d->sw, 0, d->t, 0, B) &&
             cc_dfconv(d->t, d->kp, d->prm, d->a, &post) && cc_ew(d->h, d->h, d->a, NULL, NULL, &add);
    }
    /* the masks' rows the verify can take (the block's every row went through the layers: each sees the
     * others): the drafter's norm, the target's lm_head, the candidates */
    const int R = kmax < B - 1 ? kmax : B - 1;
    ok = ok && df_norm(d, d->h, (size_t)D, R, D, 1, d->o_norm, d->fin, 0) && cc_matmul(d->lm, d->fin, 0, d->lg, 0, R);
    if (d->select) {
        CcDfTopk tk = {R, V, K, 0, V, 0, 2 * K, DF_TOPK_SPLIT};
        ok = ok && cc_dftopk(d->lg, d->tsc, d->top, &tk) && cc_copy(d->topd, 0, d->top, 0, (size_t)R * 2 * K) &&
             cc_matmul(d->hp, d->fin, 0, d->hg, 0, R) && cc_copy(d->hgd, 0, d->hg, 0, (size_t)R * d->rank);
    } else ok = ok && cc_pick_rows(d->lg, 0, d->pick, V, R);
    ok = ok && cc_submit(1);
    if (!ok) { d->on = 0; d->failed = 1; fprintf(stderr, "[dflash] a block failed on the device: drafts off\n"); return 0; }
    int k = kmax < R ? kmax : R;
    if (d->select) {
        const float *tp = (const float *)cc_ptr(d->topd), *hg = (const float *)cc_ptr(d->hgd);
        int prev = tok;
        for (int t = 0; t < k; t++) {
            const float *v = tp + (size_t)t * 2 * K, *gt = hg + (size_t)t * d->rank;
            const uint16_t *a = d->A + (size_t)prev * d->rank;
            float ag[DF_MAX_RANK];
            const int rk = d->rank;
            for (int j = 0; j < rk; j++) ag[j] = df_bf16(a[j]) * gt[j];
            int best = 0; float bs = -INFINITY, z = 0.f;
            for (int c = 0; c < K; c++) {
                int id; memcpy(&id, v + K + c, 4);
                if (id < 0 || id >= V) continue;
                const uint16_t *b = d->B + (size_t)id * d->rank;
                float s = v[c];
                for (int j = 0; j < rk; j++) s += ag[j] * df_bf16(b[j]);
                if (s > bs) { bs = s; best = c; }
            }
            for (int c = 0; c < K; c++) z += expf(v[c] - v[0]);
            int id; memcpy(&id, v + K + best, 4);
            out[t] = id; prob[t] = z > 0.f ? expf(v[best] - v[0]) / z : 0.f;
            prev = id;
        }
    } else {
        const char *pk = (const char *)cc_ptr(d->pick);
        for (int t = 0; t < k; t++) { int id; float p; memcpy(&id, pk + (size_t)t * 8, 4); memcpy(&p, pk + (size_t)t * 8 + 4, 4); out[t] = id; prob[t] = p; }
    }
    d->blocks++; d->ms += (now_s() - t0) * 1e3;
    return k;
}
static void df_report(const DfDrafter *d) {
    if (!d->on && !d->blocks) return;
    fprintf(stderr, "[dflash] %llu blocks (%.2f ms each), %llu context rows (%.3f ms each)\n", d->blocks,
            d->blocks ? d->ms / (double)d->blocks : 0.0, d->ctx_rows, d->ctx_rows ? d->ctx_ms / (double)d->ctx_rows : 0.0);
}
#endif
