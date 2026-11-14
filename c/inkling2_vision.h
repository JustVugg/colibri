/* inkling2_vision.h: Inkling-2's vision tower inside inkling.c (transformers' Inkling2VisionModel,
 * the checkpoint's names model.visual.*).
 *
 * An image comes as Inkling2ImageProcessor gives it: padded to whole tiles of patch * merge
 * pixels (40 on Inkling-2-Small), each tile one language-model token, [tokens][tile][tile][3] in
 * [-1, 1], the tokens in raster order over the image's rows x cols. Each tile is merge x merge
 * patches; the tower's sequence for one image is a class token, then every token's patches in
 * that order (token, then the patch's row and column inside it):
 *
 *   - a patch: its pixels (row, column, channel) through patch_embedding (no bias), plus the
 *     learned position table (posemb_grid x posemb_grid) resampled to the image's patch grid
 *     bilinearly (align_corners false, the border clamped); the class token is class_embedding
 *     plus cls_position_embedding. Then ln_pre.
 *   - layers: x += ls_1 * attn(ln_1(x)), x += ls_2 * mlp(ln_2(x)); the attention is over the
 *     image's whole sequence, with a 2-D rope on q and k: a patch at (row, column) of the patch
 *     grid rotates by (row + 1, column + 1), the class token by (0, 0); per head the first half
 *     of the channel pairs turn with the column, the second half with the row, at
 *     theta^(-2i / (head_dim / 2)), each pair (x0, x1) -> (x0 c - x1 s, x1 c + x0 s). The MLP is
 *     c_fc, exact GELU, c_proj, all with biases.
 *   - the merger: each token's patches side by side (merge^2 * hidden) through proj_in, GELU and
 *     proj_out, no biases: one row of the language model's width per token.
 *
 * The rows replace the embeddings (after embed_norm) of the prompt's image tokens, in order. */
#ifndef INKLING2_VISION_H
#define INKLING2_VISION_H

typedef struct {
    float *ln1w, *ln1b, *ln2w, *ln2b, *ls1, *ls2;
    Wt q, k, v, o, fc1, fc2;
    float *qb, *kb, *vb, *ob, *fc1b, *fc2b;
} Ink2VisLayer;

typedef struct Ink2Vision {
    int D, L, H, hd, I, patch, merge, grid, proj, out_D, image_tok;
    float eps, theta;
    Wt patch_w;                      /* [D, patch * patch * 3] */
    float *pos;                      /* [grid * grid, D] */
    float *cls;                      /* class_embedding + cls_position_embedding [D] */
    float *pre_w, *pre_b;
    Ink2VisLayer *layer;
    Wt proj_in, proj_out;            /* [proj, merge^2 * D], [out_D, proj] */
} Ink2Vision;

static inline float ink2v_gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

static void ink2v_layernorm(float *out, const float *x, const float *w, const float *b, int D, float eps) {
    double mu = 0, var = 0;
    for (int i = 0; i < D; i++) mu += x[i];
    mu /= D;
    for (int i = 0; i < D; i++) { double d = x[i] - mu; var += d * d; }
    const float r = (float)(1.0 / sqrt(var / D + eps));
    for (int i = 0; i < D; i++) out[i] = (float)(x[i] - mu) * r * w[i] + b[i];
}

/* the tower when the snapshot has one (config.json's vision_config and model.visual.*), else NULL;
 * its matrices stay on the CPU (load_w without the device): a picture is read once a request,
 * and the card's memory goes to the expert tier */
static Ink2Vision *ink2_vision_load(Model *m, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1); if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, n, f) != (size_t)n) {} buf[n] = 0; fclose(f);
    char *arena = NULL; jval *root = json_parse(buf, &arena);
    jval *vc = root ? json_get(root, "vision_config") : NULL;
    if (!vc || !st_has(&m->S, "model.visual.backbone.embeddings.patch_embedding.weight")) { free(buf); free(arena); return NULL; }
    Ink2Vision *v = calloc(1, sizeof *v);
    if (!v) { fprintf(stderr, "OOM\n"); exit(1); }
    v->D = (int)jnum(vc, "hidden_size", 1024);
    v->L = (int)jnum(vc, "num_hidden_layers", 23);
    v->H = (int)jnum(vc, "num_attention_heads", 16);
    v->hd = v->D / v->H;
    v->I = (int)jnum(vc, "intermediate_size", v->D * jnum(vc, "mlp_ratio", 4.0));
    v->patch = (int)jnum(vc, "patch_size", jnum(vc, "native_patch_size", 20));
    v->merge = (int)jnum(vc, "spatial_merge_size", jnum(vc, "adapter_merge_size", 2));
    v->grid = (int)jnum(vc, "posemb_grid_size", 32);
    v->proj = (int)jnum(vc, "projection_hidden_dim", 4096);
    v->eps = (float)jnum(vc, "layer_norm_eps", 1e-5);
    jval *rp = json_get(vc, "rope_parameters");
    v->theta = (float)(rp ? jnum(rp, "rope_theta", 10000.0) : 10000.0);
    v->image_tok = (int)jnum(root, "image_token_id", 200054);
    v->out_D = m->c.hidden;
    free(buf); free(arena);
    if (v->hd % 4 || v->D % v->H) { fprintf(stderr, "inkling2 vision: head width %d is not a multiple of 4\n", v->hd); exit(1); }
    char nm[256];
    #define PV(suffix) (snprintf(nm, sizeof(nm), "model.visual." suffix), nm)
    v->patch_w = load_w(m, PV("backbone.embeddings.patch_embedding.weight"), 0);
    v->pos = load_t(m, PV("backbone.embeddings.position_embedding"));
    v->cls = load_t(m, PV("backbone.embeddings.class_embedding"));
    float *cp = load_t(m, PV("backbone.embeddings.cls_position_embedding"));
    for (int i = 0; i < v->D; i++) v->cls[i] += cp[i];
    free(cp);
    v->pre_w = load_t(m, PV("backbone.ln_pre.weight"));
    v->pre_b = load_t(m, PV("backbone.ln_pre.bias"));
    v->layer = calloc((size_t)v->L, sizeof *v->layer);
    if (!v->layer) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < v->L; i++) {
        Ink2VisLayer *l = &v->layer[i];
        #define PL(suffix) (snprintf(nm, sizeof(nm), "model.visual.backbone.layers.%d." suffix, i), nm)
        l->ln1w = load_t(m, PL("ln_1.weight")); l->ln1b = load_t(m, PL("ln_1.bias"));
        l->ln2w = load_t(m, PL("ln_2.weight")); l->ln2b = load_t(m, PL("ln_2.bias"));
        l->ls1 = load_t(m, PL("ls_1.gamma")); l->ls2 = load_t(m, PL("ls_2.gamma"));
        l->q = load_w(m, PL("self_attn.q_proj.weight"), 0); l->qb = load_t(m, PL("self_attn.q_proj.bias"));
        l->k = load_w(m, PL("self_attn.k_proj.weight"), 0); l->kb = load_t(m, PL("self_attn.k_proj.bias"));
        l->v = load_w(m, PL("self_attn.v_proj.weight"), 0); l->vb = load_t(m, PL("self_attn.v_proj.bias"));
        l->o = load_w(m, PL("self_attn.out_proj.weight"), 0); l->ob = load_t(m, PL("self_attn.out_proj.bias"));
        l->fc1 = load_w(m, PL("mlp.c_fc.weight"), 0); l->fc1b = load_t(m, PL("mlp.c_fc.bias"));
        l->fc2 = load_w(m, PL("mlp.c_proj.weight"), 0); l->fc2b = load_t(m, PL("mlp.c_proj.bias"));
        #undef PL
    }
    v->proj_in = load_w(m, PV("proj_in.weight"), 0);
    v->proj_out = load_w(m, PV("proj_out.weight"), 0);
    #undef PV
    fprintf(stderr, "[inkling2] vision tower: %d layers of %d, %d-pixel patches merged %dx%d (a %d-pixel tile a token), "
                    "image token %d\n", v->L, v->D, v->patch, v->merge, v->merge, v->patch * v->merge, v->image_tok);
    return v;
}

/* y[S][O] = x[S][I] W^T + b */
static void ink2v_linear(float *y, const float *x, Wt W, const float *b, int S, int I, int O) {
    matmul_w(y, x, W, S, I, O);
    if (b) for (int64_t s = 0; s < S; s++) for (int o = 0; o < O; o++) y[s * O + o] += b[o];
}

/* bilinear taps of one axis: source index of target i of n onto a side of `side` (align_corners
 * false, the border clamped), as transformers' _interpolation_axis_taps_weights */
static void ink2v_taps(int i, int n, int side, int *t0, int *t1, float *w0, float *w1) {
    const float src = ((float)i + 0.5f) * (float)side / (float)n - 0.5f;
    const float fl = floorf(src);
    const int a = (int)fl, b = a + 1;
    *w0 = fmaxf(0.f, 1.f - fabsf(src - fl));
    *w1 = fmaxf(0.f, 1.f - fabsf(src - fl - 1.f));
    *t0 = a < 0 ? 0 : a > side - 1 ? side - 1 : a;
    *t1 = b < 0 ? 0 : b > side - 1 ? side - 1 : b;
}

/* One image: pix [rows * cols][tile][tile][3] -> out [rows * cols][out_D]. */
static void ink2_vision_image(const Ink2Vision *v, const float *pix, int rows, int cols, float *out) {
    const int D = v->D, M = v->merge, P = v->patch, T = rows * cols, NP = T * M * M, S = NP + 1;
    const int tile = P * M, PIN = P * P * 3, hd = v->hd, H = v->H, gh = rows * M, gw = cols * M;
    float *patches = falloc((int64_t)NP * PIN), *x = falloc((int64_t)S * D);
    int *prow = malloc((size_t)S * sizeof(int)), *pcol = malloc((size_t)S * sizeof(int));
    if (!prow || !pcol) { fprintf(stderr, "OOM\n"); exit(1); }
    /* the patches in the tower's order: token t, then (mh, mw) inside it */
    for (int t = 0; t < T; t++)
        for (int mh = 0; mh < M; mh++) for (int mw = 0; mw < M; mw++) {
            const int p = (t * M + mh) * M + mw;
            float *dst = patches + (int64_t)p * PIN;
            for (int ph = 0; ph < P; ph++) for (int pw = 0; pw < P; pw++) for (int ch = 0; ch < 3; ch++)
                dst[(ph * P + pw) * 3 + ch] = pix[(((int64_t)t * tile + mh * P + ph) * tile + mw * P + pw) * 3 + ch];
            prow[1 + p] = (t / cols) * M + mh;
            pcol[1 + p] = (t % cols) * M + mw;
        }
    matmul_w(x + D, patches, v->patch_w, NP, PIN, D);
    free(patches);
    for (int p = 0; p < NP; p++) {   /* the position table resampled to the patch grid */
        int h0, h1, w0, w1; float a0, a1, b0, b1;
        ink2v_taps(prow[1 + p], gh, v->grid, &h0, &h1, &a0, &a1);
        ink2v_taps(pcol[1 + p], gw, v->grid, &w0, &w1, &b0, &b1);
        const float *e00 = v->pos + ((int64_t)h0 * v->grid + w0) * D, *e01 = v->pos + ((int64_t)h0 * v->grid + w1) * D;
        const float *e10 = v->pos + ((int64_t)h1 * v->grid + w0) * D, *e11 = v->pos + ((int64_t)h1 * v->grid + w1) * D;
        float *xr = x + (int64_t)(1 + p) * D;
        for (int i = 0; i < D; i++)
            xr[i] += a0 * (b0 * e00[i] + b1 * e01[i]) + a1 * (b0 * e10[i] + b1 * e11[i]);
    }
    memcpy(x, v->cls, (size_t)D * sizeof(float));
    prow[0] = pcol[0] = -1;   /* the class token rotates by (0, 0) */
    for (int s = 0; s < S; s++) ink2v_layernorm(x + (int64_t)s * D, x + (int64_t)s * D, v->pre_w, v->pre_b, D, v->eps);
    /* the rope's angles: per row s, hd / 2 pairs, the first hd / 4 by the column, the rest by the row */
    const int npair = hd / 2, q4 = hd / 4;
    float *rc = falloc((int64_t)S * npair), *rs = falloc((int64_t)S * npair);
    for (int s = 0; s < S; s++)
        for (int j = 0; j < npair; j++) {
            const int axis_pos = j < q4 ? pcol[s] + 1 : prow[s] + 1;
            const int i = j < q4 ? j : j - q4;
            const double ang = axis_pos * pow((double)v->theta, -2.0 * i / (hd / 2));
            rc[(int64_t)s * npair + j] = (float)cos(ang);
            rs[(int64_t)s * npair + j] = (float)sin(ang);
        }
    free(prow); free(pcol);
    float *h = falloc((int64_t)S * D), *q = falloc((int64_t)S * D), *k = falloc((int64_t)S * D), *vv = falloc((int64_t)S * D);
    float *ctx = falloc((int64_t)S * D), *mlp = falloc((int64_t)S * v->I);
    const float scale = 1.f / sqrtf((float)hd);
    for (int li = 0; li < v->L; li++) {
        const Ink2VisLayer *l = &v->layer[li];
        for (int s = 0; s < S; s++) ink2v_layernorm(h + (int64_t)s * D, x + (int64_t)s * D, l->ln1w, l->ln1b, D, v->eps);
        ink2v_linear(q, h, l->q, l->qb, S, D, D);
        ink2v_linear(k, h, l->k, l->kb, S, D, D);
        ink2v_linear(vv, h, l->v, l->vb, S, D, D);
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < S; s++)
            for (int hh = 0; hh < H; hh++)
                for (int j = 0; j < npair; j++) {
                    const float c = rc[(int64_t)s * npair + j], sn = rs[(int64_t)s * npair + j];
                    float *qp = q + (int64_t)s * D + hh * hd + 2 * j, *kp = k + (int64_t)s * D + hh * hd + 2 * j;
                    const float q0 = qp[0], q1 = qp[1], k0 = kp[0], k1 = kp[1];
                    qp[0] = q0 * c - q1 * sn; qp[1] = q1 * c + q0 * sn;
                    kp[0] = k0 * c - k1 * sn; kp[1] = k1 * c + k0 * sn;
                }
        #pragma omp parallel
        {
            float *sc = malloc((size_t)S * sizeof(float));
            if (!sc) { fprintf(stderr, "OOM\n"); exit(1); }
            #pragma omp for collapse(2) schedule(static)
            for (int hh = 0; hh < H; hh++)
                for (int s = 0; s < S; s++) {
                    const float *qv = q + (int64_t)s * D + hh * hd;
                    float mx = -1e30f;
                    for (int t = 0; t < S; t++) {
                        sc[t] = dot_f32_lanes(qv, k + (int64_t)t * D + hh * hd, hd) * scale;
                        if (sc[t] > mx) mx = sc[t];
                    }
                    float den = 0.f;
                    for (int t = 0; t < S; t++) { sc[t] = expf(sc[t] - mx); den += sc[t]; }
                    float *cx = ctx + (int64_t)s * D + hh * hd;
                    for (int d = 0; d < hd; d++) cx[d] = 0.f;
                    for (int t = 0; t < S; t++) {
                        const float a = sc[t] / den, *vr = vv + (int64_t)t * D + hh * hd;
                        for (int d = 0; d < hd; d++) cx[d] += a * vr[d];
                    }
                }
            free(sc);
        }
        ink2v_linear(h, ctx, l->o, l->ob, S, D, D);
        for (int64_t s = 0; s < S; s++) for (int i = 0; i < D; i++) x[s * D + i] += l->ls1[i] * h[s * D + i];
        for (int s = 0; s < S; s++) ink2v_layernorm(h + (int64_t)s * D, x + (int64_t)s * D, l->ln2w, l->ln2b, D, v->eps);
        ink2v_linear(mlp, h, l->fc1, l->fc1b, S, D, v->I);
        for (int64_t j = 0; j < (int64_t)S * v->I; j++) mlp[j] = ink2v_gelu(mlp[j]);
        ink2v_linear(h, mlp, l->fc2, l->fc2b, S, v->I, D);
        for (int64_t s = 0; s < S; s++) for (int i = 0; i < D; i++) x[s * D + i] += l->ls2[i] * h[s * D + i];
    }
    free(q); free(k); free(vv); free(ctx); free(mlp); free(rc); free(rs);
    /* the merger: a token's M * M patches side by side, the class token left out */
    float *mid = falloc((int64_t)T * v->proj);
    matmul_w(mid, x + D, v->proj_in, T, M * M * D, v->proj);
    for (int64_t j = 0; j < (int64_t)T * v->proj; j++) mid[j] = ink2v_gelu(mid[j]);
    matmul_w(out, mid, v->proj_out, T, v->proj, v->out_D);
    free(mid); free(h); free(x);
}

#endif /* INKLING2_VISION_H */
