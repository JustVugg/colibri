/* nemotron_vision.h -- RADIO ViT + pixel-shuffle + ReLU^2 MLP projector for
 * NemotronH_Omni_Reasoning_V3 (NVIDIA-Nemotron-3.5-Super-VL-120B-A12B).
 *
 * Included by nemotron.c after Model, NemoMat, and math helpers.
 * Mirrors modeling_radio.py and NemotronH_Omni_Reasoning_V3VisionProjector in
 * modeling_nemotron_h_omni.py:
 *   - patch_generator: Linear(3*P*P -> D, bias=False) over [H/P, W/P] patches,
 *     2D bilinear CPE position embedding (align_corners=False, square-then-crop),
 *     prepend num_cls_tokens + num_registers prefix tokens (10 on release)
 *   - encoder: depth ViT blocks (LayerNorm, full bidirectional multi-head attn
 *     with fused QKV + bias, optional LayerScale, LayerNorm, GELU MLP + bias)
 *   - strip prefix tokens -> optional vision_final_layernorm -> 2x2 pixel_shuffle
 *     (ps_version="v2") -> mlp1 (RMSNorm -> Linear -> ReLU^2 -> Linear)
 */

#ifndef COLIBRI_NEMOTRON_VISION_H
#define COLIBRI_NEMOTRON_VISION_H

typedef struct {
    float *norm1_w, *norm1_b;
    float *norm2_w, *norm2_b;
    float *qkv_b, *proj_b, *fc1_b, *fc2_b;
    float *ls1, *ls2;               /* optional LayerScale lambda1 [D] */
    NemoMat qkv, proj, fc1, fc2;    /* qkv:[3D,D], proj:[D,D], fc1:[I,D], fc2:[D,I] */
} NemoRadioBlock;

typedef struct {
    int depth, d, inter, heads, hd, patch, in_ch;
    int max_img_size, max_rows, max_cols;
    int num_cls, num_reg, num_skip;
    int out_hidden, proj_hidden;
    float ln_eps, mlp_eps, downsample;
    float norm_mean[3], norm_std[3];
    int ps_v1;

    NemoMat embedder;               /* [D, 3*P*P] */
    float *pos_embed;               /* [max_rows * max_cols, D] */
    float *cls_reg_token;           /* [num_skip, D] */
    NemoRadioBlock *blocks;         /* [depth] */
    float *final_ln_w, *final_ln_b; /* optional vision_final_layernorm [D] */
    float *mlp_norm_w;              /* mlp1.0 / norm [D * 4] */
    NemoMat mlp_fc1, mlp_fc2;       /* [proj_hidden, D*4], [out_hidden, proj_hidden] */
} NemoVision;

static void nemo_layernorm(float *out, const float *in, const float *w, const float *b,
                           int n, float eps) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)in[i];
    float mean = (float)(sum / (double)n);
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        float d = in[i] - mean;
        var += (double)d * (double)d;
    }
    float inv = 1.0f / sqrtf((float)(var / (double)n) + eps);
    for (int i = 0; i < n; i++)
        out[i] = (in[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
}

static inline float nemo_gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}

/* Bilinear resize of [src_h, src_w, C] -> [dst_h, dst_w, C], matching
 * torch.nn.functional.interpolate(mode="bilinear", align_corners=False). */
static void nemo_bilinear_2d(float *dst, int dst_h, int dst_w,
                             const float *src, int src_h, int src_w, int C) {
    float scale_y = (float)src_h / (float)dst_h;
    float scale_x = (float)src_w / (float)dst_w;
    for (int y = 0; y < dst_h; y++) {
        float fy = ((float)y + 0.5f) * scale_y - 0.5f;
        if (fy < 0.0f) fy = 0.0f;
        int y0 = (int)floorf(fy);
        int y1 = y0 + 1;
        float wy = fy - (float)y0;
        if (y0 >= src_h - 1) { y0 = src_h - 1; y1 = src_h - 1; wy = 0.0f; }
        for (int x = 0; x < dst_w; x++) {
            float fx = ((float)x + 0.5f) * scale_x - 0.5f;
            if (fx < 0.0f) fx = 0.0f;
            int x0 = (int)floorf(fx);
            int x1 = x0 + 1;
            float wx = fx - (float)x0;
            if (x0 >= src_w - 1) { x0 = src_w - 1; x1 = src_w - 1; wx = 0.0f; }
            const float *p00 = src + ((size_t)y0 * src_w + x0) * C;
            const float *p01 = src + ((size_t)y0 * src_w + x1) * C;
            const float *p10 = src + ((size_t)y1 * src_w + x0) * C;
            const float *p11 = src + ((size_t)y1 * src_w + x1) * C;
            float *out = dst + ((size_t)y * dst_w + x) * C;
            float w00 = (1.0f - wy) * (1.0f - wx);
            float w01 = (1.0f - wy) * wx;
            float w10 = wy * (1.0f - wx);
            float w11 = wy * wx;
            for (int c = 0; c < C; c++)
                out[c] = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
        }
    }
}

static int nemo_vis_find(shards *S, const char *a, const char *b, char *out, size_t cap) {
    if (a && st_has(S, a)) { snprintf(out, cap, "%s", a); return 1; }
    if (b && st_has(S, b)) { snprintf(out, cap, "%s", b); return 1; }
    return 0;
}

static NemoVision *nemo_vision_load(shards *S, jval *root, int llm_hidden, float llm_eps) {
    char tname[384];
    if (!nemo_vis_find(S, "vision_model.radio_model.model.patch_generator.embedder.weight",
                          "vision_model.embeddings.patch_projection.weight", tname, sizeof(tname)))
        return NULL;
    jval *vc = root ? json_get(root, "vision_config") : NULL;
    if (!vc || vc->t != J_OBJ) return NULL;

    NemoVision *v = (NemoVision *)xcalloc(1, sizeof(NemoVision), "nemo_vision");
    v->depth = (int)jnum(vc, "num_hidden_layers", 32);
    v->d = (int)jnum(vc, "hidden_size", 1280);
    double mlp_ratio = jnum(vc, "mlp_ratio", 4.0);
    v->inter = (int)(v->d * mlp_ratio + 0.5);
    v->heads = (int)jnum(vc, "num_attention_heads", 16);
    v->hd = v->d / v->heads;
    v->patch = (int)jnum(vc, "patch_size", 16);
    v->in_ch = (int)jnum(vc, "num_channels", 3);
    v->max_img_size = (int)jnum(vc, "max_img_size", 2048);
    v->max_rows = v->max_img_size / v->patch;
    v->max_cols = v->max_img_size / v->patch;
    v->num_cls = (int)jnum(vc, "num_cls_tokens", 3);
    v->num_reg = (int)jnum(vc, "num_registers", 7);
    v->num_skip = v->num_cls + v->num_reg;
    v->ln_eps = (float)jnum(vc, "layer_norm_eps", 1e-6);
    v->mlp_eps = llm_eps;
    v->downsample = (float)jnum(root, "downsample_ratio", 0.5);
    v->out_hidden = llm_hidden;
    v->proj_hidden = (int)jnum(root, "projector_hidden_size", 20480);
    const char *psv = jstr(root, "ps_version", "v2");
    v->ps_v1 = (psv && strcmp(psv, "v1") == 0);

    float def_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float def_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    for (int c = 0; c < 3; c++) { v->norm_mean[c] = def_mean[c]; v->norm_std[c] = def_std[c]; }
    jval *jm = json_get(root, "norm_mean"), *js = json_get(root, "norm_std");
    if (jm && jm->t == J_ARR && jm->len == 3)
        for (int c = 0; c < 3; c++) v->norm_mean[c] = (float)jm->kids[c]->num;
    if (js && js->t == J_ARR && js->len == 3)
        for (int c = 0; c < 3; c++) v->norm_std[c] = (float)js->kids[c]->num;

    int patch_in = v->in_ch * v->patch * v->patch;
    nemo_mat_load(&v->embedder, S, tname, v->d, patch_in);

    if (nemo_vis_find(S, "vision_model.radio_model.model.patch_generator.pos_embed",
                         "vision_model.embeddings.position_embedding", tname, sizeof(tname)))
        v->pos_embed = nemo_load_f32(S, tname, (size_t)v->max_rows * v->max_cols * v->d);
    if (nemo_vis_find(S, "vision_model.radio_model.model.patch_generator.cls_token.token",
                         "vision_model.embeddings.cls_register_token", tname, sizeof(tname)))
        v->cls_reg_token = nemo_load_f32(S, tname, (size_t)v->num_skip * v->d);

    v->blocks = (NemoRadioBlock *)xcalloc((size_t)v->depth, sizeof(NemoRadioBlock), "radio_blocks");
    for (int i = 0; i < v->depth; i++) {
        NemoRadioBlock *b = &v->blocks[i];
        char p1[256], p2[256];
        snprintf(p1, sizeof(p1), "vision_model.radio_model.model.blocks.%d.", i);
        snprintf(p2, sizeof(p2), "vision_model.encoder.layer.%d.", i);
        char a[320], bb[320];

        snprintf(a, sizeof(a), "%snorm1.weight", p1); snprintf(bb, sizeof(bb), "%snorm1.weight", p2);
        nemo_vis_find(S, a, bb, tname, sizeof(tname));
        b->norm1_w = nemo_load_f32(S, tname, v->d);
        snprintf(a, sizeof(a), "%snorm1.bias", p1); snprintf(bb, sizeof(bb), "%snorm1.bias", p2);
        if (nemo_vis_find(S, a, bb, tname, sizeof(tname))) b->norm1_b = nemo_load_f32(S, tname, v->d);

        snprintf(a, sizeof(a), "%snorm2.weight", p1); snprintf(bb, sizeof(bb), "%snorm2.weight", p2);
        nemo_vis_find(S, a, bb, tname, sizeof(tname));
        b->norm2_w = nemo_load_f32(S, tname, v->d);
        snprintf(a, sizeof(a), "%snorm2.bias", p1); snprintf(bb, sizeof(bb), "%snorm2.bias", p2);
        if (nemo_vis_find(S, a, bb, tname, sizeof(tname))) b->norm2_b = nemo_load_f32(S, tname, v->d);

        snprintf(a, sizeof(a), "%sattn.qkv.weight", p1);
        nemo_mat_load(&b->qkv, S, a, 3 * v->d, v->d);
        snprintf(a, sizeof(a), "%sattn.qkv.bias", p1);
        if (st_has(S, a)) b->qkv_b = nemo_load_f32(S, a, 3 * v->d);

        snprintf(a, sizeof(a), "%sattn.proj.weight", p1);
        nemo_mat_load(&b->proj, S, a, v->d, v->d);
        snprintf(a, sizeof(a), "%sattn.proj.bias", p1);
        if (st_has(S, a)) b->proj_b = nemo_load_f32(S, a, v->d);

        snprintf(a, sizeof(a), "%smlp.fc1.weight", p1);
        nemo_mat_load(&b->fc1, S, a, v->inter, v->d);
        snprintf(a, sizeof(a), "%smlp.fc1.bias", p1);
        if (st_has(S, a)) b->fc1_b = nemo_load_f32(S, a, v->inter);

        snprintf(a, sizeof(a), "%smlp.fc2.weight", p1);
        nemo_mat_load(&b->fc2, S, a, v->d, v->inter);
        snprintf(a, sizeof(a), "%smlp.fc2.bias", p1);
        if (st_has(S, a)) b->fc2_b = nemo_load_f32(S, a, v->d);

        snprintf(a, sizeof(a), "%sls1.gamma", p1); snprintf(bb, sizeof(bb), "%slayer_scale1.lambda1", p2);
        if (nemo_vis_find(S, a, bb, tname, sizeof(tname))) b->ls1 = nemo_load_f32(S, tname, v->d);
        snprintf(a, sizeof(a), "%sls2.gamma", p1); snprintf(bb, sizeof(bb), "%slayer_scale2.lambda1", p2);
        if (nemo_vis_find(S, a, bb, tname, sizeof(tname))) b->ls2 = nemo_load_f32(S, tname, v->d);
    }

    if (st_has(S, "vision_projector.vision_final_layernorm.weight")) {
        v->final_ln_w = nemo_load_f32(S, "vision_projector.vision_final_layernorm.weight", v->d);
        if (st_has(S, "vision_projector.vision_final_layernorm.bias"))
            v->final_ln_b = nemo_load_f32(S, "vision_projector.vision_final_layernorm.bias", v->d);
    }

    int ds_inv = (int)(1.0f / v->downsample + 0.5f);
    int mlp_in = v->d * ds_inv * ds_inv;
    nemo_vis_find(S, "mlp1.0.weight", "vision_projector.mlp1.norm.weight", tname, sizeof(tname));
    v->mlp_norm_w = nemo_load_f32(S, tname, mlp_in);
    nemo_vis_find(S, "mlp1.1.weight", "vision_projector.mlp1.linear1.weight", tname, sizeof(tname));
    nemo_mat_load(&v->mlp_fc1, S, tname, v->proj_hidden, mlp_in);
    nemo_vis_find(S, "mlp1.3.weight", "vision_projector.mlp1.linear2.weight", tname, sizeof(tname));
    nemo_mat_load(&v->mlp_fc2, S, tname, v->out_hidden, v->proj_hidden);

    fprintf(stderr, "[nemotron] RADIO vision tower loaded: %d blocks, dim=%d, heads=%d, patch=%d, mlp1=%d->%d->%d\n",
            v->depth, v->d, v->heads, v->patch, mlp_in, v->proj_hidden, v->out_hidden);
    return v;
}

static void nemo_vision_free(NemoVision *v) {
    if (!v) return;
    nemo_mat_free(&v->embedder);
    free(v->pos_embed);
    free(v->cls_reg_token);
    for (int i = 0; i < v->depth; i++) {
        NemoRadioBlock *b = &v->blocks[i];
        free(b->norm1_w); free(b->norm1_b);
        free(b->norm2_w); free(b->norm2_b);
        free(b->qkv_b); free(b->proj_b); free(b->fc1_b); free(b->fc2_b);
        free(b->ls1); free(b->ls2);
        nemo_mat_free(&b->qkv); nemo_mat_free(&b->proj);
        nemo_mat_free(&b->fc1); nemo_mat_free(&b->fc2);
    }
    free(v->blocks);
    free(v->final_ln_w); free(v->final_ln_b);
    free(v->mlp_norm_w);
    nemo_mat_free(&v->mlp_fc1); nemo_mat_free(&v->mlp_fc2);
    free(v);
}

/* Encode one normalized CHW f32 image [3, H, W] -> [out_tokens, llm_hidden].
 * H and W must be multiples of patch * 2. Returns number of output tokens. */
static int nemo_vision_encode(const NemoVision *v, const float *chw, int H, int W, float *out_embeds) {
    int P = v->patch, D = v->d;
    int gh = H / P, gw = W / P;
    int npatches = gh * gw;
    int ntok = v->num_skip + npatches;
    int patch_in = 3 * P * P;

    /* 1. Extract patches [npatches, 3*P*P] in (c, py, px) order */
    float *patches = (float *)xmalloc((size_t)npatches * patch_in * sizeof(float), "vis_patches");
    for (int gy = 0; gy < gh; gy++) {
        for (int gx = 0; gx < gw; gx++) {
            float *dst = patches + ((size_t)gy * gw + gx) * patch_in;
            int k = 0;
            for (int c = 0; c < 3; c++)
                for (int py = 0; py < P; py++)
                    for (int px = 0; px < P; px++)
                        dst[k++] = chw[((size_t)c * H + (gy * P + py)) * W + (gx * P + px)];
        }
    }

    /* 2. Patch projection + CPE 2D bilinear pos_embed + prepend cls/register tokens */
    float *x = (float *)xcalloc((size_t)ntok * D, sizeof(float), "vis_x");
    if (v->cls_reg_token)
        memcpy(x, v->cls_reg_token, (size_t)v->num_skip * D * sizeof(float));
    nemo_mat_matmul(x + (size_t)v->num_skip * D, &v->embedder, patches, npatches);
    free(patches);

    if (v->pos_embed) {
        int max_dim = gh > gw ? gh : gw;
        float *pos_sq = (float *)xmalloc((size_t)max_dim * max_dim * D * sizeof(float), "vis_pos_sq");
        nemo_bilinear_2d(pos_sq, max_dim, max_dim, v->pos_embed, v->max_rows, v->max_cols, D);
        for (int gy = 0; gy < gh; gy++) {
            for (int gx = 0; gx < gw; gx++) {
                float *dst = x + ((size_t)v->num_skip + (size_t)gy * gw + gx) * D;
                const float *pe = pos_sq + ((size_t)gy * max_dim + gx) * D;
                for (int c = 0; c < D; c++) dst[c] += pe[c];
            }
        }
        free(pos_sq);
    }

    /* 3. Run ViT blocks */
    float *xn = (float *)xmalloc((size_t)ntok * D * sizeof(float), "vis_xn");
    float *qkv = (float *)xmalloc((size_t)ntok * 3 * D * sizeof(float), "vis_qkv");
    float *attn_out = (float *)xmalloc((size_t)ntok * D * sizeof(float), "vis_attn");
    float *proj_out = (float *)xmalloc((size_t)ntok * D * sizeof(float), "vis_proj");
    float *mlp_h = (float *)xmalloc((size_t)ntok * v->inter * sizeof(float), "vis_mlph");
    float scale = 1.0f / sqrtf((float)v->hd);

    for (int li = 0; li < v->depth; li++) {
        const NemoRadioBlock *b = &v->blocks[li];
        for (int t = 0; t < ntok; t++)
            nemo_layernorm(xn + (size_t)t * D, x + (size_t)t * D, b->norm1_w, b->norm1_b, D, v->ln_eps);

        nemo_mat_matmul(qkv, &b->qkv, xn, ntok);
        if (b->qkv_b) {
            for (int t = 0; t < ntok; t++) {
                float *row = qkv + (size_t)t * 3 * D;
                for (int j = 0; j < 3 * D; j++) row[j] += b->qkv_b[j];
            }
        }

        /* Bidirectional multi-head attention over ntok tokens */
        int Hn = v->heads, hd = v->hd;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int h = 0; h < Hn; h++) {
            float *scores = (float *)malloc((size_t)ntok * sizeof(float));
            for (int ti = 0; ti < ntok; ti++) {
                const float *q = qkv + (size_t)ti * 3 * D + h * hd;
                float mx = -1e30f;
                for (int tj = 0; tj < ntok; tj++) {
                    const float *k = qkv + (size_t)tj * 3 * D + D + h * hd;
                    float s = 0.0f;
                    for (int d = 0; d < hd; d++) s += q[d] * k[d];
                    s *= scale;
                    scores[tj] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (int tj = 0; tj < ntok; tj++) {
                    scores[tj] = expf(scores[tj] - mx);
                    sum += scores[tj];
                }
                float inv_s = 1.0f / sum;
                float *o = attn_out + (size_t)ti * D + h * hd;
                memset(o, 0, (size_t)hd * sizeof(float));
                for (int tj = 0; tj < ntok; tj++) {
                    float w = scores[tj] * inv_s;
                    const float *vv = qkv + (size_t)tj * 3 * D + 2 * D + h * hd;
                    for (int d = 0; d < hd; d++) o[d] += w * vv[d];
                }
            }
            free(scores);
        }

        nemo_mat_matmul(proj_out, &b->proj, attn_out, ntok);
        for (int t = 0; t < ntok; t++) {
            float *xr = x + (size_t)t * D;
            const float *pr = proj_out + (size_t)t * D;
            for (int j = 0; j < D; j++) {
                float val = pr[j] + (b->proj_b ? b->proj_b[j] : 0.0f);
                if (b->ls1) val *= b->ls1[j];
                xr[j] += val;
            }
        }

        for (int t = 0; t < ntok; t++)
            nemo_layernorm(xn + (size_t)t * D, x + (size_t)t * D, b->norm2_w, b->norm2_b, D, v->ln_eps);
        nemo_mat_matmul(mlp_h, &b->fc1, xn, ntok);
        for (int t = 0; t < ntok; t++) {
            float *hr = mlp_h + (size_t)t * v->inter;
            for (int j = 0; j < v->inter; j++) {
                float val = hr[j] + (b->fc1_b ? b->fc1_b[j] : 0.0f);
                hr[j] = nemo_gelu_exact(val);
            }
        }
        nemo_mat_matmul(proj_out, &b->fc2, mlp_h, ntok);
        for (int t = 0; t < ntok; t++) {
            float *xr = x + (size_t)t * D;
            const float *pr = proj_out + (size_t)t * D;
            for (int j = 0; j < D; j++) {
                float val = pr[j] + (b->fc2_b ? b->fc2_b[j] : 0.0f);
                if (b->ls2) val *= b->ls2[j];
                xr[j] += val;
            }
        }
    }
    free(xn); free(qkv); free(attn_out); free(proj_out); free(mlp_h);

    /* 4. Strip prefix tokens + optional vision_final_layernorm */
    float *feat = x + (size_t)v->num_skip * D;
    if (v->final_ln_w) {
        for (int t = 0; t < npatches; t++)
            nemo_layernorm(feat + (size_t)t * D, feat + (size_t)t * D,
                           v->final_ln_w, v->final_ln_b, D, v->ln_eps);
    }

    /* 5. Pixel shuffle (scale_factor = 0.5 -> 2x2 spatial merge -> 4*D)
     * Mirrors modeling_nemotron_h_omni.py pixel_shuffle(v2):
     *   in [gh, gw, D] -> out [gh/2, gw/2, 4*D] where out[r, c] concatenates
     *   in[2r, 2c], in[2r, 2c+1], in[2r+1, 2c], in[2r+1, 2c+1] (row-major within 2x2).
     * Wait: check exact order from pixel_shuffle:
     *   x.view(n, w, h/2, 2c) -> permute(0,2,1,3) [n, h/2, w, 2c]
     *   -> view(n, h/2, w/2, 4c) -> permute(0,2,1,3) [n, w/2, h/2, 4c]
     *   So for output (i, j) where i in 0..gh/2-1, j in 0..gw/2-1:
     *     slot 0: (2*i,   2*j)
     *     slot 1: (2*i,   2*j+1)
     *     slot 2: (2*i+1, 2*j)
     *     slot 3: (2*i+1, 2*j+1)
     *   Notice the first view splits the LAST spatial dim (gw) by 2 (putting (2*i, 2*j) and (2*i, 2*j+1)
     *   together in 2c), then after permute the second view splits gh by 2 (putting row 2*i and 2*i+1 together).
     *   In ps_version="v2", the final permute restores [gh/2, gw/2, 4c] row-major! */
    int oh = gh / 2, ow = gw / 2;
    int out_toks = oh * ow;
    int mlp_in = 4 * D;
    float *ps = (float *)xmalloc((size_t)out_toks * mlp_in * sizeof(float), "vis_ps");
    for (int i = 0; i < oh; i++) {
        for (int j = 0; j < ow; j++) {
            float *dst = ps + ((size_t)i * ow + j) * mlp_in;
            memcpy(dst + 0 * D, feat + ((size_t)(2 * i)     * gw + (2 * j))     * D, (size_t)D * sizeof(float));
            memcpy(dst + 1 * D, feat + ((size_t)(2 * i)     * gw + (2 * j + 1)) * D, (size_t)D * sizeof(float));
            memcpy(dst + 2 * D, feat + ((size_t)(2 * i + 1) * gw + (2 * j))     * D, (size_t)D * sizeof(float));
            memcpy(dst + 3 * D, feat + ((size_t)(2 * i + 1) * gw + (2 * j + 1)) * D, (size_t)D * sizeof(float));
        }
    }
    free(x);

    /* 6. mlp1: RMSNorm -> linear1 -> ReLU^2 -> linear2 */
    for (int t = 0; t < out_toks; t++)
        nemo_rmsnorm(ps + (size_t)t * mlp_in, ps + (size_t)t * mlp_in, v->mlp_norm_w, mlp_in, v->mlp_eps);
    float *mh = (float *)xmalloc((size_t)out_toks * v->proj_hidden * sizeof(float), "vis_mlp1_h");
    nemo_mat_matmul(mh, &v->mlp_fc1, ps, out_toks);
    free(ps);
    for (size_t k = 0; k < (size_t)out_toks * v->proj_hidden; k++) {
        float val = mh[k];
        mh[k] = val > 0.0f ? val * val : 0.0f;
    }
    nemo_mat_matmul(out_embeds, &v->mlp_fc2, mh, out_toks);
    free(mh);
    return out_toks;
}

#endif /* COLIBRI_NEMOTRON_VISION_H */
