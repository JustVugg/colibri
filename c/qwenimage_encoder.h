/* CPU f32 encoder for one Qwen-Image-2.1 image. Reuses the decoder kernels
 * and loads encoder.* and quant_conv.* weights.
 *
 * Reference: Diffusers AutoencoderKLQwenImage21 (Apache-2.0), commit
 * 80c7ed262aeffbeb43ef13ae04baeb9b84515a69. Like QwenImage21Pipeline, use
 * the posterior mode, not a sample, then (mode - latents_mean) / latents_std.
 * This primitive does not implement the vision encoder or reference DiT path.
 */
#ifndef COLIBRI_QWENIMAGE_ENCODER_H
#define COLIBRI_QWENIMAGE_ENCODER_H
#include "qwenimage_vae.h"
#include <limits.h>

typedef struct {
    int cin, cout, down, ft, nres;
    QivRes res[QIV_MAX_RES];
    QivConv downc;
} QieBlock;

typedef struct {
    int z_dim, in_ch, scale, nblk;
    float *mean, *std;
    QivConv conv_in, conv_out, quant;
    QieBlock blk[QIV_MAX_BLOCKS];
    QivRes mid[2];
    float *attn_g, *g_out;
    QivConv qkv, proj;
    size_t wbytes;
} QiVaeEncoder;

static void qie_free(QiVaeEncoder *v) {
    if (!v) return;
    free(v->mean);
    free(v->std);
    free(v->attn_g);
    free(v->g_out);
    qiv_free_conv(&v->conv_in);
    qiv_free_conv(&v->conv_out);
    qiv_free_conv(&v->quant);
    qiv_free_conv(&v->qkv);
    qiv_free_conv(&v->proj);
    for (int i = 0; i < 2; i++) qiv_free_res(&v->mid[i]);
    for (int i = 0; i < v->nblk; i++) {
        for (int j = 0; j < v->blk[i].nres; j++) qiv_free_res(&v->blk[i].res[j]);
        qiv_free_conv(&v->blk[i].downc);
    }
    free(v);
}

static int qie_config_int(jval *cfg, const char *name, int fallback) {
    jval *v = json_get(cfg, name);
    if (!v) return fallback;
    return v->t == J_NUM && isfinite(v->num) && v->num >= 1 && v->num <= 4096 && floor(v->num) == v->num ? (int)v->num
                                                                                                         : -1;
}

static QiVaeEncoder *qie_load(const char *dir) {
    QiVaeEncoder *v = calloc(1, sizeof *v);
    char path[4096], err[512] = "";
    char *text = NULL;
    jval *cfg = NULL;
    shards s;
    int opened = 0;
    if (!v) return NULL;
    if (!dir || snprintf(path, sizeof path, "%s/config.json", dir) >= (int)sizeof path) {
        snprintf(err, sizeof err, "invalid checkpoint path");
        goto fail;
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, sizeof err, "cannot open config.json");
        goto fail;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    if (n <= 0 || n > (1 << 20) || !(text = malloc((size_t)n + 1))) {
        fclose(f);
        snprintf(err, sizeof err, "invalid config size or out of memory");
        goto fail;
    }
    if (fread(text, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        snprintf(err, sizeof err, "cannot read config");
        goto fail;
    }
    fclose(f);
    text[n] = 0;
    cfg = json_parse_checked(text);
    if (!cfg || cfg->t != J_OBJ) {
        snprintf(err, sizeof err, "config is not an object");
        goto fail;
    }
    v->z_dim = qie_config_int(cfg, "z_dim", -1);
    v->in_ch = qie_config_int(cfg, "in_channels", 3);
    int base = qie_config_int(cfg, "base_dim", -1), nres = qie_config_int(cfg, "num_res_blocks", 2);
    jval *mult = json_get(cfg, "dim_mult"), *temporal = json_get(cfg, "temperal_downsample");
    jval *residual = json_get(cfg, "is_residual"), *patch = json_get(cfg, "patch_size");
    jval *mean = json_get(cfg, "latents_mean"), *std = json_get(cfg, "latents_std");
    if (v->z_dim <= 0 || (v->in_ch != 3 && v->in_ch != 4) || base <= 0 || nres < 1 || nres > QIV_MAX_RES || !mult ||
        mult->t != J_ARR || mult->len < 1 || mult->len > QIV_MAX_BLOCKS || !temporal || temporal->t != J_ARR ||
        temporal->len != mult->len - 1) {
        snprintf(err, sizeof err, "unsupported encoder dimensions");
        goto fail;
    }
    if (!residual || residual->t != J_BOOL || !residual->boolean || (patch && patch->t != J_NULL)) {
        snprintf(err, sizeof err, "encoder requires is_residual=true and patch_size=null");
        goto fail;
    }
    if (!mean || mean->t != J_ARR || mean->len != v->z_dim || !std || std->t != J_ARR || std->len != v->z_dim) {
        snprintf(err, sizeof err, "latents_mean/latents_std must have z_dim entries");
        goto fail;
    }
    v->mean = qiv_alloc(v->z_dim);
    v->std = qiv_alloc(v->z_dim);
    if (!v->mean || !v->std) goto fail;
    for (int i = 0; i < v->z_dim; i++) {
        if (mean->kids[i]->t != J_NUM || std->kids[i]->t != J_NUM || !isfinite(mean->kids[i]->num) ||
            !isfinite(std->kids[i]->num) || std->kids[i]->num <= 0) {
            snprintf(err, sizeof err, "invalid latent normalization");
            goto fail;
        }
        v->mean[i] = (float)mean->kids[i]->num;
        v->std[i] = (float)std->kids[i]->num;
        if (!isfinite(v->mean[i]) || !isfinite(v->std[i]) || v->std[i] <= 0) {
            snprintf(err, sizeof err, "latent normalization is not representable in f32");
            goto fail;
        }
    }
    v->nblk = mult->len;
    v->scale = 1;
    int cin = base;
    for (int i = 0; i < v->nblk; i++) {
        if (mult->kids[i]->t != J_NUM || !isfinite(mult->kids[i]->num) || mult->kids[i]->num < 1 ||
            mult->kids[i]->num > 64 || floor(mult->kids[i]->num) != mult->kids[i]->num ||
            (i + 1 < v->nblk && temporal->kids[i]->t != J_BOOL)) {
            snprintf(err, sizeof err, "invalid dim_mult or temperal_downsample");
            goto fail;
        }
        QieBlock *b = &v->blk[i];
        b->cin = cin;
        b->cout = base * (int)mult->kids[i]->num;
        b->down = i + 1 < v->nblk;
        b->ft = b->down && temporal->kids[i]->boolean ? 2 : 1;
        b->nres = nres;
        if ((b->cin * b->ft * (b->down ? 4 : 1)) % b->cout) {
            snprintf(err, sizeof err, "block %d has incompatible AvgDown3D channels", i);
            goto fail;
        }
        v->scale *= b->down ? 2 : 1;
        cin = b->cout;
    }
    st_init(&s, dir);
    opened = 1;
    if (!s.n) {
        snprintf(err, sizeof err, "no safetensors found");
        goto fail;
    }
    if (qiv_load_conv(&s, &v->wbytes, "encoder.conv_in", v->in_ch, base, 3, &v->conv_in, err, sizeof err)) goto fail;
    for (int i = 0; i < v->nblk; i++) {
        QieBlock *b = &v->blk[i];
        char p[160];
        for (int j = 0; j < b->nres; j++) {
            snprintf(p, sizeof p, "encoder.down_blocks.%d.resnets.%d", i, j);
            if (qiv_load_res(&s, &v->wbytes, p, j ? b->cout : b->cin, b->cout, &b->res[j], err, sizeof err)) goto fail;
        }
        if (b->down) {
            snprintf(p, sizeof p, "encoder.down_blocks.%d.downsampler.resample.1", i);
            if (qiv_load_conv(&s, &v->wbytes, p, b->cout, b->cout, 3, &b->downc, err, sizeof err)) goto fail;
        }
    }
    for (int i = 0; i < 2; i++) {
        char p[128];
        snprintf(p, sizeof p, "encoder.mid_block.resnets.%d", i);
        if (qiv_load_res(&s, &v->wbytes, p, cin, cin, &v->mid[i], err, sizeof err)) goto fail;
    }
    v->attn_g = qiv_read(&s, "encoder.mid_block.attentions.0.norm.gamma", cin, err, sizeof err);
    v->g_out = qiv_read(&s, "encoder.norm_out.gamma", cin, err, sizeof err);
    if (!v->attn_g || !v->g_out) goto fail;
    v->wbytes += (size_t)2 * cin * sizeof(float);
    if (qiv_load_conv(&s, &v->wbytes, "encoder.mid_block.attentions.0.to_qkv", cin, 3 * cin, 1, &v->qkv, err,
                      sizeof err) ||
        qiv_load_conv(&s, &v->wbytes, "encoder.mid_block.attentions.0.proj", cin, cin, 1, &v->proj, err, sizeof err) ||
        qiv_load_conv(&s, &v->wbytes, "encoder.conv_out", cin, 2 * v->z_dim, 3, &v->conv_out, err, sizeof err) ||
        qiv_load_conv(&s, &v->wbytes, "quant_conv", 2 * v->z_dim, 2 * v->z_dim, 1, &v->quant, err, sizeof err))
        goto fail;
    st_destroy(&s);
    json_free(cfg);
    free(text);
    return v;
fail:
    fprintf(stderr, "qwenimage encoder: %s\n", err[0] ? err : "out of memory");
    if (opened) st_destroy(&s);
    json_free(cfg);
    free(text);
    qie_free(v);
    return NULL;
}

/* Stride-2 3x3 convolution with bottom/right padding: ZeroPad2d((0,1,0,1)).
 * Bound the im2col buffer by qiv_col_floats, with a minimum of one row. */
static int qie_downsample(const QivConv *c, const float *x, int H, int W, float *y) {
    int Ho = H / 2, Wo = W / 2, K = c->cin * 9;
    size_t row = (size_t)Wo * K;
    int band = (int)(qiv_col_floats / row);
    if (band < 1) band = 1;
    if (band > Ho) band = Ho;
    float *col = qiv_alloc((size_t)band * row);
    if (!col) return -1;
    for (int y0 = 0; y0 < Ho; y0 += band) {
        int nb = Ho - y0 < band ? Ho - y0 : band;
#pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < nb; r++)
            for (int u = 0; u < Wo; u++)
                for (int kx = 0; kx < 3; kx++)
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = 2 * (y0 + r) + ky, ix = 2 * u + kx;
                        float *dst = col + ((size_t)r * Wo + u) * K + (kx * 3 + ky) * c->cin;
                        if (iy < H && ix < W) memcpy(dst, x + ((size_t)iy * W + ix) * c->cin, sizeof(float) * c->cin);
                        else memset(dst, 0, sizeof(float) * c->cin);
                    }
        qi_gemm(y + (size_t)y0 * Wo * c->cout, col, nb * Wo, &c->m, c->b);
    }
    free(col);
    return 0;
}

/* The single image occupies the last temporal slot after causal zero padding.
 * Count the padded zeros in AvgDown3D's denominator, including the first chunk. */
static void qie_avg_add(const QieBlock *b, const float *x, int H, int W, float *y) {
    int fs = b->down ? 2 : 1, factor = b->ft * fs * fs;
    int group = b->cin * factor / b->cout, Ho = H / fs, Wo = W / fs;
#pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < Ho; r++)
        for (int u = 0; u < Wo; u++)
            for (int o = 0; o < b->cout; o++) {
                float sum = 0;
                for (int j = 0; j < group; j++) {
                    int index = o * group + j, c = index / factor, sub = index % factor;
                    if (sub / (fs * fs) != b->ft - 1) continue;
                    int iy = r * fs + sub / fs % fs, ix = u * fs + sub % fs;
                    sum += x[((size_t)iy * W + ix) * b->cin + c];
                }
                y[((size_t)r * Wo + u) * b->cout + o] += sum / group;
            }
}

static int qie_valid_size(const QiVaeEncoder *v, int H, int W) {
    /* GEMMs take int row counts; reject dimensions before allocating or reading. */
    return v && v->scale > 0 && v->conv_in.cout > 0 && H > 0 && W > 0 && H % v->scale == 0 && W % v->scale == 0 &&
           (int64_t)H * W <= INT_MAX && (size_t)H * W <= (SIZE_MAX - 64) / (sizeof(float) * (size_t)v->conv_in.cout);
}

/* Input: normalized float pixels [H][W][in_ch], no resizing or alpha compositing.
 * Output: normalized posterior mode [H/scale][W/scale][z_dim], token-major.
 * No sampling, tiling or multi-frame support. The caller owns both arrays. */
static int qie_encode_f32(const QiVaeEncoder *v, const float *pixels, int H, int W, float *z) {
    if (!pixels || !z || !qie_valid_size(v, H, W)) return -1;
    size_t HW = (size_t)H * W;
    float *x = qiv_alloc(HW * v->conv_in.cout), *a = NULL, *moments = NULL;
    if (!x || qiv_conv3(&v->conv_in, (float *)pixels, H, W, NULL, 0, x, NULL)) goto fail;
    for (int i = 0; i < v->nblk; i++) {
        const QieBlock *b = &v->blk[i];
        float *source = x, *cur = x, *next;
        x = NULL;
        for (int j = 0; j < b->nres; j++) {
            next = qiv_resnet(&b->res[j], cur, H, W, j == 0);
            if (!next) {
                if (cur != source) free(cur);
                free(source);
                goto fail;
            }
            cur = next;
        }
        if (b->down) {
            next = qiv_alloc(HW / 4 * b->cout);
            if (!next || qie_downsample(&b->downc, cur, H, W, next)) {
                free(next);
                free(cur);
                free(source);
                goto fail;
            }
            free(cur);
            cur = next;
        }
        qie_avg_add(b, source, H, W, cur);
        free(source);
        x = cur;
        if (b->down) {
            H /= 2;
            W /= 2;
            HW /= 4;
        }
    }
    float *next = qiv_resnet(&v->mid[0], x, H, W, 0);
    if (!next) goto fail;
    x = next;
    if (qiv_attention_core(v->attn_g, &v->qkv, &v->proj, x, (int)HW)) goto fail;
    next = qiv_resnet(&v->mid[1], x, H, W, 0);
    if (!next) goto fail;
    x = next;
    a = qiv_alloc(HW * 2 * v->z_dim);
    moments = qiv_alloc(HW * 2 * v->z_dim);
    if (!a || !moments || qiv_conv3(&v->conv_out, x, H, W, v->g_out, 0, a, NULL)) goto fail;
    qi_gemm(moments, a, (int)HW, &v->quant.m, v->quant.b);
    /* quant_conv's first z_dim channels are the mean; the log variance is not
     * needed for the deterministic posterior mode used by reference editing. */
    for (size_t p = 0; p < HW; p++)
        for (int c = 0; c < v->z_dim; c++)
            z[p * v->z_dim + c] = (moments[p * 2 * v->z_dim + c] - v->mean[c]) / v->std[c];
    free(x);
    free(a);
    free(moments);
    return 0;
fail:
    free(x);
    free(a);
    free(moments);
    return -1;
}

/* Add opaque alpha to RGB input for an RGBA encoder.
 * Preserve supplied alpha without compositing. */
static int qie_encode(const QiVaeEncoder *v, const uint8_t *pixels, int H, int W, int channels, float *z) {
    if (!pixels || !z || !qie_valid_size(v, H, W) || (channels != v->in_ch && !(channels == 3 && v->in_ch == 4)))
        return -1;
    size_t HW = (size_t)H * W;
    float *x = qiv_alloc(HW * v->in_ch);
    if (!x) return -1;
    for (size_t p = 0; p < HW; p++)
        for (int c = 0; c < v->in_ch; c++)
            x[p * v->in_ch + c] = c < channels ? pixels[p * channels + c] / 127.5f - 1.f : 1.f;
    int rc = qie_encode_f32(v, x, H, W, z);
    free(x);
    return rc;
}
#endif
