/* qwenimage_vae_vk.h -- the VAE decoder of qwenimage_vae.h on the Vulkan chain
 * (vk_chain.h). Included by qwenimage.c in a COLI_VULKAN build after both; it runs when
 * the transformer's chain does (COLI_VK_CHAIN), and the CPU's qiv_decode otherwise or
 * when anything here fails.
 *
 * The same arithmetic as qiv_decode, the maps channel-last on the device from conv_in to
 * conv_out:
 *   - a 3x3 convolution is chain_vae.comp's taps of a band of output rows (zero padding,
 *     the nearest 2x upsample of an up block folded into the read) and one GEMM over
 *     them with the f32 weights qwenimage_vae.h already permuted to [cout][kx][ky][cin],
 *     then its bias (vkc_enc_bias);
 *   - the RMS norm + SiLU in front of a residual conv is an L2 norm times sqrt(C) by its
 *     gamma (chain_norm.comp) and a SiLU (chain_ew.comp's HC_LOW with fc 1);
 *   - a 1x1 shortcut conv is a GEMM and its bias; the DupUp3D shortcut of an up block is
 *     chain_vae.comp's gather of the block's input channels, added;
 *   - the mid block's single-head attention over every pixel (384 channels a head, past
 *     the chain's attention ops) runs on the CPU between the two mid resnets: the map
 *     comes down and goes back up (6 MB at 1024x1024);
 *   - z * std + mean and post_quant_conv (a 1x1 conv on the latent grid) run on the CPU
 *     before the upload, the clamp and the uint8 rounding (qiv_post) after the read-back.
 * The device sums in other orders than the CPU (the norms in float trees, the GEMMs by
 * tiles), so the pixels agree with the CPU's to rounding. */
#include "vk_chain.h"

#define QVV_TAPS ((size_t)64 << 20)   /* floats of a band's taps (256 MB) */

static struct {
    int ok, failed;
    VkcBuf *prm;                       /* the device convs' biases and the norms' gammas */
    VkcBuf *dup[QIV_MAX_BLOCKS];       /* the up blocks' DupUp3D tables, int32 [cout][4] */
    VkcBuf *q;                         /* a band's taps */
    unsigned long long decodes;
} g_qvv;

static void qvv_off(const char *why) {
    if (!g_qvv.failed) fprintf(stderr, "[VK] qwenimage vae: %s; the VAE decodes on the CPU\n", why);
    g_qvv.failed = 1;
}

/* every conv the device runs, in the order their parameters are laid out */
static int qvv_convs(QiVae *v, QivConv **cv) {
    int n = 0;
    cv[n++] = &v->conv_in;
    for (int m = 0; m < 2; m++) { cv[n++] = &v->mid[m].c1; cv[n++] = &v->mid[m].c2; if (v->mid[m].cin != v->mid[m].cout) cv[n++] = &v->mid[m].sc; }
    for (int i = 0; i < v->nblk; i++) {
        QivBlock *b = &v->blk[i];
        for (int j = 0; j < b->nres; j++) {
            cv[n++] = &b->res[j].c1; cv[n++] = &b->res[j].c2;
            if (b->res[j].cin != b->res[j].cout) cv[n++] = &b->res[j].sc;
        }
        if (b->up) cv[n++] = &b->upc;
    }
    cv[n++] = &v->conv_out;
    return n;
}

static int qvv_setup(QiVae *v) {
    QivConv *cv[2 + 3 * 2 + QIV_MAX_BLOCKS * (3 * QIV_MAX_RES + 1)];
    int nc = qvv_convs(v, cv);
    size_t n = 0;
    for (int k = 0; k < nc; k++) { cv[k]->bo = (int)n; n += cv[k]->cout; }
    for (int m = 0; m < 2; m++) { v->mid[m].g1o = (int)n; n += v->mid[m].cin; v->mid[m].g2o = (int)n; n += v->mid[m].cout; }
    for (int i = 0; i < v->nblk; i++)
        for (int j = 0; j < v->blk[i].nres; j++) {
            QivRes *r = &v->blk[i].res[j];
            r->g1o = (int)n; n += r->cin; r->g2o = (int)n; n += r->cout;
        }
    v->g_outo = (int)n; n += v->conv_out.cin;
    float *a = malloc(n * sizeof(float));
    if (!a) return 0;
    for (int k = 0; k < nc; k++) memcpy(a + cv[k]->bo, cv[k]->b, (size_t)cv[k]->cout * sizeof(float));
    for (int m = 0; m < 2; m++) {
        memcpy(a + v->mid[m].g1o, v->mid[m].g1, (size_t)v->mid[m].cin * sizeof(float));
        memcpy(a + v->mid[m].g2o, v->mid[m].g2, (size_t)v->mid[m].cout * sizeof(float));
    }
    for (int i = 0; i < v->nblk; i++)
        for (int j = 0; j < v->blk[i].nres; j++) {
            QivRes *r = &v->blk[i].res[j];
            memcpy(a + r->g1o, r->g1, (size_t)r->cin * sizeof(float));
            memcpy(a + r->g2o, r->g2, (size_t)r->cout * sizeof(float));
        }
    memcpy(a + v->g_outo, v->g_out, (size_t)v->conv_out.cin * sizeof(float));
    g_qvv.prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = g_qvv.prm && vkc_begin() && vkc_write(g_qvv.prm, 0, a, n * sizeof(float));
    free(a);
    for (int i = 0; ok && i < v->nblk; i++)
        if (v->blk[i].up)
            ok = (g_qvv.dup[i] = vkc_buf((size_t)v->blk[i].cout * 4 * sizeof(int), VKC_DEV)) &&
                 vkc_write(g_qvv.dup[i], 0, v->blk[i].dup, (size_t)v->blk[i].cout * 4 * sizeof(int));
    ok = ok && vkc_submit(1);
    for (int k = 0; ok && k < nc; k++)
        ok = coli_vk_tensor_ensure((ColiVkTensor **)&cv[k]->vk, cv[k]->w, NULL, 10, cv[k]->cin * cv[k]->k * cv[k]->k,
                                   cv[k]->cout, 0);
    if (!ok) { qvv_off(vkc_lost() ? "the device was lost" : "the device refused its weights"); return 0; }
    g_qvv.ok = 1;
    return 1;
}

static int qvv_bias(VkcBuf *y, size_t yo, int rows, const QivConv *c) {
    VkcEncBias b = {1, rows * c->cout, c->cout, (int)yo, c->cout, c->bo, 0};
    return vkc_enc_bias(y, g_qvv.prm, &b);
}
/* t = SiLU(RMS norm of x by the gamma at go): x * sqrt(C) * gamma / |x|, then SiLU */
static int qvv_pre(VkcBuf *x, VkcBuf *t, size_t HW, int C, int go) {
    VkcNorm nn = {(int)HW, C, 1, 0, C, 0, 0, C, 0, go, 0, VKC_NORM_L2, 1e-24f, sqrtf((float)C)};
    VkcEw si = {VKC_EW_HC_LOW, (int)(HW * C), C, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
    return vkc_norm(x, g_qvv.prm, t, &nn) && vkc_ew(t, t, NULL, NULL, NULL, &si);
}
/* y = conv3x3(x) + bias over an Ho x Wo map (x H x W, upsampled 2x first with up), in
 * bands of output rows; a submission a band */
static int qvv_conv3(const QivConv *c, VkcBuf *x, int H, int W, int up, VkcBuf *y) {
    int C = c->cin, Co = c->cout, Ho = H << up, Wo = W << up;
    size_t per_row = (size_t)Wo * 9 * C;
    int band = (int)(QVV_TAPS / per_row);
    if (band > 65535 / Wo) band = 65535 / Wo;   /* vkc_matmul's rows a call */
    if (band < 1) band = 1;
    if (band > Ho) band = Ho;
    if (!vkc_reserve(&g_qvv.q, (size_t)band * per_row * sizeof(float), VKC_DEV)) return 0;
    int ok = 1;
    for (int y0 = 0; ok && y0 < Ho; y0 += band) {
        int nb = Ho - y0 < band ? Ho - y0 : band;
        VkcVae p = {0, C, H, W, up, Ho, Wo, y0, (int)((size_t)nb * per_row), 0, 0, Co, 0};
        ok = vkc_vae(x, g_qvv.q, NULL, &p) &&
             vkc_matmul((ColiVkTensor *)c->vk, g_qvv.q, 0, y, (size_t)y0 * Wo * Co, nb * Wo) &&
             qvv_bias(y, (size_t)y0 * Wo * Co, nb * Wo, c) && vkc_submit(0) && vkc_begin();
    }
    return ok;
}
static VkcBuf *qvv_map(size_t floats) { return vkc_buf(floats * sizeof(float), VKC_DEV); }
/* one residual block; x is kept (the caller frees it), the result is new */
static VkcBuf *qvv_resnet(const QivRes *r, VkcBuf *x, int H, int W) {
    size_t HW = (size_t)H * W;
    VkcBuf *pa = qvv_map(HW * r->cin), *t = qvv_map(HW * r->cout), *pb = qvv_map(HW * r->cout), *y = qvv_map(HW * r->cout);
    VkcBuf *s = r->cin != r->cout ? qvv_map(HW * r->cout) : NULL;
    int ok = pa && t && pb && y && (r->cin == r->cout || s) &&
             qvv_pre(x, pa, HW, r->cin, r->g1o) && qvv_conv3(&r->c1, pa, H, W, 0, t) &&
             qvv_pre(t, pb, HW, r->cout, r->g2o) && qvv_conv3(&r->c2, pb, H, W, 0, y);
    for (size_t p0 = 0; ok && s && p0 < HW; p0 += 32768) {   /* the 1x1 shortcut of x, 32768 pixels a GEMM */
        int np = HW - p0 < 32768 ? (int)(HW - p0) : 32768;
        ok = vkc_matmul((ColiVkTensor *)r->sc.vk, x, p0 * r->cin, s, p0 * r->cout, np) && qvv_bias(s, p0 * r->cout, np, &r->sc);
    }
    VkcEw add = {VKC_EW_ADD, (int)(HW * r->cout), r->cout, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
    ok = ok && vkc_ew(y, y, s ? s : x, NULL, NULL, &add) && vkc_submit(0) && vkc_begin();
    vkc_free(pa); vkc_free(t); vkc_free(pb); vkc_free(s);
    if (!ok) { vkc_free(y); return NULL; }
    return y;
}

/* qiv_decode on the device: 0, or -1 (nothing written; the CPU decodes) */
static int qvv_decode(QiVae *v, const float *z, int h, int w, uint8_t *rgba, float *out_f) {
    if (g_qvv.failed || vkc_lost() || !vkc_vae_ready() || !v || !z || h <= 0 || w <= 0) return -1;
    if (!g_qvv.ok && !qvv_setup(v)) return -1;
    const int prof = getenv("QIV_PROFILE") != NULL, Z = v->z_dim, steps = 3 + v->nblk;
    int step = 0, H = h, W = w;
    size_t HW = (size_t)H * W;
    double t0 = qiv_now(), tl = t0;
#define QVV_STAGE(name) do { double tn = qiv_now(); \
        if (prof) fprintf(stderr, "[vae vk] %-10s %4dx%-4d %7.3f s\n", name, H, W, tn - tl); tl = tn; \
        if (v->progress) v->progress(v->progress_ud, ++step, steps); } while (0)
    float *a = qiv_alloc(HW * Z), *b = qiv_alloc(HW * Z), *host = NULL;
    VkcBuf *x = NULL, *bb = NULL, *nx = NULL;
    int ok = a && b;
    if (ok) {   /* z * std + mean and post_quant_conv on the CPU, as qiv_decode */
        for (size_t p = 0; p < HW; p++)
            for (int c = 0; c < Z; c++) a[p * Z + c] = z[p * Z + c] * v->std[c] + v->mean[c];
        qi_gemm(b, a, (int)HW, &v->pqc.m, v->pqc.b);
    }
    ok = ok && (bb = qvv_map(HW * Z)) && (x = qvv_map(HW * v->conv_in.cout)) && vkc_begin() &&
         vkc_write(bb, 0, b, HW * Z * sizeof(float)) && qvv_conv3(&v->conv_in, bb, H, W, 0, x);
    free(a); free(b); a = b = NULL;
    vkc_free(bb); bb = NULL;
    if (ok) QVV_STAGE("conv_in");
    /* mid: a resnet, the attention on the CPU, a resnet */
    ok = ok && (nx = qvv_resnet(&v->mid[0], x, H, W));
    if (ok) { vkc_free(x); x = nx; nx = NULL; }
    ok = ok && (host = qiv_alloc(HW * v->mid[0].cout)) && vkc_submit(1) &&
         vkc_read(x, 0, host, HW * v->mid[0].cout * sizeof(float)) && qiv_attention(v, host, (int)HW) == 0 &&
         vkc_begin() && vkc_write(x, 0, host, HW * v->mid[0].cout * sizeof(float));
    free(host); host = NULL;
    ok = ok && (nx = qvv_resnet(&v->mid[1], x, H, W));
    if (ok) { vkc_free(x); x = nx; nx = NULL; QVV_STAGE("mid"); }
    for (int i = 0; ok && i < v->nblk; i++) {
        const QivBlock *bk = &v->blk[i];
        VkcBuf *xin = x, *cur = x;   /* xin: the block's input, the DupUp3D shortcut's source */
        for (int j = 0; ok && j < bk->nres; j++) {
            ok = (nx = qvv_resnet(&bk->res[j], cur, H, W)) != NULL;
            if (ok) { if (cur != xin) vkc_free(cur); cur = nx; nx = NULL; }
        }
        if (ok && bk->up) {
            size_t HW4 = HW * 4;
            VkcBuf *y = qvv_map(HW4 * bk->cout);
            VkcVae p = {1, bk->cout, H, W, 1, 2 * H, 2 * W, 0, (int)(HW4 * bk->cout), 0, 0, bk->cout, bk->cin};
            ok = y && qvv_conv3(&bk->upc, cur, H, W, 1, y) && vkc_vae(xin, y, g_qvv.dup[i], &p) && vkc_submit(0) && vkc_begin();
            if (cur != xin) vkc_free(cur);
            vkc_free(xin);
            x = y;
            H *= 2; W *= 2; HW = (size_t)H * W;
        } else if (ok) {
            if (xin != cur) vkc_free(xin);
            x = cur;
        } else {
            if (cur != xin) vkc_free(cur);
            x = xin;
        }
        if (ok) { char nm[16]; snprintf(nm, sizeof nm, "block%d", i); QVV_STAGE(nm); }
    }
    const int OC = v->out_ch;
    VkcBuf *pre = NULL, *o = NULL, *od = NULL;
    ok = ok && (pre = qvv_map(HW * v->conv_out.cin)) && (o = qvv_map(HW * OC)) &&
         (od = vkc_buf(HW * OC * sizeof(float), VKC_DOWN)) && qvv_pre(x, pre, HW, v->conv_out.cin, v->g_outo) &&
         qvv_conv3(&v->conv_out, pre, H, W, 0, o) && vkc_copy(od, 0, o, 0, HW * OC) && vkc_submit(1);
    if (ok) qiv_post((const float *)vkc_ptr(od), HW, OC, rgba, out_f);
    vkc_free(pre); vkc_free(o); vkc_free(od); vkc_free(x);
    if (!ok) {
        if (vkc_lost()) qvv_off("the device was lost");
        else qvv_off("a step of the decode failed (the device's memory, or a shape its ops do not take)");
        return -1;
    }
    QVV_STAGE("conv_out");
    if (prof) fprintf(stderr, "[vae vk] total %.3f s\n", qiv_now() - t0);
#undef QVV_STAGE
    g_qvv.decodes++;
    return 0;
}
