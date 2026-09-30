/* qwenimage_vae.h -- the Qwen-Image-2.1 VAE decoder (diffusers AutoencoderKLQwenImage21),
 * one image at a time, pure C on top of qi_gemm.
 *
 * The diffusers class is Wan's causal video VAE specialised to images. For one
 * frame its time machinery folds away, and this file only keeps what a single
 * frame runs:
 *   - every CausalConv3d is a zero-padded Conv2d (the class squeezes the time
 *     axis and calls nn.Conv2d), so the weights are plain [Cout][Cin][k][k];
 *   - the time_conv of an "upsample3d" resample only runs from the second chunk
 *     on (the first one stores "Rep" in the feature cache), so it is never loaded;
 *   - DupUp3D with first_chunk=True keeps temporal slot factor_t-1 only, which
 *     makes the up-block shortcut a gather of input channels (qiv_dup_table).
 * Path: z*std + mean, post_quant_conv (1x1), conv_in, mid block (resnet,
 * single-head attention over all pixels, resnet), residual up blocks, RMS norm
 * + SiLU, conv_out, clamp to [-1, 1], then VaeImageProcessor.postprocess
 * ((x*0.5 + 0.5).clamp(0, 1) * 255, numpy round-half-even, uint8).
 *
 * Activations are channel-last [H][W][C]. A 3x3 conv is one qi_gemm per band of
 * output rows over a 3x (not 9x) im2col, see qiv_conv3; the weight is permuted
 * at load to [Cout][kx][ky][Cin] to match. The im2col only exists for a band of
 * output rows (qiv_col_floats), and the input rows a band has consumed
 * are handed back to the OS when nothing reads them later (qiv_release): at
 * 1024x1024 the last up blocks hold 288-channel maps of 1.2 GB each, and without
 * that the decode would not fit next to the rest of the pipeline.
 *
 * Cost: about 4.2 TFLOP at 512x512, growing with the pixel count; measured on
 * ds (Zen 4, 8 threads) 81% of the time is qi_gemm at 700-800 GFLOP/s, the rest
 * is the RMS norm + SiLU prologue, the im2col copy and the epilogues. Peak
 * memory is the weights (1.0 GB f32) plus about 1.8 GB per megapixel.
 *
 * Interface: qiv_load(dir) / qiv_decode(v, z, h, w, rgba, out_f) / qiv_free(v);
 * qiv_set_progress for a callback after each stage (8 for the real model).
 */
#ifndef COLIBRI_QWENIMAGE_VAE_H
#define COLIBRI_QWENIMAGE_VAE_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "st.h"
#include "json.h"
#include "qi_gemm.h"
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define QIV_MAX_BLOCKS 8
#define QIV_MAX_RES 8
/* im2col band budget in floats (64 MB): large enough that the GEMM sees
 * thousands of rows, small next to the activations it serves. The attention
 * score block gets 16 MB. Variables so a test can shrink both to one row and
 * cover the band edges on a tiny model. */
static size_t qiv_col_floats = (size_t)16 << 20;
static size_t qiv_attn_floats = (size_t)4 << 20;

typedef struct {
    int cin, cout, k;       /* k = 1 or 3 */
    float *w, *b;           /* w: [cout][kx][ky][cin] */
    QiMat m;
} QivConv;

typedef struct {
    int cin, cout;
    float *g1, *g2;         /* RMS norm gammas, [cin] and [cout] */
    QivConv c1, c2, sc;     /* sc only when cin != cout */
} QivRes;

typedef struct {
    int cin, cout, up, ft;  /* up: 2x upsample + DupUp3D shortcut; ft: its factor_t */
    int nres;
    QivRes res[QIV_MAX_RES];
    QivConv upc;            /* the conv after the nearest 2x upsample */
    int *dup;               /* [cout][2][2] -> input channel of the DupUp3D shortcut */
} QivBlock;

typedef struct QiVae {
    int z_dim, out_ch, scale;   /* scale: pixels per latent along each side (16) */
    float *mean, *std;          /* [z_dim] from config.json */
    QivConv pqc, conv_in;
    QivRes mid[2];
    float *attn_g;
    QivConv qkv, proj;
    int nblk;
    QivBlock blk[QIV_MAX_BLOCKS];
    float *g_out;
    QivConv conv_out;
    size_t wbytes;              /* resident weight bytes */
    void (*progress)(void *ud, int step, int steps);
    void *progress_ud;
} QiVae;

static double qiv_now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------- memory -- */

/* Hand the whole pages of buf[from, upto) (in floats) back to the OS: the caller
 * never reads them again. Only for large buffers, which glibc serves with their
 * own mmap; a page strictly inside a buffer belongs to it alone either way, and
 * free() later only writes (never reads) near the chunk edges. Elsewhere this is
 * a no-op and the peak is simply higher. */
static void qiv_release(float *buf, size_t total, size_t from, size_t upto){
#if defined(__linux__) && defined(MADV_DONTNEED)
    if (!buf || total < (8u << 20) || upto <= from) return;
    static long pg = 0;
    if (!pg) { pg = sysconf(_SC_PAGESIZE); if (pg <= 0) pg = 4096; }
    uintptr_t a = (uintptr_t)(buf + from), e = (uintptr_t)(buf + upto);
    a = (a + pg - 1) & ~(uintptr_t)(pg - 1);
    e = e & ~(uintptr_t)(pg - 1);
    if (e > a) madvise((void *)a, e - a, MADV_DONTNEED);
#else
    (void)buf; (void)total; (void)from; (void)upto;
#endif
}

static float *qiv_alloc(size_t n){
    float *p = (float *)malloc(n * sizeof(float) + 64);
    if (!p) fprintf(stderr, "qwenimage vae: out of memory (%.1f MB)\n", n * 4.0 / 1e6);
    return p;
}

/* ----------------------------------------------------------------- loading -- */

static float *qiv_read(shards *S, const char *name, int64_t numel, char *err, size_t errn){
    st_tensor *t = st_find(S, name);
    if (!t) { snprintf(err, errn, "missing tensor %s", name); return NULL; }
    if (t->numel != numel) {
        snprintf(err, errn, "tensor %s has %lld elements, expected %lld", name,
                 (long long)t->numel, (long long)numel);
        return NULL;
    }
    float *p = qiv_alloc((size_t)numel);
    if (!p) { snprintf(err, errn, "out of memory reading %s", name); return NULL; }
    st_read_f32(S, name, p, 1);
    return p;
}

static int qiv_load_conv(shards *S, QiVae *v, const char *pfx, int cin, int cout, int k,
                         QivConv *c, char *err, size_t errn){
    char nm[256];
    c->cin = cin; c->cout = cout; c->k = k;
    snprintf(nm, sizeof nm, "%s.weight", pfx);
    float *w = qiv_read(S, nm, (int64_t)cout * cin * k * k, err, errn);
    if (!w) return -1;
    if (k > 1) {
        /* [cout][cin][ky][kx] -> [cout][kx][ky][cin], the order qiv_conv3 reads taps in */
        float *p = qiv_alloc((size_t)cout * cin * k * k);
        if (!p) { free(w); snprintf(err, errn, "out of memory"); return -1; }
        for (int o = 0; o < cout; o++)
            for (int i = 0; i < cin; i++)
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++)
                        p[((size_t)o * k * k + kx * k + ky) * cin + i] = w[((size_t)o * cin + i) * k * k + ky * k + kx];
        free(w); w = p;
    }
    snprintf(nm, sizeof nm, "%s.bias", pfx);
    float *b = qiv_read(S, nm, cout, err, errn);
    if (!b) { free(w); return -1; }
    c->w = w; c->b = b;
    c->m = (QiMat){ QI_F32, cout, cin * k * k, w, NULL };
    v->wbytes += ((size_t)cout * cin * k * k + cout) * sizeof(float);
    return 0;
}

static int qiv_load_res(shards *S, QiVae *v, const char *pfx, int cin, int cout, QivRes *r,
                        char *err, size_t errn){
    char nm[256];
    r->cin = cin; r->cout = cout;
    snprintf(nm, sizeof nm, "%s.norm1.gamma", pfx);
    if (!(r->g1 = qiv_read(S, nm, cin, err, errn))) return -1;
    snprintf(nm, sizeof nm, "%s.norm2.gamma", pfx);
    if (!(r->g2 = qiv_read(S, nm, cout, err, errn))) return -1;
    v->wbytes += (size_t)(cin + cout) * sizeof(float);
    snprintf(nm, sizeof nm, "%s.conv1", pfx);
    if (qiv_load_conv(S, v, nm, cin, cout, 3, &r->c1, err, errn)) return -1;
    snprintf(nm, sizeof nm, "%s.conv2", pfx);
    if (qiv_load_conv(S, v, nm, cout, cout, 3, &r->c2, err, errn)) return -1;
    if (cin != cout) {
        snprintf(nm, sizeof nm, "%s.conv_shortcut", pfx);
        if (qiv_load_conv(S, v, nm, cin, cout, 1, &r->sc, err, errn)) return -1;
    }
    return 0;
}

static void qiv_free_conv(QivConv *c){ free(c->w); free(c->b); c->w = c->b = NULL; }
static void qiv_free_res(QivRes *r){
    free(r->g1); free(r->g2); r->g1 = r->g2 = NULL;
    qiv_free_conv(&r->c1); qiv_free_conv(&r->c2); qiv_free_conv(&r->sc);
}

static void qiv_free(QiVae *v){
    if (!v) return;
    free(v->mean); free(v->std);
    qiv_free_conv(&v->pqc); qiv_free_conv(&v->conv_in);
    qiv_free_res(&v->mid[0]); qiv_free_res(&v->mid[1]);
    free(v->attn_g); qiv_free_conv(&v->qkv); qiv_free_conv(&v->proj);
    for (int i = 0; i < v->nblk; i++) {
        for (int j = 0; j < v->blk[i].nres; j++) qiv_free_res(&v->blk[i].res[j]);
        qiv_free_conv(&v->blk[i].upc);
        free(v->blk[i].dup);
    }
    free(v->g_out); qiv_free_conv(&v->conv_out);
    free(v);
}

static int qiv_json_int(jval *cfg, const char *key, int def){
    jval *j = json_get(cfg, key);
    return (j && j->t == J_NUM) ? (int)j->num : def;
}

/* DupUp3D for one frame with first_chunk: repeat_interleave(repeats) over
 * channels, view [cout][ft][2][2], keep time slot ft-1. So output channel o at
 * sub-pixel (sy, sx) reads input channel (o*4*ft + (ft-1)*4 + 2*sy + sx) / repeats. */
static int *qiv_dup_table(int cin, int cout, int ft){
    int factor = 4 * ft;
    if ((cout * factor) % cin) return NULL;
    int repeats = cout * factor / cin;
    int *t = (int *)malloc(sizeof(int) * cout * 4);
    if (!t) return NULL;
    for (int o = 0; o < cout; o++)
        for (int s = 0; s < 4; s++)
            t[o * 4 + s] = (o * factor + (ft - 1) * 4 + s) / repeats;
    return t;
}

static QiVae *qiv_load(const char *vae_dir){
    char err[512] = "", path[4096];
    QiVae *v = (QiVae *)calloc(1, sizeof(QiVae));
    shards S; int have_s = 0;
    jval *cfg = NULL;
    char *text = NULL;
    if (!v) { fprintf(stderr, "qwenimage vae: out of memory\n"); return NULL; }

    if (snprintf(path, sizeof path, "%s/config.json", vae_dir) >= (int)sizeof path) {
        snprintf(err, sizeof err, "path too long"); goto fail; }
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, sizeof err, "cannot open %.400s", path); goto fail; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    text = (char *)malloc(n > 0 ? n + 1 : 1);
    if (!text || n <= 0 || fread(text, 1, n, f) != (size_t)n) {
        fclose(f); snprintf(err, sizeof err, "cannot read %.400s", path); goto fail; }
    fclose(f); text[n] = 0;
    cfg = json_parse_checked(text);
    if (!cfg || cfg->t != J_OBJ) { snprintf(err, sizeof err, "%.400s is not a JSON object", path); goto fail; }

    v->z_dim = qiv_json_int(cfg, "z_dim", 0);
    v->out_ch = qiv_json_int(cfg, "out_channels", 3);
    int base = qiv_json_int(cfg, "decoder_base_dim", 0);
    if (base <= 0) base = qiv_json_int(cfg, "base_dim", 0);
    int nrb = qiv_json_int(cfg, "num_res_blocks", 2);
    jval *jm = json_get(cfg, "dim_mult"), *jt = json_get(cfg, "temperal_downsample");
    jval *jr = json_get(cfg, "is_residual"), *jp = json_get(cfg, "patch_size");
    jval *jmean = json_get(cfg, "latents_mean"), *jstd = json_get(cfg, "latents_std");
    if (v->z_dim <= 0 || base <= 0 || nrb < 0 || nrb + 1 > QIV_MAX_RES || !jm || jm->t != J_ARR
        || jm->len < 1 || jm->len > QIV_MAX_BLOCKS || !jt || jt->t != J_ARR || jt->len != jm->len - 1
        || v->out_ch <= 0) {
        snprintf(err, sizeof err, "%.400s: missing or unexpected z_dim/base_dim/dim_mult/temperal_downsample", path);
        goto fail;
    }
    /* The non-residual (Wan 2.1 style) decoder and patchified output are other
     * checkpoints; refuse them rather than decode garbage. */
    if (!jr || jr->t != J_BOOL || !jr->boolean) { snprintf(err, sizeof err, "only is_residual=true is supported"); goto fail; }
    if (jp && jp->t != J_NULL) { snprintf(err, sizeof err, "patch_size must be null"); goto fail; }
    if (!jmean || jmean->t != J_ARR || jmean->len != v->z_dim || !jstd || jstd->t != J_ARR || jstd->len != v->z_dim) {
        snprintf(err, sizeof err, "latents_mean/latents_std must have z_dim=%d entries", v->z_dim); goto fail; }
    v->mean = qiv_alloc(v->z_dim); v->std = qiv_alloc(v->z_dim);
    if (!v->mean || !v->std) { snprintf(err, sizeof err, "out of memory"); goto fail; }
    for (int i = 0; i < v->z_dim; i++) { v->mean[i] = (float)jmean->kids[i]->num; v->std[i] = (float)jstd->kids[i]->num; }

    int nm = jm->len, dims[QIV_MAX_BLOCKS + 1], tup[QIV_MAX_BLOCKS];
    /* Every width is a product of two config numbers that size buffers and a
     * divisor in qiv_dup_table: positive and bounded, or refused. */
    for (int i = 0; i < nm; i++)
        if (jm->kids[i]->t != J_NUM || jm->kids[i]->num < 1 || jm->kids[i]->num > 64 || base > 4096) {
            snprintf(err, sizeof err, "dim_mult entries must be whole numbers from 1 to 64 (base %d)", base); goto fail; }
    for (int i = 0; i < nm - 1; i++)
        if (jt->kids[i]->t != J_BOOL) { snprintf(err, sizeof err, "temperal_downsample must hold booleans"); goto fail; }
    if (v->z_dim > 4096) { snprintf(err, sizeof err, "z_dim %d is out of range", v->z_dim); goto fail; }
    dims[0] = base * (int)jm->kids[nm - 1]->num;
    for (int i = 0; i < nm; i++) dims[i + 1] = base * (int)jm->kids[nm - 1 - i]->num;
    for (int i = 0; i < nm - 1; i++) tup[i] = jt->kids[nm - 2 - i]->boolean;   /* temperal_upsample = reversed */

    {
        DIR *d = opendir(vae_dir);
        if (!d) { snprintf(err, sizeof err, "cannot open directory %.400s", vae_dir); goto fail; }
        closedir(d);
    }
    st_init(&S, vae_dir); have_s = 1;
    if (S.n == 0) { snprintf(err, sizeof err, "no safetensors in %.400s", vae_dir); goto fail; }

    if (qiv_load_conv(&S, v, "post_quant_conv", v->z_dim, v->z_dim, 1, &v->pqc, err, sizeof err)) goto fail;
    if (qiv_load_conv(&S, v, "decoder.conv_in", v->z_dim, dims[0], 3, &v->conv_in, err, sizeof err)) goto fail;
    for (int i = 0; i < 2; i++) {
        char p[128]; snprintf(p, sizeof p, "decoder.mid_block.resnets.%d", i);
        if (qiv_load_res(&S, v, p, dims[0], dims[0], &v->mid[i], err, sizeof err)) goto fail;
    }
    if (!(v->attn_g = qiv_read(&S, "decoder.mid_block.attentions.0.norm.gamma", dims[0], err, sizeof err))) goto fail;
    if (qiv_load_conv(&S, v, "decoder.mid_block.attentions.0.to_qkv", dims[0], 3 * dims[0], 1, &v->qkv, err, sizeof err)) goto fail;
    if (qiv_load_conv(&S, v, "decoder.mid_block.attentions.0.proj", dims[0], dims[0], 1, &v->proj, err, sizeof err)) goto fail;
    v->scale = 1;
    v->nblk = nm;
    for (int i = 0; i < nm; i++) {
        QivBlock *b = &v->blk[i];
        b->cin = dims[i]; b->cout = dims[i + 1];
        b->up = i != nm - 1;
        b->ft = b->up && tup[i] ? 2 : 1;
        b->nres = nrb + 1;
        for (int j = 0; j < b->nres; j++) {
            char p[128]; snprintf(p, sizeof p, "decoder.up_blocks.%d.resnets.%d", i, j);
            if (qiv_load_res(&S, v, p, j ? b->cout : b->cin, b->cout, &b->res[j], err, sizeof err)) goto fail;
        }
        if (b->up) {
            char p[128]; snprintf(p, sizeof p, "decoder.up_blocks.%d.upsampler.resample.1", i);
            if (qiv_load_conv(&S, v, p, b->cout, b->cout, 3, &b->upc, err, sizeof err)) goto fail;
            if (!(b->dup = qiv_dup_table(b->cin, b->cout, b->ft))) {
                snprintf(err, sizeof err, "up block %d: DupUp3D %d->%d channels is not a whole repeat", i, b->cin, b->cout);
                goto fail;
            }
            v->scale *= 2;
        }
    }
    if (!(v->g_out = qiv_read(&S, "decoder.norm_out.gamma", dims[nm], err, sizeof err))) goto fail;
    if (qiv_load_conv(&S, v, "decoder.conv_out", dims[nm], v->out_ch, 3, &v->conv_out, err, sizeof err)) goto fail;
    v->wbytes += (size_t)(dims[0] + dims[nm]) * sizeof(float);
    if (v->out_ch != 4 && v->out_ch != 3) { snprintf(err, sizeof err, "out_channels=%d, expected 3 or 4", v->out_ch); goto fail; }
    st_destroy(&S);
    json_free(cfg); free(text);
    return v;
fail:
    fprintf(stderr, "qwenimage vae: %s\n", err[0] ? err : "load failed");
    if (have_s) st_destroy(&S);
    json_free(cfg); free(text);
    qiv_free(v);
    return NULL;
}

static void qiv_set_progress(QiVae *v, void (*cb)(void *ud, int step, int steps), void *ud){
    v->progress = cb; v->progress_ud = ud;
}

/* --------------------------------------------------------------- compute -- */

/* RMS_norm: F.normalize over channels (x / max(||x||, 1e-12)), times sqrt(C),
 * times gamma; then SiLU when asked. Same operation order as the module. */
static inline void qiv_norm_px(float *d, const float *s, int C, const float *g, float scale, int silu){
    double ss = 0;
    for (int c = 0; c < C; c++) ss += (double)s[c] * s[c];
    float den = (float)sqrt(ss);
    if (den < 1e-12f) den = 1e-12f;
    for (int c = 0; c < C; c++) {
        float v = s[c] / den * scale * g[c];
        d[c] = silu ? v / (1.f + expf(-v)) : v;
    }
}

/* What a conv adds to its output before storing it, and which inputs it may
 * give back as it goes (the caller's promise that nothing reads them later). */
typedef struct {
    const float *res;           /* y += res, same layout as y; may be y itself */
    const float *scx;           /* y += sc(scx): a 1x1 conv of the block input at output resolution */
    const QivConv *sc;
    const float *dup;           /* y += DupUp3D(dup): dup is at half the output resolution */
    int dupC; const int *dupt;
    int rel_x, rel_scx, rel_dup;
} QivEpi;

/* y[Ho][Wo][cout] = conv3x3(pre(x)) (+ epilogue), with pre = RMS norm + SiLU if
 * gamma, and x nearest-upsampled 2x first if up. Processed in bands of output
 * rows. P holds the band's input rows after pre, with a one-row halo and a zero
 * column on each side (width Wp = Wo + 2). Q interleaves three consecutive P
 * rows: Q[r][u][ky] = P[r + ky][u]. Then the nine taps of output pixel (r, u)
 * are the 9*C contiguous floats at Q[r][u], ordered [kx][ky][c], and the next
 * pixel starts 3*C floats later: a GEMM with ldx = 3*C over the rows r*Wp + u
 * reads them in place. Q costs 3 copies of the input instead of the 9 of a full
 * im2col; the two rows per band that straddle a line end are computed and
 * dropped. */
static int qiv_conv3(const QivConv *cv, float *x, int H, int W, const float *gamma, int up,
                     float *y, const QivEpi *ep){
    const int C = cv->cin, Co = cv->cout, C3 = 3 * C;
    const int Ho = H << up, Wo = W << up, Wp = Wo + 2;
    const float scale = (float)sqrt((double)C);
    const size_t rowq = (size_t)Wp * C3;
    int band = (int)(qiv_col_floats / rowq);
    if (band < 1) band = 1;
    if (band > Ho) band = Ho;
    float *P = qiv_alloc((size_t)(band + 2) * Wp * C);
    float *Q = qiv_alloc((size_t)band * rowq);
    float *T = qiv_alloc((size_t)band * Wp * Co);
    float *T2 = ep && ep->sc ? qiv_alloc((size_t)band * Wo * Co) : NULL;
    if (!P || !Q || !T || (ep && ep->sc && !T2)) { free(P); free(Q); free(T); free(T2); return -1; }
    const size_t xtot = (size_t)H * W * C;
    const int scC = ep && ep->sc ? ep->sc->cin : 0;
    size_t xrel = 0, screl = 0, duprel = 0;

    for (int y0 = 0; y0 < Ho; y0 += band) {
        const int nb = Ho - y0 < band ? Ho - y0 : band;
        /* P row r = virtual input row y0-1+r */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < nb + 2; r++)
            for (int u = 0; u < Wp; u++) {
                float *d = P + ((size_t)r * Wp + u) * C;
                int vr = y0 - 1 + r, vu = u - 1;
                if (vr < 0 || vr >= Ho || vu < 0 || vu >= Wo) { memset(d, 0, sizeof(float) * C); continue; }
                const float *s = x + ((size_t)(vr >> up) * W + (vu >> up)) * C;
                if (gamma) qiv_norm_px(d, s, C, gamma, scale, 1);
                else memcpy(d, s, sizeof(float) * C);
            }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < nb; r++)
            for (int u = 0; u < Wp; u++) {
                float *d = Q + ((size_t)r * Wp + u) * C3;
                for (int ky = 0; ky < 3; ky++)
                    memcpy(d + ky * C, P + ((size_t)(r + ky) * Wp + u) * C, sizeof(float) * C);
            }
        qi_gemm_ld(T, Co, Q, C3, nb * Wp - 2, &cv->m, cv->b);
        if (T2) qi_gemm(T2, ep->scx + (size_t)y0 * Wo * scC, nb * Wo, &ep->sc->m, ep->sc->b);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < nb; r++)
            for (int u = 0; u < Wo; u++) {
                const int p = r * Wo + u;
                float *t = T + ((size_t)r * Wp + u) * Co;
                size_t gp = (size_t)y0 * Wo + p;
                if (T2) { const float *t2 = T2 + (size_t)p * Co; for (int o = 0; o < Co; o++) t[o] += t2[o]; }
                if (ep && ep->dup) {
                    int Y = y0 + r;
                    const float *s = ep->dup + ((size_t)(Y >> 1) * (Wo >> 1) + (u >> 1)) * ep->dupC;
                    const int *tab = ep->dupt + ((Y & 1) * 2 + (u & 1));
                    for (int o = 0; o < Co; o++) t[o] += s[tab[o * 4]];
                }
                if (ep && ep->res) { const float *rr = ep->res + gp * Co; for (int o = 0; o < Co; o++) t[o] += rr[o]; }
                memcpy(y + gp * Co, t, sizeof(float) * Co);
            }
        const int y1 = y0 + nb;
        if (ep && ep->rel_x) {
            size_t upto = y1 >= Ho ? xtot : (size_t)((y1 - 1) >> up) * W * C;
            qiv_release(x, xtot, xrel, upto); if (upto > xrel) xrel = upto;
        }
        if (ep && ep->rel_scx && ep->scx) {
            size_t tot = (size_t)Ho * Wo * scC, upto = (size_t)y1 * Wo * scC;
            qiv_release((float *)ep->scx, tot, screl, upto); screl = upto;
        }
        if (ep && ep->rel_dup && ep->dup) {
            size_t tot = (size_t)(Ho >> 1) * (Wo >> 1) * ep->dupC;
            size_t upto = y1 >= Ho ? tot : (size_t)(y1 >> 1) * (Wo >> 1) * ep->dupC;
            qiv_release((float *)ep->dup, tot, duprel, upto); duprel = upto;
        }
    }
    free(P); free(Q); free(T); free(T2);
    return 0;
}

/* One residual block. x [H][W][cin] is consumed unless keep: the result is
 * written over x when the shapes allow it, otherwise into a new buffer and x is
 * freed. Returns the output or NULL on out of memory (x is then still owned by
 * the caller). */
static float *qiv_resnet(const QivRes *r, float *x, int H, int W, int keep){
    const size_t HW = (size_t)H * W;
    float *t = qiv_alloc(HW * r->cout);
    if (!t) return NULL;
    if (qiv_conv3(&r->c1, x, H, W, r->g1, 0, t, NULL)) { free(t); return NULL; }
    QivEpi ep; memset(&ep, 0, sizeof ep);
    ep.rel_x = 1;                       /* conv2 is the last reader of t */
    float *y;
    if (r->cin == r->cout) {
        ep.res = x;
        y = keep ? qiv_alloc(HW * r->cout) : x;
    } else {
        ep.scx = x; ep.sc = &r->sc; ep.rel_scx = !keep;
        y = qiv_alloc(HW * r->cout);
    }
    if (!y || qiv_conv3(&r->c2, t, H, W, r->g2, 0, y, &ep)) { free(t); if (y && y != x) free(y); return NULL; }
    free(t);
    if (y != x && !keep) free(x);
    return y;
}

/* Mid-block attention, in place: x += proj(softmax(q k^T / sqrt(C)) v), one head,
 * every pixel attends to every pixel. Queries go in blocks so the score matrix
 * stays small at 2048x2048 (16384 tokens). */
static int qiv_attention(const QiVae *v, float *x, int N){
    const int C = v->qkv.cin;
    const float scale = (float)sqrt((double)C);
    float *xn = qiv_alloc((size_t)N * C), *qkv = qiv_alloc((size_t)N * 3 * C);
    float *k = qiv_alloc((size_t)N * C), *vt = qiv_alloc((size_t)N * C);
    int qb = (int)(qiv_attn_floats / (size_t)N); if (qb < 16) qb = 16; if (qb > N) qb = N;
    float *S = qiv_alloc((size_t)qb * N);
    int rc = -1;
    if (!xn || !qkv || !k || !vt || !S) goto out;
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < N; p++) qiv_norm_px(xn + (size_t)p * C, x + (size_t)p * C, C, v->attn_g, scale, 0);
    qi_gemm(qkv, xn, N, &v->qkv.m, v->qkv.b);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < N; p++) {
        memcpy(k + (size_t)p * C, qkv + (size_t)p * 3 * C + C, sizeof(float) * C);
        for (int c = 0; c < C; c++) vt[(size_t)c * N + p] = qkv[(size_t)p * 3 * C + 2 * C + c];
    }
    const QiMat Km = { QI_F32, N, C, k, NULL }, Vm = { QI_F32, C, N, vt, NULL };
    const float isd = 1.f / scale;
    for (int i0 = 0; i0 < N; i0 += qb) {
        int m = N - i0 < qb ? N - i0 : qb;
        qi_gemm_ld(S, N, qkv + (size_t)i0 * 3 * C, 3 * C, m, &Km, NULL);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            float *s = S + (size_t)i * N, mx = -INFINITY;
            for (int j = 0; j < N; j++) { s[j] *= isd; if (s[j] > mx) mx = s[j]; }
            double sum = 0;
            for (int j = 0; j < N; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
            float inv = (float)(1.0 / sum);
            for (int j = 0; j < N; j++) s[j] *= inv;
        }
        qi_gemm(xn + (size_t)i0 * C, S, m, &Vm, NULL);        /* xn is free again: reuse for O */
    }
    qi_gemm(qkv, xn, N, &v->proj.m, v->proj.b);             /* qkv reused for proj(O) [N][C] */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)N * C; i++) x[i] += qkv[i];
    rc = 0;
out:
    free(xn); free(qkv); free(k); free(vt); free(S);
    return rc;
}

/* z: normalized latents [h][w][z_dim] (the packed DiT output, token y*w + x).
 * rgba: [scale*h][scale*w][4] uint8 top row first (alpha 255 if the model has 3
 * output channels). out_f: optional [out_ch][scale*h][scale*w] in [-1, 1].
 * Returns 0, or -1 on out of memory / bad arguments. */
static int qiv_decode(QiVae *v, const float *z, int h, int w, uint8_t *rgba, float *out_f){
    if (!v || !z || h <= 0 || w <= 0) return -1;
    const int prof = getenv("QIV_PROFILE") != NULL;
    double t0 = qiv_now(), tl = t0;
    const int Z = v->z_dim, steps = 3 + v->nblk;
    int step = 0, H = h, W = w;
    size_t HW = (size_t)H * W;
#define QIV_STAGE(name) do { double tn = qiv_now(); \
        if (prof) fprintf(stderr, "[vae] %-10s %4dx%-4d %7.3f s\n", name, H, W, tn - tl); tl = tn; \
        if (v->progress) v->progress(v->progress_ud, ++step, steps); } while (0)
    float *a = qiv_alloc(HW * Z), *b = qiv_alloc(HW * Z), *x = NULL;
    if (!a || !b) goto fail;
    for (size_t p = 0; p < HW; p++)
        for (int c = 0; c < Z; c++) a[p * Z + c] = z[p * Z + c] * v->std[c] + v->mean[c];
    qi_gemm(b, a, (int)HW, &v->pqc.m, v->pqc.b);
    free(a); a = NULL;
    x = qiv_alloc(HW * v->conv_in.cout);
    if (!x || qiv_conv3(&v->conv_in, b, H, W, NULL, 0, x, NULL)) goto fail;
    free(b); b = NULL;
    QIV_STAGE("conv_in");

    float *nx;
    if (!(nx = qiv_resnet(&v->mid[0], x, H, W, 0))) goto fail;
    x = nx;
    if (qiv_attention(v, x, (int)HW)) goto fail;
    if (!(nx = qiv_resnet(&v->mid[1], x, H, W, 0))) goto fail;
    x = nx;
    QIV_STAGE("mid");

    for (int i = 0; i < v->nblk; i++) {
        const QivBlock *bk = &v->blk[i];
        /* xin: kept for the DupUp3D shortcut of an upsampling block, so its
         * first resnet must not write over it */
        float *xin = bk->up ? x : NULL, *cur = x;
        x = NULL;
        for (int j = 0; j < bk->nres; j++) {
            if (!(nx = qiv_resnet(&bk->res[j], cur, H, W, bk->up && j == 0))) {
                if (cur != xin) free(cur);
                free(xin); goto fail;
            }
            cur = nx;
        }
        if (bk->up) {
            float *y = qiv_alloc(HW * 4 * bk->cout);
            QivEpi ep; memset(&ep, 0, sizeof ep);
            ep.dup = xin; ep.dupC = bk->cin; ep.dupt = bk->dup;
            ep.rel_x = 1; ep.rel_dup = 1;
            if (!y || qiv_conv3(&bk->upc, cur, H, W, NULL, 1, y, &ep)) {
                free(y); free(cur); free(xin); goto fail;
            }
            free(cur); free(xin);
            H *= 2; W *= 2; HW = (size_t)H * W;
            x = y;
        } else {
            x = cur;
        }
        char nm[16]; snprintf(nm, sizeof nm, "block%d", i);
        QIV_STAGE(nm);
    }

    {
        const int OC = v->out_ch;
        float *o = qiv_alloc(HW * OC);
        QivEpi ep; memset(&ep, 0, sizeof ep); ep.rel_x = 1;
        if (!o || qiv_conv3(&v->conv_out, x, H, W, v->g_out, 0, o, &ep)) { free(o); goto fail; }
        free(x); x = NULL;
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < HW; p++)
            for (int c = 0; c < 4; c++) {
                float f = c < OC ? o[p * OC + c] : 1.f;
                /* torch.clamp(-1, 1), then postprocess: (x*0.5 + 0.5).clamp(0, 1),
                 * *255 in float32, numpy round (half to even), uint8. x*0.5 is
                 * exact, so an FMA here rounds exactly like torch's two ops. */
                f = f < -1.f ? -1.f : f > 1.f ? 1.f : f;
                if (c < OC && out_f) out_f[(size_t)c * HW + p] = f;
                float u = f * 0.5f + 0.5f;
                u = u < 0.f ? 0.f : u > 1.f ? 1.f : u;
                if (rgba) rgba[p * 4 + c] = (uint8_t)nearbyintf(u * 255.f);
            }
        free(o);
    }
    QIV_STAGE("conv_out");
    if (prof) fprintf(stderr, "[vae] total %.3f s\n", qiv_now() - t0);
#undef QIV_STAGE
    return 0;
fail:
    free(a); free(b); free(x);
    fprintf(stderr, "qwenimage vae: decode failed (out of memory?)\n");
    return -1;
}

#endif
