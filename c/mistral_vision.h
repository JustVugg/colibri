/* mistral_vision.h -- Mistral-Large-4's vision tower (Pixtral) and its projector, in C.
 *
 * The patches tools/mistral_image.py cuts (14 x 14, channel, row, column: 588 values a patch,
 * row-major over the patch grid) go to the rows the language model reads at the prompt's
 * [IMG] tokens, one row per 2 x 2 block of patches. Transcribed from transformers'
 * modeling_pixtral.py and modeling_mistral3.py (the release's architecture):
 *
 *   patch_conv  a linear map of each patch (no bias), then an RMSNorm (ln_pre)
 *   layers      pre-norm, residual: bidirectional attention over the picture (16 heads of
 *               128, no bias, an axial RoPE: head_dim / 4 frequencies for the row, as many for
 *               the column, the even and the odd ones of the 1D set, concatenated and doubled,
 *               rotate-half over the whole head), then an RMSNorm and a SiLU-gated MLP
 *   projector   an RMSNorm (the text model's eps), the 2 x 2 patches of a block side by side
 *               (unfold's order: channel, then the block's row, then its column), a linear map
 *               to the tower's width (no bias), linear to the model's width, exact GELU, linear
 *
 * An mmproj's q and k rows come in interleaved rotary order (GGUF converters permute them): the
 * rotation takes the pairs (2j, 2j + 1) there, the same angles as the release's rotate-half. With
 * the halves instead, the text was right and a test picture's colours too, but its shapes
 * were all "triangles": the positions were scrambled, not the content.
 *
 * The matrices stay as the mmproj has them (BF16, F16, F32 or a block type): a block of 32
 * rows is turned to f32 where it is used, so the 1.7B tower holds 3.4 GB beside a model that
 * needs every gigabyte, not 6.8. The attention is quadratic in the patches: the gateway's
 * token ceiling (ML4_MAX_IMAGE_TOKENS) keeps a picture's cost bounded.
 */
#ifndef MISTRAL_VISION_H
#define MISTRAL_VISION_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "gguf.h"

typedef struct { int type, O, I; int64_t rb; uint8_t *data; } MvMat;   /* rows O of I, the file's blocks */
typedef struct { float *ln1, *ln2; MvMat q, k, v, o, gate, up, down; } MvLayer;
typedef struct {
    int depth, hidden, heads, head_dim, inter, patch, merge, in_ch, out_hidden;
    float eps, proj_eps, theta;
    int rope_pairs;   /* 1: q and k rows of a head in interleaved rotary order, as GGUF converters write
                       * them (row 2j is the release's row j, 2j + 1 its row j + head_dim / 2): the
                       * rotation takes the pairs (2j, 2j + 1). 0: the release's order, rotate-half */
    float *patch_w;                      /* [D, in_ch * patch * patch] */
    float *pre_ln, *mm_norm;             /* [D] */
    MvLayer *layers;
    MvMat merger, l1, l2;                /* [D, 4D], [out, D], [out, out] */
    int ready;
} MistralVision;

typedef float mv_v8 __attribute__((vector_size(32)));
static inline float mv_hsum(mv_v8 v) { float a = 0.f; for (int i = 0; i < 8; i++) a += v[i]; return a; }

/* y[s, o] = x[s, :] . w[o, :]: each thread turns 32 rows of w to f32 once and runs every row
 * of x against them, 4 rows x 4 outputs a step in 16 vector sums */
static void mv_gemm(float *y, int ldy, const float *x, int ldx, const MvMat *w, int S) {
    const int OB = 32, O = w->O, I = w->I, nob = (O + OB - 1) / OB;
    #pragma omp parallel
    {
        float *tile = (float *)malloc((size_t)OB * I * sizeof(float));
        #pragma omp for schedule(dynamic, 1)
        for (int ob = 0; ob < nob; ob++) {
            const int o0 = ob * OB, no_t = O - o0 < OB ? O - o0 : OB;
            for (int r = 0; r < no_t; r++)
                if (w->type == GQ_F32) memcpy(tile + (size_t)r * I, w->data + (size_t)(o0 + r) * w->rb, (size_t)I * sizeof(float));
                else gq_dequant_row(w->type, w->data + (size_t)(o0 + r) * w->rb, tile + (size_t)r * I, I);
            for (int s = 0; s < S; s += 4)
                for (int o = 0; o < no_t; o += 4) {
                    const int ns = S - s < 4 ? S - s : 4, no = no_t - o < 4 ? no_t - o : 4;
                    mv_v8 acc[4][4];
                    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] = (mv_v8){0};
                    const float *xr[4], *wr[4];
                    for (int a = 0; a < 4; a++) xr[a] = x + (size_t)(s + (a < ns ? a : 0)) * ldx;
                    for (int b = 0; b < 4; b++) wr[b] = tile + (size_t)(o + (b < no ? b : 0)) * I;
                    int i = 0;
                    for (; i + 8 <= I; i += 8) {
                        mv_v8 xv[4], wv[4];
                        for (int a = 0; a < 4; a++) memcpy(&xv[a], xr[a] + i, 32);
                        for (int b = 0; b < 4; b++) memcpy(&wv[b], wr[b] + i, 32);
                        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] += xv[a] * wv[b];
                    }
                    for (int a = 0; a < ns; a++)
                        for (int b = 0; b < no; b++) {
                            float t = mv_hsum(acc[a][b]);
                            for (int k = i; k < I; k++) t += xr[a][k] * wr[b][k];
                            y[(size_t)(s + a) * ldy + o0 + o + b] = t;
                        }
                }
        }
        free(tile);
    }
}

static void mv_rms(float *dst, const float *src, const float *w, int rows, int n, float eps) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *p = src + (size_t)r * n; float *q = dst + (size_t)r * n;
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)p[i] * p[i];
        float inv = 1.f / sqrtf((float)(ss / n) + eps);
        for (int i = 0; i < n; i++) q[i] = w[i] * (p[i] * inv);
    }
}

/* the axial RoPE's angle j of a head (0 <= j < head_dim / 2) at patch (row, col) */
static inline float mv_angle(const MistralVision *v, int row, int col, int j) {
    const int hd = v->head_dim, quarter = hd / 4;
    /* the 1D set inv[i] = theta^(-2i / hd), i < hd / 2: the row takes inv[0], inv[2], ...;
     * the column inv[1], inv[3], ... */
    int i = j < quarter ? 2 * j : 2 * (j - quarter) + 1;
    float inv = (float)(1.0 / pow((double)v->theta, (double)(2 * i) / (double)hd));
    return (float)(j < quarter ? row : col) * inv;
}

/* N patches of grid gh x gw (patch rows, columns) in, (gh / merge) x (gw / merge) rows of
 * out_hidden out. Returns the rows written, -1 on a bad grid or no memory. */
static int mistral_vision_forward(const MistralVision *v, const float *patches, int gh, int gw, float *out) {
    const int D = v->hidden, H = v->heads, hd = v->head_dim, F = v->inter, m = v->merge;
    const int N = gh * gw, PD = v->in_ch * v->patch * v->patch;
    if (!v->ready || gh < 1 || gw < 1 || gh % m || gw % m || H * hd != D) return -1;
    const int R = (gh / m) * (gw / m);
    float *x = (float *)malloc((size_t)N * D * sizeof(float)), *h = (float *)malloc((size_t)N * D * sizeof(float));
    float *q = (float *)malloc((size_t)N * D * sizeof(float)), *k = (float *)malloc((size_t)N * D * sizeof(float));
    float *vv = (float *)malloc((size_t)N * D * sizeof(float)), *att = (float *)malloc((size_t)N * D * sizeof(float));
    float *g = (float *)malloc((size_t)N * F * sizeof(float)), *u = (float *)malloc((size_t)N * F * sizeof(float));
    float *cs = (float *)malloc((size_t)N * (hd / 2) * 2 * sizeof(float));
    if (!x || !h || !q || !k || !vv || !att || !g || !u || !cs) {
        free(x); free(h); free(q); free(k); free(vv); free(att); free(g); free(u); free(cs); return -1;
    }
    /* patch_conv: a [D, PD] matrix over the patches, then ln_pre */
    MvMat pw = {GQ_F32, D, PD, (int64_t)PD * 4, (uint8_t *)v->patch_w};
    mv_gemm(h, D, patches, PD, &pw, N);
    mv_rms(x, h, v->pre_ln, N, D, v->eps);
    for (int p = 0; p < N; p++)   /* cos | sin of every angle at every patch */
        for (int j = 0; j < hd / 2; j++) {
            float a = mv_angle(v, p / gw, p % gw, j);
            cs[(size_t)p * hd + j] = cosf(a); cs[(size_t)p * hd + hd / 2 + j] = sinf(a);
        }
    const float scale = 1.f / sqrtf((float)hd);
    for (int l = 0; l < v->depth; l++) {
        const MvLayer *L = &v->layers[l];
        mv_rms(h, x, L->ln1, N, D, v->eps);
        mv_gemm(q, D, h, D, &L->q, N); mv_gemm(k, D, h, D, &L->k, N); mv_gemm(vv, D, h, D, &L->v, N);
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < N; p++)   /* rotate-half over each head, the patch's angles */
            for (int hh = 0; hh < H; hh++) {
                const float *c = cs + (size_t)p * hd, *s = c + hd / 2;
                float *qh = q + (size_t)p * D + hh * hd, *kh = k + (size_t)p * D + hh * hd;
                for (int j = 0; j < hd / 2; j++) {
                    /* an mmproj's q and k rows: the pairs (2j, 2j + 1); the release's: j and j + hd / 2 */
                    const int a = v->rope_pairs ? 2 * j : j, b = v->rope_pairs ? 2 * j + 1 : j + hd / 2;
                    float q1 = qh[a], q2 = qh[b], k1 = kh[a], k2 = kh[b];
                    qh[a] = q1 * c[j] - q2 * s[j]; qh[b] = q2 * c[j] + q1 * s[j];
                    kh[a] = k1 * c[j] - k2 * s[j]; kh[b] = k2 * c[j] + k1 * s[j];
                }
            }
        #pragma omp parallel
        {
            float *sc = (float *)malloc((size_t)N * sizeof(float));
            #pragma omp for collapse(2) schedule(dynamic, 16)
            for (int hh = 0; hh < H; hh++)
                for (int i = 0; i < N; i++) {
                    const float *qi = q + (size_t)i * D + hh * hd;
                    float mx = -INFINITY;
                    for (int j = 0; j < N; j++) {
                        const float *kj = k + (size_t)j * D + hh * hd;
                        mv_v8 acc = {0};
                        for (int d = 0; d < hd; d += 8) { mv_v8 a, b; memcpy(&a, qi + d, 32); memcpy(&b, kj + d, 32); acc += a * b; }
                        sc[j] = mv_hsum(acc) * scale;
                        if (sc[j] > mx) mx = sc[j];
                    }
                    double sum = 0;
                    for (int j = 0; j < N; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                    float inv = (float)(1.0 / sum), *o = att + (size_t)i * D + hh * hd;
                    for (int d = 0; d < hd; d++) o[d] = 0.f;
                    for (int j = 0; j < N; j++) {
                        const float pj = sc[j] * inv, *vj = vv + (size_t)j * D + hh * hd;
                        for (int d = 0; d < hd; d++) o[d] += pj * vj[d];
                    }
                }
            free(sc);
        }
        mv_gemm(h, D, att, D, &L->o, N);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < (size_t)N * D; t++) x[t] += h[t];
        mv_rms(h, x, L->ln2, N, D, v->eps);
        mv_gemm(g, F, h, D, &L->gate, N); mv_gemm(u, F, h, D, &L->up, N);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < (size_t)N * F; t++) g[t] = g[t] / (1.f + expf(-g[t])) * u[t];
        mv_gemm(h, D, g, F, &L->down, N);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < (size_t)N * D; t++) x[t] += h[t];
    }
    /* projector: the norm, the 2 x 2 blocks side by side in unfold's order, then the three maps */
    mv_rms(h, x, v->mm_norm, N, D, v->proj_eps);
    float *blk = q;   /* R rows of D m m: q holds N D floats, as many */
    const int bw = gw / m;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < R; r++) {
        const int br = r / bw, bc = r % bw;
        float *dst = blk + (size_t)r * D * m * m;
        for (int c = 0; c < D; c++)
            for (int ki = 0; ki < m; ki++)
                for (int kj = 0; kj < m; kj++)
                    dst[(size_t)c * m * m + ki * m + kj] = h[((size_t)(br * m + ki) * gw + bc * m + kj) * D + c];
    }
    mv_gemm(k, D, blk, D * m * m, &v->merger, R);
    const int P = v->l1.O;
    float *t1 = g;   /* R rows of P: g holds N F floats */
    mv_gemm(t1, P, k, D, &v->l1, R);
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < (size_t)R * P; t++) t1[t] = 0.5f * t1[t] * (1.f + erff(t1[t] * 0.70710678118654752f));
    mv_gemm(out, v->out_hidden, t1, P, &v->l2, R);
    free(x); free(h); free(q); free(k); free(vv); free(att); free(g); free(u); free(cs);
    return R;
}

/* ---- loading from an mmproj (arch clip, projector pixtral) ---------------------------------- */
static int mv_read_f32(gguf_ctx *g, const char *name, int64_t want, float **dst) {
    const gguf_tensor *t = gguf_mark(g, name);
    if (!t) { fprintf(stderr, "mmproj: missing %s\n", name); return -1; }
    int64_t n = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    if (n != want) { fprintf(stderr, "mmproj %s: %lld values, %lld expected\n", name, (long long)n, (long long)want); return -1; }
    *dst = (float *)malloc((size_t)n * sizeof(float));
    if (!*dst || gguf_read_f32(g, t, *dst)) { fprintf(stderr, "mmproj %s: read failed\n", name); return -1; }
    return 0;
}
static int mv_read_mat(gguf_ctx *g, const char *name, int O, int I, MvMat *w) {
    const gguf_tensor *t = gguf_mark(g, name);
    if (!t) { fprintf(stderr, "mmproj: missing %s\n", name); return -1; }
    if (t->ne[0] != I || t->ne[1] != O || t->ne[2] != 1 || t->ne[3] != 1) {
        fprintf(stderr, "mmproj %s: [%lld x %lld], [%d x %d] expected\n", name, (long long)t->ne[1], (long long)t->ne[0], O, I);
        return -1;
    }
    w->type = t->type; w->O = O; w->I = I; w->rb = gq_row_bytes(t->type, I);
    if (w->rb <= 0) { fprintf(stderr, "mmproj %s: type %s not read here\n", name, gq_type_name(t->type)); return -1; }
    w->data = (uint8_t *)malloc((size_t)w->rb * O);
    if (!w->data || gguf_read(g, t, w->data, 0, (uint64_t)w->rb * O)) { fprintf(stderr, "mmproj %s: read failed\n", name); return -1; }
    return 0;
}
/* the tower of the mmproj at path, for a model of out_hidden whose RMSNorm eps is proj_eps */
static int mistral_vision_load(MistralVision *v, const char *path, int out_hidden, float proj_eps) {
    gguf_ctx g; memset(v, 0, sizeof *v);
    if (gguf_open(&g, path)) { fprintf(stderr, "mmproj %s: cannot open\n", path); return -1; }
    v->depth = (int)gguf_get_int(&g, "clip.vision.block_count", 0);
    v->hidden = (int)gguf_get_int(&g, "clip.vision.embedding_length", 0);
    v->heads = (int)gguf_get_int(&g, "clip.vision.attention.head_count", 0);
    v->inter = (int)gguf_get_int(&g, "clip.vision.feed_forward_length", 0);
    v->patch = (int)gguf_get_int(&g, "clip.vision.patch_size", 0);
    v->merge = (int)gguf_get_int(&g, "clip.vision.spatial_merge_size", 2);
    v->eps = (float)gguf_get_float(&g, "clip.vision.attention.layer_norm_epsilon", 1e-5);
    v->theta = (float)gguf_get_float(&g, "clip.rope.freq_base", 10000.0);
    v->in_ch = 3; v->out_hidden = out_hidden; v->proj_eps = proj_eps;
    v->rope_pairs = 1;   /* an mmproj: its converter permuted q and k (checked: the shapes of a test picture,
                          * and tools/mistral_vision_ref.py, which undoes it and rotates halves) */
    if (v->depth < 1 || v->hidden < 8 || v->heads < 1 || v->hidden % v->heads || v->inter < 1 || v->patch < 1 ||
        v->merge < 1 || (v->hidden / v->heads) % 8) {
        fprintf(stderr, "mmproj %s: not a Pixtral tower this engine reads\n", path); gguf_close(&g); return -1;
    }
    v->head_dim = v->hidden / v->heads;
    const int D = v->hidden, F = v->inter, PD = v->in_ch * v->patch * v->patch;
    int bad = mv_read_f32(&g, "v.patch_embd.weight", (int64_t)D * PD, &v->patch_w) ||
              mv_read_f32(&g, "v.pre_ln.weight", D, &v->pre_ln) ||
              mv_read_f32(&g, "mm.input_norm.weight", D, &v->mm_norm) ||
              mv_read_mat(&g, "mm.patch_merger.weight", D, D * v->merge * v->merge, &v->merger) ||
              mv_read_mat(&g, "mm.1.weight", out_hidden, D, &v->l1) ||
              mv_read_mat(&g, "mm.2.weight", out_hidden, out_hidden, &v->l2);
    v->layers = (MvLayer *)calloc((size_t)v->depth, sizeof *v->layers);
    for (int l = 0; !bad && v->layers && l < v->depth; l++) {
        MvLayer *L = &v->layers[l]; char nm[96];
        #define MVN(s) (snprintf(nm, sizeof nm, "v.blk.%d.%s.weight", l, s), nm)
        bad = mv_read_f32(&g, MVN("ln1"), D, &L->ln1) || mv_read_f32(&g, MVN("ln2"), D, &L->ln2) ||
              mv_read_mat(&g, MVN("attn_q"), D, D, &L->q) || mv_read_mat(&g, MVN("attn_k"), D, D, &L->k) ||
              mv_read_mat(&g, MVN("attn_v"), D, D, &L->v) || mv_read_mat(&g, MVN("attn_out"), D, D, &L->o) ||
              mv_read_mat(&g, MVN("ffn_gate"), F, D, &L->gate) || mv_read_mat(&g, MVN("ffn_up"), F, D, &L->up) ||
              mv_read_mat(&g, MVN("ffn_down"), D, F, &L->down);
        #undef MVN
    }
    gguf_close(&g);
    if (bad || !v->layers) return -1;
    v->ready = 1;
    return 0;
}

#endif /* MISTRAL_VISION_H */
