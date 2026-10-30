/* lfm3_vision.h -- LFM3-VL's vision tower and projector (LiquidAI/LFM3-VL), in C, f32.
 *
 * The patches tools/qwen38_image.py cuts (patch 16, merge 2, two frames of a still
 * image: C x T x 16 x 16 values a patch, in merge-block order) go to the rows the
 * language model reads where the prompt has its <|image|> tokens, one row per 2 x 2
 * block of patches. Transcribed from modeling_lfm3_vl.py:
 *
 *   patchifier  a linear map of each patch (no bias), then an RMSNorm
 *   layers      pre-norm, residual: a gated 2D convolution (in_proj to B | C | x,
 *               C * depthwise 3 x 3 conv(B * x) over the patch grid, zero padded,
 *               out_proj) or bidirectional grouped-query attention over the image
 *               (RMSNorm on each query and key head, an axial RoPE: head_dim / 4
 *               frequencies for the row, as many for the column, concatenated and
 *               doubled, rotate-half over the whole head); then a SwiGLU MLP
 *   projector   a LayerNorm (with bias) on each token, the four tokens of a merge
 *               block side by side, linear (bias), exact GELU, linear (bias)
 *
 * The tokens are in merge-block order (the four of a 2 x 2 block adjacent), which
 * is the order the projector groups and the convolution has to undo to find a
 * patch's neighbours: token (row, col) is block (row / 2) * (grid_w / 2) + col / 2,
 * cell (row % 2) * 2 + col % 2.
 *
 * The attention is quadratic in the patches (a 1024 x 768 picture is 3072 of them):
 * the gateway's token ceiling (LFM3_MAX_IMAGE_TOKENS) keeps a picture's cost bounded; a
 * head's N x N scores are one buffer, 64 MB at 4096 patches.
 */
#ifndef LFM3_VISION_H
#define LFM3_VISION_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int attn;                              /* 1: attention, 0: the gated convolution */
    const float *op_norm, *ffn_norm;       /* [D] */
    const float *in_proj, *spatial, *out_proj;   /* conv: [3D, D], [D, 3 x 3], [D, D] */
    const float *q, *k, *v, *o, *qn, *kn;  /* attention: [H hd, D], [KV hd, D] x 2, [D, H hd], [hd] x 2 */
    const float *w1, *w3, *w2;             /* MLP: [F, D] x 2, [D, F] */
} L3VLayer;

typedef struct {
    int depth, hidden, heads, kv_heads, head_dim, inter, patch, merge, temporal, in_ch;
    int out_hidden, proj_hidden;
    float eps, proj_eps, theta;
    const float *patchifier;               /* [D, in_ch * temporal * patch * patch] */
    const float *emb_norm;                 /* [D] */
    L3VLayer *layers;
    const float *ln_w, *ln_b;              /* the projector's LayerNorm, [D] */
    const float *l1_w, *l1_b, *l2_w, *l2_b;   /* [P, 4D], [P]; [out, P], [out] */
} Lfm3Vision;

/* y[s, o] = x[s, :] . w[o, :] over I, rows at strides ldy, ldx, ldw: 4 rows x 4 outputs a step
 * in 16 vector sums (GCC's vector extension: AVX2 or NEON, whichever the build has), blocks of
 * 32 outputs over the threads. matmul()'s one dot product at a time read its two rows from L2
 * for every output: the tower ran at 64 GFLOP/s on 14 cores, 27 s for a 1024-token picture. */
typedef float l3v_v8 __attribute__((vector_size(32)));
static inline float l3v_hsum(l3v_v8 v) { float a = 0.f; for (int i = 0; i < 8; i++) a += v[i]; return a; }
static void l3v_gemm(float *y, int ldy, const float *x, int ldx, const float *w, int ldw, int S, int I, int O) {
    const int OB = 32, nob = (O + OB - 1) / OB, nsb = (S + 63) / 64;
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int ob = 0; ob < nob; ob++)
        for (int sb = 0; sb < nsb; sb++) {
            const int o_end = (ob + 1) * OB < O ? (ob + 1) * OB : O, s_end = (sb + 1) * 64 < S ? (sb + 1) * 64 : S;
            for (int s = sb * 64; s < s_end; s += 4)
                for (int o = ob * OB; o < o_end; o += 4) {
                    const int ns = s_end - s < 4 ? s_end - s : 4, no = o_end - o < 4 ? o_end - o : 4;
                    l3v_v8 acc[4][4];
                    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] = (l3v_v8){0};
                    const float *xr[4], *wr[4];
                    for (int a = 0; a < 4; a++) xr[a] = x + (size_t)(s + (a < ns ? a : 0)) * ldx;
                    for (int b = 0; b < 4; b++) wr[b] = w + (size_t)(o + (b < no ? b : 0)) * ldw;
                    int i = 0;
                    for (; i + 8 <= I; i += 8) {
                        l3v_v8 xv[4], wv[4];
                        for (int a = 0; a < 4; a++) memcpy(&xv[a], xr[a] + i, 32);
                        for (int b = 0; b < 4; b++) memcpy(&wv[b], wr[b] + i, 32);
                        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) acc[a][b] += xv[a] * wv[b];
                    }
                    for (int a = 0; a < ns; a++)
                        for (int b = 0; b < no; b++) {
                            float t = l3v_hsum(acc[a][b]);
                            for (int k = i; k < I; k++) t += xr[a][k] * wr[b][k];
                            y[(size_t)(s + a) * ldy + o + b] = t;
                        }
                }
        }
}
static void l3v_mm(float *y, const float *x, const float *w, int S, int I, int O) { l3v_gemm(y, O, x, I, w, I, S, I, O); }

static void l3v_rms(float *x, const float *w, int rows, int n, float eps) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float *p = x + (size_t)r * n;
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)p[i] * p[i];
        float inv = 1.f / sqrtf((float)(ss / n) + eps);
        for (int i = 0; i < n; i++) p[i] = w[i] * (p[i] * inv);
    }
}
static int l3v_index(int row, int col, int grid_w, int m) {
    return ((row / m) * (grid_w / m) + col / m) * m * m + (row % m) * m + col % m;
}

/* the rows for one image: grid_h x grid_w patches in, (grid_h / merge) x (grid_w / merge)
 * rows of out_hidden out. Returns the rows written, -1 on a bad grid or no memory. */
static int lfm3_vision_forward(const Lfm3Vision *v, const float *patches, int grid_h, int grid_w, float *out) {
    const int m = v->merge, D = v->hidden, H = v->heads, KV = v->kv_heads, hd = v->head_dim, F = v->inter;
    if (grid_h <= 0 || grid_w <= 0 || grid_h % m || grid_w % m || hd % 4 || H % KV) return -1;
    const int N = grid_h * grid_w, PD = v->in_ch * v->temporal * v->patch * v->patch, rep = H / KV;
    float *x = malloc((size_t)N * D * sizeof(float)), *h = malloc((size_t)N * D * sizeof(float));
    float *big = malloc((size_t)N * (3 * D > 2 * F ? 3 * D : 2 * F) * sizeof(float));
    float *tmp = malloc((size_t)N * D * sizeof(float));
    float *cs = malloc((size_t)N * hd * sizeof(float)), *sn = malloc((size_t)N * hd * sizeof(float));
    int *nb = malloc((size_t)N * 9 * sizeof(int));
    if (!x || !h || !big || !tmp || !cs || !sn || !nb) { free(x); free(h); free(big); free(tmp); free(cs); free(sn); free(nb); return -1; }
    /* the patch embedding and its norm */
    l3v_mm(x, patches, v->patchifier, N, PD, D);
    l3v_rms(x, v->emb_norm, N, D, v->eps);
    /* each token's place, its 3 x 3 neighbours (-1 past the edge) and its rotary table */
    {
        const int q4 = hd / 4;
        for (int r = 0; r < grid_h; r++)
            for (int c = 0; c < grid_w; c++) {
                int t = l3v_index(r, c, grid_w, m);
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int rr = r + dy, cc = c + dx;
                        nb[(size_t)t * 9 + (dy + 1) * 3 + dx + 1] = rr < 0 || rr >= grid_h || cc < 0 || cc >= grid_w ? -1 : l3v_index(rr, cc, grid_w, m);
                    }
                for (int i = 0; i < hd / 2; i++) {
                    int j = i < q4 ? i : i - q4;
                    double inv = pow((double)v->theta, -(2.0 * j) / (double)(hd / 2));
                    double a = (i < q4 ? r : c) * inv;
                    cs[(size_t)t * hd + i] = cs[(size_t)t * hd + i + hd / 2] = (float)cos(a);
                    sn[(size_t)t * hd + i] = sn[(size_t)t * hd + i + hd / 2] = (float)sin(a);
                }
            }
    }
    for (int li = 0; li < v->depth; li++) {
        const L3VLayer *L = &v->layers[li];
        memcpy(h, x, (size_t)N * D * sizeof(float));
        l3v_rms(h, L->op_norm, N, D, v->eps);
        if (L->attn) {
            /* q | k | v in big: [N, H hd], [N, KV hd], [N, KV hd] */
            float *q = big, *k = big + (size_t)N * H * hd, *vv = k + (size_t)N * KV * hd;
            l3v_mm(q, h, L->q, N, D, H * hd);
            l3v_mm(k, h, L->k, N, D, KV * hd);
            l3v_mm(vv, h, L->v, N, D, KV * hd);
            l3v_rms(q, L->qn, N * H, hd, v->eps);
            l3v_rms(k, L->kn, N * KV, hd, v->eps);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < N; t++)
                for (int hh = 0; hh < H + KV; hh++) {
                    float *p = hh < H ? q + ((size_t)t * H + hh) * hd : k + ((size_t)t * KV + hh - H) * hd;
                    const float *c = cs + (size_t)t * hd, *s = sn + (size_t)t * hd;
                    for (int i = 0; i < hd / 2; i++) {
                        float a = p[i], b = p[i + hd / 2];
                        p[i] = a * c[i] - b * s[i];
                        p[i + hd / 2] = b * c[i + hd / 2] + a * s[i + hd / 2];
                    }
                }
            /* a head at a time: its scores q k^T as a GEMM over the key head, softmax by row, then
             * the value head (transposed to rows of positions) as the second GEMM's weights */
            const float scale = 1.f / sqrtf((float)hd);
            float *sc = malloc((size_t)N * N * sizeof(float)), *vt = malloc((size_t)hd * N * sizeof(float));
            if (!sc || !vt) { free(sc); free(vt); free(x); free(h); free(big); free(tmp); free(cs); free(sn); free(nb); return -1; }
            for (int hh = 0; hh < H; hh++) {
                const int kh = hh / rep;
                l3v_gemm(sc, N, q + (size_t)hh * hd, H * hd, k + (size_t)kh * hd, KV * hd, N, hd, N);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < N; i++) {
                    float *r = sc + (size_t)i * N, best = -INFINITY, sum = 0.f;
                    for (int j = 0; j < N; j++) { r[j] *= scale; if (r[j] > best) best = r[j]; }
                    for (int j = 0; j < N; j++) { r[j] = expf(r[j] - best); sum += r[j]; }
                    float inv = 1.f / sum;
                    for (int j = 0; j < N; j++) r[j] *= inv;
                }
                if (hh % rep == 0)
                    for (int j = 0; j < N; j++)
                        for (int d = 0; d < hd; d++) vt[(size_t)d * N + j] = vv[((size_t)j * KV + kh) * hd + d];
                l3v_gemm(tmp + (size_t)hh * hd, D, sc, N, vt, N, N, N, hd);
            }
            free(sc); free(vt);
            l3v_mm(h, tmp, L->o, N, H * hd, D);
        } else {
            /* B | C | x in big, the gate B * x into tmp, the convolution over the grid into h,
             * C * conv into tmp, out_proj into h */
            l3v_mm(big, h, L->in_proj, N, D, 3 * D);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < N; t++) {
                const float *b = big + (size_t)t * 3 * D;
                for (int c = 0; c < D; c++) tmp[(size_t)t * D + c] = b[c] * b[2 * D + c];
            }
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < N; t++) {
                const int *n9 = nb + (size_t)t * 9;
                float *o = h + (size_t)t * D;
                for (int c = 0; c < D; c++) {
                    const float *w = L->spatial + (size_t)c * 9;
                    float a = 0.f;
                    for (int e = 0; e < 9; e++) if (n9[e] >= 0) a += w[e] * tmp[(size_t)n9[e] * D + c];
                    o[c] = a;
                }
            }
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < N; t++) {
                const float *b = big + (size_t)t * 3 * D;
                for (int c = 0; c < D; c++) tmp[(size_t)t * D + c] = b[D + c] * h[(size_t)t * D + c];
            }
            l3v_mm(h, tmp, L->out_proj, N, D, D);
        }
        for (size_t i = 0; i < (size_t)N * D; i++) x[i] += h[i];
        /* the MLP */
        memcpy(h, x, (size_t)N * D * sizeof(float));
        l3v_rms(h, L->ffn_norm, N, D, v->eps);
        {
            float *g = big, *u = big + (size_t)N * F;
            l3v_mm(g, h, L->w1, N, D, F);
            l3v_mm(u, h, L->w3, N, D, F);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < (size_t)N * F; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
            l3v_mm(h, g, L->w2, N, F, D);
        }
        for (size_t i = 0; i < (size_t)N * D; i++) x[i] += h[i];
    }
    /* the projector: LayerNorm per token, four tokens to a row, linear, GELU, linear */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < N; t++) {
        float *p = x + (size_t)t * D;
        double mu = 0, var = 0;
        for (int i = 0; i < D; i++) mu += p[i];
        mu /= D;
        for (int i = 0; i < D; i++) var += (p[i] - mu) * (p[i] - mu);
        float inv = 1.f / sqrtf((float)(var / D) + v->proj_eps);
        for (int i = 0; i < D; i++) p[i] = (float)(p[i] - mu) * inv * v->ln_w[i] + v->ln_b[i];
    }
    const int R = N / (m * m), W4 = D * m * m, P = v->proj_hidden;
    float *y = malloc((size_t)R * P * sizeof(float));
    if (!y) { free(x); free(h); free(big); free(tmp); free(cs); free(sn); free(nb); return -1; }
    l3v_mm(y, x, v->l1_w, R, W4, P);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)R * P; i++) {
        float z = y[i] + v->l1_b[i % P];
        y[i] = 0.5f * z * (1.f + erff(z * 0.70710678118654752f));
    }
    l3v_mm(out, y, v->l2_w, R, P, v->out_hidden);
    for (int r = 0; r < R; r++) for (int i = 0; i < v->out_hidden; i++) out[(size_t)r * v->out_hidden + i] += v->l2_b[i];
    free(y); free(x); free(h); free(big); free(tmp); free(cs); free(sn); free(nb);
    return R;
}

#endif
