/* qwen38_vision.h -- la torre vision di Qwen3.8, in C.
 *
 * Prende le patch che tools/qwen38_image.py ha estratto e restituisce i token
 * immagine gia' proiettati nello spazio del modello di testo, pronti per essere
 * infilati dove il template scrive <|vision_start|><|image_pad|><|vision_end|>.
 *
 * Trascritta da Qwen4ExpVisionModel upstream, non dedotta dal paper. I quattro
 * punti in cui una torre ViT si sbaglia in silenzio, e come stanno qui:
 *
 *   patch_embed  e' una Conv3d con kernel UGUALE allo stride, quindi e' una
 *                matrice: ogni patch appiattita (C*T*P*P) per W^T piu' il bias.
 *                Trattarla come una convoluzione vera sarebbe lavoro sprecato e
 *                un'occasione in piu' di sbagliare l'ordine.
 *
 *   pos_embed    sono 48x48 posizioni APPRESE, interpolate bilinearmente sulla
 *                griglia dell'immagine con align_corners=true. Non e' una
 *                interpolazione qualsiasi: gli indici di sorgente si calcolano
 *                da (riga, colonna) ricavate in ordine a blocchi di merge, non
 *                in ordine raster. Usare l'ordine raster da' un'immagine che
 *                sembra funzionare e ha le posizioni mescolate.
 *
 *   rope         e' 2D e VIVE INSIEME alle posizioni apprese, non al posto
 *                loro. head_dim/2 frequenze per asse, concatenate e poi
 *                raddoppiate a head_dim.
 *
 *   merger       normalizza ogni token PRIMA di raggruppare, poi unisce 4 token
 *                adiacenti in un vettore da 4*hidden. L'ordine dei 4 e' quello
 *                in cui le patch sono gia' disposte, che e' il motivo per cui
 *                qwen38_image.py le emette a blocchi di merge.
 *
 * L'attenzione e' piena e quadratica nel numero di patch. Un'immagine 1080p ne
 * produce 8160, cioe' 66 milioni di coppie per testa per blocco: il tetto sui
 * token non e' una comodita', e' un requisito. Vedi Q38_MAX_IMAGE_TOKENS.
 */
#ifndef QWEN38_VISION_H
#define QWEN38_VISION_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const float *w, *b;        /* [out, in] riga-maggiore, bias [out] */
    int out, in;
} Q38Linear;

typedef struct {
    const float *w, *b;        /* LayerNorm: scala e traslazione, [n] */
} Q38Norm;

typedef struct {
    Q38Norm norm1, norm2;
    Q38Linear qkv, proj, fc1, fc2;
} Q38VBlock;

typedef struct {
    int depth, hidden, heads, head_dim, inter, patch, merge, temporal, in_ch;
    int out_hidden, num_pos, side;      /* side = sqrt(num_pos) */
    float eps;
    Q38Linear patch_embed;
    const float *pos_embed;             /* [num_pos, hidden] */
    Q38VBlock *blocks;
    Q38Norm merger_norm;
    Q38Linear merger_fc1, merger_fc2;
} Q38Vision;

/* ---- primitive ---------------------------------------------------------- */

static void q38v_linear(float *out, const float *in, const Q38Linear *l, int rows)
{
    for (int r = 0; r < rows; r++) {
        const float *x = in + (size_t)r * l->in;
        float *y = out + (size_t)r * l->out;
        for (int o = 0; o < l->out; o++) {
            const float *w = l->w + (size_t)o * l->in;
            float acc = l->b ? l->b[o] : 0.f;
            for (int i = 0; i < l->in; i++) acc += w[i] * x[i];
            y[o] = acc;
        }
    }
}

static void q38v_layernorm(float *out, const float *in, const Q38Norm *n, int rows,
                           int width, float eps)
{
    for (int r = 0; r < rows; r++) {
        const float *x = in + (size_t)r * width;
        float *y = out + (size_t)r * width;
        double mean = 0.0;
        for (int i = 0; i < width; i++) mean += x[i];
        mean /= width;
        double var = 0.0;
        for (int i = 0; i < width; i++) { double d = x[i] - mean; var += d * d; }
        var /= width;
        float inv = (float)(1.0 / sqrt(var + eps));
        for (int i = 0; i < width; i++)
            y[i] = ((float)(x[i] - mean)) * inv * n->w[i] + n->b[i];
    }
}

/* DUE gelu, e non e' una svista del riferimento: i blocchi usano
 * ACT2FN[hidden_act], che per questo checkpoint e' gelu_pytorch_tanh, mentre il
 * merger istanzia nn.GELU(), cioe' quella ESATTA con erf. Usarne una sola le fa
 * combaciare quasi -- lo scarto misurato sul merger era 4.6e-3, invisibile senza
 * un oracolo e sufficiente a spostare i token immagine. */
static float q38v_gelu_tanh(float x)
{
    const float k = 0.7978845608028654f;          /* sqrt(2/pi) */
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static float q38v_gelu_exact(float x)
{
    return 0.5f * x * (1.f + erff(x * 0.70710678118654752f));   /* 1/sqrt(2) */
}

/* ---- posizioni ---------------------------------------------------------- */

/* (riga, colonna) della patch `i`, in ordine a blocchi di merge. E' l'inverso
 * esatto del riordinamento che qwen38_image.py applica alle patch. */
static void q38v_row_col(int i, int grid_w, int merge, int *row, int *col)
{
    int blocks_w = grid_w / merge;
    int in_col = i % merge;
    int in_row = (i / merge) % merge;
    int block_col = (i / (merge * merge)) % blocks_w;
    int block_row = i / (merge * merge * blocks_w);
    *row = block_row * merge + in_row;
    *col = block_col * merge + in_col;
}

/* I due tap bilineari e i loro pesi su un asse, align_corners=true. */
static void q38v_taps(int index, int size, int side, int *tap, float *weight)
{
    float src = (float)index * (float)(side - 1) / (float)(size > 1 ? size - 1 : 1);
    float floored = floorf(src);
    for (int t = 0; t < 2; t++) {
        int raw = (int)floored + t;
        tap[t] = raw < 0 ? 0 : (raw > side - 1 ? side - 1 : raw);
        float distance = fabsf(src - floored - (float)t);
        float w = 1.f - distance;
        weight[t] = w < 0.f ? 0.f : w;
    }
}

/* ---- il forward --------------------------------------------------------- */

/* patches: [n, in_ch*temporal*patch*patch].  out: [n/merge^2, out_hidden].
 * Ritorna il numero di token prodotti, o -1 se la griglia non e' coerente. */
static int q38_vision_forward_dbg(const Q38Vision *v, const float *patches,
                              int grid_h, int grid_w, float *out, float *last_hidden)
{
    if (grid_h <= 0 || grid_w <= 0) return -1;
    if (grid_h % v->merge || grid_w % v->merge) return -1;    /* il merger 2x2 non chiuderebbe */
    const int n = grid_h * grid_w, H = v->hidden, D = v->head_dim, nh = v->heads;
    const int unit = v->merge * v->merge;

    float *x   = (float *)calloc((size_t)n * H, sizeof(float));
    float *tmp = (float *)calloc((size_t)n * H, sizeof(float));
    float *qkv = (float *)calloc((size_t)n * 3 * H, sizeof(float));
    float *att = (float *)calloc((size_t)n * H, sizeof(float));
    float *mlp = (float *)calloc((size_t)n * v->inter, sizeof(float));
    float *cos_t = (float *)calloc((size_t)n * D, sizeof(float));
    float *sin_t = (float *)calloc((size_t)n * D, sizeof(float));
    float *scores = (float *)calloc((size_t)n, sizeof(float));
    if (!x || !tmp || !qkv || !att || !mlp || !cos_t || !sin_t || !scores) {
        free(x); free(tmp); free(qkv); free(att); free(mlp);
        free(cos_t); free(sin_t); free(scores); return -1;
    }

    /* 1. patch embedding, e le posizioni apprese sommate subito dopo. */
    q38v_linear(x, patches, &v->patch_embed, n);
    for (int i = 0; i < n; i++) {
        int row, col; q38v_row_col(i, grid_w, v->merge, &row, &col);
        int ht[2], wt[2]; float hw[2], ww[2];
        q38v_taps(row, grid_h, v->side, ht, hw);
        q38v_taps(col, grid_w, v->side, wt, ww);
        float *dst = x + (size_t)i * H;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++) {
                float weight = hw[a] * ww[b];
                if (weight == 0.f) continue;
                const float *src = v->pos_embed + (size_t)(ht[a] * v->side + wt[b]) * H;
                for (int c = 0; c < H; c++) dst[c] += weight * src[c];
            }
    }

    /* 2. RoPE 2D: meta' delle frequenze sulla riga, meta' sulla colonna, poi il
     *    vettore viene raddoppiato per coprire head_dim. */
    {
        const int half = D / 2, freqs = half / 2;
        for (int i = 0; i < n; i++) {
            int row, col; q38v_row_col(i, grid_w, v->merge, &row, &col);
            float *c = cos_t + (size_t)i * D, *s = sin_t + (size_t)i * D;
            for (int f = 0; f < freqs; f++) {
                float inv = 1.f / powf(10000.f, (float)(2 * f) / (float)half);
                float ah = (float)row * inv, aw = (float)col * inv;
                c[f] = cosf(ah);          s[f] = sinf(ah);
                c[freqs + f] = cosf(aw);  s[freqs + f] = sinf(aw);
            }
            for (int k = 0; k < half; k++) { c[half + k] = c[k]; s[half + k] = s[k]; }
        }
    }

    /* 3. i blocchi. */
    for (int layer = 0; layer < v->depth; layer++) {
        const Q38VBlock *blk = &v->blocks[layer];
        q38v_layernorm(tmp, x, &blk->norm1, n, H, v->eps);
        q38v_linear(qkv, tmp, &blk->qkv, n);

        /* rotate_half su q e k, testa per testa. Il layout di qkv e'
         * [token][3][heads][head_dim], come lo produce la reshape upstream. */
        for (int i = 0; i < n; i++) {
            const float *c = cos_t + (size_t)i * D, *s = sin_t + (size_t)i * D;
            for (int part = 0; part < 2; part++)                 /* q, k -- non v */
                for (int h = 0; h < nh; h++) {
                    float *p = qkv + ((size_t)i * 3 + part) * H + (size_t)h * D;
                    for (int d = 0; d < D / 2; d++) {
                        float a = p[d], b = p[d + D / 2];
                        p[d]         = a * c[d]         - b * s[d];
                        p[d + D / 2] = b * c[d + D / 2] + a * s[d + D / 2];
                    }
                }
        }

        const float scale = 1.f / sqrtf((float)D);
        memset(att, 0, (size_t)n * H * sizeof(float));
        for (int h = 0; h < nh; h++)
            for (int i = 0; i < n; i++) {
                const float *q = qkv + (size_t)i * 3 * H + (size_t)h * D;
                float best = -INFINITY;
                for (int j = 0; j < n; j++) {
                    const float *k = qkv + ((size_t)j * 3 + 1) * H + (size_t)h * D;
                    float acc = 0.f;
                    for (int d = 0; d < D; d++) acc += q[d] * k[d];
                    scores[j] = acc * scale;
                    if (scores[j] > best) best = scores[j];
                }
                float sum = 0.f;
                for (int j = 0; j < n; j++) { scores[j] = expf(scores[j] - best); sum += scores[j]; }
                float inv = 1.f / sum;
                float *dst = att + (size_t)i * H + (size_t)h * D;
                for (int j = 0; j < n; j++) {
                    float weight = scores[j] * inv;
                    const float *val = qkv + ((size_t)j * 3 + 2) * H + (size_t)h * D;
                    for (int d = 0; d < D; d++) dst[d] += weight * val[d];
                }
            }

        q38v_linear(tmp, att, &blk->proj, n);
        for (size_t i = 0; i < (size_t)n * H; i++) x[i] += tmp[i];

        q38v_layernorm(tmp, x, &blk->norm2, n, H, v->eps);
        q38v_linear(mlp, tmp, &blk->fc1, n);
        for (size_t i = 0; i < (size_t)n * v->inter; i++) mlp[i] = q38v_gelu_tanh(mlp[i]);
        q38v_linear(tmp, mlp, &blk->fc2, n);
        for (size_t i = 0; i < (size_t)n * H; i++) x[i] += tmp[i];
    }

    if (last_hidden) memcpy(last_hidden, x, (size_t)n * H * sizeof(float));

    /* 4. merger: normalizza ogni token, poi ne unisce merge^2 adiacenti. */
    q38v_layernorm(tmp, x, &v->merger_norm, n, H, v->eps);
    {
        const int tokens = n / unit, wide = H * unit;
        float *grouped = (float *)calloc((size_t)tokens * wide, sizeof(float));
        float *hidden = (float *)calloc((size_t)tokens * wide, sizeof(float));
        if (!grouped || !hidden) {
            free(grouped); free(hidden);
            free(x); free(tmp); free(qkv); free(att); free(mlp);
            free(cos_t); free(sin_t); free(scores); return -1;
        }
        memcpy(grouped, tmp, (size_t)tokens * wide * sizeof(float));
        q38v_linear(hidden, grouped, &v->merger_fc1, tokens);
        for (size_t i = 0; i < (size_t)tokens * wide; i++) hidden[i] = q38v_gelu_exact(hidden[i]);
        q38v_linear(out, hidden, &v->merger_fc2, tokens);
        free(grouped); free(hidden);
        free(x); free(tmp); free(qkv); free(att); free(mlp);
        free(cos_t); free(sin_t); free(scores);
        return tokens;
    }
}

static int q38_vision_forward(const Q38Vision *v, const float *patches,
                              int grid_h, int grid_w, float *out)
{ return q38_vision_forward_dbg(v, patches, grid_h, grid_w, out, NULL); }

#endif /* QWEN38_VISION_H */
