#ifndef COLIBRI_VISION_TOWER_H
#define COLIBRI_VISION_TOWER_H

/*
 * Vision transformer tower, shared by the engines whose checkpoint carries one.
 *
 * GLM-5.3-Flash is the first, and the shape is the common one for this
 * generation of vision-language models: patches are projected, run through a
 * plain pre-norm transformer with 2-D rotary positions, then merged in m x m
 * spatial blocks and projected into the language model's width. Nothing here is
 * GLM-specific except the SwiGLU clamp, which is passed in.
 *
 * Two details decide whether the output is right, and both are easy to get
 * subtly wrong:
 *
 *   - Token order is BLOCK-MAJOR over the m x m merge blocks, not row-major
 *     over the image. The reference builds position ids as
 *     reshape(h/m, m, w/m, m).transpose(1,2), so the tokens of one merge block
 *     are adjacent. The final merge relies on that adjacency, which is why the
 *     two must agree: get the order wrong and the picture is still "processed",
 *     just scrambled in a way no assertion would catch.
 *   - The 2-D rotary rotates each head's first half against the h axis and the
 *     second against w: freqs are [h*inv_freq, w*inv_freq] concatenated, then
 *     duplicated to the full head width, exactly like the text-side RoPE.
 *
 * Weights are plain f32 here. Quantized towers can come later; a 563 M
 * parameter tower is about 1 GB in bf16, which is not where this model's memory
 * goes.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int depth, hidden, heads, head_dim, intermediate;
    int patch, temporal, merge, in_channels;
    int out_hidden, proj_intermediate;
    float eps, swiglu_limit, rope_theta;
} ColiVisionConfig;

/* Row-major weights, transformers layout: a [out, in] matrix multiplies a
 * column vector of `in` values. Biases may be NULL where the checkpoint has
 * none. */
typedef struct {
    const float *norm1, *norm2;
    const float *qkv_w, *qkv_b;          /* [3*hidden, hidden], [3*hidden] */
    const float *q_norm, *k_norm;        /* [head_dim] each */
    const float *proj_w, *proj_b;        /* [hidden, hidden], [hidden] */
    const float *gate_w, *gate_b;        /* [intermediate, hidden] */
    const float *up_w, *up_b;
    const float *down_w, *down_b;        /* [hidden, intermediate] */
} ColiVisionBlock;

typedef struct {
    ColiVisionConfig config;
    const float *patch_w, *patch_b;      /* [hidden, in_channels*temporal*patch*patch] */
    const ColiVisionBlock *blocks;       /* depth entries */
    const float *post_norm;              /* [hidden] */
    const float *down_w, *down_b;        /* Conv2d [out_hidden, hidden, merge, merge] */
    const float *merger_proj;            /* [out_hidden, out_hidden] */
    const float *merger_norm_w, *merger_norm_b;
    const float *merger_gate, *merger_up;/* [proj_intermediate, out_hidden] */
    const float *merger_down;            /* [out_hidden, proj_intermediate] */
} ColiVisionTower;

static inline void coli_vision_matvec(float *out, const float *w, const float *b,
                                      const float *in, int rows, int columns) {
    for (int row = 0; row < rows; row++) {
        float sum = b ? b[row] : 0.0f;
        const float *weight = w + (size_t)row * columns;
        for (int column = 0; column < columns; column++) sum += weight[column] * in[column];
        out[row] = sum;
    }
}

static inline void coli_vision_rmsnorm(float *out, const float *in, const float *weight,
                                       int dimension, float eps) {
    float square = 0.0f;
    for (int i = 0; i < dimension; i++) square += in[i] * in[i];
    float inverse = 1.0f / sqrtf(square / dimension + eps);
    for (int i = 0; i < dimension; i++) out[i] = in[i] * inverse * weight[i];
}

/* Classic LayerNorm (mean and variance), which is what the patch merger uses -
 * the rest of the tower is RMSNorm. */
static inline void coli_vision_layernorm(float *out, const float *in, const float *weight,
                                         const float *bias, int dimension, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < dimension; i++) mean += in[i];
    mean /= dimension;
    float variance = 0.0f;
    for (int i = 0; i < dimension; i++) variance += (in[i] - mean) * (in[i] - mean);
    variance /= dimension;
    float inverse = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < dimension; i++)
        out[i] = (in[i] - mean) * inverse * weight[i] + (bias ? bias[i] : 0.0f);
}

static inline float coli_vision_silu(float value) {
    return value / (1.0f + expf(-value));
}

/* Exact GELU, the torch nn.GELU() default: erf-based, not the tanh
 * approximation. On a two-block tower the difference is invisible; on a
 * 24-block one it is not, and the two are not interchangeable. */
static inline float coli_vision_gelu(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.70710678118654752f));
}

static inline float coli_vision_clamp(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

/* Output tokens for a grid: one per m x m block. */
static inline int coli_vision_output_tokens(const ColiVisionConfig *config,
                                            int grid_h, int grid_w) {
    if (config->merge < 1 || grid_h % config->merge || grid_w % config->merge) return -1;
    return (grid_h / config->merge) * (grid_w / config->merge);
}

/* pixels: [grid_h*grid_w, in_channels*temporal*patch*patch], already in the
 * block-major token order the reference produces.
 * out: [output_tokens, out_hidden]. Returns 0, or -1 on bad arguments or
 * allocation failure. */
static inline int coli_vision_forward(float *out, const ColiVisionTower *tower,
                                      const float *pixels, int grid_h, int grid_w) {
    const ColiVisionConfig *c = &tower->config;
    if (!out || !tower || !pixels || grid_h < 1 || grid_w < 1 ||
        c->heads < 1 || c->hidden != c->heads * c->head_dim ||
        c->head_dim % 4 || coli_vision_output_tokens(c, grid_h, grid_w) < 0)
        return -1;

    const int tokens = grid_h * grid_w;
    const int hidden = c->hidden, heads = c->heads, hd = c->head_dim;
    const int patch_width = c->in_channels * c->temporal * c->patch * c->patch;
    const int rot = hd / 2;                   /* rotary width before duplication */
    const int half = rot / 2;                 /* entries per spatial axis */

    float *state = malloc((size_t)tokens * hidden * sizeof(float));
    float *cos_table = malloc((size_t)tokens * hd * sizeof(float));
    float *sin_table = malloc((size_t)tokens * hd * sizeof(float));
    float *qkv = malloc((size_t)tokens * 3 * hidden * sizeof(float));
    float *scores = malloc((size_t)tokens * sizeof(float));
    float *scratch = malloc((size_t)(hidden > c->intermediate ? hidden : c->intermediate) *
                            sizeof(float));
    float *branch = malloc((size_t)tokens * hidden * sizeof(float));
    if (!state || !cos_table || !sin_table || !qkv || !scores || !scratch || !branch) {
        free(branch); free(scratch); free(scores); free(qkv);
        free(sin_table); free(cos_table); free(state);
        return -1;
    }

    /* Patch embedding: the checkpoint's Conv3d has kernel == stride, so it is a
     * matrix applied to each flattened patch, not a convolution. */
    for (int t = 0; t < tokens; t++)
        coli_vision_matvec(state + (size_t)t * hidden, tower->patch_w, tower->patch_b,
                           pixels + (size_t)t * patch_width, hidden, patch_width);

    /* 2-D rotary tables, in the same block-major order as the tokens. */
    {
        int m = c->merge, index = 0;
        for (int bh = 0; bh < grid_h / m; bh++)
            for (int bw = 0; bw < grid_w / m; bw++)
                for (int ih = 0; ih < m; ih++)
                    for (int iw = 0; iw < m; iw++, index++) {
                        float position[2] = { (float)(bh * m + ih), (float)(bw * m + iw) };
                        float *cosine = cos_table + (size_t)index * hd;
                        float *sine = sin_table + (size_t)index * hd;
                        for (int axis = 0; axis < 2; axis++)
                            for (int j = 0; j < half; j++) {
                                float inv_freq = powf(c->rope_theta,
                                                      -(float)(2 * j) / (float)rot);
                                float angle = position[axis] * inv_freq;
                                int slot = axis * half + j;
                                cosine[slot] = cosf(angle);
                                sine[slot] = sinf(angle);
                                cosine[slot + rot] = cosine[slot];   /* cat(emb, emb) */
                                sine[slot + rot] = sine[slot];
                            }
                    }
    }

    for (int layer = 0; layer < c->depth; layer++) {
        const ColiVisionBlock *block = &tower->blocks[layer];
        /* ---- attention ---- */
        for (int t = 0; t < tokens; t++) {
            coli_vision_rmsnorm(scratch, state + (size_t)t * hidden, block->norm1, hidden, c->eps);
            coli_vision_matvec(qkv + (size_t)t * 3 * hidden, block->qkv_w, block->qkv_b,
                               scratch, 3 * hidden, hidden);
        }
        /* q/k head norms, then rotate. The reference reshapes qkv to
         * [seq, 3, heads, head_dim], so q, k and v are contiguous thirds. */
        for (int t = 0; t < tokens; t++) {
            float *row = qkv + (size_t)t * 3 * hidden;
            const float *cosine = cos_table + (size_t)t * hd;
            const float *sine = sin_table + (size_t)t * hd;
            for (int h = 0; h < heads; h++) {
                float *q = row + (size_t)h * hd;
                float *k = row + hidden + (size_t)h * hd;
                coli_vision_rmsnorm(q, q, block->q_norm, hd, c->eps);
                coli_vision_rmsnorm(k, k, block->k_norm, hd, c->eps);
                for (int pass = 0; pass < 2; pass++) {
                    float *vec = pass ? k : q;
                    float rotated[512];
                    int limit = hd < 512 ? hd : 512;
                    for (int i = 0; i < limit; i++)
                        rotated[i] = i < hd / 2 ? -vec[i + hd / 2] : vec[i - hd / 2];
                    for (int i = 0; i < limit; i++)
                        vec[i] = vec[i] * cosine[i] + rotated[i] * sine[i];
                }
            }
        }
        const float scaling = 1.0f / sqrtf((float)hd);
        for (int t = 0; t < tokens; t++) {
            float *result = branch + (size_t)t * hidden;
            for (int h = 0; h < heads; h++) {
                const float *q = qkv + (size_t)t * 3 * hidden + (size_t)h * hd;
                float maximum = -INFINITY;
                for (int s = 0; s < tokens; s++) {
                    const float *k = qkv + (size_t)s * 3 * hidden + hidden + (size_t)h * hd;
                    float dot = 0.0f;
                    for (int i = 0; i < hd; i++) dot += q[i] * k[i];
                    scores[s] = dot * scaling;
                    if (scores[s] > maximum) maximum = scores[s];
                }
                float total = 0.0f;
                for (int s = 0; s < tokens; s++) {
                    scores[s] = expf(scores[s] - maximum);
                    total += scores[s];
                }
                float *slot = result + (size_t)h * hd;
                for (int i = 0; i < hd; i++) slot[i] = 0.0f;
                for (int s = 0; s < tokens; s++) {
                    const float *v = qkv + (size_t)s * 3 * hidden + 2 * hidden + (size_t)h * hd;
                    float weight = scores[s] / total;
                    for (int i = 0; i < hd; i++) slot[i] += weight * v[i];
                }
            }
        }
        for (int t = 0; t < tokens; t++) {
            coli_vision_matvec(scratch, block->proj_w, block->proj_b,
                               branch + (size_t)t * hidden, hidden, hidden);
            for (int i = 0; i < hidden; i++) state[(size_t)t * hidden + i] += scratch[i];
        }
        /* ---- MLP ---- */
        for (int t = 0; t < tokens; t++) {
            float *gate = malloc((size_t)c->intermediate * 2 * sizeof(float));
            if (!gate) { free(branch); free(scratch); free(scores); free(qkv);
                         free(sin_table); free(cos_table); free(state); return -1; }
            float *up = gate + c->intermediate;
            coli_vision_rmsnorm(scratch, state + (size_t)t * hidden, block->norm2, hidden, c->eps);
            coli_vision_matvec(gate, block->gate_w, block->gate_b, scratch, c->intermediate, hidden);
            coli_vision_matvec(up, block->up_w, block->up_b, scratch, c->intermediate, hidden);
            for (int i = 0; i < c->intermediate; i++) {
                float g = gate[i] > c->swiglu_limit ? c->swiglu_limit : gate[i];
                float u = coli_vision_clamp(up[i], -c->swiglu_limit, c->swiglu_limit);
                gate[i] = coli_vision_silu(g) * u;
            }
            coli_vision_matvec(scratch, block->down_w, block->down_b, gate, hidden, c->intermediate);
            for (int i = 0; i < hidden; i++) state[(size_t)t * hidden + i] += scratch[i];
            free(gate);
        }
    }

    for (int t = 0; t < tokens; t++)
        coli_vision_rmsnorm(state + (size_t)t * hidden, state + (size_t)t * hidden,
                            tower->post_norm, hidden, c->eps);

    /* Spatial merge. The reference views the tokens as [-1, m, m, hidden],
     * permutes to [N, hidden, m, m] and applies a Conv2d whose kernel covers
     * the whole block, so each output token is one dot product over the
     * block's m*m*hidden values - which only lines up because the tokens of a
     * block are adjacent (see the header comment). */
    const int m = c->merge, blocks = coli_vision_output_tokens(c, grid_h, grid_w);
    float *merged = malloc((size_t)c->out_hidden * sizeof(float));
    float *gated = malloc((size_t)c->proj_intermediate * 2 * sizeof(float));
    if (!merged || !gated) {
        free(gated); free(merged); free(branch); free(scratch); free(scores);
        free(qkv); free(sin_table); free(cos_table); free(state);
        return -1;
    }
    for (int n = 0; n < blocks; n++) {
        for (int o = 0; o < c->out_hidden; o++) {
            float sum = tower->down_b ? tower->down_b[o] : 0.0f;
            for (int ch = 0; ch < hidden; ch++)
                for (int kh = 0; kh < m; kh++)
                    for (int kw = 0; kw < m; kw++) {
                        size_t weight_index = (((size_t)o * hidden + ch) * m + kh) * m + kw;
                        size_t token = (size_t)n * m * m + (size_t)kh * m + kw;
                        sum += tower->down_w[weight_index] * state[token * hidden + ch];
                    }
            merged[o] = sum;
        }
        /* Patch merger: proj, LayerNorm, GELU, then a clamped SwiGLU. */
        float *tmp = branch;                                  /* reuse: >= out_hidden */
        coli_vision_matvec(tmp, tower->merger_proj, NULL, merged, c->out_hidden, c->out_hidden);
        coli_vision_layernorm(merged, tmp, tower->merger_norm_w, tower->merger_norm_b,
                              c->out_hidden, 1e-5f);
        for (int i = 0; i < c->out_hidden; i++) merged[i] = coli_vision_gelu(merged[i]);
        float *up = gated + c->proj_intermediate;
        coli_vision_matvec(gated, tower->merger_gate, NULL, merged, c->proj_intermediate, c->out_hidden);
        coli_vision_matvec(up, tower->merger_up, NULL, merged, c->proj_intermediate, c->out_hidden);
        for (int i = 0; i < c->proj_intermediate; i++) {
            float g = gated[i] > c->swiglu_limit ? c->swiglu_limit : gated[i];
            float u = coli_vision_clamp(up[i], -c->swiglu_limit, c->swiglu_limit);
            gated[i] = coli_vision_silu(g) * u;
        }
        coli_vision_matvec(out + (size_t)n * c->out_hidden, tower->merger_down, NULL,
                           gated, c->out_hidden, c->proj_intermediate);
    }

    free(gated);
    free(merged);
    free(branch);
    free(scratch);
    free(scores);
    free(qkv);
    free(sin_table);
    free(cos_table);
    free(state);
    return 0;
}

#endif /* COLIBRI_VISION_TOWER_H */
