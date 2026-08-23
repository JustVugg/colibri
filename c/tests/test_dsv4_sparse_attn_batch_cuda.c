/* Validates dsv4_cuda_sparse_attn_batch and dsv4_cuda_matmul_bf16_batch
 * against inline CPU references that mirror coli_v4_sparse_attention_ref and
 * the compressor's bf16 projection loop. Build with cl against
 * coli_cuda_dsv4_dg.lib. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../backend_cuda_dsv4.h"

static float bf16_round(float x) {
    uint32_t bits;
    memcpy(&bits, &x, 4);
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7fffu + lsb;
    bits &= 0xffff0000u;
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

static float bf16_decode(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

static float frand(unsigned *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return ((*seed >> 8) & 0xffff) / 65536.0f - 0.5f;
}

/* CPU reference: same contract as the GPU kernel (window range + compressed
 * prefix over a linear slab), same math as coli_v4_sparse_attention_ref. */
static void ref_token(float *out, const float *q, const float *vals,
                      const float *sinks, int wo, int wn, int cn,
                      int comp_base, int heads, int dim, float scale) {
    float *scores = malloc((size_t)(wn + cn) * sizeof(*scores));
    for (int h = 0; h < heads; h++) {
        const float *qh = q + (size_t)h * dim;
        float maximum = -INFINITY;
        for (int r = 0; r < wn + cn; r++) {
            const float *v = vals + (size_t)(r < wn ? wo + r
                                                    : comp_base + r - wn) * dim;
            float s = 0.0f;
            for (int d = 0; d < dim; d++) s += qh[d] * v[d];
            s *= scale;
            scores[r] = s;
            if (s > maximum) maximum = s;
        }
        float den = expf(sinks[h] - maximum);
        float *oh = out + (size_t)h * dim;
        memset(oh, 0, (size_t)dim * sizeof(*oh));
        for (int r = 0; r < wn + cn; r++) {
            float p = expf(scores[r] - maximum);
            den += p;
            p = bf16_round(p);
            const float *v = vals + (size_t)(r < wn ? wo + r
                                                    : comp_base + r - wn) * dim;
            for (int d = 0; d < dim; d++) oh[d] += p * v[d];
        }
        for (int d = 0; d < dim; d++) oh[d] = bf16_round(oh[d] / den);
    }
    free(scores);
}

int main(void) {
    enum { HEADS = 64, DIM = 512, WINDOW = 128, T = 64, START = 200,
           COMP = 60 };
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) return 1;
    unsigned seed = 12345;

    /* ---- sparse attention ---- */
    int first = START - WINDOW + 1;
    int pre = START - first;
    int value_rows = pre + T + COMP;
    int comp_base = pre + T;
    float *q = malloc((size_t)T * HEADS * DIM * sizeof(float));
    float *vals = malloc((size_t)value_rows * DIM * sizeof(float));
    float *sinks = malloc(HEADS * sizeof(float));
    int *meta = malloc((size_t)T * 3 * sizeof(int));
    float *got = malloc((size_t)T * HEADS * DIM * sizeof(float));
    float *want = malloc((size_t)HEADS * DIM * sizeof(float));
    if (!q || !vals || !sinks || !meta || !got || !want) return 2;
    for (size_t i = 0; i < (size_t)T * HEADS * DIM; i++) q[i] = frand(&seed);
    for (size_t i = 0; i < (size_t)value_rows * DIM; i++) vals[i] = frand(&seed);
    for (int h = 0; h < HEADS; h++) sinks[h] = frand(&seed) * 4.0f;
    for (int t = 0; t < T; t++) {
        int pos = START + t;
        int wfirst = pos - WINDOW + 1;
        if (wfirst < 0) wfirst = 0;
        meta[3 * t] = wfirst - first;
        meta[3 * t + 1] = pos - wfirst + 1;
        meta[3 * t + 2] = COMP * (t + 1) / T; /* growing prefix */
    }
    float scale = 1.0f / sqrtf((float)DIM);
    if (!dsv4_cuda_sparse_attn_batch(dev, q, vals, sinks, meta, value_rows,
                                     comp_base, HEADS, DIM, T, scale, got)) {
        fprintf(stderr, "sparse_attn_batch refused\n");
        return 3;
    }
    double worst = 0.0;
    for (int t = 0; t < T; t++) {
        ref_token(want, q + (size_t)t * HEADS * DIM, vals, sinks,
                  meta[3 * t], meta[3 * t + 1], meta[3 * t + 2],
                  comp_base, HEADS, DIM, scale);
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++) {
            double diff = fabs((double)want[i] -
                               got[(size_t)t * HEADS * DIM + i]);
            if (diff > worst) worst = diff;
            if (diff > 2e-3) {
                fprintf(stderr, "attention mismatch t=%d i=%zu want=%g got=%g\n",
                        t, i, want[i], got[(size_t)t * HEADS * DIM + i]);
                return 4;
            }
        }
    }
    printf("sparse attention batch: max |diff| = %g\n", worst);

    /* ---- cached sparse attention (ring + separate chunk buffer) ----
     * Only the historic rows go into the ring; the chunk rows ride along as
     * their own buffer exactly like the engine passes them (appending the
     * chunk first would alias the oldest window rows once positions exceed
     * the window). */
    enum { LAYER = 7, TOPK = 24 };
    if (!dsv4_cuda_kv_ring_append(dev, LAYER, vals, first, pre, WINDOW,
                                  DIM) ||
        !dsv4_cuda_kv_comp_append(dev, LAYER, vals + (size_t)comp_base * DIM,
                                  0, COMP, DIM)) {
        fprintf(stderr, "kv cache seed refused\n");
        return 9;
    }
    if (!dsv4_cuda_sparse_attn_batch_cached(dev, LAYER, q,
                                            vals + (size_t)pre * DIM, START,
                                            sinks, meta, first, COMP, HEADS,
                                            DIM, T, scale, got)) {
        fprintf(stderr, "sparse_attn_batch_cached refused\n");
        return 13;
    }
    worst = 0.0;
    for (int t = 0; t < T; t++) {
        ref_token(want, q + (size_t)t * HEADS * DIM, vals, sinks,
                  meta[3 * t], meta[3 * t + 1], meta[3 * t + 2],
                  comp_base, HEADS, DIM, scale);
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++) {
            double diff = fabs((double)want[i] -
                               got[(size_t)t * HEADS * DIM + i]);
            if (diff > worst) worst = diff;
            if (diff > 2e-3) {
                fprintf(stderr, "cached attention mismatch t=%d i=%zu want=%g got=%g\n",
                        t, i, want[i], got[(size_t)t * HEADS * DIM + i]);
                return 14;
            }
        }
    }
    printf("cached sparse attention: max |diff| = %g\n", worst);

    /* ---- cached indexed sparse attention (subset selection) ---- */
    int *sel = malloc((size_t)T * TOPK * sizeof(*sel));
    int *meta_idx = malloc((size_t)T * 3 * sizeof(*meta_idx));
    int *pool = malloc((size_t)COMP * sizeof(*pool));
    float *vals_g = malloc((size_t)(WINDOW + TOPK) * DIM * sizeof(*vals_g));
    if (!sel || !meta_idx || !pool || !vals_g) return 10;
    for (int t = 0; t < T; t++) {
        int cn_full = meta[3 * t + 2];
        int sn = cn_full < TOPK ? cn_full : TOPK;
        meta_idx[3 * t] = meta[3 * t];
        meta_idx[3 * t + 1] = meta[3 * t + 1];
        meta_idx[3 * t + 2] = sn;
        for (int i = 0; i < cn_full; i++) pool[i] = i;
        for (int i = 0; i < sn; i++) { /* partial shuffle: unsorted subset */
            int j = i + (int)((frand(&seed) + 0.5f) * (cn_full - i));
            if (j >= cn_full) j = cn_full - 1;
            int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
            sel[(size_t)t * TOPK + i] = pool[i];
        }
    }
    if (!dsv4_cuda_sparse_attn_batch_cached_idx(dev, LAYER, q,
                                                vals + (size_t)pre * DIM, START,
                                                sinks, meta_idx,
                                                sel, TOPK, first, COMP, HEADS,
                                                DIM, T, scale, got)) {
        fprintf(stderr, "sparse_attn_batch_cached_idx refused\n");
        return 11;
    }
    worst = 0.0;
    for (int t = 0; t < T; t++) {
        int wo = meta_idx[3 * t], wn = meta_idx[3 * t + 1],
            sn = meta_idx[3 * t + 2];
        memcpy(vals_g, vals + (size_t)wo * DIM,
               (size_t)wn * DIM * sizeof(*vals_g));
        for (int i = 0; i < sn; i++)
            memcpy(vals_g + (size_t)(wn + i) * DIM,
                   vals + (size_t)(comp_base + sel[(size_t)t * TOPK + i]) * DIM,
                   (size_t)DIM * sizeof(*vals_g));
        ref_token(want, q + (size_t)t * HEADS * DIM, vals_g, sinks,
                  0, wn, sn, wn, HEADS, DIM, scale);
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++) {
            double diff = fabs((double)want[i] -
                               got[(size_t)t * HEADS * DIM + i]);
            if (diff > worst) worst = diff;
            if (diff > 2e-3) {
                fprintf(stderr, "idx attention mismatch t=%d i=%zu want=%g got=%g\n",
                        t, i, want[i], got[(size_t)t * HEADS * DIM + i]);
                return 12;
            }
        }
    }
    printf("cached idx sparse attention: max |diff| = %g\n", worst);
    free(sel); free(meta_idx); free(pool); free(vals_g);

    /* ---- bf16 batch matmul ---- */
    enum { O = 256, I = 4096, TT = 64 };
    uint16_t *w = malloc((size_t)O * I * sizeof(*w));
    float *x = malloc((size_t)TT * I * sizeof(float));
    float *y = malloc((size_t)TT * O * sizeof(float));
    if (!w || !x || !y) return 5;
    for (size_t i = 0; i < (size_t)O * I; i++) {
        float v = frand(&seed);
        uint32_t bits;
        memcpy(&bits, &v, 4);
        w[i] = (uint16_t)(bits >> 16);
    }
    for (size_t i = 0; i < (size_t)TT * I; i++) x[i] = frand(&seed);
    Dsv4CudaTensor *t = NULL;
    if (!dsv4_cuda_upload_bf16(&t, w, O, I, dev)) return 6;
    if (!dsv4_cuda_matmul_bf16_batch(t, x, TT, y)) {
        fprintf(stderr, "matmul_bf16_batch refused\n");
        return 7;
    }
    worst = 0.0;
    for (int tt = 0; tt < TT; tt++)
        for (int o = 0; o < O; o++) {
            float sum = 0.0f;
            for (int i = 0; i < I; i++)
                sum += bf16_decode(w[(size_t)o * I + i]) * x[(size_t)tt * I + i];
            double diff = fabs((double)sum - y[(size_t)tt * O + o]);
            double mag = fabs((double)sum) + 1.0;
            if (diff / mag > worst) worst = diff / mag;
            if (diff / mag > 1e-4) {
                fprintf(stderr, "bf16 matmul mismatch t=%d o=%d want=%g got=%g\n",
                        tt, o, sum, y[(size_t)tt * O + o]);
                return 8;
            }
        }
    printf("bf16 batch matmul: max rel diff = %g\n", worst);
    dsv4_cuda_tensor_free(t);
    dsv4_cuda_shutdown();
    puts("dsv4 sparse attn batch: OK");
    return 0;
}
