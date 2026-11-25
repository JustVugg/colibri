/* dflash_select.h -- DFlash2's path selector on the host, apart from the device code of
 * dflash_cuda.h so that a CPU test can run it (tests/test_dflash_select.c).
 *
 * Per draft position t the device leaves the top K logits (values v[0..K), ids after them,
 * v[0] the largest) and the hidden projection g_t (rank floats). A candidate b after the
 * previous draft a scores S_t(a, b) = U_t(b) + <A(a) * g_t, B(b)>; the greedy path takes the
 * best successor of the draft before.
 *
 * Sampled requests (spec_exact.h): q_t(. | a) = softmax(S_t(a, .) / temp) over the
 * candidates is a distribution conditioned on the previous draft, so a draft drawn from it is
 * a valid proposal for min(1, p/q) verification, row after row. qs != NULL fills q_t for every
 * position (sorted by score, the greedy pick first); drawn != 0 draws the path from them with
 * rnd(), otherwise the path is the greedy one, bit for bit as before. */
#ifndef DFLASH_SELECT_H
#define DFLASH_SELECT_H
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "spec_exact.h"

#define DF_SEL_MAX_RANK 512
#define DF_SEL_MAX_K SPEC_Q_MAX

static inline float df_sel_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

/* tp: k rows of 2K floats (K values, K ids as int bits); hg: k rows of rank floats; A, B: the
 * codebooks, bf16 [V][rank]. out/prob: the drafts and each one's probability among the
 * position's candidates by logit (what the gate reads, as before). Returns k. */
static int df_select_path(const float *tp, const float *hg, const uint16_t *A, const uint16_t *Bk,
                          int rank, int K, int V, int tok, int k, int *out, float *prob,
                          SpecQ *qs, int drawn, float temp, double (*rnd)(void)) {
    if (K > DF_SEL_MAX_K) K = DF_SEL_MAX_K;
    if (rank > DF_SEL_MAX_RANK) rank = DF_SEL_MAX_RANK;
    int prev = tok;
    for (int t = 0; t < k; t++) {
        const float *v = tp + (size_t)t * 2 * K, *gt = hg + (size_t)t * rank;
        const uint16_t *a = A + (size_t)prev * rank;
        float ag[DF_SEL_MAX_RANK], sc[DF_SEL_MAX_K];
        int ok[DF_SEL_MAX_K];
        for (int j = 0; j < rank; j++) ag[j] = df_sel_bf16(a[j]) * gt[j];
        int best = 0; float bs = -INFINITY, z = 0.f;
        for (int c = 0; c < K; c++) {
            int id; memcpy(&id, v + K + c, 4);
            ok[c] = id >= 0 && id < V;
            if (!ok[c]) { sc[c] = -INFINITY; continue; }
            const uint16_t *b = Bk + (size_t)id * rank;
            float s = v[c];
            for (int j = 0; j < rank; j++) s += ag[j] * df_sel_bf16(b[j]);
            sc[c] = s;
            if (s > bs) { bs = s; best = c; }
        }
        for (int c = 0; c < K; c++) z += expf(v[c] - v[0]);
        int pick = best;
        if (qs && temp > 0.f && bs > -INFINITY) {
            /* q over the valid candidates by score, the greedy pick first, then by score
             * (ties: candidate order, which is the device's logit order) */
            SpecQ *q = &qs[t];
            double w[DF_SEL_MAX_K], sum = 0.0;
            int ord[DF_SEL_MAX_K], n = 0;
            for (int c = 0; c < K; c++) if (ok[c]) ord[n++] = c;
            for (int i = 1; i < n; i++) {   /* insertion sort by score, descending, stable */
                int c = ord[i], j = i;
                while (j > 0 && sc[ord[j - 1]] < sc[c]) { ord[j] = ord[j - 1]; j--; }
                ord[j] = c;
            }
            for (int i = 0; i < n; i++) { w[i] = exp(((double)sc[ord[i]] - (double)bs) / (double)temp); sum += w[i]; }
            q->n = n;
            for (int i = 0; i < n; i++) { int id; memcpy(&id, v + K + ord[i], 4); q->id[i] = id; q->p[i] = (float)(w[i] / sum); }
            if (drawn && rnd) {
                int id = specq_draw(q, rnd());
                for (int c = 0; c < K; c++) { int cid; memcpy(&cid, v + K + c, 4); if (ok[c] && cid == id) { pick = c; break; } }
            }
        } else if (qs) {
            int id; memcpy(&id, v + K + best, 4); specq_point(&qs[t], id);
        }
        int id; memcpy(&id, v + K + pick, 4);
        out[t] = id; prob[t] = z > 0.f ? expf(v[pick] - v[0]) / z : 0.f;
        prev = id;
    }
    return k;
}
#endif
