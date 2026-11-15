/* The CPU's references for tests/test_cuda_glm53c.cu, which nvcc compiles as C++ (these headers
 * are C): delta_attention.h's KDA step, sparse_index.h's k-pooled selection, colibri.c's DSA
 * selection. */
#include "../delta_attention.h"
int glm53c_ref_kda_scratch(int H, int KD, int VD) { return coli_kda_scratch_floats(H, KD, VD); }
int glm53c_ref_kda_step(float *out, float *state, float *window, const float *qkv, const float *conv_w, const float *gate,
                        const float *beta, int H, int KD, int VD, int K, float neps, float *scratch) {
    return coli_kda_step(out, state, window, qkv, conv_w, gate, beta, H, KD, VD, K, neps, scratch);
}
/* sparse_index.h's k-pooled selection, rows q_from .. q_to - 1 */
#include "../sparse_index.h"
int glm53c_ref_width(int topk, int pool, int tail) { return coli_sparse_index_width(topk, pool, tail); }
int glm53c_ref_kpool(int *out, const float *q, const float *keys, const float *gates, const float *hw, const float *ape,
                     const unsigned char *valid, int T, int IH, int ID, int pool, int topk, int tail, int q_from, int q_to) {
    return coli_sparse_index_select_range(out, q, keys, gates, hw, ape, valid, T, IH, ID, pool, topk, tail, q_from, q_to);
}
/* colibri.c's DSA selection (attention_rows' indexer), its steps in its order: each row's scores
 * over keys 0..pos, the keep-th largest the threshold, the positions above it then those equal to
 * it, each in position order. nsel[s] 0: every position (a context not past topk, no force). */
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int glmdsa_desc(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? 1 : x > y ? -1 : 0; }
void glmdsa_ref_select(int *sel, int *nsel, const float *q, const float *hw, const float *keys, int S, int pb, int IH, int ID,
                       int topk, int force) {
    for (int s = 0; s < S; s++) {
        const int nk = pb + s + 1;
        if (nk <= topk && !force) { nsel[s] = 0; continue; }
        const int keep = nk < topk ? nk : topk;
        float *isc = (float *)malloc((size_t)nk * sizeof(float)), *tmp = (float *)malloc((size_t)nk * sizeof(float));
        const float *qi = q + (size_t)s * IH * ID, *w32 = hw + (size_t)s * IH;
        const float wsc = 1.f / sqrtf((float)IH), rs = 1.f / sqrtf((float)ID);
        for (int t = 0; t < nk; t++) {
            const float *kt = keys + (size_t)t * ID;
            float a = 0;
            for (int h = 0; h < IH; h++) {
                const float *qhp = qi + (size_t)h * ID;
                float d0 = 0; for (int i = 0; i < ID; i++) d0 += qhp[i] * kt[i];
                d0 *= rs; if (d0 > 0) a += w32[h] * d0;
            }
            isc[t] = a * wsc;
        }
        memcpy(tmp, isc, (size_t)nk * sizeof(float));
        qsort(tmp, (size_t)nk, sizeof(float), glmdsa_desc);
        const float thr = tmp[keep - 1];
        int *dst = sel + (size_t)s * topk, nd = 0;
        for (int t = 0; t < nk && nd < keep; t++) if (isc[t] > thr) dst[nd++] = t;
        for (int t = 0; t < nk && nd < keep; t++) if (isc[t] == thr) dst[nd++] = t;
        nsel[s] = nd;
        free(isc); free(tmp);
    }
}
