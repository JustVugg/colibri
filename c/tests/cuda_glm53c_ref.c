/* The CPU's references for tests/test_cuda_glm53c.cu, which nvcc compiles as C++ (these headers
 * are C): delta_attention.h's KDA step. */
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
