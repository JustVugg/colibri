/* The CPU's references for tests/test_cuda_dsv4c.cu, which nvcc compiles as C++ (these
 * headers are C): hyper_connections.h's split and write back, sparse_attn.h's attention. */
#include "../hyper_connections.h"
#include "../sparse_attn.h"
int dsv4c_ref_split(float *pre, float *post, float *comb, const float *mixes, const float *scale, const float *base, int hc, int iters, float eps) {
    return coli_hc_split_sinkhorn(pre, post, comb, mixes, scale, base, hc, iters, eps);
}
int dsv4c_ref_post(float *out, const float *branch, const float *residual, const float *post, const float *comb, int hc, int D) {
    return coli_hc_post(out, branch, residual, post, comb, hc, D);
}
void dsv4c_ref_attend(float *out, const float *q, const float *kv, const int *idx, int count, int hd, float sink, float scale, float *score) {
    coli_sparse_attend_scalar(out, q, kv, idx, count, hd, sink, scale, score);
}
