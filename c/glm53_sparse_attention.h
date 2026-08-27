#ifndef COLIBRI_GLM53_SPARSE_ATTENTION_H
#define COLIBRI_GLM53_SPARSE_ATTENTION_H

/* CPU DSA attention over per-query selected token indices. Queries/keys are
 * [sequence, heads, key_dim], values/output [sequence, heads, value_dim]. */
int coli_glm53_sparse_attention(float *output, const float *queries,
                                const float *keys, const float *values,
                                const int *indices, int sequence, int width,
                                int heads, int key_dim, int value_dim);

#endif
