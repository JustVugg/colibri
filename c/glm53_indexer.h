#ifndef COLIBRI_GLM53_INDEXER_H
#define COLIBRI_GLM53_INDEXER_H

/* Reference CPU implementation of the GLM-5.3 DSA k-pool indexer. The output
 * has sequence * (topk + pool - 1) entries and uses -1 for invalid slots. */
int coli_glm53_index_select(int *output, const float *queries,
                            const float *keys, const float *gate_scores,
                            const float *head_weights, const float *ape,
                            const unsigned char *valid, int sequence,
                            int heads, int dimension, int pool, int topk);

#endif
