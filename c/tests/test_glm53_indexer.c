#include "glm53_indexer.h"
#include "glm53_indexer_case.h"

#include <stdio.h>

int main(void) {
    int output[GLM53_INDEX_SEQUENCE * GLM53_INDEX_WIDTH];
    if (coli_glm53_index_select(
            output, glm53_index_queries, glm53_index_keys, glm53_index_gates,
            glm53_index_weights, glm53_index_ape, glm53_index_valid,
            GLM53_INDEX_SEQUENCE, GLM53_INDEX_HEADS, GLM53_INDEX_DIM,
            GLM53_INDEX_POOL, GLM53_INDEX_TOPK)) return 2;
    for (int i = 0; i < GLM53_INDEX_SEQUENCE * GLM53_INDEX_WIDTH; i++) {
        if (output[i] != glm53_index_expected[i]) {
            fprintf(stderr, "GLM53 index mismatch at %d: got %d expected %d\n",
                    i, output[i], glm53_index_expected[i]);
            return 1;
        }
    }
    puts("PASS GLM53 DSA k-pool CPU indexer");
    return 0;
}
