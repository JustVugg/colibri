#include "glm53_sparse_attention.h"
#include "glm53_sparse_attention_case.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    float output[GLM53_SA_SEQUENCE * GLM53_SA_HEADS * GLM53_SA_VALUE_DIM];
    if (coli_glm53_sparse_attention(
            output, glm53_sa_queries, glm53_sa_keys, glm53_sa_values,
            glm53_sa_indices, GLM53_SA_SEQUENCE, GLM53_SA_WIDTH,
            GLM53_SA_HEADS, GLM53_SA_KEY_DIM, GLM53_SA_VALUE_DIM)) return 2;
    float worst = 0.0f;
    for (int i = 0; i < GLM53_SA_SEQUENCE * GLM53_SA_HEADS * GLM53_SA_VALUE_DIM; i++) {
        float error = fabsf(output[i] - glm53_sa_expected[i]);
        if (error > worst) worst = error;
    }
    if (worst > 2.0e-5f) {
        fprintf(stderr, "GLM53 sparse attention mismatch: %.9g\n", worst);
        return 1;
    }
    printf("PASS GLM53 DSA sparse CPU attention: max abs %.9g\n", worst);
    return 0;
}
