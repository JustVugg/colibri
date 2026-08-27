#include "glm53_kda.h"
#include "glm53_kda_case.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    float state[GLM53_KDA_HEADS * GLM53_KDA_DIM * GLM53_KDA_DIM] = {0};
    float window[3 * GLM53_KDA_HEADS * GLM53_KDA_DIM * GLM53_KDA_KERNEL] = {0};
    float output[GLM53_KDA_HEADS * GLM53_KDA_DIM];
    float worst = 0.0f;
    for (int step = 0; step < GLM53_KDA_STEPS; step++) {
        if (coli_glm53_kda_step(
                output, state, window,
                glm53_kda_qkv + step * 3 * GLM53_KDA_HEADS * GLM53_KDA_DIM,
                glm53_kda_conv,
                glm53_kda_decay + step * GLM53_KDA_HEADS * GLM53_KDA_DIM,
                glm53_kda_beta + step * GLM53_KDA_HEADS, GLM53_KDA_HEADS,
                GLM53_KDA_DIM, GLM53_KDA_KERNEL)) return 2;
        for (int i = 0; i < GLM53_KDA_HEADS * GLM53_KDA_DIM; i++) {
            float error = fabsf(
                output[i] - glm53_kda_output[step * GLM53_KDA_HEADS * GLM53_KDA_DIM + i]);
            if (error > worst) worst = error;
        }
    }
    if (worst > 2.0e-5f) {
        fprintf(stderr, "GLM53 KDA mismatch: max abs %.9g\n", worst);
        return 1;
    }
    printf("PASS GLM53 KDA recurrent CPU kernel: max abs %.9g\n", worst);
    return 0;
}
