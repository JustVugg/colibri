#include "glm53_mhc.h"
#include "glm53_mhc_case.h"

#include <math.h>
#include <stdio.h>

static float check(const float *actual, const float *expected, int count) {
    float worst = 0.0f;
    for (int i = 0; i < count; i++) {
        float error = fabsf(actual[i] - expected[i]);
        if (error > worst) worst = error;
    }
    return worst;
}

int main(void) {
    float collapsed[GLM53_MHC_DIM], post[GLM53_MHC_COPIES];
    float comb[GLM53_MHC_COPIES * GLM53_MHC_COPIES];
    float output[GLM53_MHC_COPIES * GLM53_MHC_DIM];
    if (coli_glm53_mhc_pre(collapsed, post, comb, glm53_mhc_streams,
            glm53_mhc_function, glm53_mhc_scale, glm53_mhc_base,
            GLM53_MHC_COPIES, GLM53_MHC_DIM, GLM53_MHC_ITERATIONS,
            1.0e-5f, 1.0e-6f)) return 2;
    if (coli_glm53_mhc_post(output, glm53_mhc_branch, glm53_mhc_streams,
            post, comb, GLM53_MHC_COPIES, GLM53_MHC_DIM)) return 2;
    float worst = check(collapsed, glm53_mhc_collapsed, GLM53_MHC_DIM);
    float value = check(post, glm53_mhc_post, GLM53_MHC_COPIES);
    if (value > worst) worst = value;
    value = check(comb, glm53_mhc_comb, GLM53_MHC_COPIES * GLM53_MHC_COPIES);
    if (value > worst) worst = value;
    value = check(output, glm53_mhc_output, GLM53_MHC_COPIES * GLM53_MHC_DIM);
    if (value > worst) worst = value;
    if (worst > 2.0e-5f) {
        fprintf(stderr, "GLM53 mHC mismatch: %.9g\n", worst);
        return 1;
    }
    printf("PASS GLM53 mHC CPU collapse/expand: max abs %.9g\n", worst);
    return 0;
}
