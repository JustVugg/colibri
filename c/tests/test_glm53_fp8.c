#include <math.h>
#include <stdio.h>

#include "glm53_fp8.h"
#include "glm53_fp8_case.h"

int main(void) {
    float output[GLM53_FP8_ROWS];
    if (coli_glm53_fp8_matvec(output, glm53_fp8_weight, glm53_fp8_scales, glm53_fp8_input, GLM53_FP8_ROWS,
                              GLM53_FP8_COLUMNS) != 0)
        return 1;
    float maximum = 0.0f;
    for (int row = 0; row < GLM53_FP8_ROWS; row++) {
        float error = fabsf(output[row] - glm53_fp8_expected[row]);
        if (error > maximum) maximum = error;
    }
    if (maximum > 2.0e-5f) {
        fprintf(stderr, "GLM53 FP8 mismatch: max abs %.9g\n", maximum);
        return 1;
    }
    printf("PASS GLM53 native FP8 matvec: max abs %.9g\n", maximum);
    return 0;
}
