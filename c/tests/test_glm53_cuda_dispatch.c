#define COLI_CUDA
#define main glm53_engine_main
#include "../glm53_flash.c"
#undef main

#include "glm53_fp8_case.h"

static int calls;
static float lut[256];

int coli_cuda_init(const int *devices, int count) { return devices && count == 1; }
void coli_cuda_shutdown(void) {}
int coli_cuda_fp8_set_lut(const float *values) {
    memcpy(lut, values, sizeof(lut));
    return 1;
}
void coli_cuda_tensor_free(ColiCudaTensor *tensor) { (void)tensor; }
int coli_cuda_matmul(ColiCudaTensor **tensor, float *output, const float *input, const void *weights,
                     const float *scales, int format, int sequences, int columns, int rows, int device,
                     int group_size) {
    if (!tensor || !output || !input || !weights || !scales || format != 8 || sequences != 1 || columns != 128 ||
        rows != GLM53_FP8_ROWS || device != 0 || group_size != 0)
        return 0;
    const uint8_t *bytes = weights;
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (int column = 0; column < columns; column++)
            sum += input[column] * lut[bytes[(size_t)row * columns + column]] *
                   scales[(row / 128) * (columns / 128) + column / 128];
        output[row] = sum;
    }
    *tensor = (ColiCudaTensor *)(uintptr_t)1;
    calls++;
    return 1;
}

int main(void) {
    coli_glm53_fp8_decode_table(lut);
    g_cuda_enabled = 1;
    g_cuda_device = 0;
    Matrix matrix = {0};
    matrix.view.format = COLI_TENSOR_FP8_E4M3_BLOCK;
    matrix.view.rows = GLM53_FP8_ROWS;
    matrix.view.columns = GLM53_FP8_COLUMNS;
    matrix.view.data = glm53_fp8_weight;
    matrix.view.scales = glm53_fp8_scales;
    float output[GLM53_FP8_ROWS];
    matrix_multiply(output, glm53_fp8_input, &matrix, 1);
    if (calls != 1 || matrix.cuda == NULL) return 1;
    float maximum = 0.0f;
    for (int row = 0; row < GLM53_FP8_ROWS; row++)
        maximum = fmaxf(maximum, fabsf(output[row] - glm53_fp8_expected[row]));
    if (maximum > 2.0e-5f) {
        fprintf(stderr, "GLM53 CUDA dispatch mismatch: max abs %.9g\n", maximum);
        return 1;
    }
    matrix_free(&matrix);
    printf("PASS GLM53 CUDA dispatch with activation QDQ: max abs %.9g\n", maximum);
    return 0;
}
