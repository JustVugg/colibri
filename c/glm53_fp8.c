#include "glm53_fp8.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static float e4m3fn_decode(uint8_t value) {
    int sign = value >> 7;
    int exponent = (value >> 3) & 15;
    int mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float number = exponent ? ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7) : ldexpf((float)mantissa, -9);
    return sign ? -number : number;
}

static uint8_t e4m3fn_encode(float value) {
    if (isnan(value)) return 0x7f;
    int negative = signbit(value) != 0;
    float magnitude = fabsf(value);
    if (!magnitude) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f) return (uint8_t)((negative ? 0x80 : 0) | 0x7e);
    uint8_t best = 0;
    float distance = FLT_MAX;
    for (uint8_t code = 0; code <= 0x7e; code++) {
        float candidate = e4m3fn_decode(code);
        float next = fabsf(candidate - magnitude);
        if (next < distance || (next == distance && !(code & 1) && (best & 1))) {
            best = code;
            distance = next;
        }
    }
    return (uint8_t)(best | (negative ? 0x80 : 0));
}

int coli_glm53_fp8_matvec(float *output, const uint8_t *weight, const float *weight_scale_inv, const float *input,
                          int rows, int columns) {
    if (!output || !weight || !weight_scale_inv || !input || rows < 1 || columns < 1 || columns % 128) return -1;
    int scale_columns = columns / 128;
    float *activation = malloc((size_t)columns * sizeof(*activation));
    if (!activation) return -1;
    for (int block = 0; block < scale_columns; block++) {
        int base = block * 128;
        float maximum = 0.0f;
        for (int i = 0; i < 128; i++) maximum = fmaxf(maximum, fabsf(input[base + i]));
        float scale = maximum > 0.0f ? maximum / 448.0f : 1.0f;
        for (int i = 0; i < 128; i++)
            activation[base + i] = e4m3fn_decode(e4m3fn_encode(input[base + i] / scale)) * scale;
    }
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (int column = 0; column < columns; column++) {
            int scale_index = (row / 128) * scale_columns + column / 128;
            float value = e4m3fn_decode(weight[(size_t)row * columns + column]);
            sum += activation[column] * value * weight_scale_inv[scale_index];
        }
        output[row] = sum;
    }
    free(activation);
    return 0;
}
