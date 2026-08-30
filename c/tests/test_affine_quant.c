#include "../affine_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { OUTPUT_DIM = 3, INPUT_DIM = 32, GROUP_SIZE = 16, BATCH = 2 };

static void store_le16(uint8_t *destination, uint16_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
}

static void store_le32(uint8_t *destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
    destination[2] = (uint8_t)(value >> 16);
    destination[3] = (uint8_t)(value >> 24);
}

static uint16_t f32_to_f16(float value) {
    uint32_t bits;
    uint32_t sign, mantissa;
    int exponent;
    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16) & 0x8000u;
    exponent = (int)((bits >> 23) & 0xffu) - 127 + 15;
    mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return (uint16_t)sign;
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

static void store_scalar(uint8_t *destination, size_t index,
                         ColiAffineScalarFormat format, float value) {
    uint32_t bits;
    if (format == COLI_AFFINE_SCALAR_F32) {
        memcpy(&bits, &value, sizeof(bits));
        store_le32(destination + index * 4, bits);
    } else if (format == COLI_AFFINE_SCALAR_F16) {
        store_le16(destination + index * 2, f32_to_f16(value));
    } else {
        memcpy(&bits, &value, sizeof(bits));
        store_le16(destination + index * 2, (uint16_t)(bits >> 16));
    }
}

static uint32_t packed_value(size_t row, size_t column, unsigned bits) {
    const uint32_t mask = (UINT32_C(1) << bits) - 1u;
    return (uint32_t)(row * 17 + column * 5 + 3) & mask;
}

static float direct_reference(const ColiAffineQuantizedView *view,
                              const float *input, size_t sample, size_t row) {
    const unsigned bits = coli_affine_bits(view->format);
    const unsigned per_word = 32u / bits;
    const size_t packed_words = view->input_dim / per_word;
    const size_t groups = view->input_dim / view->group_size;
    const uint32_t mask = (UINT32_C(1) << bits) - 1u;
    const uint8_t *weight_row =
        (const uint8_t *)view->weights + row * packed_words * 4;
    double result = 0.0;

    for (size_t column = 0; column < view->input_dim; column++) {
        const uint32_t word =
            coli_affine_load_le32(weight_row + (column / per_word) * 4);
        const uint32_t quantized =
            (word >> (bits * (unsigned)(column % per_word))) & mask;
        const size_t group = row * groups + column / view->group_size;
        const float scale = coli_affine_load_scalar(
            view->scales, group, view->scalar_format);
        const float bias = coli_affine_load_scalar(
            view->biases, group, view->scalar_format);
        const float x = input[sample * view->input_dim + column];
        result += (double)(scale * (float)quantized + bias) * (double)x;
    }
    return (float)result;
}

static int run_parity(ColiAffineFormat format,
                      ColiAffineScalarFormat scalar_format) {
    const unsigned bits = coli_affine_bits(format);
    const unsigned per_word = 32u / bits;
    const size_t packed_words = INPUT_DIM / per_word;
    const size_t scalar_count = OUTPUT_DIM * (INPUT_DIM / GROUP_SIZE);
    const size_t scalar_size = coli_affine_scalar_size(scalar_format);
    uint8_t weights[OUTPUT_DIM * INPUT_DIM];
    uint8_t scales[OUTPUT_DIM * (INPUT_DIM / GROUP_SIZE) * 4];
    uint8_t biases[OUTPUT_DIM * (INPUT_DIM / GROUP_SIZE) * 4];
    float input[BATCH * INPUT_DIM], output[BATCH * OUTPUT_DIM];

    memset(weights, 0, sizeof(weights));
    memset(scales, 0, sizeof(scales));
    memset(biases, 0, sizeof(biases));
    for (size_t row = 0; row < OUTPUT_DIM; row++) {
        for (size_t word_index = 0; word_index < packed_words; word_index++) {
            uint32_t word = 0;
            for (unsigned lane = 0; lane < per_word; lane++) {
                const size_t column = word_index * per_word + lane;
                word |= packed_value(row, column, bits) << (lane * bits);
            }
            store_le32(weights + (row * packed_words + word_index) * 4, word);
        }
    }
    for (size_t index = 0; index < scalar_count; index++) {
        const float scale = (float)((int)(index % 5) - 2) * 0.125f;
        const float bias = (float)((int)(index % 3) - 1) * 0.0625f;
        store_scalar(scales, index, scalar_format, scale);
        store_scalar(biases, index, scalar_format, bias);
    }
    for (size_t index = 0; index < BATCH * INPUT_DIM; index++)
        input[index] = (float)((int)(index % 13) - 6) * 0.125f;

    ColiAffineQuantizedView view = {
        weights, scales, biases,
        OUTPUT_DIM * packed_words * 4, scalar_count * scalar_size,
        scalar_count * scalar_size,
        OUTPUT_DIM, INPUT_DIM, GROUP_SIZE, format, scalar_format
    };
    ColiAffineStatus status = coli_affine_validate(&view);
    if (status != COLI_AFFINE_OK) {
        fprintf(stderr, "affine validation failed bits=%u scalar=%d: %s\n",
                bits, (int)scalar_format, coli_affine_status_string(status));
        return 1;
    }
    status = coli_affine_matmul_ref(output, input, BATCH, &view);
    if (status != COLI_AFFINE_OK) {
        fprintf(stderr, "affine matmul failed bits=%u scalar=%d: %s\n",
                bits, (int)scalar_format, coli_affine_status_string(status));
        return 1;
    }

    for (size_t sample = 0; sample < BATCH; sample++) {
        for (size_t row = 0; row < OUTPUT_DIM; row++) {
            const float expected = direct_reference(&view, input, sample, row);
            const float actual = output[sample * OUTPUT_DIM + row];
            const float tolerance = 2e-6f * fmaxf(1.0f, fabsf(expected));
            if (fabsf(actual - expected) > tolerance) {
                fprintf(stderr,
                        "affine parity failed bits=%u scalar=%d sample=%zu row=%zu: "
                        "got %.9g want %.9g\n",
                        bits, (int)scalar_format, sample, row, actual, expected);
                return 1;
            }
        }
    }
    return 0;
}

static int test_validation(void) {
    uint32_t weight = 0;
    float scales = 1.0f, bias = 0.0f;
    float input[8] = {0}, output = 123.0f;
    ColiAffineQuantizedView view = {
        &weight, &scales, &bias, sizeof(weight), sizeof(scales), sizeof(bias),
        1, 8, 8, COLI_AFFINE_MLX_Q4, COLI_AFFINE_SCALAR_F32
    };

#define EXPECT_STATUS(expression, expected) do {                                  \
        ColiAffineStatus got = (expression);                                      \
        if (got != (expected)) {                                                  \
            fprintf(stderr, "validation line %d: got %s, want %s\n", __LINE__,  \
                    coli_affine_status_string(got),                               \
                    coli_affine_status_string(expected));                         \
            return 1;                                                             \
        }                                                                         \
    } while (0)

    EXPECT_STATUS(coli_affine_validate(&view), COLI_AFFINE_OK);
    view.biases = NULL;
    EXPECT_STATUS(coli_affine_validate(&view), COLI_AFFINE_NULL);
    view.biases = &bias;
    view.format = (ColiAffineFormat)99;
    EXPECT_STATUS(coli_affine_validate(&view), COLI_AFFINE_BAD_FORMAT);
    view.format = COLI_AFFINE_MLX_Q4;
    view.input_dim = 7;
    EXPECT_STATUS(coli_affine_validate(&view), COLI_AFFINE_BAD_SHAPE);
    view.input_dim = 8;
    view.weight_bytes--;
    EXPECT_STATUS(coli_affine_validate(&view), COLI_AFFINE_TRUNCATED);
    view.weight_bytes++;
    view.scale_bytes--;
    EXPECT_STATUS(coli_affine_validate(&view), COLI_AFFINE_TRUNCATED);
    view.scale_bytes++;
    view.bias_bytes--;
    EXPECT_STATUS(coli_affine_validate(&view), COLI_AFFINE_TRUNCATED);
    view.bias_bytes++;
    view.output_dim = SIZE_MAX;
    EXPECT_STATUS(coli_affine_validate(&view), COLI_AFFINE_OVERFLOW);
    view.output_dim = 1;
    EXPECT_STATUS(coli_affine_matmul_ref(&output, input, 0, &view),
                  COLI_AFFINE_BAD_SHAPE);
    EXPECT_STATUS(coli_affine_matmul_ref(&output, input, SIZE_MAX, &view),
                  COLI_AFFINE_OVERFLOW);
#undef EXPECT_STATUS
    if (output != 123.0f) return 1;
    return 0;
}

int main(void) {
    for (int format = COLI_AFFINE_MLX_Q4;
         format <= COLI_AFFINE_MLX_Q8; format++) {
        for (int scalar = COLI_AFFINE_SCALAR_F32;
             scalar <= COLI_AFFINE_SCALAR_BF16; scalar++) {
            if (run_parity((ColiAffineFormat)format,
                           (ColiAffineScalarFormat)scalar))
                return 1;
        }
    }
    if (test_validation()) return 1;
    puts("affine quant tests: ok");
    return 0;
}
