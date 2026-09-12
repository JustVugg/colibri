/* affine_quant.h -- portable MLX affine-quantization contract and CPU oracle.
 *
 * This format is deliberately separate from QT fmt=4.  Colibri's fmt=4 stores
 * signed int4 values with multiplicative group scales; MLX stores unsigned
 * uint32-packed values and reconstructs each group as scale*q + bias.
 */
#ifndef COLI_AFFINE_QUANT_H
#define COLI_AFFINE_QUANT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    COLI_AFFINE_MLX_Q4 = 0,
    COLI_AFFINE_MLX_Q8 = 1
} ColiAffineFormat;

typedef enum {
    COLI_AFFINE_SCALAR_F32 = 0,
    COLI_AFFINE_SCALAR_F16 = 1,
    COLI_AFFINE_SCALAR_BF16 = 2
} ColiAffineScalarFormat;

typedef enum {
    COLI_AFFINE_OK = 0,
    COLI_AFFINE_NULL = -1,
    COLI_AFFINE_BAD_FORMAT = -2,
    COLI_AFFINE_BAD_SHAPE = -3,
    COLI_AFFINE_TRUNCATED = -4,
    COLI_AFFINE_OVERFLOW = -5
} ColiAffineStatus;

/* MLX affine weights are little-endian uint32 words, packed along input_dim.
 * Scales and biases have identical scalar types and [output_dim, groups]
 * layout, where groups = input_dim/group_size.  Byte lengths are mandatory:
 * qpack sections and resident tensors must be checked before GPU registration.
 */
typedef struct {
    const void *weights;
    const void *scales;
    const void *biases;
    size_t weight_bytes;
    size_t scale_bytes;
    size_t bias_bytes;
    size_t output_dim;
    size_t input_dim;
    size_t group_size;
    ColiAffineFormat format;
    ColiAffineScalarFormat scalar_format;
} ColiAffineQuantizedView;

static inline const char *coli_affine_status_string(ColiAffineStatus status) {
    switch (status) {
        case COLI_AFFINE_OK: return "ok";
        case COLI_AFFINE_NULL: return "null affine buffer";
        case COLI_AFFINE_BAD_FORMAT: return "unsupported affine format";
        case COLI_AFFINE_BAD_SHAPE: return "invalid affine shape";
        case COLI_AFFINE_TRUNCATED: return "truncated affine buffer";
        case COLI_AFFINE_OVERFLOW: return "affine size overflow";
    }
    return "unknown affine status";
}

static inline unsigned coli_affine_bits(ColiAffineFormat format) {
    return format == COLI_AFFINE_MLX_Q4 ? 4u :
           format == COLI_AFFINE_MLX_Q8 ? 8u : 0u;
}

static inline size_t coli_affine_scalar_size(ColiAffineScalarFormat format) {
    return format == COLI_AFFINE_SCALAR_F32 ? 4u :
           (format == COLI_AFFINE_SCALAR_F16 ||
            format == COLI_AFFINE_SCALAR_BF16) ? 2u : 0u;
}

static inline int coli_affine_size_mul(size_t left, size_t right, size_t *out) {
    if (left && right > SIZE_MAX / left) return 0;
    *out = left * right;
    return 1;
}

static inline ColiAffineStatus
coli_affine_validate(const ColiAffineQuantizedView *view) {
    size_t packed_words, group_count, need_weights, need_scalars;
    unsigned bits, per_word;
    size_t scalar_size;

    if (!view || !view->weights || !view->scales || !view->biases)
        return COLI_AFFINE_NULL;
    bits = coli_affine_bits(view->format);
    scalar_size = coli_affine_scalar_size(view->scalar_format);
    if (!bits || !scalar_size) return COLI_AFFINE_BAD_FORMAT;
    if (!view->output_dim || !view->input_dim || !view->group_size)
        return COLI_AFFINE_BAD_SHAPE;

    per_word = 32u / bits;
    if (view->input_dim % view->group_size ||
        view->input_dim % per_word || view->group_size % per_word)
        return COLI_AFFINE_BAD_SHAPE;

    packed_words = view->input_dim / per_word;
    group_count = view->input_dim / view->group_size;
    if (!coli_affine_size_mul(view->output_dim, packed_words, &need_weights) ||
        !coli_affine_size_mul(need_weights, sizeof(uint32_t), &need_weights) ||
        !coli_affine_size_mul(view->output_dim, group_count, &need_scalars) ||
        !coli_affine_size_mul(need_scalars, scalar_size, &need_scalars))
        return COLI_AFFINE_OVERFLOW;
    if (view->weight_bytes < need_weights ||
        view->scale_bytes < need_scalars || view->bias_bytes < need_scalars)
        return COLI_AFFINE_TRUNCATED;
    return COLI_AFFINE_OK;
}

static inline uint16_t coli_affine_load_le16(const uint8_t *source) {
    return (uint16_t)source[0] | (uint16_t)((uint16_t)source[1] << 8);
}

static inline uint32_t coli_affine_load_le32(const uint8_t *source) {
    return (uint32_t)source[0] | ((uint32_t)source[1] << 8) |
           ((uint32_t)source[2] << 16) | ((uint32_t)source[3] << 24);
}

static inline float coli_affine_f32_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline float coli_affine_f16_to_f32(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    int exponent = (int)(((uint32_t)value >> 10) & 0x1fu);
    uint32_t mantissa = (uint32_t)value & 0x3ffu;

    if (exponent == 0) {
        if (!mantissa) return coli_affine_f32_from_bits(sign);
        exponent = 1;
        while (!(mantissa & 0x400u)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3ffu;
        exponent += 127 - 15;
    } else if (exponent == 31) {
        exponent = 255;
    } else {
        exponent += 127 - 15;
    }
    return coli_affine_f32_from_bits(sign | ((uint32_t)exponent << 23) |
                                     (mantissa << 13));
}

static inline float coli_affine_load_scalar(const void *values, size_t index,
                                             ColiAffineScalarFormat format) {
    const uint8_t *bytes = (const uint8_t *)values;
    if (format == COLI_AFFINE_SCALAR_F32)
        return coli_affine_f32_from_bits(coli_affine_load_le32(bytes + index * 4));
    if (format == COLI_AFFINE_SCALAR_F16)
        return coli_affine_f16_to_f32(coli_affine_load_le16(bytes + index * 2));
    return coli_affine_f32_from_bits(
        (uint32_t)coli_affine_load_le16(bytes + index * 2) << 16);
}

/* Correctness-first y[S,O] = x[S,I] @ dequant(W[O,I])^T.  Accumulation is
 * grouped exactly like Swiftlet's generic Metal kernel: two f32 reductions
 * (q*x and x), followed by scale*qdot + bias*xsum for each group.
 */
static inline ColiAffineStatus
coli_affine_matmul_ref(float *output, const float *input, size_t batch,
                       const ColiAffineQuantizedView *view) {
    ColiAffineStatus status = coli_affine_validate(view);
    unsigned bits, per_word;
    uint32_t mask;
    size_t packed_words, groups, unused;
    const uint8_t *weights;

    if (status != COLI_AFFINE_OK) return status;
    if (!output || !input) return COLI_AFFINE_NULL;
    if (!batch) return COLI_AFFINE_BAD_SHAPE;
    if (!coli_affine_size_mul(batch, view->input_dim, &unused) ||
        !coli_affine_size_mul(batch, view->output_dim, &unused))
        return COLI_AFFINE_OVERFLOW;

    bits = coli_affine_bits(view->format);
    per_word = 32u / bits;
    mask = (UINT32_C(1) << bits) - 1u;
    packed_words = view->input_dim / per_word;
    groups = view->input_dim / view->group_size;
    weights = (const uint8_t *)view->weights;

    for (size_t row = 0; row < view->output_dim; row++) {
        const uint8_t *weight_row = weights + row * packed_words * 4;
        for (size_t sample = 0; sample < batch; sample++) {
            const float *input_row = input + sample * view->input_dim;
            float sum = 0.0f;
            for (size_t group = 0; group < groups; group++) {
                const size_t word_base = group * (view->group_size / per_word);
                const size_t input_base = group * view->group_size;
                float quantized_dot = 0.0f, input_sum = 0.0f;

                for (size_t local = 0; local < view->group_size; local++) {
                    const size_t column = input_base + local;
                    const size_t word_index = word_base + local / per_word;
                    const uint32_t word =
                        coli_affine_load_le32(weight_row + word_index * 4);
                    const unsigned shift = bits * (unsigned)(local % per_word);
                    const float x = input_row[column];
                    quantized_dot += (float)((word >> shift) & mask) * x;
                    input_sum += x;
                }
                const size_t scalar_index = row * groups + group;
                const float scale = coli_affine_load_scalar(
                    view->scales, scalar_index, view->scalar_format);
                const float bias = coli_affine_load_scalar(
                    view->biases, scalar_index, view->scalar_format);
                sum += scale * quantized_dot + bias * input_sum;
            }
            output[sample * view->output_dim + row] = sum;
        }
    }
    return COLI_AFFINE_OK;
}

#ifdef __cplusplus
}
#endif

#endif
