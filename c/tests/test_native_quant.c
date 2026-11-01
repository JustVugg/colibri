#include "../native_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int close_enough(float left, float right) {
    return fabsf(left - right) <= 1e-6f * fmaxf(1.0f, fabsf(right));
}

static int test_fp8_qdq_simd(void) {
    float input[4097], output[4097]; uint8_t scales[4097];
    uint32_t random = 1;
    for (int i = 0; i < 4097; i++) {
        random = random * 1664525u + 1013904223u;
        input[i] = (int32_t)random * (448.0f / 2147483648.0f);
    }
    for (int i = 0; i < 128; i++) {
        float a = coli_e4m3fn_decode((uint8_t)(i % 126));
        float b = coli_e4m3fn_decode((uint8_t)(i % 126 + 1));
        input[i] = (a + b) * 0.5f;
    }
    input[0] = -0.0f; input[128] = NAN; input[255] = 448.0f;
    const size_t blocks[] = {20, 128, 257};
    for (size_t k = 0; k < sizeof(blocks) / sizeof(blocks[0]); k++) {
        size_t block = blocks[k];
        if (coli_fp8_activation_qdq_ref(output, scales, input, 4097, block)) return 1;
        for (size_t base = 0; base < 4097; base += block) {
            size_t count = 4097 - base < block ? 4097 - base : block;
            float maximum = 1e-4f;
            for (size_t i = 0; i < count; i++) maximum = fmaxf(maximum, fabsf(input[base+i]));
            int exponent; float fraction = frexpf(maximum / 448.0f, &exponent);
            if (fraction == 0.5f) exponent--;
            float scale = ldexpf(1.0f, exponent);
            if (scales[base / block] != exponent + 127) return 1;
            for (size_t i = 0; i < count; i++) {
                float normalized = fmaxf(-448.0f, fminf(448.0f, input[base+i] / scale));
                float expected = coli_e4m3fn_decode(coli_e4m3fn_encode(normalized)) * scale;
                if (memcmp(output + base + i, &expected, sizeof(expected))) return 1;
            }
        }
    }
    return 0;
}

int main(void) {
    if (test_fp8_qdq_simd()) return 1;
#ifdef __AVX2__
    for (int base = 0; base < 256; base += 8) {
        int codes[8]; float decoded[8];
        for (int lane = 0; lane < 8; lane++) codes[lane] = base + lane;
        _mm256_storeu_ps(decoded, v4_fp8_decode8(
            _mm256_loadu_si256((const __m256i *)codes)));
        for (int lane = 0; lane < 8; lane++) {
            float expected = coli_e4m3fn_decode((uint8_t)codes[lane]);
            if (isnan(expected) ? !isnan(decoded[lane]) :
                memcmp(&expected, &decoded[lane], sizeof(float)) != 0) return 1;
        }
    }
#endif
#ifdef __AVX512F__
    for (int base = 0; base < 256; base += 16) {
        int codes[16]; float decoded[16];
        for (int lane = 0; lane < 16; lane++) codes[lane] = base + lane;
        _mm512_storeu_ps(decoded, v4_fp8_decode16(
            _mm512_loadu_si512((const void *)codes)));
        for (int lane = 0; lane < 16; lane++) {
            float expected = coli_e4m3fn_decode((uint8_t)codes[lane]);
            if (isnan(expected) ? !isnan(decoded[lane]) :
                memcmp(&expected, &decoded[lane], sizeof(float)) != 0) return 1;
        }
    }
#endif
#ifdef __AVX2__
    unsigned previous_mxcsr = _mm_getcsr();
    for (int flush = 0; flush < 2; flush++) {
        _mm_setcsr(flush ? previous_mxcsr | 0x8040u : previous_mxcsr & ~0x8040u);
        for (int base = 0; base < 256; base += 16) {
            uint8_t codes[16]; float decoded[16];
            for (int lane = 0; lane < 16; lane++) codes[lane] = (uint8_t)(base + lane);
            _mm256_storeu_ps(decoded, v4_fp8_decode8_bytes(_mm_loadl_epi64((const __m128i *)codes)));
            _mm256_storeu_ps(decoded + 8, v4_fp8_decode8_bytes(_mm_loadl_epi64((const __m128i *)(codes + 8))));
            for (int lane = 0; lane < 16; lane++) {
                float expected = coli_e4m3fn_decode(codes[lane]);
                if (isnan(expected) ? !isnan(decoded[lane]) :
                    memcmp(&expected, &decoded[lane], sizeof(float))) return 1;
            }
#ifdef __AVX512F__
            _mm512_storeu_ps(decoded, v4_fp8_decode16_bytes(_mm_loadu_si128((const __m128i *)codes)));
            for (int lane = 0; lane < 16; lane++) {
                float expected = coli_e4m3fn_decode(codes[lane]);
                if (isnan(expected) ? !isnan(decoded[lane]) :
                    memcmp(&expected, &decoded[lane], sizeof(float))) return 1;
            }
#endif
        }
    }
    _mm_setcsr(previous_mxcsr);
#endif
    if (coli_bf16_round(1.00390625f) != 1.0f ||
        coli_bf16_round(1.01171875f) != 1.015625f)
        return 1;
    static const float fp4[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    for (int i = 0; i < 16; i++)
        if (!close_enough(coli_e2m1_decode((uint8_t)i), fp4[i])) return 1;
    if (!close_enough(coli_e8m0_decode(0x7e), 0.5f) ||
        !close_enough(coli_e8m0_decode(0x7f), 1.0f) ||
        !close_enough(coli_e8m0_decode(0x80), 2.0f) ||
        !isnan(coli_e8m0_decode(0xff))) return 1;
    const float *e8m0 = coli_e8m0_table();
    if (!e8m0 || e8m0 != coli_e8m0_table()) return 1;
    for (int code = 0; code < 255; code++) {
        float decoded = coli_e8m0_decode((uint8_t)code);
        if (memcmp(&e8m0[code], &decoded, sizeof(decoded)) != 0) return 1;
    }
    if (!isnan(e8m0[255])) return 1;

    static const float representable[] = {
        0.0f, 0.001953125f, 0.5f, 1.0f, 1.5f, 6.0f,
        224.0f, 448.0f, -0.5f, -6.0f, -448.0f,
    };
    for (size_t i = 0; i < sizeof(representable) / sizeof(representable[0]); i++) {
        uint8_t encoded = coli_e4m3fn_encode(representable[i]);
        if (!close_enough(coli_e4m3fn_decode(encoded), representable[i])) return 1;
    }

    float input[128], qdq[128];
    uint8_t activation_scale;
    for (int i = 0; i < 128; i++) input[i] = 1.0f;
    if (coli_fp8_activation_qdq_ref(qdq, &activation_scale, input, 128, 128) != 0)
        return 1;
    if (activation_scale != 119) return 1; /* 2^-8 */
    for (int i = 0; i < 128; i++)
        if (!close_enough(qdq[i], 1.0f)) return 1;

    float fp4_input[32], fp4_qdq[32];
    uint8_t fp4_scale;
    for (int i = 0; i < 32; i++) fp4_input[i] = 1.0f;
    if (coli_fp4_activation_qdq_ref(fp4_qdq, &fp4_scale,
                                    fp4_input, 32, 32) != 0 ||
        fp4_scale != 125) return 1; /* 2^-2 */
    for (int i = 0; i < 32; i++)
        if (!close_enough(fp4_qdq[i], 1.0f)) return 1;

    /* Tiny valid heads can end in a partial activation block. */
    float partial_input[20], partial_fp8[20], partial_fp4[20];
    uint8_t partial_fp8_scale, partial_fp4_scale;
    for (int i = 0; i < 20; i++) partial_input[i] = 1.0f;
    if (coli_fp8_activation_qdq_ref(partial_fp8, &partial_fp8_scale,
                                    partial_input, 20, 64) != 0 ||
        partial_fp8_scale != 119 ||
        coli_fp4_activation_qdq_ref(partial_fp4, &partial_fp4_scale,
                                    partial_input, 20, 32) != 0 ||
        partial_fp4_scale != 125) return 1;
    for (int i = 0; i < 20; i++)
        if (!close_enough(partial_fp8[i], 1.0f) ||
            !close_enough(partial_fp4[i], 1.0f)) return 1;

    float hadamard[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    static const float hadamard_expected[4] = {5.0f, -1.0f, -2.0f, 0.0f};
    if (coli_hadamard_bf16_ref(hadamard, 4) != 0) return 1;
    for (int i = 0; i < 4; i++)
        if (!close_enough(hadamard[i], hadamard_expected[i])) return 1;

    uint8_t weights[64];
    uint8_t scales[4];
    memset(weights, 0x44, sizeof(weights)); /* two +2.0 E2M1 values */
    memset(scales, 0x7e, sizeof(scales));   /* 0.5 */
    ColiTensorView view = {
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        weights, scales, sizeof(weights), sizeof(scales),
        1, 128, 1, 32, NULL
    };
    float output;
    if (coli_fp4_matvec_ref(&output, &view, input) != 0 ||
        !close_enough(output, 128.0f)) return 1;

    uint8_t fp8_weights[128 * 128];
    float fp8_scales[1] = {1.0f};
    memset(fp8_weights, 0x38, sizeof(fp8_weights)); /* E4M3 1.0 */
    ColiTensorView fp8_view = {
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32,
        fp8_weights, fp8_scales, sizeof(fp8_weights), sizeof(fp8_scales),
        128, 128, 128, 128, NULL
    };
    float fp8_output[128];
    if (coli_fp8_matvec_ref(fp8_output, &fp8_view, input) != 0)
        return 1;
    for (int i = 0; i < 128; i++)
        if (!close_enough(fp8_output[i], 128.0f)) return 1;

    /* Flash layout: eight rows are interleaved by column.  It must be
     * numerically equivalent to the guarded row-major reference path while
     * exercising different values in every lane. */
    uint8_t row_major[8 * 128], rows8[8 * 128];
    float row_major_out[8], rows8_out[8];
    for (int row = 0; row < 8; row++)
        for (int column = 0; column < 128; column++)
            row_major[row * 128 + column] =
                (uint8_t)(0x30 + ((row + column) & 15));
    for (int column = 0; column < 128; column++)
        for (int row = 0; row < 8; row++)
            rows8[column * 8 + row] = row_major[row * 128 + column];
    ColiTensorView row_major_view = {
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_F32,
        row_major, fp8_scales, sizeof(row_major), sizeof(fp8_scales),
        8, 128, 128, 128, NULL
    };
    ColiTensorView rows8_view = row_major_view;
    rows8_view.data = rows8;
    rows8_view.block_rows = 8;
    if (coli_fp8_matvec_ref(row_major_out, &row_major_view, input) != 0 ||
        coli_fp8_matvec_ref(rows8_out, &rows8_view, input) != 0)
        return 1;
    for (int row = 0; row < 8; row++)
        if (!close_enough(rows8_out[row], row_major_out[row])) return 1;
    puts("native quant tests: ok");
    return 0;
}
