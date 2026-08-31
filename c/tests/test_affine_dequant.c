/* coli_affine_dequant_ref gates: the loader-side expansion oracle must read
 * the SAME bits as coli_affine_matmul_ref and refuse the same malformed
 * views.
 *
 *   1. nibble order  -- a Q4 word authored as 0x76543210 must expand to
 *      logical columns 0,1,...,7 in that order (lowest bits = lowest
 *      column).  A most-significant-first reading produces the reversed
 *      sequence and fails the exact-value checks.
 *   2. affine map    -- expansion is scale*q + bias per group, with the
 *      BF16/F16/F32 scalar decodings all exercised (the values chosen are
 *      exactly representable in every one of the three, so the checks are
 *      equality, not tolerance).
 *   3. Q8 arm        -- per_word drops to 4 and the mask widens; a crafted
 *      word proves the byte order.
 *   4. matmul parity -- x @ dequant(W)^T must equal matmul_ref(x, W) on a
 *      multi-group view, so the two readings of the packed words cannot
 *      drift apart.
 *   5. refusals      -- NULL buffers, truncated scales, and a group size
 *      that does not divide input_dim come back as statuses, never as
 *      plausible output.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../affine_quant.h"

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                           \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);   /* the test values are bf16-exact */
}

/* Q4, one row of 16 logical columns in two groups of 8: word 0 counts
 * 0..7 LSB-first, word 1 counts 8..15.  Group 0 maps q -> 2q - 1, group 1
 * maps q -> 0.5q + 4; every expected value is exact in bf16/f16/f32. */
static void test_q4_nibble_order_and_affine(void) {
    const uint32_t words[2] = { 0x76543210u, 0xFEDCBA98u };
    const uint16_t scales_bf16[2] = { 0x4000u /* 2.0 */, 0x3F00u /* 0.5 */ };
    const uint16_t biases_bf16[2] = { 0xBF80u /* -1.0 */, 0x4080u /* 4.0 */ };
    ColiAffineQuantizedView view = {
        words, scales_bf16, biases_bf16,
        sizeof(words), sizeof(scales_bf16), sizeof(biases_bf16),
        1, 16, 8, COLI_AFFINE_MLX_Q4, COLI_AFFINE_SCALAR_BF16
    };
    float out[16];
    CHECK(coli_affine_dequant_ref(&view, out) == COLI_AFFINE_OK);
    for (int q = 0; q < 8; q++)
        CHECK(out[q] == 2.0f * (float)q - 1.0f);
    for (int q = 8; q < 16; q++)
        CHECK(out[q] == 0.5f * (float)q + 4.0f);
}

/* Q8: 4 values per word, byte order proven by an asymmetric word. */
static void test_q8_byte_order(void) {
    const uint32_t words[1] = { 0xFF400201u };   /* q = 1, 2, 64, 255 */
    const float scales_f32[1] = { 3.0f };
    const float biases_f32[1] = { -2.0f };
    const float expected[4] = { 1.0f, 4.0f, 190.0f, 763.0f };
    ColiAffineQuantizedView view = {
        words, scales_f32, biases_f32,
        sizeof(words), sizeof(scales_f32), sizeof(biases_f32),
        1, 4, 4, COLI_AFFINE_MLX_Q8, COLI_AFFINE_SCALAR_F32
    };
    float out[4];
    CHECK(coli_affine_dequant_ref(&view, out) == COLI_AFFINE_OK);
    for (int i = 0; i < 4; i++) CHECK(out[i] == expected[i]);
}

/* F16 scalar decoding through the same path (1.5 and 0.25 are f16-exact). */
static void test_f16_scalars(void) {
    const uint32_t words[1] = { 0x00000021u };   /* q = 1, 2, 0, 0, ... */
    const uint16_t scales_f16[1] = { 0x3E00u /* 1.5 */ };
    const uint16_t biases_f16[1] = { 0x3400u /* 0.25 */ };
    ColiAffineQuantizedView view = {
        words, scales_f16, biases_f16,
        sizeof(words), sizeof(scales_f16), sizeof(biases_f16),
        1, 8, 8, COLI_AFFINE_MLX_Q4, COLI_AFFINE_SCALAR_F16
    };
    float out[8];
    CHECK(coli_affine_dequant_ref(&view, out) == COLI_AFFINE_OK);
    CHECK(out[0] == 1.75f && out[1] == 3.25f);
    for (int i = 2; i < 8; i++) CHECK(out[i] == 0.25f);
}

/* Multi-row multi-group parity: y = matmul_ref(x, W) must equal
 * x @ dequant_ref(W)^T, or the two functions read different bits. */
static void test_matmul_parity(void) {
    enum { O = 3, I = 32, GS = 16, WORDS_PER_ROW = I / 8 };
    uint32_t words[O * WORDS_PER_ROW];
    uint16_t scales[O * (I / GS)], biases[O * (I / GS)];
    float x[I], expanded[O * I], want[O], got[O];
    for (int i = 0; i < O * WORDS_PER_ROW; i++)
        words[i] = 0x9E3779B9u * (uint32_t)(i + 1) + 0x7F4A7C15u;
    for (int i = 0; i < O * (I / GS); i++) {
        scales[i] = f32_to_bf16(0.25f + 0.125f * (float)i);
        biases[i] = f32_to_bf16(-1.0f + 0.5f * (float)(i % 3));
    }
    for (int i = 0; i < I; i++) x[i] = 0.0625f * (float)(i - I / 2);
    ColiAffineQuantizedView view = {
        words, scales, biases,
        sizeof(words), sizeof(scales), sizeof(biases),
        O, I, GS, COLI_AFFINE_MLX_Q4, COLI_AFFINE_SCALAR_BF16
    };
    CHECK(coli_affine_matmul_ref(want, x, 1, &view) == COLI_AFFINE_OK);
    CHECK(coli_affine_dequant_ref(&view, expanded) == COLI_AFFINE_OK);
    for (int o = 0; o < O; o++) {
        double sum = 0.0;
        for (int i = 0; i < I; i++) sum += (double)expanded[o * I + i] * x[i];
        got[o] = (float)sum;
        CHECK(fabsf(got[o] - want[o]) <= 1e-4f * (1.0f + fabsf(want[o])));
    }
}

static void test_refusals(void) {
    const uint32_t words[2] = { 0, 0 };
    const uint16_t scalars[2] = { 0x3F80u, 0x3F80u };
    float out[16];
    ColiAffineQuantizedView view = {
        words, scalars, scalars,
        sizeof(words), sizeof(scalars), sizeof(scalars),
        1, 16, 8, COLI_AFFINE_MLX_Q4, COLI_AFFINE_SCALAR_BF16
    };
    CHECK(coli_affine_dequant_ref(NULL, out) == COLI_AFFINE_NULL);
    CHECK(coli_affine_dequant_ref(&view, NULL) == COLI_AFFINE_NULL);

    ColiAffineQuantizedView truncated = view;
    truncated.scale_bytes = 2;               /* needs 2 groups * 2 bytes */
    CHECK(coli_affine_dequant_ref(&truncated, out) == COLI_AFFINE_TRUNCATED);

    ColiAffineQuantizedView bad_group = view;
    bad_group.group_size = 12;               /* 16 % 12 != 0 */
    CHECK(coli_affine_dequant_ref(&bad_group, out) == COLI_AFFINE_BAD_SHAPE);

    ColiAffineQuantizedView bad_format = view;
    bad_format.format = (ColiAffineFormat)7;
    CHECK(coli_affine_dequant_ref(&bad_format, out) == COLI_AFFINE_BAD_FORMAT);
}

int main(void) {
    test_q4_nibble_order_and_affine();
    test_q8_byte_order();
    test_f16_scalars();
    test_matmul_parity();
    test_refusals();
    printf("test_affine_dequant: OK\n");
    return 0;
}
