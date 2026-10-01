#ifndef COLIBRI_NATIVE_QUANT_H
#define COLIBRI_NATIVE_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#ifdef __AVX2__
#include <immintrin.h>
/* Branchless SIMD decode of 8 E4M3FN codes -> 8 f32, byte-identical to
 * coli_e4m3fn_decode for all 256 inputs (exhaustively verified). Replaces a
 * slow per-element _mm256_i32gather_ps from a 256-float LUT. E4M3FN values are
 * exact, so the f32 bit pattern is built directly: normals via integer field
 * assembly, subnormals as (float)mantissa*2^-9 (exact), NaN (code&0x7F==0x7F)
 * as canonical qNaN overwriting the sign. */
static inline __m256 v4_fp8_decode8(__m256i codes) {
    __m256i man = _mm256_and_si256(codes, _mm256_set1_epi32(7));
    __m256i exp = _mm256_and_si256(_mm256_srli_epi32(codes, 3), _mm256_set1_epi32(0xF));
    __m256i sgn = _mm256_slli_epi32(_mm256_srli_epi32(codes, 7), 31);
    __m256i nbits = _mm256_or_si256(
        _mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(120)), 23),
        _mm256_slli_epi32(man, 20));
    __m256 nval = _mm256_castsi256_ps(nbits);
    float man_factor = 1.0f / (float)(1 << 9);
    __m256 sval = _mm256_mul_ps(_mm256_cvtepi32_ps(man), _mm256_set1_ps(man_factor));
    __m256 is_sub = _mm256_castsi256_ps(_mm256_cmpeq_epi32(exp, _mm256_setzero_si256()));
    __m256i sbits = _mm256_or_si256(
        _mm256_castps_si256(_mm256_blendv_ps(nval, sval, is_sub)), sgn);
    __m256i is_nan = _mm256_cmpeq_epi32(
        _mm256_and_si256(codes, _mm256_set1_epi32(0x7F)), _mm256_set1_epi32(0x7F));
    return _mm256_castsi256_ps(
        _mm256_blendv_epi8(sbits, _mm256_set1_epi32(0x7FC00000), is_nan));
}
/* E4M3 -> FP16 with its exponent left unbiased, then an exact 2^8 scale.
 * F16C handles FP16 subnormals without a scalar FP32 denormal assist. */
static inline __m256 v4_fp8_decode8_bytes(__m128i bytes) {
#ifdef __F16C__
    __m128i codes = _mm_cvtepu8_epi16(bytes);
    __m128i magnitude = _mm_and_si128(codes, _mm_set1_epi16(127));
    __m128i half = _mm_or_si128(_mm_slli_epi16(magnitude, 7),
        _mm_slli_epi16(_mm_and_si128(codes, _mm_set1_epi16(128)), 8));
    __m256 values = _mm256_mul_ps(_mm256_cvtph_ps(half), _mm256_set1_ps(256.0f));
    __m256i nan = _mm256_cvtepi16_epi32(_mm_cmpeq_epi16(magnitude, _mm_set1_epi16(127)));
    return _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(values),
        _mm256_set1_epi32(0x7fc00000), nan));
#else
    return v4_fp8_decode8(_mm256_cvtepu8_epi32(bytes));
#endif
}

#endif

#ifdef __AVX512F__
static inline __m512 v4_fp8_decode16(__m512i codes) {
    __m512i magnitude = _mm512_and_si512(codes, _mm512_set1_epi32(127));
    __m512i bits = _mm512_add_epi32(_mm512_slli_epi32(magnitude, 20),
                                    _mm512_set1_epi32(120 << 23));
    __m512 values = _mm512_mask_mul_ps(_mm512_castsi512_ps(bits),
        _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(8)),
        _mm512_cvtepi32_ps(magnitude), _mm512_set1_ps(1.0f / 512.0f));
    __m512i sign = _mm512_slli_epi32(
        _mm512_and_si512(codes, _mm512_set1_epi32(128)), 24);
    bits = _mm512_or_si512(_mm512_castps_si512(values), sign);
    return _mm512_castsi512_ps(_mm512_mask_mov_epi32(bits,
        _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(127)),
        _mm512_set1_epi32(0x7fc00000)));
}
static inline __m512 v4_fp8_decode16_bytes(__m128i bytes) {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
    __m256i codes = _mm256_cvtepu8_epi16(bytes);
    __m256i magnitude = _mm256_and_si256(codes, _mm256_set1_epi16(127));
    __m256i half = _mm256_or_si256(_mm256_slli_epi16(magnitude, 7),
        _mm256_slli_epi16(_mm256_and_si256(codes, _mm256_set1_epi16(128)), 8));
    __m512 values = _mm512_mul_ps(_mm512_cvtph_ps(half), _mm512_set1_ps(256.0f));
    return _mm512_castsi512_ps(_mm512_mask_mov_epi32(_mm512_castps_si512(values),
        _mm256_cmpeq_epi16_mask(magnitude, _mm256_set1_epi16(127)),
        _mm512_set1_epi32(0x7fc00000)));
#else
    return v4_fp8_decode16(_mm512_cvtepu8_epi32(bytes));
#endif
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

float coli_e8m0_decode(uint8_t value);
/* Process-wide immutable decode table shared by split native-quant units. */
const float *coli_e8m0_table(void);
float coli_e2m1_decode(uint8_t nibble);
float coli_e4m3fn_decode(uint8_t value);
uint8_t coli_e4m3fn_encode(float value);
float coli_bf16_round(float value);
float coli_bf16_decode(uint16_t value);
void coli_bf16_round_array(float *values, size_t count);

/* Simulates the official dynamic E4M3 activation quantization with one E8M0
 * power-of-two scale per block. Output contains the dequantized FP32 values. */
int coli_fp8_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size);
int coli_fp4_activation_qdq_ref(float *output, uint8_t *scales,
                                const float *input, size_t length,
                                size_t block_size);
int coli_hadamard_bf16_ref(float *values, size_t length);

/* Correctness-first FP4 matvec. The input is dynamically quantized to E4M3 in
 * blocks of 128, weights are native E2M1 with one E8M0 scale per 32 K, and
 * accumulation is FP32. */
int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input);

/* #1136 convergence core: row-major fp4 matvec accumulating in the rows16
 * kernels' per-row order — (x*w)*scale folded straight into the row
 * accumulator, column by column, product rounded before the add — so a
 * rows16-packed and a row-major copy of the same matrix produce identical
 * bits. Requires I%32==0 and O%16==0 (the only shapes rows16 can pack);
 * `x` is the already-qdq'd activation. */
void coli_fp4_matvec_rows16_order(float *y, const uint8_t *q4,
                                  const uint8_t *e8s, const float *x,
                                  int I, int O);
/* Batch-major companion: independent scalar-order accumulators share each
 * decoded weight tile, so the matrix is streamed once for the whole batch. */
void coli_fp4_matmul_batch_rows16_order(float *y, const uint8_t *q4,
                                        const uint8_t *e8s, const float *x,
                                        int batch, int I, int O);

/* Correctness-first FP8 matvec for native 128x128 E4M3 weight blocks with
 * UE8M0 scales and dynamically quantized E4M3 activations. */
int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                        const float *input);

/* Growable thread-local scratch for the qdq activation buffers shared by the
 * *_ref/_pre matvec and matmul entries (defined in the NATIVE_QUANT unit).
 * Replaces the historical malloc/free pair per call; buffers are never
 * returned. Values computed from them are bit-for-bit unchanged. */
int coli_v4_qdq_scratch(size_t activation_count, size_t scales_count,
                        float **activation, uint8_t **scales);

/* Hoisted-qdq FP8 matvec: `activation` is `input` already passed through
 * coli_fp8_activation_qdq_ref once by the caller (wq_a and wkv consume the
 * same vector); `input` stays raw for the GPU path, exactly as in _ref.
 * Same checks, same compute: bit-identical to _ref. */
int coli_fp8_matvec_pre(float *output, const ColiTensorView *weight,
                        const float *input, const float *activation);

/* Optional CUDA tier (Windows engine build only, COLI_V4_GPU_TIER). The
 * engine-side wrappers in deepseek_v4.c resolve coli_cuda_dsv4.dll through
 * backend_loader_dsv4.c; the matvec_ref implementations dispatch to them when
 * the resident weight mirror (ColiTensorView.gpu) is non-NULL. */
#ifdef COLI_V4_GPU_TIER
int coli_v4_gpu_fp8_matvec(const ColiTensorView *w, float *output,
                           const float *input);
int coli_v4_gpu_matvec_grouped(const ColiTensorView *w, float *output,
                               const float *input, int groups);
int coli_v4_gpu_fp8_matmul_batch(const ColiTensorView *w, float *outputs,
                                 const float *inputs, int batch);
#endif

#ifdef __cplusplus
}
#endif

#endif
