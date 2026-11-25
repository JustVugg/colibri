/* kv16.h -- a KV cache in float16: half the RAM of float32. Each value is clamped to
 * float16's range (+-65504: a value past it is the largest one, never an infinity) and
 * rounded to the nearest float16, ties to even, the IEEE rounding: the CPU's (F16C, or the
 * integer path below), a CUDA kernel's (__float2half_rn) and a Vulkan shader's (the same
 * integer arithmetic as kv16_round_bits) give the same bits, and a row read back widens
 * to the float it was rounded to.
 *
 * Float16, not bfloat16: three more bits of mantissa. On Muse Spark's tiny oracle, whose
 * heads' outputs are normalised (a head whose values cancel magnifies their rounding), K
 * and V in bf16 moved the teacher-forced NLL from 2.0790 to 2.2305 nats a token, in f16
 * to 2.0752.
 *
 * kv16_dot_lanes is dot_f32_lanes (matmul_f32.h) over a float16 row: the same lanes in
 * the same order, so a score is the same bits whether its key comes from the cache or
 * from a step's rounded scratch. */
#ifndef COLI_KV16_H
#define COLI_KV16_H
#include <stdint.h>
#include <string.h>
#include "matmul_f32.h"   /* MATMUL_F32_LANES, MATMUL_F32_MADD */

#define KV16_MAX 65504.0f

/* the float32 bits of x rounded to float16's precision (x within +-65504): the integer
 * reference the Vulkan shader (chain_ew.comp, op 11) follows line for line */
static inline uint32_t kv16_round_bits(uint32_t u) {
    uint32_t s = u & 0x80000000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return u | 0x400000u;            /* a NaN stays one */
    if (a >= 0x38800000u)                                  /* float16's normals: 10 bits of mantissa */
        return s | ((a + 0xfffu + ((a >> 13) & 1u)) & 0xffffe000u);
    /* below 2^-14, float16's subnormals: multiples of 2^-24 */
    uint32_t e = a >> 23, m = (a & 0x7fffffu) | 0x800000u;
    if (e < 102) return s;                                 /* under 2^-25: zero (2^-25 itself a tie, to even) */
    uint32_t sh = 126 - e;                                 /* 14 .. 24: the value is m * 2^-(sh + 24) */
    uint32_t n = (m + (1u << (sh - 1)) - 1u + ((m >> sh) & 1u)) >> sh;
    float f = (float)n * 5.9604644775390625e-8f;           /* n * 2^-24, exact */
    uint32_t r; memcpy(&r, &f, 4);
    return s | r;
}
static inline float kv16_clamp(float x) { return x > KV16_MAX ? KV16_MAX : x < -KV16_MAX ? -KV16_MAX : x; }

#if defined(__FLT16_MAX__) && !defined(KV16_INTEGER)   /* KV16_INTEGER: the integer path (tests) */
typedef _Float16 kv16_t;
static inline uint16_t kv16_from(float f) { kv16_t h = (kv16_t)kv16_clamp(f); uint16_t b; memcpy(&b, &h, 2); return b; }
static inline float kv16_to(uint16_t b) { kv16_t h; memcpy(&h, &b, 2); return (float)h; }
#else
/* no _Float16 in this compiler: the bits by integer arithmetic */
static inline uint16_t kv16_from(float f) {
    f = kv16_clamp(f);
    uint32_t u; memcpy(&u, &f, 4);
    u = kv16_round_bits(u);
    uint32_t s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint16_t)(s | 0x7e00u);
    if (a < 0x38800000u) { float g; memcpy(&g, &a, 4); return (uint16_t)(s | (uint32_t)(g * 16777216.0f)); }   /* n * 2^-24 */
    return (uint16_t)(s | (((a >> 23) - 112u) << 10) | ((a >> 13) & 0x3ffu));
}
static inline float kv16_to(uint16_t b) {
    uint32_t s = ((uint32_t)b & 0x8000u) << 16, e = (b >> 10) & 0x1fu, m = b & 0x3ffu, u;
    if (e == 0) { float g = (float)m * 5.9604644775390625e-8f; memcpy(&u, &g, 4); u |= s; }
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4);
    return f;
}
#endif

/* gcc does not vectorize the _Float16 casts: with F16C the three loops below convert eight
 * values at a time by the instructions those casts compile to (round to nearest even), the
 * clamp's operands ordered so a NaN passes through as kv16_clamp lets it: the same bits.
 * Before, kv16_round took about 0.2 ms of an OLMoE decode token (16 layers' K and V rows
 * down from the Vulkan chain) and about 20 ms of a 3K-token prompt. */
#if defined(__F16C__) && defined(__AVX__) && defined(__FLT16_MAX__) && !defined(KV16_INTEGER)
#include <immintrin.h>
#define KV16_CVT8 1
static inline __m256 kv16_clamp8(__m256 v) {
    return _mm256_max_ps(_mm256_set1_ps(-KV16_MAX), _mm256_min_ps(_mm256_set1_ps(KV16_MAX), v));
}
#endif
/* x rounded in place to the floats its float16 holds */
static inline void kv16_round(float *x, int64_t n) {
    int64_t i = 0;
#ifdef KV16_CVT8
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_cvtph_ps(_mm256_cvtps_ph(kv16_clamp8(_mm256_loadu_ps(x + i)), _MM_FROUND_TO_NEAREST_INT)));
#endif
    for (; i < n; i++) x[i] = kv16_to(kv16_from(x[i]));
}
static inline void kv16_pack(uint16_t *d, const float *s, int64_t n) {
    int64_t i = 0;
#ifdef KV16_CVT8
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(d + i), _mm256_cvtps_ph(kv16_clamp8(_mm256_loadu_ps(s + i)), _MM_FROUND_TO_NEAREST_INT));
#endif
    for (; i < n; i++) d[i] = kv16_from(s[i]);
}
static inline void kv16_unpack(float *d, const uint16_t *s, int64_t n) {
    int64_t i = 0;
#ifdef KV16_CVT8
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(d + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + i))));
#endif
    for (; i < n; i++) d[i] = kv16_to(s[i]);
}

/* gcc does not vectorize a float16 -> float conversion without AVX512-FP16: the two loops
 * attention runs over every position, with F16C's conversion by hand, lane for lane the
 * scalar ones (lane j of 32 a fused multiply-add chain; the same halving tree after) */
#if defined(__F16C__) && defined(__AVX__) && defined(__FMA__) && MATMUL_F32_LANES == 32
#include <immintrin.h>
#define KV16_F16C 1
#endif

static inline float kv16_dot_lanes(const float *x, const uint16_t *w, int I) {
    float a[MATMUL_F32_LANES] = {0};
    int i = 0;
#ifdef KV16_F16C
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    for (; i + 32 <= I; i += 32) {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))),      c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8),  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8))),  c1);
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 16))), c2);
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 24))), c3);
    }
    _mm256_storeu_ps(a, c0); _mm256_storeu_ps(a + 8, c1); _mm256_storeu_ps(a + 16, c2); _mm256_storeu_ps(a + 24, c3);
#else
    for (; i + MATMUL_F32_LANES <= I; i += MATMUL_F32_LANES)
        for (int j = 0; j < MATMUL_F32_LANES; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], kv16_to(w[i+j]));
#endif
    for (int j = 0; i + j < I; j++) a[j] = MATMUL_F32_MADD(a[j], x[i+j], kv16_to(w[i+j]));
    for (int h = MATMUL_F32_LANES/2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) a[j] += a[j+h];
    return a[0];
}
/* y += a * x, the float16 row widened (the float row's own loop: y + a*x, fused). The
 * fusion is spelled out (MATMUL_F32_MADD) rather than left to the compiler, which fuses
 * y += a*x only from -O2 on: at -O1 (the ASan build) the tail would round twice while the
 * F16C body does not. */
static inline void kv16_axpy(float *y, float a, const uint16_t *x, int n) {
    int i = 0;
#ifdef KV16_F16C
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))), _mm256_loadu_ps(y + i)));
#endif
    for (; i < n; i++) y[i] = MATMUL_F32_MADD(y[i], a, kv16_to(x[i]));
}

/* Two queries against one float16 key row: converts each 8-element chunk once
 * in registers (8 accumulator ymms + 1 key ymm: no spills on AVX2), bit-identical
 * to two kv16_dot_lanes calls. */
static inline void kv16_dot_lanes_x2(float out[2], const float *x0, const float *x1, const uint16_t *w, int I) {
    float a0[MATMUL_F32_LANES] = {0}, a1[MATMUL_F32_LANES] = {0};
    int i = 0;
#ifdef KV16_F16C
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    __m256 d0 = _mm256_setzero_ps(), d1 = d0, d2 = d0, d3 = d0;
    for (; i + 32 <= I; i += 32) {
        __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i)));
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + i),      w0, c0);
        d0 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i),      w0, d0);
        __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8)));
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + i + 8),  w1, c1);
        d1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i + 8),  w1, d1);
        __m256 w2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 16)));
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + i + 16), w2, c2);
        d2 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i + 16), w2, d2);
        __m256 w3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 24)));
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + i + 24), w3, c3);
        d3 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i + 24), w3, d3);
    }
    _mm256_storeu_ps(a0, c0); _mm256_storeu_ps(a0 + 8, c1); _mm256_storeu_ps(a0 + 16, c2); _mm256_storeu_ps(a0 + 24, c3);
    _mm256_storeu_ps(a1, d0); _mm256_storeu_ps(a1 + 8, d1); _mm256_storeu_ps(a1 + 16, d2); _mm256_storeu_ps(a1 + 24, d3);
#else
    for (; i + MATMUL_F32_LANES <= I; i += MATMUL_F32_LANES)
        for (int j = 0; j < MATMUL_F32_LANES; j++) {
            float wv = kv16_to(w[i+j]);
            a0[j] = MATMUL_F32_MADD(a0[j], x0[i+j], wv);
            a1[j] = MATMUL_F32_MADD(a1[j], x1[i+j], wv);
        }
#endif
    for (int j = 0; i + j < I; j++) {
        float wv = kv16_to(w[i+j]);
        a0[j] = MATMUL_F32_MADD(a0[j], x0[i+j], wv);
        a1[j] = MATMUL_F32_MADD(a1[j], x1[i+j], wv);
    }
    for (int h = MATMUL_F32_LANES/2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) { a0[j] += a0[j+h]; a1[j] += a1[j+h]; }
    out[0] = a0[0];
    out[1] = a1[0];
}

/* Two/four outputs accumulating one float16 value row: converts x once in
 * registers, bit-identical to individual kv16_axpy calls. */
static inline void kv16_axpy_x2(float *y0, float *y1, float a0, float a1, const uint16_t *x, int n) {
    int i = 0;
#ifdef KV16_F16C
    __m256 va0 = _mm256_set1_ps(a0), va1 = _mm256_set1_ps(a1);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i)));
        _mm256_storeu_ps(y0 + i, _mm256_fmadd_ps(va0, vx, _mm256_loadu_ps(y0 + i)));
        _mm256_storeu_ps(y1 + i, _mm256_fmadd_ps(va1, vx, _mm256_loadu_ps(y1 + i)));
    }
#endif
    for (; i < n; i++) {
        float vx = kv16_to(x[i]);
        y0[i] = MATMUL_F32_MADD(y0[i], a0, vx);
        y1[i] = MATMUL_F32_MADD(y1[i], a1, vx);
    }
}
static inline void kv16_axpy_x4(float *y0, float *y1, float *y2, float *y3,
                                float a0, float a1, float a2, float a3, const uint16_t *x, int n) {
    int i = 0;
#ifdef KV16_F16C
    __m256 va0 = _mm256_set1_ps(a0), va1 = _mm256_set1_ps(a1), va2 = _mm256_set1_ps(a2), va3 = _mm256_set1_ps(a3);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i)));
        _mm256_storeu_ps(y0 + i, _mm256_fmadd_ps(va0, vx, _mm256_loadu_ps(y0 + i)));
        _mm256_storeu_ps(y1 + i, _mm256_fmadd_ps(va1, vx, _mm256_loadu_ps(y1 + i)));
        _mm256_storeu_ps(y2 + i, _mm256_fmadd_ps(va2, vx, _mm256_loadu_ps(y2 + i)));
        _mm256_storeu_ps(y3 + i, _mm256_fmadd_ps(va3, vx, _mm256_loadu_ps(y3 + i)));
    }
#endif
    for (; i < n; i++) {
        float vx = kv16_to(x[i]);
        y0[i] = MATMUL_F32_MADD(y0[i], a0, vx);
        y1[i] = MATMUL_F32_MADD(y1[i], a1, vx);
        y2[i] = MATMUL_F32_MADD(y2[i], a2, vx);
        y3[i] = MATMUL_F32_MADD(y3[i], a3, vx);
    }
}
#endif
