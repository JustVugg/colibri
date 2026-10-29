/* ggml_quants.h -- the GGUF weight formats, decoded natively.
 *
 * A GGUF file stores each tensor as rows of fixed-size blocks in one of the
 * ggml types (Q4_K, IQ2_XXS, Q8_0, MXFP4 ...). This header lets colibri use
 * those bytes as they are, with no second quantization: every type has a
 * row decoder that reproduces the reference dequantization exactly, and a
 * matrix-vector product built on it.
 *
 * Exactness contract: gq_dequant_row() returns, bit for bit, the float32
 * values the reference (gguf-py, gguf.quants.dequantize) computes. The
 * arithmetic is written in the same order as the reference and the file is
 * compiled without floating-point contraction, so a*b - c is never fused
 * into an FMA with a different rounding. tests/test_ggml_quants.py holds
 * this for every type on random blocks.
 *
 * Block layouts follow the GGUF specification (ggml-common.h in the ggml
 * project). Codebook tables live in ggml_grids.h (generated).
 */
#ifndef GGML_QUANTS_H
#define GGML_QUANTS_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml_grids.h"

#if defined(__AVX2__) || defined(__F16C__)
#include <immintrin.h>
#endif
/* AVX-512 with VNNI (vpdpbusd: four u8 x s8 products into an int32 lane) and VBMI
 * (vpmultishiftqb: bytes from arbitrary bit offsets), as on Zen 4 and Ice Lake on.
 * The kernels under it give the AVX2 kernels' bits: the same int32 lanes, the same
 * float steps in the same order. */
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && \
    defined(__AVX512VNNI__) && defined(__AVX512VBMI__) && defined(__FMA__)
#define GQ_VNNI512 1
#endif
/* |w| (unsigned) times sign(x, w), the four products of each int32 lane summed: one
 * vpdpbusd where AVX-VNNI or AVX-512 VL has it (Alder Lake, Zen 4), else maddubs then
 * madd with ones. Exact either way (a pair is at most 2*127*127 < 2^15), so the lanes
 * and every result built on them are the same bits. */
#if defined(__AVX2__)
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define GQ_DP32(aw, sx) _mm256_dpbusd_epi32(_mm256_setzero_si256(), (aw), (sx))
#elif defined(__AVXVNNI__)
#define GQ_DP32(aw, sx) _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), (aw), (sx))
#else
#define GQ_DP32(aw, sx) _mm256_madd_epi16(_mm256_maddubs_epi16((aw), (sx)), _mm256_set1_epi16(1))
#endif
#endif
#ifdef GQ_VNNI512
/* The scales of a 64-element pair as one vector, so a row's scales are one multiply:
 * w16(a0, a1, b0, b1) = a0 x4 | a1 x4 | b0 x4 | b1 x4 (the weights' per-16 scales of the
 * two sub-blocks, shared by every row), x16(dd) = dd[0] x8 | dd[1] x8 (a row's two
 * activation scales). w16 * x16 lane by lane is the AVX2 kernel's s * dd product for
 * every lane, the same single rounding. */
static inline __m512 gq_w16(float a0, float a1, float b0, float b1) {
    return _mm512_permutexvar_ps(_mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0), _mm512_castps128_ps512(_mm_set_ps(b1, b0, a1, a0)));
}
static inline __m512 gq_x16(const float *dd) {
    return _mm512_permutexvar_ps(_mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                                 _mm512_castps128_ps512(_mm_castsi128_ps(_mm_loadl_epi64((const __m128i *)dd))));
}
#define GQ_FMA2V(acc, pf, sc) \
    acc = _mm256_fmadd_ps(_mm512_castps512_ps256(pf), _mm512_castps512_ps256(sc), acc); \
    acc = _mm256_fmadd_ps(_mm512_extractf32x8_ps(pf, 1), _mm512_extractf32x8_ps(sc, 1), acc);
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__clang__)
#define GQ_EXACT_BEGIN _Pragma("clang fp contract(off)")
#define GQ_EXACT_FN
#elif defined(__GNUC__)
#define GQ_EXACT_BEGIN
#define GQ_EXACT_FN __attribute__((optimize("fp-contract=off")))
#else
#define GQ_EXACT_BEGIN
#define GQ_EXACT_FN
#endif

/* ggml type ids, as stored in GGUF tensor infos. */
enum {
    GQ_F32 = 0, GQ_F16 = 1, GQ_Q4_0 = 2, GQ_Q4_1 = 3, GQ_Q5_0 = 6, GQ_Q5_1 = 7,
    GQ_Q8_0 = 8, GQ_Q8_1 = 9, GQ_Q2_K = 10, GQ_Q3_K = 11, GQ_Q4_K = 12,
    GQ_Q5_K = 13, GQ_Q6_K = 14, GQ_Q8_K = 15, GQ_IQ2_XXS = 16, GQ_IQ2_XS = 17,
    GQ_IQ3_XXS = 18, GQ_IQ1_S = 19, GQ_IQ4_NL = 20, GQ_IQ3_S = 21, GQ_IQ2_S = 22,
    GQ_IQ4_XS = 23, GQ_I8 = 24, GQ_I16 = 25, GQ_I32 = 26, GQ_I64 = 27,
    GQ_F64 = 28, GQ_IQ1_M = 29, GQ_BF16 = 30, GQ_TQ1_0 = 34, GQ_TQ2_0 = 35,
    GQ_MXFP4 = 39, GQ_NVFP4 = 40, GQ_Q1_0 = 41, GQ_Q2_0 = 42, GQ_TYPE_MAX = 43
};

typedef struct { const char *name; int blck; int bytes; int supported; } gq_type_info;

static const gq_type_info GQ_TYPES[GQ_TYPE_MAX] = {
    [GQ_F32]     = { "F32",     1,   4, 1 },
    [GQ_F16]     = { "F16",     1,   2, 1 },
    [GQ_Q4_0]    = { "Q4_0",    32,  18, 1 },
    [GQ_Q4_1]    = { "Q4_1",    32,  20, 1 },
    [GQ_Q5_0]    = { "Q5_0",    32,  22, 1 },
    [GQ_Q5_1]    = { "Q5_1",    32,  24, 1 },
    [GQ_Q8_0]    = { "Q8_0",    32,  34, 1 },
    [GQ_Q8_1]    = { "Q8_1",    32,  36, 0 },
    [GQ_Q2_K]    = { "Q2_K",    256, 84, 1 },
    [GQ_Q3_K]    = { "Q3_K",    256, 110, 1 },
    [GQ_Q4_K]    = { "Q4_K",    256, 144, 1 },
    [GQ_Q5_K]    = { "Q5_K",    256, 176, 1 },
    [GQ_Q6_K]    = { "Q6_K",    256, 210, 1 },
    [GQ_Q8_K]    = { "Q8_K",    256, 292, 0 },
    [GQ_IQ2_XXS] = { "IQ2_XXS", 256, 66, 1 },
    [GQ_IQ2_XS]  = { "IQ2_XS",  256, 74, 1 },
    [GQ_IQ3_XXS] = { "IQ3_XXS", 256, 98, 1 },
    [GQ_IQ1_S]   = { "IQ1_S",   256, 50, 1 },
    [GQ_IQ4_NL]  = { "IQ4_NL",  32,  18, 1 },
    [GQ_IQ3_S]   = { "IQ3_S",   256, 110, 1 },
    [GQ_IQ2_S]   = { "IQ2_S",   256, 82, 1 },
    [GQ_IQ4_XS]  = { "IQ4_XS",  256, 136, 1 },
    [GQ_I8]      = { "I8",      1,   1, 0 },
    [GQ_I16]     = { "I16",     1,   2, 0 },
    [GQ_I32]     = { "I32",     1,   4, 0 },
    [GQ_I64]     = { "I64",     1,   8, 0 },
    [GQ_F64]     = { "F64",     1,   8, 0 },
    [GQ_IQ1_M]   = { "IQ1_M",   256, 56, 1 },
    [GQ_BF16]    = { "BF16",    1,   2, 1 },
    [GQ_TQ1_0]   = { "TQ1_0",   256, 54, 1 },
    [GQ_TQ2_0]   = { "TQ2_0",   256, 66, 1 },
    [GQ_MXFP4]   = { "MXFP4",   32,  17, 1 },
    [GQ_NVFP4]   = { "NVFP4",   64,  36, 1 },
    [GQ_Q1_0]    = { "Q1_0",    128, 18, 1 },
    [GQ_Q2_0]    = { "Q2_0",    64,  18, 1 },
};

/* Not GGUF types: a block type's rows interleaved in groups of 16 (gq_r16_*, below), made by
 * the engine at load from the file's rows. A matrix holds whole groups; a group stores, block
 * after block, its 16 rows' blocks as one group block (gq_r16_group_bytes: 16 blocks' bytes,
 * a little more for the K-quants, whose scales it keeps decoded). */
enum { GQ_Q2_0_R16 = 64, GQ_Q4_K_R16, GQ_Q5_K_R16, GQ_Q6_K_R16, GQ_Q3_K_R16, GQ_IQ4_XS_R16,
       GQ_Q8_0_R16, GQ_Q4_0_R16, GQ_IQ4_NL_R16, GQ_R16_END };
static inline int gq_r16_base(int t) {
    static const int8_t base[] = { GQ_Q2_0, GQ_Q4_K, GQ_Q5_K, GQ_Q6_K, GQ_Q3_K, GQ_IQ4_XS, GQ_Q8_0, GQ_Q4_0, GQ_IQ4_NL };
    return t >= GQ_Q2_0_R16 && t < GQ_R16_END ? base[t - GQ_Q2_0_R16] : -1;
}
static inline int gq_r16_group_bytes(int t) {
    static const int16_t gb[] = { 288, 2368, 2880, 3360, 1824, 2208, 544, 288, 288 };
    return t >= GQ_Q2_0_R16 && t < GQ_R16_END ? gb[t - GQ_Q2_0_R16] : -1;
}
/* the interleaved form of block type t, or -1 */
static inline int gq_r16_of(int t) {
    for (int r = GQ_Q2_0_R16; r < GQ_R16_END; r++) if (gq_r16_base(r) == t) return r;
    return -1;
}
static inline int gq_type_ok(int t) { return t >= 0 && t < GQ_TYPE_MAX && GQ_TYPES[t].blck > 0 && GQ_TYPES[t].supported; }
static inline const char *gq_type_name(int t) {
    static const char *r16[] = { "Q2_0r16", "Q4_Kr16", "Q5_Kr16", "Q6_Kr16", "Q3_Kr16", "IQ4_XSr16", "Q8_0r16", "Q4_0r16", "IQ4_NLr16" };
    if (gq_r16_base(t) >= 0) return r16[t - GQ_Q2_0_R16];
    return (t >= 0 && t < GQ_TYPE_MAX && GQ_TYPES[t].name) ? GQ_TYPES[t].name : "?";
}
/* Bytes of one row of n elements, or -1 when n is not a whole number of blocks (an
 * interleaved type: a group's bytes / 16, so 16 * row bytes steps a group). */
static inline int64_t gq_row_bytes(int t, int64_t n) {
    const int b = gq_r16_base(t);
    if (b >= 0) return n % GQ_TYPES[b].blck ? -1 : n / GQ_TYPES[b].blck * (gq_r16_group_bytes(t) / 16);
    if (t < 0 || t >= GQ_TYPE_MAX || GQ_TYPES[t].blck <= 0) return -1;
    if (n % GQ_TYPES[t].blck) return -1;
    return n / GQ_TYPES[t].blck * GQ_TYPES[t].bytes;
}

/* ---- scalar conversions (exact) ---- */
static inline float gq_f16(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            int e = -1;
            do { e++; mant <<= 1; } while (!(mant & 0x400));
            mant &= 0x3ff;
            f = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) f = sign | 0x7f800000u | (mant << 13);
    else f = sign | ((exp + 112) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
#endif
}
static inline float gq_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float r; memcpy(&r, &u, 4); return r; }
static inline uint16_t gq_rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t gq_rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
/* ggml_e8m0_to_fp32_half: the MXFP4 block exponent, already halved because
 * the value table holds e2m1 values doubled. */
static inline float gq_e8m0_half(uint8_t x) {
    uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23);
    float r; memcpy(&r, &bits, 4); return r;
}

/* ---- per-type block decoders: one block in, blck floats out ---- */

GQ_EXACT_FN static void gq_blk_q4_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; j++) {
        y[j]      = d * (float)((int)(qs[j] & 0x0F) - 8);
        y[j + 16] = d * (float)((int)(qs[j] >> 4) - 8);
    }
}
GQ_EXACT_FN static void gq_blk_q4_1(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), m = gq_f16(gq_rd16(b + 2)); const uint8_t *qs = b + 4;
    for (int j = 0; j < 16; j++) {
        y[j]      = d * (float)(qs[j] & 0x0F) + m;
        y[j + 16] = d * (float)(qs[j] >> 4) + m;
    }
}
GQ_EXACT_FN static void gq_blk_q5_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint32_t qh = gq_rd32(b + 2); const uint8_t *qs = b + 6;
    for (int j = 0; j < 16; j++) {
        int lo = (qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4);
        int hi = (qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
        y[j] = d * (float)(lo - 16); y[j + 16] = d * (float)(hi - 16);
    }
}
GQ_EXACT_FN static void gq_blk_q5_1(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), m = gq_f16(gq_rd16(b + 2));
    const uint32_t qh = gq_rd32(b + 4); const uint8_t *qs = b + 8;
    for (int j = 0; j < 16; j++) {
        int lo = (qs[j] & 0x0F) | (int)(((qh >> j) & 1) << 4);
        int hi = (qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
        y[j] = d * (float)lo + m; y[j + 16] = d * (float)hi + m;
    }
}
GQ_EXACT_FN static void gq_blk_q8_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const int8_t *qs = (const int8_t *)(b + 2);
    for (int j = 0; j < 32; j++) y[j] = (float)qs[j] * d;
}
GQ_EXACT_FN static void gq_blk_q2_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *sc = b, *qs = b + 16;
    const float d = gq_f16(gq_rd16(b + 80)), dmin = gq_f16(gq_rd16(b + 82));
    for (int i = 0; i < 256; i++) {
        int c = i >> 7, s = (i >> 5) & 3, j = i & 31, g = i >> 4;
        int q = (qs[c * 32 + j] >> (2 * s)) & 3;
        float dl = d * (float)(sc[g] & 0xF), ml = dmin * (float)(sc[g] >> 4);
        y[i] = dl * (float)q - ml;
    }
}
GQ_EXACT_FN static void gq_blk_q3_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *hm = b, *qs = b + 32, *s = b + 96;
    const float d = gq_f16(gq_rd16(b + 108));
    float dl[16];
    for (int k = 0; k < 16; k++) {
        int l = k < 8 ? (s[k] & 0x0F) : (s[k - 8] >> 4);
        int h = (s[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
        dl[k] = d * (float)((l | (h << 4)) - 32);
    }
    for (int i = 0; i < 256; i++) {
        int c = i >> 7, sh = (i >> 5) & 3, j = i & 31;
        int ql = (qs[c * 32 + j] >> (2 * sh)) & 3;
        int hb = (hm[i & 31] >> (i >> 5)) & 1;
        int q = ql - ((hb ^ 1) << 2);
        y[i] = dl[i >> 4] * (float)q;
    }
}
static inline void gq_scale_min_k4(const uint8_t *s, int g, int *sc, int *mn) {
    if (g < 4) { *sc = s[g] & 0x3F; *mn = s[g + 4] & 0x3F; }
    else {
        *sc = (s[g + 4] & 0x0F) | ((s[g - 4] >> 2) & 0x30);
        *mn = (s[g + 4] >> 4)   | ((s[g] >> 2) & 0x30);
    }
}
GQ_EXACT_FN static void gq_blk_q4_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
    const uint8_t *sc = b + 4, *qs = b + 16;
    for (int g = 0; g < 8; g++) {
        int s, m; gq_scale_min_k4(sc, g, &s, &m);
        const float d1 = d * (float)s, m1 = dmin * (float)m;
        const uint8_t *q = qs + (g >> 1) * 32; int sh = (g & 1) * 4;
        for (int j = 0; j < 32; j++) y[g * 32 + j] = d1 * (float)((q[j] >> sh) & 0x0F) - m1;
    }
}
GQ_EXACT_FN static void gq_blk_q5_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
    const uint8_t *sc = b + 4, *qh = b + 16, *qs = b + 48;
    for (int g = 0; g < 8; g++) {
        int s, m; gq_scale_min_k4(sc, g, &s, &m);
        const float d1 = d * (float)s, m1 = dmin * (float)m;
        const uint8_t *q = qs + (g >> 1) * 32; int sh = (g & 1) * 4;
        for (int j = 0; j < 32; j++) {
            int v = ((q[j] >> sh) & 0x0F) | (((qh[j] >> g) & 1) << 4);
            y[g * 32 + j] = d1 * (float)v - m1;
        }
    }
}
GQ_EXACT_FN static void gq_blk_q6_k(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
    const float d = gq_f16(gq_rd16(b + 208));
    for (int i = 0; i < 256; i++) {
        int c = i >> 7, r = i & 127;                 /* chunk of 128 */
        int nib = r >> 6, k = r & 63;                /* ql: low nibbles 0..63, high 64..127 */
        int lo = (ql[c * 64 + k] >> (4 * nib)) & 0x0F;
        int s = r >> 5, j = r & 31;                  /* qh: 2 bits, 4 shifts of 32 */
        int hi = (qh[c * 32 + j] >> (2 * s)) & 3;
        int q = (lo | (hi << 4)) - 32;
        y[i] = (d * (float)sc[i >> 4]) * (float)q;
    }
}
GQ_EXACT_FN static void gq_blk_iq2_xxs(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
    for (int ib = 0; ib < 8; ib++) {
        uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
        const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = GGML_IQ2XXS_GRID + 8 * ((a0 >> (8 * l)) & 0xFF);
            uint8_t signs = GGML_KSIGNS_IQ2XS[(a1 >> (7 * l)) & 127];
            for (int j = 0; j < 8; j++) {
                float v = db * (float)grid[j];
                y[ib * 32 + l * 8 + j] = (signs >> j) & 1 ? -v : v;
            }
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq2_xs(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sc = b + 66;
    for (int k = 0; k < 32; k++) {                   /* 32 groups of 8 */
        uint16_t q = gq_rd16(qs + 2 * k);
        int si = k >> 1;                              /* one 4-bit scale per 16 */
        int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F;
        const float db = d * (0.5f + (float)s) * 0.25f;
        const uint8_t *grid = GGML_IQ2XS_GRID + 8 * (q & 511);
        uint8_t signs = GGML_KSIGNS_IQ2XS[q >> 9];
        for (int j = 0; j < 8; j++) {
            float v = db * (float)grid[j];
            y[k * 8 + j] = (signs >> j) & 1 ? -v : v;
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq2_s(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b));
    const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
    for (int k = 0; k < 32; k++) {
        int si = k >> 1;
        int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F;
        const float db = d * (0.5f + (float)s) * 0.25f;
        int idx = qs[k] | (((qh[k >> 2] >> (2 * (k & 3))) & 3) << 8);
        const uint8_t *grid = GGML_IQ2S_GRID + 8 * idx;
        for (int j = 0; j < 8; j++) {
            float v = db * (float)grid[j];
            y[k * 8 + j] = (sg[k] >> j) & 1 ? -v : v;
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq3_xxs(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *ss = b + 66;
    for (int ib = 0; ib < 8; ib++) {
        uint32_t a = gq_rd32(ss + 4 * ib);
        const float db = d * (0.5f + (float)(a >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            uint8_t signs = GGML_KSIGNS_IQ2XS[(a >> (7 * l)) & 127];
            const uint8_t *g1 = GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l];
            const uint8_t *g2 = GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l + 1];
            for (int j = 0; j < 4; j++) {
                float v1 = db * (float)g1[j], v2 = db * (float)g2[j];
                y[ib * 32 + l * 8 + j]     = (signs >> j) & 1 ? -v1 : v1;
                y[ib * 32 + l * 8 + j + 4] = (signs >> (j + 4)) & 1 ? -v2 : v2;
            }
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq3_s(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b));
    const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
    for (int k = 0; k < 64; k++) {                   /* 64 groups of 4 */
        int ib = k >> 3;                              /* 32-element block */
        int s = (sc[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        const float db = d * (float)(1 + 2 * s);
        int idx = qs[k] | (((qh[k >> 3] >> (k & 7)) & 1) << 8);
        const uint8_t *grid = GGML_IQ3S_GRID + 4 * idx;
        for (int j = 0; j < 4; j++) {
            int e = k * 4 + j;
            float v = db * (float)grid[j];
            y[e] = (sg[e >> 3] >> (e & 7)) & 1 ? -v : v;
        }
    }
}
#define GQ_IQ1_DELTA 0.125f
GQ_EXACT_FN static void gq_blk_iq1_s(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qhp = b + 34;
    for (int ib = 0; ib < 8; ib++) {
        uint16_t qh = gq_rd16(qhp + 2 * ib);
        const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -GQ_IQ1_DELTA : GQ_IQ1_DELTA;
        for (int l = 0; l < 4; l++) {
            int idx = qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
            const int8_t *grid = GGML_IQ1S_GRID + 8 * idx;
            for (int j = 0; j < 8; j++) y[ib * 32 + l * 8 + j] = dl * ((float)grid[j] + delta);
        }
    }
}
GQ_EXACT_FN static void gq_blk_iq1_m(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *qs = b, *qh = b + 32, *scp = b + 48;
    uint16_t sc[4]; for (int i = 0; i < 4; i++) sc[i] = gq_rd16(scp + 2 * i);
    uint16_t dh = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const float d = gq_f16(dh);
    for (int k = 0; k < 32; k++) {                   /* 32 groups of 8 */
        int s16 = k >> 1;                             /* 3-bit scale per 16 */
        int s = (sc[s16 >> 2] >> (3 * (s16 & 3))) & 7;
        const float dl = d * (float)(2 * s + 1);
        int h = (qh[k >> 1] >> (4 * (k & 1))) & 0x0F;
        int idx = qs[k] | ((h & 7) << 8);
        const float delta = (h & 8) ? -GQ_IQ1_DELTA : GQ_IQ1_DELTA;
        const int8_t *grid = GGML_IQ1S_GRID + 8 * idx;
        for (int j = 0; j < 8; j++) y[k * 8 + j] = dl * ((float)grid[j] + delta);
    }
}
GQ_EXACT_FN static void gq_blk_iq4_nl(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; j++) {
        y[j]      = d * (float)GGML_KVALUES_IQ4NL[qs[j] & 0x0F];
        y[j + 16] = d * (float)GGML_KVALUES_IQ4NL[qs[j] >> 4];
    }
}
GQ_EXACT_FN static void gq_blk_iq4_xs(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint16_t sh = gq_rd16(b + 2);
    const uint8_t *sl = b + 4, *qs = b + 8;
    for (int ib = 0; ib < 8; ib++) {
        int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        const uint8_t *q = qs + 16 * ib;
        for (int j = 0; j < 16; j++) {
            y[ib * 32 + j]      = dl * (float)GGML_KVALUES_IQ4NL[q[j] & 0x0F];
            y[ib * 32 + j + 16] = dl * (float)GGML_KVALUES_IQ4NL[q[j] >> 4];
        }
    }
}
GQ_EXACT_FN static void gq_blk_mxfp4(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_e8m0_half(b[0]); const uint8_t *qs = b + 1;
    for (int j = 0; j < 16; j++) {
        y[j]      = d * (float)GGML_KVALUES_MXFP4[qs[j] & 0x0F];
        y[j + 16] = d * (float)GGML_KVALUES_MXFP4[qs[j] >> 4];
    }
}
GQ_EXACT_FN static void gq_blk_tq1_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    const uint8_t *qs = b, *qh = b + 48; const float d = gq_f16(gq_rd16(b + 52));
    int o = 0;
    for (int n = 0; n < 5; n++) for (int j = 0; j < 32; j++) {
        uint8_t v = (uint8_t)(qs[j] * pow3[n]); y[o++] = d * (float)((int)(((uint16_t)v * 3) >> 8) - 1);
    }
    for (int n = 0; n < 5; n++) for (int j = 0; j < 16; j++) {
        uint8_t v = (uint8_t)(qs[32 + j] * pow3[n]); y[o++] = d * (float)((int)(((uint16_t)v * 3) >> 8) - 1);
    }
    for (int n = 0; n < 4; n++) for (int j = 0; j < 4; j++) {
        uint8_t v = (uint8_t)(qh[j] * pow3[n]); y[o++] = d * (float)((int)(((uint16_t)v * 3) >> 8) - 1);
    }
}
GQ_EXACT_FN static void gq_blk_tq2_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *qs = b; const float d = gq_f16(gq_rd16(b + 64));
    for (int i = 0; i < 256; i++) {
        int c = i >> 7, s = (i >> 5) & 3, j = i & 31;
        y[i] = d * (float)((int)((qs[c * 32 + j] >> (2 * s)) & 3) - 1);
    }
}

/* NVFP4 block scale: unsigned E4M3 (bias 7), halved like the MXFP4 exponent
 * because the value table holds e2m1 values doubled; 0 and 0x7F decode to 0. */
static inline float gq_ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.f;
    int e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? ldexpf((float)m, -10) : ldexpf((float)(8 + m), e - 11);
}
GQ_EXACT_FN static void gq_blk_nvfp4(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const uint8_t *qs = b + 4;
    for (int sb = 0; sb < 4; sb++) {
        const float d = gq_ue4m3_half(b[sb]);
        for (int j = 0; j < 8; j++) {
            y[16 * sb + j]     = d * (float)GGML_KVALUES_MXFP4[qs[8 * sb + j] & 0x0F];
            y[16 * sb + 8 + j] = d * (float)GGML_KVALUES_MXFP4[qs[8 * sb + j] >> 4];
        }
    }
}
GQ_EXACT_FN static void gq_blk_q1_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)), nd = -d; const uint8_t *qs = b + 2;
    for (int j = 0; j < 128; j++) y[j] = (qs[j >> 3] >> (j & 7)) & 1 ? d : nd;
}
GQ_EXACT_FN static void gq_blk_q2_0(const uint8_t *b, float *y) {
    GQ_EXACT_BEGIN
    const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
    for (int j = 0; j < 64; j++) y[j] = (float)((int)((qs[j >> 2] >> ((j & 3) * 2)) & 3) - 1) * d;
}

typedef void (*gq_blk_fn)(const uint8_t *, float *);
static inline gq_blk_fn gq_blk_decoder(int t) {
    switch (t) {
    case GQ_Q4_0: return gq_blk_q4_0;     case GQ_Q4_1: return gq_blk_q4_1;
    case GQ_Q5_0: return gq_blk_q5_0;     case GQ_Q5_1: return gq_blk_q5_1;
    case GQ_Q8_0: return gq_blk_q8_0;     case GQ_Q2_K: return gq_blk_q2_k;
    case GQ_Q3_K: return gq_blk_q3_k;     case GQ_Q4_K: return gq_blk_q4_k;
    case GQ_Q5_K: return gq_blk_q5_k;     case GQ_Q6_K: return gq_blk_q6_k;
    case GQ_IQ2_XXS: return gq_blk_iq2_xxs; case GQ_IQ2_XS: return gq_blk_iq2_xs;
    case GQ_IQ2_S: return gq_blk_iq2_s;   case GQ_IQ3_XXS: return gq_blk_iq3_xxs;
    case GQ_IQ3_S: return gq_blk_iq3_s;   case GQ_IQ1_S: return gq_blk_iq1_s;
    case GQ_IQ1_M: return gq_blk_iq1_m;   case GQ_IQ4_NL: return gq_blk_iq4_nl;
    case GQ_IQ4_XS: return gq_blk_iq4_xs; case GQ_MXFP4: return gq_blk_mxfp4;
    case GQ_TQ1_0: return gq_blk_tq1_0;   case GQ_TQ2_0: return gq_blk_tq2_0;
    case GQ_NVFP4: return gq_blk_nvfp4;   case GQ_Q1_0: return gq_blk_q1_0;
    case GQ_Q2_0: return gq_blk_q2_0;
    default: return NULL;
    }
}

/* Decode n elements (a whole number of blocks) of type t into dst.
 * Returns 0, or -1 for an unsupported type or a partial block. */
static int gq_dequant_row(int t, const uint8_t *src, float *dst, int64_t n);
static int gq_r16_dequant_row(int t, const uint8_t *W, int64_t row, float *out, int I);
/* One matrix of ggml rows (or of a GQ_*_R16 form) as int8 with one f32 scale per 32 weights (the Vulkan tier's
 * VKT_SRC_I8_GS, fmt 13), times mul (an expert's ".scale" factor, else 1). Measured on
 * Qwen3.6's unsloth levels (IQ1_M to Q8_0): +0.4-0.55% relative error on top of each
 * level's own against Q8_0, and exactly the weights for Q8_0; int4 would add 7-10%. */
static void gq_rows_i8(int t, const uint8_t *src, int rows, int cols, float mul, int8_t *dst, float *sc) {
    const int64_t rb = gq_row_bytes(t, cols); const int ng = cols / 32;
    #pragma omp parallel
    {
        float *row = (float *)malloc((size_t)cols * sizeof(float));
        #pragma omp for schedule(static)
        for (int o = 0; o < rows; o++) {
            if (!row) continue;
            if (gq_r16_base(t) >= 0) gq_r16_dequant_row(t, src, o, row, cols);
            else gq_dequant_row(t, src + (int64_t)o * rb, row, cols);
            for (int g = 0; g < ng; g++) {
                const float *v = row + g * 32; float a = 0.f;
                for (int i = 0; i < 32; i++) { float f = fabsf(v[i]); if (f > a) a = f; }
                float scale = a / 127.f, inv = a > 0.f ? 127.f / a : 0.f;
                int8_t *q = dst + (int64_t)o * cols + g * 32;
                for (int i = 0; i < 32; i++) { long r = lrintf(v[i] * inv); q[i] = (int8_t)(r > 127 ? 127 : r < -127 ? -127 : r); }
                sc[(int64_t)o * ng + g] = scale * mul;
            }
        }
        free(row);
    }
}
static int gq_dequant_row(int t, const uint8_t *src, float *dst, int64_t n) {
    if (t == GQ_F32) { memcpy(dst, src, (size_t)n * 4); return 0; }
    if (t == GQ_F16) { for (int64_t i = 0; i < n; i++) dst[i] = gq_f16(gq_rd16(src + 2 * i)); return 0; }
    if (t == GQ_BF16) { for (int64_t i = 0; i < n; i++) dst[i] = gq_bf16(gq_rd16(src + 2 * i)); return 0; }
    gq_blk_fn f = gq_blk_decoder(t);
    if (!f || n % GQ_TYPES[t].blck) return -1;
    const int bl = GQ_TYPES[t].blck, by = GQ_TYPES[t].bytes;
    for (int64_t i = 0; i < n / bl; i++) f(src + i * by, dst + i * bl);
    return 0;
}


/* ---- matrix x vector on the decoded rows ------------------------------
 * y[s*O + o] = sum_i W[o, i] * x[s*I + i], W stored as O rows of type t.
 * Reference path: each row is decoded to float once and dotted with every
 * activation row. Rows are split across OpenMP threads; each thread owns a
 * row buffer. Deterministic: the per-row sum order is fixed. */
static inline float gq_dot_f32(const float *a, const float *b, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h)); h = _mm_add_ss(h, _mm_movehdup_ps(h));
    float r = _mm_cvtss_f32(h);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
#else
    float r = 0.f; for (int64_t i = 0; i < n; i++) r += a[i] * b[i]; return r;
#endif
}

/* Returns 0, or -1 for an unsupported type / shape (nothing written). */
static int gq_matmul_f32(float *y, const float *x, const uint8_t *W, int t, int S, int I, int O) {
    const int64_t rb = gq_row_bytes(t, I);
    if (rb < 0 || (t != GQ_F32 && t != GQ_F16 && t != GQ_BF16 && !gq_blk_decoder(t))) return -1;
    #pragma omp parallel
    {
        float *row = (float *)malloc((size_t)I * sizeof(float));
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            if (!row) continue;
            gq_dequant_row(t, W + (int64_t)o * rb, row, I);
            for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = gq_dot_f32(row, x + (int64_t)s * I, I);
        }
        free(row);
    }
    return 0;
}


/* ---- integer form: every type as int8 values + a scale and a min per 16 ----
 * gq_unpack_row(t, src, n, w8, sw, mw) writes, for element i of group g = i/16,
 *     w8[i] (int8)   sw[g] (float)   mw[g] (float)
 * such that  sw[g] * w8[i] - mw[g]  is, bit for bit, what gq_dequant_row gives.
 * The products are formed with the same operations as the decoders above, so
 * the identity is exact, and it is what the integer dot product below uses:
 * one kernel for every type, the type only decides how the row is unpacked.
 * IQ1 folds its +-1/8 delta into the integer (8*grid +- 1, scale /8). */
#define GQ_G 16
static inline void gq_set_g(float *sw, float *mw, int g0, int ng, float s, float m) {
    for (int g = g0; g < g0 + ng; g++) { sw[g] = s; mw[g] = m; }
}
/* sign byte -> 8 int8 of +1 / -1 (bit j set = element j negative). A generated
 * constant (ggml_grids.h): it was once filled lazily inside the parallel
 * unpack, and a thread could see "ready" before the stores that built it --
 * wrong signs on some rows, depending on timing. Found by comparing the
 * per-layer state of the AVX2, scalar and NEON builds. */
#define GQ_SIGN8(v) (GGML_SIGN8 + 8 * (v))
static inline void gq_sign8_init(void) {}
#if defined(__AVX2__)
/* the four sign bytes (bit j of byte k: element 8k+j negative) applied to 32 grid bytes as a
 * byte mask and a conditional negation, (v ^ m) - m: the values of _mm256_sign_epi8 against
 * the GQ_SIGN8 rows, without reading four of them (the grids hold no -128) */
static inline __m256i gq_signs4(__m256i v, uint32_t bits) {
    const __m256i pick = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), pick), bit), bit);
    return _mm256_sub_epi8(_mm256_xor_si256(v, m), m);
}
static inline uint32_t gq_sbytes4(int s0, int s1, int s2, int s3) {
    return (uint32_t)(s0 & 0xFF) | (uint32_t)(s1 & 0xFF) << 8 | (uint32_t)(s2 & 0xFF) << 16 | (uint32_t)(s3 & 0xFF) << 24;
}
/* four 8-byte groups (grid rows g0..g3, sign bytes s0..s3) -> 32 signed int8 */
static inline __m256i gq_grid4(const uint8_t *g0, const uint8_t *g1, const uint8_t *g2, const uint8_t *g3,
                               int s0, int s1, int s2, int s3) {
    uint64_t a, b, c, d;
    memcpy(&a, g0, 8); memcpy(&b, g1, 8); memcpy(&c, g2, 8); memcpy(&d, g3, 8);
    return gq_signs4(_mm256_set_epi64x((long long)d, (long long)c, (long long)b, (long long)a), gq_sbytes4(s0, s1, s2, s3));
}
/* A sub-block's grid rows in one gather from their indices (four of 8 bytes, or eight of
 * 4), for the types whose index takes its high bits from another field (IQ2_S, IQ3_S):
 * the scalar index math and the loads and inserts gq_grid4 needs cost more than the
 * gather there (one Zen 4 core, x86-64-v3 build: IQ2_S 2.50 -> 3.11 GB/s, IQ3_S 2.15 ->
 * 4.14). Where the index is a plain byte or field (IQ2_XXS, IQ2_XS, IQ3_XXS) the loads
 * were faster: gq_grid4 stays. */
static inline __m256i gq_gather8x4(const uint8_t *grid, __m128i ix) { return _mm256_i32gather_epi64((const long long *)grid, ix, 8); }
static inline __m256i gq_gather4x8(const uint8_t *grid, __m256i ix) { return _mm256_i32gather_epi32((const int *)grid, ix, 4); }
/* 32 bytes -> low nibbles and high nibbles as two 32-byte vectors */
static inline void gq_nib32(const uint8_t *p, __m256i *lo, __m256i *hi) {
    const __m256i m = _mm256_set1_epi8(0x0F), v = _mm256_loadu_si256((const __m256i *)p);
    *lo = _mm256_and_si256(v, m); *hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m);
}
/* 16 bytes -> 32 values: low nibbles then high nibbles, looked up in tab */
static inline __m256i gq_nib16_lut(const uint8_t *p, __m128i tab) {
    const __m128i m = _mm_set1_epi8(0x0F), v = _mm_loadu_si128((const __m128i *)p);
    __m128i lo = _mm_shuffle_epi8(tab, _mm_and_si128(v, m)), hi = _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(v, 4), m));
    return _mm256_set_m128i(hi, lo);
}
#endif
#if defined(__AVX2__)
/* Sub-block decoders of the 256-element grid types: 32 elements of sub-block ib (0..7)
 * of block b as 32 int8 in a register, and the scales of its two 16-element groups
 * (s[0], s[1]), from the block scale d (gq_bd_*). gq_ublk stores them (the unpack),
 * gq_dot_sb256 multiplies them at once: one decoder per format, so both paths hold the
 * same integers and scales by construction. */
#define GQ_SB_FN GQ_EXACT_FN static inline __attribute__((always_inline))
GQ_SB_FN float gq_bd_f16(const uint8_t *b) { return gq_f16(gq_rd16(b)); }
GQ_SB_FN float gq_bd_iq4_xs(const uint8_t *b) { return gq_f16(gq_rd16(b)); }
GQ_SB_FN float gq_bd_iq1_m(const uint8_t *b) {
    uint16_t sc[4]; for (int i = 0; i < 4; i++) sc[i] = gq_rd16(b + 48 + 2 * i);
    return gq_f16((uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)));
}
GQ_SB_FN __m256i gq_sb_iq4_xs(const uint8_t *b, float d, int ib, float *s) {
    const uint16_t sh = gq_rd16(b + 2); const uint8_t *sl = b + 4, *qs = b + 8;
    int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
    s[0] = s[1] = d * (float)(ls - 32);
    return gq_nib16_lut(qs + 16 * ib, _mm_loadu_si128((const __m128i *)GGML_KVALUES_IQ4NL));
}
GQ_SB_FN __m256i gq_sb_iq2_xxs(const uint8_t *b, float d, int ib, float *s) {
    const uint8_t *qs = b + 2;
    uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
    s[0] = s[1] = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
    const uint8_t *G = GGML_IQ2XXS_GRID;
    return gq_grid4(G + 8 * (a0 & 0xFF), G + 8 * ((a0 >> 8) & 0xFF), G + 8 * ((a0 >> 16) & 0xFF), G + 8 * (a0 >> 24),
        GGML_KSIGNS_IQ2XS[a1 & 127], GGML_KSIGNS_IQ2XS[(a1 >> 7) & 127], GGML_KSIGNS_IQ2XS[(a1 >> 14) & 127], GGML_KSIGNS_IQ2XS[(a1 >> 21) & 127]);
}
GQ_SB_FN __m256i gq_sb_iq2_xs(const uint8_t *b, float d, int ib, float *s) {
    const uint8_t *qs = b + 2, *sc = b + 66;
    for (int h = 0; h < 2; h++) { int si = 2 * ib + h; int v = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; s[h] = d * (0.5f + (float)v) * 0.25f; }
    uint16_t q0 = gq_rd16(qs + 8 * ib), q1 = gq_rd16(qs + 8 * ib + 2), q2 = gq_rd16(qs + 8 * ib + 4), q3 = gq_rd16(qs + 8 * ib + 6);
    const uint8_t *G = GGML_IQ2XS_GRID;
    return gq_grid4(G + 8 * (q0 & 511), G + 8 * (q1 & 511), G + 8 * (q2 & 511), G + 8 * (q3 & 511),
        GGML_KSIGNS_IQ2XS[q0 >> 9], GGML_KSIGNS_IQ2XS[q1 >> 9], GGML_KSIGNS_IQ2XS[q2 >> 9], GGML_KSIGNS_IQ2XS[q3 >> 9]);
}
GQ_SB_FN __m256i gq_sb_iq2_s(const uint8_t *b, float d, int ib, float *s) {
    const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
    for (int h = 0; h < 2; h++) { int si = 2 * ib + h; int v = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; s[h] = d * (0.5f + (float)v) * 0.25f; }
    /* element k = 4 ib + j: index qs[k] | ((qh[ib] >> 2 j) & 3) << 8 */
    const __m128i lo = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)gq_rd32(qs + 4 * ib)));
    const __m128i hi = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(qh[ib]), _mm_setr_epi32(0, 2, 4, 6)), _mm_set1_epi32(3));
    return gq_signs4(gq_gather8x4(GGML_IQ2S_GRID, _mm_or_si128(lo, _mm_slli_epi32(hi, 8))), gq_rd32(sg + 4 * ib));
}
GQ_SB_FN __m256i gq_sb_iq3_xxs(const uint8_t *b, float d, int ib, float *s) {
    const uint8_t *qs = b + 2, *ss = b + 66;
    uint32_t a = gq_rd32(ss + 4 * ib);
    s[0] = s[1] = d * (0.5f + (float)(a >> 28)) * 0.5f;
    uint8_t gb[32]; const uint8_t *q = qs + 8 * ib;
    for (int j = 0; j < 8; j++) memcpy(gb + 4 * j, GGML_IQ3XXS_GRID + 4 * q[j], 4);
    return gq_grid4(gb, gb + 8, gb + 16, gb + 24,
        GGML_KSIGNS_IQ2XS[a & 127], GGML_KSIGNS_IQ2XS[(a >> 7) & 127], GGML_KSIGNS_IQ2XS[(a >> 14) & 127], GGML_KSIGNS_IQ2XS[(a >> 21) & 127]);
}
GQ_SB_FN __m256i gq_sb_iq3_s(const uint8_t *b, float d, int ib, float *s) {
    const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
    int v = (sc[ib >> 1] >> (4 * (ib & 1))) & 0x0F; s[0] = s[1] = d * (float)(1 + 2 * v);
    /* element k = 8 ib + j: index qs[k] | ((qh[ib] >> j) & 1) << 8 */
    const __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 8 * ib)));
    const __m256i hi = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(qh[ib]), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)), _mm256_set1_epi32(1));
    return gq_signs4(gq_gather4x8(GGML_IQ3S_GRID, _mm256_or_si256(lo, _mm256_slli_epi32(hi, 8))), gq_rd32(sg + 4 * ib));
}
GQ_SB_FN __m256i gq_sb_iq1_s(const uint8_t *b, float d, int ib, float *s) {
    const uint8_t *qs = b + 2, *qhp = b + 34;
    uint16_t qh = gq_rd16(qhp + 2 * ib);
    s[0] = s[1] = d * (float)(2 * ((qh >> 12) & 7) + 1) * 0.125f;
    uint64_t g[4];
    for (int l = 0; l < 4; l++) memcpy(&g[l], GGML_IQ1S_GRID + 8 * (qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)), 8);
    __m256i v = _mm256_set_epi64x((long long)g[3], (long long)g[2], (long long)g[1], (long long)g[0]);
    v = _mm256_add_epi8(_mm256_add_epi8(v, v), _mm256_add_epi8(v, v));      /* 4x */
    v = _mm256_add_epi8(v, v);                                               /* 8x */
    return _mm256_add_epi8(v, _mm256_set1_epi8((qh & 0x8000) ? -1 : 1));
}
GQ_SB_FN __m256i gq_sb_iq1_m(const uint8_t *b, float d, int ib, float *s) {
    const uint8_t *qs = b, *qh = b + 32, *scp = b + 48;
    uint16_t sc[4]; for (int i = 0; i < 4; i++) sc[i] = gq_rd16(scp + 2 * i);
    for (int h = 0; h < 2; h++) { int s16 = 2 * ib + h; int v = (sc[s16 >> 2] >> (3 * (s16 & 3))) & 7; s[h] = d * (float)(2 * v + 1) * 0.125f; }
    uint64_t g[4]; int8_t ds[4];
    for (int l = 0; l < 4; l++) {
        int k = 4 * ib + l, hh = (qh[k >> 1] >> (4 * (k & 1))) & 0x0F;
        memcpy(&g[l], GGML_IQ1S_GRID + 8 * (qs[k] | ((hh & 7) << 8)), 8); ds[l] = (hh & 8) ? -1 : 1;
    }
    __m256i v = _mm256_set_epi64x((long long)g[3], (long long)g[2], (long long)g[1], (long long)g[0]);
    v = _mm256_add_epi8(v, v); v = _mm256_add_epi8(v, v); v = _mm256_add_epi8(v, v);   /* 8 * grid, byte-wise (a 16-bit shift would carry 0xFF into the neighbour) */
    __m256i dl = _mm256_set_epi64x((long long)(0x0101010101010101ull * (uint8_t)ds[3]), (long long)(0x0101010101010101ull * (uint8_t)ds[2]),
                                   (long long)(0x0101010101010101ull * (uint8_t)ds[1]), (long long)(0x0101010101010101ull * (uint8_t)ds[0]));
    return _mm256_add_epi8(v, dl);
}
/* The K-quants' sub-blocks: 32 elements, their two group scales s[] and mins m[]
 * (sw and mw of the unpack: dequantized value = s * w - m). */
GQ_SB_FN __m256i gq_sbk_q4_5(const uint8_t *b, int ib, float *s, float *m, int five) {
    const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
    int sc, mn; gq_scale_min_k4(b + 4, ib, &sc, &mn);
    s[0] = s[1] = d * (float)sc; m[0] = m[1] = dmin * (float)mn;
    const int c = ib >> 1, hi = ib & 1;
    const __m256i v = _mm256_loadu_si256((const __m256i *)((five ? b + 48 : b + 16) + 32 * c)), m4 = _mm256_set1_epi8(0x0F);
    __m256i w = _mm256_and_si256(hi ? _mm256_srli_epi16(v, 4) : v, m4);
    if (five) {
        const __m256i qh = _mm256_loadu_si256((const __m256i *)(b + 16)), bit = _mm256_set1_epi8((char)(1 << ib));
        w = _mm256_or_si256(w, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit), _mm256_set1_epi8(16)));
    }
    return w;
}
GQ_SB_FN __m256i gq_sbk_q4_k(const uint8_t *b, int ib, float *s, float *m) { return gq_sbk_q4_5(b, ib, s, m, 0); }
GQ_SB_FN __m256i gq_sbk_q5_k(const uint8_t *b, int ib, float *s, float *m) { return gq_sbk_q4_5(b, ib, s, m, 1); }
GQ_SB_FN __m256i gq_sbk_q6_k(const uint8_t *b, int ib, float *s, float *m) {
    const uint8_t *ql = b, *qhp = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
    const float d = gq_f16(gq_rd16(b + 208));
    s[0] = d * (float)sc[2 * ib]; s[1] = d * (float)sc[2 * ib + 1]; m[0] = m[1] = 0.f;
    const int c = ib >> 2, k = ib & 3;   /* k: 0 ql lo + bits 0-1, 1 ql+32 lo + 2-3, 2 ql hi + 4-5, 3 ql+32 hi + 6-7 */
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3);
    const __m256i l = _mm256_loadu_si256((const __m256i *)(ql + 64 * c + 32 * (k & 1)));
    const __m256i lo = _mm256_and_si256(k & 2 ? _mm256_srli_epi16(l, 4) : l, m4);
    const __m256i q = _mm256_loadu_si256((const __m256i *)(qhp + 32 * c));
    const __m256i h = _mm256_and_si256(k == 0 ? q : k == 1 ? _mm256_srli_epi16(q, 2) : k == 2 ? _mm256_srli_epi16(q, 4) : _mm256_srli_epi16(q, 6), m3);
    return _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(h, 4)), _mm256_set1_epi8(32));
}
GQ_SB_FN __m256i gq_sbk_q3_k(const uint8_t *b, int ib, float *s, float *m) {
    const uint8_t *hm = b, *qs = b + 32, *sc = b + 96; const float d = gq_f16(gq_rd16(b + 108));
    for (int h = 0; h < 2; h++) {
        int k = 2 * ib + h, l = k < 8 ? (sc[k] & 0x0F) : (sc[k - 8] >> 4), hh = (sc[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
        s[h] = d * (float)((l | (hh << 4)) - 32);
    }
    m[0] = m[1] = 0.f;
    const int c = ib >> 2, sh = ib & 3;   /* element 32*ib + j: bits 2*sh of qs[32c + j], bit ib of hm[j] (clear: -4) */
    const __m256i q = _mm256_loadu_si256((const __m256i *)(qs + 32 * c)), hv = _mm256_loadu_si256((const __m256i *)hm);
    const __m256i bit = _mm256_set1_epi8((char)(1 << ib));
    const __m256i sub = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hv, bit), bit), _mm256_set1_epi8(4));
    return _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(q, 2 * sh), _mm256_set1_epi8(3)), sub);
}
#define GQ_UNPACK_SBK(SB) { \
        for (int ib = 0; ib < 8; ib++) { float s2[2], m2[2]; _mm256_storeu_si256((__m256i *)(w + 32 * ib), SB(b, ib, s2, m2)); \
            sw[2 * ib] = s2[0]; sw[2 * ib + 1] = s2[1]; mw[2 * ib] = m2[0]; mw[2 * ib + 1] = m2[1]; } \
        return; }
/* gq_ublk's case for a grid type: every sub-block through its decoder */
#define GQ_UNPACK_SB(BD, SB) { const float d = BD(b); \
        for (int ib = 0; ib < 8; ib++) { float s2[2]; _mm256_storeu_si256((__m256i *)(w + 32 * ib), SB(b, d, ib, s2)); \
            sw[2 * ib] = s2[0]; sw[2 * ib + 1] = s2[1]; mw[2 * ib] = mw[2 * ib + 1] = 0.f; } \
        return; }
#endif
GQ_EXACT_FN static void gq_ublk(int t, const uint8_t *b, int8_t *w, float *sw, float *mw) {
    GQ_EXACT_BEGIN
#if defined(__AVX2__)
    /* vector unpack for the common types: same integers and scales as the
     * scalar cases below (tests/test_ggml_quants.py checks both builds) */
    switch (t) {
    case GQ_Q4_K: GQ_UNPACK_SBK(gq_sbk_q4_k)
    case GQ_Q5_K: GQ_UNPACK_SBK(gq_sbk_q5_k)
    case GQ_Q6_K: GQ_UNPACK_SBK(gq_sbk_q6_k)
    case GQ_Q4_0: {
        const float d = gq_f16(gq_rd16(b));
        const __m128i m = _mm_set1_epi8(0x0F), v = _mm_loadu_si128((const __m128i *)(b + 2)), e8 = _mm_set1_epi8(8);
        _mm_storeu_si128((__m128i *)w, _mm_sub_epi8(_mm_and_si128(v, m), e8));
        _mm_storeu_si128((__m128i *)(w + 16), _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(v, 4), m), e8));
        gq_set_g(sw, mw, 0, 2, d, 0.f); return; }
    case GQ_IQ4_NL: case GQ_MXFP4: {
        const int nl = t == GQ_IQ4_NL;
        const float d = nl ? gq_f16(gq_rd16(b)) : gq_e8m0_half(b[0]);
        const __m128i tab = _mm_loadu_si128((const __m128i *)(nl ? GGML_KVALUES_IQ4NL : GGML_KVALUES_MXFP4));
        _mm256_storeu_si256((__m256i *)w, gq_nib16_lut(b + (nl ? 2 : 1), tab));
        gq_set_g(sw, mw, 0, 2, d, 0.f); return; }
    case GQ_IQ4_XS: GQ_UNPACK_SB(gq_bd_iq4_xs, gq_sb_iq4_xs)
    case GQ_IQ2_XXS: GQ_UNPACK_SB(gq_bd_f16, gq_sb_iq2_xxs)
    case GQ_IQ2_XS: GQ_UNPACK_SB(gq_bd_f16, gq_sb_iq2_xs)
    case GQ_IQ2_S: GQ_UNPACK_SB(gq_bd_f16, gq_sb_iq2_s)
    case GQ_IQ3_XXS: GQ_UNPACK_SB(gq_bd_f16, gq_sb_iq3_xxs)
    case GQ_IQ3_S: GQ_UNPACK_SB(gq_bd_f16, gq_sb_iq3_s)
    case GQ_IQ1_S: GQ_UNPACK_SB(gq_bd_f16, gq_sb_iq1_s)
    case GQ_IQ1_M: GQ_UNPACK_SB(gq_bd_iq1_m, gq_sb_iq1_m)
    case GQ_Q2_K: {
        const uint8_t *sc = b, *qs = b + 16; const float d = gq_f16(gq_rd16(b + 80)), dmin = gq_f16(gq_rd16(b + 82));
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)(sc[g] & 0xF); mw[g] = dmin * (float)(sc[g] >> 4); }
        const __m256i m3 = _mm256_set1_epi8(3);
        for (int c = 0; c < 2; c++) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(qs + 32 * c));
            for (int sh = 0; sh < 4; sh++)
                _mm256_storeu_si256((__m256i *)(w + 128 * c + 32 * sh), _mm256_and_si256(_mm256_srli_epi16(q, 2 * sh), m3));
        }
        return; }
    case GQ_Q3_K: GQ_UNPACK_SBK(gq_sbk_q3_k)
    case GQ_TQ2_0: {
        const float d = gq_f16(gq_rd16(b + 64)); const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
        gq_set_g(sw, mw, 0, 16, d, 0.f);
        for (int c = 0; c < 2; c++) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(b + 32 * c));
            for (int sh = 0; sh < 4; sh++)
                _mm256_storeu_si256((__m256i *)(w + 128 * c + 32 * sh), _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(q, 2 * sh), m3), one));
        }
        return; }
    default: break;
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    /* the same vector unpack on NEON (AArch64): same integers and scales */
    switch (t) {
    case GQ_Q4_K: case GQ_Q5_K: {
        const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
        const uint8_t *sc = b + 4; const uint8_t *qs = t == GQ_Q4_K ? b + 16 : b + 48;
        for (int g = 0; g < 8; g++) { int s, m; gq_scale_min_k4(sc, g, &s, &m); gq_set_g(sw, mw, 2 * g, 2, d * (float)s, dmin * (float)m); }
        const uint8x16_t m4 = vdupq_n_u8(0x0F), sixteen = vdupq_n_u8(16);
        uint8x16_t qh0 = vdupq_n_u8(0), qh1 = vdupq_n_u8(0);
        if (t == GQ_Q5_K) { qh0 = vld1q_u8(b + 16); qh1 = vld1q_u8(b + 32); }
        for (int c = 0; c < 4; c++) {
            for (int h = 0; h < 2; h++) {
                uint8x16_t v = vld1q_u8(qs + 32 * c + 16 * h);
                uint8x16_t lo = vandq_u8(v, m4), hi = vshrq_n_u8(v, 4);
                if (t == GQ_Q5_K) {
                    uint8x16_t qh = h ? qh1 : qh0;
                    uint8x16_t b0 = vdupq_n_u8((uint8_t)(1 << (2 * c))), b1 = vdupq_n_u8((uint8_t)(1 << (2 * c + 1)));
                    lo = vorrq_u8(lo, vandq_u8(vtstq_u8(qh, b0), sixteen));
                    hi = vorrq_u8(hi, vandq_u8(vtstq_u8(qh, b1), sixteen));
                }
                vst1q_s8(w + 64 * c + 16 * h, vreinterpretq_s8_u8(lo));
                vst1q_s8(w + 64 * c + 32 + 16 * h, vreinterpretq_s8_u8(hi));
            }
        }
        return; }
    case GQ_Q6_K: {
        const uint8_t *ql = b, *qhp = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
        const float d = gq_f16(gq_rd16(b + 208));
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)sc[g]; mw[g] = 0.f; }
        const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3); const int8x16_t m32 = vdupq_n_s8(32);
        for (int c = 0; c < 2; c++) for (int h = 0; h < 2; h++) {
            uint8x16_t a = vld1q_u8(ql + 64 * c + 16 * h), bq = vld1q_u8(ql + 64 * c + 32 + 16 * h), q = vld1q_u8(qhp + 32 * c + 16 * h);
            #define GQ6N(L, H) vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(L, vshlq_n_u8(H, 4))), m32)
            vst1q_s8(w + 128 * c + 16 * h,      GQ6N(vandq_u8(a, m4),  vandq_u8(q, m3)));
            vst1q_s8(w + 128 * c + 32 + 16 * h, GQ6N(vandq_u8(bq, m4), vandq_u8(vshrq_n_u8(q, 2), m3)));
            vst1q_s8(w + 128 * c + 64 + 16 * h, GQ6N(vshrq_n_u8(a, 4), vandq_u8(vshrq_n_u8(q, 4), m3)));
            vst1q_s8(w + 128 * c + 96 + 16 * h, GQ6N(vshrq_n_u8(bq, 4), vshrq_n_u8(q, 6)));
            #undef GQ6N
        }
        return; }
    case GQ_Q4_0: {
        const float d = gq_f16(gq_rd16(b)); uint8x16_t v = vld1q_u8(b + 2); const int8x16_t e8 = vdupq_n_s8(8);
        vst1q_s8(w, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(15))), e8));
        vst1q_s8(w + 16, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), e8));
        gq_set_g(sw, mw, 0, 2, d, 0.f); return; }
    case GQ_IQ4_NL: case GQ_MXFP4: {
        const int nl = t == GQ_IQ4_NL;
        const float d = nl ? gq_f16(gq_rd16(b)) : gq_e8m0_half(b[0]);
        const int8x16_t tab = vld1q_s8(nl ? GGML_KVALUES_IQ4NL : GGML_KVALUES_MXFP4);
        uint8x16_t v = vld1q_u8(b + (nl ? 2 : 1));
        vst1q_s8(w, vqtbl1q_s8(tab, vandq_u8(v, vdupq_n_u8(15)))); vst1q_s8(w + 16, vqtbl1q_s8(tab, vshrq_n_u8(v, 4)));
        gq_set_g(sw, mw, 0, 2, d, 0.f); return; }
    case GQ_IQ4_XS: {
        const float d = gq_f16(gq_rd16(b)); const uint16_t sh = gq_rd16(b + 2); const uint8_t *sl = b + 4, *qs = b + 8;
        const int8x16_t tab = vld1q_s8(GGML_KVALUES_IQ4NL);
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (float)(ls - 32), 0.f);
            uint8x16_t v = vld1q_u8(qs + 16 * ib);
            vst1q_s8(w + 32 * ib, vqtbl1q_s8(tab, vandq_u8(v, vdupq_n_u8(15)))); vst1q_s8(w + 32 * ib + 16, vqtbl1q_s8(tab, vshrq_n_u8(v, 4)));
        }
        return; }
    case GQ_IQ2_XXS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a1 >> 28)) * 0.25f, 0.f);
            for (int l = 0; l < 4; l++) {
                int8x8_t g = vreinterpret_s8_u8(vld1_u8(GGML_IQ2XXS_GRID + 8 * ((a0 >> (8 * l)) & 0xFF)));
                vst1_s8(w + 32 * ib + 8 * l, vmul_s8(g, vld1_s8(GQ_SIGN8(GGML_KSIGNS_IQ2XS[(a1 >> (7 * l)) & 127]))));
            }
        }
        return; }
    case GQ_IQ2_XS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sc = b + 66;
        for (int k = 0; k < 32; k++) {
            uint16_t q = gq_rd16(qs + 2 * k); int si = k >> 1;
            if (!(k & 1)) { int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            int8x8_t g = vreinterpret_s8_u8(vld1_u8(GGML_IQ2XS_GRID + 8 * (q & 511)));
            vst1_s8(w + 8 * k, vmul_s8(g, vld1_s8(GQ_SIGN8(GGML_KSIGNS_IQ2XS[q >> 9]))));
        }
        return; }
    case GQ_IQ2_S: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
        for (int k = 0; k < 32; k++) {
            int si = k >> 1;
            if (!(k & 1)) { int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            int idx = qs[k] | (((qh[k >> 2] >> (2 * (k & 3))) & 3) << 8);
            vst1_s8(w + 8 * k, vmul_s8(vreinterpret_s8_u8(vld1_u8(GGML_IQ2S_GRID + 8 * idx)), vld1_s8(GQ_SIGN8(sg[k]))));
        }
        return; }
    case GQ_IQ3_XXS: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *ss = b + 66;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a = gq_rd32(ss + 4 * ib);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a >> 28)) * 0.5f, 0.f);
            for (int l = 0; l < 4; l++) {
                uint8_t gb[8]; memcpy(gb, GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l], 4); memcpy(gb + 4, GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l + 1], 4);
                vst1_s8(w + 32 * ib + 8 * l, vmul_s8(vreinterpret_s8_u8(vld1_u8(gb)), vld1_s8(GQ_SIGN8(GGML_KSIGNS_IQ2XS[(a >> (7 * l)) & 127]))));
            }
        }
        return; }
    case GQ_IQ3_S: {
        gq_sign8_init(); const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
        for (int ib = 0; ib < 8; ib++) { int s = (sc[ib >> 1] >> (4 * (ib & 1))) & 0x0F; gq_set_g(sw, mw, 2 * ib, 2, d * (float)(1 + 2 * s), 0.f); }
        for (int g8 = 0; g8 < 32; g8++) {
            uint8_t gb[8];
            for (int h = 0; h < 2; h++) { int k = 2 * g8 + h; int idx = qs[k] | (((qh[k >> 3] >> (k & 7)) & 1) << 8); memcpy(gb + 4 * h, GGML_IQ3S_GRID + 4 * idx, 4); }
            vst1_s8(w + 8 * g8, vmul_s8(vreinterpret_s8_u8(vld1_u8(gb)), vld1_s8(GQ_SIGN8(sg[g8]))));
        }
        return; }
    case GQ_Q2_K: {
        const uint8_t *sc = b, *qs = b + 16; const float d = gq_f16(gq_rd16(b + 80)), dmin = gq_f16(gq_rd16(b + 82));
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)(sc[g] & 0xF); mw[g] = dmin * (float)(sc[g] >> 4); }
        const uint8x16_t m3 = vdupq_n_u8(3);
        for (int c = 0; c < 2; c++) for (int h = 0; h < 2; h++) {
            uint8x16_t q = vld1q_u8(qs + 32 * c + 16 * h);
            vst1q_s8(w + 128 * c + 16 * h,      vreinterpretq_s8_u8(vandq_u8(q, m3)));
            vst1q_s8(w + 128 * c + 32 + 16 * h, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q, 2), m3)));
            vst1q_s8(w + 128 * c + 64 + 16 * h, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q, 4), m3)));
            vst1q_s8(w + 128 * c + 96 + 16 * h, vreinterpretq_s8_u8(vshrq_n_u8(q, 6)));
        }
        return; }
    default: break;
    }
#endif
    switch (t) {
    case GQ_Q4_0: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int j = 0; j < 16; j++) { w[j] = (int8_t)((qs[j] & 15) - 8); w[j + 16] = (int8_t)((qs[j] >> 4) - 8); }
        gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_Q4_1: { const float d = gq_f16(gq_rd16(b)), m = gq_f16(gq_rd16(b + 2)); const uint8_t *qs = b + 4;
        for (int j = 0; j < 16; j++) { w[j] = (int8_t)(qs[j] & 15); w[j + 16] = (int8_t)(qs[j] >> 4); }
        gq_set_g(sw, mw, 0, 2, d, -m); break; }
    case GQ_Q5_0: { const float d = gq_f16(gq_rd16(b)); const uint32_t qh = gq_rd32(b + 2); const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            w[j] = (int8_t)(((qs[j] & 15) | (int)(((qh >> j) & 1) << 4)) - 16);
            w[j + 16] = (int8_t)(((qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4)) - 16);
        }
        gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_Q5_1: { const float d = gq_f16(gq_rd16(b)), m = gq_f16(gq_rd16(b + 2));
        const uint32_t qh = gq_rd32(b + 4); const uint8_t *qs = b + 8;
        for (int j = 0; j < 16; j++) {
            w[j] = (int8_t)((qs[j] & 15) | (int)(((qh >> j) & 1) << 4));
            w[j + 16] = (int8_t)((qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4));
        }
        gq_set_g(sw, mw, 0, 2, d, -m); break; }
    case GQ_Q8_0: { const float d = gq_f16(gq_rd16(b)); memcpy(w, b + 2, 32); gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_Q2_K: { const uint8_t *sc = b, *qs = b + 16;
        const float d = gq_f16(gq_rd16(b + 80)), dmin = gq_f16(gq_rd16(b + 82));
        for (int i = 0; i < 256; i++) { int c = i >> 7, sh = (i >> 5) & 3, j = i & 31; w[i] = (int8_t)((qs[c * 32 + j] >> (2 * sh)) & 3); }
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)(sc[g] & 0xF); mw[g] = dmin * (float)(sc[g] >> 4); }
        break; }
    case GQ_Q3_K: { const uint8_t *hm = b, *qs = b + 32, *s = b + 96; const float d = gq_f16(gq_rd16(b + 108));
        for (int k = 0; k < 16; k++) {
            int l = k < 8 ? (s[k] & 0x0F) : (s[k - 8] >> 4);
            int h = (s[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
            sw[k] = d * (float)((l | (h << 4)) - 32); mw[k] = 0.f;
        }
        for (int i = 0; i < 256; i++) {
            int c = i >> 7, sh = (i >> 5) & 3, j = i & 31;
            int ql = (qs[c * 32 + j] >> (2 * sh)) & 3, hb = (hm[i & 31] >> (i >> 5)) & 1;
            w[i] = (int8_t)(ql - ((hb ^ 1) << 2));
        }
        break; }
    case GQ_Q4_K: case GQ_Q5_K: { const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
        const uint8_t *sc = b + 4, *qh = b + 16, *qs = t == GQ_Q4_K ? b + 16 : b + 48;
        for (int g = 0; g < 8; g++) {
            int s, m; gq_scale_min_k4(sc, g, &s, &m);
            gq_set_g(sw, mw, 2 * g, 2, d * (float)s, dmin * (float)m);
            const uint8_t *q = qs + (g >> 1) * 32; int sh = (g & 1) * 4;
            for (int j = 0; j < 32; j++) {
                int v = (q[j] >> sh) & 0x0F;
                if (t == GQ_Q5_K) v |= ((qh[j] >> g) & 1) << 4;
                w[g * 32 + j] = (int8_t)v;
            }
        }
        break; }
    case GQ_Q6_K: { const uint8_t *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
        const float d = gq_f16(gq_rd16(b + 208));
        for (int i = 0; i < 256; i++) {
            int c = i >> 7, r = i & 127, nib = r >> 6, k = r & 63, sh = r >> 5, j = r & 31;
            int lo = (ql[c * 64 + k] >> (4 * nib)) & 0x0F, hi = (qh[c * 32 + j] >> (2 * sh)) & 3;
            w[i] = (int8_t)((lo | (hi << 4)) - 32);
        }
        for (int g = 0; g < 16; g++) { sw[g] = d * (float)sc[g]; mw[g] = 0.f; }
        break; }
    case GQ_IQ2_XXS: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a1 >> 28)) * 0.25f, 0.f);
            for (int l = 0; l < 4; l++) {
                const uint8_t *grid = GGML_IQ2XXS_GRID + 8 * ((a0 >> (8 * l)) & 0xFF);
                uint8_t signs = GGML_KSIGNS_IQ2XS[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++) w[ib * 32 + l * 8 + j] = (int8_t)((signs >> j) & 1 ? -grid[j] : grid[j]);
            }
        }
        break; }
    case GQ_IQ2_XS: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sc = b + 66;
        for (int k = 0; k < 32; k++) {
            uint16_t q = gq_rd16(qs + 2 * k); int si = k >> 1;
            if (!(k & 1)) { int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            const uint8_t *grid = GGML_IQ2XS_GRID + 8 * (q & 511); uint8_t signs = GGML_KSIGNS_IQ2XS[q >> 9];
            for (int j = 0; j < 8; j++) w[k * 8 + j] = (int8_t)((signs >> j) & 1 ? -grid[j] : grid[j]);
        }
        break; }
    case GQ_IQ2_S: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
        for (int k = 0; k < 32; k++) {
            int si = k >> 1;
            if (!(k & 1)) { int s = (sc[si >> 1] >> (4 * (si & 1))) & 0x0F; sw[si] = d * (0.5f + (float)s) * 0.25f; mw[si] = 0.f; }
            int idx = qs[k] | (((qh[k >> 2] >> (2 * (k & 3))) & 3) << 8);
            const uint8_t *grid = GGML_IQ2S_GRID + 8 * idx;
            for (int j = 0; j < 8; j++) w[k * 8 + j] = (int8_t)((sg[k] >> j) & 1 ? -grid[j] : grid[j]);
        }
        break; }
    case GQ_IQ3_XXS: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *ss = b + 66;
        for (int ib = 0; ib < 8; ib++) {
            uint32_t a = gq_rd32(ss + 4 * ib);
            gq_set_g(sw, mw, 2 * ib, 2, d * (0.5f + (float)(a >> 28)) * 0.5f, 0.f);
            for (int l = 0; l < 4; l++) {
                uint8_t signs = GGML_KSIGNS_IQ2XS[(a >> (7 * l)) & 127];
                const uint8_t *g1 = GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l], *g2 = GGML_IQ3XXS_GRID + 4 * qs[8 * ib + 2 * l + 1];
                for (int j = 0; j < 4; j++) {
                    w[ib * 32 + l * 8 + j]     = (int8_t)((signs >> j) & 1 ? -g1[j] : g1[j]);
                    w[ib * 32 + l * 8 + j + 4] = (int8_t)((signs >> (j + 4)) & 1 ? -g2[j] : g2[j]);
                }
            }
        }
        break; }
    case GQ_IQ3_S: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
        for (int ib = 0; ib < 8; ib++) { int s = (sc[ib >> 1] >> (4 * (ib & 1))) & 0x0F; gq_set_g(sw, mw, 2 * ib, 2, d * (float)(1 + 2 * s), 0.f); }
        for (int k = 0; k < 64; k++) {
            int idx = qs[k] | (((qh[k >> 3] >> (k & 7)) & 1) << 8);
            const uint8_t *grid = GGML_IQ3S_GRID + 4 * idx;
            for (int j = 0; j < 4; j++) { int e = k * 4 + j; w[e] = (int8_t)((sg[e >> 3] >> (e & 7)) & 1 ? -grid[j] : grid[j]); }
        }
        break; }
    case GQ_IQ1_S: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2, *qhp = b + 34;
        for (int ib = 0; ib < 8; ib++) {
            uint16_t qh = gq_rd16(qhp + 2 * ib);
            const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
            int dsign = (qh & 0x8000) ? -1 : 1;
            gq_set_g(sw, mw, 2 * ib, 2, dl * 0.125f, 0.f);
            for (int l = 0; l < 4; l++) {
                const int8_t *grid = GGML_IQ1S_GRID + 8 * (qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8));
                for (int j = 0; j < 8; j++) w[ib * 32 + l * 8 + j] = (int8_t)(8 * grid[j] + dsign);
            }
        }
        break; }
    case GQ_IQ1_M: { const uint8_t *qs = b, *qh = b + 32, *scp = b + 48;
        uint16_t sc[4]; for (int i = 0; i < 4; i++) sc[i] = gq_rd16(scp + 2 * i);
        const float d = gq_f16((uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)));
        for (int k = 0; k < 32; k++) {
            int s16 = k >> 1;
            if (!(k & 1)) { int s = (sc[s16 >> 2] >> (3 * (s16 & 3))) & 7; sw[s16] = d * (float)(2 * s + 1) * 0.125f; mw[s16] = 0.f; }
            int h = (qh[k >> 1] >> (4 * (k & 1))) & 0x0F;
            const int8_t *grid = GGML_IQ1S_GRID + 8 * (qs[k] | ((h & 7) << 8));
            int dsign = (h & 8) ? -1 : 1;
            for (int j = 0; j < 8; j++) w[k * 8 + j] = (int8_t)(8 * grid[j] + dsign);
        }
        break; }
    case GQ_IQ4_NL: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int j = 0; j < 16; j++) { w[j] = GGML_KVALUES_IQ4NL[qs[j] & 15]; w[j + 16] = GGML_KVALUES_IQ4NL[qs[j] >> 4]; }
        gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_IQ4_XS: { const float d = gq_f16(gq_rd16(b)); const uint16_t sh = gq_rd16(b + 2); const uint8_t *sl = b + 4, *qs = b + 8;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0x0F) | (((sh >> (2 * ib)) & 3) << 4);
            gq_set_g(sw, mw, 2 * ib, 2, d * (float)(ls - 32), 0.f);
            const uint8_t *q = qs + 16 * ib;
            for (int j = 0; j < 16; j++) { w[ib * 32 + j] = GGML_KVALUES_IQ4NL[q[j] & 15]; w[ib * 32 + j + 16] = GGML_KVALUES_IQ4NL[q[j] >> 4]; }
        }
        break; }
    case GQ_MXFP4: { const float d = gq_e8m0_half(b[0]); const uint8_t *qs = b + 1;
        for (int j = 0; j < 16; j++) { w[j] = GGML_KVALUES_MXFP4[qs[j] & 15]; w[j + 16] = GGML_KVALUES_MXFP4[qs[j] >> 4]; }
        gq_set_g(sw, mw, 0, 2, d, 0.f); break; }
    case GQ_TQ1_0: { static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
        const uint8_t *qs = b, *qh = b + 48; const float d = gq_f16(gq_rd16(b + 52)); int o = 0;
        for (int n = 0; n < 5; n++) for (int j = 0; j < 32; j++) { uint8_t v = (uint8_t)(qs[j] * pow3[n]); w[o++] = (int8_t)((int)(((uint16_t)v * 3) >> 8) - 1); }
        for (int n = 0; n < 5; n++) for (int j = 0; j < 16; j++) { uint8_t v = (uint8_t)(qs[32 + j] * pow3[n]); w[o++] = (int8_t)((int)(((uint16_t)v * 3) >> 8) - 1); }
        for (int n = 0; n < 4; n++) for (int j = 0; j < 4; j++) { uint8_t v = (uint8_t)(qh[j] * pow3[n]); w[o++] = (int8_t)((int)(((uint16_t)v * 3) >> 8) - 1); }
        gq_set_g(sw, mw, 0, 16, d, 0.f); break; }
    case GQ_TQ2_0: { const uint8_t *qs = b; const float d = gq_f16(gq_rd16(b + 64));
        for (int i = 0; i < 256; i++) { int c = i >> 7, s = (i >> 5) & 3, j = i & 31; w[i] = (int8_t)((int)((qs[c * 32 + j] >> (2 * s)) & 3) - 1); }
        gq_set_g(sw, mw, 0, 16, d, 0.f); break; }
    case GQ_NVFP4: { const uint8_t *qs = b + 4;
        for (int sb = 0; sb < 4; sb++) {
            sw[sb] = gq_ue4m3_half(b[sb]); mw[sb] = 0.f;
            for (int j = 0; j < 8; j++) { w[16 * sb + j] = GGML_KVALUES_MXFP4[qs[8 * sb + j] & 0x0F]; w[16 * sb + 8 + j] = GGML_KVALUES_MXFP4[qs[8 * sb + j] >> 4]; }
        }
        break; }
    case GQ_Q1_0: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int j = 0; j < 128; j++) w[j] = (qs[j >> 3] >> (j & 7)) & 1 ? 1 : -1;
        gq_set_g(sw, mw, 0, 8, d, 0.f); break; }
    case GQ_Q2_0: { const float d = gq_f16(gq_rd16(b)); const uint8_t *qs = b + 2;
        for (int j = 0; j < 64; j++) w[j] = (int8_t)((int)((qs[j >> 2] >> ((j & 3) * 2)) & 3) - 1);
        gq_set_g(sw, mw, 0, 4, d, 0.f); break; }
    }
}
/* 1 if type t has an integer form (every block type does; F32/F16/BF16 do not). */
static inline int gq_has_int(int t) { return gq_blk_decoder(t) != NULL; }
/* Unpack n elements (whole blocks) of row src. 0, or -1 for a float/unknown type. */
static int gq_unpack_row(int t, const uint8_t *src, int64_t n, int8_t *w8, float *sw, float *mw) {
    if (!gq_has_int(t) || n % GQ_TYPES[t].blck) return -1;
    const int bl = GQ_TYPES[t].blck, by = GQ_TYPES[t].bytes;
    for (int64_t i = 0; i < n / bl; i++) gq_ublk(t, src + i * by, w8 + i * bl, sw + i * (bl / GQ_G), mw + i * (bl / GQ_G));
    return 0;
}


/* ---- activations in int8, and the integer dot product ----------------
 * An activation row is quantized once per call, in blocks of 32: int8 codes,
 * one float scale per 32 and the code sum per 16 (the min term of the K
 * types). A weight row in integer form (gq_unpack_row) is then dotted as
 *     sum_g dx[g/2] * (sw[g] * sum_{i in g} w8[i]*xq[i] - mw[g] * xs[g])
 * with exact integer inner sums. The only approximation against the float
 * reference is the activation's own rounding to int8, the same one colibri's
 * other integer paths make. GQ_EXACT=1 selects the float reference path. */
static void gq_quant_act_ref(const float *x, int64_t n, int8_t *xq, float *dx, int32_t *xs) {
    for (int64_t b = 0; b < n / 32; b++) {
        const float *v = x + b * 32; float amax = 0.f;
        for (int i = 0; i < 32; i++) { float a = fabsf(v[i]); if (a > amax) amax = a; }
        const float d = amax / 127.f, id = amax > 0.f ? 127.f / amax : 0.f;
        dx[b] = d;
        int32_t s0 = 0, s1 = 0;
        for (int i = 0; i < 32; i++) {
            int q = (int)lrintf(v[i] * id); if (q > 127) q = 127; if (q < -127) q = -127;
            xq[b * 32 + i] = (int8_t)q; if (i < 16) s0 += q; else s1 += q;
        }
        xs[2 * b] = s0; xs[2 * b + 1] = s1;
    }
}
#if defined(__AVX2__)
/* a block of 32 quantized as gq_quant_act_ref does it, eight lanes at a time: the
 * maximum of |v| (exact in any order), the same scalar d and 1/d, v*id rounded by
 * cvtps (the current rounding mode, lrintf's), the same clamp. Byte for byte its codes,
 * scales and sums; the scalar loop ran once per matrix and per row on one thread. */
static inline void gq_quant_block_avx2(const float *v, int8_t *q, float *d, int32_t *s16) {
    const __m256 sign = _mm256_set1_ps(-0.f);
    const __m256 v0 = _mm256_loadu_ps(v), v1 = _mm256_loadu_ps(v + 8), v2 = _mm256_loadu_ps(v + 16), v3 = _mm256_loadu_ps(v + 24);
    const __m256 a = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                   _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m)); m = _mm_max_ss(m, _mm_movehdup_ps(m));
    const float amax = _mm_cvtss_f32(m);
    const float id = amax > 0.f ? 127.f / amax : 0.f;
    *d = amax / 127.f;
    const __m256 idv = _mm256_set1_ps(id);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    const __m256i q0 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v0, idv)), lo), hi);
    const __m256i q1 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v1, idv)), lo), hi);
    const __m256i q2 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v2, idv)), lo), hi);
    const __m256i q3 = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v3, idv)), lo), hi);
    const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(q0, q1), _mm256_packs_epi32(q2, q3));
    _mm256_storeu_si256((__m256i *)q, _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
    if (s16) {
        __m256i t0 = _mm256_add_epi32(q0, q1), t1 = _mm256_add_epi32(q2, q3);
        __m128i u0 = _mm_add_epi32(_mm256_castsi256_si128(t0), _mm256_extracti128_si256(t0, 1));
        __m128i u1 = _mm_add_epi32(_mm256_castsi256_si128(t1), _mm256_extracti128_si256(t1, 1));
        u0 = _mm_hadd_epi32(u0, u1); u0 = _mm_hadd_epi32(u0, u0);
        s16[0] = _mm_cvtsi128_si32(u0); s16[1] = _mm_extract_epi32(u0, 1);
    }
}
#endif
static void gq_quant_act(const float *x, int64_t n, int8_t *xq, float *dx, int32_t *xs) {
#if defined(__AVX2__)
    for (int64_t b = 0; b < n / 32; b++) gq_quant_block_avx2(x + b * 32, xq + b * 32, dx + b, xs + 2 * b);
#else
    gq_quant_act_ref(x, n, xq, dx, xs);
#endif
}
static inline int gq_has_min(int t) { return t == GQ_Q4_1 || t == GQ_Q5_1 || t == GQ_Q2_K || t == GQ_Q4_K || t == GQ_Q5_K; }

static float gq_dot_i8(const int8_t *w8, const float *sw, const float *mw, int has_min,
                       const int8_t *xq, const float *dx, const int32_t *xs, int64_t n) {
    float acc = 0.f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 vacc = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / 32; b++) {
        __m256i w = _mm256_loadu_si256((const __m256i *)(w8 + b * 32));
        __m256i x = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
        __m256i p = GQ_DP32(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
        /* int32 lanes 0..3 are group 2b, 4..7 group 2b+1 */
        const float d = dx[b];
        __m256 sc = _mm256_set_m128(_mm_set1_ps(sw[2 * b + 1] * d), _mm_set1_ps(sw[2 * b] * d));
        vacc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), sc, vacc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h)); h = _mm_add_ss(h, _mm_movehdup_ps(h));
    acc = _mm_cvtss_f32(h);
#elif defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t vacc = vdupq_n_f32(0.f);
    for (int64_t b = 0; b < n / 32; b++) {
        int8x16_t w0 = vld1q_s8(w8 + b * 32), w1 = vld1q_s8(w8 + b * 32 + 16);
        int8x16_t x0 = vld1q_s8(xq + b * 32), x1 = vld1q_s8(xq + b * 32 + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), w0, x0), p1 = vdotq_s32(vdupq_n_s32(0), w1, x1);
#else
        int32x4_t p0 = vpaddlq_s16(vmlal_s8(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)), vget_high_s8(w0), vget_high_s8(x0)));
        int32x4_t p1 = vpaddlq_s16(vmlal_s8(vmull_s8(vget_low_s8(w1), vget_low_s8(x1)), vget_high_s8(w1), vget_high_s8(x1)));
#endif
        const float d = dx[b];
        vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p0), sw[2 * b] * d);
        vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p1), sw[2 * b + 1] * d);
    }
    acc = vaddvq_f32(vacc);
#else
    for (int64_t b = 0; b < n / 32; b++) {
        int32_t s0 = 0, s1 = 0;
        for (int i = 0; i < 16; i++) s0 += (int32_t)w8[b * 32 + i] * xq[b * 32 + i];
        for (int i = 16; i < 32; i++) s1 += (int32_t)w8[b * 32 + i] * xq[b * 32 + i];
        acc += dx[b] * (sw[2 * b] * (float)s0 + sw[2 * b + 1] * (float)s1);
    }
#endif
    if (has_min) {   /* explicit fmaf: the same bits in gq_dot_b32, whatever the compiler contracts */
        float m = 0.f;
        for (int64_t b = 0; b < n / 32; b++) m = fmaf(dx[b], fmaf(mw[2 * b], (float)xs[2 * b], mw[2 * b + 1] * (float)xs[2 * b + 1]), m);
        acc -= m;
    }
    return acc;
}

static int gq_exact_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("GQ_EXACT"); v = e && *e && *e != '0'; } return v; }

/* Q8_0 needs no unpack: the block already is 32 int8 codes and one scale. */
static inline float gq_dot_q8_0(const uint8_t *row, const int8_t *xq, const float *dx, int64_t n) {
    float acc = 0.f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 vacc = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34;
        __m256i w = _mm256_loadu_si256((const __m256i *)(blk + 2)), x = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
        __m256i p = GQ_DP32(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
        vacc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(gq_f16(gq_rd16(blk)) * dx[b]), vacc);
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(vacc), _mm256_extractf128_ps(vacc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h)); h = _mm_add_ss(h, _mm_movehdup_ps(h));
    acc = _mm_cvtss_f32(h);
#elif defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t vacc = vdupq_n_f32(0.f);
    for (int64_t b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34;
        int8x16_t w0 = vld1q_s8((const int8_t *)(blk + 2)), w1 = vld1q_s8((const int8_t *)(blk + 18));
        int8x16_t x0 = vld1q_s8(xq + b * 32), x1 = vld1q_s8(xq + b * 32 + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, x0), w1, x1);
#else
        int32x4_t p = vaddq_s32(vpaddlq_s16(vmlal_s8(vmull_s8(vget_low_s8(w0), vget_low_s8(x0)), vget_high_s8(w0), vget_high_s8(x0))),
                                vpaddlq_s16(vmlal_s8(vmull_s8(vget_low_s8(w1), vget_low_s8(x1)), vget_high_s8(w1), vget_high_s8(x1))));
#endif
        vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p), gq_f16(gq_rd16(blk)) * dx[b]);
    }
    acc = vaddvq_f32(vacc);
#else
    for (int64_t b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34; const int8_t *w = (const int8_t *)(blk + 2); int32_t si = 0;
        for (int i = 0; i < 32; i++) si += (int32_t)w[i] * xq[b * 32 + i];
        acc += gq_f16(gq_rd16(blk)) * dx[b] * (float)si;
    }
#endif
    return acc;
}

/* An activation block quantized once and shared by every matrix that reads it
 * (gate and up read the same x). */
/* l4: the sum of every four consecutive codes (I/4 a row), which turns an unsigned
 * weight code u = w + bias into w with one subtraction: sum w x = dpbusd(u, x) -
 * bias * l4, lane by lane (GQ_VNNI512). Last, so a view built field by field without
 * it leaves it NULL, and the kernels that need it fall back to the AVX2 ones. */
typedef struct { int8_t *q; float *d; int32_t *s; int S, I, cap; int32_t *l4; } gq_act;
/* l4 of n codes: the sum of each four in order */
static inline void gq_lane_sums(const int8_t *q, int64_t n, int32_t *l4) {
    int64_t k = 0;
#if defined(__AVX2__)
    const __m256i one8 = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
    for (; k + 8 <= n / 4; k += 8)   /* 1 x q per byte, pairs then quads: exact */
        _mm256_storeu_si256((__m256i *)(l4 + k), _mm256_madd_epi16(_mm256_maddubs_epi16(one8, _mm256_loadu_si256((const __m256i *)(q + 4 * k))), one16));
#endif
    for (; k < n / 4; k++) l4[k] = (int32_t)q[4 * k] + q[4 * k + 1] + q[4 * k + 2] + q[4 * k + 3];
}
static int gq_act_set(gq_act *a, const float *x, int S, int I) {
    if ((int64_t)S * I > a->cap) {
        free(a->q); free(a->d); free(a->s); free(a->l4);
        a->cap = S * I; a->q = (int8_t *)malloc((size_t)a->cap); a->d = (float *)malloc((size_t)(a->cap / 32 + 1) * sizeof(float));
        a->s = (int32_t *)malloc((size_t)(a->cap / 16 + 1) * sizeof(int32_t));
        a->l4 = (int32_t *)malloc((size_t)(a->cap / 4 + 1) * sizeof(int32_t));
        if (!a->q || !a->d || !a->s || !a->l4) { a->cap = 0; return -1; }
    }
    a->S = S; a->I = I;
    for (int s = 0; s < S; s++) gq_quant_act(x + (int64_t)s * I, I, a->q + (int64_t)s * I, a->d + (int64_t)s * (I / 32), a->s + (int64_t)s * (I / 16));
    gq_lane_sums(a->q, (int64_t)S * I, a->l4);
    return 0;
}
static void gq_act_free(gq_act *a) { free(a->q); free(a->d); free(a->s); free(a->l4); memset(a, 0, sizeof *a); }
/* dst's rows are src's rows[0..n): the same codes, scales and sums, copied (a grouped
 * MoE hands each expert its assignments' rows contiguously). 0, or -1 out of memory. */
static int gq_act_gather(gq_act *dst, const gq_act *src, const int *rows, int n) {
    const int I = src->I;
    if ((int64_t)n * I > dst->cap) {
        free(dst->q); free(dst->d); free(dst->s); free(dst->l4);
        dst->cap = n * I; dst->q = (int8_t *)malloc((size_t)dst->cap); dst->d = (float *)malloc((size_t)(dst->cap / 32 + 1) * sizeof(float));
        dst->s = (int32_t *)malloc((size_t)(dst->cap / 16 + 1) * sizeof(int32_t));
        dst->l4 = (int32_t *)malloc((size_t)(dst->cap / 4 + 1) * sizeof(int32_t));
        if (!dst->q || !dst->d || !dst->s || !dst->l4) { dst->cap = 0; return -1; }
    }
    dst->S = n; dst->I = I;
    for (int p = 0; p < n; p++) {
        const int64_t r = rows[p];
        memcpy(dst->q + (int64_t)p * I, src->q + r * I, (size_t)I);
        memcpy(dst->d + (int64_t)p * (I / 32), src->d + r * (I / 32), (size_t)(I / 32) * sizeof(float));
        memcpy(dst->s + (int64_t)p * (I / 16), src->s + r * (I / 16), (size_t)(I / 16) * sizeof(int32_t));
        memcpy(dst->l4 + (int64_t)p * (I / 4), src->l4 + r * (I / 4), (size_t)(I / 4) * sizeof(int32_t));
    }
    return 0;
}

/* per-thread unpack buffers, grown on demand and kept */
static __thread int8_t *gq_tw8; static __thread float *gq_tsw, *gq_tmw; static __thread int gq_tcap;
static inline int gq_tbuf(int I) {
    if (I > gq_tcap) {
        free(gq_tw8); free(gq_tsw); free(gq_tmw);
        gq_tw8 = (int8_t *)malloc((size_t)I); gq_tsw = (float *)malloc((size_t)(I / 16 + 1) * sizeof(float)); gq_tmw = (float *)malloc((size_t)(I / 16 + 1) * sizeof(float));
        gq_tcap = (gq_tw8 && gq_tsw && gq_tmw) ? I : 0;
    }
    return gq_tcap >= I;
}
/* one output row of W against every activation row */
#if defined(__AVX2__) && defined(__FMA__)
/* The 32-element block types without the unpack (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL,
 * MXFP4): a block's nibbles (low ones are elements 0..15, high ones 16..31), the
 * fifth bit of Q5_x, made 32 int8 in a register and multiplied as gq_dot_i8 does:
 * the same int32 lanes, the same scale product and fma, the same reduction, and for
 * Q4_1/Q5_1 the same min term, so the result is the unpack path's bit for bit. The
 * unpack cost a call and a type switch per block, 64 per 2048-wide row. */
static inline __m256i gq_bits32(uint32_t bits) {   /* byte i: 0xFF when bit i is set */
    const __m256i pick = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), pick);
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
}
#define GQ_HSUM8(v) ({ __m128 h_ = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)); \
    h_ = _mm_add_ps(h_, _mm_movehl_ps(h_, h_)); h_ = _mm_add_ss(h_, _mm_movehdup_ps(h_)); _mm_cvtss_f32(h_); })
#define GQ_FUSED_ROWS 4   /* activation rows a fused kernel takes at once; from there one unpack shared by all is cheaper */
/* S (1..GQ_FUSED_ROWS) activation rows of one weight row: y[s * ystride]. A block is
 * decoded once and multiplied into every row, each row with its own accumulator and
 * the operations it has alone, so a row's bits do not depend on S. */
static inline __attribute__((always_inline)) void gq_dot_b32_body(int t, float *y, int ystride, const uint8_t *row, const int8_t *xq,
                                                                 const float *dx, const int32_t *xs, int I, const int S) {
    const int mx = t == GQ_MXFP4, nl = t == GQ_IQ4_NL, q5 = t == GQ_Q5_0 || t == GQ_Q5_1, hasmin = t == GQ_Q4_1 || t == GQ_Q5_1;
    const int bs = GQ_TYPES[t].bytes, off = mx ? 1 : t == GQ_Q4_0 || nl ? 2 : t == GQ_Q4_1 || t == GQ_Q5_0 ? (t == GQ_Q4_1 ? 4 : 6) : 8;
    const __m128i tab = _mm_loadu_si128((const __m128i *)(mx ? GGML_KVALUES_MXFP4 : GGML_KVALUES_IQ4NL));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i bias = _mm256_set1_epi8(t == GQ_Q4_0 ? 8 : t == GQ_Q5_0 ? 16 : 0), b16 = _mm256_set1_epi8(16);
    __m256 vacc[GQ_FUSED_ROWS]; for (int s = 0; s < S; s++) vacc[s] = _mm256_setzero_ps();
    for (int64_t b = 0; b < I / 32; b++) {
        const uint8_t *blk = row + b * bs;
        const float d = mx ? gq_e8m0_half(blk[0]) : gq_f16(gq_rd16(blk));
        const __m128i v = _mm_loadu_si128((const __m128i *)(blk + off));
        __m128i lo = _mm_and_si128(v, m4), hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
        if (nl || mx) { lo = _mm_shuffle_epi8(tab, lo); hi = _mm_shuffle_epi8(tab, hi); }
        __m256i w = _mm256_set_m128i(hi, lo);
        if (q5) w = _mm256_or_si256(w, _mm256_and_si256(gq_bits32(gq_rd32(blk + (t == GQ_Q5_0 ? 2 : 4))), b16));
        w = _mm256_sub_epi8(w, bias);
        const __m256i aw = _mm256_sign_epi8(w, w);
        for (int s = 0; s < S; s++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (int64_t)s * I + b * 32));
            const __m256i p = GQ_DP32(aw, _mm256_sign_epi8(x, w));
            vacc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[(int64_t)s * (I / 32) + b]), vacc[s]);
        }
    }
    for (int s = 0; s < S; s++) {
        float acc = GQ_HSUM8(vacc[s]);
        if (hasmin) {   /* gq_dot_i8's min term, mw = -m for both halves of a block */
            const float *dxs = dx + (int64_t)s * (I / 32); const int32_t *xss = xs + (int64_t)s * (I / 16);
            float m = 0.f;
            for (int64_t b = 0; b < I / 32; b++) {
                const float mw = -gq_f16(gq_rd16(row + b * bs + 2));
                m = fmaf(dxs[b], fmaf(mw, (float)xss[2 * b], mw * (float)xss[2 * b + 1]), m);
            }
            acc -= m;
        }
        y[(int64_t)s * ystride] = acc;
    }
}
/* Q2_0 (64 elements in 18 bytes: an f16 scale, then four 2-bit codes a byte, element
 * 4k+r in bits 2r of byte k, value (code - 1) * d): the 16 code bytes become 64 int8 in
 * two registers (each byte copied to the four lanes of its elements, the lane's two bits
 * picked with a shift and a mask) and are multiplied as gq_dot_i8 does it, two 32-element
 * halves with the block's scale each, so the result is the unpack path's bit for bit. */
static inline __attribute__((always_inline)) void gq_dot_q2_0_body(float *y, int ystride, const uint8_t *row, const int8_t *xq,
                                                                  const float *dx, int I, const int S) {
    const __m256i lo_pick = _mm256_set_epi64x(0x0707070706060606LL, 0x0505050504040404LL, 0x0303030302020202LL, 0x0101010100000000LL);
    const __m256i hi_pick = _mm256_set_epi64x(0x0F0F0F0F0E0E0E0ELL, 0x0D0D0D0D0C0C0C0CLL, 0x0B0B0B0B0A0A0A0ALL, 0x0909090908080808LL);
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    const __m256i r1 = _mm256_set1_epi32(0x0000FF00), r2 = _mm256_set1_epi32(0x00FF0000), r3 = _mm256_set1_epi32((int)0xFF000000u);
    __m256 vacc[GQ_FUSED_ROWS]; for (int s = 0; s < S; s++) vacc[s] = _mm256_setzero_ps();
    for (int64_t b = 0; b < I / 64; b++) {
        const uint8_t *blk = row + b * 18; const float d = gq_f16(gq_rd16(blk));
        const __m128i q = _mm_loadu_si128((const __m128i *)(blk + 2));
        const __m256i qq = _mm256_set_m128i(q, q);
        __m256i w2[2];
        for (int h = 0; h < 2; h++) {
            const __m256i v = _mm256_shuffle_epi8(qq, h ? hi_pick : lo_pick);   /* byte i holds the codes of element i */
            __m256i w = _mm256_and_si256(v, m3);                                 /* lanes 4k: bits 0-1 */
            w = _mm256_blendv_epi8(w, _mm256_and_si256(_mm256_srli_epi16(v, 2), m3), r1);
            w = _mm256_blendv_epi8(w, _mm256_and_si256(_mm256_srli_epi16(v, 4), m3), r2);
            w = _mm256_blendv_epi8(w, _mm256_and_si256(_mm256_srli_epi16(v, 6), m3), r3);
            w2[h] = _mm256_sub_epi8(w, one);
        }
        for (int h = 0; h < 2; h++) {
            const __m256i w = w2[h], aw = _mm256_sign_epi8(w, w); const int64_t xb = 2 * b + h;
            for (int s = 0; s < S; s++) {
                const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (int64_t)s * I + xb * 32));
                const __m256i p = GQ_DP32(aw, _mm256_sign_epi8(x, w));
                vacc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[(int64_t)s * (I / 32) + xb]), vacc[s]);
            }
        }
    }
    for (int s = 0; s < S; s++) y[(int64_t)s * ystride] = GQ_HSUM8(vacc[s]);
}
#ifdef GQ_VNNI512
/* Q2_0 on AVX-512: element e of a block is bits 2e..2e+1 of its 16 code bytes, so one
 * vpmultishiftqb lifts the 64 codes into 64 bytes (qword j of the result reads bits 16j
 * on of the stream, which the qword permute puts in its lane) and vpdpbusd multiplies the
 * unsigned codes c = w + 1 into the activations: lane l = sum of elements 4l..4l+3 of
 * c x, minus the lane's activation sum is the lane of w x, gq_dot_q2_0_body's exactly.
 * The float steps are its own: cvt, then half 0 and half 1 each times d * dx into the
 * 8-lane accumulator. */
static inline __attribute__((always_inline)) void gq_dot_q2_0_body512(float *y, int ystride, const uint8_t *row, const int8_t *xq,
                                                                     const float *dx, const int32_t *l4, int I, const int S) {
    const __m512i qsel = _mm512_set_epi64(1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i ctrl = _mm512_set_epi64(0x3e3c3a3836343230LL, 0x2e2c2a2826242220LL, 0x1e1c1a1816141210LL, 0x0e0c0a0806040200LL,
                                          0x3e3c3a3836343230LL, 0x2e2c2a2826242220LL, 0x1e1c1a1816141210LL, 0x0e0c0a0806040200LL);
    const __m512i m3 = _mm512_set1_epi8(3);
    __m256 vacc[GQ_FUSED_ROWS]; for (int s = 0; s < S; s++) vacc[s] = _mm256_setzero_ps();
    for (int64_t b = 0; b < I / 64; b++) {
        const uint8_t *blk = row + b * 18; const float d = gq_f16(gq_rd16(blk));
        const __m512i q = _mm512_permutexvar_epi64(qsel, _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *)(blk + 2))));
        const __m512i c = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, q), m3);
        const __m512 ws = _mm512_set1_ps(d);
        for (int s = 0; s < S; s++) {
            const __m512i x = _mm512_loadu_si512((const void *)(xq + (int64_t)s * I + b * 64));
            const __m512i p = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), c, x),
                                               _mm512_loadu_si512((const void *)(l4 + (int64_t)s * (I / 4) + b * 16)));
            const __m512 pf = _mm512_cvtepi32_ps(p);
            const __m512 sc = _mm512_mul_ps(ws, gq_x16(dx + (int64_t)s * (I / 32) + 2 * b));
            vacc[s] = _mm256_fmadd_ps(_mm512_castps512_ps256(pf), _mm512_castps512_ps256(sc), vacc[s]);
            vacc[s] = _mm256_fmadd_ps(_mm512_extractf32x8_ps(pf, 1), _mm512_extractf32x8_ps(sc, 1), vacc[s]);
        }
    }
    for (int s = 0; s < S; s++) y[(int64_t)s * ystride] = GQ_HSUM8(vacc[s]);
}
/* any number of activation rows, GQ_FUSED_ROWS at a time: a row's bits do not depend on S */
static void gq_dot_q2_0_512(float *y, int ystride, const uint8_t *row, const gq_act *a, int I) {
    for (int s0 = 0; s0 < a->S; s0 += GQ_FUSED_ROWS) {
        const int n = a->S - s0 < GQ_FUSED_ROWS ? a->S - s0 : GQ_FUSED_ROWS;
        float *ys = y + (int64_t)s0 * ystride; const int8_t *xq = a->q + (int64_t)s0 * I;
        const float *dx = a->d + (int64_t)s0 * (I / 32); const int32_t *l4 = a->l4 + (int64_t)s0 * (I / 4);
        switch (n) {
        case 1: gq_dot_q2_0_body512(ys, ystride, row, xq, dx, l4, I, 1); break;
        case 2: gq_dot_q2_0_body512(ys, ystride, row, xq, dx, l4, I, 2); break;
        case 3: gq_dot_q2_0_body512(ys, ystride, row, xq, dx, l4, I, 3); break;
        default: gq_dot_q2_0_body512(ys, ystride, row, xq, dx, l4, I, 4); break;
        }
    }
}
#endif
static void gq_dot_q2_0(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, int I, int S) {
    switch (S) {
    case 1: gq_dot_q2_0_body(y, ystride, row, xq, dx, I, 1); break;
    case 2: gq_dot_q2_0_body(y, ystride, row, xq, dx, I, 2); break;
    case 3: gq_dot_q2_0_body(y, ystride, row, xq, dx, I, 3); break;
    default: gq_dot_q2_0_body(y, ystride, row, xq, dx, I, 4); break;
    }
}
/* S constant in each call below: the accumulators stay in registers and S=1 is the one-row kernel */
static void gq_dot_b32(int t, float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, const int32_t *xs, int I, int S) {
    switch (S) {
    case 1: gq_dot_b32_body(t, y, ystride, row, xq, dx, xs, I, 1); break;
    case 2: gq_dot_b32_body(t, y, ystride, row, xq, dx, xs, I, 2); break;
    case 3: gq_dot_b32_body(t, y, ystride, row, xq, dx, xs, I, 3); break;
    default: gq_dot_b32_body(t, y, ystride, row, xq, dx, xs, I, 4); break;
    }
}
/* Q8_0, S rows at once: the block's codes loaded and made positive once */
static inline __attribute__((always_inline)) void gq_dot_q8_0_body(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, int I, const int S) {
    __m256 vacc[GQ_FUSED_ROWS]; for (int s = 0; s < S; s++) vacc[s] = _mm256_setzero_ps();
    for (int64_t b = 0; b < I / 32; b++) {
        const uint8_t *blk = row + b * 34; const float d = gq_f16(gq_rd16(blk));
        const __m256i w = _mm256_loadu_si256((const __m256i *)(blk + 2)), aw = _mm256_sign_epi8(w, w);
        for (int s = 0; s < S; s++) {
            const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (int64_t)s * I + b * 32));
            const __m256i p = GQ_DP32(aw, _mm256_sign_epi8(x, w));
            vacc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[(int64_t)s * (I / 32) + b]), vacc[s]);
        }
    }
    for (int s = 0; s < S; s++) y[(int64_t)s * ystride] = GQ_HSUM8(vacc[s]);
}
static void gq_dot_q8_0_rows(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, int I, int S) {
    switch (S) {
    case 1: gq_dot_q8_0_body(y, ystride, row, xq, dx, I, 1); break;
    case 2: gq_dot_q8_0_body(y, ystride, row, xq, dx, I, 2); break;
    case 3: gq_dot_q8_0_body(y, ystride, row, xq, dx, I, 3); break;
    default: gq_dot_q8_0_body(y, ystride, row, xq, dx, I, 4); break;
    }
}
/* The 256-element grid types without the unpack: each sub-block from its decoder
 * (gq_sb_*, the unpack's own), multiplied at once as gq_dot_i8 does it (the same
 * lanes, group scales times dx, fma and reduction), so the result is the unpack
 * path's bit for bit, without storing and reloading the row. */
#define GQ_DOT_SB256(NAME, BYTES, BD, SB) \
GQ_SB_FN void NAME##_body(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, int I, const int S) { \
    __m256 vacc[GQ_FUSED_ROWS]; for (int s = 0; s < S; s++) vacc[s] = _mm256_setzero_ps(); \
    for (int64_t blk = 0; blk < I / 256; blk++) { \
        const uint8_t *b = row + blk * (BYTES); const float d = BD(b); \
        for (int ib = 0; ib < 8; ib++) { \
            float s2[2]; const __m256i w = SB(b, d, ib, s2), aw = _mm256_sign_epi8(w, w); const int64_t xb = blk * 8 + ib; \
            for (int s = 0; s < S; s++) { \
                const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (int64_t)s * I + xb * 32)); \
                const __m256i p = GQ_DP32(aw, _mm256_sign_epi8(x, w)); \
                const float dd = dx[(int64_t)s * (I / 32) + xb]; \
                vacc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set_m128(_mm_set1_ps(s2[1] * dd), _mm_set1_ps(s2[0] * dd)), vacc[s]); \
            } \
        } \
    } \
    for (int s = 0; s < S; s++) y[(int64_t)s * ystride] = GQ_HSUM8(vacc[s]); \
} \
GQ_EXACT_FN static void NAME(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, int I, int S) { \
    switch (S) { \
    case 1: NAME##_body(y, ystride, row, xq, dx, I, 1); break; \
    case 2: NAME##_body(y, ystride, row, xq, dx, I, 2); break; \
    case 3: NAME##_body(y, ystride, row, xq, dx, I, 3); break; \
    default: NAME##_body(y, ystride, row, xq, dx, I, 4); break; \
    } \
}
GQ_DOT_SB256(gq_dot_iq4_xs, 136, gq_bd_iq4_xs, gq_sb_iq4_xs)
GQ_DOT_SB256(gq_dot_iq2_xxs, 66, gq_bd_f16, gq_sb_iq2_xxs)
GQ_DOT_SB256(gq_dot_iq2_xs, 74, gq_bd_f16, gq_sb_iq2_xs)
GQ_DOT_SB256(gq_dot_iq2_s, 82, gq_bd_f16, gq_sb_iq2_s)
GQ_DOT_SB256(gq_dot_iq3_xxs, 98, gq_bd_f16, gq_sb_iq3_xxs)
GQ_DOT_SB256(gq_dot_iq3_s, 110, gq_bd_f16, gq_sb_iq3_s)
GQ_DOT_SB256(gq_dot_iq1_s, 50, gq_bd_f16, gq_sb_iq1_s)
GQ_DOT_SB256(gq_dot_iq1_m, 56, gq_bd_iq1_m, gq_sb_iq1_m)
/* The K-quants the same way, with gq_dot_i8's min term accumulated per row in its
 * order (explicit fmaf) beside the vector part. */
#define GQ_DOT_SBK(NAME, BYTES, SB, HASMIN) \
GQ_SB_FN void NAME##_body(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, const int32_t *xs, int I, const int S) { \
    __m256 vacc[GQ_FUSED_ROWS]; float mm[GQ_FUSED_ROWS]; \
    for (int s = 0; s < S; s++) { vacc[s] = _mm256_setzero_ps(); mm[s] = 0.f; } \
    for (int64_t blk = 0; blk < I / 256; blk++) { \
        const uint8_t *b = row + blk * (BYTES); \
        for (int ib = 0; ib < 8; ib++) { \
            float s2[2], m2[2]; const __m256i w = SB(b, ib, s2, m2), aw = _mm256_sign_epi8(w, w); const int64_t xb = blk * 8 + ib; \
            for (int s = 0; s < S; s++) { \
                const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (int64_t)s * I + xb * 32)); \
                const __m256i p = GQ_DP32(aw, _mm256_sign_epi8(x, w)); \
                const float dd = dx[(int64_t)s * (I / 32) + xb]; \
                vacc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set_m128(_mm_set1_ps(s2[1] * dd), _mm_set1_ps(s2[0] * dd)), vacc[s]); \
                if (HASMIN) { const int32_t *xss = xs + (int64_t)s * (I / 16); \
                    mm[s] = fmaf(dd, fmaf(m2[0], (float)xss[2 * xb], m2[1] * (float)xss[2 * xb + 1]), mm[s]); } \
            } \
        } \
    } \
    for (int s = 0; s < S; s++) { float acc = GQ_HSUM8(vacc[s]); if (HASMIN) acc -= mm[s]; y[(int64_t)s * ystride] = acc; } \
} \
GQ_EXACT_FN static void NAME(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, const int32_t *xs, int I, int S) { \
    switch (S) { \
    case 1: NAME##_body(y, ystride, row, xq, dx, xs, I, 1); break; \
    case 2: NAME##_body(y, ystride, row, xq, dx, xs, I, 2); break; \
    case 3: NAME##_body(y, ystride, row, xq, dx, xs, I, 3); break; \
    default: NAME##_body(y, ystride, row, xq, dx, xs, I, 4); break; \
    } \
}
GQ_DOT_SBK(gq_dot_q4_k, 144, gq_sbk_q4_k, 1)
GQ_DOT_SBK(gq_dot_q5_k, 176, gq_sbk_q5_k, 1)
GQ_DOT_SBK(gq_dot_q6_k, 210, gq_sbk_q6_k, 0)
GQ_DOT_SBK(gq_dot_q3_k, 110, gq_sbk_q3_k, 0)
typedef void (*gq_fdot_fn)(float *, int, const uint8_t *, const int8_t *, const float *, int, int);
static inline gq_fdot_fn gq_fdot_sb256(int t) {
    switch (t) {
    case GQ_IQ4_XS: return gq_dot_iq4_xs;   case GQ_IQ2_XXS: return gq_dot_iq2_xxs;
    case GQ_IQ2_XS: return gq_dot_iq2_xs;   case GQ_IQ2_S: return gq_dot_iq2_s;
    case GQ_IQ3_XXS: return gq_dot_iq3_xxs; case GQ_IQ3_S: return gq_dot_iq3_s;
    case GQ_IQ1_S: return gq_dot_iq1_s;     case GQ_IQ1_M: return gq_dot_iq1_m;
    default: return NULL;
    }
}
#endif
#ifdef GQ_VNNI512
/* ---- AVX-512 VNNI: two 32-element sub-blocks a register ----------------
 * Every kernel below takes its sub-blocks from the AVX2 decoder of its type and
 * multiplies 64 of them at once with the sign trick of gq_dot_i8: |w| unsigned
 * times x negated where w < 0 (one masked subtraction), vpdpbusd's int32 lane is
 * elements 4l..4l+3 of w x, the lanes the AVX2 maddubs/madd pair gives. The float
 * steps are the AVX2 kernel's in its order: cvt, then the first sub-block's lanes
 * and the second's, each with its own scales, into the 8-lane accumulator; the min
 * term the same fmaf chain. So every row's bits are the AVX2 kernel's, for any S. */
static inline __m512i gq_dp64(__m512i aw, __mmask64 neg, __m512i x) {
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), aw, _mm512_mask_sub_epi8(x, neg, _mm512_setzero_si512(), x));
}
static inline __m512i gq_pair(__m256i w0, __m256i w1) { return _mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1); }
#define GQ_FMA2(acc, pf, sc0, sc1) \
    acc = _mm256_fmadd_ps(_mm512_castps512_ps256(pf), sc0, acc); acc = _mm256_fmadd_ps(_mm512_extractf32x8_ps(pf, 1), sc1, acc);
/* rows of an activation block GQ_FUSED_ROWS at a time through BODY(y, ystride, row, xq, dx, xs, I, n) */
#define GQ_ROWS512(BODY, y, ystride, row, a, I) do { \
    for (int s0_ = 0; s0_ < (a)->S; s0_ += GQ_FUSED_ROWS) { \
        const int n_ = (a)->S - s0_ < GQ_FUSED_ROWS ? (a)->S - s0_ : GQ_FUSED_ROWS; \
        float *ys_ = (y) + (int64_t)s0_ * (ystride); const int8_t *xq_ = (a)->q + (int64_t)s0_ * (I); \
        const float *dx_ = (a)->d + (int64_t)s0_ * ((I) / 32); const int32_t *xs_ = (a)->s + (int64_t)s0_ * ((I) / 16); \
        switch (n_) { case 1: BODY(ys_, ystride, row, xq_, dx_, xs_, I, 1); break; \
                      case 2: BODY(ys_, ystride, row, xq_, dx_, xs_, I, 2); break; \
                      case 3: BODY(ys_, ystride, row, xq_, dx_, xs_, I, 3); break; \
                      default: BODY(ys_, ystride, row, xq_, dx_, xs_, I, 4); break; } \
    } } while (0)
#define GQ_DOT_SB256_512(NAME, BYTES, BD, SB) \
GQ_SB_FN void NAME##_512(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, const int32_t *xs, int I, const int S) { \
    (void)xs; __m256 vacc[GQ_FUSED_ROWS]; for (int s = 0; s < S; s++) vacc[s] = _mm256_setzero_ps(); \
    for (int64_t blk = 0; blk < I / 256; blk++) { \
        const uint8_t *b = row + blk * (BYTES); const float d = BD(b); \
        for (int ib = 0; ib < 8; ib += 2) { \
            float sa[2], sb[2]; const __m256i w0 = SB(b, d, ib, sa), w1 = SB(b, d, ib + 1, sb); \
            const __m512i w = gq_pair(w0, w1), aw = _mm512_abs_epi8(w); const __mmask64 neg = _mm512_movepi8_mask(w); \
            const __m512 ws = gq_w16(sa[0], sa[1], sb[0], sb[1]); const int64_t xb = blk * 8 + ib; \
            for (int s = 0; s < S; s++) { \
                const __m512 pf = _mm512_cvtepi32_ps(gq_dp64(aw, neg, _mm512_loadu_si512((const void *)(xq + (int64_t)s * I + xb * 32)))); \
                const __m512 sc = _mm512_mul_ps(ws, gq_x16(dx + (int64_t)s * (I / 32) + xb)); \
                GQ_FMA2V(vacc[s], pf, sc) \
            } \
        } \
    } \
    for (int s = 0; s < S; s++) y[(int64_t)s * ystride] = GQ_HSUM8(vacc[s]); \
}
GQ_DOT_SB256_512(gq_dot_iq4_xs, 136, gq_bd_iq4_xs, gq_sb_iq4_xs)
GQ_DOT_SB256_512(gq_dot_iq2_xxs, 66, gq_bd_f16, gq_sb_iq2_xxs)
GQ_DOT_SB256_512(gq_dot_iq2_xs, 74, gq_bd_f16, gq_sb_iq2_xs)
GQ_DOT_SB256_512(gq_dot_iq2_s, 82, gq_bd_f16, gq_sb_iq2_s)
GQ_DOT_SB256_512(gq_dot_iq3_xxs, 98, gq_bd_f16, gq_sb_iq3_xxs)
GQ_DOT_SB256_512(gq_dot_iq3_s, 110, gq_bd_f16, gq_sb_iq3_s)
GQ_DOT_SB256_512(gq_dot_iq1_s, 50, gq_bd_f16, gq_sb_iq1_s)
GQ_DOT_SB256_512(gq_dot_iq1_m, 56, gq_bd_iq1_m, gq_sb_iq1_m)
#define GQ_DOT_SBK_512(NAME, BYTES, SB, HASMIN) \
GQ_SB_FN void NAME##_512(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, const int32_t *xs, int I, const int S) { \
    __m256 vacc[GQ_FUSED_ROWS]; float mm[GQ_FUSED_ROWS]; \
    for (int s = 0; s < S; s++) { vacc[s] = _mm256_setzero_ps(); mm[s] = 0.f; } \
    for (int64_t blk = 0; blk < I / 256; blk++) { \
        const uint8_t *b = row + blk * (BYTES); \
        for (int ib = 0; ib < 8; ib += 2) { \
            float sa[2], ma[2], sb[2], mb[2]; const __m256i w0 = SB(b, ib, sa, ma), w1 = SB(b, ib + 1, sb, mb); \
            const __m512i w = gq_pair(w0, w1), aw = _mm512_abs_epi8(w); const __mmask64 neg = _mm512_movepi8_mask(w); \
            const __m512 ws = gq_w16(sa[0], sa[1], sb[0], sb[1]); const int64_t xb = blk * 8 + ib; \
            for (int s = 0; s < S; s++) { \
                const __m512 pf = _mm512_cvtepi32_ps(gq_dp64(aw, neg, _mm512_loadu_si512((const void *)(xq + (int64_t)s * I + xb * 32)))); \
                const float *dd = dx + (int64_t)s * (I / 32) + xb; \
                const __m512 sc = _mm512_mul_ps(ws, gq_x16(dd)); \
                GQ_FMA2V(vacc[s], pf, sc) \
                if (HASMIN) { const int32_t *xss = xs + (int64_t)s * (I / 16) + 2 * xb; \
                    mm[s] = fmaf(dd[0], fmaf(ma[0], (float)xss[0], ma[1] * (float)xss[1]), mm[s]); \
                    mm[s] = fmaf(dd[1], fmaf(mb[0], (float)xss[2], mb[1] * (float)xss[3]), mm[s]); } \
            } \
        } \
    } \
    for (int s = 0; s < S; s++) { float acc = GQ_HSUM8(vacc[s]); if (HASMIN) acc -= mm[s]; y[(int64_t)s * ystride] = acc; } \
}
GQ_DOT_SBK_512(gq_dot_q4_k, 144, gq_sbk_q4_k, 1)
GQ_DOT_SBK_512(gq_dot_q5_k, 176, gq_sbk_q5_k, 1)
GQ_DOT_SBK_512(gq_dot_q6_k, 210, gq_sbk_q6_k, 0)
GQ_DOT_SBK_512(gq_dot_q3_k, 110, gq_sbk_q3_k, 0)
/* Q8_0: two blocks a register, each with its own scale */
static inline __attribute__((always_inline)) void gq_dot_q8_0_512(float *y, int ystride, const uint8_t *row, const int8_t *xq,
                                                                 const float *dx, const int32_t *xs, int I, const int S) {
    (void)xs; __m256 vacc[GQ_FUSED_ROWS]; for (int s = 0; s < S; s++) vacc[s] = _mm256_setzero_ps();
    for (int64_t b = 0; b < I / 32; b += 2) {
        const uint8_t *b0 = row + b * 34, *b1 = b0 + 34; const float d0 = gq_f16(gq_rd16(b0)), d1 = gq_f16(gq_rd16(b1));
        const __m512i w = gq_pair(_mm256_loadu_si256((const __m256i *)(b0 + 2)), _mm256_loadu_si256((const __m256i *)(b1 + 2)));
        const __m512i aw = _mm512_abs_epi8(w); const __mmask64 neg = _mm512_movepi8_mask(w);
        const __m512 ws = gq_w16(d0, d0, d1, d1);
        for (int s = 0; s < S; s++) {
            const __m512 pf = _mm512_cvtepi32_ps(gq_dp64(aw, neg, _mm512_loadu_si512((const void *)(xq + (int64_t)s * I + b * 32))));
            const __m512 sc = _mm512_mul_ps(ws, gq_x16(dx + (int64_t)s * (I / 32) + b));
            GQ_FMA2V(vacc[s], pf, sc)
        }
    }
    for (int s = 0; s < S; s++) y[(int64_t)s * ystride] = GQ_HSUM8(vacc[s]);
}
/* the 32-element block types: gq_dot_b32_body's decode of one block, as it is */
static inline __attribute__((always_inline)) __m256i gq_b32_w(int t, const uint8_t *blk, float *d) {
    const int mx = t == GQ_MXFP4, nl = t == GQ_IQ4_NL, q5 = t == GQ_Q5_0 || t == GQ_Q5_1;
    const int off = mx ? 1 : t == GQ_Q4_0 || nl ? 2 : t == GQ_Q4_1 || t == GQ_Q5_0 ? (t == GQ_Q4_1 ? 4 : 6) : 8;
    const __m128i tab = _mm_loadu_si128((const __m128i *)(mx ? GGML_KVALUES_MXFP4 : GGML_KVALUES_IQ4NL));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i bias = _mm256_set1_epi8(t == GQ_Q4_0 ? 8 : t == GQ_Q5_0 ? 16 : 0), b16 = _mm256_set1_epi8(16);
    *d = mx ? gq_e8m0_half(blk[0]) : gq_f16(gq_rd16(blk));
    const __m128i v = _mm_loadu_si128((const __m128i *)(blk + off));
    __m128i lo = _mm_and_si128(v, m4), hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
    if (nl || mx) { lo = _mm_shuffle_epi8(tab, lo); hi = _mm_shuffle_epi8(tab, hi); }
    __m256i w = _mm256_set_m128i(hi, lo);
    if (q5) w = _mm256_or_si256(w, _mm256_and_si256(gq_bits32(gq_rd32(blk + (t == GQ_Q5_0 ? 2 : 4))), b16));
    return _mm256_sub_epi8(w, bias);
}
static inline __attribute__((always_inline)) void gq_dot_b32_512(int t, float *y, int ystride, const uint8_t *row, const int8_t *xq,
                                                                const float *dx, const int32_t *xs, int I, const int S) {
    const int hasmin = t == GQ_Q4_1 || t == GQ_Q5_1, bs = GQ_TYPES[t].bytes;
    __m256 vacc[GQ_FUSED_ROWS]; for (int s = 0; s < S; s++) vacc[s] = _mm256_setzero_ps();
    for (int64_t b = 0; b < I / 32; b += 2) {
        float d0, d1; const __m256i w0 = gq_b32_w(t, row + b * bs, &d0), w1 = gq_b32_w(t, row + (b + 1) * bs, &d1);
        const __m512i w = gq_pair(w0, w1), aw = _mm512_abs_epi8(w); const __mmask64 neg = _mm512_movepi8_mask(w);
        const __m512 ws = gq_w16(d0, d0, d1, d1);
        for (int s = 0; s < S; s++) {
            const __m512 pf = _mm512_cvtepi32_ps(gq_dp64(aw, neg, _mm512_loadu_si512((const void *)(xq + (int64_t)s * I + b * 32))));
            const __m512 sc = _mm512_mul_ps(ws, gq_x16(dx + (int64_t)s * (I / 32) + b));
            GQ_FMA2V(vacc[s], pf, sc)
        }
    }
    for (int s = 0; s < S; s++) {
        float acc = GQ_HSUM8(vacc[s]);
        if (hasmin) {   /* gq_dot_b32_body's min term, as it is */
            const float *dxs = dx + (int64_t)s * (I / 32); const int32_t *xss = xs + (int64_t)s * (I / 16);
            float m = 0.f;
            for (int64_t b = 0; b < I / 32; b++) {
                const float mw = -gq_f16(gq_rd16(row + b * bs + 2));
                m = fmaf(dxs[b], fmaf(mw, (float)xss[2 * b], mw * (float)xss[2 * b + 1]), m);
            }
            acc -= m;
        }
        y[(int64_t)s * ystride] = acc;
    }
}
#define GQ_B32_512(y, ystride, row, xq, dx, xs, I, n) gq_dot_b32_512(t, y, ystride, row, xq, dx, xs, I, n)
/* the AVX-512 kernel of a type, any S; 0 when the type has none here */
GQ_EXACT_FN static int gq_row_512(float *y, int ystride, const gq_act *a, const uint8_t *row, int t, int I) {
    switch (t) {
    case GQ_Q8_0:
        if (I % 64) return 0;
        GQ_ROWS512(gq_dot_q8_0_512, y, ystride, row, a, I); return 1;
    case GQ_Q4_0: case GQ_Q4_1: case GQ_Q5_0: case GQ_Q5_1: case GQ_IQ4_NL: case GQ_MXFP4:
        if (I % 64) return 0;
        GQ_ROWS512(GQ_B32_512, y, ystride, row, a, I); return 1;
    case GQ_Q4_K: GQ_ROWS512(gq_dot_q4_k_512, y, ystride, row, a, I); return 1;
    case GQ_Q5_K: GQ_ROWS512(gq_dot_q5_k_512, y, ystride, row, a, I); return 1;
    case GQ_Q6_K: GQ_ROWS512(gq_dot_q6_k_512, y, ystride, row, a, I); return 1;
    case GQ_Q3_K: GQ_ROWS512(gq_dot_q3_k_512, y, ystride, row, a, I); return 1;
    case GQ_IQ4_XS: GQ_ROWS512(gq_dot_iq4_xs_512, y, ystride, row, a, I); return 1;
    case GQ_IQ2_XXS: GQ_ROWS512(gq_dot_iq2_xxs_512, y, ystride, row, a, I); return 1;
    case GQ_IQ2_XS: GQ_ROWS512(gq_dot_iq2_xs_512, y, ystride, row, a, I); return 1;
    case GQ_IQ2_S: GQ_ROWS512(gq_dot_iq2_s_512, y, ystride, row, a, I); return 1;
    case GQ_IQ3_XXS: GQ_ROWS512(gq_dot_iq3_xxs_512, y, ystride, row, a, I); return 1;
    case GQ_IQ3_S: GQ_ROWS512(gq_dot_iq3_s_512, y, ystride, row, a, I); return 1;
    case GQ_IQ1_S: GQ_ROWS512(gq_dot_iq1_s_512, y, ystride, row, a, I); return 1;
    case GQ_IQ1_M: GQ_ROWS512(gq_dot_iq1_m_512, y, ystride, row, a, I); return 1;
    default: return 0;
    }
}
#endif
static inline void gq_row_all(float *y, int ystride, const gq_act *a, const uint8_t *row, int t, int I) {
#ifdef GQ_VNNI512
    if (t == GQ_Q2_0 && I % 64 == 0 && a->l4) { gq_dot_q2_0_512(y, ystride, row, a, I); return; }
    if (gq_row_512(y, ystride, a, row, t, I)) return;
#endif
#if defined(__AVX2__) && defined(__FMA__)
    if (t == GQ_Q8_0) {   /* no unpack to share: rows in groups of GQ_FUSED_ROWS */
        for (int s0 = 0; s0 < a->S; s0 += GQ_FUSED_ROWS) {
            const int n = a->S - s0 < GQ_FUSED_ROWS ? a->S - s0 : GQ_FUSED_ROWS;
            gq_dot_q8_0_rows(y + (int64_t)s0 * ystride, ystride, row, a->q + (int64_t)s0 * I, a->d + (int64_t)s0 * (I / 32), I, n);
        }
        return;
    }
    if (a->S <= GQ_FUSED_ROWS) {
        if (t == GQ_Q4_0 || t == GQ_Q4_1 || t == GQ_Q5_0 || t == GQ_Q5_1 || t == GQ_IQ4_NL || t == GQ_MXFP4) {
            gq_dot_b32(t, y, ystride, row, a->q, a->d, a->s, I, a->S); return;
        }
        const gq_fdot_fn f = gq_fdot_sb256(t);
        if (f) { f(y, ystride, row, a->q, a->d, I, a->S); return; }
        if (t == GQ_Q4_K) { gq_dot_q4_k(y, ystride, row, a->q, a->d, a->s, I, a->S); return; }
        if (t == GQ_Q2_0) { gq_dot_q2_0(y, ystride, row, a->q, a->d, I, a->S); return; }
        if (t == GQ_Q5_K) { gq_dot_q5_k(y, ystride, row, a->q, a->d, a->s, I, a->S); return; }
        if (t == GQ_Q6_K) { gq_dot_q6_k(y, ystride, row, a->q, a->d, a->s, I, a->S); return; }
        if (t == GQ_Q3_K) { gq_dot_q3_k(y, ystride, row, a->q, a->d, a->s, I, a->S); return; }
    }
#else
    if (t == GQ_Q8_0) {
        for (int s = 0; s < a->S; s++) y[(int64_t)s * ystride] = gq_dot_q8_0(row, a->q + (int64_t)s * I, a->d + (int64_t)s * (I / 32), I);
        return;
    }
#endif
    if (!gq_tbuf(I)) return;
    gq_unpack_row(t, row, I, gq_tw8, gq_tsw, gq_tmw);
    const int hm = gq_has_min(t);
    for (int s = 0; s < a->S; s++)
        y[(int64_t)s * ystride] = gq_dot_i8(gq_tw8, gq_tsw, gq_tmw, hm, a->q + (int64_t)s * I, a->d + (int64_t)s * (I / 32), a->s + (int64_t)s * (I / 16), I);
}
/* ---- Rows interleaved by 16 (GQ_*_R16) -------------------------------------------
 * The row kernels split a block's dot into eight lane sums of one row, each with a convert
 * and an fma, so their float work grows with rows x activation rows: on a prompt chunk the
 * trunk and the experts spent their time there, not in the integer products. Here 16 rows
 * share a register, lane r = row r: for every four consecutive elements (a quad) a group
 * block holds the 16 rows' four codes as 64 bytes, so one vpdpbusd against the activation's
 * broadcast quad advances all 16 rows, and the float work is one convert and one fma per
 * 32 elements (two converts for the types with a scale per 16) for all of them.
 * The codes are kept unsigned (u = w + bias: IQ4's values + 128, Q8_0's bytes ^ 0x80, Q3_K's
 * hmask as + 4) and the bias comes back exactly with the activation's sums (gq_act.s).
 * Numerics: per row and per 32 elements (16 where the type's scale is per 16), the exact
 * integer dot, then its scale: one rounding where the row kernels round eight lane sums,
 * so not their bits (a TF-NLL check, not a bit check); a row's result does not depend on
 * S or on the other rows of its group. Pack and unpack are exact inverses (the decoded
 * scales encode back to the same bytes): unpack gives the device tiers and the row
 * readers the file's rows. */
static inline int32_t gq_ld32(const void *p) { int32_t v; memcpy(&v, p, 4); return v; }
/* the 4 x 4 transpose of 2-bit codes Q2_0's form keeps: byte k of out, bits 2i = byte i of in,
 * bits 2k (its own inverse) */
static inline void gq_r16_t2(uint8_t *out, const uint8_t *in) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) { const uint32_t b = in[i]; v |= ((b | b << 6 | b << 12 | b << 18) & 0x03030303u) << (2 * i); }
    memcpy(out, &v, 4);
}
/* one group block from the 16 rows' blocks b[r] */
static void gq_r16_pack_blk(int t, uint8_t *o, const uint8_t *const *b) {
    for (int r = 0; r < 16; r++) {
        const uint8_t *s = b[r];
        switch (t) {
        case GQ_Q2_0_R16:   /* byte k of row r's word G: bits 2i = element 16G + 4i + k */
            memcpy(o + 2 * r, s, 2);
            for (int G = 0; G < 4; G++) gq_r16_t2(o + 32 + 64 * G + 4 * r, s + 2 + 4 * G);
            break;
        case GQ_Q4_K_R16: case GQ_Q5_K_R16: {
            const int five = t == GQ_Q5_K_R16, qo = five ? 832 : 320;
            memcpy(o + 2 * r, s, 2); memcpy(o + 32 + 2 * r, s + 2, 2);
            for (int ib = 0; ib < 8; ib++) { int sc, mn; gq_scale_min_k4(s + 4, ib, &sc, &mn); o[64 + 16 * ib + r] = (uint8_t)sc; o[192 + 16 * ib + r] = (uint8_t)mn; }
            if (five) for (int j = 0; j < 8; j++) memcpy(o + 320 + 64 * j + 4 * r, s + 16 + 4 * j, 4);
            for (int c = 0; c < 4; c++) for (int j = 0; j < 8; j++) memcpy(o + qo + (8 * c + j) * 64 + 4 * r, s + (five ? 48 : 16) + 32 * c + 4 * j, 4);
            break; }
        case GQ_Q6_K_R16:
            memcpy(o + 2 * r, s + 208, 2);
            for (int k = 0; k < 16; k++) o[32 + 16 * k + r] = s[192 + k];
            for (int c = 0; c < 2; c++) for (int a = 0; a < 2; a++) for (int j = 0; j < 8; j++)
                memcpy(o + 288 + ((2 * c + a) * 8 + j) * 64 + 4 * r, s + 64 * c + 32 * a + 4 * j, 4);
            for (int c = 0; c < 2; c++) for (int j = 0; j < 8; j++) memcpy(o + 2336 + (8 * c + j) * 64 + 4 * r, s + 128 + 32 * c + 4 * j, 4);
            break;
        case GQ_Q3_K_R16: {
            const uint8_t *sc = s + 96;
            memcpy(o + 2 * r, s + 108, 2);
            for (int k = 0; k < 16; k++) {
                const int l = k < 8 ? sc[k] & 15 : sc[k - 8] >> 4, hh = (sc[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
                o[32 + 16 * k + r] = (uint8_t)(int8_t)((l | hh << 4) - 32);
            }
            for (int c = 0; c < 2; c++) for (int j = 0; j < 8; j++) memcpy(o + 288 + (8 * c + j) * 64 + 4 * r, s + 32 + 32 * c + 4 * j, 4);
            for (int j = 0; j < 8; j++) memcpy(o + 1312 + 64 * j + 4 * r, s + 4 * j, 4);
            break; }
        case GQ_IQ4_XS_R16: {
            const uint16_t sh = gq_rd16(s + 2); const uint8_t *sl = s + 4;
            memcpy(o + 2 * r, s, 2);
            for (int ib = 0; ib < 8; ib++) {
                const int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
                o[32 + 16 * ib + r] = (uint8_t)(int8_t)(ls - 32);
            }
            for (int ib = 0; ib < 8; ib++) for (int q = 0; q < 4; q++) memcpy(o + 160 + (4 * ib + q) * 64 + 4 * r, s + 8 + 16 * ib + 4 * q, 4);
            break; }
        case GQ_Q8_0_R16:
            memcpy(o + 2 * r, s, 2);
            for (int j = 0; j < 8; j++) for (int e = 0; e < 4; e++) o[32 + 64 * j + 4 * r + e] = s[2 + 4 * j + e] ^ 0x80;
            break;
        case GQ_Q4_0_R16: case GQ_IQ4_NL_R16:
            memcpy(o + 2 * r, s, 2);
            for (int q = 0; q < 4; q++) memcpy(o + 32 + 64 * q + 4 * r, s + 2 + 4 * q, 4);
            break;
        }
    }
}
/* row r's block back from a group block */
static void gq_r16_unpack_blk(int t, const uint8_t *o, int r, uint8_t *s) {
    switch (t) {
    case GQ_Q2_0_R16:
        memcpy(s, o + 2 * r, 2);
        for (int G = 0; G < 4; G++) gq_r16_t2(s + 2 + 4 * G, o + 32 + 64 * G + 4 * r);
        break;
    case GQ_Q4_K_R16: case GQ_Q5_K_R16: {
        const int five = t == GQ_Q5_K_R16, qo = five ? 832 : 320;
        const uint8_t *sc = o + 64 + r, *mn = o + 192 + r; uint8_t *q = s + 4;
        memcpy(s, o + 2 * r, 2); memcpy(s + 2, o + 32 + 2 * r, 2);
        for (int j = 0; j < 4; j++) {
            q[j] = (uint8_t)((sc[16 * j] & 63) | ((sc[16 * (j + 4)] >> 4) << 6));
            q[j + 4] = (uint8_t)((mn[16 * j] & 63) | ((mn[16 * (j + 4)] >> 4) << 6));
            q[j + 8] = (uint8_t)((sc[16 * (j + 4)] & 15) | ((mn[16 * (j + 4)] & 15) << 4));
        }
        if (five) for (int j = 0; j < 8; j++) memcpy(s + 16 + 4 * j, o + 320 + 64 * j + 4 * r, 4);
        for (int c = 0; c < 4; c++) for (int j = 0; j < 8; j++) memcpy(s + (five ? 48 : 16) + 32 * c + 4 * j, o + qo + (8 * c + j) * 64 + 4 * r, 4);
        break; }
    case GQ_Q6_K_R16:
        memcpy(s + 208, o + 2 * r, 2);
        for (int k = 0; k < 16; k++) s[192 + k] = o[32 + 16 * k + r];
        for (int c = 0; c < 2; c++) for (int a = 0; a < 2; a++) for (int j = 0; j < 8; j++)
            memcpy(s + 64 * c + 32 * a + 4 * j, o + 288 + ((2 * c + a) * 8 + j) * 64 + 4 * r, 4);
        for (int c = 0; c < 2; c++) for (int j = 0; j < 8; j++) memcpy(s + 128 + 32 * c + 4 * j, o + 2336 + (8 * c + j) * 64 + 4 * r, 4);
        break;
    case GQ_Q3_K_R16: {
        uint8_t *sc = s + 96; memset(sc, 0, 12);
        memcpy(s + 108, o + 2 * r, 2);
        for (int k = 0; k < 16; k++) {
            const int u = (int8_t)o[32 + 16 * k + r] + 32;
            if (k < 8) sc[k] |= (uint8_t)(u & 15); else sc[k - 8] |= (uint8_t)((u & 15) << 4);
            sc[8 + (k & 3)] |= (uint8_t)(((u >> 4) & 3) << (2 * (k >> 2)));
        }
        for (int c = 0; c < 2; c++) for (int j = 0; j < 8; j++) memcpy(s + 32 + 32 * c + 4 * j, o + 288 + (8 * c + j) * 64 + 4 * r, 4);
        for (int j = 0; j < 8; j++) memcpy(s + 4 * j, o + 1312 + 64 * j + 4 * r, 4);
        break; }
    case GQ_IQ4_XS_R16: {
        uint16_t sh = 0; uint8_t *sl = s + 4; memset(sl, 0, 4);
        memcpy(s, o + 2 * r, 2);
        for (int ib = 0; ib < 8; ib++) {
            const int u = (int8_t)o[32 + 16 * ib + r] + 32;
            sl[ib >> 1] |= (uint8_t)((u & 15) << (4 * (ib & 1))); sh |= (uint16_t)(((u >> 4) & 3) << (2 * ib));
        }
        memcpy(s + 2, &sh, 2);
        for (int ib = 0; ib < 8; ib++) for (int q = 0; q < 4; q++) memcpy(s + 8 + 16 * ib + 4 * q, o + 160 + (4 * ib + q) * 64 + 4 * r, 4);
        break; }
    case GQ_Q8_0_R16:
        memcpy(s, o + 2 * r, 2);
        for (int j = 0; j < 8; j++) for (int e = 0; e < 4; e++) s[2 + 4 * j + e] = o[32 + 64 * j + 4 * r + e] ^ 0x80;
        break;
    case GQ_Q4_0_R16: case GQ_IQ4_NL_R16:
        memcpy(s, o + 2 * r, 2);
        for (int q = 0; q < 4; q++) memcpy(s + 2 + 4 * q, o + 32 + 64 * q + 4 * r, 4);
        break;
    }
}
/* rows (whole groups) of the base type's rows at src into t's form at dst (not src) */
static void gq_r16_pack(int t, uint8_t *dst, const uint8_t *src, int rows, int I) {
    if (gq_r16_base(t) < 0) return;
    const int base = gq_r16_base(t), bb = GQ_TYPES[base].bytes, nb = I / GQ_TYPES[base].blck, gb = gq_r16_group_bytes(t);
    const int64_t rb = (int64_t)nb * bb;
    if (t == GQ_Q2_0_R16) {   /* the experts' form, packed at every read: row by row, no dispatch */
        for (int g = 0; g < rows / 16; g++)
            for (int r = 0; r < 16; r++) {
                const uint8_t *row = src + (int64_t)(16 * g + r) * rb; uint8_t *o = dst + (int64_t)g * nb * gb + 4 * r;
                for (int b = 0; b < nb; b++, row += 18, o += 288) {
                    memcpy(o - 2 * r, row, 2);
                    for (int G = 0; G < 4; G++) gq_r16_t2(o + 32 + 64 * G, row + 2 + 4 * G);
                }
            }
        return;
    }
    for (int g = 0; g < rows / 16; g++)
        for (int b = 0; b < nb; b++) {
            const uint8_t *rp[16];
            for (int r = 0; r < 16; r++) rp[r] = src + (int64_t)(16 * g + r) * rb + (int64_t)b * bb;
            gq_r16_pack_blk(t, dst + ((int64_t)g * nb + b) * gb, rp);
        }
}
/* the inverse: rows of t's form back to the base type's rows */
static void gq_r16_unpack(int t, uint8_t *dst, const uint8_t *src, int rows, int I) {
    if (gq_r16_base(t) < 0) return;
    const int base = gq_r16_base(t), bb = GQ_TYPES[base].bytes, nb = I / GQ_TYPES[base].blck, gb = gq_r16_group_bytes(t);
    const int64_t rb = (int64_t)nb * bb;
    for (int g = 0; g < rows / 16; g++)
        for (int b = 0; b < nb; b++)
            for (int r = 0; r < 16; r++)
                gq_r16_unpack_blk(t, src + ((int64_t)g * nb + b) * gb, r, dst + (int64_t)(16 * g + r) * rb + (int64_t)b * bb);
}
/* row `row` of a matrix of t's form, dequantized as its base type's row */
static int gq_r16_dequant_row(int t, const uint8_t *W, int64_t row, float *out, int I) {
    if (gq_r16_base(t) < 0) return -1;
    const int base = gq_r16_base(t), bl = GQ_TYPES[base].blck, nb = I / bl, gb = gq_r16_group_bytes(t);
    const uint8_t *grp = W + (row / 16) * (int64_t)nb * gb; uint8_t blk[256];
    for (int b = 0; b < nb; b++) {
        gq_r16_unpack_blk(t, grp + (int64_t)b * gb, (int)(row % 16), blk);
        if (gq_dequant_row(base, blk, out + (int64_t)b * bl, bl)) return -1;
    }
    return 0;
}
#ifdef GQ_VNNI512
/* The kernels: 16 rows of one group (y[s * ystride + r]) against S <= 8 activation rows. */
#define GQ_R16_ARGS float *y, int ystride, const uint8_t *grp, const int8_t *xq, const float *dx, const int32_t *xs, int I, const int S
#define GQ_R16_FN static inline __attribute__((always_inline)) void
#define GQ_R16_X4(xb, j) _mm512_set1_epi32(gq_ld32((xb) + 4 * (j)))
/* the 32 elements at index g32 (of 32) of the row: codes c[8] (quad j = elements 4j..4j+3),
 * sv the rows' scales, nv their mins (MIN), codes = w + BIAS */
#define GQ_R16_ACC32(c, sv, MIN, nv, BIAS) \
    for (int s = 0; s < S; s++) { \
        const int8_t *xb_ = xq + (int64_t)s * I + 32 * (int64_t)g32; \
        __m512i p_ = _mm512_dpbusd_epi32(_mm512_setzero_si512(), (c)[0], GQ_R16_X4(xb_, 0)); \
        for (int j_ = 1; j_ < 8; j_++) p_ = _mm512_dpbusd_epi32(p_, (c)[j_], GQ_R16_X4(xb_, j_)); \
        const int32_t *x16_ = xs + (int64_t)s * (I / 16) + 2 * (int64_t)g32; const int32_t x32_ = x16_[0] + x16_[1]; \
        const float d_ = dx[(int64_t)s * (I / 32) + g32]; \
        if ((BIAS) != 0) p_ = _mm512_sub_epi32(p_, _mm512_set1_epi32((BIAS) * x32_)); \
        acc[s] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p_), _mm512_mul_ps((sv), _mm512_set1_ps(d_)), acc[s]); \
        if (MIN) acc[s] = _mm512_fnmadd_ps((nv), _mm512_set1_ps(d_ * (float)x32_), acc[s]); \
    }
/* the same with a scale per 16 elements: sv0 for quads 0-3, sv1 for 4-7 */
#define GQ_R16_ACC16(c, sv0, sv1, BIAS) \
    for (int s = 0; s < S; s++) { \
        const int8_t *xb_ = xq + (int64_t)s * I + 32 * (int64_t)g32; \
        __m512i p0_ = _mm512_dpbusd_epi32(_mm512_setzero_si512(), (c)[0], GQ_R16_X4(xb_, 0)); \
        __m512i p1_ = _mm512_dpbusd_epi32(_mm512_setzero_si512(), (c)[4], GQ_R16_X4(xb_, 4)); \
        for (int j_ = 1; j_ < 4; j_++) { p0_ = _mm512_dpbusd_epi32(p0_, (c)[j_], GQ_R16_X4(xb_, j_)); \
                                         p1_ = _mm512_dpbusd_epi32(p1_, (c)[4 + j_], GQ_R16_X4(xb_, 4 + j_)); } \
        const int32_t *x16_ = xs + (int64_t)s * (I / 16) + 2 * (int64_t)g32; \
        p0_ = _mm512_sub_epi32(p0_, _mm512_set1_epi32((BIAS) * x16_[0])); p1_ = _mm512_sub_epi32(p1_, _mm512_set1_epi32((BIAS) * x16_[1])); \
        const __m512 t_ = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p1_), (sv1), _mm512_mul_ps(_mm512_cvtepi32_ps(p0_), (sv0))); \
        acc[s] = _mm512_fmadd_ps(t_, _mm512_set1_ps(dx[(int64_t)s * (I / 32) + g32]), acc[s]); \
    }
#define GQ_R16_ZERO __m512 acc[8]; for (int s = 0; s < S; s++) acc[s] = _mm512_setzero_ps();
#define GQ_R16_STORE for (int s = 0; s < S; s++) _mm512_storeu_ps(y + (int64_t)s * ystride, acc[s]);
#define GQ_R16_SC8(p) _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(p))))
#define GQ_R16_SCU8(p) _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(p))))
#define GQ_R16_D(p) _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(p)))
/* Q2_0: per 64-element block the 16 f16 scales, then four 64-byte words G, row r's four bytes
 * at 4r: byte k holds in bits 2i the code of element 16G + 4i + k, so quad 4G + i of all 16
 * rows is one shift and one mask (a 2-bit transpose of the file's bytes 4G .. 4G + 3) */
GQ_R16_FN gq_r16_q2_0(GQ_R16_ARGS) {
    const __m512i m3 = _mm512_set1_epi8(3);
    GQ_R16_ZERO
    for (int b = 0; b < I / 64; b++) {
        const uint8_t *o = grp + (int64_t)b * 288;
        const __m512 dv = GQ_R16_D(o);
        for (int hf = 0; hf < 2; hf++) {
            const int g32 = 2 * b + hf; __m512i c[8];
            for (int G = 0; G < 2; G++) {
                const __m512i v = _mm512_loadu_si512(o + 32 + 64 * (2 * hf + G));
                c[4 * G] = _mm512_and_si512(v, m3);
                c[4 * G + 1] = _mm512_and_si512(_mm512_srli_epi16(v, 2), m3);
                c[4 * G + 2] = _mm512_and_si512(_mm512_srli_epi16(v, 4), m3);
                c[4 * G + 3] = _mm512_and_si512(_mm512_srli_epi16(v, 6), m3);
            }
            GQ_R16_ACC32(c, dv, 0, dv, 1)
        }
    }
    GQ_R16_STORE
}
/* Q4_K / Q5_K: d, dmin, the eight sub-blocks' scales and mins (a byte a row), Q5_K's high
 * bits (quad j of qh: bit ib = sub-block ib), then the nibbles: sub-blocks 2c and 2c+1 are
 * the low and high nibbles of the same 32 bytes */
GQ_R16_FN gq_r16_q45_k(GQ_R16_ARGS, const int five) {
    const int gb = five ? 2880 : 2368, qo = five ? 832 : 320;
    const __m512i m4 = _mm512_set1_epi8(0x0F), sixteen = _mm512_set1_epi8(16);
    GQ_R16_ZERO
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *o = grp + (int64_t)b * gb;
        const __m512 dv = GQ_R16_D(o), mv = GQ_R16_D(o + 32);
        for (int ib = 0; ib < 8; ib++) {
            const int g32 = 8 * b + ib; const __m128i sh = _mm_cvtsi32_si128(4 * (ib & 1));
            __m512i c[8];
            for (int j = 0; j < 8; j++) {
                c[j] = _mm512_and_si512(_mm512_srl_epi16(_mm512_loadu_si512(o + qo + (8 * (ib >> 1) + j) * 64), sh), m4);
                if (five) c[j] = _mm512_mask_add_epi8(c[j], _mm512_test_epi8_mask(_mm512_loadu_si512(o + 320 + 64 * j), _mm512_set1_epi8((char)(1 << ib))), c[j], sixteen);
            }
            const __m512 sv = _mm512_mul_ps(dv, GQ_R16_SCU8(o + 64 + 16 * ib)), nv = _mm512_mul_ps(mv, GQ_R16_SCU8(o + 192 + 16 * ib));
            GQ_R16_ACC32(c, sv, 1, nv, 0)
        }
    }
    GQ_R16_STORE
}
GQ_R16_FN gq_r16_q4_k(GQ_R16_ARGS) { gq_r16_q45_k(y, ystride, grp, xq, dx, xs, I, S, 0); }
GQ_R16_FN gq_r16_q5_k(GQ_R16_ARGS) { gq_r16_q45_k(y, ystride, grp, xq, dx, xs, I, S, 1); }
/* Q6_K: d, the 16 int8 scales, ql (sub-block 4c+k: nibble k>>1 of half k&1 of 64-byte part c),
 * qh (bits 2k of part c's 32 bytes); codes = w + 32 */
GQ_R16_FN gq_r16_q6_k(GQ_R16_ARGS) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), m30 = _mm512_set1_epi8(0x30);
    GQ_R16_ZERO
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *o = grp + (int64_t)b * 3360;
        const __m512 dv = GQ_R16_D(o);
        for (int ib = 0; ib < 8; ib++) {
            const int g32 = 8 * b + ib, cc = ib >> 2, k = ib & 3;
            const __m128i sl = _mm_cvtsi32_si128(4 * (k >> 1));
            __m512i c[8];
            for (int j = 0; j < 8; j++) {
                const __m512i lo = _mm512_and_si512(_mm512_srl_epi16(_mm512_loadu_si512(o + 288 + ((2 * cc + (k & 1)) * 8 + j) * 64), sl), m4);
                const __m512i hv = _mm512_loadu_si512(o + 2336 + (8 * cc + j) * 64);
                const __m512i h = k == 0 ? _mm512_slli_epi16(hv, 4) : k == 1 ? _mm512_slli_epi16(hv, 2) : k == 2 ? hv : _mm512_srli_epi16(hv, 2);
                c[j] = _mm512_or_si512(lo, _mm512_and_si512(h, m30));
            }
            const __m512 sv0 = _mm512_mul_ps(dv, GQ_R16_SC8(o + 32 + 32 * ib)), sv1 = _mm512_mul_ps(dv, GQ_R16_SC8(o + 48 + 32 * ib));
            GQ_R16_ACC16(c, sv0, sv1, 32)
        }
    }
    GQ_R16_STORE
}
/* Q3_K: d, the 16 scales (int8, -32 applied), qs (sub-block 4c+sh: bits 2sh of part c's 32
 * bytes), hmask (quad j: bit ib = sub-block ib, set = no -4); codes = w + 4 */
GQ_R16_FN gq_r16_q3_k(GQ_R16_ARGS) {
    const __m512i m3 = _mm512_set1_epi8(3), four = _mm512_set1_epi8(4);
    GQ_R16_ZERO
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *o = grp + (int64_t)b * 1824;
        const __m512 dv = GQ_R16_D(o);
        for (int ib = 0; ib < 8; ib++) {
            const int g32 = 8 * b + ib; const __m128i sh = _mm_cvtsi32_si128(2 * (ib & 3));
            __m512i c[8];
            for (int j = 0; j < 8; j++) {
                const __m512i q2 = _mm512_and_si512(_mm512_srl_epi16(_mm512_loadu_si512(o + 288 + (8 * (ib >> 2) + j) * 64), sh), m3);
                c[j] = _mm512_mask_add_epi8(q2, _mm512_test_epi8_mask(_mm512_loadu_si512(o + 1312 + 64 * j), _mm512_set1_epi8((char)(1 << ib))), q2, four);
            }
            const __m512 sv0 = _mm512_mul_ps(dv, GQ_R16_SC8(o + 32 + 32 * ib)), sv1 = _mm512_mul_ps(dv, GQ_R16_SC8(o + 48 + 32 * ib));
            GQ_R16_ACC16(c, sv0, sv1, 4)
        }
    }
    GQ_R16_STORE
}
/* IQ4_XS: d, the eight sub-block scales (int8, -32 applied), then per sub-block four quads of
 * bytes: low nibbles quads 0-3, high nibbles 4-7, looked up as value + 128 */
GQ_R16_FN gq_r16_iq4_xs(GQ_R16_ARGS) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i tab = _mm512_broadcast_i32x4(_mm_xor_si128(_mm_loadu_si128((const __m128i *)GGML_KVALUES_IQ4NL), _mm_set1_epi8((char)0x80)));
    GQ_R16_ZERO
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *o = grp + (int64_t)b * 2208;
        const __m512 dv = GQ_R16_D(o);
        for (int ib = 0; ib < 8; ib++) {
            const int g32 = 8 * b + ib;
            __m512i c[8];
            for (int q = 0; q < 4; q++) {
                const __m512i v = _mm512_loadu_si512(o + 160 + (4 * ib + q) * 64);
                c[q] = _mm512_shuffle_epi8(tab, _mm512_and_si512(v, m4));
                c[4 + q] = _mm512_shuffle_epi8(tab, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4));
            }
            const __m512 sv = _mm512_mul_ps(dv, GQ_R16_SC8(o + 32 + 16 * ib));
            GQ_R16_ACC32(c, sv, 0, sv, 128)
        }
    }
    GQ_R16_STORE
}
/* Q8_0: d, then the 32 bytes ^ 0x80 (w + 128) as 8 quads */
GQ_R16_FN gq_r16_q8_0(GQ_R16_ARGS) {
    GQ_R16_ZERO
    for (int b = 0; b < I / 32; b++) {
        const uint8_t *o = grp + (int64_t)b * 544; const int g32 = b;
        const __m512 dv = GQ_R16_D(o);
        __m512i c[8];
        for (int j = 0; j < 8; j++) c[j] = _mm512_loadu_si512(o + 32 + 64 * j);
        GQ_R16_ACC32(c, dv, 0, dv, 128)
    }
    GQ_R16_STORE
}
/* Q4_0 (nibble = w + 8) and IQ4_NL (looked up as value + 128): d, then 4 quads of bytes, low
 * nibbles elements 0-15, high 16-31 */
GQ_R16_FN gq_r16_q4_0_nl(GQ_R16_ARGS, const int nl) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i tab = _mm512_broadcast_i32x4(_mm_xor_si128(_mm_loadu_si128((const __m128i *)GGML_KVALUES_IQ4NL), _mm_set1_epi8((char)0x80)));
    GQ_R16_ZERO
    for (int b = 0; b < I / 32; b++) {
        const uint8_t *o = grp + (int64_t)b * 288; const int g32 = b;
        const __m512 dv = GQ_R16_D(o);
        __m512i c[8];
        for (int q = 0; q < 4; q++) {
            const __m512i v = _mm512_loadu_si512(o + 32 + 64 * q);
            c[q] = _mm512_and_si512(v, m4); c[4 + q] = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4);
            if (nl) { c[q] = _mm512_shuffle_epi8(tab, c[q]); c[4 + q] = _mm512_shuffle_epi8(tab, c[4 + q]); }
        }
        GQ_R16_ACC32(c, dv, 0, dv, nl ? 128 : 8)
    }
    GQ_R16_STORE
}
GQ_R16_FN gq_r16_q4_0(GQ_R16_ARGS) { gq_r16_q4_0_nl(y, ystride, grp, xq, dx, xs, I, S, 0); }
GQ_R16_FN gq_r16_iq4_nl(GQ_R16_ARGS) { gq_r16_q4_0_nl(y, ystride, grp, xq, dx, xs, I, S, 1); }
#define GQ_R16_N(FN) do { switch (n) { \
        case 1: FN(ys, ystride, grp, xq, dx, xs, I, 1); break; case 2: FN(ys, ystride, grp, xq, dx, xs, I, 2); break; \
        case 3: FN(ys, ystride, grp, xq, dx, xs, I, 3); break; case 4: FN(ys, ystride, grp, xq, dx, xs, I, 4); break; \
        default: FN(ys, ystride, grp, xq, dx, xs, I, n); break; } } while (0)
/* group grp's 16 rows (output rows at y .. y + 15) against every activation row of a */
static void gq_r16_rows(int t, float *y, int ystride, const uint8_t *grp, const gq_act *a, int I) {
    for (int s0 = 0; s0 < a->S; s0 += 8) {
        const int n = a->S - s0 < 8 ? a->S - s0 : 8;
        float *ys = y + (int64_t)s0 * ystride; const int8_t *xq = a->q + (int64_t)s0 * I;
        const float *dx = a->d + (int64_t)s0 * (I / 32); const int32_t *xs = a->s + (int64_t)s0 * (I / 16);
        switch (t) {
        case GQ_Q2_0_R16: GQ_R16_N(gq_r16_q2_0); break;
        case GQ_Q4_K_R16: GQ_R16_N(gq_r16_q4_k); break;
        case GQ_Q5_K_R16: GQ_R16_N(gq_r16_q5_k); break;
        case GQ_Q6_K_R16: GQ_R16_N(gq_r16_q6_k); break;
        case GQ_Q3_K_R16: GQ_R16_N(gq_r16_q3_k); break;
        case GQ_IQ4_XS_R16: GQ_R16_N(gq_r16_iq4_xs); break;
        case GQ_Q8_0_R16: GQ_R16_N(gq_r16_q8_0); break;
        case GQ_Q4_0_R16: GQ_R16_N(gq_r16_q4_0); break;
        case GQ_IQ4_NL_R16: GQ_R16_N(gq_r16_iq4_nl); break;
        }
    }
}
#endif
#if !defined(GQ_VNNI512) && defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
/* The same forms on AVX2 (x86-64-v3, AVX-VNNI laptops): a group's 16 rows as two halves of
 * eight (a quad's 64 bytes: rows 0-7 then 8-15), one half at a time and up to four activation
 * rows a pass (16 ymm registers). Every lane runs the AVX-512 kernels' integer sum and float
 * steps in the same order, so a row's result is theirs to the bit. Without VNNI the products
 * go through maddubs, which saturates a pair above 2^15: fine for codes up to 63 (every type
 * but Q8_0 and the IQ4 lookups), and those two take the signed values with the sign trick
 * (|w| times sign(x, w)) instead of the bias: the same integer. */
#define GQ_R16_AVX2 1
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define GQ_R16_VNNI 1
#define GQ_R16_DPU(acc, u, x) _mm256_dpbusd_epi32((acc), (u), (x))
#elif defined(__AVXVNNI__)
#define GQ_R16_VNNI 1
#define GQ_R16_DPU(acc, u, x) _mm256_dpbusd_avx_epi32((acc), (u), (x))
#else
#define GQ_R16_DPU(acc, u, x) _mm256_add_epi32((acc), _mm256_madd_epi16(_mm256_maddubs_epi16((u), (x)), _mm256_set1_epi16(1)))
#endif
#define GQ_R16_DPS(acc, w, x) _mm256_add_epi32((acc), _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8((x), (w))), _mm256_set1_epi16(1)))
#ifdef GQ_R16_VNNI
#define GQ_R16_BIG(acc, c, x) GQ_R16_DPU(acc, c, x)   /* Q8_0, IQ4: the biased codes */
#define GQ_R16_BIGBIAS 128
#else
#define GQ_R16_BIG(acc, c, x) GQ_R16_DPS(acc, c, x)   /* the signed values */
#define GQ_R16_BIGBIAS 0
#endif
#define GQ_R16H_ARGS float *y, int ystride, const uint8_t *grp, const int8_t *xq, const float *dx, const int32_t *xs, int I, const int S, const int h
#define GQ_R16H_FN static inline __attribute__((always_inline)) void
#define GQ_R16H_X4(xb, j) _mm256_set1_epi32(gq_ld32((xb) + 4 * (j)))
/* the int32 sums of quads j0 .. j0 + n - 1 (n even): with VNNI a vpdpbusd each; without, two
 * maddubs added in 16 bits before one madd (codes up to 63: a pair of pairs stays below 2^15) */
#ifdef GQ_R16_VNNI
#define GQ_R16H_DOTU(p, c, xb, j0, n) { p = GQ_R16_DPU(_mm256_setzero_si256(), (c)[j0], GQ_R16H_X4(xb, j0)); \
        for (int k_ = (j0) + 1; k_ < (j0) + (n); k_++) p = GQ_R16_DPU(p, (c)[k_], GQ_R16H_X4(xb, k_)); }
#else
#define GQ_R16H_DOTU(p, c, xb, j0, n) { const __m256i o16_ = _mm256_set1_epi16(1); p = _mm256_setzero_si256(); \
        for (int k_ = (j0); k_ < (j0) + (n); k_ += 2) \
            p = _mm256_add_epi32(p, _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16((c)[k_], GQ_R16H_X4(xb, k_)), \
                                                                       _mm256_maddubs_epi16((c)[k_ + 1], GQ_R16H_X4(xb, k_ + 1))), o16_)); }
#endif
/* the same for Q8_0 and the IQ4 lookups (GQ_R16_BIG: biased with VNNI, signed without) */
#define GQ_R16H_DOTB(p, c, xb, j0, n) { p = GQ_R16_BIG(_mm256_setzero_si256(), (c)[j0], GQ_R16H_X4(xb, j0)); \
        for (int k_ = (j0) + 1; k_ < (j0) + (n); k_++) p = GQ_R16_BIG(p, (c)[k_], GQ_R16H_X4(xb, k_)); }
#define GQ_R16H_ACC32(c, sv, MIN, nv, BIAS, DOT) \
    for (int s = 0; s < S; s++) { \
        const int8_t *xb_ = xq + (int64_t)s * I + 32 * (int64_t)g32; \
        __m256i p_; DOT(p_, c, xb_, 0, 8) \
        const int32_t *x16_ = xs + (int64_t)s * (I / 16) + 2 * (int64_t)g32; const int32_t x32_ = x16_[0] + x16_[1]; \
        const float d_ = dx[(int64_t)s * (I / 32) + g32]; \
        if ((BIAS) != 0) p_ = _mm256_sub_epi32(p_, _mm256_set1_epi32((BIAS) * x32_)); \
        acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p_), _mm256_mul_ps((sv), _mm256_set1_ps(d_)), acc[s]); \
        if (MIN) acc[s] = _mm256_fnmadd_ps((nv), _mm256_set1_ps(d_ * (float)x32_), acc[s]); \
    }
#define GQ_R16H_ACC16(c, sv0, sv1, BIAS) \
    for (int s = 0; s < S; s++) { \
        const int8_t *xb_ = xq + (int64_t)s * I + 32 * (int64_t)g32; \
        __m256i p0_, p1_; GQ_R16H_DOTU(p0_, c, xb_, 0, 4) GQ_R16H_DOTU(p1_, c, xb_, 4, 4) \
        const int32_t *x16_ = xs + (int64_t)s * (I / 16) + 2 * (int64_t)g32; \
        p0_ = _mm256_sub_epi32(p0_, _mm256_set1_epi32((BIAS) * x16_[0])); p1_ = _mm256_sub_epi32(p1_, _mm256_set1_epi32((BIAS) * x16_[1])); \
        const __m256 t_ = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1_), (sv1), _mm256_mul_ps(_mm256_cvtepi32_ps(p0_), (sv0))); \
        acc[s] = _mm256_fmadd_ps(t_, _mm256_set1_ps(dx[(int64_t)s * (I / 32) + g32]), acc[s]); \
    }
#define GQ_R16H_ZERO __m256 acc[4]; for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
#define GQ_R16H_STORE for (int s = 0; s < S; s++) _mm256_storeu_ps(y + (int64_t)s * ystride + 8 * h, acc[s]);
#define GQ_R16H_LD(p) _mm256_loadu_si256((const __m256i *)(p))
#define GQ_R16H_D(p) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(p)))   /* eight f16 */
#define GQ_R16H_SC8(p) _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(p))))
#define GQ_R16H_SCU8(p) _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(p))))
/* byte k of every four of v set where bit k of `bits` is: (b >> s) & 1 tested as a mask */
#define GQ_R16H_BIT(v, b) _mm256_cmpeq_epi8(_mm256_and_si256((v), _mm256_set1_epi8((char)(b))), _mm256_set1_epi8((char)(b)))
GQ_R16H_FN gq_r16h_q2_0(GQ_R16H_ARGS) {
    const __m256i m3 = _mm256_set1_epi8(3);
    GQ_R16H_ZERO
    for (int b = 0; b < I / 64; b++) {
        const uint8_t *o = grp + (int64_t)b * 288;
        const __m256 dv = GQ_R16H_D(o + 16 * h);
        for (int hf = 0; hf < 2; hf++) {
            const int g32 = 2 * b + hf; __m256i c[8];
            for (int G = 0; G < 2; G++) {
                const __m256i v = GQ_R16H_LD(o + 32 + 64 * (2 * hf + G) + 32 * h);
                c[4 * G] = _mm256_and_si256(v, m3);
                c[4 * G + 1] = _mm256_and_si256(_mm256_srli_epi16(v, 2), m3);
                c[4 * G + 2] = _mm256_and_si256(_mm256_srli_epi16(v, 4), m3);
                c[4 * G + 3] = _mm256_and_si256(_mm256_srli_epi16(v, 6), m3);
            }
            GQ_R16H_ACC32(c, dv, 0, dv, 1, GQ_R16H_DOTU)
        }
    }
    GQ_R16H_STORE
}
GQ_R16H_FN gq_r16h_q45_k(GQ_R16H_ARGS, const int five) {
    const int gb = five ? 2880 : 2368, qo = five ? 832 : 320;
    const __m256i m4 = _mm256_set1_epi8(0x0F), sixteen = _mm256_set1_epi8(16);
    GQ_R16H_ZERO
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *o = grp + (int64_t)b * gb;
        const __m256 dv = GQ_R16H_D(o + 16 * h), mv = GQ_R16H_D(o + 32 + 16 * h);
        for (int ib = 0; ib < 8; ib++) {
            const int g32 = 8 * b + ib; const __m128i sh = _mm_cvtsi32_si128(4 * (ib & 1));
            __m256i c[8];
            for (int j = 0; j < 8; j++) {
                c[j] = _mm256_and_si256(_mm256_srl_epi16(GQ_R16H_LD(o + qo + (8 * (ib >> 1) + j) * 64 + 32 * h), sh), m4);
                if (five) c[j] = _mm256_add_epi8(c[j], _mm256_and_si256(GQ_R16H_BIT(GQ_R16H_LD(o + 320 + 64 * j + 32 * h), 1 << ib), sixteen));
            }
            const __m256 sv = _mm256_mul_ps(dv, GQ_R16H_SCU8(o + 64 + 16 * ib + 8 * h)), nv = _mm256_mul_ps(mv, GQ_R16H_SCU8(o + 192 + 16 * ib + 8 * h));
            GQ_R16H_ACC32(c, sv, 1, nv, 0, GQ_R16H_DOTU)
        }
    }
    GQ_R16H_STORE
}
GQ_R16H_FN gq_r16h_q4_k(GQ_R16H_ARGS) { gq_r16h_q45_k(y, ystride, grp, xq, dx, xs, I, S, h, 0); }
GQ_R16H_FN gq_r16h_q5_k(GQ_R16H_ARGS) { gq_r16h_q45_k(y, ystride, grp, xq, dx, xs, I, S, h, 1); }
GQ_R16H_FN gq_r16h_q6_k(GQ_R16H_ARGS) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m30 = _mm256_set1_epi8(0x30);
    GQ_R16H_ZERO
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *o = grp + (int64_t)b * 3360;
        const __m256 dv = GQ_R16H_D(o + 16 * h);
        for (int ib = 0; ib < 8; ib++) {
            const int g32 = 8 * b + ib, cc = ib >> 2, k = ib & 3;
            const __m128i sl = _mm_cvtsi32_si128(4 * (k >> 1));
            __m256i c[8];
            for (int j = 0; j < 8; j++) {
                const __m256i lo = _mm256_and_si256(_mm256_srl_epi16(GQ_R16H_LD(o + 288 + ((2 * cc + (k & 1)) * 8 + j) * 64 + 32 * h), sl), m4);
                const __m256i hv = GQ_R16H_LD(o + 2336 + (8 * cc + j) * 64 + 32 * h);
                const __m256i hh = k == 0 ? _mm256_slli_epi16(hv, 4) : k == 1 ? _mm256_slli_epi16(hv, 2) : k == 2 ? hv : _mm256_srli_epi16(hv, 2);
                c[j] = _mm256_or_si256(lo, _mm256_and_si256(hh, m30));
            }
            const __m256 sv0 = _mm256_mul_ps(dv, GQ_R16H_SC8(o + 32 + 32 * ib + 8 * h)), sv1 = _mm256_mul_ps(dv, GQ_R16H_SC8(o + 48 + 32 * ib + 8 * h));
            GQ_R16H_ACC16(c, sv0, sv1, 32)
        }
    }
    GQ_R16H_STORE
}
GQ_R16H_FN gq_r16h_q3_k(GQ_R16H_ARGS) {
    const __m256i m3 = _mm256_set1_epi8(3), four = _mm256_set1_epi8(4);
    GQ_R16H_ZERO
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *o = grp + (int64_t)b * 1824;
        const __m256 dv = GQ_R16H_D(o + 16 * h);
        for (int ib = 0; ib < 8; ib++) {
            const int g32 = 8 * b + ib; const __m128i sh = _mm_cvtsi32_si128(2 * (ib & 3));
            __m256i c[8];
            for (int j = 0; j < 8; j++) {
                const __m256i q2 = _mm256_and_si256(_mm256_srl_epi16(GQ_R16H_LD(o + 288 + (8 * (ib >> 2) + j) * 64 + 32 * h), sh), m3);
                c[j] = _mm256_add_epi8(q2, _mm256_and_si256(GQ_R16H_BIT(GQ_R16H_LD(o + 1312 + 64 * j + 32 * h), 1 << ib), four));
            }
            const __m256 sv0 = _mm256_mul_ps(dv, GQ_R16H_SC8(o + 32 + 32 * ib + 8 * h)), sv1 = _mm256_mul_ps(dv, GQ_R16H_SC8(o + 48 + 32 * ib + 8 * h));
            GQ_R16H_ACC16(c, sv0, sv1, 4)
        }
    }
    GQ_R16H_STORE
}
/* the IQ4 values as the codes the products take: + 128 with VNNI, signed without */
static inline __m256i gq_r16h_iq4_tab(void) {
    const __m128i t = _mm_loadu_si128((const __m128i *)GGML_KVALUES_IQ4NL);
    return _mm256_broadcastsi128_si256(GQ_R16_BIGBIAS ? _mm_xor_si128(t, _mm_set1_epi8((char)0x80)) : t);
}
GQ_R16H_FN gq_r16h_iq4_xs(GQ_R16H_ARGS) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), tab = gq_r16h_iq4_tab();
    GQ_R16H_ZERO
    for (int b = 0; b < I / 256; b++) {
        const uint8_t *o = grp + (int64_t)b * 2208;
        const __m256 dv = GQ_R16H_D(o + 16 * h);
        for (int ib = 0; ib < 8; ib++) {
            const int g32 = 8 * b + ib;
            __m256i c[8];
            for (int q = 0; q < 4; q++) {
                const __m256i v = GQ_R16H_LD(o + 160 + (4 * ib + q) * 64 + 32 * h);
                c[q] = _mm256_shuffle_epi8(tab, _mm256_and_si256(v, m4));
                c[4 + q] = _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
            }
            const __m256 sv = _mm256_mul_ps(dv, GQ_R16H_SC8(o + 32 + 16 * ib + 8 * h));
            GQ_R16H_ACC32(c, sv, 0, sv, GQ_R16_BIGBIAS, GQ_R16H_DOTB)
        }
    }
    GQ_R16H_STORE
}
GQ_R16H_FN gq_r16h_q8_0(GQ_R16H_ARGS) {
    const __m256i flip = _mm256_set1_epi8(GQ_R16_BIGBIAS ? 0 : (char)0x80);   /* without VNNI: the bytes back to w */
    GQ_R16H_ZERO
    for (int b = 0; b < I / 32; b++) {
        const uint8_t *o = grp + (int64_t)b * 544; const int g32 = b;
        const __m256 dv = GQ_R16H_D(o + 16 * h);
        __m256i c[8];
        for (int j = 0; j < 8; j++) c[j] = _mm256_xor_si256(GQ_R16H_LD(o + 32 + 64 * j + 32 * h), flip);
        GQ_R16H_ACC32(c, dv, 0, dv, GQ_R16_BIGBIAS, GQ_R16H_DOTB)
    }
    GQ_R16H_STORE
}
GQ_R16H_FN gq_r16h_q4_0(GQ_R16H_ARGS) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    GQ_R16H_ZERO
    for (int b = 0; b < I / 32; b++) {
        const uint8_t *o = grp + (int64_t)b * 288; const int g32 = b;
        const __m256 dv = GQ_R16H_D(o + 16 * h);
        __m256i c[8];
        for (int q = 0; q < 4; q++) {
            const __m256i v = GQ_R16H_LD(o + 32 + 64 * q + 32 * h);
            c[q] = _mm256_and_si256(v, m4); c[4 + q] = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
        }
        GQ_R16H_ACC32(c, dv, 0, dv, 8, GQ_R16H_DOTU)
    }
    GQ_R16H_STORE
}
GQ_R16H_FN gq_r16h_iq4_nl(GQ_R16H_ARGS) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), tab = gq_r16h_iq4_tab();
    GQ_R16H_ZERO
    for (int b = 0; b < I / 32; b++) {
        const uint8_t *o = grp + (int64_t)b * 288; const int g32 = b;
        const __m256 dv = GQ_R16H_D(o + 16 * h);
        __m256i c[8];
        for (int q = 0; q < 4; q++) {
            const __m256i v = GQ_R16H_LD(o + 32 + 64 * q + 32 * h);
            c[q] = _mm256_shuffle_epi8(tab, _mm256_and_si256(v, m4)); c[4 + q] = _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
        }
        GQ_R16H_ACC32(c, dv, 0, dv, GQ_R16_BIGBIAS, GQ_R16H_DOTB)
    }
    GQ_R16H_STORE
}
#define GQ_R16H_N(FN) do { switch (n) { \
        case 1: FN(ys, ystride, grp, xq, dx, xs, I, 1, h); break; case 2: FN(ys, ystride, grp, xq, dx, xs, I, 2, h); break; \
        case 3: FN(ys, ystride, grp, xq, dx, xs, I, 3, h); break; default: FN(ys, ystride, grp, xq, dx, xs, I, 4, h); break; } } while (0)
static void gq_r16_rows(int t, float *y, int ystride, const uint8_t *grp, const gq_act *a, int I) {
    for (int s0 = 0; s0 < a->S; s0 += 4) {
        const int n = a->S - s0 < 4 ? a->S - s0 : 4;
        float *ys = y + (int64_t)s0 * ystride; const int8_t *xq = a->q + (int64_t)s0 * I;
        const float *dx = a->d + (int64_t)s0 * (I / 32); const int32_t *xs = a->s + (int64_t)s0 * (I / 16);
        for (int h = 0; h < 2; h++)
            switch (t) {
            case GQ_Q2_0_R16: GQ_R16H_N(gq_r16h_q2_0); break;
            case GQ_Q4_K_R16: GQ_R16H_N(gq_r16h_q4_k); break;
            case GQ_Q5_K_R16: GQ_R16H_N(gq_r16h_q5_k); break;
            case GQ_Q6_K_R16: GQ_R16H_N(gq_r16h_q6_k); break;
            case GQ_Q3_K_R16: GQ_R16H_N(gq_r16h_q3_k); break;
            case GQ_IQ4_XS_R16: GQ_R16H_N(gq_r16h_iq4_xs); break;
            case GQ_Q8_0_R16: GQ_R16H_N(gq_r16h_q8_0); break;
            case GQ_Q4_0_R16: GQ_R16H_N(gq_r16h_q4_0); break;
            case GQ_IQ4_NL_R16: GQ_R16H_N(gq_r16h_iq4_nl); break;
            }
    }
}
#endif
#if defined(GQ_VNNI512) || defined(GQ_R16_AVX2)
#define GQ_R16_ON 1   /* the interleaved forms have kernels in this build: the engine may make them */
#endif
/* ---- File rows eight at a time (gq_rows_f8) ---------------------------------------
 * The rows of a matrix as the file holds them (a host pool's experts, the engines' rows that
 * are not interleaved), eight a call, with the R16 kernels' numerics: per row and per 32
 * elements (16 where the type's scale is per 16) the exact integer dot, then its scale, the
 * same float steps in the same order. Where a type has an R16 form the results are its bits
 * (tests/test_rows_f8.c packs and compares); a row's result does not depend on S or on the
 * other rows of its call. Two ways in:
 *  - IQ4_XS and the K-quants transpose the eight rows' raw bytes in registers (32 bytes, two
 *    sub-blocks a time) into the half-group codes R16 keeps, and take every row's scales as
 *    the lanes of one vector: the unpack work of one row spread over eight.
 *  - the grid types (IQ2, IQ3) decode a row's sub-block as their row kernels do, with the
 *    signs moved onto the activation (|w| is the grid's byte; exact, x is in -127..127), and
 *    sum the eight rows' int32 lanes in a tree of horizontal adds.
 * One core of an i5-13500 and of a Zen 4 (8700GE), x86-64-v3, 512 x 2048 in cache, GB/s of
 * weights, row kernels -> these: IQ3_XXS 3.0 -> 6.9 and 3.5 -> 4.5, IQ3_S 5.5 -> 7.6 and
 * 4.4 -> 5.2, IQ2_S 3.3 -> 5.5 and 3.3 -> 4.8, IQ4_XS 12.2 -> 18.1 and 14.5 -> 23.1, Q4_K
 * 10.2 -> 24.2 and 13.2 -> 27.8, Q6_K 11.5 -> 25.2 and 14.0 -> 24.8. The grid's rows are
 * gathered for IQ3 (faster on the Intel core, even on the Zen 4) and loaded for IQ2 (the
 * gather was slower on the Zen 4). */
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define GQ_F8_ON 1
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define GQ_F8_VNNI 1
#define GQ_F8_DPU(acc, u, x) _mm256_dpbusd_epi32((acc), (u), (x))
#elif defined(__AVXVNNI__)
#define GQ_F8_VNNI 1
#define GQ_F8_DPU(acc, u, x) _mm256_dpbusd_avx_epi32((acc), (u), (x))
#else
#define GQ_F8_VNNI 0
#define GQ_F8_DPU(acc, u, x) _mm256_add_epi32((acc), _mm256_madd_epi16(_mm256_maddubs_epi16((u), (x)), _mm256_set1_epi16(1)))
#endif
#define GQ_F8_DPS(acc, w, x) _mm256_add_epi32((acc), _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8((x), (w))), _mm256_set1_epi16(1)))
/* Q8_0 and the IQ4 lookups: + 128 with VNNI (vpdpbusd does not saturate), signed without */
#if GQ_F8_VNNI
#define GQ_F8_BIG(acc, c, x) GQ_F8_DPU(acc, c, x)
#define GQ_F8_BB 128
#else
#define GQ_F8_BIG(acc, c, x) GQ_F8_DPS(acc, c, x)
#define GQ_F8_BB 0
#endif
#define GQ_F8_X4(xb, j) _mm256_set1_epi32(gq_ld32((xb) + 4 * (j)))
/* the int32 sums of quads j0 .. j0 + n - 1 of the codes c (lane r = row r), as R16's: with
 * VNNI a vpdpbusd each; without, two maddubs added in 16 bits (codes up to 63) before a madd */
#if GQ_F8_VNNI
#define GQ_F8_DOTU(p, c, xb, j0, n) { p = GQ_F8_DPU(_mm256_setzero_si256(), (c)[j0], GQ_F8_X4(xb, j0)); \
        for (int k_ = (j0) + 1; k_ < (j0) + (n); k_++) p = GQ_F8_DPU(p, (c)[k_], GQ_F8_X4(xb, k_)); }
#else
#define GQ_F8_DOTU(p, c, xb, j0, n) { const __m256i o16_ = _mm256_set1_epi16(1); p = _mm256_setzero_si256(); \
        for (int k_ = (j0); k_ < (j0) + (n); k_ += 2) \
            p = _mm256_add_epi32(p, _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16((c)[k_], GQ_F8_X4(xb, k_)), \
                                                                       _mm256_maddubs_epi16((c)[k_ + 1], GQ_F8_X4(xb, k_ + 1))), o16_)); }
#endif
#define GQ_F8_DOTB(p, c, xb, j0, n) { p = GQ_F8_BIG(_mm256_setzero_si256(), (c)[j0], GQ_F8_X4(xb, j0)); \
        for (int k_ = (j0) + 1; k_ < (j0) + (n); k_++) p = GQ_F8_BIG(p, (c)[k_], GQ_F8_X4(xb, k_)); }
/* GQ_R16H_ACC32 and GQ_R16H_ACC16, lane by lane: the 32 elements at g32, sv the rows'
 * scales, nv their mins (MIN), codes = w + BIAS */
#define GQ_F8_ACC32(c, sv, MIN, nv, BIAS, DOT) \
    for (int s = 0; s < S; s++) { \
        const int8_t *xb_ = xq + (int64_t)s * I + 32 * (int64_t)g32; \
        __m256i p_; DOT(p_, c, xb_, 0, 8) \
        const int32_t *x16_ = xs + (int64_t)s * (I / 16) + 2 * (int64_t)g32; const int32_t x32_ = x16_[0] + x16_[1]; \
        const float d_ = dx[(int64_t)s * (I / 32) + g32]; \
        if ((BIAS) != 0) p_ = _mm256_sub_epi32(p_, _mm256_set1_epi32((BIAS) * x32_)); \
        acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p_), _mm256_mul_ps((sv), _mm256_set1_ps(d_)), acc[s]); \
        if (MIN) acc[s] = _mm256_fnmadd_ps((nv), _mm256_set1_ps(d_ * (float)x32_), acc[s]); \
    }
#define GQ_F8_ACC16(c, sv0, sv1, BIAS, DOT) \
    for (int s = 0; s < S; s++) { \
        const int8_t *xb_ = xq + (int64_t)s * I + 32 * (int64_t)g32; \
        __m256i p0_, p1_; DOT(p0_, c, xb_, 0, 4) DOT(p1_, c, xb_, 4, 4) \
        const int32_t *x16_ = xs + (int64_t)s * (I / 16) + 2 * (int64_t)g32; \
        p0_ = _mm256_sub_epi32(p0_, _mm256_set1_epi32((BIAS) * x16_[0])); p1_ = _mm256_sub_epi32(p1_, _mm256_set1_epi32((BIAS) * x16_[1])); \
        const __m256 t_ = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1_), (sv1), _mm256_mul_ps(_mm256_cvtepi32_ps(p0_), (sv0))); \
        acc[s] = _mm256_fmadd_ps(t_, _mm256_set1_ps(dx[(int64_t)s * (I / 32) + g32]), acc[s]); \
    }
#define GQ_F8_ARGS float *y, int ystride, const uint8_t *row, int64_t rb, const int8_t *xq, const float *dx, const int32_t *xs, int I, const int S
#define GQ_F8_ZERO __m256 acc[4]; for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
#define GQ_F8_STORE for (int s = 0; s < S; s++) _mm256_storeu_ps(y + (int64_t)s * ystride, acc[s]);
static inline int64_t gq_ld64(const void *p) { int64_t v; memcpy(&v, p, 8); return v; }
/* c[j] lane r = dword j of w[r]: the 8 x 8 transpose of 32-bit elements */
static inline __attribute__((always_inline)) void gq_f8_t8(__m256i *c, const __m256i *w) {
    const __m256i t0 = _mm256_unpacklo_epi32(w[0], w[1]), t1 = _mm256_unpackhi_epi32(w[0], w[1]);
    const __m256i t2 = _mm256_unpacklo_epi32(w[2], w[3]), t3 = _mm256_unpackhi_epi32(w[2], w[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(w[4], w[5]), t5 = _mm256_unpackhi_epi32(w[4], w[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(w[6], w[7]), t7 = _mm256_unpackhi_epi32(w[6], w[7]);
    const __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    c[0] = _mm256_permute2x128_si256(u0, u4, 0x20); c[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    c[1] = _mm256_permute2x128_si256(u1, u5, 0x20); c[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    c[2] = _mm256_permute2x128_si256(u2, u6, 0x20); c[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    c[3] = _mm256_permute2x128_si256(u3, u7, 0x20); c[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}
/* the eight rows' 32 bytes at off, transposed: T[j] lane r = row r's bytes 4j .. 4j + 3 */
static inline __attribute__((always_inline)) void gq_f8_raw(__m256i *T, const uint8_t *const *b, int off) {
    __m256i w[8];
    for (int r = 0; r < 8; r++) w[r] = _mm256_loadu_si256((const __m256i *)(b[r] + off));
    gq_f8_t8(T, w);
}
/* the eight rows' 32-bit words, and f16 values, at off: lane r = row r */
static inline __attribute__((always_inline)) __m256i gq_f8_w32(const uint8_t *const *b, int off) {
    return _mm256_setr_epi32(gq_ld32(b[0] + off), gq_ld32(b[1] + off), gq_ld32(b[2] + off), gq_ld32(b[3] + off),
                             gq_ld32(b[4] + off), gq_ld32(b[5] + off), gq_ld32(b[6] + off), gq_ld32(b[7] + off));
}
static inline __attribute__((always_inline)) __m256 gq_f8_h16(const uint8_t *const *b, int off) {
    return _mm256_cvtph_ps(_mm_setr_epi16((short)gq_rd16(b[0] + off), (short)gq_rd16(b[1] + off), (short)gq_rd16(b[2] + off), (short)gq_rd16(b[3] + off),
                                          (short)gq_rd16(b[4] + off), (short)gq_rd16(b[5] + off), (short)gq_rd16(b[6] + off), (short)gq_rd16(b[7] + off)));
}
#define GQ_F8_B32(V, k) _mm256_and_si256(_mm256_srli_epi32((V), 8 * (k)), _mm256_set1_epi32(0xFF))   /* byte k of each lane */
#define GQ_F8_SB8(V, k) _mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_slli_epi32((V), 24 - 8 * (k)), 24))   /* the signed byte k, as a float */
#define GQ_F8_BIT(v, bit) _mm256_cmpeq_epi8(_mm256_and_si256((v), _mm256_set1_epi8((char)(bit))), _mm256_set1_epi8((char)(bit)))
/* IQ4_XS: per block the rows' d and 6-bit scales as lanes (the scale d * (ls - 32), gq_sb_iq4_xs's
 * product), then two sub-blocks' 32 bytes transposed at once, looked up as R16's codes */
GQ_SB_FN void gq_f8_iq4_xs(GQ_F8_ARGS) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m128i t0 = _mm_loadu_si128((const __m128i *)GGML_KVALUES_IQ4NL);
    const __m256i tab = _mm256_broadcastsi128_si256(GQ_F8_BB ? _mm_xor_si128(t0, _mm_set1_epi8((char)0x80)) : t0);
    GQ_F8_ZERO
    for (int64_t blk = 0; blk < I / 256; blk++) {
        const uint8_t *b[8];
        for (int r = 0; r < 8; r++) b[r] = row + r * rb + blk * 136;
        const __m256 D = gq_f8_h16(b, 0);
        const __m256i H = gq_f8_w32(b, 0), L = gq_f8_w32(b, 4);
        for (int k = 0; k < 4; k++) {
            __m256i T[8]; gq_f8_raw(T, b, 8 + 32 * k);
            for (int hf = 0; hf < 2; hf++) {
                const int ib = 2 * k + hf; const int64_t g32 = blk * 8 + ib;
                const __m256i ls = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(L, 8 * k + 4 * hf), _mm256_set1_epi32(15)),
                                                   _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(H, 16 + 2 * ib), _mm256_set1_epi32(3)), 4));
                const __m256 sv = _mm256_mul_ps(D, _mm256_cvtepi32_ps(_mm256_sub_epi32(ls, _mm256_set1_epi32(32))));
                __m256i c[8];
                for (int q = 0; q < 4; q++) {
                    const __m256i v = T[4 * hf + q];
                    c[q] = _mm256_shuffle_epi8(tab, _mm256_and_si256(v, m4));
                    c[4 + q] = _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
                }
                GQ_F8_ACC32(c, sv, 0, sv, GQ_F8_BB, GQ_F8_DOTB)
            }
        }
    }
    GQ_F8_STORE
}
/* Q4_K and Q5_K: a sub-block pair's 32 bytes (low nibbles the first, high the second), Q5_K's
 * high bits transposed once a block; scales and mins from the 12 bytes as lanes
 * (gq_scale_min_k4 on each lane) */
GQ_SB_FN void gq_f8_q45_k(GQ_F8_ARGS, const int five) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), sixteen = _mm256_set1_epi8(16), m63 = _mm256_set1_epi32(63), m15 = _mm256_set1_epi32(15), m30 = _mm256_set1_epi32(0x30);
    const int BYTES = five ? 176 : 144, qo = five ? 48 : 16;
    GQ_F8_ZERO
    for (int64_t blk = 0; blk < I / 256; blk++) {
        const uint8_t *b[8];
        for (int r = 0; r < 8; r++) b[r] = row + r * rb + blk * BYTES;
        const __m256 D = gq_f8_h16(b, 0), DM = gq_f8_h16(b, 2);
        const __m256i S0 = gq_f8_w32(b, 4), S1 = gq_f8_w32(b, 8), S2 = gq_f8_w32(b, 12);
        __m256i QH[8];
        if (five) gq_f8_raw(QH, b, 16);
        for (int c2 = 0; c2 < 4; c2++) {
            __m256i T[8]; gq_f8_raw(T, b, qo + 32 * c2);
            for (int hi = 0; hi < 2; hi++) {
                const int ib = 2 * c2 + hi; const int64_t g32 = blk * 8 + ib;
                __m256i sc, mn;
                if (ib < 4) { sc = _mm256_and_si256(_mm256_srli_epi32(S0, 8 * ib), m63); mn = _mm256_and_si256(_mm256_srli_epi32(S1, 8 * ib), m63); }
                else {
                    const int j = ib - 4;
                    sc = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(S2, 8 * j), m15), _mm256_and_si256(_mm256_srli_epi32(S0, 8 * j + 2), m30));
                    mn = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(S2, 8 * j + 4), m15), _mm256_and_si256(_mm256_srli_epi32(S1, 8 * j + 2), m30));
                }
                const __m256 sv = _mm256_mul_ps(D, _mm256_cvtepi32_ps(sc)), nv = _mm256_mul_ps(DM, _mm256_cvtepi32_ps(mn));
                __m256i c[8];
                for (int j = 0; j < 8; j++) {
                    c[j] = _mm256_and_si256(hi ? _mm256_srli_epi16(T[j], 4) : T[j], m4);
                    if (five) c[j] = _mm256_add_epi8(c[j], _mm256_and_si256(GQ_F8_BIT(QH[j], 1 << ib), sixteen));
                }
                GQ_F8_ACC32(c, sv, 1, nv, 0, GQ_F8_DOTU)
            }
        }
    }
    GQ_F8_STORE
}
/* Q6_K: per half of the block, ql's two 32-byte runs and qh's one transposed (codes = value + 32) */
GQ_SB_FN void gq_f8_q6_k(GQ_F8_ARGS) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m30 = _mm256_set1_epi8(0x30);
    GQ_F8_ZERO
    for (int64_t blk = 0; blk < I / 256; blk++) {
        const uint8_t *b[8];
        for (int r = 0; r < 8; r++) b[r] = row + r * rb + blk * 210;
        const __m256 D = gq_f8_h16(b, 208);
        __m256i SC[4];
        for (int i = 0; i < 4; i++) SC[i] = gq_f8_w32(b, 192 + 4 * i);
        for (int cc = 0; cc < 2; cc++) {
            __m256i L0[8], L1[8], HH[8];
            gq_f8_raw(L0, b, 64 * cc); gq_f8_raw(L1, b, 64 * cc + 32); gq_f8_raw(HH, b, 128 + 32 * cc);
            for (int k = 0; k < 4; k++) {
                const int ib = 4 * cc + k; const int64_t g32 = blk * 8 + ib;
                __m256i c[8];
                for (int j = 0; j < 8; j++) {
                    const __m256i l = (k & 1) ? L1[j] : L0[j];
                    const __m256i lo = _mm256_and_si256(k & 2 ? _mm256_srli_epi16(l, 4) : l, m4);
                    const __m256i hv = HH[j];
                    const __m256i hh = k == 0 ? _mm256_slli_epi16(hv, 4) : k == 1 ? _mm256_slli_epi16(hv, 2) : k == 2 ? hv : _mm256_srli_epi16(hv, 2);
                    c[j] = _mm256_or_si256(lo, _mm256_and_si256(hh, m30));
                }
                const __m256i SCd = SC[ib >> 1];
                const __m256 sv0 = _mm256_mul_ps(D, GQ_F8_SB8(SCd, 2 * (ib & 1))), sv1 = _mm256_mul_ps(D, GQ_F8_SB8(SCd, 2 * (ib & 1) + 1));
                GQ_F8_ACC16(c, sv0, sv1, 32, GQ_F8_DOTU)
            }
        }
    }
    GQ_F8_STORE
}
/* Q3_K: hmask transposed once a block, qs per half (codes = value + 4) */
GQ_SB_FN void gq_f8_q3_k(GQ_F8_ARGS) {
    const __m256i m3 = _mm256_set1_epi8(3), four = _mm256_set1_epi8(4), m15 = _mm256_set1_epi32(15), m3d = _mm256_set1_epi32(3), k32 = _mm256_set1_epi32(32);
    GQ_F8_ZERO
    for (int64_t blk = 0; blk < I / 256; blk++) {
        const uint8_t *b[8];
        for (int r = 0; r < 8; r++) b[r] = row + r * rb + blk * 110;
        const __m256 D = gq_f8_h16(b, 108);
        const __m256i SA = gq_f8_w32(b, 96), SB = gq_f8_w32(b, 100), SCC = gq_f8_w32(b, 104);
        __m256i HM[8]; gq_f8_raw(HM, b, 0);
        for (int cq = 0; cq < 2; cq++) {
            __m256i Q[8]; gq_f8_raw(Q, b, 32 + 32 * cq);
            for (int sh = 0; sh < 4; sh++) {
                const int ib = 4 * cq + sh; const int64_t g32 = blk * 8 + ib;
                __m256i c[8];
                for (int j = 0; j < 8; j++)
                    c[j] = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(Q[j], 2 * sh), m3), _mm256_and_si256(GQ_F8_BIT(HM[j], 1 << ib), four));
                __m256 sv[2];
                for (int h = 0; h < 2; h++) {
                    const int k = 2 * ib + h, m = k < 8 ? k : k - 8;
                    const __m256i by = GQ_F8_B32(m < 4 ? SA : SB, m & 3);
                    const __m256i l = k < 8 ? _mm256_and_si256(by, m15) : _mm256_srli_epi32(by, 4);
                    const __m256i hh = _mm256_and_si256(_mm256_srli_epi32(GQ_F8_B32(SCC, k & 3), 2 * (k >> 2)), m3d);
                    sv[h] = _mm256_mul_ps(D, _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_or_si256(l, _mm256_slli_epi32(hh, 4)), k32)));
                }
                GQ_F8_ACC16(c, sv[0], sv[1], 4, GQ_F8_DOTU)
            }
        }
    }
    GQ_F8_STORE
}
/* Q8_0: d, then 32 signed bytes */
GQ_SB_FN void gq_f8_q8_0(GQ_F8_ARGS) {
    const __m256i flip = _mm256_set1_epi8(GQ_F8_BB ? (char)0x80 : 0);
    GQ_F8_ZERO
    for (int64_t g32 = 0; g32 < I / 32; g32++) {
        __m256i w[8], c[8];
        const uint8_t *b[8];
        for (int r = 0; r < 8; r++) { b[r] = row + r * rb + g32 * 34; w[r] = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(b[r] + 2)), flip); }
        gq_f8_t8(c, w);
        const __m256 sv = gq_f8_h16(b, 0);
        GQ_F8_ACC32(c, sv, 0, sv, GQ_F8_BB, GQ_F8_DOTB)
    }
    GQ_F8_STORE
}
/* The grid types row by row: |w| the grid's bytes, the signs on x as +1 / -1 rows. */
#if GQ_F8_VNNI
#define GQ_F8_DPR(u, x) GQ_F8_DPU(_mm256_setzero_si256(), (u), (x))
#else
#define GQ_F8_DPR(u, x) _mm256_madd_epi16(_mm256_maddubs_epi16((u), (x)), _mm256_set1_epi16(1))
#endif
#define GQ_F8_KS(i) gq_ld64(GGML_KSIGNS_PM1 + 8 * (i))
/* lane r of the halves: the sums of p[r]'s lanes 0-3 (lo) and 4-7 (hi) */
static inline __attribute__((always_inline)) void gq_f8_hsum8x2(const __m256i *p, __m256i *lo, __m256i *hi) {
    const __m256i s01 = _mm256_hadd_epi32(p[0], p[1]), s23 = _mm256_hadd_epi32(p[2], p[3]);
    const __m256i s45 = _mm256_hadd_epi32(p[4], p[5]), s67 = _mm256_hadd_epi32(p[6], p[7]);
    const __m256i t0 = _mm256_hadd_epi32(s01, s23), t1 = _mm256_hadd_epi32(s45, s67);
    *lo = _mm256_permute2x128_si256(t0, t1, 0x20); *hi = _mm256_permute2x128_si256(t0, t1, 0x31);
}
/* four sign bytes (bit j of byte k: element 8k + j negative) as +1 / -1 */
static inline __attribute__((always_inline)) __m256i gq_f8_pm(uint32_t bits) {
    const __m256i pick = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), pick), bit), bit);
    return _mm256_or_si256(m, _mm256_set1_epi8(1));
}
/* a sub-block's grid bytes (the return) and signs (*sg), as gq_sb_* builds the values */
GQ_SB_FN __m256i gq_ug_iq3_xxs(const uint8_t *b, int ib, __m256i *sg) {
    const uint8_t *q = b + 2 + 8 * ib; const uint32_t a = gq_rd32(b + 66 + 4 * ib);
    *sg = _mm256_setr_epi64x(GQ_F8_KS(a & 127), GQ_F8_KS((a >> 7) & 127), GQ_F8_KS((a >> 14) & 127), GQ_F8_KS((a >> 21) & 127));
    return _mm256_i32gather_epi32((const int *)GGML_IQ3XXS_GRID, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)q)), 4);
}
GQ_SB_FN __m256i gq_ug_iq3_s(const uint8_t *b, int ib, __m256i *sg) {
    const uint8_t *qs = b + 2, *qh = b + 66, *sgn = b + 74;
    const __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 8 * ib)));
    const __m256i hi = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(qh[ib]), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)), _mm256_set1_epi32(1));
    *sg = gq_f8_pm(gq_rd32(sgn + 4 * ib));
    return gq_gather4x8(GGML_IQ3S_GRID, _mm256_or_si256(lo, _mm256_slli_epi32(hi, 8)));
}
GQ_SB_FN __m256i gq_ug_iq2_xxs(const uint8_t *b, int ib, __m256i *sg) {
    const uint8_t *qs = b + 2, *G = GGML_IQ2XXS_GRID;
    const uint32_t a0 = gq_rd32(qs + 8 * ib), a1 = gq_rd32(qs + 8 * ib + 4);
    *sg = _mm256_setr_epi64x(GQ_F8_KS(a1 & 127), GQ_F8_KS((a1 >> 7) & 127), GQ_F8_KS((a1 >> 14) & 127), GQ_F8_KS((a1 >> 21) & 127));
    return _mm256_setr_epi64x(gq_ld64(G + 8 * (a0 & 0xFF)), gq_ld64(G + 8 * ((a0 >> 8) & 0xFF)), gq_ld64(G + 8 * ((a0 >> 16) & 0xFF)), gq_ld64(G + 8 * (a0 >> 24)));
}
GQ_SB_FN __m256i gq_ug_iq2_xs(const uint8_t *b, int ib, __m256i *sg) {
    const uint8_t *qs = b + 2, *G = GGML_IQ2XS_GRID;
    const uint16_t q0 = gq_rd16(qs + 8 * ib), q1 = gq_rd16(qs + 8 * ib + 2), q2 = gq_rd16(qs + 8 * ib + 4), q3 = gq_rd16(qs + 8 * ib + 6);
    *sg = _mm256_setr_epi64x(GQ_F8_KS(q0 >> 9), GQ_F8_KS(q1 >> 9), GQ_F8_KS(q2 >> 9), GQ_F8_KS(q3 >> 9));
    return _mm256_setr_epi64x(gq_ld64(G + 8 * (q0 & 511)), gq_ld64(G + 8 * (q1 & 511)), gq_ld64(G + 8 * (q2 & 511)), gq_ld64(G + 8 * (q3 & 511)));
}
GQ_SB_FN __m256i gq_ug_iq2_s(const uint8_t *b, int ib, __m256i *sg) {
    const uint8_t *qs = b + 2, *sgn = b + 34, *qh = b + 66;
    const __m128i lo = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)gq_rd32(qs + 4 * ib)));
    const __m128i hi = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(qh[ib]), _mm_setr_epi32(0, 2, 4, 6)), _mm_set1_epi32(3));
    *sg = gq_f8_pm(gq_rd32(sgn + 4 * ib));
    return gq_gather8x4(GGML_IQ2S_GRID, _mm_or_si128(lo, _mm_slli_epi32(hi, 8)));
}
/* the eight rows' scales of sub-block ib, gq_sb_*'s products on lanes (sv1 the second 16's) */
#define GQ_F8_SV static inline __attribute__((always_inline)) void
GQ_F8_SV gq_sv_iq3_xxs(const uint8_t *const *b, __m256 D, int ib, __m256 *sv0, __m256 *sv1) {
    const __m256 ls = _mm256_cvtepi32_ps(_mm256_srli_epi32(gq_f8_w32(b, 66 + 4 * ib), 28));
    *sv0 = *sv1 = _mm256_mul_ps(_mm256_mul_ps(D, _mm256_add_ps(_mm256_set1_ps(0.5f), ls)), _mm256_set1_ps(0.5f));
}
GQ_F8_SV gq_sv_iq3_s(const uint8_t *const *b, __m256 D, int ib, __m256 *sv0, __m256 *sv1) {
    const __m256i v = _mm256_and_si256(_mm256_srli_epi32(gq_f8_w32(b, 106), 8 * (ib >> 1) + 4 * (ib & 1)), _mm256_set1_epi32(15));
    *sv0 = *sv1 = _mm256_mul_ps(D, _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(1), _mm256_slli_epi32(v, 1))));
}
GQ_F8_SV gq_sv_iq2_xxs(const uint8_t *const *b, __m256 D, int ib, __m256 *sv0, __m256 *sv1) {
    const __m256 ls = _mm256_cvtepi32_ps(_mm256_srli_epi32(gq_f8_w32(b, 2 + 8 * ib + 4), 28));
    *sv0 = *sv1 = _mm256_mul_ps(_mm256_mul_ps(D, _mm256_add_ps(_mm256_set1_ps(0.5f), ls)), _mm256_set1_ps(0.25f));
}
/* IQ2_XS and IQ2_S: a nibble per 16 at sc (byte ib, low then high) */
GQ_F8_SV gq_sv_iq2_n(const uint8_t *const *b, int sc, __m256 D, int ib, __m256 *sv0, __m256 *sv1) {
    const __m256i by = GQ_F8_B32(gq_f8_w32(b, sc + 4 * (ib >> 2)), ib & 3);
    *sv0 = _mm256_mul_ps(_mm256_mul_ps(D, _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_cvtepi32_ps(_mm256_and_si256(by, _mm256_set1_epi32(15))))), _mm256_set1_ps(0.25f));
    *sv1 = _mm256_mul_ps(_mm256_mul_ps(D, _mm256_add_ps(_mm256_set1_ps(0.5f), _mm256_cvtepi32_ps(_mm256_srli_epi32(by, 4)))), _mm256_set1_ps(0.25f));
}
GQ_F8_SV gq_sv_iq2_xs(const uint8_t *const *b, __m256 D, int ib, __m256 *sv0, __m256 *sv1) { gq_sv_iq2_n(b, 66, D, ib, sv0, sv1); }
GQ_F8_SV gq_sv_iq2_s(const uint8_t *const *b, __m256 D, int ib, __m256 *sv0, __m256 *sv1) { gq_sv_iq2_n(b, 74, D, ib, sv0, sv1); }
#define GQ_F8R(NAME, BYTES, UG, SV, PER16) \
GQ_SB_FN void NAME(GQ_F8_ARGS) { \
    GQ_F8_ZERO \
    for (int64_t blk = 0; blk < I / 256; blk++) { \
        const uint8_t *b[8]; \
        for (int r = 0; r < 8; r++) b[r] = row + r * rb + blk * (BYTES); \
        const __m256 D = gq_f8_h16(b, 0); \
        for (int ib = 0; ib < 8; ib++) { \
            const int64_t g32 = blk * 8 + ib; __m256i u[8], sg[8]; __m256 sv0, sv1; \
            for (int r = 0; r < 8; r++) u[r] = UG(b[r], ib, &sg[r]); \
            SV(b, D, ib, &sv0, &sv1); \
            for (int s = 0; s < S; s++) { \
                const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (int64_t)s * I + 32 * g32)); \
                __m256i p[8], P0, P1; \
                for (int r = 0; r < 8; r++) p[r] = GQ_F8_DPR(u[r], _mm256_sign_epi8(x, sg[r])); \
                gq_f8_hsum8x2(p, &P0, &P1); \
                const float d_ = dx[(int64_t)s * (I / 32) + g32]; \
                if (!(PER16)) acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(P0, P1)), _mm256_mul_ps(sv0, _mm256_set1_ps(d_)), acc[s]); \
                else { const __m256 t_ = _mm256_fmadd_ps(_mm256_cvtepi32_ps(P1), sv1, _mm256_mul_ps(_mm256_cvtepi32_ps(P0), sv0)); \
                       acc[s] = _mm256_fmadd_ps(t_, _mm256_set1_ps(d_), acc[s]); } \
            } \
        } \
    } \
    GQ_F8_STORE \
}
GQ_F8R(gq_f8_iq3_xxs, 98, gq_ug_iq3_xxs, gq_sv_iq3_xxs, 0)
GQ_F8R(gq_f8_iq3_s, 110, gq_ug_iq3_s, gq_sv_iq3_s, 0)
GQ_F8R(gq_f8_iq2_xxs, 66, gq_ug_iq2_xxs, gq_sv_iq2_xxs, 0)
GQ_F8R(gq_f8_iq2_xs, 74, gq_ug_iq2_xs, gq_sv_iq2_xs, 1)
GQ_F8R(gq_f8_iq2_s, 82, gq_ug_iq2_s, gq_sv_iq2_s, 1)
#define GQ_F8_N(CALL) do { switch (n) { case 1: { const int S = 1; CALL; } break; case 2: { const int S = 2; CALL; } break; \
        case 3: { const int S = 3; CALL; } break; default: { const int S = 4; CALL; } break; } } while (0)
#endif
/* the types and widths gq_rows_f8 takes (none in GQ_EXACT=1 mode: the float reference) */
static inline int gq_f8_ok(int t, int I) {
#ifdef GQ_F8_ON
    if (gq_exact_mode()) return 0;
    switch (t) {
    case GQ_IQ4_XS: case GQ_IQ2_XXS: case GQ_IQ2_XS: case GQ_IQ2_S: case GQ_IQ3_XXS: case GQ_IQ3_S:
    case GQ_Q4_K: case GQ_Q5_K: case GQ_Q6_K: case GQ_Q3_K: return I % 256 == 0;
    case GQ_Q8_0: return I % 32 == 0;
    default: return 0;
    }
#else
    (void)t; (void)I; return 0;
#endif
}
/* rows [0, R) of type t at row, rb bytes apart, against every activation row of a:
 * y[s * ystride + r]. R a multiple of 8, gq_f8_ok(t, I). */
static void gq_rows_f8(float *y, int ystride, const gq_act *a, const uint8_t *row, int64_t rb, int t, int I, int R) {
#ifdef GQ_F8_ON
    for (int r0 = 0; r0 + 8 <= R; r0 += 8)
        for (int s0 = 0; s0 < a->S; s0 += 4) {
            const int n = a->S - s0 < 4 ? a->S - s0 : 4;
            float *ys = y + (int64_t)s0 * ystride + r0; const uint8_t *rw = row + (int64_t)r0 * rb;
            const int8_t *xq = a->q + (int64_t)s0 * I; const float *dx = a->d + (int64_t)s0 * (I / 32); const int32_t *xs = a->s + (int64_t)s0 * (I / 16);
            switch (t) {
            case GQ_IQ4_XS: GQ_F8_N(gq_f8_iq4_xs(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            case GQ_IQ2_XXS: GQ_F8_N(gq_f8_iq2_xxs(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            case GQ_IQ2_XS: GQ_F8_N(gq_f8_iq2_xs(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            case GQ_IQ2_S: GQ_F8_N(gq_f8_iq2_s(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            case GQ_IQ3_XXS: GQ_F8_N(gq_f8_iq3_xxs(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            case GQ_IQ3_S: GQ_F8_N(gq_f8_iq3_s(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            case GQ_Q4_K: GQ_F8_N(gq_f8_q45_k(ys, ystride, rw, rb, xq, dx, xs, I, S, 0)); break;
            case GQ_Q5_K: GQ_F8_N(gq_f8_q45_k(ys, ystride, rw, rb, xq, dx, xs, I, S, 1)); break;
            case GQ_Q6_K: GQ_F8_N(gq_f8_q6_k(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            case GQ_Q3_K: GQ_F8_N(gq_f8_q3_k(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            case GQ_Q8_0: GQ_F8_N(gq_f8_q8_0(ys, ystride, rw, rb, xq, dx, xs, I, S)); break;
            }
        }
#else
    (void)y; (void)ystride; (void)a; (void)row; (void)rb; (void)t; (void)I; (void)R;
#endif
}
static inline int gq_int_ok(int t, int I) {
    const int b = gq_r16_base(t);
    if (b >= 0) return I % GQ_TYPES[b].blck == 0;   /* made only where its kernels are (GQ_R16_ON) */
    return !gq_exact_mode() && gq_has_int(t) && I % 32 == 0;
}

/* y = W x on an already quantized activation block (types with an integer form). An
 * interleaved W holds whole groups of 16 rows; O may stop inside the last one (a caller
 * reading the first O rows of a matrix packed whole, an unpadded vocabulary): that group
 * goes through a buffer of its own. */
static void gq_matmul_act(float *y, const gq_act *a, const uint8_t *W, int t, int O) {
    const int I = a->I; const int64_t rb = gq_row_bytes(t, I);
#ifdef GQ_R16_ON
    if (gq_r16_base(t) >= 0) {
        const int G = O / 16, tail = O % 16;
        #pragma omp parallel for schedule(static)
        for (int g = 0; g < G; g++) gq_r16_rows(t, y + 16 * g, O, W + (int64_t)16 * g * rb, a, I);
        if (tail) {
            float *y16 = (float *)malloc((size_t)a->S * 16 * sizeof(float));
            if (!y16) abort();
            gq_r16_rows(t, y16, 16, W + (int64_t)16 * G * rb, a, I);
            for (int r = 0; r < a->S; r++) memcpy(y + (int64_t)r * O + 16 * G, y16 + 16 * r, (size_t)tail * sizeof(float));
            free(y16);
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) gq_row_all(y + o, O, a, W + (int64_t)o * rb, t, I);
}
/* Two matrices on the same activations in one parallel region (gate and up). */
static void gq_matmul2_act(float *y1, const uint8_t *W1, int t1, float *y2, const uint8_t *W2, int t2, const gq_act *a, int O) {
    const int I = a->I; const int64_t rb1 = gq_row_bytes(t1, I), rb2 = gq_row_bytes(t2, I);
#ifdef GQ_R16_ON
    if (gq_r16_base(t1) >= 0 || gq_r16_base(t2) >= 0) { gq_matmul_act(y1, a, W1, t1, O); gq_matmul_act(y2, a, W2, t2, O); return; }
#endif
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < 2 * O; r++) {
        if (r < O) gq_row_all(y1 + r, O, a, W1 + (int64_t)r * rb1, t1, I);
        else       gq_row_all(y2 + (r - O), O, a, W2 + (int64_t)(r - O) * rb2, t2, I);
    }
}

/* y[s*O + o] = W[o,:] . x[s,:]; integer path for every block type, the float
 * reference for F32/F16/BF16, odd widths, or GQ_EXACT=1. 0 or -1. */
static int gq_matmul(float *y, const float *x, const uint8_t *W, int t, int S, int I, int O) {
    if (!gq_int_ok(t, I)) return gq_matmul_f32(y, x, W, t, S, I, O);
    if (gq_row_bytes(t, I) < 0) return -1;
    gq_act a; memset(&a, 0, sizeof a);
    if (gq_act_set(&a, x, S, I)) { gq_act_free(&a); return gq_matmul_f32(y, x, W, t, S, I, O); }
    gq_matmul_act(y, &a, W, t, O);
    gq_act_free(&a);
    return 0;
}

#endif /* GGML_QUANTS_H */
