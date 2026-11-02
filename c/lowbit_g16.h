/* lowbit_g16.h — colibri's int2-g16 and int3-g16 routed-expert formats and their kernels.
 *
 *   codes   a row of I weights (I % 128 == 0) is I/128 blocks. Each block holds the low
 *           two bits of its 128 codes as int2_g32.h does (32 bytes: byte k = elements k,
 *           k+32, k+64, k+96 in bits 0-1, 2-3, 4-5, 6-7) and, for 3 bits, then the high
 *           bits (16 bytes: byte k = elements k, k+16, k+32, ..., k+112 in bits 0..7, so
 *           a group of 32 is bits 2h and 2h+1 of the same 16 bytes: one variable shift
 *           per half of a register finds them). A block is 32 bytes at 2 bits, 48 at 3.
 *   values  code -> level, the four-level (-10 -3 3 10) or eight-level (-26 -16 -9 -3 3 9
 *           16 26) Lloyd-Max grid of a bell-shaped weight in integers: fitted per layer and
 *           matrix on Qwen3.8's experts the levels came out the same, so the grid is fixed
 *           and one pshufb decodes it.
 *   scales  one 4-bit scale per 16 inputs (two to a byte: group 2k low, 2k+1 high, so a row
 *           has I/32 bytes) under one f32 per row: w = d[row] * q * level. At the same size
 *           as one byte per 32, half the groups: on 24 Qwen3.8 experts the weights' error
 *           is 5% lower at 2 bits and 9% lower at 3.
 *   size    2.25 and 3.25 bits per weight, plus 32 bits per row.
 *   x       the activation contract of int2_g32.h (x2_quant_act): int8 per 32, one f32
 *           scale each.
 *
 * Per group of 32 the two 4-bit scales go into the integer sum itself: |level| * sign(x)
 * byte pairs (maddubs, <= 2*26*127) multiplied by a vector holding the first scale for the
 * group's first 16 elements and the second for the rest (madd, <= 15 * 2 * 6604), so the
 * group costs one f32 fma, d * dx, as int2_g32.h's does. The scalar reference defines the
 * numerics (lane l = elements 4l..4l+3 times its half's scale, one fmaf per group with
 * d*dx[g], the lanes summed as GQ_HSUM8); the AVX2 kernel reproduces it bit for bit for
 * every S. Header-only, all static, pure compute. */
#ifndef COLI_LOWBIT_G16_H
#define COLI_LOWBIT_G16_H

#include <string.h>
#include "int2_g32.h"   /* x2_quant_act, x2_hsum8_scalar, X2_MAX_ROWS */

static const int8_t XL_LEVELS2[4] = { -10, -3, 3, 10 };
static const int8_t XL_LEVELS3[8] = { -26, -16, -9, -3, 3, 9, 16, 26 };

static inline int    xl_layout_ok(int I)             { return I > 0 && I % 128 == 0; }
static inline size_t xl_block_bytes(int bits)        { return bits == 3 ? 48 : 32; }
static inline size_t xl_code_bytes(int bits, int I)  { return (size_t)I / 128 * xl_block_bytes(bits); }
static inline size_t xl_scale_bytes(int I)           { return (size_t)I / 32; }

/* code of element i of a row (tests, packing checks) */
static inline int xl_code(const uint8_t *row, int bits, int i) {
    const uint8_t *blk = row + (size_t)(i / 128) * xl_block_bytes(bits);
    int k = i % 128, h = k / 32, j = k % 32;
    int c = (blk[j] >> (2 * h)) & 3;
    if (bits == 3) c |= ((blk[32 + k % 16] >> (k / 16)) & 1) << 2;
    return c;
}
static inline int xl_level(int bits, int code) { return bits == 3 ? XL_LEVELS3[code] : XL_LEVELS2[code]; }
static inline int xl_scale(const uint8_t *q, int g16) { return (q[g16 / 2] >> (4 * (g16 & 1))) & 15; }

/* y = sum_i w[i] * x[i] for one weight row and one activation row (S = 1) */
static inline float xl_dot_ref(const uint8_t *codes, const uint8_t *q, float d, int bits,
                               const int8_t *xq, const float *dx, int I) {
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int g = 0; g < I / 32; g++) {
        const int8_t *xb = xq + (size_t)g * 32;
        const float sc = d * dx[g];
        for (int l = 0; l < 8; l++) {
            int32_t s = 0;
            for (int j = 4 * l; j < 4 * l + 4; j++) s += xl_level(bits, xl_code(codes, bits, g * 32 + j)) * xb[j];
            s *= xl_scale(q, 2 * g + (l >= 4));
            acc[l] = fmaf((float)s, sc, acc[l]);
        }
    }
    return x2_hsum8_scalar(acc);
}

#ifdef X2_HAVE_AVX2
#define XL_HAVE_AVX2 1
static inline __attribute__((always_inline)) void xl_dot_body(float *y, int ystride, const uint8_t *codes, const uint8_t *q,
                                                             float d, const int8_t *xq, const float *dx, int I,
                                                             const int S, const int bits) {
    const __m256i lut = bits == 3 ? _mm256_setr_epi8(-26, -16, -9, -3, 3, 9, 16, 26, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     -26, -16, -9, -3, 3, 9, 16, 26, 0, 0, 0, 0, 0, 0, 0, 0)
                                  : _mm256_setr_epi8(-10, -3, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                     -10, -3, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi8(1);
    const int ng = I / 32;
    const size_t bb_bytes = xl_block_bytes(bits);
    __m256 acc[X2_MAX_ROWS];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    for (int b = 0; b < I / 128; b++) {
        const uint8_t *blk = codes + (size_t)b * bb_bytes;
        const __m256i bb = _mm256_loadu_si256((const __m256i *)blk);
        /* the high plane in both halves: group h's first 16 elements are bit 2h, the rest 2h+1 */
        const __m256i hp = bits == 3 ? _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(blk + 32)))
                                     : _mm256_setzero_si256();
        for (int h = 0; h < 4; h++) {
            const int g = 4 * b + h;
            __m256i c = _mm256_and_si256(h == 0 ? bb : _mm256_srli_epi16(bb, 2 * h), m3);
            if (bits == 3) {
                const __m256i cnt = _mm256_setr_epi32(2 * h, 2 * h, 2 * h, 2 * h, 2 * h + 1, 2 * h + 1, 2 * h + 1, 2 * h + 1);
                c = _mm256_or_si256(c, _mm256_slli_epi16(_mm256_and_si256(_mm256_srlv_epi32(hp, cnt), one), 2));
            }
            const __m256i w = _mm256_shuffle_epi8(lut, c), aw = _mm256_sign_epi8(w, w);
            const int qb = q[g];
            const __m256i qv = _mm256_set_m128i(_mm_set1_epi16((short)(qb >> 4)), _mm_set1_epi16((short)(qb & 15)));
            for (int s = 0; s < S; s++) {
                const __m256i x = _mm256_loadu_si256((const __m256i *)(xq + (size_t)s * I + (size_t)g * 32));
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)), qv);
                acc[s] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[(size_t)s * ng + g]), acc[s]);
            }
        }
    }
    for (int s = 0; s < S; s++) y[(size_t)s * ystride] = x2_hsum8(acc[s]);
}
#endif

/* S (1..X2_MAX_ROWS) activation rows of one weight row; xq/dx row-major [S][I], [S][I/32] */
static inline void xl_dot(float *y, int ystride, const uint8_t *codes, const uint8_t *q, float d, int bits,
                          const int8_t *xq, const float *dx, int I, int S) {
#ifdef XL_HAVE_AVX2
#define XL_CASE(n) case n: if (bits == 3) xl_dot_body(y, ystride, codes, q, d, xq, dx, I, n, 3); \
                           else xl_dot_body(y, ystride, codes, q, d, xq, dx, I, n, 2); break;
    switch (S) { XL_CASE(1) XL_CASE(2) XL_CASE(3) default: XL_CASE(4) }
#undef XL_CASE
#else
    for (int s = 0; s < S; s++)
        y[(size_t)s * ystride] = xl_dot_ref(codes, q, d, bits, xq + (size_t)s * I, dx + (size_t)s * (I / 32), I);
#endif
}

/* y[s*O + o] = W[o,:] . x[s,:] for an int2/int3-g16 matrix: codes [O][xl_code_bytes],
 * scales [O][I/32], d [O]. As x2_matmul: activations quantized once, rows four at a time.
 * 0, or -1 when out of memory. */
static inline int xl_matmul(float *y, const float *x, const uint8_t *codes, const uint8_t *q, const float *d,
                            int bits, int S, int I, int O) {
    const size_t cb = xl_code_bytes(bits, I), qb = xl_scale_bytes(I); const int ng = I / 32;
    int8_t *xq = (int8_t *)malloc((size_t)S * I); float *dx = (float *)malloc((size_t)S * ng * sizeof(float));
    if (!xq || !dx) { free(xq); free(dx); return -1; }
    for (int s = 0; s < S; s++) x2_quant_act(x + (size_t)s * I, I, xq + (size_t)s * I, dx + (size_t)s * ng);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        for (int s0 = 0; s0 < S; s0 += X2_MAX_ROWS) {
            const int n = S - s0 < X2_MAX_ROWS ? S - s0 : X2_MAX_ROWS;
            xl_dot(y + (size_t)s0 * O + o, O, codes + (size_t)o * cb, q + (size_t)o * qb, d[o], bits,
                   xq + (size_t)s0 * I, dx + (size_t)s0 * ng, I, n);
        }
    free(xq); free(dx);
    return 0;
}


/* ---- rows interleaved by 16 (xl16_*) ---------------------------------------------
 * The row kernel above keeps eight lane sums per row with an fma each, the per-row float
 * work that held a prompt's chunk back. Here 16 rows share a register (lane r = row r):
 * a group of 16 rows holds their 16 f32 row scales, then per 32 inputs (g) a block of
 *   16 bytes   row r's scale byte of g (the two 4-bit scales)
 *   128 bytes  the low two bits: words W = 0, 1 of 64 bytes, row r's four bytes at 4r, byte k
 *              holding in bits 2i the code of element 16W + 4i + k
 *   64 bytes   (3 bits) the high bits: row r's four bytes at 4r, byte k bit i = element 4i + k
 * so quad i (elements 4i .. 4i+3) of the 16 rows is a shift and a mask (two for the high
 * bit), one pshufb to the level + bias, one vpdpbusd against the broadcast quad. The same
 * bytes as the rows (13/32 a weight at 3 bits, 9/32 at 2, and 4 a row).
 * Numerics: per 32 inputs the exact integer sum_16 level * x times each half's scale (an
 * integer below 2^24: exact as a float), then one fma with d * dx: one rounding per 32
 * where xl_dot_ref rounds eight lane sums, so not its bits. xl16_ref is the reference; the
 * input is the activation contract (x2_quant_act's bytes and scales) plus the sums per 16
 * of the bytes (gq_quant_act's xs). */
static inline size_t xl16_gblock(int bits) { return bits == 3 ? 208 : 144; }
static inline size_t xl16_group_bytes(int bits, int I) { return 64 + (size_t)(I / 32) * xl16_gblock(bits); }
static inline uint32_t xl16_rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
/* rows (whole groups) of codes [rows][xl_code_bytes], q [rows][I/32], d [rows] into dst. Word
 * by word: group h of a block of 128 is bits 2h of its 32 low bytes (byte e = element e of
 * the group) and bits 2h, 2h+1 of its 16 high bytes, so the group's two transposes are a few
 * shifts and masks on 32-bit words (this runs at every expert read). */
static void xl16_pack(uint8_t *dst, const uint8_t *codes, const uint8_t *q, const float *d, int rows, int bits, int I) {
    const size_t cb = xl_code_bytes(bits, I), qb = xl_scale_bytes(I), gbk = xl16_gblock(bits), gb = xl16_group_bytes(bits, I);
    const size_t bb = xl_block_bytes(bits);
    for (int g = 0; g < rows / 16; g++) {
        uint8_t *grp = dst + (size_t)g * gb;
        memcpy(grp, d + 16 * g, 64);
        for (int r = 0; r < 16; r++) {
            const uint8_t *row = codes + (size_t)(16 * g + r) * cb, *qr = q + (size_t)(16 * g + r) * qb;
#if defined(__AVX2__)
            /* the same transposes on vectors: a block's 32 low bytes as eight words (W = 0: words
             * 0-3, W = 1: words 4-7), word i shifted by 2i and the four of a half or-ed together;
             * the 16 high bytes as four words, word m shifted by m (bit 2h) and by m + 4 (2h + 1) */
            const __m256i m03 = _mm256_set1_epi32(0x03030303), sl2 = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
            const __m128i m01 = _mm_set1_epi32(0x01010101), slo = _mm_setr_epi32(0, 1, 2, 3), shi = _mm_setr_epi32(4, 5, 6, 7);
            for (int b = 0; b < I / 128; b++) {
                const uint8_t *blk = row + (size_t)b * bb;
                const __m256i v = _mm256_loadu_si256((const __m256i *)blk);
                const __m128i hp = bits == 3 ? _mm_loadu_si128((const __m128i *)(blk + 32)) : _mm_setzero_si128();
                for (int h = 0; h < 4; h++) {
                    const int gi = 4 * b + h; uint8_t *o = grp + 64 + (size_t)gi * gbk;
                    o[r] = qr[gi];
                    __m256i c = _mm256_sllv_epi32(_mm256_and_si256(_mm256_srli_epi32(v, 2 * h), m03), sl2);
                    c = _mm256_or_si256(c, _mm256_shuffle_epi32(c, 0x4E)); c = _mm256_or_si256(c, _mm256_shuffle_epi32(c, 0xB1));
                    const uint32_t t0 = (uint32_t)_mm256_cvtsi256_si32(c), t1 = (uint32_t)_mm256_extract_epi32(c, 4);
                    memcpy(o + 16 + 4 * r, &t0, 4); memcpy(o + 80 + 4 * r, &t1, 4);
                    if (bits == 3) {
                        __m128i t = _mm_or_si128(_mm_sllv_epi32(_mm_and_si128(_mm_srli_epi32(hp, 2 * h), m01), slo),
                                                 _mm_sllv_epi32(_mm_and_si128(_mm_srli_epi32(hp, 2 * h + 1), m01), shi));
                        t = _mm_or_si128(t, _mm_shuffle_epi32(t, 0x4E)); t = _mm_or_si128(t, _mm_shuffle_epi32(t, 0xB1));
                        const uint32_t th = (uint32_t)_mm_cvtsi128_si32(t);
                        memcpy(o + 144 + 4 * r, &th, 4);
                    }
                }
            }
            continue;
#endif
            for (int gi = 0; gi < I / 32; gi++) {
                uint8_t *o = grp + 64 + (size_t)gi * gbk;
                const uint8_t *blk = row + (size_t)(gi / 4) * bb; const int h = gi % 4;
                o[r] = qr[gi];
                for (int W = 0; W < 2; W++) {   /* byte k, bits 2i: element 16W + 4i + k = low byte 16W + 4i + k, bits 2h */
                    uint32_t t = 0;
                    for (int i = 0; i < 4; i++) t |= ((xl16_rd32(blk + 16 * W + 4 * i) >> (2 * h)) & 0x03030303u) << (2 * i);
                    memcpy(o + 16 + 64 * W + 4 * r, &t, 4);
                }
                if (bits == 3) {   /* byte k, bit i: element 4i + k = high byte (4i + k) % 16, bit 2h + i / 4 */
                    uint32_t t = 0;
                    for (int i = 0; i < 8; i++) t |= ((xl16_rd32(blk + 32 + 4 * (i % 4)) >> (2 * h + i / 4)) & 0x01010101u) << i;
                    memcpy(o + 144 + 4 * r, &t, 4);
                }
            }
        }
    }
}
/* the inverse: the rows' codes, scale bytes and row scales back */
static void xl16_unpack(uint8_t *codes, uint8_t *q, float *d, const uint8_t *src, int rows, int bits, int I) {
    const size_t cb = xl_code_bytes(bits, I), qb = xl_scale_bytes(I), gbk = xl16_gblock(bits), gb = xl16_group_bytes(bits, I);
    memset(codes, 0, cb * (size_t)rows);
    for (int g = 0; g < rows / 16; g++) {
        const uint8_t *grp = src + (size_t)g * gb;
        memcpy(d + 16 * g, grp, 64);
        for (int r = 0; r < 16; r++) {
            uint8_t *row = codes + (size_t)(16 * g + r) * cb, *qr = q + (size_t)(16 * g + r) * qb;
            for (int gi = 0; gi < I / 32; gi++) {
                const uint8_t *o = grp + 64 + (size_t)gi * gbk;
                qr[gi] = o[r];
                for (int e = 0; e < 32; e++) {
                    const int W = e / 16, i = (e % 16) / 4, k = e % 4;
                    int c = (o[16 + 64 * W + 4 * r + k] >> (2 * i)) & 3;
                    if (bits == 3) c |= ((o[144 + 4 * r + k] >> (e / 4)) & 1) << 2;
                    const int el = 32 * gi + e, kk = el % 128;
                    uint8_t *blk = row + (size_t)(el / 128) * xl_block_bytes(bits);
                    blk[kk % 32] |= (uint8_t)((c & 3) << (2 * (kk / 32)));
                    if (bits == 3) blk[32 + kk % 16] |= (uint8_t)(((c >> 2) & 1) << (kk / 16));
                }
            }
        }
    }
}
/* the numerics in plain C: row r of a group against one activation row */
static inline float xl16_ref(const uint8_t *codes, const uint8_t *q, float d, int bits,
                             const int8_t *xq, const float *dx, int I) {
    float acc = 0.f;
    for (int g = 0; g < I / 32; g++) {
        int32_t p[2] = { 0, 0 };
        for (int j = 0; j < 32; j++) p[j / 16] += xl_level(bits, xl_code(codes, bits, g * 32 + j)) * xq[(size_t)g * 32 + j];
        acc = fmaf((float)(p[0] * xl_scale(q, 2 * g) + p[1] * xl_scale(q, 2 * g + 1)), d * dx[g], acc);
    }
    return acc;
}
#if defined(__AVX2__) && defined(__FMA__)
#define XL16_ON 1
static inline int32_t xl16_ld32(const int8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
/* 16 rows (one group) against S <= 8 activation rows: y[s * ystride + r] */
static inline __attribute__((always_inline)) void xl16_body512(float *y, int ystride, const uint8_t *grp, const int8_t *xq,
                                                              const float *dx, const int32_t *xs, int I, const int S, const int bits) {
    const int B = bits == 3 ? 26 : 10;   /* levels + B are the unsigned codes */
    const __m512i lut = _mm512_broadcast_i32x4(bits == 3 ? _mm_setr_epi8(0, 10, 17, 23, 29, 35, 42, 52, 0, 0, 0, 0, 0, 0, 0, 0)
                                                         : _mm_setr_epi8(0, 7, 13, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    const __m512i m3 = _mm512_set1_epi8(3), m4 = _mm512_set1_epi8(4), m15 = _mm512_set1_epi32(15);
    const __m512 dv = _mm512_loadu_ps((const float *)grp);
    const size_t gbk = xl16_gblock(bits);
    __m512 acc[8];
    for (int s = 0; s < S; s++) acc[s] = _mm512_setzero_ps();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *o = grp + 64 + (size_t)g * gbk;
        const __m512i qb = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)o));
        const __m512 sa = _mm512_cvtepi32_ps(_mm512_and_si512(qb, m15)), sb = _mm512_cvtepi32_ps(_mm512_srli_epi32(qb, 4));
        __m512i c[8];
        for (int W = 0; W < 2; W++) {
            const __m512i v = _mm512_loadu_si512(o + 16 + 64 * W);
            for (int i = 0; i < 4; i++) c[4 * W + i] = _mm512_and_si512(i ? _mm512_srli_epi16(v, 2 * i) : v, m3);
        }
        if (bits == 3) {
            const __m512i h = _mm512_loadu_si512(o + 144);
            for (int i = 0; i < 8; i++) {
                const __m512i hb = i < 2 ? _mm512_slli_epi16(h, 2 - i) : i == 2 ? h : _mm512_srli_epi16(h, i - 2);
                c[i] = _mm512_or_si512(c[i], _mm512_and_si512(hb, m4));
            }
        }
        for (int i = 0; i < 8; i++) c[i] = _mm512_shuffle_epi8(lut, c[i]);
        for (int s = 0; s < S; s++) {
            const int8_t *xb = xq + (size_t)s * I + 32 * (size_t)g;
            __m512i p0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c[0], _mm512_set1_epi32(xl16_ld32(xb)));
            __m512i p1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c[4], _mm512_set1_epi32(xl16_ld32(xb + 16)));
            for (int i = 1; i < 4; i++) {
                p0 = _mm512_dpbusd_epi32(p0, c[i], _mm512_set1_epi32(xl16_ld32(xb + 4 * i)));
                p1 = _mm512_dpbusd_epi32(p1, c[4 + i], _mm512_set1_epi32(xl16_ld32(xb + 16 + 4 * i)));
            }
            const int32_t *x16 = xs + (size_t)s * (I / 16) + 2 * (size_t)g;
            p0 = _mm512_sub_epi32(p0, _mm512_set1_epi32(B * x16[0])); p1 = _mm512_sub_epi32(p1, _mm512_set1_epi32(B * x16[1]));
            const __m512 t = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p1), sb, _mm512_mul_ps(_mm512_cvtepi32_ps(p0), sa));   /* exact */
            acc[s] = _mm512_fmadd_ps(t, _mm512_mul_ps(dv, _mm512_set1_ps(dx[(size_t)s * (I / 32) + g])), acc[s]);
        }
    }
    for (int s = 0; s < S; s++) _mm512_storeu_ps(y + (size_t)s * ystride, acc[s]);
}
#define XL16_SMAX 8
#else
/* half h (rows 8h .. 8h+7) of a group against S <= 4 activation rows */
static inline __attribute__((always_inline)) void xl16_body256(float *y, int ystride, const uint8_t *grp, const int8_t *xq,
                                                              const float *dx, const int32_t *xs, int I, const int S, const int bits, const int h) {
    const int B = bits == 3 ? 26 : 10;
    const __m256i lut = _mm256_broadcastsi128_si256(bits == 3 ? _mm_setr_epi8(0, 10, 17, 23, 29, 35, 42, 52, 0, 0, 0, 0, 0, 0, 0, 0)
                                                              : _mm_setr_epi8(0, 7, 13, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
    const __m256i m3 = _mm256_set1_epi8(3), m4 = _mm256_set1_epi8(4), m15 = _mm256_set1_epi32(15);
    const __m256 dv = _mm256_loadu_ps((const float *)grp + 8 * h);
    const size_t gbk = xl16_gblock(bits);
    __m256 acc[4];
    for (int s = 0; s < S; s++) acc[s] = _mm256_setzero_ps();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *o = grp + 64 + (size_t)g * gbk;
        const __m256i qb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(o + 8 * h)));
        const __m256 sa = _mm256_cvtepi32_ps(_mm256_and_si256(qb, m15)), sb = _mm256_cvtepi32_ps(_mm256_srli_epi32(qb, 4));
        __m256i c[8];
        for (int W = 0; W < 2; W++) {
            const __m256i v = _mm256_loadu_si256((const __m256i *)(o + 16 + 64 * W + 32 * h));
            for (int i = 0; i < 4; i++) c[4 * W + i] = _mm256_and_si256(i ? _mm256_srli_epi16(v, 2 * i) : v, m3);
        }
        if (bits == 3) {
            const __m256i hh = _mm256_loadu_si256((const __m256i *)(o + 144 + 32 * h));
            for (int i = 0; i < 8; i++) {
                const __m256i hb = i < 2 ? _mm256_slli_epi16(hh, 2 - i) : i == 2 ? hh : _mm256_srli_epi16(hh, i - 2);
                c[i] = _mm256_or_si256(c[i], _mm256_and_si256(hb, m4));
            }
        }
        for (int i = 0; i < 8; i++) c[i] = _mm256_shuffle_epi8(lut, c[i]);
        for (int s = 0; s < S; s++) {
            const int8_t *xb = xq + (size_t)s * I + 32 * (size_t)g;
            __m256i p0 = _mm256_setzero_si256(), p1 = _mm256_setzero_si256();
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
            for (int i = 0; i < 4; i++) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                p0 = _mm256_dpbusd_epi32(p0, c[i], _mm256_set1_epi32(xl16_ld32(xb + 4 * i)));
                p1 = _mm256_dpbusd_epi32(p1, c[4 + i], _mm256_set1_epi32(xl16_ld32(xb + 16 + 4 * i)));
#else
                p0 = _mm256_dpbusd_avx_epi32(p0, c[i], _mm256_set1_epi32(xl16_ld32(xb + 4 * i)));
                p1 = _mm256_dpbusd_avx_epi32(p1, c[4 + i], _mm256_set1_epi32(xl16_ld32(xb + 16 + 4 * i)));
#endif
            }
#else   /* codes <= 52: two maddubs add in 16 bits below 2^15 */
            const __m256i one = _mm256_set1_epi16(1);
            for (int i = 0; i < 4; i += 2) {
                p0 = _mm256_add_epi32(p0, _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(c[i], _mm256_set1_epi32(xl16_ld32(xb + 4 * i))),
                         _mm256_maddubs_epi16(c[i + 1], _mm256_set1_epi32(xl16_ld32(xb + 4 * i + 4)))), one));
                p1 = _mm256_add_epi32(p1, _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(c[4 + i], _mm256_set1_epi32(xl16_ld32(xb + 16 + 4 * i))),
                         _mm256_maddubs_epi16(c[5 + i], _mm256_set1_epi32(xl16_ld32(xb + 20 + 4 * i)))), one));
            }
#endif
            const int32_t *x16 = xs + (size_t)s * (I / 16) + 2 * (size_t)g;
            p0 = _mm256_sub_epi32(p0, _mm256_set1_epi32(B * x16[0])); p1 = _mm256_sub_epi32(p1, _mm256_set1_epi32(B * x16[1]));
            const __m256 t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), sb, _mm256_mul_ps(_mm256_cvtepi32_ps(p0), sa));
            acc[s] = _mm256_fmadd_ps(t, _mm256_mul_ps(dv, _mm256_set1_ps(dx[(size_t)s * (I / 32) + g])), acc[s]);
        }
    }
    for (int s = 0; s < S; s++) _mm256_storeu_ps(y + (size_t)s * ystride + 8 * h, acc[s]);
}
#define XL16_SMAX 4
#endif
/* group grp's 16 rows (outputs y .. y + 15) against S activation rows: xq/dx/xs the
 * activation contract plus the sums per 16 (gq_act's q, d, s) */
static void xl16_rows(float *y, int ystride, const uint8_t *grp, const int8_t *xq, const float *dx, const int32_t *xs,
                      int I, int S, int bits) {
    for (int s0 = 0; s0 < S; s0 += XL16_SMAX) {
        const int n = S - s0 < XL16_SMAX ? S - s0 : XL16_SMAX;
        float *ys = y + (size_t)s0 * ystride; const int8_t *xb = xq + (size_t)s0 * I;
        const float *db = dx + (size_t)s0 * (I / 32); const int32_t *sb = xs + (size_t)s0 * (I / 16);
#if XL16_SMAX == 8
#define XL16_CALL(N, BITS) xl16_body512(ys, ystride, grp, xb, db, sb, I, N, BITS)
#else
#define XL16_CALL(N, BITS) do { xl16_body256(ys, ystride, grp, xb, db, sb, I, N, BITS, 0); xl16_body256(ys, ystride, grp, xb, db, sb, I, N, BITS, 1); } while (0)
#endif
        if (bits == 3) switch (n) { case 1: XL16_CALL(1, 3); break; case 2: XL16_CALL(2, 3); break; case 3: XL16_CALL(3, 3); break;
                                    case 4: XL16_CALL(4, 3); break; default: XL16_CALL(n, 3); break; }
        else switch (n) { case 1: XL16_CALL(1, 2); break; case 2: XL16_CALL(2, 2); break; case 3: XL16_CALL(3, 2); break;
                          case 4: XL16_CALL(4, 2); break; default: XL16_CALL(n, 2); break; }
#undef XL16_CALL
    }
}
#endif

#endif /* COLI_LOWBIT_G16_H */
