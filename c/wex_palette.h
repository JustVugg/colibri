/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef COLI_WEX_PALETTE_H
#define COLI_WEX_PALETTE_H

/* Three-bit indices use the same low-first 8 values / 3 bytes layout as
 * TurboQuantQJLCompressor._pack_indices. Each UE8M0 exponent selects an
 * eight-entry palette of native E2M1 nibbles. No floating-point work at load.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define WEX_PALETTE3_MAGIC UINT32_C(0x33504357)
#define WEX_PALETTE3_HEADER_BYTES 16u
#define WEX_PALETTE3_TABLE_BYTES (3u * 256u * 4u)
#define WEX_PALETTE3_PREFIX_BYTES (WEX_PALETTE3_HEADER_BYTES + WEX_PALETTE3_TABLE_BYTES)

#if defined(__AVX2__)
static inline __m128i wex_palette_load12(const unsigned char *src) {
    uint64_t low;
    uint32_t high;
    memcpy(&low, src, 8u);
    memcpy(&high, src + 8u, 4u);
    return _mm_or_si128(_mm_cvtsi64_si128((long long)low),
                        _mm_slli_si128(_mm_cvtsi32_si128((int)high), 8));
}
#endif

static inline void wex_palette_expand_table(const unsigned char *packed,
                                             unsigned char table[256][16]) {
    unsigned int scale, code;
    for (scale = 0; scale < 256u; scale++) {
        for (code = 0; code < 8u; code++)
            table[scale][code] = (unsigned char)((packed[scale * 4u + code / 2u] >>
                                                 ((code & 1u) * 4u)) & 15u);
        memset(table[scale] + 8u, 0, 8u);
    }
}

/* raw_bytes is a multiple of 16 (one MXFP4 scale group); scales has exactly
 * raw_bytes / 16 values. Source can overlap destination during forward
 * expansion only when the record decoder's tail-preservation rule is met.
 */
static inline void wex_unpack_palette3(const unsigned char *src,
                                        unsigned char *dst, size_t raw_bytes,
                                        const unsigned char *scales,
                                        const unsigned char table[256][16]) {
    size_t index = 0;
#if defined(__AVX2__)
    const __m256i shuffle = _mm256_setr_epi8(
        0,1,2,-128,3,4,5,-128,6,7,8,-128,9,10,11,-128,
        0,1,2,-128,3,4,5,-128,6,7,8,-128,9,10,11,-128);
    for (; index + 32u <= raw_bytes; index += 32u, src += 24u, scales += 2u) {
        __m256i lut = _mm256_set_m128i(
            _mm_loadu_si128((const __m128i *)table[scales[1]]),
            _mm_loadu_si128((const __m128i *)table[scales[0]]));
        /* Ordinary loads are safe while >=48 raw bytes remain (>=36 encoded
         * bytes). Only the final 32-byte group needs exact 12-byte loads.
         * This avoids overreads on both the mapped slab and the saved tail.
         */
        __m256i input = index + 32u < raw_bytes ?
            _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src + 12u)),
                             _mm_loadu_si128((const __m128i *)src)) :
            _mm256_set_m128i(wex_palette_load12(src + 12u), wex_palette_load12(src));
        __m256i x = _mm256_shuffle_epi8(input, shuffle);
        /* Split 24 bits into two 12-bit halves, then each half into two
         * six-bit indices. The masks discard the unused high four bits of
         * each half; no variable shifts or per-value gathers are needed.
         */
        x = _mm256_blend_epi16(x, _mm256_slli_epi32(x, 4), 0xaa);
        __m256i codes = _mm256_or_si256(
            _mm256_and_si256(x, _mm256_set1_epi32(0x003f003f)),
            _mm256_and_si256(_mm256_slli_epi16(x, 2), _mm256_set1_epi32(0x3f003f00)));
        __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(codes, _mm256_set1_epi8(7)));
        __m256i hi = _mm256_shuffle_epi8(lut,
            _mm256_and_si256(_mm256_srli_epi16(codes, 3), _mm256_set1_epi8(7)));
        _mm256_storeu_si256((__m256i *)(dst + index),
                            _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4)));
    }
#endif
    for (; index < raw_bytes; index += 16u, src += 12u, scales++) {
        const unsigned char *palette = table[*scales];
        unsigned int group, value;
        for (group = 0; group < 4u; group++) {
            uint32_t word = (uint32_t)src[group * 3u] |
                            ((uint32_t)src[group * 3u + 1u] << 8) |
                            ((uint32_t)src[group * 3u + 2u] << 16);
            for (value = 0; value < 4u; value++)
                dst[index + group * 4u + value] = (unsigned char)(
                    palette[(word >> (value * 6u)) & 7u] |
                    (palette[(word >> (value * 6u + 3u)) & 7u] << 4));
        }
    }
}

#endif
