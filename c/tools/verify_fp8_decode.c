#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* reference: exact copy of coli_e4m3fn_decode (deepseek_v4.c:14774) */
static float ref_decode(uint8_t value) {
    int sign = value >> 7, exponent = (value >> 3) & 15, mantissa = value & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float number;
    if (!exponent) number = ldexpf((float)mantissa, -9);
    else number = ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -number : number;
}

/* candidate: branchless SIMD decode of 8 e4m3fn codes -> 8 f32, aiming bit-exact */
static inline __m256 fp8_decode8(__m256i codes) {
    __m256i man  = _mm256_and_si256(codes, _mm256_set1_epi32(7));
    __m256i exp  = _mm256_and_si256(_mm256_srli_epi32(codes, 3), _mm256_set1_epi32(0xF));
    __m256i sgn  = _mm256_slli_epi32(_mm256_srli_epi32(codes, 7), 31);        /* bit7 -> bit31 */
    /* normal magnitude: (exp+120)<<23 | man<<20 */
    __m256i nbits = _mm256_or_si256(
        _mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(120)), 23),
        _mm256_slli_epi32(man, 20));
    __m256 nval = _mm256_castsi256_ps(nbits);
    /* subnormal magnitude (exp==0): (float)man * 2^-9  (exact) */
    __m256 sval = _mm256_mul_ps(_mm256_cvtepi32_ps(man), _mm256_set1_ps(ldexpf(1.0f, -9)));
    __m256 is_sub = _mm256_castsi256_ps(_mm256_cmpeq_epi32(exp, _mm256_setzero_si256()));
    __m256 mag = _mm256_blendv_ps(nval, sval, is_sub);
    __m256i sbits = _mm256_or_si256(_mm256_castps_si256(mag), sgn);           /* apply sign */
    /* NaN: (code & 0x7F) == 0x7F  -> canonical qNaN, overwrites sign */
    __m256i is_nan = _mm256_cmpeq_epi32(_mm256_and_si256(codes, _mm256_set1_epi32(0x7F)),
                                        _mm256_set1_epi32(0x7F));
    __m256i fbits = _mm256_blendv_epi8(sbits, _mm256_set1_epi32(0x7FC00000), is_nan);
    return _mm256_castsi256_ps(fbits);
}

int main(void) {
    /* what bits does NAN actually compile to here? */
    float n = NAN; uint32_t nb; memcpy(&nb, &n, 4);
    printf("platform NAN bits = 0x%08x (my const = 0x7FC00000)\n", nb);
    int mism = 0;
    for (int base = 0; base < 256; base += 8) {
        int32_t c[8]; for (int i = 0; i < 8; i++) c[i] = base + i;
        __m256i codes = _mm256_loadu_si256((const __m256i *)c);
        float out[8]; _mm256_storeu_ps(out, fp8_decode8(codes));
        for (int i = 0; i < 8; i++) {
            float r = ref_decode((uint8_t)(base + i));
            uint32_t rb, ob; memcpy(&rb, &r, 4); memcpy(&ob, &out[i], 4);
            if (rb != ob) { printf("MISMATCH code=%3d ref=0x%08x(%g) mine=0x%08x(%g)\n",
                                   base + i, rb, r, ob, out[i]); mism++; }
        }
    }
    printf(mism ? "FAIL: %d mismatches\n" : "ALL 256 CODES BIT-EXACT\n", mism);
    return mism ? 1 : 0;
}
