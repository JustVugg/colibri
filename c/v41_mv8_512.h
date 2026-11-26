/* v41_mv8_512.h -- deepseek_v41's FP8 trunk product (mv8_rows_cpu) with AVX-512 and the AVX2 loop's bits.
 * The AVX2 loop, per output o and row r, per 32-column tile: w0..w3 = e4m3_decode8 of 8 bytes each (NaN for 0x7F/0xFF);
 * acc = w0*x0, then fma(w1,x1,acc), fma(w2,x2,acc), fma(w3,x3,acc) lane by lane; h = hsum256(acc) =
 * ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)); sum += h * tile (one fused multiply-add, as gcc emits it).
 * Here a zmm carries two outputs, o in the low 256 bits and o+1 in the high: the same decode, the same multiply and
 * fmas lane by lane (the activations broadcast to both halves), the same pairwise additions inside each half (128-bit
 * blocks swapped, then lanes 2,3 onto 0,1, then lane 1 onto 0), and each output's scalar fma with its tile scale. Every
 * operation an output's value goes through is the AVX2 loop's, in its order, so the bits are the same. Output pairs
 * (2k, 2k+1) share their 32-row scale block. An odd last output, and I not a multiple of 32, are left to the caller. */
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define V41_MV8_512_HAVE 1

__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma"))) static inline __m512 v41_e4_dec16(const uint8_t *p0, const uint8_t *p1) {
    const __m128i v = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)p0), _mm_loadl_epi64((const __m128i *)p1));
    const __m512i b = _mm512_cvtepu8_epi32(v);
    const __m512i low = _mm512_and_si512(b, _mm512_set1_epi32(0x7F));
    const __m512i bits = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 24),
                                         _mm512_slli_epi32(low, 20));
    const __m512 value = _mm512_mul_ps(_mm512_castsi512_ps(bits), _mm512_set1_ps(0x1p120f));
    const __mmask16 nan = _mm512_cmpeq_epi32_mask(low, _mm512_set1_epi32(0x7F));
    return _mm512_mask_blend_ps(nan, value, _mm512_set1_ps(NAN));
}
/* hsum256 in each 256-bit half: lane 0 holds the low half's, lane 8 the high half's */
__attribute__((target("avx512f,avx512dq"))) static inline __m512 v41_hsum2(__m512 a) {
    const __m512 b = _mm512_add_ps(a, _mm512_shuffle_f32x4(a, a, _MM_SHUFFLE(2, 3, 0, 1)));   /* a_j + a_{j+4} */
    const __m512 c = _mm512_add_ps(b, _mm512_permute_ps(b, 0x0E));                           /* b0+b2, b1+b3 */
    return _mm512_add_ps(c, _mm512_permute_ps(c, 0x01));                                      /* c0+c1 */
}
/* outputs [p0*2, p1*2) of y (rows r0..r0+nr-1), I % 32 == 0; called inside the caller's parallel loop over pairs */
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma")))
static void v41_mv8_pair(float *y, int ystride, const uint8_t *q, const uint8_t *s, int I, int o,
                         const float *x, int xstride, int r0, int nr) {
    const int tiles_i = I / 32;
    const uint8_t *row0 = q + (size_t)o * I, *row1 = row0 + I;
    const uint8_t *scale = s + (size_t)(o / 32) * tiles_i;   /* o even: o and o+1 in the same 32-row block */
    float s0[32], s1[32];
    for (int r = 0; r < nr; r++) { s0[r] = 0.0f; s1[r] = 0.0f; }
    for (int t = 0; t < tiles_i; t++) {
        const int base = t * 32;
        const union { uint32_t u; float f; } tl = { (uint32_t)scale[t] << 23 };
        const float tile = tl.f;
        const __m512 w0 = v41_e4_dec16(row0 + base, row1 + base);
        const __m512 w1 = v41_e4_dec16(row0 + base + 8, row1 + base + 8);
        const __m512 w2 = v41_e4_dec16(row0 + base + 16, row1 + base + 16);
        const __m512 w3 = v41_e4_dec16(row0 + base + 24, row1 + base + 24);
        for (int r = 0; r < nr; r++) {
            const float *xr = x + (size_t)(r0 + r) * xstride + base;
            __m512 acc = _mm512_mul_ps(w0, _mm512_broadcast_f32x8(_mm256_loadu_ps(xr)));
            acc = _mm512_fmadd_ps(w1, _mm512_broadcast_f32x8(_mm256_loadu_ps(xr + 8)), acc);
            acc = _mm512_fmadd_ps(w2, _mm512_broadcast_f32x8(_mm256_loadu_ps(xr + 16)), acc);
            acc = _mm512_fmadd_ps(w3, _mm512_broadcast_f32x8(_mm256_loadu_ps(xr + 24)), acc);
            const __m512 h = v41_hsum2(acc);
            const float h0 = _mm512_cvtss_f32(h), h1 = _mm512_cvtss_f32(_mm512_shuffle_f32x4(h, h, 2));
            s0[r] = __builtin_fmaf(h0, tile, s0[r]);   /* the AVX2 loop's `sum += h * tile`, fused */
            s1[r] = __builtin_fmaf(h1, tile, s1[r]);
        }
    }
    for (int r = 0; r < nr; r++) { y[(size_t)(r0 + r) * ystride + o] = s0[r]; y[(size_t)(r0 + r) * ystride + o + 1] = s1[r]; }
}
/* quant.h's matmul_mxfp4 (I % 32 == 0 branch) for outputs o and o+1 over all S rows, with its bits. That loop, per
 * output and row: acc = 0; per 32-column group: the nibbles to doubled e2m1 int8 (pshufb), ga = x0*w0 (a multiply),
 * fma(x1,w1,ga), fma(x2,w2,ga), fma(x3,w3,ga); acc = fma(ga, scale*0.5, acc); y = hsum256(acc). Here the low half is
 * o and the high half o+1 -- each with its own group scale -- and every lane goes through the same operations. */
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma")))
static void v41_mx4_pair(float *y, const float *x, const uint8_t *q4, const uint8_t *e8s, int S, int I, int O, int o) {
    const int rb = (I + 1) / 2, ng = (I + 31) / 32;
    const uint8_t *w0 = q4 + (int64_t)o * rb, *w1 = w0 + rb, *c0 = e8s + (int64_t)o * ng, *c1 = c0 + ng;
    const __m128i lut2 = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), m4 = _mm_set1_epi8(0x0F);
    for (int s0 = 0; s0 < S; s0 += 8) {   /* up to 8 rows share each group's decode; each row keeps its own chain */
        const int ns = S - s0 < 8 ? S - s0 : 8;
        __m512 acc[8];
        for (int r = 0; r < ns; r++) acc[r] = _mm512_setzero_ps();
        for (int g = 0; g < ng; g++) {
            const __m128i b0 = _mm_loadu_si128((const __m128i *)(w0 + g * 16)), b1 = _mm_loadu_si128((const __m128i *)(w1 + g * 16));
            const __m128i l0 = _mm_and_si128(b0, m4), h0 = _mm_and_si128(_mm_srli_epi16(b0, 4), m4);
            const __m128i l1 = _mm_and_si128(b1, m4), h1 = _mm_and_si128(_mm_srli_epi16(b1, 4), m4);
            const __m128i a0 = _mm_shuffle_epi8(lut2, _mm_unpacklo_epi8(l0, h0)), a1 = _mm_shuffle_epi8(lut2, _mm_unpackhi_epi8(l0, h0));
            const __m128i e0 = _mm_shuffle_epi8(lut2, _mm_unpacklo_epi8(l1, h1)), e1 = _mm_shuffle_epi8(lut2, _mm_unpackhi_epi8(l1, h1));
            /* 8 columns of o (low) and the same 8 of o+1 (high) */
            const __m512 wk0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpacklo_epi64(a0, e0)));
            const __m512 wk1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpackhi_epi64(a0, e0)));
            const __m512 wk2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpacklo_epi64(a1, e1)));
            const __m512 wk3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_unpackhi_epi64(a1, e1)));
            const union { uint32_t u; float f; } c0v = { (uint32_t)c0[g] << 23 }, c1v = { (uint32_t)c1[g] << 23 };
            const __m512 sc = _mm512_insertf32x8(_mm512_set1_ps(c0v.f * 0.5f), _mm256_set1_ps(c1v.f * 0.5f), 1);
            for (int r = 0; r < ns; r++) {
                const float *xg = x + (int64_t)(s0 + r) * I + g * 32;
                __m512 ga = _mm512_mul_ps(_mm512_broadcast_f32x8(_mm256_loadu_ps(xg)), wk0);
                ga = _mm512_fmadd_ps(_mm512_broadcast_f32x8(_mm256_loadu_ps(xg + 8)), wk1, ga);
                ga = _mm512_fmadd_ps(_mm512_broadcast_f32x8(_mm256_loadu_ps(xg + 16)), wk2, ga);
                ga = _mm512_fmadd_ps(_mm512_broadcast_f32x8(_mm256_loadu_ps(xg + 24)), wk3, ga);
                acc[r] = _mm512_fmadd_ps(ga, sc, acc[r]);
            }
        }
        for (int r = 0; r < ns; r++) {
            const __m512 h = v41_hsum2(acc[r]);
            y[(int64_t)(s0 + r) * O + o] = _mm512_cvtss_f32(h);
            y[(int64_t)(s0 + r) * O + o + 1] = _mm512_cvtss_f32(_mm512_shuffle_f32x4(h, h, 2));
        }
    }
}
#else
#define V41_MV8_512_HAVE 0
#endif
