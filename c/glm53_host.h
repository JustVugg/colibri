/* GLM-5.3 decode host experts, native adjacent-nibble int4 weights.
 * One parallel region for gate/up, clamped SwiGLU and down. No repacking.
 * kernel 0: original i4g_dot; 1: bit-exact AVX2 g64 pipeline.
 */
#ifndef COLIBRI_GLM53_HOST_H
#define COLIBRI_GLM53_HOST_H
#include "quant.h"

typedef struct { const uint8_t *q4; const float *s; } G53HostMat;

#if defined(__AVX2__) && defined(__FMA__)
/* Preserve i4g_dot's eight lane FMAs and its horizontal tree per group. A
 * 32-byte load unpacks the full group; signed byte widening saves the eight
 * int32 bias subtractions. The unpack lane order is 0,16,32,48 below. */
static inline __m256 g53h_group64(const uint8_t *w, const float *x) {
    const __m256i mask = _mm256_set1_epi8(15), bias = _mm256_set1_epi8(8);
    __m256i b = _mm256_loadu_si256((const __m256i *)w);
    __m256i lo = _mm256_and_si256(b, mask), hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), mask);
    __m256i n0 = _mm256_sub_epi8(_mm256_unpacklo_epi8(lo, hi), bias);
    __m256i n1 = _mm256_sub_epi8(_mm256_unpackhi_epi8(lo, hi), bias);
    __m128i p0 = _mm256_castsi256_si128(n0), p1 = _mm256_castsi256_si128(n1);
    __m128i p2 = _mm256_extracti128_si256(n0, 1), p3 = _mm256_extracti128_si256(n1, 1);
    __m256 a = _mm256_setzero_ps();
#define G53H_FMA(N, X) a = _mm256_fmadd_ps(_mm256_loadu_ps(x + (X)), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(N)), a)
    G53H_FMA(p0, 0);  G53H_FMA(_mm_srli_si128(p0, 8), 8);
    G53H_FMA(p1, 16); G53H_FMA(_mm_srli_si128(p1, 8), 24);
    G53H_FMA(p2, 32); G53H_FMA(_mm_srli_si128(p2, 8), 40);
    G53H_FMA(p3, 48); G53H_FMA(_mm_srli_si128(p3, 8), 56);
#undef G53H_FMA
    return a;
}
static inline float g53h_dot64(const uint8_t *w, const float *sc, const float *x, int I) {
    float a = 0.f; int g = 0, ng = I / 64;
    /* Independent groups expose instruction parallelism; the scalar scale
     * FMAs still retire in the original group order, with identical rounding. */
    for (; g + 4 <= ng; g += 4) {
        __m256 p[4];
        for (int j = 0; j < 4; j++) p[j] = g53h_group64(w + (size_t)(g + j) * 32, x + (g + j) * 64);
        for (int j = 0; j < 4; j++) a = fmaf(hsum256(p[j]), sc[g + j], a);
    }
    for (; g < ng; g++) a = fmaf(hsum256(g53h_group64(w + (size_t)g * 32, x + g * 64)), sc[g], a);
    return a;
}

#endif

static int g53h_compute(int n, int I, int Dn, int D, int gs, const G53HostMat *g,
                        const G53HostMat *u, const G53HostMat *d, const float *x,
                        float limit, float *const *dst, int kernel) {
#if defined(__SSE4_1__) && !defined(__AVX2__)
    /* The existing four-row SSE kernel uses a different reduction order.
     * Decline fusion so the caller preserves that platform's original bits. */
    return 0;
#endif
    /* Scalar tails can contract differently when the row helper is inlined
     * into different OpenMP workers (observed with GCC 16 on Windows). Keep
     * unsupported groups and dimensions on the original per-expert path. */
    if (gs != 64 || Dn % 64 || I % 64) return 0;
    const int rbg = (Dn + 1) / 2, ngg = (Dn + gs - 1) / gs;
    const int rbd = (I + 1) / 2, ngd = (I + gs - 1) / gs;
#if defined(__AVX2__) && defined(__FMA__)
    if (kernel < 0 || kernel > 1) kernel = 0;
#else
    (void)kernel;  /* No specialized kernel is compiled on this target. */
#endif
    float *buf = malloc((size_t)2 * n * I * sizeof(float));
    if (!buf) return 0;
    float *sgb = buf, *sub = buf + (size_t)n * I;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int z = 0; z < 2 * n * I; z++) {
            const int e = z / (2 * I), r = z % (2 * I), up = r >= I, o = up ? r - I : r;
            const G53HostMat *w = up ? &u[e] : &g[e];
            const uint8_t *q = w->q4 + (size_t)o * rbg; const float *s = w->s + (size_t)o * ngg;
            float a;
#if defined(__AVX2__) && defined(__FMA__)
            if (kernel == 1) a = g53h_dot64(q, s, x, Dn);
            else
#endif
                a = i4g_dot(q, s, x, Dn, gs);
            (up ? sub : sgb)[(size_t)e * I + o] = a;
        }
        #pragma omp for schedule(static)
        for (int e = 0; e < n; e++) {
            float *ge = sgb + (size_t)e * I; const float *ue = sub + (size_t)e * I;
            for (int i = 0; i < I; i++) {
                float gv = ge[i] > limit ? limit : ge[i];
                float uv = ue[i] < -limit ? -limit : (ue[i] > limit ? limit : ue[i]);
                ge[i] = (gv / (1.f + expf(-gv))) * uv;
            }
        }
        #pragma omp for schedule(static)
        for (int z = 0; z < n * D; z++) {
            const int e = z / D, o = z % D;
            const uint8_t *q = d[e].q4 + (size_t)o * rbd; const float *s = d[e].s + (size_t)o * ngd;
#if defined(__AVX2__) && defined(__FMA__)
            if (kernel == 1) dst[e][o] = g53h_dot64(q, s, sgb + (size_t)e * I, I);
            else
#endif
                dst[e][o] = i4g_dot(q, s, sgb + (size_t)e * I, I, gs);
        }
    }
    free(buf);
    return 1;
}
#endif
