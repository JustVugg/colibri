/* v4_np_route.h -- V4_ROUTER_AVX2=1: the DeepSeek V4 bf16 router with an 8-lane
 * AVX2 FMA gate GEMV, bit-identical to the scalar route (coli_v4_route_bf16_scalar,
 * the ROUTE_BF16 unit in deepseek_v4.c). The scalar oracle and every vector fallback
 * stay on the scalar loop and never dispatch back into the AVX2 wrapper.
 *
 * The vector sum adds in another order than the scalar loop, so its logits
 * differ in the last bits and a near tie at the K-th place could flip a guess.
 * To keep the guesses *identical* to the scalar path's, every fast logit
 * carries a rigorous bound on its distance from the scalar one, and the
 * experts the bounds cannot place clearly in or out of the top K (the few
 * around the K-th place) are recomputed by the scalar loop itself
 * (coli_v4_route_bf16_rows: the ROUTE_BF16 unit, route_bf16_impl's loop).
 * route_bf16_impl then selects as always (coli_v4_route_bf16_logits) from
 * logits that are bit-identical wherever they can matter: the same indices in
 * the same order, the same weights.
 *
 * Bound. Both are sums of the same terms w_c*h_c. A chain of m FMAs or adds
 * is within gamma(m) = m*u/(1-m*u) of the exact sum, times S = sum |w_c*h_c|
 * (u = 2^-24). The vector path chains at most cols/16 + 8 + cols%32 terms
 * (32 lanes x 4 accumulators, the reduction, the tail); the scalar path one
 * chain of cols terms (cols+1 if it rounds the products first). So
 *   |fast - scalar| <= (gamma(m) + gamma(cols+1)) * S,
 * and S, summed alongside from |w| (the bf16 sign bit cleared) and |h|, is
 * itself within gamma(m) (all terms >= 0). sqrt(softplus(x)) has slope under
 * 0.33 everywhere, so a selection value moves by at most half the logit
 * bound, plus a few ulps for the float evaluation (2e-6 relative is used).
 * With lo/hi the resulting selection interval of each expert and T the K-th
 * largest lo, the K experts holding those lo values beat every expert whose
 * hi is below T, in the scalar values and in the mixed ones alike; only the
 * experts with hi >= T can be in the top K, and those are the ones recomputed.
 *
 * Pure: no globals, no OS headers. tests/test_v4_np_route.c checks it against
 * coli_v4_route_bf16 on random, near-tie, exact-tie and real gate data. */
#ifndef V4_NP_ROUTE_H
#define V4_NP_ROUTE_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ROUTE_BF16 unit (deepseek_v4.c) */
int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale);
/* Explicit oracle/fallback: never dispatches back into the AVX2 wrapper. */
int coli_v4_route_bf16_scalar(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale);
int coli_v4_route_bf16_logits(float *weights, int *indices, const float *hidden,
                              const uint16_t *gate, const float *bias,
                              const int *forced_indices, int experts, int dimension,
                              int topk, float route_scale, const float *logits);
void coli_v4_route_bf16_rows(float *logits, const uint16_t *gate, const float *hidden,
                             const int *rows, int count, int dimension);

#define V4_NP_ROUTE_MAX_EXPERTS 256   /* stack arrays; more falls back to scalar */

static int v4_exact_router_enabled(int experts,int dimension,int topk) {
#if defined(__AVX2__) && defined(__FMA__) && !defined(COLI_VULKAN)
    const char *enabled=getenv("V4_ROUTER_AVX2");
    return enabled && atoi(enabled)!=0 && experts>=1 && experts<=V4_NP_ROUTE_MAX_EXPERTS &&
           dimension>=1 && dimension<=8192 && topk>=1 && topk<=experts;
#else
    (void)experts;(void)dimension;(void)topk;return 0;
#endif
}

#if defined(__AVX2__) && defined(__FMA__) && !defined(COLI_VULKAN)
/* (COLI_VULKAN: route_bf16_impl may take its logits from the device, which
 * the bound does not describe; the router keeps the scalar call there.) */
#include <immintrin.h>
#define V4_NP_ROUTE_AVX2 1

static inline __m256 v4_np_bf16x8(const uint16_t *p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}

static inline float v4_np_hsum8(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}

static inline float v4_np_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

/* logits[r] = gate row r . hidden, sums[r] = sum_c |gate[r][c] * hidden[c]|;
 * habs: `cols` floats of scratch (|hidden|). */
static void v4_np_gemv_bf16(float *logits, float *sums, const uint16_t *gate,
                            const float *hidden, float *habs, int rows, int cols) {
    const __m256 magnitude = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    int body = cols & ~31;
    int c = 0;
    for (; c + 31 < cols; c += 32) {
        __m256 h0 = _mm256_loadu_ps(hidden + c);
        __m256 h1 = _mm256_loadu_ps(hidden + c + 8);
        __m256 h2 = _mm256_loadu_ps(hidden + c + 16);
        __m256 h3 = _mm256_loadu_ps(hidden + c + 24);
        _mm256_storeu_ps(habs + c, _mm256_and_ps(h0, magnitude));
        _mm256_storeu_ps(habs + c + 8, _mm256_and_ps(h1, magnitude));
        _mm256_storeu_ps(habs + c + 16, _mm256_and_ps(h2, magnitude));
        _mm256_storeu_ps(habs + c + 24, _mm256_and_ps(h3, magnitude));
    }
    for (; c < cols; c++) habs[c] = fabsf(hidden[c]);

    int r = 0;
    for (; r + 1 < rows; r += 2) {
        const uint16_t *row0 = gate + (size_t)r * cols;
        const uint16_t *row1 = gate + (size_t)(r + 1) * cols;
        __m256 a0_0 = _mm256_setzero_ps(), a0_1 = a0_0, a0_2 = a0_0, a0_3 = a0_0;
        __m256 s0_0 = a0_0, s0_1 = a0_0;
        __m256 a1_0 = a0_0, a1_1 = a0_0, a1_2 = a0_0, a1_3 = a0_0;
        __m256 s1_0 = a0_0, s1_1 = a0_0;
        for (int col = 0; col < body; col += 32) {
            __m256 h0 = _mm256_loadu_ps(hidden + col);
            __m256 ha0 = _mm256_loadu_ps(habs + col);
            __m256 w0_0 = v4_np_bf16x8(row0 + col);
            __m256 w1_0 = v4_np_bf16x8(row1 + col);
            a0_0 = _mm256_fmadd_ps(w0_0, h0, a0_0);
            a1_0 = _mm256_fmadd_ps(w1_0, h0, a1_0);
            s0_0 = _mm256_fmadd_ps(_mm256_and_ps(w0_0, magnitude), ha0, s0_0);
            s1_0 = _mm256_fmadd_ps(_mm256_and_ps(w1_0, magnitude), ha0, s1_0);

            __m256 h1 = _mm256_loadu_ps(hidden + col + 8);
            __m256 ha1 = _mm256_loadu_ps(habs + col + 8);
            __m256 w0_1 = v4_np_bf16x8(row0 + col + 8);
            __m256 w1_1 = v4_np_bf16x8(row1 + col + 8);
            a0_1 = _mm256_fmadd_ps(w0_1, h1, a0_1);
            a1_1 = _mm256_fmadd_ps(w1_1, h1, a1_1);
            s0_1 = _mm256_fmadd_ps(_mm256_and_ps(w0_1, magnitude), ha1, s0_1);
            s1_1 = _mm256_fmadd_ps(_mm256_and_ps(w1_1, magnitude), ha1, s1_1);

            __m256 h2 = _mm256_loadu_ps(hidden + col + 16);
            __m256 ha2 = _mm256_loadu_ps(habs + col + 16);
            __m256 w0_2 = v4_np_bf16x8(row0 + col + 16);
            __m256 w1_2 = v4_np_bf16x8(row1 + col + 16);
            a0_2 = _mm256_fmadd_ps(w0_2, h2, a0_2);
            a1_2 = _mm256_fmadd_ps(w1_2, h2, a1_2);
            s0_0 = _mm256_fmadd_ps(_mm256_and_ps(w0_2, magnitude), ha2, s0_0);
            s1_0 = _mm256_fmadd_ps(_mm256_and_ps(w1_2, magnitude), ha2, s1_0);

            __m256 h3 = _mm256_loadu_ps(hidden + col + 24);
            __m256 ha3 = _mm256_loadu_ps(habs + col + 24);
            __m256 w0_3 = v4_np_bf16x8(row0 + col + 24);
            __m256 w1_3 = v4_np_bf16x8(row1 + col + 24);
            a0_3 = _mm256_fmadd_ps(w0_3, h3, a0_3);
            a1_3 = _mm256_fmadd_ps(w1_3, h3, a1_3);
            s0_1 = _mm256_fmadd_ps(_mm256_and_ps(w0_3, magnitude), ha3, s0_1);
            s1_1 = _mm256_fmadd_ps(_mm256_and_ps(w1_3, magnitude), ha3, s1_1);
        }
        float sum0 = v4_np_hsum8(_mm256_add_ps(_mm256_add_ps(a0_0, a0_1), _mm256_add_ps(a0_2, a0_3)));
        float sum1 = v4_np_hsum8(_mm256_add_ps(_mm256_add_ps(a1_0, a1_1), _mm256_add_ps(a1_2, a1_3)));
        float mag0 = v4_np_hsum8(_mm256_add_ps(s0_0, s0_1));
        float mag1 = v4_np_hsum8(_mm256_add_ps(s1_0, s1_1));
        for (int col = body; col < cols; col++) {
            float w0 = v4_np_bf16(row0[col]);
            float w1 = v4_np_bf16(row1[col]);
            sum0 += w0 * hidden[col];
            sum1 += w1 * hidden[col];
            mag0 += fabsf(w0) * habs[col];
            mag1 += fabsf(w1) * habs[col];
        }
        logits[r] = sum0;
        logits[r + 1] = sum1;
        sums[r] = mag0;
        sums[r + 1] = mag1;
    }
    for (; r < rows; r++) {
        const uint16_t *row = gate + (size_t)r * cols;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, s0 = a0, s1 = a0;
        for (int col = 0; col < body; col += 32) {
            __m256 w0 = v4_np_bf16x8(row + col), w1 = v4_np_bf16x8(row + col + 8);
            __m256 w2 = v4_np_bf16x8(row + col + 16), w3 = v4_np_bf16x8(row + col + 24);
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(hidden + col), a0);
            a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(hidden + col + 8), a1);
            a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(hidden + col + 16), a2);
            a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(hidden + col + 24), a3);
            s0 = _mm256_fmadd_ps(_mm256_and_ps(w0, magnitude), _mm256_loadu_ps(habs + col), s0);
            s1 = _mm256_fmadd_ps(_mm256_and_ps(w1, magnitude), _mm256_loadu_ps(habs + col + 8), s1);
            s0 = _mm256_fmadd_ps(_mm256_and_ps(w2, magnitude), _mm256_loadu_ps(habs + col + 16), s0);
            s1 = _mm256_fmadd_ps(_mm256_and_ps(w3, magnitude), _mm256_loadu_ps(habs + col + 24), s1);
        }
        float sum = v4_np_hsum8(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        float magnitude_sum = v4_np_hsum8(_mm256_add_ps(s0, s1));
        for (int col = body; col < cols; col++) {
            float w = v4_np_bf16(row[col]);
            sum += w * hidden[col];
            magnitude_sum += fabsf(w) * habs[col];
        }
        logits[r] = sum;
        sums[r] = magnitude_sum;
    }
}

/* What coli_v4_route_bf16(weights, indices, hidden, gate, bias, NULL, experts,
 * dimension, topk, route_scale) returns, bit for bit, at AVX2 speed. scratch:
 * `dimension` floats. *refined (if not NULL): the experts recomputed by the
 * scalar loop, or -1 when it fell back to the plain scalar route. */
static int v4_np_route_bf16_avx2(float *weights, int *indices, const float *hidden,
                                 const uint16_t *gate, const float *bias,
                                 int experts, int dimension, int topk,
                                 float route_scale, float *scratch, int *refined) {
    __attribute__((aligned(32))) float logits[V4_NP_ROUTE_MAX_EXPERTS], sums[V4_NP_ROUTE_MAX_EXPERTS];
    __attribute__((aligned(32))) float lo[V4_NP_ROUTE_MAX_EXPERTS], hi[V4_NP_ROUTE_MAX_EXPERTS];
    __attribute__((aligned(32))) float exact[V4_NP_ROUTE_MAX_EXPERTS], best[V4_NP_ROUTE_MAX_EXPERTS];
    __attribute__((aligned(32))) int rows[V4_NP_ROUTE_MAX_EXPERTS];
    if (refined) *refined = -1;
    if (!weights || !indices || !hidden || !gate || !scratch || experts < 1 ||
        experts > V4_NP_ROUTE_MAX_EXPERTS || dimension < 1 || dimension > (1 << 20) ||
        topk < 1 || topk > experts)
        return coli_v4_route_bf16_scalar(weights, indices, hidden, gate, bias, NULL,
                                  experts, dimension, topk, route_scale);
    v4_np_gemv_bf16(logits, sums, gate, hidden, scratch, experts, dimension);
    const double u = 1.0 / 16777216.0;
    double m = (double)(dimension / 16 + 8 + (dimension & 31));
    double n = (double)dimension + 1.0;
    double gm = m * u / (1.0 - m * u), gn = n * u / (1.0 - n * u);
    float scale = (float)((gm + gn) / (1.0 - gm) * 1.001);
    for (int e = 0; e < experts; e++) {
        float b = scale * sums[e] + 1e-35f;
        float shift = bias ? bias[e] : 0.0f;
        if (!isfinite(logits[e]) || !isfinite(b) || !isfinite(shift) ||
            fabsf(logits[e]) + b > 1e30f)
            return coli_v4_route_bf16_scalar(weights, indices, hidden, gate, bias, NULL,
                                      experts, dimension, topk, route_scale);
        float x = logits[e];
        float g = sqrtf(fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x))));
        float slack = 0.5f * b + 2e-6f * (g + fabsf(shift)) + 1e-30f;
        lo[e] = g + shift - slack;
        hi[e] = g + shift + slack;
    }
    /* T = the topk-th largest lo (insertion into a descending list) */
    int held = 0;
    for (int e = 0; e < experts; e++) {
        float v = lo[e];
        if (held == topk && !(v > best[held - 1])) continue;
        int at = held < topk ? held++ : topk - 1;
        while (at > 0 && best[at - 1] < v) { best[at] = best[at - 1]; at--; }
        best[at] = v;
    }
    float threshold = best[topk - 1];
    int count = 0;
    __m256 vthresh = _mm256_set1_ps(threshold);
    int e = 0;
    for (; e + 7 < experts; e += 8) {
        __m256 vhi = _mm256_load_ps(hi + e);
        __m256 cmp = _mm256_cmp_ps(vhi, vthresh, _CMP_GE_OQ);
        int mask = _mm256_movemask_ps(cmp);
        if (__builtin_expect(mask == 0, 1)) continue;
        while (mask) {
            int bit = __builtin_ctz(mask);
            rows[count++] = e + bit;
            mask &= mask - 1;
        }
    }
    for (; e < experts; e++)
        if (hi[e] >= threshold) rows[count++] = e;
    coli_v4_route_bf16_rows(exact, gate, hidden, rows, count, dimension);
    for (int i = 0; i < count; i++) logits[rows[i]] = exact[i];
    if (refined) *refined = count;
    return coli_v4_route_bf16_logits(weights, indices, hidden, gate, bias, NULL,
                                     experts, dimension, topk, route_scale, logits);
}
#endif /* AVX2 && FMA */

#endif /* V4_NP_ROUTE_H */
