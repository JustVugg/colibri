/* sparse_attn.h -- the kernel DeepSeek V4.1 Flash reads attention with.
 *
 * One query head against a LIST of positions rather than a prefix: the window's
 * ring slots plus whatever the DSA indexer kept, concatenated. A slot the query
 * cannot reach is -1 and scores -INFINITY, which is why the list is passed
 * instead of a length. The model is MQA, so every head reads the same kv rows;
 * `kv` is one row per position, `hd` wide.
 *
 * Two passes over those rows, both of them FMA loops: scores, then the weighted
 * sum. Four accumulators in the dot because a single dependent FMA chain runs at
 * the latency of the unit and not its throughput, and the rows are 512 wide on
 * the released checkpoint -- long enough for the difference to be the whole cost.
 *
 * Numerics: the vector path folds the dot into four partial sums where the
 * scalar path keeps one, and its multiply-adds are fused where the scalar ones
 * round twice. So the two agree to fp32 accumulation and not bit for bit -- the
 * same relationship mv8 already has with its own scalar fallback. Nothing else
 * differs: the exponentials, the sink and the final division are the same
 * operations in the same order, and the division stays a division so that it
 * rounds the way the scalar one does. tests/test_sparse_attn.c holds the two
 * against each other across widths, including the ones with a tail.
 *
 * Header-only, all static, no dependency beyond immintrin -- the quant.h and
 * fused_simd.h convention.
 */
#ifndef COLI_SPARSE_ATTN_H
#define COLI_SPARSE_ATTN_H

#include <math.h>
#include <stddef.h>

#if defined(__AVX2__)
#include <immintrin.h>

/* named apart from quant.h's hsum256: this header stands alone */
static inline float coli_sa_hsum(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    low = _mm_add_ps(low, high);
    low = _mm_add_ps(low, _mm_movehl_ps(low, low));
    low = _mm_add_ss(low, _mm_shuffle_ps(low, low, 0x55));
    return _mm_cvtss_f32(low);
}
#elif defined(__aarch64__)
#include <arm_neon.h>
#define COLI_SPARSE_ATTN_NEON 1
#endif

/* The reading of the kernel, one multiply at a time. Kept as the numerical oracle
 * and for targets without AVX2 or ARM NEON. */
static inline void coli_sparse_attend_scalar(float *out, const float *q, const float *kv,
                                             const int *idx, int count, int hd,
                                             float sink, float scale, float *score) {
    float best = -1e30f;
    for (int j = 0; j < count; j++) {
        if (idx[j] < 0) { score[j] = -INFINITY; continue; }
        const float *k = kv + (size_t)idx[j] * hd;
        float dot = 0.0f;
        for (int i = 0; i < hd; i++) dot += q[i] * k[i];
        score[j] = dot * scale;
        if (score[j] > best) best = score[j];
    }
    float denom = expf(sink - best);
    for (int i = 0; i < hd; i++) out[i] = 0.0f;
    for (int j = 0; j < count; j++) {
        if (!(score[j] > -INFINITY)) continue;
        float weight = expf(score[j] - best);
        denom += weight;
        const float *k = kv + (size_t)idx[j] * hd;
        for (int i = 0; i < hd; i++) out[i] += weight * k[i];
    }
    for (int i = 0; i < hd; i++) out[i] /= denom;
}

static inline void coli_sparse_attend(float *out, const float *q, const float *kv,
                                      const int *idx, int count, int hd,
                                      float sink, float scale, float *score) {
#if !defined(__AVX2__) && !defined(COLI_SPARSE_ATTN_NEON)
    coli_sparse_attend_scalar(out, q, kv, idx, count, hd, sink, scale, score);
#elif defined(__AVX2__)
    float best = -1e30f;
    for (int j = 0; j < count; j++) {
        if (idx[j] < 0) { score[j] = -INFINITY; continue; }
        const float *k = kv + (size_t)idx[j] * hd;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 32 <= hd; i += 32) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i),      _mm256_loadu_ps(k + i),      a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8),  _mm256_loadu_ps(k + i + 8),  a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 16), _mm256_loadu_ps(k + i + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 24), _mm256_loadu_ps(k + i + 24), a3);
        }
        for (; i + 8 <= hd; i += 8)
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k + i), a0);
        float dot = coli_sa_hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        for (; i < hd; i++) dot += q[i] * k[i];
        score[j] = dot * scale;
        if (score[j] > best) best = score[j];
    }
    float denom = expf(sink - best);
    int i = 0;
    for (; i + 8 <= hd; i += 8) _mm256_storeu_ps(out + i, _mm256_setzero_ps());
    for (; i < hd; i++) out[i] = 0.0f;
    for (int j = 0; j < count; j++) {
        if (!(score[j] > -INFINITY)) continue;
        float weight = expf(score[j] - best);
        denom += weight;
        const float *k = kv + (size_t)idx[j] * hd;
        __m256 w = _mm256_set1_ps(weight);
        i = 0;
        for (; i + 8 <= hd; i += 8)
            _mm256_storeu_ps(out + i, _mm256_fmadd_ps(w, _mm256_loadu_ps(k + i),
                                                      _mm256_loadu_ps(out + i)));
        for (; i < hd; i++) out[i] += weight * k[i];
    }
    __m256 d = _mm256_set1_ps(denom);
    i = 0;
    for (; i + 8 <= hd; i += 8)
        _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(out + i), d));
    for (; i < hd; i++) out[i] /= denom;
#else
    float best = -1e30f;
    for (int j = 0; j < count; j++) {
        if (idx[j] < 0) { score[j] = -INFINITY; continue; }
        const float *k = kv + (size_t)idx[j] * hd;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 16 <= hd; i += 16) {
            a0 = vfmaq_f32(a0, vld1q_f32(q + i),      vld1q_f32(k + i));
            a1 = vfmaq_f32(a1, vld1q_f32(q + i + 4),  vld1q_f32(k + i + 4));
            a2 = vfmaq_f32(a2, vld1q_f32(q + i + 8),  vld1q_f32(k + i + 8));
            a3 = vfmaq_f32(a3, vld1q_f32(q + i + 12), vld1q_f32(k + i + 12));
        }
        float dot = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        for (; i < hd; i++) dot += q[i] * k[i];
        score[j] = dot * scale;
        if (score[j] > best) best = score[j];
    }
    float denom = expf(sink - best);
    int i = 0;
    for (; i + 4 <= hd; i += 4) vst1q_f32(out + i, vdupq_n_f32(0.0f));
    for (; i < hd; i++) out[i] = 0.0f;
    for (int j = 0; j < count; j++) {
        if (!(score[j] > -INFINITY)) continue;
        float weight = expf(score[j] - best);
        denom += weight;
        const float *k = kv + (size_t)idx[j] * hd;
        float32x4_t w = vdupq_n_f32(weight);
        i = 0;
        for (; i + 4 <= hd; i += 4)
            vst1q_f32(out + i, vfmaq_f32(vld1q_f32(out + i), w, vld1q_f32(k + i)));
        for (; i < hd; i++) out[i] += weight * k[i];
    }
    float32x4_t d = vdupq_n_f32(denom);
    i = 0;
    for (; i + 4 <= hd; i += 4)
        vst1q_f32(out + i, vdivq_f32(vld1q_f32(out + i), d));
    for (; i < hd; i++) out[i] /= denom;
#endif
}

#endif /* COLI_SPARSE_ATTN_H */
