/* fmt=6 (E8/IQ3 lattice) encoder — SIMD/OpenMP core for tools/iq3_pack.py.
 *
 * Same algorithm as the numpy encoder, byte for byte: per 32-weight sub-block,
 * try all 16 sub-scale codes, and for each take the nearest of the 256 codebook
 * entries for every 4-dim group, keeping the code with the lowest squared error.
 *
 * The numpy version is memory-bound, not compute-bound: it materialises a
 * [groups, 256] score array per candidate code, so every score is written to
 * memory and read back by argmin — ~12 KB of traffic per weight, which measured
 * out at ~10 GB/s, i.e. exactly one core's share of bandwidth. Here the whole
 * working set for a sub-block is the 8x256 float table of m.g (8 KB, L1-resident)
 * and the scores never leave registers, so the same arithmetic runs at cache
 * speed instead of DRAM speed.
 *
 * The search is written against the identity
 *     argmin_g ||m/db - g||^2 = argmin_g [ db*||g||^2 - 2*(m.g) ]        (db > 0)
 * so m.g is computed once per sub-block and shared by all 16 codes.
 *
 * The codebook is passed in from Python rather than duplicated here, so
 * tools/iq3xxs_grid.json stays the single source of truth for this side.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define QK 256        /* weights per super-block */
#define SUB 32        /* weights per sign/scale word */
#define NG 256        /* codebook entries */
#define BB 98         /* packed bytes per super-block */
#define GRP (SUB / 4) /* 4-dim groups per sub-block = 8 */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* IEEE754 binary32 -> binary16, round-to-nearest-even. Matches numpy's
 * astype(float16) exactly, including the carry out of the mantissa into the
 * exponent; the encoder must agree with numpy here because the super-scale is
 * stored as fp16 and then read back for the search. */
static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e = (x >> 23) & 0xffu;
    uint32_t man = x & 0x7fffffu;
    if (e == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? (0x200u | (man >> 13)) : 0u));
    int32_t ex = (int32_t)e - 127 + 15;
    if (ex >= 0x1f) return (uint16_t)(sign | 0x7c00u); /* overflow -> inf */
    if (ex <= 0) {
        if (ex < -10) return (uint16_t)sign; /* underflow -> +-0 */
        man |= 0x800000u;
        uint32_t sh = (uint32_t)(14 - ex);
        uint32_t r = man >> sh, rem = man & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) r++;
        return (uint16_t)(sign | r);
    }
    uint32_t r = man >> 13, rem = man & 0x1fffu;
    uint32_t o = ((uint32_t)ex << 10) | r;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) o++; /* carries into ex */
    return (uint16_t)(sign | o);
}

/* binary16 -> binary32, exact (numpy semantics, including subnormals). */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (!e) {
        if (!man) bits = sign;
        else { /* subnormal -> normalise */ int s = 0;
            uint32_t m = man;
            while (!(m & 0x400u)) {
                m <<= 1;
                s++;
            }
            bits = sign | ((uint32_t)(127 - 15 - s) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 0x1fu) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((e + 112u) << 23) | (man << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Nearest codebook entry for each of the 8 groups of one sub-block, under a
 * given db. crit[j] = db*G2[j] - 2*(m.g_j); A2 already holds -2*(m.g).
 * Ties resolve to the LOWEST index, as numpy's argmin does. */
static inline void nearest8(const float *A2, const float *G2, float db, int *idx, float *val) {
#if defined(__AVX2__)
    const __m256 vdb = _mm256_set1_ps(db);
    for (int g = 0; g < GRP; g++) {
        const float *a = A2 + (size_t)g * NG;
        __m256 best = _mm256_set1_ps(FLT_MAX);
        __m256i bi = _mm256_setzero_si256();
        __m256i jv = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i eight = _mm256_set1_epi32(8);
        for (int j = 0; j < NG; j += 8) {
            __m256 v = _mm256_fmadd_ps(vdb, _mm256_loadu_ps(G2 + j), _mm256_loadu_ps(a + j));
            __m256 lt = _mm256_cmp_ps(v, best, _CMP_LT_OQ); /* strict: first wins */
            best = _mm256_blendv_ps(best, v, lt);
            bi = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(bi), _mm256_castsi256_ps(jv), lt));
            jv = _mm256_add_epi32(jv, eight);
        }
        /* horizontal argmin over the 8 lanes; lane k held indices k, k+8, ... so
         * an exact tie must still resolve to the smaller index. */
        float bv[8];
        int bx[8];
        _mm256_storeu_ps(bv, best);
        _mm256_storeu_si256((__m256i *)bx, bi);
        float bestv = bv[0];
        int besti = bx[0];
        for (int k = 1; k < 8; k++)
            if (bv[k] < bestv || (bv[k] == bestv && bx[k] < besti)) {
                bestv = bv[k];
                besti = bx[k];
            }
        idx[g] = besti;
        val[g] = bestv;
    }
#else
    for (int g = 0; g < GRP; g++) {
        const float *a = A2 + (size_t)g * NG;
        float bestv = FLT_MAX;
        int besti = 0;
        for (int j = 0; j < NG; j++) {
            float v = db * G2[j] + a[j];
            if (v < bestv) {
                bestv = v;
                besti = j;
            }
        }
        idx[g] = besti;
        val[g] = bestv;
    }
#endif
}

/* One row, one super-block. `gs` is the codebook as 4 planes of 256 (SoA), G2
 * the squared norms, blk the 256 input weights, out the 98 packed bytes. */
static void encode_block(const float *blk, const float *const gs[4], const float *G2, uint8_t *out) {
    float mag[QK];
    uint8_t neg[QK];
    for (int i = 0; i < QK; i++) {
        neg[i] = blk[i] < 0.0f; /* -0.0 counts as positive, as numpy's where does */
        mag[i] = fabsf(blk[i]);
    }
    /* Odd-parity fix: the 8th sign of every eight is implied, so a group whose
     * true signs multiply to -1 gets its SMALLEST magnitude flipped (first
     * minimum on ties, matching argmin). */
    for (int b = 0; b < QK / 8; b++) {
        uint8_t *n = neg + b * 8;
        const float *m = mag + b * 8;
        int cnt = 0;
        for (int j = 0; j < 8; j++) cnt += n[j];
        if (cnt & 1) {
            int am = 0;
            for (int j = 1; j < 8; j++)
                if (m[j] < m[am]) am = j;
            n[am] ^= 1;
        }
    }
    /* super-scale: RMS anchor, stored fp16 and read back so the search sees the
     * value that will actually be decoded */
    double acc = 0.0;
    for (int i = 0; i < QK; i++) acc += (double)mag[i] * (double)mag[i];
    float d = sqrtf((float)(acc / QK)) / 20.0f + 1e-12f;
    uint16_t dh = f32_to_f16(d);
    memcpy(out + 96, &dh, 2);
    d = f16_to_f32(dh);

    for (int ib = 0; ib < QK / SUB; ib++) {
        const float *m = mag + ib * SUB;
        /* A2[g][j] = -2 * (m_g . g_j) — computed once, shared by all 16 codes */
        float A2[GRP * NG];
        for (int g = 0; g < GRP; g++) {
            const float m0 = m[g * 4 + 0], m1 = m[g * 4 + 1], m2 = m[g * 4 + 2], m3 = m[g * 4 + 3];
            float *a = A2 + (size_t)g * NG;
#if defined(__AVX2__)
            const __m256 v0 = _mm256_set1_ps(-2.0f * m0), v1 = _mm256_set1_ps(-2.0f * m1),
                         v2 = _mm256_set1_ps(-2.0f * m2), v3 = _mm256_set1_ps(-2.0f * m3);
            for (int j = 0; j < NG; j += 8) {
                __m256 s = _mm256_mul_ps(v0, _mm256_loadu_ps(gs[0] + j));
                s = _mm256_fmadd_ps(v1, _mm256_loadu_ps(gs[1] + j), s);
                s = _mm256_fmadd_ps(v2, _mm256_loadu_ps(gs[2] + j), s);
                s = _mm256_fmadd_ps(v3, _mm256_loadu_ps(gs[3] + j), s);
                _mm256_storeu_ps(a + j, s);
            }
#else
            for (int j = 0; j < NG; j++) a[j] = -2.0f * (m0 * gs[0][j] + m1 * gs[1][j] + m2 * gs[2][j] + m3 * gs[3][j]);
#endif
        }
        /* Choosing the code needs only the SUM of the 8 group minima — the
         * indices matter for the winner alone. So the 16 candidate passes track
         * a bare minimum (one _mm256_min_ps, no compare/blend and no index
         * bookkeeping) and the indices are recovered in a single extra pass at
         * the end: 17 cheap passes instead of 16 expensive ones.
         * err = sum_g [db^2*G2 - 2*db*(m.g)] = db * sum_g [db*G2 - 2*(m.g)]
         * = db * (sum of the minima), which is the quantity minimised. */
        int bestcode = 0;
        float besterr = 0.0f;
        for (int code = 0; code < 16; code++) {
            float db = d * (0.5f + (float)code) * 0.5f;
            if (db < 1e-20f) db = 1e-20f;
            float sum = 0.0f;
#if defined(__AVX2__)
            const __m256 vdb = _mm256_set1_ps(db);
            for (int g = 0; g < GRP; g++) {
                const float *a = A2 + (size_t)g * NG;
                /* four independent chains: a single running min over 32 vectors
                 * is a 32-deep latency chain, which is what left the search
                 * far short of the SIMD width */
                __m256 m0 = _mm256_set1_ps(FLT_MAX), m1 = m0, m2 = m0, m3 = m0;
                for (int j = 0; j < NG; j += 32) {
                    m0 = _mm256_min_ps(m0, _mm256_fmadd_ps(vdb, _mm256_loadu_ps(G2 + j), _mm256_loadu_ps(a + j)));
                    m1 = _mm256_min_ps(m1,
                                       _mm256_fmadd_ps(vdb, _mm256_loadu_ps(G2 + j + 8), _mm256_loadu_ps(a + j + 8)));
                    m2 = _mm256_min_ps(m2,
                                       _mm256_fmadd_ps(vdb, _mm256_loadu_ps(G2 + j + 16), _mm256_loadu_ps(a + j + 16)));
                    m3 = _mm256_min_ps(m3,
                                       _mm256_fmadd_ps(vdb, _mm256_loadu_ps(G2 + j + 24), _mm256_loadu_ps(a + j + 24)));
                }
                m0 = _mm256_min_ps(_mm256_min_ps(m0, m1), _mm256_min_ps(m2, m3));
                __m128 lo = _mm256_castps256_ps128(m0), hi = _mm256_extractf128_ps(m0, 1);
                lo = _mm_min_ps(lo, hi);
                lo = _mm_min_ps(lo, _mm_movehl_ps(lo, lo));
                lo = _mm_min_ss(lo, _mm_shuffle_ps(lo, lo, 1));
                sum += _mm_cvtss_f32(lo);
            }
#else
            for (int g = 0; g < GRP; g++) {
                const float *a = A2 + (size_t)g * NG;
                float b0 = FLT_MAX, b1 = FLT_MAX, b2 = FLT_MAX, b3 = FLT_MAX;
                for (int j = 0; j < NG; j += 4) {
                    float v0 = db * G2[j] + a[j];
                    if (v0 < b0) b0 = v0;
                    float v1 = db * G2[j + 1] + a[j + 1];
                    if (v1 < b1) b1 = v1;
                    float v2 = db * G2[j + 2] + a[j + 2];
                    if (v2 < b2) b2 = v2;
                    float v3 = db * G2[j + 3] + a[j + 3];
                    if (v3 < b3) b3 = v3;
                }
                float lo0 = b0 < b1 ? b0 : b1, lo1 = b2 < b3 ? b2 : b3;
                sum += lo0 < lo1 ? lo0 : lo1;
            }
#endif
            float err = db * sum;
            if (code == 0 || err < besterr) {
                besterr = err;
                bestcode = code;
            }
        }
        { /* one index pass for the winning code */
            float db = d * (0.5f + (float)bestcode) * 0.5f;
            if (db < 1e-20f) db = 1e-20f;
            int idx[GRP];
            float val[GRP];
            nearest8(A2, G2, db, idx, val);
            for (int g = 0; g < GRP; g++) out[ib * 8 + g] = (uint8_t)idx[g];
        }

        /* four 7-bit sign words + the 4-bit sub-scale code */
        uint32_t word = 0;
        for (int l = 0; l < 4; l++) {
            uint32_t seven = 0;
            const uint8_t *n = neg + ib * SUB + l * 8;
            for (int j = 0; j < 7; j++) seven |= (uint32_t)n[j] << j;
            word |= seven << (7 * l);
        }
        word |= ((uint32_t)bestcode & 0xfu) << 28;
        uint8_t wb[4] = {(uint8_t)word, (uint8_t)(word >> 8), (uint8_t)(word >> 16),
                         (uint8_t)(word >> 24)}; /* little-endian */
        memcpy(out + QK / 4 + ib * 4, wb, 4);
    }
}

/* rows: [nrow, K] float32, K % 256 == 0. out: [nrow, K/256*98] uint8.
 * grid: [256*4] float32 in weight units (the published table already halved). */
EXPORT void iq3_encode(const float *rows, int64_t nrow, int64_t K, const float *grid, uint8_t *out) {
    float gp[4][NG], G2[NG];
    for (int j = 0; j < NG; j++) {
        for (int c = 0; c < 4; c++) gp[c][j] = grid[j * 4 + c];
        G2[j] = gp[0][j] * gp[0][j] + gp[1][j] * gp[1][j] + gp[2][j] * gp[2][j] + gp[3][j] * gp[3][j];
    }
    const float *gs[4] = {gp[0], gp[1], gp[2], gp[3]};
    const int64_t nsb = K / QK;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < nrow; r++)
        for (int64_t sb = 0; sb < nsb; sb++) encode_block(rows + r * K + sb * QK, gs, G2, out + r * nsb * BB + sb * BB);
}

/* Reported by the Python side so a stale/mismatched build is visible. */
EXPORT int iq3_encode_abi(void) { return 1; }
