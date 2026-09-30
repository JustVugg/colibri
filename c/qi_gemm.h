/* qi_gemm.h -- the one GEMM of the image engine.
 *
 * Y[M][N] = X[M][K] . W[N][K]^T (+ bias[N]), X and Y f32 row-major, W row-major
 * in one of three storage formats: f32, bf16, or int8 with one f32 scale per
 * row. Every linear layer of the text encoder and the DiT, every convolution
 * of the VAE (through im2col) and both products of the attention go through
 * here, so this is where the time of an image goes.
 *
 * Why not matmul_q_batch: the text engines' kernels are GEMVs that stream the
 * weight row once per output and sweep every activation row under it. That is
 * right for decode (one or two rows) and wrong for a DiT step, where M is the
 * number of image tokens (1296 at 768x432): the activations stop fitting in
 * cache and each output row re-reads all of them. Here W is dequantized into
 * f32 panels of NR rows x KC columns, X is walked in blocks of MC rows that stay
 * in L2, and a register tile of MR x NR accumulates in FMA registers.
 *
 * The weights stay int8 or bf16 in memory; the f32 copy only ever exists as one
 * packed panel per thread. Activations stay f32: no activation quantization, so
 * the result is the f32 product of the dequantized weights up to summation
 * order. */
#ifndef COLIBRI_QI_GEMM_H
#define COLIBRI_QI_GEMM_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

enum { QI_F32 = 0, QI_BF16 = 1, QI_I8 = 2 };

typedef struct {
    int fmt;            /* QI_F32 | QI_BF16 | QI_I8 */
    int N, K;           /* rows (outputs) x columns (inputs) */
    const void *w;      /* [N][K] in fmt */
    const float *sc;    /* QI_I8: one scale per row; NULL otherwise */
    int ld;             /* row stride in elements, 0 = K. Lets attention read one head's
                         * columns of K straight out of the [T][heads*hd] projection. */
} QiMat;

#define QI_MR 6
#define QI_NR 16
#define QI_KC 256
#define QI_MC 192

static inline float qi_bf16(uint16_t h){ union { uint32_t u; float f; } v; v.u = (uint32_t)h << 16; return v.f; }

/* Pack W rows [n0, n0+nr) x columns [k0, k0+kc) as f32, k-major: panel[k*QI_NR + j].
 * Rows past nr are zero so the kernel never branches on the edge. The int8 scale
 * is applied here, so the kernel is the same for every format. */
static void qi_pack_w(float *panel, const QiMat *W, int n0, int nr, int k0, int kc){
    for (int j = 0; j < QI_NR; j++) {
        if (j >= nr) { for (int k = 0; k < kc; k++) panel[k*QI_NR + j] = 0.f; continue; }
        int64_t row = (int64_t)(n0 + j) * (W->ld ? W->ld : W->K) + k0;
        if (W->fmt == QI_F32) {
            const float *s = (const float *)W->w + row;
            for (int k = 0; k < kc; k++) panel[k*QI_NR + j] = s[k];
        } else if (W->fmt == QI_BF16) {
            const uint16_t *s = (const uint16_t *)W->w + row;
            for (int k = 0; k < kc; k++) panel[k*QI_NR + j] = qi_bf16(s[k]);
        } else {
            const int8_t *s = (const int8_t *)W->w + row;
            float sc = W->sc[n0 + j];
            for (int k = 0; k < kc; k++) panel[k*QI_NR + j] = (float)s[k] * sc;
        }
    }
}

/* acc[MR][NR] (+)= X[m0.., k0..k0+kc) . panel ; mr <= MR rows valid */
static void qi_kernel(float *Y, int ldy, const float *X, int ldx, const float *panel,
                      int kc, int mr, int nr, int accumulate){
#if defined(__AVX2__) && defined(__FMA__)
    if (mr == QI_MR) {
        __m256 c00=_mm256_setzero_ps(), c01=_mm256_setzero_ps(), c10=_mm256_setzero_ps(), c11=_mm256_setzero_ps();
        __m256 c20=_mm256_setzero_ps(), c21=_mm256_setzero_ps(), c30=_mm256_setzero_ps(), c31=_mm256_setzero_ps();
        __m256 c40=_mm256_setzero_ps(), c41=_mm256_setzero_ps(), c50=_mm256_setzero_ps(), c51=_mm256_setzero_ps();
        const float *x0 = X, *x1 = X + ldx, *x2 = X + 2*ldx, *x3 = X + 3*ldx, *x4 = X + 4*ldx, *x5 = X + 5*ldx;
        for (int k = 0; k < kc; k++) {
            __m256 b0 = _mm256_loadu_ps(panel + k*QI_NR), b1 = _mm256_loadu_ps(panel + k*QI_NR + 8);
            __m256 a;
            a = _mm256_broadcast_ss(x0 + k); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
            a = _mm256_broadcast_ss(x1 + k); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
            a = _mm256_broadcast_ss(x2 + k); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
            a = _mm256_broadcast_ss(x3 + k); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
            a = _mm256_broadcast_ss(x4 + k); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
            a = _mm256_broadcast_ss(x5 + k); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        }
        float t[QI_MR][QI_NR];
        _mm256_storeu_ps(t[0], c00); _mm256_storeu_ps(t[0]+8, c01);
        _mm256_storeu_ps(t[1], c10); _mm256_storeu_ps(t[1]+8, c11);
        _mm256_storeu_ps(t[2], c20); _mm256_storeu_ps(t[2]+8, c21);
        _mm256_storeu_ps(t[3], c30); _mm256_storeu_ps(t[3]+8, c31);
        _mm256_storeu_ps(t[4], c40); _mm256_storeu_ps(t[4]+8, c41);
        _mm256_storeu_ps(t[5], c50); _mm256_storeu_ps(t[5]+8, c51);
        for (int i = 0; i < QI_MR; i++)
            for (int j = 0; j < nr; j++)
                Y[(int64_t)i*ldy + j] = accumulate ? Y[(int64_t)i*ldy + j] + t[i][j] : t[i][j];
        return;
    }
#endif
    float t[QI_MR][QI_NR];
    memset(t, 0, sizeof t);
    for (int k = 0; k < kc; k++) {
        const float *b = panel + k*QI_NR;
        for (int i = 0; i < mr; i++) {
            float a = X[(int64_t)i*ldx + k];
            for (int j = 0; j < QI_NR; j++) t[i][j] += a * b[j];
        }
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            Y[(int64_t)i*ldy + j] = accumulate ? Y[(int64_t)i*ldy + j] + t[i][j] : t[i][j];
}

/* Y[M][N] = X[M][K] . W^T (+ bias). ldx/ldy are the row strides (K and N for
 * dense rows; larger when X or Y is a column slice of a wider matrix). */
static void qi_gemm_ld(float *Y, int ldy, const float *X, int ldx, int M, const QiMat *W, const float *bias){
    const int N = W->N, K = W->K;
    if (M <= 0 || N <= 0) return;
    const int nblocks = (N + QI_NR - 1) / QI_NR;
    const int mblocks = (M + QI_MC - 1) / QI_MC;
    /* Parallel over (m-block, n-panel) tiles; each tile walks all of K, so the
     * sum for one output is always accumulated in the same order: the result
     * does not depend on the thread count. */
    #pragma omp parallel
    {
        /* plain malloc: the kernel loads unaligned, and aligned_alloc is missing from
         * the Windows CRT */
        float *panel = (float *)malloc((size_t)QI_KC * QI_NR * sizeof(float));
        if (!panel) { fprintf(stderr, "OOM qi_gemm panel\n"); exit(1); }
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (int mb = 0; mb < mblocks; mb++)
            for (int nb = 0; nb < nblocks; nb++) {
                int m0 = mb * QI_MC, mc = M - m0 < QI_MC ? M - m0 : QI_MC;
                int n0 = nb * QI_NR, nr = N - n0 < QI_NR ? N - n0 : QI_NR;
                for (int k0 = 0; k0 < K; k0 += QI_KC) {
                    int kc = K - k0 < QI_KC ? K - k0 : QI_KC;
                    qi_pack_w(panel, W, n0, nr, k0, kc);
                    for (int i = 0; i < mc; i += QI_MR) {
                        int mr = mc - i < QI_MR ? mc - i : QI_MR;
                        qi_kernel(Y + (int64_t)(m0 + i) * ldy + n0, ldy,
                                  X + (int64_t)(m0 + i) * ldx + k0, ldx,
                                  panel, kc, mr, nr, k0 > 0);
                    }
                }
                if (bias)
                    for (int i = 0; i < mc; i++)
                        for (int j = 0; j < nr; j++) Y[(int64_t)(m0 + i) * ldy + n0 + j] += bias[n0 + j];
            }
        free(panel);
    }
}

static inline void qi_gemm(float *Y, const float *X, int M, const QiMat *W, const float *bias){
    qi_gemm_ld(Y, W->N, X, W->K, M, W, bias);
}

/* ---- int8 x int8 (opt-in): the activations quantized per row too ---------
 *
 * With VNNI one vpdpbusd does four multiply-adds per 32-bit lane where an FMA
 * does one, so a DiT step is compute-bound on a quarter of the instructions.
 * The price is an 8-bit activation (one scale per row, amax/127): that is a
 * quality question, measured against the oracle, not assumed. vpdpbusd wants
 * unsigned x signed bytes, so x is stored as q+128 and 128*sum(w) comes off the
 * int32 sum at the end. The int32 sum over the whole K cannot overflow:
 * K * 255 * 127 stays under 2^31 up to K = 66000.
 *
 * Only for QI_I8 weights and only where VNNI exists (AVX-VNNI, or AVX-512 VNNI
 * with VL); qi_gemm_act8() falls back to qi_gemm() otherwise. */
#if (defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))) && defined(__AVX2__)
#define QI_HAVE_VNNI 1
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define QI_DPBUSD(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#else
#define QI_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#endif

/* W rows [n0, n0+nr), all K (padded to Kp, a multiple of 4), as [Kp/4][16][4]
 * bytes, plus the sum of each row for the +128 correction. */
static void qi_pack_w8(int8_t *panel, int32_t *wsum, const QiMat *W, int n0, int nr, int Kp){
    const int K = W->K, ld = W->ld ? W->ld : K;
    for (int j = 0; j < QI_NR; j++) {
        int32_t s = 0;
        const int8_t *r = j < nr ? (const int8_t *)W->w + (int64_t)(n0 + j) * ld : NULL;
        for (int k = 0; k < Kp; k++) {
            int8_t v = r && k < K ? r[k] : 0;
            panel[(k >> 2) * (QI_NR * 4) + j * 4 + (k & 3)] = v;
            s += v;
        }
        wsum[j] = s;
    }
}

static void qi_kernel8(float *Y, int ldy, const uint8_t *X, int Kp, const int8_t *panel,
                       const int32_t *wsum, const float *sx, const float *sw, int mr, int nr, const float *bias){
    __m256i c[QI_MR][2];
    for (int i = 0; i < QI_MR; i++) c[i][0] = c[i][1] = _mm256_setzero_si256();
    const int kq = Kp >> 2;
    if (mr == QI_MR) {
        for (int q = 0; q < kq; q++) {
            __m256i b0 = _mm256_loadu_si256((const __m256i *)(panel + q * 64));
            __m256i b1 = _mm256_loadu_si256((const __m256i *)(panel + q * 64 + 32));
            for (int i = 0; i < QI_MR; i++) {
                int32_t a4; memcpy(&a4, X + (int64_t)i * Kp + q * 4, 4);
                __m256i a = _mm256_set1_epi32(a4);
                c[i][0] = QI_DPBUSD(c[i][0], a, b0);
                c[i][1] = QI_DPBUSD(c[i][1], a, b1);
            }
        }
    } else {
        for (int q = 0; q < kq; q++) {
            __m256i b0 = _mm256_loadu_si256((const __m256i *)(panel + q * 64));
            __m256i b1 = _mm256_loadu_si256((const __m256i *)(panel + q * 64 + 32));
            for (int i = 0; i < mr; i++) {
                int32_t a4; memcpy(&a4, X + (int64_t)i * Kp + q * 4, 4);
                __m256i a = _mm256_set1_epi32(a4);
                c[i][0] = QI_DPBUSD(c[i][0], a, b0);
                c[i][1] = QI_DPBUSD(c[i][1], a, b1);
            }
        }
    }
    int32_t t[QI_NR];
    for (int i = 0; i < mr; i++) {
        _mm256_storeu_si256((__m256i *)t, c[i][0]);
        _mm256_storeu_si256((__m256i *)(t + 8), c[i][1]);
        for (int j = 0; j < nr; j++)
            Y[(int64_t)i * ldy + j] = (float)(t[j] - 128 * wsum[j]) * sx[i] * sw[j] + (bias ? bias[j] : 0.f);
    }
}
#endif

/* Y = X . W^T with X quantized to int8 per row. W must be QI_I8. */
static void qi_gemm_act8(float *Y, int ldy, const float *X, int ldx, int M, const QiMat *W, const float *bias){
#ifdef QI_HAVE_VNNI
    const int N = W->N, K = W->K, Kp = (K + 3) & ~3;
    if (W->fmt != QI_I8 || M <= 0 || N <= 0) { qi_gemm_ld(Y, ldy, X, ldx, M, W, bias); return; }
    uint8_t *xq = (uint8_t *)malloc((size_t)M * Kp);
    float *sx = (float *)malloc(sizeof(float) * (size_t)M);
    if (!xq || !sx) { fprintf(stderr, "OOM qi_gemm_act8\n"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *r = X + (int64_t)m * ldx;
        float am = 0.f;
        for (int k = 0; k < K; k++) { float a = r[k] < 0 ? -r[k] : r[k]; if (a > am) am = a; }
        float s = am > 1e-30f ? am / 127.f : 1.f, inv = 1.f / s;
        sx[m] = s;
        uint8_t *d = xq + (int64_t)m * Kp;
        for (int k = 0; k < K; k++) { float v = r[k] * inv; int iv = (int)(v < 0 ? v - 0.5f : v + 0.5f); d[k] = (uint8_t)(iv + 128); }
        for (int k = K; k < Kp; k++) d[k] = 128;
    }
    const int nblocks = (N + QI_NR - 1) / QI_NR, mblocks = (M + QI_MC - 1) / QI_MC;
    #pragma omp parallel
    {
        int8_t *panel = (int8_t *)malloc((size_t)Kp * QI_NR);
        int32_t wsum[QI_NR];
        if (!panel) { fprintf(stderr, "OOM qi_gemm_act8 panel\n"); exit(1); }
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (int nb = 0; nb < nblocks; nb++)
            for (int mb = 0; mb < mblocks; mb++) {
                int n0 = nb * QI_NR, nr = N - n0 < QI_NR ? N - n0 : QI_NR;
                int m0 = mb * QI_MC, mc = M - m0 < QI_MC ? M - m0 : QI_MC;
                qi_pack_w8(panel, wsum, W, n0, nr, Kp);
                for (int i = 0; i < mc; i += QI_MR) {
                    int mr = mc - i < QI_MR ? mc - i : QI_MR;
                    qi_kernel8(Y + (int64_t)(m0 + i) * ldy + n0, ldy, xq + (int64_t)(m0 + i) * Kp, Kp,
                               panel, wsum, sx + m0 + i, W->sc + n0, mr, nr, bias ? bias + n0 : NULL);
                }
            }
        free(panel);
    }
    free(xq); free(sx);
#else
    qi_gemm_ld(Y, ldy, X, ldx, M, W, bias);
#endif
}

/* Per-row int8 quantization of an f32 matrix (max-abs / 127, round to nearest). */
static void qi_quantize_i8(const float *src, int N, int K, int8_t *q, float *sc){
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float *r = src + (int64_t)n * K;
        float am = 0.f;
        for (int k = 0; k < K; k++) { float a = r[k] < 0 ? -r[k] : r[k]; if (a > am) am = a; }
        float s = am > 1e-12f ? am / 127.f : 1.f, inv = 1.f / s;
        sc[n] = s;
        int8_t *d = q + (int64_t)n * K;
        for (int k = 0; k < K; k++) {
            float v = r[k] * inv;
            int iv = (int)(v < 0 ? v - 0.5f : v + 0.5f);
            if (iv > 127) iv = 127;
            if (iv < -127) iv = -127;
            d[k] = (int8_t)iv;
        }
    }
}

#endif
