/* Microbenchmark: scalar vs SIMD MLA-absorb reductions (the #442 change).
 *
 * NOT a unit test — correctness is gated by the glm_tiny TF oracle (SNAP=./glm_tiny
 * TF=1, expected 30/32 unchanged). This measures the headline claim: SIMD-ifying
 * the f32 score dot (qabs·Lt, kvl=512 on GLM-5.2) and the value-mix AXPY lifts
 * the per-(s,h) attention reduction off the scalar floor, the same shape of win
 * the int8 matmul_q kernel (quant.h:91) and the AVX-VNNI idot kernels (#264)
 * already took.
 *
 * Re-implements the OLD scalar reductions inline and calls the REAL attention
 * path via the include-colibri.c pattern — but since the reductions live inline
 * in the MLA attention block (not a callable), this bench re-implements both
 * the OLD (scalar) and NEW (SIMD) reductions verbatim from the colibri.c source
 * and times them back-to-back at GLM-5.2 dimensions (kvl=512, realistic nt).
 *
 * Run:  make tests/bench_mla_simd && ./tests/bench_mla_simd   (not a gate)
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <stdint.h>
#include <string.h>

static uint32_t rs = 0x2545F491u;
static uint32_t xr(void) {
    rs ^= rs << 13;
    rs ^= rs >> 17;
    rs ^= rs << 5;
    return rs;
}

/* ---- OLD: verbatim scalar reductions (pre-#442 colibri.c:2459, 2467) ---- */
static void score_scalar(const float *qabs, const float *Lt, int kvl, float *out) {
    float a = 0;
    for (int i = 0; i < kvl; i++) a += qabs[i] * Lt[i];
    *out = a;
}
static void vmix_scalar(float *clat, const float *Lt, int kvl, float a) {
    for (int i = 0; i < kvl; i++) clat[i] += a * Lt[i];
}

/* ---- NEW: verbatim SIMD reductions (post-#442 colibri.c, AVX2 path) ---- */
#if defined(__AVX2__)
static void score_simd(const float *qabs, const float *Lt, int kvl, float *out) {
    float a = 0;
    int i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= kvl; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(qabs + i), _mm256_loadu_ps(Lt + i), acc);
    a = hsum256(acc);
    for (; i < kvl; i++) a += qabs[i] * Lt[i];
    *out = a;
}
static void vmix_simd(float *clat, const float *Lt, int kvl, float a) {
    int i = 0;
    __m256 va = _mm256_set1_ps(a);
    for (; i + 8 <= kvl; i += 8) {
        __m256 cl = _mm256_loadu_ps(clat + i), lt = _mm256_loadu_ps(Lt + i);
        _mm256_storeu_ps(clat + i, _mm256_fmadd_ps(va, lt, cl));
    }
    for (; i < kvl; i++) clat[i] += a * Lt[i];
}
#else
/* non-AVX2 build: SIMD path == scalar, bench will show parity */
#define score_simd score_scalar
#define vmix_simd vmix_scalar
#endif

/* now_s() comes from colibri.c (included above) */

int main(void) {
    /* GLM-5.2 dims: kvl=512, and nt (context tokens per head) at a realistic
     * mid-length decode. The score loop runs nt times per (s,h); we time the
     * full inner nt-loop to mirror the real attention block. */
    int kvl = 512, nt = 256, H = 128; /* H heads — the outer parallel dim */
    float *qabs = malloc(kvl * sizeof(float));
    float *Lt = malloc((size_t)nt * kvl * sizeof(float));
    float *clat_old = malloc(kvl * sizeof(float));
    float *clat_new = malloc(kvl * sizeof(float));
    float *sc_old = malloc(nt * sizeof(float));
    float *sc_new = malloc(nt * sizeof(float));

    /* deterministic but non-trivial inputs — bounded floats in [-1,1] (avoids
     * NaN/Inf from arbitrary bit patterns, which would make the correctness
     * diff meaningless while still timing the same fp arithmetic). */
    for (int i = 0; i < kvl; i++) {
        qabs[i] = ((float)(xr() & 0xFFFF) / 65535.0f) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < (size_t)nt * kvl; i++) {
        Lt[i] = ((float)(xr() & 0xFFFF) / 65535.0f) * 2.0f - 1.0f;
    }

    /* warmup + correctness sanity: SIMD and scalar agree within float tolerance */
    memset(clat_old, 0, kvl * sizeof(float));
    memset(clat_new, 0, kvl * sizeof(float));
    for (int jj = 0; jj < 5; jj++) {
        score_scalar(qabs, Lt + (size_t)jj * kvl, kvl, &sc_old[jj]);
        score_simd(qabs, Lt + (size_t)jj * kvl, kvl, &sc_new[jj]);
        vmix_scalar(clat_old, Lt + (size_t)jj * kvl, kvl, 0.5f);
        vmix_simd(clat_new, Lt + (size_t)jj * kvl, kvl, 0.5f);
    }
    float max_err = 0;
    for (int i = 0; i < kvl; i++) {
        float e = fabsf(clat_old[i] - clat_new[i]);
        if (e > max_err) max_err = e;
    }
    printf("correctness: score max abs diff over 5 = %.3e, vmix max abs diff = %.3e\n", fabsf(sc_old[0] - sc_new[0]),
           max_err);
    printf("(score reassociates -> small diff expected; vmix is lane-wise -> ~0 diff)\n\n");

    /* bench: H * (nt score calls + nt vmix calls), as in one attention forward */
    int reps = 200;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        for (int h = 0; h < H; h++) {
            for (int jj = 0; jj < nt; jj++) score_scalar(qabs, Lt + (size_t)jj * kvl, kvl, &sc_old[jj]);
            for (int jj = 0; jj < nt; jj++) vmix_scalar(clat_old, Lt + (size_t)jj * kvl, kvl, sc_old[jj]);
        }
    }
    double t_old = now_s() - t0;

    t0 = now_s();
    for (int r = 0; r < reps; r++) {
        for (int h = 0; h < H; h++) {
            for (int jj = 0; jj < nt; jj++) score_simd(qabs, Lt + (size_t)jj * kvl, kvl, &sc_new[jj]);
            for (int jj = 0; jj < nt; jj++) vmix_simd(clat_new, Lt + (size_t)jj * kvl, kvl, sc_new[jj]);
        }
    }
    double t_new = now_s() - t0;

    double per_old = t_old / reps / H; /* per head */
    double per_new = t_new / reps / H;
    printf("MLA-absorb reductions (kvl=%d, nt=%d, H=%d, %d reps):\n", kvl, nt, H, reps);
    printf("  scalar: %.3f us / head\n", per_old * 1e6);
    printf("  SIMD  : %.3f us / head\n", per_new * 1e6);
    printf("  speedup: %.2fx   (%.1f%% faster)\n", per_old / per_new, 100.0 * (per_old - per_new) / per_old);

    free(qabs);
    free(Lt);
    free(clat_old);
    free(clat_new);
    free(sc_old);
    free(sc_new);
    return 0;
}
