/* Local A/B for st.h's bf16_to_f32_bulk/f16_to_f32_bulk vectorized tiers vs the
 * scalar per-element reference (bf16_to_f32/f16_to_f32) they replace at the
 * st_read_f32/st_read_slice_f32 call sites. Not a test gate (see bench_idot,
 * bench_mla_simd, bench_gemv_stream for the same pattern) -- build on demand:
 *
 *   make tests/bench_st_f16_bf16_simd ARCH=native
 *   ./tests/bench_st_f16_bf16_simd
 */
#include "../st.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { N = 4 * 1024 * 1024, REPS = 5 };

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double scalar_bf16(const uint16_t *src, float *dst, int64_t n) {
    double t0 = now_s();
    for (int64_t i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
    return now_s() - t0;
}
static double scalar_f16(const uint16_t *src, float *dst, int64_t n) {
    double t0 = now_s();
    for (int64_t i = 0; i < n; i++) dst[i] = f16_to_f32(src[i]);
    return now_s() - t0;
}
static double bulk_bf16(const uint16_t *src, float *dst, int64_t n) {
    double t0 = now_s(); bf16_to_f32_bulk(src, dst, n); return now_s() - t0;
}
static double bulk_f16(const uint16_t *src, float *dst, int64_t n) {
    double t0 = now_s(); f16_to_f32_bulk(src, dst, n); return now_s() - t0;
}

int main(void) {
    uint16_t *src = malloc((size_t)N * sizeof(uint16_t));
    float *a = malloc((size_t)N * sizeof(float));
    float *b = malloc((size_t)N * sizeof(float));
    if (!src || !a || !b) { fprintf(stderr, "OOM\n"); return 2; }
    /* representative mix: mostly normal-range values (real weight magnitudes),
     * a scattering of exact zeros, occasional subnormal/inf/nan bit patterns --
     * not the uniform all-65536-once sweep the exactness test uses. */
    for (int64_t i = 0; i < N; i++) {
        int64_t m = i % 97;
        if (m == 0) src[i] = 0;
        else if (m == 1) src[i] = 0x7C00;         /* +inf */
        else if (m == 2) src[i] = 0x03FF;         /* max subnormal */
        else src[i] = (uint16_t)((i * 2654435761u) & 0x7BFF);
    }

    printf("st f16/bf16 simd bench: N=%d elements (%.1f MiB src)\n", N, N * sizeof(uint16_t) / 1048576.0);

    double ts = 0, tb = 0;
    for (int r = 0; r < REPS; r++) {
        if (r & 1) { tb += bulk_bf16(src, b, N); ts += scalar_bf16(src, a, N); }
        else       { ts += scalar_bf16(src, a, N); tb += bulk_bf16(src, b, N); }
    }
    printf("bf16->f32: scalar %.6f s  bulk %.6f s  speedup %.2fx\n", ts / REPS, tb / REPS, ts / tb);

    ts = 0; tb = 0;
    for (int r = 0; r < REPS; r++) {
        if (r & 1) { tb += bulk_f16(src, b, N); ts += scalar_f16(src, a, N); }
        else       { ts += scalar_f16(src, a, N); tb += bulk_f16(src, b, N); }
    }
    printf("f16->f32:  scalar %.6f s  bulk %.6f s  speedup %.2fx\n", ts / REPS, tb / REPS, ts / tb);

    free(src); free(a); free(b);
    return 0;
}
