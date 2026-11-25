/* Bit-exactness and timing test for fused F32/F16/BF16 paths in gq_matmul_f32
 * and vectorized gq_dequant_row (ggml_quants.h). Compares against the original
 * scalar-dequant + malloc-per-thread + single-row gq_dot_f32 reference. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../ggml_quants.h"

static uint32_t g_rng = 0x12345678u;
static uint32_t xrand32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
static float frand(void) {
    return (float)(xrand32() & 0xffffffu) / 8388608.0f - 1.0f;
}
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int g_failures;
#define CHECK(ok, ...) do { if (!(ok)) { \
    fprintf(stderr, "FAIL line %d: ", __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    g_failures++; } } while (0)

/* Original reference implementation before this optimization. */
static void dequant_row_ref(int t, const uint8_t *src, float *dst, int64_t n) {
    if (t == GQ_F32) { memcpy(dst, src, (size_t)n * 4); return; }
    if (t == GQ_F16) { for (int64_t i = 0; i < n; i++) dst[i] = gq_f16(gq_rd16(src + 2 * i)); return; }
    if (t == GQ_BF16) { for (int64_t i = 0; i < n; i++) dst[i] = gq_bf16(gq_rd16(src + 2 * i)); return; }
}

static void matmul_f32_ref(float *y, const float *x, const uint8_t *W, int t, int S, int I, int O) {
    const int64_t rb = gq_row_bytes(t, I);
    #pragma omp parallel
    {
        float *row = (float *)malloc((size_t)I * sizeof(float));
        #pragma omp for schedule(static)
        for (int o = 0; o < O; o++) {
            dequant_row_ref(t, W + (int64_t)o * rb, row, I);
            for (int s = 0; s < S; s++)
                y[(int64_t)s * O + o] = gq_dot_f32(row, x + (int64_t)s * I, I);
        }
        free(row);
    }
}

static void test_all_16bit_codes(void) {
    uint16_t *all = (uint16_t *)malloc(65536 * sizeof(uint16_t));
    float *got = (float *)malloc(65536 * sizeof(float));
    float *ref = (float *)malloc(65536 * sizeof(float));
    for (int i = 0; i < 65536; i++) all[i] = (uint16_t)i;

    /* F16: check all 65536 bit patterns (normals, subnormals, +-0, +-inf, NaNs) */
    gq_dequant_row(GQ_F16, (const uint8_t *)all, got, 65536);
    dequant_row_ref(GQ_F16, (const uint8_t *)all, ref, 65536);
    int bad = 0;
    for (int i = 0; i < 65536; i++) {
        if (isnan(got[i]) && isnan(ref[i])) continue;
        if (memcmp(&got[i], &ref[i], 4) != 0) { bad++; break; }
    }
    CHECK(bad == 0, "GQ_F16 dequant_row mismatch across 65536 codes");

    /* BF16: check all 65536 bit patterns */
    gq_dequant_row(GQ_BF16, (const uint8_t *)all, got, 65536);
    dequant_row_ref(GQ_BF16, (const uint8_t *)all, ref, 65536);
    bad = 0;
    for (int i = 0; i < 65536; i++) {
        if (isnan(got[i]) && isnan(ref[i])) continue;
        if (memcmp(&got[i], &ref[i], 4) != 0) { bad++; break; }
    }
    CHECK(bad == 0, "GQ_BF16 dequant_row mismatch across 65536 codes");

    free(all); free(got); free(ref);
    if (!g_failures) printf("  ok  dequant_row all 65536 codes (F16 + BF16)\n");
}

static uint16_t float_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}
static uint16_t float_to_f16_simple(float f) {
    /* Valid finite non-NaN f16 from random [-1, 1] floats */
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int exp = (int)((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3ffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7bffu);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static void check_matmul_case(int t, int S, int I, int O) {
    int64_t rb = gq_row_bytes(t, I);
    uint8_t *W = (uint8_t *)malloc((size_t)O * (size_t)rb);
    float *x = (float *)malloc((size_t)S * (size_t)I * sizeof(float));
    float *got = (float *)malloc((size_t)S * (size_t)O * sizeof(float));
    float *ref = (float *)malloc((size_t)S * (size_t)O * sizeof(float));
    for (int i = 0; i < S * I; i++) x[i] = frand();
    for (int o = 0; o < O; o++) {
        for (int i = 0; i < I; i++) {
            float v = frand();
            if (t == GQ_F32) {
                memcpy(W + (int64_t)o * rb + (int64_t)i * 4, &v, 4);
            } else if (t == GQ_F16) {
                uint16_t h = float_to_f16_simple(v);
                memcpy(W + (int64_t)o * rb + (int64_t)i * 2, &h, 2);
            } else {
                uint16_t h = float_to_bf16(v);
                memcpy(W + (int64_t)o * rb + (int64_t)i * 2, &h, 2);
            }
        }
    }
    CHECK(gq_matmul_f32(got, x, W, t, S, I, O) == 0, "gq_matmul_f32 returned error");
    matmul_f32_ref(ref, x, W, t, S, I, O);
    int exact = memcmp(got, ref, (size_t)S * (size_t)O * sizeof(float)) == 0;
    CHECK(exact, "type=%s S=%d I=%d O=%d not bit-identical", GQ_TYPES[t].name, S, I, O);
    free(W); free(x); free(got); free(ref);
}

static void run_bench(void) {
    const int I = 2048, O = 4096;
    const int types[] = { GQ_F32, GQ_F16, GQ_BF16 };
    const int Ss[] = { 1, 32 };
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("\nBenchmark (I=%d O=%d threads=%d):\n", I, O, threads);
    for (size_t ti = 0; ti < 3; ti++) {
        int t = types[ti];
        int64_t rb = gq_row_bytes(t, I);
        uint8_t *W = (uint8_t *)malloc((size_t)O * (size_t)rb);
        for (int o = 0; o < O; o++) {
            for (int i = 0; i < I; i++) {
                float v = frand();
                if (t == GQ_F32) memcpy(W + (int64_t)o * rb + (int64_t)i * 4, &v, 4);
                else if (t == GQ_F16) { uint16_t h = float_to_f16_simple(v); memcpy(W + (int64_t)o * rb + (int64_t)i * 2, &h, 2); }
                else { uint16_t h = float_to_bf16(v); memcpy(W + (int64_t)o * rb + (int64_t)i * 2, &h, 2); }
            }
        }
        for (size_t si = 0; si < 2; si++) {
            int S = Ss[si];
            int iters = S == 1 ? 60 : 12;
            float *x = (float *)malloc((size_t)S * (size_t)I * sizeof(float));
            float *ya = (float *)malloc((size_t)S * (size_t)O * sizeof(float));
            float *yb = (float *)malloc((size_t)S * (size_t)O * sizeof(float));
            for (int i = 0; i < S * I; i++) x[i] = frand();

            matmul_f32_ref(ya, x, W, t, S, I, O);
            double t0 = now_s();
            for (int it = 0; it < iters; it++) matmul_f32_ref(ya, x, W, t, S, I, O);
            double ref_ms = (now_s() - t0) * 1000.0 / iters;

            gq_matmul_f32(yb, x, W, t, S, I, O);
            t0 = now_s();
            for (int it = 0; it < iters; it++) gq_matmul_f32(yb, x, W, t, S, I, O);
            double new_ms = (now_s() - t0) * 1000.0 / iters;

            CHECK(memcmp(ya, yb, (size_t)S * (size_t)O * sizeof(float)) == 0,
                  "bench %s S=%d mismatch", GQ_TYPES[t].name, S);
            printf("  %-4s S=%2d: ref=%6.2f ms  fused=%6.2f ms  speedup=%.2fx\n",
                   GQ_TYPES[t].name, S, ref_ms, new_ms, ref_ms / new_ms);
            free(x); free(ya); free(yb);
        }
        free(W);
    }
}

int main(int argc, char **argv) {
    printf("Testing gq_dequant_row and gq_matmul_f32 (F32/F16/BF16, bit-exactness)...\n");
    test_all_16bit_codes();
    const int types[] = { GQ_F32, GQ_F16, GQ_BF16 };
    const int Ss[] = { 1, 2, 3, 4, 16, 33 };
    const int Is[] = { 17, 64, 255, 2048 };
    for (size_t ti = 0; ti < 3; ti++)
        for (size_t si = 0; si < 6; si++)
            for (size_t ii = 0; ii < 4; ii++)
                check_matmul_case(types[ti], Ss[si], Is[ii], 37);
    if (!g_failures) printf("  ok  72 matmul cases (F32/F16/BF16 x S={1,2,3,4,16,33} x I={17,64,255,2048})\n");
    if (argc > 1 && !strcmp(argv[1], "bench")) run_bench();
    if (g_failures) { fprintf(stderr, "%d failure(s)\n", g_failures); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
