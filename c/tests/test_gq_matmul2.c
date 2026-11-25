/* Bit-exactness and timing test for single-region gq_matmul2_act across R16,
 * TILE16_BIAS, and F8 paths: every output must match two gq_matmul_act calls
 * to the bit. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../ggml_quants.h"

static uint32_t g_rng = 0x2468ace1u;
static uint32_t xrand32(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}
static float frand(void) { return (float)(xrand32() & 0xffffffu) / 8388608.0f - 1.0f; }
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

static void fill_valid_blocks(uint8_t *buf, size_t nbytes) {
    for (size_t i = 0; i < nbytes; i++) buf[i] = (uint8_t)(xrand32() & 0xffu);
    /* Avoid NaN/Inf in any f16 scale header: clear exponent top bit in every 16-bit lane */
    for (size_t i = 0; i + 1 < nbytes; i += 2) buf[i + 1] &= 0x3fu;
}

static void check_pair(const char *label, int t1, int t2, int S, int I, int O, int pack_r16) {
    int b1 = pack_r16 ? gq_r16_base(t1) : t1;
    int b2 = pack_r16 ? gq_r16_base(t2) : t2;
    int64_t rb1 = gq_row_bytes(b1, I), rb2 = gq_row_bytes(b2, I);
    int rows1 = pack_r16 ? ((O + 15) / 16) * 16 : O;
    uint8_t *raw1 = (uint8_t *)malloc((size_t)rows1 * (size_t)rb1);
    uint8_t *raw2 = (uint8_t *)malloc((size_t)rows1 * (size_t)rb2);
    fill_valid_blocks(raw1, (size_t)rows1 * (size_t)rb1);
    fill_valid_blocks(raw2, (size_t)rows1 * (size_t)rb2);
    uint8_t *W1 = raw1, *W2 = raw2;
#ifdef GQ_R16_ON
    if (pack_r16) {
        int64_t p1 = gq_row_bytes(t1, I), p2 = gq_row_bytes(t2, I);
        W1 = (uint8_t *)malloc((size_t)rows1 * (size_t)p1);
        W2 = (uint8_t *)malloc((size_t)rows1 * (size_t)p2);
        gq_r16_pack(t1, W1, raw1, rows1, I);
        gq_r16_pack(t2, W2, raw2, rows1, I);
    }
#endif
    float *x = (float *)malloc((size_t)S * (size_t)I * sizeof(float));
    for (int i = 0; i < S * I; i++) x[i] = frand();
    gq_act a; memset(&a, 0, sizeof a);
    CHECK(gq_act_set(&a, x, S, I) == 0, "gq_act_set failed");

    size_t ybytes = (size_t)S * (size_t)O * sizeof(float);
    float *y1_got = (float *)malloc(ybytes), *y2_got = (float *)malloc(ybytes);
    float *y1_ref = (float *)malloc(ybytes), *y2_ref = (float *)malloc(ybytes);

    gq_matmul2_act(y1_got, W1, t1, y2_got, W2, t2, &a, O);
    gq_matmul_act(y1_ref, &a, W1, t1, O);
    gq_matmul_act(y2_ref, &a, W2, t2, O);

    CHECK(memcmp(y1_got, y1_ref, ybytes) == 0, "%s y1 mismatch (S=%d O=%d)", label, S, O);
    CHECK(memcmp(y2_got, y2_ref, ybytes) == 0, "%s y2 mismatch (S=%d O=%d)", label, S, O);
    if (!g_failures) printf("  ok  %-24s (S=%2d I=%d O=%d)\n", label, S, I, O);

    gq_act_free(&a);
    free(x); free(y1_got); free(y2_got); free(y1_ref); free(y2_ref);
    if (W1 != raw1) free(W1);
    if (W2 != raw2) free(W2);
    free(raw1); free(raw2);
}

static void run_bench(void) {
    const int I = 2048, O = 512;   /* MoE expert gate+up shape */
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("\nBenchmark gq_matmul2_act vs 2x gq_matmul_act (I=%d O=%d threads=%d):\n", I, O, threads);
    struct { const char *name; int t; int S; int iters; } cases[] = {
        { "Q4_K   F8     S=16", GQ_Q4_K,   16, 40 },
        { "IQ2_S  TILE16 S=16", GQ_IQ2_S,  16, 20 },
        { "Q4_K   R16    S= 1", GQ_Q4_K_R16, 1, 200 },
    };
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        int t = cases[ci].t, S = cases[ci].S, iters = cases[ci].iters;
        int is_r16 = gq_r16_base(t) >= 0;
        int b = is_r16 ? gq_r16_base(t) : t;
        int64_t rb = gq_row_bytes(b, I);
        if (rb <= 0) continue;
        uint8_t *raw1 = (uint8_t *)malloc((size_t)O * (size_t)rb);
        uint8_t *raw2 = (uint8_t *)malloc((size_t)O * (size_t)rb);
        fill_valid_blocks(raw1, (size_t)O * (size_t)rb);
        fill_valid_blocks(raw2, (size_t)O * (size_t)rb);
        uint8_t *W1 = raw1, *W2 = raw2;
#ifdef GQ_R16_ON
        if (is_r16) {
            int64_t pb = gq_row_bytes(t, I);
            W1 = (uint8_t *)malloc((size_t)O * (size_t)pb);
            W2 = (uint8_t *)malloc((size_t)O * (size_t)pb);
            gq_r16_pack(t, W1, raw1, O, I);
            gq_r16_pack(t, W2, raw2, O, I);
        }
#endif
        float *x = (float *)malloc((size_t)S * (size_t)I * sizeof(float));
        for (int i = 0; i < S * I; i++) x[i] = frand();
        gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, I);
        float *y1 = (float *)malloc((size_t)S * (size_t)O * sizeof(float));
        float *y2 = (float *)malloc((size_t)S * (size_t)O * sizeof(float));

        gq_matmul_act(y1, &a, W1, t, O); gq_matmul_act(y2, &a, W2, t, O);
        double t0 = now_s();
        for (int i = 0; i < iters; i++) { gq_matmul_act(y1, &a, W1, t, O); gq_matmul_act(y2, &a, W2, t, O); }
        double ref_ms = (now_s() - t0) * 1000.0 / iters;

        gq_matmul2_act(y1, W1, t, y2, W2, t, &a, O);
        t0 = now_s();
        for (int i = 0; i < iters; i++) gq_matmul2_act(y1, W1, t, y2, W2, t, &a, O);
        double new_ms = (now_s() - t0) * 1000.0 / iters;

        printf("  %-20s: 2x=%6.3f ms  single=%6.3f ms  speedup=%.2fx\n",
               cases[ci].name, ref_ms, new_ms, ref_ms / new_ms);
        gq_act_free(&a); free(x); free(y1); free(y2);
        if (W1 != raw1) free(W1);
        if (W2 != raw2) free(W2);
        free(raw1); free(raw2);
    }
}

int main(int argc, char **argv) {
    printf("Testing single-region gq_matmul2_act (bit-exactness)...\n");
    check_pair("Q4_K F8 S=16 aligned",   GQ_Q4_K,     GQ_Q4_K,     16, 512, 64, 0);
    check_pair("Q4_K F8 S=16 tail",      GQ_Q4_K,     GQ_Q4_K,     16, 512, 53, 0);
    check_pair("IQ2_S TILE16 S=16",      GQ_IQ2_S,    GQ_IQ2_S,    16, 512, 64, 0);
    check_pair("IQ2_S TILE16 S=16 tail", GQ_IQ2_S,    GQ_IQ2_S,    16, 512, 53, 0);
    check_pair("IQ3_S TILE16 S=12",      GQ_IQ3_S,    GQ_IQ3_S,    12, 512, 48, 0);
#ifdef GQ_R16_ON
    check_pair("Q4_K_R16 S=1 aligned",   GQ_Q4_K_R16, GQ_Q4_K_R16,  1, 512, 64, 1);
    check_pair("Q4_K_R16 S=1 tail",      GQ_Q4_K_R16, GQ_Q4_K_R16,  1, 512, 53, 1);
    check_pair("Q4_K_R16 S=16 aligned",  GQ_Q4_K_R16, GQ_Q4_K_R16, 16, 512, 64, 1);
#endif
    check_pair("mixed Q4_K + Q6_K S=16", GQ_Q4_K,     GQ_Q6_K,     16, 512, 40, 0);
    if (argc > 1 && !strcmp(argv[1], "bench")) run_bench();
    if (g_failures) { fprintf(stderr, "%d failure(s)\n", g_failures); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
