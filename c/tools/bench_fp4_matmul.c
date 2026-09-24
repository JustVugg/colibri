/* bench_fp4_matmul.c — microbench for coli_fp4_matmul_batch_rows16_order.
 *
 * Measures the cost of the SIMD arm vs the scalar #else arm of the FP4 expert
 * matmul used in DeepSeek V4 CPU prefill (see Makefile.deepseek-v4: the arm is
 * selected by __AVX2__ / __ARM_NEON on the build). Build the same unit twice
 * — default flags and with EXTRA_CFLAGS=-mno-avx2 (x86) — then compare.
 *
 * Usage: bench_fp4_matmul [S] [I] [O] [iters] [ref_file]
 *   S     rows (token batch); the kernel contract caps S at 128
 *   I/O   input/output dims. Real V4-Flash experts: (S,4096,2048) w1/w3,
 *         (S,2048,4096) w2. Tiny fixture: (S,128,128).
 *   ref   optional path: first run writes a float32 reference output,
 *         second run (other arm) compares bit-exactly against it.
 *
 * Data notes:
 *   - seeded xorshift, deterministic across platforms/builds
 *   - e8m0 scale bytes limited to 120..127 (2^-8 .. 2^-1) so products can
 *     never overflow to inf/NaN — full-range random scale bytes poison any
 *     bit-comparison with NaN (NaN != NaN).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#if defined(_WIN32)
#  include <io.h>
#  define BK_ACCESS(p, m) _access(p, m)
#else
#  include <unistd.h>
#  define BK_ACCESS(p, m) access(p, m)
#endif

/* Mirrors the kernel's own arm selection (deepseek_v4.c gates on __AVX2__
 * only today): on arm64 the SIMD arm does not exist yet, so the build is the
 * scalar one even though NEON is available. */
/* Mirrors the kernel's own arm selection; scalar when both/neither. */
#if defined(__AVX2__)
#  define BK_ARM_NAME "avx2"
#elif defined(__ARM_NEON)
#  define BK_ARM_NAME "neon"
#else
#  define BK_ARM_NAME "scalar"
#endif

/* Link stubs: GPU-tier entry points referenced (never taken) by the CPU-only
 * build of this unit. Signatures from deepseek_v4_internal.h. */
int coli_v4_gpu_matvec_grouped(const void *w, float *output,
                               const float *input, int groups) { (void)w; (void)output; (void)input; (void)groups; return 1; }
int coli_v4_gpu_fp8_matvec(const void *w, float *output,
                           const float *input) { (void)w; (void)output; (void)input; return 1; }
int coli_v4_gpu_fp8_matmul_batch(const void *w, float *outputs,
                                 const float *inputs, int batch) { (void)w; (void)outputs; (void)inputs; (void)batch; return 1; }

void coli_fp4_matmul_batch_rows16_order(float *y, const uint8_t *q4,
                                        const uint8_t *e8s, const float *x,
                                        int S, int I, int O);

static double now_s(void) {
    struct timespec ts;
#if defined(__APPLE__)
    clock_gettime(CLOCK_MONOTONIC, &ts); /* works since macOS 10.12 */
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t rng_state = 0x1234abcd;
static uint32_t rng(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5; return rng_state;
}

int main(int argc, char **argv) {
    int S = argc > 1 ? atoi(argv[1]) : 128;
    int I = argc > 2 ? atoi(argv[2]) : 4096;
    int O = argc > 3 ? atoi(argv[3]) : 2048;
    int iters = argc > 4 ? atoi(argv[4]) : 7;
    const char *ref_path = argc > 5 ? argv[5] : NULL;

    if (S < 1 || I % 2 || I % 32 || O < 1) {
        fprintf(stderr, "bad shape: need S>=1, I%%2==0, I%%32==0 (FP4 group "
                        "packing), O>=1\n");
        return 2;
    }
    if (S > 128) {
        fprintf(stderr, "S=%d exceeds the kernel's batch cap (sums[128]); the "
                        "sole caller chunks prefill to <=128 rows\n", S);
        return 2;
    }

    size_t rb = (size_t)I / 2, ng = (size_t)I / 32;
    uint8_t *q4 = malloc(rb * O);
    uint8_t *e8s = malloc(ng * O);
    float *x = malloc((size_t)S * I * sizeof(float));
    float *y = malloc((size_t)S * O * sizeof(float));
    if (!q4 || !e8s || !x || !y) { fprintf(stderr, "oom\n"); return 1; }
    for (size_t i = 0; i < rb * O; i++) q4[i] = (uint8_t)rng();
    for (size_t i = 0; i < ng * O; i++) e8s[i] = (uint8_t)(120 + (rng() & 7));
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = (float)((double)(rng() >> 8) / 8388608.0 - 1.0);

    memset(y, 0, (size_t)S * O * sizeof(float));
    coli_fp4_matmul_batch_rows16_order(y, q4, e8s, x, S, I, O);

    if (ref_path) {
        FILE *f = fopen(ref_path, "rb");
        if (f) {
            size_t n = (size_t)S * O;
            float *ref = malloc(n * sizeof(float));
            if (fread(ref, sizeof(float), n, f) == n) {
                size_t bad = 0;
                for (size_t i = 0; i < n; i++) if (ref[i] != y[i]) bad++;
                printf("BITEXACT vs %s : %s (%zu/%zu differ)\n", ref_path,
                       bad ? "MISMATCH" : "IDENTICAL", bad, n);
                if (bad) return 3;
            } else printf("BITEXACT: ref file wrong size\n");
            fclose(f); free(ref);
        } else printf("BITEXACT: no ref (this run writes it)\n");
    }

    /* timing: discard first timed call (page faults), median of the rest */
    double best = 1e9, times[64]; int nt = 0;
    if (iters > 63) iters = 63;
    for (int it = 0; it < iters + 1; it++) {
        double t0 = now_s();
        coli_fp4_matmul_batch_rows16_order(y, q4, e8s, x, S, I, O);
        double dt = now_s() - t0;
        if (it > 0) { times[nt++] = dt; if (dt < best) best = dt; }
    }
    for (int i = 0; i < nt; i++) for (int j = i + 1; j < nt; j++)
        if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
    double med = times[nt / 2];
    double flops = 2.0 * S * I * O;
    printf("KERNEL S=%d I=%d O=%d arm=%s iters=%d med=%.4fs best=%.4fs "
           "gflops_med=%.2f\n",
           S, I, O, BK_ARM_NAME, iters, med, best, flops / med / 1e9);
    if (ref_path && BK_ACCESS(ref_path, 0) != 0) {
        FILE *f = fopen(ref_path, "wb");
        if (f) { fwrite(y, sizeof(float), (size_t)S * O, f); fclose(f); }
    }
    return 0;
}
