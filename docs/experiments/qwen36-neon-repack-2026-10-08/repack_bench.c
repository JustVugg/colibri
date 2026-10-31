/* repack_bench.c: how long xf_repack_pairs_signed takes for one Qwen3.6-35B-A3B
 * expert, the three calls load_expert_merged makes after every expert read from
 * disk: gate and up (512 x 2048 INT4), down (2048 x 512). The buffer is the
 * size of one expert (1.5 MB), so it stays in the CPU caches as the engine's
 * just-read expert mostly does. The calls go through a function the compiler
 * cannot inline, as in the engine: inlined next to the two mallocs below, clang
 * can prove the buffers distinct and vectorizes the scalar loop, which then takes
 * half the time it takes in load_expert_merged. Prints the median of 15 rounds
 * of 200 experts and a checksum of the output (equal for both builds: same bytes).
 *
 *   clang -O3 -I<engine c/ dir> repack_bench.c -o repack_bench && ./repack_bench
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "expert_ffn.h"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

/* Like load_expert_merged, which repacks into a slot from a buffer it cannot
 * prove distinct: not inlined, so the loop is compiled as the engine's is. */
__attribute__((noinline)) static void repack(uint8_t *dst, const uint8_t *src, int O, int I) {
    xf_repack_pairs_signed(dst, src, O, I);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(void) {
    const int hidden = 2048, inter = 512, experts = 200, rounds = 15;
    const size_t gp = (size_t)inter * hidden / 2, bytes = 3 * gp;
    uint8_t *raw = malloc(bytes), *pw = malloc(bytes);
    if (!raw || !pw) return 1;
    uint32_t x = 12345;
    for (size_t i = 0; i < bytes; i++) { x = x * 1664525u + 1013904223u; raw[i] = (uint8_t)(x >> 24); }
    double t[15];
    for (int r = -1; r < rounds; r++) {          /* round -1 warms up */
        double a = now_s();
        for (int e = 0; e < experts; e++) {
            raw[e % bytes] ^= (uint8_t)e;          /* one byte changed each time */
            repack(pw,          raw,          inter,  hidden);
            repack(pw + gp,     raw + gp,     inter,  hidden);
            repack(pw + 2 * gp, raw + 2 * gp, hidden, inter);
        }
        if (r >= 0) t[r] = (now_s() - a) / experts;
    }
    qsort(t, rounds, sizeof t[0], cmp_double);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < bytes; i++) h = (h ^ pw[i]) * 0x100000001b3ull;
#if defined(__AVX2__)
    const char *arm = "AVX2";
#elif defined(__ARM_NEON)
    const char *arm = "NEON or scalar, as the header chooses";
#else
    const char *arm = "scalar";
#endif
    printf("repack of one expert (3 x %zu bytes): median %.4f ms, min %.4f, max %.4f (%d rounds x %d experts); "
           "checksum %016llx; %s\n", gp, t[rounds / 2] * 1e3, t[0] * 1e3, t[rounds - 1] * 1e3, rounds, experts,
           (unsigned long long)h, arm);
    free(raw); free(pw);
    return 0;
}
