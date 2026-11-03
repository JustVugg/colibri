/* ramtest.c -- system memory checked from user space: GB of RAM in one block, each pass writes a pattern that
 * depends on the address and the pass over every thread, reads it back and counts the 64-bit words that
 * differ, then the same with the pattern inverted in place (read, check, write the inverse, read again).
 * Runs SECONDS seconds. A healthy machine at its settings: 0.
 * usage: ramtest GB SECONDS */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static inline uint64_t pat(uint64_t i, uint64_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull ^ seed;
    x ^= x >> 31; x *= 0x7FB5D329728EA185ull; x ^= x >> 27; x *= 0x81DADEF4BC2DD44Dull; x ^= x >> 33;
    return x;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    const double gb = argc > 1 ? atof(argv[1]) : 64, secs = argc > 2 ? atof(argv[2]) : 300;
    const size_t n = (size_t)(gb * 1073741824.0) / 8;
    uint64_t *p = malloc(n * 8);
    if (!p) { printf("no %.0f GiB\n", gb); return 1; }
    printf("%.1f GiB under test for %.0f s on %d threads\n", n * 8 / 1073741824.0, secs, omp_get_max_threads());
    unsigned long long bad = 0, passes = 0, first_i[8], first_x[8]; int nf = 0;
    const double t0 = now();
    for (uint64_t pass = 1;; pass++) {
        const uint64_t seed = pass * 0xD1B54A32D192ED03ull;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) p[i] = pat(i, seed);
        unsigned long long b1 = 0;
        #pragma omp parallel for schedule(static) reduction(+:b1)
        for (size_t i = 0; i < n; i++) {
            const uint64_t g = p[i], w = pat(i, seed);
            if (g != w) {
                b1++;
                #pragma omp critical
                if (nf < 8) { first_i[nf] = i; first_x[nf] = g ^ w; nf++; }
            }
            p[i] = ~g;
        }
        #pragma omp parallel for schedule(static) reduction(+:b1)
        for (size_t i = 0; i < n; i++) {
            const uint64_t g = p[i], w = ~pat(i, seed);
            if (g != w) {
                b1++;
                #pragma omp critical
                if (nf < 8) { first_i[nf] = i; first_x[nf] = g ^ w; nf++; }
            }
        }
        bad += b1; passes++;
        if (now() - t0 >= secs) break;
    }
    printf("%llu passes in %.0f s (%.3g words read): %llu wrong\n", passes, now() - t0, (double)passes * n * 2, bad);
    for (int i = 0; i < nf; i++) printf("  word %llu (offset %.3f GiB): bits %016llx wrong\n", first_i[i], first_i[i] * 8 / 1073741824.0, first_x[i]);
    free(p);
    return bad ? 2 : 0;
}
