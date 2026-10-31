/* test_exact_expf.c — exact_expf.h: exp(x) correctly rounded to float, and the same bits in
 * every tier.
 *
 * For each float x checked:
 *  1. exact_expf(x) is exp(x) correctly rounded. The reference is the C library's double exp
 *     rounded to float, which is the correctly rounded value unless exp(x) lies within 2^-50
 *     (relative) of the middle between two floats: a library's double exp is within one
 *     double ulp (2^-52), so anywhere else it cannot fall on the wrong side. The few "near"
 *     floats must be in near_cases[] below, whose results were worked out at 60 decimal
 *     digits (python3 decimal); the table is cut at 2^-49 so that it also holds the near
 *     cases of a library whose double exp is off by up to one ulp in the other direction;
 *  2. no argument that is not NaN comes out NaN (a rounding the scalar could not prove);
 *  3. exact_expf_n gives the scalar's bits (the AVX-512 or AVX2 tier when the build has it,
 *     the scalar loop otherwise), in place and not, and on every tail length 0-17.
 * By default every 61st bit pattern (70 million floats, about a second); --all is every one of
 * the 2^32, which is the proof. --near prints the near floats of this library and checks
 * nothing (within 2^-49: to regenerate near_cases[]). */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../exact_expf.h"

/* bits of x, bits of exp(x) correctly rounded: every float whose exp lies within 2^-49 of the
 * middle between two floats, by glibc's double exp (exp(x) in the comment) */
static const uint32_t near_cases[][2] = {
    {0x337fffffu, 0x3f800000u}, /* exp(0x1.fffffe0000000p-25) */
    {0x33800000u, 0x3f800001u}, /* exp(0x1.0000000000000p-24) */
    {0x343fffffu, 0x3f800002u}, /* exp(0x1.7ffffe0000000p-23) */
    {0x34dffffdu, 0x3f800004u}, /* exp(0x1.bffffa0000000p-22) */
    {0x356ffff9u, 0x3f800008u}, /* exp(0x1.dffff20000000p-21) */
    {0x35f7fff1u, 0x3f800010u}, /* exp(0x1.efffe20000000p-20) */
    {0x367bffe1u, 0x3f800020u}, /* exp(0x1.f7ffc20000000p-19) */
    {0x36fdffc1u, 0x3f800040u}, /* exp(0x1.fbff820000000p-18) */
    {0x377eff81u, 0x3f800080u}, /* exp(0x1.fdff020000000p-17) */
    {0x383a3ef1u, 0x3f800175u}, /* exp(0x1.747de20000000p-15) */
    {0x38e69cc1u, 0x3f80039au}, /* exp(0x1.cd39820000000p-14) */
    {0x39c6be5bu, 0x3f800c6du}, /* exp(0x1.8d7cb60000000p-12) */
    {0x39e5bb1du, 0x3f800e5du}, /* exp(0x1.cb763a0000000p-12) */
    {0x3d1a274eu, 0x3f84e8bau}, /* exp(0x1.344e9c0000000p-5) */
    {0x3fe67199u, 0x40c1a7a6u}, /* exp(0x1.cce3320000000p+0) */
    {0x4001b249u, 0x40f2cd14u}, /* exp(0x1.0364920000000p+1) */
    {0x40315b33u, 0x417fa47du}, /* exp(0x1.62b6660000000p+1) */
    {0x41cbf87bu, 0x51dc50beu}, /* exp(0x1.97f0f60000000p+4) */
    {0x4288942bu, 0x70b7a4c5u}, /* exp(0x1.1128560000000p+6) */
    {0xb3000000u, 0x3f800000u}, /* exp(-0x1.0000000000000p-25) */
    {0xbae0e25cu, 0x3f7f8fa7u}, /* exp(-0x1.c1c4b80000000p-10) */
    {0xbbb70ee8u, 0x3f7e92e8u}, /* exp(-0x1.6e1dd00000000p-8) */
    {0xbbf0edf1u, 0x3f7e1fe9u}, /* exp(-0x1.e1dbe20000000p-8) */
    {0xbc2a461au, 0x3f7d5a6eu}, /* exp(-0x1.548c340000000p-7) */
    {0xc13d6631u, 0x36f28e33u}, /* exp(-0x1.7acc620000000p+3) */
    {0xc16912cdu, 0x34fd331bu}, /* exp(-0x1.d2259a0000000p+3) */
};
#define N_NEAR (int)(sizeof near_cases / sizeof near_cases[0])

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* is d = exp(x) within thr (relative) of the middle between the two floats around it? */
static int near_middle(double d, double thr) {
    float f = (float)d, lo, hi;
    if ((double)f <= d) { lo = f; hi = nextafterf(f, INFINITY); }
    else { hi = f; lo = nextafterf(f, -INFINITY); }
    if (isinf(lo)) return 0;
    double mid = isinf(hi) ? (double)FLT_MAX + 0x1p103 : ((double)lo + (double)hi) * 0.5;
    return fabs(d - mid) <= mid * thr;
}

static int near_index(uint32_t u) {
    for (int i = 0; i < N_NEAR; i++) if (near_cases[i][0] == u) return i;
    return -1;
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  [FAIL] " __VA_ARGS__); printf("\n"); fails++; } \
                           else { printf("  [PASS] " __VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char **argv) {
    int all = argc > 1 && !strcmp(argv[1], "--all"), near_only = argc > 1 && !strcmp(argv[1], "--near");
    uint64_t step = (all || near_only) ? 1 : 61, count = (0x100000000ull + step - 1) / step;
    enum { B = 4096 };
    uint64_t wrong = 0, nan_out = 0, tier = 0, near_seen = 0, near_missing = 0, first_bad = 0;
    int64_t nblocks = (int64_t)((count + B - 1) / B);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(+:wrong, nan_out, tier, near_seen, near_missing)
#endif
    for (int64_t blk = 0; blk < nblocks; blk++) {
        float x[B], s[B], v[B];
        int n = 0;
        for (uint64_t i = (uint64_t)blk * B; i < count && n < B; i++, n++) x[n] = bitsf((uint32_t)(i * step));
        for (int j = 0; j < n; j++) s[j] = exact_expf(x[j]);
        exact_expf_n(x, v, n);
        tier += memcmp(s, v, (size_t)n * 4) != 0;
        memcpy(v, x, (size_t)n * 4);
        exact_expf_n(v, v, n); /* in place */
        tier += memcmp(s, v, (size_t)n * 4) != 0;
        for (int j = 0; j < n; j++) {
            if (x[j] != x[j]) { wrong += s[j] == s[j]; continue; }
            if (s[j] != s[j]) { nan_out++; continue; }
            double d = exp((double)x[j]);
            uint32_t u = fbits(x[j]);
            if (near_middle(d, near_only ? 0x1p-49 : 0x1p-50)) {
                near_seen++;
                if (near_only) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    printf("0x%08xu\n", u);
                    continue;
                }
                int k = near_index(u);
                if (k < 0) { near_missing++; continue; }
                wrong += fbits(s[j]) != near_cases[k][1];
                continue;
            }
            if (fbits(s[j]) != fbits((float)d)) {
                wrong++;
                if (!first_bad) first_bad = u;
            }
        }
    }
    if (near_only) { fprintf(stderr, "%llu near floats\n", (unsigned long long)near_seen); return 0; }
    const char *tier_name =
#if defined(__AVX512F__)
        "AVX-512";
#elif defined(__AVX2__)
        "AVX2";
#else
        "scalar loop";
#endif
    printf("exact_expf over %s, exact_expf_n tier: %s\n",
           all ? "all 2^32 floats" : "every 61st bit pattern (70M floats)", tier_name);
    CHECK(wrong == 0, "correctly rounded: %llu wrong (first 0x%08llx)", (unsigned long long)wrong,
          (unsigned long long)first_bad);
    CHECK(nan_out == 0, "no NaN out of a number: %llu", (unsigned long long)nan_out);
    CHECK(near_missing == 0, "near-middle floats all in the table: %llu seen, %llu missing",
          (unsigned long long)near_seen, (unsigned long long)near_missing);
    CHECK(tier == 0, "exact_expf_n == exact_expf, in place and not: %llu blocks differ", (unsigned long long)tier);

    /* the table itself, whatever the stride: each entry is the scalar's result */
    int table_bad = 0;
    for (int i = 0; i < N_NEAR; i++) table_bad += fbits(exact_expf(bitsf(near_cases[i][0]))) != near_cases[i][1];
    CHECK(table_bad == 0, "the %d near-middle floats give their 60-digit result: %d differ", N_NEAR, table_bad);

    /* tails: every length 0-17 at every offset 0-3, so a lane mask or a scalar tail cannot hide */
    int tail_bad = 0;
    float src[24], out[24], ref[24];
    uint64_t rs = 0x9E3779B97F4A7C15ull;
    for (int round = 0; round < 64; round++)
        for (int off = 0; off < 4; off++)
            for (int len = 0; len <= 17; len++) {
                for (int j = 0; j < 24; j++) {
                    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                    src[j] = (float)((double)(rs >> 11) * 0x1p-53 * 220.0 - 110.0);
                    out[j] = ref[j] = -1.0f;
                }
                for (int j = 0; j < len; j++) ref[off + j] = exact_expf(src[off + j]);
                exact_expf_n(src + off, out + off, len);
                tail_bad += memcmp(out, ref, sizeof out) != 0;
            }
    CHECK(tail_bad == 0, "tails 0-17 at offsets 0-3, nothing written past n: %d differ", tail_bad);
    return fails ? 1 : 0;
}
