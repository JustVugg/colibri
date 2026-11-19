/* coli_fmaf_rto and coli_fmaf_exact (fma_compat.h) against fmaf, bit for bit (#2882). Built for
 * the x86-64 baseline without FMA, where coli_fmaf_exact is the emulation and libm's fmaf the
 * reference (correctly rounded, in hardware when the CPU has FMA). Five sets: random bit
 * patterns (NaN, infinities, subnormals included), products cancelled by c to a few ulps,
 * products that land exactly halfway between two floats (where a sum rounded twice goes the
 * wrong way: the counter-proof checks that coli_fmaf's double rounding does miss some), the
 * same ties scaled into the subnormal floats, and small integers times scales as the kernels
 * use them. A NaN matches a NaN. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../fma_compat.h"

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t next32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}
static float from_bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t to_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float uniform(float lo, float hi) { return lo + (hi - lo) * (float)(next32() >> 8) / 16777216.f; }

static long checked, wrong;
static void check(float a, float b, float c, const char *set) {
    float want = fmaf(a, b, c), got = coli_fmaf_rto(a, b, c), got2 = coli_fmaf_exact(a, b, c);
    checked++;
    int ok = (isnan(want) && isnan(got) && isnan(got2)) ||
             (to_bits(want) == to_bits(got) && to_bits(want) == to_bits(got2));
    if (!ok) {
        if (wrong < 8)
            fprintf(stderr, "test_fmaf_exact: %s: fmaf(%a, %a, %a) = %a, coli_fmaf_rto %a, coli_fmaf_exact %a\n",
                    set, a, b, c, want, got, got2);
        wrong++;
    }
}

int main(void) {
    /* 1. random bit patterns */
    for (long i = 0; i < 6000000; i++) check(from_bits(next32()), from_bits(next32()), from_bits(next32()), "bits");
    /* 2. c cancels the product to within a few ulps of it */
    for (long i = 0; i < 3000000; i++) {
        float a = uniform(-4.f, 4.f) * ldexpf(1.f, (int)(next32() % 60) - 30), b = uniform(-4.f, 4.f);
        float c = -(a * b);
        uint32_t u = to_bits(c) + (next32() % 9) - 4;
        check(a, b, from_bits(u), "cancel");
    }
    /* 3. ties: (1 + k1 2^-12)(1 + k2 2^-12) has the bit 2^-24 set when k1 k2 is odd, halfway between
     * two floats near 1 (k1 + k2 < 2048 keeps it below 2). c either shifts it by whole float ulps, so
     * it stays a tie, or nudges it by far less than a double's ulp, so the double sum rounds back
     * onto the tie and a second rounding to float goes to the even side whichever way c pointed */
    long ties_missed_by_double = 0, ties = 0;
    for (long i = 0; i < 3000000; i++) {
        int k1 = (int)(next32() % 1024) | 1, k2 = (int)(next32() % 1024) | 1;
        int e = (int)(next32() % 200) - 100;
        float a = ldexpf(1.f + (float)k1 / 4096.f, e), b = 1.f + (float)k2 / 4096.f;
        float c = (i & 1) ? ldexpf((float)((int)(next32() % 2001) - 1000), e - 23)
                          : ldexpf((next32() & 1) ? 1.f : -1.f, e - 40 - (int)(next32() % 40));
        if (next32() & 1) { a = -a; c = -c; }
        check(a, b, c, "ties");
        ties++;
        if (to_bits((float)((double)a * (double)b + (double)c)) != to_bits(fmaf(a, b, c))) ties_missed_by_double++;
    }
    /* 4. the ties scaled to subnormal results */
    for (long i = 0; i < 1000000; i++) {
        int k1 = (int)(next32() % 2048) | 1, k2 = (int)(next32() % 2048) | 1;
        float a = ldexpf(1.f + (float)k1 / 4096.f, -126 - (int)(next32() % 20)), b = 1.f + (float)k2 / 4096.f;
        float c = ldexpf((float)((int)(next32() % 64) - 32), -149);
        check(a, b, c, "subnormal");
    }
    /* 5. the kernels' operands: an integer group sum times a scale, onto an accumulator */
    for (long i = 0; i < 3000000; i++) {
        float lane = (float)((int)(next32() % 2000001) - 1000000), sc = uniform(1e-4f, 1e-2f), acc = uniform(-50.f, 50.f);
        check(lane, sc, acc, "kernel");
    }
    printf("test_fmaf_exact: %ld cases, %ld differ from fmaf; %ld of %ld ties go the other way when the sum is "
           "rounded twice (the counter-proof)\n", checked, wrong, ties_missed_by_double, ties);
    if (wrong) return 1;
    if (ties_missed_by_double == 0) { fprintf(stderr, "test_fmaf_exact: the tie set never reaches a double rounding\n"); return 1; }
    return 0;
}
