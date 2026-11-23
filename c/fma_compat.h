/* coli_fmaf(a, b, c): a * b + c rounded once, as fmaf. Where the CPU has a fused multiply-add
 * (FP_FAST_FMAF: x86 with FMA3, arm64) it is fmaf, the one instruction, and the code is the same
 * as with fmaf (built for Haswell, qwen38's instructions are identical). On a CPU without one
 * (Sandy Bridge, Ivy Bridge, the x86-64 baseline) fmaf is a call into libm for every element,
 * which no loop vectorizes, and glibc computes it in software there: a Xeon E5-2690 v1 spent
 * 898 ms of a 1248 ms Qwen3.8 forward in the DeltaNet recurrence's fmaf calls (#2606). There the
 * product is formed in double, where it is exact, and the sum rounded to float; AVX vectorizes it
 * four wide. It is fmaf's result but where rounding the sum to double first lands on a float's
 * halfway point (a double rounding): 2 of 92.9 million cases, random bit patterns, small and
 * subnormal values and constructed ties, one ulp each. Such an ulp can turn a greedy answer
 * another way; the model's quality is the same: Qwen3.8 IQ2_XS teacher-forced over 258 tokens,
 * 2.2402 nats a token with fmaf and with this. Made exactly (TwoSum and rounding to odd, as
 * expert_ffn.h's SSE2 kernel does) the recurrence ran slower than glibc's fmaf: 558 ms a forward
 * against 208, against 52 this way (built for Sandy Bridge on an i5-13500, glibc's software fmaf
 * by GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA,-FMA4, a 585-token prompt and 64 tokens). */
#ifndef COLI_FMA_COMPAT_H
#define COLI_FMA_COMPAT_H
#include <math.h>
#include <stdint.h>
#include <string.h>

/* coli_fmaf_rto(a, b, c): fmaf's bits without a fused multiply-add, for a kernel whose result
 * must stay the same on every CPU. The product of two floats is exact in a double; the sum is
 * rounded to the nearest double. Rounding that double to float gives the correctly rounded
 * a * b + c unless the double sits exactly halfway between two floats, where the first rounding
 * may have made the tie (a float's halfway points are doubles, and rounding is monotone: a sum
 * that did not land on one stays on its side). Only there, and for a result among the subnormal
 * floats (whose halfway points the bit test below does not see), the error is recovered exactly
 * (TwoSum) and the double moved one step toward it when its last bit is even: the sum rounded to
 * odd, which one rounding to float turns into fmaf's result (Boldo and Melquiond, 2008: 53 bits
 * >= 2 * 24 + 2). An infinite or NaN sum is the double's. tests/test_fmaf_exact.c checks it
 * against fmaf on random bit patterns, cancelling sums, halfway ties and subnormals. */
static inline float coli_fmaf_rto(float a, float b, float c) {
    const double p = (double)a * (double)b, cd = (double)c;
    double s = p + cd;
    uint64_t u;
    memcpy(&u, &s, sizeof u);
    /* the 29 bits a double has below a normal float's last: 1 then 28 zeros is a halfway point */
    if ((u & 0x1fffffffu) != 0x10000000u && !(fabs(s) < 0x1p-126)) return (float)s;
    if (s - s != 0.0) return (float)s;                      /* inf or NaN */
    const double bv = s - p, err = (p - (s - bv)) + (cd - bv);
    if (err != 0.0 && !(u & 1)) {
        u += ((err > 0.0) == (s > 0.0)) ? 1 : (uint64_t)-1;
        memcpy(&s, &u, sizeof s);
    }
    return (float)s;
}

#if defined(FP_FAST_FMAF) || defined(__FP_FAST_FMAF) || defined(__FMA__) || defined(__aarch64__)
#define coli_fmaf(a, b, c) fmaf((a), (b), (c))
#define coli_fmaf_exact(a, b, c) fmaf((a), (b), (c))
#else
static inline float coli_fmaf(float a, float b, float c) { return (float)((double)a * (double)b + (double)c); }
/* coli_fmaf_exact: fmaf's bits in a build without FMA. On a CPU without it glibc's fmaf computes it
 * in software, 38.9 ns a call with glibc 2.39 on an i7-1355U with its FMA paths turned off, about
 * 50 ns on the Xeon E5-2670 of #2882 (glibc 2.41): gsgemv.h's SSE4.1 group step 0.600 -> 0.229 ns a
 * weight, gq_dot_i8's min correction 2.610 -> 0.794, one thread. A build without FMA that runs on a
 * CPU with it (ARCH=x86-64 on a newer machine) keeps fmaf, which glibc then makes in hardware and
 * which is the faster of the two there (0.141 ns against 0.217): the CPU is asked once. */
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
static inline int coli_cpu_fma(void) {
    static int has = -1;
    if (has < 0) { __builtin_cpu_init(); has = __builtin_cpu_supports("fma") ? 1 : 0; }
    return has;
}
#define coli_fmaf_exact(a, b, c) (coli_cpu_fma() ? fmaf((a), (b), (c)) : coli_fmaf_rto((a), (b), (c)))
#else
#define coli_fmaf_exact(a, b, c) coli_fmaf_rto((a), (b), (c))
#endif
#endif
#endif
