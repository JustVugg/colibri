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
#if defined(FP_FAST_FMAF) || defined(__FP_FAST_FMAF) || defined(__FMA__) || defined(__aarch64__)
#define coli_fmaf(a, b, c) fmaf((a), (b), (c))
#else
static inline float coli_fmaf(float a, float b, float c) { return (float)((double)a * (double)b + (double)c); }
#endif
#endif
