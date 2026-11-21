/* coli_silu_exact.hpp -- COLI_SYCL_SILU_CPU=2: the host's SiLU to the bit, on a SYCL device.
 *
 * glibc_expf_dev is glibc 2.39's expf as x86-64 runs it (__expf_fma: sysdeps/ieee754/flt-32/e_expf.c with the table
 * and coefficients of e_exp2f_data.c; WANT_ERRNO_UFLOW, no TOINT intrinsics). Every step that build fuses is a
 * sycl::fma here, the rest one IEEE double operation each, so the result does not depend on the compiler's contraction
 * setting. The port gives the system expf's bits on all 2^32 float inputs: checked exhaustively on the host against
 * glibc's libm, and on the card by sycl/tests/test_sycl_silu_exact.cpp (linked by GCC so its host expf is glibc's, as
 * the engine's is). (Which steps are fused matters for one: r = x*N/ln2 - k must be a single fma; computed as a
 * multiply then a subtract, two float inputs give another result.)
 *
 * glibc_form_silu_mul then does coli_fp8_host_silu_mul's float steps (sycl/fp8_host_silu.h): z = expf(-|g|), the
 * stable sigmoid 1/(1+z) or z/(1+z), g*sig, times u. The sigmoid's float quotient is formed in double and rounded once,
 * which is the correctly rounded float quotient (53 >= 2*24+2), i.e. the host's float division -- the device's own
 * float division is not correctly rounded without SYCL_IEEE_DIV_SQRT=1. Keep the written form: the numerator a select
 * of 1.0 and (double)z keeps the divide in double; LLVM may narrow `(double)a/(double)b` of two floats to a float
 * divide (allowed, 53 >= 2*24), which the device would then round its own way (the review of ef21b72b). */
#ifndef COLI_SILU_EXACT_HPP
#define COLI_SILU_EXACT_HPP
#include <sycl/sycl.hpp>
#include <cstdint>

static inline float glibc_expf_dev(float x) {
    static constexpr uint64_t T[32] = {   /* static: one copy in constant memory, not a private one a call */
        0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
        0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
        0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
        0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
        0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
        0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
        0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
        0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
    };
    constexpr double InvLn2N = 0x1.71547652b82fep+0 * 32, Shift = 0x1.8p+52;
    constexpr double C0 = 0x1.c6af84b912394p-5 / 32 / 32 / 32, C1 = 0x1.ebfce50fac4f3p-3 / 32 / 32,
                     C2 = 0x1.62e42ff0c52d6p-1 / 32;
    const uint32_t ux = sycl::bit_cast<uint32_t>(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {                                   /* |x| >= 88 (top12(88.0f)) or x is nan */
        if (ux == 0xff800000u) return 0.0f;                   /* -inf */
        if (abstop >= 0x7f8u) return x + x;                   /* nan, +inf */
        if (x > 0x1.62e42ep6f) return sycl::bit_cast<float>(0x7f800000u);   /* __math_oflowf */
        if (x < -0x1.9fe368p6f) return 0.0f;                  /* __math_uflowf */
        if (x < -0x1.9d1d9ep6f) return 0x1p-149f;             /* __math_may_uflowf: 0x1.4p-75f squared */
    }
    const double xd = (double)x;
    double kd = sycl::fma(InvLn2N, xd, Shift);
    const uint64_t ki = sycl::bit_cast<uint64_t>(kd);
    kd -= Shift;
    const double r = sycl::fma(InvLn2N, xd, -kd);
    uint64_t t = T[ki % 32];
    t += ki << (52 - 5);
    const double s = sycl::bit_cast<double>(t);
    const double z = sycl::fma(C0, r, C1);
    const double r2 = r * r;
    double y = sycl::fma(C2, r, 1.0);
    y = sycl::fma(z, r2, y);
    y = y * s;
    return (float)y;
}

static inline float glibc_form_silu_mul(float g, float u) {
    const float z = glibc_expf_dev(g >= 0.f ? -g : g);
    const float den = 1.f + z;
    const float sig = (float)((g >= 0.f ? 1.0 : (double)z) / (double)den);
    const float hidden = g * sig;
    return hidden * u;
}
#endif /* COLI_SILU_EXACT_HPP */
