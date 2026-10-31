/* exact_expf.h — exp(x) correctly rounded to float: the same bits on every platform.
 *
 * The C library's expf is a different function on each system: glibc's is not the
 * correctly rounded value for 170 648 of the 2^32 floats, and Windows' UCRT and Apple's
 * libm are other code again, so the same engine computes different bits on different
 * machines. This header gives the one answer that does not depend on the library: the
 * float nearest to exp(x), ties to even.
 *
 *   exact_expf(x)           scalar; the definition
 *   exact_expf_n(x, y, n)   y[i] = exact_expf(x[i]) for i < n, y may be x: 16 lanes on
 *                           AVX-512, 8 on AVX2 (chosen at compile time, like the rest of
 *                           the engine), the scalar loop elsewhere. Every tier gives the
 *                           scalar's bits.
 *
 * Scalar: exp(x) = 2^(k/64) exp(r), k = round(64 x / ln2), r = x - k ln2/64 with ln2/64 in
 * two pieces (k times the first is exact), all in double: a table of 2^(j/64), a degree-5
 * polynomial, then a rounding test. If y(1 - 2^-50) and y(1 + 2^-50) round to the same
 * float, exp(x) rounds there too, subnormal results included. The 8 floats that fail the
 * test carry their correctly rounded result in a table. A float that failed it without
 * being in the table would come out as NaN, which the exhaustive test would see.
 *
 * Vector: float and int32 operations only. k = round(256 x / ln2), ln2/256 in three 8-bit
 * pieces so that every product with k is exact, 2^(j/256) = hi + lo from a gathered table,
 * a Dekker product, and the same kind of test with a margin of 2^-39. The lanes it does not
 * settle (about 1 in 33 000) take exact_expf itself: there is no second slow path to prove.
 *
 * Both rely on every multiply and add being rounded on its own; a fused multiply-add
 * changes the error the tests are sized for. colibri builds with contraction on, so the two
 * functions turn it off for their own bodies: clang with its pragma, gcc with the optimize
 * attribute, which gcc drops when it inlines the body into a caller, hence noinline.
 *
 * The proof is tests/test_exact_expf.c --all: every one of the 2^32 floats against the
 * correctly rounded value, and the tier against the scalar. The constants were generated
 * with mpmath (200 bits for the scalar, 300 for the vector table); the test is what proves
 * them. Ported from trochilus (src/kernels/expf.c and the expf_f32 kernels). */
#ifndef COLI_EXACT_EXPF_H
#define COLI_EXACT_EXPF_H

#include <stdint.h>
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__clang__)
#define EXF_FN static
#define EXF_NOFUSE _Pragma("clang fp contract(off)")
#elif defined(__GNUC__)
#define EXF_FN static __attribute__((noinline, optimize("fp-contract=off")))
#define EXF_NOFUSE
#else
#define EXF_FN static
#define EXF_NOFUSE
#endif

/* k >> 6 below is floor(k / 64) for a negative k too */
_Static_assert((-1 >> 1) == -1, "exact_expf needs an arithmetic right shift of signed integers");

#define EXF_INV_LN2_64 0x1.71547652b82fep+6       /* 64 / ln2 */
#define EXF_LN2_64_HI 0x1.62e42fee00000p-7        /* ln2 / 64, a multiple of 2^-38: k * hi is exact */
#define EXF_LN2_64_LO 0x1.a39ef35793c76p-39       /* the rest of ln2 / 64 */
#define EXF_C3 0x1.5555555555555p-3               /* 1/6 */
#define EXF_C4 0x1.5555555555555p-5               /* 1/24 */
#define EXF_C5 0x1.1111111111111p-7               /* 1/120 */
#define EXF_MARGIN 0x1p-50                        /* the error the fast path is allowed */
#define EXF_OVERFLOW_X 0x1.62e42e0000000p+6f      /* the largest float whose exp is finite */
#define EXF_UNDERFLOW_X -0x1.a000000000000p+6f    /* below it exp(x) < 2^-150 rounds to zero */

/* 2^(j/64), j = 0..63, rounded to nearest */
static const double exf_pow2[64] = {
    0x1.0000000000000p+0, 0x1.02c9a3e778061p+0, 0x1.059b0d3158574p+0, 0x1.0874518759bc8p+0,
    0x1.0b5586cf9890fp+0, 0x1.0e3ec32d3d1a2p+0, 0x1.11301d0125b51p+0, 0x1.1429aaea92de0p+0,
    0x1.172b83c7d517bp+0, 0x1.1a35beb6fcb75p+0, 0x1.1d4873168b9aap+0, 0x1.2063b88628cd6p+0,
    0x1.2387a6e756238p+0, 0x1.26b4565e27cddp+0, 0x1.29e9df51fdee1p+0, 0x1.2d285a6e4030bp+0,
    0x1.306fe0a31b715p+0, 0x1.33c08b26416ffp+0, 0x1.371a7373aa9cbp+0, 0x1.3a7db34e59ff7p+0,
    0x1.3dea64c123422p+0, 0x1.4160a21f72e2ap+0, 0x1.44e086061892dp+0, 0x1.486a2b5c13cd0p+0,
    0x1.4bfdad5362a27p+0, 0x1.4f9b2769d2ca7p+0, 0x1.5342b569d4f82p+0, 0x1.56f4736b527dap+0,
    0x1.5ab07dd485429p+0, 0x1.5e76f15ad2148p+0, 0x1.6247eb03a5585p+0, 0x1.6623882552225p+0,
    0x1.6a09e667f3bcdp+0, 0x1.6dfb23c651a2fp+0, 0x1.71f75e8ec5f74p+0, 0x1.75feb564267c9p+0,
    0x1.7a11473eb0187p+0, 0x1.7e2f336cf4e62p+0, 0x1.82589994cce13p+0, 0x1.868d99b4492edp+0,
    0x1.8ace5422aa0dbp+0, 0x1.8f1ae99157736p+0, 0x1.93737b0cdc5e5p+0, 0x1.97d829fde4e50p+0,
    0x1.9c49182a3f090p+0, 0x1.a0c667b5de565p+0, 0x1.a5503b23e255dp+0, 0x1.a9e6b5579fdbfp+0,
    0x1.ae89f995ad3adp+0, 0x1.b33a2b84f15fbp+0, 0x1.b7f76f2fb5e47p+0, 0x1.bcc1e904bc1d2p+0,
    0x1.c199bdd85529cp+0, 0x1.c67f12e57d14bp+0, 0x1.cb720dcef9069p+0, 0x1.d072d4a07897cp+0,
    0x1.d5818dcfba487p+0, 0x1.da9e603db3285p+0, 0x1.dfc97337b9b5fp+0, 0x1.e502ee78b3ff6p+0,
    0x1.ea4afa2a490dap+0, 0x1.efa1bee615a27p+0, 0x1.f50765b6e4540p+0, 0x1.fa7c1819e90d8p+0,
};

/* the floats the fast path cannot settle: bits of x, bits of exp(x) correctly rounded */
#define EXF_N_EXCEPTIONS 8
static const uint32_t exf_exceptions[EXF_N_EXCEPTIONS][2] = {
    {0x377eff81u, 0x3f800080u}, /* exp(0x1.fdff020000000p-17) */
    {0x39c6be5bu, 0x3f800c6du}, /* exp(0x1.8d7cb60000000p-12) */
    {0x4001b249u, 0x40f2cd14u}, /* exp(0x1.0364920000000p+1) */
    {0x40315b33u, 0x417fa47du}, /* exp(0x1.62b6660000000p+1) */
    {0xb3000000u, 0x3f800000u}, /* exp(-0x1.0000000000000p-25) */
    {0xbae0e25cu, 0x3f7f8fa7u}, /* exp(-0x1.c1c4b80000000p-10) */
    {0xbbf0edf1u, 0x3f7e1fe9u}, /* exp(-0x1.e1dbe20000000p-8) */
    {0xc16912cdu, 0x34fd331bu}, /* exp(-0x1.d2259a0000000p+3) */
};

EXF_FN float exact_expf(float x) {
    EXF_NOFUSE
    union { float f; uint32_t u; } in, a, b, out;
    union { double d; uint64_t u; } scale;
    if (x != x) return x + x; /* NaN in, quiet NaN out */
    if (x > EXF_OVERFLOW_X) {
        out.u = 0x7F800000u; /* +infinity */
        return out.f;
    }
    if (x < EXF_UNDERFLOW_X) return 0.0f;
    double xd = (double)x;
    /* round to the nearest integer without a library call: exact for |z| < 2^51 */
    double kd = (xd * EXF_INV_LN2_64 + 0x1.8p52) - 0x1.8p52;
    int64_t k = (int64_t)kd;
    double r = (xd - kd * EXF_LN2_64_HI) - kd * EXF_LN2_64_LO;
    double p = r + r * r * (0.5 + r * (EXF_C3 + r * (EXF_C4 + r * EXF_C5)));
    double t = exf_pow2[k & 63];
    scale.u = (uint64_t)((k >> 6) + 1023) << 52; /* 2^floor(k / 64), exact */
    double y = (t + t * p) * scale.d;
    /* trusted where an error of EXF_MARGIN cannot move the rounding */
    a.f = (float)(y * (1.0 - EXF_MARGIN));
    b.f = (float)(y * (1.0 + EXF_MARGIN));
    if (a.u == b.u) return a.f;
    in.f = x;
    for (int i = 0; i < EXF_N_EXCEPTIONS; i++)
        if (exf_exceptions[i][0] == in.u) {
            out.u = exf_exceptions[i][1];
            return out.f;
        }
    out.u = 0x7FC00000u; /* NaN: tests/test_exact_expf.c --all proves no float gets here */
    return out.f;
}

#if defined(__AVX512F__) || defined(__AVX2__)
#define EXF_INVL 0x1.7154760000000p+8f    /* 256/ln2 */
#define EXF_SHIFT 0x1.8p23f               /* 1.5 2^23: adding it rounds to an integer */
#define EXF_LA 0x1.6200000000000p-9f      /* ln2/256 rounded to float = LA + LB + LC, 8 bits each */
#define EXF_LB 0x1.c800000000000p-18f
#define EXF_LC 0x1.8000000000000p-28f
#define EXF_NLL 0x1.05c6100000000p-37f    /* -(ln2/256 - (LA + LB + LC)) rounded to float */
#define EXF_C2V 0x1.0000020000000p-1f     /* 1/2 + 2^-24: minimax with C3V */
#define EXF_C3V 0x1.5555560000000p-3f     /* 1/6 */
#define EXF_X_NORM -0x1.5d589e0000000p+6f /* smallest float >= -126 ln2 */
#define EXF_X_OVF 0x1.62e42ep+6f          /* the largest float whose exp is finite */
#define EXF_X_UF -0x1.a0p+6f              /* below -104 exp(x) < 2^-150: zero */
#define EXF_D 0x1p-39f                    /* what the fast path may be off by, in units of y */
#define EXF_SPLIT 4097.0f                 /* Veltkamp: 2^12 + 1 */

/* 2^(j/256) = hi + lo, laid out hi, lo: a gather at 2j reads hi, at 2j + 1 lo */
static const float exf_tab[512] = {
    0x1.0000000000000p+0f, 0.0f, 0x1.00b1b00000000p+0f, -0x1.6950d00000000p-26f,
    0x1.0163da0000000p+0f, 0x1.3f66660000000p-25f, 0x1.0216820000000p+0f, -0x1.789fb00000000p-25f,
    0x1.02c9a40000000p+0f, -0x1.887fa00000000p-28f, 0x1.037d420000000p+0f, 0x1.c2377a0000000p-25f,
    0x1.04315e0000000p+0f, 0x1.0dcff00000000p-25f, 0x1.04e5f80000000p+0f, -0x1.a1356a0000000p-25f,
    0x1.059b0e0000000p+0f, -0x1.9d4f520000000p-25f, 0x1.0650a00000000p+0f, 0x1.c783f20000000p-25f,
    0x1.0706b20000000p+0f, 0x1.3bbedc0000000p-25f, 0x1.07bd420000000p+0f, 0x1.6e55060000000p-25f,
    0x1.0874520000000p+0f, -0x1.e2990e0000000p-26f, 0x1.092be00000000p+0f, -0x1.333f040000000p-25f,
    0x1.09e3ec0000000p+0f, 0x1.58de700000000p-25f, 0x1.0a9c7a0000000p+0f, -0x1.3831ba0000000p-26f,
    0x1.0b55860000000p+0f, 0x1.9f31220000000p-25f, 0x1.0c0f140000000p+0f, 0x1.791b220000000p-26f,
    0x1.0cc9220000000p+0f, 0x1.6e48fe0000000p-25f, 0x1.0d83b20000000p+0f, 0x1.9caef60000000p-27f,
    0x1.0e3ec40000000p+0f, -0x1.a585cc0000000p-25f, 0x1.0efa560000000p+0f, -0x1.02b1da0000000p-31f,
    0x1.0fb66a0000000p+0f, 0x1.ffda640000000p-25f, 0x1.1073020000000p+0f, 0x1.1ae4680000000p-25f,
    0x1.11301e0000000p+0f, -0x1.fdb4960000000p-25f, 0x1.11edba0000000p+0f, 0x1.6bc5560000000p-25f,
    0x1.12abdc0000000p+0f, 0x1.b0c7300000000p-30f, 0x1.136a820000000p+0f, -0x1.61bf6a0000000p-25f,
    0x1.1429aa0000000p+0f, 0x1.d525bc0000000p-25f, 0x1.14e95a0000000p+0f, -0x1.9619da0000000p-25f,
    0x1.15a98c0000000p+0f, 0x1.14b1ca0000000p-25f, 0x1.166a460000000p+0f, -0x1.71c7880000000p-25f,
    0x1.172b840000000p+0f, -0x1.c157420000000p-27f, 0x1.17ed480000000p+0f, 0x1.a56ef00000000p-26f,
    0x1.18af940000000p+0f, -0x1.dcdc860000000p-26f, 0x1.1972660000000p+0f, -0x1.f228b40000000p-26f,
    0x1.1a35be0000000p+0f, 0x1.6df96e0000000p-25f, 0x1.1af9a00000000p+0f, -0x1.fb1d780000000p-26f,
    0x1.1bbe080000000p+0f, 0x1.0117340000000p-26f, 0x1.1c82fa0000000p+0f, -0x1.5afc720000000p-25f,
    0x1.1d48740000000p+0f, -0x1.d2e8ca0000000p-25f, 0x1.1e0e760000000p+0f, -0x1.4bbfda0000000p-28f,
    0x1.1ed5020000000p+0f, 0x1.7e6c8e0000000p-27f, 0x1.1f9c180000000p+0f, 0x1.0e33940000000p-26f,
    0x1.2063b80000000p+0f, 0x1.0c519a0000000p-25f, 0x1.212be40000000p+0f, -0x1.50eafc0000000p-25f,
    0x1.21f49a0000000p+0f, -0x1.d0446e0000000p-25f, 0x1.22bdda0000000p+0f, 0x1.3c89680000000p-27f,
    0x1.2387a60000000p+0f, 0x1.ceac480000000p-25f, 0x1.2452000000000p+0f, -0x1.1f7afe0000000p-26f,
    0x1.251ce40000000p+0f, 0x1.f654c80000000p-25f, 0x1.25e8580000000p+0f, -0x1.dc26320000000p-25f,
    0x1.26b4560000000p+0f, 0x1.789f380000000p-26f, 0x1.2780e40000000p+0f, -0x1.7c441a0000000p-25f,
    0x1.284dfe0000000p+0f, 0x1.f563800000000p-28f, 0x1.291ba80000000p+0f, -0x1.4dc8920000000p-25f,
    0x1.29e9e00000000p+0f, -0x1.5c04240000000p-25f, 0x1.2ab8a60000000p+0f, 0x1.b443c40000000p-26f,
    0x1.2b87fe0000000p+0f, -0x1.e4a4ce0000000p-25f, 0x1.2c57e40000000p+0f, -0x1.a239340000000p-26f,
    0x1.2d285a0000000p+0f, 0x1.b900c20000000p-26f, 0x1.2df9620000000p+0f, -0x1.37d4ee0000000p-29f,
    0x1.2ecafa0000000p+0f, 0x1.27c5ea0000000p-25f, 0x1.2f9d240000000p+0f, 0x1.57b10e0000000p-25f,
    0x1.306fe00000000p+0f, 0x1.4636e20000000p-25f, 0x1.31432e0000000p+0f, 0x1.bdd6600000000p-25f,
    0x1.3217100000000p+0f, -0x1.d993e80000000p-27f, 0x1.32eb840000000p+0f, -0x1.15c5740000000p-26f,
    0x1.33c08c0000000p+0f, -0x1.b37d200000000p-25f, 0x1.3496260000000p+0f, 0x1.b8fe8c0000000p-26f,
    0x1.356c560000000p+0f, -0x1.b5803c0000000p-30f, 0x1.36431a0000000p+0f, 0x1.6f441e0000000p-27f,
    0x1.371a740000000p+0f, -0x1.18aac60000000p-25f, 0x1.37f2620000000p+0f, 0x1.8f3aa40000000p-27f,
    0x1.38cae60000000p+0f, 0x1.a0bb0c0000000p-25f, 0x1.39a4020000000p+0f, -0x1.23afc40000000p-26f,
    0x1.3a7db40000000p+0f, -0x1.634c020000000p-25f, 0x1.3b57fc0000000p+0f, -0x1.3930ba0000000p-32f,
    0x1.3c32dc0000000p+0f, 0x1.89d4720000000p-27f, 0x1.3d0e540000000p+0f, 0x1.3b785c0000000p-26f,
    0x1.3dea640000000p+0f, 0x1.8246840000000p-25f, 0x1.3ec70e0000000p+0f, -0x1.c75d160000000p-29f,
    0x1.3fa4500000000p+0f, 0x1.2b20060000000p-26f, 0x1.40822c0000000p+0f, 0x1.b3d0120000000p-27f,
    0x1.4160a20000000p+0f, 0x1.f72e2a0000000p-28f, 0x1.423fb20000000p+0f, 0x1.c251a20000000p-26f,
    0x1.431f5e0000000p+0f, -0x1.abd5da0000000p-26f, 0x1.43ffa40000000p+0f, -0x1.ed18b00000000p-30f,
    0x1.44e0860000000p+0f, 0x1.8624b40000000p-30f, 0x1.45c2040000000p+0f, 0x1.53e9180000000p-27f,
    0x1.46a41e0000000p+0f, 0x1.a3a00a0000000p-25f, 0x1.4786d60000000p+0f, 0x1.a2cc8e0000000p-26f,
    0x1.486a2c0000000p+0f, -0x1.47d8660000000p-25f, 0x1.494e1e0000000p+0f, 0x1.92aed20000000p-28f,
    0x1.4a32b00000000p+0f, -0x1.e505840000000p-25f, 0x1.4b17de0000000p+0f, 0x1.4db6fa0000000p-25f,
    0x1.4bfdae0000000p+0f, -0x1.593abc0000000p-25f, 0x1.4ce41c0000000p+0f, -0x1.fa0fba0000000p-26f,
    0x1.4dcb2a0000000p+0f, -0x1.8088bc0000000p-26f, 0x1.4eb2d80000000p+0f, 0x1.d8abfe0000000p-28f,
    0x1.4f9b280000000p+0f, -0x1.2c5a6c0000000p-25f, 0x1.5084180000000p+0f, -0x1.759c240000000p-29f,
    0x1.516daa0000000p+0f, 0x1.67b3200000000p-27f, 0x1.5257de0000000p+0f, 0x1.07e9de0000000p-25f,
    0x1.5342b60000000p+0f, -0x1.2c56100000000p-25f, 0x1.542e300000000p+0f, -0x1.612a5c0000000p-25f,
    0x1.551a4c0000000p+0f, 0x1.4bb2420000000p-25f, 0x1.56070e0000000p+0f, -0x1.0b77980000000p-27f,
    0x1.56f4740000000p+0f, -0x1.295b040000000p-25f, 0x1.57e27e0000000p+0f, -0x1.074ecc0000000p-26f,
    0x1.58d12e0000000p+0f, -0x1.6d07000000000p-25f, 0x1.59c0820000000p+0f, 0x1.ffc1f20000000p-26f,
    0x1.5ab07e0000000p+0f, -0x1.5bd5ec0000000p-27f, 0x1.5ba1200000000p+0f, -0x1.15e1800000000p-26f,
    0x1.5c92680000000p+0f, 0x1.4b28d60000000p-25f, 0x1.5d845a0000000p+0f, -0x1.ecce8e0000000p-25f,
    0x1.5e76f20000000p+0f, -0x1.4a5bd60000000p-25f, 0x1.5f6a320000000p+0f, 0x1.b9d6e20000000p-29f,
    0x1.605e1c0000000p+0f, -0x1.a248fe0000000p-26f, 0x1.6152ae0000000p+0f, 0x1.b37dbe0000000p-26f,
    0x1.6247ec0000000p+0f, -0x1.f8b5500000000p-25f, 0x1.633dd20000000p+0f, -0x1.736b020000000p-27f,
    0x1.6434640000000p+0f, -0x1.66679c0000000p-25f, 0x1.652ba00000000p+0f, -0x1.43704a0000000p-28f,
    0x1.6623880000000p+0f, 0x1.2a91120000000p-27f, 0x1.671c1c0000000p+0f, 0x1.c20cfe0000000p-26f,
    0x1.68155e0000000p+0f, -0x1.766ad20000000p-25f, 0x1.690f4c0000000p+0f, -0x1.cc2d580000000p-25f,
    0x1.6a09e60000000p+0f, 0x1.9fcef40000000p-26f, 0x1.6b05300000000p+0f, -0x1.62ba300000000p-26f,
    0x1.6c01280000000p+0f, -0x1.5e84a80000000p-25f, 0x1.6cfdce0000000p+0f, -0x1.15c4de0000000p-27f,
    0x1.6dfb240000000p+0f, -0x1.cd72e80000000p-27f, 0x1.6ef92a0000000p+0f, -0x1.e9b1460000000p-26f,
    0x1.6ff7e00000000p+0f, -0x1.ab9ae00000000p-26f, 0x1.70f7460000000p+0f, 0x1.bd0ba20000000p-26f,
    0x1.71f75e0000000p+0f, 0x1.1d8bee0000000p-25f, 0x1.72f8280000000p+0f, 0x1.bab4220000000p-26f,
    0x1.73f9a40000000p+0f, 0x1.14b02e0000000p-25f, 0x1.74fbd40000000p+0f, -0x1.4506800000000p-25f,
    0x1.75feb60000000p+0f, -0x1.37b3060000000p-25f, 0x1.77024c0000000p+0f, -0x1.ca923e0000000p-25f,
    0x1.7806940000000p+0f, 0x1.fbcba80000000p-25f, 0x1.790b940000000p+0f, -0x1.d4f8c20000000p-26f,
    0x1.7a11480000000p+0f, -0x1.829fd00000000p-25f, 0x1.7b17b00000000p+0f, 0x1.2ed9fc0000000p-25f,
    0x1.7c1ed00000000p+0f, 0x1.30c1320000000p-28f, 0x1.7d26a60000000p+0f, 0x1.7fc3780000000p-27f,
    0x1.7e2f340000000p+0f, -0x1.2616340000000p-25f, 0x1.7f38780000000p+0f, 0x1.2471240000000p-26f,
    0x1.8042760000000p+0f, -0x1.783cbe0000000p-25f, 0x1.814d2a0000000p+0f, 0x1.ba20dc0000000p-25f,
    0x1.82589a0000000p+0f, -0x1.accc7c0000000p-26f, 0x1.8364c20000000p+0f, -0x1.46be080000000p-28f,
    0x1.8471a40000000p+0f, 0x1.88f1ec0000000p-26f, 0x1.857f420000000p+0f, -0x1.0c149c0000000p-25f,
    0x1.868d9a0000000p+0f, -0x1.2edb440000000p-26f, 0x1.879cae0000000p+0f, -0x1.b396f20000000p-26f,
    0x1.88ac7e0000000p+0f, -0x1.9d665a0000000p-26f, 0x1.89bd0a0000000p+0f, 0x1.1e16040000000p-26f,
    0x1.8ace540000000p+0f, 0x1.15506e0000000p-27f, 0x1.8be05c0000000p+0f, -0x1.4a7a220000000p-26f,
    0x1.8cf3220000000p+0f, -0x1.29576e0000000p-25f, 0x1.8e06a60000000p+0f, -0x1.f799280000000p-28f,
    0x1.8f1aea0000000p+0f, -0x1.baa2320000000p-26f, 0x1.902fee0000000p+0f, -0x1.fafa6e0000000p-25f,
    0x1.9145b00000000p+0f, 0x1.723ff80000000p-25f, 0x1.925c360000000p+0f, -0x1.8aba040000000p-25f,
    0x1.93737c0000000p+0f, -0x1.e647440000000p-25f, 0x1.948b820000000p+0f, 0x1.6bf31c0000000p-25f,
    0x1.95a44c0000000p+0f, 0x1.790a420000000p-25f, 0x1.96bdda0000000p+0f, -0x1.6263d40000000p-26f,
    0x1.97d82a0000000p+0f, -0x1.0d8d840000000p-31f, 0x1.98f33e0000000p+0f, 0x1.1e88a80000000p-26f,
    0x1.9a0f180000000p+0f, -0x1.e6bf080000000p-25f, 0x1.9b2bb40000000p+0f, 0x1.aa7fc20000000p-25f,
    0x1.9c49180000000p+0f, 0x1.51f8480000000p-27f, 0x1.9d67420000000p+0f, -0x1.ad11ca0000000p-26f,
    0x1.9e86320000000p+0f, -0x1.8737380000000p-26f, 0x1.9fa5e80000000p+0f, 0x1.a0fe540000000p-25f,
    0x1.a0c6680000000p+0f, -0x1.2886a60000000p-26f, 0x1.a1e7ae0000000p+0f, 0x1.b1d7180000000p-25f,
    0x1.a309be0000000p+0f, 0x1.8945a60000000p-25f, 0x1.a42c980000000p+0f, 0x1.182b5e0000000p-30f,
    0x1.a5503c0000000p+0f, -0x1.b83b540000000p-25f, 0x1.a674a80000000p+0f, 0x1.5e8c0a0000000p-25f,
    0x1.a799e20000000p+0f, -0x1.99e9940000000p-25f, 0x1.a8bfe60000000p+0f, -0x1.87da340000000p-25f,
    0x1.a9e6b60000000p+0f, -0x1.50c0480000000p-25f, 0x1.ab0e520000000p+0f, 0x1.356eba0000000p-28f,
    0x1.ac36bc0000000p+0f, -0x1.6064320000000p-31f, 0x1.ad5ff40000000p+0f, -0x1.70f6220000000p-26f,
    0x1.ae89fa0000000p+0f, -0x1.a94b140000000p-26f, 0x1.afb4ce0000000p+0f, 0x1.88bcc00000000p-26f,
    0x1.b0e0720000000p+0f, 0x1.31b6cc0000000p-25f, 0x1.b20ce60000000p+0f, 0x1.93512a0000000p-25f,
    0x1.b33a2c0000000p+0f, -0x1.ec3a820000000p-26f, 0x1.b468420000000p+0f, -0x1.4916ca0000000p-25f,
    0x1.b597280000000p+0f, 0x1.bcab280000000p-25f, 0x1.b6c6e20000000p+0f, 0x1.3e38a60000000p-25f,
    0x1.b7f7700000000p+0f, -0x1.a094380000000p-25f, 0x1.b928d00000000p+0f, -0x1.bb16c40000000p-25f,
    0x1.ba5b040000000p+0f, -0x1.ebdf360000000p-25f, 0x1.bb8e0c0000000p+0f, -0x1.0cb21c0000000p-25f,
    0x1.bcc1ea0000000p+0f, -0x1.f687c60000000p-25f, 0x1.bdf69c0000000p+0f, 0x1.f9d1040000000p-27f,
    0x1.bf2c260000000p+0f, -0x1.0a387e0000000p-26f, 0x1.c062860000000p+0f, 0x1.41b33c0000000p-28f,
    0x1.c199be0000000p+0f, -0x1.3d56b20000000p-27f, 0x1.c2d1ce0000000p+0f, -0x1.8166b60000000p-26f,
    0x1.c40ab60000000p+0f, -0x1.7c2c980000000p-39f, 0x1.c544780000000p+0f, -0x1.c141380000000p-26f,
    0x1.c67f120000000p+0f, 0x1.cafa2a0000000p-25f, 0x1.c7ba880000000p+0f, 0x1.3119260000000p-25f,
    0x1.c8f6da0000000p+0f, -0x1.7f230a0000000p-25f, 0x1.ca34060000000p+0f, -0x1.15c7640000000p-25f,
    0x1.cb720e0000000p+0f, -0x1.8837cc0000000p-27f, 0x1.ccb0f20000000p+0f, 0x1.cda2ce0000000p-25f,
    0x1.cdf0b60000000p+0f, -0x1.5447800000000p-25f, 0x1.cf31560000000p+0f, -0x1.2915240000000p-26f,
    0x1.d072d40000000p+0f, 0x1.40f1300000000p-25f, 0x1.d1b5320000000p+0f, 0x1.61192e0000000p-25f,
    0x1.d2f8700000000p+0f, 0x1.01b13e0000000p-25f, 0x1.d43c8e0000000p+0f, 0x1.59543a0000000p-25f,
    0x1.d5818e0000000p+0f, -0x1.822dbc0000000p-27f, 0x1.d6c76e0000000p+0f, 0x1.0c5cda0000000p-25f,
    0x1.d80e320000000p+0f, -0x1.26cf8e0000000p-25f, 0x1.d955d80000000p+0f, -0x1.c013f20000000p-25f,
    0x1.da9e600000000p+0f, 0x1.ed99420000000p-27f, 0x1.dbe7ce0000000p+0f, -0x1.38af9e0000000p-25f,
    0x1.dd32200000000p+0f, -0x1.9fc9740000000p-25f, 0x1.de7d560000000p+0f, 0x1.0701960000000p-26f,
    0x1.dfc9740000000p+0f, -0x1.908c940000000p-25f, 0x1.e116760000000p+0f, 0x1.632fa20000000p-25f,
    0x1.e264620000000p+0f, -0x1.614bda0000000p-25f, 0x1.e3b3340000000p+0f, -0x1.3a447c0000000p-26f,
    0x1.e502ee0000000p+0f, 0x1.e2cffe0000000p-26f, 0x1.e653920000000p+0f, 0x1.19db5e0000000p-26f,
    0x1.e7a5200000000p+0f, -0x1.0e2ce00000000p-26f, 0x1.e8f7980000000p+0f, -0x1.0649180000000p-25f,
    0x1.ea4afa0000000p+0f, 0x1.52486c0000000p-27f, 0x1.eb9f480000000p+0f, 0x1.9f329c0000000p-26f,
    0x1.ecf4820000000p+0f, 0x1.b1ccfe0000000p-25f, 0x1.ee4aaa0000000p+0f, 0x1.0c42880000000p-27f,
    0x1.efa1be0000000p+0f, 0x1.cc2b440000000p-25f, 0x1.f0f9c20000000p+0f, -0x1.a4df6c0000000p-27f,
    0x1.f252b40000000p+0f, -0x1.1288ae0000000p-25f, 0x1.f3ac940000000p+0f, 0x1.1bae4e0000000p-25f,
    0x1.f507660000000p+0f, -0x1.246eb00000000p-26f, 0x1.f663280000000p+0f, -0x1.9deec20000000p-26f,
    0x1.f7bfda0000000p+0f, 0x1.b397c20000000p-25f, 0x1.f91d800000000p+0f, 0x1.121e440000000p-27f,
    0x1.fa7c180000000p+0f, 0x1.9e90d80000000p-28f, 0x1.fbdba40000000p+0f, -0x1.2da55e0000000p-25f,
    0x1.fd3c220000000p+0f, 0x1.71ee3e0000000p-25f, 0x1.fe9d960000000p+0f, 0x1.65447c0000000p-25f,
};

/* the lanes in bad, from their saved arguments, through the definition */
static void exf_fallback(const float *xs, float *y, unsigned bad) {
    while (bad) {
        int l = __builtin_ctz(bad);
        y[l] = exact_expf(xs[l]);
        bad &= bad - 1;
    }
}
#endif

#if defined(__AVX512F__)
EXF_FN void exact_expf_n(const float *x, float *y, int64_t n) {
    EXF_NOFUSE
    const __m512 invl = _mm512_set1_ps(EXF_INVL), shift = _mm512_set1_ps(EXF_SHIFT);
    const __m512 la = _mm512_set1_ps(EXF_LA), lb = _mm512_set1_ps(EXF_LB), lc = _mm512_set1_ps(EXF_LC);
    const __m512 nll = _mm512_set1_ps(EXF_NLL), split = _mm512_set1_ps(EXF_SPLIT);
    const __m512 c2 = _mm512_set1_ps(EXF_C2V), c3 = _mm512_set1_ps(EXF_C3V), d = _mm512_set1_ps(EXF_D);
    const __m512 xnorm = _mm512_set1_ps(EXF_X_NORM), xovf = _mm512_set1_ps(EXF_X_OVF);
    const __m512 xuf = _mm512_set1_ps(EXF_X_UF), g22 = _mm512_set1_ps(0x1p-22f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 kf = _mm512_sub_ps(_mm512_add_ps(_mm512_mul_ps(vx, invl), shift), shift);
        __m512i k = _mm512_cvttps_epi32(kf);
        __m512i j2 = _mm512_slli_epi32(_mm512_and_si512(k, _mm512_set1_epi32(255)), 1);
        __m512 rh = _mm512_sub_ps(_mm512_sub_ps(_mm512_sub_ps(vx, _mm512_mul_ps(kf, la)), _mm512_mul_ps(kf, lb)),
                                  _mm512_mul_ps(kf, lc));
        __m512 rl = _mm512_mul_ps(kf, nll);
        __m512 rs = _mm512_add_ps(rh, rl);
        __m512 s = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(rs, rs), _mm512_add_ps(_mm512_mul_ps(c3, rs), c2)), rl);
        __m512 th = _mm512_i32gather_ps(j2, exf_tab, 4);
        __m512 tl = _mm512_i32gather_ps(_mm512_add_epi32(j2, _mm512_set1_epi32(1)), exf_tab, 4);
        __m512 ph = _mm512_mul_ps(th, rh);
        __m512 ct = _mm512_mul_ps(th, split), cr = _mm512_mul_ps(rh, split);
        __m512 th1 = _mm512_sub_ps(ct, _mm512_sub_ps(ct, th)), th2 = _mm512_sub_ps(th, th1);
        __m512 rh1 = _mm512_sub_ps(cr, _mm512_sub_ps(cr, rh)), rh2 = _mm512_sub_ps(rh, rh1);
        __m512 pl = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(th1, rh1), ph), _mm512_mul_ps(th1, rh2));
        pl = _mm512_add_ps(_mm512_add_ps(pl, _mm512_mul_ps(th2, rh1)), _mm512_mul_ps(th2, rh2));
        __m512 yh = _mm512_add_ps(th, ph);
        __m512 e = _mm512_add_ps(_mm512_sub_ps(th, yh), ph);
        __m512 a = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(tl, rs), tl), pl), e);
        __m512 yl = _mm512_add_ps(_mm512_mul_ps(th, s), a);
        __m512i ee = _mm512_srai_epi32(k, 8);
        __m512 u1 = _mm512_add_ps(yh, _mm512_sub_ps(yl, d)), u2 = _mm512_add_ps(yh, _mm512_add_ps(yl, d));
        __m512i bits = _mm512_add_epi32(_mm512_castps_si512(u1), _mm512_slli_epi32(ee, 23));
        __mmask16 ok = _mm512_cmp_ps_mask(u1, u2, _CMP_EQ_OQ);
        __mmask16 sub = _mm512_cmp_ps_mask(vx, xnorm, _CMP_LT_OQ);
        if (sub) { /* exp(x) < 2^-126: m + y rounds y on the grid 2^-149 (bench_expf32.c finish32) */
            __m512i mb = _mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(1), ee), 23);
            __m512 m = _mm512_castsi512_ps(mb);
            __m512 g = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(-22), ee), 23));
            __m512 s2 = _mm512_add_ps(m, yh);
            __m512 v = _mm512_add_ps(_mm512_add_ps(_mm512_sub_ps(m, s2), yh), yl);
            __m512 d2 = _mm512_add_ps(_mm512_mul_ps(g, g22), d);
            __m512 w1 = _mm512_add_ps(s2, _mm512_sub_ps(v, d2)), w2 = _mm512_add_ps(s2, _mm512_add_ps(v, d2));
            bits = _mm512_mask_mov_epi32(bits, sub, _mm512_sub_epi32(_mm512_castps_si512(w1), mb));
            ok = (__mmask16)((ok & ~sub) | (_mm512_cmp_ps_mask(w1, w2, _CMP_EQ_OQ) & sub));
        }
        __mmask16 nan = _mm512_cmp_ps_mask(vx, vx, _CMP_UNORD_Q);
        __mmask16 ovf = _mm512_cmp_ps_mask(vx, xovf, _CMP_GT_OQ);
        __mmask16 uf = _mm512_cmp_ps_mask(vx, xuf, _CMP_LT_OQ);
        bits = _mm512_mask_mov_epi32(bits, ovf, _mm512_set1_epi32(0x7F800000));
        bits = _mm512_mask_mov_epi32(bits, uf, _mm512_setzero_si512());
        bits = _mm512_mask_mov_epi32(bits, nan, _mm512_castps_si512(_mm512_add_ps(vx, vx)));
        unsigned bad = (unsigned)(uint16_t)~(ok | nan | ovf | uf);
        if (bad == 0) {
            _mm512_storeu_si512((void *)(y + i), bits);
        } else { /* the arguments saved first: y may be x */
            float xs[16];
            _mm512_storeu_ps(xs, vx);
            _mm512_storeu_si512((void *)(y + i), bits);
            exf_fallback(xs, y + i, bad);
        }
    }
    for (; i < n; i++) y[i] = exact_expf(x[i]);
}
#elif defined(__AVX2__)
EXF_FN void exact_expf_n(const float *x, float *y, int64_t n) {
    EXF_NOFUSE
    const __m256 invl = _mm256_set1_ps(EXF_INVL), shift = _mm256_set1_ps(EXF_SHIFT);
    const __m256 la = _mm256_set1_ps(EXF_LA), lb = _mm256_set1_ps(EXF_LB), lc = _mm256_set1_ps(EXF_LC);
    const __m256 nll = _mm256_set1_ps(EXF_NLL), split = _mm256_set1_ps(EXF_SPLIT);
    const __m256 c2 = _mm256_set1_ps(EXF_C2V), c3 = _mm256_set1_ps(EXF_C3V), d = _mm256_set1_ps(EXF_D);
    const __m256 xnorm = _mm256_set1_ps(EXF_X_NORM), xovf = _mm256_set1_ps(EXF_X_OVF);
    const __m256 xuf = _mm256_set1_ps(EXF_X_UF), g22 = _mm256_set1_ps(0x1p-22f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 kf = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(vx, invl), shift), shift);
        __m256i k = _mm256_cvttps_epi32(kf);
        __m256i j2 = _mm256_slli_epi32(_mm256_and_si256(k, _mm256_set1_epi32(255)), 1);
        __m256 rh = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(vx, _mm256_mul_ps(kf, la)), _mm256_mul_ps(kf, lb)),
                                  _mm256_mul_ps(kf, lc));
        __m256 rl = _mm256_mul_ps(kf, nll);
        __m256 rs = _mm256_add_ps(rh, rl);
        __m256 s = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(rs, rs), _mm256_add_ps(_mm256_mul_ps(c3, rs), c2)), rl);
        __m256 th = _mm256_i32gather_ps(exf_tab, j2, 4);
        __m256 tl = _mm256_i32gather_ps(exf_tab + 1, j2, 4);
        __m256 ph = _mm256_mul_ps(th, rh);
        __m256 ct = _mm256_mul_ps(th, split), cr = _mm256_mul_ps(rh, split);
        __m256 th1 = _mm256_sub_ps(ct, _mm256_sub_ps(ct, th)), th2 = _mm256_sub_ps(th, th1);
        __m256 rh1 = _mm256_sub_ps(cr, _mm256_sub_ps(cr, rh)), rh2 = _mm256_sub_ps(rh, rh1);
        __m256 pl = _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(th1, rh1), ph), _mm256_mul_ps(th1, rh2));
        pl = _mm256_add_ps(_mm256_add_ps(pl, _mm256_mul_ps(th2, rh1)), _mm256_mul_ps(th2, rh2));
        __m256 yh = _mm256_add_ps(th, ph);
        __m256 e = _mm256_add_ps(_mm256_sub_ps(th, yh), ph);
        __m256 a = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(tl, rs), tl), pl), e);
        __m256 yl = _mm256_add_ps(_mm256_mul_ps(th, s), a);
        __m256i ee = _mm256_srai_epi32(k, 8);
        __m256 u1 = _mm256_add_ps(yh, _mm256_sub_ps(yl, d)), u2 = _mm256_add_ps(yh, _mm256_add_ps(yl, d));
        __m256i bits = _mm256_add_epi32(_mm256_castps_si256(u1), _mm256_slli_epi32(ee, 23));
        __m256 ok = _mm256_cmp_ps(u1, u2, _CMP_EQ_OQ);
        __m256 sub = _mm256_cmp_ps(vx, xnorm, _CMP_LT_OQ);
        if (_mm256_movemask_ps(sub)) { /* exp(x) < 2^-126: m + y rounds y on the grid 2^-149 */
            __m256i mb = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(1), ee), 23);
            __m256 m = _mm256_castsi256_ps(mb);
            __m256 g = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_sub_epi32(_mm256_set1_epi32(-22), ee), 23));
            __m256 s2 = _mm256_add_ps(m, yh);
            __m256 v = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(m, s2), yh), yl);
            __m256 d2 = _mm256_add_ps(_mm256_mul_ps(g, g22), d);
            __m256 w1 = _mm256_add_ps(s2, _mm256_sub_ps(v, d2)), w2 = _mm256_add_ps(s2, _mm256_add_ps(v, d2));
            __m256i sbits = _mm256_sub_epi32(_mm256_castps_si256(w1), mb);
            bits = _mm256_blendv_epi8(bits, sbits, _mm256_castps_si256(sub));
            ok = _mm256_blendv_ps(ok, _mm256_cmp_ps(w1, w2, _CMP_EQ_OQ), sub);
        }
        __m256 nan = _mm256_cmp_ps(vx, vx, _CMP_UNORD_Q);
        __m256 ovf = _mm256_cmp_ps(vx, xovf, _CMP_GT_OQ);
        __m256 uf = _mm256_cmp_ps(vx, xuf, _CMP_LT_OQ);
        bits = _mm256_blendv_epi8(bits, _mm256_set1_epi32(0x7F800000), _mm256_castps_si256(ovf));
        bits = _mm256_blendv_epi8(bits, _mm256_setzero_si256(), _mm256_castps_si256(uf));
        bits = _mm256_blendv_epi8(bits, _mm256_castps_si256(_mm256_add_ps(vx, vx)), _mm256_castps_si256(nan));
        unsigned spec = (unsigned)_mm256_movemask_ps(_mm256_or_ps(nan, _mm256_or_ps(ovf, uf)));
        unsigned bad = ~((unsigned)_mm256_movemask_ps(ok) | spec) & 0xFFu;
        if (bad == 0) {
            _mm256_storeu_si256((__m256i *)(void *)(y + i), bits);
        } else { /* the arguments saved first: y may be x */
            float xs[8];
            _mm256_storeu_ps(xs, vx);
            _mm256_storeu_si256((__m256i *)(void *)(y + i), bits);
            exf_fallback(xs, y + i, bad);
        }
    }
    for (; i < n; i++) y[i] = exact_expf(x[i]);
}
#else
EXF_FN void exact_expf_n(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = exact_expf(x[i]);
}
#endif

#endif /* COLI_EXACT_EXPF_H */
