/* k3_exact.h -- exp_f's and MinGW tanhf's float bits from double arithmetic, for a device
 * (CUDA) that cannot call either, and for the host tool that checks them on every float.
 *
 * Kimi K3's SiTU (b1 tanh(g/b1) sigmoid(g) b2 tanh(u/b2)) calls exp_f (expf's bits: e^x
 * within about 2^-62, rounded to float) and the C library's tanhf (MinGW's: tanh rounded to
 * float, as (float)tanh((double)x) and (float)tanhl(x) give). Both are the correctly rounded
 * float of the exact value except where it lies very near a point halfway between two
 * floats. Here e^x and tanh(x) are computed in double within about 2^-50 and rounded from
 * 2^-44 below and from 2^-44 above: where the two roundings agree, every number between them
 * rounds the same way, the reference's among them. The inputs where they differ (a few
 * thousand of the 2^32 floats, k3_exact_tables.h) carry the reference's bits in a sorted
 * table; tools/k3_exact_tables.c writes it and checks every other float against the
 * reference, and the CUDA test runs the device's code on all of them.
 *
 * k3x_exp_d is exp_f8_pd's arithmetic (exp_f.h), one lane at a time: the same uncertain set.
 * Every double operation is an explicit IEEE one (fused where written FMA, never contracted:
 * the host's products go through k3x_apart), so host and device give the same bits. */
#ifndef K3_EXACT_H
#define K3_EXACT_H

#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define K3X_FN static __host__ __device__ __forceinline__
#else
#define K3X_FN static inline
#include <math.h>
#endif

K3X_FN double k3x_fma(double a, double b, double c){
#ifdef __CUDA_ARCH__
    return __fma_rn(a, b, c);
#else
    return fma(a, b, c);
#endif
}
/* a product the host compiler may not fuse with the add after it (-ffp-contract) */
K3X_FN double k3x_mul(double a, double b){
#ifdef __CUDA_ARCH__
    return __dmul_rn(a, b);
#else
    double r = a * b;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __asm__("" : "+x"(r));
#endif
    return r;
#endif
}
K3X_FN double k3x_add(double a, double b){
#ifdef __CUDA_ARCH__
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
}
K3X_FN double k3x_div(double a, double b){
#ifdef __CUDA_ARCH__
    return __ddiv_rn(a, b);
#else
    return a / b;
#endif
}
K3X_FN float k3x_d2f(double a){
#ifdef __CUDA_ARCH__
    return __double2float_rn(a);
#else
    return (float)a;
#endif
}
#ifdef __CUDA_ARCH__
K3X_FN uint64_t k3x_dbits(double a){ return (uint64_t)__double_as_longlong(a); }
K3X_FN double k3x_bitsd(uint64_t u){ return __longlong_as_double((long long)u); }
K3X_FN uint32_t k3x_fbits(float a){ return __float_as_uint(a); }
K3X_FN float k3x_bitsf(uint32_t u){ return __uint_as_float(u); }
#else
K3X_FN uint64_t k3x_dbits(double a){ uint64_t u; memcpy(&u, &a, 8); return u; }
K3X_FN double k3x_bitsd(uint64_t u){ double a; memcpy(&a, &u, 8); return a; }
K3X_FN uint32_t k3x_fbits(float a){ uint32_t u; memcpy(&u, &a, 4); return u; }
K3X_FN float k3x_bitsf(uint32_t u){ float a; memcpy(&a, &u, 4); return a; }
#endif

/* e^x for x in [-150, 100]: 2^n times a degree-12 polynomial on |r| <= ln2/2 (exp_f8_pd) */
K3X_FN double k3x_exp_d(double x){
    const double shift = 0x1.8p52;
    double t = k3x_fma(x, 0x1.71547652b82fep0, shift);
    double n = t - shift;
    double r = k3x_fma(-n, 0x1.62e42fefa39efp-1, x);
    r = k3x_fma(-n, 0x1.abc9e3b39803fp-56, r);
    double p = 1.0 / 479001600.0;
    p = k3x_fma(p, r, 1.0 / 39916800.0);
    p = k3x_fma(p, r, 1.0 / 3628800.0);
    p = k3x_fma(p, r, 1.0 / 362880.0);
    p = k3x_fma(p, r, 1.0 / 40320.0);
    p = k3x_fma(p, r, 1.0 / 5040.0);
    p = k3x_fma(p, r, 1.0 / 720.0);
    p = k3x_fma(p, r, 1.0 / 120.0);
    p = k3x_fma(p, r, 1.0 / 24.0);
    p = k3x_fma(p, r, 1.0 / 6.0);
    p = k3x_fma(p, r, 0.5);
    p = k3x_fma(p, r, 1.0);
    p = k3x_fma(p, r, 1.0);
    return k3x_mul(p, k3x_bitsd((k3x_dbits(t) + 1023) << 52));
}

/* e^z - 1 for z in [0, 40]: z = n ln2 + r, |r| <= ln2/2, e^r - 1 = r (1 + r/2 + ... + r^14/15!)
 * (the terms past r^14/15! are below 2^-60 of it), then 2^n (e^r - 1) + (2^n - 1) in one
 * rounding: within about 2^-51 of e^z - 1 */
K3X_FN double k3x_expm1_d(double z){
    const double shift = 0x1.8p52;
    double t = k3x_fma(z, 0x1.71547652b82fep0, shift);
    double n = t - shift;
    double r = k3x_fma(-n, 0x1.62e42fefa39efp-1, z);
    r = k3x_fma(-n, 0x1.abc9e3b39803fp-56, r);
    double q = 1.0 / 1307674368000.0;           /* 1/15! */
    q = k3x_fma(q, r, 1.0 / 87178291200.0);     /* 1/14! */
    q = k3x_fma(q, r, 1.0 / 6227020800.0);
    q = k3x_fma(q, r, 1.0 / 479001600.0);
    q = k3x_fma(q, r, 1.0 / 39916800.0);
    q = k3x_fma(q, r, 1.0 / 3628800.0);
    q = k3x_fma(q, r, 1.0 / 362880.0);
    q = k3x_fma(q, r, 1.0 / 40320.0);
    q = k3x_fma(q, r, 1.0 / 5040.0);
    q = k3x_fma(q, r, 1.0 / 720.0);
    q = k3x_fma(q, r, 1.0 / 120.0);
    q = k3x_fma(q, r, 1.0 / 24.0);
    q = k3x_fma(q, r, 1.0 / 6.0);
    q = k3x_fma(q, r, 0.5);
    q = k3x_fma(q, r, 1.0);
    double em = k3x_mul(r, q);                  /* e^r - 1 */
    if (n == 0.0) return em;
    double p2 = k3x_bitsd((k3x_dbits(t) + 1023) << 52);   /* 2^n, n in 1..58 */
    return k3x_fma(p2, em, p2 - 1.0);
}

/* tanh(x) for |x| <= 20 (tanhf is 1 from 9.02 on): (e^2a - 1) / (e^2a - 1 + 2), a = |x|,
 * the sign put back; within about 2^-50 */
K3X_FN double k3x_tanh_d(double x){
    double a = x < 0 ? -x : x;
    double em = k3x_expm1_d(a + a);
    double y = k3x_div(em, k3x_add(em, 2.0));
    return x < 0 ? -y : y;
}

/* The two roundings of y from 2^-44 below and above; 1 when they agree (*f then holds it) */
K3X_FN int k3x_round_sure(double y, float *f){
    float lo = k3x_d2f(k3x_fma(-y, 0x1p-44, y)), hi = k3x_d2f(k3x_fma(y, 0x1p-44, y));
    *f = lo;
    return k3x_fbits(lo) == k3x_fbits(hi);
}

/* The fast parts: 1 and the reference's bits in *f, or 0 where the table decides (an
 * uncertain rounding, or a NaN: exp_f and tanhf give it back quieted). */
K3X_FN int k3x_expf_fast(float x, float *f){
    if (x != x) { *f = k3x_bitsf(k3x_fbits(x) | 0x00400000u); return 1; }
    float c = x < -150.0f ? -150.0f : x > 100.0f ? 100.0f : x;
    return k3x_round_sure(k3x_exp_d((double)c), f);
}
K3X_FN int k3x_tanhf_fast(float x, float *f){
    if (x != x) { *f = k3x_bitsf(k3x_fbits(x) | 0x00400000u); return 1; }
    float c = x < -20.0f ? -20.0f : x > 20.0f ? 20.0f : x;
    return k3x_round_sure(k3x_tanh_d((double)c), f);
}

/* The table's entry for x: n sorted (input bits, reference bits) pairs */
K3X_FN float k3x_table(const uint32_t *in, const uint32_t *out, int n, float x){
    uint32_t b = k3x_fbits(x);
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (in[mid] == b) return k3x_bitsf(out[mid]);
        if (in[mid] < b) lo = mid + 1; else hi = mid - 1;
    }
    return k3x_bitsf(0x7fc00000u);   /* not in the table: never, for the tables of k3_exact_tables.h */
}

#endif
