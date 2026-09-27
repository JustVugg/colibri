/* exact_dot.h — order-independent dot products for the opt-in exact verify mode.
 *
 * Every product is formed exactly (integer mantissas, never rounded) and accumulated by
 * exponent bin; the bins are folded into one wide two's-complement integer and rounded to
 * double ONCE, correctly (round-to-nearest-even), then to float. No summation order, no
 * contraction flag and no SIMD width can change the result, so CPU and GPU agree to the bit
 * by construction and near-ties are decided identically everywhere. The cost is throughput:
 * an integer path, no FMA, no tensor cores. That is why it is opt-in and verify-only.
 *
 * Covers the two shapes the engine's verify rows need:
 *   exd_add_ff(acc, a, b)       a*b for f32 a, b            (attention q.k, p.v over f32 rows)
 *   exd_add_wsx(acc, w, s, x)   (w*s)*x for int w, f32 s, x (quantised weight rows: int4/int8
 *                                                           * per-row/per-group scale * f32 act)
 * Products are exact int64 (24+24 or 8+24+24 mantissa bits <= 56 bits); bins hold __int128
 * partial sums (72 bits of headroom), so any n up to 2^72 products per bin is safe.
 *
 * NaN / inf: a float dot with a NaN input is NaN, with an inf input is +-inf or NaN; the exact
 * path reproduces the same *classification* (flags), so callers see the same special values.
 *
 * Copyright (c) 2026 Anomly, Inc. Licensed under the same terms as colibri (see LICENSE).
 * Author: Ry Bruscoe. */
#ifndef COLI_EXACT_DOT_H
#define COLI_EXACT_DOT_H
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__SIZEOF_INT128__)
typedef __int128 exd_i128;
#else
#error "exact_dot.h needs a 128-bit integer type (GCC/Clang); the exact verify mode is unavailable on this compiler"
#endif

/* Product exponents: f32 mantissa m (24 bits, value m*2^e with e = exp-150), normals e in
 * [-149, 104]; products e in [-298, 208]; with an 8-bit integer weight (value w) the exponent is
 * the same range. Bin index = e + EXD_BIAS. */
#define EXD_BIAS 320
#define EXD_NBIN 640
/* wide accumulator: EXD_NBIN + 64 bits of headroom, in 64-bit limbs (two's complement) */
#define EXD_LIMBS 12   /* 768 bits */

typedef struct {
    exd_i128 bin[EXD_NBIN];
    int lo, hi;          /* used bin range (inclusive); lo > hi means empty */
    int nan, pinf, ninf; /* special-value flags, order-independent by construction */
} exd_acc;

static inline void exd_init(exd_acc *a){ a->lo = EXD_NBIN; a->hi = -1; a->nan = a->pinf = a->ninf = 0; }

/* bins are zeroed lazily as the used range [lo, hi] grows (a full memset would cost more than a
 * typical 512-element dot); returns the bin to add into */
static inline exd_i128 *exd_touch(exd_acc *a, int idx){
    if(a->lo > a->hi){ a->bin[idx] = 0; a->lo = a->hi = idx; return &a->bin[idx]; }
    if(idx < a->lo){ for(int i = idx; i < a->lo; i++) a->bin[i] = 0; a->lo = idx; }
    else if(idx > a->hi){ for(int i = a->hi + 1; i <= idx; i++) a->bin[i] = 0; a->hi = idx; }
    return &a->bin[idx];
}

/* decode f32 into (signed integer mantissa, exponent) with value = m * 2^e; returns 0 for
 * zero (m=0), 1 for finite non-zero, 2 for inf, 3 for nan. */
static inline int exd_decode(float f, int64_t *m, int *e){
    uint32_t u; memcpy(&u, &f, 4);
    int s = (int)(u >> 31), ex = (int)((u >> 23) & 0xFF); uint32_t fr = u & 0x7FFFFF;
    if(ex == 0xFF){ *m = 0; *e = 0; return fr ? 3 : 2; }
    if(ex == 0){ if(!fr){ *m = 0; *e = 0; return 0; } *m = s ? -(int64_t)fr : (int64_t)fr; *e = -149; return 1; }
    int64_t mm = (int64_t)(fr | 0x800000); *m = s ? -mm : mm; *e = ex - 150; return 1;
}

static inline void exd_special(exd_acc *a, int ca, int cb, int64_t ma, int64_t mb, float fa, float fb){
    if(ca == 3 || cb == 3){ a->nan = 1; return; }
    if(ca == 2 || cb == 2){
        /* inf * 0 = nan; inf * x has the sign of the product */
        if((ca == 2 && (cb == 0)) || (cb == 2 && (ca == 0))){ a->nan = 1; return; }
        int sa = signbit(fa) ? 1 : 0, sb = signbit(fb) ? 1 : 0; (void)ma; (void)mb;
        if(sa ^ sb) a->ninf = 1; else a->pinf = 1;
    }
}

static inline void exd_add_ff(exd_acc *a, float fa, float fb){
    int64_t ma, mb; int ea, eb;
    int ca = exd_decode(fa, &ma, &ea), cb = exd_decode(fb, &mb, &eb);
    if(ca > 1 || cb > 1){ exd_special(a, ca, cb, ma, mb, fa, fb); return; }
    if(ca == 0 || cb == 0) return;
    *exd_touch(a, ea + eb + EXD_BIAS) += (exd_i128)ma * mb;
}

/* (w * s) * x with an exact integer w (|w| < 2^8), f32 scale s, f32 activation x */
static inline void exd_add_wsx(exd_acc *a, int w, float s, float x){
    if(w == 0) return;
    int64_t ms, mx; int es, ex;
    int cs = exd_decode(s, &ms, &es), cx = exd_decode(x, &mx, &ex);
    if(cs > 1 || cx > 1){ exd_special(a, cs, cx, ms, mx, s, x); return; }
    if(cs == 0 || cx == 0) return;
    *exd_touch(a, es + ex + EXD_BIAS) += (exd_i128)w * ms * mx;   /* <= 8 + 24 + 24 bits: exact in int128 */
}

/* ---- fold bins into a 768-bit two's-complement integer scaled by 2^(lo - EXD_BIAS) ---- */
static inline void exd_wide_add_shifted(uint64_t *w, exd_i128 v, int shift){
    /* add v * 2^shift into w (12 limbs, little-endian, two's complement), 0 <= shift < 704 */
    uint64_t t[EXD_LIMBS]; const uint64_t sx = (v < 0) ? ~(uint64_t)0 : 0;
    for(int i = 0; i < EXD_LIMBS; i++) t[i] = sx;
    t[0] = (uint64_t)v; t[1] = (uint64_t)(v >> 64);
    int limbs = shift >> 6, bits = shift & 63;
    if(bits){ uint64_t prev = 0; for(int i = 0; i < EXD_LIMBS; i++){ uint64_t cur = t[i]; t[i] = (cur << bits) | prev; prev = cur >> (64 - bits); } }
    if(limbs){ for(int i = EXD_LIMBS - 1; i >= 0; i--) t[i] = (i - limbs >= 0) ? t[i - limbs] : 0; }
    unsigned __int128 carry = 0;
    for(int i = 0; i < EXD_LIMBS; i++){ unsigned __int128 sum = (unsigned __int128)w[i] + t[i] + carry; w[i] = (uint64_t)sum; carry = sum >> 64; }
}

/* correctly rounded (nearest-even) conversion of a two's-complement 768-bit integer * 2^e2 */
static inline double exd_wide_to_double(const uint64_t *w, int e2){
    uint64_t m[EXD_LIMBS]; memcpy(m, w, sizeof m);
    int neg = (m[EXD_LIMBS - 1] >> 63) & 1;
    if(neg){ unsigned __int128 c = 1; for(int i = 0; i < EXD_LIMBS; i++){ unsigned __int128 v = (unsigned __int128)(~m[i]) + c; m[i] = (uint64_t)v; c = v >> 64; } }
    int top = -1;
    for(int i = EXD_LIMBS - 1; i >= 0 && top < 0; i--) if(m[i]){ int b = 63; while(!((m[i] >> b) & 1)) b--; top = i * 64 + b; }
    if(top < 0) return 0.0;
    /* take the top 54 bits (53 + round bit) and a sticky bit for the rest */
    int hi = top, lo = top - 53;                     /* bits [lo, hi] = 54 bits */
    uint64_t bits54 = 0; int sticky = 0;
    for(int b = hi; b >= lo; b--){
        int v = (b >= 0) ? (int)((m[b >> 6] >> (b & 63)) & 1) : 0;
        bits54 = (bits54 << 1) | (uint64_t)v;
    }
    for(int i = 0; i < EXD_LIMBS && !sticky; i++){
        int base = i * 64;
        if(base + 63 < lo){ if(m[i]) sticky = 1; continue; }
        for(int b = base; b < base + 64 && b < lo; b++) if((m[i] >> (b - base)) & 1){ sticky = 1; break; }
    }
    uint64_t mant = bits54 >> 1; int round = (int)(bits54 & 1);
    if(round && (sticky || (mant & 1))) mant += 1;
    /* mant may have become 2^53: ldexp handles it exactly */
    double d = ldexp((double)mant, lo + 1 + e2);   /* mant = bits [lo+1, hi] */
    return neg ? -d : d;
}

static inline double exd_finish_double(const exd_acc *a){
    if(a->nan || (a->pinf && a->ninf)) return NAN;
    if(a->pinf) return INFINITY;
    if(a->ninf) return -INFINITY;
    if(a->lo > a->hi) return 0.0;
    uint64_t w[EXD_LIMBS]; memset(w, 0, sizeof w);
    for(int i = a->lo; i <= a->hi; i++) if(a->bin[i]) exd_wide_add_shifted(w, a->bin[i], i - a->lo);
    return exd_wide_to_double(w, a->lo - EXD_BIAS);
}

static inline float exd_finish(const exd_acc *a){ return (float)exd_finish_double(a); }

/* convenience: exact f32 dot of length n */
static inline float exd_dot_ff(const float *x, const float *y, int n){
    exd_acc a; exd_init(&a);
    for(int i = 0; i < n; i++) exd_add_ff(&a, x[i], y[i]);
    return exd_finish(&a);
}
#endif /* COLI_EXACT_DOT_H */
