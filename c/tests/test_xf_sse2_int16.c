/* test_xf_sse2_int16: the SSE2 i8 expert kernel (expert_ffn.h, XF_HAVE_SSE2_INT16)
 * against the scalar reference xf_dot_i8_ref, bit for bit.
 *
 *   test_xf_sse2_int16 [fma_cases [full]]   fma_cases defaults to 20000 per worker; "full" adds the 40-token layer
 *                                    (a minute of the scalar reference loop)
 *   test_xf_sse2_int16 dump FILE     writes the bits of xf_dot_i8 / xf_dot_i8_rows / xf_moe_run on a fixed
 *                                    stream of cases; built once against this tree and once against the
 *                                    base's expert_ffn.h (where xf_dot_i8 is the reference) the two files
 *                                    must be identical: `cmp`
 *
 * What it checks, each against the reference and never against itself:
 *   fmaf     the emulated fmaf (two doubles in an SSE register, round to odd) against libm fmaf: random bit
 *            patterns, lane*scale against an accumulator over wide exponent gaps, cancellation, zeros of both
 *            signs, results in the subnormal range, and products built so that the exact sum lies a hair
 *            from a float midpoint (the double-rounding case); non-finite operands by class
 *   decode   every byte value in each of the 32 bytes of a block, both nibbles, through unit activations
 *   integer  >= 100000 random blocks and the extremes (codes all 0 / all 15, activations +-127, -128) with
 *            unit scales: the result is the exact integer
 *   rows     20 rows (2000 with "full") for each of I = 64, 128, 512, 2048, 4096 under several scale and activation
 *            distributions; xf_dot_i8 and xf_dot_i8_rows (1..4 activation rows, strides) as the reference
 *   layer    xf_moe_run / xf_moe_add, K = 8, 2048/512, S = 1, 7, 40, holes, cuts, 1 and many threads
 * On a build without XF_HAVE_SSE2_INT16 (AVX2+FMA, or not x86-64) only what the dispatch gives is checked. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../expert_ffn.h"

static uint64_t g_rng = 88172645463325252ULL;
static uint64_t rnd64(uint64_t *s) { uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *s = x; return x * 0x2545F4914F6CDD1DULL; }
static uint32_t rnd(void) { return (uint32_t)(rnd64(&g_rng) >> 24); }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
/* a float of unbiased exponent E (clamped to the finite range), random 24-bit mantissa and sign */
static float mkf(uint64_t *s, int E) {
    if (E < -149) E = -149;
    if (E > 127) E = 127;
    float f = ldexpf(1.f + (float)(rnd64(s) >> 40) / 16777216.f, E);
    return (rnd64(s) & 1) ? -f : f;
}

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* ---- fmaf ------------------------------------------------------------------- */
#ifdef XF_HAVE_SSE2_INT16
static float finite_pattern(uint64_t *s) { for (;;) { float f = bitsf((uint32_t)(rnd64(s) >> 16)); if (isfinite(f)) return f; } }
#define FMA_GENS 10
static const char *fma_gen_name[FMA_GENS] = {
    "random finite bit patterns", "lane x scale vs accumulator, exponent gap +-60", "cancellation (acc = -product +- 2 ulp)",
    "ties by construction (odd integer x power of two)", "results in the subnormal range", "24-bit x 24-bit products",
    "zeros of both signs", "exponent gap up to 250 (overflow too)", "float midpoints + residual < double ulp (lane-sized)",
    "float midpoints + residual < double ulp (24-bit factors, subnormal accumulators)" };

static void fma_case(int g, uint64_t *s, float *a, float *b, float *c) {
    switch (g) {
    case 0: *a = finite_pattern(s); *b = finite_pattern(s); *c = finite_pattern(s); break;
    case 1: { int E = (int)(rnd64(s) % 100) - 50; *a = (float)((int)(rnd64(s) % 30721) - 15360); *b = mkf(s, E);
              *c = mkf(s, E + 13 + (int)(rnd64(s) % 121) - 60); } break;
    case 2: { int E = (int)(rnd64(s) % 60) - 30; *a = (float)((int)(rnd64(s) % 30721) - 15360); *b = mkf(s, E);
              *c = -(float)((double)*a * *b); int k = (int)(rnd64(s) % 5) - 2;
              for (; k > 0; k--) *c = nextafterf(*c, INFINITY);
              for (; k < 0; k++) *c = nextafterf(*c, -INFINITY); } break;
    case 3: { int E = (int)(rnd64(s) % 80) - 40; *c = mkf(s, E); int k = (int)(rnd64(s) % (1 << 22)) * 2 + 1;
              *a = (float)((rnd64(s) & 1) ? -k : k); *b = ldexpf(1.f, E - 24 - (int)(rnd64(s) % 3)); } break;
    case 4: *c = mkf(s, (int)(rnd64(s) % 12) - 149); *a = (float)((int)(rnd64(s) % 30721) - 15360);
            *b = mkf(s, (int)(rnd64(s) % 24) - 153); break;
    case 5: *a = mkf(s, (int)(rnd64(s) % 40) - 20); *b = mkf(s, (int)(rnd64(s) % 40) - 20); *c = mkf(s, (int)(rnd64(s) % 80) - 40); break;
    case 6: *c = mkf(s, (int)(rnd64(s) % 100) - 50); *a = (float)((int)(rnd64(s) % 2001) - 1000); *b = mkf(s, (int)(rnd64(s) % 100) - 50);
            if (rnd64(s) & 1) *c = 0.f;
            if (rnd64(s) % 7 == 0) *a = 0.f;
            if (rnd64(s) % 11 == 0) *b = -0.f;
            if (rnd64(s) % 13 == 0) *c = -0.f;
            break;
    case 7: *a = (float)((int)(rnd64(s) % 30721) - 15360); *b = mkf(s, (int)(rnd64(s) % 250) - 125); *c = mkf(s, (int)(rnd64(s) % 250) - 125); break;
    case 8: {   /* A*B = 2^37 + r, 1 <= r < 255: with the scale E-61 the product is half an ulp of a float c of exponent E plus a residual
                 * under half a double ulp, so the double sum is exactly a float midpoint with a non-zero error */
        int E = (int)(rnd64(s) % 200) - 100, A; uint64_t B; double r;
        do { A = 8192 + (int)(rnd64(s) % 7169); B = (uint64_t)ceil(137438953472.0 / A); r = (double)(B * (uint64_t)A) - 137438953472.0; }
        while (!(r >= 1 && r < 255 && B < (1ULL << 24)));
        *a = (float)((rnd64(s) & 1) ? -A : A); *b = ldexpf((float)B, E - 61); *c = mkf(s, E); } break;
    default: {  /* the same with 24-bit factors, and accumulators down into the subnormals */
        int E = (int)(rnd64(s) % 250) - 149; uint32_t A, Bq; double r;
        do { A = (1u << 23) + (uint32_t)(rnd64(s) % (1u << 23)); Bq = (uint32_t)(((1ULL << 47) + A - 1) / A); r = (double)((uint64_t)A * Bq - (1ULL << 47)); }
        while (!(r >= 1 && r < 200000 && Bq < (1u << 24)));
        int q = E - 24 - 47, u = q / 2, v = q - u;
        *a = ldexpf((float)A, u); *b = ldexpf((float)Bq, v); if (rnd64(s) & 1) *a = -*a;
        *c = E < -126 ? (float)(int)(rnd64(s) % (1 << 23)) * ldexpf(1.f, -149) : mkf(s, E); } break;
    }
}
static float fma_emulated(float a, float b, float c) {
    return (float)_mm_cvtsd_f64(xf_fmaf2_sse2(_mm_set1_pd((double)a * (double)b), _mm_set1_pd((double)c)));
}
static void test_fmaf(long n) {
    long bad[FMA_GENS] = {0}, tot[FMA_GENS] = {0};
#ifdef _OPENMP
    #pragma omp parallel num_threads(omp_get_max_threads() < 4 ? omp_get_max_threads() : 4)
#endif
    {
        int th = 0;
#ifdef _OPENMP
        th = omp_get_thread_num();
#endif
        uint64_t s = 88172645463325252ULL + 7919ULL * (uint64_t)th;
        long lbad[FMA_GENS] = {0}, ltot[FMA_GENS] = {0};
        for (long it = 0; it < n; it++) {
            int g = (int)(it % FMA_GENS); float a, b, c;
            fma_case(g, &s, &a, &b, &c);
            float ref = fmaf(a, b, c), got = fma_emulated(a, b, c);
            ltot[g]++;
            if (fbits(ref) != fbits(got) && !(isnan(ref) && isnan(got))) {
                if (++lbad[g] <= 2) printf("  fmaf mismatch (gen %d) a=%a b=%a c=%a ref=%a got=%a\n", g, a, b, c, ref, got);
            }
        }
#ifdef _OPENMP
        #pragma omp critical
#endif
        for (int g = 0; g < FMA_GENS; g++) { bad[g] += lbad[g]; tot[g] += ltot[g]; }
    }
    long B = 0, T = 0;
    for (int g = 0; g < FMA_GENS; g++) { B += bad[g]; T += tot[g]; printf("fmaf gen %d (%s): %ld cases, %ld differ\n", g, fma_gen_name[g], tot[g], bad[g]); }
    CHECK(B == 0, "fmaf emulation: %ld of %ld cases differ from libm fmaf", B, T);
    /* non-finite operands: the class (NaN, +-Inf, finite) of the reference */
    const float sp[] = {INFINITY, -INFINITY, NAN, 0.f, 1.5f, -2.f, 3e38f};
    long ncls = 0, nbad = 0;
    for (size_t i = 0; i < sizeof sp / sizeof *sp; i++) for (size_t j = 0; j < sizeof sp / sizeof *sp; j++) for (size_t k = 0; k < sizeof sp / sizeof *sp; k++) {
        float ref = fmaf(sp[i], sp[j], sp[k]), got = fma_emulated(sp[i], sp[j], sp[k]); ncls++;
        int same = (isnan(ref) && isnan(got)) || fbits(ref) == fbits(got);
        if (!same) { nbad++; printf("  non-finite class: fmaf(%g, %g, %g) = %g, emulated %g\n", sp[i], sp[j], sp[k], ref, got); }
    }
    CHECK(nbad == 0, "fmaf emulation: %ld of %ld non-finite operand triples differ in class", nbad, ncls);
    printf("fmaf non-finite operand triples: %ld, %ld differ in class\n", ncls, nbad);
}
#endif

/* ---- helpers for the dot tests ------------------------------------------------ */
static void rand_bytes(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (uint8_t)rnd(); }
static void block_sums(const int8_t *xq, int I, float *xsum) {
    for (int g = 0; g < I / XF_BLOCK; g++) { int t = 0; for (int i = 0; i < XF_BLOCK; i++) t += xq[g * XF_BLOCK + i]; xsum[g] = (float)t; }
}
static float bits_cmp_ref(const uint8_t *w, const float *sc, const int8_t *xq, const float *xsum, float sx, int I, float *ref_out) {
    float ref = xf_dot_i8_ref(w, sc, xq, xsum, sx, I), got = xf_dot_i8(w, sc, xq, xsum, sx, I);
    if (ref_out) *ref_out = ref;
    return got;
}
/* distinct values compare by bits; two NaNs of the reference's class count as equal only if their bits are */
static int differs(float a, float b) { return fbits(a) != fbits(b); }

/* ---- decode: every byte value, every position ----------------------------------- */
static void test_decode(void) {
    enum { NB = 3 };                          /* the block under test is the middle of three, so a wrong block offset shows */
    uint8_t w[NB * XF_BLOCK_BYTES]; int8_t xq[NB * XF_BLOCK]; float sc[NB], xsum[NB];
    long cases = 0, bad = 0;
    for (int v = 0; v < 256; v++)
        for (int pos = 0; pos < XF_BLOCK_BYTES; pos++)
            for (int half = 0; half < 2; half++) {
                rand_bytes(w, sizeof w);
                w[XF_BLOCK_BYTES + pos] = (uint8_t)v;
                const int elem = half ? 32 + pos : pos, code = half ? (v >> 4) : (v & 15);
                memset(xq, 0, sizeof xq);
                xq[XF_BLOCK + elem] = 1;      /* unit activation: the block contributes only its element's weight */
                for (int g = 0; g < NB; g++) sc[g] = (g == 1) ? 1.f : 0.f;   /* scales 0 on the neighbours: they must not leak */
                block_sums(xq, NB * XF_BLOCK, xsum);
                float got = xf_dot_i8(w, sc, xq, xsum, 1.f, NB * XF_BLOCK), ref = xf_dot_i8_ref(w, sc, xq, xsum, 1.f, NB * XF_BLOCK);
                cases++;
                if (got != (float)(code - 8) || differs(got, ref)) { if (++bad <= 3) printf("  decode: byte %d at %d half %d: got %g, expect %d, ref %g\n", v, pos, half, got, code - 8, ref); }
            }
    CHECK(bad == 0, "decode: %ld of %ld unit-activation cases wrong", bad, cases);
    printf("decode: %ld cases (256 byte values x 32 bytes x 2 nibbles), %ld wrong\n", cases, bad);
}

/* ---- exact integers --------------------------------------------------------------- */
static long exact_block(const uint8_t *w, const int8_t *xq) {
    long t = 0;
    for (int i = 0; i < XF_BLOCK; i++) t += (long)xf_get(w, i) * xq[i];
    return t;
}
static void test_integer(long nblocks) {
    uint8_t w[XF_BLOCK_BYTES]; int8_t xq[XF_BLOCK]; float sc = 1.f, xsum;
    long bad = 0, n = 0;
    for (long b = 0; b < nblocks; b++) {
        rand_bytes(w, sizeof w);
        const int style = (int)(rnd() % 4);
        for (int i = 0; i < XF_BLOCK; i++) {
            int v = (int)(rnd() % 255) - 127;                            /* the quantizer's range */
            if (style == 1) v = (rnd() & 1) ? 127 : -127;
            if (style == 2) v = (int)(rnd() % 256) - 128;               /* -128 explicitly */
            if (style == 3) v = (rnd() % 8 == 0) ? v : 0;               /* sparse */
            xq[i] = (int8_t)v;
        }
        block_sums(xq, XF_BLOCK, &xsum);
        float got = xf_dot_i8(w, &sc, xq, &xsum, 1.f, XF_BLOCK);
        n++;
        if (got != (float)exact_block(w, xq)) { if (++bad <= 3) printf("  integer: block %ld got %g expect %ld\n", b, got, exact_block(w, xq)); }
    }
    /* extremes: all codes 0 / all 15 / alternating, against +-127, -128, uniform */
    static const int8_t xv[] = {127, -127, -128, 1, -1, 0};
    for (int cw = 0; cw < 3; cw++) for (size_t xi = 0; xi < sizeof xv; xi++) for (int alt = 0; alt < 2; alt++) {
        for (int i = 0; i < XF_BLOCK_BYTES; i++) w[i] = cw == 0 ? 0x00 : cw == 1 ? 0xFF : 0x0F;
        for (int i = 0; i < XF_BLOCK; i++) xq[i] = (alt && (i & 1)) ? (int8_t)-xv[xi] : xv[xi];
        if (alt && xv[xi] == -128) xq[1] = 127;
        block_sums(xq, XF_BLOCK, &xsum);
        float got = xf_dot_i8(w, &sc, xq, &xsum, 1.f, XF_BLOCK);
        n++;
        if (got != (float)exact_block(w, xq)) { bad++; printf("  integer extreme: w=%d x=%d alt=%d got %g expect %ld\n", cw, xv[xi], alt, got, exact_block(w, xq)); }
    }
    CHECK(bad == 0, "integer: %ld of %ld blocks not the exact integer", bad, n);
    printf("integer: %ld blocks (random + extremes, unit scale), %ld not the exact integer\n", n, bad);
}

/* ---- rows -------------------------------------------------------------------------- */
enum { SCALE_REAL, SCALE_WIDE, SCALE_SIGNED, SCALE_SPECIAL, SCALE_SUBNORMAL, SCALE_KINDS };
static const char *scale_name[SCALE_KINDS] = {"realistic", "wide exponent", "signed [-1,1)", "zeros, +-0, tiny, huge", "subnormal"};
static float scale_value(int kind) {
    switch (kind) {
    case SCALE_REAL: return 0.002f + (rnd() & 1023) / 65536.f;
    case SCALE_WIDE: return mkf(&g_rng, (int)(rnd() % 81) - 40);
    case SCALE_SIGNED: return ((int)(rnd() & 0xFFFF) - 32768) / 32768.f;
    case SCALE_SPECIAL: { uint32_t k = rnd() % 8; return k == 0 ? 0.f : k == 1 ? -0.f : k == 2 ? 1e-30f : k == 3 ? 1e36f : k == 4 ? -1e30f : k == 5 ? 1e-38f : mkf(&g_rng, (int)(rnd() % 40) - 20); }
    default: return (rnd() & 1 ? -1.f : 1.f) * ldexpf((float)(1 + rnd() % 8000000), -149 - (int)(rnd() % 6));
    }
}
/* an activation row: quantized from floats of one of several shapes, or raw int8 bytes */
static void make_act(int style, int I, int8_t *xq, float *xsum, float *sx) {
    if (style == 4) { for (int i = 0; i < I; i++) xq[i] = (int8_t)rnd(); *sx = 0.001f * (1 + rnd() % 100); block_sums(xq, I, xsum); return; }   /* raw int8, -128 included */
    float *x = malloc(sizeof(float) * I);
    for (int i = 0; i < I; i++) {
        float v = ((int)(rnd() & 0xFFFF) - 32768) / 8192.f;
        if (style == 1) v = (rnd() % 16 == 0) ? v * 40.f : v * 0.05f;           /* outliers */
        if (style == 2) v = (rnd() % 10 == 0) ? v : 0.f;                         /* sparse */
        if (style == 3) v = (rnd() & 1) ? 1.f : -1.f;                            /* uniform magnitude: saturates the codes */
        x[i] = v;
    }
    *sx = xf_act_i8(x, I, xq, xsum);
    free(x);
}
static void test_rows(int I, int nrows, int kind) {
    const size_t rb = xf_row_bytes(I); const int ng = I / XF_BLOCK;
    enum { NA = 4 };
    int8_t *xq = malloc((size_t)NA * I); float *xsum = malloc(sizeof(float) * NA * ng), sx[NA];
    uint8_t *w = malloc(rb); float *sc = malloc(sizeof(float) * ng);
    float *y = malloc(sizeof(float) * (NA * 3 + 1));
    long bad = 0, badr = 0, n = 0, nr = 0, nonfinite = 0;
    for (int r = 0; r < nrows; r++) {
        if (r % 250 == 0) for (int a = 0; a < NA; a++) make_act((a + r / 250) % 5, I, xq + (size_t)a * I, xsum + (size_t)a * ng, &sx[a]);
        rand_bytes(w, rb);
        for (int g = 0; g < ng; g++) sc[g] = scale_value(kind);
        float ref[NA];
        for (int a = 0; a < NA; a++) {
            float got = bits_cmp_ref(w, sc, xq + (size_t)a * I, xsum + (size_t)a * ng, sx[a], I, &ref[a]);
            n++; nonfinite += !isfinite(ref[a]);
            if (differs(got, ref[a])) { if (++bad <= 3) printf("  rows I=%d scale=%s row %d act %d: got %a ref %a\n", I, scale_name[kind], r, a, got, ref[a]); }
        }
        for (int cnt = 1; cnt <= NA; cnt++) for (int st = 1; st <= 3; st += 2) {   /* strides 1 and 3 */
            for (int i = 0; i < NA * 3 + 1; i++) y[i] = -777.f;
            xf_dot_i8_rows(y, st, w, sc, xq, xsum, sx, I, cnt);
            for (int a = 0; a < cnt; a++) { nr++; if (differs(y[(size_t)a * st], ref[a])) { if (++badr <= 3) printf("  rows(n=%d stride %d) I=%d row %d act %d: got %a ref %a\n", cnt, st, I, r, a, y[(size_t)a * st], ref[a]); } }
            for (int i = 0; i < NA * 3 + 1; i++) if (i % st != 0 || i / st >= cnt) { if (y[i] != -777.f) { badr++; printf("  rows wrote outside its outputs at %d\n", i); break; } }
        }
    }
    CHECK(bad == 0, "rows I=%d scale=%s: %ld of %ld xf_dot_i8 results differ from the reference", I, scale_name[kind], bad, n);
    CHECK(badr == 0, "rows I=%d scale=%s: %ld of %ld xf_dot_i8_rows results differ from the reference", I, scale_name[kind], badr, nr);
    printf("rows I=%4d %-24s: %5d weight rows, %6ld dots + %6ld batched, %ld + %ld differ (non-finite refs %ld)\n", I, scale_name[kind], nrows, n, nr, bad, badr, nonfinite);
    free(xq); free(xsum); free(w); free(sc); free(y);
}

/* ---- layer -------------------------------------------------------------------------- */
static void make_scales(float *sc, size_t n) { for (size_t i = 0; i < n; i++) sc[i] = 0.002f + (rnd() & 1023) / 65536.f; }
/* the per-token loop on the reference dot (xf_dot_i8_ref) */
static void ref_moe(float *out, const float *x, int S, int K, int H, int F, const int *idx, const float *val, const XfExpert *const *ex) {
    float *g = malloc(sizeof(float) * F), *u = malloc(sizeof(float) * F), *h = malloc(sizeof(float) * F), *y = malloc(sizeof(float) * H);
    int8_t *xq = malloc(H), *hq = malloc(F); float *xs = malloc(sizeof(float) * (H / 64)), *hs = malloc(sizeof(float) * (F / 64));
    for (int s = 0; s < S; s++) {
        float *os = out + (size_t)s * H; memset(os, 0, sizeof(float) * H);
        const float sx = xf_act_i8(x + (size_t)s * H, H, xq, xs);
        for (int k = 0; k < K; k++) {
            int i = s * K + k; if (idx[i] < 0) continue; const XfExpert *e = ex[i];
            for (int r = 0; r < F; r++) {
                g[r] = xf_dot_i8_ref(e->g4 + (size_t)r * H / 2, e->gs + (size_t)r * (H / 64), xq, xs, sx, H);
                u[r] = xf_dot_i8_ref(e->u4 + (size_t)r * H / 2, e->us + (size_t)r * (H / 64), xq, xs, sx, H);
            }
            xf_swiglu(h, g, u, F);
            const float shx = xf_act_i8(h, F, hq, hs);
            for (int r = 0; r < H; r++) y[r] = xf_dot_i8_ref(e->d4 + (size_t)r * F / 2, e->ds + (size_t)r * (F / 64), hq, hs, shx, F);
            const float w = val[i]; for (int d = 0; d < H; d++) os[d] += w * y[d];
        }
    }
    free(g); free(u); free(h); free(y); free(xq); free(hq); free(xs); free(hs);
}
/* runs the layer; with `dump` the bits of every output go to it instead of being compared */
static void layer_case(int S, int K, int H, int F, int NE, FILE *dump) {
    XfExpert *E = malloc(sizeof(XfExpert) * NE);
    for (int e = 0; e < NE; e++) {
        uint8_t *g4 = malloc((size_t)F * H / 2), *u4 = malloc((size_t)F * H / 2), *d4 = malloc((size_t)H * F / 2);
        float *gs = malloc(sizeof(float) * F * (H / 64)), *us = malloc(sizeof(float) * F * (H / 64)), *ds = malloc(sizeof(float) * H * (F / 64));
        rand_bytes(g4, (size_t)F * H / 2); rand_bytes(u4, (size_t)F * H / 2); rand_bytes(d4, (size_t)H * F / 2);
        make_scales(gs, (size_t)F * (H / 64)); make_scales(us, (size_t)F * (H / 64)); make_scales(ds, (size_t)H * (F / 64));
        XfExpert x = { g4, u4, d4, gs, us, ds }; E[e] = x;
    }
    float *x = malloc(sizeof(float) * S * H); for (int i = 0; i < S * H; i++) x[i] = ((int)(rnd() & 0xFFFF) - 32768) / 8192.f;
    int *idx = malloc(sizeof(int) * S * K); float *val = malloc(sizeof(float) * S * K); const XfExpert **ex = malloc(sizeof(void *) * S * K);
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) {
        int i = s * K + k;
        idx[i] = (k == K - 1 && (s & 1)) ? -1 : (int)(rnd() % NE);                 /* a hole in the routing now and then */
        for (int j = 0; j < k; j++) if (idx[i] >= 0 && idx[s * K + j] == idx[i]) idx[i] = (idx[i] + 1) % NE;
        val[i] = idx[i] < 0 ? 0.f : 0.05f + (rnd() & 255) / 512.f; ex[i] = idx[i] < 0 ? NULL : &E[idx[i]];
    }
    float *o1 = malloc(sizeof(float) * S * H), *o2 = malloc(sizeof(float) * S * H), *o3 = calloc((size_t)S * H, sizeof(float));
    void *scratch = malloc(xf_moe_scratch_bytes(S, K, H, F));
    int nthr[2] = {1, 0};
#ifdef _OPENMP
    nthr[1] = omp_get_max_threads(); const int keep = nthr[1];
#else
    nthr[1] = 1;
#endif
    if (!dump) ref_moe(o1, x, S, K, H, F, idx, val, ex);
    for (int t = 0; t < 2; t++) {
#ifdef _OPENMP
        omp_set_num_threads(nthr[t]);
#endif
        xf_moe_run(o2, x, S, K, H, F, idx, val, ex, 1, scratch);
        if (dump) { fwrite(o2, sizeof(float), (size_t)S * H, dump); continue; }
        int bad = 0; for (int i = 0; i < S * H; i++) bad += fbits(o1[i]) != fbits(o2[i]);
        CHECK(bad == 0, "layer S=%d K=%d H=%d F=%d threads=%d: %d of %d outputs differ from the per-token reference loop", S, K, H, F, nthr[t], bad, S * H);
        printf("layer S=%2d K=%d H=%4d F=%3d threads=%2d: %d of %d outputs differ from the reference loop\n", S, K, H, F, nthr[t], bad, S * H);
    }
#ifdef _OPENMP
    omp_set_num_threads(keep);
#endif
    /* cut into (token, expert) pairs, and into the runs a cache of `cap` experts allows */
    for (int cap = 0; cap < 3; cap++) {
        memset(o3, 0, sizeof(float) * (size_t)S * H);
        if (cap == 0) { void *sc1 = malloc(xf_moe_scratch_bytes(1, 1, H, F));
                        for (int i = 0; i < S * K; i++) xf_moe_add(o3 + (size_t)(i / K) * H, x + (size_t)(i / K) * H, 1, 1, H, F, idx + i, val + i, ex + i, 1, sc1);
                        free(sc1); }
        else { const int c = cap == 1 ? K : (NE < 2 * K ? NE : 2 * K);
               for (int s0 = 0, per; s0 < S; s0 += per) { per = xf_rows_fit(idx, s0, S, K, c, S);
                   xf_moe_add(o3 + (size_t)s0 * H, x + (size_t)s0 * H, per, K, H, F, idx + s0 * K, val + s0 * K, ex + s0 * K, 1, scratch); } }
        if (dump) { fwrite(o3, sizeof(float), (size_t)S * H, dump); continue; }
        int bad = 0; for (int i = 0; i < S * H; i++) bad += fbits(o1[i]) != fbits(o3[i]);
        CHECK(bad == 0, "layer S=%d K=%d cut=%d: %d of %d outputs differ", S, K, cap, bad, S * H);
        printf("layer S=%2d K=%d H=%4d F=%3d cut %s: %d of %d outputs differ from the reference loop\n", S, K, H, F, cap == 0 ? "into pairs" : cap == 1 ? "for a cache of K" : "for a cache of 2K", bad, S * H);
    }
    for (int e = 0; e < NE; e++) { free((void *)E[e].g4); free((void *)E[e].u4); free((void *)E[e].d4); free((void *)E[e].gs); free((void *)E[e].us); free((void *)E[e].ds); }
    free(E); free(x); free(idx); free(val); free(ex); free(o1); free(o2); free(o3); free(scratch);
}

/* ---- the fixed stream the base and this tree are compared on ------------------------ */
static void dump_stream(FILE *f) {
    g_rng = 0x9E3779B97F4A7C15ULL;
    static const int sizes[] = {64, 128, 512, 2048, 4096};
    for (size_t si = 0; si < sizeof sizes / sizeof *sizes; si++) for (int kind = 0; kind < SCALE_KINDS; kind++) {
        const int I = sizes[si], ng = I / XF_BLOCK; const size_t rb = xf_row_bytes(I);
        int8_t *xq = malloc((size_t)4 * I); float *xsum = malloc(sizeof(float) * 4 * ng), sx[4];
        uint8_t *w = malloc(rb); float *sc = malloc(sizeof(float) * ng), y[4];
        for (int r = 0; r < 600; r++) {
            if (r % 100 == 0) for (int a = 0; a < 4; a++) make_act((a + r / 100) % 5, I, xq + (size_t)a * I, xsum + (size_t)a * ng, &sx[a]);
            rand_bytes(w, rb); for (int g = 0; g < ng; g++) sc[g] = scale_value(kind);
            for (int a = 0; a < 4; a++) y[a] = xf_dot_i8(w, sc, xq + (size_t)a * I, xsum + (size_t)a * ng, sx[a], I);
            fwrite(y, sizeof y[0], 4, f);
            xf_dot_i8_rows(y, 1, w, sc, xq, xsum, sx, I, 4); fwrite(y, sizeof y[0], 4, f);
        }
        free(xq); free(xsum); free(w); free(sc);
    }
    layer_case(1, 8, 2048, 512, 16, f); layer_case(7, 8, 2048, 512, 12, f); layer_case(40, 8, 2048, 512, 24, f);
    layer_case(5, 4, 128, 192, 6, f); layer_case(3, 2, 64, 64, 2, f);
}

#ifdef XF_HAVE_SSE2_INT16
/* Calls through volatile pointers keep the compiler from reusing a result
 * across an environment change. The scalar path is the independent oracle. */
static void test_environment(void) {
    float (*volatile reference)(const uint8_t *, const float *, const int8_t *, const float *, float, int) = xf_dot_i8_ref;
    float (*volatile dispatch)(const uint8_t *, const float *, const int8_t *, const float *, float, int) = xf_dot_i8;
    enum { I = 128, NG = I / XF_BLOCK };
    uint8_t w[I / 2]; int8_t x[I]; float sc[NG], sums[NG];
    fenv_t saved;
    fegetenv(&saved);
    const unsigned csr = _mm_getcsr();
    const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    long bad = 0, cases = 0;
    for (size_t m = 0; m < sizeof modes / sizeof *modes; m++) {
        fesetround(modes[m]);
        const unsigned rounding = _mm_getcsr();
        for (unsigned flush = 0; flush < 4; flush++) {
            const unsigned control = (rounding & ~0x8040u) | ((flush & 1) ? 0x8000u : 0) | ((flush & 2) ? 0x0040u : 0);
            _mm_setcsr(control);
            for (int r = 0; r < 100; r++) {
                rand_bytes(w, sizeof w);
                for (int i = 0; i < I; i++) x[i] = (int8_t)(rnd() % 256 - 128);
                block_sums(x, I, sums);
                for (int g = 0; g < NG; g++) sc[g] = r & 1 ? bitsf(1 + rnd() % 0x7fffff) : bitsf(0x3e000000 + rnd() % 0x2000000);
                float ref = reference(w, sc, x, sums, 1.f, I);
                float got = dispatch(w, sc, x, sums, 1.f, I);
                cases++; bad += fbits(ref) != fbits(got);
                CHECK((_mm_getcsr() & 0xffc0u) == (control & 0xffc0u), "dispatch changed MXCSR controls");
            }
        }
    }
    fesetenv(&saved);
    _mm_setcsr(csr);
    CHECK(bad == 0, "rounding/FTZ/DAZ: %ld of %ld dispatch results differ", bad, cases);
    printf("rounding/FTZ/DAZ: %ld cases, %ld differ\n", cases, bad);
}
#endif

int main(int argc, char **argv) {
#ifdef XF_HAVE_AVX2
    printf("path: AVX2 (the SSE2 kernel is not in this build; the dispatch is checked against the reference)\n");
#elif defined(XF_HAVE_SSE2_INT16)
    printf("path: SSE2 int16 (PMADDWD), exact fmaf emulation; build has SSE2"
#ifdef __SSSE3__
           " SSSE3"
#endif
#ifdef __SSE4_1__
           " SSE4.1"
#endif
#ifdef __AVX__
           " AVX"
#endif
           "\n");
#else
    printf("path: scalar reference (no SSE2 kernel in this build)\n");
#endif
    if (argc >= 3 && !strcmp(argv[1], "dump")) {
        FILE *f = fopen(argv[2], "wb"); if (!f) { perror(argv[2]); return 2; }
        dump_stream(f); fclose(f); printf("dump: %s\n", argv[2]); return 0;
    }
    const long fma_cases = argc >= 2 ? atol(argv[1]) : 20000L;
    const int full = argc >= 3 && !strcmp(argv[2], "full");
#ifdef XF_HAVE_SSE2_INT16
    test_fmaf(fma_cases);
    test_environment();
#else
    (void)fma_cases; printf("fmaf: no emulation in this build\n");
#endif
    test_decode();
    test_integer(200000);
    static const int sizes[] = {64, 128, 512, 2048, 4096};
    for (size_t si = 0; si < sizeof sizes / sizeof *sizes; si++) for (int kind = 0; kind < SCALE_KINDS; kind++) test_rows(sizes[si], full ? 2000 : 20, kind);
    layer_case(1, 8, 2048, 512, 16, NULL);
    layer_case(7, 8, 2048, 512, 12, NULL);
    if (full) layer_case(40, 8, 2048, 512, 24, NULL);
    layer_case(5, 4, 128, 192, 6, NULL);
    layer_case(3, 2, 64, 64, 2, NULL);
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    printf("all passed\n"); return 0;
}
