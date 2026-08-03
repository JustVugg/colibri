/* fmt=8 (native FP8-e4m3 passthrough) CPU kernel tests: LUT exactness (all 256
 * byte codes, incl. +-0, subnormals, NaN), matmul_fp8 vs a double-precision
 * block-scale reference on block-edge (non-128-multiple O/I) shapes, a
 * non-square-block-grid case with a distinct scale per block (catches a
 * transposed/swapped block-index stride the same way test_backend_metal.mm's
 * "stride-audit" case does for the GPU kernel), and the NaN-propagation policy
 * (decode -> real IEEE NaN -> flows through the dot product; see quant.h's
 * e4m3_decode/E4M3_LUT comment for the documented policy + rationale).
 * Pure quant.h test: no Model/QT/disk dependency (see quant.h's own "pure
 * compute" header comment) -- the loader-seam (qt_resolve_fmt disambiguation,
 * qt_from_disk) is covered separately in test_fp8_load.c.
 *
 * fmt=8, PUBLIC ordinal: this format was minted fmt=6 during original
 * development of this branch, before dev's own #465 (E8/IQ3) claimed that
 * ordinal upstream; re-tagged fmt=100 (PRIVATE ORDINAL BLOCK, see colibri.c's
 * QT struct comment) from that point forward -- there was never a build in
 * this branch's history where this format was reachable as fmt=6 -- graduated
 * to fmt=7 when the maintainer assigned that ordinal on #524, and renumbered
 * to fmt=8 after #705 merged claiming 7 for MXFP4 (see colibri.c's QT
 * comment). */
#include "../quant.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                                        \
            fails++;                                                                                                   \
        }                                                                                                              \
    } while (0)

static uint64_t rng = 0xC0FFEE1234567ull;
static float rndf(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return ((int64_t)(rng & 0xFFFFF) - 0x80000) / (float)0x80000;
}
/* random byte EXCLUDING the two NaN codes (0x7F/0xFF) -- used by the accuracy/
 * magnitude tests below so a stray NaN term doesn't turn `rel>tol` trivially
 * false (NaN compares false against everything) and mask a real kernel bug.
 * NaN handling itself is checked separately, deliberately, below. */
static uint8_t rndbyte_nonan(void) {
    for (;;) {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        uint8_t b = (uint8_t)(rng & 0xFF);
        if (b != 0x7F && b != 0xFF) return b;
    }
}

/* independent reference decode (bit manipulation, not the LUT under test) --
 * cross-checked offline against torch.float8_e4m3fn byte-for-byte (256/256
 * match, RAN evidence, see the build report). Kept here so a transcription
 * bug in E4M3_LUT's 256 hex-float literals is still caught by this test even
 * though both were derived from the same formula. */
static float ref_e4m3(uint8_t b) {
    unsigned sign = (b >> 7) & 1, exp = (b >> 3) & 0xF, mant = b & 0x7;
    if (exp == 0xF && mant == 0x7) return NAN;
    float val;
    if (exp == 0) val = (float)mant * (1.0f / 8.0f) * powf(2.0f, 1.0f - 7.0f);
    else val = (1.0f + (float)mant * (1.0f / 8.0f)) * powf(2.0f, (float)exp - 7.0f);
    return sign ? -val : val;
}

static void test_lut(void) {
    int nan_codes = 0;
    for (int b = 0; b < 256; b++) {
        float got = e4m3_decode((uint8_t)b);
        float ref = ref_e4m3((uint8_t)b);
        if (isnan(ref)) {
            CHECK(isnan(got));
            nan_codes++;
            continue;
        }
        CHECK(got == ref);
        if (got == 0.f) CHECK(signbit(got) == signbit(ref)); /* +-0 must keep its sign bit */
    }
    CHECK(nan_codes == 2); /* exactly 0x7F and 0xFF */
    CHECK(isnan(e4m3_decode(0x7F)) && isnan(e4m3_decode(0xFF)));
    CHECK(e4m3_decode(0x7E) == 448.0f && e4m3_decode(0xFE) == -448.0f); /* max finite, OCP E4M3FN */
    CHECK(e4m3_decode(0x01) == 0x1.0p-9f);                              /* min positive subnormal = 2^-9 */
    CHECK(e4m3_decode(0x00) == 0.0f && !signbit(e4m3_decode(0x00)));
    CHECK(e4m3_decode(0x80) == 0.0f && signbit(e4m3_decode(0x80))); /* -0.0 */
}

/* double-precision block-scale reference: mirrors matmul_fp8's formula exactly
 * (w[o,i] = e4m3_decode(byte) * scale[o/128,i/128]) but every intermediate is
 * double so it can serve as ground truth for matmul_fp8's own float/double mix. */
static void ref_matmul_fp8(double *y, double *mag, const float *x, const uint8_t *q8, const float *bscale, int S, int I,
                           int O) {
    int64_t nblkI = fp8_nblk(I);
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q8 + (int64_t)o * I;
        int64_t blkO = o / FP8_BLOCK;
        const float *scl = bscale + blkO * nblkI;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            double a = 0, m = 0;
            for (int i = 0; i < I; i++) {
                int64_t bi = i / FP8_BLOCK;
                double term = (double)e4m3_decode(w[i]) * (double)xs[i] * (double)scl[bi];
                a += term;
                m += fabs(term);
            }
            y[(int64_t)s * O + o] = a;
            mag[(int64_t)s * O + o] = m;
        }
    }
}

/* magnitude-relative Sigma|terms| oracle (the convention test_backend_metal.mm's
 * cpu_ref_grouped/run_grouped established for fmt=4 and reuses for fmt=8 on the
 * GPU side): compare error against the sum of |terms|, not the (possibly
 * near-zero, cancellation-prone) result itself. */
static int check_close(const float *got, const double *ref, const double *mag, int n, double tol, const char *tag) {
    double worst = 0;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)got[i] - ref[i]);
        double rel = mag[i] > 1e-30 ? d / mag[i] : d;
        if (rel > worst) worst = rel;
        if (rel > tol) bad++;
    }
    if (bad) printf("  %-40s worst_rel=%.3e FAIL (%d/%d over tol)\n", tag, worst, bad, n);
    return bad == 0;
}

static void run_block_case(int O, int I, const char *tag) {
    int64_t nblkO = fp8_nblk(O), nblkI = fp8_nblk(I), nblk = nblkO * nblkI;
    uint8_t *q8 = malloc((size_t)O * I);
    float *bscale = malloc((size_t)nblk * sizeof(float));
    float *x = malloc((size_t)3 * I * sizeof(float));
    float *y = malloc((size_t)3 * O * sizeof(float));
    double *yr = malloc((size_t)3 * O * sizeof(double));
    double *mag = malloc((size_t)3 * O * sizeof(double));
    for (int64_t i = 0; i < (int64_t)O * I; i++) q8[i] = rndbyte_nonan();
    /* distinct, easily-distinguishable scale per block (b+1)*0.01 with alternating
     * sign and a random jitter -- a swapped blkO/blkI stride or a wrong nblkI in the
     * indexing picks up the WRONG block's scale, which this makes numerically loud
     * rather than "close enough to pass by luck" (same idea as the GPU stride-audit
     * case, applied here to the CPU kernel). */
    for (int64_t b = 0; b < nblk; b++) bscale[b] = (float)(b + 1) * 0.01f * ((b & 1) ? -1.f : 1.f) + rndf() * 0.001f;
    for (int i = 0; i < 3 * I; i++) x[i] = rndf();
    ref_matmul_fp8(yr, mag, x, q8, bscale, 3, I, O);
    matmul_fp8(y, x, q8, bscale, 3, I, O);
    CHECK(check_close(y, yr, mag, 3 * O, 1e-5, tag));
    free(q8);
    free(bscale);
    free(x);
    free(y);
    free(yr);
    free(mag);
}

static void test_nan_propagation(void) {
    /* one NaN byte in an otherwise-normal tensor -> that OUTPUT ROW is NaN (the
     * documented policy: propagate, rely on the existing argmax_v/dist_build net
     * downstream -- see quant.h's e4m3_decode comment). Rows that never touch the
     * NaN byte stay finite: NaN contamination is per-dot-product, not global.
     * O=3<=128 -> nblkO=1, so all three rows share the SAME nblkI=2-entry scale
     * block-row (bscale[0..1]); q8 is the only thing that differs per row. */
    enum { O = 3, I = 200 };
    static uint8_t q8[O * I];
    static float bscale[2]; /* nblkO(1)*nblkI(2) */
    for (int i = 0; i < O * I; i++) q8[i] = rndbyte_nonan();
    q8[1 * I + 37] = 0x7F; /* row 1 only, one NaN byte */
    bscale[0] = 0.1f;
    bscale[1] = 0.11f;
    static float x[I], y[O];
    for (int i = 0; i < I; i++) x[i] = rndf();
    matmul_fp8(y, x, q8, bscale, 1, I, O);
    CHECK(!isnan(y[0]) && !isnan(y[2])); /* untouched rows stay finite */
    CHECK(isnan(y[1]));                  /* contaminated row propagates NaN */
}

int main(void) {
    test_lut();
    run_block_case(2048, 6144, "block gate/up-shaped O=2048 I=6144 (spec example)");
    run_block_case(6144, 2048, "block down-shaped O=6144 I=2048");
    run_block_case(130, 200, "block edges: O,I both non-mult-128");
    run_block_case(129, 128, "block edges: O just over 128, I exact");
    run_block_case(128, 129, "block edges: O exact, I just over 128");
    run_block_case(1, 1, "degenerate 1x1 (single sub-block)");
    run_block_case(384, 6144, "non-square block grid nblkO=3 nblkI=48 (stride audit)");
    test_nan_propagation();
    if (fails) {
        printf("fp8 passthrough CPU tests: %d FAILED\n", fails);
        return 1;
    }
    printf("fp8 passthrough CPU tests: ok\n");
    return 0;
}
