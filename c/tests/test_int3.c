/* int3-g64 (fmt=5) tests: pack layout, dequant round-trip vs plain-C reference,
 * matmul_i3 (NEON/AVX-512 + scalar tail) vs reference dequant-matmul, per-row helpers,
 * the .qs-size format tag, and the quality claim in miniature (per-group int3
 * beats per-row int4 on rows with outliers — the #132 result this format ships). */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <string.h>
#include <math.h>

static int fails = 0;
static int cur_I = 0, cur_S = 0; /* shape under test, for failure triage */
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            printf("FAIL %s:%d: %s (I=%d S=%d O=%d)\n", __FILE__, __LINE__, #c, cur_I, cur_S, (int)O);                 \
            fails++;                                                                                                   \
        }                                                                                                              \
    } while (0)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float rndf(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return ((int64_t)(rng & 0xFFFFF) - 0x80000) / (float)0x80000;
}

/* reference: quantize like pack_int3_g64 but keep dequantized f32 (mirrors
 * quant_ablation._quant_last_dim(bits=3, group=64)) */
static void ref_i3_dequant(const float *w, float *dq, int O, int I) {
    int64_t ng = i3_groups(I);
    for (int o = 0; o < O; o++)
        for (int64_t g = 0; g < ng; g++) {
            int base = (int)(g * I3_GROUP), n = I - base < I3_GROUP ? I - base : I3_GROUP;
            float amax = 0;
            for (int k = 0; k < n; k++) {
                float a = fabsf(w[(int64_t)o * I + base + k]);
                if (a > amax) amax = a;
            }
            float s = amax / 3.f;
            if (s < 1e-8f) s = 1e-8f;
            for (int k = 0; k < n; k++) {
                int v = (int)lrintf(w[(int64_t)o * I + base + k] / s);
                if (v > 3) v = 3;
                if (v < -4) v = -4;
                dq[(int64_t)o * I + base + k] = (float)v * s;
            }
        }
}
static void unpack_i3(const uint8_t *q3, const float *s, float *dq, int O, int I) {
    int64_t ng = i3_groups(I), rb = i3_rowbytes(I);
    for (int o = 0; o < O; o++)
        for (int64_t g = 0; g < ng; g++) {
            const uint8_t *lo = q3 + (int64_t)o * rb + g * I3_GBYTES, *hi = lo + 16;
            int base = (int)(g * I3_GROUP), n = I - base < I3_GROUP ? I - base : I3_GROUP;
            for (int k = 0; k < n; k++) {
                unsigned u = ((lo[k >> 2] >> ((k & 3) * 2)) & 3) | (((hi[k >> 3] >> (k & 7)) & 1) << 2);
                dq[(int64_t)o * I + base + k] = (float)((int)u - 4) * s[(int64_t)o * ng + g];
            }
        }
}

int main(void) {
    const int Is[] = {64, 128, 192, 100,
                      65, 33,  7,   7168}; /* incl. short tail groups, I<64 (scalar-only group), one real GLM dim */
    enum { O = 7, MAXI = 7168 };
    static float w[(int64_t)O * MAXI], dq_ref[(int64_t)O * MAXI], dq_pk[(int64_t)O * MAXI];
    static float x[4 * MAXI], y_ref[4 * O], y_ker[4 * O];
    static uint8_t q3[(int64_t)O * (MAXI / 64 + 1) * 24];
    static float sc[(int64_t)O * (MAXI / 64 + 1)];

    for (unsigned c = 0; c < sizeof Is / sizeof *Is; c++) {
        int I = Is[c];
        cur_I = I;
        cur_S = 0;
        for (int64_t i = 0; i < (int64_t)O * I; i++) w[i] = rndf() * 0.05f;
        w[3] = 1.7f;
        w[(int64_t)2 * I + 5] = -2.2f; /* outliers */

        /* 1. pack -> unpack == reference quantize-dequantize, bit for bit */
        pack_int3_g64(w, q3, sc, O, I);
        ref_i3_dequant(w, dq_ref, O, I);
        unpack_i3(q3, sc, dq_pk, O, I);
        int bad = 0;
        for (int64_t i = 0; i < (int64_t)O * I; i++)
            if (dq_pk[i] != dq_ref[i]) bad++;
        CHECK(bad == 0);

        /* 2. matmul_i3 == matmul over the dequantized reference (fp tolerance:
         *    NEON/AVX-512 fma order differs from the scalar reference loop) */
        for (int S = 1; S <= 4; S += 3) {
            cur_S = S;
            for (int64_t i = 0; i < (int64_t)S * I; i++) x[i] = rndf();
            matmul_i3(y_ker, x, q3, sc, S, I, O);
            for (int s = 0; s < S; s++)
                for (int o = 0; o < O; o++) {
                    double a = 0;
                    for (int i = 0; i < I; i++) a += (double)dq_ref[(int64_t)o * I + i] * x[(int64_t)s * I + i];
                    y_ref[s * O + o] = (float)a;
                }
            for (int i = 0; i < S * O; i++) {
                float d = fabsf(y_ker[i] - y_ref[i]), m = fabsf(y_ref[i]) > 1 ? fabsf(y_ref[i]) : 1;
                if (d / m > 2e-4f) {
                    CHECK(!"matmul_i3 mismatch");
                    break;
                }
            }
        }

        /* 3. QT plumbing: qt_alloc(bits=3) -> qt_fill -> matmul_qt & qt_bytes & helpers */
        cur_S = 1;
        QT t;
        qt_alloc(&t, O, I, 3);
        CHECK(t.fmt == 5);
        qt_fill(&t, w, 3);
        CHECK(qt_bytes(&t) == (int64_t)O * i3_rowbytes(I) + (int64_t)O * i3_groups(I) * 4);
        matmul_qt(y_ker, x, &t, 1);
        for (int o = 0; o < O; o++) {
            double a = 0;
            for (int i = 0; i < I; i++) a += (double)dq_ref[(int64_t)o * I + i] * x[i];
            float d = fabsf(y_ker[o] - (float)a), m = fabsf((float)a) > 1 ? fabsf((float)a) : 1;
            CHECK(d / m <= 2e-4f);
        }
        float acc[MAXI];
        memset(acc, 0, I * sizeof(float));
        qt_addrow(&t, 2, 0.5f, acc);
        for (int i = 0; i < I; i++) CHECK(fabsf(acc[i] - 0.5f * dq_ref[(int64_t)2 * I + i]) <= 1e-6f);
        float yr[3];
        qt_matvec_rows(&t, 1, 3, x, yr);
        for (int j = 0; j < 3; j++) {
            double a = 0;
            for (int i = 0; i < I; i++) a += (double)dq_ref[(int64_t)(1 + j) * I + i] * x[i];
            float d = fabsf(yr[j] - (float)a), m = fabsf((float)a) > 1 ? fabsf((float)a) : 1;
            CHECK(d / m <= 2e-4f);
        }

        /* 4. format resolution through the #413 gate: fmt=5 is tagged by its distinct
         *    WEIGHT byte count. int3-g64 and grouped-int4-at-gs=64 carry the SAME scale
         *    cardinality O*ceil(I/64), so the pair (weight bytes, scale bytes) must
         *    disambiguate: same scales, int4 weights -> fmt=4/gs=64; int3 weights -> fmt=5.
         *    Only well-posed for I > 256: below that, O row scales legitimately match a
         *    1-group grouped layout too (detect_group_size probes gs up to 256), so
         *    per-row vs grouped is not distinguishable from byte counts alone.
         *    stamped_name=NULL throughout below: this suite predates and is orthogonal
         *    to the fp8 metadata-stamp feature -- none of these calls touch the
         *    fmt=1-vs-fmt=8 or fmt=6-vs-fmt=8 collisions the stamp parameter exists
         *    to resolve, so a NULL stamp cannot change any of this suite's outcomes. */
        if (I > 256) {
            int gs = -1;
            int64_t ns_g64 = (int64_t)O * i3_groups(I) * 4, ns_row = (int64_t)O * 4;
            CHECK(qt_resolve_fmt("t.i3", O, I, (int64_t)O * i3_rowbytes(I), ns_g64, &gs, NULL) == 5);
            CHECK(gs == 0);
            CHECK(qt_resolve_fmt("t.i8", O, I, (int64_t)O * I, ns_row, &gs, NULL) == 1);
            CHECK(qt_resolve_fmt("t.i4", O, I, (int64_t)O * ((I + 1) / 2), ns_row, &gs, NULL) == 2);
            CHECK(qt_resolve_fmt("t.i4g", O, I, (int64_t)O * ((I + 1) / 2), ns_g64, &gs, NULL) == 4);
            CHECK(gs == 64);
            CHECK(qt_resolve_fmt("t.i2", O, I, (int64_t)O * ((I + 3) / 4), ns_row, &gs, NULL) == 3);
        }
        free(t.q4);
        free(t.s);
    }

    /* 5. quality in miniature: on rows with outliers, per-group int3 must beat
     *    per-row int4 on reconstruction RMS (the #132 finding this format ships). */
    {
        int I = 1024;
        cur_I = I;
        cur_S = 1;
        for (int64_t i = 0; i < (int64_t)O * I; i++) w[i] = rndf() * 0.02f;
        for (int o = 0; o < O; o++) w[(int64_t)o * I + (o * 37) % I] = 1.5f; /* one outlier per row */
        ref_i3_dequant(w, dq_ref, O, I);
        QT t4;
        qt_alloc(&t4, O, I, 4);
        qt_fill(&t4, w, 4);
        double e3 = 0, e4 = 0;
        for (int o = 0; o < O; o++)
            for (int i = 0; i < I; i++) {
                float w4;
                {
                    const uint8_t *q = t4.q4 + (int64_t)o * ((I + 1) / 2);
                    uint8_t b = q[i >> 1];
                    int v = (i & 1) ? ((int)(b >> 4) - 8) : ((int)(b & 0xF) - 8);
                    w4 = (float)v * t4.s[o];
                }
                double d3 = w[(int64_t)o * I + i] - dq_ref[(int64_t)o * I + i], d4 = w[(int64_t)o * I + i] - w4;
                e3 += d3 * d3;
                e4 += d4 * d4;
            }
        CHECK(e3 < e4);
        printf("  outlier-rows RMS: int3-g64 %.3e < int4-row %.3e (ratio %.2f)\n", sqrt(e3 / ((double)O * I)),
               sqrt(e4 / ((double)O * I)), sqrt(e4 / e3));
        free(t4.q4);
        free(t4.s);
    }

    if (fails) {
        printf("int3-g64 tests: %d FAILED\n", fails);
        return 1;
    }
    printf("int3-g64 tests: ok\n");
    return 0;
}
