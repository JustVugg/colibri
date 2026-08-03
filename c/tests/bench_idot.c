/* Microbenchmark: old (single-accumulator) vs new (independent-accumulator) AVX-VNNI
 * int8/int4 dot kernels (quant.h). NOT a unit test -- test_idot.c proves correctness.
 * This measures the headline claim: breaking the serial vpdpbusd->acc chain lifts
 * per-core kernel throughput, the same win the NEON path already took ("26->63 GB/s, 2.4x").
 *
 * It re-implements the OLD single-acc AVX-VNNI kernels inline and calls the REAL (new)
 * ones via the include-colibri.c pattern -- one process, identical frozen inputs, warm
 * caches. Reports median ns/call + GB/s-of-weights + new/old ratio. Measures the kernel's
 * compute ceiling (warm caches), NOT end-to-end tok/s.
 *
 * Run:  make tests/bench_idot ARCH=native && ./tests/bench_idot   (not in TEST_BINS -- not a gate)
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <stdint.h>
#include <string.h>

static uint32_t rs = 0x2545F491u;
static uint32_t xr(void) {
    rs ^= rs << 13;
    rs ^= rs >> 17;
    rs ^= rs << 5;
    return rs;
}

/* ---- OLD kernels: verbatim copies of the pre-change __AVXVNNI__ branches (single acc) ---- */
#if defined(__AVXVNNI__) && defined(__AVX2__)
static int32_t dot_i8i8_old(const int8_t *w, const int8_t *x, int I) {
    int32_t sum = 0;
    int i = 0;
    __m128i acc = _mm_setzero_si128();
    for (; i + 16 <= I; i += 16) {
        __m128i wv = _mm_loadu_si128((const __m128i *)(w + i));
        __m128i xv = _mm_loadu_si128((const __m128i *)(x + i));
        __m128i xs = _mm_sign_epi8(xv, wv);
        acc = _mm_dpbusd_epi32(acc, _mm_abs_epi8(wv), xs);
    }
    sum = hsum128_i32(acc);
    for (; i < I; i++) sum += (int32_t)w[i] * x[i];
    return sum;
}
static int32_t dot_i4i8_old(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0;
    int i = 0;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    __m128i acc = _mm_setzero_si128();
    for (; i + 32 <= I; i += 32) {
        __m128i by = _mm_loadu_si128((const __m128i *)(w4 + (i >> 1)));
        __m128i lo = _mm_and_si128(by, m4), hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
        __m128i n0 = _mm_unpacklo_epi8(lo, hi), n1 = _mm_unpackhi_epi8(lo, hi);
        __m128i w0 = _mm_sub_epi8(n0, b8), w1 = _mm_sub_epi8(n1, b8);
        __m128i x0 = _mm_loadu_si128((const __m128i *)(x + i));
        __m128i x1 = _mm_loadu_si128((const __m128i *)(x + i + 16));
        acc = _mm_dpbusd_epi32(acc, _mm_abs_epi8(w0), _mm_sign_epi8(x0, w0));
        acc = _mm_dpbusd_epi32(acc, _mm_abs_epi8(w1), _mm_sign_epi8(x1, w1));
    }
    sum = hsum128_i32(acc);
    for (; i < I; i++) {
        uint8_t b = w4[i >> 1];
        int v = (i & 1) ? ((int)(b >> 4) - 8) : ((int)(b & 0xF) - 8);
        sum += v * x[i];
    }
    return sum;
}
#else
#error "bench_idot requires an AVX-VNNI build: make tests/bench_idot ARCH=native on an AVX-VNNI CPU"
#endif

#define I_DIM 6144
#define N_REPEAT 20000
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(void) {
    static int8_t w8[I_DIM], x8[I_DIM];
    static uint8_t w4[I_DIM / 2];
    for (int i = 0; i < I_DIM; i++) {
        w8[i] = (int8_t)(xr() & 0xFF);
        x8[i] = (int8_t)((int)(xr() % 255) - 127);
    }
    for (int i = 0; i < I_DIM / 2; i++) w4[i] = (uint8_t)(xr() & 0xFF);

    /* correctness sanity: new must equal old (bit-exact) */
    if (dot_i8i8(w8, x8, I_DIM) != dot_i8i8_old(w8, x8, I_DIM)) {
        fprintf(stderr, "MISMATCH i8i8\n");
        return 1;
    }
    if (dot_i4i8(w4, x8, I_DIM) != dot_i4i8_old(w4, x8, I_DIM)) {
        fprintf(stderr, "MISMATCH i4i8\n");
        return 1;
    }

    static double t[N_REPEAT];
    volatile int32_t sink = 0;
    const char *names[4] = {"i8i8 old", "i8i8 new", "i4i8 old", "i4i8 new"};
    double gbs[4];
    for (int k = 0; k < 4; k++) {
        for (int wi = 0; wi < 200; wi++)
            sink += (k == 0)   ? dot_i8i8_old(w8, x8, I_DIM)
                    : (k == 1) ? dot_i8i8(w8, x8, I_DIM)
                    : (k == 2) ? dot_i4i8_old(w4, x8, I_DIM)
                               : dot_i4i8(w4, x8, I_DIM); /* warmup */
        for (int r = 0; r < N_REPEAT; r++) {
            double t0 = now_s();
            sink += (k == 0)   ? dot_i8i8_old(w8, x8, I_DIM)
                    : (k == 1) ? dot_i8i8(w8, x8, I_DIM)
                    : (k == 2) ? dot_i4i8_old(w4, x8, I_DIM)
                               : dot_i4i8(w4, x8, I_DIM);
            t[r] = (now_s() - t0) * 1e9;
        }
        qsort(t, N_REPEAT, sizeof(double), cmp_d);
        double med = t[N_REPEAT / 2];
        double bytes = (k < 2) ? I_DIM : (double)I_DIM / 2; /* weight bytes touched */
        gbs[k] = bytes / med;
        printf("%-9s  %8.1f ns/call   %6.2f GB/s\n", names[k], med, gbs[k]);
    }
    printf("ratio i8i8 new/old: %.2fx   |   ratio i4i8 new/old: %.2fx\n", gbs[1] / gbs[0], gbs[3] / gbs[2]);
    (void)sink;
    return 0;
}
