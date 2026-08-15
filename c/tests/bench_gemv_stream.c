/* Microbenchmark: decode-regime GEMV bandwidth vs the machine's read ceiling.
 *
 * bench_idot measures the kernels' warm-cache compute ceiling. Decode lives in the
 * opposite regime: expert weights do NOT fit in cache, every token streams them from
 * RAM, and arithmetic intensity is ~1 MAC/byte. This measures the REAL quant.h row
 * loops (matmul_q int8, matmul_i4 int4, their own internal OMP) over a working set
 * far larger than the LLC, cycling experts so no call sees a warm weight — then a
 * read baseline in the same process, same pass, so the ratio cancels host noise.
 *
 * The int8-vs-int4 GB/s comparison is the sharp edge: equal GB/s means the decode
 * GEMV is bandwidth-bound (int4 = free 2x tokens/byte); int4 well below int8 means
 * the unpack is compute-bound on this ISA and kernel work has headroom.
 *
 * Arms: f32-act int8/int4 (IDOT=0 paths), idot int8/int4 (production decode), a FROZEN
 * baseline copy of today's plain-AVX2 dot_i4i8 (any future kernel change A/Bs against
 * it), a DEINTERLEAVED-x candidate kernel (no unpacks; bit-exact, gated inline), a warm
 * single-expert arm (ALU ceiling vs streaming), and a parallel-read ceiling.
 *
 * Run:  make tests/bench_gemv_stream ARCH=native && OMP_NUM_THREADS=N ./tests/bench_gemv_stream
 * (not in TEST_BINS -- a measurement, not a gate. On shared vCPUs the read ceiling
 * scales with N and never shows the physical RAM roof, so absolute GB/s there is
 * indicative only; the within-pass ratios are the signal. Physical AVX2-only silicon
 * is what decides whether the deinterleaved candidate is worth landing.) */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <stdint.h>
#include <string.h>

/* FROZEN baseline: verbatim copy of dev's plain-AVX2 dot_i4i8 as of 2026-08-09,
 * bench_idot's old-vs-new pattern. While quant.h is unchanged the ratio prints 1.00x;
 * any future kernel change A/Bs against this copy in one process over identical frozen
 * inputs, so host noise cancels in the ratio. */
#ifdef __AVX2__
static inline int32_t dot_i4i8_old(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0; int i = 0;
    const __m128i m4 = _mm_set1_epi8(0x0F); const __m256i b8 = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (; i + 32 <= I; i += 32) {
        __m128i by = _mm_loadu_si128((const __m128i*)(w4 + (i >> 1)));
        __m128i lo = _mm_and_si128(by, m4), hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
        __m128i n0 = _mm_unpacklo_epi8(lo, hi), n1 = _mm_unpackhi_epi8(lo, hi);
        __m256i wv = _mm256_sub_epi8(_mm256_set_m128i(n1, n0), b8);
        __m256i xv = _mm256_loadu_si256((const __m256i*)(x + i));
        __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    sum = hsum256_i32(acc);
    for (; i < I; i++) { int8_t w = (int8_t)((i & 1) ? (w4[i >> 1] >> 4) : (w4[i >> 1] & 0x0F)) - 8; sum += w * x[i]; }
    return sum;
}
static void matmul_i4_idot_old(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                               const float *scale, int S, int I, int O) {
    int rb = (I + 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) { const uint8_t *w = q4 + (int64_t)o * rb; float sc = scale[o];
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = (float)dot_i4i8_old(w, xq + (int64_t)s * I, I) * sc * sx[s]; }
}
#endif

/* CANDIDATE dot_i4i8_de: x pre-deinterleaved (even/odd elements) so lo/hi nibbles are
 * used straight after and/srli — no unpacks, no lane insert. The deinterleave runs once
 * per activation vector and amortises over every row of the projection. Products are
 * identical and integer adds are associative, so the result must stay bit-identical. */
#ifdef __AVX2__
static inline int32_t dot_i4i8_de(const uint8_t *w4, const int8_t *xe, const int8_t *xo, int I) {
    int32_t sum = 0; int i = 0;
    const __m256i m4 = _mm256_set1_epi8(0x0F); const __m256i b8 = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    for (; i + 64 <= I; i += 64) {
        __m256i by = _mm256_loadu_si256((const __m256i*)(w4 + (i >> 1)));   /* 32 B = 64 elems */
        __m256i we = _mm256_sub_epi8(_mm256_and_si256(by, m4), b8);
        __m256i wo = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(by, 4), m4), b8);
        __m256i xev = _mm256_loadu_si256((const __m256i*)(xe + (i >> 1)));
        __m256i xov = _mm256_loadu_si256((const __m256i*)(xo + (i >> 1)));
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(we, we), _mm256_sign_epi8(xev, we)), ones));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(wo, wo), _mm256_sign_epi8(xov, wo)), ones));
    }
    sum = hsum256_i32(_mm256_add_epi32(a0, a1));
    for (; i < I; i++) { int8_t w = (int8_t)((i & 1) ? (w4[i >> 1] >> 4) : (w4[i >> 1] & 0x0F)) - 8;
                         sum += w * ((i & 1) ? xo[i >> 1] : xe[i >> 1]); }
    return sum;
}
static void i4_deinterleave(const int8_t *x, int I, int8_t *xe, int8_t *xo) {
    int j = 0;
    for (; j + 2 <= I; j += 2) { xe[j >> 1] = x[j]; xo[j >> 1] = x[j + 1]; }
    if (I & 1) xe[I >> 1] = x[I - 1];
}
static void matmul_i4_idot_de(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                              const float *scale, int I, int O, int8_t *xe, int8_t *xo) {
    int rb = (I + 1) / 2;                                        /* S=1 decode shape */
    i4_deinterleave(xq, I, xe, xo);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        y[o] = (float)dot_i4i8_de(q4 + (int64_t)o * rb, xe, xo, I) * scale[o] * sx[0];
}
#endif

static uint32_t rs = 0x2545F491u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static double tnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
static double median5(double *v) { qsort(v, 5, sizeof(double), cmpd); return v[2]; }

/* GLM-like expert: gate/up [INTER,HID], down [HID,INTER]; top-K experts per token */
enum { HID = 5120, INTER = 1536, NEXP = 32, TOPK = 8, TOKENS = 24, REPS = 5 };

int main(void) {
    const int64_t w8_per = (int64_t)INTER * HID * 2 + (int64_t)HID * INTER;   /* int8 bytes/expert */
    const int64_t w4_per = w8_per / 2;
    printf("bench_gemv_stream: %d experts x %.1f MB int8 (working set %.2f GB), top-%d, S=1\n",
           NEXP, w8_per / 1e6, NEXP * w8_per / 1e9, TOPK);

    /* one contiguous slab per format, cut into per-expert views */
    int8_t  *w8 = malloc((size_t)NEXP * w8_per);
    uint8_t *w4 = malloc((size_t)NEXP * w4_per);
    float   *sc = malloc((size_t)NEXP * (INTER * 2 + HID) * sizeof(float));
    float   *x  = malloc((size_t)HID * sizeof(float));
    float   *g  = malloc((size_t)INTER * sizeof(float));
    float   *u  = malloc((size_t)INTER * sizeof(float));
    float   *d  = malloc((size_t)HID * sizeof(float));
    int8_t  *xq = malloc((size_t)HID);                            /* idot activations */
    int8_t  *gq = malloc((size_t)INTER);
    float   *rd = malloc((size_t)1 << 30);                        /* 1 GiB read baseline */
    if (!w8 || !w4 || !sc || !x || !g || !u || !d || !xq || !gq || !rd) { fprintf(stderr, "OOM\n"); return 1; }
    for (int i = 0; i < HID; i++)   xq[i] = (int8_t)(xr() % 15) - 7;
    for (int i = 0; i < INTER; i++) gq[i] = (int8_t)(xr() % 15) - 7;
    float sx1 = 0.031f;                                           /* one activation scale, S=1 */
    for (int64_t i = 0; i < (int64_t)NEXP * w8_per; i++) w8[i] = (int8_t)(xr() & 0xff);
    for (int64_t i = 0; i < (int64_t)NEXP * w4_per; i++) w4[i] = (uint8_t)(xr() & 0xff);
    for (int64_t i = 0; i < (int64_t)NEXP * (INTER * 2 + HID); i++) sc[i] = 1.0f + (float)(xr() & 7) * 0.01f;
    for (int i = 0; i < HID; i++) x[i] = (float)(int)(xr() % 17) - 8.0f;
    for (int64_t i = 0; i < ((int64_t)1 << 30) / 4; i++) rd[i] = (float)(xr() & 0xff);

    double sink = 0;                                              /* defeats DCE across arms */

    /* --- arm A: int8 GEMV, cold experts (round-robin so no reuse across calls) --- */
    double a5[5];
    for (int r = 0; r < 5; r++) {
        int e = 0; double t0 = tnow();
        for (int t = 0; t < TOKENS; t++)
            for (int k = 0; k < TOPK; k++, e = (e + 1) % NEXP) {
                const int8_t *base = w8 + (int64_t)e * w8_per;
                const float  *s    = sc + (int64_t)e * (INTER * 2 + HID);
                matmul_q(g, x, base,                              s,             1, HID, INTER);
                matmul_q(u, x, base + (int64_t)INTER * HID,       s + INTER,     1, HID, INTER);
                for (int i = 0; i < INTER; i++) g[i] *= u[i];     /* keep g live like the FFN does */
                matmul_q(d, g, base + (int64_t)INTER * HID * 2,   s + INTER * 2, 1, INTER, HID);
                sink += d[0] + d[HID - 1];
            }
        a5[r] = (double)TOKENS * TOPK * w8_per / (tnow() - t0);
    }
    double gbs8 = median5(a5) / 1e9;

    /* --- arm B: same shape, int4 packed --- */
    double b5[5];
    for (int r = 0; r < 5; r++) {
        int e = 0; double t0 = tnow();
        for (int t = 0; t < TOKENS; t++)
            for (int k = 0; k < TOPK; k++, e = (e + 1) % NEXP) {
                const uint8_t *base = w4 + (int64_t)e * w4_per;
                const float   *s    = sc + (int64_t)e * (INTER * 2 + HID);
                matmul_i4(g, x, base,                                  s,             1, HID, INTER);
                matmul_i4(u, x, base + (int64_t)INTER * HID / 2,       s + INTER,     1, HID, INTER);
                for (int i = 0; i < INTER; i++) g[i] *= u[i];
                matmul_i4(d, g, base + (int64_t)INTER * HID,           s + INTER * 2, 1, INTER, HID);
                sink += d[0] + d[HID - 1];
            }
        b5[r] = (double)TOKENS * TOPK * w4_per / (tnow() - t0);
    }
    double gbs4 = median5(b5) / 1e9;

    /* --- arm E: the PRODUCTION decode path — idot int8×int8 (dot_i8i8) --- */
    double e5[5];
    for (int r = 0; r < 5; r++) {
        int e = 0; double t0 = tnow();
        for (int t = 0; t < TOKENS; t++)
            for (int k = 0; k < TOPK; k++, e = (e + 1) % NEXP) {
                const int8_t *base = w8 + (int64_t)e * w8_per;
                const float  *s    = sc + (int64_t)e * (INTER * 2 + HID);
                matmul_q_idot(g, xq, &sx1, base,                            s,             1, HID, INTER);
                matmul_q_idot(u, xq, &sx1, base + (int64_t)INTER * HID,     s + INTER,     1, HID, INTER);
                matmul_q_idot(d, gq, &sx1, base + (int64_t)INTER * HID * 2, s + INTER * 2, 1, INTER, HID);
                sink += d[0] + d[HID - 1];
            }
        e5[r] = (double)TOKENS * TOPK * w8_per / (tnow() - t0);
    }
    double gbs8i = median5(e5) / 1e9;

    /* --- arm F: idot int4 (dot_i4i8) — the GLM fmt-4 decode kernel --- */
    double f5[5];
    for (int r = 0; r < 5; r++) {
        int e = 0; double t0 = tnow();
        for (int t = 0; t < TOKENS; t++)
            for (int k = 0; k < TOPK; k++, e = (e + 1) % NEXP) {
                const uint8_t *base = w4 + (int64_t)e * w4_per;
                const float   *s    = sc + (int64_t)e * (INTER * 2 + HID);
                matmul_i4_idot(g, xq, &sx1, base,                            s,             1, HID, INTER);
                matmul_i4_idot(u, xq, &sx1, base + (int64_t)INTER * HID / 2, s + INTER,     1, HID, INTER);
                matmul_i4_idot(d, gq, &sx1, base + (int64_t)INTER * HID,     s + INTER * 2, 1, INTER, HID);
                sink += d[0] + d[HID - 1];
            }
        f5[r] = (double)TOKENS * TOPK * w4_per / (tnow() - t0);
    }
    double gbs4i = median5(f5) / 1e9;

    /* --- arm G: the OLD single-accumulator idot int4, same inputs — the A/B --- */
    double g5[5]; double gbs4o = 0;
#ifdef __AVX2__
    for (int r = 0; r < 5; r++) {
        int e = 0; double t0 = tnow();
        for (int t = 0; t < TOKENS; t++)
            for (int k = 0; k < TOPK; k++, e = (e + 1) % NEXP) {
                const uint8_t *base = w4 + (int64_t)e * w4_per;
                const float   *s    = sc + (int64_t)e * (INTER * 2 + HID);
                matmul_i4_idot_old(g, xq, &sx1, base,                            s,             1, HID, INTER);
                matmul_i4_idot_old(u, xq, &sx1, base + (int64_t)INTER * HID / 2, s + INTER,     1, HID, INTER);
                matmul_i4_idot_old(d, gq, &sx1, base + (int64_t)INTER * HID,     s + INTER * 2, 1, INTER, HID);
                sink += d[0] + d[HID - 1];
            }
        g5[r] = (double)TOKENS * TOPK * w4_per / (tnow() - t0);
    }
    gbs4o = median5(g5) / 1e9;

    /* bit-exactness spot check across the whole working set, old vs current */
    {
        int64_t rows_checked = 0;
        for (int e = 0; e < NEXP; e++) {
            const uint8_t *base = w4 + (int64_t)e * w4_per;
            for (int o = 0; o < INTER; o += 97) {                 /* stride: cover all experts cheaply */
                int32_t a = dot_i4i8(base + (int64_t)o * (HID / 2), xq, HID);
                int32_t b = dot_i4i8_old(base + (int64_t)o * (HID / 2), xq, HID);
                if (a != b) { printf("EXACTNESS FAIL expert %d row %d: %d != %d\n", e, o, a, b); return 1; }
                rows_checked++;
            }
        }
        printf("  old-vs-new bit-exact on %lld sampled rows\n", (long long)rows_checked);
    }
#endif

    /* --- arm H: the deinterleaved-x candidate, same inputs --- */
    double h5[5]; double gbs4d = 0;
#ifdef __AVX2__
    {
        int8_t *xeb = malloc((size_t)(HID + 1) / 2), *xob = malloc((size_t)(HID + 1) / 2);
        int8_t *geb = malloc((size_t)(INTER + 1) / 2), *gob = malloc((size_t)(INTER + 1) / 2);
        if (!xeb || !xob || !geb || !gob) { fprintf(stderr, "OOM de\n"); return 1; }
        for (int r = 0; r < 5; r++) {
            int e = 0; double t0 = tnow();
            for (int t = 0; t < TOKENS; t++)
                for (int k = 0; k < TOPK; k++, e = (e + 1) % NEXP) {
                    const uint8_t *base = w4 + (int64_t)e * w4_per;
                    const float   *s    = sc + (int64_t)e * (INTER * 2 + HID);
                    matmul_i4_idot_de(g, xq, &sx1, base,                            s,             HID, INTER, xeb, xob);
                    matmul_i4_idot_de(u, xq, &sx1, base + (int64_t)INTER * HID / 2, s + INTER,     HID, INTER, xeb, xob);
                    matmul_i4_idot_de(d, gq, &sx1, base + (int64_t)INTER * HID,     s + INTER * 2, INTER, HID, geb, gob);
                    sink += d[0] + d[HID - 1];
                }
            h5[r] = (double)TOKENS * TOPK * w4_per / (tnow() - t0);
        }
        gbs4d = median5(h5) / 1e9;

        /* exactness: candidate vs current, whole sampled set + ragged tails */
        i4_deinterleave(xq, HID, xeb, xob);
        int64_t rows = 0;
        for (int e = 0; e < NEXP; e++) {
            const uint8_t *base = w4 + (int64_t)e * w4_per;
            for (int o = 0; o < INTER; o += 97) {
                int32_t a = dot_i4i8(base + (int64_t)o * (HID / 2), xq, HID);
                int32_t b = dot_i4i8_de(base + (int64_t)o * (HID / 2), xeb, xob, HID);
                if (a != b) { printf("DE EXACTNESS FAIL e%d o%d: %d != %d\n", e, o, a, b); return 1; }
                rows++;
            }
        }
        for (int I2 = 1; I2 <= 200; I2++) {                       /* ragged sizes, both parities */
            i4_deinterleave(xq, I2, xeb, xob);
            int32_t a = dot_i4i8(w4, xq, I2), b = dot_i4i8_de(w4, xeb, xob, I2);
            if (a != b) { printf("DE TAIL FAIL I=%d: %d != %d\n", I2, a, b); return 1; }
        }
        printf("  candidate bit-exact on %lld sampled rows + I=1..200 tails\n", (long long)rows);
        free(xeb); free(xob); free(geb); free(gob);
    }
#endif

    /* --- arm C: warm sanity — one expert, must beat arm A by a wide margin or the
     *     "cold" arms were never actually streaming RAM --- */
    double c5[5];
    for (int r = 0; r < 5; r++) {
        double t0 = tnow();
        for (int t = 0; t < TOKENS * TOPK; t++) {
            matmul_q(g, x, w8, sc, 1, HID, INTER);
            sink += g[0];
        }
        c5[r] = (double)TOKENS * TOPK * INTER * HID / (tnow() - t0);
    }
    double gbsw = median5(c5) / 1e9;

    /* --- arm W: warm idot4, current vs deinterleaved — one gate matrix in cache.
     *     Separates the ALU ceiling from the streaming cost: if DE wins big here but
     *     only 9% cold, the cold arms are memory-limited and kernel µops are moot. --- */
#ifdef __AVX2__
    {
        int8_t *xeb = malloc((size_t)(HID + 1) / 2), *xob = malloc((size_t)(HID + 1) / 2);
        double wc5[5], wd5[5];
        for (int r = 0; r < 5; r++) {
            double t0 = tnow();
            for (int t = 0; t < TOKENS * TOPK; t++) {
                matmul_i4_idot(g, xq, &sx1, w4, sc, 1, HID, INTER);
                sink += g[0];
            }
            wc5[r] = (double)TOKENS * TOPK * ((int64_t)INTER * HID / 2) / (tnow() - t0);
        }
        for (int r = 0; r < 5; r++) {
            double t0 = tnow();
            for (int t = 0; t < TOKENS * TOPK; t++) {
                matmul_i4_idot_de(g, xq, &sx1, w4, sc, HID, INTER, xeb, xob);
                sink += g[0];
            }
            wd5[r] = (double)TOKENS * TOPK * ((int64_t)INTER * HID / 2) / (tnow() - t0);
        }
        printf("  warm idot4 cur   : %6.2f GB/s   warm idot4 DEINT: %6.2f GB/s   (DE/cur warm: %.2fx)\n",
               median5(wc5) / 1e9, median5(wd5) / 1e9, median5(wd5) / median5(wc5));
        free(xeb); free(xob);
    }
#endif

    /* --- arm D: read ceiling — every thread streams the 1 GiB buffer --- */
    double d5[5];
    for (int r = 0; r < 5; r++) {
        double t0 = tnow(); double acc = 0;
        #pragma omp parallel for reduction(+:acc) schedule(static)
        for (int64_t i = 0; i < ((int64_t)1 << 30) / 4; i += 8) acc += rd[i];
        d5[r] = (double)(1 << 30) / (tnow() - t0);
        sink += acc;
    }
    double gbsr = median5(d5) / 1e9;

    printf("  f32-act int8 cold: %6.2f GB/s of weights  (IDOT=0 path)\n", gbs8);
    printf("  f32-act int4 cold: %6.2f GB/s of weights\n", gbs4);
    printf("  idot int8 cold   : %6.2f GB/s of weights  (production decode, dot_i8i8)\n", gbs8i);
    printf("  idot int4 cold   : %6.2f GB/s of weights  (production decode, dot_i4i8)\n", gbs4i);
    if (gbs4o > 0)
        printf("  idot int4 BASE   : %6.2f GB/s of weights  (frozen baseline copy; 1.00x while quant.h is unchanged)\n",
               gbs4o);
    if (gbs4d > 0)
        printf("  idot int4 DEINT  : %6.2f GB/s of weights  (deinterleaved x)  vs current: %.2fx  tokens vs idot8: %.2fx\n",
               gbs4d, gbs4d / gbs4i, 2.0 * gbs4d / gbs8i);
    printf("  int8 GEMV warm   : %6.2f GB/s  (kernel ceiling, weights in cache)\n", gbsw);
    printf("  read ceiling     : %6.2f GB/s  (parallel strided sum, 1 GiB)\n", gbsr);
    printf("  efficiency vs read: f32i8 %.0f%%  f32i4 %.0f%%  idot8 %.0f%%  idot4 %.0f%%\n",
           100.0 * gbs8 / gbsr, 100.0 * gbs4 / gbsr, 100.0 * gbs8i / gbsr, 100.0 * gbs4i / gbsr);
    printf("  tokens/s, idot4 vs idot8 (same weights, half the bytes): %.2fx\n", 2.0 * gbs4i / gbs8i);
    printf("  (sink %.3g)\n", sink);
    if (gbsw < gbs8 * 0.95) { printf("SANITY FAIL: warm below cold — measurement broken\n"); return 1; }
    printf("  regime of the f32-act path: %s (warm ceiling at %.0f%% of read ceiling)\n",
           gbsw < 0.7 * gbsr ? "COMPUTE-bound" : "bandwidth-bound", 100.0 * gbsw / gbsr);
    return 0;
}
