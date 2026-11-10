/* test_nofma_kernels: the kernels a build without FMA takes, against the references they must
 * reproduce bit for bit (#2715). Built twice by the Makefile on x86-64: for the x86-64 baseline
 * (SSE2) and for Sandy Bridge (AVX, no FMA); elsewhere it checks what the dispatch gives.
 *
 *   fp8    quant.h's matmul_fp8 (one row: four output rows a pass; two rows or more: the rows in
 *          the lanes of a vector) against matmul_fp8_scalar: rows 1..33 (the lane tail and whole
 *          groups), widths on and off the 128-block grid, outputs off the four-row pass, weights
 *          with the NaN codes, activations of a full 24-bit mantissa (the products round)
 *   int4   expert_ffn.h's xf_dot_f32 and xf_dot_f32_rows (one to nine rows a call, a stride)
 *          against xf_dot_f32_ref: random codes and scales over wide exponent ranges, activations
 *          with zeros of both signs and subnormals, the all-0 and all-15 codes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../quant.h"
#include "../expert_ffn.h"

static int fails = 0;
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd64(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng * 0x2545F4914F6CDD1DULL; }
static float mant_x(void) {   /* a float in +-[1,2) * 2^e with every mantissa bit random */
    uint64_t r = rnd64(); union { uint32_t b; float f; } v;
    v.b = 0x3F800000u | (uint32_t)(r & 0x7FFFFFu);
    float f = ldexpf(v.f, (int)((r >> 24) % 9) - 4);
    return (r >> 40) & 1 ? -f : f;
}
static int same_bits(const float *a, const float *b, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        if (isnan(a[i]) && isnan(b[i])) continue;
        if (memcmp(a + i, b + i, sizeof(float))) return 0;
    }
    return 1;
}

static void fp8_case(int S, int I, int O) {
    int64_t nblk = fp8_nblk(O) * fp8_nblk(I);
    uint8_t *q8 = malloc((size_t)O * I);
    float *bs = malloc((size_t)nblk * sizeof(float)), *x = malloc((size_t)S * I * sizeof(float));
    float *y = malloc((size_t)S * O * sizeof(float)), *y1 = malloc((size_t)S * O * sizeof(float));
    for (int64_t i = 0; i < (int64_t)O * I; i++) { uint8_t b = (uint8_t)rnd64(); q8[i] = (b == 0x7F || b == 0xFF) ? 0x3C : b; }
    q8[(int64_t)(O - 1) * I + I / 2] = 0x7F;   /* a NaN weight: its output row stays a NaN */
    for (int64_t b = 0; b < nblk; b++) bs[b] = ldexpf(1.f + (float)(rnd64() & 0xFFFF) / 65536.f, (int)(rnd64() % 7) - 9);
    for (int64_t i = 0; i < (int64_t)S * I; i++) x[i] = mant_x();
    matmul_fp8(y, x, q8, bs, S, I, O);
    matmul_fp8_scalar(y1, x, q8, bs, S, I, O);
    if (!same_bits(y, y1, (int64_t)S * O)) { printf("FAIL fp8 S=%d I=%d O=%d: not matmul_fp8_scalar's bits\n", S, I, O); fails++; }
    free(q8); free(bs); free(x); free(y); free(y1);
}

static float wide_f(int lo, int hi) {   /* random sign and mantissa, exponent in [lo, hi] */
    uint64_t r = rnd64(); union { uint32_t b; float f; } v;
    v.b = 0x3F800000u | (uint32_t)(r & 0x7FFFFFu);
    float f = ldexpf(v.f, lo + (int)((r >> 23) % (uint64_t)(hi - lo + 1)));
    return (r >> 40) & 1 ? -f : f;
}
static void int4_case(int I, int S, int kind) {
    const int ng = I / XF_BLOCK;
    uint8_t *w = malloc((size_t)I / 2); float *sc = malloc((size_t)ng * sizeof(float));
    float **xr = malloc((size_t)S * sizeof(float *)); const float **xc = malloc((size_t)S * sizeof(float *));
    float *y = malloc((size_t)S * 3 * sizeof(float)), *yref = malloc((size_t)S * sizeof(float));
    for (int i = 0; i < I / 2; i++) w[i] = kind == 1 ? 0x00 : kind == 2 ? 0xFF : (uint8_t)rnd64();
    for (int g = 0; g < ng; g++) sc[g] = kind == 3 ? wide_f(-140, 100) : wide_f(-12, 2);
    for (int s = 0; s < S; s++) {
        xr[s] = malloc((size_t)I * sizeof(float)); xc[s] = xr[s];
        for (int i = 0; i < I; i++) {
            uint64_t r = rnd64() % 64;
            xr[s][i] = r == 0 ? 0.f : r == 1 ? -0.f : r == 2 ? wide_f(-149, -127) : kind == 3 ? wide_f(-120, 60) : mant_x();
        }
        yref[s] = xf_dot_f32_ref(w, sc, xr[s], I);
        float one = xf_dot_f32(w, sc, xr[s], I);
        if (!same_bits(&one, yref + s, 1)) { printf("FAIL xf_dot_f32 I=%d kind %d row %d: %a, the reference %a\n", I, kind, s, one, yref[s]); fails++; }
    }
    for (int i = 0; i < S * 3; i++) y[i] = -1.f;
    xf_dot_f32_rows(y, 3, w, sc, xc, I, S);   /* every third float: the stride a matmul's output has */
    for (int s = 0; s < S; s++)
        if (!same_bits(y + (size_t)s * 3, yref + s, 1)) { printf("FAIL xf_dot_f32_rows I=%d S=%d kind %d row %d\n", I, S, kind, s); fails++; }
    for (int s = 0; s < S; s++) free(xr[s]);
    free(w); free(sc); free(xr); free(xc); free(y); free(yref);
}

int main(void) {
#if defined(__x86_64__) && defined(__AVX__) && !defined(__FMA__)
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx")) { printf("test_nofma_kernels: this CPU has no AVX, skipped\n"); return 0; }
#endif
    printf("test_nofma_kernels: fp8 %s, int4 f32 %s\n",
#ifdef COLI_FP8_NOFMA_LANES
           FP8L == 8 ? "rows in 8 AVX lanes" : "rows in 4 SSE2 lanes",
#else
           "the dispatch's kernel",
#endif
#ifdef XF_HAVE_F32_DBL
#ifdef __AVX__
           "AVX doubles"
#else
           "SSE2 doubles"
#endif
#else
           "the dispatch's kernel"
#endif
    );
    const int Ss[] = { 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33 };
    const int Is[] = { 128, 129, 200, 1000, 2048 };
    const int Os[] = { 1, 3, 4, 5, 130, 257 };
    for (unsigned a = 0; a < sizeof Ss / sizeof *Ss; a++)
        for (unsigned b = 0; b < sizeof Is / sizeof *Is; b++)
            for (unsigned c = 0; c < sizeof Os / sizeof *Os; c++) fp8_case(Ss[a], Is[b], Os[c]);
    const int I4[] = { 64, 128, 512, 2048, 4096 };
    for (unsigned b = 0; b < sizeof I4 / sizeof *I4; b++)
        for (int S = 1; S <= 9; S++)
            for (int kind = 0; kind < 4; kind++) int4_case(I4[b], S, kind);
    if (fails) { printf("test_nofma_kernels: %d FAILED\n", fails); return 1; }
    printf("test_nofma_kernels: ok\n");
    return 0;
}
