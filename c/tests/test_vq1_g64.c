/* vq1_g64.h: the vq1 kernels against their contract (as tests/test_vq2_g32.c for vq2).
 *   - the layout: a row packed here from known 10-bit indices and 4-bit scales reads back
 *     through vq1_value and vq1_scale;
 *   - vq1_dot (SIMD where built) equals vq1_dot_ref bit for bit for S = 1..4, every row on
 *     its own, with codebooks at the int8 edge (|a| + |b| = 127), and vq1_dot_ref equals a
 *     double-precision dot of the decoded weights to rounding;
 *   - vq1_matmul equals vq1_dot_ref row by row for S = 1..9. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../vq1_g64.h"

static uint32_t rs = 0x9E3779B9u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; if (fails < 10) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static float act_value(void) {
    uint32_t r = xr() % 64;
    if (r == 0) return 0.f;
    float v = ((int)(xr() % 2001) - 1000) * 1e-3f;
    return r == 1 ? v * 40.f : v;
}
/* books as the converter scales them: per coordinate max|a| + max|b| <= 127 over all
 * entries (any a plus any b stays an int8), every coordinate's bound reached */
static void books(Vq1Books *bk) {
    for (int j = 0; j < 16; j++) {
        int ta = 20 + (int)(xr() % 88), tb = 127 - ta;
        for (int e = 0; e < VQ1_BOOK; e++) {
            bk->a[e][j] = (int8_t)(e < 2 ? (e ? -ta : ta) : (int)(xr() % (2 * ta + 1)) - ta);
            bk->b[e][j] = (int8_t)(e < 2 ? (e ? -tb : tb) : (int)(xr() % (2 * tb + 1)) - tb);
        }
    }
}
/* the documented layout, written independently of the kernel's reader: per group of 64 the
 * indices a0 b0 a1 b1 a2 b2 a3 b3 at bits 10k, least significant first, in 10 bytes */
static void pack(uint8_t *row, uint8_t *q, const int *ia, const int *ib, const int *scale, int I) {
    memset(row, 0, vq1_code_bytes(I));
    for (int g = 0; g < I / 64; g++)
        for (int k = 0; k < 8; k++) {
            int v = 4 * g + k / 2, val = k % 2 ? ib[v] : ia[v];
            for (int bit = 0; bit < 10; bit++)
                if (val >> bit & 1) { int pos = 10 * k + bit; row[g * 10 + pos / 8] |= (uint8_t)(1 << (pos % 8)); }
        }
    memset(q, 0, vq1_scale_bytes(I));
    for (int g = 0; g < I / 64; g++) q[g / 2] |= (uint8_t)(scale[g] << (4 * (g & 1)));
}

int main(void) {
    static Vq1Books bk; books(&bk);
    double worst = 0; int cases = 0;
    for (int t = 0; t < 300; t++) {
        int I = 128 * (1 + t % 20), S = 1 + t % 4;
        int *ia = malloc(sizeof(int) * I / 16), *ib = malloc(sizeof(int) * I / 16), *scale = malloc(sizeof(int) * I / 64);
        uint8_t *row = malloc(vq1_code_bytes(I)), *q = malloc(vq1_scale_bytes(I));
        for (int v = 0; v < I / 16; v++) { ia[v] = t % 5 == 0 && v % 3 == 0 ? (int)(xr() & 1) : (int)(xr() % 1024); ib[v] = ia[v] < 2 ? ia[v] : (int)(xr() % 1024); }   /* sums at +-127 */
        for (int g = 0; g < I / 64; g++) scale[g] = (int)(xr() % 16);
        pack(row, q, ia, ib, scale, I);
        int lay = 1;
        for (int i = 0; i < I; i++) lay &= vq1_value(&bk, row, i) == bk.a[ia[i / 16]][i % 16] + bk.b[ib[i / 16]][i % 16];
        for (int g = 0; g < I / 64; g++) lay &= vq1_scale(q, g) == scale[g];
        CHECK(lay, "layout: I=%d does not read back", I);
        float d = 1e-5f * (float)(1 + xr() % 1000);
        float *x = malloc(sizeof(float) * S * I); int8_t *xq = malloc(S * I); float *dx = malloc(sizeof(float) * S * I / 32);
        for (int i = 0; i < S * I; i++) x[i] = act_value();
        for (int s = 0; s < S; s++) x2_quant_act(x + s * I, I, xq + s * I, dx + s * I / 32);
        if (t % 9 == 0) for (int i = 0; i < I; i++) xq[i] = (int8_t)(i & 1 ? 127 : -127);   /* the int16 pair sums at their edge */
        float y[4];
        vq1_dot(y, 1, &bk, row, q, d, xq, dx, I, S);
        for (int s = 0; s < S; s++) {
            float r = vq1_dot_ref(&bk, row, q, d, xq + s * I, dx + s * I / 32, I);
            CHECK(!memcmp(&r, &y[s], sizeof r), "vq1_dot I=%d S=%d row %d: %.9g, reference %.9g", I, S, s, y[s], r);
            float alone; vq1_dot(&alone, 1, &bk, row, q, d, xq + s * I, dx + s * I / 32, I, 1);
            CHECK(!memcmp(&alone, &y[s], sizeof alone), "vq1_dot I=%d: row %d of S=%d differs from S=1", I, s, S);
            double e = 0, mag = 0;
            for (int i = 0; i < I; i++) {
                double w = (double)bk.a[ia[i / 16]][i % 16] + bk.b[ib[i / 16]][i % 16];
                double term = (double)d * scale[i / 64] * w * xq[s * I + i] * dx[s * I / 32 + i / 32];
                e += term; mag += fabs(term);
            }
            double rel = fabs(e - r) / (mag + 1e-30); if (rel > worst) worst = rel;
            cases++;
        }
        free(ia); free(ib); free(scale); free(row); free(q); free(x); free(xq); free(dx);
    }
    for (int S = 1; S <= 9; S++) {
        int I = 640, O = 5; size_t cb = vq1_code_bytes(I), qb = vq1_scale_bytes(I);
        uint8_t *codes = malloc(O * cb), *q = malloc(O * qb); float d[5];
        for (size_t i = 0; i < O * cb; i++) codes[i] = (uint8_t)xr();   /* any 10-bit index is a book entry */
        for (size_t i = 0; i < O * qb; i++) q[i] = (uint8_t)xr();
        for (int o = 0; o < O; o++) d[o] = 1e-4f * (float)(1 + o);
        float *x = malloc(sizeof(float) * S * I), *y = malloc(sizeof(float) * S * O);
        int8_t *xq = malloc(I); float dx[640 / 32];
        for (int i = 0; i < S * I; i++) x[i] = act_value();
        CHECK(!vq1_matmul(y, x, &bk, codes, q, d, S, I, O), "vq1_matmul S=%d failed", S);
        for (int s = 0; s < S; s++) {
            x2_quant_act(x + s * I, I, xq, dx);
            for (int o = 0; o < O; o++) {
                float r = vq1_dot_ref(&bk, codes + o * cb, q + o * qb, d[o], xq, dx, I);
                CHECK(!memcmp(&r, &y[s * O + o], sizeof r), "vq1_matmul S=%d [%d,%d]: %.9g, reference %.9g", S, s, o, y[s * O + o], r);
            }
        }
        free(codes); free(q); free(x); free(y); free(xq);
    }
    CHECK(worst < 1e-6, "vq1_dot_ref against double: %.2e of the terms' magnitude", worst);
    printf("test_vq1_g64: %d dot cases, worst reference error %.2e of the terms; %s (%s)\n", cases, worst,
           fails ? "FAIL" : "ok",
#ifdef VQ1_HAVE_AVX2
           "AVX2"
#else
           "scalar"
#endif
    );
    return fails != 0;
}
