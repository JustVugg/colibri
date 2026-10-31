/* int2_g32.h: the int2-g32 routed-expert format's kernels against their contract.
 *
 *   - x2_quant_act is gq_quant_act (ggml_quants.h): the same int8 and the same scales,
 *     so int2 experts and the GGUF path see one activation;
 *   - the layout: element i's code is where the header says (a row packed here from
 *     known codes reads back through x2_code);
 *   - x2_dot (SIMD where built) equals x2_dot_ref bit for bit for S = 1..4, every row
 *     on its own, and x2_dot_ref equals a double-precision dot of the decoded weights
 *     to rounding;
 *   - x2_matmul equals x2_dot_ref row by row for S = 1..9 (its chunks of four). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../int2_g32.h"
#include "../ggml_quants.h"

static uint32_t rs = 0x9E3779B9u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; if (fails < 10) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static float act_value(void) {   /* mostly small, now and then an outlier, now and then zero */
    uint32_t r = xr() % 64;
    if (r == 0) return 0.f;
    float v = ((int)(xr() % 2001) - 1000) * 1e-3f;
    return r == 1 ? v * 40.f : v;
}

int main(void) {
    /* activation contract */
    for (int t = 0; t < 50; t++) {
        int I = 32 * (1 + t % 40);
        float *x = malloc(sizeof(float) * I); int8_t *a = malloc(I), *b = malloc(I);
        float *da = malloc(sizeof(float) * I / 32), *db = malloc(sizeof(float) * I / 32); int32_t *xs = malloc(sizeof(int32_t) * I / 16);
        for (int i = 0; i < I; i++) x[i] = t % 7 == 0 && i < 32 ? 0.f : act_value();
        x2_quant_act(x, I, a, da); gq_quant_act(x, I, b, db, xs);
        CHECK(!memcmp(a, b, I) && !memcmp(da, db, sizeof(float) * I / 32), "x2_quant_act differs from gq_quant_act at I=%d", I);
        free(x); free(a); free(b); free(da); free(db); free(xs);
    }
    /* layout */
    {
        enum { I = 256 };
        int want[I]; uint8_t row[I / 4] = {0};
        for (int i = 0; i < I; i++) {
            want[i] = (int)(xr() & 3);
            row[(i / 128) * 32 + i % 32] |= (uint8_t)(want[i] << (2 * ((i % 128) / 32)));
        }
        for (int i = 0; i < I; i++) CHECK(x2_code(row, i) == want[i], "x2_code(%d) = %d, packed %d", i, x2_code(row, i), want[i]);
    }
    /* dots */
    double worst = 0; int cases = 0;
    for (int t = 0; t < 300; t++) {
        int I = 128 * (1 + t % 24), S = 1 + t % 4;
        uint8_t *codes = malloc(I / 4), *q = malloc(I / 32);
        for (int i = 0; i < I / 4; i++) codes[i] = (uint8_t)xr();
        for (int g = 0; g < I / 32; g++) q[g] = (uint8_t)xr();
        float d = 1e-4f * (float)(1 + xr() % 1000);
        float *x = malloc(sizeof(float) * S * I); int8_t *xq = malloc(S * I); float *dx = malloc(sizeof(float) * S * I / 32);
        for (int i = 0; i < S * I; i++) x[i] = act_value();
        for (int s = 0; s < S; s++) x2_quant_act(x + s * I, I, xq + s * I, dx + s * I / 32);
        float y[4];
        x2_dot(y, 1, codes, q, d, xq, dx, I, S);
        for (int s = 0; s < S; s++) {
            float r = x2_dot_ref(codes, q, d, xq + s * I, dx + s * I / 32, I);
            CHECK(!memcmp(&r, &y[s], sizeof r), "x2_dot I=%d S=%d row %d: %.9g, reference %.9g", I, S, s, y[s], r);
            float alone; x2_dot(&alone, 1, codes, q, d, xq + s * I, dx + s * I / 32, I, 1);
            CHECK(!memcmp(&alone, &y[s], sizeof alone), "x2_dot I=%d: row %d of S=%d differs from S=1", I, s, S);
            double e = 0, mag = 0;
            for (int i = 0; i < I; i++) {
                double term = (double)d * q[i / 32] * X2_LEVELS[x2_code(codes, i)] * xq[s * I + i] * dx[s * I / 32 + i / 32];
                e += term; mag += fabs(term);
            }
            double rel = fabs(e - r) / (mag + 1e-30); if (rel > worst) worst = rel;
            cases++;
        }
        free(codes); free(q); free(x); free(xq); free(dx);
    }
    CHECK(worst < 1e-6, "x2_dot_ref against double: %.2e of the terms' magnitude", worst);
    /* matmul */
    for (int S = 1; S <= 9; S++) {
        int I = 384, O = 7; size_t cb = x2_code_bytes(I);
        uint8_t *codes = malloc(O * cb), *q = malloc(O * x2_groups(I)); float d[7];
        for (size_t i = 0; i < O * cb; i++) codes[i] = (uint8_t)xr();
        for (int i = 0; i < O * x2_groups(I); i++) q[i] = (uint8_t)xr();
        for (int o = 0; o < O; o++) d[o] = 1e-3f * (float)(1 + o);
        float *x = malloc(sizeof(float) * S * I), *y = malloc(sizeof(float) * S * O);
        int8_t *xq = malloc(I); float dx[384 / 32];
        for (int i = 0; i < S * I; i++) x[i] = act_value();
        CHECK(!x2_matmul(y, x, codes, q, d, S, I, O), "x2_matmul S=%d failed", S);
        for (int s = 0; s < S; s++) {
            x2_quant_act(x + s * I, I, xq, dx);
            for (int o = 0; o < O; o++) {
                float r = x2_dot_ref(codes + o * cb, q + o * x2_groups(I), d[o], xq, dx, I);
                CHECK(!memcmp(&r, &y[s * O + o], sizeof r), "x2_matmul S=%d [%d,%d]: %.9g, reference %.9g", S, s, o, y[s * O + o], r);
            }
        }
        free(codes); free(q); free(x); free(y); free(xq);
    }
    printf("test_int2_g32: %d dot cases, worst reference error %.2e of the terms; %s\n", cases, worst, fails ? "FAIL" : "ok");
    return fails != 0;
}
