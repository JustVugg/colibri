/* lowbit_g16.h: the int2-g16 and int3-g16 kernels against their contract.
 *   - the layout: a row packed here from known codes and 4-bit scales reads back through
 *     xl_code and xl_scale (both widths);
 *   - xl_dot (SIMD where built) equals xl_dot_ref bit for bit for S = 1..4 at 2 and 3 bits,
 *     every row on its own, and xl_dot_ref equals a double-precision dot of the decoded
 *     weights to rounding;
 *   - xl_matmul equals xl_dot_ref row by row for S = 1..9. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../lowbit_g16.h"

static uint32_t rs = 0x2545F491u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; if (fails < 10) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static float act_value(void) {
    uint32_t r = xr() % 64;
    if (r == 0) return 0.f;
    float v = ((int)(xr() % 2001) - 1000) * 1e-3f;
    return r == 1 ? v * 40.f : v;
}
/* the documented layout, written independently of the kernel's reader */
static void pack(uint8_t *row, const int *code, const int *scale, uint8_t *q, int bits, int I) {
    memset(row, 0, xl_code_bytes(bits, I)); memset(q, 0, xl_scale_bytes(I));
    for (int i = 0; i < I; i++) {
        uint8_t *blk = row + (size_t)(i / 128) * xl_block_bytes(bits);
        int k = i % 128, h = k / 32, j = k % 32;
        (void)h;
        blk[j] |= (uint8_t)((code[i] & 3) << (2 * h));
        if (bits == 3 && (code[i] & 4)) blk[32 + k % 16] |= (uint8_t)(1 << (k / 16));
    }
    for (int g = 0; g < I / 16; g++) q[g / 2] |= (uint8_t)(scale[g] << (4 * (g & 1)));
}

int main(void) {
    double worst = 0; int cases = 0;
    for (int bits = 2; bits <= 3; bits++) {
        for (int t = 0; t < 200; t++) {
            int I = 128 * (1 + t % 20), S = 1 + t % 4;
            int *code = malloc(sizeof(int) * I), *scale = malloc(sizeof(int) * I / 16);
            uint8_t *row = malloc(xl_code_bytes(bits, I)), *q = malloc(xl_scale_bytes(I));
            for (int i = 0; i < I; i++) code[i] = (int)(xr() % (bits == 3 ? 8 : 4));
            for (int g = 0; g < I / 16; g++) scale[g] = (int)(xr() % 16);
            pack(row, code, scale, q, bits, I);
            int lay = 1;
            for (int i = 0; i < I; i++) lay &= xl_code(row, bits, i) == code[i];
            for (int g = 0; g < I / 16; g++) lay &= xl_scale(q, g) == scale[g];
            CHECK(lay, "layout: int%d I=%d does not read back", bits, I);
            float d = 1e-4f * (float)(1 + xr() % 1000);
            float *x = malloc(sizeof(float) * S * I); int8_t *xq = malloc(S * I); float *dx = malloc(sizeof(float) * S * I / 32);
            for (int i = 0; i < S * I; i++) x[i] = act_value();
            for (int s = 0; s < S; s++) x2_quant_act(x + s * I, I, xq + s * I, dx + s * I / 32);
            float y[4];
            xl_dot(y, 1, row, q, d, bits, xq, dx, I, S);
            for (int s = 0; s < S; s++) {
                float r = xl_dot_ref(row, q, d, bits, xq + s * I, dx + s * I / 32, I);
                CHECK(!memcmp(&r, &y[s], sizeof r), "xl_dot int%d I=%d S=%d row %d: %.9g, reference %.9g", bits, I, S, s, y[s], r);
                float alone; xl_dot(&alone, 1, row, q, d, bits, xq + s * I, dx + s * I / 32, I, 1);
                CHECK(!memcmp(&alone, &y[s], sizeof alone), "xl_dot int%d I=%d: row %d of S=%d differs from S=1", bits, I, s, S);
                double e = 0, mag = 0;
                for (int i = 0; i < I; i++) {
                    double term = (double)d * scale[i / 16] * xl_level(bits, code[i]) * xq[s * I + i] * dx[s * I / 32 + i / 32];
                    e += term; mag += fabs(term);
                }
                double rel = fabs(e - r) / (mag + 1e-30); if (rel > worst) worst = rel;
                cases++;
            }
            free(code); free(scale); free(row); free(q); free(x); free(xq); free(dx);
        }
        for (int S = 1; S <= 9; S++) {
            int I = 384, O = 5; size_t cb = xl_code_bytes(bits, I), qb = xl_scale_bytes(I);
            uint8_t *codes = malloc(O * cb), *q = malloc(O * qb); float d[5];
            for (size_t i = 0; i < O * cb; i++) codes[i] = (uint8_t)xr();
            for (size_t i = 0; i < O * qb; i++) q[i] = (uint8_t)xr();
            for (int o = 0; o < O; o++) d[o] = 1e-3f * (float)(1 + o);
            float *x = malloc(sizeof(float) * S * I), *y = malloc(sizeof(float) * S * O);
            int8_t *xq = malloc(I); float dx[384 / 32];
            for (int i = 0; i < S * I; i++) x[i] = act_value();
            CHECK(!xl_matmul(y, x, codes, q, d, bits, S, I, O), "xl_matmul S=%d failed", S);
            for (int s = 0; s < S; s++) {
                x2_quant_act(x + s * I, I, xq, dx);
                for (int o = 0; o < O; o++) {
                    float r = xl_dot_ref(codes + o * cb, q + o * qb, d[o], bits, xq, dx, I);
                    CHECK(!memcmp(&r, &y[s * O + o], sizeof r), "xl_matmul int%d S=%d [%d,%d]: %.9g, reference %.9g", bits, S, s, o, y[s * O + o], r);
                }
            }
            free(codes); free(q); free(x); free(y); free(xq);
        }
    }
    CHECK(worst < 1e-6, "xl_dot_ref against double: %.2e of the terms' magnitude", worst);
    printf("test_lowbit_g16: %d dot cases at 2 and 3 bits, worst reference error %.2e of the terms; %s\n", cases, worst, fails ? "FAIL" : "ok");
    return fails != 0;
}
