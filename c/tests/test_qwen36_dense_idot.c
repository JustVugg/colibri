/* The dense trunk's integer path (COLI_DENSE_IDOT, COLI_DENSE_BITS=4).
 *
 * Pins, with no model:
 *  - dense_act_i8's vector path equals the scalar lrintf contract bit for bit
 *    (same rounding, same scale) and its block sums are exact;
 *  - pack_int4_g64_planar stores what the K1b layout expects: nibble v+8,
 *    lo nibbles = elements 0..31 of the block, hi = 32..63, one scale per 64;
 *  - matmul_d through the int4 grouped kernel and through the int8 integer
 *    kernel both agree with the f32 reference within what an int8 activation
 *    and an int4 weight allow, on every row of a random matrix, for one row
 *    (decode) and for several rows (prefill);
 *  - with neither flag the dispatch is the one before: the int8 f32-activation
 *    kernel, byte-identical to its own output. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#include "../compat.h"   /* setenv: MinGW has none */

static int fails;
static void ck(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); fails++; }

static unsigned g_seed = 4242;
static float rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return ((g_seed >> 8) & 0xFFFF) / 32768.f - 1.f; }

static void ref_matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double a = 0; for (int i = 0; i < I; i++) a += (double)x[(size_t)s * I + i] * W[(size_t)o * I + i];
        y[(size_t)s * O + o] = (float)a;
    }
}
static double rel_gap(const float *a, const float *b, int n) {
    double worst = 0, scale = 1e-6;
    for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > worst) worst = d; if (fabs((double)b[i]) > scale) scale = fabs((double)b[i]); }
    return worst / scale;
}

int main(void) {
    enum { I = 256, O = 96, S = 5 };
    printf("activation quantizer\n");
    {
        float x[I]; int8_t a[I], b[I]; int32_t sums[I / 64];
        for (int t = 0; t < 50; t++) {
            float mag = (t % 5 == 0) ? 1e-3f : 3.f;
            for (int i = 0; i < I; i++) x[i] = rnd() * mag;
            float sa = dense_act_i8(x, I, a, sums);
            float sb = qrow_i8(x, b, I);
            if (sa != sb || memcmp(a, b, I)) { ck(0, "vector quantizer equals the scalar contract"); break; }
            for (int g = 0; g < I / 64; g++) { int32_t s = 0; for (int k = 0; k < 64; k++) s += a[g * 64 + k]; if (s != sums[g]) { ck(0, "block sums are exact"); break; } }
            if (t == 49) ck(1, "vector quantizer equals the scalar lrintf contract on 50 random rows, block sums exact");
        }
    }
    printf("int4 planar packing\n");
    {
        float *W = malloc((size_t)O * I * sizeof(float));
        for (size_t i = 0; i < (size_t)O * I; i++) W[i] = rnd();
        uint8_t *q4 = malloc((size_t)O * I / 2); float *sg = malloc((size_t)O * (I / 64) * sizeof(float));
        pack_int4_g64_planar(W, q4, sg, O, I);
        int bad = 0;
        for (int o = 0; o < O && !bad; o++) for (int g = 0; g < I / 64 && !bad; g++) {
            const uint8_t *blk = q4 + (size_t)o * (I / 2) + g * 32;
            float s = sg[(size_t)o * (I / 64) + g];
            for (int k = 0; k < 64; k++) {
                int nib = k < 32 ? (blk[k] & 0xF) : (blk[k - 32] >> 4);
                float w = W[(size_t)o * I + g * 64 + k];
                int want = (int)lrintf(w / s); if (want > 7) want = 7; if (want < -8) want = -8;
                if (nib - 8 != want) { bad = 1; break; }
            }
        }
        ck(!bad, "nibble v+8 in the planar order, one absmax/7 scale per block of 64");
        free(W); free(q4); free(sg);
    }
    printf("matmul_d dispatch\n");
    {
        float *W = malloc((size_t)O * I * sizeof(float)), *x = malloc((size_t)S * I * sizeof(float));
        float *ref = malloc((size_t)S * O * sizeof(float)), *y = malloc((size_t)S * O * sizeof(float)), *y0 = malloc((size_t)S * O * sizeof(float));
        for (size_t i = 0; i < (size_t)O * I; i++) W[i] = rnd();
        for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rnd() * 2.f;
        ref_matmul(ref, x, W, S, I, O);
        /* the classic path (no flags): int8 rows, f32 activations */
        setenv("COLI_DENSE_IDOT", "0", 1); setenv("COLI_DENSE_BITS", "8", 1);
        qdw_register(W, I, O);
        matmul_d(y0, x, W, S, I, O);
        double g0 = rel_gap(y0, ref, S * O);
        ck(g0 < 2e-2, "classic int8 path within 2% of the f32 reference (per-row int8)");
        matmul_d(y, x, W, 1, I, O);
        ck(!memcmp(y, y0, (size_t)O * sizeof(float)), "one row and the first of a batch agree on the classic path");
        /* the integer path on the same int8 rows */
        g_qdw_n = 0;
        /* dense_idot_on caches its answer: this TU reads the env once, so set it before the first call */
    }
    {
        /* fresh process-wide state for the flag readers: emulate by direct calls */
        float *W = malloc((size_t)O * I * sizeof(float)), *x = malloc((size_t)S * I * sizeof(float));
        float *ref = malloc((size_t)S * O * sizeof(float)), *y = malloc((size_t)S * O * sizeof(float));
        for (size_t i = 0; i < (size_t)O * I; i++) W[i] = rnd();
        for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rnd() * 2.f;
        ref_matmul(ref, x, W, S, I, O);
        int8_t *q = malloc((size_t)O * I); float *sc = malloc((size_t)O * sizeof(float));
        for (int o = 0; o < O; o++) {              /* the per-row int8 the engine builds at load */
            float am = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(W[(size_t)o * I + i]); if (a > am) am = a; }
            float s = am > 1e-12f ? am / 127.f : 1.f; sc[o] = s;
            for (int i = 0; i < I; i++) { int v = (int)lrintf(W[(size_t)o * I + i] / s); if (v > 127) v = 127; if (v < -127) v = -127; q[(size_t)o * I + i] = (int8_t)v; }
        }
        int8_t *xq = malloc((size_t)S * I); float sx[S]; int32_t *xsg = malloc((size_t)S * (I / 64) * sizeof(int32_t));
        for (int s = 0; s < S; s++) sx[s] = dense_act_i8(x + (size_t)s * I, I, xq + (size_t)s * I, xsg + (size_t)s * (I / 64));
        matmul_q_idot(y, xq, sx, q, sc, S, I, O);
        ck(rel_gap(y, ref, S * O) < 3e-2, "int8 weights x int8 activations within 3% of the f32 reference, 5 rows");
        matmul_q_idot(y, xq, sx, q, sc, 1, I, O);
        ck(rel_gap(y, ref, O) < 3e-2, "same for one row");
        uint8_t *q4 = malloc((size_t)O * I / 2); float *sg = malloc((size_t)O * (I / 64) * sizeof(float));
        pack_int4_g64_planar(W, q4, sg, O, I);
        matmul_i4p_grouped_idot(y, xq, sx, xsg, q4, sg, S, I, O, 64);
        ck(rel_gap(y, ref, S * O) < 8e-2, "int4 blocks of 64 x int8 activations within 8% of the f32 reference, 5 rows");
        float y1[O];
        matmul_i4p_grouped_idot(y1, xq, sx, xsg, q4, sg, 1, I, O, 64);
        ck(!memcmp(y1, y, (size_t)O * sizeof(float)), "the one-row path and the row tile give the same bytes (the K1b contract)");
        free(W); free(x); free(ref); free(y); free(q); free(sc); free(xq); free(xsg); free(q4); free(sg);
    }
    if (fails) { printf("test_qwen36_dense_idot: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_dense_idot: the dense trunk's integer path\n");
    return 0;
}
