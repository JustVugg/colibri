/* Inkling-2's CPU kernels, model free.
 *
 * matmul_h_lanes (an AVX2 build's bf16 matrices for Inkling-2) against a double reference, a
 * row's bits the same at every S, and matmul_w taking it only for Inkling-2: with g_ink_hl
 * off a bf16 matrix still goes through matmul_h, Inkling's path, bit for bit.
 *
 * ink_xg64_expert (an int4 g64 container's expert, its rows over the threads) against the
 * same rows one after another with xf_dot_f32, bit for bit. */
#define main inkling_main_unused
#include "../inkling.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

static uint32_t rng = 12345u;
static float frand(void) {   /* [-1, 1) */
    rng = rng * 1664525u + 1013904223u;
    return (float)((rng >> 8) & 0xFFFF) / 32768.f - 1.f;
}
static uint8_t brand(void) { rng = rng * 1664525u + 1013904223u; return (uint8_t)(rng >> 24); }
static uint16_t bf16(float x) { union { float f; uint32_t u; } v = { x }; return (uint16_t)(v.u >> 16); }
static float from_bf16(uint16_t h) { union { uint32_t u; float f; } v = { (uint32_t)h << 16 }; return v.f; }

#ifdef INK_HL
static void check_lanes(int S, int I, int O) {
    uint16_t *W = malloc((size_t)O * I * sizeof(uint16_t));
    float *x = falloc((int64_t)S * I), *y = falloc((int64_t)S * O), *y1 = falloc(O);
    float *yw = falloc((int64_t)S * O), *yh = falloc((int64_t)S * O), *yhw = falloc((int64_t)S * O);
    for (int64_t i = 0; i < (int64_t)O * I; i++) W[i] = bf16(frand());
    for (int64_t i = 0; i < (int64_t)S * I; i++) x[i] = frand();
    matmul_h_lanes(y, x, W, S, I, O);
    for (int s = 0; s < S; s++) {
        for (int o = 0; o < O; o++) {
            double ref = 0, mag = 0;
            for (int i = 0; i < I; i++) {
                const double p = (double)x[(int64_t)s * I + i] * from_bf16(W[(int64_t)o * I + i]);
                ref += p; mag += fabs(p);
            }
            const double err = fabs(y[(int64_t)s * O + o] - ref);
            /* f32 lanes of I/8 terms each, then a tree of three: the bound is about (I/8 + 3) units
             * of roundoff of the sum of |terms|, 3.5e-5 of it at I = 4608 */
            CHECK(err <= 4e-5 * mag + 1e-30, "lanes S=%d I=%d row %d col %d: %g vs %g", S, I, s, o,
                  y[(int64_t)s * O + o], ref);
        }
        matmul_h_lanes(y1, x + (int64_t)s * I, W, 1, I, O);   /* the same row alone: decode */
        CHECK(!memcmp(y1, y + (int64_t)s * O, (size_t)O * sizeof(float)),
              "lanes S=%d I=%d row %d: bits differ from the row alone", S, I, s);
    }
    Wt w; memset(&w, 0, sizeof(w)); w.h = W;
    g_ink_hl = 1; matmul_w(yw, x, w, S, I, O);
    CHECK(!memcmp(yw, y, (size_t)S * O * sizeof(float)), "matmul_w with Inkling-2 on: not matmul_h_lanes (S=%d)", S);
    g_ink_hl = 0; matmul_w(yhw, x, w, S, I, O);
    matmul_h(yh, x, W, S, I, O);
    CHECK(!memcmp(yhw, yh, (size_t)S * O * sizeof(float)), "matmul_w with Inkling-2 off: not matmul_h (S=%d)", S);
    free(W); free(x); free(y); free(y1); free(yw); free(yh); free(yhw);
}
#endif

static void check_xg64(int D, int I) {
    Model m; memset(&m, 0, sizeof(m));
    m.c.hidden = D; m.c.moe_inter = I; m.c.v2 = 1; m.c.gate_cap = 4.f; m.c.up_cap = 25.f;
    Slot e; memset(&e, 0, sizeof(e));
    e.p13 = malloc((size_t)2 * I * D / 2); e.s13 = falloc((int64_t)2 * I * (D / XF_BLOCK));
    e.p2 = malloc((size_t)D * I / 2);      e.s2 = falloc((int64_t)D * (I / XF_BLOCK));
    for (int64_t i = 0; i < (int64_t)2 * I * D / 2; i++) e.p13[i] = brand();
    for (int64_t i = 0; i < (int64_t)D * I / 2; i++) e.p2[i] = brand();
    for (int64_t i = 0; i < (int64_t)2 * I * (D / XF_BLOCK); i++) e.s13[i] = 0.01f * (1.5f + frand());
    for (int64_t i = 0; i < (int64_t)D * (I / XF_BLOCK); i++) e.s2[i] = 0.01f * (1.5f + frand());
    float *x = falloc(D), *g = falloc(2 * I), *hh = falloc(D), *gr = falloc(2 * I), *hr = falloc(D);
    for (int i = 0; i < D; i++) x[i] = frand();
    ink_xg64_expert(&m, &e, x, g, hh);
    for (int r = 0; r < 2 * I; r++)
        gr[r] = xf_dot_f32(e.p13 + (int64_t)r * (D / 2), e.s13 + (int64_t)r * (D / XF_BLOCK), x, D);
    ink_glu(&m.c, gr, gr + I, I);
    for (int o = 0; o < D; o++)
        hr[o] = xf_dot_f32(e.p2 + (int64_t)o * (I / 2), e.s2 + (int64_t)o * (I / XF_BLOCK), gr, I);
    CHECK(!memcmp(g, gr, (size_t)I * sizeof(float)), "xg64 D=%d I=%d: the SwiGLU differs from the rows in order", D, I);
    CHECK(!memcmp(hh, hr, (size_t)D * sizeof(float)), "xg64 D=%d I=%d: the output differs from the rows in order", D, I);
    free(e.p13); free(e.s13); free(e.p2); free(e.s2); free(x); free(g); free(hh); free(gr); free(hr);
}

int main(void) {
#ifdef INK_HL
    const int Ss[] = { 1, 3, 4, 5, 9 };
    for (int k = 0; k < 5; k++) { check_lanes(Ss[k], 96, 37); check_lanes(Ss[k], 4608, 19); }
#else
    puts("inkling2 kernels: matmul_h_lanes not in this build (no AVX2/FMA, or vdpbf16ps): skipped");
#endif
    check_xg64(128, 64);
    check_xg64(4608, 2304);
    if (failures) { fprintf(stderr, "inkling2 kernels: %d failure(s)\n", failures); return 1; }
    puts("inkling2 kernels: ok");
    return 0;
}
