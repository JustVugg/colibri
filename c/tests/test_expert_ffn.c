/* expert_ffn.h: the SIMD kernels must reproduce the scalar reference bit for
 * bit (f32: same lane order; i8: exact integers), the reference must agree with
 * a double-precision dot, the pair->planar repack must round-trip, and a whole
 * layer through xf_moe_run must equal the per-token loop it replaces. */
#include <stdio.h>
#include <stdlib.h>
#include "../expert_ffn.h"

static unsigned g_seed = 12345;
static unsigned rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return g_seed >> 8; }
static float frand(void) { return ((int)(rnd() & 0xFFFF) - 32768) / 8192.f; }

/* signed-pairs container row (the converter's layout) with random int4 */
static void make_pairs(uint8_t *dst, int O, int I, int8_t *vals) {
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i += 2) {
        int a = (int)(rnd() & 15) - 8, b = (int)(rnd() & 15) - 8;
        vals[(size_t)o * I + i] = (int8_t)a; vals[(size_t)o * I + i + 1] = (int8_t)b;
        dst[(size_t)o * I / 2 + i / 2] = (uint8_t)((a & 0xF) | ((b & 0xF) << 4));
    }
}
static void make_scales(float *sc, size_t n) { for (size_t i = 0; i < n; i++) sc[i] = 0.002f + (rnd() & 1023) / 65536.f; }

/* double-precision dot, and the magnitude of its terms: random rows cancel
 * almost completely, so an error is judged against what was summed, not
 * against the (near-zero) result */
static double dot_ref_d(const int8_t *v, const float *sc, const float *x, int I, double *mag) {
    double a = 0, m = 0;
    for (int g = 0; g < I / 64; g++) { double d = 0; for (int i = 0; i < 64; i++) { double t = (double)v[g * 64 + i] * x[g * 64 + i]; d += t; m += fabs(t) * sc[g]; } a += d * sc[g]; }
    *mag = m; return a;
}

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static void test_row(int I, int O) {
    size_t rb = (size_t)I / 2; int ng = I / 64;
    uint8_t *pairs = malloc(rb * O), *planar = malloc(rb * O); int8_t *vals = malloc((size_t)I * O);
    float *sc = malloc(sizeof(float) * ng * O); float *x = malloc(sizeof(float) * I);
    make_pairs(pairs, O, I, vals); make_scales(sc, (size_t)ng * O); for (int i = 0; i < I; i++) x[i] = frand();
    xf_repack_pairs_signed(planar, pairs, O, I);
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i++)
        if (xf_get(planar + o * rb, i) != vals[(size_t)o * I + i]) { CHECK(0, "repack I=%d O=%d row %d elem %d: %d != %d", I, O, o, i, xf_get(planar + o * rb, i), vals[(size_t)o * I + i]); return; }
    int8_t *xq = malloc(I); float *xsum = malloc(sizeof(float) * ng); float sx = xf_act_i8(x, I, xq, xsum);
    float *xdq = malloc(sizeof(float) * I); for (int i = 0; i < I; i++) xdq[i] = xq[i] * sx;
    double worst_f = 0, worst_i = 0;
    for (int o = 0; o < O; o++) {
        const uint8_t *w = planar + o * rb; const float *s = sc + (size_t)o * ng;
        float rf = xf_dot_f32_ref(w, s, x, I), kf = xf_dot_f32(w, s, x, I);
        CHECK(rf == kf, "f32 kernel != reference at I=%d O=%d row %d: %.9g vs %.9g", I, O, o, kf, rf);
        double mg, d = dot_ref_d(vals + (size_t)o * I, s, x, I, &mg), e = fabs(d - rf) / mg; if (e > worst_f) worst_f = e;
        float ri = xf_dot_i8_ref(w, s, xq, xsum, sx, I), ki = xf_dot_i8(w, s, xq, xsum, sx, I);
        CHECK(ri == ki, "i8 kernel != reference at I=%d O=%d row %d: %.9g vs %.9g", I, O, o, ki, ri);
        double mi, di = dot_ref_d(vals + (size_t)o * I, s, xdq, I, &mi), ei = fabs(di - ri) / mi; if (ei > worst_i) worst_i = ei;
    }
    CHECK(worst_f < 1e-6, "f32 vs double: err %.3g of the summed magnitude at I=%d", worst_f, I);
    CHECK(worst_i < 1e-6, "i8 vs double (on quantized x): err %.3g of the summed magnitude at I=%d", worst_i, I);
    printf("rows I=%5d O=%5d: f32 bit-exact, i8 bit-exact, worst err/magnitude f32 %.2e i8 %.2e\n", I, O, worst_f, worst_i);
    free(pairs); free(planar); free(vals); free(sc); free(x); free(xq); free(xsum); free(xdq);
}

/* the per-token loop xf_moe_run replaces, on the same planar weights */
static void naive_moe(float *out, const float *x, int S, int K, int H, int F, const int *idx, const float *val, const XfExpert *const *ex, int mode) {
    float *g = malloc(sizeof(float) * F), *u = malloc(sizeof(float) * F), *h = malloc(sizeof(float) * F), *y = malloc(sizeof(float) * H);
    int8_t *xq = malloc(H), *hq = malloc(F); float *xs = malloc(sizeof(float) * (H / 64)), *hs = malloc(sizeof(float) * (F / 64));
    for (int s = 0; s < S; s++) {
        float *os = out + (size_t)s * H; memset(os, 0, sizeof(float) * H);
        const float *xs_ = x + (size_t)s * H; float sx = mode ? xf_act_i8(xs_, H, xq, xs) : 0.f;
        for (int k = 0; k < K; k++) {
            int i = s * K + k; if (idx[i] < 0) continue; const XfExpert *e = ex[i];
            for (int r = 0; r < F; r++) {
                if (mode) { g[r] = xf_dot_i8_ref(e->g4 + (size_t)r * H / 2, e->gs + (size_t)r * (H / 64), xq, xs, sx, H); u[r] = xf_dot_i8_ref(e->u4 + (size_t)r * H / 2, e->us + (size_t)r * (H / 64), xq, xs, sx, H); }
                else { g[r] = xf_dot_f32_ref(e->g4 + (size_t)r * H / 2, e->gs + (size_t)r * (H / 64), xs_, H); u[r] = xf_dot_f32_ref(e->u4 + (size_t)r * H / 2, e->us + (size_t)r * (H / 64), xs_, H); }
            }
            xf_swiglu(h, g, u, F);
            float shx = mode ? xf_act_i8(h, F, hq, hs) : 0.f;
            for (int r = 0; r < H; r++)
                y[r] = mode ? xf_dot_i8_ref(e->d4 + (size_t)r * F / 2, e->ds + (size_t)r * (F / 64), hq, hs, shx, F)
                            : xf_dot_f32_ref(e->d4 + (size_t)r * F / 2, e->ds + (size_t)r * (F / 64), h, F);
            float w = val[i]; for (int d = 0; d < H; d++) os[d] += w * y[d];
        }
    }
    free(g); free(u); free(h); free(y); free(xq); free(hq); free(xs); free(hs);
}

static void test_layer(int S, int K, int H, int F, int NE, int mode) {
    size_t gb = (size_t)F * H / 2, db = (size_t)H * F / 2;
    XfExpert *E = malloc(sizeof(XfExpert) * NE); uint8_t *tmp = malloc(gb); int8_t *vals = malloc((size_t)F * H);
    for (int e = 0; e < NE; e++) {
        uint8_t *g4 = malloc(gb), *u4 = malloc(gb), *d4 = malloc(db);
        float *gs = malloc(sizeof(float) * F * (H / 64)), *us = malloc(sizeof(float) * F * (H / 64)), *ds = malloc(sizeof(float) * H * (F / 64));
        make_pairs(tmp, F, H, vals); xf_repack_pairs_signed(g4, tmp, F, H);
        make_pairs(tmp, F, H, vals); xf_repack_pairs_signed(u4, tmp, F, H);
        make_pairs(tmp, H, F, vals); xf_repack_pairs_signed(d4, tmp, H, F);
        make_scales(gs, (size_t)F * (H / 64)); make_scales(us, (size_t)F * (H / 64)); make_scales(ds, (size_t)H * (F / 64));
        XfExpert x = { g4, u4, d4, gs, us, ds }; E[e] = x;
    }
    float *x = malloc(sizeof(float) * S * H); for (int i = 0; i < S * H; i++) x[i] = frand();
    int *idx = malloc(sizeof(int) * S * K); float *val = malloc(sizeof(float) * S * K); const XfExpert **ex = malloc(sizeof(void *) * S * K);
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) {
        int i = s * K + k;
        idx[i] = (k == K - 1 && (s & 1)) ? -1 : (int)(rnd() % NE);   /* a hole in the routing now and then */
        for (int j = 0; j < k; j++) if (idx[i] >= 0 && idx[s * K + j] == idx[i]) idx[i] = (idx[i] + 1) % NE;
        val[i] = idx[i] < 0 ? 0.f : 0.05f + (rnd() & 255) / 512.f; ex[i] = idx[i] < 0 ? NULL : &E[idx[i]];
    }
    float *o1 = malloc(sizeof(float) * S * H), *o2 = malloc(sizeof(float) * S * H);
    void *scratch = malloc(xf_moe_scratch_bytes(S, K, H, F));
    naive_moe(o1, x, S, K, H, F, idx, val, ex, mode);
    xf_moe_run(o2, x, S, K, H, F, idx, val, ex, mode, scratch);
    int bad = 0; for (int i = 0; i < S * H; i++) if (o1[i] != o2[i]) { if (!bad) printf("  first diff at %d: %.9g vs %.9g\n", i, o2[i], o1[i]); bad++; }
    CHECK(bad == 0, "layer S=%d K=%d H=%d F=%d mode=%d: %d of %d outputs differ from the per-token loop", S, K, H, F, mode, bad, S * H);
    printf("layer S=%2d K=%d H=%4d F=%4d mode=%d: bit-identical to the per-token loop\n", S, K, H, F, mode);
    /* the same batch cut into (token, expert) pairs, each added into out: the cut
     * qwen36 makes when its cache holds fewer experts than the batch routes to.
     * Every cut goes through xf_moe_add's one statement, so the bits must match
     * the whole run whether or not the build fuses the multiply-add. */
    float *o3 = calloc((size_t)S * H, sizeof(float)); void *sc1 = malloc(xf_moe_scratch_bytes(1, 1, H, F));
    for (int i = 0; i < S * K; i++) xf_moe_add(o3 + (size_t)(i / K) * H, x + (size_t)(i / K) * H, 1, 1, H, F, idx + i, val + i, ex + i, mode, sc1);
    bad = 0; for (int i = 0; i < S * H; i++) if (o3[i] != o2[i]) { if (!bad) printf("  first diff at %d: %.9g vs %.9g\n", i, o3[i], o2[i]); bad++; }
    CHECK(bad == 0, "layer S=%d K=%d H=%d F=%d mode=%d: %d of %d outputs differ when cut into pairs", S, K, H, F, mode, bad, S * H);
    printf("layer S=%2d K=%d H=%4d F=%4d mode=%d: bit-identical when cut into pairs\n", S, K, H, F, mode);
    free(o3); free(sc1);
    for (int e = 0; e < NE; e++) { free((void *)E[e].g4); free((void *)E[e].u4); free((void *)E[e].d4); free((void *)E[e].gs); free((void *)E[e].us); free((void *)E[e].ds); }
    free(E); free(tmp); free(vals); free(x); free(idx); free(val); free(ex); free(o1); free(o2); free(scratch);
}

int main(void) {
#ifdef XF_HAVE_AVX2
    printf("path: AVX2%s\n",
#ifdef XF_HAVE_VNNI
        " + VNNI");
#else
        " (maddubs)");
#endif
#else
    printf("path: scalar\n");
#endif
    test_row(64, 8); test_row(128, 3); test_row(512, 2048); test_row(2048, 512); test_row(4096, 16);
    /* an unreachable routing (all -1) must zero the output */
    { int idx[4] = {-1, -1, -1, -1}; float val[4] = {0}; const XfExpert *ex[4] = {0}; float x[128], out[128]; void *sc = malloc(xf_moe_scratch_bytes(1, 4, 128, 64));
      for (int i = 0; i < 128; i++) { x[i] = 1.f; out[i] = 7.f; }
      xf_moe_run(out, x, 1, 4, 128, 64, idx, val, ex, 0, sc); int nz = 0; for (int i = 0; i < 128; i++) nz += out[i] != 0.f;
      CHECK(nz == 0, "unrouted token left %d non-zero outputs", nz); free(sc); }
    test_layer(1, 8, 2048, 512, 16, 0);
    test_layer(1, 8, 2048, 512, 16, 1);
    test_layer(7, 8, 2048, 512, 12, 0);   /* prompt rows sharing experts */
    test_layer(7, 8, 2048, 512, 12, 1);
    test_layer(5, 4, 128, 192, 6, 0);
    test_layer(3, 2, 64, 64, 2, 1);
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    printf("all passed\n"); return 0;
}
