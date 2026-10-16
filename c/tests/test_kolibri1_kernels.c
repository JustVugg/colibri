/* The engine's own kernels, checked as components.
 *
 *  (a) BIT EQUALITY between S = 1 and S > 1 inside the SAME implementation (the AVX2
 *      kernel against itself, the scalar kernel against itself, bf16, the router):
 *      a row's result does not depend on S, on its position in the block or on what
 *      the neighbouring rows hold. The two implementations are NOT required to be
 *      bit-identical to each other (different reduction order inside a 128-block);
 *      how many outputs differ is printed.
 *  (b) NUMERICAL gates against an independent float64 reference (tests/
 *      kolibri1_kernel_ref.py, NumPy only; never the reference implementation): normalised RMS <= 1e-6 and
 *      scaled max <= 1e-5 per row (definitions: divided by
 *      max(RMS(ref), 1e-6)), on random, wide-range and cancelling inputs. Thresholds
 *      were fixed before the first run and are not tuned. A heavier cancelling case
 *      is reported (DIAG) and not gated.
 *  (c) RoPE tables: cos/sin of a float64 angle rounded once to float32, 1,000
 *      positions up to 262,143, gate <= 1 ulp of fp32 against NumPy; the number of
 *      bit-identical entries is recorded and bit equality is not assumed. The
 *      mimo.c float32-angle form is measured for contrast.
 *  (d) k1_route against a float64/NumPy reference on 4,096 rows of 384 logits (ids
 *      exact, weights atol 1e-6 / rtol 1e-5), a heavy-tie set, 100,000 logits for
 *      the sigmoid, the tie rule, and the non-finite refusal.
 *
 *   tests/test_kolibri1_kernels [ref dir]
 * Without a directory the reference is computed in-C with long double (and a
 * smaller route set); the make target kolibri1-kernels-check passes the NumPy one.
 */
#define _GNU_SOURCE
#define KOLIBRI1_NO_MAIN
#include <float.h>
#include "../kolibri1.c"

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; if (fails < 40) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

static const double GATE_NRMS = 1e-6, GATE_SMAX = 1e-5;

static uint64_t rng_state = 0x2545F4914F6CDD1Dull;
static uint32_t rnd32(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}
static float rndf(void) { return (rnd32() & 0xFFFFFF) / 8388608.0f - 1.0f; }       /* [-1, 1) */
static float rndn(void) { float s = 0; for (int i = 0; i < 12; i++) s += (rnd32() & 0xFFFF) / 65536.0f; return s - 6.0f; }

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void *slurp(const char *dir, const char *name, size_t want) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    void *p = malloc(want ? want : 1);
    if (fread(p, 1, want, f) != want || fgetc(f) != EOF) { fprintf(stderr, "%s: wrong size (want %zu)\n", path, want); exit(2); }
    fclose(f);
    return p;
}

/* worst per-row metrics of y[S][N] against the float64 reference */
typedef struct { double nrms, smax; int row; } Metrics;
static Metrics metrics(const float *y, const double *ref, int S, int N) {
    Metrics m = { 0, 0, 0 };
    for (int s = 0; s < S; s++) {
        double ss = 0, dd = 0, mx = 0;
        for (int i = 0; i < N; i++) {
            double r = ref[(size_t)s * N + i], d = (double)y[(size_t)s * N + i] - r;
            ss += r * r; dd += d * d; if (fabs(d) > mx) mx = fabs(d);
            if (!isfinite(y[(size_t)s * N + i])) { m.nrms = INFINITY; m.smax = INFINITY; m.row = s; return m; }
        }
        double rms = sqrt(ss / N), den = rms > 1e-6 ? rms : 1e-6;
        double nr = sqrt(dd / N) / den, sm = mx / den;
        if (nr > m.nrms) { m.nrms = nr; m.row = s; }
        if (sm > m.smax) m.smax = sm;
    }
    return m;
}

static void gate(const char *what, Metrics m, int gated) {
    printf("  %-34s nRMS %.3e  scaled max %.3e  (worst row %d)%s\n", what, m.nrms, m.smax, m.row, gated ? "" : "  DIAG, not gated");
    if (gated) CHECK(m.nrms <= GATE_NRMS && m.smax <= GATE_SMAX, "%s: nRMS %.3e (<= %.0e) scaled max %.3e (<= %.0e)",
                     what, m.nrms, GATE_NRMS, m.smax, GATE_SMAX);
}

/* ---------------------------------------------------------------- (a) bit equality */

typedef void (*fp8_fn)(float *, const float *, const uint8_t *, const float *, int, int, int);

static void bit_equal_fp8(const char *name, fp8_fn fn, int O, int I) {
    int nb_o = (int)fp8_nblk(O), nb_i = (int)fp8_nblk(I);
    uint8_t *w = malloc((size_t)O * I);
    float *grid = malloc((size_t)nb_o * nb_i * 4);
    for (size_t i = 0; i < (size_t)O * I; i++) { w[i] = (uint8_t)rnd32(); if (k1_e4m3_is_nan(w[i])) w[i] = 0x30; }
    for (int i = 0; i < nb_o * nb_i; i++) grid[i] = ldexpf(1.0f + (rnd32() & 0xFFFF) / 65536.0f, -9 + (int)(rnd32() % 5));
    const int SMAX = 37;
    float *x = malloc((size_t)SMAX * I * 4), *y1 = malloc((size_t)SMAX * O * 4), *yb = malloc((size_t)SMAX * O * 4);
    for (size_t i = 0; i < (size_t)SMAX * I; i++) x[i] = rndn();
    for (int s = 0; s < SMAX; s++) fn(y1 + (size_t)s * O, x + (size_t)s * I, w, grid, 1, I, O);       /* S = 1, one row at a time */
    long checked = 0, bad = 0;
    static const int sizes[] = { 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 37 };
    for (unsigned k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
        int S = sizes[k];
        fn(yb, x, w, grid, S, I, O);
        for (size_t i = 0; i < (size_t)S * O; i++) { checked++; bad += bits(yb[i]) != bits(y1[i]); }
    }
    /* a row's position in the block and its neighbours do not matter: shuffle the rows, run, compare */
    int perm[37];
    for (int i = 0; i < SMAX; i++) perm[i] = i;
    for (int i = SMAX - 1; i > 0; i--) { int j = (int)(rnd32() % (i + 1)); int t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    float *xs = malloc((size_t)SMAX * I * 4);
    for (int s = 0; s < SMAX; s++) memcpy(xs + (size_t)s * I, x + (size_t)perm[s] * I, (size_t)I * 4);
    fn(yb, xs, w, grid, SMAX, I, O);
    for (int s = 0; s < SMAX; s++) for (int o = 0; o < O; o++) { checked++; bad += bits(yb[(size_t)s * O + o]) != bits(y1[(size_t)perm[s] * O + o]); }
    printf("  %-34s [%d,%d]: %ld outputs compared across S in {1..37} and a shuffled block, %ld differ\n", name, O, I, checked, bad);
    CHECK(bad == 0, "%s [%d,%d]: S=1 and S>1 are not bit-identical (%ld of %ld)", name, O, I, bad, checked);
    free(w); free(grid); free(x); free(y1); free(yb); free(xs);
}

static void bit_equal_bf16(int O, int I) {
    uint16_t *w = malloc((size_t)O * I * 2);
    for (size_t i = 0; i < (size_t)O * I; i++) w[i] = (uint16_t)(bits(rndf() * 0.05f) >> 16);
    const int SMAX = 23;
    float *x = malloc((size_t)SMAX * I * 4), *y1 = malloc((size_t)SMAX * O * 4), *yb = malloc((size_t)SMAX * O * 4);
    float *r1 = malloc((size_t)SMAX * O * 4), *rb = malloc((size_t)SMAX * O * 4);
    for (size_t i = 0; i < (size_t)SMAX * I; i++) x[i] = rndn();
    for (int s = 0; s < SMAX; s++) { k1_matmul_bf16(y1 + (size_t)s * O, x + (size_t)s * I, w, 1, I, O); k1_router_logits(r1 + (size_t)s * O, x + (size_t)s * I, w, 1, I, O); }
    long checked = 0, bad = 0;
    for (int S = 2; S <= SMAX; S += 3) {
        k1_matmul_bf16(yb, x, w, S, I, O); k1_router_logits(rb, x, w, S, I, O);
        for (size_t i = 0; i < (size_t)S * O; i++) { checked += 2; bad += (bits(yb[i]) != bits(y1[i])) + (bits(rb[i]) != bits(r1[i])); }
    }
    printf("  %-34s [%d,%d]: %ld outputs compared, %ld differ\n", "k1_matmul_bf16 + router_logits", O, I, checked, bad);
    CHECK(bad == 0, "bf16 kernels: S=1 and S>1 differ (%ld)", bad);
    free(w); free(x); free(y1); free(yb); free(r1); free(rb);
}

static void test_bit_equality(void) {
    printf("(a) S = 1 vs S > 1, bit equality within one implementation\n");
#ifdef K1_HAVE_VEC
    bit_equal_fp8("k1_matmul_fp8_vec (AVX2)", k1_matmul_fp8_vec, 512, 2560);
    bit_equal_fp8("k1_matmul_fp8_vec (AVX2)", k1_matmul_fp8_vec, 130, 300);      /* partial blocks, tail of 4 */
    bit_equal_fp8("k1_matmul_fp8_vec (AVX2)", k1_matmul_fp8_vec, 19, 203);       /* odd everything */
#else
    printf("  (no AVX2+FMA in this build: the vector kernel is not compiled)\n");
#endif
    bit_equal_fp8("k1_matmul_fp8_scalar", k1_matmul_fp8_scalar, 512, 2560);
    bit_equal_fp8("k1_matmul_fp8_scalar", k1_matmul_fp8_scalar, 130, 300);
    bit_equal_fp8("k1_matmul_fp8_scalar", k1_matmul_fp8_scalar, 19, 203);
    bit_equal_fp8("k1_matmul_fp8 (dispatcher)", k1_matmul_fp8, 130, 300);
    bit_equal_bf16(640, 2560);
    bit_equal_bf16(33, 203);
#ifdef K1_HAVE_VEC
    {   /* control: the two implementations are different arithmetic, and the comparison can see it */
        int O = 256, I = 2560;
        uint8_t *w = malloc((size_t)O * I); float g[2 * 20], *x = malloc((size_t)I * 4), *ya = malloc(O * 4), *yb = malloc(O * 4);
        for (size_t i = 0; i < (size_t)O * I; i++) { w[i] = (uint8_t)rnd32(); if (k1_e4m3_is_nan(w[i])) w[i] = 0x30; }
        for (int i = 0; i < 40; i++) g[i] = ldexpf(1.0f + (rnd32() & 0xFFFF) / 65536.0f, -9);
        for (int i = 0; i < I; i++) x[i] = rndn();
        k1_matmul_fp8_vec(ya, x, w, g, 1, I, O); k1_matmul_fp8_scalar(yb, x, w, g, 1, I, O);
        int diff = 0; for (int o = 0; o < O; o++) diff += bits(ya[o]) != bits(yb[o]);
        printf("  control: vec vs scalar differ in %d of %d outputs (different reduction order; NOT required to match)\n", diff, O);
        free(w); free(x); free(ya); free(yb);
    }
#endif
}

/* ---------------------------------------------------------------- (b) numerical gates */

static char ref_dir_g[4096];

static void numeric_fp8(const char *name, int O, int I, int S, int gated) {
    char nm[128];
    int nbo = (int)fp8_nblk(O), nbi = (int)fp8_nblk(I);
    snprintf(nm, sizeof nm, "%s.codes.u8", name); uint8_t *w = slurp(ref_dir_g, nm, (size_t)O * I);
    snprintf(nm, sizeof nm, "%s.grid.f32", name); float *g = slurp(ref_dir_g, nm, (size_t)nbo * nbi * 4);
    snprintf(nm, sizeof nm, "%s.x.f32", name); float *x = slurp(ref_dir_g, nm, (size_t)S * I * 4);
    snprintf(nm, sizeof nm, "%s.y.f64", name); double *ref = slurp(ref_dir_g, nm, (size_t)S * O * 8);
    float *y = malloc((size_t)S * O * 4);
    char label[160];
    k1_matmul_fp8_scalar(y, x, w, g, S, I, O);
    snprintf(label, sizeof label, "%s [%d,%d] S=%d scalar", name, O, I, S); gate(label, metrics(y, ref, S, O), gated);
#ifdef K1_HAVE_VEC
    k1_matmul_fp8_vec(y, x, w, g, S, I, O);
    snprintf(label, sizeof label, "%s [%d,%d] S=%d vec", name, O, I, S); gate(label, metrics(y, ref, S, O), gated);
    if (S > 1) {
        for (int s = 0; s < S; s++) k1_matmul_fp8_vec(y + (size_t)s * O, x + (size_t)s * I, w, g, 1, I, O);
        snprintf(label, sizeof label, "%s [%d,%d] S=1 x %d vec", name, O, I, S); gate(label, metrics(y, ref, S, O), gated);
    }
#endif
    free(w); free(g); free(x); free(ref); free(y);
}

/* the in-C reference: long double, from the same dequantisation definition */
static void numeric_fp8_selfref(int O, int I, int S) {
    int nbo = (int)fp8_nblk(O), nbi = (int)fp8_nblk(I);
    uint8_t *w = malloc((size_t)O * I); float *g = malloc((size_t)nbo * nbi * 4), *x = malloc((size_t)S * I * 4), *y = malloc((size_t)S * O * 4);
    double *ref = malloc((size_t)S * O * 8);
    for (size_t i = 0; i < (size_t)O * I; i++) { w[i] = (uint8_t)rnd32(); if (k1_e4m3_is_nan(w[i])) w[i] = 0x30; }
    for (int i = 0; i < nbo * nbi; i++) g[i] = ldexpf(1.0f + (rnd32() & 0xFFFF) / 65536.0f, -9 + (int)(rnd32() % 5));
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rndn();
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        long double a = 0;
        for (int i = 0; i < I; i++) a += (long double)k1_dequant(w[(size_t)o * I + i], g[(o / 128) * nbi + i / 128]) * x[(size_t)s * I + i];
        ref[(size_t)s * O + o] = (double)a;
    }
    char label[96];
    k1_matmul_fp8_scalar(y, x, w, g, S, I, O); snprintf(label, sizeof label, "fp8 [%d,%d] S=%d scalar (in-C ref)", O, I, S); gate(label, metrics(y, ref, S, O), 1);
#ifdef K1_HAVE_VEC
    k1_matmul_fp8_vec(y, x, w, g, S, I, O); snprintf(label, sizeof label, "fp8 [%d,%d] S=%d vec (in-C ref)", O, I, S); gate(label, metrics(y, ref, S, O), 1);
#endif
    free(w); free(g); free(x); free(y); free(ref);
}

static void numeric_bf16(const char *name, int O, int I, int S, int router) {
    char nm[128];
    snprintf(nm, sizeof nm, "%s.w.u16", name); uint16_t *w = slurp(ref_dir_g, nm, (size_t)O * I * 2);
    snprintf(nm, sizeof nm, "%s.x.f32", name); float *x = slurp(ref_dir_g, nm, (size_t)S * I * 4);
    snprintf(nm, sizeof nm, "%s.y.f64", name); double *ref = slurp(ref_dir_g, nm, (size_t)S * O * 8);
    float *y = malloc((size_t)S * O * 4);
    char label[160];
    if (router) k1_router_logits(y, x, w, S, I, O); else k1_matmul_bf16(y, x, w, S, I, O);
    snprintf(label, sizeof label, "%s [%d,%d] S=%d %s", name, O, I, S, router ? "router_logits (double acc)" : "matmul_bf16"); gate(label, metrics(y, ref, S, O), 1);
    free(w); free(x); free(ref); free(y);
}

static void numeric_norm(const char *name, int n, int rows) {
    char nm[128];
    snprintf(nm, sizeof nm, "%s.x.f32", name); float *x = slurp(ref_dir_g, nm, (size_t)rows * n * 4);
    snprintf(nm, sizeof nm, "%s.w.f32", name); float *w = slurp(ref_dir_g, nm, (size_t)n * 4);
    snprintf(nm, sizeof nm, "%s.y.f64", name); double *ref = slurp(ref_dir_g, nm, (size_t)rows * n * 8);
    float *y = malloc((size_t)rows * n * 4);
    for (int r = 0; r < rows; r++) k1_rmsnorm(y + (size_t)r * n, x + (size_t)r * n, w, n, 1e-6f);
    char label[96]; snprintf(label, sizeof label, "%s rmsnorm n=%d", name, n); gate(label, metrics(y, ref, rows, n), 1);
    for (int r = 0; r < rows; r++) { float *t = malloc((size_t)n * 4); memcpy(t, x + (size_t)r * n, (size_t)n * 4); k1_rmsnorm(t, t, w, n, 1e-6f);   /* in place */
        CHECK(!memcmp(t, y + (size_t)r * n, (size_t)n * 4), "rmsnorm in place differs from out of place"); free(t); }
    free(x); free(w); free(ref); free(y);
}

static void numeric_norm_selfref(int n) {
    float *x = malloc(n * 4), *w = malloc(n * 4), *y = malloc(n * 4); double *ref = malloc(n * 8);
    for (int i = 0; i < n; i++) { x[i] = rndn() * 3; w[i] = 1.0f + 0.1f * rndn(); }
    long double ss = 0; for (int i = 0; i < n; i++) ss += (long double)x[i] * x[i];
    long double r = 1.0L / sqrtl(ss / n + 1e-6L);
    for (int i = 0; i < n; i++) ref[i] = (double)((long double)w[i] * x[i] * r);
    k1_rmsnorm(y, x, w, n, 1e-6f);
    char label[64]; snprintf(label, sizeof label, "rmsnorm n=%d (in-C ref)", n); gate(label, metrics(y, ref, 1, n), 1);
    free(x); free(w); free(y); free(ref);
}

/* ---------------------------------------------------------------- (c) rope */

static int ulp_dist(float a, float b) {
    int32_t x = (int32_t)bits(a), y = (int32_t)bits(b);
    if (x < 0) x = INT32_MIN - x;
    if (y < 0) y = INT32_MIN - y;
    return abs(x - y);
}

static void test_rope(const char *ref) {
    printf("(c) RoPE tables\n");
    K1Rope r; k1_rope_init(&r, 128, 10000.0);
    CHECK(r.half == 64, "rope half");
    int np = 0;
    int64_t *pos = NULL; float *rc = NULL, *rs = NULL;
    if (ref) {
        np = 0;
        /* the position count is in cases.txt; read it here */
        char path[4096]; snprintf(path, sizeof path, "%s/cases.txt", ref);
        FILE *c = fopen(path, "r"); char line[512], kind[32], nm[64]; long a, b, cc, d;
        while (c && fgets(line, sizeof line, c)) if (sscanf(line, "%31s %63s %ld %ld %ld %ld", kind, nm, &a, &b, &cc, &d) == 6 && !strcmp(kind, "rope")) np = (int)a;
        if (c) fclose(c);
        pos = slurp(ref, "rope.pos.i64", (size_t)np * 8); rc = slurp(ref, "rope.cos.f32", (size_t)np * 64 * 4); rs = slurp(ref, "rope.sin.f32", (size_t)np * 64 * 4);
    } else {
        np = 1000;
        pos = malloc(np * 8); rc = malloc((size_t)np * 64 * 4); rs = malloc((size_t)np * 64 * 4);
        for (int p = 0; p < np; p++) {
            pos[p] = p < 15 ? (int64_t[]){0, 1, 2, 511, 512, 513, 1026, 1027, 4096, 16383, 16384, 16385, 100000, 262142, 262143}[p] : rnd32() % 262144;
            for (int i = 0; i < 64; i++) {
                long double inv = 1.0L / powl(10000.0L, (long double)(2 * i) / 128.0L), a = (long double)pos[p] * inv;
                rc[p * 64 + i] = (float)cosl(a); rs[p * 64 + i] = (float)sinl(a);
            }
        }
    }
    int64_t maxpos = 0; for (int p = 0; p < np; p++) if (pos[p] > maxpos) maxpos = pos[p];
    long ident = 0, total = 0, over1 = 0; int worst = 0; double worst_abs = 0, mimo_worst = 0; int64_t mimo_pos = 0;
    float c[64], s[64];
    for (int p = 0; p < np; p++) {
        k1_rope_tables(&r, pos[p], c, s);
        for (int i = 0; i < 64; i++) {
            CHECK(isfinite(c[i]) && isfinite(s[i]), "non-finite rope entry at position %lld", (long long)pos[p]);
            int d1 = ulp_dist(c[i], rc[p * 64 + i]), d2 = ulp_dist(s[i], rs[p * 64 + i]);
            total += 2; ident += (d1 == 0) + (d2 == 0); over1 += (d1 > 1) + (d2 > 1);
            if (d1 > worst) worst = d1; if (d2 > worst) worst = d2;
            double ea = fabs((double)c[i] - rc[p * 64 + i]); if (ea > worst_abs) worst_abs = ea;
            /* contrast: mimo.c's float32 angle, inv_freq and cos in float32 */
            float inv = 1.0f / powf(10000.0f, (float)(2 * i) / 128.0f), ang = inv * (float)pos[p];
            double em = fabs((double)cosf(ang) - rc[p * 64 + i]); if (em > mimo_worst) { mimo_worst = em; mimo_pos = pos[p]; }
        }
    }
    printf("  %d positions up to %lld x 64 pairs: %ld of %ld cos/sin entries bit-identical to the %s float64 -> float32 tables, "
           "worst %d ulp, %ld entries above 1 ulp, worst abs error %.3e\n",
           np, (long long)maxpos, ident, total, ref ? "NumPy" : "long-double", worst, over1, worst_abs);
    printf("  contrast, mimo.c's float32 angle: worst |cos error| %.3e (at position %lld) -- the double path above is what the engine uses\n", mimo_worst, (long long)mimo_pos);
    CHECK(over1 == 0, "RoPE tables: %ld entries differ by more than 1 ulp (worst %d)", over1, worst);
    CHECK(mimo_worst > 1e-3, "the float32-angle contrast should be large at long positions (%.3e)", mimo_worst);
    /* rotation: a unit rotation preserves the norm of the pair, and matches the reference on random heads */
    if (ref) {
        int nv = 64;
        float *v = slurp(ref, "rope.v.f32", (size_t)nv * 128 * 4); double *yv = slurp(ref, "rope.vy.f64", (size_t)nv * 128 * 8);
        float *out = malloc((size_t)nv * 128 * 4);
        for (int k = 0; k < nv; k++) { memcpy(out + k * 128, v + k * 128, 128 * 4); k1_rope_tables(&r, pos[k], c, s); k1_rope_apply(out + k * 128, 128, c, s); }
        gate("rope_apply on 64 random heads", metrics(out, yv, nv, 128), 1);
        free(v); free(yv); free(out);
    }
    float v0[128], ch[64], sh[64];
    for (int i = 0; i < 128; i++) v0[i] = rndn();
    float n0 = 0; for (int i = 0; i < 128; i++) n0 += v0[i] * v0[i];
    k1_rope_tables(&r, 12345, ch, sh); k1_rope_apply(v0, 128, ch, sh);
    float n1 = 0; for (int i = 0; i < 128; i++) n1 += v0[i] * v0[i];
    CHECK(fabsf(n1 - n0) < 1e-4f * n0, "rope_apply must preserve the norm (%g vs %g)", n0, n1);
    float one[128] = { 0 }; one[0] = 1; k1_rope_tables(&r, 0, ch, sh); k1_rope_apply(one, 128, ch, sh);
    CHECK(one[0] == 1.0f && one[64] == 0.0f, "position 0 is the identity");
    free(pos); free(rc); free(rs);
}

/* ---------------------------------------------------------------- (d) routing */

static void test_route_set(const char *ref, const char *name, int R, int E, int K, double atol, double rtol) {
    char nm[128];
    snprintf(nm, sizeof nm, "%s.z.f32", name); float *z = slurp(ref, nm, (size_t)R * E * 4);
    snprintf(nm, sizeof nm, "%s.bias.f32", name); float *bias = slurp(ref, nm, (size_t)E * 4);
    snprintf(nm, sizeof nm, "%s.ids.i32", name); int32_t *ids = slurp(ref, nm, (size_t)R * K * 4);
    snprintf(nm, sizeof nm, "%s.w.f64", name); double *w = slurp(ref, nm, (size_t)R * K * 8);
    snprintf(nm, sizeof nm, "%s.gap.f64", name); double *gap = slurp(ref, nm, (size_t)R * 8);
    long id_bad = 0, w_bad = 0, near_tie = 0; double worst = 0, min_gap = INFINITY;
    for (int r = 0; r < R; r++) {
        K1Route out;
        int rc = k1_route(&out, z + (size_t)r * E, bias, E, K);
        CHECK(rc == 0, "k1_route failed on a finite row");
        int row_ok = 1;
        for (int k = 0; k < K; k++) {
            row_ok &= out.ids[k] == ids[r * K + k];
            double dw = fabs((double)out.w[k] - w[r * K + k]), tol = atol + rtol * fabs(w[r * K + k]);
            if (dw > tol) w_bad++;
            if (dw / tol > worst) worst = dw / tol;
        }
        if (!row_ok) id_bad++;
        if (gap[r] == 0) near_tie++;
        if (gap[r] > 0 && gap[r] < min_gap) min_gap = gap[r];
    }
    printf("  %-8s %d rows x %d logits, top-%d: %ld rows differ from the reference ids, %ld weights outside atol %.0e rtol %.0e "
           "(worst %.3f of the tolerance); rows with an exact 6th/7th tie %ld, smallest nonzero 6th/7th gap %.3e\n",
           name, R, E, K, id_bad, w_bad, atol, rtol, worst, near_tie, min_gap);
    CHECK(id_bad == 0, "%s: %ld rows select different experts than the reference", name, id_bad);
    CHECK(w_bad == 0, "%s: %ld routing weights outside the tolerance", name, w_bad);
    free(z); free(bias); free(ids); free(w); free(gap);
}

static void test_route(const char *ref) {
    printf("(d) routing\n");
    const int E = 384, K = 6;
    if (ref) {
        test_route_set(ref, "route_a", 4096, E, K, 1e-6, 1e-5);
        test_route_set(ref, "route_b", 512, E, K, 1e-6, 1e-5);
        float *x = slurp(ref, "sigmoid.x.f32", 100000 * 4); double *w = slurp(ref, "sigmoid.w.f64", 100000 * 8);
        long bad = 0; double worst = 0;
        for (int i = 0; i < 100000; i++) {
            double got = k1_sigmoid(x[i]), tol = 1e-6 + 1e-5 * fabs(w[i]), d = fabs(got - w[i]);
            CHECK(isfinite(got) && got >= 0 && got <= 1, "sigmoid(%g) = %g", x[i], got);
            if (d > tol) bad++;
            if (d / tol > worst) worst = d / tol;
        }
        printf("  sigmoid  100000 logits (|x| up to 3.4e38): %ld outside atol 1e-6 rtol 1e-5, worst %.3f of the tolerance\n", bad, worst);
        CHECK(bad == 0, "sigmoid: %ld outside the tolerance", bad);
        free(x); free(w);
    } else {
        /* self reference: long double selection and weights, a smaller set */
        float *z = malloc(E * 4), *bias = malloc(E * 4);
        long bad = 0;
        for (int e = 0; e < E; e++) bias[e] = rndf();
        for (int r = 0; r < 300; r++) {
            for (int e = 0; e < E; e++) z[e] = rndn() * (1 + r % 3);
            K1Route o; CHECK(k1_route(&o, z, bias, E, K) == 0, "route rc");
            int used[384] = { 0 };
            for (int k = 0; k < K; k++) {
                int best = -1;
                for (int e = 0; e < E; e++) if (!used[e] && (best < 0 || z[e] + bias[e] > z[best] + bias[best])) best = e;
                used[best] = 1;
                bad += o.ids[k] != best;
                long double sg = 1.0L / (1.0L + expl(-(long double)z[best]));
                bad += fabsl((long double)o.w[k] - sg) > 1e-6L + 1e-5L * sg;
            }
        }
        printf("  300 rows x 384 (in-C reference): %ld mismatches\n", bad);
        CHECK(bad == 0, "in-C route reference: %ld mismatches", bad);
        free(z); free(bias);
    }
    /* the tie rule: first (lowest) id wins an exact tie, picks come out in descending biased order */
    {
        float z[16], b[16]; K1Route o;
        for (int e = 0; e < 16; e++) { z[e] = 0.25f; b[e] = 0.0f; }
        CHECK(k1_route(&o, z, b, 16, 6) == 0, "tie rc");
        for (int k = 0; k < 6; k++) CHECK(o.ids[k] == k, "all-equal row: pick %d is id %d, want %d", k, o.ids[k], k);
        for (int e = 0; e < 16; e++) { z[e] = (e % 4) * 0.5f; b[e] = 0.0f; }        /* ids 3,7,11,15 tie at the top, then 2,6,10,14 */
        CHECK(k1_route(&o, z, b, 16, 6) == 0, "tie rc");
        int want[6] = { 3, 7, 11, 15, 2, 6 };
        for (int k = 0; k < 6; k++) CHECK(o.ids[k] == want[k], "tie-group row: pick %d is id %d, want %d", k, o.ids[k], want[k]);
        for (int e = 0; e < 16; e++) { z[e] = 0.0f; b[e] = 0.0f; }
        z[5] = -0.0f; b[5] = 0.0f; z[2] = 1.0f; b[2] = -1.0f;                          /* biased tie 0 == -0 and 1 + -1 */
        CHECK(k1_route(&o, z, b, 16, 6) == 0, "tie rc");
        CHECK(o.ids[0] == 0 && o.ids[1] == 1 && o.ids[2] == 2, "biased zero ties: ids %d %d %d", o.ids[0], o.ids[1], o.ids[2]);
        /* selection is on z + bias, the weight is sigmoid(z) alone */
        for (int e = 0; e < 16; e++) { z[e] = 0.0f; b[e] = 0.0f; }
        z[9] = -2.0f; b[9] = 5.0f; z[4] = 1.0f;
        CHECK(k1_route(&o, z, b, 16, 6) == 0, "bias rc");
        CHECK(o.ids[0] == 9, "bias must decide the selection (got %d)", o.ids[0]);
        CHECK(fabsf(o.w[0] - 1.0f / (1.0f + expf(2.0f))) < 1e-7f, "the weight of a biased pick is sigmoid(z), not sigmoid(z + bias): %g", o.w[0]);
        CHECK(o.ids[1] == 4 && fabsf(o.w[1] - 1.0f / (1.0f + expf(-1.0f))) < 1e-7f, "second pick");
        /* no renormalisation: the six weights do not sum to one */
        float s = 0; for (int k = 0; k < 6; k++) s += o.w[k];
        CHECK(fabsf(s - 1.0f) > 0.1f, "weights must not be renormalised (sum %g)", s);
        /* non-finite input is a hard error, never a silent fallback */
        float zn[16], bn[16];
        memcpy(zn, z, sizeof z); memcpy(bn, b, sizeof b);
        zn[3] = NAN; CHECK(k1_route(&o, zn, bn, 16, 6) == -1, "NaN logit must fail");
        memcpy(zn, z, sizeof z); zn[7] = INFINITY; CHECK(k1_route(&o, zn, bn, 16, 6) == -1, "inf logit must fail");
        memcpy(zn, z, sizeof z); bn[0] = NAN; CHECK(k1_route(&o, zn, bn, 16, 6) == -1, "NaN bias must fail");
        memcpy(bn, b, sizeof b);
        CHECK(k1_route(&o, z, bn, 16, 17) == -1 && k1_route(&o, z, bn, 16, 0) == -1, "K out of range must fail");
        printf("  tie rule, bias-vs-weight, no renormalisation, non-finite refusal: checked\n");
    }
}

int main(int argc, char **argv) {
    const char *ref = argc > 1 ? argv[1] : NULL;
    int fp = k1_fp_env_check();
    CHECK(fp == 0, "FTZ/DAZ must be off (code %d)", fp);
    printf("kernels: %s; reference: %s\n",
#ifdef K1_HAVE_VEC
           "AVX2+FMA vector and scalar",
#else
           "scalar only (no AVX2+FMA)",
#endif
           ref ? ref : "(none: in-C long double)");
    test_bit_equality();
    printf("(b) numerical gates: nRMS <= %.0e, scaled max <= %.0e, vs the independent float64 reference\n", GATE_NRMS, GATE_SMAX);
    if (ref) {
        snprintf(ref_dir_g, sizeof ref_dir_g, "%s", ref);
        numeric_fp8("fp8_small", 7, 192, 3, 1);
        numeric_fp8("fp8_partial", 130, 300, 9, 1);
        numeric_fp8("fp8_gate", 512, 2560, 17, 1);
        numeric_fp8("fp8_down", 2560, 512, 5, 1);
        numeric_fp8("fp8_range", 256, 640, 6, 1);
        numeric_fp8("fp8_cancel", 256, 1024, 8, 1);
        numeric_fp8("fp8_cancel_heavy", 64, 2560, 4, 0);
        numeric_bf16("bf16_head", 640, 2560, 7, 0);
        numeric_bf16("bf16_odd", 33, 203, 3, 0);
        numeric_bf16("bf16_cancel", 128, 1024, 5, 0);
        numeric_bf16("router", 384, 2560, 6, 1);
        numeric_norm("norm_2560", 2560, 8); numeric_norm("norm_128", 128, 8); numeric_norm("norm_odd", 203, 8);
        numeric_norm("norm_big", 2560, 8); numeric_norm("norm_small", 2560, 8);
    } else {
        numeric_fp8_selfref(130, 300, 9); numeric_fp8_selfref(512, 2560, 17); numeric_fp8_selfref(19, 203, 5);
        numeric_norm_selfref(2560); numeric_norm_selfref(203);
    }
    test_rope(ref);
    test_route(ref);
    if (fails) { fprintf(stderr, "test_kolibri1_kernels: %d FAILED\n", fails); return 1; }
    printf("test_kolibri1_kernels: PASS\n");
    return 0;
}
