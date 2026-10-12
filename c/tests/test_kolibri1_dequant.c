/* The engine reads FP8 block-scaled weights bit-exactly.
 *
 * Every one of the 256 e4m3 codes times each of 4096 FP32 scales (non-power-of-two,
 * subnormal-range, huge, negative, specials) is dequantised by the engine
 * (k1_dequant = e4m3 LUT * scale, one fp32 multiply) and compared BITWISE on every
 * non-NaN result against
 *   (1) an independent in-C decoder built from sign/exponent/mantissa arithmetic, and
 *   (2) when a reference directory is given (tests/kolibri1_c1_ref.py), torch's
 *       `fp8.to(float32) * scale`.
 * NaN is classified separately: exactly the codes 0x7F and 0xFF decode to NaN (e4m3fn
 * has no infinity), no other code does, they stay NaN for every finite scale, and the
 * 14 subnormal codes and the two zeros are valid finite values that compare bitwise.
 * Grid orientation: [ceil(O/128), ceil(I/128)] on [256,384], [192,128] and the real
 * [2560,6144]; a transposed grid fails the shape check and, where it is not a
 * flat-identical re-labelling, the values. Subnormal products are exact because the
 * test asserts FTZ/DAZ are off.
 *
 *   tests/test_kolibri1_dequant [ref dir]
 * Without a directory only the in-C independent decoder runs (the test-c suite); the
 * make target kolibri1-c1-check generates the torch reference and passes it.
 */
#define _GNU_SOURCE
#define KOLIBRI1_NO_MAIN
#include <float.h>
#include "../kolibri1.c"

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; if (fails < 40) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float fbits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* e4m3fn from first principles: no table, no bit tricks */
static float indep_e4m3(int b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    float v = e == 0 ? ldexpf((float)m, -9) : ldexpf((float)(8 + m), e - 10);
    return s ? -v : v;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}

static void *read_file(const char *dir, const char *name, size_t want) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    void *p = malloc(want);
    if (fread(p, 1, want, f) != want || fgetc(f) != EOF) { fprintf(stderr, "%s: wrong size (want %zu)\n", path, want); exit(2); }
    fclose(f);
    return p;
}

#define NS 4096

static void test_codes_and_scales(const char *ref) {
    float *scales = malloc(NS * sizeof(float));
    uint32_t *ref_prod = NULL, *ref_dec = NULL;
    if (ref) {
        float *s = read_file(ref, "scales.f32", NS * 4);
        memcpy(scales, s, NS * 4); free(s);
        ref_prod = read_file(ref, "products.u32", 256 * NS * 4);
        ref_dec = read_file(ref, "decode.u32", 256 * 4);
    } else {
        /* the same classes, from a fixed xorshift stream */
        for (int i = 0; i < NS; i++) {
            uint32_t r = rnd32(), e = 1 + (rnd32() % 254), m = rnd32() & 0x7FFFFF, sgn = r & 1;
            switch (i % 8) {
            case 0: scales[i] = fbits(sgn << 31 | e << 23 | m); break;
            case 1: scales[i] = 0.004f + (rnd32() % 8000) * 1e-6f; break;
            case 2: scales[i] = fbits(sgn << 31 | (1 + m % 0x7FFFFF)); break;                 /* subnormal */
            case 3: scales[i] = fbits(sgn << 31 | (uint32_t)(190 + rnd32() % 60) << 23 | m); break; /* huge */
            case 4: scales[i] = fbits(sgn << 31 | (uint32_t)(1 + rnd32() % 30) << 23 | m); break;   /* tiny normal */
            default: scales[i] = ldexpf(1.0f + (m / 8388608.0f), (int)(rnd32() % 40) - 20); break;
            }
        }
        scales[0] = 0.0f; scales[1] = -0.0f; scales[2] = 1.0f;
    }
    long n_nan = 0, n_inf = 0, n_sub = 0, n_cmp = 0, n_zero = 0, n_ref = 0;
    for (int c = 0; c < 256; c++) {
        float lut = e4m3_decode((uint8_t)c), ind = indep_e4m3(c);
        int want_nan = c == 0x7F || c == 0xFF;
        CHECK(k1_e4m3_is_nan((uint8_t)c) == want_nan, "is_nan(0x%02X) wrong", c);
        CHECK((isnan(lut) != 0) == want_nan, "LUT NaN class of 0x%02X", c);
        CHECK((isnan(ind) != 0) == want_nan, "independent decoder NaN class of 0x%02X", c);
        if (!want_nan) CHECK(bits(lut) == bits(ind), "decode 0x%02X: LUT %a vs independent %a", c, lut, ind);
        if (ref_dec) {
            float rd = fbits(ref_dec[c]);
            CHECK((isnan(rd) != 0) == want_nan, "torch NaN class of 0x%02X", c);
            if (!want_nan) CHECK(bits(lut) == ref_dec[c], "decode 0x%02X vs torch", c);
        }
#ifdef K1_HAVE_VEC
        {   /* the AVX2 register decoder agrees with the LUT on every code, NaN positions included */
            uint8_t blk[8] = { (uint8_t)c, (uint8_t)c, (uint8_t)c, (uint8_t)c, (uint8_t)c, (uint8_t)c, (uint8_t)c, (uint8_t)c };
            float v[8]; _mm256_storeu_ps(v, e4m3_decode8(blk));
            for (int k = 0; k < 8; k++)
                CHECK(want_nan ? isnan(v[k]) : bits(v[k]) == bits(lut), "decode8 0x%02X lane %d", c, k);
        }
#endif
        for (int i = 0; i < NS; i++) {
            float s = scales[i], got = k1_dequant((uint8_t)c, s), mine = ind * s;
            if (want_nan) {
                CHECK(isnan(got), "NaN code 0x%02X x scale %a must stay NaN, got %a", c, s, got);
                n_nan++;
                continue;
            }
            CHECK(!isnan(got), "finite code 0x%02X x finite scale %a is NaN", c, s);
            CHECK(bits(got) == bits(mine), "0x%02X x %a: engine %a vs independent %a", c, s, got, mine);
            n_cmp++;
            if (isinf(got)) n_inf++;
            else if (got == 0) n_zero++;
            else if (fabsf(got) < FLT_MIN) n_sub++;
            if (ref_prod) {
                CHECK(bits(got) == ref_prod[c * NS + i], "0x%02X x %a: engine %a vs torch %a", c, s, got, fbits(ref_prod[c * NS + i]));
                n_ref++;
            }
        }
    }
    printf("codes x scales: 256 x %d; %ld finite-code products compared bitwise (%ld of them to torch), "
           "%ld NaN-code products classified; results: %ld inf, %ld zero, %ld subnormal\n",
           NS, n_cmp, n_ref, n_nan, n_inf, n_zero, n_sub);
    CHECK(n_sub > 1000, "the scale set must exercise subnormal products (%ld)", n_sub);
    free(scales); free(ref_prod); free(ref_dec);
}

static void test_orientation(const char *ref, int O, int I) {
    int gr = (int)fp8_nblk(O), gc = (int)fp8_nblk(I);
    uint8_t *codes = malloc((size_t)O * I);
    float *grid = malloc((size_t)gr * gc * 4), *w = malloc((size_t)O * I * 4), *expect = NULL;
    if (ref) {
        char nm[64];
        snprintf(nm, sizeof nm, "%dx%d.codes.u8", O, I); uint8_t *c = read_file(ref, nm, (size_t)O * I); memcpy(codes, c, (size_t)O * I); free(c);
        snprintf(nm, sizeof nm, "%dx%d.grid.f32", O, I); float *g = read_file(ref, nm, (size_t)gr * gc * 4); memcpy(grid, g, (size_t)gr * gc * 4); free(g);
        snprintf(nm, sizeof nm, "%dx%d.w.f32", O, I); expect = read_file(ref, nm, (size_t)O * I * 4);
    } else {
        for (size_t i = 0; i < (size_t)O * I; i++) { codes[i] = (uint8_t)rnd32(); if (k1_e4m3_is_nan(codes[i])) codes[i] = 0x3C; }
        for (int i = 0; i < gr * gc; i++) grid[i] = ldexpf(1.0f + (rnd32() & 0xFFFF) / 65536.0f, -12 + (int)(rnd32() % 8));
        expect = malloc((size_t)O * I * 4);
        /* the independent oracle: expand the grid with explicit block loops, decode from first principles */
        for (int br = 0; br < gr; br++) for (int bc = 0; bc < gc; bc++)
            for (int o = br * 128; o < O && o < (br + 1) * 128; o++)
                for (int i = bc * 128; i < I && i < (bc + 1) * 128; i++)
                    expect[(size_t)o * I + i] = indep_e4m3(codes[(size_t)o * I + i]) * grid[br * gc + bc];
    }
    CHECK(k1_grid_ok(O, I, gr, gc), "grid [%d,%d] must be accepted for [%d,%d]", gr, gc, O, I);
    CHECK(gr == gc || !k1_grid_ok(O, I, gc, gr), "transposed grid [%d,%d] must be refused for [%d,%d]", gc, gr, O, I);
    CHECK(!k1_grid_ok(O, I, gr + 1, gc) && !k1_grid_ok(O, I, gr, gc + 1), "off-by-one grid accepted");
    k1_dequant_matrix(w, codes, grid, O, I);
    size_t bad = 0;
    for (size_t i = 0; i < (size_t)O * I; i++) bad += bits(w[i]) != bits(expect[i]);
    CHECK(bad == 0, "[%d,%d]: %zu dequantised values differ from the reference", O, I, bad);
    /* mutation: the same flat grid data read as the transposed grid (what a loader that
     * confused the two axes would hand the kernel) must NOT reproduce the reference */
    if (gr > 1 && gc > 1) {
        float *gt = malloc((size_t)gr * gc * 4), *w2 = malloc((size_t)O * I * 4);
        for (int r = 0; r < gr; r++) for (int c = 0; c < gc; c++) gt[c * gr + r] = grid[r * gc + c];   /* [gc][gr] */
        k1_dequant_matrix(w2, codes, gt, O, I);          /* flat data of the transposed grid read as [gr][gc] */
        size_t diff = 0;
        for (size_t i = 0; i < (size_t)O * I; i++) diff += bits(w2[i]) != bits(expect[i]);
        CHECK(diff > 0, "[%d,%d]: a transposed grid reproduced the reference -- the test cannot see an orientation bug", O, I);
        printf("orientation [%d,%d] grid [%d,%d]: matches the reference bitwise; transposed grid changes %zu of %zu values\n",
               O, I, gr, gc, diff, (size_t)O * I);
        free(gt); free(w2);
    } else {
        printf("orientation [%d,%d] grid [%d,%d]: matches the reference bitwise; transposed shape [%d,%d] refused\n", O, I, gr, gc, gc, gr);
    }
    free(codes); free(grid); free(w); free(expect);
}

int main(int argc, char **argv) {
    const char *ref = argc > 1 ? argv[1] : NULL;
    int fp = k1_fp_env_check();
    CHECK(fp == 0, "FTZ/DAZ must be off for exact subnormal products (code %d)", fp);
    /* the check itself bites: a flush-to-zero CPU would make 2^-126 * 0.5 vanish */
    volatile float tiny = 0x1p-126f, half = 0.5f;
    volatile float p = tiny * half;
    CHECK(p != 0.0f && p == 0x1p-127f, "subnormal arithmetic is not exact here");
    printf("reference: %s\n", ref ? ref : "(none: independent in-C decoder only)");
    test_codes_and_scales(ref);
    test_orientation(ref, 256, 384);
    test_orientation(ref, 192, 128);
    test_orientation(ref, 2560, 6144);
    if (fails) { fprintf(stderr, "test_kolibri1_dequant: %d FAILED\n", fails); return 1; }
    printf("test_kolibri1_dequant: PASS\n");
    return 0;
}
