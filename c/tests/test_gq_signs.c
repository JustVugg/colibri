/* The grid types' dot product with the signs on the activation (ggml_quants.h gq_sbu_*,
 * GQ_DOT_SBU256) against the one it replaced (the signed decoders gq_sb_* through GQ_DOT_SB256,
 * built here as they were): the same bits for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S, one to
 * four activation rows, random blocks with small scales; gq_matmul through the new kernels too.
 * And IQ1_M's decoder, which reads its fields at once, against the one that read them a load each.
 * Every grid byte is checked to be positive and below 64, which the new form relies on. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ggml_quants.h"

static uint32_t rs = 0x2545F491u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

#if defined(__AVX2__) && defined(__FMA__)
GQ_DOT_SB256(old_iq2_xxs, 66, gq_bd_f16, gq_sb_iq2_xxs)
GQ_DOT_SB256(old_iq2_xs, 74, gq_bd_f16, gq_sb_iq2_xs)
GQ_DOT_SB256(old_iq2_s, 82, gq_bd_f16, gq_sb_iq2_s)
GQ_DOT_SB256(old_iq3_xxs, 98, gq_bd_f16, gq_sb_iq3_xxs)
GQ_DOT_SB256(old_iq3_s, 110, gq_bd_f16, gq_sb_iq3_s)
/* IQ1_M's decoder as it was (a load a field), against the one that reads them at once */
GQ_SB_FN __m256i old_sb_iq1_m(const uint8_t *b, float d, int ib, float *s) {
    const uint8_t *qs = b, *qh = b + 32, *scp = b + 48;
    uint16_t sc[4]; for (int i = 0; i < 4; i++) sc[i] = gq_rd16(scp + 2 * i);
    for (int h = 0; h < 2; h++) { int s16 = 2 * ib + h; int v = (sc[s16 >> 2] >> (3 * (s16 & 3))) & 7; s[h] = d * (float)(2 * v + 1) * 0.125f; }
    uint64_t g[4]; int8_t ds[4];
    for (int l = 0; l < 4; l++) {
        int k = 4 * ib + l, hh = (qh[k >> 1] >> (4 * (k & 1))) & 0x0F;
        memcpy(&g[l], GGML_IQ1S_GRID + 8 * (qs[k] | ((hh & 7) << 8)), 8); ds[l] = (hh & 8) ? -1 : 1;
    }
    __m256i v = _mm256_set_epi64x((long long)g[3], (long long)g[2], (long long)g[1], (long long)g[0]);
    v = _mm256_add_epi8(v, v); v = _mm256_add_epi8(v, v); v = _mm256_add_epi8(v, v);
    __m256i dl = _mm256_set_epi64x((long long)(0x0101010101010101ull * (uint8_t)ds[3]), (long long)(0x0101010101010101ull * (uint8_t)ds[2]),
                                   (long long)(0x0101010101010101ull * (uint8_t)ds[1]), (long long)(0x0101010101010101ull * (uint8_t)ds[0]));
    return _mm256_add_epi8(v, dl);
}
GQ_DOT_SB256(old_iq1_m, 56, gq_bd_iq1_m, old_sb_iq1_m)
typedef void (*dot_fn)(float *, int, const uint8_t *, const int8_t *, const float *, int, int);
static int grid_ok(const uint8_t *g, size_t n, const char *name) {
    for (size_t i = 0; i < n; i++) if (g[i] == 0 || g[i] >= 64) { printf("FAIL %s grid byte %zu is %d\n", name, i, g[i]); return 0; }
    return 1;
}
#endif

int main(void) {
#if !(defined(__AVX2__) && defined(__FMA__))
    puts("test_gq_signs: the kernels are AVX2's, not in this build: nothing to check");
    return 0;
#else
    int fails = 0;
    fails += !grid_ok(GGML_IQ2XXS_GRID, sizeof GGML_IQ2XXS_GRID, "IQ2_XXS") + !grid_ok(GGML_IQ2XS_GRID, sizeof GGML_IQ2XS_GRID, "IQ2_XS") +
             !grid_ok(GGML_IQ2S_GRID, sizeof GGML_IQ2S_GRID, "IQ2_S") + !grid_ok(GGML_IQ3XXS_GRID, sizeof GGML_IQ3XXS_GRID, "IQ3_XXS") +
             !grid_ok(GGML_IQ3S_GRID, sizeof GGML_IQ3S_GRID, "IQ3_S");
    const struct { int t, bytes; dot_fn now, old; } ty[] = {
        {GQ_IQ2_XXS, 66, gq_dot_iq2_xxs, old_iq2_xxs}, {GQ_IQ2_XS, 74, gq_dot_iq2_xs, old_iq2_xs}, {GQ_IQ2_S, 82, gq_dot_iq2_s, old_iq2_s},
        {GQ_IQ3_XXS, 98, gq_dot_iq3_xxs, old_iq3_xxs}, {GQ_IQ3_S, 110, gq_dot_iq3_s, old_iq3_s},
        {GQ_IQ1_M, 56, gq_dot_iq1_m, old_iq1_m}};
    const int I = 2048, O = 96;
    float *x = malloc((size_t)4 * I * sizeof(float)), y0[4], y1[4], *ym = malloc((size_t)4 * O * sizeof(float));
    for (int i = 0; i < 4 * I; i++) x[i] = ((float)(xr() % 4001) - 2000.f) / 400.f;
    gq_act a[4]; memset(a, 0, sizeof a);
    for (int S = 1; S <= 4; S++) if (gq_act_set(&a[S - 1], x, S, I)) { puts("FAIL activation"); return 1; }
    int checked = 0;
    for (size_t k = 0; k < sizeof ty / sizeof *ty; k++) {
        const int64_t rb = gq_row_bytes(ty[k].t, I);
        if (rb <= 0) { printf("FAIL %s: no row bytes\n", gq_type_name(ty[k].t)); fails++; continue; }
        uint8_t *W = malloc((size_t)rb * O);
        for (int64_t i = 0; i < rb * O; i++) W[i] = (uint8_t)(xr() & 0xFF);
        if (ty[k].t != GQ_IQ1_M)   /* IQ1_M keeps its scale in the block's last bytes: random there */
            for (int64_t o = 0; o < O; o++) for (int64_t b = 0; b < rb; b += ty[k].bytes) { W[o * rb + b] = (uint8_t)(xr() & 0xFF); W[o * rb + b + 1] = (uint8_t)(0x18 + xr() % 16); }
        for (int S = 1; S <= 4; S++)
            for (int o = 0; o < O; o++) {
                ty[k].now(y1, 1, W + o * rb, a[S - 1].q, a[S - 1].d, I, S);
                ty[k].old(y0, 1, W + o * rb, a[S - 1].q, a[S - 1].d, I, S);
                if (memcmp(y0, y1, (size_t)S * sizeof(float))) {
                    printf("FAIL %s S=%d row %d: %.9g != %.9g\n", gq_type_name(ty[k].t), S, o, (double)y1[0], (double)y0[0]); fails++; break;
                }
                checked++;
            }
        /* the whole matrix through gq_matmul: row o of y is old_*'s */
        gq_matmul_act(ym, &a[0], W, ty[k].t, O);
        for (int o = 0; o < O; o++) {
            ty[k].old(y0, 1, W + o * rb, a[0].q, a[0].d, I, 1);
            if (memcmp(&ym[o], y0, sizeof(float))) { printf("FAIL %s gq_matmul row %d\n", gq_type_name(ty[k].t), o); fails++; break; }
        }
        free(W);
    }
    printf("test_gq_signs: %d rows and row counts, %d failures\n", checked, fails);
    if (fails) return 1;
    puts("PASS");
    return 0;
#endif
}
