#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../backend_metal.h"
#include "../quant.h"

enum { NB = 2, D = 64, II = 64, S = 3, GS = 64, SLAB = 16384 };

typedef struct {
    uint8_t *base;
    uint8_t *g, *u, *d;
    float *gs, *us, *ds;
} Expert;

static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

static void pack_matrix(uint8_t *q4, float *sc, int O, int I, int seed) {
    const int rb = (I + 1) / 2;
    const int ng = (I + GS - 1) / GS;
    memset(q4, 0, (size_t)O * rb);
    for (int o = 0; o < O; o++) {
        for (int g = 0; g < ng; g++)
            sc[(size_t)o * ng + g] = 0.0065f + 0.00017f * (float)((o + 3*g + seed) % 11);
        for (int i = 0; i < I; i++) {
            int q = ((o * 17 + i * 13 + seed * 7) % 16) - 8;
            uint8_t n = (uint8_t)(q + 8);
            uint8_t *b = &q4[(size_t)o * rb + (i >> 1)];
            if (i & 1) *b |= (uint8_t)(n << 4);
            else *b = (uint8_t)((*b & 0xf0u) | n);
        }
    }
}

static int make_expert(Expert *e, int seed) {
    void *p = NULL;
    if (posix_memalign(&p, 16384, SLAB) != 0 || !p) return 0;
    memset(p, 0, SLAB);
    e->base = (uint8_t *)p;

    const size_t q_g = (size_t)II * ((D + 1) / 2);
    const size_t s_g = (size_t)II * ((D + GS - 1) / GS) * sizeof(float);
    const size_t q_d = (size_t)D * ((II + 1) / 2);
    const size_t s_d = (size_t)D * ((II + GS - 1) / GS) * sizeof(float);

    size_t at = 0;
    e->g = e->base + at; at += q_g;
    at = align_up(at, 16); e->gs = (float *)(e->base + at); at += s_g;
    at = align_up(at, 16); e->u = e->base + at; at += q_g;
    at = align_up(at, 16); e->us = (float *)(e->base + at); at += s_g;
    at = align_up(at, 16); e->d = e->base + at; at += q_d;
    at = align_up(at, 16); e->ds = (float *)(e->base + at); at += s_d;
    if (at > SLAB) return 0;

    pack_matrix(e->g, e->gs, II, D, seed + 1);
    pack_matrix(e->u, e->us, II, D, seed + 2);
    pack_matrix(e->d, e->ds, D, II, seed + 3);
    return 1;
}

static void cpu_one(const Expert *e, const float *x, float *y, float limit) {
    float g[II], u[II], h[D];
    matmul_i4_grouped(g, x, e->g, e->gs, 1, D, II, GS);
    matmul_i4_grouped(u, x, e->u, e->us, 1, D, II, GS);
    for (int i = 0; i < II; i++) {
        float v = fminf(g[i], limit);
        float uv = fmaxf(-limit, fminf(u[i], limit));
        g[i] = (v / (1.0f + expf(-v))) * uv;
    }
    matmul_i4_grouped(h, g, e->d, e->ds, 1, II, D, GS);
    memcpy(y, h, sizeof(h));
}

int main(void) {
    const float limit = 10.0f;
    Expert ex[NB] = {0};
    if (!make_expert(&ex[0], 3) || !make_expert(&ex[1], 19)) {
        fprintf(stderr, "expert slab allocation failed\n");
        return 2;
    }
    if (!coli_metal_init() || !coli_metal_available()) {
        fprintf(stderr, "Metal unavailable\n");
        return 77;
    }
    for (int e = 0; e < NB; e++) coli_metal_register(ex[e].base, SLAB);

    const void *g[NB] = { ex[0].g, ex[1].g };
    const void *u[NB] = { ex[0].u, ex[1].u };
    const void *d[NB] = { ex[0].d, ex[1].d };
    const float *gs[NB] = { ex[0].gs, ex[1].gs };
    const float *us[NB] = { ex[0].us, ex[1].us };
    const float *ds[NB] = { ex[0].ds, ex[1].ds };

    /* Packed rows: expert 0 handles rows 0/2, expert 1 handles rows 1/2. */
    const int nr[NB] = { 2, 2 };
    const int xoff[NB] = { 0, 2 };
    const int rows[4] = { 0, 2, 1, 2 };
    const float rw[4] = { 0.75f, 0.20f, 0.55f, 0.35f };
    float xg[4 * D];
    for (int r = 0; r < 4; r++)
        for (int i = 0; i < D; i++)
            xg[(size_t)r * D + i] = ((float)(((r + 1) * 29 + i * 7) % 41) - 20.0f) / 9.0f;

    float ref[S * D]; memset(ref, 0, sizeof(ref));
    float got[S * D]; memset(got, 0, sizeof(got));
    int roff = 0;
    for (int e = 0; e < NB; e++) {
        for (int j = 0; j < nr[e]; j++) {
            float h[D];
            cpu_one(&ex[e], xg + (size_t)(xoff[e] + j) * D, h, limit);
            float *dst = ref + (size_t)rows[roff + j] * D;
            for (int k = 0; k < D; k++) dst[k] += rw[roff + j] * h[k];
        }
        roff += nr[e];
    }

    int ok = coli_metal_moe_block_clamped(NB, D, II, 4, GS,
        g, u, d, gs, us, ds, xg, xoff, nr, rows, rw, got, S, limit);
    if (!ok) {
        fprintf(stderr, "Metal routed MoE block returned fallback\n");
        return 3;
    }

    double max_abs = 0.0, max_rel = 0.0;
    int worst = -1;
    for (int i = 0; i < S * D; i++) {
        double ae = fabs((double)got[i] - (double)ref[i]);
        double re = ae / (1.0 + fabs((double)ref[i]));
        if (ae > max_abs) { max_abs = ae; worst = i; }
        if (re > max_rel) max_rel = re;
    }
    printf("GLM53 routed Metal MoE oracle: max_abs=%.9g max_rel=%.9g worst=%d\n",
           max_abs, max_rel, worst);

    const double abs_tol = 2e-4, rel_tol = 2e-4;
    int pass = (max_abs <= abs_tol || max_rel <= rel_tol);
    printf("GLM53 routed Metal MoE oracle: %s\n", pass ? "OK" : "FAIL");

    for (int e = 0; e < NB; e++) {
        coli_metal_unregister(ex[e].base);
        free(ex[e].base);
    }
    coli_metal_shutdown();
    return pass ? 0 : 4;
}
