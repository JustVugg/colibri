/* bench_expert_decode: the routed experts of one decode forward, per weight format.
 *
 * Qwen3.8 Flash Next's shapes by default (hidden 2560, expert width 640, 10 experts
 * a token, 48 layers): every forward reads layers x topk DISTINCT experts, so the
 * weights come from DRAM as in a real decode, not from a cache a small matrix would
 * sit in. Per layer: the S activation rows quantized once, gate and up for every
 * expert in one parallel region, swiglu, the hidden rows quantized, down in a second
 * region. Reports ms per forward and the bytes it streamed per second, beside the
 * machine's plain read bandwidth over the same buffers (the floor for that format).
 *
 *   bench_expert_decode [format|all] [S] [forwards]
 *   formats: q2_0 q4_k q3_k q5_k q6_k q8_0 q2_k iq2_xxs iq2_s iq3_xxs iq3_s iq4_xs
 *            iq1_m mxfp4 q4_0 vq2 vq1 int3g16 int2g16 int2g32 int4g64 int4g64i8
 * Weights are random bytes with sane scales: the kernels' cost does not depend on
 * the values. Build: make tests/bench_expert_decode (-O3 -march=native -fopenmp). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include "../ggml_quants.h"
#include "../vq2_g32.h"
#include "../vq1_g64.h"
#include "../lowbit_g16.h"
#include "../expert_ffn.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int H = 2560, F = 640, K = 10, L = 48;

enum { FMT_GQ, FMT_VQ2, FMT_VQ1, FMT_XL3, FMT_XL2, FMT_X2, FMT_I4, FMT_I4I8 };
typedef struct { const char *name; int kind, gq; } Fmt;
static const Fmt FMTS[] = {
    {"q2_0", FMT_GQ, GQ_Q2_0}, {"q4_k", FMT_GQ, GQ_Q4_K}, {"q3_k", FMT_GQ, GQ_Q3_K}, {"q5_k", FMT_GQ, GQ_Q5_K},
    {"q6_k", FMT_GQ, GQ_Q6_K}, {"q8_0", FMT_GQ, GQ_Q8_0}, {"q2_k", FMT_GQ, GQ_Q2_K}, {"iq2_xxs", FMT_GQ, GQ_IQ2_XXS},
    {"iq2_s", FMT_GQ, GQ_IQ2_S}, {"iq3_xxs", FMT_GQ, GQ_IQ3_XXS}, {"iq3_s", FMT_GQ, GQ_IQ3_S}, {"iq4_xs", FMT_GQ, GQ_IQ4_XS},
    {"iq1_m", FMT_GQ, GQ_IQ1_M}, {"mxfp4", FMT_GQ, GQ_MXFP4}, {"q4_0", FMT_GQ, GQ_Q4_0},
    {"vq2", FMT_VQ2, 0}, {"vq1", FMT_VQ1, 0}, {"int3g16", FMT_XL3, 0}, {"int2g16", FMT_XL2, 0},
    {"int2g32", FMT_X2, 0}, {"int4g64", FMT_I4, 0}, {"int4g64i8", FMT_I4I8, 0},
};

/* one matrix of O rows x I inputs in a format: codes, per-row aux (scales), d */
typedef struct { uint8_t *w; uint8_t *q; float *d; size_t wb, qb; int gq; } Mat;
typedef struct { Mat gate, up, down; } Exp;

static uint64_t rng = 88172645463325252ULL;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }
static void fill(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (uint8_t)rnd(); }

/* a 256-block type cannot hold a 640-wide row: the GGUF quantizers store such a
 * tensor (ffn_down of these experts) in a 32-block type, Q8_0 here */
static int gq_for(const Fmt *f, int I) { return I % GQ_TYPES[f->gq].blck ? GQ_Q8_0 : f->gq; }
static size_t row_bytes(const Fmt *f, int I, size_t *aux) {
    *aux = 0;
    switch (f->kind) {
    case FMT_GQ:  return (size_t)gq_row_bytes(gq_for(f, I), I);
    case FMT_VQ2: *aux = vq2_scale_bytes(I); return vq2_code_bytes(I);
    case FMT_VQ1: *aux = vq1_scale_bytes(I); return vq1_code_bytes(I);
    case FMT_XL3: *aux = xl_scale_bytes(I); return xl_code_bytes(3, I);
    case FMT_XL2: *aux = xl_scale_bytes(I); return xl_code_bytes(2, I);
    case FMT_X2:  *aux = (size_t)x2_groups(I); return x2_code_bytes(I);
    default:      *aux = (size_t)xf_groups(I) * sizeof(float); return xf_row_bytes(I);
    }
}
static void mat_make(Mat *m, const Fmt *f, int O, int I) {
    size_t aux; size_t rb = row_bytes(f, I, &aux);
    m->wb = rb; m->qb = aux; m->gq = f->kind == FMT_GQ ? gq_for(f, I) : 0;
    m->w = aligned_alloc(64, ((size_t)O * rb + 63) / 64 * 64); fill(m->w, (size_t)O * rb);
    m->q = aux ? aligned_alloc(64, ((size_t)O * aux + 63) / 64 * 64) : NULL;
    if (m->q) {
        if (f->kind == FMT_I4 || f->kind == FMT_I4I8) { float *s = (float *)m->q; for (size_t i = 0; i < (size_t)O * aux / 4; i++) s[i] = 0.01f; }
        else fill(m->q, (size_t)O * aux);
    }
    m->d = malloc((size_t)O * sizeof(float)); for (int o = 0; o < O; o++) m->d[o] = 0.001f;
    if (f->kind == FMT_GQ) {   /* sane f16 scales at the head of every block (all but IQ1_M, MXFP4) */
        int t = m->gq, bs = GQ_TYPES[t].bytes;
        if (t != GQ_MXFP4 && t != GQ_IQ1_M)
            for (size_t o = 0; o < (size_t)O; o++) for (size_t b = 0; b < rb; b += bs) { m->w[o * rb + b] = 0x00; m->w[o * rb + b + 1] = 0x20; }
    }
}
static size_t mat_bytes(const Mat *m, int O) { return (size_t)O * (m->wb + m->qb); }

static Vq2Books vq2b; static Vq1Books vq1b;

/* rows [0,O) of y[s*O+o] = W x, S rows, activations prepared by the caller */
typedef struct { gq_act ga; int8_t *xq; float *dx; float *xsum; float sx[4]; const float *x; } Act;
static void act_prep(Act *a, const Fmt *f, const float *x, int S, int I) {
    a->x = x;
    if (f->kind == FMT_GQ) { gq_act_set(&a->ga, x, S, I); return; }
    if (f->kind == FMT_I4) return;
    if (f->kind == FMT_I4I8) { for (int s = 0; s < S; s++) a->sx[s] = xf_act_i8(x + (size_t)s * I, I, a->xq + (size_t)s * I, a->xsum + (size_t)s * (I / 64)); return; }
    for (int s = 0; s < S; s++) x2_quant_act(x + (size_t)s * I, I, a->xq + (size_t)s * I, a->dx + (size_t)s * (I / 32));
}
static inline void row_dot(float *y, int ystride, const Fmt *f, const Mat *m, int o, const Act *a, int S, int I) {
    const uint8_t *w = m->w + (size_t)o * m->wb; const uint8_t *q = m->q ? m->q + (size_t)o * m->qb : NULL;
    switch (f->kind) {
    case FMT_GQ:  gq_row_all(y, ystride, &a->ga, w, m->gq, I); break;
    case FMT_VQ2: vq2_dot(y, ystride, &vq2b, w, q, m->d[o], a->xq, a->dx, I, S); break;
    case FMT_VQ1: vq1_dot(y, ystride, &vq1b, w, q, m->d[o], a->xq, a->dx, I, S); break;
    case FMT_XL3: xl_dot(y, ystride, w, q, m->d[o], 3, a->xq, a->dx, I, S); break;
    case FMT_XL2: xl_dot(y, ystride, w, q, m->d[o], 2, a->xq, a->dx, I, S); break;
    case FMT_X2:  x2_dot(y, ystride, w, q, m->d[o], a->xq, a->dx, I, S); break;
    case FMT_I4:  { const float *xr[4]; for (int s = 0; s < S; s++) xr[s] = a->x + (size_t)s * I;
                    xf_dot_f32_rows(y, ystride, w, (const float *)q, xr, I, S); } break;
    case FMT_I4I8: for (int s = 0; s < S; s++)
                      y[(size_t)s * ystride] = xf_dot_i8(w, (const float *)q, a->xq + (size_t)s * I, a->xsum + (size_t)s * (I / 64), a->sx[s], I);
                  break;
    }
}

/* the AVX-512 kernels against the AVX2 ones they must reproduce, bit for bit:
 * random rows and activations, 1..7 activation rows */
static int check_exact(void) {
    int bad = 0, tried = 0;
#ifdef GQ_VNNI512
    for (int I = 64; I <= 2560; I += I < 640 ? 64 : 640) for (int S = 1; S <= 7; S++) {
        float *x = malloc((size_t)S * I * 4); for (int i = 0; i < S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
        gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, I);
        int64_t rb = gq_row_bytes(GQ_Q2_0, I); uint8_t *row = malloc((size_t)rb);
        for (int r = 0; r < 64; r++) {
            fill(row, (size_t)rb); for (int64_t b = 0; b < rb; b += 18) { row[b] = (uint8_t)rnd(); row[b + 1] = (uint8_t)(rnd() % 0x3c); }
            float y1[8], y2[8];
            for (int s0 = 0; s0 < S; s0 += 4) gq_dot_q2_0(y1 + s0, 1, row, a.q + (int64_t)s0 * I, a.d + (int64_t)s0 * (I / 32), I, S - s0 < 4 ? S - s0 : 4);
            gq_dot_q2_0_512(y2, 1, row, &a, I);
            tried++; if (memcmp(y1, y2, (size_t)S * 4)) { bad++; if (bad < 4) printf("q2_0 mismatch I=%d S=%d: %.9g vs %.9g\n", I, S, y1[0], y2[0]); }
        }
        free(row); free(x); gq_act_free(&a);
    }
    printf("exact check q2_0: %d of %d rows differ\n", bad, tried);
    /* every other type with an AVX-512 kernel against the integer reference (unpack, gq_dot_i8;
     * Q8_0's own dot), which the AVX2 fused kernels reproduce bit for bit */
    static const int types[] = { GQ_Q8_0, GQ_Q4_0, GQ_Q4_1, GQ_Q5_0, GQ_Q5_1, GQ_IQ4_NL, GQ_MXFP4, GQ_Q4_K, GQ_Q5_K, GQ_Q6_K, GQ_Q3_K,
                                 GQ_IQ4_XS, GQ_IQ2_XXS, GQ_IQ2_XS, GQ_IQ2_S, GQ_IQ3_XXS, GQ_IQ3_S, GQ_IQ1_S, GQ_IQ1_M };
    for (size_t k = 0; k < sizeof types / sizeof *types; k++) {
        const int t = types[k]; int tbad = 0, ttried = 0;
        for (int I = 256; I <= 2560; I = I == 256 ? 768 : I == 768 ? 2560 : 9999) for (int S = 1; S <= 7; S++) {
            if (I % GQ_TYPES[t].blck) continue;
            float *x = malloc((size_t)S * I * 4); for (int i = 0; i < S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
            gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, I);
            int64_t rb = gq_row_bytes(t, I); uint8_t *row = malloc((size_t)rb);
            int8_t *w8 = malloc((size_t)I); float *sw = malloc((size_t)(I / 16 + 1) * 4), *mw = malloc((size_t)(I / 16 + 1) * 4);
            for (int r = 0; r < 24; r++) {
                fill(row, (size_t)rb);
                for (int64_t i = 1; i < rb; i += 2) row[i] &= 0xBF;   /* every f16 field finite (|v| < 2) */
                if (t == GQ_MXFP4) for (int64_t b = 0; b < rb; b += GQ_TYPES[t].bytes) row[b] = (uint8_t)(120 + rnd() % 16);
                float y1[8], y2[8];
                if (t == GQ_Q8_0) for (int s2 = 0; s2 < S; s2++) y1[s2] = gq_dot_q8_0(row, a.q + (int64_t)s2 * I, a.d + (int64_t)s2 * (I / 32), I);
                else {
                    gq_unpack_row(t, row, I, w8, sw, mw);
                    for (int s2 = 0; s2 < S; s2++)
                        y1[s2] = gq_dot_i8(w8, sw, mw, gq_has_min(t), a.q + (int64_t)s2 * I, a.d + (int64_t)s2 * (I / 32), a.s + (int64_t)s2 * (I / 16), I);
                }
                if (!gq_row_512(y2, 1, &a, row, t, I)) { printf("%s: no AVX-512 kernel at I=%d\n", gq_type_name(t), I); break; }
                ttried++;
                if (memcmp(y1, y2, (size_t)S * 4)) { tbad++; if (tbad < 3) printf("%s mismatch I=%d S=%d: %.9g vs %.9g\n", gq_type_name(t), I, S, y1[0], y2[0]); }
            }
            free(row); free(x); free(w8); free(sw); free(mw); gq_act_free(&a);
        }
        printf("exact check %-8s %d of %d rows differ\n", gq_type_name(t), tbad, ttried);
        bad += tbad; tried += ttried;
    }
#else
    printf("exact check: no AVX-512 VNNI/VBMI kernels in this build\n");
#endif
    /* our formats: the dispatchers (the AVX-512 kernels where built) against the scalar references */
    const char *names[5] = { "int2g32", "int2g16", "int3g16", "vq2", "vq1" };
    for (int f = 0; f < 5; f++) {
        int fbad = 0, ftried = 0;
        for (int I = 128; I <= 2560; I = I == 128 ? 640 : I == 640 ? 2560 : 9999) for (int S = 1; S <= 4; S++) {
            int8_t *xq = malloc((size_t)S * I); float *dx = malloc((size_t)S * (I / 32) * 4);
            for (int i = 0; i < S * I; i++) xq[i] = (int8_t)((int)(rnd() % 255) - 127);
            for (int i = 0; i < S * (I / 32); i++) dx[i] = (float)(rnd() % 1000 + 1) * 1e-4f;
            size_t cb = f == 0 ? x2_code_bytes(I) : f == 1 ? xl_code_bytes(2, I) : f == 2 ? xl_code_bytes(3, I) : f == 3 ? vq2_code_bytes(I) : vq1_code_bytes(I);
            size_t qb = f == 0 ? (size_t)x2_groups(I) : f <= 2 ? xl_scale_bytes(I) : f == 3 ? vq2_scale_bytes(I) : vq1_scale_bytes(I);
            uint8_t *codes = malloc(cb), *q = malloc(qb);
            for (int r = 0; r < 24; r++) {
                fill(codes, cb); fill(q, qb);
                const float d = (float)(rnd() % 1000 + 1) * 1e-3f;
                float y1[4], y2[4];
                for (int s2 = 0; s2 < S; s2++) {
                    const int8_t *xs = xq + (size_t)s2 * I; const float *ds = dx + (size_t)s2 * (I / 32);
                    y1[s2] = f == 0 ? x2_dot_ref(codes, q, d, xs, ds, I) : f == 1 ? xl_dot_ref(codes, q, d, 2, xs, ds, I)
                           : f == 2 ? xl_dot_ref(codes, q, d, 3, xs, ds, I) : f == 3 ? vq2_dot_ref(&vq2b, codes, q, d, xs, ds, I)
                           : vq1_dot_ref(&vq1b, codes, q, d, xs, ds, I);
                }
                if (f == 0) x2_dot(y2, 1, codes, q, d, xq, dx, I, S);
                else if (f <= 2) xl_dot(y2, 1, codes, q, d, f == 1 ? 2 : 3, xq, dx, I, S);
                else if (f == 3) vq2_dot(y2, 1, &vq2b, codes, q, d, xq, dx, I, S);
                else vq1_dot(y2, 1, &vq1b, codes, q, d, xq, dx, I, S);
                ftried++;
                if (memcmp(y1, y2, (size_t)S * 4)) { fbad++; if (fbad < 3) printf("%s mismatch I=%d S=%d: %.9g vs %.9g\n", names[f], I, S, y1[0], y2[0]); }
            }
            free(xq); free(dx); free(codes); free(q);
        }
        printf("exact check %-8s %d of %d rows differ (against the scalar reference)\n", names[f], fbad, ftried);
        bad += fbad; tried += ftried;
    }
    /* the vectorized activation quantizers against their scalar references, byte for byte */
    {
        int abad = 0, atried = 0;
        for (int r = 0; r < 2000; r++) {
            const int I = 32 * (1 + (int)(rnd() % 80)); float *x = malloc((size_t)I * 4);
            const float mag = r % 7 == 0 ? 0.f : (float)(rnd() % 100000 + 1) * 1e-3f;
            for (int i = 0; i < I; i++) x[i] = ((int)(rnd() % 20001) - 10000) * 1e-4f * mag;
            int8_t *q1 = malloc(I), *q2 = malloc(I); float *d1 = malloc(I / 32 * 4), *d2 = malloc(I / 32 * 4);
            int32_t *s1 = malloc(I / 16 * 4), *s2 = malloc(I / 16 * 4), *l1 = malloc(I / 4 * 4), *l2 = malloc(I / 4 * 4);
            gq_quant_act_ref(x, I, q1, d1, s1); gq_quant_act(x, I, q2, d2, s2);
            for (int k = 0; k < I / 4; k++) l1[k] = (int32_t)q1[4 * k] + q1[4 * k + 1] + q1[4 * k + 2] + q1[4 * k + 3];
            gq_lane_sums(q2, I, l2);
            int bad1 = memcmp(q1, q2, I) || memcmp(d1, d2, I / 32 * 4) || memcmp(s1, s2, I / 16 * 4) || memcmp(l1, l2, I / 4 * 4);
            x2_quant_act_ref(x, I, q1, d1); x2_quant_act(x, I, q2, d2);
            bad1 |= memcmp(q1, q2, I) || memcmp(d1, d2, I / 32 * 4);
            atried++; abad += bad1;
            free(x); free(q1); free(q2); free(d1); free(d2); free(s1); free(s2); free(l1); free(l2);
        }
        printf("exact check activation quantizers: %d of %d rows differ\n", abad, atried);
        bad += abad; tried += atried;
    }
    /* int4-g64 f32: the multi-row kernel against xf_dot_f32 row by row */
    {
        int ibad = 0, itried = 0;
        for (int I = 64; I <= 2560; I = I == 64 ? 640 : I == 640 ? 2560 : 9999) for (int S = 1; S <= 7; S++) {
            float *x = malloc((size_t)S * I * 4); for (int i = 0; i < S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
            const float *xr[8]; for (int s2 = 0; s2 < S; s2++) xr[s2] = x + (size_t)s2 * I;
            uint8_t *w = malloc(xf_row_bytes(I)); float *sc = malloc((size_t)xf_groups(I) * 4);
            for (int r = 0; r < 24; r++) {
                fill(w, xf_row_bytes(I)); for (int g = 0; g < xf_groups(I); g++) sc[g] = (float)(rnd() % 1000 + 1) * 1e-4f;
                float y1[8], y2[8];
                for (int s2 = 0; s2 < S; s2++) y1[s2] = xf_dot_f32(w, sc, xr[s2], I);
                xf_dot_f32_rows(y2, 1, w, sc, xr, I, S);
                itried++; if (memcmp(y1, y2, (size_t)S * 4)) { ibad++; if (ibad < 3) printf("int4 f32 mismatch I=%d S=%d\n", I, S); }
            }
            free(x); free(w); free(sc);
        }
        printf("exact check int4g64 f32 rows: %d of %d rows differ (against xf_dot_f32)\n", ibad, itried);
        bad += ibad; tried += itried;
    }
    return bad;
}

/* gq_dot_i8's AVX2 arithmetic in plain C: exact lane sums (elements 4l..4l+3 of a block),
 * fmaf per lane with sw * dx, GQ_HSUM8's order, the min term; Q8_0 with its block scale.
 * Built without any SIMD dot, so it checks every build's kernels, AVX-VNNI ones included. */
static float ref_dot(int t, const uint8_t *row, const int8_t *xq, const float *dx, const int32_t *xs, int I,
                     int8_t *w8, float *sw, float *mw) {
    float acc[8] = {0};
    if (t == GQ_Q8_0) {
        for (int b = 0; b < I / 32; b++) {
            const uint8_t *blk = row + b * 34; const int8_t *w = (const int8_t *)(blk + 2); const float sc = gq_f16(gq_rd16(blk)) * dx[b];
            for (int l = 0; l < 8; l++) { int32_t p = 0; for (int i = 4 * l; i < 4 * l + 4; i++) p += (int32_t)w[i] * xq[32 * b + i]; acc[l] = fmaf((float)p, sc, acc[l]); }
        }
        return x2_hsum8_scalar(acc);
    }
    gq_unpack_row(t, row, I, w8, sw, mw);
    for (int b = 0; b < I / 32; b++)
        for (int l = 0; l < 8; l++) {
            int32_t p = 0; for (int i = 4 * l; i < 4 * l + 4; i++) p += (int32_t)w8[32 * b + i] * xq[32 * b + i];
            acc[l] = fmaf((float)p, sw[2 * b + (l >= 4)] * dx[b], acc[l]);
        }
    float r = x2_hsum8_scalar(acc);
    if (gq_has_min(t)) {
        float m = 0.f;
        for (int b = 0; b < I / 32; b++) m = fmaf(dx[b], fmaf(mw[2 * b], (float)xs[2 * b], mw[2 * b + 1] * (float)xs[2 * b + 1]), m);
        r -= m;
    }
    return r;
}
static int check_portable(void) {
    static const int types[] = { GQ_Q2_0, GQ_Q8_0, GQ_Q4_0, GQ_Q4_1, GQ_Q5_0, GQ_Q5_1, GQ_IQ4_NL, GQ_MXFP4, GQ_Q4_K, GQ_Q5_K, GQ_Q6_K, GQ_Q3_K,
                                 GQ_IQ4_XS, GQ_IQ2_XXS, GQ_IQ2_XS, GQ_IQ2_S, GQ_IQ3_XXS, GQ_IQ3_S, GQ_IQ1_S, GQ_IQ1_M, GQ_Q2_K };
    int bad = 0, tried = 0;
    for (size_t k = 0; k < sizeof types / sizeof *types; k++) {
        const int t = types[k]; int tbad = 0, ttried = 0;
        for (int I = 256; I <= 2560; I = I == 256 ? 768 : I == 768 ? 2560 : 9999) for (int S = 1; S <= 6; S++) {
            if (I % GQ_TYPES[t].blck) continue;
            float *x = malloc((size_t)S * I * 4); for (int i = 0; i < S * I; i++) x[i] = ((int)(rnd() % 2001) - 1000) * 1e-3f;
            gq_act a; memset(&a, 0, sizeof a); gq_act_set(&a, x, S, I);
            int64_t rb = gq_row_bytes(t, I); uint8_t *row = malloc((size_t)rb);
            int8_t *w8 = malloc((size_t)I); float *sw = malloc((size_t)(I / 16 + 1) * 4), *mw = malloc((size_t)(I / 16 + 1) * 4);
            for (int r = 0; r < 12; r++) {
                fill(row, (size_t)rb);
                for (int64_t i = 1; i < rb; i += 2) row[i] &= 0xBF;
                if (t == GQ_MXFP4) for (int64_t b = 0; b < rb; b += GQ_TYPES[t].bytes) row[b] = (uint8_t)(120 + rnd() % 16);
                float y1[8], y2[8];
                for (int s2 = 0; s2 < S; s2++) y1[s2] = ref_dot(t, row, a.q + (int64_t)s2 * I, a.d + (int64_t)s2 * (I / 32), a.s + (int64_t)s2 * (I / 16), I, w8, sw, mw);
                gq_row_all(y2, 1, &a, row, t, I);
                ttried++;
                if (memcmp(y1, y2, (size_t)S * 4)) { tbad++; if (tbad < 3) printf("%s mismatch I=%d S=%d: %.9g vs %.9g\n", gq_type_name(t), I, S, y1[0], y2[0]); }
            }
            free(row); free(x); free(w8); free(sw); free(mw); gq_act_free(&a);
        }
        printf("portable check %-8s %d of %d rows differ\n", gq_type_name(t), tbad, ttried);
        bad += tbad; tried += ttried;
    }
    return bad;
}

int main(int argc, char **argv) {
    const char *want = argc > 1 ? argv[1] : "all";

    int S = argc > 2 ? atoi(argv[2]) : 1, reps = argc > 3 ? atoi(argv[3]) : 6;
    if (S < 1 || S > 4) { fprintf(stderr, "S in 1..4\n"); return 2; }
    if (getenv("BENCH_LAYERS")) L = atoi(getenv("BENCH_LAYERS"));
    fill((uint8_t *)&vq2b, sizeof vq2b); fill((uint8_t *)&vq1b, sizeof vq1b);
    for (int i = 0; i < VQ2_BOOK; i++) for (int j = 0; j < 8; j++) { vq2b.a[i][j] /= 2; vq2b.b[i][j] /= 2; }
    for (int i = 0; i < VQ1_BOOK; i++) for (int j = 0; j < 16; j++) { vq1b.a[i][j] /= 2; vq1b.b[i][j] /= 2; }
    if (!strcmp(want, "check")) return check_exact() ? 1 : 0;
    if (!strcmp(want, "check2")) return check_portable() ? 1 : 0;
    float *x = malloc((size_t)S * H * 4), *h = malloc((size_t)K * S * F * 4);
    float *g = malloc((size_t)K * S * F * 4), *u = malloc((size_t)K * S * F * 4), *y = malloc((size_t)K * S * H * 4);
    for (int i = 0; i < S * H; i++) x[i] = sinf((float)i * 0.37f);
    printf("%d threads, H=%d F=%d K=%d L=%d, S=%d, %d forwards\n", omp_get_max_threads(), H, F, K, L, S, reps);
    for (size_t k = 0; k < sizeof FMTS / sizeof *FMTS; k++) {
        const Fmt *f = &FMTS[k];
        if (strcmp(want, "all") && strcmp(want, f->name)) continue;
        int n = L * K; Exp *e = malloc((size_t)n * sizeof *e); size_t bytes = 0;
        for (int i = 0; i < n; i++) {
            mat_make(&e[i].gate, f, F, H); mat_make(&e[i].up, f, F, H); mat_make(&e[i].down, f, H, F);
            bytes += mat_bytes(&e[i].gate, F) + mat_bytes(&e[i].up, F) + mat_bytes(&e[i].down, H);
        }
        Act ax, ah[16]; memset(&ax, 0, sizeof ax); memset(ah, 0, sizeof ah);
        ax.xq = malloc((size_t)S * H); ax.dx = malloc((size_t)S * H / 32 * 4); ax.xsum = malloc((size_t)S * H / 64 * 4);
        for (int j = 0; j < K; j++) { ah[j].xq = malloc((size_t)S * F); ah[j].dx = malloc((size_t)S * F / 32 * 4); ah[j].xsum = malloc((size_t)S * F / 64 * 4); }
        double best = 1e9, sum = 0, glue = 0, best_glue = 0;
        for (int r = 0; r <= reps; r++) {   /* forward 0 warms up */
            double t0 = now(); glue = 0;
            for (int l = 0; l < L; l++) {
                const Exp *le = e + (size_t)l * K;
                act_prep(&ax, f, x, S, H);
                #pragma omp parallel for schedule(static)
                for (int it = 0; it < K * F * 2; it++) {
                    int j = it / (2 * F), rr = it % (2 * F), up = rr >= F, o = up ? rr - F : rr;
                    row_dot((up ? u : g) + (size_t)j * S * F + o, F, f, up ? &le[j].up : &le[j].gate, o, &ax, S, H);
                }
                double tk = now();
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < K; j++) {   /* the engine runs swiglu and the hidden rows' quantization in parallel too */
                    for (int i = 0; i < S * F; i++) { float gv = g[(size_t)j * S * F + i]; h[(size_t)j * S * F + i] = gv / (1.f + expf(-gv)) * u[(size_t)j * S * F + i]; }
                    act_prep(&ah[j], f, h + (size_t)j * S * F, S, F);
                }
                glue += now() - tk;
                #pragma omp parallel for schedule(static)
                for (int it = 0; it < K * H; it++) {
                    int j = it / H, o = it % H;
                    row_dot(y + (size_t)j * S * H + o, H, f, &le[j].down, o, &ah[j], S, F);
                }
            }
            double dt = now() - t0;
            if (r) { sum += dt; if (dt < best) { best = dt; best_glue = glue; } }
        }
        /* the floor: one read of the same bytes, every thread */
        double rbest = 1e9;
        for (int r = 0; r < 3; r++) {
            double t0 = now(); uint64_t acc = 0;
            #pragma omp parallel for schedule(static) reduction(^:acc)
            for (int i = 0; i < n; i++) {
                const Mat *ms[3] = { &e[i].gate, &e[i].up, &e[i].down }; const int os[3] = { F, F, H };
                for (int m = 0; m < 3; m++) {
                    const uint64_t *p = (const uint64_t *)ms[m]->w; size_t c = (size_t)os[m] * ms[m]->wb / 8;
                    for (size_t z = 0; z < c; z++) acc ^= p[z];
                }
            }
            double dt = now() - t0; if (dt < rbest) rbest = dt;
            if (acc == 42) printf(" ");
        }
        if (f->kind == FMT_GQ && gq_for(f, F) != f->gq) printf("(down in Q8_0) ");
        printf("%-10s %7.1f MB/forward  best %7.2f ms (swiglu+quant %5.2f)  mean %7.2f ms  %6.1f GB/s | read floor %6.2f ms (%5.1f GB/s)  %4.2fx floor\n",
               f->name, bytes / 1e6, best * 1e3, best_glue * 1e3, sum / reps * 1e3, bytes / best / 1e9, rbest * 1e3, bytes / rbest / 1e9, best / rbest);
        fflush(stdout);
        for (int i = 0; i < n; i++) { Mat *ms[3] = { &e[i].gate, &e[i].up, &e[i].down }; for (int m = 0; m < 3; m++) { free(ms[m]->w); free(ms[m]->q); free(ms[m]->d); } }
        free(e); free(ax.xq); free(ax.dx); free(ax.xsum);
        for (int j = 0; j < K; j++) { free(ah[j].xq); free(ah[j].dx); free(ah[j].xsum); if (f->kind == FMT_GQ) gq_act_free(&ah[j].ga); }
    }
    return 0;
}
