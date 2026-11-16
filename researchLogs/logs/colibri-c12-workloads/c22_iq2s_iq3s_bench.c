/* c22_iq2s_iq3s_bench.c -- profile of the IQ2_S/IQ3_S expert path on ARM and the
 * decode-once candidate (the C22 proposal: extend C18's Q36_EXPERT_DEC1 family to
 * the two types unsloth's UD-IQ3_XXS actually ships).
 *
 * Today (ARM): gq_row_all falls through to the generic path -- gq_unpack_row once
 * per weight row into the thread buffer, then gq_dot_i8 once per activation row.
 * Candidate: the same unpack (once per row, reused verbatim -- no new SB decoders)
 * into stack scratch, then gq_dot_i8's NEON float steps inlined per activation row
 * (per 32-block: two 16-lane vdot products, vmlaq with sw[2b]*d then sw[2b+1]*d,
 * vaddvq at the end; no has_min work -- gq_has_min is 0 for both types, so the
 * candidate's bits are the current path's).
 *
 * Shapes are the IQ3 file's: gate/up I=2048 O=512 (IQ2_S), down I=512 O=2048 (IQ3_S).
 * S sweeps the C14-measured per-expert row counts.
 *
 * Build (from c/):
 *   clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
 *     ../researchLogs/logs/colibri-c12-workloads/c22_iq2s_iq3s_bench.c -o /tmp/c22bench \
 *     -lm -L/opt/homebrew/opt/libomp/lib -lomp
 * Run twice: OMP_NUM_THREADS=1 (per-thread rates) and =6 (the serving config).
 */
#include "ggml_quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static inline double frand(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (double)(int64_t)(rs >> 40) / 8388608.0 - 1.0;
}

typedef struct {
    int t, S, I, O;
    const uint8_t *W;
    float *x, *y;
    gq_act a;
} Ctx;

/* ---- phase 2: the fused register-resident decoders (the actual C22 proposal) ----
 * Same structure as the C18 SB decoders (gq_arm_sb_iq2_xxs/iq3_xxs): per 32-element
 * sub-block, table gathers + sign expansion in registers, joining gq_arm_dot2 at once.
 * Sign bits are raw bytes here (not the XXS 7-bit table), so a 256x8 +-1 table plays
 * the role of GQ_SIGN8: one load per 8 elements. */

static int8_t C22_SIGNS8[256][8];

/* IQ2_S sub-block xb (32 elements): grid entries k = 4*xb..4*xb+3, 8 elements each.
 * Layout: d@0 qs[32]@2 sg[32]@34 qh[8]@66 sc[8]@74 (82 bytes). Scales per 16:
 * sw[2*xb] = sc[xb]&0xF, sw[2*xb+1] = sc[xb]>>4, x d*(0.5+s)*0.25. */
static inline void c22_sb_iq2_s(const uint8_t *b, float d, int xb, float *s2, int8x16_t *w0, int8x16_t *w1) {
    const uint8_t *qs = b + 2, *sg = b + 34, *qh = b + 66, *sc = b + 74;
    s2[0] = d * (0.5f + (float)(sc[xb] & 0x0F)) * 0.25f;
    s2[1] = d * (0.5f + (float)(sc[xb] >> 4)) * 0.25f;
    int8x8_t g[4];
    for (int l = 0; l < 4; l++) {
        const int k = 4 * xb + l;
        const int idx = qs[k] | (((qh[k >> 2] >> (2 * (k & 3))) & 3) << 8);
        g[l] = vmul_s8(vreinterpret_s8_u8(vld1_u8(GGML_IQ2S_GRID + 8 * idx)),
                       vld1_s8(C22_SIGNS8[(uint8_t)sg[k]]));
    }
    *w0 = vcombine_s8(g[0], g[1]); *w1 = vcombine_s8(g[2], g[3]);
}

/* IQ3_S sub-block xb (32 elements): grid entries k = 8*xb..8*xb+7, 4 elements each.
 * Layout: d@0 qs[64]@2 qh[8]@66 sg[32]@74 sc[4]@106 (110 bytes). One scale per 32:
 * d*(1+2s), s = sc nibble xb. */
static inline void c22_sb_iq3_s(const uint8_t *b, float d, int xb, float *s2, int8x16_t *w0, int8x16_t *w1) {
    const uint8_t *qs = b + 2, *qh = b + 66, *sg = b + 74, *sc = b + 106;
    const int s = (sc[xb >> 1] >> (4 * (xb & 1))) & 0x0F;
    s2[0] = s2[1] = d * (float)(1 + 2 * s);
    int8x8_t g[4];
    for (int l = 0; l < 4; l++) {
        const int k0 = 8 * xb + 2 * l, k1 = k0 + 1;
        const int idx0 = qs[k0] | (((qh[k0 >> 3] >> (k0 & 7)) & 1) << 8);
        const int idx1 = qs[k1] | (((qh[k1 >> 3] >> (k1 & 7)) & 1) << 8);
        uint8_t gb[8];
        memcpy(gb, GGML_IQ3S_GRID + 4 * idx0, 4);
        memcpy(gb + 4, GGML_IQ3S_GRID + 4 * idx1, 4);
        g[l] = vmul_s8(vreinterpret_s8_u8(vld1_u8(gb)),
                       vld1_s8(C22_SIGNS8[(uint8_t)sg[4 * xb + l]]));
    }
    *w0 = vcombine_s8(g[0], g[1]); *w1 = vcombine_s8(g[2], g[3]);
}

/* the fused dot: per activation row, decode each sub-block in registers and dot at
 * once. Float steps are gq_dot_i8's NEON order (p0 by sw[2b]*d, p1 by sw[2b+1]*d). */
#define C22_DOT_FUSED(NAME, BYTES, SB) \
static void NAME(float *y, int ystride, const uint8_t *row, const int8_t *xq, const float *dx, int I, int S) { \
    for (int s = 0; s < S; s++) { \
        float32x4_t vacc = vdupq_n_f32(0.f); \
        const int8_t *xs8 = xq + (int64_t)s * I; const float *dxs = dx + (int64_t)s * (I / 32); \
        int xb = 0; \
        for (int64_t blk = 0; blk < I / 256; blk++) { \
            const uint8_t *b = row + blk * (BYTES); const float d = gq_f16(gq_rd16(b)); \
            for (int ib = 0; ib < 8; ib++, xb++) { \
                float s2[2]; int8x16_t w0, w1; SB(b, d, ib, s2, &w0, &w1); \
                int32x4_t p0, p1; \
                gq_arm_dot2(w0, w1, vld1q_s8(xs8 + 32 * xb), vld1q_s8(xs8 + 32 * xb + 16), &p0, &p1); \
                vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p0), s2[0] * dxs[xb]); \
                vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p1), s2[1] * dxs[xb]); \
            } \
        } \
        y[(int64_t)s * ystride] = vaddvq_f32(vacc); \
    } \
}
C22_DOT_FUSED(c22_dot_fused_iq2_s, 82, c22_sb_iq2_s)
C22_DOT_FUSED(c22_dot_fused_iq3_s, 110, c22_sb_iq3_s)

/* the fused cell, dispatching by type: XXS -> the shipped C18 kernels (calibration),
 * IQ2_S/IQ3_S -> the C22 prototypes */
static double cell_fused(int reps, void *p) {
    Ctx *c = p; const int64_t rb = gq_row_bytes(c->t, c->I); double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++) {
            const uint8_t *row = c->W + (int64_t)o * rb;
            switch (c->t) {
            case GQ_IQ2_XXS: gq_dot_arm_iq2_xxs(c->y + o, c->O, row, c->a.q, c->a.d, c->I, c->S); break;
            case GQ_IQ3_XXS: gq_dot_arm_iq3_xxs(c->y + o, c->O, row, c->a.q, c->a.d, c->I, c->S); break;
            case GQ_IQ2_S:   c22_dot_fused_iq2_s(c->y + o, c->O, row, c->a.q, c->a.d, c->I, c->S); break;
            case GQ_IQ3_S:   c22_dot_fused_iq3_s(c->y + o, c->O, row, c->a.q, c->a.d, c->I, c->S); break;
            default: fprintf(stderr, "no fused kernel\n"); exit(1);
            }
        }
    return now() - t0;
}

/* ---- the candidate kernel (a weight row against all S activation rows) ----
 * The future gq_row_all insertion: stack scratch is caller-provided here. */
static void c22_dec1_row(float *y, int ystride, const gq_act *a, const uint8_t *row,
                         int t, int I, int8_t *dw, float *ds, float *mw) {
    if (gq_unpack_row(t, row, I, dw, ds, mw)) { fprintf(stderr, "unpack failed\n"); exit(1); }
    for (int s = 0; s < a->S; s++) {
        float32x4_t vacc = vdupq_n_f32(0.f);
        const int8_t *xs8 = a->q + (int64_t)s * I;
        const float *dxs = a->d + (int64_t)s * (I / 32);
        for (int b = 0; b < I / 32; b++) {
            int32x4_t p0, p1;
            gq_arm_dot2(vld1q_s8(dw + b * 32), vld1q_s8(dw + b * 32 + 16),
                        vld1q_s8(xs8 + b * 32), vld1q_s8(xs8 + b * 32 + 16), &p0, &p1);
            const float d = dxs[b];
            vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p0), ds[2 * b] * d);
            vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p1), ds[2 * b + 1] * d);
        }
        y[(int64_t)s * ystride] = vaddvq_f32(vacc);
    }
}

/* ---- cells: each returns seconds for `reps` iterations ---- */

static double cell_matmul(int reps, void *p) {
    Ctx *c = p; double t0 = now();
    for (int r = 0; r < reps; r++)
        if (gq_matmul(c->y, c->x, c->W, c->t, c->S, c->I, c->O)) { fprintf(stderr, "gq_matmul failed\n"); exit(1); }
    return now() - t0;
}
static double cell_quant(int reps, void *p) {
    Ctx *c = p; double t0 = now();
    for (int r = 0; r < reps; r++)
        if (gq_act_set(&c->a, c->x, c->S, c->I)) exit(1);
    return now() - t0;
}
/* the engine's per-row dispatch today: gq_row_all -> unpack + gq_dot_i8 */
static double cell_rowall(int reps, void *p) {
    Ctx *c = p; const int64_t rb = gq_row_bytes(c->t, c->I); double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++)
            gq_row_all(c->y + o, c->O, &c->a, c->W + (int64_t)o * rb, c->t, c->I);
    return now() - t0;
}
/* decode-only: the unpack the current and candidate paths share */
static double cell_unpack(int reps, void *p) {
    Ctx *c = p; const int I = c->I;
    int8_t *dw = malloc((size_t)I); float *ds = malloc((size_t)(I / 16) * 2 * sizeof(float));
    float *mw = ds + I / 16;
    const int64_t rb = gq_row_bytes(c->t, I); volatile double dacc = 0;
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++)
            dacc += ds[gq_unpack_row(c->t, c->W + (int64_t)o * rb, I, dw, ds, mw) ? 0 : 0];
    free(dw); free(ds);
    return now() - t0;
}
/* the candidate, single thread */
static double cell_dec1(int reps, void *p) {
    Ctx *c = p; const int I = c->I;
    int8_t *dw = malloc((size_t)I); float *ds = malloc((size_t)(I / 16) * 2 * sizeof(float));
    float *mw = ds + I / 16;
    const int64_t rb = gq_row_bytes(c->t, I); double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++)
            c22_dec1_row(c->y + o, c->O, &c->a, c->W + (int64_t)o * rb, c->t, I, dw, ds, mw);
    free(dw); free(ds);
    return now() - t0;
}
/* the candidate under the engine's dispatch shape: omp over output rows, the
 * scratch a per-call stack VLA like the future gq_row_all insertion */
static double cell_dec1_omp(int reps, void *p) {
    Ctx *c = p; const int O = c->O, I = c->I; const int64_t rb = gq_row_bytes(c->t, I);
    double t0 = now();
    for (int r = 0; r < reps; r++) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            int8_t dw[I]; float ds[I / 16], mw[I / 16];
            c22_dec1_row(c->y + o, O, &c->a, c->W + (int64_t)o * rb, c->t, I, dw, ds, mw);
        }
    }
    return now() - t0;
}

/* run a cell with reps calibrated to ~0.15 s; returns min-of-3 batches */
static double time_cell(double (*fn)(int, void *), void *ctx) {
    double est = fn(1, ctx);
    int reps = est > 0 ? (int)(0.15 / est) : 1000;
    if (reps < 1) reps = 1;
    if (reps > 200000) reps = 200000;
    double best = 1e30;
    for (int b = 0; b < 3; b++) {
        double t = fn(reps, ctx);
        if (t / reps < best) best = t / reps;
    }
    return best;
}

static void run_shape(const char *name, int t, int I, int O, const int *Ss, int nS, int threads) {
    Ctx c; memset(&c, 0, sizeof c);
    c.t = t; c.I = I; c.O = O;
    const int64_t rb = gq_row_bytes(t, I);
    c.W = malloc((size_t)O * rb);
    uint8_t *Wm = (uint8_t *)c.W;
    for (int64_t i = 0; i < (int64_t)O * rb; i++) Wm[i] = (uint8_t)(frand() * 256.0);  /* random block bytes */
    c.x = malloc((size_t)4096 * I * sizeof(float));
    c.y = malloc((size_t)4096 * O * sizeof(float));
    for (int i = 0; i < 4096 * I; i++) c.x[i] = (float)frand();
    /* sanity: the candidate must reproduce the current path's bits (same unpack,
     * gq_dot_i8's NEON float order). NaN scales compare as "not greater", as in
     * the C18 bench's precheck. */
    if (gq_act_set(&c.a, c.x, 2, I)) exit(1);
    double maxdiff = 0;
    for (int o = 0; o < 8; o++) {
        float yk[2] = {0, 0}, yp[2] = {0, 0};
        gq_row_all(yk, 1, &c.a, c.W + (int64_t)o * rb, t, I);
        int8_t *dw = malloc((size_t)I); float *ds = malloc((size_t)(I / 16) * 2 * sizeof(float));
        c22_dec1_row(yp, 1, &c.a, c.W + (int64_t)o * rb, t, I, dw, ds, ds + I / 16);
        free(dw); free(ds);
        for (int s = 0; s < 2; s++) {
            double d = fabs((double)yk[s] - yp[s]);
            if (d > maxdiff) maxdiff = d;
        }
    }
    /* fused precheck: the register-resident decoder vs the current path */
    double fused_maxdiff = 0;
    for (int o = 0; o < 8; o++) {
        float yk[2] = {0, 0}, yp[2] = {0, 0};
        gq_row_all(yk, 1, &c.a, c.W + (int64_t)o * rb, t, I);
        const uint8_t *row = c.W + (int64_t)o * rb;
        switch (t) {
        case GQ_IQ2_XXS: gq_dot_arm_iq2_xxs(yp, 1, row, c.a.q, c.a.d, I, c.a.S); break;
        case GQ_IQ3_XXS: gq_dot_arm_iq3_xxs(yp, 1, row, c.a.q, c.a.d, I, c.a.S); break;
        case GQ_IQ2_S:   c22_dot_fused_iq2_s(yp, 1, row, c.a.q, c.a.d, I, c.a.S); break;
        case GQ_IQ3_S:   c22_dot_fused_iq3_s(yp, 1, row, c.a.q, c.a.d, I, c.a.S); break;
        }
        for (int s = 0; s < 2; s++) {
            double d = fabs((double)yk[s] - yp[s]);
            if (d > fused_maxdiff) fused_maxdiff = d;
        }
    }
    printf("{\"shape\":\"%s\",\"t\":\"%s\",\"I\":%d,\"O\":%d,\"threads\":%d,\"precheck_maxdiff\":%.3e,\"fused_precheck_maxdiff\":%.3e",
           name, gq_type_name(t), I, O, threads, maxdiff, fused_maxdiff);
    for (int i = 0; i < nS; i++) {
        int S = Ss[i];
        c.S = S;
        if (gq_act_set(&c.a, c.x, S, I)) exit(1);
        double gf = 2.0 * (double)S * I * O / 1e9;
        double t_m = time_cell(cell_matmul, &c);
        double t_q = time_cell(cell_quant, &c);
        double t_r = time_cell(cell_rowall, &c);
        double t_u = time_cell(cell_unpack, &c);
        double t_d = time_cell(cell_dec1, &c);
        double t_do = time_cell(cell_dec1_omp, &c);
        double t_f = time_cell(cell_fused, &c);
        printf(",\"%d\":{\"flops\":%.3f,\"matmul_s\":%.4f,\"matmul_gflops\":%.1f,"
               "\"quant_s\":%.5f,\"rowall_s\":%.4f,\"rowall_gflops\":%.1f,"
               "\"unpack_s\":%.4f,\"dec1_s\":%.4f,\"dec1_gflops\":%.1f,"
               "\"dec1_omp_gflops\":%.1f,\"fused_gflops\":%.1f,"
               "\"unpack_share_pct\":%.1f,\"speedup_pot\":%.2f,\"speedup_omp_pot\":%.2f,\"fused_s1_gain\":%.2f}",
               S, gf,
               t_m, gf / t_m,
               t_q, t_r, gf / t_r,
               t_u, t_d, gf / t_d,
               gf / t_do, gf / t_f,
               100.0 * t_u / t_r, t_r / t_d, t_m / t_do, S == 1 ? t_r / t_f : 0.0);
    }
    printf("}\n");
    fflush(stdout);
    free(c.W); free(c.x); free(c.y); gq_act_free(&c.a);
}

int main(void) {
    for (int b = 0; b < 256; b++)
        for (int j = 0; j < 8; j++)
            C22_SIGNS8[b][j] = (b >> j) & 1 ? -1 : 1;
    int threads = 1;
#if defined(_OPENMP)
#pragma omp parallel
    { if (omp_get_thread_num() == 0) threads = omp_get_num_threads(); }
#endif
    printf("{\"info\":\"c22_iq2s_iq3s_bench\",\"threads\":%d,\"arm_neon\":%d,\"dotprod\":%d}\n",
           threads,
#if defined(__ARM_NEON)
           1,
#else
           0,
#endif
#if defined(__ARM_FEATURE_DOTPROD)
           1
#else
           0
#endif
    );
    const int Ss[] = { 1, 4, 8, 38, 137, 411, 455 };
    run_shape("gate_up", GQ_IQ2_S, 2048, 512, Ss, 7, threads);
    run_shape("down",    GQ_IQ3_S, 512,  2048, Ss, 7, threads);
    /* C18 calibration: the shipped fused XXS kernels on the same machine state */
    run_shape("gate_up_xxs", GQ_IQ2_XXS, 2048, 512, Ss, 7, threads);
    run_shape("down_xxs",    GQ_IQ3_XXS, 512,  2048, Ss, 7, threads);
    return 0;
}
