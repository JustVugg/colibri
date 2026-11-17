/* c24_q4k_bench.c -- profile of the Q4_K expert path on ARM at S == 1 and the
 * fused register-resident candidate (the C24 proposal: extend the C22 fused
 * S==1 family to the K-quants, whose serving container is Q4_K_M).
 *
 * Today (ARM): Q4_K at any S falls through gq_row_all to the generic path --
 * gq_unpack_row once per weight row into the thread buffer, then gq_dot_i8
 * once per activation row, with the min term (gq_has_min(Q4_K) = 1) folded
 * via the activation's per-16 sums: y = sum_g dx*(sw*sum(w*xq) - mw*xs).
 * The codes are UNSIGNED nibbles 0..15; no -8 offset, the min carries it.
 *
 * Candidate: the same sub-block decoders register-resident (per 32 elements:
 * two int8x16 from one nibble split, scales d*sc[i], mins dmin*m[i] via
 * gq_scale_min_k4), dotted with gq_arm_dot2 and accumulated in gq_dot_i8's
 * exact NEON float order -- two vmlaq per sub-block (p0 by sw*d, p1 by sw*d),
 * the min term as gq_dot_i8's explicit fmaf chain, final vaddvq minus the
 * min accumulator. No dec1 cell: at S > 1 the generic path is already
 * amortized (C22 phase 1's finding transfers: same unpack+dot structure),
 * and at S == 1 dec1 IS the generic path.
 *
 * Shapes are Q4_K_M's serving shapes (same geometry as the IQ3 file):
 * gate/up I=2048 O=512, down I=512 O=2048, 144-byte blocks.
 * S sweeps the C14-measured per-expert row counts.
 * Straggler sizing (no fused candidate here, rowall/unpack only): the
 * Q2_K/Q3_K/IQ4_XS early-layer types the IQ3 file ships (1-2 layers).
 * Calibration: the shipped fused XXS kernels (ties machine state to C18/C22).
 *
 * Build (from c/):
 *   clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
 *     ../researchLogs/logs/colibri-c12-workloads/c24_q4k_bench.c -o /tmp/c24bench \
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

/* ---- the fused register-resident Q4_K decoder (the C24 candidate) ----
 * Per 256-element block (144 bytes): d@0 dmin@2 sc[12]@4 qs[128]@16.
 * Sub-block ib (32 elements): scale/min pair gq_scale_min_k4(sc, ib)
 * (sw = d*s, mw = dmin*m, one per 32), codes = one nibble half of the
 * 32 bytes qs + (ib>>1)*32 -- even ib low nibble, odd ib high nibble,
 * UNSIGNED 0..15 exactly as the NEON unpack stores them. */

static inline void c24_sb_q4_k(const uint8_t *b, float d, float dmin, int ib,
                               float *s2, float *m2, int8x16_t *w0, int8x16_t *w1) {
    const uint8_t *sc = b + 4, *qs = b + 16;
    int s, m; gq_scale_min_k4(sc, ib, &s, &m);
    s2[0] = s2[1] = d * (float)s;
    m2[0] = m2[1] = dmin * (float)m;
    const uint8_t *q = qs + (ib >> 1) * 32;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t v0 = vld1q_u8(q), v1 = vld1q_u8(q + 16);
    if (ib & 1) {   /* odd sub-block: the high nibbles */
        *w0 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
        *w1 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
    } else {        /* even sub-block: the low nibbles */
        *w0 = vreinterpretq_s8_u8(vandq_u8(v0, m4));
        *w1 = vreinterpretq_s8_u8(vandq_u8(v1, m4));
    }
}

/* the fused dot: per activation row, decode each sub-block in registers and dot
 * at once, the min term in gq_dot_i8's exact fmaf chain and final order. */
static void c24_dot_fused_q4_k(float *y, int ystride, const uint8_t *row,
                               const int8_t *xq, const float *dx, const int32_t *xs, int I, int S) {
    for (int s = 0; s < S; s++) {
        float32x4_t vacc = vdupq_n_f32(0.f);
        float mm = 0.f;
        const int8_t *xs8 = xq + (int64_t)s * I;
        const float *dxs = dx + (int64_t)s * (I / 32);
        const int32_t *xss = xs + (int64_t)s * (I / 16);
        int64_t xb = 0;
        for (int64_t blk = 0; blk < I / 256; blk++) {
            const uint8_t *b = row + blk * 144;
            const float d = gq_f16(gq_rd16(b)), dmin = gq_f16(gq_rd16(b + 2));
            for (int ib = 0; ib < 8; ib++, xb++) {
                float s2[2], m2[2]; int8x16_t w0, w1;
                c24_sb_q4_k(b, d, dmin, ib, s2, m2, &w0, &w1);
                int32x4_t p0, p1;
                gq_arm_dot2(w0, w1, vld1q_s8(xs8 + 32 * xb), vld1q_s8(xs8 + 32 * xb + 16), &p0, &p1);
                const float dd = dxs[xb];
                vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p0), s2[0] * dd);
                vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p1), s2[1] * dd);
                mm = fmaf(dd, fmaf(m2[0], (float)xss[2 * xb], m2[1] * (float)xss[2 * xb + 1]), mm);
            }
        }
        y[(int64_t)s * ystride] = vaddvq_f32(vacc) - mm;
    }
}

static double cell_fused(int reps, void *p) {
    Ctx *c = p; const int64_t rb = gq_row_bytes(c->t, c->I); double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++) {
            const uint8_t *row = c->W + (int64_t)o * rb;
            switch (c->t) {
            case GQ_Q4_K:    c24_dot_fused_q4_k(c->y + o, c->O, row, c->a.q, c->a.d, c->a.s, c->I, c->S); break;
            case GQ_IQ2_XXS: gq_dot_arm_iq2_xxs(c->y + o, c->O, row, c->a.q, c->a.d, c->I, c->S); break;
            case GQ_IQ3_XXS: gq_dot_arm_iq3_xxs(c->y + o, c->O, row, c->a.q, c->a.d, c->I, c->S); break;
            default: fprintf(stderr, "no fused kernel for this type\n"); exit(1);
            }
        }
    return now() - t0;
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
/* the engine's per-row dispatch today: gq_row_all -> unpack + gq_dot_i8 (+ min) */
static double cell_rowall(int reps, void *p) {
    Ctx *c = p; const int64_t rb = gq_row_bytes(c->t, c->I); double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++)
            gq_row_all(c->y + o, c->O, &c->a, c->W + (int64_t)o * rb, c->t, c->I);
    return now() - t0;
}
/* decode-only: the unpack the current path pays per row */
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

static void run_shape(const char *name, int t, int I, int O, const int *Ss, int nS, int threads, int has_fused) {
    Ctx c; memset(&c, 0, sizeof c);
    c.t = t; c.I = I; c.O = O;
    const int64_t rb = gq_row_bytes(t, I);
    c.W = malloc((size_t)O * rb);
    uint8_t *Wm = (uint8_t *)c.W;
    for (int64_t i = 0; i < (int64_t)O * rb; i++) Wm[i] = (uint8_t)(frand() * 256.0);  /* random block bytes */
    c.x = malloc((size_t)4096 * I * sizeof(float));
    c.y = malloc((size_t)4096 * O * sizeof(float));
    for (int i = 0; i < 4096 * I; i++) c.x[i] = (float)frand();
    /* sanity: the fused candidate must reproduce the current path's bits
     * (same sub-block scales/mins/codes, gq_dot_i8's NEON float order and
     * fmaf min chain). NaN scales compare as "not greater", as in C22. */
    double fused_maxdiff = 0;
    if (has_fused && t == GQ_Q4_K) {
        if (gq_act_set(&c.a, c.x, 2, I)) exit(1);
        for (int o = 0; o < 8; o++) {
            float yk[2] = {0, 0}, yp[2] = {0, 0};
            gq_row_all(yk, 1, &c.a, c.W + (int64_t)o * rb, t, I);
            c24_dot_fused_q4_k(yp, 1, c.W + (int64_t)o * rb, c.a.q, c.a.d, c.a.s, I, c.a.S);
            for (int s = 0; s < 2; s++) {
                double d = fabs((double)yk[s] - yp[s]);
                if (d > fused_maxdiff) fused_maxdiff = d;
            }
        }
    }
    printf("{\"shape\":\"%s\",\"t\":\"%s\",\"I\":%d,\"O\":%d,\"threads\":%d,\"fused_precheck_maxdiff\":%.3e",
           name, gq_type_name(t), I, O, threads, has_fused ? fused_maxdiff : 0.0);
    for (int i = 0; i < nS; i++) {
        int S = Ss[i];
        c.S = S;
        if (gq_act_set(&c.a, c.x, S, I)) exit(1);
        double gf = 2.0 * (double)S * I * O / 1e9;
        double t_m = time_cell(cell_matmul, &c);
        double t_q = time_cell(cell_quant, &c);
        double t_r = time_cell(cell_rowall, &c);
        double t_u = time_cell(cell_unpack, &c);
        double t_f = has_fused ? time_cell(cell_fused, &c) : 0.0;
        if (has_fused)
            printf(",\"%d\":{\"flops\":%.3f,\"matmul_s\":%.4f,\"matmul_gflops\":%.1f,"
                   "\"quant_s\":%.5f,\"rowall_s\":%.4f,\"rowall_gflops\":%.1f,"
                   "\"unpack_s\":%.4f,\"fused_gflops\":%.1f,"
                   "\"unpack_share_pct\":%.1f,\"fused_s1_gain\":%.2f,\"fused_vs_matmul\":%.2f}",
                   S, gf,
                   t_m, gf / t_m,
                   t_q, t_r, gf / t_r,
                   t_u, gf / t_f,
                   100.0 * t_u / t_r, S == 1 ? t_r / t_f : 0.0, t_m / t_f);
        else
            printf(",\"%d\":{\"flops\":%.3f,\"matmul_s\":%.4f,\"matmul_gflops\":%.1f,"
                   "\"quant_s\":%.5f,\"rowall_s\":%.4f,\"rowall_gflops\":%.1f,"
                   "\"unpack_s\":%.4f,\"unpack_share_pct\":%.1f}",
                   S, gf,
                   t_m, gf / t_m,
                   t_q, t_r, gf / t_r,
                   t_u, 100.0 * t_u / t_r);
    }
    printf("}\n");
    fflush(stdout);
    free(c.W); free(c.x); free(c.y); gq_act_free(&c.a);
}

int main(void) {
    int threads = 1;
#if defined(_OPENMP)
#pragma omp parallel
    { if (omp_get_thread_num() == 0) threads = omp_get_num_threads(); }
#endif
    printf("{\"info\":\"c24_q4k_bench\",\"threads\":%d,\"arm_neon\":%d,\"dotprod\":%d}\n",
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
    /* the target: Q4_K_M's expert shapes */
    run_shape("gate_up", GQ_Q4_K, 2048, 512, Ss, 7, threads, 1);
    run_shape("down",    GQ_Q4_K, 512,  2048, Ss, 7, threads, 1);
    /* straggler sizing: the IQ3 file's early-layer types (no fused candidate yet) */
    run_shape("gate_up_q2k", GQ_Q2_K,  2048, 512, Ss, 7, threads, 0);
    run_shape("down_q2k",    GQ_Q2_K,  512,  2048, Ss, 7, threads, 0);
    run_shape("gate_up_q3k", GQ_Q3_K,  2048, 512, Ss, 7, threads, 0);
    run_shape("down_q3k",    GQ_Q3_K,  512,  2048, Ss, 7, threads, 0);
    run_shape("gate_up_iq4xs", GQ_IQ4_XS, 2048, 512, Ss, 7, threads, 0);
    run_shape("down_iq4xs",    GQ_IQ4_XS, 512,  2048, Ss, 7, threads, 0);
    /* C18 calibration: the shipped fused XXS kernels on the same machine state */
    run_shape("gate_up_xxs", GQ_IQ2_XXS, 2048, 512, Ss, 7, threads, 1);
    run_shape("down_xxs",    GQ_IQ3_XXS, 512,  2048, Ss, 7, threads, 1);
    return 0;
}
