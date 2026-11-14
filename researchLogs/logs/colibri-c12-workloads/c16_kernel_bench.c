/* c16_kernel_bench.c -- standalone profile of the ARM fused expert-matmul kernel
 * (the IQ2_XXS/IQ3_XXS grid dot the CPU-GGUF serving path runs for routed experts),
 * calling the exact engine entry points (gq_matmul / gq_row_all) plus decomposition
 * cells: activation quantize, weight-decode-only, and a decode-hoisted variant that
 * mirrors the kernel's float steps but decodes each weight row once for all rows.
 *
 * Shapes are the model's: gate/up I=2048 O=512 (IQ2_XXS), down I=512 O=2048 (IQ3_XXS);
 * S sweeps the per-expert row counts measured in C14 phase 0 (mean 411 at 12K, 455 max).
 *
 * Build (from c/):
 *   clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
 *     ../researchLogs/logs/colibri-c12-workloads/c16_kernel_bench.c -o /tmp/c16bench \
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
    /* decode-once scratch, filled by cell_pre_dec_setup */
    int8_t *dw;    /* I decoded int8 per weight row, sub-block of 32 contiguous */
    float *ds;     /* I/32 sub-block scales (s2[0]; s2[1] is the same value) */
    int decode_bytes;
} Ctx;

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
/* the engine's per-row dispatch on ARM: gq_row_all -> fused grid dot */
static double cell_rowall(int reps, void *p) {
    Ctx *c = p; const int64_t rb = gq_row_bytes(c->t, c->I); double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++)
            gq_row_all(c->y + o, c->O, &c->a, c->W + (int64_t)o * rb, c->t, c->I);
    return now() - t0;
}
/* decode-only: GQ_DOT_GRID_ARM's weight work with the dot removed */
static double cell_decode(int reps, void *p) {
    Ctx *c = p; const int64_t rb = gq_row_bytes(c->t, c->I);
    const int BYTES = c->decode_bytes; volatile double dacc = 0;
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++) {
            const uint8_t *row = c->W + (int64_t)o * rb;
            for (int s = 0; s < c->S; s++)
                for (int64_t blk = 0; blk < c->I / 256; blk++) {
                    const uint8_t *b = row + blk * BYTES; const float d = gq_f16(gq_rd16(b));
                    for (int ib = 0; ib < 8; ib++) {
                        float s2[2]; int8x16_t w0, w1;
                        if (c->t == GQ_IQ2_XXS) gq_arm_sb_iq2_xxs(b, d, ib, s2, &w0, &w1);
                        else                    gq_arm_sb_iq3_xxs(b, d, ib, s2, &w0, &w1);
                        dacc += s2[0] + s2[1];
                    }
                }
        }
    return now() - t0;
}
/* decode-once setup (outside timing): SB-decode each weight row's sub-blocks into
 * dw (32 int8 each) and ds (one scale each) */
static void pre_decode(Ctx *c) {
    const int64_t rb = gq_row_bytes(c->t, c->I);
    for (int o = 0; o < c->O; o++) {
        const uint8_t *row = c->W + (int64_t)o * rb;
        int8_t *dw = c->dw + (int64_t)o * c->I; float *ds = c->ds + (int64_t)o * (c->I / 32);
        int64_t xb = 0;
        for (int64_t blk = 0; blk < c->I / 256; blk++) {
            const uint8_t *b = row + blk * c->decode_bytes; const float d = gq_f16(gq_rd16(b));
            for (int ib = 0; ib < 8; ib++, xb++) {
                float s2[2]; int8x16_t w0, w1;
                if (c->t == GQ_IQ2_XXS) gq_arm_sb_iq2_xxs(b, d, ib, s2, &w0, &w1);
                else                    gq_arm_sb_iq3_xxs(b, d, ib, s2, &w0, &w1);
                vst1q_s8(dw + 32 * xb, w0); vst1q_s8(dw + 32 * xb + 16, w1);
                ds[xb] = s2[0];
            }
        }
    }
}
/* decode hoisted: the kernel's exact float steps (p*scale fma order), weights pre-decoded */
static double cell_pre_dot(int reps, void *p) {
    Ctx *c = p; double t0 = now();
    for (int r = 0; r < reps; r++)
        for (int o = 0; o < c->O; o++) {
            const int8_t *dw = c->dw + (int64_t)o * c->I;
            const float *ds = c->ds + (int64_t)o * (c->I / 32);
            for (int s = 0; s < c->S; s++) {
                float32x4_t vacc = vdupq_n_f32(0.f);
                const int8_t *xs8 = c->a.q + (int64_t)s * c->I;
                const float *dxs = c->a.d + (int64_t)s * (c->I / 32);
                for (int xb = 0; xb < c->I / 32; xb++) {
                    int32x4_t p0, p1;
                    gq_arm_dot2(vld1q_s8(dw + 32 * xb), vld1q_s8(dw + 32 * xb + 16),
                                vld1q_s8(xs8 + 32 * xb), vld1q_s8(xs8 + 32 * xb + 16), &p0, &p1);
                    vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p0), ds[xb] * dxs[xb]);
                    vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p1), ds[xb] * dxs[xb]);
                }
                c->y[(int64_t)s * c->O + o] = vaddvq_f32(vacc);
            }
        }
    return now() - t0;
}
/* the same decode-once dot under the engine's dispatch shape: omp over output rows */
static double cell_pre_dot_omp(int reps, void *p) {
    Ctx *c = p; double t0 = now();
    for (int r = 0; r < reps; r++) {
        const int O = c->O, S = c->S, I = c->I;
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *dw = c->dw + (int64_t)o * I;
            const float *ds = c->ds + (int64_t)o * (I / 32);
            for (int s = 0; s < S; s++) {
                float32x4_t vacc = vdupq_n_f32(0.f);
                const int8_t *xs8 = c->a.q + (int64_t)s * I;
                const float *dxs = c->a.d + (int64_t)s * (I / 32);
                for (int xb = 0; xb < I / 32; xb++) {
                    int32x4_t p0, p1;
                    gq_arm_dot2(vld1q_s8(dw + 32 * xb), vld1q_s8(dw + 32 * xb + 16),
                                vld1q_s8(xs8 + 32 * xb), vld1q_s8(xs8 + 32 * xb + 16), &p0, &p1);
                    vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p0), ds[xb] * dxs[xb]);
                    vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p1), ds[xb] * dxs[xb]);
                }
                c->y[(int64_t)s * O + o] = vaddvq_f32(vacc);
            }
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
    c.decode_bytes = GQ_TYPES[t].bytes;
    c.dw = malloc((size_t)O * I); c.ds = malloc((size_t)O * (I / 32) * sizeof(float));
    pre_decode(&c);
    /* sanity: pre_decode's stored codes must re-dot to the kernel's values */
    if (gq_act_set(&c.a, c.x, 2, I)) exit(1);
    double maxdiff = 0;
    for (int o = 0; o < 8; o++) {
        float yk = 0, yp = 0;
        gq_row_all(&yk, 1, &c.a, c.W + (int64_t)o * rb, t, I);
        /* reuse cell math for one (o, s) by direct computation */
        float32x4_t vacc = vdupq_n_f32(0.f);
        const int8_t *dw = c.dw + (int64_t)o * I; const float *ds = c.ds + (int64_t)o * (I / 32);
        const int8_t *xs8 = c.a.q; const float *dxs = c.a.d;
        for (int xb = 0; xb < I / 32; xb++) {
            int32x4_t p0, p1;
            gq_arm_dot2(vld1q_s8(dw + 32 * xb), vld1q_s8(dw + 32 * xb + 16),
                        vld1q_s8(xs8 + 32 * xb), vld1q_s8(xs8 + 32 * xb + 16), &p0, &p1);
            vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p0), ds[xb] * dxs[xb]);
            vacc = vmlaq_n_f32(vacc, vcvtq_f32_s32(p1), ds[xb] * dxs[xb]);
        }
        yp = vaddvq_f32(vacc);
        double d = fabs((double)yk - yp);
        if (d > maxdiff) maxdiff = d;
    }
    printf("{\"shape\":\"%s\",\"t\":\"%s\",\"I\":%d,\"O\":%d,\"threads\":%d,\"precheck_maxdiff\":%.3e",
           name, gq_type_name(t), I, O, threads, maxdiff);
    for (int i = 0; i < nS; i++) {
        int S = Ss[i];
        c.S = S;
        if (gq_act_set(&c.a, c.x, S, I)) exit(1);
        double gf = 2.0 * (double)S * I * O / 1e9;
        double t_m = time_cell(cell_matmul, &c);
        double t_q = time_cell(cell_quant, &c);
        double t_r = time_cell(cell_rowall, &c);
        double t_d = time_cell(cell_decode, &c);
        double t_p = time_cell(cell_pre_dot, &c);
        double t_po = time_cell(cell_pre_dot_omp, &c);
        printf(",\"%d\":{\"flops\":%.3f,\"matmul_s\":%.4f,\"matmul_gflops\":%.1f,"
               "\"quant_s\":%.5f,\"rowall_s\":%.4f,\"rowall_gflops\":%.1f,"
               "\"decode_s\":%.4f,\"pre_dot_s\":%.4f,\"pre_dot_gflops\":%.1f,"
               "\"pre_dot_omp_gflops\":%.1f,"
               "\"decode_share_pct\":%.1f,\"speedup_pot\":%.2f}",
               S, gf,
               t_m, gf / t_m,
               t_q, t_r, gf / t_r,
               t_d, t_p, gf / t_p,
               gf / t_po,
               100.0 * t_d / t_r, t_r / t_p);
    }
    printf("}\n");
    fflush(stdout);
    free(c.W); free(c.x); free(c.y); free(c.dw); free(c.ds); gq_act_free(&c.a);
}

int main(void) {
    int threads = 1;
#if defined(_OPENMP)
#pragma omp parallel
    { if (omp_get_thread_num() == 0) threads = omp_get_num_threads(); }
#endif
    printf("{\"info\":\"c16_kernel_bench\",\"threads\":%d,\"arm_neon\":%d,\"dotprod\":%d}\n",
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
    run_shape("gate_up", GQ_IQ2_XXS, 2048, 512, Ss, 7, threads);
    run_shape("down",    GQ_IQ3_XXS, 512,  2048, Ss, 7, threads);
    return 0;
}
