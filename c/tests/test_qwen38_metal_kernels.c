/* Metal kernel tests for Qwen3.8: the qwen38 kernels against a double-precision
 * reference on random data, at the real model's shapes and at the tiny
 * fixture's. No model, no checkpoint. Build and run:
 * make test-qwen38-metal-kernels
 *
 * The qwen38 kernels share qwen36's device and command batch, so a test may
 * chain q36 and q38 dispatches on the same tensors and sync once. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../qwen38_gpu.h"
#include "../ggml_quants.h"

/* ---- helpers (copies of tests/test_qwen36_kernels.c's) ------------------- */

static uint32_t g_rng = 0x2545f491u;
static float frand(void) {                      /* uniform in [-1, 1) */
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float)(g_rng & 0xffffffu) / 8388608.0f - 1.0f;
}

static int g_failures;

/* got vs ref, with |got-ref| <= tol * scale[i] (scale NULL: |ref| + 1e-3). */
static void check(const char *what, const float *got, const double *ref, const double *scale, size_t n, double tol) {
    double worst = 0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        double s = scale ? scale[i] : fabs(ref[i]) + 1e-3;
        if (!isfinite(got[i]) || !isfinite(ref[i]) || !isfinite(s)) {
            worst = INFINITY;
            at = i;
            break;
        }
        double e = fabs((double)got[i] - ref[i]) / (s > 1e-30 ? s : 1e-30);
        if (!(e <= worst)) { worst = e; at = i; }
    }
    int ok = worst <= tol;
    printf("%-34s %s  worst=%.3g at %zu (got %.9g ref %.9g)\n", what, ok ? "ok  " : "FAIL", worst, at,
           (double)got[at], ref[at]);
    if (!ok) g_failures++;
}

static q36_tensor *rand_tensor(size_t count, float scale) {
    q36_tensor *t = q36_tensor_new(count * sizeof(float));
    float *p = q36_tensor_data(t);
    for (size_t i = 0; i < count; i++) p[i] = frand() * scale;
    return t;
}

static float *F(q36_tensor *t) { return q36_tensor_data(t); }

static void sync_or_die(void) {
    if (!q36_gpu_sync()) { fprintf(stderr, "GPU batch failed\n"); exit(1); }
}

static void expect(const char *what, int ok) {
    printf("%-34s %s\n", what, ok ? "ok  " : "FAIL");
    if (!ok) g_failures++;
}

/* ---- plumbing ------------------------------------------------------------- */

/* The reference norm of the original layout: x * rsqrt(mean(x^2) + eps) * (1 + w). */
static void ref_norm(double *y, const double *x, const float *w, int rows, int dim, double eps) {
    for (int r = 0; r < rows; r++) {
        double ss = 0;
        for (int j = 0; j < dim; j++) ss += x[(size_t)r * dim + j] * x[(size_t)r * dim + j];
        double k = 1.0 / sqrt(ss / dim + eps);
        for (int j = 0; j < dim; j++) y[(size_t)r * dim + j] = x[(size_t)r * dim + j] * k * (1.0 + w[j]);
    }
}

/* q36_rmsnorm -> q38_copy -> q36_rmsnorm on the same tensors, one sync at the
 * end: the result is norm(copy(norm(x))) only if the q38 dispatch runs in the
 * q36 stream, after the first norm and before the second. (Out of order, the
 * copy reads b's zeros or the second norm reads x itself.) */
static void test_copy_interleaved(int rows, int dim, int exact) {
    size_t n = (size_t)rows * dim;
    const float eps = 1e-6f;
    q36_gpu_set_exact(exact);
    q36_tensor *a = rand_tensor(n, 3.0f), *w = rand_tensor((size_t)dim, 0.5f);
    q36_tensor *b = q36_tensor_new(n * sizeof(float));
    double *x = malloc(n * sizeof(double)), *ref = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) x[i] = F(a)[i];
    ref_norm(ref, x, F(w), rows, dim, eps);
    ref_norm(x, ref, F(w), rows, dim, eps);     /* x <- norm(copy(norm(x))) */

    int ok = q36_rmsnorm(b, a, w, rows, dim, eps, 0) && q38_copy(a, b, n) && q36_rmsnorm(b, a, w, rows, dim, eps, 0);
    sync_or_die();
    char name[64];
    snprintf(name, sizeof name, "copy interleaved %dx%d %s", rows, dim, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(b), x, NULL, n, exact ? 1e-5 : 3e-4);
    free(x); free(ref);
    q36_tensor_free(a); q36_tensor_free(w); q36_tensor_free(b);
}

/* q38_copy declines what does not fit, and enqueues nothing. */
static void test_copy_declines(void) {
    q36_tensor *x = rand_tensor(16, 1.0f), *y = q36_tensor_new(8 * sizeof(float));
    int ok = !q38_copy(y, x, 16) && !q38_copy(x, y, 16) && !q38_copy(y, x, 0) && !q38_copy(NULL, x, 4) &&
             !q38_copy(y, NULL, 4);
    sync_or_die();
    for (int i = 0; i < 8; i++) ok = ok && F(y)[i] == 0.0f;
    expect("copy declines bad sizes", ok);
    q36_tensor_free(x); q36_tensor_free(y);
}

/* ---- router and experts at Qwen3.8's width (512 experts, top-k up to 16) --- */

/* Qwen3.8's CPU router (qwen38_core.h's q38_moe_decode_ex): softmax with the maximum
 * subtracted, top-k by K passes of "the largest unused" with a strict > (ties go to the
 * lowest id), weights val / sum(top) (norm_topk). Weights in double. */
static void ref_router(int *ids, double *w, const float *logits, int E, int K) {
    float *p = malloc(sizeof(float) * E);
    float mx = logits[0];
    for (int e = 1; e < E; e++) if (logits[e] > mx) mx = logits[e];
    for (int e = 0; e < E; e++) p[e] = expf(logits[e] - mx);
    double top = 0;
    for (int z = 0; z < K; z++) {
        int best = -1;
        float bv = -1.f;
        for (int e = 0; e < E; e++) {
            int used = 0;
            for (int j = 0; j < z; j++) used |= ids[j] == e;
            if (!used && p[e] > bv) { bv = p[e]; best = e; }
        }
        ids[z] = best;
        w[z] = (double)expf(logits[best] - mx);
        top += w[z];
    }
    for (int z = 0; z < K; z++) w[z] /= top;
    free(p);
}

/* logits (rows x E) through q36_router_softmax and q36_router_select (one group):
 * the ids identical to ref_router's, the weights within tol. */
static void run_router(const char *what, const float *logits, int rows, int E, int K, int exact) {
    q36_gpu_set_exact(exact);
    size_t n = (size_t)rows * E, c = (size_t)rows * K;
    q36_tensor *lg = q36_tensor_new(n * sizeof(float)), *prob = q36_tensor_new(n * sizeof(float));
    q36_tensor *ids = q36_tensor_new(c * sizeof(int)), *wts = q36_tensor_new(c * sizeof(float));
    memcpy(F(lg), logits, n * sizeof(float));
    int *rid = malloc(sizeof(int) * c);
    double *rw = malloc(sizeof(double) * c);
    for (int r = 0; r < rows; r++) ref_router(rid + (size_t)r * K, rw + (size_t)r * K, logits + (size_t)r * E, E, K);
    int ok = q36_router_softmax(prob, lg, NULL, rows, E) && q36_router_select(ids, wts, prob, rows, E, K, 1, 1);
    sync_or_die();
    char name[96];
    snprintf(name, sizeof name, "%s %dx%d K=%d %s ids", what, rows, E, K, exact ? "exact" : "fast");
    const int *gid = q36_tensor_data(ids);
    int same = ok;
    for (size_t i = 0; i < c && same; i++) same = gid[i] == rid[i];
    expect(name, same);
    if (ok && !same)
        for (size_t i = 0; i < c; i++)
            if (gid[i] != rid[i]) { printf("    first mismatch at %zu: got %d ref %d\n", i, gid[i], rid[i]); break; }
    snprintf(name, sizeof name, "%s %dx%d K=%d %s weights", what, rows, E, K, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(wts), rw, NULL, c, exact ? 1e-5 : 3e-4);
    free(rid); free(rw);
    q36_tensor_free(lg); q36_tensor_free(prob); q36_tensor_free(ids); q36_tensor_free(wts);
}

/* Random logits, kept apart (a random permutation of a grid 0.01 apart, plus less
 * than a tenth of that of noise) so the CPU's and the GPU's float softmax pick the
 * same experts. */
static void test_router_wide(int rows, int E, int K, int exact) {
    float *l = malloc(sizeof(float) * (size_t)rows * E);
    int *perm = malloc(sizeof(int) * E);
    for (int r = 0; r < rows; r++) {
        for (int e = 0; e < E; e++) perm[e] = e;
        for (int e = E - 1; e > 0; e--) {
            int j = (int)((frand() + 1.0f) * 0.5f * (float)(e + 1));
            if (j > e) j = e;
            int t = perm[e]; perm[e] = perm[j]; perm[j] = t;
        }
        for (int e = 0; e < E; e++) l[(size_t)r * E + e] = (float)perm[e] * 0.01f - 2.0f + frand() * 0.0009f;
    }
    run_router("router_wide", l, rows, E, K, exact);
    free(l); free(perm);
}

/* Groups of equal logits: equal probabilities in both softmaxes, so the ids
 * show the tie-break. Three experts at the top (ids 400, 7, 300), a group of
 * 40 at the next value straddling the K boundary, the rest below. */
static void test_router_tie(int rows, int E, int K, int exact) {
    float *l = malloc(sizeof(float) * (size_t)rows * E);
    for (int r = 0; r < rows; r++) {
        float *lr = l + (size_t)r * E;
        for (int e = 0; e < E; e++) lr[e] = -1.0f - 0.5f * (float)((e * 37 + r) % 5);
        for (int e = 0; e < 40; e++) lr[(e * 13 + 5 * r + 3) % E] = 1.0f;
        lr[400 % E] = lr[7] = lr[300 % E] = 2.0f;
    }
    run_router("router tie", l, rows, E, K, exact);
    free(l);
}

/* q36_router_select declines more than 512 experts, more than 16, groups at the
 * wide sizes; nothing is written. */
static void test_router_wide_declines(void) {
    q36_tensor *prob = rand_tensor(2 * 600, 1.0f), *ids = q36_tensor_new(2 * 20 * 4), *w = q36_tensor_new(2 * 20 * 4);
    int took = q36_router_select(ids, w, prob, 2, 513, 10, 1, 1) + q36_router_select(ids, w, prob, 2, 512, 17, 1, 1) +
               q36_router_select(ids, w, prob, 2, 512, 10, 2, 1) + q36_router_select(ids, w, prob, 2, 300, 9, 2, 1);
    sync_or_die();
    int clean = 1;
    for (int i = 0; i < 40; i++) clean = clean && ((int *)q36_tensor_data(ids))[i] == 0 && F(w)[i] == 0.0f;
    expect("router_wide declines E=513 K=17 G>1", took == 0 && clean);
    q36_tensor_free(prob); q36_tensor_free(ids); q36_tensor_free(w);
}

/* q36_expert_sum with K choices: out = residual + sum_k w_k y_k + shared against a double
 * reference, scaled by the magnitude of the terms (|residual| + sum |w y| + |shared|). */
static void test_expert_sum(int rows, int H, int K, int exact) {
    q36_gpu_set_exact(exact);
    size_t cells = (size_t)rows * H;
    q36_tensor *y = rand_tensor(cells * K, 2.0f), *wts = rand_tensor((size_t)rows * K, 0.5f);
    q36_tensor *sh = rand_tensor(cells, 1.0f), *res = rand_tensor(cells, 4.0f), *out = q36_tensor_new(cells * 4);
    double *ref = malloc(sizeof(double) * cells), *mag = malloc(sizeof(double) * cells);
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < H; o++) {
            size_t i = (size_t)r * H + o;
            double s = 0, m = 0;
            for (int k = 0; k < K; k++) {
                double t = (double)F(wts)[r * K + k] * F(y)[((size_t)r * K + k) * H + o];
                s += t; m += fabs(t);
            }
            ref[i] = F(res)[i] + s + F(sh)[i];
            mag[i] = fabs(F(res)[i]) + m + fabs(F(sh)[i]) + 1e-3;
        }
    for (size_t i = 0; i < cells; i++) F(out)[i] = NAN;
    int ok = q36_expert_sum(out, y, wts, sh, res, rows, H, K);
    sync_or_die();
    char name[80];
    snprintf(name, sizeof name, "expert_sum %dx%d K=%d %s", rows, H, K, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(out), ref, mag, cells, exact ? 1e-5 : 3e-4);
    free(ref); free(mag);
    q36_tensor_free(y); q36_tensor_free(wts); q36_tensor_free(sh); q36_tensor_free(res); q36_tensor_free(out);
}

static uint8_t rand_byte(void) { return (uint8_t)((g_rng = g_rng * 1664525u + 1013904223u) >> 24); }

static uint16_t f16_bits(float f) {
    _Float16 h = (_Float16)f;
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}

/* rows of `in` IQ4_XS or IQ4_NL weights (type t) as test_qwen36_kernels.c's gq_fill_rows leaves
 * them: random quants and packed scales, the block's f16 scale (its first two bytes in both
 * types) of a sensible size and sign. */
static void iq4_fill_rows(int t, uint8_t *p, int rows, int in) {
    const int64_t rb = gq_row_bytes(t, in), blocks = (int64_t)rows * (in / GQ_TYPES[t].blck);
    const int bytes = GQ_TYPES[t].bytes;
    for (int64_t i = 0; i < rb * rows; i++) p[i] = rand_byte();
    for (int64_t k = 0; k < blocks; k++) {
        float d = (frand() + 1.5f) * 0.004f;
        uint16_t dh = f16_bits(frand() < -0.6f ? -d : d);
        memcpy(p + k * bytes, &dh, 2);
    }
}

/* GGUF routed experts at Qwen3.8's width and in its IQ4_XS file's types (gate and up IQ4_XS;
 * down IQ4_NL, as llama.cpp stores the 640-column ffn_down_exps of an IQ quant, 640 not being a
 * multiple of IQ4_XS's 256-weight block): `rows` tokens each routing to the same
 * NE = K experts (in a different order a row), bound in two groups of slots, slots 0..7 and
 * then slots 0..NE-9: the choice index row*K+k runs past 8 a row. gate_up and down against a
 * double reference of the dequantized SwiGLU experts (down from the GPU's h), scaled by
 * the magnitude of the products. */
static void test_expert_gq(int rows, int H, int I, int K, int exact) {
    const int t = GQ_IQ4_XS, td = GQ_IQ4_NL, NE = K, choices = rows * K;
    const int64_t rbg = gq_row_bytes(t, H), rbd = gq_row_bytes(td, I);
    const size_t up = (size_t)(rbg * I), down = up + (size_t)(rbg * I), bytes = down + (size_t)(rbd * H);
    q36_gpu_set_exact(exact);
    q36_tensor **slot = calloc((size_t)NE, sizeof(*slot));
    float sg[16], su[16], sd[16];
    for (int e = 0; e < NE; e++) {
        slot[e] = q36_tensor_new(bytes);
        uint8_t *p = q36_tensor_data(slot[e]);
        iq4_fill_rows(t, p, I, H);
        iq4_fill_rows(t, p + up, I, H);
        iq4_fill_rows(td, p + down, H, I);
        sg[e] = 0.5f + 0.25f * (frand() + 1.0f);
        su[e] = 0.5f + 0.25f * (frand() + 1.0f);
        sd[e] = 0.5f + 0.25f * (frand() + 1.0f);
    }
    int *ids = malloc(sizeof(int) * choices);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < K; k++) ids[r * K + k] = (k * (2 * r + 1) + 3 * r) % NE;   /* a permutation a row */
    q36_tensor *x = rand_tensor((size_t)rows * H, 1.0f), *pairs = q36_tensor_new((size_t)choices * 4);
    q36_tensor *h = q36_tensor_new((size_t)choices * I * 4), *y = q36_tensor_new((size_t)choices * H * 4);
    for (size_t i = 0; i < (size_t)choices * I; i++) F(h)[i] = NAN;
    for (size_t i = 0; i < (size_t)choices * H; i++) F(y)[i] = NAN;
    uint32_t *pp = q36_tensor_data(pairs);
    int ok = 1, q = 0;
    for (int g0 = 0; g0 < NE; g0 += 8) {
        int ng = NE - g0 < 8 ? NE - g0 : 8, first = q;
        q36_tensor *s8[8] = {0};
        q36_gq_experts tt = {t, t, td, {0}, {0}, {0}};
        for (int j = 0; j < ng; j++) {
            int e = g0 + j;
            s8[j] = slot[e];
            tt.sg[j] = sg[e]; tt.su[j] = su[e]; tt.sd[j] = sd[e];
            for (int p = 0; p < choices; p++)
                if (ids[p] == e) pp[q++] = (uint32_t)p | (uint32_t)j << 28;
        }
        ok = q36_expert_gate_up_gq(h, x, pairs, first, q - first, s8, rows, H, I, K, &tt) && ok;
        ok = q36_expert_down_gq(y, h, pairs, first, q - first, s8, rows, H, I, K, &tt) && ok;
    }
    sync_or_die();
    double *rh = malloc(sizeof(double) * choices * I), *hm = malloc(sizeof(double) * choices * I);
    double *ry = malloc(sizeof(double) * choices * H), *ym = malloc(sizeof(double) * choices * H);
    float *gr = malloc(sizeof(float) * H), *ur = malloc(sizeof(float) * H), *dr = malloc(sizeof(float) * I);
    for (int p = 0; p < choices; p++) {
        int e = ids[p];
        const uint8_t *w = q36_tensor_data(slot[e]);
        const float *xr = F(x) + (size_t)(p / K) * H, *hr = F(h) + (size_t)p * I;
        for (int i = 0; i < I; i++) {
            gq_dequant_row(t, w + i * rbg, gr, H);
            gq_dequant_row(t, w + up + i * rbg, ur, H);
            double g = 0, u = 0, mg = 0, mu = 0;
            for (int j = 0; j < H; j++) {
                g += (double)gr[j] * xr[j]; mg += fabs((double)gr[j] * xr[j]);
                u += (double)ur[j] * xr[j]; mu += fabs((double)ur[j] * xr[j]);
            }
            g *= sg[e]; u *= su[e]; mg *= sg[e]; mu *= su[e];
            rh[(size_t)p * I + i] = g / (1.0 + exp(-g)) * u;
            hm[(size_t)p * I + i] = (mg * fabs(u) + mu * fabs(g)) + 1e-9;
        }
        for (int o = 0; o < H; o++) {
            gq_dequant_row(td, w + down + o * rbd, dr, I);
            double v = 0, m = 0;
            for (int j = 0; j < I; j++) { v += (double)dr[j] * hr[j]; m += fabs((double)dr[j] * hr[j]); }
            ry[(size_t)p * H + o] = v * sd[e];
            ym[(size_t)p * H + o] = m * sd[e] + 1e-9;
        }
    }
    char name[96];
    snprintf(name, sizeof name, "expert_gq IQ4_XS|IQ4_NL %dx%d I=%d K=%d %s gate_up", rows, H, I, K, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(h), rh, hm, (size_t)choices * I, exact ? 2e-5 : 3e-4);
    snprintf(name, sizeof name, "expert_gq IQ4_XS|IQ4_NL %dx%d I=%d K=%d %s down", rows, H, I, K, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(y), ry, ym, (size_t)choices * H, exact ? 2e-5 : 3e-4);
    free(rh); free(hm); free(ry); free(ym); free(gr); free(ur); free(dr); free(ids);
    for (int e = 0; e < NE; e++) q36_tensor_free(slot[e]);
    free(slot);
    q36_tensor_free(x); q36_tensor_free(pairs); q36_tensor_free(h); q36_tensor_free(y);
}

/* ---- BF16 dense dot (indexer projections) --------------------------------- */

static void test_dot_bf16(int rows, int in, int out, int exact) {
    q36_gpu_set_exact(exact);
    q36_tensor *x = rand_tensor((size_t)rows * in, 1.0f), *w = q36_tensor_new((size_t)out * in * 2);
    q36_tensor *y = q36_tensor_new((size_t)rows * out * 4);
    uint16_t *wb = q36_tensor_data(w);
    for (size_t i = 0; i < (size_t)out * in; i++) {
        float f = frand() * 0.1f;
        uint32_t u;
        memcpy(&u, &f, 4);
        wb[i] = (uint16_t)(u >> 16);
    }
    size_t cells = (size_t)rows * out;
    for (size_t i = 0; i < cells; i++) F(y)[i] = NAN;
    /* scaled by the magnitude of the terms, sum |w x|: a random-sign dot cancels to far below its
     * terms, and float error follows the terms, not the result */
    double *ref = malloc(sizeof(double) * cells), *mag = malloc(sizeof(double) * cells);
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < out; o++) {
            double s = 0, m = 0;
            for (int j = 0; j < in; j++) {
                uint32_t u = (uint32_t)wb[(size_t)o * in + j] << 16;
                float wf;
                memcpy(&wf, &u, 4);
                double t = (double)wf * F(x)[(size_t)r * in + j];
                s += t; m += fabs(t);
            }
            ref[(size_t)r * out + o] = s;
            mag[(size_t)r * out + o] = m + 1e-3;
        }
    int ok = q38_dot_bf16(y, x, w, rows, in, out);
    sync_or_die();
    char name[80];
    snprintf(name, sizeof name, "dot_bf16 %dx%d->%d %s", rows, in, out, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(y), ref, mag, cells, exact ? 2e-5 : 3e-4);
    free(ref); free(mag);
    q36_tensor_free(x); q36_tensor_free(w); q36_tensor_free(y);
}

static void test_dot_bf16_declines(void) {
    q36_tensor *x = rand_tensor(2 * 32, 1.0f), *w = q36_tensor_new(4 * 32 * 2), *y = q36_tensor_new(2 * 4 * 4);
    int ok = !q38_dot_bf16(y, x, w, 2, 30, 4) && !q38_dot_bf16(y, x, w, 2, 32, 5) && !q38_dot_bf16(y, x, w, 0, 32, 4) &&
             !q38_dot_bf16(y, NULL, w, 2, 32, 4);
    sync_or_die();
    for (int i = 0; i < 8; i++) ok = ok && F(y)[i] == 0.0f;
    expect("dot_bf16 declines in=30", ok);
    q36_tensor_free(x); q36_tensor_free(w); q36_tensor_free(y);
}

/* ---- hyper-connection ops ------------------------------------------------- */

static double ref_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

static void test_hc_unary(int mode, int n, int exact) {   /* mode 0 low, 1 inj */
    q36_gpu_set_exact(exact);
    const float fc = 4.0f;
    q36_tensor *a = rand_tensor((size_t)n, 6.0f), *y = q36_tensor_new((size_t)n * 4);
    double *ref = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        double v = (double)F(a)[i] / fc;
        ref[i] = mode == 0 ? v * ref_sigmoid(v) : 2.0 * ref_sigmoid(v);
        F(y)[i] = NAN;
    }
    int ok = mode == 0 ? q38_hc_low(y, a, n, fc) : q38_hc_inj(y, a, n, fc);
    sync_or_die();
    char name[64];
    snprintf(name, sizeof name, "hc_%s n=%d %s", mode == 0 ? "low" : "inj", n, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(y), ref, NULL, (size_t)n, exact ? 1e-5 : 3e-4);
    free(ref);
    q36_tensor_free(a); q36_tensor_free(y);
}

static void test_hc_mix(int rows, int D, int C, int exact) {
    q36_gpu_set_exact(exact);
    size_t W = (size_t)C * D, n = (size_t)rows * D;
    q36_tensor *a = rand_tensor((size_t)rows * W, 4.0f), *b = rand_tensor((size_t)rows * W, 2.0f);
    q36_tensor *y = q36_tensor_new(n * 4);
    /* a signed sum over k can cancel: scaled by sum |terms| / C (Ruling 9) */
    double *ref = malloc(sizeof(double) * n), *mag = malloc(sizeof(double) * n);
    for (int r = 0; r < rows; r++)
        for (int d = 0; d < D; d++) {
            double s = 0, m = 0;
            for (int k = 0; k < C; k++) {
                double t = ref_sigmoid(F(a)[r * W + (size_t)k * D + d]) * F(b)[r * W + (size_t)k * D + d];
                s += t; m += fabs(t);
            }
            ref[(size_t)r * D + d] = s / C;
            mag[(size_t)r * D + d] = m / C + 1e-3;
            F(y)[(size_t)r * D + d] = NAN;
        }
    int ok = q38_hc_mix(y, a, b, rows, D, C, (float)C);
    sync_or_die();
    char name[64];
    snprintf(name, sizeof name, "hc_mix %dx%d C%d %s", rows, D, C, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(y), ref, mag, n, exact ? 1e-5 : 3e-4);
    free(ref); free(mag);
    q36_tensor_free(a); q36_tensor_free(b); q36_tensor_free(y);
}

static void test_hc_apply(int rows, int D, int C, int exact) {
    q36_gpu_set_exact(exact);
    size_t W = (size_t)C * D, n = (size_t)rows * W;
    q36_tensor *h = rand_tensor(n, 3.0f), *inj = rand_tensor((size_t)rows * C, 2.0f), *blk = rand_tensor((size_t)rows * D, 2.0f);
    double *ref = malloc(sizeof(double) * n);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < C; k++)
            for (int d = 0; d < D; d++)
                ref[r * W + (size_t)k * D + d] = (double)F(h)[r * W + (size_t)k * D + d] +
                    (double)F(inj)[(size_t)r * C + k] * F(blk)[(size_t)r * D + d];
    int ok = q38_hc_apply(h, inj, blk, rows, D, C);
    sync_or_die();
    char name[64];
    snprintf(name, sizeof name, "hc_apply %dx%d C%d %s", rows, D, C, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(h), ref, NULL, n, exact ? 1e-5 : 3e-4);
    free(ref);
    q36_tensor_free(h); q36_tensor_free(inj); q36_tensor_free(blk);
}

static void test_hc_norm(int rows, int D, int C, int exact) {
    q36_gpu_set_exact(exact);
    size_t W = (size_t)C * D, n = (size_t)rows * W;
    q36_tensor *x = rand_tensor(n, 3.0f), *w = rand_tensor(W, 0.5f), *y = q36_tensor_new(n * 4);
    double *xd = malloc(sizeof(double) * n), *ref = malloc(sizeof(double) * n);
    for (size_t i = 0; i < n; i++) { xd[i] = F(x)[i]; F(y)[i] = NAN; }
    for (int r = 0; r < rows; r++)
        for (int b = 0; b < C; b++)
            ref_norm(ref + r * W + (size_t)b * D, xd + r * W + (size_t)b * D, F(w) + (size_t)b * D, 1, D, 1e-6);
    int ok = q38_hc_norm(y, x, w, rows, D, C, 1e-6f);
    sync_or_die();
    char name[64];
    snprintf(name, sizeof name, "hc_norm %dx%d C%d %s", rows, D, C, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(y), ref, NULL, n, exact ? 1e-5 : 3e-4);
    free(xd); free(ref);
    q36_tensor_free(x); q36_tensor_free(w); q36_tensor_free(y);
}

static void test_hc_declines(void) {
    q36_tensor *a = rand_tensor(64, 1.0f), *y = q36_tensor_new(16 * 4);
    int ok = !q38_hc_low(y, a, 0, 4.f) && !q38_hc_low(y, a, 32, 4.f) && !q38_hc_inj(NULL, a, 4, 4.f) &&
             !q38_hc_mix(y, a, a, 2, 16, 4, 4.f) && !q38_hc_mix(y, a, a, 2, 8, 0, 4.f) &&
             !q38_hc_apply(a, y, y, 2, 16, 4) && !q38_hc_norm(y, a, a, 0, 4, 4, 1e-6f) &&
             !q38_hc_norm(y, a, a, 1, 4, 4, 0.f) && !q38_hc_norm(y, a, a, 1, 4, 4, NAN) &&
             !q38_hc_norm(y, a, a, 2, 16, 4, 1e-6f);
    sync_or_die();
    for (int i = 0; i < 16; i++) ok = ok && F(y)[i] == 0.0f;
    expect("hc declines bad sizes", ok);
    q36_tensor_free(a); q36_tensor_free(y);
}

/* The CPU's q38_gr_read in double, with the dot products on the f32 weights:
 * rms0 per stream (stream b uses weight row b of the W-long nw),
 * low = silu((down.norm)/C), mix = up.low, mixed = sum_b sigmoid(mix)*norm / C,
 * inject = 2 sigmoid((inj.norm)/C). */
static void test_gr_read(int S, int H, int C, int R, int exact) {
    q36_gpu_set_exact(exact);
    const float eps = 1e-6f;
    size_t W = (size_t)C * H;
    q36_tensor *hyper = rand_tensor((size_t)S * W, 2.0f), *nw = rand_tensor(W, 0.5f);
    q36_tensor *down = rand_tensor((size_t)R * W, 1.0f / sqrtf((float)W));
    q36_tensor *up = rand_tensor(W * R, 1.0f / sqrtf((float)R)), *injw = rand_tensor((size_t)C * W, 1.0f / sqrtf((float)W));
    q36_tensor *norm = q36_tensor_new((size_t)S * W * 4), *low0 = q36_tensor_new((size_t)S * R * 4);
    q36_tensor *low = q36_tensor_new((size_t)S * R * 4), *mix0 = q36_tensor_new((size_t)S * W * 4);
    q36_tensor *mixed = q36_tensor_new((size_t)S * H * 4), *inj0 = q36_tensor_new((size_t)S * C * 4);
    q36_tensor *inj = q36_tensor_new((size_t)S * C * 4);

    /* double reference, mirroring q38_gr_read */
    double *nm = malloc(sizeof(double) * S * W), *lo = malloc(sizeof(double) * S * R), *mx = malloc(sizeof(double) * S * W);
    double *rmixed = malloc(sizeof(double) * S * H), *rinj = malloc(sizeof(double) * S * C);
    double *smix = malloc(sizeof(double) * S * H), *sinj = malloc(sizeof(double) * S * C);
    double *xin = malloc(sizeof(double) * S * W);
    for (size_t i = 0; i < (size_t)S * W; i++) xin[i] = F(hyper)[i];
    for (int s = 0; s < S; s++)
        for (int b = 0; b < C; b++)
            ref_norm(nm + s * W + (size_t)b * H, xin + s * W + (size_t)b * H, F(nw) + (size_t)b * H, 1, H, eps);
    for (int s = 0; s < S; s++)
        for (int r = 0; r < R; r++) {
            double v = 0;
            for (size_t j = 0; j < W; j++) v += nm[s * W + j] * F(down)[(size_t)r * W + j];
            v /= C;
            lo[s * R + r] = v * ref_sigmoid(v);
        }
    for (int s = 0; s < S; s++)
        for (size_t o = 0; o < W; o++) {
            double v = 0;
            for (int r = 0; r < R; r++) v += lo[s * R + r] * F(up)[o * R + r];
            mx[s * W + o] = v;
        }
    for (int s = 0; s < S; s++)
        for (int d = 0; d < H; d++) {
            double v = 0, m = 0;
            for (int b = 0; b < C; b++) {
                double t = ref_sigmoid(mx[s * W + (size_t)b * H + d]) * nm[s * W + (size_t)b * H + d];
                v += t; m += fabs(t);
            }
            rmixed[(size_t)s * H + d] = v / C;
            smix[(size_t)s * H + d] = m / C + 1e-3;
        }
    for (int s = 0; s < S; s++)
        for (int c = 0; c < C; c++) {
            double v = 0;
            for (size_t j = 0; j < W; j++) v += nm[s * W + j] * F(injw)[(size_t)c * W + j];
            rinj[s * C + c] = 2.0 * ref_sigmoid(v / C);
            sinj[s * C + c] = fabs(rinj[s * C + c]) + 1e-3;
        }

    int ok = q38_hc_norm(norm, hyper, nw, S, H, C, eps) &&
             q36_dot_f32(low0, norm, down, S, (int)W, R) && q38_hc_low(low, low0, S * R, (float)C) &&
             q36_dot_f32(mix0, low, up, S, R, (int)W) && q38_hc_mix(mixed, mix0, norm, S, H, C, (float)C) &&
             q36_dot_f32(inj0, norm, injw, S, (int)W, C) && q38_hc_inj(inj, inj0, S * C, (float)C);
    sync_or_die();
    char name[80];
    snprintf(name, sizeof name, "gr_read composite S%d H%d %s", S, H, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else {
        /* the dots feed sigmoid/silu: scale by the output magnitude (mixed: sum of terms) */
        check(name, F(mixed), rmixed, smix, (size_t)S * H, exact ? 2e-5 : 3e-4);
        strcat(name, " inj");
        check(name, F(inj), rinj, sinj, (size_t)S * C, exact ? 2e-5 : 3e-4);
    }
    free(nm); free(lo); free(mx); free(rmixed); free(rinj); free(smix); free(sinj); free(xin);
    q36_tensor *ts[] = {hyper, nw, down, up, injw, norm, low0, low, mix0, mixed, inj0, inj};
    for (size_t i = 0; i < sizeof ts / sizeof *ts; i++) q36_tensor_free(ts[i]);
}

/* ---- PLE gate and dilated conv -------------------------------------------- */

/* qwen38_core.h's q38_ple from the gate on, ported literally in double. W = C*H. */
static void ref_rms0(double *out, const double *x, const float *w, int n, double eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * (1.0 + w[i]);
}

static void test_ple_gate(int S, int H, int C, int exact) {
    q36_gpu_set_exact(exact);
    const int W = C * H;
    const float eps = 1e-6f;
    q38_ple_shape sh = {S, C, W, H, 4, 3, 0, eps, 0};
    size_t n = (size_t)S * W;
    q36_tensor *keys = rand_tensor(n, 2.0f), *hyper = rand_tensor(n, 3.0f), *value = rand_tensor((size_t)S * H, 1.0f);
    q36_tensor *prm = rand_tensor((size_t)3 * W, 0.5f);
    q36_tensor *gated = q36_tensor_new(n * 4), *normv = q36_tensor_new(n * 4);
    for (size_t i = 0; i < n; i++) F(gated)[i] = F(normv)[i] = NAN;
    const float *nk = F(prm), *nq = F(prm) + W, *nc = F(prm) + 2 * W;
    double *x = malloc(sizeof(double) * H), *kn = malloc(sizeof(double) * H), *qn = malloc(sizeof(double) * H);
    double *rg = malloc(sizeof(double) * n), *rn = malloc(sizeof(double) * n), *sg = malloc(sizeof(double) * n);
    for (int s = 0; s < S; s++)
        for (int b = 0; b < C; b++) {
            size_t o = (size_t)s * W + (size_t)b * H;
            for (int d = 0; d < H; d++) x[d] = F(keys)[o + d];
            ref_rms0(kn, x, nk + (size_t)b * H, H, eps);
            for (int d = 0; d < H; d++) x[d] = F(hyper)[o + d];
            ref_rms0(qn, x, nq + (size_t)b * H, H, eps);
            double dot = 0, mag = 0;
            for (int d = 0; d < H; d++) { dot += kn[d] * qn[d]; mag += fabs(kn[d] * qn[d]); }
            dot /= sqrt((double)H); mag /= sqrt((double)H);
            double root = sqrt(fmax(fabs(dot), 1e-6)), g = ref_sigmoid(copysign(root, dot));
            for (int d = 0; d < H; d++) {
                double v = F(value)[(size_t)s * H + d];
                rg[o + d] = g * v;
                /* Ruling 9: the dot's error is relative to sum |terms|; it reaches gated through
                 * sqrt (slope 1/(2 root)) and sigmoid (slope <= 1/4) */
                sg[o + d] = fabs(g * v) + 1e-3 + fabs(v) * 0.25 * mag / (2.0 * root);
            }
            ref_rms0(rn + o, rg + o, nc + (size_t)b * H, H, eps);
        }
    int ok = q38_ple_gate(gated, normv, keys, hyper, value, prm, &sh);
    sync_or_die();
    char name[80];
    snprintf(name, sizeof name, "ple_gate S%d H%d C%d %s gated", S, H, C, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(gated), rg, sg, n, exact ? 2e-5 : 3e-4);
    snprintf(name, sizeof name, "ple_gate S%d H%d C%d %s normv", S, H, C, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(normv), rn, NULL, n, exact ? 1e-5 : 3e-4);
    free(x); free(kn); free(qn); free(rg); free(rn); free(sg);
    q36_tensor *ts[] = {keys, hyper, value, prm, gated, normv};
    for (size_t i = 0; i < sizeof ts / sizeof *ts; i++) q36_tensor_free(ts[i]);
}

/* The conv on random gated/normv/ring/conv, rows in order, against q38_ple's loop in double:
 * out (written or accumulated), the ring after the last row and, for s < snap_rows, the ring
 * after row s. The ring and the snaps hold copies of normv: identical bits. */
static void test_ple_conv(int S, int H, int C, int snap_rows, int accumulate, int exact) {
    q36_gpu_set_exact(exact);
    const int W = C * H, CK = 4, NG = 3, SL = (CK - 1) * NG;
    q38_ple_shape sh = {S, C, W, H, CK, NG, snap_rows, 1e-6f, accumulate};
    size_t n = (size_t)S * W, nr = (size_t)W * SL;
    q36_tensor *gated = rand_tensor(n, 1.0f), *normv = rand_tensor(n, 2.0f), *ring = rand_tensor(nr, 2.0f);
    q36_tensor *conv = rand_tensor((size_t)W * CK, 0.5f), *out = rand_tensor(n, 3.0f);
    q36_tensor *snap = snap_rows ? q36_tensor_new((size_t)snap_rows * nr * 4) : NULL;
    if (!accumulate) for (size_t i = 0; i < n; i++) F(out)[i] = NAN;
    for (size_t i = 0; i < (size_t)snap_rows * nr; i++) F(snap)[i] = NAN;
    double *ro = malloc(sizeof(double) * n), *so = malloc(sizeof(double) * n), *rr = malloc(sizeof(double) * nr);
    double *rs = malloc(sizeof(double) * (snap_rows ? snap_rows : 1) * nr);
    for (size_t i = 0; i < nr; i++) rr[i] = F(ring)[i];
    const float *cw = F(conv), *nv = F(normv), *gt = F(gated);
    for (int s = 0; s < S; s++) {
        for (int d = 0; d < W; d++) {
            size_t i = (size_t)s * W + d;
            double t0 = (double)cw[(size_t)d * CK + CK - 1] * nv[i], a = t0, m = fabs(t0);
            for (int k = 0; k < CK - 1; k++) {
                double t = cw[(size_t)d * CK + k] * rr[(size_t)d * SL + k * NG];
                a += t; m += fabs(t);
            }
            double v = gt[i] + a * ref_sigmoid(a), base = accumulate ? F(out)[i] : 0.0;
            ro[i] = base + v;
            so[i] = fabs(base) + fabs(gt[i]) + fabs(v) + m + 1e-3;   /* Ruling 9: a is a CK-term sum */
            double *rgd = rr + (size_t)d * SL;
            for (int k = 0; k < SL - 1; k++) rgd[k] = rgd[k + 1];
            rgd[SL - 1] = nv[i];
        }
        if (s < snap_rows) memcpy(rs + (size_t)s * nr, rr, sizeof(double) * nr);
    }
    int ok = q38_ple_conv(out, ring, snap, gated, normv, conv, &sh);
    sync_or_die();
    char name[96], tag[64];
    snprintf(tag, sizeof tag, "S%d H%d C%d snap%d acc%d %s", S, H, C, snap_rows, accumulate, exact ? "exact" : "fast");
    snprintf(name, sizeof name, "ple_conv %s out", tag);
    if (!ok) expect(name, 0);
    else check(name, F(out), ro, so, n, exact ? 2e-5 : 3e-4);
    snprintf(name, sizeof name, "ple_conv %s ring", tag);
    if (!ok) expect(name, 0);
    else check(name, F(ring), rr, NULL, nr, 0.0);
    for (int s = 0; s < snap_rows; s++) {
        snprintf(name, sizeof name, "ple_conv %s snap[%d]", tag, s);
        if (!ok) expect(name, 0);
        else check(name, F(snap) + (size_t)s * nr, rs + (size_t)s * nr, NULL, nr, 0.0);
    }
    free(ro); free(so); free(rr); free(rs);
    q36_tensor *ts[] = {gated, normv, ring, conv, out};
    for (size_t i = 0; i < sizeof ts / sizeof *ts; i++) q36_tensor_free(ts[i]);
    if (snap) q36_tensor_free(snap);
}

static void test_ple_declines(void) {
    const int H = 8, C = 4, W = 32, S = 2;
    q36_tensor *big = rand_tensor(4096, 1.0f), *y = q36_tensor_new((size_t)S * W * 4), *z = q36_tensor_new((size_t)S * W * 4);
    q36_tensor *small = q36_tensor_new(16 * 4);
    q38_ple_shape ok_s = {S, C, W, H, 4, 3, 0, 1e-6f, 0};
    q38_ple_shape bad_w = ok_s, bad_eps = ok_s, bad_sl = ok_s, bad_snap = ok_s, bad_s = ok_s;
    bad_w.W = 24; bad_eps.eps = 0.0f; bad_sl.CK = 12; bad_snap.snap_rows = 1; bad_s.S = 0;
    q38_ple_shape over_snap = ok_s;
    over_snap.snap_rows = 3;
    int took = q38_ple_gate(y, z, big, big, big, big, &bad_w) + q38_ple_gate(y, z, big, big, big, big, &bad_eps) +
               q38_ple_gate(y, z, big, big, big, big, &bad_s) + q38_ple_gate(y, z, big, big, big, small, &ok_s) +
               q38_ple_gate(y, small, big, big, big, big, &ok_s) + q38_ple_gate(y, z, big, big, big, big, NULL) +
               q38_ple_conv(y, big, NULL, big, big, big, &bad_w) + q38_ple_conv(y, big, NULL, big, big, big, &bad_sl) +
               q38_ple_conv(y, big, NULL, big, big, big, &bad_snap) + q38_ple_conv(y, big, big, big, big, big, &over_snap) +
               q38_ple_conv(y, small, NULL, big, big, big, &ok_s) + q38_ple_conv(small, big, NULL, big, big, big, &ok_s) +
               q38_ple_conv(y, big, NULL, big, big, big, NULL);
    sync_or_die();
    int clean = 1;
    for (int i = 0; i < S * W; i++) clean = clean && F(y)[i] == 0.0f && F(z)[i] == 0.0f;
    expect("ple declines W!=C*H SL>32 snaps sizes", took == 0 && clean);
    q36_tensor_free(big); q36_tensor_free(y); q36_tensor_free(z); q36_tensor_free(small);
}

/* ---- QSA: block-key pooling, scoring, selection, sparse attention -------- */
/* References: qwen38_core.h's q38_attention (QSA 3482-3533, attention 3543-3590), q38_rope,
 * q38_rms0, q38_block_desc, ported into doubles. */

static const float QSA_THETA = 1e7f, QSA_EPS = 1e-6f;

/* q38_rope on doubles; the angle in float as the CPU makes it (powf base, float division).
 * mag (when not NULL) gets the rotated pairs' sum |terms| + 1e-3 (the default elsewhere). */
static void ref_rope(double *x, double *mag, int dim, int rotary, int pos, float theta) {
    int half = rotary / 2;
    if (mag)
        for (int d = 0; d < dim; d++) mag[d] = fabs(x[d]) + 1e-3;
    for (int i = 0; i < half; i++) {
        float base = powf(theta, (float)(2 * i) / rotary);
        float ang = (float)pos / base;
        double co = cos((double)ang), si = sin((double)ang), a = x[i], b = x[i + half];
        x[i] = a * co - b * si;
        x[i + half] = b * co + a * si;
        if (mag) {
            mag[i] = fabs(a * co) + fabs(b * si) + 1e-3;
            mag[i + half] = fabs(b * co) + fabs(a * si) + 1e-3;
        }
    }
    if (mag)   /* the non-rotated tail keeps |ref| + 1e-3 */
        for (int d = 2 * half; d < dim; d++) mag[d] = fabs(x[d]) + 1e-3;
}

static q38_qsa_shape qsa_shape(int IQ, int ID, int R, int budget, int rotary) {
    q38_qsa_shape s = {IQ, ID, R, budget, rotary, QSA_THETA, QSA_EPS};
    return s;
}

/* pk[b] = rope(rms0(mean_r ik[b*R+r], kn), b*R) for b in [b0, b0+nb). */
static void test_qsa_pool(int ID, int R, int rotary, int b0, int nb, int exact) {
    q36_gpu_set_exact(exact);
    q38_qsa_shape sh = qsa_shape(4, ID, R, 2048, rotary);
    int nbk = b0 + nb;
    q36_tensor *ik = rand_tensor((size_t)nbk * R * ID, 2.0f), *kn = rand_tensor((size_t)ID, 0.5f);
    q36_tensor *pk = q36_tensor_new((size_t)nbk * ID * sizeof(float));
    for (size_t i = 0; i < (size_t)nbk * ID; i++) F(pk)[i] = NAN;
    size_t n = (size_t)nb * ID;
    double *ref = malloc(n * sizeof(double)), *mag = malloc(n * sizeof(double));
    for (int k = 0; k < nb; k++) {
        int b = b0 + k;
        double *x = ref + (size_t)k * ID, ss = 0;
        for (int d = 0; d < ID; d++) {
            double v = 0;
            for (int r = 0; r < R; r++) v += (double)F(ik)[((size_t)b * R + r) * ID + d] / R;
            x[d] = v;
            ss += v * v;
        }
        double rr = 1.0 / sqrt(ss / ID + QSA_EPS);
        for (int d = 0; d < ID; d++) x[d] = x[d] * rr * (1.0 + F(kn)[d]);
        ref_rope(x, mag + (size_t)k * ID, ID, rotary, b * R, QSA_THETA);
    }
    int ok = q38_qsa_pool(pk, ik, kn, b0, nb, &sh);
    sync_or_die();
    char name[96];
    snprintf(name, sizeof name, "qsa_pool ID=%d R=%d b%d+%d %s", ID, R, b0, nb, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(pk) + (size_t)b0 * ID, ref, mag, n, exact ? 1e-5 : 3e-4);
    int untouched = 1;   /* blocks below b0 are not written */
    for (size_t i = 0; i < (size_t)b0 * ID; i++) untouched = untouched && isnan(F(pk)[i]);
    snprintf(name, sizeof name, "qsa_pool b%d+%d leaves below b0 %s", b0, nb, exact ? "exact" : "fast");
    if (b0) expect(name, untouched);
    free(ref); free(mag);
    q36_tensor_free(ik); q36_tensor_free(kn); q36_tensor_free(pk);
}

/* sc[s][b] = sum_h max(0, q[s][h] . pk[b]) / sqrt(ID), b < nblk(s); scale sum |q pk| / sqrt(ID). */
static void test_qsa_score(int IQ, int ID, int R, int S, int pos_base, int exact) {
    q36_gpu_set_exact(exact);
    q38_qsa_shape sh = qsa_shape(IQ, ID, R, 2048, ID / 2);
    int nbmax = (pos_base + S) / R;
    q36_tensor *q = rand_tensor((size_t)S * IQ * ID, 1.0f), *pk = rand_tensor((size_t)nbmax * ID, 1.0f);
    q36_tensor *sc = q36_tensor_new((size_t)S * nbmax * sizeof(float));
    for (size_t i = 0; i < (size_t)S * nbmax; i++) F(sc)[i] = NAN;
    size_t cnt = 0;
    double *ref = malloc((size_t)S * nbmax * sizeof(double)), *mag = malloc((size_t)S * nbmax * sizeof(double));
    float *got = malloc((size_t)S * nbmax * sizeof(float));
    int ok = q38_qsa_score(sc, q, pk, S, pos_base, nbmax, &sh);
    sync_or_die();
    int untouched = 1;
    for (int s = 0; s < S; s++) {
        int nblk = (pos_base + s + 1) / R;
        for (int b = 0; b < nbmax; b++) {
            if (b >= nblk) { untouched = untouched && isnan(F(sc)[(size_t)s * nbmax + b]); continue; }
            double score = 0, m = 0;
            for (int h = 0; h < IQ; h++) {
                double a = 0;
                for (int d = 0; d < ID; d++) {
                    double t = (double)F(q)[((size_t)s * IQ + h) * ID + d] * F(pk)[(size_t)b * ID + d];
                    a += t;
                    m += fabs(t);
                }
                if (a > 0) score += a;
            }
            ref[cnt] = score / sqrt((double)ID);
            mag[cnt] = m / sqrt((double)ID) + 1e-3;
            got[cnt++] = F(sc)[(size_t)s * nbmax + b];
        }
    }
    char name[96];
    snprintf(name, sizeof name, "qsa_score IQ=%d ID=%d S=%d pos%d %s", IQ, ID, S, pos_base, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, got, ref, mag, cnt, exact ? 2e-5 : 3e-4);
    snprintf(name, sizeof name, "qsa_score S=%d pos%d past nblk unwritten %s", S, pos_base, exact ? "exact" : "fast");
    expect(name, ok && untouched);
    free(ref); free(mag); free(got);
    q36_tensor_free(q); q36_tensor_free(pk); q36_tensor_free(sc);
}

typedef struct { float score; int block; } ref_block;
static int ref_block_desc(const void *aa, const void *bb) {   /* q38_block_desc */
    const ref_block *a = aa, *b = bb;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return a->block - b->block;
}

/* The CPU's selection of the row at pos (q38_attention 3519-3530): n positions into list,
 * in its order; *all set when take == blocks (the GPU's count -1). */
static int ref_select(int *list, int *all, const float *scr, int pos, int R, int budget) {
    int visible = pos + 1, blocks = visible / R, tail = blocks * R, cap = budget / R;
    int take = blocks < cap ? blocks : cap, n = 0;
    ref_block *rk = malloc(sizeof(ref_block) * (size_t)(blocks ? blocks : 1));
    for (int b = 0; b < blocks; b++) rk[b] = (ref_block){scr[b], b};
    if (blocks) qsort(rk, (size_t)blocks, sizeof(ref_block), ref_block_desc);
    for (int z = 0; z < take; z++)
        for (int r = 0; r < R; r++) list[n++] = rk[z].block * R + r;
    for (int t = tail; t < visible; t++) list[n++] = t;
    *all = take == blocks;
    free(rk);
    return n;
}

static int qsa_sel_row(int R, int budget) { return budget / R * R + R; }

/* q38_qsa_select over the scores sch (S x nbmax) against ref_select, row by row: identical
 * lists, and count -1 exactly where the CPU takes every block (its list then a permutation
 * of 0..pos). */
static void run_select(const char *what, const float *sch, int S, int pos_base, int nbmax, int R, int budget, int exact) {
    q36_gpu_set_exact(exact);
    q38_qsa_shape sh = qsa_shape(4, 4, R, budget, 2);
    int sel_row = qsa_sel_row(R, budget);
    q36_tensor *sc = q36_tensor_new((size_t)S * nbmax * sizeof(float) + 4);
    q36_tensor *sel = q36_tensor_new((size_t)S * sel_row * sizeof(int));
    memcpy(F(sc), sch, (size_t)S * nbmax * sizeof(float));
    int *gs = q36_tensor_data(sel);
    for (size_t i = 0; i < (size_t)S * sel_row; i++) gs[i] = 0x7eadbeef;
    int ok = q38_qsa_select(sel, sc, S, pos_base, nbmax, sel_row, &sh);
    sync_or_die();
    int *list = malloc(sizeof(int) * (size_t)sel_row), *seen = calloc((size_t)(pos_base + S + 1), sizeof(int));
    int same = ok, contig = 0, lists = 0;
    for (int s = 0; s < S && same; s++) {
        int pos = pos_base + s, all, n = ref_select(list, &all, sch + (size_t)s * nbmax, pos, R, budget);
        const int *g = gs + (size_t)s * sel_row;
        if (all) {
            memset(seen, 0, sizeof(int) * (size_t)(pos + 1));
            int perm = n == pos + 1;
            for (int i = 0; i < n && perm; i++) perm = list[i] >= 0 && list[i] <= pos && !seen[list[i]]++;
            same = g[0] == -1 && perm;
            contig++;
            if (!same) printf("    row %d pos %d: got count %d, want -1 (cpu n %d perm %d)\n", s, pos, g[0], n, perm);
        } else {
            same = g[0] == n && !memcmp(g + 1, list, sizeof(int) * (size_t)n);
            lists++;
            if (!same) {
                printf("    row %d pos %d: got count %d want %d\n", s, pos, g[0], n);
                for (int i = 0; i < n; i++)
                    if (g[1 + i] != list[i]) { printf("    first mismatch at %d: got %d ref %d\n", i, g[1 + i], list[i]); break; }
            }
        }
    }
    char name[128];
    snprintf(name, sizeof name, "%s S=%d pos%d R=%d bud=%d (%d list %d all) %s", what, S, pos_base, R, budget, lists,
             contig, exact ? "exact" : "fast");
    expect(name, same);
    free(list); free(seen);
    q36_tensor_free(sc); q36_tensor_free(sel);
}

/* kind 0: random scores >= 0, a fifth of them 0 (no head positive); 1: eight levels (many
 * ties across the cutoff); 2: all equal; 3: all equal but every seventh block higher. */
static void test_qsa_select(const char *what, int kind, int S, int pos_base, int R, int budget, int exact) {
    int nbmax = (pos_base + S) / R;
    if (nbmax < 1) nbmax = 1;
    float *sch = malloc(sizeof(float) * (size_t)S * nbmax);
    for (size_t i = 0; i < (size_t)S * nbmax; i++) {
        float u = frand();
        int b = (int)(i % (size_t)nbmax);
        sch[i] = kind == 0 ? (u < -0.6f ? 0.0f : (u + 1.0f) * 0.75f)
               : kind == 1 ? floorf((u + 1.0f) * 4.0f) * 0.125f
               : kind == 2 ? 0.5f
                           : (b % 7 == 3 ? 0.75f : 0.5f);
    }
    run_select(what, sch, S, pos_base, nbmax, R, budget, exact);
    free(sch);
}

/* The edge positions, one row each at pos: random scores. */
static void test_qsa_select_edges(const int *pos, int np, int R, int budget, int exact) {
    for (int i = 0; i < np; i++) {
        char what[48];
        snprintf(what, sizeof what, "qsa_select edge pos=%d", pos[i]);
        test_qsa_select(what, 0, 1, pos[i], R, budget, exact);
    }
}

/* The CPU's attention of one (row, head) over a list, in double: softmax(q.k / sqrt(D)) . v
 * then * sigmoid(gate); mag = sigmoid(gate) * sum_j p_j |v_j| + 1e-3 (Ruling 9). */
static void ref_attn_head(double *o, double *mag, const float *q, const float *g, const float *K, const float *V,
                          const int *list, int n, int D) {
    double *sc = malloc(sizeof(double) * (size_t)n), mx = -INFINITY, den = 0;
    for (int j = 0; j < n; j++) {
        double a = 0;
        for (int d = 0; d < D; d++) a += (double)q[d] * K[(size_t)list[j] * D + d];
        sc[j] = a / sqrt((double)D);
        if (sc[j] > mx) mx = sc[j];
    }
    for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); den += sc[j]; }
    for (int d = 0; d < D; d++) {
        double a = 0, m = 0;
        for (int j = 0; j < n; j++) {
            double t = sc[j] / den * V[(size_t)list[j] * D + d];
            a += t;
            m += fabs(t);
        }
        double sg = ref_sigmoid(g[d]);
        o[d] = a * sg;
        mag[d] = m * sg + 1e-3;
    }
    free(sc);
}

/* mode 0: the lists ref_select makes from random scores (count -1 where it takes every
 * block); mode 1: every row -1 (contiguous 0..pos). K and V random, rounded to f16 for kv_half
 * (the reference reads the rounded values). */
static void test_attn_sel(int S, int pos_base, int heads, int kv, int D, int R, int budget, int cap, int kv_half,
                          int mode, int exact) {
    q36_gpu_set_exact(exact);
    int sel_row = qsa_sel_row(R, budget), nbmax = (pos_base + S) / R + 1;
    size_t cn = (size_t)kv * cap * D, qn = (size_t)S * heads * D;
    float *Kf = malloc(sizeof(float) * cn), *Vf = malloc(sizeof(float) * cn);
    q36_tensor *kc = q36_tensor_new(cn * (kv_half ? 2 : 4)), *vc = q36_tensor_new(cn * (kv_half ? 2 : 4));
    for (size_t i = 0; i < cn; i++) {
        Kf[i] = frand();
        Vf[i] = frand() * 2.0f;
        if (kv_half) {
            uint16_t kb = f16_bits(Kf[i]), vb = f16_bits(Vf[i]);
            ((uint16_t *)q36_tensor_data(kc))[i] = kb;
            ((uint16_t *)q36_tensor_data(vc))[i] = vb;
            _Float16 hk, hv;
            memcpy(&hk, &kb, 2); memcpy(&hv, &vb, 2);
            Kf[i] = (float)hk; Vf[i] = (float)hv;
        } else {
            F(kc)[i] = Kf[i];
            F(vc)[i] = Vf[i];
        }
    }
    q36_tensor *q = rand_tensor(qn, 4.0f), *g = rand_tensor(qn, 3.0f), *out = q36_tensor_new(qn * sizeof(float));
    q36_tensor *sel = q36_tensor_new((size_t)S * sel_row * sizeof(int));
    int *gs = q36_tensor_data(sel), *full = malloc(sizeof(int) * (size_t)(pos_base + S + sel_row));
    float *scr = malloc(sizeof(float) * (size_t)nbmax);
    double *ref = malloc(sizeof(double) * qn), *mag = malloc(sizeof(double) * qn);
    int lists = 0;
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s, *row = gs + (size_t)s * sel_row, n = pos + 1, all = 1;
        for (int b = 0; b < nbmax; b++) scr[b] = (frand() + 1.0f) * 0.5f;
        if (mode == 0) n = ref_select(row + 1, &all, scr, pos, R, budget);
        if (all) {
            row[0] = -1;
            n = pos + 1;
            for (int t = 0; t < n; t++) full[t] = t;
        } else {
            row[0] = n;
            memcpy(full, row + 1, sizeof(int) * (size_t)n);
            lists++;
        }
        for (int h = 0; h < heads; h++) {
            size_t o = ((size_t)s * heads + h) * D;
            int kvh = h / (heads / kv);
            ref_attn_head(ref + o, mag + o, F(q) + o, F(g) + o, Kf + (size_t)kvh * cap * D, Vf + (size_t)kvh * cap * D,
                          full, n, D);
        }
    }
    int ok = q38_attn_sel(out, q, g, kc, vc, sel, S, pos_base, heads, kv, D, cap, sel_row, kv_half);
    sync_or_die();
    char name[128];
    snprintf(name, sizeof name, "attn_sel S=%d pos%d h%d/%d D%d %s%s (%d list) %s", S, pos_base, heads, kv, D,
             kv_half ? "f16 " : "", mode ? "contig" : "sel", lists, exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(out), ref, mag, qn, exact ? 2e-5 : 3e-4);
    free(Kf); free(Vf); free(full); free(scr); free(ref); free(mag);
    q36_tensor_free(kc); q36_tensor_free(vc); q36_tensor_free(q); q36_tensor_free(g); q36_tensor_free(out);
    q36_tensor_free(sel);
}

/* Bad shapes and sizes enqueue nothing. */
static void test_qsa_declines(void) {
    q38_qsa_shape ok = qsa_shape(2, 4, 2, 4, 2), bad;
    q36_tensor *big = q36_tensor_new(1 << 16), *y = q36_tensor_new(256), *small = q36_tensor_new(8);
    int took = 0;
    took += q38_qsa_pool(y, big, big, 0, 4, NULL);
    bad = ok; bad.ID = 300; took += q38_qsa_pool(y, big, big, 0, 1, &bad);           /* ID > 256 */
    bad = ok; bad.rotary = 3; took += q38_qsa_pool(y, big, big, 0, 1, &bad);         /* odd rotary */
    bad = ok; bad.rotary = 6; took += q38_qsa_pool(y, big, big, 0, 1, &bad);         /* rotary > ID */
    took += q38_qsa_pool(small, big, big, 0, 4, &ok);                                 /* pk too small */
    took += q38_qsa_pool(y, small, big, 0, 4, &ok);                                   /* ikraw too small */
    took += q38_qsa_pool(y, big, big, -1, 4, &ok);
    took += q38_qsa_score(y, big, big, 4, 9, 2, &ok);                                 /* nbmax < nblk */
    took += q38_qsa_score(small, big, big, 2, 5, 3, &ok);                             /* sc too small */
    took += q38_qsa_select(y, big, 2, 9, 5, 3, &ok);                                  /* sel_row < budget/R*R + R */
    bad = ok; bad.budget = 2 * 4097; took += q38_qsa_select(big, big, 1, 9, 5, 9000, &bad);   /* budget/R > 2048 */
    took += q38_qsa_select(small, big, 2, 9, 5, 6, &ok);                              /* sel too small */
    took += q38_attn_sel(y, big, big, big, big, big, 1, 3, 2, 1, 8, 2, 7, 0);         /* cap < pos + S */
    took += q38_attn_sel(y, big, big, big, big, big, 1, 1, 3, 2, 8, 4, 7, 0);         /* heads % kv */
    took += q38_attn_sel(y, big, big, big, big, big, 1, 1, 2, 1, 300, 4, 7, 0);       /* D > 256 */
    took += q38_attn_sel(y, big, big, small, big, big, 1, 1, 2, 1, 8, 4, 7, 1);       /* kc too small */
    sync_or_die();
    int clean = 1;
    for (int i = 0; i < 64; i++) clean = clean && F(y)[i] == 0.0f;
    expect("qsa declines bad shapes and sizes", took == 0 && clean);
    q36_tensor_free(big); q36_tensor_free(y); q36_tensor_free(small);
}

/* ---- attention glue: RoPE, q/gate split, KV and indexer append ----------- */
/* qwen38_core.h's q38_attention 3464-3504 (projections, K/V/IK writes, indexer queries) and
 * 3568-3592 (q = rope(rms0(qraw[0..D)), gate = qraw[D..2D)), q38_rope, kv16_round. */

/* x rows x heads x D roped in place at pos_base + r; ref_rope per (row, head). */
static void test_rope_half(int rows, int heads, int D, int rotary, int pos_base, int exact) {
    q36_gpu_set_exact(exact);
    size_t n = (size_t)rows * heads * D;
    q36_tensor *x = rand_tensor(n, 2.0f);
    double *ref = malloc(n * sizeof(double)), *mag = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) ref[i] = F(x)[i];
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < heads; h++) {
            size_t o = ((size_t)r * heads + h) * D;
            ref_rope(ref + o, mag + o, D, rotary, pos_base + r, QSA_THETA);
        }
    int ok = q38_rope_half(x, rows, heads, D, rotary, pos_base, QSA_THETA);
    sync_or_die();
    char name[96];
    snprintf(name, sizeof name, "rope_half %dx%dx%d rot%d pos%d %s", rows, heads, D, rotary, pos_base,
             exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(x), ref, mag, n, exact ? 1e-5 : 3e-4);
    free(ref); free(mag);
    q36_tensor_free(x);
}

/* qraw [rows][heads][q D | gate D] -> q, gate [rows][heads][D], bit for bit. */
static void test_qgate_split(int rows, int heads, int D, int exact) {
    q36_gpu_set_exact(exact);
    size_t n = (size_t)rows * heads * D;
    q36_tensor *raw = rand_tensor(2 * n, 3.0f), *q = q36_tensor_new(n * 4), *g = q36_tensor_new(n * 4);
    int ok = q38_qgate_split(q, g, raw, rows, heads, D);
    sync_or_die();
    int same = ok;
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < heads; h++) {
            const float *qr = F(raw) + ((size_t)r * heads + h) * 2 * D;   /* qp + (s*QH + h)*2D */
            size_t o = ((size_t)r * heads + h) * D;
            same = same && !memcmp(F(q) + o, qr, (size_t)D * 4) && !memcmp(F(g) + o, qr + D, (size_t)D * 4);
        }
    char name[96];
    snprintf(name, sizeof name, "qgate_split %dx%dx%d %s", rows, heads, D, exact ? "exact" : "fast");
    expect(name, same);
    q36_tensor_free(raw); q36_tensor_free(q); q36_tensor_free(g);
}

/* The CPU's kv16_round: clamp to +-65504, round to nearest even. */
static uint16_t kv16_ref(float f) { return f16_bits(f > 65504.0f ? 65504.0f : f < -65504.0f ? -65504.0f : f); }

/* k, v rows x kv x D into [kv][cap][D] at pos_base + r; every other cache entry untouched. half:
 * the values include past-range, subnormal and tie cases. */
static void test_kv_append(int rows, int pos_base, int kv, int D, int cap, int half, int exact) {
    q36_gpu_set_exact(exact);
    size_t n = (size_t)rows * kv * D, cn = (size_t)kv * cap * D, es = half ? 2 : 4;
    q36_tensor *k = rand_tensor(n, 4.0f), *v = rand_tensor(n, 4.0f);
    for (size_t i = 0; i < n; i += 7) {   /* edges for the f16 rounding */
        static const float edge[6] = {70000.0f, -65520.0f, 65519.0f, 3e-6f, -1e-8f, 1.0f + 1.0f / 2048.0f};
        F(k)[i] = edge[(i / 7) % 6];
        F(v)[i] = -edge[(i / 7 + 3) % 6];
    }
    q36_tensor *kc = q36_tensor_new(cn * es), *vc = q36_tensor_new(cn * es);
    memset(q36_tensor_data(kc), 0xa5, cn * es);
    memset(q36_tensor_data(vc), 0x5a, cn * es);
    int ok = q38_kv_append(kc, vc, k, v, rows, pos_base, kv, D, cap, half);
    sync_or_die();
    int same = ok;
    for (int h = 0; h < kv; h++)
        for (int p = 0; p < cap; p++)
            for (int d = 0; d < D; d++) {
                size_t c = ((size_t)h * cap + p) * D + d;   /* Kl + (h*kv_cap + pos)*D */
                int r = p - pos_base, in = r >= 0 && r < rows;
                size_t s = ((size_t)r * kv + h) * D + d;
                for (int w = 0; w < 2; w++) {
                    const uint8_t *cb = (const uint8_t *)q36_tensor_data(w ? vc : kc) + c * es;
                    if (!in) {
                        for (size_t b = 0; b < es; b++) same = same && cb[b] == (w ? 0x5a : 0xa5);
                        continue;
                    }
                    float x = F(w ? v : k)[s];
                    if (half) {
                        uint16_t want = kv16_ref(x), got;
                        memcpy(&got, cb, 2);
                        same = same && got == want;
                    } else {
                        same = same && !memcmp(cb, &x, 4);
                    }
                }
            }
    char name[96];
    snprintf(name, sizeof name, "kv_append %d rows at %d kv%d D%d cap%d %s %s", rows, pos_base, kv, D, cap,
             half ? "f16" : "f32", exact ? "exact" : "fast");
    expect(name, same);
    q36_tensor_free(k); q36_tensor_free(v); q36_tensor_free(kc); q36_tensor_free(vc);
}

/* ip rows x (IQ+1)*ID: q[r][h] = rope(rms0(ip[r][h], qn), pos), ikc[pos] = ip[r][IQ] raw. */
static void test_idx_prep(int rows, int pos_base, int IQ, int ID, int rotary, int cap, int exact) {
    q36_gpu_set_exact(exact);
    q38_qsa_shape sh = qsa_shape(IQ, ID, 4, 2048, rotary);
    size_t row = (size_t)(IQ + 1) * ID, qn = (size_t)rows * IQ * ID;
    q36_tensor *ip = rand_tensor((size_t)rows * row, 3.0f), *w = rand_tensor((size_t)ID, 0.5f);
    q36_tensor *q = q36_tensor_new(qn * 4), *ikc = q36_tensor_new((size_t)cap * ID * 4);
    for (size_t i = 0; i < (size_t)cap * ID; i++) F(ikc)[i] = NAN;
    double *ref = malloc(qn * sizeof(double)), *mag = malloc(qn * sizeof(double)), *x = malloc(ID * sizeof(double));
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < IQ; h++) {
            size_t o = ((size_t)r * IQ + h) * ID;
            for (int d = 0; d < ID; d++) x[d] = F(ip)[(size_t)r * row + (size_t)h * ID + d];
            ref_rms0(ref + o, x, F(w), ID, QSA_EPS);
            ref_rope(ref + o, mag + o, ID, rotary, pos_base + r, QSA_THETA);
        }
    int ok = q38_idx_prep(q, ikc, ip, w, rows, pos_base, cap, &sh);
    sync_or_die();
    char name[112];
    snprintf(name, sizeof name, "idx_prep q %d rows at %d IQ%d ID%d rot%d %s", rows, pos_base, IQ, ID, rotary,
             exact ? "exact" : "fast");
    if (!ok) expect(name, 0);
    else check(name, F(q), ref, mag, qn, exact ? 1e-5 : 3e-4);
    int same = ok;   /* IKl + pos*ID = ip + s*(IQ+1)*ID + IQ*ID, bit for bit; the rest untouched */
    for (int p = 0; p < cap; p++) {
        int r = p - pos_base;
        for (int d = 0; d < ID; d++) {
            float got = F(ikc)[(size_t)p * ID + d];
            if (r >= 0 && r < rows) same = same && !memcmp(&got, F(ip) + (size_t)r * row + (size_t)IQ * ID + d, 4);
            else same = same && isnan(got);
        }
    }
    snprintf(name, sizeof name, "idx_prep ik %d rows at %d IQ%d ID%d %s", rows, pos_base, IQ, ID,
             exact ? "exact" : "fast");
    expect(name, same);
    free(ref); free(mag); free(x);
    q36_tensor_free(ip); q36_tensor_free(w); q36_tensor_free(q); q36_tensor_free(ikc);
}

/* Bad shapes and sizes enqueue nothing. */
static void test_glue_declines(void) {
    q38_qsa_shape ok = qsa_shape(2, 4, 4, 2048, 4), bad;
    q36_tensor *big = q36_tensor_new(1 << 16), *y = q36_tensor_new(256), *small = q36_tensor_new(8);
    memset(q36_tensor_data(big), 0, 1 << 16);
    int took = 0;
    took += q38_rope_half(y, 1, 2, 8, 3, 0, QSA_THETA);              /* odd rotary */
    took += q38_rope_half(y, 1, 2, 8, 10, 0, QSA_THETA);             /* rotary > D */
    took += q38_rope_half(big, 1, 1, 512, 258, 0, QSA_THETA);        /* rotary > 256 */
    took += q38_rope_half(small, 1, 2, 8, 4, 0, QSA_THETA);          /* x too small */
    took += q38_rope_half(y, 1, 2, 8, 4, -1, QSA_THETA);
    took += q38_rope_half(y, 0, 2, 8, 4, 0, QSA_THETA);
    took += q38_qgate_split(y, y, small, 1, 2, 8);                     /* qraw too small */
    took += q38_qgate_split(small, y, big, 1, 2, 8);                   /* q too small */
    took += q38_qgate_split(y, small, big, 1, 2, 8);                   /* gate too small */
    took += q38_kv_append(y, y, big, big, 2, 7, 1, 8, 8, 0);           /* pos_base + rows > cap */
    took += q38_kv_append(y, y, big, big, 1, -1, 1, 8, 8, 0);
    took += q38_kv_append(small, y, big, big, 1, 0, 1, 8, 8, 1);       /* kc too small for halves */
    took += q38_kv_append(big, y, big, big, 1, 0, 1, 8, 16, 0);        /* vc too small for floats */
    took += q38_kv_append(big, big, small, big, 1, 0, 1, 8, 16, 0);    /* k too small */
    took += q38_idx_prep(y, big, big, big, 1, 0, 4, NULL);
    bad = ok; bad.ID = 300; took += q38_idx_prep(y, big, big, big, 1, 0, 4, &bad);   /* ID > 256 */
    bad = ok; bad.rotary = 6; took += q38_idx_prep(y, big, big, big, 1, 0, 4, &bad); /* rotary > ID */
    took += q38_idx_prep(y, big, big, big, 2, 3, 4, &ok);                             /* pos_base + rows > cap */
    took += q38_idx_prep(y, small, big, big, 1, 0, 4, &ok);                           /* ikc too small */
    took += q38_idx_prep(small, big, big, big, 1, 0, 4, &ok);                         /* q too small */
    took += q38_idx_prep(y, big, small, big, 1, 0, 4, &ok);                           /* ip too small */
    sync_or_die();
    int clean = 1;
    for (int i = 0; i < 64; i++) clean = clean && F(y)[i] == 0.0f;
    expect("glue declines bad shapes and sizes", took == 0 && clean);
    q36_tensor_free(big); q36_tensor_free(y); q36_tensor_free(small);
}

/* ---- exact mode is deterministic: every op twice, same bits ------------------ */

static void fill_ff(q36_tensor *t, size_t bytes) { memset(q36_tensor_data(t), 0xff, bytes); }

/* pass 0 keeps a copy of p, pass 1 compares: byte-identical (same_bits) or FAIL. */
static void det_step(const char *name, int pass, void **saved, const void *p, size_t bytes, int ok) {
    if (pass == 0) {
        *saved = ok ? malloc(bytes) : NULL;
        if (*saved) memcpy(*saved, p, bytes);
        return;
    }
    expect(name, ok && *saved && memcmp(*saved, p, bytes) == 0);
    free(*saved);
}

static void test_exact_twice(void) {
    q36_gpu_set_exact(1);
    void *sv = NULL, *sv2 = NULL, *sv3 = NULL;
    int ok;

    {   /* dot_bf16: 4 rows, 2560 -> 640 */
        int rows = 4, in = 2560, out = 640;
        q36_tensor *x = rand_tensor((size_t)rows * in, 2.0f), *w = q36_tensor_new((size_t)out * in * 2);
        q36_tensor *y = q36_tensor_new((size_t)rows * out * 4);
        for (size_t i = 0; i < (size_t)out * in; i++) { float f = frand(); uint32_t b; memcpy(&b, &f, 4); ((uint16_t *)q36_tensor_data(w))[i] = (uint16_t)(b >> 16); }
        for (int pass = 0; pass < 2; pass++) {
            fill_ff(y, (size_t)rows * out * 4);
            ok = q38_dot_bf16(y, x, w, rows, in, out); sync_or_die();
            det_step("twice dot_bf16", pass, &sv, F(y), (size_t)rows * out * 4, ok);
        }
        q36_tensor_free(x); q36_tensor_free(w); q36_tensor_free(y);
    }
    {   /* hc_low, hc_inj, hc_mix, hc_apply, hc_norm */
        int rows = 4, D = 2560, C = 4;
        size_t W = (size_t)C * D, n = 1280;
        q36_tensor *a = rand_tensor((size_t)rows * W, 4.0f), *b = rand_tensor((size_t)rows * W, 2.0f);
        q36_tensor *y = q36_tensor_new((size_t)rows * W * 4), *h = q36_tensor_new((size_t)rows * W * 4);
        q36_tensor *inj = rand_tensor((size_t)rows * C, 2.0f), *blk = rand_tensor((size_t)rows * D, 2.0f);
        q36_tensor *hs = rand_tensor((size_t)rows * W, 3.0f), *nw = rand_tensor(W, 0.5f);
        for (int pass = 0; pass < 2; pass++) {
            fill_ff(y, n * 4);
            ok = q38_hc_low(y, a, (int)n, 4.0f); sync_or_die();
            det_step("twice hc_low", pass, &sv, F(y), n * 4, ok);
            fill_ff(y, n * 4);
            ok = q38_hc_inj(y, a, (int)n, 4.0f); sync_or_die();
            det_step("twice hc_inj", pass, &sv2, F(y), n * 4, ok);
            fill_ff(y, (size_t)rows * D * 4);
            ok = q38_hc_mix(y, a, b, rows, D, C, 4.0f); sync_or_die();
            det_step("twice hc_mix", pass, &sv3, F(y), (size_t)rows * D * 4, ok);
        }
        for (int pass = 0; pass < 2; pass++) {
            memcpy(F(h), F(hs), (size_t)rows * W * 4);
            ok = q38_hc_apply(h, inj, blk, rows, D, C); sync_or_die();
            det_step("twice hc_apply", pass, &sv, F(h), (size_t)rows * W * 4, ok);
            fill_ff(y, (size_t)rows * W * 4);
            ok = q38_hc_norm(y, a, nw, rows, D, C, 1e-6f); sync_or_die();
            det_step("twice hc_norm", pass, &sv2, F(y), (size_t)rows * W * 4, ok);
        }
        q36_tensor_free(a); q36_tensor_free(b); q36_tensor_free(y); q36_tensor_free(h);
        q36_tensor_free(inj); q36_tensor_free(blk); q36_tensor_free(hs); q36_tensor_free(nw);
    }
    {   /* ple_gate, ple_conv: H 2560, C 4, convk 4, ngram 3 */
        int S = 4, H = 2560, C = 4, SL = 9, snap_rows = 3;
        size_t W = (size_t)C * H, n = (size_t)S * W, nr = W * SL;
        q38_ple_shape sh = {S, C, (int)W, H, 4, 3, snap_rows, 1e-6f, 1};
        q36_tensor *keys = rand_tensor(n, 2.0f), *hyper = rand_tensor(n, 3.0f), *value = rand_tensor((size_t)S * H, 1.0f);
        q36_tensor *prm = rand_tensor(3 * W, 0.5f), *gated = q36_tensor_new(n * 4), *normv = q36_tensor_new(n * 4);
        q36_tensor *conv = rand_tensor(W * 4, 0.5f), *out0 = rand_tensor(n, 3.0f), *ring0 = rand_tensor(nr, 2.0f);
        q36_tensor *out = q36_tensor_new(n * 4), *ring = q36_tensor_new(nr * 4), *snap = q36_tensor_new((size_t)snap_rows * nr * 4);
        for (int pass = 0; pass < 2; pass++) {
            fill_ff(gated, n * 4); fill_ff(normv, n * 4);
            ok = q38_ple_gate(gated, normv, keys, hyper, value, prm, &sh); sync_or_die();
            det_step("twice ple_gate gated", pass, &sv, F(gated), n * 4, ok);
            det_step("twice ple_gate normv", pass, &sv2, F(normv), n * 4, ok);
        }
        for (int pass = 0; pass < 2; pass++) {
            memcpy(F(out), F(out0), n * 4); memcpy(F(ring), F(ring0), nr * 4); fill_ff(snap, (size_t)snap_rows * nr * 4);
            ok = q38_ple_conv(out, ring, snap, gated, normv, conv, &sh); sync_or_die();
            det_step("twice ple_conv out", pass, &sv, F(out), n * 4, ok);
            det_step("twice ple_conv ring", pass, &sv2, F(ring), nr * 4, ok);
            det_step("twice ple_conv snap", pass, &sv3, F(snap), (size_t)snap_rows * nr * 4, ok);
        }
        q36_tensor_free(keys); q36_tensor_free(hyper); q36_tensor_free(value); q36_tensor_free(prm); q36_tensor_free(gated);
        q36_tensor_free(normv); q36_tensor_free(conv); q36_tensor_free(out0); q36_tensor_free(ring0);
        q36_tensor_free(out); q36_tensor_free(ring); q36_tensor_free(snap);
    }
    {   /* QSA: pool, score, select, attention. IQ 4, ID 128, R 4, budget 256 (64 blocks), 24 heads, kv 2, D 256 */
        int ID = 128, R = 4, budget = 256, S = 3, pos_base = 1200, nb = 300, heads = 24, kv = 2, D = 256, cap = 1300;
        int nbmax = (pos_base + S) / R + 1, sel_row = qsa_sel_row(R, budget);
        q38_qsa_shape sh = qsa_shape(4, ID, R, budget, 64);
        q36_tensor *ik = rand_tensor((size_t)nb * R * ID, 2.0f), *kn = rand_tensor(ID, 0.5f);
        q36_tensor *pk = q36_tensor_new((size_t)nbmax * ID * 4), *q = rand_tensor((size_t)S * 4 * ID, 1.0f);
        q36_tensor *pk2 = rand_tensor((size_t)nbmax * ID, 1.0f), *sc = q36_tensor_new((size_t)S * nbmax * 4);
        q36_tensor *scq = q36_tensor_new((size_t)S * nbmax * 4), *sel = q36_tensor_new((size_t)S * sel_row * 4);
        for (size_t i = 0; i < (size_t)S * nbmax; i++) F(scq)[i] = floorf(fabsf(frand()) * 16.0f) / 16.0f;   /* ties */
        for (int pass = 0; pass < 2; pass++) {
            fill_ff(pk, (size_t)nb * ID * 4);
            ok = q38_qsa_pool(pk, ik, kn, 0, nb, &sh); sync_or_die();
            det_step("twice qsa_pool", pass, &sv, F(pk), (size_t)nb * ID * 4, ok);
            fill_ff(sc, (size_t)S * nbmax * 4);
            ok = q38_qsa_score(sc, q, pk2, S, pos_base, nbmax, &sh); sync_or_die();
            det_step("twice qsa_score", pass, &sv2, F(sc), (size_t)S * nbmax * 4, ok);
            fill_ff(sel, (size_t)S * sel_row * 4);
            ok = q38_qsa_select(sel, scq, S, pos_base, nbmax, sel_row, &sh); sync_or_die();
            det_step("twice qsa_select", pass, &sv3, q36_tensor_data(sel), (size_t)S * sel_row * 4, ok);
        }
        size_t cn = (size_t)kv * cap * D, qn = (size_t)S * heads * D;
        q36_tensor *kc = rand_tensor(cn, 1.0f), *vc = rand_tensor(cn, 2.0f);
        q36_tensor *aq = rand_tensor(qn, 4.0f), *g = rand_tensor(qn, 3.0f), *ao = q36_tensor_new(qn * 4);
        for (int pass = 0; pass < 2; pass++) {
            fill_ff(ao, qn * 4);
            ok = q38_attn_sel(ao, aq, g, kc, vc, sel, S, pos_base, heads, kv, D, cap, sel_row, 0); sync_or_die();
            det_step("twice attn_sel", pass, &sv, F(ao), qn * 4, ok);
        }
        q36_tensor_free(ik); q36_tensor_free(kn); q36_tensor_free(pk); q36_tensor_free(q); q36_tensor_free(pk2);
        q36_tensor_free(sc); q36_tensor_free(scq); q36_tensor_free(sel); q36_tensor_free(kc); q36_tensor_free(vc);
        q36_tensor_free(aq); q36_tensor_free(g); q36_tensor_free(ao);
    }
    {   /* attention glue (Task 4): rope_half, qgate_split, kv_append (floats and halves),
         * idx_prep. 24 heads, kv 2, D 256, rotary 64, theta 1e7; IQ 4, ID 128; 3 rows at 1200 */
        int rows = 3, heads = 24, kv = 2, D = 256, rot = 64, pb = 1200, cap = 1300, ID = 128;
        size_t qn = (size_t)rows * heads * D, kn = (size_t)rows * kv * D, cn = (size_t)kv * cap * D;
        q38_qsa_shape sh = qsa_shape(4, ID, 4, 256, rot);
        sh.theta = 1e7f;
        q36_tensor *x0 = rand_tensor(qn, 3.0f), *x = q36_tensor_new(qn * 4);
        q36_tensor *qraw = rand_tensor(2 * qn, 2.0f), *qq = q36_tensor_new(qn * 4), *gg = q36_tensor_new(qn * 4);
        q36_tensor *k = rand_tensor(kn, 2.0f), *v = rand_tensor(kn, 70000.0f);   /* v past the half range */
        q36_tensor *kc = q36_tensor_new(cn * 4), *vc = q36_tensor_new(cn * 4);
        q36_tensor *ip = rand_tensor((size_t)rows * 5 * ID, 2.0f), *iqn = rand_tensor(ID, 0.5f);
        q36_tensor *iq = q36_tensor_new((size_t)rows * 4 * ID * 4), *ikc = q36_tensor_new((size_t)cap * ID * 4);
        for (int pass = 0; pass < 2; pass++) {
            memcpy(F(x), F(x0), qn * 4);
            ok = q38_rope_half(x, rows, heads, D, rot, pb, 1e7f); sync_or_die();
            det_step("twice rope_half", pass, &sv, F(x), qn * 4, ok);
            fill_ff(qq, qn * 4); fill_ff(gg, qn * 4);
            ok = q38_qgate_split(qq, gg, qraw, rows, heads, D); sync_or_die();
            det_step("twice qgate_split q", pass, &sv2, F(qq), qn * 4, ok);
            det_step("twice qgate_split gate", pass, &sv3, F(gg), qn * 4, ok);
        }
        for (int half = 0; half < 2; half++)
            for (int pass = 0; pass < 2; pass++) {
                fill_ff(kc, cn * 4); fill_ff(vc, cn * 4);
                ok = q38_kv_append(kc, vc, k, v, rows, pb, kv, D, cap, half); sync_or_die();
                det_step(half ? "twice kv_append half k" : "twice kv_append k", pass, &sv, F(kc), cn * 4, ok);
                det_step(half ? "twice kv_append half v" : "twice kv_append v", pass, &sv2, F(vc), cn * 4, ok);
            }
        for (int pass = 0; pass < 2; pass++) {
            fill_ff(iq, (size_t)rows * 4 * ID * 4); fill_ff(ikc, (size_t)cap * ID * 4);
            ok = q38_idx_prep(iq, ikc, ip, iqn, rows, pb, cap, &sh); sync_or_die();
            det_step("twice idx_prep q", pass, &sv, F(iq), (size_t)rows * 4 * ID * 4, ok);
            det_step("twice idx_prep ikc", pass, &sv2, F(ikc), (size_t)cap * ID * 4, ok);
        }
        q36_tensor_free(x0); q36_tensor_free(x); q36_tensor_free(qraw); q36_tensor_free(qq); q36_tensor_free(gg);
        q36_tensor_free(k); q36_tensor_free(v); q36_tensor_free(kc); q36_tensor_free(vc);
        q36_tensor_free(ip); q36_tensor_free(iqn); q36_tensor_free(iq); q36_tensor_free(ikc);
    }
    {   /* wide router select and the lifted expert sum: 512 experts, top-k 10, H 2560 */
        int rows = 3, E = 512, K = 10, H = 2560;
        q36_tensor *lg = rand_tensor((size_t)rows * E, 3.0f), *prob = q36_tensor_new((size_t)rows * E * 4);
        q36_tensor *ids = q36_tensor_new((size_t)rows * K * 4), *wts = q36_tensor_new((size_t)rows * K * 4);
        for (size_t i = 0; i < (size_t)rows * E; i++) F(lg)[i] = floorf(F(lg)[i] * 4.0f) / 4.0f;   /* ties */
        for (int pass = 0; pass < 2; pass++) {
            fill_ff(ids, (size_t)rows * K * 4); fill_ff(wts, (size_t)rows * K * 4);
            ok = q36_router_softmax(prob, lg, NULL, rows, E) && q36_router_select(ids, wts, prob, rows, E, K, 1, 1); sync_or_die();
            det_step("twice router ids", pass, &sv, q36_tensor_data(ids), (size_t)rows * K * 4, ok);
            det_step("twice router weights", pass, &sv2, F(wts), (size_t)rows * K * 4, ok);
        }
        size_t cells = (size_t)rows * H;
        q36_tensor *y = rand_tensor(cells * K, 2.0f), *ew = rand_tensor((size_t)rows * K, 0.5f);
        q36_tensor *shd = rand_tensor(cells, 1.0f), *res = rand_tensor(cells, 4.0f), *eo = q36_tensor_new(cells * 4);
        for (int pass = 0; pass < 2; pass++) {
            fill_ff(eo, cells * 4);
            ok = q36_expert_sum(eo, y, ew, shd, res, rows, H, K); sync_or_die();
            det_step("twice expert_sum", pass, &sv3, F(eo), cells * 4, ok);
        }
        q36_tensor_free(lg); q36_tensor_free(prob); q36_tensor_free(ids); q36_tensor_free(wts);
        q36_tensor_free(y); q36_tensor_free(ew); q36_tensor_free(shd); q36_tensor_free(res); q36_tensor_free(eo);
    }
}

/* ---- QWEN38_BENCH=1: report-only microbench, median of 20 dispatches (each
 * enqueued and q36_gpu_sync'd), exact mode, decode (1 row) and a 4-row step. ---- */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
#define BENCH(label, ...) do { \
    double t_[20]; \
    for (int i_ = 0; i_ < 20; i_++) { double t0_ = now_ms(); if (!(__VA_ARGS__)) { puts("bench: call declined"); exit(1); } sync_or_die(); t_[i_] = now_ms() - t0_; } \
    qsort(t_, 20, sizeof(double), cmp_d); \
    printf("bench %-44s median %8.3f ms (min %.3f max %.3f)\n", label, (t_[9] + t_[10]) / 2, t_[0], t_[19]); \
} while (0)

static void bench_all(void) {
    q36_gpu_set_exact(1);
    for (int rows = 1; rows <= 4; rows += 3) {
        char l[96];
        int H = 2560, C = 4, D = 2560, out = 640, in = 2560;
        size_t W = (size_t)C * H;
        q36_tensor *x = rand_tensor((size_t)rows * in, 2.0f), *w = q36_tensor_new((size_t)out * in * 2), *y = q36_tensor_new((size_t)rows * W * 4);
        for (size_t i = 0; i < (size_t)out * in; i++) { float f = frand(); uint32_t b; memcpy(&b, &f, 4); ((uint16_t *)q36_tensor_data(w))[i] = (uint16_t)(b >> 16); }
        snprintf(l, sizeof l, "dot_bf16 rows %d 2560->640", rows);
        BENCH(l, q38_dot_bf16(y, x, w, rows, in, out));
        q36_tensor *a = rand_tensor((size_t)rows * W, 4.0f), *b = rand_tensor((size_t)rows * W, 2.0f);
        q36_tensor *inj = rand_tensor((size_t)rows * C, 2.0f), *blk = rand_tensor((size_t)rows * D, 2.0f), *nw = rand_tensor(W, 0.5f);
        snprintf(l, sizeof l, "hc_low rows %d x 320", rows);   BENCH(l, q38_hc_low(y, a, rows * 320, 4.0f));
        snprintf(l, sizeof l, "hc_inj rows %d x 4", rows);     BENCH(l, q38_hc_inj(y, a, rows * 4, 4.0f));
        snprintf(l, sizeof l, "hc_mix rows %d H2560 C4", rows);  BENCH(l, q38_hc_mix(y, a, b, rows, D, C, 4.0f));
        snprintf(l, sizeof l, "hc_apply rows %d H2560 C4", rows); BENCH(l, q38_hc_apply(a, inj, blk, rows, D, C));
        snprintf(l, sizeof l, "hc_norm rows %d H2560 C4", rows); BENCH(l, q38_hc_norm(y, b, nw, rows, D, C, 1e-6f));
        {
            int SL = 9, snap_rows = 0;
            size_t nr = W * SL, n = (size_t)rows * W;
            q38_ple_shape sh = {rows, C, (int)W, H, 4, 3, snap_rows, 1e-6f, 0};
            q36_tensor *keys = rand_tensor(n, 2.0f), *value = rand_tensor((size_t)rows * H, 1.0f), *prm = rand_tensor(3 * W, 0.5f);
            q36_tensor *gated = q36_tensor_new(n * 4), *normv = q36_tensor_new(n * 4), *conv = rand_tensor(W * 4, 0.5f);
            q36_tensor *ring = rand_tensor(nr, 2.0f), *po = q36_tensor_new(n * 4);
            snprintf(l, sizeof l, "ple_gate rows %d H2560 C4", rows); BENCH(l, q38_ple_gate(gated, normv, keys, a, value, prm, &sh));
            snprintf(l, sizeof l, "ple_conv rows %d W10240 k4 ng3", rows); BENCH(l, q38_ple_conv(po, ring, NULL, gated, normv, conv, &sh));
            q36_tensor_free(keys); q36_tensor_free(value); q36_tensor_free(prm); q36_tensor_free(gated);
            q36_tensor_free(normv); q36_tensor_free(conv); q36_tensor_free(ring); q36_tensor_free(po);
        }
        q36_tensor_free(x); q36_tensor_free(w); q36_tensor_free(y); q36_tensor_free(a); q36_tensor_free(b);
        q36_tensor_free(inj); q36_tensor_free(blk); q36_tensor_free(nw);
    }
    {   /* QSA at the real shapes: ID 128, IQ 4, R 4, budget 2048, 24 heads, kv 2, D 256, position 8996 */
        int ID = 128, R = 4, budget = 2048, heads = 24, kv = 2, D = 256, cap = 9008, pos_base = 8996;
        int sel_row = qsa_sel_row(R, budget);
        for (int S = 1; S <= 4; S += 3) {
            char l[96];
            int nbmax = (pos_base + S) / R + 1, nb = nbmax;
            q38_qsa_shape sh = qsa_shape(4, ID, R, budget, 64);
            q36_tensor *ik = rand_tensor((size_t)nb * R * ID, 2.0f), *kn = rand_tensor(ID, 0.5f), *pk = q36_tensor_new((size_t)nb * ID * 4);
            q36_tensor *q = rand_tensor((size_t)S * 4 * ID, 1.0f), *pk2 = rand_tensor((size_t)nbmax * ID, 1.0f), *sc = q36_tensor_new((size_t)S * nbmax * 4);
            q36_tensor *scq = q36_tensor_new((size_t)S * nbmax * 4), *sel = q36_tensor_new((size_t)S * sel_row * 4);
            for (size_t i = 0; i < (size_t)S * nbmax; i++) F(scq)[i] = fabsf(frand());
            snprintf(l, sizeof l, "qsa_pool 1 block (decode)");
            if (S == 1) BENCH(l, q38_qsa_pool(pk, ik, kn, nb - 1, 1, &sh));
            snprintf(l, sizeof l, "qsa_pool %d blocks", nb);
            if (S == 1) BENCH(l, q38_qsa_pool(pk, ik, kn, 0, nb, &sh));
            snprintf(l, sizeof l, "qsa_score S %d blocks %d", S, nbmax); BENCH(l, q38_qsa_score(sc, q, pk2, S, pos_base, nbmax, &sh));
            snprintf(l, sizeof l, "qsa_select S %d blocks %d budget 2048", S, nbmax); BENCH(l, q38_qsa_select(sel, scq, S, pos_base, nbmax, sel_row, &sh));
            size_t cn = (size_t)kv * cap * D, qn = (size_t)S * heads * D;
            q36_tensor *kc = rand_tensor(cn, 1.0f), *vc = rand_tensor(cn, 2.0f), *aq = rand_tensor(qn, 4.0f), *g = rand_tensor(qn, 3.0f), *ao = q36_tensor_new(qn * 4);
            snprintf(l, sizeof l, "attn_sel S %d 24h kv2 D256 list 2048", S);
            BENCH(l, q38_attn_sel(ao, aq, g, kc, vc, sel, S, pos_base, heads, kv, D, cap, sel_row, 0));
            q36_tensor_free(ik); q36_tensor_free(kn); q36_tensor_free(pk); q36_tensor_free(q); q36_tensor_free(pk2);
            q36_tensor_free(sc); q36_tensor_free(scq); q36_tensor_free(sel); q36_tensor_free(kc); q36_tensor_free(vc);
            q36_tensor_free(aq); q36_tensor_free(g); q36_tensor_free(ao);
        }
    }
    {   /* wide router select (512 experts, top-k 10) and the lifted expert sum (H 2560, top-k 10) */
        int rows = 1, E = 512, K = 10, H = 2560;
        q36_tensor *lg = rand_tensor((size_t)rows * E, 3.0f), *prob = q36_tensor_new((size_t)rows * E * 4);
        q36_tensor *ids = q36_tensor_new((size_t)rows * K * 4), *wts = q36_tensor_new((size_t)rows * K * 4);
        if (!q36_router_softmax(prob, lg, NULL, rows, E)) { puts("bench: softmax declined"); exit(1); }
        sync_or_die();
        BENCH("router_select_wide 1 row E512 K10", q36_router_select(ids, wts, prob, rows, E, K, 1, 1));
        size_t cells = (size_t)rows * H;
        q36_tensor *y = rand_tensor(cells * K, 2.0f), *ew = rand_tensor((size_t)rows * K, 0.5f), *shd = rand_tensor(cells, 1.0f);
        q36_tensor *res = rand_tensor(cells, 4.0f), *eo = q36_tensor_new(cells * 4);
        BENCH("expert_sum 1 row H2560 K10", q36_expert_sum(eo, y, ew, shd, res, rows, H, K));
        q36_tensor_free(lg); q36_tensor_free(prob); q36_tensor_free(ids); q36_tensor_free(wts);
        q36_tensor_free(y); q36_tensor_free(ew); q36_tensor_free(shd); q36_tensor_free(res); q36_tensor_free(eo);
    }
}

int main(void) {
    /* As tests/test_qwen36_kernels.c: GitHub's Apple Paravirtual device never completes
     * command buffers, so nothing runs there. */
    const char *device = q36_gpu_default_device_name();
    if (!*device) { puts("SKIPPED: no Metal device"); return 0; }
    if (strstr(device, "Apple Paravirtual device")) {
        printf("SKIPPED: %s: none of the kernels runs here\n", device);
        return 0;
    }
    if (!q38_gpu_init()) { puts("FAILED: no usable Metal device, or the kernels did not compile (see above)"); return 1; }
    printf("Metal device: %s\n", q36_gpu_device_name());

    for (int exact = 1; exact >= 0; exact--) {
        test_copy_interleaved(1, 2560, exact);      /* real: hidden 2560 */
        test_copy_interleaved(4, 2560, exact);
        test_copy_interleaved(3, 32, exact);        /* tiny fixture: hidden 32 */
    }
    test_copy_declines();

    for (int exact = 1; exact >= 0; exact--) {
        int rr[3] = {1, 3, 8};
        for (int i = 0; i < 3; i++) test_router_wide(rr[i], 512, 10, exact);   /* real: 512 experts, top-k 10 */
        test_router_wide(2, 512, 16, exact);
        test_router_wide(3, 4, 2, exact);                                        /* tiny fixture: 4 experts, top-k 2 */
        test_router_tie(2, 512, 10, exact);
        test_router_tie(1, 512, 16, exact);
        test_expert_sum(3, 2560, 10, exact);
        test_expert_sum(3, 32, 2, exact);
        test_expert_gq(2, 2560, 640, 10, exact);
    }
    test_router_wide_declines();

    for (int exact = 1; exact >= 0; exact--) {
        test_dot_bf16(2, 32, 12, exact);            /* tiny: IQ*ID + ID = 12 */
        int rr[3] = {1, 4, 64};
        for (int i = 0; i < 3; i++) test_dot_bf16(rr[i], 2560, 640, exact);   /* real: (4+1)*128 */
    }
    test_dot_bf16_declines();

    for (int exact = 1; exact >= 0; exact--) {
        test_hc_unary(0, 96, exact); test_hc_unary(1, 96, exact);       /* tiny: rows 3 x C 4 x ... */
        test_hc_mix(3, 32, 4, exact);
        test_hc_apply(3, 32, 4, exact);
        test_hc_norm(3, 32, 4, exact);
        int rr[2] = {1, 8};
        for (int i = 0; i < 2; i++) {                                    /* real: H 2560, C 4, rank 320 */
            test_hc_unary(0, rr[i] * 320, exact);
            test_hc_unary(1, rr[i] * 4, exact);
            test_hc_mix(rr[i], 2560, 4, exact);
            test_hc_apply(rr[i], 2560, 4, exact);
            test_hc_norm(rr[i], 2560, 4, exact);
            test_gr_read(rr[i], 2560, 4, 320, exact);
        }
        test_gr_read(3, 32, 4, 8, exact);
    }
    test_hc_declines();

    for (int exact = 1; exact >= 0; exact--) {
        test_ple_gate(3, 32, 4, exact);                     /* tiny: H 32, C 4 (W 128) */
        test_ple_conv(3, 32, 4, 0, 0, exact);
        test_ple_conv(3, 32, 4, 0, 1, exact);
        test_ple_conv(4, 32, 4, 3, 0, exact);               /* snapshots after rows 0..2 */
        int rr[2] = {1, 4};
        for (int i = 0; i < 2; i++) {                       /* real: H 2560, C 4 (W 10240), convk 4, ngram 3 */
            test_ple_gate(rr[i], 2560, 4, exact);
            test_ple_conv(rr[i], 2560, 4, 0, 0, exact);
            test_ple_conv(rr[i], 2560, 4, 0, 1, exact);
        }
        test_ple_conv(4, 2560, 4, 3, 1, exact);
    }
    test_ple_declines();

    for (int exact = 1; exact >= 0; exact--) {
        /* tiny: IQ 2, ID 4, R 2, budget 4, heads 2, kv 1, D 8 (rotary 2 = 0.25 * 8) */
        test_qsa_pool(4, 2, 2, 0, 6, exact);
        test_qsa_pool(4, 2, 2, 3, 2, exact);
        test_qsa_score(2, 4, 2, 3, 5, exact);
        test_qsa_select("qsa_select rand", 0, 4, 5, 2, 4, exact);
        test_qsa_select("qsa_select levels", 1, 4, 9, 2, 4, exact);
        test_qsa_select("qsa_select tie", 2, 2, 7, 2, 4, exact);
        int tiny_edges[5] = {0, 1, 1, 2, 5};                 /* 0, 1, R-1, R, 2R+1 */
        test_qsa_select_edges(tiny_edges, 5, 2, 4, exact);
        test_qsa_select("qsa_select rows 0..5", 0, 6, 0, 2, 4, exact);
        for (int kh = 0; kh < 2; kh++) {
            test_attn_sel(3, 5, 2, 1, 8, 2, 4, 16, kh, 0, exact);
            test_attn_sel(2, 1, 2, 1, 8, 2, 4, 16, kh, 1, exact);
        }
        /* real: IQ 4, ID 128 (rotary 64), R 4, budget 2048, heads 24, kv 2, D 256, positions to 9000 */
        test_qsa_pool(128, 4, 64, 0, 2250, exact);
        test_qsa_pool(128, 4, 64, 2240, 10, exact);
        test_qsa_score(4, 128, 4, 4, 8996, exact);
        test_qsa_score(4, 128, 4, 1, 100, exact);
        test_qsa_select("qsa_select rand", 0, 4, 8996, 4, 2048, exact);
        test_qsa_select("qsa_select levels", 1, 4, 8996, 4, 2048, exact);
        test_qsa_select("qsa_select rand", 0, 8, 2044, 4, 2048, exact);     /* 511..513 blocks: -1 and lists */
        test_qsa_select("qsa_select tie", 2, 2, 8996, 4, 2048, exact);
        test_qsa_select("qsa_select tie+7th", 3, 2, 8996, 4, 2048, exact);
        int real_edges[9] = {0, 1, 3, 4, 9, 2047, 2048, 2051, 2052};
        test_qsa_select_edges(real_edges, 9, 4, 2048, exact);
        for (int kh = 0; kh < 2; kh++) {
            test_attn_sel(4, 8996, 24, 2, 256, 4, 2048, 9008, kh, 0, exact);
            test_attn_sel(3, 2050, 24, 2, 256, 4, 2048, 9008, kh, 0, exact);
            test_attn_sel(2, 8999, 24, 2, 256, 4, 2048, 9008, kh, 1, exact);
        }
    }
    test_qsa_declines();

    for (int exact = 1; exact >= 0; exact--) {
        /* tiny: D 8, rotary 4, heads 2, kv 1; IQ 2, ID 4 */
        test_rope_half(3, 2, 8, 4, 0, exact);
        test_rope_half(3, 1, 8, 4, 5, exact);
        test_qgate_split(3, 2, 8, exact);
        for (int kh = 0; kh < 2; kh++) {
            test_kv_append(3, 0, 1, 8, 16, kh, exact);
            test_kv_append(3, 13, 1, 8, 16, kh, exact);       /* the last rows of the cache */
        }
        test_idx_prep(3, 5, 2, 4, 4, 16, exact);
        test_idx_prep(3, 5, 2, 4, 2, 16, exact);
        /* real: D 256, rotary 64, theta 1e7, heads 24, kv 2; IQ 4, ID 128; positions to 9000 */
        test_rope_half(1, 24, 256, 64, 8999, exact);
        test_rope_half(4, 24, 256, 64, 8996, exact);
        test_rope_half(64, 2, 256, 64, 0, exact);
        test_rope_half(2, 24, 256, 64, 99999, exact);          /* sampled: position 100,000 */
        test_qgate_split(1, 24, 256, exact);
        test_qgate_split(5, 24, 256, exact);
        for (int kh = 0; kh < 2; kh++) {
            test_kv_append(1, 8999, 2, 256, 9008, kh, exact);
            test_kv_append(4, 9004, 2, 256, 9008, kh, exact);  /* ends at cap */
            test_kv_append(7, 0, 2, 256, 9008, kh, exact);
        }
        test_idx_prep(1, 8999, 4, 128, 64, 9008, exact);
        test_idx_prep(4, 9004, 4, 128, 64, 9008, exact);
    }
    test_glue_declines();

    test_exact_twice();
    if (getenv("QWEN38_BENCH") && *getenv("QWEN38_BENCH") == '1') bench_all();

    printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    q36_gpu_cleanup();
    return g_failures ? 1 : 0;
}
