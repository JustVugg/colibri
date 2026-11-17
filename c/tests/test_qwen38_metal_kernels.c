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

static void test_hc_declines(void) {
    q36_tensor *a = rand_tensor(64, 1.0f), *y = q36_tensor_new(16 * 4);
    int ok = !q38_hc_low(y, a, 0, 4.f) && !q38_hc_low(y, a, 32, 4.f) && !q38_hc_inj(NULL, a, 4, 4.f) &&
             !q38_hc_mix(y, a, a, 2, 16, 4, 4.f) && !q38_hc_mix(y, a, a, 2, 8, 0, 4.f) &&
             !q38_hc_apply(a, y, y, 2, 16, 4);
    sync_or_die();
    for (int i = 0; i < 16; i++) ok = ok && F(y)[i] == 0.0f;
    expect("hc declines bad sizes", ok);
    q36_tensor_free(a); q36_tensor_free(y);
}

/* The CPU's q38_gr_read in double, with the dot products on the f32 weights:
 * rms0 per stream (weight w, shared across streams here: q36_rmsnorm takes one weight row),
 * low = silu((down.norm)/C), mix = up.low, mixed = sum_b sigmoid(mix)*norm / C,
 * inject = 2 sigmoid((inj.norm)/C). */
static void test_gr_read(int S, int H, int C, int R, int exact) {
    q36_gpu_set_exact(exact);
    const float eps = 1e-6f;
    size_t W = (size_t)C * H;
    q36_tensor *hyper = rand_tensor((size_t)S * W, 2.0f), *nw = rand_tensor((size_t)H, 0.5f);
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
    ref_norm(nm, xin, F(nw), S * C, H, eps);
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

    int ok = q36_rmsnorm(norm, hyper, nw, S * C, H, eps, 0) &&
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
        int rr[2] = {1, 8};
        for (int i = 0; i < 2; i++) {                                    /* real: H 2560, C 4, rank 320 */
            test_hc_unary(0, rr[i] * 320, exact);
            test_hc_unary(1, rr[i] * 4, exact);
            test_hc_mix(rr[i], 2560, 4, exact);
            test_hc_apply(rr[i], 2560, 4, exact);
            test_gr_read(rr[i], 2560, 4, 320, exact);
        }
        test_gr_read(3, 32, 4, 8, exact);
    }
    test_hc_declines();

    printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    q36_gpu_cleanup();
    return g_failures ? 1 : 0;
}
