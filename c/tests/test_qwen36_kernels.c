/* Metal kernel tests for Qwen3.6: every kernel, at the real model's shapes and
 * at small ones that leave tails in the loops, against a double-precision
 * reference on random data. No model, no checkpoint. Build and run:
 * make test-qwen36-kernels
 *
 * The tolerance is relative to the magnitude of the terms (sum of |a*b| for a
 * dot product), so it catches layout and index mistakes, not the last bit. Two
 * cases are tight enough to need the exact kernels' compensated sums (a long
 * delta rule, attention with one dominant score), and the row kernels and the
 * conv must give a row the same bits in batches of any size. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../qwen36_gpu.h"
#include "../qwen36_cpu_mm.h"

static uint32_t g_rng = 0x2545f491u;
static float frand(void) {                      /* uniform in [-1, 1) */
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float)(g_rng & 0xffffffu) / 8388608.0f - 1.0f;
}

static int g_failures;
static int g_test_exact;

static void test_set_exact(int on) {
    g_test_exact = !!on;
    q36_gpu_set_exact(g_test_exact);
}

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

static void check_ints(const char *what, const int32_t *got, const int32_t *ref, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) bad += got[i] != ref[i];
    printf("%-34s %s  mismatches=%zu of %zu\n", what, bad ? "FAIL" : "ok  ", bad, n);
    if (bad) g_failures++;
}

static q36_tensor *rand_tensor(size_t count, float scale) {
    q36_tensor *t = q36_tensor_new(count * sizeof(float));
    float *p = q36_tensor_data(t);
    for (size_t i = 0; i < count; i++) p[i] = frand() * scale;
    return t;
}

static q36_tensor *rand_i8(size_t count) {
    q36_tensor *t = q36_tensor_new(count);
    int8_t *p = q36_tensor_data(t);
    for (size_t i = 0; i < count; i++) p[i] = (int8_t)(frand() * 127.0f);
    return t;
}

static float *F(q36_tensor *t) { return q36_tensor_data(t); }

static void sync_or_die(void) {
    if (!q36_gpu_sync()) { fprintf(stderr, "GPU batch failed\n"); exit(1); }
}

/* ---- dense trunk --------------------------------------------------------- */

static void test_rmsnorm(int rows, int dim, int plain) {
    q36_tensor *x = rand_tensor((size_t)rows * dim, 3.0f), *w = rand_tensor((size_t)dim, 0.5f);
    q36_tensor *y = q36_tensor_new((size_t)rows * dim * 4);
    q36_rmsnorm(y, x, w, rows, dim, 1e-6f, plain);
    sync_or_die();
    double *ref = malloc(sizeof(double) * rows * dim);
    for (int r = 0; r < rows; r++) {
        double ss = 0;
        for (int j = 0; j < dim; j++) ss += (double)F(x)[r * dim + j] * F(x)[r * dim + j];
        double k = 1.0 / sqrt(ss / dim + 1e-6);
        for (int j = 0; j < dim; j++)
            ref[r * dim + j] = F(x)[r * dim + j] * k * (plain ? F(w)[j] : 1.0 + F(w)[j]);
    }
    char name[64];
    snprintf(name, sizeof name, "rmsnorm %dx%d%s", rows, dim, plain ? " plain" : "");
    check(name, F(y), ref, NULL, (size_t)rows * dim, 1e-5);
    free(ref);
    q36_tensor_free(x); q36_tensor_free(w); q36_tensor_free(y);
}

/* FAST reduction against an independent double sum. Compare full input/output
 * aliasing too: the norm reads must complete before any input is overwritten. */
static void test_rmsnorm_fast(int rows, int dim, int plain, int zero) {
    int saved_exact = g_test_exact;
    size_t n = (size_t)rows * dim;
    const float eps = 1e-6f;
    q36_tensor *x = rand_tensor(n, zero ? 0.0f : 3.0f), *w = rand_tensor((size_t)dim, 0.5f);
    q36_tensor *y = q36_tensor_new(n * sizeof(float));
    double *ref = malloc(n * sizeof(double));
    for (int r = 0; r < rows; r++) {
        size_t base = (size_t)r * dim;
        double ss = 0;
        for (int j = 0; j < dim; j++) {
            double v = F(x)[base + j];
            ss += v * v;
        }
        double scale = 1.0 / sqrt(ss / dim + (double)eps);
        for (int j = 0; j < dim; j++)
            ref[base + j] = F(x)[base + j] * scale * (plain ? F(w)[j] : 1.0 + F(w)[j]);
    }
    test_set_exact(0);
    int ok = q36_rmsnorm(y, x, w, rows, dim, eps, plain);
    ok = q36_rmsnorm(x, x, w, rows, dim, eps, plain) && ok;
    sync_or_die();
    char name[96];
    snprintf(name, sizeof name, "fast rmsnorm %dx%d%s%s", rows, dim, plain ? " plain" : " centered", zero ? " zero" : "");
    check(name, F(y), ref, NULL, n, 1e-5);
    snprintf(name, sizeof name, "fast rmsnorm inplace %dx%d", rows, dim);
    check(name, F(x), ref, NULL, n, 1e-5);
    int same = !memcmp(F(x), F(y), n * sizeof(float));
    printf("fast rmsnorm dispatch/inplace %dx%d %s\n", rows, dim, ok && same ? "ok" : "FAIL");
    g_failures += !ok + !same;
    free(ref);
    q36_tensor_free(x); q36_tensor_free(w); q36_tensor_free(y);
    test_set_exact(saved_exact);
}

static void test_dot(int rows, int in, int out, int router) {
    q36_tensor *q = rand_i8((size_t)out * in), *sc = rand_tensor((size_t)out, 0.01f);
    q36_tensor *x = rand_tensor((size_t)rows * in, 1.0f), *y = q36_tensor_new((size_t)rows * out * 4);
    if (router) q36_router_dot(y, x, q, sc, rows, in, out);
    else q36_dot_i8(y, x, q, sc, rows, in, out);
    sync_or_die();
    double *ref = malloc(sizeof(double) * rows * out), *mag = malloc(sizeof(double) * rows * out);
    const int8_t *qq = q36_tensor_data(q);
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < out; o++) {
            double s = 0, m = 0;
            for (int i = 0; i < in; i++) {
                double t = (double)qq[(size_t)o * in + i] * F(x)[(size_t)r * in + i];
                s += t;
                m += fabs(t);
            }
            ref[(size_t)r * out + o] = s * F(sc)[o];
            mag[(size_t)r * out + o] = m * fabs(F(sc)[o]) + 1e-12;
        }
    char name[64];
    snprintf(name, sizeof name, "%s %dx%d->%d", router ? "router_dot" : "dot_i8", rows, in, out);
    check(name, F(y), ref, mag, (size_t)rows * out, 3e-5);
    free(ref); free(mag);
    q36_tensor_free(q); q36_tensor_free(sc); q36_tensor_free(x); q36_tensor_free(y);
}

static void test_elementwise(int n) {
    q36_tensor *x = rand_tensor((size_t)n, 2.0f), *z = rand_tensor((size_t)n, 4.0f);
    q36_tensor *y0 = q36_tensor_new((size_t)n * 4), *y1 = q36_tensor_new((size_t)n * 4);
    q36_tensor *y2 = q36_tensor_new((size_t)n * 4), *y3 = q36_tensor_new((size_t)n * 4);
    q36_gate(y0, x, z, n, 0);
    q36_gate(y1, x, z, n, 1);
    q36_add(y2, x, z, n);
    q36_silu_mul(y3, z, x, n);           /* h = silu(g=z) * u=x */
    sync_or_die();
    double *r0 = malloc(sizeof(double) * n), *r1 = malloc(sizeof(double) * n);
    double *r2 = malloc(sizeof(double) * n), *r3 = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        double a = F(z)[i], sg = 1.0 / (1.0 + exp(-a));
        r0[i] = F(x)[i] * sg;
        r1[i] = F(x)[i] * sg * a;
        r2[i] = (double)F(x)[i] + F(z)[i];
        r3[i] = a * sg * F(x)[i];
    }
    check("gate sigmoid", F(y0), r0, NULL, (size_t)n, 1e-5);
    check("gate silu", F(y1), r1, NULL, (size_t)n, 1e-5);
    check("add", F(y2), r2, NULL, (size_t)n, 1e-6);
    check("silu_mul", F(y3), r3, NULL, (size_t)n, 1e-5);
    free(r0); free(r1); free(r2); free(r3);
    q36_tensor_free(x); q36_tensor_free(z);
    q36_tensor_free(y0); q36_tensor_free(y1); q36_tensor_free(y2); q36_tensor_free(y3);
}

/* ---- DeltaNet ---------------------------------------------------------------- */

static double softplus_d(double x) { return x > 20 ? x : log1p(exp(x)); }

/* One DeltaNet core over `rows` tokens, twice (two batches) so the state and
 * the conv history carry across calls, against a double reference. */
/* handoff: the first batch in fast mode, the second exact, as a prompt then
 * generation: the state the fast kernel leaves must read the same. */
static void test_deltanet(int rows, int H, int vh, int kh, int kd, int vd, int ck, int handoff) {
    int saved_exact = g_test_exact;
    int cd = 2 * kh * kd + vh * vd;
    q36_shape s = {0};
    s.rows = rows; s.hidden = H; s.v_heads = vh; s.k_heads = kh; s.k_dim = kd; s.v_dim = vd;
    s.conv_kernel = ck; s.conv_dim = cd;
    q36_tensor *x = rand_tensor((size_t)rows * H, 1.0f);
    q36_tensor *a = rand_tensor((size_t)vh * H, 0.05f), *b = rand_tensor((size_t)vh * H, 0.05f);
    q36_tensor *alog = rand_tensor((size_t)vh, 1.0f), *dt = rand_tensor((size_t)vh, 1.0f);
    q36_tensor *proj = rand_tensor((size_t)rows * cd, 1.0f), *w = rand_tensor((size_t)cd * ck, 0.5f);
    q36_tensor *ring = q36_tensor_new((size_t)cd * (ck - 1) * 4);
    q36_tensor *state = q36_tensor_new((size_t)2 * vh * kd * vd * 4);
    q36_tensor *decay = q36_tensor_new((size_t)rows * vh * 4), *beta = q36_tensor_new((size_t)rows * vh * 4);
    q36_tensor *conv = q36_tensor_new((size_t)rows * cd * 4), *qk = q36_tensor_new((size_t)rows * cd * 4);
    q36_tensor *out = q36_tensor_new((size_t)rows * vh * vd * 4);

    double *rs = calloc((size_t)vh * kd * vd, sizeof(double));      /* reference state */
    double *rring = calloc((size_t)cd * (ck - 1), sizeof(double));
    double *rdecay = malloc(sizeof(double) * rows * vh), *rbeta = malloc(sizeof(double) * rows * vh);
    double *rconv = malloc(sizeof(double) * rows * cd), *rqk = malloc(sizeof(double) * rows * cd);
    /* the conv and l2 errors are relative to the magnitude of the conv terms
     * (|ref| alone fails on outputs near zero, where the terms cancel) */
    double *sconv = malloc(sizeof(double) * rows * cd), *sqk = malloc(sizeof(double) * rows * cd);
    double *rout = malloc(sizeof(double) * rows * vh * vd);
    for (int pass = 0; pass < 2; pass++) {
        if (handoff) test_set_exact(pass);
        if (pass) {   /* new inputs for the second batch; state and ring carry over */
            for (size_t i = 0; i < (size_t)rows * H; i++) F(x)[i] = frand();
            for (size_t i = 0; i < (size_t)rows * cd; i++) F(proj)[i] = frand();
        }
        q36_dn_aux(decay, beta, x, a, b, alog, dt, &s);
        q36_dn_conv(conv, ring, proj, w, NULL, &s);
        q36_dn_l2(qk, conv, &s);
        q36_dn_delta(out, state, qk, conv, decay, beta, NULL, &s);
        sync_or_die();
        for (int r = 0; r < rows; r++) {
            for (int h = 0; h < vh; h++) {
                double av = 0, bv = 0;
                for (int i = 0; i < H; i++) {
                    av += (double)F(a)[(size_t)h * H + i] * F(x)[(size_t)r * H + i];
                    bv += (double)F(b)[(size_t)h * H + i] * F(x)[(size_t)r * H + i];
                }
                rdecay[r * vh + h] = exp(-exp((double)F(alog)[h]) * softplus_d(av + F(dt)[h]));
                rbeta[r * vh + h] = 1.0 / (1.0 + exp(-bv));
            }
            for (int c = 0; c < cd; c++) {
                double v = 0, terms = 0;
                for (int k = 0; k + 1 < ck; k++) {
                    v += (double)F(w)[(size_t)c * ck + k] * rring[(size_t)c * (ck - 1) + k];
                    terms += fabs((double)F(w)[(size_t)c * ck + k] * rring[(size_t)c * (ck - 1) + k]);
                }
                double raw = F(proj)[(size_t)r * cd + c];
                v += (double)F(w)[(size_t)c * ck + ck - 1] * raw;
                terms += fabs((double)F(w)[(size_t)c * ck + ck - 1] * raw);
                rconv[(size_t)r * cd + c] = v / (1.0 + exp(-v));
                sconv[(size_t)r * cd + c] = 1.1 * terms + 1e-6;     /* silu' is below 1.1 */
                for (int k = 0; k + 2 < ck; k++) rring[(size_t)c * (ck - 1) + k] = rring[(size_t)c * (ck - 1) + k + 1];
                rring[(size_t)c * (ck - 1) + ck - 2] = raw;
            }
            for (int head = 0; head < 2 * kh; head++) {
                double ss = 0;
                for (int k = 0; k < kd; k++) ss += rconv[(size_t)r * cd + head * kd + k] * rconv[(size_t)r * cd + head * kd + k];
                double scale = (head < kh ? 1.0 / sqrt((double)kd) : 1.0) / sqrt(ss + 1e-6);
                for (int k = 0; k < kd; k++) {
                    rqk[(size_t)r * cd + head * kd + k] = rconv[(size_t)r * cd + head * kd + k] * scale;
                    sqk[(size_t)r * cd + head * kd + k] = sconv[(size_t)r * cd + head * kd + k] * scale + 1e-6;
                }
            }
            for (int v = 0; v < vh * vd; v++) {
                rqk[(size_t)r * cd + 2 * kh * kd + v] = rconv[(size_t)r * cd + 2 * kh * kd + v];
                sqk[(size_t)r * cd + 2 * kh * kd + v] = sconv[(size_t)r * cd + 2 * kh * kd + v];
            }
            for (int h = 0; h < vh; h++) {
                int kk = h / (vh / kh);
                const double *q = rqk + (size_t)r * cd + kk * kd, *kv = q + kh * kd;
                for (int v = 0; v < vd; v++) {
                    double u = 0;
                    for (int k = 0; k < kd; k++) {
                        double *st = &rs[((size_t)h * kd + k) * vd + v];
                        *st *= rdecay[r * vh + h];
                        u += *st * kv[k];
                    }
                    double vin = rconv[(size_t)r * cd + 2 * kh * kd + h * vd + v];
                    double delta = (vin - u) * rbeta[r * vh + h], y = 0;
                    for (int k = 0; k < kd; k++) {
                        double *st = &rs[((size_t)h * kd + k) * vd + v];
                        *st += delta * kv[k];
                        y += q[k] * *st;
                    }
                    rout[((size_t)r * vh + h) * vd + v] = y;
                }
            }
        }
        char name[80];
        snprintf(name, sizeof name, "dn pass%d decay rows=%d vh=%d", pass, rows, vh);
        check(name, F(decay), rdecay, NULL, (size_t)rows * vh, 1e-5);
        snprintf(name, sizeof name, "dn pass%d beta", pass);
        check(name, F(beta), rbeta, NULL, (size_t)rows * vh, 1e-5);
        snprintf(name, sizeof name, "dn pass%d conv+silu cd=%d", pass, cd);
        check(name, F(conv), rconv, sconv, (size_t)rows * cd, 1e-5);
        snprintf(name, sizeof name, "dn pass%d l2 q/k, v copy", pass);
        check(name, F(qk), rqk, sqk, (size_t)rows * cd, 1e-5);
        snprintf(name, sizeof name, "dn pass%d delta output kd=%d vd=%d", pass, kd, vd);
        check(name, F(out), rout, NULL, (size_t)rows * vh * vd, 1e-4);
    }
    /* the low plane must be part of the state: high + low == reference */
    double *sum = malloc(sizeof(double) * vh * kd * vd);
    float *st = F(state), *hi = malloc(sizeof(float) * vh * kd * vd);
    size_t cells = (size_t)vh * kd * vd;
    for (size_t i = 0; i < cells; i++) { sum[i] = rs[i]; hi[i] = (float)((double)st[i] + (double)st[cells + i]); }
    check("dn state high+low vs reference", hi, sum, NULL, cells, 1e-4);
    free(sum); free(hi);
    if (handoff) test_set_exact(saved_exact);
    free(rs); free(rring); free(rdecay); free(rbeta); free(rconv); free(rqk); free(rout); free(sconv); free(sqk);
    q36_tensor_free(x); q36_tensor_free(a); q36_tensor_free(b); q36_tensor_free(alog); q36_tensor_free(dt);
    q36_tensor_free(proj); q36_tensor_free(w); q36_tensor_free(ring); q36_tensor_free(state);
    q36_tensor_free(decay); q36_tensor_free(beta); q36_tensor_free(conv); q36_tensor_free(qk); q36_tensor_free(out);
}

/* The copy conv and delta take at a row (q36_shape.snap) must be the state of
 * a batch that stops at that row, to the bit: conv history and both planes,
 * from a state that is not zero. The fast scan also handles one-row prefixes. */
static void test_deltanet_snapshot(int rows, int snap, int fast) {
    int saved_exact = g_test_exact;
    int H = 64, vh = 32, kh = 16, kd = 128, vd = 128, ck = 4, cd = 2 * kh * kd + vh * vd;
    size_t cells = (size_t)vh * kd * vd, hist = (size_t)cd * (ck - 1);
    test_set_exact(!fast);
    q36_shape s = {0};
    s.rows = rows; s.hidden = H; s.v_heads = vh; s.k_heads = kh; s.k_dim = kd; s.v_dim = vd;
    s.conv_kernel = ck; s.conv_dim = cd;
    q36_tensor *x = rand_tensor((size_t)rows * H, 1.0f), *proj = rand_tensor((size_t)rows * cd, 1.0f);
    q36_tensor *a = rand_tensor((size_t)vh * H, 0.05f), *b = rand_tensor((size_t)vh * H, 0.05f);
    q36_tensor *alog = rand_tensor((size_t)vh, 1.0f), *dt = rand_tensor((size_t)vh, 1.0f);
    q36_tensor *w = rand_tensor((size_t)cd * ck, 0.5f);
    q36_tensor *ring0 = rand_tensor(hist, 0.5f), *state0 = rand_tensor(2 * cells, 0.01f);
    for (size_t i = cells; i < 2 * cells; i++) F(state0)[i] *= 1e-7f;   /* a low plane of low bits */
    q36_tensor *ring[2], *state[2], *snap_ring = q36_tensor_new(hist * 4), *snap_state = q36_tensor_new(2 * cells * 4);
    q36_tensor *decay = q36_tensor_new((size_t)rows * vh * 4), *beta = q36_tensor_new((size_t)rows * vh * 4);
    q36_tensor *conv = q36_tensor_new((size_t)rows * cd * 4), *qk = q36_tensor_new((size_t)rows * cd * 4);
    q36_tensor *out = q36_tensor_new((size_t)rows * vh * vd * 4);
    for (int run = 0; run < 2; run++) {         /* 0: all rows, copy at snap; 1: the first snap rows */
        ring[run] = q36_tensor_upload(F(ring0), hist * 4);
        state[run] = q36_tensor_upload(F(state0), 2 * cells * 4);
        s.rows = run ? snap : rows;
        s.snap = run ? 0 : snap;
        q36_dn_aux(decay, beta, x, a, b, alog, dt, &s);
        q36_dn_conv(conv, ring[run], proj, w, snap_ring, &s);
        q36_dn_l2(qk, conv, &s);
        q36_dn_delta(out, state[run], qk, conv, decay, beta, snap_state, &s);
        sync_or_die();
    }
    int same_ring = !memcmp(F(snap_ring), F(ring[1]), hist * 4);
    int same_state = !memcmp(F(snap_state), F(state[1]), 2 * cells * 4);
    g_failures += !same_ring + !same_state;
    printf("%-34s %s\n", fast ? "dn copy at a row (fast)" : "dn copy at a row (exact)",
           same_ring && same_state ? "ok    the state of a batch that stops there" : "FAIL  differs from a shorter batch");
    printf("    rows=%d copy after %d: conv history %s, state %s\n", rows, snap, same_ring ? "same" : "DIFFERENT",
           same_state ? "same" : "DIFFERENT");
    q36_tensor *t[] = {x, proj, a, b, alog, dt, w, ring0, state0, ring[0], ring[1], state[0], state[1],
                       snap_ring, snap_state, decay, beta, conv, qk, out};
    for (size_t k = 0; k < sizeof(t) / sizeof(t[0]); k++) q36_tensor_free(t[k]);
    test_set_exact(saved_exact);
}

/* Fast mode keeps the high plane of the DeltaNet state only: the same bits as
 * a full state whose low plane is zero, for the scan (decode, short batches)
 * and the chunks (prompt batches), with the copy for serve mode. */
static void test_deltanet_high_only(int rows, int snap) {
    int saved_exact = g_test_exact;
    int H = 64, vh = 32, kh = 16, kd = 128, vd = 128, ck = 4, cd = 2 * kh * kd + vh * vd;
    size_t cells = (size_t)vh * kd * vd, hist = (size_t)cd * (ck - 1);
    test_set_exact(0);
    q36_shape s = {0};
    s.rows = rows; s.hidden = H; s.v_heads = vh; s.k_heads = kh; s.k_dim = kd; s.v_dim = vd;
    s.conv_kernel = ck; s.conv_dim = cd; s.snap = snap;
    q36_tensor *x = rand_tensor((size_t)rows * H, 1.0f), *proj = rand_tensor((size_t)rows * cd, 1.0f);
    q36_tensor *a = rand_tensor((size_t)vh * H, 0.05f), *b = rand_tensor((size_t)vh * H, 0.05f);
    q36_tensor *alog = rand_tensor((size_t)vh, 1.0f), *dt = rand_tensor((size_t)vh, 1.0f);
    q36_tensor *w = rand_tensor((size_t)cd * ck, 0.5f), *ring0 = rand_tensor(hist, 0.5f);
    q36_tensor *high = rand_tensor(cells, 0.01f);
    q36_tensor *decay = q36_tensor_new((size_t)rows * vh * 4), *beta = q36_tensor_new((size_t)rows * vh * 4);
    q36_tensor *conv = q36_tensor_new((size_t)rows * cd * 4), *qk = q36_tensor_new((size_t)rows * cd * 4);
    q36_tensor *ring = q36_tensor_upload(F(ring0), hist * 4), *snap_ring = q36_tensor_new(hist * 4);
    q36_dn_aux(decay, beta, x, a, b, alog, dt, &s);
    q36_dn_conv(conv, ring, proj, w, snap_ring, &s);
    q36_dn_l2(qk, conv, &s);
    q36_tensor *state[2], *copy[2], *out[2];
    int ok = 1;
    for (int run = 0; run < 2; run++) {         /* 0: high and zero low plane; 1: high plane only */
        size_t planes = run ? 1 : 2;
        state[run] = q36_tensor_new(planes * cells * 4);
        memcpy(F(state[run]), F(high), cells * 4);
        copy[run] = q36_tensor_new(planes * cells * 4);
        out[run] = q36_tensor_new((size_t)rows * vh * vd * 4);
        ok = q36_dn_delta(out[run], state[run], qk, conv, decay, beta, copy[run], &s) && ok;
        sync_or_die();
    }
    int same_out = !memcmp(F(out[0]), F(out[1]), (size_t)rows * vh * vd * 4);
    int same_state = !memcmp(F(state[0]), F(state[1]), cells * 4);
    int same_copy = !snap || !memcmp(F(copy[0]), F(copy[1]), cells * 4);
    int zero_low = 1;
    for (size_t i = cells; i < 2 * cells; i++) zero_low &= F(state[0])[i] == 0.0f;
    test_set_exact(1);                          /* the exact kernels refuse a state without its low plane */
    int refused = !q36_dn_delta(out[1], state[1], qk, conv, decay, beta, copy[1], &s);
    sync_or_die();
    int pass = ok && same_out && same_state && same_copy && zero_low && refused;
    g_failures += !pass;
    printf("dn high plane only rows=%-4d snap=%-3d %s\n", rows, snap,
           pass ? "ok    same bits as a zero low plane" : "FAIL");
    q36_tensor *t[] = {x, proj, a, b, alog, dt, w, ring0, high, decay, beta, conv, qk, ring, snap_ring,
                       state[0], state[1], copy[0], copy[1], out[0], out[1]};
    for (size_t k = 0; k < sizeof(t) / sizeof(t[0]); k++) q36_tensor_free(t[k]);
    test_set_exact(saved_exact);
}

/* Batches of at most 8 rows run the norm, the router's softmax and top-k and the
 * shared gate as one threadgroup per row; larger ones give each row a thread
 * (fast mode has its own norm and softmax for every batch). Both must give the
 * same bits, in both modes: the same rows through a 16-row batch and an 8-row
 * one. The probabilities carry ties, values the top-k must skip, and a row with
 * fewer eligible experts than K; the lengths leave tails. */
static int same_bits(const char *what, q36_tensor *a, q36_tensor *b, size_t bytes) {
    int same = !memcmp(q36_tensor_data(a), q36_tensor_data(b), bytes);
    printf("%-34s %s\n", what, same ? "ok    the same bits in a smaller batch" : "FAIL  differs between batch sizes");
    g_failures += !same;
    return same;
}

static void test_row_kernels(int dim, int E, int K, int exact) {
    enum { R = 16, S = 8 };
    const char *mode = exact ? "" : "fast ";
    char name[96];
    int saved_exact = g_test_exact;
    test_set_exact(exact);
    q36_tensor *x = rand_tensor((size_t)R * dim, 1.0f), *w = rand_tensor((size_t)dim, 0.5f);
    q36_tensor *y16 = q36_tensor_new((size_t)R * dim * 4), *y8 = q36_tensor_new((size_t)R * dim * 4);
    for (int plain = 0; plain < 2; plain++) {
        /* Fast mode: a few long rows (decode) take q36_rmsnorm_wide and more rows
         * q36_rmsnorm_fast, whose sums run in other orders, so there a row is
         * compared within each: 12 rows against 16, 4 against 8. */
        int split = !exact && dim >= 1024 && dim % 4 == 0, small = split ? 12 : S;
        q36_rmsnorm(y16, x, w, R, dim, 1e-6f, plain);
        q36_rmsnorm(y8, x, w, small, dim, 1e-6f, plain);
        sync_or_die();
        snprintf(name, sizeof name, "%srmsnorm by row dim=%d plain=%d", mode, dim, plain);
        same_bits(name, y16, y8, (size_t)small * dim * 4);
        if (split) {
            q36_rmsnorm(y16, x, w, S, dim, 1e-6f, plain);
            q36_rmsnorm(y8, x, w, 4, dim, 1e-6f, plain);
            sync_or_die();
            snprintf(name, sizeof name, "%srmsnorm few rows dim=%d plain=%d", mode, dim, plain);
            same_bits(name, y16, y8, (size_t)4 * dim * 4);
        }
    }
    q36_tensor *g16 = q36_tensor_new(R * 4), *g8 = q36_tensor_new(R * 4);
    q36_shared_gate(g16, x, w, R, dim);
    q36_shared_gate(g8, x, w, S, dim);
    sync_or_die();
    snprintf(name, sizeof name, "%sshared gate by row H=%d", mode, dim);
    same_bits(name, g16, g8, S * 4);

    q36_tensor *l16 = rand_tensor((size_t)R * E, 3.0f), *bias = rand_tensor((size_t)E, 0.1f);
    q36_tensor *l8 = q36_tensor_upload(F(l16), (size_t)R * E * 4);
    q36_tensor *p16 = q36_tensor_new((size_t)R * E * 4), *p8 = q36_tensor_new((size_t)R * E * 4);
    q36_router_softmax(p16, l16, bias, R, E);
    q36_router_softmax(p8, l8, bias, S, E);
    sync_or_die();
    snprintf(name, sizeof name, "%srouter softmax by row E=%d", mode, E);
    same_bits(name, p16, p8, (size_t)S * E * 4);
    snprintf(name, sizeof name, "%srouter logits + bias by row E=%d", mode, E);
    same_bits(name, l16, l8, (size_t)S * E * 4);

    float *pr = F(p16);
    for (int r = 0; r < S; r++) {               /* ties, skipped values, a short row */
        float *row = pr + (size_t)r * E;
        row[3] = row[E - 2];
        row[1] = row[5] = row[E - 1];
        if (r == 2) row[0] = -2e30f;
        if (r == 3) row[7] = 0.0f / 0.0f;
        if (r == 5) for (int e = K / 2; e < E; e++) row[e] = -1e30f;
    }
    q36_tensor *i16 = q36_tensor_new((size_t)R * K * 4), *i8 = q36_tensor_new((size_t)R * K * 4);
    q36_tensor *w16 = q36_tensor_new((size_t)R * K * 4), *w8 = q36_tensor_new((size_t)R * K * 4);
    q36_router_select(i16, w16, p16, R, E, K, 1, 1);
    q36_router_select(i8, w8, p16, S, E, K, 1, 1);
    sync_or_die();
    snprintf(name, sizeof name, "%srouter top-%d ids by row E=%d", mode, K, E);
    same_bits(name, i16, i8, (size_t)S * K * 4);
    snprintf(name, sizeof name, "%srouter top-%d weights by row", mode, K);
    same_bits(name, w16, w8, (size_t)S * K * 4);
    q36_tensor *t[] = {x, w, y16, y8, g16, g8, l16, bias, l8, p16, p8, i16, i8, w16, w8};
    for (size_t k = 0; k < sizeof(t) / sizeof(t[0]); k++) q36_tensor_free(t[k]);
    test_set_exact(saved_exact);
}

/* ---- attention ------------------------------------------------------------------ */

static const uint32_t attn_poison = UINT32_C(0x7fc12345);

static void attn_fill_poison(q36_tensor *t) {
    unsigned char *p = q36_tensor_data(t);
    for (size_t i = 0; i < q36_tensor_bytes(t); i += sizeof(attn_poison))
        memcpy(p + i, &attn_poison, sizeof(attn_poison));
}

static int attn_poison_intact(q36_tensor *t, size_t first, size_t count) {
    const unsigned char *p = q36_tensor_data(t);
    for (size_t i = first; i < first + count; i++) {
        uint32_t bits;
        memcpy(&bits, p + i * sizeof(bits), sizeof(bits));
        if (bits != attn_poison) return 0;
    }
    return 1;
}

/* The fast one-row GQA path against the existing independent double oracle.
 * The first scratch plane must contain probabilities, which also proves the
 * fast kernel ran. One float less than required must take the exact fallback.
 * Guarded views and poisoned future KV rows exercise partial QK/PV tiles. */
static void test_attention_gqa(q36_tensor *query, q36_tensor *kc, q36_tensor *vc,
                               q36_tensor *exact, const q36_shape *s,
                               const double *ref, const double *magnitude) {
    enum { GUARD = 16, PART = 256 };
    int saved_exact = g_test_exact, n = s->pos + 1;
    size_t width = (size_t)s->q_heads * s->head_dim;
    size_t parts = ((size_t)n + PART - 1) / PART;
    size_t need = (size_t)s->q_heads * s->cap + parts * width;
    q36_tensor *out_base = q36_tensor_new((GUARD + width + GUARD) * sizeof(float));
    q36_tensor *scratch_base = q36_tensor_new((GUARD + need + GUARD) * sizeof(float));
    q36_tensor *out = q36_tensor_view(out_base, GUARD * sizeof(float), width * sizeof(float));
    for (int h = 0; h < s->kv_heads; h++)
        for (int t = n; t < s->cap; t++)
            for (int d = 0; d < s->head_dim; d++) {
                size_t i = ((size_t)h * s->cap + t) * s->head_dim + d;
                memcpy(F(kc) + i, &attn_poison, sizeof(attn_poison));
                memcpy(F(vc) + i, &attn_poison, sizeof(attn_poison));
            }
    for (int short_scratch = 0; short_scratch < 2; short_scratch++) {
        size_t count = need - (size_t)short_scratch;
        q36_tensor *scratch = q36_tensor_view(scratch_base, GUARD * sizeof(float), count * sizeof(float));
        attn_fill_poison(out_base);
        attn_fill_poison(scratch_base);
        test_set_exact(0);
        int ok = q36_attn(out, query, kc, vc, scratch, s);
        sync_or_die();
        char name[96];
        snprintf(name, sizeof name, "GQA %s double oracle ctx=%d", short_scratch ? "fallback" : "fast", n);
        /* Cancellation-safe magnitude from the same double softmax/PV oracle.
         * Keep the isolated GQA test's tolerance; model quality is checked separately. */
        check(name, F(out), ref, magnitude, width, 5e-5);
        int guards = attn_poison_intact(out_base, 0, GUARD) &&
                     attn_poison_intact(out_base, GUARD + width, GUARD) &&
                     attn_poison_intact(scratch_base, 0, GUARD) &&
                     attn_poison_intact(scratch_base, GUARD + count, need - count + GUARD);
        if (short_scratch) {
            ok = ok && !memcmp(F(out), F(exact), width * sizeof(float)) &&
                 attn_poison_intact(scratch, 0, count);
        } else {
            for (int h = 0; h < s->q_heads; h++) {
                double sum = 0;
                for (int t = 0; t < n; t++) {
                    float p = F(scratch)[(size_t)h * s->cap + t];
                    if (!isfinite(p) || p < 0 || p > 1) ok = 0;
                    sum += p;
                }
                if (!(fabs(sum - 1.0) <= 2e-5)) ok = 0;
                ok = attn_poison_intact(scratch, (size_t)h * s->cap + n, (size_t)s->cap - n) && ok;
            }
        }
        printf("GQA ctx=%d scratch=%s %s (guards %s, %s)\n", n, short_scratch ? "need-1 float" : "need",
               ok && guards ? "ok" : "FAIL", guards ? "intact" : "DAMAGED",
               short_scratch ? "exact bits and unused scratch" : "normalized probabilities and untouched tails");
        g_failures += !ok + !guards;
        q36_tensor_free(scratch);
    }
    q36_tensor_free(out); q36_tensor_free(out_base); q36_tensor_free(scratch_base);
    test_set_exact(saved_exact);
}

/* The scores scratch of a prompt batch: Q36_ATTN_CHUNK rows of every head over the cache, then
 * as many rows of maxima over every 32 positions (qwen36_gpu.h). A smaller one sends the fast
 * kernels down another path without a word. */
static size_t attn_scratch(int qh, int cap) {
    return (size_t)Q36_ATTN_CHUNK * qh * (cap + (cap + 31) / 32) * 4;
}

static void test_attention(int rows, int qh, int kv, int hd, int qd, int rotary, int pos0) {
    int cap = pos0 + rows + 3;
    q36_shape s = {0};
    s.rows = rows; s.q_heads = qh; s.kv_heads = kv; s.head_dim = hd; s.q_head_dim = qd; s.rotary = rotary;
    s.pos = pos0; s.cap = cap;
    q36_tensor *q = rand_tensor((size_t)rows * qh * qd, 1.0f), *k = rand_tensor((size_t)rows * kv * hd, 1.0f);
    q36_tensor *v = rand_tensor((size_t)rows * kv * hd, 1.0f);
    q36_tensor *query = q36_tensor_new((size_t)rows * qh * hd * 4), *gate = q36_tensor_new((size_t)rows * qh * hd * 4);
    q36_tensor *key = q36_tensor_new((size_t)rows * kv * hd * 4);
    q36_tensor *kc = rand_tensor((size_t)kv * cap * hd, 1.0f), *vc = rand_tensor((size_t)kv * cap * hd, 1.0f);
    q36_tensor *rot = q36_tensor_new((size_t)rows * rotary * 4), *out = q36_tensor_new((size_t)rows * qh * hd * 4);
    for (int r = 0; r < rows; r++)
        for (int d = 0; d < rotary / 2; d++) {
            double f = pow(10000.0, -2.0 * d / rotary), ang = (double)(pos0 + r) * f;
            F(rot)[(size_t)r * rotary + 2 * d] = (float)cos(ang);
            F(rot)[(size_t)r * rotary + 2 * d + 1] = (float)sin(ang);
        }
    /* reference copies of the inputs, before the kernels touch anything */
    size_t nq = (size_t)rows * qh * hd, nk = (size_t)rows * kv * hd, nc = (size_t)kv * cap * hd;
    double *rq = malloc(sizeof(double) * nq), *rg = malloc(sizeof(double) * nq), *rk = malloc(sizeof(double) * nk);
    double *rkc = malloc(sizeof(double) * nc), *rvc = malloc(sizeof(double) * nc), *rout = malloc(sizeof(double) * nq);
    double *rmag = malloc(sizeof(double) * nq), *rqs = malloc(sizeof(double) * nq), *rks = malloc(sizeof(double) * nk);
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < qh; h++)
            for (int d = 0; d < hd; d++) {
                size_t off = ((size_t)r * qh + h) * qd + d;
                rq[((size_t)r * qh + h) * hd + d] = F(q)[off];
                rqs[((size_t)r * qh + h) * hd + d] = fabs((double)F(q)[off]) + 1e-6;
                rg[((size_t)r * qh + h) * hd + d] = qd > hd ? F(q)[off + hd] : 0.0;
            }
    for (size_t i = 0; i < nk; i++) { rk[i] = F(k)[i]; rks[i] = fabs((double)F(k)[i]) + 1e-6; }
    for (size_t i = 0; i < nc; i++) { rkc[i] = F(kc)[i]; rvc[i] = F(vc)[i]; }
    q36_attn_split(query, gate, key, q, k, &s);
    q36_attn_rope(query, rot, qh, &s);
    q36_attn_rope(key, rot, kv, &s);
    q36_attn_kv(kc, vc, key, v, &s);
    q36_attn(out, query, kc, vc, NULL, &s);
    sync_or_die();
    if (rows == 1) {                            /* decode in three steps: the same bits */
        int saved_exact = g_test_exact;
        q36_tensor *scores = q36_tensor_new((size_t)3 * qh * cap * 4), *two = q36_tensor_new((size_t)qh * hd * 4);
        test_set_exact(1);
        q36_attn(two, query, kc, vc, scores, &s);
        sync_or_die();
        int same = !memcmp(q36_tensor_data(two), q36_tensor_data(out), (size_t)qh * hd * 4);
        g_failures += !same;
        printf("%-34s %s\n", "attention decode in three steps", same ? "ok    bit-identical to one step" : "FAIL  differs from one step");
        q36_tensor_free(scores); q36_tensor_free(two);
        test_set_exact(saved_exact);
    }
    /* rope and cache append in the reference */
    for (int r = 0; r < rows; r++)
        for (int d = 0; d < rotary / 2; d++) {
            double cs = F(rot)[(size_t)r * rotary + 2 * d], sn = F(rot)[(size_t)r * rotary + 2 * d + 1];
            for (int h = 0; h < qh; h++) {   /* the error is relative to the terms: a rotated value can be near 0 */
                double *x = rq + ((size_t)r * qh + h) * hd, a = x[d], b = x[d + rotary / 2];
                double *m = rqs + ((size_t)r * qh + h) * hd;
                x[d] = a * cs - b * sn; x[d + rotary / 2] = a * sn + b * cs;
                m[d] = fabs(a * cs) + fabs(b * sn) + 1e-6; m[d + rotary / 2] = fabs(a * sn) + fabs(b * cs) + 1e-6;
            }
            for (int h = 0; h < kv; h++) {
                double *x = rk + ((size_t)r * kv + h) * hd, a = x[d], b = x[d + rotary / 2];
                double *m = rks + ((size_t)r * kv + h) * hd;
                x[d] = a * cs - b * sn; x[d + rotary / 2] = a * sn + b * cs;
                m[d] = fabs(a * cs) + fabs(b * sn) + 1e-6; m[d + rotary / 2] = fabs(a * sn) + fabs(b * cs) + 1e-6;
            }
        }
    double *rkcs = malloc(sizeof(double) * nc);  /* the k cache's scale: the old rows' values, the new rows' terms */
    for (size_t i = 0; i < nc; i++) rkcs[i] = fabs(rkc[i]) + 1e-6;
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < kv; h++)
            for (int d = 0; d < hd; d++) {
                size_t dest = ((size_t)h * cap + pos0 + r) * hd + d;
                rkc[dest] = rk[((size_t)r * kv + h) * hd + d];
                rkcs[dest] = rks[((size_t)r * kv + h) * hd + d];
                rvc[dest] = F(v)[((size_t)r * kv + h) * hd + d];
            }
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < qh; h++) {
            int kh = h / (qh / kv);
            const double *qq = rq + ((size_t)r * qh + h) * hd;
            double mx = -1e300, den = 0, *acc = calloc((size_t)hd, sizeof(double));
            double *mag = calloc((size_t)hd, sizeof(double));
            int n = pos0 + r + 1;
            double *sc = malloc(sizeof(double) * n);
            for (int t = 0; t < n; t++) {
                double sd = 0;
                for (int d = 0; d < hd; d++) sd += qq[d] * rkc[((size_t)kh * cap + t) * hd + d];
                sc[t] = sd / sqrt((double)hd);
                if (sc[t] > mx) mx = sc[t];
            }
            for (int t = 0; t < n; t++) {
                double wgt = exp(sc[t] - mx);
                den += wgt;
                for (int d = 0; d < hd; d++) {
                    double value = rvc[((size_t)kh * cap + t) * hd + d];
                    acc[d] += wgt * value;
                    mag[d] += wgt * fabs(value);
                }
            }
            for (int d = 0; d < hd; d++) {
                size_t i = ((size_t)r * qh + h) * hd + d;
                rout[i] = acc[d] / den;
                rmag[i] = mag[d] / den + 1e-6;
            }
            free(acc); free(mag); free(sc);
        }
    char name[96];
    snprintf(name, sizeof name, "attn gate split qh=%d hd=%d", qh, hd);
    check(name, F(gate), rg, NULL, nq, 0);
    snprintf(name, sizeof name, "attn rope q rotary=%d pos=%d", rotary, pos0);
    check(name, F(query), rq, rqs, nq, 1e-5);
    snprintf(name, sizeof name, "attn k cache append");
    check(name, F(kc), rkc, rkcs, nc, 1e-5);
    snprintf(name, sizeof name, "attn v cache append");
    check(name, F(vc), rvc, NULL, nc, 0);
    snprintf(name, sizeof name, "attn output rows=%d ctx=%d", rows, pos0 + rows);
    check(name, F(out), rout, NULL, nq, 1e-4);
    if (rows == 1 && qh / kv == 8 && hd % 64 == 0 && pos0 >= 127)
        test_attention_gqa(query, kc, vc, out, &s, rout, rmag);
    if (rows > 8 && hd % 64 == 0) {             /* prompt rows on the matrix units (fast path) */
        int saved_exact = g_test_exact;
        q36_tensor *scores = q36_tensor_new(attn_scratch(qh, cap)), *fast = q36_tensor_new(nq * 4);
        test_set_exact(0);
        q36_attn(fast, query, kc, vc, scores, &s);
        sync_or_die();
        snprintf(name, sizeof name, "attn fast output rows=%d ctx=%d", rows, pos0 + rows);
        check(name, F(fast), rout, NULL, nq, 1e-4);
        q36_tensor_free(scores); q36_tensor_free(fast);
        test_set_exact(saved_exact);
    }
    free(rq); free(rg); free(rk); free(rkc); free(rvc); free(rout); free(rmag); free(rqs); free(rks); free(rkcs);
    q36_tensor_free(q); q36_tensor_free(k); q36_tensor_free(v); q36_tensor_free(query); q36_tensor_free(gate);
    q36_tensor_free(key); q36_tensor_free(kc); q36_tensor_free(vc); q36_tensor_free(rot); q36_tensor_free(out);
}

/* The fast prompt attention (matrix units) against the exact kernel on the
 * same cache, at real shapes: no RoPE, so only the attention is compared. */
static void test_attention_fast(int rows, int qh, int kv, int hd, int pos0) {
    int saved_exact = g_test_exact;
    int cap = pos0 + rows + 3;
    q36_shape s = {0};
    s.rows = rows; s.q_heads = qh; s.kv_heads = kv; s.head_dim = hd; s.q_head_dim = hd; s.pos = pos0; s.cap = cap;
    size_t nq = (size_t)rows * qh * hd;
    q36_tensor *query = rand_tensor(nq, 1.0f), *kc = rand_tensor((size_t)kv * cap * hd, 1.0f);
    q36_tensor *vc = rand_tensor((size_t)kv * cap * hd, 1.0f), *exact = q36_tensor_new(nq * 4), *fast = q36_tensor_new(nq * 4);
    q36_tensor *scores = q36_tensor_new(attn_scratch(qh, cap));
    test_set_exact(1);
    q36_attn(exact, query, kc, vc, NULL, &s);
    test_set_exact(0);
    q36_attn(fast, query, kc, vc, scores, &s);
    sync_or_die();
    double *ref = malloc(sizeof(double) * nq);
    for (size_t i = 0; i < nq; i++) ref[i] = F(exact)[i];
    char name[96];
    snprintf(name, sizeof name, "attn fast vs exact rows=%d ctx=%d", rows, pos0 + rows);
    check(name, F(fast), ref, NULL, nq, 1e-4);
    free(ref);
    q36_tensor_free(query); q36_tensor_free(kc); q36_tensor_free(vc); q36_tensor_free(exact);
    q36_tensor_free(fast); q36_tensor_free(scores);
    test_set_exact(saved_exact);
}

/* A half KV cache (fast mode, q36_kv_half) against the exact kernels run on a
 * float cache that holds the same rounded values: decode (the GQA path at any
 * position), short and long prompt batches (the matrix path). q36_attn_kv must
 * round like the CPU, and the exact kernels must refuse a half cache. */
static void test_attention_half(int rows, int pos0, int hd) {
    int saved_exact = g_test_exact, qh = 16, kv = 2, cap = pos0 + rows + 3;
    q36_shape s = {0};
    s.rows = rows; s.q_heads = qh; s.kv_heads = kv; s.head_dim = hd; s.q_head_dim = hd; s.pos = pos0; s.cap = cap;
    size_t nq = (size_t)rows * qh * hd, ncache = (size_t)kv * cap * hd;
    q36_tensor *query = rand_tensor(nq, 1.0f), *kf = rand_tensor(ncache, 1.0f), *vf = rand_tensor(ncache, 1.0f);
    q36_tensor *kh = q36_tensor_new(ncache * 2), *vh = q36_tensor_new(ncache * 2);
    _Float16 *k16 = q36_tensor_data(kh), *v16 = q36_tensor_data(vh);
    for (size_t i = 0; i < ncache; i++) {        /* the float cache keeps the rounded values too */
        k16[i] = (_Float16)F(kf)[i]; F(kf)[i] = (float)k16[i];
        v16[i] = (_Float16)F(vf)[i]; F(vf)[i] = (float)v16[i];
    }
    q36_tensor *exact = q36_tensor_new(nq * 4), *half = q36_tensor_new(nq * 4);
    q36_tensor *scores = q36_tensor_new(attn_scratch(qh, cap));
    test_set_exact(1);
    int ok = q36_attn(exact, query, kf, vf, NULL, &s) && !q36_attn(half, query, kh, vh, scores, &s);
    test_set_exact(0);
    ok = q36_kv_half(&s) && q36_attn(half, query, kh, vh, scores, &s) && ok;
    /* the cache writer: rows written by q36_attn_kv round as the CPU does */
    q36_shape w = s;
    w.rows = 3; w.pos = pos0;
    q36_tensor *nk = rand_tensor((size_t)3 * kv * hd, 1.0f), *nv = rand_tensor((size_t)3 * kv * hd, 1.0f);
    ok = q36_attn_kv(kh, vh, nk, nv, &w) && ok;
    sync_or_die();
    int rounded = 1;
    for (int r = 0; r < 3; r++) for (int h = 0; h < kv; h++) for (int d = 0; d < hd; d++) {
        size_t at = ((size_t)h * cap + pos0 + r) * hd + d, from = ((size_t)r * kv + h) * hd + d;
        rounded &= k16[at] == (_Float16)F(nk)[from] && v16[at] == (_Float16)F(nv)[from];
    }
    g_failures += !ok + !rounded;
    double *ref = malloc(sizeof(double) * nq);
    for (size_t i = 0; i < nq; i++) ref[i] = F(exact)[i];
    char name[96];
    snprintf(name, sizeof name, "attn half cache rows=%d ctx=%d hd=%d", rows, pos0 + rows, hd);
    check(name, F(half), ref, NULL, nq, 1e-4);
    printf("    the exact kernels refuse it, q36_attn_kv rounds as the CPU: %s\n", ok && rounded ? "ok" : "FAIL");
    free(ref);
    q36_tensor *t[] = {query, kf, vf, kh, vh, exact, half, scores, nk, nv};
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) q36_tensor_free(t[i]);
    test_set_exact(saved_exact);
}

#if QCM_ON
/* The CPU's attention of a prompt batch's last rows (qwen36_cpu_mm.h) on a half
 * cache, against the exact kernels run on a float cache that holds the same
 * rounded values; the rows before `first` are not written. */
static void test_attention_cpu(int rows, int pos0, int first) {
    int saved_exact = g_test_exact, qh = 16, kv = 2, hd = 256, cap = pos0 + rows + 3;
    q36_shape s = {0};
    s.rows = rows; s.q_heads = qh; s.kv_heads = kv; s.head_dim = hd; s.q_head_dim = hd; s.pos = pos0; s.cap = cap;
    size_t nq = (size_t)rows * qh * hd, ncache = (size_t)kv * cap * hd, skip = (size_t)first * qh * hd;
    q36_tensor *query = rand_tensor(nq, 1.0f), *kf = rand_tensor(ncache, 1.0f), *vf = rand_tensor(ncache, 1.0f);
    q36_tensor *kh = q36_tensor_new(ncache * 2), *vh = q36_tensor_new(ncache * 2), *exact = q36_tensor_new(nq * 4);
    _Float16 *k16 = q36_tensor_data(kh), *v16 = q36_tensor_data(vh);
    for (size_t i = 0; i < ncache; i++) {
        k16[i] = (_Float16)F(kf)[i]; F(kf)[i] = (float)k16[i];
        v16[i] = (_Float16)F(vf)[i]; F(vf)[i] = (float)v16[i];
    }
    test_set_exact(1);
    int ok = q36_attn(exact, query, kf, vf, NULL, &s);
    sync_or_die();
    float *cpu = malloc(nq * sizeof(float));
    for (size_t i = 0; i < nq; i++) cpu[i] = -12345.f;
    QcmAttn a = {0};
    ok = qcm_attention(&a, cpu, F(query), k16, v16, 1, first, rows, pos0, qh, kv, hd, cap) && ok;
    qcm_attn_free(&a);
    size_t written = 0;
    for (size_t i = 0; i < skip; i++) written += cpu[i] != -12345.f;
    double *ref = malloc(sizeof(double) * nq);
    for (size_t i = 0; i < nq; i++) ref[i] = F(exact)[i];
    char name[96];
    snprintf(name, sizeof name, "attn cpu rows %d..%d ctx=%d", first, rows, pos0 + rows);
    check(name, cpu + skip, ref + skip, NULL, nq - skip, 1e-4);
    printf("    rows before %d untouched: %s\n", first, ok && !written ? "ok" : "FAIL");
    g_failures += !ok + (written > 0);
    free(ref); free(cpu);
    q36_tensor *t[] = {query, kf, vf, kh, vh, exact};
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) q36_tensor_free(t[i]);
    test_set_exact(saved_exact);
}
#endif

/* ---- MoE --------------------------------------------------------------------------- */

/* ties: few distinct logits and no bias, so many probabilities are exactly
 * equal and the selection has to follow the lowest-id rule. */
static void test_router(int rows, int E, int K, int G, int T, int ties) {
    q36_tensor *logits = rand_tensor((size_t)rows * E, 4.0f), *bias = rand_tensor((size_t)E, 0.1f);
    for (int i = 0; ties && i < rows * E; i++) F(logits)[i] = (float)((i * 7) % 5) * 0.75f;
    for (int e = 0; ties && e < E; e++) F(bias)[e] = 0;
    q36_tensor *prob = q36_tensor_new((size_t)rows * E * 4);
    q36_tensor *ids = q36_tensor_new((size_t)rows * K * 4), *wts = q36_tensor_new((size_t)rows * K * 4);
    double *rl = malloc(sizeof(double) * rows * E), *rp = malloc(sizeof(double) * rows * E);
    for (int i = 0; i < rows * E; i++) rl[i] = F(logits)[i];
    q36_router_softmax(prob, logits, bias, rows, E);
    q36_router_select(ids, wts, prob, rows, E, K, G, T);
    sync_or_die();
    int32_t *rid = malloc(sizeof(int32_t) * rows * K);
    double *rw = malloc(sizeof(double) * rows * K);
    const float *p = F(prob);
    for (int r = 0; r < rows; r++) {
        double mx = -1e300, sum = 0;
        for (int e = 0; e < E; e++) { rl[r * E + e] += F(bias)[e]; if (rl[r * E + e] > mx) mx = rl[r * E + e]; }
        for (int e = 0; e < E; e++) { rp[r * E + e] = exp(rl[r * E + e] - mx); sum += rp[r * E + e]; }
        for (int e = 0; e < E; e++) rp[r * E + e] /= sum;
        /* selection is checked on the GPU's own probabilities (ties need them) */
        int keep[256];
        for (int e = 0; e < E; e++) keep[e] = G == 1;
        if (G > 1) {
            int per = E / G, used[256] = {0};
            double gs[256];
            for (int g = 0; g < G; g++) {
                double b1 = -1e30, b2 = -1e30;
                for (int e = g * per; e < (g + 1) * per; e++) {
                    double v = p[r * E + e];
                    if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v;
                }
                gs[g] = (float)b1 + (float)b2;
            }
            for (int t = 0; t < T; t++) {
                int best = -1; double value = -1e30;
                for (int g = 0; g < G; g++) if (!used[g] && gs[g] > value) { best = g; value = gs[g]; }
                used[best] = 1;
                for (int e = best * per; e < (best + 1) * per; e++) keep[e] = 1;
            }
        }
        double s = 0;
        for (int k = 0; k < K; k++) {
            int best = -1; double value = -1e30;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int j = 0; j < k; j++) taken |= rid[r * K + j] == e;
                if (keep[e] && !taken && p[r * E + e] > value) { best = e; value = p[r * E + e]; }
            }
            rid[r * K + k] = best;
            rw[r * K + k] = value;
            s += value;
        }
        for (int k = 0; k < K; k++) rw[r * K + k] /= s;
    }
    char name[80];
    snprintf(name, sizeof name, "router softmax rows=%d E=%d", rows, E);
    check(name, F(prob), rp, NULL, (size_t)rows * E, 1e-5);
    snprintf(name, sizeof name, "router select K=%d G=%d T=%d%s", K, G, T, ties ? " ties" : "");
    check_ints(name, q36_tensor_data(ids), rid, (size_t)rows * K);
    snprintf(name, sizeof name, "router weights");
    check(name, F(wts), rw, NULL, (size_t)rows * K, 1e-6);
    free(rl); free(rp); free(rid); free(rw);
    q36_tensor_free(logits); q36_tensor_free(bias); q36_tensor_free(prob); q36_tensor_free(ids); q36_tensor_free(wts);
}

static void test_shared_gate(int rows, int H) {
    q36_tensor *x = rand_tensor((size_t)rows * H, 1.0f), *w = rand_tensor((size_t)H, 0.05f);
    q36_tensor *g = q36_tensor_new((size_t)rows * 4), *g1 = q36_tensor_new((size_t)rows * 4);
    q36_tensor *y = rand_tensor((size_t)rows * H, 1.0f);
    double *ry = malloc(sizeof(double) * rows * H), *rg = malloc(sizeof(double) * rows), one[64];
    q36_shared_gate(g, x, w, rows, H);
    q36_shared_gate(g1, x, NULL, rows, H);
    sync_or_die();
    for (int i = 0; i < rows * H; i++) ry[i] = F(y)[i];
    q36_scale_rows(y, g, rows, H);
    sync_or_die();
    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int i = 0; i < H; i++) s += (double)F(x)[(size_t)r * H + i] * F(w)[i];
        rg[r] = 1.0 / (1.0 + exp(-s));
        one[r] = 1.0;
        for (int i = 0; i < H; i++) ry[(size_t)r * H + i] *= F(g)[r];
    }
    check("shared_gate", F(g), rg, NULL, (size_t)rows, 1e-5);
    check("shared_gate without weight", F(g1), one, NULL, (size_t)rows, 0);
    check("scale_rows", F(y), ry, NULL, (size_t)rows * H, 1e-6);
    free(ry); free(rg);
    q36_tensor_free(x); q36_tensor_free(w); q36_tensor_free(g); q36_tensor_free(g1); q36_tensor_free(y);
}

/* An expert slot as the cache holds it: planar int4 gate|up|down, then scales. */
static q36_tensor *rand_slot(int H, int I, size_t *scales) {
    size_t n = (size_t)H * I;
    *scales = 3 * n / 2;
    q36_tensor *t = q36_tensor_new(*scales + 3 * (n / 64) * 4);
    uint8_t *p = q36_tensor_data(t);
    for (size_t i = 0; i < 3 * n / 2; i++) p[i] = (uint8_t)((g_rng = g_rng * 1664525u + 1013904223u) >> 24);
    float *sc = (float *)(p + *scales);
    for (size_t i = 0; i < 3 * n / 64; i++) sc[i] = (frand() + 1.5f) * 0.01f;
    return t;
}

/* row o of a planar int4 matrix with `in` inputs, dotted with x, in double */
static double dot4_ref(const uint8_t *w, const float *s, const float *x, int in, int o, double *mag) {
    w += (size_t)o * (in / 2);
    s += (size_t)o * (in / 64);
    double sum = 0, m = 0;
    for (int b = 0; b < in / 64; b++)
        for (int j = 0; j < 64; j++) {
            uint8_t byte = w[b * 32 + (j & 31)];
            int q = (j < 32 ? (byte & 15) : (byte >> 4)) - 8;
            double t = (double)q * x[b * 64 + j] * s[b];
            sum += t;
            m += fabs(t);
        }
    if (mag) *mag = m;
    return sum;
}

/* rows tokens choose K distinct experts each out of E; the choices are sorted
 * by expert and computed in groups of K slots, as the graph does. */
static void test_experts(int H, int I, int K, int E, int rows) {
    size_t scales = 0, n = (size_t)H * I;
    int choices = rows * K;
    q36_tensor **slot = calloc((size_t)E, sizeof(*slot));
    for (int e = 0; e < E; e++) slot[e] = rand_slot(H, I, &scales);
    q36_tensor *x = rand_tensor((size_t)rows * H, 1.0f), *wts = q36_tensor_new((size_t)choices * 4);
    q36_tensor *shared = rand_tensor((size_t)rows * H, 1.0f), *resid = rand_tensor((size_t)rows * H, 1.0f);
    q36_tensor *h = q36_tensor_new((size_t)choices * I * 4), *y = q36_tensor_new((size_t)choices * H * 4);
    q36_tensor *out = q36_tensor_new((size_t)rows * H * 4), *pairs = q36_tensor_new((size_t)choices * 4);
    int *ids = malloc(sizeof(int) * choices), *first = calloc((size_t)E + 1, sizeof(int));
    int *experts = malloc(sizeof(int) * E), ne = 0;
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < K; k++) {
            int e, taken;
            do {
                e = (int)((g_rng = g_rng * 1664525u + 1013904223u) >> 8) % E;
                taken = 0;
                for (int j = 0; j < k; j++) taken |= ids[r * K + j] == e;
            } while (taken);
            ids[r * K + k] = e;
            F(wts)[r * K + k] = frand() * 0.5f;
        }
    uint32_t *pp = q36_tensor_data(pairs);
    for (int p = 0; p < choices; p++) first[ids[p] + 1]++;
    for (int e = 0; e < E; e++) {
        if (first[e + 1]) experts[ne++] = e;
        first[e + 1] += first[e];
    }
    for (int i = 0, q = 0; i < ne; i++)
        for (int p = 0; p < choices; p++)
            if (ids[p] == experts[i]) pp[q++] = (uint32_t)p | (uint32_t)(i % K) << 28;
    for (int g0 = 0; g0 < ne; g0 += K) {
        int ng = ne - g0 < K ? ne - g0 : K, a = first[experts[g0]], b = first[experts[g0 + ng - 1] + 1];
        q36_tensor *s8[8] = {0};
        for (int j = 0; j < ng; j++) s8[j] = slot[experts[g0 + j]];
        q36_expert_gate_up(h, x, pairs, a, b - a, s8, rows, H, I, K, scales);
        q36_expert_down(y, h, pairs, a, b - a, s8, rows, H, I, K, scales);
    }
    q36_expert_sum(out, y, wts, shared, resid, rows, H, K);
    sync_or_die();
    double *rh = malloc(sizeof(double) * choices * I), *hm = malloc(sizeof(double) * choices * I);
    double *ry = malloc(sizeof(double) * choices * H), *ym = malloc(sizeof(double) * choices * H);
    double *rout = malloc(sizeof(double) * rows * H), *mag = malloc(sizeof(double) * rows * H);
    for (int p = 0; p < choices; p++) {
        const uint8_t *w = q36_tensor_data(slot[ids[p]]);
        const float *sc = (const float *)(w + scales), *xr = F(x) + (size_t)(p / K) * H;
        for (int i = 0; i < I; i++) {
            double mg, mu, g = dot4_ref(w, sc, xr, H, i, &mg), u = dot4_ref(w + n / 2, sc + n / 64, xr, H, i, &mu);
            rh[(size_t)p * I + i] = g / (1.0 + exp(-g)) * u;
            hm[(size_t)p * I + i] = (mg * fabs(u) + mu * fabs(g)) + 1e-9;
        }
        for (int o = 0; o < H; o++) {      /* from the GPU's h, so each kernel is checked alone */
            double mk;
            ry[(size_t)p * H + o] = dot4_ref(w + n, sc + 2 * n / 64, F(h) + (size_t)p * I, I, o, &mk);
            ym[(size_t)p * H + o] = mk + 1e-9;
        }
    }
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < H; o++) {
            size_t i = (size_t)r * H + o;
            double sum = 0, m = 0;
            for (int k = 0; k < K; k++) {
                float wk = F(wts)[r * K + k], yk = F(y)[(size_t)(r * K + k) * H + o];
                sum += (double)wk * yk;
                m += fabs((double)wk * yk);
            }
            rout[i] = (double)F(resid)[i] + sum + F(shared)[i];
            mag[i] = m + fabs(F(resid)[i]) + fabs(F(shared)[i]) + 1e-9;
        }
    char name[80];
    snprintf(name, sizeof name, "expert gate_up H=%d I=%d K=%d E=%d rows=%d", H, I, K, E, rows);
    check(name, F(h), rh, hm, (size_t)choices * I, 3e-5);
    snprintf(name, sizeof name, "expert down (%d experts, %d groups)", ne, (ne + K - 1) / K);
    check(name, F(y), ry, ym, (size_t)choices * H, 3e-5);
    snprintf(name, sizeof name, "expert sum + shared + residual");
    check(name, F(out), rout, mag, (size_t)rows * H, 3e-6);
    free(rh); free(hm); free(ry); free(ym); free(rout); free(mag); free(ids); free(first); free(experts);
    for (int e = 0; e < E; e++) q36_tensor_free(slot[e]);
    free(slot);
    q36_tensor_free(x); q36_tensor_free(wts); q36_tensor_free(shared); q36_tensor_free(resid);
    q36_tensor_free(h); q36_tensor_free(y); q36_tensor_free(out); q36_tensor_free(pairs);
}

#if QCM_ON
/* The CPU's dense products (qwen36_cpu_mm.h): int8 rows times their scales converted to floats, all of
 * them (qcm_convert, two matrices stacked) or one part of each head (qcm_convert_rows), then BLAS into
 * rows wider than the product; against a double reference. A second conversion of the same matrix is
 * skipped, another matrix is converted. */
static void test_cpu_dense(int rows, int in, int out) {
    int8_t *q = malloc((size_t)2 * out * in);
    float *sc = malloc(sizeof(float) * 2 * out), *x = malloc(sizeof(float) * (size_t)rows * in);
    for (size_t i = 0; i < (size_t)2 * out * in; i++) q[i] = (int8_t)(frand() * 127.0f);
    for (int o = 0; o < 2 * out; o++) sc[o] = 0.001f + 0.01f * fabsf(frand());
    for (size_t i = 0; i < (size_t)rows * in; i++) x[i] = frand();
    int ld = 2 * out + 3;                       /* rows wider than the product, an odd width */
    float *y = malloc(sizeof(float) * (size_t)rows * ld);
    double *ref = malloc(sizeof(double) * (size_t)rows * 2 * out), *mag = malloc(sizeof(double) * (size_t)rows * 2 * out);
    QcmWeights w = {0};
    const int8_t *qs[2] = {q, q + (size_t)out * in};
    const float *ss[2] = {sc, sc + out};
    int ok = qcm_convert(&w, 2, qs, ss, out, in);
    float *before = w.w;
    ok = qcm_convert(&w, 2, qs, ss, out, in) && w.w == before && ok;     /* held: nothing converted */
    for (size_t i = 0; i < (size_t)rows * ld; i++) y[i] = -12345.f;
    qcm_mm(y + 1, ld, x, in, w.w, rows, in, 2 * out);
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < 2 * out; o++) {
            double s = 0, m = 0;
            for (int i = 0; i < in; i++) {
                double t = (double)q[(size_t)o * in + i] * sc[o] * x[(size_t)r * in + i];
                s += t;
                m += fabs(t);
            }
            ref[(size_t)r * 2 * out + o] = s;
            mag[(size_t)r * 2 * out + o] = m + 1e-12;
        }
    float *got = malloc(sizeof(float) * (size_t)rows * 2 * out);
    size_t edges = 0;
    for (int r = 0; r < rows; r++) {
        memcpy(got + (size_t)r * 2 * out, y + (size_t)r * ld + 1, sizeof(float) * 2 * out);
        edges += y[(size_t)r * ld] != -12345.f;
        edges += y[(size_t)r * ld + 1 + 2 * out] != -12345.f;
        edges += y[(size_t)r * ld + 2 + 2 * out] != -12345.f;
    }
    char name[96];
    snprintf(name, sizeof name, "cpu dense %dx%d->2x%d", rows, in, out);
    check(name, got, ref, mag, (size_t)rows * 2 * out, 1e-5);
    /* the second half of each pair of heads of `group` rows, as the attention's gate */
    int group = 32, heads = 2 * out / (2 * group);
    ok = qcm_convert_rows(&w, q, sc, heads * group, in, group, 2 * group, group) && ok;
    qcm_mm(y, heads * group, x, in, w.w, rows, in, heads * group);
    double *rr = malloc(sizeof(double) * (size_t)rows * heads * group), *rm = malloc(sizeof(double) * (size_t)rows * heads * group);
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < heads * group; o++) {
            int src = o / group * 2 * group + group + o % group;
            rr[(size_t)r * heads * group + o] = ref[(size_t)r * 2 * out + src];
            rm[(size_t)r * heads * group + o] = mag[(size_t)r * 2 * out + src];
        }
    snprintf(name, sizeof name, "cpu dense rows, %d heads of 2x%d", heads, group);
    check(name, y, rr, rm, (size_t)rows * heads * group, 1e-5);
    printf("    conversion held, columns around untouched: %s\n", ok && !edges ? "ok" : "FAIL");
    g_failures += !ok + (edges > 0);
    qcm_weights_free(&w);
    free(q); free(sc); free(x); free(y); free(ref); free(mag); free(got); free(rr); free(rm);
}
#endif

#if QCM_ON
/* The CPU's shared expert (qwen36_cpu_mm.h): (silu(x Wg) * (x Wu)) Wd, each row times sigmoid(x . gate);
 * against a double reference from the same int8 weights. */
static void test_cpu_shared(int rows, int H, int SI) {
    int8_t *q[3];
    float *sc[3];
    int outs[3] = {SI, SI, H}, ins[3] = {H, H, SI};
    for (int k = 0; k < 3; k++) {
        q[k] = malloc((size_t)outs[k] * ins[k]);
        sc[k] = malloc(sizeof(float) * outs[k]);
        for (size_t i = 0; i < (size_t)outs[k] * ins[k]; i++) q[k][i] = (int8_t)(frand() * 127.0f);
        for (int o = 0; o < outs[k]; o++) sc[k][o] = 0.0005f + 0.002f * fabsf(frand());
    }
    float *x = malloc(sizeof(float) * (size_t)rows * H), *gate = malloc(sizeof(float) * H);
    float *out = malloc(sizeof(float) * (size_t)rows * H);
    for (size_t i = 0; i < (size_t)rows * H; i++) x[i] = frand();
    for (int i = 0; i < H; i++) gate[i] = 0.05f * frand();
    QcmShared j;
    memset(&j, 0, sizeof j);
    j.post = x; j.shared = out; j.gate = gate; j.rows = rows; j.H = H; j.SI = SI;
    for (int k = 0; k < 3; k++) { j.q[k] = q[k]; j.sc[k] = sc[k]; }
    int ok = qcm_shared_part(&j);
    double *ref = malloc(sizeof(double) * (size_t)rows * H), *mag = malloc(sizeof(double) * (size_t)rows * H);
    double *h = malloc(sizeof(double) * SI), *hm = malloc(sizeof(double) * SI);
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * H;
        for (int i = 0; i < SI; i++) {
            double g = 0, u = 0, mg = 0, mu = 0;
            for (int c = 0; c < H; c++) {
                double tg = (double)q[0][(size_t)i * H + c] * sc[0][i] * xr[c], tu = (double)q[1][(size_t)i * H + c] * sc[1][i] * xr[c];
                g += tg; u += tu; mg += fabs(tg); mu += fabs(tu);
            }
            double sg = 1.0 / (1.0 + exp(-g));
            h[i] = g * sg * u;
            hm[i] = (mg * fabs(u) + mu * fabs(g)) + 1e-9;
        }
        double gs = 0;
        for (int c = 0; c < H; c++) gs += (double)xr[c] * gate[c];
        double s = 1.0 / (1.0 + exp(-gs));
        for (int o = 0; o < H; o++) {
            double y = 0, m = 0;
            for (int i = 0; i < SI; i++) {
                double w = (double)q[2][(size_t)o * SI + i] * sc[2][o];
                y += w * h[i];
                m += fabs(w) * hm[i];
            }
            ref[(size_t)r * H + o] = y * s;
            mag[(size_t)r * H + o] = m * s + 1e-9;
        }
    }
    char name[96];
    snprintf(name, sizeof name, "cpu shared expert rows=%d H=%d SI=%d", rows, H, SI);
    check(name, out, ref, mag, (size_t)rows * H, 1e-5);
    g_failures += !ok;
    qcm_shared_release(&j);
    for (int k = 0; k < 3; k++) { free(q[k]); free(sc[k]); }
    free(x); free(gate); free(out); free(ref); free(mag); free(h); free(hm);
}
#endif

#if QCM_ON
/* The CPU's routed experts (qwen36_cpu_mm.h): planar int4 slots decoded with NEON, then BLAS, each
 * choice's row of y written in place; against a double reference from the same weights. More than
 * QCM_EXPERT_ROWS choices for an expert take two products. Rows of y no choice owns stay untouched. */
static void test_cpu_experts(int H, int I, int K, int E, int rows) {
    size_t scales = 0, n = (size_t)H * I;
    int choices = rows * K;
    q36_tensor **slot = calloc((size_t)E, sizeof(*slot));
    for (int e = 0; e < E; e++) slot[e] = rand_slot(H, I, &scales);
    q36_tensor *x = rand_tensor((size_t)rows * H, 1.0f);
    int *ids = malloc(sizeof(int) * choices), *first = calloc((size_t)E + 1, sizeof(int));
    uint32_t *pairs = malloc(sizeof(uint32_t) * choices);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < K; k++) {
            int e, taken;
            do {
                e = (int)((g_rng = g_rng * 1664525u + 1013904223u) >> 8) % E;
                taken = 0;
                for (int j = 0; j < k; j++) taken |= ids[r * K + j] == e;
            } while (taken);
            ids[r * K + k] = e;
        }
    for (int p = 0; p < choices; p++) first[ids[p] + 1]++;
    for (int e = 0; e < E; e++) first[e + 1] += first[e];
    for (int e = 0, q = 0; e < E; e++)
        for (int p = 0; p < choices; p++)
            if (ids[p] == e) pairs[q++] = (uint32_t)p | (uint32_t)(e % 8) << 28;   /* slot bits, as the graph's */
    float *y = malloc(sizeof(float) * (size_t)(choices + 1) * H);
    for (size_t i = 0; i < (size_t)(choices + 1) * H; i++) y[i] = -12345.f;
    QcmShared j;
    memset(&j, 0, sizeof j);
    j.post = F(x);
    j.ey = y;
    j.H = H; j.I = I; j.K = K; j.scales = scales; j.rows = rows;
    for (int e = 0; e < E; e++)
        if (first[e + 1] > first[e]) j.ex[j.nex++] = (QcmExpert){q36_tensor_data(slot[e]), pairs + first[e], first[e + 1] - first[e]};
    int ok = qcm_experts_run(&j);
    double *ry = malloc(sizeof(double) * choices * H), *ym = malloc(sizeof(double) * choices * H), *hh = malloc(sizeof(double) * I);
    float *hf = malloc(sizeof(float) * I);
    for (int p = 0; p < choices; p++) {
        const uint8_t *w = q36_tensor_data(slot[ids[p]]);
        const float *sc = (const float *)(w + scales), *xr = F(x) + (size_t)(p / K) * H;
        for (int i = 0; i < I; i++) {
            double mg, mu, g = dot4_ref(w, sc, xr, H, i, &mg), u = dot4_ref(w + n / 2, sc + n / 64, xr, H, i, &mu);
            hh[i] = g / (1.0 + exp(-g)) * u;
            hf[i] = (float)hh[i];
        }
        for (int o = 0; o < H; o++) {
            double mk;
            ry[(size_t)p * H + o] = dot4_ref(w + n, sc + 2 * n / 64, hf, I, o, &mk);
            ym[(size_t)p * H + o] = mk + 1e-6;
        }
    }
    size_t written = 0;
    for (int i = 0; i < H; i++) written += y[(size_t)choices * H + i] != -12345.f;
    char name[96];
    snprintf(name, sizeof name, "cpu experts H=%d I=%d K=%d E=%d rows=%d", H, I, K, E, rows);
    check(name, y, ry, ym, (size_t)choices * H, 1e-4);
    printf("    the row after the last choice untouched, the job ok: %s\n", ok && !written ? "ok" : "FAIL");
    g_failures += !ok + (written > 0);
    qcm_shared_release(&j);
    free(ry); free(ym); free(hh); free(hf); free(y); free(ids); free(first); free(pairs);
    for (int e = 0; e < E; e++) q36_tensor_free(slot[e]);
    free(slot);
    q36_tensor_free(x);
}
#endif

/* ---- file mapping windows ---------------------------------------------------------- */

static void test_map_views(void) {
    size_t page = 16384, bytes = (size_t)96 << 20;       /* larger than one window */
    void *base = NULL;
    if (posix_memalign(&base, page, bytes)) { puts("map views: allocation failed"); g_failures++; return; }
    for (size_t i = 0; i < bytes / 4; i++) ((float *)base)[i] = (float)(i % 1000003);   /* exact under +0 */
    q36_map *m = q36_map_new(base, bytes);
    size_t offs[] = {0, 1000, ((size_t)31 << 20) + 12, ((size_t)64 << 20) - 8, ((size_t)90 << 20)};
    size_t lens[] = {4096, (size_t)20 << 20, (size_t)40 << 20, 64, (size_t)6 << 20};
    int bad = 0;
    for (int i = 0; i < 5; i++) {
        q36_tensor *v = q36_map_view(m, offs[i], lens[i]);
        if (!v || memcmp(q36_tensor_data(v), (char *)base + offs[i], lens[i])) bad++;
        /* the GPU sees the same bytes: copy the first word through a kernel */
        q36_tensor *out = q36_tensor_new(4);
        if (v && (offs[i] % 4) == 0 && lens[i] >= 4) {
            q36_tensor *zero = q36_tensor_new(4);
            q36_add(out, v, zero, 1);
            q36_gpu_sync();
            if (memcmp(q36_tensor_data(out), (char *)base + offs[i], 4)) bad++;
            q36_tensor_free(zero);
        }
        q36_tensor_free(out);
        q36_tensor_free(v);
    }
    printf("%-34s %s  views=5 bad=%d\n", "map views over 96 MiB", bad ? "FAIL" : "ok  ", bad);
    if (bad) g_failures++;
    q36_map_free(m);
    free(base);
}

/* Two single-token updates use the same compensated state and outputs as an
 * exact two-row batch, including the first token's snapshot. Nonzero low
 * planes and dimensions that leave partial SIMD groups exercise the handoff. */
static void test_delta_row_bits(int kd, int vd) {
    int saved_exact = g_test_exact;
    int vh = 4, kh = 2, cd = 2 * kh * kd + vh * vd;
    size_t cells = (size_t)vh * kd * vd, width = (size_t)vh * vd;
    q36_shape s = {0};
    s.rows = 2; s.v_heads = vh; s.k_heads = kh; s.k_dim = kd; s.v_dim = vd; s.conv_dim = cd; s.snap = 1;
    q36_tensor *qk = rand_tensor((size_t)2 * cd, 0.05f), *conv = rand_tensor((size_t)2 * cd, 1.0f);
    q36_tensor *decay = rand_tensor((size_t)2 * vh, 0.1f), *beta = rand_tensor((size_t)2 * vh, 0.2f);
    q36_tensor *state[2], *out[2], *snap[2];
    for (int i = 0; i < 2 * vh; i++) { F(decay)[i] += 0.8f; F(beta)[i] += 0.5f; }
    state[0] = rand_tensor(2 * cells, 0.01f);
    for (size_t i = cells; i < 2 * cells; i++) F(state[0])[i] *= 1e-7f;
    state[1] = q36_tensor_upload(F(state[0]), 2 * cells * 4);
    for (int run = 0; run < 2; run++) {
        out[run] = q36_tensor_new(2 * width * 4);
        snap[run] = q36_tensor_new(2 * cells * 4);
    }
    test_set_exact(1);
    int ok = q36_dn_delta(out[0], state[0], qk, conv, decay, beta, snap[0], &s);
    for (int row = 0; row < 2; row++) {
        q36_tensor *qr = q36_tensor_view(qk, (size_t)row * cd * 4, (size_t)cd * 4);
        q36_tensor *cr = q36_tensor_view(conv, (size_t)row * cd * 4, (size_t)cd * 4);
        q36_tensor *dr = q36_tensor_view(decay, (size_t)row * vh * 4, (size_t)vh * 4);
        q36_tensor *br = q36_tensor_view(beta, (size_t)row * vh * 4, (size_t)vh * 4);
        q36_tensor *yr = q36_tensor_view(out[1], (size_t)row * width * 4, width * 4);
        s.rows = 1; s.snap = row ? 0 : 1;
        ok = q36_dn_delta(yr, state[1], qr, cr, dr, br, snap[1], &s) && ok;
        q36_tensor_free(qr); q36_tensor_free(cr); q36_tensor_free(dr); q36_tensor_free(br); q36_tensor_free(yr);
    }
    sync_or_die();
    int same_state = !memcmp(F(state[0]), F(state[1]), 2 * cells * 4);
    int same_out = !memcmp(F(out[0]), F(out[1]), 2 * width * 4);
    int same_snap = !memcmp(F(snap[0]), F(snap[1]), 2 * cells * 4);
    printf("delta row bits kd=%d vd=%d     %s (state %s, output %s, snapshot %s)\n", kd, vd,
           ok && same_state && same_out && same_snap ? "ok" : "FAIL", same_state ? "same" : "DIFFERENT",
           same_out ? "same" : "DIFFERENT", same_snap ? "same" : "DIFFERENT");
    g_failures += !ok + !same_state + !same_out + !same_snap;
    for (int run = 0; run < 2; run++) {
        q36_tensor_free(state[run]); q36_tensor_free(out[run]); q36_tensor_free(snap[run]);
    }
    q36_tensor_free(qk); q36_tensor_free(conv); q36_tensor_free(decay); q36_tensor_free(beta);
    test_set_exact(saved_exact);
}

/* Shared projections use the fast dense reduction only in fast mode. The
 * router's ordered reduction stays the exact-mode reference. */
static void test_shared_dot_modes(int rows, int in, int out) {
    int saved_exact = g_test_exact;
    q36_tensor *q = rand_i8((size_t)out * in), *sc = rand_tensor((size_t)out, 0.01f);
    q36_tensor *x = rand_tensor((size_t)rows * in, 1.f), *y[4];
    size_t bytes = (size_t)rows * out * 4;
    for (int i = 0; i < 4; i++) y[i] = q36_tensor_new(bytes);
    test_set_exact(0);
    int ok = q36_dot_i8(y[0], x, q, sc, rows, in, out) && q36_shared_dot(y[1], x, q, sc, rows, in, out);
    test_set_exact(1);
    ok = q36_router_dot(y[2], x, q, sc, rows, in, out) && q36_shared_dot(y[3], x, q, sc, rows, in, out) && ok;
    sync_or_die();
    int fast_same = !memcmp(F(y[0]), F(y[1]), bytes), exact_same = !memcmp(F(y[2]), F(y[3]), bytes);
    printf("shared modes %dx%d->%d %s (fast %s, exact %s)\n", rows, in, out,
           ok && fast_same && exact_same ? "ok" : "FAIL", fast_same ? "dense" : "DIFFERENT", exact_same ? "router" : "DIFFERENT");
    g_failures += !ok + !fast_same + !exact_same;
    for (int i = 0; i < 4; i++) q36_tensor_free(y[i]);
    q36_tensor_free(q); q36_tensor_free(sc); q36_tensor_free(x);
    test_set_exact(saved_exact);
}

/* q36_dot_i8_cols: the first `cols` columns of a fast prompt product, in rows `out` wide: the same
 * bits as the whole product's, the other columns untouched, and refused in exact mode and for 8 rows. */
static void test_dot_cols(int rows, int in, int out, int cols) {
    int saved_exact = g_test_exact;
    test_set_exact(0);
    q36_tensor *q = rand_i8((size_t)out * in), *sc = rand_tensor((size_t)out, 0.01f);
    q36_tensor *x = rand_tensor((size_t)rows * in, 1.f);
    size_t n = (size_t)rows * out;
    q36_tensor *whole = q36_tensor_new(n * 4), *part = q36_tensor_new(n * 4);
    for (size_t i = 0; i < n; i++) F(part)[i] = -12345.f;
    int ok = q36_dot_i8(whole, x, q, sc, rows, in, out) && q36_dot_i8_cols(part, out, x, q, sc, rows, in, cols);
    int refused = !q36_dot_i8_cols(part, out, x, q, sc, 8, in, cols);
    test_set_exact(1);
    refused = !q36_dot_i8_cols(part, out, x, q, sc, rows, in, cols) && refused;
    sync_or_die();
    size_t bad = 0, written = 0;
    for (size_t i = 0; i < n; i++) {
        if ((int)(i % out) < cols) bad += memcmp(F(part) + i, F(whole) + i, 4) != 0;
        else written += F(part)[i] != -12345.f;
    }
    printf("dot_i8_cols %dx%d->%d, %d columns %s (different %zu, others written %zu, refused %s)\n", rows, in, out,
           cols, ok && !bad && !written && refused ? "ok" : "FAIL", bad, written, refused ? "yes" : "NO");
    g_failures += !ok + (bad > 0) + (written > 0) + !refused;
    q36_tensor_free(q); q36_tensor_free(sc); q36_tensor_free(x); q36_tensor_free(whole); q36_tensor_free(part);
    test_set_exact(saved_exact);
}

/* q36_dot_i8_heads: the query parts of heads of `group` query then `group` gate outputs, packed,
 * the same bits as those columns of the whole product (the kernel computes each output alone),
 * and refused where the matrix kernel does not apply (exact mode, 8 rows). */
static void test_dot_heads(int rows, int in, int heads, int group) {
    int saved_exact = g_test_exact;
    test_set_exact(0);
    int out = 2 * heads * group, half = heads * group;
    q36_tensor *q = rand_i8((size_t)out * in), *sc = rand_tensor((size_t)out, 0.01f);
    q36_tensor *x = rand_tensor((size_t)rows * in, 1.f);
    q36_tensor *whole = q36_tensor_new((size_t)rows * out * 4), *part = q36_tensor_new((size_t)rows * half * 4);
    int ok = q36_dot_i8(whole, x, q, sc, rows, in, out) &&
             q36_dot_i8_heads(part, x, q, sc, rows, in, half, group, 2 * group);
    int refused = !q36_dot_i8_heads(part, x, q, sc, 8, in, half, group, 2 * group);
    test_set_exact(1);
    refused = !q36_dot_i8_heads(part, x, q, sc, rows, in, half, group, 2 * group) && refused;
    sync_or_die();
    size_t bad = 0;
    for (int r = 0; r < rows; r++)
        for (int o = 0; o < half; o++) {
            const float *a = F(part) + (size_t)r * half + o, *b = F(whole) + (size_t)r * out + o / group * 2 * group + o % group;
            bad += memcmp(a, b, 4) != 0;
        }
    printf("dot_i8_heads %dx%d, %d heads of 2x%d %s (different %zu, refused %s)\n", rows, in, heads, group,
           ok && !bad && refused ? "ok" : "FAIL", bad, refused ? "yes" : "NO");
    g_failures += !ok + (bad > 0) + !refused;
    q36_tensor_free(q); q36_tensor_free(sc); q36_tensor_free(x); q36_tensor_free(whole); q36_tensor_free(part);
    test_set_exact(saved_exact);
}

/* Decode's two routers in one dispatch (q36_router_dot2): in exact mode the
 * bits of two q36_router_dot calls, in fast mode of two q36_dot_i8 calls. */
static void test_router_dot2(int in, int out) {
    int saved_exact = g_test_exact;
    q36_tensor *q = rand_i8((size_t)out * in), *sc = rand_tensor((size_t)out, 0.01f), *x = rand_tensor((size_t)in, 1.0f);
    q36_tensor *q2 = rand_i8((size_t)out * in), *sc2 = rand_tensor((size_t)out, 0.01f), *x2 = rand_tensor((size_t)in, 1.0f);
    q36_tensor *y = q36_tensor_new((size_t)out * 4), *y2 = q36_tensor_new((size_t)out * 4);
    q36_tensor *r = q36_tensor_new((size_t)out * 4), *r2 = q36_tensor_new((size_t)out * 4);
    for (int exact = 1; exact >= 0; exact--) {
        test_set_exact(exact);
        int ok = q36_router_dot2(y, x, q, sc, y2, x2, q2, sc2, in, out) &&
                 (exact ? q36_router_dot(r, x, q, sc, 1, in, out) && q36_router_dot(r2, x2, q2, sc2, 1, in, out)
                        : q36_dot_i8(r, x, q, sc, 1, in, out) && q36_dot_i8(r2, x2, q2, sc2, 1, in, out));
        sync_or_die();
        int same = ok && !memcmp(F(y), F(r), (size_t)out * 4) && !memcmp(F(y2), F(r2), (size_t)out * 4);
        printf("router_dot2 %dx%d %-5s = two single calls: %s\n", in, out, exact ? "exact" : "fast", same ? "ok" : "FAIL");
        g_failures += !same;
    }
    q36_tensor *t[] = {q, sc, x, q2, sc2, x2, y, y2, r, r2};
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) q36_tensor_free(t[i]);
    test_set_exact(saved_exact);
}

/* The keep-alive (q36_gpu_keepalive): its thread runs q36_keepalive on a second
 * queue until the deadline, the engine's batches give their results beside it,
 * and 0 stops it at once. */
static void test_keepalive(void) {
    uint64_t before = q36_gpu_keepalive_runs();
    q36_gpu_keepalive(0.3);
    test_rmsnorm(3, 2048, 0);                   /* the engine's kernels beside it */
    struct timespec pause = {0, 400 * 1000 * 1000};
    nanosleep(&pause, NULL);                    /* past the deadline: the thread waits */
    uint64_t ran = q36_gpu_keepalive_runs() - before, idle = q36_gpu_keepalive_runs();
    nanosleep(&pause, NULL);
    int asleep = q36_gpu_keepalive_runs() == idle;
    q36_gpu_keepalive(5.0);
    q36_gpu_keepalive(0);                       /* renewed, then stopped */
    nanosleep(&pause, NULL);
    uint64_t stopped = q36_gpu_keepalive_runs();
    nanosleep(&pause, NULL);
    const char *off = getenv("QWEN36_METAL_KEEPALIVE");
    int ok = (!off || strcmp(off, "0")) ? ran > 0 : ran == 0;   /* QWEN36_METAL_KEEPALIVE=0: never starts */
    ok = ok && asleep && q36_gpu_keepalive_runs() == stopped && q36_gpu_sync();
    printf("keep-alive: %llu dispatches in 0.3 s, then none past the deadline or after 0: %s\n",
           (unsigned long long)ran, ok ? "ok" : "FAIL");
    g_failures += !ok;
}

/* The exact conv (q36_dn_conv) gives a row the same bits in a batch of any
 * size: `rows` rows at once against the same rows in batches of `part`, the
 * history carried from one batch to the next. */
static void test_dn_conv_split(int rows, int part, int cd, int ck) {
    int saved_exact = g_test_exact;
    test_set_exact(1);
    q36_shape s = {0};
    s.conv_dim = cd; s.conv_kernel = ck;
    q36_tensor *x = rand_tensor((size_t)rows * cd, 1.0f), *w = rand_tensor((size_t)cd * ck, 0.5f);
    q36_tensor *ring1 = rand_tensor((size_t)cd * (ck - 1), 1.0f), *ring2 = q36_tensor_new((size_t)cd * (ck - 1) * 4);
    memcpy(F(ring2), F(ring1), (size_t)cd * (ck - 1) * 4);
    q36_tensor *y1 = q36_tensor_new((size_t)rows * cd * 4), *y2 = q36_tensor_new((size_t)rows * cd * 4);
    s.rows = rows;
    int ok = q36_dn_conv(y1, ring1, x, w, NULL, &s);
    for (int r0 = 0; ok && r0 < rows; r0 += part) {
        int n = rows - r0 < part ? rows - r0 : part;
        q36_tensor *xv = q36_tensor_view(x, (size_t)r0 * cd * 4, (size_t)n * cd * 4);
        q36_tensor *yv = q36_tensor_view(y2, (size_t)r0 * cd * 4, (size_t)n * cd * 4);
        s.rows = n;
        ok = q36_dn_conv(yv, ring2, xv, w, NULL, &s);
        q36_tensor_free(xv); q36_tensor_free(yv);
    }
    sync_or_die();
    int same = ok && !memcmp(F(y1), F(y2), (size_t)rows * cd * 4) && !memcmp(F(ring1), F(ring2), (size_t)cd * (ck - 1) * 4);
    printf("dn conv exact, %d rows = batches of %d: %s\n", rows, part, same ? "ok" : "FAIL");
    g_failures += !same;
    q36_tensor *t[] = {x, w, ring1, ring2, y1, y2};
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) q36_tensor_free(t[i]);
    test_set_exact(saved_exact);
}

/* The exact delta rule over a long prompt: its state is two planes (high and
 * low), so after many rows with a decay near 1 it stays within float rounding
 * of a double recurrence driven by the GPU's own decay, beta, q, k and v; one
 * float plane drifts past that. */
static void test_deltanet_long(int rows) {
    int saved_exact = g_test_exact, H = 64, vh = 2, kh = 1, kd = 32, vd = 16, ck = 4, cd = 2 * kh * kd + vh * vd;
    test_set_exact(1);
    q36_shape s = {0};
    s.rows = rows; s.hidden = H; s.v_heads = vh; s.k_heads = kh; s.k_dim = kd; s.v_dim = vd;
    s.conv_kernel = ck; s.conv_dim = cd;
    q36_tensor *x = rand_tensor((size_t)rows * H, 1.0f);
    q36_tensor *a = rand_tensor((size_t)vh * H, 0.01f), *b = rand_tensor((size_t)vh * H, 0.05f);
    q36_tensor *alog = q36_tensor_new((size_t)vh * 4), *dt = q36_tensor_new((size_t)vh * 4);
    for (int h = 0; h < vh; h++) { F(alog)[h] = -7.0f + 0.1f * frand(); F(dt)[h] = 0.5f; }   /* decay about 0.999 */
    q36_tensor *proj = rand_tensor((size_t)rows * cd, 1.0f), *w = rand_tensor((size_t)cd * ck, 0.5f);
    q36_tensor *ring = q36_tensor_new((size_t)cd * (ck - 1) * 4), *state = q36_tensor_new((size_t)2 * vh * kd * vd * 4);
    q36_tensor *decay = q36_tensor_new((size_t)rows * vh * 4), *beta = q36_tensor_new((size_t)rows * vh * 4);
    q36_tensor *conv = q36_tensor_new((size_t)rows * cd * 4), *qk = q36_tensor_new((size_t)rows * cd * 4);
    q36_tensor *out = q36_tensor_new((size_t)rows * vh * vd * 4);
    int ok = q36_dn_aux(decay, beta, x, a, b, alog, dt, &s) && q36_dn_conv(conv, ring, proj, w, NULL, &s) &&
             q36_dn_l2(qk, conv, &s) && q36_dn_delta(out, state, qk, conv, decay, beta, NULL, &s);
    sync_or_die();
    size_t cells = (size_t)vh * kd * vd;
    double *rs = calloc(cells, sizeof(double)), *ref = malloc(sizeof(double) * cells);
    for (int r = 0; r < rows; r++)
        for (int h = 0; h < vh; h++) {
            const float *k = F(qk) + (size_t)r * cd + kh * kd + (h / (vh / kh)) * kd;
            for (int v = 0; v < vd; v++) {
                double u = 0;
                for (int i = 0; i < kd; i++) {
                    double *st = &rs[((size_t)h * kd + i) * vd + v];
                    *st *= F(decay)[r * vh + h];
                    u += *st * k[i];
                }
                double delta = ((double)F(conv)[(size_t)r * cd + 2 * kh * kd + h * vd + v] - u) * F(beta)[r * vh + h];
                for (int i = 0; i < kd; i++) rs[((size_t)h * kd + i) * vd + v] += delta * k[i];
            }
        }
    float *got = malloc(sizeof(float) * cells);
    for (size_t i = 0; i < cells; i++) { got[i] = (float)((double)F(state)[i] + (double)F(state)[cells + i]); ref[i] = rs[i]; }
    char name[80];
    snprintf(name, sizeof name, "dn exact state after %d rows", rows);
    if (!ok) { puts("dn exact long: a kernel refused the shape FAIL"); g_failures++; }
    check(name, got, ref, NULL, cells, 2e-6);
    free(rs); free(ref); free(got);
    q36_tensor *t[] = {x, a, b, alog, dt, proj, w, ring, state, decay, beta, conv, qk, out};
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) q36_tensor_free(t[i]);
    test_set_exact(saved_exact);
}

/* Exact attention whose first position takes nearly all the weight and 8191
 * later ones about 2e-9 each: a plain float sum drops every small term (the
 * denominator would stay 1), the compensated one keeps their 1.7e-5. Both the
 * one-group kernel (no scratch) and decode's three steps (scratch). */
static void test_attention_dominant(int rows, int scratch) {
    int saved_exact = g_test_exact, hd = 64, cap = 8192, pos0 = cap - rows;
    test_set_exact(1);
    q36_shape s = {0};
    s.rows = rows; s.q_heads = 1; s.kv_heads = 1; s.head_dim = hd; s.q_head_dim = hd; s.pos = pos0; s.cap = cap;
    q36_tensor *query = rand_tensor((size_t)rows * hd, 1.0f), *kc = rand_tensor((size_t)cap * hd, 0.01f);
    q36_tensor *vc = rand_tensor((size_t)cap * hd, 1.0f), *out = q36_tensor_new((size_t)rows * hd * 4);
    q36_tensor *sc = scratch ? q36_tensor_new(attn_scratch(1, cap)) : NULL;
    double qq = 0;
    for (int d = 0; d < hd; d++) qq += (double)F(query)[d] * F(query)[d];
    for (int d = 0; d < hd; d++) F(kc)[d] = (float)(F(query)[d] * 20.0 * sqrt((double)hd) / qq);
    int ok = q36_attn(out, query, kc, vc, sc, &s);
    sync_or_die();
    double *ref = malloc(sizeof(double) * rows * hd);
    for (int r = 0; r < rows; r++) {
        int n = pos0 + r + 1;
        double *sco = malloc(sizeof(double) * n), mx = -INFINITY, sum = 0;
        for (int i = 0; i < n; i++) {
            double v = 0;
            for (int d = 0; d < hd; d++) v += (double)F(query)[(size_t)r * hd + d] * F(kc)[(size_t)i * hd + d];
            sco[i] = v / sqrt((double)hd);
            if (sco[i] > mx) mx = sco[i];
        }
        for (int i = 0; i < n; i++) { sco[i] = exp(sco[i] - mx); sum += sco[i]; }
        for (int d = 0; d < hd; d++) {
            double v = 0;
            for (int i = 0; i < n; i++) v += sco[i] * F(vc)[(size_t)i * hd + d];
            ref[(size_t)r * hd + d] = v / sum;
        }
        free(sco);
    }
    char name[96];
    snprintf(name, sizeof name, "attn exact, one dominant of %d, %s", cap, scratch ? "3 steps" : "one group");
    if (!ok) { printf("%s: refused FAIL\n", name); g_failures++; }
    check(name, F(out), ref, NULL, (size_t)rows * hd, 3e-6);
    free(ref);
    q36_tensor_free(query); q36_tensor_free(kc); q36_tensor_free(vc); q36_tensor_free(out); q36_tensor_free(sc);
    test_set_exact(saved_exact);
}

int main(void) {
    /* GitHub's Apple Silicon runners expose Metal through the Apple Paravirtual device, where command
     * buffers never complete (tests/test_backend_metal.mm skips it too), and which need not meet the
     * engine's requirements: there only the kernels' compilation is checked. */
    const char *device = q36_gpu_default_device_name();
    if (!*device) { puts("SKIPPED: no Metal device"); return 0; }
    if (strstr(device, "Apple Paravirtual device")) {
        if (!q36_gpu_compile_check()) { printf("FAILED: %s: the kernels did not compile (see above)\n", device); return 1; }
        printf("SKIPPED: %s: the kernels compile, none runs here\n", device);
        return 0;
    }
    /* The engine needs the same: a Metal device with unified memory, and kernels that compile on it. */
    if (!q36_gpu_init()) { puts("FAILED: no usable Metal device, or the kernels did not compile (see above)"); return 1; }
    test_set_exact(1);                         /* do not inherit the shell's mode for legacy checks */
    printf("Metal device: %s\n", q36_gpu_device_name());

    /* real model shapes (Qwen3.6-35B-A3B) */
    test_rmsnorm(1, 2048, 0);
    test_rmsnorm(7, 2048, 0);
    test_rmsnorm(3, 128, 1);
    test_dot(1, 2048, 8192, 0);
    test_dot(5, 2048, 1024, 0);
    test_dot(1, 4096, 2048, 0);
    test_dot(1, 2048, 256, 1);
    test_dot(3, 2048, 256, 1);
    test_elementwise(8192);
    test_deltanet(1, 2048, 32, 16, 128, 128, 4, 0);
    test_deltanet(5, 2048, 32, 16, 128, 128, 4, 0);
    test_attention(1, 16, 2, 256, 512, 64, 37);
    test_attention(4, 16, 2, 256, 512, 64, 0);
    test_router(1, 256, 8, 1, 1, 0);
    test_router(6, 256, 8, 1, 1, 0);
    test_router(3, 256, 8, 8, 4, 0);
    test_shared_gate(3, 2048);
    test_experts(2048, 512, 8, 40, 1);
    test_experts(2048, 512, 8, 40, 9);

    /* small shapes */
    test_rmsnorm(2, 64, 0);
    test_dot(2, 64, 32, 0);
    test_dot(2, 64, 8, 1);
    test_deltanet(2, 64, 2, 1, 8, 8, 4, 0);
    test_attention(2, 2, 1, 8, 16, 8, 3);
    test_router(2, 8, 8, 1, 1, 0);
    test_experts(64, 64, 8, 8, 5);
    test_experts(128, 64, 4, 13, 33);

    test_map_views();

    /* last, so the cases above keep their random inputs */
    test_dot(2, 72, 5, 1);                      /* no fma tail, a scalar run after the 16-wide loads */
    test_dot(3, 75, 9, 1);                      /* all three parts of the router dot */
    test_router(4, 256, 8, 1, 1, 1);
    test_router(2, 256, 8, 8, 4, 1);
    test_attention(1, 16, 2, 256, 512, 64, 8191);   /* decode at 8192 positions */
    test_attention(1, 16, 2, 256, 512, 64, 100);    /* positions not a multiple of 32 */
    test_attention(1, 4, 2, 64, 128, 32, 70);       /* head_dim below 256 */
    /* Prompt batches on the fast FP32 path: within the tolerance of the
     * double reference, not bit-identical to the exact kernel. */
    test_set_exact(0);
    test_dot(64, 2048, 8192, 0);
    test_dot(67, 2048, 512, 0);                 /* a short tile of rows */
    test_dot(33, 64, 64, 0);
    test_dot(9, 4096, 2048, 0);
    test_attention(72, 2, 1, 64, 128, 32, 37);      /* prompt chunks of 64 + 8 rows */
    test_attention_fast(100, 16, 2, 256, 300);      /* real shapes, two chunks */
    test_attention_half(1, 5, 256);                 /* decode near the start: GQA, one part */
    test_attention_half(1, 300, 256);               /* decode over two parts */
    test_attention_half(1, 8191, 256);
    test_attention_half(4, 100, 256);               /* a short batch on the matrix path */
    test_attention_half(70, 200, 256);              /* two chunks */
    test_attention_half(20, 50, 64);                /* the 16x32 tiles of head_dim below 256 */
    test_attention_half(20, 50, 128);
    test_attention_half(20, 50, 192);
    test_set_exact(1);                       /* routed experts of a prompt batch, both paths */
    test_experts(2048, 512, 8, 40, 33);
    test_set_exact(0);
    test_experts(2048, 512, 8, 40, 33);
    test_experts(64, 64, 8, 8, 21);
    test_attention(9, 4, 2, 64, 128, 32, 0);        /* a short first batch */
    test_deltanet(40, 2048, 32, 16, 128, 128, 4, 0);  /* the fast delta rule (q36_dn_scan) */
    test_deltanet(24, 2048, 32, 16, 128, 128, 4, 1);  /* fast prompt, then exact */
    test_deltanet(17, 64, 4, 2, 32, 12, 4, 0);        /* one key per lane, a part threadgroup */
    test_deltanet(10, 64, 2, 1, 96, 8, 4, 1);         /* three keys per lane */
    test_deltanet_snapshot(20, 13, 0);                  /* the state copied at a row */
    test_deltanet_snapshot(5, 1, 0);                    /* after the first row */
    test_deltanet_snapshot(9, 9, 0);                    /* after the last row */
    test_deltanet_snapshot(40, 21, 1);
    test_deltanet_snapshot(30, 30, 1);
    for (int exact = 0; exact < 2; exact++) {   /* the row kernels, in both modes */
        test_row_kernels(2048, 256, 8, exact);  /* the real shapes */
        test_row_kernels(75, 100, 4, exact);    /* tails of every loop */
        test_row_kernels(64, 8, 8, exact);      /* a small router: every expert chosen */
    }
    test_delta_row_bits(128, 128);
    test_delta_row_bits(96, 35);
    test_delta_row_bits(1, 7);
    test_set_exact(0);
    test_dot(1, 4, 7, 0);                       /* fewer vector groups than lanes */
    test_dot(3, 68, 19, 0);                     /* partial 128-input SIMD tile */
    test_dot(1, 75, 7, 0);                      /* fallback when in is not divisible by four */
    test_shared_dot_modes(1, 2048, 512);
    test_shared_dot_modes(8, 512, 2048);
    test_shared_dot_modes(9, 64, 64);
    test_shared_dot_modes(3, 75, 19);
    test_set_exact(0);
    test_deltanet(1, 2048, 32, 16, 128, 128, 4, 0); /* float scan in generation */
    test_deltanet(8, 64, 4, 2, 32, 12, 4, 1);      /* short fast batch, exact handoff */
    test_deltanet_snapshot(5, 1, 1);               /* snapshot from a one-row fast prefix */
    test_deltanet_snapshot(8, 8, 1);
    test_deltanet_high_only(1, 0);                 /* decode: the scan */
    test_deltanet_high_only(1, 1);
    test_deltanet_high_only(6, 4);                 /* a short batch: the scan */
    test_deltanet_high_only(100, 37);              /* a prompt batch: chunks of 32, split at the copy */
    test_attention(1, 16, 2, 256, 512, 64, 127);   /* fast GQA threshold: 128 positions */
    test_attention(1, 16, 2, 256, 512, 64, 256);   /* second PV part with one position */
    test_attention(1, 16, 2, 256, 512, 64, 8192);  /* long GQA tail, full/short scratch */
    test_rmsnorm_fast(1, 2048, 0, 0);             /* decode, centered gamma */
    test_rmsnorm_fast(1, 2048, 1, 0);             /* decode, plain gamma */
    test_rmsnorm_fast(32, 128, 1, 0);             /* DeltaNet head normalization */
    test_rmsnorm_fast(3, 257, 0, 0);              /* incomplete SIMD tail */
    test_rmsnorm_fast(2, 33, 1, 1);               /* zero norm and positive epsilon */
    /* Both router paths against the independent double reference. FAST
     * covers decode and complete/large matrix batches; EXACT stays ordered. */
    test_set_exact(0);
    test_dot(1, 2048, 256, 1);
    test_dot(64, 2048, 256, 1);
    test_dot(512, 2048, 256, 1);
    test_dot(3, 75, 9, 1);                       /* scalar-width fallback remains valid */
    test_set_exact(1);
    test_dot(1, 2048, 256, 1);
    test_dot(64, 2048, 256, 1);
    test_set_exact(0);
    test_experts(64, 64, 8, 8, 12);            /* short slots: one partial tile each */
    test_experts(64, 64, 8, 8, 48);            /* a full tile and a partial one per slot */
    test_experts(64, 64, 8, 8, 28);            /* tails of 3 or 4 row blocks: q36_expert_*_mm4 */
    test_experts(64, 64, 8, 8, 76);            /* a full tile and a tail of 1 or 2 row blocks */
    test_set_exact(0);
    test_experts(192, 64, 3, 13, 9);            /* boundary stays on prompt matrix path */
    test_dot_heads(40, 2048, 16, 256);            /* the attention query parts, real shapes */
    test_dot_heads(9, 64, 3, 32);                 /* groups of one SIMD group, a tile of 128 across heads */
#if QCM_ON
    test_attention_cpu(200, 1100, 136);           /* the CPU's last rows, real shapes, two chunks */
    test_attention_cpu(40, 0, 7);                 /* from the first position, one chunk */
#endif
    test_dot_cols(64, 4096, 2048, 1408);          /* the output projection's GPU columns */
#if QCM_ON
    test_cpu_experts(2048, 512, 2, 3, 700);       /* the CPU's routed experts, real shapes */
    test_cpu_experts(128, 64, 2, 2, 600);         /* more than 512 choices: two products */
    test_cpu_dense(40, 2048, 1024);               /* the CPU's dense products, real width */
    test_cpu_dense(33, 64, 128);
    test_cpu_shared(40, 2048, 512);               /* the CPU's shared expert, real shapes */
    test_cpu_shared(9, 64, 64);
#endif
    test_dot_cols(33, 64, 256, 128);              /* a short tile of rows */
    test_router_dot2(2048, 256);                  /* decode's two routers in one dispatch */
    test_router_dot2(75, 9);
    test_dn_conv_split(12, 8, 8192, 4);           /* more than 8 rows against batches of 8 and 4 */
    test_dn_conv_split(20, 3, 96, 4);
    test_deltanet_long(4096);                     /* the two-plane state over a long prompt */
    test_attention_dominant(1, 0);                /* compensated sums: one-group kernel, decode */
    test_attention_dominant(1, 1);                /* decode's three steps */
    test_attention_dominant(2, 0);                /* prompt rows */
    test_keepalive();

    printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    q36_gpu_cleanup();
    return g_failures ? 1 : 0;
}
