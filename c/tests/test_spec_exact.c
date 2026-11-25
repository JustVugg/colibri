/* tests/test_spec_exact.c -- spec_exact.h: the drawn-draft rule emits tokens that follow
 * the target exactly, keeps 1 - TV(p, q) of its drafts, reduces to the old rule for a
 * point mass, stays exact across a two-row chain, and the nucleus it settles against is
 * the serve samplers' nucleus.  Also verifies that settling in probability space avoids
 * the top-p ban-and-resample leak (banning a rejected draft in logits and re-applying
 * top-p lets tokens outside the original nucleus through).
 *
 * Monte Carlo with fixed seeds: N draws per case, tolerances ~5 sigma. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../spec_exact.h"

#ifdef _WIN32
static void test_setenv(const char *k, const char *v) { _putenv_s(k, v); }
static void test_unsetenv(const char *k) { _putenv_s(k, ""); }
#else
static void test_setenv(const char *k, const char *v) { setenv(k, v, 1); }
static void test_unsetenv(const char *k) { unsetenv(k); }
#endif

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static uint64_t g_s = 0x243F6A8885A308D3ULL;
static double U(void) {   /* xorshift64*, 53-bit uniform in [0, 1) */
    g_s ^= g_s << 13; g_s ^= g_s >> 7; g_s ^= g_s << 17;
    return (double)(g_s >> 11) * (1.0 / 9007199254740992.0);
}
static double gauss(void) { double a = U() + 1e-12, b = U(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

enum { V = 64, N = 1000000 };

static double tv(const double *a, const float *b, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += fabs(a[i] - b[i]); return s / 2;
}

/* target logits and a drafter's: the drafter sees the target through noise and its
 * own temperature (sharper or flatter) */
static void rows(float *lt, float *ld, double noise, double sharp) {
    for (int i = 0; i < V; i++) {
        lt[i] = (float)(2.0 * gauss() - 0.08 * i);
        ld[i] = (float)(sharp * (lt[i] + noise * gauss()));
    }
}

static void one_case(const char *name, double noise, double sharp, float temp, float top_p, int point) {
    float lt[V], ld[V], p[V], scratch[V];
    rows(lt, ld, noise, sharp);
    if (spec_exact_target(p, lt, V, temp, top_p)) { CHECK(0, "%s: target", name); return; }
    SpecQ q;
    if (point) { float tmp[V]; spec_exact_target(tmp, ld, V, temp, 1.f); int a = 0; for (int i = 1; i < V; i++) if (ld[i] > ld[a]) a = i; specq_point(&q, a); }
    else specq_from_logits(&q, ld, V, temp, top_p, SPEC_Q_MAX);
    double hist[V] = {0}; long kept = 0;
    for (int t = 0; t < N; t++) {
        int d = specq_draw(&q, U()), k;
        memcpy(scratch, p, sizeof p);
        int tok = spec_exact_settle(scratch, V, &q, d, U(), U(), &k);
        hist[tok] += 1.0; kept += k;
    }
    for (int i = 0; i < V; i++) hist[i] /= N;
    double out_tv = tv(hist, p, V), acc = (double)kept / N;
    double want = point ? spec_exact_keep_point(p, V, &q) : spec_exact_keep_drawn(p, V, &q);
    double tv_pq = 1.0 - spec_exact_keep_drawn(p, V, &q);
    printf("%-28s TV(p,q)=%.4f  keep=%.4f (expected %.4f)  TV(out,p)=%.5f\n", name, tv_pq, acc, want, out_tv);
    CHECK(out_tv < 0.004, "%s: output does not follow p (TV %.5f)", name, out_tv);
    CHECK(fabs(acc - want) < 0.003, "%s: acceptance %.4f, expected %.4f", name, acc, want);
    for (int i = 0; i < V; i++) if (p[i] == 0.f) CHECK(hist[i] == 0.0, "%s: token %d outside the nucleus emitted", name, i);
}

/* Two rows, each target row conditional on the token before (a Markov chain), each
 * drafter row conditional on the DRAFT before: the joint distribution of the two
 * emitted tokens must be p1(a) p2(b | a). */
enum { V2 = 8, N2 = 2000000 };
static void chain_case(void) {
    float lt1[V2], ld1[V2], lt2[V2][V2], ld2[V2][V2];
    float p1[V2], p2[V2][V2];
    SpecQ q1, q2[V2];
    for (int i = 0; i < V2; i++) { lt1[i] = (float)(1.5 * gauss()); ld1[i] = (float)(lt1[i] + 0.7 * gauss()); }
    spec_exact_target(p1, lt1, V2, 1.f, 1.f);
    specq_from_logits(&q1, ld1, V2, 1.f, 1.f, SPEC_Q_MAX);
    for (int a = 0; a < V2; a++) {
        for (int i = 0; i < V2; i++) { lt2[a][i] = (float)(1.5 * gauss()); ld2[a][i] = (float)(lt2[a][i] + 0.7 * gauss()); }
        spec_exact_target(p2[a], lt2[a], V2, 1.f, 1.f);
        specq_from_logits(&q2[a], ld2[a], V2, 1.f, 1.f, SPEC_Q_MAX);
    }
    static double joint[V2][V2];
    memset(joint, 0, sizeof joint);
    float s[V2];
    for (int t = 0; t < N2; t++) {
        int d1 = specq_draw(&q1, U()), d2 = specq_draw(&q2[d1], U()), k;
        memcpy(s, p1, sizeof s);
        int a = spec_exact_settle(s, V2, &q1, d1, U(), U(), &k), b;
        if (k) { memcpy(s, p2[a], sizeof s); b = spec_exact_settle(s, V2, &q2[a], d2, U(), U(), &k); }
        else {   /* the round ends; the next token is a plain draw from p2(. | a) */
            double u = U(), c = 0; b = V2 - 1;
            for (int i = 0; i < V2; i++) { c += p2[a][i]; if (u < c) { b = i; break; } }
        }
        joint[a][b] += 1.0;
    }
    double err = 0;
    for (int a = 0; a < V2; a++) for (int b = 0; b < V2; b++) err += fabs(joint[a][b] / N2 - (double)p1[a] * p2[a][b]);
    printf("%-28s TV(joint, p1*p2)=%.5f\n", "two-row chain", err / 2);
    CHECK(err / 2 < 0.004, "two-row chain: joint TV %.5f", err / 2);
}

/* the nucleus of spec_exact_target against a full sort */
static void nucleus_case(void) {
    enum { VN = 5000 };
    static float lo[VN], p[VN], r[VN];
    static SpecExactRank rk[VN];
    int bad = 0;
    for (int trial = 0; trial < 40; trial++) {
        for (int i = 0; i < VN; i++) lo[i] = (float)(trial % 3 == 0 ? floor(4 * gauss()) : 3 * gauss());   /* ties too */
        float temp = 0.4f + 0.05f * trial, top_p = trial % 4 == 0 ? 1.f : 0.5f + 0.012f * trial;
        spec_exact_target(p, lo, VN, temp, top_p);
        float mx = lo[0]; for (int i = 1; i < VN; i++) if (lo[i] > mx) mx = lo[i];
        double sum = 0; for (int i = 0; i < VN; i++) { rk[i].p = expf((lo[i] - mx) / temp); rk[i].id = i; sum += rk[i].p; }
        qsort(rk, VN, sizeof rk[0], spec_exact_rank_desc);
        double cut = (top_p > 0 && top_p < 1) ? top_p * sum : sum, kept = 0; int n = 0;
        while (n < VN && kept < cut) kept += rk[n++].p;
        for (int i = 0; i < VN; i++) r[i] = 0.f;
        for (int i = 0; i < n; i++) r[rk[i].id] = (float)(rk[i].p / kept);
        for (int i = 0; i < VN; i++) if (fabsf(r[i] - p[i]) > 1e-7f) { bad++; break; }
    }
    printf("%-28s %d of 40 rows differ from the full sort\n", "nucleus", bad);
    CHECK(bad == 0, "nucleus: %d rows differ", bad);
}

/* Banning a rejected draft in the logits and re-applying top-p is not exact:
 * it shifts the nucleus and lets tokens outside the original nucleus through,
 * whereas spec_exact_settle on the target probabilities stays exact. */
static void top_p_ban_leak_case(void) {
    float lo[V], p[V], s[V];
    for (int i = 0; i < V; i++) lo[i] = (float)(2.0 * gauss() - 0.05 * i);
    float temp = 1.f, top_p = 0.7f;
    spec_exact_target(p, lo, V, temp, top_p);
    int d = 0; for (int i = 1; i < V; i++) if (lo[i] > lo[d]) d = i;   /* greedy draft */
    double old_h[V] = {0}, new_h[V] = {0};
    for (int t = 0; t < N; t++) {
        /* naive: keep with p(d), else ban d in the logits and sample with top-p again */
        if (U() < p[d]) old_h[d] += 1;
        else {
            float l2[V]; memcpy(l2, lo, sizeof l2); l2[d] = -INFINITY;
            spec_exact_target(s, l2, V, temp, top_p);
            double u = U(), c = 0; int b = V - 1;
            for (int i = 0; i < V; i++) { c += s[i]; if (u < c) { b = i; break; } }
            old_h[b] += 1;
        }
        SpecQ q; specq_point(&q, d); int k;
        memcpy(s, p, sizeof s);
        new_h[spec_exact_settle(s, V, &q, d, U(), U(), &k)] += 1;
    }
    for (int i = 0; i < V; i++) { old_h[i] /= N; new_h[i] /= N; }
    double outside = 0; for (int i = 0; i < V; i++) if (p[i] == 0.f) outside += old_h[i];
    printf("%-28s ban-in-logits TV=%.4f (%.4f outside nucleus), settle TV=%.5f\n",
           "top-p rejection", tv(old_h, p, V), outside, tv(new_h, p, V));
    CHECK(tv(new_h, p, V) < 0.004, "point-mass settle not exact under top-p");
    CHECK(outside > 0.005, "ban-in-logits leak did not reproduce (test lost its point)");
}

static void gate_case(void) {
    float lt[V], ld[V], p[V];
    SpecSoftGate g;
    test_setenv("COLI_SPEC_SOFT", "auto"); spec_soft_gate_init(&g, 0);
    CHECK(g.mode == 2, "gate: auto");
    test_setenv("COLI_SPEC_SOFT", "0"); spec_soft_gate_init(&g, 2);
    CHECK(g.mode == 0, "gate: 0");
    test_setenv("COLI_SPEC_SOFT", "1"); spec_soft_gate_init(&g, 0);
    CHECK(g.mode == 1 && !spec_soft_gate_drawn(&g, 0.f), "gate: 1, greedy never draws");
    test_unsetenv("COLI_SPEC_SOFT");
    /* a drafter as sharp as the model: drawn wins */
    spec_soft_gate_init(&g, 2);
    for (int r = 0; r < 64; r++) {
        rows(lt, ld, 0.3, 1.0); SpecQ q;
        spec_exact_target(p, lt, V, 1.f, 1.f); specq_from_logits(&q, ld, V, 1.f, 1.f, SPEC_Q_MAX);
        CHECK(r >= SPEC_SOFT_WARMUP || !spec_soft_gate_drawn(&g, 1.f), "gate: drew during warm-up");
        spec_soft_gate_observe(&g, p, V, &q);
    }
    printf("%-28s matched drafter: drawn %.3f vs point %.3f -> %s\n", "gate", g.keep_drawn, g.keep_point,
           spec_soft_gate_drawn(&g, 1.f) ? "drawn" : "point");
    CHECK(spec_soft_gate_drawn(&g, 1.f), "gate: matched drafter should draw");
    /* a peaked model, a flat drafter: the point mass wins */
    spec_soft_gate_init(&g, 2);
    for (int r = 0; r < 64; r++) {
        rows(lt, ld, 0.0, 0.15); SpecQ q;
        spec_exact_target(p, lt, V, 0.5f, 1.f); specq_from_logits(&q, ld, V, 0.5f, 1.f, SPEC_Q_MAX);
        spec_soft_gate_observe(&g, p, V, &q);
    }
    printf("%-28s flat drafter:    drawn %.3f vs point %.3f -> %s\n", "gate", g.keep_drawn, g.keep_point,
           spec_soft_gate_drawn(&g, 1.f) ? "drawn" : "point");
    CHECK(!spec_soft_gate_drawn(&g, 1.f), "gate: flat drafter should keep the point mass");
}

int main(void) {
    one_case("drawn, T=1 noise .3", 0.3, 1.0, 1.f, 1.f, 0);
    one_case("drawn, T=.6 top-p .9", 0.3, 1.0, 0.6f, 0.9f, 0);
    one_case("drawn, T=1 noise 1", 1.0, 1.0, 1.f, 1.f, 0);
    one_case("drawn, sharp drafter", 0.3, 2.5, 1.f, 1.f, 0);
    one_case("drawn, flat drafter", 0.3, 0.3, 1.f, 1.f, 0);
    one_case("point, T=1 noise .3", 0.3, 1.0, 1.f, 1.f, 1);
    one_case("point, T=.7 top-p .8", 0.3, 1.0, 0.7f, 0.8f, 1);
    chain_case();
    nucleus_case();
    top_p_ban_leak_case();
    gate_case();
    if (g_fail) { printf("test_spec_exact: %d FAILED\n", g_fail); return 1; }
    printf("test_spec_exact: ok\n");
    return 0;
}
