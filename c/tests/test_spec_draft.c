/* spec_draft.h on its own: what prompt lookup proposes, and what the gate decides.
 *
 *   lookup  the longest suffix wins over a shorter one, the most recent occurrence
 *           over an older one, the proposal stops at the history's end (a
 *           continuation that runs into the suffix itself is fine), k caps it,
 *           min_n and max_n bound the n-gram, and nothing is proposed without an
 *           earlier occurrence or a token after it;
 *   gate    the expected tokens E(k) and the value of a depth follow the formula
 *           in the header; acceptance moves toward what was judged (and only the
 *           judged positions move); a measured forward cost replaces the fitted
 *           line; drafting stops when the measured cost outweighs the acceptance,
 *           comes back through the probe (sooner while the deeper verify's cost is
 *           unmeasured), and a plain step is forced after SPEC_REFRESH forwards
 *           without one; off drafts every proposal in full.
 *
 * cc -O2 tests/test_spec_draft.c -o tests/test_spec_draft -lm && ./tests/test_spec_draft */
#include <math.h>
#include <stdio.h>
#include "../spec_draft.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int same(const int *a, const int *b, int n) { for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0; return 1; }

static void test_lookup(void) {
    int out[8] = {0};
    /* the suffix "3 4" occurred at 2..3 and is followed by 5 6 7 */
    int h1[] = {1, 2, 3, 4, 5, 6, 7, 9, 3, 4};
    int n = spec_lookup(h1, 10, 2, 4, 5, out);
    int w1[] = {5, 6, 7, 9, 3};
    CHECK(n == 5 && same(out, w1, 5), "bigram continuation: n=%d", n);
    n = spec_lookup(h1, 10, 2, 4, 2, out);
    CHECK(n == 2 && out[0] == 5 && out[1] == 6, "k caps the proposal: n=%d", n);
    /* the most recent earlier occurrence wins: "8 9" at 0 (then 1) and at 3 (then 2) */
    int h2[] = {8, 9, 1, 8, 9, 2, 0, 8, 9};
    n = spec_lookup(h2, 9, 2, 4, 3, out);
    CHECK(n == 3 && out[0] == 2 && out[1] == 0 && out[2] == 8, "most recent occurrence: n=%d first %d", n, out[0]);
    /* the longest suffix wins: "7 8 9" occurs once (then 5), "8 9" more recently (then 6) */
    int h3[] = {7, 8, 9, 5, 1, 8, 9, 6, 2, 7, 8, 9};
    n = spec_lookup(h3, 12, 2, 4, 1, out);
    CHECK(n == 1 && out[0] == 5, "longest suffix: n=%d got %d", n, out[0]);
    n = spec_lookup(h3, 12, 2, 2, 1, out);
    CHECK(n == 1 && out[0] == 6, "max_n 2 takes the bigram: n=%d got %d", n, out[0]);
    /* a periodic history: the continuation runs up to the end of the history */
    int h4[] = {1, 2, 1, 2, 1, 2};
    n = spec_lookup(h4, 6, 2, 4, 5, out);
    CHECK(n >= 1 && out[0] == 1, "periodic: n=%d first %d", n, out[0]);
    for (int i = 0; i < n; i++) CHECK(out[i] == (i % 2 ? 2 : 1), "periodic token %d = %d", i, out[i]);
    /* nothing to find: no earlier occurrence, a history too short, min_n above what matches */
    int h5[] = {1, 2, 3, 4, 5};
    CHECK(spec_lookup(h5, 5, 2, 4, 5, out) == 0, "no repeat, no proposal");
    CHECK(spec_lookup(h5, 1, 1, 4, 5, out) == 0, "one token, no proposal");
    int h6[] = {4, 1, 2, 4};   /* unigram 4 repeats, no bigram does */
    CHECK(spec_lookup(h6, 4, 2, 4, 5, out) == 0, "min_n 2: a unigram match is not enough");
    n = spec_lookup(h6, 4, 1, 4, 5, out);
    CHECK(n == 3 && out[0] == 1 && out[1] == 2 && out[2] == 4, "min_n 1: the unigram's continuation, n=%d", n);
    CHECK(spec_lookup(h1, 10, 2, 4, 0, out) == 0, "k 0");
}

static void test_gate(void) {
    SpecGate g;
    spec_gate_init(&g, 0);
    double e;
    /* the priors: MTP 0.8, 0.6, 0.5; no forward measured, rho = SPEC_RHO0 */
    double v1 = spec_gate_value(&g, SPEC_SRC_MTP, 1, &e);
    CHECK(fabs(e - 1.8) < 1e-12 && fabs(v1 - 1.8 / (1.0 + SPEC_RHO0)) < 1e-12, "E(1) %g value %g", e, v1);
    double v3 = spec_gate_value(&g, SPEC_SRC_MTP, 3, &e);
    CHECK(fabs(e - (1 + 0.8 + 0.48 + 0.24)) < 1e-12 && fabs(v3 - e / (1.0 + 3 * SPEC_RHO0)) < 1e-12, "E(3) %g value %g", e, v3);
    CHECK(spec_gate_value(&g, SPEC_SRC_MTP, 0, NULL) == 1.0, "a plain step is worth 1");
    /* acceptance moves toward what was judged, and only the judged positions move */
    double a0 = g.acc[SPEC_SRC_LOOKUP][0], a2 = g.acc[SPEC_SRC_LOOKUP][2];
    spec_gate_result(&g, SPEC_SRC_LOOKUP, 2, 1);   /* draft 1 stood, draft 2 did not, 3+ unseen */
    CHECK(g.acc[SPEC_SRC_LOOKUP][0] > a0 && g.acc[SPEC_SRC_LOOKUP][1] < 0.5 && g.acc[SPEC_SRC_LOOKUP][2] == a2,
          "judged positions: %g %g %g", g.acc[SPEC_SRC_LOOKUP][0], g.acc[SPEC_SRC_LOOKUP][1], g.acc[SPEC_SRC_LOOKUP][2]);
    CHECK(g.prop[SPEC_SRC_LOOKUP][0] == 1 && g.hit[SPEC_SRC_LOOKUP][0] == 1 && g.prop[SPEC_SRC_LOOKUP][1] == 1 &&
          g.hit[SPEC_SRC_LOOKUP][1] == 0 && g.prop[SPEC_SRC_LOOKUP][2] == 0, "counts");
    /* measured costs: a row that costs nothing extra makes every depth pay */
    spec_gate_init(&g, 0);
    for (int i = 0; i < 4; i++) { spec_gate_forward(&g, 1, 0.100); spec_gate_forward(&g, 4, 0.100); }
    CHECK(fabs(spec_gate_rows_cost(&g, 4) - 1.0) < 1e-9 && g.rho < 0.1 + 1e-9, "free rows: cost %g rho %g",
          spec_gate_rows_cost(&g, 4), g.rho);
    double v;
    CHECK(spec_gate_pick(&g, SPEC_SRC_MTP, 3, &v) == 3 && v > 2.0, "free rows: the deepest verify (value %g)", v);
    /* rows that cost a full forward each, with 50% acceptance, never pay */
    spec_gate_init(&g, 0);
    for (int i = 0; i < 8; i++) {
        spec_gate_forward(&g, 1, 0.1);
        for (int r = 2; r <= 6; r++) spec_gate_forward(&g, r, 0.1 * r);
        spec_gate_result(&g, SPEC_SRC_LOOKUP, 1, 0);
        spec_gate_result(&g, SPEC_SRC_LOOKUP, 2, 1);
    }
    int picked = 0, probes = 0;
    for (int i = 0; i < SPEC_PROBE * 3; i++) {
        int k = spec_gate_pick(&g, SPEC_SRC_LOOKUP, 5, &v);
        if (k) { picked++; CHECK(k == 1, "a probe drafts one deeper than the best (0), got %d", k); }
        spec_gate_forward(&g, 1, 0.1);   /* the plain steps keep T(1) fresh */
    }
    probes = (int)g.probes[SPEC_SRC_LOOKUP];
    CHECK(picked == 3 && probes == 3, "costly rows: only the probes draft (%d picked, %d probes)", picked, probes);
    /* a deeper verify whose cost is barely measured is probed every SPEC_PROBE_NEW picks */
    spec_gate_init(&g, 0);
    for (int i = 0; i < 8; i++) { spec_gate_forward(&g, 1, 0.1); spec_gate_forward(&g, 2, 0.13); spec_gate_result(&g, SPEC_SRC_MTP, 2, 1); }
    int deeper = 0;
    for (int i = 0; i < 16; i++) {
        int k = spec_gate_pick(&g, SPEC_SRC_MTP, 3, NULL);
        if (k == 2) deeper++;
        CHECK(k == 1 || k == 2, "best 1 or its probe, got %d", k);
    }
    CHECK(deeper == 16 / SPEC_PROBE_NEW, "an unmeasured depth probed every %d picks: %d probes in 16", SPEC_PROBE_NEW, deeper);
    /* T(1) is kept measured: SPEC_REFRESH forwards without a plain step force one */
    spec_gate_init(&g, 0);
    spec_gate_forward(&g, 1, 0.1);
    for (int i = 0; i < SPEC_REFRESH; i++) spec_gate_forward(&g, 3, 0.11);
    CHECK(spec_gate_pick(&g, SPEC_SRC_MTP, 3, NULL) == 0, "a plain step after %d verifies", SPEC_REFRESH);
    spec_gate_forward(&g, 1, 0.1);
    CHECK(spec_gate_pick(&g, SPEC_SRC_MTP, 3, NULL) > 0, "drafting again once T(1) is fresh");
    /* the extra cost of a source's deeper drafts counts */
    spec_gate_init(&g, 0);
    spec_gate_forward(&g, 1, 0.1); spec_gate_forward(&g, 3, 0.1);
    double before = spec_gate_value(&g, SPEC_SRC_MTP, 2, NULL);
    spec_gate_draft_cost(&g, SPEC_SRC_MTP, 0.05);
    double after = spec_gate_value(&g, SPEC_SRC_MTP, 2, NULL);
    CHECK(after < before && fabs(before / after - 1.5) < 1e-9, "draft cost: %g -> %g", before, after);
    /* off: every proposal in full */
    spec_gate_init(&g, 1);
    for (int i = 0; i < 4; i++) spec_gate_forward(&g, 6, 10.0);
    CHECK(spec_gate_pick(&g, SPEC_SRC_LOOKUP, 5, NULL) == 5, "gate off drafts the whole proposal");
}

int main(void) {
    test_lookup();
    test_gate();
    if (fails) { printf("test_spec_draft: %d failure(s)\n", fails); return 1; }
    printf("test_spec_draft: prompt lookup and the gate behave as spec_draft.h says\n");
    return 0;
}
