/* test_degrade_zero.c — unit tests for the DEGRADE_ZERO miss-slot zero-fill logic.
 *
 * Strategy: mirrors test_ablate.c — a standalone mini-harness that reimplements
 * only the routing data structures the DEGRADE_ZERO block in moe() touches.
 * No colibri.c include, no Model, no weights, no disk I/O needed.
 *
 * The block under test (colibri.c, "DEGRADE_ZERO: zero-fill miss slots..."):
 *   1. Computes aggregate gate weight per unique expert across the batch.
 *   2. Marks cache hits as always-keep; marks cold misses below g_degrade_tau
 *      for zeroing.
 *   3. Rescue rule: if all of a position's experts would be dropped, reinstates
 *      the highest-gate-weight miss so no position has zero routed experts.
 *   4. Rewrites idxs[]/ws[]/keff[] removing dropped experts, renormalises weights,
 *      compacts uniq[], increments g_degrade_dropped.
 *
 * Properties verified:
 *   P1  OFF-BY-DEFAULT  — with g_degrade_zero=0, routing is byte-identical to
 *       the input (nothing is dropped or renormalised).
 *   P2  HITS-ARE-SAFE   — experts already resident (simulated via a mock
 *       "resident" set) are never dropped regardless of gate weight.
 *   P3  TAU-GATE        — cold misses with aggregate gate weight >= tau are kept;
 *       those below tau are dropped and counted in g_degrade_dropped.
 *   P4  RENORM-NORMTOPK — after dropping, surviving gate weights renormalise to
 *       sum 1 * routed_scale when norm_topk=1 (GLM-5.2 default).
 *   P5  RENORM-NONNORM  — with norm_topk=0 the surviving weights are rescaled by
 *       old_sum/new_sum to preserve output magnitude.
 *   P6  RESCUE-RULE     — when all of a position's experts are cold misses below
 *       tau, the highest-weight one is reinstated; g_degrade_dropped is NOT
 *       inflated for the rescued expert.
 *   P7  PREFILL-GUARD   — with S>4 (prefill batch) the block is a no-op; nothing
 *       is dropped even if g_degrade_zero=1 and all experts are below tau.
 *   P8  COUNTER         — g_degrade_dropped accumulates correctly across calls.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", (msg)); g_fails++; } \
    else         { printf("  ok:   %s\n", (msg)); } \
} while (0)
#define CHECKF(cond, msg, ...) do { \
    if (!(cond)) { printf("  FAIL: " msg "\n", __VA_ARGS__); g_fails++; } \
    else         { printf("  ok:   " msg "\n", __VA_ARGS__); } \
} while (0)

/* ---- mirror of the globals the block reads -------------------------------- */
static int   g_degrade_zero = 0;
static float g_degrade_tau  = 0.03f;
static long long g_degrade_dropped = 0;

/* ---- mirror of the Cfg fields the block reads ----------------------------- */
typedef struct { int norm_topk; float routed_scale; } Cfg;

/* ---- resident-set mock: a flat array of expert ids considered "in cache" -- */
#define MAX_RESIDENT 32
static int g_resident[MAX_RESIDENT];
static int g_nresident = 0;

static int is_resident(int eid) {
    for (int i = 0; i < g_nresident; i++)
        if (g_resident[i] == eid) return 1;
    return 0;
}

/* ---- the drop logic extracted verbatim from colibri.c moe() --------------- *
 * Parameters match the local variables in moe() at the insertion point:
 *   idxs[S*K], ws[S*K], keff[S], uniq[nu], nu, S, K, c->norm_topk/routed_scale.
 * Returns the new nu after compaction. */
static int degrade_zero_apply(int *idxs, float *ws, int *keff,
                               int *uniq, int nu, int S, int K,
                               const Cfg *c)
{
    if (!g_degrade_zero || S > 4) return nu;

    /* 1. aggregate gate weight per unique expert */
    float *dg_wsum = calloc((size_t)nu, sizeof(float));
    for (int s = 0; s < S; s++)
        for (int kk = 0; kk < keff[s]; kk++) {
            int e = idxs[s * K + kk];
            for (int j = 0; j < nu; j++)
                if (uniq[j] == e) { dg_wsum[j] += ws[s * K + kk]; break; }
        }

    /* 2. keep = resident OR gate weight >= tau */
    unsigned char *dg_keep = calloc((size_t)nu, 1);
    for (int j = 0; j < nu; j++)
        if (is_resident(uniq[j]) || dg_wsum[j] >= g_degrade_tau) dg_keep[j] = 1;

    /* 3. rescue: no position may end up with zero routed experts */
    /* build seen[] from current dg_keep */
    int *seen = calloc((size_t)256, sizeof(int));   /* expert ids < 256 in tests */
    for (int j = 0; j < nu; j++) if (dg_keep[j]) seen[uniq[j]] = 1;
    for (int s = 0; s < S; s++) {
        int alive = 0;
        for (int kk = 0; kk < keff[s] && !alive; kk++)
            if (seen[idxs[s * K + kk]]) alive = 1;
        if (alive || keff[s] <= 0) continue;
        /* reinstate highest-gate-weight miss */
        int be = -1; float bw = -1e30f;
        for (int kk = 0; kk < keff[s]; kk++) {
            float wv = ws[s * K + kk];
            if (wv > bw) { bw = wv; be = idxs[s * K + kk]; }
        }
        if (be < 0) be = idxs[s * K];
        seen[be] = 1;
        for (int j = 0; j < nu; j++)
            if (uniq[j] == be && !dg_keep[j]) { dg_keep[j] = 1; break; }
    }
    free(seen);

    /* 4. count dropped, then apply */
    int dg_dropped = 0;
    for (int j = 0; j < nu; j++) if (!dg_keep[j]) dg_dropped++;

    if (dg_dropped) {
        g_degrade_dropped += dg_dropped;

        /* rebuild seen[] from final dg_keep */
        int *seen2 = calloc((size_t)256, sizeof(int));
        for (int j = 0; j < nu; j++) if (dg_keep[j]) seen2[uniq[j]] = 1;

        for (int s = 0; s < S; s++) {
            int w = 0; float sold = 0, snew = 0;
            for (int kk = 0; kk < keff[s]; kk++) {
                int e = idxs[s * K + kk]; float wv = ws[s * K + kk];
                sold += wv;
                if (seen2[e]) { idxs[s * K + w] = e; ws[s * K + w] = wv; snew += wv; w++; }
            }
            if (w < keff[s]) {
                keff[s] = w;
                if (c->norm_topk && w > 0) {
                    float sm = 0;
                    for (int kk = 0; kk < w; kk++) sm += ws[s * K + kk];
                    sm += 1e-20f;
                    for (int kk = 0; kk < w; kk++) ws[s * K + kk] /= sm;
                    for (int kk = 0; kk < w; kk++) ws[s * K + kk] *= c->routed_scale;
                } else if (w > 0 && snew > 1e-20f && sold > snew) {
                    float sc = sold / snew;
                    for (int kk = 0; kk < w; kk++) ws[s * K + kk] *= sc;
                }
            }
        }

        /* compact uniq[] */
        int nu2 = 0;
        for (int j = 0; j < nu; j++) if (dg_keep[j]) uniq[nu2++] = uniq[j];
        nu = nu2;
        free(seen2);
    }

    free(dg_wsum);
    free(dg_keep);
    return nu;
}

/* ---- helpers -------------------------------------------------------------- */
static float sum_weights(const float *ws, int K, const int *keff, int S) {
    float s = 0;
    for (int i = 0; i < S; i++)
        for (int kk = 0; kk < keff[i]; kk++)
            s += ws[i * K + kk];
    return s;
}

static int expert_in_uniq(const int *uniq, int nu, int eid) {
    for (int j = 0; j < nu; j++) if (uniq[j] == eid) return 1;
    return 0;
}

static int expert_in_routing(const int *idxs, const int *keff, int S, int K, int eid) {
    for (int s = 0; s < S; s++)
        for (int kk = 0; kk < keff[s]; kk++)
            if (idxs[s * K + kk] == eid) return 1;
    return 0;
}

/* ---- tests ---------------------------------------------------------------- */

/* P1: g_degrade_zero=0 — block is a no-op */
static void test_off_by_default(void) {
    printf("\nP1: off-by-default\n");
    Cfg c = {1, 1.0f};
    /* S=1, K=2: experts 10 (w=0.8) and 11 (w=0.01, below tau) */
    int idxs[2] = {10, 11};
    float ws[2] = {0.8f, 0.01f};
    int keff[1] = {2};
    int uniq[2] = {10, 11}; int nu = 2;

    g_degrade_zero = 0;
    g_degrade_dropped = 0;
    nu = degrade_zero_apply(idxs, ws, keff, uniq, nu, 1, 2, &c);

    CHECK(nu == 2,           "nu unchanged when off");
    CHECK(keff[0] == 2,      "keff unchanged when off");
    CHECK(g_degrade_dropped == 0, "counter unchanged when off");
}

/* P2: resident experts are never dropped regardless of gate weight */
static void test_hits_are_safe(void) {
    printf("\nP2: hits-are-safe\n");
    Cfg c = {1, 1.0f};
    /* Expert 5 is resident with gate weight 0.001 (well below tau=0.03) */
    g_resident[0] = 5; g_nresident = 1;
    /* S=1, K=2: expert 5 (resident, w=0.001) and expert 7 (miss, w=0.8) */
    int idxs[2] = {5, 7};
    float ws[2] = {0.001f, 0.8f};
    int keff[1] = {2};
    int uniq[2] = {5, 7}; int nu = 2;

    g_degrade_zero = 1; g_degrade_tau = 0.03f; g_degrade_dropped = 0;
    nu = degrade_zero_apply(idxs, ws, keff, uniq, nu, 1, 2, &c);

    CHECK(expert_in_uniq(uniq, nu, 5), "resident expert 5 kept despite low gate weight");
    CHECK(expert_in_uniq(uniq, nu, 7), "above-tau miss expert 7 kept");
    CHECK(g_degrade_dropped == 0,      "nothing dropped");

    g_nresident = 0;
}

/* P3: cold misses below tau are dropped; at/above tau are kept */
static void test_tau_gate(void) {
    printf("\nP3: tau-gate\n");
    Cfg c = {1, 1.0f};
    /* S=1, K=3: expert 1 (w=0.7, above tau), expert 2 (w=0.03, exactly tau),
     *           expert 3 (w=0.02, below tau) — all cold misses */
    int idxs[3] = {1, 2, 3};
    float ws[3] = {0.7f, 0.03f, 0.02f};
    int keff[1] = {3};
    int uniq[3] = {1, 2, 3}; int nu = 3;

    g_degrade_zero = 1; g_degrade_tau = 0.03f; g_degrade_dropped = 0;
    nu = degrade_zero_apply(idxs, ws, keff, uniq, nu, 1, 3, &c);

    CHECK(expert_in_uniq(uniq, nu, 1),  "expert 1 (w=0.7) kept");
    CHECK(expert_in_uniq(uniq, nu, 2),  "expert 2 (w=0.03, exactly tau) kept");
    CHECK(!expert_in_uniq(uniq, nu, 3), "expert 3 (w=0.02, below tau) dropped");
    CHECK(g_degrade_dropped == 1,       "counter incremented by 1");
    CHECK(keff[0] == 2,                 "keff reduced to 2");
}

/* P4: renormalisation with norm_topk=1 (GLM-5.2 default) */
static void test_renorm_normtopk(void) {
    printf("\nP4: renorm with norm_topk=1\n");
    Cfg c = {1, 2.0f};   /* routed_scale=2.0 to verify it's applied */
    /* S=1, K=2: expert 10 (w=0.6, keep), expert 11 (w=0.02, drop) */
    int idxs[2] = {10, 11};
    float ws[2] = {0.6f, 0.02f};
    int keff[1] = {2};
    int uniq[2] = {10, 11}; int nu = 2;

    g_degrade_zero = 1; g_degrade_tau = 0.03f; g_degrade_dropped = 0;
    nu = degrade_zero_apply(idxs, ws, keff, uniq, nu, 1, 2, &c);

    CHECK(nu == 1,      "uniq compacted to 1");
    CHECK(keff[0] == 1, "keff=1 after drop");
    /* with norm_topk=1: ws[0] = (0.6/0.6) * routed_scale = 1.0 * 2.0 = 2.0 */
    CHECKF(fabsf(ws[0] - 2.0f) < 1e-5f,
           "renormed weight = routed_scale (got %.5f, want 2.0)", ws[0]);
}

/* P5: renormalisation with norm_topk=0 (preserve output magnitude) */
static void test_renorm_nonnorm(void) {
    printf("\nP5: renorm with norm_topk=0\n");
    Cfg c = {0, 1.0f};
    /* S=1, K=2: expert 20 (w=0.6, keep), expert 21 (w=0.02, drop) */
    int idxs[2] = {20, 21};
    float ws[2] = {0.6f, 0.02f};
    int keff[1] = {2};
    int uniq[2] = {20, 21}; int nu = 2;

    g_degrade_zero = 1; g_degrade_tau = 0.03f; g_degrade_dropped = 0;
    nu = degrade_zero_apply(idxs, ws, keff, uniq, nu, 1, 2, &c);

    /* with norm_topk=0: ws[0] = 0.6 * (0.62 / 0.6) = 0.62 (old_sum/new_sum scaling) */
    float expected = 0.6f * (0.62f / 0.6f);
    CHECKF(fabsf(ws[0] - expected) < 1e-5f,
           "rescaled weight preserves magnitude (got %.5f, want %.5f)", ws[0], expected);
}

/* P6: rescue rule — all experts are cold misses below tau */
static void test_rescue_rule(void) {
    printf("\nP6: rescue rule\n");
    Cfg c = {1, 1.0f};
    /* S=1, K=2: both experts are cold misses below tau.
     * Expert 30 has the higher gate weight — it must be rescued. */
    int idxs[2] = {30, 31};
    float ws[2] = {0.025f, 0.01f};
    int keff[1] = {2};
    int uniq[2] = {30, 31}; int nu = 2;

    g_degrade_zero = 1; g_degrade_tau = 0.03f; g_degrade_dropped = 0;
    nu = degrade_zero_apply(idxs, ws, keff, uniq, nu, 1, 2, &c);

    CHECK(keff[0] >= 1,                  "position not left with 0 routed experts");
    CHECK(expert_in_routing(idxs, keff, 1, 2, 30), "highest-weight expert 30 rescued");
    CHECK(!expert_in_routing(idxs, keff, 1, 2, 31), "lower-weight expert 31 dropped");
    /* only expert 31 counts as dropped; rescued expert 30 does not */
    CHECK(g_degrade_dropped == 1,        "only truly-dropped expert counted");
}

/* P7: prefill guard — S>4 means the block must be a complete no-op */
static void test_prefill_guard(void) {
    printf("\nP7: prefill guard (S=8)\n");
    Cfg c = {1, 1.0f};
    /* S=8, K=1: all experts are cold misses well below tau */
    int idxs[8]  = {0, 1, 2, 3, 4, 5, 6, 7};
    float ws[8]  = {0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
    int keff[8]  = {1, 1, 1, 1, 1, 1, 1, 1};
    int uniq[8]  = {0, 1, 2, 3, 4, 5, 6, 7}; int nu = 8;

    g_degrade_zero = 1; g_degrade_tau = 0.03f; g_degrade_dropped = 0;
    nu = degrade_zero_apply(idxs, ws, keff, uniq, nu, 8, 1, &c);

    CHECK(nu == 8,               "uniq unchanged for prefill batch");
    CHECK(g_degrade_dropped == 0, "counter unchanged for prefill batch");
}

/* P8: counter accumulates correctly across two calls */
static void test_counter_accumulates(void) {
    printf("\nP8: counter accumulates across calls\n");
    Cfg c = {1, 1.0f};
    g_degrade_zero = 1; g_degrade_tau = 0.03f; g_degrade_dropped = 0;

    /* call 1: drop 1 expert */
    {
        int idxs[2] = {40, 41}; float ws[2] = {0.8f, 0.01f};
        int keff[1] = {2}; int uniq[2] = {40, 41}; int nu = 2;
        degrade_zero_apply(idxs, ws, keff, uniq, nu, 1, 2, &c);
    }
    CHECK(g_degrade_dropped == 1, "counter=1 after first call");

    /* call 2: drop 2 experts */
    {
        int idxs[3] = {50, 51, 52}; float ws[3] = {0.8f, 0.01f, 0.01f};
        int keff[1] = {3}; int uniq[3] = {50, 51, 52}; int nu = 3;
        degrade_zero_apply(idxs, ws, keff, uniq, nu, 1, 3, &c);
    }
    CHECK(g_degrade_dropped == 3, "counter=3 after second call (cumulative)");
}

/* ---- main ----------------------------------------------------------------- */
int main(void) {
    printf("test_degrade_zero: DEGRADE_ZERO miss-slot zero-fill logic\n");
    test_off_by_default();
    test_hits_are_safe();
    test_tau_gate();
    test_renorm_normtopk();
    test_renorm_nonnorm();
    test_rescue_rule();
    test_prefill_guard();
    test_counter_accumulates();
    printf("\n");
    if (g_fails) {
        printf("test_degrade_zero: %d FAILED\n", g_fails);
        return 1;
    }
    printf("test_degrade_zero: all tests passed\n");
    return 0;
}
