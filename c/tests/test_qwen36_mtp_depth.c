/* The MTP head's depth for the next verify (qwen36.c q36_mtp_depth). A fixed depth of 2 or
 * more that the drafts' probabilities trim (Q36_MTP_PSTEP, on by default) takes a plain
 * step every SPEC_REFRESH forwards: the trim prices verify rows with T(1) and rho, and at a
 * fixed depth no other step measures T(1). One draft (the default), an untrimmed depth, the
 * gate off and forced drafts never take it: no trim prices the head's rows there. */
#include <stdio.h>
#include <string.h>

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int fails;

static void expect(int depth, int pstep, int off, int force, int want) {
    Q36Spec sp; memset(&sp, 0, sizeof sp);
    spec_gate_init(&sp.gate, off);
    sp.mtp_depth = depth; sp.mtp_pstep = pstep; sp.mtp_force = force;
    for (int i = 0; i < SPEC_REFRESH; i++) spec_gate_forward(&sp.gate, depth + 1, 0.01);
    int k = q36_mtp_depth(&sp, Q36_SPEC_ROWS, NULL);
    if (k != want) {
        printf("FAIL: depth %d, pstep %d, gate off %d, force %d: %d drafts after %d verifies (want %d)\n",
               depth, pstep, off, force, k, SPEC_REFRESH, want);
        fails++;
    }
    spec_gate_forward(&sp.gate, 1, 0.01);   /* the plain step: the fixed depth again */
    if (q36_mtp_depth(&sp, Q36_SPEC_ROWS, NULL) != depth) {
        printf("FAIL: depth %d, pstep %d: not the fixed depth after a plain step\n", depth, pstep); fails++;
    }
}

/* Q36_SPEC_DECLINE: a fixed depth that loses to a plain step is not drafted, but for a probe on a
 * generation's first declined step and every SPEC_PROBE declined steps after it; drafts that are
 * accepted on the probes bring drafting back. Without T(1), with the gate off or the switch off:
 * the fixed depth as before. */
static void decline(void) {
    Q36Spec sp; memset(&sp, 0, sizeof sp);
    spec_gate_init(&sp.gate, 0);
    sp.mtp_depth = 7; sp.mtp_pstep = 1; sp.decline = 1;
#define WANT(c, ...) do { if (!(c)) { printf("FAIL: decline: " __VA_ARGS__); printf("\n"); fails++; } } while (0)
    WANT(q36_mtp_depth(&sp, Q36_SPEC_ROWS, NULL) == 7, "T(1) not measured: the fixed depth");
    /* the 93k-position run: a 51 ms step, a 181 ms block, rows nearly free, a third of the first drafts stand */
    spec_gate_forward(&sp.gate, 1, 0.051);
    for (int i = 0; i < 8; i++) { spec_gate_pre_cost(&sp.gate, SPEC_SRC_MTP, 0.181); spec_gate_forward(&sp.gate, 2, 0.056); }
    for (int i = 0; i < 24; i++) spec_gate_result(&sp.gate, SPEC_SRC_MTP, 1, i % 3 == 0);
    int drafted = 0, n = 4 * SPEC_PROBE;
    for (int i = 0; i < n; i++) {
        double v = 0; int k = q36_mtp_depth(&sp, Q36_SPEC_ROWS, &v);
        drafted += k > 0;
        WANT(k == 0 ? v == 1.0 : k == 7, "a pick of %d", k);
        if (i == 0) WANT(k == 7, "a generation's first declined step is the probe");
    }
    WANT(drafted == 4, "a losing block: %d of %d steps drafted (want one probe every %d)", drafted, n, SPEC_PROBE);
    WANT(sp.gate.declined[SPEC_SRC_MTP] == (unsigned)(n - 4) && sp.gate.probes[SPEC_SRC_MTP] == 4, "the declines and probes counted");
    { Q36Spec off = sp; off.decline = 0; WANT(q36_mtp_depth(&off, Q36_SPEC_ROWS, NULL) == 7, "the switch off: the fixed depth"); }
    { Q36Spec go = sp; go.gate.off = 1; WANT(q36_mtp_depth(&go, Q36_SPEC_ROWS, NULL) == 7, "the gate off: the fixed depth"); }
    { Q36Spec f = sp; f.mtp_force = 'a'; WANT(q36_mtp_depth(&f, Q36_SPEC_ROWS, NULL) == 7, "forced drafts: the fixed depth"); }
    /* a kept losing gate carried into a new generation: first step plain to re-measure T(1),
     * then the first declined step probes immediately so short generations still probe */
    {
        Q36Spec next; memset(&next, 0, sizeof next);
        spec_gate_init(&next.gate, 0);
        spec_gate_carry(&next.gate, &sp.gate);
        next.mtp_depth = 7; next.mtp_pstep = 1; next.decline = 1;
        WANT(q36_mtp_depth(&next, Q36_SPEC_ROWS, NULL) == 0, "carried gate: first step plain to measure T(1)");
        spec_gate_forward(&next.gate, 1, 0.051);
        WANT(q36_mtp_depth(&next, Q36_SPEC_ROWS, NULL) == 7, "carried losing gate: first declined step probes");
        WANT(q36_mtp_depth(&next, Q36_SPEC_ROWS, NULL) == 0, "carried losing gate: second step declines");
    }
    /* the context changes (a cheap block, drafts that stand): the probes see it and drafting is back */
    int steps = 0, run = 0;
    while (steps < 4000 && run < 3 * SPEC_PROBE) {
        steps++;
        if (q36_mtp_depth(&sp, Q36_SPEC_ROWS, NULL)) {
            run++;
            spec_gate_pre_cost(&sp.gate, SPEC_SRC_MTP, 0.004); spec_gate_forward(&sp.gate, 8, 0.07);
            spec_gate_result(&sp.gate, SPEC_SRC_MTP, 7, 7);
        } else { run = 0; spec_gate_forward(&sp.gate, 1, 0.051); }
    }
    WANT(run == 3 * SPEC_PROBE, "drafting never came back in %d steps", steps);
    WANT(steps < 40 * SPEC_PROBE, "drafting came back only after %d steps", steps);
    /* a winning drafter is never declined */
    spec_gate_init(&sp.gate, 0); sp.decline_n = 0;
    spec_gate_forward(&sp.gate, 1, 0.019); spec_gate_pre_cost(&sp.gate, SPEC_SRC_MTP, 0.004); spec_gate_forward(&sp.gate, 8, 0.03);
    for (int i = 0; i < 8; i++) spec_gate_result(&sp.gate, SPEC_SRC_MTP, 7, 5);
    for (int i = 0; i < 2 * SPEC_PROBE; i++) WANT(q36_mtp_depth(&sp, Q36_SPEC_ROWS, NULL) == 7, "a winning drafter declined");
#undef WANT
}

int main(void) {
    decline();
    expect(1, 1, 0, 0, 1);   /* one draft: never a plain step */
    expect(2, 1, 0, 0, 0);   /* two trimmed drafts: the plain step */
    expect(3, 1, 0, 0, 0);
    expect(2, 0, 0, 0, 2);   /* Q36_MTP_PSTEP=0: the depth as asked */
    expect(2, 1, 1, 0, 2);   /* COLI_SPEC_GATE=0 */
    expect(2, 1, 0, 'a', 2); /* Q36_MTP_FORCE */
    if (!fails) printf("OK test_qwen36_mtp_depth: a plain step every %d forwards only for a trimmed depth of 2+\n",
                       SPEC_REFRESH);
    return fails ? 1 : 0;
}
