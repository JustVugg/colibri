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

int main(void) {
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
