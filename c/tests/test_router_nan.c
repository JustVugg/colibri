/* Regression for non-finite-logit poisoning of the ROUTER (upstream of sampling).
 *
 * test_logit_nan.c hardened the sampling side. The routing side was not, and it
 * runs first: the three top-K selection loops in moe() start from best=-1 and use
 * it as an index as soon as the scan ends. choice[e] = sigmoidf(logit[e]) + bias,
 * and sigmoidf(NaN) is NaN, so `choice[e] > bv` is false for every e — best stays
 * -1. The old code then did idx[kk]=best; w[kk]=logit[best], which is:
 *
 *   - a heap OOB read of logits_all[s*E - 1] at s==0;
 *   - three OOB writes at index -1 (eusage / eheat / elast);
 *   - a stack buffer underflow on the VLA `unsigned char seen[E]` in FASE B;
 *   - uniq[nu++] = -1 handed to expert_load().
 *
 * Fix under test: router_best_or_fallback() maps a failed scan to a deterministic
 * in-range expert and warns once. Degrade + diagnose, never index out of range.
 *
 * No model file needed: exercises the guard and the exact scan that trips it. */
#include <assert.h>
#include <math.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

/* the selection scan as written in moe(), verbatim in shape */
static int scan_best(const float *choice, int E, const int *taken, int ntaken) {
    int best = -1;
    float bv = -1e30f;
    for (int e = 0; e < E; e++) {
        int tk = 0;
        for (int j = 0; j < ntaken; j++)
            if (taken[j] == e) {
                tk = 1;
                break;
            }
        if (!tk && choice[e] > bv) {
            bv = choice[e];
            best = e;
        }
    }
    return best;
}

int main(void) {
    const int E = 8;

    /* --- the trigger is real: an all-NaN choice[] leaves best==-1 --- */
    {
        float choice[8];
        for (int e = 0; e < E; e++) choice[e] = sigmoidf(NAN) + 0.0f;
        for (int e = 0; e < E; e++) assert(choice[e] != choice[e] && "sigmoidf(NaN) is NaN");
        assert(scan_best(choice, E, NULL, 0) == -1 && "all-NaN scan selects nothing");
    }

    /* one NaN logit is enough to poison a slot once the finite ones are taken */
    {
        float choice[8] = {NAN, NAN, NAN, NAN, NAN, NAN, NAN, 2.f};
        int taken[1] = {7};
        assert(scan_best(choice, E, taken, 1) == -1 && "remaining candidates all NaN");
    }

    /* --- the guard: passthrough when the scan succeeded --- */
    assert(router_best_or_fallback(5, 0, E, 0) == 5);
    assert(router_best_or_fallback(0, 3, E, 0) == 0 && "expert 0 is a valid pick, not a failure");

    /* --- the guard: a failed scan degrades to a deterministic in-range expert --- */
    for (int kk = 0; kk < E; kk++) {
        int b = router_best_or_fallback(-1, kk, E, 0);
        assert(b >= 0 && b < E && "fallback must be a valid expert index");
        assert(b == kk && "distinct kk -> distinct expert, so a slot is never duplicated");
    }
    /* Ksel may exceed E on a malformed config: still in range, never -1 */
    for (int kk = E; kk < E + 4; kk++) {
        int b = router_best_or_fallback(-1, kk, E, 0);
        assert(b >= 0 && b < E && "fallback stays in range past E");
    }

    /* --- end to end: the accounting arrays are indexed safely after the guard --- */
    {
        float choice[8];
        for (int e = 0; e < E; e++) choice[e] = NAN;
        unsigned char seen[8];
        memset(seen, 0, sizeof seen);
        int idx[4];
        int Ksel = 4;
        for (int kk = 0; kk < Ksel; kk++) {
            int best = scan_best(choice, E, idx, kk);
            best = router_best_or_fallback(best, kk, E, 0);
            idx[kk] = best;
        }
        for (int kk = 0; kk < Ksel; kk++) {
            assert(idx[kk] >= 0 && idx[kk] < E && "no OOB index reaches eusage/eheat/elast");
            seen[idx[kk]] = 1; /* would have been seen[-1] before the fix */
        }
    }

    printf("OK test_router_nan: top-K selection guards best==-1\n");
    return 0;
}
