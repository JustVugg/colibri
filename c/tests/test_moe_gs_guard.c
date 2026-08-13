/* Regression for moe()'s MB_BUILD subset-consistency guard (review finding F2).
 *
 * The trap: coli_metal_moe_block(_begin) takes ONE fmt/qgs for an entire GPU batch of
 * experts (+ optionally the fused shared expert). MB_BUILD already refused to fold the
 * shared expert into a batch whose fmt disagreed (`l->sh_gate.fmt==shf`), but for fmt=4
 * (grouped int4) format equality alone isn't enough: two fmt=4 tensors can still disagree
 * on GROUP SIZE, and moe_submit has no per-member gs -- every expert in the batch is
 * dequantized with the SAME qgs, so a silently-included gs mismatch would corrupt every
 * row belonging to the odd one out. mb_gs_compat() is the guard: it decides whether a
 * candidate (fmt,gs) is safe to fold into a batch already established at (est_fmt,est_gs).
 *
 * This test exercises mb_gs_compat() directly rather than moe() end-to-end: MB_BUILD is a
 * macro #define'd (and #undef'd) entirely inside moe()'s own function body, so it isn't
 * reachable from outside moe() even via the "#include colibri.c" seam this file uses --
 * and moe() itself needs a fully-populated Model/Layer/ESlot/ecache/router-output
 * apparatus no existing test constructs, which would make a true end-to-end harness
 * heavy and fragile relative to what it proves. mb_gs_compat() is not a re-implementation
 * of MB_BUILD's guard logic -- it IS that logic (MB_BUILD calls it directly, at both the
 * routed-expert and the shared-expert call sites) -- so exercising it directly is a
 * faithful, low-risk proxy for the integration behavior. See WORKER_REPORT_integ_postwave.md
 * UNCERTAINTIES for the case against the heavier alternative.
 *
 * Also documents, by assertion (not just prose), the two gaps this guard deliberately
 * does NOT cover: per-expert fmt heterogeneity (g vs g of a different expert), and
 * per-tensor-role gs heterogeneity within one expert/shared-expert triple (g vs u vs d).
 * Both are pre-existing, out of scope for F2 -- see the guard's own comment in colibri.c. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <assert.h>

int main(void){
    /* --- nothing established yet (est_fmt<0): always compatible, any (fmt,gs) --- */
    assert(mb_gs_compat(-1,0, 4,128)==1 && "first member of a subset is always accepted");
    assert(mb_gs_compat(-1,0, 2,0)==1   && "first member, non-grouped fmt");

    /* --- established fmt=4: SAME gs is compatible (the common, correct case) --- */
    assert(mb_gs_compat(4,128, 4,128)==1 && "matching gs within a fmt=4 subset: ok");
    assert(mb_gs_compat(4,64,  4,64)==1  && "matching gs (different value): ok");

    /* --- established fmt=4: DIFFERENT gs is the F2 trap -- must be rejected --- */
    assert(mb_gs_compat(4,128, 4,64)==0  && "gs mismatch within a fmt=4 subset: must reject");
    assert(mb_gs_compat(4,64,  4,128)==0 && "gs mismatch, reversed order: must reject");
    assert(mb_gs_compat(4,128, 4,127)==0 && "off-by-one gs mismatch: must reject (no slack)");

    /* --- established fmt!=4 (no group size to disagree on): always compatible,
     *     REGARDLESS of what the candidate claims -- fmt heterogeneity itself is a
     *     separate, pre-existing gap this guard does not cover (documented, not fixed:
     *     MB_BUILD has never checked per-expert fmt agreement, fmt=4 or otherwise). --- */
    assert(mb_gs_compat(2,0, 4,999)==1  && "est fmt=2 (ungrouped): gs never checked");
    assert(mb_gs_compat(1,0, 4,999)==1  && "est fmt=1 (int8): gs never checked");

    /* --- candidate fmt!=4 against an established fmt=4 batch: also out of scope
     *     (the candidate's OWN fmt mismatch is what would corrupt it, not gs -- and
     *     fmt agreement for the shared expert is still checked separately by MB_BUILD's
     *     TRY_SH condition before mb_gs_compat is ever consulted). --- */
    assert(mb_gs_compat(4,128, 2,0)==1  && "candidate fmt=2 against fmt=4 batch: gs moot");

    printf("OK test_moe_gs_guard: mb_gs_compat (review F2) accepts matching gs, "
           "rejects mismatched gs, documents its fmt-heterogeneity non-scope\n");
    return 0;
}
