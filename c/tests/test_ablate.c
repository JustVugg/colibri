/* ============================================================================
 * test_ablate.c — standalone unit tests for the expert causal-ablation logic
 * (abl.h).  No model, no weights: it links the SAME abl.h the engine links and
 * drives a faithful MINIATURE of colibri's moe() routing+accumulate (FASE A
 * selection -> route-around compaction -> module-swap remap -> norm_topk ->
 * routed_scale ; FASE C weighted accumulate with the contribution skip) with
 * synthetic router state, applying the EXACT hook arithmetic the ablation hooks
 * insert into moe().  This is the on-box proof that the ablation decision logic
 * and per-item reset are correct without needing to load a 744B model.
 *
 * Required properties:
 *   P1 BASELINE == UNPATCHED — mode 0 leaves the selected set, weights, and the
 *      accumulated MoE output byte-identical to the un-hooked reference.
 *   P2 ABLATING AN UNUSED EXPERT IS A NO-OP — configuring any mode on a (layer,
 *      expert) cell that this token does not select changes nothing.
 *   P3 PER-ITEM RESET — abl_reset() returns to OFF; an item's spec never leaks
 *      into the next item (re-running after reset reproduces the baseline).
 * plus the SEMANTICS each arm must have:
 *   S1 CONTRIBUTION (mode 1) — routing (idx,w) identical to baseline; the ablated
 *      expert's weighted output is removed from the sum.
 *   S2 ROUTE-AROUND (mode 2) — the ablated expert is dropped from the selected
 *      set (Ke-1), survivors renormalised (sum 1 * routed_scale).
 *   S3 MODULE-SWAP (mode 3) — the ablated slot's expert id is remapped to the
 *      substitute, the routed weight preserved.
 * Exit 0 = all pass.
 * ==========================================================================*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../abl.h"

static int fails = 0;
#define CHECK(cond, msg)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf("  FAIL: %s\n", msg);                                                                               \
            fails++;                                                                                                   \
        } else printf("  ok:   %s\n", msg);                                                                            \
    } while (0)

enum { E = 32, KE = 8 };
static const float ROUTED_SCALE = 2.5f; /* stand-in for c->routed_scale */

/* Deterministic distinct "expert output" scalar for expert e (stands in for the
 * D-vector each expert contributes; a scalar suffices to test the accumulate). */
static float expert_out(int layer, int e) { return (float)(1 + (layer * 131 + e * 7) % 97); }

/* FASE A selection: pick KE distinct experts for (item,layer), strictly
 * decreasing gate; w[] initialised to the (pre-norm) gate, as in colibri. */
static void select_topk(long item, int layer, int *idx, float *w) {
    int step = 3, base = (int)((item * 7 + layer * 11) % E);
    for (int kk = 0; kk < KE; kk++) {
        int e = (base + kk * step) % E;
        idx[kk] = e;
        w[kk] = 0.9f - 0.05f * (float)kk; /* pre-norm sigmoid gate (colibri: w[kk]=logit[best]) */
    }
}

/* Mirror of the moe() FASE-A tail AFTER selection, WITH the ablation hooks the
 * engine inserts (route-around compaction, then module-swap remap, then norm_topk,
 * then routed_scale).  Writes the final idx/w and returns Ke. */
static int finalize_routing(const Abl *a, int layer, int *idx, float *w, int Ke) {
    /* --- ABLATE route-around (mode 2): drop ablated cells from the set --- */
    if (a->mode == 2) {
        int wr = 0;
        for (int kk = 0; kk < Ke; kk++) {
            if (abl_route_around(a, layer, idx[kk])) continue; /* skip ablated */
            idx[wr] = idx[kk];
            w[wr] = w[kk];
            wr++;
        }
        Ke = wr;
    }
    /* --- ABLATE module-swap (mode 3): remap the slot's expert, keep weight --- */
    if (a->mode == 3) {
        for (int kk = 0; kk < Ke; kk++) {
            int to = abl_swap_target(a, layer, idx[kk]);
            if (to >= 0) idx[kk] = to;
        }
    }
    /* --- norm_topk over the (possibly reduced) set, then routed_scale --- */
    float sm = 0.f;
    for (int kk = 0; kk < Ke; kk++) sm += w[kk];
    sm += 1e-20f;
    for (int kk = 0; kk < Ke; kk++) w[kk] /= sm;
    for (int kk = 0; kk < Ke; kk++) w[kk] *= ROUTED_SCALE;
    return Ke;
}

/* Mirror of the moe() FASE-C accumulate WITH the contribution skip (mode 1). */
static float accumulate(const Abl *a, int layer, const int *idx, const float *w, int Ke) {
    float out = 0.f;
    for (int kk = 0; kk < Ke; kk++) {
        int e = idx[kk];
        if (a->mode == 1 && abl_zero_contrib(a, layer, e)) continue; /* contribution -> 0 */
        out += w[kk] * expert_out(layer, e);
    }
    return out;
}

/* Full mini-moe for one token: select -> finalize(hooks) -> accumulate. */
static float mini_moe(const Abl *a, long item, int layer, int *idx_out, float *w_out, int *Ke_out) {
    int idx[KE];
    float w[KE];
    select_topk(item, layer, idx, w);
    int Ke = finalize_routing(a, layer, idx, w, KE);
    float o = accumulate(a, layer, idx, w, Ke);
    if (idx_out) memcpy(idx_out, idx, Ke * sizeof(int));
    if (w_out) memcpy(w_out, w, Ke * sizeof(float));
    if (Ke_out) *Ke_out = Ke;
    return o;
}

int main(void) {
    const long IT = 12345;
    const int L = 41;
    Abl off;
    abl_reset(&off);

    /* ----- baseline reference (mode 0) ----- */
    int b_idx[KE];
    float b_w[KE];
    int b_Ke;
    float b_out = mini_moe(&off, IT, L, b_idx, b_w, &b_Ke);
    printf("baseline: Ke=%d out=%.6f  sel=[", b_Ke, b_out);
    for (int k = 0; k < b_Ke; k++) {
        printf("%d ", b_idx[k]);
    }
    printf("]\n");

    /* ===== P1  baseline == unpatched (mode 0 identical to un-hooked ref) ===== */
    /* un-hooked reference: run the selection+norm+accumulate with NO Abl at all. */
    {
        int idx[KE];
        float w[KE];
        select_topk(IT, L, idx, w);
        float sm = 0;
        for (int k = 0; k < KE; k++) sm += w[k];
        sm += 1e-20f;
        for (int k = 0; k < KE; k++) {
            w[k] /= sm;
            w[k] *= ROUTED_SCALE;
        }
        float ref = 0;
        for (int k = 0; k < KE; k++) ref += w[k] * expert_out(L, idx[k]);
        int same_sel = 1;
        for (int k = 0; k < KE; k++)
            if (idx[k] != b_idx[k]) same_sel = 0;
        CHECK(b_Ke == KE && same_sel, "P1 mode0 keeps the full selected set");
        CHECK(fabsf(ref - b_out) < 1e-6f, "P1 mode0 output == un-hooked reference (byte-identical path)");
    }

    /* ===== P2  ablating an UNUSED expert is a no-op (all three modes) ===== */
    /* pick a cell NOT selected by this token (and, for route/swap, a distinct id) */
    int used[E];
    memset(used, 0, sizeof(used));
    for (int k = 0; k < b_Ke; k++) used[b_idx[k]] = 1;
    int unused = -1;
    for (int e = 0; e < E; e++)
        if (!used[e]) {
            unused = e;
            break;
        }
    int sub = -1;
    for (int e = 0; e < E; e++)
        if (!used[e] && e != unused) {
            sub = e;
            break;
        }
    CHECK(unused >= 0 && sub >= 0, "P2 setup: found an unused expert + a substitute id");
    for (int mode = 1; mode <= 3; mode++) {
        Abl a;
        abl_reset(&a);
        int Ls[1] = {L}, Es[1] = {unused}, Ts[1] = {sub};
        abl_set_item(&a, mode, Ls, Es, (mode == 3 ? Ts : NULL), 1);
        int idx[KE];
        float w[KE];
        int Ke;
        float o = mini_moe(&a, IT, L, idx, w, &Ke);
        int same = (Ke == b_Ke);
        for (int k = 0; k < Ke && same; k++)
            if (idx[k] != b_idx[k] || fabsf(w[k] - b_w[k]) > 1e-6f) same = 0;
        char msg[80];
        snprintf(msg, sizeof(msg), "P2 mode %d on an unused cell -> identical routing+output", mode);
        CHECK(same && fabsf(o - b_out) < 1e-6f, msg);
        /* also on a DIFFERENT layer than the one ablated -> abl functions never match */
        CHECK(!abl_zero_contrib(&a, L + 1, unused) && !abl_route_around(&a, L + 1, unused) &&
                  abl_swap_target(&a, L + 1, unused) < 0,
              "P2 (layer key is respected)");
    }

    /* ===== S1  CONTRIBUTION (mode 1): routing identical, target expert removed ===== */
    {
        int tgt = b_idx[2]; /* an expert this token DOES select */
        Abl a;
        abl_reset(&a);
        int Ls[1] = {L}, Es[1] = {tgt};
        abl_set_item(&a, 1, Ls, Es, NULL, 1);
        int idx[KE];
        float w[KE];
        int Ke;
        float o = mini_moe(&a, IT, L, idx, w, &Ke);
        int route_same = (Ke == b_Ke);
        for (int k = 0; k < Ke && route_same; k++)
            if (idx[k] != b_idx[k] || fabsf(w[k] - b_w[k]) > 1e-6f) route_same = 0;
        CHECK(route_same, "S1 contribution leaves the routing (idx,w) UNCHANGED");
        float expect = b_out - b_w[2] * expert_out(L, tgt); /* just that term removed */
        CHECK(fabsf(o - expect) < 1e-3f, "S1 contribution removes exactly the ablated expert's weighted output");
    }

    /* ===== S2  ROUTE-AROUND (mode 2): expert dropped, survivors renormalised ===== */
    {
        int tgt = b_idx[2];
        Abl a;
        abl_reset(&a);
        int Ls[1] = {L}, Es[1] = {tgt};
        abl_set_item(&a, 2, Ls, Es, NULL, 1);
        int idx[KE];
        float w[KE];
        int Ke;
        float o = mini_moe(&a, IT, L, idx, w, &Ke);
        (void)o;
        int present = 0;
        for (int k = 0; k < Ke; k++)
            if (idx[k] == tgt) present = 1;
        CHECK(Ke == b_Ke - 1 && !present, "S2 route-around drops the ablated expert (Ke-1, absent)");
        float sm = 0;
        for (int k = 0; k < Ke; k++) sm += w[k];
        CHECK(fabsf(sm - ROUTED_SCALE) < 1e-4f, "S2 survivors renormalise to sum==routed_scale");
    }

    /* ===== S3  MODULE-SWAP (mode 3): slot remapped to substitute, weight kept ===== */
    {
        int tgt = b_idx[2];
        int to = sub;
        Abl a;
        abl_reset(&a);
        int Ls[1] = {L}, Es[1] = {tgt}, Ts[1] = {to};
        abl_set_item(&a, 3, Ls, Es, Ts, 1);
        int idx[KE];
        float w[KE];
        int Ke;
        float o = mini_moe(&a, IT, L, idx, w, &Ke);
        int has_to = 0, has_tgt = 0;
        for (int k = 0; k < Ke; k++) {
            if (idx[k] == to) has_to = 1;
            if (idx[k] == tgt) has_tgt = 1;
        }
        CHECK(Ke == b_Ke && has_to && !has_tgt, "S3 module-swap remaps the slot id (target in, original out)");
        float expect = b_out - b_w[2] * expert_out(L, tgt) + b_w[2] * expert_out(L, to);
        CHECK(fabsf(o - expect) < 1e-3f, "S3 module-swap substitutes the output at the same routed weight");
    }

    /* ===== P3  PER-ITEM RESET: a spec must not leak into the next item ===== */
    {
        Abl a;
        abl_reset(&a);
        int Ls[1] = {L}, Es[1] = {b_idx[2]};
        abl_set_item(&a, 1, Ls, Es, NULL, 1);
        float o1 = mini_moe(&a, IT, L, NULL, NULL, NULL);
        CHECK(fabsf(o1 - b_out) > 1e-6f, "P3 setup: the ablated item differs from baseline");
        abl_reset(&a); /* <-- per-item reset */
        CHECK(a.mode == 0 && a.n == 0, "P3 abl_reset restores the OFF state");
        float o2 = mini_moe(&a, IT, L, NULL, NULL, NULL);
        CHECK(fabsf(o2 - b_out) < 1e-6f, "P3 after reset the next item reproduces the baseline (no leak)");
    }

    printf("\n%s  (%d failure%s)\n", fails ? "TEST FAIL" : "ALL PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
