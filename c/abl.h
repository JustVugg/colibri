#ifndef ABL_H
#define ABL_H
/* ============================================================================
 * abl.h — per-item EXPERT CAUSAL-ABLATION config for colibri's moe().
 *
 * A research instrument, off the hot path: OFF by default, byte-identical to
 * upstream unless a run explicitly enables it via ABLATE_SCORE=<manifest>.
 *
 * PURPOSE.  Let a named set of (layer, expert) cells be ablated for ONE
 * teacher-forced item, in one of three modes, so the causal effect of an expert
 * on the FINAL logits can be measured against the same item's baseline. This is
 * the causal complement to the routing-affinity Expert Atlas (#175): the Atlas
 * shows what routes WHERE; ablation shows what an expert actually DOES.
 *
 *   mode 0  OFF             — no ablation; the forward is byte-identical to the
 *                             un-ablated build (every abl_*() below is inert).
 *   mode 1  CONTRIBUTION    — routing is UNCHANGED (the expert is still selected,
 *                             counted, and traced); only its weighted output
 *                             contribution to the MoE block is replaced with 0.
 *                             Isolates "what does this expert's output add,
 *                             holding the gate fixed". (CPU forward path.)
 *   mode 2  ROUTE-AROUND    — the cell is dropped from the token's selected set
 *                             BEFORE the routed weights are renormalised, so the
 *                             remaining kept experts absorb the mass (norm_topk
 *                             over the survivors).  Isolates "can the model cope
 *                             without this expert by reweighting the others".
 *                             This is the arm that mirrors EXPERT_BUDGET-style
 *                             dropping, per token, on a chosen expert.
 *   mode 3  MODULE-SWAP     — the cell's routing slot is remapped to a substitute
 *                             expert (swap_to) keeping the original routed weight,
 *                             i.e. a different learned module stands in.  A cheap
 *                             deterministic-replaceability probe.
 *
 * The three moe() hooks (route-around compaction + module-swap remap in FASE A;
 * contribution skip in the FASE C CPU accumulate) are ALL guarded by g_abl.mode,
 * so when mode==0 none fires and output stays identical to the un-ablated build.
 * Modes 2/3 edit the SELECTED set before the eusage/eheat/elast counters and the
 * norm_topk/routed_scale/ROUTE_TRACE emit, so all downstream accounting reflects
 * the post-ablation routing.
 *
 * PER-ITEM RESET (the anti-leak invariant).  The ABLATE_SCORE driver calls
 * abl_reset() then abl_set_item() for each item, so an item's ablation spec can
 * never leak into the next item.  This header holds no cross-item state and has
 * no Model/Cfg dependency, so the decision logic unit-tests against the SAME
 * source the engine links (tests/test_ablate.c).
 * ==========================================================================*/

#define ABL_MAX_CELLS 16 /* per item: 1 for single-expert tests, a few for cohort screens */

typedef struct Abl {
    int mode;                   /* 0 off, 1 contribution, 2 route-around, 3 module-swap */
    int n;                      /* # ablated cells for THIS item                         */
    int layer[ABL_MAX_CELLS];   /* absolute layer index                                  */
    int expert[ABL_MAX_CELLS];  /* expert id within the layer                            */
    int swap_to[ABL_MAX_CELLS]; /* mode-3 substitute expert id; -1 for modes 0/1/2       */
} Abl;

/* PER-ITEM RESET: clear to the OFF state.  After this every abl_*() is inert. */
static inline void abl_reset(Abl *a) {
    a->mode = 0;
    a->n = 0;
    for (int i = 0; i < ABL_MAX_CELLS; i++) {
        a->layer[i] = -1;
        a->expert[i] = -1;
        a->swap_to[i] = -1;
    }
}

/* Configure the ablation for the current item.  n is clamped to ABL_MAX_CELLS
 * (fail-safe: never overrun).  T may be NULL for modes 0/1/2 (no swap targets).
 * Returns the number of cells actually installed. */
static inline int abl_set_item(Abl *a, int mode, const int *L, const int *E, const int *T, int n) {
    if (n < 0) n = 0;
    if (n > ABL_MAX_CELLS) n = ABL_MAX_CELLS;
    a->mode = mode;
    a->n = n;
    for (int i = 0; i < n; i++) {
        a->layer[i] = L[i];
        a->expert[i] = E[i];
        a->swap_to[i] = (T ? T[i] : -1);
    }
    return n;
}

/* mode 2: is (layer,e) to be dropped from this token's selected set? */
static inline int abl_route_around(const Abl *a, int layer, int e) {
    if (a->mode != 2) return 0;
    for (int i = 0; i < a->n; i++)
        if (a->layer[i] == layer && a->expert[i] == e) return 1;
    return 0;
}

/* mode 1: is (layer,e)'s weighted contribution to be zeroed (routing kept)? */
static inline int abl_zero_contrib(const Abl *a, int layer, int e) {
    if (a->mode != 1) return 0;
    for (int i = 0; i < a->n; i++)
        if (a->layer[i] == layer && a->expert[i] == e) return 1;
    return 0;
}

/* mode 3: substitute expert id for (layer,e), or -1 if this cell is not swapped. */
static inline int abl_swap_target(const Abl *a, int layer, int e) {
    if (a->mode != 3) return -1;
    for (int i = 0; i < a->n; i++)
        if (a->layer[i] == layer && a->expert[i] == e) return a->swap_to[i];
    return -1;
}

#endif /* ABL_H */
