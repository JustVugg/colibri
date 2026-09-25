/* qwen36_cap_for_ram(): the cap<=0 ("auto") sentinel derives the expert-cache
 * slots/layer from host RAM instead of the old hardcoded default of 16
 * (bare CLI omission) / 16 (Segment API's memory_limit_bytes==0 fallback).
 * Pure function -- no Model*, no globals, no I/O -- same testability
 * contract as k3_cap_for_ram/coli_resolve_cap (test_k3_ram_budget.c,
 * test_cap_precedence.c). An explicit cap>0 never reaches this function at
 * all (main()/qwen36_segment_engine_open() only call it when cap<=0), so
 * "explicit cap always wins" is verified at the integration level
 * (test_qwen36_cap_budget.py), not here. Likewise, RAM_GB parsing itself
 * (getenv/atof) happens in model_init_range, not this function -- an
 * unparseable or negative RAM_GB reaches this function as ram_gb_override<=0
 * (atof's own behavior on garbage input), which is exactly what the
 * "override<=0 falls back to the computed budget" case below exercises. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

/* Roughly qwen36-i4-gs64 shape: hidden=2048 inter=512 n_experts=256. */
enum { HIDDEN = 2048, INTER = 512, N_EXPERTS = 256 };

int main(void) {
    /* RAM_GB override wins over the resident+avail*0.88 computed budget:
     * same resident/avail, only the override differs -> different cap. The
     * outparams (slot_gb, budget_gb) are threaded through rather than
     * recomputed by any caller, so budget_gb must equal the override
     * exactly when one is given. */
    {
        double slot_gb_a = 0, slot_gb_b = 0, budget_a = 0, budget_b = 0;
        int cap_computed = qwen36_cap_for_ram(4.83, 20.0, 0.0, HIDDEN, INTER, N_EXPERTS, 40, 0, &slot_gb_a, &budget_a);
        int cap_override  = qwen36_cap_for_ram(4.83, 20.0, 64.0, HIDDEN, INTER, N_EXPERTS, 40, 0, &slot_gb_b, &budget_b);
        CHECK(cap_override > cap_computed);
        CHECK(slot_gb_a == slot_gb_b); /* override changes the budget, not the per-slot size */
        CHECK(budget_a == 4.83 + 20.0 * 0.88);
        CHECK(budget_b == 64.0);
    }

    /* A negative or zero override (e.g. RAM_GB unset, or set to garbage that
     * atof() parses as 0, or set to a negative number) is not an error --
     * it must fall back to the computed budget exactly as if RAM_GB had
     * never been read, not propagate a negative/zero budget. */
    {
        double budget_unset = 0, budget_negative = 0, budget_zero = 0;
        int cap_unset = qwen36_cap_for_ram(4.83, 20.0, 0.0, HIDDEN, INTER, N_EXPERTS, 40, 0, NULL, &budget_unset);
        int cap_negative = qwen36_cap_for_ram(4.83, 20.0, -5.0, HIDDEN, INTER, N_EXPERTS, 40, 0, NULL, &budget_negative);
        int cap_zero = qwen36_cap_for_ram(4.83, 20.0, 0.0, HIDDEN, INTER, N_EXPERTS, 40, 0, NULL, &budget_zero);
        CHECK(cap_unset == cap_negative && cap_unset == cap_zero);
        CHECK(budget_unset == budget_negative && budget_unset == budget_zero);
    }

    /* int4 (xf_mode) halves the per-slot byte cost of int8, so for identical
     * budget/geometry it derives a cap at least as large, strictly larger in
     * the constrained-budget middle of the range (not floored, not clamped
     * at n_experts on either side). */
    {
        double slot_gb_int8 = 0, slot_gb_int4 = 0;
        int cap_int8 = qwen36_cap_for_ram(4.83, 3.0, 0.0, HIDDEN, INTER, N_EXPERTS, 40, 0, &slot_gb_int8, NULL);
        int cap_int4 = qwen36_cap_for_ram(4.83, 3.0, 0.0, HIDDEN, INTER, N_EXPERTS, 40, 1, &slot_gb_int4, NULL);
        CHECK(slot_gb_int4 < slot_gb_int8);
        CHECK(cap_int4 >= cap_int8);
        CHECK(cap_int4 > 1 && cap_int4 < N_EXPERTS);
        CHECK(cap_int8 > 1 && cap_int8 < N_EXPERTS);
        CHECK(cap_int4 > cap_int8);
    }

    /* A budget at or below the resident floor still returns a usable cap>=1
     * -- never 0, which would leave every layer's cache empty (expert_get
     * would then find no slot to evict and hang; see the comment above the
     * cap<0 CLI check in main()). */
    {
        int cap = qwen36_cap_for_ram(4.83, 0.0, 4.83, HIDDEN, INTER, N_EXPERTS, 40, 0, NULL, NULL);
        CHECK(cap == 1);
        int cap_neg_room = qwen36_cap_for_ram(4.83, 0.0, 1.0, HIDDEN, INTER, N_EXPERTS, 40, 0, NULL, NULL);
        CHECK(cap_neg_room == 1);
    }

    /* A huge budget clamps at n_experts, never runs past it. */
    {
        int cap = qwen36_cap_for_ram(4.83, 1000.0, 0.0, HIDDEN, INTER, N_EXPERTS, 40, 0, NULL, NULL);
        CHECK(cap == N_EXPERTS);
    }

    /* n_active_layers scales the derived cap: the same total budget spread
     * over twice as many layers yields roughly half the per-layer slots
     * (Segment/Edge partial-model builds pass layer_end-layer_begin here,
     * not always the model's full n_layers). */
    {
        int cap_40 = qwen36_cap_for_ram(4.83, 3.0, 0.0, HIDDEN, INTER, N_EXPERTS, 40, 0, NULL, NULL);
        int cap_80 = qwen36_cap_for_ram(4.83, 3.0, 0.0, HIDDEN, INTER, N_EXPERTS, 80, 0, NULL, NULL);
        CHECK(cap_80 < cap_40);
        CHECK(cap_80 >= cap_40 / 2 - 1 && cap_80 <= cap_40 / 2 + 1);
    }

    /* Degenerate inputs don't crash and still return a floored cap. */
    {
        CHECK(qwen36_cap_for_ram(4.83, 3.0, 0.0, HIDDEN, INTER, N_EXPERTS, 0, 0, NULL, NULL) >= 1);
        CHECK(qwen36_cap_for_ram(4.83, 3.0, 0.0, HIDDEN, INTER, N_EXPERTS, -1, 0, NULL, NULL) >= 1);
        CHECK(qwen36_cap_for_ram(0.0, 0.0, 0.0, HIDDEN, INTER, N_EXPERTS, 40, 0, NULL, NULL) >= 1);
    }

    printf(fails ? "test_qwen36_cap_precedence: FAIL (%d)\n" : "test_qwen36_cap_precedence: PASS\n", fails);
    return fails != 0;
}
