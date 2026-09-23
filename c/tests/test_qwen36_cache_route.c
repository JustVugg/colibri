/* CACHE_ROUTE in the qwen36 engine: route_select() against a residency table.
 *
 * The lever changes which experts run, so what it must never do is spelled
 * out here: touch the true top-J, look past the top-M window, rank an expert
 * the group mask excluded, or report agreement it does not have. The two
 * residency levels (VRAM tier over RAM cache) and ROUTE_ALPHA are pinned too.
 * No Model, no weights: the function takes the softmax row and a callback. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include <stdio.h>
#include <string.h>

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { fails++; printf("  FAIL: %s\n", what); }
}

/* residency table for the callback: level per expert id */
static int8_t g_lvl[64];
static int lvl_table(void *ctx, int e) { (void)ctx; return g_lvl[e]; }

/* softmax row where expert e has mass proportional to (E - e): rank == id. */
static void ranked_row(float *pr, int E) {
    float sum = 0; for (int e = 0; e < E; e++) { pr[e] = (float)(E - e); sum += pr[e]; }
    for (int e = 0; e < E; e++) pr[e] /= sum;
}

static int has(const int *idx, int K, int e) { for (int k = 0; k < K; k++) if (idx[k] == e) return 1; return 0; }

int main(void) {
    enum { E = 16, K = 4 };
    float pr[E]; uint8_t keep[E]; int idx[K]; float val[K]; RouteStats st;
    ranked_row(pr, E); memset(keep, 1, sizeof keep);

    /* nothing resident: the plain top-K, no substitution, full agreement, zero KL */
    memset(g_lvl, 0, sizeof g_lvl); memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, 2, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(idx[0]==0 && idx[1]==1 && idx[2]==2 && idx[3]==3, "no_resident_expert_means_the_true_top_k");
    check(st.swaps == 0 && st.agree_hit == K && st.agree_tot == K, "no_substitution_reports_full_agreement");
    check(st.kl_n == 1 && st.kl_sum == 0.0, "identical_choice_has_zero_kl");
    check(val[0] == pr[0] && val[3] == pr[3], "val_carries_the_raw_mass_of_each_chosen_expert");

    /* a RAM-resident expert inside the window replaces the unresident tail */
    memset(g_lvl, 0, sizeof g_lvl); g_lvl[7] = 1; memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, 2, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(idx[0]==0 && idx[1]==1, "true_top_j_is_taken_first");
    check(idx[2]==7 && idx[3]==2, "resident_expert_in_window_fills_before_the_unresident_ranking");
    check(st.swaps == 1 && st.swaps_vram == 0 && st.agree_hit == 3, "one_substitution_is_counted_once_and_not_as_vram");
    check(st.kl_sum > 0.0, "a_substitution_shows_up_as_positive_kl");

    /* the sacred top-J is taken even when a lower rank is resident and it is not */
    memset(g_lvl, 0, sizeof g_lvl); g_lvl[2] = 1; g_lvl[3] = 1; g_lvl[4] = 1; g_lvl[5] = 1; memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, 2, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(idx[0]==0 && idx[1]==1 && idx[2]==2 && idx[3]==3, "unresident_top_j_is_never_displaced");
    check(st.swaps == 0, "resident_experts_already_in_the_top_k_are_not_substitutions");

    /* VRAM outranks RAM inside the window, whatever their rank order */
    memset(g_lvl, 0, sizeof g_lvl); g_lvl[5] = 1; g_lvl[9] = 2; memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, 2, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(idx[2]==9 && idx[3]==5, "vram_resident_fills_before_ram_resident");
    check(st.swaps == 2 && st.swaps_vram == 1, "swaps_to_vram_are_counted_separately");

    /* residency outside the top-M window is ignored */
    memset(g_lvl, 0, sizeof g_lvl); g_lvl[12] = 2; g_lvl[15] = 1; memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, 2, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(!has(idx, K, 12) && !has(idx, K, 15) && idx[2]==2 && idx[3]==3, "resident_expert_past_the_window_is_not_chosen");

    /* ROUTE_J=0: even rank 0 may be replaced; ROUTE_J=K: nothing may */
    memset(g_lvl, 0, sizeof g_lvl); g_lvl[6] = 1; g_lvl[7] = 1; g_lvl[8] = 1; g_lvl[9] = 1; memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, 0, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(idx[0]==6 && idx[1]==7 && idx[2]==8 && idx[3]==9 && st.swaps == 4, "route_j_zero_lets_residents_take_every_slot");
    memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, K, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(idx[0]==0 && idx[3]==3 && st.swaps == 0, "route_j_equal_to_k_disables_substitution");

    /* the group mask is respected: an excluded expert is never ranked or chosen */
    memset(g_lvl, 0, sizeof g_lvl); g_lvl[5] = 2; memset(&st, 0, sizeof st);
    keep[1] = 0; keep[5] = 0;
    route_select(pr, keep, E, K, 2, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(!has(idx, K, 1) && !has(idx, K, 5), "masked_experts_are_never_chosen_even_when_resident");
    check(idx[0]==0 && idx[1]==2 && idx[2]==3 && idx[3]==4, "masked_ranking_shifts_the_true_top_k");
    memset(keep, 1, sizeof keep);

    /* ROUTE_ALPHA scales only the substitute's mass, before moe() renormalises */
    memset(g_lvl, 0, sizeof g_lvl); g_lvl[7] = 1; memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, 2, 12, 0.f, 0.5f, lvl_table, NULL, idx, val, &st);
    check(idx[2]==7 && val[2] == pr[7]*0.5f && val[0] == pr[0] && val[3] == pr[2], "alpha_halves_the_substitute_and_nothing_else");

    /* ROUTE_P: the window is the mass, not M. Ranks 0..4 hold 70/136 = 0.515 of the
     * mass, so P=0.5 closes the window at rank 4: rank 4 is inside, rank 5 outside. */
    memset(g_lvl, 0, sizeof g_lvl); g_lvl[4] = 1; g_lvl[5] = 1; memset(&st, 0, sizeof st);
    route_select(pr, keep, E, K, 2, 12, 0.5f, 1.f, lvl_table, NULL, idx, val, &st);
    check(has(idx, K, 4) && !has(idx, K, 5), "route_p_bounds_the_window_by_cumulative_mass");

    /* fewer eligible experts than K: the remainder is marked -1, not garbage */
    memset(g_lvl, 0, sizeof g_lvl); memset(&st, 0, sizeof st);
    memset(keep, 0, sizeof keep); keep[3] = 1; keep[9] = 1;
    route_select(pr, keep, E, K, 2, 12, 0.f, 1.f, lvl_table, NULL, idx, val, &st);
    check(idx[0]==3 && idx[1]==9 && idx[2]==-1 && idx[3]==-1 && st.slots == 2, "short_candidate_list_pads_with_minus_one");

    if (fails) { printf("test_qwen36_cache_route: %d fallimenti\n", fails); return 1; }
    printf("test_qwen36_cache_route: ok\n");
    return 0;
}
