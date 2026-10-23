/* qt_replan: the resident set follows the prompt.
 *
 * The warmstart fills VRAM from a heat file, i.e. from what earlier prompts
 * routed to. A new prompt routes elsewhere, and the LFRU tick corrects that
 * one expert per sixteen tokens. qt_replan takes this prompt's own routing
 * counts (the engine's prefill counts) and swaps residents the prompt never
 * touched for its most-routed non-residents, budget-neutral, through the same
 * victim-first swap the tick uses, in strict count order and only while the
 * newcomer's count beats the victim's. Evicted experts are reported through
 * qt_evicted_take so the engine can rebuild their CPU copies before the miss
 * path needs them. Fake CUDA backend, no GPU, no toolkit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"   /* setenv: MinGW has none */
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

enum { NL = 2, NE = 8, D = 64, IH = 32, TOPK = 2 };

static int resident_set(int layer, int *out) {   /* returns count, fills sorted eids */
    int n = 0;
    pthread_mutex_lock(&G.mx);
    for (int e = 0; e < NE; e++) if (qs(layer, e)->resident) out[n++] = e;
    pthread_mutex_unlock(&G.mx);
    return n;
}
static int same(const int *a, int na, const int *b, int nb) {
    if (na != nb) return 0;
    for (int i = 0; i < na; i++) if (a[i] != b[i]) return 0;
    return 1;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1); setenv("COLI_PLACE", "off", 1);
    setenv("QT_UPLOAD_SYNC", "", 1);
    fake_ndev = 1; fake_uploads = 0; fake_overwrites = 0;

    /* budget: exactly eight experts, four per layer once noted in interleaved order */
    size_t exp_bytes = 3 * dev_alloc_footprint((size_t)D * IH / 2) + 3 * dev_alloc_footprint((2 * IH + D) / 3 * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(8 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts (int4 mode, per-row scales)");

    static unsigned char g4[NL][NE][D * IH / 2], u4[NL][NE][D * IH / 2], d4[NL][NE][D * IH / 2];
    static float sc[NL][NE][2 * IH + D];
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++) {
        memset(g4[l][e], (unsigned char)(e + 1), sizeof g4[l][e]); memset(u4[l][e], (unsigned char)(e + 2), sizeof u4[l][e]);
        memset(d4[l][e], (unsigned char)(e + 3), sizeof d4[l][e]);
        for (int i = 0; i < 2 * IH + D; i++) sc[l][e][i] = 1.0f;
        /* experts 0..3 of both layers fit the budget and go up; 4..7 only leave
         * their RAM pointers with the tier, which is what a real warmstart does
         * for the experts it cannot place */
        qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sc[l][e], sc[l][e] + IH, sc[l][e] + 2 * IH);
    }
    qt_fill_wait();
    int res[NE]; int n = resident_set(0, res);
    { int want[] = {0, 1, 2, 3}; check(same(res, n, want, 4), "warmstart: layer 0 experts 0..3 resident, 4..7 not"); }
    n = resident_set(1, res);
    { int want[] = {0, 1, 2, 3}; check(same(res, n, want, 4), "warmstart: layer 1 experts 0..3 resident, 4..7 not"); }
    check(fake_uploads == 3 * 8, "eight experts uploaded (three tensors each)");
    size_t used0 = G.used[0];

    /* the prompt's counts: it never routed to 0 and 1, a little to 2 and 3,
     * mostly to 4 and 5, once to 6, never to 7 */
    uint32_t freq[NE] = {0, 0, 5, 7, 10, 9, 1, 0};

    /* layer 0, cap 1: only the best pair, 4 (10) for 0 (count 0, the coldest resident) */
    ColiCudaTensor *victim_tg; pthread_mutex_lock(&G.mx); victim_tg = qs(0, 0)->tg; pthread_mutex_unlock(&G.mx);
    int planned = qt_replan(0, freq, 1);
    check(planned == 1, "cap 1 plans exactly one swap");
    qt_fill_wait();
    pthread_mutex_lock(&G.mx);
    check(qs(0, 4)->tg == victim_tg && qs(0, 0)->tg == NULL, "the swap is in place: the newcomer took the victim's device buffers");
    pthread_mutex_unlock(&G.mx);
    check(fake_overwrites == 3 && fake_uploads == 3 * 8, "three overwrites, no new upload");
    n = resident_set(0, res);
    { int want[] = {1, 2, 3, 4}; check(same(res, n, want, 4), "after the first swap: 4 in, 0 out"); }
    n = resident_set(1, res);
    { int want[] = {0, 1, 2, 3}; check(same(res, n, want, 4), "a layer-0 re-plan leaves layer 1 alone"); }
    check(G.used[0] == used0, "a swap is budget-neutral");
    int ls[8], es[8];
    int nev = qt_evicted_take(ls, es, 8);
    check(nev == 1 && ls[0] == 0 && es[0] == 0, "the evicted expert is reported once, as (layer 0, expert 0)");
    check(qt_evicted_take(ls, es, 8) == 0, "and not a second time");

    /* no cap: 5 (9) for 1 (count 0); 6 (1) against the next victim 2 (5) is
     * not worth a swap, so the plan stops there */
    planned = qt_replan(0, freq, 100);
    check(planned == 1, "the second plan holds one pair: a newcomer must beat its victim's count");
    qt_fill_wait();
    n = resident_set(0, res);
    { int want[] = {2, 3, 4, 5}; check(same(res, n, want, 4), "after the second swap: 5 in, 1 out; 2 and 3 stay"); }
    check(fake_overwrites == 6 && fake_uploads == 3 * 8, "again in place: six overwrites so far, still no new upload");
    check(G.inplace == 2, "the tier counts both swaps as in-place");
    nev = qt_evicted_take(ls, es, 8);
    check(nev == 1 && es[0] == 1, "expert 1 reported as evicted");
    check(G.rp_n == G.rp_i, "nothing left pending: the queue was idle, the plan drained at once");

    /* same counts again: the set already matches the prompt */
    check(qt_replan(0, freq, 100) == 0, "a re-plan on the same counts plans nothing");
    check(qt_replan(0, NULL, 100) == 0 && qt_replan(0, freq, 0) == 0 && qt_replan(NL, freq, 100) == 0,
          "NULL counts, a zero cap or a layer out of range plan nothing");

    /* layer 1 from its own counts: 7 (10) takes the place of a never-routed
     * resident; 0 is resident and routed, so it stays */
    uint32_t freq1[NE] = {9, 0, 0, 0, 0, 0, 0, 10};
    fake_overwrite_fail = 1;   /* a backend without the entry point: the swap must still land, by free + upload */
    check(qt_replan(1, freq1, 100) == 1, "layer 1: one swap, 7 for one of the count-0 residents");
    qt_fill_wait();
    fake_overwrite_fail = 0;
    check(fake_overwrites == 6 && fake_uploads == 3 * 9, "without overwrite support the swap falls back to a fresh upload");
    n = resident_set(1, res);
    check(n == 4 && res[0] == 0 && res[3] == 7, "layer 1 keeps 0 and now holds 7");
    n = resident_set(0, res);
    { int want[] = {2, 3, 4, 5}; check(same(res, n, want, 4), "layer 0 untouched by the layer-1 plan"); }
    nev = qt_evicted_take(ls, es, 8);
    check(nev == 1 && ls[0] == 1, "the layer-1 victim is reported with its layer");

    /* the whole-table form (layer < 0) plans across layers, per device: layer 0's
     * expert 6 (routed once) is worth more than layer 1's never-routed residents,
     * which the per-layer form could not trade against each other */
    uint32_t all[NL * NE]; memcpy(all, freq, sizeof freq); memcpy(all + NE, freq1, sizeof freq1);
    check(qt_replan(-1, all, 100) == 1, "the whole-table form plans one cross-layer swap");
    qt_fill_wait();
    check(resident_set(0, res) == 5 && resident_set(1, res) == 3, "layer 0 gained a resident, layer 1 lost one");
    check(G.used[0] == used0, "still budget-neutral");
    check(qt_replan(-1, all, 100) == 0, "and then nothing is left worth a swap");

    /* the decode marker: stats after the mark count from here */
    qt_stats_mark();
    check(G.mk_on && G.mk_hits == 0 && G.mk_miss == 0, "the marker snapshots the counters (none yet)");
    qt_stats();

    qt_shutdown();
    if (fails) { printf("test_qwen36_tier_replan: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_tier_replan: the resident set follows the prompt's counts, budget-neutral, evictions reported\n");
    return 0;
}
