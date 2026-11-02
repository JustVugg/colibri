/* The CUDA tier with a GGUF's experts (qt_init_gguf), on the fake backend: the file's rows go
 * up as the backend's block formats, gate/up and down of each layer's own types.
 *   - With the arena (the default): one device allocation, slots of the biggest layer's
 *     expert, every expert uploaded into a slot (coli_cuda_tensor_upload_at), and a swap may
 *     pair layers of any types, the newcomer taking the victim's slot.
 *   - Exact slots (qt_gguf_exact_slots, qwen36's default): each expert in the room of its own
 *     layer's matrices, carved from one allocation, and a swap pairs slots of one size.
 *   - Without it (the arena refused, or QT_GGUF_ARENA=0): an expert costs what three device
 *     allocations take, and a swap pairs only layers of the same types: a swap into another
 *     layer's types freed and allocated device memory under a dense chain's hybrid layer,
 *     which deadlocked it (the device waiting for the host's word, the host's next launch
 *     behind the free). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}
static void drain(int want) {
    for (int i = 0; i < 1000 && fake_uploads < want; i++) {
        struct timespec ts = {0, 2000000};
        nanosleep(&ts, NULL);
    }
}

enum { NL = 3, NE = 4, D = 256, IH = 256, TOPK = 2 };
/* layers 0 and 1: IQ2_XXS gate/up, IQ2_S down; layer 2 a bigger down (IQ3_S), as published files mix them */
static const int tg[NL] = {16, 16, 16}, td[NL] = {22, 22, 21};
static const size_t gb[NL] = {(size_t)IH * 66, (size_t)IH * 66, (size_t)IH * 66}, db[NL] = {(size_t)D * 82, (size_t)D * 82, (size_t)D * 110};

static void run(int arena) {
    printf("%s\n", arena ? "the arena:" : "a device allocation a matrix:");
    fake_alloc_fail = !arena;   /* the arena refused: the tier falls back */
    if (!qt_init_gguf(NL, NE, D, IH, NE, TOPK, tg, td, gb, db, 0)) { printf("  FAIL: the tier refused a GGUF's experts\n"); fails++; return; }
    check(G.wfmt == QT_GGUF, "a GGUF's tier holds its block formats");
    check(G_arena_on == arena, arena ? "the arena is up" : "no arena when the device refuses it");
    size_t pg = (gb[2] + 3 * (size_t)IH) / 4 * 4, pd = (db[2] + 3 * (size_t)D) / 4 * 4;
    if (arena) check(G.exp_bytes == 2 * qt_pad256(pg) + qt_pad256(pd), "an expert costs its slot: the biggest layer's rows, no rounding");
    else check(G.exp_bytes == 2 * dev_alloc_footprint(pg) + dev_alloc_footprint(pd), "an expert costs the biggest layer's three allocations");
    int f, fd;
    check(qt_chain_format(&f, NULL) && f == QT_GGUF, "the chain sees a GGUF's formats");
    check(qt_chain_layer_format(0, &f, &fd) && f == 116 && fd == 122, "layer 0: IQ2_XXS gate/up, IQ2_S down");
    check(qt_chain_layer_format(2, &f, &fd) && f == 116 && fd == 121, "layer 2: IQ3_S down");
    check(swap_ok(0, 1) && (arena ? swap_ok(0, 2) : !swap_ok(0, 2) && !swap_ok(2, 1)),
          arena ? "the arena's slots take any layer's types" : "without the arena, swaps pair only same-typed layers");

    /* every expert goes up as the file's rows, its own types */
    unsigned char *g = malloc(gb[2]), *u = malloc(gb[2]), *d = malloc(db[2]);
    memset(g, 0x31, gb[2]); memset(u, 0x32, gb[2]); memset(d, 0x33, db[2]);
    int layers[NL * NE], eids[NL * NE];
    int planned = qt_plan_fill(layers, eids, NL * NE);
    check(planned == NL * NE, "every expert fits the fake device");
    int before = fake_uploads, at0 = fake_upload_at;
    for (int i = 0; i < planned; i++) qt_note_planned(layers[i], eids[i], g, u, d, NULL, NULL, NULL);
    drain(before + 3 * planned);
    qt_fill_wait();
    check(fake_uploads == before + 3 * planned, "three matrices an expert went up");
    check(arena ? fake_upload_at == at0 + 3 * planned : fake_upload_at == at0, arena ? "into the arena's slots" : "as tensors of their own");
    if (arena) {
        int distinct = 1, used = 0;
        for (int i = 0; i < NL * NE; i++) { if (G.slot[i].aslot) used++; for (int j = 0; j < i; j++) if (G.slot[i].aslot && G.slot[i].aslot == G.slot[j].aslot) distinct = 0; }
        check(used == NL * NE && distinct && G_anfree[0] == G_aslots[0] - NL * NE, "each resident holds a slot of its own");
    }

    /* the re-plan: candidates in layer 0, residents in layers 1 and 2, layer 2's the coldest */
    pthread_mutex_lock(&G.mx);
    for (int e = 0; e < NE; e++) res_set(qs(0, e), 0);   /* layer 0 not resident (its tensors and slots stay, as an eviction leaves them) */
    pthread_mutex_unlock(&G.mx);
    uint32_t *counts = calloc(NL * NE, sizeof *counts);
    for (int e = 0; e < NE; e++) { counts[e] = 100 + e; counts[NE + e] = 5; }
    int np = qt_replan(-1, counts, 8);
    check(np == NE, "each of layer 0's experts found a victim");
    qt_fill_wait();
    int vl[16], ve[16], nv = qt_evicted_take(vl, ve, 16), want = 1;
    for (int i = 0; i < nv; i++) want &= arena ? vl[i] == 2 : vl[i] == 1;
    check(nv == NE && want, arena ? "the coldest go, layer 2's (IQ3_S): the slot takes layer 0's types" :
                                    "the victims are layer 1's (IQ2_S down), never layer 2's (IQ3_S)");
    qt_shutdown();
    check(!G_arena[0] && !G_arena_on, "the arena goes with the tier");
    free(g); free(u); free(d); free(counts);
}

/* A dense chain takes the arena (qt_chain_own): the tier tells it the slots and their residents,
 * then places, swaps and computes none, whatever it is handed */
static void run_own(void) {
    printf("the chain takes the arena:\n");
    fake_alloc_fail = 0;
    if (!qt_init_gguf(NL, NE, D, IH, NE, TOPK, tg, td, gb, db, 0)) { printf("  FAIL: the tier refused a GGUF's experts\n"); fails++; return; }
    unsigned char *g = malloc(gb[2]), *u = malloc(gb[2]), *d = malloc(db[2]);
    memset(g, 0x31, gb[2]); memset(u, 0x32, gb[2]); memset(d, 0x33, db[2]);
    int layers[NL * NE], eids[NL * NE];
    int planned = qt_plan_fill(layers, eids, NL * NE), before = fake_uploads;
    for (int i = 0; i < planned; i++) qt_note_planned(layers[i], eids[i], g, u, d, NULL, NULL, NULL);
    drain(before + 3 * planned);
    qt_fill_wait();
    void *base; size_t slot, ou, od; int n;
    check(qt_chain_arena(&base, &slot, &ou, &od, &n) && base == G_arena[0] && slot == G_aslot && ou == G_aoff_u && od == G_aoff_d &&
          n == G_aslots[0], "the arena's base, slot and offsets");
    int same = 1;
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) same &= qt_chain_slot(l, e) == qs(l, e)->aslot - 1;
    check(same, "each resident's slot");
    qt_chain_own();
    ColiCudaTensor *tg_, *tu_, *td_;
    check(!qt_chain_resident(0, 0, 0, &tg_, &tu_, &td_), "no resident answers once the chain moves them");
    check(qt_chain_slot(0, 0) >= 0, "the slots it took still answer");
    int at = fake_uploads;
    pthread_mutex_lock(&G.mx);
    res_set(qs(1, 2), 0);   /* a resident gone, as the chain's eviction leaves the tier's books */
    pthread_mutex_unlock(&G.mx);
    qt_note(1, 2, g, u, d, NULL, NULL, NULL); qt_note_planned(1, 2, g, u, d, NULL, NULL, NULL);
    check(!qt_note_wanted(1, 2), "no note wanted");
    int ids[TOPK] = {0, 1};
    float x[D]; memset(x, 0, sizeof x);
    check(qt_issue(0, ids, TOPK, x) == 0, "no group issued");
    for (int t = 0; t < 40; t++) qt_chain_count(0, ids, TOPK);   /* ticks: the tier's own swaps would come now */
    qt_fill_wait();
    check(fake_uploads == at, "nothing uploaded");
    qt_shutdown();
    check(!G_chain_own, "the next tier starts the tier's own");
    free(g); free(u); free(d);
}

/* Exact slots: layers 0 and 1 take one size, layer 2 (its bigger down) another; the slots tile
 * the arena with no gap and no overlap, and the re-plan's victims are the same size's, layer 1's,
 * though layer 2's are the coldest */
static void run_exact(void) {
    printf("exact slots:\n");
    fake_alloc_fail = 0;
    qt_gguf_exact_slots(1);
    if (!qt_init_gguf(NL, NE, D, IH, NE, TOPK, tg, td, gb, db, 0)) { printf("  FAIL: the tier refused a GGUF's experts\n"); fails++; qt_gguf_exact_slots(0); return; }
    check(G_exact && G_arena_on, "the exact arena is up");
    size_t pg = (gb[0] + 3 * (size_t)IH) / 4 * 4, pd0 = (db[0] + 3 * (size_t)D) / 4 * 4, pd2 = (db[2] + 3 * (size_t)D) / 4 * 4;
    size_t s0 = 2 * qt_pad256(pg) + qt_pad256(pd0), s2 = 2 * qt_pad256(pg) + qt_pad256(pd2), all = (size_t)NE * (2 * s0 + s2);
    check(qt_ebytes(0) == s0 && qt_ebytes(1) == s0 && qt_ebytes(2) == s2 && s0 < s2 && G_ncls == 2, "each layer's slot holds its own matrices: two sizes");
    check((size_t)G_aslots[0] * 256 == all, "one allocation of every expert's slot");
    check(swap_ok(0, 1) && !swap_ok(0, 2) && !swap_ok(2, 1), "a swap pairs slots of one size");
    void *base; size_t sl, ou, od; int n;
    check(!qt_chain_arena(&base, &sl, &ou, &od, &n), "no chain is told of slots of one size");

    unsigned char *g = malloc(gb[2]), *u = malloc(gb[2]), *d = malloc(db[2]);
    memset(g, 0x31, gb[2]); memset(u, 0x32, gb[2]); memset(d, 0x33, db[2]);
    int layers[NL * NE], eids[NL * NE];
    int planned = qt_plan_fill(layers, eids, NL * NE), before = fake_uploads, at0 = fake_upload_at;
    check(planned == NL * NE, "every expert fits");
    for (int i = 0; i < planned; i++) qt_note_planned(layers[i], eids[i], g, u, d, NULL, NULL, NULL);
    drain(before + 3 * planned);
    qt_fill_wait();
    check(fake_uploads == before + 3 * planned && fake_upload_at == at0 + 3 * planned, "three matrices an expert into the arena");
    check(G_abump == all && G.used[0] == all, "the arena carved and the budget counted each expert's slot");
    int tiled = 1;
    for (int i = 0; i < NL * NE; i++) {
        size_t oi = (size_t)(G.slot[i].aslot - 1) * 256, ei = oi + qt_ebytes(i / NE);
        if (!G.slot[i].aslot || ei > all) { tiled = 0; continue; }
        for (int j = 0; j < i; j++) {
            size_t oj = (size_t)(G.slot[j].aslot - 1) * 256, ej = oj + qt_ebytes(j / NE);
            if (G.slot[j].aslot && oi < ej && oj < ei) tiled = 0;
        }
    }
    check(tiled, "each resident in a slot of its own, inside the arena, none overlapping");

    pthread_mutex_lock(&G.mx);
    for (int e = 0; e < NE; e++) res_set(qs(0, e), 0);
    pthread_mutex_unlock(&G.mx);
    uint32_t *counts = calloc(NL * NE, sizeof *counts);
    for (int e = 0; e < NE; e++) { counts[e] = 100 + e; counts[NE + e] = 5; }
    int np = qt_replan(-1, counts, 8);
    check(np == NE, "each of layer 0's experts found a victim");
    qt_fill_wait();
    int vl[16], ve[16], nv = qt_evicted_take(vl, ve, 16), want = 1;
    for (int i = 0; i < nv; i++) want &= vl[i] == 1;
    check(nv == NE && want, "the victims are layer 1's, of layer 0's size, never layer 2's");
    check(G.used[0] == all, "a swap moves no budget");
    qt_shutdown();
    qt_gguf_exact_slots(0);
    check(!G_arena[0] && !G_arena_on && !G_exact && !G_lsl, "the arena and its sizes go with the tier");
    free(g); free(u); free(d); free(counts);
}

int main(void) {
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    fake_block_elems = 0;
    check(!qt_init_gguf(NL, NE, D, IH, NE, TOPK, tg, td, gb, db, 0), "a backend that takes no block format refuses a GGUF's experts");
    fake_block_elems = 256;
    run(1);
    run(0);
    run_own();
    run_exact();
    printf(fails ? "qwen36 tier gguf: %d FAIL\n" : "qwen36 tier gguf: ok\n", fails);
    return fails != 0;
}
