/* The int4 arena's slots under the chain's promotions, on the fake backend (COLI_CUDA_EXPERT_ARENA).
 *
 * With the arena an expert's three tensors are views of one slot of its device's arena, and the tier keeps the slot's
 * number (aslot) to hand the slot on when the expert is evicted. qt_chain_promote / qt_chain_promote_done move the
 * victim's tensors to the newcomer: the slot's number must move with them. After every step the books must hold:
 * every resident owns one slot of its own, its tensors point into it and the slot holds its bytes; a non-resident owns
 * none; the free slots, the residents' slots and the slots of pairs in flight are every slot once.
 *
 * What went wrong before (campaign records, g5-evidence/h4-load/RESULT.md, "Known problems"): the newcomer got the
 * views without the number, the victim kept it while not resident. Evicting the newcomer then stranded the slot;
 * and the victim, promoted again later into another victim's slot, carried the stale number into its next eviction,
 * where the tier wrote a newcomer into the slot the first newcomer still computed from. Also checked: a failed copy
 * with nothing in flight gives the victim's slot back with its budget, and the miss path refills it; a refused batch
 * whose copies may still land leaves its slots out of use, budget and all; the victim of a tier swap still in the
 * upload queue is never promoted (it keeps its tensors and slot until the uploader takes them: promoted, it took a
 * second victim's tensors over its own and the uploader then wrote the swap's newcomer into the slot it computed
 * from); qt_init refused after making the arena and the pool leaves neither behind; a staging buffer still out when
 * the tier shuts down keeps the pool alive (no write into freed page-locked memory), a later release lands in it,
 * and the next tier returns it and makes its own. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"   /* setenv, unsetenv */
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

enum { NL = 2, NE = 8, D = 64, IH = 32, TOPK = 2, GS = 16,
       MB = D * IH / 2,                        /* a matrix of packed nibbles: 1024 bytes */
       SGU = IH * (D / GS), SD = D * (IH / GS), /* scale floats: gate or up 128, down 128 */
       NSLOT = 8 };                             /* the budget: eight of the sixteen experts */

static unsigned char g4[NL][NE][MB], u4[NL][NE][MB], d4[NL][NE][MB];
static float sg[NL][NE][SGU], su[NL][NE][SGU], sd[NL][NE][SD];

static void fill_sources(void) {
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        const int k = l * NE + e;
        for (int i = 0; i < MB; i++) {
            g4[l][e][i] = (unsigned char)(7 * k + 1 + i);
            u4[l][e][i] = (unsigned char)(7 * k + 2 + 5 * i);
            d4[l][e][i] = (unsigned char)(7 * k + 3 + 3 * i);
        }
        for (int i = 0; i < SGU; i++) { sg[l][e][i] = 1.0f + k; su[l][e][i] = 2.0f + k; }
        for (int i = 0; i < SD; i++) sd[l][e][i] = 3.0f + k;
    }
}
/* the slot holds expert (l, e)'s gate matrix: as the tier's upload stages it (offset-binary nibbles, XOR 0x88) or as
 * the chain's promotion copies it (qt_chain_promote_stage: the RAM's bytes as they are) */
static int slot_holds(int l, int e, const unsigned char *b) {
    if (!memcmp(b, g4[l][e], MB)) return 1;
    for (int i = 0; i < MB; i++) if (b[i] != (unsigned char)(g4[l][e][i] ^ 0x88)) return 0;
    return 1;
}
static const char *at(const char *when, const char *what) { static char m[256]; snprintf(m, sizeof m, "%s: %s", when, what); return m; }
/* the books, as the head comment says; returns the number of free slots */
static int check_books(const char *when) {
    unsigned char seen[NSLOT];
    memset(seen, 0, sizeof seen);
    int distinct = 1, views = 1, bytes = 1, nonres = 1, cover = 1;
    pthread_mutex_lock(&G.mx);
    const int nslots = G_aslots[0];
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        QSlot *s = qs(l, e);
        if (!s->resident) {
            if (s->queued && s->aslot > 0) {   /* a victim in flight: its slot until qt_chain_promote_done */
                const int a = s->aslot - 1;
                if (a < 0 || a >= nslots || seen[a]) distinct = 0; else seen[a] = 1;
            } else if (s->aslot != 0 || s->tg || s->tu || s->td) nonres = 0;
            continue;
        }
        const int a = s->aslot - 1;
        if (a < 0 || a >= nslots || seen[a]) { distinct = 0; continue; }
        seen[a] = 1;
        const unsigned char *b = (const unsigned char *)G_arena[0] + (size_t)a * G_aslot;
        if (!s->tg || !s->tu || !s->td || s->tg->w != b || s->tu->w != b + MB || s->td->w != b + 2 * MB) views = 0;
        if (!slot_holds(l, e, b)) bytes = 0;
    }
    for (int k = 0; k < G_anfree[0]; k++) {
        const int a = G_afree[0][k];
        if (a < 0 || a >= nslots || seen[a]) cover = 0; else seen[a] = 1;
    }
    int unseen = 0;
    for (int a = 0; a < nslots; a++) if (!seen[a]) unseen++;
    if (unseen != G_slots_lost) cover = 0;   /* a refused batch's slots: out of use, neither free nor owned */
    const int nfree = G_anfree[0];
    pthread_mutex_unlock(&G.mx);
    check(nslots == NSLOT, at(when, "the arena has the budget's eight slots"));
    check(distinct, at(when, "every resident (and every victim in flight) owns a slot of its own"));
    check(views, at(when, "a resident's tensors point at its own slot's matrices"));
    check(bytes, at(when, "a resident's slot holds that expert's bytes"));
    check(nonres, at(when, "a non-resident owns no slot and no tensors"));
    check(cover, at(when, "the free slots, the owned slots and the slots left out of use are every slot once"));
    return nfree;
}
static void step(int layer, int a, int b) {   /* a row's routes, the misses offered as the engine's miss path does */
    int ids[2] = { a, b };
    qt_chain_count(layer, ids, 2);
    for (int k = 0; k < 2; k++) if (ids[k] >= 0 && !qt_is_resident(layer, ids[k]))
        qt_note(layer, ids[k], g4[layer][ids[k]], u4[layer][ids[k]], d4[layer][ids[k]],
                sg[layer][ids[k]], su[layer][ids[k]], sd[layer][ids[k]]);
}
/* the chain's copy of a pair, as it lands in the victim's buffers (the slot's matrices, then the scales after them) */
static void land(const QtPromo *p) { qt_chain_promote_stage((void *)p->tg->w, p); }
/* the tier's own swap (as stream_promote_locked or the re-plan make one): newcomer (nl, ne) into the victim's place */
static int tier_swap(int nl, int ne, int vl, int ve) {
    pthread_mutex_lock(&G.mx);
    QSlot *v = qs(vl, ve);
    int ok = v->resident && !v->queued;
    if (ok) { res_set(v, 0); ok = enqueue_locked(nl, ne, vl, ve, 0); if (!ok) res_set(v, 1); }
    pthread_mutex_unlock(&G.mx);
    qt_fill_wait();
    return ok;
}
static int find_resident_not(int avoid1, int avoid2, int *l_out, int *e_out) {   /* a resident other than two slot indices */
    int r = 0;
    pthread_mutex_lock(&G.mx);
    for (int i = 0; i < NL * NE && !r; i++)
        if (i != avoid1 && i != avoid2 && G.slot[i].resident && !G.slot[i].queued) { *l_out = i / NE; *e_out = i % NE; r = 1; }
    pthread_mutex_unlock(&G.mx);
    return r;
}
static int find_nonresident_not(int avoid1, int avoid2, int *l_out, int *e_out) {
    int r = 0;
    pthread_mutex_lock(&G.mx);
    for (int i = 0; i < NL * NE && !r; i++)
        if (i != avoid1 && i != avoid2 && !G.slot[i].resident && !G.slot[i].queued) { *l_out = i / NE; *e_out = i % NE; r = 1; }
    pthread_mutex_unlock(&G.mx);
    return r;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1); setenv("COLI_PLACE", "off", 1);
    setenv("QT_UPLOAD_SYNC", "", 1);
    fake_ndev = 1;
    fill_sources();
    /* the slot: three 1024-byte matrices, then 384 floats of scales padded to 256: 4608 bytes; eight and a half of them */
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (NSLOT * 4608.0 + 2304.0) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    setenv("COLI_CUDA_EXPERT_ARENA", "1", 1);

    printf("promotions with the arena\n");
    check(qt_init(NL, NE, D, IH, NE, TOPK, GS, 1 /* int4 */) && G_arena_on && G_aslot == 4608, "the tier starts with the int4 arena, slots of 4608 bytes");
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++)
        qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sg[l][e], su[l][e], sd[l][e]);
    qt_fill_wait();
    check(check_books("warm start") == 0, "warm start: every slot taken");

    /* three steps: layer 0 routes 4 and 5, layer 1 routes 0 and 5; the third makes three promotions */
    QtPromo pr[8];
    for (int t = 0; t < 2; t++) { step(0, 4, 5); step(1, 0, 5); check(qt_chain_promote(8, pr) == 0, "not yet worth a swap"); }
    step(0, 4, 5); step(1, 0, 5);
    int np = qt_chain_promote(8, pr);
    check(np == 3, "three promotions");
    if (np != 3) { printf("FAILED: %d failures (no promotions to follow)\n", fails + 1); return 1; }
    check_books("three pairs in flight");
    land(&pr[0]); qt_chain_promote_done(pr[0].ci, pr[0].vi, 1);
    land(&pr[1]); qt_chain_promote_done(pr[1].ci, pr[1].vi, 1);
    const int failed_layer = pr[2].vi / NE;
    qt_chain_promote_done(pr[2].ci, pr[2].vi, 0);   /* this copy failed before a byte went: nothing in flight */
    check(check_books("two landed, one failed") == 1, "the failed copy's slot is free again");
    check(G.used[0] + qt_ebytes(failed_layer) <= G.budget[0], "and its budget is back, so a plain upload may take it");
    int nl, ne;
    check(find_nonresident_not(pr[0].vi, pr[1].vi, &nl, &ne), "a non-resident expert for the miss path");
    qt_note(nl, ne, g4[nl][ne], u4[nl][ne], d4[nl][ne], sg[nl][ne], su[nl][ne], sd[nl][ne]);
    qt_fill_wait();
    check(check_books("the miss path offered an expert") == 0 && qt_is_resident(nl, ne), "the miss path refilled the free slot");

    /* the tier evicts a promoted newcomer: its slot goes to the next expert, nothing is stranded */
    check(find_nonresident_not(pr[0].vi, pr[1].vi, &nl, &ne), "a non-resident expert to swap in");
    check(tier_swap(nl, ne, pr[1].ci / NE, pr[1].ci % NE), "the tier swaps a newcomer in for the second promoted expert");
    check(check_books("a promoted expert evicted by the tier") == 0, "no free slot: the newcomer took the evicted one's");

    /* the first victim comes back through a promotion (into another victim's slot), then the tier evicts it: the slot
     * it hands on must be the one it holds now, not the one the first newcomer took from it */
    const int v1 = pr[0].vi, c1 = pr[0].ci;
    for (int t = 0; t < 3; t++) step(v1 / NE, v1 % NE, -1);
    np = qt_chain_promote(8, pr);
    check(np == 1 && pr[0].ci == v1, "the first victim is promoted again");
    if (np == 1 && pr[0].ci == v1) {
        land(&pr[0]); qt_chain_promote_done(pr[0].ci, pr[0].vi, 1);
        check_books("the first victim resident again in another victim's slot");
        check(find_nonresident_not(c1, v1, &nl, &ne), "another non-resident expert");
        check(tier_swap(nl, ne, v1 / NE, v1 % NE), "the tier evicts the first victim");
        check_books("the first victim evicted again");
        pthread_mutex_lock(&G.mx);
        const QSlot *s = qs(c1 / NE, c1 % NE);
        const int ok = s->resident && s->aslot > 0 &&
                       slot_holds(c1 / NE, c1 % NE, (const unsigned char *)G_arena[0] + (size_t)(s->aslot - 1) * G_aslot);
        pthread_mutex_unlock(&G.mx);
        check(ok, "the first newcomer still computes from its own bytes (nothing was written into its slot)");
    }
    /* the remaining residents, evicted one after another by the tier: every slot is handed on, none twice */
    for (int t = 0; t < 4; t++) {
        int rl, re;
        if (!find_resident_not(-1, -1, &rl, &re) || !find_nonresident_not(rl * NE + re, -1, &nl, &ne)) break;
        tier_swap(nl, ne, rl, re);
    }
    check_books("four more swaps");
    qt_shutdown();
    check(!G_arena_on && G_arena[0] == NULL && fake_pipe_allocs == fake_pipe_frees, "the shutdown returns the arena");

    printf("a tier swap's victim still in the upload queue is not promoted\n");
    check(qt_init(NL, NE, D, IH, NE, TOPK, GS, 1) && G_arena_on, "a fresh tier with the arena");
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++)
        qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sg[l][e], su[l][e], sd[l][e]);
    qt_fill_wait();
    check(check_books("warm start again") == 0, "every slot taken");
    qt_chain_hold(1);   /* a chain forward holds the tier: the uploader touches no victim until it ends */
    pthread_mutex_lock(&G.mx);
    QSlot *w1 = qs(0, 0), *w2 = qs(1, 1);
    int q1 = w1->resident && !w1->queued && !qs(0, 6)->resident, q2 = w2->resident && !w2->queued && !qs(0, 7)->resident;
    if (q1) { res_set(w1, 0); q1 = enqueue_locked(0, 6, 0, 0, 0); if (!q1) res_set(w1, 1); }   /* (0,6) in for (0,0) */
    if (q2) { res_set(w2, 0); q2 = enqueue_locked(0, 7, 1, 1, 0); if (!q2) res_set(w2, 1); }   /* (0,7) in for (1,1) */
    for (int i = 0; i < NL * NE; i++) G.slot[i].heat = 1;
    w1->heat = w2->heat = 100;   /* both victims routed again in this conversation: the chain would want them back */
    pthread_mutex_unlock(&G.mx);
    check(q1 && q2, "two tier swaps queued behind the forward");
    for (int t = 0; t < 3; t++) { int a = 0, b = 1; qt_chain_count(0, &a, 1); qt_chain_count(1, &b, 1); }
    np = qt_chain_promote(8, pr);
    check(np == 0, "neither victim is promoted while its swap waits for the uploader");
    for (int k = 0; k < np; k++) qt_chain_promote_done(pr[k].ci, pr[k].vi, 0);
    qt_chain_hold(0);   /* the forward ends: the uploader runs both swaps */
    qt_fill_wait();
    check(check_books("both swaps done") == 0 && qt_is_resident(0, 6) && qt_is_resident(0, 7),
          "the swaps' newcomers resident, every slot owned once");
    qt_shutdown();

    printf("a refused promotion batch: its slots stay out of use\n");
    check(qt_init(NL, NE, D, IH, NE, TOPK, GS, 1) && G_arena_on && G_slots_lost == 0, "a fresh tier");
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++)
        qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sg[l][e], su[l][e], sd[l][e]);
    qt_fill_wait();
    for (int t = 0; t < 2; t++) { step(0, 4, 5); step(1, 0, 5); qt_chain_promote(8, pr); }
    step(0, 4, 5); step(1, 0, 5);
    np = qt_chain_promote(8, pr);
    check(np >= 1, "promotions to refuse");
    const size_t used_before = G.used[0];
    for (int k = 0; k < np; k++) qt_chain_promote_done(pr[k].ci, pr[k].vi, -1);   /* refused after some copies went */
    check(G_slots_lost == np, "each refused pair's slot counted out of use");
    check(G.used[0] == used_before, "their budget still counted: nothing is uploaded where a late copy may land");
    check(check_books("a refused batch") == 0, "no slot freed");
    qt_shutdown();

    printf("qt_init refused after the arena and the pool\n");
    setenv("COLI_CUDA_EXPERT_ARENA", "2", 1);
    setenv("QT_FAIL_INIT_LATE", "1", 1);
    const int pa = fake_pipe_allocs, ha = fake_host_allocs;
    check(!qt_init(NL, NE, D, IH, NE, TOPK, GS, 1), "qt_init refuses (the test's late failure)");
    check(fake_pipe_allocs > pa && fake_host_allocs > ha, "it had made the arena and the staging pool");
    check(!G_arena_on && G_arena[0] == NULL && fake_pipe_allocs == fake_pipe_frees, "the arena is returned and its flag cleared");
    check(G_stg[0] == NULL && !G_stg_closing[0], "the staging pool is returned");
    unsetenv("QT_FAIL_INIT_LATE");

    printf("a staging buffer still out when the tier shuts down\n");
    check(qt_init(NL, NE, D, IH, NE, TOPK, GS, 1) && G_stg[0] != NULL, "the tier starts with the arena and the pool");
    uint8_t *out = qt_stg_take(0);   /* as an engine thread staging an expert outside the tier's lock holds one */
    check(out != NULL && qt_stg_owns(out), "a buffer taken from the pool");
    qt_shutdown();
    check(G_stg[0] != NULL && G_stg_closing[0], "the shutdown keeps the pool while the buffer is out");
    check(qt_stg_take(0) == NULL, "and hands out no buffer from it");
    uint8_t *const kept = G_stg[0];
    check(qt_init(NL, NE, D, IH, NE, TOPK, GS, 1), "a new tier starts while the buffer is still out");
    check(G_stg[0] == kept && G_stg_closing[0], "without a pool of its own while the old one is kept");
    qt_shutdown();
    memset(out, 0x5a, 64);          /* the staging thread may still write it: the memory is not freed */
    qt_stg_release(out, (float *)(out + G_aoff_s));
    check(G_stg_nfree[0] == QT_STG_N, "the late release lands in the pool");
    const int hf = fake_host_frees;
    check(qt_init(NL, NE, D, IH, NE, TOPK, GS, 1), "the next tier starts");
    check(fake_host_frees == hf + 1 && G_stg[0] != NULL && !G_stg_closing[0] && G_stg_nfree[0] == QT_STG_N,
          "it returns the kept pool (every buffer back) and makes its own");
    qt_shutdown();
    check(G_stg[0] == NULL && !G_stg_closing[0], "and its shutdown returns that one");

    printf("%s: %d failures\n", fails ? "FAILED" : "PASSED", fails);
    return fails != 0;
}
