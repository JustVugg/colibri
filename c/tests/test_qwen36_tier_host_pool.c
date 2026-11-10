/* qt_chain_host_follow: the host pool follows the card's residents.
 *
 * The warm start page-locks the RAM copy of the experts it leaves off the card (qt_note_host),
 * and the chain copies a prompt chunk's non-resident experts from there. The chain's promotions
 * then swap experts between the card and the RAM: the one that left was never in the pool, and
 * a prompt chunk's layer holding it went to the host's step whole (q36cc_stage_table). Here a
 * warm start as the engine's, three promotions (one copy failed), then the follow: the pool is
 * the non-residents again, nothing else; a page-lock refused, whole or for the scales only,
 * leaves nothing locked; every page-lock undone at the exit. A per-row int4 container's slots
 * (pairs, gate, up and down in one block, a scale a row) join the pool as they are (#2734), a slot
 * whose matrices lie apart does not; why the pool is off is named. Fake CUDA backend (its
 * page-locks counted), no GPU, no toolkit. The tier gives the chain's KV mirrors memory back
 * (qt_chain_give_back): its coldest residents, or with their addresses the residents of its
 * coldest pages, whole. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"   /* setenv: MinGW has none */
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

/* planar int4 per group of 64, as the engine's slots on a Qwen3.6 container */
enum { NL = 2, NE = 8, TOPK = 2, D = 64, IH = 64, MB = D * IH / 2, NSC = 2 * IH * (D / 64) + D * (IH / 64) };
static unsigned char pw[NL][NE][3 * MB];
static float sc[NL][NE][NSC];

static int pooled(int l, int e) { const void *w; const float *s; size_t a, b; return qt_chain_host(l, e, &w, &a, &s, &b); }
static int resident(int l, int e) { int r; pthread_mutex_lock(&G.mx); r = qs(l, e)->resident; pthread_mutex_unlock(&G.mx); return r; }
/* the pool is exactly the non-residents */
static int pool_is_rest(void) {
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) if (pooled(l, e) == resident(l, e)) return 0;
    return 1;
}
/* a synthetic device address for each resident's matrices (gate, up, down of slot i back to back,
 * 512 KiB each: four to a 2 MiB page), as the chain's cc_tensor_dev gives the tier */
static uintptr_t slot_page(int i, int j) { return ((uintptr_t)0x10000000 + ((uintptr_t)i * 3 + (uintptr_t)j) * 0x80000) >> 21; }
static int page_addr(const struct ColiCudaTensor *t, uintptr_t *at) {
    for (int i = 0; i < NL * NE; i++) {
        const QSlot *x = &G.slot[i];
        const struct ColiCudaTensor *m3[3] = {x->tg, x->tu, x->td};
        for (int j = 0; j < 3; j++) if (t && m3[j] == t) { *at = (uintptr_t)0x10000000 + ((uintptr_t)i * 3 + (uintptr_t)j) * 0x80000; return 1; }
    }
    return 0;
}
static void note(int l, int e, int host) {
    const unsigned char *g = pw[l][e];
    if (host) qt_note_host(l, e, g, g + MB, g + 2 * MB, sc[l][e], sc[l][e] + IH, sc[l][e] + 2 * IH);
    else qt_note_block(l, e, g, g + MB, g + 2 * MB, sc[l][e], sc[l][e] + IH, sc[l][e] + 2 * IH);
}
static void step(int layer, int a, int b) {   /* a row's routes, the misses offered as the engine's miss path does */
    int ids[2] = {a, b};
    qt_chain_count(layer, ids, 2);
    for (int k = 0; k < 2; k++) {
        int e = ids[k];
        if (e >= 0 && !qt_is_resident(layer, e))
            qt_note(layer, e, pw[layer][e], pw[layer][e] + MB, pw[layer][e] + 2 * MB, sc[layer][e], sc[layer][e] + IH, sc[layer][e] + 2 * IH);
    }
}
/* a warm start as the engine's (tier_warmstart): every expert in RAM, eight on the card, the rest page-locked */
static void warm(void) {
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++) note(l, e, 0);
    qt_fill_wait();
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) if (!resident(l, e)) note(l, e, 1);
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1); setenv("COLI_PLACE", "off", 1);
    setenv("QT_UPLOAD_SYNC", "", 1);
    fake_ndev = 1; fake_host_reg_ok = 1; fake_planar_rows = 1;
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        for (size_t i = 0; i < sizeof pw[l][e]; i++) pw[l][e][i] = (unsigned char)((i * 37u + (unsigned)l * 13u + (unsigned)e * 7u) & 0xFFu);
        for (int i = 0; i < NSC; i++) sc[l][e][i] = 0.25f * (float)(i + e + 1);
    }
    const unsigned long long eb = 3 * MB + NSC * sizeof(float);   /* an expert's page-locked bytes */
    size_t exp_bytes = 3 * dev_alloc_footprint(MB) + 3 * dev_alloc_footprint((NSC / 3) * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(8 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);

    qt_source_planar(1);
    check(qt_init(NL, NE, D, IH, NE, TOPK, 64, 1), "tier starts (int4 gs=64, planar source)");
    warm();
    int chg[16], over, j, l;
    check(qt_chain_host_count() == 8 && fake_host_regs == 16 && G_host_bytes == 8 * eb,
          "warm start: the eight experts off the card page-locked (weights and scales), 16 registrations");
    check(pool_is_rest(), "warm start: the pool is the non-residents");
    check(qt_chain_host_follow(&j, &l) == 0 && j == 0 && l == 0 && fake_host_regs == 16, "nothing moved: the follow changes nothing");
    qt_chain_changes(chg, 16, &over);   /* the recording starts */

    /* the conversation: layer 0 routes 4 and 5, layer 1 routes 0 and 5, three times */
    for (int t = 0; t < 3; t++) { step(0, 4, 5); step(1, 0, 5); }
    QtPromo pr[8];
    int np = qt_chain_promote(8, pr);
    check(np == 3, "three promotions: layer 0's 4 and 5, layer 1's 5");
    if (np != 3) { printf("test_qwen36_tier_host_pool: %d failure(s)\n", fails); return 1; }
    qt_chain_promote_done(pr[0].ci, pr[0].vi, 1);
    qt_chain_promote_done(pr[1].ci, pr[1].vi, 1);
    qt_chain_promote_done(pr[2].ci, pr[2].vi, 0);   /* a copy that failed: the victim's buffers hold neither */
    int vl[3], ve[3];
    for (int k = 0; k < 3; k++) vl[k] = pr[k].vi / NE, ve[k] = pr[k].vi % NE;
    check(!resident(vl[0], ve[0]) && !resident(vl[1], ve[1]) && !resident(vl[2], ve[2]) &&
          !pooled(vl[0], ve[0]) && !pooled(vl[1], ve[1]) && !pooled(vl[2], ve[2]),
          "before the follow: the three victims are off the card and not in the pool (a prompt chunk's layer would go to the host)");
    check(resident(pr[0].ci / NE, pr[0].ci % NE) && pooled(pr[0].ci / NE, pr[0].ci % NE),
          "before the follow: a newcomer on the card keeps its RAM page-locked");
    qt_chain_changes(chg, 16, &over);   /* the promotions' changes, read */

    int regs = fake_host_regs, unregs = fake_host_unregs;
    check(qt_chain_host_follow(&j, &l) == 1 && j == 3 && l == 2, "the follow: three victims join the pool, the two newcomers on the card leave it");
    check(pool_is_rest(), "after the follow: the pool is the non-residents (the failed pair's newcomer stays in it)");
    check(qt_chain_host_count() == 9 && G_host_bytes == 9 * eb, "nine in the pool (8 - 2 + 3), its bytes counted");
    check(fake_host_regs == regs + 6 && fake_host_unregs == unregs + 4, "six registrations (three experts) and four releases (two)");
    check(qt_chain_changes(chg, 16, &over) == 0 && over == 0, "the follow moves no resident: no change for the chain's table");
    check(qt_chain_host_follow(&j, &l) == 0 && j == 0 && l == 0 && fake_host_regs == regs + 6, "a second follow changes nothing");

    /* the page-lock refused: nothing joins, nothing is left locked */
    for (int t = 0; t < 3; t++) step(0, 6, 7);
    np = qt_chain_promote(8, pr);
    check(np >= 1, "another promotion");
    for (int k = 0; k < np; k++) qt_chain_promote_done(pr[k].ci, pr[k].vi, 1);
    fake_host_reg_ok = 0;
    regs = fake_host_regs; unregs = fake_host_unregs;
    qt_chain_host_follow(&j, &l);
    check(j == 0 && l == np && fake_host_regs == regs && fake_host_unregs == unregs + 2 * np,
          "the page-lock refused: no victim joins, the newcomers still leave");
    check(!pooled(pr[0].vi / NE, pr[0].vi % NE) && qt_chain_host_count() == 9 - np, "the victim stays out, the count says so");
    fake_host_reg_ok = 1;
    check(qt_chain_host_follow(&j, &l) == 1 && j == np && l == 0 && pool_is_rest(), "the next follow takes them");

    /* the weights locked, the scales refused: the weights are released, two page-locks an expert in the pool */
    for (int t = 0; t < 3; t++) step(1, 6, 7);
    np = qt_chain_promote(8, pr);
    check(np >= 1, "a promotion in the other layer");
    for (int k = 0; k < np; k++) qt_chain_promote_done(pr[k].ci, pr[k].vi, 1);
    fake_host_reg_bytes = 3 * MB;
    qt_chain_host_follow(&j, &l);
    check(j == 0 && l == np && !pooled(pr[0].vi / NE, pr[0].vi % NE), "the scales refused: no victim joins, the newcomers leave");
    check(fake_host_regs - fake_host_unregs == 2 * qt_chain_host_count(), "the scales refused: no weights left page-locked");
    fake_host_reg_bytes = 0;
    check(qt_chain_host_follow(&j, &l) == 1 && j == np && l == 0 && pool_is_rest() &&
          fake_host_regs - fake_host_unregs == 2 * qt_chain_host_count(), "the next follow takes them");

    qt_shutdown();
    check(fake_host_unregs == fake_host_regs, "the exit releases every page-lock the pool took");

    /* COLI_CUDA_HOST_POOL=0: no pool, and the follow leaves it so */
    setenv("COLI_CUDA_HOST_POOL", "0", 1);
    qt_source_planar(1);
    check(qt_init(NL, NE, D, IH, NE, TOPK, 64, 1), "tier starts again, the pool off");
    regs = fake_host_regs;
    warm();
    for (int t = 0; t < 3; t++) step(0, 4, 5);
    np = qt_chain_promote(8, pr);
    for (int k = 0; k < np; k++) qt_chain_promote_done(pr[k].ci, pr[k].vi, 1);
    check(qt_chain_host_follow(&j, &l) == 0 && j == 0 && l == 0 && qt_chain_host_count() == 0 && fake_host_regs == regs,
          "COLI_CUDA_HOST_POOL=0: nothing page-locked, before or after");
    check(qt_chain_host_off() && !strcmp(qt_chain_host_off(), "COLI_CUDA_HOST_POOL=0"), "COLI_CUDA_HOST_POOL=0: the startup names it");
    qt_source_planar(0);
    qt_shutdown();
    setenv("COLI_CUDA_HOST_POOL", "", 1);

    /* a per-row int4 container (#2734): the slot's pairs as the residents' tensors hold them, gate, up and
     * down in one block, a scale a row (IH, IH, D); the pool takes them as they are, not planar */
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts (int4 per row, pairs)");
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++) note(l, e, 0);
    qt_fill_wait();
    regs = fake_host_regs;
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++)
        if (!resident(l, e)) { const unsigned char *g = pw[l][e]; qt_note_host(l, e, g, g + MB, g + 2 * MB, sc[l][e], sc[l][e] + IH, sc[l][e] + 2 * IH); }
    {
        const void *w; const float *s; size_t a = 0, b = 0; int lo = -1, eo = -1;
        for (int l = 0; l < NL && lo < 0; l++) for (int e = 0; e < NE; e++) if (!resident(l, e)) { lo = l; eo = e; break; }
        check(qt_chain_host_count() == 8 && fake_host_regs == regs + 16 && pool_is_rest(),
              "per row: the eight experts off the card page-locked, the pool the non-residents");
        check(lo >= 0 && qt_chain_host(lo, eo, &w, &a, &s, &b) && w == pw[lo][eo] && a == 3 * MB && s == sc[lo][eo] &&
              b == (2 * IH + D) * sizeof(float), "per row: an expert's block and its row scales (2 IH + D), as the slot holds them");
        check(!qt_chain_host_planar() && !qt_chain_host_off(), "per row: the pool is on, its int4 in pairs");
    }
    qt_shutdown();
    check(fake_host_unregs == fake_host_regs, "per row: the exit releases every page-lock");

    /* the three matrices apart (the slot's old layout): each such expert is refused */
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts again (per row)");
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++) note(l, e, 0);
    qt_fill_wait();
    regs = fake_host_regs;
    static unsigned char apart[3][MB];   /* up after down: not the block the pool copies */
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++)
        if (!resident(l, e)) qt_note_host(l, e, apart[0], apart[2], apart[1], sc[l][e], sc[l][e] + IH, sc[l][e] + 2 * IH);
    check(qt_chain_host_count() == 0 && fake_host_regs == regs, "gate, up and down in three allocations: nothing page-locked");
    qt_shutdown();

    /* memory back for the chain's KV mirrors (qt_chain_give_back, #2734): the coldest residents
     * leave the card, their tensors freed, the budget down by as much; the CPU's rebuild list has
     * them, the chain sees the move, and the next follow takes them into the pool */
    qt_source_planar(1);
    check(qt_init(NL, NE, D, IH, NE, TOPK, 64, 1), "tier starts (planar), for the give-back");
    warm();
    {
        int hot[2], nh = 0;
        for (int e = 0; e < NE && nh < 2; e++) if (resident(0, e)) hot[nh++] = e;
        for (int t = 0; t < 4; t++) step(0, hot[0], hot[1]);   /* this conversation's routes */
        int frees = fake_tensor_frees, res0 = 0, res1 = 0;
        unsigned ep = qt_chain_epoch();
        size_t bud = G.budget[0], used = G.used[0], eb = G.exp_bytes;
        for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) res0 += resident(l, e);
        size_t got = qt_chain_give_back(2 * eb - 1, NULL);
        for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) res1 += resident(l, e);
        check(got == 2 * eb && res1 == res0 - 2 && fake_tensor_frees == frees + 6,
              "give back two experts' bytes less one: two residents leave, their three tensors freed");
        check(resident(0, hot[0]) && resident(0, hot[1]), "the conversation's experts stay on the card");
        check(G.used[0] == used - 2 * eb && G.budget[0] == bud - 2 * eb, "used and budget down by what left");
        check(qt_chain_epoch() != ep, "the chain sees the residents moved");
        int el[8], ee[8];
        check(qt_evicted_take(el, ee, 8) == 2 && !resident(el[0], ee[0]) && !resident(el[1], ee[1]),
              "the CPU's rebuild list has the two");
        check(!pooled(el[0], ee[0]) && qt_chain_host_follow(&j, &l) == 1 && j == 2 && pool_is_rest(),
              "the next follow takes them into the host pool");
        check(qt_chain_give_back(0, NULL) == 0 && fake_tensor_frees == frees + 6, "nothing asked, nothing back");
    }
    qt_shutdown();

    /* by page (an address for each matrix): the driver packs the 512 KiB matrices four to a 2 MiB
     * page and releases a page only when it is empty, so the residents go by page: every resident
     * with a matrix in the picked page, and no other; the conversation's experts stay */
    check(qt_init(NL, NE, D, IH, NE, TOPK, 64, 1), "tier starts (planar), for the give-back by page");
    warm();
    {
        int hot[2], nh = 0;
        for (int e = 0; e < NE && nh < 2; e++) if (resident(1, e)) hot[nh++] = e;
        for (int t = 0; t < 4; t++) step(1, hot[0], hot[1]);
        int frees = fake_tensor_frees, res0 = 0, res1 = 0;
        for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) res0 += resident(l, e);
        uint8_t was[NL * NE];
        for (int i = 0; i < NL * NE; i++) was[i] = (uint8_t)resident(i / NE, i % NE);
        size_t got = qt_chain_give_back(1, page_addr);
        for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) res1 += resident(l, e);
        int out = res0 - res1;
        uintptr_t pgs[3 * NL * NE]; int npg = 0, ok_page = out > 0;
        for (int i = 0; i < NL * NE; i++) if (was[i] && !resident(i / NE, i % NE))
            for (int j = 0; j < 3; j++) pgs[npg++] = slot_page(i, j);
        /* one page every resident that left has a matrix in */
        uintptr_t common = 0; int found = 0;
        for (int a = 0; a < npg && !found; a++) {
            int all = 1;
            for (int i = 0; i < NL * NE && all; i++) if (was[i] && !resident(i / NE, i % NE))
                all = slot_page(i, 0) == pgs[a] || slot_page(i, 1) == pgs[a] || slot_page(i, 2) == pgs[a];
            if (all) { common = pgs[a]; found = 1; }
        }
        ok_page = ok_page && found;
        for (int i = 0; i < NL * NE && ok_page; i++)   /* nothing left on the card with a matrix there */
            if (resident(i / NE, i % NE)) for (int j = 0; j < 3; j++) ok_page = ok_page && slot_page(i, j) != common;
        check(ok_page && got == (size_t)out * G.exp_bytes && fake_tensor_frees == frees + 3 * out,
              "by page: the residents of one page leave, all of them and no other, their tensors freed");
        check(resident(1, hot[0]) && resident(1, hot[1]), "by page: the conversation's experts stay on the card");
    }
    qt_shutdown();
    qt_source_planar(0);

    /* a planar container's pool says so; the RAM short of every expert turns it off, by name */
    qt_source_planar(1);
    check(qt_init(NL, NE, D, IH, NE, TOPK, 64, 1), "tier starts (planar)");
    warm();
    check(qt_chain_host_count() == 8 && qt_chain_host_planar() && !qt_chain_host_off(), "planar: the pool is on, its int4 planar");
    qt_shutdown();
    check(qt_init(NL, NE, D, IH, NE / 2, TOPK, 64, 1), "tier starts with the RAM holding half of a layer's experts");
    check(qt_chain_host_off() && !strcmp(qt_chain_host_off(), "the RAM holds fewer than every expert"), "the RAM short: the startup names it");
    qt_source_planar(0);
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_host_pool: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_tier_host_pool: after promotions the host pool is the non-residents again, every page-lock released at the exit; a per-row container's pairs join it; memory back from the coldest residents, or by page\n");
    return 0;
}
