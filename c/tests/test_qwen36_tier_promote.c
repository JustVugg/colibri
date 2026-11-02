/* qt_chain_promote: the resident set follows the conversation while the CUDA chain runs
 * the routed experts in its frame.
 *
 * With the experts in the frame the tier sees routes only through qt_chain_count. The
 * warm start's set comes from the history; a conversation on another topic misses it.
 * After a decode step the chain asks for promotions: the step's misses routed at least
 * twice in this conversation, the worth the most first (the heat, which carries the
 * history and this conversation, and this conversation's routes once more), each against
 * the coldest of the residents drawn, where it clears the tier's admission margin. The
 * engine's miss path offers each miss to the tier (qt_note: its heat), as here. The tier hands
 * over the pair -- the newcomer's RAM bytes, the victim's device tensors -- and marks both
 * queued; the chain copies on its own stream and reports qt_chain_promote_done. The
 * residency changes reach the chain's table through qt_chain_changes. Fake CUDA backend,
 * no GPU, no toolkit. */
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
static int in_set(const int *a, int n, int v) { for (int i = 0; i < n; i++) if (a[i] == v) return 1; return 0; }
static unsigned char (*G4)[NE][64 * 32 / 2], (*U4)[NE][64 * 32 / 2], (*D4)[NE][64 * 32 / 2]; static float (*SC)[NE][2 * 32 + 64];
static void step(int layer, int a, int b) {   /* a row's routes: the books, and the misses offered as the engine's miss path does */
    int ids[2] = {a, b};
    qt_chain_count(layer, ids, 2);
    for (int k = 0; k < 2; k++) if (ids[k] >= 0 && !qt_is_resident(layer, ids[k]))
        qt_note(layer, ids[k], G4[layer][ids[k]], U4[layer][ids[k]], D4[layer][ids[k]], SC[layer][ids[k]], SC[layer][ids[k]] + IH, SC[layer][ids[k]] + 2 * IH);
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
    G4 = g4; U4 = u4; D4 = d4; SC = sc;
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++) {
        memset(g4[l][e], (unsigned char)(e + 1), sizeof g4[l][e]); memset(u4[l][e], (unsigned char)(e + 2), sizeof u4[l][e]);
        memset(d4[l][e], (unsigned char)(e + 3), sizeof d4[l][e]);
        for (int i = 0; i < 2 * IH + D; i++) sc[l][e][i] = 1.0f;
        qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sc[l][e], sc[l][e] + IH, sc[l][e] + 2 * IH);
    }
    qt_fill_wait();
    int res[NE]; int n = resident_set(0, res);
    check(n == 4 && res[0] == 0 && res[3] == 3, "warmstart: layer 0 experts 0..3 resident, 4..7 not");

    int chg[16], over;
    check(qt_chain_changes(chg, 16, &over) == 0 && over == 1, "the first read of the changes says: read every entry (the recording starts now)");
    check(qt_chain_changes(chg, 16, &over) == 0 && over == 0, "nothing moved since");

    /* a step of the conversation: layer 0 routes to 4 and 5 (misses) and 0 (a hit); once is not enough */
    step(0, 4, 5); step(1, 0, 5);
    QtPromo pr[8];
    check(qt_chain_promote(8, pr) == 0, "a miss routed once is not promoted (COLI_CUDA_CHAIN_PROMOTE_MIN 2)");
    n = resident_set(0, res);
    check(n == 4 && res[0] == 0 && res[3] == 3, "and nothing moved");
    /* twice: two routes, but a score of 4 (heat 2 and 2 routes) does not clear a never-routed resident's 1 by the margin (a quarter and 4) */
    step(0, 4, 5); step(1, 0, 5);
    check(qt_chain_promote(8, pr) == 0, "two routes do not clear the admission margin over a resident the history holds");

    /* the third step: layer 0's 4 and 5 and layer 1's 5 score 6 against the never-routed residents' 1 */
    step(0, 4, 5); step(1, 0, 5);
    int np = qt_chain_promote(8, pr);
    check(np == 3, "three promotions: layer 0's 4 and 5, layer 1's 5");
    int ok_pairs = np == 3;
    for (int k = 0; k < np && ok_pairs; k++) {
        int cl = pr[k].ci / NE, ce = pr[k].ci % NE, vl = pr[k].vi / NE, ve = pr[k].vi % NE;
        if (ve < 0 || ve > 3 || (cl == 0 && ce != 4 && ce != 5) || (cl == 1 && ce != 5) || (vl == 1 && ve == 0)) ok_pairs = 0;   /* any layer's cold resident may give way */
        if (pr[k].g != g4[cl][ce] || pr[k].gs != sc[cl][ce] || pr[k].mb != (size_t)D * IH / 2 || pr[k].sb_gu != IH * sizeof(float) || pr[k].sb_d != D * sizeof(float)) ok_pairs = 0;
        pthread_mutex_lock(&G.mx);
        if (!pr[k].tg || pr[k].tg != qs(vl, ve)->tg || !qs(cl, ce)->queued || !qs(vl, ve)->queued || qs(vl, ve)->resident) ok_pairs = 0;
        pthread_mutex_unlock(&G.mx);
    }
    check(ok_pairs, "each pair: a victim among the never-routed residents of either layer (layer 1 keeps its routed 0), the newcomer's RAM "
                    "bytes and sizes, the victim's tensors, both queued, the victim no longer resident");
    check(pr[0].vi != pr[1].vi && pr[1].vi != pr[2].vi && pr[0].vi != pr[2].vi, "three newcomers take three victims");
    check(resident_set(0, res) + resident_set(1, res) == 5, "five residents left until the copies land");
    int nc = qt_chain_changes(chg, 16, &over);
    check(nc == 3 && over == 0 && in_set(chg, nc, pr[0].vi) && in_set(chg, nc, pr[1].vi) && in_set(chg, nc, pr[2].vi), "the three evictions reported to the chain");
    int ls[8], es[8];
    check(qt_evicted_take(ls, es, 8) == 3, "and to the engine, for their int8 copies");
    check(qt_chain_promote(8, pr + 4) == 0, "nothing else to promote: the step's misses were taken");

    /* the copies landed: the newcomers take the victims' tensors */
    ColiCudaTensor *t0 = pr[0].tg;
    qt_chain_promote_done(pr[0].ci, pr[0].vi, 1);
    qt_chain_promote_done(pr[1].ci, pr[1].vi, 1);
    n = resident_set(0, res);
    check(in_set(res, n, 4) && in_set(res, n, 5) && n + resident_set(1, res) == 7, "layer 0's 4 and 5 resident, seven residents in all");
    pthread_mutex_lock(&G.mx);
    check(qs(pr[0].ci / NE, pr[0].ci % NE)->tg == t0 && qs(pr[0].vi / NE, pr[0].vi % NE)->tg == NULL &&
          !qs(pr[0].ci / NE, pr[0].ci % NE)->queued && !qs(pr[0].vi / NE, pr[0].vi % NE)->queued, "the newcomer holds the victim's tensors, neither is queued");
    pthread_mutex_unlock(&G.mx);
    ColiCudaTensor *g, *u, *d;
    check(qt_chain_resident(0, 4, 0, &g, &u, &d) && g && qt_chain_resident(0, 5, 0, &g, &u, &d), "the chain finds them resident");
    check(!qt_chain_resident(pr[0].vi / NE, pr[0].vi % NE, 0, &g, &u, &d), "and the victim not");
    nc = qt_chain_changes(chg, 16, &over);
    check(nc == 2 && over == 0 && in_set(chg, nc, pr[0].ci) && in_set(chg, nc, pr[1].ci), "the two arrivals reported");
    check(qt_chain_promoted() == 2 && G.swaps == 3 && G.inplace == 2, "two landed so far, three swaps counted");

    /* a copy that failed: the victim's buffers hold neither expert; the newcomer stays out */
    qt_chain_promote_done(pr[2].ci, pr[2].vi, 0);
    n = resident_set(1, res);
    check(!in_set(res, n, 5) && n + resident_set(0, res) == 7, "5 of layer 1 not among the seven residents");
    pthread_mutex_lock(&G.mx);
    check(qs(1, 5)->tg == NULL && !qs(1, 5)->queued && qs(pr[2].vi / NE, pr[2].vi % NE)->tg == NULL && !qs(pr[2].vi / NE, pr[2].vi % NE)->queued,
          "neither holds tensors, neither is queued");
    pthread_mutex_unlock(&G.mx);
    check(qt_chain_promoted() == 2, "a failed copy is not a landed promotion");

    /* a newcomer is not worth a swap against a resident the conversation routes as often */
    for (int t = 0; t < 3; t++) step(0, 4, 6);   /* 6 three times (score 6); 4 (resident, routed) scores 12 by now */
    np = qt_chain_promote(8, pr);
    check(np == 1 && pr[0].ci == 6, "6 (three routes) is promoted against a never-routed resident");
    check(pr[0].vi != 4 && pr[0].vi != 5 && pr[0].vi != NE + 0 && pr[0].vi != NE + 5, "4, 5 and layer 1's 0 and 5 (routed in this conversation) are not its victim");
    qt_chain_promote_done(pr[0].ci, pr[0].vi, 1);
    for (int t = 0; t < 3; t++) step(0, 7, -1);
    np = qt_chain_promote(8, pr);
    check(np == 1 && pr[0].ci == 7, "7 (three routes) takes another never-routed resident");
    qt_chain_promote_done(pr[0].ci, pr[0].vi, 1);
    n = resident_set(0, res);
    check(in_set(res, n, 4) && in_set(res, n, 5) && in_set(res, n, 6) && in_set(res, n, 7) && n + resident_set(1, res) == 7, "layer 0 holds 4..7 now, seven residents in all");
    for (int t = 0; t < 3; t++) { step(0, 0, 1); step(0, 2, 3); }   /* 0..3: the two out of each pair get three routes, a score of 6 */
    np = qt_chain_promote(8, pr);
    check(np == 0, "0..3 with three routes each do not displace 4..7 (scores 6 and more, routed as often or more: the margin holds), nor layer 1's residents");

    /* the conversation's counts halve every QT_CONV_DECAY ticks (a tick a layer-0 row) */
    uint32_t c4; pthread_mutex_lock(&G.mx); c4 = qs(0, 4)->conv; pthread_mutex_unlock(&G.mx);
    { int ids[1] = {-1}; for (int t = 0; t < QT_CONV_DECAY; t++) qt_chain_count(0, ids, 1); }
    uint32_t c4b; pthread_mutex_lock(&G.mx); c4b = qs(0, 4)->conv; pthread_mutex_unlock(&G.mx);
    check(c4 > 0 && c4b < c4, "the counts decayed");

    /* no rate, or the chain's brake: nothing moves, and the step's misses are dropped (the list is a step's) */
    for (int t = 0; t < 3; t++) step(0, 0, 1);   /* 0 and 1 miss now (4..7 hold layer 0), three routes each */
    check(qt_chain_promote(0, pr) == 0 && qt_chain_promote(8, NULL) == 0, "no rate, no out: nothing");
    check(qt_chain_promote(8, pr) == 0, "the braked step's misses were dropped: the next step begins empty");
    step(0, 0, 1);
    check(qt_chain_promote(8, pr) == 1 && pr[0].ci == 0, "the step after routes 0 again: 0 (four routes, a score of 8) takes the coldest of 4..7");
    {
        unsigned char stg[3 * (D * IH / 2) + (2 * IH + D) * sizeof(float)];
        qt_chain_promote_stage(stg, &pr[0]);
        check(memcmp(stg, g4[0][0], sizeof g4[0][0]) == 0 &&
              memcmp(stg + sizeof g4[0][0], u4[0][0], sizeof u4[0][0]) == 0 &&
              memcmp(stg + 2 * sizeof g4[0][0], d4[0][0], sizeof d4[0][0]) == 0 &&
              memcmp(stg + 3 * sizeof g4[0][0], sc[0][0], sizeof sc[0][0]) == 0,
              "pair-ordered int4: qt_chain_promote_stage copies weights and scales as they are");
    }
    qt_chain_promote_done(pr[0].ci, pr[0].vi, 1);

    qt_shutdown();

    /* planar int4 per group of 64 (qt_source_planar(1)): promoted and converted to signed pairs */
    enum { D64 = 64, IH64 = 64, MB64 = D64 * IH64 / 2, NSC64 = 2 * IH64 * (D64 / 64) + D64 * (IH64 / 64) };
    exp_bytes = 3 * dev_alloc_footprint(MB64) + 3 * dev_alloc_footprint((NSC64 / 3) * sizeof(float));
    snprintf(gb, sizeof gb, "%.15f", (double)(8 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    qt_source_planar(1);
    check(qt_init(NL, NE, D64, IH64, NE, TOPK, 64, 1), "tier starts (int4 gs=64, planar source)");
    static unsigned char raw[NL][NE][3 * MB64], pw[NL][NE][3 * MB64];
    static float sc64[NL][NE][NSC64];
    for (int e = 0; e < NE; e++) for (int l = 0; l < NL; l++) {
        for (size_t i = 0; i < sizeof raw[l][e]; i++)
            raw[l][e][i] = (unsigned char)((i * 37u + (unsigned)l * 13u + (unsigned)e * 7u) & 0xFFu);
        for (size_t b = 0; b < sizeof raw[l][e]; b += 32) {
            const unsigned char *s = raw[l][e] + b;
            unsigned char *d = pw[l][e] + b;
            for (int k = 0; k < 32; k++) {
                unsigned lo = (unsigned)(s[k >> 1] >> ((k & 1) * 4)) & 0xFu;
                unsigned hi = (unsigned)(s[16 + (k >> 1)] >> ((k & 1) * 4)) & 0xFu;
                d[k] = (unsigned char)((lo ^ 8u) | ((hi ^ 8u) << 4));
            }
        }
        for (int i = 0; i < NSC64; i++) sc64[l][e][i] = 0.25f * (float)(i + e + 1);
        qt_note_block(l, e, pw[l][e], pw[l][e] + MB64, pw[l][e] + 2 * MB64,
                      sc64[l][e], sc64[l][e] + IH64, sc64[l][e] + 2 * IH64);
    }
    qt_fill_wait();
    for (int t = 0; t < 3; t++) {
        int ids[2] = {4, -1};
        qt_chain_count(0, ids, 2);
        qt_note(0, 4, pw[0][4], pw[0][4] + MB64, pw[0][4] + 2 * MB64,
                sc64[0][4], sc64[0][4] + IH64, sc64[0][4] + 2 * IH64);
    }
    np = qt_chain_promote(8, pr);
    check(np == 1 && pr[0].ci == 4 && pr[0].mb == MB64, "planar int4 gs=64: expert 4 promoted");
    if (np == 1) {
        unsigned char stg[3 * MB64 + NSC64 * sizeof(float)];
        qt_chain_promote_stage(stg, &pr[0]);
        check(memcmp(stg, raw[0][4], sizeof raw[0][4]) == 0 &&
              memcmp(stg + sizeof raw[0][4], sc64[0][4], sizeof sc64[0][4]) == 0,
              "planar int4 gs=64: qt_chain_promote_stage converts planar blocks back to exact signed pairs and copies scales");
        qt_chain_promote_done(pr[0].ci, pr[0].vi, 1);
    }
    qt_source_planar(0);
    qt_shutdown();
    if (fails) { printf("test_qwen36_tier_promote: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_tier_promote: the chain's promotions follow the conversation's routes; the pairs, the changes and the counts are right\n");
    return 0;
}
