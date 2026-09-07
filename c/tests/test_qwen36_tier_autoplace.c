/* Automatic placement of the dense trunk (COLI_PLACE unset or "auto").
 *
 * The rule under test, stated once: a trunk component goes into VRAM when the
 * bytes it saves on the memory bus per token beat the bytes the experts it
 * displaces would have saved -- dense weights are read every token, an expert
 * with the probability a token routes to it, and the CPU reads an int4 expert
 * as an int8 slot (twice the VRAM bytes). Everything else here is that rule
 * applied: no heat means the trunk wins, hot marginal experts mean it loses,
 * an explicit list is obeyed, "off" places nothing, and whatever lands in
 * VRAM comes out of that device's expert budget, on one card and on two.
 *
 * Fake backend, no GPU: free VRAM is whatever the test says it is. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { D = 64, IH = 32 };
#define MiB (1024ull * 1024ull)

/* every case starts from a clean tier: no offers, no auto decision, env set */
static void fresh(const char *place, const char *gpus, int ndev, const char *budget_gb) {
    G_offer_n = 0; G_auto_on = 0;
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", gpus, 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_PLACE", place, 1);                 /* "" == unset == auto */
    setenv("CUDA_EXPERT_GB", budget_gb, 1);        /* the per-device allowance */
    setenv("HEAT_FILE", "", 1);
    fake_ndev = ndev;
}

static int start(int nl, int ne, int topk) {
    return qt_init(nl, ne, D, IH, ne, topk, 0 /* per-row */, 1 /* int4 */);
}

/* HEAT_FILE with every expert at the same heat: p_e = topk / n_experts */
static const char *write_uniform_heat(int nl, int ne) {
    static char path[256];
    snprintf(path, sizeof path, "qwen36_autoplace_heat_%d_%d.bin", nl, ne);
    FILE *f = fopen(path, "wb");
    uint32_t hdr[3] = { 0x51544831u, (uint32_t)nl, (uint32_t)ne };
    fwrite(hdr, 4, 3, f);
    for (int i = 0; i < nl * ne; i++) { uint32_t h = 1000; fwrite(&h, 4, 1, f); }
    fclose(f);
    return path;
}

int main(void) {
    printf("qwen36 tier automatic placement\n");

    /* ---- 1. auto, one device: everything offered fits, budget shrinks by it ---- */
    printf(" 1. auto, one device, trunk fits\n");
    fresh("", "0", 1, "0.0625");                          /* 64 MiB allowance */
    qt_trunk_offer("lmhead", 0, 10 * MiB);
    for (int l = 0; l < 8; l++) if (l % 4 != 3) qt_trunk_offer("dnproj", l, 2 * MiB);
    check(start(8, 16, 4), "tier starts in auto mode");
    check(qt_place_of("lmhead", 0) == 0, "lmhead lands on device 0");
    check(qt_place_of("dnproj", 0) == 0 && qt_place_of("dnproj", 6) == 0, "offered dnproj layers land on device 0");
    check(qt_place_of("dnproj", 3) == QT_PLACE_CPU, "an attention layer (never offered) stays on the CPU");
    check(qt_place_of("attnproj", 0) == QT_PLACE_CPU, "components not yet placed by auto stay on the CPU");
    check(G.ndev == 1 && G.on, "one card keeps experts and trunk together (no role-split reservation in auto)");
    check(G.budget[0] == 64 * MiB - 22 * MiB, "expert budget = allowance minus the 22 MiB of trunk placed");
    check(G_lmh.dev_ok && G_lmh.dev == 0, "lm_head device is armed for qt_lmhead_init");
    qt_shutdown();

    /* ---- 2. auto: a trunk item larger than the allowance stays on the CPU ---- */
    printf(" 2. auto, trunk does not fit\n");
    fresh("", "0", 1, "0.0625");
    qt_trunk_offer("lmhead", 0, 100 * MiB);
    qt_trunk_offer("dnproj", 0, 2 * MiB);
    check(start(8, 16, 4), "tier starts");
    check(qt_place_of("lmhead", 0) == QT_PLACE_CPU, "100 MiB lmhead cannot fit 64 MiB: CPU");
    check(qt_place_of("dnproj", 0) == 0, "the 2 MiB projection still fits and is placed");
    check(G.budget[0] == 64 * MiB - 2 * MiB, "budget shrinks only by what was placed");
    qt_shutdown();

    /* ---- 3. heat decides: hot marginal experts keep their VRAM ---- */
    printf(" 3. heat: hot marginal experts beat the trunk, cold ones do not\n");
    {
        /* nl=1, ne=48. Allowance for exactly 16 experts, so 32 of 48 do not fit
         * and the trunk would displace real residents. Offer one expert-sized
         * projection (k=1). Uniform heat -> p_e = topk/48. */
        enum { NL = 1, NE = 48 };
        const char *heat = write_uniform_heat(NL, NE);
        char gb[64];
        /* exp_bytes for D=64, IH=32, per-row int4, at cudaMalloc granularity:
         * three 1 KiB matrices charged 1 MiB each, three scale tables 10 KiB each */
        size_t exp_bytes = 3 * dev_alloc_footprint((size_t)D * IH / 2)
                         + 3 * dev_alloc_footprint((2 * IH + D) / 3 * sizeof(float));
        snprintf(gb, sizeof gb, "%.15f", (double)(16 * exp_bytes + exp_bytes / 2) / 1073741824.0);
        /* the allowance as the tier will parse it back from the string */
        size_t allowance = (size_t)(atof(gb) * 1024.0 * 1024.0 * 1024.0);

        /* topk 32: p = 0.667, CPU factor 2 -> 1.33 bytes/token per VRAM byte > 1.0: trunk loses */
        fresh("", "0", 1, gb); setenv("HEAT_FILE", heat, 1);
        qt_trunk_offer("dnproj", 0, exp_bytes);
        check(start(NL, NE, 32), "tier starts with a heat file");
        check(G.heat0 != NULL, "heat table loaded");
        check(qt_place_of("dnproj", 0) == QT_PLACE_CPU,
              "with p_marginal 0.67 the displaced expert is worth 1.33 bytes/token per byte: trunk stays on CPU");
        check(G.budget[0] == allowance, "nothing placed, budget untouched");
        qt_shutdown();

        /* topk 8: p = 0.167 -> 0.33 < 1.0: trunk wins */
        fresh("", "0", 1, gb); setenv("HEAT_FILE", heat, 1);
        qt_trunk_offer("dnproj", 0, exp_bytes);
        check(start(NL, NE, 8), "tier starts");
        check(qt_place_of("dnproj", 0) == 0, "with p_marginal 0.17 the trunk item is worth more: placed");
        check(G.budget[0] == allowance - exp_bytes, "budget shrinks by the placed item");
        qt_shutdown();

        /* no heat, topk 32: uniform assumption p = 32/48 as well -- same answer as with the file */
        fresh("", "0", 1, gb);
        qt_trunk_offer("dnproj", 0, exp_bytes);
        check(start(NL, NE, 32), "tier starts without heat");
        check(qt_place_of("dnproj", 0) == QT_PLACE_CPU, "without a heat file the uniform estimate gives the same verdict");
        qt_shutdown();

        /* room to spare: allowance for 100 experts, only 48 exist -> displaces nothing, always placed */
        snprintf(gb, sizeof gb, "%.15f", (double)(100 * exp_bytes) / 1073741824.0);
        fresh("", "0", 1, gb); setenv("HEAT_FILE", heat, 1);
        qt_trunk_offer("dnproj", 0, exp_bytes);
        check(start(NL, NE, 32), "tier starts");
        check(qt_place_of("dnproj", 0) == 0, "VRAM nobody would use goes to the trunk regardless of heat");
        qt_shutdown();
        remove(heat);
    }

    /* ---- 4. an explicit list wins over auto, and still pays for its trunk ---- */
    printf(" 4. explicit COLI_PLACE\n");
    fresh("lmhead=cpu,dnproj=0", "0", 1, "0.0625");
    qt_trunk_offer("lmhead", 0, 10 * MiB);
    for (int l = 0; l < 8; l++) if (l % 4 != 3) qt_trunk_offer("dnproj", l, 2 * MiB);
    check(start(8, 16, 4), "tier starts with an explicit list");
    check(!G_auto_on, "auto is off when a list is given");
    check(qt_place_of("lmhead", 0) == QT_PLACE_CPU, "lmhead=cpu is obeyed even though it would fit");
    check(qt_place_of("dnproj", 2) == 0, "dnproj=0 is obeyed");
    check(G.budget[0] == 64 * MiB - 12 * MiB, "the explicit list's 12 MiB of projections come out of the expert budget too");
    qt_shutdown();

    /* ---- 5. off: nothing placed, full budget ---- */
    printf(" 5. COLI_PLACE=off\n");
    fresh("off", "0", 1, "0.0625");
    qt_trunk_offer("lmhead", 0, 10 * MiB);
    qt_trunk_offer("dnproj", 0, 2 * MiB);
    check(start(8, 16, 4), "tier starts with placement off");
    check(qt_place_of("lmhead", 0) == QT_PLACE_CPU && qt_place_of("dnproj", 0) == QT_PLACE_CPU, "off places nothing");
    check(G.budget[0] == 64 * MiB, "off leaves the whole allowance to the experts");
    qt_shutdown();

    /* ---- 6. auto, two devices: greedy by room, both keep experts ---- */
    printf(" 6. auto, two devices\n");
    fresh("", "0,1", 2, "0.0625");
    qt_trunk_offer("lmhead", 0, 10 * MiB);
    for (int l = 0; l < 8; l++) if (l % 4 != 3) qt_trunk_offer("dnproj", l, 9 * MiB);
    check(start(8, 16, 4), "tier starts on two fake devices");
    check(G.ndev == 2, "auto keeps experts on both devices");
    check(qt_place_of("lmhead", 0) == 0, "lmhead goes to the first device (both equal, first wins)");
    /* greedy by most room, 9 MiB each, after lmhead dev0=54 dev1=64:
       l0 -> dev1 (55), l1 -> dev1 (46), l2 -> dev0 (45), l4 -> dev1 (37),
       l5 -> dev0 (36), l6 -> dev1 (27): dev0 holds lmhead+2 = 28 MiB, dev1 4 = 36 MiB */
    check(qt_place_of("dnproj", 0) == 1 && qt_place_of("dnproj", 1) == 1 && qt_place_of("dnproj", 2) == 0 &&
          qt_place_of("dnproj", 4) == 1 && qt_place_of("dnproj", 5) == 0 && qt_place_of("dnproj", 6) == 1,
          "projections go to whichever device has the most room left");
    check(G.budget[0] == 64 * MiB - 28 * MiB && G.budget[1] == 64 * MiB - 36 * MiB,
          "each device's expert budget is its allowance minus its own trunk");
    qt_shutdown();

    /* ---- 7. nothing offered: auto is a no-op, budgets are the allowance ---- */
    printf(" 7. auto with nothing offered\n");
    fresh("", "0", 1, "0.0625");
    check(start(8, 16, 4), "tier starts");
    check(qt_place_of("lmhead", 0) == QT_PLACE_CPU && G.budget[0] == 64 * MiB, "no offers: nothing placed, full budget");
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_autoplace: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_autoplace: ok\n");
    return 0;
}
