/* The automatic trunk placement is a prediction the engine measures at startup
 * (#1652: on four Tesla M10 every placed component ran slower than the CPU).
 * When the probe says the GPU loses, qt_trunk_withdraw() must put every offer
 * back on the CPU, lm_head included, and return the bytes the trunk had taken
 * to the expert budget, so the warmstart that follows fills that VRAM with
 * experts instead. A hand-written COLI_PLACE is the user's word: not auto, not
 * withdrawn. Fake CUDA backend, no GPU, no toolkit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"   /* setenv: MinGW has none */
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

enum { NL = 2, NE = 8, D = 64, IH = 32, TOPK = 2 };

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1);
    setenv("COLI_PLACE", "", 1);                    /* "" == unset == auto */
    fake_ndev = 1; fake_uploads = 0;

    /* room for 8 experts plus a little; each offer is 1 expert worth of bytes */
    size_t exp_bytes = 3 * dev_alloc_footprint((size_t)D * IH / 2) + 3 * dev_alloc_footprint((2 * IH + D) / 3 * sizeof(float));
    size_t cap = 8 * exp_bytes + exp_bytes / 2;
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)cap / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);

    qt_trunk_offer("lmhead", 0, exp_bytes);
    qt_trunk_offer("dnproj", 0, exp_bytes); qt_trunk_offer("dnout", 0, exp_bytes);
    qt_trunk_offer("attnproj", 1, exp_bytes); qt_trunk_offer("shexp", 1, exp_bytes);
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts (int4 mode, cap == n_experts)");
    check(qt_place_is_auto(), "COLI_PLACE unset: the placement is automatic");
    check(qt_place_of("lmhead", 0) == 0 && qt_place_of("dnproj", 0) == 0 && qt_place_of("dnout", 0) == 0 &&
          qt_place_of("attnproj", 1) == 0 && qt_place_of("shexp", 1) == 0, "all five offers placed on the one device");
    check(G_lmh.dev_ok, "lm_head has a device before the withdrawal");
    size_t before = G.budget[0];
    check(before + 5 * exp_bytes <= cap && before + 5 * exp_bytes + exp_bytes > cap,
          "the expert budget is the allowance minus the five placed offers");

    qt_trunk_withdraw("test");
    check(qt_place_of("lmhead", 0) == QT_PLACE_CPU && qt_place_of("dnproj", 0) == QT_PLACE_CPU &&
          qt_place_of("dnout", 0) == QT_PLACE_CPU && qt_place_of("attnproj", 1) == QT_PLACE_CPU &&
          qt_place_of("shexp", 1) == QT_PLACE_CPU, "after the withdrawal every offer answers CPU");
    check(!G_lmh.dev_ok, "lm_head has no device any more: qt_lmhead_init will refuse");
    check(G.budget[0] == before + 5 * exp_bytes, "the trunk's bytes are back in the expert budget");
    check(G_trunk_bytes[0] == 0, "no trunk bytes are charged to the device");
    check(qt_lmhead_init((const int8_t *)"x", (const float *)"x", 1, 1) == 0, "a later lm_head upload is refused");
    check(fake_uploads == 0, "nothing was uploaded: the withdrawal came before any trunk upload");

    /* a hand-written list is not automatic: nothing to withdraw, nothing changes */
    setenv("COLI_PLACE", "lmhead=0", 1);
    check(!qt_place_is_auto(), "an explicit COLI_PLACE list is not automatic");
    size_t held = G.budget[0];
    qt_trunk_withdraw("test again");
    check(G.budget[0] == held, "withdrawing an explicit placement is a no-op");

    qt_shutdown();
    if (fails) { printf("test_qwen36_tier_withdraw: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_tier_withdraw: the automatic trunk placement can be withdrawn, budget restored\n");
    return 0;
}
