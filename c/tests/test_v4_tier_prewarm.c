/* Unit tests for DeepSeek V4 stock-storage VRAM tier prewarm ranking
 * (c/v4_tier_prewarm.h): global frequency ordering, deterministic tie-breaking,
 * capacity cap enforcement, zero-usage exclusion / no-history fallback, and
 * read-ahead clamping. Pure CPU.
 *
 * Build and run:
 *   make -C c tests/test_v4_tier_prewarm && c/tests/test_v4_tier_prewarm */
#include "../v4_tier_prewarm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(cond, msg) do {                                                \
    if (!(cond)) {                                                           \
        fprintf(stderr, "FAIL (%s:%d): %s\n", __FILE__, __LINE__, (msg));    \
        failures++;                                                          \
    }                                                                        \
} while (0)

/* 1. Most-used experts selected first across layers, with deterministic
 *    tie-breaking (expert id ascending, then layer id ascending). */
static void test_ordering_and_ties(void) {
    /* 3 layers x 4 experts */
    uint64_t usage[3 * 4] = {
        /* layer 0: e0=10, e1=50, e2=0,  e3=20 */
        10, 50, 0, 20,
        /* layer 1: e0=50, e1=20, e2=90, e3=0  */
        50, 20, 90, 0,
        /* layer 2: e0=20, e1=50, e2=10, e3=90 */
        20, 50, 10, 90,
    };
    ColiV4TierRank out[12];
    size_t n = coli_v4_tier_prewarm_select(usage, 3, 4, out, 12);
    /* 10 nonzero entries out of 12 */
    CHECK(n == 10, "expected 10 nonzero ranked experts");
    if (n != 10) return;

    /* Expected order:
     * freq 90: (1, 2), (2, 3)         -- expert 2 < expert 3
     * freq 50: (1, 0), (0, 1), (2, 1) -- expert 0 < expert 1; layer 0 < layer 2
     * freq 20: (2, 0), (1, 1), (0, 3) -- expert 0 < expert 1 < expert 3
     * freq 10: (0, 0), (2, 2)         -- expert 0 < expert 2
     */
    const ColiV4TierRank want[10] = {
        {1, 2, 90}, {2, 3, 90},
        {1, 0, 50}, {0, 1, 50}, {2, 1, 50},
        {2, 0, 20}, {1, 1, 20}, {0, 3, 20},
        {0, 0, 10}, {2, 2, 10},
    };
    for (size_t i = 0; i < 10; i++) {
        CHECK(out[i].layer == want[i].layer &&
              out[i].expert == want[i].expert &&
              out[i].frequency == want[i].frequency,
              "rank entry mismatch");
    }
}

/* 2. Cap strictly respected when nonzero experts exceed capacity. */
static void test_cap_respected(void) {
    uint64_t usage[2 * 5] = {
        5, 100, 30, 80, 10,
        70, 20, 90, 40, 60,
    };
    ColiV4TierRank out[5];
    memset(out, 0x5a, sizeof(out));
    size_t n = coli_v4_tier_prewarm_select(usage, 2, 5, out, 3);
    CHECK(n == 3, "expected cap of 3");
    CHECK(out[0].layer == 0 && out[0].expert == 1 && out[0].frequency == 100,
          "top 1 should be (0,1)=100");
    CHECK(out[1].layer == 1 && out[1].expert == 2 && out[1].frequency == 90,
          "top 2 should be (1,2)=90");
    CHECK(out[2].layer == 0 && out[2].expert == 3 && out[2].frequency == 80,
          "top 3 should be (0,3)=80");
    /* Sentinel slots beyond cap must remain untouched */
    CHECK(out[3].layer == 0x5a5a5a5a && out[4].layer == 0x5a5a5a5a,
          "must not write past cap");
}

/* 3. Clean fallback when no usage history exists (all zeros or NULL). */
static void test_no_history_fallback(void) {
    uint64_t zeros[4 * 8];
    memset(zeros, 0, sizeof(zeros));
    ColiV4TierRank out[8];
    memset(out, 0x7f, sizeof(out));

    size_t n = coli_v4_tier_prewarm_select(zeros, 4, 8, out, 8);
    CHECK(n == 0, "all-zero usage should select 0 experts");
    CHECK(out[0].layer == 0x7f7f7f7f, "all-zero usage should not modify out");

    CHECK(coli_v4_tier_prewarm_select(NULL, 4, 8, out, 8) == 0,
          "NULL usage should select 0 experts");
    CHECK(coli_v4_tier_prewarm_select(zeros, 0, 8, out, 8) == 0,
          "zero layers should select 0 experts");
    CHECK(coli_v4_tier_prewarm_select(zeros, 4, 0, out, 8) == 0,
          "zero experts should select 0 experts");
    CHECK(coli_v4_tier_prewarm_select(zeros, 4, 8, out, 0) == 0,
          "zero cap should select 0 experts");
    CHECK(coli_v4_tier_prewarm_select(zeros, 4, 8, NULL, 8) == 0,
          "NULL out should select 0 experts");
}

/* 4. Read-ahead clamping bounds. */
static void test_ahead_clamp(void) {
    CHECK(coli_v4_tier_prewarm_ahead_clamp(0, 16) == 1, "0 clamps to 1");
    CHECK(coli_v4_tier_prewarm_ahead_clamp(-5, 16) == 1, "negative clamps to 1");
    CHECK(coli_v4_tier_prewarm_ahead_clamp(1, 16) == 1, "1 stays 1");
    CHECK(coli_v4_tier_prewarm_ahead_clamp(4, 16) == 4, "4 stays 4");
    CHECK(coli_v4_tier_prewarm_ahead_clamp(99, 16) == COLI_V4_TIER_PREWARM_AHEAD_MAX,
          "large value clamps to AHEAD_MAX");
    CHECK(coli_v4_tier_prewarm_ahead_clamp(8, 3) == 3,
          "clamps to slots_per_layer when smaller");
    CHECK(coli_v4_tier_prewarm_ahead_clamp(4, 0) == 4,
          "slots_per_layer=0 does not clamp to 0");
}

int main(void) {
    test_ordering_and_ties();
    test_cap_respected();
    test_no_history_fallback();
    test_ahead_clamp();
    if (failures) {
        fprintf(stderr, "test_v4_tier_prewarm: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_v4_tier_prewarm: PASS\n");
    return 0;
}
