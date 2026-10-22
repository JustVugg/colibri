/* qfe_keep (qwen36_fast_experts.h) without a model: which choices stay, what a
 * left-out choice becomes, and that the kept weights are not touched. */
#include "../qwen36_fast_experts.h"
#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, ...) do { checks++; if (!(c)) { failures++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    unsigned char none[16] = {0}, all[16], some[16] = {0};
    memset(all, 1, sizeof all);

    /* Nothing in memory, keep 5 of 8: the three lightest go, whatever their position. */
    int32_t ids[8] = {3, 9, 1, 12, 7, 4, 15, 0};
    float w[8] = {0.05f, 0.30f, 0.02f, 0.15f, 0.10f, 0.08f, 0.04f, 0.26f};
    int32_t ids0[8]; float w0[8];
    memcpy(ids0, ids, sizeof ids); memcpy(w0, w, sizeof w);
    int left = qfe_keep(ids, w, 8, 5, none);
    CHECK(left == 3, "left %d, expected 3", left);
    /* by weight: 0.30 (9), 0.26 (0), 0.15 (12), 0.10 (7), 0.08 (4) stay; 0.05 (3), 0.04 (15), 0.02 (1) go */
    const int gone[8] = {1, 0, 1, 0, 0, 0, 1, 0};
    for (int i = 0; i < 8; i++) {
        if (gone[i]) CHECK(ids[i] == 9 && w[i] == 0.0f, "slot %d: id %d weight %g, expected 9 and 0", i, ids[i], w[i]);
        else CHECK(ids[i] == ids0[i] && w[i] == w0[i], "slot %d changed: id %d weight %g", i, ids[i], w[i]);
    }

    /* Everything in memory: nothing changes. */
    memcpy(ids, ids0, sizeof ids); memcpy(w, w0, sizeof w);
    CHECK(qfe_keep(ids, w, 8, 2, all) == 0, "experts in memory were left out");
    CHECK(!memcmp(ids, ids0, sizeof ids) && !memcmp(w, w0, sizeof w), "a row in memory changed");

    /* Only some in memory, keep 2: the two heaviest stay, the others stay only if present. */
    memcpy(ids, ids0, sizeof ids); memcpy(w, w0, sizeof w);
    some[12] = some[1] = 1;                     /* 0.15 and 0.02 are in memory */
    left = qfe_keep(ids, w, 8, 2, some);
    CHECK(left == 4, "left %d, expected 4", left);
    CHECK(ids[1] == 9 && w[1] == w0[1] && ids[7] == 0 && w[7] == w0[7], "the two heaviest did not stay");
    CHECK(ids[3] == 12 && w[3] == w0[3] && ids[2] == 1 && w[2] == w0[2], "experts in memory were left out");
    CHECK(w[0] == 0 && w[4] == 0 && w[5] == 0 && w[6] == 0, "experts not in memory stayed");
    float sum = 0, sum0 = 0;
    for (int i = 0; i < 8; i++) { sum += w[i]; if (some[ids0[i]] || ids0[i] == 9 || ids0[i] == 0) sum0 += w0[i]; }
    CHECK(sum == sum0, "the kept weights are not the router's (%g against %g)", sum, sum0);

    /* keep >= k, or a row of one: nothing to do; keep 0 still keeps the heaviest. */
    memcpy(ids, ids0, sizeof ids); memcpy(w, w0, sizeof w);
    CHECK(qfe_keep(ids, w, 8, 8, none) == 0 && !memcmp(w, w0, sizeof w), "keep 8 left something out");
    CHECK(qfe_keep(ids, w, 1, 0, none) == 0, "a single choice was left out");
    left = qfe_keep(ids, w, 8, 0, none);
    CHECK(left == 7 && ids[1] == 9 && w[1] == w0[1], "keep 0 did not keep the heaviest (left %d)", left);

    /* Equal weights: the router's order decides. */
    int32_t tid[4] = {5, 6, 7, 8};
    float tw[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    CHECK(qfe_keep(tid, tw, 4, 2, none) == 2, "ties: wrong count");
    CHECK(tid[0] == 5 && tid[1] == 6 && tw[2] == 0 && tw[3] == 0 && tid[2] == 5, "ties: not in the router's order");

    printf("test_qwen36_fast_experts: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
