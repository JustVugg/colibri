/* qfe_keep and qfe_keep_guessed (qwen36_fast_experts.h) without a model: which
 * choices stay, what a left-out choice becomes, that the kept weights are not
 * touched, and how many stayed only because they were guessed. */
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

    /* qfe_keep_guessed, keep 5 and sure 3. By weight 9, 0 and 12 always stay; 7 and 4 (ranks 3 and 4) only if
     * cheap; 3, 15 and 1 (ranks 5 to 7) only if present. */
    unsigned char cheap[16] = {0}, pres[16] = {0};
    int guessed = 0;
    memcpy(ids, ids0, sizeof ids); memcpy(w, w0, sizeof w);
    left = qfe_keep_guessed(ids, w, 8, 5, 3, none, none, &guessed);
    CHECK(left == 5 && guessed == 0, "guessed, nothing cheap: left %d, kept as guessed %d, expected 5 and 0", left,
          guessed);
    CHECK(ids[1] == 9 && ids[7] == 0 && ids[3] == 12 && w[1] == w0[1] && w[7] == w0[7] && w[3] == w0[3],
          "guessed: the three sure ones did not stay");
    CHECK(ids[4] == 9 && w[4] == 0 && ids[5] == 9 && w[5] == 0, "guessed: ranks 3 and 4 stayed without being cheap");

    /* 15 cached (so cheap too), 7, 3 and 1 guessed: 7 (rank 3) and 15 (rank 6) stay, and only 7 stayed because
     * it was guessed; 4 (rank 4) goes; 3 (rank 5) and 1 (rank 7) go too, since after the keep heaviest only the
     * cache counts. */
    memcpy(ids, ids0, sizeof ids); memcpy(w, w0, sizeof w);
    pres[15] = 1;
    memcpy(cheap, pres, sizeof cheap); cheap[7] = cheap[3] = cheap[1] = 1;
    guessed = 0;
    left = qfe_keep_guessed(ids, w, 8, 5, 3, pres, cheap, &guessed);
    CHECK(left == 3, "guessed: left %d, expected 3", left);
    CHECK(guessed == 1, "guessed: %d kept as guessed, expected 1 (7)", guessed);
    CHECK(ids[4] == 7 && w[4] == w0[4], "guessed: a guessed rank 3 was left out");
    CHECK(ids[5] == 9 && w[5] == 0, "guessed: rank 4, neither guessed nor cached, stayed");
    CHECK(ids[0] == 9 && w[0] == 0, "guessed: rank 5, guessed but not cached, stayed");
    CHECK(ids[6] == 15 && w[6] == w0[6], "guessed: rank 6, cached, was left out");
    CHECK(ids[2] == 9 && w[2] == 0, "guessed: rank 7, guessed but not cached, stayed");

    /* sure = keep: no middle ranks, qfe_keep's row; sure 0 still keeps the heaviest; keep 8 with every expert
     * cheap leaves nothing out; a row of one is left alone. */
    int32_t a_ids[8], b_ids[8]; float a_w[8], b_w[8];
    memcpy(a_ids, ids0, sizeof a_ids); memcpy(a_w, w0, sizeof a_w);
    memcpy(b_ids, ids0, sizeof b_ids); memcpy(b_w, w0, sizeof b_w);
    CHECK(qfe_keep_guessed(a_ids, a_w, 8, 5, 5, pres, none, NULL) == qfe_keep(b_ids, b_w, 8, 5, pres),
          "sure = keep: another count than qfe_keep");
    CHECK(!memcmp(a_ids, b_ids, sizeof a_ids) && !memcmp(a_w, b_w, sizeof a_w), "sure = keep: not qfe_keep's row");
    memcpy(ids, ids0, sizeof ids); memcpy(w, w0, sizeof w);
    left = qfe_keep_guessed(ids, w, 8, 5, 0, none, none, NULL);
    CHECK(left == 7 && ids[1] == 9 && w[1] == w0[1], "guessed, sure 0: the heaviest did not stay (left %d)", left);
    memcpy(ids, ids0, sizeof ids); memcpy(w, w0, sizeof w);
    CHECK(qfe_keep_guessed(ids, w, 8, 8, 3, none, all, NULL) == 0 && !memcmp(ids, ids0, sizeof ids) &&
          !memcmp(w, w0, sizeof w), "guessed, keep 8, all cheap: something was left out");
    CHECK(qfe_keep_guessed(ids, w, 1, 1, 1, none, none, NULL) == 0, "guessed: a single choice was left out");

    printf("test_qwen36_fast_experts: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
