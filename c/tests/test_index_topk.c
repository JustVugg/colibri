/* The DeepSeek V4.1 indexer's top-k (v4_topk.h's radix sort) against the selection it replaced, which
 * is the definition: the same positions, in the same order, for every row. Rows with
 * ties (scores drawn from a handful of values), unreachable positions, fewer reachable
 * positions than topk, topk of one, and a NaN. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../index_topk.h"

static unsigned g_seed = 12345;
static unsigned rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return (g_seed >> 8) & 0xffffff; }

static int g_rows;
static int check(const float *score, int n, int topk, const char *what) {
    g_rows++;
    int want[4096], got[4096];
    static uint64_t work[2 * 4096];
    int tw = coli_index_topk_scan(score, n, topk, want);
    int tg = coli_index_topk(score, n, topk, got, work);
    if (tw != tg || memcmp(want, got, (size_t)topk * sizeof(int))) {
        fprintf(stderr, "FAIL %s: n %d topk %d taken %d vs %d\n", what, n, topk, tw, tg);
        for (int k = 0; k < topk && k < 12; k++) fprintf(stderr, "  %d: %d vs %d\n", k, want[k], got[k]);
        return 1;
    }
    return 0;
}

int main(void) {
    int fails = 0;
    float score[4096];
    for (int trial = 0; trial < 3000; trial++) {
        int n = 1 + (int)(rnd() % 700);
        int topk = 1 + (int)(rnd() % 600);
        int levels = 1 + (int)(rnd() % (trial % 3 == 0 ? 4 : 1000));   /* few levels = many ties */
        int reach = (int)(rnd() % (n + 1));                            /* positions past it unreachable */
        for (int j = 0; j < n; j++) {
            if (j >= reach && trial % 2) { score[j] = -INFINITY; continue; }
            if (rnd() % 17 == 0) { score[j] = -INFINITY; continue; }
            score[j] = (float)(rnd() % levels) - (float)levels / 2.0f;
            if (rnd() % 23 == 0) score[j] = 0.0f;
        }
        fails += check(score, n, topk, "random");
    }
    /* every position reachable and fewer than topk: all of them, in order */
    for (int j = 0; j < 100; j++) score[j] = (float)((j * 37) % 11);
    fails += check(score, 100, 512, "all fit");
    /* nothing reachable */
    for (int j = 0; j < 50; j++) score[j] = -INFINITY;
    fails += check(score, 50, 8, "none");
    /* the long-prompt shape: 575 positions, keep 512 */
    for (int j = 0; j < 575; j++) score[j] = (float)(rnd() % 100000) / 7.0f;
    fails += check(score, 575, 512, "long");
    /* +inf is a score like any other, and -0 ties +0 (the lower position wins) */
    for (int j = 0; j < 30; j++) score[j] = (float)(j % 4);
    score[5] = INFINITY; score[22] = INFINITY; score[9] = -0.0f; score[3] = 0.0f; score[13] = -INFINITY;
    for (int k = 1; k <= 30; k++) fails += check(score, 30, k, "inf and signed zero");
    /* nothing to keep, or nothing to keep from */
    for (int j = 0; j < 20; j++) score[j] = (float)(j % 3);
    fails += check(score, 20, 0, "topk 0");
    fails += check(score, 0, 5, "n 0");
    { int want[8] = {7, 7, 7, 7, 7, 7, 7, 7}, got[8] = {7, 7, 7, 7, 7, 7, 7, 7}; static uint64_t work[64]; g_rows++;
      int tw = coli_index_topk_scan(score, 20, -3, want), tg = coli_index_topk(score, 20, -3, got, work);
      if (tw != tg || memcmp(want, got, sizeof want)) { fprintf(stderr, "FAIL topk negative: %d vs %d\n", tw, tg); fails++; } }
    /* a NaN goes through the scan */
    for (int j = 0; j < 40; j++) score[j] = (float)(j % 5);
    score[17] = NAN;
    fails += check(score, 40, 10, "nan");
    /* NaNs in random rows: the scan's answer, whatever it makes of them */
    for (int trial = 0; trial < 500; trial++) {
        int n = 2 + (int)(rnd() % 200), topk = 1 + (int)(rnd() % 100);
        for (int j = 0; j < n; j++) score[j] = (float)(rnd() % 50);
        for (int k = 0; k < 1 + (int)(rnd() % 3); k++) score[rnd() % n] = NAN;
        fails += check(score, n, topk, "random nan");
    }
    if (fails) { fprintf(stderr, "test_index_topk: %d of %d rows differ\n", fails, g_rows); return 1; }
    printf("test_index_topk: %d rows, the radix choice matches the scan; ok\n", g_rows);
    return 0;
}
