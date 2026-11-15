/* index_topk.h -- the DeepSeek V4.1 indexer's choice of compressed positions.
 *
 * A query keeps the `topk` best-scoring compressed positions it can reach (a score of
 * -INFINITY marks one it cannot), then lists them in position order. Among equal scores
 * the lower position wins, because the original selection scanned positions upward and
 * replaced its pick only on a strictly greater score.
 *
 * coli_index_topk_scan is that selection as first written: topk passes over every
 * position, each checking the picks so far, which is topk^2 x positions and became most
 * of a long prompt's prefill and of a long context's decode step. coli_index_topk makes
 * the same choice with the V4 indexer's radix top-k (v4_topk.h: score descending, then
 * position ascending, -0 equal to +0): when every reachable position fits it takes them
 * all, otherwise it ranks the row, keeps the first topk, which are all reachable because
 * -INFINITY ranks last, and puts them back in position order. v4_topk_desc refuses a row
 * holding a NaN, which has no place in that order; such a row goes through the scan,
 * which is the definition.
 *
 * `row` gets the chosen positions in increasing order and -1 after them; the return
 * value is how many were chosen. `work` is the caller's scratch of 2 * n uint64_t.
 */
#ifndef COLI_INDEX_TOPK_H
#define COLI_INDEX_TOPK_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "v4_topk.h"

static inline int coli_index_topk_scan(const float *score, int n, int topk, int *row) {
    for (int k = 0; k < topk; k++) row[k] = -1;
    int taken = 0;
    for (int k = 0; k < topk; k++) {
        int best = -1;
        for (int j = 0; j < n; j++) {
            if (score[j] == -INFINITY) continue;
            int already = 0;
            for (int u = 0; u < taken; u++) if (row[u] == j) { already = 1; break; }
            if (already) continue;
            if (best < 0 || score[j] > score[best]) best = j;
        }
        if (best < 0) break;
        row[taken++] = best;
    }
    for (int a = 0; a < taken; a++)
        for (int b = a + 1; b < taken; b++)
            if (row[b] < row[a]) { int swap = row[a]; row[a] = row[b]; row[b] = swap; }
    return taken;
}

static inline int coli_index_topk_pos(const void *pa, const void *pb) {
    int a = *(const int *)pa, b = *(const int *)pb;
    return a < b ? -1 : a > b;
}

static inline int coli_index_topk(const float *score, int n, int topk, int *row, uint64_t *work) {
    if (topk <= 0) return 0;   /* the scan keeps nothing and writes nothing then */
    int valid = 0;
    for (int j = 0; j < n; j++) valid += score[j] != -INFINITY;
    int taken = 0;
    if (valid <= topk) {       /* every reachable position, already in order */
        for (int j = 0; j < n; j++) if (score[j] != -INFINITY) row[taken++] = j;
    } else {
        taken = v4_topk_desc(row, score, n, topk, work);
        if (taken < 0) return coli_index_topk_scan(score, n, topk, row);
        qsort(row, (size_t)taken, sizeof(int), coli_index_topk_pos);
    }
    for (int k = taken; k < topk; k++) row[k] = -1;
    return taken;
}

#endif /* COLI_INDEX_TOPK_H */
