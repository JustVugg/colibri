#ifndef COLI_V41_TOPK_H
#define COLI_V41_TOPK_H

#include <stdint.h>
#include <math.h>
#include <string.h>

/* Higher score wins; exact ties keep the lower original position, matching the
 * previous left-to-right scan with a strict `>` comparison. */
static inline int v41_topk_better(float sa, int ia, float sb, int ib) {
    if (sa > sb) return 1;
    if (sb > sa) return 0;
    return sa == sb && ia < ib;
}

/* The heap root is the least desirable retained candidate. */
static inline int v41_topk_worse(float sa, int ia, float sb, int ib) {
    if (sa < sb) return 1;
    if (sb < sa) return 0;
    return sa == sb && ia > ib;
}

/* Select up to `limit` finite-sentinel scores and return their positions in
 * ascending order. `out` is used as bounded heap scratch then compacted in place;
 * `selected` must have `width` bytes. Negative finite scores remain eligible. */
static inline int v41_topk_positions(const float *scores, int width, int limit,
                                     int *out, uint8_t *selected) {
    if (width <= 0 || limit <= 0) return 0;
    if (limit > width) limit = width;
    /* Preserve the old left-to-right behavior for malformed numeric rows. NaN
     * comparisons do not define a total order, so do not feed them to a heap. */
    for (int i = 0; i < width; i++) {
        if (!isnan(scores[i])) continue;
        int count = 0;
        for (int k = 0; k < limit; k++) {
            int best = -1;
            for (int j = 0; j < width; j++) {
                if (scores[j] == -INFINITY) continue;
                int already = 0;
                for (int u = 0; u < count; u++)
                    if (out[u] == j) { already = 1; break; }
                if (already) continue;
                if (best < 0 || scores[j] > scores[best]) best = j;
            }
            if (best < 0) break;
            out[count++] = best;
        }
        for (int a = 0; a < count; a++)
            for (int b = a + 1; b < count; b++)
                if (out[b] < out[a]) { int swap = out[a]; out[a] = out[b]; out[b] = swap; }
        for (int k = count; k < limit; k++) out[k] = -1;
        return count;
    }
    int count = 0;
    for (int i = 0; i < width; i++) {
        float value = scores[i];
        if (value == -INFINITY) continue;
        if (count < limit) {
            int at = count++;
            out[at] = i;
            while (at > 0) {
                int parent = (at - 1) / 2;
                int parent_i = out[parent];
                if (!v41_topk_worse(scores[out[at]], out[at],
                                    scores[parent_i], parent_i)) break;
                out[parent] = out[at];
                out[at] = parent_i;
                at = parent;
            }
            continue;
        }
        int root = out[0];
        if (!v41_topk_better(value, i, scores[root], root)) continue;
        out[0] = i;
        int at = 0;
        for (;;) {
            int left = at * 2 + 1;
            if (left >= count) break;
            int worst = left;
            int right = left + 1;
            if (right < count) {
                int li = out[left], ri = out[right];
                if (v41_topk_worse(scores[ri], ri, scores[li], li)) worst = right;
            }
            int wi = out[worst], ai = out[at];
            if (!v41_topk_worse(scores[wi], wi, scores[ai], ai)) break;
            out[worst] = ai;
            out[at] = wi;
            at = worst;
        }
    }

    memset(selected, 0, (size_t)width);
    for (int i = 0; i < count; i++) selected[out[i]] = 1;
    int sorted = 0;
    for (int i = 0; i < width; i++)
        if (selected[i]) out[sorted++] = i;
    for (int i = sorted; i < limit; i++) out[i] = -1;
    return sorted;
}

#endif
