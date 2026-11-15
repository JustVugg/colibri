/* Fast mode, decode: fewer experts read from the disk (QWEN36_FAST_EXPERTS=J).
 *
 * A token's J likeliest routed experts are always used; the others of its
 * top-k only when the expert cache holds them. The kept weights stay as the
 * router wrote them, divided by the probability mass of the whole top-k, so a
 * left-out expert's share is simply missing instead of being spread over the
 * others: renormalizing over the kept experts would raise their gain above
 * what the model was trained with (arXiv 2609.04575 studies this on
 * Qwen3.6-35B-A3B). QWEN36_FAST_GUESSED=R narrows the J likeliest: the R - 1
 * likeliest are always used, the next ones only when the expert cache holds
 * them or the layer before guessed them, so that their reads had already
 * begun (qfe_keep_guessed). Header-only; the graph supplies which experts the
 * cache holds and which were guessed.
 */
#ifndef QWEN36_FAST_EXPERTS_H
#define QWEN36_FAST_EXPERTS_H
#include <stdint.h>

/* One token's top-k choices (ids and weights, in any order). The keep heaviest
 * stay (at least one); each other one stays where present[id] is nonzero, and
 * otherwise points at the heaviest expert with weight 0: it costs no read and
 * no cache slot, and that expert's FFN runs once more for it, at weight 0.
 * Returns how many were left out. */
static inline int qfe_keep(int32_t *ids, float *weights, int k, int keep, const unsigned char *present) {
    int order[8], left = 0;
    if (k < 2 || k > 8 || keep >= k) return 0;
    if (keep < 1) keep = 1;
    for (int i = 0; i < k; i++) {               /* heaviest first; ties keep the router's order */
        int j = i;
        while (j > 0 && weights[order[j - 1]] < weights[i]) { order[j] = order[j - 1]; j--; }
        order[j] = i;
    }
    int32_t heaviest = ids[order[0]];
    for (int r = keep; r < k; r++) {
        int i = order[r];
        if (present[ids[i]]) continue;
        ids[i] = heaviest;
        weights[i] = 0;
        left++;
    }
    return left;
}

/* As qfe_keep, with a rule for the middle ranks (QWEN36_FAST_GUESSED): of the
 * keep heaviest, the sure heaviest stay (at least one) and the others only
 * where cheap[id] is nonzero (in the cache, or guessed for the layer: read
 * anyway); the ones after the keep heaviest stay where present[id] is
 * nonzero. A left-out choice points at the heaviest expert with weight 0, as
 * in qfe_keep. Returns how many were left out; when guessed is not NULL,
 * *guessed grows by the middle ranks that stayed only by cheap[id] (guessed,
 * not present). */
static inline int qfe_keep_guessed(int32_t *ids, float *weights, int k, int keep, int sure,
                                   const unsigned char *present, const unsigned char *cheap, int *guessed) {
    int order[8], left = 0;
    if (k < 2 || k > 8) return 0;
    if (keep > k) keep = k;
    if (sure < 1) sure = 1;
    for (int i = 0; i < k; i++) {               /* heaviest first; ties keep the router's order */
        int j = i;
        while (j > 0 && weights[order[j - 1]] < weights[i]) { order[j] = order[j - 1]; j--; }
        order[j] = i;
    }
    int32_t heaviest = ids[order[0]];
    for (int r = sure; r < k; r++) {
        int i = order[r];
        if (guessed && r < keep && cheap[ids[i]] && !present[ids[i]]) (*guessed)++;
        if (r < keep ? cheap[ids[i]] : present[ids[i]]) continue;
        ids[i] = heaviest;
        weights[i] = 0;
        left++;
    }
    return left;
}

#endif
