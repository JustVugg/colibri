/* Stock-storage VRAM tier prewarm ranking for DeepSeek V4.
 *
 * Ranks (layer, expert) pairs globally by cumulative usage frequency
 * descending, with deterministic tie-breaking (expert id ascending, then layer
 * id ascending). Never-used experts (usage == 0) are excluded so that when no
 * usage history exists, prewarm selects 0 experts and falls back cleanly to
 * demand loading, matching coli_v4_expert_store_prewarm's no-history policy. */
#ifndef COLI_V4_TIER_PREWARM_H
#define COLI_V4_TIER_PREWARM_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

enum { COLI_V4_TIER_PREWARM_AHEAD_MAX = 8 };

typedef struct {
    int layer;
    int expert;
    uint64_t frequency;
} ColiV4TierRank;

static inline int coli_v4_tier_rank_compare(const void *a, const void *b) {
    const ColiV4TierRank *x = (const ColiV4TierRank *)a;
    const ColiV4TierRank *y = (const ColiV4TierRank *)b;
    if (x->frequency != y->frequency) return x->frequency > y->frequency ? -1 : 1;
    if (x->expert != y->expert) return x->expert < y->expert ? -1 : 1;
    return (x->layer > y->layer) - (x->layer < y->layer);
}

/* Clamp the read-ahead width (COLI_V4_TIER_PREWARM_AHEAD) to [1, AHEAD_MAX]
 * and never above a layer's RAM slot budget when slots_per_layer > 0. */
static inline int coli_v4_tier_prewarm_ahead_clamp(int requested,
                                                   int slots_per_layer) {
    int ahead = requested;
    if (ahead > COLI_V4_TIER_PREWARM_AHEAD_MAX)
        ahead = COLI_V4_TIER_PREWARM_AHEAD_MAX;
    if (slots_per_layer > 0 && ahead > slots_per_layer)
        ahead = slots_per_layer;
    if (ahead < 1) ahead = 1;
    return ahead;
}

/* Fill `out` (capacity `cap`) with up to `cap` most-used (layer, expert) pairs
 * from `usage[layers * experts_per_layer]`. Never-used experts (usage == 0)
 * are excluded. Returns the number of entries written to `out` (0 when no
 * usage history exists or inputs are empty/invalid). */
static inline size_t coli_v4_tier_prewarm_select(
    const uint64_t *usage, int layers, int experts_per_layer,
    ColiV4TierRank *out, size_t cap) {
    if (!usage || !out || layers <= 0 || experts_per_layer <= 0 || cap == 0)
        return 0;
    size_t total = (size_t)layers * (size_t)experts_per_layer;
    size_t nonzero = 0;
    for (size_t i = 0; i < total; i++) {
        if (usage[i] > 0) nonzero++;
    }
    if (!nonzero) return 0;
    ColiV4TierRank *ranked = (ColiV4TierRank *)malloc(nonzero * sizeof(*ranked));
    if (!ranked) return 0;
    size_t n = 0;
    for (int layer = 0; layer < layers; layer++) {
        size_t base = (size_t)layer * (size_t)experts_per_layer;
        for (int expert = 0; expert < experts_per_layer; expert++) {
            uint64_t freq = usage[base + (size_t)expert];
            if (freq > 0) {
                ranked[n++] = (ColiV4TierRank){layer, expert, freq};
            }
        }
    }
    qsort(ranked, n, sizeof(*ranked), coli_v4_tier_rank_compare);
    size_t take = n < cap ? n : cap;
    for (size_t i = 0; i < take; i++) out[i] = ranked[i];
    free(ranked);
    return take;
}

#endif /* COLI_V4_TIER_PREWARM_H */
