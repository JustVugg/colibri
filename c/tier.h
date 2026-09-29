#ifndef COLIBRI_TIER_H
#define COLIBRI_TIER_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Shared admission contract for every adaptive resident tier.  Widen before
 * adding the margin: a saturated uint32 heat counter must become sticky, not
 * wrap the threshold and admit a colder expert. */
static int tier_should_promote(uint32_t hot, uint32_t cold){
    uint64_t threshold=(uint64_t)cold+((uint64_t)cold>>2)+4u;
    return (uint64_t)hot>threshold;
}

static uint32_t tier_decay_value(uint32_t heat){ return heat>>1; }

/* Pick one RAM/VRAM hot-store slot to replace from recent routing heat.
 * The fixed margin handles tiny samples; the 25% margin prevents ping-pong. */
static int tier_pick_swap(const uint32_t *heat, int nexpert,
                          const int *pinned, int npin,
                          int *slot, int *eid, long *gain){
    if(!heat || !pinned || npin<1 || nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++) if(heat[pinned[z]]<heat[pinned[cold]]) cold=z;
    int hot=-1; uint32_t fh=0;
    for(int e=0;e<nexpert;e++){
        int resident=0;
        for(int z=0;z<npin;z++) if(pinned[z]==e){ resident=1; break; }
        if(!resident && heat[e]>fh){ fh=heat[e]; hot=e; }
    }
    if(hot<0) return 0;
    uint32_t fc=heat[pinned[cold]];
    if(!tier_should_promote(fh,fc)) return 0;
    *slot=cold; *eid=hot; *gain=(long)fh-(long)fc;
    return 1;
}

/* LFRU: frequency is the primary signal; recency breaks close calls. A recent
 * access contributes at most 255 points while one frequency count is worth
 * 256, so a merely recent expert cannot displace a genuinely hotter one. */
static uint64_t tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock){
    uint32_t age=clock-last, recent=age<255?255-age:0;
    return ((uint64_t)heat<<8)|recent;
}

static int tier_pick_lfru(const uint32_t *heat, const uint32_t *last, uint32_t clock,
                          int nexpert, const int *pinned, int npin,
                          int *slot, int *eid, long *gain){
    if(!heat||!last||!pinned||npin<1||nexpert<1) return 0;
    int cold=0;
    for(int z=1;z<npin;z++)
        if(tier_lfru_score(heat[pinned[z]],last[pinned[z]],clock)<
           tier_lfru_score(heat[pinned[cold]],last[pinned[cold]],clock)) cold=z;
    int hot=-1; uint64_t hs=0;
    for(int e=0;e<nexpert;e++){
        int resident=0; for(int z=0;z<npin;z++) if(pinned[z]==e){resident=1;break;}
        uint64_t score=tier_lfru_score(heat[e],last[e],clock);
        if(!resident&&(hot<0||score>hs)){ hot=e; hs=score; }
    }
    if(hot<0) return 0;
    uint64_t cs=tier_lfru_score(heat[pinned[cold]],last[pinned[cold]],clock);
    /* Retain the existing 25%+4-frequency hysteresis in score units. */
    if(hs<=cs+(cs>>2)+(4u<<8)) return 0;
    *slot=cold; *eid=hot; *gain=(long)((hs-cs)>>8); return 1;
}

static void tier_decay(uint32_t *heat, int nexpert){
    for(int e=0;e<nexpert;e++) heat[e]=tier_decay_value(heat[e]);
}

/* ------------------------------------------------------------------------
 * Phase 1 memory tiering: temporal-locality guards for eviction policy.
 *
 * Two independent, O(1)-lookup guards over the most recent routing events of
 * a layer.  Both are DATA-STAGING ONLY: they influence which cache slot is
 * chosen as an eviction victim, never routing, weights, logits or KV.
 *
 *   1. Ring: per layer, the last `depth` routed expert sets, each stored as
 *      a (nexpert+63)/64-word bitmap.  Membership = the expert was routed
 *      in one of the last `depth` events.
 *   2. Rolling two-generation Bloom filter: per layer, two 64-word (4096
 *      bit) generations.  Every push inserts 2 bits (two splitmix64-style
 *      hashes of layer|eid|salt) into the current generation; every
 *      `bloom_window` pushes the older generation is cleared and becomes
 *      the insertion target, so lookups see a window of up to 2*window
 *      pushes with a false-positive rate only, never false negatives for
 *      the recent window.
 *
 * All storage is allocated once in tier_recent_init(); push/test/bonus
 * never allocate.  For 16 layers x 64 experts x depth 16 the ring is 2 KB
 * and the bloom 16 KB.  A zeroed (never initialised) struct is safe: every
 * accessor degrades to "not protected".
 * ------------------------------------------------------------------------ */

#define TIER_RECENT_DEPTH 16   /* default ring depth (routing events) */
#define TIER_BLOOM_WORDS  64   /* 64 words = 4096 bits per generation */

typedef struct {
    int nlayer, nexpert;
    int depth;             /* ring depth; 0 = ring disabled */
    int words;             /* 64-bit words per routed set */
    int bloom_window;      /* pushes per bloom generation; 0 = disabled */
    uint64_t *ring;        /* [nlayer * depth * words] */
    uint32_t *ring_pos;    /* [nlayer] next ring slot to overwrite */
    uint64_t *bloom;       /* [nlayer * 2 * TIER_BLOOM_WORDS] */
    uint8_t  *bloom_gen;   /* [nlayer] current generation (0/1) */
    uint32_t *bloom_count; /* [nlayer] pushes in the current generation */
    uint64_t bonus;        /* heat units granted when protected (RECENT_BONUS) */
} TierRecent;

static inline uint64_t tier_recent_mix(uint64_t x){
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* Splitmix64-style hash over (layer, eid, salt); salt picks the bit. */
static inline uint64_t tier_recent_hash(int layer, int eid, uint64_t salt){
    uint64_t key = ((uint64_t)(uint32_t)layer << 32) | (uint32_t)eid;
    return tier_recent_mix(key ^ (salt * 0x9E3779B97F4A7C15ULL));
}

static inline void tier_recent_free(TierRecent *tr){
    if(!tr) return;
    free(tr->ring); free(tr->ring_pos);
    free(tr->bloom); free(tr->bloom_gen); free(tr->bloom_count);
    memset(tr, 0, sizeof(*tr));
}

/* depth <= 0 disables the ring, bloom_window <= 0 the Bloom filter; both
 * disabled leaves an inert struct.  Returns 0 on success, -1 on OOM. */
static inline int tier_recent_init(TierRecent *tr, int nlayer, int nexpert,
                                   int depth, int bloom_window){
    if(!tr) return -1;
    memset(tr, 0, sizeof(*tr));
    tr->bonus = 1;
    if(nlayer < 1 || nexpert < 1) return -1;
    tr->nlayer = nlayer; tr->nexpert = nexpert;
    if(depth > 0){
        tr->depth = depth;
        tr->words = (nexpert + 63) / 64;
        tr->ring = calloc((size_t)nlayer * (size_t)depth * (size_t)tr->words,
                          sizeof(uint64_t));
        tr->ring_pos = calloc((size_t)nlayer, sizeof(uint32_t));
        if(!tr->ring || !tr->ring_pos){ tier_recent_free(tr); return -1; }
    }
    if(bloom_window > 0){
        tr->bloom_window = bloom_window;
        tr->bloom = calloc((size_t)nlayer * 2 * TIER_BLOOM_WORDS,
                           sizeof(uint64_t));
        tr->bloom_gen = calloc((size_t)nlayer, 1);
        tr->bloom_count = calloc((size_t)nlayer, sizeof(uint32_t));
        if(!tr->bloom || !tr->bloom_gen || !tr->bloom_count){
            tier_recent_free(tr); return -1;
        }
    }
    return 0;
}

static inline void tier_recent_bloom_insert(TierRecent *tr, int layer, int eid){
    uint64_t *gen = tr->bloom +
                    ((size_t)layer * 2 + tr->bloom_gen[layer]) * TIER_BLOOM_WORDS;
    uint64_t h0 = tier_recent_hash(layer, eid, 0);
    uint64_t h1 = tier_recent_hash(layer, eid, 1);
    gen[(h0 >> 6) & (TIER_BLOOM_WORDS - 1)] |= 1ULL << (h0 & 63);
    gen[(h1 >> 6) & (TIER_BLOOM_WORDS - 1)] |= 1ULL << (h1 & 63);
}

static inline int tier_recent_bloom_test(const TierRecent *tr, int layer, int eid){
    const uint64_t *g0 = tr->bloom + (size_t)layer * 2 * TIER_BLOOM_WORDS;
    const uint64_t *g1 = g0 + TIER_BLOOM_WORDS;
    uint64_t h0 = tier_recent_hash(layer, eid, 0);
    uint64_t h1 = tier_recent_hash(layer, eid, 1);
    uint64_t m0 = 1ULL << (h0 & 63), m1 = 1ULL << (h1 & 63);
    int w0 = (int)((h0 >> 6) & (TIER_BLOOM_WORDS - 1));
    int w1 = (int)((h1 >> 6) & (TIER_BLOOM_WORDS - 1));
    return ((g0[w0] & m0) && (g0[w1] & m1)) ||
           ((g1[w0] & m0) && (g1[w1] & m1));
}

/* One routing event: record the routed expert set for this layer.  Callers
 * serialise concurrent pushes (olmoe holds g_pilot_mx), so no locking here. */
static inline void tier_recent_push(TierRecent *tr, int layer,
                                    const int *eids, int k){
    if(!tr || !eids || layer < 0 || layer >= tr->nlayer) return;
    if(tr->ring && tr->depth > 0){
        int pos = (int)tr->ring_pos[layer];
        uint64_t *set = tr->ring +
                        ((size_t)layer * tr->depth + (size_t)pos) * tr->words;
        memset(set, 0, (size_t)tr->words * sizeof(uint64_t));
        for(int i = 0; i < k; i++){
            int e = eids[i];
            if(e < 0 || e >= tr->nexpert) continue;
            set[e >> 6] |= 1ULL << (e & 63);
        }
        tr->ring_pos[layer] = (uint32_t)((pos + 1) % tr->depth);
    }
    if(tr->bloom && tr->bloom_window > 0){
        for(int i = 0; i < k; i++){
            int e = eids[i];
            if(e < 0 || e >= tr->nexpert) continue;
            tier_recent_bloom_insert(tr, layer, e);
        }
        if(++tr->bloom_count[layer] >= (uint32_t)tr->bloom_window){
            tr->bloom_count[layer] = 0;
            tr->bloom_gen[layer] ^= 1;
            memset(tr->bloom + ((size_t)layer * 2 + tr->bloom_gen[layer]) *
                   TIER_BLOOM_WORDS, 0,
                   TIER_BLOOM_WORDS * sizeof(uint64_t));
        }
    }
}

/* Ring OR Bloom membership: 1 = routed recently in this layer. */
static inline int tier_recent_protected(const TierRecent *tr, int layer, int eid){
    if(!tr || layer < 0 || layer >= tr->nlayer ||
       eid < 0 || eid >= tr->nexpert) return 0;
    if(tr->ring && tr->depth > 0){
        const uint64_t *sets = tr->ring + (size_t)layer * tr->depth * tr->words;
        uint64_t mask = 1ULL << (eid & 63);
        int word = eid >> 6;
        for(int i = 0; i < tr->depth; i++)
            if(sets[(size_t)i * tr->words + word] & mask) return 1;
    }
    if(tr->bloom && tr->bloom_window > 0)
        return tier_recent_bloom_test(tr, layer, eid);
    return 0;
}

/* Score bonus in heat units; nonzero only for protected experts. */
static inline uint64_t tier_recent_bonus(const TierRecent *tr, int layer, int eid){
    if(!tier_recent_protected(tr, layer, eid)) return 0;
    return tr->bonus;
}

/* ------------------------------------------------------------------------
 * Phase 2 memory tiering: inter-layer routing transition chains.
 *
 * A compact saturating table of inter-layer routing transitions
 * P(expert_next | expert_curr) for consecutive layers L-1 -> L, so the PILOT
 * thread can stage reads for layer L+1 (plus one chained step L+2) the moment
 * layer L's routing is known.  counts[(layer-1)*E*E + e*E + e2] counts how
 * often e2 is routed at `layer` right after e was routed at `layer-1`
 * (callers feed decode-only single-token forwards).
 *
 * DATA-STAGING ONLY: the table influences which expert read is speculated,
 * never routing, weights, logits or KV.  A zeroed (never initialised) struct
 * is safe: enabled == 0 and every accessor degrades to "nothing to predict".
 *
 * Size: (nlayer-1)*E*E uint16.  OLMoE (16 layers x 64 experts) is
 * 15*4096 = 61440 entries = 122,880 bytes.  Allocation is refused (the
 * struct stays disabled) above TIER_MARKOV_MAX_ENTRIES (4M entries = 8 MB)
 * or for nexpert > TIER_MARKOV_MAX_EXPERTS, so small-RAM budgets are
 * honoured.  All hot paths are O(E) or better per lookup.
 * ------------------------------------------------------------------------ */

#define TIER_MARKOV_MAX_ENTRIES ((size_t)4194304)  /* 4M uint16 = 8 MB cap */
#define TIER_MARKOV_MAX_EXPERTS 512
#define TIER_MARKOV_TOP_MAX     8                  /* top-k selection bound */

typedef struct {
    uint16_t *counts;      /* [(nlayer-1) * nexpert * nexpert], saturating */
    int nlayer, nexpert;
    int decay_every;       /* observations between halvings; 0 = never */
    uint64_t tokens;       /* observe() calls since init (decay clock) */
    int enabled;
} TierMarkov;

static inline void tier_markov_free(TierMarkov *mk){
    if(!mk) return;
    free(mk->counts);
    memset(mk, 0, sizeof(*mk));
}

/* Returns 0 on success.  On refusal (bad dims, table over the cap, OOM) the
 * struct is left zeroed and disabled and -1 is returned: callers keep the
 * feature off, they never get a half-built table. */
static inline int tier_markov_init(TierMarkov *mk, int nlayer, int nexpert,
                                   int decay_every){
    if(!mk) return -1;
    memset(mk, 0, sizeof(*mk));
    if(nlayer < 2 || nexpert < 1 || nexpert > TIER_MARKOV_MAX_EXPERTS)
        return -1;
    size_t n = (size_t)(nlayer - 1) * (size_t)nexpert * (size_t)nexpert;
    if(n > TIER_MARKOV_MAX_ENTRIES) return -1;
    mk->counts = calloc(n, sizeof(uint16_t));
    if(!mk->counts) return -1;
    mk->nlayer = nlayer;
    mk->nexpert = nexpert;
    mk->decay_every = decay_every > 0 ? decay_every : 0;
    mk->tokens = 0;
    mk->enabled = 1;
    return 0;
}

/* One decode transition: layer L (>= 1) just routed cur_eids[0..k), and
 * prev_bits holds the routed set of layer L-1 as a (nexpert+7)/8-byte bitmap.
 * Increments counts[(layer-1)*E*E + e*E + e2] for every prev-routed e and
 * every current e2, saturating at 65535.  O(k * popcount(prev_bits)). */
static inline void tier_markov_observe(TierMarkov *mk, int layer,
                                       const uint8_t *prev_bits,
                                       const int *cur_eids, int k){
    if(!mk || !mk->enabled || !prev_bits || !cur_eids) return;
    if(layer < 1 || layer >= mk->nlayer) return;
    int E = mk->nexpert;
    uint16_t *base = mk->counts +
                     (size_t)(layer - 1) * (size_t)E * (size_t)E;
    int words = (E + 7) / 8;
    for(int w = 0; w < words; w++){
        unsigned b = prev_bits[w];
        while(b){
            int bit = 0;
#if defined(__GNUC__) || defined(__clang__)
            bit = __builtin_ctz(b);
#else
            { unsigned t = b; while(!(t & 1u)){ t >>= 1; bit++; } }
#endif
            int e = (w << 3) | bit;
            b &= b - 1;
            if(e >= E) continue;               /* tail padding bits */
            uint16_t *row = base + (size_t)e * (size_t)E;
            for(int i = 0; i < k; i++){
                int e2 = cur_eids[i];
                if(e2 < 0 || e2 >= E) continue;
                if(row[e2] < 0xFFFFu) row[e2]++;
            }
        }
    }
    mk->tokens++;
}

/* Top-`top` successor ids by count for (layer-1 -> layer, expert e), highest
 * first; returns how many were found (zero counts are skipped, ties keep the
 * lower eid).  One O(nexpert * top) insertion pass, top clamped to
 * TIER_MARKOV_TOP_MAX.  Returns 0 when the struct is inert. */
static inline int tier_markov_top(const TierMarkov *mk, int layer, int e,
                                  int *out, int top){
    if(!mk || !mk->enabled || !out || top < 1) return 0;
    if(layer < 1 || layer >= mk->nlayer || e < 0 || e >= mk->nexpert) return 0;
    if(top > TIER_MARKOV_TOP_MAX) top = TIER_MARKOV_TOP_MAX;
    const uint16_t *row = mk->counts +
        ((size_t)(layer - 1) * (size_t)mk->nexpert + (size_t)e) *
        (size_t)mk->nexpert;
    uint16_t vals[TIER_MARKOV_TOP_MAX] = {0};
    int n = 0;
    for(int e2 = 0; e2 < mk->nexpert; e2++){
        uint16_t v = row[e2];
        if(!v) continue;
        int pos = n;
        while(pos > 0 && vals[pos - 1] < v) pos--;
        if(pos >= top) continue;               /* outside the current top */
        int last = n < top ? n : top - 1;
        for(int j = last; j > pos; j--){ vals[j] = vals[j-1]; out[j] = out[j-1]; }
        vals[pos] = v; out[pos] = e2;
        if(n < top) n++;
    }
    return n;
}

/* Halve every counter: stale transitions fade (a decayed 1 becomes 0).
 * Callers run it every `decay_every` observations; O(table). */
static inline void tier_markov_decay(TierMarkov *mk){
    if(!mk || !mk->enabled) return;
    size_t n = (size_t)(mk->nlayer - 1) * (size_t)mk->nexpert *
               (size_t)mk->nexpert;
    for(size_t i = 0; i < n; i++) mk->counts[i] >>= 1;
}

#endif
