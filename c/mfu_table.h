#ifndef COLIBRI_MFU_TABLE_H
#define COLIBRI_MFU_TABLE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MFU popularity table — PLAN_cache Task 1 (§5.1).
 *
 * A flat uint32 counter per expert ([layer * 256 + expert]; 43 layers x 256
 * experts = 11008 counters = 44032 bytes, L1-resident), incremented O(1) per
 * ROUTED selection only — decode tokens and MTP/spec drafts (§10). Prefill
 * batch accesses and PILOT prefetches NEVER call mfu_record(): counting
 * reflects routing demand, nothing else.
 *
 * This table is the SINGLE authority for popularity: reorder ranking, the MFU
 * cutoff, MFU membership, MFU victim selection, boot prefill, and GPU fill
 * all read it.
 *
 * Persistence: mfu_save()/mfu_load() — magic "MFU1" + version +
 * overflow_shift + total_accesses + the 44 KB counts. Load decay is OFF by
 * default (see MFU_DECAY_SHIFT_DEFAULT below): the persisted counts ARE the
 * next boot's warm-fill seed (GPU fill-to-free prefill + CPU pool warm),
 * and truncating the tail at load would starve exactly the preload decision
 * it exists to serve. Opt-in decay (env COLI_DSV4_VK_MFU_DECAY_SHIFT=N>0)
 * remains for A/B — last request's traffic then dominates the ranking and
 * old history only biases.
 * Corrupt/truncated/bad-magic/missing file -> cold start: mfu_load returns
 * -1 and leaves the caller's zeroed (mfu_init'ed) table untouched.
 *
 * delta_counts[] is the per-request decomposition for the reorder ranking
 * (§5.4): incremented alongside counts in mfu_record (same O(1) path),
 * reset to zero at each reorder (mfu_reset_deltas). The reorder ranking is
 * rank_score = delta + (counts - delta) == live counts, so mfu_scale_down
 * shifts BOTH arrays identically and the decomposition stays in lockstep
 * through the (effectively-never-fires) overflow guard — the invariant
 * "rank_score == live count" survives.
 *
 * Precondition of mfu_record: 0 <= layer < 43, 0 <= expert < 256 (router
 * indices; the O(1) hot path is deliberately unchecked, matching the plan).
 */

#define MFU_NUM_LAYERS 43
#define MFU_EXPERTS_PER_LAYER 256
#define MFU_NUM_EXPERTS (MFU_NUM_LAYERS * MFU_EXPERTS_PER_LAYER) /* 11008 */

/* High-water guard: a counter reaches 2^31 after ~8.3M routed tokens
 * (258 selections/token ~= 96 days at this host's ~1 tok/s). Cheap
 * insurance — a uniform divide preserves the ranking. */
#define MFU_OVERFLOW_HIGHWATER (UINT32_C(1) << 31)

/* Decay/guard shift (env COLI_DSV4_VK_MFU_DECAY_SHIFT, default 0 = NO load
 * decay). Applied ONLY by the overflow guard (mfu_scale_down) unless the env
 * opts back in for load too (mfu_load shifts by decay_shift when > 0).
 *
 * v5.3 (2026-09-03): 2 -> 0. The load-decay history: 4 (÷16) -> 2 (÷4,
 * PLAN_gpu_mfu_sizing T6) -> 0. ÷16 and ÷4 were tuned when the GPU-MFU boot
 * prefill was capped small and only needed the decayed table's head; the
 * warm-start consumers now fill to their FULL budget — the GPU tier
 * fill-to-free (down to the 1 GiB spare) and the CPU pool warm (every FREE
 * slab, ~30 GiB here) — and both need the persisted table's long tail. The
 * ÷4 truncation measured 2026-09-03: 3713 nonzero saved -> 1141 survivors,
 * so a 2449-slot pool warmed only 1141 experts (494 GPU + 647 CPU) even
 * though the pool had room for all of them. Decay also compounds per
 * restart (overflow_shift accumulates), not per wall-clock time, so a
 * frequently restarted server erases stable popularity within days.
 * Recency needs no load-side help: the request-boundary reorder re-ranks
 * from LIVE counts every request, so a topic shift self-heals at the first
 * boundary of the new session.
 *
 * The OVERFLOW GUARD is unaffected by the default: mfu_scale_down always
 * shifts by at least 1 when a counter crosses MFU_OVERFLOW_HIGHWATER (the
 * uint32 wrap protection is a correctness invariant, not recency). */
#define MFU_DECAY_SHIFT_DEFAULT 0u

/* Persistence format ("MFU1" magic + version). */
#define MFU_FILE_MAGIC "MFU1"
#define MFU_FILE_VERSION 1u

typedef struct {
    uint32_t counts[MFU_NUM_EXPERTS]; /* [layer*256+expert]; the authority */
    uint64_t total_accesses;          /* routed selections since creation */
    uint32_t overflow_shift;          /* CUMULATIVE >> shifts applied to
                                         counts — overflow guards AND load
                                         decays. Stored so saved data stays
                                         comparable across restarts (a file
                                         that has been through N restarts has
                                         been decayed N times; this field
                                         records the total scale). */
    uint32_t decay_shift;             /* runtime: load-decay + guard shift
                                         (env COLI_DSV4_VK_MFU_DECAY_SHIFT,
                                         default 4). Not persisted. */
    uint32_t delta_counts[MFU_NUM_EXPERTS]; /* per-request decomposition
                                               (§5.4), reset at each reorder */
} MfuTable;

/* Zero the table; decay_shift from COLI_DSV4_VK_MFU_DECAY_SHIFT (default 0 =
 * no load decay; N in 1..31 opts into >>N-at-load, >=32 clamps to no decay). */
void mfu_init(MfuTable *m);

/* Global right-shift of counts[] AND delta_counts[] by m->decay_shift —
 * lockstep so the decomposition survives (§5.4). Bumps overflow_shift.
 * ~µs at 11008 entries; called only by the high-water guard. */
void mfu_scale_down(MfuTable *m);

/* Zero delta_counts[] — call at each reorder boundary (§8 step 1). */
void mfu_reset_deltas(MfuTable *m);

/* Save: magic + version + overflow_shift + total_accesses + counts, written
 * to "<path>.tmp" then renamed over the target. 0 on success, -1 on error. */
int mfu_save(const MfuTable *m, const char *path);

/* Load: validates magic/version/length; on success applies >>decay_shift to
 * the counts (the persistence round-trip is the decay) and bumps
 * overflow_shift by the same amount; delta_counts start zeroed. Returns 0 on
 * success, -1 on any failure (cold start — the caller's table is untouched). */
int mfu_load(MfuTable *m, const char *path);

/* O(1) record for a ROUTED selection (decode + MTP/spec drafts ONLY — §10). */
static inline void mfu_record(MfuTable *m, int layer, int expert) {
    size_t idx = (size_t)layer * MFU_EXPERTS_PER_LAYER + (size_t)expert;
    uint32_t *c = &m->counts[idx];
    if (*c < MFU_OVERFLOW_HIGHWATER) { /* high-water guard: rare branch */
        *c += 1;
        m->total_accesses += 1;
    } else {
        mfu_scale_down(m);             /* global >>decay_shift, ~µs */
        *c = 1;
        m->total_accesses += 1;
    }
    m->delta_counts[idx] += 1;         /* per-request decomposition, lockstep */
}

static inline uint32_t mfu_count(const MfuTable *m, int layer, int expert) {
    return m->counts[(size_t)layer * MFU_EXPERTS_PER_LAYER + (size_t)expert];
}

static inline uint32_t mfu_delta(const MfuTable *m, int layer, int expert) {
    return m->delta_counts[(size_t)layer * MFU_EXPERTS_PER_LAYER + (size_t)expert];
}

#ifdef __cplusplus
}
#endif
#endif /* COLIBRI_MFU_TABLE_H */
