/* kv_prefix.h — reuse the attention state a previous turn already built.
 *
 * A chat client resends the whole transcript every turn. colibri.c has pinned
 * each conversation to a KV slot since #639, so its turn N prefills only the
 * new text; inkling.c, kimi_k3.c and deepseek were written without it and
 * re-processed turns 1..N-1 from scratch. The cost of a message therefore grew
 * with the length of the conversation, and every replayed position pulled its
 * experts off disk again — on a streaming engine that is the dominant cost, not
 * a rounding error. Measured on DeepSeek V4, a second turn that reused 82% of
 * its prompt took 61s instead of 320s.
 *
 * THE IDEA. After a turn, the engine's state covers some number of positions.
 * If the next prompt BEGINS with the token sequence that produced them, that
 * state already IS the state at that position: skip the reset and prefill only
 * the tail. No snapshot, no rewind — a prompt that diverges anywhere starts
 * over.
 *
 * WHY A RECORD AND NOT A COUNTER. It is tempting to derive the reusable length
 * from the caller's own bookkeeping ("prompt_count + generated - 1"). That
 * invariant differs per engine — whether the last sampled token was fed back,
 * whether a chunked prefill ran to completion, whether generation stopped early
 * — and getting it wrong does not crash: it silently answers from a state that
 * belongs to a different conversation. So the ids are recorded WHERE THEY ARE
 * FED, and the record is the only description of the state anyone consults.
 *
 * INVARIANT: fed[0..len-1] are exactly the token ids the current state was
 * built from, in position order. Everything else follows from it.
 *
 * TAINT. Some inputs are not described by their token ids. Inkling's audio
 * frames all carry the same id (c->audio_tok) while the mel payload differs, so
 * an id-only comparison would cheerfully "match" two different clips. A state
 * that consumed such an input is marked tainted and is never reused.
 */
#ifndef KV_PREFIX_H
#define KV_PREFIX_H

#include <stdlib.h>
#include <string.h>

typedef struct {
    int *fed;      /* token ids at positions 0..len-1 */
    int  len;      /* positions the state currently covers */
    int  cap;      /* allocated positions */
    int  tainted;  /* state consumed something token ids cannot describe */
} kv_prefix;

/* Size the record to the KV it describes. Call it wherever the KV buffers are
 * (re)allocated: growing them discards the positions fed[] refers to, so the
 * record has to be dropped at the same moment. Returns 0 on allocation
 * failure, in which case reuse is simply disabled — this is an optimisation
 * and must never be the reason a turn fails. */
static inline int kv_prefix_alloc(kv_prefix *p, int cap) {
    if (!p) return 0;
    free(p->fed);
    p->fed = (cap > 0) ? (int *)calloc((size_t)cap, sizeof(int)) : NULL;
    p->cap = p->fed ? cap : 0;
    p->len = 0;
    p->tainted = 0;
    return p->fed != NULL;
}

/* Forget the state. Pair this with whatever zeroes the engine's own KV/conv/
 * recurrent buffers, so the two can never disagree. */
static inline void kv_prefix_clear(kv_prefix *p) {
    if (!p) return;
    p->len = 0;
    p->tainted = 0;
}

static inline void kv_prefix_free(kv_prefix *p) {
    if (!p) return;
    free(p->fed);
    p->fed = NULL;
    p->cap = p->len = 0;
    p->tainted = 0;
}

/* Grow the record to `cap`, keeping the first `keep` positions.
 *
 * For engines whose KV buffers are re-allocated when a longer prompt arrives.
 * If those buffers are grown by COPYING their contents, the positions survive
 * and so must the record — otherwise reuse can never fire in the one case it
 * exists for: a conversation whose prompt gets longer every turn. Returns 0 if
 * the record could not be preserved, and leaves it empty rather than stale:
 * the caller must then treat the state as unreusable. */
static inline int kv_prefix_grow(kv_prefix *p, int cap, int keep) {
    if (!p || cap <= 0) return 0;
    int *grown = (int *)calloc((size_t)cap, sizeof(int));
    if (!grown) { kv_prefix_free(p); return 0; }
    if (keep > p->len) keep = p->len;
    if (keep > cap)    keep = cap;
    if (keep > 0 && p->fed) memcpy(grown, p->fed, (size_t)keep * sizeof(int));
    free(p->fed);
    p->fed = grown;
    p->cap = cap;
    p->len = keep > 0 ? keep : 0;
    return 1;
}

/* Record n tokens fed at absolute positions pos0..pos0+n-1. Out-of-range
 * writes drop the record rather than truncating it: a partial record would
 * claim coverage the state does not have. */
static inline void kv_prefix_record(kv_prefix *p, const int *ids,
                                    int pos0, int n) {
    if (!p || !p->fed || !ids || n <= 0 || pos0 < 0) return;
    if (pos0 + n > p->cap) { p->len = 0; return; }
    memcpy(p->fed + pos0, ids, (size_t)n * sizeof(int));
    if (pos0 + n > p->len) p->len = pos0 + n;
}

/* Mark the state as holding something the ids do not describe (see TAINT). */
static inline void kv_prefix_taint(kv_prefix *p) {
    if (p) p->tainted = 1;
}

/* How many leading tokens of this prompt the state already holds: either all
 * `len` of them, or none.
 *
 * Requires at least one NEW token (len < n). A prompt that is a prefix of, or
 * equal to, the recorded sequence would need the state REWOUND, which nothing
 * here can do — and prefilling zero tokens leaves the caller with no final
 * hidden state to sample from. */
static inline int kv_prefix_reuse(const kv_prefix *p, const int *ids, int n) {
    if (!p || !p->fed || !ids) return 0;
    if (p->tainted || p->len <= 0 || p->len >= n) return 0;
    if (memcmp(p->fed, ids, (size_t)p->len * sizeof(int)) != 0) return 0;
    return p->len;
}

#endif /* KV_PREFIX_H */
