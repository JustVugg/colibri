/* test_kv_prefix — the reuse decision, exhaustively.
 *
 * This is the piece that must not be wrong. A bad reuse length does not crash:
 * it answers the user from an attention state built out of a different
 * conversation, and the output looks like a plausible reply. So every rejection
 * rule gets its own case here, including the ones that look paranoid.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../kv_prefix.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

int main(void) {
    kv_prefix p = {0};
    CHECK(kv_prefix_alloc(&p, 16), "alloc failed");

    const int turn1[] = {5, 6, 7};
    const int turn2[] = {5, 6, 7, 8, 9};          /* extends turn1 */
    const int other[] = {5, 6, 99, 8, 9};         /* diverges at index 2 */

    /* nothing recorded yet */
    CHECK(kv_prefix_reuse(&p, turn2, 5) == 0, "empty record must not be reused");

    kv_prefix_record(&p, turn1, 0, 3);
    CHECK(p.len == 3, "len=%d, want 3", p.len);
    CHECK(kv_prefix_reuse(&p, turn2, 5) == 3, "an extending prompt reuses all 3");
    CHECK(kv_prefix_reuse(&p, other, 5) == 0, "a diverging prompt reuses nothing");

    /* Equal length: reusing everything would leave no token to prefill, so the
     * caller would have no final hidden state to sample from. */
    CHECK(kv_prefix_reuse(&p, turn1, 3) == 0, "equal-length prompt must not reuse");

    /* Shorter prompt: the state holds MORE than asked for and cannot be
     * rewound. Reusing 3 positions for a 2-token prompt would answer with a
     * token the user deleted still in context. */
    const int shorter[] = {5, 6};
    CHECK(kv_prefix_reuse(&p, shorter, 2) == 0, "shorter prompt must not reuse");

    /* Divergence at the very first token. */
    const int first[] = {1, 6, 7, 8};
    CHECK(kv_prefix_reuse(&p, first, 4) == 0, "divergence at 0 reuses nothing");

    /* Divergence at the LAST recorded token — the boundary a memcmp length
     * that is off by one would let through. */
    const int last[] = {5, 6, 70, 8};
    CHECK(kv_prefix_reuse(&p, last, 4) == 0, "divergence at len-1 reuses nothing");

    /* Appending: the record must describe the whole state, not the last write. */
    const int gen[] = {8, 9};
    kv_prefix_record(&p, gen, 3, 2);
    CHECK(p.len == 5, "len=%d after append, want 5", p.len);
    const int turn3[] = {5, 6, 7, 8, 9, 10};
    CHECK(kv_prefix_reuse(&p, turn3, 6) == 5, "reuse must cover prompt+generated");

    /* Taint: same ids, different payload (Inkling audio). */
    kv_prefix_taint(&p);
    CHECK(kv_prefix_reuse(&p, turn3, 6) == 0, "a tainted state is never reused");
    kv_prefix_clear(&p);
    CHECK(p.tainted == 0 && p.len == 0, "clear must drop both len and taint");

    /* A write past the end drops the record instead of truncating it: claiming
     * coverage the state does not have is the one failure that answers wrongly. */
    CHECK(kv_prefix_alloc(&p, 4), "realloc failed");
    kv_prefix_record(&p, turn1, 0, 3);
    const int spill[] = {1, 2, 3};
    kv_prefix_record(&p, spill, 3, 3);            /* 3+3 > cap 4 */
    CHECK(p.len == 0, "overflowing write must drop the record, len=%d", p.len);
    CHECK(kv_prefix_reuse(&p, turn2, 5) == 0, "dropped record must not be reused");

    /* kv_prefix_alloc RESTARTS: for callers that discard the KV outright. */
    CHECK(kv_prefix_alloc(&p, 32), "alloc failed");
    CHECK(p.len == 0 && p.tainted == 0, "alloc must reset the record");

    /* kv_prefix_grow PRESERVES: for callers that grow the KV by copying it.
     * This is the case CI caught — inkling's kv_alloc re-allocated on every
     * longer prompt, so reuse could never fire in a growing conversation, the
     * one situation it exists for. */
    kv_prefix_record(&p, turn1, 0, 3);
    kv_prefix_record(&p, gen, 3, 2);                       /* 5 positions held */
    CHECK(kv_prefix_grow(&p, 64, 5), "grow failed");
    CHECK(p.cap == 64, "cap=%d after grow, want 64", p.cap);
    CHECK(p.len == 5, "len=%d after grow, want 5 preserved", p.len);
    CHECK(kv_prefix_reuse(&p, turn3, 6) == 5,
          "a preserved record must still be reusable after growing");

    /* keep is clamped by what is actually held: a caller that copied FEWER
     * positions than it claims must not end up describing uninitialised ones. */
    CHECK(kv_prefix_grow(&p, 128, 999), "grow failed");
    CHECK(p.len == 5, "keep must clamp to len, got %d", p.len);

    /* keep=0 grows without preserving: the caller discarded the KV contents. */
    CHECK(kv_prefix_grow(&p, 256, 0), "grow failed");
    CHECK(p.len == 0, "keep=0 must leave the record empty, got %d", p.len);
    CHECK(kv_prefix_reuse(&p, turn3, 6) == 0, "an emptied record reuses nothing");

    /* Shrinking below what is held keeps only what fits, never past the end. */
    kv_prefix_record(&p, turn1, 0, 3);
    CHECK(kv_prefix_grow(&p, 2, 3), "grow failed");
    CHECK(p.len <= 2 && p.cap == 2, "len=%d cap=%d must stay within cap",
          p.len, p.cap);

    /* Defensive: a NULL record is inert, not a crash. Engines disable reuse by
     * leaving fed==NULL when the allocation fails. */
    kv_prefix nul = {0};
    CHECK(kv_prefix_reuse(&nul, turn2, 5) == 0, "NULL record reuses nothing");
    kv_prefix_record(&nul, turn1, 0, 3);          /* must not crash */
    kv_prefix_clear(&nul);
    kv_prefix_free(&nul);

    kv_prefix_free(&p);
    CHECK(p.fed == NULL && p.cap == 0, "free must clear the record");

    if (failures) { fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    puts("kv_prefix: ok");
    return 0;
}
