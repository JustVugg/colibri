/* Corpus draft source (COLI_DRAFT_CORPUS): the proposal lookup must be a pure
 * suffix match over the frozen ids, and it must never propose anything the
 * verifier would then have to reject for a structural reason.
 *
 * Why this test exists: corpus_draft() is the only NEW way tokens can enter
 * spec_decode's draft buffer. The verification path downstream is unchanged and
 * already lossless, but that guarantee only holds if the proposals themselves
 * are well-formed: within bounds, never past a span separator (-1), never
 * longer than the caller's cap, and empty when no suffix matches. A malformed
 * proposal would index draft[]/batch[] out of range in spec_decode long before
 * verification ever ran.
 *
 * The contract under test:
 *   1. longest-suffix wins: an n=8 match is preferred over an n=3 one
 *   2. most-recent-first: with two matches of equal length, the later corpus
 *      occurrence is proposed (recency = the freshest continuation)
 *   3. the -1 span separator terminates a proposal (spans never bleed together)
 *   4. g <= k always, and g == 0 when the context suffix is absent
 *   5. short contexts (< minn) and empty corpora propose nothing
 *
 * In-memory only (no scratch files, no model), so it builds clean on the
 * Windows MinGW CI job. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int g_nfails = 0;

static void check(int cond, const char *what) {
    if (!cond) {
        printf("FAIL: %s\n", what);
        g_nfails++;
    }
}

/* corpus_draft reads the file-scope g_corp/g_corp_n; set them directly so the
 * test needs no filesystem (corpus_load's parsing is exercised end-to-end by
 * the engine itself, this is about the lookup). */
static void set_corpus(int *ids, long n) {
    g_corp = ids;
    g_corp_n = n;
}

int main(void) {
    int out[64];

    /* 1. exact suffix match proposes the continuation, longest suffix wins.
     * corpus: ... 10 11 12 13 [14 15 16] ... and a decoy short match of "13". */
    {
        static int c[] = {90, 91, 13, 77, 78, 10, 11, 12, 13, 14, 15, 16, 17};
        set_corpus(c, (long)(sizeof(c) / sizeof(c[0])));
        int ctx[] = {1, 2, 3, 10, 11, 12, 13}; /* 4-long suffix 10 11 12 13 */
        int g = corpus_draft(ctx, 7, out, 4, 3, 8);
        check(g == 4, "longest-suffix match proposes k tokens");
        check(g >= 1 && out[0] == 14, "proposal continues the matched span (14)");
        if (g == 4) check(out[1] == 15 && out[2] == 16 && out[3] == 17, "proposal is contiguous");
        /* the decoy (13 at index 2) must NOT win: its continuation is 77 */
        check(!(g >= 1 && out[0] == 77), "longer suffix beats the shorter decoy");
    }

    /* 2. recency: two identical 3-long matches, the later one is proposed. */
    {
        static int c[] = {7, 8, 9, 100, 101, 7, 8, 9, 200, 201};
        set_corpus(c, (long)(sizeof(c) / sizeof(c[0])));
        int ctx[] = {0, 7, 8, 9};
        int g = corpus_draft(ctx, 4, out, 2, 3, 8);
        check(g == 2, "recency case proposes k tokens");
        check(g >= 1 && out[0] == 200, "most recent occurrence wins (200, not 100)");
    }

    /* 3. the span separator (-1) stops a proposal: no bleed across spans. */
    {
        static int c[] = {5, 6, 7, 42, -1, 99, 98};
        set_corpus(c, (long)(sizeof(c) / sizeof(c[0])));
        int ctx[] = {4, 5, 6, 7};
        int g = corpus_draft(ctx, 4, out, 8, 3, 8);
        check(g == 1, "proposal stops at the -1 span separator");
        check(g >= 1 && out[0] == 42, "the pre-separator token is still proposed");
    }

    /* 4. cap and absence. */
    {
        static int c[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        set_corpus(c, (long)(sizeof(c) / sizeof(c[0])));
        int ctx[] = {1, 2, 3};
        int g = corpus_draft(ctx, 3, out, 3, 3, 8);
        check(g <= 3, "proposal never exceeds the caller's cap");
        int ctx2[] = {555, 556, 557};
        check(corpus_draft(ctx2, 3, out, 8, 3, 8) == 0, "absent suffix proposes nothing");
    }

    /* 5. degenerate inputs are silent, not crashes. */
    {
        static int c[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        set_corpus(c, (long)(sizeof(c) / sizeof(c[0])));
        int ctx[] = {1, 2};
        check(corpus_draft(ctx, 2, out, 8, 3, 8) == 0, "context shorter than minn proposes nothing");
        check(corpus_draft(ctx, 2, out, 0, 3, 8) == 0, "k=0 proposes nothing");
        set_corpus(NULL, 0);
        int ctx3[] = {1, 2, 3, 4};
        check(corpus_draft(ctx3, 4, out, 8, 3, 8) == 0, "empty corpus proposes nothing");
    }

    set_corpus(NULL, 0); /* don't free static storage */
    if (g_nfails) {
        printf("corpus_draft: %d FAILED\n", g_nfails);
        return 1;
    }
    printf("corpus_draft: suffix lookup, recency, span separator, caps ok\n");
    return 0;
}
