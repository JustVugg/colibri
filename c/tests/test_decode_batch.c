#include <assert.h>
#include <stdio.h>

#include "../decode_batch.h"

static void test_rows_use_their_own_sequence_storage(void)
{
    float sequence_a[4 * 3] = {0};
    float sequence_b[4 * 3] = {0};

    float *a2 = coli_kv_row(sequence_a, 2, 3);
    float *b1 = coli_kv_row(sequence_b, 1, 3);
    a2[0] = 20.0f;
    b1[2] = 12.0f;

    assert(a2 == &sequence_a[6]);
    assert(b1 == &sequence_b[3]);
    assert(sequence_a[6] == 20.0f);
    assert(sequence_b[5] == 12.0f);
    assert(sequence_a[5] == 0.0f);
    assert(sequence_b[6] == 0.0f);
}

static void test_const_reader_selects_the_same_row(void)
{
    float storage[5 * 7] = {0};
    const float *row = coli_kv_row(storage, 4, 7);

    assert(row == &storage[28]);
}

static void test_submit_header(void)
{
    ColiSubmit sub;
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95", &sub));
    assert(sub.id == 42 && sub.slot == 3 && sub.bytes == 17);
    assert(sub.max_tokens == 64 && sub.temperature > .69f && sub.top_p > .94f);
    assert(!coli_submit_parse("SUBMIT 1 -1 2 3 0.7 1", &sub));
    assert(!coli_submit_parse("SUBMIT 1 0 2 0 0.7 1", &sub));
    assert(!coli_submit_parse("SUBMIT 1 0 2 3 4 1", &sub));
    assert(!coli_submit_parse("SUBMIT 0 0 2 3 1 1", &sub));
    assert(!coli_submit_parse("SUBMIT 1 0 2 3 nan 1", &sub));
    assert(!coli_submit_parse("SUBMIT 1 0 2 3 1 inf", &sub));
    assert(coli_submit_parse("SUBMIT 1 0 16777216 3 1 1", &sub));
    assert(!coli_submit_parse("SUBMIT 1 0 16777217 3 1 1", &sub));
    assert(!coli_submit_parse("SUBMIT 1 0 2 3 1 1 trailing", &sub));
    /* optional 7th field: per-request grammar length (0 when absent) */
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95", &sub) && sub.gbytes == 0);
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 512", &sub) && sub.gbytes == 512);
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 1048576", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 1048577", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 512 extra", &sub));
}

/* U7a extension fields: key=value tokens from the 8th field onward. The two
 * legacy arms stay byte-identical in behavior (asserted above); the third arm
 * accepts only known keys with in-range values, so an unknown or malformed
 * extension rejects the whole frame exactly like a trailing garbage field. */
static void test_submit_extension_fields(void)
{
    ColiSubmit sub;
    /* legacy headers leave the new capabilities off */
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95", &sub));
    assert(sub.logprobs == 0 && sub.tok_ids == 0);
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 512", &sub));
    assert(sub.logprobs == 0 && sub.tok_ids == 0);
    /* the 8-field opt-in forms */
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=5", &sub));
    assert(sub.logprobs == 5 && sub.tok_ids == 0 && sub.gbytes == 0);
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 ids=1", &sub));
    assert(sub.tok_ids == 1 && sub.logprobs == 0);
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 512 logprobs=32 ids=1", &sub));
    assert(sub.logprobs == 32 && sub.tok_ids == 1 && sub.gbytes == 512);
    assert(coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=0 ids=0", &sub));
    assert(sub.logprobs == 0 && sub.tok_ids == 0);
    /* rejects: over-cap, out-of-range, unknown key, malformed token */
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=33", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 ids=2", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 seed=7", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 =5", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=5x", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=5 junk", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs", &sub));
    /* glued numeric field + key: one malformed field, NOT "512" + an opt-in
     * (%llu would otherwise stop at the first letter and silently split it) */
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 512logprobs=5", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0logprobs=5", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0ids=1", &sub));
    /* duplicate keys: rejected outright, never silently last-wins */
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=3 logprobs=5", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 ids=1 ids=1", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 ids=0 ids=0", &sub));
    /* overflowing values: rejected by the bounded digit loop, no sscanf %llu
     * ever sees the value (overflow there is UB; a wrapping libc would fold
     * 2^64+5 back to an in-range 5) */
    assert(!coli_submit_parse(
        "SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=18446744073709551621", &sub));
    assert(!coli_submit_parse(
        "SUBMIT 42 3 17 64 0.7 0.95 0 ids=18446744073709551617", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=329", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 0 logprobs=1000", &sub));
    /* an extended header still validates the base fields */
    assert(!coli_submit_parse("SUBMIT 0 3 17 64 0.7 0.95 0 logprobs=5", &sub));
    assert(!coli_submit_parse("SUBMIT 42 3 17 0 0.7 0.95 0 logprobs=5", &sub));
}

/* U7a token-ID intake: the ids the caller formatted are exactly the ids that
 * come back out (the zero-round-trip property) -- and malformed input never
 * parses partially. */
static void test_ids_parse_round_trip(void)
{
    /* fixed known-good example: format then parse must reproduce the array */
    const int ids[6] = {0, 154822, 154824, 12, 7, 99999};
    char buf[128];
    int out[8], n, k;
    n = snprintf(buf, sizeof(buf), "%d %d %d %d %d %d",
                 ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
    assert(n > 0);
    k = coli_ids_parse(buf, (size_t)n, out, 8, 160000);
    assert(k == 6);
    for (int i = 0; i < 6; i++) assert(out[i] == ids[i]);

    /* whitespace forms: leading/trailing/newlines are all separators */
    k = coli_ids_parse("  5\n7\t9  ", 9, out, 8, 100);
    assert(k == 3 && out[0] == 5 && out[1] == 7 && out[2] == 9);

    /* rejects: sign, non-digit, vocab bound (exact and above), garbage tail */
    assert(coli_ids_parse("-1", 2, out, 8, 100) == -1);
    assert(coli_ids_parse("5x", 2, out, 8, 100) == -1);
    assert(coli_ids_parse("1 two 3", 7, out, 8, 100) == -1);
    assert(coli_ids_parse("100", 3, out, 8, 100) == -1);   /* id == vocab */
    assert(coli_ids_parse("99", 2, out, 8, 100) == 1 && out[0] == 99);
    assert(coli_ids_parse("101", 3, out, 8, 100) == -1);
    assert(coli_ids_parse("123456789012345", 15, out, 8, 100) == -1);

    /* empty payload parses to zero ids (the caller's EMPTY_PROMPT arm) */
    assert(coli_ids_parse("   ", 3, out, 8, 100) == 0);

    /* overflow reports the cap so the caller's context check refuses loudly,
     * mirroring tok_encode's stop-at-cap contract (#401) */
    assert(coli_ids_parse("1 2 3 4 5", 9, out, 4, 100) == 4);
    assert(coli_ids_parse("1 2 3 4", 7, out, 4, 100) == 4);
}

int main(void)
{
    test_rows_use_their_own_sequence_storage();
    test_const_reader_selects_the_same_row();
    test_submit_header();
    test_submit_extension_fields();
    test_ids_parse_round_trip();
    puts("decode batch helper tests: ok");
    return 0;
}
