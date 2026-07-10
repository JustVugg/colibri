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
}

int main(void)
{
    test_rows_use_their_own_sequence_storage();
    test_const_reader_selects_the_same_row();
    test_submit_header();
    puts("decode batch helper tests: ok");
    return 0;
}
