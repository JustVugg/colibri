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

int main(void)
{
    test_rows_use_their_own_sequence_storage();
    test_const_reader_selects_the_same_row();
    puts("decode batch helper tests: ok");
    return 0;
}
