/* tools/wex4_verify on small synthetic WEX4 containers (the in-place upgrade of a WEX1 file,
 * tests/wex4_fixture.h): an intact file passes and prints the root the fixture computed; a byte
 * changed in a record or in a blob fails with 1; a changed index or header page, a WEX1 file and a
 * truncated file do not open (2); a record hash changed in the index and resealed (a new root)
 * fails on that record. One thread and four give the same verdicts. */
#define WEX4_VERIFY_NO_MAIN
#include "../tools/wex4_verify.c"
#include "wex4_fixture.h"
#include <assert.h>

enum { LAYERS = 2, EXPERTS = 3, ROWS = 40, SCALE_COLS = 4, WEIGHT_COLS = 64 };
#define SLOTS (LAYERS * EXPERTS)
#define RAW (3u * ROWS * (SCALE_COLS + WEIGHT_COLS))   /* 8160 bytes: two pages */

static const char *path = "test_wex4_verify.tmp.wex";

static void put(const unsigned char *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    assert(f && fwrite(data, 1, bytes, f) == bytes && fclose(f) == 0);
}

/* A WEX1 file: header page, table at 4096, then the records in slot order. */
static unsigned char *wex1(size_t *bytes) {
    size_t table = (size_t)wex4_align(SLOTS * sizeof(WexEntry)), record = (size_t)wex4_align(RAW);
    size_t total = WEX4_PAGE + table + SLOTS * record;
    unsigned char *out = calloc(1, total);
    uint32_t seed = 12345u;
    assert(out);
    WexHeader *h = (WexHeader *)out;
    h->magic = WEX_MAGIC_V1; h->version = 1; h->num_layers = LAYERS; h->num_experts = EXPERTS;
    h->aligned_expert_bytes = record; h->table_offset = WEX4_PAGE;
    WexEntry *e = (WexEntry *)(out + WEX4_PAGE);
    for (int i = 0; i < SLOTS; i++) {
        e[i].file_offset = WEX4_PAGE + table + (uint64_t)i * record;
        e[i].total_bytes = (uint32_t)record;
        e[i].scale_bytes = 3u * ROWS * SCALE_COLS; e[i].weight_bytes = 3u * ROWS * WEIGHT_COLS;
        for (int m = 0; m < 3; m++) {
            e[i].scale_shapes[m][0] = e[i].weight_shapes[m][0] = ROWS;
            e[i].scale_shapes[m][1] = SCALE_COLS; e[i].weight_shapes[m][1] = WEIGHT_COLS;
            e[i].scale_offsets[m] = (uint32_t)(m * ROWS * SCALE_COLS);
            e[i].weight_offsets[m] = (uint32_t)(3u * ROWS * SCALE_COLS + m * ROWS * WEIGHT_COLS);
        }
        for (uint32_t j = 0; j < RAW; j++) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            out[e[i].file_offset + j] = (unsigned char)seed;
        }
    }
    *bytes = total;
    return out;
}

static int run(int threads, char root[65]) { return wex4_verify_file(path, threads, root, stdout); }

int main(void) {
    size_t old_bytes, bytes;
    char root[65], got[65], other[65];
    unsigned char *old = wex1(&old_bytes), *w4 = wex4_fixture_from_wex1(old, old_bytes, &bytes, root);
    const WexEntry *e = (const WexEntry *)(old + WEX4_PAGE);
    uint64_t index_offset;
    memcpy(&index_offset, w4 + 32, 8);
    setbuf(stdout, NULL);

    for (int threads = 1; threads <= 4; threads += 3) {
        put(w4, bytes);
        assert(run(threads, got) == 0 && !strcmp(got, root));   /* intact: the root to pin */

        w4[e[4].file_offset + RAW - 1] ^= 1u;                  /* one byte of record 4 */
        put(w4, bytes);
        assert(run(threads, got) == 1 && !strcmp(got, root));   /* the index is fine: same root, record fails */
        w4[e[4].file_offset + RAW - 1] ^= 1u;

        w4[e[2].file_offset + RAW] ^= 1u;                      /* record 2's zero padding */
        put(w4, bytes);
        assert(run(threads, got) == 1);
        w4[e[2].file_offset + RAW] ^= 1u;

        w4[WEX4_PAGE + 5] ^= 1u;                               /* the old table (blob kind 17) */
        put(w4, bytes);
        assert(run(threads, got) == 1);
        w4[WEX4_PAGE + 5] ^= 1u;

        w4[index_offset + 70] ^= 1u;                           /* the index, not resealed */
        put(w4, bytes);
        assert(run(threads, got) == 2 && !got[0]);
        w4[index_offset + 70] ^= 1u;

        w4[60] ^= 1u;                                          /* the header page */
        put(w4, bytes);
        assert(run(threads, got) == 2 && !got[0]);
        w4[60] ^= 1u;

        put(w4, bytes - WEX4_PAGE);                            /* truncated */
        assert(run(threads, got) == 2);
        put(old, old_bytes);                                   /* the WEX1 form: no root */
        assert(run(threads, got) == 2 && !got[0]);

        /* record 1's SHA-256 in the index changed and resealed: a valid file with another root whose
         * record 1 does not match it */
        unsigned char *copy = malloc(bytes);
        assert(copy);
        memcpy(copy, w4, bytes);
        copy[index_offset + 2u * WEX4_BLOB_BYTES + WEX4_RECORD_BYTES + 36u] ^= 1u;
        wex4_fixture_reseal(copy, other);
        put(copy, bytes);
        assert(run(threads, got) == 1 && !strcmp(got, other) && strcmp(other, root));
        free(copy);
    }
    remove(path);
    free(old); free(w4);
    puts("test_wex4_verify: intact file passes with the fixture's root; changed record, padding and blob bytes fail; "
         "damaged index / header, truncated and WEX1 files do not open; a resealed wrong record hash fails (1 and 4 threads)");
    return 0;
}
