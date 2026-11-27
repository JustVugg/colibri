/* tools/wex4_verify on small synthetic WEX4 containers (the in-place upgrade of a WEX1 file,
 * tests/wex4_fixture.h): an intact file passes and prints the root the fixture computed; a byte
 * changed in a record or in a blob fails with 1; a changed index or header page, a WEX1 file and a
 * truncated file do not open (2); a record hash changed in the index and resealed (a new root)
 * fails on that record. One thread and four give the same verdicts.
 *
 * Then wex4.h's own rules, which a resealed index has to pass: fields of the index or header page
 * changed and both hashes recomputed (as Wex-WRN2's conformance vectors are made), each refused
 * with the status of the rule it breaks; a record of no expert is accepted. And the parts of
 * wex4.h the engines do not call: wex4_find, wex4_decode_record on a RAW record, and a WBP1
 * (codec 6) record through wex4_bp_encode and wex4_bp_decode, refused under another owner or
 * with a byte changed. */
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

/* wex4_parse_header, then wex4_open, on a file held in memory: 0 or the status of the first rule broken */
static int open_in_memory(const unsigned char *w4, size_t bytes, Wex4View *v) {
    Wex4Header h;
    int rc = wex4_parse_header(w4, bytes, &h);
    return rc ? rc : wex4_open(&h, w4 + h.index_offset, (size_t)wex4_align(h.index_bytes), v);
}

/* One or two fields of the index or header page changed (little-endian, `width` bytes) and both
 * hashes resealed, so that the change reaches the index's rules. */
static void index_rules(const unsigned char *w4, size_t bytes) {
    uint64_t index_offset, index_bytes, rec0, total, nbytes5;
    uint32_t order0, owner0, maximum;
    memcpy(&index_offset, w4 + 32, 8); memcpy(&index_bytes, w4 + 40, 8);
    assert(index_bytes % WEX4_PAGE);   /* the index ends inside its last page: a byte after it to change */
    const size_t Q = WEX4_RECORD_BYTES, Z = WEX4_TENSOR_BYTES, R = (size_t)index_offset + 2u * WEX4_BLOB_BYTES,
                 T = R + SLOTS * Q, O = T + SLOTS * 6u * Z, N = O + SLOTS * 6u * 4u;
    memcpy(&rec0, w4 + R, 8); memcpy(&owner0, w4 + R + 24, 4); memcpy(&order0, w4 + O, 4);
    memcpy(&total, w4 + 88, 8); memcpy(&maximum, w4 + 68, 4); memcpy(&nbytes5, w4 + T + 5 * Z + 88, 8);
    const struct { const char *what; size_t at[2]; unsigned width[2]; uint64_t value[2]; int want; } cases[] = {
        {"resealed unchanged", {0, 0}, {0, 0}, {0, 0}, WEX4_OK},
        {"the last record of no expert, its owner its index", {R + 5 * Q + 18, R + 5 * Q + 24}, {2, 4}, {0, 5}, WEX4_OK},
        {"a record of no expert whose owner is not its index", {R + 5 * Q + 18, R + 5 * Q + 24}, {2, 4}, {0, 4}, WEX4_E_RECORD},
        {"two records at one offset", {R + Q, 0}, {8, 0}, {rec0, 0}, WEX4_E_LAYOUT},
        {"a nonzero byte after the index", {(size_t)(index_offset + index_bytes), 0}, {1, 0}, {1, 0}, WEX4_E_INDEX},
        {"a name twice in the name order", {O + 4, 0}, {4, 0}, {order0, 0}, WEX4_E_NAME},
        {"a name that is not UTF-8", {N, 0}, {1, 0}, {0xff, 0}, WEX4_E_NAME},
        {"a shape whose product overflows", {T + 8, 0}, {8, 0}, {UINT64_C(1) << 62, 0}, WEX4_E_TENSOR},
        {"rank 9", {T + 7, 0}, {1, 0}, {9, 0}, WEX4_E_TENSOR},
        {"two tensors at one offset", {T + Z + 80, 0}, {8, 0}, {0, 0}, WEX4_E_TENSOR},
        {"a size that is not the shape's", {T + 5 * Z + 88, 0}, {8, 0}, {nbytes5 + 1, 0}, WEX4_E_TENSOR},
        {"a tensor past its record's end", {T + 5 * Z + 8, T + 5 * Z + 88}, {8, 8}, {ROWS + 1, nbytes5 + WEIGHT_COLS},
         WEX4_E_TENSOR},
        {"an unknown record flag", {R + 18, 0}, {2, 0}, {WEX4_REC_EXPERT | 4u, 0}, WEX4_E_RECORD},
        {"a value-exact RAW record", {R + 18, 0}, {2, 0}, {WEX4_REC_EXPERT | WEX4_REC_VALUE_EXACT, 0}, WEX4_E_RECORD},
        {"a RAW record with a blob", {R + 20, 0}, {4, 0}, {1, 0}, WEX4_E_RECORD},
        {"stored bytes not whole pages", {R + 8, 0}, {4, 0}, {8191, 0}, WEX4_E_RECORD},
        {"a record with more tensors than are left", {R + 5 * Q + 32, 0}, {4, 0}, {7, 0}, WEX4_E_RECORD},
        {"a coded (WBP1) record no smaller than RAW", {R + 16, 0}, {2, 0}, {WEX4_CODEC_WBP1, 0}, WEX4_E_RECORD},
        {"an expert record of codec 7", {R + 16, R + 8}, {2, 4}, {7, WEX4_PAGE}, WEX4_E_EXPERT},
        {"expert owners not increasing", {R + Q + 24, 0}, {4, 0}, {owner0, 0}, WEX4_E_EXPERT},
        {"an expert owner outside the grid", {R + 24, 0}, {4, 0}, {(uint64_t)LAYERS << 16, 0}, WEX4_E_EXPERT},
        {"total_decoded one byte more", {88, 0}, {8, 0}, {total + 1, 0}, WEX4_E_RECORD},
        {"max_decoded one byte less", {68, 0}, {4, 0}, {maximum - 1u, 0}, WEX4_E_RECORD},
        {"a WNM2 blob of another size", {(size_t)index_offset, 0}, {4, 0}, {WEX4_BLOB_WNM2, 0}, WEX4_E_BLOB},
    };
    unsigned char *c = malloc(bytes);
    int bad = 0;
    assert(c);
    for (size_t k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        char root[65];
        Wex4View v;
        memcpy(c, w4, bytes);
        for (int j = 0; j < 2; j++) if (cases[k].width[j]) memcpy(c + cases[k].at[j], &cases[k].value[j], cases[k].width[j]);
        wex4_fixture_reseal(c, root);
        int got = open_in_memory(c, bytes, &v);
        printf("%-4s %s: wex4.h %d\n", got == cases[k].want ? "ok" : "FAIL", cases[k].what, got);
        bad += got != cases[k].want;
    }
    free(c);
    assert(!bad);
}

/* wex4_find and wex4_decode_record on the fixture's records; a WBP1 record of bf16-like data. */
static void unused_by_engines(const unsigned char *w4, size_t bytes) {
    Wex4View v;
    Wex4Record rec;
    Wex4Tensor t;
    const char *name = "layers.1.ffn.experts.0.w2.weight";   /* record 3's fifth tensor */
    unsigned char *out = malloc(RAW);
    assert(out && !open_in_memory(w4, bytes, &v));
    int64_t at = wex4_find(&v, (const unsigned char *)name, strlen(name));
    assert(at == 3 * 6 + 4);
    wex4_get_tensor(&v, (uint32_t)at, &t);
    assert(t.record == 3 && t.name_len == strlen(name) && !memcmp(t.name, name, t.name_len));
    assert(wex4_find(&v, (const unsigned char *)"layers.1.ffn.experts.0.w2", 25) == -1);
    wex4_get_record(&v, 3, &rec);
    assert(!wex4_decode_record(&v, 3, w4 + rec.offset, out, NULL, 1) && !memcmp(out, w4 + rec.offset, RAW));
    assert(wex4_decode_record(&v, 2, w4 + rec.offset, out, NULL, 1) == WEX4_E_HASH);   /* another record's bytes */
    free(out);

    enum { N = 24576 };
    const uint32_t owner = 0x00010002u;
    unsigned char *data = malloc(N), *coded = calloc(1, 2 * N + 65536), *back = malloc(N);
    size_t written = 0;
    uint32_t seed = 777u;
    assert(data && coded && back);
    for (uint32_t i = 0; i < N; i += 2) {   /* bf16 values near 1.0 */
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        uint16_t value = (uint16_t)(0x3f80u + (seed & 0x3fu) - 0x20u);
        memcpy(data + i, &value, 2);
    }
    assert(!wex4_bp_encode(data, N, 2, 12, 1, owner, coded, 2 * N + 65536, &written) && written && !(written % WEX4_PAGE));
    assert(!wex4_bp_decode(coded, written, N, owner, back) && !memcmp(back, data, N));
    assert(wex4_bp_decode(coded, written, N, owner + 1u, back));   /* the record names another owner */
    assert(wex4_bp_decode(coded, written, N - 2, owner, back));     /* or another size */
    coded[WEX4_BP_HEADER + 600] ^= 0x10u;                           /* a byte of the first plane's streams */
    assert(wex4_bp_decode(coded, written, N, owner, back));
    free(data); free(coded); free(back);
}

/* FIPS 180-4 known-answer vectors plus scalar-vs-accelerated differential tests across edge
 * lengths, unaligned pointers, and incremental chunked updates. */
static void sha256_scalar_ref(const unsigned char *p, size_t n, unsigned char out[32]) {
    Wex4Sha s;
    uint64_t bits = (uint64_t)n * 8u;
    unsigned char pad[72] = {0x80}, len[8];
    size_t fill, pad_len;
    unsigned int i;
    wex4_sha_init(&s);
    while (n >= 64u) { wex4_sha_block(s.h, p); p += 64u; n -= 64u; }
    if (n) memcpy(s.buf, p, n);
    fill = n;
    pad_len = (size_t)((119u - fill) & 63u) + 1u;
    for (i = 0; i < 8u; i++) len[i] = (unsigned char)(bits >> (56u - i * 8u));
    for (i = 0; i < pad_len; i++) {
        s.buf[fill++] = pad[i];
        if (fill == 64u) { wex4_sha_block(s.h, s.buf); fill = 0; }
    }
    for (i = 0; i < 8u; i++) {
        s.buf[fill++] = len[i];
        if (fill == 64u) { wex4_sha_block(s.h, s.buf); fill = 0; }
    }
    assert(fill == 0);
    for (i = 0; i < 32u; i++) out[i] = (unsigned char)(s.h[i / 4u] >> (24u - (i % 4u) * 8u));
}

static void sha256_hex(const unsigned char d[32], char hex[65]) {
    for (unsigned int i = 0; i < 32u; i++) snprintf(hex + i * 2u, 3u, "%02x", d[i]);
}

static void sha256_tests(void) {
    static const struct { const char *msg; const char *hex; } kat[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
         "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
         "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
    };
    unsigned char d_hw[32], d_sc[32], d_st[32];
    char hex[65];
    for (size_t k = 0; k < sizeof kat / sizeof kat[0]; k++) {
        size_t len = strlen(kat[k].msg);
        wex4_sha256((const unsigned char *)kat[k].msg, len, d_hw);
        sha256_scalar_ref((const unsigned char *)kat[k].msg, len, d_sc);
        sha256_hex(d_hw, hex);
        assert(!strcmp(hex, kat[k].hex));
        assert(!memcmp(d_hw, d_sc, 32u));
    }
    /* FIPS 180-4 1,000,000 x 'a' vector */
    {
        unsigned char *million = malloc(1000000u);
        assert(million);
        memset(million, 'a', 1000000u);
        wex4_sha256(million, 1000000u, d_hw);
        sha256_hex(d_hw, hex);
        assert(!strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
        free(million);
    }
    /* Scalar vs accelerated differential on edge lengths, unaligned starts, and chunked updates */
    static const size_t lens[] = {0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 4096, 65537};
    static const size_t chunks[] = {1, 7, 63, 64, 65, 1000};
    unsigned char *buf = malloc(65537u + 16u);
    uint32_t seed = 0x12345678u;
    assert(buf);
    for (size_t i = 0; i < 65537u + 16u; i++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        buf[i] = (unsigned char)seed;
    }
    for (size_t li = 0; li < sizeof lens / sizeof lens[0]; li++) {
        size_t n = lens[li];
        const unsigned char *src = buf + (li & 7u);   /* test unaligned pointers too */
        wex4_sha256(src, n, d_hw);
        sha256_scalar_ref(src, n, d_sc);
        assert(!memcmp(d_hw, d_sc, 32u));
        for (size_t ci = 0; ci < sizeof chunks / sizeof chunks[0]; ci++) {
            Wex4Sha s;
            size_t off = 0, step = chunks[ci];
            wex4_sha_init(&s);
            while (off < n) {
                size_t take = n - off < step ? n - off : step;
                wex4_sha_update(&s, src + off, take);
                off += take;
            }
            wex4_sha_final(&s, d_st);
            assert(!memcmp(d_st, d_sc, 32u));
        }
    }
    free(buf);
}

int main(void) {
    sha256_tests();
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
    index_rules(w4, bytes);
    unused_by_engines(w4, bytes);
    free(old); free(w4);
    puts("test_wex4_verify: intact file passes with the fixture's root; changed record, padding and blob bytes fail; "
         "damaged index / header, truncated and WEX1 files do not open; a resealed wrong record hash fails (1 and 4 threads); "
         "a resealed index that breaks a rule of wex4.h is refused with that rule's status; wex4_find, wex4_decode_record "
         "and a WBP1 round trip");
    return 0;
}
