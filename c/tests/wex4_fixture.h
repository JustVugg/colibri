/* tests/wex4_fixture.h -- the WEX4 form of a WEX1 (raw) expert container, built in memory the way
 * the in-place upgrade lays it out: the old bytes unchanged, then the old header page (blob kind
 * 16) and the index, and a new header page. The old table is blob kind 17. Every slot with a
 * record becomes one RAW EXPERT record of 6 U8 tensors (w1..w3 scales, then w1..w3 weights).
 * Needs the WEX1 records contiguous behind the table, in slot order. For tests only. */
#ifndef COLI_TESTS_WEX4_FIXTURE_H
#define COLI_TESTS_WEX4_FIXTURE_H
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../wex4.h"

static const char *wex4_fixture_names;
static const uint32_t *wex4_fixture_name_at, *wex4_fixture_name_len;
static int wex4_fixture_by_name(const void *a, const void *b) {
    uint32_t i = *(const uint32_t *)a, j = *(const uint32_t *)b, li = wex4_fixture_name_len[i], lj = wex4_fixture_name_len[j];
    int c = memcmp(wex4_fixture_names + wex4_fixture_name_at[i], wex4_fixture_names + wex4_fixture_name_at[j], li < lj ? li : lj);
    return c ? c : (li > lj) - (li < lj);
}

/* old: a WEX1 file of old_bytes; returns the WEX4 file (malloc) and its root as 64 hex digits. */
static unsigned char *wex4_fixture_from_wex1(const unsigned char *old, size_t old_bytes, size_t *bytes, char root[65]) {
    const WexHeader *h1 = (const WexHeader *)old;
    const WexEntry *table = (const WexEntry *)(old + h1->table_offset);
    uint32_t slots = h1->num_layers * h1->num_experts, n = 0, t, i, m;
    size_t table_bytes = (size_t)slots * sizeof(WexEntry), at = 0, name_bytes = 0;
    uint64_t total = 0, maximum = 0, raw = 0, aligned = 0;
    char *names = malloc((size_t)slots * 6u * 48u);
    uint32_t *name_at = malloc((size_t)slots * 6u * 4u), *name_len = malloc((size_t)slots * 6u * 4u),
             *order = malloc((size_t)slots * 6u * 4u), *slot_of = malloc((size_t)slots * 4u);
    assert(h1->magic == WEX_MAGIC_V1 && names && name_at && name_len && order && slot_of);
    for (i = 0; i < slots; i++) {
        if (!table[i].total_bytes) continue;
        assert(!wex_entry_raw_sizes(&table[i], &raw, &aligned));
        total += raw; if (raw > maximum) maximum = raw;
        for (m = 0; m < 6u; m++) {
            t = n * 6u + m;
            name_at[t] = (uint32_t)name_bytes;
            name_len[t] = (uint32_t)snprintf(names + name_bytes, (size_t)slots * 6u * 48u - name_bytes,
                                             "layers.%u.ffn.experts.%u.w%u.%s", i / h1->num_experts,
                                             i % h1->num_experts, m % 3u + 1u, m < 3u ? "scale" : "weight");
            name_bytes += name_len[t];
            order[t] = t;
        }
        slot_of[n++] = i;
    }
    wex4_fixture_names = names; wex4_fixture_name_at = name_at; wex4_fixture_name_len = name_len;
    qsort(order, (size_t)n * 6u, 4u, wex4_fixture_by_name);
    size_t index_bytes = 2u * WEX4_BLOB_BYTES + (size_t)n * WEX4_RECORD_BYTES + (size_t)n * 6u * (WEX4_TENSOR_BYTES + 4u) + name_bytes;
    size_t index_offset = old_bytes + WEX4_PAGE, file = index_offset + (size_t)wex4_align(index_bytes);
    unsigned char *out = calloc(1, file), *x;
    assert(out && !(old_bytes % WEX4_PAGE));
    memcpy(out, old, old_bytes);
    memcpy(out + old_bytes, old, WEX4_PAGE);
    x = out + index_offset;
#define W4F_PUT(value, type) do { type t_ = (type)(value); memcpy(x + at, &t_, sizeof t_); at += sizeof t_; } while (0)
    {
        struct { uint32_t kind; uint64_t offset, size; } blobs[2] = {{17u, h1->table_offset, table_bytes}, {16u, old_bytes, WEX4_PAGE}};
        for (i = 0; i < 2u; i++) {
            W4F_PUT(blobs[i].kind, uint32_t); W4F_PUT(0, uint32_t); W4F_PUT(blobs[i].offset, uint64_t); W4F_PUT(blobs[i].size, uint64_t);
            wex4_sha256(out + blobs[i].offset, (size_t)blobs[i].size, x + at); at += 32; W4F_PUT(0, uint64_t);
        }
    }
    for (i = 0; i < n; i++) {
        const WexEntry *e = &table[slot_of[i]];
        assert(!wex_entry_raw_sizes(e, &raw, &aligned) && aligned == e->total_bytes);
        W4F_PUT(e->file_offset, uint64_t); W4F_PUT(e->total_bytes, uint32_t); W4F_PUT(raw, uint32_t);
        W4F_PUT(WEX4_CODEC_RAW, uint16_t); W4F_PUT(WEX4_REC_EXPERT, uint16_t); W4F_PUT(0, uint32_t);
        W4F_PUT((slot_of[i] / h1->num_experts) << 16 | slot_of[i] % h1->num_experts, uint32_t);
        W4F_PUT(i * 6u, uint32_t); W4F_PUT(6, uint32_t);
        wex4_sha256(out + e->file_offset, e->total_bytes, x + at); at += 32; W4F_PUT(0, uint32_t);
    }
    for (i = 0; i < n; i++) {
        const WexEntry *e = &table[slot_of[i]];
        for (m = 0; m < 6u; m++) {
            const uint16_t *shape = m < 3u ? e->scale_shapes[m] : e->weight_shapes[m - 3u];
            t = i * 6u + m;
            W4F_PUT(name_at[t], uint32_t); W4F_PUT(name_len[t], uint16_t); W4F_PUT(2, uint8_t); W4F_PUT(2, uint8_t);
            W4F_PUT(shape[0], uint64_t); W4F_PUT(shape[1], uint64_t);
            for (int d = 2; d < 8; d++) W4F_PUT(0, uint64_t);
            W4F_PUT(i, uint32_t); W4F_PUT(1, uint32_t);
            W4F_PUT(m < 3u ? e->scale_offsets[m] : e->weight_offsets[m - 3u], uint64_t);
            W4F_PUT((uint64_t)shape[0] * shape[1], uint64_t);
        }
    }
    for (t = 0; t < n * 6u; t++) W4F_PUT(order[t], uint32_t);
    memcpy(x + at, names, name_bytes); at += name_bytes;
    assert(at == index_bytes);
    x = out; at = 0;
    memset(out, 0, WEX4_PAGE);
    W4F_PUT(WEX4_MAGIC, uint32_t); W4F_PUT(4, uint16_t); W4F_PUT(0, uint16_t); W4F_PUT(0, uint64_t); W4F_PUT(0, uint64_t);
    W4F_PUT(file, uint64_t); W4F_PUT(index_offset, uint64_t); W4F_PUT(index_bytes, uint64_t);
    W4F_PUT(2, uint32_t); W4F_PUT(n, uint32_t); W4F_PUT(n * 6u, uint32_t); W4F_PUT(h1->num_layers, uint32_t);
    W4F_PUT(h1->num_experts, uint32_t); W4F_PUT(maximum, uint32_t); W4F_PUT(name_bytes, uint64_t); W4F_PUT(0, uint64_t);
    W4F_PUT(total, uint64_t);
    wex4_sha256(out + index_offset, index_bytes, out + at);
    wex4_sha256(out, WEX4_HEADER_HASHED, out + WEX4_HEADER_HASHED);
#undef W4F_PUT
    for (i = 0; i < 32u; i++) snprintf(root + i * 2u, 3, "%02x", out[WEX4_HEADER_HASHED + i]);
    free(names); free(name_at); free(name_len); free(order); free(slot_of);
    *bytes = file;
    return out;
}

/* Reseal after an edit of the index: index hash, then header hash; root gets the new root. */
static void wex4_fixture_reseal(unsigned char *w4, char root[65]) {
    uint64_t index_offset, index_bytes;
    memcpy(&index_offset, w4 + 32, 8); memcpy(&index_bytes, w4 + 40, 8);
    wex4_sha256(w4 + index_offset, (size_t)index_bytes, w4 + 96);
    wex4_sha256(w4, WEX4_HEADER_HASHED, w4 + WEX4_HEADER_HASHED);
    for (int i = 0; i < 32; i++) snprintf(root + i * 2, 3, "%02x", w4[WEX4_HEADER_HASHED + i]);
}
#endif
