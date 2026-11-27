#include "../wex_format.h"
#include <stdio.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void put_u32(unsigned char *p, uint32_t value) {
    p[0] = (unsigned char)value;
    p[1] = (unsigned char)(value >> 8);
    p[2] = (unsigned char)(value >> 16);
    p[3] = (unsigned char)(value >> 24);
}

static WexEntry make_entry(int compressed) {
    WexEntry entry;
    unsigned int m;
    memset(&entry, 0, sizeof(entry));
    entry.file_offset = 2u * WEX_ALIGN;
    entry.scale_bytes = 3u * 64u * 64u;
    entry.weight_bytes = 3u * 64u * 1024u;
    entry.total_bytes = entry.scale_bytes + entry.weight_bytes;
    if (compressed) {
        entry.flags = (1u << WEX_ENTRY_SCALE_PAGES_SHIFT) | WEX_ENTRY_SCALE_TILES;
        entry.total_bytes = WEX_ALIGN + entry.weight_bytes;
    }
    for (m = 0; m < 3; m++) {
        entry.scale_shapes[m][0] = 64;
        entry.scale_shapes[m][1] = 64;
        entry.weight_shapes[m][0] = 64;
        entry.weight_shapes[m][1] = 1024;
        entry.scale_offsets[m] = m * 4096u;
        entry.weight_offsets[m] = entry.scale_bytes + m * 65536u;
    }
    return entry;
}

static WexHeader make_header(int version) {
    WexHeader header;
    WexEntry raw = make_entry(0);
    memset(&header, 0, sizeof(header));
    header.magic = version == 1 ? WEX_MAGIC_V1 : version == 2 ? WEX_MAGIC_V2 : WEX_MAGIC_V3;
    header.version = (uint32_t)version;
    header.num_layers = 1;
    header.num_experts = 1;
    header.aligned_expert_bytes = raw.total_bytes;
    header.table_offset = WEX_ALIGN;
    header.flags = version == 1 ? 0 : version == 2 ? WEX_HEADER_SCALE_TILES :
                   WEX_HEADER_SCALE_TILES | WEX_HEADER_PALETTE3 | WEX_HEADER_DEDUP;
    return header;
}

/* Test encoder is intentionally independent of the production bit reader. */
static size_t encode_tile(unsigned char *encoded, const unsigned char *raw,
                          size_t count, unsigned int base, unsigned int bits) {
    size_t index, bytes = (count * bits + 7u) / 8u;
    memset(encoded, 0, bytes + 2u);
    encoded[0] = (unsigned char)base;
    encoded[1] = (unsigned char)bits;
    for (index = 0; index < count; index++) {
        unsigned int bit;
        unsigned int delta = (unsigned int)raw[index] - base;
        for (bit = 0; bit < bits; bit++) {
            size_t position = index * bits + bit;
            if (delta & (1u << bit))
                encoded[2u + position / 8u] |= (unsigned char)(1u << (position % 8u));
        }
    }
    return bytes + 2u;
}

static void test_bit_vectors(void) {
    static const unsigned char known[] = {119, 3, 0x88, 0xc6, 0xfa};
    unsigned char decoded[WEX_SCALE_TILE_VALUES + 7u];
    unsigned char raw[WEX_SCALE_TILE_VALUES];
    unsigned char encoded[WEX_SCALE_TILE_VALUES + 2u];
    unsigned int width;
    size_t index, count;
    CHECK(!wex_unpack_scale_tiles(known, sizeof(known), decoded, 8));
    for (index = 0; index < 8; index++) CHECK(decoded[index] == 119u + index);
    for (width = 0; width <= 8; width++) {
        for (count = 1; count <= WEX_SCALE_TILE_VALUES; count += 17u) {
            unsigned int base = width == 8 ? 0 : 119;
            unsigned int mask = (1u << width) - 1u;
            size_t bytes;
            for (index = 0; index < count; index++)
                raw[index] = (unsigned char)(base + ((index * 37u) & mask));
            /* For 7-bit values use base 0 to keep every byte representable. */
            if (width == 7) {
                base = 0;
                for (index = 0; index < count; index++)
                    raw[index] = (unsigned char)((index * 37u) & mask);
            }
            bytes = encode_tile(encoded, raw, count, base, width);
            CHECK(!wex_unpack_scale_tiles(encoded, bytes, decoded, count));
            CHECK(!memcmp(raw, decoded, count));
            CHECK(wex_unpack_scale_tiles(encoded, bytes - 1u, decoded, count));
            CHECK(wex_unpack_scale_tiles(encoded, bytes + 1u, decoded, count));
        }
    }
    {
        unsigned char constant[] = {255, 0};
        unsigned char overflow[] = {255, 1, 1};
        unsigned char bad_width[] = {0, 9};
        unsigned char bad_padding[] = {119, 1, 0x80};
        CHECK(!wex_unpack_scale_tiles(constant, 2, decoded, WEX_SCALE_TILE_VALUES));
        for (index = 0; index < WEX_SCALE_TILE_VALUES; index++) CHECK(decoded[index] == 255);
        CHECK(wex_unpack_scale_tiles(overflow, sizeof(overflow), decoded, 1));
        CHECK(wex_unpack_scale_tiles(bad_width, sizeof(bad_width), decoded, 1));
        CHECK(wex_unpack_scale_tiles(bad_padding, sizeof(bad_padding), decoded, 1));
        CHECK(wex_unpack_scale_tiles(NULL, 1, decoded, 1));
        CHECK(!wex_unpack_scale_tiles(NULL, 0, NULL, 0));
    }
}

/* Exercise both the bounded fast path and legal high-base streams that need
 * individual overflow checks. All tail lengths and group boundaries matter.
 */
static void test_tile_boundaries(void) {
    unsigned char raw[WEX_SCALE_TILE_VALUES];
    unsigned char encoded[WEX_SCALE_TILE_VALUES + 3u];
    unsigned char guarded[WEX_SCALE_TILE_VALUES + 2u];
    unsigned int width, variant;
    size_t count, index;
    for (width = 0; width <= 8u; width++) {
        unsigned int mask = (1u << width) - 1u;
        for (variant = 0; variant < 4u; variant++) {
            unsigned int base = variant == 0 ? 0 : variant == 1 ? 255u - mask :
                                variant == 2 ? 255u : (mask ? 256u - mask : 119u);
            unsigned int max_delta = 255u - base;
            if (max_delta > mask) max_delta = mask;
            for (count = 1; count <= WEX_SCALE_TILE_VALUES; count++) {
                size_t bytes;
                for (index = 0; index < count; index++)
                    raw[index] = (unsigned char)(base + ((index * 37u + count) % (max_delta + 1u)));
                bytes = encode_tile(encoded, raw, count, base, width);
                memset(guarded, 0xa5, sizeof(guarded));
                CHECK(!wex_unpack_scale_tiles(encoded, bytes, NULL, count));
                CHECK(!wex_unpack_scale_tiles(encoded, bytes, guarded + 1u, count));
                CHECK(!memcmp(raw, guarded + 1u, count));
                CHECK(guarded[0] == 0xa5 && guarded[count + 1u] == 0xa5);
                CHECK(wex_unpack_scale_tiles(encoded, bytes - 1u, NULL, count));
                CHECK(wex_unpack_scale_tiles(encoded, bytes + 1u, NULL, count));
                if ((count * width) & 7u) {
                    encoded[bytes - 1u] |= 0x80u;
                    CHECK(wex_unpack_scale_tiles(encoded, bytes, NULL, count));
                    CHECK(wex_unpack_scale_tiles(encoded, bytes, guarded + 1u, count));
                }
            }
        }
    }
    /* A valid first tile must not hide overflow in a later, partial tile. */
    {
        unsigned char mixed[] = {119, 0, 255, 1, 1};
        CHECK(wex_unpack_scale_tiles(mixed, sizeof(mixed), NULL, WEX_SCALE_TILE_VALUES + 1u));
    }
}

static void test_headers_and_entries(void) {
    WexHeader header = make_header(1), changed;
    WexEntry raw = make_entry(0), compressed = make_entry(1), entry, missing;
    uint64_t file_bytes = raw.file_offset + raw.total_bytes, aligned;
    size_t bytes = 99, table_aligned = 99;
    CHECK(sizeof(WexHeader) == 40 && sizeof(WexEntry) == 72);
    CHECK(WEX_MAGIC == WEX_MAGIC_V1 && WEX_MAGIC_V1 != WEX_MAGIC_V2);
    CHECK(!wex_validate_header(&header, file_bytes, 1, 1, &bytes, &table_aligned));
    CHECK(bytes == 72 && table_aligned == WEX_ALIGN);
    CHECK(!wex_validate_entry(&header, &raw, file_bytes));
    CHECK(!wex_record_read_offset(&raw));
    CHECK(wex_validate_entry(&header, &compressed, file_bytes));
    header = make_header(2);
    CHECK(!wex_validate_entry(&header, &raw, file_bytes));
    CHECK(!wex_validate_entry(&header, &compressed, file_bytes));
    CHECK(wex_record_read_offset(&compressed) == 2u * WEX_ALIGN);
    memset(&missing, 0, sizeof(missing));
    CHECK(!wex_validate_entry(&header, &missing, file_bytes));
    missing.scale_bytes = 1;
    CHECK(wex_validate_entry(&header, &missing, file_bytes));

    changed = header; changed.magic = WEX_MAGIC_V1;
    CHECK(wex_validate_header(&changed, file_bytes, 1, 1, NULL, NULL));
    changed = header; changed.version = 99;
    CHECK(wex_validate_header(&changed, file_bytes, 1, 1, NULL, NULL));
    changed = header; changed.flags |= 2u;
    CHECK(wex_validate_header(&changed, file_bytes, 1, 1, NULL, NULL));
    changed = make_header(1); changed.flags = WEX_HEADER_SCALE_TILES;
    CHECK(wex_validate_header(&changed, file_bytes, 1, 1, NULL, NULL));
    changed = header; changed.reserved = 1;
    CHECK(wex_validate_header(&changed, file_bytes, 1, 1, NULL, NULL));
    changed = header; changed.num_layers = 0;
    CHECK(wex_validate_header(&changed, file_bytes, 0, 1, NULL, NULL));
    changed = header; changed.num_layers = INT_MAX; changed.num_experts = INT_MAX;
    CHECK(wex_validate_header(&changed, UINT64_MAX, 0, 0, NULL, NULL));
    changed = header; changed.aligned_expert_bytes = UINT64_MAX;
    CHECK(wex_validate_header(&changed, file_bytes, 1, 1, NULL, NULL));
    changed = header; changed.table_offset++;
    CHECK(wex_validate_header(&changed, file_bytes, 1, 1, NULL, NULL));
    changed = header; changed.table_offset = UINT64_MAX - (WEX_ALIGN - 1u);
    CHECK(wex_validate_header(&changed, UINT64_MAX, 1, 1, NULL, NULL));
    CHECK(wex_validate_header(&header, file_bytes, 2, 1, NULL, NULL));
    CHECK(wex_validate_header(&header, file_bytes, -1, 1, NULL, NULL));
    CHECK(wex_validate_header(&header, 2u * WEX_ALIGN - 1u, 1, 1, NULL, NULL));
    CHECK(wex_aligned_size(UINT64_MAX, &aligned));
    CHECK(WEX_ALIGN_UP(UINT64_C(0x100000001), WEX_ALIGN) == UINT64_C(0x100001000));

    entry = raw; entry.scale_offsets[1]++;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = raw; entry.weight_offsets[2]++;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = raw; entry.scale_bytes++;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = raw; entry.weight_bytes++;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = raw; entry.scale_shapes[0][1]++;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = raw; entry.total_bytes--;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = raw; entry.file_offset = WEX_ALIGN;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = raw; entry.file_offset++;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = raw; entry.flags = 2;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    CHECK(wex_validate_entry(&header, &raw, file_bytes - 1u));
    entry = compressed; entry.flags = WEX_ENTRY_SCALE_TILES;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    entry = compressed; entry.flags = (3u << WEX_ENTRY_SCALE_PAGES_SHIFT) | WEX_ENTRY_SCALE_TILES;
    CHECK(wex_validate_entry(&header, &entry, file_bytes));
    changed = header; changed.flags = 0;
    CHECK(wex_validate_entry(&changed, &compressed, file_bytes));
    changed = header; changed.aligned_expert_bytes -= WEX_ALIGN;
    CHECK(wex_validate_entry(&changed, &compressed, file_bytes));
}

static void test_record_decoder(void) {
    WexEntry entry = make_entry(1), raw_entry = make_entry(0), changed;
    size_t capacity = raw_entry.total_bytes;
    unsigned char *disk = (unsigned char *)calloc(1, entry.total_bytes);
    unsigned char *allocated = (unsigned char *)malloc(capacity + 32u);
    unsigned char *slab = allocated + 16u;
    unsigned char *expected = (unsigned char *)malloc(entry.scale_bytes);
    size_t read_offset = wex_record_read_offset(&entry), stream_bytes = 0;
    size_t tile, index;
    CHECK(disk && allocated && expected);
    for (tile = 0; tile < entry.scale_bytes / WEX_SCALE_TILE_VALUES; tile++) {
        unsigned int width = (unsigned int)(tile % 4u);
        unsigned int mask = (1u << width) - 1u;
        unsigned char *values = expected + tile * WEX_SCALE_TILE_VALUES;
        for (index = 0; index < WEX_SCALE_TILE_VALUES; index++)
            values[index] = (unsigned char)(119u + ((index * 13u) & mask));
        stream_bytes += encode_tile(disk + WEX_SCALE_STREAM_HEADER_BYTES + stream_bytes,
                                    values, WEX_SCALE_TILE_VALUES, 119, width);
    }
    CHECK(stream_bytes + WEX_SCALE_STREAM_HEADER_BYTES < WEX_ALIGN);
    put_u32(disk, WEX_SCALE_STREAM_MAGIC);
    put_u32(disk + 4u, entry.scale_bytes);
    put_u32(disk + 8u, (uint32_t)stream_bytes);
    for (index = 0; index < entry.weight_bytes; index++)
        disk[WEX_ALIGN + index] = (unsigned char)(index * 13u);
    memset(allocated, 0xa5, capacity + 32u);
    memcpy(slab + read_offset, disk, entry.total_bytes);
    CHECK(!wex_decode_record(&entry, slab, capacity));
    CHECK(!memcmp(slab, expected, entry.scale_bytes));
    CHECK(!memcmp(slab + entry.scale_bytes, disk + WEX_ALIGN, entry.weight_bytes));
    for (index = 0; index < 16; index++) {
        CHECK(allocated[index] == 0xa5);
        CHECK(allocated[16u + capacity + index] == 0xa5);
    }
    CHECK(!wex_decode_record(&raw_entry, slab, capacity));
    CHECK(wex_decode_record(&entry, slab, capacity - 1u));
    changed = entry; changed.flags = 2;
    CHECK(wex_decode_record(&changed, slab, capacity));
    changed = entry; changed.total_bytes += WEX_ALIGN;
    CHECK(wex_decode_record(&changed, slab, capacity));

    /* Each bad stream is rejected before any destination or weight changes. */
    for (tile = 0; tile < 7; tile++) {
        unsigned char *before = (unsigned char *)malloc(capacity);
        CHECK(before);
        memset(slab, 0xcc, capacity);
        memcpy(slab + read_offset, disk, entry.total_bytes);
        if (tile == 0) slab[read_offset] ^= 1u;
        if (tile == 1) put_u32(slab + read_offset + 4u, entry.scale_bytes - 1u);
        if (tile == 2) put_u32(slab + read_offset + 8u, WEX_ALIGN);
        if (tile == 3) slab[read_offset + WEX_SCALE_STREAM_HEADER_BYTES + 1u] = 9;
        if (tile == 4) put_u32(slab + read_offset + 8u, (uint32_t)stream_bytes - 1u);
        if (tile == 5) put_u32(slab + read_offset + 8u, (uint32_t)stream_bytes + 1u);
        /* Last tile has width 3 and nonzero residuals: force late overflow
         * after the fast validator has accepted all preceding tiles.
         */
        if (tile == 6) slab[read_offset + WEX_SCALE_STREAM_HEADER_BYTES + stream_bytes - 98u] = 255;
        memcpy(before, slab, capacity);
        CHECK(wex_decode_record(&entry, slab, capacity));
        CHECK(!memcmp(before, slab, capacity));
        free(before);
    }
    free(expected);
    free(allocated);
    free(disk);
}

static void test_palette_record(int small) {
    WexEntry raw_entry = make_entry(0), entry;
    WexHeader header = make_header(3);
    unsigned char *disk, *allocated, *slab, *expected, *before;
    uint64_t raw, aligned;
    size_t capacity, stream_bytes = 0, pos, matrix, scale_cursor = 0, weight_cursor = 0;
    size_t offset;
    if (small) {
        raw_entry.scale_bytes = 1000;
        raw_entry.weight_bytes = 16000;
        for (matrix = 0; matrix < 3; matrix++) {
            unsigned int rows = matrix == 2 ? 998 : 1;
            raw_entry.scale_shapes[matrix][0] = (uint16_t)rows;
            raw_entry.scale_shapes[matrix][1] = 1;
            raw_entry.weight_shapes[matrix][0] = (uint16_t)rows;
            raw_entry.weight_shapes[matrix][1] = 16;
            raw_entry.scale_offsets[matrix] = (uint32_t)scale_cursor;
            raw_entry.weight_offsets[matrix] = raw_entry.scale_bytes + (uint32_t)weight_cursor;
            scale_cursor += rows; weight_cursor += rows * 16u;
        }
    }
    CHECK(!wex_entry_raw_sizes(&raw_entry, &raw, &aligned));
    capacity = (size_t)aligned;
    disk = (unsigned char *)calloc(1, capacity);
    expected = (unsigned char *)calloc(1, capacity);
    allocated = (unsigned char *)malloc(capacity + 32u);
    before = (unsigned char *)malloc(capacity);
    CHECK(disk && expected && allocated && before);
    slab = allocated + 16u;
    for (pos = 0; pos < raw_entry.scale_bytes; pos++) expected[pos] = (unsigned char)(pos * 7u);
    for (pos = 0; pos < raw_entry.scale_bytes; pos += WEX_SCALE_TILE_VALUES) {
        size_t count = raw_entry.scale_bytes - pos;
        if (count > WEX_SCALE_TILE_VALUES) count = WEX_SCALE_TILE_VALUES;
        stream_bytes += encode_tile(disk + WEX_PALETTE3_PREFIX_BYTES + stream_bytes,
                                   expected + pos, count, 0, 8);
    }
    for (matrix = 0; matrix < 3; matrix++) {
        size_t group, count = (size_t)raw_entry.scale_shapes[matrix][0] * raw_entry.scale_shapes[matrix][1];
        unsigned char *tables = disk + WEX_PALETTE3_HEADER_BYTES + matrix * 1024u;
        unsigned int scale, code;
        for (scale = 0; scale < 256; scale++) {
            for (code = 0; code < 8; code++)
                tables[scale * 4u + code / 2u] |= (unsigned char)(
                    ((scale * 3u + code * 5u + matrix * 7u) & 15u) << ((code & 1u) * 4u));
        }
        for (group = 0; group < count; group++) {
            unsigned char codes[32], encoded[14];
            unsigned int nibble;
            scale = expected[raw_entry.scale_offsets[matrix] + group];
            for (nibble = 0; nibble < 32u; nibble++) {
                unsigned int symbol;
                codes[nibble] = (unsigned char)((nibble * 3u + group) & 7u);
                symbol = (tables[scale * 4u + codes[nibble] / 2u] >> ((codes[nibble] & 1u) * 4u)) & 15u;
                expected[raw_entry.weight_offsets[matrix] + group * 16u + nibble / 2u] |=
                    (unsigned char)(symbol << ((nibble & 1u) * 4u));
            }
            CHECK(encode_tile(encoded, codes, 32u, 0, 3) == 14u);
            memcpy(disk + WEX_PALETTE3_PREFIX_BYTES + stream_bytes +
                   (raw_entry.weight_offsets[matrix] - raw_entry.scale_bytes) / 4u * 3u + group * 12u,
                   encoded + 2u, 12u);
        }
    }
    entry = raw_entry;
    entry.flags = WEX_ENTRY_PALETTE3;
    entry.total_bytes = (uint32_t)WEX_ALIGN_UP(WEX_PALETTE3_PREFIX_BYTES + stream_bytes +
                                            entry.weight_bytes / 4u * 3u, WEX_ALIGN);
    CHECK(entry.total_bytes < capacity);
    put_u32(disk, WEX_PALETTE3_MAGIC);
    put_u32(disk + 4u, (uint32_t)stream_bytes);
    put_u32(disk + 8u, entry.weight_bytes / 4u * 3u);
    header.aligned_expert_bytes = capacity;
    CHECK(!wex_validate_entry(&header, &entry, entry.file_offset + entry.total_bytes));
    { WexHeader old = make_header(2); CHECK(wex_validate_entry(&old, &entry, entry.file_offset + entry.total_bytes)); }
    offset = wex_record_read_offset(&entry);
    CHECK(offset == capacity - entry.total_bytes);
    memset(allocated, 0xa5, capacity + 32u);
    memcpy(slab + offset, disk, entry.total_bytes);
    CHECK(!wex_decode_record(&entry, slab, capacity));
    CHECK(!memcmp(slab, expected, (size_t)raw));
    for (pos = 0; pos < 16u; pos++) CHECK(allocated[pos] == 0xa5 && allocated[capacity + 16u + pos] == 0xa5);
    CHECK(wex_decode_record(&entry, slab, capacity - 1u));
    for (pos = 0; pos < 6; pos++) {
        memset(slab, 0xcc, capacity);
        memcpy(slab + offset, disk, entry.total_bytes);
        if (pos == 0) slab[offset] ^= 1u;
        if (pos == 1) put_u32(slab + offset + 4u, 0);
        if (pos == 2) put_u32(slab + offset + 4u, UINT32_MAX);
        if (pos == 3) put_u32(slab + offset + 8u, 1);
        if (pos == 4) put_u32(slab + offset + 12u, 1);
        if (pos == 5) slab[offset + WEX_PALETTE3_PREFIX_BYTES + 1u] = 9;
        memcpy(before, slab, capacity);
        CHECK(wex_decode_record(&entry, slab, capacity));
        CHECK(!memcmp(slab, before, capacity));
    }
    free(before); free(allocated); free(expected); free(disk);
}

static void test_v3_alias_manifest(void) {
    WexHeader header = make_header(3);
    WexEntry entries[2];
    uint64_t file_bytes;
    entries[0] = make_entry(0);
    entries[1] = entries[0];
    header.num_experts = 2;
    file_bytes = entries[0].file_offset + entries[0].total_bytes;
    CHECK(!wex_validate_table(&header, entries, file_bytes));
    header.flags &= ~WEX_HEADER_DEDUP;
    CHECK(wex_validate_table(&header, entries, file_bytes));
    header.flags |= WEX_HEADER_DEDUP;
    entries[1].scale_shapes[0][0] = entries[1].weight_shapes[0][0] = 32;
    entries[1].scale_shapes[0][1] = 128;
    entries[1].weight_shapes[0][1] = 2048;
    CHECK(!wex_validate_entry(&header, &entries[1], file_bytes));
    CHECK(wex_validate_table(&header, entries, file_bytes));
    entries[1] = entries[0];
    CHECK(wex_validate_table(&header, entries, file_bytes + WEX_ALIGN));
}

/* V1/V2: the header's slab size is bounded by the table, at most twice the
 * largest aligned decoded record (one page when the table has no record). */
static void test_v12_slab_bound(void) {
    int version;
    for (version = 1; version <= 2; version++) {
        WexHeader header = make_header(version);
        WexEntry entries[2], swap;
        uint64_t record, file_bytes;
        unsigned int m;
        entries[0] = make_entry(version == 2);
        entries[1] = make_entry(0);   /* a smaller raw record after it: 32 rows */
        for (m = 0; m < 3; m++) {
            entries[1].scale_shapes[m][0] = entries[1].weight_shapes[m][0] = 32;
            entries[1].scale_offsets[m] = m * 2048u;
            entries[1].weight_offsets[m] = 3u * 2048u + m * 32768u;
        }
        entries[1].scale_bytes = 3u * 2048u;
        entries[1].weight_bytes = 3u * 32768u;
        entries[1].total_bytes = (uint32_t)WEX_ALIGN_UP(entries[1].scale_bytes + entries[1].weight_bytes, WEX_ALIGN);
        entries[1].file_offset = entries[0].file_offset + entries[0].total_bytes;
        header.num_experts = 2;
        record = header.aligned_expert_bytes;   /* the larger of the two, as the writers store it */
        file_bytes = entries[1].file_offset + entries[1].total_bytes;
        CHECK(!wex_validate_table(&header, entries, file_bytes));
        header.aligned_expert_bytes = 2u * record;   /* a rounded-up slab is accepted */
        CHECK(!wex_validate_table(&header, entries, file_bytes));
        header.aligned_expert_bytes = 2u * record + WEX_ALIGN;
        CHECK(!wex_validate_header(&header, file_bytes, 1, 2, NULL, NULL));
        CHECK(!wex_validate_entry(&header, &entries[0], file_bytes) && !wex_validate_entry(&header, &entries[1], file_bytes));
        CHECK(wex_validate_table(&header, entries, file_bytes));
        header.aligned_expert_bytes = UINT32_MAX - (WEX_ALIGN - 1u);   /* 4 GiB - 4096: the largest the header check takes */
        CHECK(!wex_validate_header(&header, file_bytes, 1, 2, NULL, NULL));
        CHECK(wex_validate_table(&header, entries, file_bytes));
        header.aligned_expert_bytes = record - WEX_ALIGN;   /* below the larger record: the per-entry rule */
        CHECK(wex_validate_table(&header, entries, file_bytes));
        /* the bound follows the largest record, wherever it is: above, it is the
         * first one; here the last (V1/V2 tables may list records in any order) */
        swap = entries[0]; entries[0] = entries[1]; entries[1] = swap;
        header.aligned_expert_bytes = 2u * record;
        CHECK(!wex_validate_table(&header, entries, file_bytes));
        header.aligned_expert_bytes = 2u * record + WEX_ALIGN;
        CHECK(wex_validate_table(&header, entries, file_bytes));
        swap = entries[0]; entries[0] = entries[1]; entries[1] = swap;
        /* without the larger record, the bound is twice the smaller one */
        memset(&entries[0], 0, sizeof(entries[0]));
        header.aligned_expert_bytes = 2u * record;
        CHECK(wex_validate_table(&header, entries, file_bytes));
        header.aligned_expert_bytes = 2u * entries[1].total_bytes + WEX_ALIGN;
        CHECK(wex_validate_table(&header, entries, file_bytes));
        header.aligned_expert_bytes = 2u * entries[1].total_bytes;
        CHECK(!wex_validate_table(&header, entries, file_bytes));
        /* no record at all: one page */
        memset(&entries[1], 0, sizeof(entries[1]));
        CHECK(wex_validate_table(&header, entries, file_bytes));
        header.aligned_expert_bytes = WEX_ALIGN;
        CHECK(!wex_validate_table(&header, entries, file_bytes));
    }
}

/* WNR2: train a model on synthetic experts, encode with the reference encoder,
 * decode into a slab, and reject every malformed field. */
static uint32_t nr2_rand(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static void nr2_fill(const WexEntry *entry, unsigned char *raw, uint32_t seed) {
    /* FP4-like codes: magnitudes skewed small, sign random; scales near 120 */
    static const unsigned char mag[16] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7};
    uint32_t s = seed, i;
    for (i = 0; i < entry->scale_bytes; i++) raw[i] = (unsigned char)(118u + nr2_rand(&s) % 5u);
    for (i = 0; i < entry->weight_bytes; i++) {
        uint32_t r = nr2_rand(&s);
        unsigned char a = (unsigned char)(mag[r & 15u] | ((r >> 4) & 8u)), b = (unsigned char)(mag[(r >> 8) & 15u] | ((r >> 12) & 8u));
        raw[entry->scale_bytes + i] = (unsigned char)(a | b << 4);
    }
}
static void nr2_views(const WexEntry *entry, unsigned char *raw, const unsigned char *scales[3],
                      const unsigned char *weights[3]) {
    unsigned int m;
    for (m = 0; m < 3u; m++) { scales[m] = raw + entry->scale_offsets[m]; weights[m] = raw + entry->weight_offsets[m]; }
}

static void test_nr2_codec(void) {
    WexEntry entry = make_entry(0);
    WexHeader header = make_header(3);
    WexNr2Layout layout;
    WexNr2Model model;
    WexNr2Record view;
    uint64_t raw_bytes, aligned_raw;
    size_t written, capacity, read_offset, total;
    static uint64_t wcounts[3u * WEX_NR2_CONTEXTS * 16u], scounts[3u * 256u];
    unsigned char *blob = malloc(WEX_NR2_MODEL_BYTES), *raw = malloc(entry.scale_bytes + entry.weight_bytes);
    unsigned char *record, *slab, *bad, page[WEX_ALIGN];
    const unsigned char *scales[3], *weights[3];
    unsigned int chunk_case, m;
    uint64_t at; uint32_t bytes, id;
    CHECK(blob && raw);
    CHECK(!wex_entry_raw_sizes(&entry, &raw_bytes, &aligned_raw));
    wex_nr2_layout(&entry, &layout);
    for (m = 0; m < 3u; m++) CHECK(layout.symbol_count[m] == 64u * 1024u * 2u && layout.scale_count[m] == 4096u);
    memset(wcounts, 0, sizeof(wcounts)); memset(scounts, 0, sizeof(scounts));
    for (id = 1; id <= 3u; id++) {   /* train on three experts, encode a fourth */
        nr2_fill(&entry, raw, id * 7919u);
        nr2_views(&entry, raw, scales, weights);
        wex_nr2_count(&layout, scales, weights, wcounts, scounts);
    }
    wex_nr2_model_build(wcounts, scounts, blob);
    CHECK(!wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model));
    CHECK(wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES - 2u, &model));
    {   /* a zero-width symbol, a wrong total, a wrong magic: all refused */
        WexNr2Model broken;
        unsigned char keep[4];
        memcpy(keep, blob + 34u, 2u); blob[34] = blob[35] = 0;
        CHECK(wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &broken)); memcpy(blob + 34u, keep, 2u);
        memcpy(keep, blob + 64u, 2u); blob[64] ^= 1u;
        CHECK(wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &broken)); memcpy(blob + 64u, keep, 2u);
        memcpy(keep, blob, 4u); blob[0] ^= 1u;
        CHECK(wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &broken)); memcpy(blob, keep, 4u);
    }
    CHECK(!wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model));
    nr2_fill(&entry, raw, 424242u);
    nr2_views(&entry, raw, scales, weights);
    capacity = (size_t)aligned_raw + WEX_ALIGN;
    record = malloc(capacity); slab = malloc(capacity); bad = malloc(capacity);
    CHECK(record && slab && bad);
    for (chunk_case = 0; chunk_case < 3u; chunk_case++) {
        static const uint32_t wchunk[3] = {4096u, 512u, 16384u}, schunk[3] = {2048u, 256u, 16384u};
        unsigned char *outs[3], *outw[3];
        memset(record, 0, capacity);
        CHECK(!wex_nr2_encode(&model, &layout, scales, weights, wchunk[chunk_case], schunk[chunk_case],
                              record, capacity, &written));
        total = (written + 4095u) & ~(size_t)4095u;
        CHECK(total < aligned_raw);   /* the synthetic source compresses */
        CHECK(!wex_nr2_parse(record, total, &layout, model.id, &view));
        CHECK(view.weight_chunk == wchunk[chunk_case] && view.scale_chunk == schunk[chunk_case]);
        /* decode through the slab API: record at the read offset, sentinels around */
        entry.flags = WEX_ENTRY_NEURAL_RANS2;
        entry.total_bytes = (uint32_t)total;
        read_offset = wex_record_read_offset(&entry);
        CHECK(read_offset == aligned_raw - total);
        memset(slab, 0xa5, capacity);
        memcpy(slab + read_offset, record, total);
        CHECK(wex_decode_record(&entry, slab, capacity));        /* no model: refused */
        CHECK(!wex_decode_nr2_record(&entry, slab, capacity, &model));
        CHECK(!memcmp(slab, raw, (size_t)raw_bytes));
        for (m = 0; m < 3u; m++) { outs[m] = bad + entry.scale_offsets[m]; outw[m] = bad + entry.weight_offsets[m]; }
        CHECK(!wex_nr2_decode(&view, &layout, &model, outs, outw) && !memcmp(bad, raw, (size_t)raw_bytes));
        /* malformed records are refused and leave the slab untouched */
#define NR2_REJECT(mutate) do { \
            memcpy(bad, record, total); mutate; \
            memset(slab, 0x5a, capacity); memcpy(slab + read_offset, bad, total); \
            CHECK(wex_decode_nr2_record(&entry, slab, capacity, &model)); \
            CHECK(slab[0] == 0x5a && slab[read_offset - 1u] == 0x5a); \
        } while (0)
        NR2_REJECT(bad[0] ^= 1u);                                  /* magic */
        NR2_REJECT(bad[4] ^= 1u);                                  /* model id */
        NR2_REJECT(put_u32(bad + 16u, wchunk[chunk_case] * 3u));   /* chunk not a power of two */
        NR2_REJECT(put_u32(bad + 16u, 256u));                      /* weight chunk too small */
        NR2_REJECT(put_u32(bad + 20u, 32768u));                    /* scale chunk too large */
        NR2_REJECT(bad[27] = 1u);                                  /* reserved */
        NR2_REJECT(put_u32(bad + 28u, 1u));                        /* reserved */
        NR2_REJECT(put_u32(bad + 12u, view.weight_payload_bytes + 1u));   /* payload size */
        NR2_REJECT(bad[WEX_NR2_HEADER_BYTES] = 3u; bad[WEX_NR2_HEADER_BYTES + 1u] = 0);  /* stream < 4 bytes */
        NR2_REJECT(bad[WEX_NR2_HEADER_BYTES] ^= 1u);               /* lengths no longer add up */
        NR2_REJECT(memset(bad + view.metadata_bytes, 0, 4u));     /* initial state below L */
        NR2_REJECT(bad[view.metadata_bytes] = 0x80u);              /* initial state above 256 L */
        if (written < total) NR2_REJECT(bad[total - 1u] = 1u);     /* non-zero padding */
        NR2_REJECT(memset(bad + view.metadata_bytes + view.scale_payload_bytes + 4u, 0, 64u));  /* corrupted payload */
#undef NR2_REJECT
        CHECK(wex_nr2_parse(record, total - WEX_ALIGN, &layout, 0, &view));   /* truncated */
        CHECK(wex_nr2_parse(record, total, &layout, model.id + 1u, &view));    /* other model */
        entry.flags = 0;
    }
    /* container side: flag, entry validation and the model descriptor */
    header.flags |= WEX_HEADER_NEURAL_RANS2;
    header.table_offset = 2u * WEX_ALIGN + WEX_ALIGN * 14u;
    CHECK(!wex_validate_header(&header, 64u * WEX_ALIGN, 0, 0, NULL, NULL));
    entry.flags = WEX_ENTRY_NEURAL_RANS2;
    entry.total_bytes = (uint32_t)total;
    entry.file_offset = header.table_offset + WEX_ALIGN;
    CHECK(!wex_validate_entry(&header, &entry, 64u * WEX_ALIGN + aligned_raw));
    header.flags &= ~WEX_HEADER_NEURAL_RANS2;
    CHECK(wex_validate_entry(&header, &entry, 64u * WEX_ALIGN + aligned_raw));
    header.flags |= WEX_HEADER_NEURAL_RANS2;
    entry.total_bytes = (uint32_t)aligned_raw;   /* no smaller than raw: not a WNR2 record */
    CHECK(wex_validate_entry(&header, &entry, 64u * WEX_ALIGN + aligned_raw));
    memset(page, 0, sizeof(page));
    memcpy(page, &header, sizeof(header));
    put_u32(page + 64u, WEX_NR2_DESCRIPTOR_MAGIC); put_u32(page + 68u, WEX_NR2_MODEL_BYTES);
    put_u32(page + 72u, WEX_ALIGN); put_u32(page + 76u, 0); put_u32(page + 80u, model.id);
    CHECK(!wex_nr2_model_location(&header, page, &at, &bytes, &id) && at == WEX_ALIGN &&
          bytes == WEX_NR2_MODEL_BYTES && id == model.id);
    put_u32(page + 72u, WEX_ALIGN + 512u); CHECK(wex_nr2_model_location(&header, page, &at, &bytes, &id));
    put_u32(page + 72u, 12u * WEX_ALIGN); CHECK(wex_nr2_model_location(&header, page, &at, &bytes, &id));  /* runs into the table */
    put_u32(page + 72u, WEX_ALIGN); put_u32(page + 84u, 1u); CHECK(wex_nr2_model_location(&header, page, &at, &bytes, &id));
    put_u32(page + 84u, 0); put_u32(page + 80u, 0); CHECK(wex_nr2_model_location(&header, page, &at, &bytes, &id));
    put_u32(page + 80u, model.id); header.flags &= ~WEX_HEADER_NEURAL_RANS2;
    CHECK(wex_nr2_model_location(&header, page, &at, &bytes, &id));
    free(blob); free(raw); free(record); free(slab); free(bad);
}

/* G2a: the WNR2 encoder output must stay byte-identical. FNV-1a of the
 * unpadded record for 3 seeded synthetic experts x 2 chunk pairs, recorded on
 * master 3b604bb before any WNR3 change. */
static void test_nr2_golden(void) {
    static const uint32_t golden[3][2] = {{0xf28a1ab7u, 0xf21a38d5u}, {0xd286a3cdu, 0xbfbf3718u},
                                             {0x8e1a5e5au, 0x2912c05bu}};   /* model b118b72e */
    static const uint32_t wchunk[2] = {2048u, 512u}, schunk[2] = {512u, 256u};
    WexEntry entry = make_entry(0);
    WexNr2Layout layout;
    WexNr2Model model;
    static uint64_t wcounts[3u * WEX_NR2_CONTEXTS * 16u], scounts[3u * 256u];
    unsigned char *blob = malloc(WEX_NR2_MODEL_BYTES), *raw = malloc(entry.scale_bytes + entry.weight_bytes);
    size_t capacity = entry.scale_bytes + entry.weight_bytes + WEX_ALIGN, written;
    unsigned char *record = malloc(capacity);
    const unsigned char *scales[3], *weights[3];
    uint32_t seed, c;
    CHECK(blob && raw && record);
    wex_nr2_layout(&entry, &layout);
    memset(wcounts, 0, sizeof(wcounts)); memset(scounts, 0, sizeof(scounts));
    for (seed = 1; seed <= 2u; seed++) {
        nr2_fill(&entry, raw, seed * 104729u);
        nr2_views(&entry, raw, scales, weights);
        wex_nr2_count(&layout, scales, weights, wcounts, scounts);
    }
    wex_nr2_model_build(wcounts, scounts, blob);
    CHECK(!wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model));
    for (seed = 0; seed < 3u; seed++) {
        nr2_fill(&entry, raw, 20261110u + seed * 31u);
        nr2_views(&entry, raw, scales, weights);
        for (c = 0; c < 2u; c++) {
            uint32_t h;
            CHECK(!wex_nr2_encode(&model, &layout, scales, weights, wchunk[c], schunk[c], record, capacity, &written));
            h = wex_nr2_fnv1a(record, written);
            if (golden[seed][c] != h) {
                fprintf(stderr, "WNR2 golden seed %u chunk %u: %08x (want %08x), model %08x\n",
                        seed, c, h, golden[seed][c], model.id);
                CHECK(golden[seed][c] == h);
            }
        }
    }
    free(blob); free(raw); free(record);
}

/* ---- WNR3 ------------------------------------------------------------- */

static void test_nr3_hash_vectors(void) {
    unsigned char key[256], hashes[1024], data[1024];
    static const unsigned char scale5[5] = {0x7f, 0x7e, 0x80, 0x7f, 0x7d};
    uint32_t i, ident[4] = {0x299dc586u, 0x00270078u, 2165u, 0x0f787978u};
    for (i = 0; i < 256u; i++) {   /* SMHasher Murmur3A verification value */
        key[i] = (unsigned char)i;
        put_u32(hashes + i * 4u, wex_nr3_h32(key, i, 256u - i));
    }
    CHECK(wex_nr3_h32(hashes, 1024u, 0) == 0xb0f57ee3u);
    for (i = 0; i < 16u; i++) key[i] = (unsigned char)i;
    CHECK(wex_nr3_h32(key, 0, 0) == 0);
    CHECK(wex_nr3_h32(key, 0, WEX_NR3_HASH_SEED) == 0xe7f741f1u);
    CHECK(wex_nr3_h32(key, 16u, WEX_NR3_HASH_SEED) == 0x31ce0919u);
    for (i = 0; i < 1024u; i++) data[i] = (unsigned char)((37u * i + 11u) & 0x77u);
    CHECK(wex_nr3_stream_x0(ident, data, 1024u) == 0x00e485b2u);
    ident[2] = 719u;
    CHECK(wex_nr3_stream_x0(ident, scale5, 5u) == 0x00c976d8u);
    {   /* rule g: densest adjacent pair, the first one on ties */
        static const unsigned char t1[4] = {10, 11, 20, 21}, t2[5] = {5, 5, 6, 7, 7}, t3[3] = {254, 255, 255};
        CHECK(wex_nr3_anchor(t1, 4u) == 10u && wex_nr3_anchor(t2, 5u) == 5u && wex_nr3_anchor(t3, 3u) == 254u);
    }
}

/* Synthetic MXFP4 expert with every WNR3 case: general blocks (max 7 with odd
 * magnitudes), even blocks (ceil form), even blocks at scale 254 / 255, max
 * 0..4 blocks, all-zero blocks with random signs (-0), and, in five_matrix, one
 * block of max 5 (the ceil gate fails for that matrix). */
static void nr3_fill(const WexEntry *e, unsigned char *raw, uint32_t seed, int five_matrix) {
    static const unsigned char general[16] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7};
    static const unsigned char even[8] = {0, 0, 2, 2, 4, 5, 6, 7};
    uint32_t s = seed, m, b, k;
    for (m = 0; m < 3u; m++) {
        uint32_t blocks = (uint32_t)e->scale_shapes[m][0] * e->scale_shapes[m][1];
        unsigned char *sc = raw + e->scale_offsets[m], *w = raw + e->weight_offsets[m];
        for (b = 0; b < blocks; b++) {
            uint32_t kind = nr2_rand(&s) % 8u;
            unsigned char *q = w + (size_t)b * 16u;
            sc[b] = (unsigned char)(118u + nr2_rand(&s) % 5u);
            if (kind == 4u && b % 37u == 0) sc[b] = (unsigned char)(b & 1u ? 255u : 254u);
            for (k = 0; k < 32u; k++) {
                uint32_t r = nr2_rand(&s);
                uint32_t a = kind < 4u ? general[r & 15u] : kind < 6u ? even[r & 7u] : kind == 6u ? (r & 7u) % 5u : 0u;
                uint32_t x = a | ((r >> 8) & 8u);
                if (k < 2u && kind < 6u) x = (x & 8u) | 7u;   /* max 7 */
                if (k & 1u) q[k / 2u] = (unsigned char)(q[k / 2u] | x << 4); else q[k / 2u] = (unsigned char)x;
            }
        }
        if ((int)m == five_matrix) memset(w + 16u * 3u, 0x55, 16u);
    }
}

static void nr3_split(const WexEntry *e, unsigned char *raw, unsigned char *scales[3], unsigned char *weights[3]) {
    unsigned int m;
    for (m = 0; m < 3u; m++) { scales[m] = raw + e->scale_offsets[m]; weights[m] = raw + e->weight_offsets[m]; }
}

/* coded = source -> Z (cz) -> ceil form; returns the record flags. */
static uint32_t nr3_prepare(const WexEntry *e, const WexNr2Layout *layout, const unsigned char *source,
                            unsigned char *coded, int cz, uint64_t *mapped, uint32_t stats[3][3]) {
    unsigned char *scales[3], *weights[3];
    uint64_t z = 0;
    memcpy(coded, source, (size_t)e->scale_bytes + e->weight_bytes);
    if (cz) z = wex_nr3_canonical_zero(coded + e->scale_bytes, e->weight_bytes);
    if (mapped) *mapped = z;
    nr3_split(e, coded, scales, weights);
    return wex_nr3_ceil_forward(layout, scales, weights, stats) | (cz ? WEX_NR3_FLAG_CZ : 0u);
}

/* Sign-symmetric weight counts (f[s] == f[s ^ 8], as real models nearly are):
 * the case where a WNR2 sign flip decodes without error. */
static void nr3_symmetrize(uint64_t *wcounts, int keep_zero) {
    size_t row, s;
    for (row = 0; row < 3u * WEX_NR2_CONTEXTS; row++)
        for (s = keep_zero ? 1u : 0; s < 8u; s++) {
            uint64_t c = wcounts[row * 16u + s] + wcounts[row * 16u + s + 8u];
            wcounts[row * 16u + s] = wcounts[row * 16u + s + 8u] = c;
        }
}

static void nr3_train(const WexEntry *e, int cz, unsigned char *blob, WexNr2Model *model, int symmetric) {
    static uint64_t wcounts[3u * WEX_NR2_CONTEXTS * 16u], scounts[3u * 256u];
    size_t raw = (size_t)e->scale_bytes + e->weight_bytes;
    unsigned char *source = malloc(raw), *coded = malloc(raw), *scales[3], *weights[3];
    WexNr2Layout layout;
    uint32_t id;
    CHECK(source && coded);
    wex_nr2_layout(e, &layout);
    memset(wcounts, 0, sizeof(wcounts)); memset(scounts, 0, sizeof(scounts));
    for (id = 1; id <= 3u; id++) {
        nr3_fill(e, source, id * 6007u, -1);
        nr3_prepare(e, &layout, source, coded, cz, NULL, NULL);
        nr3_split(e, coded, scales, weights);
        wex_nr3_count(&layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights, wcounts, scounts);
    }
    if (symmetric) nr3_symmetrize(wcounts, cz);
    wex_nr2_model_build(wcounts, scounts, blob);
    CHECK(!wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, model));
    free(source); free(coded);
}

/* Z(source) == expected, and they differ exactly in the source's 0x8 nibbles. */
static void nr3_check_canonical(const unsigned char *source, const unsigned char *decoded, size_t scale_bytes,
                                size_t raw, uint64_t mapped) {
    uint64_t eights = 0;
    size_t i;
    CHECK(!memcmp(source, decoded, scale_bytes));
    for (i = scale_bytes; i < raw; i++) {
        unsigned char d = (unsigned char)(source[i] ^ decoded[i]);
        CHECK(!(d & 15u) || ((source[i] & 15u) == 8u && (decoded[i] & 15u) == 0));
        CHECK(!(d >> 4) || ((source[i] >> 4) == 8u && (decoded[i] >> 4) == 0));
        eights += ((source[i] & 15u) == 8u) + ((source[i] >> 4) == 8u);
        CHECK(((decoded[i] & 15u) != 8u) && ((decoded[i] >> 4) != 8u));
    }
    CHECK(eights == mapped);
}

static void test_nr3_codec(void) {
    static const uint32_t wchunk[4] = {512u, 2048u, 4096u, 16384u}, schunk[4] = {256u, 512u, 2048u, 16384u};
    WexEntry entry = make_entry(0);
    WexNr2Layout layout;
    WexNr2Model models[2];
    WexNr2Record view;
    uint64_t raw_bytes, aligned_raw, mapped;
    size_t written, total, capacity, read_offset;
    unsigned char *blobs[2] = {malloc(WEX_NR2_MODEL_BYTES), malloc(WEX_NR2_MODEL_BYTES)};
    unsigned char *source, *coded, *expected, *record, *slab, *bad, *scales[3], *weights[3];
    uint32_t stats[3][3], flags, owner = 39u << 16 | 120u, m;
    int cz, c, swapped = 0, stale_scale = 0;
    CHECK(blobs[0] && blobs[1] && !wex_entry_raw_sizes(&entry, &raw_bytes, &aligned_raw));
    capacity = (size_t)aligned_raw + WEX_ALIGN;
    source = malloc(capacity); coded = malloc(capacity); expected = malloc(capacity);
    record = malloc(capacity); slab = malloc(capacity); bad = malloc(capacity);
    CHECK(source && coded && expected && record && slab && bad);
    wex_nr2_layout(&entry, &layout);
    nr3_train(&entry, 0, blobs[0], &models[0], 0);
    nr3_train(&entry, 1, blobs[1], &models[1], 0);
    CHECK(models[0].id != models[1].id);
    nr3_fill(&entry, source, 4242u, 1);
    for (cz = 0; cz < 2; cz++) {
        const WexNr2Model *model = &models[cz], *other = &models[!cz];
        flags = nr3_prepare(&entry, &layout, source, coded, cz, &mapped, stats);
        CHECK(flags == (5u | (cz ? WEX_NR3_FLAG_CZ : 0u)));   /* matrix 1 has a max-5 block */
        CHECK(stats[1][1] == 1u && stats[1][0] == 0 && stats[0][0] && stats[2][0] && stats[0][1] == 0);
        CHECK(stats[0][2] + stats[2][2] > 0);   /* even blocks at scale 255 stay */
        CHECK(cz ? mapped > 0 : mapped == 0);
        memcpy(expected, source, (size_t)raw_bytes);
        if (cz) wex_nr3_canonical_zero(expected + entry.scale_bytes, entry.weight_bytes);
        nr3_split(&entry, coded, scales, weights);
        for (c = 0; c < 4; c++) {
            uint32_t s, t, n_streams;
            memset(record, 0, capacity);
            CHECK(!wex_nr3_encode(model, &layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                                  wchunk[c], schunk[c], flags, owner, record, capacity, &written));
            total = (written + 4095u) & ~(size_t)4095u;
            CHECK(total < aligned_raw);
            CHECK(!wex_nr3_parse(record, total, &layout, model->id, &view));
            CHECK(wex_nr2_parse(record, total, &layout, model->id, &view));   /* not a WNR2 record */
            CHECK(!wex_nr3_parse(record, total, &layout, model->id, &view));
            CHECK(view.flags == flags && view.owner == owner && view.magic == WEX_NR3_MAGIC);
            for (m = 0; m < 3u; m++) CHECK(view.median[m] == wex_nr3_anchor(scales[m], layout.scale_count[m]));
            entry.flags = WEX_ENTRY_NEURAL_RANS3;
            entry.total_bytes = (uint32_t)total;
            read_offset = wex_record_read_offset(&entry);
            CHECK(read_offset == aligned_raw - total);
            {   /* owner, and the record's CZ flag against the container's flag 64 */
                uint32_t mine = cz ? WEX_HEADER_CANONICAL_ZERO : 0u, other_profile = mine ^ WEX_HEADER_CANONICAL_ZERO;
                CHECK(wex_nr3_owner_ok(&entry, record, 39u, 120u, mine) && !wex_nr3_owner_ok(&entry, record, 39u, 121u, mine));
                CHECK(!wex_nr3_owner_ok(&entry, record, 39u, 120u, other_profile));
            }
            memset(slab, 0xa5, capacity);
            memcpy(slab + read_offset, record, total);
            CHECK(wex_decode_record(&entry, slab, capacity));           /* needs its model */
            CHECK(wex_decode_nr2_record(&entry, slab, capacity, other)); /* another model */
            CHECK(!wex_decode_nr2_record(&entry, slab, capacity, model));
            CHECK(!memcmp(slab, expected, (size_t)raw_bytes));
            if (cz) nr3_check_canonical(source, slab, entry.scale_bytes, (size_t)raw_bytes, mapped);
            else CHECK(!memcmp(slab, source, (size_t)raw_bytes));        /* byte-exact without CZ */
            entry.flags = WEX_ENTRY_NEURAL_RANS2;                        /* WNR3 record under codec 4 */
            memset(slab, 0x5a, capacity); memcpy(slab + read_offset, record, total);
            CHECK(wex_decode_nr2_record(&entry, slab, capacity, model) && slab[0] == 0x5a);
            entry.flags = WEX_ENTRY_NEURAL_RANS3;
#define NR3_REJECT(mutate) do { \
                memcpy(bad, record, total); mutate; \
                memset(slab, 0x5a, capacity); memcpy(slab + read_offset, bad, total); \
                CHECK(wex_decode_nr2_record(&entry, slab, capacity, model)); \
                CHECK(slab[0] == 0x5a && slab[read_offset - 1u] == 0x5a); \
            } while (0)
            for (s = 0x10u; s <= 0x80u; s <<= 1) NR3_REJECT(bad[27] |= (unsigned char)s);   /* reserved flags */
            NR3_REJECT(bad[27] ^= 1u);                          /* CEIL bit */
            NR3_REJECT(bad[27] ^= 2u);                          /* CEIL bit of the gate-failed matrix */
            NR3_REJECT(bad[27] ^= WEX_NR3_FLAG_CZ);             /* CZ bit */
            NR3_REJECT(bad[24]++);                              /* anchor */
            NR3_REJECT(bad[26]--);
            NR3_REJECT(put_u32(bad + 28u, owner ^ 1u));          /* owner */
            NR3_REJECT(put_u32(bad + 4u, other->id));            /* other model id (parse) */
            NR3_REJECT(bad[0] ^= 1u);                           /* magic */
            NR3_REJECT(bad[view.metadata_bytes + 5u] ^= 0x40u);  /* scale payload */
            NR3_REJECT(bad[view.metadata_bytes + view.scale_payload_bytes + 9u] ^= 1u);   /* weight payload */
            if (written < total) NR3_REJECT(bad[total - 1u] = 1u);   /* padding */
            /* decoded under the other model: parses, every hash fails */
            memcpy(bad, record, total); put_u32(bad + 4u, other->id);
            memset(slab, 0x5a, capacity); memcpy(slab + read_offset, bad, total);
            CHECK(wex_decode_nr2_record(&entry, slab, capacity, other) && slab[0] == 0x5a);
            /* two equal-length streams swapped: same structure, other index g */
            n_streams = view.scale_streams + view.weight_streams;
            for (s = 0; s < n_streams && !swapped; s++)
                for (t = s + 1u; t < n_streams; t++) {
                    uint32_t ls = wex_nr_u16(view.lengths + s * 2u), lt = wex_nr_u16(view.lengths + t * 2u), at_s = 0, at_t = 0, u;
                    if (ls != lt) continue;
                    for (u = 0; u < t; u++) { uint32_t l = wex_nr_u16(view.lengths + u * 2u); if (u < s) at_s += l; at_t += l; }
                    memcpy(bad, record, total);
                    memcpy(bad + view.metadata_bytes + at_s, record + view.metadata_bytes + at_t, ls);
                    memcpy(bad + view.metadata_bytes + at_t, record + view.metadata_bytes + at_s, ls);
                    if (!memcmp(bad, record, total)) continue;
                    NR3_REJECT(memcpy(bad + view.metadata_bytes + at_s, record + view.metadata_bytes + at_t, ls);
                               memcpy(bad + view.metadata_bytes + at_t, record + view.metadata_bytes + at_s, ls));
                    swapped++;
                    break;
                }
            {   /* scale stream 0 re-encoded with one scale moved where srel stays clamped (the weight
                 * contexts do not change) under the stream's ORIGINAL x0: only its hash sees it */
                static unsigned char tmp[2u * WEX_NR2_MAX_CHUNK + 8u], data[WEX_NR2_MAX_CHUNK];
                uint32_t count = layout.scale_count[0] < view.scale_chunk ? layout.scale_count[0] : view.scale_chunk;
                uint32_t len0 = wex_nr_u16(view.lengths), ident[4] = {model->id, owner, 0, view.word24}, k;
                uint32_t x0 = wex_nr3_stream_x0(ident, scales[0], count);
                WexNr2Record other_view;
                memcpy(data, scales[0], count);
                for (k = 0; k < count; k++) {
                    int d = (int)data[k] - (int)view.median[0];
                    unsigned char keep = data[k], *p = tmp + sizeof(tmp);
                    uint32_t state = x0, j;
                    if (!((d >= 3 && keep < 255u) || (d <= -3 && keep > 0))) continue;
                    data[k] = (unsigned char)(d >= 3 ? keep + 1u : keep - 1u);
                    for (j = count; j-- > 0;) CHECK(!wex_nr2_put(&state, &p, tmp, wex_nr2_scale_cdf(model, 0), data[j]));
                    CHECK(!wex_nr2_flush(state, &p, tmp));
                    data[k] = keep;
                    {   /* splice it in: new length, scale payload size and padding */
                        uint32_t nl = (uint32_t)(tmp + sizeof(tmp) - p), used = view.metadata_bytes +
                            view.scale_payload_bytes + view.weight_payload_bytes;
                        size_t new_total = ((size_t)used - len0 + nl + 4095u) & ~(size_t)4095u, at;
                        memset(bad, 0, capacity);
                        memcpy(bad, record, view.metadata_bytes);
                        bad[WEX_NR2_HEADER_BYTES] = (unsigned char)nl; bad[WEX_NR2_HEADER_BYTES + 1u] = (unsigned char)(nl >> 8);
                        put_u32(bad + 8u, view.scale_payload_bytes - len0 + nl);
                        memcpy(bad + view.metadata_bytes, p, nl);
                        memcpy(bad + view.metadata_bytes + nl, record + view.metadata_bytes + len0, used - view.metadata_bytes - len0);
                        CHECK(new_total < aligned_raw && !wex_nr3_parse(bad, new_total, &layout, model->id, &other_view));
                        entry.total_bytes = (uint32_t)new_total;
                        at = wex_record_read_offset(&entry);
                        memset(slab, 0x5a, capacity); memcpy(slab + at, bad, new_total);
                        CHECK(wex_decode_nr2_record(&entry, slab, capacity, model) && slab[0] == 0x5a);
                        entry.total_bytes = (uint32_t)total;
                    }
                    stale_scale++;
                    break;
                }
                memcpy(bad, record, total); bad[27] |= 0x10u;   /* reserved flag bit: refused by the parser itself */
                CHECK(wex_nr3_parse(bad, total, &layout, 0, &other_view));
            }
#undef NR3_REJECT
            CHECK(wex_nr3_parse(record, total - WEX_ALIGN, &layout, 0, &view));   /* truncated */
            CHECK(!wex_nr3_parse(record, total, &layout, 0, &view));
            entry.flags = 0;
        }
        /* a WNR2 record under a codec-5 entry */
        {
            const unsigned char *ss[3], *ww[3];
            for (m = 0; m < 3u; m++) { ss[m] = scales[m]; ww[m] = weights[m]; }   /* coded data: the model fits it */
            CHECK(!wex_nr2_encode(model, &layout, ss, ww, 2048u, 512u, record, capacity, &written));
            total = (written + 4095u) & ~(size_t)4095u;
            entry.flags = WEX_ENTRY_NEURAL_RANS3; entry.total_bytes = (uint32_t)total;
            read_offset = wex_record_read_offset(&entry);
            memset(slab, 0x5a, capacity); memcpy(slab + read_offset, record, total);
            CHECK(wex_decode_nr2_record(&entry, slab, capacity, model) && slab[0] == 0x5a);
            entry.flags = 0;
        }
    }
    CHECK(swapped && stale_scale);
    /* crafted records the encoder would never write, decoded by the reference */
    flags = nr3_prepare(&entry, &layout, source, coded, 0, NULL, NULL);
    nr3_split(&entry, coded, scales, weights);
    CHECK(wex_nr3_encode(&models[1], &layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                         2048u, 512u, flags | WEX_NR3_FLAG_CZ, owner, record, capacity, &written));   /* -0 under CZ */
    for (c = 0; c < 2; c++) {
        unsigned char anchor[3];
        uint32_t b = 0, cflags = flags;
        if (c == 0) cflags |= WEX_NR3_FLAG_CZ;   /* CZ record carrying symbol 8 */
        else {                                    /* ceil record with a max-5 block at coded scale 0 */
            for (b = 0; b < layout.scale_count[0] && wex_nr3_block_max(weights[0] + (size_t)b * 16u, NULL) != 5u; b++) {}
            CHECK(b < layout.scale_count[0]);
            scales[0][b] = 0;
        }
        for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(scales[m], layout.scale_count[m]);
        CHECK(!wex_nrx_encode(&models[0], &layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                              2048u, 512u, anchor, WEX_NR3_MAGIC, cflags, owner, record, capacity, &written));
        total = (written + 4095u) & ~(size_t)4095u;
        CHECK(!wex_nr3_parse(record, total, &layout, models[0].id, &view));
        entry.flags = WEX_ENTRY_NEURAL_RANS3; entry.total_bytes = (uint32_t)total;
        read_offset = wex_record_read_offset(&entry);
        memset(slab, 0x5a, capacity); memcpy(slab + read_offset, record, total);
        CHECK(wex_decode_nr2_record(&entry, slab, capacity, &models[0]) && slab[0] == 0x5a);
        if (c == 1) {   /* the same record decodes once the scale is legal again */
            scales[0][b] = 1;
            for (m = 0; m < 3u; m++) anchor[m] = wex_nr3_anchor(scales[m], layout.scale_count[m]);
            CHECK(!wex_nrx_encode(&models[0], &layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                                  2048u, 512u, anchor, WEX_NR3_MAGIC, cflags, owner, record, capacity, &written));
            total = (written + 4095u) & ~(size_t)4095u;
            entry.total_bytes = (uint32_t)total; read_offset = wex_record_read_offset(&entry);
            memset(slab, 0x5a, capacity); memcpy(slab + read_offset, record, total);
            CHECK(!wex_decode_nr2_record(&entry, slab, capacity, &models[0]));
        }
        entry.flags = 0;
    }
    {   /* uniform random nibbles do not compress: the builder writes them raw (codec 0) */
        uint32_t r = 77u, i;
        for (i = 0; i < raw_bytes; i++) source[i] = (unsigned char)nr2_rand(&r);
        flags = nr3_prepare(&entry, &layout, source, coded, 0, NULL, NULL);
        nr3_split(&entry, coded, scales, weights);
        written = 0;   /* too large for the buffer, or no smaller than raw */
        CHECK(wex_nr3_encode(&models[0], &layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                             2048u, 512u, flags, owner, record, capacity, &written) ||
              ((written + 4095u) & ~(size_t)4095u) >= aligned_raw);
    }
    free(blobs[0]); free(blobs[1]); free(source); free(coded); free(expected); free(record); free(slab); free(bad);
}

/* Corruption audit: every single-bit flip in the header, the length table, the
 * first scale stream and the first and last weight streams of every matrix of
 * a small expert is either rejected (slab untouched) or decodes to exactly the
 * expected slab. WNR3 must accept no wrong output; WNR2 (same source, same
 * bits) is counted for comparison. */
static void nr3_audit_entry(WexEntry *e) {
    unsigned int m;
    memset(e, 0, sizeof(*e));
    e->scale_bytes = 3u * 256u; e->weight_bytes = 3u * 4096u;
    for (m = 0; m < 3u; m++) {
        e->scale_shapes[m][0] = e->weight_shapes[m][0] = 16;
        e->scale_shapes[m][1] = 16; e->weight_shapes[m][1] = 256;
        e->scale_offsets[m] = m * 256u;
        e->weight_offsets[m] = e->scale_bytes + m * 4096u;
    }
}

static void nr3_audit_record(const WexEntry *base, uint32_t codec, const unsigned char *record, size_t total,
                             const WexNr2Model *model, const unsigned char *expected, uint64_t *flips,
                             uint64_t *rejected, uint64_t *harmless, uint64_t *wrong) {
    WexEntry entry = *base;
    WexNr2Layout layout;
    WexNr2Record view;
    uint64_t raw, aligned;
    size_t capacity, read_offset, at, ranges[16][2], nranges = 0, r;
    unsigned char *slab;
    uint32_t bit;
    CHECK(!wex_entry_raw_sizes(&entry, &raw, &aligned));
    entry.flags = codec; entry.total_bytes = (uint32_t)total;
    read_offset = wex_record_read_offset(&entry);
    capacity = (size_t)aligned;
    slab = malloc(capacity);
    CHECK(slab);
    wex_nr2_layout(&entry, &layout);
    CHECK(!wex_nrx_parse(record, total, &layout, model->id, wex_entry_nr_magic(&entry), &view));
    ranges[nranges][0] = 0; ranges[nranges++][1] = view.metadata_bytes;   /* header + lengths */
    {
        uint32_t s, w0 = view.scale_streams;
        uint32_t per = wex_nr2_streams(layout.symbol_count[0], view.weight_chunk);
        size_t off = view.metadata_bytes;
        for (s = 0; s < view.scale_streams + view.weight_streams; s++) {
            size_t len = wex_nr_u16(view.lengths + s * 2u);
            int take = s == 0 || (s >= w0 && ((s - w0) % per == 0 || (s - w0) % per == per - 1u));
            if (take && nranges < 16u) { ranges[nranges][0] = off; ranges[nranges++][1] = off + len; }
            off += len;
        }
    }
    for (r = 0; r < nranges; r++)
        for (at = ranges[r][0]; at < ranges[r][1]; at++)
            for (bit = 0; bit < 8u; bit++) {
                memset(slab, 0x5a, capacity);
                memcpy(slab + read_offset, record, total);
                slab[read_offset + at] ^= (unsigned char)(1u << bit);
                ++*flips;
                if (wex_decode_nr2_record(&entry, slab, capacity, model)) {
                    CHECK(slab[0] == 0x5a);
                    ++*rejected;
                } else if (!memcmp(slab, expected, (size_t)raw)) ++*harmless;
                else ++*wrong;
            }
    free(slab);
}

static void test_nr3_audit(void) {
    WexEntry entry;
    WexNr2Layout layout;
    WexNr2Model models[2];
    uint64_t raw, aligned, f[2] = {0, 0}, rej[2] = {0, 0}, ok[2] = {0, 0}, wrong[2] = {0, 0};
    unsigned char *blobs[2] = {malloc(WEX_NR2_MODEL_BYTES), malloc(WEX_NR2_MODEL_BYTES)};
    unsigned char *source, *coded, *expected, *record, *scales[3], *weights[3];
    const unsigned char *ss[3], *ww[3];
    size_t written, cap;
    uint32_t flags, m;
    WexEntry big = make_entry(0);
    nr3_audit_entry(&entry);
    CHECK(!wex_entry_raw_sizes(&entry, &raw, &aligned));
    cap = (size_t)aligned + WEX_ALIGN;
    source = malloc(cap); coded = malloc(cap); expected = malloc(cap); record = calloc(1, cap);
    CHECK(blobs[0] && blobs[1] && source && coded && expected && record);
    nr3_train(&big, 1, blobs[1], &models[1], 1);   /* CZ model; WNR2 gets the plain-trained one */
    {
        static uint64_t wcounts[3u * WEX_NR2_CONTEXTS * 16u], scounts[3u * 256u];
        unsigned char *t = malloc((size_t)big.scale_bytes + big.weight_bytes);
        WexNr2Layout bl;
        CHECK(t);
        wex_nr2_layout(&big, &bl);
        memset(wcounts, 0, sizeof(wcounts)); memset(scounts, 0, sizeof(scounts));
        nr3_fill(&big, t, 6007u, -1);
        for (m = 0; m < 3u; m++) { ss[m] = t + big.scale_offsets[m]; ww[m] = t + big.weight_offsets[m]; }
        wex_nr2_count(&bl, ss, ww, wcounts, scounts);
        nr3_symmetrize(wcounts, 0);
        wex_nr2_model_build(wcounts, scounts, blobs[0]);
        CHECK(!wex_nr2_model_open(blobs[0], WEX_NR2_MODEL_BYTES, &models[0]));
        free(t);
    }
    wex_nr2_layout(&entry, &layout);
    nr3_fill(&entry, source, 1311u, -1);
    for (m = 0; m < 3u; m++) { ss[m] = source + entry.scale_offsets[m]; ww[m] = source + entry.weight_offsets[m]; }
    CHECK(!wex_nr2_encode(&models[0], &layout, ss, ww, 512u, 256u, record, cap, &written));
    CHECK(((written + 4095u) & ~(size_t)4095u) < aligned);
    nr3_audit_record(&entry, WEX_ENTRY_NEURAL_RANS2, record, (written + 4095u) & ~(size_t)4095u, &models[0], source,
                     &f[0], &rej[0], &ok[0], &wrong[0]);
    flags = nr3_prepare(&entry, &layout, source, coded, 1, NULL, NULL);
    memcpy(expected, source, (size_t)raw);
    wex_nr3_canonical_zero(expected + entry.scale_bytes, entry.weight_bytes);
    nr3_split(&entry, coded, scales, weights);
    memset(record, 0, cap);
    CHECK(!wex_nr3_encode(&models[1], &layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                          512u, 256u, flags, 7u, record, cap, &written));
    CHECK(((written + 4095u) & ~(size_t)4095u) < aligned);
    nr3_audit_record(&entry, WEX_ENTRY_NEURAL_RANS3, record, (written + 4095u) & ~(size_t)4095u, &models[1], expected,
                     &f[1], &rej[1], &ok[1], &wrong[1]);
    printf("WNR3 corruption audit: WNR2 %llu flips, %llu rejected, %llu harmless, %llu WRONG accepted; "
           "WNR3+CZ %llu flips, %llu rejected, %llu harmless, %llu wrong accepted.\n",
           (unsigned long long)f[0], (unsigned long long)rej[0], (unsigned long long)ok[0], (unsigned long long)wrong[0],
           (unsigned long long)f[1], (unsigned long long)rej[1], (unsigned long long)ok[1], (unsigned long long)wrong[1]);
    CHECK(f[1] > 10000u && wrong[1] == 0 && wrong[0] > 0);   /* WNR2 lets sign flips through, WNR3 none */
    free(blobs[0]); free(blobs[1]); free(source); free(coded); free(expected); free(record);
}

static void test_nr3_container_rules(void) {
    WexEntry entry = make_entry(0);
    WexHeader header = make_header(3);
    unsigned char page[WEX_ALIGN];
    uint64_t raw, aligned, at;
    uint32_t bytes, id;
    CHECK(!wex_entry_raw_sizes(&entry, &raw, &aligned));
    header.flags = WEX_HEADER_SCALE_TILES | WEX_HEADER_NEURAL_RANS3;
    header.table_offset = 16u * WEX_ALIGN;
    CHECK(!wex_validate_header(&header, 64u * WEX_ALIGN, 0, 0, NULL, NULL));
    header.flags |= WEX_HEADER_CANONICAL_ZERO;
    CHECK(!wex_validate_header(&header, 64u * WEX_ALIGN, 0, 0, NULL, NULL));
    header.flags |= WEX_HEADER_NEURAL_RANS2; CHECK(wex_validate_header(&header, 64u * WEX_ALIGN, 0, 0, NULL, NULL));
    header.flags &= ~WEX_HEADER_NEURAL_RANS2;
    header.flags |= WEX_HEADER_NEURAL_RANS; CHECK(wex_validate_header(&header, 64u * WEX_ALIGN, 0, 0, NULL, NULL));
    header.flags = WEX_HEADER_NEURAL_RANS2 | WEX_HEADER_CANONICAL_ZERO;   /* 64 needs 32 */
    CHECK(wex_validate_header(&header, 64u * WEX_ALIGN, 0, 0, NULL, NULL));
    header.flags = WEX_HEADER_SCALE_TILES | WEX_HEADER_NEURAL_RANS3 | WEX_HEADER_DEDUP;   /* WNR3 never dedups */
    CHECK(wex_validate_header(&header, 64u * WEX_ALIGN, 0, 0, NULL, NULL));
    header.flags = WEX_HEADER_SCALE_TILES | WEX_HEADER_NEURAL_RANS3;
    entry.flags = WEX_ENTRY_NEURAL_RANS3;
    entry.total_bytes = (uint32_t)aligned - WEX_ALIGN;
    entry.file_offset = header.table_offset + WEX_ALIGN;
    CHECK(wex_record_read_offset(&entry) == WEX_ALIGN);
    CHECK(!wex_validate_entry(&header, &entry, 64u * WEX_ALIGN + aligned));
    entry.flags = WEX_ENTRY_NEURAL_RANS2;   /* codec 4 in a flag-32 container */
    CHECK(wex_validate_entry(&header, &entry, 64u * WEX_ALIGN + aligned));
    entry.flags = WEX_ENTRY_NEURAL_RANS3 | (1u << 8);   /* bits 8-31 must be 0 */
    CHECK(wex_validate_entry(&header, &entry, 64u * WEX_ALIGN + aligned));
    entry.flags = WEX_ENTRY_NEURAL_RANS3;
    entry.total_bytes = (uint32_t)aligned;  /* no smaller than raw */
    CHECK(wex_validate_entry(&header, &entry, 64u * WEX_ALIGN + aligned));
    entry.total_bytes = (uint32_t)aligned - WEX_ALIGN;
    header.flags = WEX_HEADER_SCALE_TILES | WEX_HEADER_NEURAL_RANS2;   /* codec 5 without flag 32 */
    CHECK(wex_validate_entry(&header, &entry, 64u * WEX_ALIGN + aligned));
    header.flags = WEX_HEADER_SCALE_TILES | WEX_HEADER_NEURAL_RANS3;
    memset(page, 0, sizeof(page));
    put_u32(page + 64u, WEX_NR2_DESCRIPTOR_MAGIC); put_u32(page + 68u, WEX_NR2_MODEL_BYTES);
    put_u32(page + 72u, WEX_ALIGN); put_u32(page + 80u, 0x1234u);
    CHECK(!wex_nr2_model_location(&header, page, &at, &bytes, &id) && id == 0x1234u);
    header.flags = WEX_HEADER_SCALE_TILES;
    CHECK(wex_nr2_model_location(&header, page, &at, &bytes, &id));
}

int main(void) {
    test_nr3_hash_vectors();
    test_nr3_container_rules();
    test_nr3_codec();
    test_nr3_audit();
    test_nr2_golden();
    test_nr2_codec();
    test_bit_vectors();
    test_tile_boundaries();
    test_headers_and_entries();
    test_record_decoder();
    test_palette_record(0);
    test_palette_record(1);
    test_v3_alias_manifest();
    test_v12_slab_bound();
    puts("WEX format: V1/V2/V3, all bit widths, palette/scale boundaries, overlapping expansion, WNR2 round trips and malformed streams passed.");
    return 0;
}
