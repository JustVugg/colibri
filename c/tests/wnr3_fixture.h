/* tests/wnr3_fixture.h -- WNR3 (codec 5) records for the tests, written with the reference
 * encoder of wex_neural_rans.h: a WNM2 model trained on a set of decoded expert records, one
 * record encoded, and the WNR3 form of a WEX4 container of RAW expert records (as
 * tests/wex4_fixture.h writes them): the same index tensors, every expert record re-encoded as
 * WNR3 (canonical zero on request), one WNM2 blob in front of the records. For tests only. */
#ifndef COLI_TESTS_WNR3_FIXTURE_H
#define COLI_TESTS_WNR3_FIXTURE_H
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../wex4.h"

#define WNR3F_WEIGHT_CHUNK 4096u
#define WNR3F_SCALE_CHUNK 256u

static void wnr3f_split(const WexEntry *e, unsigned char *raw, unsigned char *scales[3], unsigned char *weights[3]) {
    for (unsigned m = 0; m < 3u; m++) { scales[m] = raw + e->scale_offsets[m]; weights[m] = raw + e->weight_offsets[m]; }
}

/* coded = raw -> canonical zero (cz) -> ceil form; returns the record flags. */
static uint32_t wnr3f_code(const WexEntry *e, const WexNr2Layout *layout, const unsigned char *raw, unsigned char *coded,
                           int cz) {
    unsigned char *scales[3], *weights[3];
    memcpy(coded, raw, (size_t)e->scale_bytes + e->weight_bytes);
    if (cz) wex_nr3_canonical_zero(coded + e->scale_bytes, e->weight_bytes);
    wnr3f_split(e, coded, scales, weights);
    return wex_nr3_ceil_forward(layout, scales, weights, NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
}

/* A WNM2 model (blob of WEX_NR2_MODEL_BYTES) trained on n decoded records of entry e's layout. */
static void wnr3f_train(const WexEntry *e, const unsigned char *const *raw, size_t n, int cz, unsigned char *blob,
                        WexNr2Model *model) {
    static uint64_t wcounts[3u * WEX_NR2_CONTEXTS * 16u], scounts[3u * 256u];
    size_t bytes = (size_t)e->scale_bytes + e->weight_bytes;
    unsigned char *coded = malloc(bytes), *scales[3], *weights[3];
    WexNr2Layout layout;
    assert(coded);
    wex_nr2_layout(e, &layout);
    memset(wcounts, 0, sizeof wcounts); memset(scounts, 0, sizeof scounts);
    for (size_t i = 0; i < n; i++) {
        wnr3f_code(e, &layout, raw[i], coded, cz);
        wnr3f_split(e, coded, scales, weights);
        wex_nr3_count(&layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights, wcounts, scounts);
    }
    wex_nr2_model_build(wcounts, scounts, blob);
    assert(!wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, model));
    free(coded);
}

/* One record: raw (entry e's decoded layout) as WNR3 into out (zeroed to the page); returns the
 * stored size (a multiple of 4096) or 0 when it does not encode smaller than the RAW record. */
static size_t wnr3f_encode(const WexNr2Model *model, const WexEntry *e, const unsigned char *raw, int cz, uint32_t owner,
                           unsigned char *out, size_t capacity) {
    size_t bytes = (size_t)e->scale_bytes + e->weight_bytes, written = 0, stored;
    unsigned char *coded = malloc(bytes), *scales[3], *weights[3];
    WexNr2Layout layout;
    uint64_t raw_bytes, aligned;
    assert(coded && !wex_entry_raw_sizes(e, &raw_bytes, &aligned));
    wex_nr2_layout(e, &layout);
    uint32_t flags = wnr3f_code(e, &layout, raw, coded, cz);
    wnr3f_split(e, coded, scales, weights);
    memset(out, 0, capacity);
    int rc = wex_nr3_encode(model, &layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                            WNR3F_WEIGHT_CHUNK, WNR3F_SCALE_CHUNK, flags, owner, out, capacity, &written);
    free(coded);
    if (rc) return 0;
    stored = (written + 4095u) & ~(size_t)4095u;
    return stored < aligned && stored <= capacity ? stored : 0;
}

/* The WNR3 form of the WEX4 container w4 (bytes): every RAW expert record encoded as WNR3 with a
 * model trained on all of them (cz: canonical zero, and the value-exact flags), behind one WNM2
 * blob at page 1; the other index tables are w4's. offset[cell] (layers x experts of w4's grid,
 * may be NULL) gets each record's offset, 0 for a cell without one; root its new root. Returns
 * the file (malloc), or NULL when a record does not encode smaller than RAW. */
static unsigned char *wnr3f_from_wex4(const unsigned char *w4, size_t bytes, int cz, size_t *out_bytes, char root[65],
                                      size_t *offset) {
    Wex4Header h;
    Wex4View v;
    Wex4Record rec;
    WexEntry e;
    WexNr2Model model;
    unsigned char *blob = malloc(WEX_NR2_MODEL_BYTES);
    assert(blob && !wex4_parse_header(w4, bytes, &h));
    const unsigned char *index = w4 + h.index_offset;
    assert(!wex4_open(&h, index, (size_t)wex4_align(h.index_bytes), &v));
    const unsigned char **raw = malloc((size_t)h.records * sizeof *raw);
    size_t n = 0;
    for (uint32_t r = 0; r < h.records; r++) {
        wex4_get_record(&v, r, &rec);
        assert(rec.codec == WEX4_CODEC_RAW && (rec.flags & WEX4_REC_EXPERT));
        raw[n++] = w4 + rec.offset;
    }
    assert(!wex4_expert_entry(&v, 0, &e));
    wnr3f_train(&e, raw, n, cz, blob, &model);
    /* the new file: page 0, the blob, the records, the index (blobs: the one WNM2 blob) */
    const size_t blob_stored = (size_t)wex4_align(WEX_NR2_MODEL_BYTES), old_tables = (size_t)h.blobs * WEX4_BLOB_BYTES;
    const size_t index_bytes = (size_t)h.index_bytes - old_tables + WEX4_BLOB_BYTES;
    size_t capacity = WEX4_PAGE + blob_stored + (size_t)h.records * (size_t)wex4_align(h.max_decoded) +
                      (size_t)wex4_align(index_bytes);
    unsigned char *out = calloc(1, capacity);
    size_t at = WEX4_PAGE + blob_stored;
    assert(out);
    memcpy(out + WEX4_PAGE, blob, WEX_NR2_MODEL_BYTES);
    unsigned char *x = malloc(index_bytes);
    assert(x);
    {   /* blob 1: kind WNM2, the model's id */
        unsigned char *p = x;
        uint32_t kind = WEX4_BLOB_WNM2, id = model.id;
        uint64_t boff = WEX4_PAGE, bbytes = WEX_NR2_MODEL_BYTES;
        memset(p, 0, WEX4_BLOB_BYTES);
        memcpy(p, &kind, 4); memcpy(p + 4, &id, 4); memcpy(p + 8, &boff, 8); memcpy(p + 16, &bbytes, 8);
        wex4_sha256(blob, WEX_NR2_MODEL_BYTES, p + 24);
    }
    memcpy(x + WEX4_BLOB_BYTES, index + old_tables, (size_t)h.index_bytes - old_tables);   /* records, tensors, ... */
    if (offset) memset(offset, 0, (size_t)h.layers * h.experts * sizeof *offset);
    for (uint32_t r = 0; r < h.records; r++) {
        unsigned char *p = x + WEX4_BLOB_BYTES + (size_t)r * WEX4_RECORD_BYTES;
        wex4_get_record(&v, r, &rec);
        assert(!wex4_expert_entry(&v, r, &e));
        size_t stored = wnr3f_encode(&model, &e, w4 + rec.offset, cz, rec.owner, out + at, capacity - at);
        if (!stored) { free(out); free(x); free(blob); free(raw); return NULL; }
        uint64_t off64 = at;
        uint32_t stored32 = (uint32_t)stored, blob1 = 1;
        uint16_t codec = WEX4_CODEC_WNR3, flags = (uint16_t)(WEX4_REC_EXPERT | (cz ? WEX4_REC_VALUE_EXACT : 0u));
        memcpy(p, &off64, 8); memcpy(p + 8, &stored32, 4); memcpy(p + 16, &codec, 2); memcpy(p + 18, &flags, 2);
        memcpy(p + 20, &blob1, 4);
        wex4_sha256(out + at, stored, p + 36);
        if (offset) offset[(size_t)(rec.owner >> 16) * h.experts + (rec.owner & 0xffffu)] = at;
        at += stored;
    }
    const size_t index_offset = at, file = index_offset + (size_t)wex4_align(index_bytes);
    memcpy(out + index_offset, x, index_bytes);
    {   /* the header page: w4's, with the new blobs, sizes, offsets, value-exact feature and hashes */
        uint64_t required = cz ? WEX4_REQ_VALUE_EXACT : 0, file64 = file, ioff = index_offset, ibytes = index_bytes;
        uint32_t blobs = 1;
        memcpy(out, w4, WEX4_PAGE);
        memcpy(out + 8, &required, 8); memcpy(out + 24, &file64, 8); memcpy(out + 32, &ioff, 8);
        memcpy(out + 40, &ibytes, 8); memcpy(out + 48, &blobs, 4);
        wex4_sha256(out + index_offset, index_bytes, out + 96);
        wex4_sha256(out, WEX4_HEADER_HASHED, out + WEX4_HEADER_HASHED);
    }
    for (int i = 0; i < 32; i++) snprintf(root + i * 2, 3, "%02x", out[WEX4_HEADER_HASHED + i]);
    free(x); free(blob); free(raw);
    *out_bytes = file;
    {   /* what an engine checks at open */
        Wex4Header check;
        Wex4View view;
        assert(!wex4_parse_header(out, file, &check) &&
               !wex4_open(&check, out + index_offset, (size_t)wex4_align(index_bytes), &view));
    }
    return out;
}

/* In the WNR3 container w4 (bytes, as wnr3f_from_wex4 wrote it): the record of expert `owner`
 * (layer << 16 | expert) encoded again from raw, its decoded bytes, under the name new_owner, its
 * hash in the index updated and the file resealed (root: the new root). The record is whole and
 * its hash matches, but it says it is another expert. 0, or -1 when its size would change. */
static int wnr3f_rename(unsigned char *w4, size_t bytes, uint32_t owner, uint32_t new_owner, const unsigned char *raw,
                        char root[65]) {
    Wex4Header h;
    Wex4View v;
    Wex4Blob blob;
    WexNr2Model model;
    assert(!wex4_parse_header(w4, bytes, &h) && !wex4_open(&h, w4 + h.index_offset, (size_t)wex4_align(h.index_bytes), &v));
    wex4_get_blob(&v, 0, &blob);
    assert(!wex_nr2_model_open(w4 + blob.offset, (size_t)blob.bytes, &model));
    for (uint32_t r = 0; r < h.records; r++) {
        Wex4Record rec;
        WexEntry e;
        wex4_get_record(&v, r, &rec);
        if (rec.owner != owner) continue;
        assert(!wex4_expert_entry(&v, r, &e));
        unsigned char *out = calloc(1, rec.stored);
        size_t stored = out ? wnr3f_encode(&model, &e, raw, !!(rec.flags & WEX4_REC_VALUE_EXACT), new_owner, out,
                                           rec.stored) : 0;
        if (stored != rec.stored) { free(out); return -1; }
        memcpy(w4 + rec.offset, out, stored);
        wex4_sha256(out, stored, (unsigned char *)rec.sha);   /* rec.sha points into w4's index */
        free(out);
        wex4_sha256(w4 + h.index_offset, (size_t)h.index_bytes, w4 + 96);
        wex4_sha256(w4, WEX4_HEADER_HASHED, w4 + WEX4_HEADER_HASHED);
        for (int i = 0; i < 32; i++) snprintf(root + i * 2, 3, "%02x", w4[WEX4_HEADER_HASHED + i]);
        return 0;
    }
    return -1;
}
#endif
