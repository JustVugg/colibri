/* WNR3 encoder side for tools/build_nr3.py (ctypes). CPU only.
 * Build: gcc -O2 -shared -o wex_nr3_tools.dll tools/wex_nr3_tools.c
 * Every function is thread-safe (no globals); buffers belong to the caller.
 * The codec itself lives in wex_neural_rans.h / wex_format.h; this file only
 * wires it to whole experts (WexEntry + canonical raw slab). */
#include "../wex_format.h"

#ifdef _WIN32
#define WEX_NR3_EXPORT __declspec(dllexport)
#else
#define WEX_NR3_EXPORT __attribute__((visibility("default")))
#endif

WEX_NR3_EXPORT uint32_t wex_nr3_tools_abi(void) { return 1u; }

static int nr3_views(const WexEntry *e, const unsigned char *slab, WexNr2Layout *layout,
                     unsigned char *scales[3], unsigned char *weights[3], uint64_t *raw) {
    uint64_t r, aligned;
    unsigned int m;
    if (!e || !slab || wex_entry_raw_sizes(e, &r, &aligned)) return -1;
    wex_nr2_layout(e, layout);
    for (m = 0; m < 3u; m++) {
        scales[m] = (unsigned char *)slab + e->scale_offsets[m];
        weights[m] = (unsigned char *)slab + e->weight_offsets[m];
    }
    if (raw) *raw = r;
    return 0;
}

/* coded = source -> canonical zero (cz) -> ceil form. Returns the record
 * flags (ceil mask | CZ) or -1. stats (16 u32, optional): [3m + k] = matrix m
 * {transformed blocks, max-5 blocks, even blocks kept at 255}, [9..11] rule-g
 * anchors of the coded scales, [12] flags, [13] / [14] low / high word of the
 * count of -0 nibbles mapped to +0. */
WEX_NR3_EXPORT int wex_nr3_prepare(const WexEntry *e, const unsigned char *source, int cz,
                                   unsigned char *coded, uint32_t *stats) {
    WexNr2Layout layout;
    unsigned char *scales[3], *weights[3];
    uint32_t st[3][3], flags, m;
    uint64_t raw, mapped = 0;
    if (!coded || nr3_views(e, coded, &layout, scales, weights, &raw)) return -1;
    memcpy(coded, source, (size_t)raw);
    if (cz) mapped = wex_nr3_canonical_zero(coded + e->scale_bytes, e->weight_bytes);
    flags = wex_nr3_ceil_forward(&layout, scales, weights, st) | (cz ? WEX_NR3_FLAG_CZ : 0u);
    if (stats) {
        for (m = 0; m < 3u; m++) {
            stats[3u * m] = st[m][0]; stats[3u * m + 1u] = st[m][1]; stats[3u * m + 2u] = st[m][2];
            stats[9u + m] = wex_nr3_anchor(scales[m], layout.scale_count[m]);
        }
        stats[12] = flags; stats[13] = (uint32_t)mapped; stats[14] = (uint32_t)(mapped >> 32); stats[15] = 0;
    }
    return (int)flags;
}

/* Training counts of one coded expert, added to wc[3*567*16] / sc[3*256]. */
WEX_NR3_EXPORT int wex_nr3_count_expert(const WexEntry *e, const unsigned char *coded, uint64_t *wc, uint64_t *sc) {
    WexNr2Layout layout;
    unsigned char *scales[3], *weights[3];
    if (!wc || !sc || nr3_views(e, coded, &layout, scales, weights, NULL)) return -1;
    wex_nr3_count(&layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights, wc, sc);
    return 0;
}

/* WNM2 blob (WEX_NR2_MODEL_BYTES) from counts; *id = its FNV-1a id. */
WEX_NR3_EXPORT int wex_nr3_build_model(const uint64_t *wc, const uint64_t *sc, unsigned char *blob, uint32_t *id) {
    WexNr2Model model;
    if (!wc || !sc || !blob) return -1;
    wex_nr2_model_build(wc, sc, blob);
    if (wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model)) return -1;
    if (id) *id = model.id;
    return 0;
}

WEX_NR3_EXPORT int wex_nr3_model_id(const unsigned char *blob, uint32_t *id) {
    WexNr2Model model;
    if (!id || wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model)) return -1;
    *id = model.id;
    return 0;
}

/* Encode a coded expert: out (capacity cap, >= 4x the raw slab is always
 * enough) receives the record zero-padded to *total (used bytes in *used).
 * Returns 0, 1 when the record is not smaller than the aligned raw slab (the
 * builder then stores codec 0 with the SOURCE bytes), -1 on error. */
WEX_NR3_EXPORT int wex_nr3_encode_record(const unsigned char *blob, const WexEntry *e, const unsigned char *coded,
                                         uint32_t flags, uint32_t owner, uint32_t wchunk, uint32_t schunk,
                                         unsigned char *out, size_t cap, size_t *used, size_t *total) {
    WexNr2Model model;
    WexNr2Layout layout;
    unsigned char *scales[3], *weights[3];
    uint64_t raw, aligned;
    size_t written = 0, padded;
    if (!out || !used || !total || wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model) ||
        nr3_views(e, coded, &layout, scales, weights, &raw) || wex_entry_raw_sizes(e, &raw, &aligned)) return -1;
    if (wex_nr3_encode(&model, &layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                       wchunk, schunk, flags, owner, out, cap, &written)) return -1;
    padded = (written + 4095u) & ~(size_t)4095u;
    if (padded > cap) return -1;
    memset(out + written, 0, padded - written);
    *used = written; *total = padded;
    return padded >= aligned ? 1 : 0;
}

/* CPU reference decode of a record (entry codec 5, total_bytes = record size)
 * into out (raw slab bytes). Returns the decoder status (0 / -1); out is
 * untouched unless 0. Checks the owner (layer << 16 | expert) first. */
WEX_NR3_EXPORT int wex_nr3_decode(const unsigned char *blob, const WexEntry *e, const unsigned char *record,
                                  uint32_t layer, uint32_t expert, unsigned char *out) {
    WexNr2Model model;
    uint64_t raw, aligned;
    size_t at;
    unsigned char *slab;
    int status;
    if (!e || !record || !out || wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model) ||
        wex_entry_raw_sizes(e, &raw, &aligned) || !wex_entry_nr_model(e) || e->total_bytes >= aligned) return -1;
    /* no container here: the record's own profile; build_nr3.py checks each record's
     * CZ bit against the container header it writes */
    if (!wex_nr3_owner_ok(e, record, layer, expert,
                          record[27] & WEX_NR3_FLAG_CZ ? WEX_HEADER_CANONICAL_ZERO : 0u)) return -1;
    slab = (unsigned char *)malloc((size_t)aligned);
    if (!slab) return -1;
    at = wex_record_read_offset(e);
    memcpy(slab + at, record, e->total_bytes);
    status = wex_decode_nr2_record(e, slab, (size_t)aligned, &model);
    if (!status) memcpy(out, slab, (size_t)raw);
    free(slab);
    return status;
}

/* 0 iff the record decodes on the CPU to exactly `expected` (source, or
 * Z(source) for a CZ record) for (layer, expert). */
WEX_NR3_EXPORT int wex_nr3_verify(const unsigned char *blob, const WexEntry *e, const unsigned char *record,
                                  uint32_t layer, uint32_t expert, const unsigned char *expected) {
    uint64_t raw, aligned;
    unsigned char *out;
    int result;
    if (!e || !expected || wex_entry_raw_sizes(e, &raw, &aligned)) return -1;
    out = (unsigned char *)malloc((size_t)raw);
    if (!out) return -1;
    result = wex_nr3_decode(blob, e, record, layer, expert, out) || memcmp(out, expected, (size_t)raw) ? -1 : 0;
    free(out);
    return result;
}

/* Z: -0 -> +0 in place over the weight bytes of a raw slab; returns the count. */
WEX_NR3_EXPORT uint64_t wex_nr3_canonicalize(const WexEntry *e, unsigned char *slab) {
    return e && slab ? wex_nr3_canonical_zero(slab + e->scale_bytes, e->weight_bytes) : 0;
}

/* Container structure: header page (4096 B), the model blob the descriptor
 * names, the full entry table and the file size. 0 when valid. */
WEX_NR3_EXPORT int wex_nr3_check_container(const unsigned char *page, const unsigned char *blob,
                                           const WexEntry *entries, uint64_t file_bytes) {
    WexHeader hdr;
    WexNr2Model model;
    uint64_t at;
    uint32_t bytes, id;
    if (!page || !blob || !entries) return -1;
    memcpy(&hdr, page, sizeof(hdr));
    if (wex_validate_header(&hdr, file_bytes, 0, 0, NULL, NULL) || !(hdr.flags & WEX_HEADER_NEURAL_RANS3) ||
        wex_nr2_model_location(&hdr, page, &at, &bytes, &id) ||
        wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model) || model.id != id ||
        wex_validate_table(&hdr, entries, file_bytes)) return -1;
    return 0;
}
