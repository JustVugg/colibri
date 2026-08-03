/* rans_ctypes.c — thin exported-symbol bridge over rans.h for the Python
 * offline tools (repack_rans.py, rans_verify.py), built as a shared library
 * by `make rans` (same optional-accelerator shape as tools/iq3_encode.c:
 * the Python side falls back to a slow pure-Python codec producing the SAME
 * bytes when this library is absent, and the Python e2e test pins the two
 * implementations byte-identical).
 *
 * Everything in rans.h is static; these wrappers are the only symbols the
 * library exports. No engine code is included or touched. */
#include "../rans.h"

#ifdef _WIN32
#define RC_EXPORT __declspec(dllexport)
#else
#define RC_EXPORT
#endif

/* This bridge serves the int4-rans256-g0 TOOLS, so unlike the format-generic
 * header entry points it can and does pin n_streams to the format's 256 —
 * a zero or wrong-for-format value is refused by name before any division
 * or decode dispatch (the header entries additionally refuse zero). */
static int rc__nstreams_ok(uint32_t n_streams) { return n_streams == RANS_NSTREAMS; }

/* Encode n nibbles into one chunk record. Returns the record length, or the
 * negated rans_err on failure. out_cap should come from rc_record_bound. */
RC_EXPORT int64_t rc_record_encode(const uint8_t *nibbles, uint64_t n, const uint32_t *freq, const uint32_t *start,
                                   const uint16_t *slot_to_symbol, uint32_t scale_bits, uint32_t n_streams,
                                   uint8_t *out, uint64_t out_cap) {
    if (!rc__nstreams_ok(n_streams)) return -(int64_t)RANS_E_NSTREAMS;
    rans_table t;
    rans_err e = rans_table_init(&t, scale_bits, freq, start, slot_to_symbol);
    if (e != RANS_OK) return -(int64_t)e;
    uint64_t len = 0;
    e = rans_record_encode(nibbles, n, &t, n_streams, out, out_cap, &len);
    rans_table_free(&t);
    if (e != RANS_OK) return -(int64_t)e;
    return (int64_t)len;
}

RC_EXPORT uint64_t rc_record_bound(uint64_t n_symbols, uint32_t n_streams) {
    if (!rc__nstreams_ok(n_streams)) return 0; /* callers treat 0 as refusal */
    return rans_record_bound(n_symbols, n_streams);
}

/* Parse + fully validate a record blob; fills n_symbols/packed_bytes on
 * success. Returns a rans_err. */
RC_EXPORT int32_t rc_record_parse(const uint8_t *blob, uint64_t blob_len, uint32_t n_streams, uint64_t *n_symbols,
                                  uint64_t *packed_bytes) {
    if (!rc__nstreams_ok(n_streams)) return (int32_t)RANS_E_NSTREAMS;
    rans_record rec;
    rans_err e = rans_record_parse(blob, blob_len, n_streams, &rec);
    if (e == RANS_OK) {
        if (n_symbols) *n_symbols = rec.n_symbols;
        if (packed_bytes) *packed_bytes = rec.packed_bytes;
    }
    return (int32_t)e;
}

/* Parse + decode a record blob to packed bytes with the requested path
 * (path < 0 => rans_path_select(): env-honouring best arm). out_packed must
 * hold the record's packed_bytes; the CALLER must satisfy the RANS_SLACK
 * contract (blob allocated with >= RANS_SLACK readable bytes past blob_len —
 * the Python side copies each record into a padded buffer). */
RC_EXPORT int32_t rc_record_decode(const uint8_t *blob, uint64_t blob_len, uint32_t n_streams, const uint32_t *freq,
                                   const uint32_t *start, const uint16_t *slot_to_symbol, uint32_t scale_bits,
                                   int32_t path, uint8_t *out_packed) {
    if (!rc__nstreams_ok(n_streams)) return (int32_t)RANS_E_NSTREAMS;
    rans_record rec;
    rans_err e = rans_record_parse(blob, blob_len, n_streams, &rec);
    if (e != RANS_OK) return (int32_t)e;
    rans_table t;
    e = rans_table_init(&t, scale_bits, freq, start, slot_to_symbol);
    if (e != RANS_OK) return (int32_t)e;
    rans_path p = (path < 0) ? rans_path_select() : (rans_path)path;
    e = rans_record_decode_packed(&rec, &t, n_streams, p, out_packed);
    rans_table_free(&t);
    return (int32_t)e;
}

/* Checked per-stream decode (verification invariants: state range, exact
 * consumption, final state). Returns a rans_err. */
RC_EXPORT int32_t rc_stream_decode_checked(const uint8_t *in_buf, uint64_t in_size, uint64_t n, const uint32_t *freq,
                                           const uint32_t *start, const uint16_t *slot_to_symbol, uint32_t scale_bits,
                                           uint8_t *out_nibbles) {
    return (int32_t)rans_decode_stream_checked(in_buf, (size_t)in_size, (size_t)n, freq, start, slot_to_symbol,
                                               scale_bits, out_nibbles);
}

RC_EXPORT const char *rc_err_name(int32_t e) { return rans_err_name((rans_err)(e < 0 ? -e : e)); }

RC_EXPORT const char *rc_path_name(void) { return rans_path_name(rans_path_select()); }
