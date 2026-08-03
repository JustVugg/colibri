/* rans.h — static-table byte-renormalized rANS (range Asymmetric Numeral
 * Systems) codec for 4-bit (nibble) alphabets, plus the `int4-rans256-g0`
 * chunk-record reader/writer the offline tools (tools/repack_rans.py,
 * tools/rans_verify.py) and a future engine decode stage share.
 *
 * Header-only, all functions static, dependency-free C99 — same shape as
 * st.h/quant.h: the engine is a single translation unit and tests include
 * this header directly. No engine types are used or touched here.
 *
 * CODEC (ryg_rans-style construction, Fabian Giesen's public-domain
 * reference design): 32-bit state, byte renormalization, division-based
 * encode step, table-lookup decode step. Encode walks the input BACKWARDS
 * and fills the output buffer from the END backwards (standard rANS LIFO
 * construction); decode reads the resulting byte range FORWARDS and emits
 * symbols in the ORIGINAL forward order.
 *
 * RECORD FORMAT (`int4-rans256-g0`, one record = one tensor's nibble
 * stream; all integers little-endian; docs/int4-rans256-g0.md is the full
 * writeup):
 *
 *   offset 0:  n_symbols     u64   -- nibble count
 *   offset 8:  packed_bytes  u64   -- original packed-byte count
 *                                     (== ceil(n_symbols/2)), stored so a
 *                                     reader can size its output buffer
 *                                     without recomputing it
 *   offset 16: stream_offsets[N+1]  u32 x (N+1)  -- N = n_streams (256 for
 *                                     this format); offsets[i] is the byte
 *                                     offset, relative to the start of
 *                                     `payload`, where stream i begins;
 *                                     offsets[N] is the total payload length
 *   then:      zero-pad to the next 16-byte boundary (derived, never stored)
 *   payload:   N independent rANS byte streams, concatenated in stream
 *              order: stream i is payload[offsets[i] : offsets[i+1]]
 *   then:      zero-pad to the next 16-byte boundary (derived, never stored)
 *
 * INTERLEAVING: nibble at logical index j belongs to stream (j % N) and is
 * the (j / N)-th symbol in that stream's own order. Every stream is an
 * ordinary self-contained single-state rANS stream over the SAME shared
 * static table; there is no cross-stream coupling. Round-robin (not block)
 * assignment is what makes wide decoders output-coalesced: lane l of a
 * group based at stream b emits logical position r*N+b+l each round, so a
 * group of G lanes writes G contiguous nibbles = G/2 whole packed bytes.
 *
 * BATCHED DECODE ARMS + ENVELOPE: scalar (branchy, structurally the
 * reference decoder), scalar_bf (branch-free closed-form renorm — the same
 * algebra the SIMD arms use, kept so that algebra is testable on machines
 * without the ISA), NEON and AVX-512 (F+BW). Each vector arm follows the
 * same discipline as quant.h's AVX-512 i4 accumulator: a compile-time ISA
 * gate, a first-use encode/decode round-trip selftest (a failing arm is
 * disabled loudly, never used), and an env kill-switch (RANS_NEON=0 /
 * RANS_AVX512=0). RANS_PATH=scalar|scalar_bf|neon|avx512 forces one arm and
 * REFUSES (no silent downgrade) if it is not available: a forced path that
 * quietly runs something else would turn "we tested avx512" into a false
 * claim. All arms are byte-identical to scalar by contract — this is a
 * codec, not a float kernel; there is no tolerance, only identity.
 *
 * SLACK CONTRACT: the branch-free and SIMD arms read 4 bytes at a cursor
 * that may sit at a stream's end, and gather 4 bytes at a symbol-table
 * index that may be the last slot. Callers of the batched decode entry
 * points must therefore hand in (a) a record buffer with at least
 * RANS_SLACK readable bytes past its end and (b) a rans_table whose sym8
 * mirror was built by rans_table_init (which allocates the slack). The
 * offline tools copy each record into a slack-padded scratch buffer; an
 * engine consumer must do the equivalent. The strictly scalar entry points
 * (rans_decode_stream / RANS_PATH_SCALAR) never over-read.
 */
#ifndef COLI_RANS_H
#define COLI_RANS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ENDIANNESS PRECONDITION: record header integers (n_symbols, packed_bytes,
 * stream_offsets) are little-endian on the wire, and this implementation
 * reads/writes them with native-order memcpy — correct only on little-endian
 * hosts. Refuse to compile elsewhere rather than emit byte-swapped records
 * that silently break the cross-host byte-identity contract. */
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "rans.h assumes a little-endian host (record integers are little-endian)"
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define RANS_L (1u << 23) /* lower bound of the normalization interval */
#define RANS_ALPHABET 16
#define RANS_NSTREAMS 256      /* int4-rans256-g0's stream count */
#define RANS_SLACK 64          /* readable bytes required past record buffers */
#define RANS_SCALE_BITS_MAX 15 /* rans_table_init's upper bound on scale_bits */

/* ---- scalar stream codec (ported unmodified from the proven prototype) --- */

/* Encode n nibbles (values 0..15) into out_buf writing BACKWARDS from
 * out_buf+out_cap. freq[16] must sum to 1<<scale_bits and start[] must be its
 * exclusive prefix sum; every symbol present in the input needs freq > 0.
 * Returns the byte OFFSET where the valid stream begins (stream occupies
 * out_buf[offset..out_cap)), or (size_t)-1 if out_cap was too small. */
static size_t rans_encode_stream(const uint8_t *nibbles, size_t n, const uint32_t *freq, const uint32_t *start,
                                 uint32_t scale_bits, uint8_t *out_buf, size_t out_cap) {
    uint32_t x = RANS_L;
    uint8_t *ptr = out_buf + out_cap; /* write backwards from the end */
    for (size_t i = n; i-- > 0;) {
        /* mask to the nibble alphabet: an out-of-range input byte must never
         * index past freq[16]/start[16] (record-level entry points refuse
         * such input by name BEFORE encoding; the mask makes this primitive
         * memory-safe for any bytes regardless) */
        uint32_t s = nibbles[i] & 15u;
        uint32_t f = freq[s];
        uint32_t st = start[s];
        /* renormalize: keep x below x_max so the update keeps x in [L, L*256) */
        uint32_t x_max = ((RANS_L >> scale_bits) << 8) * f;
        while (x >= x_max) {
            if (ptr <= out_buf) return (size_t)-1; /* overflow guard */
            *--ptr = (uint8_t)(x & 0xff);
            x >>= 8;
        }
        x = ((x / f) << scale_bits) + (x % f) + st;
    }
    for (int b = 0; b < 4; b++) { /* flush final state, 4 bytes */
        if (ptr <= out_buf) return (size_t)-1;
        *--ptr = (uint8_t)(x & 0xff);
        x >>= 8;
    }
    return (size_t)(ptr - out_buf);
}

/* Decode n nibbles from the stream produced by rans_encode_stream.
 * slot_to_symbol maps a slot (x & (M-1)) to its symbol — the O(1) lookup that
 * makes decode fast. Returns 0 on success, 1 on buffer underrun. The
 * `else break` in the renorm is load-bearing: the encoder stops feeding bytes
 * once drained, so a stream's final symbols legally decode with x < RANS_L;
 * a decoder that reads past `end` instead produces wrong tail symbols. */
static int rans_decode_stream(const uint8_t *in_buf, size_t in_size, size_t n, const uint32_t *freq,
                              const uint32_t *start, const uint16_t *slot_to_symbol, uint32_t scale_bits,
                              uint8_t *out_nibbles) {
    const uint8_t *ptr = in_buf;
    const uint8_t *end = in_buf + in_size;
    uint32_t mask = (1u << scale_bits) - 1;
    uint32_t x = 0;
    for (int b = 0; b < 4; b++) {
        if (ptr >= end) return 1;
        x = (x << 8) | *ptr++;
    }
    for (size_t i = 0; i < n; i++) {
        uint32_t slot = x & mask;
        uint32_t s = slot_to_symbol[slot];
        x = freq[s] * (x >> scale_bits) + slot - start[s];
        while (x < RANS_L) {
            if (ptr >= end) break;
            x = (x << 8) | *ptr++;
        }
        out_nibbles[i] = (uint8_t)s;
    }
    return 0;
}

/* Checked variant for verification: additionally enforces the invariants a
 * genuine encoder output always satisfies, so a validator can refuse a
 * corrupted stream WITHOUT ground-truth bytes. Returns one of RANS_OK /
 * RANS_E_STREAM_STATE_RANGE / RANS_E_STREAM_UNDERRUN /
 * RANS_E_STREAM_LEFTOVER / RANS_E_STREAM_FINAL_STATE (see rans_err below). */

/* ---- named error codes --------------------------------------------------- */

typedef enum {
    RANS_OK = 0,
    /* record-level refusals (rans_record_parse) */
    RANS_E_TRUNCATED,         /* blob shorter than its own framing claims  */
    RANS_E_EMPTY,             /* n_symbols == 0                            */
    RANS_E_COUNT_MISMATCH,    /* packed_bytes != ceil(n_symbols/2)         */
    RANS_E_OVERSIZE,          /* n_symbols impossibly large for payload    */
    RANS_E_OFFSET_FIRST,      /* stream_offsets[0] != 0                    */
    RANS_E_OFFSETS_MONOTONIC, /* stream_offsets not non-decreasing         */
    RANS_E_STREAM_SHORT,      /* a stream shorter than its 4-byte state    */
    RANS_E_LENGTH_MISMATCH,   /* blob length != derived framing length     */
    RANS_E_PAD_NONZERO,       /* a derived padding byte is not zero        */
    RANS_E_UNALIGNED,         /* record buffer not 4-byte aligned          */
    RANS_E_NSTREAMS,          /* n_streams zero / wrong for the format     */
    /* table-level refusals (rans_table_init) */
    RANS_E_TABLE_SCALE,    /* scale_bits out of range (1..15)           */
    RANS_E_TABLE_FREQ_SUM, /* freq[] does not sum to M                  */
    RANS_E_TABLE_START,    /* start[] not the prefix sum of freq[]      */
    RANS_E_TABLE_SLOT,     /* slot_to_symbol inconsistent with freq[]   */
    /* stream-level refusals (rans_decode_stream_checked)                   */
    RANS_E_STREAM_STATE_RANGE, /* initial state outside [L, 256L)           */
    RANS_E_STREAM_UNDERRUN,    /* fewer than 4 bytes for the initial state  */
    RANS_E_STREAM_LEFTOVER,    /* decode finished with bytes unconsumed     */
    RANS_E_STREAM_FINAL_STATE, /* final state != L (not an encoder output)  */
    /* encode-side refusals */
    RANS_E_SYMBOL_UNCODABLE, /* input contains a value > 15 or a symbol
                                whose table frequency is zero             */
    RANS_E_SCRATCH,          /* encoder scratch/output bound exhausted
                                (a bound failure, NOT a memory condition) */
    /* misc */
    RANS_E_NOMEM,
    RANS_E_PATH_UNAVAILABLE /* RANS_PATH forced an arm this build/CPU/env
                               cannot provide (never silently downgraded) */
} rans_err;

static const char *rans_err_name(rans_err e) {
    switch (e) {
    case RANS_OK:
        return "OK";
    case RANS_E_TRUNCATED:
        return "E_TRUNCATED";
    case RANS_E_EMPTY:
        return "E_EMPTY";
    case RANS_E_COUNT_MISMATCH:
        return "E_COUNT_MISMATCH";
    case RANS_E_OVERSIZE:
        return "E_OVERSIZE";
    case RANS_E_OFFSET_FIRST:
        return "E_OFFSET_FIRST";
    case RANS_E_OFFSETS_MONOTONIC:
        return "E_OFFSETS_MONOTONIC";
    case RANS_E_STREAM_SHORT:
        return "E_STREAM_SHORT";
    case RANS_E_LENGTH_MISMATCH:
        return "E_LENGTH_MISMATCH";
    case RANS_E_PAD_NONZERO:
        return "E_PAD_NONZERO";
    case RANS_E_UNALIGNED:
        return "E_UNALIGNED";
    case RANS_E_NSTREAMS:
        return "E_NSTREAMS";
    case RANS_E_SYMBOL_UNCODABLE:
        return "E_SYMBOL_UNCODABLE";
    case RANS_E_SCRATCH:
        return "E_SCRATCH";
    case RANS_E_TABLE_SCALE:
        return "E_TABLE_SCALE";
    case RANS_E_TABLE_FREQ_SUM:
        return "E_TABLE_FREQ_SUM";
    case RANS_E_TABLE_START:
        return "E_TABLE_START";
    case RANS_E_TABLE_SLOT:
        return "E_TABLE_SLOT";
    case RANS_E_STREAM_STATE_RANGE:
        return "E_STREAM_STATE_RANGE";
    case RANS_E_STREAM_UNDERRUN:
        return "E_STREAM_UNDERRUN";
    case RANS_E_STREAM_LEFTOVER:
        return "E_STREAM_LEFTOVER";
    case RANS_E_STREAM_FINAL_STATE:
        return "E_STREAM_FINAL_STATE";
    case RANS_E_NOMEM:
        return "E_NOMEM";
    case RANS_E_PATH_UNAVAILABLE:
        return "E_PATH_UNAVAILABLE";
    }
    return "E_UNKNOWN";
}

static rans_err rans_decode_stream_checked(const uint8_t *in_buf, size_t in_size, size_t n, const uint32_t *freq,
                                           const uint32_t *start, const uint16_t *slot_to_symbol, uint32_t scale_bits,
                                           uint8_t *out_nibbles) {
    const uint8_t *ptr = in_buf;
    const uint8_t *end = in_buf + in_size;
    uint32_t mask = (1u << scale_bits) - 1;
    uint32_t x = 0;
    if (in_size < 4) return RANS_E_STREAM_UNDERRUN;
    for (int b = 0; b < 4; b++) x = (x << 8) | *ptr++;
    /* the encoder's state invariant: x stays in [L, 256L) at every step, so
     * the flushed initial state must land there too */
    if (x < RANS_L) return RANS_E_STREAM_STATE_RANGE; /* x < 256L is implicit: 32-bit */
    for (size_t i = 0; i < n; i++) {
        uint32_t slot = x & mask;
        uint32_t s = slot_to_symbol[slot];
        x = freq[s] * (x >> scale_bits) + slot - start[s];
        while (x < RANS_L) {
            if (ptr >= end) break;
            x = (x << 8) | *ptr++;
        }
        out_nibbles[i] = (uint8_t)s;
    }
    /* a genuine encoder output is consumed exactly and drains back to the
     * encoder's initial state — anything else is corruption */
    if (ptr != end) return RANS_E_STREAM_LEFTOVER;
    if (x != RANS_L) return RANS_E_STREAM_FINAL_STATE;
    return RANS_OK;
}

/* ---- shared static table -------------------------------------------------- */

typedef struct {
    uint32_t scale_bits;
    uint32_t M; /* 1u << scale_bits */
    uint32_t freq[RANS_ALPHABET];
    uint32_t start[RANS_ALPHABET];
    const uint16_t *slot_to_symbol; /* borrowed, M entries */
    uint8_t *sym8;                  /* owned u8 mirror, M + RANS_SLACK bytes:
                                       halves the hot-table footprint and lets
                                       the SIMD arms use a scale-1 dword gather */
} rans_table;

/* Validate freq/start/slot_to_symbol coherence and build the sym8 mirror.
 * slot_to_symbol is borrowed (must outlive the table); sym8 is owned — free
 * with rans_table_free. On failure the table is zeroed and the named error
 * returned; decode with an unvalidated table would be garbage in a way no
 * downstream byte comparison could explain, so this refuses instead. */
static rans_err rans_table_init(rans_table *t, uint32_t scale_bits, const uint32_t *freq, const uint32_t *start,
                                const uint16_t *slot_to_symbol) {
    memset(t, 0, sizeof(*t));
    if (scale_bits < 1 || scale_bits > 15) return RANS_E_TABLE_SCALE;
    uint32_t M = 1u << scale_bits;
    uint64_t sum = 0;
    for (int s = 0; s < RANS_ALPHABET; s++) {
        if (start[s] != sum) return RANS_E_TABLE_START;
        sum += freq[s];
    }
    if (sum != M) return RANS_E_TABLE_FREQ_SUM;
    for (uint32_t i = 0; i < M; i++) {
        uint16_t s = slot_to_symbol[i];
        if (s >= RANS_ALPHABET) return RANS_E_TABLE_SLOT;
        if (i < start[s] || i >= start[s] + freq[s]) return RANS_E_TABLE_SLOT;
    }
    uint8_t *sym8 = (uint8_t *)calloc((size_t)M + RANS_SLACK, 1);
    if (!sym8) return RANS_E_NOMEM;
    for (uint32_t i = 0; i < M; i++) sym8[i] = (uint8_t)slot_to_symbol[i];
    t->scale_bits = scale_bits;
    t->M = M;
    memcpy(t->freq, freq, sizeof(t->freq));
    memcpy(t->start, start, sizeof(t->start));
    t->slot_to_symbol = slot_to_symbol;
    t->sym8 = sym8;
    return RANS_OK;
}

static void rans_table_free(rans_table *t) {
    if (!t) return;
    free(t->sym8);
    memset(t, 0, sizeof(*t));
}

/* ---- chunk record: writer ------------------------------------------------- */

static uint64_t rans_round16(uint64_t v) { return (v + 15u) & ~(uint64_t)15u; }

/* Derived framing sizes. Header = n_symbols + packed_bytes + offsets table,
 * padded; the payload pad closes the record. Never stored in any field — the
 * padding rule is computed identically by writer and reader, which avoids a
 * redundant length field that could disagree with the derived value. */
static uint64_t rans_record_header_bytes(uint32_t n_streams) {
    return rans_round16(16u + ((uint64_t)n_streams + 1u) * 4u);
}

/* Worst-case record size for n nibbles. A symbol with table frequency f
 * costs at most log2(M/f) bits, so the per-stream worst case over any valid
 * table (f >= 1) is scale_bits bits/symbol — 1.875 bytes/symbol at the
 * maximum scale_bits of 15 — plus the 4 flushed state bytes. This bound
 * uses RANS_SCALE_BITS_MAX so it is table-independent and safe for every
 * table rans_table_init accepts (the old "n + n/2" rule was FALSE for
 * freq=1 symbols at scale_bits >= 13). Returns 0 for n_streams == 0. */
static uint64_t rans_record_bound(uint64_t n_symbols, uint32_t n_streams) {
    if (n_streams == 0) return 0;
    uint64_t per = n_symbols / n_streams + 1u;
    uint64_t per_bytes = (per * RANS_SCALE_BITS_MAX + 7u) / 8u + 4u + 64u;
    return rans_record_header_bytes(n_streams) + rans_round16(per_bytes * n_streams);
}

/* Encode n nibbles as one int4-rans256-g0 chunk record (n_streams round-robin
 * interleaved streams, shared table). Writes at most rans_record_bound()
 * bytes into out; *out_len receives the exact record length. Deterministic:
 * same input + same table => byte-identical output. Returns RANS_OK, or:
 * RANS_E_EMPTY (n == 0), RANS_E_NSTREAMS (n_streams == 0),
 * RANS_E_SYMBOL_UNCODABLE (an input value > 15, or a symbol whose table
 * frequency is zero — detected by a pre-scan BEFORE any encoding work),
 * RANS_E_SCRATCH (a codec bound was exceeded — a logic/bound failure, never
 * reported as a memory condition), RANS_E_NOMEM (malloc failed). */
static rans_err rans_record_encode(const uint8_t *nibbles, uint64_t n, const rans_table *t, uint32_t n_streams,
                                   uint8_t *out, uint64_t out_cap, uint64_t *out_len) {
    if (n_streams == 0) return RANS_E_NSTREAMS;
    if (n == 0) return RANS_E_EMPTY;
    /* pre-scan: refuse uncodable input by name, before touching `out` */
    for (uint64_t j = 0; j < n; j++)
        if (nibbles[j] > 15u || t->freq[nibbles[j]] == 0) return RANS_E_SYMBOL_UNCODABLE;
    uint64_t head = rans_record_header_bytes(n_streams);
    if (out_cap < head) return RANS_E_SCRATCH;
    uint64_t sub_max = n / n_streams + 1u;
    /* per-stream scratch: a freq>=1 symbol costs at most scale_bits bits, so
     * ceil(sub_max*scale_bits/8) + 4 flush bytes (+ slack) always fits */
    uint64_t scap = (sub_max * t->scale_bits + 7u) / 8u + 4u + 64u;
    uint8_t *sub = (uint8_t *)malloc(sub_max ? sub_max : 1);
    uint8_t *enc = (uint8_t *)malloc(scap);
    if (!sub || !enc) {
        free(sub);
        free(enc);
        return RANS_E_NOMEM;
    }

    memset(out, 0, head);
    uint64_t packed_bytes = n / 2u + (n & 1u);
    memcpy(out, &n, 8);
    memcpy(out + 8, &packed_bytes, 8);

    uint64_t pos = 0; /* payload cursor */
    uint8_t *payload = out + head;
    for (uint32_t i = 0; i < n_streams; i++) {
        uint64_t ns = 0; /* gather sub-stream i */
        for (uint64_t j = i; j < n; j += n_streams) sub[ns++] = nibbles[j];
        size_t off = rans_encode_stream(sub, ns, t->freq, t->start, t->scale_bits, enc, scap);
        if (off == (size_t)-1) {
            free(sub);
            free(enc);
            return RANS_E_SCRATCH;
        }
        uint64_t len = scap - off;
        if (head + pos + len > out_cap) {
            free(sub);
            free(enc);
            return RANS_E_SCRATCH;
        }
        memcpy(payload + pos, enc + off, len);
        uint32_t pos32 = (uint32_t)pos;
        memcpy(out + 16 + (uint64_t)i * 4u, &pos32, 4);
        pos += len;
        if (pos > 0xFFFFFFFFu) {
            free(sub);
            free(enc);
            return RANS_E_SCRATCH;
        }
    }
    uint32_t total32 = (uint32_t)pos;
    memcpy(out + 16 + (uint64_t)n_streams * 4u, &total32, 4);
    uint64_t total = head + rans_round16(pos);
    if (total > out_cap) {
        free(sub);
        free(enc);
        return RANS_E_SCRATCH;
    }
    memset(payload + pos, 0, rans_round16(pos) - pos);
    *out_len = total;
    free(sub);
    free(enc);
    return RANS_OK;
}

/* ---- chunk record: reader ------------------------------------------------- */

typedef struct {
    uint64_t n_symbols;
    uint64_t packed_bytes;
    const uint32_t *stream_offsets; /* n_streams+1 entries, payload-relative */
    const uint8_t *payload;
    uint64_t payload_len;
} rans_record;

/* Parse + validate one record blob (a tensor's full byte range). TRUST-
 * VERIFY-REFUSE: every malformation class returns its own named error and
 * the record is unusable on failure; there is no partial acceptance. The
 * blob is borrowed and must be 4-byte aligned (any malloc'd or numpy-backed
 * buffer qualifies; refused by name otherwise) so the stream_offsets
 * pointer handed to the decode kernels indexes with defined behavior.
 * NOTE for the batched decode arms: the blob must additionally satisfy the
 * RANS_SLACK contract (header comment) — parsing alone does not read past
 * blob_len, and it performs no allocation (in particular nothing
 * proportional to the untrusted n_symbols field). */
static rans_err rans_record_parse(const uint8_t *blob, uint64_t blob_len, uint32_t n_streams, rans_record *out) {
    memset(out, 0, sizeof(*out));
    if (n_streams == 0) return RANS_E_NSTREAMS;
    if (((uintptr_t)blob & 3u) != 0) return RANS_E_UNALIGNED;
    uint64_t head = rans_record_header_bytes(n_streams);
    if (blob_len < head) return RANS_E_TRUNCATED;
    uint64_t n_symbols, packed_bytes;
    memcpy(&n_symbols, blob, 8);
    memcpy(&packed_bytes, blob + 8, 8);
    if (n_symbols == 0) return RANS_E_EMPTY;
    /* non-wrapping form of packed_bytes == ceil(n_symbols/2): the additive
     * form (n_symbols+1)/2 wraps at UINT64_MAX and would accept
     * packed_bytes == 0 for it */
    if (packed_bytes != n_symbols / 2u + (n_symbols & 1u)) return RANS_E_COUNT_MISMATCH;
    uint32_t off_prev, off_cur;
    memcpy(&off_prev, blob + 16, 4);
    if (off_prev != 0) return RANS_E_OFFSET_FIRST;
    for (uint32_t i = 0; i < n_streams; i++) {
        memcpy(&off_cur, blob + 16 + ((uint64_t)i + 1u) * 4u, 4);
        if (off_cur < off_prev) return RANS_E_OFFSETS_MONOTONIC;
        /* every stream carries at least its 4-byte flushed state (true even
         * for streams that encode zero symbols) */
        if (off_cur - off_prev < 4u) return RANS_E_STREAM_SHORT;
        off_prev = off_cur;
    }
    uint64_t payload_len = off_prev; /* == offsets[n_streams] */
    if (head + payload_len > blob_len) return RANS_E_TRUNCATED;
    if (blob_len != head + rans_round16(payload_len)) return RANS_E_LENGTH_MISMATCH;
    /* amplification bound: a symbol costs at least log2(M/(M-1)) bits under
     * any valid table (present symbols of a genuine record cost more; the
     * degenerate freq==M single-symbol table costs ~0 per symbol but still
     * satisfies this bound easily). n_symbols beyond payload_len*8*M_max is
     * impossible for ANY table this format admits — refuse the
     * decompression bomb here, before any consumer sizes buffers from
     * n_symbols. (No overflow: payload_len < 2^32, M_max = 2^15.) */
    if (n_symbols > payload_len * 8u * (uint64_t)(1u << RANS_SCALE_BITS_MAX)) return RANS_E_OVERSIZE;
    const uint32_t *offs = (const uint32_t *)(blob + 16); /* aligned: checked */
    /* derived padding must be zero: bits hiding in the pad would make two
     * "identical" containers differ and defeat writer determinism */
    for (uint64_t i = 16u + ((uint64_t)n_streams + 1u) * 4u; i < head; i++)
        if (blob[i] != 0) return RANS_E_PAD_NONZERO;
    for (uint64_t i = head + payload_len; i < blob_len; i++)
        if (blob[i] != 0) return RANS_E_PAD_NONZERO;
    out->n_symbols = n_symbols;
    out->packed_bytes = packed_bytes;
    out->stream_offsets = offs;
    out->payload = blob + head;
    out->payload_len = payload_len;
    return RANS_OK;
}

/* ---- batched decode: path selection + envelope ---------------------------- */

typedef enum {
    RANS_PATH_SCALAR = 0,    /* portable, branchy renorm (mirrors the oracle) */
    RANS_PATH_SCALAR_BF = 1, /* portable, branch-free renorm (the SIMD algebra) */
    RANS_PATH_NEON = 2,
    RANS_PATH_AVX512 = 3,
    RANS_PATH_INVALID = -1
} rans_path;

static const char *rans_path_name(rans_path p) {
    switch (p) {
    case RANS_PATH_SCALAR:
        return "scalar";
    case RANS_PATH_SCALAR_BF:
        return "scalar_bf";
    case RANS_PATH_NEON:
        return "neon";
    case RANS_PATH_AVX512:
        return "avx512";
    default:
        return "invalid";
    }
}

#define RANS_G_SCALAR 8u
#define RANS_G_NEON 16u
#define RANS_G_AVX512 16u

static uint32_t rans_path_group_width(rans_path p) {
    if (p == RANS_PATH_AVX512) return RANS_G_AVX512;
    if (p == RANS_PATH_NEON) return RANS_G_NEON;
    return RANS_G_SCALAR;
}

/* forward declaration: the selftest below round-trips through the kernels */
static rans_err rans_record_decode_packed(const rans_record *rec, const rans_table *t, uint32_t n_streams, rans_path p,
                                          uint8_t *out_packed);

/* First-use round-trip selftest for one vector arm (same envelope discipline
 * as quant.h's AVX-512 accumulator selftest):
 * encode a fixed pseudo-random nibble block with the scalar encoder, decode
 * it with the arm under test AND the scalar arm, and demand byte identity
 * with the input and with each other. Runs once per process per arm; a
 * failing arm is reported on stderr and never used. */
static int rans__selftest_arm(rans_path p) {
    /* NSYM: even (so the group fast path — the thing under test — actually
     * runs) but NOT a multiple of NT, so group_tail runs too; large enough
     * that the rare-symbol states below reliably visit the multi-byte-renorm
     * bands (verified by re-running the sabotage exercise against the
     * selftest itself, not assumed). */
    enum { NT = 32, NSYM = 32 * 40 + 6 };
    uint8_t data[NSYM];
    uint32_t rng = 0x9E3779B9u;
    for (int i = 0; i < NSYM; i++) {
        rng = rng * 1664525u + 1013904223u;
        /* mostly the dominant symbol, with every rare symbol appearing: the
         * low-frequency outliers below drive the decoder through post-update
         * states far below RANS_L (multi-byte renorm refills) where a
         * renorm bug in a vector arm hides from comfortable tables. The f=3
         * group matters: power-of-two freqs keep floor(log2 x) on a rigid
         * lattice that never visits some refill bands (measured, not
         * theorized) — this table shape is load-bearing, found by a
         * deliberate-sabotage exercise. */
        data[i] = (uint8_t)(((rng >> 24) % 3 == 0) ? 1 + ((rng >> 8) % 14) : 15);
    }
    uint32_t sb = 10, M = 1u << sb;
    uint32_t freq[RANS_ALPHABET] = {0}, start[RANS_ALPHABET] = {0};
    for (int s = 1; s <= 7; s++) freq[s] = 1;  /* rare symbols */
    for (int s = 8; s <= 14; s++) freq[s] = 3; /* rare, off-lattice */
    freq[15] = M - 7 - 21;                     /* dominant symbol */
    uint64_t acc = 0;
    for (int s = 0; s < RANS_ALPHABET; s++) {
        start[s] = (uint32_t)acc;
        acc += freq[s];
    }
    uint16_t slot[1u << 10];
    for (int s = 0; s < RANS_ALPHABET; s++)
        for (uint32_t i = 0; i < freq[s]; i++) slot[start[s] + i] = (uint16_t)s;

    rans_table t;
    if (rans_table_init(&t, sb, freq, start, slot) != RANS_OK) return 0;
    uint64_t cap = rans_record_bound(NSYM, NT) + RANS_SLACK;
    uint8_t *rec_buf = (uint8_t *)calloc(cap, 1);
    uint8_t out_a[(NSYM + 1) / 2], out_b[(NSYM + 1) / 2], packed_ref[(NSYM + 1) / 2];
    int ok = 0;
    uint64_t rec_len = 0;
    if (rec_buf && rans_record_encode(data, NSYM, &t, NT, rec_buf, cap, &rec_len) == RANS_OK) {
        rans_record rec;
        if (rans_record_parse(rec_buf, rec_len, NT, &rec) == RANS_OK) {
            /* MACHINE-CHECKED band coverage: the load-bearing property of
             * this fixture is that it drives multi-byte renorm refills
             * (post-update x < 2^15 => refill count kb == 2) — the band a
             * comfortable table never visits and where a vector-arm renorm
             * bug hides. Count kb with the oracle-shaped loop and FAIL the
             * selftest if the fixture ever stops covering it, instead of
             * trusting a comment. (kb == 3 needs x < 2^7, impossible
             * mid-stream: post-update x >= L >> scale_bits >= 2^8.) */
            uint64_t kb2 = 0;
            for (uint32_t i = 0; i < NT; i++) {
                const uint8_t *ptr = rec.payload + rec.stream_offsets[i];
                const uint8_t *e2 = rec.payload + rec.stream_offsets[i + 1];
                uint32_t x = 0;
                for (int b2 = 0; b2 < 4; b2++) x = (x << 8) | *ptr++;
                for (uint64_t j = i; j < rec.n_symbols; j += NT) {
                    uint32_t sl2 = x & (M - 1u);
                    uint32_t sy2 = slot[sl2];
                    x = freq[sy2] * (x >> sb) + sl2 - start[sy2];
                    uint32_t kb = 0;
                    while (x < RANS_L) {
                        if (ptr >= e2) break;
                        x = (x << 8) | *ptr++;
                        kb++;
                    }
                    if (kb >= 2) kb2++;
                }
            }
            if (kb2 == 0) {
                fprintf(stderr, "[rans] selftest fixture lost multi-byte "
                                "renorm coverage — fixture bug, arm not trusted\n");
                free(rec_buf);
                rans_table_free(&t);
                return 0;
            }
            memset(packed_ref, 0, sizeof(packed_ref));
            for (int i = 0; i < NSYM; i++) packed_ref[i >> 1] |= (uint8_t)(data[i] << ((i & 1) * 4));
            memset(out_a, 0xAA, sizeof(out_a));
            memset(out_b, 0x55, sizeof(out_b));
            if (rans_record_decode_packed(&rec, &t, NT, RANS_PATH_SCALAR, out_a) == RANS_OK &&
                rans_record_decode_packed(&rec, &t, NT, p, out_b) == RANS_OK &&
                memcmp(out_a, packed_ref, sizeof(packed_ref)) == 0 &&
                memcmp(out_b, packed_ref, sizeof(packed_ref)) == 0)
                ok = 1;
        }
    }
    free(rec_buf);
    rans_table_free(&t);
    if (!ok) fprintf(stderr, "[rans] %s selftest FAILED: arm disabled, scalar fallback\n", rans_path_name(p));
    return ok;
}

static int rans_path_available(rans_path p) {
    switch (p) {
    case RANS_PATH_SCALAR:
    case RANS_PATH_SCALAR_BF:
        return 1;
#ifdef __ARM_NEON
    case RANS_PATH_NEON: {
        static int cached = -1;              /* -1 unknown, else 0/1 */
        const char *e = getenv("RANS_NEON"); /* kill-switch */
        if (e && atoi(e) == 0) return 0;
        if (cached < 0) cached = rans__selftest_arm(RANS_PATH_NEON);
        return cached;
    }
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    case RANS_PATH_AVX512: {
        static int cached = -1;
        const char *e = getenv("RANS_AVX512"); /* kill-switch */
        if (e && atoi(e) == 0) return 0;
#if defined(__GNUC__) || defined(__clang__)
        if (!(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))) return 0;
#endif
        if (cached < 0) cached = rans__selftest_arm(RANS_PATH_AVX512);
        return cached;
    }
#endif
    default:
        return 0;
    }
}

static rans_path rans_path_best(void) {
    if (rans_path_available(RANS_PATH_AVX512)) return RANS_PATH_AVX512;
    if (rans_path_available(RANS_PATH_NEON)) return RANS_PATH_NEON;
    /* scalar_bf over scalar: the closed-form renorm removes a data-dependent,
     * badly-predicted branch from the inner loop (measured ~1.9x on an
     * M-series host in the prototype round); both are byte-exact. */
    return RANS_PATH_SCALAR_BF;
}

/* Honours RANS_PATH=scalar|scalar_bf|neon|avx512. A forced arm that this
 * build/CPU/env cannot provide returns RANS_PATH_INVALID — NEVER a silent
 * downgrade. */
static rans_path rans_path_select(void) {
    const char *e = getenv("RANS_PATH");
    if (!e || !*e) return rans_path_best();
    rans_path p = RANS_PATH_INVALID;
    if (!strcmp(e, "scalar")) p = RANS_PATH_SCALAR;
    else if (!strcmp(e, "scalar_bf")) p = RANS_PATH_SCALAR_BF;
    else if (!strcmp(e, "neon")) p = RANS_PATH_NEON;
    else if (!strcmp(e, "avx512")) p = RANS_PATH_AVX512;
    else return RANS_PATH_INVALID;
    return rans_path_available(p) ? p : RANS_PATH_INVALID;
}

/* ---- batched decode: group kernels ---------------------------------------- */

/* Everything a group kernel needs about the record it is decoding. */
typedef struct {
    const rans_table *tab;
    const uint8_t *payload;
    const uint32_t *offs; /* n_streams+1 */
    uint64_t n;           /* n_symbols, even on the fast path */
    uint32_t N;           /* n_streams, even */
    uint8_t *out;         /* packed output, n/2 bytes */
} rans_kctx;

/* Big-endian 4-byte state load (the format's stream-init rule). */
static uint32_t rans_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

/* Number of COMPLETE rounds for the group of G streams based at stream b: a
 * round r is complete iff every lane's logical position r*N+b+l is a real
 * symbol, i.e. r*N + b + G - 1 < n. */
static uint64_t rans_full_rounds(uint64_t n, uint32_t N, uint32_t b, uint32_t G) {
    if (n < (uint64_t)b + G) return 0;
    return (n - (uint64_t)b - G) / N + 1;
}

/* Branchy (oracle-shaped) decode of the final, partially-populated round of
 * a group, given the per-lane states/cursors the main loop left behind.
 * k = min(G, n-p) is even on the fast path (n even, p = R*N+b even). */
static void rans_group_tail(const rans_kctx *K, uint32_t b, uint32_t G, uint64_t R, uint32_t *x, uint32_t *cur,
                            const uint32_t *end) {
    uint64_t p = R * (uint64_t)K->N + b;
    if (p >= K->n) return;
    uint64_t rem = K->n - p;
    uint32_t k = (rem < (uint64_t)G) ? (uint32_t)rem : G;
    const uint32_t sb = K->tab->scale_bits;
    const uint32_t mask = K->tab->M - 1u;
    const uint32_t *fq = K->tab->freq;
    const uint32_t *st = K->tab->start;
    const uint8_t *sym = K->tab->sym8;
    const uint8_t *pl = K->payload;
    uint8_t s[RANS_G_AVX512];
    for (uint32_t l = 0; l < k; l++) {
        uint32_t xx = x[l], c = cur[l], e = end[l];
        uint32_t slot = xx & mask;
        uint32_t sy = sym[slot];
        xx = fq[sy] * (xx >> sb) + slot - st[sy];
        while (xx < RANS_L) {
            if (c < e) xx = (xx << 8) | pl[c++];
            else break;
        }
        x[l] = xx;
        cur[l] = c;
        s[l] = (uint8_t)sy;
    }
    uint8_t *op = K->out + (p >> 1);
    for (uint32_t i = 0; i < k / 2u; i++) op[i] = (uint8_t)(s[2 * i] | (s[2 * i + 1] << 4));
}

static void rans_kernel_scalar(const rans_kctx *K, uint32_t g0, uint32_t g1) {
    const uint32_t G = RANS_G_SCALAR;
    const uint32_t sb = K->tab->scale_bits;
    const uint32_t mask = K->tab->M - 1u;
    const uint32_t *fq = K->tab->freq;
    const uint32_t *st = K->tab->start;
    const uint8_t *sym = K->tab->sym8;
    const uint8_t *pl = K->payload;
    const uint64_t n = K->n;
    const uint32_t N = K->N;
    const size_t ostride = N >> 1;
    for (uint32_t g = g0; g < g1; g++) {
        uint32_t b = g * G;
        uint32_t x[RANS_G_SCALAR], cur[RANS_G_SCALAR], end[RANS_G_SCALAR];
        for (uint32_t l = 0; l < G; l++) {
            cur[l] = K->offs[b + l];
            end[l] = K->offs[b + l + 1];
            x[l] = rans_be32(pl + cur[l]);
            cur[l] += 4;
        }
        uint64_t R = rans_full_rounds(n, N, b, G);
        uint8_t *op = K->out + (b >> 1);
        for (uint64_t r = 0; r < R; r++) {
            uint8_t s[RANS_G_SCALAR];
            for (uint32_t l = 0; l < G; l++) {
                uint32_t xx = x[l], c = cur[l];
                const uint32_t e = end[l];
                uint32_t slot = xx & mask;
                uint32_t sy = sym[slot];
                xx = fq[sy] * (xx >> sb) + slot - st[sy];
                /* byte-at-a-time renorm, verbatim from the stream decoder */
                while (xx < RANS_L) {
                    if (c < e) xx = (xx << 8) | pl[c++];
                    else break;
                }
                x[l] = xx;
                cur[l] = c;
                s[l] = (uint8_t)sy;
            }
            for (uint32_t i = 0; i < G / 2u; i++) op[i] = (uint8_t)(s[2 * i] | (s[2 * i + 1] << 4));
            op += ostride;
        }
        rans_group_tail(K, b, G, R, x, cur, end);
    }
}

/* Branch-free twin: same output, closed-form renorm. The renorm loop appends
 * whole bytes, so the count it consumes is a pure function of x —
 *   kb = (x < 2^23) + (x < 2^15) + (x < 2^7), capped by bytes remaining —
 * and x' = (x << 8*kb) | (next kb bytes, big-endian). This is the exact
 * algebra the NEON/AVX-512 arms use, kept scalar so it is testable on any
 * machine. Reads 4 bytes at the cursor => needs the RANS_SLACK contract. */
static void rans_kernel_scalar_bf(const rans_kctx *K, uint32_t g0, uint32_t g1) {
    const uint32_t G = RANS_G_SCALAR;
    const uint32_t sb = K->tab->scale_bits;
    const uint32_t mask = K->tab->M - 1u;
    const uint32_t *fq = K->tab->freq;
    const uint32_t *st = K->tab->start;
    const uint8_t *sym = K->tab->sym8;
    const uint8_t *pl = K->payload;
    const uint64_t n = K->n;
    const uint32_t N = K->N;
    const size_t ostride = N >> 1;
    for (uint32_t g = g0; g < g1; g++) {
        uint32_t b = g * G;
        uint32_t x[RANS_G_SCALAR], cur[RANS_G_SCALAR], end[RANS_G_SCALAR];
        for (uint32_t l = 0; l < G; l++) {
            cur[l] = K->offs[b + l];
            end[l] = K->offs[b + l + 1];
            x[l] = rans_be32(pl + cur[l]);
            cur[l] += 4;
        }
        uint64_t R = rans_full_rounds(n, N, b, G);
        uint8_t *op = K->out + (b >> 1);
        for (uint64_t r = 0; r < R; r++) {
            uint8_t s[RANS_G_SCALAR];
            for (uint32_t l = 0; l < G; l++) {
                uint32_t xx = x[l], c = cur[l];
                uint32_t slot = xx & mask;
                uint32_t sy = sym[slot];
                xx = fq[sy] * (xx >> sb) + slot - st[sy];
                uint32_t kb = (uint32_t)(xx < (1u << 23)) + (uint32_t)(xx < (1u << 15)) + (uint32_t)(xx < (1u << 7));
                uint32_t rem = end[l] - c;
                if (rem < kb) kb = rem;
                uint32_t be = rans_be32(pl + c); /* RANS_SLACK over-read */
                uint32_t val = (kb == 0) ? 0u : (be >> (8u * (4u - kb)));
                xx = (kb == 0) ? xx : ((xx << (8u * kb)) | val);
                c += kb;
                x[l] = xx;
                cur[l] = c;
                s[l] = (uint8_t)sy;
            }
            for (uint32_t i = 0; i < G / 2u; i++) op[i] = (uint8_t)(s[2 * i] | (s[2 * i + 1] << 4));
            op += ostride;
        }
        rans_group_tail(K, b, G, R, x, cur, end);
    }
}

#ifdef __ARM_NEON
/* NEON group kernel: 16 streams per group in four 4-lane state vectors.
 * NEON has no gather, so the two per-symbol memory operations (slot->symbol
 * lookup, renorm-byte fetch) stay scalar per lane — but both are issued
 * EARLY, before this round's arithmetic needs them, so all 32 independent
 * loads are in flight together; the state update, the closed-form renorm
 * count/merge (the scalar_bf algebra), the 16-entry freq/start lookups
 * (vqtbl1q byte-table lookups on lo/hi bytes of the u16-narrowed tables) and
 * the fused nibble repack (vuzp + shl + orr) are all vectorized. G=16 rather
 * than 8 amortizes the per-round scalar<->vector traffic; measured ~1.15x
 * over scalar_bf single-thread on an M-series host (a G=8 variant measured
 * 0.88x — the width is load-bearing). Byte-identical to the scalar arms by
 * construction, enforced by the first-use selftest + the identity sweep in
 * tests/test_rans.c. Needs the RANS_SLACK contract (4-byte fetches at
 * cursors that may sit at a stream end). */
static void rans_kernel_neon(const rans_kctx *K, uint32_t g0, uint32_t g1) {
    const uint32_t G = RANS_G_NEON; /* 16 */
    const uint32_t sb = K->tab->scale_bits;
    const uint8_t *sym = K->tab->sym8;
    const uint8_t *pl = K->payload;
    const uint64_t n = K->n;
    const uint32_t N = K->N;
    const size_t ostride = N >> 1;

    /* freq/start fit u16 (M <= 32768): narrow to two 16-byte tables each and
     * look 16 lanes up with one vqtbl1q per table byte. */
    uint8_t fq_lo[16], fq_hi[16], st_lo[16], st_hi[16];
    for (int s = 0; s < RANS_ALPHABET; s++) {
        fq_lo[s] = (uint8_t)(K->tab->freq[s] & 0xFF);
        fq_hi[s] = (uint8_t)(K->tab->freq[s] >> 8);
        st_lo[s] = (uint8_t)(K->tab->start[s] & 0xFF);
        st_hi[s] = (uint8_t)(K->tab->start[s] >> 8);
    }
    const uint8x16_t vfq_lo = vld1q_u8(fq_lo), vfq_hi = vld1q_u8(fq_hi);
    const uint8x16_t vst_lo = vld1q_u8(st_lo), vst_hi = vld1q_u8(st_hi);
    const uint32x4_t vmask = vdupq_n_u32(K->tab->M - 1u);
    const uint32x4_t t23 = vdupq_n_u32(1u << 23);
    const uint32x4_t t15 = vdupq_n_u32(1u << 15);
    const uint32x4_t t7 = vdupq_n_u32(1u << 7);
    const uint32x4_t one = vdupq_n_u32(1);
    const int32x4_t vsb_neg = vdupq_n_s32(-(int32_t)sb);
    const int32x4_t m32 = vdupq_n_s32(-32);

    for (uint32_t g = g0; g < g1; g++) {
        uint32_t b = g * G;
        uint32_t xa[RANS_G_NEON], ca[RANS_G_NEON], ea[RANS_G_NEON];
        for (uint32_t l = 0; l < G; l++) {
            ca[l] = K->offs[b + l];
            ea[l] = K->offs[b + l + 1];
            xa[l] = rans_be32(pl + ca[l]);
            ca[l] += 4;
        }
        uint32x4_t vx[4], vc[4], ve[4];
        for (int q = 0; q < 4; q++) {
            vx[q] = vld1q_u32(xa + 4 * q);
            vc[q] = vld1q_u32(ca + 4 * q);
            ve[q] = vld1q_u32(ea + 4 * q);
        }
        uint64_t R = rans_full_rounds(n, N, b, G);
        uint8_t *op = K->out + (b >> 1);

        for (uint64_t r = 0; r < R; r++) {
            uint8_t sarr[RANS_G_NEON];
            uint32_t slt[RANS_G_NEON], cl[RANS_G_NEON], bew[RANS_G_NEON];
            /* early per-lane loads: renorm words at the current cursors and
             * symbols at the current slots — independent of this round's
             * arithmetic, so all of them can be in flight at once */
            uint32x4_t sl[4];
            for (int q = 0; q < 4; q++) {
                sl[q] = vandq_u32(vx[q], vmask);
                vst1q_u32(slt + 4 * q, sl[q]);
                vst1q_u32(cl + 4 * q, vc[q]);
            }
            for (uint32_t l = 0; l < G; l++) {
                sarr[l] = sym[slt[l]];
                bew[l] = rans_be32(pl + cl[l]); /* RANS_SLACK over-read */
            }
            uint8x16_t sidx = vld1q_u8(sarr);
            uint8x16_t flo = vqtbl1q_u8(vfq_lo, sidx), fhi = vqtbl1q_u8(vfq_hi, sidx);
            uint8x16_t slo = vqtbl1q_u8(vst_lo, sidx), shi = vqtbl1q_u8(vst_hi, sidx);
            uint16x8_t f16a = vorrq_u16(vmovl_u8(vget_low_u8(flo)), vshlq_n_u16(vmovl_u8(vget_low_u8(fhi)), 8));
            uint16x8_t f16b = vorrq_u16(vmovl_u8(vget_high_u8(flo)), vshlq_n_u16(vmovl_u8(vget_high_u8(fhi)), 8));
            uint16x8_t s16a = vorrq_u16(vmovl_u8(vget_low_u8(slo)), vshlq_n_u16(vmovl_u8(vget_low_u8(shi)), 8));
            uint16x8_t s16b = vorrq_u16(vmovl_u8(vget_high_u8(slo)), vshlq_n_u16(vmovl_u8(vget_high_u8(shi)), 8));
            uint32x4_t f[4] = {vmovl_u16(vget_low_u16(f16a)), vmovl_u16(vget_high_u16(f16a)),
                               vmovl_u16(vget_low_u16(f16b)), vmovl_u16(vget_high_u16(f16b))};
            uint32x4_t st[4] = {vmovl_u16(vget_low_u16(s16a)), vmovl_u16(vget_high_u16(s16a)),
                                vmovl_u16(vget_low_u16(s16b)), vmovl_u16(vget_high_u16(s16b))};
            for (int q = 0; q < 4; q++) {
                /* x = f*(x>>sb) + slot - start */
                vx[q] = vaddq_u32(vmulq_u32(f[q], vshlq_u32(vx[q], vsb_neg)), vsubq_u32(sl[q], st[q]));
                /* closed-form renorm count, capped by bytes remaining */
                uint32x4_t kb = vandq_u32(vcltq_u32(vx[q], t23), one);
                kb = vaddq_u32(kb, vandq_u32(vcltq_u32(vx[q], t15), one));
                kb = vaddq_u32(kb, vandq_u32(vcltq_u32(vx[q], t7), one));
                kb = vminq_u32(kb, vsubq_u32(ve[q], vc[q]));
                /* merge: x' = (x << 8k) | (be >> 8*(4-k)); vshlq with a
                 * negative count is a logical right shift and |count| >= 32
                 * yields 0, so the kb==0 lane computes (x<<0)|0 = x — the
                 * scalar_bf ternary with no branch */
                uint32x4_t be = vld1q_u32(bew + 4 * q);
                int32x4_t shl = vreinterpretq_s32_u32(vshlq_n_u32(kb, 3));
                int32x4_t shr = vaddq_s32(shl, m32);
                vx[q] = vorrq_u32(vshlq_u32(vx[q], shl), vshlq_u32(be, shr));
                vc[q] = vaddq_u32(vc[q], kb);
            }
            /* fused repack: 16 nibbles -> 8 packed bytes */
            {
                uint8x8x2_t z = vuzp_u8(vget_low_u8(sidx), vget_high_u8(sidx));
                vst1_u8(op, vorr_u8(z.val[0], vshl_n_u8(z.val[1], 4)));
            }
            op += ostride;
        }
        for (int q = 0; q < 4; q++) {
            vst1q_u32(xa + 4 * q, vx[q]);
            vst1q_u32(ca + 4 * q, vc[q]);
        }
        rans_group_tail(K, b, G, R, xa, ca, ea);
    }
}
#endif /* __ARM_NEON */

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* AVX-512 group kernel: 16 streams per group, 16 lanes. Ported from the
 * proven prototype kernel (byte-exact on all shipped chunks in its round,
 * 8.7 GB/s batched on a Zen 5 host). Two things get cheap at this width:
 * the 16-entry freq/start lookups collapse to ONE vpermd each (a zmm holds
 * exactly 16 dwords), and each round's 16 nibbles emit as one 8-byte packed
 * store (vpmovqb). Uses only F+BW. Needs the RANS_SLACK contract (dword
 * gathers at cursors that may sit at a stream end and at sym8[M-1]). */
static void rans_kernel_avx512(const rans_kctx *K, uint32_t g0, uint32_t g1) {
    const uint32_t G = RANS_G_AVX512;
    const uint32_t sb = K->tab->scale_bits;
    const uint8_t *pl = K->payload;
    const uint8_t *sym = K->tab->sym8;
    const uint64_t n = K->n;
    const uint32_t N = K->N;
    const size_t ostride = N >> 1;

    const __m512i vmask = _mm512_set1_epi32((int)(K->tab->M - 1u));
    const __m512i vff = _mm512_set1_epi32(0xFF);
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i four = _mm512_set1_epi32(4);
    const __m512i eight = _mm512_set1_epi32(8);
    const __m512i t23 = _mm512_set1_epi32((int)(1u << 23));
    const __m512i t15 = _mm512_set1_epi32((int)(1u << 15));
    const __m512i t7 = _mm512_set1_epi32((int)(1u << 7));
    const __m512i bswap = _mm512_set4_epi32(0x0C0D0E0F, 0x08090A0B, 0x04050607, 0x00010203);
    const __m512i vfreq = _mm512_loadu_si512((const void *)K->tab->freq);
    const __m512i vstart = _mm512_loadu_si512((const void *)K->tab->start);

    for (uint32_t g = g0; g < g1; g++) {
        uint32_t b = g * G;
        uint32_t xa[RANS_G_AVX512], ca[RANS_G_AVX512], ea[RANS_G_AVX512];
        for (uint32_t l = 0; l < G; l++) {
            ca[l] = K->offs[b + l];
            ea[l] = K->offs[b + l + 1];
            xa[l] = rans_be32(pl + ca[l]);
            ca[l] += 4;
        }
        __m512i vx = _mm512_loadu_si512((const void *)xa);
        __m512i vcur = _mm512_loadu_si512((const void *)ca);
        __m512i vend = _mm512_loadu_si512((const void *)ea);

        uint64_t R = rans_full_rounds(n, N, b, G);
        uint8_t *op = K->out + (b >> 1);

        for (uint64_t r = 0; r < R; r++) {
            __m512i slot = _mm512_and_si512(vx, vmask);
            __m512i s = _mm512_and_si512(_mm512_i32gather_epi32(slot, (const void *)sym, 1), vff);
            __m512i f = _mm512_permutexvar_epi32(s, vfreq);
            __m512i st = _mm512_permutexvar_epi32(s, vstart);

            vx = _mm512_add_epi32(_mm512_mullo_epi32(f, _mm512_srli_epi32(vx, (int)sb)), _mm512_sub_epi32(slot, st));

            __m512i kb = _mm512_setzero_si512();
            kb = _mm512_mask_add_epi32(kb, _mm512_cmplt_epu32_mask(vx, t23), kb, one);
            kb = _mm512_mask_add_epi32(kb, _mm512_cmplt_epu32_mask(vx, t15), kb, one);
            kb = _mm512_mask_add_epi32(kb, _mm512_cmplt_epu32_mask(vx, t7), kb, one);
            kb = _mm512_min_epu32(kb, _mm512_sub_epi32(vend, vcur));

            __m512i be = _mm512_shuffle_epi8(_mm512_i32gather_epi32(vcur, (const void *)pl, 1), bswap);
            __m512i shl = _mm512_mullo_epi32(kb, eight);
            __m512i shr = _mm512_mullo_epi32(_mm512_sub_epi32(four, kb), eight);
            vx = _mm512_or_si512(_mm512_sllv_epi32(vx, shl), _mm512_srlv_epi32(be, shr));
            vcur = _mm512_add_epi32(vcur, kb);

            /* fused repack: byte 0 of each 64-bit element is the packed byte,
             * vpmovqb collects all eight into one 8-byte store */
            __m512i p = _mm512_or_si512(s, _mm512_srli_epi64(s, 28));
            __m128i packed8 = _mm512_cvtepi64_epi8(p);
            long long v = _mm_cvtsi128_si64(packed8);
            memcpy(op, &v, 8);
            op += ostride;
        }
        _mm512_storeu_si512((void *)xa, vx);
        _mm512_storeu_si512((void *)ca, vcur);
        _mm512_storeu_si512((void *)ea, vend);
        rans_group_tail(K, b, G, R, xa, ca, ea);
    }
}
#endif /* AVX512F+BW */

/* ---- batched decode: dispatch --------------------------------------------- */

/* Portable per-stream reference: whole record to one NIBBLE per byte. Works
 * for any n_symbols/n_streams combination, including odd n_symbols. */
static void rans_record_decode_nibbles(const rans_record *rec, const rans_table *t, uint32_t n_streams,
                                       uint8_t *out_nibbles) {
    const uint32_t sb = t->scale_bits;
    const uint32_t mask = t->M - 1u;
    const uint32_t *fq = t->freq;
    const uint32_t *st = t->start;
    const uint16_t *sym = t->slot_to_symbol;
    for (uint32_t i = 0; i < n_streams; i++) {
        const uint8_t *ptr = rec->payload + rec->stream_offsets[i];
        const uint8_t *e = rec->payload + rec->stream_offsets[i + 1];
        uint32_t x = 0;
        for (int b = 0; b < 4; b++) x = (x << 8) | *ptr++;
        for (uint64_t j = i; j < rec->n_symbols; j += n_streams) {
            uint32_t slot = x & mask;
            uint32_t s = sym[slot];
            x = fq[s] * (x >> sb) + slot - st[s];
            while (x < RANS_L) {
                if (ptr < e) x = (x << 8) | *ptr++;
                else break;
            }
            out_nibbles[j] = (uint8_t)s;
        }
    }
}

/* Fast path applies when whole groups tile the streams and the fused repack
 * can own whole output bytes; otherwise 0 => use the generic path. */
static uint32_t rans_record_group_count(const rans_record *rec, uint32_t n_streams, rans_path p) {
    uint32_t G = rans_path_group_width(p);
    if (n_streams % G) return 0;
    if (rec->n_symbols & 1u) return 0;
    return n_streams / G;
}

/* Decode stream groups [g0,g1) straight into PACKED bytes. Each call writes
 * only the bytes its own streams own, so disjoint group ranges may run
 * concurrently on the same output buffer with no synchronization — this is
 * the parallel entry point a threaded consumer (E2) partitions work with. */
static void rans_record_decode_groups(const rans_record *rec, const rans_table *t, uint32_t n_streams, rans_path p,
                                      uint32_t g0, uint32_t g1, uint8_t *out_packed) {
    rans_kctx K;
    K.tab = t;
    K.payload = rec->payload;
    K.offs = rec->stream_offsets;
    K.n = rec->n_symbols;
    K.N = n_streams;
    K.out = out_packed;
    switch (p) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    case RANS_PATH_AVX512:
        rans_kernel_avx512(&K, g0, g1);
        return;
#endif
#ifdef __ARM_NEON
    case RANS_PATH_NEON:
        rans_kernel_neon(&K, g0, g1);
        return;
#endif
    case RANS_PATH_SCALAR_BF:
        rans_kernel_scalar_bf(&K, g0, g1);
        return;
    default:
        rans_kernel_scalar(&K, g0, g1);
        return;
    }
}

/* Whole record to packed bytes (out_packed must hold rec->packed_bytes).
 * Dispatches to the requested arm's fast path, or the generic per-stream
 * path when the record shape cannot use groups. RANS_PATH_INVALID refuses. */
static rans_err rans_record_decode_packed(const rans_record *rec, const rans_table *t, uint32_t n_streams, rans_path p,
                                          uint8_t *out_packed) {
    /* n_streams == 0 would decode zero streams and return uninitialized
     * scratch as "successful" output — refuse before any path can run */
    if (n_streams == 0) return RANS_E_NSTREAMS;
    if (p == RANS_PATH_INVALID) return RANS_E_PATH_UNAVAILABLE;
        /* an arm this build did not compile refuses rather than silently running
         * scalar: a quiet downgrade would let "we tested <arm>" become a false
         * claim (same rule rans_path_select applies to the env override) */
#ifndef __ARM_NEON
    if (p == RANS_PATH_NEON) return RANS_E_PATH_UNAVAILABLE;
#endif
#if !(defined(__AVX512F__) && defined(__AVX512BW__))
    if (p == RANS_PATH_AVX512) return RANS_E_PATH_UNAVAILABLE;
#endif
    uint32_t ng = rans_record_group_count(rec, n_streams, p);
    if (ng == 0) {
        uint8_t *scratch = (uint8_t *)malloc((size_t)rec->n_symbols);
        if (!scratch) return RANS_E_NOMEM;
        rans_record_decode_nibbles(rec, t, n_streams, scratch);
        uint64_t np = rec->packed_bytes;
        for (uint64_t k = 0; k < np; k++) {
            uint8_t lo = scratch[2 * k];
            uint8_t hi = (2 * k + 1 < rec->n_symbols) ? scratch[2 * k + 1] : 0;
            out_packed[k] = (uint8_t)(lo | (hi << 4));
        }
        free(scratch);
        return RANS_OK;
    }
    rans_record_decode_groups(rec, t, n_streams, p, 0, ng, out_packed);
    return RANS_OK;
}

#endif /* COLI_RANS_H */
