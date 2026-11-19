/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#ifndef KAIRA_WEX4_H
#define KAIRA_WEX4_H

/* WEX4: self-describing lossless tensor container (docs/NEXT-FORMAT-SPEC.md).
 * Prototype reader, validator and codecs; no engine dependency. Every integer
 * on disk is little-endian and is read byte by byte (no struct casts).
 * Validation works in place on the header page and the index: no allocation,
 * no parsing into heap structures. A file is used only after wex4_parse_header
 * and wex4_open both returned 0. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "wex_format.h"   /* WexEntry, WNR3 codec, rANS primitives, MurmurHash3 */
/* The Wex-WRN2 copy (c/wex4.h) is this file unchanged; its research tools reach the fast NR3
 * decoder through a shim (wex4/wex4_tools.c). No KAIRA engine calls wex4_decode_record, so
 * this keeps the reference decoder. */
#define WEX4_NR3_DECODE wex_nr2_decode

#define WEX4_MAGIC UINT32_C(0x57455834)
#define WEX4_PAGE 4096u
#define WEX4_HEADER_HASHED 4064u               /* header_sha256 = SHA-256(page[0..4064)) at 4064 */
#define WEX4_REQ_VALUE_EXACT UINT64_C(1)       /* some records are value-exact, not byte-exact */
#define WEX4_REQUIRED_KNOWN WEX4_REQ_VALUE_EXACT
#define WEX4_OPT_ST_SOURCES UINT64_C(1)        /* meta holds the safetensors sources (tools only) */
#define WEX4_MAX_BLOBS (1u << 16)
#define WEX4_MAX_RECORDS (1u << 24)
#define WEX4_MAX_TENSORS (1u << 24)
#define WEX4_MAX_INDEX (UINT64_C(1) << 28)
#define WEX4_MAX_DECODED (1u << 30)            /* per record; larger tensors are split into parts */
#define WEX4_BLOB_BYTES 64u
#define WEX4_RECORD_BYTES 72u
#define WEX4_TENSOR_BYTES 96u
#define WEX4_CODEC_RAW 0u
#define WEX4_CODEC_WNR3 5u
#define WEX4_CODEC_WBP1 6u
#define WEX4_REC_VALUE_EXACT 1u
#define WEX4_REC_EXPERT 2u
#define WEX4_BLOB_WNM2 1u
#define WEX4_BP_MAGIC UINT32_C(0x31504257)     /* "WBP1" */
#define WEX4_BP_HEADER 16u

enum { WEX4_OK = 0, WEX4_E_ARG = -1, WEX4_E_MAGIC = -2, WEX4_E_HASH = -3, WEX4_E_VERSION = -4, WEX4_E_FEATURE = -5,
       WEX4_E_HEADER = -6, WEX4_E_INDEX = -7, WEX4_E_BLOB = -8, WEX4_E_RECORD = -9, WEX4_E_TENSOR = -10,
       WEX4_E_NAME = -11, WEX4_E_LAYOUT = -12, WEX4_E_EXPERT = -13, WEX4_E_DATA = -14, WEX4_E_UNSUPPORTED = -15 };

/* ---- SHA-256 (FIPS 180-4) ------------------------------------------------ */
typedef struct { uint32_t h[8]; uint64_t n; unsigned char buf[64]; } Wex4Sha;
static const uint32_t wex4_sha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#define WEX4_ROR(x, r) ((x) >> (r) | (x) << (32 - (r)))
static inline void wex4_sha_block(uint32_t h[8], const unsigned char *p) {
    uint32_t w[64], s[8], i;
    for (i = 0; i < 16u; i++)
        w[i] = (uint32_t)p[i * 4u] << 24 | (uint32_t)p[i * 4u + 1u] << 16 | (uint32_t)p[i * 4u + 2u] << 8 | p[i * 4u + 3u];
    for (; i < 64u; i++)
        w[i] = w[i - 16u] + (WEX4_ROR(w[i - 15u], 7) ^ WEX4_ROR(w[i - 15u], 18) ^ w[i - 15u] >> 3) + w[i - 7u] +
               (WEX4_ROR(w[i - 2u], 17) ^ WEX4_ROR(w[i - 2u], 19) ^ w[i - 2u] >> 10);
    memcpy(s, h, sizeof(s));
    for (i = 0; i < 64u; i++) {
        uint32_t t1 = s[7] + (WEX4_ROR(s[4], 6) ^ WEX4_ROR(s[4], 11) ^ WEX4_ROR(s[4], 25)) +
                      ((s[4] & s[5]) ^ (~s[4] & s[6])) + wex4_sha_k[i] + w[i];
        uint32_t t2 = (WEX4_ROR(s[0], 2) ^ WEX4_ROR(s[0], 13) ^ WEX4_ROR(s[0], 22)) +
                      ((s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]));
        s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1; s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
    }
    for (i = 0; i < 8u; i++) h[i] += s[i];
}
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>
#include <cpuid.h>
/* The same compression function with the SHA extensions (about 6x the portable one): what makes a
 * SHA-256 of every record at its first read affordable. Chosen at run time. */
__attribute__((target("sha,sse4.1,ssse3")))
static void wex4_sha_block_ni(uint32_t h[8], const unsigned char *p) {
    const __m128i swap = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i m[4], t = _mm_loadu_si128((const __m128i *)h), s1 = _mm_loadu_si128((const __m128i *)(h + 4)), s0, a, b;
    unsigned int i;
    t = _mm_shuffle_epi32(t, 0xb1);
    s1 = _mm_shuffle_epi32(s1, 0x1b);
    s0 = _mm_alignr_epi8(t, s1, 8);
    s1 = _mm_blend_epi16(s1, t, 0xf0);
    a = s0; b = s1;
    for (i = 0; i < 16u; i++) {
        __m128i w;
        if (i < 4u) m[i] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + i * 16u)), swap);
        else {
            w = _mm_sha256msg1_epu32(m[i & 3u], m[(i + 1u) & 3u]);
            w = _mm_add_epi32(w, _mm_alignr_epi8(m[(i + 3u) & 3u], m[(i + 2u) & 3u], 4));
            m[i & 3u] = _mm_sha256msg2_epu32(w, m[(i + 3u) & 3u]);
        }
        w = _mm_add_epi32(m[i & 3u], _mm_loadu_si128((const __m128i *)(wex4_sha_k + i * 4u)));
        s1 = _mm_sha256rnds2_epu32(s1, s0, w);
        s0 = _mm_sha256rnds2_epu32(s0, s1, _mm_shuffle_epi32(w, 0x0e));
    }
    s0 = _mm_add_epi32(s0, a); s1 = _mm_add_epi32(s1, b);
    t = _mm_shuffle_epi32(s0, 0x1b);
    s1 = _mm_shuffle_epi32(s1, 0xb1);
    _mm_storeu_si128((__m128i *)h, _mm_blend_epi16(t, s1, 0xf0));
    _mm_storeu_si128((__m128i *)(h + 4), _mm_alignr_epi8(s1, t, 8));
}
static inline int wex4_sha_ni(void) {
    static int known = -1;
    if (known < 0) {
        unsigned int a, b = 0, c, d;
        known = __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b >> 29 & 1u);
    }
    return known;
}
#define WEX4_SHA_BLOCK(h, p) (wex4_sha_ni() ? wex4_sha_block_ni(h, p) : wex4_sha_block(h, p))
#else
#define WEX4_SHA_BLOCK(h, p) wex4_sha_block(h, p)
#endif
static inline void wex4_sha_init(Wex4Sha *s) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, iv, sizeof(iv)); s->n = 0;
}
static inline void wex4_sha_update(Wex4Sha *s, const unsigned char *p, size_t n) {
    size_t fill = (size_t)(s->n & 63u);
    s->n += n;
    if (fill) {
        size_t take = 64u - fill < n ? 64u - fill : n;
        memcpy(s->buf + fill, p, take); p += take; n -= take;
        if (fill + take < 64u) return;
        WEX4_SHA_BLOCK(s->h, s->buf);
    }
    for (; n >= 64u; p += 64u, n -= 64u) WEX4_SHA_BLOCK(s->h, p);
    if (n) memcpy(s->buf, p, n);
}
static inline void wex4_sha_final(Wex4Sha *s, unsigned char out[32]) {
    uint64_t bits = s->n * 8u;
    unsigned char pad[72] = {0x80}, len[8];
    unsigned int i;
    for (i = 0; i < 8u; i++) len[i] = (unsigned char)(bits >> (56u - i * 8u));
    wex4_sha_update(s, pad, (size_t)((119u - (s->n & 63u)) & 63u) + 1u);
    wex4_sha_update(s, len, 8u);
    for (i = 0; i < 32u; i++) out[i] = (unsigned char)(s->h[i / 4u] >> (24u - (i % 4u) * 8u));
}
static inline void wex4_sha256(const unsigned char *p, size_t n, unsigned char out[32]) {
    Wex4Sha s;
    wex4_sha_init(&s); wex4_sha_update(&s, p, n); wex4_sha_final(&s, out);
}

/* ---- Wire access ---------------------------------------------------------- */
static inline uint64_t wex4_u64(const unsigned char *p) { return wex_nr_u32(p) | (uint64_t)wex_nr_u32(p + 4u) << 32; }
static inline uint64_t wex4_align(uint64_t n) { return (n + (WEX4_PAGE - 1u)) & ~(uint64_t)(WEX4_PAGE - 1u); }

typedef struct {
    uint32_t major, minor, blobs, records, tensors, layers, experts, max_decoded;
    uint64_t required, optional, file_bytes, index_offset, index_bytes, names_bytes, meta_bytes, total_decoded;
    unsigned char index_sha[32], root[32];   /* root = header_sha256: the container's identity */
} Wex4Header;
typedef struct { uint32_t kind, id; uint64_t offset, bytes; const unsigned char *sha; } Wex4Blob;
typedef struct {
    uint64_t offset;
    uint32_t stored, decoded, codec, flags, blob, owner, first_tensor, tensor_count;
    const unsigned char *sha;
} Wex4Record;
typedef struct {
    const unsigned char *name;
    uint32_t name_len, dtype, rank, record, record_count;
    uint64_t shape[8], offset, nbytes;
} Wex4Tensor;
typedef struct {
    Wex4Header h;
    const unsigned char *blobs, *records, *tensors, *order, *names, *meta;
} Wex4View;

/* Bits per element of dtype codes 1..22 (the safetensors dtypes); 0 = unknown, opaque bytes. */
static const unsigned char wex4_dtype_bits[23] = {0, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 64, 64, 64, 64, 4, 6, 6, 8, 8};
#define WEX4_DT_U8 2u
#define WEX4_DT_I8 3u
#define WEX4_DT_F8_E8M0 6u

/* Validate the 4096-byte header page of a file of file_bytes bytes. */
static inline int wex4_parse_header(const unsigned char *page, uint64_t file_bytes, Wex4Header *h) {
    unsigned char digest[32];
    uint64_t need;
    if (!page || !h) return WEX4_E_ARG;
    if (wex_nr_u32(page) != WEX4_MAGIC) return WEX4_E_MAGIC;
    wex4_sha256(page, WEX4_HEADER_HASHED, digest);
    if (memcmp(digest, page + WEX4_HEADER_HASHED, 32u)) return WEX4_E_HASH;
    h->major = wex_nr_u16(page + 4u); h->minor = wex_nr_u16(page + 6u);
    if (h->major != 4u) return WEX4_E_VERSION;
    h->required = wex4_u64(page + 8u); h->optional = wex4_u64(page + 16u);
    if (h->required & ~WEX4_REQUIRED_KNOWN) return WEX4_E_FEATURE;
    h->file_bytes = wex4_u64(page + 24u); h->index_offset = wex4_u64(page + 32u); h->index_bytes = wex4_u64(page + 40u);
    h->blobs = wex_nr_u32(page + 48u); h->records = wex_nr_u32(page + 52u); h->tensors = wex_nr_u32(page + 56u);
    h->layers = wex_nr_u32(page + 60u); h->experts = wex_nr_u32(page + 64u); h->max_decoded = wex_nr_u32(page + 68u);
    h->names_bytes = wex4_u64(page + 72u); h->meta_bytes = wex4_u64(page + 80u); h->total_decoded = wex4_u64(page + 88u);
    memcpy(h->index_sha, page + 96u, 32u); memcpy(h->root, digest, 32u);
    if (h->file_bytes != file_bytes || file_bytes < WEX4_PAGE || (file_bytes & (WEX4_PAGE - 1u)) ||
        h->blobs > WEX4_MAX_BLOBS || h->records > WEX4_MAX_RECORDS || h->tensors > WEX4_MAX_TENSORS ||
        h->names_bytes > WEX4_MAX_INDEX || h->meta_bytes > WEX4_MAX_INDEX ||
        h->layers > 65536u || h->experts > 65536u || !h->layers != !h->experts ||
        (uint64_t)h->layers * h->experts > WEX4_MAX_RECORDS ||   /* bounds the legacy entry table */
        h->max_decoded > WEX4_MAX_DECODED || (!h->records && (h->max_decoded || h->total_decoded)))
        return WEX4_E_HEADER;
    need = (uint64_t)h->blobs * WEX4_BLOB_BYTES + (uint64_t)h->records * WEX4_RECORD_BYTES +
           (uint64_t)h->tensors * (WEX4_TENSOR_BYTES + 4u) + h->names_bytes + h->meta_bytes;
    if (need != h->index_bytes || need > WEX4_MAX_INDEX || h->index_offset < WEX4_PAGE ||
        (h->index_offset & (WEX4_PAGE - 1u)) || h->index_offset > file_bytes ||
        wex4_align(need) > file_bytes - h->index_offset || (!need && h->index_offset != WEX4_PAGE))
        return WEX4_E_HEADER;
    return WEX4_OK;
}

static inline void wex4_get_blob(const Wex4View *v, uint32_t i, Wex4Blob *b) {
    const unsigned char *p = v->blobs + (size_t)i * WEX4_BLOB_BYTES;
    b->kind = wex_nr_u32(p); b->id = wex_nr_u32(p + 4u); b->offset = wex4_u64(p + 8u); b->bytes = wex4_u64(p + 16u);
    b->sha = p + 24u;
}
static inline void wex4_get_record(const Wex4View *v, uint32_t i, Wex4Record *r) {
    const unsigned char *p = v->records + (size_t)i * WEX4_RECORD_BYTES;
    r->offset = wex4_u64(p); r->stored = wex_nr_u32(p + 8u); r->decoded = wex_nr_u32(p + 12u);
    r->codec = wex_nr_u16(p + 16u); r->flags = wex_nr_u16(p + 18u); r->blob = wex_nr_u32(p + 20u);
    r->owner = wex_nr_u32(p + 24u); r->first_tensor = wex_nr_u32(p + 28u); r->tensor_count = wex_nr_u32(p + 32u);
    r->sha = p + 36u;
}
static inline void wex4_get_tensor(const Wex4View *v, uint32_t i, Wex4Tensor *t) {
    const unsigned char *p = v->tensors + (size_t)i * WEX4_TENSOR_BYTES;
    unsigned int d;
    t->name = v->names + wex_nr_u32(p); t->name_len = wex_nr_u16(p + 4u); t->dtype = p[6]; t->rank = p[7];
    for (d = 0; d < 8u; d++) t->shape[d] = wex4_u64(p + 8u + d * 8u);
    t->record = wex_nr_u32(p + 72u); t->record_count = wex_nr_u32(p + 76u);
    t->offset = wex4_u64(p + 80u); t->nbytes = wex4_u64(p + 88u);
}

/* Strict UTF-8 (shortest form, no surrogates, <= U+10FFFF) without NUL. */
static inline int wex4_utf8_ok(const unsigned char *p, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned int c = p[i], extra, k;
        uint32_t cp;
        if (!c) return 0;
        if (c < 0x80u) { i++; continue; }
        if (c >= 0xc2u && c <= 0xdfu) { extra = 1; cp = c & 0x1fu; }
        else if (c >= 0xe0u && c <= 0xefu) { extra = 2; cp = c & 0x0fu; }
        else if (c >= 0xf0u && c <= 0xf4u) { extra = 3; cp = c & 0x07u; }
        else return 0;
        if (n - i <= extra) return 0;
        for (k = 1; k <= extra; k++) {
            if ((p[i + k] & 0xc0u) != 0x80u) return 0;
            cp = cp << 6 | (p[i + k] & 0x3fu);
        }
        if ((extra == 2 && (cp < 0x800u || (cp >= 0xd800u && cp <= 0xdfffu))) ||
            (extra == 3 && (cp < 0x10000u || cp > 0x10ffffu))) return 0;
        i += extra + 1u;
    }
    return 1;
}

/* Legacy 72-byte entry of an EXPERT record (3 scale + 3 weight tensors): what
 * the engine store and the CUDA helper already consume. Validates the slab layout. */
static inline int wex4_expert_entry(const Wex4View *v, uint32_t r, WexEntry *e) {
    Wex4Record rec;
    Wex4Tensor t;
    uint64_t raw, aligned;
    uint32_t m;
    if (!v || !e || r >= v->h.records) return WEX4_E_ARG;
    wex4_get_record(v, r, &rec);
    if (!(rec.flags & WEX4_REC_EXPERT) || rec.tensor_count != 6u) return WEX4_E_EXPERT;
    memset(e, 0, sizeof(*e));
    e->file_offset = rec.offset; e->total_bytes = rec.stored; e->flags = rec.codec;
    for (m = 0; m < 6u; m++) {
        wex4_get_tensor(v, rec.first_tensor + m, &t);
        if (t.rank != 2u || t.record_count != 1u || !t.shape[0] || !t.shape[1] || t.shape[0] > 65535u ||
            t.shape[1] > 65535u) return WEX4_E_EXPERT;
        if (m < 3u) {
            if (t.dtype != WEX4_DT_U8 && t.dtype != WEX4_DT_F8_E8M0) return WEX4_E_EXPERT;
            e->scale_shapes[m][0] = (uint16_t)t.shape[0]; e->scale_shapes[m][1] = (uint16_t)t.shape[1];
            e->scale_offsets[m] = (uint32_t)t.offset; e->scale_bytes += (uint32_t)t.nbytes;
        } else {
            if (t.dtype != WEX4_DT_U8 && t.dtype != WEX4_DT_I8) return WEX4_E_EXPERT;
            e->weight_shapes[m - 3u][0] = (uint16_t)t.shape[0]; e->weight_shapes[m - 3u][1] = (uint16_t)t.shape[1];
            e->weight_offsets[m - 3u] = (uint32_t)t.offset; e->weight_bytes += (uint32_t)t.nbytes;
        }
    }
    if (wex_entry_raw_sizes(e, &raw, &aligned) || raw != rec.decoded) return WEX4_E_EXPERT;
    return WEX4_OK;
}

/* Validate the whole index (index_len = index_bytes rounded up to 4096, as read
 * from disk). On success every offset, size and name in the view is in bounds. */
static inline int wex4_open(const Wex4Header *h, const unsigned char *index, size_t index_len, Wex4View *v) {
    unsigned char digest[32];
    uint64_t pos, total = 0, pending_bytes = 0, name_at = 0, i64;
    uint32_t r, t = 0, pending = 0, maximum = 0, b = 0, i;
    int64_t last_owner = -1;
    int index_done;
    Wex4Record rec;
    Wex4Tensor ten;
    Wex4Blob blob;
    if (!h || !v || (!index && h->index_bytes) || index_len != wex4_align(h->index_bytes)) return WEX4_E_ARG;
    wex4_sha256(index, (size_t)h->index_bytes, digest);
    if (memcmp(digest, h->index_sha, 32u)) return WEX4_E_HASH;
    for (i64 = h->index_bytes; i64 < index_len; i64++) if (index[i64]) return WEX4_E_INDEX;
    v->h = *h;
    v->blobs = index;
    v->records = v->blobs + (size_t)h->blobs * WEX4_BLOB_BYTES;
    v->tensors = v->records + (size_t)h->records * WEX4_RECORD_BYTES;
    v->order = v->tensors + (size_t)h->tensors * WEX4_TENSOR_BYTES;
    v->names = v->order + (size_t)h->tensors * 4u;
    v->meta = v->names + (size_t)h->names_bytes;

    /* tensors: canonical name pool, dtype/shape/size */
    for (i = 0; i < h->tensors; i++) {
        const unsigned char *p = v->tensors + (size_t)i * WEX4_TENSOR_BYTES;
        uint64_t elements = 1;
        unsigned int d, bits;
        if (wex_nr_u32(p) != name_at || wex_nr_u16(p + 4u) > h->names_bytes - name_at) return WEX4_E_NAME;
        wex4_get_tensor(v, i, &ten);
        name_at += ten.name_len;
        if (!wex4_utf8_ok(ten.name, ten.name_len)) return WEX4_E_NAME;
        if (ten.rank > 8u) return WEX4_E_TENSOR;
        for (d = 0; d < 8u; d++) {
            if (d >= ten.rank) { if (ten.shape[d]) return WEX4_E_TENSOR; continue; }
            if (ten.shape[d] && elements > UINT64_MAX / ten.shape[d]) return WEX4_E_TENSOR;
            elements *= ten.shape[d];
        }
        bits = ten.dtype < 23u ? wex4_dtype_bits[ten.dtype] : 0u;
        if (bits && (elements > UINT64_MAX / 64u || ten.nbytes > UINT64_MAX / 8u || elements * bits != ten.nbytes * 8u))
            return WEX4_E_TENSOR;
    }
    if (name_at != h->names_bytes) return WEX4_E_NAME;
    /* name order: strictly increasing names => a permutation and unique names */
    for (i = 0; i < h->tensors; i++) {
        uint32_t at = wex_nr_u32(v->order + (size_t)i * 4u);
        Wex4Tensor prev;
        int c;
        if (at >= h->tensors) return WEX4_E_NAME;
        wex4_get_tensor(v, at, &ten);
        if (i) {
            wex4_get_tensor(v, wex_nr_u32(v->order + (size_t)(i - 1u) * 4u), &prev);
            c = memcmp(prev.name, ten.name, prev.name_len < ten.name_len ? prev.name_len : ten.name_len);
            if (c > 0 || (!c && prev.name_len >= ten.name_len)) return WEX4_E_NAME;
        }
    }

    /* blobs */
    for (i = 0; i < h->blobs; i++) {
        wex4_get_blob(v, i, &blob);
        if (!blob.bytes || blob.bytes > WEX4_MAX_DECODED || wex4_u64(blob.sha + 32u)) return WEX4_E_BLOB;
        if (blob.kind == WEX4_BLOB_WNM2 && (blob.bytes != WEX_NR2_MODEL_BYTES || !blob.id)) return WEX4_E_BLOB;
    }

    /* records and the tensors that tile them */
    for (r = 0; r < h->records; r++) {
        uint64_t cursor = 0;
        uint32_t k;
        wex4_get_record(v, r, &rec);
        if (!rec.stored || (rec.stored & (WEX4_PAGE - 1u)) || !rec.decoded || rec.decoded > WEX4_MAX_DECODED ||
            rec.stored > wex4_align(rec.decoded) || (rec.flags & ~(WEX4_REC_VALUE_EXACT | WEX4_REC_EXPERT)) ||
            wex_nr_u32(rec.sha + 32u) || rec.blob > h->blobs) return WEX4_E_RECORD;
        if (rec.codec == WEX4_CODEC_RAW ? rec.stored != wex4_align(rec.decoded) || rec.blob
                                        : rec.stored == wex4_align(rec.decoded)) return WEX4_E_RECORD;
        /* WBP1: every stream of at most 32768 bytes costs at least 6 bytes, which bounds the expansion (~5461x) */
        if (rec.codec == WEX4_CODEC_WBP1 &&
            (rec.blob || rec.flags || rec.stored < wex4_align(WEX4_BP_HEADER + 6u * (((uint64_t)rec.decoded + 32767u) >> 15))))
            return WEX4_E_RECORD;
        if ((rec.flags & WEX4_REC_VALUE_EXACT) &&
            (rec.codec != WEX4_CODEC_WNR3 || !(h->required & WEX4_REQ_VALUE_EXACT))) return WEX4_E_RECORD;
        if (rec.codec == WEX4_CODEC_WNR3) {
            if (!(rec.flags & WEX4_REC_EXPERT) || !rec.blob) return WEX4_E_RECORD;
            wex4_get_blob(v, rec.blob - 1u, &blob);
            if (blob.kind != WEX4_BLOB_WNM2) return WEX4_E_RECORD;
        }
        total += rec.decoded;
        if (rec.decoded > maximum) maximum = rec.decoded;
        if (pending) {   /* continuation part of tensor t - 1 */
            if (rec.tensor_count || rec.first_tensor != t - 1u || (rec.flags & WEX4_REC_EXPERT) ||
                rec.owner != r || rec.decoded > pending_bytes) return WEX4_E_RECORD;
            pending_bytes -= rec.decoded;
            if (!--pending && pending_bytes) return WEX4_E_RECORD;
            continue;
        }
        if (!rec.tensor_count || rec.first_tensor != t || rec.tensor_count > h->tensors - t) return WEX4_E_RECORD;
        for (k = 0; k < rec.tensor_count; k++) {
            wex4_get_tensor(v, t + k, &ten);
            if (ten.record != r || !ten.nbytes) return WEX4_E_TENSOR;
            if (ten.record_count == 1u) {
                if (ten.offset != cursor || ten.nbytes > rec.decoded - cursor) return WEX4_E_TENSOR;
                cursor += ten.nbytes;
            } else {
                if (ten.record_count < 2u || rec.tensor_count != 1u || ten.offset || (rec.flags & WEX4_REC_EXPERT) ||
                    ten.record_count > h->records - r || ten.nbytes <= rec.decoded) return WEX4_E_TENSOR;
                pending = ten.record_count - 1u; pending_bytes = ten.nbytes - rec.decoded; cursor = rec.decoded;
            }
        }
        if (cursor != rec.decoded) return WEX4_E_TENSOR;
        t += rec.tensor_count;
        if (rec.flags & WEX4_REC_EXPERT) {
            WexEntry e;
            if (rec.codec != WEX4_CODEC_RAW && rec.codec != WEX4_CODEC_WNR3) return WEX4_E_EXPERT;
            if ((rec.owner >> 16) >= h->layers || (rec.owner & 0xffffu) >= h->experts ||
                (int64_t)rec.owner <= last_owner || wex4_expert_entry(v, r, &e)) return WEX4_E_EXPERT;
            last_owner = rec.owner;
            if (rec.codec == WEX4_CODEC_WNR3) {   /* smallest legal WNR3 record: longest chunks, 4-byte streams */
                uint64_t streams = 0;
                unsigned int m;
                for (m = 0; m < 3u; m++)
                    streams += wex_nr2_streams((uint32_t)e.scale_shapes[m][0] * e.scale_shapes[m][1], WEX_NR2_MAX_CHUNK) +
                               wex_nr2_streams((uint32_t)e.weight_shapes[m][0] * e.weight_shapes[m][1] * 2u, WEX_NR2_MAX_CHUNK);
                if (rec.stored < wex4_align(WEX_NR2_HEADER_BYTES + streams * 6u)) return WEX4_E_EXPERT;
            }
        } else if (rec.owner != r) return WEX4_E_RECORD;
    }
    if (pending || total != h->total_decoded || maximum != h->max_decoded) return WEX4_E_RECORD;
    if (h->layers && last_owner < 0) return WEX4_E_EXPERT;   /* a grid needs at least one expert record */
    for (; t < h->tensors; t++) {   /* zero-size tensors close the table */
        wex4_get_tensor(v, t, &ten);
        if (ten.nbytes || ten.record_count || ten.record || ten.offset) return WEX4_E_TENSOR;
    }

    /* header page, blobs, records and index tile the file exactly, in offset order */
    pos = WEX4_PAGE; r = 0; index_done = !h->index_bytes;
    for (;;) {
        uint64_t size;
        if (b < h->blobs && (wex4_get_blob(v, b, &blob), blob.offset == pos)) { size = wex4_align(blob.bytes); b++; }
        else if (r < h->records && (wex4_get_record(v, r, &rec), rec.offset == pos)) { size = rec.stored; r++; }
        else if (!index_done && h->index_offset == pos) { size = index_len; index_done = 1; }
        else break;
        if (size > h->file_bytes - pos) return WEX4_E_LAYOUT;
        pos += size;
    }
    if (b != h->blobs || r != h->records || !index_done || pos != h->file_bytes) return WEX4_E_LAYOUT;
    return WEX4_OK;
}

/* Tensor index by name (binary search over the name order), or -1. */
static inline int64_t wex4_find(const Wex4View *v, const unsigned char *name, size_t len) {
    uint32_t lo = 0, hi = v->h.tensors;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2u, at = wex_nr_u32(v->order + (size_t)mid * 4u);
        Wex4Tensor t;
        int c;
        wex4_get_tensor(v, at, &t);
        c = memcmp(t.name, name, t.name_len < len ? t.name_len : len);
        if (!c && t.name_len == len) return at;
        if (c < 0 || (!c && t.name_len < len)) lo = mid + 1u; else hi = mid;
    }
    return -1;
}

/* ---- WBP1: byte-plane rANS (codec 6) -------------------------------------
 * Plane p holds byte p of every P-byte element. A plane is either order-0 rANS
 * (the WNR3 scale-stream coder with a per-record CDF, x0 = L + h23) or raw with
 * one MurmurHash3 word per chunk, so every chunk is verified where it is decoded.
 * transform 1 (floats): every little-endian element is rotated left by one bit
 * before the split, so the top plane is the whole exponent and the sign joins
 * the incompressible mantissa bits. Hashes cover the coded (rotated) bytes. */
typedef struct {
    uint32_t planes, chunk, mask, transform, n[8], streams[8];
    const unsigned char *cdf[8], *table[8], *payload[8];
} Wex4Bp;
static inline uint32_t wex4_bp_plane_bytes(uint32_t decoded, uint32_t planes, uint32_t p) {
    return decoded > p ? (decoded - p + planes - 1u) / planes : 0u;
}
/* Rotate every little-endian `planes`-byte element by one bit: left (encode) or right (decode). */
static inline void wex4_bp_rotate(unsigned char *data, size_t bytes, uint32_t planes, int left) {
    size_t i;
    uint32_t k;
    for (i = 0; i + planes <= bytes; i += planes) {
        if (left) {
            unsigned int carry = data[i + planes - 1u] >> 7;
            for (k = 0; k < planes; k++) {
                unsigned int next = data[i + k] >> 7;
                data[i + k] = (unsigned char)(data[i + k] << 1 | carry); carry = next;
            }
        } else {
            unsigned int carry = data[i] & 1u;
            for (k = planes; k-- > 0;) {
                unsigned int next = data[i + k] & 1u;
                data[i + k] = (unsigned char)(data[i + k] >> 1 | carry << 7); carry = next;
            }
        }
    }
}
static inline int wex4_bp_parse(const unsigned char *rec, size_t total, uint32_t decoded, uint32_t owner, Wex4Bp *b) {
    uint64_t at = WEX4_BP_HEADER;
    uint32_t p, s;
    if (!rec || !b || total < WEX4_PAGE || (total & (WEX4_PAGE - 1u)) || wex_nr_u32(rec) != WEX4_BP_MAGIC ||
        wex_nr_u32(rec + 4u) != owner || wex_nr_u32(rec + 8u) != decoded || !decoded || decoded > WEX4_MAX_DECODED) return -1;
    b->planes = rec[12]; b->mask = rec[14]; b->transform = rec[15];
    if ((b->planes != 1u && b->planes != 2u && b->planes != 4u && b->planes != 8u) || rec[13] < 10u || rec[13] > 15u ||
        b->mask >> b->planes || b->transform > 1u || (b->transform && (b->planes == 1u || decoded % b->planes))) return -1;
    b->chunk = 1u << rec[13];
    for (p = 0; p < b->planes; p++) {
        b->n[p] = wex4_bp_plane_bytes(decoded, b->planes, p);
        b->streams[p] = wex_nr2_streams(b->n[p], b->chunk);
        b->cdf[p] = NULL;
        if (!(b->mask >> p & 1u)) continue;
        if (!b->n[p] || total - at < 514u) return -1;
        b->cdf[p] = rec + at; at += 514u;
        if (!wex_nr2_cdf_ok(b->cdf[p], 256u)) return -1;
    }
    for (p = 0; p < b->planes; p++) {
        uint64_t bytes = (uint64_t)b->streams[p] * (b->mask >> p & 1u ? 2u : 4u);
        if (total - at < bytes) return -1;
        b->table[p] = rec + at; at += bytes;
    }
    for (p = 0; p < b->planes; p++) {
        uint64_t bytes = b->n[p];
        if (b->mask >> p & 1u)
            for (s = 0, bytes = 0; s < b->streams[p]; s++) {
                uint32_t length = wex_nr_u16(b->table[p] + (size_t)s * 2u);
                if (length < 4u) return -1;
                bytes += length;
            }
        if (total - at < bytes) return -1;
        b->payload[p] = rec + at; at += bytes;
    }
    if (wex4_align(at) != total) return -1;
    for (; at < total; at++) if (rec[at]) return -1;
    return 0;
}
/* Decode into out[0..decoded); out is scratch until this returns 0. */
static inline int wex4_bp_decode(const unsigned char *rec, size_t total, uint32_t decoded, uint32_t owner, unsigned char *out) {
    Wex4Bp b;
    uint32_t p, g = 0, ident[4];
    unsigned char *tmp, *lut;
    if (!out || wex4_bp_parse(rec, total, decoded, owner, &b)) return -1;
    tmp = (unsigned char *)malloc((size_t)b.chunk + WEX_NR2_TOTAL);
    if (!tmp) return -1;
    lut = tmp + b.chunk;   /* slot -> symbol of the plane being decoded */
    ident[0] = WEX4_BP_MAGIC; ident[1] = owner; ident[3] = wex_nr_u16(rec + 12u) | (uint32_t)rec[15] << 24;
    for (p = 0; p < b.planes; p++) {
        const unsigned char *payload = b.payload[p];
        uint32_t first, s = 0, j;
        if (b.cdf[p])
            for (j = 0; j < 256u; j++)
                memset(lut + wex_nr_u16(b.cdf[p] + j * 2u), (int)j,
                       wex_nr_u16(b.cdf[p] + j * 2u + 2u) - wex_nr_u16(b.cdf[p] + j * 2u));
        for (first = 0; first < b.n[p]; first += b.chunk, s++, g++) {
            uint32_t count = b.n[p] - first < b.chunk ? b.n[p] - first : b.chunk;
            ident[2] = g;
            if (b.cdf[p]) {
                uint32_t length = wex_nr_u16(b.table[p] + (size_t)s * 2u), state;
                const unsigned char *ptr, *end;
                if (wex_nr2_stream_open(payload, length, &ptr, &end, &state)) goto fail;
                for (j = 0; j < count; j++) {   /* wex_nr2_decode_symbol with a table instead of the binary search */
                    uint32_t slot = state & (WEX_NR2_TOTAL - 1u), symbol = lut[slot];
                    uint32_t start = wex_nr_u16(b.cdf[p] + symbol * 2u);
                    state = (wex_nr_u16(b.cdf[p] + symbol * 2u + 2u) - start) * (state >> WEX_NR2_PRECISION) + slot - start;
                    while (state < WEX_NR_L) {
                        if (ptr == end) goto fail;
                        state = state << 8 | *ptr++;
                    }
                    tmp[j] = (unsigned char)symbol;
                }
                if (ptr != end || state != wex_nr3_stream_x0(ident, tmp, count)) goto fail;
                payload += length;
            } else {
                if (wex_nr3_h32_bytes(wex_nr3_ident_hash(ident), payload, count, 16u) != wex_nr_u32(b.table[p] + (size_t)s * 4u))
                    goto fail;
                memcpy(tmp, payload, count);
                payload += count;
            }
            for (j = 0; j < count; j++) out[(size_t)(first + j) * b.planes + p] = tmp[j];
        }
    }
    if (b.transform) wex4_bp_rotate(out, decoded, b.planes, 0);
    free(tmp);
    return 0;
fail:
    free(tmp);
    return -1;
}
/* Canonical encoder: writes the padded record, *written = its size (a multiple
 * of 4096). The caller stores it only when it is smaller than the raw record. */
static inline int wex4_bp_encode(const unsigned char *data, uint32_t decoded, uint32_t planes, uint32_t chunk_log2,
                                 uint32_t transform, uint32_t owner, unsigned char *out, size_t capacity, size_t *written) {
    uint32_t chunk, p, g = 0, mask = 0, n[8], streams[8], ident[4];
    unsigned char *plane = NULL, *coded = NULL, *enc[8] = {0}, *lens[8] = {0}, cdf[8][514], tmp[65536 + 16];
    uint64_t enc_bytes[8] = {0}, at;
    int result = -1;
    if (!data || !out || !written || !decoded || decoded > WEX4_MAX_DECODED ||
        (planes != 1u && planes != 2u && planes != 4u && planes != 8u) || chunk_log2 < 10u || chunk_log2 > 15u ||
        transform > 1u || (transform && (planes == 1u || decoded % planes))) return -1;
    chunk = 1u << chunk_log2;
    plane = (unsigned char *)malloc(wex4_bp_plane_bytes(decoded, planes, 0));
    if (!plane) return -1;
    if (transform) {
        coded = (unsigned char *)malloc(decoded);
        if (!coded) { free(plane); return -1; }
        memcpy(coded, data, decoded);
        wex4_bp_rotate(coded, decoded, planes, 1);
        data = coded;
    }
    ident[0] = WEX4_BP_MAGIC; ident[1] = owner; ident[3] = planes | chunk_log2 << 8 | transform << 24;
    for (p = 0; p < planes; p++) {
        uint64_t counts[256] = {0};
        uint32_t i, first, s = 0;
        n[p] = wex4_bp_plane_bytes(decoded, planes, p);
        streams[p] = wex_nr2_streams(n[p], chunk);
        if (!n[p]) continue;
        for (i = 0; i < n[p]; i++) counts[plane[i] = data[(size_t)i * planes + p]]++;
        wex_nr2_quantize(counts, 256u, cdf[p]);
        enc[p] = (unsigned char *)malloc((size_t)n[p] * 2u + (size_t)streams[p] * 8u);
        lens[p] = (unsigned char *)malloc((size_t)streams[p] * 4u);
        if (!enc[p] || !lens[p]) goto finish;
        for (first = 0; first < n[p]; first += chunk, s++) {
            uint32_t count = n[p] - first < chunk ? n[p] - first : chunk, j, state, length;
            unsigned char *end = tmp + sizeof(tmp), *cursor = end;
            ident[2] = g + s;
            state = wex_nr3_stream_x0(ident, plane + first, count);
            for (j = count; j-- > 0;)
                if (wex_nr2_put(&state, &cursor, tmp, cdf[p], plane[first + j])) goto finish;
            if (wex_nr2_flush(state, &cursor, tmp)) goto finish;
            length = (uint32_t)(end - cursor);
            if (length > 65535u) goto finish;
            memcpy(enc[p] + enc_bytes[p], cursor, length); enc_bytes[p] += length;
            lens[p][s * 2u] = (unsigned char)length; lens[p][s * 2u + 1u] = (unsigned char)(length >> 8);
        }
        if (514u + 2u * (uint64_t)streams[p] + enc_bytes[p] < 4u * (uint64_t)streams[p] + n[p]) mask |= 1u << p;
        else {   /* raw plane: one MurmurHash3 word per chunk */
            for (first = 0, s = 0; first < n[p]; first += chunk, s++) {
                uint32_t count = n[p] - first < chunk ? n[p] - first : chunk, h;
                ident[2] = g + s;
                h = wex_nr3_h32_bytes(wex_nr3_ident_hash(ident), plane + first, count, 16u);
                lens[p][s * 4u] = (unsigned char)h; lens[p][s * 4u + 1u] = (unsigned char)(h >> 8);
                lens[p][s * 4u + 2u] = (unsigned char)(h >> 16); lens[p][s * 4u + 3u] = (unsigned char)(h >> 24);
            }
            memcpy(enc[p], plane, n[p]); enc_bytes[p] = n[p];
        }
        g += streams[p];
    }
    at = WEX4_BP_HEADER;
    for (p = 0; p < planes; p++) at += (mask >> p & 1u ? 514u + 2u * (uint64_t)streams[p] : 4u * (uint64_t)streams[p]) + enc_bytes[p];
    if (wex4_align(at) > capacity) goto finish;
    memset(out, 0, (size_t)wex4_align(at));
    {
        uint32_t fields[3] = {WEX4_BP_MAGIC, owner, decoded}, i;
        for (i = 0; i < 12u; i++) out[i] = (unsigned char)(fields[i / 4u] >> ((i % 4u) * 8u));
        out[12] = (unsigned char)planes; out[13] = (unsigned char)chunk_log2; out[14] = (unsigned char)mask;
        out[15] = (unsigned char)transform;
    }
    at = WEX4_BP_HEADER;
    for (p = 0; p < planes; p++) if (mask >> p & 1u) { memcpy(out + at, cdf[p], 514u); at += 514u; }
    for (p = 0; p < planes; p++) {
        size_t bytes = (size_t)streams[p] * (mask >> p & 1u ? 2u : 4u);
        if (bytes) memcpy(out + at, lens[p], bytes);
        at += bytes;
    }
    for (p = 0; p < planes; p++) { if (enc_bytes[p]) memcpy(out + at, enc[p], (size_t)enc_bytes[p]); at += enc_bytes[p]; }
    *written = (size_t)wex4_align(at);
    result = 0;
finish:
    for (p = 0; p < 8u; p++) { free(enc[p]); free(lens[p]); }
    free(plane); free(coded);
    return result;
}

/* Decode record r from its stored bytes (rec.stored of them) into out
 * (rec.decoded bytes; scratch until this returns 0). model: the opened WNM2
 * blob of a codec-5 record, else NULL. check_sha also verifies the record's
 * SHA-256 before decoding. */
static inline int wex4_decode_record(const Wex4View *v, uint32_t r, const unsigned char *stored, unsigned char *out,
                                     const WexNr2Model *model, int check_sha) {
    Wex4Record rec;
    uint32_t i;
    if (!v || !stored || !out || r >= v->h.records) return WEX4_E_ARG;
    wex4_get_record(v, r, &rec);
    if (check_sha) {
        unsigned char digest[32];
        wex4_sha256(stored, rec.stored, digest);
        if (memcmp(digest, rec.sha, 32u)) return WEX4_E_HASH;
    }
    if (rec.codec == WEX4_CODEC_RAW) {
        for (i = rec.decoded; i < rec.stored; i++) if (stored[i]) return WEX4_E_DATA;
        memcpy(out, stored, rec.decoded);
        return WEX4_OK;
    }
    if (rec.codec == WEX4_CODEC_WBP1)
        return wex4_bp_decode(stored, rec.stored, rec.decoded, rec.owner, out) ? WEX4_E_DATA : WEX4_OK;
    if (rec.codec == WEX4_CODEC_WNR3) {
        WexEntry e;
        WexNr2Layout layout;
        WexNr2Record view;
        Wex4Blob blob;
        unsigned char *scales[3], *weights[3];
        if (!model || wex4_expert_entry(v, r, &e)) return WEX4_E_ARG;
        wex4_get_blob(v, rec.blob - 1u, &blob);
        /* the record names its owner and its value-exact (canonical zero) form; both must match the index */
        if (model->id != blob.id || wex_nr_u32(stored + 28u) != rec.owner ||
            !(stored[27] & WEX_NR3_FLAG_CZ) != !(rec.flags & WEX4_REC_VALUE_EXACT)) return WEX4_E_DATA;
        wex_nr2_layout(&e, &layout);
        if (wex_nrx_parse(stored, rec.stored, &layout, model->id, WEX_NR3_MAGIC, &view)) return WEX4_E_DATA;
        for (i = 0; i < 3u; i++) { scales[i] = out + e.scale_offsets[i]; weights[i] = out + e.weight_offsets[i]; }
        return WEX4_NR3_DECODE(&view, &layout, model, scales, weights) ? WEX4_E_DATA : WEX4_OK;
    }
    return WEX4_E_UNSUPPORTED;
}

#endif /* KAIRA_WEX4_H */
