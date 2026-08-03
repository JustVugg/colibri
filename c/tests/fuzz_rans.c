/* Structured-mutation fuzz for rans.h parse+decode against hostile records.
 * Every mutant blob is placed in a fresh heap buffer of exactly
 * blob_len + RANS_SLACK bytes so ASan catches any read past the documented
 * slack contract. Decode output buffer is exactly packed_bytes.
 *
 * Seeded and bounded (7 base records x 700 structured mutants). Build+run
 * via `make fuzz-rans` (ASan+UBSan, non-recoverable). Exit nonzero on any
 * cross-arm divergence; sanitizers abort on any memory/UB report. NOT part
 * of TEST_BINS: the sanitizer flags differ from the normal test build.
 * Adopted from the E1 review round's audit harness (its first run found the
 * u64-wrap acceptance in rans_record_parse that E_COUNT_MISMATCH's
 * non-wrapping form now refuses). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../rans.h"

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static int parsed_ok = 0, refused = 0, decoded_ok = 0, decode_refused = 0, skipped_alloc = 0;

static void exercise(const uint8_t *mut, uint64_t mlen, const rans_table *t, uint32_t N) {
    /* fresh allocation: blob + slack ONLY */
    uint8_t *blob = (uint8_t *)malloc(mlen + RANS_SLACK);
    if (!blob) return;
    memcpy(blob, mut, mlen);
    memset(blob + mlen, 0, RANS_SLACK);
    rans_record rec;
    rans_err e = rans_record_parse(blob, mlen, N, &rec);
    if (e != RANS_OK) {
        refused++;
        free(blob);
        return;
    }
    parsed_ok++;
    if (rec.n_symbols > (1ull << 24) || rec.packed_bytes > (1ull << 28)) {
        skipped_alloc++;
        free(blob);
        return;
    }
    uint8_t *out = (uint8_t *)malloc(rec.packed_bytes ? rec.packed_bytes : 1);
    if (!out) {
        skipped_alloc++;
        free(blob);
        return;
    }
    const rans_path arms[] = {RANS_PATH_SCALAR, RANS_PATH_SCALAR_BF, RANS_PATH_NEON};
    uint8_t *ref = NULL;
    for (int a = 0; a < 3; a++) {
        memset(out, 0xEE, rec.packed_bytes);
        e = rans_record_decode_packed(&rec, t, N, arms[a], out);
        if (e != RANS_OK) {
            decode_refused++;
            continue;
        }
        decoded_ok++;
        if (a == 0) {
            ref = (uint8_t *)malloc(rec.packed_bytes);
            memcpy(ref, out, rec.packed_bytes);
        } else if (ref && memcmp(ref, out, rec.packed_bytes) != 0) {
            fprintf(stderr, "ARM DIVERGENCE on mutant (arm %d)\n", a);
            exit(2);
        }
    }
    /* checked per-stream decode over every stream too */
    for (uint32_t i = 0; i < N; i++) {
        uint64_t off0 = rec.stream_offsets[i], off1 = rec.stream_offsets[i + 1];
        uint64_t ns = 0;
        for (uint64_t j = i; j < rec.n_symbols; j += N) ns++;
        if (ns > (1ull << 26)) break;
        uint8_t *nib = (uint8_t *)malloc(ns ? ns : 1);
        rans_decode_stream_checked(rec.payload + off0, off1 - off0, ns, t->freq, t->start, t->slot_to_symbol,
                                   t->scale_bits, nib);
        free(nib);
    }
    free(ref);
    free(out);
    free(blob);
}

int main(void) {
    /* base table: the 15-symbol rare-mix shape */
    uint32_t sb = 14, M = 1u << 14;
    uint32_t freq[16] = {0}, start[16] = {0};
    for (int s = 1; s <= 7; s++) freq[s] = 1;
    for (int s = 8; s <= 14; s++) freq[s] = 3;
    freq[15] = M - 7 - 21;
    uint64_t acc = 0;
    for (int s = 0; s < 16; s++) {
        start[s] = (uint32_t)acc;
        acc += freq[s];
    }
    uint16_t *slot = (uint16_t *)malloc(sizeof(uint16_t) * M);
    for (int s = 0; s < 16; s++)
        for (uint32_t i = 0; i < freq[s]; i++) slot[start[s] + i] = (uint16_t)s;
    rans_table t;
    if (rans_table_init(&t, sb, freq, start, slot) != RANS_OK) return 1;

    const uint32_t N = RANS_NSTREAMS;
    const uint64_t nsyms[] = {2, 255, 256, 511, 512, 8192, 65537};
    for (size_t bi = 0; bi < sizeof(nsyms) / sizeof(nsyms[0]); bi++) {
        uint64_t n = nsyms[bi];
        uint8_t *data = (uint8_t *)malloc(n);
        for (uint64_t i = 0; i < n; i++) data[i] = (uint8_t)((rnd() % 3 == 0) ? 1 + (rnd() % 14) : 15);
        uint64_t cap = rans_record_bound(n, N) + RANS_SLACK;
        uint8_t *base = (uint8_t *)calloc(cap, 1);
        uint64_t blen = 0;
        if (rans_record_encode(data, n, &t, N, base, cap, &blen) != RANS_OK) return 3;
        /* sanity: unmutated parses+decodes */
        exercise(base, blen, &t, N);

        uint8_t *mut = (uint8_t *)malloc(blen + 64);
        for (int it = 0; it < 700; it++) {
            uint64_t mlen = blen;
            memcpy(mut, base, blen);
            switch (it % 7) {
            case 0: { /* random byte corruption, 1-8 bytes anywhere */
                int k = 1 + (int)(rnd() % 8);
                for (int j = 0; j < k; j++) mut[rnd() % blen] ^= (uint8_t)(1 + (rnd() & 0xFF));
                break;
            }
            case 1: { /* header field attacks */
                uint64_t v;
                switch (rnd() % 6) {
                case 0:
                    v = 0;
                    break;
                case 1:
                    v = UINT64_MAX;
                    break;
                case 2:
                    v = UINT64_MAX - 1;
                    break;
                case 3:
                    v = (uint64_t)1 << 32;
                    break;
                case 4:
                    v = n + 1;
                    break;
                default:
                    v = n - 1;
                    break;
                }
                if (rnd() & 1) memcpy(mut, &v, 8);
                else memcpy(mut + 8, &v, 8);
                /* half the time keep packed_bytes consistent with lying n */
                if (rnd() & 1) {
                    uint64_t nn;
                    memcpy(&nn, mut, 8);
                    uint64_t pb = (nn + 1) / 2;
                    memcpy(mut + 8, &pb, 8);
                }
                break;
            }
            case 2: { /* offsets attacks */
                uint32_t idx = rnd() % (N + 1);
                uint32_t v;
                switch (rnd() % 5) {
                case 0:
                    v = 0xFFFFFFFFu;
                    break;
                case 1:
                    v = 0;
                    break;
                case 2:
                    v = (uint32_t)(rnd());
                    break;
                case 3: {
                    uint32_t cur;
                    memcpy(&cur, mut + 16 + 4 * (uint64_t)idx, 4);
                    v = cur + 1;
                    break;
                }
                default: {
                    uint32_t cur;
                    memcpy(&cur, mut + 16 + 4 * (uint64_t)idx, 4);
                    v = cur ? cur - 1 : 0;
                    break;
                }
                }
                memcpy(mut + 16 + 4 * (uint64_t)idx, &v, 4);
                break;
            }
            case 3: /* truncate */
                mlen = rnd() % blen;
                break;
            case 4: /* extend with garbage */
                mlen = blen + 1 + rnd() % 63;
                for (uint64_t j = blen; j < mlen; j++) mut[j] = (uint8_t)rnd();
                break;
            case 5: { /* swap two offsets (reversed ranges) */
                uint32_t i1 = rnd() % (N + 1), i2 = rnd() % (N + 1);
                uint32_t a1, a2;
                memcpy(&a1, mut + 16 + 4 * (uint64_t)i1, 4);
                memcpy(&a2, mut + 16 + 4 * (uint64_t)i2, 4);
                memcpy(mut + 16 + 4 * (uint64_t)i1, &a2, 4);
                memcpy(mut + 16 + 4 * (uint64_t)i2, &a1, 4);
                break;
            }
            default: { /* payload bit flips near stream boundaries */
                uint32_t i1 = rnd() % N;
                uint32_t o;
                memcpy(&o, base + 16 + 4 * (uint64_t)i1, 4);
                uint64_t head = rans_record_header_bytes(N);
                uint64_t p = head + o + (rnd() % 8);
                if (p < blen) mut[p] ^= (uint8_t)(1u << (rnd() % 8));
                break;
            }
            }
            if (mlen == 0) {
                refused++;
                continue;
            }
            exercise(mut, mlen, &t, N);
        }
        free(mut);
        free(base);
        free(data);
    }
    printf("fuzz done: parsed_ok=%d refused=%d decoded_ok=%d decode_refused=%d skipped=%d\n", parsed_ok, refused,
           decoded_ok, decode_refused, skipped_alloc);
    rans_table_free(&t);
    free(slot);
    return 0;
}
