/* wex_decode.c on the CPU, no model and no GPU: the WNM2 model registry, a WEX4 container's
 * model blob checked against its index hash, and the in-place decode of WNR3 (codec 5) records
 * made with the reference encoder (tests/wnr3_fixture.h), plain and canonical zero. A record
 * whose model is not registered, or any record with a changed bit, must be refused, never
 * decoded to other bytes; decodes on several threads at once must agree. The engines include
 * wex_decode.c with COLI_WEX_DECODE_API static inline, and so does this test. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"   /* setenv / unsetenv on Windows */
#define COLI_WEX_DECODE_API static inline
#include "../wex_decode.c"
#include "wex4_fixture.h"
#include "wnr3_fixture.h"

enum { ROWS = 64, COLS = 256, SCALE_M = ROWS * COLS / 32, WEIGHT_M = ROWS * COLS / 2 };
#define RAW (3u * (SCALE_M + WEIGHT_M))            /* 26,112 bytes: 7 pages aligned */
#define ALIGNED ((size_t)WEX_ALIGN_UP(RAW, WEX_ALIGN))

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)
static void ok(const char *what, int before) { if (failures == before) printf("ok   %s\n", what); }

static WexEntry entry_of(void) {
    WexEntry e;
    memset(&e, 0, sizeof e);
    e.scale_bytes = 3u * SCALE_M; e.weight_bytes = 3u * WEIGHT_M; e.total_bytes = (uint32_t)ALIGNED;
    for (unsigned m = 0; m < 3u; m++) {
        e.scale_shapes[m][0] = e.weight_shapes[m][0] = ROWS;
        e.scale_shapes[m][1] = COLS / 32; e.weight_shapes[m][1] = COLS / 2;
        e.scale_offsets[m] = m * SCALE_M; e.weight_offsets[m] = 3u * SCALE_M + m * WEIGHT_M;
    }
    return e;
}

/* A record an MXFP4 expert could hold: scales near 2^-7, nibbles mostly small. negzero: some -0. */
static void fill(unsigned char *raw, uint32_t seed, int negzero) {
    static const unsigned char mag[16] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4, 6, 7};
    for (size_t j = 0; j < RAW; j++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        if (j < 3u * SCALE_M) { raw[j] = (unsigned char)(118u + seed % 5u); continue; }
        unsigned lo = mag[seed & 15u], hi = mag[(seed >> 4) & 15u];
        if (lo) lo |= seed >> 8 & 8u; else if (negzero && !(seed % 97u)) lo = 8u;
        if (hi) hi |= seed >> 9 & 8u;
        raw[j] = (unsigned char)(lo | hi << 4);
    }
}

typedef struct { const WexEntry *e; const unsigned char *record; size_t stored; const unsigned char *want; int same; } Job;
static void *decode_job(void *arg) {
    Job *j = arg;
    unsigned char *slab = malloc(ALIGNED);
    if (!slab) return NULL;
    memcpy(slab + wex_record_read_offset(j->e), j->record, j->stored);
    j->same = !coli_wex_decode_record(j->e, slab, ALIGNED) && !memcmp(slab, j->want, RAW);
    free(slab);
    return NULL;
}

/* A WEX1 file of layers x 2 experts of this layout (records behind the table, cell order). */
static unsigned char *wex1(unsigned char (*raw)[RAW], int layers, size_t *bytes) {
    size_t cells = (size_t)layers * 2, table = (size_t)WEX_ALIGN_UP(cells * sizeof(WexEntry), WEX_ALIGN);
    size_t total = WEX_ALIGN + table + cells * ALIGNED, at = WEX_ALIGN + table;
    unsigned char *out = calloc(1, total);
    if (!out) return NULL;
    WexHeader *h = (WexHeader *)out;
    h->magic = WEX_MAGIC_V1; h->version = 1; h->num_layers = (uint32_t)layers; h->num_experts = 2;
    h->aligned_expert_bytes = (uint32_t)ALIGNED; h->table_offset = WEX_ALIGN;
    for (size_t c = 0; c < cells; c++, at += ALIGNED) {
        WexEntry *e = (WexEntry *)(out + WEX_ALIGN) + c;
        *e = entry_of();
        e->file_offset = at;
        memcpy(out + at, raw[c], RAW);
    }
    *bytes = total;
    return out;
}

int main(void) {
    static unsigned char raw[4][RAW], want[RAW], record[ALIGNED], slab[ALIGNED + WEX_ALIGN];
    unsigned char *blob = malloc(WEX_NR2_MODEL_BYTES), *blob_cz = malloc(WEX_NR2_MODEL_BYTES);
    const WexEntry plain = entry_of();
    WexEntry e = plain;
    WexNr2Model model, model_cz;
    const unsigned char *train[4];
    uint32_t id = 0, again = 0;
    int before;
    setbuf(stdout, NULL);
    unsetenv("DSV4_CUDA_MOE_MASKED");
    if (!blob || !blob_cz) return 1;
    for (int i = 0; i < 4; i++) { fill(raw[i], 20261116u + 7919u * (uint32_t)i, i == 3); train[i] = raw[i]; }
    wnr3f_train(&plain, train, 3, 0, blob, &model);
    wnr3f_train(&plain, train, 4, 1, blob_cz, &model_cz);

    before = failures;
    {
        unsigned char *bad = malloc(WEX_NR2_MODEL_BYTES);
        CHECK(bad);
        memcpy(bad, blob, WEX_NR2_MODEL_BYTES);
        bad[0] ^= 1u;                                              /* not "WNM2" */
        CHECK(coli_wex_register_model(bad, WEX_NR2_MODEL_BYTES, &id) == -1);
        CHECK(coli_wex_register_model(blob, WEX_NR2_MODEL_BYTES - 1, &id) == -1);
        free(bad);
    }
    CHECK(!coli_wex_register_model(blob, WEX_NR2_MODEL_BYTES, &id) && id == model.id && wex_nr2_model_count == 1);
    ok("an invalid model blob is refused; a valid one registers and returns its id", before);
    before = failures;
    CHECK(!coli_wex_register_model(blob, WEX_NR2_MODEL_BYTES, &again) && again == id && wex_nr2_model_count == 1);
    ok("the same blob again: the same id, nothing added", before);

    before = failures;
    {   /* as a WEX4 container gives it: the blob, its zero padding to the page, the index's SHA-256 and id */
        size_t stored = (size_t)wex4_align(WEX_NR2_MODEL_BYTES);
        unsigned char *page = calloc(1, stored), sha[32];
        CHECK(page);
        memcpy(page, blob_cz, WEX_NR2_MODEL_BYTES);
        wex4_sha256(blob_cz, WEX_NR2_MODEL_BYTES, sha);
        sha[5] ^= 1u;
        CHECK(coli_wex4_register_model(page, WEX_NR2_MODEL_BYTES, stored, sha, model_cz.id) == -1);
        sha[5] ^= 1u;
        page[stored - 1] = 1u;
        CHECK(coli_wex4_register_model(page, WEX_NR2_MODEL_BYTES, stored, sha, model_cz.id) == -1);
        page[stored - 1] = 0;
        CHECK(coli_wex4_register_model(page, WEX_NR2_MODEL_BYTES, stored, sha, model_cz.id ^ 1u) == -1);
        CHECK(wex_nr2_model_count == 1);
        CHECK(!coli_wex4_register_model(page, WEX_NR2_MODEL_BYTES, stored, sha, model_cz.id) && wex_nr2_model_count == 2);
        free(page);
    }
    ok("coli_wex4_register_model: the index's blob registers; another SHA-256, nonzero padding or another id do not",
       before);

    before = failures;
    size_t stored = wnr3f_encode(&model, &plain, raw[0], 0, 3u << 16 | 1u, record, sizeof record);
    CHECK(stored && stored < ALIGNED);
    e.flags = WEX_ENTRY_NEURAL_RANS3; e.total_bytes = (uint32_t)stored;
    size_t at = wex_record_read_offset(&e);
    CHECK(at == ALIGNED - stored);
    CHECK(wex_nr3_owner_ok(&e, record, 3u, 1u, 0) && !wex_nr3_owner_ok(&e, record, 3u, 2u, 0) &&
          !wex_nr3_owner_ok(&e, record, 3u, 1u, WEX_HEADER_CANONICAL_ZERO));
    memset(slab, 0xa5, sizeof slab);
    memcpy(slab + at, record, stored);
    CHECK(!coli_wex_decode_record(&e, slab, ALIGNED) && !memcmp(slab, raw[0], RAW));
    CHECK(coli_wex_decode_record(&e, slab, ALIGNED - 1) == -1);   /* a slab smaller than the decoded page */
    ok("a WNR3 record decodes in place to the record it was made from", before);

    before = failures;
    {
        WexEntry z = plain;
        unsigned char rec_cz[ALIGNED];
        size_t n = wnr3f_encode(&model_cz, &plain, raw[3], 1, 0u << 16 | 0u, rec_cz, sizeof rec_cz);
        uint64_t mapped;
        CHECK(n);
        z.flags = WEX_ENTRY_NEURAL_RANS3; z.total_bytes = (uint32_t)n;
        memcpy(want, raw[3], RAW);
        mapped = wex_nr3_canonical_zero(want + plain.scale_bytes, plain.weight_bytes);
        CHECK(mapped > 0 && memcmp(want, raw[3], RAW));
        CHECK(wex_nr3_owner_ok(&z, rec_cz, 0, 0, WEX_HEADER_CANONICAL_ZERO) && !wex_nr3_owner_ok(&z, rec_cz, 0, 0, 0));
        memcpy(slab + wex_record_read_offset(&z), rec_cz, n);
        CHECK(!coli_wex_decode_record(&z, slab, ALIGNED) && !memcmp(slab, want, RAW));
    }
    ok("a canonical-zero record decodes to the record with -0 stored as +0", before);

    before = failures;
    memcpy(slab, raw[1], RAW); memset(slab + RAW, 0, ALIGNED - RAW);
    CHECK(!coli_wex_decode_record(&plain, slab, ALIGNED) && !memcmp(slab, raw[1], RAW));
    ok("a RAW entry: nothing to do, the slab is the record", before);

    before = failures;
    {
        unsigned char other_blob[WEX_NR2_MODEL_BYTES], rec2[ALIGNED];
        WexNr2Model other;
        const unsigned char *one[1] = {raw[2]};
        WexEntry o = plain;
        wnr3f_train(&plain, one, 1, 0, other_blob, &other);
        size_t n = wnr3f_encode(&other, &plain, raw[2], 0, 0, rec2, sizeof rec2);
        CHECK(n && other.id != model.id && other.id != model_cz.id);
        o.flags = WEX_ENTRY_NEURAL_RANS3; o.total_bytes = (uint32_t)n;
        memcpy(slab + wex_record_read_offset(&o), rec2, n);
        CHECK(coli_wex_decode_record(&o, slab, ALIGNED) == -1);
    }
    ok("a record whose model is not registered: refused", before);

    before = failures;
    {
        unsigned changed = 0, refused = 0, harmless = 0, wrong = 0;
        uint32_t seed = 99u;
        for (unsigned k = 0; k < 64u * 8u + 4096u; k++) {
            size_t bit;
            if (k < 512u) bit = k;
            else { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; bit = 512u + seed % (stored * 8u - 512u); }
            memcpy(slab + at, record, stored);
            slab[at + bit / 8u] ^= (unsigned char)(1u << (bit % 8u));
            changed++;
            if (coli_wex_decode_record(&e, slab, ALIGNED)) refused++;
            else if (!memcmp(slab, raw[0], RAW)) harmless++;   /* a bit of the record's zero padding */
            else wrong++;
        }
        CHECK(!wrong && refused + harmless == changed && refused > changed * 9u / 10u);
        printf("     %u single-bit changes: %u refused, %u decoded to the right record anyway, %u wrong\n", changed,
               refused, harmless, wrong);
    }
    ok("every bit of the first 64 bytes and 4,096 sampled bits elsewhere changed one at a time: never a wrong record",
       before);

    before = failures;
    {
        pthread_t threads[8];
        Job jobs[8];
        for (int t = 0; t < 8; t++) {
            jobs[t] = (Job){&e, record, stored, raw[0], 0};
            CHECK(!pthread_create(&threads[t], NULL, decode_job, &jobs[t]));
        }
        for (int t = 0; t < 8; t++) { pthread_join(threads[t], NULL); CHECK(jobs[t].same); }
    }
    ok("8 threads decoding the same record at once: all right", before);

    before = failures;
    CHECK(!coli_wex_container_refused(0, "test") && !coli_wex_container_refused(WEX_HEADER_CANONICAL_ZERO, "test"));
    setenv("DSV4_CUDA_MOE_MASKED", "1", 1);
    CHECK(coli_wex_container_refused(WEX_HEADER_CANONICAL_ZERO, "test") && !coli_wex_container_refused(0, "test"));
    unsetenv("DSV4_CUDA_MOE_MASKED");
    ok("coli_wex_container_refused: a canonical-zero container under DSV4_CUDA_MOE_MASKED=1 only", before);

    before = failures;
    for (int cz = 0; cz < 2; cz++) {
        size_t old_bytes, w4_bytes, nr3_bytes, offset[4];
        char root[65], root3[65];
        unsigned char *old = wex1(raw, 2, &old_bytes), *w4 = old ? wex4_fixture_from_wex1(old, old_bytes, &w4_bytes, root) : NULL;
        unsigned char *nr3 = w4 ? wnr3f_from_wex4(w4, w4_bytes, cz, &nr3_bytes, root3, offset) : NULL;
        Wex4Header h;
        Wex4View v;
        CHECK(nr3 && strcmp(root, root3));
        if (!nr3) { free(old); free(w4); continue; }
        CHECK(!wex4_parse_header(nr3, nr3_bytes, &h) && h.blobs == 1 && !!(h.required & WEX4_REQ_VALUE_EXACT) == cz);
        CHECK(!wex4_open(&h, nr3 + h.index_offset, (size_t)wex4_align(h.index_bytes), &v));
        Wex4Blob b;
        wex4_get_blob(&v, 0, &b);
        CHECK(b.kind == WEX4_BLOB_WNM2 &&
              !coli_wex4_register_model(nr3 + b.offset, (size_t)b.bytes, (size_t)wex4_align(b.bytes), b.sha, b.id));
        for (uint32_t r = 0; r < h.records; r++) {
            Wex4Record rec;
            WexEntry x;
            wex4_get_record(&v, r, &rec);
            CHECK(rec.codec == WEX4_CODEC_WNR3 && !wex4_expert_entry(&v, r, &x) && x.flags == WEX_ENTRY_NEURAL_RANS3);
            CHECK(offset[(rec.owner >> 16) * 2 + (rec.owner & 0xffffu)] == rec.offset);
            CHECK(wex_nr3_owner_ok(&x, nr3 + rec.offset, rec.owner >> 16, rec.owner & 0xffffu,
                                   cz ? WEX_HEADER_CANONICAL_ZERO : 0));
            memcpy(slab + wex_record_read_offset(&x), nr3 + rec.offset, rec.stored);
            memcpy(want, raw[r], RAW);
            if (cz) wex_nr3_canonical_zero(want + plain.scale_bytes, plain.weight_bytes);
            CHECK(!coli_wex_decode_record(&x, slab, ALIGNED) && !memcmp(slab, want, RAW));
        }
        free(old); free(w4); free(nr3);
    }
    ok("the WNR3 form of a WEX4 container, plain and canonical zero: it opens, its blob registers, every record "
       "decodes to its RAW record", before);

    before = failures;
    {   /* the registry holds 16 models: 3 are in it (two above; the plain container's; the canonical-zero
         * container's is blob_cz again, trained on the same records) */
        unsigned char *more = malloc(WEX_NR2_MODEL_BYTES);
        int added = 0;
        CHECK(more && wex_nr2_model_count == 3);
        for (uint32_t s = 0; more && wex_nr2_model_count < WEX_NR2_MAX_MODELS; s++) {
            const unsigned char *one[1] = {raw[s % 4u]};
            WexNr2Model m;
            fill(raw[s % 4u], 777u + s, 0);
            wnr3f_train(&plain, one, 1, 0, more, &m);
            if (!coli_wex_register_model(more, WEX_NR2_MODEL_BYTES, &id)) added++;
        }
        fill(raw[0], 31337u, 0);
        const unsigned char *one[1] = {raw[0]};
        WexNr2Model m;
        wnr3f_train(&plain, one, 1, 0, more, &m);
        CHECK(added == 13 && coli_wex_register_model(more, WEX_NR2_MODEL_BYTES, &id) == -1);
        CHECK(!coli_wex_register_model(blob, WEX_NR2_MODEL_BYTES, &again) && again == model.id);   /* known: still fine */
        free(more);
    }
    ok("16 models fit; a 17th is refused, one already registered is still found", before);

    free(blob); free(blob_cz);
    if (failures) { fprintf(stderr, "test_wex_decode: %d failures\n", failures); return 1; }
    puts("test_wex_decode: ok");
    return 0;
}
