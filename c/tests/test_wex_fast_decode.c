/* wex_fast_decode.h against the reference decoder of wex_decode.c (wex_decode_nr2_record, wex_nr2_decode of
 * wex_neural_rans.h): the same status and, when both accept, the same bytes.
 *
 *   test_wex_fast_decode [--mutate N] [--seed S]                     synthetic records only (make test-c)
 *   test_wex_fast_decode [--mutate N] [--seed S] CONTAINER.wex [R]   also R real records (default 4) of a WNR3
 *                                                                    WEX4 container, and mini-records cut from them
 *   test_wex_fast_decode --bench CONTAINER.wex [R]                   ms per record, reference against fast
 *
 * equal: every record decoded by the reference, then by every variant of the fast decoder (kernel 0; kernel 1
 * with 1, 2, 3, 4, 6, 8 streams, fused or not) on 1 and 8 threads and by the default entry point; the slab-level
 * wex_fast_decode_record against wex_decode_nr2_record on the whole slab.
 * mutate: N mini-records (random cuts of decoded records, re-encoded with the reference encoder, random chunks and
 * owner; some with a twist the hashes do not see: -0 nibbles with and without the canonical-zero flag, a ceil
 * underflow, ceil flags that do not match), then byte-level damage to them; each one the parser accepts goes to
 * the reference and to every variant: same status, same bytes. From Wex-WRN2's fast-cpu-decode harness.
 * Every buffer a decoder touches sits against a page without access (Windows: PAGE_NOACCESS; elsewhere mmap +
 * PROT_NONE): the record copied without its zero padding flush against the upper guard, the outputs flush against
 * the upper or the lower one, the model too, so a read or a write one byte outside faults. */
#if !defined(_WIN32) && !defined(_DEFAULT_SOURCE)
#define _DEFAULT_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

#include "../compat.h"   /* setenv / unsetenv on Windows */
#define COLI_WEX_DECODE_API static inline
#include "../wex_decode.c"
#include "../wex_fast_decode.h"
#include "wnr3_fixture.h"

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)
static void ok(const char *what, int before) { if (failures == before) printf("ok   %s\n", what); }

/* ---- guarded buffers ------------------------------------------------------- */
typedef struct { unsigned char *lo, *hi; size_t page; } Guard;   /* usable [lo, hi) */
static Guard guard_alloc(size_t bytes) {
    Guard g;
#ifdef _WIN32
    const size_t page = 4096u, pages = (bytes + page - 1u) / page + !bytes;
    DWORD old;
    unsigned char *base = (unsigned char *)VirtualAlloc(NULL, (pages + 2u) * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!base || !VirtualProtect(base, page, PAGE_NOACCESS, &old) ||
        !VirtualProtect(base + (pages + 1u) * page, page, PAGE_NOACCESS, &old)) { fprintf(stderr, "guard_alloc\n"); exit(2); }
#else
    const size_t page = (size_t)sysconf(_SC_PAGESIZE), pages = (bytes + page - 1u) / page + !bytes;
    unsigned char *base = (unsigned char *)mmap(NULL, (pages + 2u) * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == (unsigned char *)MAP_FAILED || mprotect(base, page, PROT_NONE) ||
        mprotect(base + (pages + 1u) * page, page, PROT_NONE)) { fprintf(stderr, "guard_alloc\n"); exit(2); }
#endif
    g.lo = base + page; g.hi = g.lo + pages * page; g.page = page;
    return g;
}
static void guard_free(Guard g) {
#ifdef _WIN32
    VirtualFree(g.lo - g.page, 0, MEM_RELEASE);
#else
    munmap(g.lo - g.page, (size_t)(g.hi - g.lo) + 2u * g.page);
#endif
}

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
#endif
}

static uint64_t rng_state = 1;
static uint64_t rnd(void) {   /* xorshift64* */
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return rng_state * UINT64_C(2685821657736338717);
}
static uint32_t rnd_below(uint32_t n) { return (uint32_t)((rnd() >> 32) % n); }

/* ---- the variants ------------------------------------------------------------ */
typedef struct { const char *name; WexFastOptions o; } Variant;
static const Variant variants[] = {
    {"k0: reference loop per stream", {1, 0, 1, 0}},
    {"k1: 1 stream", {1, 1, 1, 0}},       {"k1: 2 streams", {1, 1, 2, 0}},       {"k1: 3 streams", {1, 1, 3, 0}},
    {"k1: 4 streams", {1, 1, 4, 0}},      {"k1: 6 streams", {1, 1, 6, 0}},       {"k1: 8 streams", {1, 1, 8, 0}},
    {"k1: 1 stream, fused", {1, 1, 1, 1}}, {"k1: 2 streams, fused", {1, 1, 2, 1}}, {"k1: 3 streams, fused", {1, 1, 3, 1}},
    {"k1: 4 streams, fused", {1, 1, 4, 1}}, {"k1: 6 streams, fused", {1, 1, 6, 1}}, {"k1: 8 streams, fused", {1, 1, 8, 1}},
};
#define VARIANTS (sizeof(variants) / sizeof(variants[0]))
static WexFastModel *tables_for(const WexNr2Model *model) {
    WexFastModel *t = malloc(sizeof *t);
    if (!t) { fprintf(stderr, "no memory\n"); exit(2); }
    wex_fast_model_init(t, model);
    return t;
}

/* Output buffers of one decoder: six guarded regions, used flush high or flush low. */
typedef struct { Guard gs[3], gw[3]; unsigned char *scales[3], *weights[3]; } Out;
static void out_alloc(Out *o, const WexNr2Layout *max) {
    for (unsigned m = 0; m < 3u; m++) { o->gs[m] = guard_alloc(max->scale_count[m]); o->gw[m] = guard_alloc(max->symbol_count[m] / 2u); }
}
static void out_free(Out *o) {
    for (unsigned m = 0; m < 3u; m++) { guard_free(o->gs[m]); guard_free(o->gw[m]); }
}
static void out_place(Out *o, const WexNr2Layout *l, int high) {
    for (unsigned m = 0; m < 3u; m++) {
        o->scales[m] = high ? o->gs[m].hi - l->scale_count[m] : o->gs[m].lo;
        o->weights[m] = high ? o->gw[m].hi - l->symbol_count[m] / 2u : o->gw[m].lo;
    }
}
static void out_poison(Out *o, const WexNr2Layout *l) {
    for (unsigned m = 0; m < 3u; m++) { memset(o->scales[m], 0xa5, l->scale_count[m]); memset(o->weights[m], 0xa5, l->symbol_count[m] / 2u); }
}
static int out_equal(const Out *a, const Out *b, const WexNr2Layout *l) {
    for (unsigned m = 0; m < 3u; m++)
        if (memcmp(a->scales[m], b->scales[m], l->scale_count[m]) || memcmp(a->weights[m], b->weights[m], l->symbol_count[m] / 2u))
            return 0;
    return 1;
}
/* The unpadded record flush against the guard page, the view pointed at the copy. */
static void tighten(WexNr2Record *v, const unsigned char *record, Guard g) {
    size_t used = (size_t)v->metadata_bytes + v->scale_payload_bytes + v->weight_payload_bytes;
    unsigned char *copy = g.hi - used;
    memcpy(copy, record, used);
    v->lengths = copy + WEX_NR2_HEADER_BYTES;
    v->scale_payload = copy + v->metadata_bytes;
    v->weight_payload = v->scale_payload + v->scale_payload_bytes;
}

/* ---- records ------------------------------------------------------------------ */
typedef struct { WexEntry entry; WexNr2Layout layout; unsigned char *record; size_t stored; } Rec;   /* record: unpadded or padded */

/* equal on one record: every variant on 1 and 8 threads, the default entry point, and the slab-level call. */
static int equal_one(const char *what, const WexNr2Model *model, const WexFastModel *tables, const Rec *r) {
    WexNr2Record view;
    Out ref, fast;
    Guard rec;
    int status, bad = 0;
    uint64_t raw, aligned;
    if (wex_nr3_parse(r->record, r->stored, &r->layout, model->id, &view)) { printf("%s: does not parse\n", what); return 1; }
    out_alloc(&ref, &r->layout); out_alloc(&fast, &r->layout);
    out_place(&ref, &r->layout, 1);
    rec = guard_alloc((size_t)view.metadata_bytes + view.scale_payload_bytes + view.weight_payload_bytes);
    tighten(&view, r->record, rec);
    status = wex_nr2_decode(&view, &r->layout, model, ref.scales, ref.weights);
    if (status) { printf("%s: the reference refuses it\n", what); return 1; }
    for (size_t v = 0; v < VARIANTS; v++)
        for (int threads = 1; threads <= 8; threads += 7) {
            WexFastOptions o = variants[v].o;
            o.threads = threads;
            out_place(&fast, &r->layout, (int)((v + (size_t)threads) & 1u));
            out_poison(&fast, &r->layout);
            status = wex_fast_decode_with(tables, &view, &r->layout, model, fast.scales, fast.weights, &o);
            if (status || !out_equal(&ref, &fast, &r->layout)) {
                printf("%s: MISMATCH %s, %d threads: status %d\n", what, variants[v].name, threads, status);
                bad++;
            }
        }
    out_place(&fast, &r->layout, 0);
    out_poison(&fast, &r->layout);
    if (wex_fast_decode(&view, &r->layout, model, fast.scales, fast.weights) || !out_equal(&ref, &fast, &r->layout)) {
        printf("%s: MISMATCH wex_fast_decode\n", what); bad++;
    }
    /* the slab-level calls, in place, as an engine makes them */
    if (!wex_entry_raw_sizes(&r->entry, &raw, &aligned)) {
        const size_t at = wex_record_read_offset(&r->entry);
        unsigned char *a = malloc((size_t)aligned), *b = malloc((size_t)aligned);
        static const WexFastOptions def = {1, 1, WEX_FAST_LANES, 1};
        CHECK(a && b);
        memset(a, 0x5a, (size_t)aligned); memset(b, 0x5a, (size_t)aligned);
        memcpy(a + at, r->record, r->entry.total_bytes); memcpy(b + at, r->record, r->entry.total_bytes);
        if (wex_decode_nr2_record(&r->entry, a, (size_t)aligned, model) ||
            wex_fast_decode_record(&r->entry, b, (size_t)aligned, model, tables, &def) || memcmp(a, b, (size_t)raw)) {
            printf("%s: MISMATCH wex_fast_decode_record against wex_decode_nr2_record\n", what); bad++;
        }
        free(a); free(b);
    }
    out_free(&ref); out_free(&fast); guard_free(rec);
    return bad;
}

/* ---- synthetic records ----------------------------------------------------------- */
enum { ROWS = 128, COLS = 2048, SCALE_M = ROWS * COLS / 32, WEIGHT_M = ROWS * COLS / 2 };
#define SRAW (3u * (SCALE_M + WEIGHT_M))   /* 417,792 bytes; 8,192 scale blocks a matrix */
static WexEntry synth_entry(void) {
    WexEntry e;
    uint64_t raw, aligned;
    memset(&e, 0, sizeof e);
    e.scale_bytes = 3u * SCALE_M; e.weight_bytes = 3u * WEIGHT_M;
    for (unsigned m = 0; m < 3u; m++) {
        e.scale_shapes[m][0] = e.weight_shapes[m][0] = ROWS;
        e.scale_shapes[m][1] = COLS / 32; e.weight_shapes[m][1] = COLS / 2;
        e.scale_offsets[m] = m * SCALE_M; e.weight_offsets[m] = 3u * SCALE_M + m * WEIGHT_M;
    }
    wex_entry_raw_sizes(&e, &raw, &aligned);
    e.total_bytes = (uint32_t)aligned;
    return e;
}
/* An MXFP4 expert-like record: scales near 2^-7, nibbles mostly small, some blocks of maximum 6 or 7 (the ceil
 * form), negzero: some -0 codes. */
static void synth_fill(unsigned char *raw, uint32_t seed, int negzero) {
    static const unsigned char mag[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6};
    for (size_t j = 0; j < SRAW; j++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        if (j < 3u * SCALE_M) { raw[j] = (unsigned char)(116u + seed % 9u); continue; }
        unsigned lo = mag[seed & 15u], hi = mag[(seed >> 4) & 15u];
        if (!(seed % 61u)) lo = 7u;
        if (lo) lo |= seed >> 8 & 8u; else if (negzero && !(seed % 97u)) lo = 8u;
        if (hi) hi |= seed >> 9 & 8u;
        raw[j] = (unsigned char)(lo | hi << 4);
    }
}
/* raw as a WNR3 record with these chunks into r (cz: canonical zero first). */
static int synth_encode(const WexNr2Model *model, const unsigned char *raw, int cz, uint32_t wchunk, uint32_t schunk,
                        uint32_t owner, Rec *r) {
    unsigned char *coded = malloc(SRAW), *scales[3], *weights[3];
    size_t written = 0, cap = 2u * SRAW + 65536u;
    int rc;
    r->entry = synth_entry();
    wex_nr2_layout(&r->entry, &r->layout);
    r->record = calloc(1, cap);
    if (!coded || !r->record) return -1;
    uint32_t flags = wnr3f_code(&r->entry, &r->layout, raw, coded, cz);
    wnr3f_split(&r->entry, coded, scales, weights);
    rc = wex_nr3_encode(model, &r->layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                        wchunk, schunk, flags, owner, r->record, cap, &written);
    free(coded);
    if (rc) return -1;
    r->stored = (written + 4095u) & ~(size_t)4095u;
    r->entry.flags = WEX_ENTRY_NEURAL_RANS3;
    r->entry.total_bytes = (uint32_t)r->stored;
    return 0;
}

/* ---- a real container ------------------------------------------------------------ */
typedef struct { FILE *f; Wex4Header h; Wex4View v; unsigned char *index; const WexNr2Model *model; } Container;
static int read_at(FILE *f, void *p, size_t n, uint64_t off) {
#ifdef _WIN32
    if (_fseeki64(f, (long long)off, SEEK_SET)) return -1;
#else
    if (fseeko(f, (off_t)off, SEEK_SET)) return -1;
#endif
    return fread(p, 1, n, f) == n ? 0 : -1;
}
static int container_open(const char *path, Container *c) {
    unsigned char page[WEX4_PAGE];
    uint64_t size;
    c->f = fopen(path, "rb");
    if (!c->f) return -1;
#ifdef _WIN32
    _fseeki64(c->f, 0, SEEK_END); size = (uint64_t)_ftelli64(c->f);
#else
    fseeko(c->f, 0, SEEK_END); size = (uint64_t)ftello(c->f);
#endif
    if (read_at(c->f, page, WEX4_PAGE, 0) || wex4_parse_header(page, size, &c->h)) return -2;
    size_t n = (size_t)wex4_align(c->h.index_bytes);
    c->index = malloc(n);
    if (!c->index || read_at(c->f, c->index, n, c->h.index_offset) || wex4_open(&c->h, c->index, n, &c->v)) return -3;
    c->model = NULL;
    for (uint32_t i = 0; i < c->h.blobs; i++) {
        Wex4Blob bl;
        wex4_get_blob(&c->v, i, &bl);
        if (bl.kind != WEX4_BLOB_WNM2) continue;
        size_t s = (size_t)wex4_align(bl.bytes);
        unsigned char *p = malloc(s);
        if (!p || read_at(c->f, p, s, bl.offset) || coli_wex4_register_model(p, (size_t)bl.bytes, s, bl.sha, bl.id)) return -4;
        free(p);
        c->model = wex_nr2_find(bl.id, NULL);
    }
    return c->model ? 0 : -5;
}
/* Record k of count, spread over the container's WNR3 expert records; stored bytes, padded. */
static int container_record(Container *c, uint32_t k, uint32_t count, Rec *r, char *name, size_t name_bytes) {
    uint32_t list = 0, want, i;
    Wex4Record rec;
    for (i = 0; i < c->h.records; i++) { wex4_get_record(&c->v, i, &rec); list += rec.codec == WEX4_CODEC_WNR3 && (rec.flags & WEX4_REC_EXPERT); }
    if (!list) return -1;
    want = (uint32_t)((uint64_t)k * list / count + list / (2u * count));
    for (i = 0; i < c->h.records; i++) {
        wex4_get_record(&c->v, i, &rec);
        if (rec.codec != WEX4_CODEC_WNR3 || !(rec.flags & WEX4_REC_EXPERT)) continue;
        if (want--) continue;
        if (wex4_expert_entry(&c->v, i, &r->entry)) return -1;
        wex_nr2_layout(&r->entry, &r->layout);
        r->stored = rec.stored;
        r->record = malloc(rec.stored);
        if (!r->record || read_at(c->f, r->record, rec.stored, rec.offset)) return -1;
        snprintf(name, name_bytes, "L%u E%u", rec.owner >> 16, rec.owner & 0xffffu);
        return 0;
    }
    return -1;
}

/* ---- mutate (mini-records) ----------------------------------------------------------- */
#define MINI_MAX_BLOCKS 2500u
#define MINI_CAP ((3u * MINI_MAX_BLOCKS * 17u) * 4u + (1u << 17))
typedef struct {
    WexNr2Layout layout;
    unsigned char record[MINI_CAP];
    size_t total;
    int expect;   /* 0 accept, 1 reject, 2 whatever the reference says */
} Mini;
typedef struct { unsigned char *scales[3], *weights[3]; WexNr2Layout layout; } Source;   /* decoded, no -0 */

static void put32(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24); }
static void put16(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }

/* A valid-by-construction WNR3 record over a random cut of decoded data, with optional semantic twists whose
 * hashes are still right (so the decoders run to the end and reach the canonical-zero / ceil checks). */
static int make_mini(const WexNr2Model *model, const Source *src, int nsrc, Mini *mini, unsigned char *plain_s[3],
                     unsigned char *plain_w[3]) {
    static unsigned char cs[3][MINI_MAX_BLOCKS], cw[3][MINI_MAX_BLOCKS * 16u];
    unsigned char *scales[3], *weights[3], anchor[3];
    const uint32_t kind = rnd_below(20);
    uint32_t m, i, flags, cz = 1, wchunk, schunk;
    size_t written;
    mini->expect = 0;
    for (m = 0; m < 3u; m++) {
        const Source *s = &src[rnd_below((uint32_t)nsrc)];
        const uint32_t sm = rnd_below(3), have = s->layout.scale_count[sm];
        const uint32_t cap = have - 1u < MINI_MAX_BLOCKS ? have - 1u : MINI_MAX_BLOCKS;
        const uint32_t blocks = rnd_below(10) && cap > 40u ? 40u + rnd_below(cap - 40u) : 1u + rnd_below(cap < 40u ? cap : 40u);
        const uint32_t at = rnd_below(have - blocks);
        mini->layout.scale_count[m] = blocks; mini->layout.symbol_count[m] = blocks * 32u;
        memcpy(cs[m], s->scales[sm] + at, blocks);
        memcpy(cw[m], s->weights[sm] + (size_t)at * 16u, (size_t)blocks * 16u);
        scales[m] = cs[m]; weights[m] = cw[m];
        if (kind == 10 || kind == 11)   /* noise: every context, long streams */
            for (i = 0; i < blocks * 16u; i++) {
                uint32_t v = (uint32_t)rnd() & 0xffu;
                if ((v & 15u) == 8u) v &= 0xf0u;
                if ((v >> 4) == 8u) v &= 0x0fu;
                cw[m][i] = (unsigned char)v;
                if (!(i % 16u)) cs[m][i / 16u] = (unsigned char)(kind == 10 ? 118u + rnd_below(12) : rnd());
            }
        if (kind == 12) memset(cw[m], rnd_below(2) ? 0 : 0x11, (size_t)blocks * 16u);   /* near-zero entropy */
        if (kind == 13 || kind == 14)   /* -0 nibbles: legal without the CZ flag (13), a CZ violation with it (14) */
            for (i = 0; i < 1u + rnd_below(4); i++) {
                const uint32_t at8 = rnd_below(blocks * 32u);
                cw[m][at8 / 2u] = (unsigned char)((cw[m][at8 / 2u] & (at8 & 1u ? 0x0fu : 0xf0u)) | (at8 & 1u ? 0x80u : 0x08u));
            }
    }
    if (kind == 13) cz = 0;
    if (kind == 14) mini->expect = 1;
    for (m = 0; m < 3u; m++) {
        memcpy(plain_s[m], cs[m], mini->layout.scale_count[m]);
        memcpy(plain_w[m], cw[m], mini->layout.symbol_count[m] / 2u);
    }
    flags = wex_nr3_ceil_forward(&mini->layout, scales, weights, NULL) | (cz ? WEX_NR3_FLAG_CZ : 0u);
    if (kind == 15) {   /* ceil underflow: a coded block of maximum 5 with coded scale 0 in a ceil matrix */
        const uint32_t block = rnd_below(mini->layout.scale_count[m = rnd_below(3)]);
        for (i = 0; i < 16u; i++) cw[m][block * 16u + i] = (unsigned char)(rnd_below(6) | rnd_below(6) << 4);
        cw[m][block * 16u + rnd_below(16)] = 0x15;
        cs[m][block] = 0;
        flags |= 1u << m;
        mini->expect = 1;
    }
    if (kind == 16) { flags ^= 1u + rnd_below(7); mini->expect = 2; }   /* ceil flags that do not match the data */
    for (m = 0; m < 3u; m++) anchor[m] = kind == 17 ? (unsigned char)rnd() : wex_nr3_anchor(scales[m], mini->layout.scale_count[m]);
    wchunk = 512u << rnd_below(rnd_below(8) ? 3 : 6);
    schunk = 256u << rnd_below(rnd_below(8) ? 3 : 7);
    if (wex_nrx_encode(model, &mini->layout, (const unsigned char *const *)scales, (const unsigned char *const *)weights,
                       wchunk, schunk, anchor, WEX_NR3_MAGIC, flags, (uint32_t)rnd(), mini->record, MINI_CAP - 4096u, &written)) return -1;
    mini->total = (written + 4095u) & ~(size_t)4095u;
    memset(mini->record + written, 0, mini->total - written);
    return 0;
}

/* Byte-level damage to a valid record; the parser decides whether the decoders get to see it. */
static void mutate(Mini *x) {
    WexNr2Record v;
    unsigned char *r = x->record, *lengths = r + WEX_NR2_HEADER_BYTES;
    uint32_t streams, used, meta, s, k, a, b, n;
    if (wex_nr3_parse(r, x->total, &x->layout, 0, &v)) return;
    streams = v.scale_streams + v.weight_streams; meta = v.metadata_bytes;
    used = meta + v.scale_payload_bytes + v.weight_payload_bytes;
    switch (rnd_below(16)) {
    case 0: r[rnd_below(used)] ^= (unsigned char)(1u << rnd_below(8)); break;
    case 1: r[rnd_below(used)] = (unsigned char)rnd(); break;
    case 2: r[4u + rnd_below(28)] ^= (unsigned char)(1u << rnd_below(8)); break;
    case 3: r[WEX_NR2_HEADER_BYTES + rnd_below(2u * streams)] ^= (unsigned char)(1u << rnd_below(8)); break;
    case 4:   /* move the boundary between two neighbouring streams */
        if (streams < 2u) break;
        s = rnd_below(streams - 1u); k = 1u + rnd_below(8);
        a = wex_nr_u16(lengths + 2u * s); b = wex_nr_u16(lengths + 2u * s + 2u);
        if (rnd_below(2)) { if (a < 4u + k || b + k > 65535u) break; a -= k; b += k; }
        else { if (b < 4u + k || a + k > 65535u) break; a += k; b -= k; }
        if (s + 1u == v.scale_streams) {   /* the boundary between the two payloads moves too */
            put32(r + 8u, v.scale_payload_bytes + a - wex_nr_u16(lengths + 2u * s));
            put32(r + 12u, v.weight_payload_bytes + b - wex_nr_u16(lengths + 2u * s + 2u));
        }
        put16(lengths + 2u * s, a); put16(lengths + 2u * s + 2u, b);
        break;
    case 5:   /* truncate the last stream */
        a = wex_nr_u16(lengths + 2u * (streams - 1u)); k = 1u + rnd_below(6);
        if (a < 4u + k) break;
        put16(lengths + 2u * (streams - 1u), a - k); put32(r + 12u, v.weight_payload_bytes - k);
        memset(r + used - k, 0, k);
        x->total = ((size_t)used - k + 4095u) & ~(size_t)4095u;
        break;
    case 6:   /* extend the last stream into the zero padding */
        a = wex_nr_u16(lengths + 2u * (streams - 1u)); k = 1u + rnd_below(6);
        if (a + k > 65535u || used + k > x->total) break;
        put16(lengths + 2u * (streams - 1u), a + k); put32(r + 12u, v.weight_payload_bytes + k);
        break;
    case 7: {   /* swap two neighbouring streams (bytes and lengths) */
        static unsigned char tmp[65536];
        uint32_t at = meta;
        if (streams < 2u) break;
        s = rnd_below(streams - 1u);
        if (s + 1u == v.scale_streams) break;
        for (k = 0; k < s; k++) at += wex_nr_u16(lengths + 2u * k);
        a = wex_nr_u16(lengths + 2u * s); b = wex_nr_u16(lengths + 2u * s + 2u);
        memcpy(tmp, r + at, a); memmove(r + at, r + at + a, b); memcpy(r + at + b, tmp, a);
        put16(lengths + 2u * s, b); put16(lengths + 2u * s + 2u, a);
        break;
    }
    case 8:   /* header fields the parser accepts: anchors, flags, owner */
        k = rnd_below(4);
        if (k < 3u) r[24u + k] = (unsigned char)rnd();
        else if (rnd_below(2)) r[27] ^= (unsigned char)(1u << rnd_below(4));
        else put32(r + 28u, (uint32_t)rnd());
        break;
    case 9: {   /* a stream's initial state */
        uint32_t at = meta;
        s = rnd_below(streams);
        for (k = 0; k < s; k++) at += wex_nr_u16(lengths + 2u * k);
        if (rnd_below(2)) put32(r + at, (uint32_t)rnd());
        else r[at] = (unsigned char)(rnd_below(2) ? 0x80u | rnd() : 0);   /* state >= 2^31 or < 2^24 */
        break;
    }
    case 10:   /* a run of bytes in the payload */
        n = 1u + rnd_below(64); a = meta + rnd_below(used - meta);
        k = rnd_below(3);
        for (b = a; b < used && b < a + n; b++) r[b] = (unsigned char)(k == 0 ? 0 : k == 1 ? 0xff : rnd());
        break;
    case 11: for (n = 2u + rnd_below(4); n; n--) r[meta + rnd_below(used - meta)] ^= (unsigned char)(1u << rnd_below(8)); break;
    case 12: {   /* stray bytes appended to one stream: it decodes to the right state, only "pointer at end" fails */
        uint32_t at = meta;
        s = rnd_below(streams); k = 1u + rnd_below(3);
        a = wex_nr_u16(lengths + 2u * s);
        if (a + k > 65535u || (((size_t)used + k + 4095u) & ~(size_t)4095u) > MINI_CAP) break;
        for (b = 0; b <= s; b++) at += wex_nr_u16(lengths + 2u * b);
        memmove(r + at + k, r + at, used - at);
        for (b = 0; b < k; b++) r[at + b] = (unsigned char)(rnd_below(2) ? 0 : rnd());
        put16(lengths + 2u * s, a + k);
        if (s < v.scale_streams) put32(r + 8u, v.scale_payload_bytes + k); else put32(r + 12u, v.weight_payload_bytes + k);
        used += k;
        x->total = ((size_t)used + 4095u) & ~(size_t)4095u;
        memset(r + used, 0, x->total - used);
        break;
    }
    case 13: case 14: {   /* drop the last byte of a stream whose successor starts with the same byte: the reference
                             runs out of bytes; a decoder that reads ahead finds the right byte in the wrong stream */
        uint32_t at = meta, pick = rnd_below(streams), found = 0, found_at = 0;
        for (s = 0; s + 1u < streams; s++) {
            a = wex_nr_u16(lengths + 2u * s);
            at += a;
            if (a > 4u && r[at - 1u] == r[at] && (!found_at || s <= pick)) { found = s; found_at = at; }
        }
        if (!found_at) break;
        memmove(r + found_at - 1u, r + found_at, used - found_at);
        put16(lengths + 2u * found, wex_nr_u16(lengths + 2u * found) - 1u);
        if (found < v.scale_streams) put32(r + 8u, v.scale_payload_bytes - 1u); else put32(r + 12u, v.weight_payload_bytes - 1u);
        used--;
        r[used] = 0;
        x->total = ((size_t)used + 4095u) & ~(size_t)4095u;
        break;
    }
    default:   /* the last bytes of the record: the checked path and the hard end */
        r[used - 1u - rnd_below(used - meta < 80u ? used - meta : 80u)] ^= (unsigned char)(1u << rnd_below(8));
    }
}

static int run_mutate(const char *what, const WexNr2Model *model, const Source *src, int nsrc, long wanted) {
    enum { POOL = 32 };
    static Mini pool[POOL], work;
    WexFastModel *tables = tables_for(model);
    WexNr2Layout max;
    Out ref, fast;
    Guard rec = guard_alloc(MINI_CAP), plain[6];
    unsigned char *plain_s[3], *plain_w[3];
    long reached = 0, parse_rejected = 0, accepted = 0, rejected = 0, made = 0, runs = 0, kinds[3] = {0, 0, 0};
    long mutated = 0, mutated_rejected = 0, unchanged = 0;
    int pool_used = 0;
    for (unsigned m = 0; m < 3u; m++) {
        max.scale_count[m] = MINI_MAX_BLOCKS; max.symbol_count[m] = MINI_MAX_BLOCKS * 32u;
        plain[m] = guard_alloc(MINI_MAX_BLOCKS); plain[3 + m] = guard_alloc(MINI_MAX_BLOCKS * 16u);
        plain_s[m] = plain[m].lo; plain_w[m] = plain[3 + m].lo;
    }
    out_alloc(&ref, &max); out_alloc(&fast, &max);
    while (reached < wanted) {
        WexNr2Record view;
        int status, fresh = pool_used < POOL || !rnd_below(4);
        if (fresh) {
            if (make_mini(model, src, nsrc, &work, plain_s, plain_w)) continue;
            made++;
        } else {
            const Mini *base = &pool[rnd_below((uint32_t)pool_used)];
            work = *base;
            mutate(&work);
            if (work.total == base->total && !memcmp(work.record, base->record, work.total)) { unchanged++; continue; }
        }
        if (wex_nr3_parse(work.record, work.total, &work.layout, model->id, &view)) {
            if (fresh) { printf("%s: a constructed record did not parse\n", what); return 1; }
            parse_rejected++;
            continue;
        }
        reached++;
        tighten(&view, work.record, rec);
        out_place(&ref, &work.layout, (int)(reached & 1));
        status = wex_nr2_decode(&view, &work.layout, model, ref.scales, ref.weights);
        if (status) rejected++; else accepted++;
        if (!fresh) { mutated++; mutated_rejected += status != 0; }
        if (fresh) {
            kinds[work.expect]++;
            if ((work.expect == 0 && status) || (work.expect == 1 && !status)) {
                printf("%s: constructed record: reference status %d, expected %d\n", what, status, work.expect); return 1;
            }
            if (work.expect == 0) {   /* the round trip gives the data back */
                for (unsigned m = 0; m < 3u; m++)
                    if (memcmp(ref.scales[m], plain_s[m], work.layout.scale_count[m]) ||
                        memcmp(ref.weights[m], plain_w[m], work.layout.symbol_count[m] / 2u)) { printf("%s: round trip differs\n", what); return 1; }
                pool[pool_used < POOL ? pool_used++ : (int)rnd_below(POOL)] = work;
            }
        }
        out_place(&fast, &work.layout, (int)(reached >> 1 & 1));
        if (wex_fast_decode(&view, &work.layout, model, fast.scales, fast.weights) != status ||
            (!status && !out_equal(&ref, &fast, &work.layout))) { printf("%s: MISMATCH at case %ld (wex_fast_decode)\n", what, reached); return 1; }
        for (size_t v = 0; v < VARIANTS; v++) {
            WexFastOptions o = variants[v].o;
            int got;
            o.threads = (reached + (long)v) % 5 ? 1 : 3;
            out_place(&fast, &work.layout, (int)((reached + (long)v) & 1));
            out_poison(&fast, &work.layout);
            got = wex_fast_decode_with(tables, &view, &work.layout, model, fast.scales, fast.weights, &o);
            runs++;
            if (got != status || (!status && !out_equal(&ref, &fast, &work.layout))) {
                printf("%s: MISMATCH at case %ld (%s, %d threads): reference %d, fast %d\n", what, reached, variants[v].name, o.threads, status, got);
                return 1;
            }
        }
    }
    printf("     %s: %ld records at the decoders (%ld accepted, %ld rejected by the reference); %ld byte-mutated (%ld rejected); "
           "%ld constructed (expected accept %ld, reject %ld, either %ld); %ld more stopped at the parser, %ld unchanged dropped; "
           "%ld fast decodes, 0 mismatches\n", what, reached, accepted, rejected, mutated, mutated_rejected, made, kinds[0], kinds[1],
           kinds[2], parse_rejected, unchanged, runs);
    free(tables);
    return 0;
}

/* A decoded record as a mutate source; -0 codes cleared (make_mini wants none in its source). */
static void source_of(const Rec *r, const WexNr2Model *model, Source *s) {
    WexNr2Record view;
    s->layout = r->layout;
    for (unsigned m = 0; m < 3u; m++) {
        s->scales[m] = malloc(r->layout.scale_count[m]); s->weights[m] = malloc(r->layout.symbol_count[m] / 2u);
        if (!s->scales[m] || !s->weights[m]) exit(2);
    }
    if (wex_nr3_parse(r->record, r->stored, &r->layout, model->id, &view) ||
        wex_nr2_decode(&view, &r->layout, model, s->scales, s->weights)) { fprintf(stderr, "source does not decode\n"); exit(2); }
    for (unsigned m = 0; m < 3u; m++)
        for (size_t k = 0; k < r->layout.symbol_count[m] / 2u; k++) {
            unsigned char *b = s->weights[m] + k;
            if ((*b & 15u) == 8u) *b &= 0xf0u;
            if ((*b >> 4) == 8u) *b &= 0x0fu;
        }
}

/* ---- bench ------------------------------------------------------------------------- */
static int run_bench(const WexNr2Model *model, Rec *recs, int n) {
    WexFastModel *tables = tables_for(model);
    uint64_t raw, aligned;
    double t, best;
    int bad = 0, threads_max = 1;
#ifdef _OPENMP
    threads_max = omp_get_max_threads() < 8 ? omp_get_max_threads() : 8;
#endif
    if (wex_entry_raw_sizes(&recs[0].entry, &raw, &aligned)) return 1;
    unsigned char *slab[8];
    for (int i = 0; i < 8; i++) if (!(slab[i] = malloc((size_t)aligned))) return 1;
    printf("%d records of %llu decoded bytes, fast decoder: %s kernel 1, %d streams, fused; best of 3 passes\n", n,
           (unsigned long long)raw, wex_fast_kernels(), WEX_FAST_LANES);
    printf("| decoder | 1 thread, ms/record | %d threads in one record, ms/record | %d records at once, ms/record |\n|---|---:|---:|---:|\n",
           threads_max, threads_max);
    for (int dec = 0; dec < 2; dec++) {
        static const WexFastOptions one = {1, 1, WEX_FAST_LANES, 1};
        WexFastOptions wide = one;
        double col[3];
        wide.threads = threads_max;
        for (int mode = 0; mode < 3; mode++) {
            best = 1e30;
            if (dec == 0 && mode == 1) { col[1] = -1; continue; }   /* the reference has no threads of its own */
            for (int pass = 0; pass < 3; pass++) {
                t = now_ms();
                if (mode < 2)
                    for (int i = 0; i < n; i++) {
                        const size_t at = wex_record_read_offset(&recs[i].entry);
                        memcpy(slab[0] + at, recs[i].record, recs[i].entry.total_bytes);
                        bad |= dec ? wex_fast_decode_record(&recs[i].entry, slab[0], (size_t)aligned, model, tables, mode ? &wide : &one)
                                   : wex_decode_nr2_record(&recs[i].entry, slab[0], (size_t)aligned, model);
                    }
                else {
                    int done = 0;
                    for (int i0 = 0; i0 < n; i0 += threads_max) {
                        const int k = n - i0 < threads_max ? n - i0 : threads_max;
#ifdef _OPENMP
#pragma omp parallel for num_threads(k) schedule(static, 1) reduction(|:bad)
#endif
                        for (int i = 0; i < k; i++) {
                            const Rec *r = &recs[i0 + i];
                            const size_t at = wex_record_read_offset(&r->entry);
                            memcpy(slab[i] + at, r->record, r->entry.total_bytes);
                            bad |= dec ? wex_fast_decode_record(&r->entry, slab[i], (size_t)aligned, model, tables, &one)
                                       : wex_decode_nr2_record(&r->entry, slab[i], (size_t)aligned, model);
                        }
                        done += k;
                    }
                    (void)done;
                }
                t = (now_ms() - t) / n;
                if (t < best) best = t;
            }
            col[mode] = best;
        }
        if (dec == 0) printf("| reference (wex_decode_nr2_record) | %.1f | - | %.1f |\n", col[0], col[2]);
        else printf("| fast (wex_fast_decode_record) | %.1f | %.1f | %.1f |\n", col[0], col[1], col[2]);
    }
    for (int i = 0; i < 8; i++) free(slab[i]);
    free(tables);
    if (bad) printf("a decode failed\n");
    return bad != 0;
}

int main(int argc, char **argv) {
    long mutations = 300;
    const char *container = NULL;
    int bench = 0, real_count = 4, before;
    setbuf(stdout, NULL);
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mutate") && i + 1 < argc) mutations = atol(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) { uint64_t s = strtoull(argv[++i], NULL, 0); rng_state = s ? s : 1u; } /* every nonzero seed is its own stream; 0 is 1 (xorshift needs a nonzero state) */
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!container) container = argv[i];
        else real_count = atoi(argv[i]);
    }
    if (real_count < 1) real_count = 1;
    printf("     fast decoder: kernel 1 on %s here (WEX_FAST_LANES %d)\n", wex_fast_kernels(), WEX_FAST_LANES);

    if (bench) {
        Container c;
        Rec *recs = calloc((size_t)real_count, sizeof *recs);
        char name[32];
        if (!container || container_open(container, &c)) { fprintf(stderr, "cannot open %s as a WNR3 WEX4 container\n", container ? container : "(none)"); return 2; }
        for (int k = 0; k < real_count; k++)
            if (container_record(&c, (uint32_t)k, (uint32_t)real_count, &recs[k], name, sizeof name)) { fprintf(stderr, "record %d\n", k); return 2; }
        return run_bench(c.model, recs, real_count);
    }

    /* synthetic: a model trained on 4 records, records with several chunk sizes, plain and canonical zero */
    static unsigned char raw[4][SRAW];
    unsigned char *blob = malloc(WEX_NR2_MODEL_BYTES), *blob_cz = malloc(WEX_NR2_MODEL_BYTES);
    Guard gm = guard_alloc(WEX_NR2_MODEL_BYTES), gm_cz = guard_alloc(WEX_NR2_MODEL_BYTES);
    WexNr2Model model, model_cz;
    const WexEntry e = synth_entry();
    const unsigned char *train[4];
    Rec synth[6];
    Source src[6];
    if (!blob || !blob_cz) return 2;
    for (int i = 0; i < 4; i++) { synth_fill(raw[i], 20261116u + 104729u * (uint32_t)i, i >= 2); train[i] = raw[i]; }
    wnr3f_train(&e, train, 4, 0, blob, &model);
    wnr3f_train(&e, train, 4, 1, blob_cz, &model_cz);
    memcpy(gm.hi - WEX_NR2_MODEL_BYTES, blob, WEX_NR2_MODEL_BYTES);         /* the models flush against a guard page */
    memcpy(gm_cz.hi - WEX_NR2_MODEL_BYTES, blob_cz, WEX_NR2_MODEL_BYTES);
    if (wex_nr2_model_open(gm.hi - WEX_NR2_MODEL_BYTES, WEX_NR2_MODEL_BYTES, &model) ||
        wex_nr2_model_open(gm_cz.hi - WEX_NR2_MODEL_BYTES, WEX_NR2_MODEL_BYTES, &model_cz)) return 2;
    WexFastModel *t_plain = tables_for(&model), *t_cz = tables_for(&model_cz);

    before = failures;
    {
        static const struct { int raw, cz; uint32_t wchunk, schunk; } plan[6] = {
            {0, 0, 4096, 256}, {1, 0, 512, 256}, {2, 0, 16384, 16384}, {2, 1, 4096, 256}, {3, 1, 1024, 2048}, {0, 1, 2048, 512}};
        for (int i = 0; i < 6; i++) {
            const WexNr2Model *m = plan[i].cz ? &model_cz : &model;
            char what[64];
            snprintf(what, sizeof what, "synthetic %d (%s, chunks %u/%u)", i, plan[i].cz ? "CZ" : "plain", plan[i].wchunk, plan[i].schunk);
            CHECK(!synth_encode(m, raw[plan[i].raw], plan[i].cz, plan[i].wchunk, plan[i].schunk, (uint32_t)i << 16 | 7u, &synth[i]));
            CHECK(!equal_one(what, m, plan[i].cz ? t_cz : t_plain, &synth[i]));
        }
    }
    ok("synthetic WNR3 records (plain and canonical zero, 6 chunk sizes): every variant on 1 and 8 threads, the default "
       "entry point and the slab-level call give the reference's bytes", before);

    before = failures;
    {   /* a record whose model is not the one given, a record changed in one bit: refused by both, slab untouched */
        uint64_t raw_b, aligned;
        static const WexFastOptions def = {1, 1, WEX_FAST_LANES, 1};
        CHECK(!wex_entry_raw_sizes(&synth[0].entry, &raw_b, &aligned));
        unsigned char *a = malloc((size_t)aligned), *b = malloc((size_t)aligned), *keep = malloc((size_t)aligned);
        const size_t at = wex_record_read_offset(&synth[0].entry);
        unsigned refused = 0, wrong = 0, changed = 0;
        uint32_t seed = 4242u;
        CHECK(a && b && keep);
        memset(a, 0x5a, (size_t)aligned);
        memcpy(a + at, synth[0].record, synth[0].entry.total_bytes);
        memcpy(b, a, (size_t)aligned);
        CHECK(wex_fast_decode_record(&synth[0].entry, b, (size_t)aligned, &model_cz, t_cz, &def) == -1 && !memcmp(a, b, (size_t)aligned));
        CHECK(wex_fast_decode_record(&synth[0].entry, b, (size_t)aligned - 1, &model, t_plain, &def) == -1);
        for (unsigned k = 0; k < 512u; k++) {
            size_t bit;
            int ra, rb;
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            bit = k < 128u ? k : seed % (synth[0].entry.total_bytes * 8u);
            memcpy(a + at, synth[0].record, synth[0].entry.total_bytes);
            a[at + bit / 8u] ^= (unsigned char)(1u << (bit % 8u));
            memcpy(b, a, (size_t)aligned); memcpy(keep, a, (size_t)aligned);
            ra = wex_decode_nr2_record(&synth[0].entry, a, (size_t)aligned, &model);
            rb = wex_fast_decode_record(&synth[0].entry, b, (size_t)aligned, &model, t_plain, &def);
            changed++;
            if (ra != rb || memcmp(a, b, (size_t)aligned)) wrong++;
            if (rb) { refused++; if (memcmp(b, keep, (size_t)aligned)) wrong++; }
            memset(a, 0x5a, at); memset(b, 0x5a, at);
        }
        printf("     %u single-bit changes of a record: %u refused by both, %u decoded by both to the same bytes, %u differ\n",
               changed, refused, changed - refused - wrong, wrong);
        CHECK(!wrong && refused > changed / 2u);
        free(a); free(b); free(keep);
    }
    ok("another model, a slab one byte short, 512 single-bit changes (the first 128 bits, then sampled): the same status as the reference, the same "
       "bytes, a refused record leaves the slab as it was", before);

    before = failures;
    {   /* coli_wex_decode_record: the fast decoder where the CPU has its kernel, the reference with
         * COLI_WEX_FAST_DECODE=0; the reference's bytes and verdicts either way */
        uint32_t id = 0;
        uint64_t raw_b, aligned;
        CHECK(!coli_wex_register_model(blob, WEX_NR2_MODEL_BYTES, &id) && id == model.id);
        CHECK(!coli_wex_register_model(blob_cz, WEX_NR2_MODEL_BYTES, &id) && id == model_cz.id);
        CHECK(!wex_entry_raw_sizes(&synth[0].entry, &raw_b, &aligned));
        unsigned char *a = malloc((size_t)aligned), *b = malloc((size_t)aligned);
        CHECK(a && b);
        for (int mode = 0; mode < 3; mode++) {   /* unset, "1", "0" */
            const int fast = mode < 2 && wexf_cpu_simd();
            const uint64_t before_count = coli_wex_fast_decode_count();
            if (mode == 0) unsetenv("COLI_WEX_FAST_DECODE");
            else setenv("COLI_WEX_FAST_DECODE", mode == 1 ? "1" : "0", 1);
            CHECK(!strcmp(coli_wex_cpu_decoder(), fast ? wex_fast_kernels() : "reference"));
            for (int i = 0; i < 6; i++) {
                const WexEntry *x = &synth[i].entry;
                const size_t at = wex_record_read_offset(x);
                memset(a, 0x5a, (size_t)aligned);
                memcpy(a + at, synth[i].record, x->total_bytes);
                memcpy(b, a, (size_t)aligned);
                CHECK(!wex_decode_nr2_record(x, b, (size_t)aligned, i >= 3 ? &model_cz : &model));
                CHECK(!coli_wex_decode_record(x, a, (size_t)aligned) && !memcmp(a, b, (size_t)raw_b));
                memset(a, 0x5a, at);   /* a bit changed: refused, the slab as it was */
                memcpy(a + at, synth[i].record, x->total_bytes);
                a[at + 4096u + (size_t)i * 977u] ^= 0x10u;
                memcpy(b, a, (size_t)aligned);
                CHECK(coli_wex_decode_record(x, a, (size_t)aligned) == -1 && !memcmp(a, b, (size_t)aligned));
            }
            CHECK(coli_wex_fast_decode_count() - before_count == (fast ? 6u : 0u));
        }
        unsetenv("COLI_WEX_FAST_DECODE");
        free(a); free(b);
    }
    ok("coli_wex_decode_record: the fast decoder where the CPU has its kernel (COLI_WEX_FAST_DECODE unset or 1), the "
       "reference with COLI_WEX_FAST_DECODE=0; the reference's bytes, a damaged record refused and the slab as it was",
       before);

    before = failures;
    {   /* coli_wex_decode_record_to: the record in a buffer of its own (flush against a guard page, read only), each of
         * the six parts in a buffer flush against a guard page; the reference's bytes with the variable unset, 1 and 0
         * and with every fixed stream count; a damaged record refused, the record itself untouched */
        uint64_t raw_b, aligned;
        CHECK(!wex_entry_raw_sizes(&synth[0].entry, &raw_b, &aligned));
        size_t most = 0;
        for (int i = 0; i < 6; i++) if (synth[i].entry.total_bytes > most) most = synth[i].entry.total_bytes;
        unsigned char *ref = malloc((size_t)aligned), *keep = malloc(most);
        Guard part[6];
        unsigned char *scales[3], *weights[3];
        CHECK(ref && keep);
        for (int m = 0; m < 3; m++) {
            const size_t sb = (size_t)synth[0].entry.scale_shapes[m][0] * synth[0].entry.scale_shapes[m][1];
            const size_t wb = (size_t)synth[0].entry.weight_shapes[m][0] * synth[0].entry.weight_shapes[m][1];
            part[m] = guard_alloc(sb); part[3 + m] = guard_alloc(wb);
            scales[m] = part[m].hi - sb; weights[m] = part[3 + m].hi - wb;
        }
        static const char *const lanes[] = {NULL, "1", "2", "3", "4", "6", "8"};
        for (int mode = 0; mode < 3 + 6; mode++) {   /* unset, "1", "0", then COLI_WEX_FAST_LANES=1..8 */
            const int fast = mode != 2 && wexf_cpu_simd();
            const uint64_t before_count = coli_wex_fast_decode_count();
            if (mode == 0 || mode > 2) unsetenv("COLI_WEX_FAST_DECODE");
            else setenv("COLI_WEX_FAST_DECODE", mode == 1 ? "1" : "0", 1);
            if (mode > 2) setenv("COLI_WEX_FAST_LANES", lanes[mode - 2], 1);
            else unsetenv("COLI_WEX_FAST_LANES");
            if (mode > 2) CHECK(coli_wex_fast_lanes() == atoi(lanes[mode - 2]));
            for (int i = 0; i < 6; i++) {
                const WexEntry *x = &synth[i].entry;
                const size_t at = wex_record_read_offset(x);
                Guard g = guard_alloc(x->total_bytes);
                unsigned char *rec = g.hi - x->total_bytes;
                memcpy(rec, synth[i].record, x->total_bytes);
                memset(ref, 0x5a, (size_t)aligned);
                memcpy(ref + at, synth[i].record, x->total_bytes);
                CHECK(!wex_decode_nr2_record(x, ref, (size_t)aligned, i >= 3 ? &model_cz : &model));
                for (int m = 0; m < 3; m++) {
                    memset(scales[m], 0xa5, (size_t)x->scale_shapes[m][0] * x->scale_shapes[m][1]);
                    memset(weights[m], 0xa5, (size_t)x->weight_shapes[m][0] * x->weight_shapes[m][1]);
                }
                CHECK(!coli_wex_decode_record_to(x, rec, scales, weights));
                for (int m = 0; m < 3; m++) {
                    CHECK(!memcmp(scales[m], ref + x->scale_offsets[m], (size_t)x->scale_shapes[m][0] * x->scale_shapes[m][1]));
                    CHECK(!memcmp(weights[m], ref + x->weight_offsets[m], (size_t)x->weight_shapes[m][0] * x->weight_shapes[m][1]));
                }
                CHECK(!memcmp(rec, synth[i].record, x->total_bytes));
                rec[4096u + (size_t)i * 977u] ^= 0x10u;   /* a bit changed: refused, the record as it was given */
                memcpy(keep, rec, x->total_bytes);
                CHECK(coli_wex_decode_record_to(x, rec, scales, weights) == -1 && !memcmp(rec, keep, x->total_bytes));
                guard_free(g);
            }
            CHECK(coli_wex_fast_decode_count() - before_count == (fast ? 6u : 0u));
        }
        unsetenv("COLI_WEX_FAST_DECODE");
        unsetenv("COLI_WEX_FAST_LANES");
        {   /* a NULL part, a RAW entry, a record whose model is not registered: refused */
            WexEntry raw_entry = synth[0].entry;
            unsigned char *none[3] = {scales[0], NULL, scales[2]};
            unsigned char *foreign = malloc(synth[0].entry.total_bytes);
            CHECK(foreign);
            memcpy(foreign, synth[0].record, synth[0].entry.total_bytes);
            foreign[4] ^= 0xffu;   /* the model id at bytes 4..7 */
            raw_entry.flags = 0;
            CHECK(coli_wex_decode_record_to(&synth[0].entry, synth[0].record, none, weights) == -1);
            CHECK(coli_wex_decode_record_to(&raw_entry, synth[0].record, scales, weights) == -1);
            CHECK(coli_wex_decode_record_to(&synth[0].entry, foreign, scales, weights) == -1);
            free(foreign);
        }
        /* the stream count measured on the first records (unset): one of the measured counts once enough were decoded */
        if (wexf_cpu_simd()) {
            const int n = coli_wex_fast_lanes();
            CHECK(n == 3 || n == 4 || n == 6 || n == 8);
            printf("     stream count measured here: %d\n", n);
        }
        for (int k = 0; k < 6; k++) guard_free(part[k]);
        free(ref); free(keep);
    }
    ok("coli_wex_decode_record_to: the record decoded straight into six caller buffers (guarded), the reference's bytes "
       "with COLI_WEX_FAST_DECODE unset, 1 and 0 and every COLI_WEX_FAST_LANES; a damaged record, a missing buffer, a RAW "
       "entry or an unregistered model refused, the record untouched", before);

    before = failures;
    for (int i = 0; i < 6; i++) source_of(&synth[i], i == 3 || i == 4 || i == 5 ? &model_cz : &model, &src[i]);
    {   /* plain sources under the plain model, CZ ones under the CZ model */
        long half = mutations / 2 > 0 ? mutations / 2 : 1;
        CHECK(!run_mutate("synthetic, plain model", &model, src, 3, half));
        CHECK(!run_mutate("synthetic, CZ model", &model_cz, src + 3, 3, mutations - half > 0 ? mutations - half : 1));
    }
    ok("mutation test on synthetic mini-records: same status as the reference, same bytes when both accept", before);

    if (container) {
        Container c;
        Rec *recs = calloc((size_t)real_count, sizeof *recs);
        Source *rs = calloc((size_t)real_count, sizeof *rs);
        WexFastModel *tables;
        char name[32], what[96];
        before = failures;
        if (container_open(container, &c)) { fprintf(stderr, "cannot open %s as a WNR3 WEX4 container\n", container); return 2; }
        tables = tables_for(c.model);
        for (int k = 0; k < real_count; k++) {
            CHECK(!container_record(&c, (uint32_t)k, (uint32_t)real_count, &recs[k], name, sizeof name));
            snprintf(what, sizeof what, "real record %s (%u stored bytes)", name, recs[k].entry.total_bytes);
            CHECK(!equal_one(what, c.model, tables, &recs[k]));
            printf("     %s: equal\n", what);
        }
        ok("real WNR3 records: every variant on 1 and 8 threads, the default entry point and the slab-level call give "
           "the reference's bytes", before);
        before = failures;
        for (int k = 0; k < real_count && k < 16; k++) source_of(&recs[k], c.model, &rs[k]);
        CHECK(!run_mutate("real, the container's model", c.model, rs, real_count < 16 ? real_count : 16, mutations));
        ok("mutation test on mini-records cut from the real records, the container's model", before);
        free(tables);
    }

    free(t_plain); free(t_cz); free(blob); free(blob_cz);
    if (failures) { fprintf(stderr, "test_wex_fast_decode: %d failures\n", failures); return 1; }
    puts("test_wex_fast_decode: ok");
    return 0;
}
