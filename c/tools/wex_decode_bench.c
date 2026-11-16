/* wex_decode_bench -- WNR3 decode and record read speed on records of a real WEX4 container.
 *
 *   tools/wex_decode_bench decode FILE [--records N] [--seed S] [--passes P] [--lanes 2,4,6,8] [--threads T]
 *   tools/wex_decode_bench io FILE [--records N] [--seed S] [--passes P] [--threads T]
 *
 * decode: N WNR3 expert records picked by the seed (default 16, seed 1) are read into memory, then decoded
 * one at a time on one thread by the reference decoder (wex_decode_nr2_record), by the fast decoder
 * (wex_fast_decode_record, kernel 1, each stream count of --lanes, fused) and, when COLI_WEX_CUDA_LIB names
 * the CUDA helper, by coli_wex_decode_record (one record a call) and the helper's host batch; then T records
 * at once on T threads (default: the OpenMP team, at most 8) by the reference and the fast decoder. Every
 * decoded record is hashed and compared with the reference's: a difference is a failure (exit 1).
 * io: N records of any codec picked by the seed, read whole: uncached (Windows FILE_FLAG_NO_BUFFERING, Linux
 * O_DIRECT, macOS F_NOCACHE; posix_fadvise DONTNEED first where O_DIRECT is refused), then through the OS cache
 * twice (the second pass is the warm one), on 1 thread and on T threads.
 * Times are per record, the best and the median of P passes (decode default 3, the reference 1; io 1).
 * The last line is one JSON object ("JSON {...}") with every number, for tools/wex_bench.py.
 * The file is opened read-only. Paths are not printed. */
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE   /* O_DIRECT */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define COLI_WEX_DECODE_API static inline
#include "../wex_decode.c"
#include "../wex_fast_decode.h"

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

static void *aligned_alloc_page(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n ? n : 1, 4096);
#else
    void *p = NULL;
    return posix_memalign(&p, 4096, n ? n : 1) ? NULL : p;
#endif
}
static void aligned_free_page(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ---- the file: buffered or uncached positional reads ------------------------------------ */
#ifdef _WIN32
typedef HANDLE BFile;
#define BFILE_BAD INVALID_HANDLE_VALUE
static BFile bfile_open(const char *path, int uncached, const char **how) {
    *how = uncached ? "FILE_FLAG_NO_BUFFERING" : "buffered";
    return CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                       FILE_FLAG_OVERLAPPED | (uncached ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_RANDOM_ACCESS), NULL);
}
static void bfile_close(BFile f) { CloseHandle(f); }
static uint64_t bfile_size(BFile f) { LARGE_INTEGER n; return GetFileSizeEx(f, &n) ? (uint64_t)n.QuadPart : 0; }
static int bfile_read(BFile f, void *buf, size_t n, uint64_t off) {
    unsigned char *p = (unsigned char *)buf;
    while (n) {
        const DWORD want = n > (64u << 20) ? (64u << 20) : (DWORD)n;
        OVERLAPPED ov;
        DWORD got = 0;
        BOOL ok;
        memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)off; ov.OffsetHigh = (DWORD)(off >> 32);
        ov.hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!ov.hEvent) return -1;
        ok = ReadFile(f, p, want, &got, &ov);
        if (!ok && GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(f, &ov, &got, TRUE);
        CloseHandle(ov.hEvent);
        if (!ok || got != want) return -1;
        p += want; off += want; n -= want;
    }
    return 0;
}
#else
typedef int BFile;
#define BFILE_BAD (-1)
static BFile bfile_open(const char *path, int uncached, const char **how) {
    int fd;
    *how = "buffered";
    if (!uncached) return open(path, O_RDONLY);
#if defined(__APPLE__)
    fd = open(path, O_RDONLY);
    if (fd >= 0 && fcntl(fd, F_NOCACHE, 1) == 0) *how = "F_NOCACHE";
    return fd;
#else
#ifdef O_DIRECT
    fd = open(path, O_RDONLY | O_DIRECT);
    if (fd >= 0) { *how = "O_DIRECT"; return fd; }
#endif
    fd = open(path, O_RDONLY);
#ifdef POSIX_FADV_DONTNEED
    if (fd >= 0 && posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0) *how = "posix_fadvise DONTNEED (O_DIRECT refused)";
    else *how = "buffered (O_DIRECT and posix_fadvise refused: not cold)";
#else
    *how = "buffered (no O_DIRECT: not cold)";
#endif
    return fd;
#endif
}
static void bfile_close(BFile f) { close(f); }
static uint64_t bfile_size(BFile f) { struct stat s; return fstat(f, &s) ? 0 : (uint64_t)s.st_size; }
static int bfile_read(BFile f, void *buf, size_t n, uint64_t off) {
    unsigned char *p = (unsigned char *)buf;
    while (n) {
        const ssize_t got = pread(f, p, n, (off_t)off);
        if (got <= 0) return -1;
        p += got; off += (uint64_t)got; n -= (size_t)got;
    }
    return 0;
}
#endif

/* ---- the container ----------------------------------------------------------------------- */
typedef struct { BFile f; Wex4Header h; Wex4View v; unsigned char *index; uint64_t size; } Container;
static int container_open(const char *path, Container *c) {
    const char *how;
    unsigned char *page = (unsigned char *)aligned_alloc_page(WEX4_PAGE);
    memset(c, 0, sizeof *c);
    c->f = bfile_open(path, 0, &how);
    if (c->f == BFILE_BAD || !page) return -1;
    c->size = bfile_size(c->f);
    if (bfile_read(c->f, page, WEX4_PAGE, 0) || wex4_parse_header(page, c->size, &c->h)) return -2;
    const size_t n = (size_t)wex4_align(c->h.index_bytes);
    c->index = (unsigned char *)aligned_alloc_page(n);
    if (!c->index || bfile_read(c->f, c->index, n, c->h.index_offset) || wex4_open(&c->h, c->index, n, &c->v)) return -3;
    aligned_free_page(page);
    return 0;
}
/* every WNM2 blob registered (the helper too, when COLI_WEX_CUDA_LIB names it) */
static int container_models(Container *c) {
    int registered = 0;
    for (uint32_t i = 0; i < c->h.blobs; i++) {
        Wex4Blob bl;
        wex4_get_blob(&c->v, i, &bl);
        if (bl.kind != WEX4_BLOB_WNM2) continue;
        const size_t s = (size_t)wex4_align(bl.bytes);
        unsigned char *p = (unsigned char *)aligned_alloc_page(s);
        if (!p || bfile_read(c->f, p, s, bl.offset) || coli_wex4_register_model(p, (size_t)bl.bytes, s, bl.sha, bl.id)) {
            aligned_free_page(p);
            return -1;
        }
        aligned_free_page(p);
        registered++;
    }
    return registered;
}

static uint64_t rng = 1, seed = 1;
static uint64_t rnd(void) { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return rng * UINT64_C(2685821657736338717); }

/* n distinct record indices, picked by the seed among the records `want` accepts (a partial Fisher-Yates) */
static int pick_records(const Container *c, int wnr3_only, int n, uint32_t *out) {
    uint32_t *all = (uint32_t *)malloc((size_t)c->h.records * sizeof *all), count = 0;
    if (!all) return -1;
    for (uint32_t i = 0; i < c->h.records; i++) {
        Wex4Record r;
        wex4_get_record(&c->v, i, &r);
        if (!(r.flags & WEX4_REC_EXPERT)) continue;
        if (wnr3_only && r.codec != WEX4_CODEC_WNR3) continue;
        all[count++] = i;
    }
    if ((uint32_t)n > count) n = (int)count;
    for (int k = 0; k < n; k++) {
        const uint32_t j = (uint32_t)k + (uint32_t)(rnd() % (count - (uint32_t)k));
        const uint32_t t = all[k]; all[k] = all[j]; all[j] = t;
        out[k] = all[k];
    }
    free(all);
    return n;
}

/* a fast 64-bit hash of decoded bytes (identity check only, not a digest) */
static uint64_t hash_bytes(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(0xcbf29ce484222325) ^ n, w;
    size_t i = 0;
    for (; i + 8u <= n; i += 8u) { memcpy(&w, p + i, 8); h = (h ^ w) * UINT64_C(0x100000001b3); h ^= h >> 29; }
    for (; i < n; i++) h = (h ^ p[i]) * UINT64_C(0x100000001b3);
    return h;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
typedef struct { double best, median; int passes; } Timing;
static Timing timing_of(double *per_pass, int passes) {
    Timing t;
    qsort(per_pass, (size_t)passes, sizeof *per_pass, cmp_double);
    t.best = per_pass[0];
    t.median = passes & 1 ? per_pass[passes / 2] : 0.5 * (per_pass[passes / 2 - 1] + per_pass[passes / 2]);
    t.passes = passes;
    return t;
}

/* ---- decode ------------------------------------------------------------------------------- */
typedef struct {
    WexEntry entry;
    const WexNr2Model *model;
    const WexFastModel *tables;
    unsigned char *stored;   /* the record as read, entry.total_bytes */
    uint64_t want;           /* the reference's hash */
    char name[24];
} Rec;

#define MAX_ARMS 24
typedef struct { char name[64]; Timing t; int threads, mismatches; double gbs_decoded; } Arm;
static Arm arms[MAX_ARMS];
static int n_arms;

static void arm_print(const Arm *a) {
    printf("| %-44s | %7d | %9.2f | %9.2f | %8.3f | %s |\n", a->name, a->threads, a->t.best, a->t.median, a->gbs_decoded,
           a->mismatches ? "MISMATCH" : "same bytes");
}

enum { DEC_REF, DEC_FAST, DEC_COLI };
/* one pass, one thread: ms per record; mismatches counted against the reference's hash (reference arm: sets it) */
static double pass_serial(Rec *recs, int n, unsigned char *slab, size_t cap, uint64_t raw, int dec, const WexFastOptions *o,
                          int set_hash, int *mismatches, double *timed_total) {
    double total = 0;
    for (int i = 0; i < n; i++) {
        Rec *r = &recs[i];
        const size_t at = wex_record_read_offset(&r->entry);
        int status;
        memcpy(slab + at, r->stored, r->entry.total_bytes);
        const double t0 = now_ms();
        status = dec == DEC_REF ? wex_decode_nr2_record(&r->entry, slab, cap, r->model)
               : dec == DEC_FAST ? wex_fast_decode_record(&r->entry, slab, cap, r->model, r->tables, o)
               : coli_wex_decode_record(&r->entry, slab, cap);
        total += now_ms() - t0;
        const uint64_t h = status ? 0 : hash_bytes(slab, (size_t)raw);
        if (set_hash) r->want = h;
        if (status || h != r->want) (*mismatches)++;
    }
    if (timed_total) *timed_total = total;
    return total / n;
}

/* one pass, T records at once on T threads (wall time of the pass / n) */
static double pass_parallel(Rec *recs, int n, unsigned char **slabs, size_t cap, uint64_t raw, int dec, const WexFastOptions *o,
                            int threads, int *mismatches) {
    int bad = 0;
    double total = 0;
    for (int i0 = 0; i0 < n; i0 += threads) {
        const int k = n - i0 < threads ? n - i0 : threads;
        for (int i = 0; i < k; i++) {
            const Rec *r = &recs[i0 + i];
            memcpy(slabs[i] + wex_record_read_offset(&r->entry), r->stored, r->entry.total_bytes);
        }
        const double t0 = now_ms();   /* the decodes of one group, timed; the copies and the hashes are not */
#ifdef _OPENMP
#pragma omp parallel for num_threads(k) schedule(static, 1) reduction(+ : bad)
#endif
        for (int i = 0; i < k; i++) {
            Rec *r = &recs[i0 + i];
            const int status = dec == DEC_REF ? wex_decode_nr2_record(&r->entry, slabs[i], cap, r->model)
                             : wex_fast_decode_record(&r->entry, slabs[i], cap, r->model, r->tables, o);
            if (status) bad++;
        }
        total += now_ms() - t0;
        for (int i = 0; i < k; i++)
            if (hash_bytes(slabs[i], (size_t)raw) != recs[i0 + i].want) bad++;
    }
    *mismatches += bad;
    return total / n;
}

static int run_decode(Container *c, int n, int passes, const int *lanes, int n_lanes, int threads) {
    uint32_t *idx = (uint32_t *)malloc((size_t)n * sizeof *idx);
    Rec *recs = (Rec *)calloc((size_t)n, sizeof *recs);
    const char *helper = getenv("COLI_WEX_CUDA_LIB");
    const int want_helper = helper && *helper && strcmp(helper, "0");
    uint64_t raw = 0, aligned = 0, stored_bytes = 0;
    double per_pass[64];
    int failures = 0;
    if (!idx || !recs) return 2;
    uint32_t wnr3 = 0;
    for (uint32_t i = 0; i < c->h.records; i++) {
        Wex4Record r;
        wex4_get_record(&c->v, i, &r);
        wnr3 += r.codec == WEX4_CODEC_WNR3;
    }
    if (!wnr3) {   /* a RAW container: nothing is decoded, the engines read its records as they are */
        printf("decode: no WNR3 records here (a RAW container): nothing to decode\n");
        printf("JSON {\"stage\":\"decode\",\"skipped\":\"no WNR3 records (RAW container)\",\"records\":0,\"arms\":[]}\n");
        return 0;
    }
    if (container_models(c) <= 0) { fprintf(stderr, "the container has no WNM2 model blob that registers: not a WNR3 container?\n"); return 2; }
    n = pick_records(c, 1, n, idx);
    if (n <= 0) { fprintf(stderr, "no WNR3 expert records in this container\n"); return 2; }
    for (int k = 0; k < n; k++) {
        Wex4Record rec;
        uint64_t r_raw, r_al;
        wex4_get_record(&c->v, idx[k], &rec);
        if (wex4_expert_entry(&c->v, idx[k], &recs[k].entry)) { fprintf(stderr, "record %u: no entry\n", idx[k]); return 2; }
        recs[k].stored = (unsigned char *)aligned_alloc_page((size_t)wex4_align(rec.stored));
        if (!recs[k].stored || bfile_read(c->f, recs[k].stored, (size_t)wex4_align(rec.stored), rec.offset)) {
            fprintf(stderr, "record %u: read failed\n", idx[k]); return 2;
        }
        recs[k].model = wex_nr2_find(wex_read_u32_le(recs[k].stored + 4u), NULL);
        if (!recs[k].model) { fprintf(stderr, "record %u names a model the container does not hold\n", idx[k]); return 2; }
        recs[k].tables = wex_fast_models[recs[k].model - wex_nr2_models];
        snprintf(recs[k].name, sizeof recs[k].name, "L%u E%u", rec.owner >> 16, rec.owner & 0xffffu);
        if (wex_entry_raw_sizes(&recs[k].entry, &r_raw, &r_al)) return 2;
        if (r_raw > raw) raw = r_raw;
        if (r_al > aligned) aligned = r_al;
        stored_bytes += recs[k].entry.total_bytes;
    }
    const size_t cap = (size_t)aligned;
    unsigned char *slab = (unsigned char *)malloc(cap);
    if (threads < 1) threads = 1;
    if (threads > 8) threads = 8;
    if (threads > n) threads = n;
    unsigned char *slabs[8];   /* T records at once, and the helper's host batch of up to 8 */
    const int n_slabs = n < 8 ? n : 8;
    for (int i = 0; i < n_slabs; i++) if (!(slabs[i] = (unsigned char *)malloc(cap))) return 2;
    if (!slab) return 2;
    const int simd = wexf_cpu_simd();
    printf("decode: %d WNR3 records (seeded), %llu decoded bytes each, stored/decoded %.3f; fast decoder kernel: %s; "
           "WEX_FAST_LANES (build default) %d\n", n, (unsigned long long)raw, (double)stored_bytes / ((double)raw * n),
           wex_fast_kernels(), WEX_FAST_LANES);
    printf("records:");
    for (int k = 0; k < n; k++) printf(" %s", recs[k].name);
    printf("\n| arm | threads | best ms/record | median ms/record | GB/s decoded (best) | bytes |\n|---|---:|---:|---:|---:|---|\n");

    /* the reference first: its hashes are the ones every other arm must give */
    {
        Arm *a = &arms[n_arms++];
        const int ref_passes = passes > 1 ? 1 : passes;
        snprintf(a->name, sizeof a->name, "reference (wex_decode_nr2_record)");
        a->threads = 1;
        for (int p = 0; p < ref_passes; p++) per_pass[p] = pass_serial(recs, n, slab, cap, raw, DEC_REF, NULL, p == 0, &a->mismatches, NULL);
        a->t = timing_of(per_pass, ref_passes);
        a->gbs_decoded = (double)raw / (a->t.best * 1e6);
        arm_print(a);
        if (a->mismatches) { printf("the reference refused %d records: stopping\n", a->mismatches); return 1; }
    }
    /* the fast decoder, each stream count */
    for (int l = 0; l < (simd ? n_lanes : 1) && n_arms < MAX_ARMS - 6; l++) {
        Arm *a = &arms[n_arms++];
        WexFastOptions o = {1, 1, simd ? lanes[l] : WEX_FAST_LANES, 1};
        if (simd) snprintf(a->name, sizeof a->name, "fast %s, %d streams%s", wex_fast_kernels(), lanes[l], lanes[l] == WEX_FAST_LANES ? " (default)" : "");
        else snprintf(a->name, sizeof a->name, "fast, no SIMD kernel here (scalar)");
        a->threads = 1;
        for (int p = 0; p < passes; p++) per_pass[p] = pass_serial(recs, n, slab, cap, raw, DEC_FAST, &o, 0, &a->mismatches, NULL);
        a->t = timing_of(per_pass, passes);
        a->gbs_decoded = (double)raw / (a->t.best * 1e6);
        arm_print(a);
        failures += a->mismatches;
    }
    /* T records at once */
    if (threads > 1) {
        static const WexFastOptions one = {1, 1, WEX_FAST_LANES, 1};
        for (int dec = DEC_REF; dec <= DEC_FAST; dec++) {
            Arm *a = &arms[n_arms++];
            const int pp = dec == DEC_REF ? 1 : passes;
            snprintf(a->name, sizeof a->name, "%s, %d records at once", dec == DEC_REF ? "reference" : "fast (default streams)", threads);
            a->threads = threads;
            for (int p = 0; p < pp; p++) per_pass[p] = pass_parallel(recs, n, slabs, cap, raw, dec, &one, threads, &a->mismatches);
            a->t = timing_of(per_pass, pp);
            a->gbs_decoded = (double)raw / (a->t.best * 1e6);
            arm_print(a);
            failures += a->mismatches;
        }
    }
    /* the CUDA helper */
    unsigned int features = 0;
    if (want_helper) {
        features = coli_wex_cuda_features();
        if (!(features & COLI_WEX_CUDA_NR3)) printf("COLI_WEX_CUDA_LIB is set, but no helper with WNR3 decode opened (features %u)\n", features);
    }
    if (features & COLI_WEX_CUDA_NR3) {
        Arm *a = &arms[n_arms++];
        const uint64_t before = coli_wex_cuda_decode_count();
        double first = 0;
        int first_bad = 0;
        snprintf(a->name, sizeof a->name, "CUDA helper, coli_wex_decode_record");
        a->threads = 1;
        first = pass_serial(recs, n, slab, cap, raw, DEC_COLI, NULL, 0, &first_bad, NULL);   /* allocates its workspaces */
        for (int p = 0; p < passes; p++) per_pass[p] = pass_serial(recs, n, slab, cap, raw, DEC_COLI, NULL, 0, &a->mismatches, NULL);
        a->t = timing_of(per_pass, passes);
        a->gbs_decoded = (double)raw / (a->t.best * 1e6);
        a->mismatches += first_bad;
        arm_print(a);
        const uint64_t on_gpu = coli_wex_cuda_decode_count() - before;
        printf("     the helper decoded %llu of %d calls (first pass %.2f ms/record, with its workspace allocation)\n",
               (unsigned long long)on_gpu, n * (passes + 1), first);
        if (on_gpu != (uint64_t)n * (uint64_t)(passes + 1)) { printf("     some records went back to the CPU\n"); a->mismatches++; }
        failures += a->mismatches;
        if (features & COLI_WEX_CUDA_BATCH_HOST) {
            Arm *b = &arms[n_arms++];
            const int batch = n < 8 ? n : 8;
            const WexEntry *entries[8];
            const unsigned char *records[8];
            int status[8];
            snprintf(b->name, sizeof b->name, "CUDA helper, host batch of %d", batch);
            b->threads = 1;
            for (int p = 0; p < passes + 1; p++) {   /* pass 0 is a warm-up */
                double total = 0;
                for (int i0 = 0; i0 < n; i0 += batch) {
                    const int k = n - i0 < batch ? n - i0 : batch;
                    for (int i = 0; i < k; i++) { entries[i] = &recs[i0 + i].entry; records[i] = recs[i0 + i].stored; }
                    const double t0 = now_ms();
                    const int bad = coli_wex_cuda_decode_batch_to_host(k, entries, records, 0, slabs, status);
                    total += now_ms() - t0;
                    for (int i = 0; i < k; i++)
                        if (bad || status[i] || hash_bytes(slabs[i], (size_t)raw) != recs[i0 + i].want) b->mismatches += p > 0;
                }
                if (p > 0) per_pass[p - 1] = total / n;
            }
            b->t = timing_of(per_pass, passes);
            b->gbs_decoded = (double)raw / (b->t.best * 1e6);
            arm_print(b);
            failures += b->mismatches;
        }
    }
    printf("%s\n", failures ? "FAILED: some arm did not give the reference's bytes" : "every arm gave the reference's bytes");
    printf("JSON {\"stage\":\"decode\",\"records\":%d,\"seed\":%llu,\"decoded_bytes\":%llu,\"stored_bytes\":%llu,"
           "\"kernel\":\"%s\",\"build_lanes\":%d,\"helper_features\":%u,\"identical\":%s,\"arms\":[",
           n, (unsigned long long)seed, (unsigned long long)raw, (unsigned long long)stored_bytes, wex_fast_kernels(), WEX_FAST_LANES,
           features, failures ? "false" : "true");
    for (int i = 0; i < n_arms; i++)
        printf("%s{\"name\":\"%s\",\"threads\":%d,\"best_ms\":%.3f,\"median_ms\":%.3f,\"passes\":%d,\"gbs\":%.4f,\"mismatches\":%d}",
               i ? "," : "", arms[i].name, arms[i].threads, arms[i].t.best, arms[i].t.median, arms[i].t.passes, arms[i].gbs_decoded,
               arms[i].mismatches);
    printf("],\"records_picked\":[");
    for (int k = 0; k < n; k++) printf("%s\"%s\"", k ? "," : "", recs[k].name);
    printf("]}\n");
    return failures ? 1 : 0;
}

/* ---- io ---------------------------------------------------------------------------------- */
typedef struct { uint64_t offset, bytes; } Span;
static double io_pass(BFile f, const Span *s, int n, unsigned char **bufs, int threads, int *bad) {
    int fail = 0;
    const double t0 = now_ms();
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) reduction(+ : fail)
#endif
    for (int i = 0; i < n; i++) {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        fail += bfile_read(f, bufs[t], (size_t)s[i].bytes, s[i].offset) != 0;
    }
    *bad += fail;
    return now_ms() - t0;
}

static int run_io(const char *path, Container *c, int n, int passes, int threads) {
    uint32_t *idx = (uint32_t *)malloc((size_t)n * sizeof *idx);
    Span *s = (Span *)malloc((size_t)n * sizeof *s);
    uint64_t total = 0, largest = 0;
    unsigned char *bufs[8];
    const char *how_cold, *how_warm;
    int bad = 0;
    double per_pass[64];
    if (!idx || !s) return 2;
    n = pick_records(c, 0, n, idx);
    if (n <= 0) { fprintf(stderr, "no expert records\n"); return 2; }
    int raw_records = 0;
    for (int k = 0; k < n; k++) {
        Wex4Record r;
        wex4_get_record(&c->v, idx[k], &r);
        s[k].offset = r.offset; s[k].bytes = wex4_align(r.stored);
        total += s[k].bytes;
        if (s[k].bytes > largest) largest = s[k].bytes;
        raw_records += r.codec != WEX4_CODEC_WNR3;
    }
    if (threads < 1) threads = 1;
    if (threads > 8) threads = 8;
    for (int i = 0; i < threads; i++) if (!(bufs[i] = (unsigned char *)aligned_alloc_page((size_t)largest))) return 2;
    printf("io: %d records (seeded; %d RAW, %d WNR3), %.1f MB, read whole, one record a request\n", n, raw_records, n - raw_records,
           total / 1e6);
    printf("| read | threads | best GB/s | median GB/s | ms/record (best) | how |\n|---|---:|---:|---:|---:|---|\n");
    struct { const char *name; int uncached, threads; double best, median, ms; const char *how; } row[4];
    int rows = 0;
    for (int t = 1; t <= threads; t = t == 1 && threads > 1 ? threads : t + threads) {
        /* uncached */
        BFile f = bfile_open(path, 1, &how_cold);
        if (f == BFILE_BAD) { fprintf(stderr, "cannot open the file uncached\n"); return 2; }
        for (int p = 0; p < passes; p++) {
#if !defined(_WIN32) && !defined(__APPLE__) && defined(POSIX_FADV_DONTNEED)
            posix_fadvise(f, 0, 0, POSIX_FADV_DONTNEED);
#endif
            per_pass[p] = io_pass(f, s, n, bufs, t, &bad);
        }
        bfile_close(f);
        Timing tc = timing_of(per_pass, passes);
        row[rows].name = "uncached"; row[rows].uncached = 1; row[rows].threads = t; row[rows].how = how_cold;
        row[rows].best = total / (tc.best * 1e6); row[rows].median = total / (tc.median * 1e6); row[rows].ms = tc.best / n;
        printf("| uncached | %d | %.3f | %.3f | %.2f | %s |\n", t, row[rows].best, row[rows].median, row[rows].ms, how_cold);
        rows++;
        /* through the cache: a first pass loads it, the timed passes are warm */
        f = bfile_open(path, 0, &how_warm);
        if (f == BFILE_BAD) return 2;
        io_pass(f, s, n, bufs, t, &bad);
        for (int p = 0; p < passes; p++) per_pass[p] = io_pass(f, s, n, bufs, t, &bad);
        bfile_close(f);
        Timing tw = timing_of(per_pass, passes);
        row[rows].name = "warm"; row[rows].uncached = 0; row[rows].threads = t; row[rows].how = "OS cache, after one pass";
        row[rows].best = total / (tw.best * 1e6); row[rows].median = total / (tw.median * 1e6); row[rows].ms = tw.best / n;
        printf("| warm | %d | %.3f | %.3f | %.2f | %s |\n", t, row[rows].best, row[rows].median, row[rows].ms, row[rows].how);
        rows++;
        if (threads == 1) break;
    }
    if (bad) printf("FAILED: %d reads failed\n", bad);
    printf("JSON {\"stage\":\"io\",\"records\":%d,\"raw_records\":%d,\"bytes\":%llu,\"ok\":%s,\"rows\":[", n, raw_records,
           (unsigned long long)total, bad ? "false" : "true");
    for (int i = 0; i < rows; i++)
        printf("%s{\"read\":\"%s\",\"threads\":%d,\"best_gbs\":%.4f,\"median_gbs\":%.4f,\"ms_per_record\":%.3f,\"how\":\"%s\"}",
               i ? "," : "", row[i].name, row[i].threads, row[i].best, row[i].median, row[i].ms, row[i].how);
    printf("]}\n");
    return bad ? 1 : 0;
}

static void usage(const char *argv0) {
    fprintf(stderr, "usage: %s decode|io FILE [--records N] [--seed S] [--passes P] [--lanes 2,4,6,8] [--threads T]\n", argv0);
}

int main(int argc, char **argv) {
    int n = 16, passes = -1, threads = 0, lanes[8] = {2, 4, 6, 8}, n_lanes = 4;
    Container c;
    setbuf(stdout, NULL);
    if (argc < 3 || (strcmp(argv[1], "decode") && strcmp(argv[1], "io"))) { usage(argv[0]); return 2; }
    const int io = !strcmp(argv[1], "io");
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--records") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) { seed = strtoull(argv[++i], NULL, 0); rng = seed ? seed : 1u; }
        else if (!strcmp(argv[i], "--passes") && i + 1 < argc) passes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lanes") && i + 1 < argc) {
            char *p = argv[++i];
            n_lanes = 0;
            while (*p && n_lanes < 8) {
                const int l = (int)strtol(p, &p, 10);
                if (l == 1 || l == 2 || l == 3 || l == 4 || l == 6 || l == 8) lanes[n_lanes++] = l;
                else { fprintf(stderr, "--lanes: each of 1, 2, 3, 4, 6, 8\n"); return 2; }
                if (*p == ',') p++;
            }
        } else { usage(argv[0]); return 2; }
    }
    if (n < 1) n = 1;
    if (passes < 1) passes = io ? 1 : 3;
    if (passes > 64) passes = 64;
    if (threads < 1) {
#ifdef _OPENMP
        threads = omp_get_max_threads();
#else
        threads = 1;
#endif
    }
    if (container_open(argv[2], &c)) { fprintf(stderr, "the file does not open as WEX4\n"); return 2; }
    printf("container: WEX4 %u.%u, %u records, grid %u x %u, %.2f GB; root ", c.h.major, c.h.minor, c.h.records, c.h.layers,
           c.h.experts, c.size / 1e9);
    for (int i = 0; i < 32; i++) printf("%02x", c.h.root[i]);
    printf(" (header page hash; tools/wex4_verify checks the records)\n");
    return io ? run_io(argv[2], &c, n, passes, threads) : run_decode(&c, n, passes, lanes, n_lanes, threads);
}
