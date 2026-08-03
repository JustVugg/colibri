/* test_ssd_probe.c -- #379/#386 storage-probe hardening contract.
 *
 * Pins the probe's decision layer, which is pure C by design (colibri.c):
 *   - coli_ssd_cache_parse(): the STRICT .coli_ssd grammar, driven by the
 *     shared vector file tests/fixtures/ssd_cache_vectors.txt so the C reader
 *     and the Python reader (resource_plan.parse_ssd_cache, pinned by
 *     test_resource_plan.py over the SAME vectors) cannot drift apart.
 *   - coli_ssd_cache_format() -> parse round-trip: what the writer emits, the
 *     reader trusts.
 *   - coli_ssd_probe_cold_tiles() + coli_ssd_probe_vetoed(): cold-range
 *     steering and the contamination veto over injected residency vectors --
 *     all-cold, all-warm, mixed, and exactly-at-floor, no mincore, no files.
 * On __APPLE__ it additionally exercises coli_ssd_probe_cached() end-to-end
 * against a scratch dir: v2-with-matching-st_dev is trusted without touching
 * any shard; mismatched st_dev, legacy and garbage caches all fall through to
 * a re-probe; and a freshly written (page-cache-warm) shard is VETOED -- no
 * cache file is written and no bandwidth number escapes.
 * Portable: the pure layer builds and runs on every platform, same as
 * test_cap_precedence.c. */
#define COLI_SSD_PROBE_TEST 1

#ifdef __APPLE__
/* pread recorder (#386 r2, FR2-2): pins that probe_raw actually READS the
 * offsets the steering selected -- the decision function alone proved
 * mutable out from under the suite (a probe_raw that ignores tiles[] and
 * reads r11-style random offsets passed every round-1 test). unistd.h is
 * included BEFORE the macro below so its include guard keeps the real
 * prototype; every pread in the engine TU then routes through the recorder,
 * which delegates to the real call. */
#include <unistd.h>
#include <sys/types.h>
static long long g_pread_rec[4096];
static int g_pread_nrec;
static int g_pread_rec_on;
static ssize_t coli_test_pread(int fd, void *buf, size_t n, off_t off) {
    if (g_pread_rec_on && g_pread_nrec < 4096) g_pread_rec[g_pread_nrec++] = (long long)off;
    return pread(fd, buf, n, off);
}
#define pread coli_test_pread
#endif /* __APPLE__ */

#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#ifdef __APPLE__
#undef pread
#endif

static int failures = 0;

#define CHECK(cond, ...)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL ");                                                                                  \
            fprintf(stderr, __VA_ARGS__);                                                                              \
            fprintf(stderr, "\n");                                                                                     \
            failures++;                                                                                                \
        }                                                                                                              \
    } while (0)

/* ---- shared grammar vectors ------------------------------------------- */

/* \n \r \t \0 \s \\ unescape; returns payload length (may contain NULs). */
static size_t unescape(const char *in, char *out, size_t outsz) {
    size_t n = 0;
    for (size_t i = 0; in[i] && n + 1 < outsz; i++) {
        char c = in[i];
        if (c == '\\' && in[i + 1]) {
            i++;
            switch (in[i]) {
            case 'n':
                c = '\n';
                break;
            case 'r':
                c = '\r';
                break;
            case 't':
                c = '\t';
                break;
            case '0':
                c = '\0';
                break;
            case 's':
                c = ' ';
                break;
            case '\\':
                c = '\\';
                break;
            default:
                fprintf(stderr, "FAIL bad escape \\%c in vector file\n", in[i]);
                failures++;
                break;
            }
        }
        out[n++] = c;
    }
    return n;
}

static void run_vectors(void) {
    FILE *f = fopen("tests/fixtures/ssd_cache_vectors.txt", "r");
    if (!f) {
        perror("tests/fixtures/ssd_cache_vectors.txt");
        failures++;
        return;
    }
    char line[512];
    int nvec = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        if (len && line[len - 1] == '\n') line[--len] = 0;
        if (!len || line[0] == '#') continue;
        /* fields are tab-separated; the payload is everything after the last
         * expectation field and may itself contain literal spaces */
        char *kind = line;
        char *f1 = strchr(line, '\t');
        if (f1) *f1++ = 0;
        char payload[256];
        size_t plen = 0;
        double want_gbs = 0;
        unsigned long long want_dev = 0;
        int want_kind = 0;
        if (!strcmp(kind, "garbage")) {
            want_kind = 0;
            plen = f1 ? unescape(f1, payload, sizeof(payload)) : 0;
        } else if (!strcmp(kind, "legacy")) {
            want_kind = 1;
            char *f2 = f1 ? strchr(f1, '\t') : NULL;
            if (!f1 || !f2) {
                CHECK(0, "vector line malformed: %s", kind);
                continue;
            }
            *f2++ = 0;
            want_gbs = strtod(f1, NULL);
            plen = unescape(f2, payload, sizeof(payload));
        } else if (!strcmp(kind, "v2")) {
            want_kind = 2;
            char *f2 = f1 ? strchr(f1, '\t') : NULL;
            char *f3 = f2 ? strchr(f2 + 1, '\t') : NULL;
            if (!f1 || !f2 || !f3) {
                CHECK(0, "vector line malformed: %s", kind);
                continue;
            }
            *f2++ = 0;
            *f3++ = 0;
            want_gbs = strtod(f1, NULL);
            want_dev = strtoull(f2, NULL, 10);
            plen = unescape(f3, payload, sizeof(payload));
        } else {
            CHECK(0, "unknown vector kind: %s", kind);
            continue;
        }
        double gbs = 0;
        unsigned long long dev = 0;
        int got = coli_ssd_cache_parse(payload, plen, &gbs, &dev);
        CHECK(got == want_kind, "vector %d (%s, %zu bytes): kind %d, want %d", nvec, kind, plen, got, want_kind);
        if (got == want_kind && want_kind >= 1)
            CHECK(gbs == want_gbs, "vector %d: gbs %.6f, want %.6f", nvec, gbs, want_gbs);
        if (got == want_kind && want_kind == 2)
            CHECK(dev == want_dev, "vector %d: st_dev %llu, want %llu", nvec, dev, want_dev);
        nvec++;
    }
    fclose(f);
    CHECK(nvec >= 40, "vector file suspiciously short: %d vectors", nvec);
    fprintf(stderr, "grammar vectors: %d checked\n", nvec);
}

/* ---- writer -> reader round-trip --------------------------------------- */

static void run_roundtrip(void) {
    static const double vals[] = {0.001, 0.5, 4.0, 8.894, 9.742, 22.139, 999.999};
    static const unsigned long long devs[] = {0, 1, 16777233, 18446744073709551615ull};
    for (size_t i = 0; i < sizeof(vals) / sizeof(*vals); i++)
        for (size_t j = 0; j < sizeof(devs) / sizeof(*devs); j++) {
            char buf[COLI_SSD_CACHE_MAX + 1];
            int n = coli_ssd_cache_format(buf, sizeof(buf), vals[i], devs[j]);
            CHECK(n > 0, "format(%.3f,%llu) failed", vals[i], devs[j]);
            if (n <= 0) continue;
            double gbs = 0;
            unsigned long long dev = 0;
            int kind = coli_ssd_cache_parse(buf, (size_t)n, &gbs, &dev);
            CHECK(kind == 2, "round-trip '%s': kind %d", buf, kind);
            /* the writer rounds to %.3f; the reader must return exactly that */
            CHECK(kind == 2 && dev == devs[j] && gbs > 0 && gbs < 1000 &&
                      (gbs > vals[i] - 0.0005 && gbs < vals[i] + 0.0005),
                  "round-trip '%s': gbs %.6f dev %llu (want ~%.3f %llu)", buf, gbs, dev, vals[i], devs[j]);
        }
}

/* ---- steering + veto over injected residency vectors ------------------- */

#define PG 16384u                     /* Apple Silicon page size */
#define PER (COLI_SSD_PROBE_BLK / PG) /* 256 pages per 4MB tile */

static void run_steering(void) {
    enum { NT = 20 }; /* 20 tiles = 80 MB */
    size_t npages = (size_t)NT * PER;
    unsigned char *vec = calloc(npages + PER, 1);
    long long tiles[NT + 2];
    long long cold = 0;
    size_t n;

    /* all-cold: every tile qualifies, offsets are the tile starts, no veto */
    n = coli_ssd_probe_cold_tiles(vec, npages, PG, tiles, &cold);
    CHECK(n == NT, "all-cold: %zu tiles, want %d", n, NT);
    CHECK(cold == (long long)NT * COLI_SSD_PROBE_BLK, "all-cold: cold_bytes %lld", cold);
    CHECK(!coli_ssd_probe_vetoed(cold), "all-cold must not veto");
    for (size_t i = 0; i < n; i++)
        CHECK(tiles[i] == (long long)i * COLI_SSD_PROBE_BLK, "all-cold: tile %zu at %lld", i, tiles[i]);

    /* all-warm: nothing qualifies -> veto */
    memset(vec, 1, npages);
    n = coli_ssd_probe_cold_tiles(vec, npages, PG, tiles, &cold);
    CHECK(n == 0 && cold == 0, "all-warm: %zu tiles, cold %lld", n, cold);
    CHECK(coli_ssd_probe_vetoed(cold), "all-warm must veto");

    /* mixed: ONE warm page poisons its whole tile (a 4MB read across it would
     * be part RAM); warm page in tile 0 and tile 5 -> 18 qualifying tiles,
     * none of them at offset 0 or 5*4MB */
    memset(vec, 0, npages);
    vec[3] = 1;
    vec[5 * PER + PER / 2] = 1;
    n = coli_ssd_probe_cold_tiles(vec, npages, PG, tiles, &cold);
    CHECK(n == NT - 2, "mixed: %zu tiles, want %d", n, NT - 2);
    for (size_t i = 0; i < n; i++)
        CHECK(tiles[i] != 0 && tiles[i] != 5ll * COLI_SSD_PROBE_BLK, "mixed: poisoned tile %lld selected", tiles[i]);
    CHECK(!coli_ssd_probe_vetoed(cold), "mixed with 72MB cold must not veto");

    /* exactly at the floor: 16 cold tiles = 64 MB passes, 15 = 60 MB vetoes */
    memset(vec, 1, npages);
    for (size_t t = 0; t < 16; t++) memset(vec + t * PER, 0, PER);
    n = coli_ssd_probe_cold_tiles(vec, npages, PG, tiles, &cold);
    CHECK(n == 16 && cold == COLI_SSD_PROBE_COLD_FLOOR, "floor: %zu tiles, cold %lld", n, cold);
    CHECK(!coli_ssd_probe_vetoed(cold), "exactly-at-floor (64MB) must not veto");
    memset(vec + 15 * PER, 1, PER);
    n = coli_ssd_probe_cold_tiles(vec, npages, PG, tiles, &cold);
    CHECK(n == 15, "floor-1: %zu tiles", n);
    CHECK(coli_ssd_probe_vetoed(cold), "one tile below the floor must veto");

    /* partial tail tile (npages not a multiple of PER) is never selected:
     * a short window cannot supply a full 4MB read */
    memset(vec, 0, npages + PER / 2);
    n = coli_ssd_probe_cold_tiles(vec, npages + PER / 2, PG, tiles, &cold);
    CHECK(n == NT, "partial tail: %zu tiles, want %d", n, NT);

    /* 4K pages (Intel macs): 1024 pages per tile, same tiling */
    n = coli_ssd_probe_cold_tiles(vec, 3 * (COLI_SSD_PROBE_BLK / 4096), 4096, tiles, &cold);
    CHECK(n == 3, "4K pages: %zu tiles, want 3", n);

    free(vec);
}

/* ---- end-to-end cache decisions (darwin: real files, no real model) ----- */
#ifdef __APPLE__
static void write_file(const char *dir, const char *name, const char *data) {
    char p[1400];
    snprintf(p, sizeof(p), "%s/%s", dir, name);
    FILE *f = fopen(p, "wb"); /* exact bytes -- these fixtures feed a strict parser */
    if (!f) {
        perror(p);
        failures++;
        return;
    }
    fputs(data, f);
    fclose(f);
}

static int read_file(const char *dir, const char *name, char *out, size_t outsz) {
    char p[1400];
    snprintf(p, sizeof(p), "%s/%s", dir, name);
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    size_t n = fread(out, 1, outsz - 1, f);
    out[n] = 0;
    fclose(f);
    return (int)n;
}

static void run_cached_decisions(void) {
    const char *tmp = getenv("TMPDIR");
    char dirbuf[1200];
    snprintf(dirbuf, sizeof(dirbuf), "%s/coli_ssd_probe_test.XXXXXX", (tmp && strlen(tmp) < 1100) ? tmp : "/tmp");
    if (!mkdtemp(dirbuf)) {
        perror("mkdtemp");
        failures++;
        return;
    }
    struct stat st;
    if (stat(dirbuf, &st) != 0) {
        failures++;
        return;
    }
    unsigned long long dev = (unsigned long long)st.st_dev;
    char cache[128], back[160];

    /* v2 + matching st_dev: trusted as-is -- no shard is even needed */
    snprintf(cache, sizeof(cache), "v2 7.500 %llu\n", dev);
    write_file(dirbuf, ".coli_ssd", cache);
    double got = coli_ssd_probe_cached(dirbuf);
    CHECK(got == 7.5, "matching v2 cache: got %.3f, want 7.500", got);

    /* v2 from another volume: NOT trusted -> re-probe (no shard here, so the
     * probe cannot run and reports -1); the stale cache file is left in place */
    snprintf(cache, sizeof(cache), "v2 7.500 %llu\n", dev + 1);
    write_file(dirbuf, ".coli_ssd", cache);
    got = coli_ssd_probe_cached(dirbuf);
    CHECK(got == -1, "foreign-volume v2 cache must re-probe: got %.3f", got);

    /* legacy bare number: NOT trusted -> re-probe (the upgrade-to-v2 write
     * happens only when the re-probe succeeds; grammar + round-trip above pin
     * the upgraded format) */
    write_file(dirbuf, ".coli_ssd", "9.500\n");
    got = coli_ssd_probe_cached(dirbuf);
    CHECK(got == -1, "legacy cache must re-probe: got %.3f", got);

    /* garbage: NOT trusted -> re-probe */
    write_file(dirbuf, ".coli_ssd", "inf\n");
    got = coli_ssd_probe_cached(dirbuf);
    CHECK(got == -1, "garbage cache must re-probe: got %.3f", got);

    /* contamination veto, end to end: a freshly WRITTEN 80MB shard is page-
     * cache resident, so the probe must refuse to measure it -- and must NOT
     * write any .coli_ssd */
    {
        char p[1400];
        snprintf(p, sizeof(p), "%s/out-00000.safetensors", dirbuf);
        FILE *f = fopen(p, "wb");
        if (f) {
            char *mb = malloc(1024 * 1024);
            memset(mb, 0xA5, 1024 * 1024);
            for (int i = 0; i < 80; i++) fwrite(mb, 1, 1024 * 1024, f);
            free(mb);
            fclose(f);
            char cp[1400];
            snprintf(cp, sizeof(cp), "%s/.coli_ssd", dirbuf);
            unlink(cp); /* start with no cache at all */
            int veto = COLI_SSD_VETO_NONE;
            double raw = coli_ssd_probe_raw(dirbuf, &veto);
            if (veto == COLI_SSD_VETO_NONE) {
                /* the OS evicted a fresh 80MB write already? environment too
                 * unusual to assert on -- report and move on, the pure veto
                 * tests above still hold the line */
                fprintf(stderr, "note: fresh shard not resident (raw=%.3f), skipping veto e2e\n", raw);
            } else {
                CHECK(raw == -1, "vetoed raw probe must report -1: got %.3f", raw);
                CHECK(veto == COLI_SSD_VETO_RESIDENT, "fully-allocated warm shard must veto as RESIDENT, not %d", veto);
                got = coli_ssd_probe_cached(dirbuf);
                CHECK(got == -1, "vetoed cached probe must report -1: got %.3f", got);
                CHECK(read_file(dirbuf, ".coli_ssd", back, sizeof(back)) == -1,
                      "vetoed probe must not write .coli_ssd (found: %s)", back);
            }
            unlink(p);
        }
    }
    /* scrub the scratch dir */
    char cp[1400];
    snprintf(cp, sizeof(cp), "%s/.coli_ssd", dirbuf);
    unlink(cp);
    rmdir(dirbuf);
}

/* ---- round 2: sparse / small / steering-pin arms ------------------------ */

static char *make_scratch_dir(char *dirbuf, size_t sz) {
    const char *tmp = getenv("TMPDIR");
    snprintf(dirbuf, sz, "%s/coli_ssd_probe_r2.XXXXXX", (tmp && strlen(tmp) < 1100) ? tmp : "/tmp");
    if (!mkdtemp(dirbuf)) {
        perror("mkdtemp");
        failures++;
        return NULL;
    }
    return dirbuf;
}

/* write `mb` MB through F_NOCACHE so the pages land on disk but NOT in the
 * page cache (the confirmation battery's cold-file technique) */
static int write_cold_mb(const char *path, int mb) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    fcntl(fd, F_NOCACHE, 1);
    char *buf = malloc(1u << 20);
    if (!buf) {
        close(fd);
        return -1;
    }
    memset(buf, 0x5A, 1u << 20);
    int ok = 1;
    for (int i = 0; i < mb && ok; i++) ok = write(fd, buf, 1u << 20) == (ssize_t)(1u << 20);
    free(buf);
    close(fd);
    return ok ? 0 : -1;
}

/* fraction of the byte range [lo,hi) currently page-cache resident */
static double resident_share(const char *path, long long lo, long long hi) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -1;
    }
    size_t page_sz = (size_t)getpagesize();
    size_t npages = ((size_t)st.st_size + page_sz - 1) / page_sz;
    unsigned char *vec = malloc(npages);
    void *map = vec ? mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
    double share = -1;
    if (map != MAP_FAILED && mincore(map, (size_t)st.st_size, (char *)vec) == 0) {
        size_t p0 = (size_t)(lo / (long long)page_sz), p1 = (size_t)(hi / (long long)page_sz);
        if (p1 > npages) p1 = npages;
        size_t res = 0;
        for (size_t i = p0; i < p1; i++) res += (size_t)(vec[i] & 1);
        share = p1 > p0 ? (double)res / (double)(p1 - p0) : -1;
    }
    if (map != MAP_FAILED) munmap(map, (size_t)st.st_size);
    free(vec);
    close(fd);
    return share;
}

/* F1: an under-allocated (sparse / still-downloading) shard must be vetoed
 * BEFORE any read: its never-resident pages are holes, and pread of a hole
 * is VFS zero-fill at RAM speed (auditor repro: 73.3 GB/s latched). */
static void run_sparse_veto(void) {
    char dirbuf[1200];
    if (!make_scratch_dir(dirbuf, sizeof(dirbuf))) return;
    char p[1400];
    snprintf(p, sizeof(p), "%s/out-00000.safetensors", dirbuf);
    int fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror(p);
        failures++;
        rmdir(dirbuf);
        return;
    }
    char *mb = malloc(1u << 20);
    memset(mb, 0xA5, 1u << 20);
    for (int i = 0; i < 100; i++) write(fd, mb, 1u << 20); /* 100 MB real data */
    free(mb);
    ftruncate(fd, (off_t)400 * 1024 * 1024); /* + 300 MB of holes */
    close(fd);
    /* Precondition read RAW from stat, deliberately NOT via the gate function
     * under test (a mutated gate must not be able to self-disable this arm):
     * the fixture is 100 MB real of 400 MB, so anything under half allocated
     * proves the filesystem kept the hole. */
    struct stat st;
    long long alloc = stat(p, &st) == 0 ? (long long)st.st_blocks * 512 : -1;
    if (alloc < 0 || alloc * 2 >= (long long)st.st_size) {
        fprintf(stderr, "note: filesystem allocated the ftruncate hole (blocks*512=%lld); skipping sparse e2e\n",
                alloc);
    } else {
        int veto = COLI_SSD_VETO_NONE;
        double raw = coli_ssd_probe_raw(dirbuf, &veto);
        CHECK(raw == -1, "sparse shard must not yield a measurement: got %.3f", raw);
        CHECK(veto == COLI_SSD_VETO_SPARSE, "sparse shard must veto as SPARSE, got %d", veto);
        double got = coli_ssd_probe_cached(dirbuf);
        char back[160];
        CHECK(got == -1, "sparse cached probe must report -1: got %.3f", got);
        CHECK(read_file(dirbuf, ".coli_ssd", back, sizeof(back)) == -1,
              "sparse probe must not write .coli_ssd (found: %s)", back);
    }
    /* Precedence pin: a shard that is BOTH under-allocated and below the
     * window floor reports SPARSE -- the allocation check runs first, and
     * "finish the download" is the actionable advice of the two. */
    {
        int fd2 = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd2 >= 0) {
            char five[5 * 1024 * 1024];
            memset(five, 0x3C, sizeof(five));
            write(fd2, five, sizeof(five));          /* 5 MB real data */
            ftruncate(fd2, (off_t)30 * 1024 * 1024); /* sparse AND small */
            close(fd2);
            long long alloc2 = stat(p, &st) == 0 ? (long long)st.st_blocks * 512 : -1;
            if (alloc2 >= 0 && alloc2 * 2 < (long long)st.st_size) {
                int veto2 = COLI_SSD_VETO_NONE;
                coli_ssd_probe_raw(dirbuf, &veto2);
                CHECK(veto2 == COLI_SSD_VETO_SPARSE, "sparse+small shard must report SPARSE (allocation first), got %d",
                      veto2);
            }
        }
    }
    unlink(p);
    char cp[1400];
    snprintf(cp, sizeof(cp), "%s/.coli_ssd", dirbuf);
    unlink(cp);
    rmdir(dirbuf);
}

/* F4: the picker must return the LARGEST shard, not the readdir-first match
 * -- a stray small shard in the dir would otherwise starve the probe below
 * the 64 MB floor forever while a full-size neighbor sits untouched. Unit
 * half pins the contract on pick_shard directly; e2e half proves the probe
 * then measures where a first-match picker would veto SMALL. */
static void run_largest_shard(void) {
    char dirbuf[1200];
    if (!make_scratch_dir(dirbuf, sizeof(dirbuf))) return;
    char small[1400], large[1400];
    snprintf(small, sizeof(small), "%s/aaa-small.safetensors", dirbuf);
    snprintf(large, sizeof(large), "%s/zzz-large.safetensors", dirbuf);
    if (write_cold_mb(small, 30) != 0 || write_cold_mb(large, 72) != 0) {
        perror("largest-shard fixture");
        failures++;
        unlink(small);
        unlink(large);
        rmdir(dirbuf);
        return;
    }
    char picked[1280];
    coli_ssd_probe_pick_shard(dirbuf, picked, sizeof(picked));
    CHECK(!strcmp(picked, large), "picker must choose the largest shard: got '%s'", picked);
    if (resident_share(large, 0, (long long)72 * 1024 * 1024) < 0.05) {
        int veto = COLI_SSD_VETO_NONE;
        double gbs = coli_ssd_probe_raw(dirbuf, &veto);
        CHECK(veto == COLI_SSD_VETO_NONE, "with a 72MB cold shard present the probe must not veto: got %d", veto);
        CHECK(gbs > 0, "with a 72MB cold shard present the probe must measure: got %.3f", gbs);
    } else {
        fprintf(stderr, "note: large shard not cold, skipping largest-shard e2e half\n");
    }
    unlink(small);
    unlink(large);
    char cp[1400];
    snprintf(cp, sizeof(cp), "%s/.coli_ssd", dirbuf);
    unlink(cp);
    rmdir(dirbuf);
}

/* F4 honesty: a shard that can never offer 64 MB of probe windows is SMALL,
 * not "page-cache resident" -- even when it is stone cold. */
static void run_small_veto(void) {
    char dirbuf[1200];
    if (!make_scratch_dir(dirbuf, sizeof(dirbuf))) return;
    char p[1400];
    snprintf(p, sizeof(p), "%s/out-00000.safetensors", dirbuf);
    if (write_cold_mb(p, 30) != 0) {
        perror(p);
        failures++;
        rmdir(dirbuf);
        return;
    }
    int veto = COLI_SSD_VETO_NONE;
    double raw = coli_ssd_probe_raw(dirbuf, &veto);
    CHECK(raw == -1, "30MB shard must not yield a measurement: got %.3f", raw);
    CHECK(veto == COLI_SSD_VETO_SMALL, "cold 30MB shard must veto as SMALL, got %d", veto);
    unlink(p);
    rmdir(dirbuf);
}

/* FR2-2: the steering must be USED, not merely computed. Half the shard is
 * cold (F_NOCACHE-written), half deliberately warmed; every offset probe_raw
 * actually preads (recorded by the coli_test_pread shim) must fall in the
 * cold half. A probe_raw that ignores tiles[] and reads r11-style random
 * offsets lands ~half its reads in the warm half and fails here. */
static void run_steering_pin(void) {
    enum { MB_TOTAL = 144, MB_COLD = 72 }; /* 36 tiles: 18 cold + 18 warm */
    char dirbuf[1200];
    if (!make_scratch_dir(dirbuf, sizeof(dirbuf))) return;
    char p[1400];
    snprintf(p, sizeof(p), "%s/out-00000.safetensors", dirbuf);
    if (write_cold_mb(p, MB_TOTAL) != 0) {
        perror(p);
        failures++;
        rmdir(dirbuf);
        return;
    }
    long long warm_lo = (long long)MB_COLD * 1024 * 1024;
    long long warm_hi = (long long)MB_TOTAL * 1024 * 1024;
    /* warm the second half with plain buffered reads */
    int fd = open(p, O_RDONLY);
    if (fd >= 0) {
        char *buf = malloc(1u << 20);
        for (long long off = warm_lo; off < warm_hi; off += 1u << 20)
            if (pread(fd, buf, 1u << 20, (off_t)off) <= 0) break;
        free(buf);
        close(fd);
    }
    double cold_res = resident_share(p, 0, warm_lo);
    double warm_res = resident_share(p, warm_lo, warm_hi);
    if (!(cold_res >= 0 && cold_res < 0.05 && warm_res > 0.90)) {
        /* residency preconditions not met on this box: report, don't flake */
        fprintf(stderr, "note: residency prep off (cold %.0f%%, warm %.0f%%), skipping steering pin\n", cold_res * 100,
                warm_res * 100);
    } else {
        g_pread_nrec = 0;
        g_pread_rec_on = 1;
        int veto = COLI_SSD_VETO_NONE;
        double gbs = coli_ssd_probe_raw(dirbuf, &veto);
        g_pread_rec_on = 0;
        CHECK(veto == COLI_SSD_VETO_NONE, "half-cold shard must not veto (72MB cold): got %d", veto);
        CHECK(gbs > 0, "half-cold shard must measure: got %.3f", gbs);
        CHECK(g_pread_nrec > 0, "probe made no preads");
        for (int i = 0; i < g_pread_nrec; i++) {
            CHECK(g_pread_rec[i] >= 0 && g_pread_rec[i] < warm_lo, "pread %d at %lld outside the cold half [0,%lld)", i,
                  g_pread_rec[i], warm_lo);
            CHECK(g_pread_rec[i] % COLI_SSD_PROBE_BLK == 0, "pread %d at %lld is not tile-aligned", i, g_pread_rec[i]);
        }
        fprintf(stderr, "steering pin: %d preads, all in the cold half\n", g_pread_nrec);
    }
    unlink(p);
    char cp[1400];
    snprintf(cp, sizeof(cp), "%s/.coli_ssd", dirbuf);
    unlink(cp);
    rmdir(dirbuf);
}
#endif /* __APPLE__ */

/* F1's decision layer, portable: st_blocks*512 must cover st_size within
 * 12.5% slack; the boundary sits exactly at size - size/8. */
static void run_underallocated(void) {
    const long long GBl = 1024ll * 1024 * 1024;
    CHECK(coli_ssd_probe_underallocated(100ll * 1024 * 1024, 10 * GBl),
          "auditor repro: 100MB real of 10GB must read as under-allocated");
    CHECK(!coli_ssd_probe_underallocated(10 * GBl, 10 * GBl), "fully allocated must pass");
    CHECK(!coli_ssd_probe_underallocated(10 * GBl + 4096, 10 * GBl), "metadata rounding above size must pass");
    long long size = 800ll * 1024 * 1024;
    CHECK(!coli_ssd_probe_underallocated(size - size / 8, size), "exactly at the slack edge must pass");
    CHECK(coli_ssd_probe_underallocated(size - size / 8 - 1, size), "one byte past the slack must veto");
    CHECK(coli_ssd_probe_underallocated(0, size), "zero allocation must veto");
}

int main(void) {
    run_vectors();
    run_roundtrip();
    run_steering();
    run_underallocated();
#ifdef __APPLE__
    run_cached_decisions();
    run_sparse_veto();
    run_small_veto();
    run_largest_shard();
    run_steering_pin();
#endif
    if (failures) {
        fprintf(stderr, "ssd probe tests: %d FAILURE(S)\n", failures);
        return 1;
    }
    puts("ssd probe tests: ok");
    return 0;
}
