/* Replica-aware range reads (st_read_range_rep): the O_DIRECT window has to
 * return exactly the bytes a plain read would, from any replica, at any file
 * offset, without ever writing past the caller's payload -- and it has to fall
 * back to the buffered path when the caller did not reserve the scratch the
 * aligned window needs.
 *
 * The safety property is the interesting one. The optimized path reads from the
 * block boundary BELOW the payload, so the bytes before `out` are the caller's
 * scratch and the bytes after `out + nbytes` are a wall it must never cross.
 * The guards below are that wall. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../st.h"

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#define RMDIR(p) _rmdir(p)
#else
#include <sys/stat.h>
#include <unistd.h>
#define MKDIR(p) mkdir(p, 0777)
#define RMDIR(p) rmdir(p)
#endif

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define DIR_A "tmp_range_a"   /* primary */
#define DIR_B "tmp_range_b"   /* byte-identical copy, a different "drive" */
#define DIR_C "tmp_range_c"   /* same size, divergent header: must be refused */
#define DIR_E "tmp_range_e"   /* empty: nothing to add */

/* Payload big enough that a read spans several blocks and still ends on a
 * non-block boundary, so the aligned bulk and the buffered tail both happen. */
#define PAYLOAD_FLOATS 16693
#define PAYLOAD_BYTES (PAYLOAD_FLOATS * 4)

static uint8_t *g_truth;

/* One tensor, F32, occupying the whole payload. The header is padded to an
 * 8-byte multiple by the format, which leaves the data at an offset that is
 * 8-aligned but generally NOT block-aligned -- exactly the case the window math
 * exists for, and one the caller cannot choose. */
static int write_model(const char *dir, int corrupt_header) {
    char hdr[160], path[256];
    int hl = snprintf(hdr, sizeof(hdr),
        "{\"t0\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[0,%d]}}",
        PAYLOAD_FLOATS, PAYLOAD_BYTES);
    if (hl < 0 || hl >= (int)sizeof(hdr)) return -1;
    if (corrupt_header) hdr[hl - 2] = ' ';
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    uint64_t n = (uint64_t)hl;
    if (fwrite(&n, 8, 1, f) != 1) { fclose(f); return -1; }
    if (fwrite(hdr, 1, (size_t)hl, f) != (size_t)hl) { fclose(f); return -1; }
    if (fwrite(g_truth, 1, PAYLOAD_BYTES, f) != PAYLOAD_BYTES) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

static void cleanup(void) {
    const char *dirs[] = { DIR_A, DIR_B, DIR_C, DIR_E };
    for (int i = 0; i < 4; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/model.safetensors", dirs[i]);
        remove(path);
        RMDIR(dirs[i]);
    }
}

/* Read [off, off+n) through `rep` the way an engine does, and hold the result to
 * the truth plus a wall of guard bytes past the payload. */
static int read_and_check(shards *S, const st_tensor *t, int rep, int64_t off, int64_t n,
                          int direct, int reserve_scratch) {
    /* `off` is relative to the tensor's payload; the reader works in file
     * offsets, and the format leaves the payload at an 8-aligned-but-not-
     * block-aligned offset after the header. */
    int64_t file_off = t->off + off;
    size_t slack = reserve_scratch ? ST_DIRECT_ALIGN : 0;
    uint8_t *buf = NULL;
    size_t want = (size_t)n + slack + ST_DIRECT_ALIGN;   /* extra tail for the guards */
    if (posix_memalign((void **)&buf, ST_DIRECT_ALIGN, want) != 0) return 1;
    memset(buf, 0xA5, want);
    /* reserve_scratch=0 hands over a page-aligned destination, so `out - pad` is
     * not aligned and the buffered path is the only correct one. */
    uint8_t *out = reserve_scratch ? buf + ST_DIRECT_ALIGN + (file_off % ST_DIRECT_ALIGN)
                                   : buf + ST_DIRECT_ALIGN;
    int64_t before = out - buf;
    st_read_range_rep(S, S->fds[0], rep, file_off, n, out, n, 1, direct, "range test");
    int bad = memcmp(out, g_truth + off, (size_t)n) != 0;
    for (size_t i = (size_t)before + (size_t)n; i < want; i++)
        if (buf[i] != 0xA5) { bad |= 2; break; }
    if (bad)
        fprintf(stderr, "rep=%d off=%lld n=%lld direct=%d reserve=%d -> %s\n",
                rep, (long long)file_off, (long long)n, direct, reserve_scratch,
                (bad & 1) ? "WRONG BYTES" : "WROTE PAST THE PAYLOAD");
    compat_aligned_free(buf);
    return bad;
}

/* Boundary, sub-block, multi-block and non-block-terminated ranges, plus a
 * window that would run past EOF and must be clamped to it. */
static const struct { int64_t off, len; } g_ranges[] = {
    { 0, 1 }, { 0, 4096 }, { 1, 4095 }, { 7, 8189 },
    { 1000, 10000 }, { 4096, 4096 * 3 + 77 }, { 33333, 20000 },
    { PAYLOAD_BYTES - 1000, 1000 },      /* tail: aligned window crosses EOF */
    { PAYLOAD_BYTES - 4096, 4096 },
    { 65536, 1 },
};

int main(void) {
    cleanup();
    CHECK(MKDIR(DIR_A) == 0 && MKDIR(DIR_B) == 0 && MKDIR(DIR_C) == 0 && MKDIR(DIR_E) == 0);
    g_truth = malloc(PAYLOAD_BYTES);
    CHECK(g_truth != NULL);
    srand(20260914);
    for (int i = 0; i < PAYLOAD_BYTES; i++) g_truth[i] = (uint8_t)(rand() & 0xFF);
    CHECK(write_model(DIR_A, 0) == 0);
    CHECK(write_model(DIR_B, 0) == 0);
    CHECK(write_model(DIR_C, 1) == 0);

    shards S;
    st_init(&S, DIR_A);
    CHECK(S.nfd == 1);
    const st_tensor *t = st_find(&S, "t0");
    CHECK(t != NULL && t->nbytes == PAYLOAD_BYTES);

    const int nranges = (int)(sizeof(g_ranges) / sizeof(g_ranges[0]));
    for (int i = 0; i < nranges; i++)
        for (int direct = 0; direct <= 1; direct++) {
            CHECK(read_and_check(&S, t, 0, g_ranges[i].off, g_ranges[i].len, direct, 1) == 0);
            /* No scratch reserved: the same bytes, through the buffered path. */
            CHECK(read_and_check(&S, t, 0, g_ranges[i].off, g_ranges[i].len, direct, 0) == 0);
        }

    /* A replica answers the same bytes, and one that is not byte-identical is
     * never trusted. */
    CHECK(st_mirror_add(&S, DIR_C) == 0);
    CHECK(st_mirror_add(&S, DIR_E) == 0);
    CHECK(st_mirror_add(&S, DIR_B) == 1);
    CHECK(S.nrep == 1);
    for (int i = 0; i < nranges; i++)
        for (int direct = 0; direct <= 1; direct++)
            CHECK(read_and_check(&S, t, 1, g_ranges[i].off, g_ranges[i].len, direct, 1) == 0);

    /* A replica index that has no copy of this shard is not an error: the read
     * stays on the primary and returns the same bytes. */
    st_mirror_reset(&S);
    CHECK(S.nrep == 0);
    CHECK(read_and_check(&S, t, 1, 1000, 10000, 1, 1) == 0);
    CHECK(read_and_check(&S, t, ST_MAX_MIR - 1, 1000, 10000, 1, 1) == 0);

    st_destroy(&S);
    free(g_truth);
    cleanup();
    printf("test_st_range_rep: ok\n");
    return 0;
}
