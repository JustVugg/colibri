/* test_st_shared_read.c -- a shard's second descriptor (st.h: st_shard_pair, st_shard_enter).
 *
 * The reader that finds a shard free reads through its first descriptor, a reader
 * that finds it taken through the second. In the first half of this test the two
 * descriptors of a pair are deliberately two DIFFERENT files, so the bytes that
 * come back say which descriptor served the read: no counter, no timing. The
 * second half pairs a file with a second descriptor of itself, as st_open_fd does
 * on Windows, and has a team of threads read it at once.
 *
 * On Linux and macOS no shard ever gets a second descriptor (compat_open_overlapped
 * is -1 there, a positional pread already overlaps); the mechanism is the same
 * code and is tested everywhere. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define FSZ (1u << 20)
#define FILE_A "test_st_shared_a.tmp"
#define FILE_B "test_st_shared_b.tmp"
#define SALT_B 0x5a

static uint8_t pat(uint32_t i, uint8_t salt) { return (uint8_t)(((i * 2654435761u) >> 24) ^ salt); }
static int same(const uint8_t *b, uint32_t off, uint32_t n, uint8_t salt) {
    for (uint32_t i = 0; i < n; i++) if (b[i] != pat(off + i, salt)) return 0;
    return 1;
}
static int write_file(const char *path, uint8_t salt) {
    uint8_t *all = (uint8_t *)malloc(FSZ);
    FILE *f = fopen(path, "wb");
    if (!all || !f) return 0;
    for (uint32_t i = 0; i < FSZ; i++) all[i] = pat(i, salt);
    int ok = fwrite(all, 1, FSZ, f) == FSZ;
    fclose(f); free(all);
    return ok;
}

int main(void) {
    CHECK(write_file(FILE_A, 0) && write_file(FILE_B, SALT_B));
    int a = open(FILE_A, COMPAT_O_RDONLY), b = open(FILE_B, COMPAT_O_RDONLY);
    CHECK(a >= 0 && b >= 0);
    uint8_t buf[4096];

    /* no second descriptor: the one there is, and nothing to take */
    CHECK(st_shard_enter(a) == a && st_shard_enter(a) == a);
    st_shard_leave(a, a);
    st_pread_full(a, buf, sizeof buf, 100, "unpaired");
    CHECK(same(buf, 100, sizeof buf, 0));
    CHECK(st_shard_enter(-1) == -1 && st_shard_enter(ST_MAX_PAIRED_FD) == ST_MAX_PAIRED_FD);

    st_shard_pair(a, b);

    /* a lone reader stays on the first descriptor, read after read */
    for (uint32_t i = 0; i < 3; i++) {
        st_pread_full(a, buf, sizeof buf, 4096 * i, "alone");
        CHECK(same(buf, 4096 * i, sizeof buf, 0));
    }

    /* a reader that finds the shard taken goes through the second, and so does the next */
    int via = st_shard_enter(a);
    CHECK(via == a);
    CHECK(st_shard_enter(a) == b);
    st_shard_leave(a, b);                       /* leaving the second descriptor frees nothing */
    CHECK(st_shard_enter(a) == b);
    st_pread_full(a, buf, sizeof buf, 8192, "taken");
    CHECK(same(buf, 8192, sizeof buf, SALT_B));

    /* the shard is free again once its reader leaves, and after every read */
    st_shard_leave(a, via);
    st_pread_full(a, buf, sizeof buf, 8192, "released");
    CHECK(same(buf, 8192, sizeof buf, 0));
    st_pread_full(a, buf, sizeof buf, 12288, "released again");
    CHECK(same(buf, 12288, sizeof buf, 0));

    /* unpaired: taking it no longer sends anyone elsewhere (the second descriptor is closed with the pair) */
    st_shard_pair(a, -1);
    CHECK(st_shard_enter(a) == a && st_shard_enter(a) == a);

    /* a team on a shard paired with itself: every thread gets its own bytes */
    int a2 = compat_open_overlapped(FILE_A);
    if (a2 < 0) a2 = open(FILE_A, COMPAT_O_RDONLY);
    CHECK(a2 >= 0);
    st_shard_pair(a, a2);
    int bad = 0;
    #pragma omp parallel for num_threads(8) schedule(static, 1) reduction(+:bad)
    for (int t = 0; t < 8; t++) {
        uint8_t *tb = (uint8_t *)malloc(1u << 16);
        uint32_t s = 2463534242u * (uint32_t)(t + 1);
        for (int i = 0; tb && i < 400; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            uint32_t len = 1 + (s >> 8) % (1u << 16), off = s % (FSZ - len);
            st_pread_full(a, tb, len, off, "team");
            if (!same(tb, off, len, 0)) bad++;
        }
        if (!tb) bad++;
        free(tb);
    }
    CHECK(bad == 0);
    via = st_shard_enter(a);                    /* the team left the shard free */
    CHECK(via == a);
    st_shard_leave(a, via);

    st_shard_pair(a, -1);
    close(a);
    remove(FILE_A); remove(FILE_B);
    puts("st shared read tests: ok");
    return 0;
}
