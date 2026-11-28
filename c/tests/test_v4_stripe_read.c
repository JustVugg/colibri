/* test_v4_stripe_read.c -- V4_STRIPE split-size, alignment, and striped
 * replica-read reconstruction unit test (CPU-only). */
#include "../deepseek_v4_internal.h"
#include "../compat.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(cond, ...) do {                                              \
    if (!(cond)) {                                                         \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
        fprintf(stderr, __VA_ARGS__);                                      \
        fputc('\n', stderr);                                               \
        failures++;                                                        \
    }                                                                      \
} while (0)

static int write_all(int fd, const void *buf, size_t n) {
    const unsigned char *p = (const unsigned char *)buf;
    size_t done = 0;
    while (done < n) {
        ssize_t w = write(fd, p + done, n - done);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        done += (size_t)w;
    }
    return 0;
}

/* Write a valid single-tensor safetensors file of ~payload_bytes filled with
 * deterministic pseudo-random bytes so both replicas pass st_mirror_add's
 * content verification and any misalignment or swap is caught immediately. */
static int write_replica_file(const char *path, const unsigned char *payload,
                              size_t payload_bytes, uint64_t *data_offset_out) {
    char header[256];
    int hlen = snprintf(
        header, sizeof(header),
        "{\"w\":{\"dtype\":\"U8\",\"shape\":[%zu],\"data_offsets\":[0,%zu]}}",
        payload_bytes, payload_bytes);
    if (hlen <= 0 || (size_t)hlen >= sizeof(header)) return -1;
    uint64_t header_len = (uint64_t)hlen;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | COMPAT_O_BINARY, 0644);
    if (fd < 0) return -1;
    int rc = 0;
    if (write_all(fd, &header_len, sizeof(header_len)) != 0 ||
        write_all(fd, header, (size_t)hlen) != 0 ||
        write_all(fd, payload, payload_bytes) != 0) {
        rc = -1;
    }
    close(fd);
    if (rc == 0 && data_offset_out)
        *data_offset_out = sizeof(header_len) + header_len;
    return rc;
}

static void reset_mir_stats(void) {
    for (int r = 0; r < 1 + ST_MAX_MIR; r++) {
        __atomic_store_n(&g_v4_mir_bytes[r], 0, __ATOMIC_RELAXED);
        __atomic_store_n(&g_v4_mir_nread[r], 0, __ATOMIC_RELAXED);
    }
}

static void test_split_size_and_alignment(void) {
    const size_t align = 4096;
    const size_t one_mib = (size_t)1 << 20;

    /* Below 1 MiB threshold -> disabled (returns 0). */
    CHECK(coli_v4_stripe_head_bytes(0, 128, 128) == 0,
          "0 bytes must not stripe");
    CHECK(coli_v4_stripe_head_bytes(one_mib - align, 128, 128) == 0,
          "below 1 MiB must not stripe");

    /* Unaligned direct_length -> disabled (returns 0). */
    CHECK(coli_v4_stripe_head_bytes(one_mib + 123, 128, 128) == 0,
          "unaligned direct_length must not stripe");

    /* Invalid shares -> disabled (returns 0). */
    CHECK(coli_v4_stripe_head_bytes(2 * one_mib, 0, 128) == 0,
          "zero share_rep must not stripe");
    CHECK(coli_v4_stripe_head_bytes(2 * one_mib, 128, 0) == 0,
          "zero share_other must not stripe");
    CHECK(coli_v4_stripe_head_bytes(2 * one_mib, -10, 128) == 0,
          "negative share_rep must not stripe");

    /* Equal shares on 1 MiB, 2 MiB, 12 MiB, and odd-page 12 MiB + 4 KiB. */
    const size_t lengths[] = {
        one_mib, 2 * one_mib, 12 * one_mib, 12 * one_mib + align
    };
    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) {
        size_t len = lengths[i];
        size_t head = coli_v4_stripe_head_bytes(len, 128, 128);
        size_t tail = len - head;
        CHECK(head >= align && tail >= align,
              "equal split len=%zu: head=%zu tail=%zu both >= 4K", len, head, tail);
        CHECK((head & (align - 1)) == 0 && (tail & (align - 1)) == 0,
              "equal split len=%zu: head=%zu tail=%zu 4K-aligned", len, head, tail);
        CHECK(head + tail == len,
              "equal split len=%zu: sum matches", len);
        size_t diff = head > tail ? head - tail : tail - head;
        CHECK(diff <= align,
              "equal split len=%zu: |head-tail|=%zu <= 4K", len, diff);
    }

    /* Asymmetric 3:1 shares (e.g. Gen5 + Gen4 or NVMe + slower drive). */
    {
        size_t len = 12 * one_mib;
        size_t head = coli_v4_stripe_head_bytes(len, 192, 64);
        size_t tail = len - head;
        CHECK((head & (align - 1)) == 0 && (tail & (align - 1)) == 0,
              "3:1 split must be 4K-aligned (head=%zu tail=%zu)", head, tail);
        CHECK(head == 9 * one_mib && tail == 3 * one_mib,
              "3:1 split of 12 MiB expected 9/3 MiB, got %zu/%zu", head, tail);
    }

    /* Asymmetric 1:3 shares (routed replica slower than mirror). */
    {
        size_t len = 12 * one_mib;
        size_t head = coli_v4_stripe_head_bytes(len, 64, 192);
        size_t tail = len - head;
        CHECK((head & (align - 1)) == 0 && (tail & (align - 1)) == 0,
              "1:3 split must be 4K-aligned (head=%zu tail=%zu)", head, tail);
        CHECK(head == 3 * one_mib && tail == 9 * one_mib,
              "1:3 split of 12 MiB expected 3/9 MiB, got %zu/%zu", head, tail);
    }

    /* Extreme skew where head rounds down below 4 KiB -> safe fallback (0). */
    CHECK(coli_v4_stripe_head_bytes(one_mib, 1, 1000) == 0,
          "sub-page head must return 0");
}

static void test_striped_replica_reconstruction(void) {
    const size_t payload_bytes = 2 * ((size_t)1 << 20) + 12345;
    unsigned char *payload = (unsigned char *)malloc(payload_bytes);
    CHECK(payload != NULL, "malloc payload");
    if (!payload) return;

    uint32_t rng = 0x12345678u;
    for (size_t i = 0; i < payload_bytes; i++) {
        rng = rng * 1664525u + 1013904223u;
        payload[i] = (unsigned char)(rng >> 24);
    }

    char dir0[] = "colibri-v4-stripe0-XXXXXX";
    char dir1[] = "colibri-v4-stripe1-XXXXXX";
    char path0[256] = {0}, path1[256] = {0}, error[256] = {0};
    int have_dir0 = mkdtemp(dir0) != NULL;
    int have_dir1 = mkdtemp(dir1) != NULL;
    CHECK(have_dir0 && have_dir1, "mkdtemp replica dirs");
    if (!have_dir0 || !have_dir1) {
        if (have_dir0) rmdir(dir0);
        if (have_dir1) rmdir(dir1);
        free(payload);
        return;
    }

    snprintf(path0, sizeof(path0), "%s/model.safetensors", dir0);
    snprintf(path1, sizeof(path1), "%s/model.safetensors", dir1);
    uint64_t data_off0 = 0, data_off1 = 0;
    CHECK(write_replica_file(path0, payload, payload_bytes, &data_off0) == 0,
          "write replica 0");
    CHECK(write_replica_file(path1, payload, payload_bytes, &data_off1) == 0,
          "write replica 1");
    CHECK(data_off0 == data_off1, "replica header offsets match");

    const size_t read_len = 1536u * 1024u + 333u; /* > 1 MiB, unaligned length */
    const size_t rel_off = 137u;                  /* unaligned offset */
    const uint64_t file_off = data_off0 + rel_off;
    const size_t slab_cap = read_len + 16384u;
    unsigned char *slab = NULL;
    CHECK(posix_memalign((void **)&slab, 4096, slab_cap) == 0 && slab != NULL,
          "posix_memalign slab");

    /* ---- Case 1: Single replica fallback with V4_STRIPE=1 ---- */
    if (slab) {
        unsetenv("COLI_MODEL_MIRROR");
        unsetenv("SNAP_MIRROR");
        setenv("V4_STRIPE", "1", 1);

        ColiSafetensorsIndex *idx1 = NULL;
        CHECK(coli_st_index_open(&idx1, dir0, error, sizeof(error)) == 0,
              "open single-replica index: %s", error);
        if (idx1) {
            if (idx1->dfds[0] < 0 && idx1->fds[0] >= 0)
                idx1->dfds[0] = dup(idx1->fds[0]);
            CHECK(coli_st_mirror_nrep() == 1,
                  "single-replica nrep expected 1, got %d",
                  coli_st_mirror_nrep());
            reset_mir_stats();
            memset(slab, 0xAA, slab_cap);
            CHECK(coli_v4_test_read_direct_window(idx1, 0, 0, slab, file_off,
                                                  read_len, 0, NULL) == 0,
                  "single-replica read_direct_window succeeded");
            CHECK(memcmp(slab, payload + rel_off, read_len) == 0,
                  "single-replica read byte-identical");
            CHECK(__atomic_load_n(&g_v4_mir_nread[0], __ATOMIC_RELAXED) == 1 &&
                  __atomic_load_n(&g_v4_mir_nread[1], __ATOMIC_RELAXED) == 0,
                  "single-replica accounting stays on replica 0");
            coli_st_index_close(idx1);
        }
    }

    /* ---- Case 2: Dual-replica tests (opt-in off, explicit 0, on, shift, fallback) ---- */
    if (slab) {
        setenv("COLI_MODEL_MIRROR", dir1, 1);
        setenv("COLI_DISK_WEIGHTS", "3,1", 1);

        ColiSafetensorsIndex *idx2 = NULL;
        CHECK(coli_st_index_open(&idx2, dir0, error, sizeof(error)) == 0,
              "open dual-replica index: %s", error);
        if (idx2) {
            int mir_rc = coli_st_mirror_setup(idx2, dir0, 8);
            CHECK(mir_rc == 1, "coli_st_mirror_setup returned %d", mir_rc);
            CHECK(coli_st_mirror_nrep() == 2,
                  "dual-replica nrep expected 2, got %d",
                  coli_st_mirror_nrep());
            int s0 = coli_st_mirror_share(0);
            int s1 = coli_st_mirror_share(1);
            CHECK(s0 > s1 && s0 + s1 == 256,
                  "3:1 disk weights shares s0=%d s1=%d", s0, s1);

            /* Ensure direct fds are valid even on filesystems without O_DIRECT. */
            if (idx2->dfds[0] < 0 && idx2->fds[0] >= 0)
                idx2->dfds[0] = dup(idx2->fds[0]);
            if (idx2->mdfds[0][0] < 0 && idx2->mfds[0][0] >= 0)
                idx2->mdfds[0][0] = dup(idx2->mfds[0][0]);

            /* 2a. Strictly opt-in: V4_STRIPE unset must NOT stripe. */
            unsetenv("V4_STRIPE");
            reset_mir_stats();
            memset(slab, 0xBB, slab_cap);
            CHECK(coli_v4_test_read_direct_window(idx2, 0, 0, slab, file_off,
                                                  read_len, 0, NULL) == 0,
                  "opt-in default read succeeded");
            CHECK(memcmp(slab, payload + rel_off, read_len) == 0,
                  "opt-in default read byte-identical");
            CHECK(__atomic_load_n(&g_v4_mir_nread[0], __ATOMIC_RELAXED) == 1 &&
                  __atomic_load_n(&g_v4_mir_nread[1], __ATOMIC_RELAXED) == 0,
                  "V4_STRIPE unset must not touch replica 1");

            /* 2b. V4_STRIPE=0 explicit disable must NOT stripe. */
            setenv("V4_STRIPE", "0", 1);
            reset_mir_stats();
            memset(slab, 0xCC, slab_cap);
            CHECK(coli_v4_test_read_direct_window(idx2, 0, 0, slab, file_off,
                                                  read_len, 0, NULL) == 0,
                  "V4_STRIPE=0 read succeeded");
            CHECK(memcmp(slab, payload + rel_off, read_len) == 0,
                  "V4_STRIPE=0 read byte-identical");
            CHECK(__atomic_load_n(&g_v4_mir_nread[0], __ATOMIC_RELAXED) == 1 &&
                  __atomic_load_n(&g_v4_mir_nread[1], __ATOMIC_RELAXED) == 0,
                  "V4_STRIPE=0 must not touch replica 1");

            /* 2c. V4_STRIPE=1 striped read (rep=0, unaligned offset, memmove path). */
            setenv("V4_STRIPE", "1", 1);
            reset_mir_stats();
            memset(slab, 0xDD, slab_cap);
            CHECK(coli_v4_test_read_direct_window(idx2, 0, 0, slab, file_off,
                                                  read_len, 0, NULL) == 0,
                  "V4_STRIPE=1 rep=0 read succeeded");
            CHECK(memcmp(slab, payload + rel_off, read_len) == 0,
                  "V4_STRIPE=1 rep=0 byte-identical reconstruction");
            uint64_t b0 = __atomic_load_n(&g_v4_mir_bytes[0], __ATOMIC_RELAXED);
            uint64_t b1 = __atomic_load_n(&g_v4_mir_bytes[1], __ATOMIC_RELAXED);
            uint64_t n0 = __atomic_load_n(&g_v4_mir_nread[0], __ATOMIC_RELAXED);
            uint64_t n1 = __atomic_load_n(&g_v4_mir_nread[1], __ATOMIC_RELAXED);
            CHECK(n0 == 1 && n1 == 1,
                  "V4_STRIPE=1 must read both replicas (n0=%llu n1=%llu)",
                  (unsigned long long)n0, (unsigned long long)n1);
            CHECK(b0 + b1 == read_len,
                  "V4_STRIPE=1 byte accounting sum %llu == %zu",
                  (unsigned long long)(b0 + b1), read_len);
            CHECK(b0 > b1 && b1 > 0,
                  "V4_STRIPE=1 3:1 weights: b0=%llu > b1=%llu > 0",
                  (unsigned long long)b0, (unsigned long long)b1);

            /* 2d. V4_STRIPE=1 with shift != NULL (V4_SLAB_SHIFT path) and rep=1. */
            reset_mir_stats();
            memset(slab, 0xEE, slab_cap);
            size_t shift = 0;
            CHECK(coli_v4_test_read_direct_window(idx2, 0, 1, slab, file_off,
                                                  read_len, 0, &shift) == 0,
                  "V4_STRIPE=1 rep=1 shift read succeeded");
            CHECK(shift == (size_t)(file_off & 4095u),
                  "shift=%zu matches file_off & 4095 (%zu)",
                  shift, (size_t)(file_off & 4095u));
            CHECK(memcmp(slab + shift, payload + rel_off, read_len) == 0,
                  "V4_STRIPE=1 rep=1 shift byte-identical reconstruction");
            b0 = __atomic_load_n(&g_v4_mir_bytes[0], __ATOMIC_RELAXED);
            b1 = __atomic_load_n(&g_v4_mir_bytes[1], __ATOMIC_RELAXED);
            CHECK(b0 > b1 && b0 + b1 == read_len,
                  "V4_STRIPE=1 rep=1: replica 0 still gets larger share (b0=%llu b1=%llu)",
                  (unsigned long long)b0, (unsigned long long)b1);

            /* 2e. Fallback when mirror direct fd is missing (-1). */
            int saved_mdfd = idx2->mdfds[0][0];
            idx2->mdfds[0][0] = -1;
            reset_mir_stats();
            memset(slab, 0x55, slab_cap);
            CHECK(coli_v4_test_read_direct_window(idx2, 0, 0, slab, file_off,
                                                  read_len, 0, NULL) == 0,
                  "missing mirror direct fd fallback succeeded");
            CHECK(memcmp(slab, payload + rel_off, read_len) == 0,
                  "missing mirror direct fd fallback byte-identical");
            CHECK(__atomic_load_n(&g_v4_mir_nread[0], __ATOMIC_RELAXED) == 1 &&
                  __atomic_load_n(&g_v4_mir_nread[1], __ATOMIC_RELAXED) == 0,
                  "missing mirror direct fd falls back to replica 0");
            idx2->mdfds[0][0] = saved_mdfd;

            coli_st_index_close(idx2);
        }
        unsetenv("COLI_MODEL_MIRROR");
        unsetenv("COLI_DISK_WEIGHTS");
        unsetenv("V4_STRIPE");
    }

    if (slab) compat_aligned_free(slab);
    unlink(path0);
    unlink(path1);
    rmdir(dir0);
    rmdir(dir1);
    free(payload);
}

int main(void) {
    test_split_size_and_alignment();
    test_striped_replica_reconstruction();
    if (failures) {
        fprintf(stderr, "test_v4_stripe_read: %d failure(s)\n", failures);
        return 1;
    }
    puts("DeepSeek-V4 V4_STRIPE replica-read tests: ok");
    return 0;
}
