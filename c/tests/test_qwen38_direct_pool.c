/* Qwen3.8 direct expert reads (Q38_EXPERT_DIRECT) and 32-buffer aligned pool.
 * Verifies:
 *  - opt-in default (Q38_EXPERT_DIRECT=0) keeps buffered reads and prefetch on;
 *  - Q38_EXPERT_DIRECT=1 reads unaligned interior ranges directly into the pool
 *    with bit-exact bytes, and falls back to buffered I/O past EOF or when all
 *    Q38_DIRECT_BUFS slots are busy;
 *  - varying OpenMP team sizes reuse the fixed pool without leaking buffers or
 *    leaving any slot busy. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
    return 1; \
} } while (0)

static int make_temp_shard(char *path, size_t path_cap, unsigned char **ref_out, int64_t size) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = getenv("TEMP");
    if (!tmp || !*tmp) tmp = getenv("TMP");
    if (!tmp || !*tmp) tmp = "/tmp";
    snprintf(path, path_cap, "%s/colibri_q38_direct_%d.bin", tmp, (int)getpid());
    unsigned char *ref = (unsigned char *)malloc((size_t)size);
    if (!ref) return -1;
    for (int64_t i = 0; i < size; i++)
        ref[i] = (unsigned char)((i * 131u + (i >> 8) * 17u + 0x5Au) & 0xFFu);
    FILE *fp = fopen(path, "wb");
    if (!fp) { free(ref); return -1; }
    size_t wrote = fwrite(ref, 1, (size_t)size, fp);
    fclose(fp);
    if (wrote != (size_t)size) { remove(path); free(ref); return -1; }
    *ref_out = ref;
    return 0;
}

static int open_direct_fd(const char *path) {
#ifdef O_DIRECT
    return open(path, COMPAT_O_RDONLY | O_DIRECT);
#elif defined(__APPLE__) || defined(_WIN32)
    return compat_open_direct(path);
#else
    (void)path;
    return -1;
#endif
}

int main(void) {
    /* 1. Default env behavior: Q38_EXPERT_DIRECT is opt-in (0), prefetch defaults to 1. */
    unsetenv("Q38_EXPERT_DIRECT");
    unsetenv("Q38_EXPERT_PREFETCH");
    int d0 = q38_env_bool("Q38_EXPERT_DIRECT", 0);
    int p0 = q38_env_bool("Q38_EXPERT_PREFETCH", !d0);
    CHECK(d0 == 0 && p0 == 1);

    /* Q38_EXPERT_DIRECT=1 turns prefetch default off unless explicitly enabled. */
    setenv("Q38_EXPERT_DIRECT", "1", 1);
    int d1 = q38_env_bool("Q38_EXPERT_DIRECT", 0);
    int p1 = q38_env_bool("Q38_EXPERT_PREFETCH", !d1);
    CHECK(d1 == 1 && p1 == 0);

    setenv("Q38_EXPERT_PREFETCH", "1", 1);
    int p1_override = q38_env_bool("Q38_EXPERT_PREFETCH", !d1);
    CHECK(p1_override == 1);
    unsetenv("Q38_EXPERT_PREFETCH");
    unsetenv("Q38_EXPERT_DIRECT");

    /* 2. Build a real file with a non-block-aligned tail (32 KiB + 777 bytes). */
    const int64_t fsize = 8 * ST_DIRECT_ALIGN + 777;
    char path[512];
    unsigned char *ref = NULL;
    CHECK(make_temp_shard(path, sizeof path, &ref, fsize) == 0);

    int fd = open(path, COMPAT_O_RDONLY);
    int dfd = open_direct_fd(path);
    CHECK(fd >= 0);

    Model m;
    memset(&m, 0, sizeof m);
    m.S.nfd = 1;
    m.S.fds[0] = fd;
    m.S.dfds[0] = dfd;
    m.S.sizes[0] = fsize;

    unsigned char out[16384];

    /* 2a. expert_direct=0: always buffered, never increments g_q38_direct_reads. */
    m.expert_direct = 0;
    unsigned long long c0 = g_q38_direct_reads;
    memset(out, 0, sizeof out);
    q38_read_expert_range(&m, fd, 123, 5000, out, sizeof out, "test buffered");
    CHECK(memcmp(out, ref + 123, 5000) == 0);
    CHECK(g_q38_direct_reads == c0);

    /* 2b. expert_direct=1, unaligned interior range: uses direct pool when dfd >= 0. */
    m.expert_direct = 1;
    c0 = g_q38_direct_reads;
    memset(out, 0, sizeof out);
    q38_read_expert_range(&m, fd, 333, 9123, out, sizeof out, "test direct interior");
    CHECK(memcmp(out, ref + 333, 9123) == 0);
    if (dfd >= 0) CHECK(g_q38_direct_reads == c0 + 1);

    /* 2c. expert_direct=1, tail range whose enclosing 4 KiB blocks pass EOF: falls back buffered. */
    c0 = g_q38_direct_reads;
    memset(out, 0, sizeof out);
    int64_t tail_off = fsize - 1000;
    q38_read_expert_range(&m, fd, tail_off, 1000, out, sizeof out, "test direct tail fallback");
    CHECK(memcmp(out, ref + tail_off, 1000) == 0);
    CHECK(g_q38_direct_reads == c0);

    /* 2d. Pool exhaustion: mark all Q38_DIRECT_BUFS slots busy -> falls back to buffered cleanly. */
    for (int i = 0; i < Q38_DIRECT_BUFS; i++)
        __atomic_store_n(&g_q38_dbuf[i].busy, 1, __ATOMIC_RELEASE);
    c0 = g_q38_direct_reads;
    memset(out, 0, sizeof out);
    q38_read_expert_range(&m, fd, 512, 4096, out, sizeof out, "test pool exhausted");
    CHECK(memcmp(out, ref + 512, 4096) == 0);
    CHECK(g_q38_direct_reads == c0);
    for (int i = 0; i < Q38_DIRECT_BUFS; i++)
        __atomic_store_n(&g_q38_dbuf[i].busy, 0, __ATOMIC_RELEASE);

    /* 3. Varying OpenMP team sizes: verify pool stays bounded and leak-free. */
    int mismatches = 0;
    const int teams[] = {1, 2, 4, 8, 3, 6, 2, 8, 1, 4};
    for (int r = 0; r < (int)(sizeof(teams) / sizeof(teams[0])); r++) {
        int nt = teams[r];
        #pragma omp parallel for num_threads(nt) reduction(+:mismatches)
        for (int j = 0; j < 24; j++) {
            int64_t off = 17 + (int64_t)(j % 5) * 1009;
            int64_t len = 4100 + (int64_t)(j % 3) * 1024;
            unsigned char local[8192];
            q38_read_expert_range(&m, fd, off, len, local, sizeof local, "test parallel direct");
            if (memcmp(local, ref + off, (size_t)len) != 0) mismatches++;
        }
    }
    CHECK(mismatches == 0);

    int pooled = 0, busy = 0;
    for (int i = 0; i < Q38_DIRECT_BUFS; i++) {
        if (g_q38_dbuf[i].p != NULL) pooled++;
        if (__atomic_load_n(&g_q38_dbuf[i].busy, __ATOMIC_ACQUIRE)) busy++;
    }
    CHECK(busy == 0);
    CHECK(pooled <= Q38_DIRECT_BUFS);
    if (dfd >= 0) CHECK(pooled >= 1);

    if (dfd >= 0 && dfd != fd) close(dfd);
    close(fd);
    remove(path);
    free(ref);

    printf("test_qwen38_direct_pool: PASS (dfd=%s, direct_reads=%llu, pooled=%d/%d)\n",
           dfd >= 0 ? "yes" : "no", g_q38_direct_reads, pooled, Q38_DIRECT_BUFS);
    return 0;
}
