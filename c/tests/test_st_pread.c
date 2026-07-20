/* st_pread_full: chunk loop + honest truncation errors.
 * Built with -DST_PREAD_CHUNK=7 so a ~100-byte tensor takes many pread calls —
 * exercising the loop that production only needs past 2^31 bytes (one pread
 * caps there on Linux; big bf16 tensors exceed it). Also forks a child against
 * a truncated shard and requires exit(1) with a "short read" message instead
 * of the old perror("... : Success"). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static void write_snap(const char *dir, int truncate_bytes) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    unsigned char data[96];
    for (int i = 0; i < 96; i++) data[i] = (unsigned char)(i * 7 + 3);
    const char *hdr = "{\"t\":{\"dtype\":\"U8\",\"shape\":[96],\"data_offsets\":[0,96]}}";
    uint64_t hlen = strlen(hdr);
    FILE *f = fopen(path, "wb");
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, hlen, f);
    fwrite(data, 1, (size_t)(96 - truncate_bytes), f);
    fclose(f);
}

int main(void) {
    /* relative to the CWD, per test_stops: MinGW .exe files resolve Windows
     * paths and "/tmp" is not one */
    char dir[] = "test_st_pread_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }

    /* 1) chunk loop: 96-byte tensor read 7 bytes at a time, content exact */
    write_snap(dir, 0);
    shards S; st_init(&S, dir);
    unsigned char out[96] = {0};
    st_read_raw(&S, "t", out, 0);
    for (int i = 0; i < 96; i++) CHECK(out[i] == (unsigned char)(i * 7 + 3));

#ifndef _WIN32
    /* 2) shard truncated AFTER st_init (init validates static bounds, so the
     * pread path only fires when the file shrinks underneath a live handle):
     * child must exit(1) with an honest message, not perror's "Success" */
    char shard[512]; snprintf(shard, sizeof(shard), "%s/model.safetensors", dir);
    struct stat sb; CHECK(stat(shard, &sb) == 0);
    CHECK(truncate(shard, sb.st_size - 40) == 0);
    int pipefd[2]; CHECK(pipe(pipefd) == 0);
    pid_t pid = fork(); CHECK(pid >= 0);
    if (pid == 0) {
        dup2(pipefd[1], 2); close(pipefd[0]); close(pipefd[1]);
        unsigned char buf[96];
        st_read_raw(&S, "t", buf, 0);    /* inherited handles; must exit(1) inside */
        _exit(42);                        /* reaching here = bug */
    }
    close(pipefd[1]);
    char err[512] = {0};
    ssize_t n = read(pipefd[0], err, sizeof(err)-1); (void)n;
    close(pipefd[0]);
    int status = 0; waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    CHECK(strstr(err, "short read") != NULL);
    CHECK(strstr(err, "Success") == NULL);
#else
    /* fork/pipe/truncate are POSIX; Windows still runs the chunk-loop check */
    printf("test_st_pread: truncation subtest skipped on Windows\n");
#endif

    char cmd[600];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd), "rmdir /s /q %s", dir);
#else
    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
#endif
    if (system(cmd)) {}
    printf("test_st_pread: chunk loop + honest truncation error: ok\n");
    return 0;
}
