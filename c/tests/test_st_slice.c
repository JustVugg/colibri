/* Bounds-checked raw tensor byte slices, including native one-byte F8 and
 * eight-byte I64 tensors.  The reader must validate before pread and must not
 * turn a zero-length DONTNEED into a request for the rest of the shard. */
#define _GNU_SOURCE
#include <stdint.h>
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

static void write_snap(const char *dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    const char *hdr =
        "{\"f8\":{\"dtype\":\"F8_E4M3\",\"shape\":[3,5],\"data_offsets\":[0,15]},"
        "\"ids\":{\"dtype\":\"I64\",\"shape\":[4],\"data_offsets\":[15,47]},"
        "\"f32\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[47,51]}}";
    uint64_t hlen = strlen(hdr);
    unsigned char bytes[51];
    for (int i = 0; i < (int)sizeof(bytes); i++) bytes[i] = (unsigned char)(0xa0 + i);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, hlen, f);
    fwrite(bytes, 1, sizeof(bytes), f);
    fclose(f);
}

#ifndef _WIN32
static int expect_failure(shards *S, const char *name, int64_t off, int64_t nbytes,
                          void *out, int64_t cap, const char *needle) {
    int pipefd[2]; CHECK(pipe(pipefd) == 0);
    pid_t pid = fork(); CHECK(pid >= 0);
    if (pid == 0) {
        dup2(pipefd[1], 2); close(pipefd[0]); close(pipefd[1]);
        st_read_slice_raw_cap(S, name, off, nbytes, out, cap, 0);
        _exit(42);
    }
    close(pipefd[1]);
    char err[512] = {0};
    ssize_t n = read(pipefd[0], err, sizeof(err) - 1); (void)n;
    close(pipefd[0]);
    int status = 0; waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    CHECK(strstr(err, needle) != NULL);
    return 0;
}
#endif

int main(void) {
    char dir[] = "test_st_slice_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    write_snap(dir);

    shards S; st_init(&S, dir);
    CHECK(st_numel(&S, "f8") == 15);
    CHECK(st_nbytes(&S, "f8") == 15);
    CHECK(st_numel(&S, "ids") == 4);
    CHECK(st_nbytes(&S, "ids") == 32);

    unsigned char f8[5] = {0};
    st_read_slice_raw_cap(&S, "f8", 7, 5, f8, sizeof(f8), 0);
    for (int i = 0; i < 5; i++) CHECK(f8[i] == (unsigned char)(0xa0 + 7 + i));

    uint64_t ids[2] = {0};
    st_read_slice_raw_cap(&S, "ids", 8, sizeof(ids), ids, sizeof(ids), 0);
    unsigned char *id_bytes = (unsigned char *)ids;
    for (int i = 0; i < (int)sizeof(ids); i++)
        CHECK(id_bytes[i] == (unsigned char)(0xa0 + 15 + 8 + i));

    /* Empty slices are valid, but must leave the destination untouched. */
    unsigned char sentinel = 0x5a;
    st_read_slice_raw_cap(&S, "f8", 15, 0, &sentinel, 0, 1);
    CHECK(sentinel == 0x5a);
    st_read_slice_f32(&S, "f32", 1, 0, NULL, 1);

#ifndef _WIN32
    unsigned char out[16] = {0};
    CHECK(expect_failure(&S, "f8", -1, 1, out, sizeof(out), "negative") == 0);
    CHECK(expect_failure(&S, "f8", 0, -1, out, sizeof(out), "negative") == 0);
    CHECK(expect_failure(&S, "f8", 0, 16, out, sizeof(out), "out of tensor byte bounds") == 0);
    CHECK(expect_failure(&S, "f8", 0, 8, out, 7, "destination holds") == 0);
    CHECK(expect_failure(&S, "f8", 0, 1, NULL, 1, "NULL destination") == 0);
    CHECK(expect_failure(&S, "f8", 15, 1, out, sizeof(out), "out of tensor byte bounds") == 0);

    /* Exercise the absolute-offset overflow guard independently of the tensor
     * range checks.  This mutates only the child-inherited index. */
    st_tensor *f8_tensor = st_find(&S, "f8");
    CHECK(f8_tensor != NULL);
    int64_t saved_off = f8_tensor->off;
    int64_t saved_nbytes = f8_tensor->nbytes;
    f8_tensor->off = INT64_MAX;
    f8_tensor->nbytes = INT64_MAX;
    CHECK(expect_failure(&S, "f8", 1, 1, out, sizeof(out), "overflows int64") == 0);
    f8_tensor->off = saved_off;
    f8_tensor->nbytes = saved_nbytes;
#else
    printf("test_st_slice: rejection subtests skipped on Windows\n");
#endif

    st_destroy(&S);
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    CHECK(remove(path) == 0);
#ifdef _WIN32
    CHECK(_rmdir(dir) == 0);
#else
    CHECK(rmdir(dir) == 0);
#endif
    printf("test_st_slice: raw F8/I64 slices and bounds: ok\n");
    return 0;
}
