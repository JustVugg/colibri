/* Safetensors shape validation, including st_init() exit(1) paths.
 *
 * st_init() deliberately terminates the process for hostile containers, so
 * the parent writes one container per case and invokes this same binary in
 * child mode. This keeps the refusal checks real on Windows as well as POSIX.
 */
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

typedef struct {
    const char *shape;
    int accept;
    int rank;
    int64_t numel;
    int data_bytes;
} shape_case;

static const shape_case CASES[] = {
    {"[2,\"3\"]", 0, 0, 0, 0},
    {"[2,3.5]", 0, 0, 0, 0},
    {"[2,-1]", 0, 0, 0, 0},
    {"[9223372036854775807,2]", 0, 0, 0, 0},
    {"[4611686018427387904,3]", 0, 0, 0, 0},
    {"[1,1,1,1,1,1,1,1]", 1, 8, 1, 1},
    {"[1,1,1,1,1,1,1,1,1]", 0, 0, 0, 0},
    {"[]", 1, 0, 1, 1},
    {"[0,4096]", 1, 2, 0, 0},
};

static int write_case(const char *dir, const shape_case *test) {
    char path[512];
    char header[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    int header_bytes = snprintf(
        header, sizeof(header),
        "{\"t\":{\"dtype\":\"U8\",\"shape\":%s,\"data_offsets\":[0,%d]}}",
        test->shape, test->data_bytes);
    if (header_bytes < 0 || (size_t)header_bytes >= sizeof(header)) return -1;

    FILE *file = fopen(path, "wb");
    if (!file) return -1;
    uint64_t hlen = (uint64_t)header_bytes;
    unsigned char byte = 0;
    int failed =
        fwrite(&hlen, 8, 1, file) != 1 ||
        fwrite(header, 1, (size_t)header_bytes, file) != (size_t)header_bytes ||
        (test->data_bytes && fwrite(&byte, 1, 1, file) != 1);
    if (fclose(file) != 0) failed = 1;
    return failed ? -1 : 0;
}

static int child_case(int index, const char *dir) {
    if (index < 0 || index >= (int)(sizeof(CASES) / sizeof(CASES[0]))) return 90;
    const shape_case *test = &CASES[index];
    shards S;
    st_init(&S, dir);

    /* A refusing case reaching here means st_init() incorrectly accepted it. */
    if (!test->accept) return 91;
    st_tensor *tensor = st_find(&S, "t");
    if (!tensor || tensor->rank != test->rank || tensor->numel != test->numel)
        return 92;
    for (int i = 0; i < tensor->rank; i++) {
        int64_t expected =
            test->rank == 2 && test->numel == 0 ? (i == 0 ? 0 : 4096) : 1;
        if (tensor->shape[i] != expected) return 93;
    }
    for (int i = tensor->rank; i < ST_MAX_RANK; i++)
        if (tensor->shape[i] != 0) return 94;
    return 0;
}

static int subprocess_exit_code(const char *self, int index, const char *dir) {
    char command[1536];
    char error_path[512];
    snprintf(error_path, sizeof(error_path), "%s/stderr.txt", dir);
#ifdef _WIN32
    int n = snprintf(command, sizeof(command),
                     "call \"%s\" --shape-child %d \"%s\" 2>\"%s\"",
                     self, index, dir, error_path);
#else
    int n = snprintf(command, sizeof(command),
                     "\"%s\" --shape-child %d \"%s\" 2>\"%s\"",
                     self, index, dir, error_path);
#endif
    if (n < 0 || (size_t)n >= sizeof(command)) return -1;
    int status = system(command);
#ifdef _WIN32
    return status;
#else
    if (status < 0 || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
#endif
}

static void remove_case(const char *dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    remove(path);
    snprintf(path, sizeof(path), "%s/stderr.txt", dir);
    remove(path);
#ifdef _WIN32
    _rmdir(dir);
#else
    rmdir(dir);
#endif
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--shape-child"))
        return child_case(atoi(argv[2]), argv[3]);

    CHECK(argc >= 1 && argv[0] && *argv[0]);
    for (int i = 0; i < (int)(sizeof(CASES) / sizeof(CASES[0])); i++) {
        char dir[] = "test_st_shape_XXXXXX";
        CHECK(mkdtemp(dir) != NULL);
        CHECK(write_case(dir, &CASES[i]) == 0);
        int code = subprocess_exit_code(argv[0], i, dir);
        char error_path[512];
        char error[1024] = {0};
        snprintf(error_path, sizeof(error_path), "%s/stderr.txt", dir);
        FILE *error_file = fopen(error_path, "rb");
        CHECK(error_file != NULL);
        size_t error_bytes = fread(error, 1, sizeof(error) - 1, error_file);
        CHECK(!ferror(error_file));
        fclose(error_file);
        error[error_bytes] = 0;
        remove_case(dir);
        if (CASES[i].accept) {
            CHECK(code == 0);
            CHECK(error_bytes == 0);
        } else {
            CHECK(code == 1);
            CHECK(strstr(error, "tensor 't'") != NULL);
            CHECK(strstr(error, "shape") != NULL);
        }
    }

    printf("test_st_shape: hostile dimensions, rank limits, scalar and zero shape: ok\n");
    return 0;
}
