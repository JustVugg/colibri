/* st_die_missing: with model.safetensors.index.json present, the message says
 * which of three things happened -- the name is not in this checkpoint at all
 * (wrong engine for the directory), the shard the index maps it to is absent,
 * or that shard is here but does not declare the tensor (bad copy). Without
 * an index the historical file-count wording stands. POSIX only: the function
 * exits, so each case runs in a forked child whose stderr is captured. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
int main(void) { puts("test_st_missing: skipped on Windows (fork)"); return 0; }
#else
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>
#include "../st.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
static int write_shard(const char *dir, const char *file, const char *name) {
    char hdr[256], path[512];
    int hl = snprintf(hdr, sizeof hdr, "{\"%s\":{\"dtype\":\"F32\",\"shape\":[8],\"data_offsets\":[0,32]}}", name);
    snprintf(path, sizeof path, "%s/%s", dir, file);
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    uint64_t n = (uint64_t)hl; fwrite(&n, 8, 1, f); fwrite(hdr, 1, (size_t)hl, f);
    float d[8] = {0}; fwrite(d, 4, 8, f); fclose(f); return 0;
}
static int put(const char *dir, const char *file, const char *text) {
    char p[512]; snprintf(p, sizeof p, "%s/%s", dir, file);
    FILE *f = fopen(p, "wb"); if (!f) return -1; fputs(text, f); fclose(f); return 0;
}
static void wipe(const char *dir) {
    const char *names[] = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"};
    for (int i = 0; i < 3; i++) { char p[512]; snprintf(p, sizeof p, "%s/%s", dir, names[i]); remove(p); }
    rmdir(dir);
}
/* run st_init + st_read_f32(name) in a child; return its stderr in buf, its exit code */
static int run_case(const char *dir, const char *name, char *buf, size_t cap) {
    int pfd[2]; if (pipe(pfd)) return -1;
    fflush(stdout); fflush(stderr);
    pid_t pid = fork(); if (pid < 0) return -1;
    if (pid == 0) {
        close(pfd[0]); dup2(pfd[1], 2); close(pfd[1]);
        shards S; memset(&S, 0, sizeof S); st_init(&S, dir);
        float v[8]; st_read_f32(&S, name, v, 0);
        _exit(0);
    }
    close(pfd[1]); size_t got = 0; ssize_t r;
    while (got + 1 < cap && (r = read(pfd[0], buf + got, cap - 1 - got)) > 0) got += (size_t)r;
    buf[got] = 0; close(pfd[0]);
    int status = 0; waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int main(void) {
    const char *D = "tmp_st_missing"; char err[8192];
    wipe(D); CHECK(mkdir(D, 0755) == 0);
    CHECK(write_shard(D, "model-00001-of-00002.safetensors", "a") == 0);
    CHECK(write_shard(D, "model-00002-of-00002.safetensors", "b") == 0);

    /* 1. no index: the file-count wording, "complete" branch */
    CHECK(run_case(D, "model.embed_tokens.weight", err, sizeof err) == 1);
    CHECK(strstr(err, "NOT the usual truncated download") != NULL);
    CHECK(strstr(err, "model.safetensors.index.json") == NULL);

    /* 2. index present, name absent from it: wrong engine for this checkpoint */
    CHECK(put(D, "model.safetensors.index.json", "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00002-of-00002.safetensors\"}}") == 0);
    CHECK(run_case(D, "model.embed_tokens.weight", err, sizeof err) == 1);
    CHECK(strstr(err, "never had a") != NULL && strstr(err, "not this") != NULL);
    CHECK(strstr(err, "coli info --model") != NULL);

    /* 3. index maps the name to a shard that is not here */
    CHECK(put(D, "model.safetensors.index.json", "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"c\":\"model-00003-of-00003.safetensors\"}}") == 0);
    CHECK(run_case(D, "c", err, sizeof err) == 1);
    CHECK(strstr(err, "MISSING") != NULL && strstr(err, "model-00003-of-00003.safetensors") != NULL);

    /* 4. index maps the name to a shard that is here but does not declare it */
    CHECK(put(D, "model.safetensors.index.json", "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"c\":\"model-00002-of-00002.safetensors\"}}") == 0);
    CHECK(run_case(D, "c", err, sizeof err) == 1);
    CHECK(strstr(err, "header does not declare it") != NULL);

    /* and a present tensor still reads fine with the index around */
    { shards S; memset(&S, 0, sizeof S); st_init(&S, D); float v[8]; CHECK(st_read_f32(&S, "a", v, 0) == 8); st_destroy(&S); }
    wipe(D);

    /* 5. an index path that does not fit the buffer is refused, not opened truncated:
     * with a 1193-char directory, "<dir>/model" is all snprintf would keep of the name.
     * The case has to build that directory, so it runs where the file system takes a
     * path that long: on macOS PATH_MAX is 1024 and mkdir answers ENAMETOOLONG, and
     * there the case steps aside -- the Linux runs cover the guard. Any other mkdir
     * failure is a real one and fails the test. */
    {
        char dir[1200] = "tmp_st_long", p[1300]; size_t n = strlen(dir);
        int made = mkdir(dir, 0755) == 0;
        while (made && n < 1193) {
            size_t k = 1193 - n - 1 > 200 ? 200 : 1193 - n - 1;
            dir[n++] = '/'; memset(dir + n, 'd', k); n += k; dir[n] = 0;
            made = mkdir(dir, 0755) == 0;
        }
        int too_long = !made && errno == ENAMETOOLONG, refused = 1;
        if (made) {
            snprintf(p, sizeof p, "%s/model", dir);
            FILE *f = fopen(p, "wb"); CHECK(f != NULL); fputs("{\"weight_map\":{\"a\":\"x\"}}", f); fclose(f);
            st_index ix; memset(&ix, 0, sizeof ix); st_index_load(&ix, dir);
            refused = ix.root == NULL && ix.map == NULL;
            remove(p);
        }
        for (char *s; (s = strrchr(dir, '/')) != NULL; *s = 0) rmdir(dir);
        rmdir(dir);
        CHECK(made || too_long);
        CHECK(refused);
        if (!made) puts("st_index_load long-path guard: skipped, the file system caps the path");
    }
    puts("st_die_missing diagnosis tests: ok");
    return 0;
}
#endif
