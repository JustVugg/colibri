/* #1479: two shards carrying the same tensor name are resolved by the
 * checkpoint's model.safetensors.index.json (the copy the index maps the name
 * to is read, the other ignored), whichever order the directory scan meets
 * them in; without an index, or with an index that does not settle it, the
 * historical refusal stands. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#define RMDIR(p) _rmdir(p)
#else
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#define MKDIR(p) mkdir(p, 0755)
#define RMDIR(p) rmdir(p)
#endif
#include "../st.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

/* one safetensors file with tensor `n1` = base+0..7 and, if n2, `n2` = base2+0..7 */
static int write_shard(const char *dir, const char *file, const char *n1, float base, const char *n2, float base2) {
    char hdr[512], path[512];
    int hl = n2 ? snprintf(hdr, sizeof hdr, "{\"%s\":{\"dtype\":\"F32\",\"shape\":[8],\"data_offsets\":[0,32]},"
                                            "\"%s\":{\"dtype\":\"F32\",\"shape\":[8],\"data_offsets\":[32,64]}}", n1, n2)
                : snprintf(hdr, sizeof hdr, "{\"%s\":{\"dtype\":\"F32\",\"shape\":[8],\"data_offsets\":[0,32]}}", n1);
    snprintf(path, sizeof path, "%s/%s", dir, file);
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); return -1; }
    uint64_t n = (uint64_t)hl; fwrite(&n, 8, 1, f); fwrite(hdr, 1, (size_t)hl, f);
    float d[8]; for (int i = 0; i < 8; i++) d[i] = base + i; fwrite(d, 4, 8, f);
    if (n2) { for (int i = 0; i < 8; i++) d[i] = base2 + i; fwrite(d, 4, 8, f); }
    fclose(f); return 0;
}
static int write_index(const char *dir, const char *json) {
    char path[512]; snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    fputs(json, f); fclose(f); return 0;
}
static void wipe(const char *dir) {
    const char *names[] = { "model-00001-of-00001.safetensors", "model-overlay-00001.safetensors",
                            "aaa-overlay.safetensors", "model.safetensors.index.json" };
    for (int i = 0; i < 4; i++) { char p[512]; snprintf(p, sizeof p, "%s/%s", dir, names[i]); remove(p); }
    RMDIR(dir);
}
static int read8(shards *S, const char *name, float *out) { return st_read_f32(S, name, out, 0) == 8; }

int main(void) {
    const char *D = "tmp_overlay";
    float v[8];

    /* 1. base scanned first, overlay second: the index points at the overlay -> replace in place */
    wipe(D); CHECK(MKDIR(D) == 0);
    CHECK(write_shard(D, "model-00001-of-00001.safetensors", "a", 1.f, "b", 10.f) == 0);
    CHECK(write_shard(D, "model-overlay-00001.safetensors", "a", 100.f, NULL, 0) == 0);
    CHECK(write_index(D, "{\"metadata\":{},\"weight_map\":{\"a\":\"model-overlay-00001.safetensors\",\"b\":\"model-00001-of-00001.safetensors\"}}") == 0);
    { shards S; memset(&S, 0, sizeof S); st_init(&S, D);
      CHECK(S.n == 2);
      CHECK(read8(&S, "a", v) && v[0] == 100.f && v[7] == 107.f);
      CHECK(read8(&S, "b", v) && v[0] == 10.f);
      st_destroy(&S); }
    puts("overlay after base: index copy read, stale copy ignored");

    /* 2. overlay scanned first (sorts before "model-"): the stale base copy is skipped */
    wipe(D); CHECK(MKDIR(D) == 0);
    CHECK(write_shard(D, "aaa-overlay.safetensors", "a", 100.f, NULL, 0) == 0);
    CHECK(write_shard(D, "model-00001-of-00001.safetensors", "a", 1.f, "b", 10.f) == 0);
    CHECK(write_index(D, "{\"weight_map\":{\"a\":\"aaa-overlay.safetensors\",\"b\":\"model-00001-of-00001.safetensors\"}}") == 0);
    { shards S; memset(&S, 0, sizeof S); st_init(&S, D);
      CHECK(S.n == 2);
      CHECK(read8(&S, "a", v) && v[0] == 100.f);
      CHECK(read8(&S, "b", v) && v[0] == 10.f);
      st_destroy(&S); }
    puts("overlay before base: index copy read, stale copy ignored");

    /* 3. the index keeps the base copy: the overlay's is the one ignored */
    wipe(D); CHECK(MKDIR(D) == 0);
    CHECK(write_shard(D, "model-00001-of-00001.safetensors", "a", 1.f, "b", 10.f) == 0);
    CHECK(write_shard(D, "model-overlay-00001.safetensors", "a", 100.f, NULL, 0) == 0);
    CHECK(write_index(D, "{\"weight_map\":{\"a\":\"model-00001-of-00001.safetensors\",\"b\":\"model-00001-of-00001.safetensors\"}}") == 0);
    { shards S; memset(&S, 0, sizeof S); st_init(&S, D);
      CHECK(S.n == 2);
      CHECK(read8(&S, "a", v) && v[0] == 1.f);
      st_destroy(&S); }
    puts("index prefers the base: overlay copy ignored");

#ifndef _WIN32
    /* 4. no index, and 5. an index that does not list the name: still refused (exit 1) */
    for (int variant = 0; variant < 2; variant++) {
        wipe(D); CHECK(MKDIR(D) == 0);
        CHECK(write_shard(D, "model-00001-of-00001.safetensors", "a", 1.f, "b", 10.f) == 0);
        CHECK(write_shard(D, "model-overlay-00001.safetensors", "a", 100.f, NULL, 0) == 0);
        if (variant == 1) CHECK(write_index(D, "{\"weight_map\":{\"b\":\"model-00001-of-00001.safetensors\"}}") == 0);
        fflush(stdout); fflush(stderr);   /* the child inherits the buffers: flush, or it replays them */
        pid_t pid = fork(); CHECK(pid >= 0);
        if (pid == 0) { if (!freopen("/dev/null", "w", stderr)) _exit(2); shards S; memset(&S, 0, sizeof S); st_init(&S, D); _exit(0); }
        int status = 0; CHECK(waitpid(pid, &status, 0) == pid);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    }
    puts("no index / index silent on the name: duplicate still refused");
#endif
    wipe(D);
    puts("safetensors overlay tests: ok");
    return 0;
}
