/* Multi-SSD mirror (st_mirror_add & friends): read-only model copies are
 * accepted only when byte-identical in size and safetensors header, reads on
 * any replica return the same bytes, and divergent/missing copies degrade
 * to the primary instead of being trusted. Fixture dirs are created in the
 * current working directory and removed on exit. */
#include <stdint.h>
#include <stdio.h>
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

#define CHECK(condition)                                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition);                              \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define DIR_A "tmp_mirror_a" /* primary */
#define DIR_B "tmp_mirror_b" /* identical copy */
#define DIR_C "tmp_mirror_c" /* same size, divergent header */
#define DIR_D "tmp_mirror_d" /* different size */
#define DIR_E "tmp_mirror_e" /* empty (missing file) */
#define DIR_F "tmp_mirror_f" /* second identical copy (multi-SSD) */

/* one-tensor safetensors file; flip lets us corrupt one header byte and
 * pad lets us grow the payload, both without changing anything else */
static int write_model(const char *dir, int flip, int pad) {
    char hdr[128], path[256];
    int hl = snprintf(hdr, sizeof(hdr), "{\"t0\":{\"dtype\":\"F32\",\"shape\":[8],\"data_offsets\":[0,32]}}");
    if (flip) hdr[hl - 2] = ' '; /* inside the JSON header, same length */
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    uint64_t n = (uint64_t)hl;
    fwrite(&n, 8, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    float data[8] = {1, -2, 3.5f, 0, 42, -0.5f, 7, 8};
    fwrite(data, 4, 8, f);
    for (int i = 0; i < pad; i++) fputc(0, f);
    fclose(f);
    return 0;
}

static void cleanup(void) {
    const char *dirs[] = {DIR_A, DIR_B, DIR_C, DIR_D, DIR_E, DIR_F};
    for (int i = 0; i < 6; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/model.safetensors", dirs[i]);
        remove(path);
        RMDIR(dirs[i]); /* remove() deletes only files on Windows: stale dirs made every 2nd run fail */
    }
}

int main(void) {
    cleanup();
    CHECK(MKDIR(DIR_A) == 0 && MKDIR(DIR_B) == 0 && MKDIR(DIR_C) == 0 && MKDIR(DIR_D) == 0 && MKDIR(DIR_E) == 0 &&
          MKDIR(DIR_F) == 0);
    CHECK(write_model(DIR_A, 0, 0) == 0);
    CHECK(write_model(DIR_B, 0, 0) == 0);
    CHECK(write_model(DIR_C, 1, 0) == 0);
    CHECK(write_model(DIR_D, 0, 64) == 0);
    CHECK(write_model(DIR_F, 0, 0) == 0);

    shards S;
    st_init(&S, DIR_A);
    CHECK(S.n == 1 && S.nfd == 1);
    st_tensor *t = st_find(&S, "t0");
    CHECK(t != NULL);

    /* without a mirror: replica 0 is the identity, replica 1 is absent */
    CHECK(st_fd_rep(&S, t->fd, 0) == t->fd);
    CHECK(st_fd_rep(&S, t->fd, 1) == -1);

    /* identical copy: accepted, and both replicas serve the same bytes */
    CHECK(st_mirror_init(&S, DIR_B) == 1);
    int mfd = st_fd_rep(&S, t->fd, 1);
    CHECK(mfd >= 0 && mfd != t->fd);
#ifdef _WIN32
    CHECK(st_direct_fd_rep(&S, t->fd, 1) >= 0);
#endif
    float a[8], b[8];
    CHECK(pread(t->fd, a, t->nbytes, t->off) == t->nbytes);
    CHECK(pread(mfd, b, t->nbytes, t->off) == t->nbytes);
    CHECK(memcmp(a, b, sizeof(a)) == 0);
    st_prefetch_rep(&S, "t0", 1); /* smoke: WILLNEED on the mirror fd */
    st_prefetch_rep(&S, "t0", 0);

    /* divergent header, same size: rejected */
    CHECK(st_mirror_init(&S, DIR_C) == 0);
    CHECK(st_fd_rep(&S, t->fd, 1) == -1);

    /* different size: rejected */
    CHECK(st_mirror_init(&S, DIR_D) == 0);

    /* missing file: rejected (partial mirror with zero shards) */
    CHECK(st_mirror_init(&S, DIR_E) == 0);

    /* unknown fd never maps to a replica */
    CHECK(st_fd_rep(&S, 987654, 1) == -1);

    /* ---- multi-SSD: two identical copies become replicas 1 and 2 ---- */
    st_mirror_reset(&S);
    CHECK(st_mirror_add(&S, DIR_B) == 1);
    CHECK(st_mirror_add(&S, DIR_F) == 1);
    CHECK(S.nrep == 2);
    int f1 = st_fd_rep(&S, t->fd, 1), f2 = st_fd_rep(&S, t->fd, 2);
    CHECK(f1 >= 0 && f2 >= 0 && f1 != f2 && f1 != t->fd && f2 != t->fd);
    float c2[8];
    CHECK(pread(f2, c2, t->nbytes, t->off) == t->nbytes);
    CHECK(memcmp(a, c2, sizeof(a)) == 0);
    st_prefetch_rep(&S, "t0", 2); /* smoke: WILLNEED on the second mirror */

    /* a divergent dir claims no replica slot; existing replicas survive */
    CHECK(st_mirror_add(&S, DIR_C) == 0);
    CHECK(S.nrep == 2);
    CHECK(st_fd_rep(&S, t->fd, 1) == f1 && st_fd_rep(&S, t->fd, 2) == f2);
    CHECK(st_fd_rep(&S, t->fd, 3) == -1);

    /* reset drops every replica */
    st_mirror_reset(&S);
    CHECK(S.nrep == 0);
    CHECK(st_fd_rep(&S, t->fd, 1) == -1);

    cleanup();
    puts("safetensors mirror tests: ok");
    return 0;
}
