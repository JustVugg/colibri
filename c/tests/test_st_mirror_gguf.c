/* A GGUF's mirror (st_mirror_add on a GGUF primary). The safetensors check compares the header's
 * JSON, and a GGUF's first eight bytes are its magic and version, not a length, so a byte-identical
 * copy of a GGUF part was refused ("header differs") and the engine read the primary drive only.
 * A GGUF copy is accepted when it has the primary's size, the same bytes up to where the first
 * tensor starts (the key/value table and the tensor table), and the same 64 KiB at the tensor
 * data's start, end and 14 places between. Anything else stays on the primary. Fixtures are
 * created in the current directory and removed on exit. */
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

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

#define HEAD  (256 << 10)             /* where the first tensor starts: past the first sample, so the head check stands alone */
#define DATA  (3 << 20)               /* tensor data */
#define NDIR  7
static const char *dirs[NDIR] = {"tmp_gm_a", "tmp_gm_b", "tmp_gm_c", "tmp_gm_d", "tmp_gm_e", "tmp_gm_f", "tmp_gm_g"};
#define FILE_NAME "model-00001-of-00001.gguf"

/* a GGUF-looking part: magic, version 3, the rest of the head and the data pseudo-random.
 * flip_at >= 0 flips one byte there; pad grows the file. */
static int write_part(const char *dir, long flip_at, int pad) {
    char path[256]; snprintf(path, sizeof path, "%s/%s", dir, FILE_NAME);
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); return -1; }
    uint8_t *b = malloc(HEAD + DATA); if (!b) return -1;
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < HEAD + DATA; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; b[i] = (uint8_t)x; }
    memcpy(b, "GGUF", 4); b[4] = 3; b[5] = b[6] = b[7] = 0;
    if (flip_at >= 0) b[flip_at] ^= 0x5a;
    fwrite(b, 1, HEAD + DATA, f);
    for (int i = 0; i < pad; i++) fputc(0, f);
    fclose(f); free(b); return 0;
}
static void cleanup(void) {
    for (int i = 0; i < NDIR; i++) { char p[256]; snprintf(p, sizeof p, "%s/%s", dirs[i], FILE_NAME); remove(p); RMDIR(dirs[i]); }
}

int main(void) {
    cleanup();
    for (int i = 0; i < NDIR; i++) CHECK(MKDIR(dirs[i]) == 0);
    CHECK(write_part(dirs[0], -1, 0) == 0);          /* primary */
    CHECK(write_part(dirs[1], -1, 0) == 0);          /* identical copy */
    CHECK(write_part(dirs[2], 100000, 0) == 0);      /* a byte flipped in the key/value table, between two samples */
    CHECK(write_part(dirs[3], HEAD + DATA - 1, 0) == 0);   /* a byte flipped in the data's last 64 KiB */
    CHECK(write_part(dirs[4], -1, 64) == 0);         /* a different size */
    CHECK(write_part(dirs[6], HEAD, 0) == 0);        /* a byte flipped at the data's start */
    /* dirs[5] stays empty: a missing file */

    /* the primary as an engine's index holds it: one part, one tensor at HEAD */
    shards S; memset(&S, 0, sizeof S);
    char path[256]; snprintf(path, sizeof path, "%s/%s", dirs[0], FILE_NAME);
    int fd = open(path, O_RDONLY); CHECK(fd >= 0);
    S.nfd = 1; S.fds[0] = fd; S.dfds[0] = -2; S.paths[0] = strdup(path);
    S.sizes[0] = lseek(fd, 0, SEEK_END);
    S.cap = 1; S.n = 1; S.t = calloc(1, sizeof(st_tensor));
    S.t[0].name = strdup("t0"); S.t[0].fd = fd; S.t[0].off = HEAD; S.t[0].nbytes = DATA;

    CHECK(st_mirror_add(&S, dirs[1]) == 1);          /* identical: accepted */
    CHECK(S.nrep == 1);
    int mfd = st_fd_rep(&S, fd, 1);
    CHECK(mfd >= 0 && mfd != fd);
    uint8_t a[64], b[64];
    CHECK(pread(fd, a, 64, HEAD + 12345) == 64 && pread(mfd, b, 64, HEAD + 12345) == 64 && !memcmp(a, b, 64));
    st_mirror_reset(&S);

    CHECK(st_mirror_add(&S, dirs[2]) == 0 && S.nrep == 0);   /* head differs: refused */
    CHECK(st_mirror_add(&S, dirs[3]) == 0 && S.nrep == 0);   /* data's end differs: refused */
    CHECK(st_mirror_add(&S, dirs[6]) == 0 && S.nrep == 0);   /* data's start differs: refused */
    CHECK(st_mirror_add(&S, dirs[4]) == 0 && S.nrep == 0);   /* size differs: refused */
    CHECK(st_mirror_add(&S, dirs[5]) == 0 && S.nrep == 0);   /* missing: refused */
    CHECK(st_fd_rep(&S, fd, 1) == -1);

    /* two identical copies are two replicas, as for safetensors */
    CHECK(write_part(dirs[5], -1, 0) == 0);
    CHECK(st_mirror_add(&S, dirs[1]) == 1 && st_mirror_add(&S, dirs[5]) == 1 && S.nrep == 2);

    cleanup();
    puts("GGUF mirror tests: ok");
    return 0;
}
