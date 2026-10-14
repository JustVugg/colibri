/* own_file.h: the engine's files beside a model are opened only as regular files
 * of their own. A symlink planted under one of their names (a .coli_kv pointing
 * at a file elsewhere) must be refused without touching its target. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../own_file.h"

#ifdef _WIN32
#include <direct.h>
#include <process.h>
#define mkdir_(p) _mkdir(p)
#define pid_() _getpid()
#else
#include <unistd.h>
#define mkdir_(p) mkdir((p), 0700)
#define pid_() getpid()
#endif

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static char dir[512];

static const char *at(const char *name){
    static char path[4][600]; static int k;
    k = (k + 1) & 3;
    snprintf(path[k], sizeof path[k], "%s/%s", dir, name);
    return path[k];
}

#ifndef _WIN32
static int put(const char *path, const char *text){
    FILE *f = fopen(path, "wb");
    if(!f) return -1;
    fputs(text, f);
    return fclose(f);
}
#endif

/* the file's bytes, or "" when it cannot be read */
static const char *get(const char *path){
    static char buf[256];
    FILE *f = fopen(path, "rb");
    buf[0] = 0;
    if(f){ size_t n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0; fclose(f); }
    return buf;
}

int main(void){
    const char *base = getenv("TMPDIR");
#ifdef _WIN32
    if(!base) base = getenv("TEMP");
#endif
    if(!base) base = "/tmp";
    snprintf(dir, sizeof dir, "%s/own_file_%ld", base, (long)pid_());
    CHECK(mkdir_(dir) == 0);
    FILE *f;

    /* a name that does not exist is created; "wb" truncates a file of its own */
    CHECK((f = coli_own_fopen(at(".coli_kv"), "wb")) != NULL);
    CHECK(fputs("first and longer", f) >= 0 && fclose(f) == 0);
    CHECK(!strcmp(get(at(".coli_kv")), "first and longer"));
    CHECK((f = coli_own_fopen(at(".coli_kv"), "wb")) != NULL);
    CHECK(fputs("second", f) >= 0 && fclose(f) == 0);
    CHECK(!strcmp(get(at(".coli_kv")), "second"));

    /* "r+b" keeps the bytes and writes in place; "rb" reads */
    CHECK((f = coli_own_fopen(at(".coli_kv"), "r+b")) != NULL);
    CHECK(fseek(f, 0, SEEK_SET) == 0 && fputc('S', f) == 'S' && fclose(f) == 0);
    CHECK(!strcmp(get(at(".coli_kv")), "Second"));
    CHECK((f = coli_own_fopen(at(".coli_kv"), "rb")) != NULL);
    char line[16] = {0};
    CHECK(fread(line, 1, sizeof line - 1, f) == 6 && !strcmp(line, "Second"));
    fclose(f);

    /* "w" (text) as the history uses it */
    CHECK((f = coli_own_fopen(at(".coli_usage.tmp"), "w")) != NULL);
    CHECK(fprintf(f, "0 1 2\n") > 0 && fclose(f) == 0);

    /* a missing file is not created by a read mode */
    errno = 0;
    CHECK(coli_own_fopen(at("absent"), "rb") == NULL && errno == ENOENT);
    CHECK(coli_own_fopen(at("absent"), "r+b") == NULL);
    CHECK(!coli_own_is_link(at("absent")));

    /* a directory under the name is refused in every mode */
    CHECK(mkdir_(at("sub")) == 0);
    CHECK(coli_own_fopen(at("sub"), "rb") == NULL);
    CHECK(coli_own_fopen(at("sub"), "wb") == NULL);
    CHECK(!coli_own_is_link(at("sub")));

    /* an unknown mode is refused */
    errno = 0;
    CHECK(coli_own_fopen(at(".coli_kv"), "ab") == NULL && errno == EINVAL);

#ifndef _WIN32
    /* a planted symlink is never followed: every mode fails, the target keeps its
     * bytes, and a dangling link creates nothing at its target */
    CHECK(put(at("victim"), "keep me") == 0);
    CHECK(symlink(at("victim"), at("link")) == 0);
    CHECK(coli_own_is_link(at("link")));
    const char *modes[] = {"rb", "r+b", "wb", "w", "w+b"};
    for(size_t i = 0; i < sizeof modes / sizeof *modes; i++){
        errno = 0;
        CHECK(coli_own_fopen(at("link"), modes[i]) == NULL && errno == ELOOP);
    }
    CHECK(!strcmp(get(at("victim")), "keep me"));
    CHECK(symlink(at("not-yet"), at("dangling")) == 0);
    CHECK(coli_own_fopen(at("dangling"), "wb") == NULL);
    struct stat st;
    CHECK(stat(at("not-yet"), &st) != 0 && errno == ENOENT);

    /* a symlinked directory is reported, and a link to a directory refused */
    CHECK(symlink(at("sub"), at("sublink")) == 0);
    CHECK(coli_own_is_link(at("sublink")));
    CHECK(coli_own_fopen(at("sublink"), "wb") == NULL);

    /* a FIFO under the name is refused at once, in a write mode too (no reader) */
    CHECK(mkfifo(at("fifo"), 0600) == 0);
    CHECK(coli_own_fopen(at("fifo"), "rb") == NULL && errno == EINVAL);
    CHECK(coli_own_fopen(at("fifo"), "wb") == NULL);
    CHECK(coli_own_fopen(at("fifo"), "r+b") == NULL);

    /* the descriptor is blocking again once accepted */
    CHECK((f = coli_own_fopen(at(".coli_kv"), "rb")) != NULL);
    {
        int fl = fcntl(fileno(f), F_GETFL);
        CHECK(fl != -1 && !(fl & O_NONBLOCK));
    }
    fclose(f);

    unlink(at("fifo")); unlink(at("sublink")); unlink(at("dangling"));
    unlink(at("link")); unlink(at("victim"));
#endif
    remove(at(".coli_usage.tmp")); remove(at(".coli_kv"));
#ifdef _WIN32
    _rmdir(at("sub")); _rmdir(dir);
#else
    rmdir(at("sub")); rmdir(dir);
#endif
    printf("own_file: OK\n");
    return 0;
}
