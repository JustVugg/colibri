/* Shared by the size-trust tests: write a one-shard safetensors container whose
 * tensors declare exactly the dtype, shape and byte count the case needs, then
 * run each case in a child and inspect how it ended. */
#ifndef ST_FIXTURE_H
#define ST_FIXTURE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/wait.h>
typedef struct { const char *name, *dtype; int64_t numel, nbytes; } FxT;
static int fx_write(const char *dir, const FxT *t, int n){
    char hdr[8192], path[512]; int len = 0; int64_t off = 0;
    len += snprintf(hdr + len, sizeof hdr - len, "{");
    for (int i = 0; i < n; i++, off += t[i - 1].nbytes)
        len += snprintf(hdr + len, sizeof hdr - len,
                        "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
                        i ? "," : "", t[i].name, t[i].dtype, (long long)t[i].numel,
                        (long long)off, (long long)(off + t[i].nbytes));
    len += snprintf(hdr + len, sizeof hdr - len, "}");
    while (len % 8) hdr[len++] = ' ';
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    uint64_t hl = (uint64_t)len; fwrite(&hl, 8, 1, f); fwrite(hdr, 1, (size_t)len, f);
    for (int64_t i = 0; i < off; i++) fputc(0x3c, f);   /* 0x3c3c3c3c = 0.0115f */
    return fclose(f);
}
/* Run `self --child <c> <dir>`; returns the exit code and fills `log` with stderr. */
static int fx_child(const char *self, int c, const char *dir, char *log, size_t cap){
    char err[600], cmd[1600];
    snprintf(err, sizeof err, "%s/stderr.txt", dir);
    snprintf(cmd, sizeof cmd, "\"%s\" --child %d \"%s\" >/dev/null 2>\"%s\"", self, c, dir, err);
    int st = system(cmd), got = st >= 0 && WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    log[0] = 0; FILE *f = fopen(err, "rb");
    if (f) { size_t r = fread(log, 1, cap - 1, f); log[r] = 0; fclose(f); }
    remove(err);
    return got;
}
static void fx_cleanup(const char *dir){
    char p[600]; snprintf(p, sizeof p, "%s/model.safetensors", dir); remove(p); rmdir(dir);
}
/* A refusal is exit 1 naming the tensor, with no sanitizer report: ASan also
 * exits 1, so the exit code alone cannot tell a refusal from a heap overflow. */
static int fx_refused(int got, const char *log, const char *tensor){
    return got == 1 && strstr(log, tensor) && strstr(log, "refusing") &&
           !strstr(log, "Sanitizer");
}
#endif
