/* canonical.qc: the INT8 dense weights of a Qwen3.6 model, converted once.
 *
 * The file holds every matrix the engine keeps in row-scaled INT8 (attention
 * and DeltaNet projections, routers, shared experts and lm_head), with the
 * bytes the engine computes when it converts them at load. The engine maps
 * it and takes the matrices from it instead of converting them at start.
 *
 * Layout, in the host's byte order (little-endian on every target Colibri
 * builds for):
 *   header  "QWEN36Q8", u32 version, u32 count, u64 fingerprint, u64 bytes
 *   table   count entries: name[104], i32 in, i32 out, u64 q, u64 scales
 *   data    each matrix on a 16 KiB boundary: int8 [out][in], then float [out]
 * The offsets follow from the list of matrices alone (qdf_layout). The
 * fingerprint covers that list, QDF_QUANTIZER and the first and last 128
 * bytes of every source matrix: a file made from other weights (a fine-tune
 * of the same shape too) or for another quantizer is refused instead of used.
 * It samples the source, it is not a checksum: an edit confined to the middle
 * of a matrix is not seen, and the INT8 rows of the file are not verified.
 * qwen36-canonical writes the file. */
#ifndef COLI_QWEN36_DENSE_FILE_H
#define COLI_QWEN36_DENSE_FILE_H
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "compat.h"

#define QDF_MAGIC "QWEN36Q8"
#define QDF_VERSION 2u
#define QDF_ALIGN ((uint64_t)16384)
#define QDF_NAME 104
#define QDF_SEED 0xcbf29ce484222325ull
/* The quantizer whose bytes the file holds (the engine's qw_quantize: per row,
 * scale = max|w| / 127, or 1 for a row below 1e-12, values lrintf-rounded and
 * clamped to +-127). Part of the fingerprint: another quantizer means another
 * file, so it changes whenever qw_quantize does (tests/test_qwen36_dense_file.c
 * fails until it does). */
#define QDF_QUANTIZER "rows-amax127-lrintf-v1"

typedef struct { char magic[8]; uint32_t version, count; uint64_t fingerprint, bytes; } QdfHeader;
typedef struct { char name[QDF_NAME]; int32_t I, O; uint64_t q, sc; } QdfEntry;
typedef struct { const char *name; int I, O; } QdfSpec;
typedef struct { compat_ro_map map; const unsigned char *base; size_t bytes; int count; } QdfFile;
_Static_assert(sizeof(QdfHeader) == 32 && sizeof(QdfEntry) == 128, "canonical.qc layout");

/* FNV-1a, 64 bits: catches a wrong file, not an attack. */
static inline uint64_t qdf_hash(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

static inline uint64_t qdf_round(uint64_t n, uint64_t to) { return (n + to - 1) / to * to; }

static inline int qdf_finite(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

/* The table of a file holding these matrices; returns the file size, or 0 for
 * a name that does not fit or an empty shape. */
static inline uint64_t qdf_layout(const QdfSpec *s, int n, QdfEntry *e) {
    uint64_t at = qdf_round(sizeof(QdfHeader) + (uint64_t)n * sizeof(QdfEntry), QDF_ALIGN);
    for (int i = 0; i < n; i++) {
        if (strlen(s[i].name) >= QDF_NAME || s[i].I <= 0 || s[i].O <= 0) return 0;
        memset(&e[i], 0, sizeof(e[i]));
        memcpy(e[i].name, s[i].name, strlen(s[i].name));
        e[i].I = s[i].I;
        e[i].O = s[i].O;
        e[i].q = at;
        e[i].sc = qdf_round(at + (uint64_t)s[i].I * (uint64_t)s[i].O, 4);
        at = qdf_round(e[i].sc + 4 * (uint64_t)s[i].O, QDF_ALIGN);
    }
    return at;
}

static inline int qdf_error(char *err, size_t cap, const char *fmt, ...) {
    if (err && cap) { va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap); }
    return 0;
}

static inline void qdf_close(QdfFile *f) {
    if (f && f->base) compat_unmap_readonly(&f->map);
    if (f) memset(f, 0, sizeof(*f));
}

/* Maps path and checks it against the matrices the engine expects and the
 * fingerprint of its source weights. Scales must be finite and positive. */
static inline int qdf_open(QdfFile *f, const char *path, const QdfSpec *s, int n, uint64_t fingerprint,
                           char *err, size_t cap) {
    memset(f, 0, sizeof(*f));
    QdfEntry *want = (QdfEntry *)calloc(n > 0 ? (size_t)n : 1, sizeof(*want));
    if (!want) return qdf_error(err, cap, "%s: out of memory", path);
    uint64_t bytes = n > 0 ? qdf_layout(s, n, want) : 0;
    int fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) { free(want); return qdf_error(err, cap, "%s: %s", path, strerror(errno)); }
    struct stat st;
    const void *data = NULL;
    int mapped = !fstat(fd, &st) && st.st_size >= (off_t)sizeof(QdfHeader) &&
                 !compat_map_readonly(fd, 0, (size_t)st.st_size, &f->map, &data);
    close(fd);
    if (!mapped) { free(want); return qdf_error(err, cap, "%s: cannot map the file", path); }
    f->base = (const unsigned char *)data;
    f->bytes = (size_t)st.st_size;
    f->count = n;
    QdfHeader h;
    memcpy(&h, f->base, sizeof(h));
    const char *why = NULL;
    if (memcmp(h.magic, QDF_MAGIC, 8)) why = "is not a canonical.qc file";
    else if (h.version != QDF_VERSION) why = "was written by another version of the converter";
    else if (!bytes || h.count != (uint32_t)n || h.bytes != f->bytes || f->bytes != bytes)
        why = "does not hold the matrices of this model";
    else if (h.fingerprint != fingerprint) why = "was made from other weights or for another quantizer";
    const QdfEntry *table = (const QdfEntry *)(f->base + sizeof(h));
    for (int i = 0; !why && i < n; i++) {
        if (memcmp(&table[i], &want[i], sizeof(QdfEntry))) why = "has a different matrix table";
        const float *sc = (const float *)(f->base + want[i].sc);
        for (int o = 0; !why && o < want[i].O; o++)
            if (!qdf_finite(sc[o]) || !(sc[o] > 0.f)) why = "has an invalid scale";
    }
    free(want);
    if (why) {
        qdf_close(f);
        return qdf_error(err, cap, "%s %s: rebuild it with qwen36-canonical, or remove it", path, why);
    }
    return 1;
}

/* The matrix `name`: its INT8 rows [O][I] and O scales, in the mapping. */
static inline int qdf_find(const QdfFile *f, const char *name, int I, int O,
                           const int8_t **q, const float **sc) {
    const QdfEntry *t = (const QdfEntry *)(f->base + sizeof(QdfHeader));
    for (int i = 0; f->base && i < f->count; i++) {
        if (strcmp(t[i].name, name)) continue;
        if (t[i].I != I || t[i].O != O) return 0;
        *q = (const int8_t *)(f->base + t[i].q);
        *sc = (const float *)(f->base + t[i].sc);
        return 1;
    }
    return 0;
}

/* Header and table of a new file; the writer then puts each matrix at the
 * offsets of its entry and pads the file to `bytes`. */
static inline int qdf_write_header(FILE *out, const QdfEntry *e, int n, uint64_t fingerprint, uint64_t bytes) {
    QdfHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QDF_MAGIC, 8);
    h.version = QDF_VERSION;
    h.count = (uint32_t)n;
    h.fingerprint = fingerprint;
    h.bytes = bytes;
    return fwrite(&h, sizeof(h), 1, out) == 1 && fwrite(e, sizeof(*e), (size_t)n, out) == (size_t)n;
}

/* Zero bytes up to file offset `to` (sequential writer). */
static inline int qdf_pad(FILE *out, uint64_t *at, uint64_t to) {
    static const char zeros[4096];
    while (*at < to) {
        size_t k = to - *at < sizeof(zeros) ? (size_t)(to - *at) : sizeof(zeros);
        if (fwrite(zeros, 1, k, out) != k) return 0;
        *at += k;
    }
    return *at == to;
}
#endif
