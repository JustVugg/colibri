/* gguf.h -- read a GGUF model in place.
 *
 * Parses the header of a GGUF file (or of every part of a split model,
 * NAME-00001-of-000NN.gguf) into a metadata table and a tensor table, and
 * reads tensor bytes straight from the file with pread. Nothing is mapped and
 * nothing is copied: the engine asks for the rows it needs, where they are.
 *
 * The header is untrusted input (files come from any mirror), so every count,
 * length and offset is checked against the file size and against overflow
 * before it is used; a malformed file is refused with a message, never read
 * past its end.
 *
 * Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */
#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "ggml_quants.h"

enum { GGUF_T_U8 = 0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16, GGUF_T_U32, GGUF_T_I32,
       GGUF_T_F32, GGUF_T_BOOL, GGUF_T_STRING, GGUF_T_ARRAY, GGUF_T_U64, GGUF_T_I64,
       GGUF_T_F64, GGUF_T_COUNT };

typedef struct {
    char *key;
    int type;                 /* GGUF_T_* */
    union { uint64_t u; int64_t i; double f; } v;   /* scalar value */
    char *s;                  /* GGUF_T_STRING */
    int atype;                /* GGUF_T_ARRAY: element type */
    uint64_t n;               /* GGUF_T_ARRAY: element count */
    void *arr;                /* numeric array elements, native width */
    char **strs;              /* string array elements */
} gguf_kv;

typedef struct {
    char *name;
    int type;                 /* ggml type (GQ_*) */
    int n_dims;
    int64_t ne[4];            /* ggml order: ne[0] is the contiguous (row) length */
    int file;                 /* which part holds it */
    uint64_t off;             /* absolute offset in that part */
    int64_t nbytes;
    int used;                 /* set by the loader that consumed it (see gguf_mark) */
} gguf_tensor;

typedef struct {
    int n_files;
    int *fds;
    char **paths;
    int n_kv; gguf_kv *kv;
    int n_tensors; gguf_tensor *t;
    /* open-addressing name index over t[] */
    int *idx; int idx_cap;
    char err[512];
} gguf_ctx;

/* Header-only mode (GGUF_CHECK): the file is the header of a model fetched
 * with an HTTP range request, so tensor data is legitimately absent. Offsets
 * and sizes are still parsed and checked for alignment and overflow; only the
 * "lies inside the file" check is skipped. Reading a tensor then fails. */
static int gguf_header_only;

/* ---------------- header reader ---------------- */
typedef struct { FILE *f; uint64_t pos, size; int bad; char *err; } gguf_rd;

static int gguf_rd_bytes(gguf_rd *r, void *dst, uint64_t n) {
    if (r->bad) return -1;
    if (n > r->size - r->pos || r->pos > r->size) { r->bad = 1; snprintf(r->err, 512, "truncated header at byte %llu", (unsigned long long)r->pos); return -1; }
    if (n && fread(dst, 1, (size_t)n, r->f) != (size_t)n) { r->bad = 1; snprintf(r->err, 512, "read error at byte %llu", (unsigned long long)r->pos); return -1; }
    r->pos += n; return 0;
}
static uint64_t gguf_rd_u64(gguf_rd *r) { uint64_t v = 0; gguf_rd_bytes(r, &v, 8); return v; }
static uint32_t gguf_rd_u32(gguf_rd *r) { uint32_t v = 0; gguf_rd_bytes(r, &v, 4); return v; }
static char *gguf_rd_str(gguf_rd *r) {
    uint64_t n = gguf_rd_u64(r);
    if (r->bad) return NULL;
    if (n > r->size - r->pos || n > (1u << 30)) { r->bad = 1; snprintf(r->err, 512, "string of %llu bytes at %llu exceeds the file", (unsigned long long)n, (unsigned long long)r->pos); return NULL; }
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) { r->bad = 1; snprintf(r->err, 512, "out of memory"); return NULL; }
    if (gguf_rd_bytes(r, s, n)) { free(s); return NULL; }
    s[n] = 0; return s;
}
static int gguf_scalar_size(int t) {
    switch (t) {
    case GGUF_T_U8: case GGUF_T_I8: case GGUF_T_BOOL: return 1;
    case GGUF_T_U16: case GGUF_T_I16: return 2;
    case GGUF_T_U32: case GGUF_T_I32: case GGUF_T_F32: return 4;
    case GGUF_T_U64: case GGUF_T_I64: case GGUF_T_F64: return 8;
    default: return 0;
    }
}
static void gguf_scalar_decode(int t, const void *p, gguf_kv *kv) {
    switch (t) {
    case GGUF_T_U8: case GGUF_T_BOOL: kv->v.u = *(const uint8_t *)p; kv->v.i = (int64_t)kv->v.u; break;
    case GGUF_T_I8:  kv->v.i = *(const int8_t *)p; break;
    case GGUF_T_U16: { uint16_t x; memcpy(&x, p, 2); kv->v.u = x; break; }
    case GGUF_T_I16: { int16_t x; memcpy(&x, p, 2); kv->v.i = x; break; }
    case GGUF_T_U32: { uint32_t x; memcpy(&x, p, 4); kv->v.u = x; break; }
    case GGUF_T_I32: { int32_t x; memcpy(&x, p, 4); kv->v.i = x; break; }
    case GGUF_T_F32: { float x; memcpy(&x, p, 4); kv->v.f = x; break; }
    case GGUF_T_U64: memcpy(&kv->v.u, p, 8); break;
    case GGUF_T_I64: memcpy(&kv->v.i, p, 8); break;
    case GGUF_T_F64: memcpy(&kv->v.f, p, 8); break;
    }
}

static uint64_t gguf_hash(const char *s) { uint64_t h = 1469598103934665603ull; while (*s) { h ^= (uint8_t)*s++; h *= 1099511628211ull; } return h; }

static int gguf_find_idx(const gguf_ctx *g, const char *name) {
    if (!g->idx_cap) return -1;
    uint64_t h = gguf_hash(name);
    for (int k = 0; k < g->idx_cap; k++) {
        int slot = g->idx[(h + (uint64_t)k) % (uint64_t)g->idx_cap];
        if (slot < 0) return -1;
        if (!strcmp(g->t[slot].name, name)) return slot;
    }
    return -1;
}
static const gguf_tensor *gguf_find(const gguf_ctx *g, const char *name) {
    int i = gguf_find_idx(g, name); return i < 0 ? NULL : &g->t[i];
}
/* Find and record that the caller consumed it: an engine can then name every
 * tensor of the file it did NOT use, instead of silently ignoring it. */
static const gguf_tensor *gguf_mark(const gguf_ctx *g, const char *name) {
    const gguf_tensor *t = gguf_find(g, name);
    if (t) ((gguf_tensor *)t)->used = 1;
    return t;
}
static const gguf_kv *gguf_get(const gguf_ctx *g, const char *key) {
    for (int i = 0; i < g->n_kv; i++) if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}
/* Integer/float/string getters: `def` when the key is absent or of the wrong kind. */
static int64_t gguf_get_int(const gguf_ctx *g, const char *key, int64_t def) {
    const gguf_kv *k = gguf_get(g, key);
    if (!k) return def;
    switch (k->type) {
    case GGUF_T_U8: case GGUF_T_U16: case GGUF_T_U32: case GGUF_T_U64: case GGUF_T_BOOL: return (int64_t)k->v.u;
    case GGUF_T_I8: case GGUF_T_I16: case GGUF_T_I32: case GGUF_T_I64: return k->v.i;
    default: return def;
    }
}
static double gguf_get_float(const gguf_ctx *g, const char *key, double def) {
    const gguf_kv *k = gguf_get(g, key);
    if (!k) return def;
    if (k->type == GGUF_T_F32 || k->type == GGUF_T_F64) return k->v.f;
    if (k->type == GGUF_T_ARRAY || k->type == GGUF_T_STRING) return def;
    return (double)gguf_get_int(g, key, (int64_t)def);
}
static const char *gguf_get_str(const gguf_ctx *g, const char *key) {
    const gguf_kv *k = gguf_get(g, key); return (k && k->type == GGUF_T_STRING) ? k->s : NULL;
}

static int gguf_parse_one(gguf_ctx *g, int part, const char *path, int want_kv) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(g->err, sizeof g->err, "%s: %s", path, strerror(errno)); return -1; }
    struct stat st; if (fstat(fileno(f), &st)) { fclose(f); snprintf(g->err, sizeof g->err, "%s: stat failed", path); return -1; }
    gguf_rd r = { f, 0, (uint64_t)st.st_size, 0, g->err };
    char magic[4];
    if (gguf_rd_bytes(&r, magic, 4) || memcmp(magic, "GGUF", 4)) { fclose(f); if (!r.bad) snprintf(g->err, sizeof g->err, "%s: not a GGUF file", path); return -1; }
    uint32_t ver = gguf_rd_u32(&r);
    if (!r.bad && (ver < 2 || ver > 3)) { fclose(f); snprintf(g->err, sizeof g->err, "%s: GGUF version %u is not supported (2 or 3)", path, ver); return -1; }
    uint64_t nt = gguf_rd_u64(&r), nkv = gguf_rd_u64(&r);
    if (r.bad || nt > (1u << 24) || nkv > (1u << 20)) { fclose(f); if (!r.bad) snprintf(g->err, sizeof g->err, "%s: implausible header counts (%llu tensors, %llu keys)", path, (unsigned long long)nt, (unsigned long long)nkv); return -1; }
    uint64_t alignment = 32;
    for (uint64_t k = 0; k < nkv && !r.bad; k++) {
        gguf_kv kv; memset(&kv, 0, sizeof kv);
        kv.key = gguf_rd_str(&r);
        kv.type = (int)gguf_rd_u32(&r);
        if (r.bad) { free(kv.key); break; }
        if (kv.type == GGUF_T_STRING) kv.s = gguf_rd_str(&r);
        else if (kv.type == GGUF_T_ARRAY) {
            kv.atype = (int)gguf_rd_u32(&r); kv.n = gguf_rd_u64(&r);
            if (r.bad) { free(kv.key); break; }
            if (kv.atype == GGUF_T_STRING) {
                if (kv.n > (r.size - r.pos) / 8) { r.bad = 1; snprintf(g->err, sizeof g->err, "%s: string array %s too long", path, kv.key); free(kv.key); break; }
                kv.strs = (char **)calloc((size_t)kv.n + 1, sizeof(char *));
                if (!kv.strs) { r.bad = 1; snprintf(g->err, sizeof g->err, "out of memory"); free(kv.key); break; }
                for (uint64_t j = 0; j < kv.n && !r.bad; j++) kv.strs[j] = gguf_rd_str(&r);
            } else {
                int es = gguf_scalar_size(kv.atype);
                if (!es || kv.n > (r.size - r.pos) / (uint64_t)es) { r.bad = 1; snprintf(g->err, sizeof g->err, "%s: array %s: bad element type %d or length", path, kv.key, kv.atype); free(kv.key); break; }
                kv.arr = malloc((size_t)(kv.n * (uint64_t)es) + 1);
                if (!kv.arr) { r.bad = 1; snprintf(g->err, sizeof g->err, "out of memory"); free(kv.key); break; }
                gguf_rd_bytes(&r, kv.arr, kv.n * (uint64_t)es);
            }
        } else {
            int es = gguf_scalar_size(kv.type);
            if (!es) { r.bad = 1; snprintf(g->err, sizeof g->err, "%s: key %s has unknown type %d", path, kv.key, kv.type); free(kv.key); break; }
            uint8_t buf[8]; if (!gguf_rd_bytes(&r, buf, (uint64_t)es)) gguf_scalar_decode(kv.type, buf, &kv);
        }
        if (!strcmp(kv.key, "general.alignment") && kv.type == GGUF_T_U32) alignment = kv.v.u;
        if (want_kv) {
            gguf_kv *nk = (gguf_kv *)realloc(g->kv, sizeof(gguf_kv) * (size_t)(g->n_kv + 1));
            if (!nk) { r.bad = 1; snprintf(g->err, sizeof g->err, "out of memory"); break; }
            g->kv = nk; g->kv[g->n_kv++] = kv;
        } else {
            free(kv.key); free(kv.s); free(kv.arr);
            if (kv.strs) { for (uint64_t j = 0; j < kv.n; j++) free(kv.strs[j]); free(kv.strs); }
        }
    }
    if (!r.bad && (alignment == 0 || (alignment & (alignment - 1)) || alignment > (1u << 20))) { r.bad = 1; snprintf(g->err, sizeof g->err, "%s: bad alignment %llu", path, (unsigned long long)alignment); }
    int t0 = g->n_tensors;
    if (!r.bad && nt) {
        gguf_tensor *nt2 = (gguf_tensor *)realloc(g->t, sizeof(gguf_tensor) * (size_t)(g->n_tensors + nt));
        if (!nt2) { r.bad = 1; snprintf(g->err, sizeof g->err, "out of memory"); } else g->t = nt2;
    }
    for (uint64_t k = 0; k < nt && !r.bad; k++) {
        gguf_tensor *t = &g->t[g->n_tensors];
        memset(t, 0, sizeof *t);
        t->name = gguf_rd_str(&r);
        uint32_t nd = gguf_rd_u32(&r);
        if (r.bad) { free(t->name); break; }
        if (nd < 1 || nd > 4) { r.bad = 1; snprintf(g->err, sizeof g->err, "%s: tensor %s has %u dims", path, t->name, nd); free(t->name); break; }
        t->n_dims = (int)nd;
        int64_t numel = 1;
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < nd; d++) {
            uint64_t v = gguf_rd_u64(&r);
            if (v == 0 || v > (1ull << 40) || (numel > 0 && (int64_t)v > INT64_MAX / numel)) { r.bad = 1; snprintf(g->err, sizeof g->err, "%s: tensor %s dimension %u = %llu", path, t->name, d, (unsigned long long)v); break; }
            t->ne[d] = (int64_t)v; numel *= (int64_t)v;
        }
        t->type = (int)gguf_rd_u32(&r);
        uint64_t rel = gguf_rd_u64(&r);
        if (r.bad) { free(t->name); break; }
        int64_t rb = gq_row_bytes(t->type, t->ne[0]);
        if (rb < 0) { r.bad = 1; snprintf(g->err, sizeof g->err, "%s: tensor %s: type %d (%s) cannot hold rows of %lld", path, t->name, t->type, gq_type_name(t->type), (long long)t->ne[0]); free(t->name); break; }
        t->nbytes = rb * (numel / t->ne[0]);
        t->file = part; t->off = rel;          /* made absolute below */
        g->n_tensors++;
    }
    if (!r.bad) {
        uint64_t data = (r.pos + alignment - 1) / alignment * alignment;
        for (int i = t0; i < g->n_tensors; i++) {
            gguf_tensor *t = &g->t[i];
            if (t->off % alignment || (!gguf_header_only && (t->off > r.size || data > r.size - t->off || (uint64_t)t->nbytes > r.size - data - t->off))) {
                r.bad = 1; snprintf(g->err, sizeof g->err, "%s: tensor %s lies outside the file (offset %llu, %lld bytes)", path, t->name, (unsigned long long)t->off, (long long)t->nbytes); break;
            }
            t->off += data;
        }
    }
    fclose(f);
    return r.bad ? -1 : 0;
}

static void gguf_close(gguf_ctx *g) {
    if (!g) return;
    for (int i = 0; i < g->n_files; i++) { if (g->fds && g->fds[i] >= 0) close(g->fds[i]); if (g->paths) free(g->paths[i]); }
    free(g->fds); free(g->paths);
    for (int i = 0; i < g->n_kv; i++) {
        gguf_kv *k = &g->kv[i]; free(k->key); free(k->s); free(k->arr);
        if (k->strs) { for (uint64_t j = 0; j < k->n; j++) free(k->strs[j]); free(k->strs); }
    }
    free(g->kv);
    for (int i = 0; i < g->n_tensors; i++) free(g->t[i].name);
    free(g->t); free(g->idx);
    char err[sizeof g->err]; memcpy(err, g->err, sizeof err);   /* gguf_open's reason outlives the close */
    memset(g, 0, sizeof *g); memcpy(g->err, err, sizeof err);
}

/* 1 if path names a GGUF file (by magic), 0 otherwise. */
static int gguf_is_file(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    char m[4]; int ok = fread(m, 1, 4, f) == 4 && !memcmp(m, "GGUF", 4);
    fclose(f); return ok;
}

/* Open a GGUF model: a single file, or the first part of a split model
 * (NAME-00001-of-000NN.gguf), in which case every part is opened. */
static int gguf_open(gguf_ctx *g, const char *path) {
    memset(g, 0, sizeof *g);
    const char *filename = strrchr(path, '/');
    filename = filename ? filename + 1 : path;
#ifdef _WIN32
    const char *backslash = strrchr(filename, '\\');
    if (backslash) filename = backslash + 1;
#endif
    size_t flen = strlen(filename);
    const char *of = flen >= 20 ? filename + flen - 14 : NULL;
    int parts = 1;
    size_t plen = strlen(path);
    if (of && !strncmp(of, "-of-", 4) && of - filename >= 6 &&
        plen >= 5 && !strcmp(path + plen - 5, ".gguf")) {
        const char *num = of - 5;
        int a = atoi(num), b = atoi(of + 4);
        if (num[-1] == '-' && a == 1 && b >= 1 && b <= 9999) parts = b;
    }
    g->n_files = parts;
    g->fds = (int *)malloc(sizeof(int) * (size_t)parts);
    g->paths = (char **)calloc((size_t)parts, sizeof(char *));
    if (!g->fds || !g->paths) { snprintf(g->err, sizeof g->err, "out of memory"); gguf_close(g); return -1; }
    for (int p = 0; p < parts; p++) g->fds[p] = -1;
    for (int p = 0; p < parts; p++) {
        char *pp = strdup(path);
        if (!pp) { snprintf(g->err, sizeof g->err, "out of memory"); gguf_close(g); return -1; }
        if (parts > 1) { char *n = pp + ((of - 5) - path); char tmp[16]; snprintf(tmp, sizeof tmp, "%05d", (p + 1) % 100000); memcpy(n, tmp, 5); }
        g->paths[p] = pp;
        if (gguf_parse_one(g, p, pp, p == 0)) { gguf_close(g); return -1; }
#ifdef _WIN32
        g->fds[p] = open(pp, O_RDONLY | O_BINARY);
#else
        g->fds[p] = open(pp, O_RDONLY);
#endif
        if (g->fds[p] < 0) { snprintf(g->err, sizeof g->err, "%s: %s", pp, strerror(errno)); gguf_close(g); return -1; }
    }
    g->idx_cap = g->n_tensors * 2 + 1;
    g->idx = (int *)malloc(sizeof(int) * (size_t)g->idx_cap);
    if (!g->idx) { snprintf(g->err, sizeof g->err, "out of memory"); gguf_close(g); return -1; }
    for (int i = 0; i < g->idx_cap; i++) g->idx[i] = -1;
    for (int i = 0; i < g->n_tensors; i++) {
        if (gguf_find_idx(g, g->t[i].name) >= 0) { snprintf(g->err, sizeof g->err, "duplicate tensor %s", g->t[i].name); gguf_close(g); return -1; }
        uint64_t h = gguf_hash(g->t[i].name);
        for (int k = 0; k < g->idx_cap; k++) { int s = (int)((h + (uint64_t)k) % (uint64_t)g->idx_cap); if (g->idx[s] < 0) { g->idx[s] = i; break; } }
    }
    return 0;
}

/* Read n bytes at byte `off` inside tensor t into dst. 0 or -1. */
static int gguf_read(const gguf_ctx *g, const gguf_tensor *t, void *dst, int64_t off, int64_t n) {
    if (off < 0 || n < 0 || off > t->nbytes || n > t->nbytes - off) return -1;
    uint8_t *p = (uint8_t *)dst; uint64_t pos = t->off + (uint64_t)off; int64_t left = n;
    while (left > 0) {
#ifdef _WIN32   /* positional, as compat_pread: _lseeki64 + _read share the seek and race between threads */
        HANDLE h = (HANDLE)_get_osfhandle(g->fds[t->file]);
        if (h == INVALID_HANDLE_VALUE) return -1;
        DWORD want = left > (1 << 30) ? (DWORD)(1 << 30) : (DWORD)left, got = 0;
        OVERLAPPED ov; memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)(pos & 0xFFFFFFFFu); ov.OffsetHigh = (DWORD)(pos >> 32);
        if (!ReadFile(h, p, want, &got, &ov)) return -1;
        int64_t r = (int64_t)got;
#else
        ssize_t r = pread(g->fds[t->file], p, (size_t)(left > (1 << 30) ? (1 << 30) : left), (off_t)pos);
#endif
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; }
        p += r; pos += (uint64_t)r; left -= r;
    }
    return 0;
}

/* Whole tensor to float32 (any supported type). dst holds numel floats. */
static int gguf_read_f32(const gguf_ctx *g, const gguf_tensor *t, float *dst) {
    int64_t numel = t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    if (t->type == GQ_F32) return gguf_read(g, t, dst, 0, t->nbytes);
    uint8_t *raw = (uint8_t *)malloc((size_t)t->nbytes);
    if (!raw) return -1;
    int rc = gguf_read(g, t, raw, 0, t->nbytes);
    if (!rc) {
        int64_t rows = numel / t->ne[0], rb = gq_row_bytes(t->type, t->ne[0]);
        for (int64_t r = 0; r < rows && !rc; r++) rc = gq_dequant_row(t->type, raw + r * rb, dst + r * t->ne[0], t->ne[0]);
    }
    free(raw); return rc;
}

#endif /* GGUF_H */
