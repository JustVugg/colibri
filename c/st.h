/* Indicizzazione e lettura on-demand di tensori da piu' file safetensors.
 * Equivale a Shards in engine.py, ma:
 *   - legge con pread (niente mmap) + posix_fadvise(DONTNEED) -> le pagine NON
 *     restano residenti nel processo. E' la correzione del bug di RSS: cosi' la
 *     RAM di picco resta densa+cache, non l'intero modello. (vedi memoria mmap-rss-bug)
 *   - converte sempre in float32 in uscita (BF16/F16/F32 supportati). */
#ifndef ST_H
#define ST_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include "json.h"
#include "compat.h"

typedef struct {
    char   *name;
    int     fd;
    int64_t off;       /* offset assoluto del dato dentro al file */
    int64_t nbytes;
    int     dtype;     /* 0=BF16 1=F16 2=F32 */
    int64_t numel;
} st_tensor;

typedef struct {
    st_tensor *t;
    int        n, cap;
    int        fds[512];
    int        dfds[512];  /* gemelli O_DIRECT (aperti pigramente): -2 = non ancora provato */
    char      *paths[512];
    int        nfd;
    int       *hidx;      /* hash map nome->indice (open addressing): con ~120k tensori
                           * (GLM: 256 expert x 78 layer x 3 x 2) la scansione lineare
                           * costava decine di secondi/token (misurato sul primo run reale) */
    int        hcap;
} shards;
#define ST_MAX_SHARDS 512

static uint64_t st_hash(const char *s){
    uint64_t h=1469598103934665603ULL;
    while(*s){ h^=(unsigned char)*s++; h*=1099511628211ULL; }
    return h;
}

static int st_dtype_code(const char *s) {
    if (!strcmp(s, "BF16")) return 0;
    if (!strcmp(s, "F16"))  return 1;
    if (!strcmp(s, "F32"))  return 2;
    if (!strcmp(s, "U8"))   return 3;   /* dati quantizzati (int4 packed / int8) */
    if (!strcmp(s, "I8"))   return 3;
    fprintf(stderr, "unsupported dtype: %s\n", s); exit(1);
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {            /* subnormale o zero */
        if (man == 0) u = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; u = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {  /* inf/nan */
        u = sign | 0x7F800000 | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

static int st_open_fd(shards *S, const char *path) {
    for (int i = 0; i < S->nfd; i++) if (!strcmp(S->paths[i], path)) return S->fds[i];
    int fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    S->paths[S->nfd] = strdup(path); S->fds[S->nfd] = fd;
#ifdef O_DIRECT
    S->dfds[S->nfd] = open(path, COMPAT_O_RDONLY | O_DIRECT);   /* eager: lookup poi thread-safe */
#elif defined(__APPLE__)
    S->dfds[S->nfd] = compat_open_direct(path);          /* macOS: F_NOCACHE ~ O_DIRECT */
#else
    S->dfds[S->nfd] = -1;                                /* niente equivalente: solo buffered */
#endif
    S->nfd++;
    return fd;
}

/* fd gemello O_DIRECT dello stesso file (bypassa la page cache: il buffered read su
 * ext4-in-VHDX si strozza a ~0.8 GB/s, O_DIRECT arriva a 2.3+; misurato). -1 se non disponibile. */
static int st_direct_fd(shards *S, int fd) {
    for (int i = 0; i < S->nfd; i++) if (S->fds[i] == fd) return S->dfds[i];
    return -1;
}

/* indicizza tutti i model-*.safetensors in snap_dir */
static void st_init(shards *S, const char *snap_dir) {
    memset(S, 0, sizeof(*S));
    S->cap = 4096; S->t = calloc(S->cap, sizeof(st_tensor));
    /* raccoglie ordinatamente i nomi dei file shard */
    static char files[ST_MAX_SHARDS][1024]; int nf = 0;
    DIR *d = opendir(snap_dir); struct dirent *e;
    if (!d) { perror(snap_dir); exit(1); }
    while ((e = readdir(d))) {
        const char *dot = strrchr(e->d_name, '.');
        if (dot && !strcmp(dot, ".safetensors")) {  /* model.safetensors o model-0000N-of-... */
            if (nf >= ST_MAX_SHARDS) { fprintf(stderr, "too many shards (>%d): raise ST_MAX_SHARDS\n", ST_MAX_SHARDS); exit(1); }
            snprintf(files[nf++], 1024, "%s/%s", snap_dir, e->d_name);
        }
    }
    closedir(d);
    for (int a = 0; a < nf; a++) for (int b = a+1; b < nf; b++)
        if (strcmp(files[a], files[b]) > 0) { char tmp[1024]; strcpy(tmp, files[a]); strcpy(files[a], files[b]); strcpy(files[b], tmp); }

    for (int fi = 0; fi < nf; fi++) {
        int fd = st_open_fd(S, files[fi]);
        uint64_t hlen;
        if (pread(fd, &hlen, 8, 0) != 8) { perror("pread hlen"); exit(1); }
        char *hdr = malloc(hlen + 1);
        if (pread(fd, hdr, hlen, 8) != (ssize_t)hlen) { perror("pread hdr"); exit(1); }
        hdr[hlen] = 0;
        int64_t data_start = 8 + (int64_t)hlen;
        char *arena = NULL;
        jval *root = json_parse(hdr, &arena);
        for (int i = 0; i < root->len; i++) {
            const char *name = root->keys[i];
            if (!strcmp(name, "__metadata__")) continue;
            jval *m = root->kids[i];
            jval *dt = json_get(m, "dtype");
            jval *off = json_get(m, "data_offsets");
            jval *shp = json_get(m, "shape");
            int64_t a0 = (int64_t)off->kids[0]->num, b0 = (int64_t)off->kids[1]->num;
            int64_t numel = 1; for (int k = 0; k < shp->len; k++) numel *= (int64_t)shp->kids[k]->num;
            if (S->n == S->cap) { S->cap *= 2; S->t = realloc(S->t, S->cap*sizeof(st_tensor)); }
            st_tensor *t = &S->t[S->n++];
            t->name = strdup(name); t->fd = fd; t->off = data_start + a0;
            t->nbytes = b0 - a0; t->dtype = st_dtype_code(dt->str); t->numel = numel;
        }
        free(arena); /* i jval restano leakati: ok, una tantum all'avvio */
        free(hdr);
    }
    /* indice hash costruito a fine indicizzazione (gli indici restano validi dopo i realloc) */
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = malloc(S->hcap * sizeof(int));
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t h = st_hash(S->t[i].name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (S->hcap - 1);
        S->hidx[h] = i;
    }
}

static st_tensor *st_find(shards *S, const char *name) {
    if (S->hidx) {
        uint64_t h = st_hash(name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) {
            st_tensor *t = &S->t[S->hidx[h]];
            if (!strcmp(t->name, name)) return t;
            h = (h + 1) & (S->hcap - 1);
        }
        return NULL;
    }
    for (int i = 0; i < S->n; i++) if (!strcmp(S->t[i].name, name)) return &S->t[i];
    return NULL;
}
static int st_has(shards *S, const char *name) { return st_find(S, name) != NULL; }

/* prefetch ASINCRONO: dice al kernel di iniziare a leggere le pagine del tensore in
 * background (readahead). Serve a sovrapporre l'I/O degli expert col calcolo: si
 * prefetcha tutto il set di expert di un layer, poi le pread sincrone trovano la cache
 * gia' calda. No-op se il tensore non esiste (es. il primo .qs prima della lettura). */
static void st_prefetch(shards *S, const char *name) {
    st_tensor *t = st_find(S, name);
    if (t) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* legge un tensore in un buffer float32 fornito dal chiamante (numel float).
 * drop=1 -> consiglia al kernel di scartare le pagine (per gli expert in streaming). */
static int64_t st_read_f32(shards *S, const char *name, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    void *raw = malloc(t->nbytes);
    if (pread(t->fd, raw, t->nbytes, t->off) != t->nbytes) { perror("pread data"); exit(1); }
    if (t->dtype == 2) {
        memcpy(out, raw, t->nbytes);
    } else if (t->dtype == 0) {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = bf16_to_f32(p[i]);
    } else {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = f16_to_f32(p[i]);
    }
    free(raw);
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    return t->numel;
}

static int64_t st_numel(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->numel : -1;
}
static int64_t st_nbytes(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->nbytes : -1;
}

/* legge i byte GREZZI di un tensore (nessuna conversione di dtype): per i pesi gia'
 * quantizzati int4/int8 del nostro container (dtype U8). drop=1 -> fadvise DONTNEED. */
static void st_read_raw(shards *S, const char *name, void *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (pread(t->fd, out, t->nbytes, t->off) != t->nbytes) { perror("pread raw"); exit(1); }
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
}

/* legge una FETTA di un tensore: n_elems a partire dall'elemento elem_off.
 * Serve per gli expert fusi di GLM (un tensore = blocco [E, ...]): si legge il
 * solo expert richiesto via pread del sotto-range, niente lettura dell'intero blocco. */
static void st_read_slice_f32(shards *S, const char *name, int64_t elem_off, int64_t n_elems, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    int64_t boff = t->off + elem_off * esz, nb = n_elems * esz;
    void *raw = malloc(nb);
    if (pread(t->fd, raw, nb, boff) != nb) { perror("pread slice"); exit(1); }
    if (t->dtype == 2) memcpy(out, raw, nb);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = bf16_to_f32(p[i]); }
    else { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    if (drop) posix_fadvise(t->fd, boff, nb, POSIX_FADV_DONTNEED);
}

#endif
