/* Indicizzazione e lettura on-demand di tensori da piu' file safetensors.
 * Equivale a Shards in engine.py, ma:
 *   - legge con pread (niente mmap) + posix_fadvise(DONTNEED) -> le pagine NON
 *     restano residenti nel processo. E' la correzione del bug di RSS: cosi' la
 *     RAM di picco resta densa+cache, non l'intero modello. (vedi memoria mmap-rss-bug)
 *   - converte sempre in float32 in uscita (BF16/F16/F32 supportati). */
#ifndef ST_H
#define ST_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>   /* ldexpf per ue8m0_to_f32 */
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include "json.h"
#include "compat.h"

/* tetto sulla dimensione dell'header safetensors: gli header reali sono piccoli
 * (KB..pochi MB). Un file crafted che dichiara un hlen enorme causerebbe una
 * malloc gigante prima ancora di leggere: lo respingiamo. */
#define ST_MAX_HEADER (512ll << 20)
#define ST_MAX_RANK 8

typedef struct {
    char   *name;
    int     fd;
    int64_t off;       /* offset assoluto del dato dentro al file */
    int64_t nbytes;
    int     dtype;     /* 0=BF16 1=F16 2=F32 3=U8/I8 4=F8_E4M3 5=F8_E8M0 6=I64 */
    int64_t numel;
    int     rank;
    int64_t shape[ST_MAX_RANK];
} st_tensor;

typedef struct {
    st_tensor *t;
    int        n, cap;
    int        fds[512];
    int        dfds[512];  /* gemelli O_DIRECT (aperti pigramente): -2 = non ancora provato */
    char      *paths[512];
    int64_t    sizes[512];  /* indexed primary shard sizes, parallel to fds/paths */
    int        nfd;
#define ST_MAX_MIR 4       /* extra read replicas beyond the primary (multi-SSD) */
    int        mfds[ST_MAX_MIR][512];  /* MIRROR: fds of replica copy r+1 (multi-SSD), -1 = absent */
    int        mdfds[ST_MAX_MIR][512]; /* O_DIRECT twins of the replica copies, -1 = absent */
    int        nmirror[ST_MAX_MIR];    /* files accepted into replica r+1 */
    int        nrep;       /* registered replica copies (0 = mirror inactive) */
    int       *hidx;      /* hash map nome->indice (open addressing): con ~120k tensori
                           * (GLM: 256 expert x 78 layer x 3 x 2) la scansione lineare
                           * costava decine di secondi/token (misurato sul primo run reale) */
    int        hcap;
    /* FORMAT METADATA STAMP (reference impl, see colibri.c's qt_verify_fmt_stamp):
     * per-tensor {name -> format NAME string} pairs collected from every shard's
     * __metadata__["colibri.fmt"] JSON blob (safetensors __metadata__ values are
     * always strings, so colibri.fmt's value is itself JSON text, parsed a
     * second time -- see st_init_multi below). Small in practice (only the
     * tensors a stamping tool selected, a subset of S->t), so a flat array +
     * linear st_fmt_stamp() lookup is fine; this is a reference implementation,
     * not a framework -- no hash map for a handful-to-low-thousands of entries.
     * Both arrays own strdup'd strings, intentionally leaked like the rest of
     * st_init_multi's one-time startup parsing (see the json_parse callers
     * below). */
    char     **fmt_name;   /* stamped tensor name */
    char     **fmt_val;    /* stamped format NAME string */
    int        fmt_n, fmt_cap;
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
    /* --- tipi dei checkpoint nativi fp8 (DeepSeek-V4, GLM-5.2-FP8 non ripacchettati) ---
     * PRIMA di questi, st_init faceva exit(1) su un checkpoint DeepSeek-V4 al primo
     * tensore I64, senza arrivare ai pesi. Sono INDICIZZATI qui e letti dal percorso
     * dei byte grezzi (st_read_raw); i lettori float li RIFIUTANO PER NOME invece di
     * caderci dentro -- vedi il commento in st_read_f32. Il loro codice numerico e'
     * nuovo e nessun codice esistente cambia: 0/1/2/3 restano quelli di prima. */
    if (!strcmp(s, "F8_E4M3") || !strcmp(s, "F8_E4M3FN") ||
        !strcmp(s, "float8_e4m3fn")) return 4;
    if (!strcmp(s, "F8_E8M0") || !strcmp(s, "F8_E8M0FNU")) return 5;
    if (!strcmp(s, "I64") || !strcmp(s, "U64")) return 6;
    fprintf(stderr, "unsupported dtype: %s\n", s); exit(1);
}

/* Byte per elemento. UNICO posto che lo sa: prima la formula era ripetuta in tre
 * punti come `dtype==2 ? 4 : 2`, che con soli 0/1/2/3 era corretta e con i tipi
 * nuovi avrebbe silenziosamente detto "2 byte" per un I64 da 8. */
static inline int st_dtype_esz(int dtype) {
    switch (dtype) {
        case 2: return 4;                 /* F32 */
        case 3: case 4: case 5: return 1; /* U8/I8, F8_E4M3, F8_E8M0 */
        case 6: return 8;                 /* I64/U64 */
        default: return 2;                /* BF16, F16 */
    }
}

/* Nome leggibile, per i messaggi di rifiuto. */
static inline const char *st_dtype_name(int dtype) {
    switch (dtype) {
        case 0: return "BF16"; case 1: return "F16"; case 2: return "F32";
        case 3: return "U8/I8"; case 4: return "F8_E4M3"; case 5: return "F8_E8M0";
        case 6: return "I64"; default: return "?";
    }
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
    struct stat sb;
    if (fstat(fd, &sb) != 0) { perror("fstat shard"); close(fd); exit(1); }
    S->paths[S->nfd] = strdup(path); S->fds[S->nfd] = fd;
    S->sizes[S->nfd] = (int64_t)sb.st_size;
#ifdef O_DIRECT
    S->dfds[S->nfd] = open(path, COMPAT_O_RDONLY | O_DIRECT);   /* eager: lookup poi thread-safe */
#elif defined(__APPLE__) || defined(_WIN32)
    S->dfds[S->nfd] = compat_open_direct(path);          /* macOS: F_NOCACHE; Windows: NO_BUFFERING */
#else
    S->dfds[S->nfd] = -1;                                /* niente equivalente: solo buffered */
#endif
    S->nfd++;
    return fd;
}

/* fd gemello O_DIRECT dello stesso file (bypassa la page cache: il buffered read su
 * ext4-in-VHDX si strozza a ~0.8 GB/s, O_DIRECT arriva a 2.3+; misurato). -1 se non disponibile. */
static int st_fidx(shards *S, int fd) {
    for (int i = 0; i < S->nfd; i++) if (S->fds[i] == fd) return i;
    return -1;
}
static int st_direct_fd(shards *S, int fd) {
    int i = st_fidx(S, fd); return i < 0 ? -1 : S->dfds[i];
}

/* ---- MIRROR (multi-SSD): read-only copies of the model on other drives ----
 * st_fd_rep/st_direct_fd_rep: fd of replica `rep` (0 = primary, 1..nrep =
 * mirrors) for the SAME file identified by its primary fd. -1 if absent. */
static int st_fd_rep(shards *S, int fd, int rep) {
    if (!rep) return fd;
    if (rep > S->nrep) return -1;
    int i = st_fidx(S, fd); return i < 0 ? -1 : S->mfds[rep-1][i];
}
static int st_direct_fd_rep(shards *S, int fd, int rep) {
    if (!rep) return st_direct_fd(S, fd);
    if (rep > S->nrep) return -1;
    int i = st_fidx(S, fd); return i < 0 ? -1 : S->mdfds[rep-1][i];
}

/* Registers <dir>/<basename> as read replica S->nrep+1 of every already-indexed
 * shard. A file is accepted ONLY if its size and safetensors header are
 * byte-identical to the primary: the data_offsets then match by construction,
 * so every pread is valid on any copy. Missing or divergent files simply stay
 * on the primary (a mirror may be partial, e.g. a smaller SSD holding only the
 * expert shards). Returns the number of accepted files; a dir contributing 0
 * files claims no replica slot. Mirrors are NEVER written to: .coli_usage /
 * .coli_kv keep deriving from the primary alone. */
static void st_mirror_reset(shards *S) {           /* re-init: drop every replica */
    for (int r = 0; r < S->nrep; r++) for (int i = 0; i < S->nfd; i++) {
        if (S->mfds[r][i] >= 0) close(S->mfds[r][i]);
        if (S->mdfds[r][i] >= 0) close(S->mdfds[r][i]);
    }
    memset(S->nmirror, 0, sizeof(S->nmirror));
    S->nrep = 0;
}
static int st_mirror_add(shards *S, const char *dir) {
    if (S->nrep >= ST_MAX_MIR) return 0;
    int r = S->nrep;
    for (int i = 0; i < ST_MAX_SHARDS; i++) { S->mfds[r][i] = -1; S->mdfds[r][i] = -1; }
    S->nmirror[r] = 0;
    for (int i = 0; i < S->nfd; i++) {
        const char *base = strrchr(S->paths[i], '/');
#ifdef _WIN32
        const char *b2 = strrchr(S->paths[i], '\\');
        if (b2 && (!base || b2 > base)) base = b2;
#endif
        base = base ? base + 1 : S->paths[i];
        char mp[2048]; snprintf(mp, sizeof(mp), "%s/%s", dir, base);
        int mfd = open(mp, COMPAT_O_RDONLY);
        if (mfd < 0) continue;               /* partial mirror: this shard stays on the primary */
        int64_t sza = lseek(S->fds[i], 0, SEEK_END), szb = lseek(mfd, 0, SEEK_END);
        if (sza != szb) {
            fprintf(stderr, "[MIRROR] %s: size differs from the primary copy — file skipped\n", mp);
            close(mfd); continue;
        }
        uint64_t ha = 0, hb = 0; int ok = 1;   /* identical header => identical data_offsets */
        if (pread(S->fds[i], &ha, 8, 0) != 8 || pread(mfd, &hb, 8, 0) != 8 ||
            ha != hb || ha == 0 || ha > (uint64_t)256 << 20 || (int64_t)(8 + ha) > sza) ok = 0;
        if (ok) {
            char *ba = malloc(ha), *bb = malloc(ha);
            if (!ba || !bb || pread(S->fds[i], ba, ha, 8) != (ssize_t)ha ||
                pread(mfd, bb, ha, 8) != (ssize_t)ha || memcmp(ba, bb, ha)) ok = 0;
            free(ba); free(bb);
        }
        if (!ok) {
            fprintf(stderr, "[MIRROR] %s: header differs from the primary copy — file skipped\n", mp);
            close(mfd); continue;
        }
        S->mfds[r][i] = mfd;
#ifdef O_DIRECT
        S->mdfds[r][i] = open(mp, COMPAT_O_RDONLY | O_DIRECT);
#elif defined(__APPLE__) || defined(_WIN32)
        S->mdfds[r][i] = compat_open_direct(mp);
#endif
        S->nmirror[r]++;
    }
    if (S->nmirror[r] > 0) { S->nrep++; return S->nmirror[r]; }
    return 0;
}
/* backward-compatible single-mirror entry point */
static int st_mirror_init(shards *S, const char *dir) {
    st_mirror_reset(S);
    return st_mirror_add(S, dir);
}

/* indicizza tutti i model-*.safetensors in snap_dir */
/* pread completo: chunk-loop (una singola pread si ferma a ~2^31 byte su Linux
 * — i tensori bf16 grandi la superano), riprova su EINTR e riporta un errore
 * ONESTO: perror stampava "Success" su una short-read (errno resta 0), lo
 * stesso sintomo corretto in glm.c per #236. ST_PREAD_CHUNK e' sovrascrivibile
 * per i test. EN: full pread — chunk loop (one pread caps at ~2^31 bytes and
 * big bf16 tensors exceed it), EINTR retry, honest short-read errors.
 * Exits on failure, like every st.h reader. */
#ifndef ST_PREAD_CHUNK
#define ST_PREAD_CHUNK (1u << 30)
#endif
static void st_pread_full(int fd, void *buf, int64_t n, int64_t off, const char *tag) {
    char *p = (char *)buf;
    int64_t got = 0;
    while (got < n) {
        int64_t want = n - got;
        if (want > (int64_t)ST_PREAD_CHUNK) want = ST_PREAD_CHUNK;
        ssize_t r = pread(fd, p + got, (size_t)want, off + got);
        if (r < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "%s: %s (off %lld, %lld/%lld bytes)\n", tag, strerror(errno),
                    (long long)off, (long long)got, (long long)n);
            exit(1);
        }
        if (r == 0) {
            fprintf(stderr, "%s: short read at EOF (off %lld, %lld/%lld bytes) — truncated file?\n",
                    tag, (long long)off, (long long)got, (long long)n);
            exit(1);
        }
        got += r;
    }
}

/* Stamps are a resident-tensor convention (see docs/FORMATS.md's "Stamp-map
 * scan bound"): a handful to a few hundred entries per model
 * (q_a/q_b/kv_a/kv_b_proj, o_proj, shared-expert and dense-MLP gate/up/down),
 * NEVER the tens of thousands of routed-expert tensors a large MoE model
 * carries (tools/repack_fp8_passthrough.py never stamps routed experts). A
 * container whose combined __metadata__["colibri.fmt"] entries exceed this
 * cap is not using the convention as designed -- CAP, not a switch to a hash
 * table. Precisely what this bounds: the colibri.fmt blob is json_parse'd in
 * FULL before the per-entry check below fires, so the parse allocation
 * itself is bounded by ST_MAX_HEADER (the shard-header size cap), not by
 * this constant -- what the cap bounds is the PERSISTENT fmt_name/fmt_val
 * strdup arrays on `shards` (and every later st_fmt_stamp linear scan over
 * them), which would otherwise grow with an adversarial map. Refuse loudly
 * rather than carry an absurd stamp map forward. */
#define ST_FMT_STAMP_MAX 4096

/* Parses one shard's __metadata__["colibri.fmt"] value (a safetensors metadata
 * value is always a plain string, so colibri.fmt's VALUE is itself JSON text --
 * a flat {tensor_name: format_name} object -- parsed a second time here) and
 * appends every entry to S->fmt_name/fmt_val. Absent __metadata__, or a
 * __metadata__ without a colibri.fmt key, is NOT an error: that's simply an
 * unstamped container, and byte-arithmetic inference alone decides, exactly as
 * before this feature existed (see qt_verify_fmt_stamp in colibri.c). A
 * colibri.fmt key that IS present but doesn't parse into that shape is refused
 * loudly -- same "untrusted container" discipline qt_resolve_fmt applies
 * elsewhere: a stamp the engine cannot make sense of must not be silently
 * ignored (that would be indistinguishable from a real mismatch going
 * unnoticed).
 *
 * DISCOVERY-TIME ABORT SURFACE: every exit(1) below (malformed stamp value,
 * malformed entry, or the ST_FMT_STAMP_MAX cap) fires from inside
 * st_init_multi's shard-header-parse loop -- i.e. at CONTAINER DISCOVERY
 * time, while the engine is still building its tensor index, before it has
 * resolved a single tensor against the model's architecture or read one byte
 * of weight data. This is coarser-grained and EARLIER than
 * qt_resolve_fmt/qt_verify_fmt_stamp's own per-tensor refusals in colibri.c
 * (which fire much later, during weight load, once a specific tensor's
 * [O,I] shape and stamp are both known): a malformed stamp anywhere in any
 * shard aborts the ENTIRE model load immediately, before the user ever sees
 * which layer or tensor was implicated -- these messages name a shard FILE,
 * never a tensor, which is how to tell this abort surface apart from the
 * later per-tensor one at a glance. See docs/FORMATS.md's own section on
 * this. */
static void st_fmt_stamp_ingest(shards *S, jval *root, const char *shard_path) {
    jval *meta = json_get(root, "__metadata__");
    if (!meta || meta->t != J_OBJ) return;                /* no metadata object: unstamped, fine */
    jval *stamp = json_get(meta, "colibri.fmt");
    if (!stamp) return;                                    /* no stamp key: unstamped, fine */
    if (stamp->t != J_STR) {
        fprintf(stderr, "%s: __metadata__[\"colibri.fmt\"] is not a JSON string -- malformed stamp, refusing (untrusted container)\n",
                shard_path); exit(1); }
    char *arena2 = NULL;
    jval *inner = json_parse(stamp->str, &arena2);
    if (!inner || inner->t != J_OBJ) {
        fprintf(stderr, "%s: __metadata__[\"colibri.fmt\"] does not parse as a JSON object -- malformed stamp, refusing (untrusted container)\n",
                shard_path); exit(1); }
    for (int i = 0; i < inner->len; i++) {
        jval *v = inner->kids[i];
        if (v->t != J_STR) {
            fprintf(stderr, "%s: colibri.fmt entry '%s' is not a string -- malformed stamp, refusing (untrusted container)\n",
                    shard_path, inner->keys[i]); exit(1); }
        /* DUPLICATE CLAIMS (user-ratified design, register D8): at most one
         * DISTINCT format claim per tensor name, container-wide. An entry
         * that repeats an already-ingested claim verbatim is tolerated and
         * collapsed to one entry (idempotent -- a centralized-manifest
         * writer may legally stamp the same map into every shard, and a
         * shard may stamp tensors it does not itself contain; there is no
         * locality constraint). An entry that CONTRADICTS an earlier claim
         * refuses by name: a container that disagrees with itself about a
         * tensor's format is corrupted or hostile, and the previous
         * first-wins behavior made the outcome depend on shard enumeration
         * order (st_scan_dir is raw readdir order, not sorted) while
         * mis-diagnosing the real problem downstream as a stamp/inference
         * mismatch -- or hiding it entirely when the enumeration happened
         * to favor the agreeing claim. The refusal names the tensor and
         * both format names; the EARLIER claim's shard file is not named
         * (per-entry shard provenance isn't stored, and adding it just for
         * this message would be new plumbing -- only the current shard's
         * path is in scope here). Linear rescan per entry is O(n^2) worst
         * case, bounded by ST_FMT_STAMP_MAX at one-time startup. */
        int dup = -1;
        for (int k = 0; k < S->fmt_n; k++)
            if (!strcmp(S->fmt_name[k], inner->keys[i])) { dup = k; break; }
        if (dup >= 0) {
            if (!strcmp(S->fmt_val[dup], v->str)) continue;   /* agreeing duplicate: keep one */
            fprintf(stderr, "%s: __metadata__[\"colibri.fmt\"] stamps tensor '%s' as '%s', but an "
                    "earlier shard's map already stamped it '%s' -- conflicting format claims, "
                    "refusing (untrusted container)\n",
                    shard_path, inner->keys[i], v->str, S->fmt_val[dup]); exit(1); }
        if (S->fmt_n >= ST_FMT_STAMP_MAX) {
            fprintf(stderr, "%s: __metadata__[\"colibri.fmt\"] stamps more than %d tensor names across "
                    "this container's shards -- stamps are a resident-tensor convention (docs/FORMATS.md), "
                    "not a bulk migration path; a container stamping this many names is malformed, "
                    "refusing (untrusted container)\n",
                    shard_path, ST_FMT_STAMP_MAX); exit(1); }
        if (S->fmt_n == S->fmt_cap) {
            S->fmt_cap = S->fmt_cap ? S->fmt_cap * 2 : 16;
            char **nn = (char**)realloc(S->fmt_name, S->fmt_cap * sizeof(char*));
            if (!nn) { fprintf(stderr, "%s: OOM reallocating format-stamp table (fmt_name, %d entries)\n",
                        shard_path, S->fmt_cap); exit(1); }
            S->fmt_name = nn;
            char **nv = (char**)realloc(S->fmt_val, S->fmt_cap * sizeof(char*));
            if (!nv) { fprintf(stderr, "%s: OOM reallocating format-stamp table (fmt_val, %d entries)\n",
                        shard_path, S->fmt_cap); exit(1); }
            S->fmt_val = nv;
        }
        /* Neither strdup here is benign to drop: a NULL fmt_name would be dereferenced by
         * the duplicate-scan strcmp above on the very next entry, and a NULL fmt_val is
         * worse -- the process would exit 0 with no diagnostic, st_fmt_stamp() would then
         * return NULL for a tensor that IS stamped, and the container would silently read
         * as unstamped: qt_verify_fmt_stamp's TRUST-VERIFY-REFUSE check (colibri.c) is a
         * no-op on an unstamped tensor, so one failed allocation would quietly disable it
         * on an untrusted-container path. Refuse instead. */
        char *sn = strdup(inner->keys[i]), *sv = strdup(v->str);
        if (!sn || !sv) { fprintf(stderr, "%s: OOM duplicating __metadata__[\"colibri.fmt\"] stamp "
                    "for tensor '%s' -- refusing (a dropped stamp would silently disable format "
                    "verification for this tensor)\n", shard_path, inner->keys[i]); exit(1); }
        S->fmt_name[S->fmt_n] = sn;
        S->fmt_val[S->fmt_n]  = sv;
        S->fmt_n++;
    }
    json_free(inner);
    free(arena2);  /* always NULL (json_parse never populates it -- see j_dup). */
}

/* Stamped format NAME for `name`, or NULL if this tensor carries no stamp
 * (either because no shard stamped it, or the container predates this
 * feature). Linear scan: S->fmt_n is a subset of S->n (only stamped tensors),
 * small in practice -- see the shards struct comment.
 *
 * SCOPE: .qs-BACKED TENSORS ONLY. This function itself will happily look up
 * ANY name that got stamped -- st_fmt_stamp_ingest doesn't know or check
 * whether a stamped name belongs to a quantized (.qs-backed) tensor -- but
 * qt_from_disk (colibri.c) only ever CALLS this inside its `st_has(name+
 * ".qs")` branch, i.e. only for a tensor that already carries a quantized
 * scale sidecar. A colibri.fmt entry naming a raw f32/bf16 weight, a norm, a
 * router, or embed/lm_head is stored here like any other entry but never
 * looked up: it is silently ignored BY DESIGN, not an oversight -- the
 * convention exists to disambiguate a byte-count collision among quantized
 * formats, and only a .qs-backed tensor can have one. See docs/FORMATS.md's
 * "Scope: .qs-backed tensors only". */
static const char *st_fmt_stamp(shards *S, const char *name) {
    for (int i = 0; i < S->fmt_n; i++)
        if (!strcmp(S->fmt_name[i], name)) return S->fmt_val[i];
    return NULL;
}

/* Scan one directory for *.safetensors shards, appending to files[] (dedup by
 * basename, so a list of directories acts as a SEARCH PATH: the same shard
 * present on two drives is taken from the first-listed one only). *added
 * returns how many shards this dir contributed. */
static void st_scan_dir(const char *dir, char files[][1024], int *nf, int *added) {
    DIR *d = opendir(dir); struct dirent *e;
    if (!d) { perror(dir); exit(1); }
    int base_n = *nf;
    while ((e = readdir(d))) {
        const char *dot = strrchr(e->d_name, '.');
        if (dot && !strcmp(dot, ".safetensors")) {  /* model.safetensors o model-0000N-of-... */
            int dup = 0;
            for (int i = 0; i < *nf; i++) {
                const char *b = strrchr(files[i], '/');
#ifdef _WIN32
                const char *b2 = strrchr(files[i], '\\'); if (b2 && (!b || b2 > b)) b = b2;
#endif
                b = b ? b + 1 : files[i];
                if (!strcmp(b, e->d_name)) { dup = 1; break; }  /* already taken from a higher-priority drive */
            }
            if (dup) continue;
            if (*nf >= ST_MAX_SHARDS) { fprintf(stderr, "too many shards (>%d): raise ST_MAX_SHARDS\n", ST_MAX_SHARDS); exit(1); }
            snprintf(files[(*nf)++], 1024, "%s/%s", dir, e->d_name);
        }
    }
    closedir(d);
    if (added) *added = *nf - base_n;
}

/* Index shards from snap_dir, optionally SPLIT across extra drives listed in
 * extra_dirs (';' or ',' separated). Each shard lives on exactly ONE drive
 * (no duplication — unlike the dual-SSD mirror); a demand pread hits whichever
 * drive holds that shard, so concurrent expert loads parallelise across drives
 * and combined capacity is used. Scales to N drives. Metadata (config /
 * tokenizer / .coli_usage / .coli_kv) is read from snap_dir only. */
static void st_init_multi(shards *S, const char *snap_dir, const char *extra_dirs) {
    memset(S, 0, sizeof(*S));
    S->cap = 4096; S->t = calloc(S->cap, sizeof(st_tensor));
    /* Duplicate-tensor-name detector, indexed-shard scope.
     * st_find()/S->hidx (below) aren't built until every shard has been read, so a
     * duplicate name across two shards would otherwise go unnoticed until AFTER both are
     * already appended to S->t -- by then the collision has already happened silently:
     * st_find() would return whichever shard was appended first, while a container-wide
     * __metadata__["colibri.fmt"] stamp for that name (st_fmt_stamp_ingest above) would
     * still apply uniformly. A directory blending two containers under the same tensor
     * names (e.g. an FP8 copy and an int8 copy of the same o_proj) would then silently read
     * one shard's bytes labeled with a format that describes the other's -- no error, no
     * warning, plausible numbers. This is a DIFFERENT concern from st_fmt_stamp_ingest's
     * own duplicate check a few lines above: that dedups format CLAIMS keyed by name (and
     * already refuses on a disagreeing claim); this dedups the tensor DATA entries
     * themselves, the actual byte ranges a read resolves to. Kept as its own temporary
     * open-addressing table (same probe scheme st_hash()/S->hidx use below) rather than a
     * linear rescan, because a real container indexes on the order of 1e5 tensors and an
     * O(n^2) scan at that scale is not a rounding error. Grown in lockstep with S->cap
     * (below) and freed at the end of this function, just before S->hidx is built. */
    int dup_cap = 1; while (dup_cap < S->cap * 2) dup_cap <<= 1;
    int *dup_idx = (int*)malloc(dup_cap * sizeof(int));
    if (!dup_idx) { fprintf(stderr, "OOM allocating duplicate-tensor-name detector (%d entries)\n",
                dup_cap); exit(1); }
    for (int di = 0; di < dup_cap; di++) dup_idx[di] = -1;
    /* raccoglie ordinatamente i nomi dei file shard */
    static char files[ST_MAX_SHARDS][1024]; int nf = 0;
    int c0 = 0; st_scan_dir(snap_dir, files, &nf, &c0);
    int ndir = 1;
    if (extra_dirs && *extra_dirs) {
        char buf[4096]; snprintf(buf, sizeof(buf), "%s", extra_dirs);
        char *p = buf;
        while (p && *p) {
            char *sep = p; while (*sep && *sep != ';' && *sep != ',') sep++;
            int last = (*sep == 0); *sep = 0;
            while (*p == ' ') p++;
            size_t plen = strlen(p); while (plen > 0 && p[plen-1] == ' ') p[--plen] = 0;
            if (*p) {
                int cN = 0; st_scan_dir(p, files, &nf, &cN);
                fprintf(stderr, "[SPLIT] +%s -> %d shard(s)\n", p, cN);
                ndir++;
            }
            p = last ? NULL : sep + 1;
        }
        fprintf(stderr, "[SPLIT] model across %d dir(s): %d shard(s) total (primary %s -> %d shard(s)), no duplication\n",
                ndir, nf, snap_dir, c0);
    }
    for (int a = 0; a < nf; a++) for (int b = a+1; b < nf; b++)
        if (strcmp(files[a], files[b]) > 0) { char tmp[1024]; memcpy(tmp, files[a], 1024); memcpy(files[a], files[b], 1024); memcpy(files[b], tmp, 1024); }

    for (int fi = 0; fi < nf; fi++) {
        int fd = st_open_fd(S, files[fi]);
        int fidx = st_fidx(S, fd);
        if (fidx < 0) { fprintf(stderr, "%s: indexed shard fd is missing\n", files[fi]); exit(1); }
        int64_t fsz = S->sizes[fidx];
        uint64_t hlen;
        st_pread_full(fd, &hlen, 8, 0, "pread hlen");
        /* file malevolo/troncato: hlen deve stare nel file dopo gli 8 byte di
         * prefisso e sotto il tetto. Senza questo bound hlen+1 puo' andare in
         * overflow (malloc(0) e poi hdr[hlen]=0 fuori limiti) o forzare una
         * malloc gigante. */
        if (fsz < 8 || hlen > (uint64_t)(fsz - 8) || hlen > (uint64_t)ST_MAX_HEADER) {
            fprintf(stderr, "%s: bad safetensors header length %llu (file %lld bytes)\n",
                    files[fi], (unsigned long long)hlen, (long long)fsz); exit(1); }
        char *hdr = malloc(hlen + 1);
        if (!hdr) { perror("malloc safetensors header"); exit(1); }
        st_pread_full(fd, hdr, (int64_t)hlen, 8, "pread hdr");
        hdr[hlen] = 0;
        int64_t data_start = 8 + (int64_t)hlen;
        char *arena = NULL;
        jval *root = json_parse(hdr, &arena);
        if (!root || root->t != J_OBJ) {
            fprintf(stderr, "%s: safetensors header is not a JSON object\n", files[fi]); exit(1); }
        st_fmt_stamp_ingest(S, root, files[fi]);
        for (int i = 0; i < root->len; i++) {
            const char *name = root->keys[i];
            if (!strcmp(name, "__metadata__")) continue;
            jval *m = root->kids[i];
            jval *dt = json_get(m, "dtype");
            jval *off = json_get(m, "data_offsets");
            jval *shp = json_get(m, "shape");
            /* un header crafted puo' omettere i campi o dare tipi sbagliati:
             * senza questi guard si dereferenzia NULL (json_get) o si legge
             * off->kids[0/1] oltre i limiti dell'array. */
            if (!dt || dt->t != J_STR || !off || off->t != J_ARR || off->len < 2 ||
                !shp || shp->t != J_ARR || shp->len > ST_MAX_RANK) {
                fprintf(stderr, "%s: tensor '%s' has malformed dtype/data_offsets/shape\n",
                        files[fi], name); exit(1); }
            jval *off_a = off->kids[0], *off_b = off->kids[1];
            if (!off_a || !off_b || off_a->t != J_NUM || off_b->t != J_NUM ||
                !isfinite(off_a->num) || !isfinite(off_b->num) ||
                off_a->num < 0.0 || off_b->num < 0.0 ||
                off_a->num >= ldexp(1.0, 63) || off_b->num >= ldexp(1.0, 63) ||
                floor(off_a->num) != off_a->num || floor(off_b->num) != off_b->num) {
                fprintf(stderr, "%s: tensor '%s' has invalid data_offsets\n",
                        files[fi], name); exit(1); }
            int64_t a0 = (int64_t)off_a->num, b0 = (int64_t)off_b->num;
            /* offset dichiarati dal file: non-negativi, ordinati e dentro al
             * file. Altrimenti nbytes=b0-a0 diventa negativo -> malloc((size_t))
             * gigante e la memcpy in st_read_f32 sfora il buffer del chiamante;
             * oppure off punta fuori dal file. */
            if (b0 < a0 || b0 > INT64_MAX - data_start || data_start + b0 > fsz) {
                fprintf(stderr, "%s: tensor '%s' data_offsets [%lld,%lld] out of file bounds (%lld)\n",
                        files[fi], name, (long long)a0, (long long)b0, (long long)fsz); exit(1); }
            /* SEC: lo shape viene da un file non fidato (mirror). Senza il guard
             * di overflow, uno shape tipo [65535,65535,65535,...] fa avvolgere
             * numel a un valore piccolo/negativo che poi passerebbe il cross-check
             * numel*esz==nbytes in st_read_f32, riaprendo l'OOB. */
            int64_t numel = 1; int bad_shape = 0;
            for (int k = 0; k < shp->len; k++) {
                jval *dim = shp->kids[k];
                if (!dim || dim->t != J_NUM || !isfinite(dim->num) ||
                    dim->num < 0.0 || dim->num >= ldexp(1.0, 63) ||
                    floor(dim->num) != dim->num) { bad_shape = 1; break; }
                int64_t d = (int64_t)dim->num;
                if (d != 0 && numel > INT64_MAX / d) { bad_shape = 1; break; }
                numel *= d;
            }
            if (bad_shape) {
                fprintf(stderr, "%s: tensor '%s' shape overflows int64 — refusing (hostile or corrupt file)\n",
                        files[fi], name); exit(1); }
            /* Refuse if `name` was already indexed from an earlier shard (see the
             * detector's block comment near the top of this function). Must run before
             * this tensor is appended, and is intentionally NOT the same check as
             * st_fmt_stamp_ingest's -- that one only ever sees the small subset of names a
             * stamping tool chose to annotate; this one sees every tensor. */
            {
                uint64_t dh = st_hash(name) & (dup_cap - 1);
                while (dup_idx[dh] >= 0) {
                    if (!strcmp(S->t[dup_idx[dh]].name, name)) {
                        int prev_fidx = st_fidx(S, S->t[dup_idx[dh]].fd);
                        fprintf(stderr, "%s: tensor '%s' is also indexed from shard '%s' -- duplicate "
                                "tensor name across indexed shards, refusing (a directory holding both "
                                "would silently read one shard's bytes for this name while any "
                                "__metadata__[\"colibri.fmt\"] stamp for it still applies uniformly)\n",
                                files[fi], name, prev_fidx >= 0 ? S->paths[prev_fidx] : "<unknown>");
                        exit(1);
                    }
                    dh = (dh + 1) & (dup_cap - 1);
                }
            }
            if (S->n == S->cap) {
                S->cap *= 2;
                st_tensor *nt = (st_tensor*)realloc(S->t, S->cap * sizeof(st_tensor));
                if (!nt) { fprintf(stderr, "OOM reallocating shard tensor array\n"); exit(1); }
                S->t = nt;
                /* keep the duplicate-name detector's load factor low as S->t grows;
                 * rehash into a table twice the new capacity. */
                int new_dup_cap = dup_cap * 2;
                int *ndup = (int*)malloc(new_dup_cap * sizeof(int));
                if (!ndup) { fprintf(stderr, "OOM growing duplicate-tensor-name detector (%d entries)\n",
                            new_dup_cap); exit(1); }
                for (int di = 0; di < new_dup_cap; di++) ndup[di] = -1;
                for (int di = 0; di < dup_cap; di++) {
                    if (dup_idx[di] < 0) continue;
                    uint64_t h = st_hash(S->t[dup_idx[di]].name) & (new_dup_cap - 1);
                    while (ndup[h] >= 0) h = (h + 1) & (new_dup_cap - 1);
                    ndup[h] = dup_idx[di];
                }
                free(dup_idx); dup_idx = ndup; dup_cap = new_dup_cap;
            }
            st_tensor *t = &S->t[S->n++];
            memset(t, 0, sizeof(*t));
            t->name = strdup(name); t->fd = fd; t->off = data_start + a0;
            t->nbytes = b0 - a0; t->dtype = st_dtype_code(dt->str); t->numel = numel;
            t->rank = shp->len;
            /* Register this name in the duplicate detector now that t->name is
             * stable, so the next shard's scan (above) sees it. */
            { uint64_t dh = st_hash(t->name) & (dup_cap - 1);
              while (dup_idx[dh] >= 0) dh = (dh + 1) & (dup_cap - 1);
              dup_idx[dh] = S->n - 1; }
            for (int k = 0; k < t->rank; k++) t->shape[k] = (int64_t)shp->kids[k]->num;
            /* cross-check the declared element count against the byte span for FLOAT
             * dtypes: st_read_f32 writes `numel` floats (BF16/F16 loop or F32 memcpy)
             * into a caller-sized buffer, so a header with numel != nbytes/esz is an
             * OOB write primitive. U8/I8 (raw quant bytes) are read by byte count, so
             * their numel is unused by the read path and legitimately may differ. */
            { int esz = st_dtype_esz(t->dtype);
              if (t->dtype != 3 &&
                  (numel > INT64_MAX / esz || t->nbytes != numel * (int64_t)esz)) {
                  fprintf(stderr, "%s: tensor '%s' numel %lld disagrees with byte span %lld (esz %d)\n",
                          files[fi], name, (long long)numel, (long long)t->nbytes, esz); exit(1); } }
        }
        json_free(root);
        free(arena);
        free(hdr);
    }
    free(dup_idx);  /* duplicate detector: one-time-startup scratch, superseded by S->hidx below */
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

/* backward-compatible single-directory entry point */
static void st_init(shards *S, const char *snap_dir) { st_init_multi(S, snap_dir, NULL); }

/* Long-lived embedding consumers (the Segment runtime is the first one) can
 * open and close several model ranges in one process.  The historical CLI
 * exits after one model and intentionally relied on process teardown; expose a
 * real cleanup path without changing that CLI lifecycle. */
static void st_destroy(shards *S) {
    if (!S) return;
    st_mirror_reset(S);
    for (int index = 0; index < S->nfd; index++) {
        if (S->fds[index] >= 0) close(S->fds[index]);
        if (S->dfds[index] >= 0 && S->dfds[index] != S->fds[index])
            close(S->dfds[index]);
        free(S->paths[index]);
    }
    for (int index = 0; index < S->n; index++) free(S->t[index].name);
    for (int index = 0; index < S->fmt_n; index++) {
        free(S->fmt_name[index]);
        free(S->fmt_val[index]);
    }
    free(S->fmt_name);
    free(S->fmt_val);
    free(S->hidx);
    free(S->t);
    memset(S, 0, sizeof(*S));
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

/* A missing CORE tensor is almost never an engine bug: the converter writes the final
 * norm and lm_head into the LAST shards, so a transfer that stopped early indexes
 * nearly everything and then dies on the first tensor from the gap. Bare "missing
 * model.norm.weight" gave the user no way to tell that apart from a wrong --model or a
 * real bug (#586, #583). Report what was actually found, and let the filenames say
 * whether shards are absent: the HF layout declares the total in `-of-NNNNN`, the
 * converter layout at least reveals a truncated tail. */
static void st_die_missing(shards *S, const char *name) {
    fprintf(stderr, "missing %s\n\n", name);
    if (S->nfd == 0 || S->n == 0) {
        fprintf(stderr, "  No safetensors tensors were indexed at all. --model must point AT the\n"
                        "  container directory -- the one holding the *.safetensors shards.\n");
        exit(1);
    }
    int lo = -1, hi = -1, declared = 0, numbered = 0;
    for (int i = 0; i < S->nfd; i++) {
        const char *b = strrchr(S->paths[i], '/');
#ifdef _WIN32
        const char *b2 = strrchr(S->paths[i], '\\');
        if (b2 && (!b || b2 > b)) b = b2;
#endif
        b = b ? b + 1 : S->paths[i];
        int idx, tot;
        if (sscanf(b, "model-%d-of-%d", &idx, &tot) == 2) declared = tot;
        else if (sscanf(b, "out-%d", &idx) != 1) continue;   /* out-mtp-* etc: not numbered */
        numbered++;
        if (lo < 0 || idx < lo) lo = idx;
        if (idx > hi) hi = idx;
    }
    int complete = declared > 0 && numbered >= declared;
    fprintf(stderr, "  indexed %d tensors from %d shard file(s)\n", S->n, S->nfd);
    if (declared > 0 && numbered < declared)
        fprintf(stderr, "  shard filenames declare %d shards, but only %d are here -- %d MISSING\n",
                declared, numbered, declared - numbered);
    else if (declared > 0)
        fprintf(stderr, "  shard filenames declare %d shards and %d are here\n", declared, numbered);
    else if (numbered > 0)
        fprintf(stderr, "  shards numbered %05d..%05d, %d file(s)%s\n", lo, hi, numbered,
                numbered == hi - lo + 1 ? " (contiguous: a gap would be at the tail)" : " (GAPS in the numbering)");
    /* Follow the evidence: telling someone who already has every declared shard to
     * re-download sends them round a loop that cannot help them. */
    if (complete) {
        fprintf(stderr,
            "\n  '%s' is a core tensor the engine cannot run without, and every shard the\n"
            "  filenames declare is present -- so this is NOT the usual truncated download.\n"
            "  Either the container was built without this tensor, or the engine is looking\n"
            "  for the wrong name. Please report it with this output.\n", name);
        exit(1);
    }
    /* The final norm and lm_head sit at the END of the model, so an interrupted transfer
     * loses them first -- worth saying, but only where it's true. */
    if (strstr(name, "model.norm") || strstr(name, "lm_head"))
        fprintf(stderr, "\n  '%s' is written into one of the LAST shards, so a container whose tail\n"
                        "  never arrived indexes almost everything and then fails exactly here.\n", name);
    else
        fprintf(stderr, "\n  '%s' is a core tensor: the engine cannot run without it.\n", name);
    fprintf(stderr,
        "  An incomplete transfer is far more likely than a corrupt engine. Re-running the\n"
        "  download resumes only the missing shards; the HF xet backend is known to stall\n"
        "  mid-transfer (#452), so disable it:\n"
        "      HF_HUB_DISABLE_XET=1 hf download <repo> --local-dir <model-dir>\n"
        "  If your shard count is already complete, that IS a bug worth reporting.\n");
    exit(1);
}

/* prefetch ASINCRONO: dice al kernel di iniziare a leggere le pagine del tensore in
 * background (readahead). Serve a sovrapporre l'I/O degli expert col calcolo: si
 * prefetcha tutto il set di expert di un layer, poi le pread sincrone trovano la cache
 * gia' calda. No-op se il tensore non esiste (es. il primo .qs prima della lettura). */
static void st_prefetch(shards *S, const char *name) {
    st_tensor *t = st_find(S, name);
    if (t) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* like st_prefetch, but on replica `rep`'s drive: the WILLNEED must warm the
 * page cache of the SAME fd the later demand pread will hit. */
static void st_prefetch_rep(shards *S, const char *name, int rep) {
    st_tensor *t = st_find(S, name);
    if (!t) return;
    int fd = st_fd_rep(S, t->fd, rep);
    if (fd < 0) fd = t->fd;
    posix_fadvise(fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* legge un tensore in un buffer float32 fornito dal chiamante (numel float).
 * drop=1 -> consiglia al kernel di scartare le pagine (per gli expert in streaming). */
static int64_t st_read_f32(shards *S, const char *name, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    /* SEC: numel viene dallo shape, nbytes dagli offset — due campi indipendenti
     * del file. Se non concordano, la memcpy F32 (nbytes) o i loop BF16/F16
     * (numel elementi da un raw di soli nbytes) sforano il buffer del chiamante,
     * che e' dimensionato sul config, non sul file. Il chiamante che alloca su
     * st_numel resta coerente; questo blocca l'ingresso ostile a monte. */
    /* I tipi non-float si leggono con st_read_raw, non qui. Senza questo rifiuto
     * cadrebbero nel ramo `else` in fondo, che assume F16: un tensore F8_E4M3 o
     * I64 verrebbe letto come mezze-precisioni e produrrebbe numeri plausibili e
     * sbagliati, in silenzio. Con soli i dtype 0/1/2/3 il fallthrough era corretto;
     * dal momento in cui ne esistono altri, non lo e' piu'. */
    if (t->dtype >= 3) {
        fprintf(stderr, "%s: tensor '%s' is %s — not a float tensor; read it with st_read_raw\n",
                name, name, st_dtype_name(t->dtype)); exit(1); }
    int esz = st_dtype_esz(t->dtype);
    if (t->numel < 0 || t->numel > t->nbytes / esz || t->numel * (int64_t)esz != t->nbytes) {
        fprintf(stderr, "%s: tensor '%s' shape/bytes mismatch (numel %lld, %lld bytes, dtype %d) — refusing (hostile or corrupt file)\n",
                name, name, (long long)t->numel, (long long)t->nbytes, t->dtype); exit(1); }
    void *raw = malloc(t->nbytes);
    if (!raw) { fprintf(stderr, "malloc %lld bytes for tensor %s failed\n", (long long)t->nbytes, name); exit(1); }
    st_pread_full(t->fd, raw, t->nbytes, t->off, "pread data");
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

/* like st_read_f32 but refuses to write more than `cap` floats into `out`.
 * Callers that size `out` from CONFIG dims (qt_alloc's O*I, per-row scales) must
 * use this: a crafted header whose tensor holds more elements than the config
 * shape would otherwise overrun `out`. Callers that size `out` from st_numel are
 * self-consistent and may keep using st_read_f32. */
static int64_t st_read_f32_cap(shards *S, const char *name, float *out, int64_t cap, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->numel < 0 || t->numel > cap) {
        fprintf(stderr, "tensor %s: numel %lld exceeds destination capacity %lld\n",
                name, (long long)t->numel, (long long)cap); exit(1); }
    return st_read_f32(S, name, out, drop);
}

static int64_t st_numel(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->numel : -1;
}
static int64_t st_nbytes(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->nbytes : -1;
}

/* --- ue8m0_to_f32 / st_read_scale_f32: sidecar di scale a 1 byte ------------
 *
 * UE8M0 e' un esponente potenza-di-due senza segno e senza mantissa: il valore
 * e' 2^(v-127), e 0xff e' NaN. Un byte per scala invece di quattro.
 *
 * Serve perche' un checkpoint fp8 nativo (DeepSeek-V4, e in generale
 * quantization_config.scale_fmt == "ue8m0") scrive le scale di blocco cosi',
 * mentre i container ripacchettati da noi le scrivono in f32. La GEOMETRIA e'
 * identica -- stessa forma, stesso significato, stessa moltiplicazione -- cambia
 * solo la codifica del numero.
 *
 * Si espande a f32 UNA VOLTA al caricamento invece di decodificare nel kernel:
 * le scale sono ~1/16384 dei byte dei pesi (mezzo MB per gli 8,4 GB densi di
 * DeepSeek-V4), quindi il costo in memoria e' trascurabile e matmul_fp8 resta
 * UNA sola implementazione, senza un ramo dentro il ciclo caldo. E' la stessa
 * scelta gia' fatta in kimi_k3.c per le scale ue8m0 di MXFP4 (`mx4_scale`).
 *
 * NaN: 0xff decodifica a un NaN IEEE reale e lo si lascia propagare, coerente
 * con la politica gia' documentata per i pesi fp8 in quant.h -- la rete di
 * sicurezza sta a valle, sul sampler (test_logit_nan.c), non qui. */
static inline float ue8m0_to_f32(uint8_t v) {
    if (v == 0xff) { uint32_t n = 0x7fc00000u; float f; memcpy(&f, &n, 4); return f; }
    /* ldexpf e NON il trucco `(uint32_t)v << 23`: quel trucco e' esatto per
     * v in [1,254], ma a v==0 produce il pattern di bit 0x00000000, che in IEEE
     * 754 e' ZERO ESATTO e non 2^-127. Un blocco di pesi con quella scala
     * verrebbe azzerato invece che reso quasi-zero -- differenza piccola in
     * ampiezza, ma e' comunque un valore sbagliato, e 2^-127 e' rappresentabile
     * come subnormale. Costa solo al caricamento (una volta per scala), quindi
     * si paga la chiamata e si tiene la correttezza. */
    return ldexpf(1.0f, (int)v - 127);
}

/* Legge un sidecar di scale in `out` come f32, accettando SIA F32 SIA F8_E8M0.
 * Rifiuta qualunque altro dtype per nome. `cap` e' il numero massimo di float
 * che il chiamante ha allocato, come in st_read_f32_cap. */
static int64_t st_read_scale_f32(shards *S, const char *name, float *out, int64_t cap, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->numel < 0 || t->numel > cap) {
        fprintf(stderr, "scale %s: numel %lld exceeds destination capacity %lld\n",
                name, (long long)t->numel, (long long)cap); exit(1); }
    if (t->dtype == 2 || t->dtype == 0 || t->dtype == 1) return st_read_f32(S, name, out, drop);
    if (t->dtype != 5) {
        fprintf(stderr, "scale %s: dtype %s is neither F32 nor F8_E8M0\n",
                name, st_dtype_name(t->dtype)); exit(1); }
    /* stessa validazione byte-vs-numel dei percorsi float: 1 byte per scala */
    if (t->nbytes != t->numel) {
        fprintf(stderr, "scale %s: F8_E8M0 numel %lld disagrees with %lld bytes\n",
                name, (long long)t->numel, (long long)t->nbytes); exit(1); }
    uint8_t *raw = (uint8_t*)malloc((size_t)t->nbytes);
    if (!raw) { fprintf(stderr, "malloc %lld bytes for scale %s failed\n", (long long)t->nbytes, name); exit(1); }
    st_pread_full(t->fd, raw, t->nbytes, t->off, "pread ue8m0 scale");
    for (int64_t i = 0; i < t->numel; i++) out[i] = ue8m0_to_f32(raw[i]);
    free(raw);
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    return t->numel;
}

/* legge i byte GREZZI di un tensore (nessuna conversione di dtype): per i pesi gia'
 * quantizzati int4/int8 del nostro container (dtype U8). drop=1 -> fadvise DONTNEED.
 *
 * CALLER CONTRACT: this reads `t->nbytes` -- a length declared by the file header --
 * into `out`, and has no bound of its own. st_init cannot supply one: it deliberately
 * skips its numel*esz==nbytes cross-check for dtype 3, because packed quant bytes
 * legitimately have numel != nbytes. So the caller MUST establish that its destination
 * is at least t->nbytes before calling. Today all callers do, by three routes:
 *   - colibri.c sizes the buffer from st_nbytes() itself, and qt_resolve_fmt validates
 *     both byte counts against [O,I];
 *   - kimi_k3.c makes the byte count the branch predicate (`if(t->nbytes==O*I ...)`),
 *     so identifying the format and validating it are the same act;
 *   - olmoe.c compares against a config-derived want_w and refuses by name.
 * A new caller with none of those wants st_read_raw_cap below. */
static void st_read_raw(shards *S, const char *name, void *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    st_pread_full(t->fd, out, t->nbytes, t->off, "pread raw");
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
}

/* st_read_raw with the bound made explicit: `cap` is the byte capacity of `out`, in the
 * same position and spirit as st_read_f32_cap's element cap. Refuses rather than writing
 * past the destination, so a caller that has an expected size need not invent its own
 * check -- and one that has none cannot silently do the wrong thing. */
static void st_read_raw_cap(shards *S, const char *name, void *out, int64_t cap, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->nbytes < 0 || t->nbytes > cap) {
        fprintf(stderr, "%s: tensor declares %lld bytes, destination holds %lld — refusing "
                "(untrusted container)\n", name, (long long)t->nbytes, (long long)cap); exit(1); }
    st_read_raw(S, name, out, drop);
}

/* Read an exact physical range from one indexed shard.  This is intentionally
 * separate from tensor slices: callers may use it only after validating a
 * checkpoint-specific packing invariant (for example, two adjacent tensors).
 * The shard membership, file bounds and destination capacity remain enforced
 * here so a malformed header cannot turn that optimization into an unchecked
 * pread. */
static void st_read_range_raw_cap(shards *S, int fd, int64_t off,
                                  int64_t nbytes, void *out, int64_t cap,
                                  int drop, const char *tag) {
    int fidx=S?st_fidx(S,fd):-1;
    if(fidx<0){fprintf(stderr,"physical range uses an unindexed shard fd\n");exit(1);}
    if(off<0||nbytes<0||cap<0||nbytes>cap||
       off>S->sizes[fidx]||nbytes>S->sizes[fidx]-off||
       (uint64_t)nbytes>SIZE_MAX){
        fprintf(stderr,"physical shard range [%lld,+%lld) exceeds file/destination bounds "
                       "(file %lld, cap %lld) — refusing (untrusted container)\n",
                (long long)off,(long long)nbytes,(long long)S->sizes[fidx],
                (long long)cap);exit(1);
    }
    if(nbytes>0&&!out){fprintf(stderr,"physical range has a NULL destination\n");exit(1);}
    if(nbytes>0)st_pread_full(fd,out,nbytes,off,tag&&*tag?tag:"pread physical range");
    if(drop&&nbytes>0)posix_fadvise(fd,off,nbytes,POSIX_FADV_DONTNEED);
}

/* Read-only view of one tensor's exact stored bytes.  Unlike st_read_raw this
 * performs no allocation or copy; unlike a naked mmap pointer it carries the
 * aligned OS view/handle required for cleanup.  Callers must still validate
 * dtype and shape for their own format before consuming `data`. */
typedef struct {
    compat_ro_map map;
    const void *data;
    int64_t nbytes;
} st_mapped_raw;

static int st_map_raw(shards *S, const char *name, st_mapped_raw *out) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); return -1; }
    if (!out || t->nbytes <= 0 || (uint64_t)t->nbytes > SIZE_MAX) {
        errno = EINVAL; return -1;
    }
    memset(out, 0, sizeof(*out));
    if (compat_map_readonly(t->fd, t->off, (size_t)t->nbytes,
                            &out->map, &out->data) != 0) return -1;
    out->nbytes = t->nbytes;
    return 0;
}

static void st_unmap_raw(st_mapped_raw *mapped) {
    if (!mapped) return;
    compat_unmap_readonly(&mapped->map);
    mapped->data = NULL;
    mapped->nbytes = 0;
}

/* Read an exact byte slice from a tensor into a caller-owned buffer.  This is
 * the raw counterpart of st_read_slice_f32: byte_off/nbytes are relative to
 * the tensor (not the shard), and cap is the actual byte capacity of `out`.
 * It is deliberately usable for one-byte F8 rows and eight-byte I64 rows;
 * there is no dtype conversion or element-size assumption here. */
static void st_read_slice_raw_cap(shards *S, const char *name, int64_t byte_off,
                                  int64_t nbytes, void *out, int64_t cap, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (byte_off < 0 || nbytes < 0) {
        fprintf(stderr, "slice %s [%lld,+%lld) has negative offset/length\n",
                name, (long long)byte_off, (long long)nbytes); exit(1);
    }
    if (cap < 0 || nbytes > cap) {
        fprintf(stderr, "slice %s requests %lld bytes, destination holds %lld — refusing\n",
                name, (long long)nbytes, (long long)cap); exit(1);
    }
    if (t->nbytes < 0 || byte_off > t->nbytes || nbytes > t->nbytes - byte_off) {
        fprintf(stderr, "slice %s [%lld,+%lld) out of tensor byte bounds (%lld)\n",
                name, (long long)byte_off, (long long)nbytes, (long long)t->nbytes); exit(1);
    }
    if ((uint64_t)nbytes > SIZE_MAX) {
        fprintf(stderr, "slice %s requests %lld bytes, exceeds host size_t\n",
                name, (long long)nbytes); exit(1);
    }
    if (nbytes > 0 && !out) {
        fprintf(stderr, "slice %s has a NULL destination for %lld bytes\n",
                name, (long long)nbytes); exit(1);
    }
    if (t->off < 0 || byte_off > INT64_MAX - t->off) {
        fprintf(stderr, "slice %s file offset overflows int64\n", name); exit(1);
    }
    int64_t boff = t->off + byte_off;
    st_pread_full(t->fd, out, nbytes, boff, "pread raw slice");
    /* A zero length to posix_fadvise means "through EOF" on POSIX, which is
     * broader than this request.  Do not issue it for an empty slice. */
    if (drop && nbytes > 0) posix_fadvise(t->fd, boff, nbytes, POSIX_FADV_DONTNEED);
}

/* legge una FETTA di un tensore: n_elems a partire dall'elemento elem_off.
 * Serve per gli expert fusi di GLM (un tensore = blocco [E, ...]): si legge il
 * solo expert richiesto via pread del sotto-range, niente lettura dell'intero blocco. */
static void st_read_slice_f32(shards *S, const char *name, int64_t elem_off, int64_t n_elems, float *out, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->dtype >= 3) {   /* stesso motivo di st_read_f32 sopra */
        fprintf(stderr, "slice %s: tensor is %s — not a float tensor\n",
                name, st_dtype_name(t->dtype)); exit(1); }
    int esz = st_dtype_esz(t->dtype);
    if (elem_off < 0 || n_elems < 0 || elem_off > t->numel || n_elems > t->numel - elem_off) {   /* keep the slice inside the tensor; subtraction avoids overflow (#1) */
        fprintf(stderr, "slice %s [%lld,+%lld) out of tensor bounds (numel %lld)\n",
                name, (long long)elem_off, (long long)n_elems, (long long)t->numel); exit(1); }
    if (elem_off > INT64_MAX / esz || n_elems > INT64_MAX / esz ||
        t->off < 0 || elem_off * esz > INT64_MAX - t->off ||
        (uint64_t)(n_elems * esz) > SIZE_MAX || (n_elems > 0 && !out)) {
        fprintf(stderr, "slice %s byte arithmetic/destination is invalid — refusing\n", name); exit(1); }
    int64_t boff = t->off + elem_off * esz, nb = n_elems * esz;
    void *raw = nb ? malloc((size_t)nb) : NULL;
    if (nb && !raw) { fprintf(stderr, "malloc %lld bytes for slice %s failed\n", (long long)nb, name); exit(1); }
    if (nb) st_pread_full(t->fd, raw, nb, boff, "pread slice");   /* dev #331: chunked + EINTR + honest short-read */
    if (nb) {
        if (t->dtype == 2) memcpy(out, raw, (size_t)nb);
        else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = bf16_to_f32(p[i]); }
        else { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = f16_to_f32(p[i]); }
    }
    free(raw);
    if (drop && nb) posix_fadvise(t->fd, boff, nb, POSIX_FADV_DONTNEED);
}

#endif
