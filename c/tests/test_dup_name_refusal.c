/* Duplicate-tensor-name refusal (st.h, st_init_multi's shard-ingestion loop):
 *
 *  - named refusal when two indexed shards carry the same tensor name -- the message must
 *    name the tensor and BOTH shard files;
 *  - the same refusal under a HOSTILE tensor name (printf metacharacters including %n,
 *    non-UTF8 bytes, oversized length) -- the message prints attacker-controlled bytes,
 *    so this asserts they only ever travel as a %s argument, never as format;
 *  - a no-duplicates control (two shards, distinct names, both findable);
 *  - checked malloc on the detector's own scratch table: dup_idx's initial allocation
 *    (unconditional, every load) and ndup's growth allocation (only when S->cap doubles
 *    past its initial 4096 tensors), exercised with REAL allocation-failure injection.
 *
 * The malloc injection is targeted by exact allocation SIZE rather than call ordinal:
 * st.h/json.h make several unrelated malloc() calls per load (json arena, the safetensors
 * header buffer, S->hidx, ...) whose count shifts with fixture shape. dup_idx's and ndup's
 * sizes are fully determined by S->cap/dup_cap arithmetic for a given fixture, so failing
 * the FIRST call of that exact size hits the intended allocation without an ordinal scan,
 * and without colliding with S->hidx's malloc (built AFTER dup_idx is freed).
 *
 * The shadow uses the same #define-before-include / #undef-after / call-the-real-function
 * trick tests/test_fp8_load.c already uses for mlock/munlock -- except malloc must still
 * work for every OTHER call in the process, so the shadow fails only the first call of the
 * targeted size and passes everything else through to the real allocator.
 *
 * Refusal cases fork a child (mirrors tests/test_st_pread.c's own idiom): the refusal code
 * calls exit(1) in place, so the only way to observe it without killing this test binary is
 * a subprocess, with stderr captured through a pipe for message-content assertions. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif

#ifndef _WIN32
/* Forward-declare the shadow under its OWN name before the macro below hijacks every
 * "malloc" token that follows (including inside <stdlib.h>'s own declarations, pulled in a
 * second time by st.h -- harmless, include-guarded no-ops -- and every call site inside
 * st.h/json.h). Without this forward declaration the macro-expanded call sites would
 * implicitly declare test_malloc_seam, which for a pointer-returning function is a real bug
 * (truncated to int on LP64), not just a warning. */
static void *test_malloc_seam(size_t n);
#define malloc test_malloc_seam
#endif

#include "../st.h"          /* pulls in json.h and compat.h */

#ifndef _WIN32
#undef malloc

/* Size-targeted, not ordinal (see file header) -- 0 disables injection, and only the FIRST
 * malloc of exactly this size fails, so a coincidental later call of the same size (there
 * isn't one before the target for these fixtures, but the guard costs nothing) still
 * succeeds. */
static size_t g_malloc_fail_size = 0;
static int    g_malloc_fail_hit  = 0;

/* Defined AFTER the #undef above, so "malloc" here is the real libc function -- declared
 * under its real name by <stdlib.h> (included transitively by st.h, itself pulled in above)
 * before the #define ever took effect. */
static void *test_malloc_seam(size_t n){
    if (g_malloc_fail_size != 0 && n == g_malloc_fail_size && !g_malloc_fail_hit) {
        g_malloc_fail_hit = 1;
        return NULL;
    }
    return malloc(n);
}
static void reset_seam(void){
    g_malloc_fail_size = 0; g_malloc_fail_hit = 0;
}
#endif

static int g_nfails = 0;
static void check(int cond, const char *what){
    if (!cond) { printf("FAIL: %s\n", what); g_nfails++; }
}

/* ---- safetensors fixture writer: one U8 tensor ---- */
static void write_shard(const char *path, const char *tname, const unsigned char *data, int nbytes) {
    char hdr[1024]; int hl;
    hl = snprintf(hdr, sizeof(hdr),
        "{\"%s\":{\"dtype\":\"U8\",\"shape\":[%d],\"data_offsets\":[0,%d]}}",
        tname, nbytes, nbytes);
    uint64_t hlen = (uint64_t)hl;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    fwrite(data, 1, (size_t)nbytes, f);
    fclose(f);
}

static void rmfile(const char *p) { unlink(p); }

#ifndef _WIN32
/* ---- fork/pipe/waitpid harness, mirrors tests/test_st_pread.c's inline idiom ---- */
static void run_forked(void (*fn)(void), int *exit_code, char *errbuf, size_t errbuf_sz) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { *exit_code = -1; errbuf[0] = 0; return; }
    pid_t pid = fork();
    if (pid < 0) { *exit_code = -1; errbuf[0] = 0; return; }
    if (pid == 0) {
        dup2(pipefd[1], 2); close(pipefd[0]); close(pipefd[1]);
        fn();
        _exit(42);  /* reaching here means fn() did NOT refuse -- a bug, not a crash */
    }
    close(pipefd[1]);
    size_t off = 0; ssize_t n;
    while (off < errbuf_sz - 1 && (n = read(pipefd[0], errbuf + off, errbuf_sz - 1 - off)) > 0) off += (size_t)n;
    errbuf[off] = 0;
    close(pipefd[0]);
    int status = 0; waitpid(pid, &status, 0);
    *exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
#endif

/* ---- global the forked subtest bodies below read (set by the parent right before fork) ---- */
static char g_dirA[512];

/* ==================================================================================
 * Duplicate tensor name across two indexed shards -- must refuse by name, naming the
 * tensor and BOTH shard files.
 * ================================================================================== */
#ifndef _WIN32
static void fn_dup_collision(void) {
    shards S; st_init(&S, g_dirA);
    (void)S;
}
#endif

static void test_duplicate_name(void) {
#ifndef _WIN32
    char dir[] = "test_dup_name_refusal_dup_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); check(0, "dup: mkdtemp"); return; }
    snprintf(g_dirA, sizeof(g_dirA), "%s", dir);
    char pa[600], pb[600];
    snprintf(pa, sizeof(pa), "%s/a-00000.safetensors", dir);
    snprintf(pb, sizeof(pb), "%s/b-00000.safetensors", dir);
    unsigned char data[4] = {1, 2, 3, 4};
    write_shard(pa, "shared.weight", data, 4);
    write_shard(pb, "shared.weight", data, 4);

    int exit_code; char err[2048];
    reset_seam();
    run_forked(fn_dup_collision, &exit_code, err, sizeof(err));
    check(exit_code == 1, "dup: duplicate tensor name across shards exits(1)");
    check(strstr(err, "shared.weight") != NULL, "dup: message names the tensor");
    check(strstr(err, "a-00000.safetensors") != NULL, "dup: message names the earlier shard");
    check(strstr(err, "b-00000.safetensors") != NULL, "dup: message names the shard that triggered the refusal");
    check(strstr(err, "duplicate tensor name") != NULL, "dup: message states the reason");

    rmfile(pa); rmfile(pb);
    char cmd[700]; snprintf(cmd, sizeof(cmd), "rmdir %s", dir); if (system(cmd)) {}
#else
    printf("duplicate-name refusal: skipped on Windows (no fork)\n");
#endif
}

/* ==================================================================================
 * Hostile tensor name in the duplicate fixture: the refusal message prints an
 * attacker-controlled string (the tensor name comes straight from the shard header), so
 * the name must only ever travel as a %s ARGUMENT, never as format. This fixture's name
 * carries printf metacharacters (%s/%n/%x -- %n is the one that turns a format-string
 * slip into a write primitive), non-UTF8 bytes, and an oversized-but-valid length; the
 * refusal must still fire cleanly (exit 1, no signal) and render the name verbatim.
 * ================================================================================== */
static void build_hostile_name(char *out, size_t cap) {
    size_t off = 0;
    off += (size_t)snprintf(out + off, cap - off, "evil.%%s.%%n.%%x.");
    out[off++] = (char)0x80; out[off++] = (char)0xff; out[off++] = (char)0xc3; out[off++] = '.';
    while (off < 200 && off < cap - 1) out[off++] = 'A';   /* oversized, still a legal JSON key */
    out[off] = 0;
}

static void test_hostile_name_duplicate(void) {
#ifndef _WIN32
    char hostile[256]; build_hostile_name(hostile, sizeof(hostile));

    char dir[] = "test_dup_name_refusal_evil_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); check(0, "hostile: mkdtemp"); return; }
    snprintf(g_dirA, sizeof(g_dirA), "%s", dir);
    char pa[600], pb[600];
    snprintf(pa, sizeof(pa), "%s/a-00000.safetensors", dir);
    snprintf(pb, sizeof(pb), "%s/b-00000.safetensors", dir);
    unsigned char data[4] = {1, 2, 3, 4};
    write_shard(pa, hostile, data, 4);
    write_shard(pb, hostile, data, 4);

    int exit_code; char err[4096];
    reset_seam();
    run_forked(fn_dup_collision, &exit_code, err, sizeof(err));
    check(exit_code == 1, "hostile: refusal exits(1) cleanly -- no crash, no signal");
    check(strstr(err, "%s.%n.%x") != NULL, "hostile: printf metacharacters render verbatim, not interpreted");
    { char hb[5] = {(char)0x80, (char)0xff, (char)0xc3, '.', 0};
      check(strstr(err, hb) != NULL, "hostile: non-UTF8 bytes pass through untouched"); }
    check(strstr(err, hostile) != NULL, "hostile: full oversized name renders verbatim");
    check(strstr(err, "duplicate tensor name") != NULL, "hostile: message still states the reason");

    rmfile(pa); rmfile(pb);
    char cmd[700]; snprintf(cmd, sizeof(cmd), "rmdir %s", dir); if (system(cmd)) {}
#else
    printf("hostile-name refusal: skipped on Windows (no fork)\n");
#endif
}

/* ---- control: two shards, DISTINCT tensor names -- must load clean, both findable ---- */
static void test_control_no_duplicates(void) {
    char dir[] = "test_dup_name_refusal_ctrl_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); check(0, "control: mkdtemp"); return; }
    char pa[600], pb[600];
    snprintf(pa, sizeof(pa), "%s/a-00000.safetensors", dir);
    snprintf(pb, sizeof(pb), "%s/b-00000.safetensors", dir);
    unsigned char da[4] = {1, 2, 3, 4}, db[3] = {9, 8, 7};
    write_shard(pa, "tensor.one", da, 4);
    write_shard(pb, "tensor.two", db, 3);

    shards S; st_init(&S, dir);
    check(S.n == 2, "control: both tensors indexed, no false-positive refusal");
    st_tensor *t1 = st_find(&S, "tensor.one");
    st_tensor *t2 = st_find(&S, "tensor.two");
    check(t1 != NULL && t1->nbytes == 4, "control: tensor.one resolves");
    check(t2 != NULL && t2->nbytes == 3, "control: tensor.two resolves");

    rmfile(pa); rmfile(pb);
    char cmd[700]; snprintf(cmd, sizeof(cmd), "rmdir %s", dir); if (system(cmd)) {}
}

/* ==================================================================================
 * The detector's OWN two malloc() calls: dup_idx's initial allocation (unconditional,
 * every st_init call) and ndup's growth allocation (only on S->cap growth past its
 * initial 4096 tensors).
 * ================================================================================== */
static void fn_dupidx_load(void) {
    shards S; st_init(&S, g_dirA);
    (void)S;
}

/* One shard, one U8 tensor -- the smallest fixture that reaches st_init_multi's dup
 * detector at all. S->cap starts at a fixed 4096, so dup_idx's size never depends on
 * fixture shape: dup_cap = next pow2 >= cap*2 = 8192, malloc size = 8192*sizeof(int). */
static void setup_dupidx_fixture(char dir_out[512]) {
    char dir[] = "test_dup_name_refusal_dupidx_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); exit(1); }
    snprintf(dir_out, 512, "%s", dir);
    char p[600]; snprintf(p, sizeof(p), "%s/a-00000.safetensors", dir);
    unsigned char data[4] = {1, 2, 3, 4};
    write_shard(p, "w", data, 4);
}
static void teardown_dupidx_fixture(const char *dir) {
    char p[600]; snprintf(p, sizeof(p), "%s/a-00000.safetensors", dir);
    rmfile(p);
    char cmd[700]; snprintf(cmd, sizeof(cmd), "rmdir %s", dir); if (system(cmd)) {}
}

/* One shard, 4097 distinct 1-byte U8 tensors -- the 4097th tensor pushes S->n past the
 * fixed initial S->cap(4096), forcing S->cap to double and, in lockstep, the dup
 * detector's own ndup growth: new_dup_cap = dup_cap*2 = 16384, malloc size =
 * 16384*sizeof(int) -- distinct from dup_idx's 8192*sizeof(int), so size-targeting hits
 * this specific growth call and not the initial one. */
static void write_shard_many(const char *path, int ntensors) {
    size_t hdr_cap = (size_t)ntensors * 96 + 64;
    char *hdr = (char *)malloc(hdr_cap);
    if (!hdr) { perror("malloc"); exit(1); }
    size_t off = 0;
    off += (size_t)snprintf(hdr + off, hdr_cap - off, "{");
    for (int i = 0; i < ntensors; i++) {
        off += (size_t)snprintf(hdr + off, hdr_cap - off,
            "%s\"t%d\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[%d,%d]}",
            i == 0 ? "" : ",", i, i, i + 1);
    }
    off += (size_t)snprintf(hdr + off, hdr_cap - off, "}");
    uint64_t hlen = (uint64_t)off;
    unsigned char *data = (unsigned char *)calloc(1, (size_t)ntensors);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, off, f);
    fwrite(data, 1, (size_t)ntensors, f);
    fclose(f);
    free(hdr); free(data);
}
static void setup_ndup_fixture(char dir_out[512]) {
    char dir[] = "test_dup_name_refusal_ndup_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); exit(1); }
    snprintf(dir_out, 512, "%s", dir);
    char p[600]; snprintf(p, sizeof(p), "%s/a-00000.safetensors", dir);
    write_shard_many(p, 4097);
}
static void teardown_ndup_fixture(const char *dir) {
    char p[600]; snprintf(p, sizeof(p), "%s/a-00000.safetensors", dir);
    rmfile(p);
    char cmd[700]; snprintf(cmd, sizeof(cmd), "rmdir %s", dir); if (system(cmd)) {}
}

#ifndef _WIN32
static void test_malloc_injection(void) {
    /* dup_idx: initial allocation, every st_init call. */
    {
        char dir[512]; setup_dupidx_fixture(dir);
        snprintf(g_dirA, sizeof(g_dirA), "%s", dir);
        reset_seam();
        g_malloc_fail_size = 8192 * sizeof(int);
        int exit_code; char err[2048];
        run_forked(fn_dupidx_load, &exit_code, err, sizeof(err));
        check(exit_code == 1, "malloc: dup_idx initial allocation exits(1) on injected OOM");
        check(strstr(err, "OOM allocating duplicate-tensor-name detector") != NULL,
              "malloc: dup_idx failure message names the detector");
        teardown_dupidx_fixture(dir);
        reset_seam();   /* undo the fork's fail_size in the parent */
    }
    /* ndup: growth allocation, only when S->cap doubles past 4096 tensors. */
    {
        char dir[512]; setup_ndup_fixture(dir);
        snprintf(g_dirA, sizeof(g_dirA), "%s", dir);
        reset_seam();
        g_malloc_fail_size = 16384 * sizeof(int);
        int exit_code; char err[4096];
        run_forked(fn_dupidx_load, &exit_code, err, sizeof(err));
        check(exit_code == 1, "malloc: ndup growth allocation exits(1) on injected OOM");
        check(strstr(err, "OOM growing duplicate-tensor-name detector") != NULL,
              "malloc: ndup failure message names the detector");
        teardown_ndup_fixture(dir);
        reset_seam();
    }
}
#endif

/* ---- malloc control: same two fixtures, no injection -- must load clean ---- */
static void test_malloc_control(void) {
    {
        char dir[512]; setup_dupidx_fixture(dir);
        shards S; st_init(&S, dir);
        check(S.n == 1, "malloc control: dup_idx fixture loads clean with no injection");
        teardown_dupidx_fixture(dir);
    }
    {
        char dir[512]; setup_ndup_fixture(dir);
        shards S; st_init(&S, dir);
        check(S.n == 4097, "malloc control: 4097-tensor fixture (forces ndup growth) loads clean");
        teardown_ndup_fixture(dir);
    }
}

int main(void) {
    test_duplicate_name();
    test_hostile_name_duplicate();
    test_control_no_duplicates();
#ifndef _WIN32
    test_malloc_injection();
#else
    printf("malloc allocation-failure injection: skipped on Windows (no fork)\n");
#endif
    test_malloc_control();

    if (g_nfails) { printf("test_dup_name_refusal: %d FAILED\n", g_nfails); return 1; }
    printf("test_dup_name_refusal: duplicate-tensor-name refusal, hostile-name refusal, "
           "no-duplicate control, detector malloc allocation-failure injection (dup_idx/ndup) "
           "-- ok\n");
    return 0;
}
