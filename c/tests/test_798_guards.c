/* Guards for the #798 defect set (unchecked allocation / snprintf results on the
 * config- and shard-parsing path):
 *
 *  - st.h: checked realloc on the colibri.fmt stamp table (fmt_name/fmt_val) growth.
 *  - st.h: checked strdup on the two colibri.fmt stamp-ingest strings.
 *  - json.h: checked realloc on j_parse_val's object (keys/kids) and array (kids)
 *    growth paths.
 *  - route_trace.h: checked snprintf return in rt_save's temp-path build.
 *
 * The allocation guards are exercised with REAL allocation-failure injection, not just a
 * growth-path regression: realloc/strdup are shadowed with the same #define-before-include /
 * #undef-after / call-the-real-function-from-the-shadow trick tests/test_fp8_load.c already
 * uses for mlock/munlock -- except realloc/strdup, unlike mlock/munlock, must still work for
 * every OTHER call in the process, so the shadow counts calls and only fails the ONE ordinal
 * a given subtest targets, passing every other call through to the real allocator. Each
 * fixture is sized so that ordinal is fixed and documented at the point of use. This proves
 * the specific lines the guards touch actually refuse on a failed allocation, not merely
 * that the file still compiles with a NULL-check added.
 *
 * Refusal cases fork a child (mirrors tests/test_st_pread.c's own idiom): the guard code
 * calls exit(1) in place, so the only way to observe it without killing this test binary is
 * a subprocess, with stderr captured through a pipe for message-content assertions. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif

#ifndef _WIN32
/* Forward-declare the shadows under their OWN names before the macros below hijack every
 * "realloc"/"strdup" token that follows (including inside <stdlib.h>/<string.h>'s own
 * declarations, pulled in a second time by st.h -- harmless, include-guarded no-ops -- and
 * every call site inside st.h/json.h). Without this forward declaration the macro-expanded
 * call sites would implicitly declare test_realloc_seam/test_strdup_seam, which for a
 * pointer-returning function is a real bug (truncated to int on LP64), not just a warning. */
static void *test_realloc_seam(void *p, size_t n);
static char *test_strdup_seam(const char *s);
static void *test_malloc_seam(size_t n);
static int   test_snprintf_seam(char *buf, size_t sz, const char *fmt, ...);
#define realloc test_realloc_seam
#define strdup  test_strdup_seam
#define malloc  test_malloc_seam
/* snprintf may already be a (function-like) fortify macro from <stdio.h>; replace it
 * cleanly rather than redefining over it. */
#undef  snprintf
#define snprintf test_snprintf_seam
#endif

#include "../st.h"          /* pulls in json.h and compat.h */
#include "../route_trace.h"

#ifndef _WIN32
#undef realloc
#undef strdup
#undef malloc
#undef snprintf

static long g_realloc_n = 0, g_realloc_fail_at = -1;
static long g_strdup_n  = 0, g_strdup_fail_at  = -1;
static long g_malloc_n  = 0, g_malloc_fail_at  = -1;
static long g_snprintf_n = 0, g_snprintf_fail_at = -1;

/* When the snprintf seam fails, it plants THIS plausible-looking path in the caller's
 * buffer before returning -1 (C says the buffer is indeterminate after an encoding
 * error; "indeterminate" legitimately includes stale valid-looking garbage). An
 * UNGUARDED caller that treats a negative return as success then happily fopen()s it,
 * which is exactly the defect being probed -- and makes it deterministic to observe. */
#define SNPRINTF_SEAM_SENTINEL "test_798_guards_SHOULD_NOT_EXIST.tmp"

/* Defined AFTER the #undef above, so "realloc"/"strdup" here are the real libc functions --
 * declared under their real names by <stdlib.h>/<string.h> (included transitively by st.h,
 * itself pulled in above) before the #define ever took effect. */
static void *test_realloc_seam(void *p, size_t n){
    g_realloc_n++;
    if (g_realloc_n == g_realloc_fail_at) return NULL;
    return realloc(p, n);
}
static char *test_strdup_seam(const char *s){
    g_strdup_n++;
    if (g_strdup_n == g_strdup_fail_at) return NULL;
    return strdup(s);
}
static void *test_malloc_seam(size_t n){
    g_malloc_n++;
    if (g_malloc_n == g_malloc_fail_at) return NULL;
    return malloc(n);
}
static int test_snprintf_seam(char *buf, size_t sz, const char *fmt, ...){
    g_snprintf_n++;
    if (g_snprintf_n == g_snprintf_fail_at) {
        if (sz > sizeof(SNPRINTF_SEAM_SENTINEL)) memcpy(buf, SNPRINTF_SEAM_SENTINEL, sizeof(SNPRINTF_SEAM_SENTINEL));
        return -1;
    }
    va_list ap; va_start(ap, fmt);
    int r = vsnprintf(buf, sz, fmt, ap);
    va_end(ap);
    return r;
}
static void reset_seam(void){
    g_realloc_n = 0; g_realloc_fail_at = -1;
    g_strdup_n  = 0; g_strdup_fail_at  = -1;
    g_malloc_n  = 0; g_malloc_fail_at  = -1;
    g_snprintf_n = 0; g_snprintf_fail_at = -1;
}
#endif

static int g_nfails = 0;
static void check(int cond, const char *what){
    if (!cond) { printf("FAIL: %s\n", what); g_nfails++; }
}

/* ---- safetensors fixture writer: one U8 tensor, optional colibri.fmt stamp for it ---- */
static void write_shard(const char *path, const char *tname, const unsigned char *data, int nbytes,
                         const char *stamp_name, const char *stamp_val) {
    char hdr[1024]; int hl;
    if (stamp_name) {
        hl = snprintf(hdr, sizeof(hdr),
            "{\"%s\":{\"dtype\":\"U8\",\"shape\":[%d],\"data_offsets\":[0,%d]},"
            "\"__metadata__\":{\"colibri.fmt\":\"{\\\"%s\\\":\\\"%s\\\"}\"}}",
            tname, nbytes, nbytes, stamp_name, stamp_val);
    } else {
        hl = snprintf(hdr, sizeof(hdr),
            "{\"%s\":{\"dtype\":\"U8\",\"shape\":[%d],\"data_offsets\":[0,%d]}}",
            tname, nbytes, nbytes);
    }
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

/* ---- globals the forked subtest bodies below read (set by the parent right before fork) ---- */
static char g_dirA[512];
static char g_jsonbuf[512];

/* ==================================================================================
 * st.h fmt-table realloc + fmt strdup guards: one shard, one tensor, one colibri.fmt
 * stamp entry for that same tensor. Call ordering inside st_init_multi for THIS
 * fixture (verified against the current source, documented so a future edit that
 * shifts it is caught by a wrong ordinal rather than a silently-vacuous test):
 *   strdup #1 = st_open_fd's S->paths[] strdup(path)
 *   realloc #1 = fmt_name growth (fmt_cap 0 -> 16)
 *   realloc #2 = fmt_val growth
 *   strdup #2 = strdup(inner->keys[i])   (fmt_name entry, "sn" in the source)
 *   strdup #3 = strdup(v->str)           (fmt_val entry,  "sv" in the source)
 *   strdup #4 = tensor append's t->name = strdup(name)
 * The fixture is a single tensor with a single metadata key, both short strings, so no
 * OTHER realloc/strdup fires first (json.h's own growth needs >8 keys/array elements or
 * a >64-byte unescaped string; none of that applies here).
 * ================================================================================== */
static void fn_load_fmt_fixture(void) {
    shards S; st_init(&S, g_dirA);
    (void)S;
}

static void setup_fmt_fixture(char dir_out[512]) {
    char dir[] = "test_798_guards_fmt_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); exit(1); }
    snprintf(dir_out, 512, "%s", dir);
    char p[600]; snprintf(p, sizeof(p), "%s/model.safetensors", dir);
    unsigned char data[4] = {1, 2, 3, 4};
    write_shard(p, "w", data, 4, "w", "int8-row");
}

static void teardown_fmt_fixture(const char *dir) {
    char p[600]; snprintf(p, sizeof(p), "%s/model.safetensors", dir);
    rmfile(p);
    char cmd[700]; snprintf(cmd, sizeof(cmd), "rmdir %s", dir); if (system(cmd)) {}
}

#ifndef _WIN32
static void oom_subtest(const char *label, long realloc_fail_at, long strdup_fail_at,
                         const char *want_substr1, const char *want_substr2) {
    char dir[512]; setup_fmt_fixture(dir);
    snprintf(g_dirA, sizeof(g_dirA), "%s", dir);

    reset_seam();
    g_realloc_fail_at = realloc_fail_at;
    g_strdup_fail_at  = strdup_fail_at;
    int exit_code; char err[2048];
    run_forked(fn_load_fmt_fixture, &exit_code, err, sizeof(err));

    char msg[256]; snprintf(msg, sizeof(msg), "%s: exits(1) on injected allocation failure", label);
    check(exit_code == 1, msg);
    if (want_substr1) {
        snprintf(msg, sizeof(msg), "%s: message contains '%s'", label, want_substr1);
        check(strstr(err, want_substr1) != NULL, msg);
    }
    if (want_substr2) {
        snprintf(msg, sizeof(msg), "%s: message contains '%s'", label, want_substr2);
        check(strstr(err, want_substr2) != NULL, msg);
    }
    teardown_fmt_fixture(dir);
    /* run_forked's fail_at/counters were set for the CHILD; reset them in the parent too,
     * or the next direct (unforked) call in this process would inherit a stale fail_at
     * and refuse for real. */
    reset_seam();
}
#endif

static void test_fmt_table_injection(void) {
#ifndef _WIN32
    /* fmt_name table growth (realloc #1) */
    oom_subtest("fmt_name realloc", 1, -1, "OOM reallocating format-stamp table", "fmt_name");
    /* fmt_val table growth (realloc #2) */
    oom_subtest("fmt_val realloc", 2, -1, "OOM reallocating format-stamp table", "fmt_val");
    /* strdup of the stamp's tensor-name key (strdup #2, "sn" in the source) */
    oom_subtest("fmt_name strdup (sn)", -1, 2, "OOM duplicating", "colibri.fmt");
    /* strdup of the stamp's format-value string (strdup #3, "sv" in the source) */
    oom_subtest("fmt_val strdup (sv)", -1, 3, "OOM duplicating", "colibri.fmt");
#else
    printf("st.h allocation-failure injection: skipped on Windows (no fork)\n");
#endif
}

/* ---- control: same fixture, no injection -- must load clean and stamp correctly ---- */
static void test_fmt_table_control(void) {
    char dir[512]; setup_fmt_fixture(dir);
    shards S; st_init(&S, dir);
    check(S.n == 1, "fmt control: tensor indexed");
    const char *stamp = st_fmt_stamp(&S, "w");
    check(stamp != NULL && !strcmp(stamp, "int8-row"), "fmt control: stamp ingested correctly");
    teardown_fmt_fixture(dir);
}

/* ==================================================================================
 * json.h object/array growth realloc, exercised directly via json_parse() (no st.h
 * involved) so the failing call's ordinal is trivially known: a fresh process, first
 * realloc call.
 * ================================================================================== */
#ifndef _WIN32
static void fn_json_object_growth(void) {
    char *arena = NULL;
    json_parse(g_jsonbuf, &arena);
}
static void fn_json_array_growth(void) {
    char *arena = NULL;
    json_parse(g_jsonbuf, &arena);
}
#endif

static void test_json_growth_injection(void) {
#ifndef _WIN32
    /* object with 9 keys -- the 9th insertion (index 8, v->len==cap==8) is where
     * j_parse_val's object branch grows keys[] (realloc #1) then kids[] (realloc #2). */
    snprintf(g_jsonbuf, sizeof(g_jsonbuf),
        "{\"k0\":0,\"k1\":1,\"k2\":2,\"k3\":3,\"k4\":4,\"k5\":5,\"k6\":6,\"k7\":7,\"k8\":8}");
    {
        reset_seam(); g_realloc_fail_at = 1;
        int exit_code; char err[1024];
        run_forked(fn_json_object_growth, &exit_code, err, sizeof(err));
        check(exit_code == 1, "json object: keys[] growth exits(1) on injected OOM");
        check(strstr(err, "OOM parsing JSON object") != NULL, "json object: keys[] failure message");
    }
    {
        reset_seam(); g_realloc_fail_at = 2;
        int exit_code; char err[1024];
        run_forked(fn_json_object_growth, &exit_code, err, sizeof(err));
        check(exit_code == 1, "json object: kids[] growth exits(1) on injected OOM");
        check(strstr(err, "OOM parsing JSON object") != NULL, "json object: kids[] failure message");
    }

    /* array with 9 elements -- the 9th insertion grows kids[] once (realloc #1). */
    snprintf(g_jsonbuf, sizeof(g_jsonbuf), "[0,1,2,3,4,5,6,7,8]");
    {
        reset_seam(); g_realloc_fail_at = 1;
        int exit_code; char err[1024];
        run_forked(fn_json_array_growth, &exit_code, err, sizeof(err));
        check(exit_code == 1, "json array: kids[] growth exits(1) on injected OOM");
        check(strstr(err, "OOM parsing JSON array") != NULL, "json array: failure message");
    }
    reset_seam();   /* undo the last fork's fail_at in the parent (see the fmt-table comment) */
#else
    printf("json.h allocation-failure injection: skipped on Windows (no fork)\n");
#endif
}

/* ---- control: same shapes, no injection -- growth path still parses correctly ---- */
static void test_json_growth_control(void) {
    char *arena = NULL;
    jval *obj = json_parse(
        "{\"k0\":0,\"k1\":1,\"k2\":2,\"k3\":3,\"k4\":4,\"k5\":5,\"k6\":6,\"k7\":7,\"k8\":8}", &arena);
    check(obj && obj->t == J_OBJ && obj->len == 9, "json control: 9-key object grows and parses");
    check(obj && obj->len == 9 && obj->kids[8]->num == 8.0, "json control: last key survives the growth");

    char *arena2 = NULL;
    jval *arr = json_parse("[0,1,2,3,4,5,6,7,8]", &arena2);
    check(arr && arr->t == J_ARR && arr->len == 9, "json control: 9-element array grows and parses");
    check(arr && arr->len == 9 && arr->kids[8]->num == 8.0, "json control: last element survives the growth");
}

/* ==================================================================================
 * json.h object/array INITIAL malloc (the cap=8 keys[]/kids[] allocations at the top
 * of j_parse_val's '{' and '[' branches), exercised like the growth reallocs above but
 * through the malloc seam. Ordinals for the minimal fixtures below (j_new is calloc
 * and does not tick the malloc counter; string parsing mallocs come AFTER these):
 *   "{\"a\":1}"  ->  malloc #1 = v->keys, malloc #2 = v->kids   (object branch)
 *   "[1]"        ->  malloc #1 = v->kids                        (array branch)
 * ================================================================================== */
static void test_json_initial_alloc_injection(void) {
#ifndef _WIN32
    snprintf(g_jsonbuf, sizeof(g_jsonbuf), "{\"a\":1}");
    {
        reset_seam(); g_malloc_fail_at = 1;
        int exit_code; char err[1024];
        run_forked(fn_json_object_growth, &exit_code, err, sizeof(err));
        check(exit_code == 1, "json object: initial keys[] malloc exits(1) on injected OOM");
        check(strstr(err, "OOM parsing JSON object") != NULL, "json object: initial keys[] failure message");
    }
    {
        reset_seam(); g_malloc_fail_at = 2;
        int exit_code; char err[1024];
        run_forked(fn_json_object_growth, &exit_code, err, sizeof(err));
        check(exit_code == 1, "json object: initial kids[] malloc exits(1) on injected OOM");
        check(strstr(err, "OOM parsing JSON object") != NULL, "json object: initial kids[] failure message");
    }
    snprintf(g_jsonbuf, sizeof(g_jsonbuf), "[1]");
    {
        reset_seam(); g_malloc_fail_at = 1;
        int exit_code; char err[1024];
        run_forked(fn_json_array_growth, &exit_code, err, sizeof(err));
        check(exit_code == 1, "json array: initial kids[] malloc exits(1) on injected OOM");
        check(strstr(err, "OOM parsing JSON array") != NULL, "json array: initial kids[] failure message");
    }
    reset_seam();
#else
    printf("json.h initial-malloc failure injection: skipped on Windows (no fork)\n");
#endif
}

/* ==================================================================================
 * rt_save's snprintf truncation guard (route_trace.h).
 *
 * The return-value check alone (rt_save == 0 on an overlong path) is NOT proof the guard
 * exists: on hosts whose PATH_MAX is below the truncated length, the unguarded code's
 * fopen() of the truncated path fails with ENAMETOOLONG and rt_save returns 0 anyway, for
 * the wrong reason. What distinguishes the guard is that it refuses BEFORE touching the
 * filesystem and says so on stderr -- so the message assertion below (forked, stderr
 * captured) is the load-bearing check, and the return-value checks are kept as a plain
 * regression on the caller-visible contract.
 * ================================================================================== */
#ifndef _WIN32
static char g_long_path[2200];
static void fn_rt_save_overlong(void) {
    int r = rt_save(g_long_path, 0);
    _exit(r == 0 ? 0 : 43);   /* 43 = saved despite the overlong path -- guard missing */
}
#endif

static void test_route_trace_truncation(void) {
    rt_init("test_798_guards_engine", 1, 1);
    int ids[1] = {0};
    rt_count(0, ids, 1);

    /* control: an ordinary short path saves fine */
    const char *ok_path = "test_798_guards_rt.tmp";
    check(rt_save(ok_path, 1) == 1, "route_trace control: normal-length path saves");
    remove(ok_path);
    { char tmp[600]; snprintf(tmp, sizeof(tmp), "%s.tmp", ok_path); remove(tmp); }

    /* a path long enough that "%s.tmp" (4 extra bytes + NUL) overflows rt_save's tmp[2100] */
    char long_path[2200];
    memset(long_path, 'x', sizeof(long_path) - 1);
    long_path[sizeof(long_path) - 1] = 0;
    check(rt_save(long_path, 1) == 0, "route_trace: overlong path is not saved (quiet)");

#ifndef _WIN32
    /* the load-bearing check: the refusal is the GUARD's (its own diagnostic on stderr),
     * not a downstream fopen() failure on the truncated path. */
    memcpy(g_long_path, long_path, sizeof(g_long_path));
    int exit_code; char err[4096];
    run_forked(fn_rt_save_overlong, &exit_code, err, sizeof(err));
    check(exit_code == 0, "route_trace: overlong path returns 0 (forked)");
    check(strstr(err, "path too long") != NULL,
          "route_trace: refusal is the truncation guard's own diagnostic, not a downstream fopen error");
    check(strstr(err, "[STATS]") != NULL, "route_trace: diagnostic uses the [STATS] prefix");
#else
    printf("route_trace stderr assertion: skipped on Windows (no fork)\n");
#endif
}

/* ==================================================================================
 * rt_save on a NEGATIVE snprintf return (encoding error, not truncation). C leaves the
 * buffer indeterminate on error, so the seam plants a plausible path in it and returns
 * -1: code that compares only `>= sizeof(tmp)` reads -1 as success and fopen()s the
 * planted path. The guard must treat ret < 0 as failure through the same refusal path
 * (return 0, refuse BEFORE touching the filesystem). rt_save's temp-path snprintf is
 * snprintf #1 in the forked child: rt_save makes no other snprintf call before it and
 * the child calls nothing else compiled under the seam.
 * ================================================================================== */
#ifndef _WIN32
static void fn_rt_save_neg_snprintf(void) {
    int r = rt_save("test_798_guards_rt_neg", 0);
    _exit(r == 0 ? 0 : 44);   /* 44 = saved despite the failed snprintf -- guard missing */
}
#endif

static void test_route_trace_negative_snprintf(void) {
#ifndef _WIN32
    rt_init("test_798_guards_engine", 1, 1);
    int ids[1] = {0};
    rt_count(0, ids, 1);

    remove(SNPRINTF_SEAM_SENTINEL);
    remove("test_798_guards_rt_neg");

    reset_seam(); g_snprintf_fail_at = 1;
    int exit_code; char err[2048];
    run_forked(fn_rt_save_neg_snprintf, &exit_code, err, sizeof(err));
    reset_seam();

    check(exit_code == 0, "route_trace: negative snprintf return refuses (rt_save == 0)");
    check(strstr(err, "snprintf error") != NULL,
          "route_trace: refusal is the guard's own negative-return diagnostic");
    FILE *f = fopen(SNPRINTF_SEAM_SENTINEL, "r");
    FILE *g = fopen("test_798_guards_rt_neg", "r");
    check(f == NULL && g == NULL,
          "route_trace: negative snprintf return touched no file (no sentinel, no rename target)");
    if (f) fclose(f);
    if (g) fclose(g);
    remove(SNPRINTF_SEAM_SENTINEL);
    remove("test_798_guards_rt_neg");
#else
    printf("route_trace negative-snprintf injection: skipped on Windows (no fork)\n");
#endif
}

int main(void) {
    test_fmt_table_injection();
    test_fmt_table_control();
    test_json_growth_injection();
    test_json_growth_control();
    test_json_initial_alloc_injection();
    test_route_trace_truncation();
    test_route_trace_negative_snprintf();

    if (g_nfails) { printf("test_798_guards: %d FAILED\n", g_nfails); return 1; }
    printf("test_798_guards: st.h fmt-table realloc/strdup allocation-failure injection, "
           "json.h object/array initial-malloc + growth allocation-failure injection, "
           "route_trace.h snprintf truncation + negative-return guard -- ok\n");
    return 0;
}
