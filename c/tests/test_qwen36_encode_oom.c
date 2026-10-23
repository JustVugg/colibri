/* push_id and bpe_piece's growth reallocs were unchecked: a failed realloc left the
 * buffer NULL and the very next store wrote through it. Injects a real allocation
 * failure (not just a code read) and asserts the engine refuses loudly instead of
 * crashing on a NULL write, mirroring tests/test_798_guards.c's shadow-realloc
 * technique -- qwen38.c's own push_id/bpe_piece already guard this exact pattern
 * (q38_encode_realloc / q38_encode_oom); qwen36.c never got the equivalent fix. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif

#ifndef _WIN32
/* Forward-declare the shadow under its own name before the macro below hijacks every
 * "realloc" token that follows, mirroring tests/test_798_guards.c's technique. */
static void *test_realloc_seam(void *p, size_t n);
#define realloc test_realloc_seam
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#ifndef _WIN32
#undef realloc

static long g_realloc_n = 0, g_realloc_fail_at = -1;

static void *test_realloc_seam(void *p, size_t n){
    g_realloc_n++;
    if (g_realloc_n == g_realloc_fail_at) return NULL;
    return realloc(p, n);
}
static void reset_seam(void){ g_realloc_n = 0; g_realloc_fail_at = -1; }
#endif

static int g_nfails = 0;
static void check(int cond, const char *what){
    if (!cond) { printf("FAIL: %s\n", what); g_nfails++; }
}

#ifndef _WIN32
/* fork/pipe/waitpid harness, mirrors tests/test_798_guards.c's inline idiom */
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

/* push_id: cap starts at 2, so the 3rd push (n==cap) is the realloc growth call --
 * the first and only realloc in a fresh child, so ordinal 1 targets it precisely. */
static void fn_push_id_grow(void) {
    int *ids = malloc(2 * sizeof(int));
    int n = 0, cap = 2;
    push_id(&ids, &n, &cap, 10);
    push_id(&ids, &n, &cap, 20);
    push_id(&ids, &n, &cap, 30);   /* n==cap(2) here: triggers the realloc */
    free(ids);
}

/* bpe_piece: scap starts at 16, so a 17-byte piece triggers exactly one realloc (the
 * byte-symbol table growth) before the merge loop is ever reached -- an empty
 * (never-loaded) merge table is never consulted on the failing path. */
static void fn_bpe_piece_grow(void) {
    build_byte_sym();
    int cap = 16, n = 0; int *ids = malloc((size_t)cap * sizeof(int));
    const char *piece = "abcdefghijklmnopq";  /* 17 bytes */
    bpe_piece(piece, (int)strlen(piece), &ids, &n, &cap);
    free(ids);
}

static void test_push_id_oom(void) {
    reset_seam();
    g_realloc_fail_at = 1;
    int exit_code; char err[512];
    run_forked(fn_push_id_grow, &exit_code, err, sizeof(err));
    check(exit_code == 1, "push_id: exits(1) on injected realloc failure");
    check(strstr(err, "OOM reallocating token id buffer") != NULL,
          "push_id: message names the buffer");
    reset_seam();
}

static void test_bpe_piece_oom(void) {
    reset_seam();
    g_realloc_fail_at = 1;
    int exit_code; char err[512];
    run_forked(fn_bpe_piece_grow, &exit_code, err, sizeof(err));
    check(exit_code == 1, "bpe_piece: exits(1) on injected realloc failure");
    check(strstr(err, "OOM reallocating BPE symbol buffer") != NULL,
          "bpe_piece: message names the buffer");
    reset_seam();
}
#endif

/* ---- controls: same growth paths, no injection -- must succeed normally ---- */
static void test_push_id_control(void) {
    int *ids = malloc(2 * sizeof(int));
    int n = 0, cap = 2;
    push_id(&ids, &n, &cap, 10);
    push_id(&ids, &n, &cap, 20);
    push_id(&ids, &n, &cap, 30);
    check(n == 3 && cap >= 3 && ids[0] == 10 && ids[1] == 20 && ids[2] == 30,
          "push_id control: grows and stores correctly without injection");
    free(ids);
}

static void test_bpe_piece_control(void) {
    build_byte_sym();
    int cap = 16, n = 0; int *ids = malloc((size_t)cap * sizeof(int));
    const char *piece = "abcdefghijklmnopq";
    bpe_piece(piece, (int)strlen(piece), &ids, &n, &cap);
    check(n == 17, "bpe_piece control: one id per byte with no merge table loaded");
    free(ids);
}

int main(void) {
#ifndef _WIN32
    test_push_id_oom();
    test_bpe_piece_oom();
#else
    printf("qwen36 encode allocation-failure injection: skipped on Windows (no fork)\n");
#endif
    test_push_id_control();
    test_bpe_piece_control();

    if (g_nfails) { printf("%d check(s) failed\n", g_nfails); return 1; }
    printf("OK\n");
    return 0;
}
