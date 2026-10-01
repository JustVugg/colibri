/* llama.c's load_t must check a tensor's declared element count against the
 * count the config implies BEFORE it allocates or reads. The forward pass reads
 * every weight with CONFIG dims: a short tensor sized from its own header was
 * read past its end at the first rmsnorm_row() (heap overflow under ASan), and
 * an oversized one would be silently truncated.
 * Case 0: right-sized -- loads, and the forward's own consumer reads it cleanly.
 * Case 1: one element where the config implies D -- refused by name.
 * Case 2: 2*D elements where the config implies D -- refused by name.
 *
 * A refusal exits the process in place, so each case runs in a forked child
 * with stderr captured through a pipe -- the idiom tests/test_798_guards.c and
 * tests/test_st_pread.c use -- and an overrun in the child is what the ASan
 * job watches for. tests/st_fixture.h, which this test used to include, was
 * removed upstream, so the one-tensor F32 shard is written inline instead. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif

#define main llama_main_unused
#include "../llama.c"
#undef main

enum { D = 64 };
static const int64_t sizes[3] = { D, 1, 2 * D };
static const char *g_dir;

/* One F32 tensor `name` of n elements in <dir>/model.safetensors. */
static void write_shard(const char *dir, const char *name, int64_t n) {
    char path[1024]; snprintf(path, sizeof path, "%s/model.safetensors", dir);
    char hdr[256];
    int hl = snprintf(hdr, sizeof hdr,
        "{\"%s\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[0,%lld]}}",
        name, (long long)n, (long long)(n * 4));
    uint64_t hlen = (uint64_t)hl;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(&hlen, 8, 1, f);
    fwrite(hdr, 1, (size_t)hl, f);
    for (int64_t i = 0; i < n; i++) { float v = 1.f; fwrite(&v, sizeof v, 1, f); }
    fclose(f);
}

static int child(void) {
    static Model m; memset(&m, 0, sizeof m);
    st_init(&m.S, g_dir);
    float *w = load_t(&m, "w", D), x[D], out[D];
    for (int i = 0; i < D; i++) x[i] = 1.f;
    rmsnorm_row(out, x, w, D, 1e-6f);   /* the forward pass's own consumer, at config D */
    fprintf(stderr, "loaded; out[0] %g\n", out[0]);
    free(w);
    return 0;
}

#ifndef _WIN32
/* Fork the child, capture its stderr, return its exit status. */
static int run_forked(char *errbuf, size_t errbuf_sz) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { snprintf(errbuf, errbuf_sz, "pipe failed"); return -1; }
    pid_t pid = fork();
    if (pid < 0) { snprintf(errbuf, errbuf_sz, "fork failed"); return -1; }
    if (pid == 0) {
        dup2(pipefd[1], 2); close(pipefd[0]); close(pipefd[1]);
        _exit(child());
    }
    close(pipefd[1]);
    size_t off = 0; ssize_t n;
    while (off < errbuf_sz - 1 &&
           (n = read(pipefd[0], errbuf + off, errbuf_sz - 1 - off)) > 0) off += (size_t)n;
    errbuf[off] = 0; close(pipefd[0]);
    int status = 0; waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}
#endif

int main(int argc, char **argv) {
    (void)argc; (void)argv;
#ifdef _WIN32
    printf("SKIP test_llama_read_trust: refusal exits the process, no fork on Windows\n");
    return 0;
#else
    int fails = 0;
    for (int c = 0; c < 3; c++) {
        char dir[] = "test_llama_read_trust_XXXXXX";
        if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
        g_dir = dir;
        write_shard(dir, "w", sizes[c]);
        char log[16384] = {0};
        int got = run_forked(log, sizeof log);
        int ok = c ? (got != 0 && strstr(log, "w: ")) :
                     (got == 0 && !strstr(log, "Sanitizer"));
        if (!ok) {
            fails++;
            printf("FAIL: case %d (%lld elements, config %d): exit %d, want %s\n"
                   "--- child stderr ---\n%s\n",
                   c, (long long)sizes[c], D, got,
                   c ? "a refusal naming w" : "a clean load", log);
        }
        char p[600]; snprintf(p, sizeof p, "%s/model.safetensors", dir);
        remove(p); rmdir(dir);
    }
    if (fails) { printf("test_llama_read_trust: %d failure(s)\n", fails); return 1; }
    printf("OK test_llama_read_trust: right-sized tensor loads, short and oversized tensors refused before any config-sized read\n");
    return 0;
#endif
}
