/* decode_id_to_bytes() must stay inside the vocab piece it decodes.
 *
 * The <0xXX> byte-fallback recogniser probed pc[0..5], but the && chain only
 * establishes that pc[0..2] are non-NUL before it reads pc[5], so a populated
 * 3- or 4-byte piece starting "<0x" was read past its own allocation. On an
 * ordinary heap block that 1-2 byte overread is silent, so the piece is placed
 * flush against a PROT_NONE guard page, turning it into a real fault, and each
 * decode runs in a forked child so the fault is a named failure. A vacant vocab
 * slot (an id inside g_tok_n that the tokenizer never populated) must decode to
 * nothing rather than dereference NULL. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
static int g_fails = 0;
static int decode_isolated(int id, unsigned char *out, int *outn, int *signalled){
    int fd[2];
    if (pipe(fd) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) { close(fd[0]); close(fd[1]); return -1; }
    if (pid == 0){
        close(fd[0]);
        unsigned char msg[2 + 256]; int tn = 0;
        decode_id_to_bytes(id, msg + 2, &tn);          /* the read under test */
        if (tn < 0 || tn > 256) _exit(3);
        msg[0] = (unsigned char)(tn & 0xFF);
        msg[1] = (unsigned char)((tn >> 8) & 0xFF);
        ssize_t w = write(fd[1], msg, (size_t)(2 + tn));
        (void)w;
        _exit(0);
    }
    close(fd[1]);
    unsigned char msg[2 + 256];
    ssize_t got = read(fd[0], msg, sizeof msg);
    close(fd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    *signalled = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
    if (*signalled || WEXITSTATUS(status) != 0 || got < 2) return -1;
    int tn = (int)msg[0] | ((int)msg[1] << 8);
    if (got < 2 + tn) return -1;
    memcpy(out, msg + 2, (size_t)tn);
    *outn = tn;
    return 0;
}
static void expect_decode(int id, const char *what, const char *want){
    unsigned char out[256]; int n = -1, sig = 0;
    int nwant = (int)strlen(want);
    if (decode_isolated(id, out, &n, &sig) != 0){
        g_fails++;
        printf("FAIL: %-28s id %d %s (signal %d)\n", what, id,
               sig ? "died" : "did not return cleanly", sig);
    } else if (n != nwant || memcmp(out, want, (size_t)nwant) != 0){
        g_fails++;
        printf("FAIL: %-28s id %d gave %d bytes, want %d [%s]\n", what, id, n, nwant, want);
    }
}
/* Decodes `id` with its piece's NUL on the last byte before a guard page. */
static void expect_no_read_past_piece(int id, const char *piece){
    long pg = sysconf(_SC_PAGESIZE);
    char *base = mmap(NULL, (size_t)pg * 2, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED || mprotect(base + pg, (size_t)pg, PROT_NONE) != 0){
        g_fails++; printf("FAIL: %-28s could not arm a guard page\n", piece); return;
    }
    size_t len = strlen(piece);
    char *slot = base + pg - (long)(len + 1);
    memcpy(slot, piece, len + 1);
    char *saved = g_tok[id];
    g_tok[id] = slot;
    char what[64]; snprintf(what, sizeof what, "guarded '%s'", piece);
    expect_decode(id, what, piece);
    g_tok[id] = saved;
    munmap(base, (size_t)pg * 2);
}
int main(void){
    const char *path = "test_qwen36_tok_decode.json";
    /* ids 2..29 are vacant; 30 is a short piece starting "<0x", 31 the real form */
    const char *json =
        "{\"model\":{\"vocab\":{\"X\":0,\".\":1,\"<0x\":30,\"<0x41>\":31},\"merges\":[]}}";
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(json, 1, strlen(json), f) != strlen(json) || fclose(f)){
        printf("FAIL: cannot write %s\n", path); return 1;
    }
    load_tokenizer(path);
    remove(path);
    if (g_tok_n != 32 || g_tok[20] != NULL){
        printf("FAIL: fixture g_tok_n=%d, want 32 with id 20 vacant\n", g_tok_n); return 1;
    }
    expect_decode(0,  "in-vocab control", "X");
    expect_decode(31, "'<0x41>' byte fallback", "A");   /* gate must not suppress it */
    expect_decode(30, "short '<0x' piece", "<0x");
    expect_no_read_past_piece(30, "<0x");               /* pc[5] is 2 past the NUL */
    expect_no_read_past_piece(30, "<0x4");              /* pc[5] is 1 past the NUL */
    expect_decode(20, "vacant slot", "");
    expect_decode(32, "out-of-range id", "");
    if (g_fails){ printf("test_qwen36_tok_decode: %d failure(s)\n", g_fails); return 1; }
    printf("OK test_qwen36_tok_decode: decode stays inside the piece\n");
    return 0;
}
