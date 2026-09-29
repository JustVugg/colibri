/* The expert weight format is decided by dtype + element count, and the
 * adapters' numeric_class reports that same decision. Before, the label came
 * from a byte-count int4 probe, so a BF16/F16 container advertised
 * "qwen36/f32-int8/cpu-v1". Every accepted format and each near-miss that must
 * be refused are pinned here, including a genuine F32 tensor (not a supported
 * container format) and a 2-byte-per-element tensor with the wrong dtype. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#ifndef _WIN32
#include <sys/wait.h>
#endif
static int g_fails = 0;
static void expect_bits(const char *what, int dtype, int64_t numel, int64_t nbytes,
                        int64_t want_w, int want){
    st_tensor t = {0};
    t.dtype = dtype; t.numel = numel; t.nbytes = nbytes;
    int got = expert_weight_bits(&t, want_w);
    if (got != want){ g_fails++; printf("FAIL: %-26s got %d want %d\n", what, got, want); }
}
#ifndef _WIN32
/* Exit status of load_expert_merged on one expert: weight of dtype `wdtype`
 * (full numel, 1 byte each) in a container whose format probe said
 * `fp8_container`, with `nsc` f32 scales (-1 = no scale tensor). */
static int load_exit(int wdtype, int fp8_container, int64_t nsc){
    const int hidden = 32, inter = 64;
    const int64_t N = 3 * (int64_t)hidden * inter;
    const char *wn = "model.layers.0.mlp.experts.0.merged_weight";
    const char *sn = "model.layers.0.mlp.experts.0.qs";
    int fd = open("/dev/zero", O_RDONLY), active = 0;
    if (fd < 0) return -1;
    st_tensor tensors[2] = {
        { .name = (char *)wn, .fd = fd, .nbytes = N, .dtype = wdtype, .numel = N },
        { .name = (char *)sn, .fd = fd, .nbytes = nsc * (int64_t)sizeof(float),
          .dtype = 2, .numel = nsc }
    };
    Model m = {0}; Slot s = {0};
    m.c.hidden = hidden; m.c.inter = inter; m.c.expert_fp8 = fp8_container;
    m.active_of = &active; m.S.t = tensors; m.S.n = nsc < 0 ? 1 : 2;
    slot_ensure_allocated(&m, &s);
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0){ load_expert_merged(&m, 0, 0, &s); _exit(0); }
    int status = -1;
    int ok = pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status);
    close(fd); free(s.g); free(s.gs);
    return ok ? WEXITSTATUS(status) : -1;
}
static void expect_load(const char *what, int wdtype, int fp8_container, int64_t nsc, int want){
    int got = load_exit(wdtype, fp8_container, nsc);
    if (got != want){ g_fails++; printf("FAIL: %-34s exit %d want %d\n", what, got, want); }
}
#endif
int main(void){
    const int64_t N = 3 * 64 * 32;             /* 3 * inter * hidden */
    expect_bits("bf16",               0, N,     2 * N, N, 16);
    expect_bits("f16",                1, N,     2 * N, N, 16);
    expect_bits("int8",               3, N,     N,     N, 8);
    expect_bits("packed int4",        3, N / 2, N / 2, N, 4);
    expect_bits("f32 (unsupported)",  2, N,     4 * N, N, 0);
    expect_bits("f32 at 2N bytes",    2, N / 2, 2 * N, N, 0);   /* size matches bf16 */
    expect_bits("fp8 at N bytes",     4, N,     N,     N, 8);   /* dtype distinguishes it from int8 */
    expect_bits("bf16 wrong numel",   0, N / 2, 2 * N, N, 0);
    if (expert_weight_bits(NULL, N) != 0){ g_fails++; printf("FAIL: missing tensor\n"); }
#ifndef _WIN32
    /* 64x32 expert: one 128x128 block per matrix (3 scales) for FP8, per-row
     * 2*64+32 = 160 scales for int8. */
    expect_load("FP8 without scales",               4, 1, -1,  1);
    expect_load("FP8 with block scales",            4, 1, 3,   0);
    expect_load("I8 expert in an FP8 container",    3, 1, 3,   1);
    expect_load("F8 expert in an int8 container",   4, 0, 160, 1);
    expect_load("I8 with per-row scales",           3, 0, 160, 0);
#endif
    if (g_fails){ printf("test_qwen36_expert_format: %d failure(s)\n", g_fails); return 1; }
    printf("OK test_qwen36_expert_format: dtype + numel decide the expert format\n");
    return 0;
}
