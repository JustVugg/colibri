/* The expert weight format is decided by dtype + element count, and the
 * adapters' numeric_class reports that same decision. Before, the label came
 * from a byte-count int4 probe, so a BF16/F16 container advertised
 * "qwen36/f32-int8/cpu-v1". Every accepted format and each near-miss that must
 * be refused are pinned here, including a genuine F32 tensor (not a supported
 * container format) and a 2-byte-per-element tensor with the wrong dtype. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
static int g_fails = 0;
static void expect_bits(const char *what, int dtype, int64_t numel, int64_t nbytes,
                        int64_t want_w, int want){
    st_tensor t = {0};
    t.dtype = dtype; t.numel = numel; t.nbytes = nbytes;
    int got = expert_weight_bits(&t, want_w);
    if (got != want){ g_fails++; printf("FAIL: %-26s got %d want %d\n", what, got, want); }
}
int main(void){
    const int64_t N = 3 * 64 * 32;             /* 3 * inter * hidden */
    expect_bits("bf16",               0, N,     2 * N, N, 16);
    expect_bits("f16",                1, N,     2 * N, N, 16);
    expect_bits("int8",               3, N,     N,     N, 8);
    expect_bits("packed int4",        3, N / 2, N / 2, N, 4);
    expect_bits("f32 (unsupported)",  2, N,     4 * N, N, 0);
    expect_bits("f32 at 2N bytes",    2, N / 2, 2 * N, N, 0);   /* size matches bf16 */
    expect_bits("fp8 at N bytes",     4, N,     N,     N, 0);   /* size matches int8 */
    expect_bits("bf16 wrong numel",   0, N / 2, 2 * N, N, 0);
    if (expert_weight_bits(NULL, N) != 0){ g_fails++; printf("FAIL: missing tensor\n"); }
    if (g_fails){ printf("test_qwen36_expert_format: %d failure(s)\n", g_fails); return 1; }
    printf("OK test_qwen36_expert_format: dtype + numel decide the expert format\n");
    return 0;
}
