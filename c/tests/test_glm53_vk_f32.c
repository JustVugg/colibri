/* The production GLM-5.3 dispatch must upload resident f32 matrices for both
 * decode and prefill, and reserve their bytes before sizing the expert tier.
 * A numerical comparison alone would pass when every call falls back to CPU.
 * An f32 matrix whose floats are all bf16 widened (a bf16 checkpoint read into
 * f32) goes up as fmt 11, one whose floats are all f16 (an f16 checkpoint) as
 * fmt 14, each in half the bytes, and the budget counts it so; any other stays
 * fmt 10. */
#define main glm53_main_unused
#include "../glm53.c"
#undef main

enum { I = 64, O = 128, S = 64 };

static int run_case(const char *name, float *weights, int want_fmt, const float *x) {
    static float cpu[S * O], gpu[S * O];
    int failed = 0;
    Mat w = {.fmt = 0, .rows = O, .columns = I, .resident = 1, .f = weights};
    if (coli_vk_f32_fmt(weights, (size_t)I * O) != want_fmt) {
        fprintf(stderr, "FAIL: %s matrix taken as fmt %d, not %d\n", name, coli_vk_f32_fmt(weights, (size_t)I * O), want_fmt);
        return 1;
    }
    if (glm53_mat_dev_bytes(&w) != coli_vk_tensor_payload(want_fmt, I, O, 0) ||
        glm53_mat_dev_bytes(&w) < (size_t)I * O * (want_fmt == 10 ? 4 : 2)) {
        fprintf(stderr, "FAIL: %s residents counted at %zu bytes in the dense GPU budget\n", name, glm53_mat_dev_bytes(&w));
        return 1;
    }
    const int rows[] = {1, 5, S};
    for (size_t k = 0; k < sizeof(rows) / sizeof(rows[0]); k++) {
        const int n = rows[k];
        g_vk_dense = 0;
        mm(cpu, &w, x, n);
        g_vk_dense = 1;
        unsigned long long before = coli_vk_matmul_calls();
        mm(gpu, &w, x, n);
        if (coli_vk_matmul_calls() == before) {
            fprintf(stderr, "FAIL: %s S=%d f32 dispatch did no Vulkan matmul\n", name, n);
            failed = 1;
        }
        for (int i = 0; i < n * O; i++) {
            if (!isfinite(gpu[i]) || fabsf(cpu[i] - gpu[i]) > 1e-5f * (1.f + fabsf(cpu[i]))) {
                fprintf(stderr, "FAIL: %s S=%d output[%d]: CPU=%g Vulkan=%g\n", name, n, i, cpu[i], gpu[i]);
                failed = 1;
                break;
            }
        }
    }
    ColiVkTensorInfo ti;
    if (!w.vk || !coli_vk_tensor_info((ColiVkTensor *)w.vk, &ti) || ti.fmt != want_fmt) {
        fprintf(stderr, "FAIL: %s matrix not on the device as fmt %d\n", name, want_fmt);
        failed = 1;
    }
    /* Exact prefix/pin checks fix the kernel to GEMV: a cold prompt and its
     * resumed suffix must then have identical arithmetic at every batch size.
     * Default GEMM/GEMV selection is checked numerically above instead. */
    const char *gemm_min = getenv("COLI_VK_GEMM_MIN_S");
    if (gemm_min && !strcmp(gemm_min, "0")) {
        static float single[S * O];
        for (int t = 0; t < S; t++) mv(single + t * O, &w, x + t * I);
        if (memcmp(single, gpu, sizeof(single))) {
            fprintf(stderr, "FAIL: %s GEMV rows differ between a full batch and individual rows\n", name);
            failed = 1;
        }
        mm(gpu, &w, x + (S - 5) * I, 5);
        if (memcmp(single + (S - 5) * O, gpu, 5 * O * sizeof(float))) {
            fprintf(stderr, "FAIL: %s GEMV resumed suffix differs from the full batch\n", name);
            failed = 1;
        }
    }
    /* Nonresident weights still belong to the CPU and take no dense budget. */
    coli_vk_tensor_free((ColiVkTensor *)w.vk);
    w.vk = NULL; w.resident = 0;
    g_vk_dense = 0;
    mm(cpu, &w, x, S);
    g_vk_dense = 1;
    unsigned long long before = coli_vk_matmul_calls();
    mm(gpu, &w, x, S);
    if (coli_vk_matmul_calls() != before || glm53_mat_dev_bytes(&w) != 0 || w.vk ||
        memcmp(cpu, gpu, sizeof(cpu))) {
        fprintf(stderr, "FAIL: nonresident %s matrix changed placement or result\n", name);
        failed = 1;
    }
    return failed;
}

int main(void) {
    static float f32[I * O], bf16[I * O], f16[I * O], x[S * I];
    for (int i = 0; i < I * O; i++) f32[i] = (float)(i % 19 - 9) / 19.f;
    /* ten bits after the point: every one an f16, 999/1024 and others no bf16 */
    for (int i = 0; i < I * O; i++) f16[i] = (float)((i * 37) % 2001 - 1000) / 1024.f;
    for (int i = 0; i < I * O; i++) {   /* the same values cut to bf16, as a bf16 checkpoint gives them */
        uint32_t u; memcpy(&u, &f32[i], 4); u &= 0xffff0000u; memcpy(&bf16[i], &u, 4);
    }
    for (int i = 0; i < S * I; i++) x[i] = (float)(i % 17 - 8) / 17.f;
    if (!coli_vk_init("shaders/qmatmul.spv")) {
        fputs("FAIL: no Vulkan device\n", stderr);
        return 1;
    }
    int failed = run_case("f32", f32, 10, x);
    failed |= run_case("bf16-exact", bf16, 11, x);
    failed |= run_case("f16-exact", f16, 14, x);
    coli_vk_shutdown();
    if (!failed) puts("PASS: glm53 f32 Vulkan decode, prefill and dense budget (fmt 10, and fmt 11 or 14 where exact)");
    return failed;
}
