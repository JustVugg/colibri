/* The production GLM-5.3 dispatch must upload resident f32 matrices for both
 * decode and prefill, and reserve their bytes before sizing the expert tier.
 * A numerical comparison alone would pass when every call falls back to CPU. */
#define main glm53_main_unused
#include "../glm53.c"
#undef main

int main(void) {
    enum { I = 64, O = 128, S = 64 };
    float weights[I * O], x[S * I], cpu[S * O], gpu[S * O];
    for (int i = 0; i < I * O; i++) weights[i] = (float)(i % 19 - 9) / 19.f;
    for (int i = 0; i < S * I; i++) x[i] = (float)(i % 17 - 8) / 17.f;
    Mat w = {.fmt = 0, .rows = O, .columns = I, .resident = 1, .f = weights};
    if (glm53_mat_dev_bytes(&w) != sizeof(weights)) {
        fputs("FAIL: f32 residents missing from the dense GPU budget\n", stderr);
        return 1;
    }
    if (!coli_vk_init("shaders/qmatmul.spv")) {
        fputs("FAIL: no Vulkan device\n", stderr);
        return 1;
    }
    int failed = 0;
    const int rows[] = {1, 5, S};
    for (size_t k = 0; k < sizeof(rows) / sizeof(rows[0]); k++) {
        const int n = rows[k];
        g_vk_dense = 0;
        mm(cpu, &w, x, n);
        g_vk_dense = 1;
        unsigned long long before = coli_vk_matmul_calls();
        mm(gpu, &w, x, n);
        if (coli_vk_matmul_calls() == before) {
            fprintf(stderr, "FAIL: S=%d f32 dispatch did no Vulkan matmul\n", n);
            failed = 1;
        }
        for (int i = 0; i < n * O; i++) {
            if (!isfinite(gpu[i]) || fabsf(cpu[i] - gpu[i]) > 1e-5f * (1.f + fabsf(cpu[i]))) {
                fprintf(stderr, "FAIL: S=%d output[%d]: CPU=%g Vulkan=%g\n", n, i, cpu[i], gpu[i]);
                failed = 1;
                break;
            }
        }
    }
    /* Exact prefix/pin checks fix the kernel to GEMV: a cold prompt and its
     * resumed suffix must then have identical arithmetic at every batch size.
     * Default GEMM/GEMV selection is checked numerically above instead. */
    const char *gemm_min = getenv("COLI_VK_GEMM_MIN_S");
    if (gemm_min && !strcmp(gemm_min, "0")) {
        float single[S * O];
        for (int t = 0; t < S; t++) mv(single + t * O, &w, x + t * I);
        if (memcmp(single, gpu, sizeof(single))) {
            fputs("FAIL: f32 GEMV rows differ between a full batch and individual rows\n", stderr);
            failed = 1;
        }
        mm(gpu, &w, x + (S - 5) * I, 5);
        if (memcmp(single + (S - 5) * O, gpu, 5 * O * sizeof(float))) {
            fputs("FAIL: f32 GEMV resumed suffix differs from the full batch\n", stderr);
            failed = 1;
        }
    }
    /* Nonresident weights still belong to the CPU and take no dense budget. */
    w.resident = 0;
    unsigned long long before = coli_vk_matmul_calls();
    mm(gpu, &w, x, S);
    if (coli_vk_matmul_calls() != before || glm53_mat_dev_bytes(&w) != 0 ||
        memcmp(cpu, gpu, sizeof(cpu))) {
        fputs("FAIL: nonresident f32 matrix changed placement or result\n", stderr);
        failed = 1;
    }
    coli_vk_tensor_free((ColiVkTensor *)w.vk);
    coli_vk_shutdown();
    if (!failed) puts("PASS: glm53 f32 Vulkan decode, prefill and dense budget");
    return failed;
}
