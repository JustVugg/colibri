/* Run the Qwen-Image CPU and Metal projection paths with matching inputs. */
#ifndef COLI_METAL
#include <stdio.h>
int main(void){
    fputs("Build with METAL=1 to run the Qwen-Image Metal benchmark.\n", stderr);
    return 1;
}
#else
#define main qwenimage_main
#include "../qwenimage.c"
#undef main

static uint32_t sample(uint32_t *state){
    *state = *state * 1664525u + 1013904223u;
    return *state >> 16;
}

static int bench(int m, int k, int n){
    Lin l = {0};
    l.m.N = n; l.m.K = k;
    lin_alloc_i8(&l, n, k, 1);
    if (!l.metal_weight) return 1;
    size_t inputs = (size_t)m * k, outputs = (size_t)m * n;
    float *x = fmalloc(inputs), *cpu = fmalloc(outputs), *gpu = fmalloc(outputs);
    uint32_t seed = 42;
    int8_t *weight = (int8_t *)l.m.w;
    float *scales = (float *)l.m.sc;
    for (size_t i = 0; i < (size_t)k * n; i++) weight[i] = (int)(sample(&seed) % 255) - 127;
    for (int i = 0; i < n; i++) scales[i] = (1 + sample(&seed) % 32) / 4096.0f;
    for (size_t i = 0; i < inputs; i++) x[i] = ((int)(sample(&seed) % 2048) - 1024) / 512.0f;
    int failed = qi_metal_linear(gpu, x, 8, &l) != 0;
    for (int i = -1; i < 3 && !failed; i++) {
        double start, cpu_s, gpu_s;
        int ok;
        if (i % 2 == 0) {
            start = now_s(); qi_gemm(cpu, x, m, &l.m, NULL); cpu_s = now_s() - start;
            start = now_s(); ok = qi_metal_linear(gpu, x, m, &l); gpu_s = now_s() - start;
        } else {
            start = now_s(); ok = qi_metal_linear(gpu, x, m, &l); gpu_s = now_s() - start;
            start = now_s(); qi_gemm(cpu, x, m, &l.m, NULL); cpu_s = now_s() - start;
        }
        double max_diff = 0, max_cpu = 0;
        failed = !ok;
        for (size_t p = 0; p < outputs && !failed; p++) {
            failed = !isfinite(cpu[p]) || !isfinite(gpu[p]);
            max_diff = fmax(max_diff, fabs((double)cpu[p] - gpu[p]));
            max_cpu = fmax(max_cpu, fabs(cpu[p]));
        }
        double normalized = max_diff / fmax(max_cpu, 1e-30);
        failed |= normalized > 2e-5;
        printf("%d,%d,%d,%d,%.9f,%.9f,%.9g,%s\n",
               m, k, n, i, cpu_s, gpu_s, normalized, failed ? "fail" : "ok");
        fflush(stdout);
    }
    free(x); free(cpu); free(gpu);
    lin_free(&l);
    return failed;
}

int main(void){
    if (!q36_gpu_init()) { fprintf(stderr, "Metal unavailable\n"); return 1; }
    q36_gpu_set_exact(0);
    g_metal_ready = 1;
    puts("M,K,N,run,cpu_s,metal_s,normalized_max_error,quality");
    int rc = bench(33, 64, 96) || bench(1024, 4096, 4096) ||
             bench(1024, 4096, 12288) || bench(1024, 12288, 4096) ||
             bench(4096, 4096, 4096);
    qi_metal_cleanup();
    return rc;
}
#endif
