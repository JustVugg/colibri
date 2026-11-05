#include "../backend_metal.h"
#include "../ggml_quants.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t rng_state = 0x6d2b79f5u;
static uint32_t next_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static void fill_inputs(uint8_t *weights, size_t weight_bytes, int I, int O,
                        float *x) {
    for (size_t i = 0; i < weight_bytes; i++) weights[i] = (uint8_t)next_u32();
    for (int row = 0; row < O; row++)
        for (int block = 0; block < I / 256; block++) {
            uint8_t *scale = weights + (size_t)row * (I / 256) * 66 + block * 66;
            scale[0] = 0x00; scale[1] = 0x3c; /* finite f16 1.0 */
        }
    for (int i = 0; i < I; i++)
        x[i] = ((int)(next_u32() % 2001) - 1000) * 0.001f;
}

static int compare(const char *label, const float *actual, const float *reference,
                   int O, double tolerance) {
    double max_abs = 0.0, max_ref = 0.0, sum_sq = 0.0, sum_ref_sq = 0.0;
    for (int i = 0; i < O; i++) {
        double d = fabs((double)actual[i] - reference[i]);
        if (!isfinite(actual[i]) || !isfinite(reference[i])) {
            printf("  %-28s non-finite result MISMATCH\n", label);
            return 0;
        }
        if (d > max_abs) max_abs = d;
        if (fabs(reference[i]) > max_ref) max_ref = fabs(reference[i]);
        sum_sq += d * d;
        sum_ref_sq += (double)reference[i] * reference[i];
    }
    double max_norm = max_abs / (max_ref + 1e-12);
    double rms_norm = sqrt(sum_sq / (sum_ref_sq + 1e-24));
    printf("  %-28s max_abs=%.7g max_norm=%.7g rms_norm=%.7g %s\n",
           label, max_abs, max_norm, rms_norm,
           max_norm <= tolerance ? "ok" : "MISMATCH");
    return max_norm <= tolerance;
}

static int run_shape(int I, int O) {
    const size_t row_bytes = (size_t)(I / 256) * 66;
    const size_t weight_bytes = (size_t)O * row_bytes;
    const size_t page = 16384;
    const size_t alloc_bytes = (weight_bytes + page - 1) & ~(page - 1);
    uint8_t *weights = NULL;
    if (posix_memalign((void **)&weights, page, alloc_bytes)) return 1;
    float *x = (float *)malloc((size_t)I * sizeof(float));
    float *cpu = (float *)malloc((size_t)O * sizeof(float));
    float *exact = (float *)malloc((size_t)O * sizeof(float));
    float *gpu_registered = (float *)malloc((size_t)O * sizeof(float));
    float *gpu_copied = (float *)malloc((size_t)O * sizeof(float));
    if (!x || !cpu || !exact || !gpu_registered || !gpu_copied) {
        free(weights); free(x); free(cpu); free(exact); free(gpu_registered); free(gpu_copied); return 1;
    }
    fill_inputs(weights, weight_bytes, I, O, x);
    gq_act act; memset(&act, 0, sizeof act);
    if (!gq_int_ok(GQ_IQ2_XXS, I) ||
        gq_act_set(&act, x, 1, I) ||
        gq_matmul(cpu, x, weights, GQ_IQ2_XXS, 1, I, O) ||
        gq_matmul_f32(exact, x, weights, GQ_IQ2_XXS, 1, I, O)) {
        fprintf(stderr, "CPU IQ2_XXS reference unavailable for I=%d O=%d\n", I, O);
        gq_act_free(&act); free(weights); free(x); free(cpu); free(exact); free(gpu_registered); free(gpu_copied); return 1;
    }

    coli_metal_register(weights, alloc_bytes);
    int registered_ok = coli_metal_iq2_xxs_gemv(NULL, gpu_registered, act.q, act.d,
                                                weights, I, O);
    coli_metal_unregister(weights);

    ColiMetalTensor *tensor = NULL;
    int copied_ok = coli_metal_iq2_xxs_gemv(&tensor, gpu_copied, act.q, act.d,
                                             weights, I, O);
    int ok = registered_ok && copied_ok;
    if (registered_ok) {
        ok &= compare("registered vs CPU int8", gpu_registered, cpu, O, 1e-5);
        ok &= compare("registered vs dequant F32", gpu_registered, exact, O, 0.01);
    }
    if (copied_ok) {
        ok &= compare("copied vs CPU int8", gpu_copied, cpu, O, 1e-5);
        ok &= compare("copied vs dequant F32", gpu_copied, exact, O, 0.01);
    }
    if (!registered_ok || !copied_ok)
        fprintf(stderr, "Metal IQ2_XXS GEMV unavailable for I=%d O=%d\n", I, O);
    if (copied_ok) coli_metal_tensor_free(tensor);
    gq_act_free(&act);
    free(weights); free(x); free(cpu); free(exact); free(gpu_registered); free(gpu_copied);
    return ok ? 0 : 1;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int compare_double(const void *a, const void *b) {
    double x=*(const double *)a, y=*(const double *)b;
    return (x>y)-(x<y);
}
static double median(double *values, int count) {
    qsort(values,(size_t)count,sizeof *values,compare_double);
    return count&1 ? values[count/2] : (values[count/2-1]+values[count/2])*0.5;
}
static double ns_ms(uint64_t ns) { return (double)ns/1e6; }

static int benchmark_shape(int I, int O, int warm, int cpu_runs) {
    const size_t row_bytes=(size_t)(I/256)*66, weight_bytes=(size_t)O*row_bytes;
    const size_t page=16384, alloc_bytes=(weight_bytes+page-1)&~(page-1);
    uint8_t *weights=NULL;
    if (posix_memalign((void **)&weights,page,alloc_bytes)) return 1;
    float *x=(float *)malloc((size_t)I*sizeof(float));
    float *cpu=(float *)malloc((size_t)O*sizeof(float));
    float *gpu=(float *)malloc((size_t)O*sizeof(float));
    if (!x || !cpu || !gpu) {
        free(weights); free(x); free(cpu); free(gpu); return 1;
    }
    fill_inputs(weights,weight_bytes,I,O,x);
    coli_metal_register(weights,alloc_bytes);

    double start=now_ms();
    gq_act act; memset(&act,0,sizeof act);
    int prep_ok=!gq_act_set(&act,x,1,I);
    double prep_reused_ms=0;
    if (prep_ok) {
        start=now_ms();
        for (int i=0;i<warm;i++)
            if (gq_act_set(&act,x,1,I)) prep_ok=0;
        prep_reused_ms=(now_ms()-start)/warm;
    }

    double prep_fresh_ms=0;
    for (int i=0;i<warm;i++) {
        gq_act fresh; memset(&fresh,0,sizeof fresh);
        start=now_ms();
        if (gq_act_set(&fresh,x,1,I)) prep_ok=0;
        gq_act_free(&fresh);
        prep_fresh_ms+=(now_ms()-start)/warm;
    }

    ColiMetalIq2Timing cold={0};
    start=now_ms();
    int cold_ok=prep_ok && coli_metal_iq2_xxs_gemv_profile(
        NULL,gpu,act.q,act.d,weights,I,O,&cold);
    double cold_wall_ms=now_ms()-start;

    double total[32], resolve[32], weight_copy[32], allocation[32], input_copy[32];
    double submit_wait[32], kernel[32], output_copy[32];
    int warm_ok=1;
    for (int i=0;i<warm;i++) {
        ColiMetalIq2Timing t={0};
        if (!coli_metal_iq2_xxs_gemv_profile(NULL,gpu,act.q,act.d,weights,I,O,&t))
            warm_ok=0;
        total[i]=ns_ms(t.total_ns); resolve[i]=ns_ms(t.weight_resolve_ns);
        weight_copy[i]=ns_ms(t.weight_copy_ns); allocation[i]=ns_ms(t.scratch_allocation_ns);
        input_copy[i]=ns_ms(t.activation_copy_ns); submit_wait[i]=ns_ms(t.encode_submit_wait_ns);
        kernel[i]=ns_ms(t.gpu_execution_ns); output_copy[i]=ns_ms(t.output_copy_ns);
    }

    int cpu_ok=prep_ok;
    double cpu_kernel[32], cpu_e2e[32];
    for (int i=0;i<cpu_runs;i++) {
        start=now_ms();
        gq_matmul_act(cpu,&act,weights,GQ_IQ2_XXS,O);
        cpu_kernel[i]=(now_ms()-start);
        start=now_ms();
        if (gq_matmul(cpu,x,weights,GQ_IQ2_XXS,1,I,O)) cpu_ok=0;
        cpu_e2e[i]=(now_ms()-start);
    }
    double total_med=median(total,warm), resolve_med=median(resolve,warm);
    double weight_copy_med=median(weight_copy,warm), alloc_med=median(allocation,warm);
    double input_copy_med=median(input_copy,warm), submit_med=median(submit_wait,warm);
    double kernel_med=median(kernel,warm), output_med=median(output_copy,warm);
    double cpu_kernel_med=median(cpu_kernel,cpu_runs), cpu_e2e_med=median(cpu_e2e,cpu_runs);
    double submit_non_gpu=submit_med>kernel_med?submit_med-kernel_med:0;
    double effective_gb_s=kernel_med>0 ? (double)weight_bytes/(kernel_med*1e6) : 0;
    double metal_e2e=prep_fresh_ms+total_med;
    coli_metal_unregister(weights);

    printf("\nIQ2_XXS %s GEMV (I=%d O=%d, weights=%.2f MiB, %d warm samples):\n",
           (I>=4096 && O>=2048)?"large":"small",I,O,(double)weight_bytes/(1024.0*1024.0),warm);
    printf("  activation preparation: reused quantization=%.4f ms; fresh allocation+quantize+free=%.4f ms\n",
           prep_reused_ms,prep_fresh_ms);
    printf("  cold first call: wall=%.4f ms, profile total=%.4f ms, resolve=%.4f ms, scratch allocation=%.4f ms, submit+sync=%.4f ms, GPU=%.4f ms\n",
           cold_wall_ms,ns_ms(cold.total_ns),ns_ms(cold.weight_resolve_ns),
           ns_ms(cold.scratch_allocation_ns),ns_ms(cold.encode_submit_wait_ns),
           ns_ms(cold.gpu_execution_ns));
    printf("  warm medians: total=%.4f ms, resolve=%.4f ms, weight-copy=%.4f ms, scratch-allocation=%.4f ms, activation-copy=%.4f ms, submit+sync=%.4f ms, GPU=%.4f ms, output-copy=%.4f ms\n",
           total_med,resolve_med,weight_copy_med,alloc_med,input_copy_med,
           submit_med,kernel_med,output_med);
    printf("  inferred non-GPU dispatch/sync=%.4f ms; effective weight-read rate=%.2f GB/s\n",
           submit_non_gpu,effective_gb_s);
    printf("  CPU: prequantized kernel=%.4f ms; full gq_matmul=%.4f ms; Metal end-to-end estimate=%.4f ms; CPU/Metal=%.2fx\n",
           cpu_kernel_med,cpu_e2e_med,metal_e2e,cpu_e2e_med/metal_e2e);
    gq_act_free(&act);
    free(weights); free(x); free(cpu); free(gpu);
    return cold_ok && warm_ok && cpu_ok ? 0 : 1;
}

int main(void) {
    if (!coli_metal_init()) {
        fprintf(stderr, "Metal IQ2_XXS pipeline unavailable\n");
        return 1;
    }
    int failed = run_shape(256, 1) | run_shape(1024, 13) | run_shape(4096, 32);
    failed |= benchmark_shape(256,1,16,8);
    failed |= benchmark_shape(1024,13,16,8);
    failed |= benchmark_shape(4096,2048,16,5);
    failed |= benchmark_shape(5120,17408,8,3);
    coli_metal_shutdown();
    return failed ? 1 : 0;
}
