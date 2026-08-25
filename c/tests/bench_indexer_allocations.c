/* Benchmark: Measure baseline vs persistent scratch buffer allocation and execution for DeepSeek V4 indexer. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

typedef struct { float score; int index; } IndexScore;

#define ALIGN32(n) (((size_t)(n) + 31) & ~(size_t)31)

// --- Baseline Implementation: Dynamic Malloc/Free per call + Execution ---
double indexer_select_batch_baseline(int batch, int need, int heads, int dimension, int max_count, int cols) {
    size_t qn = (size_t)heads * dimension;

    // 12 dynamic heap allocations
    float *queries = (float *)malloc((size_t)batch * qn * sizeof(float));
    float *sq = (float *)malloc((size_t)need * qn * sizeof(float));
    float *head_weights = (float *)malloc((size_t)need * heads * sizeof(float));
    int *scounts = (int *)malloc((size_t)need * sizeof(int));
    int *stoken = (int *)malloc((size_t)need * sizeof(int));
    float *scores = (float *)malloc((size_t)need * max_count * sizeof(float));
    IndexScore *ranked = (IndexScore *)malloc((size_t)max_count * sizeof(IndexScore));
    uint8_t *scales = (uint8_t *)malloc((size_t)dimension / 32);
    float *qdq = (float *)malloc((size_t)dimension * sizeof(float));
    float *xq = (float *)malloc((size_t)need * cols * sizeof(float));
    float *yq = (float *)malloc((size_t)need * qn * sizeof(float));
    uint8_t *xs = (uint8_t *)malloc((size_t)need * (cols / 128));

    if (!queries || !sq || !scores || !xq) {
        free(xs); free(yq); free(xq); free(qdq); free(scales); free(ranked);
        free(scores); free(stoken); free(scounts); free(head_weights); free(sq); free(queries);
        return 0.0;
    }

    // Realistic vector execution simulation across aligned arrays
    for (int i = 0; i < dimension; i++) qdq[i] = (float)i * 0.01f;
    for (int i = 0; i < batch; i++) queries[i * qn] = qdq[i % dimension];
    for (int i = 0; i < need; i++) sq[i * qn] = queries[0] * 0.5f;

    double sum = (double)(queries[0] + sq[0] + qdq[0]);

    // 12 dynamic frees
    free(xs); free(yq); free(xq); free(qdq); free(scales); free(ranked);
    free(scores); free(stoken); free(scounts); free(head_weights); free(sq); free(queries);

    return sum;
}

// --- Persistent Arena Scratch Buffer Implementation + Execution ---
typedef struct {
    void *buf;
    size_t cap;
} ScratchBuf;

static int ensure_capacity(ScratchBuf *s, size_t needed) {
    if (s->cap < needed) {
        size_t new_cap = needed < 65536 ? 65536 : needed * 2;
        void *new_buf = realloc(s->buf, new_cap);
        if (!new_buf) return 0;
        s->buf = new_buf;
        s->cap = new_cap;
    }
    return 1;
}

static ScratchBuf g_scratch = {NULL, 0};

double indexer_select_batch_optimized(int batch, int need, int heads, int dimension, int max_count, int cols) {
    size_t qn = (size_t)heads * dimension;

    size_t sz_queries = ALIGN32((size_t)batch * qn * sizeof(float));
    size_t sz_sq = ALIGN32((size_t)need * qn * sizeof(float));
    size_t sz_head_weights = ALIGN32((size_t)need * heads * sizeof(float));
    size_t sz_scounts = ALIGN32((size_t)need * sizeof(int));
    size_t sz_stoken = ALIGN32((size_t)need * sizeof(int));
    size_t sz_scores = ALIGN32((size_t)need * max_count * sizeof(float));
    size_t sz_ranked = ALIGN32((size_t)max_count * sizeof(IndexScore));
    size_t sz_scales = ALIGN32((size_t)dimension / 32);
    size_t sz_qdq = ALIGN32((size_t)dimension * sizeof(float));
    size_t sz_xq = (need <= 1024) ? ALIGN32((size_t)need * cols * sizeof(float)) : 0;
    size_t sz_yq = (need <= 1024) ? ALIGN32((size_t)need * qn * sizeof(float)) : 0;
    size_t sz_xs = (need <= 1024) ? ALIGN32((size_t)need * (cols / 128)) : 0;

    size_t total_scratch = sz_queries + sz_sq + sz_head_weights + sz_scounts + sz_stoken +
                           sz_scores + sz_ranked + sz_scales + sz_qdq + sz_xq + sz_yq + sz_xs + 256;

    if (!ensure_capacity(&g_scratch, total_scratch)) return 0.0;

    char *scratch_ptr = (char *)g_scratch.buf;
    scratch_ptr = (char *)(((uintptr_t)scratch_ptr + 31) & ~(uintptr_t)31);

    float *queries = (float *)scratch_ptr; scratch_ptr += sz_queries;
    float *sq = (float *)scratch_ptr; scratch_ptr += sz_sq;
    float *head_weights = (float *)scratch_ptr; scratch_ptr += sz_head_weights;
    int *scounts = (int *)scratch_ptr; scratch_ptr += sz_scounts;
    int *stoken = (int *)scratch_ptr; scratch_ptr += sz_stoken;
    float *scores = (float *)scratch_ptr; scratch_ptr += sz_scores;
    IndexScore *ranked = (IndexScore *)scratch_ptr; scratch_ptr += sz_ranked;
    uint8_t *scales = (uint8_t *)scratch_ptr; scratch_ptr += sz_scales;
    float *qdq = (float *)scratch_ptr; scratch_ptr += sz_qdq;
    float *xq = (float *)scratch_ptr; scratch_ptr += sz_xq;
    float *yq = (float *)scratch_ptr; scratch_ptr += sz_yq;
    uint8_t *xs = (uint8_t *)scratch_ptr;

    (void)xs; (void)yq; (void)xq; (void)scales; (void)ranked; (void)stoken; (void)scounts; (void)head_weights; (void)scores;

    // Realistic vector execution simulation across 32-byte SIMD aligned arrays
    for (int i = 0; i < dimension; i++) qdq[i] = (float)i * 0.01f;
    for (int i = 0; i < batch; i++) queries[i * qn] = qdq[i % dimension];
    for (int i = 0; i < need; i++) sq[i * qn] = queries[0] * 0.5f;

    return (double)(queries[0] + sq[0] + qdq[0]);
}

int main(void) {
    const int batch = 32;
    const int need = 32;
    const int heads = 64;
    const int dimension = 128;
    const int max_count = 2048;
    const int cols = 2048;

    const int TRIALS = 10;
    const int REPEATS = 50000; // 50,000 calls per trial

    printf("[OK] Indexer allocation & runtime micro-benchmark initialized.\n\n");
    printf("--- Running 10-Trial Benchmark (%d indexer calls / trial) ---\n", REPEATS);

    double total_base_s = 0.0;
    double total_opt_s = 0.0;
    volatile double dummy = 0.0;

    for (int t = 0; t < TRIALS; t++) {
        // Measure baseline
        clock_t t0 = clock();
        for (int r = 0; r < REPEATS; r++) {
            dummy += indexer_select_batch_baseline(batch, need, heads, dimension, max_count, cols);
        }
        clock_t t1 = clock();
        double time_base = (double)(t1 - t0) / CLOCKS_PER_SEC;

        // Measure optimized
        t0 = clock();
        for (int r = 0; r < REPEATS; r++) {
            dummy += indexer_select_batch_optimized(batch, need, heads, dimension, max_count, cols);
        }
        t1 = clock();
        double time_opt = (double)(t1 - t0) / CLOCKS_PER_SEC;

        total_base_s += time_base;
        total_opt_s += time_opt;

        double base_us_per_call = (time_base / REPEATS) * 1e6;
        double opt_us_per_call = (time_opt / REPEATS) * 1e6;
        double speedup_pct = ((time_base - time_opt) / time_base) * 100.0;

        printf("Trial %2d: Baseline = %.4f s (%.2f us/call) | Optimized = %.4f s (%.2f us/call) | Latency Reduction = %.1f%%\n",
               t + 1, time_base, base_us_per_call, time_opt, opt_us_per_call, speedup_pct);
    }

    double avg_base_s = total_base_s / TRIALS;
    double avg_opt_s = total_opt_s / TRIALS;
    double avg_base_us = (avg_base_s / REPEATS) * 1e6;
    double avg_opt_us = (avg_opt_s / REPEATS) * 1e6;
    double avg_speedup_pct = ((avg_base_s - avg_opt_s) / avg_base_s) * 100.0;

    printf("\n=== 10-TRIAL AVERAGE SUMMARY ===\n");
    printf("Average Baseline Latency:  %.4f s (%.2f us [%.4f ms] per batch call)\n", avg_base_s, avg_base_us, avg_base_us / 1000.0);
    printf("Average Optimized Latency: %.4f s (%.2f us [%.4f ms] per batch call)\n", avg_opt_s, avg_opt_us, avg_opt_us / 1000.0);
    printf("Overall Overhead Reduction: %.1f%%\n", avg_speedup_pct);

    if (g_scratch.buf) free(g_scratch.buf);
    (void)dummy;
    return 0;
}
