/* Separately rounded CPU oracle; real-model verification uses head_bf16_dot. */
#include "../backend_cuda_dsv4.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

static float head_bf16_dot(const uint16_t *w,const float *x,int cols) {
    float sum = 0.f;
    for (int i = 0; i < cols; i++) {
        uint32_t bits = (uint32_t)w[i] << 16;
        float value; memcpy(&value, &bits, sizeof(value));
        volatile float product = value * x[i];
        sum += product;
    }
    return sum;
}
static int head_scores_argmax(const float *scores,int rows,int *token,float *logit) {
    *token = -1; *logit = -FLT_MAX;
    for (int i = 0; i < rows; i++) if (scores[i] > *logit) { *token = i; *logit = scores[i]; }
    return *token < 0 ? -1 : 0;
}

static unsigned seed = 12345;
static unsigned next_random(void) { seed = seed * 1664525u + 1013904223u; return seed; }
static void test_batch(Dsv4CudaTensor *head,const uint16_t *weights,int rows,int cols) {
    enum { MAX_BATCH = 9 };
    float *inputs = malloc((size_t)MAX_BATCH * cols * sizeof(float));
    float *cpu = malloc((size_t)MAX_BATCH * rows * sizeof(float));
    float *gpu = malloc((size_t)MAX_BATCH * rows * sizeof(float));
    int ids[MAX_BATCH]; float values[MAX_BATCH];
    assert(inputs && cpu && gpu);
    for (int item = 0; item < MAX_BATCH; item++)
        for (int col = 0; col < cols; col++)
            inputs[(size_t)item * cols + col] = item == 2 ? -0.f
                : (float)((int)((next_random() >> 8) % 20001) - 10000) / 10003.f;
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row++)
        for (int item = 0; item < MAX_BATCH; item++)
            cpu[(size_t)item * rows + row] = head_bf16_dot(
                weights + (size_t)row * cols, inputs + (size_t)item * cols, cols);
    assert(!dsv4_cuda_head_scores_batch_exact(head, inputs, 0, gpu));
    assert(!dsv4_cuda_head_scores_batch_exact(head, inputs, 129, gpu));
    assert(!dsv4_cuda_head_scores_batch_exact(head, NULL, 3, gpu));
    assert(!dsv4_cuda_head_scores_batch_exact(head, inputs, 3, NULL));
    assert(!dsv4_cuda_head_argmax_batch_exact(head, inputs, 3, NULL, values));
    assert(!dsv4_cuda_head_argmax_batch_exact(head, inputs, 3, ids, NULL));
    int batches[] = {1, 3, 4, 5, 9, 2};
    for (size_t b = 0; b < sizeof(batches)/sizeof(*batches); b++) {
        int batch = batches[b];
        assert(dsv4_cuda_head_scores_batch_exact(head, inputs, batch, gpu));
        assert(!memcmp(cpu, gpu, (size_t)batch * rows * sizeof(float)));
        assert(dsv4_cuda_head_argmax_batch_exact(head, inputs, batch, ids, values));
        for (int item = 0; item < batch; item++) {
            int id; float value;
            assert(!head_scores_argmax(cpu + (size_t)item * rows, rows, &id, &value));
            assert(ids[item] == id && !memcmp(values + item, &value, sizeof(value)));
        }
    }
    free(gpu); free(cpu); free(inputs);
}

static void test_nonfinite(int device) {
    uint16_t weights[32 * 8] = {0};
    float inputs[3 * 8], scores[3 * 32], values[3]; int ids[3];
    for (int i = 0; i < 3 * 8; i++) inputs[i] = 1.f;
    weights[0] = 0x7fc0; weights[8] = 0x7f80; weights[16] = 0x7f80;
    Dsv4CudaTensor *head = NULL;
    assert(dsv4_cuda_upload_head_exact(&head, weights, 32, 8, device));
    assert(dsv4_cuda_head_scores_batch_exact(head, inputs, 3, scores));
    assert(dsv4_cuda_head_argmax_batch_exact(head, inputs, 3, ids, values));
    for (int item = 0; item < 3; item++) {
        assert(isnan(scores[item * 32]) && ids[item] == 1 && values[item] == INFINITY);
    }
    for (int i = 0; i < 3 * 8; i++) inputs[i] = NAN;
    assert(dsv4_cuda_head_argmax_batch_exact(head, inputs, 3, ids, values));
    for (int item = 0; item < 3; item++) assert(ids[item] == -1 && values[item] == -FLT_MAX);
    dsv4_cuda_tensor_free(head);
}

int main(int argc,char **argv) {
#ifndef __AVX2__
    puts("head CUDA oracle requires the AVX2 head contract");
    return 77;
#else
    int devices[] = {0, 5};
    assert(dsv4_cuda_init(devices, 2));
    int shapes[][2] = {{32, 256}, {160, 4104}, {129280, 4096}};
    int shape_count = argc == 2 && !strcmp(argv[1], "--small") ? 2 : 3;
    for (int dev = 0; dev < 2; dev++) test_nonfinite(devices[dev]);
    for (int dev = 0; dev < 2; dev++) for (int shape = 0; shape < shape_count; shape++) {
        int rows = shapes[shape][0], cols = shapes[shape][1];
        uint16_t *weights = malloc((size_t)rows * cols * sizeof(*weights));
        float *input = malloc((size_t)cols * sizeof(*input));
        float *cpu = malloc((size_t)rows * sizeof(*cpu));
        float *gpu = malloc((size_t)rows * sizeof(*gpu));
        assert(weights && input && cpu && gpu);
        for (size_t i = 0; i < (size_t)rows * cols; i++)
            weights[i] = (uint16_t)(0x3800 + (next_random() >> 16) % 1024) | (next_random() & 0x8000);
        // Equal maximum rows exercise first-token tie breaking.
        for (int i = 0; i < 2 * cols; i++) weights[i] = 0x3f80;
        Dsv4CudaTensor *head = NULL;
        assert(!dsv4_cuda_upload_head_exact(&head, weights, rows - 1, cols, devices[dev]));
        assert(!dsv4_cuda_upload_head_exact(&head, weights, rows, cols - 1, devices[dev]));
        assert(!head);
        assert(dsv4_cuda_upload_head_exact(&head, weights, rows, cols, devices[dev]));
        assert(dsv4_cuda_tensor_bytes(head) == (long long)rows * cols * 2);
        assert(!dsv4_cuda_upload_head_exact(&head, weights, rows, cols, devices[dev]));
        assert(!dsv4_cuda_head_scores_exact(head, NULL, gpu));
        for (int trial = 0; trial < 3; trial++) {
            for (int i = 0; i < cols; i++) input[i] = trial == 2 ? 0.f : (next_random() >> 8) % 100001 / 100003.f;
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < rows; row++)
                cpu[row] = head_bf16_dot(weights + (size_t)row * cols, input, cols);
            if (trial == 0) {
                int distinguishes_fma = 0;
                for (int row = 2; row < 32; row++) {
                    float fused = 0.f;
                    for (int i = 0; i < cols; i++) {
                        uint32_t bits = (uint32_t)weights[(size_t)row * cols + i] << 16;
                        float w; memcpy(&w, &bits, sizeof(w));
                        fused = fmaf(w, input[i], fused);
                    }
                    distinguishes_fma += memcmp(&fused, cpu + row, sizeof(fused)) != 0;
                }
                assert(distinguishes_fma > 0);
            }
            assert(dsv4_cuda_head_scores_exact(head, input, gpu));
            assert(!memcmp(cpu, gpu, (size_t)rows * sizeof(*cpu)));
            int a, b; float av, bv;
            assert(!head_scores_argmax(cpu, rows, &a, &av));
            assert(!head_scores_argmax(gpu, rows, &b, &bv));
            assert(a == 0 && b == a && av == bv);
        }
        printf("device=%d rows=%d cols=%d logits=exact trials=3 tie=first zero=exact bytes=%lld\n",
               devices[dev], rows, cols, dsv4_cuda_tensor_bytes(head));
        test_batch(head, weights, rows, cols);
        puts("batched head: logits and compact argmax exact; batch=1,3,4,5,9,2");
        dsv4_cuda_tensor_free(head);
        free(weights); free(input); free(cpu); free(gpu);
    }
    dsv4_cuda_shutdown();
    return 0;
#endif
}
