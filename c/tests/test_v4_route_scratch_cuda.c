/* Real DLL regression for the native router's retained activation. The
 * existing transient-allocation path is the numerical oracle; GPU/CPU
 * reduction association is intentionally outside this test's contract. */
#include "../backend_cuda_dsv4.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../compat.h"

int coli_v4_test_gpu_route_mirrors(float *, int *, const float *, void *, void *,
                                  const int *, int, float);
void coli_v4_test_gpu_route_scratch_clear(void);
long long coli_v4_test_gpu_route_scratch_capacity(int);

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "router scratch: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    exit(1); \
} } while (0)

enum { EXPERTS = 256, TOPK = 6, SMALL = 128, LARGE = 4096, INPUTS = 4 };
static const int fixed_ids[TOPK] = {5, 77, 128, 202, 17, 251};

typedef struct Fixture {
    Dsv4CudaTensor *small_gate, *large_gate, *bias, *invalid_bias;
    float input[INPUTS][LARGE];
} Fixture;

static void fixture_init(Fixture *fixture, int cycle) {
    memset(fixture, 0, sizeof(*fixture));
    float *weights = malloc((size_t)EXPERTS * LARGE * sizeof(*weights));
    CHECK(weights);
    for (int shape = 0; shape < 2; shape++) {
        int dimension = shape ? LARGE : SMALL;
        for (int expert = 0; expert < EXPERTS; expert++) {
            for (int i = 0; i < dimension; i++) {
                uint32_t bits = (uint32_t)(expert + 1) * UINT32_C(0x9e3779b9) ^
                                (uint32_t)(i + 5) * UINT32_C(0x85ebca6b);
                weights[(size_t)expert * dimension + i] =
                    (float)((int)(bits >> 24) - 128) * (1.f / 2048.f);
            }
        }
        Dsv4CudaTensor **gate = shape ? &fixture->large_gate : &fixture->small_gate;
        CHECK(dsv4_cuda_upload_f32(gate, weights, EXPERTS, dimension, 0));
    }
    free(weights);
    float bias[EXPERTS];
    for (int expert = 0; expert < EXPERTS; expert++)
        bias[expert] = (float)(expert % 17 - 8) * (1.f / 256.f);
    CHECK(dsv4_cuda_upload_f32(&fixture->bias, bias, EXPERTS, 1, 0));
    CHECK(dsv4_cuda_upload_f32(&fixture->invalid_bias, bias, 32, 1, 0));
    for (int sample = 0; sample < INPUTS; sample++) {
        for (int i = 0; i < LARGE; i++) {
            uint32_t bits = (uint32_t)(i + 3) * UINT32_C(0x27d4eb2d) ^
                (uint32_t)(sample + 7 + cycle * 11) * UINT32_C(0x165667b1);
            fixture->input[sample][i] =
                (float)((int)(bits >> 25) - 64) * (1.f / 128.f);
        }
    }
}

static void fixture_free(Fixture *fixture) {
    dsv4_cuda_tensor_free(fixture->invalid_bias);
    dsv4_cuda_tensor_free(fixture->bias);
    dsv4_cuda_tensor_free(fixture->large_gate);
    dsv4_cuda_tensor_free(fixture->small_gate);
}

static void valid_result(const int ids[TOPK], const float weights[TOPK],
                         const int *fixed) {
    float sum = 0.f;
    for (int i = 0; i < TOPK; i++) {
        CHECK(ids[i] >= 0 && ids[i] < EXPERTS);
        CHECK(isfinite(weights[i]) && weights[i] >= 0.f);
        if (fixed) CHECK(ids[i] == fixed[i]);
        for (int j = 0; j < i; j++) CHECK(ids[i] != ids[j]);
        sum += weights[i];
    }
    CHECK(fabsf(sum - 2.5f) < 0.00001f);
}

static void compare_paths(Fixture *fixture, int dimension, int with_bias,
                          int with_fixed, int sample) {
    Dsv4CudaTensor *gate = dimension == LARGE ? fixture->large_gate : fixture->small_gate;
    Dsv4CudaTensor *bias = with_bias ? fixture->bias : NULL;
    const int *fixed = with_fixed ? fixed_ids : NULL;
    int old_ids[TOPK], new_ids[TOPK];
    float old_weights[TOPK], new_weights[TOPK];
    long long previous_capacity = coli_v4_test_gpu_route_scratch_capacity(0);
    CHECK(setenv("V4_GPU_ROUTE_REUSE", "0", 1) == 0);
    CHECK(coli_v4_test_gpu_route_mirrors(old_weights, old_ids,
        fixture->input[sample], gate, bias, fixed, dimension, 2.5f) == 0);
    CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == previous_capacity);
    CHECK(setenv("V4_GPU_ROUTE_REUSE", "1", 1) == 0);
    CHECK(coli_v4_test_gpu_route_mirrors(new_weights, new_ids,
        fixture->input[sample], gate, bias, fixed, dimension, 2.5f) == 0);
    CHECK(coli_v4_test_gpu_route_scratch_capacity(0) >= dimension);
    CHECK(memcmp(old_ids, new_ids, sizeof(old_ids)) == 0);
    CHECK(memcmp(old_weights, new_weights, sizeof(old_weights)) == 0);
    valid_result(new_ids, new_weights, fixed);
}

static void refused_then_valid(Fixture *fixture) {
    int ids[TOPK], saved_ids[TOPK];
    float weights[TOPK], saved_weights[TOPK], short_input[SMALL];
    memcpy(short_input, fixture->input[0], sizeof(short_input));
    for (int i = 0; i < TOPK; i++) { ids[i] = -123; weights[i] = -456.f; }
    memcpy(saved_ids, ids, sizeof(ids));
    memcpy(saved_weights, weights, sizeof(weights));
    CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == LARGE);
    for (int mode = 0; mode < 2; mode++) {
        CHECK(setenv("V4_GPU_ROUTE_REUSE", mode ? "1" : "0", 1) == 0);
        /* The retained capacity must not authorize reading beyond the
         * caller's smaller logical input, even after a larger route. */
        CHECK(coli_v4_test_gpu_route_mirrors(weights, ids, short_input,
            fixture->large_gate, NULL, NULL, SMALL, 2.5f) != 0);
        CHECK(memcmp(ids, saved_ids, sizeof(ids)) == 0);
        CHECK(memcmp(weights, saved_weights, sizeof(weights)) == 0);
        /* Upload succeeds, then backend bias validation refuses the route:
         * failure must drain before the next valid call reuses scratch. */
        CHECK(coli_v4_test_gpu_route_mirrors(weights, ids, fixture->input[1],
            fixture->large_gate, fixture->invalid_bias, NULL, LARGE, 2.5f) != 0);
        CHECK(memcmp(ids, saved_ids, sizeof(ids)) == 0);
        CHECK(memcmp(weights, saved_weights, sizeof(weights)) == 0);
        CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == LARGE);
        compare_paths(fixture, LARGE, 1, 0, mode);
    }
}

static double now_ms(void) {
    struct timespec stamp;
    CHECK(timespec_get(&stamp, TIME_UTC) == TIME_UTC);
    return stamp.tv_sec * 1000. + stamp.tv_nsec * 0.000001;
}

static void timing(Fixture *fixture) {
    int ids[TOPK];
    float weights[TOPK];
    for (int mode = 0; mode < 2; mode++) {
        coli_v4_test_gpu_route_scratch_clear();
        CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == 0);
        CHECK(setenv("V4_GPU_ROUTE_REUSE", mode ? "1" : "0", 1) == 0);
        for (int i = 0; i < 8; i++)
            CHECK(coli_v4_test_gpu_route_mirrors(weights, ids,
                fixture->input[i % INPUTS], fixture->large_gate, fixture->bias,
                NULL, LARGE, 2.5f) == 0);
        double began = now_ms();
        for (int i = 0; i < 128; i++)
            CHECK(coli_v4_test_gpu_route_mirrors(weights, ids,
                fixture->input[i % INPUTS], fixture->large_gate, fixture->bias,
                NULL, LARGE, 2.5f) == 0);
        printf("router scratch reuse=%d calls=128 call_ms=%.6f\n",
               mode, (now_ms() - began) / 128.);
        CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == (mode ? LARGE : 0));
    }
}

int main(int argc, char **argv) {
    const int dimensions[] = {SMALL, LARGE, SMALL};
    int device = 0;
    CHECK(argc <= 2);
    CHECK(argc == 1 || !strcmp(argv[1], "0") || !strcmp(argv[1], "1"));
    int cycle = argc == 2 ? atoi(argv[1]) : 0;
    /* The runtime loader admits one backend lifetime per process. Run this
     * executable with 0 and 1 in separate processes to cover both fixtures. */
    {
        CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == 0);
        if (!dsv4_cuda_init(&device, 1)) return 77;
        Fixture fixture;
        fixture_init(&fixture, cycle);
        coli_v4_test_gpu_route_scratch_clear();
        coli_v4_test_gpu_route_scratch_clear();
        CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == 0);
        for (int shape = 0; shape < 3; shape++) {
            for (int bias = 0; bias < 2; bias++)
                for (int fixed = 0; fixed < 2; fixed++)
                    for (int sample = 0; sample < INPUTS; sample++)
                        compare_paths(&fixture, dimensions[shape], bias, fixed, sample);
            CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == (shape ? LARGE : SMALL));
            if (shape == 1) refused_then_valid(&fixture);
        }
        if (cycle == 0) timing(&fixture);
        fixture_free(&fixture);
        coli_v4_test_gpu_route_scratch_clear();
        CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == 0);
        coli_v4_test_gpu_route_scratch_clear();
        dsv4_cuda_shutdown();
        CHECK(coli_v4_test_gpu_route_scratch_capacity(0) == 0);
        printf("router scratch: bitwise parity, growth, refusal recovery, shutdown cycle=%d OK\n", cycle);
    }
    puts("CUDA router scratch: all checks OK");
    return 0;
}
