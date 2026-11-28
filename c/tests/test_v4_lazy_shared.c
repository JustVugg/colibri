/* test_v4_lazy_shared.c — V4_LAZY_SHARED opt-in deferral and fallback contract.
 *
 * Verifies:
 *   1. V4_LAZY_SHARED defaults to 0 (opt-in) and parses 0/1/empty cleanly.
 *   2. Deferral engages only when lazy_enabled, GPU tier is active, hybrid and
 *      hot_split are off, and all three GPU shared-expert tensors exist.
 *   3. When the fused GPU MoE path computes the shared expert, the host
 *      shared-expert forward is skipped (0 host calls).
 *   4. When falling back to GPU expert-group, peer/CUDA-lost CPU fallback, or
 *      the CPU MoE path, the host shared-expert forward runs on demand and
 *      produces bit-identical output to the eager baseline. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../compat.h"
#include "../deepseek_v4_lazy_shared.h"

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

enum { TEST_DIM = 16 };

typedef struct MockSharedCtx {
    const float *input;
    int fail_code;
} MockSharedCtx;

static int mock_host_shared_forward(void *opaque, float *shared_output) {
    const MockSharedCtx *ctx = (const MockSharedCtx *)opaque;
    if (ctx->fail_code) return ctx->fail_code;
    for (int i = 0; i < TEST_DIM; i++) {
        float x = ctx->input[i];
        /* Deterministic non-linear transform standing in for shared SwiGLU. */
        shared_output[i] = x * 0.375f + (float)(i + 1) * 0.0625f;
    }
    return 0;
}

typedef enum MoePath {
    PATH_GPU_FUSED_OK = 0,
    PATH_GPU_GROUP_FALLBACK = 1,
    PATH_GPU_LOST_CPU_FALLBACK = 2,
    PATH_CPU_DIRECT = 3,
} MoePath;

static int run_mock_moe(float *output, const float *input, const float *routed,
                        int lazy_enabled, MoePath path, int *out_host_calls) {
    float shared_output[TEST_DIM];
    float expert_output[TEST_DIM];
    /* Poison shared_output so reading it without running ensure is caught. */
    for (int i = 0; i < TEST_DIM; i++) shared_output[i] = -99999.0f;
    memset(output, 0, sizeof(float) * TEST_DIM);

    int deferred = coli_v4_lazy_shared_should_defer(
        lazy_enabled, /*has_gpu=*/1, /*hybrid=*/0, /*hot_split=*/0,
        /*w1=*/1, /*w3=*/1, /*w2=*/1);
    ColiV4LazyShared shared;
    coli_v4_lazy_shared_init(&shared, /*shared_here=*/1, deferred);
    MockSharedCtx ctx = { .input = input, .fail_code = 0 };
    int result = 0;

    if (!deferred)
        coli_v4_lazy_shared_ensure(&shared, &result, shared_output,
                                   mock_host_shared_forward, &ctx);

    if (path == PATH_GPU_FUSED_OK) {
        /* Fused GPU MoE computes routed + shared on device; host shared is
         * never touched. */
        float gpu_shared[TEST_DIM];
        mock_host_shared_forward(&ctx, gpu_shared);
        for (int i = 0; i < TEST_DIM; i++)
            output[i] = routed[i] + gpu_shared[i];
    } else if (path == PATH_GPU_GROUP_FALLBACK) {
        /* dsv4_cuda_moe returned 0 -> fallback to dsv4_cuda_expert_group +
         * host shared_output. */
        coli_v4_lazy_shared_ensure(&shared, &result, shared_output,
                                   mock_host_shared_forward, &ctx);
        for (int i = 0; i < TEST_DIM; i++)
            expert_output[i] = routed[i];
        if (!result)
            for (int i = 0; i < TEST_DIM; i++)
                output[i] = expert_output[i] + shared_output[i];
    } else if (path == PATH_GPU_LOST_CPU_FALLBACK) {
        /* dsv4_cuda_moe failed with peer/CUDA lost -> CPU routed + host
         * shared_output. */
        coli_v4_lazy_shared_ensure(&shared, &result, shared_output,
                                   mock_host_shared_forward, &ctx);
        for (int i = 0; i < TEST_DIM; i++)
            output[i] += routed[i];
        if (!result)
            for (int i = 0; i < TEST_DIM; i++)
                output[i] = output[i] + shared_output[i];
    } else {
        /* Direct CPU fallback path. */
        coli_v4_lazy_shared_ensure(&shared, &result, shared_output,
                                   mock_host_shared_forward, &ctx);
        for (int i = 0; i < TEST_DIM; i++)
            output[i] += routed[i];
        if (!result)
            for (int i = 0; i < TEST_DIM; i++)
                output[i] = output[i] + shared_output[i];
    }

    *out_host_calls = shared.host_calls;
    return result;
}

int main(void) {
    /* 1. Env var parsing: opt-in, default off. */
    unsetenv("V4_LAZY_SHARED");
    CHECK(coli_v4_lazy_shared_enabled() == 0, "unset V4_LAZY_SHARED must default to 0");
    setenv("V4_LAZY_SHARED", "", 1);
    CHECK(coli_v4_lazy_shared_enabled() == 0, "empty V4_LAZY_SHARED must be 0");
    setenv("V4_LAZY_SHARED", "0", 1);
    CHECK(coli_v4_lazy_shared_enabled() == 0, "V4_LAZY_SHARED=0 must be 0");
    setenv("V4_LAZY_SHARED", "1", 1);
    CHECK(coli_v4_lazy_shared_enabled() == 1, "V4_LAZY_SHARED=1 must be 1");
    unsetenv("V4_LAZY_SHARED");

    /* 2. Deferral gating conditions. */
    CHECK(coli_v4_lazy_shared_should_defer(0, 1, 0, 0, 1, 1, 1) == 0,
          "default off must not defer");
    CHECK(coli_v4_lazy_shared_should_defer(1, 1, 0, 0, 1, 1, 1) == 1,
          "all conditions satisfied must defer");
    CHECK(coli_v4_lazy_shared_should_defer(1, 0, 0, 0, 1, 1, 1) == 0,
          "no GPU store must not defer");
    CHECK(coli_v4_lazy_shared_should_defer(1, 1, 1, 0, 1, 1, 1) == 0,
          "hybrid mode must not defer");
    CHECK(coli_v4_lazy_shared_should_defer(1, 1, 0, 1, 1, 1, 1) == 0,
          "hot_split must not defer");
    CHECK(coli_v4_lazy_shared_should_defer(1, 1, 0, 0, 0, 1, 1) == 0,
          "missing GPU w1 must not defer");
    CHECK(coli_v4_lazy_shared_should_defer(1, 1, 0, 0, 1, 0, 1) == 0,
          "missing GPU w3 must not defer");
    CHECK(coli_v4_lazy_shared_should_defer(1, 1, 0, 0, 1, 1, 0) == 0,
          "missing GPU w2 must not defer");

    /* 3. End-to-end skip vs fallback bit-exactness. */
    float input[TEST_DIM], routed[TEST_DIM];
    for (int i = 0; i < TEST_DIM; i++) {
        input[i] = (float)(i - 7) * 0.25f;
        routed[i] = (float)(i * 3 - 10) * 0.125f;
    }

    float eager_out[TEST_DIM], lazy_out[TEST_DIM];
    int eager_calls = -1, lazy_calls = -1;

    /* Fused GPU path: lazy skips host shared forward and matches eager bits. */
    CHECK(run_mock_moe(eager_out, input, routed, 0, PATH_GPU_FUSED_OK, &eager_calls) == 0,
          "eager fused GPU path failed");
    CHECK(eager_calls == 1, "eager path must run host shared once, got %d", eager_calls);
    CHECK(run_mock_moe(lazy_out, input, routed, 1, PATH_GPU_FUSED_OK, &lazy_calls) == 0,
          "lazy fused GPU path failed");
    CHECK(lazy_calls == 0, "lazy fused GPU path must skip host shared, got %d", lazy_calls);
    CHECK(memcmp(eager_out, lazy_out, sizeof(eager_out)) == 0,
          "lazy fused GPU output must be bit-identical to eager");

    /* Fallback paths: lazy runs host shared on demand and matches eager bits. */
    const MoePath fallbacks[] = {
        PATH_GPU_GROUP_FALLBACK,
        PATH_GPU_LOST_CPU_FALLBACK,
        PATH_CPU_DIRECT,
    };
    for (size_t k = 0; k < sizeof(fallbacks) / sizeof(fallbacks[0]); k++) {
        CHECK(run_mock_moe(eager_out, input, routed, 0, fallbacks[k], &eager_calls) == 0,
              "eager fallback %zu failed", k);
        CHECK(run_mock_moe(lazy_out, input, routed, 1, fallbacks[k], &lazy_calls) == 0,
              "lazy fallback %zu failed", k);
        CHECK(lazy_calls == 1, "lazy fallback %zu must run host shared once, got %d",
              k, lazy_calls);
        CHECK(memcmp(eager_out, lazy_out, sizeof(eager_out)) == 0,
              "lazy fallback %zu output must be bit-identical to eager", k);
    }

    /* 4. Idempotency, shared_here=0, and error propagation. */
    {
        float shared_buf[TEST_DIM] = {0};
        ColiV4LazyShared s;
        MockSharedCtx ctx = { .input = input, .fail_code = 0 };
        int rc = 0;
        coli_v4_lazy_shared_init(&s, 1, 1);
        coli_v4_lazy_shared_ensure(&s, &rc, shared_buf, mock_host_shared_forward, &ctx);
        coli_v4_lazy_shared_ensure(&s, &rc, shared_buf, mock_host_shared_forward, &ctx);
        CHECK(rc == 0 && s.host_calls == 1,
              "second ensure must be a no-op (rc=%d calls=%d)", rc, s.host_calls);

        coli_v4_lazy_shared_init(&s, 0, 1);
        coli_v4_lazy_shared_ensure(&s, &rc, shared_buf, mock_host_shared_forward, &ctx);
        CHECK(rc == 0 && s.host_calls == 0,
              "shared_here=0 must not run host shared (calls=%d)", s.host_calls);

        MockSharedCtx fail_ctx = { .input = input, .fail_code = -7 };
        coli_v4_lazy_shared_init(&s, 1, 1);
        coli_v4_lazy_shared_ensure(&s, &rc, shared_buf, mock_host_shared_forward, &fail_ctx);
        CHECK(rc == -7 && s.host_calls == 1,
              "error must propagate (rc=%d calls=%d)", rc, s.host_calls);
        coli_v4_lazy_shared_ensure(&s, &rc, shared_buf, mock_host_shared_forward, &ctx);
        CHECK(rc == -7 && s.host_calls == 1,
              "subsequent ensure after error must not re-run");
    }

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("test_v4_lazy_shared: ok\n");
    return 0;
}
