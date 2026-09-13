#include "../glm53_gpu.h"
#include "../backend_cuda.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    int create_calls;
    int probe_calls;
    int destroy_calls;
    int last_device;
    int create_ok;
    uint64_t caps;
    ColiGpuContext *ctx;
} FakeGpu;

static FakeGpu *g_fake_gpu;

static int fake_create(ColiGpuContext **out, int device) {
    FakeGpu *fake = g_fake_gpu;
    fake->create_calls++;
    fake->last_device = device;
    if (!fake->create_ok) return 0;
    *out = fake->ctx;
    return 1;
}

static int fake_probe(ColiGpuContext *ctx, uint64_t required_caps) {
    FakeGpu *fake = (FakeGpu *)ctx;
    fake->probe_calls++;
    return (required_caps & ~fake->caps) == 0;
}

static void fake_destroy(ColiGpuContext *ctx) {
    FakeGpu *fake = (FakeGpu *)ctx;
    fake->destroy_calls++;
}

static ColiGlm53GpuOps fake_ops(void) {
    ColiGlm53GpuOps ops;
    ops.context_create = fake_create;
    ops.context_probe = fake_probe;
    ops.context_destroy = fake_destroy;
    return ops;
}

static int expect_mode(const char *text, ColiGlm53GpuMode want) {
    ColiGlm53GpuMode got = COLI_GLM53_GPU_MODE_CPU;
    if (!coli_glm53_gpu_mode_parse(text, &got)) {
        fprintf(stderr, "parse failed for %s\n", text ? text : "(null)");
        return 0;
    }
    if (got != want) {
        fprintf(stderr, "wrong mode for %s: got %d want %d\n",
                text ? text : "(null)", got, want);
        return 0;
    }
    return 1;
}

static int test_mode_parsing(void) {
    ColiGlm53GpuMode got = COLI_GLM53_GPU_MODE_AUTO;
    return expect_mode(NULL, COLI_GLM53_GPU_MODE_AUTO) &&
           expect_mode("", COLI_GLM53_GPU_MODE_AUTO) &&
           expect_mode("auto", COLI_GLM53_GPU_MODE_AUTO) &&
           expect_mode("gpu", COLI_GLM53_GPU_MODE_GPU) &&
           expect_mode("cpu", COLI_GLM53_GPU_MODE_CPU) &&
           !coli_glm53_gpu_mode_parse("cuda", &got);
}

static int test_cpu_mode_does_not_initialize_gpu(void) {
    FakeGpu fake = {0};
    ColiGlm53GpuBackend backend;
    char err[128] = {0};
    g_fake_gpu = &fake;
    fake.ctx = (ColiGpuContext *)&fake;
    fake.create_ok = 1;
    fake.caps = COLI_GLM53_GPU_REQUIRED_CAPS;
    coli_glm53_gpu_backend_init(&backend);
    if (!coli_glm53_gpu_backend_select(&backend, COLI_GLM53_GPU_MODE_CPU,
                                       fake_ops(), (int)(long)&fake,
                                       COLI_GLM53_GPU_REQUIRED_CAPS,
                                       err, sizeof(err))) return 0;
    if (fake.create_calls || fake.probe_calls || fake.destroy_calls) {
        fprintf(stderr, "CPU mode touched GPU: create=%d probe=%d destroy=%d\n",
                fake.create_calls, fake.probe_calls, fake.destroy_calls);
        return 0;
    }
    return coli_glm53_gpu_backend_selected(&backend) == COLI_GLM53_BACKEND_CPU;
}

static int test_auto_falls_back_when_pipeline_is_not_advertised(void) {
    FakeGpu fake = {0};
    ColiGlm53GpuBackend backend;
    char err[128] = {0};
    g_fake_gpu = &fake;
    fake.ctx = (ColiGpuContext *)&fake;
    fake.create_ok = 1;
    fake.caps = COLI_GPU_CAP_STREAM_ORDERED | COLI_GPU_CAP_INT4_GS64;
    coli_glm53_gpu_backend_init(&backend);
    if (!coli_glm53_gpu_backend_select(&backend, COLI_GLM53_GPU_MODE_AUTO,
                                       fake_ops(), (int)(long)&fake,
                                       COLI_GLM53_GPU_REQUIRED_CAPS,
                                       err, sizeof(err))) return 0;
    if (fake.create_calls != 1 || fake.probe_calls != 1 || fake.destroy_calls != 1) {
        fprintf(stderr, "auto probe failure calls: create=%d probe=%d destroy=%d\n",
                fake.create_calls, fake.probe_calls, fake.destroy_calls);
        return 0;
    }
    return coli_glm53_gpu_backend_selected(&backend) == COLI_GLM53_BACKEND_CPU;
}

static int test_gpu_mode_fails_when_pipeline_is_not_advertised(void) {
    FakeGpu fake = {0};
    ColiGlm53GpuBackend backend;
    char err[128] = {0};
    g_fake_gpu = &fake;
    fake.ctx = (ColiGpuContext *)&fake;
    fake.create_ok = 1;
    fake.caps = COLI_GPU_CAP_STREAM_ORDERED | COLI_GPU_CAP_INT4_GS64;
    coli_glm53_gpu_backend_init(&backend);
    if (coli_glm53_gpu_backend_select(&backend, COLI_GLM53_GPU_MODE_GPU,
                                      fake_ops(), (int)(long)&fake,
                                      COLI_GLM53_GPU_REQUIRED_CAPS,
                                      err, sizeof(err))) {
        fprintf(stderr, "gpu mode accepted failed probe\n");
        return 0;
    }
    if (!strstr(err, "probe")) {
        fprintf(stderr, "gpu failure did not mention probe: %s\n", err);
        return 0;
    }
    return coli_glm53_gpu_backend_selected(&backend) == COLI_GLM53_BACKEND_UNSELECTED;
}

static int test_successful_gpu_selection_and_destroy(void) {
    FakeGpu fake = {0};
    ColiGlm53GpuBackend backend;
    char err[128] = {0};
    g_fake_gpu = &fake;
    fake.ctx = (ColiGpuContext *)&fake;
    fake.create_ok = 1;
    fake.caps = COLI_GLM53_GPU_REQUIRED_CAPS;
    coli_glm53_gpu_backend_init(&backend);
    if (!coli_glm53_gpu_backend_select(&backend, COLI_GLM53_GPU_MODE_GPU,
                                       fake_ops(), (int)(long)&fake,
                                       COLI_GLM53_GPU_REQUIRED_CAPS,
                                       err, sizeof(err))) return 0;
    if (coli_glm53_gpu_backend_selected(&backend) != COLI_GLM53_BACKEND_GPU) return 0;
    if (coli_glm53_gpu_backend_context(&backend) != fake.ctx) return 0;
    coli_glm53_gpu_backend_destroy(&backend, fake_ops());
    return fake.destroy_calls == 1 &&
           coli_glm53_gpu_backend_selected(&backend) == COLI_GLM53_BACKEND_UNSELECTED;
}

static int test_backend_selection_locked_during_request(void) {
    FakeGpu fake = {0};
    ColiGlm53GpuBackend backend;
    char err[128] = {0};
    g_fake_gpu = &fake;
    fake.ctx = (ColiGpuContext *)&fake;
    fake.create_ok = 1;
    fake.caps = COLI_GLM53_GPU_REQUIRED_CAPS;
    coli_glm53_gpu_backend_init(&backend);
    if (!coli_glm53_gpu_backend_select(&backend, COLI_GLM53_GPU_MODE_GPU,
                                       fake_ops(), (int)(long)&fake,
                                       COLI_GLM53_GPU_REQUIRED_CAPS,
                                       err, sizeof(err))) return 0;
    if (!coli_glm53_gpu_backend_begin_request(&backend)) return 0;
    if (coli_glm53_gpu_backend_select(&backend, COLI_GLM53_GPU_MODE_CPU,
                                      fake_ops(), (int)(long)&fake,
                                      COLI_GLM53_GPU_REQUIRED_CAPS,
                                      err, sizeof(err))) {
        fprintf(stderr, "backend selection changed during an active request\n");
        return 0;
    }
    coli_glm53_gpu_backend_end_request(&backend);
    coli_glm53_gpu_backend_destroy(&backend, fake_ops());
    return coli_glm53_gpu_backend_selected(&backend) == COLI_GLM53_BACKEND_UNSELECTED;
}

static int test_runtime_failure_switches_future_requests_to_cpu(void) {
    FakeGpu fake = {0};
    ColiGlm53GpuBackend backend;
    char err[128] = {0};
    g_fake_gpu = &fake;
    fake.ctx = (ColiGpuContext *)&fake;
    fake.create_ok = 1;
    fake.caps = COLI_GLM53_GPU_REQUIRED_CAPS;
    coli_glm53_gpu_backend_init(&backend);
    if (!coli_glm53_gpu_backend_select(
            &backend, COLI_GLM53_GPU_MODE_GPU, fake_ops(), 0,
            COLI_GLM53_GPU_REQUIRED_CAPS, err, sizeof(err)) ||
        !coli_glm53_gpu_backend_begin_request(&backend) ||
        !coli_glm53_gpu_backend_fail_request(&backend) ||
        coli_glm53_gpu_backend_selected(&backend) !=
            COLI_GLM53_BACKEND_CPU)
        return 0;
    coli_glm53_gpu_backend_end_request(&backend);
    if (!coli_glm53_gpu_backend_begin_request(&backend) ||
        coli_glm53_gpu_backend_selected(&backend) !=
            COLI_GLM53_BACKEND_CPU)
        return 0;
    coli_glm53_gpu_backend_end_request(&backend);
    coli_glm53_gpu_backend_destroy(&backend, fake_ops());
    return fake.destroy_calls == 1;
}

int main(void) {
    if (!test_mode_parsing()) return 1;
    if (!test_cpu_mode_does_not_initialize_gpu()) return 1;
    if (!test_auto_falls_back_when_pipeline_is_not_advertised()) return 1;
    if (!test_gpu_mode_fails_when_pipeline_is_not_advertised()) return 1;
    if (!test_successful_gpu_selection_and_destroy()) return 1;
    if (!test_backend_selection_locked_during_request()) return 1;
    if (!test_runtime_failure_switches_future_requests_to_cpu()) return 1;
    puts("glm53 gpu mode selection: ok");
    return 0;
}
