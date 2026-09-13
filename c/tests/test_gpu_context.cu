#include "../backend_cuda.h"

#include <cstdio>
#include <cstdlib>

static int test_create_probe_sync_destroy(int device) {
    ColiGpuContext *ctx = nullptr;
    const uint64_t caps = COLI_GPU_CAP_STREAM_ORDERED |
                          COLI_GPU_CAP_INT4_GS64;
    if (!coli_gpu_context_create(&ctx, device) || !ctx) {
        std::fprintf(stderr, "context create failed on device %d\n", device);
        return 0;
    }
    if (!coli_gpu_context_healthy(ctx)) {
        std::fprintf(stderr, "new context is unhealthy\n");
        return 0;
    }
    if (!coli_gpu_context_probe(ctx, caps)) {
        std::fprintf(stderr, "baseline capability probe failed\n");
        return 0;
    }
    if (!coli_gpu_context_sync(ctx)) {
        std::fprintf(stderr, "stream sync failed\n");
        return 0;
    }
    coli_gpu_context_destroy(ctx);
    return 1;
}

static int test_pipeline_capability_is_not_advertised_until_full_forward(int device) {
    ColiGpuContext *ctx = nullptr;
    if (!coli_gpu_context_create(&ctx, device)) return 0;
    int ok = !coli_gpu_context_probe(ctx, COLI_GPU_CAP_PIPELINE) &&
             !coli_gpu_context_probe(ctx, COLI_GPU_CAP_STREAM_ORDERED |
                                          COLI_GPU_CAP_INT4_GS64 |
                                          COLI_GPU_CAP_PIPELINE);
    coli_gpu_context_destroy(ctx);
    if (!ok) std::fprintf(stderr, "pipeline capability advertised before Task 6\n");
    return ok;
}

static int test_invalid_device_is_rejected(void) {
    ColiGpuContext *ctx = nullptr;
    if (coli_gpu_context_create(&ctx, -1) || ctx) {
        std::fprintf(stderr, "negative device accepted\n");
        return 0;
    }
    if (coli_gpu_context_create(&ctx, 99999) || ctx) {
        std::fprintf(stderr, "out-of-range device accepted\n");
        coli_gpu_context_destroy(ctx);
        return 0;
    }
    return 1;
}

static int test_unsupported_capability_is_rejected(int device) {
    ColiGpuContext *ctx = nullptr;
    if (!coli_gpu_context_create(&ctx, device)) return 0;
    const uint64_t unknown_cap = 1ull << 63;
    int ok = !coli_gpu_context_probe(ctx, COLI_GPU_CAP_STREAM_ORDERED | unknown_cap);
    coli_gpu_context_destroy(ctx);
    if (!ok) std::fprintf(stderr, "unknown capability accepted\n");
    return ok;
}

static int test_unhealthy_state_is_sticky(int device) {
    ColiGpuContext *ctx = nullptr;
    if (!coli_gpu_context_create(&ctx, device)) return 0;
    coli_gpu_context_mark_unhealthy(ctx);
    int ok = !coli_gpu_context_healthy(ctx) &&
             !coli_gpu_context_probe(ctx, COLI_GPU_CAP_STREAM_ORDERED) &&
             !coli_gpu_context_sync(ctx);
    coli_gpu_context_destroy(ctx);
    if (!ok) std::fprintf(stderr, "unhealthy context remained usable\n");
    return ok;
}

int main(int argc, char **argv) {
    int device = argc > 1 ? std::atoi(argv[1]) : 0;
    if (!test_invalid_device_is_rejected()) return 1;
    if (!test_create_probe_sync_destroy(device)) return 1;
    if (!test_pipeline_capability_is_not_advertised_until_full_forward(device)) return 1;
    if (!test_unsupported_capability_is_rejected(device)) return 1;
    if (!test_unhealthy_state_is_sticky(device)) return 1;
    std::printf("gpu context: ok on device %d\n", device);
    return 0;
}
