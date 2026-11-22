#include "../qwen36_cuda_profile.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;

enum { available = (1u << 9) - 1u };
#define GIB(n) ((size_t)(n) << 30)

static void check_value(unsigned actual, unsigned expected, const char *what) {
    checks++;
    if (actual != expected) {
        failures++;
        fprintf(stderr, "%s: expected %#x, got %#x\n", what, expected, actual);
    }
}

static void check_device(int major, int minor, const char *name, size_t bytes, int expected) {
    check_value(q36_cuda_device_profile(major, minor, name, bytes), expected, name);
    check_value(q36_cuda_device_profile(major, minor + 1, name, bytes), Q36_GPU_NONE,
                "correct name with incompatible device architecture");
    char derived[128];
    snprintf(derived, sizeof derived, "%s (other SKU)", name);
    check_value(q36_cuda_device_profile(major, minor, derived, bytes), Q36_GPU_NONE,
                "compatible architecture with only a matching name prefix");
}

static void check_mask(int profile, int reused, int fresh, int cold, int generation,
                       int cache3090, unsigned expected) {
    char context[144];
    snprintf(context, sizeof context, "profile %d, reused %d, fresh %d, cold %d, decode %d, live cache %d",
             profile, reused, fresh, cold, generation, cache3090);
    check_value(q36_request_features(profile, reused, fresh, cold, generation, cache3090),
                expected, context);
}


int main(void) {
    check_device(8, 9, "NVIDIA L4", GIB(22), Q36_GPU_L4);
    check_device(8, 9, "L4", GIB(22), Q36_GPU_L4);
    check_device(7, 5, "Tesla T4", GIB(15), Q36_GPU_NONE);
    check_device(8, 6, "NVIDIA GeForce RTX 3080", GIB(10), Q36_GPU_NONE);
    check_device(8, 0, "NVIDIA A100-SXM4-80GB", GIB(80), Q36_GPU_NONE);
    check_value(q36_cuda_device_profile(8, 9, NULL, 0), Q36_GPU_NONE, "missing device name");
    check_value(q36_device_features(Q36_GPU_NONE), 0, "ordinary mode");
    check_value(q36_device_features(-1), 0, "unknown mode");
    check_value(q36_device_features(Q36_GPU_L4), available, "L4 available features");
    for (int generation = 0; generation <= 1; generation++) {
        check_mask(Q36_GPU_NONE, 0, 64, 1, generation, 0, 0);
        check_mask(Q36_GPU_L4, 0, 64, 1, generation, 0, available);
        check_mask(Q36_GPU_L4, 5785, 1298, 0, generation, 1, available);
    }
    printf("Qwen3.6 CUDA profiles: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
