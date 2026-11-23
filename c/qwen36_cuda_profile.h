#ifndef QWEN36_CUDA_PROFILE_H
#define QWEN36_CUDA_PROFILE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    Q36_GPU_NONE = 0,
    Q36_GPU_5090 = 1,
    Q36_GPU_PRO6000 = 2,
    Q36_GPU_4090 = 3,
    Q36_GPU_L40S = 4,
    Q36_GPU_L4,
    Q36_GPU_A100,
    Q36_GPU_3090,
    Q36_GPU_PRO4000,
};

enum {
    Q36_PROMPT_CACHE = 1u << 0,
    Q36_EXPERT_ROUTE = 1u << 1,
    Q36_FLASH_ATTN = 1u << 2,
    Q36_ASYNC_COPIES = 1u << 3,
    Q36_SHORT_MATMUL = 1u << 4,
    Q36_TILED_MATMUL = 1u << 5,
    Q36_ACTIVATION_CACHE = 1u << 6,
    Q36_SHARED_SWIGLU = 1u << 7,
    Q36_FUSED_NORM = 1u << 8,
    Q36_DNCONV = 1u << 9,
    Q36_DNREC = 1u << 10,
    Q36_EXPERT_MATMUL = 1u << 11,
    Q36_LONG_ATTN = 1u << 12,
};
#define Q36_ALL_FEATURES ((1u << 13) - 1u)
#define Q36_AVAILABLE_FEATURES ((1u << 13) - 1u)

/* Match the complete device name and compute capability for each profile. */
static inline int q36_cuda_device_profile(int major, int minor, const char *name,
                                          size_t total_global_mem) {
    static const struct {
        int profile, major, minor;
        const char *name, *alias;
        uint64_t min_memory, max_memory;
    } devices[] = {
        {Q36_GPU_L4, 8, 9, "NVIDIA L4", "L4", 0, 0},
    };

    if (!name) return Q36_GPU_NONE;
    for (size_t i = 0; i < sizeof(devices) / sizeof(devices[0]); ++i) {
        if (devices[i].major != major || devices[i].minor != minor) continue;
        if (strcmp(devices[i].name, name) != 0 &&
            (!devices[i].alias || strcmp(devices[i].alias, name) != 0))
            continue;
        if (devices[i].max_memory &&
            (total_global_mem < devices[i].min_memory || total_global_mem >= devices[i].max_memory))
            continue;
        return devices[i].profile;
    }
    return Q36_GPU_NONE;
}

static inline int q36_has_request_rules(int profile) {
    (void)profile;
    return 0;
}

/* Capacity and initialization use the union of the permitted requests and phases. */
static inline unsigned q36_device_features(int profile) {
    switch (profile) {
    case Q36_GPU_L4:
        return Q36_AVAILABLE_FEATURES;
    default:
        return 0;
    }
}

static inline int q36_3090_uses_exact_cache(int reused, int fresh) {
    (void)reused;
    (void)fresh;
    return 0;
}

static inline unsigned q36_request_features(int profile, int reused, int fresh, int cold,
                                             int generation, int cache3090) {
    (void)cold;
    (void)generation;
    (void)reused;
    (void)fresh;
    (void)cache3090;
    unsigned mask = q36_device_features(profile);
    return mask;
}

static inline unsigned q36_default_features(int profile) {
    return q36_request_features(profile, 0, 0, 0, 0, 0);
}

#endif
