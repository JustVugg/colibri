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
        {Q36_GPU_A100, 8, 0, "NVIDIA A100-SXM4-40GB", NULL, UINT64_C(38) << 30, UINT64_C(42) << 30},
        {Q36_GPU_3090, 8, 6, "NVIDIA GeForce RTX 3090", NULL, 0, 0},
        {Q36_GPU_PRO4000, 12, 0, "NVIDIA RTX PRO 4000 Blackwell", NULL, 0, 0},
        {Q36_GPU_4090, 8, 9, "NVIDIA GeForce RTX 4090", NULL, 0, 0},
        {Q36_GPU_5090, 12, 0, "NVIDIA GeForce RTX 5090", NULL, 0, 0},
        {Q36_GPU_PRO6000, 12, 0, "NVIDIA RTX PRO 6000 Blackwell Server Edition", NULL, 0, 0},
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
    switch (profile) {
    case Q36_GPU_4090:
    case Q36_GPU_5090:
    case Q36_GPU_PRO6000:
        return 1;
    default:
        return 0;
    }
}

#define Q36_ROW(n) (UINT32_C(1) << (n))
#define Q36_ROWS(first, last) (Q36_ROW((last) + 1) - Q36_ROW(first))
#define Q36_ANY_ROW Q36_ROWS(0, 16)

/* Use the full logical request in both phases, before splitting into forwards.
 * Cold means the chain has not run a forward when the request begins.
 * Row 0 covers requests outside the measured matrix. */
static inline uint32_t q36_request_row(int reused, int fresh, int cold) {
    enum { ANY_TEMPERATURE = -1 };
    static const struct {
        int reused, fresh;
        int cold;
    } shapes[] = {
        {   0,    64, 1 },
        {   0,   512, 0 },
        {   0,    64, 0 },
        {   0,   512, 1 },
        {   0,  2048, ANY_TEMPERATURE },
        {   0,  4096, ANY_TEMPERATURE },
        {   0,  8192, ANY_TEMPERATURE },
        {   0, 16384, ANY_TEMPERATURE },
        {   0,   663, ANY_TEMPERATURE },
        { 720,  1234, ANY_TEMPERATURE },
        {1986,  2262, ANY_TEMPERATURE },
        {4335,  1406, ANY_TEMPERATURE },
        {5785,  1298, ANY_TEMPERATURE },
        {7137,   904, ANY_TEMPERATURE },
        {8065,   348, ANY_TEMPERATURE },
        {8880,    35, ANY_TEMPERATURE },
    };

    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        if (shapes[i].reused == reused && shapes[i].fresh == fresh &&
            (shapes[i].cold == ANY_TEMPERATURE || shapes[i].cold == (cold != 0)))
            return Q36_ROW(i + 1);
    }
    return Q36_ROW(0);
}

static inline unsigned q36_row_features(int profile, uint32_t rows, int generation) {
    struct PhaseRules { uint32_t read, generate; };
    /* General permissions include row 0. Each allowlist applies to measured rows. */
    enum {
        RTX4090_TILED_MATMUL_ROWS = Q36_ANY_ROW &
            ~(Q36_ROW(2) | Q36_ROW(4) | Q36_ROW(9) | Q36_ROWS(15, 16)),
    };
    static const struct PhaseRules rtx4090[13] = {
        { Q36_ANY_ROW & ~Q36_ROWS(14, 15), Q36_ANY_ROW }, /* prompt cache */
        { 0, Q36_ANY_ROW }, /* expert route */
        { Q36_ANY_ROW, Q36_ANY_ROW }, /* flash attn */
        { Q36_ANY_ROW, 0 }, /* async copies */
        { 0, Q36_ANY_ROW }, /* short matmul */
        { RTX4090_TILED_MATMUL_ROWS, RTX4090_TILED_MATMUL_ROWS }, /* tiled matmul */
        { Q36_ANY_ROW, Q36_ANY_ROW }, /* activation cache */
        { Q36_ANY_ROW, Q36_ANY_ROW }, /* shared swiglu */
        { Q36_ANY_ROW, 0 }, /* fused norm */
        { 0, 0 }, /* dnconv */
        { Q36_ROW(1), Q36_ANY_ROW & ~Q36_ROW(14) }, /* dnrec */
        { Q36_ANY_ROW, Q36_ANY_ROW }, /* expert matmul */
        { Q36_ROW(16), 0 }, /* long attn */
    };
    static const struct PhaseRules rtx5090[13] = {
        { Q36_ANY_ROW & ~Q36_ROWS(13, 16),
          Q36_ANY_ROW & ~Q36_ROW(16) }, /* prompt cache */
        { 0, 0 }, /* expert route */
        { Q36_ANY_ROW, Q36_ROWS(9, 16) }, /* flash attn */
        { Q36_ANY_ROW,
          Q36_ANY_ROW & ~(Q36_ROW(10) | Q36_ROW(12) | Q36_ROW(14)) }, /* async copies */
        { 0, 0 }, /* short matmul */
        { Q36_ROWS(5, 14), Q36_ROWS(5, 14) & ~Q36_ROW(12) }, /* tiled matmul */
        { Q36_ANY_ROW, 0 }, /* activation cache */
        { Q36_ANY_ROW, Q36_ANY_ROW }, /* shared swiglu */
        { 0, 0 }, /* fused norm */
        { 0, 0 }, /* dnconv */
        { Q36_ROW(1), Q36_ANY_ROW }, /* dnrec */
        { Q36_ANY_ROW, 0 }, /* expert matmul */
        { Q36_ROW(16), 0 }, /* long attn */
    };
    static const struct PhaseRules pro6000[13] = {
        { Q36_ANY_ROW & ~Q36_ROWS(14, 15),
          Q36_ANY_ROW & ~Q36_ROWS(14, 16) }, /* prompt cache */
        { 0, Q36_ROW(14) }, /* expert route */
        { Q36_ANY_ROW, Q36_ANY_ROW }, /* flash attn */
        { Q36_ANY_ROW & ~Q36_ROW(1), 0 }, /* async copies */
        { 0, Q36_ROW(10) }, /* short matmul */
        { Q36_ROWS(5, 8) | Q36_ROWS(10, 13), 0 }, /* tiled matmul */
        { 0, 0 }, /* activation cache */
        { Q36_ANY_ROW, Q36_ANY_ROW }, /* shared swiglu */
        { Q36_ANY_ROW, 0 }, /* fused norm */
        { 0, 0 }, /* dnconv */
        { 0, 0 }, /* dnrec */
        { Q36_ANY_ROW, 0 }, /* expert matmul */
        { Q36_ROW(16), Q36_ROW(16) }, /* long attn */
    };

    const struct PhaseRules *rules;
    switch (profile) {
    case Q36_GPU_4090:
        rules = rtx4090;
        break;
    case Q36_GPU_5090:
        rules = rtx5090;
        break;
    case Q36_GPU_PRO6000:
        rules = pro6000;
        break;
    default:
        return 0;
    }
    unsigned permissions = 0;
    for (unsigned i = 0; i < 13; i++) {
        uint32_t allowed = generation ? rules[i].generate : rules[i].read;
        if (allowed & rows) permissions |= 1u << i;
    }
    return permissions & Q36_AVAILABLE_FEATURES;
}

/* Capacity and initialization use the union of the permitted requests and phases. */
static inline unsigned q36_device_features(int profile) {
    switch (profile) {
    case Q36_GPU_L4:
        return Q36_AVAILABLE_FEATURES;
    case Q36_GPU_A100:
    case Q36_GPU_3090:
        return Q36_AVAILABLE_FEATURES & ~(Q36_EXPERT_ROUTE | Q36_SHARED_SWIGLU |
                                       Q36_FUSED_NORM | Q36_LONG_ATTN);
    case Q36_GPU_PRO4000:
        return Q36_AVAILABLE_FEATURES & ~(Q36_ACTIVATION_CACHE | Q36_DNCONV |
                                       Q36_DNREC | Q36_LONG_ATTN);
    case Q36_GPU_4090:
    case Q36_GPU_5090:
    case Q36_GPU_PRO6000:
        return q36_row_features(profile, Q36_ANY_ROW, 0) |
               q36_row_features(profile, Q36_ANY_ROW, 1);
    default:
        return 0;
    }
}

static inline int q36_3090_uses_exact_cache(int reused, int fresh) {
    return (reused == 5785 && fresh == 1298) || (reused == 7137 && fresh == 904) ||
           (reused == 8065 && fresh == 348);
}

static inline unsigned q36_request_features(int profile, int reused, int fresh, int cold,
                                             int generation, int cache3090) {
    if (q36_has_request_rules(profile))
        return q36_row_features(profile, q36_request_row(reused, fresh, cold), generation);
    unsigned mask = q36_device_features(profile);
    if (profile == Q36_GPU_A100) {
        if (generation) mask &= ~(Q36_FLASH_ATTN | Q36_ASYNC_COPIES);
        else if (!(cold && reused == 0 && fresh == 64)) mask &= ~Q36_DNREC;
    }
    if (profile == Q36_GPU_3090 && cache3090 && q36_3090_uses_exact_cache(reused, fresh))
        mask &= ~Q36_PROMPT_CACHE;
    return mask;
}

static inline unsigned q36_default_features(int profile) {
    return q36_request_features(profile, 0, 0, 0, 0, 0);
}

#undef Q36_ANY_ROW
#undef Q36_ROWS
#undef Q36_ROW

#endif
