#include "../qwen36_cuda_profile.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;

enum { available = (1u << 13) - 1u };
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

typedef struct { const char *read, *generate; } ExpectedPermissions;

/* The leftmost character corresponds to feature bit 0. */
static unsigned permission_bits(const char *flags) {
    unsigned mask = 0;
    for (int i = 0; flags[i]; i++)
        if (flags[i] == '1') mask |= 1u << i;
    return mask;
}

static void check_matrix(void) {
    /* Full policy oracle in qualified request order. */
    static const struct { int reused, fresh, cold; } measures[] = {
        {   0,    64, 1 },
        {   0,   512, 0 },
        {   0,    64, 0 },
        {   0,   512, 1 },
        {   0,  2048, 1 },
        {   0,  4096, 0 },
        {   0,  8192, 0 },
        {   0, 16384, 0 },
        {   0,   663, 1 },
        { 720,  1234, 0 },
        {1986,  2262, 0 },
        {4335,  1406, 0 },
        {5785,  1298, 0 },
        {7137,   904, 0 },
        {8065,   348, 0 },
        {8880,    35, 0 },
    };
    enum { MEASURE_COUNT = sizeof(measures) / sizeof(measures[0]) };

    static const struct {
        int profile;
        ExpectedPermissions defaults;
        ExpectedPermissions matrix[MEASURE_COUNT];
    } cases[] = {
        { Q36_GPU_4090, { "1011011110010", "1110111100110" }, {
            { "1011011110110", "1110111100110" },
            { "1011001110010", "1110101100110" },
            { "1011011110010", "1110111100110" },
            { "1011001110010", "1110101100110" },
            { "1011011110010", "1110111100110" },
            { "1011011110010", "1110111100110" },
            { "1011011110010", "1110111100110" },
            { "1011011110010", "1110111100110" },
            { "1011001110010", "1110101100110" },
            { "1011011110010", "1110111100110" },
            { "1011011110010", "1110111100110" },
            { "1011011110010", "1110111100110" },
            { "1011011110010", "1110111100110" },
            { "0011011110010", "1110111100010" },
            { "0011001110010", "1110101100110" },
            { "1011001110011", "1110101100110" },
        } },
        { Q36_GPU_5090, { "1011001100010", "1001000100100" }, {
            { "1011001100110", "1001000100100" },
            { "1011001100010", "1001000100100" },
            { "1011001100010", "1001000100100" },
            { "1011001100010", "1001000100100" },
            { "1011011100010", "1001010100100" },
            { "1011011100010", "1001010100100" },
            { "1011011100010", "1001010100100" },
            { "1011011100010", "1001010100100" },
            { "1011011100010", "1011010100100" },
            { "1011011100010", "1010010100100" },
            { "1011011100010", "1011010100100" },
            { "1011011100010", "1010000100100" },
            { "0011011100010", "1011010100100" },
            { "0011011100010", "1010010100100" },
            { "0011001100010", "1011000100100" },
            { "0011001100011", "0011000100100" },
        } },
        { Q36_GPU_PRO6000, { "1011000110010", "1010000100000" }, {
            { "1010000110010", "1010000100000" },
            { "1011000110010", "1010000100000" },
            { "1011000110010", "1010000100000" },
            { "1011000110010", "1010000100000" },
            { "1011010110010", "1010000100000" },
            { "1011010110010", "1010000100000" },
            { "1011010110010", "1010000100000" },
            { "1011010110010", "1010000100000" },
            { "1011000110010", "1010000100000" },
            { "1011010110010", "1010100100000" },
            { "1011010110010", "1010000100000" },
            { "1011010110010", "1010000100000" },
            { "1011010110010", "1010000100000" },
            { "0011000110010", "0110000100000" },
            { "0011000110010", "0010000100000" },
            { "1011000110011", "0010000100001" },
        } },
        { Q36_GPU_L40S, { "1101000000010", "1101100000010" }, {
            { "1101000000110", "1101100000010" },
            { "1101001000010", "1101100000010" },
            { "1101000000010", "1101100000010" },
            { "1101001000010", "1101100100010" },
            { "1101011000010", "1101100000010" },
            { "1101011000010", "1101100000010" },
            { "1101011000010", "1101100000010" },
            { "1111011000010", "1101100100010" },
            { "1101101001010", "1101100000010" },
            { "1101011000010", "1100100000010" },
            { "1101001000010", "1101100000010" },
            { "1101011000010", "1001100000010" },
            { "0111011000010", "1101100100010" },
            { "0101011000010", "1101100000010" },
            { "0101001000010", "1101100100010" },
            { "1111000000010", "1101100000010" },
        } },
    };

    for (size_t gpu = 0; gpu < sizeof cases / sizeof *cases; gpu++) {
        int profile = cases[gpu].profile;
        ExpectedPermissions defaults = cases[gpu].defaults;
        unsigned possible = permission_bits(defaults.read) | permission_bits(defaults.generate);
        for (size_t row = 0; row < MEASURE_COUNT; row++) {
            int reused = measures[row].reused, fresh = measures[row].fresh, cold = measures[row].cold;
            ExpectedPermissions expected = cases[gpu].matrix[row];
            possible |= permission_bits(expected.read) | permission_bits(expected.generate);
            for (int generation = 0; generation <= 1; generation++) {
                unsigned row_mask = permission_bits(generation ? expected.generate : expected.read);
                unsigned fallback = permission_bits(generation ? defaults.generate : defaults.read);
                check_mask(profile, reused, fresh, cold, generation, 0, row_mask);
                if (fresh != 64 && fresh != 512)
                    check_mask(profile, reused, fresh, !cold, generation, 0, row_mask);
                check_mask(profile, reused, fresh - 1, cold, generation, 0, fallback);
                check_mask(profile, reused, fresh + 1, cold, generation, 0, fallback);
                check_mask(profile, reused + 1, fresh, cold, generation, 0, fallback);
                if (reused)
                    check_mask(profile, reused - 1, fresh + 1, cold, generation, 0, fallback);
            }
        }
        check_value(q36_device_features(profile), possible, "union of matrix permissions");
    }
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
    check_device(8, 0, "NVIDIA A100-SXM4-40GB", GIB(39), Q36_GPU_A100);
    check_value(q36_cuda_device_profile(8, 0, "NVIDIA A100-SXM4-40GB", GIB(80)),
                Q36_GPU_NONE, "40GB profile rejects an oversized device");
    unsigned a100 = Q36_ALL_FEATURES & ~4482u;
    check_value(q36_device_features(Q36_GPU_A100), a100, "A100 available features");
    check_mask(Q36_GPU_A100, 0, 64, 1, 0, 0, a100);
    check_mask(Q36_GPU_A100, 0, 64, 0, 0, 0, a100 & ~Q36_DNREC);
    check_mask(Q36_GPU_A100, 1, 64, 1, 0, 0, a100 & ~Q36_DNREC);
    check_mask(Q36_GPU_A100, 0, 63, 1, 0, 0, a100 & ~Q36_DNREC);
    check_mask(Q36_GPU_A100, 0, 64, 1, 1, 0, a100 & ~(Q36_FLASH_ATTN | Q36_ASYNC_COPIES));
    check_device(8, 6, "NVIDIA GeForce RTX 3090", GIB(24), Q36_GPU_3090);
    check_device(8, 6, "NVIDIA GeForce RTX 3090 Ti", GIB(24), Q36_GPU_NONE);
    unsigned rtx3090 = Q36_ALL_FEATURES & ~4482u;
    check_value(q36_device_features(Q36_GPU_3090), rtx3090, "RTX 3090 available features");
    const int live[][2] = {{5785, 1298}, {7137, 904}, {8065, 348}};
    for (size_t i = 0; i < sizeof live / sizeof *live; i++) {
        int reused = live[i][0], fresh = live[i][1];
        for (int generation = 0; generation <= 1; generation++) {
            check_mask(Q36_GPU_3090, reused, fresh, 0, generation, 1, rtx3090 & ~Q36_PROMPT_CACHE);
            check_mask(Q36_GPU_3090, reused, fresh, 0, generation, 0, rtx3090);
            check_mask(Q36_GPU_3090, reused, fresh + 1, 0, generation, 1, rtx3090);
            check_mask(Q36_GPU_3090, reused - 1, fresh + 1, 0, generation, 1, rtx3090);
        }
    }
    check_device(12, 0, "NVIDIA RTX PRO 4000 Blackwell", GIB(24), Q36_GPU_PRO4000);
    check_device(12, 0, "NVIDIA RTX PRO 4000 Blackwell SFF Edition", GIB(24), Q36_GPU_PRO4000);
    unsigned pro4000 = Q36_ALL_FEATURES & ~5696u;
    check_value(q36_device_features(Q36_GPU_PRO4000), pro4000, "RTX PRO 4000 available features");
    check_mask(Q36_GPU_PRO4000, 0, 64, 1, 0, 0, pro4000);
    check_mask(Q36_GPU_PRO4000, 0, 64, 1, 1, 0, pro4000);
    check_device(8, 9, "NVIDIA GeForce RTX 4090", GIB(24), Q36_GPU_4090);
    check_device(12, 0, "NVIDIA GeForce RTX 5090", GIB(32), Q36_GPU_5090);
    check_device(12, 0, "NVIDIA RTX PRO 6000 Blackwell Server Edition", GIB(96), Q36_GPU_PRO6000);
    check_device(12, 0, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", GIB(96), Q36_GPU_NONE);
    check_device(8, 9, "NVIDIA L40S", GIB(48), Q36_GPU_L40S);
    check_device(8, 9, "NVIDIA L40", GIB(48), Q36_GPU_NONE);
    check_matrix();
    printf("Qwen3.6 CUDA profiles: %u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
