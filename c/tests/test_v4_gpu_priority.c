#define main unused_merged_test_main
#include "test_deepseek_v4.c"
#undef main

int coli_v4_test_gpu_frequency_victim(const uint64_t *, const uint64_t *, int, uint64_t);
int coli_v4_test_gpu_ram_dedupe_ring(void);

int main(void) {
    uint64_t frequency[64], clock[64];
    for (int i = 0; i < 64; i++) {
        frequency[i] = 100;
        clock[i] = (uint64_t)(i + 1);
    }
    frequency[0] = 50000;
    frequency[12] = 1;
    clock[0] = 0;
    /* An old, heavily used expert survives a newer, rarely used one. */
    assert(coli_v4_test_gpu_frequency_victim(frequency, clock, 64, 100) == 12);
    frequency[12] = 100;
    frequency[4] = 2;
    frequency[5] = 2;
    assert(coli_v4_test_gpu_frequency_victim(frequency, clock, 64, 100) == 4);
    /* Never recycle a view just attached by the current forward. */
    frequency[63] = 0;
    clock[63] = 100;
    assert(coli_v4_test_gpu_frequency_victim(frequency, clock, 64, 100) == 4);
    for (int i = 0; i < 8; i++) {
        frequency[i] = 100;
        clock[i] = (uint64_t)(i + 1);
    }
    frequency[6] = 1;
    /* A nearly full eight-entry cache must retain the current layer's
     * newly attached views even if one has a very low frequency. */
    assert(coli_v4_test_gpu_frequency_victim(frequency, clock, 8, 10) == 0);

    /* Opt-in V4_GPU_HOT_CACHE=1: RAM dedupe bitmap (ram_first) + 16 transient ring slots. */
    assert(coli_v4_test_gpu_ram_dedupe_ring() == 0);

    char directory[] = "colibri-v4-gpu-priority-XXXXXX", path[512], error[256];
    assert(mkdtemp(directory));
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    assert(!write_fixture_experts(path, 8));
    setenv("COLI_V4_AUTOPIN", "0", 1);
    setenv("USAGE_SAVE", "0", 1);
    setenv("COLI_V4_PREWARM", "0", 1);
    setenv("COLI_V4_ROWS16", "0", 1);
    ColiDeepSeekV4ExpertStoreOptions options = {directory, 1, 8, 8 * 51, -1, UINT64_MAX, 0, 0};
    ColiExpertStore *store = NULL;
    assert(!coli_deepseek_v4_expert_store_open(&options, &store, error, sizeof(error)));
    ColiExpertKey key = {0, 3};
    assert(coli_v4_expert_store_frequency(store, key) == 0);
    coli_v4_expert_store_note_routed(store, key);
    coli_v4_expert_store_note_routed(store, key);
    assert(coli_v4_expert_store_frequency(store, key) == 2);
    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    assert(stats.requests == 0 && stats.misses == 0 && stats.bytes_read == 0);
    store->ops->destroy(store);
    assert(!scratch_remove(directory, path));
    puts("v4 GPU priority: frequency beats recency, live views protected, RAM dedupe + 16 transient ring slots, CUDA history counted without RAM reads: ok");
    return 0;
}
