/* Unit test for q38_cache_grow_ok (free-RAM floor freezing expert RAM cache
 * growth instead of pushing new slabs into the pagefile), with no model file
 * and no GPU. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void test_above_floor(void) {
    g_q38_ram_floor_gb = 0.0;
    LCache lc = {0};
    lc.cap = 4;

    lc.n = 0;
    CHECK(q38_cache_grow_ok(&lc) == 1 && lc.cap == 4, "above floor, cold layer grows");

    lc.n = 2;
    CHECK(q38_cache_grow_ok(&lc) == 1 && lc.cap == 4, "above floor, partial layer grows");

    lc.n = 4;
    CHECK(q38_cache_grow_ok(&lc) == 0 && lc.cap == 4, "above floor, full layer stops at cap");
}

static void test_under_floor(void) {
    /* Every machine has less than 1e9 GB of available RAM. */
    g_q38_ram_floor_gb = 1e9;

    /* Partially filled layer: freezes cap at n and declines further growth. */
    LCache lc = {0};
    lc.cap = 8;
    lc.n = 2;
    CHECK(q38_cache_grow_ok(&lc) == 0 && lc.cap == 2,
          "under floor, partial layer freezes cap at n (got cap=%d)", lc.cap);
    CHECK(q38_cache_grow_ok(&lc) == 0 && lc.cap == 2,
          "under floor, frozen layer stays frozen");

    /* Cold layer with cap > 1: clamps cap to 1 so the layer still gets its
     * first slot, then stops growing once n reaches 1. */
    LCache cold = {0};
    cold.cap = 8;
    cold.n = 0;
    CHECK(q38_cache_grow_ok(&cold) == 1 && cold.cap == 1,
          "under floor, cold layer clamps cap to 1 and allows first slot (got cap=%d)", cold.cap);
    cold.n = 1;
    CHECK(q38_cache_grow_ok(&cold) == 0 && cold.cap == 1,
          "under floor, cold layer stops after first slot (got cap=%d)", cold.cap);

    /* Cold layer already at cap == 1: always allows its first slot. */
    LCache single = {0};
    single.cap = 1;
    single.n = 0;
    CHECK(q38_cache_grow_ok(&single) == 1 && single.cap == 1,
          "under floor, cap=1 cold layer allows first slot");
}

static void test_batch_declines_under_floor(void) {
    Model m;
    memset(&m, 0, sizeof m);
    m.expert_parallel_reads = 1;
    m.c.layers = 2;
    m.c.experts = 8;

    LCache cache[2] = {0};
    cache[0].cap = 4;
    cache[0].n = 2;
    cache[1].cap = 4;
    cache[1].n = 0;
    m.cache = cache;

    int experts[3] = {0, 1, 2};
    Slot *sel[3] = {0};

    g_q38_ram_floor_gb = 1e9;

    /* Layer 0 has 2 slots filled; demand of 3 must freeze cap to 2 and decline. */
    int ok = q38_expert_get_batch(&m, 0, experts, 3, sel);
    CHECK(ok == 0 && cache[0].cap == 2 && cache[0].n == 2,
          "batch under floor freezes partial layer cap to n and declines (ok=%d cap=%d n=%d)",
          ok, cache[0].cap, cache[0].n);
    CHECK(m.expert_batch_fallback == Q38_EXPERT_BATCH_FALLBACK_CACHE_CAPACITY,
          "batch fallback reason is CACHE_CAPACITY (got %d)", (int)m.expert_batch_fallback);

    /* Layer 1 is cold (n=0); demand of 2 must clamp cap to 1 and decline so
     * callers fall back to single-slot serial streaming instead of allocating
     * a multi-slot batch under low RAM. */
    ok = q38_expert_get_batch(&m, 1, experts, 2, sel);
    CHECK(ok == 0 && cache[1].cap == 1 && cache[1].n == 0,
          "batch under floor clamps cold layer cap to 1 and declines (ok=%d cap=%d n=%d)",
          ok, cache[1].cap, cache[1].n);
}

int main(void) {
    CHECK(g_q38_ram_floor_gb == 1.0, "default floor is 1.0 GB (got %.2f)", g_q38_ram_floor_gb);
    test_above_floor();
    if (compat_mem_available_gb() > 0.0) {
        test_under_floor();
        test_batch_declines_under_floor();
    } else {
        printf("SKIP under-floor checks: compat_mem_available_gb() unavailable\n");
    }
    g_q38_ram_floor_gb = 1.0;
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
