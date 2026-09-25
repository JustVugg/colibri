/* Model-free regression for Qwen3.6's expert -> slot index. */
#define COLI_CACHE_INDEX_TEST 1
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

static void init_cache(Model *m, int experts, int cap) {
    memset(m, 0, sizeof(*m));
    m->c.n_layers = 1; m->c.n_experts = experts;
    m->cache = calloc(1, sizeof(LCache));
    LCache *lc = &m->cache[0]; lc->cap = cap;
    lc->slots = calloc((size_t)cap, sizeof(Slot));
    lc->slot_by_expert = malloc((size_t)experts * sizeof(int));
    for (int e = 0; e < experts; e++) lc->slot_by_expert[e] = -1;
}

static void free_cache(Model *m) {
    free(m->cache[0].slot_by_expert);
    free(m->cache[0].slots);
    free(m->cache);
}

static void check_lookup_scaling(int cap) {
    Model m; init_cache(&m, cap, cap); LCache *lc = &m.cache[0]; lc->n = cap;
    for (int i = 0; i < cap; i++) cache_publish(&m, 0, &lc->slots[i], i);
    g_slot_index_probes = 0;
    CHECK(slot_indexed(&m, 0, cap-1) == &lc->slots[cap-1],
          "last-slot lookup failed at cap %d", cap);
    CHECK(g_slot_index_probes == 1,
          "indexed lookup used %llu probes at cap %d, expected 1",
          (unsigned long long)g_slot_index_probes, cap);
    printf("qwen36 lookup probes: cap=%d indexed=%llu legacy-worst=%d\n",
           cap, (unsigned long long)g_slot_index_probes, cap);
    free_cache(&m);
}

/* The eviction rule the demand path and the PILOT worker share: a slot a moe
 * run is computing from (held) or one being loaded is never given up, and a
 * pinned one only when nothing unpinned is left. */
static void check_victim(void) {
    Model m; init_cache(&m, 6, 3); LCache *lc = &m.cache[0]; lc->n = 3;
    for (int i = 0; i < 3; i++) { cache_publish(&m, 0, &lc->slots[i], i); lc->slots[i].used = (uint64_t)(i + 1); }
    CHECK(slot_victim(lc, 0) == 0, "LRU victim is not the oldest slot");

    Slot *h = expert_hold(&m, 0, 0);      /* a resident expert: the hit path holds too */
    CHECK(h == &lc->slots[0] && slot_busy(h), "expert_hold did not hold the resident slot");
    CHECK(slot_victim(lc, 0) == 1 && slot_victim(lc, 1) == 1, "a held slot was chosen for eviction");

    lc->slots[1].pinned = 1; slot_hold(&lc->slots[2]); slot_hold(&lc->slots[2]);   /* routed twice in one run */
    CHECK(slot_victim(lc, 0) == -1, "an unpinned victim came from pinned and held slots");
    CHECK(slot_victim(lc, 1) == 1, "the pinned fallback missed the one pinned free slot");
    cache_hide(&m, 0, &lc->slots[1]);    /* now being loaded */
    CHECK(slot_victim(lc, 1) == -1, "a slot being loaded was chosen for eviction");

    slot_release(&lc->slots[2]);
    CHECK(slot_victim(lc, 1) == -1, "one release freed a slot held twice");
    slot_release(&lc->slots[2]);
    CHECK(slot_victim(lc, 1) == 2, "a fully released slot stayed unevictable");
    slot_release(h);
    CHECK(slot_victim(lc, 1) == 0, "the released oldest slot is not the victim");
    free_cache(&m);
}

int main(void) {
    Model m; init_cache(&m, 6, 2); LCache *lc = &m.cache[0]; lc->n = 2;
    cache_publish(&m, 0, &lc->slots[0], 1);
    cache_publish(&m, 0, &lc->slots[1], 2);
    CHECK(slot_indexed(&m, 0, 1) == &lc->slots[0] &&
          slot_indexed(&m, 0, 2) == &lc->slots[1], "initial publication mismatch");

    cache_hide(&m, 0, &lc->slots[0]);
    CHECK(lc->slot_by_expert[1] == -1 && slot_indexed(&m, 0, 1) == NULL,
          "in-flight slot retained the evicted mapping");
    cache_publish(&m, 0, &lc->slots[0], 3);
    CHECK(slot_indexed(&m, 0, 2) == &lc->slots[1] &&
          slot_indexed(&m, 0, 3) == &lc->slots[0], "reuse damaged the index");

    lc->slot_by_expert[4] = 1;
    CHECK(slot_indexed(&m, 0, 4) == NULL, "stale entry served wrong weights");
    CHECK(slot_indexed(&m, 0, -1) == NULL && slot_indexed(&m, 0, 6) == NULL,
          "out-of-range lookup was accepted");

    free_cache(&m);
    check_victim();
    check_lookup_scaling(44);
    check_lookup_scaling(219);
    if (failures) { fprintf(stderr,"qwen36 cache index: %d failure(s)\n", failures); return 1; }
    puts("qwen36 cache index: ok");
    return 0;
}
