/* Model-free regression for OLMoE's O(1) LRU victim list (#1050). */
#define COLI_VICTIM_TEST 1
#define COLI_CACHE_INDEX_TEST 1
#define main olmoe_main_unused
#include "../olmoe.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

static void init_cache(Model *m, int experts, int cap) {
    memset(m, 0, sizeof(*m));
    m->c.n_layers = 1; m->c.n_experts = experts;
    m->cache = calloc(1, sizeof(LCache));
    LCache *lc = &m->cache[0];
    lc->cap = cap;
    lc->slots = calloc((size_t)cap, sizeof(Slot));
    lc->slot_by_expert = malloc((size_t)experts * sizeof(int));
    for (int e = 0; e < experts; e++) lc->slot_by_expert[e] = -1;
    lc->ev_head = lc->ev_tail = -1;
}

static void free_cache(Model *m) {
    free(m->cache[0].slot_by_expert);
    free(m->cache[0].slots);
    free(m->cache);
}

/* Legacy oracle: oldest unpinned resident, else oldest non-in-flight. */
static int oracle_pick(LCache *lc, int allow_pinned) {
    int lru = -1;
    for (int i = 0; i < lc->n; i++) {
        if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
        if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
    }
    if (lru >= 0 || !allow_pinned) return lru;
    for (int i = 0; i < lc->n; i++) {
        if (lc->slots[i].eid < 0) continue;
        if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
    }
    return lru;
}

static void publish(Model *m, int slot, int eid, int pinned, uint64_t used) {
    LCache *lc = &m->cache[0];
    Slot *s = &lc->slots[slot];
    if (slot >= lc->n) lc->n = slot + 1;
    cache_publish(m, 0, s, eid);
    s->pinned = pinned;
    s->used = used;
    victim_refile(lc, s, slot);
}

static void check_basic_order(void) {
    Model m; init_cache(&m, 8, 4); LCache *lc = &m.cache[0];
    publish(&m, 0, 10, 0, 1);
    publish(&m, 1, 11, 0, 2);
    publish(&m, 2, 12, 0, 3);
    publish(&m, 3, 13, 0, 4);
    CHECK(victim_pick(lc, 1) == 0, "LRU should be slot 0 (used=1)");
    victim_touch(lc, &lc->slots[0], 0); lc->slots[0].used = 5;
    CHECK(victim_pick(lc, 1) == 1, "after touch(0), LRU should be slot 1");
    /* Pin the current LRU — it must leave the ev-list. */
    lc->slots[1].pinned = 1;
    victim_refile(lc, &lc->slots[1], 1);
    CHECK(victim_pick(lc, 0) == 2, "pinned LRU skipped when allow_pinned=0");
    CHECK(victim_pick(lc, 1) == 2, "next unpinned is slot 2");
    /* Hide (in-flight) the next victim. */
    cache_hide(&m, 0, &lc->slots[2]);
    CHECK(victim_pick(lc, 1) == 3, "in-flight slot skipped");
    cache_hide(&m, 0, &lc->slots[3]);
    /* Slot 0 is still an unpinned resident (touched earlier). */
    CHECK(victim_pick(lc, 0) == 0, "slot 0 still unpinned on ev-list");
    CHECK(victim_pick(lc, 1) == 0, "ev-list head is slot 0");
    /* Pin the last unpinned slot → only pinned residents remain. */
    lc->slots[0].pinned = 1;
    victim_refile(lc, &lc->slots[0], 0);
    CHECK(victim_pick(lc, 0) == -1, "no unpinned resident");
    CHECK(victim_pick(lc, 1) == 1, "fallback to oldest pinned (used=2)");
    free_cache(&m);
}

static void check_matches_oracle(void) {
    Model m; init_cache(&m, 64, 16); LCache *lc = &m.cache[0];
    uint64_t clock = 1;
    /* Fill */
    for (int i = 0; i < 16; i++) publish(&m, i, i, 0, clock++);
    /* Mix of touches, pins, hides, republishes */
    for (int step = 0; step < 200; step++) {
        int op = step % 5;
        int i = (step * 7) % lc->n;
        Slot *s = &lc->slots[i];
        if (op == 0 && s->eid >= 0 && !s->pinned) {
            s->used = clock++;
            victim_touch(lc, s, i);
        } else if (op == 1 && s->eid >= 0) {
            s->pinned = 1;
            victim_refile(lc, s, i);
        } else if (op == 2 && s->eid >= 0) {
            s->pinned = 0;
            s->used = clock++;
            victim_refile(lc, s, i);
        } else if (op == 3 && s->eid >= 0) {
            cache_hide(&m, 0, s);
        } else {
            int eid = 100 + (step % 50);
            cache_publish(&m, 0, s, eid);
            s->pinned = (step % 11 == 0);
            s->used = clock++;
            victim_refile(lc, s, i);
        }
        int got0 = victim_pick(lc, 0), exp0 = oracle_pick(lc, 0);
        int got1 = victim_pick(lc, 1), exp1 = oracle_pick(lc, 1);
        CHECK(got0 == exp0, "step %d allow0: got %d expected %d", step, got0, exp0);
        CHECK(got1 == exp1, "step %d allow1: got %d expected %d", step, got1, exp1);
        if (failures > 8) break;
    }
    free_cache(&m);
}

static void check_hot_path_is_o1(void) {
    Model m; init_cache(&m, 256, 64); LCache *lc = &m.cache[0];
    for (int i = 0; i < 64; i++) publish(&m, i, i, 0, (uint64_t)(i + 1));
    g_victim_picks = g_victim_fallback_scans = g_victim_fallback_len = 0;
    for (int i = 0; i < 1000; i++) {
        int v = victim_pick(lc, 1);
        CHECK(v == 0, "stable LRU should stay at slot 0, got %d", v);
        /* Touch a non-LRU slot — list head must not change. */
        int t = 1 + (i % 63);
        lc->slots[t].used = 1000 + (uint64_t)i;
        victim_touch(lc, &lc->slots[t], t);
    }
    CHECK(g_victim_picks == 1000, "expected 1000 picks, got %llu",
          (unsigned long long)g_victim_picks);
    CHECK(g_victim_fallback_scans == 0,
          "hot path must not fall back to O(cap) scan (%llu scans, %llu len)",
          (unsigned long long)g_victim_fallback_scans,
          (unsigned long long)g_victim_fallback_len);
    printf("olmoe victim picks: %llu  fallback_scans: %llu  fallback_len: %llu\n",
           (unsigned long long)g_victim_picks,
           (unsigned long long)g_victim_fallback_scans,
           (unsigned long long)g_victim_fallback_len);
    free_cache(&m);
}

int main(void) {
    check_basic_order();
    check_matches_oracle();
    check_hot_path_is_o1();
    if (failures) {
        fprintf(stderr, "olmoe victim index: %d failure(s)\n", failures);
        return 1;
    }
    puts("olmoe victim index: ok");
    return 0;
}
