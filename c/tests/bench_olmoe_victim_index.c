/* Eviction-pressure micro-benchmark for the #1342 recency-list victim pick:
 * list O(1) pick vs the legacy O(cap) scan it replaces, under the same state
 * shape (cap=219, publish-stamped used, every miss at full cap). Wall-clock
 * evidence for the PR description; not part of the automated suite. */
#define COLI_VICTIM_TEST 1
#define main olmoe_main_unused
#include "../olmoe.c"
#undef main

static double bench_now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
    enum { CAP = 219, CYCLES = 2000000 };
    Model m;
    memset(&m, 0, sizeof m);
    m.c.n_layers = 1; m.c.n_experts = CAP;
    m.cache = calloc(1, sizeof(LCache));
    LCache *lc = &m.cache[0];
    lc->cap = CAP; lc->n = CAP;
    lc->slots = calloc(CAP, sizeof(Slot));
    lc->slot_by_expert = malloc(CAP * sizeof(int));
    for (int e = 0; e < CAP; e++) lc->slot_by_expert[e] = -1;
    lc->ev_head = lc->ev_tail = -1;
    lc->pin_head = lc->pin_tail = -1;
    m.clock = 0;

    /* fill: publish every slot (publish refiles onto the ev list), used in order */
    for (int i = 0; i < CAP; i++) {
        cache_publish(&m, 0, &lc->slots[i], i);
        lc->slots[i].used = ++m.clock;
    }

    /* list victim: 2M pick+hide+publish cycles (one eviction each) */
    double t0 = bench_now_s();
    for (int k = 0; k < CYCLES; k++) {
        int v = victim_pick(&m, lc);
        if (v < 0) { fprintf(stderr, "no victim\n"); return 1; }
        cache_hide(&m, 0, &lc->slots[v]);
        cache_publish(&m, 0, &lc->slots[v], v);
        lc->slots[v].used = ++m.clock;
    }
    double dt_list = bench_now_s() - t0;

    /* legacy scan on the same shape: min used over n slots, 2M times */
    double t1 = bench_now_s();
    long long legacy_steps = 0;
    for (int k = 0; k < CYCLES; k++) {
        int lru = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
        }
        if (lru < 0) { fprintf(stderr, "legacy: no victim\n"); return 1; }
        legacy_steps += lc->n;
        volatile int sink = lru; (void)sink;
    }
    double dt_scan = bench_now_s() - t1;

    printf("list pick:      %.4f s for %d cycles (%.1f ns/cycle)\n",
           dt_list, CYCLES, dt_list * 1e9 / CYCLES);
    printf("legacy scan:    %.4f s for %d cycles (%.1f ns/cycle, %lld steps)\n",
           dt_scan, CYCLES, dt_scan * 1e9 / CYCLES, legacy_steps);
    printf("speedup at cap=%d: %.1fx\n", CAP, dt_scan / (dt_list > 1e-9 ? dt_list : 1e-9));
    return 0;
}