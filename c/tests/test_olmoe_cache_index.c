/* Model-free regression for OLMoE's expert -> slot index.  No checkpoint,
 * tokenizer, GPU, or meaningful RAM is required. */
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
    for (int i = 0; i < cap; i++) { lc->slots[i].eid = i; cache_publish(&m, 0, &lc->slots[i], i); }
    g_slot_index_probes = 0;
    CHECK(slot_indexed(&m, 0, cap-1) == &lc->slots[cap-1],
          "last-slot lookup failed at cap %d", cap);
    CHECK(g_slot_index_probes == 1,
          "indexed lookup used %llu probes at cap %d, expected 1",
          (unsigned long long)g_slot_index_probes, cap);
    printf("olmoe lookup probes: cap=%d indexed=%llu legacy-worst=%d\n",
           cap, (unsigned long long)g_slot_index_probes, cap);
    free_cache(&m);
}

/* The first read of a scenario stays open until the test releases it; every
 * later read returns at once.  g_io_loads counts them all. */
static pthread_mutex_t g_io_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_io_cv = PTHREAD_COND_INITIALIZER;
static int g_io_loads, g_io_open, g_io_release;

static void held_load(Model *m, int layer, int eid, Slot *s) {
    (void)m; (void)layer; (void)eid; (void)s;
    pthread_mutex_lock(&g_io_mx);
    if (g_io_loads++ == 0) {
        g_io_open = 1;
        pthread_cond_broadcast(&g_io_cv);
        while (!g_io_release) pthread_cond_wait(&g_io_cv, &g_io_mx);
    }
    pthread_mutex_unlock(&g_io_mx);
}

static int io_get(int *v) {
    pthread_mutex_lock(&g_io_mx); int x = *v; pthread_mutex_unlock(&g_io_mx);
    return x;
}

static void io_release(void) {
    pthread_mutex_lock(&g_io_mx);
    g_io_release = 1;
    pthread_cond_broadcast(&g_io_cv);
    pthread_mutex_unlock(&g_io_mx);
}

/* expert_get marks the expert routed under g_pilot_mx and keeps the lock until
 * it either waits or starts its own read: seeing the mark means it got there. */
static int routed(Model *m, int eid) {
    pthread_mutex_lock(&g_pilot_mx);
    int r = m->ehit && m->ehit[0][eid];
    pthread_mutex_unlock(&g_pilot_mx);
    return r;
}

typedef struct { Model *m; int eid, done; Slot *got; } Req;

static void *demand_thread(void *p) {
    Req *r = p; expert_get(r->m, 0, r->eid, &r->got);
    __atomic_store_n(&r->done, 1, __ATOMIC_RELEASE); return NULL;
}

static void *prefetch_thread(void *p) {
    Req *r = p; pilot_realload(r->m, 0, r->eid);
    __atomic_store_n(&r->done, 1, __ATOMIC_RELEASE); return NULL;
}

/* Bounded wait: a lost wakeup must fail the test, not hang the suite. */
static void wait_done(Req *r, pthread_t t, const char *who) {
    for (int ms = 0; ms < 5000 && !__atomic_load_n(&r->done, __ATOMIC_ACQUIRE); ms++) sleep_ms(1);
    if (!__atomic_load_n(&r->done, __ATOMIC_ACQUIRE)) {
        fprintf(stderr, "FAIL %s:%d: %s never returned\n", __FILE__, __LINE__, who);
        exit(1);
    }
    pthread_join(t, NULL);
}

static int copies(Model *m, int eid) {
    LCache *lc = &m->cache[0]; int n = 0;
    for (int i = 0; i < lc->n; i++) n += lc->slots[i].eid == eid;
    return n;
}

static void init_loader(Model *m) {
    init_cache(m, 8, 4);
    m->c.inter = 4; m->c.hidden = 4;
    m->is_queued = calloc(8, 1); m->is_pinned = calloc(8, 1);
    g_io_loads = g_io_open = g_io_release = 0;
    g_test_expert_load = held_load;
}

static void free_loader(Model *m) {
    LCache *lc = &m->cache[0];
    for (int i = 0; i < lc->n; i++) { free(lc->slots[i].g); free(lc->slots[i].gs); }
    if (m->ehit) { free(m->ehit[0]); free(m->ehit); }
    free(m->is_queued); free(m->is_pinned);
    free_cache(m);
    g_test_expert_load = NULL;
}

/* The prefetcher is reading expert 3 when the forward pass routes to it: the
 * forward pass must take the prefetched copy, not read it a second time. */
static void check_demand_joins_prefetch(void) {
    Model m; init_loader(&m);
    m.is_queued[3] = 1;
    Req pf = { &m, 3, 0, NULL }, dm = { &m, 3, 0, NULL };
    pthread_t tp, td;
    pthread_create(&tp, NULL, prefetch_thread, &pf);
    for (int ms = 0; ms < 5000 && !io_get(&g_io_open); ms++) sleep_ms(1);
    CHECK(io_get(&g_io_open), "prefetch never started its read");
    pthread_create(&td, NULL, demand_thread, &dm);
    for (int ms = 0; ms < 5000 && !routed(&m, 3); ms++) sleep_ms(1);
    io_release();
    wait_done(&pf, tp, "pilot_realload");
    wait_done(&dm, td, "expert_get");
    CHECK(g_io_loads == 1, "expert 3 read %d times, prefetch already had it in flight", g_io_loads);
    CHECK(copies(&m, 3) == 1, "expert 3 resident in %d slots", copies(&m, 3));
    CHECK(dm.got && dm.got == slot_indexed(&m, 0, 3), "forward pass got a slot the index does not name");
    free_loader(&m);
}

/* The forward pass is reading expert 4 when a prefetch for it comes up: the
 * prefetch must drop the request at once, without a read or a wait. */
static void check_prefetch_skips_demand(void) {
    Model m; init_loader(&m);
    Req dm = { &m, 4, 0, NULL };
    pthread_t td;
    pthread_create(&td, NULL, demand_thread, &dm);
    for (int ms = 0; ms < 5000 && !io_get(&g_io_open); ms++) sleep_ms(1);
    CHECK(io_get(&g_io_open), "forward pass never started its read");
    pthread_mutex_lock(&g_pilot_mx); m.is_queued[4] = 1; pthread_mutex_unlock(&g_pilot_mx);
    pilot_realload(&m, 0, 4);
    int loads = io_get(&g_io_loads);
    CHECK(loads == 1, "expert 4 read %d times, forward pass already had it in flight", loads);
    CHECK(m.is_queued[4] == 0, "dropped prefetch stayed queued");
    io_release();
    wait_done(&dm, td, "expert_get");
    CHECK(copies(&m, 4) == 1, "expert 4 resident in %d slots", copies(&m, 4));
    free_loader(&m);
}

/* Two callers route to expert 5 at once: the second waits for the first read
 * and must be woken by its publish. */
static void check_demand_joins_demand(void) {
    Model m; init_loader(&m);
    Req a = { &m, 5, 0, NULL }, b = { &m, 5, 0, NULL };
    pthread_t ta, tb;
    pthread_create(&ta, NULL, demand_thread, &a);
    for (int ms = 0; ms < 5000 && !io_get(&g_io_open); ms++) sleep_ms(1);
    CHECK(io_get(&g_io_open), "first caller never started its read");
    pthread_mutex_lock(&g_pilot_mx); m.ehit[0][5] = 0; pthread_mutex_unlock(&g_pilot_mx);
    pthread_create(&tb, NULL, demand_thread, &b);
    for (int ms = 0; ms < 5000 && !routed(&m, 5); ms++) sleep_ms(1);
    io_release();
    wait_done(&a, ta, "first expert_get");
    wait_done(&b, tb, "second expert_get");
    CHECK(g_io_loads == 1, "expert 5 read %d times by two callers", g_io_loads);
    CHECK(a.got && a.got == b.got, "two callers got different slots for expert 5");
    free_loader(&m);
}

int main(void) {
    Model m; init_cache(&m, 6, 2); LCache *lc = &m.cache[0]; lc->n = 2;
    cache_publish(&m, 0, &lc->slots[0], 1);
    cache_publish(&m, 0, &lc->slots[1], 2);
    CHECK(slot_indexed(&m, 0, 1) == &lc->slots[0] &&
          slot_indexed(&m, 0, 2) == &lc->slots[1], "initial publication mismatch");

    cache_hide(&m, 0, &lc->slots[1]);
    CHECK(lc->slot_by_expert[2] == -1 && slot_indexed(&m, 0, 2) == NULL,
          "in-flight slot retained the evicted mapping");
    cache_publish(&m, 0, &lc->slots[1], 3);
    CHECK(slot_indexed(&m, 0, 1) == &lc->slots[0] &&
          slot_indexed(&m, 0, 3) == &lc->slots[1], "reuse damaged the index");

    lc->slot_by_expert[4] = 0;
    CHECK(slot_indexed(&m, 0, 4) == NULL, "stale entry served wrong weights");
    CHECK(slot_indexed(&m, 0, -1) == NULL && slot_indexed(&m, 0, 6) == NULL,
          "out-of-range lookup was accepted");

    free_cache(&m);
    check_lookup_scaling(44);
    check_lookup_scaling(219);
    check_demand_joins_prefetch();
    check_prefetch_skips_demand();
    check_demand_joins_demand();
    if (failures) { fprintf(stderr,"olmoe cache index: %d failure(s)\n", failures); return 1; }
    puts("olmoe cache index: ok");
    return 0;
}
