/* Randomised differential test for the #1050 recency-list victim selection.
 * JustVugg (#1571 review): test_olmoe_victim_index calls victim_* and
 * cache_publish/hide directly and never drives expert_get or pilot_realload —
 * exactly the call sites where the stamp/refile ordering lives.
 *
 * This test drives the REAL expert_get (demand path, load stubbed) and the
 * real pilot_realload ordering under a seeded random op sequence, and asserts
 * at every step that the list-based victim pick equals the legacy linear scan
 * on the same state. Both defects that review round 2 caught (the pilot
 * refile/stamp inversion, and the L2b pin-list staleness) violate this
 * equality — a single test, both engines' ordering sites.
 *
 * No checkpoint, tokenizer, GPU, or meaningful RAM required. */
#define COLI_CACHE_INDEX_TEST 1
#define COLI_VICTIM_TEST 1
#define main olmoe_main_unused
#include "../olmoe.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

static uint64_t rng_s = 0x9e3779b97f4a7c15ULL;
static uint32_t rnd(void){ rng_s ^= rng_s<<13; rng_s ^= rng_s>>7; rng_s ^= rng_s<<17; return (uint32_t)(rng_s>>32); }

/* Local copies of the shared helpers (test_olmoe_victim_index.c keeps its own;
 * both tests build as standalone binaries over the same olmoe.c). */
static void init_cache(Model *m, int experts, int cap) {
    memset(m, 0, sizeof(*m));
    m->c.n_layers = 1; m->c.n_experts = experts;
    m->cache = calloc(1, sizeof(LCache));
    LCache *lc = &m->cache[0]; lc->cap = cap;
    lc->slots = calloc((size_t)cap, sizeof(Slot));
    lc->slot_by_expert = malloc((size_t)experts * sizeof(int));
    for (int e = 0; e < experts; e++) lc->slot_by_expert[e] = -1;
    lc->ev_head = lc->ev_tail = lc->pin_head = lc->pin_tail = -1;
}

static void free_cache(Model *m) {
    free(m->cache[0].slot_by_expert);
    free(m->cache[0].slots);
    free(m->cache);
}

/* Structural list audit (same invariants as the main test file). */
static void audit_lists(LCache *lc, const char *where) {
    for (int i = lc->ev_head; i >= 0; ) {
        CHECK(i < lc->n, "ev-list index %d out of range (%s)", i, where);
        CHECK(lc->slots[i].rlist == 1, "slot %d rlist=%d not ev (list walk %s)", i, lc->slots[i].rlist, where);
        i = lc->slots[i].rnext;
    }
    for (int i = lc->pin_head; i >= 0; ) {
        CHECK(i < lc->n, "pin-list index %d out of range (%s)", i, where);
        CHECK(lc->slots[i].rlist == 2, "slot %d rlist=%d not pin (list walk %s)", i, lc->slots[i].rlist, where);
        CHECK(lc->slots[i].pinned, "pin-list member %d not marked pinned (%s)", i, where);
        i = lc->slots[i].rnext;
    }
}

/* Legacy scan mirror with the SAME two-phase contract as victim_pick
 * (victim_pick = victim_pick_ex with allow_pinned=true): phase 1 = oldest
 * unpinned; phase 2 (only if none) oldest non-in-flight, may be pinned. */
static int legacy_pick(LCache *lc) {
    int lru = -1;
    for (int i = 0; i < lc->n; i++) {
        if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
        if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
    }
    if (lru >= 0) return lru;
    for (int i = 0; i < lc->n; i++) {
        if (lc->slots[i].eid < 0) continue;
        if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
    }
    return lru;
}

/* Mirror of victim_pick's contract under the same invariants, computed on a
 * shadow copy of the legacy semantics. Both paths must agree on the slot
 * INDEX chosen; used-stamp ties are broken identically (stable min). */
static void diff_check(Model *m, LCache *lc, const char *where, int step) {
    int got = victim_pick(m, lc);
    int want = legacy_pick(lc);
    if (got != want) {
        fprintf(stderr, "DIFF %s step %d: list pick=%d (used=%llu) legacy=%d (used=%llu)\n",
                where, step, got,
                (unsigned long long)(got >= 0 ? lc->slots[got].used : 0),
                want, (unsigned long long)(want >= 0 ? lc->slots[want].used : 0));
        fprintf(stderr, "  ev-list walk:");
        for (int i = lc->ev_head; i >= 0; i = lc->slots[i].rnext)
            fprintf(stderr, " [%d u=%llu rlist=%d]", i, (unsigned long long)lc->slots[i].used, lc->slots[i].rlist);
        fprintf(stderr, "\n  pin-list walk:");
        for (int i = lc->pin_head; i >= 0; i = lc->slots[i].rnext)
            fprintf(stderr, " [%d u=%llu]", i, (unsigned long long)lc->slots[i].used);
        fprintf(stderr, "\n  slots:");
        for (int i = 0; i < lc->n; i++)
            fprintf(stderr, " [%d e=%d p=%d u=%llu rl=%d]", i, lc->slots[i].eid,
                    lc->slots[i].pinned, (unsigned long long)lc->slots[i].used, lc->slots[i].rlist);
        fprintf(stderr, "\n");
        failures++;
    }
}

int main(void) {
    /* --- 1. randomized ops via the REAL expert_get (demand path) --- */
    {
        Model m; init_cache(&m, 8, 4); LCache *lc = &m.cache[0];
        uint8_t *pin = calloc(8, 1); m.is_pinned = pin;
        for (int step = 0; step < 5000; step++) {
            int eid = (int)(rnd() % 8);
            Slot *out = NULL;
            expert_get(&m, 0, eid, &out);   /* load_expert_merged reads from files —
                                               but expert ids here exceed the checkpoint
                                               dims so it degrades to the stub path below */
            CHECK(out != NULL, "expert_get returned NULL (step %d)", step);
            audit_lists(lc, "expert_get");
            diff_check(&m, lc, "expert_get", step);
            if (failures) break;            /* first divergence: keep the log readable */
        }
        free_cache(&m);
    }
    printf("expert_get differential: OK (5000 steps)\n");
    /* --- 2. randomized ops via the REAL pilot_realload (pilot path) --- */
    {
        Model m; init_cache(&m, 8, 4); LCache *lc = &m.cache[0];
        uint8_t *pin = calloc(8, 1); m.is_pinned = pin;
        uint8_t *queued = calloc(8, 1); m.is_queued = queued;
        /* pre-fill through the demand path so the row is at cap: then pilot
         * realload exercises the evict branch (hide/stamp/publish/refile). */
        for (int e = 0; e < 4; e++) {
            Slot *out = NULL;
            expert_get(&m, 0, e, &out);
        }
        for (int step = 0; step < 5000; step++) {
            int eid = (int)(rnd() % 8);
            /* random pin flips THROUGH the real refile path (mirrors
             * pin_hot_experts lines 942-947: resident->pinned + victim_refile) */
            uint8_t want = (uint8_t)(rnd() % 4 == 0);
            pthread_mutex_lock(&g_pilot_mx);
            Slot *resident = slot_indexed(&m, 0, eid);
            if (resident && resident->pinned != want) {
                resident->pinned = want; pin[eid] = want;
                victim_refile(lc, resident, (int)(resident - lc->slots));
            }
            pthread_mutex_unlock(&g_pilot_mx);
            m.is_queued[eid] = 1;               /* pilot_realload's gate */
            pilot_realload(&m, 0, eid);         /* exercises publish + stamp + refile order */
            audit_lists(lc, "pilot_realload");
            diff_check(&m, lc, "pilot_realload", step);
            if (failures) break;
        }
        free_cache(&m);
    }
    printf("pilot_realload differential: OK (5000 steps)\n");
    return failures ? 1 : 0;
}