/* Model-free regression for OLMoE's recency-list LRU victim selection (#1050).
 * Complements test_olmoe_cache_index.c (PR #1215): the victim *scan* path.
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

/* Fill the cache to cap with distinct experts via the real publish path.
 * fill order slot i = expert i, used stamps increase i+1. */
static void fill(Model *m, int cap) {
    LCache *lc = &m->cache[0];
    for (int i = 0; i < cap; i++) {
        lc->slots[i].eid = i;
        lc->slots[i].used = ++m->clock;
        lc->n = i + 1;
        cache_publish(m, 0, &lc->slots[i], i);
    }
}

/* Structural list audit: both lists well-formed, exactly the right members. */
static void audit_lists(LCache *lc, const char *where) {
    (void)where;
    int seen = 0;
    for (int i = lc->ev_head; i >= 0; ) {
        CHECK(i < lc->n, "ev-list index %d out of range (%s)", i, where);
        CHECK(lc->slots[i].rlist == 1, "slot %d rlist=%d not ev (list walk)", i, lc->slots[i].rlist);
        int nx = lc->slots[i].rnext;
        if (nx >= 0) CHECK(lc->slots[nx].rprev == i, "broken ev link %d->%d", i, nx);
        seen++;
        if (nx < 0) { CHECK(i == lc->ev_tail, "ev walk ends at %d but tail=%d", i, lc->ev_tail); break; }
        i = nx;
    }
    for (int i = lc->pin_head; i >= 0; ) {
        CHECK(i < lc->n, "pin-list index %d out of range", i);
        CHECK(lc->slots[i].rlist == 2, "slot %d rlist=%d not pin (list walk)", i, lc->slots[i].rlist);
        int nx = lc->slots[i].rnext;
        if (nx >= 0) CHECK(lc->slots[nx].rprev == i, "broken pin link %d->%d", i, nx);
        seen++;
        if (nx < 0) { CHECK(i == lc->pin_tail, "pin walk ends at %d but tail=%d", i, lc->pin_tail); break; }
        i = nx;
    }
    /* in-flight slots (eid<0) are legitimately off-list, so audit expects
     * exactly the resident count, not lc->n (fill() makes all resident;
     * hide() removes one by design). */
    int resident = 0;
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid >= 0) resident++;
    CHECK(seen == resident, "%s: %d slots on lists but %d resident (n=%d)",
          where, seen, resident, lc->n);
}

int main(void) {
    /* --- 1. victim pick: head of ev-list is exact LRU --- */
    {
        Model m; init_cache(&m, 6, 3); LCache *lc = &m.cache[0];
        fill(&m, 3);                       /* slot0 oldest (used=1), slot2 newest */
        lc->slots[1].used = ++m.clock;     /* touch slot 1 */
        victim_touch(lc, &lc->slots[1], 1);
        audit_lists(lc, "touch");
        int v = victim_pick(&m, lc);
        CHECK(v == 0, "victim picked slot %d, expected 0 (exact LRU order)", v);
        free_cache(&m);
    }

    /* --- 2. pinned slot is never the victim while an unpinned exists --- */
    {
        Model m; init_cache(&m, 6, 2); LCache *lc = &m.cache[0];
        fill(&m, 2);
        lc->slots[0].pinned = 1;
        victim_refile(lc, &lc->slots[0], 0);     /* pin the LRU */
        audit_lists(lc, "pin-flip");
        int v = victim_pick(&m, lc);
        CHECK(v == 1, "victim picked pinned slot 0 (got %d)", v);
        free_cache(&m);
    }

    /* --- 3. all pinned -> fallback returns the oldest non-in-flight (pin head) --- */
    {
        Model m; init_cache(&m, 6, 2); LCache *lc = &m.cache[0];
        fill(&m, 2);
        lc->slots[0].pinned = lc->slots[1].pinned = 1;
        victim_refile(lc, &lc->slots[0], 0);
        victim_refile(lc, &lc->slots[1], 1);
        int v = victim_pick(&m, lc);
        CHECK(v == 0, "all-pinned fallback must return oldest slot 0 (got %d)", v);
        free_cache(&m);
    }

    /* --- 4. in-flight (eid==-1 via hide) excluded, restored after publish --- */
    {
        Model m; init_cache(&m, 6, 2); LCache *lc = &m.cache[0];
        fill(&m, 2);
        cache_hide(&m, 0, &lc->slots[0]);        /* eid=-1, unlinks */
        audit_lists(lc, "hide");
        int v = victim_pick(&m, lc);
        CHECK(v == 1, "in-flight slot selected as victim (got %d)", v);
        cache_publish(&m, 0, &lc->slots[0], 9);  /* resident again */
        lc->slots[0].used = ++m.clock;
        victim_refile(lc, &lc->slots[0], 0);
        audit_lists(lc, "republish");
        v = victim_pick(&m, lc);
        CHECK(v == 1, "after publish the oldest victim is still slot 1 (got %d)", v);
        free_cache(&m);
    }

    /* --- 5. kill-switch: COLI_VICTIM_SCAN=1 restores the legacy scan --- */
    {
        Model m; init_cache(&m, 6, 3); LCache *lc = &m.cache[0];
        fill(&m, 3);
        g_victim_scan_mode = 1;
        g_victim_scans = 0; g_victim_scan_len = 0;
        int v = victim_pick(&m, lc);
        g_victim_scan_mode = 0;
        CHECK(v == 0, "legacy scan picked slot %d, expected 0", v);
        CHECK(g_victim_scans == 1 && g_victim_scan_len == 3,
              "legacy counters wrong: scans=%llu len=%llu",
              (unsigned long long)g_victim_scans, (unsigned long long)g_victim_scan_len);
        free_cache(&m);
    }

    /* --- 8. DELAYED pin-flip order (Sol-r1 M3 differential): older slot pinned
     * after a newer one must become pin-head — the all-pinned fallback and the
     * legacy scan must agree on the same victim. --- */
    {
        Model m; init_cache(&m, 6, 3); LCache *lc = &m.cache[0];
        fill(&m, 3);                       /* used: slot0=1, slot1=2, slot2=3 */
        lc->slots[2].pinned = 1;           /* pin NEWEST first -> pin list [2] */
        victim_refile(lc, &lc->slots[2], 2);
        lc->slots[0].pinned = 1;           /* pin OLDER later -> must land at pin HEAD */
        victim_refile(lc, &lc->slots[0], 0);
        lc->slots[1].pinned = 1;
        victim_refile(lc, &lc->slots[1], 1);
        audit_lists(lc, "delayed pin");
        int v = victim_pick(&m, lc);       /* all pinned -> pin head */
        CHECK(v == 0, "all-pinned fallback picked %d, scan order says slot 0 (used=1)", v);
        /* legacy scan agreement */
        g_victim_scan_mode = 1;
        int vs = victim_pick(&m, lc);
        g_victim_scan_mode = 0;
        CHECK(vs == v, "legacy scan (%d) and list pick (%d) disagree after delayed flips", vs, v);
        /* speculation must NOT displace pinned slots at all */
        int vx = victim_pick_ex(&m, lc, 0);
        CHECK(vx == -1, "victim_pick_ex(allow_pinned=0) returned %d — speculation would evict pinned", vx);
        free_cache(&m);
    }

    /* --- 9. partial-pin state: pick_ex skips pinned, pick takes pin head only as last resort --- */
    {
        Model m; init_cache(&m, 6, 3); LCache *lc = &m.cache[0];
        fill(&m, 3);
        lc->slots[0].pinned = 1; victim_refile(lc, &lc->slots[0], 0);
        int v = victim_pick_ex(&m, lc, 0);
        CHECK(v == 1, "pick_ex picked %d, expected slot 1 (LRU unpinned)", v);
        int vp = victim_pick(&m, lc);
        CHECK(vp == 1, "full pick picked %d, expected slot 1 (unpinned LRU wins over pin head)", vp);
        free_cache(&m);
    }

    /* --- 10. churn: publish/hide/touch sequence keeps lists exact --- */
    {
        Model m; init_cache(&m, 8, 4); LCache *lc = &m.cache[0];
        fill(&m, 4);
        /* evict LRU (slot 0) style churn: hide + republish with new expert */
        for (int round = 0; round < 6; round++) {
            int v = victim_pick(&m, lc);
            CHECK(v >= 0 && v < lc->n, "churn pick %d out of range", v);
            cache_hide(&m, 0, &lc->slots[v]);
            audit_lists(lc, "churn hide");
            cache_publish(&m, 0, &lc->slots[v], (v + 3) % 8);
            lc->slots[v].used = ++m.clock;
            victim_refile(lc, &lc->slots[v], v);
            audit_lists(lc, "churn publish");
            /* touch a random other slot */
            int t = (v + 1 + round) % lc->n;
            lc->slots[t].used = ++m.clock;
            victim_touch(lc, &lc->slots[t], t);
            audit_lists(lc, "churn touch");
        }
        free_cache(&m);
    }

    /* --- 7. scaling: list pick cost is O(1) with cap --- */
    {
        Model m; init_cache(&m, 64, 64); LCache *lc = &m.cache[0];
        fill(&m, 64);
        int v = victim_pick(&m, lc);
        CHECK(v == 0, "cap=64 victim should be slot 0 head (got %d)", v);
        /* no linear probes on the list path (probes only in legacy mode) */
        g_victim_scan_len = 0;
        (void)victim_pick(&m, lc);
        CHECK(g_victim_scan_len == 0, "list path did linear work (%llu slots)",
              (unsigned long long)g_victim_scan_len);
        free_cache(&m);
    }

    if (failures) { fprintf(stderr,"olmoe victim index: %d failure(s)\n", failures); return 1; }
    puts("olmoe victim index: ok");
    return 0;
}