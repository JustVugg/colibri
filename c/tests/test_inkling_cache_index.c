/* Model-free regression for Inkling's expert -> slot index. The fixture uses
 * two tiny slots: no checkpoint, tokenizer, GPU or meaningful RAM required. */
#define COLI_CACHE_INDEX_TEST 1
#define main inkling_main_unused
#include "../inkling.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

static void init_cache(Model *m){
    memset(m,0,sizeof(*m));
    m->c.n_layers=1; m->c.n_experts=6; m->c.hidden=2; m->c.moe_inter=2;
    m->cache=calloc(1,sizeof(LCache));
    m->cache[0].cap=2;
    m->cache[0].slots=calloc(2,sizeof(Slot));
    m->cache[0].slot_by_expert=malloc(6*sizeof(int));
    for(int e=0;e<6;e++) m->cache[0].slot_by_expert[e]=-1;
}

static void free_cache(Model *m){
    for(int i=0;i<m->cache[0].cap;i++){
        free(m->cache[0].slots[i].f13);
        free(m->cache[0].slots[i].f2);
    }
    free(m->cache[0].slot_by_expert);
    free(m->cache[0].slots);
    free(m->cache);
}

static void check_lookup_scaling(int cap){
    Model m; memset(&m,0,sizeof(m));
    m.c.n_layers=1; m.c.n_experts=cap;
    m.cache=calloc(1,sizeof(LCache));
    LCache *lc=&m.cache[0]; lc->cap=lc->n=cap;
    lc->slots=calloc((size_t)cap,sizeof(Slot));
    lc->slot_by_expert=malloc((size_t)cap*sizeof(int));
    for(int i=0;i<cap;i++){ lc->slots[i].eid=i; lc->slot_by_expert[i]=i; }
    g_slot_index_probes=0;
    CHECK(slot_indexed(&m,0,cap-1)==&lc->slots[cap-1],
          "last-slot lookup failed at cap %d",cap);
    CHECK(g_slot_index_probes==1,
          "indexed lookup used %llu probes at cap %d, expected 1",
          (unsigned long long)g_slot_index_probes,cap);
    printf("inkling lookup probes: cap=%d indexed=%llu legacy-worst=%d\n",
           cap,(unsigned long long)g_slot_index_probes,cap);
    free(lc->slot_by_expert); free(lc->slots); free(m.cache);
}

/* Tiny real tensor bytes exercise production slot_fill and moe_ex. The scalar
 * reference calls the same model one row at a time, so it never holds a round's
 * cache slots across another acquire. No checkpoint or optional GPU is needed. */
enum { ROUND_E = 16, ROUND_D = 2, ROUND_I = 2, ROUND_S = 19 };
static void init_round_model(Model *m, int cap, int pin_limit, int pins, int topk,
                             st_tensor *tensors) {
    memset(m, 0, sizeof(*m));
    m->c.n_layers = 1; m->c.n_experts = ROUND_E;
    m->c.hidden = ROUND_D; m->c.moe_inter = ROUND_I;
    m->c.topk = topk; m->c.route_scale = 1.f;
    m->npin = pin_limit;
    m->cache = calloc(1, sizeof(LCache));
    if (!m->cache) exit(2);
    LCache *lc = m->cache;
    lc->cap = cap; lc->slots = calloc((size_t)cap, sizeof(Slot));
    lc->slot_by_expert = malloc(ROUND_E * sizeof(int));
    if (!lc->slots || !lc->slot_by_expert) exit(2);
    for (int e = 0; e < ROUND_E; e++) lc->slot_by_expert[e] = -1;
    for (int i = 0; i < cap; i++) lc->slots[i].eid = -1;
    m->S.t = tensors; m->S.n = 2;
    for (int e = 0; e < pins; e++) {
        Slot *slot = slot_acquire(m, 0, e);
        slot_fill(m, 0, slot); slot->pinned = 1;
    }
}
static void free_round_model(Model *m) {
    free_cache(m);
    if (m->ehit) { free(m->ehit[0]); free(m->ehit); }
}
static void check_pinned_rounds(const char *name, int cap, int pin_limit,
                               int pins, int mixed, int topk, int cpu_pairs) {
    FILE *weights = tmpfile();
    if (!weights) { perror("tmpfile expert weights"); exit(2); }
    float data[ROUND_E * (2 * ROUND_I * ROUND_D + ROUND_D * ROUND_I)];
    for (size_t i = 0; i < sizeof(data) / sizeof(data[0]); i++)
        data[i] = (float)((int)(i % 17) - 8) / 13.f;
    if (fwrite(data, sizeof(data), 1, weights) != 1 || fflush(weights)) exit(2);
    st_tensor tensors[2] = {
        {.name = "model.layers.0.mlp.experts.gate_up_proj",
         .fd = fileno(weights), .dtype = 2, .off = 0},
        {.name = "model.layers.0.mlp.experts.down_proj",
         .fd = fileno(weights), .dtype = 2,
         .off = ROUND_E * 2 * ROUND_I * ROUND_D * (int64_t)sizeof(float)},
    };
    Model batch, scalar;
    init_round_model(&batch, cap, pin_limit, pins, topk, tensors);
    init_round_model(&scalar, cap, pin_limit, pins, topk, tensors);
    Layer layer; memset(&layer, 0, sizeof(layer));
    float bias[ROUND_E] = {0}; layer.rbias = bias; layer.rgs = 1.f;
    float x[ROUND_S * ROUND_D], logits[ROUND_S * ROUND_E];
    float actual[ROUND_S * ROUND_D], expected[ROUND_S * ROUND_D];
    for (int s = 0; s < ROUND_S; s++) {
        int expert = mixed && s % 4 == 0 ? 0 : 2 + s % (ROUND_E - 2);
        for (int d = 0; d < ROUND_D; d++)
            x[s * ROUND_D + d] = (float)(s + d + 1) / 23.f;
        for (int e = 0; e < ROUND_E; e++)
            logits[s * ROUND_E + e] = e == expert ? 2.f : -2.f;
        if (!cpu_pairs)
            moe_ex(&scalar, &layer, 0, x + s * ROUND_D, 1,
                   expected + s * ROUND_D, logits + s * ROUND_E, NULL);
    }
    if (cpu_pairs) {
        int ids[ROUND_S * 2]; uint8_t want[ROUND_S * 2];
        float got[ROUND_S * 2 * ROUND_D], ref[ROUND_S * 2 * ROUND_D];
        float scratch[2 * ROUND_I];
        for (int i = 0; i < ROUND_S * topk; i++) {
            ids[i] = mixed && i % 4 == 0 ? 0 : 2 + i % (ROUND_E - 2);
            if (i % 7 == 3) ids[i] = -1;
            want[i] = i % 5 != 2;
            for (int d = 0; d < ROUND_D; d++)
                got[i * ROUND_D + d] = ref[i * ROUND_D + d] = 123.f;
        }
        for (int i = 0; i < ROUND_S * topk; i++) {
            int one[2] = {-1, -1}; uint8_t take[2] = {0, 0};
            one[i % topk] = ids[i]; take[i % topk] = want[i];
            ink_vk_cpu_pairs(&scalar, 0, x + (i / topk) * ROUND_D,
                             topk, one, take,
                             ref + (i / topk) * topk * ROUND_D, scratch);
        }
        ink_vk_cpu_pairs(&batch, 0, x, ROUND_S * topk, ids, want, got, scratch);
        CHECK(!memcmp(got, ref, (size_t)ROUND_S * topk * ROUND_D * sizeof(float)),
              "%s CPU fallback differs from per-pair reference", name);
        for (int i = 0; i < ROUND_S * topk; i++)
            for (int d = 0; d < ROUND_D; d++) {
                float value = got[i * ROUND_D + d];
                CHECK(isfinite(value), "%s non-finite pair %d", name, i);
                CHECK((value != 123.f) == (want[i] && ids[i] >= 0),
                      "%s pair %d was skipped or an unused row was changed", name, i);
            }
    } else {
        moe_ex(&batch, &layer, 0, x, ROUND_S, actual, logits, NULL);
        int nonzero = 0;
        for (int i = 0; i < ROUND_S * ROUND_D; i++) {
            CHECK(isfinite(actual[i]), "%s non-finite output %d", name, i);
            nonzero |= actual[i] != 0.f;
        }
        CHECK(nonzero, "%s fixture produced only zeros", name);
    }
    CHECK(cpu_pairs || !memcmp(actual, expected, sizeof(actual)),
          "%s batched outputs differ from single-row reference", name);
    for (int e = 0; e < pins; e++) {
        Slot *slot = slot_indexed(&batch, 0, e);
        CHECK(slot && slot->pinned && slot->filled,
              "%s lost pinned expert %d", name, e);
    }
    printf("inkling routed rounds: %s scalar agreement, pins=%d/%d cap=%d\n",
           name, pins, pin_limit, cap);
    fflush(stdout);
    free_round_model(&batch); free_round_model(&scalar); fclose(weights);
}

int main(int argc, char **argv){
    if (argc == 2 && !strcmp(argv[1], "cpu-pairs")) {
        check_pinned_rounds("CPU fallback misses", 10, 2, 2, 0, 2, 1);
        check_pinned_rounds("CPU fallback masks", 10, 2, 2, 1, 2, 1);
        return failures ? 1 : 0;
    }
    Model m; init_cache(&m); LCache *lc=&m.cache[0];

    Slot *one=slot_acquire(&m,0,1); one->filled=1;
    Slot *two=slot_acquire(&m,0,2); two->filled=1;
    CHECK(lc->slot_by_expert[1]==0&&lc->slot_by_expert[2]==1,
          "initial publications are not indexed");
    CHECK(slot_find(&m,0,1)==one,"indexed hit did not return expert 1");

    /* Expert 1 was just touched, so expert 2 is the deterministic LRU victim. */
    Slot *three=slot_acquire(&m,0,3); three->filled=1;
    CHECK(three==two,"wrong LRU slot was reused");
    CHECK(lc->slot_by_expert[2]==-1&&slot_indexed(&m,0,2)==NULL,
          "evicted expert 2 kept a mapping");
    CHECK(slot_indexed(&m,0,1)==one&&slot_indexed(&m,0,3)==three,
          "slot reuse damaged a live or new mapping");

    /* Pinning does not change identity: the other slot must be reused. */
    one->pinned=1;
    Slot *four=slot_acquire(&m,0,4); four->filled=1;
    CHECK(four==three,"pinned slot was selected as victim");
    CHECK(slot_indexed(&m,0,1)==one&&slot_indexed(&m,0,3)==NULL&&
          slot_indexed(&m,0,4)==four,"pin/eviction index mismatch");

    /* Defense in depth: a corrupt entry must never serve another expert. */
    lc->slot_by_expert[5]=0;
    CHECK(slot_indexed(&m,0,5)==NULL,"stale entry served the wrong weights");
    CHECK(slot_indexed(&m,0,-1)==NULL&&slot_indexed(&m,0,6)==NULL,
          "out-of-range lookup was accepted");

    free_cache(&m);
    check_lookup_scaling(44);
    check_lookup_scaling(219);
    {
        double avail = mem_avail_bytes();
        double probe = compat_mem_available_gb();
        CHECK(avail > 0.0, "mem_avail_bytes returned 0; auto cap would be 16 experts/layer");
        double gb = avail / 1e9;
        double gap = gb > probe ? gb - probe : probe - gb;
        CHECK(gap < 0.25, "mem_avail_bytes diverged from the shared probe (%.3f vs %.3f GB)", gb, probe);
    }
    check_pinned_rounds("no pins", 10, 0, 0, 0, 1, 0);
    check_pinned_rounds("one-slot cache", 1, 0, 0, 1, 1, 0);
    check_pinned_rounds("pinned misses", 10, 2, 2, 0, 1, 0);
    check_pinned_rounds("pinned hits and repeated experts", 10, 2, 2, 1, 1, 0);
    check_pinned_rounds("sparse pin history", 12, 4, 2, 1, 1, 0);
    check_pinned_rounds("top-k two", 10, 2, 2, 1, 2, 0);
    check_pinned_rounds("CPU fallback misses", 10, 2, 2, 0, 2, 1);
    check_pinned_rounds("CPU fallback masks", 10, 2, 2, 1, 2, 1);
    g_topp = 0.5f;
    check_pinned_rounds("top-p trimmed pairs", 10, 2, 2, 1, 2, 0);
    g_topp = 0.f;
    if(failures){ fprintf(stderr,"inkling cache index: %d failure(s)\n",failures); return 1; }
    puts("inkling cache index: ok");
    return 0;
}
