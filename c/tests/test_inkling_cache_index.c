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

int main(void){
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
    if(failures){ fprintf(stderr,"inkling cache index: %d failure(s)\n",failures); return 1; }
    puts("inkling cache index: ok");
    return 0;
}
