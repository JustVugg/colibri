/* Model-free regression for Kimi K3's expert -> slot index and ws[] promotion.
 * Synthetic pointers prove that swapping a loaded working slot still recycles
 * the victim allocation; no checkpoint or substantial memory is involved. */
#define COLI_CACHE_INDEX_TEST 1
#define main kimi_k3_main_unused
#include "../kimi_k3.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

static void init_cache(Model *m){
    memset(m,0,sizeof(*m));
    m->c.n_layers=1; m->c.n_experts=6;
    m->ecache=calloc(1,sizeof(LCache));
    m->ecache[0].cap=2;
    m->ecache[0].s=calloc(2,sizeof(Slot));
    m->ecache[0].slot_by_expert=malloc(6*sizeof(int));
    for(int i=0;i<2;i++) m->ecache[0].s[i].eid=-1;
    for(int e=0;e<6;e++) m->ecache[0].slot_by_expert[e]=-1;
}

static void check_lookup_scaling(int cap){
    Model m; memset(&m,0,sizeof(m));
    m.c.n_layers=1; m.c.n_experts=cap;
    m.ecache=calloc(1,sizeof(LCache));
    LCache *lc=&m.ecache[0]; lc->cap=lc->n=cap;
    lc->s=calloc((size_t)cap,sizeof(Slot));
    lc->slot_by_expert=malloc((size_t)cap*sizeof(int));
    for(int i=0;i<cap;i++){ lc->s[i].eid=i; lc->slot_by_expert[i]=i; }
    g_slot_index_probes=0;
    CHECK(slot_indexed(&m,0,cap-1)==&lc->s[cap-1],
          "last-slot lookup failed at cap %d",cap);
    CHECK(g_slot_index_probes==1,
          "indexed lookup used %llu probes at cap %d, expected 1",
          (unsigned long long)g_slot_index_probes,cap);
    printf("kimi lookup probes: cap=%d indexed=%llu legacy-worst=%d\n",
           cap,(unsigned long long)g_slot_index_probes,cap);
    free(lc->slot_by_expert); free(lc->s); free(m.ecache);
}

int main(void){
    Model m; init_cache(&m); LCache *lc=&m.ecache[0];
    unsigned char a=1,b=2,c=3;

    m.ws[0]=(Slot){.eid=1,.base=&a,.buf=&a};
    Slot *dst=&lc->s[lc->n++]; cache_promote(&m,0,dst,&m.ws[0]);
    m.ws[0]=(Slot){.eid=2,.base=&b,.buf=&b};
    dst=&lc->s[lc->n++]; cache_promote(&m,0,dst,&m.ws[0]);
    CHECK(lc->slot_by_expert[1]==0&&lc->slot_by_expert[2]==1,
          "initial promotions are not indexed");
    uint64_t hits=m.hits;
    CHECK(slot_find(&m,0,1)==&lc->s[0]&&m.hits==hits+1,
          "indexed hit did not preserve Kimi hit accounting");

    /* Replace expert 2. Its allocation must move back to ws[] for reuse. */
    m.ws[0]=(Slot){.eid=3,.base=&c,.buf=&c};
    cache_promote(&m,0,&lc->s[1],&m.ws[0]);
    CHECK(lc->slot_by_expert[2]==-1&&slot_indexed(&m,0,2)==NULL,
          "evicted expert 2 kept a mapping");
    CHECK(slot_indexed(&m,0,1)==&lc->s[0]&&slot_indexed(&m,0,3)==&lc->s[1],
          "promotion damaged a live or new mapping");
    CHECK(m.ws[0].eid==2&&m.ws[0].base==&b&&m.ws[0].buf==&b,
          "promotion failed to recycle the victim working buffer");

    lc->slot_by_expert[4]=0;
    CHECK(slot_indexed(&m,0,4)==NULL,"stale entry served the wrong weights");
    CHECK(slot_indexed(&m,0,-1)==NULL&&slot_indexed(&m,0,6)==NULL,
          "out-of-range lookup was accepted");

    free(lc->slot_by_expert); free(lc->s); free(m.ecache);
    check_lookup_scaling(44);
    check_lookup_scaling(219);
    if(failures){ fprintf(stderr,"kimi cache index: %d failure(s)\n",failures); return 1; }
    puts("kimi cache index: ok");
    return 0;
}
