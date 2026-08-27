/* Model-free regression for GLM's pin + LRU expert indices, including the
 * negative PILOT reservation state.  No checkpoint or large allocation. */
#define COLI_CACHE_INDEX_TEST 1
#define main colibri_main_unused
#include "../colibri.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); failures++; } } while (0)

static void init_cache(Model *m,int experts,int cap){
    memset(m,0,sizeof(*m));
    m->c.n_layers=1; m->c.n_experts=experts;
    int nr=2;
    m->ecache=calloc((size_t)nr,sizeof(ESlot*));
    m->ecn=calloc((size_t)nr,sizeof(int));
    m->pin=calloc((size_t)nr,sizeof(ESlot*));
    m->npin=calloc((size_t)nr,sizeof(int));
    m->ecache_slot_by_expert=calloc((size_t)nr,sizeof(int*));
    m->pin_slot_by_expert=calloc((size_t)nr,sizeof(int*));
    m->ecache[0]=calloc((size_t)cap,sizeof(ESlot));
    m->pin[0]=calloc(2,sizeof(ESlot));
    m->ecap=cap;
    for(int l=0;l<nr;l++){
        m->ecache_slot_by_expert[l]=malloc((size_t)experts*sizeof(int));
        m->pin_slot_by_expert[l]=malloc((size_t)experts*sizeof(int));
        for(int e=0;e<experts;e++){
            m->ecache_slot_by_expert[l][e]=-1;
            m->pin_slot_by_expert[l][e]=-1;
        }
    }
}

static void free_cache(Model *m){
    for(int l=0;l<2;l++){
        free(m->ecache_slot_by_expert[l]);
        free(m->pin_slot_by_expert[l]);
    }
    free(m->ecache[0]); free(m->pin[0]);
    free(m->ecache); free(m->ecn); free(m->pin); free(m->npin);
    free(m->ecache_slot_by_expert); free(m->pin_slot_by_expert);
}

static void check_lookup_scaling(int cap){
    Model m; init_cache(&m,cap,cap); m.ecn[0]=cap;
    for(int i=0;i<cap;i++) ecache_publish(&m,0,&m.ecache[0][i],i);
    g_glm_slot_index_probes=0;
    CHECK(ecache_indexed(&m,0,cap-1,0)==&m.ecache[0][cap-1],
          "last-slot lookup failed at cap %d",cap);
    CHECK(g_glm_slot_index_probes==1,
          "indexed lookup used %llu probes at cap %d, expected 1",
          (unsigned long long)g_glm_slot_index_probes,cap);
    printf("glm lookup probes: cap=%d indexed=%llu legacy-worst=%d\n",
           cap,(unsigned long long)g_glm_slot_index_probes,cap);
    free_cache(&m);
}

int main(void){
    Model m; init_cache(&m,6,2); m.ecn[0]=2;
    ecache_publish(&m,0,&m.ecache[0][0],1);
    ecache_publish(&m,0,&m.ecache[0][1],2);
    CHECK(ecache_indexed(&m,0,1,0)==&m.ecache[0][0]&&
          ecache_indexed(&m,0,2,0)==&m.ecache[0][1],"initial LRU publication mismatch");

    ecache_reserve(&m,0,&m.ecache[0][1],3);
    CHECK(m.ecache_slot_by_expert[0][2]==-1,"victim mapping survived reservation");
    CHECK(ecache_indexed(&m,0,3,0)==NULL&&
          ecache_indexed(&m,0,3,1)==&m.ecache[0][1],
          "PILOT reservation visibility is wrong");
    ecache_publish(&m,0,&m.ecache[0][1],3);
    CHECK(ecache_indexed(&m,0,3,0)==&m.ecache[0][1],"reservation did not publish");
    ecache_hide(&m,0,&m.ecache[0][1]);
    CHECK(m.ecache_slot_by_expert[0][3]==-1,"hidden slot retained its mapping");

    m.npin[0]=2;
    m.pin[0][0].eid=4; pin_index(&m,0,&m.pin[0][0]);
    m.pin[0][1].eid=5; pin_index(&m,0,&m.pin[0][1]);
    CHECK(pin_indexed(&m,0,5)==&m.pin[0][1]&&expert_is_resident(&m,0,4),
          "hot-store index lookup failed");
    pin_unindex(&m,0,&m.pin[0][0]);
    CHECK(m.pin_slot_by_expert[0][4]==-1,"REPIN unindex left the old identity");
    m.pin[0][0].eid=2; pin_index(&m,0,&m.pin[0][0]);
    CHECK(pin_indexed(&m,0,2)==&m.pin[0][0],"REPIN publication did not index the new identity");

    m.ecache_slot_by_expert[0][0]=0;
    CHECK(ecache_indexed(&m,0,0,0)==NULL,"stale LRU entry served wrong weights");
    m.pin_slot_by_expert[0][0]=0;
    CHECK(pin_indexed(&m,0,0)==NULL,"stale pin entry served wrong weights");
    CHECK(ecache_indexed(&m,0,-1,0)==NULL&&pin_indexed(&m,0,6)==NULL,
          "out-of-range lookup was accepted");

    free_cache(&m);
    check_lookup_scaling(44);
    check_lookup_scaling(219);
    if(failures){ fprintf(stderr,"glm cache index: %d failure(s)\n",failures); return 1; }
    puts("glm cache index: ok");
    return 0;
}
