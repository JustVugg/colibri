/* CPU-only reservation/reader fixture. No model, GPU, checkpoint, or disk I/O.
 * Build from the repository root:
 * clang -O2 -std=c11 -pthread c/tests/test_qwen36_layer_loader.c -o /tmp/qwen-layer-loader
 * The fixture models terminal GPU completion as a caller event. It does not
 * test actual Metal completion, row-window numerics, or runtime graph wiring.
 */
#define _POSIX_C_SOURCE 200809L
#include "../qwen36_layer_loader.h"
#include <stdio.h>
#include <unistd.h>

enum { KEYS=1024, SLOTS=303, GROUP=8, CALLBACK_SCRATCH=64 };
typedef struct { uint64_t magic;int key;unsigned char bytes[116]; } Payload;
typedef struct {
    QwenGlobalCache cache;
    QwenLayerLoader loader;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    unsigned allocations,allocated,released,loads,advice,many,active,peak_active;
    unsigned loads_by_key[KEYS];
    int fail_slot,fail_key,block[KEYS],entered[KEYS],allowed[KEYS];
    int block_advice_key,advice_entered[KEYS],advice_allowed[KEYS],advised[KEYS];
    int require_own_hint;
    int finish_done;
} Fixture;
static int checks,failures;
#define CHECK(c,...) do { __atomic_fetch_add(&checks,1,__ATOMIC_RELAXED); if(!(c)) { \
    __atomic_fetch_add(&failures,1,__ATOMIC_RELAXED); \
    fprintf(stderr,"FAIL line %d: ",__LINE__);fprintf(stderr,__VA_ARGS__);fputc('\n',stderr); } } while(0)
#define REQUIRE(c,...) do { if(!(c)) { CHECK(0,__VA_ARGS__);exit(2); } } while(0)
static struct timespec deadline(void) {
    struct timespec t;clock_gettime(CLOCK_REALTIME,&t);t.tv_sec+=10;return t;
}
static void fill(Payload *p,int key) {
    p->key=key;for(size_t i=0;i<sizeof(p->bytes);i++)p->bytes[i]=(unsigned char)(key*17+(int)i);
}
static void verify(void *payload,int key) {
    Payload *p=payload;CHECK(p && p->magic==UINT64_C(0x1a9e5cafe5312026),"invalid payload for %d",key);
    if(!p)return;
    CHECK(p->key==key,"expected key %d, got %d",key,p->key);
    for(size_t i=0;i<sizeof(p->bytes);i++)CHECK(p->bytes[i]==(unsigned char)(key*17+(int)i),"torn bytes key %d",key);
}
static void *allocate_payload(void *context,size_t slot) {
    Fixture *f=context;pthread_mutex_lock(&f->mutex);f->allocations++;
    int fail=(int)slot==f->fail_slot;pthread_mutex_unlock(&f->mutex);
    CHECK(slot<SLOTS,"allocation outside the existing payload pool");if(fail)return NULL;
    Payload *p=calloc(1,sizeof(*p));REQUIRE(p,"tiny fixture allocation failed");
    p->magic=UINT64_C(0x1a9e5cafe5312026);p->key=-1;
    pthread_mutex_lock(&f->mutex);f->allocated++;pthread_mutex_unlock(&f->mutex);return p;
}
static int load_payload(void *context,int key,void *payload) {
    Fixture *f=context;pthread_mutex_lock(&f->mutex);
    f->loads++;f->loads_by_key[key]++;f->active++;if(f->active>f->peak_active)f->peak_active=f->active;
    if(f->require_own_hint)CHECK(f->advised[key],"read began before its own hint completed");
    f->entered[key]=1;pthread_cond_broadcast(&f->changed);struct timespec limit=deadline();
    while(f->block[key] && !f->allowed[key])
        REQUIRE(!pthread_cond_timedwait(&f->changed,&f->mutex,&limit),"load callback gate timed out");
    int fail=key==f->fail_key;pthread_mutex_unlock(&f->mutex);
    fill(payload,key); /* failure also dirties the destination: it must be hidden. */
    pthread_mutex_lock(&f->mutex);f->active--;pthread_cond_broadcast(&f->changed);pthread_mutex_unlock(&f->mutex);
    return !fail;
}
static void release_payload(void *context,void *payload) {
    Fixture *f=context;Payload *p=payload;
    CHECK(p->magic==UINT64_C(0x1a9e5cafe5312026),"double release or stale payload");p->magic=0;
    pthread_mutex_lock(&f->mutex);f->released++;pthread_mutex_unlock(&f->mutex);free(p);
}
static void advice_payload(void *context,int key) {
    Fixture *f=context;pthread_mutex_lock(&f->mutex);f->advice_entered[key]=1;
    pthread_cond_broadcast(&f->changed);struct timespec limit=deadline();
    while(key==f->block_advice_key && !f->advice_allowed[key])
        REQUIRE(!pthread_cond_timedwait(&f->changed,&f->mutex,&limit),"advice callback gate timed out");
    f->advised[key]=1;f->advice++;pthread_mutex_unlock(&f->mutex);
}
static int load_many_payload(void *context,const int *keys,void *const *payloads,unsigned n) {
    Fixture *f=context;pthread_mutex_lock(&f->mutex);f->many++;pthread_mutex_unlock(&f->mutex);
    int ok=1;for(unsigned i=0;i<n;i++)ok=load_payload(context,keys[i],payloads[i]) && ok;return ok;
}
static void fixture_init(Fixture *f,int initialize_loader) {
    memset(f,0,sizeof(*f));f->fail_slot=f->fail_key=f->block_advice_key=-1;
    REQUIRE(!pthread_mutex_init(&f->mutex,NULL) && !pthread_cond_init(&f->changed,NULL),"fixture synchronization init");
    REQUIRE(qgc_init(&f->cache,KEYS,SLOTS,GROUP,sizeof(Payload),CALLBACK_SCRATCH,64,f,
                    allocate_payload,load_payload,release_payload),"cache init");
    /* No advise callback: reader-only, unless a case sets one (set_advice). */
    f->cache.load_many=load_many_payload;
    if(initialize_loader)REQUIRE(qll_init(&f->loader,&f->cache),"layer loader init");
}
static void set_advice(Fixture *f) {
    pthread_mutex_lock(&f->cache.mutex);f->cache.advise=advice_payload;pthread_mutex_unlock(&f->cache.mutex);
}
static void ledger(Fixture *f,int idle) {
    QgcStats s=qgc_snapshot(&f->cache);QllStats t=qll_snapshot(&f->loader);
    CHECK(s.allocated+s.reserved<=SLOTS*sizeof(Payload) && s.peak_charge<=SLOTS*sizeof(Payload),"payload budget exceeded");
    CHECK(s.scratch<=QLL_READERS*CALLBACK_SCRATCH && s.peak_scratch<=QLL_READERS*CALLBACK_SCRATCH,"callback scratch bound exceeded");
    CHECK(t.peak_readers<=QLL_READERS && t.peak_selected<=QLL_MAX_KEYS,"bounded readers/union exceeded");
    CHECK(t.metadata_bytes==sizeof(QwenLayerLoader) && t.stack_bytes>=QLL_READERS*QLL_STACK_BYTES && (t.guard_bytes || QGC_SYSTEM_GUARD),"worker metadata/stacks not accounted");
    if(idle)CHECK(!s.active && !s.loading && !s.borrowed && !s.reserved && !s.scratch && !t.held && !t.reading && !t.queued && !t.advising,"reservation did not drain");
}
static void fixture_done(Fixture *f) {
    ledger(f,1);REQUIRE(qll_destroy(&f->loader),"loader destroy");
    REQUIRE(qll_destroy(&f->loader),"second loader destroy");
    REQUIRE(qgc_destroy(&f->cache),"cache destroy after reader join");
    CHECK(f->allocated==f->released,"payload lifetime leak: allocated=%u released=%u",f->allocated,f->released);
    CHECK(!f->active,"callback survived cache destroy");
    pthread_cond_destroy(&f->changed);pthread_mutex_destroy(&f->mutex);
}
static void keys_range(int *keys,int first,unsigned n) {for(unsigned i=0;i<n;i++)keys[i]=first+(int)i;}
static void borrow(Fixture *f,const int *keys,unsigned n) {
    void *payloads[GROUP]={0};REQUIRE(qll_wait(&f->loader,keys,n,payloads),"READY group borrow failed");
    for(unsigned i=0;i<n;i++){verify(payloads[i],keys[i]);for(unsigned j=0;j<i;j++)if(keys[j]==keys[i])CHECK(payloads[j]==payloads[i],"duplicate did not alias");}
}
static void entered(Fixture *f,int key) {
    struct timespec limit=deadline();pthread_mutex_lock(&f->mutex);
    while(!f->entered[key])REQUIRE(!pthread_cond_timedwait(&f->changed,&f->mutex,&limit),"key %d did not enter callback",key);
    pthread_mutex_unlock(&f->mutex);
}
static void allow(Fixture *f,int key) {
    pthread_mutex_lock(&f->mutex);f->allowed[key]=1;pthread_cond_broadcast(&f->changed);pthread_mutex_unlock(&f->mutex);
}
static void advice_entered(Fixture *f,int key) {
    struct timespec limit=deadline();pthread_mutex_lock(&f->mutex);
    while(!f->advice_entered[key])REQUIRE(!pthread_cond_timedwait(&f->changed,&f->mutex,&limit),"key %d did not enter advice callback",key);
    pthread_mutex_unlock(&f->mutex);
}
static void advice_allow(Fixture *f,int key,int next_blocked_key) {
    pthread_mutex_lock(&f->mutex);f->advice_allowed[key]=1;f->block_advice_key=next_blocked_key;
    pthread_cond_broadcast(&f->changed);pthread_mutex_unlock(&f->mutex);
}
static void seed(Fixture *f,int slot,int key,uint64_t used) {
    void *payload=allocate_payload(f,(size_t)slot);REQUIRE(payload,"seed allocation failed");fill(payload,key);
    pthread_mutex_lock(&f->cache.mutex);QgcEntry *e=&f->cache.entries[slot];
    REQUIRE(!e->payload && e->state==QGC_EMPTY,"seed destination occupied");
    *e=(QgcEntry){.key=key,.state=QGC_READY,.used=used,.payload=payload};f->cache.index[key]=slot;
    if(used>f->cache.clock)f->cache.clock=used;
    f->cache.stats.allocated+=sizeof(Payload);f->cache.stats.peak_charge=f->cache.stats.allocated;
    pthread_mutex_unlock(&f->cache.mutex);
}

static void all_keys_duplicate_and_budget(void) {
    Fixture f;fixture_init(&f,1);int keys[QLL_MAX_KEYS];keys_range(keys,0,QLL_MAX_KEYS);
    REQUIRE(qll_begin(&f.loader),"begin empty cache");
    CHECK(!qll_begin(&f.loader),"nested begin accepted");
    REQUIRE(qll_append(&f.loader,keys,128),"first discovery prefix");
    REQUIRE(qll_append(&f.loader,keys,128),"duplicate discovery prefix");
    REQUIRE(qll_append(&f.loader,keys+128,128),"second discovery prefix");
    for(unsigned i=0;i<QLL_MAX_KEYS;i+=GROUP)borrow(&f,keys+i,GROUP);
    int duplicates[GROUP]={0,0,1,1,2,2,3,3};borrow(&f,duplicates,GROUP);
    QllStats s=qll_snapshot(&f.loader);QgcStats c=qgc_snapshot(&f.cache);
    CHECK(s.selected==256 && s.misses==256 && s.hits==0 && s.duplicate_keys==128,"dedup counts");
    CHECK(s.load_calls==256 && s.load_bytes==256*sizeof(Payload) && c.borrowed==256,"physical calls/bytes or pins");
    CHECK(!f.advice && !f.many,"layer path invoked advisory callback or nested load pool");
    int extra=300,invalid[2]={400,-1};
    CHECK(!qll_append(&f.loader,&extra,1) && !qll_append(&f.loader,invalid,2),"invalid/oversize append accepted");
    CHECK(qll_snapshot(&f.loader).selected==256,"invalid append changed reservation");
    QwenLayerLoader copied=f.loader;
    CHECK(!qll_finish(&copied,1) && !qll_begin(&copied),"copied reservation acquired ownership");
    CHECK(qgc_snapshot(&f.cache).borrowed==256,"copied owner damaged pins");
    REQUIRE(qll_finish(&f.loader,0),"finish full union");ledger(&f,1);
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,keys,256),"second full union");
    for(unsigned i=0;i<256;i+=GROUP)borrow(&f,keys+i,GROUP);
    s=qll_snapshot(&f.loader);CHECK(s.load_calls==256 && s.hits==256,"repeated union reread resident records");
    REQUIRE(qll_finish(&f.loader,0),"finish repeated union");fixture_done(&f);
}

typedef struct {Fixture *f;int result,done;QgcLease lease;} Foreign;
static void *wrong_owner(void *argument) {
    Foreign *p=argument;Fixture *f=p->f;int key=0;void *payload=NULL;
    CHECK(!qll_append(&f->loader,&key,1),"foreign append accepted");
    CHECK(!qll_wait(&f->loader,&key,1,&payload),"foreign wait exposed payload");
    CHECK(!qll_ready(&f->loader,&key,1),"foreign ready check answered");
    CHECK(!qll_finish(&f->loader,1),"foreign finish released pins");
    CHECK(!qll_destroy(&f->loader),"foreign destroy joined active readers");return NULL;
}
static void *ordinary_waiter(void *argument) {
    Foreign *p=argument;int key=700;
    p->result=qgc_acquire(&p->f->cache,&key,1,NULL,&p->lease);
    if(p->result){verify(qgc_payload(&p->f->cache,&p->lease,0),key);CHECK(qgc_release_group(&p->f->cache,&p->lease),"ordinary waiter release");}
    __atomic_store_n(&p->done,1,__ATOMIC_RELEASE);return NULL;
}
static void wait_cache_state(Fixture *f,int want_stopping,unsigned want_waiters) {
    uint64_t until=qgc_now_ns()+UINT64_C(10000000000);
    for(;;) {
        pthread_mutex_lock(&f->cache.mutex);
        int ready=(!want_stopping || f->cache.stopping) && f->cache.stats.waiters==want_waiters;
        pthread_mutex_unlock(&f->cache.mutex);if(ready)return;
        REQUIRE(qgc_now_ns()<until,"cache waiter/stop state not reached");
        struct timespec pause={0,1000000};nanosleep(&pause,NULL);
    }
}
static void individual_ready_owner_and_fifo(void) {
    Fixture f;fixture_init(&f,1);f.block[1]=1;int keys[2]={0,1};
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,keys,2),"two-key reservation");entered(&f,1);
    void *borrowed=NULL;REQUIRE(qll_wait(&f.loader,keys,1,&borrowed),"ready key waited for unrelated callback");verify(borrowed,0);
    int absent=300;
    CHECK(qll_ready(&f.loader,keys,1),"loaded key not ready");
    CHECK(!qll_ready(&f.loader,keys,2),"group with a loading key reported ready");
    CHECK(!qll_ready(&f.loader,&absent,1),"key never appended reported ready");
    CHECK(!qgc_contains(&f.cache,1) && !qgc_peek(&f.cache,1),"incomplete payload published");
    pthread_mutex_lock(&f.cache.mutex);int i=qll_find_locked(&f.loader,1);
    CHECK(qll_job_matches_locked(&f.loader,(unsigned)i,f.loader.generation),"active job identity failed");
    CHECK(!qll_job_matches_locked(&f.loader,(unsigned)i,f.loader.generation+1),"stale generation accepted");
    pthread_mutex_unlock(&f.cache.mutex);
    Foreign foreign={.f=&f};pthread_t thread;REQUIRE(!pthread_create(&thread,NULL,wrong_owner,&foreign),"foreign thread");pthread_join(thread,NULL);
    QgcLease nested={0};CHECK(!qgc_acquire(&f.cache,keys,1,NULL,&nested),"ordinary nested lease accepted");
    CHECK(!qgc_prefetch(&f.cache,702),"prefetch displaced an active reservation");
    Foreign waiter={.f=&f};REQUIRE(!pthread_create(&thread,NULL,ordinary_waiter,&waiter),"ordinary waiter thread");wait_cache_state(&f,0,1);
    CHECK(!__atomic_load_n(&waiter.done,__ATOMIC_ACQUIRE),"ordinary acquire crossed layer ownership");
    int more[254];keys_range(more,2,254);REQUIRE(qll_append(&f.loader,more,254),"append after first GPU borrow");
    borrow(&f,more,GROUP);verify(borrowed,0);CHECK(qgc_snapshot(&f.cache).borrowed==256,"borrow pins not held across append");
    allow(&f,1);REQUIRE(qll_finish(&f.loader,0),"finish owner after GPU terminal");
    pthread_join(thread,NULL);CHECK(waiter.result==1,"ordinary waiter did not resume");fixture_done(&f);
}

static void hit_protection_forced_retention_and_unseen_eviction(void) {
    Fixture f;fixture_init(&f,1);for(int i=0;i<SLOTS;i++)seed(&f,i,i,(uint64_t)i+1);
    REQUIRE(qll_begin(&f.loader),"begin populated cache");
    int pair[2]={400,0};REQUIRE(qll_append(&f.loader,pair,2),"miss plus oldest hit");borrow(&f,pair,2);
    CHECK(f.cache.index[0]==0 && f.cache.index[400]==1,"append did not pin all its hits before choosing victims");
    REQUIRE(qll_finish(&f.loader,0),"finish protected hit");
    /* Start from a known full cache again to isolate the largest union. */
    fixture_done(&f);fixture_init(&f,1);for(int i=0;i<SLOTS;i++)seed(&f,i,i,(uint64_t)i+1);
    int keys[256];for(int i=0;i<8;i++)keys[i]=295+i;keys_range(keys+8,400,248);
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,keys,256),"full working-hit plus retained-miss union");
    for(unsigned i=0;i<256;i+=GROUP)borrow(&f,keys+i,GROUP);
    QllStats s=qll_snapshot(&f.loader);
    CHECK(s.hits==8 && s.misses==248 && s.forced_admissions>0 && s.evictions==248,"forced admission accounting");
    for(int i=0;i<8;i++)CHECK(f.cache.index[295+i]==295+i,"working hit moved or evicted");
    CHECK(qgc_snapshot(&f.cache).allocated==SLOTS*sizeof(Payload) && f.allocated==SLOTS,"full pool allocated an extra payload");
    REQUIRE(qll_finish(&f.loader,0),"finish all pins");fixture_done(&f);
    fixture_init(&f,1);for(int i=0;i<SLOTS;i++)seed(&f,i,i,(uint64_t)i+1);
    int new_key=400,late_key=0;
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,&new_key,1),"first discovery miss");borrow(&f,&new_key,1);
    CHECK(!qgc_contains(&f.cache,late_key),"fixture did not exercise unseen-hit eviction");
    REQUIRE(qll_append(&f.loader,&late_key,1),"later discovery of evicted hit");borrow(&f,&late_key,1);
    s=qll_snapshot(&f.loader);CHECK(s.misses==2 && s.load_calls==2,"unseen hit reload hidden from accounting");
    REQUIRE(qll_finish(&f.loader,0),"finish late discovery");fixture_done(&f);
}

static void *cancel_observer(void *argument) {
    Fixture *f=argument;QwenGlobalCache *c=&f->cache;struct timespec limit=deadline();
    pthread_mutex_lock(&c->mutex);
    while(!f->loader.cancelled)
        REQUIRE(!pthread_cond_timedwait(&c->changed,&c->mutex,&limit),"finish did not reach cancellation drain");
    CHECK(f->loader.reading==4 && c->active && c->entries[0].refs==1,"cancel released payload before callback terminal");
    CHECK(!__atomic_load_n(&f->finish_done,__ATOMIC_ACQUIRE),"finish returned before blocked readers completed");
    CHECK(c->stats.reserved==4*sizeof(Payload),"queued fresh reservations not uncharged on cancel");
    pthread_mutex_unlock(&c->mutex);
    for(int key=1;key<=4;key++)allow(f,key);return NULL;
}
static void cancellation_drains_readers_and_borrows(void) {
    Fixture f;fixture_init(&f,1);int ready=0;
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,&ready,1),"initial ready borrow");
    void *borrowed=NULL;REQUIRE(qll_wait(&f.loader,&ready,1,&borrowed),"initial ready pointer");
    for(int key=1;key<=12;key++)f.block[key]=1;
    int keys[12];keys_range(keys,1,12);REQUIRE(qll_append(&f.loader,keys,12),"queued cancellation records");
    for(int key=1;key<=4;key++)entered(&f,key);
    verify(borrowed,0); /* Simulated GPU work completes before the owner finish. */
    pthread_t thread;REQUIRE(!pthread_create(&thread,NULL,cancel_observer,&f),"cancellation observer");
    REQUIRE(qll_finish(&f.loader,1),"cancel finish");__atomic_store_n(&f.finish_done,1,__ATOMIC_RELEASE);pthread_join(thread,NULL);
    QllStats s=qll_snapshot(&f.loader);
    CHECK(s.cancelled_records==8 && s.completed_after_cancel==4 && s.load_calls==5,"cancel physical calls/completions accounting");
    CHECK(s.peak_readers==4 && f.peak_active==4,"four dedicated readers not exercised");
    CHECK(!f.released,"cancel freed reusable cache payloads");fixture_done(&f);
}

static void failures_hide_partial_records(void) {
    Fixture f;fixture_init(&f,1);f.fail_slot=0;int key=0;void *payload=(void *)1;
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,&key,1),"allocation failure admission");
    CHECK(!qll_wait(&f.loader,&key,1,&payload) && !payload,"allocation failure exposed a payload");
    REQUIRE(qll_finish(&f.loader,1),"allocation failure drain");QllStats s=qll_snapshot(&f.loader);
    CHECK(s.allocation_failures==1 && !s.load_calls && s.load_failures==1,"allocation failure counted as physical read");
    CHECK(!qgc_contains(&f.cache,key) && !f.allocated,"allocation failure leaked charged bytes");fixture_done(&f);
    fixture_init(&f,1);int ready=0;
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,&ready,1),"pre-failure ready record");borrow(&f,&ready,1);
    for(int k=1;k<=16;k++)f.block[k]=1;f.fail_key=1;
    int keys[16];keys_range(keys,1,16);REQUIRE(qll_append(&f.loader,keys,16),"partial failure group");
    for(int k=1;k<=4;k++)entered(&f,k);allow(&f,1);
    payload=(void *)1;CHECK(!qll_wait(&f.loader,keys,1,&payload) && !payload,"failed dirty record was published");
    CHECK(!qgc_contains(&f.cache,1),"failed load remains READY");
    CHECK(qgc_snapshot(&f.cache).active && qgc_snapshot(&f.cache).borrowed==17,"error released GPU borrows before explicit terminal drain");
    for(int k=2;k<=4;k++)allow(&f,k);
    REQUIRE(qll_finish(&f.loader,1),"partial failure drain");s=qll_snapshot(&f.loader);
    CHECK(s.load_failures==1 && s.cancelled_records==12 && s.load_calls==5,"partial failure traffic/cancel accounting");
    CHECK(qgc_contains(&f.cache,0),"independent completed record unnecessarily invalidated");fixture_done(&f);
}

static void *stop_cache(void *argument) {
    Foreign *p=argument;p->result=qgc_stop(&p->f->cache);__atomic_store_n(&p->done,1,__ATOMIC_RELEASE);return NULL;
}
static void init_under_ordinary_lease_and_shutdown(void) {
    Fixture f;fixture_init(&f,0);int key=500;QgcLease lease={0};
    REQUIRE(qgc_acquire(&f.cache,&key,1,NULL,&lease),"ordinary lease before lazy init");
    REQUIRE(qll_init(&f.loader,&f.cache),"lazy init rejected valid active cache");
    CHECK(!qll_begin(&f.loader),"begin crossed previous ordinary lease");
    REQUIRE(qgc_release_group(&f.cache,&lease),"release ordinary lease before layer");
    key=0;f.block[key]=1;REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,&key,1),"shutdown reservation");entered(&f,key);
    Foreign stop={.f=&f};pthread_t thread;REQUIRE(!pthread_create(&thread,NULL,stop_cache,&stop),"cache stop thread");wait_cache_state(&f,1,0);
    CHECK(!__atomic_load_n(&stop.done,__ATOMIC_ACQUIRE),"cache stop crossed live callback/reservation");
    void *payload=(void *)1;CHECK(!qll_wait(&f.loader,&key,1,&payload) && !payload,"stopping cache exposed a new borrow");
    allow(&f,key);REQUIRE(qll_finish(&f.loader,1),"owner shutdown drain");pthread_join(thread,NULL);
    CHECK(stop.result && !qll_begin(&f.loader),"cache stop or admission stop contract");fixture_done(&f);
}

typedef struct {Fixture *f;int first,last,reverse;} ReleaseOrder;
static void *release_in_order(void *argument) {
    ReleaseOrder *r=argument;
    for(int t=r->first;t<=r->last;t++) {
        int key=r->reverse?r->last-(t-r->first):t;allow(r->f,key);
        /* Wait for actual READY publication, not merely callback entry. */
        QwenGlobalCache *c=&r->f->cache;pthread_mutex_lock(&c->mutex);struct timespec limit=deadline();
        while(c->index[key]<0 || c->entries[c->index[key]].state!=QGC_READY)
            REQUIRE(!pthread_cond_timedwait(&c->changed,&c->mutex,&limit),"ordered READY publication timed out");
        pthread_mutex_unlock(&c->mutex);
    }
    return NULL;
}
static void completion_order_does_not_change_policy(void) {
    Fixture a,b;fixture_init(&a,1);fixture_init(&b,1);int keys[4]={0,1,2,3};
    Fixture *runs[2]={&a,&b};
    for(int run=0;run<2;run++) {
        Fixture *f=runs[run];for(int k=0;k<4;k++)f->block[k]=1;
        REQUIRE(qll_begin(&f->loader) && qll_append(&f->loader,keys,4),"ordered completion reservation");
        for(int k=0;k<4;k++)entered(f,k);
        ReleaseOrder order={f,0,3,run};pthread_t thread;
        REQUIRE(!pthread_create(&thread,NULL,release_in_order,&order),"completion ordering thread");
        borrow(f,keys,4);pthread_join(thread,NULL);REQUIRE(qll_finish(&f->loader,0),"ordered finish");
    }
    CHECK(a.cache.clock==b.cache.clock && a.cache.policy.references==b.cache.policy.references && a.cache.policy.epoch==b.cache.policy.epoch,"completion order changed history clock");
    CHECK(!memcmp(a.cache.policy.history,b.cache.policy.history,KEYS*sizeof(QwenCacheHistory)),"completion order changed policy history");
    CHECK(!memcmp(a.cache.index,b.cache.index,KEYS*sizeof(int)),"completion order changed residency index");
    for(int i=0;i<SLOTS;i++)CHECK(a.cache.entries[i].key==b.cache.entries[i].key && a.cache.entries[i].used==b.cache.entries[i].used,"completion order changed victim metadata");
    fixture_done(&a);fixture_done(&b);
}
static void advice_is_bounded_and_does_not_block_ready_groups(void) {
    Fixture f;fixture_init(&f,1);seed(&f,0,900,1);
    set_advice(&f);
    f.block_advice_key=0;f.require_own_hint=1;
    int first[9]={0,1,2,3,4,5,6,7,900},next[8];keys_range(next,8,8);
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,first,9),"first advised batch");advice_entered(&f,0);
    CHECK(!f.loads,"read started before its first hint completed");
    /* This append and READY borrow run while an advice callback is blocked,
     * proving neither the main thread nor cache mutex is held by the hint. */
    REQUIRE(qll_append(&f.loader,next,8),"append while background advice is blocked");borrow(&f,first+8,1);
    advice_allow(&f,0,1);advice_entered(&f,1);
    /* The second hint of the SAME captured eight-record batch is blocked.
     * The first record must be readable now, before that batch can finish:
     * publishing a batch only at its end would not pass. */
    borrow(&f,first,1);
    QllStats early=qll_snapshot(&f.loader);
    CHECK(early.advising && early.advise_calls==1 && early.load_calls==1,
          "first READY record did not overlap the rest of its advice batch");
    CHECK(!qgc_contains(&f.cache,1) && !qgc_contains(&f.cache,2),"unpublished advice prefix exposed payloads");
    advice_allow(&f,1,8);advice_entered(&f,8);
    borrow(&f,first,8);
    QllStats s=qll_snapshot(&f.loader);CHECK(s.advising && s.load_calls>=8,"prior batch readers stopped behind later advice batch");
    CHECK(!qgc_contains(&f.cache,8),"not-yet-announced batch became READY");
    advice_allow(&f,8,-1);borrow(&f,next,8);REQUIRE(qll_finish(&f.loader,0),"advised layer finish");
    s=qll_snapshot(&f.loader);
    CHECK(s.advice_batches==2 && s.advise_calls==16 && s.load_calls==16 && s.hits==1,"advice accounting counted hits, duplicates or missing records");
    CHECK(!f.advised[900] && !f.many,"resident key hinted or nested load pool used");fixture_done(&f);
}
static void *cancel_advice_observer(void *argument) {
    Fixture *f=argument;QwenGlobalCache *c=&f->cache;struct timespec limit=deadline();
    pthread_mutex_lock(&c->mutex);
    while(!f->loader.cancelled)
        REQUIRE(!pthread_cond_timedwait(&c->changed,&c->mutex,&limit),"finish did not reach active advice drain");
    CHECK(f->loader.advising && !f->loader.reading && c->active,"active advice lost its reservation");
    CHECK(!c->stats.reserved && c->entries[0].refs==1,"cancelled hints leaked reservations or released pins before drain");
    CHECK(!__atomic_load_n(&f->finish_done,__ATOMIC_ACQUIRE),"finish returned before advice callback terminal");
    pthread_mutex_unlock(&c->mutex);advice_allow(f,0,-1);return NULL;
}
static void cancellation_drains_active_advice(void) {
    Fixture f;fixture_init(&f,1);set_advice(&f);
    f.block_advice_key=0;int keys[8];keys_range(keys,0,8);
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,keys,8),"cancellable advice plan");advice_entered(&f,0);
    pthread_t thread;REQUIRE(!pthread_create(&thread,NULL,cancel_advice_observer,&f),"advice cancellation observer");
    REQUIRE(qll_finish(&f.loader,1),"advice cancellation drain");__atomic_store_n(&f.finish_done,1,__ATOMIC_RELEASE);pthread_join(thread,NULL);
    QllStats s=qll_snapshot(&f.loader);
    CHECK(s.advise_calls==1 && !s.load_calls && s.cancelled_records==8 && !f.allocated,"cancelled advice started extra hints/reads or allocated payloads");
    fixture_done(&f);
}
static void *cancel_progress_observer(void *argument) {
    Fixture *f=argument;QwenGlobalCache *c=&f->cache;struct timespec limit=deadline();
    pthread_mutex_lock(&c->mutex);
    while(!f->loader.cancelled)
        REQUIRE(!pthread_cond_timedwait(&c->changed,&c->mutex,&limit),"progressive cancel did not reach drain");
    CHECK(f->loader.advising && f->loader.reading==1 && c->active,
          "progressive cancel lost advisor or reader ownership");
    CHECK(c->stats.reserved==sizeof(Payload) && c->entries[0].refs==1,
          "progressive cancel uncharged a running read or released its pin");
    CHECK(!__atomic_load_n(&f->finish_done,__ATOMIC_ACQUIRE),"finish crossed active reader/advisor");
    pthread_mutex_unlock(&c->mutex);allow(f,0);
    pthread_mutex_lock(&c->mutex);limit=deadline();
    while(f->loader.reading)
        REQUIRE(!pthread_cond_timedwait(&c->changed,&c->mutex,&limit),"cancelled read did not complete");
    CHECK(f->loader.advising && c->entries[0].refs==1 && c->entries[0].state==QGC_READY,
          "reader completion released the reservation before advisor terminal");
    CHECK(!__atomic_load_n(&f->finish_done,__ATOMIC_ACQUIRE),"finish ignored remaining advisor");
    pthread_mutex_unlock(&c->mutex);advice_allow(f,1,-1);return NULL;
}
static void cancellation_drains_progressive_reader_and_advisor(void) {
    Fixture f;fixture_init(&f,1);set_advice(&f);
    f.block_advice_key=1;f.block[0]=1;f.require_own_hint=1;
    int keys[8];keys_range(keys,0,8);
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,keys,8),"progressive cancellation reservation");
    advice_entered(&f,1);entered(&f,0);
    pthread_t thread;REQUIRE(!pthread_create(&thread,NULL,cancel_progress_observer,&f),"progressive cancellation observer");
    REQUIRE(qll_finish(&f.loader,1),"progressive cancellation drain");
    __atomic_store_n(&f.finish_done,1,__ATOMIC_RELEASE);pthread_join(thread,NULL);
    QllStats s=qll_snapshot(&f.loader);
    CHECK(s.advise_calls==2 && s.load_calls==1 && s.cancelled_records==7 && s.completed_after_cancel==1,
          "progressive cancellation performed extra callbacks or lost accounting");
    CHECK(f.allocated==1 && !f.released,"progressive cancellation leaked or released cache payloads");
    fixture_done(&f);
}
static void small_cache_is_refused(void) {
    Fixture f;fixture_init(&f,0);
    CHECK(qll_cache_eligible(&f.cache),"standard 303-slot cache is not eligible");
    QwenGlobalCache small;
    REQUIRE(qgc_init(&small,KEYS,151,GROUP,sizeof(Payload),CALLBACK_SCRATCH,64,&f,
                    allocate_payload,load_payload,release_payload),"small valid cache init");
    CHECK(!qll_cache_eligible(&small),"small valid cache did not request legacy fallback");
    CHECK(!qll_init(&f.loader,&small),"small cache initialized an unsupported reservation");
    REQUIRE(qgc_destroy(&small),"small cache cleanup");
    REQUIRE(qll_init(&f.loader,&f.cache) && qll_begin(&f.loader),"standard cache reservation");
    REQUIRE(qll_finish(&f.loader,0),"standard cache reservation finish");fixture_done(&f);
}
/* A layer's hit with the lowest demand score is pinned before any miss of
 * the same layer is placed: the miss takes another retained slot. */
static void layer_hits_are_pinned_before_misses(void) {
    Fixture f;fixture_init(&f,1);
    for(int s=0;s<SLOTS-GROUP;s++)seed(&f,s,s,(uint64_t)s+1);   /* every retained slot full: key s in slot s */
    pthread_mutex_lock(&f.cache.mutex);                          /* every key but 0 in demand */
    for(int k=1;k<SLOTS-GROUP;k++)f.cache.policy.history[k]=(QwenCacheHistory){f.cache.policy.epoch,100,0};
    pthread_mutex_unlock(&f.cache.mutex);
    int keys[2]={0,900};
    REQUIRE(qll_begin(&f.loader) && qll_append(&f.loader,keys,2),"a hit and a miss of one layer");
    borrow(&f,keys,2);REQUIRE(qll_finish(&f.loader,0),"pinned hit finish");
    CHECK(qgc_contains(&f.cache,0) && qgc_contains(&f.cache,900),"the pinned hit or its miss is not resident");
    fixture_done(&f);
}
/* A hang must fail, not sit forever in CI. MinGW has no alarm(), so a
 * detached thread ends the run instead (as test_qwen36_tier_shutdown.c does). */
static void *watchdog(void *arg) {
    (void)arg;struct timespec limit={60,0};
    while(nanosleep(&limit,&limit) && errno==EINTR) {}
    (void)!write(2,"FAIL: hung for 60 s\n",20);_exit(1);
}
static void start_watchdog(void) {
    pthread_t thread;if(!pthread_create(&thread,NULL,watchdog,NULL))pthread_detach(thread);
}
int main(void) {
    start_watchdog();
    small_cache_is_refused();
    all_keys_duplicate_and_budget();individual_ready_owner_and_fifo();
    hit_protection_forced_retention_and_unseen_eviction();cancellation_drains_readers_and_borrows();
    failures_hide_partial_records();init_under_ordinary_lease_and_shutdown();
    completion_order_does_not_change_policy();
    advice_is_bounded_and_does_not_block_ready_groups();cancellation_drains_active_advice();
    cancellation_drains_progressive_reader_and_advisor();layer_hits_are_pinned_before_misses();
    printf("layer loader CPU fixture: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
