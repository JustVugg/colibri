/* Synthetic K=8 cache groups, deterministic callback barriers, bounded waits.
 * No model, checkpoint, GPU, or inference. Build from the repository root:
 * clang -O2 -std=c11 -pthread c/tests/test_qwen36_global_cache.c -o /tmp/qwen-global-cache
 */
#define _POSIX_C_SOURCE 200809L
#include "../qwen36_global_cache.h"
#include <stdio.h>
#include <unistd.h>

static int checks,failures;
#define CHECK(c,...) do {__atomic_fetch_add(&checks,1,__ATOMIC_RELAXED);if(!(c)){ \
    __atomic_fetch_add(&failures,1,__ATOMIC_RELAXED);fprintf(stderr,"FAIL line %d: ",__LINE__); \
    fprintf(stderr,__VA_ARGS__);fputc('\n',stderr);}} while(0)
#define REQUIRE(c,...) do {if(!(c)){CHECK(0,__VA_ARGS__);exit(2);}} while(0)
enum {GROUP=8,SCRATCH=64};
typedef struct {uint64_t magic;int key;unsigned char bytes[116];} Payload;
typedef struct {
    QwenGlobalCache cache;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    unsigned allocations,allocated,loads,released;
    int fail_allocation,fail_key,pause_kind,pause_value,blocked,proceed;
    int order[8],order_count;
} Fixture;
static struct timespec deadline(void) {
    struct timespec t;clock_gettime(CLOCK_REALTIME,&t);t.tv_sec+=10;return t;
}
static void pause_callback(Fixture *f,int kind,int value) {
    if(f->pause_kind!=kind || f->pause_value!=value)return;
    f->blocked=1;pthread_cond_broadcast(&f->changed);
    struct timespec limit=deadline();
    while(!f->proceed)REQUIRE(!pthread_cond_timedwait(&f->changed,&f->mutex,&limit),"callback barrier timed out");
}
static void *allocate_payload(void *context,size_t slot) {
    Fixture *f=context;pthread_mutex_lock(&f->mutex);
    unsigned call=++f->allocations;
    CHECK(slot<f->cache.slot_count,"allocator slot outside pool");
    pause_callback(f,1,(int)call);
    int fail=f->fail_allocation==(int)call;pthread_mutex_unlock(&f->mutex);
    if(fail)return NULL;
    Payload *p=calloc(1,sizeof(*p));REQUIRE(p!=NULL,"tiny payload allocation failed");
    p->magic=UINT64_C(0x71aacafe53122026);p->key=-1;
    pthread_mutex_lock(&f->mutex);f->allocated++;pthread_mutex_unlock(&f->mutex);
    return p;
}
static int load_payload(void *context,int key,void *payload) {
    Fixture *f=context;Payload *p=payload;
    pthread_mutex_lock(&f->mutex);f->loads++;pause_callback(f,2,key);
    int fail=f->fail_key==key;pthread_mutex_unlock(&f->mutex);
    CHECK(p->magic==UINT64_C(0x71aacafe53122026),"loader received invalid allocation");
    p->key=key;for(size_t i=0;i<sizeof(p->bytes);i++)p->bytes[i]=(unsigned char)(key*13+(int)i);
    return !fail;
}
static void release_payload(void *context,void *payload) {
    Fixture *f=context;Payload *p=payload;
    CHECK(p->magic==UINT64_C(0x71aacafe53122026),"release received stale allocation");
    p->magic=0;pthread_mutex_lock(&f->mutex);f->released++;pthread_mutex_unlock(&f->mutex);free(p);
}
static void fixture_init_with_age(Fixture *f,size_t slots,uint64_t age_interval) {
    memset(f,0,sizeof(*f));f->fail_key=-1;
    REQUIRE(!pthread_mutex_init(&f->mutex,NULL) && !pthread_cond_init(&f->changed,NULL),"fixture synchronization");
    REQUIRE(qgc_init(&f->cache,64,slots,GROUP,sizeof(Payload),SCRATCH,age_interval,f,
                     allocate_payload,load_payload,release_payload),"cache init");
}
static void fixture_init(Fixture *f,size_t slots) {fixture_init_with_age(f,slots,8);}
static void budget(Fixture *f) {
    QgcStats s=qgc_snapshot(&f->cache);
    CHECK(s.allocated+s.reserved<=f->cache.budget && s.peak_charge<=f->cache.budget,"payload budget exceeded");
    CHECK(s.scratch<=SCRATCH && s.peak_scratch<=SCRATCH,"scratch reservation exceeded");
    CHECK(s.allocated+s.reserved+s.scratch<=f->cache.budget+SCRATCH,"combined reservation exceeded");
}
static void idle(Fixture *f) {
    QgcStats s=qgc_snapshot(&f->cache);budget(f);
    CHECK(!s.borrowed && !s.active && !s.loading && !s.waiters && !s.reserved && !s.scratch,"cache retained active reservations");
}
static void fixture_done(Fixture *f) {
    REQUIRE(qgc_destroy(&f->cache),"cache destroy refused while idle");
    CHECK(f->released==f->allocated,"destroy released %u/%u payloads",f->released,f->allocated);
    CHECK(qgc_destroy(&f->cache) && f->released==f->allocated,"second destroy changed releases");
    pthread_cond_destroy(&f->changed);pthread_mutex_destroy(&f->mutex);
}
static void sequence(int *keys,int first) {for(int i=0;i<GROUP;i++)keys[i]=first+i;}
static void check_group(Fixture *f,QgcLease *lease,const int *keys,unsigned n) {
    for(unsigned i=0;i<n;i++) {
        Payload *p=qgc_payload(&f->cache,lease,i);
        if(keys[i]<0){CHECK(!p,"hole exposed a payload");continue;}
        CHECK(p && p->magic==UINT64_C(0x71aacafe53122026) && p->key==keys[i],"wrong payload for selected key %d",keys[i]);
        if(p)for(size_t j=0;j<sizeof(p->bytes);j++)CHECK(p->bytes[j]==(unsigned char)(keys[i]*13+(int)j),"payload byte corrupted");
        for(unsigned j=0;j<i;j++)if(keys[j]==keys[i])CHECK(p==qgc_payload(&f->cache,lease,j),"duplicate key received different storage");
    }
    CHECK(!qgc_payload(&f->cache,lease,n),"out-of-group payload exposed");budget(f);
}
static void wait_blocked(Fixture *f) {
    struct timespec limit=deadline();pthread_mutex_lock(&f->mutex);
    while(!f->blocked)REQUIRE(!pthread_cond_timedwait(&f->changed,&f->mutex,&limit),"callback did not reach barrier");
    pthread_mutex_unlock(&f->mutex);
}
static void proceed(Fixture *f) {
    pthread_mutex_lock(&f->mutex);f->proceed=1;pthread_cond_broadcast(&f->changed);pthread_mutex_unlock(&f->mutex);
}
/* These waits observe actual queue/stop state; elapsed time never starts an
 * operation or establishes ordering. The sleep only yields during observation. */
static void wait_cache(Fixture *f,unsigned waiters,int stopping) {
    uint64_t until=qgc_now_ns()+UINT64_C(10000000000);
    for(;;) {
        pthread_mutex_lock(&f->cache.mutex);
        int ready=f->cache.stats.waiters==waiters && (!stopping || f->cache.stopping);
        pthread_mutex_unlock(&f->cache.mutex);if(ready)return;
        REQUIRE(qgc_now_ns()<until,"expected waiter/stop state was not reached");
        struct timespec pause={0,1000000};nanosleep(&pause,NULL);
    }
}
typedef struct {
    Fixture *f;pthread_t thread;int keys[GROUP],operation,tag,hold,go,ready,done,result,cancel;
    QgcLease lease; /* start() zero-initializes the entire Worker before use. */
} Worker;
static void *worker_main(void *argument) {
    Worker *w=argument;Fixture *f=w->f;int result;
    if(w->operation==1)result=qgc_prefetch(&f->cache,w->keys[0]);
    else if(w->operation==2)result=qgc_stop(&f->cache);
    else result=qgc_acquire(&f->cache,w->keys,GROUP,&w->cancel,&w->lease);
    if(result==1 && !w->operation)check_group(f,&w->lease,w->keys,GROUP);
    pthread_mutex_lock(&f->mutex);w->result=result;w->ready=1;
    if(result==1 && !w->operation && w->tag){REQUIRE(f->order_count<8,"order record overflow");f->order[f->order_count++]=w->tag;}
    pthread_cond_broadcast(&f->changed);struct timespec limit=deadline();
    while(result==1 && w->hold && !w->go)REQUIRE(!pthread_cond_timedwait(&f->changed,&f->mutex,&limit),"worker lease barrier timed out");
    pthread_mutex_unlock(&f->mutex);
    if(result==1 && !w->operation)CHECK(qgc_release_group(&f->cache,&w->lease),"worker release failed");
    pthread_mutex_lock(&f->mutex);w->done=1;pthread_cond_broadcast(&f->changed);pthread_mutex_unlock(&f->mutex);return NULL;
}
static void start(Worker *w,Fixture *f,int first,int tag,int hold,int operation) {
    memset(w,0,sizeof(*w));w->f=f;w->tag=tag;w->hold=hold;w->operation=operation;sequence(w->keys,first);
    REQUIRE(!pthread_create(&w->thread,NULL,worker_main,w),"worker start");
}
static void wait_ready(Worker *w) {
    Fixture *f=w->f;struct timespec limit=deadline();pthread_mutex_lock(&f->mutex);
    while(!w->ready)REQUIRE(!pthread_cond_timedwait(&f->changed,&f->mutex,&limit),"worker never completed admission");
    pthread_mutex_unlock(&f->mutex);
}
static void let_go(Worker *w) {
    pthread_mutex_lock(&w->f->mutex);w->go=1;pthread_cond_broadcast(&w->f->changed);pthread_mutex_unlock(&w->f->mutex);
}
static void join(Worker *w) {REQUIRE(!pthread_join(w->thread,NULL),"worker join");}

static void minimum_capacity_and_leases(void) {
    Fixture f;fixture_init(&f,GROUP);int keys[GROUP];sequence(keys,0);QgcLease lease={0};
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"capacity-eight group refused");
    check_group(&f,&lease,keys,GROUP);QgcLease original=lease,other={0};
    CHECK(!qgc_acquire(&f.cache,keys,GROUP,NULL,&lease) && !memcmp(&lease,&original,sizeof(lease)),"nested acquire damaged its own lease");
    CHECK(!qgc_acquire(&f.cache,keys,GROUP,NULL,&other),"same-thread nested acquisition accepted");
    CHECK(!qgc_stop(&f.cache),"owner stopped cache while holding a group");
    QgcStats s=qgc_snapshot(&f.cache);CHECK(s.borrowed==8 && s.allocated==8*sizeof(Payload),"full group not charged/borrowed");
    CHECK(qgc_release_group(&f.cache,&lease),"first release failed");
    CHECK(!qgc_release_group(&f.cache,&lease),"double release accepted");idle(&f);
    for(int i=0;i<GROUP;i++)CHECK(qgc_contains(&f.cache,i) && qgc_peek(&f.cache,i),"normal release discarded ready working bytes");
    QgcStats before_repeat=qgc_snapshot(&f.cache);
    /* Consecutive prefill rows use the same K=8 set. All eight slots are the
     * working window here: retaining READY bytes must avoid repeated loads. */
    for(int row=0;row<16;row++) {
        REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"repeated working group refused");
        check_group(&f,&lease,keys,GROUP);
        CHECK(qgc_release_group(&f.cache,&lease),"repeated working group release failed");
    }
    s=qgc_snapshot(&f.cache);
    CHECK(s.loads==before_repeat.loads && s.misses==before_repeat.misses &&
          s.hits-before_repeat.hits==16*GROUP && f.allocated==GROUP,
          "consecutive K8 working groups reread or reallocated experts");
    sequence(keys,8);REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"second full group refused");
    CHECK(!qgc_release_group(&f.cache,&original) && !qgc_payload(&f.cache,&original,0),"stale lease reached a new generation");
    check_group(&f,&lease,keys,GROUP);CHECK(qgc_release_group(&f.cache,&lease),"second release failed");
    int duplicates[GROUP]={2,2,-1,5,5,-1,7,7};
    REQUIRE(qgc_acquire(&f.cache,duplicates,GROUP,NULL,&lease),"duplicates/holes refused");
    check_group(&f,&lease,duplicates,GROUP);s=qgc_snapshot(&f.cache);
    CHECK(s.borrowed==6 && f.allocated==8,"duplicate borrows or bounded storage incorrect");
    CHECK(qgc_release_group(&f.cache,&lease),"duplicate release failed");
    int invalid[GROUP]={-2};CHECK(!qgc_acquire(&f.cache,invalid,GROUP,NULL,&lease),"invalid key admitted");
    CHECK(!qgc_acquire(&f.cache,keys,GROUP+1,NULL,&lease),"oversized group admitted");
    idle(&f);fixture_done(&f);
}
static void protect_selected_hits(void) {
    Fixture f;fixture_init(&f,16);int keys[GROUP];sequence(keys,0);QgcLease lease={0};Payload *before[GROUP];
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"retained warmup");
    for(int i=0;i<GROUP;i++)before[i]=qgc_payload(&f.cache,&lease,(unsigned)i);
    CHECK(qgc_release_group(&f.cache,&lease),"warmup release");
    /* Seed admission evidence while quiescent: key8 must displace the only
     * unselected resident7, even though miss8 precedes all seven hits. */
    pthread_mutex_lock(&f.cache.mutex);for(int i=0;i<8;i++)qcp_touch(&f.cache.policy,8,1);pthread_mutex_unlock(&f.cache.mutex);
    int mixed[GROUP]={8,0,1,2,3,4,5,6};QgcStats old=qgc_snapshot(&f.cache);
    REQUIRE(qgc_acquire(&f.cache,mixed,GROUP,NULL,&lease),"mixed group refused");check_group(&f,&lease,mixed,GROUP);
    for(int i=1;i<GROUP;i++)CHECK(qgc_payload(&f.cache,&lease,(unsigned)i)==before[i-1],"selected hit was evicted by earlier miss");
    QgcStats now=qgc_snapshot(&f.cache);
    CHECK(now.hits-old.hits==7 && now.misses-old.misses==1 && now.loads-old.loads==1,"selected-hit accounting changed");
    CHECK(!qgc_contains(&f.cache,7) && qgc_contains(&f.cache,8),"eligible unselected victim not replaced");
    CHECK(qgc_release_group(&f.cache,&lease),"mixed release");
    old=qgc_snapshot(&f.cache);REQUIRE(qgc_acquire(&f.cache,mixed,GROUP,NULL,&lease),"all-hit group refused");
    now=qgc_snapshot(&f.cache);CHECK(now.hits-old.hits==8 && now.loads==old.loads,"all selected hits not preserved");
    CHECK(qgc_release_group(&f.cache,&lease),"all-hit release");idle(&f);fixture_done(&f);
}
static void cache_and_lease_identity(void) {
    Fixture a,b;fixture_init(&a,GROUP);fixture_init(&b,GROUP);
    int ka[GROUP],kb[GROUP];sequence(ka,0);sequence(kb,8);
    QgcLease la={0},lb={0};
    REQUIRE(qgc_acquire(&a.cache,ka,GROUP,NULL,&la),"cache A acquisition");
    QgcLease saved=la;
    CHECK(!qgc_acquire(&b.cache,kb,GROUP,NULL,&la) && !memcmp(&saved,&la,sizeof(la)),
          "cache B overwrote cache A's held lease");
    CHECK(!qgc_snapshot(&b.cache).active && !b.allocated,"rejected foreign lease mutated cache B");
    REQUIRE(qgc_acquire(&b.cache,kb,GROUP,NULL,&lb),"cache B acquisition");
    CHECK(la.generation==lb.generation && pthread_equal(la.owner,lb.owner),"collision fixture did not share owner/generation");
    CHECK(!qgc_payload(&b.cache,&la,0) && !qgc_payload(&a.cache,&lb,0),"lease exposed another cache's payload");
    CHECK(!qgc_release_group(&b.cache,&la) && !qgc_release_group(&a.cache,&lb),"foreign lease released another cache");
    QgcLease copied_a=la,copied_b=lb;
    CHECK(!qgc_payload(&a.cache,&copied_a,0) && !qgc_payload(&b.cache,&copied_b,0),"copied live lease exposed payload");
    CHECK(!qgc_release_group(&a.cache,&copied_a) && !qgc_release_group(&b.cache,&copied_b),"copied live lease released original ownership");
    CHECK(qgc_snapshot(&a.cache).borrowed==GROUP && qgc_snapshot(&b.cache).borrowed==GROUP,"identity rejection changed borrows");
    check_group(&a,&la,ka,GROUP);check_group(&b,&lb,kb,GROUP);
    CHECK(qgc_release_group(&a.cache,&la),"original A lease no longer releases");
    check_group(&b,&lb,kb,GROUP);CHECK(qgc_release_group(&b.cache,&lb),"original B lease no longer releases");
    idle(&a);idle(&b);fixture_done(&a);fixture_done(&b);
}
static void idle_stop_clears_working(void) {
    Fixture f;fixture_init(&f,GROUP);int keys[GROUP];sequence(keys,24);QgcLease lease={0};
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"idle-stop warmup");
    CHECK(qgc_release_group(&f.cache,&lease),"idle-stop warmup release");idle(&f);
    for(int i=0;i<GROUP;i++)CHECK(qgc_contains(&f.cache,keys[i]),"idle-stop fixture lacks ready working entry");
    CHECK(qgc_stop(&f.cache),"idle stop failed");idle(&f);
    for(int i=0;i<GROUP;i++)CHECK(!qgc_contains(&f.cache,keys[i]) && !qgc_peek(&f.cache,keys[i]),"idle stop retained a working index");
    CHECK(qgc_snapshot(&f.cache).allocated==GROUP*sizeof(Payload),"idle stop lost allocated-empty payload charge");
    CHECK(!qgc_acquire(&f.cache,keys,GROUP,NULL,&lease) && !qgc_prefetch(&f.cache,0),"idle stop accepted new work");
    fixture_done(&f);
}
typedef struct {Fixture *f;pthread_t thread;int stop;uint64_t attempts;} PrefetchStorm;
static void *storm_main(void *argument) {
    PrefetchStorm *s=argument;unsigned key=32;
    while(!__atomic_load_n(&s->stop,__ATOMIC_ACQUIRE)) {
        qgc_prefetch(&s->f->cache,(int)key);key=32+(key+1)%8;
        __atomic_fetch_add(&s->attempts,1,__ATOMIC_RELAXED);
    }
    return NULL;
}
static void fifo_and_prefetch_priority(void) {
    Fixture f;fixture_init(&f,16);int keys[GROUP];sequence(keys,0);QgcLease lease={0};
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"foreground owner");
    Payload copies[GROUP];Payload *pointers[GROUP];for(int i=0;i<GROUP;i++){pointers[i]=qgc_payload(&f.cache,&lease,(unsigned)i);copies[i]=*pointers[i];}
    Worker a,b;start(&a,&f,8,1,1,0);wait_cache(&f,1,0);start(&b,&f,16,2,1,0);wait_cache(&f,2,0);
    PrefetchStorm storm={.f=&f};REQUIRE(!pthread_create(&storm.thread,NULL,storm_main,&storm),"prefetch storm start");
    uint64_t until=qgc_now_ns()+UINT64_C(10000000000);
    while(__atomic_load_n(&storm.attempts,__ATOMIC_RELAXED)<1000)REQUIRE(qgc_now_ns()<until,"prefetch producer did not run");
    for(int i=0;i<GROUP;i++)CHECK(!memcmp(pointers[i],&copies[i],sizeof(Payload)),"queued demand changed an active payload");
    CHECK(qgc_release_group(&f.cache,&lease),"foreground release");wait_ready(&a);
    pthread_mutex_lock(&f.mutex);CHECK(a.result==1 && !b.ready && f.order_count==1 && f.order[0]==1,"FIFO first waiter lost order");pthread_mutex_unlock(&f.mutex);
    CHECK(!qgc_snapshot(&f.cache).prefetch_loads,"prefetch passed queued demand");let_go(&a);wait_ready(&b);
    pthread_mutex_lock(&f.mutex);CHECK(b.result==1 && f.order_count==2 && f.order[1]==2,"FIFO second waiter lost order");pthread_mutex_unlock(&f.mutex);
    CHECK(!qgc_snapshot(&f.cache).prefetch_loads,"prefetch displaced foreground progress");
    __atomic_store_n(&storm.stop,1,__ATOMIC_RELEASE);REQUIRE(!pthread_join(storm.thread,NULL),"prefetch storm join");
    let_go(&b);join(&a);join(&b);idle(&f);fixture_done(&f);
}
static void admitted_prefetch(void) {
    Fixture f;fixture_init(&f,16);f.pause_kind=2;f.pause_value=30;Worker prefetch,demand;
    start(&prefetch,&f,30,0,0,1);wait_blocked(&f);QgcStats s=qgc_snapshot(&f.cache);budget(&f);
    CHECK(s.loading && !s.active && !s.borrowed && s.reserved==sizeof(Payload) && s.scratch==SCRATCH,"prefetch reservation not visible");
    CHECK(!qgc_contains(&f.cache,30),"loading prefetch published before completion");
    start(&demand,&f,0,1,1,0);wait_cache(&f,1,0);CHECK(!qgc_prefetch(&f.cache,31),"new prefetch passed waiting demand");
    proceed(&f);join(&prefetch);wait_ready(&demand);
    /* Publication may legitimately be followed by demand evicting this cold
     * prefetched key; successful completion, not later residency, is required. */
    CHECK(prefetch.result==1 && demand.result==1,"admitted prefetch could not finish before demand");
    CHECK(qgc_snapshot(&f.cache).prefetch_loads==1,"prefetch completion accounting");
    let_go(&demand);join(&demand);idle(&f);fixture_done(&f);
}
static void cancellation_and_stop(void) {
    Fixture f;fixture_init(&f,16);int keys[GROUP];sequence(keys,0);QgcLease lease={0};
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"cancellation owner");
    Worker a,b;start(&a,&f,8,1,0,0);wait_cache(&f,1,0);start(&b,&f,16,2,0,0);wait_cache(&f,2,0);
    __atomic_store_n(&a.cancel,1,__ATOMIC_RELEASE);qgc_wake(&f.cache);join(&a);wait_cache(&f,1,0);
    CHECK(!a.result,"cancelled waiter acquired a group");CHECK(qgc_release_group(&f.cache,&lease),"cancellation owner release");
    join(&b);CHECK(b.result==1 && f.order_count==1 && f.order[0]==2,"cancelled ticket blocked next waiter");idle(&f);fixture_done(&f);

    fixture_init(&f,GROUP);f.pause_kind=2;f.pause_value=2;
    start(&a,&f,0,1,0,0);wait_blocked(&f);start(&b,&f,16,2,0,0);wait_cache(&f,1,0);
    Worker stop;start(&stop,&f,0,0,0,2);wait_cache(&f,0,1);join(&b);
    CHECK(!b.result,"stop did not reject queued demand");proceed(&f);join(&a);join(&stop);
    CHECK(!a.result && stop.result==1,"stop left the in-flight group admitted");idle(&f);
    for(int i=0;i<GROUP;i++)CHECK(!qgc_contains(&f.cache,i),"stopped working group remained indexed");
    CHECK(!qgc_prefetch(&f.cache,40) && !qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"stopped cache admitted new work");fixture_done(&f);
}
static void pending_allocation_and_cancel(void) {
    Fixture f;fixture_init(&f,GROUP);f.pause_kind=1;f.pause_value=1;Worker a;
    start(&a,&f,0,1,1,0);wait_blocked(&f);QgcStats s=qgc_snapshot(&f.cache);budget(&f);
    CHECK(!s.allocated && s.reserved==sizeof(Payload) && s.borrowed==1 && s.active && s.loading,"pending allocation escaped accounting");
    proceed(&f);wait_ready(&a);s=qgc_snapshot(&f.cache);
    CHECK(a.result==1 && s.allocated==f.cache.budget && s.peak_charge==f.cache.budget && s.peak_scratch==SCRATCH,"pool peak accounting");
    let_go(&a);join(&a);idle(&f);fixture_done(&f);

    fixture_init(&f,GROUP);f.pause_kind=2;f.pause_value=2;start(&a,&f,0,1,0,0);wait_blocked(&f);
    __atomic_store_n(&a.cancel,1,__ATOMIC_RELEASE);qgc_wake(&f.cache);proceed(&f);join(&a);
    s=qgc_snapshot(&f.cache);CHECK(!a.result && s.allocated==3*sizeof(Payload),"cancelled load lost allocated-empty storage charge");idle(&f);
    int keys[GROUP];sequence(keys,8);QgcLease lease={0};REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"cache failed to recover after active cancellation");
    check_group(&f,&lease,keys,GROUP);CHECK(qgc_release_group(&f.cache,&lease),"post-cancel release");idle(&f);fixture_done(&f);
}
static void failure_cleanup(void) {
    for(int allocation_failure=0;allocation_failure<2;allocation_failure++) {
        Fixture f;fixture_init(&f,GROUP);
        if(allocation_failure)f.fail_allocation=3;else f.fail_key=2;
        int keys[GROUP];sequence(keys,0);QgcLease lease={0};
        CHECK(!qgc_acquire(&f.cache,keys,GROUP,NULL,&lease) && !lease.held,"callback failure left a live lease");idle(&f);
        QgcStats s=qgc_snapshot(&f.cache);
        CHECK(s.allocated==(size_t)(allocation_failure?2:3)*sizeof(Payload),"failed group lost allocation charge");
        CHECK(s.load_failures==1 && s.allocation_failures==(uint64_t)allocation_failure,"callback failure accounting");
        for(int i=0;i<GROUP;i++)CHECK(!qgc_contains(&f.cache,i),"failed working group left stale index");
        f.fail_allocation=0;f.fail_key=-1;sequence(keys,8);
        REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"failed reservation blocked subsequent demand");check_group(&f,&lease,keys,GROUP);
        CHECK(f.allocated==GROUP,"allocated-empty slots were leaked instead of reused");
        CHECK(qgc_release_group(&f.cache,&lease),"post-failure release");idle(&f);fixture_done(&f);
    }
    Fixture f;fixture_init(&f,16);f.fail_key=40;
    CHECK(qgc_prefetch(&f.cache,40)==-1 && !qgc_contains(&f.cache,40),"prefetch failure published data");idle(&f);
    CHECK(qgc_snapshot(&f.cache).allocated==sizeof(Payload),"failed prefetch allocation stopped being charged");
    f.fail_key=-1;CHECK(qgc_prefetch(&f.cache,40)==1 && f.allocated==1,"prefetch retry failed to reuse its empty allocation");
    idle(&f);fixture_done(&f);
}
typedef struct {int key[16];uint64_t used[16],clock;} LruReference;
static unsigned lru_group(LruReference *lru,int first) {
    unsigned hits=0;
    for(int key=first;key<first+GROUP;key++) {
        int found=-1,victim=0;uint64_t now=++lru->clock;
        for(int i=0;i<16;i++)if(lru->key[i]==key){found=i;break;}
        if(found>=0){lru->used[found]=now;hits++;continue;}
        for(int i=0;i<16;i++) {
            if(lru->key[i]<0){victim=i;break;}
            if(lru->used[i]<lru->used[victim])victim=i;
        }
        lru->key[victim]=key;lru->used[victim]=now;
    }
    return hits;
}
static unsigned replay_group(Fixture *f,int first) {
    int keys[GROUP];sequence(keys,first);QgcLease lease={0};QgcStats before=qgc_snapshot(&f->cache);
    REQUIRE(qgc_acquire(&f->cache,keys,GROUP,NULL,&lease),"manager replay group refused");
    for(int i=0;i<GROUP;i++) {
        Payload *p=qgc_payload(&f->cache,&lease,(unsigned)i);
        CHECK(p && p->magic==UINT64_C(0x71aacafe53122026) && p->key==keys[i],"manager replay returned wrong key");
        if(p)for(size_t j=0;j<sizeof(p->bytes);j++)CHECK(p->bytes[j]==(unsigned char)(keys[i]*13+(int)j),"manager replay payload corrupted");
    }
    CHECK(qgc_snapshot(&f->cache).borrowed==GROUP,"manager replay lost whole-group protection");
    CHECK(qgc_release_group(&f->cache,&lease),"manager replay release failed");
    QgcStats after=qgc_snapshot(&f->cache);
    CHECK(after.hits-before.hits+after.misses-before.misses==GROUP,"manager replay reference accounting");
    CHECK(after.loads-before.loads==after.misses-before.misses,"manager replay reread a hit or skipped a miss");
    budget(f);return (unsigned)(after.hits-before.hits);
}
static void actual_manager_scan_and_hot_set(void) {
    Fixture f;fixture_init_with_age(&f,16,256);
    LruReference lru={0};for(int i=0;i<16;i++)lru.key[i]=-1;
    unsigned scan_hits=0,lru_scan_hits=0;
    for(int sweep=0;sweep<256;sweep++) {
        unsigned hits=0,lru_hits=0;
        for(int layer=0;layer<4;layer++) {
            hits+=replay_group(&f,layer*GROUP);lru_hits+=lru_group(&lru,layer*GROUP);
        }
        if(sweep) {
            CHECK(hits==8,"actual manager scan oscillated at sweep %d (%u hits)",sweep,hits);
            CHECK(!lru_hits,"twice-capacity LRU reference unexpectedly hit");
            scan_hits+=hits;lru_scan_hits+=lru_hits;
        }
        /* Hits alone could hide a rotating retained subset. Its original
         * eight keys must also remain in the protected retention region. */
        pthread_mutex_lock(&f.cache.mutex);
        for(int i=0;i<GROUP;i++)CHECK(f.cache.entries[i].state==QGC_READY && f.cache.entries[i].key==i,"uniform scan replaced retained key %d",i);
        pthread_mutex_unlock(&f.cache.mutex);
    }
    CHECK(scan_hits==255*8 && lru_scan_hits==0,"actual manager steady scan comparison");
    int first_full=-1,lru_first_full=-1;unsigned tail_hits=0;
    /* Keep the same manager, history and payload pool. The new disjoint set
     * fills both halves of the 16-slot budget, so one working window cannot
     * produce full hits without genuine retained-set adaptation. */
    for(int cycle=1;cycle<=128;cycle++) {
        unsigned hits=replay_group(&f,32);hits+=replay_group(&f,40);
        unsigned lru_hits=lru_group(&lru,32);lru_hits+=lru_group(&lru,40);
        if(hits==16 && first_full<0)first_full=cycle;
        if(lru_hits==16 && lru_first_full<0)lru_first_full=cycle;
        if(first_full>0)CHECK(hits==16,"actual manager lost a fully resident hot set at cycle %d",cycle);
        if(cycle>112)tail_hits+=hits;
    }
    CHECK(first_full>0 && first_full<=65,"actual manager hot-set adaptation exceeded 65 cycles (first=%d)",first_full);
    CHECK(lru_first_full==2,"fitting hot-set LRU reference failed to adapt immediately");
    CHECK(tail_hits==16*16,"actual manager hot-set hits did not remain stable");
    for(int key=32;key<48;key++)CHECK(qgc_contains(&f.cache,key),"adapted hot key %d is absent",key);
    QgcStats stats=qgc_snapshot(&f.cache);
    CHECK(stats.loads==f.loads && f.allocated==16 && stats.peak_charge==16*sizeof(Payload),"actual manager callback/payload accounting");
    printf("actual manager 32-key scan: hits=%u, LRU=%u after warmup; disjoint 16-key hot set: full-hit cycle=%d (LRU=%d), final 16 cycles=%u/256 hits\n",
           scan_hits,lru_scan_hits,first_full,lru_first_full,tail_hits);
    idle(&f);fixture_done(&f);
}
/* The optional advise callback announces each distinct miss once, before the
 * group's first load, and changes nothing else: an identical cache without it
 * sees the same hits, misses, bypasses, evictions, loads and payloads. */
static int advised_keys[GROUP * 4],advised_count;
static unsigned advised_loads_seen[GROUP * 4];
static void advise_key(void *context,int key) {
    Fixture *f=context;
    REQUIRE(advised_count<(int)(sizeof advised_keys/sizeof *advised_keys),"too many advise calls in one group");
    pthread_mutex_lock(&f->mutex);advised_loads_seen[advised_count]=f->loads;pthread_mutex_unlock(&f->mutex);
    advised_keys[advised_count++]=key;
}
static void advise_is_side_effect_free(void) {
    Fixture plain,advised;fixture_init(&plain,24);fixture_init(&advised,24);advised.cache.advise=advise_key;
    uint32_t seed=12345;unsigned expected_calls=0,calls=0;
    for(int round=0;round<400;round++) {
        int keys[GROUP];
        for(int i=0;i<GROUP;i++) {
            seed=seed*1103515245u+12345u;int r=(int)((seed>>16)%70);
            keys[i]=r>=64?-1:(round%3?r%40:r);  /* holes, a hot subset, full range */
        }
        if(round%7==0)keys[5]=keys[2];         /* duplicate within a group */
        int expected[GROUP],n_expected=0;
        for(int i=0;i<GROUP;i++) {
            if(keys[i]<0 || qgc_contains(&advised.cache,keys[i]))continue;
            int dup=0;for(int j=0;j<n_expected;j++)dup|=expected[j]==keys[i];
            if(!dup)expected[n_expected++]=keys[i];
        }
        unsigned loads_before=advised.loads;advised_count=0;
        QgcLease a={0},b={0};
        REQUIRE(qgc_acquire(&plain.cache,keys,GROUP,NULL,&a) && qgc_acquire(&advised.cache,keys,GROUP,NULL,&b),"acquire");
        check_group(&plain,&a,keys,GROUP);check_group(&advised,&b,keys,GROUP);
        CHECK(advised_count==n_expected,"round %d: %d advise calls for %d distinct misses",round,advised_count,n_expected);
        for(int i=0;i<advised_count && i<n_expected;i++) {
            CHECK(advised_keys[i]==expected[i],"round %d: advised key %d, expected %d",round,advised_keys[i],expected[i]);
            CHECK(advised_loads_seen[i]==loads_before,"round %d: advise after a load of the same group",round);
        }
        expected_calls+=(unsigned)n_expected;calls+=(unsigned)advised_count;
        REQUIRE(qgc_release_group(&plain.cache,&a) && qgc_release_group(&advised.cache,&b),"release");
        QgcStats x=qgc_snapshot(&plain.cache),y=qgc_snapshot(&advised.cache);
        CHECK(x.hits==y.hits && x.misses==y.misses && x.bypasses==y.bypasses && x.evictions==y.evictions &&
              x.loads==y.loads && plain.loads==advised.loads,"round %d: advise changed cache behaviour",round);
        for(int key=0;key<64;key++)CHECK(qgc_contains(&plain.cache,key)==qgc_contains(&advised.cache,key),
                                         "round %d: residency of key %d differs",round,key);
    }
    QgcStats s=qgc_snapshot(&advised.cache);
    CHECK(calls==expected_calls && calls==s.misses,"advise calls %u, expected %u, misses %llu",calls,expected_calls,(unsigned long long)s.misses);
    printf("advise: %u announcements for %llu misses over 400 groups; hits=%llu evictions=%llu bypasses=%llu identical\n",
           calls,(unsigned long long)s.misses,(unsigned long long)s.hits,(unsigned long long)s.evictions,(unsigned long long)s.bypasses);
    idle(&plain);idle(&advised);fixture_done(&plain);fixture_done(&advised);
}
/* With load_many a group's misses are read together; the cache must make
 * the choices it makes one load at a time: the same slots, hits, misses,
 * bypasses, evictions, loads and residency. The callback loads backwards. */
static int many_calls;
static int load_many_backwards(void *context,const int *keys,void *const *payloads,unsigned n) {
    int ok=1;many_calls++;
    for(int i=(int)n-1;i>=0;i--)ok=load_payload(context,keys[i],payloads[i]) && ok;
    return ok;
}
static void load_many_matches_sequential(void) {
    Fixture plain,many;fixture_init(&plain,24);fixture_init(&many,24);many.cache.load_many=load_many_backwards;
    uint32_t seed=777;
    for(int round=0;round<400;round++) {
        int keys[GROUP];
        for(int i=0;i<GROUP;i++) {
            seed=seed*1103515245u+12345u;int r=(int)((seed>>16)%70);
            keys[i]=r>=64?-1:(round%3?r%40:r);  /* holes, a hot subset, full range */
        }
        if(round%7==0)keys[5]=keys[2];         /* duplicate within a group */
        QgcLease a={0},b={0};
        REQUIRE(qgc_acquire(&plain.cache,keys,GROUP,NULL,&a) && qgc_acquire(&many.cache,keys,GROUP,NULL,&b),"acquire");
        check_group(&plain,&a,keys,GROUP);check_group(&many,&b,keys,GROUP);
        for(int i=0;i<GROUP;i++)CHECK(a.slots[i]==b.slots[i],"round %d: key %d in slot %d, one at a time in %d",
                                      round,keys[i],b.slots[i],a.slots[i]);
        REQUIRE(qgc_release_group(&plain.cache,&a) && qgc_release_group(&many.cache,&b),"release");
        QgcStats x=qgc_snapshot(&plain.cache),y=qgc_snapshot(&many.cache);
        CHECK(x.hits==y.hits && x.misses==y.misses && x.bypasses==y.bypasses && x.evictions==y.evictions &&
              x.loads==y.loads && plain.loads==many.loads,"round %d: load_many changed cache behaviour",round);
        for(int key=0;key<64;key++)CHECK(qgc_contains(&plain.cache,key)==qgc_contains(&many.cache,key),
                                         "round %d: residency of key %d differs",round,key);
    }
    QgcStats s=qgc_snapshot(&many.cache);
    printf("load_many: %d calls for %llu loads over 400 groups; hits=%llu evictions=%llu bypasses=%llu identical\n",
           many_calls,(unsigned long long)s.loads,(unsigned long long)s.hits,(unsigned long long)s.evictions,
           (unsigned long long)s.bypasses);
    idle(&plain);idle(&many);fixture_done(&plain);fixture_done(&many);
    /* a failed read: no lease, nothing of the group resident, its slots reused */
    Fixture f;fixture_init(&f,GROUP);f.cache.load_many=load_many_backwards;f.fail_key=2;
    int keys[GROUP];sequence(keys,0);QgcLease lease={0};
    CHECK(!qgc_acquire(&f.cache,keys,GROUP,NULL,&lease) && !lease.held,"failed load_many left a live lease");idle(&f);
    CHECK(qgc_snapshot(&f.cache).load_failures==GROUP,"a failed load_many fails its whole group");
    for(int i=0;i<GROUP;i++)CHECK(!qgc_contains(&f.cache,i),"failed load_many left a stale index");
    f.fail_key=-1;sequence(keys,8);
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"failed load_many blocked later demand");check_group(&f,&lease,keys,GROUP);
    CHECK(f.allocated==GROUP,"slots of the failed group were not reused");
    CHECK(qgc_release_group(&f.cache,&lease),"release after a failed load_many");idle(&f);fixture_done(&f);
}
/* Decode keeps the experts whose next use is expected soonest. The fixture's
 * 64 keys are 4 layers of 16 experts, with top-8 groups. */
static void expected_reuse_in_decode(void) {
    enum {L=4,E=16};
    QwenExpectedPolicy *q=qep_new(64,L,E,GROUP);REQUIRE(q,"estimates allocation");
    int keys[GROUP];sequence(keys,0);q->decode=1;
    qep_begin(q,keys,GROUP);CHECK(q->group_active,"a layer's top-k is a decode group");
    qep_end(q,1);
    double chosen=0.5*q->beta;chosen+=q->alpha;
    CHECK(q->probability[0]==chosen && q->probability[GROUP]==0.5*q->beta && q->probability[E]==0.5,
          "one group updates the estimates of its own layer only");
    int mixed[3]={0,E,2*E};qep_begin(q,mixed,3);CHECK(!q->group_active,"a mixed group is not a decode group");
    qep_free(q);

    Fixture f;fixture_init(&f,24);                   /* 16 retained slots, 8 working */
    REQUIRE(qgc_expected_enable(&f.cache,L,E,GROUP),"estimates enable");
    CHECK(!qgc_expected_enable(&f.cache,L,E,GROUP),"estimates enabled twice");
    REQUIRE(qgc_expected_phase(&f.cache,1,1),"decode phase");
    QgcLease lease={0};
    for(int layer=0;layer<2;layer++) {               /* layers 0 and 1 fill the retained slots */
        sequence(keys,layer*E);
        REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"decode group");check_group(&f,&lease,keys,GROUP);
        REQUIRE(qgc_release_group(&f.cache,&lease),"decode release");
    }
    for(int k=0;k<2*E;k++)if(k%E<GROUP)
        CHECK(f.cache.index[k]>=0 && (size_t)f.cache.index[k]<f.cache.retained,"key %d not retained",k);
    /* Layer 2 has just run: its experts are expected after the retained ones,
     * so they take working slots and evict nothing. */
    sequence(keys,2*E);
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"layer 2 group");check_group(&f,&lease,keys,GROUP);
    REQUIRE(qgc_release_group(&f.cache,&lease),"layer 2 release");
    for(int i=0;i<GROUP;i++)CHECK(f.cache.index[2*E+i]<0 || (size_t)f.cache.index[2*E+i]>=f.cache.retained,
                                  "layer 2 expert %d displaced a retained one",i);
    CHECK(qgc_snapshot(&f.cache).evictions==0,"an expert expected later evicted one expected sooner");
    /* An expert chosen almost every time is expected soon: it takes the slot of
     * the retained expert expected farthest away, layer 1's oldest. */
    int hot=2*E+GROUP;
    pthread_mutex_lock(&f.cache.mutex);
    f.cache.expected->probability[hot]=0.99;f.cache.expected->repeat_numerator[hot]=0.99;
    f.cache.expected->repeat_denominator[hot]=1;
    pthread_mutex_unlock(&f.cache.mutex);
    sequence(keys,2*E+1);
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"hot group");check_group(&f,&lease,keys,GROUP);
    REQUIRE(qgc_release_group(&f.cache,&lease),"hot release");
    CHECK(f.cache.index[hot]>=0 && (size_t)f.cache.index[hot]<f.cache.retained,"the hot expert was not retained");
    CHECK(!qgc_contains(&f.cache,E) && qgc_snapshot(&f.cache).evictions==1,"the victim was not layer 1's oldest expert");
    /* A decode group that is not one layer's top-k keeps the frequency policy. */
    REQUIRE(qgc_acquire(&f.cache,mixed,3,NULL,&lease),"a mixed decode group was refused");
    REQUIRE(qgc_release_group(&f.cache,&lease),"mixed release");
    idle(&f);fixture_done(&f);
}
/* The decode estimates against a plain model of their definition. At each
 * visit of a layer every expert's probability and repeat counts decay by beta,
 * the chosen experts' probabilities gain alpha, and an expert chosen at the
 * previous visit gains alpha in its repeat denominator, and in its numerator
 * when chosen again. The expected time of an expert just chosen uses the
 * repeat ratio, of any other its probability. */
static double reference_time(const double *P,const double *N,const double *D,const unsigned char *last,
                             const unsigned char *now,int layer,int L,int E,int key) {
    int distance=((key/E)+L-layer)%L;if(!distance)distance=L;
    double next=P[key];int selected=key/E==layer ? now[key%E] : last[key];
    if(selected && D[key]>0){next=N[key]/D[key];if(next<0)next=0;if(next>1)next=1;}
    if(P[key]<=0)return next==1 ? (double)distance : INFINITY;
    return (double)distance+(double)L*(1.0-next)/P[key];
}
static void expected_policy_reference(void) {
    enum {L=4,E=16,K=4,KEYS=L*E,STEPS=400};
    QwenExpectedPolicy *q=qep_new(KEYS,L,E,K);REQUIRE(q,"estimates allocation");q->decode=1;
    double half=1.0;for(int i=0;i<16;i++)half*=q->beta;
    CHECK(fabs(half-0.5)<1e-15,"the estimates do not halve every 16 visits: (1-alpha)^16 = %.17g",half);
    double P[KEYS],N[KEYS],D[KEYS],prior=(double)K/E;unsigned char last[KEYS]={0},now[E];
    for(int k=0;k<KEYS;k++){P[k]=prior;N[k]=prior*prior;D[k]=prior;}
    uint32_t seed=2026;int worst_step=-1;double worst=0;
    for(int step=0;step<STEPS;step++) {
        int layer=step%L,keys[K];memset(now,0,sizeof now);
        for(int i=0;i<K;) {                         /* half of the choices from a hot quarter */
            seed=seed*1103515245u+12345u;int e=(int)((seed>>16)%(seed&1 ? E/4 : E));
            if(!now[e]){now[e]=1;keys[i++]=layer*E+e;}
        }
        qep_begin(q,keys,K);CHECK(q->group_active,"step %d: a layer's top-k is a decode group",step);
        for(int k=0;k<KEYS;k++) {
            double want=reference_time(P,N,D,last,now,layer,L,E,k),got=qep_expected_time(q,(size_t)k);
            double error=fabs(got-want)/want;
            if(!(error<=1e-12) && (worst_step<0 || !(error<=worst))){worst=error;worst_step=step;}
        }
        qep_end(q,1);
        for(int e=0;e<E;e++) {
            int k=layer*E+e;P[k]*=q->beta;N[k]*=q->beta;D[k]*=q->beta;
            if(now[e])P[k]+=q->alpha;
            if(last[k]){D[k]+=q->alpha;if(now[e])N[k]+=q->alpha;}
            last[k]=now[e];
        }
        for(int k=0;k<KEYS;k++)if(fabs(q->probability[k]-P[k])>1e-12 || fabs(q->repeat_numerator[k]-N[k])>1e-12 ||
                                  fabs(q->repeat_denominator[k]-D[k])>1e-12 || q->last_selected[k]!=last[k]) {
            CHECK(0,"step %d: the estimates of key %d differ from the model",step,k);step=STEPS;break;
        }
    }
    CHECK(worst_step<0,"expected times differ from the model (relative %.3g at step %d)",worst,worst_step);
    qep_free(q);
    /* An expert never chosen in 16 visits of its layer keeps half its estimate. */
    q=qep_new(KEYS,L,E,K);REQUIRE(q,"estimates allocation");q->decode=1;
    int keys[K];for(int i=0;i<K;i++)keys[i]=i;
    for(int visit=0;visit<16;visit++){qep_begin(q,keys,K);qep_end(q,1);}
    CHECK(fabs(q->probability[K]-prior/2)<1e-15 && fabs(q->probability[E]-prior)<1e-15,
          "after 16 visits without it an expert's estimate is %.17g, not %.17g",q->probability[K],prior/2);
    qep_free(q);
}
/* In decode a group's own hit may be the retained expert expected farthest
 * away: the misses of the same group must not take its slot. */
static void expected_keeps_own_hits(void) {
    enum {L=4,E=16};Fixture f;fixture_init(&f,24);int keys[GROUP];QgcLease lease={0};
    REQUIRE(qgc_expected_enable(&f.cache,L,E,GROUP) && qgc_expected_phase(&f.cache,1,1),"estimates");
    for(int layer=0;layer<2;layer++) {               /* layers 0 and 1 fill the retained slots */
        sequence(keys,layer*E);
        REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"decode group");REQUIRE(qgc_release_group(&f.cache,&lease),"release");
    }
    pthread_mutex_lock(&f.cache.mutex);              /* expert 0 is now chosen almost never */
    f.cache.expected->probability[0]=1e-6;f.cache.expected->repeat_numerator[0]=0;f.cache.expected->repeat_denominator[0]=1;
    pthread_mutex_unlock(&f.cache.mutex);
    int mixed[GROUP]={0,8,9,10,11,12,13,14};         /* its own hit and seven misses of layer 0 */
    REQUIRE(qgc_acquire(&f.cache,mixed,GROUP,NULL,&lease),"group with a far hit");check_group(&f,&lease,mixed,GROUP);
    REQUIRE(qgc_release_group(&f.cache,&lease),"far hit release");idle(&f);fixture_done(&f);
}
/* A miss expected exactly as soon as the farthest retained expert does not
 * evict it: only a sooner one does. */
static void expected_ties_keep_the_resident(void) {
    enum {L=4,E=16};Fixture f;fixture_init(&f,24);int keys[GROUP];QgcLease lease={0};
    REQUIRE(qgc_expected_enable(&f.cache,L,E,GROUP) && qgc_expected_phase(&f.cache,1,1),"estimates");
    for(int layer=2;layer<4;layer++) {               /* layers 2 and 3 fill the retained slots */
        sequence(keys,layer*E);
        REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"decode group");REQUIRE(qgc_release_group(&f.cache,&lease),"release");
    }
    pthread_mutex_lock(&f.cache.mutex);QwenExpectedPolicy *q=f.cache.expected;
    for(int e=0;e<E;e++) {                           /* layer 2: next use after 8 layer steps; layer 3: soon */
        q->probability[2*E+e]=0.5;q->repeat_numerator[2*E+e]=0.5;q->repeat_denominator[2*E+e]=1;
        q->probability[3*E+e]=0.99;q->repeat_numerator[3*E+e]=0.99;q->repeat_denominator[3*E+e]=1;
    }
    pthread_mutex_unlock(&f.cache.mutex);
    QgcStats before=qgc_snapshot(&f.cache);sequence(keys,2*E+GROUP);   /* layer 2's other experts: misses, also after 8 */
    REQUIRE(qgc_acquire(&f.cache,keys,GROUP,NULL,&lease),"tied group");check_group(&f,&lease,keys,GROUP);
    REQUIRE(qgc_release_group(&f.cache,&lease),"tied release");
    CHECK(qgc_snapshot(&f.cache).evictions==before.evictions,"a miss expected as soon as a retained expert evicted it");
    for(int e=0;e<GROUP;e++)CHECK(qgc_contains(&f.cache,2*E+e),"retained expert %d of layer 2 was replaced",e);
    idle(&f);fixture_done(&f);
}
/* Queued demand keeps its turn: a group behind another waiter is not served
 * while that waiter is first, though no lease is held, and prefetch does not
 * start a load meanwhile. The first waiter is a stand-in queued by hand. */
static void queue_order_without_a_lease(void) {
    Fixture f;fixture_init(&f,24);QgcWaiter first={NULL,NULL};      /* retained slots left empty for prefetch */
    pthread_mutex_lock(&f.cache.mutex);f.cache.first=f.cache.last=&first;f.cache.stats.waiters=1;pthread_mutex_unlock(&f.cache.mutex);
    Worker b;start(&b,&f,0,1,0,0);
    uint64_t until=qgc_now_ns()+UINT64_C(10000000000);int behind=0;
    for(;;) {                                         /* b queued behind the stand-in, or served past it */
        pthread_mutex_lock(&f.cache.mutex);unsigned queued=f.cache.stats.waiters;pthread_mutex_unlock(&f.cache.mutex);
        pthread_mutex_lock(&f.mutex);int served=b.ready;pthread_mutex_unlock(&f.mutex);
        if(queued==2){behind=1;break;}
        if(served)break;
        REQUIRE(qgc_now_ns()<until,"the group neither queued nor was served");
        struct timespec pause={0,1000000};nanosleep(&pause,NULL);
    }
    CHECK(behind,"a group was served ahead of the waiter before it");
    CHECK(!qgc_prefetch(&f.cache,40) && !qgc_snapshot(&f.cache).prefetch_loads,"prefetch started a load while demand was queued");
    pthread_mutex_lock(&f.cache.mutex);qgc_unqueue(&f.cache,&first);pthread_mutex_unlock(&f.cache.mutex);
    join(&b);CHECK(b.result==1,"the queued group was not served once first");
    CHECK(qgc_prefetch(&f.cache,40)==1 && qgc_snapshot(&f.cache).prefetch_loads==1,"prefetch did not load with nothing queued");
    idle(&f);fixture_done(&f);
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
    minimum_capacity_and_leases();protect_selected_hits();cache_and_lease_identity();idle_stop_clears_working();fifo_and_prefetch_priority();
    admitted_prefetch();cancellation_and_stop();pending_allocation_and_cancel();failure_cleanup();
    actual_manager_scan_and_hot_set();advise_is_side_effect_free();load_many_matches_sequential();
    expected_reuse_in_decode();expected_policy_reference();expected_keeps_own_hits();expected_ties_keep_the_resident();
    queue_order_without_a_lease();
    printf("qwen36 global cache: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
