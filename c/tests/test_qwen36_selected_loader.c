/* Real FIFO cache admission, worker-owned leases and cross-thread read views.
 * Barriers, not sleeps, establish the cancellation and contention boundaries;
 * the short pauses only yield while a queue state is observed. */
#include <limits.h>
#include "../qwen36_selected_loader.h"
#include <stdio.h>
#include <unistd.h>
static int checks,failures;
#define CHECK(c,...) do {__atomic_fetch_add(&checks,1,__ATOMIC_RELAXED);if(!(c)){ \
    __atomic_fetch_add(&failures,1,__ATOMIC_RELAXED);fprintf(stderr,"FAIL line %d: ",__LINE__); \
    fprintf(stderr,__VA_ARGS__);fputc('\n',stderr);}} while(0)
#define REQUIRE(c,...) do {if(!(c)){CHECK(0,__VA_ARGS__);exit(2);}} while(0)
typedef struct {
    pthread_mutex_t mutex;pthread_cond_t changed;
    int block,block_key,fail_key,entered,proceed;unsigned loads;
} Fixture;
static void *allocate(void *context,size_t slot){(void)context;(void)slot;return malloc(64);}
/* With block, a load (of block_key, or of any key when it is -1) waits for proceed. */
static int load(void *context,int key,void *payload) {
    Fixture*f=context;pthread_mutex_lock(&f->mutex);f->loads++;
    if(f->block && (f->block_key<0 || key==f->block_key)) {
        f->entered=1;pthread_cond_broadcast(&f->changed);while(!f->proceed)pthread_cond_wait(&f->changed,&f->mutex);
    }
    int fail=key==f->fail_key;pthread_mutex_unlock(&f->mutex);memset(payload,key,64);return !fail;
}
static void release(void *context,void *payload){(void)context;free(payload);}
static void idle(QwenGlobalCache*c) {
    QgcStats s=qgc_snapshot(c);
    CHECK(!s.active&&!s.borrowed&&!s.loading&&!s.reserved&&!s.scratch&&!s.waiters,"cache retained active reservations");
}
static int cancelled(void *p){(void)p;return 1;}
static void *abort_job(void *p){qsl_finish(p,1);return NULL;}
static void let_proceed(Fixture *f){pthread_mutex_lock(&f->mutex);f->proceed=1;pthread_cond_broadcast(&f->changed);pthread_mutex_unlock(&f->mutex);}
static int loader_ready(QwenSelectedLoader *w){pthread_mutex_lock(&w->mutex);int r=w->ready;pthread_mutex_unlock(&w->mutex);return r;}
/* 1 once the cache has n waiters, 0 if w's group was served first; at most 10 s. */
static int queued(QwenGlobalCache *c,unsigned n,QwenSelectedLoader *w) {
    uint64_t until=qgc_now_ns()+UINT64_C(10000000000);
    for(;;) {
        pthread_mutex_lock(&c->mutex);unsigned q=c->stats.waiters;pthread_mutex_unlock(&c->mutex);
        if(q==n)return 1;if(w && loader_ready(w))return 0;
        REQUIRE(qgc_now_ns()<until,"the cache never had %u waiter(s)",n);
        struct timespec t={0,1000000};nanosleep(&t,NULL);
    }
}
/* 1 once w's worker has handed the hits over, 0 after 10 s without them. */
static int hits_handed_over(QwenSelectedLoader *w) {
    uint64_t until=qgc_now_ns()+UINT64_C(10000000000);
    for(;;) {
        pthread_mutex_lock(&w->mutex);int r=w->hits_ready;pthread_mutex_unlock(&w->mutex);
        if(r)return 1;if(qgc_now_ns()>=until)return 0;
        struct timespec t={0,1000000};nanosleep(&t,NULL);
    }
}
/* A hang must fail, not sit forever in CI. MinGW has no alarm(), so a
 * detached thread ends the run instead (as test_qwen36_tier_shutdown.c does). */
static void *watchdog(void *arg) {
    (void)arg;struct timespec limit={60,0};
    while(nanosleep(&limit,&limit) && errno==EINTR) {}
    (void)!write(2,"FAIL: hung for 60 s\n",20);_exit(1);
}
int main(void) {
    pthread_t guard;if(!pthread_create(&guard,NULL,watchdog,NULL))pthread_detach(guard);
    Fixture f={.mutex=PTHREAD_MUTEX_INITIALIZER,.changed=PTHREAD_COND_INITIALIZER,.block_key=-1,.fail_key=-1};
    QwenGlobalCache cache;
    REQUIRE(qgc_init(&cache,64,8,8,64,128,64,&f,allocate,load,release),"cache init");
    QwenSelectedLoader a,b;REQUIRE(qsl_init(&a,&cache)&&qsl_init(&b,&cache),"loader init");
    int keys[8];void *views[8];for(int i=0;i<8;i++)keys[i]=i;
    REQUIRE(qsl_start(&a,keys,8)&&qsl_wait(&a,NULL,NULL,views),"first group");
    CHECK(qgc_snapshot(&cache).borrowed==8,"the group is not borrowed while in use");
    CHECK(!qsl_start(&a,keys,8),"a second group started on a busy loader");
    for(int i=0;i<8;i++)CHECK(((unsigned char*)views[i])[0]==i,"first group payload %d",i);
    /* The inference thread cannot release or use the cache lease API for a
     * worker-owned lease, even though its published payload views are valid. */
    pthread_mutex_lock(&cache.mutex);QgcLease *lease=cache.active_lease;pthread_mutex_unlock(&cache.mutex);
    CHECK(lease&&!qgc_release_group(&cache,lease)&&qgc_payload(&cache,lease,0)==NULL,"another thread used the worker's lease");
    for(int i=0;i<8;i++)keys[i]=8+i;
    REQUIRE(qsl_start(&b,keys,8),"second loader");CHECK(!qgc_prefetch(&cache,32),"prefetch passed queued demand");
    qsl_finish(&a,0);REQUIRE(qsl_wait(&b,NULL,NULL,views),"second loader after the first finished");
    for(int i=0;i<8;i++)CHECK(((unsigned char*)views[i])[0]==8+i,"second group payload %d",i);
    qsl_finish(&b,0);idle(&cache);
    /* Cancel a queued borrower while the other worker retains the whole group. */
    REQUIRE(qsl_start(&a,keys,8)&&qsl_wait(&a,NULL,NULL,views),"retaining group");
    for(int i=0;i<8;i++)keys[i]=16+i;
    REQUIRE(qsl_start(&b,keys,8),"queued borrower");CHECK(queued(&cache,1,&b),"the borrower was served past a held group");
    qsl_finish(&b,1);
    CHECK(qgc_snapshot(&cache).borrowed==8 && !qgc_snapshot(&cache).waiters,"cancelling the queued borrower changed the held group");
    qsl_finish(&a,0);idle(&cache);
    /* Cancellation during an actual read drains that read, prevents remaining
     * loads, then reuses the same worker for a later successful request. */
    f.block=1;f.entered=f.proceed=0;unsigned before=f.loads;
    REQUIRE(qsl_start(&a,keys,8),"group to cancel");pthread_mutex_lock(&f.mutex);
    while(!f.entered)pthread_cond_wait(&f.changed,&f.mutex);pthread_mutex_unlock(&f.mutex);
    CHECK(!qsl_wait(&a,cancelled,NULL,views),"a cancelled wait gave a group");
    pthread_t aborter;REQUIRE(!pthread_create(&aborter,NULL,abort_job,&a),"aborter start");
    /* Wait on the cache condition until cancellation has actually been sent. */
    pthread_mutex_lock(&cache.mutex);while(!qgc_cancelled(&a.cancel))pthread_cond_wait(&cache.changed,&cache.mutex);pthread_mutex_unlock(&cache.mutex);
    let_proceed(&f);
    REQUIRE(!pthread_join(aborter,NULL),"aborter join");CHECK(f.loads==before+1,"cancellation did not stop the remaining loads");
    idle(&cache);f.block=0;
    REQUIRE(qsl_start(&a,keys,8)&&qsl_wait(&a,NULL,NULL,views),"group after a cancellation");qsl_finish(&a,0);idle(&cache);
    /* The hits are handed over while a miss of the same group is still read. */
    int warm[8],mixed[8]={32,33,34,35,48,49,50,51},other[8];void *hits[8];
    for(int i=0;i<8;i++){warm[i]=32+i;other[i]=56+i;}
    REQUIRE(qsl_start(&a,warm,8)&&qsl_wait(&a,NULL,NULL,views),"warm group");qsl_finish(&a,0);
    pthread_mutex_lock(&f.mutex);f.block=1;f.block_key=48;f.entered=f.proceed=0;pthread_mutex_unlock(&f.mutex);
    REQUIRE(qsl_start(&a,mixed,8),"mixed group");
    memset(hits,0,sizeof hits);int early=hits_handed_over(&a);unsigned found=early?qsl_wait_hits(&a,hits):0;
    pthread_mutex_lock(&f.mutex);int reading=!f.proceed;pthread_mutex_unlock(&f.mutex);
    CHECK(early && reading,"the hits were not handed over before the miss was read");
    CHECK(found==4,"%u hits handed over, 4 expected",found);
    for(int i=0;i<8;i++)CHECK(i<4 ? hits[i] && ((unsigned char*)hits[i])[0]==32+i : !hits[i],"hand-over entry %d",i);
    let_proceed(&f);REQUIRE(qsl_wait(&a,NULL,NULL,views),"mixed group after the read");
    for(int i=0;i<8;i++)CHECK(((unsigned char*)views[i])[0]==mixed[i],"mixed group payload %d",i);
    qsl_finish(&a,0);f.block=0;f.block_key=-1;idle(&cache);
    /* A group that fails after the hand-over keeps its lease until qsl_finish:
     * its hits stay valid, and another loader waits for them. */
    REQUIRE(qsl_start(&a,warm,8)&&qsl_wait(&a,NULL,NULL,views),"warm group again");qsl_finish(&a,0);
    f.fail_key=48;REQUIRE(qsl_start(&a,mixed,8),"failing group");
    memset(hits,0,sizeof hits);found=qsl_wait_hits(&a,hits);CHECK(found==4,"failing group: %u hits handed over",found);
    CHECK(!qsl_wait(&a,NULL,NULL,views),"a failed read gave a group");
    REQUIRE(qsl_start(&b,other,8),"loader behind the failed group");
    CHECK(queued(&cache,1,&b),"another loader was served before the failed group's qsl_finish");
    for(int i=0;i<4;i++)CHECK(hits[i] && ((unsigned char*)hits[i])[0]==32+i,"hit %d overwritten before qsl_finish",i);
    qsl_finish(&a,0);REQUIRE(qsl_wait(&b,NULL,NULL,views),"the loader behind the failed group");
    for(int i=0;i<8;i++)CHECK(((unsigned char*)views[i])[0]==other[i],"group after the failure, payload %d",i);
    qsl_finish(&b,0);f.fail_key=-1;idle(&cache);
    /* Destruction also drains a ready, unused job and joins before cache free. */
    REQUIRE(qsl_start(&a,keys,8)&&qsl_wait(&a,NULL,NULL,views),"group before destruction");
    qsl_destroy(&a);qsl_destroy(&b);idle(&cache);
    CHECK(qgc_destroy(&cache),"cache destroy");pthread_cond_destroy(&f.changed);pthread_mutex_destroy(&f.mutex);
    printf("selected loader: FIFO progress, thread-owned leases, cancellation during read, hits before the misses "
           "and kept after a failure, reuse and joined teardown: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
