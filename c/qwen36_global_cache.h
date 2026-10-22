/* One cache of expert payloads for the whole model, however its users split
 * the work. One FIFO foreground lease covers a complete group through compute.
 * Prefetch never queues ahead of demand. Cancellation is cooperative (flag +
 * qgc_wake), not pthread_cancel. Stop admission, join callers, then destroy.
 * A callback must complete without re-entering the cache. I/O may finish while
 * demand/shutdown waits; publication never needs a new admission lease. */
#ifndef QWEN36_GLOBAL_CACHE_H
#define QWEN36_GLOBAL_CACHE_H
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "qwen36_cache_policy.h"
#include "qwen36_expected_policy.h"

/* The loaders' threads run on small stacks of their own (256 KiB) with one
 * page of guard below. Windows guards a thread's stack itself, MinGW has no
 * sysconf, and its pthread_attr_setguardsize is a stub that returns ENOTSUP:
 * there *guard is 0. */
#ifdef _WIN32
#define QGC_SYSTEM_GUARD 1
#else
#define QGC_SYSTEM_GUARD 0
#endif
static inline int qgc_thread_stack(pthread_attr_t *attr,size_t stack,size_t *guard) {
#ifdef _WIN32
    *guard=0;return pthread_attr_setstacksize(attr,stack);
#else
    long page=sysconf(_SC_PAGESIZE);*guard=page>0?(size_t)page:4096;
    return pthread_attr_setstacksize(attr,stack) || pthread_attr_setguardsize(attr,*guard);
#endif
}

#define QGC_MAX_GROUP 64
enum { QGC_EMPTY, QGC_LOADING, QGC_READY };
typedef struct {
    int key, state;
    unsigned refs;
    uint64_t used;
    void *payload;
} QgcEntry;
typedef struct QgcWaiter { struct QgcWaiter *prev,*next; } QgcWaiter;
typedef struct {
    uint64_t hits,misses,bypasses,evictions,loads,load_failures,allocation_failures;
    uint64_t prefetch_loads,prefetch_skips,admission_wait_ns,load_ns;
    size_t allocated,reserved,peak_charge,scratch,peak_scratch,metadata;
    unsigned borrowed,waiters,active,loading;
} QgcStats;
typedef struct {
    const void *cache;
    uint64_t generation;
    pthread_t owner;
    unsigned held,n;
    int slots[QGC_MAX_GROUP];
} QgcLease;
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    QwenCachePolicy policy;
    QwenExpectedPolicy *expected; /* decode-only metadata, outside the payload budget */
    QgcEntry *entries;
    int *index;
    size_t key_count,slot_count,retained,group_cap,payload_bytes,scratch_limit,budget;
    uint64_t clock,generation;
    pthread_t owner;
    QgcLease *active_lease;
    int active,loading,stopping,initialized;
    QgcWaiter *first,*last;
    QgcStats stats;
    void *context;
    void *(*allocate)(void *,size_t);
    int (*load)(void *,int,void *);
    void (*release)(void *,void *);
    /* Optional, set after qgc_init: announces a miss before the group's
     * sequential loads (kernel readahead), without the cache mutex held.
     * The active lease remains held. Must not touch the cache. */
    void (*advise)(void *,int);
    /* Optional, set after qgc_init: loads n misses of one group together
     * (in parallel if it likes), as n calls of load would; the scratch of
     * its extra threads is its own. Must not touch the cache. */
    int (*load_many)(void *,const int *,void *const *,unsigned);
} QwenGlobalCache;

static uint64_t qgc_now_ns(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
}
static int qgc_cancelled(const int *cancel) {
    return cancel && __atomic_load_n(cancel,__ATOMIC_ACQUIRE);
}
static int qgc_init(QwenGlobalCache *c,size_t keys,size_t slots,size_t group,
                    size_t payload,size_t scratch,uint64_t age_interval,
                    void *context,void *(*allocate)(void *,size_t),
                    int (*load)(void *,int,void *),void (*release)(void *,void *)) {
    memset(c,0,sizeof(*c));
    if(!keys || keys>INT32_MAX || !group || group>QGC_MAX_GROUP || slots<group ||
       slots>INT32_MAX || !payload || slots>SIZE_MAX/payload ||
       slots>SIZE_MAX/sizeof(QgcEntry) || keys>SIZE_MAX/sizeof(int) ||
       !allocate || !load || !release)return 0;
    c->entries=calloc(slots,sizeof(*c->entries));c->index=malloc(keys*sizeof(*c->index));
    if(!c->entries || !c->index || !qcp_init(&c->policy,keys,age_interval))goto fail;
    if(pthread_mutex_init(&c->mutex,NULL))goto fail;
    if(pthread_cond_init(&c->changed,NULL)){pthread_mutex_destroy(&c->mutex);goto fail;}
    c->key_count=keys;c->slot_count=slots;c->retained=slots-group;c->group_cap=group;
    c->payload_bytes=payload;c->scratch_limit=scratch;c->budget=slots*payload;
    c->context=context;c->allocate=allocate;c->load=load;c->release=release;c->initialized=1;
    c->stats.metadata=sizeof(*c)+slots*sizeof(*c->entries)+keys*sizeof(*c->index)+qcp_payload_bytes(&c->policy);
    for(size_t i=0;i<keys;i++)c->index[i]=-1;
    for(size_t i=0;i<slots;i++)c->entries[i].key=-1;
    return 1;
fail:
    free(c->entries);free(c->index);qcp_free(&c->policy);memset(c,0,sizeof(*c));return 0;
}
/* Decode reuse estimates (qwen36_expected_policy.h), enabled once before any
 * lookup. The graph sets the phase at the start of every forward, when no
 * lease or load is in flight. */
static int qgc_expected_enable(QwenGlobalCache *c,size_t layers,size_t experts,size_t topk) {
    if (!c || !c->initialized) return 0;
    QwenExpectedPolicy *p=qep_new(c->key_count,layers,experts,topk); if (!p) return 0;
    pthread_mutex_lock(&c->mutex);
    int ok=topk<=c->group_cap && !c->expected && !c->stopping && !c->active && !c->loading && !c->first && !c->clock &&
           !c->stats.hits && !c->stats.misses && c->stats.metadata<=SIZE_MAX-p->bytes;
    if (ok) { c->expected=p; c->stats.metadata+=p->bytes; }
    pthread_mutex_unlock(&c->mutex); if (!ok) qep_free(p); return ok;
}
static int qgc_expected_phase(QwenGlobalCache *c,int S,int pos) {
    if (!c || !c->expected) return 1;
    pthread_mutex_lock(&c->mutex);
    int ok=!c->active && !c->loading && !c->first && !c->stopping;
    if (ok) { c->expected->decode=S==1 && pos>0; c->expected->group_active=0; }
    pthread_mutex_unlock(&c->mutex); return ok;
}
/* Mutex held. Empty allocated storage remains fully charged. */
static void qgc_hide(QwenGlobalCache *c,int slot) {
    QgcEntry *e=&c->entries[slot];
    if(e->key>=0 && c->index[e->key]==slot)c->index[e->key]=-1;
    e->key=-1;e->state=QGC_EMPTY;
}
static int qgc_retained_slot(QwenGlobalCache *c,int key) {
    int victim=-1;
    if (c->expected && c->expected->group_active) {
        double farthest=0;
        for (size_t i=0;i<c->retained;i++) {
            QgcEntry *e=&c->entries[i];
            if (e->refs || e->state==QGC_LOADING) continue;
            if (e->state==QGC_EMPTY) return (int)i;
            double next=qep_expected_time(c->expected,(size_t)e->key);
            if (victim<0 || next>farthest || (next==farthest && e->used<c->entries[victim].used)) {
                victim=(int)i; farthest=next;
            }
        }
        return victim>=0 && qep_expected_time(c->expected,(size_t)key)<farthest ? victim : -1;
    }
    for(size_t i=0;i<c->retained;i++) {
        QgcEntry *e=&c->entries[i];
        if(e->refs || e->state==QGC_LOADING)continue;
        if(e->state==QGC_EMPTY)return (int)i;
        if(victim<0 || qcp_score(&c->policy,(size_t)e->key)<qcp_score(&c->policy,(size_t)c->entries[victim].key) ||
           (qcp_score(&c->policy,(size_t)e->key)==qcp_score(&c->policy,(size_t)c->entries[victim].key) && e->used<c->entries[victim].used))victim=(int)i;
    }
    return victim>=0 && qcp_admit(&c->policy,(size_t)key,(size_t)c->entries[victim].key)?victim:-1;
}
/* Mutex held on entry/return; only one loader is admitted. Payload and scratch
 * reservations remain visible while allocation/read callbacks run unlocked. */
static int qgc_load_slot(QwenGlobalCache *c,int slot,int key) {
    QgcEntry *e=&c->entries[slot];int fresh=e->payload==NULL;
    if(c->loading || (fresh && c->stats.allocated+c->stats.reserved>c->budget-c->payload_bytes))abort();
    if(e->state==QGC_READY)c->stats.evictions++;
    qgc_hide(c,slot);e->key=key;e->state=QGC_LOADING;c->index[key]=slot;
    c->loading=1;c->stats.scratch=c->scratch_limit;
    if(c->stats.scratch>c->stats.peak_scratch)c->stats.peak_scratch=c->stats.scratch;
    if(fresh)c->stats.reserved+=c->payload_bytes;
    size_t charge=c->stats.allocated+c->stats.reserved;
    if(charge>c->stats.peak_charge)c->stats.peak_charge=charge;
    uint64_t start=qgc_now_ns();void *payload=e->payload;
    pthread_mutex_unlock(&c->mutex);
    if(fresh)payload=c->allocate(c->context,(size_t)slot);
    int ok=payload && c->load(c->context,key,payload);
    pthread_mutex_lock(&c->mutex);
    if(fresh) {
        c->stats.reserved-=c->payload_bytes;
        if(payload){e->payload=payload;c->stats.allocated+=c->payload_bytes;}
        else c->stats.allocation_failures++;
    }
    c->stats.loads++;c->stats.load_ns+=qgc_now_ns()-start;
    c->loading=0;c->stats.scratch=0;
    if(ok){e->state=QGC_READY;e->used=++c->clock;}
    else{c->stats.load_failures++;qgc_hide(c,slot);}
    pthread_cond_broadcast(&c->changed);return ok;
}
/* Mutex held on entry/return; the group's misses as one admitted loader, the
 * reads left to load_many. Every slot was chosen and borrowed before. */
static int qgc_load_slots(QwenGlobalCache *c,const int *slots,const int *keys,unsigned n) {
    void *payload[QGC_MAX_GROUP];int fresh[QGC_MAX_GROUP];
    if(c->loading)abort();
    for(unsigned i=0;i<n;i++) {
        QgcEntry *e=&c->entries[slots[i]];fresh[i]=e->payload==NULL;
        if(fresh[i] && c->stats.allocated+c->stats.reserved>c->budget-c->payload_bytes)abort();
        if(e->state==QGC_READY)c->stats.evictions++;
        qgc_hide(c,slots[i]);e->key=keys[i];e->state=QGC_LOADING;c->index[keys[i]]=slots[i];
        if(fresh[i])c->stats.reserved+=c->payload_bytes;
        payload[i]=e->payload;
    }
    c->loading=1;c->stats.scratch=c->scratch_limit;
    if(c->stats.scratch>c->stats.peak_scratch)c->stats.peak_scratch=c->stats.scratch;
    size_t charge=c->stats.allocated+c->stats.reserved;
    if(charge>c->stats.peak_charge)c->stats.peak_charge=charge;
    uint64_t start=qgc_now_ns();
    pthread_mutex_unlock(&c->mutex);
    int ok=1;
    for(unsigned i=0;i<n;i++)if(fresh[i] && !(payload[i]=c->allocate(c->context,(size_t)slots[i])))ok=0;
    ok=ok && c->load_many(c->context,keys,payload,n);
    pthread_mutex_lock(&c->mutex);
    for(unsigned i=0;i<n;i++)if(fresh[i]) {
        QgcEntry *e=&c->entries[slots[i]];c->stats.reserved-=c->payload_bytes;
        if(payload[i]){e->payload=payload[i];c->stats.allocated+=c->payload_bytes;}
        else c->stats.allocation_failures++;
    }
    c->stats.loads+=n;c->stats.load_ns+=qgc_now_ns()-start;
    c->loading=0;c->stats.scratch=0;
    for(unsigned i=0;i<n;i++) {
        QgcEntry *e=&c->entries[slots[i]];
        if(ok){e->state=QGC_READY;e->used=++c->clock;}
        else{c->stats.load_failures++;qgc_hide(c,slots[i]);}
    }
    pthread_cond_broadcast(&c->changed);return ok;
}
static void qgc_unqueue(QwenGlobalCache *c,QgcWaiter *w) {
    if(w->prev)w->prev->next=w->next;else c->first=w->next;
    if(w->next)w->next->prev=w->prev;else c->last=w->prev;
    c->stats.waiters--;pthread_cond_broadcast(&c->changed);
}
static int qgc_release_locked(QwenGlobalCache *c,QgcLease *g) {
    if(!g->held || g->cache!=c || c->active_lease!=g || !c->active || g->generation!=c->generation ||
       !pthread_equal(g->owner,pthread_self()) || !pthread_equal(c->owner,pthread_self()))return 0;
    for(unsigned i=0;i<g->n;i++)if(g->slots[i]>=0) {
        QgcEntry *e=&c->entries[g->slots[i]];if(!e->refs)abort();e->refs--;
    }
    for(size_t i=c->retained;i<c->slot_count;i++) {
        if(c->entries[i].refs || c->entries[i].state==QGC_LOADING)abort();
        if(c->stopping)qgc_hide(c,(int)i);
    }
    /* The bounded working window may retain READY bytes opportunistically.
     * Keeping its last group avoids rereading every row of a layer-major
     * prefill while admission history matures. These entries have no protected
     * retention: the next group may replace any unborrowed window entry. */
    c->active=0;c->active_lease=NULL;g->held=0;g->cache=NULL;pthread_cond_broadcast(&c->changed);return 1;
}
/* Lease objects are zero-initialized by the caller and remain at one address
 * until release. They are not copyable ownership handles. With hits, the
 * group's hits are handed over as soon as they are borrowed (payload i, NULL
 * for a miss), before any miss is read: their compute need not wait for the
 * disk. If the group then fails, the lease stays held, so that the hits stay
 * valid until its owner releases it; without hits a failed group releases its
 * lease itself. */
typedef void (*QgcHits)(void *context,void *const *payload,unsigned n);
static int qgc_acquire_early(QwenGlobalCache *c,const int *keys,unsigned n,const int *cancel,QgcLease *g,
                             QgcHits hits,void *hits_context);
static int qgc_acquire(QwenGlobalCache *c,const int *keys,unsigned n,const int *cancel,QgcLease *g) {
    return qgc_acquire_early(c,keys,n,cancel,g,NULL,NULL);
}
static int qgc_acquire_early(QwenGlobalCache *c,const int *keys,unsigned n,const int *cancel,QgcLease *g,
                             QgcHits hits,void *hits_context) {
    if(!c || !c->initialized || !keys || !g || g->held || n>c->group_cap)return 0;
    for(unsigned i=0;i<n;i++)if(keys[i]<-1 || (keys[i]>=0 && (size_t)keys[i]>=c->key_count))return 0;
    pthread_mutex_lock(&c->mutex);
    if(c->stopping || qgc_cancelled(cancel) || (c->active && pthread_equal(c->owner,pthread_self()))) {
        pthread_mutex_unlock(&c->mutex);return 0;
    }
    int requested[QGC_MAX_GROUP];memcpy(requested,keys,n*sizeof(int));keys=requested;
    memset(g,0,sizeof(*g));g->n=n;for(unsigned i=0;i<n;i++)g->slots[i]=-1;
    QgcWaiter w={c->last,NULL};if(c->last)c->last->next=&w;else c->first=&w;c->last=&w;c->stats.waiters++;
    while(c->first!=&w || c->active || c->loading) {
        if(c->stopping || qgc_cancelled(cancel))break;
        uint64_t start=qgc_now_ns();
        if(pthread_cond_wait(&c->changed,&c->mutex))abort();
        c->stats.admission_wait_ns+=qgc_now_ns()-start;
    }
    int refused=c->stopping || qgc_cancelled(cancel);qgc_unqueue(c,&w);
    if(refused){pthread_mutex_unlock(&c->mutex);return 0;}
    c->active=1;c->owner=pthread_self();if(!++c->generation)++c->generation;
    g->held=1;g->cache=c;g->generation=c->generation;g->owner=c->owner;c->active_lease=g;
    qep_begin(c->expected,keys,n);
    /* Protect EVERY selected hit before choosing any miss victim. */
    for(unsigned i=0;i<n;i++)if(keys[i]>=0) {
        qcp_touch(&c->policy,(size_t)keys[i],1);
        int slot=c->index[keys[i]];
        if(slot>=0 && c->entries[slot].state==QGC_READY) {
            c->entries[slot].refs++;c->entries[slot].used=++c->clock;g->slots[i]=slot;c->stats.hits++;
        }
    }
    if(hits) {      /* without the mutex, as the advice below: the active lease keeps every other admission out */
        void *early[QGC_MAX_GROUP]={0};
        for(unsigned i=0;i<n;i++)if(g->slots[i]>=0)early[i]=c->entries[g->slots[i]].payload;
        pthread_mutex_unlock(&c->mutex);
        hits(hits_context,early,n);
        pthread_mutex_lock(&c->mutex);
    }
    /* Announce every distinct miss before the first read, so the device can
     * serve them together. The calls run without the mutex, so a slow one
     * does not block presence lookups: the active lease keeps every other
     * admission out, and the selected hits are already borrowed. */
    if(c->advise) {
        int advice_keys[QGC_MAX_GROUP];unsigned advice_n=0;
        for(unsigned i=0;i<n;i++)if(keys[i]>=0 && g->slots[i]<0) {
            unsigned j=0;while(j<i && keys[j]!=keys[i])j++;
            if(j==i)advice_keys[advice_n++]=keys[i];
        }
        for(unsigned i=0;i<advice_n;i++) {
            if(c->stopping || qgc_cancelled(cancel))goto failed;
            pthread_mutex_unlock(&c->mutex);
            c->advise(c->context,advice_keys[i]);
            pthread_mutex_lock(&c->mutex);
        }
        if(c->stopping || qgc_cancelled(cancel))goto failed;
    }
    /* With load_many every miss gets its slot first and the reads go
     * together: the slots are the ones the loop picks one load at a time,
     * since each is borrowed as soon as it is chosen. */
    int fill_slots[QGC_MAX_GROUP],fill_keys[QGC_MAX_GROUP];unsigned fills=0;
    for(unsigned i=0;i<n;i++)if(keys[i]>=0 && g->slots[i]<0) {
        if(c->stopping || qgc_cancelled(cancel))goto failed;
        for(unsigned j=0;j<i;j++)if(keys[j]==keys[i] && g->slots[j]>=0) {
            g->slots[i]=g->slots[j];c->entries[g->slots[i]].refs++;c->stats.hits++;break;
        }
        if(g->slots[i]>=0)continue;
        c->stats.misses++;int slot=qgc_retained_slot(c,keys[i]);
        if(slot<0) {
            for(size_t j=c->retained;j<c->slot_count;j++) {
                QgcEntry *e=&c->entries[j];if(e->refs || e->state==QGC_LOADING)continue;
                if(e->state==QGC_EMPTY){slot=(int)j;break;}
                if(slot<0 || e->used<c->entries[slot].used)slot=(int)j;
            }
            c->stats.bypasses++;
        }
        if(slot<0)abort(); /* full-group reservation prevents hold-and-wait */
        c->entries[slot].refs++;g->slots[i]=slot;
        if(c->load_many){fill_slots[fills]=slot;fill_keys[fills++]=keys[i];}
        else if(!qgc_load_slot(c,slot,keys[i]))goto failed;
    }
    if(fills && !qgc_load_slots(c,fill_slots,fill_keys,fills))goto failed;
    if(c->stopping || qgc_cancelled(cancel))goto failed;
    qep_end(c->expected,1);
    pthread_mutex_unlock(&c->mutex);return 1;
failed:
    qep_end(c->expected,0);
    if(hits){pthread_mutex_unlock(&c->mutex);return 0;}    /* the hits may be in use: the owner releases */
    qgc_release_locked(c,g);
    for(size_t i=c->retained;i<c->slot_count;i++)qgc_hide(c,(int)i);
    pthread_mutex_unlock(&c->mutex);return 0;
}
static void *qgc_payload(QwenGlobalCache *c,const QgcLease *g,unsigned i) {
    pthread_mutex_lock(&c->mutex);void *p=NULL;
    if(g->held && g->cache==c && c->active_lease==g && c->active && g->generation==c->generation && i<g->n && g->slots[i]>=0 &&
       pthread_equal(g->owner,pthread_self()) && c->entries[g->slots[i]].state==QGC_READY)p=c->entries[g->slots[i]].payload;
    pthread_mutex_unlock(&c->mutex);return p;
}
static int qgc_release_group(QwenGlobalCache *c,QgcLease *g) {
    pthread_mutex_lock(&c->mutex);int ok=qgc_release_locked(c,g);pthread_mutex_unlock(&c->mutex);return ok;
}
/* Best effort only: 1 when the key is resident or was loaded, 0 when skipped,
 * -1 when its load failed. Queued demand outranks every new prefetch load.
 * Nothing in the engine calls it yet. */
static int qgc_prefetch(QwenGlobalCache *c,int key) {
    if(!c || !c->initialized || key<0 || (size_t)key>=c->key_count)return 0;
    pthread_mutex_lock(&c->mutex);
    if(c->stopping || c->active || c->loading || c->first){c->stats.prefetch_skips++;pthread_mutex_unlock(&c->mutex);return 0;}
    if(c->index[key]>=0){pthread_mutex_unlock(&c->mutex);return 1;}
    int slot=qgc_retained_slot(c,key);
    if(slot<0){c->stats.prefetch_skips++;pthread_mutex_unlock(&c->mutex);return 0;}
    int ok=qgc_load_slot(c,slot,key);if(ok)c->stats.prefetch_loads++;
    pthread_mutex_unlock(&c->mutex);return ok?1:-1;
}
/* Presence only: payload pointers require a lease, never this lookup. */
static int qgc_contains(QwenGlobalCache *c,int key) {
    if(key<0 || (size_t)key>=c->key_count)return 0;
    pthread_mutex_lock(&c->mutex);int i=c->index[key];int yes=i>=0 && c->entries[i].state==QGC_READY;
    pthread_mutex_unlock(&c->mutex);return yes;
}
/* Presence/debug metadata only. The stable descriptor is NOT a read lease. */
static void *qgc_peek(QwenGlobalCache *c,int key) {
    if(key<0 || (size_t)key>=c->key_count)return NULL;
    pthread_mutex_lock(&c->mutex);int i=c->index[key];
    void *p=i>=0 && c->entries[i].state==QGC_READY?c->entries[i].payload:NULL;
    pthread_mutex_unlock(&c->mutex);return p;
}
static QgcStats qgc_snapshot(QwenGlobalCache *c) {
    pthread_mutex_lock(&c->mutex);QgcStats s=c->stats;
    s.active=c->active;s.loading=c->loading;
    for(size_t i=0;i<c->slot_count;i++)s.borrowed+=c->entries[i].refs;
    pthread_mutex_unlock(&c->mutex);return s;
}
static void qgc_wake(QwenGlobalCache *c) {
    pthread_mutex_lock(&c->mutex);pthread_cond_broadcast(&c->changed);pthread_mutex_unlock(&c->mutex);
}
static int qgc_stop(QwenGlobalCache *c) {
    pthread_mutex_lock(&c->mutex);
    if(c->active && pthread_equal(c->owner,pthread_self())){pthread_mutex_unlock(&c->mutex);return 0;}
    c->stopping=1;pthread_cond_broadcast(&c->changed);
    while(c->active || c->loading || c->stats.waiters)if(pthread_cond_wait(&c->changed,&c->mutex))abort();
    for(size_t i=c->retained;i<c->slot_count;i++)qgc_hide(c,(int)i);
    pthread_mutex_unlock(&c->mutex);return 1;
}
/* Caller has stopped/joined users; no callback or external borrower may remain. */
static int qgc_destroy(QwenGlobalCache *c) {
    if(!c || !c->initialized)return 1;
    if(!qgc_stop(c))return 0;
    for(size_t i=0;i<c->slot_count;i++)if(c->entries[i].payload)c->release(c->context,c->entries[i].payload);
    free(c->entries);free(c->index);qcp_free(&c->policy);qep_free(c->expected);
    pthread_cond_destroy(&c->changed);pthread_mutex_destroy(&c->mutex);memset(c,0,sizeof(*c));return 1;
}
#endif
