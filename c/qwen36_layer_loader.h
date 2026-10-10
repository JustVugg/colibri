/* A prompt layer's experts, loaded while the GPU runs the layer's trunk.
 * The inference thread owns begin/append/wait/finish: append adds the experts
 * that a window of rows chose, wait borrows a group of them once loaded. Four
 * reader threads load the misses straight into reserved cache slots. When the
 * cache has an advise callback, one reader first announces each appended
 * batch, outside the cache mutex, and a record can be read as soon as its own
 * hint is out; earlier batches never wait behind later hints.
 *
 * Pointers from wait stay valid until finish, across later appends. The
 * caller must wait for the GPU to finish with them before finish or destroy,
 * even after a failure. Cancellation never interrupts a running callback.
 * Destroy the loader before its cache, and keep it at one address.
 *
 * Unlike the ordinary lease, every miss is admitted to the retained slots
 * (the layer needs all of them at once) and keys are touched in append order,
 * so a prompt leaves a different set of experts in the cache for decode.
 */
#ifndef QWEN36_LAYER_LOADER_H
#define QWEN36_LAYER_LOADER_H
#include "qwen36_global_cache.h"

#define QLL_MAX_KEYS 256
#define QLL_READERS 4
#define QLL_STACK_BYTES (256u * 1024u)
enum { QLL_QUEUED, QLL_READING, QLL_READY, QLL_FAILED, QLL_CANCELLED };
typedef struct {
    int key, slot, state, fresh;
    uint64_t generation;
} QllItem;
typedef struct {
    uint64_t reservations, selected, hits, misses, duplicate_keys;
    uint64_t forced_admissions, evictions, load_calls, load_bytes, load_failures;
    uint64_t allocation_failures, cancelled_records, cancelled_reservations;
    uint64_t completed_after_cancel, waits, wait_ns, drain_ns, callback_ns;
    uint64_t advice_batches, advise_calls, advise_ns;
    size_t metadata_bytes, stack_bytes, guard_bytes, callback_scratch_bound;
    unsigned peak_readers, peak_selected, held, queued, reading, selected_now;
    unsigned advising;
} QllStats;
typedef struct {
    QwenGlobalCache *cache;
    const void *identity;
    pthread_t threads[QLL_READERS], owner;
    QllItem items[QLL_MAX_KEYS];
    unsigned queue[QLL_MAX_KEYS];
    unsigned n, head, tail, advised, reading, thread_count;
    uint64_t generation;
    int initialized, stop, held, cancelled, failed, advising;
    QllStats stats;
} QwenLayerLoader;

/* All loader state and cache metadata use the cache mutex. Callback payloads
 * are inaccessible until publication; no callback re-enters either object. */
static int qll_epoch_locked(const QwenLayerLoader *w) {
    const QwenGlobalCache *c=w->cache;
    return w->identity==w && w->held && c->active && !c->active_lease &&
           w->generation==c->generation && pthread_equal(w->owner,c->owner);
}
static int qll_owner_locked(const QwenLayerLoader *w) {
    return qll_epoch_locked(w) && pthread_equal(w->owner,pthread_self());
}
static int qll_find_locked(const QwenLayerLoader *w,int key) {
    for(unsigned i=0;i<w->n;i++)if(w->items[i].key==key)return (int)i;
    return -1;
}
/* Also used by the CPU fixture to prove a stale generation cannot publish. */
static int qll_job_matches_locked(const QwenLayerLoader *w,unsigned i,uint64_t generation) {
    if(i>=w->n || !qll_epoch_locked(w) || generation!=w->generation)return 0;
    const QllItem *j=&w->items[i];const QwenGlobalCache *c=w->cache;
    if(j->generation!=generation || j->slot<0 || (size_t)j->slot>=c->slot_count ||
       j->key<0 || (size_t)j->key>=c->key_count)return 0;
    const QgcEntry *e=&c->entries[j->slot];
    return j->state==QLL_READING && e->refs==1 && e->state==QGC_LOADING &&
           e->key==j->key && c->index[j->key]==j->slot;
}
static void qll_loading_locked(QwenLayerLoader *w) {
    QwenGlobalCache *c=w->cache;
    c->loading=(w->reading || w->head<w->tail || w->advising) ? 1 : 0;
    c->stats.scratch=(size_t)w->reading*c->scratch_limit;
    if(c->stats.scratch>c->stats.peak_scratch)c->stats.peak_scratch=c->stats.scratch;
    if(w->reading>w->stats.peak_readers)w->stats.peak_readers=w->reading;
}
static void qll_cancel_locked(QwenLayerLoader *w) {
    QwenGlobalCache *c=w->cache;
    if(!w->cancelled){w->cancelled=1;w->stats.cancelled_reservations++;}
    while(w->head<w->tail) {
        QllItem *j=&w->items[w->queue[w->head++]];
        if(j->state!=QLL_QUEUED)abort();
        if(j->fresh) {
            if(c->stats.reserved<c->payload_bytes)abort();
            c->stats.reserved-=c->payload_bytes;
        }
        qgc_hide(c,j->slot);j->state=QLL_CANCELLED;w->stats.cancelled_records++;
    }
    qll_loading_locked(w);pthread_cond_broadcast(&c->changed);
}
static void *qll_worker(void *context) {
    QwenLayerLoader *w=context;QwenGlobalCache *c=w->cache;
    pthread_mutex_lock(&c->mutex);
    for(;;) {
        int do_advice=0;
        while(!w->stop) {
            if(w->held && (c->stopping || w->cancelled || w->failed) && w->head<w->tail)qll_cancel_locked(w);
            int enabled=c->advise!=NULL;
            if(w->held && !w->cancelled && !w->failed && !c->stopping) {
                if(enabled && !w->advising && w->advised<w->tail){do_advice=1;break;}
                if(w->head<(enabled?w->advised:w->tail))break;
            }
            if(pthread_cond_wait(&c->changed,&c->mutex))abort();
        }
        if(w->stop)break;
        if(!qll_epoch_locked(w))abort();
        if(do_advice) {
            unsigned first=w->advised,end=w->tail;w->advising=1;w->stats.advice_batches++;
            qll_loading_locked(w);pthread_cond_broadcast(&c->changed);
            for(unsigned p=first;p<end;p++) {
                if(c->stopping || w->cancelled || w->failed)break;
                int key=w->items[w->queue[p]].key;uint64_t start=qgc_now_ns();
                pthread_mutex_unlock(&c->mutex);c->advise(c->context,key);
                uint64_t elapsed=qgc_now_ns()-start;pthread_mutex_lock(&c->mutex);
                w->stats.advise_calls++;w->stats.advise_ns+=elapsed;
                if(!c->stopping && !w->cancelled && !w->failed) {
                    w->advised=p+1;
                    pthread_cond_broadcast(&c->changed);
                }
            }
            w->advising=0;
            if(c->stopping || w->cancelled || w->failed)qll_cancel_locked(w);
            else w->advised=end;
            qll_loading_locked(w);pthread_cond_broadcast(&c->changed);continue;
        }
        unsigned i=w->queue[w->head++];QllItem *j=&w->items[i];
        if(j->state!=QLL_QUEUED)abort();
        j->state=QLL_READING;w->reading++;qll_loading_locked(w);
        int key=j->key,slot=j->slot,fresh=j->fresh;
        uint64_t generation=j->generation,start=qgc_now_ns();
        void *payload=c->entries[slot].payload;
        pthread_cond_broadcast(&c->changed);
        pthread_mutex_unlock(&c->mutex);
        if(fresh)payload=c->allocate(c->context,(size_t)slot);
        int called=payload!=NULL;
        int ok=called && c->load(c->context,key,payload);
        uint64_t elapsed=qgc_now_ns()-start;
        pthread_mutex_lock(&c->mutex);
        /* Main cannot end or reuse this reservation while reading != 0.
         * A mismatch is an internal ownership violation, never permission
         * to publish a record into another reservation or silently continue. */
        if(!qll_job_matches_locked(w,i,generation))abort();
        QgcEntry *e=&c->entries[slot];
        if(fresh) {
            if(c->stats.reserved<c->payload_bytes)abort();
            c->stats.reserved-=c->payload_bytes;
            if(payload){e->payload=payload;c->stats.allocated+=c->payload_bytes;}
            else{c->stats.allocation_failures++;w->stats.allocation_failures++;}
        }
        c->stats.loads++;c->stats.load_ns+=elapsed;w->stats.callback_ns+=elapsed;
        if(called){w->stats.load_calls++;w->stats.load_bytes+=c->payload_bytes;}
        if(ok) {
            e->state=QGC_READY;j->state=QLL_READY;
            if(w->cancelled)w->stats.completed_after_cancel++;
        } else {
            c->stats.load_failures++;w->stats.load_failures++;
            qgc_hide(c,slot);j->state=QLL_FAILED;w->failed=1;
        }
        w->reading--;
        if(w->failed || c->stopping)qll_cancel_locked(w);
        qll_loading_locked(w);pthread_cond_broadcast(&c->changed);
    }
    pthread_mutex_unlock(&c->mutex);return NULL;
}

/* Cache mutex held. At least 256 retained slots guarantee a victim for every
 * expert of a layer; READY working-window hits pinned in place only add to it. */
static int qll_geometry_locked(const QwenGlobalCache *c) {
    return c->retained>=QLL_MAX_KEYS;
}
/* Also used by the graph before any trunk/state/KV work, so a small cache
 * selects the legacy schedule rather than failing a prefill. */
static int qll_cache_eligible(QwenGlobalCache *c) {
    if(!c || !c->initialized)return 0;
    pthread_mutex_lock(&c->mutex);int ok=qll_geometry_locked(c);pthread_mutex_unlock(&c->mutex);
    return ok;
}
/* Init once, or again only after destroy. No cache history is modified here. */
static int qll_init(QwenLayerLoader *w,QwenGlobalCache *c) {
    if(!w || !c || !c->initialized || c->scratch_limit>SIZE_MAX/QLL_READERS)return 0;
    if(!qll_cache_eligible(c))return 0;
    memset(w,0,sizeof(*w));w->cache=c;w->identity=w;
    pthread_attr_t attr;if(pthread_attr_init(&attr))return 0;
    size_t stack=QLL_STACK_BYTES,guard=0;
#ifdef PTHREAD_STACK_MIN
    if(stack<PTHREAD_STACK_MIN)stack=PTHREAD_STACK_MIN;
#endif
    if(qgc_thread_stack(&attr,stack,&guard)) {
        pthread_attr_destroy(&attr);memset(w,0,sizeof(*w));return 0;
    }
    w->initialized=1;
    for(;w->thread_count<QLL_READERS;w->thread_count++)
        if(pthread_create(&w->threads[w->thread_count],&attr,qll_worker,w))break;
    pthread_attr_destroy(&attr);
    if(w->thread_count!=QLL_READERS) {
        pthread_mutex_lock(&c->mutex);w->stop=1;pthread_cond_broadcast(&c->changed);pthread_mutex_unlock(&c->mutex);
        for(unsigned i=0;i<w->thread_count;i++)pthread_join(w->threads[i],NULL);
        memset(w,0,sizeof(*w));return 0;
    }
    w->stats.metadata_bytes=sizeof(*w);w->stats.stack_bytes=stack*QLL_READERS;
    w->stats.guard_bytes=guard*QLL_READERS;
    /* A conservative bound for the existing callback contract, not a new
     * allocation. The planar file callback uses the charged slot in place. */
    w->stats.callback_scratch_bound=c->scratch_limit*QLL_READERS;
    return 1;
}
static int qll_begin(QwenLayerLoader *w) {
    if(!w || !w->initialized || w->identity!=w)return 0;
    QwenGlobalCache *c=w->cache;pthread_mutex_lock(&c->mutex);
    if(w->held || w->stop || c->stopping || c->active || c->loading || c->first ||
       !qll_geometry_locked(c)) {
        pthread_mutex_unlock(&c->mutex);return 0;
    }
    for(size_t i=0;i<c->slot_count;i++)if(c->entries[i].refs || c->entries[i].state==QGC_LOADING) {
        pthread_mutex_unlock(&c->mutex);return 0;
    }
    w->n=w->head=w->tail=w->advised=w->reading=0;w->cancelled=w->failed=w->advising=0;
    w->owner=pthread_self();c->owner=w->owner;c->active=1;c->active_lease=NULL;
    if(!++c->generation)++c->generation;
    w->generation=c->generation;w->held=1;w->stats.reservations++;
    pthread_mutex_unlock(&c->mutex);return 1;
}
/* Choose only an unpinned retained entry. Ties use the same score/LRU ranking
 * as qgc, but this scope records and overrides qcp_admit rejection. */
static int qll_victim_locked(QwenLayerLoader *w,int key) {
    QwenGlobalCache *c=w->cache;int victim=-1;
    for(size_t i=0;i<c->retained;i++) {
        QgcEntry *e=&c->entries[i];if(e->refs || e->state==QGC_LOADING)continue;
        if(e->state==QGC_EMPTY)return (int)i;
        if(victim<0 || qcp_score(&c->policy,(size_t)e->key)<qcp_score(&c->policy,(size_t)c->entries[victim].key) ||
           (qcp_score(&c->policy,(size_t)e->key)==qcp_score(&c->policy,(size_t)c->entries[victim].key) &&
            e->used<c->entries[victim].used))victim=(int)i;
    }
    if(victim>=0 && !qcp_admit(&c->policy,(size_t)key,(size_t)c->entries[victim].key))w->stats.forced_admissions++;
    return victim;
}
static int qll_append(QwenLayerLoader *w,const int *keys,unsigned n) {
    if(!w || !w->initialized || w->identity!=w || (n && !keys) || n>QLL_MAX_KEYS)return 0;
    QwenGlobalCache *c=w->cache;pthread_mutex_lock(&c->mutex);
    if(!qll_owner_locked(w) || w->cancelled || w->failed || c->stopping) {
        pthread_mutex_unlock(&c->mutex);return 0;
    }
    int unique[QLL_MAX_KEYS];unsigned add=0,duplicates=0;
    for(unsigned i=0;i<n;i++) {
        if(keys[i]<0 || (size_t)keys[i]>=c->key_count) {pthread_mutex_unlock(&c->mutex);return 0;}
        unsigned j=0;while(j<add && unique[j]!=keys[i])j++;
        if(qll_find_locked(w,keys[i])>=0 || j<add){duplicates++;continue;}
        unique[add++]=keys[i];
    }
    if(add>QLL_MAX_KEYS-w->n){pthread_mutex_unlock(&c->mutex);return 0;}
    unsigned first=w->n;w->stats.duplicate_keys+=duplicates;
    /* Pin every newly discovered hit before any miss can evict one of them. */
    for(unsigned i=0;i<add;i++) {
        int key=unique[i],slot=c->index[key];QllItem *j=&w->items[w->n++];
        *j=(QllItem){.key=key,.slot=-1,.state=QLL_QUEUED,.generation=w->generation};
        qcp_touch(&c->policy,(size_t)key,1);w->stats.selected++;
        if(slot>=0 && c->entries[slot].state==QGC_READY) {
            QgcEntry *e=&c->entries[slot];if(e->refs)abort();
            e->refs=1;e->used=++c->clock;j->slot=slot;j->state=QLL_READY;
            c->stats.hits++;w->stats.hits++;
        }
    }
    for(unsigned i=first;i<w->n;i++) {
        QllItem *j=&w->items[i];if(j->slot>=0)continue;
        int slot=qll_victim_locked(w,j->key);
        if(slot<0)abort(); /* init capacity plus <=256 pins proves a victim. */
        QgcEntry *e=&c->entries[slot];int fresh=e->payload==NULL;
        if(fresh && c->stats.allocated+c->stats.reserved>c->budget-c->payload_bytes)abort();
        if(e->state==QGC_READY){c->stats.evictions++;w->stats.evictions++;}
        qgc_hide(c,slot);e->key=j->key;e->state=QGC_LOADING;e->refs=1;e->used=++c->clock;
        c->index[j->key]=slot;j->slot=slot;j->fresh=fresh;
        if(fresh)c->stats.reserved+=c->payload_bytes;
        size_t charge=c->stats.allocated+c->stats.reserved;
        if(charge>c->stats.peak_charge)c->stats.peak_charge=charge;
        c->stats.misses++;w->stats.misses++;w->queue[w->tail++]=i;
    }
    if(w->n>w->stats.peak_selected)w->stats.peak_selected=w->n;
    qll_loading_locked(w);pthread_cond_broadcast(&c->changed);
    pthread_mutex_unlock(&c->mutex);return 1;
}
/* Wait only for these already discovered keys. No extra refcount is needed:
 * their layer pins remain held until finish after terminal GPU completion. */
static int qll_wait(QwenLayerLoader *w,const int *keys,unsigned n,void **payloads) {
    if(!w || !w->initialized || w->identity!=w || !keys || !payloads || !n || n>QGC_MAX_GROUP)return 0;
    for(unsigned i=0;i<n;i++)payloads[i]=NULL;
    QwenGlobalCache *c=w->cache;pthread_mutex_lock(&c->mutex);
    if(!qll_owner_locked(w) || n>c->group_cap){pthread_mutex_unlock(&c->mutex);return 0;}
    int selected[QGC_MAX_GROUP];
    for(unsigned i=0;i<n;i++)if((selected[i]=qll_find_locked(w,keys[i]))<0) {
        pthread_mutex_unlock(&c->mutex);return 0;
    }
    for(;;) {
        if(w->failed || w->cancelled || c->stopping) {
            if(c->stopping)qll_cancel_locked(w);
            pthread_mutex_unlock(&c->mutex);return 0;
        }
        unsigned ready=0;for(unsigned i=0;i<n;i++)ready+=w->items[selected[i]].state==QLL_READY;
        if(ready==n)break;
        uint64_t start=qgc_now_ns();w->stats.waits++;
        if(pthread_cond_wait(&c->changed,&c->mutex))abort();
        w->stats.wait_ns+=qgc_now_ns()-start;
    }
    for(unsigned i=0;i<n;i++) {
        QllItem *j=&w->items[selected[i]];QgcEntry *e=&c->entries[j->slot];
        if(e->key!=j->key || e->state!=QGC_READY || !e->refs || !e->payload)abort();
        payloads[i]=e->payload;
    }
    pthread_mutex_unlock(&c->mutex);return 1;
}
/* 1 when every key of the group is loaded, so wait would return at once. */
static int qll_ready(QwenLayerLoader *w,const int *keys,unsigned n) {
    if(!w || !w->initialized || w->identity!=w || !keys || !n || n>QGC_MAX_GROUP)return 0;
    QwenGlobalCache *c=w->cache;pthread_mutex_lock(&c->mutex);
    int ok=qll_owner_locked(w) && !w->failed && !w->cancelled && !c->stopping;
    for(unsigned i=0;ok && i<n;i++){int s=qll_find_locked(w,keys[i]);ok=s>=0 && w->items[s].state==QLL_READY;}
    pthread_mutex_unlock(&c->mutex);return ok;
}
/* GPU terminal is a caller obligation. finish returns lifecycle success, even
 * when wait already reported a load failure. A wrong owner is refused without
 * mutation. Idle finish is a no-op; cancelled callbacks are drained, not killed. */
static int qll_finish(QwenLayerLoader *w,int cancel) {
    if(!w || !w->initialized)return 1;
    if(w->identity!=w)return 0;
    QwenGlobalCache *c=w->cache;pthread_mutex_lock(&c->mutex);
    if(!w->held){pthread_mutex_unlock(&c->mutex);return 1;}
    if(!qll_owner_locked(w)){pthread_mutex_unlock(&c->mutex);return 0;}
    if(cancel || w->failed || c->stopping)qll_cancel_locked(w);
    uint64_t start=qgc_now_ns();
    while(w->reading || w->head<w->tail || w->advising)
        if(pthread_cond_wait(&c->changed,&c->mutex))abort();
    w->stats.drain_ns+=qgc_now_ns()-start;
    for(unsigned i=0;i<w->n;i++) {
        QgcEntry *e=&c->entries[w->items[i].slot];
        if(e->refs!=1 || e->state==QGC_LOADING)abort();
        e->refs=0;
    }
    if(c->stopping)for(size_t i=c->retained;i<c->slot_count;i++)qgc_hide(c,(int)i);
    c->active=0;c->active_lease=NULL;c->loading=0;c->stats.scratch=0;w->held=0;
    pthread_cond_broadcast(&c->changed);pthread_mutex_unlock(&c->mutex);return 1;
}
static int qll_held(QwenLayerLoader *w) {
    if(!w || !w->initialized || w->identity!=w)return 0;
    pthread_mutex_lock(&w->cache->mutex);int held=w->held;pthread_mutex_unlock(&w->cache->mutex);return held;
}
static QllStats qll_snapshot(QwenLayerLoader *w) {
    QllStats s={0};if(!w || !w->initialized || w->identity!=w)return s;
    pthread_mutex_lock(&w->cache->mutex);s=w->stats;s.held=w->held;
    s.queued=w->held?w->tail-w->head:0;s.reading=w->reading;s.selected_now=w->held?w->n:0;
    s.advising=(unsigned)w->advising;
    pthread_mutex_unlock(&w->cache->mutex);return s;
}
static int qll_destroy(QwenLayerLoader *w) {
    if(!w || !w->initialized)return 1;
    if(!qll_finish(w,1))return 0;
    QwenGlobalCache *c=w->cache;pthread_mutex_lock(&c->mutex);
    w->stop=1;pthread_cond_broadcast(&c->changed);pthread_mutex_unlock(&c->mutex);
    for(unsigned i=0;i<w->thread_count;i++)if(pthread_join(w->threads[i],NULL))abort();
    memset(w,0,sizeof(*w));return 1;
}
#endif
