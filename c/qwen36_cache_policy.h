/* Demand-frequency admission with history independent of resident slots.
 * The cache manager serializes calls and chooses an eligible victim; this
 * policy neither owns weights nor changes borrow/eviction lifetimes.
 */
#ifndef COLI_QWEN36_CACHE_POLICY_H
#define COLI_QWEN36_CACHE_POLICY_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t epoch;
    uint32_t frequency, pending;
} QwenCacheHistory;

typedef struct {
    QwenCacheHistory *history;
    size_t key_count;
    uint64_t age_interval, references, epoch;
} QwenCachePolicy;

/* Initialize once, then free before reinitializing. The single allocation is
 * fixed by the model's key count; requests and misses cannot grow it. */
static inline int qcp_init(QwenCachePolicy *p,size_t key_count,uint64_t age_interval) {
    if(!p)return 0;
    memset(p,0,sizeof(*p));
    if(!key_count || !age_interval || key_count>SIZE_MAX/sizeof(*p->history))return 0;
    p->history=calloc(key_count,sizeof(*p->history));
    if(!p->history)return 0;
    p->key_count=key_count;p->age_interval=age_interval;return 1;
}
static inline void qcp_free(QwenCachePolicy *p) {
    if(p){free(p->history);memset(p,0,sizeof(*p));}
}
/* Exact allocated history bytes, excluding the embedded policy descriptor. */
static inline size_t qcp_payload_bytes(const QwenCachePolicy *p) {
    return p && p->history ? p->key_count*sizeof(*p->history) : 0;
}
static inline uint32_t qcp_score(const QwenCachePolicy *p,size_t key) {
    if(!p || !p->history || key>=p->key_count)return 0;
    const QwenCacheHistory *h=&p->history[key];
    uint64_t age=p->epoch-h->epoch;
    if(!age)return h->frequency;
    /* Only completed epochs inform admission. Counting the current reference
     * here gives the latest scan miss a rounding advantage at aging boundaries
     * and can rotate an otherwise equally frequent retained set. */
    uint64_t completed=(uint64_t)(h->frequency>>1)+h->pending;
    uint32_t score=completed>UINT32_MAX ? UINT32_MAX : (uint32_t)completed;
    return age>32 ? 0 : score>>(unsigned)(age-1);
}
static inline void qcp_advance(QwenCachePolicy *p) {
    if(p->references<p->age_interval-1){p->references++;return;}
    p->references=0;
    if(p->epoch==UINT64_MAX) {
        /* Rare wrap: materialize scores before rebasing. This preserves the
         * pending one-epoch decay without a wrapping subtraction or lost age. */
        for(size_t i=0;i<p->key_count;i++) {
            uint32_t score=qcp_score(p,i);
            if(p->history[i].epoch!=p->epoch)p->history[i].pending=0;
            p->history[i].frequency=score;
            p->history[i].epoch=0;
        }
        p->epoch=0;
    }
    p->epoch++;
}
/* Call exactly once for each requested key, whether hit, admitted or bypassed.
 * Prefetch (demand==0) changes neither frequency nor the demand-reference clock.
 * Return the completed-epoch retention score, not the current pending count.
 * At epoch completion, old evidence halves and that epoch's demands are added.
 * In the initial epoch all scores are zero: empty slots can fill, ties bypass. */
static inline uint32_t qcp_touch(QwenCachePolicy *p,size_t key,int demand) {
    if(!p || !p->history || key>=p->key_count)return 0;
    if(!demand)return qcp_score(p,key);
    QwenCacheHistory *h=&p->history[key];
    if(h->epoch!=p->epoch) {
        h->frequency=qcp_score(p,key);h->pending=0;h->epoch=p->epoch;
    }
    if(h->pending!=UINT32_MAX)h->pending++;
    qcp_advance(p);
    return qcp_score(p,key);
}
/* Fill empty retained slots in the manager. Replacement is strictly greater:
 * equally frequent scan entries bypass retention instead of cycling the cache. */
static inline int qcp_admit(const QwenCachePolicy *p,size_t candidate,size_t victim) {
    return p && p->history && candidate<p->key_count && victim<p->key_count &&
           qcp_score(p,candidate)>qcp_score(p,victim);
}
#endif
