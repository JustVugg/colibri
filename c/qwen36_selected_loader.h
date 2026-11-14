/* Bounded selected-expert loader. The worker owns the QgcLease at one address
 * from acquire through release. The inference thread only borrows published
 * payload pointers; finish/cancel is legal after its GPU work is terminal.
 * No speculative admission, lease transfer or second cache is involved. */
#ifndef QWEN36_SELECTED_LOADER_H
#define QWEN36_SELECTED_LOADER_H
#include "qwen36_global_cache.h"
typedef struct {
    QwenGlobalCache *cache;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    int initialized,started,stop,pending,ready,release,ok,cancel,hits_ready;
    unsigned n;int keys[QGC_MAX_GROUP];void *payload[QGC_MAX_GROUP],*hits[QGC_MAX_GROUP];
    size_t stack_bytes,guard_bytes;
    double acquire_s;
} QwenSelectedLoader;
/* qgc_acquire_early's hand-over, on the worker thread: the hits for qsl_wait_hits. */
static void qsl_hits(void *context,void *const *payload,unsigned n) {
    QwenSelectedLoader *w=context;
    pthread_mutex_lock(&w->mutex);
    memcpy(w->hits,payload,n*sizeof(void*));w->hits_ready=1;
    pthread_cond_broadcast(&w->changed);pthread_mutex_unlock(&w->mutex);
}
static void *qsl_worker(void *context) {
    QwenSelectedLoader *w=context;
    pthread_mutex_lock(&w->mutex);
    for(;;) {
        while(!w->pending&&!w->stop)pthread_cond_wait(&w->changed,&w->mutex);
        if(w->stop)break;
        pthread_mutex_unlock(&w->mutex);
        QgcLease lease={0};uint64_t start=qgc_now_ns();
        int ok=qgc_acquire_early(w->cache,w->keys,w->n,&w->cancel,&lease,qsl_hits,w);
        double seconds=(qgc_now_ns()-start)*1e-9;
        void *payload[QGC_MAX_GROUP]={0};
        if(ok)for(unsigned i=0;i<w->n;i++)payload[i]=qgc_payload(w->cache,&lease,i);
        pthread_mutex_lock(&w->mutex);
        w->ok=ok;w->acquire_s=seconds;memcpy(w->payload,payload,sizeof payload);w->ready=1;
        pthread_cond_broadcast(&w->changed);
        while(!w->release&&!w->stop)pthread_cond_wait(&w->changed,&w->mutex);
        pthread_mutex_unlock(&w->mutex);
        if(lease.held&&!qgc_release_group(w->cache,&lease))abort();
        pthread_mutex_lock(&w->mutex);
        memset(w->payload,0,sizeof w->payload);memset(w->hits,0,sizeof w->hits);
        w->pending=w->ready=w->release=w->hits_ready=0;
        pthread_cond_broadcast(&w->changed);
    }
    pthread_mutex_unlock(&w->mutex);return NULL;
}
static int qsl_init(QwenSelectedLoader *w,QwenGlobalCache *cache) {
    memset(w,0,sizeof(*w));if(!cache||!cache->initialized)return 0;w->cache=cache;
    if(pthread_mutex_init(&w->mutex,NULL))return 0;
    if(pthread_cond_init(&w->changed,NULL)){pthread_mutex_destroy(&w->mutex);return 0;}
    w->initialized=1;pthread_attr_t attr;
    if(pthread_attr_init(&attr))goto failed;
    w->stack_bytes=256u*1024;
    int error=qgc_thread_stack(&attr,w->stack_bytes,&w->guard_bytes);
    if(!error)error=pthread_create(&w->thread,&attr,qsl_worker,w);
    pthread_attr_destroy(&attr);
    if(!error){w->started=1;return 1;}
failed:
    pthread_cond_destroy(&w->changed);pthread_mutex_destroy(&w->mutex);memset(w,0,sizeof(*w));return 0;
}
static int qsl_start(QwenSelectedLoader *w,const int *keys,unsigned n) {
    if(!w||!w->started||!keys||!n||n>w->cache->group_cap)return 0;
    pthread_mutex_lock(&w->mutex);
    if(w->pending||w->stop){pthread_mutex_unlock(&w->mutex);return 0;}
    memcpy(w->keys,keys,n*sizeof(int));w->n=n;w->ok=0;w->acquire_s=0;
    __atomic_store_n(&w->cancel,0,__ATOMIC_RELEASE);w->pending=1;
    pthread_cond_broadcast(&w->changed);pthread_mutex_unlock(&w->mutex);return 1;
}
/* A timed condition wait lets the inference thread observe serving cancellation
 * while disk I/O completes. Cancellation cannot reclaim an in-progress read. */
static int qsl_wait(QwenSelectedLoader *w,int (*cancelled)(void *),void *context,void **payload) {
    pthread_mutex_lock(&w->mutex);
    while(w->pending&&!w->ready) {
        pthread_mutex_unlock(&w->mutex);int cancel=cancelled&&cancelled(context);pthread_mutex_lock(&w->mutex);
        if(cancel){pthread_mutex_unlock(&w->mutex);return 0;}
        struct timespec until;clock_gettime(CLOCK_REALTIME,&until);until.tv_nsec+=20000000;
        if(until.tv_nsec>=1000000000){until.tv_sec++;until.tv_nsec-=1000000000;}
        if(!w->ready)pthread_cond_timedwait(&w->changed,&w->mutex,&until);
    }
    int ok=w->pending&&w->ready&&w->ok;
    if(ok)memcpy(payload,w->payload,w->n*sizeof(void*));
    pthread_mutex_unlock(&w->mutex);return ok;
}
/* The group's hits as soon as the worker has borrowed them (NULL for a miss),
 * while the misses may still be reading: the number of hits, 0 on failure. The
 * payloads stay valid until qsl_finish, as qsl_wait's, even if the group then
 * fails: the worker keeps the lease until then. */
static unsigned qsl_wait_hits(QwenSelectedLoader *w,void **payload) {
    pthread_mutex_lock(&w->mutex);
    while(w->pending&&!w->hits_ready&&!w->ready)pthread_cond_wait(&w->changed,&w->mutex);
    unsigned found=0;
    if(w->pending&&w->hits_ready)
        for(unsigned i=0;i<w->n;i++){payload[i]=w->hits[i];found+=w->hits[i]!=NULL;}
    pthread_mutex_unlock(&w->mutex);return found;
}
/* On abort, stop further loads and drain the current read; on success, keep the
 * same owner alive through compute, then release on the worker thread. */
static void qsl_finish(QwenSelectedLoader *w,int cancel) {
    if(!w||!w->started)return;
    if(cancel){__atomic_store_n(&w->cancel,1,__ATOMIC_RELEASE);qgc_wake(w->cache);}
    pthread_mutex_lock(&w->mutex);w->release=1;pthread_cond_broadcast(&w->changed);
    while(w->pending)pthread_cond_wait(&w->changed,&w->mutex);
    w->release=0;pthread_mutex_unlock(&w->mutex);
}
static void qsl_destroy(QwenSelectedLoader *w) {
    if(!w||!w->initialized)return;
    qsl_finish(w,1);pthread_mutex_lock(&w->mutex);w->stop=1;
    pthread_cond_broadcast(&w->changed);pthread_mutex_unlock(&w->mutex);
    if(w->started&&pthread_join(w->thread,NULL))abort();
    pthread_cond_destroy(&w->changed);pthread_mutex_destroy(&w->mutex);memset(w,0,sizeof(*w));
}
#endif
