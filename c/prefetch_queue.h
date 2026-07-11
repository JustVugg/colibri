#ifndef COLIBRI_PREFETCH_QUEUE_H
#define COLIBRI_PREFETCH_QUEUE_H

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define COLI_PREFETCH_CAP 4096
typedef struct { int layer, expert; } ColiPrefetchItem;
typedef struct {
    pthread_mutex_t mu; pthread_cond_t ready;
    ColiPrefetchItem items[COLI_PREFETCH_CAP];
    unsigned read, write; int stopped, layers, experts; unsigned char *pending;
} ColiPrefetchQueue;

static inline int coli_prefetch_init(ColiPrefetchQueue *q, int layers, int experts){
    if(!q||layers<1||experts<1) return 0;
    memset(q,0,sizeof(*q)); q->layers=layers; q->experts=experts;
    q->pending=calloc((size_t)layers*experts,1); if(!q->pending) return 0;
    if(pthread_mutex_init(&q->mu,NULL)||pthread_cond_init(&q->ready,NULL)){
        free(q->pending); q->pending=NULL; return 0;
    }
    return 1;
}

/* 1 queued, 0 duplicate, -1 full/stopped/invalid. */
static inline int coli_prefetch_push(ColiPrefetchQueue *q, int layer, int expert){
    if(!q||layer<0||layer>=q->layers||expert<0||expert>=q->experts) return -1;
    pthread_mutex_lock(&q->mu);
    size_t key=(size_t)layer*q->experts+expert;
    int rc;
    if(q->stopped||q->write-q->read>=COLI_PREFETCH_CAP) rc=-1;
    else if(q->pending[key]) rc=0;
    else { q->pending[key]=1; q->items[q->write++&(COLI_PREFETCH_CAP-1)]=(ColiPrefetchItem){layer,expert};
           pthread_cond_signal(&q->ready); rc=1; }
    pthread_mutex_unlock(&q->mu); return rc;
}

static inline int coli_prefetch_pop(ColiPrefetchQueue *q, ColiPrefetchItem *item){
    pthread_mutex_lock(&q->mu);
    while(q->read==q->write&&!q->stopped) pthread_cond_wait(&q->ready,&q->mu);
    if(q->stopped){ pthread_mutex_unlock(&q->mu); return 0; }
    *item=q->items[q->read++&(COLI_PREFETCH_CAP-1)];
    pthread_mutex_unlock(&q->mu); return 1;
}

static inline void coli_prefetch_done(ColiPrefetchQueue *q, ColiPrefetchItem item){
    pthread_mutex_lock(&q->mu);
    q->pending[(size_t)item.layer*q->experts+item.expert]=0;
    pthread_mutex_unlock(&q->mu);
}

static inline void coli_prefetch_stop(ColiPrefetchQueue *q){
    pthread_mutex_lock(&q->mu); q->stopped=1; pthread_cond_broadcast(&q->ready);
    pthread_mutex_unlock(&q->mu);
}

static inline void coli_prefetch_destroy(ColiPrefetchQueue *q){
    pthread_cond_destroy(&q->ready); pthread_mutex_destroy(&q->mu);
    free(q->pending); q->pending=NULL;
}

#endif
