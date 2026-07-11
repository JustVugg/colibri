#ifndef COLIBRI_CPU_POOL_H
#define COLIBRI_CPU_POOL_H

#include <pthread.h>
#include <stdlib.h>

typedef void (*CpuPoolFn)(void *ctx, int begin, int end);

typedef struct CpuPool CpuPool;
typedef struct { CpuPool *pool; int id; } CpuPoolWorker;
struct CpuPool {
    pthread_mutex_t mu;
    pthread_cond_t work, done;
    pthread_t *threads;
    CpuPoolWorker *workers;
    CpuPoolFn fn;
    void *ctx;
    int count, items, completed, generation, stop;
};

static void *cpu_pool_worker(void *opaque){
    CpuPoolWorker *w=opaque; CpuPool *p=w->pool; int seen=0;
    pthread_mutex_lock(&p->mu);
    for(;;){
        while(!p->stop&&seen==p->generation) pthread_cond_wait(&p->work,&p->mu);
        if(p->stop) break;
        seen=p->generation; CpuPoolFn fn=p->fn; void *ctx=p->ctx; int n=p->items;
        int begin=(int)((long long)n*w->id/p->count);
        int end=(int)((long long)n*(w->id+1)/p->count);
        pthread_mutex_unlock(&p->mu);
        if(begin<end) fn(ctx,begin,end);
        pthread_mutex_lock(&p->mu);
        if(++p->completed==p->count) pthread_cond_signal(&p->done);
    }
    pthread_mutex_unlock(&p->mu); return NULL;
}

static int cpu_pool_init(CpuPool *p,int count){
    if(!p||count<1) return 0;
    *p=(CpuPool){0}; p->count=count;
    if(pthread_mutex_init(&p->mu,NULL)||pthread_cond_init(&p->work,NULL)||
       pthread_cond_init(&p->done,NULL)) return 0;
    p->threads=calloc((size_t)count,sizeof(*p->threads));
    p->workers=calloc((size_t)count,sizeof(*p->workers));
    if(!p->threads||!p->workers) return 0;
    for(int i=0;i<count;i++){
        p->workers[i]=(CpuPoolWorker){p,i};
        if(pthread_create(&p->threads[i],NULL,cpu_pool_worker,&p->workers[i])){
            pthread_mutex_lock(&p->mu); p->stop=1; pthread_cond_broadcast(&p->work); pthread_mutex_unlock(&p->mu);
            for(int j=0;j<i;j++) pthread_join(p->threads[j],NULL);
            return 0;
        }
    }
    return 1;
}

static void cpu_pool_for(CpuPool *p,int items,CpuPoolFn fn,void *ctx){
    if(!p||!p->threads||items<1||!fn) return;
    pthread_mutex_lock(&p->mu);
    p->items=items; p->fn=fn; p->ctx=ctx; p->completed=0; p->generation++;
    pthread_cond_broadcast(&p->work);
    while(p->completed<p->count) pthread_cond_wait(&p->done,&p->mu);
    pthread_mutex_unlock(&p->mu);
}

static void cpu_pool_destroy(CpuPool *p){
    if(!p||!p->threads) return;
    pthread_mutex_lock(&p->mu); p->stop=1; pthread_cond_broadcast(&p->work); pthread_mutex_unlock(&p->mu);
    for(int i=0;i<p->count;i++) pthread_join(p->threads[i],NULL);
    free(p->threads); free(p->workers); p->threads=NULL; p->workers=NULL;
    pthread_cond_destroy(&p->work); pthread_cond_destroy(&p->done); pthread_mutex_destroy(&p->mu);
}

#endif
