#ifndef COLI_URING_H
#define COLI_URING_H

/* Minimal Linux io_uring reader.  Colibri owns the ring from one thread, queues
 * a batch of positioned reads, and reaps CQEs without a userspace spin loop. */
#ifdef __linux__

#include <errno.h>
#include <linux/io_uring.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MAP_POPULATE
#define MAP_POPULATE 0
#endif

typedef struct {
    int fd;
    struct io_uring_params p;
    void *sq_map, *cq_map, *sqes_map;
    size_t sq_map_sz, cq_map_sz, sqes_map_sz;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_entries, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask, *cq_entries;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
} ColiUring;

static inline unsigned coli_uring_load_acquire(unsigned *p){
    return __atomic_load_n(p,__ATOMIC_ACQUIRE);
}
static inline void coli_uring_store_release(unsigned *p,unsigned v){
    __atomic_store_n(p,v,__ATOMIC_RELEASE);
}

static inline void coli_uring_close(ColiUring *r){
    if(!r) return;
    if(r->sqes_map && r->sqes_map!=MAP_FAILED) munmap(r->sqes_map,r->sqes_map_sz);
    if(r->cq_map && r->cq_map!=MAP_FAILED && r->cq_map!=r->sq_map) munmap(r->cq_map,r->cq_map_sz);
    if(r->sq_map && r->sq_map!=MAP_FAILED) munmap(r->sq_map,r->sq_map_sz);
    if(r->fd>=0) close(r->fd);
    memset(r,0,sizeof(*r)); r->fd=-1;
}

static inline int coli_uring_init(ColiUring *r,unsigned entries){
    memset(r,0,sizeof(*r)); r->fd=-1;
    r->fd=(int)syscall(SYS_io_uring_setup,entries,&r->p);
    if(r->fd<0) return -1;

    r->sq_map_sz=r->p.sq_off.array+r->p.sq_entries*sizeof(unsigned);
    r->cq_map_sz=r->p.cq_off.cqes+r->p.cq_entries*sizeof(struct io_uring_cqe);
    if(r->p.features&IORING_FEAT_SINGLE_MMAP){
        size_t n=r->sq_map_sz>r->cq_map_sz?r->sq_map_sz:r->cq_map_sz;
        r->sq_map=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,r->fd,IORING_OFF_SQ_RING);
        if(r->sq_map==MAP_FAILED){ r->sq_map=NULL; coli_uring_close(r); return -1; }
        r->sq_map_sz=n; r->cq_map=r->sq_map; r->cq_map_sz=n;
    }else{
        r->sq_map=mmap(NULL,r->sq_map_sz,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,r->fd,IORING_OFF_SQ_RING);
        if(r->sq_map==MAP_FAILED){ r->sq_map=NULL; coli_uring_close(r); return -1; }
        r->cq_map=mmap(NULL,r->cq_map_sz,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,r->fd,IORING_OFF_CQ_RING);
        if(r->cq_map==MAP_FAILED){ r->cq_map=NULL; coli_uring_close(r); return -1; }
    }
    r->sqes_map_sz=r->p.sq_entries*sizeof(struct io_uring_sqe);
    r->sqes_map=mmap(NULL,r->sqes_map_sz,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,r->fd,IORING_OFF_SQES);
    if(r->sqes_map==MAP_FAILED){ r->sqes_map=NULL; coli_uring_close(r); return -1; }

    char *sq=(char*)r->sq_map,*cq=(char*)r->cq_map;
    r->sq_head=(unsigned*)(sq+r->p.sq_off.head);
    r->sq_tail=(unsigned*)(sq+r->p.sq_off.tail);
    r->sq_mask=(unsigned*)(sq+r->p.sq_off.ring_mask);
    r->sq_entries=(unsigned*)(sq+r->p.sq_off.ring_entries);
    r->sq_array=(unsigned*)(sq+r->p.sq_off.array);
    r->cq_head=(unsigned*)(cq+r->p.cq_off.head);
    r->cq_tail=(unsigned*)(cq+r->p.cq_off.tail);
    r->cq_mask=(unsigned*)(cq+r->p.cq_off.ring_mask);
    r->cq_entries=(unsigned*)(cq+r->p.cq_off.ring_entries);
    r->sqes=(struct io_uring_sqe*)r->sqes_map;
    r->cqes=(struct io_uring_cqe*)(cq+r->p.cq_off.cqes);
    return 0;
}

static inline int coli_uring_set_workers(ColiUring *r,unsigned workers){
    unsigned limits[2]={workers,workers};          /* bounded, unbounded io-wq workers */
    return (int)syscall(SYS_io_uring_register,r->fd,IORING_REGISTER_IOWQ_MAX_WORKERS,limits,2);
}

static inline int coli_uring_prep_read(ColiUring *r,int fd,void *buf,size_t len,
                                       int64_t off,uint64_t user_data){
    if(!len || len>UINT32_MAX){ errno=EINVAL; return -1; }
    unsigned head=coli_uring_load_acquire(r->sq_head);
    unsigned tail=__atomic_load_n(r->sq_tail,__ATOMIC_RELAXED);
    if(tail-head>=*r->sq_entries){ errno=EAGAIN; return -1; }
    unsigned idx=tail&*r->sq_mask;
    struct io_uring_sqe *sqe=&r->sqes[idx];
    memset(sqe,0,sizeof(*sqe));
    sqe->opcode=IORING_OP_READ;
    /* Cold regular-file reads are allowed to execute inline during
     * io_uring_enter() unless forced async.  That serializes the submitter on
     * filesystems without native nonblocking buffered reads and destroys the
     * intended I/O/compute overlap.  io-wq gives the ring a real bounded worker
     * pool while CQEs retain completion ordering/ownership here. */
    sqe->flags=IOSQE_ASYNC;
    sqe->fd=fd;
    sqe->off=(uint64_t)off;
    sqe->addr=(uint64_t)(uintptr_t)buf;
    sqe->len=(uint32_t)len;
    sqe->user_data=user_data;
    r->sq_array[idx]=idx;
    coli_uring_store_release(r->sq_tail,tail+1);
    return 0;
}

static inline int coli_uring_enter(ColiUring *r,unsigned min_complete){
    for(;;){
        unsigned head=coli_uring_load_acquire(r->sq_head);
        unsigned tail=coli_uring_load_acquire(r->sq_tail);
        unsigned submit=tail-head;
        unsigned flags=min_complete?IORING_ENTER_GETEVENTS:0;
        int n=(int)syscall(SYS_io_uring_enter,r->fd,submit,min_complete,flags,NULL,0);
        if(n>=0) return n;
        if(errno!=EINTR) return -1;
    }
}

static inline int coli_uring_peek(ColiUring *r,struct io_uring_cqe *out){
    unsigned head=__atomic_load_n(r->cq_head,__ATOMIC_RELAXED);
    unsigned tail=coli_uring_load_acquire(r->cq_tail);
    if(head==tail) return 0;
    *out=r->cqes[head&*r->cq_mask];
    coli_uring_store_release(r->cq_head,head+1);
    return 1;
}

#endif /* __linux__ */
#endif /* COLI_URING_H */
