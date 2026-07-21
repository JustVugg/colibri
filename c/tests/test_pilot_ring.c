/* Concurrency test for the SPMC pilot ring claim (pilot_ring_claim): with N worker
 * threads draining a single-producer ring, every enqueued item must be claimed by
 * EXACTLY ONE worker -- none lost, none double-claimed, none torn by ring wrap.
 *
 * This gates the new lock-free primitive without needing a model (the slot
 * reservation logic in pilot_realload is exercised end-to-end by the token-exact
 * oracle; here we prove the ring claim itself). Includes colibri.c for the real
 * pilot_ring_claim / pilot_q / pilot_w / pilot_r (same pattern as the other tests).
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#include <pthread.h>
#include <stdint.h>

#define NITEMS   300000
#define NTHREADS 8
static _Atomic int seen[NITEMS];
static _Atomic long total_claimed = 0;
static _Atomic int  torn = 0;
static _Atomic int  producer_done = 0;

static void *consumer(void *arg){
    (void)arg;
    for(;;){
        int v, e;
        if(pilot_ring_claim(&v, &e)){
            if(v < 0 || v >= NITEMS || e != v){          /* out-of-range or mismatched => torn read leaked through */
                atomic_store_explicit(&torn, 1, memory_order_relaxed);
                continue;
            }
            atomic_fetch_add_explicit(&seen[v], 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&total_claimed, 1, memory_order_relaxed);
        } else if(atomic_load_explicit(&producer_done, memory_order_acquire)){
            unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
            unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
            if(r == w) return NULL;                       /* producer finished AND ring drained */
        }
    }
}

int main(void){
    for(int i=0;i<NITEMS;i++) atomic_store_explicit(&seen[i],0,memory_order_relaxed);
    pthread_t th[NTHREADS];
    for(int i=0;i<NTHREADS;i++) pthread_create(&th[i],NULL,consumer,NULL);

    for(int i=0;i<NITEMS;i++){                            /* single producer: only pilot_w */
        for(;;){
            unsigned w = __atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
            unsigned r = __atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
            if(w - r < 4096){                            /* ring has space */
                atomic_store_explicit(&pilot_q[w & 4095].l, i, memory_order_relaxed);
                atomic_store_explicit(&pilot_q[w & 4095].e, i, memory_order_relaxed);
                __atomic_store_n(&pilot_w, w+1, __ATOMIC_RELEASE);
                break;
            }
            /* full: let consumers drain (they advance pilot_r via CAS) */
        }
    }
    atomic_store_explicit(&producer_done, 1, memory_order_release);
    for(int i=0;i<NTHREADS;i++) pthread_join(th[i],NULL);

    int fail=0;
    if(atomic_load_explicit(&torn,memory_order_relaxed)){
        fprintf(stderr,"FAIL: torn/overwritten ring slot observed\n"); fail=1; }
    long tc=atomic_load_explicit(&total_claimed,memory_order_relaxed);
    if(tc!=NITEMS){ fprintf(stderr,"FAIL: total claimed %ld != %d\n",tc,NITEMS); fail=1; }
    for(int i=0;i<NITEMS && !fail;i++){
        int s=atomic_load_explicit(&seen[i],memory_order_relaxed);
        if(s!=1){ fprintf(stderr,"FAIL: item %d claimed %d times (want exactly 1)\n",i,s); fail=1; }
    }
    if(!fail) printf("test_pilot_ring: ok — %d items across %d workers, each claimed exactly once\n",NITEMS,NTHREADS);
    return fail;
}
