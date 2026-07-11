#include <assert.h>
#include <stdio.h>
#include "../prefetch_queue.h"

int main(void){
    ColiPrefetchQueue q; ColiPrefetchItem item;
    assert(coli_prefetch_init(&q,3,4));
    assert(coli_prefetch_push(&q,1,2)==1);
    assert(coli_prefetch_push(&q,1,2)==0);
    assert(coli_prefetch_pop(&q,&item)==1 && item.layer==1 && item.expert==2);
    assert(coli_prefetch_push(&q,1,2)==0);
    coli_prefetch_done(&q,item);
    assert(coli_prefetch_push(&q,1,2)==1);
    assert(coli_prefetch_push(&q,3,0)==-1);
    assert(coli_prefetch_pop(&q,&item)==1); coli_prefetch_done(&q,item);
    coli_prefetch_stop(&q);
    assert(coli_prefetch_pop(&q,&item)==0);
    assert(coli_prefetch_push(&q,0,0)==-1);
    coli_prefetch_destroy(&q);
    puts("prefetch queue tests: ok");
    return 0;
}
