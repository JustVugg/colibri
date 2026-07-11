#include <stdio.h>
#include "../cpu_pool.h"

typedef struct { int *seen; } Ctx;
static void mark(void *opaque,int begin,int end){
    Ctx *ctx=opaque; for(int i=begin;i<end;i++) ctx->seen[i]++;
}
int main(void){
    CpuPool pool={0}; int seen[257]={0}; Ctx ctx={seen};
    if(!cpu_pool_init(&pool,7)) return 1;
    for(int pass=0;pass<1000;pass++) cpu_pool_for(&pool,257,mark,&ctx);
    cpu_pool_destroy(&pool);
    for(int i=0;i<257;i++) if(seen[i]!=1000){ fprintf(stderr,"pool index %d ran %d times\n",i,seen[i]); return 1; }
    puts("cpu pool tests: ok"); return 0;
}
