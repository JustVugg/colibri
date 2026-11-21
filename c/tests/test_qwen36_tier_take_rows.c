/* Host fake backend: borrowed outputs may be recycled once issue_open closes. */
#include <assert.h>
#define coli_cuda_expert_group_take fixture_take
#include "qwen36_fake_cuda.h"
#undef coli_cuda_expert_group_take
static int failed=-1,calls[2];
static float device_rows[2][6];
const float *coli_cuda_expert_group_take(int device){
    assert(device>=0&&device<2);calls[device]++;
    return device==failed?NULL:device_rows[device];
}
#include "../qwen36_tier.c"

static void prepare(void){
    G.on=1;G.ndev=2;G.D=3;G.dev[0]=0;G.dev[1]=1;
    G.issue_open=1;G.is_cnt[0]=1;G.is_cnt[1]=2;
    G.is_k[0][0]=2;G.is_k[1][0]=0;G.is_k[1][1]=3;
    float data[2][6]={{21,22,23,0,0,0},{11,12,13,31,32,33}};
    memcpy(device_rows,data,sizeof data);calls[0]=calls[1]=0;failed=-1;
}
static void released(int c0,int c1){
    assert(calls[0]==c0&&calls[1]==c1);
    assert(!G.issue_open&&!G.is_cnt[0]&&!G.is_cnt[1]);
}
static void *recycle(void *unused){
    (void)unused;pthread_mutex_lock(&G.mx);
    while(G.issue_open)pthread_cond_wait(&G.cv_take,&G.mx);
    memset(device_rows,0,sizeof device_rows);
    pthread_mutex_unlock(&G.mx);return NULL;
}
int main(void){
    pthread_mutex_init(&G.mx,NULL);pthread_cond_init(&G.cv_take,NULL);
    float sentinel[20],out[20],want[20];
    for(int i=0;i<20;i++)sentinel[i]=(float)(100+i);
    memcpy(want,sentinel,sizeof want);
    for(int d=0;d<3;d++){want[d]=11+d;want[10+d]=21+d;want[15+d]=31+d;}
    prepare();memcpy(out,sentinel,sizeof out);
    pthread_t waiter;assert(!pthread_create(&waiter,NULL,recycle,NULL));
    assert(qt_take_rows(13,4,out,5));assert(!pthread_join(waiter,NULL));
    released(1,1);assert(!memcmp(out,want,sizeof out));
    for(int i=0;i<12;i++)assert(((float*)device_rows)[i]==0);

    /* Both backend failures and malformed masks/geometry leave every row,
     * including previously computed CPU ranks, byte-for-byte untouched. */
    for(int test=0;test<12;test++){
        prepare();memcpy(out,sentinel,sizeof out);
        uint32_t mask=13;int K=4;size_t stride=5;float *dest=out;
        switch(test){
        case 0:failed=0;break;
        case 1:failed=1;break;
        case 2:mask=5;break;                 /* missing issued rank */
        case 3:mask=15;break;                /* nonexistent GPU rank */
        case 4:G.is_k[1][1]=2;break;         /* duplicate rank across devices */
        case 5:G.is_k[1][1]=-1;break;
        case 6:G.is_k[1][1]=4;break;
        case 7:K=33;break;
        case 8:stride=2;break;
        case 9:stride=SIZE_MAX;break;
        case 10:dest=NULL;break;
        case 11:mask=0;break;                /* drain even a bad zero mask */
        }
        assert(!qt_take_rows(mask,K,dest,stride));released(1,1);
        assert(!memcmp(out,sentinel,sizeof out));
    }
    prepare();G.is_cnt[0]=G.is_cnt[1]=0;
    assert(qt_take_rows(0,0,NULL,0));released(0,0);
    prepare();G.is_cnt[0]=1;G.is_cnt[1]=0;G.is_k[0][0]=31;
    float large[96]={0};assert(qt_take_rows(1u<<31,32,large,3));released(1,0);
    assert(large[93]==21&&large[94]==22&&large[95]==23);
    prepare();G.is_cnt[0]=2;G.is_cnt[1]=0;G.is_k[0][0]=0;G.is_k[0][1]=1;
    assert(qt_take_rows(3,2,out,3));released(1,0); /* all ranks on one device */
    G.on=0;assert(qt_take_rows(0,0,NULL,0));assert(!qt_take_rows(1,0,NULL,0));
    pthread_cond_destroy(&G.cv_take);pthread_mutex_destroy(&G.mx);
    puts("tier raw ranked collection: ok (failure, masks, ownership, wakeup)");return 0;
}
