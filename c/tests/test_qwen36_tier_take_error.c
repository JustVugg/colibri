/* A failed device must not silently drop an expert or publish a partial sum. */
#include <stdio.h>
#define coli_cuda_expert_group_take fixture_take
#include "qwen36_fake_cuda.h"
#undef coli_cuda_expert_group_take
static int failed=-1, calls[2];
static float rows[2][2]={{2,4},{8,12}};
const float *coli_cuda_expert_group_take(int device) {
    calls[device]++;
    return device==failed?NULL:rows[device];
}
#include "../qwen36_tier.c"
int main(void) {
    G.on=1;G.ndev=2;G.D=2;G.dev[0]=0;G.dev[1]=1;
    pthread_mutex_init(&G.mx,NULL);pthread_cond_init(&G.cv_take,NULL);
    for(failed=-1;failed<2;failed++){
        G.issue_open=1;G.is_cnt[0]=G.is_cnt[1]=1;
        G.is_k[0][0]=0;G.is_k[1][0]=1;calls[0]=calls[1]=0;
        float val[2]={0.5f,0.25f},out[2]={10,20};
        int ok=qt_take(3,val,2,out);
        float want0=failed<0?13:10,want1=failed<0?25:20;
        if(ok!=(failed<0)||out[0]!=want0||out[1]!=want1||
           calls[0]!=1||calls[1]!=1||G.is_cnt[0]||G.is_cnt[1]||G.issue_open){
            fprintf(stderr,"take failed case %d: status=%d output=%g,%g calls=%d,%d\n",
                    failed,ok,out[0],out[1],calls[0],calls[1]);return 1;
        }
    }
    G.issue_open=1;
    if(!qt_take(0,NULL,0,NULL)||G.issue_open) return 1;
    G.on=0;
    if(!qt_take(0,NULL,0,NULL)||qt_take(1,NULL,0,NULL)) return 1;
    pthread_cond_destroy(&G.cv_take);pthread_mutex_destroy(&G.mx);
    puts("tier take error: ok");return 0;
}
