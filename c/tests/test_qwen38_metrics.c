/* Qwen3.8 request metrics: timer deltas, request-local hit rate, strict TPOT
 * throughput, and the shared PROF wire schema. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#define QWEN38_TEST_SERVE
#include "../qwen38.c"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);return 1; \
} } while(0)

static int close_enough(double a,double b){ return fabs(a-b)<1e-12; }

int main(void){
    CHECK(q38_decode_rate(0,1.0)==0.0);
    CHECK(q38_decode_rate(1,1e-9)==0.0);
    CHECK(q38_decode_rate(2,0.5)==2.0);
    CHECK(q38_decode_rate(3,0.5)==4.0);
    CHECK(q38_decode_rate(3,0.0)==0.0);

    CHECK(q38_cache_hit_percent(0,0)==0.0);
    CHECK(q38_cache_hit_percent(3,0)==100.0);
    CHECK(q38_cache_hit_percent(0,7)==0.0);
    CHECK(q38_cache_hit_percent(3,1)==75.0);

    Q38Timers before={0},after={0};
    for(int i=0;i<Q38_TM_COUNT;i++){
        before.seconds[i]=(double)i;
        after.seconds[i]=(double)i+0.25*(i+1);
    }
    before.forwards=7;after.forwards=11;
    Q38Timers delta=q38_tm_delta(&after,&before);
    for(int i=0;i<Q38_TM_COUNT;i++)CHECK(close_enough(delta.seconds[i],0.25*(i+1)));
    CHECK(delta.forwards==4);

    Q38Timers profile={0};
    profile.seconds[Q38_TM_EXPERT_READ]=0.1;
    profile.seconds[Q38_TM_FP8_EXPAND]=0.2;
    profile.seconds[Q38_TM_ROUTED_EXPERT]=0.3;
    profile.seconds[Q38_TM_SHARED_EXPERT]=0.4;
    profile.seconds[Q38_TM_DELTANET]=0.5;
    profile.seconds[Q38_TM_QSA_INDEX]=0.6;
    profile.seconds[Q38_TM_QSA_ATTENTION]=0.7;
    profile.seconds[Q38_TM_LM_HEAD]=0.8;
    profile.forwards=4;
    char line[256];
    CHECK(q38_format_prof(line,sizeof line,2.0,7,3,&profile)>0);
    CHECK(!strcmp(line,"PROF 2.000 7 3 0.100 0.300 0.700 1.800 0.800 4\n"));
    volatile size_t short_capacity=8;
    CHECK(q38_format_prof(line,short_capacity,2.0,7,3,&profile)<0);

    puts("qwen38 metrics: request deltas, TPOT rate, PROF schema: ok");
    return 0;
}
