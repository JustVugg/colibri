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

    QtCounters at={10,20,5,1,0},mid={12,23,9,1,2},end={15,24,9,3,2};
    QtCounters tier[2]={q38_qt_delta(&mid,&at),q38_qt_delta(&end,&mid)};
    CHECK(tier[0].hits==2&&tier[0].miss==3&&tier[0].uploads==4&&tier[0].q_full_skips==2);
    CHECK(tier[1].hits==3&&tier[1].miss==1&&tier[1].uploads==0&&tier[1].swaps==2);
    uint64_t ram_hits[2]={7,1},ram_misses[2]={4,0};
    char split[512];
    CHECK(q38_format_expert_split(split,sizeof split,"r1",ram_hits,ram_misses,0,tier)==84);
    CHECK(!strcmp(split,"[qwen38] request r1 experts: prefill RAM 7 hits, 4 disk | decode RAM 1 hits, 0 disk\n"));
    CHECK(q38_format_expert_split(split,sizeof split,"r1",ram_hits,ram_misses,1,tier)>0);
    CHECK(!strcmp(split,"[qwen38] request r1 experts: prefill RAM 7 hits, 4 disk | decode RAM 1 hits, 0 disk"
                        " | VRAM hit rate: prefill 40.0 % (2 hits, 3 CPU), decode 75.0 % (3 hits, 1 CPU)"
                        " | uploads 4, swaps 2, q_skips 2\n"));
    /* no room for the tier part; room for the RAM part's text but not its newline; just room */
    volatile size_t split_short=64,split_no_newline=84,split_exact=85;
    CHECK(q38_format_expert_split(split,split_short,"r1",ram_hits,ram_misses,1,tier)<0);
    CHECK(q38_format_expert_split(split,split_no_newline,"r1",ram_hits,ram_misses,0,tier)<0);
    CHECK(q38_format_expert_split(split,split_exact,"r1",ram_hits,ram_misses,0,tier)==84);
    CHECK(strlen(split)==84&&split[83]=='\n');

    puts("qwen38 metrics: request deltas, TPOT rate, PROF schema, expert split: ok");
    return 0;
}
