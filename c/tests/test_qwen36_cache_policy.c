/* Tiny deterministic policy simulation: no model, GPU or file I/O.
 * clang -O2 -std=c11 c/tests/test_qwen36_cache_policy.c -o /tmp/qwen-cache-policy
 */
#include "../qwen36_cache_policy.h"
#include <stdio.h>

static int checks,failures;
#define CHECK(c,...) do {checks++;if(!(c)){failures++; \
    fprintf(stderr,"FAIL line %d: ",__LINE__);fprintf(stderr,__VA_ARGS__);fputc('\n',stderr);}} while(0)

static void basic(void) {
    QwenCachePolicy p={0};
    CHECK(!qcp_init(NULL,4,4),"null policy initialized");
    CHECK(!qcp_init(&p,0,4) && !qcp_payload_bytes(&p),"zero key count accepted");
    CHECK(!qcp_init(&p,4,0) && !p.history,"zero age interval accepted");
    CHECK(!qcp_init(&p,SIZE_MAX/sizeof(QwenCacheHistory)+1,4),"history byte overflow accepted");
    CHECK(qcp_init(&p,4,4),"allocation failed");
    CHECK(qcp_payload_bytes(&p)==4*sizeof(QwenCacheHistory),"metadata accounting changed");
    CHECK(!qcp_touch(&p,0,1) && !qcp_touch(&p,0,1) && p.history[0].pending==2,"initial demands were not kept pending");
    CHECK(!qcp_touch(&p,1,1) && !qcp_admit(&p,0,1),"incomplete epoch influenced admission");
    CHECK(qcp_touch(&p,0,1)==3 && p.epoch==1 && !p.references,"reference boundary aging changed");
    CHECK(qcp_score(&p,1)==1 && qcp_admit(&p,0,1),"strict larger completed score rejected");
    qcp_touch(&p,1,1);for(int i=0;i<3;i++)qcp_touch(&p,2,1);
    CHECK(qcp_score(&p,0)==1 && !qcp_admit(&p,0,1) && !qcp_admit(&p,1,0),"ties must bypass");
    QwenCacheHistory history[4];memcpy(history,p.history,sizeof(history));
    uint64_t epoch=p.epoch,references=p.references;
    for(int i=0;i<10000;i++)qcp_touch(&p,(size_t)i%4,0);
    CHECK(p.epoch==epoch && p.references==references && !memcmp(history,p.history,sizeof(history)),"prefetch polluted demand history");
    size_t invalid=p.key_count;
    CHECK(!qcp_touch(&p,invalid,1) && p.references==references,"invalid key advanced clock");
    CHECK(!qcp_admit(&p,invalid,0) && !qcp_admit(&p,0,invalid),"invalid key admitted");
    for(int i=0;i<4;i++)qcp_touch(&p,2,1);
    CHECK(!qcp_score(&p,0) && qcp_score(&p,2)==5,"lazy aging differs from eager shifts");
    qcp_free(&p);qcp_free(&p);
    CHECK(!p.history && !p.key_count && !p.epoch && !qcp_payload_bytes(&p),"free did not clear policy");
}
static void saturation_and_wrap(void) {
    QwenCachePolicy p={0};CHECK(qcp_init(&p,3,4),"allocation failed");
    p.history[0].frequency=UINT32_MAX;p.history[0].pending=UINT32_MAX;
    CHECK(qcp_touch(&p,0,1)==UINT32_MAX && p.history[0].pending==UINT32_MAX,"pending frequency wrapped at saturation");
    p.history[0].pending=0;
    for(int i=0;i<3;i++)qcp_touch(&p,1,1);
    CHECK(qcp_score(&p,0)==UINT32_MAX/2,"saturated frequency failed to age");
    p.epoch=UINT64_MAX;p.references=3;
    p.history[0]=(QwenCacheHistory){UINT64_MAX,16,2};
    p.history[1]=(QwenCacheHistory){UINT64_MAX-1,8,2};
    p.history[2]=(QwenCacheHistory){UINT64_MAX-33,UINT32_MAX,UINT32_MAX};
    CHECK(!qcp_score(&p,2),"33 epochs did not erase stale frequency");
    CHECK(qcp_touch(&p,2,1)==1 && p.epoch==1 && !p.references,"epoch wrap did not rebase");
    CHECK(qcp_score(&p,0)==10 && qcp_score(&p,1)==3,"epoch wrap lost or doubled decay");
    p.history[0]=(QwenCacheHistory){0,UINT32_MAX,UINT32_MAX};
    CHECK(qcp_score(&p,0)==UINT32_MAX,"completed frequency addition wrapped");
    qcp_free(&p);
    CHECK(qcp_init(&p,1,UINT64_MAX),"largest interval rejected");
    p.references=UINT64_MAX-1;p.history[0].frequency=6;
    CHECK(qcp_touch(&p,0,1)==4 && p.epoch==1 && !p.references,"reference counter overflow");
    qcp_free(&p);
}
/* Independent eager reference checks keys left untouched across many epochs,
 * including uneven epoch boundaries and speculative touches between demands. */
static void eager_reference(void) {
    enum{N=17,INTERVAL=13};QwenCachePolicy p={0};
    uint32_t score[N]={0},pending[N]={0},random=0x91a73b5du;
    unsigned references=0;CHECK(qcp_init(&p,N,INTERVAL),"reference allocation");
    for(int step=0;step<8192;step++) {
        random^=random<<13;random^=random>>17;random^=random<<5;
        size_t key=(random>>3)%N;int demand=(random&7)!=0;
        if(demand) {
            pending[key]++;
            if(++references==INTERVAL) {
                references=0;
                for(int i=0;i<N;i++){score[i]=score[i]/2+pending[i];pending[i]=0;}
            }
        }
        CHECK(qcp_touch(&p,key,demand)==score[key],"lazy/eager touch disagreement at step %d",step);
        if(step%64==0)for(int i=0;i<N;i++)
            CHECK(qcp_score(&p,(size_t)i)==score[i],"lazy/eager stale-key disagreement at step %d key %d",step,i);
    }
    qcp_free(&p);
}

enum {KEYS=32,RETAINED=12,WORKING=4,TOTAL=RETAINED+WORKING};
typedef struct {int key;uint64_t used;} Entry;
typedef struct {
    QwenCachePolicy policy;
    Entry retained[RETAINED],lru[TOTAL];
    uint64_t clock;
} Simulation;
static void simulation_init(Simulation *s) {
    memset(s,0,sizeof(*s));
    CHECK(qcp_init(&s->policy,KEYS,KEYS*8),"simulation allocation");
    for(int i=0;i<RETAINED;i++)s->retained[i].key=-1;
    for(int i=0;i<TOTAL;i++)s->lru[i].key=-1;
}
static int policy_access(Simulation *s,int key) {
    qcp_touch(&s->policy,(size_t)key,1);uint64_t now=++s->clock;
    for(int i=0;i<RETAINED;i++)if(s->retained[i].key==key){s->retained[i].used=now;return 1;}
    int victim=-1;
    for(int i=0;i<RETAINED;i++) {
        if(s->retained[i].key<0){victim=i;break;}
        uint32_t score=qcp_score(&s->policy,(size_t)s->retained[i].key);
        uint32_t prior=victim<0?UINT32_MAX:qcp_score(&s->policy,(size_t)s->retained[victim].key);
        if(victim<0 || score<prior || (score==prior && s->retained[i].used<s->retained[victim].used))victim=i;
    }
    if(s->retained[victim].key<0 || qcp_admit(&s->policy,(size_t)key,(size_t)s->retained[victim].key))
        s->retained[victim]=(Entry){key,now};
    return 0;
}
static int lru_access(Simulation *s,int key) {
    int victim=0;
    for(int i=0;i<TOTAL;i++) {
        if(s->lru[i].key==key){s->lru[i].used=s->clock;return 1;}
        if(s->lru[i].key<0){victim=i;break;}
        if(s->lru[i].used<s->lru[victim].used)victim=i;
    }
    s->lru[victim]=(Entry){key,s->clock};return 0;
}
static void uniform_sweeps(void) {
    Simulation s;simulation_init(&s);int policy_hits=0,lru_hits=0;
    for(int round=0;round<256;round++) {
        int hits=0;
        for(int layer=0;layer<8;layer++) {
            int working[WORKING],loaded=0;
            for(int expert=0;expert<4;expert++) {
                int key=layer*4+expert;int hit=policy_access(&s,key);hits+=hit;
                if(!hit)working[loaded++]=key;
                int lru_hit=lru_access(&s,key);if(round)lru_hits+=lru_hit;
            }
            CHECK(loaded<=WORKING,"working group exceeded fixed reservation");
            for(int i=0;i<loaded;i++)CHECK(working[i]/4==layer,"working group retained a previous layer");
        }
        if(round){policy_hits+=hits;CHECK(hits==RETAINED,"uniform sweep %d rotated retained cache (%d hits)",round,hits);}
    }
    CHECK(policy_hits==255*RETAINED && lru_hits==0,"twice-capacity scan comparison changed");
    printf("uniform 32-key sweeps, 16 total slots: frequency=%d hits, LRU=%d hits after warmup\n",policy_hits,lru_hits);
    qcp_free(&s.policy);
}
static void changing_hot_set(void) {
    Simulation s;simulation_init(&s);
    for(int round=0;round<64;round++)for(int key=0;key<RETAINED;key++)policy_access(&s,key);
    uint32_t remembered=qcp_score(&s.policy,0);
    /* Eviction is only a manager action: scores remain in the policy. */
    s.retained[0].key=-1;
    CHECK(qcp_score(&s.policy,0)==remembered && remembered>0,"eviction erased demand history");
    policy_access(&s,0);
    int first_all_hit=-1,tail_hits=0;
    for(int round=0;round<128;round++) {
        int hits=0;
        for(int key=KEYS-RETAINED;key<KEYS;key++)hits+=policy_access(&s,key);
        if(hits==RETAINED && first_all_hit<0)first_all_hit=round;
        if(round>=112)tail_hits+=hits;
    }
    /* Fixed fixture limit, not a universal latency promise: after 64 old-set
     * sweeps the disjoint new set must fully hit within 65 replacement sweeps. */
    CHECK(first_all_hit>=0 && first_all_hit<=65,"new hot set did not replace old residents within 65 rounds");
    CHECK(tail_hits==16*RETAINED,"new hot set failed to stabilize");
    for(int i=0;i<RETAINED;i++)CHECK(s.retained[i].key>=KEYS-RETAINED,"old hot key remained resident");
    printf("hot-set change: first full-hit round=%d, final 16 rounds=%d/%d hits\n",first_all_hit,tail_hits,16*RETAINED);
    qcp_free(&s.policy);
}
int main(void) {
    basic();saturation_and_wrap();eager_reference();uniform_sweeps();changing_hot_set();
    printf("qwen36 cache policy: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
