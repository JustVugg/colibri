#include <stdio.h>
#include "../tier.h"

static int fail(const char *message){
    fprintf(stderr,"tier test failed: %s\n",message);
    return 1;
}

int main(void){
    if(!tier_should_promote(130,100)) return fail("shared hysteresis admits hot expert");
    if(tier_should_promote(129,100)) return fail("shared hysteresis blocks marginal expert");
    if(tier_should_promote(UINT32_MAX-1,UINT32_MAX))
        return fail("saturated hysteresis threshold wrapped");
    if(tier_decay_value(9)!=4) return fail("shared heat decay");
    uint32_t heat[6]={20,2,8,3,30,1};
    int pinned[2]={0,1}, slot=-1, eid=-1; long gain=0;
    if(!tier_pick_swap(heat,6,pinned,2,&slot,&eid,&gain)) return fail("hot expert not promoted");
    if(slot!=1 || eid!=4 || gain!=28) return fail("wrong promotion candidate");

    uint32_t stable[4]={20,18,24,4}; int resident[2]={0,1};
    if(tier_pick_swap(stable,4,resident,2,&slot,&eid,&gain)) return fail("hysteresis did not block churn");

    tier_decay(heat,6);
    if(heat[0]!=10 || heat[1]!=1 || heat[4]!=15) return fail("heat decay");

    uint32_t freq[5]={10,10,2,18,18}, last[5]={10,90,95,20,99};
    int live[2]={0,1};
    if(!tier_pick_lfru(freq,last,100,5,live,2,&slot,&eid,&gain)) return fail("LFRU promotion");
    if(slot!=0||eid!=4) return fail("LFRU did not prefer recent ties");

    /* Phase 3 demand-eviction policy: the LRU key is the slot's `used` clock,
     * verbatim; the LFRU key is the hand-computed score (heat<<8 | recent,
     * recent = 255 - age clamped); auto switches at cap 12/13; unknown
     * policies fall back to lru. */
    if(tier_demand_victim_key(TIER_DEMAND_LRU,8,7,90,100,42)!=42)
        return fail("lru demand key is not the used clock");
    if(tier_demand_lfru_score(3,90,100)!=(((uint64_t)3<<8)|245))
        return fail("LFRU demand score (age 10 -> recent 245)");
    if(tier_demand_victim_key(TIER_DEMAND_LFRU,8,3,90,100,42)!=(((uint64_t)3<<8)|245))
        return fail("LFRU demand key is not the hand-computed score");
    if(tier_demand_lfru_score(0,200,100)!=255)
        return fail("last_access in the future must clamp age to 0");
    if(tier_demand_lfru_score(1,0,255)!=((uint64_t)1<<8))
        return fail("age >= 255 must contribute no recency");
    if(tier_demand_victim_key(TIER_DEMAND_AUTO,12,3,90,100,42)!=(((uint64_t)3<<8)|245))
        return fail("auto at cap 12 must use LFRU");
    if(tier_demand_victim_key(TIER_DEMAND_AUTO,13,3,90,100,42)!=42)
        return fail("auto at cap 13 must use LRU");
    if(tier_demand_use_lfru(7,8) || tier_demand_victim_key(7,8,3,90,100,42)!=42)
        return fail("unknown policy must fall back to lru");
#if defined(__unix__) || defined(__APPLE__)
    setenv("DEMAND_POLICY","lfru",1);
    if(tier_demand_policy_env()!=TIER_DEMAND_LFRU) return fail("DEMAND_POLICY=lfru not parsed");
    setenv("DEMAND_POLICY","auto",1);
    if(tier_demand_policy_env()!=TIER_DEMAND_AUTO) return fail("DEMAND_POLICY=auto not parsed");
    setenv("DEMAND_POLICY","bogus",1);
    if(tier_demand_policy_env()!=TIER_DEMAND_LRU) return fail("unknown DEMAND_POLICY must fall back to lru");
    unsetenv("DEMAND_POLICY");
    if(tier_demand_policy_env()!=TIER_DEMAND_LRU) return fail("unset DEMAND_POLICY must default to lru");
#endif
    puts("tier tests: ok");
    return 0;
}
