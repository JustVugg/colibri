#include <stdio.h>
#include "../tier.h"

static int fail(const char *message){
    fprintf(stderr,"tier test failed: %s\n",message);
    return 1;
}

int main(void){
    uint32_t heat[6]={20,2,8,3,30,1};
    int pinned[2]={0,1}, slot=-1, eid=-1; long gain=0;
    if(!tier_pick_swap(heat,6,pinned,2,&slot,&eid,&gain)) return fail("hot expert not promoted");
    if(slot!=1 || eid!=4 || gain!=28) return fail("wrong promotion candidate");

    uint32_t stable[4]={20,18,24,4}; int resident[2]={0,1};
    if(tier_pick_swap(stable,4,resident,2,&slot,&eid,&gain)) return fail("hysteresis did not block churn");

    tier_decay(heat,6);
    if(heat[0]!=10 || heat[1]!=1 || heat[4]!=15) return fail("heat decay");
    puts("tier tests: ok");
    return 0;
}
