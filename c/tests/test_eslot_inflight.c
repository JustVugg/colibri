/* Async CUDA groups may borrow an LRU slot until stream completion.  The
 * oldest slot is therefore not necessarily an eligible eviction victim.
 * And a slot whose slab was freed by rss_guard (#1034) is only reusable
 * while the row's live-slab count stays under ecap: reusing it re-allocates
 * a slab, which is growth, not eviction. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>

/* dev's eslot_lru_victim before DEMAND_POLICY, verbatim: the default policy
 * must pick the same victim on every row state. */
static int legacy_lru_victim(ESlot *slots,int n,int ecap){
    int lru=-1, empty=-1, live=0;
    for(int i=0;i<n;i++){
        ESlot *s=&slots[i];
        if(s->slab || s->eid<-1) live++;
        if(eslot_busy(s) || s->eid<-1) continue;
        if(!s->slab){ if(s->eid==-1 && empty<0) empty=i; continue; }
        if(s->eid==-1) return i;
        if(lru<0 || s->used<slots[lru].used) lru=i;
    }
    if(empty>=0 && live<ecap) return empty;
    return lru;
}

/* Default output unchanged: with DEMAND_POLICY unset the policy resolves to
 * lru and, even with live eheat/elast rows that would make LFRU disagree,
 * every victim matches the legacy scan.  Also asserts lfru really engages. */
static int check_default_matches_legacy(void){
    enum { NE=64, NS=12 };
    static uint32_t heat_row[NE], last_row[NE];
    uint32_t *heat[1]={heat_row}, *last[1]={last_row};
    uint8_t dummy[4];
    Model m; memset(&m,0,sizeof(m));
    unsetenv("DEMAND_POLICY");
    m.demand_policy=tier_demand_policy_env();
    if(m.demand_policy!=TIER_DEMAND_LRU) return 20;
    m.eheat=heat; m.elast=last; m.eaccess_clock=1000;
    uint32_t rng=12345u; int lfru_differs=0;
    for(int it=0;it<20000;it++){
        ESlot slots[NS]; memset(slots,0,sizeof(slots));
        int n=1+(int)((rng=rng*1103515245u+12345u)>>16)%NS;
        int ecap=1+(int)((rng=rng*1103515245u+12345u)>>16)%NS;
        for(int e=0;e<NE;e++){
            heat_row[e]=((rng=rng*1103515245u+12345u)>>16)%32;
            last_row[e]=((rng=rng*1103515245u+12345u)>>16)%1000;
        }
        for(int i=0;i<n;i++){
            uint32_t r=(rng=rng*1103515245u+12345u)>>8;
            slots[i].slab=(r%8)?dummy:NULL;
            slots[i].eid=(r%13==0)?-1:(r%17==0)?-5:(int)((r>>4)%NE);
            slots[i].used=(r>>12)%997;
            if(r%11==0) eslot_acquire(&slots[i]);
        }
        int want=legacy_lru_victim(slots,n,ecap);
        if(eslot_lru_victim(&m,0,slots,n,ecap)!=want) return 21;
        m.demand_policy=TIER_DEMAND_LFRU;
        if(eslot_lru_victim(&m,0,slots,n,ecap)!=want) lfru_differs=1;
        m.demand_policy=TIER_DEMAND_LRU;
    }
    return lfru_differs?0:22;
}

int main(void){
    { int rc=check_default_matches_legacy(); if(rc) return rc; }
    /* Zeroed model: demand_policy=TIER_DEMAND_LRU (0) and no eheat/elast rows,
     * i.e. the legacy LRU scan the assertions below pin down. */
    Model m; memset(&m,0,sizeof(m));
    uint8_t dummy[4];
    ESlot slots[3]={0};
    for(int i=0;i<3;i++){ slots[i].slab=dummy; slots[i].used=(uint64_t)i+1; }

    if(eslot_lru_victim(&m,0,slots,3,3)!=0) return 1;
    eslot_acquire(&slots[0]);
    if(eslot_lru_victim(&m,0,slots,3,3)!=1) return 2;
    eslot_acquire(&slots[1]); eslot_acquire(&slots[2]);
    if(eslot_lru_victim(&m,0,slots,3,3)!=-1) return 3;

    eslot_release(&slots[0]); eslot_release(&slots[1]); eslot_release(&slots[2]);
    if(eslot_lru_victim(&m,0,slots,3,3)!=0) return 4;

    /* #1034: slot svuotato da rss_guard (eid=-1, slab=NULL) */
    slots[1].eid=-1; slots[1].slab=NULL;
    if(eslot_lru_victim(&m,0,slots,3,2)!=0) return 5;   /* live=2>=ecap=2: eviction, non crescita */
    if(eslot_lru_victim(&m,0,slots,3,3)!=1) return 6;   /* live=2<ecap=3: il vuoto si puo' riusare */

    /* slot libero che possiede ancora lo slab: riuso a costo zero, sempre preferito */
    slots[0].eid=-1;
    if(eslot_lru_victim(&m,0,slots,3,2)!=0) return 7;

    /* prenotazione in volo (eid<-1): mai vittima, e conta come slab vivo */
    slots[0].eid=0; slots[2].eid=-5;
    if(eslot_lru_victim(&m,0,slots,3,2)!=0) return 8;   /* live=2 (slot0 + prenotazione) >= ecap */
    if(eslot_lru_victim(&m,0,slots,3,3)!=1) return 9;   /* live=2<ecap=3: di nuovo il vuoto */

    puts("test_eslot_inflight: ok");
    return 0;
}
