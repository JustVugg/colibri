/* Async CUDA groups may borrow an LRU slot until stream completion.  The
 * oldest slot is therefore not necessarily an eligible eviction victim.
 * And a slot whose slab was freed by rss_guard (#1034) is only reusable
 * while the row's live-slab count stays under ecap: reusing it re-allocates
 * a slab, which is growth, not eviction. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>

int main(void){
    uint8_t dummy[4];
    ESlot slots[3]={0};
    for(int i=0;i<3;i++){ slots[i].slab=dummy; slots[i].used=(uint64_t)i+1; }

    if(eslot_lru_victim(slots,3,3)!=0) return 1;
    eslot_acquire(&slots[0]);
    if(eslot_lru_victim(slots,3,3)!=1) return 2;
    eslot_acquire(&slots[1]); eslot_acquire(&slots[2]);
    if(eslot_lru_victim(slots,3,3)!=-1) return 3;

    eslot_release(&slots[0]); eslot_release(&slots[1]); eslot_release(&slots[2]);
    if(eslot_lru_victim(slots,3,3)!=0) return 4;

    /* #1034: slot svuotato da rss_guard (eid=-1, slab=NULL) */
    slots[1].eid=-1; slots[1].slab=NULL;
    if(eslot_lru_victim(slots,3,2)!=0) return 5;   /* live=2>=ecap=2: eviction, non crescita */
    if(eslot_lru_victim(slots,3,3)!=1) return 6;   /* live=2<ecap=3: il vuoto si puo' riusare */

    /* slot libero che possiede ancora lo slab: riuso a costo zero, sempre preferito */
    slots[0].eid=-1;
    if(eslot_lru_victim(slots,3,2)!=0) return 7;

    /* prenotazione in volo (eid<-1): mai vittima, e conta come slab vivo */
    slots[0].eid=0; slots[2].eid=-5;
    if(eslot_lru_victim(slots,3,2)!=0) return 8;   /* live=2 (slot0 + prenotazione) >= ecap */
    if(eslot_lru_victim(slots,3,3)!=1) return 9;   /* live=2<ecap=3: di nuovo il vuoto */

    puts("test_eslot_inflight: ok");
    return 0;
}
