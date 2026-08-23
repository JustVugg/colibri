/* Async CUDA groups may borrow a slot until stream completion, so the next slot
 * under the clock hand is not automatically an eligible victim.
 * Two invariants are covered together:
 *  - clock: a slot with its reference bit set survives one sweep (second chance),
 *    borrowed/reserved slots are skipped, and the hand rotates fairly;
 *  - #1034: a slot whose slab was freed by rss_guard is only reusable while the
 *    row's live-slab count stays under ecap, since reusing it re-allocates a slab,
 *    which is growth rather than eviction. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>

int main(void){
    uint8_t dummy[4];

    /* ---- clock behaviour (victims must own a slab to be evictable) ---- */
    ESlot slots[4]={0};
    int hand=0;
    for(int i=0;i<4;i++){ slots[i].eid=10+i; slots[i].slab=dummy; slots[i].recent=1; }

    /* Every bit set: the first sweep clears them, the second evicts slot 0. */
    if(eslot_clock_victim(slots,4,&hand,4)!=0) return 1;
    if(hand!=1) return 2;
    for(int i=0;i<4;i++) if(slots[i].recent) return 3;

    /* A re-referenced slot 0 is spared; the hand is already past it anyway. */
    slots[0].recent=1;
    if(eslot_clock_victim(slots,4,&hand,4)!=1) return 4;

    /* Borrowed slot is skipped, and slot 0 spends its second chance. */
    hand=0; slots[0].recent=1; eslot_acquire(&slots[1]);
    if(eslot_clock_victim(slots,4,&hand,4)!=2) return 5;
    slots[2].eid=-3; /* another pilot worker's reservation: never a victim */
    if(eslot_clock_victim(slots,4,&hand,4)!=3) return 6;

    /* Everything in flight or reserved: no victim at all. */
    eslot_acquire(&slots[0]); eslot_acquire(&slots[2]); eslot_acquire(&slots[3]);
    if(eslot_clock_victim(slots,4,&hand,4)!=-1) return 7;

    /* A free slot that still owns its slab is the cheapest victim, bit or no bit. */
    eslot_release(&slots[0]); eslot_release(&slots[1]);
    eslot_release(&slots[2]); eslot_release(&slots[3]);
    hand=0; slots[2].eid=-1;
    for(int i=0;i<4;i++) slots[i].recent=1;
    if(eslot_clock_victim(slots,4,&hand,4)!=2) return 8;

    /* ---- #1034: an emptied slot is growth, so the cap gates its reuse ---- */
    ESlot row[3]={0};
    int rh=0;
    for(int i=0;i<3;i++){ row[i].eid=20+i; row[i].slab=dummy; row[i].recent=0; }

    /* rss_guard emptied slot 1 (eid=-1, slab=NULL); live slabs = 2. */
    row[1].eid=-1; row[1].slab=NULL;
    rh=0; if(eslot_clock_victim(row,3,&rh,2)!=0) return 9;   /* live=2>=ecap: evict an owner */
    rh=0; if(eslot_clock_victim(row,3,&rh,3)!=1) return 10;  /* live=2<ecap: reuse the empty */

    /* A free slot that still owns its slab beats the empty one even at the cap. */
    row[0].eid=-1;
    rh=0; if(eslot_clock_victim(row,3,&rh,2)!=0) return 11;

    /* An in-flight reservation (eid<-1) is never a victim and counts as live. */
    row[0].eid=20; row[0].recent=0; row[2].eid=-5;
    rh=0; if(eslot_clock_victim(row,3,&rh,2)!=0) return 12;  /* live=2 (slot0 + reservation) */
    rh=0; if(eslot_clock_victim(row,3,&rh,3)!=1) return 13;  /* under the cap: the empty again */
    puts("test_eslot_inflight: ok");
    return 0;
}
