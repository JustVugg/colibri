/* Async CUDA groups may borrow an LRU slot until stream completion.  The
 * oldest slot is therefore not necessarily an eligible eviction victim.
 * And a slot whose slab was freed by rss_guard (#1034) is only reusable
 * while the row's live-slab count stays under ecap: reusing it re-allocates
 * a slab, which is growth, not eviction. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>

static _Atomic int reservation_writer_stop;
static ESlot *reservation_writer_slot;
static uint8_t reservation_writer_byte;

static void *reservation_payload_writer(void *unused){
    (void)unused;
    while(!atomic_load_explicit(&reservation_writer_stop,memory_order_relaxed)){
        __atomic_store_n(&reservation_writer_slot->slab,&reservation_writer_byte,__ATOMIC_RELAXED);
        __atomic_store_n(&reservation_writer_slot->slab,NULL,__ATOMIC_RELAXED);
    }
    return NULL;
}

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
    slots[0].eid=0; slots[2].eid=-5; slots[2].slab=NULL;
    if(eslot_lru_victim(slots,3,2)!=0) return 8;   /* live=2 (slot0 + prenotazione) >= ecap */
    if(eslot_lru_victim(slots,3,3)!=1) return 9;   /* live=2<ecap=3: di nuovo il vuoto */

    /* A blocking PILOT fills a reserved slot outside g_pilot_mx while another
     * worker may scan victims under that mutex.  The victim scan must decide
     * from the negative reservation without touching the changing payload. */
    reservation_writer_slot=&slots[2];
    atomic_store_explicit(&reservation_writer_stop,0,memory_order_relaxed);
    pthread_t writer;
    if(pthread_create(&writer,NULL,reservation_payload_writer,NULL)) return 10;
    int race_bad=0;
    for(int i=0;i<200000;i++) if(eslot_lru_victim(slots,3,2)!=0){ race_bad=1; break; }
    atomic_store_explicit(&reservation_writer_stop,1,memory_order_relaxed);
    pthread_join(writer,NULL);
    __atomic_store_n(&slots[2].slab,NULL,__ATOMIC_RELAXED);
    if(race_bad) return 11;

    puts("test_eslot_inflight: ok");
    return 0;
}
