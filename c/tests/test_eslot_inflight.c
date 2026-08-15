/* Async CUDA groups may borrow an LRU slot until stream completion.  The
 * oldest slot is therefore not necessarily an eligible eviction victim. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>

int main(void){
    ESlot slots[3]={0};
    slots[0].used=1; slots[1].used=2; slots[2].used=3;

    if(eslot_lru_victim(slots,3)!=0) return 1;
    eslot_acquire(&slots[0]);
    if(eslot_lru_victim(slots,3)!=1) return 2;
    eslot_acquire(&slots[1]); eslot_acquire(&slots[2]);
    if(eslot_lru_victim(slots,3)!=-1) return 3;

    eslot_release(&slots[0]); eslot_release(&slots[1]); eslot_release(&slots[2]);
    if(eslot_lru_victim(slots,3)!=0) return 4;
    puts("test_eslot_inflight: ok");
    return 0;
}
