#include <assert.h>
#include <stdio.h>
#include "../memory_pressure.h"

int main(void){
    assert(coli_safe_pin_bytes(100,120,.25)==30);
    assert(coli_safe_pin_bytes(20,120,.25)==20);
    assert(coli_safe_pin_bytes(20,120,2)==20);
    assert(coli_safe_pin_bytes(20,0,.25)==0);
    assert(!coli_memory_pressure(20,10,40,40,1));
    assert(coli_memory_pressure(9,10,40,40,1));
    assert(coli_memory_pressure(20,10,38,40,1));
    assert(!coli_memory_pressure(20,10,39.5,40,1));
    puts("memory pressure tests: ok");
    return 0;
}
