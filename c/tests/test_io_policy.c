#include <stdio.h>
#include <stdlib.h>
#include "../io_policy.h"

static int expect(int got,int want){
    if(got==want) return 0;
    fprintf(stderr,"got %d, want %d\n",got,want); return 1;
}

int main(void){
    unsetenv("IO_THREADS"); unsetenv("IO_OVERLAP_THREADS"); unsetenv("IO_IDLE_THREADS");
    if(expect(expert_io_threads(20,1),8)||expect(expert_io_threads(20,0),20)) return 1;
    setenv("IO_OVERLAP_THREADS","4",1); setenv("IO_IDLE_THREADS","12",1);
    if(expect(expert_io_threads(20,1),4)||expect(expert_io_threads(20,0),12)) return 1;
    setenv("IO_THREADS","6",1);
    if(expect(expert_io_threads(20,1),6)||expect(expert_io_threads(20,0),6)) return 1;
    puts("io policy tests: ok"); return 0;
}
