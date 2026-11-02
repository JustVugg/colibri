/* probe_v4.c: coli_v4_os_available_memory (deepseek_v4.c's resource-plan unit, as the Makefile builds it with
 * -DCOLI_V4_UNIT_RESOURCE_PLAN) against compat.h's probe, and the microseconds a call. Built from c/. */
#define COLI_V4_UNIT_RESOURCE_PLAN 1
#include "deepseek_v4.c"
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
int main(void){
    double t0 = now_us(), v = 0; for (int i = 0; i < 2000; i++) v += (double)coli_v4_os_available_memory(); double us = (now_us() - t0) / 2000;
    printf("coli_v4_os_available_memory %.4f GB | compat_mem_available_gb %.4f GB | %.2f us a call (mean %.4f)\n",
           coli_v4_os_available_memory() / 1e9, compat_mem_available_gb(), us, v / 2000 / 1e9);
    return 0;
}
