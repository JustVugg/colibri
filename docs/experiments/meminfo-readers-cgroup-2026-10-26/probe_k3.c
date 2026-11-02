/* probe_k3.c: k3_mem_avail (kimi_k3.c, included as test_k3_ram_budget.c does) against compat.h's probe and
 * /proc/meminfo, and the microseconds a call. Built from c/tests/ in each tree. */
#define main coli_k3_main_unused
#include "../kimi_k3.c"
#undef main
#include <time.h>
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
int main(void){
    FILE *f = fopen("/proc/meminfo", "r"); char ln[256]; double kb = 0;
    while (f && fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemAvailable: %lf", &kb) == 1) break;
    if (f) fclose(f);
    double t0 = now_us(), v = 0; for (int i = 0; i < 2000; i++) v += k3_mem_avail(); double us = (now_us() - t0) / 2000;
    printf("k3_mem_avail %.4f GB | compat_mem_available_gb %.4f GB | MemAvailable kb*1024/1e9 %.4f, kb/1e6 %.4f | %.2f us a call (mean %.4f)\n",
           k3_mem_avail(), compat_mem_available_gb(), kb * 1024 / 1e9, kb / 1e6, us, v / 2000);
    return 0;
}
