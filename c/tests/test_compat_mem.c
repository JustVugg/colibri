/* #1500: compat_mem_available_gb() must measure something on every platform
 * the release ships for. olmoe sized its expert cache from a Linux-only probe
 * that returned 0 on Windows and macOS, and 0 meant one slot per layer. This
 * runs on all three CI legs and pins the shared probe itself. */
#include <stdio.h>
#include "../compat.h"
int main(void) {
    double avail = compat_mem_available_gb();
    printf("available RAM: %.2f GB\n", avail);
    if (!(avail > 0.0)) { fprintf(stderr, "compat_mem_available_gb() returned %.3f on this platform\n", avail); return 1; }
    if (avail > 100000.0) { fprintf(stderr, "implausible available RAM %.1f GB\n", avail); return 1; }
    puts("compat mem probe: ok");
    return 0;
}
