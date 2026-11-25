/* compat_rss_gb() and compat_now_s(), the one definition the engines share. Each engine had
 * its own rss_gb, and the copies disagreed on the unit: GLM-5.3 read macOS's bytes as KB. This
 * runs on the three CI legs and pins the unit by what the process is known to hold. */
#include <stdio.h>
#include <stdlib.h>
#include "../compat.h"
int main(void) {
    const double t0 = compat_now_s();
    const double before = compat_rss_gb();
    const size_t bytes = (size_t)256 << 20;
    volatile char *p = malloc(bytes);
    if (!p) { fputs("no memory for the test\n", stderr); return 1; }
    for (size_t i = 0; i < bytes; i += 1024) p[i] = (char)i;   /* resident, not just reserved */
    const double after = compat_rss_gb(), grown = after - before;
    free((void *)p);
    const double t1 = compat_now_s();
    printf("peak resident %.3f GiB before, %.3f after 256 MiB; %.3f s\n", before, after, t1 - t0);
    if (!(before > 0.0 && before < 4.0)) { fprintf(stderr, "implausible resident set of an empty process: %.3f GiB\n", before); return 1; }
    if (!(grown > 0.20 && grown < 0.45)) { fprintf(stderr, "256 MiB read as %.3f GiB\n", grown); return 1; }
    if (!(t1 >= t0 && t1 - t0 < 60.0)) { fprintf(stderr, "the clock went from %.6f to %.6f\n", t0, t1); return 1; }
    puts("compat rss and clock: ok");
    return 0;
}
