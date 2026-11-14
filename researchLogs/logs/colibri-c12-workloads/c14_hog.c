/* c14_hog.c -- co-tenant memory hog for the C14 pressure arm.
 * Allocates N GB of anonymous memory, touches every page once, then re-touches
 * one page per 64 KB stride every 15 s so the pages stay resident (defeating
 * any lazy eviction) without meaningful CPU load. SIGTERM exits.
 *
 *   clang -O2 c14_hog.c -o /tmp/c14_hog && /tmp/c14_hog 5
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

static volatile char sink;

static void on_term(int sig) { (void)sig; _exit(0); }

int main(int argc, char **argv) {
    double gb = argc > 1 ? atof(argv[1]) : 5.0;
    signal(SIGTERM, on_term);
    size_t bytes = (size_t)(gb * 1024.0 * 1024.0 * 1024.0);
    char *p = malloc(bytes);
    if (!p) { fprintf(stderr, "hog: malloc %zu failed\n", bytes); return 1; }
    /* touch every page (first byte of each 4 KiB page) */
    for (size_t i = 0; i < bytes; i += 4096) { p[i] = (char)(i >> 12); sink = p[i]; }
    fprintf(stderr, "hog: %.1f GB touched and held\n", gb);
    /* periodic re-touch, one byte per 64 KiB, to stay resident */
    for (;;) {
        struct timespec t = {15, 0};
        nanosleep(&t, NULL);
        for (size_t i = 0; i < bytes; i += 65536) { p[i] = (char)(i >> 16); sink = p[i]; }
    }
}
