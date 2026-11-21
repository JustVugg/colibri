/* sycl_test_args.h -- the migrated GPU tests' device choice (test_cuda_chain.dp.cpp,
 * test_backend_cuda.dp.cpp), the same in both:
 *   --list              every GPU's profile (coli_sycl_print_profiles: the ordinals the engine's
 *                       COLI_GPUS names, the architecture, sub-group sizes, EUs, local memory, what
 *                       the matrix engines take) and exit 0: the first guarded contact with a device,
 *                       before any kernel
 *   --device N[,M..]    the ordinals to open (coli_cuda_init's list); else COLI_GPUS=N[,M..] (the
 *                       engine's variable); else device 0
 * Anything else is a usage error (exit 2). */
#ifndef SYCL_TEST_ARGS_H
#define SYCL_TEST_ARGS_H
#include "../coli_sycl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* fills devices[0..max) from the arguments or the environment and returns the count (1 and
 * device 0 when nothing says otherwise); -1 once --list has printed (the caller exits 0) */
static int sycl_test_devices(int argc, char **argv, int *devices, int max) {
    const char *list = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--list")) { coli_sycl_print_profiles(); return -1; }
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) list = argv[++i];
        else if (!strncmp(argv[i], "--device=", 9)) list = argv[i] + 9;
        else { fprintf(stderr, "usage: %s [--list] [--device N[,M..]]   (or COLI_GPUS=N[,M..])\n", argv[0]); exit(2); }
    }
    if (!list) list = getenv("COLI_GPUS");
    int n = 0;
    if (list && *list) {
        for (const char *s = list; *s && n < max; ) {
            char *e;
            long v = strtol(s, &e, 10);
            if (e == s || v < 0 || (*e && *e != ',')) { fprintf(stderr, "bad device list: %s\n", list); exit(2); }
            devices[n++] = (int)v;
            s = *e == ',' ? e + 1 : e;
        }
    }
    if (!n) devices[n++] = 0;
    return n;
}
#endif
