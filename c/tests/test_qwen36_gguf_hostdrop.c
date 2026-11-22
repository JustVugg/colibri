/* A GGUF matrix the CUDA tier holds gives its host rows back (q36_gq_drop), and the CPU reads it
 * back from the file, the same bytes, when it needs it after all (q36_gq_reload).
 *
 * The whole engine (qwen36.c's main) on a small dense GGUF, which tests/test_qwen36_gguf_hostdrop.py
 * writes, against the fake CUDA backend: it takes every block format (fake_block_elems) and
 * computes none of them.
 *   FAKE_MATMUL_FAIL=1  every device matmul fails, so the tier turns each handle off at its first
 *                       product and the CPU runs that matrix from then on: the logits must be the
 *                       CPU run's (COLI_CUDA=0), bit for bit.
 *   FAKE_MATMUL_FAIL=0  the device answers (garbage: the fake computes no block format); only the
 *                       memory counts.
 * At the first device matmul (every matrix placed, none read back yet) and at exit it prints the
 * process's private bytes and working set and the drop and reload counts:
 *   HOSTDROP placed private=<bytes> ws=<bytes> dropped=<n> bytes=<bytes> reloaded=<n>
 * Include order as in test_qwen36_tier_int8_engine.c. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static void hostdrop_line(const char *when) {
    size_t pv = 0, ws = 0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX c;
    memset(&c, 0, sizeof c); c.cb = sizeof c;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&c, sizeof c)) { pv = c.PrivateUsage; ws = c.WorkingSetSize; }
#else
    unsigned long sz, res, sh;   /* pages: resident, of them file-backed or shared */
    FILE *f = fopen("/proc/self/statm", "r");
    if (f && fscanf(f, "%lu %lu %lu", &sz, &res, &sh) == 3) { pv = (size_t)(res - sh) * sysconf(_SC_PAGESIZE); ws = (size_t)res * sysconf(_SC_PAGESIZE); }
    if (f) fclose(f);
#endif
    printf("HOSTDROP %s private=%zu ws=%zu dropped=%d bytes=%zu reloaded=%d\n",
           when, pv, ws, g_q36_gq_ndrop, g_q36_gq_dropped, g_q36_gq_nreload);
    fflush(stdout);
}
static void hostdrop_placed(void) { static int once; if (!once++) hostdrop_line("placed"); }
static void hostdrop_exit(void) { hostdrop_line("exit"); }

int main(int argc, char **argv) {
    const char *f = getenv("FAKE_MATMUL_FAIL");
    fake_block_elems = 32; fake_free_bytes = 8ull << 30;
    fake_matmul_fail = f && *f == '1';
    fake_matmul_hook = hostdrop_placed;
    atexit(hostdrop_exit);
    return qwen36_main_unused(argc, argv);
}
