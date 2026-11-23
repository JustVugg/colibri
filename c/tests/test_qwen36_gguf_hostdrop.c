/* A GGUF matrix the CUDA tier holds gives its host rows back (q36_gq_drop), and the CPU reads it
 * back from the file, the same bytes, when it needs it after all (q36_gq_reload). The loader reads
 * no matrix the tier takes: its upload reads the rows from the file through one staging buffer
 * (q36_gq_stream), so the load holds the matrices the CPU keeps and that buffer, no more.
 *
 * The whole engine (qwen36.c's main) on a small dense GGUF, which tests/test_qwen36_gguf_hostdrop.py
 * writes, against the fake CUDA backend: it takes every block format (fake_block_elems) and
 * computes none of them.
 *   FAKE_MATMUL_FAIL=1  every device matmul fails, so the tier turns each handle off at its first
 *                       product and the CPU runs that matrix from then on: the logits must be the
 *                       CPU run's (COLI_CUDA=0), bit for bit.
 *   FAKE_MATMUL_FAIL=0  the device answers (garbage: the fake computes no block format); only the
 *                       memory counts.
 *   FAKE_UPLOAD_ROWS=0  a backend without coli_cuda_tensor_upload_rows: each matrix goes up whole,
 *                       read into RAM first (qdw_gq_rows), as before the staging buffer.
 *   HOSTDROP_ROPE_PERM=1  q and k rows un-permuted at load as a muse-spark file's (g_q36_rope_perm).
 * At the first device matmul (every matrix placed, none read back yet) and at exit it prints the
 * process's private bytes and working set, their peaks so far, the drop and reload counts, the
 * uploads, the row ranges, and a hash of every block-format row the device was given, in order:
 *   HOSTDROP placed private=<bytes> ws=<bytes> peak_private=<bytes> peak_ws=<bytes> dropped=<n>
 *            bytes=<bytes> reloaded=<n> uploads=<n> ranges=<n> rows=<hex>
 * Include order as in test_qwen36_tier_int8_engine.c. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static uint64_t rows_hash = 1469598103934665603ull;   /* FNV-1a */
static void hostdrop_rows(int fmt, int I, int row0, int n, const void *rows) {
    const uint8_t *p = rows;
    const size_t len = (size_t)gq_row_bytes(fmt - 100, I) * (size_t)n;
    (void)row0;
    for (size_t i = 0; i < len; i++) { rows_hash ^= p[i]; rows_hash *= 1099511628211ull; }
}
static void hostdrop_line(const char *when) {
    size_t pv = 0, ws = 0, ppv = 0, pws = 0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX c;
    memset(&c, 0, sizeof c); c.cb = sizeof c;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&c, sizeof c)) {
        pv = c.PrivateUsage; ws = c.WorkingSetSize; ppv = c.PeakPagefileUsage; pws = c.PeakWorkingSetSize;
    }
#else
    unsigned long sz, res, sh;   /* pages: resident, of them file-backed or shared */
    char line[256];
    FILE *f = fopen("/proc/self/statm", "r");
    if (f && fscanf(f, "%lu %lu %lu", &sz, &res, &sh) == 3) { pv = (size_t)(res - sh) * sysconf(_SC_PAGESIZE); ws = (size_t)res * sysconf(_SC_PAGESIZE); }
    if (f) fclose(f);
    f = fopen("/proc/self/status", "r");   /* VmHWM: the peak resident set, for both peaks */
    while (f && fgets(line, sizeof line, f)) if (!strncmp(line, "VmHWM:", 6)) ppv = pws = (size_t)strtoull(line + 6, NULL, 10) << 10;
    if (f) fclose(f);
#endif
    printf("HOSTDROP %s private=%zu ws=%zu peak_private=%zu peak_ws=%zu dropped=%d bytes=%zu reloaded=%d uploads=%d ranges=%d rows=%016llx\n",
           when, pv, ws, ppv, pws, g_q36_gq_ndrop, g_q36_gq_dropped, g_q36_gq_nreload, fake_uploads, fake_upload_ranges,
           (unsigned long long)rows_hash);
    fflush(stdout);
}
static void hostdrop_placed(void) { static int once; if (!once++) hostdrop_line("placed"); }
static void hostdrop_exit(void) { hostdrop_line("exit"); }

int main(int argc, char **argv) {
    const char *f = getenv("FAKE_MATMUL_FAIL"), *r = getenv("FAKE_UPLOAD_ROWS"), *p = getenv("HOSTDROP_ROPE_PERM");
    fake_block_elems = 32; fake_free_bytes = 8ull << 30;
    fake_matmul_fail = f && *f == '1';
    fake_upload_rows_absent = r && *r == '0';
    g_q36_rope_perm = p && *p == '1';   /* q and k rows un-permuted at load, a head at a time (muse-spark's files) */
    fake_matmul_hook = hostdrop_placed;
    fake_rows_hook = hostdrop_rows;
    atexit(hostdrop_exit);
    return qwen36_main_unused(argc, argv);
}
