/* The CUDA helper's loader in wex_decode.c (COLI_WEX_CUDA_LIB), no GPU: a CPU stand-in with the
 * helper's exports (tests/fake_wex_cuda.c) loaded the way an engine loads kaira_wex_cuda, and
 * WNR3 records made by the reference encoder (tests/wnr3_fixture.h). The loader runs once per
 * process, so each case runs in a child process of its own with its own environment.
 *
 * With the helper, coli_wex_decode_record (and coli_wex_decode_record_to, straight into
 * six buffers) hands it a WNR3 record and gets the CPU decoder's bytes; a record it does not decode (unavailable, or a feature it lacks) is decoded on the CPU,
 * with the same bytes; a damaged record is refused either way. The device and batch calls give
 * each record's status, write only the destinations of records that decoded, and report 1 when
 * the helper lacks their feature. A missing library, one of another ABI, COLI_WEX_CUDA_LIB=0 or
 * unset (even with a kaira_wex_cuda beside the executable: nothing is looked for) leave every
 * record on the CPU. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../compat.h"   /* setenv / unsetenv on Windows */
#define COLI_WEX_DECODE_API static inline
#include "../wex_decode.c"
#include "wex4_fixture.h"
#include "wnr3_fixture.h"

#ifdef _WIN32
#define DLEXT ".dll"
#else
#define DLEXT ".so"
#endif

enum { ROWS = 64, COLS = 256, SCALE_M = ROWS * COLS / 32, WEIGHT_M = ROWS * COLS / 2 };
#define RAW (3u * (SCALE_M + WEIGHT_M))
#define ALIGNED ((size_t)WEX_ALIGN_UP(RAW, WEX_ALIGN))
#define FEATURES (1u | 2u | 4u | 16u | 32u)   /* the stand-in's default */

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

static WexEntry entry_of(void) {
    WexEntry e;
    memset(&e, 0, sizeof e);
    e.scale_bytes = 3u * SCALE_M; e.weight_bytes = 3u * WEIGHT_M; e.total_bytes = (uint32_t)ALIGNED;
    for (unsigned m = 0; m < 3u; m++) {
        e.scale_shapes[m][0] = e.weight_shapes[m][0] = ROWS;
        e.scale_shapes[m][1] = COLS / 32; e.weight_shapes[m][1] = COLS / 2;
        e.scale_offsets[m] = m * SCALE_M; e.weight_offsets[m] = 3u * SCALE_M + m * WEIGHT_M;
    }
    return e;
}

static void fill(unsigned char *raw, uint32_t seed) {
    static const unsigned char mag[16] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4, 6, 7};
    for (size_t j = 0; j < RAW; j++) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        if (j < 3u * SCALE_M) { raw[j] = (unsigned char)(118u + seed % 5u); continue; }
        unsigned lo = mag[seed & 15u], hi = mag[(seed >> 4) & 15u];
        if (lo) lo |= seed >> 8 & 8u;
        if (hi) hi |= seed >> 9 & 8u;
        raw[j] = (unsigned char)(lo | hi << 4);
    }
}

/* Three experts as WNR3 records with their model registered; record 1 has one bit changed. */
static unsigned char raw[3][RAW], rec[3][ALIGNED];
static WexEntry ent[3];

static void make_records(void) {
    static unsigned char blob[WEX_NR2_MODEL_BYTES];
    const unsigned char *train[3];
    WexNr2Model model;
    uint32_t id = 0;
    for (int i = 0; i < 3; i++) { fill(raw[i], 20261116u + 7919u * (uint32_t)i); train[i] = raw[i]; }
    WexEntry plain = entry_of();
    wnr3f_train(&plain, train, 3, 0, blob, &model);
    CHECK(!coli_wex_register_model(blob, WEX_NR2_MODEL_BYTES, &id) && id == model.id);
    for (int i = 0; i < 3; i++) {
        size_t stored = wnr3f_encode(&model, &plain, raw[i], 0, (uint32_t)i, rec[i], ALIGNED);
        CHECK(stored && stored < ALIGNED);
        ent[i] = plain; ent[i].flags = WEX_ENTRY_NEURAL_RANS3; ent[i].total_bytes = (uint32_t)stored;
    }
    rec[1][200] ^= 0x10u;
}

/* coli_wex_decode_record of record i in a slab: its status, and 1 in *same when the slab holds raw[i] */
static int decode(int i, int *same) {
    static unsigned char slab[ALIGNED];
    memset(slab, 0xA5, sizeof slab);
    memcpy(slab + wex_record_read_offset(&ent[i]), rec[i], ent[i].total_bytes);
    int status = coli_wex_decode_record(&ent[i], slab, sizeof slab);
    *same = !memcmp(slab, raw[i], RAW);
    return status;
}

/* coli_wex_decode_record_to of record i into six buffers laid out as the slab: its status, *same as above */
static int decode_to(int i, int *same) {
    static unsigned char out[RAW];
    unsigned char *scales[3], *weights[3];
    memset(out, 0xA5, sizeof out);
    for (unsigned m = 0; m < 3u; m++) { scales[m] = out + ent[i].scale_offsets[m]; weights[m] = out + ent[i].weight_offsets[m]; }
    int status = coli_wex_decode_record_to(&ent[i], rec[i], scales, weights);
    *same = !memcmp(out, raw[i], RAW);
    return status;
}

static int untouched(const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; i++) if (p[i] != 0xA5) return 0;
    return 1;
}

/* device buffers of record slot k (host memory for the stand-in): scales then weights, as the slab */
static unsigned char dev[3][RAW];
static unsigned char *dscale[9], *dweight[9];
static void device_views(void) {
    memset(dev, 0xA5, sizeof dev);
    for (int k = 0; k < 3; k++)
        for (unsigned m = 0; m < 3u; m++) {
            dscale[3 * k + m] = dev[k] + ent[0].scale_offsets[m];
            dweight[3 * k + m] = dev[k] + ent[0].weight_offsets[m];
        }
}

static int child(const char *name) {
    int same = 0, status[3];
    make_records();
    const unsigned int bits = coli_wex_cuda_features();
    if (!strcmp(name, "helper")) {
        CHECK(bits == FEATURES);
        CHECK(!decode(0, &same) && same && coli_wex_cuda_decode_count() == 1);
        CHECK(decode(1, &same) == -1);                                     /* refused by the helper, then the CPU */
        CHECK(!decode(2, &same) && same && coli_wex_cuda_decode_count() == 2);
        CHECK(!decode_to(0, &same) && same && coli_wex_cuda_decode_count() == 3);   /* straight to buffers: the helper */
        CHECK(decode_to(1, &same) == -1 && coli_wex_cuda_decode_count() == 3);
        __atomic_store_n(&wex_cuda.host_decodes, UINT64_C(2), __ATOMIC_RELAXED);   /* the counts below, as before */
        device_views();
        CHECK(!coli_wex_cuda_decode_to_device(&ent[0], rec[0], 0, dscale, dweight) && !memcmp(dev[0], raw[0], RAW));
        CHECK(coli_wex_cuda_decode_to_device(&ent[1], rec[1], 0, dscale + 3, dweight + 3) == -1 && untouched(dev[1], RAW));
        CHECK(coli_wex_cuda_device_decode_count() == 1);
        const WexEntry *es[3] = {&ent[0], &ent[1], &ent[2]};
        const unsigned char *rs[3] = {rec[0], rec[1], rec[2]};
        device_views();
        CHECK(coli_wex_cuda_decode_batch_to_device(3, es, rs, 0, dscale, dweight, status) == 3 &&
              status[0] == 1 && status[1] == 1 && status[2] == 1 && untouched(dev[0], sizeof dev));   /* no feature 8 */
        unsigned char *dst[3] = {dev[0], dev[1], dev[2]};
        CHECK(coli_wex_cuda_decode_batch_to_host(3, es, rs, 0, dst, status) == 1 && status[0] == 0 && status[1] == -1 &&
              status[2] == 0 && !memcmp(dev[0], raw[0], RAW) && untouched(dev[1], RAW) && !memcmp(dev[2], raw[2], RAW));
        CHECK(coli_wex_cuda_decode_count() == 4);
        CHECK(!coli_wex_cuda_host_register(dev, sizeof dev) && !coli_wex_cuda_host_unregister(dev));
    } else if (!strcmp(name, "batch-device")) {   /* FAKE_WEX_FEATURES=63 */
        const WexEntry *es[2] = {&ent[2], &ent[0]};
        const unsigned char *rs[2] = {rec[2], rec[0]};
        CHECK(bits == 63u);
        device_views();
        CHECK(!coli_wex_cuda_decode_batch_to_device(2, es, rs, 0, dscale, dweight, status) && !status[0] && !status[1] &&
              !memcmp(dev[0], raw[2], RAW) && !memcmp(dev[1], raw[0], RAW) && coli_wex_cuda_device_decode_count() == 2);
    } else if (!strcmp(name, "host-fail")) {      /* FAKE_WEX_HOST_FAIL=1 */
        CHECK(bits == FEATURES);
        CHECK(!decode(0, &same) && same && coli_wex_cuda_decode_count() == 0);
        CHECK(decode(1, &same) == -1);
        CHECK(!decode_to(2, &same) && same && coli_wex_cuda_decode_count() == 0);   /* the CPU after the helper's 1 */
        CHECK(decode_to(1, &same) == -1);
    } else if (!strcmp(name, "batch-fail")) {     /* FAKE_WEX_BATCH_FAIL=0 */
        const WexEntry *es[2] = {&ent[0], &ent[2]};
        const unsigned char *rs[2] = {rec[0], rec[2]};
        unsigned char *dst[2] = {dev[0], dev[1]};
        device_views();
        CHECK(coli_wex_cuda_decode_batch_to_host(2, es, rs, 0, dst, status) == 1 && status[0] == 1 && status[1] == 0 &&
              !memcmp(dev[1], raw[2], RAW) && coli_wex_cuda_decode_count() == 1);
    } else if (!strcmp(name, "no-nr3")) {         /* FAKE_WEX_FEATURES=5: no WNR3 in the helper */
        CHECK(bits == (1u | 4u));
        CHECK(!decode(0, &same) && same && coli_wex_cuda_decode_count() == 0);
        CHECK(!decode_to(0, &same) && same && coli_wex_cuda_decode_count() == 0);
        device_views();
        CHECK(coli_wex_cuda_decode_to_device(&ent[0], rec[0], 0, dscale, dweight) == 1 && untouched(dev[0], RAW));
    } else if (!strcmp(name, "cpu")) {   /* no helper: every record on the CPU */
        CHECK(bits == 0);
        CHECK(!decode(0, &same) && same && decode(1, &same) == -1 && coli_wex_cuda_decode_count() == 0);
        CHECK(!decode_to(0, &same) && same && decode_to(1, &same) == -1 && coli_wex_cuda_decode_count() == 0);
        device_views();
        CHECK(coli_wex_cuda_decode_to_device(&ent[0], rec[0], 0, dscale, dweight) == 1 && untouched(dev[0], RAW));
        CHECK(coli_wex_cuda_host_register(dev, sizeof dev) == 1);
    } else return 2;
    return failures ? 1 : 0;
}

static char self[4096];
static int spawn(const char *name) {
    fflush(stdout); fflush(stderr);
#ifdef _WIN32
    return (int)_spawnl(_P_WAIT, self, self, "child", name, NULL);
#else
    pid_t pid = fork();
    if (pid == 0) { execl(self, self, "child", name, (char *)NULL); _exit(127); }
    int st = 0;
    if (pid < 0 || waitpid(pid, &st, 0) != pid) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
#endif
}

static void run(const char *name, const char *what) {
    int rc = spawn(name);
    printf("  %s %s%s\n", rc ? "FAIL" : "ok  ", what, rc ? " (see the lines above)" : "");
    failures += rc != 0;
}

static int copy_file(const char *from, const char *to) {
    FILE *in = fopen(from, "rb"), *out = in ? fopen(to, "wb") : NULL;
    char buffer[65536];
    size_t n;
    int ok = in && out;
    while (ok && (n = fread(buffer, 1, sizeof buffer, in)) > 0) ok = fwrite(buffer, 1, n, out) == n;
    if (in) fclose(in);
    if (out && fclose(out)) ok = 0;
    return ok ? 0 : -1;
}

int main(int argc, char **argv) {
    char fake[4200], beside[4200];
    setbuf(stdout, NULL);
    if (argc == 3 && !strcmp(argv[1], "child")) return child(argv[2]);
#ifndef COLI_WEX_CUDA_LOADER
    (void)argc; (void)argv;
    printf("test_wex_cuda_loader: skipped (the helper is loaded on Windows and Linux only)\n");
    return 0;
#else
    unsetenv("FAKE_WEX_ABI"); unsetenv("FAKE_WEX_FEATURES"); unsetenv("FAKE_WEX_HOST_FAIL"); unsetenv("FAKE_WEX_BATCH_FAIL");
    unsetenv("DSV4_CUDA_MOE_MASKED");
    /* absolute paths: this executable, the stand-in beside it in tests/ */
#ifdef _WIN32
    if (!_fullpath(self, argv[0], sizeof self)) return 1;
    if (!strstr(self, ".exe") && strlen(self) + 5 < sizeof self) strcat(self, ".exe");
#else
    if (!realpath(argv[0], self)) return 1;
#endif
    size_t dir = strlen(self);
    while (dir && self[dir - 1] != '/' && self[dir - 1] != '\\') dir--;
    snprintf(fake, sizeof fake, "%.*sfake_wex_cuda" DLEXT, (int)dir, self);
    snprintf(beside, sizeof beside, "%.*skaira_wex_cuda" DLEXT, (int)dir, self);
    remove(beside);

    setenv("COLI_WEX_CUDA_LIB", fake, 1);
    run("helper", "the helper decodes WNR3 records to host slabs and device buffers with the CPU decoder's bytes; "
                  "a damaged record refused, its destinations untouched; the device batch reports 1 without feature 8");
    setenv("FAKE_WEX_FEATURES", "63", 1);
    run("batch-device", "with feature 8: a batch to device buffers, each record where it was asked");
    setenv("FAKE_WEX_FEATURES", "5", 1);
    run("no-nr3", "a helper without feature 16: WNR3 records decode on the CPU, the device call reports 1");
    unsetenv("FAKE_WEX_FEATURES");
    setenv("FAKE_WEX_HOST_FAIL", "1", 1);
    run("host-fail", "the helper does not decode (CUDA unavailable): the CPU decodes the record, same bytes");
    unsetenv("FAKE_WEX_HOST_FAIL");
    setenv("FAKE_WEX_BATCH_FAIL", "0", 1);
    run("batch-fail", "a host batch whose copy back fails for one record: status 1 for it, the other one written");
    unsetenv("FAKE_WEX_BATCH_FAIL");
    setenv("FAKE_WEX_ABI", "2", 1);
    run("cpu", "a helper of another ABI: not used, every record on the CPU");
    unsetenv("FAKE_WEX_ABI");
    setenv("COLI_WEX_CUDA_LIB", "0", 1);
    run("cpu", "COLI_WEX_CUDA_LIB=0: no helper, every record on the CPU");
    char missing[4300];
    snprintf(missing, sizeof missing, "%s.missing", fake);
    setenv("COLI_WEX_CUDA_LIB", missing, 1);
    run("cpu", "COLI_WEX_CUDA_LIB names no file: every record on the CPU");
    unsetenv("COLI_WEX_CUDA_LIB");
    run("cpu", "COLI_WEX_CUDA_LIB unset: no helper, every record on the CPU");
    if (copy_file(fake, beside)) { printf("  FAIL cannot copy %s\n", fake); failures++; }
    else {
        run("cpu", "COLI_WEX_CUDA_LIB unset and kaira_wex_cuda" DLEXT " beside the executable: not looked for, the CPU");
        remove(beside);
    }
    if (failures) { printf("test_wex_cuda_loader: %d failure(s)\n", failures); return 1; }
    printf("test_wex_cuda_loader: ok\n");
    return 0;
#endif
}
