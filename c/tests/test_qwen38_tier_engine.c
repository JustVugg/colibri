/* The qwen38 engine on the fake CUDA backend, driven through its own main().
 *
 * What the tier's unit test (test_qwen36_tier_fp8.c) cannot see is the
 * engine side: that q38_tier_start() brings the tier up from a real FP8
 * fixture (scale banks resident, LUT published, fmt 8), that the decode loop
 * hands qt_note the slab bytes of native slots only, that the reduced CPU
 * list (mask -> cpu_idx/cpu_rank -> batch load) leaves the arithmetic alone,
 * and that recycling the RAM slot right after qt_note corrupts nothing. The
 * fake backend refuses every issue, so the CPU computes all experts and the
 * oracle tokens/logits must still match exactly; the hit/miss counters pin
 * that residency was consulted anyway. Real-GPU parity is a hardware run.
 *
 * Needs ./qwen38_tiny_fp8 (make qwen38-tiny-fp8-generate). Include order as
 * in test_qwen36_tier_int8_engine.c: engine first, then the fake backend,
 * then the tier source, so the tier's statics (G, qs()) are readable here. */
#define main qwen38_main_unused
#include "../qwen38.c"
#undef main

#ifdef _WIN32
#undef setenv
#undef unsetenv
#define unsetenv(name) _putenv_s(name, "")
#endif

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int t_fails;
static void tk(int ok, const char *what) {
    if (ok) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    t_fails++;
}

int main(void) {
    /* the fixture: 4 layers x 4 experts, hidden 32, inter 8, top-2, all
     * routed experts native e4m3 with one 128x128 block scale per matrix */
    setenv("SNAP", "./qwen38_tiny_fp8", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("NOSTREAM", "1", 1);
    setenv("USAGE_SAVE", "0", 1);               /* leave the fixture directory alone */
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    /* the fake backend counts matmuls but computes nothing, so a trunk matrix
     * placed on it would answer garbage; keep the trunk on the CPU here (the
     * placement itself is pinned by test_qwen36_tier_dense) */
    setenv("Q38_TRUNK_GPU", "0", 1);
    /* room for six of the sixteen experts, so promotion has to swap */
    size_t exp_bytes = 3 * dev_alloc_footprint(32 * 8) + 3 * dev_alloc_footprint(1 * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(6 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    fake_ndev = 1; fake_uploads = 0; fake_lut_published = 0;

    /* cap 1 < top-2: every expert is reloaded into the single RAM slot each
     * time -- the slot is recycled right after the tier looked at it */
    char *argv[] = { (char *)"qwen38", (char *)"1", (char *)"8", (char *)"./qwen38_tiny_fp8/ref.json" };
    int rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "engine on the fake tier reproduces the oracle tokens and logits (cap 1)");

    tk(G.on, "tier started from q38_tier_start on the FP8 fixture");
    tk(G.wfmt == 8, "streaming mode, weight format 8");
    tk(G.exp_bytes == exp_bytes, "exp_bytes charged at cudaMalloc granularity (three matrices, three scale tables)");
    tk(fake_lut_published, "e4m3 LUT published before the first upload");
    tk(fake_uploads > 0 && fake_uploads % 3 == 0, "uploads come in gate/up/down triples");
    tk(last_fmt == 8 && last_bytes == (size_t)G.D * G.Ih, "fmt 8 uploads carry one byte per element");

    pthread_mutex_lock(&G.mx);
    unsigned long long hits = 0; for (int i = 0; i < G.ndev; i++) hits += G.hits[i];
    unsigned long long miss = G.miss;
    int residents = 0, stale = 0;
    for (int l = 0; l < G.nl; l++) for (int e = 0; e < G.ne; e++) {
        residents += qs(l, e)->resident;
        stale += (qs(l, e)->g4 != NULL || qs(l, e)->gs != NULL);
    }
    size_t used = G.used[0];
    pthread_mutex_unlock(&G.mx);
    tk(hits + miss > 0, "qt_issue was consulted for routed experts");
    tk(hits > 0, "resident experts were routed (hits counted even though the fake backend refuses to issue)");
    tk(residents > 0 && residents <= 6, "residency stays within the six-expert budget");
    tk(used == (size_t)residents * G.exp_bytes, "bytes accounted exactly per resident");
    tk(stale == 0, "no slot keeps a pointer into the engine's recycled RAM slot");

    qt_shutdown();
    tk(!G_fp8_stream && !G.on, "shutdown leaves the mode");

    /* Second pass: the dense trunk on the fake tier. Every dense matrix of the
     * fixture is offered (Q38_TRUNK_MIN_KB=0), quantized to int8 per row and
     * "uploaded"; the fake backend now computes fmt 1 from the uploaded bytes,
     * so the run must reproduce the oracle within its limits like the CPU int8
     * reference (Q38_TRUNK_CPU_INT8=1) does, and every offered matrix must have
     * been placed and released again at shutdown. */
    printf(" trunk on the fake tier\n");
    setenv("Q38_TRUNK_GPU", "1", 1);
    setenv("Q38_TRUNK_MIN_KB", "0", 1);
    setenv("COLI_PLACE", "auto", 1);             /* the placer decides per offer */
    setenv("CUDA_EXPERT_GB", "0.5", 1);          /* room for the trunk and every expert */
    fake_dense_compute = 1; fake_uploads = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "engine with the int8 trunk on the fake tier stays within the oracle's limits");
    tk(g_trunk_n > 0 && qt_dense_count() == g_trunk_n, "every offered dense matrix was placed and uploaded");
    tk(fake_uploads >= g_trunk_n, "one upload per placed matrix");
    qt_shutdown();
    tk(qt_dense_count() == 0, "dense handles released at shutdown");

    if (t_fails) { printf("test_qwen38_tier_engine: %d failure(s)\n", t_fails); return 1; }
    printf("test_qwen38_tier_engine: ok\n");
    return 0;
}
