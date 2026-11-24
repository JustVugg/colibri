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
 * Needs ./qwen38_tiny_fp8 (make qwen38-tiny-fp8-generate). Given a snapshot
 * converted to int4-g64 experts on argv (make qwen38-tiny-int4-check passes
 * ./qwen38_tiny_int4), a third pass checks that the tier declines those
 * experts -- it streams e4m3 bytes only -- and that the engine then computes
 * them on the CPU and reproduces the int4 reference. Include order as in
 * test_qwen36_tier_int8_engine.c: engine first, then the fake backend, then
 * the tier source, so the tier's statics (G, qs()) are readable here. */
#define main qwen38_main_unused
#include "../qwen38.c"
#undef main

#ifdef _WIN32
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

/* The load's own steps, model_init then q38_tier_start, with the trunk placed on the
 * fake tier: a placed matrix keeps only its int8 rows from then on (its BF16 rows freed
 * at placement, not later in q38_trunk_cpu_int8), and resident_weight_bytes follows.
 * Q38_TRUNK_CPU_INT8=0 (keep_bf16) leaves the BF16 rows and the count alone. */
static void trunk_bf16_at_placement(int keep_bf16) {
    Model m; model_init(&m, "./qwen38_tiny_fp8", 1, 8);
    uint64_t expect = m.resident_weight_bytes;
    q38_tier_start(&m, 1);
    int placed = 0, bf16 = 0, q8 = 0;
    for (int i = 0; i < g_trunk_n; i++) {
        Q38Weight *w = g_trunk[i].w;
        if (!w->gpu) continue;
        placed++; bf16 += w->data != NULL; q8 += w->q8 != NULL;
        if (!keep_bf16) expect = expect - q38_weight_bytes(w) + (uint64_t)w->rows * w->cols + (uint64_t)w->rows * sizeof(float);
    }
    tk(placed > 0 && placed == qt_dense_count(), "model_init + q38_tier_start placed the trunk");
    tk(q8 == placed, "every placed matrix has its int8 rows");
    if (keep_bf16) tk(bf16 == placed, "Q38_TRUNK_CPU_INT8=0: every placed matrix keeps its BF16 rows");
    else tk(bf16 == 0, "every placed matrix's BF16 rows freed at placement");
    tk(m.resident_weight_bytes == expect, "resident_weight_bytes counts int8 rows and scales for what was freed");
    qt_shutdown();
    q38_model_free(&m);
}

int main(int argc, char **argv_in) {
    /* the fixture: 4 layers x 4 experts, hidden 32, inter 8, top-2, all
     * routed experts native e4m3 with one 128x128 block scale per matrix */
    setenv("SNAP", "./qwen38_tiny_fp8", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("NOSTREAM", "1", 1);
    setenv("USAGE_SAVE", "0", 1);               /* leave the fixture directory alone */
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    /* Deterministic residency: the uploader is a thread, and on a two-vCPU
     * runner it did not run once during the eight tokens (queue of six,
     * 0 uploads, 0 hits, then five FAILs). With QT_UPLOAD_SYNC every
     * promotion is resident before the next group is issued. */
    setenv("QT_UPLOAD_SYNC", "1", 1);
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
    qt_fill_wait();   /* the last token's promotions, before the counters are read */

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
    size_t budget_off = G.budget[0], trunk_off = G_trunk_bytes[0];
    qt_shutdown();
    tk(qt_dense_count() == 0, "dense handles released at shutdown");
    trunk_bf16_at_placement(0);
    setenv("Q38_TRUNK_CPU_INT8", "0", 1);
    trunk_bf16_at_placement(1);
    unsetenv("Q38_TRUNK_CPU_INT8");

    /* The same with COLI_CUDA_CHAIN=1: the chain's device memory (q38cc_reserve_bytes) is
     * offered as a component of its own before the FP8 tier sizes its expert budget, as on
     * a GGUF, so the budget is the one above less the chain's bytes. Without the offer the
     * promotions fill what the chain allocates after q38_tier_start. */
    printf(" the CUDA chain's reserve beside the FP8 tier\n");
    setenv("COLI_CUDA_CHAIN", "1", 1);
    G_offer_n = 0;                               /* offers are per load; this process opens another model */
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "engine with the chain on the fake tier stays within the oracle's limits");
    size_t chain = 0; int chain_dev = QT_PLACE_CPU;
    for (int o = 0; o < G_offer_n; o++)
        if (!strcmp(G_offer[o].name, "chain")) { chain = G_offer[o].bytes; chain_dev = G_offer[o].dev; }
    tk(chain > 0 && chain_dev == G.dev[0], "the chain's reserve offered and placed on its card");
    tk(G_trunk_bytes[0] == trunk_off + chain, "the trunk's bytes on the card count the chain's reserve");
    tk(G.budget[0] == budget_off - chain, "the expert budget leaves the chain's reserve");
    qt_shutdown();
    setenv("COLI_CUDA_CHAIN", "0", 1);

    /* COLI_CUDA_TIER_WARM=1: the FP8 tier's warm start. Through the load's own steps
     * (model_init, q38_tier_start) and no forward yet, it makes the heat file's six experts
     * resident, those and no other, one upload triple each; unset (the default) the tier starts
     * empty as before. Then through main() with an expert history (COLI_USAGE) instead: the plan
     * is the history's most used first, and the run still gives the oracle's tokens and logits. */
    printf(" COLI_CUDA_TIER_WARM=1: the FP8 tier warm before the first prompt (6 of 16 experts fit)\n");
    {
        static const char hf[] = "./qwen38_tiny_fp8/warm_heat.bin", hist[] = "./qwen38_tiny_fp8/warm_usage.txt";
        static const int hot[6][2] = { { 0, 1 }, { 1, 3 }, { 2, 0 }, { 2, 2 }, { 3, 1 }, { 3, 3 } };
        uint32_t heat[3 + 16] = { 0x51544831u, 4, 4 };
        for (int i = 0; i < 6; i++) heat[3 + hot[i][0] * 4 + hot[i][1]] = 100u + i;
        setenv("Q38_TRUNK_GPU", "0", 1); setenv("COLI_PLACE", "off", 1); setenv("CUDA_EXPERT_GB", gb, 1);
        fake_dense_compute = 0;
        for (int a = 0; a < 2; a++) {
            FILE *f = fopen(hf, "wb"); if (f) { fwrite(heat, 4, 19, f); fclose(f); }   /* qt_shutdown saves over it */
            setenv("HEAT_FILE", hf, 1);
            if (a) setenv("COLI_CUDA_TIER_WARM", "1", 1); else unsetenv("COLI_CUDA_TIER_WARM");
            G_offer_n = 0;
            Model m; model_init(&m, "./qwen38_tiny_fp8", 1, 8);
            q38_tier_start(&m, 1);
            int up0 = fake_uploads;
            q38_tier_warm(&m, 0);
            pthread_mutex_lock(&G.mx);
            int on = G.on, res = 0, same = 1;
            for (int l = 0; l < 4; l++) for (int e = 0; e < 4; e++) {
                int r = qs(l, e)->resident; res += r; same &= r == (heat[3 + l * 4 + e] != 0);
            }
            pthread_mutex_unlock(&G.mx);
            if (a) {
                tk(on && res == 6 && same, "on: the heat file's six experts resident before any forward, and no other");
                tk(fake_uploads - up0 == 18, "one gate/up/down upload triple each");
            } else tk(on && res == 0 && fake_uploads == up0, "unset (the default): the FP8 tier starts empty, as before");
            qt_shutdown();
            q38_model_free(&m);
        }
        setenv("HEAT_FILE", "", 1); remove(hf);
        FILE *f = fopen(hist, "wb");
        if (f) {
            fprintf(f, "-1 4 4\n-2 1 %u\n", rt_hash("qwen38"));
            for (int l = 0; l < 4; l++) for (int e = 0; e < 4; e++) fprintf(f, "%d %d %d\n", l, e, l >= 2 ? 10 * (l * 4 + e) : 1);
            fclose(f);
        }
        setenv("COLI_USAGE", hist, 1); setenv("COLI_CUDA_TIER_WARM", "1", 1);
        G_offer_n = 0;
        rc =qwen38_main_unused(4, argv);
        tk(rc == 0, "on, through main() with an expert history: the oracle's tokens and logits");
        pthread_mutex_lock(&G.mx);
        int order = G.on && G.heat0 && G.fill_order;
        for (int i = 0; order && i < 6; i++) order = G.fill_order[i] == 15 - i;
        pthread_mutex_unlock(&G.mx);
        tk(order, "the warm start planned the history's most used first: layer 3's four, then (2,3) (2,2)");
        qt_shutdown();
        unsetenv("COLI_USAGE"); unsetenv("COLI_CUDA_TIER_WARM"); remove(hist);
        setenv("Q38_TRUNK_GPU", "1", 1); setenv("COLI_PLACE", "auto", 1); setenv("CUDA_EXPERT_GB", "0.5", 1);
    }

    if (argc > 1) {
        /* Third pass: the int4-g64 sidecar. The tier must stay off -- no
         * upload, no routed expert ever consulted -- and the CPU must answer
         * every expert from the sidecar, matching the int4 reference. */
        printf(" int4-g64 experts (%s)\n", argv_in[1]);
        char ref[2048]; snprintf(ref, sizeof ref, "%s/ref_int4.json", argv_in[1]);
        char *argv4[] = { (char *)"qwen38", (char *)"2", (char *)"8", ref };
        setenv("SNAP", argv_in[1], 1);
        setenv("Q38_EXPERT_INT4", "1", 1);
        fake_uploads = 0; fake_dense_compute = 0;
        unsigned long long before = G.miss; for (int i = 0; i < QT_MAX_DEV; i++) before += G.hits[i];
        rc = qwen38_main_unused(4, argv4);
        tk(rc == 0, "engine with int4 experts under COLI_CUDA=1 reproduces the int4 reference");
        tk(!G.on && !G_fp8_stream, "the tier declined: it streams native FP8 experts only");
        tk(fake_uploads == 0 && qt_dense_count() == 0, "nothing was uploaded, experts or trunk");
        unsigned long long consulted = G.miss; for (int i = 0; i < QT_MAX_DEV; i++) consulted += G.hits[i];
        tk(consulted == before, "qt_issue never took a routed expert off the CPU");
        unsetenv("Q38_EXPERT_INT4");
    }

    if (t_fails) { printf("test_qwen38_tier_engine: %d failure(s)\n", t_fails); return 1; }
    printf("test_qwen38_tier_engine: ok\n");
    return 0;
}
