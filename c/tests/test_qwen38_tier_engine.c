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
 * them on the CPU and reproduces the int4 reference. Given the experts-mxfp4/
 * fixture instead (make qwen38-tiny-mxfp4-check passes ./qwen38_tiny_mxfp4),
 * the third pass brings the tier up in its fmt 7 copy mode and checks the
 * logits: with every group refused they are the CPU run's to the bit; with the
 * fake computing the residents through the engine's own matmul_mxfp4 the tokens
 * are the oracle's and the logits within 1e-6 of the CPU's (the device's rows
 * join the sum after the CPU's), the same bits run to run. Include order as in
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

/* A file's bytes (a DUMP's last logits); NULL when it cannot be read. */
static float *slurp_f32(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); float *v = NULL; *n = 0;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    if (len > 0 && (v = (float *)malloc((size_t)len)) && fread(v, 1, (size_t)len, f) == (size_t)len) *n = (size_t)len / sizeof(float);
    else { free(v); v = NULL; }
    fclose(f);
    return v;
}
/* max |a - b| over max |b|, or -1 when either file is missing or the lengths differ;
 * *same: the two files hold the same bytes */
static double dump_rel(const char *a, const char *b, int *same) {
    size_t na, nb; float *x = slurp_f32(a, &na), *y = slurp_f32(b, &nb);
    double r = -1;
    *same = 0;
    if (x && y && na == nb && na) {
        double mx = 0, mag = 0;
        for (size_t i = 0; i < na; i++) {
            double d = fabs((double)x[i] - y[i]);
            if (!(d <= mx)) mx = d;   /* a NaN wins */
            if (fabs(y[i]) > mag) mag = fabs(y[i]);
        }
        r = mag > 0 ? mx / mag : mx;
        *same = !memcmp(x, y, na * sizeof(float));
    }
    free(x); free(y);
    return r;
}
/* The fake card computing a group of MXFP4 residents as the engine computes an expert
 * (matmul_mxfp4, silu(gate) * up, down), from the bytes the tier uploaded. */
static float mx_out[QT_MAX_ROWS * 256];
static int mx_groups;
static int mx_compute(ColiCudaTensor *const *g, ColiCudaTensor *const *u, ColiCudaTensor *const *d,
                      int count, const float *x) {
    const int D = g[0]->I, I = g[0]->O;
    float eg[256], eu[256];
    if (D > 256 || I > 256) return 0;
    for (int j = 0; j < count; j++) {
        if (g[j]->fmt != 7 || !g[j]->owned || !u[j]->owned || !d[j]->owned) return 0;
        matmul_mxfp4(eg, x + (size_t)j * D, (const uint8_t *)g[j]->w, (const uint8_t *)g[j]->sc, 1, D, I);
        matmul_mxfp4(eu, x + (size_t)j * D, (const uint8_t *)u[j]->w, (const uint8_t *)u[j]->sc, 1, D, I);
        for (int i = 0; i < I; i++) eg[i] = q38_silu(eg[i]) * eu[i];
        matmul_mxfp4(mx_out + (size_t)j * D, eg, (const uint8_t *)d[j]->w, (const uint8_t *)d[j]->sc, 1, I, D);
    }
    fake_group_out = mx_out; mx_groups++;
    return 1;
}
static int mx_sync_shared;
static int mx_sync_compute(ColiCudaTensor *const *g, ColiCudaTensor *const *u, ColiCudaTensor *const *d,
                           const int *rows, int count, float *y, const float *x) {
    const int D = g[0]->I, I = g[0]->O;
    float eg[256], eu[256];
    if (D > 256 || I > 256) return 0;
    int off = 0;
    for (int j = 0; j < count; j++) {
        if (g[j]->fmt != 7 || !g[j]->owned || !u[j]->owned || !d[j]->owned) return 0;
        if (rows[j] > 1) mx_sync_shared++;
        for (int r = 0; r < rows[j]; r++, off++) {
            matmul_mxfp4(eg, x + (size_t)off * D, (const uint8_t *)g[j]->w, (const uint8_t *)g[j]->sc, 1, D, I);
            matmul_mxfp4(eu, x + (size_t)off * D, (const uint8_t *)u[j]->w, (const uint8_t *)u[j]->sc, 1, D, I);
            for (int i = 0; i < I; i++) eg[i] = q38_silu(eg[i]) * eu[i];
            matmul_mxfp4(y + (size_t)off * D, eg, (const uint8_t *)d[j]->w, (const uint8_t *)d[j]->sc, 1, I, D);
        }
    }
    return 1;
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

    char mxi[2048]; FILE *mxf = NULL;
    if (argc > 1) { snprintf(mxi, sizeof mxi, "%s/experts-mxfp4/index.json", argv_in[1]); mxf = fopen(mxi, "rb"); }
    if (mxf) {
        fclose(mxf);
        /* Third pass: the experts-mxfp4/ sidecar on the tier (fmt 7 copy mode). The CPU's run
         * first, then the tier with every group refused (the CPU computes every expert: the
         * same bits), then the fake card computing the residents (twice: the same bits). */
        const char *snap = argv_in[1];
        printf(" mxfp4 experts on the tier (%s)\n", snap);
        char ref[2048], cpu[2048], refused[2048], dev1[2048], dev2[2048];
        snprintf(ref, sizeof ref, "%s/ref.json", snap);
        snprintf(cpu, sizeof cpu, "%s/tier_cpu.f32", snap);
        snprintf(refused, sizeof refused, "%s/tier_refused.f32", snap);
        snprintf(dev1, sizeof dev1, "%s/tier_dev1.f32", snap);
        snprintf(dev2, sizeof dev2, "%s/tier_dev2.f32", snap);
        char *argvm[] = { (char *)"qwen38", (char *)"1", (char *)"8", ref };
        setenv("SNAP", snap, 1);
        setenv("Q38_EXPERT_LOWBIT", "mxfp4", 1);
        setenv("Q38_TRUNK_GPU", "0", 1);         /* the trunk on the CPU: only the experts move */
        setenv("COLI_PLACE", "off", 1);
        G_offer_n = 0;                           /* the passes above left their offers */
        fake_dense_compute = 0;
        /* the fixture's expert: gate/up [8, 32] and down [32, 8], 128 nibble bytes each, and
         * 8 + 8 + 32 scale bytes; room for six of the sixteen, so promotion has to swap */
        size_t mx_bytes = 3 * dev_alloc_footprint(32 * 8 / 2) + 2 * dev_alloc_footprint(8) + dev_alloc_footprint(32);
        snprintf(gb, sizeof gb, "%.15f", (double)(6 * mx_bytes + mx_bytes / 2) / 1073741824.0);
        setenv("CUDA_EXPERT_GB", gb, 1);

        setenv("COLI_CUDA", "0", 1); setenv("DUMP", cpu, 1);
        rc = qwen38_main_unused(4, argvm);
        tk(rc == 0, "the CPU on the mxfp4 sidecar reproduces the oracle (cap 1)");

        setenv("COLI_CUDA", "1", 1); setenv("DUMP", refused, 1);
        fake_uploads = 0;
        rc = qwen38_main_unused(4, argvm);
        qt_fill_wait();
        tk(rc == 0, "the engine on the mxfp4 tier reproduces the oracle tokens and logits");
        tk(G.on && G.wfmt == 7 && G_mx4_stream && qt_streaming(), "tier started from q38_tier_start in its fmt 7 copy mode");
        tk(G.exp_bytes == mx_bytes, "exp_bytes: three nibble matrices and three scale BYTE tables");
        tk(fake_uploads > 0 && fake_uploads % 3 == 0 && last_fmt == 7 && last_bytes == (size_t)G.D * G.Ih / 2,
           "fmt 7 uploads in triples, half a byte per weight");
        pthread_mutex_lock(&G.mx);
        unsigned long long mhits = 0; for (int i = 0; i < G.ndev; i++) mhits += G.hits[i];
        int mres = 0, mstale = 0;
        for (int l = 0; l < G.nl; l++) for (int e = 0; e < G.ne; e++) {
            mres += qs(l, e)->resident;
            mstale += (qs(l, e)->g4 != NULL || qs(l, e)->gs != NULL);
        }
        size_t mused = G.used[0];
        pthread_mutex_unlock(&G.mx);
        tk(mhits > 0 && mres > 0 && mres <= 6 && mused == (size_t)mres * G.exp_bytes,
           "residents routed (hits), within the six-expert budget, bytes accounted per resident");
        tk(mstale == 0, "no slot keeps a pointer into the engine's records");
        int same = 0; double rel = dump_rel(refused, cpu, &same);
        tk(rel >= 0 && same, "every group refused: the last logits are the CPU run's to the bit");
        qt_shutdown();
        tk(!G_mx4_stream && !G.on, "shutdown leaves the mode");

        fake_own_mx4 = 1; fake_group_hook = mx_compute; mx_groups = 0;
        setenv("DUMP", dev1, 1);
        rc = qwen38_main_unused(4, argvm);
        tk(rc == 0, "the fake card computing the residents: the oracle tokens and logits");
        int computed = mx_groups;
        qt_shutdown();
        setenv("DUMP", dev2, 1);
        rc = qwen38_main_unused(4, argvm);
        tk(rc == 0, "and again");
        qt_shutdown();
        /* the chain beside it: its MoE kernels have no fmt 7, so the routed experts take the
         * host's step and the tier's groups compute the residents (the trunk on the fake card) */
        setenv("COLI_CUDA_CHAIN", "1", 1); setenv("Q38_TRUNK_GPU", "1", 1); setenv("Q38_TRUNK_MIN_KB", "0", 1);
        setenv("COLI_PLACE", "auto", 1); setenv("CUDA_EXPERT_GB", "0.5", 1); unsetenv("DUMP");
        G_offer_n = 0; fake_dense_compute = 1; mx_groups = 0;
        rc = qwen38_main_unused(4, argvm);
        tk(rc == 0, "with COLI_CUDA_CHAIN=1 and the trunk on the fake card: within the oracle's limits");
        int fmt7 = -1;
        tk(G.on && G.wfmt == 7 && !qt_chain_format(&fmt7, NULL) && mx_groups > 0,
           "the chain left the routed experts to the tier's groups (no chain format for fmt 7)");
        qt_shutdown();
        setenv("COLI_CUDA_CHAIN", "0", 1); setenv("Q38_TRUNK_GPU", "0", 1); setenv("COLI_PLACE", "off", 1);
        fake_dense_compute = 0;
        fake_own_mx4 = 0; fake_group_hook = NULL; fake_group_out = NULL;
        tk(computed > 0, "the card computed resident experts");
        rel = dump_rel(dev1, cpu, &same);
        printf("  last logits, card vs CPU: max |d| / max |CPU| = %.3g%s\n", rel, same ? " (identical)" : "");
        tk(rel >= 0 && rel <= 1e-6, "within 1e-6 of the CPU's (the device's rows join the sum after the CPU's)");
        dump_rel(dev2, dev1, &same);
        tk(same, "two runs with the same residency give the same bits");

        /* Fourth pass: opt-in CUDA-tier prefill levers (Q38_TIER_PREFILL_ROWWISE,
         * Q38_TIER_PREFILL_GROUP, Q38_WAVE_OMP, QT_PREFILL_GROUP, QT_FREEZE,
         * Q38_PREFILL_LAYER_MAJOR). Write a 6-expert heat file so residency is known. */
        {
            char hf[2048], d_row[2048], d_grp[2048], d_wav[2048], d_pg[2048], d_pg3[2048], d_ref0[2048], d_ref1[2048];
            char d_pc[2048], d_lm[2048];
            snprintf(hf, sizeof hf, "%s/tier_heat.bin", snap);
            snprintf(d_row, sizeof d_row, "%s/tier_row.f32", snap);
            snprintf(d_grp, sizeof d_grp, "%s/tier_grp.f32", snap);
            snprintf(d_wav, sizeof d_wav, "%s/tier_wav.f32", snap);
            snprintf(d_pg, sizeof d_pg, "%s/tier_pg.f32", snap);
            snprintf(d_pg3, sizeof d_pg3, "%s/tier_pg3.f32", snap);
            snprintf(d_ref0, sizeof d_ref0, "%s/tier_ref0.f32", snap);
            snprintf(d_ref1, sizeof d_ref1, "%s/tier_ref1.f32", snap);
            snprintf(d_pc, sizeof d_pc, "%s/tier_pc.f32", snap);
            snprintf(d_lm, sizeof d_lm, "%s/tier_lm.f32", snap);
            uint32_t heat[3 + 16] = { 0x48454154u, 4, 4, 10, 0, 10, 0,  0, 10, 0, 10,  10, 0, 0, 0,  0, 10, 0, 0 };
            FILE *hfp = fopen(hf, "wb");
            if (hfp) { fwrite(heat, sizeof heat, 1, hfp); fclose(hfp); }

            /* Cap 4 so half-cache waves (2 experts) exercise Q38_GROUP_OVERLAP as well. */
            char *argvm4[] = { (char *)"qwen38", (char *)"4", (char *)"8", ref };
            G_offer_n = 0;   /* clear the chain/trunk offers left by the pass above */
            fake_own_mx4 = 1; fake_group_hook = mx_compute; fake_sync_group_hook = mx_sync_compute;
            setenv("HEAT_FILE", hf, 1); setenv("QT_NO_WARMSTART", "0", 1);
            setenv("QT_FREEZE", "1", 1);
            snprintf(gb, sizeof gb, "%.15f", (double)(6 * mx_bytes + mx_bytes / 2) / 1073741824.0);
            setenv("CUDA_EXPERT_GB", gb, 1);

            printf(" Q38_TIER_PREFILL_ROWWISE / GROUP / WAVE_OMP on MXFP4 records\n");
            setenv("Q38_TIER_PREFILL_ROWWISE", "1", 1); setenv("Q38_TIER_PREFILL_GROUP", "0", 1);
            setenv("DUMP", d_row, 1);
            int r0 = qwen38_main_unused(4, argvm4);
            QtCounters q0; qt_counters(&q0);
            qt_shutdown();

            setenv("Q38_TIER_PREFILL_GROUP", "1", 1); setenv("Q38_WAVE_OMP", "0", 1);
            setenv("DUMP", d_grp, 1);
            int r1 = qwen38_main_unused(4, argvm4);
            qt_shutdown();

            unsigned long long w0 = g_q38_wave_experts;
            setenv("Q38_WAVE_OMP", "1", 1);
            setenv("DUMP", d_wav, 1);
            int r2 = qwen38_main_unused(4, argvm4);
            unsigned long long w1 = g_q38_wave_experts;
            qt_shutdown();
            unsetenv("Q38_WAVE_OMP");

            int same_grp = 0, same_wav = 0;
            dump_rel(d_grp, d_row, &same_grp);
            dump_rel(d_wav, d_row, &same_wav);
            tk(r0 == 0 && r1 == 0 && same_grp, "Q38_TIER_PREFILL_GROUP=1 matches per-row Q38_TIER_PREFILL_ROWWISE=1 to the bit");
            tk(r2 == 0 && same_wav && w1 > w0, "Q38_WAVE_OMP=1 matches per-row logits to the bit and ran wave regions");

            printf(" QT_PREFILL_GROUP=1 + QT_FREEZE=1 on MXFP4 records\n");
            int sh0 = mx_sync_shared;
            unsigned long long c0 = G_pg_calls, rw0 = G_pg_rows, rf0 = G_pg_refused;
            setenv("QT_PREFILL_GROUP", "1", 1); setenv("DUMP", d_pg, 1);
            int rp1 = qwen38_main_unused(4, argvm4);
            QtCounters q1; qt_counters(&q1);
            unsigned long long c1 = G_pg_calls - c0, rw1 = G_pg_rows - rw0, rf1 = G_pg_refused - rf0;
            qt_shutdown();

            c0 = G_pg_calls; rw0 = G_pg_rows;
            setenv("QT_PREFILL_GROUP_ROWS", "3", 1); setenv("DUMP", d_pg3, 1);
            int rp2 = qwen38_main_unused(4, argvm4);
            QtCounters q2; qt_counters(&q2);
            unsigned long long c2 = G_pg_calls - c0, rw2 = G_pg_rows - rw0;
            qt_shutdown();
            unsetenv("QT_PREFILL_GROUP_ROWS");

            int same_pg = 0, same_pg3 = 0;
            dump_rel(d_pg, d_row, &same_pg);
            dump_rel(d_pg3, d_row, &same_pg3);
            tk(rp1 == 0 && c1 > 0 && rf1 == 0 && rw1 > c1 && mx_sync_shared > sh0,
               "QT_PREFILL_GROUP=1 ran grouped device calls with multiple rows per expert");
            tk(rp2 == 0 && c2 > c1 && rw2 == rw1, "QT_PREFILL_GROUP_ROWS=3: same hit rows in more calls");
            tk(q1.hits == q0.hits && q1.miss == q0.miss && q2.hits == q0.hits && q2.miss == q0.miss && q1.swaps == 0,
               "QT_PREFILL_GROUP=1 matches per-row hit/miss counts; QT_FREEZE=1 prevents swaps");
            tk(same_pg && same_pg3, "QT_PREFILL_GROUP=1 (default and 3 rows/call) matches per-row logits to the bit");

            /* Every group refused: QT_PREFILL_GROUP=1 falls back to CPU and matches per-row refused to the bit. */
            fake_group_hook = NULL; fake_sync_group_hook = NULL;
            unsetenv("QT_PREFILL_GROUP"); setenv("Q38_TIER_PREFILL_GROUP", "0", 1);
            setenv("DUMP", d_ref0, 1);
            int rr0 = qwen38_main_unused(4, argvm4);
            qt_shutdown();

            c0 = G_pg_calls; rf0 = G_pg_refused;
            setenv("Q38_TIER_PREFILL_GROUP", "1", 1); setenv("QT_PREFILL_GROUP", "1", 1);
            setenv("DUMP", d_ref1, 1);
            int rr1 = qwen38_main_unused(4, argvm4);
            unsigned long long cr1 = G_pg_calls - c0, rfr1 = G_pg_refused - rf0;
            qt_shutdown();
            int same_ref = 0;
            dump_rel(d_ref1, d_ref0, &same_ref);
            tk(rr0 == 0 && rr1 == 0 && cr1 == 0 && rfr1 == rw1 && same_ref,
               "QT_PREFILL_GROUP=1 with every group refused matches per-row refused logits to the bit");

            /* QT_FREEZE=1 with room to promote: unfrozen uploads > 0, frozen uploads == 0 and swaps == 0. */
            fake_group_hook = mx_compute; fake_sync_group_hook = mx_sync_compute;
            snprintf(gb, sizeof gb, "%.15f", (double)(12 * mx_bytes + mx_bytes / 2) / 1073741824.0);
            setenv("CUDA_EXPERT_GB", gb, 1);
            setenv("COLI_CUDA_TIER_WARM", "0", 1);
            unsetenv("QT_FREEZE"); unsetenv("QT_PREFILL_GROUP"); unsetenv("DUMP");
            int ru0 = qwen38_main_unused(4, argvm4);
            QtCounters qu0; qt_counters(&qu0);
            qt_shutdown();
            setenv("QT_FREEZE", "1", 1);
            int ru1 = qwen38_main_unused(4, argvm4);
            QtCounters qu1; qt_counters(&qu1);
            qt_shutdown();
            unsetenv("COLI_CUDA_TIER_WARM");
            tk(ru0 == 0 && ru1 == 0 && qu0.uploads > 0 && qu1.uploads == 0 && qu1.swaps == 0,
               "QT_FREEZE=1 prevents uploads and swaps after warm start");

            printf(" Q38_PREFILL_LAYER_MAJOR=1 on the CUDA chain + MXFP4 tier\n");
            setenv("COLI_CUDA_CHAIN", "1", 1); setenv("COLI_CUDA_CHAIN_ROWS", "2", 1);
            setenv("Q38_TRUNK_GPU", "1", 1); setenv("Q38_TRUNK_MIN_KB", "0", 1);
            setenv("COLI_PLACE", "auto", 1); setenv("CUDA_EXPERT_GB", "0.5", 1);
            setenv("QT_FREEZE", "1", 1); setenv("QT_PREFILL_GROUP", "1", 1);
            G_offer_n = 0; fake_dense_compute = 1;
            setenv("Q38_PREFILL_LAYER_MAJOR", "0", 1); setenv("DUMP", d_pc, 1);
            int rlm0 = qwen38_main_unused(4, argvm4);
            qt_shutdown();
            unsigned long long lm0 = g_q38_lm_pieces;
            setenv("Q38_PREFILL_LAYER_MAJOR", "1", 1); setenv("Q38_PREFILL_LAYER_MAJOR_ROWS", "4", 1);
            setenv("DUMP", d_lm, 1);
            G_offer_n = 0;
            int rlm1 = qwen38_main_unused(4, argvm4);
            unsigned long long lm1 = g_q38_lm_pieces;
            qt_shutdown();
            int same_lm = 0;
            dump_rel(d_lm, d_pc, &same_lm);
            tk(rlm0 == 0 && rlm1 == 0 && lm1 > lm0 && same_lm,
               "Q38_PREFILL_LAYER_MAJOR=1 (2-row pieces, 4-row segments) matches piece-by-piece logits to the bit");

            setenv("COLI_CUDA_CHAIN", "0", 1); unsetenv("COLI_CUDA_CHAIN_ROWS");
            setenv("Q38_TRUNK_GPU", "0", 1); setenv("COLI_PLACE", "off", 1);
            unsetenv("Q38_PREFILL_LAYER_MAJOR"); unsetenv("Q38_PREFILL_LAYER_MAJOR_ROWS");
            unsetenv("Q38_TIER_PREFILL_ROWWISE"); unsetenv("Q38_TIER_PREFILL_GROUP");
            unsetenv("QT_PREFILL_GROUP"); unsetenv("QT_FREEZE");
            setenv("HEAT_FILE", "", 1); setenv("QT_NO_WARMSTART", "1", 1);
            fake_dense_compute = 0; fake_own_mx4 = 0; fake_group_hook = NULL; fake_sync_group_hook = NULL; fake_group_out = NULL;
            remove(hf); remove(d_row); remove(d_grp); remove(d_wav); remove(d_pg); remove(d_pg3);
            remove(d_ref0); remove(d_ref1); remove(d_pc); remove(d_lm);
        }
        unsetenv("Q38_EXPERT_LOWBIT"); unsetenv("DUMP");
    } else if (argc > 1) {
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
