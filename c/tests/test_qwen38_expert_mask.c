/* Q38_EXPERT_MASK (qwen38_core.h): routing restricted to a per-layer allow-list, the
 * engine driven through its own main() over the tiny FP8 fixture (4 layers x 4 experts,
 * top-2), first on the CPU, then on the fake CUDA backend (tests/qwen36_fake_cuda.h).
 *
 *  - argv[1], the mask tools/q38_expert_mask.py built from a known history (the Makefile
 *    writes both): it holds the expected top-2 per layer, every traced row picks only
 *    those experts with gates summing to 1, and so does every pick the run's history
 *    counts, prompt rows batched or not;
 *  - unset, and a mask that allows every expert: the same logits to the bit and the same
 *    history, both passing the oracle;
 *  - malformed masks stop the engine;
 *  - on the fake tier only allowed experts reach VRAM, each misses at most once (its first
 *    route) and none routes outside the mask; with room for fewer, the rest miss as usual;
 *  - on the fake CUDA chain the routes stay inside the mask, and an allow-all mask gives
 *    the unset run's logits to the bit;
 *  - a masked run leaves the routing history alone unless USAGE_SAVE=1.
 *
 * Needs ./qwen38_tiny_fp8 (make qwen38-tiny-fp8-generate); run by qwen38-expert-mask-check. */
#define main qwen38_main_unused
#include "../qwen38.c"
#undef main

#ifdef _WIN32
#undef unsetenv
#define unsetenv(name) _putenv_s(name, "")
#endif

#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

#define FX "./qwen38_tiny_fp8"
static int t_fails;
static void tk(int ok, const char *what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    t_fails += !ok;
}
/* the expected mask, the tool's top-2 of the Makefile's history: allowed[layer][expert] */
static const int fx_allow[4][4] = {{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 1, 1}, {1, 1, 0, 0}};

static int fx_read(const char *path, float *buf, int max) {
    int n = 0; FILE *f = fopen(path, "rb");
    if (f) { n = (int)fread(buf, sizeof(float), (size_t)max, f); fclose(f); }
    remove(path);
    return n;
}
static long fx_slurp(const char *path, char *buf, long max) {
    long n = 0; FILE *f = fopen(path, "rb");
    if (f) { n = (long)fread(buf, 1, (size_t)max, f); fclose(f); }
    return n;
}
static void fx_write(const char *path, const char *text) {
    FILE *f = fopen(path, "w"); if (f) { fputs(text, f); fclose(f); }
}
/* one run: rc, last logits (n); trace: ROUTE_TRACE (the process's first run only: the
 * engine opens the stream once); history: the run's routing counts (COLI_USAGE) */
static int fx_run(const char *mask, const char *trace, const char *history, float *logits, int *n) {
    char *argv[] = { (char *)"qwen38", (char *)"4", (char *)"8", (char *)FX "/ref.json" };
    if (mask) setenv("Q38_EXPERT_MASK", mask, 1); else unsetenv("Q38_EXPERT_MASK");
    if (trace) setenv("ROUTE_TRACE", trace, 1);
    if (history) { remove(history); setenv("COLI_USAGE", history, 1); setenv("USAGE_SAVE", "1", 1); }
    setenv("DUMP", FX "/mask_logits.f32", 1);
    int rc = qwen38_main_unused(4, argv);
    unsetenv("DUMP"); unsetenv("ROUTE_TRACE"); unsetenv("Q38_EXPERT_MASK"); unsetenv("COLI_USAGE");
    setenv("USAGE_SAVE", "0", 1);   /* leave the fixture's own history alone */
    *n = fx_read(FX "/mask_logits.f32", logits, 256);
    return rc;
}
/* every traced row routes inside fx_allow with gates summing to 1 (4 decimals a gate) */
static int fx_trace_obeys(const char *path, int *rows) {
    FILE *f = fopen(path, "r"); if (!f) return 0;
    int ok = 1; char line[512]; *rows = 0;
    while (fgets(line, sizeof line, f)) {
        int call, row, layer, used; char *p = line;
        if (sscanf(p, "%d %d %d%n", &call, &row, &layer, &used) != 3 || layer < 0 || layer > 3) { ok = 0; break; }
        p += used;
        int id, k = 0; float g, sum = 0;
        while (sscanf(p, " %d:%f%n", &id, &g, &used) == 2) {
            p += used; k++; sum += g;
            if (id < 0 || id > 3 || !fx_allow[layer][id]) ok = 0;
        }
        if (k != 2 || fabsf(sum - 1.f) > 2e-4f) ok = 0;
        (*rows)++;
    }
    fclose(f);
    return ok && *rows > 0;
}
/* every count of a run's history ("<layer> <expert> <count>") inside fx_allow; *total: the picks */
static int fx_history_obeys(const char *path, long *total) {
    FILE *f = fopen(path, "r"); if (!f) return 0;
    int ok = 1, l, e; unsigned c; *total = 0;
    while (fscanf(f, "%d %d %u", &l, &e, &c) == 3) {
        if (l < 0) continue;   /* the header records */
        if (l > 3 || e < 0 || e > 3 || !fx_allow[l][e]) ok = 0;
        *total += c;
    }
    fclose(f);
    return ok && *total > 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <mask from tools/q38_expert_mask.py>\n", argv[0]); return 2; }
    const char *tool_mask = argv[1];
    setenv("SNAP", FX, 1);
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("NOSTREAM", "1", 1);
    setenv("USAGE_SAVE", "0", 1);
    unsetenv("COLI_CUDA");
    float l0[256], l1[256], l2[256]; int n0, n1, n2, rows = 0;
    long picks = 0;
    static char t0[1 << 16], t1[1 << 16];

    printf(" the tool's mask (top-2 of a known history, CPU)\n");
    fx_run(tool_mask, FX "/mask_trace.txt", NULL, l2, &n2);   /* rc: the oracle's tokens may need other experts */
    int same = g_q38_mask != NULL;
    for (int l = 0; l < 4 && same; l++) for (int e = 0; e < 4; e++) same &= g_q38_mask[l * 4 + e] == fx_allow[l][e];
    tk(same, "the file holds the history's top-2 per layer (the lower id on a tie)");
    tk(fx_trace_obeys(FX "/mask_trace.txt", &rows), "ROUTE_TRACE: every row routed inside the mask, gates sum to 1");
    printf("  %d routed rows traced\n", rows);
    remove(FX "/mask_trace.txt");
    for (int batch = 0; batch < 2; batch++) {
        setenv("Q38_PREFILL_BATCH", batch ? "1" : "0", 1);
        fx_run(tool_mask, NULL, FX "/mask_hist.txt", l2, &n2);
        tk(fx_history_obeys(FX "/mask_hist.txt", &picks) && picks == 2L * rows,
           batch ? "Q38_PREFILL_BATCH=1: every pick inside the mask (the run's history)"
                 : "Q38_PREFILL_BATCH=0: every pick inside the mask (the run's history)");
        tk(g_q38_mask_outside == 0, "no route outside the mask counted");
    }
    unsetenv("Q38_PREFILL_BATCH");
    remove(FX "/mask_hist.txt");

    printf(" mask unset against a mask that allows every expert\n");
    int rc0 = fx_run(NULL, NULL, FX "/mask_hist0.txt", l0, &n0);
    tk(rc0 == 0 && n0 > 0, "unset: the oracle tokens and logits");
    tk(g_q38_mask == NULL, "unset: no mask loaded");
    tk(n2 == n0 && memcmp(l0, l2, (size_t)n0 * sizeof(float)), "the restricted run computed other logits");
    fx_write(FX "/mask_all.txt", "# every expert of every layer\n0 0 1 2 3\n1 3 2 1 0\n2 0 1 2 3\n3 0 1 2 3   # trailing comment\n\n");
    int rc1 = fx_run(FX "/mask_all.txt", NULL, FX "/mask_hist1.txt", l1, &n1);
    tk(rc1 == 0 && g_q38_mask != NULL, "allow-all mask loaded, the oracle passes");
    tk(n1 == n0 && !memcmp(l0, l1, (size_t)n0 * sizeof(float)), "allow-all: the unset run's last logits to the bit");
    long s0 = fx_slurp(FX "/mask_hist0.txt", t0, sizeof t0), s1 = fx_slurp(FX "/mask_hist1.txt", t1, sizeof t1);
    tk(s0 > 0 && s0 == s1 && !memcmp(t0, t1, (size_t)s0), "allow-all: the same picks (the run's history, byte for byte)");
    remove(FX "/mask_hist0.txt"); remove(FX "/mask_hist1.txt"); remove(FX "/mask_all.txt");

    printf(" malformed masks stop the engine\n");
    const char *bad[] = { "0 1\n", "0 1 1\n", "0 1 9\n", "7 0 1\n", "0 0 1\n0 2 3\n", "0 0 x\n", "layer 0 1\n" };
    const char *why[] = { "fewer than the top-k", "an expert twice", "an expert out of range", "a layer out of range",
                          "a layer twice", "a non-number", "a line without its layer" };
    for (int i = 0; i < 7; i++) {
        fx_write(FX "/mask_bad.txt", bad[i]);
        char what[96]; snprintf(what, sizeof what, "refused: %s", why[i]);
        tk(fx_run(FX "/mask_bad.txt", NULL, NULL, l2, &n2) != 0 && n2 == 0, what);
    }
    remove(FX "/mask_bad.txt");
    tk(fx_run(FX "/no_such_mask.txt", NULL, NULL, l2, &n2) != 0, "refused: a missing file");

    /* the FP8 fixture's tier has no warm start: it fills as the routed experts pass, so each
     * allowed expert misses once, at its first route, and nothing outside the mask ever loads
     * (a GGUF's warm start plans the allowed experts only, qt_plan_allow) */
    printf(" fake CUDA tier: only the allowed experts reach VRAM\n");
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_UPLOAD_SYNC", "1", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    setenv("Q38_TRUNK_GPU", "0", 1);   /* the fake computes no trunk matrix */
    size_t exp_bytes = 3 * dev_alloc_footprint(32 * 8) + 3 * dev_alloc_footprint(1 * sizeof(float));
    for (int room = 10; room >= 6; room -= 4) {   /* 10: all 8 allowed fit; 6: two stay out */
        char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(room * exp_bytes + exp_bytes / 2) / 1073741824.0);
        setenv("CUDA_EXPERT_GB", gb, 1);
        fake_ndev = 1;
        fx_run(tool_mask, NULL, NULL, l2, &n2);
        tk(G.on, "tier up");
        int outside = 0, inside = 0;
        for (int l = 0; l < 4; l++) for (int e = 0; e < 4; e++) {
            int r = qs(l, e)->resident;
            if (fx_allow[l][e]) inside += r; else outside += r;
        }
        unsigned long long hits = 0; for (int i = 0; i < G.ndev; i++) hits += G.hits[i];
        printf("  room %d: %d allowed experts resident, %d others; hits %llu, misses %llu\n", room, inside, outside, hits, (unsigned long long)G.miss);
        tk(outside == 0, "no expert outside the mask was loaded");
        tk(g_q38_mask_outside == 0, "no route outside the mask");
        if (room == 10) {
            tk(inside == 8, "every allowed expert resident");
            tk(hits > 0 && G.miss <= 8, "routed experts hit VRAM, each allowed one missed at most once (its first route)");
        } else {
            tk(inside == 6, "the budget's six allowed experts resident");
            tk(G.miss > 0, "the two left out miss as usual");
        }
        qt_shutdown();
    }
    /* the installed path: every layer on the CUDA chain, which routes on the card through the
     * mask's bias row (or hands the host the device's router logits) */
    printf(" fake CUDA chain (COLI_CUDA_CHAIN=1)\n");
    setenv("CUDA_EXPERT_GB", "0.5", 1);
    setenv("COLI_PLACE", "auto", 1);
    setenv("Q38_TRUNK_GPU", "1", 1);
    setenv("Q38_TRUNK_MIN_KB", "0", 1);
    setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_dense_compute = 1; fake_ndev = 1; fake_chain_frames = 0;
    fx_run(tool_mask, NULL, FX "/mask_hist.txt", l2, &n2);
    unsigned long long chits = 0; for (int i = 0; i < G.ndev; i++) chits += G.hits[i];
    printf("  chain: hits %llu, misses %llu\n", chits, (unsigned long long)G.miss);
    tk(fake_chain_frames > 8, "the forwards ran through the chain's frames");
    tk(fx_history_obeys(FX "/mask_hist.txt", &picks), "every pick inside the mask (the run's history)");
    tk(g_q38_mask_outside == 0 && chits > 0 && G.miss <= 8, "no route outside the mask, routed experts hit VRAM, each allowed one missed at most once");
    qt_shutdown(); remove(FX "/mask_hist.txt");
    fake_chain_frames = 0;
    fx_run(NULL, NULL, NULL, l0, &n0); qt_shutdown();
    fx_write(FX "/mask_all.txt", "0 0 1 2 3\n1 0 1 2 3\n2 0 1 2 3\n3 0 1 2 3\n");
    fx_run(FX "/mask_all.txt", NULL, NULL, l1, &n1); qt_shutdown();
    tk(fake_chain_frames > 16 && n0 > 0 && n1 == n0 && !memcmp(l0, l1, (size_t)n0 * sizeof(float)),
       "chain: an allow-all mask gives the unset run's last logits to the bit");
    unsetenv("COLI_CUDA_CHAIN"); unsetenv("CUDA_EXPERT_GB");

    printf(" the routing history under a mask\n");
    setenv("Q38_EXPERT_MASK", FX "/mask_all.txt", 1);
    unsetenv("USAGE_SAVE");
    Cfg fc = {0}; fc.layers = 4; fc.experts = 4; fc.topk = 2;
    int loaded = q38_mask_load(&fc);
    const char *sv = getenv("USAGE_SAVE");
    tk(loaded && sv && !strcmp(sv, "0"), "USAGE_SAVE unset: a masked run does not save the history");
    setenv("USAGE_SAVE", "1", 1);
    loaded = q38_mask_load(&fc); sv = getenv("USAGE_SAVE");
    tk(loaded && sv && !strcmp(sv, "1"), "USAGE_SAVE=1 kept");
    unsetenv("Q38_EXPERT_MASK"); remove(FX "/mask_all.txt");
    q38_mask_load(&fc);
    setenv("USAGE_SAVE", "0", 1);

    if (t_fails) { printf("test_qwen38_expert_mask: %d failure(s)\n", t_fails); return 1; }
    printf("test_qwen38_expert_mask: ok\n");
    return 0;
}
