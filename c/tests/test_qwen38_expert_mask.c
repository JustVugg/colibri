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
 *  - the soft variants (Q38_EXPERT_MASK_FROM_RANK / _TAU): q38_mask_pick's picks and
 *    counters on a made-up row; on the engine, FROM_RANK=top-k and a tiny TAU give the
 *    unset run's bits, TAU near 1 keeps the history inside the mask (CPU and chain), bad
 *    values stop it;
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
    char *argv[] = { (char *)"qwen38", (char *)"4", (char *)"8", (char *)FX "/ref.json", NULL };
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
    /* the engine's main would re-exec this process once for the OpenMP hot team (omp_tune.h,
     * Linux, CUDA off): the test's own argv would come back as the engine's */
    setenv("COLI_NO_OMP_TUNE", "1", 1);
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
        tk(g_q38_mask_routed == 2ULL * rows && g_q38_mask_replaced == 0, "every pick counted as routed, none replaced");
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

    printf(" soft variants: q38_mask_pick on a made-up router row (6 experts, top-3, 2 3 4 allowed)\n");
    {
        static const uint8_t al[6] = {0, 0, 1, 1, 1, 0};
        static const float p[6] = {.30f, .25f, .05f, .20f, .10f, .10f};   /* sums to 1: `all` */
        struct { int from; double tau; int norm; int want[3]; unsigned long long out, rep; const char *what; } cs[] = {
            {0, 0, 1, {3, 4, 2}, 0, 0, "hard: the allowed top-3"},
            {1, 0, 1, {0, 3, 4}, 1, 0, "FROM_RANK=1: rank 0 over every expert, then the best allowed"},
            {2, 0, 1, {0, 1, 3}, 2, 0, "FROM_RANK=2"},
            {3, 0, 1, {0, 1, 3}, 2, 0, "FROM_RANK=3 (the top-k): the unmasked top-3"},
            {3, .35, 1, {0, 4, 3}, 1, 1, "TAU=0.35, norm_topk: gate .333 gives its slot to 4, gate .40 stays"},
            {3, .28, 0, {0, 4, 3}, 1, 1, "TAU=0.28, softmax gates: .25 replaced, .30 stays"},
            {3, .05, 1, {0, 1, 3}, 2, 0, "TAU=0.05: every outside gate above it, none replaced"},
            {1, .9, 1, {2, 3, 4}, 0, 1, "FROM_RANK=1 and TAU=0.9: rank 0's gate .50 replaced by 2"},
        };
        for (size_t i = 0; i < sizeof cs / sizeof *cs; i++) {
            int idx[3]; float val[3];
            g_q38_mask_from = cs[i].from; g_q38_mask_tau = cs[i].tau;
            g_q38_mask_routed = g_q38_mask_outside = g_q38_mask_replaced = 0;
            q38_mask_pick(al, p, 6, 3, 0, cs[i].norm, 1.0, idx, val);
            int ok = g_q38_mask_routed == 3 && g_q38_mask_outside == cs[i].out && g_q38_mask_replaced == cs[i].rep;
            for (int z = 0; z < 3; z++) ok &= idx[z] == cs[i].want[z] && val[z] == p[idx[z]];
            tk(ok, cs[i].what);
        }
        int idx[3]; float val[3];
        g_q38_mask_routed = 0;
        q38_mask_pick(NULL, p, 6, 3, 0, 1, 1.0, idx, val);
        tk(idx[0] == 0 && idx[1] == 1 && idx[2] == 3 && g_q38_mask_routed == 0, "an unmasked layer: the plain top-3, not counted");
        g_q38_mask_from = 0; g_q38_mask_tau = 0;
    }

    printf(" soft variants on the engine (tool mask, CPU)\n");
    struct { const char *rank, *tau; int same; const char *what; } sa[] = {
        {"2", NULL, 1, "FROM_RANK=2 (the top-k): the unset run's last logits to the bit"},
        {NULL, "0.000000001", 1, "TAU=1e-9: nothing replaced, the unset run's last logits to the bit"},
        {NULL, "0.999999", 0, "TAU=0.999999: every outside pick replaced, the history inside the mask"},
        {"1", NULL, 0, "FROM_RANK=1: at most rank 0 outside the mask, nothing replaced"},
    };
    for (int i = 0; i < 4; i++) {
        if (sa[i].rank) setenv("Q38_EXPERT_MASK_FROM_RANK", sa[i].rank, 1);
        if (sa[i].tau) setenv("Q38_EXPERT_MASK_TAU", sa[i].tau, 1);
        fx_run(tool_mask, NULL, sa[i].same ? NULL : FX "/mask_hist.txt", l1, &n1);
        unsetenv("Q38_EXPERT_MASK_FROM_RANK"); unsetenv("Q38_EXPERT_MASK_TAU");
        printf("  routed %llu, outside %llu, replaced %llu\n", g_q38_mask_routed, g_q38_mask_outside, g_q38_mask_replaced);
        if (sa[i].same)
            tk(n1 == n0 && !memcmp(l0, l1, (size_t)n0 * sizeof(float)) && g_q38_mask_routed > 0 && g_q38_mask_replaced == 0, sa[i].what);
        else if (i == 2)
            tk(g_q38_mask_replaced > 0 && g_q38_mask_outside == 0 && fx_history_obeys(FX "/mask_hist.txt", &picks)
               && picks == (long)g_q38_mask_routed, sa[i].what);
        else
            tk(g_q38_mask_replaced == 0 && g_q38_mask_routed > 0 && 2 * g_q38_mask_outside <= g_q38_mask_routed, sa[i].what);
        remove(FX "/mask_hist.txt");
    }
    setenv("Q38_EXPERT_MASK_FROM_RANK", "1", 1);
    fx_run(NULL, NULL, NULL, l1, &n1);
    unsetenv("Q38_EXPERT_MASK_FROM_RANK");
    tk(g_q38_mask == NULL && n1 == n0 && !memcmp(l0, l1, (size_t)n0 * sizeof(float)), "FROM_RANK without a mask: ignored, the unset run's bits");
    const char *bad_env[][2] = {{"Q38_EXPERT_MASK_FROM_RANK", "0"}, {"Q38_EXPERT_MASK_FROM_RANK", "3"},
                                {"Q38_EXPERT_MASK_FROM_RANK", "1x"}, {"Q38_EXPERT_MASK_TAU", "0"},
                                {"Q38_EXPERT_MASK_TAU", "1"}, {"Q38_EXPERT_MASK_TAU", "abc"}};
    for (int i = 0; i < 6; i++) {
        setenv(bad_env[i][0], bad_env[i][1], 1);
        char what[96]; snprintf(what, sizeof what, "refused: %s=%s", bad_env[i][0], bad_env[i][1]);
        tk(fx_run(tool_mask, NULL, NULL, l2, &n2) != 0 && n2 == 0, what);
        unsetenv(bad_env[i][0]);
    }

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
    /* Q38_ROUTE_TAU=t (NOT exact, default off), on the same tier with room for 6 of the 16 experts
     * and no mask: unset and 0 are the same run to the bit; a high t (top-2 gates sum to 1, so
     * nearly every non-resident pick qualifies) replaces picks by residents and leaves the tier
     * fewer misses; with a mask, or outside [0, 1), the engine stops */
    printf(" Q38_ROUTE_TAU on the fake tier (room for 6 of 16 experts)\n");
    {
        static const char *val[3] = { NULL, "0", "0.999" };
        float lt[3][256]; int nt[3], rct[3]; unsigned long long rep[3], routed[3], kept[3], miss[3];
        for (int a = 0; a < 3; a++) {
            if (val[a]) setenv("Q38_ROUTE_TAU", val[a], 1); else unsetenv("Q38_ROUTE_TAU");
            fake_ndev = 1;
            rct[a] = fx_run(NULL, NULL, NULL, lt[a], &nt[a]);
            rep[a] = g_q38_mask_replaced; routed[a] = g_q38_mask_routed; kept[a] = g_q38_mask_outside; miss[a] = G.miss;
            tk(a == 2 ? g_q38_route_tau > 0 && !g_q38_mask : g_q38_route_tau == 0, a == 2 ? "0.999: on, no mask" : "unset / 0: off");
            qt_shutdown();
        }
        unsetenv("Q38_ROUTE_TAU");
        printf("  t 0.999: %llu picks routed, %llu replaced, %llu kept though not in VRAM; tier misses %llu (off %llu)\n",
               routed[2], rep[2], kept[2], miss[2], miss[0]);
        tk(nt[0] > 0 && nt[0] == nt[1] && rct[0] == rct[1] && !memcmp(lt[0], lt[1], (size_t)nt[0] * sizeof(float)) && rep[0] == 0 && rep[1] == 0,
           "Q38_ROUTE_TAU=0 is the unset run to the bit, nothing replaced");
        tk(rep[2] > 0 && routed[2] > 0 && miss[2] < miss[0] && miss[2] <= kept[2],
           "on: picks were replaced by residents; the tier misses fewer, at most the picks it kept");
        setenv("Q38_ROUTE_TAU", "1.5", 1);
        tk(fx_run(NULL, NULL, NULL, lt[0], &nt[0]) != 0 && nt[0] == 0, "Q38_ROUTE_TAU=1.5: refused, the engine stops");
        setenv("Q38_ROUTE_TAU", "0.5", 1);
        tk(fx_run(tool_mask, NULL, NULL, lt[0], &nt[0]) != 0 && nt[0] == 0, "Q38_ROUTE_TAU with Q38_EXPERT_MASK: refused");
        unsetenv("Q38_ROUTE_TAU"); qt_shutdown();
    }
    /* Q38_ROUTE_BIAS on the host's pick (q38_bias_pick): b added to the resident experts' terms
     * for the pick only, the values the model's own terms */
    printf(" Q38_ROUTE_BIAS: the host's pick\n");
    {
        /* terms of a softmax (all = 1): 0 .30, 1 .25, 2 .18, 3 .15, 4 .08, 5 .04 */
        const float p[6] = {.30f, .25f, .18f, .15f, .08f, .04f};
        const uint8_t on[6] = {0, 0, 1, 1, 1, 0}, all1[6] = {1, 1, 1, 1, 1, 1};
        struct { double b; const uint8_t *res; int want[3]; unsigned long long moved; const char *what; } cs[] = {
            {0.0, on, {0, 1, 2}, 0, "b = 0: the plain top-3"},
            {.05, on, {0, 1, 2}, 0, "b = .05: 2 (.23) stays below 1 (.25), 3 (.20) below 2: the plain top-3"},
            {.08, on, {0, 2, 1}, 0, "b = .08: 2 (.26) passes 1 (.25), 3 (.23) does not: the same three, 2 at rank 1"},
            {.11, on, {0, 2, 3}, 1, "b = .11: 2 (.29) and 3 (.26) pass 1 (.25): 1 gives its slot to 3"},
            {.50, on, {2, 3, 4}, 2, "b = .5: the three residents first, 0 and 1 dropped"},
            {.50, all1, {0, 1, 2}, 0, "every expert resident: the plain top-3 (a uniform shift)"},
        };
        for (size_t i = 0; i < sizeof cs / sizeof *cs; i++) {
            int idx[3]; float val[3];
            g_q38_route_bias = cs[i].b; g_q38_rb_routed = g_q38_rb_moved = 0;
            q38_bias_pick(cs[i].res, p, 6, 3, 0, 1.0, idx, val);
            int ok = g_q38_rb_routed == 3 && g_q38_rb_moved == cs[i].moved;
            for (int z = 0; z < 3; z++) ok &= idx[z] == cs[i].want[z] && val[z] == p[idx[z]];
            if (!ok) printf("    got %d %d %d, moved %llu\n", idx[0], idx[1], idx[2], g_q38_rb_moved);
            tk(ok, cs[i].what);
        }
        /* random rows: b = 0 is the plain pick bit for bit; with any b every value is the model's own term */
        unsigned s = 20261124u; long bad0 = 0, badv = 0; const int T = 50000;
#define FX_RND() (s = s * 1664525u + 1013904223u, s >> 8)
        for (int t = 0; t < T; t++) {
            float q[64]; uint8_t res[64]; int ia[Q38_MAX_TOPK], ib[Q38_MAX_TOPK]; float va[Q38_MAX_TOPK], vb[Q38_MAX_TOPK];
            int E = 2 + (int)(FX_RND() % 62), K = 1 + (int)(FX_RND() % (E < 10 ? E : 10));
            double tot = 0;
            for (int e = 0; e < E; e++) { q[e] = (float)(FX_RND() % 100000) / 100000.f; tot += q[e]; res[e] = FX_RND() % 2; }
            g_q38_route_bias = 0; q38_bias_pick(res, q, E, K, 0, tot, ia, va);
            q38_mask_pick(NULL, q, E, K, 0, 1, tot, ib, vb);
            bad0 += memcmp(ia, ib, (size_t)K * sizeof(int)) != 0;
            g_q38_route_bias = (double)(FX_RND() % 100) / 1000.0;
            q38_bias_pick(res, q, E, K, 0, tot, ia, va);
            for (int z = 0; z < K; z++) badv += va[z] != q[ia[z]];
        }
#undef FX_RND
        g_q38_route_bias = 0; g_q38_rb_routed = g_q38_rb_moved = 0;
        tk(bad0 == 0, "random rows, b = 0: the plain top-k on every row");
        tk(badv == 0, "random rows, any b: every value the model's own term");
    }
    printf(" Q38_ROUTE_BIAS on the engine (fake tier, room for 6 of 16 experts)\n");
    {
        static const char *val[3] = { NULL, "0", "0.5" };
        float lb[3][256]; int nb[3], rcb[3]; unsigned long long moved[3], miss[3];
        for (int a = 0; a < 3; a++) {
            if (val[a]) setenv("Q38_ROUTE_BIAS", val[a], 1); else unsetenv("Q38_ROUTE_BIAS");
            fake_ndev = 1;
            rcb[a] = fx_run(NULL, NULL, NULL, lb[a], &nb[a]);
            moved[a] = g_q38_rb_moved; miss[a] = G.miss;
            qt_shutdown();
        }
        unsetenv("Q38_ROUTE_BIAS");
        printf("  b 0.5: %llu of %llu decode picks moved onto the card; tier misses %llu (off %llu)\n",
               moved[2], g_q38_rb_routed, miss[2], miss[0]);
        tk(nb[0] > 0 && nb[0] == nb[1] && rcb[0] == rcb[1] && !memcmp(lb[0], lb[1], (size_t)nb[0] * sizeof(float)) && moved[1] == 0,
           "Q38_ROUTE_BIAS=0 is the unset run to the bit");
        tk(moved[2] > 0 && miss[2] <= miss[0], "b = 0.5: decode picks moved onto the card, no more tier misses than off");
        const char *bad_rb[][2] = {{"Q38_ROUTE_BIAS", "1.5"}, {"Q38_ROUTE_BIAS_FILE", "-0.1"}, {"Q38_ROUTE_BIAS", "x"}};
        for (int i = 0; i < 3; i++) {
            setenv(bad_rb[i][0], bad_rb[i][1], 1);
            char what[96]; snprintf(what, sizeof what, "refused: %s=%s", bad_rb[i][0], bad_rb[i][1]);
            tk(fx_run(NULL, NULL, NULL, lb[0], &nb[0]) != 0 && nb[0] == 0, what);
            unsetenv(bad_rb[i][0]);
        }
        setenv("Q38_ROUTE_BIAS", "0.05", 1);
        tk(fx_run(tool_mask, NULL, NULL, lb[0], &nb[0]) != 0 && nb[0] == 0, "Q38_ROUTE_BIAS with Q38_EXPERT_MASK: refused");
        setenv("Q38_ROUTE_TAU", "0.5", 1);
        tk(fx_run(NULL, NULL, NULL, lb[0], &nb[0]) != 0 && nb[0] == 0, "Q38_ROUTE_BIAS with Q38_ROUTE_TAU: refused");
        unsetenv("Q38_ROUTE_TAU"); unsetenv("Q38_ROUTE_BIAS"); qt_shutdown();
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
    /* a soft variant: the chain's device route knows the hard mask only, so the host routes */
    setenv("Q38_EXPERT_MASK_TAU", "0.999999", 1);
    fake_chain_frames = 0;
    fx_run(tool_mask, NULL, FX "/mask_hist.txt", l2, &n2);
    unsetenv("Q38_EXPERT_MASK_TAU");
    tk(fake_chain_frames > 8 && g_q38_mask_replaced > 0 && g_q38_mask_outside == 0 && fx_history_obeys(FX "/mask_hist.txt", &picks),
       "chain, TAU=0.999999: every outside pick replaced, the history inside the mask");
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
