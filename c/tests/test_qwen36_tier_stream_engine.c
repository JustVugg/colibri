/* The qwen36 engine on the fake CUDA tier, driven through its own main(), on an
 * int4 gs64 container (#2090).
 *
 *   test_qwen36_tier_stream_engine CAP SNAP CPU_LOGITS REF_JSON
 *
 * What the tier's unit test (test_qwen36_tier_stream.c) cannot see is the
 * engine side: that the shared expert kernel (expert_ffn.h) now stays on under
 * COLI_CUDA=1, that the tier starts at a cap below n_experts in copy mode and
 * its warmstart reads the planned experts into a slot of its own, that the
 * decode loop issues first and hands the tier the planar bytes of the experts
 * the CPU holds, and that none of it moves the arithmetic. The fake backend
 * refuses every issue, so the CPU computes every expert, and the last logits
 * must equal CPU_LOGITS, which the CPU build wrote at the same cap with the
 * same environment (the int4 fixture does not match its f32 oracle by design,
 * so REF_JSON only supplies the prompt). The hit counter pins that residency
 * was consulted all the same. Real-GPU parity is a hardware run.
 *
 * Include order as in test_qwen36_tier_int8_engine.c: engine first, then the
 * fake backend, then the tier source, so the tier's statics are readable. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void tk(int ok, const char *what) {
    if (ok) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

static float *read_f32(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    float *v = bytes > 0 ? malloc((size_t)bytes) : NULL;
    *n = v && fread(v, 1, (size_t)bytes, f) == (size_t)bytes ? (size_t)bytes / sizeof(float) : 0;
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s CAP SNAP CPU_LOGITS REF_JSON\n", argv[0]); return 2; }
    int cap = atoi(argv[1]);
    char dump[64]; snprintf(dump, sizeof dump, "qwen36_tier_stream_%d.f32", cap);
    setenv("SNAP", argv[2], 1);
    setenv("DUMP", dump, 1);
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("NOSTREAM", "1", 1);
    setenv("COLI_DENSE_I8", "0", 1);            /* as the CPU reference run */
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("COLI_PLACE", "off", 1);             /* the fake backend computes no trunk */
    setenv("HEAT_FILE", "", 1);
    /* every promotion resident before the next group: deterministic residency
     * on a two-vCPU runner (test_qwen38_tier_engine.c has the measurement) */
    setenv("QT_UPLOAD_SYNC", "1", 1);
    /* the tiny geometry: hidden 64, inter 64, gs 64 -> 2 KiB per matrix, 64 or
     * 64 scales; room for six experts of the 64, so the warmstart plans six and
     * promotion has to swap */
    size_t exp_bytes = 3 * dev_alloc_footprint(64 * 64 / 2) + 3 * dev_alloc_footprint(64 * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(6 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    fake_ndev = 1; fake_uploads = 0;

    printf(" cap %d\n", cap);
    char *args[] = { (char *)"qwen36", argv[1], (char *)"4", argv[4] };
    int rc = qwen36_main_unused(4, args);
    printf("  (engine exit %d against the f32 oracle: int4 drifts from it by design)\n", rc);
    qt_fill_wait();

    tk(xf_mode(NULL) == 1, "the shared expert kernel stays on under COLI_CUDA=1");
    tk(G.on && G.wfmt == 4 && G_src_planar, "the tier started on int4, from the planar layout");
    tk(G.exp_bytes == exp_bytes, "bytes per expert at cudaMalloc granularity");
    tk(qt_streaming() == (cap < G.ne), cap < G.ne ? "copy mode below full residency" : "pointer mode at full residency");
    tk(fake_uploads > 0 && fake_uploads % 3 == 0 && last_fmt == 4, "uploads in gate/up/down triples, fmt 4");
    pthread_mutex_lock(&G.mx);
    unsigned long long hits = 0; for (int i = 0; i < G.ndev; i++) hits += G.hits[i];
    int residents = 0, stale = 0;
    for (int l = 0; l < G.nl; l++) for (int e = 0; e < G.ne; e++) {
        residents += qs(l, e)->resident;
        stale += qs(l, e)->g4 != NULL;
    }
    size_t used = G.used[0];
    pthread_mutex_unlock(&G.mx);
    tk(hits > 0, "resident experts were routed (hits counted though the fake backend refuses to issue)");
    tk(residents > 0 && residents <= 6 && used == (size_t)residents * G.exp_bytes, "residency within the six-expert budget, bytes exact");
    if (cap < G.ne) tk(stale == 0, "no slot keeps a pointer into the engine's recycled RAM slots");

    size_t na = 0, nb = 0;
    float *a = read_f32(dump, &na), *b = read_f32(argv[3], &nb);
    tk(a && b && na == nb && na > 0, "both runs dumped their last logits");
    if (a && b && na == nb && na > 0) {
        double top = 0, diff = 0;
        for (size_t i = 0; i < na; i++) {
            double d = fabs((double)a[i] - b[i]); if (d > diff) diff = d;
            if (fabs(b[i]) > top) top = fabs(b[i]);
        }
        printf("  last logits: max |tier - cpu| %.3g of max |cpu| %.3g\n", diff, top);
        tk(diff <= 1e-4 * (top > 1 ? top : 1), "the last logits equal the CPU build's (same tokens all the way)");
    }
    free(a); free(b);
    remove(dump);

    if (fails) { printf("test_qwen36_tier_stream_engine: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_stream_engine: ok\n");
    return 0;
}
