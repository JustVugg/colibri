/* CPU gate for the qwen36 qpack MLX-affine routed-expert path.
 *
 * Proves the engine-side glue with no GPU in the room:
 *   - qq_open refuses geometry that does not match the loaded model (layer
 *     count, expert count, hidden, inter) and accepts the matching container;
 *   - qq_expert_forward equals an independent composition of the affine CPU
 *     reference (gate/up/down + SwiGLU + weighted accumulate) per expert;
 *   - moe() with the store active produces exactly the router-weighted sum of
 *     reference expert outputs (shared expert zeroed), i.e. the wiring feeds
 *     the right activations to the right experts with the right weights.
 *
 * The container is synthesized here the same way tests/test_qpack.c does it;
 * the Metal-vs-CPU parity gates on the REAL Swiftlet fixture live in
 * tests/test_qwen36_qpack_metal.c and run on Apple hardware. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../qpack.h"

#include <sys/stat.h>

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                            \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

/* Fixture geometry: Q4, per_word=8, group_size=8.
 * gate/up: [INTER, HIDDEN]; down: [HIDDEN, INTER]. */
enum { FX_LAYERS = 2, FX_EXPERTS = 2, FX_HIDDEN = 16, FX_INTER = 8,
       FX_GS = 8, FX_STRIDE = 16384 };

static const char fx_layout[] =
    "{\"expertCount\":2,\"layerCount\":2,\"expertStride\":16384,"
    "\"linearLayers\":[true,false],\"sections\":["
    "{\"name\":\"gate_proj.weight\",\"dtype\":\"U32\","
    "\"shape\":[8,2],\"offset\":0,\"size\":64},"
    "{\"name\":\"gate_proj.scales\",\"dtype\":\"F32\","
    "\"shape\":[8,2],\"offset\":64,\"size\":64},"
    "{\"name\":\"gate_proj.biases\",\"dtype\":\"F32\","
    "\"shape\":[8,2],\"offset\":128,\"size\":64},"
    "{\"name\":\"up_proj.weight\",\"dtype\":\"U32\","
    "\"shape\":[8,2],\"offset\":192,\"size\":64},"
    "{\"name\":\"up_proj.scales\",\"dtype\":\"F32\","
    "\"shape\":[8,2],\"offset\":256,\"size\":64},"
    "{\"name\":\"up_proj.biases\",\"dtype\":\"F32\","
    "\"shape\":[8,2],\"offset\":320,\"size\":64},"
    "{\"name\":\"down_proj.weight\",\"dtype\":\"U32\","
    "\"shape\":[16,1],\"offset\":384,\"size\":64},"
    "{\"name\":\"down_proj.scales\",\"dtype\":\"F32\","
    "\"shape\":[16,1],\"offset\":448,\"size\":64},"
    "{\"name\":\"down_proj.biases\",\"dtype\":\"F32\","
    "\"shape\":[16,1],\"offset\":512,\"size\":64}]}";

static void fx_path(char *out, size_t cap, const char *root, const char *name) {
    int n = snprintf(out, cap, "%s/%s", root, name);
    CHECK(n > 0 && (size_t)n < cap);
}

static void fx_mkdir(const char *path) {
#ifdef _WIN32
    CHECK(_mkdir(path) == 0);
#else
    CHECK(mkdir(path, 0700) == 0);
#endif
}

static void fx_write(const char *path, const void *data, size_t size) {
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(data, 1, size, f) == size);
    CHECK(fclose(f) == 0);
}

static void fx_store_f32(unsigned char *at, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    at[0] = (unsigned char)bits;
    at[1] = (unsigned char)(bits >> 8);
    at[2] = (unsigned char)(bits >> 16);
    at[3] = (unsigned char)(bits >> 24);
}

/* One expert blob: deterministic per (layer, expert) so every expert is a
 * different matrix and a routing/index mix-up shows up as a numeric miss. */
static void fx_fill_expert(unsigned char *blob, size_t layer, size_t expert) {
    uint32_t state = 0x9e3779b9u ^ (uint32_t)(layer * 131u + expert * 17u + 5u);
    /* weights: gate@0 up@192 down@384, 16 u32 words each */
    static const size_t woff[3] = { 0, 192, 384 };
    for (int p = 0; p < 3; p++)
        for (int w = 0; w < 16; w++) {
            state = state * 1664525u + 1013904223u;
            blob[woff[p] + (size_t)w * 4 + 0] = (unsigned char)state;
            blob[woff[p] + (size_t)w * 4 + 1] = (unsigned char)(state >> 8);
            blob[woff[p] + (size_t)w * 4 + 2] = (unsigned char)(state >> 16);
            blob[woff[p] + (size_t)w * 4 + 3] = (unsigned char)(state >> 24);
        }
    /* scales/biases: gate@64/128 up@256/320 down@448/512, 16 f32 each */
    static const size_t soff[3] = { 64, 256, 448 };
    for (int p = 0; p < 3; p++)
        for (int i = 0; i < 16; i++) {
            float scale = 0.03125f * (float)((int)((layer + expert + i + p) % 7) - 3);
            float bias  = 0.015625f * (float)((int)((layer * 3 + expert + i) % 5) - 2);
            if (scale == 0.0f) scale = 0.046875f;
            fx_store_f32(blob + soff[p] + (size_t)i * 4, scale);
            fx_store_f32(blob + soff[p] + 64 + (size_t)i * 4, bias);
        }
}

static void fx_build(const char *root) {
    char path[1024];
    fx_mkdir(root);
    fx_path(path, sizeof(path), root, "packed_experts");
    fx_mkdir(path);
    fx_path(path, sizeof(path), root, "packed_experts/layout.json");
    fx_write(path, fx_layout, sizeof(fx_layout) - 1);
    for (size_t layer = 0; layer < FX_LAYERS; layer++) {
        unsigned char *bytes = calloc((size_t)FX_EXPERTS * FX_STRIDE, 1);
        CHECK(bytes != NULL);
        for (size_t expert = 0; expert < FX_EXPERTS; expert++)
            fx_fill_expert(bytes + expert * FX_STRIDE, layer, expert);
        char name[64];
        int n = snprintf(name, sizeof(name),
                         "packed_experts/layer_%02zu.bin", layer);
        CHECK(n > 0 && (size_t)n < sizeof(name));
        fx_path(path, sizeof(path), root, name);
        fx_write(path, bytes, (size_t)FX_EXPERTS * FX_STRIDE);
        free(bytes);
    }
    char manifest[1024];
    int n = snprintf(
        manifest, sizeof(manifest),
        "{\"magic\":\"QPACK\",\"version\":1,\"modelName\":\"fixture\","
        "\"sourceCheckpoint\":\"fixture\",\"quantBits\":4,"
        "\"quantGroupSize\":%d,\"files\":{"
        "\"packed_experts/layout.json\":%zu,"
        "\"packed_experts/layer_00.bin\":%d,"
        "\"packed_experts/layer_01.bin\":%d}}",
        FX_GS, sizeof(fx_layout) - 1,
        FX_EXPERTS * FX_STRIDE, FX_EXPERTS * FX_STRIDE);
    CHECK(n > 0 && (size_t)n < sizeof(manifest));
    fx_path(path, sizeof(path), root, "manifest.json");
    fx_write(path, manifest, (size_t)n);
}

static void fx_cleanup(const char *root) {
    char path[1024];
    for (size_t layer = 0; layer < FX_LAYERS; layer++) {
        char name[64];
        snprintf(name, sizeof(name), "packed_experts/layer_%02zu.bin", layer);
        fx_path(path, sizeof(path), root, name);
        remove(path);
    }
    fx_path(path, sizeof(path), root, "packed_experts/layout.json");
    remove(path);
    fx_path(path, sizeof(path), root, "manifest.json");
    remove(path);
    fx_path(path, sizeof(path), root, "packed_experts");
#ifdef _WIN32
    _rmdir(path); _rmdir(root);
#else
    rmdir(path); rmdir(root);
#endif
}

/* Independent oracle: read the blob with the raw qpack reader and compose the
 * affine CPU reference by hand.  out[FX_HIDDEN] += weight * expert(x). */
static void oracle_expert(const ColiQpackReader *reader, int layer, int eid,
                          const float *x, float weight, float *out) {
    unsigned char blob[FX_STRIDE];
    char err[256];
    CHECK(coli_qpack_read_expert(reader, (size_t)layer, (size_t)eid,
                                 blob, sizeof(blob), err, sizeof(err)) == 0);
    ColiAffineQuantizedView vg, vu, vd;
    CHECK(coli_qpack_affine_view(reader, blob, sizeof(blob), "gate_proj",
                                 &vg, err, sizeof(err)) == 0);
    CHECK(coli_qpack_affine_view(reader, blob, sizeof(blob), "up_proj",
                                 &vu, err, sizeof(err)) == 0);
    CHECK(coli_qpack_affine_view(reader, blob, sizeof(blob), "down_proj",
                                 &vd, err, sizeof(err)) == 0);
    float g[FX_INTER], u[FX_INTER], h[FX_HIDDEN];
    CHECK(coli_affine_matmul_ref(g, x, 1, &vg) == COLI_AFFINE_OK);
    CHECK(coli_affine_matmul_ref(u, x, 1, &vu) == COLI_AFFINE_OK);
    for (int i = 0; i < FX_INTER; i++)
        g[i] = (g[i] / (1.f + expf(-g[i]))) * u[i];
    CHECK(coli_affine_matmul_ref(h, g, 1, &vd) == COLI_AFFINE_OK);
    for (int d = 0; d < FX_HIDDEN; d++) out[d] += weight * h[d];
}

static double rel_err(const float *got, const float *want, size_t n) {
    double maxabs = 0.0, ymax = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)got[i]) || !isfinite((double)want[i]))
            return (double)INFINITY;
        double d = fabs((double)got[i] - (double)want[i]);
        if (d > maxabs) maxabs = d;
        double m = fabs((double)want[i]);
        if (m > ymax) ymax = m;
    }
    return maxabs / (ymax + 1e-9);
}

static int fails = 0;
static void ck(int cond, const char *what) {
    if (cond) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

static void case_open_validation(const char *root) {
    char err[256];
    printf("qq_open geometry validation\n");
    err[0] = 0;
    ck(!qq_open(root, FX_LAYERS + 1, FX_EXPERTS, FX_HIDDEN, FX_INTER,
                err, sizeof(err)) && err[0], "wrong layer count refused");
    err[0] = 0;
    ck(!qq_open(root, FX_LAYERS, FX_EXPERTS + 2, FX_HIDDEN, FX_INTER,
                err, sizeof(err)) && err[0], "wrong expert count refused");
    err[0] = 0;
    ck(!qq_open(root, FX_LAYERS, FX_EXPERTS, FX_HIDDEN * 2, FX_INTER,
                err, sizeof(err)) && err[0], "wrong hidden refused");
    err[0] = 0;
    ck(!qq_open(root, FX_LAYERS, FX_EXPERTS, FX_HIDDEN, FX_INTER + 8,
                err, sizeof(err)) && err[0], "wrong inter refused");
    ck(!qq_active(), "store stays inactive after refusals");
    err[0] = 0;
    ck(qq_open(root, FX_LAYERS, FX_EXPERTS, FX_HIDDEN, FX_INTER,
               err, sizeof(err)), "matching geometry accepted");
    ck(qq_active(), "store active after open");
}

static void case_expert_forward(const char *root) {
    printf("qq_expert_forward vs independent affine reference\n");
    ColiQpackReader reader;
    char err[256];
    CHECK(coli_qpack_open(&reader, root, err, sizeof(err)) == 0);

    float x[FX_HIDDEN];
    for (int i = 0; i < FX_HIDDEN; i++)
        x[i] = 0.0625f * (float)((i * 7 + 3) % 13 - 6);

    uint64_t metal0, cpu0;
    qq_counts(&metal0, &cpu0);
    int all_ok = 1;
    double worst = 0.0;
    for (int layer = 0; layer < FX_LAYERS; layer++)
        for (int eid = 0; eid < FX_EXPERTS; eid++) {
            float got[FX_HIDDEN], want[FX_HIDDEN];
            for (int d = 0; d < FX_HIDDEN; d++) { got[d] = 0.25f; want[d] = 0.25f; }
            float w = 0.5f + 0.125f * (float)(layer * FX_EXPERTS + eid);
            if (!qq_expert_forward(layer, eid, x, w, got)) { all_ok = 0; continue; }
            oracle_expert(&reader, layer, eid, x, w, want);
            double e = rel_err(got, want, FX_HIDDEN);
            if (e > worst) worst = e;
        }
    uint64_t metal1, cpu1;
    qq_counts(&metal1, &cpu1);
    printf("  expert forward worst nerr=%.2e\n", worst);
    ck(all_ok, "every expert forward succeeded");
    ck(worst < 1e-6, "expert forward matches the affine reference");
    ck(cpu1 - cpu0 == (uint64_t)FX_LAYERS * FX_EXPERTS * 3,
       "three CPU-reference projections per expert");
    ck(metal1 == metal0, "no Metal dispatch on the CPU build");
    /* A real out buffer, so these exercise the range check itself and not
     * the null-argument refusal in front of it. */
    float sink[FX_HIDDEN];
    memset(sink, 0, sizeof(sink));
    ck(!qq_expert_forward(FX_LAYERS, 0, x, 1.f, sink),
       "out-of-range layer refused");
    ck(!qq_expert_forward(0, FX_EXPERTS, x, 1.f, sink),
       "out-of-range expert refused");
    coli_qpack_close(&reader);
}

static void case_moe_wiring(const char *root) {
    printf("moe() one-layer wiring through the qpack store\n");
    ColiQpackReader reader;
    char err[256];
    CHECK(coli_qpack_open(&reader, root, err, sizeof(err)) == 0);

    enum { S = 2, ISH = 4 };
    Model m;
    memset(&m, 0, sizeof(m));
    m.c.hidden = FX_HIDDEN; m.c.inter = FX_INTER;
    m.c.n_experts = FX_EXPERTS; m.c.topk = FX_EXPERTS;   /* both experts run */
    m.c.n_group = 1; m.c.shared_inter = ISH; m.c.eps = 1e-6f;
    Layer l;
    memset(&l, 0, sizeof(l));
    float gate[FX_EXPERTS * FX_HIDDEN];
    for (int i = 0; i < FX_EXPERTS * FX_HIDDEN; i++)
        gate[i] = 0.05f * (float)((i * 5 + 1) % 9 - 4);
    float sh_zero[ISH * FX_HIDDEN];
    memset(sh_zero, 0, sizeof(sh_zero));
    l.gate = gate;
    l.sh_g = sh_zero; l.sh_u = sh_zero; l.sh_d = sh_zero;   /* shared adds 0 */

    float x[S * FX_HIDDEN];
    for (int i = 0; i < S * FX_HIDDEN; i++)
        x[i] = 0.09375f * (float)((i * 11 + 5) % 15 - 7);

    float out[S * FX_HIDDEN];
    moe(&m, &l, 0, x, S, out);

    /* Expected: per position, softmax the router logits, take both experts,
     * renormalize (sum already 1), accumulate reference expert outputs. */
    float want[S * FX_HIDDEN];
    memset(want, 0, sizeof(want));
    for (int s = 0; s < S; s++) {
        const float *xs = x + s * FX_HIDDEN;
        float logits[FX_EXPERTS];
        for (int e = 0; e < FX_EXPERTS; e++) {
            float acc = 0.f;
            for (int i = 0; i < FX_HIDDEN; i++)
                acc += xs[i] * gate[e * FX_HIDDEN + i];
            logits[e] = acc;
        }
        softmax_row(logits, FX_EXPERTS);
        float sum = 0.f;
        for (int e = 0; e < FX_EXPERTS; e++) sum += logits[e];
        for (int e = 0; e < FX_EXPERTS; e++)
            oracle_expert(&reader, 0, e, xs, logits[e] / sum,
                          want + s * FX_HIDDEN);
    }
    double e = rel_err(out, want, (size_t)S * FX_HIDDEN);
    printf("  moe one-layer nerr=%.2e\n", e);
    ck(e < 1e-5, "moe() equals router-weighted reference experts");
    coli_qpack_close(&reader);
}

/* With the pool covering every expert (default clamps to the container
 * total), the run so far must have filled each expert exactly once and
 * evicted nothing -- the resident-equivalent fast path. */
static void case_default_pool_stats(void) {
    printf("default pool bounds\n");
    int n_slots = 0;
    uint64_t ev = 0, fills = 0;
    qq_slot_stats(&n_slots, &ev, &fills);
    ck(n_slots == FX_LAYERS * FX_EXPERTS,
       "default pool clamps to the container total");
    ck(ev == 0, "no eviction while the pool covers every expert");
    ck(fills == FX_LAYERS * FX_EXPERTS, "every expert filled exactly once");
}

/* A pool SMALLER than the expert population must still be exact.
 * One slot serving four experts forces an evict/refill on every miss, and
 * every forward must still equal the independent affine reference. */
static void case_bounded_slots(const char *root) {
    printf("bounded slot pool: evict/refill parity (1 slot, %d experts)\n",
           FX_LAYERS * FX_EXPERTS);
    ColiQpackReader reader;
    char err[256];
    CHECK(coli_qpack_open(&reader, root, err, sizeof(err)) == 0);
    qq_set_slot_count(1);
    CHECK(qq_open(root, FX_LAYERS, FX_EXPERTS, FX_HIDDEN, FX_INTER,
                  err, sizeof(err)));
    int n_slots = 0;
    uint64_t ev = 0, fills = 0;
    qq_slot_stats(&n_slots, &ev, &fills);
    ck(n_slots == 1, "pool bounded to one slot");

    float x[FX_HIDDEN];
    for (int i = 0; i < FX_HIDDEN; i++)
        x[i] = 0.0625f * (float)((i * 5 + 2) % 13 - 6);
    double worst = 0.0;
    int all_ok = 1;
    for (int pass = 0; pass < 2; pass++)
        for (int layer = 0; layer < FX_LAYERS; layer++)
            for (int eid = 0; eid < FX_EXPERTS; eid++) {
                float got[FX_HIDDEN], want[FX_HIDDEN];
                memset(got, 0, sizeof(got));
                memset(want, 0, sizeof(want));
                if (!qq_expert_forward(layer, eid, x, 1.f, got)) {
                    all_ok = 0;
                    continue;
                }
                oracle_expert(&reader, layer, eid, x, 1.f, want);
                double e = rel_err(got, want, FX_HIDDEN);
                if (e > worst) worst = e;
            }
    qq_slot_stats(&n_slots, &ev, &fills);
    printf("  evict/refill worst nerr=%.2e  evictions=%llu fills=%llu\n",
           worst, (unsigned long long)ev, (unsigned long long)fills);
    ck(all_ok, "every forward through the one-slot pool succeeded");
    ck(worst < 1e-6, "evict/refill forwards match the affine reference");
    /* The open probe fills (0,0); pass 1 hits it and misses the other 3;
     * pass 2 misses all 4 -- every later miss displaces a live expert. */
    ck(fills == 8, "eight fills through the one slot");
    ck(ev == 7, "seven fills evicted a live expert");
    coli_qpack_close(&reader);
}

/* The stale-read refusal the slot contract requires.  Evict, refill the
 * SAME slot with a different expert, then present the old handle: dispatch
 * must refuse it without touching the accumulator, and the refilled slot
 * must compute the NEW expert's numbers. */
static void case_stale_refs(const char *root) {
    printf("slot generations: stale ref refused after evict/refill\n");
    ColiQpackReader reader;
    char err[256];
    CHECK(coli_qpack_open(&reader, root, err, sizeof(err)) == 0);
    CHECK(qq_active());   /* one-slot store still open from the case above */

    float x[FX_HIDDEN];
    for (int i = 0; i < FX_HIDDEN; i++)
        x[i] = 0.09375f * (float)((i * 3 + 1) % 11 - 5);

    QqExpertRef ref;
    CHECK(qq_expert_acquire(0, 0, &ref));
    float out[FX_HIDDEN];
    memset(out, 0, sizeof(out));
    ck(qq_ref_forward(ref, x, 1.f, out), "fresh ref dispatches");

    /* Evict (0,0): the only slot is refilled with (1,1). */
    QqExpertRef other;
    CHECK(qq_expert_acquire(1, 1, &other));
    ck(other.slot == ref.slot, "refill reused the same slot");
    ck(other.generation != ref.generation, "refill bumped the generation");

    float sink[FX_HIDDEN];
    for (int i = 0; i < FX_HIDDEN; i++) sink[i] = 42.f;
    ck(!qq_ref_forward(ref, x, 1.f, sink), "stale ref refused at dispatch");
    int untouched = 1;
    for (int i = 0; i < FX_HIDDEN; i++)
        if (sink[i] != 42.f) untouched = 0;
    ck(untouched, "refused dispatch leaves the accumulator untouched");

    /* The refilled slot computes the NEW expert, not stale bytes. */
    float got[FX_HIDDEN], want[FX_HIDDEN];
    memset(got, 0, sizeof(got));
    memset(want, 0, sizeof(want));
    ck(qq_ref_forward(other, x, 1.f, got), "refilled ref dispatches");
    oracle_expert(&reader, 1, 1, x, 1.f, want);
    double e = rel_err(got, want, FX_HIDDEN);
    printf("  refilled expert nerr=%.2e\n", e);
    ck(e < 1e-6, "refilled slot equals the reference for the NEW expert");

    QqExpertRef bogus = { 12345, other.generation };
    ck(!qq_ref_forward(bogus, x, 1.f, got), "out-of-range slot ref refused");

    /* Re-acquire (0,0): valid again under a fresh generation. */
    QqExpertRef again;
    CHECK(qq_expert_acquire(0, 0, &again));
    ck(again.generation != ref.generation,
       "re-acquired expert carries a new generation");
    memset(got, 0, sizeof(got));
    memset(want, 0, sizeof(want));
    ck(qq_ref_forward(again, x, 1.f, got), "re-acquired ref dispatches");
    oracle_expert(&reader, 0, 0, x, 1.f, want);
    ck(rel_err(got, want, FX_HIDDEN) < 1e-6,
       "re-acquired expert matches the reference");

    qq_close();
    ck(!qq_active(), "store inactive after close");
    qq_set_slot_count(0);
    coli_qpack_close(&reader);
}

int main(void) {
    const char *root = "qpack_qwen36_cpu_fixture.tmp";
    fx_cleanup(root);   /* stale dir from an aborted run */
    fx_build(root);

    case_open_validation(root);
    case_expert_forward(root);
    case_moe_wiring(root);
    case_default_pool_stats();
    qq_close();
    ck(!qq_active(), "store inactive after close");

    case_bounded_slots(root);
    case_stale_refs(root);

    fx_cleanup(root);
    printf(fails ? "qwen36 qpack CPU gate: FAILED\n"
                 : "qwen36 qpack CPU gate: ok\n");
    return fails ? 1 : 0;
}
