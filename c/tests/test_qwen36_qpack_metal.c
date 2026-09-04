/* Apple parity gates for the qwen36 qpack MLX-affine routed-expert path
 * (Metal vs the CPU affine reference, bounded slots).  The slot pool is
 * forced SMALLER than one layer's expert population (EVICT_SLOTS), so the
 * gates below cannot pass without live eviction/refill of the whole-slot
 * Metal buffers during the runs.  Three gates; the first two run the SAME
 * engine code with the dispatch flipped by qq_force_cpu():
 *
 *   1. one-layer parity  -- moe() on one transformer layer's MoE block:
 *      routed experts through coli_metal_matmul_affine vs the CPU affine
 *      reference, same router, same shared expert.
 *   2. logit parity      -- an end-to-end step() (prefill S=5 + one decode
 *      token) through the full tiny hybrid stack: DeltaNet + Gated Attention
 *      + MoE on every layer, logits compared between the two dispatch modes.
 *      Attention and DeltaNet run on the CPU in BOTH modes: only the routed
 *      experts move, which is exactly this slice's claim.
 *
 * Both gates run the REAL Swiftlet-generated fixture: set
 * QWEN36_QPACK_FIXTURE=<dir> to a swiftlet-repack output for
 * fixtures/tiny-model-q4.  Without it the binary SKIPS with a distinct
 * marker -- it never pretends to have proven parity.  The logit gate is
 * pinned to the tiny-model-q4 geometry (the DeltaNet/attention dims are not
 * in the container); a container with any other geometry fails loudly.
 *
 *   3. stale-ref refusal   -- acquire an expert, cycle the pool until its
 *      slot is refilled with a different expert, then present the old
 *      generation ref: dispatch must refuse it with ZERO Metal or CPU
 *      projections, and a fresh acquire must dispatch on Metal again.
 *
 * Dispatch is proven, not assumed: qq_counts() must show every routed
 * projection of the Metal run on the GPU and none on the CPU fallback, the
 * pool stats must show evictions actually happened during the parity runs,
 * and coli_metal_stats must show the resident slot buffers bounded by the
 * pool size (EVICT_SLOTS whole-slot buffers total, not three handles per
 * expert). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../qpack.h"
#include "../backend_metal.h"

#define CHECK(condition) do {                                                   \
    if (!(condition)) {                                                         \
        fprintf(stderr, "%s:%d: check failed: %s\n",                            \
                __FILE__, __LINE__, #condition);                                \
        exit(1);                                                                \
    }                                                                           \
} while (0)

/* tiny-model-q4 geometry (fixtures/tiny-model-q4/config.json in Swiftlet);
 * the qpack container carries hidden/inter/expert/layer counts, everything
 * else here is the dense half the logit gate synthesizes deterministically. */
enum {
    TINY_HIDDEN = 64, TINY_INTER = 32, TINY_EXPERTS = 8, TINY_LAYERS = 8,
    TINY_TOPK = 2, TINY_VOCAB = 128, TINY_SHARED_INTER = 32,
    TINY_Q_HEADS = 4, TINY_KV_HEADS = 2, TINY_HEAD_DIM = 16,
    TINY_ROTARY = 4,                       /* head_dim * partial_rotary 0.25 */
    TINY_DN_VHEADS = 4, TINY_DN_KHEADS = 2, TINY_DN_KDIM = 8,
    TINY_DN_VDIM = 8, TINY_DN_CONVK = 4,
    TINY_DN_CONV_DIM = 2 * TINY_DN_KHEADS * TINY_DN_KDIM +
                       TINY_DN_VHEADS * TINY_DN_VDIM,      /* 64 */
    PREFILL_S = 5, MAX_T = 16
};

/* Pool smaller than one layer's expert population, so the parity gates
 * cannot pass without live eviction/refill. */
enum { EVICT_SLOTS = 5 };

static int fails = 0;

static double rel_err(const float *a, const float *b, size_t n) {
    double maxabs = 0.0, ymax = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)a[i]) || !isfinite((double)b[i]))
            return (double)INFINITY;
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > maxabs) maxabs = d;
        double m = fabs((double)b[i]);
        if (m > ymax) ymax = m;
    }
    return maxabs / (ymax + 1e-9);
}

/* Deterministic dense-weight synthesis (LCG, seed carried by the caller). */
static float lcg_float(uint32_t *state, float scale) {
    *state = *state * 1664525u + 1013904223u;
    return scale * ((float)(*state >> 8) / 8388608.0f - 1.0f);
}

static float *tw(int64_t n, float scale, uint32_t *state) {
    float *w = falloc(n);
    for (int64_t i = 0; i < n; i++) w[i] = lcg_float(state, scale);
    return w;
}

/* ---- gate 1: one-layer MoE parity, Metal vs CPU reference ---------------- */
static int case_one_layer(int hidden, int inter, int n_experts) {
    enum { S = 3 };
    printf("one-layer MoE parity (S=%d, topk=%d, E=%d)\n",
           S, TINY_TOPK, n_experts);
    Model m;
    memset(&m, 0, sizeof(m));
    m.c.hidden = hidden; m.c.inter = inter; m.c.n_experts = n_experts;
    m.c.topk = TINY_TOPK; m.c.n_group = 1; m.c.eps = 1e-6f;
    m.c.shared_inter = inter < TINY_SHARED_INTER ? inter : TINY_SHARED_INTER;
    Layer l;
    memset(&l, 0, sizeof(l));
    uint32_t seed = 0xC3C3C3C3u;
    l.gate = tw((int64_t)n_experts * hidden, 0.5f, &seed);
    l.sh_g = tw((int64_t)m.c.shared_inter * hidden, 0.06f, &seed);
    l.sh_u = tw((int64_t)m.c.shared_inter * hidden, 0.06f, &seed);
    l.sh_d = tw((int64_t)hidden * m.c.shared_inter, 0.06f, &seed);
    l.sh_gate = tw(hidden, 0.06f, &seed);
    float *x = tw((int64_t)S * hidden, 0.4f, &seed);
    float *out_cpu = falloc((int64_t)S * hidden);
    float *out_gpu = falloc((int64_t)S * hidden);

    qq_force_cpu(1);
    moe(&m, &l, 0, x, S, out_cpu);
    uint64_t metal0, cpu0, metal1, cpu1;
    qq_counts(&metal0, &cpu0);
    qq_force_cpu(0);
    moe(&m, &l, 0, x, S, out_gpu);
    qq_counts(&metal1, &cpu1);

    double nerr = rel_err(out_gpu, out_cpu, (size_t)S * hidden);
    uint64_t want_metal = (uint64_t)S * TINY_TOPK * 3;
    int dispatched = metal1 - metal0 == want_metal && cpu1 == cpu0;
    int ok = dispatched && nerr < 1e-4;
    printf("  moe one-layer            nerr=%.2e  metal=%llu/%llu cpu-fallback=%llu  %s\n",
           nerr, (unsigned long long)(metal1 - metal0),
           (unsigned long long)want_metal,
           (unsigned long long)(cpu1 - cpu0),
           ok ? "ok" : "*** MISMATCH");
    free(l.gate); free(l.sh_g); free(l.sh_u); free(l.sh_d); free(l.sh_gate);
    free(x); free(out_cpu); free(out_gpu);
    return ok ? 0 : 1;
}

/* ---- gate 2: end-to-end logit parity on the tiny hybrid stack ------------ */
static Model g_m;   /* static like the engine's own main(): outlives threads */

static void tiny_model_build(void) {
    Model *m = &g_m;
    memset(m, 0, sizeof(*m));
    Cfg *c = &m->c;
    c->hidden = TINY_HIDDEN; c->n_layers = TINY_LAYERS;
    c->q_heads = TINY_Q_HEADS; c->kv_heads = TINY_KV_HEADS;
    c->head_dim = TINY_HEAD_DIM; c->k_head_dim = TINY_HEAD_DIM;
    c->v_head_dim = TINY_HEAD_DIM;
    c->q_head_dim = 2 * TINY_HEAD_DIM;          /* attn_output_gate */
    c->o_in = TINY_Q_HEADS * TINY_HEAD_DIM;
    c->rope_dim = TINY_ROTARY; c->rotary_dim = TINY_ROTARY;
    c->n_experts = TINY_EXPERTS; c->topk = TINY_TOPK;
    c->inter = TINY_INTER; c->shared_inter = TINY_SHARED_INTER;
    c->vocab = TINY_VOCAB;
    c->n_group = 1; c->topk_group = 0;
    c->theta = 10000000.0f; c->eps = 1e-6f;
    c->partial_rotary_factor = 0.25f;
    c->norm_topk = 1; c->has_qk_norm = 1; c->has_bias = 0;
    c->attn_output_gate = 1;
    c->dn_vheads = TINY_DN_VHEADS; c->dn_kheads = TINY_DN_KHEADS;
    c->dn_kdim = TINY_DN_KDIM; c->dn_vdim = TINY_DN_VDIM;
    c->dn_convk = TINY_DN_CONVK; c->dn_conv_dim = TINY_DN_CONV_DIM;
    c->is_attn = calloc(TINY_LAYERS, 1);
    CHECK(c->is_attn);
    for (int i = 0; i < TINY_LAYERS; i++)
        c->is_attn[i] = (i % 4 == 3);           /* full_attention_interval 4 */

    uint32_t seed = 0x51E77E1u;
    const int D = c->hidden;
    const int q_out = c->q_heads * c->q_head_dim;
    const int kv_out = c->kv_heads * c->k_head_dim;
    const int value_dim = c->dn_vheads * c->dn_vdim;
    m->embed = tw((int64_t)c->vocab * D, 0.5f, &seed);
    m->lm_head = tw((int64_t)c->vocab * D, 0.25f, &seed);
    m->final_norm = tw(D, 0.1f, &seed);
    m->L = calloc(TINY_LAYERS, sizeof(Layer));
    m->DN_rec = calloc(TINY_LAYERS, sizeof(float *));
    m->DN_conv = calloc(TINY_LAYERS, sizeof(float *));
    CHECK(m->L && m->DN_rec && m->DN_conv);
    for (int i = 0; i < TINY_LAYERS; i++) {
        Layer *l = &m->L[i];
        l->in_ln = tw(D, 0.1f, &seed);
        l->post_ln = tw(D, 0.1f, &seed);
        l->gate = tw((int64_t)c->n_experts * D, 0.5f, &seed);
        l->sh_g = tw((int64_t)c->shared_inter * D, 0.06f, &seed);
        l->sh_u = tw((int64_t)c->shared_inter * D, 0.06f, &seed);
        l->sh_d = tw((int64_t)D * c->shared_inter, 0.06f, &seed);
        l->sh_gate = tw(D, 0.06f, &seed);
        if (c->is_attn[i]) {
            l->q = tw((int64_t)q_out * D, 0.06f, &seed);
            l->k = tw((int64_t)kv_out * D, 0.06f, &seed);
            l->v = tw((int64_t)kv_out * D, 0.06f, &seed);
            l->o = tw((int64_t)D * c->o_in, 0.06f, &seed);
            l->qn = tw(c->head_dim, 0.1f, &seed);
            l->kn = tw(c->k_head_dim, 0.1f, &seed);
        } else {
            l->dn_qkv = tw((int64_t)c->dn_conv_dim * D, 0.06f, &seed);
            l->dn_z = tw((int64_t)value_dim * D, 0.06f, &seed);
            l->dn_b = tw((int64_t)c->dn_vheads * D, 0.06f, &seed);
            l->dn_a = tw((int64_t)c->dn_vheads * D, 0.06f, &seed);
            l->dn_conv = tw((int64_t)c->dn_conv_dim * c->dn_convk, 0.2f, &seed);
            l->dn_dtbias = tw(c->dn_vheads, 0.02f, &seed);
            l->dn_alog = tw(c->dn_vheads, 0.1f, &seed);
            l->dn_norm = tw(c->dn_vdim, 0.1f, &seed);
            l->dn_out = tw((int64_t)D * value_dim, 0.06f, &seed);
            m->DN_rec[i] = calloc((size_t)c->dn_vheads * c->dn_kdim *
                                  c->dn_vdim, sizeof(float));
            m->DN_conv[i] = calloc((size_t)c->dn_conv_dim *
                                   (c->dn_convk - 1), sizeof(float));
            CHECK(m->DN_rec[i] && m->DN_conv[i]);
        }
    }
    m->max_t = MAX_T;
    ensure_kv(m);
}

static void tiny_model_reset(void) {
    reset_recurrent(&g_m);
    g_m.kv_len = 0;
    g_m.token_count = 0;
    g_m.freq_token_count = 0;
}

/* One full run: prefill PREFILL_S tokens, then one decode token.  Copies the
 * two logit rows into caller storage. */
static void tiny_model_run(float *prefill_logits, float *decode_logits) {
    static const int ids[PREFILL_S] = { 1, 7, 42, 99, 3 };
    static const int decode_id = 64;
    tiny_model_reset();
    float *lo = step(&g_m, ids, PREFILL_S, 0);
    memcpy(prefill_logits, lo, TINY_VOCAB * sizeof(float));
    free(lo);
    lo = step(&g_m, &decode_id, 1, PREFILL_S);
    memcpy(decode_logits, lo, TINY_VOCAB * sizeof(float));
    free(lo);
}

static int case_logits(void) {
    printf("end-to-end logit parity (tiny hybrid stack, prefill S=%d + 1 decode)\n",
           PREFILL_S);
    tiny_model_build();
    float cpu_pre[TINY_VOCAB], cpu_dec[TINY_VOCAB];
    float gpu_pre[TINY_VOCAB], gpu_dec[TINY_VOCAB];

    qq_force_cpu(1);
    tiny_model_run(cpu_pre, cpu_dec);
    uint64_t metal0, cpu0, metal1, cpu1;
    qq_counts(&metal0, &cpu0);
    qq_force_cpu(0);
    tiny_model_run(gpu_pre, gpu_dec);
    qq_counts(&metal1, &cpu1);

    double nerr_pre = rel_err(gpu_pre, cpu_pre, TINY_VOCAB);
    double nerr_dec = rel_err(gpu_dec, cpu_dec, TINY_VOCAB);
    /* every layer's MoE, every position, topk experts, 3 projections each */
    uint64_t want_metal =
        (uint64_t)TINY_LAYERS * (PREFILL_S + 1) * TINY_TOPK * 3;
    int dispatched = metal1 - metal0 == want_metal && cpu1 == cpu0;
    int ok_pre = nerr_pre < 1e-4, ok_dec = nerr_dec < 1e-4;
    printf("  logits prefill (S=%d)     nerr=%.2e  %s\n",
           PREFILL_S, nerr_pre, ok_pre ? "ok" : "*** MISMATCH");
    printf("  logits decode  (pos=%d)   nerr=%.2e  %s\n",
           PREFILL_S, nerr_dec, ok_dec ? "ok" : "*** MISMATCH");
    printf("  routed dispatch          metal=%llu/%llu cpu-fallback=%llu  %s\n",
           (unsigned long long)(metal1 - metal0),
           (unsigned long long)want_metal,
           (unsigned long long)(cpu1 - cpu0),
           dispatched ? "ok" : "*** NOT ALL ON METAL");
    return (ok_pre && ok_dec && dispatched) ? 0 : 1;
}

/* ---- gate 3: stale slot refs are refused on the Metal path too ----------- */
static int case_stale_ref(void) {
    printf("slot generations: stale ref refused after evict/refill (Metal)\n");
    qq_force_cpu(0);
    QqExpertRef ref;
    CHECK(qq_expert_acquire(0, 0, &ref));
    /* With EVICT_SLOTS slots, acquiring EVICT_SLOTS other experts must
     * displace (0,0). */
    for (int e = 1; e <= EVICT_SLOTS; e++) {
        QqExpertRef spin;
        CHECK(qq_expert_acquire(0, e, &spin));
    }
    uint64_t metal0, cpu0, metal1, cpu1;
    qq_counts(&metal0, &cpu0);
    float x[TINY_HIDDEN], out[TINY_HIDDEN];
    for (int i = 0; i < TINY_HIDDEN; i++) {
        x[i] = 0.03125f * (float)((i * 7 + 2) % 17 - 8);
        out[i] = 5.f;
    }
    int refused = !qq_ref_forward(ref, x, 1.f, out);
    qq_counts(&metal1, &cpu1);
    int untouched = 1;
    for (int i = 0; i < TINY_HIDDEN; i++)
        if (out[i] != 5.f) untouched = 0;
    int no_dispatch = metal1 == metal0 && cpu1 == cpu0;
    /* A fresh acquire of the same expert is valid again -- new generation --
     * and its three projections land on Metal. */
    QqExpertRef again;
    CHECK(qq_expert_acquire(0, 0, &again));
    memset(out, 0, sizeof(out));
    int redispatched = qq_ref_forward(again, x, 1.f, out) &&
                       again.generation != ref.generation;
    qq_counts(&metal1, &cpu1);
    int on_metal = metal1 - metal0 == 3 && cpu1 == cpu0;
    int ok = refused && untouched && no_dispatch && redispatched && on_metal;
    printf("  stale ref: refused=%d accumulator-untouched=%d zero-dispatch=%d;"
           " re-acquired metal=%llu/3 cpu-fallback=%llu  %s\n",
           refused, untouched, no_dispatch,
           (unsigned long long)(metal1 - metal0),
           (unsigned long long)(cpu1 - cpu0),
           ok ? "ok" : "*** MISMATCH");
    return ok ? 0 : 1;
}

int main(void) {
    const char *fixture = getenv("QWEN36_QPACK_FIXTURE");
    if (!fixture || !*fixture) {
        printf("SKIPPED: QWEN36_QPACK_FIXTURE not set -- qpack parity gates "
               "need a Swiftlet fixture (swiftlet-repack of tiny-model-q4)\n");
        return 0;
    }
    if (!coli_metal_init()) {
        printf("Metal backend initialization failed -- this gate needs real "
               "Apple hardware\n");
        return 1;
    }
    ColiMetalAffineCapability capability = coli_metal_affine_capability();
    if (capability == COLI_METAL_AFFINE_CAP_SIMD_WIDTH_UNSUPPORTED) {
        printf("SKIPPED: MLX affine pipelines need 32-lane simdgroups\n");
        coli_metal_shutdown();
        return 0;
    }
    if (capability != COLI_METAL_AFFINE_CAP_READY) {
        printf("qwen36 qpack metal parity: FAIL (affine capability=%d)\n",
               (int)capability);
        coli_metal_shutdown();
        return 1;
    }

    /* Derive the container geometry the same way the engine will. */
    ColiQpackReader reader;
    char err[256];
    if (coli_qpack_open(&reader, fixture, err, sizeof(err))) {
        printf("qwen36 qpack metal parity: FAIL (open %s: %s)\n", fixture, err);
        coli_metal_shutdown();
        return 1;
    }
    size_t stride = reader.layout.expert_stride;
    unsigned char *blob = malloc(stride);
    CHECK(blob);
    CHECK(coli_qpack_read_expert(&reader, 0, 0, blob, stride,
                                 err, sizeof(err)) == 0);
    ColiAffineQuantizedView probe;
    CHECK(coli_qpack_affine_view(&reader, blob, stride, "gate_proj",
                                 &probe, err, sizeof(err)) == 0);
    int hidden = (int)probe.input_dim, inter = (int)probe.output_dim;
    int n_layers = (int)reader.layout.layer_count;
    int n_experts = (int)reader.layout.expert_count;
    printf("fixture %s: layers=%d experts=%d hidden=%d inter=%d Q%d gs=%d "
           "scalars=%s\n",
           fixture, n_layers, n_experts, hidden, inter, reader.quant_bits,
           reader.quant_group_size,
           probe.scalar_format == COLI_AFFINE_SCALAR_F32 ? "f32" :
           probe.scalar_format == COLI_AFFINE_SCALAR_F16 ? "f16" : "bf16");
    /* The logit gate synthesizes the dense half from tiny-model-q4's config;
     * any other geometry means the wrong fixture was passed. */
    int is_tiny = hidden == TINY_HIDDEN && inter == TINY_INTER &&
                  n_layers == TINY_LAYERS && n_experts == TINY_EXPERTS;
    free(blob);
    coli_qpack_close(&reader);
    if (!is_tiny) {
        printf("qwen36 qpack metal parity: FAIL (fixture is not tiny-model-q4:"
               " want layers=%d experts=%d hidden=%d inter=%d)\n",
               TINY_LAYERS, TINY_EXPERTS, TINY_HIDDEN, TINY_INTER);
        coli_metal_shutdown();
        return 1;
    }

    size_t count_before = 0, bytes_before = 0;
    coli_metal_stats(&count_before, &bytes_before);
    /* Force the pool SMALLER than one layer's expert count so both
     * parity gates run through constant eviction/refill -- the bounded pool
     * is the path under test, not a fully resident cache. */
    qq_set_slot_count(EVICT_SLOTS);
    if (!qq_open(fixture, n_layers, n_experts, hidden, inter,
                 err, sizeof(err))) {
        printf("qwen36 qpack metal parity: FAIL (qq_open: %s)\n", err);
        coli_metal_shutdown();
        return 1;
    }
    {
        int pool = 0;
        qq_slot_stats(&pool, NULL, NULL);
        printf("slot pool: %d slots for %d experts x %d layers\n",
               pool, n_experts, n_layers);
        if (pool != EVICT_SLOTS) {
            printf("qwen36 qpack metal parity: FAIL (pool=%d, want %d)\n",
                   pool, EVICT_SLOTS);
            qq_close();
            coli_metal_shutdown();
            return 1;
        }
    }

    fails |= case_one_layer(hidden, inter, n_experts);
    fails |= case_logits();
    fails |= case_stale_ref();

    {
        int pool = 0;
        uint64_t ev = 0, fills = 0;
        qq_slot_stats(&pool, &ev, &fills);
        size_t count_mid = 0, bytes_mid = 0;
        coli_metal_stats(&count_mid, &bytes_mid);
        int evicted = ev > 0 && fills > (uint64_t)EVICT_SLOTS;
        int bounded = count_mid - count_before == (size_t)EVICT_SLOTS;
        printf("  slot pool churn          evictions=%llu fills=%llu pool=%d  %s\n",
               (unsigned long long)ev, (unsigned long long)fills, pool,
               evicted ? "ok" : "*** NO EVICTION DURING PARITY");
        printf("  resident slot buffers    %zu buffers / %zu bytes (bounded by the pool)  %s\n",
               count_mid - count_before, bytes_mid - bytes_before,
               bounded ? "ok" : "*** MISMATCH");
        if (!evicted || !bounded) fails = 1;
    }

    qq_close();
    qq_set_slot_count(0);
    size_t count_after = 0, bytes_after = 0;
    coli_metal_stats(&count_after, &bytes_after);
    if (count_after != count_before || bytes_after != bytes_before) {
        printf("  resident handle leak: %zu tensors / %zu bytes left behind"
               "  *** MISMATCH\n",
               count_after - count_before, bytes_after - bytes_before);
        fails = 1;
    } else {
        printf("  resident affine handles released (count/bytes restored)  ok\n");
    }

    printf(fails ? "qwen36 qpack metal parity: FAILED\n"
                 : "qwen36 qpack metal parity: ok\n");
    coli_metal_shutdown();
    return fails ? 1 : 0;
}
