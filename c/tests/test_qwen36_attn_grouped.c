/* Bit-exactness and timing test for grouped GQA attention in qwen36.c.
 * Compares the grouped-by-(row, kv_head) loop against the original per-head
 * loop across kv16 and f32 caches, prefill and decode, attention sinks, and
 * sliding-window rings. No checkpoint required. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static uint32_t g_test_rng = 0x9e3779b9u;
static float test_frand(void) {
    g_test_rng ^= g_test_rng << 13;
    g_test_rng ^= g_test_rng >> 17;
    g_test_rng ^= g_test_rng << 5;
    return (float)(g_test_rng & 0xffffffu) / 8388608.0f - 1.0f;
}

static int g_failures;
#define CHECK(ok, ...) do { if (!(ok)) { \
    fprintf(stderr, "FAIL line %d: ", __LINE__); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    g_failures++; } } while (0)

/* Reference: the original per-head loop before GQA grouping. */
static void attention_core_ref(Model *m, Layer *l, int layer,
                               const float *query, const float *k, const float *vv,
                               int S, int pos_base, int win, int ring, float *ctx) {
    Cfg *c = &m->c;
    const int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, kvd = c->k_head_dim;
    const int q_per_kv = H / KV;
    const float scale = 1.f / sqrtf((float)hd);
    const Q36Row *mr = m->mux_rows;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int kvh = hh / q_per_kv;
            int qpos = mr ? mr[s].pos : pos_base + s;
            const void *Kl = mr ? mr[s].seq->K[layer] : m->K[layer];
            const void *Vl = mr ? mr[s].seq->V[layer] : m->V[layer];
            const float *qv = query + ((int64_t)s * H + hh) * hd;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *sc = m->attn_sc + (int64_t)tid * m->kv_cap;
            int t0 = win > 0 && qpos - win + 1 > 0 ? qpos - win + 1 : 0;
            int sp0 = mr ? qpos : pos_base;
            int tb = ring ? (sp0 > t0 ? sp0 : t0) : qpos + 1;
            const float *Kb = k + (int64_t)(mr ? s : 0) * KV * kvd + kvh * kvd;
            const float *Vb = vv + (int64_t)(mr ? s : 0) * KV * kvd + kvh * kvd;
            for (int t = t0; t <= qpos; t++) {
                float acc = t < tb ? q36_kv_dot(qv, Kl, q36_kv_at(m, layer, kvh, t), kvd)
                                   : dot_f32_lanes(qv, Kb + (int64_t)(t - sp0) * KV * kvd, kvd);
                sc[t] = acc * scale;
            }
            if (l->sinks) {
                float mx = l->sinks[hh];
                for (int t = t0; t <= qpos; t++) if (sc[t] > mx) mx = sc[t];
                float sum = expf(l->sinks[hh] - mx);
                for (int t = t0; t <= qpos; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                float inv = 1.f / sum;
                for (int t = t0; t <= qpos; t++) sc[t] *= inv;
            } else softmax_row(sc + t0, qpos + 1 - t0);
            float *cx = ctx + ((int64_t)s * H + hh) * hd;
            for (int dd = 0; dd < kvd; dd++) cx[dd] = 0;
            for (int t = t0; t <= qpos; t++) {
                float a = sc[t];
                if (t < tb) { q36_kv_axpy(cx, a, Vl, q36_kv_at(m, layer, kvh, t), kvd); continue; }
                const float *vrow = Vb + (int64_t)(t - sp0) * KV * kvd;
                for (int dd = 0; dd < kvd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
}

static void fill_rand(float *p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) p[i] = test_frand() * scale;
}

static void init_f32_qw(QW *w, int I, int O, float scale) {
    memset(w, 0, sizeof *w);
    w->I = I;
    w->O = O;
    float *buf = falloc((int64_t)O * I);
    fill_rand(buf, (size_t)O * I, scale);
    w->w = buf;
}

static void free_f32_qw(QW *w) {
    free((void *)w->w);
    memset(w, 0, sizeof *w);
}

static void run_case(const char *name, int H, int KV, int hd, int D,
                     int max_t, int kv16, int use_sinks, int win, int ring,
                     int hist_len, int S, int threads) {
    g_q36_kv16 = kv16;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    Model m;
    memset(&m, 0, sizeof m);
    m.c.n_layers = 1;
    m.c.hidden = D;
    m.c.q_heads = H;
    m.c.kv_heads = KV;
    m.c.head_dim = hd;
    m.c.k_head_dim = hd;
    m.c.q_head_dim = hd;   /* no gate, exercises full attention core directly */
    m.c.rotary_dim = hd / 2;
    m.c.theta = 10000.f;
    m.c.eps = 1e-6f;
    m.c.window = win;
    uint8_t win_layer[1] = { (uint8_t)(win > 0 ? 1 : 0) };
    uint8_t is_attn[1] = { 1 };
    m.c.win_layer = win_layer;
    m.c.is_attn = is_attn;

    m.max_t = max_t;
    m.kv_cap = max_t;
    m.kv_ring = ring;
    m.attn_sc_thr = threads > 0 ? threads : 1;
    m.attn_sc = falloc((int64_t)m.attn_sc_thr * m.kv_cap);
    if (ring) {
        m.ring_pos = malloc((size_t)ring * sizeof(int));
        for (int i = 0; i < ring; i++) m.ring_pos[i] = -1;
    }

    int kv_rows = ring ? ring : max_t;
    size_t kv_bytes = (size_t)KV * kv_rows * hd * q36_kv_es();
    void *K_buf = calloc(1, kv_bytes);
    void *V_buf = calloc(1, kv_bytes);
    m.K = malloc(sizeof(void *)); m.K[0] = K_buf;
    m.V = malloc(sizeof(void *)); m.V[0] = V_buf;

    Layer l;
    memset(&l, 0, sizeof l);
    init_f32_qw(&l.q, D, H * hd, 0.2f);
    init_f32_qw(&l.k, D, KV * hd, 0.2f);
    init_f32_qw(&l.v, D, KV * hd, 0.2f);
    init_f32_qw(&l.o, H * hd, D, 0.2f);
    if (use_sinks) {
        l.sinks = falloc(H);
        fill_rand(l.sinks, (size_t)H, 1.5f);
    }

    /* Seed history in K/V cache */
    float *tmp = falloc((int64_t)KV * hd);
    for (int t = 0; t < hist_len; t++) {
        fill_rand(tmp, (size_t)KV * hd, 0.5f);
        if (kv16) kv16_round(tmp, (int64_t)KV * hd);
        for (int kvh = 0; kvh < KV; kvh++)
            q36_kv_put(K_buf, q36_kv_at(&m, 0, kvh, t), tmp + (int64_t)kvh * hd, hd);
        fill_rand(tmp, (size_t)KV * hd, 0.5f);
        if (kv16) kv16_round(tmp, (int64_t)KV * hd);
        for (int kvh = 0; kvh < KV; kvh++)
            q36_kv_put(V_buf, q36_kv_at(&m, 0, kvh, t), tmp + (int64_t)kvh * hd, hd);
        if (ring) m.ring_pos[t % ring] = t;
    }
    free(tmp);

    /* Keep a snapshot of K/V and ring_pos so ref and got see identical initial state */
    void *K_snap = malloc(kv_bytes); memcpy(K_snap, K_buf, kv_bytes);
    void *V_snap = malloc(kv_bytes); memcpy(V_snap, V_buf, kv_bytes);
    int *rp_snap = ring ? malloc((size_t)ring * sizeof(int)) : NULL;
    if (ring) memcpy(rp_snap, m.ring_pos, (size_t)ring * sizeof(int));

    float *x = falloc((int64_t)S * D);
    fill_rand(x, (size_t)S * D, 0.5f);
    float *out_got = falloc((int64_t)S * D);
    float *out_ref = falloc((int64_t)S * D);

    /* Run optimized attention() */
    attention(&m, &l, 0, x, S, hist_len, out_got);

    /* Restore K/V state and run reference path */
    memcpy(K_buf, K_snap, kv_bytes);
    memcpy(V_buf, V_snap, kv_bytes);
    if (ring) memcpy(m.ring_pos, rp_snap, (size_t)ring * sizeof(int));

    float *q = falloc((int64_t)S * H * hd);
    float *k = falloc((int64_t)S * KV * hd);
    float *vv = falloc((int64_t)S * KV * hd);
    matmul_d(q, x, &l.q, S, D, H * hd);
    matmul_d(k, x, &l.k, S, D, KV * hd);
    matmul_d(vv, x, &l.v, S, D, KV * hd);
    for (int s = 0; s < S; s++) {
        int pos = hist_len + s;
        for (int hh = 0; hh < H; hh++)
            rope_head_partial(q + ((int64_t)s * H + hh) * hd, pos, m.c.rotary_dim, hd, m.c.theta);
        for (int kvh = 0; kvh < KV; kvh++)
            rope_head_partial(k + ((int64_t)s * KV + kvh) * hd, pos, m.c.rotary_dim, hd, m.c.theta);
    }
    if (kv16) { kv16_round(k, (int64_t)S * KV * hd); kv16_round(vv, (int64_t)S * KV * hd); }
    if (!ring) {
        for (int s = 0; s < S; s++) for (int kvh = 0; kvh < KV; kvh++) {
            q36_kv_put(K_buf, q36_kv_at(&m, 0, kvh, hist_len + s), k + ((int64_t)s * KV + kvh) * hd, hd);
            q36_kv_put(V_buf, q36_kv_at(&m, 0, kvh, hist_len + s), vv + ((int64_t)s * KV + kvh) * hd, hd);
        }
    }
    float *ctx_ref = falloc((int64_t)S * H * hd);
    attention_core_ref(&m, &l, 0, q, k, vv, S, hist_len, win, ring, ctx_ref);
    matmul_d(out_ref, ctx_ref, &l.o, S, H * hd, D);

    int exact = memcmp(out_got, out_ref, (size_t)S * D * sizeof(float)) == 0;
    if (!exact) {
        float maxd = 0.f;
        int at = -1;
        for (int i = 0; i < S * D; i++) {
            float d = fabsf(out_got[i] - out_ref[i]);
            if (d > maxd) { maxd = d; at = i; }
        }
        CHECK(0, "%s: mismatch max_diff=%.9g at %d (got=%.9g ref=%.9g)",
              name, (double)maxd, at, at >= 0 ? (double)out_got[at] : 0.0, at >= 0 ? (double)out_ref[at] : 0.0);
    } else {
        printf("  ok  %s (S=%d hist=%d kv16=%d sinks=%d ring=%d thr=%d)\n",
               name, S, hist_len, kv16, use_sinks, ring, threads);
    }

    free(q); free(k); free(vv); free(ctx_ref);
    free(x); free(out_got); free(out_ref);
    free(K_snap); free(V_snap); free(rp_snap);
    free(K_buf); free(V_buf); free(m.K); free(m.V);
    free(m.attn_sc); free(m.attn_sc_grp); free(m.ring_pos);
    free_f32_qw(&l.q); free_f32_qw(&l.k); free_f32_qw(&l.v); free_f32_qw(&l.o);
    free(l.sinks);
}

static void run_bench(int threads) {
    const int H = 16, KV = 2, hd = 256, max_t = 8192 + 512;
    g_q36_kv16 = 1;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    Model m; memset(&m, 0, sizeof m);
    m.c.n_layers = 1; m.c.q_heads = H; m.c.kv_heads = KV; m.c.head_dim = hd; m.c.k_head_dim = hd;
    uint8_t is_attn[1] = { 1 }; m.c.is_attn = is_attn;
    m.max_t = max_t; m.kv_cap = max_t;
    m.attn_sc_thr = threads; m.attn_sc = falloc((int64_t)threads * max_t);
    size_t kv_bytes = (size_t)KV * max_t * hd * sizeof(uint16_t);
    uint16_t *K_buf = malloc(kv_bytes), *V_buf = malloc(kv_bytes);
    m.K = malloc(sizeof(void *)); m.K[0] = K_buf;
    m.V = malloc(sizeof(void *)); m.V[0] = V_buf;
    Layer l; memset(&l, 0, sizeof l);
    float *tmp = falloc((int64_t)KV * hd);
    for (int t = 0; t < max_t; t++) {
        fill_rand(tmp, (size_t)KV * hd, 0.2f); kv16_pack(K_buf + (int64_t)t * hd, tmp, hd);
        kv16_pack(K_buf + ((int64_t)max_t + t) * hd, tmp + hd, hd);
        fill_rand(tmp, (size_t)KV * hd, 0.2f); kv16_pack(V_buf + (int64_t)t * hd, tmp, hd);
        kv16_pack(V_buf + ((int64_t)max_t + t) * hd, tmp + hd, hd);
    }
    free(tmp);

    struct { const char *label; int S; int pos_base; int iters; } cases[] = {
        { "prefill S=512  pos=0   ", 512,    0,  6 },
        { "prefill S=512  pos=2048", 512, 2048,  4 },
        { "decode  S=1    pos=4096",   1, 4096, 80 },
        { "decode  S=1    pos=8192",   1, 8192, 40 },
    };
    printf("\nBenchmark (H=%d KV=%d hd=%d kv16=1 threads=%d):\n", H, KV, hd, threads);
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        int S = cases[ci].S, pb = cases[ci].pos_base, iters = cases[ci].iters;
        float *query = falloc((int64_t)S * H * hd);
        float *ctx_a = falloc((int64_t)S * H * hd);
        float *ctx_b = falloc((int64_t)S * H * hd);
        fill_rand(query, (size_t)S * H * hd, 0.2f);

        /* Warm up and time ref */
        attention_core_ref(&m, &l, 0, query, NULL, NULL, S, pb, 0, 0, ctx_a);
        double t0 = now_s();
        for (int i = 0; i < iters; i++)
            attention_core_ref(&m, &l, 0, query, NULL, NULL, S, pb, 0, 0, ctx_a);
        double ref_ms = (now_s() - t0) * 1000.0 / iters;

        /* Warm up and time grouped (real attention_scores_ctx from qwen36.c) */
        attention_scores_ctx(&m, &l, 0, query, NULL, NULL, S, pb, 0, 0, ctx_b);
        t0 = now_s();
        for (int i = 0; i < iters; i++)
            attention_scores_ctx(&m, &l, 0, query, NULL, NULL, S, pb, 0, 0, ctx_b);
        double grp_ms = (now_s() - t0) * 1000.0 / iters;

        int gq = H / KV;
        int64_t want = S == 1 ? (int64_t)threads : (int64_t)4 * threads;
        while (gq > 1 && (int64_t)S * KV * ((H / KV) / gq) < want) {
            int d = gq - 1; while (d > 1 && (H / KV) % d) d--; gq = d;
        }
        if (S == 1 && gq < 4) gq = 1;
        CHECK(memcmp(ctx_a, ctx_b, (size_t)S * H * hd * sizeof(float)) == 0,
              "bench %s not bit-identical", cases[ci].label);
        printf("  %s (gq=%d): ref=%.2f ms  grouped=%.2f ms  speedup=%.2fx\n",
               cases[ci].label, gq, ref_ms, grp_ms, ref_ms / grp_ms);
        free(query); free(ctx_a); free(ctx_b);
    }
    free(K_buf); free(V_buf); free(m.K); free(m.V); free(m.attn_sc); free(m.attn_sc_grp);
}

int main(int argc, char **argv) {
    int orig_threads = 1;
#ifdef _OPENMP
    orig_threads = omp_get_max_threads();
#endif
    printf("Testing qwen36 grouped GQA attention (bit-exactness)...\n");
    run_case("kv16 prefill gq=8",     16, 2, 256, 64, 256, 1, 0,  0,  0,  32, 64, 8);
    run_case("kv16 decode  gq=4",     16, 2, 256, 64, 256, 1, 0,  0,  0, 128,  1, 4);
    run_case("kv16 decode  gq=2",     16, 2, 256, 64, 256, 1, 0,  0,  0, 128,  1, 8);
    run_case("kv16 decode  gq=1",     16, 2, 256, 64, 256, 1, 0,  0,  0, 128,  1, 16);
    run_case("f32  prefill gq=8",     16, 2, 128, 64, 256, 0, 0,  0,  0,  16, 48, 8);
    run_case("kv16 sinks   gq=8",     16, 2,  64, 64, 256, 1, 1,  0,  0,  20, 40, 4);
    run_case("kv16 ring    gq=4",     12, 3,  64, 48, 256, 1, 0, 32, 64,  40, 24, 4);
    run_case("kv16 odd_hd  gq=4",      8, 2, 100, 48, 128, 1, 1,  0,  0,  15, 24, 4);
    if (argc > 1 && !strcmp(argv[1], "bench")) run_bench(orig_threads);
    if (g_failures) { fprintf(stderr, "%d failure(s)\n", g_failures); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
