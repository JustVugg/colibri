/* olmoe: a prompt's MoE expert by expert (moe_by_expert) must give the bits of
 * the one-row loop.
 *
 * moe_routed() takes a prompt (S > 1) through moe_by_expert(), which groups the
 * block's (row, rank) pairs by expert and multiplies each expert against all of
 * its rows at once (matmul_q_rows). A decode step (S == 1) keeps the one-row
 * loop. Both must agree to the byte, or a prompt and the same tokens fed one at
 * a time would build different KV caches.
 *
 * What this file asserts, on a one-layer model held in memory (every expert
 * resident, so expert_get never reads a file):
 *   1. matmul_q_rows equals matmul_q row by row, memcmp, for n = 1..9 rows;
 *   2. moe_routed on S rows equals S calls of moe_routed on one row, memcmp,
 *      for S below, at and past OLMOE_PREFILL_ROWS (one, two and three blocks),
 *      with norm_topk off and on;
 *   3. the same with IDOT and FUSED3 forced on, where the build has them;
 *   4. hits + misses still count one request per (row, rank) pair.
 *
 * Build: make -C c tests/test_olmoe_moe_by_expert
 */
#define OLMOE_TESTING 1
#define main coli_olmoe_main_unused
#include "../olmoe.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { D = 128, I = 64, E = 8, K = 3 };   /* % 16 == 0: the IDOT and FUSED3 gates */

static uint32_t rng_state = 0x2545f491u;
static uint32_t xr32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float frand(void) { return (float)((double)(xr32() % 20001) / 10000.0 - 1.0); }
static void fill_q(int8_t *q, int64_t n) { for (int64_t i = 0; i < n; i++) q[i] = (int8_t)((int)(xr32() % 256) - 128); }
static void fill_s(float *s, int n) { for (int i = 0; i < n; i++) s[i] = 0.001f + (float)(xr32() % 1000) * 1e-5f; }

static int failures;
static void check(int ok, const char *what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

/* one layer, E experts all resident in slot e, indexed */
static void model_init_mem(Model *m) {
    memset(m, 0, sizeof *m);
    m->c.hidden = D; m->c.inter = I; m->c.n_experts = E; m->c.topk = K; m->c.n_layers = 1;
    m->hot_pinned = 1;   /* no route counters: route_trace.h is not set up here */
    m->cache = calloc(1, sizeof(LCache));
    LCache *lc = &m->cache[0];
    lc->slots = calloc(E, sizeof(Slot));
    lc->slot_by_expert = malloc(E * sizeof(int));
    lc->n = lc->cap = E;
    for (int e = 0; e < E; e++) {
        Slot *s = &lc->slots[e];
        s->g = malloc((size_t)I * D); s->u = malloc((size_t)I * D); s->d = malloc((size_t)D * I);
        s->gs = malloc(I * sizeof(float)); s->us = malloc(I * sizeof(float)); s->ds = malloc(D * sizeof(float));
        fill_q(s->g, (int64_t)I * D); fill_q(s->u, (int64_t)I * D); fill_q(s->d, (int64_t)D * I);
        fill_s(s->gs, I); fill_s(s->us, I); fill_s(s->ds, D);
        s->eid = e; lc->slot_by_expert[e] = e;
    }
}

static void rows_vs_one(const char *mode) {
    const int O = 96, II = 160;   /* O past one thread's share; II % 16 == 0 */
    int8_t *q = malloc((size_t)O * II); float *sc = malloc(O * sizeof(float));
    float *x = malloc((size_t)9 * II * sizeof(float));
    float *ya = malloc((size_t)9 * O * sizeof(float)), *yb = malloc((size_t)9 * O * sizeof(float));
    fill_q(q, (int64_t)O * II); fill_s(sc, O);
    for (int i = 0; i < 9 * II; i++) x[i] = frand();
    int same = 1;
    for (int n = 1; n <= 9; n++) {
        const float *xs[9]; float *ys[9];
        for (int t = 0; t < n; t++) { xs[t] = x + (int64_t)t * II; ys[t] = ya + (int64_t)t * O; }
        matmul_q_rows(ys, xs, n, q, sc, II, O);
        for (int t = 0; t < n; t++) matmul_q(yb + (int64_t)t * O, xs[t], q, sc, II, O);
        if (memcmp(ya, yb, (size_t)n * O * sizeof(float))) same = 0;
    }
    char what[128]; snprintf(what, sizeof what, "%s: matmul_q_rows == matmul_q per row, n = 1..9", mode);
    check(same, what);
    free(q); free(sc); free(x); free(ya); free(yb);
}

static void prompt_vs_rows(Model *m, const char *mode) {
    const int sizes[] = { 2, 5, OLMOE_PREFILL_ROWS, OLMOE_PREFILL_ROWS + 1, 2 * OLMOE_PREFILL_ROWS + 37 };
    for (int norm = 0; norm <= 1; norm++) {
        m->c.norm_topk = norm;
        for (size_t z = 0; z < sizeof sizes / sizeof *sizes; z++) {
            int S = sizes[z];
            float *x = malloc((size_t)S * D * sizeof(float));
            float *la = malloc((size_t)S * E * sizeof(float)), *lb = malloc((size_t)S * E * sizeof(float));
            float *oa = malloc((size_t)S * D * sizeof(float)), *ob = malloc((size_t)S * D * sizeof(float));
            for (int i = 0; i < S * D; i++) x[i] = frand();
            for (int i = 0; i < S * E; i++) la[i] = lb[i] = 4.f * frand();
            uint64_t req0 = m->hits + m->miss;
            moe_routed(m, 0, x, S, la, oa);                 /* the prompt: expert by expert */
            uint64_t req = m->hits + m->miss - req0;
            for (int s = 0; s < S; s++)                     /* the same rows, one at a time */
                moe_routed(m, 0, x + (int64_t)s * D, 1, lb + (int64_t)s * E, ob + (int64_t)s * D);
            char what[160];
            snprintf(what, sizeof what, "%s: prompt of %d rows (norm_topk %d) == %d one-row steps, memcmp",
                     mode, S, norm, S);
            check(!memcmp(oa, ob, (size_t)S * D * sizeof(float)), what);
            snprintf(what, sizeof what, "%s: prompt of %d rows counts %d requests (%llu)",
                     mode, S, S * K, (unsigned long long)req);
            check(req == (uint64_t)S * K, what);
            free(x); free(la); free(lb); free(oa); free(ob);
        }
    }
}

int main(void) {
    static Model m;
    model_init_mem(&m);
    unsetenv("IDOT");

    rows_vs_one("default");
    prompt_vs_rows(&m, "default");

#if defined(HAVE_FAST_DOT_I8)
    matmul_q_idot_force = 1;
    rows_vs_one("IDOT=1");
    prompt_vs_rows(&m, "IDOT=1");
    matmul_q_idot_force = 0;
#else
    printf("SKIP  IDOT: no fast int8 dot in this build\n");
#endif
#if defined(__AVX2__)
    g_fused3 = 1;
    prompt_vs_rows(&m, "FUSED3=1");
    g_fused3 = 0;
#else
    printf("SKIP  FUSED3: no AVX2 in this build\n");
#endif

    printf("\n%s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
