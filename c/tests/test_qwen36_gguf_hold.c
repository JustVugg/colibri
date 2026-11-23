/* Model-free regression: the GGUF CPU MoE (moe_gq_decode, moe_gq_run) holds the
 * slots it computes from, so the PILOT worker's loads cannot evict them and
 * write another expert's bytes under the matmul. */
#define COLI_CACHE_INDEX_TEST 1
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

enum { D = 256, I = 512, E = 6, K = 2, CAP = 3, STEPS = 400, ROWS = 3, WIDE = 12 };

static Model M;
static volatile int stop_worker;

static void put(FILE *f, const void *p, size_t n) { if (fwrite(p, 1, n, f) != n) { perror("fwrite"); exit(1); } }
static void put_u32(FILE *f, uint32_t v) { put(f, &v, 4); }
static void put_u64(FILE *f, uint64_t v) { put(f, &v, 8); }
static void put_str(FILE *f, const char *s) { put_u64(f, strlen(s)); put(f, s, strlen(s)); }
static void pad32(FILE *f) { while (ftell(f) % 32) fputc(0, f); }

/* gate, up and down for E experts as Q8_0, every expert with its own codes */
static void write_gguf(const char *path) {
    static const char *name[3] = { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight" };
    const uint64_t rows[3] = { I, I, D }, cols[3] = { D, D, I };
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); exit(1); }
    put(f, "GGUF", 4); put_u32(f, 3); put_u64(f, 3); put_u64(f, 1);
    put_str(f, "general.architecture"); put_u32(f, 8); put_str(f, "qwen35moe");
    uint64_t off = 0;
    for (int t = 0; t < 3; t++) {
        put_str(f, name[t]); put_u32(f, 3); put_u64(f, cols[t]); put_u64(f, rows[t]); put_u64(f, E);
        put_u32(f, GQ_Q8_0); put_u64(f, off);
        off += (E * rows[t] * (cols[t] / 32) * 34 + 31) / 32 * 32;
    }
    pad32(f);
    uint32_t seed = 1;
    for (int t = 0; t < 3; t++) {
        for (uint64_t b = 0; b < E * rows[t] * (cols[t] / 32); b++) {
            uint8_t blk[34] = { 0x29, 0x20 };   /* f16 0.0098 */
            for (int i = 2; i < 34; i++) { seed = seed * 1664525u + 1013904223u; blk[i] = (uint8_t)(seed >> 24); }
            put(f, blk, 34);
        }
        pad32(f);
    }
    fclose(f);
}

static void setup(const char *path) {
    if (gguf_open(&g_gg, path)) { fprintf(stderr, "open: %s\n", g_gg.err); exit(1); }
    g_gguf = 1;
    Cfg *c = &M.c; c->n_layers = 1; c->n_experts = E; c->topk = K; c->inter = I; c->hidden = D;
    M.L = calloc(1, sizeof(Layer));
    M.gqx = calloc(1, sizeof(GqExp));
    GqExp *x = &M.gqx[0];
    x->g = gguf_find(&g_gg, "blk.0.ffn_gate_exps.weight"); x->u = gguf_find(&g_gg, "blk.0.ffn_up_exps.weight");
    x->d = gguf_find(&g_gg, "blk.0.ffn_down_exps.weight");
    x->tg = x->g->type; x->tu = x->u->type; x->td = x->d->type;
    x->eb_g = gq_row_bytes(x->tg, D) * I; x->eb_u = gq_row_bytes(x->tu, D) * I; x->eb_d = gq_row_bytes(x->td, I) * D;
    M.gqe_max = x->eb_g + x->eb_u + x->eb_d;
    gguf_expert_held(&M, x);
    M.cache = calloc(1, sizeof(LCache)); LCache *lc = &M.cache[0]; lc->cap = CAP;
    lc->slots = calloc(CAP, sizeof(Slot));
    lc->slot_by_expert = malloc(E * sizeof(int));
    for (int e = 0; e < E; e++) lc->slot_by_expert[e] = -1;
    M.is_pinned = calloc(E, 1); M.is_queued = calloc(E, 1);
}

/* the PILOT worker's load, one candidate of this layer after another */
static void *worker(void *arg) {
    (void)arg;
    for (unsigned n = 0; !stop_worker; n++) {
        const int e = (int)(n % E);
        pthread_mutex_lock(&g_pilot_mx); M.is_queued[e] = 1; pthread_mutex_unlock(&g_pilot_mx);
        pilot_realload(&M, 0, e);
    }
    return NULL;
}

/* step `it` with S rows: S = 1 is moe_gq_decode, more is moe_gq_run's own loop (a verify's
 * rows up to GQ_EXACT_ROWS, a prompt's above) */
static void moe_step(int it, int S, const float *x, float *out) {
    int idx[WIDE * K]; float val[WIDE * K];
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) { idx[s * K + k] = (it + s + k * 2) % E; val[s * K + k] = 1.f / (float)(k + 1); }
    memset(out, 0, sizeof(float) * (size_t)S * D);
    moe_gq_run(&M, 0, x, S, out, idx, val);
}

static int differing(int S, const float *x) {
    float *ref = falloc((int64_t)STEPS * S * D), *out = falloc((int64_t)S * D);
    for (int it = 0; it < STEPS; it++) moe_step(it, S, x, ref + (int64_t)it * S * D);
    /* QWEN36_TEST_OUTPUT: the reference appended to a file, to compare two revisions with cmp */
    const char *keep = getenv("QWEN36_TEST_OUTPUT");
    if (keep && *keep) {
        FILE *f = fopen(keep, "ab"); if (!f) { perror(keep); exit(1); }
        put(f, ref, sizeof(float) * (size_t)STEPS * S * D); fclose(f);
    }
    stop_worker = 0;
    pthread_t th; pthread_create(&th, NULL, worker, NULL);
    int bad = 0;
    for (int it = 0; it < STEPS; it++) {
        moe_step(it, S, x, out);
        bad += memcmp(out, ref + (int64_t)it * S * D, sizeof(float) * (size_t)S * D) != 0;
    }
    stop_worker = 1; pthread_join(th, NULL);
    free(ref); free(out);
    return bad;
}

int main(void) {
#ifdef _OPENMP
    omp_set_num_threads(4);   /* a team of every core fights the loader's thread for them */
#endif
    char path[] = "/tmp/qwen36-gguf-hold-XXXXXX";
    int fd = mkstemp(path); if (fd < 0) { perror("mkstemp"); return 1; }
    close(fd);
    write_gguf(path);
    setup(path);
    float *x = falloc((int64_t)WIDE * D);
    for (int i = 0; i < WIDE * D; i++) x[i] = sinf(0.37f * (float)i);
    const int one = differing(1, x), rows = differing(ROWS, x), wide = differing(WIDE, x);
    int busy = 0;
    for (int i = 0; i < M.cache[0].n; i++) busy += slot_busy(&M.cache[0].slots[i]);
    gguf_close(&g_gg); unlink(path);
    if (one || rows || wide || busy) {
        fprintf(stderr, "FAIL: with the PILOT load running, %d of %d decode steps, %d of %d %d-row steps and %d of %d %d-row steps differ; %d slots still held\n",
                one, STEPS, rows, STEPS, ROWS, wide, STEPS, WIDE, busy);
        return 1;
    }
    puts("qwen36 GGUF expert hold OK");
    return 0;
}
