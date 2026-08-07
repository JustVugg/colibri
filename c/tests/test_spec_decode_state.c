/* Stateful spec_decode callers use kv_out as both the live history length and
 * the .coli_kv record count.  At an NGEN cap the last emitted token therefore
 * has to take the normal forward path before kv_out advances.  Otherwise MORE
 * starts from the preceding token and persistence silently omits the visible
 * tail of the answer.
 *
 * A zero-layer toy model keeps this test dependency-free.  Its embedding and
 * lm_head implement an explicit token->next-token table, while m.n_fw proves
 * the final state was actually forwarded rather than only counted. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

enum { V = 8, D = 8 };

typedef struct {
    Model m;
    KVState kv;
} Toy;

static void reset_decode_globals(int draft) {
    g_temp = 0;
    g_nuc = 1;
    g_draft = draft;
    g_nstop = 0;
    g_intr = 0;
    g_repin = 0;
    g_ram_budget_gb = 0;
    memset(&g_grd, 0, sizeof(g_grd));
}

static void toy_init(Toy *t, const int next[V]) {
    memset(t, 0, sizeof(*t));
    t->m.c.hidden = D;
    t->m.c.vocab = V;
    t->m.c.n_layers = 0;
    t->m.c.kv_lora = 1;
    t->m.c.qk_rope = 1;
    t->m.c.eps = 1e-5f;
    t->m.max_t = 64;
    t->m.kv = &t->kv;

    t->m.embed.fmt = 0;
    t->m.embed.O = V;
    t->m.embed.I = D;
    t->m.embed.qf = calloc((size_t)V * D, sizeof(float));
    t->m.lm_head.fmt = 0;
    t->m.lm_head.O = V;
    t->m.lm_head.I = D;
    t->m.lm_head.qf = calloc((size_t)V * D, sizeof(float));
    t->m.final_norm = calloc(D, sizeof(float));
    if (!t->m.embed.qf || !t->m.lm_head.qf || !t->m.final_norm) {
        fprintf(stderr, "OOM building spec_decode toy\n");
        exit(1);
    }
    for (int i = 0; i < V; i++) {
        t->m.embed.qf[(int64_t)i * D + i] = 1;
        t->m.final_norm[i] = 1;
        t->m.lm_head.qf[(int64_t)next[i] * D + i] = 1;
    }
}

static void toy_free(Toy *t) {
    if (t->kv.disk_fp) fclose(t->kv.disk_fp);
    free(t->kv.disk_buf);
    free(t->m.embed.qf);
    free(t->m.lm_head.qf);
    free(t->m.final_norm);
}

static float *choose(int token) {
    float *lo = falloc(V);
    for (int i = 0; i < V; i++) lo[i] = 0;
    lo[token] = 1;
    return lo;
}

static int test_ngen_zero_and_one_shot(void) {
    const int next[V] = {1, 2, 3, 4, 5, 6, 7, 0};
    Toy t;
    toy_init(&t, next);
    reset_decode_globals(0);

    int all[8] = {0, 1};
    int emitted[4] = {-1, -1, -1, -1};
    EmitStore out = {emitted, 0};
    int kv = -1;
    CHECK(spec_decode(&t.m, all, 2, 0, -1, choose(2), emit_store, &out, &kv, NULL) == 0);
    CHECK(out.n == 0 && kv == 2 && t.m.n_fw == 0);

    /* kv_out=NULL is the one-shot path: keep its saved terminal forward. */
    CHECK(spec_decode(&t.m, all, 2, 1, -1, choose(2), emit_store, &out, NULL, NULL) == 1);
    CHECK(out.n == 1 && emitted[0] == 2 && all[2] == 2);
    CHECK(t.m.n_fw == 0);

    toy_free(&t);
    return 0;
}

static int test_stateful_ngen_and_more(void) {
    const int next[V] = {1, 2, 3, 4, 5, 6, 7, 0};
    Toy t;
    toy_init(&t, next);
    reset_decode_globals(0);

    int all[12] = {0, 1};
    int emitted[4] = {-1, -1, -1, -1};
    EmitStore out = {emitted, 0};
    int kv = -1;
    CHECK(spec_decode(&t.m, all, 2, 1, -1, choose(2), emit_store, &out, &kv, NULL) == 1);
    CHECK(out.n == 1 && emitted[0] == 2 && all[2] == 2);
    CHECK(kv == 3 && all[kv - 1] == emitted[0]);       /* MORE seed */
    CHECK(t.m.n_fw == 1);                              /* committed, not just counted */

    /* Mirror run_serve's MORE path: regenerate logits from the committed tail. */
    float *lo = step(&t.m, all + kv - 1, 1, kv - 1);
    CHECK(spec_decode(&t.m, all, kv, 1, -1, lo, emit_store, &out, &kv, NULL) == 1);
    CHECK(out.n == 2 && emitted[1] == 3 && all[3] == 3);
    CHECK(kv == 4 && all[kv - 1] == emitted[1]);
    CHECK(t.m.n_fw == 2);

    toy_free(&t);
    return 0;
}

static int test_persisted_tail(void) {
    const int next[V] = {1, 2, 3, 4, 5, 6, 7, 0};
    const char *path = "test_spec_decode_state.coli_kv";
    Toy t;
    toy_init(&t, next);
    reset_decode_globals(0);
    remove(path);
    snprintf(t.kv.disk_path, sizeof(t.kv.disk_path), "%s", path);

    int all[8] = {0, 1};
    int emitted[2] = {-1, -1};
    EmitStore out = {emitted, 0};
    int kv = -1;
    CHECK(spec_decode(&t.m, all, 2, 1, -1, choose(2), emit_store, &out, &kv, NULL) == 1);
    kv_disk_append(&t.m, all, kv);
    if (t.kv.disk_fp) {
        fclose(t.kv.disk_fp);
        t.kv.disk_fp = NULL;
    }

    KVState loaded_kv = {0};
    int loaded[16] = {-1};
    snprintf(loaded_kv.disk_path, sizeof(loaded_kv.disk_path), "%s", path);
    t.m.kv = &loaded_kv;
    CHECK(kv_disk_load(&t.m, loaded, 16) == 3);
    CHECK(loaded[0] == 0 && loaded[1] == 1 && loaded[2] == 2);

    remove(path);
    toy_free(&t);
    return 0;
}

static int test_accepted_speculative_tokens(void) {
    const int next[V] = {0, 2, 3, 1, 4, 5, 6, 7};
    Toy t;
    toy_init(&t, next);
    reset_decode_globals(2);

    /* Appending 2 repeats the earlier [1,2] bigram, proposing [3,1]. */
    int all[12] = {1, 2, 3, 1};
    int emitted[4] = {-1, -1, -1, -1};
    EmitStore out = {emitted, 0};
    int kv = -1;
    CHECK(spec_decode(&t.m, all, 4, 3, -1, choose(2), emit_store, &out, &kv, NULL) == 3);
    CHECK(out.n == 3);
    CHECK(emitted[0] == 2 && emitted[1] == 3 && emitted[2] == 1);
    CHECK(kv == 7);
    CHECK(t.m.n_fw == 1);  /* one batch committed next + both accepted drafts */

    toy_free(&t);
    return 0;
}

static int test_eos_and_non_limit_stop(void) {
    const int eos = 7;
    int next[V] = {1, 2, 3, 4, 5, 6, 7, 0};
    Toy t;
    toy_init(&t, next);
    reset_decode_globals(0);

    int all[12] = {0, 1};
    int emitted[4] = {-1, -1, -1, -1};
    EmitStore out = {emitted, 0};
    int kv = -1;
    CHECK(spec_decode(&t.m, all, 2, 3, eos, choose(eos), emit_store, &out, &kv, NULL) == 0);
    CHECK(out.n == 0 && kv == 2 && t.m.n_fw == 0);
    toy_free(&t);

    /* Emit token 2, forward it, then stop normally when its logits choose EOS. */
    next[2] = eos;
    toy_init(&t, next);
    reset_decode_globals(0);
    memset(emitted, -1, sizeof(emitted));
    out.dst = emitted;
    out.n = 0;
    kv = -1;
    CHECK(spec_decode(&t.m, all, 2, 3, eos, choose(2), emit_store, &out, &kv, NULL) == 1);
    CHECK(out.n == 1 && emitted[0] == 2 && kv == 3);
    CHECK(t.m.n_fw == 1);
    toy_free(&t);
    return 0;
}

int main(void) {
    CHECK(test_ngen_zero_and_one_shot() == 0);
    CHECK(test_stateful_ngen_and_more() == 0);
    CHECK(test_persisted_tail() == 0);
    CHECK(test_accepted_speculative_tokens() == 0);
    CHECK(test_eos_and_non_limit_stop() == 0);
    puts("spec_decode state tests: ok");
    return 0;
}
