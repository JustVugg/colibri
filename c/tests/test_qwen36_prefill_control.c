/* Deterministic interruption of the real hybrid forward, using the same tiny
 * fixture as the oracle job. A wall-clock CANCEL can arrive before layer zero
 * and miss the partially advanced DeltaNet state that this test exercises. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int failures;
#define CHECK(ok, msg) do { if (!(ok)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, msg); failures++; } } while (0)

typedef struct { int calls, at, result; } Poll;
static int poll_at_boundary(void *ctx) {
    Poll *p = ctx;
    return ++p->calls == p->at ? p->result : 0;
}

static void fresh_request(Model *m) {
    reset_recurrent(m); m->kv_len = 0;
    pin_drop(); m->first_step = 1; m->resident_collecting = 0;
    memset(m->seen, 0, (size_t)m->c.n_layers * m->c.n_experts);
    memset(m->momentum_logits, 0, (size_t)m->c.n_layers * m->c.n_experts * sizeof(float));
}

static int all_zero(const float *v, size_t n) {
    for (size_t i = 0; i < n; i++) if (v[i] != 0.f) return 0;
    return 1;
}

static void check_cleared(Model *m) {
    CHECK(m->kv_len == 0 && m->kvp.len == 0, "partial prefix must be invalidated");
    CHECK(!g_pin_logit && !g_pin_use_logit, "pin pointers must be cleared");
    for (int i = 0; i < g_pins.n; i++)
        CHECK(g_pins.slot[i].len == 0, "a saved pin cannot survive an invalid prefix record");
    CHECK(m->first_step && !m->resident_collecting, "abort must not finalize resident collection");
    Cfg *c = &m->c;
    for (int i = 0; i < c->n_layers; i++) if (!c->is_attn[i]) {
        CHECK(all_zero(m->DN_rec[i], (size_t)c->dn_vheads*c->dn_kdim*c->dn_vdim),
              "partial DeltaNet recurrence survived");
        CHECK(all_zero(m->DN_conv[i], (size_t)c->dn_conv_dim*(c->dn_convk-1)),
              "partial DeltaNet convolution survived");
    }
}

static void continuation(Model *m, const int *ids, int n, float *out) {
    float *lo = step(m, ids, n, 0);
    for (int t = 0; t < 8; t++) {
        memcpy(out + (size_t)t*m->c.vocab, lo, m->c.vocab*sizeof(float));
        int id = 0;
        for (int v = 1; v < m->c.vocab; v++) if (lo[v] > lo[id]) id = v;
        free(lo);
        lo = t < 7 ? step(m, &id, 1, n + t) : NULL;
    }
}

static void interrupted(Model *m, const int *ids, int n, const float *reference,
                        int boundary, int prefix, int result) {
    fresh_request(m);
    if (prefix) {
        float *lo = step(m, ids, prefix, 0);
        pin_save(m, ids, prefix, lo);
        free(lo);
        CHECK(pin_restore(m, ids, n) == prefix, "test must begin with a real restored pin");
        g_pin_use_logit = 1;
        CHECK(kv_prefix_reuse(&m->kvp, ids, n) == prefix, "test must reuse a real prefix");
    }
    int tokens = m->token_count;
    Poll poll = {0, boundary + 1, result};
    Q36StepControl control = {poll_at_boundary, &poll, 0};
    m->resident_mode = 1; m->first_step = 1;
    float *lo = step_ex_control(m, ids + prefix, n - prefix, prefix, 1, &control);
    CHECK(lo == NULL && control.result == result, "interruption must propagate without logits");
    free(lo);
    CHECK(poll.calls == boundary + 1, "must stop at the requested layer boundary");
    CHECK(m->token_count == tokens, "partial work must not commit prompt tokens");
    check_cleared(m);
    CHECK(!kv_prefix_reuse(&m->kvp, ids, n), "retry must not reuse interrupted state");
    m->resident_mode = 0;
    /* Do not reset here: recovery must rely on the abort cleanup itself. */
    float *got = falloc((int64_t)8*m->c.vocab);
    continuation(m, ids, n, got);
    CHECK(!memcmp(got, reference, (size_t)8*m->c.vocab*sizeof(float)),
          "retry and seven decode steps must match cold logits byte for byte");
    free(got);
}

int main(int argc, char **argv) {
    const char *snap = getenv("QWEN36_TINY");
    if (!snap || !*snap) { puts("SKIP test_qwen36_prefill_control: set QWEN36_TINY"); return 0; }
    int cap = argc > 1 ? atoi(argv[1]) : 1;
    static Model m;
    model_init(&m, snap, cap, 8);
    g_expert_gs = m.c.expert_gs;
    g_expert_is_int4 = container_layer_is_int4(&m, 0);
    m.max_t = 1100; ensure_kv(&m);
    int ids[1008];
    for (int i = 0; i < 1008; i++) ids[i] = (i*17 + 11) % m.c.vocab;
    float *reference = falloc((int64_t)8*m.c.vocab);
    fresh_request(&m); continuation(&m, ids, 64, reference);
    /* Every boundary includes DeltaNet, attention, before-first and after-last.
     * A saved prefix exercises the corruption a cold-only test cannot catch. */
    for (int prefix = 0; prefix <= 8; prefix += 8)
        for (int b = 0; b <= m.c.n_layers; b++)
            for (int result = Q36_CONTROL_STOP; result <= Q36_CONTROL_CANCEL; result++)
                interrupted(&m, ids, 64, reference, b, prefix, result);
    free(reference);
    /* Same full-prompt forward with and without the callback. Compare ALL
     * logits, not just argmax, and ensure each layer has only one polling pass. */
    const int lengths[] = {1, 7, 1000};
    for (int r = 0; r < 3; r++) {
        int rows = lengths[r];
        for (int prefix = 0; prefix <= 8; prefix += 8) {
            fresh_request(&m);
            if (prefix) free(step(&m, ids, prefix, 0));
            float *expected = step(&m, ids + prefix, rows, prefix);
            fresh_request(&m);
            if (prefix) free(step(&m, ids, prefix, 0));
            Poll poll = {0, -1, Q36_CONTROL_CANCEL};
            Q36StepControl control = {poll_at_boundary, &poll, 0};
            float *got = step_ex_control(&m, ids + prefix, rows, prefix, 1, &control);
            CHECK(control.result == 0 && poll.calls == m.c.n_layers + 1,
                  "uninterrupted prefill must traverse layers only once");
            CHECK(got && !memcmp(expected, got, m.c.vocab*sizeof(float)),
                  "polling must not change full-prompt logits");
            CHECK(m.kv_len == rows + prefix && m.kvp.len == rows + prefix,
                  "completed prefix must remain reusable");
            free(expected); free(got);
        }
    }
    pin_drop();
    printf("qwen36 prefill control: cap=%d, %d failures\n", cap, failures);
    return failures ? 1 : 0;
}
