/* test_kv_ring_grow — kv_alloc's grow-with-copy over ring-buffered sliding layers.
 *
 * The collision this guards (#830 x #786): grow-with-copy re-lays-out K/V
 * linearly, but a sliding layer's ring stores position t at row t % window.
 * Copy a wrapped ring linearly and the cache comes back silently rotated:
 * nothing crashes, attention just reads the wrong rows once prefix reuse
 * engages past the window. So kv_alloc must steal ring buffers whose row
 * count is unchanged, and only linear-copy layers whose row count grows.
 *
 * Three cases, each asserting every surviving position is at the slot
 * attention will read it from (t % rows):
 *   A. wrapped ring + global grow  — ring stolen intact, global re-laid-out
 *   B. window >= max_t transition  — linear sliding buffer grows INTO a ring
 *   C. keep == 0                   — grow with nothing to preserve
 */
#define main coli_inkling_main_unused
#include "../inkling.c"
#undef main

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)

/* one distinctive value per (layer, head, position, dim) */
static float val(int li, int h, int t, int d) {
    return (float)(li*100000 + h*10000 + t*10 + d + 1);
}

/* replicate attention's append: position t lands at slot t % rows */
static void sim_append(Model *m, int li, int t) {
    Cfg *c = &m->c;
    int kv = L_KV(c,li), hd = L_HD(c,li);
    int rows = kv_ring_rows(c, li, m->max_t);
    for (int h = 0; h < kv; h++) for (int d = 0; d < hd; d++) {
        m->K[li][((int64_t)h*rows + t % rows)*hd + d] = val(li,h,t,d);
        m->V[li][((int64_t)h*rows + t % rows)*hd + d] = -val(li,h,t,d);
    }
}

/* assert position t is readable at slot t % rows, both K and V */
static void expect(Model *m, int li, int t) {
    Cfg *c = &m->c;
    int kv = L_KV(c,li), hd = L_HD(c,li);
    int rows = kv_ring_rows(c, li, m->max_t);
    for (int h = 0; h < kv; h++) for (int d = 0; d < hd; d++) {
        float got = m->K[li][((int64_t)h*rows + t % rows)*hd + d];
        CHECK(got == val(li,h,t,d), "K L%d h%d t%d d%d: got %.0f want %.0f (rows=%d)",
              li, h, t, d, got, val(li,h,t,d), rows);
        got = m->V[li][((int64_t)h*rows + t % rows)*hd + d];
        CHECK(got == -val(li,h,t,d), "V L%d h%d t%d d%d: got %.0f want %.0f (rows=%d)",
              li, h, t, d, got, -val(li,h,t,d), rows);
    }
}

static void model_setup(Model *m, int window) {
    memset(m, 0, sizeof(*m));
    Cfg *c = &m->c;
    c->n_layers = 2;
    c->local[0] = 1;                 /* L0 sliding, L1 global */
    c->window = window;
    c->swa_kv = 2; c->swa_hd = 3;    /* sliding: 2 KV heads x 3 dims */
    c->n_kv   = 1; c->head_dim = 3;  /* global:  1 KV head  x 3 dims */
}

static void model_teardown(Model *m) {
    for (int i = 0; i < m->c.n_layers; i++) { free(m->K[i]); free(m->V[i]); }
    free(m->K); free(m->V);
    kv_prefix_free(&m->kvp);
}

int main(void) {
    Model m;

    /* A. wrapped ring survives a grow untouched; global layer re-lays-out.
     * window=4, alloc 6: L0 is a 4-row ring, positions 0..5 wrap twice. */
    model_setup(&m, 4);
    kv_alloc(&m, 6);
    CHECK(kv_ring_rows(&m.c, 0, m.max_t) == 4, "L0 rows want 4");
    CHECK(kv_ring_rows(&m.c, 1, m.max_t) == 6, "L1 rows want 6");
    for (int t = 0; t < 6; t++) { sim_append(&m, 0, t); sim_append(&m, 1, t); }
    m.kv_len = 6;
    float *ring_before = m.K[0];
    kv_alloc(&m, 12);
    CHECK(m.K[0] == ring_before, "L0 ring must be stolen, not copied");
    CHECK(kv_ring_rows(&m.c, 1, m.max_t) == 12, "L1 rows want 12 after grow");
    CHECK(m.kv_len == 6, "kv_len must survive the grow, got %d", m.kv_len);
    for (int t = 2; t < 6; t++) expect(&m, 0, t);   /* last window=4 positions */
    for (int t = 0; t < 6; t++) expect(&m, 1, t);   /* all kept, new stride   */
    model_teardown(&m);

    /* B. transition: window >= max_t (linear) grows past the window (ring).
     * window=8, alloc 6: L0 rows=6, linear. Grow to 20: L0 rows=8, copy. */
    model_setup(&m, 8);
    kv_alloc(&m, 6);
    CHECK(kv_ring_rows(&m.c, 0, m.max_t) == 6, "L0 rows want 6 (window >= max_t)");
    for (int t = 0; t < 6; t++) { sim_append(&m, 0, t); sim_append(&m, 1, t); }
    m.kv_len = 6;
    kv_alloc(&m, 20);
    CHECK(kv_ring_rows(&m.c, 0, m.max_t) == 8, "L0 rows want 8 after grow");
    for (int t = 0; t < 6; t++) expect(&m, 0, t);   /* linear copy into the ring */
    for (int t = 0; t < 6; t++) expect(&m, 1, t);
    model_teardown(&m);

    /* C. keep == 0: grow allocates, copies nothing, crashes on nothing. */
    model_setup(&m, 4);
    kv_alloc(&m, 6);
    m.kv_len = 0;
    kv_alloc(&m, 12);
    CHECK(m.kv_len == 0, "kv_len want 0, got %d", m.kv_len);
    sim_append(&m, 0, 0); expect(&m, 0, 0);          /* buffers usable */
    model_teardown(&m);

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("test_kv_ring_grow: OK (ring stolen, linear re-laid-out, transition copied)\n");
    return 0;
}
