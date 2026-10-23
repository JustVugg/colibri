/* A failed deeper verify must release its unfinished rollback slot, preserve
 * complete slots, and permit a later retry. Inject every allocation failure
 * while growing from one slot to three, including the PLE copy. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

static void *live[128];
static int nlive, track, fail_at = -1, attempts;

static int refuse(void) { return track && attempts++ == fail_at; }
static void *remember(void *p) {
    if (track && p) {
        if (nlive == 128) abort();
        live[nlive++] = p;
    }
    return p;
}
static void *spec_malloc(size_t n) { return refuse() ? NULL : remember(malloc(n)); }
static void *spec_calloc(size_t n, size_t s) { return refuse() ? NULL : remember(calloc(n, s)); }
static void spec_free(void *p) {
    for (int i = 0; i < nlive; i++) if (live[i] == p) { live[i] = live[--nlive]; break; }
    free(p);
}

#define malloc spec_malloc
#define calloc spec_calloc
#define free spec_free
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#undef malloc
#undef calloc
#undef free

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

int main(void) {
    const int per_slot = 7; /* two pointer tables, two recurrent layers, PLE */
    for (int failure = 0; failure < 2 * per_slot; failure++) {
        Model m = {0};
        m.c.layers = 3;
        m.c.is_attn = calloc(3, sizeof(*m.c.is_attn));
        m.c.is_attn[1] = 1;
        m.c.dn_vheads = m.c.dn_kdim = m.c.dn_vdim = m.c.dn_conv_dim = 2;
        m.c.dn_convk = m.c.ple_convk = m.c.hc_width = m.c.ngram_size = 2;
        m.PLE_conv_state = calloc(4, sizeof(float));
        track = 1; fail_at = -1; attempts = 0;
        CHECK(q38_spec_alloc(&m, 1));
        CHECK(nlive == per_slot);
        float *first = m.snap_rec[0][0];
        first[0] = 19.0f;
        fail_at = failure; attempts = 0;
        CHECK(!q38_spec_alloc(&m, 3));
        CHECK(m.snap_slots == 1 + failure / per_slot);
        CHECK(nlive == m.snap_slots * per_slot);
        CHECK(!m.snap_rec[m.snap_slots] && !m.snap_conv[m.snap_slots] && !m.snap_ple[m.snap_slots]);
        CHECK(m.snap_rec[0][0] == first && first[0] == 19.0f);
        fail_at = -1; attempts = 0;
        CHECK(q38_spec_alloc(&m, 3));
        CHECK(m.snap_slots == 3 && nlive == 3 * per_slot);
        CHECK(m.snap_rec[0][0] == first && first[0] == 19.0f);
        track = 0;
        q38_model_free(&m);
        CHECK(nlive == 0);
    }
    puts("qwen38 speculative allocation: 14 failures release unfinished slots, preserve state and retry cleanly");
    return 0;
}
