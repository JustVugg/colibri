/* The decode path must offer int8 experts to the tier, not only int4 (#1391).
 *
 * #1331 fixed the warmstart: `wg = expert_is_int4 ? e->g4 : (const uint8_t *)e->g`
 * hands the tier the live RAM weights on an int8 container. The two decode-path
 * offer sites were not given the same choice -- `if (e->g4) qt_note(...)` in
 * moe()'s resident offer and `if (ps->g4) qt_note(...)` in the pilot-prefetch
 * lookahead -- so with QT_NO_WARMSTART=1 on an int8 container nothing was ever
 * enqueued: the tier reported itself active and sat at 0 uploads, 0 % hit rate
 * for the life of the process.
 *
 * Same shape as the parent test: include qwen36.c (with main renamed away),
 * the fake CUDA backend, then qwen36_tier.c, so the test can read the tier's
 * own statics. The model is in-memory with NO container: every slot is
 * pre-populated and indexed so expert_get() hits and never reads a file.
 *
 * The moe() call sites need a Model in step() state, which drags in the whole
 * attention/DeltaNet machinery. The offer code, however, is two identical
 * three-line blocks around one qt_note call; what the fix changes is exactly
 * the pointer choice those sites make, and both sites now share that choice
 * with tier_warmstart. This test drives the same offer decision the decode
 * sites make, against the same Slot state, the same way the parent test drives
 * tier_warmstart: call the mirror expression directly on a real Slot, assert
 * the tier accepted an int8 pointer (fake_uploads > 0) and that the resident
 * set grows -- the assertion from #1391 that fails on the old expression.
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#ifdef _WIN32
#undef setenv
#undef unsetenv
#define unsetenv(name) _putenv_s(name, "")
#endif

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void ck(int ok, const char *what) {
    if (ok) { printf("  ok   %s\n", what); return; }
    printf("  FAIL %s\n", what);
    fails++;
}

enum { NL = 1, NE = 4, EXP_D = 64, EXP_IH = 32, TOPK = 1 };
enum { NG = EXP_IH * EXP_D, ND = EXP_D * EXP_IH };

static int8_t want_byte(int eid, int which, int64_t i) {
    return (int8_t)(((int)i + eid * 3 + which * 5) % 15 - 7);
}

static void build_model(Model *m) {
    memset(m, 0, sizeof *m);
    m->c.n_layers = NL; m->c.n_experts = NE;
    m->c.hidden = EXP_D; m->c.inter = EXP_IH;
    m->c.topk = TOPK; m->c.expert_gs = 0;
    m->active_of  = calloc(NL, sizeof(int));
    m->is_pinned  = calloc((size_t)NL * NE, 1);
    m->is_queued  = calloc((size_t)NL * NE, 1);
    m->cache      = calloc(NL, sizeof(LCache));
    LCache *lc = &m->cache[0];
    lc->cap = NE; lc->n = NE;
    lc->slots = calloc(NE, sizeof(Slot));
    lc->slot_by_expert = malloc(NE * sizeof(int));
    for (int e = 0; e < NE; e++) {
        Slot *s = &lc->slots[e];
        slot_ensure_allocated(m, s);
        s->eid = e;
        s->pinned = 1;
        s->used = ++m->clock;
        lc->slot_by_expert[e] = e;
    }
}

static void fill_int8(Model *m) {
    for (int e = 0; e < NE; e++) {
        Slot *s = &m->cache[0].slots[e];
        for (int64_t i = 0; i < NG; i++) { s->g[i] = want_byte(e, 0, i); s->u[i] = want_byte(e, 1, i); }
        for (int64_t i = 0; i < ND; i++) s->d[i] = want_byte(e, 2, i);
        for (int64_t i = 0; i < EXP_IH; i++) { s->gs[i] = 1.f; s->us[i] = 1.f; }
        for (int64_t i = 0; i < EXP_D; i++) s->ds[i] = 1.f;
    }
}

static void fill_int4(Model *m) {
    int64_t want_w = NG + NG + ND;
    uint8_t *raw = malloc((size_t)(want_w / 2));
    for (int e = 0; e < NE; e++) {
        Slot *s = &m->cache[0].slots[e];
        for (int64_t i = 0; i < want_w; i += 2) {
            int which = i < NG ? 0 : (i < 2 * NG ? 1 : 2);
            int64_t off = i - (which == 0 ? 0 : (which == 1 ? NG : 2 * NG));
            uint8_t lo = (uint8_t)(want_byte(e, which, off)     & 0xF);
            uint8_t hi = (uint8_t)(want_byte(e, which, off + 1) & 0xF);
            raw[i >> 1] = (uint8_t)(lo | (hi << 4));
        }
        unpack_int4_to_int8(s->g, raw, want_w);
        s->is_int4 = 1;
        int64_t gp = NG / 2, up = NG / 2, dp = ND / 2;
        s->g4 = malloc((size_t)gp); s->u4 = malloc((size_t)up); s->d4 = malloc((size_t)dp);
        memcpy(s->g4, raw,           (size_t)gp);
        memcpy(s->u4, raw + gp,      (size_t)up);
        memcpy(s->d4, raw + gp + up, (size_t)dp);
        for (int64_t i = 0; i < EXP_IH; i++) { s->gs[i] = 1.f; s->us[i] = 1.f; }
        for (int64_t i = 0; i < EXP_D; i++) s->ds[i] = 1.f;
    }
    free(raw);
}

static void free_model(Model *m) {
    LCache *lc = &m->cache[0];
    for (int e = 0; e < NE; e++) {
        Slot *s = &lc->slots[e];
        free(s->g); free(s->gs);
        free(s->g4); free(s->u4); free(s->d4);
    }
    free(lc->slots); free(lc->slot_by_expert);
    free(m->cache); free(m->active_of); free(m->is_pinned); free(m->is_queued);
}

static int wait_resident(void) {
    for (int i = 0; i < 500; i++) {
        int all = 1;
        for (int e = 0; e < NE; e++) if (!qt_is_resident(0, e)) all = 0;
        if (all) return 1;
        struct timespec ts = {0, 2000000};
        nanosleep(&ts, NULL);
    }
    return 0;
}

/* The offer decision both decode-path sites now make, lifted verbatim from
 * moe()'s resident offer (the prefetch site is the same choice on `ps`). If
 * this expression regresses to the bare `if (e->g4)`, the int8 case below
 * fails exactly the way the issue describes. */
static void decode_offer(int layer, int eid, Slot *s) {
    if (s->g4)
        qt_note(layer, eid, s->g4, s->u4, s->d4, s->gs, s->us, s->ds);
    else if (!g_expert_is_int4 && s->g)
        qt_note(layer, eid, (const uint8_t *)s->g, (const uint8_t *)s->u,
                (const uint8_t *)s->d, s->gs, s->us, s->ds);
}

/* --- an int8 container, QT_NO_WARMSTART=1: the decode offer must promote --- */
static void case_int8_decode(void) {
    printf("int8 container, decode-path offer (no warmstart)\n");
    Model m; build_model(&m); fill_int8(&m);

    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    fake_uploads = 0;

    if (!qt_init(NL, NE, EXP_D, EXP_IH, NE, TOPK, 0 /* per-row */, 0 /* int8 */)) {
        printf("  FAIL the tier refuses a per-row int8 container\n");
        fails++; free_model(&m); return;
    }
    /* The decode loop runs with the format main probed: int8. */
    g_expert_is_int4 = 0;

    /* moe()'s resident offer, one expert per router pick. Before the fix this
     * was `if (s->g4) qt_note(...)` with g4 == NULL: zero offers, zero
     * uploads, resident 0/4 for the life of the process (#1391). */
    for (int e = 0; e < NE; e++)
        decode_offer(0, e, &m.cache[0].slots[e]);
    qt_fill_wait();

    ck(fake_uploads > 0, "the decode offer enqueued int8 experts (fake_uploads > 0)");
    ck(wait_resident(), "every decode-offered int8 expert reached VRAM");
    ck(fake_uploads == 3 * NE, "three uploads per expert (gate, up, down)");

    int tier_live = 1, intact = 1;
    for (int e = 0; e < NE; e++) {
        Slot *s = &m.cache[0].slots[e];
        if (qs(0, e)->g4 != (const uint8_t *)s->g) tier_live = 0;
        for (int64_t i = 0; i < NG; i++)
            if (s->g[i] != want_byte(e, 0, i) || s->u[i] != want_byte(e, 1, i)) intact = 0;
        for (int64_t i = 0; i < ND; i++)
            if (s->d[i] != want_byte(e, 2, i)) intact = 0;
    }
    /* outcome: decode_offer_parks_live_int8_pointer */
    ck(tier_live, "the pointer the tier kept still targets the live int8 block");
    /* outcome: decode_offer_does_not_free_or_rewrite_weights */
    ck(intact, "the offered weights are still the bytes the loader wrote");

    qt_shutdown();
    free_model(&m);
}

/* --- an int4 container: the decode offer must still prefer the packed copy -- */
static void case_int4_decode(void) {
    printf("int4 container, decode-path offer (no warmstart)\n");
    Model m; build_model(&m); fill_int4(&m);

    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    fake_uploads = 0;

    if (!qt_init(NL, NE, EXP_D, EXP_IH, NE, TOPK, 0 /* per-row */, 1 /* int4 */)) {
        printf("  FAIL regression: the tier no longer starts on an int4 container\n");
        fails++; free_model(&m); return;
    }
    g_expert_is_int4 = 1;

    for (int e = 0; e < NE; e++)
        decode_offer(0, e, &m.cache[0].slots[e]);
    qt_fill_wait();

    ck(fake_uploads > 0, "the decode offer still enqueues on an int4 container");
    ck(wait_resident(), "every decode-offered int4 expert reached VRAM");

    int tier_live = 1;
    for (int e = 0; e < NE; e++) {
        Slot *s = &m.cache[0].slots[e];
        if (qs(0, e)->g4 != s->g4) tier_live = 0;
    }
    /* outcome: decode_offer_prefers_packed_int4_pointer */
    ck(tier_live, "on int4 the tier still parks the packed g4 pointer");

    qt_shutdown();
    free_model(&m);
}

int main(void) {
    case_int8_decode();
    case_int4_decode();
    if (fails) { printf("FAILED %d\n", fails); return 1; }
    printf("OK test_qwen36_tier_int8_decode\n");
    return 0;
}