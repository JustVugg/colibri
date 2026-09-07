/* The warmstart must not free the int8 weights the tier still points at (#1341).
 *
 * The defect lives in the engine, not the tier: the warmstart hands the tier
 * the live RAM weights -- `wg = expert_is_int4 ? e->g4 : (const uint8_t *)e->g`
 * -- and qt_note_planned parks that pointer in the tier's slot. The very next
 * statement used to free it unconditionally:
 *
 *     if (!keep8 && e->g) { free(e->g); e->g = e->u = e->d = NULL; }
 *
 * with a comment claiming an LFRU eviction "rematerializes from g4". That is
 * true only for an int4 container. On an int8 container e->g4 is NULL: there is
 * no second copy, slot_ensure_int8() returns early (`if (s->g || !s->g4)
 * return;`), and the CPU fallback in the decode loop dereferences a NULL e->g.
 * The tier's stored pointer is dangling on top of that, so any later stage()
 * reads freed memory.
 *
 * Proving that needs the ENGINE, not just the tier -- the two earlier tier
 * tests would pass with the engine freeing everything underneath them. So this
 * one includes qwen36.c the way tests/test_qwen36_ctx.c does, builds an
 * in-memory int8 model with NO container behind it (every slot pre-populated
 * and indexed, so expert_get takes its hit path and never reads a file), and
 * drives tier_warmstart() directly against the fake CUDA backend shared with
 * the other tier tests.
 *
 * Include order matters: qwen36.c first (it never references coli_cuda_*),
 * then the fake backend, then qwen36_tier.c -- which puts the tier's own
 * statics (qs(), G) in this TU, so the test can read back the exact pointer
 * the tier kept. No symbol collides between the two sources.
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

/* compat.h (pulled in by qwen36.c) maps setenv/unsetenv to
 * SetEnvironmentVariableA, which updates the Win32 environment block -- but
 * getenv() reads the CRT's own copy and never sees it, so the tier would
 * silently not get COLI_CUDA here. Drop that mapping before the fake backend
 * installs its _putenv_s version, and unset the same way. Same reasoning as
 * tests/test_qwen36_ctx.c. */
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
enum { NG = EXP_IH * EXP_D, ND = EXP_D * EXP_IH };   /* gate/up and down elements */

/* Byte the setup writes at element i of matrix `which` (0=g,1=u,2=d) of expert
 * eid. Kept in int4 range so the same generator can feed the packed model. */
static int8_t want_byte(int eid, int which, int64_t i) {
    return (int8_t)(((int)i + eid * 3 + which * 5) % 15 - 7);
}

/* An in-memory model with one layer of int8 experts and no container: cap ==
 * n_experts (what qt_init demands) and every slot published in the layer's
 * index, so expert_get() hits and load_expert_merged() is never reached. */
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
        slot_ensure_allocated(m, s);      /* the engine's own g|u|d block */
        s->eid = e;
        s->pinned = 1;                    /* nothing may evict during the test */
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

/* Same weights, but stored the way an int4 container stores them: packed
 * g4|u4|d4 as the source of truth plus the unpacked int8 copy, exactly as
 * load_expert_merged() leaves a slot when the tier is running. */
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

/* qt_fill_wait() returns when the queue is empty, and the uploader dequeues
 * BEFORE uploading -- so wait on the observable end state instead. */
static int wait_resident(void) {
    for (int i = 0; i < 500; i++) {
        int all = 1;
        for (int e = 0; e < NE; e++) if (!qt_is_resident(0, e)) all = 0;
        if (all) return 1;
        struct timespec ts = {0, 2000000};          /* 2 ms */
        nanosleep(&ts, NULL);
    }
    return 0;
}

/* Every weight the warmstart touched must still read back as written: an int8
 * expert has no second copy to rebuild it from. */
static int bytes_intact(Slot *s, int eid) {
    for (int64_t i = 0; i < NG; i++)
        if (s->g[i] != want_byte(eid, 0, i) || s->u[i] != want_byte(eid, 1, i)) return 0;
    for (int64_t i = 0; i < ND; i++)
        if (s->d[i] != want_byte(eid, 2, i)) return 0;
    return 1;
}

/* --- an int8 container: the warmstart must leave the weights computable ---- */
static void case_int8(void) {
    printf("int8 container\n");
    Model m; build_model(&m); fill_int8(&m);

    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    unsetenv("QT_NO_WARMSTART");
    unsetenv("COLI_KEEP_INT8");
    fake_uploads = 0;

    if (!qt_init(NL, NE, EXP_D, EXP_IH, NE, TOPK, 0 /* per-row */, 0 /* int8 */)) {
        printf("  FAIL the tier refuses a per-row int8 container\n");
        fails++; free_model(&m); return;
    }
    tier_warmstart(&m, 0);
    qt_fill_wait();

    ck(wait_resident(), "every planned int8 expert reached VRAM");
    ck(fake_uploads == 3 * NE, "three uploads per planned expert (gate, up, down)");

    int computable = 1, intact = 1, tier_live = 1;
    for (int e = 0; e < NE; e++) {
        Slot *s; expert_get(&m, 0, e, &s);
        slot_ensure_int8(&m, s);          /* the decode loop's CPU fallback */
        if (!s->g || !s->u || !s->d) { computable = 0; intact = 0; }
        else if (!bytes_intact(s, e)) intact = 0;
        if (qs(0, e)->g4 != (const uint8_t *)s->g) tier_live = 0;
    }
    /* outcome: planned_int8_experts_stay_computable_on_cpu */
    ck(computable, "a planned int8 expert still has weights for the CPU fallback");
    ck(intact, "those weights are the bytes the loader wrote");
    /* outcome: tier_pointer_still_targets_live_weights */
    ck(tier_live, "the pointer the tier kept still targets the live int8 block");

    qt_shutdown();
    free_model(&m);
}

/* --- an int4 container: the RSS optimisation must survive the fix ---------- */
static void case_int4(void) {
    printf("int4 container\n");
    Model m; build_model(&m); fill_int4(&m);

    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    unsetenv("QT_NO_WARMSTART");
    unsetenv("COLI_KEEP_INT8");
    fake_uploads = 0;

    if (!qt_init(NL, NE, EXP_D, EXP_IH, NE, TOPK, 0 /* per-row */, 1 /* int4 */)) {
        printf("  FAIL regression: the tier no longer starts on an int4 container\n");
        fails++; free_model(&m); return;
    }
    tier_warmstart(&m, 1);
    qt_fill_wait();
    ck(wait_resident(), "every planned int4 expert reached VRAM");

    int freed = 1, remat = 1;
    for (int e = 0; e < NE; e++) {
        Slot *s; expert_get(&m, 0, e, &s);
        if (s->g) freed = 0;
        slot_ensure_int8(&m, s);
        if (!s->g || !bytes_intact(s, e)) remat = 0;
    }
    /* outcome: int4_path_still_frees_its_second_copy */
    ck(freed, "int4 still drops the int8 copy right after staging (peak RSS)");
    ck(remat, "slot_ensure_int8 rebuilds it from the packed g4/u4/d4");

    qt_shutdown();
    free_model(&m);
}

int main(void) {
    case_int8();
    case_int4();
    if (fails) { printf("FAILED %d\n", fails); return 1; }
    printf("OK test_qwen36_tier_int8_engine\n");
    return 0;
}
