/* The CUDA tier's arena for the fp8 experts (Qwen3.8: the tier's copy mode; COLI_CUDA_EXPERT_ARENA), on
 * the fake backend.
 *   - off: three uploads an expert, a swap overwrites the victim's tensors in place, as before;
 *   - 1: one arena, an expert ONE copy of its slot (gate | up | down | scales) and three views; a swap
 *     drops the victim's views and refills ITS slot; no device allocation after the arena's;
 *   - 2: the same, the staging (swaps too) from the page-locked pool, every buffer back;
 *   - in all three the device holds the same bytes, expert for expert, as the kernels address them;
 *   - every tensor is released exactly once, the arena once;
 *   - the arena refused, the view refused (a backend without it), the pool refused. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../compat.h"   /* setenv, unsetenv */

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { NL = 2, NE = 16, D = 256, IH = 128, CAP = 4, TOPK = 2, R = 3,
       MB = D * IH,                                      /* a matrix, a byte an element: 32768 */
       NSC = ((IH + 127) / 128) * ((D + 127) / 128) };   /* block scales a matrix: 2 */

static unsigned char slab[NL][NE][3 * MB];   /* gate | up | down */
static float scales[NL][NE][3 * NSC];        /* gate | up | down */
static float lut[256];

static void wait_idle(void) {
    for (int i = 0; i < 2000; i++) {
        pthread_mutex_lock(&G.mx);
        int pending = G.qn + G.inflight;
        pthread_mutex_unlock(&G.mx);
        if (!pending) break;
        struct timespec ts = {0, 2000000}; nanosleep(&ts, NULL);
    }
}
static void note(int l, int e) {
    qt_note(l, e, slab[l][e], slab[l][e] + MB, slab[l][e] + 2 * MB,
            scales[l][e], scales[l][e] + NSC, scales[l][e] + 2 * NSC);
    wait_idle();
}
static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static uint64_t src_digest(int l, int e) {
    uint64_t h = fnv(1469598103934665603ull, slab[l][e], 3 * MB);
    return fnv(h, scales[l][e], 3 * NSC * sizeof(float));
}
/* what the device holds for a resident, read through its tensors as a group's descriptor would */
static uint64_t dev_digest(int l, int e) {
    uint64_t h = 0;
    pthread_mutex_lock(&G.mx);
    QSlot *s = qs(l, e);
    if (s->resident && s->tg && s->tu && s->td && s->tg->w && s->tu->w && s->td->w && s->tg->sc && s->tu->sc && s->td->sc) {
        h = 1469598103934665603ull;
        h = fnv(h, s->tg->w, MB); h = fnv(h, s->tu->w, MB); h = fnv(h, s->td->w, MB);
        h = fnv(h, s->tg->sc, NSC * sizeof(float)); h = fnv(h, s->tu->sc, NSC * sizeof(float)); h = fnv(h, s->td->sc, NSC * sizeof(float));
    }
    pthread_mutex_unlock(&G.mx);
    return h;
}
static int shapes_ok(int l, int e) {
    pthread_mutex_lock(&G.mx);
    QSlot *s = qs(l, e);
    int ok = s->tg && s->tu && s->td && s->tg->fmt == 8 && s->tu->fmt == 8 && s->td->fmt == 8 &&
             s->tg->I == D && s->tg->O == IH && s->tu->I == D && s->tu->O == IH && s->td->I == IH && s->td->O == D;
    pthread_mutex_unlock(&G.mx);
    return ok;
}
static int slot_of(int l, int e) {
    pthread_mutex_lock(&G.mx); int a = qs(l, e)->aslot - 1; pthread_mutex_unlock(&G.mx);
    return a;
}
/* the views at the slot's offsets, the slot the engine's bytes, its padding zero */
static int slot_ok(int l, int e) {
    int ok = 0;
    pthread_mutex_lock(&G.mx);
    QSlot *s = qs(l, e);
    const int a = s->aslot - 1;
    if (a >= 0 && a < G_aslots[0] && s->tg && s->tu && s->td) {
        const unsigned char *b = (const unsigned char *)G_arena[0] + (size_t)a * G_aslot;
        const float *bs = (const float *)(b + G_aoff_s);
        const size_t nsb = 3 * NSC * sizeof(float);
        ok = s->tg->w == b && s->tu->w == b + MB && s->td->w == b + 2 * MB &&
             s->tg->sc == bs && s->tu->sc == bs + NSC && s->td->sc == bs + 2 * NSC &&
             !memcmp(b, slab[l][e], 3 * MB) && !memcmp(bs, scales[l][e], nsb);
        for (size_t i = G_aoff_s + nsb; ok && i < G_aslot; i++) if (b[i]) ok = 0;
    }
    pthread_mutex_unlock(&G.mx);
    return ok;
}
static size_t off_bytes(void) { return 3 * dev_alloc_footprint(MB) + 3 * dev_alloc_footprint(NSC * sizeof(float)); }
static size_t slot_bytes(void) { return 3 * (size_t)MB + qt_pad256(3 * NSC * sizeof(float)); }
static void set_budget(size_t per) {   /* an allowance for exactly R experts of `per` bytes */
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(R * per + per / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
}
static void reset_counts(void) {
    fake_uploads = fake_overwrites = fake_pipe_uploads = fake_f8_views = fake_tensor_frees = 0;
    fake_pipe_allocs = fake_pipe_frees = fake_host_allocs = 0; fake_pipe_upload_bytes = 0;
}

#define CHECK(c, what) do { snprintf(m, sizeof m, "%s: %s", mode, what); check((c), m); } while (0)
/* R experts into an empty tier, a cold one refused, a hot one swapped in for the coldest; dig[]: what the
 * device holds for the four experts that became resident */
static void run(const char *mode, int arena, int pool, uint64_t dig[4]) {
    char m[240];
    memset(dig, 0, 4 * sizeof dig[0]);
    if (arena) setenv("COLI_CUDA_EXPERT_ARENA", pool ? "2" : "1", 1); else unsetenv("COLI_CUDA_EXPERT_ARENA");
    set_budget(arena ? slot_bytes() : off_bytes());
    reset_counts();
    if (!qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut)) { CHECK(0, "the tier starts"); return; }
    CHECK(G.wfmt == 8 && qt_streaming(), "format 8, copy mode");
    if (!arena) {
        CHECK(!G_nat_arena && !G_arena_on && !G_arena[0] && !G_stg, "no arena and no pool without the switch");
        CHECK(G.exp_bytes == off_bytes(), "an expert budgeted at its six device allocations, as before");
        CHECK(fake_pipe_allocs == 0 && fake_f8_views == 0 && fake_host_allocs == 0, "nothing allocated, no view asked for");
    } else {
        CHECK(G_nat_arena && G_arena_on && G_arena[0], "the arena is up");
        CHECK(G_aoff_s == 3 * (size_t)MB && G_aslot == slot_bytes(), "a slot is gate | up | down, then the scales, padded to 256");
        CHECK(G.exp_bytes == G_aslot, "an expert is budgeted at its slot");
        CHECK(G_aslots[0] == R && G_anfree[0] == R, "as many slots as the budget holds, all free");
        CHECK(fake_pipe_allocs == 1, "one device allocation: the arena");
        CHECK(fake_f8_views == 1 && fake_tensor_frees == 1, "the backend's view asked for once, and released");
        if (pool) CHECK(G_stg && fake_host_allocs == 1 && G_stg_nfree == QT_STG_N, "the page-locked pool is up, full");
        else CHECK(!G_stg && fake_host_allocs == 0, "no page-locked pool at 1");
    }
    const int v0 = fake_f8_views, f0 = fake_tensor_frees;

    note(0, 5); note(0, 6); note(1, 7);
    CHECK(qt_is_resident(0, 5) && qt_is_resident(0, 6) && qt_is_resident(1, 7), "the budget's experts are resident");
    if (!arena) CHECK(fake_uploads == 3 * R && fake_pipe_uploads == 0, "three uploads an expert, no slot copy");
    else {
        CHECK(fake_uploads == 0, "no upload of a matrix, so no device allocation an expert");
        CHECK(fake_pipe_uploads == R && fake_pipe_upload_bytes == (size_t)R * G_aslot, "ONE copy an expert, of its whole slot");
        CHECK(fake_f8_views == v0 + 3 * R && fake_tensor_frees == f0, "three views an expert, none released");
        const int a = slot_of(0, 5), b = slot_of(0, 6), c = slot_of(1, 7);
        CHECK(a >= 0 && b >= 0 && c >= 0 && a != b && a != c && b != c, "every expert in a slot of its own");
        CHECK(G_anfree[0] == 0 && fake_pipe_allocs == 1, "no free slot left, still one device allocation");
        CHECK(slot_ok(0, 5) && slot_ok(0, 6) && slot_ok(1, 7), "a slot holds the engine's bytes, the views point into it, its padding is zero");
        if (pool) CHECK(G_stg_used == (unsigned long long)R && G_stg_nfree == QT_STG_N, "every expert staged through the pool, every buffer back");
        else CHECK(G_stg_used == 0, "none through a pool at 1");
    }
    CHECK(shapes_ok(0, 5) && shapes_ok(1, 7), "the tensors carry format 8 and the shapes");
    {   /* qt_cold_refresh: qt_note_wanted's bar recomputed without a tick (a prompt pass): the least heat of the
         * residents not queued; with none counted, "none" (the heats and flags are put back after) */
        pthread_mutex_lock(&G.mx);
        QSlot *ra = qs(0, 5), *rb = qs(0, 6), *rc = qs(1, 7);
        const uint32_t ha = ra->heat, hb = rb->heat, hc = rc->heat;
        ra->heat = 7; rb->heat = 3; rc->heat = 9;
        for (int di = 0; di < G.ndev; di++) G_cold_heat[di] = 12345u;
        pthread_mutex_unlock(&G.mx);
        qt_cold_refresh();
        pthread_mutex_lock(&G.mx); uint32_t got = G_cold_heat[0]; rb->queued = 1; pthread_mutex_unlock(&G.mx);
        CHECK(got == 3, "qt_cold_refresh takes the least heat of the residents");
        qt_cold_refresh();
        pthread_mutex_lock(&G.mx); got = G_cold_heat[0]; ra->queued = rc->queued = 1; pthread_mutex_unlock(&G.mx);
        CHECK(got == 7, "a queued resident does not count");
        qt_cold_refresh();
        pthread_mutex_lock(&G.mx); got = G_cold_heat[0];
        ra->queued = rb->queued = rc->queued = 0; ra->heat = ha; rb->heat = hb; rc->heat = hc;
        pthread_mutex_unlock(&G.mx);
        CHECK(got == 0xFFFFFFFFu, "no resident counted: none");
    }
    {   /* qt_prefill_group (Qwen3.8's Q38_PREFILL_GPU): residents of one layer with their rows, one synchronous call */
        CHECK(qt_resident_home(0, 5) == 0 && qt_resident_home(1, 7) == 0 && qt_resident_home(0, 8) == -1,
              "qt_resident_home: a resident's device, -1 for the others");
        const int eids[2] = {5, 6}, rows[2] = {2, 1}, bad[2] = {5, 8};
        static float px[3 * D], py[3 * D];
        const int c0 = fake_group_calls;
        CHECK(qt_prefill_group(0, eids, rows, 2, px, py) == 1 && fake_group_calls == c0 + 1 && fake_group_rows >= 3,
              "two residents' three rows in one call");
        CHECK(py[0] == 0.f && py[(size_t)1 * D] == 1.f && py[(size_t)2 * D] == 1000.f, "the outputs expert by expert, row by row");
        CHECK(qt_prefill_group(0, bad, rows, 2, px, py) == 0 && fake_group_calls == c0 + 1, "an expert not resident: refused, nothing computed");
        pthread_mutex_lock(&G.mx); G.issue_open = 1; pthread_mutex_unlock(&G.mx);
        CHECK(qt_prefill_group(0, eids, rows, 2, px, py) == 0 && fake_group_calls == c0 + 1, "refused while a decode group is open");
        pthread_mutex_lock(&G.mx); G.issue_open = 0; pthread_mutex_unlock(&G.mx);
        fake_group_fail = 1;
        CHECK(qt_prefill_group(0, eids, rows, 2, px, py) == 0, "the backend's refusal is passed on");
        fake_group_fail = 0;
        pthread_mutex_lock(&G.mx); const int open = G.issue_open; pthread_mutex_unlock(&G.mx);
        CHECK(open == 0, "issue_open released after the calls");
    }
    dig[0] = dev_digest(0, 5); dig[1] = dev_digest(0, 6); dig[2] = dev_digest(1, 7);
    CHECK(dig[0] == src_digest(0, 5) && dig[1] == src_digest(0, 6) && dig[2] == src_digest(1, 7), "the device holds the engine's bytes");
    pthread_mutex_lock(&G.mx); size_t used = G.used[0]; pthread_mutex_unlock(&G.mx);
    CHECK(used == (size_t)R * G.exp_bytes, "the budget counts R experts");

    note(0, 8);   /* heat 1 against residents of heat 1 */
    CHECK(!qt_is_resident(0, 8) && qt_is_resident(0, 5), "an expert no hotter than the coldest resident is refused");
    CHECK(fake_pipe_uploads == (arena ? R : 0) && fake_uploads == (arena ? 0 : 3 * R), "and nothing is copied for it");

    /* the coldest resident is the first of the equals: layer 0, expert 5 */
    const int vslot = arena ? slot_of(0, 5) : -1;
    for (int i = 0; i < 64 && !qt_is_resident(0, 9); i++) note(0, 9);
    CHECK(qt_is_resident(0, 9) && !qt_is_resident(0, 5), "a hot expert is swapped in for the coldest resident");
    CHECK(qt_is_resident(0, 6) && qt_is_resident(1, 7), "the other residents stay");
    pthread_mutex_lock(&G.mx);
    used = G.used[0]; const int swaps = (int)G.swaps;
    const int vgone = !qs(0, 5)->tg && !qs(0, 5)->tu && !qs(0, 5)->td && qs(0, 5)->aslot == 0;
    pthread_mutex_unlock(&G.mx);
    CHECK(swaps == 1 && used == (size_t)R * G.exp_bytes, "one swap, budget-neutral");
    CHECK(vgone, "the victim holds no tensor and no slot");
    if (!arena) CHECK(fake_overwrites == 3 && fake_uploads == 3 * R, "without the arena the swap overwrites the victim's tensors in place");
    else {
        CHECK(slot_of(0, 9) == vslot, "the newcomer is in the victim's slot");
        CHECK(G_anfree[0] == 0 && fake_pipe_allocs == 1, "no slot freed or added, no device allocation");
        CHECK(fake_overwrites == 0, "no overwrite through a view");
        CHECK(fake_pipe_uploads == R + 1 && fake_pipe_upload_bytes == (size_t)(R + 1) * G_aslot, "the swap is ONE copy of the slot");
        CHECK(fake_f8_views == v0 + 3 * (R + 1) && fake_tensor_frees == f0 + 3, "the victim's three views released once, three made");
        CHECK(slot_ok(0, 9) && slot_ok(0, 6) && slot_ok(1, 7), "the slot holds the newcomer; the others' slots are as they were");
        if (pool) CHECK(G_stg_used == (unsigned long long)(R + 1) && G_stg_nfree == QT_STG_N, "the swap staged through the pool too, every buffer back");
    }
    dig[3] = dev_digest(0, 9);
    CHECK(dig[3] == src_digest(0, 9), "the device holds the newcomer's bytes");
    CHECK(dev_digest(0, 6) == dig[1] && dev_digest(1, 7) == dig[2], "a swap changes no other resident's bytes");

    qt_shutdown();
    if (arena) {
        CHECK(!G_arena_on && !G_arena[0] && !G_stg, "shutdown releases the arena and the pool");
        CHECK(fake_pipe_allocs == 1 && fake_pipe_frees == 1, "the arena freed once");
        CHECK(fake_tensor_frees == fake_f8_views, "every view released exactly once");
    } else CHECK(fake_tensor_frees == fake_uploads, "every tensor released exactly once");
}

int main(void) {
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        const int k = l * NE + e;
        for (size_t i = 0; i < sizeof slab[l][e]; i++) slab[l][e][i] = (unsigned char)((i * 7 + k * 13 + (i >> 9)) & 0xFF);
        for (int i = 0; i < 3 * NSC; i++) scales[l][e][i] = 0.001f * (k + 1) + 0.0001f * i;
    }
    for (int i = 0; i < 256; i++) lut[i] = (float)i;
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    fake_ndev = 1; fake_keep_f8 = 1;   /* the default's uploads read back after the tier freed its staging */
    const char *mode; char m[240];

    uint64_t d0[4], d1[4], d2[4];
    printf(" off\n"); run("off", 0, 0, d0);
    printf(" 1\n");   run("1", 1, 0, d1);
    printf(" 2\n");   run("2", 1, 1, d2);
    check(d0[0] && !memcmp(d0, d1, sizeof d0), "1: the device holds the bytes of the default's three uploads, expert for expert");
    check(d0[0] && !memcmp(d0, d2, sizeof d0), "2: the device holds the bytes of the default's three uploads, expert for expert");

    printf(" arena refused\n"); mode = "arena refused";
    setenv("COLI_CUDA_EXPERT_ARENA", "2", 1); set_budget(off_bytes()); reset_counts();
    fake_alloc_fail = 1;
    int up = qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut);
    fake_alloc_fail = 0;
    CHECK(up, "the tier starts all the same");
    if (up) {
        CHECK(!G_nat_arena && !G_arena_on && !G_stg && fake_host_allocs == 0, "no arena, no pool");
        CHECK(G.exp_bytes == off_bytes(), "an expert budgeted at its device allocations again");
        note(0, 1);
        CHECK(qt_is_resident(0, 1) && fake_uploads == 3 && fake_pipe_uploads == 0 && fake_f8_views == 0, "three uploads an expert, no slot");
        CHECK(dev_digest(0, 1) == src_digest(0, 1), "the engine's bytes");
        qt_shutdown();
        CHECK(fake_tensor_frees == 3 && fake_pipe_frees == 0, "three tensors released, no arena to free");
    }

    printf(" view refused\n"); mode = "view refused";
    setenv("COLI_CUDA_EXPERT_ARENA", "2", 1); set_budget(off_bytes()); reset_counts();
    fake_f8_view_fail = 1;
    up = qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut);
    CHECK(up, "the tier starts all the same");
    if (up) {
        CHECK(!G_nat_arena && !G_arena_on && !G_arena[0] && !G_stg, "the arena is given back, no pool");
        CHECK(fake_pipe_allocs == 1 && fake_pipe_frees == 1 && fake_host_allocs == 0, "allocated once, freed once, no page-locked block");
        CHECK(G.exp_bytes == off_bytes(), "an expert budgeted at its device allocations again");
        note(0, 2);
        CHECK(qt_is_resident(0, 2) && fake_uploads == 3 && fake_pipe_uploads == 0, "three uploads an expert");
        CHECK(dev_digest(0, 2) == src_digest(0, 2), "the engine's bytes");
        qt_shutdown();
        CHECK(fake_tensor_frees == 3 && fake_pipe_frees == 1, "three tensors released, the arena not freed again");
    }
    fake_f8_view_fail = 0;

    printf(" pool refused\n"); mode = "pool refused";
    setenv("COLI_CUDA_EXPERT_ARENA", "2", 1); set_budget(slot_bytes()); reset_counts();
    fake_host_alloc_fail = 1;
    up = qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut);
    fake_host_alloc_fail = 0;
    CHECK(up, "the tier starts");
    if (up) {
        CHECK(G_nat_arena && G_arena_on && !G_stg, "the arena stays, without the pool");
        note(0, 3);
        CHECK(qt_is_resident(0, 3) && fake_uploads == 0 && fake_pipe_uploads == 1 && fake_pipe_upload_bytes == G_aslot, "still one copy of the slot, from one malloc'd block");
        CHECK(slot_ok(0, 3) && G_stg_used == 0, "the slot holds the engine's bytes");
        qt_shutdown();
        CHECK(fake_tensor_frees == fake_f8_views && fake_pipe_frees == 1, "every view and the arena released once");
    }

    unsetenv("COLI_CUDA_EXPERT_ARENA");
    if (fails) { printf("test_qwen36_tier_fp8_arena: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_fp8_arena: ok\n");
    return 0;
}
