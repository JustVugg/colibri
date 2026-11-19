/* The CUDA tier's arena for the native int4 format with grouped scales (COLI_CUDA_EXPERT_ARENA),
 * on the fake backend.
 *
 * Without the switch an expert goes up as three tensors, each with its own device allocations for
 * the weights and for the scales (coli_cuda_tensor_upload_g): a warm start of 10 240 experts made
 * 61 916 allocations and 62 000 waited copies. With it the tier allocates one arena a device, and
 * an expert is one copy of its three matrices and one of its scales into its slot
 * (coli_cuda_slot_fill_s4), the tensors pointing into the slot (coli_cuda_tensor_at_g). With the
 * switch at 2 the staging is a pool of page-locked buffers (coli_cuda_host_alloc) and an expert is
 * repacked outside the tier's lock (enqueue_staged), from as many loader threads as call.
 *
 *   - off (the default): no slot is filled, every expert is three uploads, as before;
 *   - 1: every expert in a slot of its own, the slot's bytes the staged bytes (the three matrices
 *     back to back, the scales after), the tensors at the slot's offsets;
 *   - 1, the arena refused by the device: the tier falls back to an allocation a matrix;
 *   - 2: the same slots, every expert staged through the pool, every buffer back in it after the
 *     warm start; from one thread and from many, more experts than the queue holds; through
 *     qt_note_block and through qt_plan_fill + qt_note_planned;
 *   - 2, the page-locked block refused: the arena without the pool. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

#include "../compat.h"   /* setenv, unsetenv */

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { NL = 4, NE = 32, D = 64, IH = 32, TOPK = 2, GS = 16,
       MB = D * IH / 2,                       /* a matrix of packed nibbles: 1024 bytes */
       SGU = IH * (D / GS), SD = D * (IH / GS) /* scale floats: gate or up 128, down 128 */ };
/* NL * NE = 128 experts: more than the upload queue holds (QT_QCAP 48), so loaders wait for room */

static unsigned char g4[NL][NE][MB], u4[NL][NE][MB], d4[NL][NE][MB];
static float sg[NL][NE][SGU], su[NL][NE][SGU], sd[NL][NE][SD];

static void fill_sources(void) {
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        const int k = l * NE + e;
        for (int i = 0; i < MB; i++) {
            g4[l][e][i] = (unsigned char)(3 * k + 1 + i);
            u4[l][e][i] = (unsigned char)(3 * k + 2 + 5 * i);
            d4[l][e][i] = (unsigned char)(3 * k + 3 + 7 * i);
        }
        for (int i = 0; i < SGU; i++) { sg[l][e][i] = 1.0f + k + 0.001f * i; su[l][e][i] = 2.0f + k + 0.001f * i; }
        for (int i = 0; i < SD; i++) sd[l][e][i] = 3.0f + k + 0.001f * i;
    }
}
/* the warm start through qt_note_block: from one thread, or from every thread OpenMP gives */
static void stage_all(int threads) {
    if (threads) {
        #pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++)
            qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sg[l][e], su[l][e], sd[l][e]);
    } else {
        for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++)
            qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sg[l][e], su[l][e], sd[l][e]);
    }
    qt_fill_wait();
}
/* the engine's own form: qt_plan_fill hands out the set, loader threads note what they were given */
static void stage_planned(void) {
    static int pl[NL * NE], pe[NL * NE];
    const int n = qt_plan_fill(pl, pe, NL * NE);
    check(n == NL * NE, "qt_plan_fill plans every expert");
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        const int l = pl[i], e = pe[i];
        qt_note_planned(l, e, g4[l][e], u4[l][e], d4[l][e], sg[l][e], su[l][e], sd[l][e]);
    }
    qt_fill_wait();
}
static int residents(void) {
    int r = 0, q = 0;
    pthread_mutex_lock(&G.mx);
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) { r += qs(l, e)->resident; q += qs(l, e)->queued; }
    pthread_mutex_unlock(&G.mx);
    return q ? -q : r;   /* anything still queued when the wait returned is a failure of its own */
}
/* every expert in a slot of its own, the slot the staged bytes, the tensors at its offsets */
static void check_slots(const char *mode) {
    static unsigned char seen[NL * NE];
    memset(seen, 0, sizeof seen);
    int distinct = 1, placed = 1, bytes_ok = 1, scales_ok = 1, shape_ok = 1;
    pthread_mutex_lock(&G.mx);
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        QSlot *s = qs(l, e);
        const int a = s->aslot - 1;
        if (a < 0 || a >= NL * NE || seen[a]) { distinct = 0; continue; }
        seen[a] = 1;
        const unsigned char *b = (const unsigned char *)G_arena[0] + (size_t)a * G_aslot;
        const float *bs = (const float *)(b + G_aoff_s);
        if (!s->tg || !s->tu || !s->td) { placed = 0; continue; }
        if (s->tg->w != b || s->tu->w != b + MB || s->td->w != b + 2 * MB ||
            s->tg->sc != bs || s->tu->sc != bs + SGU || s->td->sc != bs + 2 * SGU) placed = 0;
        /* the staging's bytes: the nibbles as the backend takes them, offset-binary (stage(): XOR 0x88);
         * the fake's fill copies them as they are, where the device then makes them signed */
        for (int i = 0; i < MB; i++)
            if (b[i] != (g4[l][e][i] ^ 0x88) || b[MB + i] != (u4[l][e][i] ^ 0x88) || b[2 * MB + i] != (d4[l][e][i] ^ 0x88)) { bytes_ok = 0; break; }
        if (memcmp(bs, sg[l][e], sizeof sg[l][e]) || memcmp(bs + SGU, su[l][e], sizeof su[l][e]) ||
            memcmp(bs + 2 * SGU, sd[l][e], sizeof sd[l][e])) scales_ok = 0;
        if (s->tg->fmt != 4 || s->tg->gs != GS || s->tg->I != D || s->tg->O != IH ||
            s->td->I != IH || s->td->O != D || s->td->gs != GS) shape_ok = 0;
    }
    const int nfree = G_anfree[0];
    pthread_mutex_unlock(&G.mx);
    char m[160];
    snprintf(m, sizeof m, "%s: every expert has a slot of its own", mode); check(distinct, m);
    snprintf(m, sizeof m, "%s: the tensors point at the slot's matrices and scales", mode); check(placed, m);
    snprintf(m, sizeof m, "%s: a slot holds the staged matrices back to back", mode); check(bytes_ok, m);
    snprintf(m, sizeof m, "%s: a slot holds the staged scales after them", mode); check(scales_ok, m);
    snprintf(m, sizeof m, "%s: the tensors carry the format, the group size and the shapes", mode); check(shape_ok, m);
    snprintf(m, sizeof m, "%s: no free slot is left when every expert is resident", mode); check(nfree == 0, m);
}
static int up(const char *what) {
    fake_uploads = 0; fake_slot_fills = 0; fake_host_allocs = 0;
    if (!qt_init(NL, NE, D, IH, NE, TOPK, GS, 1 /* int4 */)) { printf("  FAIL: the tier should start (%s)\n", what); fails++; return 0; }
    return 1;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    fill_sources();

    /* ---- off: as before -------------------------------------------------------------- */
    unsetenv("COLI_CUDA_EXPERT_ARENA");
    if (!up("switch off")) return 1;
    check(!G_nat_arena && !G_arena_on && !G_stg, "no arena and no pool without the switch");
    stage_all(0);
    check(residents() == NL * NE, "switch off: every expert resident");
    check(fake_slot_fills == 0, "switch off: no slot is filled");
    check(fake_uploads == 3 * NL * NE, "switch off: three uploads an expert");
    qt_shutdown();

    /* ---- 1: an expert a slot ----------------------------------------------------------- */
    setenv("COLI_CUDA_EXPERT_ARENA", "1", 1);
    if (!up("switch 1")) return 1;
    check(G_nat_arena && G_arena_on && G_arena[0], "1: the arena is up");
    check(!G_stg && fake_host_allocs == 0, "1: no page-locked pool");
    check(G_aslots[0] == NL * NE, "1: a slot for every expert");
    check(G_aoff_s == 3 * MB && G_aslot == 3 * MB + (2 * SGU + SD) * sizeof(float), "1: the slot is three matrices, then the scales");
    check(G.exp_bytes == G_aslot, "1: an expert is budgeted at its slot");
    stage_all(0);
    check(residents() == NL * NE, "1: every expert resident");
    check(fake_slot_fills == NL * NE, "1: one fill an expert");
    check_slots("1");
    qt_shutdown();
    check(!G_arena_on && !G_arena[0], "1: shutdown releases the arena");

    /* ---- 1, the arena refused: an allocation a matrix again -------------------------------- */
    fake_alloc_fail = 1;
    if (!up("arena refused")) return 1;
    fake_alloc_fail = 0;
    check(!G_nat_arena && !G_arena_on, "a refused arena leaves the tier without one");
    check(G.exp_bytes != G_aslot, "and an expert budgeted at its allocations");
    stage_all(0);
    check(residents() == NL * NE, "arena refused: every expert resident all the same");
    check(fake_slot_fills == 0 && fake_uploads == 3 * NL * NE, "arena refused: three uploads an expert");
    qt_shutdown();

    /* ---- 2: page-locked staging, repacked outside the lock; one loader thread ---------------- */
    setenv("COLI_CUDA_EXPERT_ARENA", "2", 1);
    if (!up("switch 2")) return 1;
    check(G_nat_arena && G_arena_on && G_stg, "2: the arena and the staging pool are up");
    check(fake_host_allocs == 1, "2: one page-locked block");
    check(G_stg_nfree == QT_STG_N, "2: the pool starts full");
    stage_all(0);
    check(residents() == NL * NE, "2: every expert resident when the wait returns, none queued");
    check(fake_slot_fills == NL * NE, "2: one fill an expert");
    check(G_stg_used == NL * NE, "2: every expert staged through the pool");
    check(G_stg_nfree == QT_STG_N, "2: every buffer back in the pool after the warm start");
    check(G.inflight == 0 && G.qn == 0, "2: nothing in flight, the queue empty");
    check_slots("2");
    qt_shutdown();
    check(!G_stg && !G_arena_on, "2: shutdown releases the pool and the arena");

    /* ---- 2, many loader threads, more experts than the queue holds --------------------------- */
    if (!up("switch 2, threads")) return 1;
    stage_all(1);
    check(residents() == NL * NE, "2 threads: every expert resident when the wait returns, none queued");
    check(fake_slot_fills == NL * NE, "2 threads: one fill an expert, none twice");
    check(G_stg_nfree == QT_STG_N, "2 threads: every buffer back in the pool");
    check(G.inflight == 0 && G.qn == 0, "2 threads: nothing in flight, the queue empty");
    check_slots("2 threads");
    qt_shutdown();

    /* ---- 2, the engine's form: qt_plan_fill, then qt_note_planned from many threads --------------- */
    if (!up("switch 2, planned")) return 1;
    stage_planned();
    check(residents() == NL * NE, "2 planned: every expert resident when the wait returns, none queued");
    check(fake_slot_fills == NL * NE, "2 planned: one fill an expert");
    check(G_stg_nfree == QT_STG_N, "2 planned: every buffer back in the pool");
    {
        size_t used = 0;
        pthread_mutex_lock(&G.mx); used = G.used[0]; pthread_mutex_unlock(&G.mx);
        check(used == (size_t)NL * NE * G_aslot, "2 planned: the budget holds every expert once");
    }
    check_slots("2 planned");
    qt_shutdown();

    /* ---- 2, the page-locked block refused: the arena without the pool --------------------------- */
    fake_host_alloc_fail = 1;
    if (!up("pool refused")) return 1;
    fake_host_alloc_fail = 0;
    check(G_nat_arena && G_arena_on && !G_stg, "pool refused: the arena stays, without the pool");
    stage_all(1);
    check(residents() == NL * NE, "pool refused: every expert resident");
    check(fake_slot_fills == NL * NE && G_stg_used == 0, "pool refused: one fill an expert, none through the pool");
    check_slots("pool refused");
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_int4_arena: %d failures\n", fails); return 1; }
    printf("test_qwen36_tier_int4_arena: ok\n");
    return 0;
}
