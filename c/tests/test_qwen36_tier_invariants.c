/* Invariants of the qwen36 VRAM expert tier, checked rather than assumed.
 *
 * #1339, #1340 and #1341 were each found by reading, and each had a
 * one-scenario regression test written after the fact. This file states the
 * rules those scenarios are instances of, and checks the rules directly on
 * the fake CUDA backend (tests/qwen36_fake_cuda.h), so the next drift fails
 * here before anyone reads the code again:
 *
 *   1. Budget accounting balances. On every device, bytes in use never exceed
 *      the budget, and once the queue is drained, bytes in use equal exactly
 *      resident experts times bytes per expert: every reservation was either
 *      consumed by an upload or handed back. Checked on one and on two
 *      devices, and across the path where a planned expert is never handed
 *      over -- the reservation must not leak with it.
 *
 *   2. Shutdown returns from every wait state at once. Four places sleep on
 *      cv_take (the uploader's victim wait, qt_note_block, qt_note_planned,
 *      qt_fill_wait); #1340 was shutdown waking none of them. Here all four
 *      are parked at the same time, under a portable watchdog, and shutdown
 *      has to bring every one of them home.
 *
 *   3. Issue geometry holds under random routing. Random resident sets,
 *      random K up to the row limit, one to three devices, many seeds: every
 *      device block lies inside the replica buffer, blocks of different
 *      devices never overlap, mask bits name exactly the routed experts that
 *      were resident on a device whose issue succeeded, and hits plus misses
 *      add up to everything routed. Under ASan (make test-asan) this doubles
 *      as a fuzz for the class of #1339. The row limit is read from the array
 *      the rows index, so the stride cannot drift from it again.
 *
 * Everything runs in the plain CPU build; no GPU, no toolkit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

/* ---- portable watchdog: a hang is a failure, not a stuck CI job ----------
 * Static state on purpose: the thread outlives the function that armed it,
 * and a pointer into that function's frame is a stack-use-after-return the
 * moment it returns -- the same lifetime mistake as #1277, caught here by
 * ASan on the first draft of this very test. Disarmed after a successful
 * shutdown so a late wake-up cannot fail a run that already passed. */
static volatile int watchdog_armed;
static int watchdog_seconds;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { watchdog_seconds, 0 };
    nanosleep(&limit, NULL);
    if (!watchdog_armed) return NULL;
    fprintf(stderr, "FAIL: tier did not come home within %d s (hang, see #1340)\n", watchdog_seconds);
    _exit(1);
}
static void arm_watchdog(int seconds) {
    pthread_t t;
    watchdog_seconds = seconds; watchdog_armed = 1;
    pthread_create(&t, NULL, watchdog, NULL);
    pthread_detach(t);
}
static void disarm_watchdog(void) { watchdog_armed = 0; }

/* wait (2 ms polls) until a G.mx-guarded condition holds, or give up */
#define WAIT_UNTIL(cond, max_polls)                                    \
    do {                                                               \
        for (int wu_i = 0; wu_i < (max_polls); wu_i++) {               \
            pthread_mutex_lock(&G.mx); int wu_ok = (cond);             \
            pthread_mutex_unlock(&G.mx);                               \
            if (wu_ok) break;                                          \
            struct timespec wu_ts = {0, 2000000}; nanosleep(&wu_ts, NULL); \
        }                                                              \
    } while (0)

/* the queue being empty is not the same as the uploader being done: the last
 * dequeued expert is still queued=1 until its upload returns */
static int all_settled(void) {
    if (G.qn != 0) return 0;
    for (int l = 0; l < G.nl; l++) for (int e = 0; e < G.ne; e++) if (qs(l, e)->queued) return 0;
    return 1;
}
#define WAIT_IDLE() WAIT_UNTIL(all_settled(), 1000)

/* ---- fake weights: one set per expert, recognisable bytes ---------------- */
enum { D = 64, IH = 32 };
#define MB4 (D * IH / 2)
#define NSC (2 * IH + D)             /* per-row scales: gs=0 -> sc_gu = IH, sc_d = D */

static unsigned char g4s[64][MB4], u4s[64][MB4], d4s[64][MB4];
static float scs[64][NSC];
static void make_weights(int ne) {
    for (int e = 0; e < ne; e++) {
        memset(g4s[e], (unsigned char)(e + 1), MB4);
        memset(u4s[e], (unsigned char)(e + 2), MB4);
        memset(d4s[e], (unsigned char)(e + 3), MB4);
        for (int i = 0; i < NSC; i++) scs[e][i] = 1.0f;
    }
}
#define NOTE(fn, l, e) fn((l), (e), g4s[e], u4s[e], d4s[e], scs[e], scs[e] + IH, scs[e] + 2 * IH)

static int start_tier(int ndev, int nl, int ne, int topk, const char *budget_gb) {
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", ndev == 1 ? "0" : ndev == 2 ? "0,1" : "0,1,2", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    /* "auto" is the tier's own spelling of "no explicit budget" (free minus
     * 1 GiB); unsetenv would be POSIX-only and MinGW has no equivalent */
    setenv("CUDA_EXPERT_GB", budget_gb ? budget_gb : "auto", 1);
    fake_ndev = ndev;
    fake_uploads = 0;
    fake_issue_hook = NULL;
    return qt_init(nl, ne, D, IH, ne, topk, 0 /* per-row */, 1 /* int4 */);
}

static size_t resident_on(int dev) {
    size_t n = 0;
    for (int l = 0; l < G.nl; l++)
        for (int e = 0; e < G.ne; e++)
            if (home(e) == dev && qs(l, e)->resident) n++;
    return n;
}

/* Bytes per expert as the tier computed it, plus the budget that holds
 * exactly `n` of them and not n+1 (so the fill stops on budget, not on the
 * end of the list). Written as GiB text because that is the knob. */
static void budget_for(int n, char *out, size_t len) {
    double gib = ((double)G.exp_bytes * n + G.exp_bytes / 2) / 1073741824.0;
    snprintf(out, len, "%.12f", gib);
}

/* ======================================================================== */
/* 1. budget accounting                                                     */
/* ======================================================================== */
static void check_balance(const char *ctx) {
    char msg[160];
    pthread_mutex_lock(&G.mx);
    for (int di = 0; di < G.ndev; di++) {
        size_t res = resident_on(di);
        snprintf(msg, sizeof msg, "%s: dev %d used %zu > budget %zu", ctx, di, G.used[di], G.budget[di]);
        check(G.used[di] <= G.budget[di], msg);
        snprintf(msg, sizeof msg, "%s: dev %d used %zu != %zu resident x %zu B (a reservation leaked or was double-returned)",
                 ctx, di, G.used[di], res, G.exp_bytes);
        check(G.used[di] == res * G.exp_bytes, msg);
    }
    pthread_mutex_unlock(&G.mx);
}

static void test_budget_accounting(int ndev) {
    enum { NL = 2, NE = 16 };
    make_weights(NE);
    /* first start only to learn exp_bytes, then restart with a budget that
     * holds 5 experts per device: 10 of 32 fit, the rest must stay CPU */
    if (!start_tier(ndev, NL, NE, 4, NULL)) { check(0, "tier did not start"); return; }
    char b[64]; budget_for(5, b, sizeof b);
    qt_shutdown();
    if (!start_tier(ndev, NL, NE, 4, b)) { check(0, "tier did not start with a small budget"); return; }

    int layers[NL * NE], eids[NL * NE];
    int planned = qt_plan_fill(layers, eids, NL * NE);
    check(planned == 5 * ndev, "plan_fill should reserve exactly the budget: 5 experts per device");
    pthread_mutex_lock(&G.mx);
    for (int di = 0; di < G.ndev; di++)
        check(G.used[di] == 5 * G.exp_bytes, "plan_fill reservations must equal 5 x exp_bytes per device");
    pthread_mutex_unlock(&G.mx);

    /* hand over all but the last planned expert normally; the last one is
     * reported the way the warmstart reports an expert whose loader came back
     * empty -- with NULL weights */
    for (int i = 0; i < planned - 1; i++) NOTE(qt_note_planned, layers[i], eids[i]);
    qt_note_planned(layers[planned - 1], eids[planned - 1], NULL, NULL, NULL, NULL, NULL, NULL);
    qt_fill_wait();
    WAIT_IDLE();

    /* a second hand-over of an already-resident expert must not return a
     * reservation it does not hold (double return would drive used below
     * what is resident) */
    NOTE(qt_note_planned, layers[0], eids[0]);
    /* the free-running path (decode-time note) on a non-planned expert on a
     * full device must be refused without touching the books */
    int extra_l = NL - 1, extra_e = NE - 1;
    while (qs(extra_l, extra_e)->planned || qs(extra_l, extra_e)->resident) extra_e--;
    NOTE(qt_note, extra_l, extra_e);
    /* and make every remaining expert known to the tier (g4 set, upload
     * refused by the full budget) so an LFRU swap has candidates on each
     * device -- exactly the population decode-time qt_note builds up */
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) NOTE(qt_note, l, e);
    WAIT_IDLE();

    pthread_mutex_lock(&G.mx);
    QSlot *orphan = qs(layers[planned - 1], eids[planned - 1]);
    int orphan_dev = home(eids[planned - 1]);
    size_t used_orphan_dev = G.used[orphan_dev], res_orphan_dev = resident_on(orphan_dev);
    pthread_mutex_unlock(&G.mx);
    /* The rule under test: a reservation is consumed or returned. The
     * planned-but-never-delivered expert holds neither a tensor nor a right
     * to the budget; if its bytes are still counted, the device is smaller
     * than the tier believes for the life of the process, and
     * "if(resident||queued||planned) continue" never reconsiders it. */
    check(!orphan->resident && !orphan->queued && !orphan->planned,
          "an expert reported with no weights must end up neither resident, queued nor planned");
    check(used_orphan_dev == res_orphan_dev * G.exp_bytes,
          "undelivered planned expert: its reservation is still counted (leak: planned=1 keeps it and the "
          "budget never gets those bytes back)");
    check_balance(ndev == 1 ? "1 device, after warmstart" : "2 devices, after warmstart");

    /* budget is per device: a device must not overflow because another has room */
    for (int di = 0; di < G.ndev; di++)
        check(resident_on(di) <= 5, "no device may hold more than its own budget allows");

    /* exercise one LFRU swap end to end and re-check the books: a swap is
     * budget-neutral by construction, so used must not move */
    pthread_mutex_lock(&G.mx);
    size_t used_before[QT_MAX_DEV]; memcpy(used_before, G.used, sizeof used_before);
    int vict_l = -1, vict_e = -1, hot_l = -1, hot_e = -1;
    for (int l = 0; l < NL && vict_l < 0; l++)
        for (int e = 0; e < NE; e++)
            if (qs(l, e)->resident) { vict_l = l; vict_e = e; break; }
    for (int l = 0; l < NL && hot_l < 0; l++)
        for (int e = 0; e < NE; e++)
            if (!qs(l, e)->resident && !qs(l, e)->queued && qs(l, e)->g4 && home(e) == home(vict_e)) {
                hot_l = l; hot_e = e; break; }
    int swapped = 0;
    if (vict_l >= 0 && hot_l >= 0) {
        qs(vict_l, vict_e)->resident = 0;
        swapped = enqueue_locked(hot_l, hot_e, vict_l, vict_e, 0);
        if (!swapped) qs(vict_l, vict_e)->resident = 1;
    }
    pthread_mutex_unlock(&G.mx);
    check(swapped, "an LFRU swap between a resident and a hot non-resident on the same device must enqueue");
    WAIT_IDLE();
    pthread_mutex_lock(&G.mx);
    for (int di = 0; di < G.ndev; di++)
        check(G.used[di] == used_before[di], "an LFRU swap is budget-neutral: used must not move");
    pthread_mutex_unlock(&G.mx);
    check(qt_is_resident(hot_l, hot_e) && !qt_is_resident(vict_l, vict_e), "after the swap the hot expert is resident and the victim is not");
    check_balance("after one LFRU swap");

    qt_shutdown();
    check(!G.on, "shutdown must switch the tier off");
}

/* ======================================================================== */
/* 2. shutdown wakes every waiter                                            */
/* ======================================================================== */
static int t_note_block_done, t_note_planned_done, t_fill_wait_done;
static void *th_note_block(void *a)   { (void)a; NOTE(qt_note_block, 1, 31);   t_note_block_done = 1;   return NULL; }
static void *th_note_planned(void *a) { (void)a; NOTE(qt_note_planned, 1, 30); t_note_planned_done = 1; return NULL; }
static void *th_fill_wait(void *a)    { (void)a; qt_fill_wait();               t_fill_wait_done = 1;    return NULL; }

static void test_shutdown_wakes_everyone(void) {
    enum { NL = 2, NE = 32 };            /* 64 experts: more swap candidates than the queue holds */
    make_weights(NE);
    if (!start_tier(1, NL, NE, 1, NULL)) { check(0, "tier did not start"); return; }
    char b[64]; budget_for(1, b, sizeof b);
    qt_shutdown();
    if (!start_tier(1, NL, NE, 1, b)) { check(0, "tier did not start with a one-expert budget"); return; }

    /* one resident expert, everything else known to the tier but refused by
     * the budget -- exactly the population an LFRU swap needs */
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) if (l || e != 30) if (l != 1 || (e != 30 && e != 31)) NOTE(qt_note, l, e);
    WAIT_IDLE();
    check(qt_is_resident(0, 0), "expert (0,0) should have taken the one-expert budget");

    /* open a group and never take it: issue_open stays set, which is the
     * state the uploader's victim wait keys on */
    fake_issue_hook = NULL;                 /* issue fails -> mask 0, but issue_open is set regardless */
    float x[D]; for (int i = 0; i < D; i++) x[i] = (float)i;
    int e0 = 0; (void)qt_issue(0, &e0, 1, x);
    pthread_mutex_lock(&G.mx);
    check(G.issue_open == 1, "a group must be open after qt_issue");
    /* fill the queue to capacity with swaps against the one victim, so that
     * the uploader parks on the victim wait AND the queue reads full */
    qs(0, 0)->resident = 0;
    int queued = 0;
    for (int e = 1; e < NE && queued < QT_QCAP; e++)
        if (enqueue_locked(0, e, 0, 0, 0)) queued++;
    for (int e = 0; e < NE - 2 && queued < QT_QCAP; e++)      /* (1,30) and (1,31) stay for the waiter threads */
        if (enqueue_locked(1, e, 0, 0, 0)) queued++;
    int qn_now = G.qn;
    pthread_mutex_unlock(&G.mx);
    check(queued >= 1, "at least one swap must be enqueued to park the uploader");
    /* the uploader dequeues exactly one and parks; the rest stay queued */
    WAIT_UNTIL(G.qn == qn_now - 1, 500);
    pthread_mutex_lock(&G.mx);
    int parked_qn = G.qn;
    /* top the queue back up to full so qt_note_block / qt_note_planned have to wait */
    for (int l = 0; l < NL && G.qn < QT_QCAP; l++)
        for (int e = 0; e < NE && G.qn < QT_QCAP; e++)
            if (!(l == 1 && e >= 30) && !qs(l, e)->resident && !qs(l, e)->queued && qs(l, e)->g4)
                enqueue_locked(l, e, 0, 0, 0);
    int full = (G.qn == QT_QCAP);
    pthread_mutex_unlock(&G.mx);
    (void)parked_qn;
    check(full, "the upload queue must be full for the blocking note paths to park");

    /* three more waiters, each on cv_take through a different entry point */
    pthread_t a, bth, c;
    t_note_block_done = t_note_planned_done = t_fill_wait_done = 0;
    pthread_create(&a, NULL, th_note_block, NULL);
    pthread_create(&bth, NULL, th_note_planned, NULL);
    pthread_create(&c, NULL, th_fill_wait, NULL);
    struct timespec settle = {0, 50000000}; nanosleep(&settle, NULL);   /* 50 ms: let them park */
    check(!t_note_block_done && !t_note_planned_done && !t_fill_wait_done,
          "with a full queue and an open group all three callers must be parked before shutdown");

    arm_watchdog(10);
    qt_shutdown();
    pthread_join(a, NULL); pthread_join(bth, NULL); pthread_join(c, NULL);
    disarm_watchdog();
    check(!G.on, "shutdown_returns_with_four_waiters_parked");
    check(t_note_block_done && t_note_planned_done && t_fill_wait_done,
          "every cv_take waiter must return once shutdown has been requested");
    /* the abandoned swaps left the books consistent: the victim still holds
     * its tensor and says so, no queued flag survives */
    pthread_mutex_lock(&G.mx);
    int stale_queued = 0;
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) stale_queued += qs(l, e)->queued;
    pthread_mutex_unlock(&G.mx);
    check(qs(0, 0)->resident == 1 && qs(0, 0)->tg != NULL, "the victim of an abandoned swap keeps its tensor and its resident flag");
    check(stale_queued == 0, "no expert may still read as queued after shutdown drained or abandoned the queue");
}

/* ======================================================================== */
/* 3. issue geometry under random routing                                    */
/* ======================================================================== */
enum { MAX_REC = 8 };
static struct { int device, count; const float *x; } rec[MAX_REC];
static int nrec;
static int fail_device = -1;
static int record_issue(int device, int count, const float *x) {
    if (nrec < MAX_REC) { rec[nrec].device = device; rec[nrec].count = count; rec[nrec].x = x; nrec++; }
    return device != fail_device;
}

static uint32_t rng_state;
static uint32_t rng(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }

static void test_issue_geometry(int ndev, int seeds) {
    enum { NL = 1, NE = 64 };
    const int ROWS = (int)(sizeof G.is_k[0] / sizeof G.is_k[0][0]);   /* the array the rows index */
    make_weights(NE);
    if (!start_tier(ndev, NL, NE, ROWS, NULL)) { check(0, "tier did not start"); return; }
    check(G.is_x_floats == (size_t)G.ndev * ROWS * G.D,
          "replica buffer must hold ROWS rows per device, ROWS read from is_k, not a second literal");
    /* everything resident: 2 GiB fake budget per device holds all 64 */
    for (int e = 0; e < NE; e++) NOTE(qt_note_block, 0, e);
    qt_fill_wait();
    WAIT_IDLE();
    for (int e = 0; e < NE; e++) if (!qt_is_resident(0, e)) { check(0, "warmstart left an expert non-resident"); break; }

    fake_issue_hook = record_issue;
    float x[D]; for (int i = 0; i < D; i++) x[i] = (float)i;
    float val[32]; for (int k = 0; k < 32; k++) val[k] = 1.0f;
    float out[D];
    uint64_t hits0 = 0, miss0 = 0;

    for (int s = 1; s <= seeds; s++) {
        rng_state = 0x9E3779B9u * (uint32_t)s + 1u;
        /* random resident set: knock out a random subset for this round */
        int knocked[NE]; memset(knocked, 0, sizeof knocked);
        pthread_mutex_lock(&G.mx);
        for (int e = 0; e < NE; e++) if ((rng() & 3) == 0) { knocked[e] = 1; qs(0, e)->resident = 0; }
        pthread_mutex_unlock(&G.mx);
        /* random K distinct experts, random failing device on some rounds */
        int K = 1 + (int)(rng() % (uint32_t)ROWS);
        int eids[32]; int used[NE]; memset(used, 0, sizeof used);
        for (int k = 0; k < K; k++) { int e; do e = (int)(rng() % NE); while (used[e]); used[e] = 1; eids[k] = e; }
        fail_device = (rng() % 4 == 0) ? (int)(rng() % (uint32_t)ndev) : -1;

        pthread_mutex_lock(&G.mx);
        for (int di = 0; di < G.ndev; di++) hits0 += G.hits[di];
        miss0 = G.miss; uint64_t hits_before = hits0, miss_before = miss0; hits0 = 0;
        pthread_mutex_unlock(&G.mx);

        nrec = 0;
        uint32_t mask = qt_issue(0, eids, K, x);

        /* (a) every block inside the buffer, (b) blocks pairwise disjoint,
           (c) each block at its device's slot: base + device_index * ROWS * D */
        for (int i = 0; i < nrec; i++) {
            const float *lo = G.is_x, *hi = G.is_x + G.is_x_floats;
            check(rec[i].x >= lo && rec[i].x + (size_t)rec[i].count * G.D <= hi, "issue block outside the replica buffer");
            int di = -1; for (int d = 0; d < G.ndev; d++) if (G.dev[d] == rec[i].device) di = d;
            check(di >= 0 && rec[i].x == G.is_x + (size_t)di * ROWS * G.D, "issue block not at its device's slot");
            for (int j = 0; j < i; j++) {
                int disjoint = (rec[i].x + (size_t)rec[i].count * G.D <= rec[j].x) ||
                               (rec[j].x + (size_t)rec[j].count * G.D <= rec[i].x);
                check(disjoint, "issue blocks of two devices overlap");
            }
        }
        /* (d) the mask names exactly: routed, resident, on a device whose issue succeeded */
        uint32_t expect = 0; int expect_hits = 0, expect_miss = 0;
        for (int k = 0; k < K; k++) {
            int e = eids[k];
            if (knocked[e]) { expect_miss++; continue; }
            expect_hits++;
            if (home(e) != fail_device) expect |= 1u << k;
        }
        check(mask == expect, "mask bits must be exactly the resident experts on devices whose issue succeeded");
        /* (e) hits + misses account for every routed expert */
        pthread_mutex_lock(&G.mx);
        uint64_t hits_after = 0; for (int di = 0; di < G.ndev; di++) hits_after += G.hits[di];
        check(hits_after - hits_before == (uint64_t)expect_hits && G.miss - miss_before == (uint64_t)expect_miss,
              "hits + misses must add up to the routed experts");
        pthread_mutex_unlock(&G.mx);
        /* (f) take closes the group whatever the fake returned (is_cnt is
           re-initialised by the next qt_issue, so it is not a contract here) */
        memset(out, 0, sizeof out);
        qt_take(mask, val, K, out);
        pthread_mutex_lock(&G.mx);
        check(G.issue_open == 0, "qt_take must close the group");
        for (int e = 0; e < NE; e++) if (knocked[e]) qs(0, e)->resident = 1;   /* restore for the next round */
        pthread_mutex_unlock(&G.mx);
        if (fails) { printf("  (stopping the geometry sweep at seed %d, ndev %d, K %d)\n", s, ndev, K); break; }
    }
    fake_issue_hook = NULL;
    qt_shutdown();
}

int main(void) {
    printf("qwen36 tier invariants\n");
    printf(" 1. budget accounting, 1 device\n");  test_budget_accounting(1);
    printf(" 1. budget accounting, 2 devices\n"); test_budget_accounting(2);
    printf(" 2. shutdown wakes every waiter\n");  test_shutdown_wakes_everyone();
    printf(" 3. issue geometry, random routing, 1/2/3 devices x 200 seeds\n");
    test_issue_geometry(1, 200); test_issue_geometry(2, 200); test_issue_geometry(3, 200);
    if (fails) { printf("test_qwen36_tier_invariants: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_invariants: ok\n");
    return 0;
}
