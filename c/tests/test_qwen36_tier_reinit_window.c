/* qt_init must be refused while a teardown is running, and the shutdown must be
 * synchronous (#1564, defect 2: the re-init guard was blind to the teardown
 * window).
 *
 * The defect. The re-init guard was `if(G.on) return 0;`, and qt_shutdown
 * cleared G.on at the TOP of the teardown -- before the slow part. For the
 * whole of that window (the drain, pthread_join, the backend drain, the frees)
 * the guard read "never started" and therefore ACCEPTED a re-init:
 *
 *     memset(&G,0,sizeof G)          -- over the running teardown, over the
 *                                      mutex it is about to park on, and over
 *                                      the once-only claim, which comes back
 *                                      as 0 --
 *
 * Measured, with the re-init landing in the middle of the teardown:
 *
 *     #0 __pthread_clockjoin_ex
 *     #1 qt_shutdown () at qwen36_tier.c:1243   <- pthread_join(G.th,NULL)
 *     PROBE_B2_EXIT=139
 *     [hook] re-init during teardown returned 1 (0=refused, 1=ACCEPTED)
 *     teardown_latch_now=0                        <- the claim, re-armed
 *     PROBE_REINIT_EXIT=124
 *
 * And the losers of the teardown did not wait: a caller that lost the
 * once-only claim returned immediately, so it could come back while the tier
 * was still being torn down (LOSER_RETURNED_BEFORE_TEARDOWN_COMPLETE=1).
 *
 * What is checked, all of it reported by the code that decides it:
 *   - a qt_init issued while the teardown is parked mid-flight is REFUSED, and
 *     it changes nothing: same live tensors, same storage, G.on untouched.
 *     The rendezvous is the existing qt_test_drain_leave_hook, so the teardown
 *     is held inside its own slow window while the re-init is issued -- no wall
 *     clock, and a teardown that deleted its slow window could not park there
 *     for the test to talk to.
 *   - G.on is still 1 in that window. It is no longer what admits callers, and
 *     a flag that is only cleared at the END is what let the guard read "never
 *     started" in the first place.
 *   - the teardown then runs to completion and frees every live tensor once.
 *   - a concurrent qt_shutdown does not come back before that is done: the
 *     loser is released only when the teardown's own teardown-complete seam
 *     fires, and it is the teardown that signals, so "the loser returned first"
 *     is a positive observation rather than a race.
 *   - and the lifecycle is still REUSABLE afterwards: 400 sequential
 *     init/teardown cycles, each of which would hang or double-free if any
 *     once-only claim were still being re-armed.
 *
 * Fake CUDA backend, no GPU and no model file, like the other tier tests. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

/* ---- fake weights, recognisable bytes per expert ------------------------- */
enum { D = 64, IH = 32, NEXP = 4 };
#define MB4 (D * IH / 2)
#define NSC (2 * IH + D)
static unsigned char g4s[NEXP][MB4], u4s[NEXP][MB4], d4s[NEXP][MB4];
static float scs[NEXP][NSC];
static void make_weights(void) {
    for (int e = 0; e < NEXP; e++) {
        memset(g4s[e], (unsigned char)(e + 1), MB4);
        memset(u4s[e], (unsigned char)(e + 2), MB4);
        memset(d4s[e], (unsigned char)(e + 3), MB4);
        for (int i = 0; i < NSC; i++) scs[e][i] = 1.0f;
    }
}
#define NOTE(fn, l, e) fn((l), (e), g4s[e], u4s[e], d4s[e], scs[e], scs[e] + IH, scs[e] + 2 * IH)

/* ---- the rendezvous ------------------------------------------------------ */
static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
static int at_leave, leave_release;   /* the teardown parked in its slow window */
static int reinit_done, reinit_result;
static int losshome, loser_returned, teardown_complete, loser_saw_complete;
static int teardown_entries;

/* The teardown thread, with G.mx held, past the drains and before any
 * teardown work: the middle of the slow window the re-init used to walk into. */
static void drain_leave_hook(void) {
    pthread_mutex_lock(&mx);
    if (++teardown_entries > 1) {
        /* Only a caller that WON the teardown can reach this, so a second
         * arrival is the double teardown itself, named where it is decided. */
        fflush(stdout);
        fprintf(stderr, "  FAIL: the teardown body was entered %d times for one tier\n",
                teardown_entries);
        fflush(stderr);
        _exit(1);
    }
    at_leave = 1; pthread_cond_broadcast(&cv);
    while (!leave_release) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);
}

/* The teardown thread, once the storage is gone: the point a waiting loser is
 * released at, so "a loser returned before the teardown finished" is reported
 * by the teardown rather than guessed at afterwards. */
static void gate_freed_hook(void) {
    pthread_mutex_lock(&mx);
    teardown_complete = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
}

/* A second caller that arrives while the first is still tearing down. */
static void *loser(void *a) {
    (void)a;
    pthread_mutex_lock(&mx);
    losshome = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    qt_shutdown();                       /* must WAIT, not return */
    pthread_mutex_lock(&mx);
    loser_saw_complete = teardown_complete;
    loser_returned = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    return NULL;
}

static void *reinitter(void *a) {
    (void)a;
    int r = qt_init(1, NEXP, D, IH, NEXP, 2, 0, 1);
    pthread_mutex_lock(&mx);
    reinit_result = r; reinit_done = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    return NULL;
}

static void *tearder(void *a) { (void)a; qt_shutdown(); return NULL; }

/* A wedge must fail loudly rather than sit in CI. A backstop, never a verdict. */
static volatile int alive = 1;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { 60, 0 };
    nanosleep(&limit, NULL);
    if (alive) { (void)!write(2, "FAIL: re-init window test hung\n", sizeof("FAIL: re-init window test hung\n") - 1); _exit(1); }
    return NULL;
}

static int boot(void) {
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("CUDA_EXPERT_GB", "auto", 1);
    fake_ndev = 1;
    fake_uploads = 0; fake_frees = 0; fake_live_tensors = 0;
    fake_issue_hook = NULL; fake_upload_hook = NULL; fake_free_hook = NULL;
    return qt_init(1, NEXP, D, IH, NEXP, 2, 0, 1);
}

int main(int argc, char **argv) {
    int cycles = argc > 1 ? atoi(argv[1]) : 400;
    if (cycles < 1) cycles = 1;
    printf("qwen36 tier re-init window: no init may be accepted while a teardown runs\n");
    make_weights();

    pthread_t wd;
    if (pthread_create(&wd, NULL, watchdog, NULL) == 0) pthread_detach(wd);

    if (!boot()) { printf("  FAIL: tier did not start\n"); return 1; }
    /* Resident tensors on purpose: a re-init that memsets over a running
     * teardown loses the slot array, and the free count is how the test sees
     * that the teardown still happened exactly once. */
    for (int e = 0; e < NEXP; e++) NOTE(qt_note_block, 0, e);
    qt_fill_wait();
    int live = fake_live_tensors;
    check(live > 0, "no live tensors: the re-init would have nothing to destroy");

    /* -DQT_TEST_HOOKS on this rule; production objects never define these. */
    qt_test_drain_leave_hook = drain_leave_hook;
    qt_test_gate_freed_hook  = gate_freed_hook;

    pthread_t st, rt, lt;
    if (pthread_create(&st, NULL, tearder, NULL) != 0) { check(0, "spawn teardown"); return 1; }
    /* Park the teardown in the middle of its own slow window. */
    pthread_mutex_lock(&mx);
    while (!at_leave) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);

    check(G.on == 1,
          "G.on must not be cleared while the teardown is still running: a flag "
          "cleared at the TOP is what made this window readable as 'never started'");
    check(G.slot != NULL, "the storage must still be live inside the teardown window");

    /* ---- the re-init that must be refused ------------------------------- */
    if (pthread_create(&rt, NULL, reinitter, NULL) != 0) { check(0, "spawn reinit"); return 1; }
    pthread_mutex_lock(&mx);
    while (!reinit_done) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);
    check(reinit_result == 0,
          "qt_init during TEARING_DOWN must be refused (0), not accepted (1)");
    check(G.on == 1 && G.slot != NULL,
          "a refused re-init must leave the running teardown's state untouched");
    check(fake_live_tensors == live && fake_frees == 0,
          "a refused re-init must not free, upload or lose anything");

    /* ---- a concurrent second shutdown must WAIT -------------------------- */
    if (pthread_create(&lt, NULL, loser, NULL) != 0) { check(0, "spawn loser"); return 1; }
    pthread_mutex_lock(&mx);
    while (!losshome) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);
    /* The loser is inside qt_shutdown while the teardown is still parked. It
     * must not be back yet -- and we do not wait to find that out: we let the
     * teardown finish and then read the flag the loser recorded. */
    check(!loser_returned, "a losing qt_shutdown must not return while the teardown is still running");

    /* ---- let the teardown finish ---------------------------------------- */
    pthread_mutex_lock(&mx);
    leave_release = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    pthread_join(rt, NULL);
    pthread_join(lt, NULL);
    pthread_join(st, NULL);

    check(loser_returned, "the losing qt_shutdown must return once the teardown is done");
    check(loser_saw_complete,
          "a losing qt_shutdown returned before the teardown had completed: "
          "the shutdown is not synchronous");
    check(teardown_complete, "the teardown must reach its complete state");
    check(teardown_entries == 1, "the teardown body must be entered exactly once");
    check(fake_live_tensors == 0, "the teardown must free every live tensor");
    check(fake_frees == live, "the teardown must free each live tensor exactly once");
    check(G.on == 0 && G.slot == NULL && G.is_x == NULL,
          "the teardown must complete: tier off, storage gone");

    /* ---- the lifecycle has to be REUSABLE --------------------------------
     * A once-only claim that a re-init can hand back is exactly what would
     * show up here: the second cycle's teardown either does nothing (a claim
     * that is still set) or runs twice over freed storage.
     *
     * The counter is rebased every cycle: each of these runs its own teardown
     * and legitimately reaches the seam once, so the "entered twice" tripwire
     * is about a SECOND caller inside ONE teardown, not across cycles.
     *
     * Each cycle uploads and tears down REAL tensors, so a cycle whose teardown
     * ran twice would drive the live count negative or hand freed pointers back
     * to free() -- which is how a re-armable claim announces itself. */
    for (int i = 0; i < cycles; i++) {
        teardown_entries = 0;
        if (!qt_init(1, NEXP, D, IH, NEXP, 2, 0, 1)) {
            printf("  FAIL: sequential re-init refused at cycle %d after a completed teardown\n", i);
            fails++;
            break;
        }
        for (int e = 0; e < NEXP; e++) NOTE(qt_note_block, 0, e);
        qt_fill_wait();
        if (fake_live_tensors != live) {
            printf("  FAIL: cycle %d has %d live tensors, expected %d\n", i, fake_live_tensors, live);
            fails++;
            break;
        }
        qt_shutdown();
        if (fake_live_tensors != 0) {
            printf("  FAIL: cycle %d left %d live tensors\n", i, fake_live_tensors);
            fails++;
            break;
        }
        if (teardown_entries != 1) {
            printf("  FAIL: cycle %d entered the teardown %d times, expected 1\n", i, teardown_entries);
            fails++;
            break;
        }
    }
    check(fake_frees == live * (cycles + 1),
          "every cycle must free exactly its own live tensors, none twice");
    alive = 0;
    if (fails) { printf("test_qwen36_tier_reinit_window: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_reinit_window: ok (re-init refused mid-teardown, %d sequential cycles)\n", cycles);
    return 0;
}
