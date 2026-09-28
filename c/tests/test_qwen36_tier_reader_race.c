/* A caller admitted into the tier must never be able to touch storage a
 * teardown has already freed (#1564, defect 1: the admission race).
 *
 * The defect. Twelve sibling entry points began with
 *
 *     if(!G.on) return 0;
 *     pthread_mutex_lock(&G.mx);
 *     ... read G.slot ... qs(layer,eid) is &G.slot[...]
 *
 * and G.on was read with NOTHING held across the two. A caller that had already
 * passed the test could be descheduled there; the teardown then ran to
 * completion (G.slot=NULL, then pthread_mutex_destroy(&G.mx)); and the caller
 * resumed into a destroyed mutex and a NULL slot array. Measured under gdb
 * non-stop:
 *
 *   Thread 4 "probe_reader" at qwen36_tier.c:854  (G.on already read as 1)
 *   teardown done: G.on=0  G.slot=(nil)
 *   SIGSEGV at qwen36_tier.c:855  int r = qs(layer,eid)->resident;
 *
 * So this is not "8 callers x 400 does not reproduce it" -- it needs
 * preemption inside a sub-millisecond window, and 8x400 is the wrong tool. It
 * is reproduced here DETERMINISTICALLY instead, through a seam, with no wall
 * clock anywhere.
 *
 * How the interleaving is forced. The reader is parked in a test-only hook that
 * sits exactly where the defect lives: after the admission decision, before
 * the first storage read. While it sits there it is a caller the tier has
 * admitted and has not served, and its G.mx is not taken.
 *
 * Then the teardown runs, and the test releases the reader from whichever
 * teardown-side event arrives FIRST:
 *
 *   - "the teardown is about to WAIT for the caller inside"  (the fixed tier):
 *     the release happens BEFORE the free, the reader reads live storage, and
 *     it finishes before the teardown can touch anything. Correct.
 *   - "the teardown has FREED the storage"  (the defect, where entering the
 *     tier is not counted so there is nothing to wait for): the release happens
 *     AFTER the free, and the reader resumes into the destroyed mutex and the
 *     NULL slot array -- SIGSEGV.
 *
 * Both are self-reports from the two ends of the teardown's exclusive section,
 * so the verdict never depends on how long anything took, and a teardown that
 * deletes the wait is caught by its silence rather than by a watchdog. The
 * mutant does not merely "fail a check": it dies in the reader, which is the
 * failure being repaired.
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

/* ---- the rendezvous ------------------------------------------------------
 * One mutex, one condvar, three self-reports. Which of the two teardown-side
 * events arrives first is the whole result, so the reader is released from
 * here and not from a timer. */
static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
static int reader_parked, reader_go;      /* the reader's own handshake      */
static int teardown_blocked, teardown_freed;
static unsigned event_seq;                /* monotonic: who observed storage */
static int reader_seq, freed_seq;
static int reader_val;

/* Fires on the READER thread, with the gate held and G.mx NOT taken: admitted,
 * storage untouched. This is the window `if(!G.on)` + lock leaves open. */
static void reader_pre_hook(void) {
    pthread_mutex_lock(&mx);
    reader_parked = 1; pthread_cond_broadcast(&cv);
    while (!reader_go) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);
}

/* Fires on the TEARDOWN thread, inside its exclusive entry, at the moment it is
 * about to wait for the callers still inside. A teardown with nothing to wait
 * for never fires it, and that silence IS the defect. */
static void gate_block_hook(void) {
    pthread_mutex_lock(&mx);
    teardown_blocked = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
}

/* Fires on the TEARDOWN thread once the storage is gone. */
static void gate_freed_hook(void) {
    int n = (int)__atomic_add_fetch(&event_seq, 1, __ATOMIC_ACQ_REL);
    pthread_mutex_lock(&mx);
    freed_seq = n;
    teardown_freed = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
}

static void *reader(void *a) {
    (void)a;
    reader_val = qt_is_resident(0, 0);
    int n = (int)__atomic_add_fetch(&event_seq, 1, __ATOMIC_ACQ_REL);
    pthread_mutex_lock(&mx); reader_seq = n; pthread_mutex_unlock(&mx);
    return NULL;
}
static void *tearder(void *a) { (void)a; qt_shutdown(); return NULL; }

/* A wedged teardown must fail loudly rather than sit in CI. Same shape as the
 * watchdog in the other tier tests: a backstop, never a verdict. */
static volatile int alive = 1;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { 60, 0 };
    nanosleep(&limit, NULL);
    if (alive) { (void)!write(2, "FAIL: reader race test hung\n", sizeof("FAIL: reader race test hung\n") - 1); _exit(1); }
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

int main(void) {
    printf("qwen36 tier reader race: an admitted caller must not resume into a freed tier\n");
    make_weights();

    pthread_t wd;
    if (pthread_create(&wd, NULL, watchdog, NULL) == 0) pthread_detach(wd);

    if (!boot()) { printf("  FAIL: tier did not start\n"); return 1; }
    /* Resident experts on purpose. An empty tier has nothing to read, so the
     * reader would find a NULL deref no matter what and the test would prove
     * nothing about the interleaving. */
    for (int e = 0; e < NEXP; e++) NOTE(qt_note_block, 0, e);
    qt_fill_wait();
    int live = fake_live_tensors;
    check(live > 0, "no live tensors: the reader has nothing to read");

    /* The control read, with every seam still inert: this is the value the
     * reader must still see after it has been parked and released. */
    int expect = qt_is_resident(0, 0);
    check(expect == 1, "expert 0 must be resident before the race");
    if (!live || expect != 1) { alive = 0; printf("test_qwen36_tier_reader_race: %d failure(s)\n", fails); return 1; }

    /* -DQT_TEST_HOOKS on this rule; NULL elsewhere, and the production objects
     * never define them at all. */
    qt_test_reader_pre_hook = reader_pre_hook;
    qt_test_gate_block_hook  = gate_block_hook;
    qt_test_gate_freed_hook  = gate_freed_hook;

    pthread_t rt, st;
    if (pthread_create(&rt, NULL, reader, NULL) != 0) { check(0, "spawn reader"); return 1; }
    pthread_mutex_lock(&mx);
    while (!reader_parked) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);
    /* The reader is now inside the tier: admitted, G.mx not taken, no storage
     * read yet. That is precisely the state the defect abandons a caller in. */
    if (pthread_create(&st, NULL, tearder, NULL) != 0) { check(0, "spawn teardown"); return 1; }

    pthread_mutex_lock(&mx);
    while (!teardown_blocked && !teardown_freed) pthread_cond_wait(&cv, &mx);
    int freed_first = teardown_freed;      /* 1 = the defect's world */
    reader_go = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);

    pthread_join(rt, NULL);
    pthread_join(st, NULL);

    check(!freed_first,
          "the teardown freed the storage while a caller was still inside the tier");
    check(teardown_blocked,
          "qt_shutdown must wait for the caller that is already inside the tier");
    check(reader_seq > 0 && freed_seq > 0 && reader_seq < freed_seq,
          "the reader must have finished before the teardown freed the storage");
    check(reader_val == expect,
          "the reader must return the residency bit it read from LIVE storage");
    check(fake_live_tensors == 0, "the teardown must still free every live tensor");
    check(fake_frees == live, "the teardown must free each live tensor exactly once");
    check(G.on == 0 && G.slot == NULL && G.is_x == NULL,
          "the teardown must complete: tier off, storage gone");

    /* The lifecycle has to be reusable, or the fix would have traded a crash
     * for a tier that can only be built once per process. */
    check(boot(), "qt_init after a completed teardown must build a new tier");
    check(qt_is_resident(0, 0) == 0, "the new tier starts empty");
    qt_shutdown();

    alive = 0;
    if (fails) { printf("test_qwen36_tier_reader_race: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_reader_race: ok (the reader observed live storage, the teardown waited for it)\n");
    return 0;
}
