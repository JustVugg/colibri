/* qt_shutdown called from several threads at once must tear the tier down
 * exactly once (#1564).
 *
 * The defect: qt_shutdown began with a bare `if(!G.on) return;` and cleared
 * G.on only near the END of the teardown. That is check-then-act with no lock
 * across it, and it is not a theoretical one -- two threads reaching
 * qt_shutdown together both read "on", both fell through, and both ran the
 * whole teardown: pthread_join(G.th) twice, coli_cuda_tensor_free on every
 * expert slot twice, free() of G.slot/G.is_x/G.fill_order/G.heat0 twice and
 * pthread_cond_destroy + pthread_mutex_destroy of the same objects twice.
 * A sequential second shutdown passed, which is exactly why it survived: the
 * second caller always arrives after the first has finished, and the flag is
 * already clear by then. Concurrent is the only case that breaks, and it
 * broke as a SIGSEGV (exit 139) from the double free.
 *
 * What is checked, per iteration, on a freshly booted tier:
 *   - the N callers overlap at a barrier, so they really are simultaneous
 *     rather than accidentally serialised by the scheduler;
 *   - the teardown body is entered EXACTLY once. This is counted at
 *     qt_test_drain_leave_hook, which sits past the once-only claim, so the
 *     verdict comes from the code that decides it and not from a wall clock.
 *     It has to be counted there: with the claim deleted the losing callers
 *     carry on to pthread_join(G.th) a thread the winner already reaped, and
 *     glibc never returns from that second join -- so every verdict obtained
 *     after the fact arrives only via the 60 s watchdog, and reports a hang
 *     rather than the double teardown that caused it;
 *   - every tensor live before the shutdown is freed EXACTLY once, which is
 *     the double-free assertion: a second teardown drives the live count
 *     negative, or aborts inside free() before it can be observed;
 *   - the teardown is complete when they are all home (tier off, G.slot gone).
 *
 * And once at the end, on a live tier: a second qt_init must be refused. G
 * holds the teardown latch and qt_init rebuilds G from zero, so an init
 * allowed to run against a live tier hands the once-only claim straight back
 * (#1564) while the old uploader is still parked on the mutex that same
 * memset wipes.
 *
 * The experts are made RESIDENT first, deliberately: with nothing uploaded
 * there is nothing to double-free and the test would pass against the defect.
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

/* ---- how many callers got INTO the teardown -------------------------------
 * The claim under test is qt_test_drain_leave_hook: it sits after the G.waiters
 * drain and before any teardown work, and only a caller that WON the once-only
 * claim can reach it. So one teardown means exactly one arrival, whatever the
 * callers do to each other -- a self-report, not a clock.
 *
 * Counting after the fact was not enough. With the claim deleted the extra
 * callers do not merely arrive late, they go on to pthread_join(G.th) a
 * thread the first one already reaped, and glibc blocks forever on that
 * double join -- so the verdict was only ever reachable through the 60 s
 * watchdog, i.e. through a wall clock, which is the same mistake round 2 made
 * and the reason this file was judged unsound. A second arrival is already
 * past the point of no return, so the seam reports it the moment it happens
 * and ends the process: the failure is named, numbered and instant. */
static int teardown_entries;
static void count_teardown_entry(void) {
    if (__atomic_add_fetch(&teardown_entries, 1, __ATOMIC_ACQ_REL) > 1) {
        fflush(stdout);
        fprintf(stderr,
                "  FAIL: the teardown body was entered %d times for one tier: "
                "qt_shutdown ran its teardown more than once (#1564)\n",
                teardown_entries);
        fflush(stderr);
        _exit(1);
    }
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

/* ---- start gate: every caller is released at the same instant -------------
 * Hand-rolled rather than pthread_barrier_t, which MinGW does not have, and
 * it matches the mutex/condvar style the other tier tests already use. The
 * point is that the N callers really do overlap: without a shared release they
 * would arrive in sequence and the second one would just see the tier off. */
static pthread_mutex_t bar_mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  bar_cv = PTHREAD_COND_INITIALIZER;
static int bar_count, bar_target;
static void bar_arrive(void) {
    pthread_mutex_lock(&bar_mx);
    if (++bar_count >= bar_target) { bar_count = 0; pthread_cond_broadcast(&bar_cv); }
    else while (bar_count) pthread_cond_wait(&bar_cv, &bar_mx);
    pthread_mutex_unlock(&bar_mx);
}

/* Optional per-caller offset, so a later caller lands INSIDE the first one's
 * teardown window rather than colliding with it at the `if(!G.on)` check.
 * That is what makes the double free itself -- and not merely the colliding
 * pthread_join -- the thing that shows up. */
static long stagger_us;
static void *worker(void *a) {
    long idx = (long)a;
    bar_arrive();
    if (stagger_us && idx) {
        struct timespec ts = {0, stagger_us * 1000 * idx};
        nanosleep(&ts, NULL);
    }
    qt_shutdown();
    return NULL;
}

/* A wedged teardown must fail loudly rather than sit in CI. Same shape as the
 * watchdog in test_qwen36_tier_shutdown: a thread that sleeps, then exits. */
static volatile int alive = 1;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { 60, 0 };
    nanosleep(&limit, NULL);
    if (alive) { (void)!write(2, "FAIL: concurrent qt_shutdown hung\n", 34); _exit(1); }
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
    int nthreads = argc > 1 ? atoi(argv[1]) : 8;
    int iters    = argc > 2 ? atoi(argv[2]) : 400;
    stagger_us   = argc > 3 ? atol(argv[3]) : 0;
    if (nthreads < 2) nthreads = 2;
    if (nthreads > 64) nthreads = 64;
    make_weights();
    bar_target = nthreads;

    printf("qwen36 tier concurrent shutdown\n");
    printf(" %d callers on qt_shutdown at a barrier x %d iterations, stagger %ld us\n",
           nthreads, iters, stagger_us);

    /* Compiled in by -DQT_TEST_HOOKS on this rule; NULL elsewhere, and the
     * production objects never define them at all. */
    qt_test_drain_leave_hook = count_teardown_entry;

    pthread_t wd;
    if (pthread_create(&wd, NULL, watchdog, NULL) == 0) pthread_detach(wd);

    for (int it = 0; it < iters; it++) {
        if (!boot()) { printf("  FAIL: tier did not start at iteration %d\n", it); fails++; break; }
        for (int e = 0; e < NEXP; e++) NOTE(qt_note_block, 0, e);
        qt_fill_wait();
        int live = fake_live_tensors, frees_before = fake_frees;
        if (live <= 0) {
            printf("  FAIL: iteration %d has no live tensors to double-free\n", it);
            fails++; break;
        }

        pthread_t t[64];
        teardown_entries = 0;
        for (int i = 0; i < nthreads; i++)
            if (pthread_create(&t[i], NULL, worker, (void *)(long)i) != 0) { printf("  FAIL: spawn\n"); fails++; }
        for (int i = 0; i < nthreads; i++) pthread_join(t[i], NULL);

        /* The latch's own verdict, read where it is decided. Reaching here at
         * all means the hook let it through, so this is the belt to the hook's
         * braces -- and it still catches a latch that let a second caller past
         * a hook the mutant had also removed. */
        if (teardown_entries != 1) {
            printf("  FAIL: iteration %d entered the teardown %d times, expected 1\n",
                   it, teardown_entries);
            fails++;
            break;
        }

        /* A teardown that ran twice frees every tensor twice: the second pass
         * hands the same pointers to free() a second time. Here that shows up
         * as a count that cannot be right. */
        int freed = fake_frees - frees_before;
        if (freed != live) {
            printf("  FAIL: iteration %d freed %d tensors for %d live ones"
                   " (teardown ran %d times)\n", it, freed, live, freed / live);
            fails++;
            break;
        }
        if (fake_live_tensors != 0) {
            printf("  FAIL: iteration %d left %d live tensors\n", it, fake_live_tensors);
            fails++; break;
        }
        if (G.on || G.slot || G.is_x) {
            printf("  FAIL: iteration %d: teardown incomplete (on=%d slot=%p is_x=%p)\n",
                   it, G.on, (void *)G.slot, (void *)G.is_x);
            fails++; break;
        }
    }

    /* A second qt_init must not hand the once-only claim back (#1564).
     * G.teardown lives inside G and qt_init rebuilds G from zero, so an init
     * that is allowed to run against a live tier re-arms the latch the barrier
     * loop above is there to prove is held -- and re-arms it while the old
     * uploader is still parked on the mutex the same memset wipes. The refusal
     * has to leave the running tier alone, not just return 0, so both halves
     * are checked: the init is refused, and the teardown that follows is still
     * the only one. */
    if (!fails) {
        int live;
        if (!boot()) check(0, "tier did not start for the second-init case");
        else {
            for (int e = 0; e < NEXP; e++) NOTE(qt_note_block, 0, e);
            qt_fill_wait();
            live = fake_live_tensors;
            check(live > 0, "second-init: no live tensors to protect");
            check(qt_init(1, NEXP, D, IH, NEXP, 2, 0, 1) == 0,
                  "a second qt_init while the tier is live must be refused, not memset over the running uploader");
            check(G.on && G.slot && qt_ready() && fake_live_tensors == live,
                  "the refused second init must leave the live tier exactly as it was");

            teardown_entries = 0;
            qt_shutdown();
            check(teardown_entries == 1,
                  "after a refused second init the teardown must still run exactly once: the latch was re-armed");
            check(fake_live_tensors == 0,
                  "after a refused second init the teardown must still free every live tensor");
        }
    }

    alive = 0;
    if (fails) { printf("test_qwen36_tier_shutdown_concurrent: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_shutdown_concurrent: ok (%d callers x %d iterations, no segfault, no double free)\n",
           nthreads, iters);
    return 0;
}
