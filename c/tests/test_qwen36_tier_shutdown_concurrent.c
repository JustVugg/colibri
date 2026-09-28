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
 *   - every tensor live before the shutdown is freed EXACTLY once, which is
 *     the double-free assertion: a second teardown drives the live count
 *     negative, or aborts inside free() before it can be observed;
 *   - the teardown is complete when they are all home (tier off, G.slot gone).
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
        for (int i = 0; i < nthreads; i++)
            if (pthread_create(&t[i], NULL, worker, (void *)(long)i) != 0) { printf("  FAIL: spawn\n"); fails++; }
        for (int i = 0; i < nthreads; i++) pthread_join(t[i], NULL);

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

    alive = 0;
    if (fails) { printf("test_qwen36_tier_shutdown_concurrent: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_shutdown_concurrent: ok (%d callers x %d iterations, no segfault, no double free)\n",
           nthreads, iters);
    return 0;
}
