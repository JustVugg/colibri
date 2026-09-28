/* A qt_init must never queue for the exclusive section behind a running
 * teardown's back (#1564, review of c33679dc, HIGH 1).
 *
 * The defect. qt_init called qt_gate_xenter() FIRST and only looked at the
 * lifecycle state once it held the section. With a caller still counted inside
 * the tier during half one of a teardown, that init raised qg_want_x and parked;
 * the teardown then reached its own qt_gate_xenter(), found qg_want_x taken and
 * returned -- "unreachable" -- with the state stuck at QT_TEARING_DOWN and every
 * tensor still allocated. qt_shutdown returning is supposed to mean the
 * teardown is complete; here it meant nothing had been freed at all.
 *
 * The interleaving is forced through seams, never a clock:
 *   1. a reader is admitted and parks on qt_test_reader_pre_hook (qg_live=1);
 *   2. the teardown starts and parks on qt_test_drain_leave_hook, i.e. in
 *      TEARING_DOWN, before its exclusive half;
 *   3. qt_init is issued. It either returns (refused at the door: right) or
 *      queues -- qt_test_gate_block_hook fires on the init thread (the defect);
 *   4. the teardown is released. It either queues behind the reader
 *      (gate_block_hook again: right) or comes straight back (the defect);
 *   5. the reader is released and everything is joined.
 * Each step waits on a self-report from the code that decides it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"
static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}
enum { D = 64, IH = 32, NEXP = 4 };
#define MB4 (D * IH / 2)
#define NSC (2 * IH + D)
static unsigned char g4s[NEXP][MB4], u4s[NEXP][MB4], d4s[NEXP][MB4];
static float scs[NEXP][NSC];
static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
static int reader_parked, reader_release;
static int td_parked, td_release, td_returned;
static int init_done, init_result, init_queued;
static int td_queued;
static pthread_t init_tid, td_tid;
static int td_state_at_return, td_on_at_return, td_slot_at_return;
static void reader_pre_hook(void) {
    pthread_mutex_lock(&mx);
    reader_parked = 1; pthread_cond_broadcast(&cv);
    while (!reader_release) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);
}
static void drain_leave_hook(void) {
    pthread_mutex_lock(&mx);
    td_parked = 1; pthread_cond_broadcast(&cv);
    while (!td_release) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);
}
/* Runs with qg_mx held, on whichever thread is about to wait for the gate. */
static void gate_block_hook(void) {
    pthread_mutex_lock(&mx);
    if (pthread_equal(pthread_self(), init_tid)) init_queued = 1;
    if (pthread_equal(pthread_self(), td_tid)) td_queued = 1;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
}
static void *reader(void *a) { (void)a; (void)qt_is_resident(0, 0); return NULL; }
static void *initter(void *a) {
    (void)a;
    int r = qt_init(1, NEXP, D, IH, NEXP, 2, 0, 1);
    pthread_mutex_lock(&mx);
    init_result = r; init_done = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    return NULL;
}
static void *tearer(void *a) {
    (void)a;
    qt_shutdown();
    pthread_mutex_lock(&mx);
    td_state_at_return = qg_get(); td_on_at_return = G.on; td_slot_at_return = G.slot != NULL;
    td_returned = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    return NULL;
}
static volatile int alive = 1;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { 60, 0 };
    nanosleep(&limit, NULL);
    if (alive) { (void)!write(2, "FAIL: init-order test hung\n", 27); _exit(2); }
    return NULL;
}
#define WAIT_UNTIL(cond) do { pthread_mutex_lock(&mx); while (!(cond)) pthread_cond_wait(&cv, &mx); pthread_mutex_unlock(&mx); } while (0)
int main(void) {
    printf("qwen36 tier init order: an init must not queue behind a teardown\n");
    for (int e = 0; e < NEXP; e++) {
        memset(g4s[e], e + 1, MB4); memset(u4s[e], e + 2, MB4); memset(d4s[e], e + 3, MB4);
        for (int i = 0; i < NSC; i++) scs[e][i] = 1.0f;
    }
    pthread_t wd;
    if (pthread_create(&wd, NULL, watchdog, NULL) == 0) pthread_detach(wd);
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("CUDA_EXPERT_GB", "auto", 1);
    if (!qt_init(1, NEXP, D, IH, NEXP, 2, 0, 1)) { printf("  FAIL: tier did not start\n"); return 1; }
    for (int e = 0; e < NEXP; e++)
        qt_note_block(0, e, g4s[e], u4s[e], d4s[e], scs[e], scs[e] + IH, scs[e] + 2 * IH);
    qt_fill_wait();
    int live = fake_live_tensors;
    check(live > 0, "resident tensors exist, so a skipped teardown is visible");
    qt_test_reader_pre_hook = reader_pre_hook;
    qt_test_drain_leave_hook = drain_leave_hook;
    qt_test_gate_block_hook = gate_block_hook;
    pthread_t rt;
    pthread_create(&rt, NULL, reader, NULL);
    WAIT_UNTIL(reader_parked);                       /* qg_live == 1 */
    pthread_mutex_lock(&mx);
    pthread_create(&td_tid, NULL, tearer, NULL);
    pthread_mutex_unlock(&mx);
    WAIT_UNTIL(td_parked);                           /* state == TEARING_DOWN */
    check(qg_get() == QT_TEARING_DOWN, "the teardown is parked inside TEARING_DOWN");
    pthread_mutex_lock(&mx);
    pthread_create(&init_tid, NULL, initter, NULL);
    pthread_mutex_unlock(&mx);
    WAIT_UNTIL(init_done || init_queued);
    printf("INIT_QUEUED_BEFORE_SHUTDOWN_XENTER=%d\n", init_queued);
    check(!init_queued, "qt_init queued for the exclusive section during TEARING_DOWN");
    pthread_mutex_lock(&mx);
    td_release = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    WAIT_UNTIL(td_returned || td_queued);
    int returned_early = td_returned;
    if (returned_early)
        printf("SHUTDOWN_RETURN_STATE=%d\nSHUTDOWN_RETURN_G_ON=%d\nSHUTDOWN_RETURN_SLOT_LIVE=%d\n",
               td_state_at_return, td_on_at_return, td_slot_at_return);
    check(!returned_early,
          "qt_shutdown returned before half two: it must wait for the reader, then free");
    pthread_mutex_lock(&mx);
    reader_release = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    pthread_join(rt, NULL); pthread_join(td_tid, NULL); pthread_join(init_tid, NULL);
    printf("INIT_RESULT=%d\n", init_result);
    check(init_result == 0, "qt_init during TEARING_DOWN is refused");
    check(qg_get() == QT_DEAD, "after qt_shutdown returns the state is QT_DEAD");
    check(G.slot == NULL && G.on == 0, "after qt_shutdown returns the storage is gone");
    check(fake_live_tensors == 0, "every resident tensor was freed");
    qt_test_reader_pre_hook = NULL; qt_test_drain_leave_hook = NULL; qt_test_gate_block_hook = NULL;
    /* A tier stuck in TEARING_DOWN would park the qt_shutdown below forever. */
    if (fails) { printf("test_qwen36_tier_init_order: %d failure(s)\n", fails); return 1; }
    /* and the lifecycle is still reusable */
    check(qt_init(1, NEXP, D, IH, NEXP, 2, 0, 1) == 1, "a fresh init after the teardown is accepted");
    qt_shutdown();
    check(qg_get() == QT_DEAD, "and tears down again");
    alive = 0;
    if (fails) { printf("test_qwen36_tier_init_order: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_init_order: ok\n");
    return 0;
}
