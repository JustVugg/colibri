/* A qt_shutdown that finds QT_DEAD must not answer from a count it snapshotted
 * OUTSIDE the teardown's exclusive section (#1564, review of 6917c3f6).
 *
 * The defect 6917c3f6 introduced while closing HIGH A. That commit made the
 * DEAD fast path free resident trunk storage instead of returning at once:
 *
 *     int state = qg_get(), trunk = qg_trunk;
 *     ...
 *     if(state==QT_LIVE || !trunk) return;
 *
 * and its message claimed the fast path "may only return when there is nothing
 * left to free". Under concurrency that claim is false, because qg_trunk is not
 * a count of trunk WORK -- it is a count of trunk TENSORS, and qt_dnproj_init
 * only moves it once the upload RETURNS:
 *
 *     if(!qt_gate_enter_trunk()) return 0;    <- qg_live++ : the caller is
 *                                                 counted from HERE
 *     int had = G_dnp[layer].t != NULL;
 *     int ok  = coli_cuda_tensor_upload(...); <- a caller is INSIDE here, and
 *                                                 qg_trunk has not moved
 *     else { ...; if(!had) qg_trunk++; }      <- only HERE does it move
 *     qt_gate_leave();
 *
 * So a qt_dnproj_init admitted at QT_DEAD that is parked inside the upload is
 * invisible to the fast path: it reads qg_trunk==0 and returns. The shutdown is
 * then over, and the parked init finishes, leaving a live G_dnp[].t behind a
 * completed qt_shutdown. The reviewer's deterministic interleaving:
 *
 *     state_before=0 shutdown_frees=0 live_tensors_after_shutdown=0
 *     live_tensors_final=1 qg_trunk=1 ESCAPE=1
 *
 * That is HIGH A's class again -- storage resident after the shutdown -- and the
 * same-shaped re-init is then answered by the backend's cached path with the
 * PREVIOUS generation's weights. The count is a plain int read on that line and
 * written at the increment with qg_mx held on NEITHER side, so a lost update
 * between two trunk inits admitted together is a second way to read a stale
 * zero; fake_cached_upload below models the cached path exactly.
 *
 * What is asserted here, and how:
 *
 *  - ESCAPE, the deterministic core: after an overlapping qt_dnproj_init and
 *    qt_shutdown, NOTHING is resident -- no live tensor, no G_dnp[].t, no
 *    ownership count, exactly one free. Checked after both threads have JOINED,
 *    so the verdict is not a timing judgement.
 *  - The same property over several rounds, so no single unlucky interleaving
 *    carries it.
 *  - The consequence, exactly as HIGH A stated it: a later same-shaped init
 *    must ALLOCATE, never inherit, and the layer must compute its own weights.
 *
 * There is ONE bounded wait per round and it only decides the ORDER in which
 * the shutdown and the parked upload are released -- never the verdict. It is
 * asymmetric on purpose: a correct teardown is blocked on qg_live for as long
 * as that upload is parked, so it cannot return inside the window however slow
 * the box is, and the wait can never manufacture a false ESCAPE. It only lets
 * the buggy teardown, which has nothing to wait for, be caught returning. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>
#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { PI = 8, PO = 4, LAYER = 5, ROUNDS = 24, BIAS_MS = 15 };
/* Two weight sets, same shape. gen A is 1..N, gen B is -1..-N, so a matmul
 * answered with the wrong one is a wrong answer and not a tolerance question. */
static int8_t qa[PI * PO], qb[PI * PO];
static float sa[PO], sb[PO], px[PI], py[PO];
static void expect_b(float *y) {
    for (int o = 0; o < PO; o++) {
        float a = 0.f;
        for (int i = 0; i < PI; i++) a += px[i] * (float)qb[(size_t)o * PI + i];
        y[o] = a * sb[o];
    }
}

/* ---- the seam: park a qt_dnproj_init INSIDE its upload ------------------- */
static pthread_mutex_t hx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  hc = PTHREAD_COND_INITIALIZER;
static int armed;         /* the next upload parks itself                   */
static int parked;        /* the upload is parked inside the fake backend   */
static int released;      /* let the parked upload run to completion        */
static int sd_returned;   /* qt_shutdown() came back                       */
static int sd_collected;  /* main has looked at that verdict               */

/* Called by the fake backend as the FIRST statement of every upload, before
 * the cached-tensor test and before anything is allocated -- so a thread parked
 * here has been admitted by qt_gate_enter_trunk (it holds a qg_live slot) and
 * has NOT yet reached the qg_trunk increment. That is exactly the window the
 * DEAD fast path cannot see. */
static void upload_hook(int fmt) {
    (void)fmt;
    pthread_mutex_lock(&hx);
    if (!armed) { pthread_mutex_unlock(&hx); return; }
    armed = 0;
    parked = 1;
    pthread_cond_broadcast(&hc);
    while (!released) pthread_cond_wait(&hc, &hx);
    pthread_mutex_unlock(&hx);
}

static int init_rc;
static void *init_thread(void *arg) {
    (void)arg;
    init_rc = qt_dnproj_init(LAYER, qa, sa, PI, PO, 0);
    return NULL;
}
static void *shutdown_thread(void *arg) {
    (void)arg;
    qt_shutdown();
    pthread_mutex_lock(&hx);
    sd_returned = 1;
    pthread_cond_broadcast(&hc);
    /* Hold the verdict where main can see it before this thread goes away, so
     * "the shutdown returned while the upload was still parked" is observed
     * rather than inferred from a later memory dump. */
    while (!sd_collected) pthread_cond_wait(&hc, &hx);
    pthread_mutex_unlock(&hx);
    return NULL;
}
static struct timespec deadline_ms(long ms) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    t.tv_sec  += ms / 1000;
    t.tv_nsec += (ms % 1000) * 1000000L;
    if (t.tv_nsec >= 1000000000L) { t.tv_sec++; t.tv_nsec -= 1000000000L; }
    return t;
}
static int returned_while_parked_any;   /* a shutdown came back mid-upload */
static volatile int alive = 1;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { 120, 0 };
    nanosleep(&limit, NULL);
    if (alive) { (void)!write(2, "FAIL: trunk drain test hung\n", 26); _exit(2); }
    return NULL;
}

/* One overlap. Returns 1 if the teardown let a tensor escape it. */
static int one_round(void) {
    pthread_mutex_lock(&hx);
    armed = 1; parked = 0; released = 0; sd_returned = 0; sd_collected = 0;
    pthread_mutex_unlock(&hx);
    fake_upload_hook = upload_hook;
    int frees_before = fake_frees, live_before = fake_live_tensors;

    pthread_t ti, ts;
    pthread_create(&ti, NULL, init_thread, NULL);
    pthread_mutex_lock(&hx);
    while (!parked) pthread_cond_wait(&hc, &hx);
    pthread_mutex_unlock(&hx);

    pthread_create(&ts, NULL, shutdown_thread, NULL);
    /* Let the shutdown run while the upload is still parked. A correct teardown
     * is blocked on qg_live here; the buggy one returns at once. */
    int returned_while_parked = 0;
    pthread_mutex_lock(&hx);
    if (!sd_returned) {
        struct timespec d = deadline_ms(BIAS_MS);
        while (!sd_returned && pthread_cond_timedwait(&hc, &hx, &d) != ETIMEDOUT) { }
        returned_while_parked = sd_returned;
    } else {
        returned_while_parked = 1;
    }
    released = 1;
    pthread_cond_broadcast(&hc);
    pthread_mutex_unlock(&hx);

    pthread_join(ti, NULL);
    pthread_mutex_lock(&hx);
    while (!sd_returned) pthread_cond_wait(&hc, &hx);
    sd_collected = 1;
    pthread_cond_broadcast(&hc);
    pthread_mutex_unlock(&hx);
    pthread_join(ts, NULL);
    fake_upload_hook = NULL;

    int freed = fake_frees - frees_before;
    int left  = fake_live_tensors - live_before;
    if (returned_while_parked) returned_while_parked_any = 1;
    printf("  round: returned_while_parked=%d init_rc=%d freed=%d live_left=%d "
           "dnp_t=%s qg_trunk=%d state=%d\n",
           returned_while_parked, init_rc, freed, left,
           G_dnp[LAYER].t ? "set" : "NULL", qg_trunk, qg_get());
    /* An init that was refused rather than raced has produced nothing to
     * escape, and would otherwise read as a vacuous pass. */
    check(init_rc == 1, "the trunk init was admitted, not refused at the door");
    /* The teardown owes this round exactly the tensor the init created. */
    return !(freed == 1 && left == 0 && G_dnp[LAYER].t == NULL && qg_trunk == 0);
}

int main(void) {
    printf("qwen36 tier trunk drain: a DEAD shutdown waits for the init it raced\n");
    for (int i = 0; i < PI * PO; i++) { qa[i] = (int8_t)(i + 1); qb[i] = (int8_t)(-(i + 1)); }
    for (int o = 0; o < PO; o++) { sa[o] = 1.0f; sb[o] = 1.0f; }
    for (int i = 0; i < PI; i++) px[i] = 1.0f;
    pthread_t wd;
    if (pthread_create(&wd, NULL, watchdog, NULL) == 0) pthread_detach(wd);
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("CUDA_EXPERT_GB", "auto", 1);
    /* The real backend's cached-tensor path, plus real fmt-1 arithmetic so
     * "which weights is this layer computing with" has an oracle here. */
    fake_cached_upload = 1; fake_dense_compute = 1;

    check(qg_get() == QT_DEAD, "the expert tier has never been started");
    int escapes = 0;
    for (int r = 0; r < ROUNDS; r++) {
        if (one_round()) { escapes++; break; }
    }
    /* The consequence, and the only part of this that is not the teardown's
     * own bookkeeping: a later same-shaped init must ALLOCATE and the layer must
     * compute ITS OWN weights. Run whether or not a round escaped -- on an
     * escape the slot is still occupied, and this is where that becomes the
     * previous generation's weights. */
    int hits_before = fake_cached_hits, ups_before = fake_uploads;
    int re = qt_dnproj_init(LAYER, qb, sb, PI, PO, 0);
    int stale = fake_cached_hits - hits_before;
    int realloc = fake_uploads - ups_before;
    memset(py, 0, sizeof py);
    int gpu = qt_dnproj_matmul(LAYER, py, px, PI, PO);
    float want[PO];
    expect_b(want);
    int wrong = 0;
    for (int o = 0; o < PO; o++) if (py[o] != want[o]) wrong = 1;
    printf("REINIT=%d STALE_TENSOR_REUSED=%d NEW_ALLOC=%d POST_REINIT_GPU=%d "
           "STALE_WEIGHTS=%d\n", re, stale, realloc, gpu, wrong);
    check(re == 1, "the projection re-uploads after the teardown");
    check(stale == 0, "no same-shaped init was answered by the cached path");
    check(realloc == 1, "the same-shaped init allocated a fresh tensor");
    check(gpu == 1 && !wrong, "the layer computes on the weights it was given");
    qt_shutdown();
    printf("ESCAPE=%d RETURNED_WHILE_PARKED=%d ROUNDS=%d\n",
           escapes, returned_while_parked_any, ROUNDS);
    check(escapes == 0, "no qt_dnproj_init escaped the qt_shutdown it overlapped");
    check(returned_while_parked_any == 0,
          "qt_shutdown never returned while a trunk caller was still inside");
    check(fake_live_tensors == 0, "no tensor outlived the teardown");
    alive = 0;
    if (fails) { printf("test_qwen36_tier_trunk_drain: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_trunk_drain: ok\n");
    return 0;
}
