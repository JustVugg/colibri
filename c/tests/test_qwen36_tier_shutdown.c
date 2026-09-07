/* qt_shutdown hangs when a group is open and a swap is queued (#1340).
 *
 * The defect: qt_shutdown sets G.th_stop and signals only G.cv, then
 * pthread_join()s the uploader. But the uploader's LFRU victim path waits on
 * G.cv_take -- "while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,
 * &G.mx)" -- and nothing ever broadcasts G.cv_take from qt_shutdown. If a
 * caller issued a group (qt_issue, which sets G.issue_open=1) and never took
 * it (qt_take is what clears issue_open and broadcasts G.cv_take), a queued
 * swap parks the uploader on that wait forever, and qt_shutdown's
 * pthread_join never returns.
 *
 * This test reproduces exactly that sequence with the fake CUDA backend
 * (tests/qwen36_fake_cuda.h, no GPU needed): budget one expert in, issue it
 * without taking it, queue an LFRU swap directly (what qt_lfru_tick_locked
 * does), then call qt_shutdown() under a 10 s watchdog so a hang fails loudly
 * instead of wedging CI. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

static int issue_ok(int device, int count, const float *x) {
    (void)device; (void)count; (void)x; return 1;
}

/* A hang must fail, not sit forever in CI. alarm()/SIGALRM would do it on
 * POSIX, but MinGW has neither, and the Windows job is the one that proved
 * it. A detached thread that sleeps and then checks a flag does the same
 * job everywhere the test already builds (it uses pthread and nanosleep). */
static volatile int shutdown_done = 0;
static void *hang_watchdog(void *arg) {
    (void)arg;
    for (int i = 0; i < 100 && !shutdown_done; i++) {   /* 100 x 100 ms = 10 s */
        struct timespec ts = {0, 100000000};
        nanosleep(&ts, NULL);
    }
    if (!shutdown_done) {
        (void)!write(2, "FAIL: qt_shutdown hung\n", 23);
        _exit(1);
    }
    return NULL;
}

/* Poll a G.mx-guarded predicate without a busy spin. */
static int wait_for_resident(int layer, int eid, int want, int max_polls) {
    for (int i = 0; i < max_polls; i++) {
        pthread_mutex_lock(&G.mx);
        int r = qs(layer, eid)->resident;
        pthread_mutex_unlock(&G.mx);
        if (r == want) return 1;
        struct timespec ts = {0, 2000000};   /* 2 ms */
        nanosleep(&ts, NULL);
    }
    return 0;
}

int main(void) {
    enum { D = 64 };
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    /* budget for one expert and a half, at the allocator's granularity (three
     * int4 matrices of D*32/2 bytes plus three scale tables, each charged
     * what cudaMalloc takes): enough for one expert, not two */
    size_t exp_bytes = 3 * dev_alloc_footprint((size_t)D * 32 / 2) + 3 * dev_alloc_footprint((2 * 32 + D) / 3 * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    fake_ndev = 1;

    if (!qt_init(1, 2, D, 32, 2, 1, 0 /* per-row */, 1 /* int4 */)) {
        printf("  FAIL: il tier non parte\n");
        return 1;
    }

    static unsigned char g4[2][D * 32 / 2], u4[2][D * 32 / 2], d4[2][D * 32 / 2];
    static float sc[2][2 * 32 + D];
    for (int eid = 0; eid < 2; eid++) {
        memset(g4[eid], (unsigned char)(eid + 1), sizeof g4[eid]);
        memset(u4[eid], (unsigned char)(eid + 2), sizeof u4[eid]);
        memset(d4[eid], (unsigned char)(eid + 3), sizeof d4[eid]);
        for (int i = 0; i < 2 * 32 + D; i++) sc[eid][i] = 1.0f;
    }
    /* Expert 0 fits the budget and gets uploaded; expert 1's pointers are
     * stored (needed for the later swap) but the budget refuses its upload. */
    qt_note(0, 0, g4[0], u4[0], d4[0], sc[0], sc[0] + 32, sc[0] + 2 * 32);
    qt_note(0, 1, g4[1], u4[1], d4[1], sc[1], sc[1] + 32, sc[1] + 2 * 32);
    wait_for_resident(0, 0, 1, 500);
    check(qt_is_resident(0, 0), "expert 0 should have fit the one-expert budget");
    check(!qt_is_resident(0, 1), "expert 1 should have been refused by the exhausted budget");

    /* Open a group on the resident expert and deliberately never take it --
     * that is what leaves G.issue_open set going into shutdown. */
    fake_issue_hook = issue_ok;
    float x[D]; for (int i = 0; i < D; i++) x[i] = (float)i;
    int resident_eid = 0;
    uint32_t mask = qt_issue(0, &resident_eid, 1, x);
    check(mask == 1u, "issuing the resident expert should set mask bit 0");

    /* Queue an LFRU swap directly -- exactly what qt_lfru_tick_locked does. */
    pthread_mutex_lock(&G.mx);
    qs(0, resident_eid)->resident = 0;
    int queued = enqueue_locked(0, 1, 0, resident_eid, 0);
    pthread_mutex_unlock(&G.mx);
    check(queued, "the swap should have been enqueued (queue has room, victim provided)");

    /* Give the uploader a moment to dequeue the swap and park on the victim
     * wait (poll G.qn==0 under the lock, up to 1s). */
    int parked = 0;
    for (int i = 0; i < 500; i++) {
        pthread_mutex_lock(&G.mx);
        int qn = G.qn;
        pthread_mutex_unlock(&G.mx);
        if (qn == 0) { parked = 1; break; }
        struct timespec ts = {0, 2000000};
        nanosleep(&ts, NULL);
    }
    check(parked, "the uploader should have dequeued the swap within 1s");

    pthread_t watchdog;
    if (pthread_create(&watchdog, NULL, hang_watchdog, NULL) == 0)
        pthread_detach(watchdog);
    qt_shutdown();
    shutdown_done = 1;
    check(!G.on, "shutdown_returns_while_a_group_is_open");
    /* The abandoned swap must leave the victim exactly as the open group left
     * it, and must not have driven the incoming expert's upload after
     * shutdown began. The uploader is already joined here, so reading G
     * needs no lock. */
    check(qs(0, resident_eid)->resident && qs(0, resident_eid)->tg,
          "shutdown_abandons_the_swap_instead_of_freeing_the_victim");
    check(!qs(0, 1)->queued && !qs(0, 1)->resident && G.uploads == 1,
          "shutdown_abandons_the_swap_instead_of_uploading_the_incoming_expert");

    if (fails) { printf("test_qwen36_tier_shutdown: %d fallimenti\n", fails); return 1; }
    printf("test_qwen36_tier_shutdown: ok\n");
    return 0;
}
