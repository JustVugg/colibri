/* qt_fill_wait() returns before the last upload has landed.
 *
 * The defect: qt_fill_wait waited for the upload QUEUE to drain (G.qn == 0).
 * The uploader frees the ring slot when it DEQUEUES an entry, before it calls
 * the backend, so the queue is empty while the last expert is still being
 * copied to the device. Whoever called qt_fill_wait then sees that expert as
 * queued=1, resident=0 -- and the engine's warmstart frees the RAM int8 copy
 * of every planned expert right after qt_fill_wait returns, trusting that
 * they are all in VRAM. #1360 met this as a one-in-fifteen failure of
 * test_qwen36_tier_multidev and papered over it in the test with a poll.
 *
 * This test makes the race deterministic instead of probabilistic: the fake
 * backend's upload hook sleeps a few milliseconds per tensor, so the uploader
 * is always mid-copy when the queue empties. With the old wait the last
 * expert is caught queued=1 every run; with the fix (an in-flight count that
 * the uploader decrements only after the slot is resident) qt_fill_wait
 * blocks until every enqueued expert has actually landed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

#include "../compat.h"   /* setenv: MinGW has none, and qwen36_tier.c does not
                          * pull it in the way an engine .c does */

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

/* 3 ms per tensor, 9 ms per expert: long against the microseconds the
 * dequeue-to-upload gap takes, short against the test budget. */
static void slow_upload(int fmt) {
    (void)fmt;
    struct timespec ts = {0, 3000000};
    nanosleep(&ts, NULL);
}

int main(void) {
    enum { NL = 1, NE = 8, D = 64, IH = 32, TOPK = 2 };
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    fake_upload_hook = slow_upload;

    if (!qt_init(NL, NE, D, IH, NE, TOPK, 0 /* per-row */, 1 /* int4 */)) {
        printf("  FAIL: the tier should start on the fake backend\n");
        return 1;
    }

    static unsigned char g4[NE][D * IH / 2], u4[NE][D * IH / 2], d4[NE][D * IH / 2];
    static float sc[NE][2 * IH + D];   /* gs=0 -> sc_gu=IH, sc_d=D */
    for (int eid = 0; eid < NE; eid++) {
        memset(g4[eid], (unsigned char)(eid + 1), sizeof g4[eid]);
        memset(u4[eid], (unsigned char)(eid + 2), sizeof u4[eid]);
        memset(d4[eid], (unsigned char)(eid + 3), sizeof d4[eid]);
        for (int i = 0; i < 2 * IH + D; i++) sc[eid][i] = 1.0f;
        qt_note_block(0, eid, g4[eid], u4[eid], d4[eid], sc[eid], sc[eid] + IH, sc[eid] + 2 * IH);
    }

    qt_fill_wait();

    /* The property, read under the tier's own lock the instant the wait
     * returns: nothing is still queued, and everything we enqueued is
     * resident. No polling, no sleeping -- the wait IS the guarantee. */
    int queued = 0, resident = 0;
    pthread_mutex_lock(&G.mx);
    for (int eid = 0; eid < NE; eid++) { queued += qs(0, eid)->queued; resident += qs(0, eid)->resident; }
    pthread_mutex_unlock(&G.mx);
    check(queued == 0, "an expert is still queued when qt_fill_wait returns");
    check(resident == NE, "not every enqueued expert is resident when qt_fill_wait returns");
    check(fake_uploads == 3 * NE, "each expert should have uploaded exactly three tensors by the time the wait returns");

    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_fill_wait: %d failures\n", fails); return 1; }
    printf("test_qwen36_tier_fill_wait: ok\n");
    return 0;
}
