/* The tier's per-device input-replica block overruns G.is_x (#1339).
 *
 * The defect: qt_init allocated G.is_x as 32*D floats total (one device's
 * worth), but qt_issue strides each device's block by 8*D floats and then
 * writes up to 32 rows into it -- device di's block starts at 8*di*D but can
 * span 32*D floats, so any di>0 with a full 32-row issue writes past both its
 * own slice and the end of the allocation. With two devices and 32 resident
 * experts routed to device 1, device 1's block starts at 8*D and ends at
 * 8*D+32*D = 40*D, well past the 32*D-float buffer -- silent heap corruption
 * that only ASan (or a real GPU driver) would ever complain about.
 *
 * This uses the same fake CUDA backend as test_qwen36_tier_int8.c
 * (tests/qwen36_fake_cuda.h) with fake_ndev=2 and a fake_issue_hook that
 * records exactly which (device, x pointer, row count) each issue call used,
 * so the test can check the recorded blocks against the real allocation
 * bounds without a GPU or ASan. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { MAX_REC = 8 };
static struct { int device, count; const float *x; } records[MAX_REC];
static int nrec;
static int record_issue(int device, int count, const float *x) {
    if (nrec < MAX_REC) {
        records[nrec].device = device; records[nrec].count = count; records[nrec].x = x;
        nrec++;
    }
    return 1;
}

int main(void) {
    enum { NL = 1, NE = 64, D = 64, IH = 32, TOPK = 32 };
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0,1", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    fake_ndev = 2;

    if (!qt_init(NL, NE, D, IH, NE, TOPK, 0 /* per-row */, 1 /* int4 */)) {
        printf("  FAIL: il tier non parte con due device finti\n");
        return 1;
    }

    /* Make every expert resident: qt_note_block blocks on queue space, so a
     * single qt_fill_wait() after the loop is enough to drain the rest. */
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
    /* qt_fill_wait returns when the QUEUE is empty; the expert the uploader
     * dequeued last is still queued=1 (not yet resident) until its upload
     * returns. Checking residency right here raced that upload and failed
     * about one run in fifteen. Wait for the uploader to settle. */
    for (int i = 0; i < 1000; i++) {
        int pending = 0;
        pthread_mutex_lock(&G.mx);
        for (int eid = 0; eid < NE; eid++) pending += qs(0, eid)->queued;
        pthread_mutex_unlock(&G.mx);
        if (!pending) break;
        struct timespec ts = {0, 2000000};
        nanosleep(&ts, NULL);
    }
    for (int eid = 0; eid < NE; eid++)
        check(qt_is_resident(0, eid), "expert did not become resident during warmstart");

    fake_issue_hook = record_issue;
    float x[D];
    for (int i = 0; i < D; i++) x[i] = (float)i;

    /* All 32 issued experts are odd -> home(eid)=eid%2 routes every one of
     * them to device 1, the case that overruns today. */
    int eids[32];
    for (int k = 0; k < 32; k++) eids[k] = 2 * k + 1;
    uint32_t mask = qt_issue(0, eids, 32, x);
    check(mask == 0xFFFFFFFFu, "issuing 32 resident odd-numbered experts did not set all 32 mask bits");
    check(nrec == 1 && records[0].device == 1 && records[0].count == 32,
          "32 odd-numbered experts should issue as one 32-row group on device 1");

    int base2 = nrec;
    int eids2[32];
    for (int k = 0; k < 16; k++) { eids2[k] = 2 * k; eids2[16 + k] = 2 * k + 1; }
    uint32_t mask2 = qt_issue(0, eids2, 32, x);
    check(mask2 == 0xFFFFFFFFu, "issuing a mixed even/odd batch did not set all 32 mask bits");

    int inside = 1;
    for (int i = 0; i < nrec; i++) {
        const float *lo = G.is_x, *hi = G.is_x + G.is_x_floats;
        if (!(records[i].x >= lo && records[i].x + (size_t)records[i].count * G.D <= hi))
            inside = 0;
    }
    check(inside, "blocks_stay_inside_the_replica_buffer");

    check(nrec == base2 + 2, "the mixed batch should issue once per device (two calls)");
    if (nrec == base2 + 2) {
        const float *a0 = records[base2].x, *a1 = records[base2 + 1].x;
        size_t c0 = (size_t)records[base2].count, c1 = (size_t)records[base2 + 1].count;
        int disjoint = (a0 + c0 * G.D <= a1) || (a1 + c1 * G.D <= a0);
        check(disjoint, "device_blocks_do_not_overlap");
    }

    float val[32]; for (int k = 0; k < 32; k++) val[k] = 1.0f;
    float out[D]; memset(out, 0, sizeof out);
    qt_take(mask2, val, 32, out);   /* take returns NULL in the fake backend; fine */

    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_multidev: %d fallimenti\n", fails); return 1; }
    printf("test_qwen36_tier_multidev: ok\n");
    return 0;
}
