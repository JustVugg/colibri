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

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */

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
    /* qt_fill_wait now returns only once every enqueued expert is resident
     * (test_qwen36_tier_fill_wait pins that), so residency can be read right
     * here; the poll this used to need is gone with the race it papered over. */
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

    /* In-bounds and disjoint are both necessary and both insufficient: a
     * stride SMALLER than the row capacity keeps every block inside the
     * allocation while sliding device 1's window down onto device 0's, and a
     * single-row issue on device 1 is disjoint from device 0's under any
     * stride at all. The two checks below pin the stride itself. */
    check(sizeof G.is_k[0] / sizeof G.is_k[0][0] == QT_MAX_ROWS &&
          G.is_x_floats == (size_t)G.ndev * (sizeof G.is_k[0] / sizeof G.is_k[0][0]) * (size_t)G.D,
          "replica_block_stride_equals_the_per_device_row_capacity");

    check(nrec == base2 + 2, "the mixed batch should issue once per device (two calls)");
    if (nrec == base2 + 2) {
        const float *a0 = records[base2].x, *a1 = records[base2 + 1].x;
        size_t c0 = (size_t)records[base2].count, c1 = (size_t)records[base2 + 1].count;
        int disjoint = (a0 + c0 * G.D <= a1) || (a1 + c1 * G.D <= a0);
        check(disjoint, "device_blocks_do_not_overlap");
        /* Where, not just whether: device di's block starts at
         * G.is_x + di*QT_MAX_ROWS*D. This is the one assertion that fails on
         * the old 8*D stride even when device 1 carries a single row. */
        int dev0 = records[base2].device == 0 ? base2 : base2 + 1;
        int dev1 = dev0 == base2 ? base2 + 1 : base2;
        check(records[dev0].device == 0 && records[dev1].device == 1,
              "the mixed batch should issue once on device 0 and once on device 1");
        check(records[dev0].x == G.is_x,
              "device_0_block_starts_at_the_buffer_base");
        check(records[dev1].x == G.is_x + (size_t)QT_MAX_ROWS * (size_t)G.D,
              "device_1_block_starts_one_full_row_capacity_in");
    }

    float val[32]; for (int k = 0; k < 32; k++) val[k] = 1.0f;
    float out[D]; memset(out, 0, sizeof out);
    qt_take(mask2, val, 32, out);   /* take returns NULL in the fake backend; fine */

    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_multidev: %d fallimenti\n", fails); return 1; }
    printf("test_qwen36_tier_multidev: ok\n");
    return 0;
}
