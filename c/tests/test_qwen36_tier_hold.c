/* qt_chain_hold: no upload while a chain's forward holds the tier, a free slot's included.
 *
 * The CUDA chain's frame may hold a kernel that waits for a word the host writes later in the
 * same frame (a hybrid layer: the host's misses, qwen36_cuda_chain.h). A synchronous copy or
 * allocation the uploader makes meanwhile waits for that kernel, the host's next launch waits
 * for the uploader's driver lock, and the frame never ends: on a cold tier, whose uploads are
 * all free slots', a first prompt of a few tokens hung 9 times in 10 on an RTX PRO 4000. A
 * victim's swap waited for the hold already; a free slot's upload did not.
 *
 * Asserted, with the fake backend (no GPU): uploads queued while the tier is held wait until
 * it is released, and taking the hold waits for an upload already in flight. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../compat.h"   /* setenv: MinGW has none */
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { printf("  %s: %s\n", ok ? "ok" : "FAIL", what); if (!ok) fails++; }

enum { NL = 2, NE = 8, D = 64, IH = 32, TOPK = 2 };

static void sleep_ms(int ms) { struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L}; nanosleep(&ts, NULL); }
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }

static volatile int held, upload_while_held, slow_started, slow_done;
static void watch_hook(int fmt) { (void)fmt; if (held) upload_while_held = 1; }
static void slow_hook(int fmt) {   /* the first tensor of the next upload takes 200 ms, as a real copy can */
    (void)fmt;
    if (held) upload_while_held = 1;
    if (!slow_started) { slow_started = 1; sleep_ms(200); slow_done = 1; }
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1); setenv("COLI_PLACE", "off", 1);
    setenv("QT_UPLOAD_SYNC", "", 1);
    fake_ndev = 1; fake_uploads = 0;
    size_t exp_bytes = 3 * dev_alloc_footprint((size_t)D * IH / 2) + 3 * dev_alloc_footprint((2 * IH + D) / 3 * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(16 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "the tier starts, room for every expert");

    static unsigned char g4[NL][NE][D * IH / 2], u4[NL][NE][D * IH / 2], d4[NL][NE][D * IH / 2];
    static float sc[NL][NE][2 * IH + D];
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        memset(g4[l][e], e + 1, sizeof g4[l][e]); memset(u4[l][e], e + 2, sizeof u4[l][e]); memset(d4[l][e], e + 3, sizeof d4[l][e]);
        for (int i = 0; i < 2 * IH + D; i++) sc[l][e][i] = 1.0f;
    }

    /* free slots' uploads queued while a forward holds the tier: none runs until the release */
    fake_upload_hook = watch_hook;
    qt_chain_hold(1); held = 1;
    for (int e = 0; e < 4; e++) qt_note_block(0, e, g4[0][e], u4[0][e], d4[0][e], sc[0][e], sc[0][e] + IH, sc[0][e] + 2 * IH);
    sleep_ms(150);
    check(fake_uploads == 0 && !upload_while_held, "no free slot's upload while the chain holds the tier");
    held = 0; qt_chain_hold(0);
    qt_fill_wait();
    int resident = 0;
    for (int e = 0; e < 4; e++) resident += qt_is_resident(0, e);
    check(resident == 4 && fake_uploads == 12, "released: the four experts land, three tensors each");

    /* an upload in flight when a forward takes the hold: the hold waits for it */
    fake_upload_hook = slow_hook;
    qt_note_block(1, 0, g4[1][0], u4[1][0], d4[1][0], sc[1][0], sc[1][0] + IH, sc[1][0] + 2 * IH);
    double t0 = now_ms();
    while (!slow_started && now_ms() - t0 < 5000) sleep_ms(1);
    check(slow_started, "the slow upload started");
    double t1 = now_ms();
    qt_chain_hold(1); held = 1;
    double waited = now_ms() - t1;
    check(slow_done && qt_is_resident(1, 0), "taking the hold waited for the upload in flight to land");
    printf("    (the hold waited %.0f ms)\n", waited);
    qt_note_block(1, 1, g4[1][1], u4[1][1], d4[1][1], sc[1][1], sc[1][1] + IH, sc[1][1] + 2 * IH);
    sleep_ms(100);
    check(!qt_is_resident(1, 1) && !upload_while_held, "and the next one waits for the release");
    held = 0; qt_chain_hold(0);
    qt_fill_wait();
    check(qt_is_resident(1, 1), "which lands after it");

    qt_shutdown();
    printf("test_qwen36_tier_hold: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
