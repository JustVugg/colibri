/* The tier's fp8 streaming mode (Qwen3.8), on the fake backend.
 *
 * What changes against the qwen36 modes and is pinned here:
 *   - cap < n_experts is accepted: the experts do not all live in RAM;
 *   - bytes per expert count one byte per element plus three 128x128 block
 *     scale tables, and the upload is fmt=8 with exactly those scales;
 *   - staging copies the e4m3 bytes unchanged (no 0x88 XOR: on whole bytes it
 *     would be corruption) and the block scales in order gate|up|down;
 *   - the tier keeps NO pointer into the engine's slot after qt_note returns,
 *     because that slot is recycled by the next token;
 *   - promotion happens when the bytes pass by: a non-resident expert noted
 *     on a full device evicts the coldest resident there if it is hotter, a
 *     colder one is refused, and the swap is budget-neutral;
 *   - the e4m3 decode table is published to the backend before any upload.
 *
 * No GPU, no toolkit; the fake backend records what the tier sends. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

/* toy Qwen3.8 geometry: hidden 256, inter 128 -> gate/up [128,256] = 2x2 blocks,
 * down [256,128] = 2x1 -- more than one block per matrix on purpose */
enum { NL = 2, NE = 16, D = 256, IH = 128, CAP = 4, TOPK = 2 };
#define NBD ((D + 127) / 128)
#define NBI ((IH + 127) / 128)
#define NSC (NBI * NBD)                      /* scales per matrix */

static unsigned char slab[NE][3 * D * IH];   /* gate|up|down e4m3 bytes, one per expert */
static float scales[NE][3 * NSC];
static float lut[256];

static void wait_idle(void) {
    for (int i = 0; i < 1000; i++) {
        pthread_mutex_lock(&G.mx);
        int pending = G.qn;
        for (int l = 0; l < G.nl; l++) for (int e = 0; e < G.ne; e++) pending += qs(l, e)->queued;
        pthread_mutex_unlock(&G.mx);
        if (!pending) break;
        struct timespec ts = {0, 2000000}; nanosleep(&ts, NULL);
    }
}
#define NOTE(fn, l, e) fn((l), (e), slab[e], slab[e] + D * IH, slab[e] + 2 * D * IH, \
                          scales[e], scales[e] + NSC, scales[e] + 2 * NSC)

int main(void) {
    for (int e = 0; e < NE; e++) {
        for (size_t i = 0; i < sizeof slab[e]; i++) slab[e][i] = (unsigned char)((i * 7 + e * 13) & 0xFF);
        for (int i = 0; i < 3 * NSC; i++) scales[e][i] = 0.001f * (e + 1) + 0.0001f * i;
    }
    for (int i = 0; i < 256; i++) lut[i] = (float)i;
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);

    /* ---- 1. the mode starts with cap < n_experts, publishes the LUT, sizes fmt 8 ---- */
    printf(" 1. init\n");
    /* allowance for exactly 3 experts on the one device */
    size_t exp_bytes = 3ull * D * IH + 3ull * NSC * sizeof(float) + 4096;
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(3 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    fake_ndev = 1; fake_uploads = 0; fake_lut_published = 0;
    check(qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut), "fp8 streaming mode must start with cap 4 < 16 experts");
    check(G.wfmt == 8, "weight format is 8");
    check(fake_lut_published, "e4m3 LUT published to the backend before any upload");
    check(G.exp_bytes == exp_bytes, "exp_bytes = 3*D*IH bytes + 3 block-scale tables + slack");
    check(G.sc_gu == NSC && G.sc_d == NSC, "one scale per 128x128 block per matrix");

    /* ---- 2. a note uploads fmt 8, bytes intact, scales in order, pointers dropped ---- */
    printf(" 2. note -> upload\n");
    NOTE(qt_note, 0, 5);
    wait_idle();
    check(qt_is_resident(0, 5), "noted expert becomes resident");
    check(fake_uploads == 3, "three tensors uploaded for one expert");
    check(last_fmt == 8, "upload format is 8");
    check(last_bytes == (size_t)D * IH, "fmt 8 carries one byte per element");
    int intact = 1;
    for (size_t i = 0; i < captured_len; i++) if (captured[i] != slab[5][i]) { intact = 0; break; }
    check(intact, "e4m3 bytes staged unchanged (no XOR)");
    pthread_mutex_lock(&G.mx);
    check(qs(0, 5)->g4 == NULL && qs(0, 5)->gs == NULL, "no pointer into the engine slot survives qt_note");
    pthread_mutex_unlock(&G.mx);

    /* ---- 3. fill the budget, then promotion-at-note ---- */
    printf(" 3. fill, then swap at note\n");
    NOTE(qt_note, 0, 6); NOTE(qt_note, 0, 7);
    wait_idle();
    check(qt_is_resident(0, 6) && qt_is_resident(0, 7), "budget holds three experts");
    pthread_mutex_lock(&G.mx);
    check(G.used[0] == 3 * G.exp_bytes, "used == 3 x exp_bytes");
    pthread_mutex_unlock(&G.mx);
    /* a fourth, cold expert (heat 1) must not displace anyone (residents have heat 1 too) */
    NOTE(qt_note, 0, 8);
    wait_idle();
    check(!qt_is_resident(0, 8), "an expert no hotter than the coldest resident is refused");
    check(qt_is_resident(0, 5) && qt_is_resident(0, 6) && qt_is_resident(0, 7), "residents untouched by a refused newcomer");
    /* make expert 9 hot: note it many times; each note is a chance to promote */
    for (int i = 0; i < 64 && !qt_is_resident(0, 9); i++) { NOTE(qt_note, 0, 9); wait_idle(); }
    check(qt_is_resident(0, 9), "a hot expert noted repeatedly gets promoted when the device is full");
    int residents = 0; for (int e = 0; e < NE; e++) residents += qt_is_resident(0, e);
    check(residents == 3, "the promotion was a swap: still three residents");
    pthread_mutex_lock(&G.mx);
    check(G.used[0] == 3 * G.exp_bytes, "swap is budget-neutral");
    check(G.swaps >= 1, "swap counted");
    int stale_ptr = 0; for (int e = 0; e < NE; e++) stale_ptr += (qs(0, e)->g4 != NULL);
    pthread_mutex_unlock(&G.mx);
    check(stale_ptr == 0, "no slot retains a pointer in streaming mode");

    /* ---- 4. issue on residents uses the fake issue path ---- */
    printf(" 4. issue mask\n");
    fake_issue_hook = NULL;                  /* fake issue returns 0 -> mask cleared, but hits counted */
    float x[D]; for (int i = 0; i < D; i++) x[i] = 1.0f;
    int eids[TOPK] = { 9, 8 };
    uint32_t mask = qt_issue(0, eids, TOPK, x);
    check(mask == 0, "fake issue refuses -> everything handed back to the CPU");
    float out[D]; memset(out, 0, sizeof out); float val[TOPK] = {1, 1};
    qt_take(mask, val, TOPK, out);
    pthread_mutex_lock(&G.mx);
    check(G.hits[0] == 1 && G.miss == 1, "one resident routed (hit), one not (miss)");
    pthread_mutex_unlock(&G.mx);
    qt_shutdown();
    check(!G_fp8_stream, "shutdown leaves streaming mode");

    /* ---- 5. no LUT -> no fp8 tier ---- */
    printf(" 5. refusals\n");
    fake_lut_published = 0;
    check(!qt_init_fp8(NL, NE, D, IH, CAP, TOPK, NULL), "without a decode table the fp8 tier stays off");
    check(!G_fp8_stream, "a refused init does not leave the flag set");
    /* and the classic modes still insist on cap == n_experts */
    check(!qt_init(NL, NE, D, IH, CAP, TOPK, 0, 1), "int4 mode still refuses cap < n_experts");

    if (fails) { printf("test_qwen36_tier_fp8: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_fp8: ok\n");
    return 0;
}
