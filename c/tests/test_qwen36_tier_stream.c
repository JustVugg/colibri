/* The tier's streaming mode for qwen36's own int4 format, on the fake backend.
 *
 * qwen36 used to need cap == n_experts for the VRAM tier: the tier kept raw
 * pointers into RAM slots that were never recycled. qt_init_stream reuses the
 * Qwen3.8 streaming lifecycle for int4 gs64 experts, so a box whose RAM cannot
 * hold every expert still gets the GPU. Pinned here:
 *   - qt_init refuses cap < n_experts, qt_init_stream accepts it (int4, gs64);
 *   - the upload is fmt 4 with group scales, staged with the 0x88 XOR;
 *   - no pointer into the engine's slot survives qt_note or qt_note_planned;
 *   - qt_touch keeps a routed resident hot, so a newcomer no hotter than it is
 *     refused and the resident is not the one swapped out;
 *   - shutdown leaves streaming mode, and a later full-residency init does not
 *     inherit it.
 * No GPU, no toolkit; the fake backend records what the tier sends. */
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

/* toy qwen36 geometry, gs 64: gate/up [IH,D], down [D,IH], int4 packed */
enum { NL = 2, NE = 16, D = 128, IH = 64, GS = 64, CAP = 4, TOPK = 2 };
#define MB   (D * IH / 2)                    /* packed bytes per matrix */
#define NGU  (IH * ((D + GS - 1) / GS))      /* gate/up scales */
#define NDS  (D * ((IH + GS - 1) / GS))      /* down scales */

static unsigned char pk[NE][3 * MB];         /* g|u|d packed int4, one per expert */
static float scl[NE][2 * NGU + NDS];

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
#define NOTE(fn, l, e) fn((l), (e), pk[e], pk[e] + MB, pk[e] + 2 * MB, \
                          scl[e], scl[e] + NGU, scl[e] + 2 * NGU)

static int no_slot_pointer(void) {
    int n = 0;
    pthread_mutex_lock(&G.mx);
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) n += qs(l, e)->g4 != NULL || qs(l, e)->gs != NULL;
    pthread_mutex_unlock(&G.mx);
    return n == 0;
}

int main(void) {
    for (int e = 0; e < NE; e++) {
        for (size_t i = 0; i < sizeof pk[e]; i++) pk[e][i] = (unsigned char)((i * 5 + e * 11) & 0xFF);
        for (int i = 0; i < 2 * NGU + NDS; i++) scl[e][i] = 0.01f * (e + 1) + 0.001f * i;
    }
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    fake_ndev = 1;

    /* allowance for exactly 3 experts */
    size_t exp_bytes = 3 * dev_alloc_footprint((size_t)MB)
                     + 3 * dev_alloc_footprint((2 * NGU + NDS) / 3 * sizeof(float));
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(3 * exp_bytes + exp_bytes / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);

    /* ---- 1. full residency still refuses cap < n_experts; streaming accepts ---- */
    printf(" 1. init\n");
    check(!qt_init(NL, NE, D, IH, CAP, TOPK, GS, 1), "qt_init keeps refusing cap < n_experts");
    check(!qt_streaming(), "a refused qt_init is not streaming");
    fake_uploads = 0;
    check(qt_init_stream(NL, NE, D, IH, CAP, TOPK, GS, 1), "qt_init_stream starts with cap 4 < 16 experts");
    check(qt_streaming(), "qt_streaming reports the mode");
    check(G.wfmt == 4 && G.egs == GS, "int4 gs64 format kept (not fmt 8)");
    check(!G_fp8_stream, "int4 streaming is not the fp8 format");
    check(G.exp_bytes == exp_bytes, "exp_bytes = three int4 matrices + three scale tables");

    /* ---- 2. a note uploads fmt 4 grouped, XOR-staged, and keeps no pointer ---- */
    printf(" 2. note -> upload\n");
    NOTE(qt_note, 0, 5);
    wait_idle();
    check(qt_is_resident(0, 5), "noted expert becomes resident");
    check(fake_uploads == 3, "three tensors uploaded for one expert");
    check(last_fmt == 4, "upload format is 4 (int4)");
    int xored = captured_len > 0;
    for (size_t i = 0; i < captured_len; i++) if (captured[i] != (unsigned char)(pk[5][i] ^ 0x88)) { xored = 0; break; }
    check(xored, "int4 nibbles staged as offset-binary (XOR 0x88)");
    check(no_slot_pointer(), "no pointer into the engine slot survives qt_note");

    /* the engine's slot is recycled right after: overwrite it, the upload is unaffected */
    memset(pk[5], 0, sizeof pk[5]);
    check(qt_is_resident(0, 5), "recycling the RAM slot does not touch the resident");

    /* ---- 3. qt_touch keeps routed residents hot ---- */
    printf(" 3. touch vs newcomer\n");
    NOTE(qt_note, 0, 6); NOTE(qt_note, 0, 7);
    wait_idle();
    check(qt_is_resident(0, 6) && qt_is_resident(0, 7), "budget holds three experts");
    for (int i = 0; i < 8; i++) { qt_touch(0, 5); qt_touch(0, 6); qt_touch(0, 7); }
    /* newcomer noted 8 times: as hot as the touched residents, never hotter */
    for (int i = 0; i < 8; i++) { NOTE(qt_note, 0, 8); wait_idle(); }
    check(!qt_is_resident(0, 8), "a newcomer no hotter than touched residents is refused");
    check(qt_is_resident(0, 5) && qt_is_resident(0, 6) && qt_is_resident(0, 7), "touched residents stay");
    /* without touches, 6 goes cold relative to a hammered newcomer */
    for (int i = 0; i < 64; i++) { qt_touch(0, 5); qt_touch(0, 7); }
    for (int i = 0; i < 64 && !qt_is_resident(0, 9); i++) { NOTE(qt_note, 0, 9); wait_idle(); }
    check(qt_is_resident(0, 9), "a hot newcomer is promoted when the device is full");
    check(!qt_is_resident(0, 6), "the untouched (coldest) resident is the one swapped out");
    check(qt_is_resident(0, 5) && qt_is_resident(0, 7), "touched residents survive the swap");
    pthread_mutex_lock(&G.mx);
    check(G.used[0] == 3 * G.exp_bytes, "swap is budget-neutral");
    pthread_mutex_unlock(&G.mx);
    check(no_slot_pointer(), "still no slot pointer after swaps");

    /* ---- 4. warmstart path: plan + note_planned copies and forgets ---- */
    printf(" 4. planned\n");
    int pl[NL * NE], pe[NL * NE];
    int wn = qt_plan_fill(pl, pe, NL * NE);
    check(wn == 0, "device already full: nothing planned");
    qt_shutdown();
    check(!qt_streaming() && !G_stream, "shutdown leaves streaming mode");

    check(qt_init_stream(NL, NE, D, IH, CAP, TOPK, GS, 1), "restart in streaming mode");
    wn = qt_plan_fill(pl, pe, NL * NE);
    check(wn == 3, "plan fills the three-expert budget");
    for (int i = 0; i < wn; i++) NOTE(qt_note_planned, pl[i], pe[i]);
    qt_fill_wait();
    int res = 0; for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) res += qt_is_resident(l, e);
    check(res == 3, "planned experts are resident after qt_fill_wait");
    check(no_slot_pointer(), "qt_note_planned keeps no slot pointer in streaming mode");
    qt_shutdown();

    /* ---- 5. a later full-residency init does not inherit streaming ---- */
    printf(" 5. no leak into full residency\n");
    check(qt_init(NL, NE, D, IH, NE, TOPK, GS, 1), "full residency init with cap == n_experts");
    check(!qt_streaming(), "full residency is not streaming after a streaming run");
    NOTE(qt_note, 1, 3);
    wait_idle();
    pthread_mutex_lock(&G.mx);
    check(qs(1, 3)->g4 == pk[3], "full residency keeps the slot pointer, as before");
    pthread_mutex_unlock(&G.mx);
    qt_shutdown();

    if (fails) { printf("qwen36 tier streaming: %d FAILED\n", fails); return 1; }
    printf("qwen36 tier streaming tests: ok\n");
    return 0;
}
