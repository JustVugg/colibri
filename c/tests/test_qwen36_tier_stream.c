/* The qwen36 tier below full residency, on the fake backend (#2090).
 *
 * A 32 GB box holds 182 of Qwen3.6's 256 experts a layer, and the tier used to
 * refuse anything but cap == n_experts: no experts in VRAM, and no trunk either,
 * since the trunk is placed inside the same init. Pinned here:
 *   - cap < n_experts starts the int4 tier in copy mode (qt_streaming), and a
 *     noted expert is copied, not pointed at: no pointer into the engine's
 *     recycled slot survives qt_note;
 *   - the engine's planar int4 (expert_ffn.h, qt_source_planar) is staged as
 *     exactly the bytes the container's signed pairs stage to, scales included,
 *     so the GPU computes the same expert whichever layout the RAM keeps;
 *   - a resident's hits keep it warm: qt_issue counts them, so in copy mode
 *     (where the engine notes only what the CPU computes) a newcomer evicts the
 *     coldest resident, not one the routing keeps choosing;
 *   - the mixed layout (-1), int8 with grouped scales and experts=cpu start the
 *     tier for the dense trunk only: no expert budget, nothing uploaded;
 *   - cap >= n_experts keeps the pointer mode cap == n_experts always had (a
 *     larger cap was refused too).
 *
 * No GPU, no toolkit; the fake backend records what the tier sends. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */
#include "../expert_ffn.h"

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

/* gate/up [IH][D], down [D][IH], int4 with one scale per 64 inputs */
enum { NL = 2, NE = 16, D = 128, IH = 64, GS = 64, CAP = 4, TOPK = 2 };
enum { MB = D * IH / 2, NSG = IH * (D / GS), NSD = D * (IH / GS) };

static uint8_t pairs[NE][3 * MB];    /* the container's signed pairs, gate|up|down */
static uint8_t planar[NE][3 * MB];   /* the same experts as expert_ffn.h keeps them */
static float scales[NE][2 * NSG + NSD];

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
#define NOTE(l, e) qt_note((l), (e), planar[e], planar[e] + MB, planar[e] + 2 * MB, \
                           scales[e], scales[e] + NSG, scales[e] + 2 * NSG)
/* one routed token on layer 0: the fake backend refuses the group, so the
 * mask comes back empty and the hits are counted all the same */
static void route(int a, int b) {
    float x[D]; for (int i = 0; i < D; i++) x[i] = 1.0f;
    int eids[TOPK] = { a, b };
    uint32_t mask = qt_issue(0, eids, TOPK, x);
    float out[D], val[TOPK] = { 0.5f, 0.5f };
    qt_take(mask, val, TOPK, out);
}
static size_t expert_bytes(void) {
    return 3 * dev_alloc_footprint(MB) + 2 * dev_alloc_footprint(NSG * sizeof(float))
         + dev_alloc_footprint(NSD * sizeof(float));
}
static void allow_experts(int n) {
    char gb[64];
    snprintf(gb, sizeof gb, "%.15f", (double)(n * expert_bytes() + expert_bytes() / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
}

int main(void) {
    uint32_t r = 2090;
    for (int e = 0; e < NE; e++) {
        for (int i = 0; i < 3 * MB; i++) { r = r * 1664525u + 1013904223u; pairs[e][i] = (uint8_t)(r >> 24); }
        xf_repack_pairs_signed(planar[e],          pairs[e],          IH, D);
        xf_repack_pairs_signed(planar[e] + MB,     pairs[e] + MB,     IH, D);
        xf_repack_pairs_signed(planar[e] + 2 * MB, pairs[e] + 2 * MB, D,  IH);
        for (int i = 0; i < 2 * NSG + NSD; i++) scales[e][i] = 0.001f * (e + 1) + 0.0001f * i;
    }
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    fake_ndev = 1;

    /* ---- 1. cap < n_experts: the int4 tier in copy mode ---- */
    printf(" 1. init below full residency\n");
    allow_experts(3);
    qt_source_planar(1);
    check(qt_init(NL, NE, D, IH, CAP, TOPK, GS, 1), "the int4 tier starts with cap 4 < 16 experts");
    check(qt_streaming(), "in copy mode");
    check(qt_experts_on(), "with room for experts");
    check(G.wfmt == 4 && G.egs == GS, "int4 with groups of 64");
    check(G.exp_bytes == expert_bytes(), "bytes per expert: three int4 matrices and three scale tables");

    /* ---- 2. planar staging = pair staging, byte for byte ---- */
    printf(" 2. planar -> pairs\n");
    {
        static uint8_t a[3 * MB], b[3 * MB];
        static float sa[2 * NSG + NSD], sb[2 * NSG + NSD];
        int same = 1;
        for (int e = 0; e < NE && same; e++) {
            G_src_planar = 1;
            stage(a, sa, 0, planar[e], planar[e] + MB, planar[e] + 2 * MB, scales[e], scales[e] + NSG, scales[e] + 2 * NSG);
            G_src_planar = 0;
            stage(b, sb, 0, pairs[e], pairs[e] + MB, pairs[e] + 2 * MB, scales[e], scales[e] + NSG, scales[e] + 2 * NSG);
            same = !memcmp(a, b, sizeof a) && !memcmp(sa, sb, sizeof sa);
        }
        G_src_planar = 1;
        check(same, "every expert's planar bytes stage to what its signed pairs stage to, scales included");
        int offset = 1;
        for (int i = 0; i < MB; i++) if (b[i] != (uint8_t)(pairs[NE - 1][i] ^ 0x88)) { offset = 0; break; }
        check(offset, "which is the pairs in offset binary (XOR 0x88), the backend's fmt 4");
    }

    /* ---- 3. a note copies: upload in fmt 4, no pointer kept ---- */
    printf(" 3. note -> copy -> upload\n");
    fake_uploads = 0;
    NOTE(0, 5);
    wait_idle();
    check(qt_is_resident(0, 5), "the noted expert becomes resident");
    check(fake_uploads == 3 && last_fmt == 4, "three tensors uploaded, fmt 4");
    int up = 1;
    for (size_t i = 0; i < captured_len; i++) if (captured[i] != (uint8_t)(pairs[5][i] ^ 0x88)) { up = 0; break; }
    check(captured_len == MB && up, "the gate that reached the device is the container's, in offset binary");
    pthread_mutex_lock(&G.mx);
    check(qs(0, 5)->g4 == NULL && qs(0, 5)->gs == NULL, "no pointer into the engine's slot survives qt_note");
    pthread_mutex_unlock(&G.mx);

    /* ---- 4. heat: hits keep residents warm ---- */
    printf(" 4. a resident the routing keeps choosing is not the one evicted\n");
    NOTE(0, 6); NOTE(0, 7);
    wait_idle();
    check(qt_is_resident(0, 6) && qt_is_resident(0, 7), "the budget holds three experts");
    for (int i = 0; i < 20; i++) route(5, 6);   /* 5 and 6 hit, 7 is never routed again */
    pthread_mutex_lock(&G.mx);
    check(qs(0, 5)->heat == 21 && qs(0, 6)->heat == 21 && qs(0, 7)->heat == 1, "hits counted by qt_issue, once per token");
    check(G.hits[0] == 40 && G.miss == 0, "forty hits, no miss");
    pthread_mutex_unlock(&G.mx);
    for (int i = 0; i < 8 && !qt_is_resident(0, 9); i++) { NOTE(0, 9); wait_idle(); }
    check(qt_is_resident(0, 9), "a newcomer warmer than the coldest resident is promoted");
    check(!qt_is_resident(0, 7), "in place of the resident nobody routes");
    check(qt_is_resident(0, 5) && qt_is_resident(0, 6), "the hot residents stay");
    for (int i = 0; i < 10; i++) { NOTE(0, 10); wait_idle(); }
    check(!qt_is_resident(0, 10), "a newcomer no warmer than the coldest resident by the hysteresis is refused");
    pthread_mutex_lock(&G.mx);
    int residents = 0, stale = 0;
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) { residents += qs(l, e)->resident; stale += qs(l, e)->g4 != NULL; }
    check(residents == 3 && G.used[0] == 3 * G.exp_bytes, "swaps are budget-neutral");
    check(stale == 0, "no slot keeps a pointer in copy mode");
    pthread_mutex_unlock(&G.mx);
    qt_shutdown();
    check(!qt_streaming() && !G_src_planar, "shutdown leaves copy mode and the planar source");

    /* ---- 5. no expert format the tier takes: the trunk only ---- */
    printf(" 5. trunk only\n");
    struct { int is_int4, gs; const char *place, *what; } only[] = {
        { -1, GS, "off",         "the mixed layout" },
        {  0, GS, "off",         "int8 experts with grouped scales" },
        {  1, GS, "experts=cpu", "COLI_PLACE experts=cpu" },
    };
    for (size_t i = 0; i < sizeof only / sizeof *only; i++) {
        char what[160];
        setenv("COLI_PLACE", only[i].place, 1);
        snprintf(what, sizeof what, "%s: the tier starts (for the trunk)", only[i].what);
        check(qt_init(NL, NE, D, IH, NE, TOPK, only[i].gs, only[i].is_int4), what);
        snprintf(what, sizeof what, "%s: no experts, no expert budget", only[i].what);
        check(!qt_experts_on() && G.budget[0] == 0, what);
        fake_uploads = 0;
        NOTE(0, 5);
        wait_idle();
        snprintf(what, sizeof what, "%s: a note uploads nothing", only[i].what);
        check(fake_uploads == 0 && !qt_is_resident(0, 5), what);
        qt_shutdown();
    }
    setenv("COLI_PLACE", "off", 1);

    /* ---- 6. full residency keeps the pointer mode ---- */
    printf(" 6. cap >= n_experts\n");
    check(qt_init(NL, NE, D, IH, 2 * NE, TOPK, GS, 1) && !qt_streaming(), "cap 32 > 16 experts: the tier starts, no copy mode");
    qt_shutdown();
    qt_source_planar(1);
    check(qt_init(NL, NE, D, IH, NE, TOPK, GS, 1) && !qt_streaming(), "cap 16 == 16 experts: no copy mode");
    NOTE(0, 3);
    wait_idle();
    pthread_mutex_lock(&G.mx);
    check(qs(0, 3)->g4 == planar[3], "the tier keeps the pointer into the slot, which is never recycled");
    pthread_mutex_unlock(&G.mx);
    check(qt_is_resident(0, 3), "and uploads from it");
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_stream: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_stream: ok\n");
    return 0;
}
