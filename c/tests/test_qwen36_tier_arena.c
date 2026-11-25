/* QT_ARENA=1 exact-size slab slots for FP8 and int4 experts + copy-mode promotion
 * accounting, verified on the fake CUDA backend with QT_COLD_CHECK enabled.
 *
 * Pins:
 *   1. FP8 (fmt=8) and int4-g64 (fmt=4) with QT_ARENA=1 compute 256B-aligned
 *      exact slot sizes and upload via coli_cuda_tensor_upload_view into slabs.
 *   2. Copy-mode swap reuses the victim's arena slot (both in-place overwrite
 *      and overwrite-refused fallback re-upload), keeping slab count at 1.
 *   3. qt_note_n adds n heat in one call and triggers promotion cleanly.
 *   4. Fragmented card (fake_pipe_max) halves slabs down to 1 slot without
 *      clamping budget; completely full card clamps budget.
 *   5. Backend without coli_cuda_tensor_upload_view falls back to cudaMalloc.
 *   6. Shutdown frees every allocated slab.
 */
#define QT_COLD_CHECK 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { NL = 2, NE = 16, D = 256, IH = 128, CAP = 4, TOPK = 2 };
#define NBD ((D + 127) / 128)
#define NBI ((IH + 127) / 128)
#define NSC (NBI * NBD)

static unsigned char fp8_w[NE][3 * D * IH];
static float fp8_s[NE][3 * NSC];
static float lut[256];

/* int4-g64 buffers */
#define MB4 ((size_t)D * IH / 2)
#define SC4_GU ((size_t)IH * (D / 64))
#define SC4_D  ((size_t)D * (IH / 64))
static unsigned char i4_w[NE][3 * MB4];
static float i4_s[NE][2 * SC4_GU + SC4_D];

static void wait_idle(void) {
    for (int i = 0; i < 1000; i++) {
        pthread_mutex_lock(&G.mx);
        int pending = G.qn;
        for (int l = 0; l < G.nl; l++)
            for (int e = 0; e < G.ne; e++) pending += qs(l, e)->queued;
        pthread_mutex_unlock(&G.mx);
        if (!pending) break;
        struct timespec ts = {0, 2000000}; nanosleep(&ts, NULL);
    }
}

#define NOTE_FP8(l, e, n) qt_note_n((l), (e), (n), \
    fp8_w[e], fp8_w[e] + D * IH, fp8_w[e] + 2 * D * IH, \
    fp8_s[e], fp8_s[e] + NSC, fp8_s[e] + 2 * NSC)

#define NOTE_I4(l, e, n) qt_note_n((l), (e), (n), \
    i4_w[e], i4_w[e] + MB4, i4_w[e] + 2 * MB4, \
    i4_s[e], i4_s[e] + SC4_GU, i4_s[e] + 2 * SC4_GU)

static size_t align256(size_t x) { return (x + 255u) / 256u * 256u; }

int main(void) {
    for (int e = 0; e < NE; e++) {
        memset(fp8_w[e], (unsigned char)(e + 1), sizeof fp8_w[e]);
        for (int i = 0; i < 3 * NSC; i++) fp8_s[e][i] = 0.01f * (e + 1);
        memset(i4_w[e], (unsigned char)(e + 3), sizeof i4_w[e]);
        for (size_t i = 0; i < 2 * SC4_GU + SC4_D; i++) i4_s[e][i] = 0.02f * (e + 1);
    }
    for (int i = 0; i < 256; i++) lut[i] = (float)i;

    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    setenv("QT_ARENA", "1", 1);

    /* ---- 1. FP8 arena sizing, slab upload, in-place + fallback swap ---- */
    printf(" 1. FP8 QT_ARENA=1\n");
    const size_t w_pad = align256((size_t)D * IH);
    const size_t s_pad = align256((size_t)NSC * sizeof(float));
    const size_t slot = 3 * w_pad + 3 * s_pad;
    const int per_slab = (int)(QT_AR_SLAB / slot);
    const size_t want_exp = slot + (2 * 1048576u + per_slab - 1) / per_slab;
    const size_t cuda_exp = 3 * dev_alloc_footprint((size_t)D * IH)
                          + 3 * dev_alloc_footprint((size_t)NSC * sizeof(float));

    char gb[64];
    snprintf(gb, sizeof gb, "%.15f", (double)(3 * want_exp + want_exp / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);

    fake_ndev = 1; fake_uploads = 0; fake_overwrites = 0; fake_overwrite_fail = 0;
    fake_upload_view_n = 0; fake_upload_view_absent = 0;
    fake_pipe_allocs = 0; fake_pipe_frees = 0; fake_pipe_bytes = 0; fake_pipe_max = 0;

    check(qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut), "fp8 arena init");
    check(G_ar.on == 1, "arena enabled");
    check(G_ar.slot == slot, "exact 256B-aligned slot size");
    check(G.exp_bytes == want_exp, "exp_bytes matches slot + slab rounding share");
    check(G.exp_bytes < cuda_exp, "arena exp_bytes smaller than per-matrix cudaMalloc footprint");

    for (int e = 0; e < 3; e++) NOTE_FP8(0, e, 1);
    wait_idle();
    check(qt_is_resident(0, 0) && qt_is_resident(0, 1) && qt_is_resident(0, 2), "3 experts resident");
    check(fake_pipe_allocs == 1, "one slab allocated for all 3 slots");
    check(fake_pipe_bytes == 3 * slot, "slab bounded by budget room (3 slots)");
    check(fake_upload_view_n == 9, "9 view uploads (3 matrices x 3 experts)");
    pthread_mutex_lock(&G.mx);
    check(G.used[0] == 3 * G.exp_bytes, "used == 3 * exp_bytes");
    pthread_mutex_unlock(&G.mx);

    /* swap via qt_note_n with in-place overwrite */
    NOTE_FP8(0, 5, 10);
    wait_idle();
    check(qt_is_resident(0, 5), "qt_note_n(10) promotes expert 5");
    check(fake_overwrites == 3, "in-place overwrite used on arena view");
    check(fake_pipe_allocs == 1, "swap kept existing slab");

    /* swap with overwrite refused: re-uploads into victim's slot, still 1 slab */
    fake_overwrite_fail = 1;
    NOTE_FP8(0, 6, 30);
    wait_idle();
    check(qt_is_resident(0, 6), "overwrite-refused swap promotes expert 6");
    check(fake_upload_view_n == 12 && fake_pipe_allocs == 1, "3 more view uploads into victim slot, still 1 slab");
    fake_overwrite_fail = 0;
    qt_shutdown();
    check(fake_pipe_frees == fake_pipe_allocs, "shutdown frees all slabs");

    /* ---- 2. Fragmented card: slab halving (3 -> 1 + 2), budget not clamped ---- */
    printf(" 2. fragmented card slab halving\n");
    fake_pipe_max = 2 * slot;
    fake_pipe_allocs = fake_pipe_frees = 0; fake_pipe_bytes = 0; fake_upload_view_n = 0;
    check(qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut) && G_ar.on, "arena init on fragmented card");
    size_t budget0 = G.budget[0];
    for (int e = 0; e < 3; e++) { NOTE_FP8(0, e, 1); wait_idle(); }
    check(qt_is_resident(0, 0) && qt_is_resident(0, 1) && qt_is_resident(0, 2), "all 3 experts resident via halved slabs");
    check(fake_pipe_allocs == 2 && fake_pipe_bytes == 3 * slot, "1-slot slab then 2-slot slab (3 slots total)");
    check(G.budget[0] == budget0, "budget not clamped on fragmentation");
    qt_shutdown();
    check(fake_pipe_frees == fake_pipe_allocs, "shutdown frees halved slabs");

    /* ---- 3. Completely full card: 0 slabs -> budget clamped ---- */
    printf(" 3. full card clamp\n");
    fake_pipe_max = 1;
    fake_pipe_allocs = fake_pipe_frees = 0; fake_pipe_bytes = 0;
    check(qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut) && G_ar.on, "arena init on full card");
    NOTE_FP8(0, 0, 1); wait_idle();
    check(!qt_is_resident(0, 0) && G.budget[0] == G.used[0], "no slab fits: expert not resident, budget clamped");
    qt_shutdown();
    fake_pipe_max = 0;

    /* ---- 4. Older backend without upload_view falls back to cudaMalloc ---- */
    printf(" 4. fallback without upload_view\n");
    fake_upload_view_absent = 1;
    check(qt_init_fp8(NL, NE, D, IH, CAP, TOPK, lut) && !G_ar.on && G.exp_bytes == cuda_exp,
          "absent upload_view keeps cudaMalloc path");
    qt_shutdown();
    fake_upload_view_absent = 0;

    /* ---- 5. int4-g64 with QT_ARENA=1 ---- */
    printf(" 5. int4-g64 QT_ARENA=1\n");
    const size_t i4_slot = 3 * align256(MB4) + 2 * align256(SC4_GU * sizeof(float)) + align256(SC4_D * sizeof(float));
    const int i4_per = (int)(QT_AR_SLAB / i4_slot);
    const size_t i4_exp = i4_slot + (2 * 1048576u + i4_per - 1) / i4_per;
    snprintf(gb, sizeof gb, "%.15f", (double)(2 * i4_exp + i4_exp / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    fake_pipe_allocs = fake_pipe_frees = 0; fake_pipe_bytes = 0; fake_upload_view_n = 0;
    check(qt_init(NL, NE, D, IH, CAP, TOPK, 64, 1) && G_ar.on && G_ar.slot == i4_slot, "int4-g64 arena init");
    NOTE_I4(0, 0, 1); NOTE_I4(0, 1, 1);
    wait_idle();
    check(qt_is_resident(0, 0) && qt_is_resident(0, 1), "int4-g64 arena experts resident");
    check(fake_pipe_allocs == 1 && fake_upload_view_n == 6, "1 slab and 6 view uploads for 2 int4 experts");
    qt_shutdown();
    check(fake_pipe_frees == fake_pipe_allocs, "int4-g64 arena shutdown frees slabs");

    printf("test_qwen36_tier_arena: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
