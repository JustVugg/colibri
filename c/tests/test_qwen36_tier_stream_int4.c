/* GLM's owned int4-gs64 streaming mode, on the fake CUDA backend. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { NL = 2, NE = 16, D = 128, IH = 64, CAP = 4, TOPK = 2 };
#define MB ((size_t)D * IH / 2)
#define SCGU ((size_t)IH * ((D + 63) / 64))
#define SCD ((size_t)D * ((IH + 63) / 64))

static unsigned char slab[3 * MB];
static float scales[2 * SCGU + SCD];

static void wait_idle(void) {
    for (int i = 0; i < 1000; i++) {
        pthread_mutex_lock(&G.mx);
        int pending = G.inflight;
        pthread_mutex_unlock(&G.mx);
        if (!pending) return;
        struct timespec ts = {0, 2000000}; nanosleep(&ts, NULL);
    }
}

static void slow_upload(int fmt) {
    (void)fmt;
    struct timespec ts = {0, 20000000}; nanosleep(&ts, NULL);
}

static int issue_ok(int device, int count, const float *x) {
    (void)device; (void)count; (void)x;
    return 1;
}

static int freed_while_open;
static void note_free(ColiCudaTensor *tensor) {
    (void)tensor;
    if (G.issue_open) freed_while_open++;
}

static void refill_inputs(int seed) {
    for (size_t i = 0; i < sizeof slab; i++) slab[i] = (unsigned char)(i * 7 + seed);
    for (size_t i = 0; i < sizeof scales / sizeof *scales; i++)
        scales[i] = (float)(seed * 1000 + (int)i) / 1000.0f;
}

int main(void) {
    for (size_t i = 0; i < sizeof slab; i++) slab[i] = (unsigned char)(i * 7 + 3);
    for (size_t i = 0; i < sizeof scales / sizeof *scales; i++) scales[i] = (float)i / 1000.0f;
    unsigned char original[sizeof slab]; memcpy(original, slab, sizeof slab);
    float original_scales[sizeof scales / sizeof *scales];
    memcpy(original_scales, scales, sizeof scales);

    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("CUDA_EXPERT_GB", "0.000274658203125", 1);
    char heat_path[] = "/tmp/qtier-stream-int4-XXXXXX";
    int heat_fd = mkstemp(heat_path);
    check(heat_fd >= 0, "creates a streaming heat-file fixture");
    if (heat_fd >= 0) {
        uint32_t header[3] = {0x51544831u, NL, NE};
        uint32_t hot[NL * NE];
        for (int i = 0; i < NL * NE; i++) hot[i] = 1000u + (uint32_t)i;
        check(write(heat_fd, header, sizeof header) == (ssize_t)sizeof header &&
              write(heat_fd, hot, sizeof hot) == (ssize_t)sizeof hot,
              "writes a valid streaming heat-file fixture");
        close(heat_fd);
    }
    setenv("HEAT_FILE", heat_path, 1);
    fake_free_bytes = 2ull << 30;
    fake_uploads = fake_plain_issues = fake_clamped_issues = 0;

    check(!qt_init_stream_int4(0, NE, D, IH, CAP, TOPK, 10.0f) &&
          !qt_init_stream_int4(NL, NE, D, IH, 0, TOPK, 10.0f) &&
          !qt_init_stream_int4(NL, NE, D, IH, NE + 1, TOPK, 10.0f) &&
          !qt_init_stream_int4(NL, NE, D, IH, CAP, 0, 10.0f) &&
          !qt_init_stream_int4(NL, NE, D, IH, CAP, TOPK, 0.0f) &&
          !qt_init_stream_int4(NL, NE, D, IH, CAP, TOPK, NAN),
          "owned int4 rejects invalid geometry and clamp limits");

    check(qt_init_stream_int4(NL, NE, D, IH, CAP, TOPK, 10.0f),
          "owned int4 starts with cap < n_experts");
    check(G.wfmt == 4 && G.egs == 64, "format is int4 group-size 64");
    check(G.sc_gu == SCGU && G.sc_d == SCD, "all group scales are counted");
    check(G.budget[0] / G.exp_bytes == 6,
          "numeric CUDA_EXPERT_GB limits the resident expert count");
    int inherited_heat = G.heat0 != NULL;
    for (int i = 0; i < NL * NE; i++) inherited_heat |= G.slot[i].heat != 0;
    check(!inherited_heat, "owned int4 ignores HEAT_FILE at startup");
    check(qt_resident_count() == 0 && qt_resident_bytes() == 0,
          "resident telemetry starts empty without warmstart");

    fake_upload_hook = slow_upload;
    qt_note(0, 5, slab, slab + MB, slab + 2 * MB,
            scales, scales + SCGU, scales + 2 * SCGU);
    memset(slab, 0, sizeof slab);
    memset(scales, 0, sizeof scales);
    wait_idle();
    check(qt_is_resident(0, 5), "noted expert becomes resident");
    check(qt_resident_count() == 1 && qt_resident_bytes() == G.exp_bytes,
          "resident telemetry counts the promoted expert allocation");
    check(fake_uploads == 3 && last_fmt == 4, "three grouped int4 matrices uploaded");
    int intact = captured_len[0] == MB;
    for (size_t i = 0; intact && i < MB; i++)
        intact = captured[0][i] == (unsigned char)(original[i] ^ 0x88);
    check(intact, "upload owns the packed bytes before slot reuse");
    int all_pieces = 1;
    const size_t nsc[3] = { SCGU, SCGU, SCD };
    const size_t scoff[3] = { 0, SCGU, 2 * SCGU };
    for (int p = 0; p < 3; p++) {
        all_pieces &= captured_len[p] == MB;
        all_pieces &= captured_scale_count[p] == nsc[p];
        for (size_t i = 0; all_pieces && i < MB; i++)
            all_pieces &= captured[p][i] == (unsigned char)(original[(size_t)p * MB + i] ^ 0x88);
        for (size_t i = 0; all_pieces && i < nsc[p]; i++)
            all_pieces &= captured_scales[p][i] == original_scales[scoff[p] + i];
    }
    check(all_pieces, "gate/up/down weights and scales are staged in order");
    pthread_mutex_lock(&G.mx);
    check(!qs(0, 5)->g4 && !qs(0, 5)->gs, "no engine-slot pointer survives qt_note");
    pthread_mutex_unlock(&G.mx);

    float x[D] = {0}, out[D] = {0}, val[TOPK] = {1, 1};
    int eids[TOPK] = {5, 6};
    uint32_t mask = qt_issue(0, eids, TOPK, x);
    check(mask == 0, "failed fake issue returns resident route to CPU");
    check(fake_clamped_issues == 1 && fake_plain_issues == 0,
          "owned int4 uses only the clamped grouped path");
    check(fake_last_swiglu_limit == 10.0f, "SwiGLU limit reaches the backend");
    qt_take(mask, val, TOPK, out);

    /* Even a zero resident mask still closes the issue started by qt_issue.
     * Otherwise the next promotion parks forever behind issue_open. */
    int miss_only = 0;
    mask = qt_issue(0, &miss_only, 1, x);
    check(mask == 0 && G.issue_open, "zero-mask issue still opens a paired issue");
    qt_take(mask, val, 1, out);
    check(!G.issue_open, "zero-mask qt_take closes the paired issue");

    /* A failed second upload must release the successful first allocation and
     * return its reservation instead of stranding a partial expert. */
    refill_inputs(11);
    fake_upload_hook = NULL;
    fake_fail_upload = fake_uploads + 2;
    int live_before = fake_live_tensors, frees_before = fake_frees;
    qt_note(0, 6, slab, slab + MB, slab + 2 * MB,
            scales, scales + SCGU, scales + 2 * SCGU);
    wait_idle();
    check(!qt_is_resident(0, 6), "partial upload failure leaves no resident expert");
    check(fake_live_tensors == live_before && fake_frees == frees_before + 1,
          "partial upload failure frees the successful tensor");
    pthread_mutex_lock(&G.mx);
    check(G.used[0] == G.exp_bytes, "partial upload failure returns the reserved budget");
    pthread_mutex_unlock(&G.mx);
    fake_fail_upload = 0;
    pthread_mutex_lock(&G.mx);
    G.budget[0] = 6 * G.exp_bytes;
    pthread_mutex_unlock(&G.mx);

    /* Fill the automatic allowance, then force a hotter candidate to swap. The
     * uploader must wait for qt_take before freeing the issued victim. */
    refill_inputs(17);
    qt_note(0, 6, slab, slab + MB, slab + 2 * MB,
            scales, scales + SCGU, scales + 2 * SCGU);
    wait_idle();
    check(qt_is_resident(0, 6), "expert uploads again after a transient failure");
    int resident = 0;
    for (int e = 0; e < NE; e++) resident += qt_is_resident(0, e);
    while (resident < (int)(G.budget[0] / G.exp_bytes)) {
        int eid = 7 + resident;
        refill_inputs(eid);
        qt_note(0, eid, slab, slab + MB, slab + 2 * MB,
                scales, scales + SCGU, scales + 2 * SCGU);
        wait_idle();
        resident = 0;
        for (int e = 0; e < NE; e++) resident += qt_is_resident(0, e);
    }
    pthread_mutex_lock(&G.mx);
    qs(0, 5)->heat = 1;
    pthread_mutex_unlock(&G.mx);
    fake_issue_hook = issue_ok;
    fake_free_hook = note_free;
    int victim = 5;
    mask = qt_issue(0, &victim, 1, x);
    check(mask == 1u, "resident victim has an outstanding GPU issue");
    refill_inputs(29);
    for (int i = 0; i < 16; i++)
        qt_note(0, 15, slab, slab + MB, slab + 2 * MB,
                scales, scales + SCGU, scales + 2 * SCGU);
    struct timespec pause = {0, 20000000}; nanosleep(&pause, NULL);
    check(freed_while_open == 0 && qt_is_resident(0, 5) == 0,
          "queued swap does not free its in-flight victim");
    qt_take(mask, val, 1, out);
    wait_idle();
    check(qt_is_resident(0, 15), "hot candidate becomes resident after take");
    check(freed_while_open == 0, "victim tensors are freed only after qt_take");
    resident = 0;
    for (int e = 0; e < NE; e++) resident += qt_is_resident(0, e);
    check(resident == (int)(G.budget[0] / G.exp_bytes),
          "swap keeps the resident count budget-neutral");
    fake_free_hook = NULL;

    /* Repeated promotions must remain budget-neutral: every swap replaces
     * exactly three tensors and never grows the live allocation count. */
    const size_t used_after_fill = G.used[0];
    const int live_after_fill = fake_live_tensors;
    const int uploads_before_stress = fake_uploads;
    const int frees_before_stress = fake_frees;
    enum { SWAPS = 24 };
    for (int n = 0; n < SWAPS; n++) {
        int cold = -1, hot = -1;
        pthread_mutex_lock(&G.mx);
        for (int e = 0; e < NE; e++) {
            QSlot *s = qs(0, e);
            if (s->resident) {
                s->heat = 100000;
                if (cold < 0) cold = e;
            } else if (!s->queued && hot < 0) hot = e;
        }
        qs(0, cold)->heat = 1;
        qs(0, hot)->heat = 200000;
        pthread_mutex_unlock(&G.mx);

        refill_inputs(100 + n);
        qt_note(0, hot, slab, slab + MB, slab + 2 * MB,
                scales, scales + SCGU, scales + 2 * SCGU);
        wait_idle();
        check(qt_is_resident(0, hot) && !qt_is_resident(0, cold),
              "hot candidate replaces the selected cold resident");

        resident = 0;
        for (int e = 0; e < NE; e++) resident += qt_is_resident(0, e);
        check(resident == (int)(G.budget[0] / G.exp_bytes) &&
              G.used[0] == used_after_fill &&
              fake_live_tensors == live_after_fill,
              "repeated swaps keep resident, budget, and allocation counts stable");
    }
    check(fake_uploads == uploads_before_stress + 3 * SWAPS &&
          fake_frees == frees_before_stress + 3 * SWAPS,
          "each repeated swap uploads and frees exactly three tensors");

    int resident_tensors = fake_live_tensors;
    qt_shutdown();
    check(qt_resident_count() == 0 && qt_resident_bytes() == 0,
          "resident telemetry clears at shutdown");
    check(resident_tensors > 0 && fake_live_tensors == 0,
          "shutdown frees every resident expert tensor");
    check(!G.slot && !G.is_x && !G.heat0 && !G.fill_order,
          "shutdown frees tier-owned host allocations");
    FILE *heat = fopen(heat_path, "rb");
    uint32_t kept_header[3] = {0, 0, 0};
    check(heat && fread(kept_header, sizeof kept_header, 1, heat) == 1 &&
          kept_header[0] == 0x51544831u && kept_header[1] == NL && kept_header[2] == NE,
          "owned int4 does not overwrite HEAT_FILE at shutdown");
    if (heat) fclose(heat);
    unlink(heat_path);
    unsetenv("HEAT_FILE");

    check(!qt_init(NL, NE, D, IH, CAP, TOPK, 64, 1),
          "ordinary qwen int4 still rejects partial RAM residency");

    setenv("CUDA_EXPERT_GB", "auto", 1);
    fake_free_bytes = 2ull << 30;
    check(qt_init_stream_int4(NL, NE, D, IH, CAP, TOPK, 10.0f),
          "owned int4 starts with automatic sizing");
    check(G.budget[0] == fake_free_bytes - (3ull << 29),
          "automatic GLM sizing reserves 1.5 GiB");
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_stream_int4: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_stream_int4: ok\n");
    return 0;
}
