/* The tier's MXFP4 copy mode (Qwen3.8's experts-mxfp4/ sidecar), on the fake backend.
 *
 * Pinned here: qt_init_mxfp4 starts with cap < n_experts and refuses an odd width;
 * an expert counts half a byte per weight plus one ue8m0 BYTE per row and 32 inputs
 * (not an f32), a short last group rounded up; the upload is fmt 7 with gate and up
 * [Ih, D] and down [D, Ih]; staging copies the nibbles unchanged (no 0x88 XOR, which
 * int4 needs) and the scale bytes in order gate|up|down; the tier keeps no pointer
 * into the engine's slot; a swap writes the newcomer into the victim's tensors with
 * its own scale bytes; the chains are told the residents have no format of theirs
 * (qt_chain_format 0: their MoE kernels have no fmt 7); shutdown leaves the mode. No GPU. */
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

/* gate/up [IH, D]: D/32 = 3 scale bytes a row; down [D, IH]: IH = 40 is a group of 32
 * and a short one of 8, so 2 a row */
enum { NL = 2, NE = 8, D = 96, IH = 40, CAP = 2, TOPK = 2 };
#define MB (D * IH / 2)
#define NGU (IH * ((D + 31) / 32))
#define ND (D * ((IH + 31) / 32))

static unsigned char slab[NE][3 * MB];
static unsigned char e8[NE][2 * NGU + ND];

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
#define NOTE(l, e) qt_note((l), (e), slab[e], slab[e] + MB, slab[e] + 2 * MB, \
                           (const float *)e8[e], (const float *)(e8[e] + NGU), (const float *)(e8[e] + 2 * NGU))

int main(void) {
    for (int e = 0; e < NE; e++) {
        for (size_t i = 0; i < sizeof slab[e]; i++) slab[e][i] = (unsigned char)((i * 7 + e * 13) & 0xFF);
        for (size_t i = 0; i < sizeof e8[e]; i++) e8[e][i] = (unsigned char)(100 + (i + e) % 50);
    }
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("HEAT_FILE", "", 1);
    size_t want = 3 * dev_alloc_footprint(MB) + 2 * dev_alloc_footprint(NGU) + dev_alloc_footprint(ND);
    /* allowance for exactly two experts on the one device */
    char gb[64]; snprintf(gb, sizeof gb, "%.15f", (double)(2 * want + want / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    fake_ndev = 1; fake_uploads = 0;

    printf(" 1. init\n");
    check(!qt_init_mxfp4(NL, NE, D + 1, IH, CAP, TOPK), "odd hidden refused");
    check(!qt_init_mxfp4(NL, NE, D, IH + 1, CAP, TOPK), "odd inter refused");
    check(!G.on && !G_mx4_stream, "a refused init leaves the tier off and the mode unset");
    check(qt_init_mxfp4(NL, NE, D, IH, CAP, TOPK), "mxfp4 mode starts with cap 2 < 8 experts");
    check(G.wfmt == 7 && qt_streaming(), "fmt 7, copy mode");
    check(G.sc_gu == NGU && G.sc_d == ND, "one scale per row and 32 inputs, the short last group counted");
    check(G.exp_bytes == want, "exp_bytes = three nibble matrices + three ue8m0 BYTE tables at cudaMalloc granularity");
    check(qt_mb_gu(0) == MB && qt_mb_d(0) == MB && qt_scb(10) == 10, "staging sizes: half a byte a weight, a byte a scale");
    int fmt = -1;
    check(!qt_chain_format(&fmt, NULL) && fmt == -1, "the chains see no format of theirs (no fmt 7 MoE kernel)");
    int fg = -1, fd = -1;
    check(!qt_chain_layer_format(0, &fg, &fd) && fg == -1 && fd == -1, "nor a layer's");

    printf(" 2. note -> upload\n");
    fake_own_mx4 = 1;   /* the fake keeps copies: the swap below writes into them */
    NOTE(0, 3);
    wait_idle();
    check(qt_is_resident(0, 3), "noted expert becomes resident");
    check(fake_uploads == 3 && last_fmt == 7, "three fmt 7 uploads");
    check(last_bytes == (size_t)MB, "half a byte per weight");
    check(captured_len == (size_t)MB && !memcmp(captured, slab[3], MB), "nibbles staged unchanged (no XOR)");
    check(captured_e8_len[0] == NGU && !memcmp(captured_e8[0], e8[3], NGU), "gate scale bytes");
    check(captured_e8_len[1] == NGU && !memcmp(captured_e8[1], e8[3] + NGU, NGU), "up scale bytes");
    check(captured_e8_len[2] == ND && !memcmp(captured_e8[2], e8[3] + 2 * NGU, ND), "down scale bytes");
    pthread_mutex_lock(&G.mx);
    QSlot *s = qs(0, 3);
    check(s->tg->fmt == 7 && s->tg->I == D && s->tg->O == IH, "gate is [Ih, D]");
    check(s->tu->fmt == 7 && s->tu->I == D && s->tu->O == IH, "up is [Ih, D]");
    check(s->td->fmt == 7 && s->td->I == IH && s->td->O == D, "down is [D, Ih]");
    check(s->g4 == NULL && s->gs == NULL, "no pointer into the engine slot survives qt_note");
    pthread_mutex_unlock(&G.mx);

    printf(" 3. swap in place\n");
    NOTE(0, 4);
    wait_idle();
    check(qt_is_resident(0, 4) && G.used[0] == 2 * G.exp_bytes, "the budget holds two experts");
    for (int i = 0; i < 64 && !qt_is_resident(0, 5); i++) { NOTE(0, 5); wait_idle(); }
    check(qt_is_resident(0, 5), "a hot expert takes a resident's place");
    check(fake_overwrites == 3, "written into the victim's three tensors");
    pthread_mutex_lock(&G.mx);
    s = qs(0, 5);
    check(G.used[0] == 2 * G.exp_bytes, "the swap is budget-neutral");
    check(s->tg->owned && !memcmp(s->tg->w, slab[5], MB) && !memcmp(s->td->w, slab[5] + 2 * MB, MB),
          "the newcomer's nibbles in the victim's tensors");
    check(!memcmp(s->tu->sc, e8[5] + NGU, NGU) && !memcmp(s->td->sc, e8[5] + 2 * NGU, ND),
          "and its scale bytes");
    pthread_mutex_unlock(&G.mx);

    printf(" 4. issue and shutdown\n");
    float x[D]; for (int i = 0; i < D; i++) x[i] = 1.0f;
    int eids[TOPK] = { 5, 6 };
    uint32_t mask = qt_issue(0, eids, TOPK, x);
    check(mask == 0, "the fake refuses the group: everything back to the CPU");
    float out[D]; memset(out, 0, sizeof out); float val[TOPK] = {1, 1};
    check(qt_take(mask, val, TOPK, out), "take with nothing issued");
    qt_shutdown();
    fake_own_mx4 = 0;
    check(!G_mx4_stream && !qt_streaming() && !G.on, "shutdown leaves mxfp4 mode");
    static float lut[256];
    check(qt_init_fp8(NL, NE, 256, 128, CAP, TOPK, lut) && G.wfmt == 8, "an fp8 init afterwards is fmt 8 again");
    check(qt_scb(3) == 3 * sizeof(float), "and its scales f32");
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_mxfp4: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_mxfp4: ok\n");
    return 0;
}
