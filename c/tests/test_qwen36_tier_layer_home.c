/* QT_HOME=layer: expert homes per layer range, the cards as a pipeline.
 *
 * By default expert eid lives on device eid % ndev in every layer, so every
 * layer's group joins across the cards and the slower card paces each layer
 * (measured: take 5.2 ms/token on two unequal cards against 0.2 on one). With
 * QT_HOME=layer every expert of a layer lives on that layer's device, the
 * automatic placer puts the layer's trunk components on the same device and
 * lm_head on the last layer's, and a group is issued to exactly one device.
 * Fake CUDA backend, two fake devices, no GPU. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../compat.h"
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("  FAIL: %s\n", what); fails++; } }

enum { NL = 8, NE = 8, D = 64, IH = 32, TOPK = 2 };
#define MiB (1024ull * 1024ull)

static int issued_dev[64], issued_n;
static int record_issue(int device, int count, const float *x) { (void)count; (void)x; if (issued_n < 64) issued_dev[issued_n++] = device; return 1; }

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0,1", 1); setenv("QT_NO_WARMSTART", "1", 1);
    setenv("HEAT_FILE", "", 1); setenv("COLI_PLACE", "", 1); setenv("QT_HOME", "layer", 1); setenv("QT_LAYER_SPLIT", "", 1);
    setenv("CUDA_EXPERT_GB", "0.0625", 1);        /* 64 MiB per card */
    fake_ndev = 2; fake_uploads = 0; fake_issue_hook = record_issue;

    /* offers: lmhead once, dnproj and dnout for the DeltaNet layers (every layer but 3 and 7) */
    G_offer_n = 0; G_auto_on = 0;
    qt_trunk_offer("lmhead", 0, 4 * MiB);
    for (int l = 0; l < NL; l++) if (l % 4 != 3) { qt_trunk_offer("dnproj", l, 2 * MiB); qt_trunk_offer("dnout", l, 1 * MiB); }
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts on two fake devices in layer mode");
    check(G_home_layer && G.ndev == 2, "layer homes are on, both cards kept");
    /* equal allowances: layers 0-3 on device index 0, 4-7 on 1 */
    int ok = 1; for (int l = 0; l < NL; l++) if (G_layer_dev[l] != (l < 4 ? 0 : 1)) ok = 0;
    check(ok, "equal allowances split the layers in half");
    ok = 1; for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) if (home2(l, e) != G_layer_dev[l]) ok = 0;
    check(ok, "every expert of a layer is homed on that layer's device");
    check(qt_place_of("dnproj", 0) == 0 && qt_place_of("dnout", 0) == 0 && qt_place_of("dnproj", 6) == 1 && qt_place_of("dnout", 6) == 1,
          "the automatic placer puts a layer's trunk on the layer's device");
    check(qt_place_of("lmhead", 0) == 1, "lm_head goes to the last layer's device");
    check(G.budget[0] == 64 * MiB - 9 * MiB && G.budget[1] == 64 * MiB - 13 * MiB, "each card's budget charges its own layers' trunk (3 x 3 MiB; 3 x 3 + 4 MiB)");

    /* residents follow the homes: note every expert, wait, read back */
    static unsigned char g4[NL][NE][D * IH / 2], u4[NL][NE][D * IH / 2], d4[NL][NE][D * IH / 2];
    static float sc[NL][NE][2 * IH + D];
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) {
        memset(g4[l][e], 1, sizeof g4[l][e]); memset(u4[l][e], 2, sizeof u4[l][e]); memset(d4[l][e], 3, sizeof d4[l][e]);
        for (int i = 0; i < 2 * IH + D; i++) sc[l][e][i] = 1.f;
        qt_note_block(l, e, g4[l][e], u4[l][e], d4[l][e], sc[l][e], sc[l][e] + IH, sc[l][e] + 2 * IH);
    }
    qt_fill_wait();
    int res0 = 0, res1 = 0;
    pthread_mutex_lock(&G.mx);
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) if (qs(l, e)->resident) { if (G_layer_dev[l] == 0) res0++; else res1++; }
    pthread_mutex_unlock(&G.mx);
    check(res0 == 4 * NE && res1 == 4 * NE, "all 64 experts resident, 32 per card, each on its layer's card");
    check(G.used[0] == 4 * NE * G.exp_bytes && G.used[1] == 4 * NE * G.exp_bytes, "the bytes were charged to the layer's card");

    /* a group goes to one device: layer 0 to the first card, layer 7 to the second, mixed eids notwithstanding */
    float x[D]; for (int i = 0; i < D; i++) x[i] = 0.5f;
    int eids[TOPK] = { 1, 6 }; const float val[TOPK] = { 0.5f, 0.5f }; float out[D];
    issued_n = 0;
    uint32_t m0 = qt_issue(0, eids, TOPK, x); qt_take(m0, val, TOPK, out);
    uint32_t m7 = qt_issue(7, eids, TOPK, x); qt_take(m7, val, TOPK, out);
    check(m0 == 3u && m7 == 3u, "both experts of each group hit VRAM");
    check(issued_n == 2 && issued_dev[0] == 0 && issued_dev[1] == 1, "layer 0's group went to device 0 alone, layer 7's to device 1 alone");
    qt_shutdown();

    /* QT_LAYER_SPLIT overrides the proportional split */
    setenv("QT_LAYER_SPLIT", "6", 1); G_offer_n = 0; G_auto_on = 0; fake_uploads = 0;
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts with an explicit split");
    ok = 1; for (int l = 0; l < NL; l++) if (G_layer_dev[l] != (l < 6 ? 0 : 1)) ok = 0;
    check(ok, "QT_LAYER_SPLIT=6: layers 0-5 on the first card, 6-7 on the second");
    qt_shutdown();

    /* one card: layer mode is a no-op, expert homes as before */
    setenv("COLI_GPUS", "0", 1); fake_ndev = 1; G_offer_n = 0; G_auto_on = 0;
    check(qt_init(NL, NE, D, IH, NE, TOPK, 0, 1), "tier starts on one card");
    check(!G_home_layer && home2(3, 5) == 0, "one card: no layer homes needed");
    qt_shutdown();

    if (fails) { printf("test_qwen36_tier_layer_home: %d failure(s)\n", fails); return 1; }
    printf("OK test_qwen36_tier_layer_home: experts and trunk homed per layer range, one device per group\n");
    return 0;
}
