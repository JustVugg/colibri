/* Real-GPU parity for qwen36's FP8 tier at the production expert shape.
 * The fake tier test proves staging and residency. This test proves the whole
 * qt_init_fp8 -> note -> fmt=8 upload -> issue -> take path against quant.h's
 * CPU E4M3/block-scale reference on a real CUDA backend. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../compat.h"
#ifdef COLI_CUDA
#include "../qwen36_tier.c"
#include "../quant.h"
static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}
enum { NL = 1, NE = 2, D = 2048, IH = 768, TOPK = 2 };
#define MB ((size_t)D * IH)
#define NSC ((size_t)((D + 127) / 128) * ((IH + 127) / 128))
static unsigned char w[NE][3][MB];
static float sc[NE][3][NSC];
static unsigned rng_state = 8;
static unsigned rng(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }
static unsigned char random_e4m3(void) {
    unsigned char b = (unsigned char)rng();
    if ((b & 0x7f) == 0x7f) b &= (unsigned char)~1u;
    return b;
}
static void fill_expert(int e) {
    for (int m = 0; m < 3; m++) {
        for (size_t i = 0; i < MB; i++) w[e][m][i] = random_e4m3();
        for (size_t i = 0; i < NSC; i++)
            sc[e][m][i] = ldexpf(1.f + (rng() & 0xffff) / 65535.f, -12);
    }
}
static void cpu_expert(float *out, int e, const float *x, float *gate, float *up) {
    matmul_fp8(gate, x, w[e][0], sc[e][0], 1, D, IH);
    matmul_fp8(up, x, w[e][1], sc[e][1], 1, D, IH);
    for (int i = 0; i < IH; i++) gate[i] = (gate[i] / (1.f + expf(-gate[i]))) * up[i];
    matmul_fp8(out, gate, w[e][2], sc[e][2], 1, IH, D);
}
static int wait_resident(int eid) {
    for (int i = 0; i < 30000; i++) {
        if (qt_is_resident(0, eid)) return 1;
        struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL);
    }
    return qt_is_resident(0, eid);
}
static void compare(const float *got, const float *ref, const char *what) {
    double max_abs = 0, max_rel = 0; int at = 0;
    for (int i = 0; i < D; i++) {
        double a = fabs((double)got[i] - ref[i]);
        double r = a / (1.0 + fabs((double)ref[i]));
        if (r > max_rel) { max_rel = r; max_abs = a; at = i; }
    }
    int ok = max_rel <= 2e-3;
    printf("  %s: %s (max rel %.2e abs %.2e at %d)\n",
           what, ok ? "ok" : "MISMATCH", max_rel, max_abs, at);
    if (!ok) fails++;
}
int main(void) {
    if (coli_cuda_available_device_count() < 1) {
        printf("test_qwen36_tier_fp8_parity: skip (no CUDA device)\n");
        return 0;
    }
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("CUDA_EXPERT_GB", "0.1", 1);
    check(qt_init_fp8(NL, NE, D, IH, NE, TOPK, E4M3_LUT),
          "FP8 tier starts at qwen36 expert geometry");
    static float x[D], gate[IH], up[IH], ref0[D], ref1[D], refm[D], out[D];
    for (int i = 0; i < D; i++) x[i] = (float)((int)(rng() & 0xffff) - 32768) / 32768.f;

    fill_expert(0);
    qt_note(0, 0, w[0][0], w[0][1], w[0][2], sc[0][0], sc[0][1], sc[0][2]);
    check(wait_resident(0), "expert 0 uploads and becomes resident");
    cpu_expert(ref0, 0, x, gate, up);
    int eid = 0; float val[1] = {1.f};
    uint32_t mask = qt_issue(0, &eid, 1, x);
    check(mask == 1u, "resident FP8 expert is issued");
    memset(out, 0, sizeof out); qt_take(mask, val, 1, out);
    compare(out, ref0, "single expert GPU vs CPU");

    fill_expert(1);
    qt_note(0, 1, w[1][0], w[1][1], w[1][2], sc[1][0], sc[1][1], sc[1][2]);
    check(wait_resident(1), "expert 1 uploads and becomes resident");
    cpu_expert(ref1, 1, x, gate, up);
    for (int i = 0; i < D; i++) refm[i] = 0.7f * ref0[i] + 0.3f * ref1[i];
    int pair[2] = {0, 1}; float weights[2] = {0.7f, 0.3f};
    mask = qt_issue(0, pair, 2, x);
    check(mask == 3u, "both FP8 experts are issued as one group");
    memset(out, 0, sizeof out); qt_take(mask, weights, 2, out);
    compare(out, refm, "two-expert weighted group GPU vs CPU");
    qt_stats(); qt_shutdown();
    if (fails) { printf("test_qwen36_tier_fp8_parity: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_fp8_parity: ok\n");
    return 0;
}
#else
int main(void) {
    printf("test_qwen36_tier_fp8_parity: skip (built without CUDA)\n");
    return 0;
}
#endif
