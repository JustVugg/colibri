/* A REFUSED qt_init_stream_int4 / qt_init_fp8 must leave the running tier
 * exactly as it was (#1564, review of c33679dc, HIGH 3).
 *
 * The defect. Both wrappers set their mode globals (G_int4_stream and the
 * SwiGLU limit, G_fp8_stream and the LUT) BEFORE qt_init decided anything, and
 * cleared the flag again on refusal. Against a live streaming tier the refusal
 * therefore switched G_int4_stream off: the next issue took the plain grouped
 * path instead of the clamped one, and stage() began XOR-flipping bytes that
 * were already offset-binary. A refusal must be side-effect free.
 *
 * No threads needed: the tier is live, the second init is refused by the
 * lifecycle state, and the mode is read both directly and by what the next
 * issue does at the fake backend. */
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
static const float other_lut[256];
static int issue_ok(int device, int count, const float *x) { (void)device; (void)count; (void)x; return 1; }
int main(void) {
    printf("qwen36 tier refused init: a refusal changes nothing\n");
    for (size_t i = 0; i < sizeof slab; i++) slab[i] = (unsigned char)(i * 7 + 3);
    for (size_t i = 0; i < sizeof scales / sizeof *scales; i++) scales[i] = (float)i / 1000.0f;
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("COLI_PLACE", "off", 1); setenv("CUDA_EXPERT_GB", "0.000274658203125", 1);
    unsetenv("HEAT_FILE");
    int first = qt_init_stream_int4(NL, NE, D, IH, CAP, TOPK, 10.0f);
    printf("FIRST_INIT_RESULT=%d\n", first);
    if (!first) { printf("  FAIL: streaming tier did not start\n"); return 1; }
    qt_note(0, 5, slab, slab + MB, slab + 2 * MB, scales, scales + SCGU, scales + 2 * SCGU);
    qt_fill_wait();
    check(qt_is_resident(0, 5), "the noted expert is resident");
    printf("STREAM_FLAG_BEFORE_REFUSAL=%d\n", G_int4_stream);
    int second = qt_init_stream_int4(NL, NE, D, IH, CAP, TOPK, 3.0f);
    printf("SECOND_INIT_RESULT=%d\nSTREAM_FLAG_AFTER_REFUSAL=%d\n", second, G_int4_stream);
    check(second == 0, "a second stream init against a live tier is refused");
    check(G_int4_stream == 1, "refused stream re-init disabled the live tier mode");
    check(G_stream_swiglu_limit == 10.0f, "refused stream re-init rewrote the live SwiGLU limit");
    int third = qt_init_fp8(NL, NE, D, IH, CAP, TOPK, other_lut);
    check(third == 0, "an fp8 init against a live tier is refused");
    check(G_fp8_stream == 0 && G_fp8_lut == NULL, "refused fp8 init switched the live tier's mode");
    check(G_int4_stream == 1, "refused fp8 init disabled the live stream mode");
    fake_issue_hook = issue_ok;
    fake_plain_issues = fake_clamped_issues = 0;
    float x[D] = {0}, out[D] = {0}, val[1] = {1};
    int eid = 5;
    uint32_t mask = qt_issue(0, &eid, 1, x);
    qt_take(mask, val, 1, out);
    printf("PLAIN_ISSUES=%d\nCLAMPED_ISSUES=%d\n", fake_plain_issues, fake_clamped_issues);
    check(fake_clamped_issues == 1 && fake_plain_issues == 0,
          "after the refusal the live tier still issues clamped, not plain");
    check(fake_last_swiglu_limit == 10.0f, "the backend still gets the live tier's limit");
    fake_issue_hook = NULL;
    qt_shutdown();
    check(G_int4_stream == 0 && G_fp8_stream == 0, "the teardown resets the mode");
    /* a fresh stream init after the teardown still works, and a plain one
     * after that does not inherit the stream mode */
    check(qt_init_stream_int4(NL, NE, D, IH, CAP, TOPK, 7.0f) == 1 && G_int4_stream == 1 &&
          G_stream_swiglu_limit == 7.0f, "a stream init after the teardown is accepted with its own limit");
    qt_shutdown();
    if (fails) { printf("test_qwen36_tier_refused_init: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_refused_init: ok\n");
    return 0;
}
