/* Real-GPU parity for the int4-gs64 streaming tier (the glm53 path).
 *
 * The fake-backend tests prove the tier's bookkeeping -- staging order, budget
 * neutrality, swap safety. They cannot prove the numbers: the corruption the
 * glm53 acceptance run saw (plausible-looking word fragments in
 * reasoning_content with the tier on, clean output with COLI_CUDA=0) lives
 * somewhere between the XOR-0x88 staging, the upload, offset_to_signed_s4 and
 * the g4 group kernels -- all of which the fake backend replaces with a memcpy.
 * This test runs one expert's six pieces through the engine's own CPU
 * reference (matmul_i4_grouped + the clamped SwiGLU glm53 uses) and through
 * the tier's full note -> upload -> issue -> take round trip on a real device,
 * at glm53-flash geometry, and compares. */
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

/* glm53-flash: hidden 4096, moe_inter 2048, gs 64, swiglu_limit 10. */
enum { NL = 1, NE = 4, D = 4096, IH = 2048, GS = 64, TOPK = 4 };
#define MB ((size_t)D * IH / 2)
#define SCGU ((size_t)IH * (D / GS))
#define SCD ((size_t)D * (IH / GS))
static const float LIMIT = 10.0f;

static unsigned char w[NE][3][MB];
static float sc[NE][3][SCGU < SCD ? SCD : SCGU];   /* gate/up: SCGU, down: SCD (equal here) */

static unsigned rng_state = 12345;
static unsigned rng(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }

static void fill_expert(int e, float scale_lo, float scale_hi) {
    for (int m = 0; m < 3; m++)
        for (size_t i = 0; i < MB; i++) w[e][m][i] = (unsigned char)rng();
    size_t n[3] = { SCGU, SCGU, SCD };
    for (int m = 0; m < 3; m++)
        for (size_t i = 0; i < n[m]; i++)
            sc[e][m][i] = scale_lo + (scale_hi - scale_lo) * (rng() & 0xFFFF) / 65535.0f;
}

/* glm53's CPU expert path: mv(fmt=4) + swiglu_clamped + mv(fmt=4). */
static void cpu_expert(float *out, int e, const float *x, float *sg, float *su) {
    matmul_i4_grouped(sg, x, w[e][0], sc[e][0], 1, D, IH, GS);
    matmul_i4_grouped(su, x, w[e][1], sc[e][1], 1, D, IH, GS);
    for (int i = 0; i < IH; i++) {
        float g = sg[i] > LIMIT ? LIMIT : sg[i];
        float u = su[i] < -LIMIT ? -LIMIT : (su[i] > LIMIT ? LIMIT : su[i]);
        sg[i] = (g / (1.0f + expf(-g))) * u;
    }
    matmul_i4_grouped(out, sg, w[e][2], sc[e][2], 1, IH, D, GS);
}

static int wait_resident(int layer, int eid, int ms) {
    for (int i = 0; i < ms; i++) {
        if (qt_is_resident(layer, eid)) return 1;
        struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL);
    }
    return qt_is_resident(layer, eid);
}

static int compare(const float *got, const float *ref, const char *what) {
    double max_abs = 0, max_rel = 0; int at = 0;
    for (int i = 0; i < D; i++) {
        double a = fabs((double)got[i] - ref[i]);
        double r = a / (1.0 + fabs((double)ref[i]));
        if (r > max_rel) { max_rel = r; max_abs = a; at = i; }
    }
    int ok = max_rel <= 2e-3;
    printf("  %s: %s (max rel %.2e abs %.2e at %d, got %.4f ref %.4f)\n",
           what, ok ? "ok" : "MISMATCH", max_rel, max_abs, at, got[at], ref[at]);
    if (!ok) fails++;
    return ok;
}

int main(void) {
    if (coli_cuda_available_device_count() < 1) {
        printf("test_qwen36_tier_stream_int4_parity: skip (no CUDA device)\n");
        return 0;
    }
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("COLI_PLACE", "off", 1);
    setenv("CUDA_EXPERT_GB", "0.2", 1);

    check(qt_init_stream_int4(NL, NE, D, IH, NE, TOPK, LIMIT),
          "stream int4 tier starts at glm53 geometry");

    /* Case 1: moderate scales, single expert, unit weight. */
    fill_expert(1, 0.005f, 0.02f);
    qt_note(0, 1, w[1][0], w[1][1], w[1][2], sc[1][0], sc[1][1], sc[1][2]);
    check(wait_resident(0, 1, 30000), "expert 1 uploads and becomes resident");

    static float x[D], sg[IH], su[IH], ref[D], out[D];
    for (int i = 0; i < D; i++) x[i] = (float)((int)(rng() & 0xFFFF) - 32768) / 32768.0f;
    cpu_expert(ref, 1, x, sg, su);

    int one = 1;
    uint32_t mask = qt_issue(0, &one, 1, x);
    check(mask == 1u, "resident expert is issued to the GPU");
    float val[1] = { 1.0f };
    memset(out, 0, sizeof out);
    qt_take(mask, val, 1, out);
    compare(out, ref, "single expert GPU vs CPU");

    /* Case 2: a K-expert group with distinct routing weights (the decode
     * shape: one token, topk distinct experts, qt_take merges rows). */
    fill_expert(2, 0.005f, 0.02f);
    fill_expert(3, 0.005f, 0.02f);
    qt_note(0, 2, w[2][0], w[2][1], w[2][2], sc[2][0], sc[2][1], sc[2][2]);
    qt_note(0, 3, w[3][0], w[3][1], w[3][2], sc[3][0], sc[3][1], sc[3][2]);
    check(wait_resident(0, 2, 30000) && wait_resident(0, 3, 30000),
          "experts 2 and 3 become resident");

    float ref2[D], ref3[D], refm[D];
    cpu_expert(ref2, 2, x, sg, su);
    cpu_expert(ref3, 3, x, sg, su);
    for (int i = 0; i < D; i++) refm[i] = 0.7f * ref2[i] + 0.3f * ref3[i];
    int pair[2] = { 2, 3 };
    float pval[2] = { 0.7f, 0.3f };
    mask = qt_issue(0, pair, 2, x);
    check(mask == 3u, "both resident experts are issued");
    memset(out, 0, sizeof out);
    qt_take(mask, pval, 2, out);
    compare(out, refm, "two-expert weighted group GPU vs CPU");

    /* Case 3: scales large enough that the SwiGLU clamp actually fires on
     * both sides -- the clamped kernel epilogue is a separate code path. */
    fill_expert(0, 0.08f, 0.25f);
    qt_note(0, 0, w[0][0], w[0][1], w[0][2], sc[0][0], sc[0][1], sc[0][2]);
    check(wait_resident(0, 0, 30000), "expert 0 becomes resident");
    int clamped_hits = 0;
    {
        float tsg[IH], tsu[IH];
        matmul_i4_grouped(tsg, x, w[0][0], sc[0][0], 1, D, IH, GS);
        matmul_i4_grouped(tsu, x, w[0][1], sc[0][1], 1, D, IH, GS);
        for (int i = 0; i < IH; i++)
            clamped_hits += (tsg[i] > LIMIT) || (tsu[i] > LIMIT) || (tsu[i] < -LIMIT);
    }
    printf("  clamp fires on %d/%d intermediate rows\n", clamped_hits, IH);
    cpu_expert(ref, 0, x, sg, su);
    int zero = 0;
    mask = qt_issue(0, &zero, 1, x);
    check(mask == 1u, "expert 0 is issued");
    memset(out, 0, sizeof out);
    qt_take(mask, val, 1, out);
    if (clamped_hits > 0)
        compare(out, ref, "clamped expert GPU vs CPU");
    else
        printf("  NOTE: random data did not exercise the clamp; rerun with larger scales\n");

    qt_stats();
    qt_shutdown();
    if (fails) { printf("test_qwen36_tier_stream_int4_parity: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_stream_int4_parity: ok\n");
    return 0;
}

#else /* !COLI_CUDA */
int main(void) {
    printf("test_qwen36_tier_stream_int4_parity: skip (built without CUDA)\n");
    return 0;
}
#endif
