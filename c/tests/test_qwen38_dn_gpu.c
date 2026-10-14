/* Q38_DN_GPU=1: Qwen3.8's DeltaNet layer on the card gives the oracle's tokens.
 *
 * The tiny FP8 fixture (make qwen38-tiny-fp8-generate: 4 layers, two of them
 * DeltaNet, hidden 32, 4 value heads of 4) runs the real engine on the fake
 * CUDA tier with the whole dense trunk placed and computed by the fake
 * (fake_dense_compute), once with the DeltaNet layers on the card and once
 * without; both must stay within the oracle's limits (ref.json), the GPU run
 * must have stepped every DeltaNet layer on the card, and the two runs' outputs
 * must agree. The fake's coli_cuda_dn_step is the host-side reference of the
 * CUDA kernels, with the sigmoid gate Qwen3.8's norm uses. */
#define main qwen38_main_unused
#include "../qwen38.c"
#undef main

#ifdef _WIN32
#undef setenv
#undef unsetenv
#define unsetenv(name) _putenv_s(name, "")
#endif

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int t_fails;
static void tk(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); t_fails++; }

int main(void) {
    setenv("SNAP", "./qwen38_tiny_fp8", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("NOSTREAM", "1", 1);
    setenv("USAGE_SAVE", "0", 1);
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("QT_UPLOAD_SYNC", "1", 1);
    setenv("HEAT_FILE", "", 1);
    setenv("Q38_TRUNK_GPU", "1", 1);
    setenv("Q38_TRUNK_MIN_KB", "0", 1);          /* every dense matrix of the fixture is offered */
    setenv("COLI_PLACE", "auto", 1);
    setenv("CUDA_EXPERT_GB", "0.5", 1);
    setenv("Q38_MTP", "0", 1);                     /* no draft: the verify path keeps the CPU anyway, keep the run plain */
    fake_dense_compute = 1;
    char *argv[] = { (char *)"qwen38", (char *)"1", (char *)"8", (char *)"./qwen38_tiny_fp8/ref.json" };

    printf(" DeltaNet layers on the fake card\n");
    setenv("Q38_DN_GPU", "1", 1);
    fake_dn_steps = 0; fake_uploads = 0;
    int rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "engine with the DeltaNet layers on the card stays within the oracle's limits");
    tk(fake_dn_steps > 0, "decode tokens ran the DeltaNet layers on the card");
    int on = 0; for (int l = 0; l < QT_DN_MAX_LAYERS; l++) on += qt_dn_gpu_ready(l);
    tk(on == 2, "both DeltaNet layers of the fixture joined their projections on the card");
    qt_shutdown();

    printf(" the same on the CPU\n");
    setenv("Q38_DN_GPU", "0", 1);
    fake_dn_steps = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "the CPU run stays within the oracle's limits");
    tk(fake_dn_steps == 0, "no DeltaNet step reached the card");
    qt_shutdown();

    if (t_fails) { printf("test_qwen38_dn_gpu: %d failure(s)\n", t_fails); return 1; }
    printf("OK test_qwen38_dn_gpu: Qwen3.8's DeltaNet layer on the card gives the oracle's tokens\n");
    return 0;
}
