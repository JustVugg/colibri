/* COLI_CUDA_CHAIN=1: Qwen3.8's layers as one chain on the card give the oracle's tokens.
 *
 * The tiny FP8 fixture (make qwen38-tiny-fp8-generate: 4 layers, two of them
 * DeltaNet, the PLE layer, QSA with its indexer, hidden 32) runs the real engine on
 * the fake CUDA tier with the whole dense trunk placed and computed by the fake
 * (fake_dense_compute), the chain's ops answered by the fake's host-side table
 * (tests/qwen36_fake_cuda.h, the kernels' arithmetic): once with the chain on, once
 * with a frame that fails mid-run (the state rebuilt on the CPU, the run finishing
 * there), once without the chain. Every run must stay within the oracle's limits
 * (ref.json), the chain runs must have set up and run every layer on the device.
 * Then the promotions' brake (q38cc_braked) on frames written by hand: the fixture has
 * no GGUF, so no host pool and no promotion runs above. */
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
/* a frame's books as q38cc_dm_books keeps them: its routes, miss of them read from the pool */
static int brake_frame(Q38CChain *b, unsigned routes, unsigned miss) { b->routes += routes; b->pool_reads += miss; return q38cc_braked(b); }

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
    setenv("Q38_MTP", "0", 1);                     /* no draft: the run stays plain */
    fake_dense_compute = 1;
    char *argv[] = { (char *)"qwen38", (char *)"1", (char *)"8", (char *)"./qwen38_tiny_fp8/ref.json" };

    printf(" a prompt chunk's rows\n");
    unsetenv("COLI_CUDA_CHAIN_ROWS");
    tk(q38cc_rows_env() == 2048, "2048 rows a chunk unset: a 1628-token prompt's pooled experts over the bus once");
    setenv("COLI_CUDA_CHAIN_ROWS", "512", 1);
    tk(q38cc_rows_env() == 512, "COLI_CUDA_CHAIN_ROWS=512 the rows before");
    setenv("COLI_CUDA_CHAIN_ROWS", "0", 1);
    int lo = q38cc_rows_env();
    setenv("COLI_CUDA_CHAIN_ROWS", "99999", 1);
    tk(lo == 1 && q38cc_rows_env() == 8192, "within 1..8192");
    unsetenv("COLI_CUDA_CHAIN_ROWS");

    printf(" the chain on the fake card\n");
    setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_chain_frames = 0; fake_chain_fail_at = 0;
    int rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "engine with every layer on the chain stays within the oracle's limits");
    tk(fake_chain_frames > 8, "the forwards ran through the chain's frames");
    qt_shutdown();

    printf(" a frame that fails mid-run\n");
    fake_chain_frames = 0; fake_chain_fail_at = 30;   /* past the first forward's frames: a decode token's frame fails */
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "the state rebuilt on the CPU, the run within the oracle's limits");
    tk(fake_chain_frames >= 30, "the fault fired");
    fake_chain_fail_at = 0;
    qt_shutdown();

    printf(" the same without the chain\n");
    setenv("COLI_CUDA_CHAIN", "0", 1);
    fake_chain_frames = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "the per-matrix run stays within the oracle's limits");
    tk(fake_chain_frames == 0, "no frame reached the card");
    qt_shutdown();

    printf(" MTP draft rejection with the chain on the fake card\n");
    setenv("SNAP", "./qwen38_tiny_fp8_mtp", 1);
    setenv("Q38_MTP", "1", 1);
    setenv("Q38_MTP_FORCE", "reject", 1);
    argv[3] = (char *)"./qwen38_tiny_fp8_mtp/ref.json";
    setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_chain_frames = 0; fake_chain_fail_at = 0; fake_dn_snapshots = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "rejected MTP drafts roll back and remain within the oracle's limits");
    tk(fake_chain_frames > 8, "rejected MTP verification ran through chain frames");
    tk(fake_dn_snapshots > 0, "MTP verification wrote DeltaNet chain snapshots");
    unsetenv("Q38_MTP_FORCE");
    setenv("Q38_MTP", "0", 1);
    qt_shutdown();

    printf(" the promotions' brake\n");
    unsetenv("COLI_CUDA_CHAIN_PROMOTE_HITS");
    tk(q38cc_promote_hits() == Q38CC_PROMO_HITS, "unset: the default threshold");
    setenv("COLI_CUDA_CHAIN_PROMOTE_HITS", "150", 1);
    int h_hi = q38cc_promote_hits();
    setenv("COLI_CUDA_CHAIN_PROMOTE_HITS", "-3", 1);
    tk(h_hi == 100 && q38cc_promote_hits() == 0, "the threshold clamped to 0-100");
    unsetenv("COLI_CUDA_CHAIN_PROMOTE_HITS");
    static Q38CChain b0;
    Q38CChain b = b0; b.promo_hits = 90;
    tk(brake_frame(&b, 0, 0) == 0, "no routes yet: no brake");
    tk(brake_frame(&b, 1000, 100) == 1, "90 % of the routes on the card: the frame promotes nothing");
    tk(brake_frame(&b, 1000, 300) == 0, "the window at 80 %: promotions again");
    tk(brake_frame(&b, 100000, 2000) == 1, "a prompt chunk weighs its rows: 97.6 % over the window");
    tk(b.pr_braked == 2, "the braked frames counted");
    b = b0; b.promo_hits = 90;
    tk(brake_frame(&b, 1000, 101) == 0, "just under the threshold: no brake");
    b = b0; b.promo_hits = 90;
    int held = 0;
    brake_frame(&b, 100000, 100000);   /* a frame that found nothing on the card */
    for (int f = 1; f < Q38CC_PROMO_WIN; f++) held += brake_frame(&b, 1000, 0);
    tk(held == 0, "a frame of misses holds the window while it is in it");
    tk(brake_frame(&b, 1000, 0) == 1, "past the window it is forgotten");
    b = b0; b.promo_hits = 0;
    tk(brake_frame(&b, 1000, 0) == 0 && b.pr_braked == 0, "COLI_CUDA_CHAIN_PROMOTE_HITS=0: never");

    if (t_fails) { printf("test_qwen38_cuda_chain: %d failure(s)\n", t_fails); return 1; }
    printf("OK test_qwen38_cuda_chain: Qwen3.8's chain on the card gives the oracle's tokens\n");
    return 0;
}
