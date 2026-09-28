/* A qt_shutdown over LIVE trunk-only storage must actually free it (#1564,
 * review of e6c70eb7, HIGH A).
 *
 * The defect. G_dnp[] lives OUTSIDE G on purpose -- the dense projections are
 * uploaded and used with the expert tier never started, which is why they are
 * admitted by qt_gate_enter_trunk() (which admits QT_DEAD). Nothing tracked
 * that ownership, and the top of qt_shutdown was:
 *
 *     if(qg_get()!=QT_LIVE){ ...wait for TEARING_DOWN...; return; }
 *
 * So a tier that had never been started -- state QT_DEAD, exactly what the
 * early return tests for -- returned at once and freed NOTHING. G_dnp[].t
 * stayed resident: qt_dnproj_matmul kept running on it through the trunk gate
 * (POST_SHUTDOWN_GPU=1), and a later qt_dnproj_init of the SAME SHAPE was
 * swallowed by the backend's cached-tensor path -- backend_cuda.cu returns 1
 * from a non-NULL *tensor whose fmt/I/O/device/gs match, WITHOUT copying
 * (STALE_TENSOR_REUSED=1) -- so the new generation answered with the previous
 * generation's weights. A silent wrong answer, not a fallback.
 *
 * The backend's cached path is modelled here (fake_cached_upload, below): the
 * plain fake always allocates, so against it the stale upload would look like
 * a fresh one and this test would pass with the defect in place. Everything
 * else is counted at the fake, and nothing here uses a clock to decide. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"
static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}
enum { PI = 8, PO = 4, LAYER = 5 };
/* Two weight sets, same shape. gen A is 1..N, gen B is -1..-N: a matmul that
 * answers with the wrong one is a wrong answer, not a tolerance question. */
static int8_t qa[PI * PO], qb[PI * PO];
static float sa[PO], sb[PO], px[PI], py[PO];
/* What the layer MUST compute once it holds gen B's weights. */
static void expect_b(float *y) {
    for (int o = 0; o < PO; o++) {
        float a = 0.f;
        for (int i = 0; i < PI; i++) a += px[i] * (float)qb[(size_t)o * PI + i];
        y[o] = a * sb[o];
    }
}
static volatile int alive = 1;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { 60, 0 };
    nanosleep(&limit, NULL);
    if (alive) { (void)!write(2, "FAIL: trunk teardown test hung\n", 29); _exit(2); }
    return NULL;
}
int main(void) {
    printf("qwen36 tier trunk teardown: shutdown frees storage it did not start\n");
    for (int i = 0; i < PI * PO; i++) { qa[i] = (int8_t)(i + 1); qb[i] = (int8_t)(-(i + 1)); }
    for (int o = 0; o < PO; o++) { sa[o] = 1.0f; sb[o] = 1.0f; }
    for (int i = 0; i < PI; i++) px[i] = 1.0f;
    pthread_t wd;
    if (pthread_create(&wd, NULL, watchdog, NULL) == 0) pthread_detach(wd);
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("CUDA_EXPERT_GB", "auto", 1);
    /* The real backend's cached-tensor path: a non-NULL *tensor is matched
     * against the request, not replaced by it. Plus real fmt-1 arithmetic, so
     * "which weights is this layer actually computing with" has an oracle here
     * instead of a pointer comparison. */
    fake_cached_upload = 1; fake_dense_compute = 1;
    /* ---- trunk-only: the expert tier is never started -------------------- */
    check(qg_get() == QT_DEAD, "the expert tier has never been started");
    check(qt_dnproj_init(LAYER, qa, sa, PI, PO, 0) == 1, "the projection uploads with the tier off");
    check(fake_live_tensors == 1 && G_dnp[LAYER].t != NULL, "the projection is resident");
    check(G_dnp[LAYER].on == 1, "and the layer is placed on the device");
    int frees_before = fake_frees;
    /* ---- the teardown of a tier that is already DEAD --------------------- */
    qt_shutdown();
    int post_shutdown_gpu = qt_dnproj_matmul(LAYER, py, px, PI, PO);
    printf("LIVE_AFTER=%d FREES=%d DNP_ON=%d\n", fake_live_tensors, fake_frees - frees_before, G_dnp[LAYER].on);
    printf("POST_SHUTDOWN_GPU=%d\n", post_shutdown_gpu);
    check(fake_frees - frees_before == 1, "qt_shutdown freed the live trunk-only tensor");
    check(fake_live_tensors == 0 && G_dnp[LAYER].t == NULL, "no trunk tensor survived the shutdown");
    check(G_dnp[LAYER].on == 0, "the layer is back on the CPU");
    check(qg_trunk == 0, "the trunk ownership count is back to zero");
    check(post_shutdown_gpu == 0, "no projection matmul runs on storage the shutdown freed");
    /* ---- a later init must copy ITS OWN weights, not inherit the old ------ */
    int hits_before = fake_cached_hits, ups_before = fake_uploads;
    int re = qt_dnproj_init(LAYER, qb, sb, PI, PO, 0);
    int stale = fake_cached_hits - hits_before;
    printf("REINIT=%d STALE_TENSOR_REUSED=%d NEW_ALLOC=%d\n",
           re, stale, fake_uploads - ups_before);
    check(re == 1, "the projection re-uploads after the shutdown");
    check(stale == 0, "the same-shaped init was swallowed by the backend's cached path");
    check(fake_uploads - ups_before == 1, "the same-shaped init allocated a fresh tensor");
    memset(py, 0, sizeof py);
    check(qt_dnproj_matmul(LAYER, py, px, PI, PO) == 1, "the layer computes on the new generation");
    float want[PO];
    expect_b(want);
    int wrong = 0;
    for (int o = 0; o < PO; o++) if (py[o] != want[o]) wrong = 1;
    printf("POST_REINIT_GPU=%d STALE_WEIGHTS=%d\n", qt_dnproj_matmul(LAYER, py, px, PI, PO), wrong);
    check(!wrong, "the projection answered with the weights it was given");
    /* and a second shutdown over THAT storage frees it again -- the fix is
     * ownership, not a one-shot clear */
    frees_before = fake_frees;
    qt_shutdown();
    printf("SECOND_FREES=%d\n", fake_frees - frees_before);
    check(fake_frees - frees_before == 1 && fake_live_tensors == 0,
          "a second qt_shutdown frees the new generation's tensor too");
    check(qt_dnproj_matmul(LAYER, py, px, PI, PO) == 0, "and the layer is CPU-only again");
    /* ---- nothing above needed a live tier: shutdown twice more is a no-op -- */
    frees_before = fake_frees;
    qt_shutdown(); qt_shutdown();
    check(fake_frees == frees_before, "a shutdown over nothing frees nothing");
    check(qg_get() == QT_DEAD, "and the state is still QT_DEAD");
    /* ---- and the count is kept honest by the OTHER teardown too ----------- */
    /* The projections are placed while the tier IS live too, and half two is
     * the path that frees those. A count left positive there would send every
     * later redundant shutdown through an exclusive section with nothing left
     * to free -- correct, but it stops the DEAD fast path from being one. */
    enum { NL = 1, NE = 4, D = 64, IH = 32 };
    if (!qt_init(NL, NE, D, IH, NE, 2, 0, 1)) { printf("  FAIL: tier did not start\n"); return 1; }
    check(qt_dnproj_init(LAYER, qb, sb, PI, PO, 0) == 1, "the projection uploads with the tier on");
    check(qg_trunk == 1, "one trunk tensor is owned");
    qt_shutdown();
    check(qg_get() == QT_DEAD && fake_live_tensors == 0, "the live-tier teardown freed it as well");
    check(qg_trunk == 0, "and left the ownership count at zero");
    frees_before = fake_frees;
    qt_shutdown();
    check(fake_frees == frees_before && qg_trunk == 0,
          "a redundant shutdown after it still takes the DEAD fast path");
    check(qt_dnproj_matmul(LAYER, py, px, PI, PO) == 0, "and the layer stays on the CPU");
    alive = 0;
    if (fails) { printf("test_qwen36_tier_trunk_teardown: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_trunk_teardown: ok\n");
    return 0;
}
