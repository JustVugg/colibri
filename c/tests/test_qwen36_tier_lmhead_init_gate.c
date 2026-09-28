/* qt_lmhead_init must not read G_lmh before the lifecycle admits it (#1564,
 * review of e6c70eb7, HIGH B).
 *
 * The defect. qt_lmhead_init opened with
 *
 *     if(!G_lmh.dev_ok||!q||!sc) return 0;
 *     if(!qt_gate_enter()) return 0;
 *
 * -- a read of the CURRENT generation's placement, with nothing held, one line
 * above the admission. A caller descheduled between the two crosses a whole
 * shutdown and a whole re-init, is then ADMITTED by the gate (the new
 * generation is live, which is all the gate asks) and, carrying the OLD
 * generation's dev_ok, its host weights and its host scales, installs them into
 * a generation configured with COLI_PLACE=off, which had decided this lm_head
 * belongs on the CPU. The upload SUCCEEDS, so nothing reports it: the new
 * generation answers tokens with the previous model's weights.
 *
 * This is the same defect class as qt_lmhead_matmul's pre-gate `G_lmh.on`, and
 * the difference is the whole of it. The matmul survives that read because the
 * backend refuses a NULL cached tensor and the call falls back to the CPU
 * (backend_cuda.cu). The init path has no such rejection: it CREATES the
 * tensor, so there is nothing to refuse it. One guard, two outcomes.
 *
 * The interleaving is forced through the pre-gate seam, never a clock: the
 * stale caller parks with dev_ok already read, the main thread tears the
 * generation down and starts a new one, and only then releases it. */
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
enum { NL = 1, NE = 4, D = 64, IH = 32, LI = 8, LO = 4 };
/* Generation A's weights, all positive; generation C's, all negative. Same
 * shape, so the backend's cached path would happily keep A's. */
static int8_t qa[LI * LO], qc[LI * LO];
static float sca[LO], scc[LO], px[LI], py[LO];
static void expect_c(float *y) {
    for (int o = 0; o < LO; o++) {
        float a = 0.f;
        for (int i = 0; i < LI; i++) a += px[i] * (float)qc[(size_t)o * LI + i];
        y[o] = a * scc[o];
    }
}
static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
static int parked, release, fired, stale_result = -1, stale_uploads_before, stale_ups;
static pthread_t stale_tid;
static void pre_gate_hook(void) {
    pthread_mutex_lock(&mx);
    if (fired) { pthread_mutex_unlock(&mx); return; }
    fired = 1;
    parked = 1; pthread_cond_broadcast(&cv);
    while (!release) pthread_cond_wait(&cv, &mx);
    pthread_mutex_unlock(&mx);
}
static void *stale_caller(void *a) {
    (void)a;
    /* the host buffers of the generation that is about to be torn down */
    int r = qt_lmhead_init(qa, sca, LI, LO);
    pthread_mutex_lock(&mx);
    stale_result = r; stale_ups = fake_uploads - stale_uploads_before;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    return NULL;
}
static volatile int alive = 1;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { 60, 0 };
    nanosleep(&limit, NULL);
    if (alive) { (void)!write(2, "FAIL: lmhead init gate test hung\n", 30); _exit(2); }
    return NULL;
}
#define WAIT_UNTIL(cond) do { pthread_mutex_lock(&mx); while (!(cond)) pthread_cond_wait(&cv, &mx); pthread_mutex_unlock(&mx); } while (0)
int main(void) {
    printf("qwen36 tier lmhead init: G_lmh is read only after admission\n");
    for (int i = 0; i < LI * LO; i++) { qa[i] = (int8_t)(i + 1); qc[i] = (int8_t)(-(i + 1)); }
    for (int o = 0; o < LO; o++) { sca[o] = 1.0f; scc[o] = 1.0f; }
    for (int i = 0; i < LI; i++) px[i] = 1.0f;
    pthread_t wd;
    if (pthread_create(&wd, NULL, watchdog, NULL) == 0) pthread_detach(wd);
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("CUDA_EXPERT_GB", "auto", 1);
    unsetenv("HEAT_FILE");
    fake_cached_upload = 1; fake_dense_compute = 1;
    /* ---- generation A: lm_head placed on the device ----------------------- */
    setenv("COLI_PLACE", "lmhead=0", 1);
    if (!qt_init(NL, NE, D, IH, NE, 2, 0, 1)) { printf("  FAIL: gen A did not start\n"); return 1; }
    check(G_lmh.dev_ok == 1, "gen A places the lm_head on a device");
    check(qt_lmhead_init(qa, sca, LI, LO) == 1, "gen A uploads its lm_head");
    check(G_lmh.on == 1, "gen A's lm_head is live");
    float want_a[LO];
    { float a[LO]; for (int o = 0; o < LO; o++) { float s = 0.f;
        for (int i = 0; i < LI; i++) s += px[i] * (float)qa[(size_t)o * LI + i];
        a[o] = s * sca[o]; } memcpy(want_a, a, sizeof a); }
    check(qt_lmhead_matmul(py, px, LI, LO) == 1 && memcmp(py, want_a, sizeof py) == 0,
          "gen A's lm_head computes with gen A's weights");
    /* ---- park the stale caller with dev_ok ALREADY read ------------------- */
    stale_uploads_before = fake_uploads;
    qt_test_lmhead_pre_gate_hook = pre_gate_hook;
    pthread_create(&stale_tid, NULL, stale_caller, NULL);
    WAIT_UNTIL(parked);
    printf("STALE_CALLER_PARKED=%d\n", parked);
    /* ---- teardown gen A, and start gen B with the lm_head on the CPU ------ */
    qt_shutdown();
    check(qg_get() == QT_DEAD && G_lmh.t == NULL && G_lmh.on == 0,
          "gen A is gone, lm_head tensor and all");
    setenv("COLI_PLACE", "off", 1);
    if (!qt_init(NL, NE, D, IH, NE, 2, 0, 1)) { printf("  FAIL: gen B did not start\n"); return 1; }
    printf("NEW_GENERATION_DEV_OK=%d ON=%d\n", G_lmh.dev_ok, G_lmh.on);
    check(G_lmh.dev_ok == 0 && G_lmh.on == 0, "gen B keeps the lm_head on the CPU (COLI_PLACE=off)");
    /* ---- release the caller that read dev_ok in gen A -------------------- */
    pthread_mutex_lock(&mx);
    qt_test_lmhead_pre_gate_hook = NULL;
    release = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    pthread_join(stale_tid, NULL);
    printf("STALE_INIT_RESULT=%d STALE_UPLOADS=%d NEW_GENERATION_DEV_OK=%d ON=%d\n",
           stale_result, stale_ups, G_lmh.dev_ok, G_lmh.on);
    check(stale_result == 0, "the caller that read dev_ok in gen A was refused by gen B");
    check(stale_ups == 0, "it uploaded nothing into gen B");
    check(G_lmh.dev_ok == 0 && G_lmh.on == 0 && G_lmh.t == NULL,
          "gen B is exactly as configured: no lm_head tensor, no device placement");
    int mm = fake_matmuls, mmres = qt_lmhead_matmul(py, px, LI, LO);
    printf("POST_STALE_MATMUL=%d (1=backend was called)\n", mmres ? fake_matmuls - mm : 0);
    check(mmres == 0, "gen B still answers the lm_head from the CPU");
    /* ---- positive control: a placed generation still installs ------------ */
    qt_shutdown();
    setenv("COLI_PLACE", "lmhead=0", 1);
    if (!qt_init(NL, NE, D, IH, NE, 2, 0, 1)) { printf("  FAIL: gen C did not start\n"); return 1; }
    check(G_lmh.dev_ok == 1, "gen C places the lm_head on a device");
    check(qt_lmhead_init(qc, scc, LI, LO) == 1, "gen C uploads its own lm_head");
    check(G_lmh.on == 1, "gen C's lm_head is live");
    memset(py, 0, sizeof py);
    float want_c[LO];
    expect_c(want_c);
    check(qt_lmhead_matmul(py, px, LI, LO) == 1 && memcmp(py, want_c, sizeof py) == 0,
          "gen C's lm_head computes with gen C's weights, not gen A's");
    qt_shutdown();
    check(G_lmh.t == NULL && G_lmh.on == 0 && fake_live_tensors == 0,
          "the last teardown freed the lm_head tensor");
    alive = 0;
    if (fails) { printf("test_qwen36_tier_lmhead_init_gate: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_lmhead_init_gate: ok\n");
    return 0;
}
