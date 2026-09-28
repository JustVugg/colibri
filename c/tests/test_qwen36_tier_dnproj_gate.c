/* qt_dnproj_matmul must be counted by the lifecycle gate, because qt_shutdown
 * frees G_dnp[].t (#1564, review of c33679dc, HIGH 2) -- and must keep working
 * with the expert tier never started, which is why it was left ungated.
 *
 * The defect. qt_dnproj_matmul read G_dnp[layer] and called the backend with no
 * gate at all, so a teardown saw qg_live==0, took its exclusive section and
 * freed the tensor the backend had already captured: valgrind "Invalid read ...
 * free'd by qt_shutdown". Here the backend is paused right after it captured
 * the tensor (fake_matmul_hook, like backend_cuda.cu picking up *tensor), and
 * the teardown reports which way it went, no clock involved:
 *   - qt_test_gate_block_hook on the teardown thread: it is WAITING for the
 *     matmul (right);
 *   - fake_free_hook with the captured tensor: it FREED it under the matmul
 *     (the defect, DNPROJ_TENSOR_FREED_WHILE_MATMUL_ACTIVE=1).
 * Also checked: the projections work with the tier in QT_DEAD (never started),
 * a matmul arriving during TEARING_DOWN is refused to the CPU path instead of
 * reaching storage, and after the teardown the layer is back on the CPU. */
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
enum { D = 64, IH = 32, NEXP = 4, PI = 16, PO = 8 };
static int8_t pq[PI * PO];
static float psc[PO], px[PI], py[PO];
static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
static int mm_parked, mm_release, mm_armed, mm_result = -1;
static ColiCudaTensor *mm_tensor;
static int freed_while_active, td_queued, td_returned;
static pthread_t td_tid;
static void matmul_hook(ColiCudaTensor *t) {
    pthread_mutex_lock(&mx);
    if (mm_armed) {
        mm_armed = 0; mm_tensor = t; mm_parked = 1; pthread_cond_broadcast(&cv);
        while (!mm_release) pthread_cond_wait(&cv, &mx);
        mm_tensor = NULL;
    }
    pthread_mutex_unlock(&mx);
}
static void free_hook(ColiCudaTensor *t) {
    pthread_mutex_lock(&mx);
    if (mm_tensor && t == mm_tensor) { freed_while_active = 1; pthread_cond_broadcast(&cv); }
    pthread_mutex_unlock(&mx);
}
static void gate_block_hook(void) {
    pthread_mutex_lock(&mx);
    if (pthread_equal(pthread_self(), td_tid)) { td_queued = 1; pthread_cond_broadcast(&cv); }
    pthread_mutex_unlock(&mx);
}
static void *matmuller(void *a) {
    (void)a;
    int r = qt_dnproj_matmul(0, py, px, PI, PO);
    pthread_mutex_lock(&mx); mm_result = r; pthread_cond_broadcast(&cv); pthread_mutex_unlock(&mx);
    return NULL;
}
static void *tearer(void *a) {
    (void)a; qt_shutdown();
    pthread_mutex_lock(&mx); td_returned = 1; pthread_cond_broadcast(&cv); pthread_mutex_unlock(&mx);
    return NULL;
}
static volatile int alive = 1;
static void *watchdog(void *arg) {
    (void)arg;
    struct timespec limit = { 60, 0 };
    nanosleep(&limit, NULL);
    if (alive) { (void)!write(2, "FAIL: dnproj gate test hung\n", 28); _exit(2); }
    return NULL;
}
#define WAIT_UNTIL(cond) do { pthread_mutex_lock(&mx); while (!(cond)) pthread_cond_wait(&cv, &mx); pthread_mutex_unlock(&mx); } while (0)
int main(void) {
    printf("qwen36 tier dnproj gate: no projection matmul on teardown-freed storage\n");
    for (int i = 0; i < PI * PO; i++) pq[i] = (int8_t)(i % 7);
    for (int i = 0; i < PO; i++) psc[i] = 1.0f;
    pthread_t wd;
    if (pthread_create(&wd, NULL, watchdog, NULL) == 0) pthread_detach(wd);
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("CUDA_EXPERT_GB", "auto", 1);
    /* ---- tier off: the projections stand on their own ------------------- */
    check(qg_get() == QT_DEAD, "the expert tier has never been started");
    check(qt_dnproj_init(1, pq, psc, PI, PO, 0) == 1, "dnproj uploads with the tier off");
    int m0 = fake_matmuls;
    check(qt_dnproj_matmul(1, py, px, PI, PO) == 1 && fake_matmuls == m0 + 1,
          "dnproj runs on the GPU with the tier off");
    check(qt_dnproj_matmul(2, py, px, PI, PO) == 0, "an unplaced layer stays on the CPU");
    /* ---- tier on, and a matmul inside the backend when the teardown comes */
    if (!qt_init(1, NEXP, D, IH, NEXP, 2, 0, 1)) { printf("  FAIL: tier did not start\n"); return 1; }
    check(qt_dnproj_init(0, pq, psc, PI, PO, 0) == 1, "dnproj uploads with the tier on");
    fake_matmul_hook = matmul_hook;
    fake_free_hook = free_hook;
    qt_test_gate_block_hook = gate_block_hook;
    mm_armed = 1;
    pthread_t mt;
    pthread_create(&mt, NULL, matmuller, NULL);
    WAIT_UNTIL(mm_parked);
    check(mm_tensor != NULL, "the backend captured the projection tensor");
    pthread_mutex_lock(&mx);
    pthread_create(&td_tid, NULL, tearer, NULL);
    pthread_mutex_unlock(&mx);
    WAIT_UNTIL(td_queued || freed_while_active || td_returned);
    printf("DNPROJ_TENSOR_FREED_WHILE_MATMUL_ACTIVE=%d\n", freed_while_active);
    check(!freed_while_active, "qt_shutdown freed G_dnp[].t while a matmul was using it");
    if (td_queued) {
        check(qg_get() == QT_TEARING_DOWN, "the teardown waits in TEARING_DOWN");
        int before = fake_matmuls;
        check(qt_dnproj_matmul(0, py, px, PI, PO) == 0 && fake_matmuls == before,
              "a matmul arriving during the teardown is refused to the CPU path");
    }
    pthread_mutex_lock(&mx);
    mm_release = 1; pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mx);
    pthread_join(mt, NULL); pthread_join(td_tid, NULL);
    check(mm_result == 1, "the admitted matmul completed on the GPU");
    check(qg_get() == QT_DEAD && fake_live_tensors == 0, "the teardown finished and freed everything");
    check(qt_dnproj_matmul(0, py, px, PI, PO) == 0, "after the teardown the layer is back on the CPU");
    alive = 0;
    if (fails) { printf("test_qwen36_tier_dnproj_gate: %d failure(s)\n", fails); return 1; }
    printf("test_qwen36_tier_dnproj_gate: ok\n");
    return 0;
}
