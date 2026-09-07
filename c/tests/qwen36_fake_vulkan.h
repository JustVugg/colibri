/* qwen36_fake_vulkan.h -- fake Vulkan backend for the qwen36 tier tests.
 *
 * The Vulkan twin of qwen36_fake_cuda.h: defines every coli_vk_* symbol the
 * tier's Vulkan shim (qwen36_tier.c, -DCOLI_VULKAN) links against, with the
 * signatures from backend_vulkan.h, and RECORDS what it receives. A test that
 * includes this and then ../qwen36_tier.c runs the shim's control flow with no
 * device, no libvulkan and no shaders -- in `make check` and under the
 * sanitizers on every platform -- while the numerics keep their own gate in
 * test_qwen36_tier_vk (VK=1, real device).
 *
 * Settable knobs:
 *   fake_vk_available     - what coli_vk_init returns and coli_vk_available
 *                           reports afterwards (default 1).
 *   fake_vk_budget_known  - whether coli_vk_mem_budget succeeds (default 1);
 *                           0 reproduces a driver without VK_EXT_memory_budget.
 *   fake_vk_budget_gb /
 *   fake_vk_used_gb       - the figures coli_vk_mem_budget reports.
 *   fake_vk_issue_hook    - called by coli_vk_expert_group_issue with the row
 *                           count and the input pointer; its return value is
 *                           what issue returns. NULL (the default) returns 1.
 *   fake_vk_take_ok       - coli_vk_expert_group_take returns this (default 1);
 *                           when 1 it fills row j of y with the value j+1 over
 *                           fake_vk_take_D floats, which the test sets to the
 *                           tier's D, so qt_take's accumulation is checkable.
 * Recorded:
 *   fake_vk_inits, fake_vk_shutdowns, fake_vk_uploads, fake_vk_frees,
 *   fake_vk_last_fmt, fake_vk_last_gs, fake_vk_last_bytes,
 *   fake_vk_last_issue_count. */
#ifndef QWEN36_FAKE_VULKAN_H
#define QWEN36_FAKE_VULKAN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* MinGW has no setenv: the tier reads its configuration from the environment
 * and the test must be able to set it on Windows too. */
#if defined(_WIN32)
static int test_setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
#define setenv test_setenv
#endif
#include "../backend_vulkan.h"

struct ColiVkTensor { int fmt, I, O, gs; const void *w; };

static int fake_vk_available = 1;
static int fake_vk_budget_known = 1;
static double fake_vk_budget_gb = 2.0, fake_vk_used_gb = 0.0;
static int (*fake_vk_issue_hook)(int count, const float *x) = NULL;
static int fake_vk_take_ok = 1;
static int fake_vk_take_D = 0;

static int fake_vk_inits, fake_vk_shutdowns, fake_vk_uploads, fake_vk_frees;
static int fake_vk_last_fmt = -1, fake_vk_last_gs = -1;
static size_t fake_vk_last_bytes;
static size_t fake_vk_live_tensors, fake_vk_live_bytes;
static int fake_vk_last_issue_count;

int coli_vk_init(const char *spv_path) { (void)spv_path; fake_vk_inits++; return fake_vk_available; }
void coli_vk_shutdown(void) { fake_vk_shutdowns++; }
int coli_vk_available(void) { return fake_vk_available; }
const char *coli_vk_default_spv(char *buf, size_t n) { snprintf(buf, n, "fake.spv"); return buf; }
void coli_vk_mem_info(size_t *used_bytes, size_t *tensor_count) {
    if (used_bytes) *used_bytes = fake_vk_live_bytes;
    if (tensor_count) *tensor_count = fake_vk_live_tensors;
}
int coli_vk_mem_budget(double *used_gb, double *budget_gb) {
    if (!fake_vk_budget_known) return 0;
    if (used_gb) *used_gb = fake_vk_used_gb;
    if (budget_gb) *budget_gb = fake_vk_budget_gb;
    return 1;
}
int coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales,
                          int fmt, int I, int O, int grp) {
    (void)scales;
    if (*tensor) return 1;                       /* ensure: a second call is a no-op */
    ColiVkTensor *t = (ColiVkTensor *)calloc(1, sizeof *t);
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->gs = grp; t->w = weights;
    *tensor = t;
    fake_vk_uploads++;
    fake_vk_last_fmt = fmt; fake_vk_last_gs = grp;
    fake_vk_last_bytes = (size_t)I * O / (fmt == 1 ? 1 : 2);
    fake_vk_live_tensors++; fake_vk_live_bytes += fake_vk_last_bytes;
    return 1;
}
void coli_vk_tensor_free(ColiVkTensor *t) {
    if (!t) return;
    fake_vk_frees++;
    fake_vk_live_tensors--;
    fake_vk_live_bytes -= (size_t)t->I * t->O / (t->fmt == 1 ? 1 : 2);
    free(t);
}
int coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                               ColiVkTensor *const *downs, const int *rows, int count,
                               const float *x) {
    (void)gates; (void)ups; (void)downs; (void)rows;
    fake_vk_last_issue_count = count;
    if (fake_vk_issue_hook) return fake_vk_issue_hook(count, x);
    return 1;
}
int coli_vk_expert_group_take(float *y) {
    if (!fake_vk_take_ok) return 0;
    for (int j = 0; j < fake_vk_last_issue_count; j++)
        for (int d = 0; d < fake_vk_take_D; d++) y[(size_t)j * fake_vk_take_D + d] = (float)(j + 1);
    return 1;
}

#endif /* QWEN36_FAKE_VULKAN_H */
