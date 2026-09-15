/* qwen36_fake_cuda.h -- fake CUDA backend shared by the qwen36 tier tests.
 *
 * Defines every coli_cuda_* symbol qwen36_tier.c links against (its
 * signatures come from backend_cuda.h, which the tier includes on its own)
 * and RECORDS what it receives, so a test can assert on real upload/issue
 * traffic without a GPU or the CUDA toolkit. A test that only checked
 * "qt_init returns 1" would pass even with the tier fully broken.
 *
 * Three settable hooks beyond plain recording:
 *   fake_ndev        - device count returned by coli_cuda_available_device_count
 *                       and coli_cuda_device_count (default 1).
 *   fake_issue_hook   - called by coli_cuda_expert_group_issue with the issuing
 *                       device (taken from g[0]->device), the row count and the
 *                       input pointer; its return value is what issue returns.
 *                       NULL (the default) reproduces the old always-0 stub.
 *   fake_upload_hook  - called at the start of every tensor upload, on the
 *                       uploader thread, with the tensor's fmt. A test that
 *                       needs an upload to take TIME (a real cudaMemcpy does)
 *                       sleeps here; NULL (the default) uploads instantly. */
#ifndef QWEN36_FAKE_CUDA_H
#define QWEN36_FAKE_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "../backend_cuda.h"

struct ColiCudaTensor { int fmt, I, O, device, gs; const void *w; };

static int fake_uploads, fake_frees, fake_live_tensors;
static int fake_fail_upload;
static int last_fmt = -1;
static size_t last_bytes;
static unsigned char captured[3][4096];
static size_t captured_len[3];
static float captured_scales[3][4096];
static size_t captured_scale_count[3];

static int fake_ndev = 1;
static size_t fake_free_bytes = 2ull << 30;    /* what coli_cuda_mem_info reports as free */
static int (*fake_issue_hook)(int device, int count, const float *x) = NULL;
static void (*fake_upload_hook)(int fmt) = NULL;
static void (*fake_free_hook)(ColiCudaTensor *tensor) = NULL;
static int fake_plain_issues, fake_clamped_issues, fake_takes;
static float fake_last_swiglu_limit;
static float fake_take_output[4096];
static int fake_take_returns_output;

static int upload_common(ColiCudaTensor **t, const void *w, const float *s,
                         int fmt, int I, int O, int device, int gs) {
    if (fake_upload_hook) fake_upload_hook(fmt);
    const int call = ++fake_uploads;
    if (fake_fail_upload == call) return 0;
    ColiCudaTensor *n = (ColiCudaTensor *)calloc(1, sizeof *n);
    n->fmt = fmt; n->I = I; n->O = O; n->device = device; n->gs = gs; n->w = w;
    *t = n;
    fake_live_tensors++;
    last_fmt = fmt;
    last_bytes = (size_t)I * O / ((fmt == 1 || fmt == 8) ? 1 : 2);
    if (call <= 3) {
        int i = call - 1;
        captured_len[i] = last_bytes < sizeof captured[i] ? last_bytes : sizeof captured[i];
        memcpy(captured[i], w, captured_len[i]);
        size_t ns = fmt == 8 ? (size_t)((O + 127) / 128) * ((I + 127) / 128)
                             : gs > 0 ? (size_t)O * ((I + gs - 1) / gs)
                                      : (size_t)O;
        captured_scale_count[i] = ns < sizeof captured_scales[i] / sizeof captured_scales[i][0]
                                ? ns : sizeof captured_scales[i] / sizeof captured_scales[i][0];
        memcpy(captured_scales[i], s, captured_scale_count[i] * sizeof(float));
    }
    return 1;
}
int coli_cuda_tensor_upload(ColiCudaTensor **t, const void *w, const float *s,
                            int fmt, int I, int O, int device) {
    return upload_common(t, w, s, fmt, I, O, device, 0);
}
int coli_cuda_tensor_upload_g(ColiCudaTensor **t, const void *w, const float *s,
                              int fmt, int I, int O, int device, int gs) {
    return upload_common(t, w, s, fmt, I, O, device, gs);
}
void coli_cuda_tensor_free(ColiCudaTensor *t) {
    if (t) {
        if (fake_free_hook) fake_free_hook(t);
        fake_frees++; fake_live_tensors--;
    }
    free(t);
}
int coli_cuda_available_device_count(void) { return fake_ndev; }
int coli_cuda_device_count(void) { return fake_ndev; }
int coli_cuda_init(const int *d, int n) { (void)d; (void)n; return 1; }
static int fake_lut_published;
int coli_cuda_fp8_set_lut(const float *lut) { fake_lut_published = lut != NULL; return lut != NULL; }
void coli_cuda_shutdown(void) {}
int coli_cuda_mem_info(int device, size_t *freeb, size_t *total) {
    (void)device;
    *freeb = fake_free_bytes; *total = 4ull << 30;   /* 2 GiB liberi by default */
    return 1;
}
int coli_cuda_expert_group_issue(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                 ColiCudaTensor *const *d, const int *rows,
                                 int count, const float *x) {
    (void)u; (void)d; (void)rows;
    fake_plain_issues++;
    if (fake_issue_hook) return fake_issue_hook(count > 0 ? g[0]->device : -1, count, x);
    return 0;
}
int coli_cuda_expert_group_issue_clamped(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                         ColiCudaTensor *const *d, const int *rows,
                                         int count, const float *x, float limit) {
    (void)u; (void)d; (void)rows;
    fake_clamped_issues++; fake_last_swiglu_limit = limit;
    if (fake_issue_hook) return fake_issue_hook(count > 0 ? g[0]->device : -1, count, x);
    return 0;
}
const float *coli_cuda_expert_group_take(int device) {
    (void)device; fake_takes++;
    return fake_take_returns_output ? fake_take_output : NULL;
}
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d, double *kernel, double *d2h) {
    if (calls) *calls = 0; if (experts) *experts = 0; if (rows) *rows = 0;
    if (h2d) *h2d = 0; if (kernel) *kernel = 0; if (d2h) *d2h = 0;
}
void coli_cuda_stats(int device, size_t *count, size_t *bytes) {
    (void)device; if (count) *count = 0; if (bytes) *bytes = 0;
}

/* dense GEMV on a resident tensor (lm_head / DeltaNet projections placed on a
 * device). Counted, never computed: the placement tests check WHERE work
 * went; the arithmetic has its own oracle in the CUDA build. Parameters are
 * unused on purpose (CFLAGS carry -Wno-unused-parameter). */
static int fake_matmuls;
int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x, const void *weights, const float *scales, int fmt, int S, int I, int O, int device, int gs) {
    fake_matmuls++; return 1;
}

#endif /* QWEN36_FAKE_CUDA_H */
