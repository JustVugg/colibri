/* decide_cuda.h -- the decision engines' encoders on a CUDA device (COLI_CUDA=1):
 * laya.c (ModernBERT and its decision head) and gliner_decide.c (DeBERTa-v3).
 * decide_vk.h's twin, docs/cuda.md, "The decision engines".
 *
 * With COLI_CUDA=1 the device opens once the weights are loaded, and a forward runs on
 * it as frames of the CUDA chain (cuda_chain.h: cc_matmul over the matrices' resident
 * copies, and the encoder ops of chain_enc.comp, cc_enc_*): the embedding rows go up,
 * the activations stay on the device to the rows the scorer reads, one submission a
 * forward. The matrices go up once as f32 (fmt 0), the checkpoint's values; activations
 * are f32, so a forward differs from the CPU's only in the order of the sums: the
 * decisions are the CPU's, the probabilities within rounding (tests/cuda_decide_compare.py
 * holds them to the CPU's).
 *
 * The device is COLI_GPUS' first (or COLI_GPU's; device 0 without either). COLI_CUDA_CHAIN=0
 * keeps the forward on the CPU. A device lost mid-forward (COLI_CUDA_CHAIN_FAULT=n fakes it
 * at the n-th frame) leaves that forward and every later one to the CPU, which recomputes it
 * from the ids: a decision has no state to recover.
 *
 * Without COLI_CUDA in the build, or unset at run time, none of this exists or runs. With
 * COLI_VULKAN=1 as well, the Vulkan device takes the forward (decide_vk.h) and this one
 * does not open. */
#ifndef COLI_DECIDE_CUDA_H
#define COLI_DECIDE_CUDA_H
#ifdef COLI_CUDA
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "backend_cuda.h"
#include "cuda_chain.h"

typedef struct { const void *w; ColiCudaTensor *t; int off; } DccSlot;
#define DCC_SLOTS 2048
static struct {
    int ready;              /* the device and its chain are open */
    int device;
    const char *engine;
    unsigned n_mat;
    size_t mat_bytes;
    DccSlot slot[DCC_SLOTS];
    unsigned long long fwd_dev, fwd_cpu, rows_dev, frames, fault_at;
    double dev_ms;
} g_dcc;

static double dcc_now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void dcc_shutdown(void)
{
    if (!g_dcc.ready) return;
    for (int i = 0; i < DCC_SLOTS; i++)
        if (g_dcc.slot[i].t) { coli_cuda_tensor_free(g_dcc.slot[i].t); g_dcc.slot[i].t = NULL; }
    cc_shutdown();
    coli_cuda_shutdown();
    g_dcc.ready = 0;
}

/* COLI_CUDA=1: the device and the chain's context, or a line saying why the CPU runs it */
static void dcc_init(const char *engine)
{
    memset(&g_dcc, 0, sizeof g_dcc);
    g_dcc.engine = engine;
    const char *e = getenv("COLI_CUDA");
    if (!e || !*e || !strcmp(e, "0")) return;
    const char *vk = getenv("COLI_VULKAN");
    if (vk && *vk && strcmp(vk, "0")) {
        fprintf(stderr, "[CUDA] %s: COLI_VULKAN=1 takes the forward; CUDA stays closed\n", engine);
        return;
    }
    const char *ch = getenv("COLI_CUDA_CHAIN");
    if (ch && *ch && atoi(ch) == 0) {
        fprintf(stderr, "[CUDA] %s: COLI_CUDA_CHAIN=0: the forward on the CPU\n", engine);
        return;
    }
    const char *gl = getenv("COLI_GPUS");
    if (!gl || !*gl) gl = getenv("COLI_GPU");
    int dev = gl && *gl ? atoi(gl) : 0;
    if (!cc_available() || !cc_enc_ready()) {
        fprintf(stderr, "[CUDA] %s: the CUDA backend has no encoder ops (an older build): the forward on the CPU\n", engine);
        return;
    }
    if (!coli_cuda_init(&dev, 1) || !cc_init(dev)) {
        fprintf(stderr, "[CUDA] %s: device %d did not come up: the forward on the CPU\n", engine, dev);
        return;
    }
    const char *f = getenv("COLI_CUDA_CHAIN_FAULT");
    g_dcc.fault_at = f && *f ? strtoull(f, NULL, 10) : 0;
    g_dcc.device = dev;
    g_dcc.ready = 1;
    atexit(dcc_shutdown);
    fprintf(stderr, "[CUDA] %s: forward on device %d (COLI_CUDA_CHAIN=0: the CPU)\n", engine, dev);
}

/* The device copy of W (f32 rows, uploaded on first use); NULL when it cannot be */
static ColiCudaTensor *dcc_tensor(const QiMat *W)
{
    if (!g_dcc.ready || W->fmt != QI_F32 || W->ld || !W->w) return NULL;
    uintptr_t k = (uintptr_t)W->w;
    unsigned h = (unsigned)((k >> 4) * 2654435761u) % DCC_SLOTS;
    for (int n = 0; n < DCC_SLOTS; n++, h = (h + 1) % DCC_SLOTS) {
        DccSlot *s = &g_dcc.slot[h];
        if (s->w == W->w) return s->off ? NULL : s->t;
        if (s->w) continue;
        s->w = W->w;
        if (!coli_cuda_tensor_upload(&s->t, W->w, NULL, 0, W->K, W->N, g_dcc.device)) { s->t = NULL; s->off = 1; return NULL; }
        g_dcc.n_mat++;
        g_dcc.mat_bytes += (size_t)W->N * W->K * sizeof(float);
        return s->t;
    }
    return NULL;
}

/* The norms', biases' and tables' floats, packed for one upload. */
typedef struct { float *v; size_t n, cap; int failed; } DccPrm;
static int dcc_prm_add(DccPrm *P, const float *v, size_t n)
{
    if (!v) return -1;
    if (P->n + n > P->cap) {
        size_t cap = P->cap ? P->cap : 4096;
        while (cap < P->n + n) cap *= 2;
        float *g = (float *)realloc(P->v, cap * sizeof(float));
        if (!g) { P->failed = 1; return -1; }
        P->v = g; P->cap = cap;
    }
    if (P->n + n > (size_t)INT32_MAX) { P->failed = 1; return -1; }
    memcpy(P->v + P->n, v, n * sizeof(float));
    P->n += n;
    return (int)(P->n - n);
}
/* a device buffer holding P's floats (one upload, in a frame of its own) */
static CcBuf *dcc_prm_upload(DccPrm *P)
{
    if (P->failed || !P->n) return NULL;
    CcBuf *b = cc_buf(P->n * sizeof(float), CC_DEV);
    if (!b) return NULL;
    if (!(cc_begin() && cc_write(b, 0, P->v, P->n * sizeof(float)) && cc_submit(1))) { cc_free(b); return NULL; }
    return b;
}

/* a forward's frame; COLI_CUDA_CHAIN_FAULT=n fails the n-th */
static int dcc_begin(void)
{
    g_dcc.frames++;
    if (g_dcc.fault_at && g_dcc.frames == g_dcc.fault_at) {
        fprintf(stderr, "[CUDA] %s: frame %llu failed (COLI_CUDA_CHAIN_FAULT)\n", g_dcc.engine, g_dcc.frames);
        return 0;
    }
    return cc_begin();
}
/* a forward that did not finish: this one and every later one on the CPU */
static void dcc_lost(void)
{
    fprintf(stderr, "[CUDA] %s: the device was lost: the forward runs on the CPU\n", g_dcc.engine);
    if (!cc_lost()) cc_finish();
    g_dcc.ready = 0;
}

/* At the end of a request: the line the tests read (N > 0 shows the device really ran). */
static void dcc_report(void)
{
    if (!g_dcc.n_mat && !g_dcc.fwd_dev) return;
    CcStats st;
    memset(&st, 0, sizeof st);
    if (g_dcc.ready) cc_stats(&st);
    fprintf(stderr, "[CUDA] %s chain: %llu forwards on the device (%llu rows, %.1f ms), %llu on the CPU, "
            "%u matrices resident (%.1f MiB), %llu frames (%llu ops, %llu matmuls), %.1f ms waiting\n",
            g_dcc.engine, g_dcc.fwd_dev, g_dcc.rows_dev, g_dcc.dev_ms, g_dcc.fwd_cpu, g_dcc.n_mat,
            g_dcc.mat_bytes / 1048576.0, st.frames, st.ops, st.matmuls, st.wait_ms);
}
#endif
#endif
