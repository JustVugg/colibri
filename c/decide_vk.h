/* decide_vk.h -- the decision engines' encoders on a Vulkan device (COLI_VULKAN=1):
 * laya.c (ModernBERT and its decision head) and gliner_decide.c (DeBERTa-v3).
 * docs/vulkan.md, "The decision engines".
 *
 * A decision is one forward over a prompt-sized batch of rows and no generation, so it
 * is matrix products end to end. With COLI_VULKAN=1 the device opens once the weights
 * are loaded, and a forward runs one of two ways:
 *   - on the device (COLI_VK_CHAIN, dvk_decide_chain below): the engine records the
 *     whole encoder (and Laya's head layers) as frames of the dense chain (vk_chain.h:
 *     vkc_matmul and the chain_enc.comp ops), the activations staying on the device from
 *     the embedding rows to the rows the scorer reads; one submission per forward;
 *   - matrix by matrix (COLI_VK_CHAIN=0): every dense matrix through coli_vk_matmul (the
 *     tiled GEMM from the backend's threshold), the norms, the attention and the rest on
 *     the CPU (COLI_VK_DENSE=0 keeps the matrices there too).
 * Both read the same resident copy of a matrix, uploaded during setup as f32 (fmt 10)
 * or exact f16 values (fmt 14), through the backend's staged path where
 * the card needs it. Activations stay f32 on both, so a forward differs from the CPU's
 * only in the order of the sums: the decisions are the CPU's, the probabilities within
 * rounding (the tests hold them to the CPU's; tests/vulkan_engines.sh decide).
 * A device that fails mid-forward (lost) leaves the forward to the CPU, which recomputes
 * it from the ids: a decision has no state to recover.
 *
 * Without COLI_VULKAN in the build, or unset at run time, none of this exists or runs. */
#ifndef COLI_DECIDE_VK_H
#define COLI_DECIDE_VK_H
#ifdef COLI_VULKAN
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend_vulkan.h"
#include "vk_chain.h"

typedef struct { const void *w; ColiVkTensor *t; int off, fmt; } DvkSlot;
#define DVK_SLOTS 2048
static struct {
    int ready;              /* the device is open */
    int dense;              /* matrices one by one through coli_vk_matmul (COLI_VK_CHAIN=0) */
    int chain;              /* the whole forward on the device */
    int f16;                /* the checkpoint stores f16: matrices go up as f16 where every value is one */
    const char *engine;
    unsigned n_f16, n_f32;
    DvkSlot slot[DVK_SLOTS];
    unsigned long long fwd_dev, fwd_cpu, rows_dev;
    double dev_ms;
} g_dvk;

static double dvk_now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* COLI_VK_CHAIN set: 0 matrix by matrix, any other number the forward on the device.
 * Unset: on a GPU, discrete or integrated (measured on a Radeon 780M: docs/vulkan.md);
 * off on a CPU device (Lavapipe), where the device is the CPU. */
static int dvk_decide_chain(const char *engine)
{
    const char *e = getenv("COLI_VK_CHAIN");
    char why[160];
    int on;
    if (e && *e) { on = atoi(e) != 0; snprintf(why, sizeof why, "COLI_VK_CHAIN=%s", e); }
    else if (coli_vk_device_shares_ram() && !coli_vk_device_integrated()) {
        on = 0; snprintf(why, sizeof why, "a CPU device; COLI_VK_CHAIN=1 turns it on");
    } else {
        on = 1;
        snprintf(why, sizeof why, "%s; COLI_VK_CHAIN=0 matrix by matrix",
                 coli_vk_device_integrated() ? "an integrated GPU: measured faster" : "a discrete GPU");
    }
    fprintf(stderr, "[VK] %s: forward on the device %s (%s)\n", engine, on ? "on" : "off", why);
    return on;
}

static void dvk_init(const char *engine)
{
    memset(&g_dvk, 0, sizeof g_dvk);
    g_dvk.engine = engine;
    if (!coli_vk_init_env(engine)) return;   /* COLI_VULKAN unset, or no device: the CPU */
    g_dvk.ready = 1;
    atexit(coli_vk_shutdown);
    g_dvk.chain = dvk_decide_chain(engine);
    if (g_dvk.chain && !(vkc_init() && vkc_enc_ready())) {
        fprintf(stderr, "[VK] %s: the chain's shaders are missing (chain_enc.spv): matrix by matrix\n", engine);
        g_dvk.chain = 0;
    }
    if (g_dvk.chain) atexit(vkc_shutdown);   /* registered after the backend's: runs before it */
    g_dvk.dense = !g_dvk.chain && coli_vk_dense();
}

/* f32 -> f16 bits when the value is exactly an f16 (normal, subnormal or zero), else -1 */
static int dvk_f16_exact(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u, mant = x & 0x7fffffu;
    int e = (int)((x >> 23) & 0xff) - 127;
    if ((x & 0x7fffffffu) == 0) return (int)sign;
    if (e > 15 || e < -24 || ((x >> 23) & 0xff) == 0xff) return -1;
    if (e >= -14) {                                   /* normal: 10 mantissa bits */
        if (mant & 0x1fffu) return -1;
        return (int)(sign | (uint32_t)((e + 15) << 10) | (mant >> 13));
    }
    int shift = -14 - e;                              /* subnormal: 1.m >> shift, 10 bits */
    uint32_t m = mant | 0x800000u;
    if (m & ((1u << (13 + shift)) - 1)) return -1;
    return (int)(sign | (m >> (13 + shift)));
}

/* The device copy of W, uploaded on first use: f16 (fmt 14) when the engine says its
 * checkpoint is f16 and every value of W is exactly one (the same values, half the
 * bytes the device reads), else the f32 rows (fmt 10). NULL when it cannot be (that
 * matrix then stays on the CPU). */
static ColiVkTensor *dvk_tensor(const QiMat *W)
{
    if (!g_dvk.ready || W->fmt != QI_F32 || W->ld || !W->w) return NULL;
    uintptr_t k = (uintptr_t)W->w;
    unsigned h = (unsigned)((k >> 4) * 2654435761u) % DVK_SLOTS;
    for (int n = 0; n < DVK_SLOTS; n++, h = (h + 1) % DVK_SLOTS) {
        DvkSlot *s = &g_dvk.slot[h];
        if (s->w == W->w) return s->off ? NULL : s->t;
        if (s->w) continue;
        s->w = W->w;
        const float *f = (const float *)W->w;
        size_t n_ = (size_t)W->N * W->K;
        uint16_t *hb = g_dvk.f16 && W->K % 2 == 0 ? (uint16_t *)malloc(n_ * sizeof(uint16_t)) : NULL;
        int exact = hb != NULL;
        for (size_t i = 0; exact && i < n_; i++) {
            int b = dvk_f16_exact(f[i]);
            if (b < 0) exact = 0; else hb[i] = (uint16_t)b;
        }
        s->fmt = exact ? 14 : 10;
        int ok = coli_vk_tensor_ensure(&s->t, exact ? (const void *)hb : W->w, NULL, s->fmt, W->K, W->N, 0);
        free(hb);
        if (!ok) { s->off = 1; return NULL; }
        if (exact) g_dvk.n_f16++; else g_dvk.n_f32++;
        return s->t;
    }
    return NULL;
}

/* Y = X W^T + bias through the device, matrix by matrix; 0 = the caller runs it. */
static int dvk_gemm(float *Y, const float *X, int M, const QiMat *W, const float *bias)
{
    if (!g_dvk.dense || M < 1 || M > 65535) return 0;
#ifdef _OPENMP
    if (omp_in_parallel()) return 0;
#endif
    ColiVkTensor *t = dvk_tensor(W);
    ColiVkTensorInfo ti;
    if (!t || !coli_vk_tensor_info(t, &ti) || !coli_vk_matmul(&t, Y, X, W->w, NULL, ti.fmt, M, W->K, W->N, 0)) return 0;
    if (bias) {
        const int N = W->N;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < M; r++)
            for (int j = 0; j < N; j++) Y[(int64_t)r * N + j] += bias[j];
    }
    return 1;
}

/* The norms', biases' and tables' floats, packed for one upload. */
typedef struct { float *v; size_t n, cap; int failed; } DvkPrm;
static int dvk_prm_add(DvkPrm *P, const float *v, size_t n)
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
static VkcBuf *dvk_prm_upload(DvkPrm *P)
{
    if (P->failed || !P->n) return NULL;
    VkcBuf *b = vkc_buf(P->n * sizeof(float), VKC_DEV);
    if (!b) return NULL;
    if (!(vkc_begin() && vkc_write(b, 0, P->v, P->n * sizeof(float)) && vkc_submit(1))) { vkc_free(b); return NULL; }
    return b;
}

/* At the end of a request: the line the tests read (N > 0 shows the device really ran). */
static void dvk_report(void)
{
    if (!g_dvk.ready) return;
    VkcStats st;
    memset(&st, 0, sizeof st);
    if (g_dvk.chain) vkc_stats(&st);
    size_t bytes = 0, tensors = 0;
    coli_vk_mem_info(&bytes, &tensors);
    fprintf(stderr, "[VK] %s: %llu matmuls on the GPU (%zu matrices resident, %.1f MiB)\n", g_dvk.engine,
            coli_vk_matmul_calls() + st.matmuls, tensors, bytes / 1048576.0);
    if (g_dvk.chain)
        fprintf(stderr, "[VK] %s chain: %llu forwards on the device (%llu rows, %.1f ms), %llu on the CPU, "
                "%llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), %.1f ms waiting, %.1f MiB of chain buffers\n",
                g_dvk.engine, g_dvk.fwd_dev, g_dvk.rows_dev, g_dvk.dev_ms, g_dvk.fwd_cpu,
                st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, st.dev_bytes / 1048576.0);
    if (g_dvk.chain) vkc_prof_print();
}
#endif
#endif
