/* cu_dense.h -- an engine's dense matrices on the CUDA device, a matrix at a time.
 *
 * The routed-expert tier on a CUDA device (cu_tier.h) leaves an engine's dense part on the
 * CPU, which reads all of it every token: on DeepSeek V4 Flash UD-IQ1_S that was half of a
 * decode step. An engine hands each dense matmul to cud_mm first. The matrix goes to the
 * device the first time it is used, in coli_cuda_matmul's formats (0 f32, 1 int8 rows with a
 * scale each, 4 int4 in groups, 100 + t a GGUF's blocks in ggml type t, a matrix the CPU holds
 * 16 rows interleaved going up as the file's rows), and runs there from then on: the GEMV for a decode step's rows, the tensor cores'
 * tiles for a prompt chunk's. Its handle lives in the engine's own matrix struct. A matrix the
 * device does not take, or one past the budget, stays the CPU's for good. Matrices go up in
 * the order they are first used, the layers' order, so a budget below the whole dense part
 * holds its first layers.
 *
 * The budget (cud_open): the whole dense part where it fits in half of what the device has
 * free less the tier's reserve (COLI_VK_TIER_RESERVE_GB), else none: the routed experts keep
 * the card. Measured on an RTX PRO 4000 (24 GB): DeepSeek V4 Flash UD-IQ1_S's 6.2 GiB went up
 * whole, decode 4.1 -> 7.2-7.7 tok/s; Kimi K3 UD-IQ1_S's first 16.7 of 41.6 GiB took the
 * tier from 3463 experts to 865 (the card's share of the routed experts 36% -> 16.5%, more of
 * them from the disk), and a request had not finished in the time it took before. The engine
 * passes the budget to the tier as its dense bytes (VktConfig.dense_bytes): the tier's budget
 * leaves the room. COLI_CUDA_DENSE=0 keeps the dense part on the CPU, COLI_CUDA_DENSE_GB sets
 * the budget (a part of it too: its first layers).
 *
 * Calls come from the engine thread (the one that called cud_open) outside OpenMP regions;
 * any other call returns 0 and the CPU computes. Include it once, in a COLI_CUDA build. */
#ifndef CU_DENSE_H
#define CU_DENSE_H
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "backend_cuda.h"
#include "ggml_quants.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int on, dev, mats, left;
    size_t budget, used;
    unsigned long long calls, rows;
    pthread_t thread;
} CuDense;
static CuDense g_cud;

/* the bytes a matrix takes on the device, 0 for a format it does not take */
static size_t cud_bytes(int fmt, int O, int I, int gs) {
    if (O < 1 || I < 1) return 0;
    if (fmt == 0) return (size_t)O * I * 4;
    if (fmt == 1) return (size_t)O * I + (size_t)O * 4;
    if (fmt == 4) return gs > 0 ? (size_t)O * ((I + 1) / 2) + (size_t)O * ((I + gs - 1) / gs) * 4 : 0;
    if (fmt >= 100) {
        if (gq_r16_base(fmt - 100) >= 0) fmt = 100 + gq_r16_base(fmt - 100);   /* the file's rows */
        const int blk = coli_cuda_block_elems(fmt);
        const int64_t rb = gq_row_bytes(fmt - 100, I);
        return blk > 0 && I % 32 == 0 && I % blk == 0 && rb > 0 ? (size_t)O * (size_t)((rb + 3) / 4 * 4) : 0;
    }
    return 0;
}

/* The dense part on device dev, `dense` bytes of it as the engine counts them (its matrices'
 * cud_bytes). Returns the budget, the bytes the engine gives the tier as its dense part; 0
 * when it stays on the CPU. */
static size_t cud_open(const char *engine, int dev, size_t dense) {
    const char *e = getenv("COLI_CUDA_DENSE"), *g = getenv("COLI_CUDA_DENSE_GB");
    const double GiB = 1073741824.0;
    g_cud = (CuDense){0};
    if ((e && *e && atoi(e) == 0) || !dense) return 0;
    size_t free_b = 0, total = 0;
    if (!coli_cuda_mem_info(dev, &free_b, &total)) return 0;
    const char *r = getenv("COLI_VK_TIER_RESERVE_GB");
    const double reserve = (r && *r ? atof(r) : 1.0) * GiB;
    const double room = (double)free_b > reserve ? ((double)free_b - reserve) * 0.5 : 0;
    double budget = g && *g ? atof(g) * GiB : (double)dense <= room ? (double)dense : 0;
    if (budget > (double)dense) budget = (double)dense;
    if (budget < 1) {
        fprintf(stderr, "[CUDA] %s: the dense matrices (%.2f GiB) stay on the CPU: more than half of the device's free "
                        "memory, which the routed experts keep (COLI_CUDA_DENSE_GB sets a budget)\n", engine, dense / GiB);
        return 0;
    }
    g_cud.on = 1;
    g_cud.dev = dev;
    g_cud.budget = (size_t)budget;
    g_cud.thread = pthread_self();
    fprintf(stderr, "[CUDA] %s: the dense matrices to the device as they are first used, %.2f of %.2f GiB "
                    "(COLI_CUDA_DENSE=0 keeps them on the CPU, COLI_CUDA_DENSE_GB sets it)\n",
            engine, g_cud.budget / GiB, dense / GiB);
    return g_cud.budget;
}

/* y[S][O] = x[S][I] W^T on the device: 1 done there, 0 the CPU's. *h is the matrix's
 * handle (NULL before its first use), *no set once it stays the CPU's. */
static int cud_mm(void **h, int *no, float *y, const float *x, int S, int fmt, const void *w, const float *s,
                  int O, int I, int gs) {
    if (!g_cud.on || *no || S < 1 || !w || !pthread_equal(pthread_self(), g_cud.thread)) return 0;
#ifdef _OPENMP
    if (omp_in_parallel()) return 0;
#endif
    const int held = fmt >= 100 && gq_r16_base(fmt - 100) >= 0 ? fmt : 0;   /* 16 rows interleaved */
    if (held) fmt = 100 + gq_r16_base(fmt - 100);
    if (!*h) {
        const size_t b = cud_bytes(fmt, O, I, gs);
        const int room = b && g_cud.used + b <= g_cud.budget;
        uint8_t *rows = NULL;
        if (room && held && O % 16 == 0 && (rows = (uint8_t *)malloc((size_t)gq_row_bytes(fmt - 100, I) * O)))
            gq_r16_unpack(held - 100, rows, (const uint8_t *)w, O, I);
        const int up = room && (!held || rows) &&
                       coli_cuda_matmul((ColiCudaTensor **)h, y, x, rows ? (const void *)rows : w, s, fmt, S, I, O,
                                        g_cud.dev, fmt == 4 ? gs : 0);
        free(rows);
        if (!up) {
            if (*h) coli_cuda_tensor_free((ColiCudaTensor *)*h);
            *h = NULL;
            *no = 1;
            g_cud.left++;
            return 0;
        }
        g_cud.used += b;
        g_cud.mats++;
    } else if (!coli_cuda_matmul((ColiCudaTensor **)h, y, x, w, s, fmt, S, I, O, g_cud.dev, fmt == 4 ? gs : 0)) {
        return 0;
    }
    g_cud.calls++;
    g_cud.rows += (unsigned long long)S;
    return 1;
}

/* Matrices found by their host address, for an engine that makes its views on the fly (a
 * group's rows of a larger matrix, a view built for each call): an entry keeps the handle and
 * the flag cud_mm keeps. Only for memory the engine keeps until it closes; a buffer that holds
 * one matrix and then another would find the first one's copy. */
typedef struct { const void *data; int fmt, O, I, no; void *h; } CudEntry;
static CudEntry *g_cud_map;
static size_t g_cud_cap, g_cud_used;
static size_t cud_hash(const void *data, size_t cap) {
    uint64_t h = (uint64_t)(uintptr_t)data * 0x9E3779B97F4A7C15ull;
    return (size_t)(h >> 20) & (cap - 1);
}
/* the entry of a matrix, made on its first use; NULL when the map cannot grow */
static CudEntry *cud_entry(const void *data, int fmt, int O, int I) {
    for (size_t at = g_cud_cap ? cud_hash(data, g_cud_cap) : 0; g_cud_cap; at = (at + 1) & (g_cud_cap - 1)) {
        CudEntry *e = &g_cud_map[at];
        if (!e->data) break;
        if (e->data == data && e->fmt == fmt && e->O == O && e->I == I) return e;
    }
    if ((g_cud_used + 1) * 2 > g_cud_cap) {
        size_t cap = g_cud_cap ? g_cud_cap * 2 : 256;
        CudEntry *map = (CudEntry *)calloc(cap, sizeof(*map));
        if (!map) return NULL;
        for (size_t i = 0; i < g_cud_cap; i++) {
            if (!g_cud_map[i].data) continue;
            size_t at = cud_hash(g_cud_map[i].data, cap);
            while (map[at].data) at = (at + 1) & (cap - 1);
            map[at] = g_cud_map[i];
        }
        free(g_cud_map);
        g_cud_map = map;
        g_cud_cap = cap;
    }
    size_t at = cud_hash(data, g_cud_cap);
    while (g_cud_map[at].data) at = (at + 1) & (g_cud_cap - 1);
    g_cud_map[at] = (CudEntry){data, fmt, O, I, 0, NULL};
    g_cud_used++;
    return &g_cud_map[at];
}
/* the map's device copies freed and the map with them */
static void cud_map_free(void) {
    for (size_t i = 0; i < g_cud_cap; i++)
        if (g_cud_map[i].h) coli_cuda_tensor_free((ColiCudaTensor *)g_cud_map[i].h);
    free(g_cud_map);
    g_cud_map = NULL;
    g_cud_cap = g_cud_used = 0;
}

static void cud_report(const char *engine) {
    if (g_cud.mats || g_cud.left)
        fprintf(stderr, "[CUDA] %s: %d dense matrices on the device (%.2f GiB), %d left to the CPU; %llu calls, %llu rows\n",
                engine, g_cud.mats, g_cud.used / 1073741824.0, g_cud.left, g_cud.calls, g_cud.rows);
}
#endif /* CU_DENSE_H */
