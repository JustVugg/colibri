/* The CUDA expert tier's decode batch (coli_cuda_expert_group_issue/take) at a big
 * model's decode shape: one row through K experts of E, each layer its own experts (48
 * of them), with the host's round trip; then the chain's decode kernels
 * (coli_cuda_moe_async) on the same experts without it. BENCH_SHAPE "D I K E", BENCH_FMT
 * 2|4, BENCH_GS (fmt 4); COLI_CUDA_GROUP_MOE=0 times the grouped kernels. Build from c/:
 *   nvcc -O2 -std=c++17 -arch=native tests/bench_group_cuda.cu -o tests/bench_group_cuda */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>
#include "../backend_cuda.cu"
#include "../cuda_chain.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(void) {
    int D = 6144, I = 2048, K = 8, E = 64, fmt = 4, gs = 64;
    const char *sh = getenv("BENCH_SHAPE"), *fe = getenv("BENCH_FMT"), *ge = getenv("BENCH_GS");
    if (sh) sscanf(sh, "%d %d %d %d", &D, &I, &K, &E);
    if (fe) fmt = atoi(fe);
    if (ge) gs = atoi(ge);
    if (fmt != 4) gs = 0;
    int devs[1] = {0};
    if (!coli_cuda_init(devs, 1)) { printf("cuda init failed\n"); return 1; }
    int rbD = (D + 1) / 2, rbI = (I + 1) / 2, ngD = gs ? (D + gs - 1) / gs : 1, ngI = gs ? (I + gs - 1) / gs : 1;
    uint8_t *wg = (uint8_t *)malloc((size_t)I * rbD), *wd = (uint8_t *)malloc((size_t)D * rbI);
    float *sg = (float *)malloc((size_t)I * ngD * 4), *sd = (float *)malloc((size_t)D * ngI * 4);
    for (size_t i = 0; i < (size_t)I * rbD; i++) wg[i] = rand() & 255;
    for (size_t i = 0; i < (size_t)D * rbI; i++) wd[i] = rand() & 255;
    for (size_t i = 0; i < (size_t)I * ngD; i++) sg[i] = 0.002f;
    for (size_t i = 0; i < (size_t)D * ngI; i++) sd[i] = 0.002f;
    ColiCudaTensor **tg = (ColiCudaTensor **)calloc(E, sizeof *tg), **tu = (ColiCudaTensor **)calloc(E, sizeof *tu),
                   **td = (ColiCudaTensor **)calloc(E, sizeof *td);
    for (int e = 0; e < E; e++)
        if (!coli_cuda_tensor_upload_g(&tg[e], wg, sg, fmt, D, I, 0, gs) || !coli_cuda_tensor_upload_g(&tu[e], wg, sg, fmt, D, I, 0, gs) ||
            !coli_cuda_tensor_upload_g(&td[e], wd, sd, fmt, I, D, 0, gs)) { printf("upload failed\n"); return 1; }
    /* BENCH_S > 1: a prompt chunk of S rows, each to K distinct experts at random, through the
     * sync group (coli_cuda_expert_group), the rows expert by expert */
    int S = getenv("BENCH_S") ? atoi(getenv("BENCH_S")) : 1;
    if (S > 1) {
        float *xs = (float *)malloc((size_t)S * D * 4), *xg = (float *)malloc((size_t)S * K * D * 4), *yg = (float *)malloc((size_t)S * K * D * 4);
        for (size_t i = 0; i < (size_t)S * D; i++) xs[i] = (rand() / (float)RAND_MAX - .5f) * 2.f;
        int *pick = (int *)malloc((size_t)S * K * sizeof(int)), *cnt = (int *)calloc(E, sizeof(int)), *rws = (int *)malloc(E * sizeof(int));
        ColiCudaTensor **sg2 = (ColiCudaTensor **)malloc(E * sizeof *sg2), **su2 = (ColiCudaTensor **)malloc(E * sizeof *su2), **sd2 = (ColiCudaTensor **)malloc(E * sizeof *sd2);
        double bestp = 1e9; int nlp = 4;
        for (int r = -1; r < 3; r++) {
            double t0 = now();
            for (int l = 0; l < nlp; l++) {
                memset(cnt, 0, E * sizeof(int));
                for (int s2 = 0; s2 < S; s2++) for (int k = 0; k < K; k++) {
                    int e, dup; do { e = rand() % E; dup = 0; for (int j = 0; j < k; j++) dup |= pick[s2 * K + j] == e; } while (dup);
                    pick[s2 * K + k] = e; cnt[e]++;
                }
                int count = 0, at = 0;
                for (int e = 0; e < E; e++) {
                    if (!cnt[e]) continue;
                    sg2[count] = tg[e]; su2[count] = tu[e]; sd2[count] = td[e]; rws[count++] = cnt[e];
                    for (int i = 0; i < S * K; i++) if (pick[i] == e) memcpy(xg + (size_t)(at++) * D, xs + (size_t)(i / K) * D, (size_t)D * 4);
                }
                if (!coli_cuda_expert_group(sg2, su2, sd2, rws, count, yg, xg)) { printf("group failed\n"); return 1; }
            }
            double dt = now() - t0;
            if (r >= 0 && dt < bestp) bestp = dt;
        }
        printf("cuda group prompt fmt %d gs %d S %d K %d E %d [D %d I %d]: %.0f us per layer (with the host's gather)\n", fmt, gs, S, K, E, D, I, bestp * 1e6 / nlp);
        return 0;
    }
    float *x = (float *)malloc((size_t)K * D * 4);
    for (size_t i = 0; i < (size_t)K * D; i++) x[i] = (rand() / (float)RAND_MAX - .5f) * 2.f;
    int *rows = (int *)malloc(K * sizeof(int));
    for (int k = 0; k < K; k++) rows[k] = 1;
    ColiCudaTensor **sgt = (ColiCudaTensor **)malloc(K * sizeof *sgt), **sut = (ColiCudaTensor **)malloc(K * sizeof *sut),
                   **sdt = (ColiCudaTensor **)malloc(K * sizeof *sdt);
    int nl = 48; double best = 1e9;
    for (int r = -1; r < 5; r++) {
        double t0 = now();
        for (int l = 0; l < nl; l++) {
            for (int k = 0; k < K; k++) { int e = (l * K + k) % E; sgt[k] = tg[e]; sut[k] = tu[e]; sdt[k] = td[e]; }
            if (!coli_cuda_expert_group_issue(sgt, sut, sdt, rows, K, x) || !coli_cuda_expert_group_take(0)) { printf("batch failed\n"); return 1; }
        }
        double dt = now() - t0;
        if (r >= 0 && dt < best) best = dt;
    }
    double bytes = (double)K * (2.0 * I * rbD + (double)D * rbI);
    printf("cuda group fmt %d gs %d K %d [D %d I %d]: %.1f us per layer (%.1f GB/s with the round trip)\n",
           fmt, gs, K, D, I, best * 1e6 / nl, bytes * nl / best / 1e9);
    /* the chain's decode kernel (coli_cuda_moe_async) on the same experts: a table of all E,
     * each layer's K ids, S 1, parts 1, timed with events over the 48 layers */
    {
        CcExpert *ht = (CcExpert *)calloc(E, sizeof(CcExpert));
        for (int e = 0; e < E; e++) {
            const void *w; const float *s; int gq;
            coli_cuda_tensor_dev(tg[e], &w, &s, &gq); ht[e].g = w; ht[e].gs = s;
            coli_cuda_tensor_dev(tu[e], &w, &s, &gq); ht[e].u = w; ht[e].us = s;
            coli_cuda_tensor_dev(td[e], &w, &s, &gq); ht[e].d = w; ht[e].ds = s;
        }
        CcExpert *dt; int *dids; float *dx, *dh;
        cudaMalloc(&dt, E * sizeof(CcExpert)); cudaMemcpy(dt, ht, E * sizeof(CcExpert), cudaMemcpyHostToDevice);
        int *hids = (int *)malloc((size_t)nl * K * 4);
        for (int l = 0; l < nl; l++) for (int k = 0; k < K; k++) hids[l * K + k] = (l * K + k) % E;
        cudaMalloc(&dids, (size_t)nl * K * 4); cudaMemcpy(dids, hids, (size_t)nl * K * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&dx, (size_t)D * 4); cudaMemcpy(dx, x, (size_t)D * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&dh, (size_t)K * (I + D) * 4);
        cudaStream_t st; cudaStreamCreate(&st);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        CcMoe mp; memset(&mp, 0, sizeof mp);
        mp.S = 1; mp.K = K; mp.D = D; mp.I = I; mp.fmt = fmt; mp.gs = gs; mp.parts = 1; mp.x_row = D; mp.i_row = K; mp.E = E;
        float bestm = 1e9;
        for (int r = -1; r < 5; r++) {
            cudaEventRecord(e0, st);
            for (int l = 0; l < nl; l++) { mp.i_off = l * K; if (!coli_cuda_moe_async(0, dt, dx, dids, NULL, dh, NULL, NULL, &mp, st)) { printf("moe_async refused\n"); r = 9; break; } }
            cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (r >= 0 && r < 9 && ms < bestm) bestm = ms;
        }
        if (bestm < 1e9) printf("cuda moe_async fmt %d gs %d K %d [D %d I %d]: %.1f us per layer (%.1f GB/s)\n",
                                fmt, gs, K, D, I, bestm * 1e3 / nl, bytes * nl / (bestm * 1e-3) / 1e9);
    }
    return 0;
}
