/* V4_ATTN_BATCH_PROJ: dsv4_cuda_proj_batch (n fp8 decode matvecs over one input in
 * one device round trip) against dsv4_cuda_matvec called once per tensor, bit for
 * bit, at the DeepSeek V4 decode shapes: wq_a 1024x4096, wkv 512x4096 (the pair the
 * attention uses), and a three-tensor batch with wq_b-sized 8192x1024. Random e4m3
 * weights with random e8m0 block scales; NaN codes excluded. Also checks that a
 * bf16 tensor is refused (the caller then takes the per-call path). Exit 77 = the
 * backend did not start (no usable CUDA device; on Windows also no coli_cuda_dsv4.dll
 * beside the binary). A DLL without the export fails the batch check.
 *
 * Run with `make -C c dsv4-proj-batch-test` (Windows: a coli_cuda_dsv4.dll built from
 * this tree next to the test binary, the CUDA bin directory on PATH; elsewhere the
 * backend objects are linked). It also prints the time of
 * both paths, as information only. */
#include "../backend_cuda_dsv4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
#endif
}

static uint64_t rng = 0x2545f4914f6cdd1dull;
static uint32_t next(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

/* Random fp8 e4m3 weights (no NaN code) and e8m0 block scales near 1. */
static Dsv4CudaTensor *random_fp8(int O, int I, int device) {
    size_t wn = (size_t)O * I, sn = (size_t)((O + 127) / 128) * ((I + 127) / 128);
    uint8_t *w = malloc(wn), *s = malloc(sn);
    Dsv4CudaTensor *t = NULL;
    if (!w || !s) { free(w); free(s); return NULL; }
    for (size_t i = 0; i < wn; i++) {
        uint8_t b = (uint8_t)next();
        if ((b & 0x7fu) == 0x7fu) b ^= 0x01u;     /* keep NaN codes out */
        w[i] = b;
    }
    for (size_t i = 0; i < sn; i++) s[i] = (uint8_t)(120 + next() % 12);
    int ok = dsv4_cuda_upload_fp8(&t, w, s, O, I, device);
    free(s); free(w);
    return ok ? t : NULL;
}

static int check_batch(const char *name, int n, Dsv4CudaTensor **t, const int *rows,
                       int I, const float *x, int device) {
    int total = 0;
    for (int i = 0; i < n; i++) total += rows[i];
    float *want = malloc((size_t)total * sizeof(float));
    float *got = malloc((size_t)total * sizeof(float));
    if (!want || !got) return 2;
    int off = 0;
    for (int i = 0; i < n; i++) {
        if (!dsv4_cuda_matvec(t[i], want + off, x)) { fprintf(stderr, "%s: per-call matvec failed\n", name); return 3; }
        off += rows[i];
    }
    /* Twice: the second call reuses the output buffer and the stream state. */
    for (int rep = 0; rep < 2; rep++) {
        memset(got, 0, (size_t)total * sizeof(float));
        if (!dsv4_cuda_proj_batch(n, t, rows, x, got)) {
            fprintf(stderr, "%s: dsv4_cuda_proj_batch refused a valid batch\n", name);
            return 4;
        }
        if (memcmp(want, got, (size_t)total * sizeof(float)) != 0) {
            for (int k = 0; k < total; k++)
                if (memcmp(&want[k], &got[k], sizeof(float)) != 0) {
                    fprintf(stderr, "%s: mismatch rep=%d at %d: per-call=%g batch=%g\n",
                            name, rep, k, want[k], got[k]);
                    break;
                }
            return 5;
        }
    }
    printf("%s: batch == per-call matvec, %d outputs, %d floats: OK\n", name, n, total);
    /* Information only: the per-call sequence and the batch, 200 calls each. */
    double t0 = now_ms();
    for (int it = 0; it < 200; it++)
        for (int i = 0, o = 0; i < n; o += rows[i], i++)
            if (!dsv4_cuda_matvec(t[i], want + o, x)) return 3;
    double t1 = now_ms();
    for (int it = 0; it < 200; it++)
        if (!dsv4_cuda_proj_batch(n, t, rows, x, got)) return 4;
    double t2 = now_ms();
    printf("%s: per-call %.1f us, batch %.1f us\n", name, (t1 - t0) * 5.0, (t2 - t1) * 5.0);
    (void)device;
    free(got); free(want);
    return 0;
}

int main(void) {
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) return 77;
    const int H = 4096;
    float *x = malloc((size_t)H * sizeof(float));
    if (!x) return 2;
    for (int round = 0; round < 3; round++) {
        for (int i = 0; i < H; i++) x[i] = ((float)(next() % 2001) - 1000.0f) / 500.0f;
        /* wq_a + wkv: the attention's pair. */
        Dsv4CudaTensor *pair[2] = {random_fp8(1024, H, dev), random_fp8(512, H, dev)};
        int prow[2] = {1024, 512};
        if (!pair[0] || !pair[1]) return 3;
        int rc = check_batch("wq_a+wkv", 2, pair, prow, H, x, dev);
        if (rc) return rc;
        /* three tensors of mixed shapes, one of them the wq_b-sized 8192x1024. */
        Dsv4CudaTensor *tri[3] = {random_fp8(8192, 1024, dev), random_fp8(256, 1024, dev), random_fp8(64, 1024, dev)};
        int trow[3] = {8192, 256, 64};
        if (!tri[0] || !tri[1] || !tri[2]) return 3;
        float *x1 = malloc(1024 * sizeof(float));
        if (!x1) return 2;
        for (int i = 0; i < 1024; i++) x1[i] = x[i];
        rc = check_batch("wq_b-shaped x3", 3, tri, trow, 1024, x1, dev);
        free(x1);
        if (rc) return rc;
        for (int i = 0; i < 2; i++) dsv4_cuda_tensor_free(pair[i]);
        for (int i = 0; i < 3; i++) dsv4_cuda_tensor_free(tri[i]);
    }
    /* A bf16 tensor is refused: the caller takes the per-call path. */
    {
        uint16_t *bw = calloc((size_t)512 * H, sizeof(uint16_t));
        Dsv4CudaTensor *bt = NULL;
        if (!bw || !dsv4_cuda_upload_bf16(&bt, bw, 512, H, dev)) return 3;
        float *out = malloc(512 * sizeof(float));
        int refused = !dsv4_cuda_proj_batch(1, &bt, (const int[]){512}, x, out);
        if (!refused) { fprintf(stderr, "bf16 tensor accepted by proj batch\n"); return 6; }
        printf("bf16 tensor refused by proj batch: OK\n");
        free(out); dsv4_cuda_tensor_free(bt); free(bw);
    }
    dsv4_cuda_proj_batch_release();
    dsv4_cuda_shutdown();
    free(x);
    puts("dsv4 proj batch: OK");
    return 0;
}
