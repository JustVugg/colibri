/* V4_SPARSE_FAST: the decode step (one token) of dsv4_cuda_sparse_attn_batch_cached and
 * of its indexed variant, written to a file so two processes can be compared bit for
 * bit (the switch is read once per process):
 *   V4_SPARSE_FAST=0 ./dsv4_sparse_fast_cuda_test old.bin
 *   V4_SPARSE_FAST=1 ./dsv4_sparse_fast_cuda_test new.bin      (then cmp old.bin new.bin)
 * DeepSeek V4 decode shape: 64 heads of 512, a 128-row window ring that wraps, the
 * step's own row in the chunk, up to 2000 compressed rows (more than one 1024-item tile
 * of the fast output kernel) and an indexed selection of 512. Random rows, sinks and
 * queries. It also prints the time of one realistic call (window 128 + 512 selected),
 * as information. Exit 77: no CUDA device. */
#include "../backend_cuda_dsv4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

enum { HEADS = 64, DIM = 512, WINDOW = 128, COMP = 2000, SEL = 512, LAYER = 3 };

static uint32_t rng = 0x9e3779b9u;
/* Full 24-bit mantissas: with short ones every product would be exact and a fused
 * multiply-add could not be told apart from a separate multiply and add. */
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float)(rng >> 8) / 8388608.0f - 1.0f;
}
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

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s out.bin\n", argv[0]); return 2; }
    int dev = 0;
    if (!dsv4_cuda_init(&dev, 1)) { puts("dsv4 sparse fast: no CUDA device, skipped"); return 77; }
    FILE *out = fopen(argv[1], "wb");
    if (!out) return 2;
    float *row = malloc((size_t)DIM * sizeof(float));
    float *comp = malloc((size_t)COMP * DIM * sizeof(float));
    float *q = malloc((size_t)HEADS * DIM * sizeof(float));
    float *got = malloc((size_t)HEADS * DIM * sizeof(float));
    float sinks[HEADS];
    int *sel = malloc(SEL * sizeof(int)), *pool = malloc(COMP * sizeof(int));
    if (!row || !comp || !q || !got || !sel || !pool) return 2;
    for (int h = 0; h < HEADS; h++) sinks[h] = frand();
    for (size_t i = 0; i < (size_t)COMP * DIM; i++) comp[i] = frand();
    if (!dsv4_cuda_kv_comp_append(dev, LAYER, comp, 0, COMP, DIM)) { fprintf(stderr, "comp append refused\n"); return 3; }
    const float scale = 1.0f / sqrtf((float)DIM);

    /* (position, compressed rows): the ring fills, wraps, and the item count crosses
     * the 1024-item tile. Positions only grow, so the ring holds the right rows. */
    static const int pos[] = {0, 1, 50, 127, 128, 300, 700, 1500, 5000};
    static const int cns[] = {0, 3, 17, 64, 200, 512, 1100, 1999, 2000};
    /* The ring exists once a row went in: position 0's history row (at p = 0 the step's
     * own row comes from the chunk instead, as in the engine). */
    for (int d = 0; d < DIM; d++) row[d] = frand();
    if (!dsv4_cuda_kv_ring_append(dev, LAYER, row, 0, 1, WINDOW, DIM)) { fprintf(stderr, "ring append refused\n"); return 3; }
    int appended = 1, calls = 0;
    for (size_t s = 0; s < sizeof pos / sizeof pos[0]; s++) {
        int p = pos[s], cn = cns[s];
        for (; appended < p; appended++) {        /* history rows of positions < p */
            for (int d = 0; d < DIM; d++) row[d] = frand();
            if (!dsv4_cuda_kv_ring_append(dev, LAYER, row, appended, 1, WINDOW, DIM)) return 3;
        }
        for (int d = 0; d < DIM; d++) row[d] = frand();   /* this step's own row: the chunk */
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++) q[i] = frand();
        int wn = p + 1 < WINDOW ? p + 1 : WINDOW;
        int meta[3] = {0, wn, cn};
        int abs_base = p - wn + 1;
        if (!dsv4_cuda_sparse_attn_batch_cached(dev, LAYER, q, row, p, sinks, meta, abs_base, COMP,
                                                HEADS, DIM, 1, scale, got)) {
            fprintf(stderr, "cached refused at p=%d cn=%d\n", p, cn);
            return 4;
        }
        fwrite(got, sizeof(float), (size_t)HEADS * DIM, out); calls++;
        int sn = cn < SEL ? cn : SEL;                    /* the indexed variant: a shuffled subset */
        for (int i = 0; i < COMP; i++) pool[i] = i;
        for (int i = 0; i < sn; i++) {
            int j = i + (int)(rng % (uint32_t)(COMP - i)); rng = rng * 1664525u + 1013904223u;
            int t = pool[i]; pool[i] = pool[j]; pool[j] = t; sel[i] = pool[i];
        }
        int meta_idx[3] = {0, wn, sn};
        if (!dsv4_cuda_sparse_attn_batch_cached_idx(dev, LAYER, q, row, p, sinks, meta_idx, sel, SEL,
                                                    abs_base, COMP, HEADS, DIM, 1, scale, got)) {
            fprintf(stderr, "cached idx refused at p=%d sn=%d\n", p, sn);
            return 5;
        }
        fwrite(got, sizeof(float), (size_t)HEADS * DIM, out); calls++;
    }
    /* Information only: one realistic decode call, window 128 + 512 selected rows. */
    int meta_idx[3] = {0, WINDOW, SEL};
    double t0 = now_ms();
    for (int it = 0; it < 200; it++)
        if (!dsv4_cuda_sparse_attn_batch_cached_idx(dev, LAYER, q, row, 5000, sinks, meta_idx, sel, SEL,
                                                    5000 - WINDOW + 1, COMP, HEADS, DIM, 1, scale, got)) return 6;
    double dt = (now_ms() - t0) / 200.0;
    fclose(out);
    const char *flag = getenv("V4_SPARSE_FAST");
    printf("dsv4 sparse fast: %d calls written (V4_SPARSE_FAST=%s); window 128 + 512 selected: %.1f us per call\n",
           calls, flag && *flag ? flag : "unset", dt * 1000.0);
    dsv4_cuda_shutdown();
    free(pool); free(sel); free(got); free(q); free(comp); free(row);
    return 0;
}
