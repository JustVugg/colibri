/* Synthetic peak-RSS proxy for the dense-int8-during-load change (QW/load_tq
 * in qwen36.c). NOT a test gate, NOT an end-to-end model-load benchmark: no
 * full-size Qwen3.6 checkpoint (~70 GB) is available in this environment, so
 * this isolates the actual malloc/free pattern the refactor changes, at
 * matrix sizes representative of Qwen3.6-35B-A3B's dense projections
 * (hidden=4096-ish square matrices), repeated for a layer count comparable to
 * a real model (48 dense matrices -- roughly q/k/v/o/gate/shared x 8 layers).
 *
 * OLD pattern (qdw_register, pre-#1271-adjacent refactor): allocate+read every
 * matrix's f32 copy first, THEN make a second pass quantizing all of them --
 * so all N f32 copies and all N int8 copies are resident at the same time at
 * the crossover point.
 * NEW pattern (load_tq): allocate+read one matrix's f32 copy, quantize it,
 * free the f32 copy, move to the next -- at most one f32 copy is ever
 * resident alongside the accumulating int8 copies.
 *
 * Reports peak RSS (Linux /proc/self/status VmHWM) for both patterns in
 * separate process runs (VmHWM is monotonic non-decreasing for a process's
 * lifetime, so OLD and NEW must run as separate processes, not sequentially
 * in one -- an in-process "old then new" run would report the OLD pattern's
 * peak forever after). Build on demand:
 *
 *   make tests/bench_load_tq_peak_rss ARCH=native
 *   ./tests/bench_load_tq_peak_rss old
 *   ./tests/bench_load_tq_peak_rss new
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

enum { HIDDEN = 4096, OUT = 4096, N_MATRICES = 48 };

static long vmhwm_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "VmHWM:", 6)) { sscanf(line + 6, "%ld", &kb); break; }
    }
    fclose(f);
    return kb;
}

static void fill(float *w, int64_t n, int salt) {
    for (int64_t i = 0; i < n; i++) w[i] = (float)(((i * 2654435761u + salt) % 2003) - 1000) * 0.001f;
}

static void quantize_row_major(const float *w, int I, int O, int8_t *q, float *sc) {
    for (int o = 0; o < O; o++) {
        const float *r = w + (int64_t)o * I; float am = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        float s = am > 1e-12f ? am / 127.f : 1.f; sc[o] = s; float inv = 1.f / s;
        int8_t *d = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(r[i] * inv); if (v > 127) v = 127; if (v < -127) v = -127; d[i] = (int8_t)v; }
    }
}

static void run_old(void) {
    int64_t n = (int64_t)HIDDEN * OUT;
    float **f32s = malloc(N_MATRICES * sizeof(float*));
    /* pass 1: "load" every matrix's f32 copy (as model_init_range used to,
     * for every dense matrix in the whole model, before any quantization) */
    for (int m = 0; m < N_MATRICES; m++) {
        f32s[m] = malloc((size_t)n * sizeof(float));
        fill(f32s[m], n, m);
    }
    printf("after loading all %d f32 matrices: VmHWM=%ld MiB\n", N_MATRICES, vmhwm_kb() / 1024);
    /* pass 2: quantize every matrix (as the old post-hoc qdw_register loop
     * did), freeing each f32 copy only after ALL quantization is done */
    int8_t **qs = malloc(N_MATRICES * sizeof(int8_t*));
    float **scs = malloc(N_MATRICES * sizeof(float*));
    for (int m = 0; m < N_MATRICES; m++) {
        qs[m] = malloc((size_t)n);
        scs[m] = malloc((size_t)OUT * sizeof(float));
        quantize_row_major(f32s[m], HIDDEN, OUT, qs[m], scs[m]);
    }
    printf("after quantizing all (pre-free): VmHWM=%ld MiB\n", vmhwm_kb() / 1024);
    for (int m = 0; m < N_MATRICES; m++) free(f32s[m]);
    printf("[old pattern] peak VmHWM=%ld MiB\n", vmhwm_kb() / 1024);
}

static void run_new(void) {
    int64_t n = (int64_t)HIDDEN * OUT;
    int8_t **qs = malloc(N_MATRICES * sizeof(int8_t*));
    float **scs = malloc(N_MATRICES * sizeof(float*));
    /* load_tq: read one matrix's f32 copy, quantize it, free it, next matrix */
    for (int m = 0; m < N_MATRICES; m++) {
        float *w = malloc((size_t)n * sizeof(float));
        fill(w, n, m);
        qs[m] = malloc((size_t)n);
        scs[m] = malloc((size_t)OUT * sizeof(float));
        quantize_row_major(w, HIDDEN, OUT, qs[m], scs[m]);
        free(w);
    }
    printf("[new pattern] peak VmHWM=%ld MiB\n", vmhwm_kb() / 1024);
}

int main(int argc, char **argv) {
    if (argc != 2 || (strcmp(argv[1], "old") && strcmp(argv[1], "new"))) {
        fprintf(stderr, "usage: %s old|new\n", argv[0]); return 2;
    }
    printf("N_MATRICES=%d each %dx%d f32 (%.1f MiB) -- %s pattern\n",
           N_MATRICES, HIDDEN, OUT, (double)HIDDEN * OUT * 4 / 1048576.0, argv[1]);
    if (!strcmp(argv[1], "old")) run_old(); else run_new();
    return 0;
}
