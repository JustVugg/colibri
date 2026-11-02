/* Microbenchmark: one decode token's attention core in qwen36, olmoe and inkling,
 * with the q.k score as the engines run it today and through matmul_f32.h's lanes.
 *   scalar   `acc += qv[d]*kv[d]`, one chain per score (qwen36.c, olmoe.c, inkling.c)
 *   lanes    dot_f32_lanes(qv, kv, hd)
 * Everything else is the engines' loop as it is: per query head, the scores
 * against every cached key, softmax_row, then the value mix. One call is one
 * layer at S=1 with the context already in the cache; a token costs that times
 * the number of attention layers (Qwen3.6-35B-A3B: 10, OLMoE: 16).
 *
 * NOT a unit test: tests/test_matmul_f32 gates the lane kernel's exactness. Each
 * arm runs SAMPLES timed samples (each the mean of enough calls to last ~50 ms)
 * after one warm-up sample, arms interleaved per sample; the median is reported,
 * then every raw sample.
 *
 * Run:  make tests/bench_attn_f32 && ./tests/bench_attn_f32
 *       OMP_NUM_THREADS=1 ./tests/bench_attn_f32   (single thread) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../matmul_f32.h"

#define SAMPLES 5
#define MAX_THREADS 256

typedef struct { int H, KV, hd, T; const float *q, *K, *V; float *ctx, *sc; } Attn;

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* qwen36.c's decode loop, the score computed by `dot` */
#define ATTN_CORE(name, DOT)                                                          \
static void name(const Attn *a) {                                                     \
    int q_per_kv = a->H / a->KV, hd = a->hd, qpos = a->T - 1;                         \
    float scale = 1.f / sqrtf((float)hd);                                             \
    _Pragma("omp parallel for schedule(static)")                                      \
    for (int hh = 0; hh < a->H; hh++) {                                               \
        int kvh = hh / q_per_kv, tid = 0;                                             \
        IF_OMP(tid = omp_get_thread_num();)                                           \
        const float *qv = a->q + (int64_t)hh * hd;                                    \
        float *sc = a->sc + (int64_t)tid * a->T;                                      \
        for (int t = 0; t <= qpos; t++) {                                             \
            const float *kv = a->K + ((int64_t)kvh * a->T + t) * hd;                  \
            float acc; DOT;                                                           \
            sc[t] = acc * scale;                                                      \
        }                                                                             \
        softmax_row(sc, qpos + 1);                                                    \
        float *cx = a->ctx + (int64_t)hh * hd;                                        \
        for (int dd = 0; dd < hd; dd++) cx[dd] = 0;                                   \
        for (int t = 0; t <= qpos; t++) {                                             \
            const float *vrow = a->V + ((int64_t)kvh * a->T + t) * hd;                \
            float w = sc[t]; for (int dd = 0; dd < hd; dd++) cx[dd] += w * vrow[dd];  \
        }                                                                             \
    }                                                                                 \
}
#ifdef _OPENMP
#define IF_OMP(x) x
#else
#define IF_OMP(x)
#endif
ATTN_CORE(attn_scalar, acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd] * kv[dd])
ATTN_CORE(attn_lanes, acc = dot_f32_lanes(qv, kv, hd))

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmp_d(const void *a, const void *b){ double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }

typedef void (*attn_fn)(const Attn *);
static double sample_us(attn_fn f, const Attn *a, int reps){
    attn_fn volatile call = f;   /* opaque call: without OpenMP the unused result would be optimized away */
    double t = now_s(); for (int r = 0; r < reps; r++) call(a); return (now_s() - t) / reps * 1e6;
}

static void fill(float *x, size_t n, float amp, uint64_t *rng){
    for (size_t i = 0; i < n; i++){ *rng^=*rng<<13; *rng^=*rng>>7; *rng^=*rng<<17;
        x[i] = ((float)(*rng>>40)/(float)(1<<24)*2.f-1.f)*amp; }
}

int main(void){
    static const struct { const char *name; int H, KV, hd, T; } shapes[] = {
        {"Qwen3.6-35B-A3B", 16, 2, 256,  1024}, {"Qwen3.6-35B-A3B", 16, 2, 256,  8192},
        {"Qwen3.6-35B-A3B", 16, 2, 256, 32768}, {"OLMoE-1B-7B",    16, 16, 128, 1024},
        {"OLMoE-1B-7B",     16, 16, 128, 4096} };
    static const attn_fn arm[] = { attn_scalar, attn_lanes };
    static const char *arm_name[] = { "scalar", "lanes" };
    enum { ARMS = 2 };
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    printf("threads: %d\n", threads);
    printf("%-16s %-22s %10s %10s  median us/layer; scalar/lanes\n", "model", "shape", "scalar", "lanes");
    double raw[sizeof shapes / sizeof shapes[0]][ARMS][SAMPLES];
    for (size_t k = 0; k < sizeof shapes / sizeof shapes[0]; k++){
        Attn a = { shapes[k].H, shapes[k].KV, shapes[k].hd, shapes[k].T, 0, 0, 0, 0, 0 };
        size_t cache = (size_t)a.KV * a.T * a.hd;
        float *q = malloc(sizeof(float) * (size_t)a.H * a.hd), *K = malloc(sizeof(float) * cache),
              *V = malloc(sizeof(float) * cache);
        a.ctx = malloc(sizeof(float) * (size_t)a.H * a.hd);
        a.sc = malloc(sizeof(float) * (size_t)threads * a.T);
        fill(q, (size_t)a.H * a.hd, 1.f, &rng); fill(K, cache, 1.f, &rng); fill(V, cache, 1.f, &rng);
        a.q = q; a.K = K; a.V = V;
        int reps = 1; while (sample_us(attn_scalar, &a, reps) * reps < 50e3 && reps < 1<<20) reps *= 2;
        for (int r = 0; r < ARMS; r++) sample_us(arm[r], &a, reps);   /* warm-up */
        for (int s = 0; s < SAMPLES; s++) for (int r = 0; r < ARMS; r++) raw[k][r][s] = sample_us(arm[r], &a, reps);
        double t[ARMS][SAMPLES];
        memcpy(t, raw[k], sizeof t);
        for (int r = 0; r < ARMS; r++) qsort(t[r], SAMPLES, sizeof(double), cmp_d);
        printf("%-16s H=%-2d KV=%-2d hd=%-3d T=%-5d %10.1f %10.1f  %.2fx\n", shapes[k].name,
               a.H, a.KV, a.hd, a.T, t[0][SAMPLES/2], t[1][SAMPLES/2], t[0][SAMPLES/2] / t[1][SAMPLES/2]);
        free(q); free(K); free(V); free(a.ctx); free(a.sc);
    }
    printf("raw samples, us/layer, in run order:\n");
    for (size_t k = 0; k < sizeof shapes / sizeof shapes[0]; k++)
        for (int r = 0; r < ARMS; r++){
            printf("%s T=%d %s:", shapes[k].name, shapes[k].T, arm_name[r]);
            for (int s = 0; s < SAMPLES; s++) printf(" %.1f", raw[k][r][s]);
            printf("\n");
        }
    return 0;
}
