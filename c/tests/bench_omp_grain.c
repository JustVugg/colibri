/* bench_omp_grain — what a parallel region costs before it does any work.
 *
 * The engine wraps work in `#pragma omp parallel for` without asking how much
 * work is in there. That is right for an expert matmul and wrong for a loop
 * with one iteration: OpenMP still forks and joins the team, and at decode
 * (S=1) several of the engine's regions have exactly that shape.
 *
 * This measures the floor: a region whose body does nothing measurable, so the
 * number IS the fork/join cost. Multiply by regions-per-token times layers to
 * see what it means for a real model.
 *
 *   make -C c bench-omp-grain
 *
 * Reports nanoseconds per region for the team as configured (OMP_NUM_THREADS),
 * against the same loop run inline via `if(0)`. Runs long enough that the
 * difference is not timer noise -- the engine-level tiny-model replay is 15 ms
 * end to end and cannot resolve this at all.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Sink the loop body writes to, so neither version can be optimised away. */
static volatile long g_sink;

/* One iteration per region: the decode shape (S == 1). */
static double bench_region(long reps, int iters, int parallel) {
    long acc = 0;
    double t0 = now_s();
    for (long r = 0; r < reps; r++) {
#pragma omp parallel for reduction(+ : acc) if (parallel)
        for (int i = 0; i < iters; i++) acc += i + r;
    }
    double dt = now_s() - t0;
    g_sink = acc;
    return dt;
}

/* Median of n runs: the first is always slower (team not yet warm) and a
 * scheduler hiccup should not become the headline number. */
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static double median_ns(long reps, int iters, int parallel, int runs) {
    double *v = malloc((size_t)runs * sizeof(double));
    if (!v) return -1;
    for (int k = 0; k < runs; k++) v[k] = bench_region(reps, iters, parallel) / (double)reps * 1e9;
    qsort(v, (size_t)runs, sizeof(double), cmp_d);
    double m = v[runs / 2];
    free(v);
    return m;
}

int main(int argc, char **argv) {
    long reps = argc > 1 ? atol(argv[1]) : 200000;
    int runs = argc > 2 ? atoi(argv[2]) : 7;
    if (reps < 1) reps = 1;
    if (runs < 1) runs = 1;

    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp master
        nthreads = omp_get_num_threads();
    }
#else
    printf("built without OpenMP -- nothing to measure\n");
    return 0;
#endif

    printf("bench_omp_grain: %ld regions per sample, %d samples, team of %d\n", reps, runs, nthreads);
    printf("  %-12s %14s %14s %12s\n", "iterations", "parallel (ns)", "inline (ns)", "overhead");
    for (int iters = 1; iters <= 64; iters *= 8) {
        double par = median_ns(reps, iters, 1, runs);
        double seq = median_ns(reps, iters, 0, runs);
        printf("  %-12d %14.1f %14.1f %11.1fx\n", iters, par, seq, seq > 0 ? par / seq : 0.0);
    }

    /* The number the engine cares about: one iteration, which is what S=1
     * decode gives every per-position region. */
    double one_par = median_ns(reps, 1, 1, runs);
    double one_seq = median_ns(reps, 1, 0, runs);
    double cost = one_par - one_seq;
    printf("\n  fork/join floor: %.0f ns per region (team of %d)\n", cost, nthreads);
    printf("  a 78-layer model paying this once per layer per token: %.2f ms/token\n", cost * 78.0 / 1e6);
    return 0;
}
