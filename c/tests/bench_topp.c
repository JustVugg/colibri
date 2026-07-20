/* Microbenchmark: old (full-vocab qsort) vs new (heap partial-select) top-p truncation.
 *
 * This is NOT a unit test -- test_topp.c proves correctness. This measures the headline
 * claim of #335: that replacing the O(V log V) qsort over all 151936 vocab entries with
 * an O(V) heapify + k*log-V pops is materially faster per call, which is the win the issue
 * was opened for.
 *
 * It re-implements the OLD dist_build inline (qsort + scan) on a private buffer so the A/B
 * runs in one process, same inputs, same warm caches -- a controlled comparison. It calls
 * the REAL (new) dist_build via the include-glm.c pattern on the global g_pbuf.
 *
 * Methodology (chosen to be honest, not to flatter the change):
 *   - V = 151936 (the actual GLM-5.2 vocab), g_nuc swept across the values that matter
 *     for serving: 0.5 / 0.9 (serve default) / 0.95 / 0.99.
 *   - Three logit shapes: (a) realistic peaked -- one hot token, long exponential tail,
 *     the shape real language-model logits take; (b) uniform -- worst case for the heap,
 *     maximum pop count; (c) a plateau of ties, to exercise the tie path.
 *   - Each (shape, nuc) is timed over N_REPEAT=2000 iterations, with the RNG/logits frozen
 *     so both algorithms do IDENTICAL work. We report median ns/call and the new/old ratio.
 *   - A warmup pass primes caches before timing.
 *
 * Run:  make tests/bench_topp && ./tests/bench_topp   (not in TEST_BINS -- not a gate)
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <math.h>
#include <stdint.h>

/* ---- the OLD algorithm, verbatim from dev before #354, on a private buffer ---- */
static float *s_pbuf; static int *s_pidx; static double *s_ref;
static int cmp_pdesc_old(const void *a, const void *b){
    double pa = s_ref[*(const int*)a], pb = s_ref[*(const int*)b];
    return pa < pb ? 1 : pa > pb ? -1 : 0; }
static void dist_build_old(const float *lo, int V, double temp, double nuc){
    double mx = lo[0]; for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
    double s = 0, invt = 1.0 / (temp > 1e-4 ? temp : 1e-4);
    for (int i = 0; i < V; i++){ s_ref[i] = exp((lo[i]-mx)*invt); s += s_ref[i]; }
    for (int i = 0; i < V; i++) s_ref[i] /= s;
    if (nuc > 0 && nuc < 1.0){
        for (int i = 0; i < V; i++) s_pidx[i] = i;
        qsort(s_pidx, V, sizeof(int), cmp_pdesc_old);
        double cum = 0; int keep = V;
        for (int i = 0; i < V; i++){ cum += s_ref[s_pidx[i]]; if (cum >= nuc){ keep = i+1; break; } }
        double s2 = 0;
        for (int i = keep; i < V; i++) s_ref[s_pidx[i]] = 0;
        for (int i = 0; i < keep; i++) s2 += s_ref[s_pidx[i]];
        for (int i = 0; i < keep; i++) s_ref[s_pidx[i]] /= s2;
    }
    (void)s_pbuf;
}

/* ---- timing: median of N_REPEAT runs in ns/call, sorted ascending ---- */
#define N_REPEAT 2000
static double bench_ns(void (*fn)(const float*,int,double,double),
                       const float *lo, int V, double temp, double nuc){
    static double ts[N_REPEAT];
    for (int r = 0; r < N_REPEAT; r++){
        double t0 = now_s();
        fn(lo, V, temp, nuc);
        ts[r] = (now_s() - t0) * 1e9;
    }
    /* insertion sort the N_REPEAT samples (small), take median */
    for (int a = 1; a < N_REPEAT; a++){ double k = ts[a]; int b = a-1;
        while (b >= 0 && ts[b] > k){ ts[b+1] = ts[b]; b--; } ts[b+1] = k; }
    return ts[N_REPEAT/2];
}

/* the NEW algorithm is the real dist_build, but it writes g_pbuf (not a private buf).
 * Wrap it so the bench signature matches, and set the globals it reads. */
static void dist_build_new(const float *lo, int V, double temp, double nuc){
    g_temp = (float)temp; g_nuc = (float)nuc;
    dist_build(lo, V);
}

/* deterministic logit fill for three shapes */
static uint32_t brng = 0xA5A5A5A5u;
static double brand(void){ brng ^= brng << 13; brng ^= brng >> 17; brng ^= brng << 5;
    return (double)(brng >> 8) * (1.0 / 16777216.0); }
static void fill_realistic(float *lo, int V){     /* one hot, exponential tail -- like real logits */
    for (int i = 0; i < V; i++) lo[i] = (float)(-4.0 * brand() - (double)i * 0.0001);
    lo[0] = 6.f; lo[V/50] = 4.f; lo[V/200] = 3.f;
}
static void fill_uniform(float *lo, int V){       /* worst case for the heap: max pop count */
    for (int i = 0; i < V; i++) lo[i] = 0.f;
}
static void fill_plateau(float *lo, int V){       /* ties: blocks of equal value */
    for (int i = 0; i < V; i++) lo[i] = (float)(-(double)(i / 50));
}

int main(void){
    int V = 151936;
    float *lo = malloc((size_t)V * sizeof(float));
    s_ref  = malloc((size_t)V * sizeof(double));
    s_pidx = malloc((size_t)V * sizeof(int));
    /* force the new dist_build to allocate g_pbuf/g_pidx at full V once */
    g_temp = 0.7f; g_nuc = 0.9f; dist_build(lo, V);

    double temp = 0.7;
    struct { const char *name; void (*fill)(float*,int); } shapes[] = {
        { "realistic", fill_realistic },
        { "uniform",   fill_uniform   },
        { "plateau",   fill_plateau   },
    };
    double nucs[] = { 0.5, 0.9, 0.95, 0.99 };

    printf("bench_topp: top-p truncation, old (qsort) vs new (heap)  V=%d  temp=%.2f\n", V, temp);
    printf("%-12s %6s %14s %14s %9s %9s\n", "shape", "nuc", "old ns/call", "new ns/call", "speedup", "keep");
    printf("-----------------------------------------------------------------------------\n");

    for (size_t sh = 0; sh < sizeof(shapes)/sizeof(shapes[0]); sh++){
        shapes[sh].fill(lo, V);
        for (size_t ni = 0; ni < sizeof(nucs)/sizeof(nucs[0]); ni++){
            double nuc = nucs[ni];
            /* warmup both paths so caches/branch predictors are primed */
            for (int w = 0; w < 50; w++){ dist_build_old(lo, V, temp, nuc); dist_build_new(lo, V, temp, nuc); }
            double t_old = bench_ns(dist_build_old, lo, V, temp, nuc);
            double t_new = bench_ns(dist_build_new, lo, V, temp, nuc);
            /* keep count = non-zero entries the new path leaves (== old's keep) */
            int keep = 0; for (int i = 0; i < V; i++) if (g_pbuf[i] != 0.f) keep++;
            printf("%-12s %6.2f %14.0f %14.0f %8.2fx %9d\n",
                   shapes[sh].name, nuc, t_old, t_new, t_old / t_new, keep);
        }
        printf("\n");
    }

    printf("bench_topp: done (lower ns is better; speedup = old/new)\n");
    free(lo); free(s_ref); free(s_pidx);
    return 0;
}
