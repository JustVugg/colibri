/* Microbenchmark: old (full-qsort) vs new (quickselect partial-select) DSA top-keep.
 *
 * This is NOT a unit test -- test_dsa_select.c proves correctness. This measures the
 * headline claim of #356: that replacing the O(nk log nk) qsort over all nk context
 * scores with an O(nk) partial_select_desc is materially faster per call, which is
 * the win the issue was opened for -- and that the win GROWS with context length
 * (because quickselect is linear average, qsort is n-log-n).
 *
 * It re-implements the OLD top-keep inline (qsort + threshold + scans) on a private
 * buffer so the A/B runs in one process, same inputs, same warm caches -- a controlled
 * comparison. It calls the REAL (new) partial_select_desc via the include-glm.c
 * pattern, replicating the production threshold derivation + scans.
 *
 * Methodology (chosen to be honest, not to flatter the change):
 *   - keep = 2048 (the real GLM-5.2 index_topk), nk swept across context lengths from
 *     the 2049 activation boundary up to 65536 (a long conversation).
 *   - Three score shapes: (a) realistic peaked -- a few hot keys, long tail, the shape
 *     real DSA attention scores take; (b) uniform random -- no structure; (c) a plateau
 *     of ties, to exercise the boundary-membership path.
 *   - Each (shape, nk) is timed over N_REPEAT=2000 iterations, with the scores frozen
 *     so both algorithms do IDENTICAL work. We report median ns/call and the new/old
 *     ratio. A warmup pass primes caches before timing.
 *
 * Run:  make tests/bench_dsa_select && ./tests/bench_dsa_select  (not in TEST_BINS)
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <math.h>
#include <stdint.h>

/* ---- the OLD algorithm, verbatim from dev before #356, on a private buffer ---- */
static int cmp_pdesc_old(const void *a, const void *b){
    float x=*(const float*)a, y=*(const float*)b; return x<y?1:x>y?-1:0; }
static void keep_old(const float *isc, int nk, int keep, int *dst, int *nd_out){
    float *tmp=malloc((size_t)nk*sizeof(float));
    memcpy(tmp,isc,(size_t)nk*sizeof(float));
    qsort(tmp,(size_t)nk,sizeof(float),cmp_pdesc_old);
    float thr=tmp[keep-1]; int nd=0;
    for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
    for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
    free(tmp); *nd_out=nd;
}

/* ---- timing: median of N_REPEAT runs in ns/call, sorted ascending ---- */
#define N_REPEAT 2000
static double bench_ns(void (*fn)(const float*,int,int,int*,int*),
                       const float *isc, int nk, int keep){
    static double ts[N_REPEAT]; int *dst=malloc((size_t)nk*sizeof(int)); int nd;
    for(int r=0;r<N_REPEAT;r++){
        double t0=now_s();
        fn(isc,nk,keep,dst,&nd);
        ts[r]=(now_s()-t0)*1e9;
    }
    for(int a=1;a<N_REPEAT;a++){ double k=ts[a]; int b=a-1;
        while(b>=0 && ts[b]>k){ ts[b+1]=ts[b]; b--; } ts[b+1]=k; }
    free(dst);
    return ts[N_REPEAT/2];
}

/* Sort an array of doubles ascending (median-of-medians aggregation below). */
static void dsort(double *a, int n){
    for(int s=1;s<n;s++){ double k=a[s]; int b=s-1;
        while(b>=0 && a[b]>k){ a[b+1]=a[b]; b--; } a[b+1]=k; }
}

/* the NEW algorithm calls the real partial_select_desc + replicates the production
 * threshold derivation and position scans. */
static void keep_new(const float *isc, int nk, int keep, int *dst, int *nd_out){
    float *tmp=malloc((size_t)nk*sizeof(float));
    memcpy(tmp,isc,(size_t)nk*sizeof(float));
    partial_select_desc(tmp,nk,keep);
    float thr=tmp[0]; for(int t=1;t<keep;t++) if(tmp[t]<thr) thr=tmp[t];
    int nd=0;
    for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
    for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
    free(tmp); *nd_out=nd;
}

/* deterministic score fill for three shapes */
static uint32_t brng = 0xA5A5A5A5u;
static void brng_seed(uint32_t s){ brng = s; }
static double brand(void){ brng ^= brng << 13; brng ^= brng >> 17; brng ^= brng << 5;
    return (double)(brng >> 8) * (1.0 / 16777216.0); }
static void fill_realistic(float *isc, int nk){   /* few hot, long distinct tail */
    for(int i=0;i<nk;i++) isc[i]=(float)(-1.0 - brand()*4.0);
    isc[0]=3.f; isc[nk/50<nk?nk/50:nk-1]=1.f; isc[nk/200<nk?nk/200:nk-1]=0.5f;
}
static void fill_uniform(float *isc, int nk){     /* no structure */
    for(int i=0;i<nk;i++) isc[i]=(float)(brand()*1000.0);
}
static void fill_plateau(float *isc, int nk){     /* tie blocks -> boundary path */
    for(int i=0;i<nk;i++) isc[i]=(float)(-(double)(i/7));
}

/* Multi-seed aggregation. Per @KingIcyCreamProjects (#357 thread): with a single frozen
 * input per cell, quickselect's deterministic median-of-three pivot means one lucky input
 * can spike a single nk row (an observed ~75x at nk=8192 on a 9950X3D that was really a
 * ~13-40x algorithm). Two bugs compounded it: (1) the old bench drew ONE input per cell;
 * (2) brng was never reset, so each cell's input depended on every prior cell's draws --
 * reordering nks[] silently shifted all later inputs. Both fixed here: brng is reseeded
 * per (shape, nk, seed), and we take the MEDIAN of N_SEEDS independent inputs, each itself
 * a median over N_REPEAT timing reps. A lucky pivot now moves one of the N_SEEDS samples,
 * not the reported number. */

int main(void){
    int keep = 2048;   /* GLM-5.2 index_topk */
    int nks[] = {2049, 4096, 8192, 16384, 32768, 65536};
    const int N_SEEDS = 11;   /* odd so the median is a real sample, not interpolated */
    float *isc = malloc((size_t)65536*sizeof(float));
    int *dst   = malloc((size_t)65536*sizeof(int)); int nd;
    double *old_seeds = malloc((size_t)N_SEEDS*sizeof(double));
    double *new_seeds = malloc((size_t)N_SEEDS*sizeof(double));

    struct { const char *name; void (*fill)(float*,int); } shapes[] = {
        { "realistic", fill_realistic },
        { "uniform",   fill_uniform   },
        { "plateau",   fill_plateau   },
    };

    printf("bench_dsa_select: DSA top-keep, old (qsort) vs new (partial-select)  keep=%d  (median of %d seeds x %d reps)\n",
           keep, N_SEEDS, N_REPEAT);
    printf("%-12s %7s %14s %14s %9s\n", "shape", "nk", "old ns/call", "new ns/call", "speedup");
    printf("------------------------------------------------------------------------\n");

    for(size_t sh=0; sh<sizeof(shapes)/sizeof(shapes[0]); sh++){
        for(size_t ni=0; ni<sizeof(nks)/sizeof(nks[0]); ni++){
            int nk=nks[ni];
            int bad = 0;
            for(int sd=0; sd<N_SEEDS; sd++){
                /* reseed per (shape,nk,seed) so each cell's input is reproducible and
                 * independent of cell ordering, and so lucky pivots are sampled, not fixed. */
                brng_seed(0xA5A5A5A5u + (uint32_t)(sd*0x9E3779B9u));
                shapes[sh].fill(isc,nk);
                /* warmup both paths so caches/branch predictors are primed */
                for(int w=0; w<50; w++){ keep_old(isc,nk,keep,dst,&nd); keep_new(isc,nk,keep,dst,&nd); }
                /* sanity: both must keep exactly `keep` (correctness is test_dsa_select's
                 * job, but a count divergence here would make the timing meaningless) */
                keep_old(isc,nk,keep,dst,&nd); int na=nd;
                keep_new(isc,nk,keep,dst,&nd); int nb=nd;
                if(na!=keep || nb!=keep){ bad++; continue; }
                old_seeds[sd] = bench_ns(keep_old,isc,nk,keep);
                new_seeds[sd] = bench_ns(keep_new,isc,nk,keep);
            }
            if(bad == N_SEEDS){
                printf("%-12s %7d   (BAD COUNTS on all %d seeds, skipped)\n",
                       shapes[sh].name, nk, bad);
                continue;
            }
            /* report median-of-seed-medians: robust to a single lucky/unlucky pivot */
            dsort(old_seeds, N_SEEDS);
            dsort(new_seeds, N_SEEDS);
            double t_old = old_seeds[N_SEEDS/2];
            double t_new = new_seeds[N_SEEDS/2];
            printf("%-12s %7d %14.0f %14.0f %8.2fx\n",
                   shapes[sh].name, nk, t_old, t_new, t_old/t_new);
        }
        printf("\n");
    }

    printf("bench_dsa_select: done (lower ns is better; speedup = old/new)\n");
    free(isc); free(dst); free(old_seeds); free(new_seeds);
    return 0;
}
